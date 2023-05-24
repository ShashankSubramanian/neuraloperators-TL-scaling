import os, sys, time
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_utils import get_data_loader
from utils.optimizer_utils import set_scheduler, set_optimizer
from utils.loss_utils import LossMSE
from utils.misc_utils import compute_grad_norm, vis_fields, l2_err
from utils.domains import DomainXY
from utils.sweeps import sweep_name_suffix
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict

# models
import models.ffn
import models.fno

def print_mem():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

def set_seed(params, world_size):
    seed = params.seed
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

class Trainer():
    """ trainer class """
    def __init__(self, params, args):
        self.sweep_id = args.sweep_id
        self.root_dir = args.root_dir
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True
        
        self.log_to_screen = params.log_to_screen and self.world_rank==0
        self.log_to_wandb = params.log_to_wandb and self.world_rank==0
        params['name'] = args.config + '_' + args.run_num
        params['group'] = 'op_' + args.config
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        self.params = params
        self.params.device = self.device

    def init_exp_dir(self, exp_dir):
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                os.makedirs(os.path.join(exp_dir, 'wandb/'))
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False

    def launch(self):

        if self.sweep_id:
            if self.world_rank==0:
                with wandb.init() as run:
                    hpo_config = wandb.config
                    self.params.update_params(hpo_config)
                    self.modify_bs_for_subsampling()
                    logging.info(self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id))
                    run.name = self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id)
                    self.name = run.name
                    self.params.name = self.name
                    exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.name])
                    self.init_exp_dir(exp_dir)
                    logging.info('HPO sweep %s, trial cfg %s'%(self.sweep_id, self.name))
                    self.build_and_run()
            else:
                self.build_and_run()

        else:
            self.modify_bs_for_subsampling()
            exp_dir = os.path.join(*[self.root_dir, 'expts', self.config, self.run_num])
            self.init_exp_dir(exp_dir)
            if self.log_to_wandb:
                wandb.init(dir=os.path.join(exp_dir, "wandb"),
                           config=self.params.params, name=self.params.name, group=self.params.group, project=self.params.project, 
                           entity=self.params.entity, resume=self.params.resuming)
            self.build_and_run()



    def build_and_run(self):

        if self.sweep_id and dist.is_initialized():
            # Broadcast sweep config to other ranks
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            assert self.world_rank == rank
            if rank != 0:
                self.params = None
            self.params = comm.bcast(self.params, root=0)
            self.params.device = self.device # dont broadcast 0s device

        if self.world_rank == 0:
            logging.info(self.params.log())

        set_seed(self.params, self.world_size)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        # dump the yaml used
        if self.world_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in self.params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(self.params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams,  hpfile )

        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_path, dist.is_initialized(), train=True, pack=self.params.pack_data)
        self.val_data_loader, self.val_dataset, self.valid_sampler = get_data_loader(self.params, self.params.val_path, dist.is_initialized(), train=False, pack=self.params.pack_data)

        # domain grid
        self.domain = DomainXY(self.params)

        
        if self.params.model == 'fno':
            self.model = models.fno.fno(self.params).to(self.device)
        else:
            assert(False), "Error, model arch invalid."

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank])



        self.optimizer = set_optimizer(self.params, self.model)

        self.scheduler = set_scheduler(self.params, self.optimizer)

        if self.params.loss_func == "mse":
            self.loss_func = LossMSE(self.params, self.model)
        else:
            assert(False), "Error,  loss func invalid."

        self.iters = 0
        self.startEpoch = 0

        if hasattr(self.params, 'weights'):
            self.params.resuming = False
            logging.info("Loading IC weights %s"%self.params.weights)
            self.load_model(self.params.weights)

        if self.params.resuming:
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)

        self.epoch = self.startEpoch
        self.logs = {}
        self.train_loss = self.data_loss = self.bc_loss = self.pde_loss = self.grad = 0.0
        n_params = count_parameters(self.model)
        if self.log_to_screen:
            logging.info(self.model)
            logging.info('number of model parameters: {}'.format(n_params))

        # launch training
        self.train()

    def train(self):
        if self.log_to_screen:
            logging.info("Starting training loop...")
        best_loss = np.inf

        best_epoch = 0
        best_err = 1
        self.logs['best_epoch'] = best_epoch
        plot_figs = self.params.plot_figs

        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch = epoch
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            # train
            tr_time = self.train_one_epoch()
            val_time, fields = self.val_one_epoch()
            self.logs['wt_norm'] = self.get_model_wt_norm(self.model)

            if self.params.scheduler == 'reducelr':
                self.scheduler.step(self.logs['train_loss'])
            elif self.params.scheduler == 'cosine':
                self.scheduler.step()

            if self.logs['val_loss'] <= best_loss:
                is_best_loss = True
                best_loss = self.logs['val_loss']
                best_err = self.logs['val_err']
            else:
                is_best_loss = False
            self.logs['best_val_loss'] = best_loss
            self.logs['best_val_err'] = best_err

            best_epoch = self.epoch if is_best_loss else best_epoch
            self.logs['best_epoch'] = best_epoch

            if self.params.save_checkpoint:
                if self.world_rank == 0:
                    #checkpoint at the end of every epoch
                    if is_best_loss:
                        self.save_logs(tag="_best")
                    self.save_checkpoint(self.params.checkpoint_path, is_best=is_best_loss)

            if self.log_to_wandb:
                # log visualizations every epoch
                if plot_figs:
                    fig = vis_fields(fields, self.params, self.domain)
                    self.logs['vis'] = wandb.Image(fig)
                    plt.close(fig)
                self.logs['learning_rate'] = self.optimizer.param_groups[0]['lr']
                self.logs['time_per_epoch'] = tr_time
                wandb.log(self.logs, step=self.epoch+1)

            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec; with {}/{} in tr/val'.format(self.epoch+1, time.time()-start, tr_time, val_time))
                logging.info('Loss (total = data + bc + pde) {} = {} + {} + {}'.format(self.logs['train_loss'], self.logs['data_loss'],
                self.logs['bc_loss'], self.logs['pde_loss']))


        if self.log_to_wandb:
            wandb.finish()

    
    def get_model_wt_norm(self, model):
        n = 0
        for p in model.parameters():
            p_norm = p.data.detach().norm(2)
            n += p_norm.item()**2
        n = n**0.5
        return n

    def save_logs(self, tag=""):
        with open(os.path.join(self.params.experiment_dir, "logs"+tag+".txt"), "w") as f:
            f.write("epoch,{}\n".format(self.epoch))
            for k, v in self.logs.items():
                f.write("{},{}\n".format(k,v))


    def train_one_epoch(self):
        tr_time = 0
        self.model.train()

        # buffers for logs
        logs_buff = torch.zeros((6), dtype=torch.float32, device=self.device)
        self.logs['train_loss'] = logs_buff[0].view(-1)
        self.logs['data_loss'] = logs_buff[1].view(-1)
        self.logs['bc_loss'] = logs_buff[2].view(-1)
        self.logs['pde_loss'] = logs_buff[3].view(-1)
        self.logs['grad'] = logs_buff[4].view(-1)
        self.logs['tr_err'] = logs_buff[5].view(-1)


        for i, (inputs, targets) in enumerate(self.train_data_loader):
            self.iters += 1
            data_start = time.time()
            if not self.params.pack_data: # send to gpu if not already packed in the dataloader
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            tr_start = time.time()

            self.model.zero_grad()
            u = self.model(inputs)

            loss_data = self.loss_func.data(inputs, u, targets)
            loss_pde = self.loss_func.pde(inputs, u, targets)
            loss_bc = self.loss_func.bc(inputs, u, targets)
            loss = loss_data + loss_bc + loss_pde

            loss.backward()
            self.optimizer.step()

            grad_norm = compute_grad_norm(self.model.parameters())
            tr_err = l2_err(u.detach(), targets.detach())
    
            # add all the minibatch losses
            self.logs['train_loss'] += loss.detach()
            self.logs['data_loss'] += loss_data.detach()
            self.logs['bc_loss'] += loss_bc.detach()
            self.logs['pde_loss'] += loss_pde.detach()
            self.logs['grad'] += grad_norm
            self.logs['tr_err'] += tr_err

            tr_time += time.time() - tr_start

        self.logs['train_loss'] /= len(self.train_data_loader)
        self.logs['data_loss'] /= len(self.train_data_loader)
        self.logs['bc_loss'] /= len(self.train_data_loader)
        self.logs['pde_loss'] /= len(self.train_data_loader)
        self.logs['grad'] /= len(self.train_data_loader)
        self.logs['tr_err'] /= len(self.train_data_loader)

        logs_to_reduce = ['train_loss', 'data_loss', 'bc_loss', 'pde_loss', 'grad', 'tr_err']

        if dist.is_initialized():
            for key in logs_to_reduce:
                dist.all_reduce(self.logs[key].detach())
                # todo change loss to unscaled
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        return tr_time

    def val_one_epoch(self):
        self.model.eval() # need gradients
        #self.model.train() # need gradients
        val_start = time.time()

        logs_buff = torch.zeros((2), dtype=torch.float32, device=self.device)
        self.logs['val_err'] = logs_buff[0].view(-1)
        self.logs['val_loss'] = logs_buff[1].view(-1)
        idx = np.random.randint(0, len(self.val_data_loader))
        img_idx = np.random.randint(0, self.params.local_valid_batch_size)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_data_loader):
                if not self.params.pack_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                u = self.model(inputs)
                loss_data = self.loss_func.data(inputs, u, targets)
                loss_pde = self.loss_func.pde(inputs, u, targets)
                loss_bc = self.loss_func.bc(inputs, u, targets)
                loss = loss_data + loss_bc + loss_pde
                self.logs['val_err'] += l2_err(u.detach(), targets.detach())
                self.logs['val_loss'] += loss.detach()
                if i == idx: 
                    source = inputs[img_idx,0].detach().cpu().numpy() 
                    soln = targets[img_idx,0].detach().cpu().numpy()
                    pred = u[img_idx,0].detach().cpu().numpy()
                    pde_res = 0*pred
                    temp = 0*pred

        fields = [source, soln, pred, pde_res, temp]

        self.logs['val_loss'] /= len(self.val_data_loader)
        self.logs['val_err'] /= len(self.val_data_loader)
        if dist.is_initialized():
            for key in ['val_loss', 'val_err']:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        val_time = time.time() - val_start

        return val_time, fields

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler is not None else None)}, checkpoint_path)
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if  self.scheduler is not None else None)}, checkpoint_path.replace('.tar', '_best.tar'))

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank)) 
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)

        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank)) 
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)
 
    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def modify_bs_for_subsampling(self):
        '''Reduce batchsize for very small datasets'''
        sz = self.params.subsample
        if sz >= 512:
            fac = np.log2(sz) - 8
            self.params.batch_size = int(128/2**fac)
