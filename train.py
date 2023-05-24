import os, sys, time
import argparse
import torch
import wandb
import matplotlib.pyplot as plt
import logging
import torch.distributed as dist
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.trainer import Trainer

if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/operators.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    trainer = Trainer(params, args)

    if args.sweep_id and trainer.world_rank==0:
        logging.disable(logging.CRITICAL)
        wandb.agent(args.sweep_id, function=trainer.launch, count=1, entity=trainer.params.entity, project=trainer.params.project) 
    else:
        trainer.launch()

    if dist.is_initialized():
        dist.barrier()

    logging.info('DONE')
