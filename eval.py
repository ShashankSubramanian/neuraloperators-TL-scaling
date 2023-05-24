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
from utils.inferencer import Inferencer
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
import numpy as np


if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/operators.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--sweep", default='none', type=str)
    args = parser.parse_args()


    params = YParams(os.path.abspath(args.yaml_config), args.config)
    logging.info("Starting config {}".format(args.config))

    if hasattr(params, 'weights'):
        logging.info("with weights {}".format(params.weights))
    else:
        logging.info("no weight vector")

    inferencer = Inferencer(params, args)
    if inferencer.world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams,  hpfile )
    inferencer.launch()
    if dist.is_initialized():
        dist.barrier()
    logging.info("Finished config {}".format(args.config))


    logging.info('DONE')
