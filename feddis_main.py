"""
pmvae_main.py

main entry point to start experiments for Nature MI submission

"""
import yaml
import logging
import argparse
import sys
import os
import wandb
import torch
from datetime import datetime

sys.path.insert(0, '/MONAI')
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))

from monai.ILIA.projects.feddis.FedPlanner import FedDisFedPlanner


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--log_level', type=str, default='INFO', metavar='L',
                        help='log level from : ["INFO", "DEBUG", "WARNING", "ERROR"]')

    parser.add_argument('--config_path', type=str, default='./config/feddis_config.yaml', metavar='C',
                        help='path to configuration yaml file')
    return parser


if __name__ == "__main__":
    arg_parser = add_args(argparse.ArgumentParser(description='FedBrain'))
    args = arg_parser.parse_args()
    if args.log_level == 'INFO': logging.basicConfig(level=logging.INFO)
    elif args.log_level == 'DEBUG': logging.basicConfig(level=logging.DEBUG)
    elif args.log_level == 'WARNING': logging.basicConfig(level=logging.WARNING)
    elif args.log_level == 'ERROR': logging.basicConfig(level=logging.ERROR)
    config_file = None
    logging.info(
        '------------------------------- FEDERATED UNSUPERVISED ANOMALY DETECTION -------------------------------')
    try:
        stream_file = open(args.config_path, 'r')
        config_file = yaml.load(stream_file, Loader=yaml.FullLoader)
    except:
        logging.error('[FedDis::main] ERROR: Invalid configuration file at: {}'.format(args.config_path))
        exit()

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    config_file['collaborator']['params']['checkpoint_path'] += date_time
    planner = FedDisFedPlanner(config_file=config_file)  # Run experiment
    exp_name = planner.fed_plan['experiment']['name']
    method_name = planner.fed_plan['name']
    logging.info(exp_name, method_name)

    config_dict = dict(
        yaml=config_file,
        params=planner.fed_plan
    )
    wandb.init(project=planner.fed_plan['experiment']['name'], name=planner.fed_plan['name'],
               config=config_dict, id=date_time)

    planner.init_components(log_wandb=True)
    if planner.fed_plan['experiment']['task'] == 'train':
        planner.start_training()
    else:
        global_models = dict()
        for client_name in planner.fed_plan['experiment']['models']:
            logging.info('Loaded client from: ' + planner.fed_plan['experiment']['models'][client_name])
            global_models[client_name] = torch.load(planner.fed_plan['experiment']['models'][client_name])
        planner.start_evaluations(global_models)
