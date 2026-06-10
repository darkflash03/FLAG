import warnings
warnings.filterwarnings("ignore")
import argparse
from utils import *

from train_graph_diffusion import train_graph_diffusion
from train_graph_latent_diffusion import train_graph_latent_diffusion
from test_graph_diffusion import test_graph_diffusion
from test_graph_latent_diffusion import test_graph_latent_diffusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/graph_diffusion.yaml')

    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = config['pipeline']

    if pipeline == 'train_graph_latent_diffusion':
        train_graph_latent_diffusion(config)
    elif pipeline == 'train_graph_diffusion':
        train_graph_diffusion(config)
    elif pipeline == 'test_graph_diffusion':
        test_graph_diffusion(config)
    elif pipeline == 'test_graph_latent_diffusion':
        test_graph_latent_diffusion(config)
    else:
        raise ValueError(f'pipeline {pipeline} not recognized')

