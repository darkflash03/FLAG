from dataset import *
from sdes import *
from models import *
from samplers import *

def register_datasets():
    dataset_dict = {}
    dataset_dict['graph_dataset'] = GraphDataset
    return dataset_dict

def register_sdes():
    sde_dict = {}
    sde_dict['vesde'] = VESDE
    return sde_dict

def register_models():
    model_dict = {}
    model_dict['dit'] = Dit
    model_dict['dit_repa'] = DitRepa
    model_dict['graph_dit'] = GraphDit
    model_dict['graph_model'] = GraphModel
    model_dict['graph_dit_repa'] = GraphDitRepa
    return model_dict

def register_samplers():
    samplers_dict = {}
    samplers_dict['graph_ode_sampler'] = graph_ode_sampler
    samplers_dict['graph_diffusion_ode_sampler'] = graph_diffusion_ode_sampler
    return samplers_dict