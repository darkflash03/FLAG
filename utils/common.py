import yaml
import numpy as np
import random
import os
import torch
from tqdm import tqdm
import logging
from .register import register_datasets, register_sdes, register_models, register_samplers
import pandas as pd

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        running_config = yaml.load(f, Loader=yaml.FullLoader)
    return running_config

def save_config(config_path, config):
    with open(config_path, 'w', encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("seeding done !")

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)

def load_dataset(data_config, mode='train'):
    if mode == 'train':
        slidename_lst = np.genfromtxt(data_config['train_slides_list'], dtype=str)
    elif mode == 'valid':
        slidename_lst = np.genfromtxt(data_config['valid_slides_list'], dtype=str)
    elif mode == 'test':
        slidename_lst = np.genfromtxt(data_config['test_slides_list'], dtype=str)
    else:
        raise ValueError(f'loading dataset mode {mode} not supported')

    try:
        print(f"{mode} slide num: ", len(slidename_lst))
    except:
        slidename_lst = [slidename_lst]
        print(f"{mode} slide num: ", len(slidename_lst))

    dataset_dict = register_datasets()
    if data_config['dataset_name'] not in dataset_dict.keys():
        raise ValueError(f'dataset {data_config["dataset_name"]} not supported')
    dataset_handler = dataset_dict[data_config['dataset_name']]

    if 'pca' in data_config['dataset_name']:
        data_config['stat_path'] = os.path.join(data_config['data_dir'], "processed_data", data_config['stat_filename'])
        data_config['transformer_path'] = os.path.join(data_config['data_dir'], "processed_data", data_config['transformer_filename'])

    selected_genes = np.genfromtxt(os.path.join(data_config['data_dir'], "processed_data", data_config['gene_list_filename']), dtype=str)

    datasets = []
    for slidename in tqdm(slidename_lst):
        data_config['wsi_path'] = os.path.join(data_config['data_dir'], 'wsis', f"{slidename}.tif")
        data_config['adata_path'] = os.path.join(data_config['data_dir'], "st", f"{slidename}.h5ad")
        data_config['feature_path'] = os.path.join(data_config['feature_dir'], f"{slidename}_{data_config['img_encoder_name']}.pt")
        data_config['selected_genes'] = selected_genes

        #llm's cell level and gene level files
        llm_cell_level_embeddings_path = os.path.join(data_config['llm_embedding_dir'], f"{slidename}.csv")
        llm_gene_level_embeddings_path = os.path.join(data_config['llm_embedding_dir'], f"{slidename}_gene_embeddings.npy")
        llm_gene_level_mask_path = os.path.join(data_config['llm_embedding_dir'], f"{slidename}_gene_mask.npy")

        if os.path.exists(llm_cell_level_embeddings_path):
            data_config['llm_cell_level_embeddings_path'] = llm_cell_level_embeddings_path
        if os.path.exists(llm_gene_level_embeddings_path):
            data_config['llm_gene_level_embeddings_path'] = llm_gene_level_embeddings_path
        if os.path.exists(llm_gene_level_mask_path):
            data_config['llm_gene_level_mask_path'] = llm_gene_level_mask_path

        if 'supervise_predict_dir' in data_config.keys():
            supervise_predicts_path = os.path.join(data_config['supervise_predict_dir'], f"{slidename}.csv")
            if os.path.exists(supervise_predicts_path):
                data_config['supervise_predicts_path'] = supervise_predicts_path

        dataset = dataset_handler(data_config)
        datasets.append(dataset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    return dataset, selected_genes

def load_model(model_config, gene_size):
    model_dict = register_models()
    model_config['node_dim'] = gene_size
    model = model_dict[model_config['name']](model_config)
    return model

def load_sde(sde_config):
    sde_dict = register_sdes()
    sde = sde_dict[sde_config['name']](sde_config)
    return sde

def load_sampler_fn(sde_config):
    samplers_dict = register_samplers()
    sampler_fn = samplers_dict[sde_config['sampler']]
    return sampler_fn

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def save_checkpoint(model, ema, optimizer, checkpoint_dir, epoch):
    if ema is not None:
        checkpoint = {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "opt": optimizer.state_dict()
        }
    else:
        checkpoint = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict()
        }
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def save_results(barcodes, preds, slides, selected_genes, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    barcodes = np.concatenate(barcodes, axis=0).squeeze()
    slides = np.concatenate(slides, axis=0).squeeze()
    preds = np.concatenate(preds, axis=0).squeeze()

    if preds.shape[0] != len(barcodes) or len(slides) != len(barcodes):
        raise ValueError("barcodes, slides, preds amounts not same!")

    if preds.shape[1] != len(selected_genes):
        raise ValueError("preds cols not match selected_genes amount")

    unique_slides = np.unique(slides)
    for slide in unique_slides:
        mask = slides == slide
        preds_slide = preds[mask]
        barcodes_slide = barcodes[mask]
        df = pd.DataFrame(preds_slide, index=barcodes_slide, columns=selected_genes)
        filename = f"{slide}.csv"
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=True)