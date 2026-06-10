from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset.utils import *
from losses import *

def test_graph_diffusion(config):
    print("start test_graph_diffusion pipeline ...")
    seeding(config['seed'])
    device = config['device']
    mode = config['mode']

    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']
    test_config = config['test_config']

    dataset_name = data_config['dataset_name']
    crop_size = data_config['crop_size']
    model_name = model_config['name']

    result_folder = os.path.join(train_config['result_dir'], f"{mode}/{dataset_name}_crop_size_{crop_size}",
                                 model_name)
    os.makedirs(result_folder, exist_ok=True)

    save_config(os.path.join(result_folder, 'config.yaml'), config)

    # Initialize TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(result_folder, "runs", current_time)
    writer = SummaryWriter(log_dir=log_dir)

    logger = create_logger(log_dir)
    logger.info(f"Experiment directory created at {result_folder}")

    test_dataset, selected_genes = load_dataset(data_config, mode='test')
    gene_size = len(selected_genes)

    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False)
    model = load_model(model_config, gene_size).to(device)
    ema = EMA(model, decay=0.999)
    ckpt_path = os.path.join(result_folder, test_config['ckpt'])
    print("loading ckpt from: ", ckpt_path)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    ema.shadow = ckpt['ema']
    model.eval()

    sde = load_sde(config['sde_config'])
    sampler_fn = load_sampler_fn(config['sde_config'])

    all_gt_coords = []
    all_gt_genes = []
    all_pred_genes = []
    loss_edge = 0

    ema.apply_shadow()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    continue
            batch_gt_genes = batch['x_gt'].squeeze(0).cpu()
            batch_gt_coords = batch['coords'].squeeze(0).cpu()
            all_gt_coords.append(batch_gt_coords.numpy())
            all_gt_genes.append(batch_gt_genes.numpy())
            batch_pred_genes = 0
            batch_loss_edge = 0
            for ix in range(test_config['sample_num']):
                pred_x, pred_e = sampler_fn(config, ema.model, sde, batch['cond_x'], batch['cond_e'], batch['mask'])
                pred_x = pred_x.squeeze(0).cpu()
                pred_x[pred_x < 0] = 0
                batch_pred_genes += pred_x
                batch_loss_edge += float(edge_loss(pred_e, batch, config))
            all_pred_genes.append(batch_pred_genes / test_config['sample_num'])
            loss_edge += batch_loss_edge / test_config['sample_num']

        ema.restore()
        if test_config['mode'] == 'eval':
            val_metrics = compute_metrics(all_pred_genes, all_gt_genes)
            nan_count = val_metrics['nan_count']
            valid_count = len(selected_genes) - nan_count
            logger.info(f"Test Metrics: MSE={val_metrics['MSE']:.5f}, "
                        f"MAE={val_metrics['MAE']:.5f}, "
                        f"nan_count={val_metrics['nan_count']:.5f},"
                        f"PCC-10={val_metrics['PCC-10']:.5f}, "
                        f"PCC-50={val_metrics['PCC-50']:.5f}, "
                        f"PCC-{valid_count}={val_metrics[f'PCC-{valid_count}']:.5f}")

            logger.info(f"Test Edge loss={loss_edge / len(test_loader):.5f}, ")

            metric_path = ckpt_path[:-3] + '_eval.csv'
            eval_result = {
                'MSE': [val_metrics['MSE']],
                'MAE': [val_metrics['MAE']],
                'PCC-10': [val_metrics['PCC-10']],
                'PCC-50': [val_metrics['PCC-50']],
                f'PCC-{valid_count}': [val_metrics[f'PCC-{valid_count}']]
            }
            eval_result = pd.DataFrame(eval_result)
            eval_result.to_csv(metric_path, index=False)

        else:
            raise ValueError(f"Test mode {test_config['mode']} not supported")