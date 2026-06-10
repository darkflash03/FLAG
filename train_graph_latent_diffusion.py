from utils import *
from datetime import datetime
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dataset.utils import *
from losses import *

def train_graph_latent_diffusion(config):
    print("start train_graph_latent_diffusion pipeline ...")

    seeding(config['seed'])
    device = config['device']
    mode = config['mode']

    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']

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

    train_dataset, selected_genes = load_dataset(data_config, mode='train')
    gene_size = len(selected_genes)

    valid_dataset, _ = load_dataset(data_config, mode='valid')

    train_loader = make_loader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = make_loader(valid_dataset, batch_size=train_config['batch_size'], shuffle=False)

    model = load_model(model_config, gene_size).to(device)
    ema = EMA(model, decay=0.999)
    sde = load_sde(config['sde_config'])
    sampler_fn = load_sampler_fn(config['sde_config'])

    opt = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])

    best_pcc = -1
    epochs_no_improve = 0
    patience = train_config['patience']

    accum_step = 0

    for epoch in range(train_config['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}|")
        total_loss = 0
        epoch_gene_loss = 0
        epoch_cell_embed_loss = 0
        epoch_gene_embed_loss = 0
        for batch in pbar:
            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    continue
            if config['mode'] == "graph_fixed":
                loss, logs, _ = graph_fixed_loss(model, sde, batch, config)
            elif config['mode'] == "repa_graph_fixed":
                loss, logs, _ = repa_graph_fixed_loss(model, sde, batch, config)
            elif config['mode'] == "repa_cell_graph_fixed":
                loss, logs, _ = repa_cell_graph_fixed_loss(model, sde, batch, config)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
            accum_step += 1

            if accum_step % train_config['gradient_accumulate_every'] == 0:
                opt.step()
                opt.zero_grad()
                ema.update()
                writer.add_scalar(f"Train-Step/loss", loss, accum_step)

            pbar.set_postfix({"Loss": float(loss), **{k: float(v) for k, v in logs.items()}})
            total_loss += float(loss)
            epoch_gene_loss += float(logs['loss_x'])

            if config['mode'] == "repa_graph_fixed":
                epoch_gene_embed_loss += float(logs['loss_gene_embed'])
            elif config['mode'] == "repa_cell_graph_fixed":
                epoch_cell_embed_loss += float(logs['loss_cell_embed'])


        if config['mode'] == "graph_fixed":
            logger.info(f"Epoch={epoch} | "
                        f"Average Training Loss: {total_loss / len(train_loader):.5f} |"
                        f"Average Node Loss: {epoch_gene_loss / len(train_loader):.5f} |"
                        )
        elif config['mode'] == "repa_graph_fixed":
            logger.info(f"Epoch={epoch} | "
                        f"Average Training Loss: {total_loss / len(train_loader):.5f} |"
                        f"Average Node Loss: {epoch_gene_loss / len(train_loader):.5f} |"
                        f"Average gene_embed Loss: {epoch_gene_embed_loss / len(train_loader):.5f} |"
                        )
            writer.add_scalar(f"Train-Epoch/gene_embed_loss", epoch_gene_embed_loss / len(train_loader), epoch)
        elif config['mode'] == "repa_cell_graph_fixed":
            logger.info(f"Epoch={epoch} | "
                        f"Average Training Loss: {total_loss / len(train_loader):.5f} |"
                        f"Average Node Loss: {epoch_gene_loss / len(train_loader):.5f} |"
                        f"Average cell_embed Loss: {epoch_cell_embed_loss / len(train_loader):.5f} |"
                        )
            writer.add_scalar(f"Train-Epoch/cell_embed_loss", epoch_cell_embed_loss / len(train_loader), epoch)

        writer.add_scalar(f"Train-Epoch/total_loss", total_loss / len(train_loader), epoch)
        writer.add_scalar(f"Train-Epoch/node_loss", epoch_gene_loss / len(train_loader), epoch)

        if epoch % train_config['eval_interval'] == 0:
            model.eval()
            ema.apply_shadow()
            with torch.no_grad():
                all_gt_genes = []
                all_pred_genes = []

                total_val_batches = len(valid_loader)
                print("total_val_batches: ", total_val_batches)
                sample_num = min(5, total_val_batches)
                subset_indices = set(random.sample(range(total_val_batches), sample_num))

                for i, batch in enumerate(tqdm(valid_loader, desc=f"Quick Val (Sampled {sample_num})")):
                    if i not in subset_indices:
                        continue
                    for k in batch:
                        try:
                            batch[k] = batch[k].to(device)
                        except:
                            continue

                    pred_x = sampler_fn(config, ema.model, sde, batch['cond_x'], batch['cond_e'], batch['mask'], steps=20,
                                        mode=mode).squeeze(0).cpu()
                    x_gt = batch['x_gt'].squeeze(0).cpu()

                    all_gt_genes.append(x_gt.numpy())
                    all_pred_genes.append(pred_x.numpy())

            ema.restore()
            val_metrics = compute_metrics(all_pred_genes, all_gt_genes)
            nan_count = val_metrics['nan_count']
            valid_count = len(selected_genes) - nan_count

            logger.info(f"Epoch={epoch} | Quick Validation Metrics: MSE={val_metrics['MSE']:.5f}, "
                        f"MAE={val_metrics['MAE']:.5f}, "
                        f"nan_count={val_metrics['nan_count']:.5f}, "
                        f"PCC-10={val_metrics['PCC-10']:.5f}, "
                        f"PCC-50={val_metrics['PCC-50']:.5f}, "
                        f"PCC-{valid_count}={val_metrics[f'PCC-{valid_count}']:.5f}")

            writer.add_scalar(f"Validation/MSE", val_metrics['MSE'], epoch)
            writer.add_scalar(f"Validation/MAE", val_metrics['MAE'], epoch)
            writer.add_scalar(f"Validation/PCC-{valid_count}", val_metrics[f'PCC-{valid_count}'], epoch)

            val_pcc = val_metrics['PCC-188']

            if val_pcc > best_pcc:
                best_pcc = val_pcc
                epochs_no_improve = 0

            else:
                epochs_no_improve += 1
                logger.info(
                    f"No improvement in validation pcc, epochs without improvement: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement")
                break

            checkpoint_path = save_checkpoint(model, ema, opt, result_folder, epoch)

    writer.close()
    logger.info(f"Training completed. Best validation pcc: {best_pcc:.5f}, Checkpoint: {checkpoint_path}")