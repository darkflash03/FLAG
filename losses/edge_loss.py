import torch

def edge_loss(pred_e, batch, config):
    x = batch["x_gt"]
    e = batch["e_gt"]
    mask = batch["mask"]
    B, N, G = x.shape
    device = x.device

    # mode switches
    if config['mode'] == "graph_diffusion_fixed":
        use_edge = False
    elif config['mode'] == "graph_diffusion_learned":
        use_edge = True
    else:
        raise ValueError(config['mode'])

    # node loss
    m = mask.float().unsqueeze(-1)
    pair_m = (m.unsqueeze(2) * m.unsqueeze(1))
    diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
    if use_edge:
        loss_e = (((pred_e - e)**2) * pair_m).masked_fill(diag, 0.0).sum() / \
                 pair_m.masked_fill(diag, 0.0).sum().clamp_min(1.0)
    else:
        loss_e = torch.tensor(0.0, device=device)
    return loss_e