import torch

def graph_fixed_loss(model, sde, batch, config):
    x = batch["x_gt"]
    e = batch["e_gt"]
    cond_x = batch["cond_x"]
    cond_e = batch["cond_e"]
    mask = batch["mask"]
    B, N, G = x.shape
    device = x.device

    t = torch.rand(B, device=device)
    t_min = config['sde_config']['t_min']
    t = t_min + (1 - t_min) * t

    xt, nx, stdx = sde.perturb(x, t)
    et = torch.zeros_like(e)

    sx = model(xt, et, t, cond_x, cond_e, mask)

    # x DSM
    wx = stdx ** 2
    target_x = (xt - x) / wx
    m = mask.float().unsqueeze(-1)
    loss_x = (wx * (sx + target_x) ** 2 * m).sum() / m.sum() / G
    loss = loss_x

    logs = {"loss_x": loss_x.detach()}
    return loss, logs, None
