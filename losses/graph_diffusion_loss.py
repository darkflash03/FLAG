import torch
from dataset.graph_dataset import GraphDataset


def fourier_time_emb(t: torch.Tensor, C: int) -> torch.Tensor:
    device = t.device
    freqs = torch.exp(torch.linspace(0, 6, C//2, device=device))
    ang = t[:, None] * freqs[None, :] * 2 * 3.14159
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

def graph_diffusion_loss(model, sde, batch, config):
    x = batch["x_gt"]
    e = batch["e_gt"]
    cond_x = batch["cond_x"]
    cond_e = batch["cond_e"]
    mask = batch["mask"]
    B, N, G = x.shape
    device = x.device

    if config['mode'] == "graph_diffusion_fixed":
        cond_e_in = cond_e
        use_edge = False
    elif config['mode'] == "graph_diffusion_learned":
        cond_e_in = cond_e
        use_edge = True
    else:
        raise ValueError("mode must be graph_diffusion_fixed or graph_diffusion_learned")

    # time
    t = torch.rand(B, device=device)
    t_min = config['sde_config']['t_min']
    t = t_min + (1 - t_min) * t
    t_emb = fourier_time_emb(t, config['model_config']['d_model'])

    # perturb
    xt, nx, stdx = sde.perturb(x, t)
    if use_edge:
        et, ne, stde = sde.perturb(e, t)
    else:
        et = torch.zeros_like(e)
        stde = torch.ones_like(e)

    sx, se = model(xt, et, t_emb, cond_x, cond_e_in, mask)

    # x DSM
    wx = stdx ** 2
    target_x = (xt - x) / wx
    m = mask.float().unsqueeze(-1)
    loss_x = (wx * (sx + target_x) ** 2 * m).sum() / m.sum() / G  ##node上的loss

    # e DSM (optional)
    pair_m = (m.unsqueeze(2) * m.unsqueeze(1))
    diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
    if use_edge:
        we = stde ** 2
        target_e = (et - e) / we
        loss_e = ((we * (se + target_e) ** 2) * pair_m).masked_fill(diag, 0.0).sum() / \
                 pair_m.masked_fill(diag, 0.0).sum().clamp(min=1.0)          #edge上的loss
    else:
        loss_e = torch.tensor(0.0, device=device)

    if config['mode'] == "graph_diffusion_learned" and 'lambda_consistency' in config['train_config'].keys():
        x0_hat = xt + wx * sx
        e0_hat = et + we * se
        with torch.no_grad():
            pcc_x_pred = []
            for b in range(B):
                pcc_x_pred.append(GraphDataset._pcc(x0_hat[b], mask[b]))
            pcc_x_pred = torch.stack(pcc_x_pred, dim=0)
        l_cons = (e0_hat - pcc_x_pred).abs() * pair_m
        l_cons = l_cons.masked_fill(diag, 0.0).sum() / pair_m.masked_fill(diag, 0.0).sum().clamp(min=1.0)
    else:
        l_cons = torch.tensor(0.0, device=device)

    loss = loss_x + config['train_config']['lambda_edge'] * loss_e + config['train_config']['lambda_consistency'] * l_cons
    logs = {"loss_x": loss_x.detach(), "loss_e": loss_e.detach(), "loss_cons": l_cons.detach()}
    return loss, logs, None