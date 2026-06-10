import torch
import torch.nn.functional as F

def repa_cell_graph_fixed_loss(model, sde, batch, config):
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

    sx, zs = model(xt, et, t, cond_x, cond_e, mask)
    cells, _ = sx.shape

    # x DSM
    wx = stdx ** 2
    target_x = (xt - x) / wx
    m = mask.float().unsqueeze(-1)
    loss_x = (wx * (sx + target_x) ** 2 * m).sum() / m.sum() / G

    zs_tilde = batch['llm_cell_level_embeddings'].reshape(cells, -1)
    z = torch.mean(zs[0], dim=1)  # (cells, D)
    z_tilde = zs_tilde  # (cells, D)

    # 归一化，按最后一维
    z = F.normalize(z, dim=-1)
    z_tilde = F.normalize(z_tilde, dim=-1)

    # 逐元素点积得到余弦相似度： (B, N)
    cos = (z * z_tilde).sum(dim=-1)
    # print("cos: ", cos.shape)

    proj_loss = (-cos).mean()
    logs = {"loss_x": loss_x.detach(), 'loss_cell_embed': proj_loss}
    loss = loss_x + 0.5 * proj_loss
    return loss, logs, None