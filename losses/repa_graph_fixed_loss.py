import torch
import torch.nn.functional as F

def repa_graph_fixed_loss(model, sde, batch, config):
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

    z = zs[0]  # (cells, G, D)
    #print("z: ", z.shape)
    zs_tilde = batch['llm_gene_level_embeddings'].reshape(cells, G, -1)
    #print("zs_tilde: ", zs_tilde.shape)
    zs_mask = batch['llm_gene_level_mask'].reshape(cells, G)
    #print("zs_mask: ", zs_mask.shape)

    z_tilde = zs_tilde  # (cells, G, D)
    mask = zs_mask > 0.5  # (cells, N) -> bool

    # 归一化，按最后一维
    z = F.normalize(z, dim=-1)
    z_tilde = F.normalize(z_tilde, dim=-1)

    # 逐元素点积得到余弦相似度： (B, N)
    cos = (z * z_tilde).sum(dim=-1)

    # 只对 mask==True 的位置取样，计算 -cos 的平均
    # 等价于原来每个有效 j 的 mean_flat(-(z_j * z_tilde_j).sum(-1)) 并在最后除以 z_cnt
    valid_cos = cos[mask]  # (M_valid,)
    z_cnt = valid_cos.numel()

    if z_cnt > 0:
        proj_loss = (-valid_cos).mean()
    else:
        proj_loss = torch.zeros([], device=cos.device, dtype=cos.dtype)

    logs = {"loss_x": loss_x.detach(), 'loss_gene_embed': proj_loss}
    loss = loss_x + 0.5 * proj_loss
    return loss, logs, None