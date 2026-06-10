import torch

def fourier_time_emb(t: torch.Tensor, C: int) -> torch.Tensor:
    device = t.device
    freqs = torch.exp(torch.linspace(0, 6, C//2, device=device))
    ang = t[:, None] * freqs[None, :] * 2 * 3.14159
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

@torch.no_grad()
def model_score(config, model, x, e, t, cond_x, cond_e, mask):
    C = config['model_config']['d_model']
    t_emb = fourier_time_emb(t, C)
    sx, se = model(x, e, t_emb, cond_x, cond_e, mask)
    return sx, se

@torch.no_grad()
def symmetrize_e(e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    e = 0.5 * (e + e.transpose(1, 2))
    B, N, _, C = e.shape
    diag = torch.eye(N, device=e.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
    e = e.masked_fill(diag, 0.0)
    if mask is not None:
        m = mask.float().unsqueeze(-1)
        pair_m = m.unsqueeze(2) * m.unsqueeze(1)
        e = e * pair_m
    return e

@torch.no_grad()
def graph_diffusion_ode_sampler(config, model, sde, cond_x, cond_e, mask, steps=100):
    device = cond_x.device
    B, N, G = cond_x.size(0), cond_x.size(1), config['model_config']['node_dim']

    if config['mode'] == "graph_diffusion_fixed":
        cond_e_in = cond_e
        use_edge = False
    elif config['mode'] == "graph_diffusion_learned":
        cond_e_in = cond_e
        use_edge = True
    else:
        raise ValueError(config['mode'])

    x = torch.randn(B, N, G, device=device) * sde.sigma_max

    if use_edge:
        e = torch.randn(B, N, N, 1, device=device) * sde.sigma_max
        e = symmetrize_e(e, mask)
    else:
        e = torch.zeros(B, N, N, 1, device=device)

    t_end = config['sde_config']['t_min']
    ts = torch.linspace(1.0, t_end, steps + 1, device=device)

    for k in range(steps):
        t = ts[k].expand(B)
        t_next = ts[k + 1].expand(B)
        dt = (t_next - t)
        g = sde.diffusion(t)
        sx, se = model_score(config, model, x, e, t, cond_x, cond_e_in, mask)
        drift_x = -0.5 * (g.view(B, 1, 1) ** 2) * sx
        if use_edge:
            drift_e = -0.5 * (g.view(B, 1, 1, 1) ** 2) * se
        else:
            drift_e = torch.zeros_like(se)
        x_euler = x + drift_x * dt.view(B, 1, 1)
        e_euler = e + drift_e * dt.view(B, 1, 1, 1)

        # second evaluation at t_next
        g_next = sde.diffusion(t_next)
        sx2, se2 = model_score(config, model, x_euler, e_euler, t_next, cond_x, cond_e_in, mask)
        drift_x2 = -0.5 * (g_next.view(B, 1, 1) ** 2) * sx2
        if use_edge:
            drift_e2 = -0.5 * (g_next.view(B, 1, 1, 1) ** 2) * se2
        else:
            drift_e2 = torch.zeros_like(se2)
        x = x + 0.5 * (drift_x + drift_x2) * dt.view(B, 1, 1)
        if use_edge:
            e = e + 0.5 * (drift_e + drift_e2) * dt.view(B, 1, 1, 1)
            e = symmetrize_e(e, mask)

    # === Tweedie clean 投影（从 t_end -> t=0）===
    sigma_end = sde.sigma(ts[-1]).view(B, 1, 1)
    sx_end, se_end = model_score(config, model, x, e, ts[-1].expand(B), cond_x, cond_e, mask)
    x = x + (sigma_end ** 2) * sx_end
    e = e + (sigma_end.view(B, 1, 1, 1) ** 2) * se_end
    e = symmetrize_e(e, mask)
    return x, e