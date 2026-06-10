import torch

@torch.no_grad()
def model_score(model, x, e, t, cond_x, cond_e, mask):
    sx = model(x, e, t, cond_x, cond_e, mask)
    return sx

@torch.no_grad()
def graph_ode_sampler(config, model, sde, cond_x, cond_e, mask, steps=100, mode='graph_fixed'):
    """Deterministic ODE sampler for VE-SDE using Heun's method (RK2)."""
    device = cond_x.device
    B, N, G = cond_x.size(0), cond_x.size(1), config['model_config']['node_dim']

    x = torch.randn(B, N, G, device=device) * sde.sigma_max

    e = torch.zeros(B, N, N, 1, device=device)

    t_end = config['sde_config']['t_min']
    ts = torch.linspace(1.0, t_end, steps + 1, device=device)

    for k in range(steps):
        t = ts[k].expand(B)
        t_next = ts[k + 1].expand(B)
        dt = (t_next - t)
        g = sde.diffusion(t)
        if mode == 'repa_graph_fixed' or mode == 'repa_cell_graph_fixed':
            sx, _ = model_score(model, x, e, t, cond_x, cond_e, mask)
        else:
            sx = model_score(model, x, e, t, cond_x, cond_e, mask)
        drift_x = -0.5 * (g.view(B, 1) ** 2) * sx
        x_euler = x + drift_x * dt.view(B, 1)
        # second evaluation at t_next
        g_next = sde.diffusion(t_next)
        if mode == 'repa_graph_fixed' or mode == 'repa_cell_graph_fixed':
            sx2, _ = model_score(model, x_euler, e, t_next, cond_x, cond_e, mask)
        else:
            sx2 = model_score(model, x_euler, e, t_next, cond_x, cond_e, mask)
        drift_x2 = -0.5 * (g_next.view(B, 1) ** 2) * sx2
        x = x + 0.5 * (drift_x + drift_x2) * dt.view(B, 1)

    # === Tweedie clean （from t_end -> t=0）===
    sigma_end = sde.sigma(ts[-1]).view(B, 1)
    t_final = ts[-1].expand(B)
    if mode == 'repa_graph_fixed' or mode == 'repa_cell_graph_fixed':
        sx_end, _ = model_score(model, x, e, t_final, cond_x, cond_e, mask)
    else:
        sx_end = model_score(model, x, e, t_final, cond_x, cond_e, mask)
    x = x + (sigma_end ** 2) * sx_end
    return x