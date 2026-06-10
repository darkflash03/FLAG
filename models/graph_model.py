import torch
import torch.nn as nn
from models.layer import GraphTransformerBlock

class GraphModel(nn.Module):
    def __init__(self, model_config):
        super(GraphModel, self).__init__()
        self.model_config = model_config
        G = self.model_config['node_dim']
        C = self.model_config['d_model']
        self.x_in = nn.Sequential(
            nn.Linear(G, C),
            nn.GELU(),
            nn.Linear(C, C),
        )
        self.e_in = nn.Sequential(
            nn.Linear(self.model_config['edge_dim'], C),
            nn.GELU(),
            nn.Linear(C, C),
        )
        self.t_in = nn.Sequential(
            nn.Linear(C, C),
            nn.SiLU(),
            nn.Linear(C, C)
        )

        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                node_dim=C,
                edge_dim=C,
                attn_dim=C,
                t_dim=C,
                cond_x_dim=self.model_config['node_cond_dim'],
                cond_e_dim=(self.model_config['edge_cond_dim'] if self.model_config['use_cond_e_in_attn'] else 0),
                n_heads=self.model_config['n_heads'],
                dropout=self.model_config['dropout'],
                gated_ff=True,
                ada_norm=self.model_config['ada_norm']
            )
            for _ in range(self.model_config['n_layers'])
        ])


        self.x_out = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C, G)
        )
        self.e_out = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C, self.model_config['edge_dim'])
        )

    def forward(self, x, e, t_emb, cond_x, cond_e, mask):
        B, N, _ = x.shape
        diag = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(diag, 0.0)

        x = self.x_in(x)
        e = self.e_in(e)
        t = self.t_in(t_emb)

        for blk in self.blocks:
            x, e = blk(x, e, t, cond_x, cond_e if self.model_config['use_cond_e_in_attn'] else None, mask)
        sx = self.x_out(x)
        se = self.e_out(e)
        se = 0.5*(se + se.transpose(1, 2))
        se = se.masked_fill(diag, 0.0)
        return sx, se