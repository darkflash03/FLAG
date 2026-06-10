from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

class AdaLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim, zero_init=True):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_gb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim*2)
        )
        if zero_init:
            nn.init.zeros_(self.to_gb[-1].weight)
            nn.init.zeros_(self.to_gb[-1].bias)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x0 = self.ln(x)
        g,b = self.to_gb(z).chunk(2, dim=-1)
        while g.dim() < x0.dim():
            g = g.unsqueeze(1)
            b = b.unsqueeze(1)
        return x0*(1+g) + b

class CondNodePairProj(nn.Module):
    def __init__(self, cond_x_dim: int, attn_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads; self.head_dim = attn_dim//n_heads
        self.cx_q_mul = nn.Linear(cond_x_dim, attn_dim, bias=False)
        self.cx_k_mul = nn.Linear(cond_x_dim, attn_dim, bias=False)
        self.cx_q_add = nn.Linear(cond_x_dim, attn_dim, bias=False)
        self.cx_k_add = nn.Linear(cond_x_dim, attn_dim, bias=False)
        self.cx_out_mul = nn.Linear(cond_x_dim, attn_dim, bias=False)
        self.cx_out_add = nn.Linear(cond_x_dim, attn_dim, bias=False)
    def _reshape(self, t: Tensor) -> Tensor:
        return rearrange(t, "b n (h d) -> (b h) n d", h=self.n_heads)
    def pair_mul(self, cond_x: Tensor) -> Tensor:
        q = self._reshape(self.cx_q_mul(cond_x)); k = self._reshape(self.cx_k_mul(cond_x))
        return q.unsqueeze(2)*k.unsqueeze(1)
    def pair_add(self, cond_x: Tensor) -> Tensor:
        q = self._reshape(self.cx_q_add(cond_x)); k = self._reshape(self.cx_k_add(cond_x))
        return q.unsqueeze(2)+k.unsqueeze(1)
    def out_film(self, cond_x: Tensor) -> Tuple[Tensor,Tensor]:
        return self.cx_out_mul(cond_x), self.cx_out_add(cond_x)

class CondEdgeProj(nn.Module):
    def __init__(self, cond_e_dim: int, attn_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads; self.head_dim = attn_dim//n_heads
        self.ce_mul = nn.Linear(cond_e_dim, attn_dim, bias=False)
        self.ce_add = nn.Linear(cond_e_dim, attn_dim, bias=False)
        self.ce_out_mul = nn.Linear(cond_e_dim, attn_dim, bias=False)
        self.ce_out_add = nn.Linear(cond_e_dim, attn_dim, bias=False)
    def _reshape(self, t: Tensor) -> Tensor:
        return rearrange(t, "b n m (h d) -> (b h) n m d", h=self.n_heads)
    def pair_mul(self, cond_e: Tensor) -> Tensor:
        return self._reshape(self.ce_mul(cond_e))
    def pair_add(self, cond_e: Tensor) -> Tensor:
        return self._reshape(self.ce_add(cond_e))
    def out_film(self, cond_e: Tensor) -> Tuple[Tensor,Tensor]:
        return self.ce_out_mul(cond_e), self.ce_out_add(cond_e)

class GraphTransformerBlock(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 attn_dim: int,
                 t_dim: int,
                 cond_x_dim: int,
                 cond_e_dim: int,
                 n_heads=8,
                 dropout=0.1,
                 gated_ff=True,
                 ada_norm=True):
        super().__init__()
        assert attn_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = attn_dim//n_heads
        self.scale = self.head_dim ** -0.5
        self.ada_norm = ada_norm

        self.to_q = nn.Linear(node_dim, attn_dim, bias=False)
        self.to_k = nn.Linear(node_dim, attn_dim, bias=False)
        self.to_v = nn.Linear(node_dim, attn_dim, bias=False)

        self.to_e_mul = nn.Linear(edge_dim, attn_dim, bias=False)
        self.to_e_add = nn.Linear(edge_dim, attn_dim, bias=False)

        self.cx = CondNodePairProj(cond_x_dim, attn_dim, n_heads)
        self.ce = CondEdgeProj(cond_e_dim, attn_dim, n_heads)

        self.to_x_out = nn.Sequential(nn.Linear(attn_dim, node_dim), nn.Dropout(dropout))
        self.to_e_out = nn.Sequential(nn.Linear(attn_dim, edge_dim), nn.Dropout(dropout))

        def _ff(dim):
            if gated_ff:
                class GEGLU(nn.Module):
                    def __init__(self, d_in, d_hid):
                        super().__init__()
                        self.proj = nn.Linear(d_in, d_hid*2)
                    def forward(self, x):
                        x1,g = self.proj(x).chunk(2, dim=-1)
                        return F.gelu(g)*x1
                proj = GEGLU(dim, attn_dim*2)
            else:
                proj = nn.Sequential(nn.Linear(dim, attn_dim*2), nn.GELU())
            return nn.Sequential(nn.LayerNorm(dim), proj, nn.Dropout(dropout), nn.Linear(attn_dim*2, dim))
        self.ff_x = _ff(node_dim)
        self.ff_e = _ff(edge_dim)

        if ada_norm:
            self.z_fuse = nn.Sequential(nn.Linear(t_dim + node_dim + edge_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
            self.ga_x_norm = AdaLayerNorm(node_dim, t_dim)
            self.ga_e_norm = AdaLayerNorm(edge_dim, t_dim)
            self.ff_x_norm = AdaLayerNorm(node_dim, t_dim)
            self.ff_e_norm = AdaLayerNorm(edge_dim, t_dim)
        else:
            self.ga_x_norm = nn.LayerNorm(node_dim)
            self.ga_e_norm = nn.LayerNorm(edge_dim)
            self.ff_x_norm = nn.LayerNorm(node_dim)
            self.ff_e_norm = nn.LayerNorm(edge_dim)

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta  = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.delta = nn.Parameter(torch.tensor(0.1))

        self.pool_cx = nn.Linear(cond_x_dim, node_dim, bias=False)
        self.pool_ce = nn.Linear(cond_e_dim, edge_dim, bias=False)
        nn.init.zeros_(self.pool_cx.weight)
        nn.init.zeros_(self.pool_ce.weight)

    def _pooled(self, cond_x: Optional[Tensor], cond_e: Optional[Tensor], mask: Optional[Tensor], node_dim: int, edge_dim: int):
        B = mask.shape[0] if mask is not None else (cond_x.shape[0] if cond_x is not None else cond_e.shape[0])
        device = (cond_x if cond_x is not None else cond_e).device
        # light-weight linear pools created on-the-fly to keep API minimal
        if cond_x is None:
            px = torch.zeros(B, node_dim, device=device)
        else:
            if mask is None:
                px = self.pool_cx(cond_x).mean(dim=1)
            else:
                m = mask.float().unsqueeze(-1)
                px = (self.pool_cx(cond_x)*m).sum(dim=1)/m.sum(dim=1).clamp_min(1.0)
        if cond_e is None:
            pe = torch.zeros(B, edge_dim, device=device)
        else:
            if mask is None:
                pe = self.pool_ce(cond_e).mean(dim=(1,2))
            else:
                m = mask.float()
                pair_m = (m.unsqueeze(2)*m.unsqueeze(1)).unsqueeze(-1)
                pe = (self.pool_ce(cond_e)*pair_m).sum(dim=(1,2))/pair_m.sum(dim=(1,2)).clamp_min(1.0)
        return px, pe

    def forward(self, x: Tensor, e: Tensor, t_emb: Tensor, cond_x: Optional[Tensor], cond_e: Optional[Tensor], mask: Optional[Tensor]):
        B,N,node_dim = x.shape
        H = self.n_heads
        D = self.head_dim

        # Ada/Pre-Norm
        if self.ada_norm:
            px, pe = self._pooled(cond_x, cond_e, mask, node_dim, e.shape[-1])
            z = torch.cat([t_emb, px, pe], dim=-1)
            z = self.z_fuse(z)
            x_norm = self.ga_x_norm(x, z)
            e_norm = self.ga_e_norm(e, z)
        else:
            x_norm = self.ga_x_norm(x)
            e_norm = self.ga_e_norm(e)

        x_mask = mask.unsqueeze(-1) if mask is not None else None
        if x_mask is not None:
            e_mask1 = x_mask.unsqueeze(2)
            e_mask2 = x_mask.unsqueeze(1)
        else:
            e_mask1 = e_mask2 = None

        q = rearrange(self.to_q(x_norm), "b n (h d) -> (b h) n d", h=H)
        k = rearrange(self.to_k(x_norm), "b n (h d) -> (b h) n d", h=H)
        v = rearrange(self.to_v(x_norm), "b n (h d) -> (b h) n d", h=H)
        if x_mask is not None:
            q = q * rearrange(x_mask, "b n 1 -> (b) n 1")
            k = k * rearrange(x_mask, "b n 1 -> (b) n 1")
            v = v * rearrange(x_mask, "b n 1 -> (b) n 1")

        sim = q.unsqueeze(2) * k.unsqueeze(1) * (D ** -0.5)  # (B*H,N,N,D)

        e_mul = rearrange(self.to_e_mul(e_norm), "b n m (h d) -> (b h) n m d", h=H)
        e_add = rearrange(self.to_e_add(e_norm), "b n m (h d) -> (b h) n m d", h=H)
        if e_mask1 is not None:
            em = rearrange(e_mask1*e_mask2, "b n m 1 -> (b) n m 1")
            e_mul = e_mul*em
            e_add = e_add*em

        # conditional modulations (only if provided)
        if cond_e is not None:
            ce_mul = self.ce.pair_mul(cond_e)
            ce_add = self.ce.pair_add(cond_e)
            if e_mask1 is not None:
                em = rearrange(e_mask1*e_mask2, "b n m 1 -> (b) n m 1")
                ce_mul = ce_mul*em
                ce_add = ce_add*em
        else:
            ce_mul = ce_add = 0.
        if cond_x is not None:
            cx_mul = self.cx.pair_mul(cond_x)
            cx_add = self.cx.pair_add(cond_x)
            if e_mask1 is not None:
                em = rearrange(e_mask1*e_mask2, "b n m 1 -> (b) n m 1")
                cx_mul = cx_mul*em
                cx_add = cx_add*em
        else:
            cx_mul = cx_add = 0.

        sim = (1. + e_mul + self.alpha*ce_mul + self.beta*cx_mul) * sim + e_add + self.gamma*ce_add + self.delta*cx_add

        if e_mask1 is not None:
            pair_mask = (e_mask1 * e_mask2).bool()
            attn_mask = rearrange(pair_mask, "b n m 1 -> (b) n m 1").bool()
            sim = sim.masked_fill(~attn_mask, float("-inf"))
        attn = sim.softmax(dim=2)

        out = (attn * v.unsqueeze(1)).sum(dim=2)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=H)
        if cond_x is not None:
            mul, add = self.cx.out_film(cond_x)
            if x_mask is not None:
                mul = mul*x_mask
                add = add*x_mask
            out = (1.+mul)*out + add
        out = self.to_x_out(out)
        if x_mask is not None:
            out = out * x_mask
        x = x + out

        e_out = rearrange(sim, "(b h) n m d -> b n m (h d)", h=H)
        if cond_e is not None:
            mul, add = self.ce.out_film(cond_e)
            if e_mask1 is not None:
                mul = mul*(e_mask1*e_mask2)
                add = add*(e_mask1*e_mask2)
            e_out = (1.+mul)*e_out + add
        e_out = self.to_e_out(e_out)
        if e_mask1 is not None:
            e_out = e_out*(e_mask1*e_mask2)

        if self.ada_norm:
            x = x + self.ff_x(self.ff_x_norm(x, z))
            e = e + self.ff_e(self.ff_e_norm(e + e_out, z))
        else:
            x = x + self.ff_x(self.ff_x_norm(x))
            e = e + self.ff_e(self.ff_e_norm(e + e_out))
        return x, e