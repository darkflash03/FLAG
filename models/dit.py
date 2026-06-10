import torch
import torch.nn as nn
from models.utils import GeneJointEmbedding, TimestepEmbedder, DiTBlock, FinalLayer

class Dit(nn.Module):
    def __init__(self, model_config):
        super(Dit, self).__init__()

        img_dim = model_config['img_dim']
        hidden_dim = model_config['hidden_dim']
        node_dim = model_config['node_dim']
        num_heads = model_config['num_heads']
        dit_num_blocks = model_config['dit_num_blocks']
        mlp_ratio = model_config['mlp_ratio']

        self.gene_joint_embed = GeneJointEmbedding(node_dim, hidden_dim)
        # time step embedding
        self.time_embed = TimestepEmbedder(hidden_dim)
        # label embedding (input label is already in embedding form, here just reorganize the size using linear layer)
        self.label_embed = nn.Sequential(
            nn.Linear(img_dim, img_dim, bias=True),
            nn.SiLU(),
            nn.Linear(img_dim, hidden_dim, bias=True),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(dit_num_blocks)
        ])

        self.final_layer = FinalLayer(hidden_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.label_embed[0].weight, std=0.02)
        nn.init.normal_(self.label_embed[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond_x, t):
        """
        Forward pass of DiT.
        x: (N, NumGene) tensor of inputs
        t: (N,) tensor of diffusion timesteps
        y: (N, 1024) tensor of conditions
        """
        x = x.float()
        y = cond_x.float()

        x = self.gene_joint_embed(x)  # (N, NumGene, hidden_dim) [gene_joint_ebd]
        t = self.time_embed(t)  # (N, hidden_dim) [time_ebd]
        y = self.label_embed(y)  # (N, hidden_dim) [label_ebd]
        c = t.squeeze() + y  # (N, hidden_dim) [condition]
        for block in self.blocks:
            x = block(x, c)  # (N, NumGene, hidden_dim)
        x = self.final_layer(x, c).squeeze(1)  # (N, NumGene)
        return x