import torch.nn as nn
import torch
from models.graph_model import GraphModel
from models.utils import GeneJointEmbedding, DiTBlock, FinalLayer, TimestepEmbedder

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

class GraphDitRepa(nn.Module):
    def __init__(self, model_config):
        super(GraphDitRepa, self).__init__()
        self.graph_backbone = GraphModel(model_config)

        self.model_config = model_config
        self.d_gene = self.model_config['d_gene']  ##gene映射到的维度
        self.time_dim = self.model_config['d_model']

        img_dim = model_config['node_cond_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.node_dim = model_config['node_dim']
        num_heads = model_config['num_heads']
        dit_num_blocks = model_config['dit_num_blocks']
        mlp_ratio = model_config['mlp_ratio']

        self.encoder_layer = model_config['encoder_layer']
        self.gene_joint_embed = GeneJointEmbedding(self.node_dim, self.hidden_dim)

        self.time_embed = TimestepEmbedder(self.hidden_dim)
        # label embedding (input label is already in embedding form, here just reorganize the size using linear layer)
        self.label_embed = nn.Sequential(
            nn.Linear(img_dim, img_dim, bias=True),
            nn.SiLU(),
            nn.Linear(img_dim, self.hidden_dim, bias=True),
        )

        # graph node -> cond 投影
        self.node2gene = nn.Sequential(
            nn.Linear(self.node_dim, self.d_gene, bias=True),
            nn.SiLU(),
            nn.Linear(self.d_gene, self.hidden_dim, bias=True)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(dit_num_blocks)
        ])

        self.projectors = nn.ModuleList([
            build_mlp(self.hidden_dim, model_config['projector_dim'], model_config['z_dim'])
        ])

        self.final_layer = FinalLayer(self.hidden_dim)
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

    def forward(self, x, e, t, cond_x, cond_e, mask):
        t_emb = self.time_embed(t)  # (N, hidden_dim) [time_ebd]
        z, se = self.graph_backbone(x, e, t_emb, cond_x, cond_e, mask)
        z = z.squeeze(0)
        cond_graph = self.node2gene(z)

        x = x.squeeze(0).float()
        x = self.gene_joint_embed(x)  # (N, NumGene, hidden_dim) [gene_joint_ebd]

        batch_size = x.size(0)

        c = t_emb.squeeze() + cond_graph  # (N, hidden_dim) [condition]
        for i, block in enumerate(self.blocks):
            x = block(x, c)  # (N, NumGene, hidden_dim)
            if (i + 1) == self.encoder_layer:
                zs = [projector(x.reshape(-1, self.hidden_dim)).reshape(batch_size, self.node_dim, -1) for projector in self.projectors]
        x = self.final_layer(x, c).squeeze(1)  # (N, NumGene)
        return x, zs