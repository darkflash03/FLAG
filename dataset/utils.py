from torch_geometric.data import Batch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    batch_graph_list = []
    for info in batch:
        if info is None:
            continue
        batch_graph_list.append(info['subgraphs'])
    batch_graph = Batch.from_data_list(batch_graph_list)
    return batch_graph

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def build_graph_with_all_effects_fully_connected(cell_positions, cell_features):
    """
    构建全连接图，每个节点与所有其他节点（除自身）连接。
    Args:
        cell_positions: 细胞空间坐标，形状 (n_cells, 2)
    Returns:
        edge_index: 边索引，形状 (2, n_cells * n_cells)
        edge_attr: 边属性，包含距离权重 形状 (n_cells * n_cells, 1)
    """
    n_cells = len(cell_positions)
    edge_index = []
    dist_weights = []
    img_sims = []

    # 为每对节点 (i, j) 创建边
    for i in range(n_cells):
        for j in range(n_cells):
            if i == j:  # 跳过自连接
                edge_index.append([i, i])
                dist_weights.append(0)
                img_sims.append(0)
                continue
            edge_index.append([i, j])
            # 计算距离权重 (高斯核)
            dist = np.linalg.norm(cell_positions[i] - cell_positions[j])
            sigma = 112 * 2  # 默认patch大小224 * 224
            weight = np.exp(-dist**2 / (2 * sigma**2))
            dist_weights.append(weight)

            # 图像相似度 (保持cosine，模式匹配)
            img_sim = cosine_sim(cell_features[i], cell_features[j])
            img_sims.append(img_sim)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    dist_weights = torch.tensor(dist_weights, dtype=torch.float).unsqueeze(1)
    img_sims = torch.tensor(img_sims, dtype=torch.float).unsqueeze(1)
    edge_attr = torch.cat([dist_weights, img_sims], dim=1)
    return edge_index, edge_attr

def make_loader(dataset, batch_size, shuffle=True):
    def _collate(batch):
        maxN = max(b["x_gt"].shape[0] for b in batch)
        G = batch[0]["x_gt"].shape[1]
        Cx = batch[0]["cond_x"].shape[1]
        Ce = batch[0]["cond_e"].shape[2]
        llm_embed_size = batch[0]["llm_cell_level_embeddings"].shape[-1]

        B = len(batch)
        x = torch.zeros(B, maxN, G)
        cond_x = torch.zeros(B, maxN, Cx)
        cond_e = torch.zeros(B, maxN, maxN, Ce)
        e = torch.zeros(B, maxN, maxN, 1)
        mask = torch.zeros(B, maxN, dtype=torch.bool)
        llm_cell_level_embeddings = torch.zeros(B, maxN, llm_embed_size)
        llm_gene_level_embeddings = torch.zeros(B, maxN, G, llm_embed_size)
        llm_gene_level_mask = torch.zeros(B, maxN, G)

        if 'supervise_predicts' in batch[0].keys():
            supervise_predicts = torch.zeros(B, maxN, G)

        coords = torch.zeros(B, maxN, 2)
        barcodes = []
        slides = []

        for i,b in enumerate(batch):
            n = b["x_gt"].shape[0]
            x[i,:n] = b["x_gt"]
            cond_x[i,:n] = b["cond_x"]
            cond_e[i,:n,:n] = b["cond_e"]
            e[i,:n,:n] = b["e_gt"]
            mask[i,:n] = b["mask"]
            llm_cell_level_embeddings[i, :n, :] = b["llm_cell_level_embeddings"]
            llm_gene_level_embeddings[i, :n, :, :] = b["llm_gene_level_embeddings"]
            llm_gene_level_mask[i, :n, :] = b["llm_gene_level_mask"]

            if 'supervise_predicts' in b.keys():
                supervise_predicts[i, :n, :] = b["supervise_predicts"]

            coords[i, :n, :] = b["coord"]
            barcodes.append(b["barcode"])
            slides.append(b["slide"])

        # enforce symmetry & zero-diag
        cond_e = 0.5*(cond_e + cond_e.transpose(1,2))
        e = 0.5*(e + e.transpose(1,2))

        for i in range(B):
            e[i].diagonal(dim1=0, dim2=1).zero_()

        barcodes = np.concatenate(barcodes, axis=0)
        slides = np.concatenate(slides, axis=0)
        data_dict = {
            "x_gt": x,
            "e_gt": e,
            "cond_x": cond_x,
            "cond_e": cond_e,
            "mask": mask,
            "llm_cell_level_embeddings": llm_cell_level_embeddings,
            "llm_gene_level_embeddings": llm_gene_level_embeddings,
            "llm_gene_level_mask": llm_gene_level_mask,
            "coords": coords,
            "barcodes": barcodes,
            "slides": slides
        }
        if 'supervise_predicts' in batch[0].keys():
            data_dict["supervise_predicts"] = supervise_predicts
        return data_dict
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)