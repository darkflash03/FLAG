import anndata
import pandas as pd
from PIL import Image
from dataset.utils import *

Image.MAX_IMAGE_PIXELS = None

class GraphDataset(Dataset):
    def __init__(self, data_config):
        super(GraphDataset, self).__init__()
        adata_path = data_config['adata_path']
        feature_path = data_config['feature_path']
        selected_genes = data_config['selected_genes']

        adata = anndata.read_h5ad(adata_path)
        slide_name = adata_path.split('/')[-1][:-5]
        self.slide_name = slide_name
        features = torch.load(feature_path, map_location='cpu')[:, 0, :].numpy()

        barcodes = adata.obs_names
        spatial_coords = adata.obsm["spatial"]

        expr = adata[:, selected_genes].X.toarray()
        coor_df = pd.DataFrame(spatial_coords, columns=["x", "y"], index=barcodes)
        expr_df = pd.DataFrame(expr, columns=selected_genes, index=barcodes)
        feature_dim = features.shape[1]
        feature_columns = [f'feature_{i}' for i in range(feature_dim)]
        feature_df = pd.DataFrame(features, columns=feature_columns, index=barcodes)

        spot_idx_to_remove = list(
            set(expr_df.index[expr_df.isnull().all(axis=1)]) | set(expr_df.index[expr_df.sum(axis=1) == 0]))
        spot_idx_to_keep = list(set(expr_df.index) - set(spot_idx_to_remove))

        ##loading llm embedding
        llm_cell_level_embeddings = pd.read_csv(data_config['llm_cell_level_embeddings_path'], index_col=0)
        emb_size = llm_cell_level_embeddings.shape[-1]

        keep_set = set(spot_idx_to_keep)

        final_mask_array = llm_cell_level_embeddings.index.isin(keep_set)

        final_spot_idx_to_keep = llm_cell_level_embeddings.index[final_mask_array].tolist()

        self.barcodes = np.array(final_spot_idx_to_keep)

        cols = [str(i) for i in range(emb_size)]
        self.llm_cell_level_embeddings = llm_cell_level_embeddings.loc[final_mask_array, cols].values

        self.llm_gene_level_embeddings = np.load(data_config['llm_gene_level_embeddings_path'], mmap_mode='r')[
            final_mask_array]
        self.llm_gene_level_mask = np.load(data_config['llm_gene_level_mask_path'], mmap_mode='r')[final_mask_array]

        self.llm_gene_level_embeddings = np.array(self.llm_gene_level_embeddings)
        self.llm_gene_level_mask = np.array(self.llm_gene_level_mask)

        print("cell embeddings: ", self.llm_cell_level_embeddings.shape)
        print("gene embeddings: ", self.llm_gene_level_embeddings.shape)

        self.cell_positions = coor_df.loc[final_spot_idx_to_keep][["x", "y"]].values
        self.cell_features = feature_df.loc[final_spot_idx_to_keep][feature_columns].values
        self.true_gene_expression = np.log2(expr_df.loc[final_spot_idx_to_keep].values + 1)

        self.crop_size = data_config['crop_size']
        self.stride_size = data_config['stride_size']

        max_x = int(np.max(self.cell_positions[:, 0]))
        max_y = int(np.max(self.cell_positions[:, 1]))

        self.wsi_width = max_x + self.crop_size
        self.wsi_height = max_y + self.crop_size

        self.sub_infos = self.__initialize__()


    def __initialize__(self):
        wsi_width, wsi_height = self.wsi_width, self.wsi_height
        print("wsi_width: ", wsi_width, "wsi_height: ", wsi_height)
        sub_infos = []
        for x in range(0, wsi_width, self.stride_size):
            for y in range(0, wsi_height, self.stride_size):
                x_start, y_start = x, y
                x_end, y_end = min(x + self.crop_size, wsi_width), min(y + self.crop_size, wsi_height)

                mask = (self.cell_positions[:, 0] >= x_start) & (self.cell_positions[:, 0] < x_end) & \
                       (self.cell_positions[:, 1] >= y_start) & (self.cell_positions[:, 1] < y_end)
                sub_cell_indices = np.where(mask)[0]

                if len(sub_cell_indices) < 1:  # skip cell_num < 1 patch
                    continue

                sub_barcodes = self.barcodes[sub_cell_indices]
                sub_cell_positions = self.cell_positions[sub_cell_indices]
                sub_cell_features = self.cell_features[sub_cell_indices]
                sub_gene_expression = self.true_gene_expression[sub_cell_indices]
                sub_llm_cell_level_embeddings = self.llm_cell_level_embeddings[sub_cell_indices]
                sub_llm_gene_level_embeddings = self.llm_gene_level_embeddings[sub_cell_indices]
                sub_llm_gene_level_mask = self.llm_gene_level_mask[sub_cell_indices]

                sort_key = sub_cell_positions[:, 0] * self.crop_size + sub_cell_positions[:, 1]
                sorted_idx = np.argsort(sort_key)

                sub_barcodes = sub_barcodes[sorted_idx]
                sub_cell_positions = sub_cell_positions[sorted_idx]
                sub_cell_features = sub_cell_features[sorted_idx]
                sub_gene_expression = sub_gene_expression[sorted_idx]
                sub_llm_cell_level_embeddings = sub_llm_cell_level_embeddings[sorted_idx]
                sub_llm_gene_level_embeddings = sub_llm_gene_level_embeddings[sorted_idx]
                sub_llm_gene_level_mask = sub_llm_gene_level_mask[sorted_idx]

                _, edge_attr = build_graph_with_all_effects_fully_connected(sub_cell_positions, sub_cell_features)

                sub_cell_num = sub_cell_positions.shape[0]
                edge_attr = edge_attr.reshape([sub_cell_num, sub_cell_num, -1])
                mask = torch.ones(sub_cell_num).bool()
                sub_gene_expression = torch.tensor(sub_gene_expression, dtype=torch.float)
                e_gt = self._pcc(sub_gene_expression, mask)

                sub_llm_cell_level_embeddings = torch.tensor(sub_llm_cell_level_embeddings, dtype=torch.float)
                sub_llm_gene_level_embeddings = torch.tensor(sub_llm_gene_level_embeddings, dtype=torch.float)
                sub_llm_gene_level_mask = torch.tensor(sub_llm_gene_level_mask, dtype=torch.float)

                sub_infos.append(
                    {
                        "x_gt": sub_gene_expression,
                        "e_gt": e_gt,
                        "cond_x": torch.tensor(sub_cell_features, dtype=torch.float),
                        "cond_e": torch.tensor(edge_attr, dtype=torch.float),
                        "mask": mask,
                        "llm_cell_level_embeddings": sub_llm_cell_level_embeddings,
                        "llm_gene_level_embeddings": sub_llm_gene_level_embeddings,
                        "llm_gene_level_mask": sub_llm_gene_level_mask,
                        "coord": torch.tensor(sub_cell_positions, dtype=torch.float),
                        "barcode": sub_barcodes,
                        "slide": [self.slide_name for i in range(len(sub_barcodes))],
                    }
                )
        return sub_infos

    def __len__(self):
        return len(self.sub_infos)

    def __getitem__(self, idx):
        return self.sub_infos[idx]

    @staticmethod
    def _pcc(x, mask):
        N, G = x.shape
        m = mask.float().unsqueeze(1)
        x = x * m
        mean = (x.sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0))
        xc = (x - mean) * m
        cov = (xc @ xc.T) / (G - 1)
        var = (xc.pow(2).sum(dim=1) / (G - 1)).clamp_min(1e-8)
        std = var.sqrt()
        pcc = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-8)
        pcc = torch.nan_to_num(pcc, nan=0.0, posinf=0.0, neginf=0.0)
        pcc = 0.5 * (pcc + pcc.T)
        pcc.fill_diagonal_(0.0)
        return pcc.unsqueeze(-1)