import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

# === Dataset for Peptides (Updated for HLA-A26-01) ===
class TestPeptideDataset(InMemoryDataset):
    def __init__(self, root="dataset/test_peptide_HLA-B51-01", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return "data_test_peptide_HLA-B51-01.pt"

# === Dataset for MHCs (Updated for HLA-A26-01) ===
class TestMHCDataset(InMemoryDataset):
    def __init__(self, root="dataset/test_mhc_HLA-B51-01", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return "data_test_mhc_HLA-B51-01.pt"

# === CNN Layer ===
class CNN(nn.Module):
    def __init__(self, hidden_size):
        super(CNN, self).__init__()
        kernel_size = [7, 11, 15]
        self.conv_xt_1 = nn.Conv1d(128, 64, kernel_size[0], padding=(kernel_size[0]-1)//2)
        self.conv_xt_2 = nn.Conv1d(128, 64, kernel_size[1], padding=(kernel_size[1]-1)//2)
        self.conv_xt_3 = nn.Conv1d(128, 64, kernel_size[2], padding=(kernel_size[2]-1)//2)
        self.line1 = nn.Linear(64, 512)
        self.line2 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = (self.conv_xt_1(x) + self.conv_xt_2(x) + self.conv_xt_3(x)) / 3
        x = x.permute(0, 2, 1)
        x = self.line2(self.dropout(self.relu(self.line1(x))))
        return x

# === Peptide Feature Extraction ===
class Peptide_feature_extraction(nn.Module):
    def __init__(self, hidden_size=128, input_dim=128, hidden_dim=256, num_features_xd=23):
        super().__init__()
        self.hidden_size = hidden_size
        self.x_embedding = nn.Embedding(num_features_xd, input_dim)
        self.x_embedding2 = nn.Embedding(num_features_xd, input_dim)
        self.CNN = CNN(hidden_size)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.1, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=0.1, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_size, dropout=0.1, concat=False)
        self.line_emb = nn.Linear(1280, 128)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, data, device):
        x, edge_index = data.x, data.edge_index
        emb = data.emb
        node_len = data.peptide_len

        x = x.squeeze(-1).long()
        x_r = self.x_embedding(x)
        x_g = self.x_embedding2(x)

        x = self.relu(self.conv1(x_g, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))

        emb_graph = global_mean_pool(x, data.batch)
        emb = F.relu(self.line_emb(emb))

        flag = 0
        out_graph, out_seq, out_r, mask = [], [], [], []

        for i in node_len:
            count_i = i
            mask.append([1]*count_i + [0]*(50-count_i))  # Changed from 30 to 50
            x1 = torch.cat([x[flag:flag+count_i], torch.zeros(50-count_i, self.hidden_size, device=device)], dim=0)  # Changed from 30 to 50
            emb1 = torch.cat([emb[flag:flag+count_i], torch.zeros(50-count_i, 128, device=device)], dim=0)  # Changed from 30 to 50
            x_r1 = torch.cat([x_r[flag:flag+count_i], torch.zeros(50-count_i, 128, device=device)], dim=0)
            out_graph.append(x1)
            out_seq.append(emb1)
            out_r.append(x_r1)
            flag += count_i

        out_graph = torch.stack(out_graph).to(device)
        out_seq = torch.stack(out_seq).to(device)
        out_r = torch.stack(out_r).to(device)
        mask_tensor = torch.tensor(mask, dtype=torch.float, device=device)

        out_r = (out_r + out_seq) / 2
        out_seq_cnn = self.CNN(out_r)
        emb_seq = (out_seq_cnn * mask_tensor.unsqueeze(-1)).mean(1)

        return out_seq_cnn, out_graph, mask_tensor, mask_tensor, emb_seq, emb_graph

# === MHC Feature Extraction ===
class MHC_feature_extraction(nn.Module):
    def __init__(self, hidden_size=128, input_dim=128, hidden_dim=256, num_features_xd=23):
        super().__init__()
        self.hidden_size = hidden_size
        self.x_embedding = nn.Embedding(num_features_xd, input_dim)
        self.x_embedding2 = nn.Embedding(num_features_xd, input_dim)
        self.CNN = CNN(hidden_size)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.1, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=0.1, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_size, dropout=0.1, concat=False)
        self.line_emb = nn.Linear(1280, 128)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, data, device):
        x, edge_index = data.x, data.edge_index
        emb = data.emb
        node_len = data.mhc_len

        x = x.squeeze(-1).long()
        x_r = self.x_embedding(x)
        x_g = self.x_embedding2(x)

        x = self.relu(self.conv1(x_g, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))

        emb_graph = global_mean_pool(x, data.batch)
        emb = F.relu(self.line_emb(emb))

        flag = 0
        out_graph, out_seq, out_r, mask = [], [], [], []

        for i in node_len:
            count_i = i
            mask.append([1]*count_i + [0]*(50-count_i))
            x1 = torch.cat([x[flag:flag+count_i], torch.zeros(50-count_i, self.hidden_size, device=device)], dim=0)
            emb1 = torch.cat([emb[flag:flag+count_i], torch.zeros(50-count_i, 128, device=device)], dim=0)
            x_r1 = torch.cat([x_r[flag:flag+count_i], torch.zeros(50-count_i, 128, device=device)], dim=0)
            out_graph.append(x1)
            out_seq.append(emb1)
            out_r.append(x_r1)
            flag += count_i

        out_graph = torch.stack(out_graph).to(device)
        out_seq = torch.stack(out_seq).to(device)
        out_r = torch.stack(out_r).to(device)
        mask_tensor = torch.tensor(mask, dtype=torch.float, device=device)

        out_r = (out_r + out_seq) / 2
        out_seq_cnn = self.CNN(out_r)
        emb_seq = (out_seq_cnn * mask_tensor.unsqueeze(-1)).mean(1)

        return out_seq_cnn, out_graph, mask_tensor, mask_tensor, emb_seq, emb_graph