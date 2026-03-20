import torch
from torch.utils.data import Dataset, SubsetRandomSampler
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from feature_extraction import MHC_feature_extraction, Peptide_feature_extraction, TestMHCDataset, TestPeptideDataset
from cross_attention import CrossAttention
from torch_geometric.loader import DataLoader as GeometricDataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, f1_score
import copy

# Set device to GPU 2
device = torch.device('cuda:1')
print(f"Training on device: {device}")

hidden_dim = 128

# Training hyperparameters
EPOCH = 150
BATCH_SIZE = 1024
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 1e-5
SEED = 1

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)

set_seed(SEED)

# Custom dataset to pair MHC and peptide data
class MHCPeptideDataset(Dataset):
    def __init__(self, mhc_dataset, peptide_dataset):
        self.mhc_dataset = mhc_dataset
        self.peptide_dataset = peptide_dataset
        self.length = len(self.mhc_dataset)
        print(f"Paired dataset length: {self.length}")

    def __getitem__(self, index):
        return self.mhc_dataset[index], self.peptide_dataset[index]

    def __len__(self):
        return self.length

# MHC-Peptide Interaction Model
class MHC_Peptide_Model(nn.Module):
    def __init__(self, hidden_dim=128):
        super(MHC_Peptide_Model, self).__init__()
        self.mhc_feature_model = MHC_feature_extraction(hidden_size=hidden_dim)
        self.peptide_feature_model = Peptide_feature_extraction(hidden_size=hidden_dim)
        self.cross_attention = CrossAttention(hidden_dim=hidden_dim)
        self.line1 = nn.Linear(hidden_dim * 2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        self.mhc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.pep1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.mhc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.pep2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, mhc_batch, peptide_batch):
        mhc_out_seq, mhc_out_graph, mhc_mask_seq, mhc_mask_graph, mhc_emb_seq, mhc_emb_graph = self.mhc_feature_model(mhc_batch, device)
        pep_out_seq, pep_out_graph, pep_mask_seq, pep_mask_graph, pep_emb_seq, pep_emb_graph = self.peptide_feature_model(peptide_batch, device)
        context_layer, attention_score = self.cross_attention(
            [mhc_out_seq, mhc_out_graph, pep_out_seq, pep_out_graph],
            [mhc_mask_seq.to(device), mhc_mask_graph.to(device), pep_mask_seq.to(device), pep_mask_graph.to(device)],
            device
        )
        out_mhc = context_layer[-1][0]
        out_pep = context_layer[-1][1]
        mhc_cross_seq = (out_mhc[:, 0:50] * (mhc_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)
        mhc_cross_graph = (out_mhc[:, 50:100] * (mhc_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)
        mhc_cross = (mhc_cross_seq + mhc_cross_graph) / 2 + (mhc_emb_seq + mhc_emb_graph) / 2
        mhc_cross = self.mhc2(self.dropout(self.relu(self.mhc1(mhc_cross))))
        pep_cross_seq = (out_pep[:, 0:50] * (pep_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)
        pep_cross_graph = (out_pep[:, 50:100] * (pep_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)
        pep_cross = (pep_cross_seq + pep_cross_graph) / 2 + (pep_emb_seq + pep_emb_graph) / 2
        pep_cross = self.pep2(self.dropout(self.relu(self.pep1(pep_cross))))
        out = torch.cat((mhc_cross, pep_cross), dim=1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        return out

# Create directories for saving models and checkpoints
mhc_name = "HLA-B51-01"
os.makedirs(f'save/{mhc_name}', exist_ok=True)
os.makedirs(f'checkpoint/{mhc_name}', exist_ok=True)

# Load datasets using the updated dataset classes
train_mhc_dataset = TestMHCDataset(root=f"dataset/test_mhc_{mhc_name}")
train_peptide_dataset = TestPeptideDataset(root=f"dataset/test_peptide_{mhc_name}")

# Create paired dataset
dataset = MHCPeptideDataset(train_mhc_dataset, train_peptide_dataset)

# Split dataset into 80-10-10
dataset_size = len(dataset)
indices = list(range(dataset_size))
split_train = int(np.floor(0.8 * dataset_size))
split_val = int(np.floor(0.1 * dataset_size)) + split_train

np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[:split_train], indices[split_train:split_val], indices[split_val:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create data loaders
train_loader = GeometricDataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, drop_last=False, sampler=train_sampler)
val_loader = GeometricDataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, drop_last=False, sampler=val_sampler)
test_loader = GeometricDataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, drop_last=False, sampler=test_sampler)

# Save test set to CSV before training
original_df = pd.read_csv(f'Final_HLA-B51_01.csv')  # Assuming this is the original dataset
test_df = original_df.iloc[test_indices].reset_index(drop=True)
test_df.to_csv(f'test_HLA-B51-01.csv', index=False)
print(f"Test set saved to test_HLA-B51-01.csv with {len(test_df)} samples before training.")

# Initialize model and optimizer
model = MHC_Peptide_Model(hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Checkpointing
checkpoint_path = f'checkpoint/{mhc_name}/mhc_peptide_checkpoint_seed{SEED}_{mhc_name}.pth'
start_epoch = 0
best_pcc = -1
best_model_state = None

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_pcc = checkpoint['best_pcc']
    print(f"Resuming training from {mhc_name}, epoch {start_epoch}, best PCC: {best_pcc:.4f}")

# Initialize loss function and metrics
loss_fct = nn.MSELoss()
max_p = -1
max_s = -1
max_rmse = 0
max_mae = 0
max_mse = 0
max_auc = 0
max_f1 = 0

# Create a metrics file for this MHC
metrics_file = f'metrics_{mhc_name}.txt'
if not os.path.exists(metrics_file):
    with open(metrics_file, 'w') as f:
        f.write(f"{'epo':<6}{'pcc':<20}{'scc':<20}{'rmse':<20}{'mae':<20}{'mse':<20}{'auc':<20}{'f1':<20}\n")

# Training loop
for epo in range(start_epoch, EPOCH):
    # Train
    train_loss = 0
    model.train()
    for step, (mhc_batch, peptide_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        pre = model(mhc_batch.to(device), peptide_batch.to(device))
        loss = loss_fct(pre.squeeze(dim=1), mhc_batch.y.float().to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation (every 10 epochs)
    if (epo + 1) % 10 == 0:
        with torch.set_grad_enabled(False):
            val_loss = 0
            model.eval()
            y_label = []
            y_pred = []
            y_pred_binary = []
            for step, (mhc_batch_val, peptide_batch_val) in enumerate(val_loader):
                label = mhc_batch_val.y.float().to(device)
                score = model(mhc_batch_val.to(device), peptide_batch_val.to(device))
                n = torch.squeeze(score, 1)
                logits = n.detach().cpu().numpy()
                label_ids = label.cpu().numpy()
                loss_v = loss_fct(n, label)
                y_label.extend(label_ids.flatten().tolist())
                y_pred.extend(logits.flatten().tolist())
                y_pred_binary.extend((logits > 0.5).astype(int).flatten().tolist())
                val_loss += loss_v.item()

            # Compute validation metrics
            p = pearsonr(y_label, y_pred)[0]
            s = spearmanr(y_label, y_pred)[0]
            rmse = np.sqrt(mean_squared_error(y_label, y_pred))
            mae = mean_absolute_error(y_label, y_pred)
            mse = mean_squared_error(y_label, y_pred)
            y_label_binary = (np.array(y_label) > 0.5).astype(int)
            auc = roc_auc_score(y_label_binary, y_pred_binary) if len(set(y_label_binary)) > 1 else 0.0
            f1 = f1_score(y_label_binary, y_pred_binary) if len(set(y_label_binary)) > 1 else 0.0

            # Save metrics and model if PCC improves
            print(f'epo: {epo}, pcc: {p:.4f}, scc: {s:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}, mse: {mse:.4f}, auc: {auc:.4f}, f1: {f1:.4f}')
            if p > max_p:
                max_p = p
                max_s = s
                max_rmse = rmse
                max_mae = mae
                max_mse = mse
                max_auc = auc
                max_f1 = f1
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, f'save/{mhc_name}/mhc_peptide_model_seed{SEED}_{mhc_name}.pth')
                with open(metrics_file, 'a') as f:
                    f.write(f"{epo:<6}{p:<20.6f}{s:<20.6f}{rmse:<20.6f}{mae:<20.6f}{mse:<20.6f}{auc:<20.6f}{f1:<20.6f}\n")

    # Save checkpoint after every epoch
    checkpoint = {
        'epoch': epo,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_pcc': max_p
    }
    torch.save(checkpoint, checkpoint_path)

# Test
with torch.set_grad_enabled(False):
    test_loss = 0
    model.eval()
    y_label = []
    y_pred = []
    y_pred_binary = []
    for step, (mhc_batch_test, peptide_batch_test) in enumerate(test_loader):
        label = mhc_batch_test.y.float().to(device)
        score = model(mhc_batch_test.to(device), peptide_batch_test.to(device))
        n = torch.squeeze(score, 1)
        logits = n.detach().cpu().numpy()
        label_ids = label.cpu().numpy()
        loss_t = loss_fct(n, label)
        y_label.extend(label_ids.flatten().tolist())
        y_pred.extend(logits.flatten().tolist())
        y_pred_binary.extend((logits > 0.5).astype(int).flatten().tolist())
        test_loss += loss_t.item()

    # Compute test metrics
    p = pearsonr(y_label, y_pred)[0]
    s = spearmanr(y_label, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_label, y_pred))
    mae = mean_absolute_error(y_label, y_pred)
    mse = mean_squared_error(y_label, y_pred)
    y_label_binary = (np.array(y_label) > 0.5).astype(int)
    auc = roc_auc_score(y_label_binary, y_pred_binary) if len(set(y_label_binary)) > 1 else 0.0
    f1 = f1_score(y_label_binary, y_pred_binary) if len(set(y_label_binary)) > 1 else 0.0

    print(f'Test Metrics: PCC: {p:.4f}, SCC: {s:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}')

# Save test metrics to file
with open(f'test_metrics_{mhc_name}.txt', 'w') as f:
    f.write(f'Test Metrics for {mhc_name}:\n')
    f.write(f'PCC: {p:.4f}\n')
    f.write(f'SCC: {s:.4f}\n')
    f.write(f'RMSE: {rmse:.4f}\n')
    f.write(f'MAE: {mae:.4f}\n')
    f.write(f'MSE: {mse:.4f}\n')
    f.write(f'AUC: {auc:.4f}\n')
    f.write(f'F1: {f1:.4f}\n')

# Clear checkpoint after training completes
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
print(f"Training completed for {mhc_name}, checkpoint cleared.")