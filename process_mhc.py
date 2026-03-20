import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
from tqdm import tqdm

class TestMHCDataset(InMemoryDataset):
    def __init__(self,
                 root="dataset/test_mhc_HLA-B51-01",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 batch_size=512):

        self.root = root
        self.batch_size = batch_size

        # Load updated dataset
        csv_file_path = 'Final_HLA-B51_01.csv'
        self.df = pd.read_csv(csv_file_path)

        # Updated folder paths for embeddings and contact maps
        self.contact_map_folder_path = 'contact_maps/mhcs_HLA-B51-01'
        self.emb_folder_path = 'embeddings/mhcs_HLA-B51-01'
        self.peptide_contact_map_folder_path = 'contact_maps/peptides_HLA-B51-01'
        self.peptide_emb_folder_path = 'embeddings/peptides_HLA-B51-01'

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return "data_test_mhc_HLA-B51-01.pt"

    def process(self):
        data_list = []
        skipped_rows = 0
        total_rows = len(self.df)

        for start_idx in tqdm(range(0, total_rows, self.batch_size), desc="Processing test MHC dataset"):
            end_idx = min(start_idx + self.batch_size, total_rows)
            batch_df = self.df[start_idx:end_idx]

            batch_data = []
            for index, row in batch_df.iterrows():
                mhc_id = row['MHC_ID']
                peptide_id = row['Peptide_ID']
                sequence = row['MHC'][:511] if len(row['MHC']) > 512 else row['MHC']

                # Paths
                mhc_contact_file_path = os.path.join(self.contact_map_folder_path, f"{mhc_id}.npy")
                peptide_contact_file_path = os.path.join(self.peptide_contact_map_folder_path, f"{peptide_id}.npy")
                mhc_emb_file_path = os.path.join(self.emb_folder_path, f"{mhc_id}.npy")
                peptide_emb_file_path = os.path.join(self.peptide_emb_folder_path, f"{peptide_id}.npy")

                if (os.path.exists(mhc_contact_file_path) and
                    os.path.exists(peptide_contact_file_path) and
                    os.path.exists(mhc_emb_file_path) and
                    os.path.exists(peptide_emb_file_path)):

                    matrix = np.load(mhc_contact_file_path)
                    matrix[matrix < 0.075] = 0
                    matrix[matrix >= 0.075] = 1
                    np.fill_diagonal(matrix, 0)

                    indices = [char_to_index(c) for c in sequence]
                    edges = np.argwhere(matrix == 1)

                    x = torch.tensor(indices, dtype=torch.long)
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    y = torch.tensor([row['Y']], dtype=torch.float32)
                    m_id = row['MHC_ID']
                    p_id = row['Peptide_ID']

                    mhc_emb = np.load(mhc_emb_file_path)
                    if mhc_emb.shape[-1] != 1280:
                        print(f"Warning: MHC embedding for {mhc_id} has shape {mhc_emb.shape}, expected [1280].")
                        mhc_emb = mhc_emb.flatten()[:1280]
                        if len(mhc_emb) < 1280:
                            mhc_emb = np.pad(mhc_emb, (0, 1280 - len(mhc_emb)), mode='constant')
                    mhc_emb = torch.tensor(mhc_emb, dtype=torch.float32)

                    mhc_len = len(indices)
                    data = Data(x=x, edge_index=edge_index, y=y, m_id=m_id, p_id=p_id, emb=mhc_emb, mhc_len=mhc_len)
                    batch_data.append(data)
                else:
                    skipped_rows += 1
                    if not os.path.exists(mhc_contact_file_path):
                        print(f"MHC contact map not found: {mhc_id}")
                    if not os.path.exists(peptide_contact_file_path):
                        print(f"Peptide contact map not found: {peptide_id}")
                    if not os.path.exists(mhc_emb_file_path):
                        print(f"MHC embedding not found: {mhc_id}")
                    if not os.path.exists(peptide_emb_file_path):
                        print(f"Peptide embedding not found: {peptide_id}")

            data_list.extend(batch_data)

        print(f"Total rows processed: {total_rows}")
        print(f"Skipped rows (missing files): {skipped_rows}")
        print(f"Successfully processed rows: {len(data_list)}")
        print("Saving processed test MHC dataset for HLA-B51-01...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Map amino acid to index
def char_to_index(char):
    mapping = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20, 'B': 21, 'Z': 22
    }
    return mapping.get(char, 20)  # Use 'X' index by default

if __name__ == "__main__":
    dataset = TestMHCDataset(batch_size=256)
    print(f"Test dataset loaded with {len(dataset)} samples for HLA-B51-01.")