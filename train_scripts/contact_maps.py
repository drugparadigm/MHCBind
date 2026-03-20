import torch
import esm
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

if torch.cuda.device_count() > 7:
    device = torch.device('cuda:1')
    print("Using device:", device)
    print("GPU available:", torch.cuda.get_device_name(7))
else:
    device = torch.device('cpu')
    print("cuda:7 not available — falling back to CPU")
    
def generate_contact_maps(unique_sequences, folder_name, sequence_type, threshold, batch_size=512):
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    # Create output folder if needed
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    skipped_rows = 0
    total_ids = len(unique_sequences)

    # Convert dictionary to list of tuples for batching
    all_data = [(seq_id, seq) for seq_id, seq in unique_sequences.items()]

    # Batch processing
    for i in tqdm(range(0, total_ids, batch_size), desc=f"Processing {sequence_type}"):
        batch_data = all_data[i:i+batch_size]

        # Filter invalid sequences
        filtered_data = []
        for seq_id, seq in batch_data:
            if seq and all(c in alphabet.all_toks for c in seq):
                filtered_data.append((seq_id, seq))
            else:
                print(f"Skipping invalid sequence for {sequence_type} ID {seq_id}: {seq}")
                skipped_rows += 1

        if not filtered_data:
            continue

        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(filtered_data)
            if torch.cuda.is_available():
                batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            contact_maps = results["contacts"].cpu().numpy()

            for j, (seq_id, _) in enumerate(filtered_data):
                contact_map = contact_maps[j]
                np.fill_diagonal(contact_map, 0)
                contact_map[contact_map < threshold] = 0
                contact_map[contact_map >= threshold] = 1

                save_path = os.path.join(folder_name, f"{seq_id}.npy")
                np.save(save_path, contact_map)

        except Exception as e:
            print(f"Error in batch {i}-{i+batch_size} for {sequence_type}: {e}")
            skipped_rows += len(filtered_data)

    print(f"\nTotal {sequence_type} IDs processed: {total_ids}")
    print(f"Skipped {sequence_type} IDs (failed/invalid): {skipped_rows}")
    print(f"Successfully processed {sequence_type} IDs: {total_ids - skipped_rows}")

if __name__ == "__main__":
    # Load updated dataset
    df = pd.read_csv("Final_HLA-B51_01.csv")

    # Extract unique peptides
    unique_peptides = {}
    for idx, row in df.iterrows():
        seq = row['Peptide']
        peptide_id = row['Peptide_ID']
        unique_peptides[peptide_id] = seq

    # Extract unique MHCs
    unique_mhcs = {}
    for idx, row in df.iterrows():
        seq = row['MHC']
        mhc_id = row['MHC_ID']
        unique_mhcs[mhc_id] = seq

    # Generate peptide contact maps
    print(f"Found {len(unique_peptides)} unique Peptide_IDs.")
    print("Generating contact maps for peptides with threshold 0.05...")
    generate_contact_maps(unique_peptides, "contact_maps/peptides_HLA-B51-01", "peptides", threshold=0.05)

    # Generate MHC contact maps
    print(f"Found {len(unique_mhcs)} unique MHC_IDs.")
    print("Generating contact maps for MHCs with threshold 0.075...")
    generate_contact_maps(unique_mhcs, "contact_maps/mhcs_HLA-B51-01", "MHCs", threshold=0.075)