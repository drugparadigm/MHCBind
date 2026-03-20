import torch
import esm
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

# Set device to GPU if available
if torch.cuda.device_count() > 7:
    device = torch.device('cuda:1')
    print("Using device:", device)
    print("GPU available:", torch.cuda.get_device_name(7))
else:
    device = torch.device('cpu')
    print("cuda:7 not available — falling back to CPU")


def generate_embeddings(unique_sequences, folder_name, sequence_type, batch_size=512):
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    skipped_rows = 0
    total_ids = len(unique_sequences)

    # Batch processing
    batch_data = [(seq_id, seq) for seq_id, seq in unique_sequences.items()]
    for i in tqdm(range(0, total_ids, batch_size), desc=f"Processing {sequence_type}"):
        batch = batch_data[i:i + batch_size]
        valid_data = []
        invalid_ids = []

        # Validate sequences
        for seq_id, seq in batch:
            if not seq or not all(c in alphabet.all_toks for c in seq):
                print(f"Skipping invalid sequence for {sequence_type} ID {seq_id}: {seq}")
                skipped_rows += 1
                invalid_ids.append(seq_id)
            else:
                valid_data.append((seq_id, seq))

        if not valid_data:
            continue

        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(valid_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            embeddings = results["representations"][33].cpu().numpy()

            for idx, seq_id in enumerate(batch_labels):
                if seq_id in invalid_ids:
                    continue
                save_path = os.path.join(folder_name, f"{seq_id}.npy")
                np.save(save_path, embeddings[idx])
        except Exception as e:
            print(f"Error generating embeddings for {sequence_type} batch starting at ID {batch_labels[0] if batch_labels else 'unknown'}: {e}")
            skipped_rows += len(valid_data)
            continue

    print(f"Total {sequence_type} IDs processed: {total_ids}")
    print(f"Skipped {sequence_type} IDs (failed embedding generation): {skipped_rows}")
    print(f"Successfully processed {sequence_type} IDs: {total_ids - skipped_rows}")

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("Final_HLA-B51_01.csv")

    # Extract unique peptides
    unique_peptides = {row['Peptide_ID']: row['Peptide'] for _, row in df.iterrows()}
    unique_mhcs = {row['MHC_ID']: row['MHC'] for _, row in df.iterrows()}

    print(f"Found {len(unique_peptides)} unique Peptide_IDs.")
    print("Generating embeddings for peptides...")
    generate_embeddings(unique_peptides, "embeddings/peptides_HLA-B51-01", "peptides")

    print(f"Found {len(unique_mhcs)} unique MHC_IDs.")
    print("Generating embeddings for MHCs...")
    generate_embeddings(unique_mhcs, "embeddings/mhcs_HLA-B51-01", "MHCs")