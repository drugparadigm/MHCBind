# MHCBind
MHCBind processes peptide and MHC sequences, extracts features, and trains a deep learning model using attention mechanisms to predict binding interactions.
Follow the steps below in order to run the complete pipeline:
-->Step 1: Generate Contact Maps
   python train_scripts/contact_maps.py
-->Step 2: Generate Embeddings
   python train_scripts/embeddings.py
-->Step 3: Process MHC Sequences
   python train_scripts/process_mhc.py
-->Step 4: Process Peptide Sequences
   python train_scripts/process_pep.py
-->Step 5: Feature Extraction 
   python train_scripts/feature_extraction.py
-->Step 6: Cross-Attention Module
   python train_scripts/cross_attention.py
-->Step 7: Train the Model
   python train_scripts/train.py

Run scripts sequentially to avoid missing intermediate outputs.
