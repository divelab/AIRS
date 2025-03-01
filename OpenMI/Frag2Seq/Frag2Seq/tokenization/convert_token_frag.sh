#!/bin/bash


python convert_token_frag.py \
--base_folder processed_crossdock_noH_ca_only_temp_smiles_reorder_cutoff_15 \
--save_folder ../seq/two_digit_smiles_reorder_cutoff_15_ligand_only_frag/ \
--process_lig_only