#!/bin/sh

gpu=2

CUDA_VISIBLE_DEVICES=${gpu} \
python train_3D/train.py \
--run_name frag2seq \
--batch_size 64 \
--num_props 0 \
--max_epochs 2000 \
--root_path ./seq/two_digit_smiles_reorder_cutoff_15_ligand_only_frag/train_frag_seq \
--output_tokenizer_dir ../seq/storage/two_digit_smiles_reorder_cutoff_15_ligand_only_frag/train_frag_seq/tokenizer \
--max_len 512 \
--n_layer 12 \
--n_head 12 \
--protein_embedding_path "./seq/two_digit_smiles_reorder_cutoff_15_ligand_only_frag/protein_embedding_train.lmdb" \
--protein_embedding_val_path "./seq/two_digit_smiles_reorder_cutoff_15_ligand_only_frag/protein_embedding_val.lmdb" \
--mode 'cross' \
--ESM_protein \

