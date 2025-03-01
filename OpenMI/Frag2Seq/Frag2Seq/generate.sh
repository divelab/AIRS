#!/bin/bash

gpu=0

top_k=90
temp=0.7
epoch=2000


CUDA_VISIBLE_DEVICES=${gpu} \
python train_3D/generate.py \
--run_name frag2seq \
--batch_size 32 \
--num_props 0 \
--max_epochs 250 \
--root_path ./seq/two_digit_smiles_reorder_cutoff_15_ligand_only_frag/train_frag_seq \
--output_tokenizer_dir ./seq/storage/two_digit_smiles_reorder_cutoff_15_ligand_only_frag/train_frag_seq/tokenizer \
--max_len 512 \
--n_layer 12 \
--n_head 12 \
--epoch ${epoch} \
--top_k ${top_k} \
--save_path ./sample_output/two_digit_smiles_reorder_cutoff_15_ligand_only_cross_ml_512_frag_ep${epoch}_temp_${temp}_topk_${top_k}/raw_seq \
--test_lmdb_path ./seq/two_digit_smiles_reorder_cutoff_15_ligand_only_frag/protein_embedding_test.lmdb \
--model_root_path <path to checkpoint root folder> \
--mode 'cross' \
--ESM_protein \
--sample_repeats 100 \
--seed 56 \
--temp $temp