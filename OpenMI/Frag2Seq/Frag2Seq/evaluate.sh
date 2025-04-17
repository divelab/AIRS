#!/bin/bash

save_dir=<path to the generated sequences>


python evaluate.py \
--save_dir ${save_dir} \
--ca_only \
--largest_frag \
--use_openbabel \
--train_data_foler "two_digit_smiles_reorder_cutoff_15_ligand_only_frag" \



### calculate vina score

python analysis/docking.py \
--pdbqt_dir ./sample_output/test_pdbqt/ \
--sdf_dir ./sample_output/${save_dir}/processed_mol_sdf/ \
--out_dir ./sample_output/vina_output/${save_dir} \
--write_csv \
--write_dict \
--dataset crossdocked


python save_docking_result.py \
--save_dir ${save_dir}