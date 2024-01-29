#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_3.out

# source activate mlfold

# path_to_PDB="./train_pdbs/generated_20.pdb"
gpu=( 1 1 2 3 )

root="/mnt/data/shared/congfu/protein-generation/E3_Protein/code/"
# path="20221129_1101protein_autoencoder_AFPDBdata_128_complete_new_designV11_klh1e-4_optimized_distloss0.5_torsion0.5_attn_tanh_two_pooling_lr_1e-3/scTM/generated/"
path="20221209_0937no_h_update_in_pooling_protein_autoencoder_AFPDBdata_128_complete_new_designV19corrected_klh1e-4_optimized_distloss0.5_torsion0.5_noaa_attn_tanh_two_pooling_lr_1e-3/scTM/generated/"

for ((i=0; i<4; i++)); do
    (
    for ((j=0; j<25; j++)); do
        let var=i\*25+j
        # if [ var == 780 ]
        # then
        #     break
        # fi
        path_to_PDB=$root$path"pdbs/generated_"$var".pdb"

        output_dir=$root$path"ProteinMPNN/"
        if [ ! -d $output_dir ]
        then
            mkdir -p $output_dir
        fi

        chains_to_design="A"

        CUDA_VISIBLE_DEVICES=${gpu[i]} \
        python ../protein_mpnn_run.py \
                --pdb_path $path_to_PDB \
                --pdb_path_chains "$chains_to_design" \
                --out_folder $output_dir \
                --num_seq_per_target 8 \
                --sampling_temp "0.1" \
                --seed 37 \
                --batch_size 1 \
                --ca_only \
                --protein_num $var
    done
    ) &
done


