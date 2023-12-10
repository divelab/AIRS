#!/bin/sh

GPU=3

autoencoder_path=<set this variable> # name of root folder containing the trained autoencoder model (folder's name starts with time stamp)
suffix='trained_models/checkpoint_bst_rmsd.pt' # name of the trained autoencoder model, do not change this
eval_epoch=<set this variable> # epoch of latent diffusion model to evaluate
save_folder="evaluation_result" # name of folder to save evaluation results
num_samples=<set this variable> # number of samples to generate
latent_data_name=<set this variable> # name of latent data
diffusion_train_data="${latent_data_name}_train" # name of latent training data, do not change this
diffusion_model_path=<set this variable> # name of root folder containing the trained diffusion model
diffusion_generate_data_path="${diffusion_model_path}_eval/sampled_molecules/${save_folder}" # do not change this



##################################################
### generate latent samples using diffusion model
##################################################
echo "=====Generate latent samples using trained diffusion model====="
cd ../diffusion/


CUDA_VISIBLE_DEVICES=${GPU} \
python gen_latent_protein.py \
--n_samples=2000 \
--model_path='./outputs/'$diffusion_model_path \
--save_to_pt=True \
--batch_size_gen=100 \
--eval_epoch=${eval_epoch} \
--suffix=$save_folder \
--change_scale=False \

##################################################
### generate distribution plot for latent data and decoded data
##################################################
cd ../
echo "=====Generate distribution plot====="

CUDA_VISIBLE_DEVICES=${GPU} \
python ./gen_distribution_plot.py \
--model_path=$autoencoder_path \
--suffix=$suffix \
--diffusion_train_data=$diffusion_train_data \
--diffusion_generate_data=$diffusion_generate_data_path \
--save_folder=$save_folder


##################################################
### decode samples
##################################################
echo "=====Decode samples====="

CUDA_VISIBLE_DEVICES=${GPU} \
python ./decode_samples_for_scTM.py \
--model_path=$autoencoder_path \
--suffix=$suffix \
--diffusion_train_data=$diffusion_train_data \
--diffusion_generate_data=$diffusion_generate_data_path \
--save_folder=$save_folder \
--num_samples=$num_samples \


##################################################
# use ProteinMPNN to generate sequence for generated structures
##################################################
echo "=====Predict sequence using ProteinMPNN====="

cd ./ProteinMPNN/examples/

gpu=( 5 5 6 6 )

root="../../"

if [ "$save_folder" == '' ]; then
   seq_path=$autoencoder_path'/scTM/generated/'
else
   seq_path=$autoencoder_path'/'$save_folder'/scTM/generated/'
fi

echo $seq_path

let sample_per_thread=$num_samples/4

for ((i=0; i<4; i++)); do
   (
   for ((j=0; j<$sample_per_thread; j++)); do
       let var=i\*$sample_per_thread+j
#        if [ var == 780 ]
#        then
#            break
#        fi
       path_to_PDB=$root$seq_path"pdbs/generated_"$var".pdb"

       output_dir=$root$seq_path"ProteinMPNN/"
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

echo '========Finished generating sequence with ProteinMPNN========='

wait

##################################################
## use OmegaFold to predict structure from sequence
##################################################
echo "=====OmegaFold predict structures====="

cd ../../
conda deactivate
conda activate omegafold

root="./"

rm -rf $root$seq_path'ProteinMPNN/omegafold_predictions/'

python ./omegafold_across_gpus.py \
$root$seq_path'ProteinMPNN/seqs/'*.fa \
-o $root$seq_path'ProteinMPNN/omegafold_predictions/' \
-g 5 5 6 6 7 7

cd ./script/
conda deactivate
conda activate protein

echo "=====Finished OmegaFold predict structures====="


##################################################
## Calculate scTM score
##################################################
echo "=====Calculating scTM score====="

root="../"
if [ "$save_folder" == '' ]; then
   path=$root$autoencoder_path'/scTM/'
else
   path=$root$autoencoder_path'/'$save_folder'/scTM/'
fi


python ../calculate_scTM.py \
--root=$path \
--num_samples=$num_samples

echo "=====Finished calculating scTM score====="