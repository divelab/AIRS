# number of replicates of experiments
n_exp=1 # TODO

# config data
path=./data/

V1e4=ns_V1e-4_N10000_T30.mat
swe=2D_rdb_NA_NA.h5
cos=ns_V0.0001_N1200_T30_cos4.mat
swearena=ShallowWater2D

# data with T (number of steps into the future to predict)
declare -A datas=(
["$V1e4"]=20
["$swe"]=24
["$cos"]=10
["$swearena"]=9
)

# data with superres dataset
declare -A super_datas=(
["$V1e4"]=ns_data_V1e-4_N20_T50_R256test.mat
["$cos"]=ns_V0.0001_N1200_T30_cos4_super.mat
)

data_names=( # TODO before running: uncommented datasets will be used for training
#$V1e4 # Navier Stokes
#$cos # Navier Stokes with symmetric forcing
#$swe # Shallow water equations (PDE Bench)
$swearena # Shallow water equation (PDE Arena)
)

ntrain=1000
nvalid=100
ntest=100

# model config; model with GPU
declare -A models=( # TODO before running: uncommented models will be trained on the assigned GPU
#["FNO2d"]=0
#["FNO2d_aug"]=3
#["FNO2d_aug-rf"]=6
["GFNO2d_p4"]=0
#["GFNO2d_p4m"]=3
#["GFNO2d_p4_steer"]=6
#["GFNO2d_p4m_steer"]=3
#["Ghybrid2d_p4"]=6
#["Ghybrid2d_p4m"]=7
#["radialNO2d_p4"]=9
#["radialNO2d_p4m"]=4
#["Unet_Rot_M2d"]=7
#["Unet_Rot2d"]=7
#["FNO3d"]=0
#["FNO3d_aug"]=3
#["FNO3d_aug-rf"]=5
#["GFNO3d_p4"]=0
#["GFNO3d_p4m"]=0
#["radialNO3d_p4"]=7
#["radialNO3d_p4m"]=4
#["Unet_Rot_3D"]=3
)

declare -A widths=( # number of channels
["FNO2d"]=20
["FNO2d_aug"]=20
["FNO2d_aug-rf"]=20
["GFNO2d_p4"]=10
["GFNO2d_p4m"]=7
["GFNO2d_p4_steer"]=15
["GFNO2d_p4m_steer"]=11
["Ghybrid2d_p4"]=20
["Ghybrid2d_p4m"]=20
["radialNO2d_p4"]=40
["radialNO2d_p4m"]=50
["Unet_Rot_M2d"]=32
["FNO3d"]=20
["FNO3d_aug"]=20
["FNO3d_aug-rf"]=20
["GFNO3d_p4"]=11
["GFNO3d_p4m"]=7
["radialNO3d_p4"]=60
["radialNO3d_p4m"]=80
["Unet_Rot_3D"]=32
)

suffix="" # TODO

# loop over model types
for model in "${!models[@]}"; do

  # model config
  gpu="${models[$model]}"
  epochs=100 # markov/ oneshot & recurrent (100 / 500)
  strategy=teacher_forcing # TODO markov/ recurrent /teacher_forcing
  modes=12 # 3d/ 2d data (8 / 12)
  batch_size=20 # 3d/ 2d data (1 / 20)
  if [[ $model == *"3"* ]]; then # 3d/ 2d data
    epochs=500
    strategy=oneshot
    modes=8
    batch_size=10
  fi

  width="${widths[$model]}"

  (
  # loop over datasets
  for data in "${data_names[@]}"; do

    # data config
    T="${datas[$data]}"
    data_name=$data

    if [ "$data" = "$swe" ] && [ "$ntrain" = 1000 ]; then
      ntrain=800
    elif [ "$data" = "$swearena" ]; then
      modes=32
      if [[ $model == *"3"* ]]; then
        modes=22
      fi
      ntrain=5600
      nvalid=1120
      ntest=1120
      if [[ "$model" = "radialNO2d_p4m" ]]; then
        width=55
      elif [[ "$model" = "Unet_Rot_M2d" ]]; then
        width=44
      elif [[ "$model" = "Unet_Rot_3D" ]]; then
        width=90
      elif [[ "$model" = "GFNO3d_p4m" ]]; then
        width=8
      fi
    fi

    model_name="$model"
    if [[ "$model_name" =~ rf ]]; then
      model_name="${model_name::-3}"
    fi
    if [[ "$model_name" =~ Ghybrid ]]; then
      n_equiv=3
      model="$model_name$n_equiv"
    fi

    # perform replicates
    for rep in $(seq 1 $n_exp); do

      args=(
      --seed="$rep"
      --data_path="$path$data"
      --results_path="./results/$data_name/$model/" # TODO
      --strategy="$strategy"
      --T="$T"
      --ntrain="$ntrain"
      --nvalid="$nvalid"
      --ntest="$ntest"
      --model_type="$model_name"
      --modes="$modes"
      --width="$width"
      --batch_size="$batch_size"
      --epochs="$epochs"
      --suffix="seed$rep$suffix"
      --txt_suffix=$data_name\_$model\_seed$rep
      --learning_rate=1e-3
      --early_stopping=100
      )
      if [ "$data" = "$swearena" ]; then
        args+=( --time_pad )
      else
        args+=( --super )
        if [ "$data" = "$V1e4" ] || [ "$data" = "$cos" ]; then
          super_data="${super_datas[$data]}"
          args+=( --super_path="$path$super_data" )
        fi
      fi
      if [[ "$model" =~ rf ]]; then
        args+=( --reflection )
      fi
      if [[ "$model" =~ Ghybrid ]]; then
        args+=( --n_equiv="$n_equiv" )
        if [[ "$model" =~ p4m ]]; then
          args+=( --Gwidth=7 )
        else
          args+=( --Gwidth=10 )
        fi
      fi

#      if [ $gpu = 8 ]; then
#        ((gpu++))
#      fi

      echo "${args[@]}"
      printf "\n\n"
      printf "\n\nTraining $model$width with $strategy strategy on GPU $gpu with data $data, T=$T (trial $rep, ntrain $ntrain)\n\n"
#      CUDA_VISIBLE_DEVICES=$gpu python experiments.py "${args[@]}" & # --verbose & # TODO: uncomment to run
#      gpu=$(( 1 * rep + 7 ))
      ((gpu++))
      sleep 4s
    done

  done
   ) &
done