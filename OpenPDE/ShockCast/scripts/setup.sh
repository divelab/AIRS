conda create -n shockCast python==3.10 -y  
conda activate shockCast
conda install -c conda-forge ffmpeg -y
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# https://stackoverflow.com/a/56889729/10965084
python -m pip install -r ../requirements.txt
python -m pip install -e ../.
echo done
