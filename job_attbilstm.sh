#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J rep_pink
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=200GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 12:00
### added outputs and errors to files
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Error_%J.err

echo "Runnin script..."

source $HOME/miniconda3/bin/activate
conda activate semi_active_seiz
nvidia-smi > ./gpu_stats.csv
nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory --format=csv -l 10 >> ./gpu_stats.csv &
python3 main.py --file_path /work3/theb/boston_scalp_new.hdf5 --bckg_stride 23 --seiz_stride 23 --num_workers 0 --bckg_rate_val 1 --bckg_rate_train 1 --lr 1e-5 --prefetch_data_from_seg True --epochs 1 --weight_decay 1e-2 --train_val_test True --use_weighted_loss False --padding True --standardise False --anno_based_seg False --dropoutprob 0.4 --window_length 23 --model_type AttBiLSTM
pkill nvidia-smi
