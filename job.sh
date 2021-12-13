#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J test_subj_7_allbckg
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
python3 main.py --file_path /work3/theb/boston_scalp.hdf5 --bckg_stride 2 --seiz_stride 1 --bckg_rate_val None --bckg_rate_train None --lr 1e-5 --prefetch_data_from_seg True --epochs 100 --weight_decay 1e-2 --train_val_test False --use_weighted_loss True --padding True --standardise False --anno_based_seg True --dropoutprob 0.6 --val_subj [6]
pkill nvidia-smi
