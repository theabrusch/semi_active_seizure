#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J temple_test_fulldata
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=200GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 24:00
### added outputs and errors to files
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Error_%J.err

echo "Runnin script..."

source $HOME/miniconda3/bin/activate
conda activate semi_active_seiz
python3 main.py --job_name temple_test_fulldata --file_path /work3/theb/temple_seiz_full.hdf5 --window_length 2 --bckg_stride 2 --seiz_stride 1 --bckg_rate_val 1 --bckg_rate_train 1 --lr 1e-4 --prefetch_data_from_seg True --epochs 200 --weight_decay 1e-2 --train_val_test True --glob_avg_pool True --padding False --standardise False --anno_based_seg True --dropoutprob 0.3 --optimizer RMSprop --use_weighted_loss True --scheduler MultistepLR --milestones [50,150,200]


