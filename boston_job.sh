#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J test_23
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
python3 main.py --run_folder boston_new --job_name test_subj_23 --file_path /work3/theb/boston_scalp.hdf5 --bckg_stride 2 --seiz_stride 0.1 --bckg_rate_val 1 --bckg_rate_train 1 --lr 1e-5 --prefetch_data_from_seg True --epochs 150 --weight_decay 1e-2 --train_val_test False --glob_avg_pool False --padding True --standardise False --anno_based_seg True --dropoutprob 0.6 --cnn_dropoutprob 0 --test_subj [22] --optimizer RMSprop
