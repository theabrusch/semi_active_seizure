#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J test_kfold
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=200GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 01:00
### added outputs and errors to files
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Error_%J.err

echo "Runnin script..."

source $HOME/miniconda3/bin/activate
conda activate semi_active_seiz
python3 kfoldmain.py --run_folder kfold_temple --job_name test_kfold --file_path /work3/theb/temple_seiz_full.hdf5 --window_length 2 --bckg_stride 1 --seiz_stride 1 --bckg_rate_val 1 --bckg_rate_train 1 --lr 1e-4 --epochs 5 --weight_decay 1e-2 --glob_avg_pool False --padding True --anno_based_seg True --dropoutprob 0.6 --optimizer RMSprop --milestones [30,80,100] --scheduler MultistepLR --seiz_classes 'fnsz' 'gnsz' 'cpsz' 'spsz' 'tcsz' 'seiz' 'absz' 'tnsz' 'mysz' --eval_seiz_classes 'fnsz' 'gnsz' 'cpsz' 'spsz' 'tcsz' 'seiz' 'absz' 'tnsz' 'mysz' --split 0 --n_splits 5

