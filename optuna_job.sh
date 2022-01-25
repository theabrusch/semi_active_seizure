#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J test_optuna_full_run_valsplit1_100epochs
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
python3 cross_val_main.py --run_folder optuna_new --job_name test_optuna_full_run_split2_100epochs --file_path /work3/theb/temple_seiz_full.hdf5 --n_trials 50 --epochs 100 --time_out 79200 --stride [0.5,1,1.5,2] --bckg_rate [1,2] --load_existing True --seiz_strat True --seiz_classes 'fnsz' 'gnsz' 'cpsz' 'spsz' 'tcsz' 'seiz' 'absz' 'tnsz' 'mysz' --split 2 --val_split 1
