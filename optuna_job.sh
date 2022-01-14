#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J test_optuna_full_run_no_bckg
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
python3 cross_val_main.py --run_folder optuna --job_name test_optuna_full_run_no_bckg --file_path /work3/theb/temple_seiz.hdf5 --n_trials 50 --epochs 50 --time_out 72000 --stride [0.5,1,2] --bckg_rate [1,2]
