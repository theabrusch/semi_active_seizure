#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J no_background_rate
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
python3 main.py --file_path /work3/theb/boston_scalp_new.hdf5 --bckg_stride 1 --seiz_stride 1 --num_workers 0 --bckg_rate None  --lr 1e-5 --prefetch_data_from_seg True --epochs 90 --weight_decay 1e-2
