#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J kfold_split34_dropout
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
python3 kfoldmain.py --run_folder kfold_temple --job_name split_2_dropout --file_path /work3/theb/temple_seiz_full.hdf5 --window_length 2 --bckg_stride 1 --seiz_stride 1 --bckg_rate_val 1 --bckg_rate_train 1 --lr 1e-5 --epochs 150 --weight_decay 1e-2 --glob_avg_pool False --padding False --anno_based_seg True --dropoutprob 0.6 --cnn_dropoutprob 0 --optimizer RMSprop --milestones [50,120,150] --scheduler None --seiz_classes 'fnsz' 'gnsz' 'cpsz' 'spsz' 'tcsz' 'seiz' 'absz' 'tnsz' 'mysz' --eval_seiz_classes 'fnsz' 'gnsz' 'cpsz' 'spsz' 'tcsz' 'seiz' 'absz' 'tnsz' 'mysz' --split 2 --n_splits 5 --onlytrainseiz None --save_best_model True --val_split None

