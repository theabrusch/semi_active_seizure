#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J transfer_recs_bck2
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
python3 main_transfer.py --run_folder transfer_recs/transfer_val3  --job_name transfer_recs_bckg2 --file_path /work3/theb/temple_seiz_full.hdf5 --model_path /zhome/89/a/117273/Desktop/semi_active_seizure/models/checkpoints/excl_val_3/final_model.pt --window_length 2 --bckg_stride 2 --seiz_stride 2 --bckg_rate_train 2 --lr 1e-5 --epochs 30 --weight_decay 1e-2 --glob_avg_pool False --padding False --anno_based_seg True --dropoutprob 0.6 --optimizer RMSprop --use_weighted_loss True --milestones [70,150] --scheduler None --seiz_classes 'fnsz' 'gnsz' 'cpsz' 'spsz' 'tcsz' 'seiz' 'absz' 'tnsz' 'mysz' --batch_size 10 --val_split 3 --split 2 --lambda_cons 0 --use_subjects 3

