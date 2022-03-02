#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J kfold_final_analysis_fnsz_model
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
python3 main_analysis.py --run_folder kfold_temple_tcsz --job_name fnsz_eval --file_path /work3/theb/temple_seiz_full.hdf5 --window_length 2 --glob_avg_pool False --padding True --dropoutprob 0.4709 --cnn_dropoutprob 0.3246 --seiz_classes 'fnsz' --split 3 --n_splits 5 --onlytrainseiz None --orig_split True --model_path '/zhome/89/a/117273/Desktop/semi_active_seizure/models/final_models/2022-03-01 13:59:16.444104finalsplit_test_fnsz_valsplit_both/final_model_105_sensspec.pt'
