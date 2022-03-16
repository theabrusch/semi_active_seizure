#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J test_visualization
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
python3 main_visualization.py --run_folder kfold_temple_tcsz --job_name test_visualization --file_path /work3/theb/temple_seiz_full.hdf5 --window_length 2 --glob_avg_pool False --padding True --dropoutprob 0.4709 --cnn_dropoutprob 0.3246 --seiz_classes 'fnsz' 'cpsz' 'gnsz' 'tcsz' 'tnsz' --split 3 --n_splits 5 --onlytrainseiz None --model_path '/zhome/89/a/117273/Desktop/semi_active_seizure/models/final_models/2022-03-05 13:42:41.618951finalsplit_test_fnsz_cpsz_gnsz_valsplit_f1/final_model_49.pt' --n_iterations 400
