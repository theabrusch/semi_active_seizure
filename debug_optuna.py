from cross_val_main import main
import argparse
# job name
parser = argparse.ArgumentParser()
# job name
parser.add_argument('--job_name', type = str, default='nojobname')
parser.add_argument('--run_folder', type = str, default='notspec')
# datagen
parser.add_argument('--file_path', type = str)
parser.add_argument('--window_length', type=float, default = 2)
parser.add_argument('--stride', type=eval, default=[0.05, 0.1, 0.2, 0.5, 0.7, 1, 1.5, 2])
parser.add_argument('--bckg_rate_val', type=eval, default=20) # None or value
parser.add_argument('--bckg_rate_train', type=eval, default=1)
parser.add_argument('--batch_size', type=eval, default=512)
# protocol(s) to use for training
parser.add_argument('--protocol', type=str, default= 'all')

# model
parser.add_argument('--model_type', type=str, default='BaselineCNN')
parser.add_argument('--glob_avg_pool', type=eval, default=False)
parser.add_argument('--dropoutprob', type=float, default=0.4)
parser.add_argument('--padding', type=eval, default=False)       

# Training parameters
parser.add_argument('--optimizer', type = str, default = 'RMSprop')
parser.add_argument('--scheduler', type = str, default = None)
parser.add_argument('--milestones', type = eval, default = [50, 130, 150])
parser.add_argument('--use_weighted_loss', type=eval, default=True)
parser.add_argument('--epochs', type=int, default=1)

# optuna params
parser.add_argument('--n_trials', type=int, default=1)
parser.add_argument('--time_out', type=int, default=1000)



args = parser.parse_args(['--file_path','/Volumes/GoogleDrive/Mit drev/Matematisk modellering/Speciale/semi_active_seizure/data/hdf5/temple_seiz_small_3.hdf5'])
main(args)