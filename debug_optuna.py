from cross_val_main import main
import argparse
# job name
parser = argparse.ArgumentParser()
# job name
parser.add_argument('--job_name', type = str, default='nojobname')
parser.add_argument('--run_folder', type = str, default='notspec')
# split
parser.add_argument('--split', type = int, default = 0)
parser.add_argument('--val_split', type = int, default = 0)
parser.add_argument('--n_splits', type = int, default = 5)
parser.add_argument('--n_val_splits', type = int, default = 5)
# datagen
parser.add_argument('--seed', type = int, default = 20)
parser.add_argument('--seiz_classes', nargs='+', default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
parser.add_argument('--file_path', type = str)
parser.add_argument('--seiz_strat', type = eval, default = False)
parser.add_argument('--window_length', type=float, default = 2)
parser.add_argument('--stride', type=eval, default=[0.5, 1, 1.5, 2])
parser.add_argument('--bckg_rate', type=eval, default=[1, 2, 5]) # None or value
parser.add_argument('--batch_size', type=eval, default=512)
# protocol(s) to use for training
parser.add_argument('--protocol', type=str, default= 'all')

# model
parser.add_argument('--model_type', type=str, default='BaselineCNN')      

# Training parameters
parser.add_argument('--optimizer', type = str, default = 'RMSprop')
parser.add_argument('--scheduler', type = str, default = None)
parser.add_argument('--milestones', type = eval, default = [50, 130, 150])
parser.add_argument('--use_weighted_loss', type=eval, default=True)
parser.add_argument('--epochs', type=int, default=150)

# optuna params
parser.add_argument('--load_existing', type=eval, default=False)
parser.add_argument('--n_trials', type=int, default=1)
parser.add_argument('--time_out', type=int, default=600)

args = parser.parse_args(['--file_path','/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5'])
main(args)