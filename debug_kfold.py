from kfoldmain import main
import argparse
# job name
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
# job name
parser.add_argument('--job_name', type = str, default='nojobname')
parser.add_argument('--run_folder', type = str, default='notspec')
# datagen
parser.add_argument('--file_path', type = str)
parser.add_argument('--seed', type = int, default = 20)
parser.add_argument('--split', type = int, default = 3)
parser.add_argument('--val_split', type = eval, default = 2)
parser.add_argument('--n_splits', type = int, default = 5)
parser.add_argument('--choose_orig_val', type = eval, default = False)
parser.add_argument('--orig_split', type = eval, default = True)
# exclude subjects that have 1 or more seizures not included in analysis
parser.add_argument('--excl_seiz', type = eval, default = False) 
# exclude seizure types to include in training but not evaluation
parser.add_argument('--onlytrainseiz', default = None)
parser.add_argument('--seiz_classes', nargs = '+', default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
parser.add_argument('--eval_seiz_classes', nargs = '+', default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
parser.add_argument('--window_length', type=float, default = 2)
parser.add_argument('--bckg_stride', type=eval, default=None)
parser.add_argument('--seiz_stride', type=eval, default=None)
parser.add_argument('--bckg_rate_val', type=eval, default=20) # None or value
parser.add_argument('--bckg_rate_train', type=eval, default=1)
parser.add_argument('--anno_based_seg', type=eval, default=True)
parser.add_argument('--train_val_test', type=eval, default=False)
parser.add_argument('--batch_size', type=eval, default=512)
# protocol(s) to use for training
parser.add_argument('--protocol', type=str, default= 'all')

# model
parser.add_argument('--model_type', type=str, default='BaselineCNN')
parser.add_argument('--glob_avg_pool', type=eval, default=False)
parser.add_argument('--dropoutprob', type=float, default=0.4)
parser.add_argument('--cnn_dropoutprob', type=float, default=0.4)
parser.add_argument('--padding', type=eval, default=False)       
parser.add_argument('--save_best_model', type=eval, default=False)      
parser.add_argument('--model_path', type=str, default=None)

# Training parameters
parser.add_argument('--optimizer', type = str, default = 'RMSprop')
parser.add_argument('--scheduler', type = str, default = None)
parser.add_argument('--milestones', type = eval, default = [50, 130, 150])
parser.add_argument('--use_weighted_loss', type=eval, default=True)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type = float, default=1e-3)

args = parser.parse_args(['--file_path','/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5'
                          ])
main(args)

