from main import main
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type = str)
parser.add_argument('--window_length', type=float, default = 2)
parser.add_argument('--num_workers', type=int, default = 0)
parser.add_argument('--bckg_stride', type=eval, default=None)
parser.add_argument('--seiz_stride', type=eval, default=None)
parser.add_argument('--bckg_rate_val', type=eval, default=20) # None or value
parser.add_argument('--bckg_rate_train', type=eval, default=1)
parser.add_argument('--anno_based_seg', type=eval, default=False)
parser.add_argument('--prefetch_data_dir', type=eval, default=False)
parser.add_argument('--use_train_seed', type=eval, default=True)
parser.add_argument('--prefetch_data_from_seg', type=eval, default=False)
parser.add_argument('--train_val_test', type=eval, default=False)
parser.add_argument('--val_subj', type = eval, default=None)
parser.add_argument('--sens', type = eval, default=0)
parser.add_argument('--standardise', type = eval, default=False)
# model
parser.add_argument('--model_type', type=str, default='BaselineCNN')
parser.add_argument('--dropoutprob', type=eval, default=0.4)
parser.add_argument('--lstm_units', type=eval, default=140)
parser.add_argument('--dense_units', type=eval, default=70)
parser.add_argument('--padding', type=eval, default=False)       

# Training parameters
parser.add_argument('--use_weighted_loss', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type = float, default=1e-3)

args = parser.parse_args(['--file_path','data/hdf5/boston_scalp_sub.hdf5', '--window_length', '2',
                          '--bckg_stride', '1', '--seiz_stride', '1',
                          '--anno_based_seg', 'False', '--model_type', 'AttBiLSTM',
                          '--train_val_test', 'False'])
main(args)