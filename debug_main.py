from main import main
import argparse
parser = argparse.ArgumentParser()

# datagen
parser.add_argument('--file_path', type = str)
parser.add_argument('--window_length', type=float, default = 2)
parser.add_argument('--bckg_stride', type=eval, default=None)
parser.add_argument('--num_workers', type=eval, default=0)
parser.add_argument('--seiz_stride', type=eval, default=None)
parser.add_argument('--bckg_rate', default=None) # None or value
parser.add_argument('--anno_based_seg', type=bool, default=False)
parser.add_argument('--prefetch_data_dir', type=eval, default=False)
parser.add_argument('--prefetch_data_from_seg', type=eval, default=False)

# model
parser.add_argument('--model_type', type=str, default='BaselineCNN')
parser.add_argument('--dropoutprob', type=float, default=0.2)
parser.add_argument('--padding', type=bool, default=False)       

# Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=3e-4)

args = parser.parse_args(['--file_path','data/hdf5/temple_seiz_sub.hdf5', '--window_length', '2', '--bckg_stride', '1', '--seiz_stride', '1', '--bckg_rate', 'None', '--anno_based_seg', 'False',
                          '--model_type', 'BaselineCNN'])
main(args)