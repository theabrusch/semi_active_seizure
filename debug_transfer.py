from main_visualization import main
import argparse
# job name
parser = argparse.ArgumentParser()
# job name
parser.add_argument('--job_name', type = str, default='nojobname')
parser.add_argument('--run_folder', type = str, default='notspec')
# datagen
parser.add_argument('--file_path', type = str)
parser.add_argument('--seed', type = int, default = 20)
parser.add_argument('--split', type = int, default = 0)
parser.add_argument('--val_split', type = eval, default = None)
parser.add_argument('--n_splits', type = int, default = 5)
parser.add_argument('--choose_orig_val', type = eval, default = False)
# exclude seizure types to include in training but not evaluation
parser.add_argument('--onlytrainseiz', default = None)
parser.add_argument('--eval_seiz_classes', nargs = '+', default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
parser.add_argument('--seiz_classes', nargs = '+', default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
parser.add_argument('--window_length', type=float, default = 2)
parser.add_argument('--batch_size', type=eval, default=512)
# protocol(s) to use for training
parser.add_argument('--protocol', type=str, default= 'all')

# model
parser.add_argument('--model_type', type=str, default='BaselineCNN')
parser.add_argument('--model_path', type=str)
parser.add_argument('--glob_avg_pool', type=eval, default=False)
parser.add_argument('--dropoutprob', type=float, default=0.4)
parser.add_argument('--cnn_dropoutprob', type=float, default=0.4)
parser.add_argument('--padding', type=eval, default=True)              

    # visualization
parser.add_argument('--n_iterations', type = int, default = 2)

args = parser.parse_args(['--file_path','/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 
                          '--window_length', '2',
                          '--model_path', '/Users/theabrusch/Desktop/Speciale_data/final_final_model.pt'
                          ])
main(args)