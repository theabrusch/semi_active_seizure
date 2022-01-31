from main_transfer_ssl import main
import argparse
# job name
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type = str)
parser.add_argument('--run_folder', default = 'None')
parser.add_argument('--job_name', default = 'None')
parser.add_argument('--seed', type = int, default = 20)
parser.add_argument('--min_seiz', default = 20)
parser.add_argument('--onlytrainseiz', default = None)
parser.add_argument('--val_split', type = int, default = 1)
parser.add_argument('--split', type = int, default = 2)
# minimum ratio of background in transfer dataset
parser.add_argument('--min_ratio', default = 20)
# exclude seizure types to include in training but not evaluation
parser.add_argument('--seiz_classes', nargs = '+', default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
parser.add_argument('--window_length', type=float, default = 2)
parser.add_argument('--bckg_stride', type=eval, default=None)
parser.add_argument('--seiz_stride', type=eval, default=None)
parser.add_argument('--bckg_rate_val', type=eval, default=20) # None or value
parser.add_argument('--bckg_rate_train', type=eval, default=None)
parser.add_argument('--anno_based_seg', type=eval, default=True)
parser.add_argument('--train_val_test', type=eval, default=False)
parser.add_argument('--batch_size', type=eval, default=64)
# protocol(s) to use for training
parser.add_argument('--protocol', type=str, default= 'all')

# model
parser.add_argument('--model_type', type=str, default='BaselineCNN')
parser.add_argument('--glob_avg_pool', type=eval, default=False)
parser.add_argument('--dropoutprob', type=float, default=0.4)
parser.add_argument('--padding', type=eval, default=True)     
parser.add_argument('--model_path', type=str)    

# Training parameters
parser.add_argument('--optimizer', type = str, default = 'RMSprop')
parser.add_argument('--scheduler', type = str, default = None)
parser.add_argument('--milestones', type = eval, default = [50, 130, 150])
parser.add_argument('--use_weighted_loss', type=eval, default=True)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type = float, default=1e-3)

trans_subjs = ['/test/00008174', '/test/00008544', '/train/00000609',
       '/train/00000820', '/train/00001891', '/train/00005452',
       '/train/00006230', '/train/00006520', '/train/00008204',
       '/train/00008345', '/train/00010418', '/test/00000629',
       '/test/00000795', '/test/00001770', '/test/00001981',
       '/test/00002289', '/test/00003281', '/test/00004594',
       '/test/00004671', '/test/00005031', '/test/00005943',
       '/test/00006900', '/test/00007633', '/train/00000017',
       '/train/00000032', '/train/00000054', '/train/00000077',
       '/train/00000107', '/train/00000148', '/train/00000175',
       '/train/00000184', '/train/00000216', '/train/00000244',
       '/train/00000254', '/train/00000289', '/train/00000427',
       '/train/00000458', '/train/00000502', '/train/00000577',
       '/train/00000598', '/train/00000604', '/train/00000659',
       '/train/00000705', '/train/00000762', '/train/00000908',
       '/train/00000924', '/train/00000929', '/train/00001026',
       '/train/00001030', '/train/00001050', '/train/00001217',
       '/train/00001246', '/train/00001267', '/train/00001317',
       '/train/00001331']

args = parser.parse_args(['--file_path','/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', '--window_length', '2',
                          '--bckg_stride', '2', '--seiz_stride', '2', 
                          '--model_path', '/Users/theabrusch/Desktop/Speciale_data/final_model.pt'
                          ])
main(args)