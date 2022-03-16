from dataapi import data_collection as dc
import argparse
import torch
import numpy as np
import yaml
import pickle
from prettytable import PrettyTable
from src.data import train_val_split, datagenerator
from src.models import get_model, get_optim, get_loss, train_model, metrics
from src.visualization import perturbation
from datetime import datetime
from src.models.metrics import sensitivity, specificity, accuracy
from sklearn.metrics import f1_score, precision_score
from torch.utils.tensorboard import SummaryWriter

print('done loading packages')
def params_to_tb(writer, args):
    t = PrettyTable(['Argument', 'Value'])
    param_dict = vars(args)
    for key, val in param_dict.items():
        t.add_row([key, val])
    writer.add_text("args", t.get_html_string(), global_step=0)

def main(args):
    writer = SummaryWriter('../runs/' + args.run_folder + '/' + args.model_type +\
                           '_'+ str(datetime.now()) + '_' + \
                            args.job_name + '_split_' + str(args.split))
    params_to_tb(writer, args)
    with open('configuration.yml', 'r') as file:
        config = yaml.safe_load(file)

    # get split
    splitdict = dict()
    splitdict['hdf5_path'] = args.file_path
    splitdict['split'] = args.split
    splitdict['only_train_seiz'] = args.onlytrainseiz
    splitdict['val_split'] = args.val_split
    splitdict['n_val_splits'] = 7
    splitdict['excl_seiz'] = False
    splitdict['seiz_classes'] = args.seiz_classes
    splitdict['n_splits'] = args.n_splits
    splitdict['choose_orig_val'] = False
    splitdict['orig_split'] = True
    
    if args.val_split is None:
        _, test = train_val_split.get_kfold(**splitdict)
    else:
        _, test, _ = train_val_split.get_kfold(**splitdict)
    # get testloader
    datagen = config['data_gen']
    datagen['seed'] = args.seed
    datagen['seiz_classes'] = args.eval_seiz_classes
    datagen['hdf5_path'] = args.file_path
    datagen['window_length'] = args.window_length
    datagen['prefetch_data_from_seg'] = False
    datagen['protocol'] = args.protocol
    datagen['batch_size'] = args.batch_size
    datagen['use_train_seed'] = False
    datagen['bckg_stride'] = None
    datagen['seiz_stride'] = None
    datagen['bckg_rate'] = None
    datagen['anno_based_seg'] = False

    # load model
    model_config = config['model_kwargs']
    model_config['model'] = args.model_type
    model_config['cnn_dropoutprob'] = args.cnn_dropoutprob
    model_config['dropoutprob'] = args.dropoutprob
    model_config['glob_avg_pool'] = args.glob_avg_pool
    model_config['padding'] = args.padding
    model_config['input_shape'] = (20,500)

    # load trained model
    model = get_model.get_model(model_config)
    checkpoint = torch.load(args.model_path, map_location = 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # save temporary results for further analysis
    pickle_path = 'data/predictions/'+ args.job_name + str(datetime.now()) + '_split_' + str(args.split) + '_pertmaps.pickle'

    pert_total = dict()
    j = 1
    for subj in test:
        print('Computing perturbations for subject', j, 'out of', len(test))
        j += 1
        datasegment = datagenerator.SegmentData(**datagen,
                                                subjects_to_use = [subj])
        segment, norm_coef = datasegment.segment_data(split = 'test')
        seiz_types = segment['seiz_types'].unique()
        pert_total[subj] = dict()
        #do perturbation test for background and different seizure types separately
        for seiz in seiz_types:
            seiz_seg = segment[segment['seiz_types']==seiz].reset_index()
            dataset = datagenerator.DataGenerator(**datagen, subjects_to_use=[subj], split = 'test',
                                                  segments = segment, norm_coef = norm_coef)
            #extract segments
            input_data = np.zeros((len(seiz_seg), 20, 500))

            for i, seg in seiz_seg.iterrows():
                temp = dataset._get_segment(seg)
                input_data[i,:,:] = temp

            # get perturbations
            if seiz in args.seiz_classes:
                label = 1
            else:
                label = 0
            
            pert_map = perturbation.perturbation_maps(model, input_data, args.n_iterations, 
                                                      correct_wrong=args.correct_wrong, label = label)
            pert_total[subj][seiz] = pert_map

            with open(pickle_path, 'wb') as fp:
                pickle.dump(pert_total, fp)

    writer.close()

if __name__ == '__main__':
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
    parser.add_argument('--n_iterations', type = int, default = 100)
    parser.add_argument('--correct_wrong', type = eval, default = False)

    args = parser.parse_args()
    main(args)
