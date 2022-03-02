from dataapi import data_collection as dc
import argparse
import torch
import yaml
import pickle
from prettytable import PrettyTable
from src.data import get_generator, train_val_split
from src.models import get_model, get_optim, get_loss, train_model, metrics
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

    time_start = datetime.now()
    # get split
    splitdict = dict()
    splitdict['hdf5_path'] = args.file_path
    splitdict['split'] = args.split
    splitdict['only_train_seiz'] = args.onlytrainseiz
    splitdict['val_split'] = args.val_split
    splitdict['n_val_splits'] = 7
    splitdict['excl_seiz'] = args.excl_seiz
    splitdict['seiz_classes'] = args.seiz_classes
    splitdict['n_splits'] = args.n_splits
    splitdict['choose_orig_val'] = args.choose_orig_val
    splitdict['orig_split'] = args.orig_split
    
    _, test = train_val_split.get_kfold(**splitdict)

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
    test_dataloader = get_generator.get_dataset_cross_val(data_gen = datagen, 
                                                            subjs_to_use = test,
                                                            split = 'test',
                                                            writer = writer)
    
    # load model
    model_config = config['model_kwargs']
    model_config['model'] = args.model_type
    model_config['cnn_dropoutprob'] = args.cnn_sdropoutprob
    model_config['dropoutprob'] = args.dropoutprob
    model_config['glob_avg_pool'] = args.glob_avg_pool
    model_config['padding'] = args.padding
    model_config['input_shape'] = test_dataloader.dataset._get_X_shape()

    # load trained model
    model = get_model.get_model(model_config)
    checkpoint = torch.load(args.model_path, map_location = 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model_train = train_model.model_train(model = model, 
                                          optimizer = None, 
                                          loss_fn = None, 
                                          val_loss = None,
                                          writer = writer,
                                          scheduler = None,
                                          choose_best = False)


    test_dataloader.dataset.return_seiz_type = True
    y_pred, y_true, seiz_types, probability = model_train.eval(test_dataloader, return_probability=True, return_seiz_type = True)

    segments = test_dataloader.dataset.segments
    segments['seiz prob'] = probability

    # save results for further analysis
    pickle_path = 'data/predictions/'+ args.job_name + '_split_' + str(args.split) + '_results.pickle'

    with open(pickle_path, 'wb') as fp:
        pickle.dump(segments, fp)

    # calculate metrics
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    acc = accuracy(y_true, y_pred)

    writer.add_scalar('test_final/sensitivity', sens)
    writer.add_scalar('test_final/specificity', spec)
    writer.add_scalar('test_final/f1', f1)
    writer.add_scalar('test_final/precision', prec)
    writer.add_scalar('test_final/accuracy', acc)

    # calculate metrics for different seizure types
    import numpy as np
    uni_seiz_types = np.unique(seiz_types)
    t = PrettyTable(['Seizure type', 'Sensitivity', 'Number of segments'])        
    for seiz in uni_seiz_types:
        if seiz != 'bckg':
            idx = seiz_types == seiz
            y_true_temp = y_true[idx]
            y_pred_temp = y_pred[idx]
            sens_temp = sensitivity(y_true_temp, y_pred_temp)
            t.add_row([seiz, sens_temp, len(y_true_temp)])

    writer.add_text("Seizure specific performance", t.get_html_string(), global_step=0)

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
    parser.add_argument('--orig_split', type = eval, default=False)
    # exclude subjects that have 1 or more seizures not included in analysis
    parser.add_argument('--excl_seiz', type = eval, default = False) 
    # exclude seizure types to include in training but not evaluation
    parser.add_argument('--onlytrainseiz', default = None)
    parser.add_argument('--seiz_classes', nargs = '+', default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
    parser.add_argument('--seiz_classes', nargs = '+', default=['fnsz', 'tcsz'])
    parser.add_argument('--window_length', type=float, default = 2)
    parser.add_argument('--batch_size', type=eval, default=512)
    # protocol(s) to use for training
    parser.add_argument('--protocol', type=str, default= 'all')

    # model
    parser.add_argument('--model_type', type=str, default='BaselineCNN')
    parser.add_argument('--glob_avg_pool', type=eval, default=False)
    parser.add_argument('--dropoutprob', type=float, default=0.4)
    parser.add_argument('--cnn_dropoutprob', type=float, default=0.4)
    parser.add_argument('--padding', type=eval, default=False)              

    args = parser.parse_args()
    main(args)
