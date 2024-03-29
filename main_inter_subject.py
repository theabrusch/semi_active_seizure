from dataapi import data_collection as dc
import argparse
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
    splitdict['excl_seiz'] = args.excl_seiz
    splitdict['seiz_classes'] = args.seiz_classes
    splitdict['n_splits'] = 5
    splitdict['seed'] = args.seed
    splitdict['test_frac'] = args.test_frac

    train, test = train_val_split.get_kfold(**splitdict)
    splitdict['subjects'] = train
    
    train_records, test_records = train_val_split.get_inter_subjects(**splitdict) 

    # get trainloader
    train_datagen = config['data_gen']
    train_datagen['seed'] = args.seed
    train_datagen['seiz_classes'] = args.seiz_classes
    train_datagen['hdf5_path'] = args.file_path
    train_datagen['window_length'] = args.window_length
    train_datagen['bckg_stride'] = args.bckg_stride
    train_datagen['seiz_stride'] = args.seiz_stride
    train_datagen['bckg_rate'] = args.bckg_rate_train
    train_datagen['anno_based_seg'] = args.anno_based_seg
    train_datagen['prefetch_data_from_seg'] = True
    train_datagen['protocol'] = args.protocol
    train_datagen['batch_size'] = args.batch_size
    train_datagen['use_train_seed'] = True

    train_dataloader = get_generator.get_dataset_transfer(data_gen = train_datagen, 
                                                            subjs_to_use = train, 
                                                            records_to_use = train_records, 
                                                            split = 'train',
                                                            writer = None)

    # get test loader
    test_datagen = train_datagen.copy()
    test_datagen['bckg_stride'] = None
    test_datagen['seiz_stride'] = None
    test_datagen['bckg_rate'] = None
    test_datagen['anno_based_seg'] = False
    test_datagen['seiz_classes'] = args.eval_seiz_classes
    test_dataloader = get_generator.get_dataset_cross_val(data_gen = test_datagen, 
                                                            subjs_to_use = test,
                                                            split = 'test',
                                                            writer = writer)
    # val data loader
    val_dataloader = get_generator.get_dataset_transfer(data_gen = test_datagen, 
                                                            subjs_to_use = train, 
                                                            records_to_use = test_records, 
                                                            split = 'val',
                                                            writer = None)

    # load model
    model_config = config['model_kwargs']
    model_config['model'] = args.model_type
    model_config['dropoutprob'] = args.dropoutprob
    model_config['glob_avg_pool'] = args.glob_avg_pool
    model_config['padding'] = args.padding
    model_config['input_shape'] = train_dataloader.dataset._get_X_shape()
    model = get_model.get_model(model_config)

    # train model
    optim_config = config['fit']['optimizer']
    optim_config['optimizer'] = args.optimizer
    optim_config['scheduler'] = args.scheduler
    optim_config['milestones'] = args.milestones
    optim_config['model'] = args.model_type
    optim_config['lr'] = args.lr
    optim_config['weight_decay'] = args.weight_decay
    optimizer, scheduler = get_optim.get_optim(model, optim_config)

    fit_config = config['fit']

    if args.use_weighted_loss:
        fit_config['weight'] = train_dataloader.dataset.bckg_rate
    else:
        fit_config['weight'] = None

    loss_fn = get_loss.get_loss(**fit_config)
    fit_config['weight'] = val_dataloader.dataset.bckg_rate
    test_loss = get_loss.get_loss(**fit_config)

    model_train = train_model.model_train(model = model, 
                                            optimizer = optimizer, 
                                            loss_fn = loss_fn, 
                                            val_loss = test_loss,
                                            writer = writer,
                                            scheduler = scheduler,
                                            choose_best = False)

    time = datetime.now()
    train_loss, val_loss = model_train.train(train_loader = train_dataloader,
                                            val_loader = val_dataloader,
                                            test_loader = test_dataloader,
                                            job_name = args.job_name,
                                            safe_best_model= args.save_best_model,
                                            epochs = args.epochs)
                                            
    print('Training model for', args.epochs, 'epochs took', datetime.now()-time, '.')
    print('Total time', datetime.now()-time_start, '.')

    #Evaluation
    test_dataloader.dataset.return_seiz_type = True
    y_pred, y_true, seiz_types, probability = model_train.eval(test_dataloader, return_probability=True,
                                                                return_seiz_type = True)

    segments = test_dataloader.dataset.segments
    segments['y pred'] = y_pred
    segments['output proba'] = probability

    # save results for further analysis
    pickle_path = args.job_name + '_split_' + str(args.split) + '_results.pickle'

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

    val_dataloader.dataset.return_seiz_type = True
    y_pred, y_true, seiz_types, probability = model_train.eval(val_dataloader, return_probability=True,
                                                                return_seiz_type = True)

    segments = val_dataloader.dataset.segments
    segments['y pred'] = y_pred
    segments['output proba'] = probability

    # save results for further analysis
    pickle_path = args.job_name + '_split_' + str(args.split) + '_val_results.pickle'

    with open(pickle_path, 'wb') as fp:
        pickle.dump(segments, fp)

    # calculate metrics
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    acc = accuracy(y_true, y_pred)

    writer.add_scalar('val_final/sensitivity', sens)
    writer.add_scalar('val_final/specificity', spec)
    writer.add_scalar('val_final/f1', f1)
    writer.add_scalar('val_final/precision', prec)
    writer.add_scalar('val_final/accuracy', acc)

    uni_seiz_types = np.unique(seiz_types)
    t = PrettyTable(['Seizure type', 'Sensitivity', 'Number of segments'])        
    for seiz in uni_seiz_types:
        if seiz != 'bckg':
            idx = seiz_types == seiz
            y_true_temp = y_true[idx]
            y_pred_temp = y_pred[idx]
            sens_temp = sensitivity(y_true_temp, y_pred_temp)
            t.add_row([seiz, sens_temp, len(y_true_temp)])
    writer.add_text("Val. Seizure specific performance", t.get_html_string(), global_step=0)

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
    parser.add_argument('--test_frac', type = float, default = 1/3)
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
    parser.add_argument('--padding', type=eval, default=False)       
    parser.add_argument('--save_best_model', type=eval, default=False)       

    # Training parameters
    parser.add_argument('--optimizer', type = str, default = 'RMSprop')
    parser.add_argument('--scheduler', type = str, default = None)
    parser.add_argument('--milestones', type = eval, default = [50, 130, 150])
    parser.add_argument('--use_weighted_loss', type=eval, default=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type = float, default=1e-3)

    args = parser.parse_args()
    main(args)
