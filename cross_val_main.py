from re import S
from dataapi import data_collection as dc
import argparse
import logging
import sys
import numpy as np
import yaml
from prettytable import PrettyTable
from src.data import get_generator, train_val_split
from src.models import get_model, get_optim, get_loss, train_model, metrics
from datetime import datetime
from src.models.metrics import sensitivity, specificity, accuracy
from sklearn.metrics import f1_score, precision_score
from torch.utils.tensorboard import SummaryWriter
import optuna
import pickle
from pathlib import Path

class LogParamsToTB:
    def __init__(self, writer):
        self.writer = writer
        self.time = datetime.now()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        params = trial.params
        pruned = trial.state == optuna.trial.TrialState.PRUNED
        args_name = 'args'
        f1 = trial.value
        sens = trial.user_attrs['sens']
        spec = trial.user_attrs['spec']
        f1_seq = trial.intermediate_values
        self.writer.add_text(args_name+'/f1_seq'+str(trial.number), str(f1_seq), global_step=0)
        t = PrettyTable(['Argument', 'Value'])
        run_time = (datetime.now()-self.time).total_seconds()
        t.add_row(['time', run_time])
        for key, val in params.items():
            t.add_row([key, val])

        t.add_row(['pruned', pruned])
        t.add_row(['f1', f1])
        t.add_row(['sens', sens])
        t.add_row(['spec', spec])
        self.writer.add_text(args_name+'/params' + str(trial.number), t.get_html_string(), global_step=0)
        self.time = datetime.now()

def main(args):
    writer = SummaryWriter('../runs/' + args.run_folder + '/' + args.model_type +\
                           '_split_' + str(args.split)+'_'+str(args.val_split)+ \
                            str(datetime.now()) + '_' + \
                            args.job_name)
    with open('configuration.yml', 'r') as file:
        config = yaml.safe_load(file)

    splitdict = config['data_gen'].copy()
    splitdict['hdf5_path'] = args.file_path
    splitdict['split'] = args.split
    splitdict['seiz_classes'] = args.seiz_classes
    splitdict['n_splits'] = args.n_splits
    splitdict['n_val_splits'] = args.n_val_splits
    splitdict['val_split'] = args.val_split
    # get split
    train, val, test = train_val_split.get_kfold(**splitdict)
    split = {'train': train, 'val': val, 'test': test}

    train, val, test = split['train'], split['val'], split['test']
    print('Train:',train)
    print('Validation:', val)
    print('Test:', test)

    # validation loader
    datagen = config['data_gen'].copy()
    datagen['seiz_classes'] = args.seiz_classes
    datagen['hdf5_path'] = args.file_path
    datagen['window_length'] = args.window_length
    datagen['bckg_stride'] = args.window_length
    datagen['seiz_stride'] = args.window_length
    datagen['bckg_rate'] = None
    datagen['anno_based_seg'] = True
    datagen['prefetch_data_from_seg'] = True
    datagen['standardise'] = False
    datagen['use_train_seed'] = True
    datagen['subj_strat'] = False
    datagen['batch_size'] = args.batch_size

    val_dataloader = get_generator.get_dataset_cross_val(data_gen = datagen, 
                                                         subjs_to_use = val, 
                                                         writer = writer)
    fit_config = config['fit']
    fit_config['weight'] = val_dataloader.dataset.bckg_rate
    # get weighted validation loss
    val_loss_fn = get_loss.get_loss(**fit_config)
    # optimize
    def objective(trial):
        # get datasets and dataloaders
        datagen = config['data_gen']
        datagen['seiz_classes'] = args.seiz_classes
        datagen['hdf5_path'] = args.file_path
        datagen['window_length'] = args.window_length
        stride = trial.suggest_categorical('stride', args.stride)
        datagen['bckg_stride'] = stride
        datagen['seiz_stride'] = stride
        bckg_rate = trial.suggest_categorical('bckg_rate', args.bckg_rate)
        datagen['bckg_rate'] = bckg_rate
        datagen['anno_based_seg'] = True
        datagen['prefetch_data_from_seg'] = True
        datagen['standardise'] = False
        datagen['use_train_seed'] = True
        datagen['subj_strat'] = False
        datagen['protocol'] = 'all'
        datagen['batch_size'] = args.batch_size

        train_dataloader = get_generator.get_dataset_cross_val(data_gen = datagen, subjs_to_use=train)

        # load model
        model_config = config['model_kwargs']
        model_config['model'] = args.model_type
        if not args.separate_dropout:
            dropout = trial.suggest_float('dropout', 0, 0.7)
            model_config['dropoutprob'] = dropout
            model_config['cnn_dropoutprob'] = dropout
        else:
            model_config['dropoutprob'] = trial.suggest_float('dropout', 0, 0.7)
            model_config['cnn_dropoutprob'] = trial.suggest_float('cnn_dropout', 0, 0.7)
            
        model_config['glob_avg_pool'] = trial.suggest_categorical('glob_avg_pool', [True, False])
        model_config['padding'] = trial.suggest_categorical('padding', [True, False])
        model_config['input_shape'] = train_dataloader.dataset._get_X_shape()
        model = get_model.get_model(model_config)

        # train model
        optim_config = config['fit']['optimizer']
        optim_config['optimizer'] = args.optimizer
        optim_config['scheduler'] = args.scheduler
        optim_config['milestones'] = args.milestones
        optim_config['model'] = args.model_type
        optim_config['lr'] = trial.suggest_loguniform('lr', 1e-6, 1e-2)
        optim_config['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-1)
        optimizer, scheduler = get_optim.get_optim(model, optim_config)

        if args.use_weighted_loss:
            fit_config['weight'] = bckg_rate
        else:
            fit_config['weight'] = None

        loss_fn = get_loss.get_loss(**fit_config)

        choose_best = False
        model_train = train_model.model_train(model = model, 
                                                optimizer = optimizer, 
                                                loss_fn = loss_fn, 
                                                val_loss = val_loss_fn,
                                                writer = writer,
                                                scheduler = scheduler,
                                                choose_best = choose_best)

        sensspec, sens, spec, f1, prec = model_train.train(train_loader = train_dataloader,
                                                val_loader = val_dataloader,
                                                test_loader = None,
                                                epochs = args.epochs,
                                                early_stopping=True,
                                                trial = trial)
        trial.set_user_attr('sens', sens)
        trial.set_user_attr('spec', spec)
        trial.set_user_attr('sensspec', sensspec)
        trial.set_user_attr('prec', prec)
        return f1

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    callback = LogParamsToTB(writer)

    if not args.load_existing:
        job_name = args.job_name + str(np.random.rand())
    else:
        job_name = args.job_name

    study = optuna.study.create_study(study_name = job_name, 
                                      direction = 'maximize', 
                                      pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience = 5),
                                      storage = 'sqlite:///data/optuna_trials_final_new.db',
                                      load_if_exists = args.load_existing)
    study.optimize(objective, 
                   args.n_trials, 
                   timeout = args.time_out,
                   callbacks=[callback])

    df = study.trials_dataframe()
    file_name = 'data/optuna_trials/' + job_name + str(datetime.now()) + '.csv'
    df.to_csv(file_name)

if __name__ == '__main__':
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
    parser.add_argument('--separate_dropout', type=eval, default=False)      

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

    args = parser.parse_args()
    main(args)
