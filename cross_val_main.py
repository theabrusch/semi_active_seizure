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

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        params = trial.params
        pruned = trial.state == optuna.trial.TrialState.PRUNED
        f1 = trial.value
        sens = trial.user_attrs['sens']
        spec = trial.user_attrs['spec']
        f1_seq = trial.intermediate_values

        t = PrettyTable(['Argument', 'Value'])
        for key, val in params.items():
            t.add_row([key, val])

        t.add_row(['pruned', pruned])
        t.add_row(['f1', f1])
        t.add_row(['sens', sens])
        t.add_row(['spec', spec])
        args_name = 'args' + str(trial.number)
        self.writer.add_text(args_name, t.get_html_string(), global_step=0)


def main(args):
    writer = SummaryWriter('../runs/' + args.run_folder + '/' + args.model_type +\
                           '_'+ str(datetime.now()) + '_' + \
                            args.job_name)
    with open('configuration.yml', 'r') as file:
        config = yaml.safe_load(file)

    splitdict = config['data_gen']
    splitdict['hdf5_path'] = args.file_path
    splitdict['protocol'] = 'all'
    # get split
    split_path = 'data/optuna_trials/optuna_split_temple/' 
    p = Path(split_path)
    p.mkdir(parents=True, exist_ok=True)
    try:
        with open(split_path + args.job_name + '.pkl', 'rb') as fp:
            split = pickle.load(fp)
    except:
        train, val, test = train_val_split.train_val_test_split(**splitdict)
        split = {'train': train, 'val': val, 'test': test}
        with open(split_path + args.job_name + '.pkl', 'wb') as fp:
            pickle.dump(split, fp)

    train, val, test = split['train'], split['val'], split['test']

    # validation loader
    datagen = config['data_gen']
    datagen['hdf5_path'] = args.file_path
    datagen['window_length'] = args.window_length
    datagen['bckg_stride'] = args.window_length
    datagen['seiz_stride'] = args.window_length
    datagen['bckg_rate'] = None
    datagen['anno_based_seg'] = False
    datagen['prefetch_data_from_seg'] = True
    datagen['standardise'] = False
    datagen['use_train_seed'] = True
    datagen['subj_strat'] = False
    datagen['batch_size'] = args.batch_size

    val_dataloader = get_generator.get_dataset_cross_val(data_gen = datagen, subjs_to_use = val)

    # optimize
    def objective(trial):
        # get datasets and dataloaders
        datagen = config['data_gen']
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
        model_config['dropoutprob'] = trial.suggest_float('dropout', 0, 1)
        model_config['glob_avg_pool'] = trial.suggest_categorical('glob_avg_pool', [True, False])
        model_config['padding'] = True
        model_config['input_shape'] = train_dataloader.dataset._get_X_shape()
        model = get_model.get_model(model_config)

        # train model
        optim_config = config['fit']['optimizer']
        optim_config['optimizer'] = args.optimizer
        optim_config['scheduler'] = args.scheduler
        optim_config['milestones'] = args.milestones
        optim_config['model'] = args.model_type
        optim_config['lr'] = trial.suggest_float('lr', 1e-6, 1e-2)
        optim_config['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-1)
        optimizer, scheduler = get_optim.get_optim(model, optim_config)

        fit_config = config['fit']

        if args.use_weighted_loss:
            fit_config['weight'] = bckg_rate
        else:
            fit_config['weight'] = None

        loss_fn = get_loss.get_loss(**fit_config)

        choose_best = False
        model_train = train_model.model_train(model = model, 
                                                optimizer = optimizer, 
                                                loss_fn = loss_fn, 
                                                writer = None,
                                                scheduler = scheduler,
                                                choose_best = choose_best)

        f1, sens, spec = model_train.train(train_loader = train_dataloader,
                                                val_loader = val_dataloader,
                                                test_loader = None,
                                                epochs = args.epochs,
                                                trial = trial)
        trial.set_user_attr('sens', sens)
        trial.set_user_attr('spec', spec)
        return f1

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    callback = LogParamsToTB(writer)

    if not args.load_existing:
        job_name = args.job_name + str(np.random.rand())
    else:
        job_name = args.job_name

    study = optuna.study.create_study(study_name = job_name, 
                                      direction = 'maximize', 
                                      pruner = optuna.pruners.MedianPruner(),
                                      storage = 'sqlite:///data/optuna_trials.db',
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
    # datagen
    parser.add_argument('--file_path', type = str)
    parser.add_argument('--window_length', type=float, default = 2)
    parser.add_argument('--stride', type=eval, default=[0.05, 0.1, 0.2, 0.5, 0.7, 1, 1.5, 2])
    parser.add_argument('--bckg_rate', type=eval, default=[1, 2, 5]) # None or value
    parser.add_argument('--batch_size', type=eval, default=512)
    # protocol(s) to use for training
    parser.add_argument('--protocol', type=str, default= 'all')

    # model
    parser.add_argument('--model_type', type=str, default='BaselineCNN')
    parser.add_argument('--glob_avg_pool', type=eval, default=False)
    parser.add_argument('--dropoutprob', type=float, default=0.4)
    parser.add_argument('--padding', type=eval, default=False)       

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
