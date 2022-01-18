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

    # optimize
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        
        for i in range(3):
            x+=0.1
            y+=0.1
            trial.report((x - 2) ** 2 + (y - 3) ** 2, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        sens = x
        spec = y

        trial.set_user_attr('sens', sens)
        trial.set_user_attr('spec', spec)
        return (x - 2) ** 2 + (y - 3) ** 2

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
    file_name = 'data/optuna_trials/' + args.jobname + str(datetime.now()) + '.csv'
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
