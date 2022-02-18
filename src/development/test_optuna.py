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


study = optuna.study.load_study('test_optuna_full_run_split2_valsplit1', storage = 'sqlite:///Users/theabrusch/Desktop/Speciale_data/optuna_trials.db')