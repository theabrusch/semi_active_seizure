from dataapi import data_collection as dc
import numpy as np
from src.data import datagenerator
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models import baselinemodels
import torch
from src.data import train_val_split
import yaml
from src.data import get_generator

with open('configuration.yml', 'r') as file:
    config = yaml.safe_load(file)

train_dataloader, val_dataloader = get_generator.get_generator(config['data_gen'])

train_dataset = datagenerator.DataGenerator(**config['data_gen'])
# Setup datasets
hdf5_path = 'data/hdf5/temple_seiz_sub.hdf5'
train, val = train_val_split.train_val_split(hdf5_path, 0.8)
background_rate = 10

train_dataset = datagenerator.DataGenerator(hdf5_path, 
                                            window_length = 2, stride = [1,2],
                                            protocol = 'train', signal_name = 'TCP', 
                                            bckg_rate = background_rate, anno_based_seg = True,
                                            subjects_to_use=train, 
                                            prefetch_data_dir = False,
                                            prefetch_data_from_seg = True)
val_dataset = datagenerator.DataGenerator(hdf5_path, 
                                          window_length = 2, stride = [1,2], 
                                          protocol = 'train', signal_name = 'TCP', 
                                          bckg_rate = background_rate, anno_based_seg=True,
                                          subjects_to_use=val, prefetch_data_from_seg = True)

# Setup dataloaders
train_weights = train_dataset.weights
train_sampler = WeightedRandomSampler(train_weights, num_samples=train_dataset.__len__(), replacement = True)
train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_weights = val_dataset.weights
val_sampler = WeightedRandomSampler(val_weights, num_samples=val_dataset.__len__(), replacement = True)
val_dataloader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)