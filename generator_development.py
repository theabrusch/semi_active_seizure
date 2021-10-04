from dataapi import data_collection as dc
import numpy as np
from src.data import datagenerator
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models import baselinemodels
import torch

dataset = datagenerator.DataGenerator('data/hdf5/temple_seiz.hdf5', 
                                      window_length = 2, 
                                      protocol = 'test', signal_name = 'TCP', 
                                      bckg_rate = 1, anno_based_seg=True)
weights = dataset.samples['weight']
sampler = WeightedRandomSampler(weights, num_samples=dataset.__len__(), replacement = True)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
temp = next(iter(dataloader))


model = baselinemodels.BaselineCNN(input_shape=(20,500))
out = model(torch.Tensor(temp[0]))