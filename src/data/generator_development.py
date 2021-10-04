from dataapi import data_collection as dc
from h5py._hl import dataset
import numpy as np
#from src.data import datagenerator
from torch.utils.data import DataLoader, WeightedRandomSampler
import datagenerator

dataset = datagenerator.DataGenerator('data/hdf5/temple_seiz.hdf5', 
                                      window_length = 4, stride = [4, 2], 
                                      protocol = 'test', signal_name = 'TCP', 
                                      bckg_rate = 1, anno_based_seg=True)
weights = dataset.samples['weight']
sampler = WeightedRandomSampler(weights, num_samples=dataset.__len__(), replacement = True)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
