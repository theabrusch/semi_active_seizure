from dataapi import data_collection as dc
import numpy as np
from src.data import datagenerator
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models import baselinemodels
import torch
from src.data import train_val_split

hdf5_path = 'data/hdf5/temple_seiz_sub.hdf5'
train, val = train_val_split.train_val_split(hdf5_path, 0.8)

train_dataset = datagenerator.DataGenerator(hdf5_path, 
                                            window_length = 2, stride = 2,
                                            protocol = 'train', signal_name = 'TCP', 
                                            bckg_rate = 1, anno_based_seg = False,
                                            subjects_to_use=train[0:2], 
                                            prefetch_data_dir = False,
                                            prefetch_data_from_seg = False)
val_dataset = datagenerator.DataGenerator(hdf5_path, 
                                          window_length = 2, stride = 1, 
                                          protocol = 'train', signal_name = 'TCP', 
                                          bckg_rate = 20, anno_based_seg=True,
                                          subjects_to_use=val[0:2])
temp = train_dataset.__getitem__(0)
train_weights = train_dataset.weights
train_sampler = WeightedRandomSampler(train_weights, num_samples=train_dataset.__len__(), replacement = True)
train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_weights = val_dataset.weights
val_sampler = WeightedRandomSampler(val_weights, num_samples=val_dataset.__len__(), replacement = True)
val_dataloader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)

temp = next(iter(train_dataloader))
model = baselinemodels.BaselineCNN(input_shape=train_dataset._get_X_shape())
out = model(torch.Tensor(temp[0]))

seiz_duration = dict()
bckg_duration = dict()

F = dc.File('data/hdf5/temple_seiz.hdf5', 'r')

for protocol in F.keys():
    seiz_duration[protocol] = []
    bckg_duration[protocol] = []
    for subj in F[protocol].keys():
        for rec in F[protocol][subj].keys():
            annos = F[protocol][subj][rec]['Annotations']
            for anno in annos:
                if anno['Name'] in ['cpsz', 'gnsz', 'spsz', 'tcsz']:
                    seiz_duration[protocol].append(anno['Duration'])
                else:
                    bckg_duration[protocol].append(anno['Duration'])