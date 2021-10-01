from dataapi import data_collection as dc
from h5py._hl import dataset
import numpy as np
#from src.data import datagenerator
from torch.utils.data import DataLoader, WeightedRandomSampler
import datagenerator

dataset = datagenerator.DataGenerator('data/hdf5/temple_seiz.hdf5', 
                                         window_length = 4, protocol = 'train', 
                                         signal_name = 'TCP', bckg_rate = 20)
#weights = dataset.samples['weight']
#sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement = False)
dataloader = DataLoader(dataset, batch_size=64)#, sampler=sampler)

subj = F['train/00006904']
rec = F['train/00006904/s004_t000']
sig = F['train/00006904/s004_t000/TCP']
annos = rec['Annotations']

seiz_classes = ['cpsz', 'gnsz', 'spsz', 'tcsz']
one_hot_label = np.zeros((len(sig), 2))
for anno in rec['Annotations']:
    anno_start = (anno['Start'] - rec.start_time)*sig.fs
    anno_end = anno_start+anno['Duration']*sig.fs
    if anno['Name'].lower() in seiz_classes:
        one_hot_label[round(anno_start):round(anno_end),1] = 1
    else:
        one_hot_label[round(anno_start):round(anno_end),0] = 1


time_vector = sig._generate_time_vector(sig.start_time, sig.start_time + sig.duration, fs = 1/4)

rec2 = subj['s004_t001']
sig2 = rec2['TCP']
time_vector = sig2._generate_time_vector(sig2.start_time, sig2.start_time + sig2.duration, fs = 1/4)
