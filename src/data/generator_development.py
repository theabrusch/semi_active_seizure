from dataapi import data_collection as dc
import numpy as np
#from src.data import datagenerator
import datagenerator

F = dc.File('data/hdf5/temple_seiz.hdf5', 'r')

dataloader = datagenerator.DataGenerator('data/hdf5/temple_seiz.hdf5', 
                                         window_length=4, protocol = 'train', 
                                         signal_name='TCP')
item = dataloader.__getitem__(0)

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
