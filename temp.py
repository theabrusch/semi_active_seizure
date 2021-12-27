import numpy as np
import pyedflib
from dataapi import data_collection as dc

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/boston_scalp.hdf5'

f = dc.File(file_name, 'r')

seiz_subjs = []
seiz_dur = 0
total_dur = 0
stats = dict()

for subj in f['train'].keys():
    subject = f['train'][subj]
    stats[subj] = dict()
    stats[subj]['total dur'] = 0
    stats[subj]['seizures'] = 0
    stats[subj]['seiz dur'] = 0
    for rec in subject.keys():
        record = subject[rec]
        for anno in record['Annotations']:
            if anno['Name'] in ['cpsz', 'gnsz', 'spsz', 'tcsz', 'seiz']:
                seiz = True
                stats[subj]['seizures'] += 1
                stats[subj]['seiz dur'] += anno['Duration']
        stats[subj]['total dur'] += record.duration


for subj in stats.keys():
    print('Subject', subj)
    print('Number of seizures', stats[subj]['seizures'])
    print('Seizure duration', stats[subj]['seiz dur'])
    print('Seizure percent', stats[subj]['seiz dur']*100/stats[subj]['total dur'])
    print('Total duration', stats[subj]['total dur']/60/60)
    print('\n')