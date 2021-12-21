import numpy as np
import pyedflib
from dataapi import data_collection as dc

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz.hdf5'

f = dc.File(file_name, 'r')

seiz_subjs = []
seiz_dur = 0
total_dur = 0

for subj in f['train'].keys():
    subject = f['train'][subj]
    seiz = False
    for rec in subject.keys():
        record = subject[rec]
        for anno in record['Annotations']:
            if anno['Name'] in ['cpsz', 'gnsz', 'spsz', 'tcsz', 'seiz']:
                seiz = True
                seiz_dur += anno['Duration']
        total_dur+=record.duration
    if seiz:
        seiz_subjs.append(subj)
