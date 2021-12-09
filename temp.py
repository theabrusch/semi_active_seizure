import numpy as np
import pyedflib
from dataapi import data_collection as dc

file = open('/Users/theabrusch/Desktop/Speciale_data/chb-mit-scalp-eeg-database-1.0.0/chb09/chb09_06.edf.seizures')
lines = file.readlines()

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/boston_scalp.hdf5'

f = dc.File(file_name, 'r')
total_dur = 0

for subj in f['train'].keys():
    subject = f['train'][subj]
    dur = np.sum(subject.attrs['time']['Duration'])
    total_dur += dur

annotations = f.get_children(object_type=dc.Annotations)
seiz = 0

for annos in annotations:
    for anno in annos:
        if anno['Name'] == 'seiz':
            seiz += 1
