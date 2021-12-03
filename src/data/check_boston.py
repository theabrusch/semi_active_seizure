from dataapi import data_collection as dc
import numpy as np
import pickle
from src.data import train_val_split


F = dc.File('/Users/theabrusch/Desktop/Speciale_data/boston_scalp_new.hdf5', 'r')
subject = F['train/chb12']
seiz_recs = train_val_split.get_seiz_recs(subject, seiz_classes=['seiz'])
print(seiz_recs)

total_dur = 0
seiz_dur = 0
seiz_rec_dur = 0 

for subj in F['train'].keys():
    records = list(F['train'][subj].keys())
    uni, count = np.unique(records, return_counts=True)
    idx = (count > 1)
    print(subj)
    subj_anno_dur = 0
    for rec in records:
        record = F['train'][subj][rec]
        total_dur += record.duration
        seiz_rec = 0
        for anno in record['Annotations']:
            if anno['Name'].lower() == 'seiz':
                subj_anno_dur += anno['Duration']
                seiz_rec = 1
        if seiz_rec == 1:
            seiz_rec_dur += record.duration
    print(subj_anno_dur)
    seiz_dur += subj_anno_dur
