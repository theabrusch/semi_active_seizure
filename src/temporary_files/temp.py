import numpy as np
import pyedflib
from dataapi import data_collection as dc
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists
import scipy
import pandas as pd

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/boston_scalp.hdf5'
f = dc.File(file_name, 'r')

annos = f.get_children(object_type=dc.Annotations, get_obj = True)
window_length = 2
anno_stride = 1/256
segments = 0
segments_total = 0
total_seiz_dur = 0
for anno in annos:
    for an in anno:
        windows = (an['Duration']-window_length)/anno_stride + 1
        if an['Name'].lower() == 'seiz':
            segments += windows
            total_seiz_dur += an['Duration']
        segments_total+=windows

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5'
f = dc.File(file_name, 'r')

annos = f.get_children(object_type=dc.Annotations, get_obj = True)
anno_names = dict()
seiz_priority = ['mysz', 'absz', 'spsz', 'tnsz', 'tcsz', 'cpsz', 'gnsz', 'fnsz']
total_seiz_dur = 0
for anno in annos:
    for an in anno:
        if an['Name'] in seiz_priority:
            total_seiz_dur+=an['Duration']
        if an['Name'] not in anno_names.keys():
            anno_names[an['Name']] = [an['Duration']]
        else:
            anno_names[an['Name']].append(an['Duration'])

n_stat_epi = 0
for name in anno_names.keys():
    print(name)
    print('Mean', np.mean(anno_names[name]))
    print('Min', np.min(anno_names[name]))
    print('Max', np.max(anno_names[name]), '\n')
    anno = np.array(anno_names[name])
    n_stat_epi += len(anno[anno>5*60])
    print('N. status epilepticus', len(anno[anno>5*60]))

seiz_dur = []
for name in anno_names.keys():
    if name != 'bckg':
        seiz_dur = np.append(seiz_dur, anno_names[name])

plt.hist(seiz_dur, bins = 100, range = (0,300))
plt.show()

subj_recs = []
rec_dur = np.array([])
time_between = []
for protocol in f.keys():
    prot = f[protocol]
    for subj in prot.keys():
        subject = prot[subj]
        subj_recs.append(len(subject.attrs['time']))
        for rec in subject.attrs['time']['Name']:
            if 's027_t000' in rec:
                print(protocol)
                print(subj)
                print(rec)
        rec_dur = np.append(rec_dur, subject.attrs['time']['Duration'])
        time_temp = max(subject.attrs['time']['Start'])-min(subject.attrs['time']['Start'])
        time_between.append(time_temp)




pd.to_datetime(max(time_between), unit='s')