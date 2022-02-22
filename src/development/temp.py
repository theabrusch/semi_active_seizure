import numpy as np
import pyedflib
from dataapi import data_collection as dc
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists
import scipy
import pandas as pd

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/boston_scalp_newnew.hdf5'
f = dc.File(file_name, 'r+')

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

sens = np.array([0.8171, 0.7568, 0.5453, 0.3070, 0.4523])
spec = np.array([0.7672, 0.9025, 0.9019, 0.9262, 0.8637])
prec = np.array([0.11, 0.3385, 0.3667, 0.2535, 0.1113])
f1 = np.array([19.39, 46.78, 43.85, 27.77, 17.86])
seiz = np.array([5407, 10511, 15858, 17128, 9616])
bckg = np.array([153508, 148033, 152268, 209861, 229772])
tp_temp = sens*seiz
tn_temp = bckg*spec
fn_temp = seiz-tp_temp
fp_temp = bckg-tn_temp
fn = sum(seiz-tp_temp)
fp = sum(bckg-tn_temp)
tp = sum(tp_temp)
tn = sum(tn_temp)
tot_sens = tp/(tp+fn)
tot_spec = tn/(tn+fp)



import numpy as np
sens = [70.46, 74.71, 48.77, 62.30, 95.04, 32.53, 90.24, 22.94, 97.87, 83.33, 90.10, 43.22, 12.13, 11.11, 44.68, 68.89, 85.03, 42.33, 79.19, 58.28, 93.33, 67.13, 60.84]
spec = [97.36, 97.72, 99.37, 91.86, 56.10, 78.68, 95.86, 93.68, 48.98, 96.59, 92.47, 72.95, 58.28, 66.56, 94.30, 81.68, 91.34, 97.92, 98.80, 88.41, 96.02, 96.55, 91.91]
acc = [97.29, 97.69, 99.22, 91.84, 56.25, 78.65, 95.85, 92.77, 49.04, 96.56, 92.46, 72.55, 58.07, 66.45, 93.61, 81.67, 91.31, 97.78, 98.75, 88.32, 96.02, 96.42, 91.53]


sens_7 = [90.24, 87.80, 87.80, 87.80]
spec_7 = [95.86, 97.80, 96.41, 96.52]
acc_7 = [95.85, 97.78, 96.40, 96.51]

sens_14 = [11.11, 10, 10, 5.56]
spec_14 = [66.56, 71.18, 88.65, 69.57]
acc_14 = [66.45, 71.06, 88.49, 69.46]

sens = [71.69,91.46,55.5,83.77,20.96,60,87.07,85.4,
        28.14, 80.46, 91.84,68.66, 27.11,51.53,98.58,
        50.74,15.66,74.17,37.5, 52.98, 83.81, 5.56]