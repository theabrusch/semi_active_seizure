import numpy as np
import pyedflib
from dataapi import data_collection as dc
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5'
f = dc.File(file_name, 'r')

annos = f.get_children(object_type=dc.Annotations, get_obj = True)
anno_names = dict()

for anno in annos:
    for an in anno:
        if an['Name'] not in anno_names.keys():
            anno_names[an['Name']] = [an['Duration']]
        else:
            anno_names[an['Name']].append(an['Duration'])

for name in anno_names.keys():
    print(name)
    print('Mean', np.mean(anno_names[name]))
    print('Min', np.min(anno_names[name]))
    print('Max', np.max(anno_names[name]), '\n')

seiz_subjs = []
seizures = 0
fnsz = 0
seiz_subjs_all = []
seiz_dur = 0
total_dur = 0
stats = dict()
subjects = f.get_children(dc.Subject, get_obj=False)
for subj in subjects:
    subject = f[subj]
    stats[subj] = dict()
    stats[subj]['total dur'] = 0
    stats[subj]['seiz dur'] = []
    seiz_subj = False
    subj_exclude = False
    for rec in subject.keys():
        record = subject[rec]
        for anno in record['Annotations']:
            if anno['Name'] in ['gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']:
                seiz_subj = True
                seizures += 1
                stats[subj]['seiz dur'].append(anno['Duration'])
            elif anno['Name'] == 'fnsz':
                fnsz += 1
                subj_exclude = True
    if not subj_exclude and seiz_subj:
        seiz_subjs.append(subj)
    if seiz_subj:
        seiz_subjs_all.append(subj)

    stats[subj]['total dur'] += record.duration

seiz_per_subj = []
for subj in stats.keys():
    if len(stats[subj]['seiz dur']) > 0:
        seiz_per_subj.append(len(stats[subj]['seiz dur']))