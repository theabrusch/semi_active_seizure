import numpy as np
import pyedflib
from dataapi import data_collection as dc
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/boston_scalp.hdf5'
f = dc.File(file_name, 'r')

annos = f.get_children(object_type=dc.Annotations, get_obj = True)
window_length = 2
anno_stride = 1/256
segments = 0
segments_total = 0
for anno in annos:
    for an in anno:
        windows = (an['Duration']-window_length)/anno_stride + 1
        if an['Name'].lower() == 'seiz':
            segments += windows
        segments_total+=windows

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

seiz_subjs = dict()
seizures = 0
fnsz = 0
seiz_subjs_all = []
seiz_dur = 0
total_dur = 0
stats = dict()
seiz_priority = ['mysz', 'absz', 'spsz', 'tnsz', 'tcsz', 'cpsz', 'gnsz', 'fnsz']
subjects = f.get_children(dc.Subject, get_obj=False)
for subj in subjects:
    subject = f[subj]
    stats[subj] = dict()
    stats[subj]['total dur'] = 0
    stats[subj]['seiz dur'] = []
    stats[subj]['seizure types'] = []
    seiz_subj = False
    subj_exclude = False
    for rec in subject.keys():
        record = subject[rec]
        for anno in record['Annotations']:
            if anno['Name'] in ['fnsz','gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']:
                seiz_subj = True
                seizures += 1
                stats[subj]['seizure types'].append(anno['Name'])
                stats[subj]['seiz dur'].append(anno['Duration'])
    if seiz_subj:
        seiz_subjs_all.append(subj)
        pri_seiz = [seiz for seiz in seiz_priority if seiz in stats[subj]['seizure types']][0]
        if pri_seiz in seiz_subjs.keys():
            seiz_subjs[pri_seiz].append(subj)
        else:
            seiz_subjs[pri_seiz] = [subj]
    #stats[subj]['total dur'] += record.duration

for subj in stats.keys():
    if 'seiz dur' not in stats[subj].keys():
        print(subj) 

seiz_per_subj = []
seiz_types_per_subj = []
for subj in stats.keys():
    if len(stats[subj]['seiz dur']) > 0:
        seiz_per_subj.append(len(stats[subj]['seiz dur']))
    unique_seizures, counts = np.unique(stats[subj]['seizure types'], return_counts = True)
    if len(unique_seizures) > 0:
        seiz_types_per_subj.append(len(unique_seizures))
        if len(unique_seizures) > 1:
            print(unique_seizures, counts)


seizures = 0
fnsz = 0
seiz_subjs_all = []
seiz_dur = 0
total_dur = 0
stats = dict()
seiz_priority = ['mysz', 'absz', 'spsz', 'tnsz', 'tcsz', 'cpsz', 'gnsz', 'fnsz']
subjects = f.get_children(dc.Subject, get_obj=False)
for subj in subjects:
    subject = f[subj]
    stats[subj] = dict()
    stats[subj]['seizure types'] = []
    stats[subj]['seiz dur'] = []
    seiz_subj = False
    subj_exclude = False
    for rec in subject.keys():
        record = subject[rec]
        seiz_rec = False
        stats[subj][rec] = []
        for anno in record['Annotations']:
            stats[subj][rec].append(anno['Name'])
            if anno['Name'] in ['fnsz','gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']:
                seiz_subj = True
                seizures += 1
                stats[subj]['seizure types'].append(anno['Name'])
                stats[subj]['seiz dur'].append(anno['Duration'])
    if seiz_subj:
        seiz_subjs_all.append(subj)
    #stats[subj]['total dur'] += record.duration
subjects = []
for subj in seiz_subjs_all:
    for key in stats[subj].keys():
        if key != 'seizure types' and key != 'seiz dur':
            annos, counts = np.unique(stats[subj][key], return_counts = True)
            if 'bckg' in annos and len(annos) > 2:
                if subj not in subjects:
                    subjects.append(subj)
                print(subj, rec)
                print(annos, counts)
            elif 'bckg' not in annos and len(annos) > 1:
                if subj not in subjects:
                    subjects.append(subj)
                print(subj, rec)
                print(annos, counts)