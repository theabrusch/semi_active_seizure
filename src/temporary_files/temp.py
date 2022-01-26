import numpy as np
import pyedflib
from dataapi import data_collection as dc
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists
import scipy

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


seiz_all = 0
fnsz = 0
seiz_subjs_all = []
incl_subj = []
incl_seiz = 0
seiz_dur = 0
fnsz = 0
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
    seizures = 0 
    for rec in subject.keys():
        record = subject[rec]
        seiz_rec = False
        stats[subj][rec] = []
        for anno in record['Annotations']:
            stats[subj][rec].append(anno['Name'])
            if anno['Name'] in ['gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']:
                seiz_subj = True
                seizures += 1
                stats[subj]['seizure types'].append(anno['Name'])
                stats[subj]['seiz dur'].append(anno['Duration'])
            if anno['Name'] == 'fnsz':
                subj_exclude = True
                fnsz+=1
    if seiz_subj:
        seiz_subjs_all.append(subj)
        seiz_all += seizures
    if seiz_subj and not subj_exclude:
        incl_subj.append(subj)
        incl_seiz += seizures
    #stats[subj]['total dur'] += record.duration

subjects = 0
for subj in stats:
    seiz_types, counts = np.unique(stats[subj]['seizure types'], return_counts = True)
    if len(seiz_types) > 1:
        print(seiz_types, counts)
        subjects += 1


subjects = []
mult = 0
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


test_nofnsz =  ['/train/00008345', '/train/00008615', '/train/00004456', '/train/00000006', '/train/00008029', '/train/00012046', '/train/00011999', '/train/00007937', '/train/00002380', '/train/00010421', '/train/00000630', '/train/00010088', '/train/00008018', '/train/00010639', '/train/00008628', '/train/00005533', '/train/00000184', '/train/00002348', '/train/00010364', '/train/00005575', '/train/00011455', '/train/00011596', '/train/00013336', '/train/00011981', '/train/00007795', '/train/00001482', '/train/00010301', '/train/00005476', '/train/00005526', '/train/00012786', '/train/00000762', '/train/00004774', '/test/00001981', '/train/00009097', '/train/00007279', '/train/00000289', '/train/00009734', '/train/00007296', '/train/00011081', '/train/00005103', '/train/00004512', '/train/00010461', '/train/00012615', '/train/00009880', '/train/00001317', '/train/00007828', '/train/00001944', '/train/00009902', '/train/00002235', '/train/00001851', '/train/00002271', '/train/00009762', '/train/00009245']
test_full= ['/train/00007835', '/train/00001843', '/train/00012759', '/test/00001027', '/train/00007235', '/train/00006103', '/train/00001052', '/train/00007793', '/train/00006452', '/train/00008345', '/train/00008615', '/train/00004456', '/train/00000006', '/train/00008029', '/train/00012046', '/train/00011999', '/train/00007937', '/train/00002380', '/train/00010421', '/train/00000630', '/train/00010088', '/train/00008018', '/train/00010639', '/train/00008628', '/train/00005533', '/train/00000184', '/train/00002348', '/train/00010364', '/train/00005575', '/train/00011455', '/train/00011596', '/train/00013336', '/train/00011981', '/train/00007795', '/train/00001482', '/train/00010301', '/train/00005476', '/train/00005526', '/train/00012786', '/train/00000762', '/train/00004774', '/test/00001981', '/train/00009097', '/train/00007279', '/train/00000289', '/train/00009734', '/train/00007296', '/train/00011081', '/train/00005103', '/train/00004512', '/train/00010461', '/train/00012615', '/train/00009880', '/train/00001317', '/train/00007828', '/train/00001944', '/train/00009902', '/train/00002235', '/train/00001851', '/train/00002271', '/train/00009762', '/train/00009245']

same = 0
for subj in test_nofnsz:
    if subj in test_full:
        same+=1


sens= np.array([65.90,48.40,46.89,21.77,16.64])
spec = np.array([92.27,97.67,93.78,96.06,96.31])
acc = np.array([91.37,94.41,89.36,90.46,93.11])


