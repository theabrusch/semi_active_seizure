from dataapi import data_collection as dc
import numpy as np
import pickle
from src.data import train_val_split

create_temple_sub = False
create_boston_sub = False
create_boston_small = True

if create_temple_sub:
    print('Creating subsample from the Temple dataset.')
    with open('data/temple_seiz_seiz_subjs.pickle', 'rb') as fb:
        seiz_subjs = pickle.load(fb)

    sub_seiz = np.random.choice(seiz_subjs['seiz'], size = int(0.25*len(seiz_subjs['seiz'])))
    sub_non_seiz = np.random.choice(seiz_subjs['non seiz'], size = int(0.25*len(seiz_subjs['non seiz'])))
    sub_sample = np.append(sub_seiz, sub_non_seiz)

    F = dc.File('data/hdf5/temple_seiz.hdf5', 'r')
    subF = dc.File('data/hdf5/temple_seiz_sub.hdf5', 'a')

    new_train = subF.create_group('train')
    i = 1
    for subj in sub_sample:
        print('Copying subject', i, 'out of', len(sub_sample))
        i+=1
        subject = F['train'][subj]
        if subj not in new_train.keys():
            train_subj = new_train.create_subject(subj)
        else: 
            train_subj = new_train[subj]

        for att in subject.attrs.keys():
            train_subj.attrs[att] = subject.attrs[att]
        
        for rec in subject.keys():
            record = subject[rec]
            if rec not in train_subj.keys():
                train_record = train_subj.create_record(rec)
            else:
                train_record = train_subj[rec]
            for att in record.attrs.keys():
                train_record.attrs[att]= record.attrs[att]

            if 'TCP' in train_record.keys():
                newsig = train_record['TCP']
            else:
                newsig = train_record.create_signal('TCP', data = record['TCP'][()])

            for att in record['TCP'].attrs.keys():
                newsig.attrs[att] = record['TCP'].attrs[att]
            if 'Annotations' in train_record.keys():
                annotations = train_record['Annotations']
            else:
                annotations = train_record.create_annotations()
            if len(record['Annotations']) > 0:
                annotations.append(record['Annotations'][()])
                annotations.remove_dublicates()
            subF.flush()
    F.close()
    subF.close()


if create_boston_sub:
    F = dc.File('data/hdf5/boston_scalp_new.hdf5', 'r')
    subF = dc.File('data/hdf5/boston_scalp_sub.hdf5', 'a')

    new_train = subF.create_group('train')
    i = 0
    for subj in F['train'].keys():
        print('Copying subject', i, 'out of', len(F['train'].keys()))
        i+=1
        subject = F['train'][subj]
        if subj not in new_train.keys():
            train_subj = new_train.create_subject(subj)
        else: 
            train_subj = new_train[subj]

        for att in subject.attrs.keys():
            train_subj.attrs[att] = subject.attrs[att]
        
        #check for seizure records
        seiz_recs = train_val_split.get_seiz_recs(subject, seiz_names=['seiz'])
        rec_seiz = np.random.choice(seiz_recs['seiz'], size = int(0.25*len(seiz_recs['seiz'])))
        rec_non_seiz = np.random.choice(seiz_recs['non seiz'], size = int(0.25*len(seiz_recs['non seiz'])))
        rec_sample = np.append(rec_seiz, rec_non_seiz)

        for rec in rec_sample:
            record = subject[rec]
            if rec not in train_subj.keys():
                train_record = train_subj.create_record(rec)
            else:
                train_record = train_subj[rec]
            for att in record.attrs.keys():
                train_record.attrs[att]= record.attrs[att]

            if 'EEG' in train_record.keys():
                newsig = train_record['EEG']
            else:
                newsig = train_record.create_signal('EEG', data = record['EEG'][()])

            for att in record['EEG'].attrs.keys():
                newsig.attrs[att] = record['EEG'].attrs[att]
            if 'Annotations' in train_record.keys():
                annotations = train_record['Annotations']
            else:
                annotations = train_record.create_annotations()
            if len(record['Annotations']) > 0:
                annotations.append(record['Annotations'][()])
                annotations.remove_dublicates()
            subF.flush()
    F.close()
    subF.close()


if create_boston_small:
    F = dc.File('/Users/theabrusch/Desktop/Speciale_data/boston_scalp_new.hdf5', 'r')
    subF = dc.File('data/hdf5/boston_scalp_small.hdf5', 'w')

    new_train = subF.create_group('train')
    i = 0
    subjects = list(F['train'].keys())
    subjs_to_use = np.random.choice(subjects, 2)
    for subj in subjs_to_use:
        print('Copying subject', i, 'out of', len(subjs_to_use))
        i+=1
        subject = F['train'][subj]
        if subj not in new_train.keys():
            train_subj = new_train.create_subject(subj)
        else: 
            train_subj = new_train[subj]

        for att in subject.attrs.keys():
            train_subj.attrs[att] = subject.attrs[att]
        
        #check for seizure records
        seiz_recs = train_val_split.get_seiz_recs(subject, seiz_classes=['seiz'])
        if len(seiz_recs['seiz'])>1:
            rec_seiz = np.random.choice(seiz_recs['seiz'], size = 2)
        else:
            rec_seiz = np.random.choice(seiz_recs['seiz'], size = 1)

        rec_sample = rec_seiz

        for rec in rec_sample:
            record = subject[rec]
            if rec not in train_subj.keys():
                train_record = train_subj.create_record(rec)
            else:
                train_record = train_subj[rec]
            for att in record.attrs.keys():
                train_record.attrs[att]= record.attrs[att]

            if 'EEG' in train_record.keys():
                newsig = train_record['EEG']
            else:
                newsig = train_record.create_signal('EEG', data = record['EEG'][()])

            for att in record['EEG'].attrs.keys():
                newsig.attrs[att] = record['EEG'].attrs[att]
            if 'Annotations' in train_record.keys():
                annotations = train_record['Annotations']
            else:
                annotations = train_record.create_annotations()
            if len(record['Annotations']) > 0:
                annotations.append(record['Annotations'][()])
                annotations.remove_dublicates()
            subF.flush()
    F.close()
    subF.close()

