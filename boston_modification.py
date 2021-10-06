from dataapi import data_collection as dc
import numpy as np
import pandas as pd

select_channels = False
rereference = False
create_new_file = True
create_bckg_anno = False

F = dc.File('data/hdf5/boston_scalp_mod.hdf5', 'r+')
record = F['train']['chb01/03']
sig = record['EEG']

channels = dict()

for split in F:
    for subj in F[split]:
        for rec in F[split][subj]:
            record = F[split][subj][rec]
            if 'EEG' in record.keys():
                chNames = record['EEG'].attrs['chNames']
                for ch in chNames:
                    if ch in channels.keys():
                        channels[ch] += 1
                    else:
                        channels[ch] = 1
channels_common = []
single_ref_ch1 = []
single_ref_ch2 = []


for ch in channels:
    if channels[ch] > 656:
        channels_common.append(ch)
        ch1 = ch.split('-')[0]
        ch2 = ch.split('-')[1]
        single_ref_ch1.append(ch1)
        single_ref_ch2.append(ch2)

records = F.get_children(object_type = dc.Record, get_obj=False)
common_channel_recs = []
mis_channel_recs = []

for rec in records:
    record = F[rec]
    if 'EEG' in record.keys():
        chNames = record['EEG'].attrs['chNames']
        ch_rec = []
        for ch in chNames:
            if ch in channels_common:
                ch_rec.append(ch)
        if len(ch_rec) == 18:
            common_channel_recs.append(rec)
        else:
            mis_channel_recs.append(rec)

if select_channels:
    for rec in records:
        if rec in common_channel_recs:
            record = F[rec]
            signal = record['EEG']
            chDF = pd.DataFrame(signal[()], columns = signal.attrs['chNames'])
            common_channel_df = chDF[channels_common]
            newsig = signal._append_to_record(common_channel_df.values, dset_name = 'EEG',
                                            new_ch_names = channels_common, inplace = True)
            newsig.attrs['chNames'] = channels_common
            F.flush()

if rereference:
    for rec in mis_channel_recs:
        record = F[rec]
        signal = record['EEG'] 
        channels_no_ref = [] 
        for ch in signal.attrs['chNames']:
            if ch.split('-')[0] == '01':
                channel = 'O1'
            else:
                channel = ch.split('-')[0]
            channels_no_ref.append(channel)
        chDF = pd.DataFrame(signal[()], columns = channels_no_ref)
        # Rereference
        bipolar_ref = chDF[single_ref_ch1].values - chDF[single_ref_ch2].values
        newsig = signal._append_to_record(bipolar_ref, dset_name = 'EEG',
                                          new_ch_names = channels_common, inplace = True)
        newsig.attrs['chNames'] = channels_common

if create_bckg_anno:
    records = F.get_children(dc.Record, get_obj=False)
    for rec in records:
        record = F[rec]
        annotations = record['Annotations']
        if len(record['Annotations']) > 0:
            print('hej')
        bckg = annotations._get_unlabeled_data(annotations[()], record.start_time, 
                                               record.start_time + record.duration,
                                               unlabled_name="bckg")
        annotations.append(bckg)
        annotations.remove_dublicates()

        F.flush()
F.close()

if create_new_file:
    F = dc.File('data/hdf5/boston_scalp_mod.hdf5', 'r')
    newF = dc.File('data/hdf5/boston_scalp_new.hdf5', 'a')
    new_train = newF.create_group('train')
    for subj in F['train'].keys():
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

            newF.flush()

    F.close()
    newF.close()

                