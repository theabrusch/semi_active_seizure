from dataapi import data_collection as dc
import numpy as np
import pandas as pd

select_channels = False
rereference = True

F = dc.File('data/hdf5/boston_scalp.hdf5', 'r+')
record = F['train']['chb01/03']
sig = record['EEG']

channels = dict()

for split in F:
    for subj in F[split]:
        for rec in F[split][subj]:
            record = F[split][subj][rec]
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
    if channels[ch] == 656:
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
    chNames = record['EEG'].attrs['chNames']
    ch_rec = []
    for ch in chNames:
        if ch in channels_common:
            ch_rec.append(ch)
    if len(ch_rec) == 18:
        common_channel_recs.append(rec)
    else:
        mis_channel_recs.append(rec)

records = F.get_children(object_type = dc.Record, get_obj=False)

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




F.close()

                