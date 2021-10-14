import mne
import re
import pandas as pd
import numpy as np
seizures = open('seizures.txt', 'r')
lines = seizures.readlines()
files = dict()

for line in lines:
    if 'SZ' in line.strip():
        seiz = line.strip()
        files[seiz] = dict()
        files[seiz] = []
    else:
        parts = line.strip().split(', ')
        file_name = parts[0].split('/')[-1].split('.')[0]
        edf_path = 'data/dataplot/' + file_name + '.edf'
        files[seiz].append([edf_path, float(parts[1]), float(parts[2])])

CH_LST_1020 = ['T5', 'T6', 'T4', 'T3', 'F7', 'F8', 'FP1', 'FP2', 'F4',
              'F3', 'CZ', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
montage = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'C3-CZ', 'FP1-F3',
            'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
            'C4-T4', 'CZ-C4', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
anodes = []
catodes = []
for channel in montage:
    parts = channel.split('-')
    anodes.append(parts[0])
    catodes.append(parts[1])

file_to_plot = files['FNSZ'][1]

mne_file = mne.io.read_raw_edf(file_to_plot[0], preload=True)

channels_to_use = []
channels_dict = dict()

for ch in mne_file.info.ch_names:
    channel = ch.split(' ')[-1].split('-')[0]
    if channel in CH_LST_1020:
        channels_to_use.append(ch)
        channels_dict[channel] = ch
channels_df = pd.DataFrame(channels_dict, index = [0])
rec_anodes = list(channels_df[anodes].values[0])
rec_catodes = list(channels_df[catodes].values[0])

mne_file = mne_file.pick_channels(channels_to_use)
bip_ref =  mne.set_bipolar_reference(mne_file, anode=rec_anodes,
                                        cathode=rec_catodes, ch_name = montage)

dur = file_to_plot[2] - file_to_plot[1]
order = list(range(0,len(channels_to_use), 2))
order = np.append(order, list(range(1,len(channels_to_use), 2))) 
mne.viz.plot_raw(mne_file, duration = 60, start = 1410, order = order,
                 n_channels = 17, show_scrollbars=False, scalings = dict(eeg = 'auto'))
mne.viz.plot_raw(bip_ref, duration = 60, start = 1410,
                 n_channels = 20, show_scrollbars=False, scalings = dict(eeg = 'auto'))