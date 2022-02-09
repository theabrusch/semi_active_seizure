import mne
import re
import pandas as pd
import numpy as np
from dataapi import data_collection as dc
import matplotlib.pyplot as plt
import pyedflib

#TCSZ

seizure_types = dict()
file = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')
records = file.get_children(dc.Record, get_obj = False)

for rec in records:
    record = file[rec]
    for anno in record['Annotations']:
        if anno['Name'] in ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']:
            if anno['Name'] in seizure_types.keys():
                if rec not in seizure_types[anno['Name']]:
                    seizure_types[anno['Name']].append(rec)
            else:
                seizure_types[anno['Name']] = [rec]

record = file[seizure_types['tcsz'][4]]
signal = record['TCP']
annos = record['Annotations']
#start = int((annos[1]['Start']-record.start_time - 20)*signal.fs)
start = 25*signal.fs
end = 35*signal.fs
anno_start_line = annos[1]['Start']-record.start_time
anno_end_line = anno_start_line + annos[1]['Duration']
scale_sig = signal[0:40*signal.fs,:]

fig, ax = plt.subplots(nrows = 10, figsize = (15,5), sharex = 'row', gridspec_kw = {'hspace': 0})

for ch in range(10):
    channel = signal[start:end, ch]
    time = np.linspace(start/signal.fs, end/signal.fs, len(channel))
    ax[ch].plot(time, channel, linewidth = 0.7)

    ax[ch].set_xlim([min(time), max(time)])
    ax[ch].set_ylim([np.min(scale_sig), np.max(scale_sig)])
    
    ax[ch].spines['right'].set_visible(False)
    ax[ch].spines['left'].set_visible(False)
    ax[ch].spines['top'].set_visible(False)
    ax[ch].spines['bottom'].set_visible(False)
    if not ch == 9:
        ax[ch].get_xaxis().set_ticks([])
    else:
        ax[ch].set_xlabel('Time (s)', fontsize = 14)
        ax[ch].tick_params(axis = 'x', labelsize = 14)
    ax[ch].get_yaxis().set_ticks([])

    ax[ch].set_ylabel(signal.attrs['chNames'][ch], rotation = 0, labelpad = 30, fontsize = 14)
    ax[ch].axvspan(anno_start_line, anno_end_line, np.min(signal), np.max(signal), facecolor = 'bisque')

plt.show()

#CPSZ
record = file[seizure_types['cpsz'][-1]]
signal = record['TCP']
# read labels per channel
annos = open('/Users/theabrusch/Desktop/Speciale_data/edf/train/01_tcp_ar/134/00013407/s001_2015_09_28/00013407_s001_t013.lbl_bi', 'r')
montage = dict()
translate_montage = dict()
for line in annos:
    if 'montage' in line:
        number = int(line.split(' ')[2].strip(','))
        channel = line.split(' ')[3].strip(':')
        montage[channel] = dict()
        montage[channel]['number'] = number
        translate_montage[number] = channel
    elif 'symbols' in line:
        anno_dict = eval(line.split('=')[1])
    elif 'label' in line:
        label_string = line.split('=')[1].split(',')
        ch_number = int(label_string[4])
        channel = translate_montage[ch_number]
        start = eval(label_string[2])
        end = eval(label_string[3])
        one_hot_label = np.array(eval(','.join(label_string[5:]).split('}')[0])).astype(int)
        label = anno_dict[np.where(one_hot_label==1)[0][0]]
        if label == 'seiz':
            montage[channel]['anno'] = (start, end, label)



start = 25*signal.fs#int((annos[1]['Start']-record.start_time - 10)*signal.fs)
annos = record['Annotations']
end = 35*signal.fs
anno_start_line = annos[1]['Start']-record.start_time
anno_end_line = anno_start_line + annos[1]['Duration']

fig, ax = plt.subplots(nrows = 10, figsize = (15,5), sharex = 'row', gridspec_kw = {'hspace': 0})

for ch in range(10, 20):
    channel = signal[start:end, ch]
    channel_name = signal.attrs['chNames'][ch]

    ch = ch-10
    if 'anno' in montage[channel_name].keys():
        anno_start = montage[channel_name]['anno'][0]
        anno_end = montage[channel_name]['anno'][1]
        ax[ch].axvspan(anno_start, anno_end,-55, 55, facecolor = 'bisque')

    ax[ch].set_ylabel(channel_name, rotation = 0, labelpad = 30, fontsize = 14)
    time = np.linspace(start/signal.fs, end/signal.fs, len(channel))
    ax[ch].plot(time, channel, linewidth = 0.7)

    ax[ch].set_xlim([min(time), max(time)])
    ax[ch].set_ylim([-55, 55])
    
    ax[ch].spines['right'].set_visible(False)
    ax[ch].spines['left'].set_visible(False)
    ax[ch].spines['top'].set_visible(False)
    ax[ch].spines['bottom'].set_visible(False)
    if not ch == 9:
        ax[ch].get_xaxis().set_ticks([])
    else:
        ax[ch].set_xlabel('Time (s)', fontsize = 14)
        ax[ch].tick_params(axis = 'x', labelsize = 14)
    ax[ch].get_yaxis().set_ticks([])
plt.show()

#FNSZ

record = file[seizure_types['fnsz'][0]]
signal = record['TCP']
# read labels per channel
annos = open('/Users/theabrusch/Desktop/Speciale_data/edf/train/01_tcp_ar/131/00013145/s006_2015_09_03/00013145_s006_t007.lbl_bi', 'r')
montage = dict()
translate_montage = dict()
for line in annos:
    if 'montage' in line:
        number = int(line.split(' ')[2].strip(','))
        channel = line.split(' ')[3].strip(':')
        montage[channel] = dict()
        montage[channel]['number'] = number
        translate_montage[number] = channel
    elif 'symbols' in line:
        anno_dict = eval(line.split('=')[1])
    elif 'label' in line:
        label_string = line.split('=')[1].split(',')
        ch_number = int(label_string[4])
        channel = translate_montage[ch_number]
        start = eval(label_string[2])
        end = eval(label_string[3])
        one_hot_label = np.array(eval(','.join(label_string[5:]).split('}')[0])).astype(int)
        label = anno_dict[np.where(one_hot_label==1)[0][0]]
        if label == 'seiz':
            montage[channel]['anno'] = (start, end, label)



start = 280*signal.fs#int((annos[1]['Start']-record.start_time - 10)*signal.fs)
annos = record['Annotations']
end = 290*signal.fs
anno_start_line = annos[1]['Start']-record.start_time
anno_end_line = anno_start_line + annos[1]['Duration']
scale_sig = signal[start:end, 2:]

fig, ax = plt.subplots(nrows = 20, figsize = (15,5), sharex = 'row', gridspec_kw = {'hspace': 0})

for ch in range(20):
    channel = signal[start:end, ch]
    channel_name = signal.attrs['chNames'][ch]

    ch = ch
    if 'anno' in montage[channel_name].keys():
        anno_start = montage[channel_name]['anno'][0]
        anno_end = montage[channel_name]['anno'][1]
        ax[ch].axvspan(anno_start, anno_end,np.min(scale_sig), np.max(scale_sig), facecolor = 'bisque')

    ax[ch].set_ylabel(channel_name, rotation = 0, labelpad = 30, fontsize = 14)
    time = np.linspace(start/signal.fs, end/signal.fs, len(channel))
    ax[ch].plot(time, channel, linewidth = 0.7)

    ax[ch].set_xlim([min(time), max(time)])
    ax[ch].set_ylim([np.min(scale_sig), np.max(scale_sig)])
    
    ax[ch].spines['right'].set_visible(False)
    ax[ch].spines['left'].set_visible(False)
    ax[ch].spines['top'].set_visible(False)
    ax[ch].spines['bottom'].set_visible(False)
    if not ch == 19:
        ax[ch].get_xaxis().set_ticks([])
    else:
        ax[ch].set_xlabel('Time (s)', fontsize = 14)
        ax[ch].tick_params(axis = 'x', labelsize = 14)
    ax[ch].get_yaxis().set_ticks([])
plt.show()
