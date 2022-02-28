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
end = 27*signal.fs
anno_start_line = annos[1]['Start']-record.start_time
anno_end_line = anno_start_line + annos[1]['Duration']
scale_sig = signal[start:end,:]

fig, ax = plt.subplots(nrows = 20, figsize = (8,5), sharex = 'row',
                        sharey = True, gridspec_kw = {'hspace': 0})
for ch in range(20):
    channel = signal[start:end, ch]
    time = np.linspace(start/signal.fs, end/signal.fs, len(channel))
    #time = np.linspace(0,2, len(channel))
    ax[ch].plot(time, channel, linewidth = 0.7)

    ax[ch].set_xlim([min(time), max(time)])
    ax[ch].set_ylim([-500, 500])
    
    ax[ch].spines['right'].set_visible(False)
    ax[ch].spines['left'].set_visible(False)
    ax[ch].spines['top'].set_visible(False)
    #
    if not ch == 19:
        ax[ch].get_xaxis().set_ticks([])
        ax[ch].spines['bottom'].set_visible(False)
    else:
        ax[ch].spines['bottom'].set_visible(False)
        ax[ch].set_xlabel('Time (s)', fontsize = 14)
        ax[ch].tick_params(axis = 'x', labelsize = 12)
    #if ch == 10:
    #    text_string = 'Y-range: [' + str(np.min(scale_sig)) + ', ' + str(np.max(scale_sig))
    #    plt.subplot_adjust(right = 2)
    #    ax[ch].text(29, 0, text_string)
    ax[ch].get_yaxis().set_ticks([])
    #ax[ch].tick_params(axis = 'y', labelsize = 14)


    #ax[ch].set_ylabel(signal.attrs['chNames'][ch], rotation = 0, labelpad = 30, fontsize = 14)
    #ax[ch].axvspan(anno_start_line, anno_end_line, np.min(signal), np.max(signal), facecolor = 'bisque')

plt.show()

fig, ax = plt.subplots(figsize=(8,5))

plt.rcParams["text.usetex"] = False

subsig = signal[start:end, :]
im = ax.imshow(subsig.T, aspect = 'auto', extent = [25,27, 20, 0])
ax.tick_params(axis = 'x', labelsize = 14)
ax.set_xlabel('Time (s)', fontsize = 14)
ax.set_ylabel('Channel number', fontsize = 14)
ax.get_xaxis().set_ticks(np.arange(25,27.5,0.5))
ax.tick_params(axis = 'y', labelsize = 14)
ax.get_yaxis().set_ticks(np.arange(0,20,2).astype(int))
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Amplitude ' +r'($\mu V$)', fontsize = 14, rotation=270, labelpad = 20)
cbar.ax.tick_params(axis = 'y', labelsize = 14)

ax.set_yticks([])
ax.set_xticks([])

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

#eventbased scoring
fig, ax =plt.subplots(nrows = 2, figsize=(10,4))
time = np.arange(0, len(signal)/250, step=1/250) 
# reference
ax[0].plot(time, signal[:,0].T)
ax[0].vlines([2,4,6,8,10,12,14, 16, 18], ymin=-100, ymax=150, color ='black', linewidth=1)
ax[0].axvspan(5, 15,-100, 150, facecolor = 'bisque')
#ax.axvspan(660, 700,-100, 150, facecolor = 'bisque')
ax[0].set_xlim([0,20])
ax[0].set_ylim([-10,30])
ax[0].set_yticks([])
ax[0].set_xticks([])
ax[0].set_title('Reference annotation', fontsize = 14)
# hypothesis
ax[1].plot(time, signal[:,0].T)
ax[1].vlines([2,4,6,8,10,12,14, 16, 18], ymin=-100, ymax=150, color ='black', linewidth=1)
ax[1].axvspan(6, 10,-100, 150, facecolor = 'bisque')
ax[1].axvspan(14, 18,-100, 150, facecolor = 'bisque')
ax[1].axvspan(2, 4,-100, 150, facecolor = 'bisque')
ax[1].set_title('Hypothesis annotation', fontsize = 14)
#ax.axvspan(660, 700,-100, 150, facecolor = 'bisque')
ax[1].set_xlim([0,20])
ax[1].set_ylim([-10,30])
ax[1].set_yticks([])
ax[1].set_xticks(list(range(21)))
ax[1].tick_params('x', labelsize = 12)
ax[1].set_xlabel('Time (s)', fontsize = 14)
plt.subplots_adjust(bottom = 0.2, hspace = 0.4)
plt.show()

#[610,620,630,640,650,660,670, 680, 690]
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


## Plot postprocessing

#eventbased scoring
fig, ax =plt.subplots(nrows = 4, figsize=(20,5))
time = np.arange(0, 50, step=1) 
# reference
ax_n = 0
ax[ax_n].set_xticks(np.arange(0, 51, step=2), labels=[])
ax[ax_n].set_yticks([])
ax[ax_n].axvspan(0, 10,-1, 2, facecolor = 'lightsalmon', label = 'Background (p<0.50)')
ax[ax_n].text(4,0.5,'0.11', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(10, 22,-1, 2, facecolor = 'lightgreen', label = 'Seizure (p>0.50)')
ax[ax_n].text(15,0.5,'0.80', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(22, 24,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(22.2,0.5,'0.18', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(24, 28,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(25,0.5,'0.80', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(28, 32,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(29,0.5,'0.39', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(32, 36,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(33.3,0.5,'0.59', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(36, 42,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(38,0.5,'0.39', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(42, 44,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(42.2,0.5,'0.90', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(44, 50,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(45.5,0.5,'0.39', fontsize = 10, fontweight = 'semibold')

ax[ax_n].set_ylim([0,1])
ax[ax_n].set_xlim([0,50])

# reference
ax_n = 1
ax[ax_n].set_xticks(np.arange(0, 51, step=2), labels=[])
ax[ax_n].set_yticks([])
ax[ax_n].axvspan(0, 10,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(4,0.5,'0.11', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(10, 22,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(15,0.5,'0.80', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(22, 24,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(22.2,0.5,'0.18', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(24, 28,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(25,0.5,'0.80', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(28, 42,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(34,0.5,'0.39', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(42, 44,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(42.2,0.5,'0.90', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(44, 50,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(45.5,0.5,'0.39', fontsize = 10, fontweight = 'semibold')

ax[ax_n].set_ylim([0,1])
ax[ax_n].set_xlim([0,50])

# reference
ax_n = 2
ax[ax_n].set_xticks(np.arange(0, 51, step=2), labels=[])
ax[ax_n].set_yticks([])
ax[ax_n].axvspan(0, 10,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(4,0.5,'0.11', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(10, 22,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(15,0.5,'0.80', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(22, 24,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(22.2,0.5,'0.18', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(24, 28,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(25,0.5,'0.80', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(28, 50,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(38,0.5,'0.39', fontsize = 10, fontweight = 'semibold')

ax[ax_n].set_ylim([0,1])
ax[ax_n].set_xlim([0,50])

# reference
ax_n = 3
ax[ax_n].set_yticks([])
ax[ax_n].axvspan(0, 10,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(4,0.5,'0.11', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(10, 28,-1, 2, facecolor = 'lightgreen')
ax[ax_n].text(18,0.5,'0.80', fontsize = 10, fontweight = 'semibold')
ax[ax_n].axvspan(28, 50,-1, 2, facecolor = 'lightsalmon')
ax[ax_n].text(38,0.5,'0.39', fontsize = 10, fontweight = 'semibold')

ax[ax_n].set_xticks(np.arange(0, 51, step=2), fontsize = 12)
ax[ax_n].set_xlabel('Time (s)', fontsize = 14)

ax[ax_n].set_ylim([0,1])
ax[ax_n].set_xlim([0,50])

fig.subplots_adjust(bottom = 0.2, hspace = 1)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc = 'lower center', bbox_to_anchor=(0.5, 0), borderaxespad=0.5,
            bbox_transform = plt.gcf().transFigure, ncol = 4, fontsize = 14)

plt.show()


