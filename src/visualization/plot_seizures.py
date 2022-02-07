import mne
import re
import pandas as pd
import numpy as np
from dataapi import data_collection as dc
import matplotlib.pyplot as plt

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
start = int((annos[1]['Start']-record.start_time - 10)*signal.fs)
end = int((annos[1]['Start']-record.start_time + 30)*signal.fs)
anno_start_line = annos[1]['Start']-record.start_time
anno_end_line = anno_start_line + annos[1]['Duration']

fig, ax = plt.subplots(nrows = 10, sharex = 'row', gridspec_kw = {'hspace': 0})

for ch in range(10):
    channel = signal[start:end, ch]
    time = np.linspace(start/signal.fs, end/signal.fs, len(channel))
    ax[ch].plot(time, channel, linewidth = 0.7)

    ax[ch].set_xlim([min(time), max(time)])
    ax[ch].set_ylim([min(channel), max(channel)])
    
    ax[ch].spines['right'].set_visible(False)
    ax[ch].spines['left'].set_visible(False)
    ax[ch].spines['top'].set_visible(False)
    ax[ch].spines['bottom'].set_visible(False)
    if not ch == 9:
        ax[ch].get_xaxis().set_ticks([])
    else:
        ax[ch].set_xlabel('Time (s)', fontsize = 12)
        #ax[ch].set_xticks(ticks = [280,290,300,310,320,330,340], fontsize = 12)
    ax[ch].get_yaxis().set_ticks([])

    ax[ch].set_ylabel(signal.attrs['chNames'][ch], rotation = 0, labelpad = 20, fontsize = 12)
    ax[ch].axvspan(anno_start_line, anno_end_line, min(channel), max(channel), facecolor = 'bisque')

plt.show()

