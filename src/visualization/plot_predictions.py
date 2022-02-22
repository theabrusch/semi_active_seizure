import pandas as pd
import pickle
from src.analysis import analysis
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
from dataapi import data_collection as dc

def plot_predictions(rec_name, rec_pred, anno_pred, channels, time_start, time_end, y_min, y_max):
    ## plot records
    f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')
    record = f[rec_name]
    annos = record['Annotations']
    signal = record['TCP']

    anno_start_line = []
    anno_end_line = []
    for anno in annos:  
        if anno['Name'] in ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']:
            start = anno['Start']-record.start_time
            anno_start_line.append(start)
            anno_end_line.append(start + anno['Duration'])

    fig, ax = plt.subplots(nrows = 9, figsize = (15,5), sharex = 'row', gridspec_kw = {'hspace': 0})
    for i in range(len(channels)):
        ch = channels[i]
        channel = signal[:, ch]
        channel_name = signal.attrs['chNames'][ch]

        ch = i
        for j,an in enumerate(anno_start_line):
            ax[ch].axvspan(an, anno_end_line[j], np.min(signal), np.max(signal), facecolor = 'bisque')

        ax[ch].set_ylabel(channel_name, rotation = 0, labelpad = 30, fontsize = 14)
        time = np.linspace(0, record.duration, len(channel))
        ax[ch].plot(time, channel, linewidth = 0.7)

        ax[ch].set_xlim([time_start, time_end])
        ax[ch].set_ylim([y_min, y_max])
        
        ax[ch].spines['right'].set_visible(False)
        ax[ch].spines['left'].set_visible(False)
        ax[ch].spines['top'].set_visible(False)
        ax[ch].spines['bottom'].set_visible(False)
        ax[ch].get_xaxis().set_ticks([])
        ax[ch].get_yaxis().set_ticks([])

    ax[5].get_xaxis().set_ticks([])
    ax[5].get_yaxis().set_ticks([])

    ax[5].spines['right'].set_visible(False)
    ax[5].spines['left'].set_visible(False)
    ax[5].spines['top'].set_visible(False)
    ax[5].spines['bottom'].set_visible(False)

    time_pred = np.arange(0, len(rec_pred)*2, 2)

    ax[6].plot(time_pred, rec_pred, color = 'black')
    ax[6].set_xlim([time_start, time_end])
    ax[6].hlines(0.7, time_start, time_end, linestyles ='dashed')
    ax[6].set_ylim([-0.2,1.1])
    ax[6].set_ylabel('Seiz. prob.', fontsize = 14)
    ax[6].get_xaxis().set_ticks([])


    ax[7].get_xaxis().set_ticks([])
    ax[7].get_yaxis().set_ticks([])

    ax[7].spines['right'].set_visible(False)
    ax[7].spines['left'].set_visible(False)
    ax[7].spines['top'].set_visible(False)
    ax[7].spines['bottom'].set_visible(False)
    bin_pred = np.zeros(len(time_pred))

    for anno in anno_pred:
        if anno['Name'] == 1:
            st = int(anno['Start']/2)
            end = int((anno['Start'] + anno['Duration'])/2)
            bin_pred[st:end] = 1

    ax[8].plot(time_pred, bin_pred, color = 'black')
    ax[8].set_yticks([0, 1], ['bckg', 'seiz'], fontsize = 12)
    ax[8].set_xlabel('Time (s)', fontsize = 14)
    #ax[8].set_ylabel('Pred.', fontsize = 14)
    ax[8].set_xlim([time_start, time_end])


    ax[8].tick_params('x', labelsize=12)
    ax[8].tick_params('y', labelsize=12)

    plt.tight_layout()
    f.close()
    return fig