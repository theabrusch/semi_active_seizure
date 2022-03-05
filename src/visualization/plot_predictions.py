import pandas as pd
import pickle
from src.analysis import analysis
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
from dataapi import data_collection as dc

def plot_predictions(rec_name, rec_pred, anno_pred, channels, seiz_eval, time_start, time_end, y_min, y_max):
    ## plot records
    f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')
    record = f[rec_name]
    annos = record['Annotations']
    signal = record['TCP']
    seiz_types = ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']

    anno_start_line = []
    anno_end_line = []

    diff_seiz_start = []
    diff_seiz_end= []
    for anno in annos:  
        if anno['Name'] in seiz_eval:
            start = anno['Start']-record.start_time
            anno_start_line.append(start)
            anno_end_line.append(start + anno['Duration'])
        elif anno['Name'] in seiz_types:
            start = anno['Start']-record.start_time
            diff_seiz_start.append(start)
            diff_seiz_end.append(start + anno['Duration'])


    if len(channels) <= 5:
        height = 5
    elif len(channels) <= 10:
        height = 5
    else:
        height = 7.5

    fig, ax = plt.subplots(nrows = len(channels)+4, figsize = (15,height), sharex = 'row', gridspec_kw = {'hspace': 0})
    for i in range(len(channels)):
        ch = channels[i]
        channel = signal[:, ch]
        channel_name = signal.attrs['chNames'][ch]

        ch = i
        for j,an in enumerate(anno_start_line):
            ax[ch].axvspan(an, anno_end_line[j], np.min(signal), np.max(signal), facecolor = 'bisque')

        for j,an in enumerate(diff_seiz_start):
            ax[ch].axvspan(an, diff_seiz_end[j], np.min(signal), np.max(signal), facecolor = 'lavender')

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
    
    ax[len(channels)].get_xaxis().set_ticks([])
    ax[len(channels)].get_yaxis().set_ticks([])

    ax[len(channels)].spines['right'].set_visible(False)
    ax[len(channels)].spines['left'].set_visible(False)
    ax[len(channels)].spines['top'].set_visible(False)
    ax[len(channels)].spines['bottom'].set_visible(False)

    if not rec_pred is None:
        time_pred = np.arange(0, len(rec_pred)*2, 2)
        ax[len(channels)+1].plot(time_pred, rec_pred, color = 'black')
        ax[len(channels)+1].set_xlim([time_start, time_end])
        ax[len(channels)+1].hlines(0.7, time_start, time_end, linestyles ='dashed')
        ax[len(channels)+1].set_ylim([-0.2,1.1])
        ax[len(channels)+1].set_ylabel(f'Seiz. \n prob.', fontsize = 14, rotation = 0, labelpad = 30)
        ax[len(channels)+1].get_xaxis().set_ticks([])
        ax[len(channels)+1].yaxis.tick_right()
        ax[len(channels)+1].yaxis.set_ticks([0.7])


    ax[len(channels)+2].get_xaxis().set_ticks([])
    ax[len(channels)+2].get_yaxis().set_ticks([])

    ax[len(channels)+2].spines['right'].set_visible(False)
    ax[len(channels)+2].spines['left'].set_visible(False)
    ax[len(channels)+2].spines['top'].set_visible(False)
    ax[len(channels)+2].spines['bottom'].set_visible(False)
    if not anno_pred is None:
        bin_pred = np.zeros(len(time_pred))

        for anno in anno_pred:
            if anno['Name'] == 1:
                st = int(anno['Start']/2)
                end = int((anno['Start'] + anno['Duration'])/2)
                bin_pred[st:end] = 1

        ax[len(channels)+3].plot(time_pred, bin_pred, color = 'black')
        ax[len(channels)+3].set_yticks([0, 1], ['bckg', 'seiz'], fontsize = 12)
        ax[len(channels)+3].set_xlabel('Time (s)', fontsize = 14)
        #ax[8].set_ylabel('Pred.', fontsize = 14)
        ax[len(channels)+3].set_xlim([time_start, time_end])


        ax[len(channels)+3].tick_params('x', labelsize=12)
        ax[len(channels)+3].tick_params('y', labelsize=12)

    plt.tight_layout()
    f.close()
    return fig


def visualize_seizures(rec_name, rec_pred, channels, time_start, time_end, y_min, y_max,
                       model_names = None, rec_pred_second = None, scale_individually = False):
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
            
    time_pred = np.arange(0, len(rec_pred)*2, 2)
    if rec_pred_second is not None:
        nrows = len(channels) + 4
    else:
        nrows = len(channels) + 2
    fig, ax = plt.subplots(nrows = nrows, figsize = (15,6), sharex = 'row', gridspec_kw = {'hspace': 0})
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
        ax[ch].vlines(time_pred, np.min(signal), np.max(signal), linestyles = 'solid', color = 'black')
        ax[ch].hlines(0, time_start, time_end, color = 'black', linestyles ='dashed', linewidth =0.5)

        ax[ch].set_xlim([time_start, time_end])
        if scale_individually:
            ax[ch].set_ylim([np.min(channel), np.max(channel)])
        else:
            ax[ch].set_ylim([y_min, y_max])
        
        ax[ch].spines['right'].set_visible(False)
        ax[ch].spines['left'].set_visible(False)
        ax[ch].spines['top'].set_visible(False)
        ax[ch].spines['bottom'].set_visible(False)
        ax[ch].get_xaxis().set_ticks([])
        ax[ch].get_yaxis().set_ticks([])

    ax[i+1].get_xaxis().set_ticks([])
    ax[i+1].get_yaxis().set_ticks([])

    ax[i+1].spines['right'].set_visible(False)
    ax[i+1].spines['left'].set_visible(False)
    ax[i+1].spines['top'].set_visible(False)
    ax[i+1].spines['bottom'].set_visible(False)

    ax[i+2].plot(time_pred+1, rec_pred, color = 'black')
    ax[i+2].set_xlim([time_start, time_end])
    ax[i+2].hlines(0.7, time_start, time_end, linestyles ='dashed')
    ax[i+2].set_ylim([-0.2,1.1])
    ax[i+2].set_ylabel(f'Seiz. \n prob.', fontsize = 14, rotation = 0, labelpad = 30)
    ax[i+2].yaxis.tick_right()
    ax[i+2].set_yticks([0.7])
    ax[i+2].tick_params('x', labelsize=12)
    ax[i+2].tick_params('y', labelsize=12)

    if rec_pred_second is None:
        ax[i+2].set_xlabel('Time (s)', fontsize = 14)
    else:
        ax[i+2].get_xaxis().set_ticks([])
        ax[i+2].set_title(model_names[0], fontsize = 10)

        ax[i+3].get_xaxis().set_ticks([])
        ax[i+3].get_xaxis().set_ticks([])

        ax[i+3].get_yaxis().set_ticks([])
        ax[i+3].spines['right'].set_visible(False)
        ax[i+3].spines['left'].set_visible(False)
        ax[i+3].spines['top'].set_visible(False)
        ax[i+3].spines['bottom'].set_visible(False)

        ax[i+4].plot(time_pred+1, rec_pred_second, color = 'black')
        ax[i+4].set_xlim([time_start, time_end])
        ax[i+4].hlines(0.7, time_start, time_end, linestyles ='dashed')
        ax[i+4].set_ylim([-0.2,1.1])
        ax[i+4].set_ylabel(f'Seiz. \n prob.', fontsize = 14, rotation = 0, labelpad = 30)
        ax[i+4].yaxis.tick_right()
        ax[i+4].set_yticks([0.7])
        ax[i+4].tick_params('x', labelsize=12)
        ax[i+4].tick_params('y', labelsize=12)
        ax[i+4].set_xlabel('Time (s)', fontsize = 14)
        ax[i+4].set_title(model_names[1], fontsize = 10)


    plt.tight_layout()
    f.close()
    return fig