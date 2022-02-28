import pandas as pd
import pickle
from src.analysis import analysis
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
from dataapi import data_collection as dc
from src.visualization import plot_predictions

with open('/Users/theabrusch/Desktop/Speciale_data/finalsplit_test_tcsz_split_3_results.pickle', 'rb') as rb:
    res = pickle.load(rb)

recs = res['rec'].unique()
res['seiz prob'] = res['y pred']
rec_stats = []
rec_stats_seiz_collect = []
annos_pred = []
for thresh in [0,3,5]:
    postprocs = analysis.Postprocessing(segments = res, 
                                        fs = 1/2, 
                                        prob_thresh = 0.7, 
                                        dur_thresh = thresh, 
                                        statemachine = False,
                                        post_proces=['duration_filt', 'target_agg'])
    anno = postprocs.postproces()
    annos_pred.append(anno)

    OVLP = analysis.AnyOverlap(anno, res, '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', margin=0)
    TP, FN, FP, TN, total_recdur, anno_stats, recstats_collect, recstats_seiz = OVLP.compute_performance()
    rec_stats.append(recstats_collect)
    rec_stats_seiz_collect.append(recstats_seiz)



# Analyse TCSZ
f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')

TCSZ0 =  rec_stats_seiz_collect[0][rec_stats_seiz_collect[0]['seiz_type'] == 'tcsz']
TCSZ2 =  rec_stats_seiz_collect[2][rec_stats_seiz_collect[2]['seiz_type'] == 'tcsz']
rec = '/train/00008889/s002_t008'
res_badtcsz = res[res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = list(range(len(record['TCP'].attrs['chNames'])))

fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_badtcsz['seiz prob'],  
                                          channels = channels, 
                                          time_start = 0, 
                                          time_end = 230, 
                                          y_min = -500, 
                                          y_max = 500)
plt.show()


rec = '/train/00008889/s002_t008'
res_goodtcsz = res[res['rec']==rec]
seiz_goodtcsz = res_goodtcsz[res_goodtcsz['seiz_types']=='tcsz']
record = f[rec]
annos = record['Annotations']

channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_goodtcsz['seiz prob'], 
                                        annos_pred[2][rec], channels,
                                        0, 560, -500, 500)
plt.show()

channels = list(range(20))

fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_goodtcsz['seiz prob'],  
                                          channels = channels, 
                                          time_start = 200, 
                                          time_end = 220, 
                                          y_min = -500, 
                                          y_max = 500)
plt.show()


TCSZ_pred = res[res['seiz_types']=='tcsz']
TCSZ_pred['label pred']=(TCSZ_pred['seiz prob'] >=0.7).astype(int)
seg = TCSZ_pred.iloc[0]
rec = seg['rec']
segment = f[rec]['TCP'][seg['startseg']:seg['endseg'],:]
energy = np.mean(abs(segment), axis = 0)
max_energy = np.max(energy)
sort_energy = np.sort(energy)
diff = np.mean(sort_energy[-5:])-np.mean(sort_energy[:5])

energy_diff_correct_tcsz = []
energy_diff_wrong_tcsz = []
max_energy_correct_tcsz = []
max_energy_wrong_tcsz = []

for i, seg in TCSZ_pred.iterrows():
    rec = seg['rec']
    segment = f[rec]['TCP'][seg['startseg']:seg['endseg'],:]
    energy = np.mean(abs(segment), axis = 0)
    sort_energy = np.sort(energy)
    diff = np.mean(sort_energy[-5:])-np.mean(sort_energy[:5])   
    max_energy = np.max(energy)
    if seg['label'] == seg['label pred']:
        energy_diff_correct_tcsz.append(diff)
        max_energy_correct_tcsz.append(max_energy)
    else:
        energy_diff_wrong_tcsz.append(diff)
        max_energy_wrong_tcsz.append(max_energy)


# Analyze all seizures
seizures = res[res['label']==1]
seizures['label pred']=(seizures['seiz prob'] >=0.7).astype(int)
energy_diff_correct_all = []
energy_diff_wrong_all = []
max_energy_correct_all = []
max_energy_wrong_all= []

for i, seg in seizures.iterrows():
    rec = seg['rec']
    segment = f[rec]['TCP'][seg['startseg']:seg['endseg'],:]
    energy = np.mean(abs(segment), axis = 0)
    diff = np.max(energy) - np.min(energy)
    max_energy = np.max(energy)
    if seg['label'] == seg['label pred']:
        energy_diff_correct_all.append(diff)
        max_energy_correct_all.append(max_energy)
    else:
        energy_diff_wrong_all.append(diff)
        max_energy_wrong_all.append(max_energy)



