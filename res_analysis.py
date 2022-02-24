import pandas as pd
import pickle
from src.analysis import analysis
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
from dataapi import data_collection as dc
from src.visualization import plot_predictions

with open('/Users/theabrusch/Desktop/Speciale_data/finalsplit_test_choosebest_split_3_results.pickle', 'rb') as rb:
    res = pickle.load(rb)

seiz_types = res['seiz_types'].unique()

for seiz in seiz_types:
    idx = res['seiz_types'] == seiz
    print(seiz)
    y_true = res['label'][res['seiz_types'] == seiz]
    y_pred = (res['y pred'][res['seiz_types'] == seiz] > 0.5).astype(int)
    print(recall_score(y_true=y_true, y_pred = y_pred))

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

seiz0 = rec_stats[2][rec_stats[0]['seizures']>0]
bckg0 = rec_stats[2][rec_stats[0]['seizures']==0]
bckg0.hist('FP', bins = 50, range = [1,50])
seiz0.hist('FP', bins = 50, range = [1,50])
plt.show()

#Plot records
#FNSZ
res_rec = res['seiz prob'][res['rec']=='/train/00013145/s006_t007']
f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')
record = f['/train/00013145/s006_t007']
annos = record['Annotations']
channels = [2,3,4,5,6]
fig = plot_predictions.plot_predictions('/train/00013145/s006_t007', res_rec, 
                                        annos_pred[0]['/train/00013145/s006_t007'], channels,
                                        0, 650, -200, 200)
plt.show()


# FNSZ
rec = '/test/00008174/s001_t001'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        0, 1200, -1000, 1000)
plt.show()

# TCSZ
rec = '/train/00008889/s002_t008'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [10,11,12,13,14]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        0, 620, -2000, 2000)
plt.show()


# TCSZ
rec = '/train/00008889/s003_t006'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [10,11,12,13,14]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        0, 600, -500, 500)
plt.show()


# TCSZ
rec = '/train/00008889/s002_t005'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        0, 560, -500, 500)
plt.show()

# background
rec = '/train/00009510/s002_t000'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        0, 560, -200, 200)
plt.show()


bckg0 = rec_stats[0][rec_stats[0]['seizures'] == 0]
bckg5 = rec_stats[1][rec_stats[1]['seizures'] == 0]

#background filtered
rec = '/test/00008174/s002_t007'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[1][rec], channels,
                                        0, 300, -200, 200)
plt.show()

rec = '/train/00010106/s002_t006'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[2][rec], channels,
                                        0, 1000, -200, 200)
plt.show()


# Analyse TCSZ
TCSZ0 =  rec_stats_seiz_collect[0][rec_stats_seiz_collect[0]['seiz_type'] == 'tcsz']
TCSZ2 =  rec_stats_seiz_collect[2][rec_stats_seiz_collect[2]['seiz_type'] == 'tcsz']
rec = '/train/00008889/s002_t005'
res_badtcsz = res[res['rec']==rec]
record = f[rec]
annos = record['Annotations']
sig_seiz = np.mean(np.abs(record['TCP'][66000:70500,:]))
temp = np.mean(np.abs(record['TCP'][0:500,:]))

rec = '/train/00008889/s002_t005'
res_goodtcsz = res[res['rec']==rec]
seiz_goodtcsz = res_goodtcsz[res_goodtcsz['seiz_types']=='tcsz']
record = f[rec]
annos = record['Annotations']
sig_seiz = np.mean(np.abs(record['TCP'][48500:49000,:]))
temp = np.mean(np.abs(record['TCP'][0:500,:]))