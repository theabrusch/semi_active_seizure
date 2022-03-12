import pandas as pd
import pickle
from src.analysis import analysis
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
from dataapi import data_collection as dc
from src.visualization import plot_predictions

with open('/Users/theabrusch/Desktop/Speciale_data/finalsplit_test_fnsz_cpsz_gnsz_valsplit_f1_split_3_results.pickle', 'rb') as rb:
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
                                        label_fs = 1/2, 
                                        orig_fs = 250,
                                        prob_thresh = 0.7, 
                                        dur_thresh = thresh, 
                                        statemachine = False,
                                        post_proces=['duration_filt', 'target_agg'])
    anno, res = postprocs.postproces()
    annos_pred.append(anno)

    OVLP = analysis.AnyOverlap(anno, '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', seiz_eval = None, margin=0)
    TP, FN, FP, TN, total_recdur, anno_stats, recstats_collect, recstats_seiz = OVLP.compute_performance()
    rec_stats.append(recstats_collect)
    rec_stats_seiz_collect.append(recstats_seiz)


temp = recstats_seiz[((recstats_seiz['seiz_type'] == 'fnsz')|(recstats_seiz['seiz_type'] == 'gnsz')|(recstats_seiz['seiz_type'] == 'cpsz'))]
gnsz = rec_stats_seiz_collect[2][(rec_stats_seiz_collect[0]['seiz_type'] == 'gnsz')]
cpsz = recstats_seiz[(recstats_seiz['seiz_type'] == 'cpsz')]
gnsz = recstats_seiz[(recstats_seiz['seiz_type'] == 'gnsz')]
tcsz = recstats_seiz[(recstats_seiz['seiz_type'] == 'tcsz')]

#Plot records
#FNSZ
res_rec = res['seiz prob'][res['rec']=='/train/00013145/s006_t007']
f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')
record = f['/train/00013145/s006_t007']
annos = record['Annotations']
channels = [2,3,4,5,6]
fig = plot_predictions.plot_predictions('/train/00013145/s006_t007', res_rec, 
                                        annos_pred[0]['/train/00013145/s006_t007'], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'],
                                        0, 650, -500, 500)
plt.show()


# FNSZ
rec = '/test/00008174/s001_t001'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'],
                                        0, 1200, -1000, 1000)
plt.show()

# TCSZ
rec = '/train/00008889/s002_t008'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [10,11,12,13,14]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'],
                                        0, 620, -2000, 2000)
plt.show()


# TCSZ
rec = '/train/00008889/s003_t006'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [10,11,12,13,14]

fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'], 
                                        0, 600, -500, 500)
plt.show()


# TCSZ
rec = '/train/00008889/s002_t005'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'], 
                                        0, 560, -500, 500)
plt.show()

# background
rec = '/train/00009510/s002_t000'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'],
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
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[0][rec], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'],
                                        0, 300, -300, 300)
plt.show()

rec = '/train/00010106/s002_t006'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[2][rec], channels,
                                        0, 1000, -200, 200)
plt.show()

