import pandas as pd
import pickle
from src.analysis import analysis
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
from dataapi import data_collection as dc
from src.visualization import plot_predictions


files = ['/Users/theabrusch/Desktop/Speciale_data/fnsz_gnsz_cpsz_full_eval_split_3_results.pickle', '/Users/theabrusch/Desktop/Speciale_data/finalsplit_test_choosebest_split_3_results.pickle']
models = ['rev', 'full']
res = dict()
annos_pred = dict()
recstats = dict()
rectstats_seiz = dict()
thresh = 5    

for i in range(len(files)):
    with open(files[i], 'rb') as rb:
        res[models[i]] = pickle.load(rb)
    if 'seiz prob' not in res[models[i]].keys():
        res[models[i]]['seiz prob'] = res[models[i]]['y pred']
    postprocs = analysis.Postprocessing(segments = res[models[i]], 
                                            label_fs = 1/2, 
                                            orig_fs = 250,
                                            prob_thresh = 0.7, 
                                            dur_thresh = thresh, 
                                            statemachine = False,
                                            post_proces=['duration_filt', 'target_agg'])
    annos_pred[models[i]], res[models[i]] = postprocs.postproces()

    OVLP = analysis.AnyOverlap(pred_annos=annos_pred[models[i]], hdf5_path='/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', seiz_eval = None, margin=0)
    TP, FN, FP, TN, total_recdur, anno_stats, recstats[models[i]], rectstats_seiz[models[i]] = OVLP.compute_performance()



#Plot records
#FNSZ
f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')

model = 'full'

rec = '/train/00013145/s006_t007'
res_temp = res[model]
res_rec = res_temp[res_temp['rec']==rec]
record = f[rec]

annos = record['Annotations']

model2 = 'full'
res_temp2 = res[model2]
res_rec2 = res_temp[res_temp['rec']==rec]

channels = [2,3,4,5,6]
fig = plot_predictions.plot_predictions(rec, res_rec['seiz prob'], 
                                        annos_pred[model][rec], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'],
                                        0, 650, -500, 500, rec_pred2=res_rec2['seiz prob'], anno_pred2=annos_pred[model2][rec])
plt.show()


# FNSZ
rec = '/train/00010088/s010_t001'
res_rec = res['seiz prob'][res['rec']==rec]
record = f[rec]
annos = record['Annotations']
channels = [0,1,2,3,4]
fig = plot_predictions.plot_predictions(rec, res_rec, annos_pred[2][rec], channels,
                                        ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'],
                                        0, 1300, -1000, 1000)
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

