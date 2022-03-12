import pandas as pd
import pickle
from src.analysis import analysis
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
from dataapi import data_collection as dc
from src.visualization import plot_predictions
import shutil
import pyedflib
from datetime import timedelta
import datetime

files = ['/Users/theabrusch/Desktop/Speciale_data/finalsplit_test_fnsz_cpsz_gnsz_valsplit_f1_split_3_results.pickle', '/Users/theabrusch/Desktop/Speciale_data/finalsplit_test_choosebest_split_3_results.pickle']
models = ['fnsz', 'full']
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

# Analyse TCSZ
f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')

cpsz_full = rectstats_seiz['full'][rectstats_seiz['full']['seiz_type']=='cpsz'].reset_index()
fnsz_full = rectstats_seiz['full'][rectstats_seiz['full']['seiz_type']=='fnsz'].reset_index()
gnsz_full = rectstats_seiz['full'][rectstats_seiz['full']['seiz_type']=='gnsz'].reset_index()

cpsz_res = rectstats_seiz['fnsz'][rectstats_seiz['fnsz']['seiz_type']=='cpsz'].reset_index()
fnsz_res = rectstats_seiz['fnsz'][rectstats_seiz['fnsz']['seiz_type']=='fnsz'].reset_index()
gnsz_res = rectstats_seiz['fnsz'][rectstats_seiz['fnsz']['seiz_type']=='gnsz'].reset_index()


#FNSZ first seizures
fnsz_joined = fnsz_res[((fnsz_res['hit']==1)&(fnsz_full['hit']==0))]

model = 'full'

rec = '/train/00013145/s003_t005'
res_temp = res[model]
res_rec = res_temp[res_temp['rec']==rec]
record = f[rec]

annos = record['Annotations']
channels = list(range(len(record['TCP'].attrs['chNames'])))

fig = plot_predictions.plot_predictions(rec, res_rec['seiz prob'], 
                                        annos_pred[model][rec], channels,
                                        ['fnsz'], 200, 400, -300, 300)

model2 = 'fnsz'
res_temp2 = res[model2]
res_rec2 = res_temp2[res_temp2['rec']==rec]

fig = plot_predictions.plot_predictions(rec, res_rec2['seiz prob'], 
                                        annos_pred[model2][rec], channels,
                                        ['fnsz'], 200, 400, -300, 300)

fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_rec['seiz prob'],  
                                          channels = channels, 
                                          time_start = 350, 
                                          time_end = 360, 
                                          y_min = -100, 
                                          y_max = 100,
                                          rec_pred_second = res_rec2['seiz prob'],
                                          model_names = ['Full model', 'Small model'])
plt.show()


#FNSZ second seizures
model = 'full'

rec = '/train/00010106/s003_t002'
res_temp = res[model]
res_rec = res_temp[res_temp['rec']==rec]
record = f[rec]

annos = record['Annotations']
channels = list(range(len(record['TCP'].attrs['chNames'])))

fig = plot_predictions.plot_predictions(rec, res_rec['seiz prob'], 
                                        annos_pred[model][rec], channels,
                                        ['fnsz'], 350, 600, -500, 500)


model2 = 'fnsz'
res_temp2 = res[model2]
res_rec2 = res_temp2[res_temp2['rec']==rec]

fig = plot_predictions.plot_predictions(rec, res_rec2['seiz prob'], 
                                        annos_pred[model2][rec], channels,
                                        ['fnsz'], 350, 600, -500, 500)
plt.show()


fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_rec['seiz prob'],  
                                          channels = channels, 
                                          time_start = 505, 
                                          time_end = 515, 
                                          y_min = -100, 
                                          y_max = 100,
                                          rec_pred_second = res_rec2['seiz prob'],
                                          model_names = ['Full model', 'Small model'])
plt.show()



#FNSZ third seizures
fnsz_joined = fnsz_res[((fnsz_res['hit']==0)&(fnsz_full['hit']==1))]

model = 'full'

rec = '/train/00010843/s010_t000'
res_temp = res[model]
res_rec = res_temp[res_temp['rec']==rec]
record = f[rec]

annos = record['Annotations']
channels = list(range(len(record['TCP'].attrs['chNames'])))

fig = plot_predictions.plot_predictions(rec, res_rec['seiz prob'], 
                                        annos_pred[model][rec], channels,
                                        ['fnsz'], 50, 150, -100, 100)
plt.show()

model2 = 'fnsz'
res_temp2 = res[model2]
res_rec2 = res_temp2[res_temp2['rec']==rec]

fig = plot_predictions.plot_predictions(rec, res_rec2['seiz prob'], 
                                        annos_pred[model2][rec], channels,
                                        ['fnsz'], 50, 150, -100, 100)
plt.show()


fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_rec['seiz prob'],  
                                          channels = channels, 
                                          time_start = 88, 
                                          time_end = 98, 
                                          y_min = -100, 
                                          y_max = 100,
                                          rec_pred_second = res_rec2['seiz prob'],
                                          model_names = ['Full model', 'Small model'])
plt.show()

#CPSZ
cpsz_joined = cpsz_res[((cpsz_res['hit']==1)&(cpsz_full['hit']==0))]
rec = '/train/00000883/s011_t007'

model = 'fnsz'

res_temp = res[model]
res_rec = res_temp[res_temp['rec']==rec]
record = f[rec]

annos = record['Annotations']
channels = list(range(len(record['TCP'].attrs['chNames'])))

fig = plot_predictions.plot_predictions(rec, res_rec['seiz prob'], 
                                        annos_pred[model][rec], channels,
                                        ['cpsz'], 0, 750, -300, 300)
plt.show()


# GNSZ
gnsz_joined = gnsz_res[((gnsz_res['hit']==1)&(gnsz_full['hit']==0))]
rec = '/train/00011870/s001_t003'


model = 'full'

res_temp = res[model]
res_rec = res_temp[res_temp['rec']==rec]
record = f[rec]

annos = record['Annotations']
channels = list(range(len(record['TCP'].attrs['chNames'])))

fig = plot_predictions.plot_predictions(rec, res_rec['seiz prob'], 
                                        annos_pred[model][rec], channels,
                                        ['gnsz'], 0, 1500, -500, 500)
plt.show()

model2 = 'fnsz'
res_temp2 = res[model2]
res_rec2 = res_temp2[res_temp2['rec']==rec]

fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_rec['seiz prob'],  
                                          channels = channels, 
                                          time_start = 200, 
                                          time_end = 210, 
                                          y_min = -100, 
                                          y_max = 100,
                                          rec_pred_second = res_rec2['seiz prob'],
                                          model_names = ['Full model', 'Small model'])
plt.show()



# GNSZ
rec = '/train/00012707/s003_t009'


model = 'fnsz'

res_temp = res[model]
res_rec = res_temp[res_temp['rec']==rec]
record = f[rec]

annos = record['Annotations']
channels = list(range(len(record['TCP'].attrs['chNames'])))

fig = plot_predictions.plot_predictions(rec, res_rec['seiz prob'], 
                                        annos_pred[model][rec], channels,
                                        ['gnsz'], 100, 300, -500, 500)
plt.show()

model2 = 'fnsz'
res_temp2 = res[model2]
res_rec2 = res_temp2[res_temp2['rec']==rec]

fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_rec['seiz prob'],  
                                          channels = channels, 
                                          time_start = 440, 
                                          time_end = 450, 
                                          y_min = -100, 
                                          y_max = 100,
                                          rec_pred_second = res_rec2['seiz prob'],
                                          model_names = ['Full model', 'Small model'])
plt.show()