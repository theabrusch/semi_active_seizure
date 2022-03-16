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

files = ['/Users/theabrusch/Desktop/Speciale_data/fullmodel_valsplit_results.pickle', '/Users/theabrusch/Desktop/Speciale_data/fnsz_gnsz_cpsz_full_eval_split_3_results.pickle', '/Users/theabrusch/Desktop/Speciale_data/finalsplit_test_choosebest_split_3_results.pickle']
models = ['val', 'fnsz', 'full']
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

cpsz = rectstats_seiz['full'][rectstats_seiz['full']['seiz_type']=='cpsz']
fnsz = rectstats_seiz['fnsz'][rectstats_seiz['fnsz']['seiz_type']=='fnsz']
tcsz = rectstats_seiz['full'][rectstats_seiz['full']['seiz_type']=='tcsz']
gnsz = rectstats_seiz['full'][rectstats_seiz['full']['seiz_type']=='gnsz']

val_tcsz = rectstats_seiz['val'][rectstats_seiz['val']['seiz_type']=='tcsz']

model = 'val'

rec = '/train/00005426/s009_t001'
res_temp = res[model]
res_rec = res_temp[res_temp['rec']==rec]
record = f[rec]

annos = record['Annotations']

channels = list(range(5))
fig = plot_predictions.plot_predictions(rec, res_rec['seiz prob'], 
                                        annos_pred[model][rec], channels,
                                        ['tcsz'], 0, 1048, -500, 500)
plt.show()

model2 = 'tcsz'
res_temp2 = res[model2]
res_rec2 = res_temp2[res_temp2['rec']==rec]
channels = list(range(len(record['TCP'].attrs['chNames'])))

fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_rec['seiz prob'],  
                                          channels = channels, 
                                          time_start = 400, 
                                          time_end = 410, 
                                          y_min = -100, 
                                          y_max = 100)
                                          #rec_pred_second = res_rec2['seiz prob'],
                                          #model_names = ['Full model', 'TCSZ model'])
plt.show()

rec = '/train/00006520/s003_t003'
record = f[rec]

annos = record['Annotations']
channels = list(range(20))
fig = plot_predictions.plot_predictions(rec, None, 
                                        None, channels,
                                        ['tcsz', 'fnsz'], 140, 190, -500, 500)
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


ch_hdr = record['TCP'].attrs['edfSignalHdr']
temp_ch = np.array(('P4-O2', 'uV', 250, 5482.288, -5482.28, 32767, -32767, '[HP:0.000 Hz LP:0.0 Hz N:0.0]', 'Unknown'), dtype = ch_hdr.dtype)
ch_hdr = np.append(ch_hdr, temp_ch)
record['TCP'].attrs['edfSignalHdr'] = ch_hdr

record.export_to_edf('/Users/theabrusch/Desktop/Speciale_data/edf/00008889_s002_t008.edf',
                     ['TCP'], with_annotations=True)

subjects = f.get_children(dc.Subject, get_obj=False)
tcsz_subj_recs = dict()
for subj in subjects:
    subject = f[subj]
    for rec in subject.keys():
        record = subject[rec]
        for anno in record['Annotations']:
            if anno['Name']=='tcsz' and subj not in tcsz_subj_recs.keys():
                tcsz_subj_recs[subj] = [rec]
            elif anno['Name']=='tcsz': 
                tcsz_subj_recs[subj].append(rec)