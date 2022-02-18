import pandas as pd
import pickle
from src.analysis import analysis

with open('/Users/theabrusch/Desktop/Speciale_data/finalsplit_0_split2_split_3_results.pickle', 'rb') as rb:
    res = pickle.load(rb)

recs = res['rec'].unique()
res['seiz prob'] = res['y pred']
postprocs = analysis.Postprocessing(segments = res, fs = 1/2, prob_thresh = 0.90, dur_thresh = 6, post_proces=['duration_filt', 'target_agg'])
annos = postprocs.postproces()

OVLP = analysis.AnyOverlap(annos, res, '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5')
TP, FN, FP, TN, FP_bckg = OVLP.compute_performance()