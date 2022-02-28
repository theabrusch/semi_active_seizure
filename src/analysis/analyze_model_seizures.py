import pandas as pd
import pickle
from src.analysis import analysis
import matplotlib.pyplot as plt
import numpy as np
from dataapi import data_collection as dc
from src.visualization import plot_predictions
from src.models import get_model
import torch

with open('/Users/theabrusch/Desktop/Speciale_data/finalsplit_test_choosebest_split_3_results.pickle', 'rb') as rb:
    res = pickle.load(rb)

model_dict = {'model': 'BaselineCNN',
              'input_shape': (20,500), 
              'cnn_dropoutprob': 0.3246, 
              'dropoutprob': 0.4709 , 
              'padding': True, 
              'glob_avg_pool': False,
              'model_summary': False}
final_model = get_model.get_model(model_dict)
checkpoint = torch.load('/Users/theabrusch/Desktop/Speciale_data/final_final_model.pt', map_location = 'cpu')
final_model.load_state_dict(checkpoint['model_state_dict'])
final_model.eval()

f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')


rec = '/train/00008889/s002_t008'
res_goodtcsz = res[res['rec']==rec]
seiz_goodtcsz = res_goodtcsz[res_goodtcsz['seiz_types']=='tcsz']
record = f[rec]
annos = record['Annotations']
channels = list(range(len(record['TCP'].attrs['chNames'])))
fig = plot_predictions.visualize_seizures(rec_name=rec, 
                                          rec_pred = res_goodtcsz['y pred'],  
                                          channels = channels, 
                                          time_start = 208, 
                                          time_end = 220, 
                                          y_min = -500, 
                                          y_max = 500)
plt.show()
prev_seg =  record['TCP'][270*250:272*250,:]
segment = record['TCP'][210*250:212*250,:]

bad_ch = np.where(np.sum(((segment>500) | (segment <-500)),axis =0)>0)[0]
good_ch = np.array([0,1,2,3,4,6,17])
rand_good = np.random.choice(good_ch, len(bad_ch))

mod_seg = segment
mod_seg[:,bad_ch] = prev_seg[:,rand_good]
mod_seg_torch = torch.Tensor(segment.T).unsqueeze(0)

_, features, out = final_model(mod_seg_torch, return_features = True)

out[:,0].backward()
gradients = final_model.get_activations_gradient()
mean_gradients = torch.mean(gradients, dim=[0, 2, 3]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

features = features.detach()
actmap = torch.mean(torch.nn.functional.relu(mean_gradients*features), dim = 1).detach()
actmap /= torch.max(actmap)
plt.imshow(actmap.squeeze(), vmin = 0, vmax = 1)
plt.show()