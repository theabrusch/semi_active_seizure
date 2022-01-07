import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import os

csv_files = glob.glob('src/visualization/tensorboard/temple_newmetric/*.csv')

df = None
for file in csv_files:
    df_temp = pd.read_csv(file)
    metric = file.split('/')[-1].split('-')[-1].split('_')[-1].split('.')[0]
    if df is None:
        df = df_temp
        df[metric] = df['Value']
        del df['Value']
    else:
        df[metric] = df_temp['Value']

df['new metric'] = 2*(df['sens']*df['spec'])/(df['sens']+df['spec'])

new_met_max = np.argmax(df['new metric'])
f1_max = np.argmax(df['f1'])

plt.plot(df['sens'])
plt.plot(df['spec'])
plt.vlines(new_met_max, 0, 1, colors = 'black', zorder = 3)
plt.vlines(f1_max, 0, 1, colors = 'blue', zorder = 3)
plt.show()