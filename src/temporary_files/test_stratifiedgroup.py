import numpy as np
import pandas as pd
import pyedflib
from dataapi import data_collection as dc
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold

file_name = '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5'
f = dc.File(file_name, 'r')

seiz_subjects = []
seizures = []
seiz_priority = ['mysz', 'absz', 'spsz', 'tnsz', 'tcsz', 'cpsz', 'gnsz', 'fnsz']
subjects = f.get_children(dc.Subject, get_obj = False)

i=1
for subj in subjects:
    print(i, 'out of', len(subjects))
    i+=1
    for rec in f[subj]:
        record = f[subj][rec]
        annos = record['Annotations']
        for anno in annos:
            if anno['Name'] in seiz_priority:
                seiz_subjects.append(subj)
                seizures.append(anno['Name'])

df = pd.DataFrame({'subj': seiz_subjects, 'seiz': seizures})
df_grouped = df.groupby('subj').agg(['unique', 'nunique'])['seiz']
one_seiz = df_grouped[df_grouped['nunique'] == 1]
one_seiz['unique'] = one_seiz['unique'].explode() 
fnsz = one_seiz[one_seiz['unique'] == 'fnsz'].index

stratkfold = StratifiedGroupKFold()
splits = stratkfold.split(seiz_subjects, seizures, seiz_subjects)
temp = list(splits)
i = 1
for split in temp:
    test = np.unique(np.array(seiz_subjects)[split[1]])
    train = np.unique(np.array(seiz_subjects)[split[0]])
    print('\n Split', i)
    i+=1
    print('Test subjects:', len(test))
    print('Train subjects:', len(train))
    test_seiz, test_counts = np.unique(np.array(seizures)[split[1]], return_counts = True)
    train_seiz, train_counts = np.unique(np.array(seizures)[split[0]], return_counts = True)

    print('Test distribution')
    for (seiz, count) in zip(test_seiz, test_counts):
        seiz_percent = count/np.sum(test_counts)
        print(seiz, ':', seiz_percent)
    

    print('Train distribution')
    for (seiz, count) in zip(train_seiz, train_counts):
        seiz_percent = count/np.sum(train_counts)
        print(seiz, ':', seiz_percent)