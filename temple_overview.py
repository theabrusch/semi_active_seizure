from src.data import get_generator, train_val_split
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, KFold
from dataapi import data_collection as dc
import numpy as np

splitdict = dict()
splitdict['hdf5_path'] = '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5'
splitdict['only_train_seiz'] = False
splitdict['val_split'] = None
splitdict['n_val_splits'] = 5
splitdict['excl_seiz'] = False
splitdict['seiz_classes'] = ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']
splitdict['n_splits'] = 5

seiz_subjs = train_val_split.get_seiz_kfoldsubjs('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'all',
                                                    seiz_classes = ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'],
                                                    excl_seiz = False, 
                                                    pickle_path = None)
stratgroupsplit = StratifiedGroupKFold(n_splits = 5)
seizsplits = list(stratgroupsplit.split(seiz_subjs['seiz']['subjects'], seiz_subjs['seiz']['seizures'], seiz_subjs['seiz']['subjects']))
kfold = KFold(n_splits=5)
bckg_splits = list(kfold.split(seiz_subjs['non seiz']))

f = dc.File('/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5', 'r')
overview = dict()
seiz_durs = []
bckg_durs = []
for split in range(5):
    test_seiz = np.unique(np.array(seiz_subjs['seiz']['subjects'])[seizsplits[split][1]])
    test_bckg = np.unique(np.array(seiz_subjs['non seiz'])[bckg_splits[split][1]])
    test = test_seiz
    overview[split] = dict()
    overview[split]['Seizures'] = dict()
    overview[split]['Subjects'] = dict()
    bckg_dur = 0
    seiz_dur = 0
    print('Split', split, '. N subjects:', len(test_seiz) + len(test_bckg))
    for subj in test:
        subject = f[subj]
        annos = subject.get_children(dc.Annotations, get_obj = True)
        overview[split]['Subjects'][subj] = []
        for anno in annos:
            for an in anno:
                if an['Name'] in splitdict['seiz_classes']:
                    seiz_dur += an['Duration']
                    overview[split]['Subjects'][subj].append(an['Name'])
                    if an['Name'] in overview[split]:
                        overview[split]['Seizures'][an['Name']] += 1
                    else:
                        overview[split]['Seizures'][an['Name']] = 1
                else:
                    bckg_dur += an['Duration']
    seiz_durs.append(seiz_dur)
    bckg_durs.append(bckg_dur)

test_seiz = np.unique(np.array(seiz_subjs['seiz']['subjects'])[seizsplits[3][1]])
tnsz_subjs = []
for subj in test_seiz:
    subject = f[subj]
    annos = subject.get_children(dc.Annotations, get_obj = True)
    for anno in annos:
        for an in anno:
            if an['Name'] == 'tnsz':
                if not subj in tnsz_subjs:
                    tnsz_subjs.append(subj)