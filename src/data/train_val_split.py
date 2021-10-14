from dataapi import data_collection as dc
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def train_val_split(hdf5_path, train_percent, seiz_classes, seed, **kwargs):
    dset = hdf5_path.split('/')[-1].split('.')[0]
    pickle_path = 'data/' + dset + '_' + 'seiz_subjs.pickle'
    
    if seed == 'None':
        seed = None

    # get subjects with seizure and without to ensure an equal split
    try:
        with open(pickle_path, 'rb') as fp:
            seiz_subjs = pickle.load(fp)
    except:
        print('Extracting seizure subjects and non seizure subjects.')
        seiz_subjs = get_seiz_subjs(hdf5_path, 'train', seiz_classes, pickle_path)

    train_seiz, val_seiz = train_test_split(seiz_subjs['seiz'], 
                                            train_size=train_percent, 
                                            random_state=seed)
    if len(seiz_subjs['non seiz']) > 0:
        train_non_seiz, val_non_seiz = train_test_split(seiz_subjs['non seiz'], 
                                                        train_size=train_percent, 
                                                        random_state=seed)
    else:
        train_non_seiz = []
        val_non_seiz = []

    train = np.append(train_seiz, train_non_seiz)
    val = np.append(val_seiz, val_non_seiz)

    return train, val

def get_seiz_subjs(hdf5_path, protocol, seiz_classes, pickle_path=None):
    F = dc.File(hdf5_path, 'r')
    proto = F[protocol]
    seiz_subjs = dict()
    seiz_subjs['seiz'] = []
    seiz_subjs['non seiz'] = []

    i = 1
    for subj in proto.keys():
        print('Subject', i, 'out of', len(proto.keys()))
        i+=1
        seiz = 0
        for rec in proto[subj].keys():
            annos = proto[subj][rec]['Annotations']
            for anno in annos:
                if anno['Name'].lower() in seiz_classes:
                    seiz = 1
        if seiz == 1:
            seiz_subjs['seiz'].append(subj)
        else:
            seiz_subjs['non seiz'].append(subj)
    if pickle_path is not None:
        with open(pickle_path, 'wb') as fp:
            pickle.dump(seiz_subjs, fp)

    return seiz_subjs


def get_seiz_recs(subject, seiz_classes, pickle_path=None):
    seiz_recs = dict()
    seiz_recs['seiz'] = []
    seiz_recs['non seiz'] = []

    for rec in subject.keys():
        seiz = 0
        annos = subject[rec]['Annotations']
        for anno in annos:
            if anno['Name'].lower() in seiz_classes:
                seiz = 1
        if seiz == 1:
            seiz_recs['seiz'].append(rec)
        else:
            seiz_recs['non seiz'].append(rec)

        if pickle_path is not None:
            with open(pickle_path, 'wb') as fp:
                pickle.dump(seiz_recs, fp)

    return seiz_recs

