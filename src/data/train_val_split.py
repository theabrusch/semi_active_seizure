from dataapi import data_collection as dc
import pickle
import numpy as np
import warnings
from sklearn.model_selection import train_test_split

def train_val_split(hdf5_path, train_percent, seiz_classes, 
                    protocol, seed, test_subj = None, excl_seiz_classes = [], **kwargs):
    dset = hdf5_path.split('/')[-1].split('.')[0]
    pickle_path = 'data/' + dset + '_' + 'seiz_subjs_excl_seiz' + str(excl_seiz_classes) + '.pickle'
    if seed == 'None':
        seed = None
    if test_subj is None:
        # get subjects with seizure and without to ensure an equal split
        try:
            with open(pickle_path, 'rb') as fp:
                seiz_subjs = pickle.load(fp)
        except:
            print('Extracting seizure subjects and non seizure subjects.')
            seiz_subjs = get_seiz_subjs(hdf5_path, protocol, 
                                        seiz_classes, excl_seiz_classes, 
                                        pickle_path)

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
    else:
        F = dc.File(hdf5_path, 'r')
        subjs = list(F['train'].keys())
        val = [subjs[i] for i in range(len(subjs)) if i in test_subj]
        train = [subjs[i] for i in range(len(subjs)) if i not in test_subj]
        if len(val) == 0:
            TypeError('Length of validation set:', len(val))
    return train, val

def train_val_test_split(hdf5_path, seed, seiz_classes, test_subj = None, val_subj=None,
                         train_percent = 0.7, val_percent = 0.2,  **kwargs):
    '''
    Split dataset such that two subjects are in the
    validation set and 1 subject in the test set.
    Does not consider whether subjects contain seizures 
    or not and therefore throws an error if the Temple
    dataset is used. 
    '''
    dset = hdf5_path.split('/')[-1].split('.')[0]
    F = dc.File(hdf5_path, 'r')

    if 'boston' in dset:
        subjs = F.get_children(object_type=dc.Subject, get_obj = False)
        F.close()

        if test_subj is None:
            if val_subj is None:
                train_val, test = train_test_split(subjs, 
                                                    test_size = 1, 
                                                    random_state=seed)
                train, val = train_test_split(train_val, 
                                              test_size = 2, 
                                              random_state=seed)       
            else:
                val = [subjs[i] for i in range(len(subjs)) if i in val_subj]
                train_test = [subjs[i] for i in range(len(subjs)) if i not in val_subj]
                train, test = train_test_split(train_test, 
                                              test_size = 1, 
                                              random_state=seed)      
        else:

            if val_subj is None:
                test = [subjs[i] for i in range(len(subjs)) if i in test_subj]
                train_val = [subjs[i] for i in range(len(subjs)) if i not in test_subj]
                train, val = train_test_split(train_val, 
                                              test_size = 2, 
                                              random_state=seed)
            else:
                overlap = [i in test_subj for i in val_subj]
                if any(overlap):
                    warnings.warn('There is an overlap between test and validation set.')
                test = [subjs[i] for i in range(len(subjs)) if i in test_subj]
                val = [subjs[i] for i in range(len(subjs)) if i in val_subj]
                train = [subjs[i] for i in range(len(subjs)) if i not in test_subj and i not in val_subj]
        
        if len(test) == 0 or len(val) == 0:
            TypeError('Length of test set:', len(test), '. Length of validation set:', len(val))

    else:
        pickle_path = 'data/' + dset + '_' + 'seiz_subjs_new.pickle'
        try:
            with open(pickle_path, 'rb') as fp:
                seiz_subjs = pickle.load(fp)
        except:
            print('Extracting seizure subjects and non seizure subjects.')
            seiz_subjs = get_seiz_subjs(hdf5_path, 'all', seiz_classes, pickle_path)

        train_seiz, val_test_seiz = train_test_split(seiz_subjs['seiz'], 
                                                    train_size=train_percent, 
                                                    random_state=seed)
        if len(seiz_subjs['non seiz']) > 0:
            train_non_seiz, val_test_non_seiz = train_test_split(seiz_subjs['non seiz'], 
                                                                train_size=train_percent, 
                                                                random_state=seed)
        else:
            train_non_seiz = []
            val_test_non_seiz = []
        val_percent = val_percent/(1-train_percent)
        val_seiz, test_seiz = train_test_split(val_test_seiz, 
                                                train_size=val_percent, 
                                                random_state=seed)
        if len(val_test_non_seiz) > 0:
            val_non_seiz, test_non_seiz = train_test_split(val_test_non_seiz, 
                                                            train_size=val_percent, 
                                                            random_state=seed)
        else:
            val_non_seiz = []
            test_non_seiz = []
        train = np.append(train_seiz, train_non_seiz)
        val = np.append(val_seiz, val_non_seiz)
        test = np.append(test_seiz, test_non_seiz)

    return train, val, test

def get_seiz_subjs(hdf5_path, protocol, seiz_classes, excl_seiz_classes, pickle_path=None):
    F = dc.File(hdf5_path, 'r')
    if not protocol == 'all': 
        proto = F[protocol]
        subjects = proto.get_children(object_type = dc.Subject, get_obj = False)
    else:
        subjects = F.get_children(object_type = dc.Subject, get_obj = False)
    seiz_subjs = dict()
    seiz_subjs['seiz'] = []
    seiz_subjs['non seiz'] = []

    i = 1
    for subj in subjects:
        print('Subject', i, 'out of', len(subjects))
        i+=1
        seiz = 0
        excl_seiz = 0
        for rec in F[subj].keys():
            annos = F[subj][rec]['Annotations']
            for anno in annos:
                if anno['Name'].lower() in seiz_classes:
                    seiz = 1
                elif anno['Name'].lower() in excl_seiz_classes:
                    excl_seiz = 1
        if excl_seiz == 0:
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

