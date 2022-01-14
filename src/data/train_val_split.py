from dataapi import data_collection as dc
import pickle
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def train_val_split(hdf5_path, 
                    train_percent, 
                    protocol,
                    seed, 
                    seiz_strat = False, 
                    test_subj = None, 
                    excl_seiz_classes = [], 
                    **kwargs):

    dset = hdf5_path.split('/')[-1].split('.')[0]
    pickle_path = 'data/' + dset + '_' + 'seiz_subjs_seiz_strat.pickle'
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
                                        excl_seiz_classes, 
                                        pickle_path)
        #remove subjects that contain seizures
        if excl_seiz_classes is not None:
            for seiz in seiz_subjs['seiz'].keys():
                if seiz in excl_seiz_classes:
                    del seiz_subjs['seiz'][seiz]

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
        subjs = F.get_children(object_type=dc.Subject, get_obj = False)
        val = [subjs[i] for i in range(len(subjs)) if i in test_subj]
        train = [subjs[i] for i in range(len(subjs)) if i not in test_subj]
        if len(val) == 0:
            TypeError('Length of validation set:', len(val))
    return train, val

def train_val_test_split(hdf5_path, 
                         seed, 
                         seiz_classes,
                         test_subj = None, 
                         seiz_strat = True,
                         val_subj = None,
                         train_percent = 0.7, 
                         val_percent = 0.15,  
                         **kwargs):
    '''
    Split dataset such that two subjects are in the
    validation set and 1 subject in the test set (if the Boston dataset
    is used). For the Temple dataset the subjects are split according 
    to seizures. 
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
        pickle_path = 'data/'+dset+'seiz_subjs_seiz_strat.pickle'
        np.random.seed(seed) # set numpy seed
        try:
            with open(pickle_path, 'rb') as fp:
                seiz_subjs = pickle.load(fp)
        except:
            print('Extracting seizure subjects and non seizure subjects.')
            seiz_subjs = get_seiz_subjs(hdf5_path, 'all', pickle_path)
        # remove subjects that only contain the seizure type
        # to be excluded
        temp = list(seiz_subjs['seiz'].keys())
        for seiz in temp:
            if seiz not in seiz_classes:
                del seiz_subjs['seiz'][seiz]
        val_percent_temp = round(val_percent/(1-train_percent), 2)
        if seiz_strat: 
            print(seiz_classes)
            # distribute seizure types over train, validation and test sets 
            train_seiz = np.array([])
            val_seiz = np.array([])
            missing_seizures = np.array([])
            test_seiz = np.array([])
            for seiz in seiz_subjs['seiz'].keys():
                if len(seiz_subjs['seiz'][seiz]) < 10:
                    missing_seizures = np.append(missing_seizures, seiz_subjs['seiz'][seiz])
                else:
                    train_seiz_temp, val_test_seiz = train_test_split(seiz_subjs['seiz'][seiz], 
                                                                        train_size=train_percent, 
                                                                        random_state=seed)
                    val_seiz_temp, test_seiz_temp = train_test_split(val_test_seiz, 
                                                                    train_size=val_percent_temp, 
                                                                    random_state=seed)
                    train_seiz = np.append(train_seiz, train_seiz_temp)
                    val_seiz = np.append(val_seiz, val_seiz_temp)
                    test_seiz = np.append(test_seiz, test_seiz_temp)
                
            if len(missing_seizures) > 0:
                train_seiz_temp, val_test_seiz = train_test_split(missing_seizures, 
                                                                    train_size=train_percent, 
                                                                    random_state=seed)
                val_seiz_temp, test_seiz_temp = train_test_split(val_test_seiz, 
                                                                 train_size=val_percent_temp, 
                                                                 random_state=seed)
                train_seiz = np.append(train_seiz, train_seiz_temp)
                val_seiz = np.append(val_seiz, val_seiz_temp)
                test_seiz = np.append(test_seiz, test_seiz_temp)
        else:
            temp = []
            val_percent = round(val_percent/(1-train_percent), 2)
            for seiz in seiz_subjs['seiz'].keys():
                temp = np.append(temp, seiz_subjs['seiz'][seiz])

            seiz_subjs['seiz'] = shuffle(temp)
            train_seiz, val_test_seiz = train_test_split(seiz_subjs['seiz'], 
                                                        train_size=train_percent, 
                                                        random_state=seed)
            val_seiz, test_seiz = train_test_split(val_test_seiz, 
                                                    train_size=val_percent, 
                                                    random_state=seed)
        if len(seiz_subjs['non seiz']) > 0:
            train_non_seiz, val_test_non_seiz = train_test_split(seiz_subjs['non seiz'], 
                                                                train_size=train_percent, 
                                                                random_state=seed)
        else:
            train_non_seiz = []
            val_test_non_seiz = []
        if len(val_test_non_seiz) > 0:
            val_non_seiz, test_non_seiz = train_test_split(val_test_non_seiz, 
                                                            train_size=val_percent_temp, 
                                                            random_state=seed)
        else:
            val_non_seiz = []
            test_non_seiz = []
        train = np.append(train_seiz, train_non_seiz)
        val = np.append(val_seiz, val_non_seiz)
        test = np.append(test_seiz, test_non_seiz)
    return train, val, test

def get_seiz_subjs(hdf5_path, protocol, pickle_path=None):
    F = dc.File(hdf5_path, 'r')
    if not protocol == 'all': 
        proto = F[protocol]
        subjects = proto.get_children(object_type = dc.Subject, get_obj = False)
    else:
        subjects = F.get_children(object_type = dc.Subject, get_obj = False)
    seiz_subjs = dict()
    seiz_subjs['seiz'] = dict()
    seiz_subjs['non seiz'] = []
    seiz_priority = ['seiz', 'mysz', 'absz', 'spsz', 'tnsz', 'tcsz', 'cpsz', 'gnsz', 'fnsz']
    i = 1
    for subj in subjects:
        print('Subject', i, 'out of', len(subjects))
        i+=1
        seiz = 0
        seizure_types = []
        for rec in F[subj].keys():
            annos = F[subj][rec]['Annotations']
            for anno in annos:
                if anno['Name'].lower() in seiz_priority:
                    seiz = 1
                    if not anno['Name'].lower() in seizure_types:
                        seizure_types.append(anno['Name'])
        if seiz == 1:
            pri_seiz = [seiz for seiz in seiz_priority if seiz in seizure_types][0]
            if pri_seiz in seiz_subjs['seiz'].keys():
                seiz_subjs['seiz'][pri_seiz].append(subj)
            else:
                seiz_subjs['seiz'][pri_seiz] = [subj]
        else:
            seiz_subjs['non seiz'].append(subj)

    if len(seiz_subjs['seiz'].keys()) == 1:
        seiz_subjs['seiz'] = seiz_subjs['seiz'][list(seiz_subjs).keys()[0]]

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

