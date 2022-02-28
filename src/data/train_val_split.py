from doctest import testmod, testsource
from dataapi import data_collection as dc
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, KFold
from sklearn.utils import shuffle

def train_val_split(hdf5_path, 
                    protocol,
                    seed, 
                    seiz_strat = False, 
                    test_subj = None, 
                    train_percent =0.5, 
                    excl_seiz_classes = [], 
                    **kwargs):

    dset = hdf5_path.split('/')[-1].split('.')[0]
    pickle_path = 'data/' + dset + '_' + 'seiz_subjs_seiz_strat_new.pickle'
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
        if test_subj is not None and val_subj is not None:
            test = test_subj
            val = val_subj
            subjs = F.get_children(object_type=dc.Subject, get_obj = False)
            F.close()
            train = [subj for subj in subjs if subj not in test and subj not in val]
        else:
            #pickle_path = 'data/'+dset+'seiz_subjs_seiz_strat_new.pickle'
            print('Extracting seizure subjects and non seizure subjects.')
            seiz_subjs = get_seiz_subjs(hdf5_path, 'all', seiz_classes = seiz_classes, pickle_path=None)
            # remove subjects that only contain the seizure type
            # to be excluded
            temp = list(seiz_subjs['seiz'].keys())
            #for seiz in temp:
            #    if seiz not in seiz_classes:
            #        del seiz_subjs['seiz'][seiz]
            val_percent_temp = round(val_percent/(1-train_percent), 2)
            if seiz_strat: 
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

def get_seiz_subjs(hdf5_path, protocol, seiz_classes, excl_seiz=False, pickle_path=None):
    F = dc.File(hdf5_path, 'r')
    if not protocol == 'all': 
        proto = F[protocol]
        subjects = proto.get_children(object_type = dc.Subject, get_obj = False)
    else:
        subjects = F.get_children(object_type = dc.Subject, get_obj = False)
    seiz_subjs = dict()
    seiz_subjs['seiz'] = dict()
    seiz_subjs['non seiz'] = []
    i = 1
    for subj in subjects:
        i+=1
        seiz = 0
        excl_subj = False
        seizure_types = []
        for rec in F[subj].keys():
            annos = F[subj][rec]['Annotations']
            for anno in annos:
                if anno['Name'].lower() in seiz_classes:
                    seiz = 1
                    seizure_types.append(anno['Name'])
                elif not anno['Name'].lower() == 'bckg':
                    excl_subj = True
        if seiz == 1:
            if (excl_seiz and not excl_subj) or not excl_seiz:
                seiz, counts = np.unique(seizure_types, return_counts = True)
                seiz_sort = seiz[np.argmax(counts)]
                if seiz_sort in seiz_subjs['seiz'].keys():
                    seiz_subjs['seiz'][seiz_sort].append(subj)
                else:
                    seiz_subjs['seiz'][seiz_sort] = [subj]
        elif not excl_subj:
            seiz_subjs['non seiz'].append(subj)

    if len(seiz_subjs['seiz'].keys()) == 1:
        seiz_subjs['seiz'] = seiz_subjs['seiz'][list(seiz_subjs).keys()[0]]

    if pickle_path is not None:
        with open(pickle_path, 'wb') as fp:
            pickle.dump(seiz_subjs, fp)

    return seiz_subjs

def get_kfold(hdf5_path, 
                split,
                seiz_classes,
                only_train_seiz = None,
                val_split = None,
                excl_seiz = False,
                n_splits = 5,
                n_val_splits = 7,
                choose_orig_val = True,
                orig_split = False,
                **kwargs):

    if orig_split:
        split_seiz_classes = ['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz']
    else:
        split_seiz_classes = seiz_classes
    seiz_subjs = get_seiz_kfoldsubjs(hdf5_path, 'all',
                                     seiz_classes = split_seiz_classes,
                                     excl_seiz = excl_seiz, 
                                     pickle_path = None)

    # stratified splitting on seizure type
    stratgroupsplit = StratifiedGroupKFold(n_splits = n_splits)
    seiz_splits = list(stratgroupsplit.split(seiz_subjs['seiz']['subjects'], seiz_subjs['seiz']['seizures'], seiz_subjs['seiz']['subjects']))
    seiz_split = seiz_splits[split]
    # regular splitting on non-seizure subjects
    kfold = KFold(n_splits=n_splits)
    bckg_splits = list(kfold.split(seiz_subjs['non seiz']))
    bckg_split = bckg_splits[split]

    test_seiz = np.unique(np.array(seiz_subjs['seiz']['subjects'])[seiz_split[1]])
    train_bckg = np.unique(np.array(seiz_subjs['non seiz'])[bckg_split[0]])
    test_bckg = np.unique(np.array(seiz_subjs['non seiz'])[bckg_split[1]])

    if orig_split:
        seiz_idx = [i for i in range(len(seiz_subjs['seiz']['seizures'])) if seiz_subjs['seiz']['seizures'][i] in seiz_classes]
        spec_seiz_subjs = np.unique(seiz_subjs['seiz']['subjects'][seiz_idx])
        test_seiz = [subj for subj in test_seiz if subj in spec_seiz_subjs]
        test_bckg = []

    test = np.append(test_seiz, test_bckg)

    if val_split is None:
        train_seiz = np.unique(np.array(seiz_subjs['seiz']['subjects'])[seiz_split[0]])

        # move subjects to training split if they only contain the 
        # seizure defined in only train seiz
        if only_train_seiz is not None:
            print('Moving subjects with only seizure type', only_train_seiz, 'to training set.')
            df = pd.DataFrame({'subj': seiz_subjs['seiz']['subjects'], 'seiz': seiz_subjs['seiz']['seizures']})
            df_grouped = df.groupby('subj').agg(['unique', 'nunique'])['seiz']
            one_seiz = df_grouped[df_grouped['nunique'] == 1]
            one_seiz['unique'] = one_seiz['unique'].explode() 
            only_train = one_seiz[one_seiz['unique'] == only_train_seiz].index
            moved_seiz = 0
            for subj in only_train:
                if subj in test_seiz:
                    idx = np.where(test_seiz == subj)
                    test_seiz = np.delete(test_seiz, idx)
                    train_seiz = np.append(train_seiz, subj)
                    moved_seiz+=1
            print('Moved', moved_seiz, 'to training set.')

        train = np.append(train_seiz, train_bckg)

        return train, test
    elif choose_orig_val: #choose one of the other splits for validation
        if val_split == split:
            val_split = np.random.choice([i for i in range(n_splits) if i != split])
            print('Same split is chosen for testing and validation split. Randomly setting validation set to', val_split)
        seiz_val_split = seiz_splits[val_split]
        bckg_val_split = bckg_splits[val_split]
        val_seiz = np.unique(np.array(seiz_subjs['seiz']['subjects'])[seiz_val_split[1]])
        val_bckg = np.unique(np.array(seiz_subjs['non seiz'])[bckg_val_split[1]])
        train_seiz = np.unique(np.array(seiz_subjs['seiz']['subjects'])[seiz_split[0]])

        val = np.append(val_seiz, val_bckg)
        train_temp = np.append(train_seiz, train_bckg)

        train = np.array([subj for subj in train_temp if subj not in val])
        
        return train, val, test
    else: #split training set again to get validaition and training set
        train_subj_split = np.array(seiz_subjs['seiz']['subjects'])[seiz_split[0]]
        train_seiz_split = np.array(seiz_subjs['seiz']['seizures'])[seiz_split[0]]
        # split seizures
        stratgroupsplit = StratifiedGroupKFold(n_splits = n_val_splits)
        seiz_splits = stratgroupsplit.split(train_subj_split, train_seiz_split, train_subj_split) 
        # split background
        kfold = KFold(n_splits=n_val_splits)
        bckg_splits = kfold.split(train_bckg)

        seiz_split = list(seiz_splits)[val_split]
        bckg_split = list(bckg_splits)[val_split]

        # get train and val seizures
        train_seiz = np.unique(np.array(train_subj_split)[seiz_split[0]])
        val_seiz = np.unique(np.array(train_subj_split)[seiz_split[1]])

        # get train and val background
        val_bckg = np.unique(np.array(train_bckg)[bckg_split[1]])
        train_bckg = np.unique(np.array(train_bckg)[bckg_split[0]])

        if orig_split:
            train_seiz = [subj for subj in train_seiz if subj in spec_seiz_subjs]
            val_seiz = [subj for subj in val_seiz if subj in spec_seiz_subjs]
            train_bckg = np.random.choice(train_bckg, size = int(len(train_seiz)*2))
            val_bckg = []
        
        train = np.append(train_seiz, train_bckg)
        val = np.append(val_seiz, val_bckg)

        return train, val, test

def get_seiz_kfoldsubjs(hdf5_path, protocol, seiz_classes, excl_seiz=False, pickle_path=None):
    '''
    Get all seizure and non seizure subjects in format to use for kfold stratified group splitting
    '''
    F = dc.File(hdf5_path, 'r')
    if not protocol == 'all': 
        proto = F[protocol]
        subjects = proto.get_children(object_type = dc.Subject, get_obj = False)
    else:
        subjects = F.get_children(object_type = dc.Subject, get_obj = False)
    seiz_subjs = dict()
    seiz_subjs['seiz'] = dict()
    seiz_subjs['seiz']['subjects'] = np.array([])
    seiz_subjs['seiz']['seizures'] = np.array([])
    seiz_subjs['non seiz'] = []
    i = 1
    for subj in subjects:
        i+=1
        seiz = 0
        excl_subj = False
        seizure_types = []
        for rec in F[subj].keys():
            annos = F[subj][rec]['Annotations']
            for anno in annos:
                if anno['Name'].lower() in seiz_classes:
                    seizure_types.append(anno['Name'])
                    seiz = 1
                elif not anno['Name'].lower() == 'bckg':
                    excl_subj = True
        if seiz == 1:
            # if there is a seizure present in the subject and 
            # if we are not excluding seizure types, append the 
            # subject
            if (excl_seiz and not excl_subj) or not excl_seiz:
                subject = [subj]*len(seizure_types)
                seiz_subjs['seiz']['subjects'] = np.append(seiz_subjs['seiz']['subjects'], subject)
                seiz_subjs['seiz']['seizures'] = np.append(seiz_subjs['seiz']['seizures'], seizure_types)
        elif not excl_subj:
            seiz_subjs['non seiz'].append(subj)
    if pickle_path is not None:
        with open(pickle_path, 'wb') as fp:
            pickle.dump(seiz_subjs, fp)
    return seiz_subjs

def get_transfer_subjects(hdf5_path, subjects, seiz_classes,
                          min_seiz = 20, min_ratio = 2, 
                          test_frac = 1/3, **kwargs):
    '''
    Function for splitting subjects into seizure records to use
    for transferring knowledge and into testing 
    '''
    file = dc.File(hdf5_path, 'r')
    transfer_subjects = []
    transfer_records = dict()
    test_records = dict()
    for subj in subjects:
        subject = file[subj]
        seiz_recs = get_seiz_recs(subject, seiz_classes)
        if len(seiz_recs['seiz']['rec']) > 1:
            # choose 1 record with seizure to use for transferring
            seiz_pd = pd.DataFrame(seiz_recs['seiz']).sort_values(by='rec').reset_index(drop=True)
            bckg_pd = pd.DataFrame(seiz_recs['non seiz']).sort_values(by='rec').reset_index(drop=True)
            n_seiz_recs = len(seiz_pd)
            if test_frac == 0:
                dur = 0
                i = 0
                transfer = []
                while (i+1) < n_seiz_recs and dur < min_seiz:
                    transfer.append(seiz_pd['rec'].values[i])
                    dur += seiz_pd['seiz dur'].values[i]
                    i += 1
                test = np.array(seiz_pd['rec'].values[i:])
                transfer = np.array(transfer)
                seiz_dur = np.sum(seiz_pd['seiz dur'].values[:i])
                bckg_dur = np.sum(seiz_pd['bckg dur'].values[:i])
                transfer_ratio = bckg_dur / seiz_dur

                # if any non seizure records, sample enough background records
                i = 0
                n_bckg = len(seiz_recs['non seiz']['rec'])
                while transfer_ratio < min_ratio and  i < n_bckg:
                    transfer=np.append(transfer,bckg_pd['rec'].values[i])
                    bckg_dur += bckg_pd['bckg dur'].values[i]
                    transfer_ratio = bckg_dur / seiz_dur
                    i += 1
                test_bckg = bckg_pd['rec'].values[i:]
                test = np.append(test, test_bckg)
                transfer_records[subj] = transfer
            else:                
                test_recs_seiz = int(np.ceil(test_frac*len(seiz_pd)))
                # select last test_recs (int) for test and put the remaining in transfer
                test = np.array(seiz_pd['rec'].values[-test_recs_seiz:])
                transfer_seiz = np.array(seiz_pd['rec'].values[:-test_recs_seiz])
                transfer_bckg = []
                if len(bckg_pd) > 0:
                    test_recs_bckg = int(np.ceil(test_frac*len(bckg_pd)))
                    test = np.append(test, bckg_pd['rec'].values[-test_recs_bckg:])
                    transfer_bckg = np.array(bckg_pd['rec'].values[:-test_recs_bckg])
                    transfer = np.append(transfer_seiz, transfer_bckg)
            
                transfer_records[subj] = dict()
                transfer_records[subj]['seiz'] = transfer_seiz
                transfer_records[subj]['bckg'] = transfer_bckg

            test_records[subj] = test
            transfer_subjects.append(subj)

    return transfer_subjects, transfer_records, test_records


def get_inter_subjects(hdf5_path, subjects, seiz_classes, seed,
                       test_frac = 1/3, **kwargs):
    '''
    Function for splitting subjects into seizure records to use
    for transferring knowledge and into testing 
    '''
    file = dc.File(hdf5_path, 'r')
    train_subjects = []
    train_records = dict()
    test_records = dict()
    for subj in subjects:
        subject = file[subj]
        seiz_recs = get_seiz_recs(subject, seiz_classes)
        train_recs_seiz = np.array([])
        test_recs_seiz = np.array([])
        test_recs_bckg = np.array([])
        train_recs_bckg = np.array([])

        if len(seiz_recs['seiz']['rec']) > 1:
            # randomnly choose test_frac of seizure records to use for testing
            n_seiz = int(np.ceil(len(seiz_recs['seiz']['rec'])*test_frac))
            train_recs_seiz, test_recs_seiz  = train_test_split(seiz_recs['seiz']['rec'], 
                                                                test_size = n_seiz,
                                                                random_state = seed)
        elif len(seiz_recs['seiz']['rec']) == 1:
            train_recs_seiz = np.array(seiz_recs['seiz']['rec'])

        if len(seiz_recs['non seiz']['rec']) > 1:
            n_bckg = int(np.ceil(test_frac*len(seiz_recs['non seiz']['rec'])*test_frac))
            train_recs_bckg, test_recs_bckg  = train_test_split(seiz_recs['non seiz']['rec'], 
                                                                test_size = n_bckg,
                                                                random_state = seed)
        elif len(seiz_recs['non seiz']['rec']) == 0:
            train_recs_bckg = np.array(seiz_recs['non seiz']['rec'])

        train_recs = np.append(train_recs_seiz, train_recs_bckg)
        test_recs = np.append(test_recs_seiz, test_recs_bckg)
        train_records[subj] = train_recs
        test_records[subj] = test_recs

    return train_records, test_records


def get_seiz_recs(subject, seiz_classes, pickle_path=None):
    seiz_recs = dict()
    seiz_recs['seiz'] = {'rec': [], 'seiz dur': [], 'bckg dur': []}
    seiz_recs['non seiz'] = {'rec': [], 'bckg dur': []}

    for rec in subject.keys():
        seiz = 0
        annos = subject[rec]['Annotations']
        seiz_dur = 0
        bckg_dur = 0

        for anno in annos:
            if anno['Name'].lower() in seiz_classes:
                seiz_dur += anno['Duration']
                seiz = 1
            elif anno['Name'].lower() == 'bckg':
                bckg_dur += anno['Duration']

        if seiz == 1:
            seiz_recs['seiz']['rec'].append(rec)
            seiz_recs['seiz']['seiz dur'].append(seiz_dur)
            seiz_recs['seiz']['bckg dur'].append(bckg_dur)
        else:
            seiz_recs['non seiz']['rec'].append(rec)
            seiz_recs['non seiz']['bckg dur'].append(bckg_dur)

        if pickle_path is not None:
            with open(pickle_path, 'wb') as fp:
                pickle.dump(seiz_recs, fp)

    return seiz_recs

