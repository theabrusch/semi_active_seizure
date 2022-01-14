from dataapi import data_collection as dc
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import warnings

class DataGenerator(Dataset):
    def __init__(self, 
                 hdf5_path, 
                 signal_name,
                 subjects_to_use,
                 return_seiz_type = False,
                 norm_coef = None,
                 segments = None,
                 standardise = True,
                 bckg_rate = None, 
                 prefetch_data_from_seg = False,
                 **kwargs):
        '''
        Wrapper for the Pytorch dataset that takes the segmentation and seizes the samples.
        ------------------------------
        hdf5_path: str
            Path to hdf5 file that should be used to build the
            generator. 
        signal_name: str
            Name of the signal to segment for training. 
        prefetch_data_from_seg: bool
            If True, the data is prefetched from a precomputed 
            segmentation. 
        '''

        self.hdf5_path = hdf5_path
        self.signal_name = signal_name
        self.standardise = standardise
        self.data_file = dc.File(hdf5_path, 'r')
        self.bckg_rate = bckg_rate
        self.subjects_to_use = subjects_to_use
        self.prefetch_data_from_seg = prefetch_data_from_seg
        self.return_seiz_type = return_seiz_type
    
        # Check if the normalisation coefficients have been calculated and saved
        if standardise:
            self.norm_coef = norm_coef
            
        self.segments = segments

        self.bckg_samples = len(self.segments['bckg'])
        self.seiz_samples = len(self.segments['seiz'])

        # Set the background rate to the 
        if self.bckg_rate == 'None' or self.bckg_rate is None:
            self.bckg_rate = self.bckg_samples/self.seiz_samples
        elif self.bckg_rate > self.bckg_samples/self.seiz_samples:
            print('Background rate is too high compared to ratio.',
                  'Setting background rate to', 
                  self.bckg_samples/self.seiz_samples, '.')
            self.bckg_rate = self.bckg_samples/self.seiz_samples

        # Create weights for sampling. If background rate is 
        # 1 a batch should contain 50% background and 50% 
        # seizure. 
        bckg_weight = 1/self.bckg_samples*self.bckg_rate
        seiz_weight = 1/self.seiz_samples

        self.segments['bckg']['weight'] = bckg_weight
        self.segments['seiz']['weight'] = seiz_weight
        
        samptemp = self.segments['seiz'].append(self.segments['bckg'], 
                                                    ignore_index = True)
        if self.prefetch_data_from_seg:                
            print('Starting prefetch of data from segmentation...')
            samples = self._prefetch_from_seg(samptemp)
        else:
            samples = samptemp.to_records(index = False)

        self.samples = samples


    def __len__(self):
        '''
        Set the number of samples in the dataset relative to the 
        number of seizures and the bckg_rate such that the seizures
        are not heavily oversampled in each epoch. This is passed 
        to the sampler. 
        ''' 
        return int((1+self.bckg_rate)*self.seiz_samples)

    def __getitem__(self, idx):
        '''
        Get segment based on idx. This is the function
        that the dataloader calls when sampling during 
        training. 
        '''
        item = self.samples[idx]
        if self.prefetch_data_from_seg:
            # if data has been prefetched, simply take the 
            # sample
            sample = item[0]
            label = int(item[1])
            seiz_type = item[2]
        else:
            # if data has not been prefetched, get the sample
            # using the start and end of the segment
            label = int(item['label'])
            sample = self._get_segment(item)
            seiz_type = item['seiz_types']

        if self.return_seiz_type:
            return sample, label, seiz_type
        else:
            return sample, label
    
    def _get_segment(self, item):
        '''
        Get EEG segment based on the path and start and end samples
        given in item.
        --------------
        item: pd.DataFrame
            Row from pd.DataFrame containing information about the
            segment to be sampled. 
        '''
        sig = self.data_file[item['path']]
        seg = sig[item['startseg']:item['endseg'],:]
        # Standardise with respect to record
        if self.standardise:
            mean = np.mean(seg)
            std = np.std(seg)
            seg = (seg-mean)/std
            #mean = self.norm_coef[item['path']]['mean']
            #std = self.norm_coef[item['path']]['std']
            #seg = (seg-mean)/std
        
        return seg.T
    
    def _prefetch_from_seg(self, seg):
        '''
        If segmentation has already been done,
        the data can be prefetched by calling
        _get_segment() and saving all samples in 
        memory.
        Inputs
        ----------------------------------------
        seg: pd.DataFrame
            Segmentation to use for getting samples. 
        
        Outputs
        ----------------------------------------
        samples: list 
            List of tuples where each tuple is 
            (sample, label). 
        '''
        samples = []

        for i in range(len(seg)):
            if (i+1)%1000 == 0:
                print('Prefetching segment', (i+1), 'out of', len(seg))
            item = seg.loc[i,:]
            sample = self._get_segment(item)
            label = item['label']
            seiz_type = item['seiz_types']
            samples.append((sample, label, seiz_type))
        
        # close data file since it is no longer needed
        self.data_file.close()
        return samples
    
    def _get_X_shape(self):
        temp = self.__getitem__(0)
        return temp[0].shape
    

class TestGenerator(Dataset):
    def __init__(self, 
                 hdf5_path, 
                 signal_name,
                 window_length, 
                 seiz_classes,
                 standardise = False,
                 bckg_stride = None,
                 seiz_stride = None, 
                 bckg_rate = None, 
                 subjects_to_use = 'all', 
                 prefetch_data_dir = True,
                 prefetch_data_from_seg = False,
                 **kwargs):
        '''
        Wrapper for the Pytorch dataset that segments and samples the 
        EEG records according to the window length.
        ------------------------------
        hdf5_path: str
            Path to hdf5 file that should be used to build the
            generator. 
        signal_name: str
            Name of the signal to segment for training. 
        window_length: float
            Length of windows for the data to be segmented into
        stride: float or None
            Stride between windows to creat overlap. If None is given, 
            the stride is set to window length (ie. no overlap).
        bckg_rate: int or None
            Number of background segments to include in the dataset 
            per seizure segment. If None all background examples are 
            used. 
        anno_based_seg: bool
            If True, the segmentation is based on the annotations in
            the dataset. If False, the segmentation starts from the
            beginning of each record. 
        subjects_to_use: list or 'all'
            If list, the subjects in the list are included in the 
            dataset. If not a list, all subjects are included. 
        prefetch_data_dir: bool
            If True, the segmentation is computed from scratch and
            the samples are prefetched and saved in memory. 
        prefetch_data_from_seg: bool
            If True, the data is prefetched from a precomputed 
            segmentation. 
        '''

        self.hdf5_path = hdf5_path
        self.data_file = dc.File(hdf5_path, 'r')
        self.window_length = window_length
        if bckg_stride is None and seiz_stride is None:
            self.stride = self.window_length
        elif seiz_stride is None:
            self.stride = bckg_stride
        elif bckg_stride is None:
            self.stride = seiz_stride
        else:
            self.stride = [bckg_stride, seiz_stride]
        self.signal_name = signal_name
        self.subjects_to_use = subjects_to_use
        self.prefetch_data_dir = prefetch_data_dir
        self.prefetch_data_from_seg = prefetch_data_from_seg
        self.bckg_rate = bckg_rate
        self.seiz_classes = seiz_classes
        self.standardise = standardise
    
        print('Starting prefetch of data directly from records.')
        self.segments, self.labels_collect, self.paths = self._prefetch()

    def __len__(self):
        '''
        Set the number of samples in the dataset relative to the 
        number of seizures and the bckg_rate such that the seizures
        are not heavily oversampled in each epoch. This is passed 
        to the sampler. 
        ''' 
        return len(self.segments)

    def __getitem__(self, idx):
        '''
        Get segment based on idx. This is the function
        that the dataloader calls when sampling during 
        training. 
        '''
        
        if self.prefetch_data_dir or self.prefetch_data_from_seg:
            # if data has been prefetched, simply take the 
            # sample
            sample = self.segments[idx]
            label = int(self.labels_collect[idx])
        else:
            # if data has not been prefetched, get the sample
            # using the start and end of the segment
            item = self.samples[idx]
            label = int(item['label'])
            sample = self._get_segment(item)

        return sample, label
    
    def _prefetch(self):
        '''
        If the segmentation has not already been computed, 
        the prefetching can be done directly. 
        Outputs
        -------------------------------------------------
        segments: dict
            Dictionary with all samples. Divided into 
            seizure and background. 
        '''

        segments = []
        
        i=0
        for subj in self.subjects_to_use:
            print('Segmenting data for subject', i + 1, 'out of', len(self.subjects_to_use))
            i+=1
            for rec in self.data_file[subj].keys():
                record = self.data_file[subj][rec]

                for sig in self.signal_name:
                    if sig in record.keys():
                        signal = record[sig]

                # Calculate normalisation coefficients for each record
                mean = np.mean(signal)
                std = np.std(signal)
                
                labels, samples = self._record_based_segment(record, prefetch = True)
                path = np.empty(len(labels), dtype = str)
                path[:] =  subj + '/' + rec + '/' + sig

                if self.standardise:
                    samples = (samples - mean)/std

                if len(segments)>0:
                    segments= np.append(segments, samples, axis = 0)
                    labels_collect = np.append(labels_collect, labels, axis = 0) 
                    paths = np.append(paths, path)
                else:
                    segments = samples
                    labels_collect = labels
                    paths = path

        self.data_file.close()
        return segments, labels_collect, paths

    def _get_X_shape(self):
        temp = self.__getitem__(0)
        return temp[0].shape
    
    def _segment_data(self, calc_norm_coef):
        '''
        Build pandas DataFrame containing pointers to the different
        segments to sample when generating data. 
        '''
        protocol = self.data_file[self.protocol]

        segments = dict() 
        segments['seiz'] = pd.DataFrame()
        segments['bckg'] = pd.DataFrame()
        labels = ['bckg', 'seiz']

        if calc_norm_coef:
            self.norm_coef = dict()
        
        i=0
        for subj in protocol.keys():
            print('Segmenting data for subject', i + 1, 'out of', len(protocol.keys()))
            i+=1
            for rec in protocol[subj].keys():
                record = protocol[subj][rec]
                for sig in self.signal_name:
                    if sig in record.keys():
                        signal = record[sig]
                        signal_name = sig
                path = self.protocol + '/' + subj + '/' + rec + '/' + signal_name

                # Calculate normalisation coefficients for each record
                if calc_norm_coef:
                    mean = np.mean(signal)
                    std = np.std(signal)
                
                if self.anno_based_seg:
                    labels, start_win, end_win = self._anno_based_segment(record)
                else:
                    labels, start_win, end_win = self._record_based_segment(record)
                    
                seg_rec = pd.DataFrame({'startseg': start_win.astype(int), 
                                        'endseg': end_win.astype(int), 
                                        'label': labels})
                seg_rec['path'] = path
                seg_rec['subj'] = subj
                seg_rec['rec'] = subj + '/' + rec
                if calc_norm_coef:
                    self.norm_coef[path] = dict()
                    self.norm_coef[path]['mean'] = mean
                    self.norm_coef[path]['std'] = std

                seg_rec_pos = seg_rec[seg_rec['label'] == 1]
                seg_rec_neg = seg_rec[seg_rec['label'] == 0]
                segments['seiz'] = segments['seiz'].append(seg_rec_pos)
                segments['bckg'] = segments['bckg'].append(seg_rec_neg)


        # save segmentation as pickle for future use
        with open(self.pickle_path, 'wb') as fp:
            pickle.dump(segments, fp)
        
        if calc_norm_coef:
            with open(self.norm_coef_path, 'wb') as fp:
                pickle.dump(self.norm_coef, fp)

        return segments
    
    def _record_based_segment(self, record, prefetch=False):
        # Get annotation on sample basis
        one_hot_label = self._anno_to_one_hot(record)
        for sig in self.signal_name:
            if sig in record.keys():
                signal = record[sig]
        channels = len(signal.attrs['chNames'])
        
        windows = int((record.duration-self.window_length)/self.stride)+1
        window_samples = int(self.window_length*signal.fs)
        stride_samples = int(self.stride*signal.fs)

        if prefetch:
            samples = np.zeros((windows, channels, window_samples))
            labels = np.zeros(windows)
        else:
            labels = np.zeros(windows)
            start_win = np.zeros(windows)
            end_win = np.zeros(windows)

        for win in range(windows):
            sw =int(win * stride_samples)
            ew = int(sw + window_samples)
            if prefetch:
                samples[win,:,:] = signal[sw:ew,:].T
            else: 
                start_win[win] = sw
                end_win[win] = ew
            # set label to seizure if any seizure is present in the segment
            labels[win] = int(np.sum(one_hot_label[sw:ew,:], axis = 0)[1]>0)
        if prefetch:
            return labels, samples
        else:
            return labels, start_win, end_win
    
    def _anno_to_one_hot(self, record):
        '''
        Create one hot encoding of annotations
        '''
        for sig in self.signal_name:
            if sig in record.keys():
                signal = record[sig]
        annos = record['Annotations']
        one_hot_label = np.zeros((len(signal), 2))
        one_hot_label[:, 0] = 1

        for anno in annos:
            anno_start = (anno['Start'] - record.start_time)*signal.fs
            anno_end = anno_start+anno['Duration']*signal.fs
            if anno['Name'].lower() in self.seiz_classes:
                one_hot_label[round(anno_start):round(anno_end),1] = 1
                one_hot_label[:, 0] = 0
            else:
                one_hot_label[round(anno_start):round(anno_end),0] = 1
        
        return one_hot_label

class SegmentData():
    
    def __init__(self, 
                 hdf5_path, 
                 signal_name,
                 window_length, 
                 seiz_classes,
                 bckg_rate,
                 sens = 0,
                 subj_strat = False,
                 use_train_seed = False,
                 standardise = True,
                 bckg_stride = None,
                 seiz_stride = None, 
                 anno_based_seg = False,
                 subjects_to_use = 'all',
                 **kwargs):
        '''
        Wrapper for the segmenting the data.
        ------------------------------
        hdf5_path: str
            Path to hdf5 file that should be used to build the
            generator. 
        signal_name: str
            Name of the signal to segment for training. 
        window_length: float
            Length of windows for the data to be segmented into
        stride: float or None
            Stride between windows to creat overlap. If None is given, 
            the stride is set to window length (ie. no overlap).
        anno_based_seg: bool
            If True, the segmentation is based on the annotations in
            the dataset. If False, the segmentation starts from the
            beginning of each record. 
        subjects_to_use: list or 'all'
            If list, the subjects in the list are included in the 
            dataset. If not a list, all subjects are included. 
        '''

        self.hdf5_path = hdf5_path
        self.use_train_seed = use_train_seed
        self.data_file = dc.File(hdf5_path, 'r')
        self.window_length = window_length
        self.bckg_rate = bckg_rate
        self.subj_strat = subj_strat

        if bckg_stride is None and seiz_stride is None:
            self.stride = self.window_length
        elif seiz_stride is None:
            self.stride = bckg_stride
        elif bckg_stride is None:
            self.stride = seiz_stride
        else:
            self.stride = [bckg_stride, seiz_stride]

        self.subjects_to_use = subjects_to_use
        self.anno_based_seg = anno_based_seg
        self.seiz_classes = seiz_classes
        self.standardise = standardise
        self.sens = sens
        if isinstance(self.stride, list) and not anno_based_seg:
            self.stride = self.stride[0]
            warnings.warn('The segmentation is not based on annotations '+\
                          'so the stride is set to ' + str(self.stride), UserWarning)
                        

        # Define paths for saving the segmentation
        self.signal_name = signal_name
        dset = hdf5_path.split('/')[-1].split('.')[0]
        stride_string = ''.join(str(self.stride).split(' '))
        self.pickle_path = 'data/' + dset  + \
                           '_winlen_' + str(window_length) + '_anno_seg_'\
                           + str(anno_based_seg)+'_stride_' + stride_string + \
                            'seiz_classes_' + str(self.seiz_classes)
        self.norm_coef_path = 'data/' + dset + \
                              '_norm_coef.pickle'
        if self.standardise:
            self.calc_norm_coef = True
        else:
            self.calc_norm_coef = False

    def segment_data(self):
        '''
        Build pandas DataFrame containing pointers to the different
        segments to sample when generating data. 
        '''
        segments = dict() 
        segments['seiz'] = pd.DataFrame()
        segments['bckg'] = pd.DataFrame()
        labels = ['bckg', 'seiz']

        if self.calc_norm_coef:
            self.norm_coef = dict()
        else:
            self.norm_coef = None
        
        i=0
        if not isinstance(self.subjects_to_use, list) and not isinstance(self.subjects_to_use, np.ndarray):
            subjects = self.data_file.get_children(object_type=dc.Subject, get_obj = False)
        else:
            subjects = self.subjects_to_use

        for subj in subjects:
            print('Segmenting data for subject', i + 1, 'out of', len(subjects))
            i+=1
            subj_name = subj.split('/')[-1]
            subj_path = self.pickle_path + '_' + subj_name + '.pickle'
            compute_subj = False
            try:
                with open(subj_path, 'rb') as fp:
                    subj_seg = pickle.load(fp)
            except:
                compute_subj = True
            
            if compute_subj:
                subj_seg = dict()
                subj_seg['seiz'] = pd.DataFrame()
                subj_seg['bckg'] = pd.DataFrame()
                subj_seg['norm_coef'] = dict()
                for rec in self.data_file[subj].keys():
                    record = self.data_file[subj][rec]
                    for sig in self.signal_name:
                        if sig in record.keys():
                            signal = record[sig]
                            signal_name = sig
                    path =  subj + '/' + rec + '/' + signal_name

                    # Calculate normalisation coefficients for each record
                    mean = np.mean(signal)
                    std = np.std(signal)
                    
                    if self.anno_based_seg:
                        labels, start_win, end_win, seiz_types = self._anno_based_segment(record)
                    else:
                        labels, start_win, end_win, seiz_types = self._record_based_segment(record)
                        
                    seg_rec = pd.DataFrame({'startseg': start_win.astype(int), 
                                            'endseg': end_win.astype(int), 
                                            'label': labels, 
                                            'seiz_types': seiz_types})
                    seg_rec['path'] = path
                    seg_rec['subj'] = subj
                    seg_rec['rec'] = subj + '/' + rec
                    subj_seg['norm_coef'][path] = dict()
                    subj_seg['norm_coef'][path]['mean'] = mean
                    subj_seg['norm_coef'][path]['std'] = std

                    seg_rec_pos = seg_rec[seg_rec['label'] == 1]
                    seg_rec_neg = seg_rec[seg_rec['label'] == 0]
                    subj_seg['seiz'] = subj_seg['seiz'].append(seg_rec_pos)
                    subj_seg['bckg'] = subj_seg['bckg'].append(seg_rec_neg)

                with open(subj_path, 'wb') as fp:
                    pickle.dump(subj_seg, fp)

            if self.calc_norm_coef:
                self.norm_coef.update(subj_seg['norm_coef'])
            
            segments['seiz'] = segments['seiz'].append(subj_seg['seiz'])
            if not self.bckg_rate is None and self.use_train_seed and self.subj_strat:
                seiz_samples = len(subj_seg['seiz'])
                bckg_tot = int(self.bckg_rate*seiz_samples)
                segments['bckg'] = segments['bckg'].append(subj_seg['bckg'].sample(n=bckg_tot))
            else:
                segments['bckg'] = segments['bckg'].append(subj_seg['bckg'])
        
        self.segments = segments

        if not self.bckg_rate is None and self.use_train_seed and not self.subj_strat:
            seiz_samples = len(self.segments['seiz'])
            bckg_tot = int(self.bckg_rate*seiz_samples)
            self.segments['bckg'] = self.segments['bckg'].sample(n = bckg_tot)

        return self.segments, self.norm_coef
    
    def _record_based_segment(self, record):
        # Get annotation on sample basis
        one_hot_label, classes = self._anno_to_one_hot(record)
        for sig in self.signal_name:
            if sig in record.keys():
                signal = record[sig]
        channels = len(signal.attrs['chNames'])
        
        windows = int((record.duration-self.window_length)/self.stride)+1
        window_samples = int(self.window_length*signal.fs)
        stride_samples = int(self.stride*signal.fs)


        labels = np.array([])
        start_win = np.array([])
        end_win = np.array([])
        seiz_types = np.array([])
        incl_seiz = False
        excl_seiz = False

        for win in range(windows):
            sw =int(win*stride_samples)
            ew = int(sw + window_samples)
            use_sample = False
            if np.sum(one_hot_label[sw:ew,:], axis = 0)[1]>window_samples*self.sens:
                lab = 1
                use_sample = True
                incl_seiz = True
            elif np.sum(one_hot_label[sw:ew,:], axis = 0)[0]>0.95*window_samples:
                lab = 0
                use_sample = True
            else:
                excl_seiz = True

            if use_sample:
                if lab == 0:
                    seiz_type  = 'bckg'
                else:
                    class_types = np.unique(classes[sw:ew])
                    if len(class_types) > 1 and 'bckg' in class_types:
                        class_types = [cl for cl in class_types if cl != 'bckg']
                    seiz_type = class_types[0]


                start_win = np.append(start_win, np.array([sw]), axis = 0)
                end_win = np.append(end_win, np.array([ew]), axis = 0)
                labels = np.append(labels, np.array([lab]), axis = 0)
                seiz_types = np.append(seiz_types, np.array([seiz_type]), axis = 0)
        
        if excl_seiz and not incl_seiz:
            # if the only contains excluded seizure types and no included seizure types
            labels = np.array([])
            start_win = np.array([])
            end_win = np.array([])
            seiz_types = np.array([])
            
        return labels, start_win, end_win, seiz_types
    
    def _anno_to_one_hot(self, record):
        '''
        Create one hot encoding of annotations
        '''
        for sig in self.signal_name:
            if sig in record.keys():
                signal = record[sig]
        annos = record['Annotations']
        one_hot_label = np.zeros((len(signal), 2))
        seiz_types = np.empty(len(signal), object)

        for anno in annos:
            anno_start = (anno['Start'] - record.start_time)*signal.fs
            anno_end = anno_start+anno['Duration']*signal.fs
            seiz_types[int(np.floor(anno_start)):int(np.ceil(anno_end))] = anno['Name'].lower()
            if anno['Name'].lower() in self.seiz_classes:
                one_hot_label[int(np.floor(anno_start)):int(np.floor(anno_end)),1] = 1
                one_hot_label[int(np.floor(anno_start)):int(np.floor(anno_end)),0] = 0
            elif anno['Name'].lower() == 'bckg':
                one_hot_label[int(np.floor(anno_start)):int(np.floor(anno_end)),0] = 1
        
        return one_hot_label, seiz_types

    def _anno_based_segment(self, record):
        for sig in self.signal_name:
            if sig in record.keys():
                signal = record[sig]
        channels = len(signal.attrs['chNames'])
        annos = record['Annotations']

        if isinstance(self.stride, int):
            stride = [self.stride, self.stride]
        else:
            stride = self.stride
        excl_seiz = False
        incl_seiz = False
        i = 0
        for anno in annos:
            anno_start = int((anno['Start'] - record.start_time)*signal.fs)
            window_samples = self.window_length*signal.fs
            use_anno = False
            if anno['Name'].lower() in self.seiz_classes:
                anno_stride = stride[1]
                windows = (anno['Duration']-self.window_length)/anno_stride + 1
                lab = 1
                incl_seiz = True
                use_anno = True
            elif anno['Name'].lower() == 'bckg':
                anno_stride = stride[0]
                windows = (anno['Duration']-self.window_length)/anno_stride + 1
                lab = 0
                use_anno = True
            else: 
                excl_seiz = True

            if use_anno:
                stride_samples = anno_stride*signal.fs
                windows = int(windows)

                if windows <= 0:
                    print('Annotation', anno['Name'], 'in record', record.name, 'is too short for selected window length.')
                    label = np.array([])
                    seiz_type = np.array([])
                    sw = np.array([])
                    ew = np.array([])
                else:
                    label = np.zeros(windows)
                    label[:] = lab
                    seiz_type = [anno['Name']]*windows
                    sw = anno_start + np.array([win*stride_samples for win in range(windows)])
                    ew = sw + window_samples
            
                if i == 0:
                    start_win = sw
                    end_win = ew
                    labels = label
                    seiz_types = np.array(seiz_type)
                else:
                    start_win = np.append(start_win, sw)
                    end_win = np.append(end_win, ew)
                    labels = np.append(labels, label)
                    seiz_types = np.append(seiz_types, seiz_type)
                i+=1

        if excl_seiz and not incl_seiz:
            # if the only contains excluded seizure types and no included seizure types
            labels = np.array([])
            start_win = np.array([])
            end_win = np.array([])
            seiz_types = np.array([])

        return labels, start_win, end_win, seiz_types
