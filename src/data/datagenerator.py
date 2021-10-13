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
                 protocol, 
                 signal_name,
                 window_length, 
                 seiz_classes,
                 stride = None, 
                 bckg_rate = None, 
                 anno_based_seg = False,
                 subjects_to_use = 'all', 
                 prefetch_data_dir = False,
                 prefetch_data_from_seg = False,
                 **kwargs):
        '''
        Wrapper for the Pytorch dataset that segments and samples the 
        EEG records according to the window length.
        ------------------------------
        hdf5_path: str
            Path to hdf5 file that should be used to build the
            generator. 
        protocol: str
            Train or test
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
        if stride is None:
            self.stride = self.window_length
        else:
            self.stride = stride

        self.subjects_to_use = subjects_to_use
        self.protocol = protocol
        self.prefetch_data_dir = prefetch_data_dir
        self.prefetch_data_from_seg = prefetch_data_from_seg
        self.bckg_rate = bckg_rate
        self.anno_based_seg = anno_based_seg
        self.seiz_classes = seiz_classes
        
        if isinstance(self.stride, list) and not anno_based_seg:
            self.stride = self.stride[0]
            warnings.warn('The segmentation is not based on annotations '+\
                          'so the stride is set to ' + str(self.stride), UserWarning)
                        

        # Define paths for saving the segmentation
        self.signal_name = signal_name
        dset = hdf5_path.split('/')[-1].split('.')[0]
        stride_string = ''.join(str(stride).split(' '))
        self.pickle_path = 'data/' + dset + '_' + protocol + '_'\
                           + signal_name + '_winlen_' + str(window_length) + '_anno_seg_'\
                           + str(anno_based_seg)+'_stride_' + stride_string + '.pickle'
        self.norm_coef_path = 'data/' + dset + '_' + protocol + '_'\
                              + signal_name + '_norm_coef.pickle'
        calc_norm_coef = False

        if self.prefetch_data_dir and self.prefetch_data_from_seg:
            warn_str = 'prefetch_data_dir and prefetch_data_from_seg' + \
                       'cannot both be true. Setting prefetch_data_dir' + \
                       'to False.'
            warnings.warn(warn_str, UserWarning)
            self.prefetch_data_from_seg = False

        # Check if the normalisation coefficients have been calculated and saved
        try:
            with open(self.norm_coef_path, 'rb') as fp:
                self.norm_coef = pickle.load(fp)
        except:
            calc_norm_coef = True

        # Check if the segmentation has been computed and saved
        if not self.prefetch_data_dir:
            try:
                print('Trying to load segmentation from disk.')
                with open(self.pickle_path, 'rb') as fp:
                    self.segments = pickle.load(fp)
                print('Succesfully loaded segmentation.')
            except:
                print('Segmentation not computed, starting computation of segmentation.')
                self.segments = self._segment_data(calc_norm_coef)
            if isinstance(subjects_to_use, list) or isinstance(subjects_to_use, np.ndarray):
                self.segments['bckg'] = self.segments['bckg'][self.segments['bckg']['subj'].isin(subjects_to_use)]
                self.segments['seiz'] = self.segments['seiz'][self.segments['seiz']['subj'].isin(subjects_to_use)]
            self.bckg_samples = len(self.segments['bckg'])
            self.seiz_samples = len(self.segments['seiz'])
        else:
            print('Starting prefetch of data directly from records.')
            self.segments = self._prefetch()
            self.bckg_samples = len(self.segments['bckg']['samples'])
            self.seiz_samples = len(self.segments['seiz']['samples'])   

        # Set the background rate to the 
        if self.bckg_rate == 'None':
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

        if not self.prefetch_data_dir:
            self.segments['bckg']['weight'] = bckg_weight
            self.segments['seiz']['weight'] = seiz_weight

            # Create collected sample matrix
            segments = self.segments['seiz'].append(self.segments['bckg'], 
                                                   ignore_index = True)
            samptemp = shuffle(segments).reset_index()
            self.weights = samptemp['weight']
            if self.prefetch_data_from_seg:
                print('Starting prefetch of data from segmentation...')
                samples = self._prefetch_from_seg(samptemp)
            else:
                samples = samptemp.to_records(index = False)
            self.samples = samples
        else:
            weights = np.zeros(self.bckg_samples + self.seiz_samples)
            weights[:self.seiz_samples] = seiz_weight
            weights[self.seiz_samples::] = bckg_weight
            samp = np.append(self.segments['seiz']['samples'], self.segments['bckg']['samples'], axis = 0)
            lab = np.append(self.segments['seiz']['label'], self.segments['bckg']['label'], axis = 0)
            self.samples = list(zip(samp, lab))
            self.samples, self.weights = shuffle(self.samples, weights)
        
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
        if self.prefetch_data_dir or self.prefetch_data_from_seg:
            # if data has been prefetched, simply take the 
            # sample
            sample = item[0]
            label = int(item[1])
        else:
            # if data has not been prefetched, get the sample
            # using the start and end of the segment
            label = int(item['label'])
            sample = self._get_segment(item)

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
        mean = self.norm_coef[item['path']]['mean']
        std = self.norm_coef[item['path']]['std']
        seg = (seg-mean)/std
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
            samples.append((sample, label))
        
        # close data file since it is no longer needed
        self.data_file.close()
        return samples

    
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
        protocol = self.data_file[self.protocol]

        segments = dict() 
        segments['seiz'] = dict()
        segments['bckg'] = dict()
        
        i=0
        for subj in self.subjects_to_use:
            print('Segmenting data for subject', i + 1, 'out of', len(protocol.keys()))
            i+=1
            for rec in protocol[subj].keys():
                record = protocol[subj][rec]
                signal = record[self.signal_name]

                # Calculate normalisation coefficients for each record
                mean = np.mean(signal)
                std = np.std(signal, axis = 0)
                
                if self.anno_based_seg:
                    labels, samples = self._anno_based_segment(record, prefetch = True)
                else:
                    labels, samples = self._record_based_segment(record, prefetch = True)

                samples = (samples - mean)/std
                pos_samples = labels == 1
                neg_samples = labels == 0

                if 'samples' in segments['seiz'].keys():
                    segments['seiz']['samples'] = np.append(segments['seiz']['samples'], samples[pos_samples], axis = 0)
                    segments['bckg']['samples'] = np.append(segments['bckg']['samples'], samples[neg_samples], axis = 0)
                    segments['seiz']['label'] = np.append(segments['seiz']['label'], labels[pos_samples], axis = 0)
                    segments['bckg']['label'] = np.append(segments['bckg']['label'], labels[neg_samples], axis = 0)
                else:
                    segments['seiz']['samples'] = samples[pos_samples]
                    segments['bckg']['samples'] = samples[neg_samples]
                    segments['seiz']['label'] = labels[pos_samples]
                    segments['bckg']['label'] = labels[neg_samples]
        
        self.data_file.close()
        return segments

    
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
                signal = record[self.signal_name]
                path = self.protocol + '/' + subj + '/' + rec + '/' + self.signal_name

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
        signal = record[self.signal_name]
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
            sw =int(win*stride_samples)
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
        signal = record[self.signal_name]
        annos = record['Annotations']
        one_hot_label = np.zeros((len(signal), 2))

        for anno in annos:
            anno_start = (anno['Start'] - record.start_time)*signal.fs
            anno_end = anno_start+anno['Duration']*signal.fs
            if anno['Name'].lower() in self.seiz_classes:
                one_hot_label[round(anno_start):round(anno_end),1] = 1
            else:
                one_hot_label[round(anno_start):round(anno_end),0] = 1
        
        return one_hot_label

    def _anno_based_segment(self, record, prefetch=False):
        signal = record[self.signal_name]
        channels = len(signal.attrs['chNames'])
        annos = record['Annotations']

        if isinstance(self.stride, int):
            stride = [self.stride, self.stride]
        else:
            stride = self.stride

        i = 0
        for anno in annos:
            anno_start = int((anno['Start'] - record.start_time)*signal.fs)
            window_samples = self.window_length*signal.fs

            if anno['Name'].lower() in self.seiz_classes:
                anno_stride = stride[1]
                windows = (anno['Duration']-self.window_length)/anno_stride + 1
                lab = 1
            else:
                anno_stride = stride[0]
                windows = (anno['Duration']-self.window_length)/anno_stride + 1
                lab = 0

            stride_samples = anno_stride*signal.fs
            if windows%1 != 0:
                if int(anno_start - ((windows%1)*signal.fs)/2) > 0:
                    anno_start = int(anno_start - ((windows%1)*signal.fs)/2)
                if anno_start + np.ceil(windows)*stride_samples + window_samples < record.duration*signal.fs:
                    windows = int(np.ceil(windows))
                else:
                    windows = int(windows)
            else:
                windows = int(windows)

            if windows <= 0:
                print('Annotation', anno['Name'], 'in record', record.name, 'is too short for selected window length.')
                label = np.array([])
            else:
                label = np.zeros(windows)
                label[:] = lab
            
            sw = anno_start + np.array([win*stride_samples for win in range(windows)])
            ew = sw + window_samples

            if prefetch:
                samples = np.zeros((windows, channels, window_samples))
                for win in range(windows):
                    samples[win,:,:] = signal[sw[win]:ew[win],:].T
        
            if i == 0:
                if prefetch:
                    samples_collect = samples
                else:
                    start_win = sw
                    end_win = ew
                labels = label
            else:
                if prefetch:
                    samples_collect = np.append(samples_collect, samples, axis = 0)
                else:
                    start_win = np.append(start_win, sw)
                    end_win = np.append(end_win, ew)
                labels = np.append(labels, label)
            i+=1
        
        if prefetch:
            return labels, samples_collect
        else:
            return labels, start_win, end_win