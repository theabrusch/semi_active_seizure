from dataapi import data_collection as dc
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import warnings

class DataGenerator(Dataset):
    def __init__(self, hdf5_path, window_length, protocol, signal_name,
                 stride=None, bckg_rate = None, anno_based_seg = False):
        '''
        Wrapper for the Pytorch dataset that segments and samples the 
        EEG records according to the window length.
        ------------------------------
        hdf5_path: str
            Path to hdf5 file that should be used to build the
            generator
        window_length: float
            Length of windows for the data to be segmented into
        protocol: str
            Train or test
        signal_name: str
            Name of the signal to segment for training. 
        bckg_rate: int or None
            Number of background segments to include in the dataset 
            per seizure segment. If None all background examples are 
            used. 
        '''

        self.hdf5_path = hdf5_path
        self.data_file = dc.File(hdf5_path, 'r')
        self.window_length = window_length
        if stride is None:
            self.stride = self.window_length
        else:
            self.stride = stride

        self.protocol = protocol
        self.bckg_rate = bckg_rate
        self.anno_based_seg = anno_based_seg
        
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

        # Check if the normalisation coefficients have been calculated and saved
        try:
            with open(self.norm_coef_path, 'rb') as fp:
                self.norm_coef = pickle.load(fp)
        except:
            calc_norm_coef = True

        # Check if the segmentation has been computed and saved
        try:
            with open(self.pickle_path, 'rb') as fp:
                self.segments = pickle.load(fp)
        except:
            self.segments = self._segment_data(calc_norm_coef)
        
        self.bckg_samples = len(self.segments['bckg'])
        self.seiz_samples = len(self.segments['seiz'])

        # Set the background rate to the 
        if self.bckg_rate is None:
            self.bckg_rate = self.bckg_samples/self.seiz_samples
        elif self.bckg_rate > self.bckg_samples/self.seiz_samples:
            self.bckg_rate = self.bckg_samples/self.seiz_samples

        # Create weights for sampling. If background rate is 
        # 1 a batch should contain 50% background and 50% 
        # seizure. 
        bckg_weight = 1/self.bckg_samples*self.bckg_rate
        seiz_weight = 1/self.seiz_samples

        self.segments['bckg']['weight'] = bckg_weight
        self.segments['seiz']['weight'] = seiz_weight

        # Create collected sample matrix
        self.samples = self.segments['seiz'].append(self.segments['bckg'], 
                                                    ignore_index = True)
        self.samples = shuffle(self.samples).reset_index()
        
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
        item = self.samples.iloc[idx]
        label = int(item['label'])
        sample = self._get_segment(item)

        return sample.T, label
    
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
        mean = np.expand_dims(self.norm_coef[item['path']]['mean'], 0)
        std = np.expand_dims(self.norm_coef[item['path']]['std'], 0)
        seg = (seg-mean)/std
        return seg
    
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
            print('Segmenting data for subject', i, 'out of', len(protocol.keys()))
            i+=1
            for rec in protocol[subj].keys():
                record = protocol[subj][rec]
                signal = record[self.signal_name]
                path = self.protocol + '/' + subj + '/' + rec + '/' + self.signal_name

                # Calculate normalisation coefficients for each record
                if calc_norm_coef:
                    mean = np.mean(signal, axis = 0)
                    std = np.std(signal, axis = 0)
                
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
    
    def _record_based_segment(self, record):
        # Get annotation on sample basis
        one_hot_label = self._anno_to_one_hot(record)
        signal = record[self.signal_name]
        
        windows = int((record.duration-self.window_length)/self.stride)+1
        window_samples = self.window_length*signal.fs
        stride_samples = self.stride*signal.fs

        labels = np.zeros(windows)
        start_win = np.zeros(windows)
        end_win = np.zeros(windows)

        for win in range(windows):
            sw = win*stride_samples
            ew = sw + window_samples
            start_win[win] = sw
            end_win[win] = ew
            # set label to seizure if any seizure is present in the segment
            labels[win] = int(np.sum(one_hot_label[sw:ew,:], axis = 0)[1]>0)
        return labels, start_win, end_win
    
    def _anno_to_one_hot(self, record):
        '''
        Create one hot encoding of annotations
        '''
        signal = record[self.signal_name]
        annos = record['Annotations']
        one_hot_label = np.zeros((len(signal), 2))
        seiz_classes = ['cpsz', 'gnsz', 'spsz', 'tcsz']

        for anno in annos:
            anno_start = (anno['Start'] - record.start_time)*signal.fs
            anno_end = anno_start+anno['Duration']*signal.fs
            if anno['Name'].lower() in seiz_classes:
                one_hot_label[round(anno_start):round(anno_end),1] = 1
            else:
                one_hot_label[round(anno_start):round(anno_end),0] = 1
        
        return one_hot_label

    def _anno_based_segment(self, record):
        signal = record[self.signal_name]
        annos = record['Annotations']
        seiz_classes = ['cpsz', 'gnsz', 'spsz', 'tcsz']
        if isinstance(self.stride, int):
            stride = [self.stride, self.stride]
        else:
            stride = self.stride

        i = 0
        for anno in annos:
            anno_start = int((anno['Start'] - record.start_time)*signal.fs)
            window_samples = self.window_length*signal.fs

            if anno['Name'].lower() in seiz_classes:
                anno_stride = stride[1]
                windows = int((anno['Duration']-self.window_length)/anno_stride + 1)
                label = np.ones(windows)
            else:
                anno_stride = stride[0]
                windows = int((anno['Duration']-self.window_length)/anno_stride + 1)
                label = np.zeros(windows)

            stride_samples = anno_stride*signal.fs
            sw = anno_start + np.array([win*stride_samples for win in range(windows)])
            ew = sw + window_samples

            if i == 0:
                start_win = sw
                end_win = ew
                labels = label
            else:
                start_win = np.append(start_win, sw)
                end_win = np.append(end_win, ew)
                labels = np.append(labels, label)
            i+=1
        
        return labels, start_win, end_win