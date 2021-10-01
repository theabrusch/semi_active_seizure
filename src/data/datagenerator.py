from dataapi import data_collection as dc
import numpy as np
import pandas as pd
import pickle

class DataGenerator():
    def __init__(self, hdf5_path, window_length, protocol, signal_name):
        '''
        hdf5_path: str
            Path to hdf5 file that should be used to build the
            generator
        window_length: float
            Length of windows for the data to be segmented into
        protocol: str
            Train or test
        '''

        self.hdf5_path = hdf5_path
        self.data_file = dc.File(hdf5_path, 'r')
        self.window_length = window_length
        self.protocol = protocol
        self.signal_name = signal_name
        dset = hdf5_path.split('/')[-1].split('.')[0]
        self.pickle_path = 'data/' + dset + '_' + protocol + '_'\
                           + signal_name + '_winlen_' + str(window_length) + '.pickle'
        self.norm_coef_path = 'data/' + dset + '_' + protocol + '_'\
                              + signal_name + '_norm_coef.pickle'
        calc_norm_coef = False

        try:
            with open(self.norm_coef_path, 'rb') as fp:
                self.norm_coef = pickle.load(fp)
        except:
            calc_norm_coef = True

        try:
            with open(self.pickle_path, 'rb') as fp:
                self.segments = pickle.load(fp)
        except:
            self.segments = self._segment_data(calc_norm_coef)
    
    def __getitem__(self, idx):
        item = self.segments.iloc[idx]
        label = int(item['label'])
        sample = self._get_segment(item)
        return sample, label
    
    def _get_segment(self, item):
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
        segments = pd.DataFrame()

        if calc_norm_coef:
            self.norm_coef = dict()
        
        i=0
        for subj in protocol.keys():
            print('Segmenting data for subject', i, 'out of', len(protocol.keys()))
            i+=1
            for rec in protocol[subj].keys():
                record = protocol[subj][rec]
                signal = record[self.signal_name]
                if calc_norm_coef:
                    mean = np.mean(signal, axis = 0)
                    std = np.std(signal, axis = 0)
                one_hot_label = self._anno_to_one_hot(record)
                windows = int(record.duration/self.window_length)
                window_samples = self.window_length*signal.fs

                path = self.protocol + '/' + subj + '/' + rec + '/' + self.signal_name
                labels = np.zeros(windows)
                start_win = np.zeros(windows)
                end_win = np.zeros(windows)

                for win in range(windows):
                    sw = win*window_samples
                    ew = (win+1)*window_samples
                    start_win[win] = sw
                    end_win[win] = ew
                    # set label to seizure if any seizure is present in the segment
                    labels[win] = int(np.sum(one_hot_label[sw:ew,:], axis = 0)[1]>0)
                
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

                segments = segments.append(seg_rec)

        # save segmentation as pickle for future use
        with open(self.pickle_path, 'wb') as fp:
            pickle.dump(segments, fp)
        
        if calc_norm_coef:
            with open(self.norm_coef_path, 'wb') as fp:
                pickle.dump(self.norm_coef, fp)

        return segments
    
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