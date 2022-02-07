import numpy as np
import pandas as pd
from collections import namedtuple

class Postprocessing():
    def __init__(self, 
                segments, 
                fs, 
                prob_thresh = 0.75, 
                dur_thresh = 1,
                post_proces = ['duration_filt', 'target_agg']):
                
        self.segments = segments
        self.fs = fs
        self.prob_thresh = prob_thresh
        self.dur_thresh = dur_thresh
        self.post_proces = post_proces

        proba = segments['seiz prob'].values
        recs = segments['path'].values
        self.annos = self.post_proces(proba, recs)

        return self.annos

    def postproces(self, proba, recs):
        uni_recs = np.unique(recs)
        for rec in uni_recs:
            rec_idx = np.array(recs) == rec
            rec_pred = proba[rec_idx]
            y_pred = self._probabilistic_filtering(rec_pred)
            if 'duration_filt' in self.post_proces:
                y_pred = self._duration_filtering(y_pred)
            annos = self._to_events(y_pred)
            if 'target_agg' in self.post_proces:
                annos = self._target_aggregation(annos)
        return annos
    
    def _probabilistic_filtering(self, probability):
        '''
        Set threshold for seizure prediction
        '''
        y_pred = (probability > self.prob_thresh).astype(int)
        return y_pred

    def _to_events(self, rec_pred):
        '''
        Convert binary events to annotation style
        '''
        anno_rec = []
        starts, lengths, values = self._rle(rec_pred)
        starts = starts*1/self.fs
        duration = lengths*1/self.fs
        for (start, dur, val) in zip(starts, duration, values):
            anno = dict({'Name': val, 'Start': start, 'Duration': dur})
            anno_rec.append(anno)
        return anno_rec
    
    def _rle(self, y):
        """ run length encoding.  
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """

        y = np.array(y)
        n = len(y)

        starts = np.r_[0, np.flatnonzero(~np.isclose(y[1:], y[:-1], equal_nan=True)) + 1]
        lengths = np.diff(np.r_[starts, n])
        values = y[starts]

        return starts, lengths, values
    
    def _target_aggregation(self, annos):
        i = 0
        while i < len(annos):
            anno = annos[i]
            if anno['Name'] == 1:
                start = anno['Start']
                dur = anno['Duration']
                w = -0.0083*dur*dur + 0.45*dur - 0.66
                end = start + dur 
                new_end = end + w
                if new_end > end:
                    for j in range(i+1, len(annos)):
                        anno_temp = annos[j]
                        temp_start = anno_temp['Start']
                        temp_end = temp_start + anno_temp['Duration']
                        if new_end <= temp_end and new_end >= temp_start:
                            if anno_temp['Name']== 1:
                                new_dur = temp_end - start
                                annos[i]['Duration'] = new_dur
                                del annos[i+1:j+1]
                                i = j
                            break
            i += 1
        return annos

    def _duration_filtering(self, rec_pred):
        '''
        Set duration filtering for seizure prediction
        '''
        # convert threshold from seconds to number of windows
        threshold = self.dur_thresh/self.fs
        starts, lengths, values = self._rle(rec_pred)
        for i in range(len(starts)):
            if values[i] == 1:
                if lengths[i] < threshold:
                    end = starts[i] + lengths[i]
                    rec_pred[starts[i]:end] = 0
        return rec_pred.astype(int)

