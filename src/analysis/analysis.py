import numpy as np
import pandas as pd
from collections import namedtuple
from dataapi import data_collection as dc

class Postprocessing():
    def __init__(self, 
                segments, 
                label_fs, 
                orig_fs,
                prob_thresh = 0.75, 
                dur_thresh = 1,
                p_up = 0.1,
                statemachine = False,
                post_proces = ['duration_filt', 'target_agg']):
                
        self.segments = segments
        self.fs = label_fs
        self.orig_fs = orig_fs
        self.prob_thresh = prob_thresh
        self.p_up = p_up,
        self.dur_thresh = dur_thresh
        self.post_proces = post_proces
        self.statemachine = statemachine

    def postproces(self):
        proba = self.segments['seiz prob'].values
        recs = self.segments['rec'].values
        uni_recs = np.unique(recs)
        annos_collect = dict()
        total_rec_mod = pd.DataFrame()
        for rec in uni_recs:
            rec_idx = np.array(recs) == rec
            rec_mod = self.check_for_missing_val(rec_idx)
            total_rec_mod = total_rec_mod.append(rec_mod, ignore_index=True)
            rec_pred = rec_mod['seiz prob']
            if self.statemachine:
                y_pred = self.statemachinepostprocess(rec_pred)
            else:
                y_pred = self._probabilistic_filtering(rec_pred)
            if 'duration_filt' in self.post_proces:
                y_pred = self._duration_filtering(y_pred)
            annos = self._to_events(y_pred)
            if 'target_agg' in self.post_proces:
                annos = self._target_aggregation(annos)
            annos_collect[rec] = annos
        return annos_collect, total_rec_mod
    
    def check_for_missing_val(self, rec_idx):
        rec = self.segments[rec_idx]
        diff_samples =  int(1/self.fs*self.orig_fs)
        startseg_diff = (rec['startseg'].diff() >diff_samples).sum()
        if not startseg_diff > 0:
            return rec
        else:
            startseg = 0
            for i, seg in rec.iterrows():
                if seg['startseg'] - startseg > diff_samples:
                    n_missing = int((seg['startseg'] - startseg)/diff_samples-1)
                    miss_startseg = np.array([startseg+(n+1)*diff_samples for n in range(n_missing)])
                    miss_endseg = miss_startseg+diff_samples
                    temp = pd.DataFrame({'startseg': miss_startseg, 'endseg': miss_endseg})
                    temp['label'] = np.nan
                    temp['seiz_types'] = np.nan
                    temp['path'] = seg['path']
                    temp['subj'] = seg['subj']
                    temp['rec'] = seg['rec']
                    temp['y pred'] = np.nan
                    temp['seiz prob'] = np.nan
                    rec_mod = rec.append(temp, ignore_index = True).reset_index()
                startseg = seg['startseg']
            rec_mod = rec_mod.sort_values(by='startseg').reset_index()
            return rec_mod



    
    def _probabilistic_filtering(self, probability):
        '''
        Set threshold for seizure prediction
        '''
        y_pred = (probability > self.prob_thresh).astype(int)
        y_pred[np.isnan(probability)] = np.nan
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
        threshold = self.dur_thresh*self.fs
        starts, lengths, values = self._rle(rec_pred)
        for i in range(len(starts)):
            if values[i] == 1:
                if lengths[i] < threshold:
                    end = starts[i] + lengths[i]
                    rec_pred[starts[i]:end] = 0
        return rec_pred

    def statemachinepostprocess(self, rec_pred):
        state = 'neg'
        new_pred = np.zeros(len(rec_pred))
        for i in range(len(rec_pred)):
            if state == 'neg':
                if rec_pred[i] < self.prob_thresh:
                    new_pred[i] = 0
                else:
                    state = 'pos'
                    new_pred[i] = 1
            elif state == 'pos':
                if rec_pred[i] + self.p_up < self.prob_thresh:
                    state = 'smooth'
                    new_pred[i] = 0
                else:
                    new_pred[i] = 1
            elif state == 'smooth':
                if rec_pred[i] + self.p_up < self.prob_thresh:
                    state = 'neg'
                    new_pred[i] = 0
                    new_pred[i-1] = 0
                else:
                    state = 'pos'
                    new_pred[i] = 1
                    new_pred[i-1] = 1
        
        return new_pred



class AnyOverlap():
    def __init__(self, pred_annos, segments, hdf5_path, seiz_eval, margin = 1) -> None:
        self.pred_annos = pred_annos
        #self.segments = segments
        self.hdf5_path = hdf5_path
        self.margin = margin
        self.seiz_eval = seiz_eval
    
    def compute_performance(self):
        recs = self.pred_annos.keys()
        file = dc.File(self.hdf5_path, 'r')
        seiz_classes = self.seiz_eval
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        total_recdur = 0
        anno_stats = dict()
        recstats_seiz = pd.DataFrame()
        recstats_collect = pd.DataFrame()
        for rec in recs:
            record = file[rec]
            rec_true_annos = record['Annotations']
            rec_true_annos = self._convert_true_anno(rec_true_annos, record.start_time, seiz_classes)
            rec_pred_annos = self.pred_annos[rec]
            FN_annos, TP_annos, FP_annos, TN_annos, recstat = self._ovlp(rec_true_annos, rec_pred_annos)
            recstat['rec'] = rec
            recstats_seiz = recstats_seiz.append(recstat, ignore_index=True)

            anno_stats[rec] = dict()
            anno_stats[rec]['TP annos'] = TP_annos
            anno_stats[rec]['FP annos'] = FP_annos
            anno_stats[rec]['FN annos'] = FN_annos

            temp = pd.DataFrame({'rec': rec, 'TP': len(TP_annos), 'FP': len(FP_annos),
                                 'FN': len(FN_annos), 'TN': len(TN_annos), 'dur': record.duration,
                                 'seizures':len(TP_annos)+ len(FN_annos)}, index = [0])

            recstats_collect = recstats_collect.append(temp, ignore_index=True)
            TP += len(TP_annos)
            FN += len(FN_annos)
            FP += len(FP_annos)
            TN += len(TN_annos)
            total_recdur += record.duration

            
        
        return TP, FN, FP, TN, total_recdur, anno_stats, recstats_collect, recstats_seiz


    def _convert_true_anno(self, annos, rec_start, seiz_classes):
        conv_annos = []
        for anno in annos:
            start = anno['Start']-rec_start
            if anno['Name'] in seiz_classes:
                lab = 1
            elif anno['Name'] == 'bckg':
                lab = 0
            else:
                lab = np.nan
            an = {'Name': lab, 'Start': start, 'Duration': anno['Duration'], 'class': anno['Name']}
            conv_annos.append(an)
        return conv_annos

    def _ovlp(self, rec_true_annos, rec_pred_annos):
        FN_annos = []
        TP_annos = []
        FP_annos = []
        TN_annos = []
        rec_stat = pd.DataFrame()

        for anno in rec_true_annos:
            true_stop = anno['Start'] + anno['Duration']
            hit = False
            if not np.isnan(anno['Name']):
                for pred_anno in rec_pred_annos:
                    if pred_anno['Name'] == anno['Name']:
                        pred_stop = pred_anno['Start'] + pred_anno['Duration']
                        if anno['Start'] < pred_stop + self.margin and  true_stop > pred_anno['Start']-self.margin:
                            hit = True
                            break
                if hit:
                    if anno['Name'] ==1:
                        TP_annos.append(anno)
                        temp = pd.DataFrame({'seiz_type': anno['class'], 'hit': 1, 'dur' : anno['Duration']}, index =[0])
                        rec_stat = rec_stat.append(temp, ignore_index=True)
                    else:
                        TN_annos.append(anno)
                else:
                    if anno['Name'] == 1:
                        FN_annos.append(anno)
                        temp = pd.DataFrame({'seiz_type': anno['class'], 'hit': 0, 'dur' : anno['Duration']}, index =[0])
                        rec_stat = rec_stat.append(temp, ignore_index=True)

        for anno in rec_pred_annos:
            true_stop = anno['Start'] + anno['Duration']
            hit = False
            if anno['Name'] == 1:
                for pred_anno in rec_true_annos:
                    if pred_anno['Name'] == 1:
                        pred_stop = pred_anno['Start'] + pred_anno['Duration']
                        if anno['Start'] < pred_stop + self.margin and  true_stop > pred_anno['Start']-self.margin:
                            hit = True
                            break
                if not hit:
                    FP_annos.append(anno)

        return FN_annos, TP_annos, FP_annos, TN_annos, rec_stat
                    
