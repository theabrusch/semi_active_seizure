from src.data import datagenerator, train_val_split
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from torch import Generator
import numpy as np
from sklearn.utils import shuffle
from typing import Iterator
from dataapi import data_collection as dc
from prettytable import PrettyTable

class SeizSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset, seed = None) -> None:
        self.dataset = dataset
        self.seiz_samples = list(range(self.dataset.seiz_samples))
        self.bckg_rate = int(self.dataset.bckg_rate*self.dataset.seiz_samples)

        if not seed:
            self.bckg_samples = list(range(self.dataset.seiz_samples, self.dataset.seiz_samples+self.dataset.bckg_samples))
        else:
            self.bckg_samples = list(range(self.dataset.seiz_samples, self.dataset.seiz_samples+self.bckg_rate))
            
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        # sample all seizure samples and a fixed number of background samples
        if self.seed:
            samples = np.append(self.seiz_samples, self.bckg_samples)
        else:
            samples = np.append(self.seiz_samples, np.random.choice(self.bckg_samples, self.bckg_rate, replace = False))
        return iter(shuffle(samples))

    def __len__(self) -> int:
        return self.dataset.__len__()

def add_seiztypes_to_summary(seizure_types, summarywriter, name):
    uni_seiz_types = np.unique(seizure_types)
    n_total = len(seizure_types)
    t = PrettyTable(['Seizure type', 'Number', 'Percent (%)'])
    for seiz in uni_seiz_types:
        n_seiz = (seizure_types == seiz).sum()
        t.add_row([seiz, n_seiz, n_seiz*100/n_total])
    t.add_row(['Total', n_total, 100])
    text_name = name + '_seizure_counts'
    summarywriter.add_text(text_name, t.get_html_string(), global_step=0)

def get_dataset(data_gen, split = None, summarywriter=None):
    if data_gen['gen_type'] == 'DataGenerator':
        if split is None:
            if data_gen['train_val_test']:
                train, val, test = train_val_split.train_val_test_split(**data_gen)
                print('Test subject(s)', test)
            else:
                train, val = train_val_split.train_val_split(**data_gen)
            print('Training subjects', train)
            print('Val subjects', val)
        else:
            train = split['train']
            val = split['val']
        print('Initialising training dataset.')
        datasegment = datagenerator.SegmentData(**data_gen,
                                                bckg_rate=data_gen['bckg_rate_train'],
                                                subjects_to_use = train)
        segment, norm_coef = datasegment.segment_data()
        train_dataset = datagenerator.DataGenerator(**data_gen, subjects_to_use=train,
                                                    bckg_rate=data_gen['bckg_rate_train'],
                                                    segments = segment, norm_coef = norm_coef)
        #add train seizure types to summary
        if not summarywriter is None:
            seizure_types = train_dataset.segments['seiz']['seiz_types']
            add_seiztypes_to_summary(seizure_types, summarywriter, 'train')

        print('Initialising validation dataset.')
        data_gen['use_train_seed'] = True
        data_gen['bckg_rate'] = data_gen['bckg_rate_val']
        data_gen['seiz_classes'] = data_gen['eval_seiz_classes']
        datasegment = datagenerator.SegmentData(**data_gen,
                                                subjects_to_use = val)
        segment, norm_coef = datasegment.segment_data()
        val_dataset = datagenerator.DataGenerator(**data_gen, subjects_to_use=val,
                                                  #bckg_rate=data_gen['bckg_rate_val'],
                                                  segments = segment, norm_coef=norm_coef)
        if not summarywriter is None:
            seizure_types = val_dataset.segments['seiz']['seiz_types']
            add_seiztypes_to_summary(seizure_types, summarywriter, 'validation')

    if split is None:
        if data_gen['train_val_test']:
            return train_dataset, val_dataset, test
        else:
            return train_dataset, val_dataset
    else:
        return train_dataset, val_dataset

def get_generator(train_dataset, val_dataset, generator_kwargs):
    train_sampler = SeizSampler(train_dataset, seed = generator_kwargs['use_train_seed'])
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size = generator_kwargs['batch_size'], 
                                  sampler = train_sampler,
                                  pin_memory = True)
    val_sampler = SeizSampler(val_dataset, seed = True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size = generator_kwargs['val_batch_size'], 
                                sampler = val_sampler,
                                pin_memory = True)

    return train_dataloader, val_dataloader

def get_dataset_cross_val(data_gen, subjs_to_use, writer = None):
    datasegment = datagenerator.SegmentData(**data_gen,
                                                subjects_to_use = subjs_to_use)
    segment, norm_coef = datasegment.segment_data()
    dataset = datagenerator.DataGenerator(**data_gen, subjects_to_use=subjs_to_use,
                                        segments = segment, norm_coef = norm_coef)
    sampler = SeizSampler(dataset, seed = True)
    val_dataloader = DataLoader(dataset, 
                                batch_size = data_gen['batch_size'], 
                                sampler = sampler,
                                pin_memory = True)
    if writer is not None:
        seizure_types = dataset.segments['seiz']['seiz_types']
        add_seiztypes_to_summary(seizure_types, writer, 'val')
    return val_dataloader

def get_test_generator(data_gen, generator_kwargs, test_subj, summarywriter=None):
    if data_gen['gen_type'] == 'DataGenerator':
        print('Initialising test dataset.')
        dset = data_gen['hdf5_path'].split('/')[-1].split('.')[0]
        if  'temple' in dset and data_gen['protocol'] == 'train':
            F = dc.File(data_gen['hdf5_path'], 'r')
            test_subj = F['test'].get_children(object_type = dc.Subject, get_obj = False)
        if data_gen['eval_seiz_classes'] is not None:
            data_gen['seiz_classes'] = data_gen['eval_seiz_classes']
        datasegment = datagenerator.SegmentData(**data_gen,
                                                subjects_to_use = test_subj)
        segment, norm_coef = datasegment.segment_data()
        val_dataset = datagenerator.DataGenerator(**data_gen, subjects_to_use=test_subj,
                                                  return_seiz_type = True,
                                                  segments = segment, norm_coef=norm_coef)
        if not summarywriter is None:
            seizure_types = val_dataset.segments['seiz']['seiz_types']
            add_seiztypes_to_summary(seizure_types, summarywriter, 'test')
        sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size = generator_kwargs['val_batch_size'],
                                    sampler = sampler)
    
    return val_dataloader

