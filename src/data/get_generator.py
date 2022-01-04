from src.data import datagenerator, train_val_split
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from torch import Generator
import numpy as np
from sklearn.utils import shuffle
from typing import Iterator

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


def get_dataset(data_gen):
    if data_gen['gen_type'] == 'DataGenerator':
        if data_gen['train_val_test']:
            train, val, test = train_val_split.train_val_test_split(**data_gen)
            print('Test subject(s)', test)
        else:
            train, val = train_val_split.train_val_split(**data_gen)
        print('Training subjects', train)
        print('Val subjects', val)
        print('Initialising training dataset.')
        datasegment = datagenerator.SegmentData(**data_gen,
                                                bckg_rate=data_gen['bckg_rate_train'],
                                                subjects_to_use = train)
        segment, norm_coef = datasegment.segment_data()
        train_dataset = datagenerator.DataGenerator(**data_gen, subjects_to_use=train,
                                                    bckg_rate=data_gen['bckg_rate_train'],
                                                    segments = segment, norm_coef = norm_coef)
        print('Number of seizure segments in training set:', train_dataset.seiz_samples)
        
        print('Initialising validation dataset.')
        data_gen['use_train_seed'] = True
        data_gen['bckg_rate'] = data_gen['bckg_rate_val']
        datasegment = datagenerator.SegmentData(**data_gen,
                                                subjects_to_use = val)
        segment, norm_coef = datasegment.segment_data()
        val_dataset = datagenerator.DataGenerator(**data_gen, subjects_to_use=val,
                                                  #bckg_rate=data_gen['bckg_rate_val'],
                                                  segments = segment, norm_coef=norm_coef)
        print('Number of seizure segments in validation set', val_dataset.seiz_samples)
    if data_gen['train_val_test']:
        return train_dataset, val_dataset, test
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

def get_test_generator(data_gen, generator_kwargs, test_subj):
    if data_gen['gen_type'] == 'DataGenerator':
        print('Initialising test dataset.')
        dset = data_gen['hdf5_path'].split('/')[-1].split('.')[0]
        datasegment = datagenerator.SegmentData(**data_gen,
                                                subjects_to_use = test_subj)
        segment, norm_coef = datasegment.segment_data()
        val_dataset = datagenerator.DataGenerator(**data_gen, subjects_to_use=test_subj,
                                                  #bckg_rate=data_gen['bckg_rate_val'],
                                                  segments = segment, norm_coef=norm_coef)
        print('Number of seizure segments in test set:', (val_dataset.seiz_samples))
        sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size = generator_kwargs['val_batch_size'],
                                    sampler = sampler)
    
    return val_dataloader

