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
        self.bckg_samples = list(range(self.dataset.seiz_samples, self.dataset.seiz_samples+self.dataset.bckg_samples))
        self.bckg_rate = int(self.dataset.bckg_rate*self.dataset.seiz_samples)
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        # sample all seizure samples and a fixed number of background samples
        np.random.seed(self.seed)
        samples = np.append(self.seiz_samples, np.random.choice(self.bckg_samples, self.bckg_rate, replace = False))
        np.random.seed(None)
        return iter(shuffle(samples, random_state = self.seed))

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
        train_dataset = datagenerator.DataGenerator(**data_gen, bckg_rate=data_gen['bckg_rate_train'],
                                                    subjects_to_use = train)
        print('Number of seizure segments in training set:', train_dataset.seiz_samples)
        print('Initialising validation dataset.')
        val_dataset = datagenerator.DataGenerator(**data_gen, bckg_rate=data_gen['bckg_rate_val'],
                                                  subjects_to_use = val)
        print('Number of seizure segments in validation set', val_dataset.seiz_samples)
    if data_gen['train_val_test']:
        return train_dataset, val_dataset, test
    else:
        return train_dataset, val_dataset

def get_generator(train_dataset, val_dataset, generator_kwargs):
    if generator_kwargs['use_train_seed']:
        seed = int(np.random.uniform(0, 2**32))
    else:
        seed = None
    train_sampler = SeizSampler(train_dataset, seed = seed)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size = generator_kwargs['batch_size'], 
                                  sampler = train_sampler,
                                  num_workers = generator_kwargs['num_workers'],
                                  pin_memory = True)
    seed = int(np.random.uniform(0, 2**32))
    val_sampler = SeizSampler(val_dataset, seed = seed)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size = generator_kwargs['val_batch_size'], 
                                sampler = val_sampler,
                                num_workers = generator_kwargs['num_workers'],
                                pin_memory = True)

    return train_dataloader, val_dataloader

def get_test_generator(data_gen, generator_kwargs, val_subj):
    if data_gen['gen_type'] == 'DataGenerator':
        print('Initialising validation dataset.')
        val_dataset = datagenerator.TestGenerator(**data_gen, 
                                                  subjects_to_use = val_subj)
        print('Number of seizure segments in test set:', val_dataset.seiz_samples)
        sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size = generator_kwargs['val_batch_size'],
                                    sampler = sampler)
    
    return val_dataloader

