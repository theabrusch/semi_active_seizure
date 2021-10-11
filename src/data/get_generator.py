from src.data import datagenerator, train_val_split
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_dataset(data_gen):
    if data_gen['gen_type'] == 'DataGenerator':
        train, val = train_val_split.train_val_split(**data_gen)
        print('Initialising training dataset.')
        train_dataset = datagenerator.DataGenerator(**data_gen, 
                                                    subjects_to_use = train)
        print('Initialising validation dataset.')
        val_dataset = datagenerator.DataGenerator(**data_gen, 
                                                  subjects_to_use = val)
    
    return train_dataset, val_dataset

def get_generator(train_dataset, val_dataset, generator_kwargs):
    train_weights = train_dataset.weights
    train_sampler = WeightedRandomSampler(train_weights, 
                                          num_samples=train_dataset.__len__(), 
                                          replacement = True)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=generator_kwargs['batch_size'], 
                                  sampler=train_sampler)
    val_weights = val_dataset.weights
    val_sampler = WeightedRandomSampler(val_weights, 
                                        num_samples=val_dataset.__len__(), 
                                        replacement = True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=generator_kwargs['val_batch_size'], 
                                sampler=val_sampler)

    return train_dataloader, val_dataloader
