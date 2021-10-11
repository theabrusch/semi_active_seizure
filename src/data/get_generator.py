from src.data import datagenerator, train_val_split
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_generator(generator_kwargs):
    if generator_kwargs['gen_type'] == 'DataGenerator':
        train, val = train_val_split.train_val_split(**generator_kwargs)
        print('Initialising training dataset.')
        train_dataset = datagenerator.DataGenerator(**generator_kwargs, 
                                                    subjects_to_use = train)
        print('Initialising validation dataset.')
        val_dataset = datagenerator.DataGenerator(**generator_kwargs, 
                                                  subjects_to_use = val)

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

