from dataapi import data_collection as dc
import numpy as np
import pickle
import yaml
from src.data import get_generator
from src.models import get_model, get_optim, get_loss, train_model

with open('configuration.yml', 'r') as file:
    config = yaml.safe_load(file)

# get datasets and dataloaders
train_dataset, val_dataset = get_generator.get_dataset(config['data_gen'])
train_dataloader, val_dataloader = get_generator.get_generator(train_dataset,
                                                               val_dataset,
                                                               config['generator_kwargs'])
# load model
model_config = config['model_kwargs']
model_config['input_shape'] = train_dataset._get_X_shape()
model = get_model.get_model(model_config)

# train model
optim_config = config['fit']['optimizer']
optimizer, scheduler = get_optim.get_optim(model.parameters(), optim_config)

config['fit']['weight'] = train_dataset.bckg_rate
loss_fn = get_loss.get_loss(**config['fit'])

train_model = train_model.model_train(model, 
                                      optimizer, 
                                      loss_fn, 
                                      scheduler)

train_loss, val_loss = train_model.train(train_dataloader,
                                         val_dataloader,
                                         config['fit']['n_epochs'])

with open('train_loss.pickle', 'wb') as f:
    pickle.dump(train_loss, f)

with open('val_loss.pickle', 'w') as f:
    pickle.dump(val_loss, f)
