from dataapi import data_collection as dc
import numpy as np
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
loss_fn = get_loss.get_loss(**config['fit'])

train_model = train_model.model_train(model, 
                                      optimizer, 
                                      loss_fn, 
                                      scheduler)

train_loss, val_loss = train_model.train(train_dataloader,
                                         val_dataloader,
                                         config['fit']['n_epochs'])

temp = next(iter(train_dataloader))
out_temp = train_model.model(temp[0].float())
out_temp2 = model(temp[0].float())

