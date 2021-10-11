from dataapi import data_collection as dc
import numpy as np
import yaml
from src.data import get_generator
from src.models import get_model

with open('configuration.yml', 'r') as file:
    config = yaml.safe_load(file)

train_dataset, val_dataset = get_generator.get_dataset(config['data_gen'])
train_dataloader, val_dataloader = get_generator.get_generator(train_dataset,
                                                               val_dataset,
                                                               config['generator_kwargs'])
model_config = config['model_kwargs']
model_config['input_shape'] = train_dataset._get_X_shape()

model = get_model.get_model(model_config)