from dataapi import data_collection as dc
import argparse
import yaml
from prettytable import PrettyTable
from src.data import get_generator
from src.models import get_model, get_optim, get_loss, train_model
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

print('done loading packages')
def params_to_tb(writer, args):
    t = PrettyTable(['Argument', 'Value'])
    param_dict = vars(args)
    for key, val in param_dict.items():
        t.add_row([key, val])
    writer.add_text("args", t.get_html_string(), global_step=0)

def main(args):
    writer = SummaryWriter('../runs/'+ args.model_type + str(datetime.now()))
    params_to_tb(writer, args)
    with open('configuration.yml', 'r') as file:
        config = yaml.safe_load(file)

    time_start = datetime.now()

    # get datasets and dataloaders
    datagen = config['data_gen']
    datagen['hdf5_path'] = args.file_path
    datagen['window_length'] = args.window_length
    datagen['bckg_stride'] = args.bckg_stride
    datagen['seiz_stride'] = args.seiz_stride
    datagen['bckg_rate'] = args.bckg_rate
    datagen['anno_based_seg'] = args.anno_based_seg
    datagen['prefetch_data_dir'] = args.prefetch_data_dir
    datagen['prefetch_data_from_seg'] = args.prefetch_data_from_seg

    gen_args = config['generator_kwargs']
    gen_args['num_workers'] = args.num_workers

    train_dataset, val_dataset = get_generator.get_dataset(datagen)
    train_dataloader, val_dataloader = get_generator.get_generator(train_dataset,
                                                                val_dataset,
                                                                gen_args)
    print('Data loader initialization took', datetime.now()-time_start, '.')

    # load model
    model_config = config['model_kwargs']
    model_config['model'] = args.model_type
    model_config['dropoutprob'] = args.dropoutprob
    model_config['padding'] = args.padding
    model_config['input_shape'] = train_dataset._get_X_shape()
    model = get_model.get_model(model_config)

    # train model
    optim_config = config['fit']['optimizer']
    optim_config['lr'] = args.lr
    optimizer, scheduler = get_optim.get_optim(model.parameters(), optim_config)

    fit_config = config['fit']
    fit_config['weight'] = train_dataset.bckg_rate
    loss_fn = get_loss.get_loss(**fit_config)

    model_train = train_model.model_train(model = model, 
                                        optimizer = optimizer, 
                                        loss_fn = loss_fn, 
                                        writer = writer,
                                        scheduler = scheduler)
    time = datetime.now()
    train_loss, val_loss = model_train.train(train_dataloader,
                                            val_dataloader,
                                            args.epochs)
    print('Training model for', args.epochs, 'epochs took', datetime.now()-time, '.')
    print('Total time', datetime.now()-time_start, '.')
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # datagen
    parser.add_argument('--file_path', type = str)
    parser.add_argument('--window_length', type=float, default = 2)
    parser.add_argument('--num_workers', type=int, default = 0)
    parser.add_argument('--bckg_stride', type=eval, default=None)
    parser.add_argument('--seiz_stride', type=eval, default=None)
    parser.add_argument('--bckg_rate', type=eval, default=None) # None or value
    parser.add_argument('--anno_based_seg', type=bool, default=False)
    parser.add_argument('--prefetch_data_dir', type=bool, default=False)
    parser.add_argument('--prefetch_data_from_seg', type=bool, default=False)

    # model
    parser.add_argument('--model_type', type=str, default='BaselineCNN')
    parser.add_argument('--dropoutprob', type=float, default=0.2)
    parser.add_argument('--padding', type=bool, default=False)       

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)

    args = parser.parse_args()
    main(args)
