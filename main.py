from dataapi import data_collection as dc
import argparse
import yaml
from prettytable import PrettyTable
from src.data import get_generator
from src.models import get_model, get_optim, get_loss, train_model, metrics
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
    datagen['bckg_rate_val'] = args.bckg_rate_val
    datagen['bckg_rate_train'] = args.bckg_rate_train
    datagen['anno_based_seg'] = args.anno_based_seg
    datagen['prefetch_data_from_seg'] = args.prefetch_data_from_seg
    datagen['train_val_test'] = args.train_val_test
    datagen['val_subj'] = args.val_subj
    datagen['sens'] = args.sens
    datagen['standardise'] = args.standardise

    gen_args = config['generator_kwargs']
    gen_args['batch_size'] = args.batch_size
    gen_args['use_train_seed'] = args.use_train_seed

    if args.train_val_test:
        train_dataset, val_dataset, test = get_generator.get_dataset(datagen)
    else:
        train_dataset, val_dataset = get_generator.get_dataset(datagen)

    train_dataloader, val_dataloader = get_generator.get_generator(train_dataset,
                                                                    val_dataset,
                                                                    gen_args)
    print('Data loader initialization took', datetime.now()-time_start, '.')

    temp = next(iter(train_dataloader))
    # load model
    model_config = config['model_kwargs']
    model_config['model'] = args.model_type
    model_config['lstm_units'] = args.lstm_units
    model_config['dense_units'] = args.dense_units
    model_config['dropoutprob'] = args.dropoutprob
    model_config['padding'] = args.padding
    model_config['input_shape'] = train_dataset._get_X_shape()
    model = get_model.get_model(model_config)

    # train model
    optim_config = config['fit']['optimizer']
    optim_config['optimizer'] = args.optimizer
    optim_config['model'] = args.model_type
    optim_config['lr'] = args.lr
    optim_config['weight_decay'] = args.weight_decay
    optimizer, scheduler = get_optim.get_optim(model, optim_config)

    fit_config = config['fit']

    if args.use_weighted_loss:
        fit_config['weight'] = train_dataset.bckg_rate
    else:
        fit_config['weight'] = None

    loss_fn = get_loss.get_loss(**fit_config)

    if not datagen['train_val_test']:
        # if validation set is the same as test set, then use the final model
        choose_best = False
    else:
        choose_best = True

    model_train = train_model.model_train(model = model, 
                                        optimizer = optimizer, 
                                        loss_fn = loss_fn, 
                                        writer = writer,
                                        scheduler = scheduler,
                                        choose_best = choose_best)
    metric_names = ['sensitivity', 'specificity', 'accuracy']
    metrics_compute = metrics.get_metrics(metric_names)

    time = datetime.now()
    train_loss, val_loss = model_train.train(train_dataloader,
                                            val_dataloader,
                                            args.epochs)
                                            
    print('Training model for', args.epochs, 'epochs took', datetime.now()-time, '.')
    print('Total time', datetime.now()-time_start, '.')

    if config['general']['run_test']:
        datagen['bckg_stride'] = None
        datagen['seiz_stride'] = None
        datagen['bckg_rate'] = None
        datagen['anno_based_seg'] = False
        datagen['prefetch_data_dir'] = True
        datagen['prefetch_data_from_seg'] = False
        
        if args.train_val_test:
            test_loader = get_generator.get_test_generator(datagen, gen_args, test)
        else:
            test_loader = get_generator.get_test_generator(datagen, gen_args, val_dataset.subjects_to_use)

        y_pred = model_train.eval(test_loader)
        y_true = test_loader.dataset.labels_collect

        for i in range(len(metric_names)):
            met = metrics_compute[i](y_true, y_pred)
            writer.add_scalar('test/' + metric_names[i], met)
            print(metric_names[i], met)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # datagen
    parser.add_argument('--file_path', type = str)
    parser.add_argument('--window_length', type=float, default = 2)
    parser.add_argument('--bckg_stride', type=eval, default=None)
    parser.add_argument('--seiz_stride', type=eval, default=None)
    parser.add_argument('--bckg_rate_val', type=eval, default=20) # None or value
    parser.add_argument('--bckg_rate_train', type=eval, default=1)
    parser.add_argument('--use_train_seed', type=eval, default=True)
    parser.add_argument('--anno_based_seg', type=eval, default=False)
    parser.add_argument('--prefetch_data_from_seg', type=eval, default=False)
    parser.add_argument('--train_val_test', type=eval, default=False)
    parser.add_argument('--val_subj', type = eval, default=None)
    parser.add_argument('--standardise', type = eval, default=False)
    parser.add_argument('--sens', type = eval, default=0)
    parser.add_argument('--batch_size', type=eval, default=512)

    # model
    parser.add_argument('--model_type', type=str, default='BaselineCNN')
    parser.add_argument('--dropoutprob', type=float, default=0.4)
    parser.add_argument('--lstm_units', type=eval, default=140)
    parser.add_argument('--dense_units', type=eval, default=70)
    parser.add_argument('--padding', type=eval, default=False)       

    # Training parameters
    parser.add_argument('--optimizer', type = eval, default = 'RMSprop')
    parser.add_argument('--use_weighted_loss', type=eval, default=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type = float, default=1e-3)

    args = parser.parse_args()
    main(args)
