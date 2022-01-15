from dataapi import data_collection as dc
import argparse
import yaml
from prettytable import PrettyTable
from src.data import get_generator
from src.models import get_model, get_optim, get_loss, train_model, metrics
from datetime import datetime
from src.models.metrics import sensitivity, specificity, accuracy
from sklearn.metrics import f1_score, precision_score
from torch.utils.tensorboard import SummaryWriter

print('done loading packages')
def params_to_tb(writer, args):
    t = PrettyTable(['Argument', 'Value'])
    param_dict = vars(args)
    for key, val in param_dict.items():
        t.add_row([key, val])
    writer.add_text("args", t.get_html_string(), global_step=0)

def main(args):
    writer = SummaryWriter('../runs/' + args.run_folder + '/' + args.model_type +\
                           '_'+ str(datetime.now()) + '_' + \
                            args.job_name)
    params_to_tb(writer, args)
    with open('configuration.yml', 'r') as file:
        config = yaml.safe_load(file)

    time_start = datetime.now()

    # get datasets and dataloaders
    datagen = config['data_gen']
    datagen['seed'] = args.seed
    datagen['seiz_classes'] = args.seiz_classes
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
    datagen['test_subj'] = args.test_subj
    datagen['sens'] = args.sens
    datagen['standardise'] = args.standardise
    datagen['use_train_seed'] = args.use_train_seed
    datagen['subj_strat'] = args.subj_strat
    datagen['protocol'] = args.protocol
    datagen['seiz_strat'] = args.seizure_strat

    gen_args = config['generator_kwargs']
    gen_args['batch_size'] = args.batch_size
    gen_args['use_train_seed'] = args.use_train_seed

    if args.train_val_test:
        train_dataset, val_dataset, test = get_generator.get_dataset(datagen, summarywriter = writer)
        split_text = 'Train subjects: ' + str(train_dataset.subjects_to_use) + \
                 '. Validation subjects: ' + str(val_dataset.subjects_to_use) + \
                  '. Test subjects: ' + str(test) + '.'
    else:
        train_dataset, val_dataset = get_generator.get_dataset(datagen, summarywriter = writer)
        split_text = 'Train subjects: ' + str(train_dataset.subjects_to_use) + \
                 '. Validation subjects: ' + str(val_dataset.subjects_to_use)+ '.'

    train_dataloader, val_dataloader = get_generator.get_generator(train_dataset,
                                                                    val_dataset,
                                                                    gen_args)
    print('Data loader initialization took', datetime.now()-time_start, '.')
    writer.add_text('Split', split_text, global_step = 0)

    # Get test loader
    test_datagen = datagen.copy()
    test_datagen['bckg_stride'] = None
    test_datagen['seiz_stride'] = None
    test_datagen['bckg_rate'] = None
    test_datagen['anno_based_seg'] = False
    test_datagen['prefetch_data_from_seg'] = True
    test_datagen['use_train_seed'] = False

    if not args.train_val_test:
        test_loader = get_generator.get_test_generator(test_datagen,
                                                       gen_args,
                                                       val_dataset.subjects_to_use, 
                                                       summarywriter=writer)
    else:
        test_loader = get_generator.get_test_generator(test_datagen, 
                                                       gen_args, 
                                                       test, 
                                                       writer)

    # load model
    model_config = config['model_kwargs']
    model_config['model'] = args.model_type
    model_config['lstm_units'] = args.lstm_units
    model_config['dense_units'] = args.dense_units
    model_config['dropoutprob'] = args.dropoutprob
    model_config['glob_avg_pool'] = args.glob_avg_pool
    model_config['padding'] = args.padding
    model_config['input_shape'] = train_dataset._get_X_shape()
    model = get_model.get_model(model_config)

    # train model
    optim_config = config['fit']['optimizer']
    optim_config['optimizer'] = args.optimizer
    optim_config['scheduler'] = args.scheduler
    optim_config['milestones'] = args.milestones
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
        track_test = True
    else:
        choose_best = False
        track_test = True

    model_train = train_model.model_train(model = model, 
                                            optimizer = optimizer, 
                                            loss_fn = loss_fn, 
                                            writer = writer,
                                            scheduler = scheduler,
                                            choose_best = choose_best)

    time = datetime.now()
    train_loss, val_loss = model_train.train(train_loader = train_dataloader,
                                            val_loader = val_dataloader,
                                            test_loader = test_loader,
                                            epochs = args.epochs)
                                            
    print('Training model for', args.epochs, 'epochs took', datetime.now()-time, '.')
    print('Total time', datetime.now()-time_start, '.')

    if config['general']['run_test']:
        y_pred, y_true, seiz_types = model_train.eval(test_loader, return_seiz_type = True)

        # calculate metrics
        sens = sensitivity(y_true, y_pred)
        spec = specificity(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        acc = accuracy(y_true, y_pred)

        writer.add_scalar('test_final/sensitivity', sens)
        writer.add_scalar('test_final/specificity', spec)
        writer.add_scalar('test_final/f1', f1)
        writer.add_scalar('test_final/precision', prec)
        writer.add_scalar('test_final/accuracy', acc)

        # calculate metrics for different seizure types
        import numpy as np
        uni_seiz_types = np.unique(seiz_types)
        t = PrettyTable(['Seizure type', 'Sensitivity', 'Number of segments'])        
        for seiz in uni_seiz_types:
            if seiz != 'bckg':
                idx = seiz_types == seiz
                y_true_temp = y_true[idx]
                y_pred_temp = y_pred[idx]
                sens_temp = sensitivity(y_true_temp, y_pred_temp)
                t.add_row([seiz, sens_temp, len(y_true_temp)])

        writer.add_text("Seizure specific performance", t.get_html_string(), global_step=0)

        # get seizure specific performance on validation set
        val_dataloader.dataset.return_seiz_type = True
        y_pred, y_true, seiz_types = model_train.eval(val_dataloader, return_seiz_type = True)
        uni_seiz_types = np.unique(seiz_types)
        t = PrettyTable(['Seizure type', 'Sensitivity', 'Number of segments'])        
        for seiz in uni_seiz_types:
            if seiz != 'bckg':
                idx = seiz_types == seiz
                y_true_temp = y_true[idx]
                y_pred_temp = y_pred[idx]
                sens_temp = sensitivity(y_true_temp, y_pred_temp)
                t.add_row([seiz, sens_temp, len(y_true_temp)])

        writer.add_text("Seizure specific performance, validation", t.get_html_string(), global_step=0)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # job name
    parser.add_argument('--job_name', type = str, default='nojobname')
    parser.add_argument('--run_folder', type = str, default='notspec')
    # datagen
    parser.add_argument('--seed', type = int, default = 20)
    parser.add_argument('--seiz_classes', type = eval, default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
    parser.add_argument('--excl_seiz_classes', type = eval, default = [])
    parser.add_argument('--file_path', type = str)
    parser.add_argument('--window_length', type=float, default = 2)
    parser.add_argument('--bckg_stride', type=eval, default=None)
    parser.add_argument('--seiz_stride', type=eval, default=None)
    parser.add_argument('--bckg_rate_val', type=eval, default=20) # None or value
    parser.add_argument('--bckg_rate_train', type=eval, default=1)
    parser.add_argument('--use_train_seed', type=eval, default=True)
    parser.add_argument('--subj_strat', type=eval, default=False)
    parser.add_argument('--anno_based_seg', type=eval, default=False)
    parser.add_argument('--prefetch_data_from_seg', type=eval, default=False)
    parser.add_argument('--train_val_test', type=eval, default=False)
    parser.add_argument('--val_subj', type = eval, default=None)
    parser.add_argument('--test_subj', type = eval, default=None)
    parser.add_argument('--standardise', type = eval, default=False)
    parser.add_argument('--sens', type = eval, default=0)
    parser.add_argument('--seizure_strat', type = eval, default = False)
    parser.add_argument('--batch_size', type=eval, default=512)
    # protocol(s) to use for training
    parser.add_argument('--protocol', type=str, default= 'all')

    # model
    parser.add_argument('--model_type', type=str, default='BaselineCNN')
    parser.add_argument('--glob_avg_pool', type=eval, default=False)
    parser.add_argument('--dropoutprob', type=float, default=0.4)
    parser.add_argument('--lstm_units', type=eval, default=140)
    parser.add_argument('--dense_units', type=eval, default=70)
    parser.add_argument('--padding', type=eval, default=False)       

    # Training parameters
    parser.add_argument('--optimizer', type = str, default = 'RMSprop')
    parser.add_argument('--scheduler', type = str, default = None)
    parser.add_argument('--milestones', type = eval, default = [50, 130, 150])
    parser.add_argument('--use_weighted_loss', type=eval, default=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type = float, default=1e-3)

    args = parser.parse_args()
    main(args)
