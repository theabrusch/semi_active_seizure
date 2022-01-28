from dataapi import data_collection as dc
import argparse
import yaml
import torch
from prettytable import PrettyTable
from src.data import get_generator, train_val_split
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
        if not key == 'transfer_subjects':
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
    # get split on records for subjects in val set
    splitdict = dict()
    splitdict['hdf5_path'] = args.file_path
    splitdict['split'] = args.split
    splitdict['only_train_seiz'] = False
    splitdict['val_split'] = args.val_split
    splitdict['n_val_splits'] = 5
    splitdict['excl_seiz'] = False
    splitdict['seiz_classes'] = args.seiz_classes
    splitdict['n_splits'] = 5
    splitdict['seed'] = args.seed
    splitdict['min_seiz'] = args.min_seiz
    if not args.bckg_rate_train is None:
        if args.min_ratio < args.bckg_rate_train:
            args.min_ratio = args.bckg_rate_train
    splitdict['min_ratio'] = args.min_ratio

    train, val, test = train_val_split.get_kfold(**splitdict)
    print('Train:', train)
    print('Val:', val)
    print('Test:', test)
    splitdict['subjects'] = val

    transfer_subjects, transfer_records, test_records = train_val_split.get_transfer_subjects(**splitdict)

    # get trainloader
    train_datagen = config['data_gen']
    train_datagen['seed'] = args.seed
    train_datagen['seiz_classes'] = args.seiz_classes
    train_datagen['hdf5_path'] = args.file_path
    train_datagen['window_length'] = args.window_length
    train_datagen['bckg_stride'] = args.bckg_stride
    train_datagen['seiz_stride'] = args.seiz_stride
    train_datagen['bckg_rate'] = args.bckg_rate_train
    train_datagen['anno_based_seg'] = True
    train_datagen['prefetch_data_from_seg'] = True
    train_datagen['protocol'] = 'all'
    train_datagen['batch_size'] = args.batch_size
    train_datagen['use_train_seed'] = False

    # get test loader
    test_datagen = train_datagen.copy()
    test_datagen['bckg_stride'] = None
    test_datagen['seiz_stride'] = None
    test_datagen['bckg_rate'] = None
    test_datagen['anno_based_seg'] = False
    test_datagen['batch_size'] = 512
    
    # initialise tabel for initial and final results
    t_dataset = PrettyTable(['Subject', 'Transfer seiz', 'Transfer bckg', 'Total', 'Ratio'])
    t_res = PrettyTable(['Subject', 'I. sens', 'F. sens', 'I. spec', 'F. spec', 'I. f1', 'F. f1',\
                         'I. sensspec', 'F. sensspec'])
    for subj in transfer_subjects:
        transfer_dataloader = get_generator.get_dataset_transfer(data_gen = train_datagen, 
                                                                subjs_to_use = [subj], 
                                                                records_to_use = transfer_records, 
                                                                split = subj, 
                                                                writer = writer)
        trans_seiz = transfer_dataloader.dataset.seiz_samples
        trans_bckg = transfer_dataloader.dataset.bckg_samples
        print('Trans. Seiz samples', trans_seiz)
        print('Trans. Bckg samples', trans_bckg)
        t_dataset.add_row([subj, trans_seiz, trans_bckg, trans_seiz + trans_bckg, \
                           trans_bckg/trans_seiz])

        test_dataloader = get_generator.get_dataset_transfer(data_gen = test_datagen, 
                                                                subjs_to_use = [subj], 
                                                                records_to_use = test_records, 
                                                                split = subj, 
                                                                writer = writer)
        print('Test seiz samples', test_dataloader.dataset.seiz_samples)
        print('Test bckg samples', test_dataloader.dataset.bckg_samples)
        
        # load model
        model_config = config['model_kwargs']
        model_config['model'] = args.model_type
        model_config['dropoutprob'] = args.dropoutprob
        model_config['glob_avg_pool'] = args.glob_avg_pool
        model_config['padding'] = args.padding
        model_config['input_shape'] = transfer_dataloader.dataset._get_X_shape()
        target_model = get_model.get_model(model_config)
        source_model = get_model.get_model(model_config)

        # load weights of trained model
        checkpoint = torch.load(args.model_path, map_location = 'cpu')
        target_model.load_state_dict(checkpoint['model_state_dict'])
        source_model.load_state_dict(checkpoint['model_state_dict'])

        # train model
        optim_config = config['fit']['optimizer']
        optim_config['optimizer'] = args.optimizer
        optim_config['scheduler'] = args.scheduler
        optim_config['milestones'] = args.milestones
        optim_config['model'] = args.model_type
        optim_config['lr'] = args.lr
        optim_config['weight_decay'] = args.weight_decay
        optimizer, scheduler = get_optim.get_optim(target_model, optim_config)

        fit_config = config['fit']

        if args.use_weighted_loss:
            fit_config['weight'] = transfer_dataloader.dataset.bckg_rate
        else:
            fit_config['weight'] = None

        class_loss = get_loss.get_loss(**fit_config)
        loss_fn = get_loss.TransferLoss(class_loss)
        fit_config['weight'] = test_dataloader.dataset.bckg_rate
        test_loss = get_loss.get_loss(**fit_config)

        model_train = train_model.model_train_ssltf(target_model = target_model, 
                                                    source_model = source_model,
                                                    optimizer = optimizer, 
                                                    loss_fn = loss_fn, 
                                                    val_loss = test_loss,
                                                    writer = writer,
                                                    scheduler = scheduler,
                                                    choose_best = False)
        
        # evaluate test error before transfer
        #y_pred, y_true = model_train.eval(test_dataloader, return_seiz_type = False)
        # calculate metrics
        #sens_init = sensitivity(y_true, y_pred)
        #spec_init = specificity(y_true, y_pred)
        #f1_init = f1_score(y_true, y_pred)
        #prec_init = precision_score(y_true, y_pred)
        #acc_init = accuracy(y_true, y_pred)
        #sensspec_init = 2*sens_init*spec_init/(sens_init+spec_init)

        #writer.add_scalar('test_initial/sensitivity_' + subj, sens_init)
        #writer.add_scalar('test_initial/specificity_' + subj, spec_init)
        #writer.add_scalar('test_initial/f1_' + subj, f1_init)
        #writer.add_scalar('test_initial/precision_' + subj, prec_init)
        #writer.add_scalar('test_initial/accuracy_' + subj, acc_init)


        time = datetime.now()
        train_loss, val_loss = model_train.train_transfer(train_loader = transfer_dataloader,
                                                            val_loader = test_dataloader,
                                                            transfer_subj = subj,
                                                            epochs = args.epochs)
                                                
        print('Training model for', args.epochs, 'epochs took', datetime.now()-time, '.')
        print('Total time', datetime.now()-time_start, '.')

        # test model after transfer
        y_pred, y_true = model_train.eval(test_dataloader, return_seiz_type = False)

        # calculate metrics
        sens_fin = sensitivity(y_true, y_pred)
        spec_fin = specificity(y_true, y_pred)
        f1_fin = f1_score(y_true, y_pred)
        prec_fin = precision_score(y_true, y_pred)
        acc_fin = accuracy(y_true, y_pred)
        sensspec_fin = 2*sens_fin*spec_fin/(sens_fin+spec_fin)


        writer.add_scalar('test_final/sensitivity_' + subj, sens_fin)
        writer.add_scalar('test_final/specificity_' + subj, spec_fin)
        writer.add_scalar('test_final/f1_' + subj, f1_fin)
        writer.add_scalar('test_final/precision_' + subj, prec_fin)
        writer.add_scalar('test_final/accuracy_' + subj, acc_fin)

        t_res.add_row([subj, round(sens_init,3), round(sens_fin,3),\
                       round(spec_init,3), round(spec_fin,3), round(f1_init,3), \
                       round(f1_fin,3), round(sensspec_init,3), round(sensspec_fin, 3)])
    
    writer.add_text("transfer_results", t_res.get_html_string(), global_step=0)
    writer.add_text("transfer_datasets", t_dataset.get_html_string(), global_step=0)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # job name
    parser.add_argument('--job_name', type = str, default='nojobname')
    parser.add_argument('--run_folder', type = str, default='notspec')
    # datagen
    parser.add_argument('--file_path', type = str)
    parser.add_argument('--seed', type = int, default = 20)
    # exclude seizure types to include in training but not evaluation
    parser.add_argument('--onlytrainseiz', default = None)
    parser.add_argument('--val_split', type = int, default = 1)
    parser.add_argument('--split', type = int, default = 2)
    # minimum amount of seizure in transfer dataset
    parser.add_argument('--min_seiz', default = 20)
    # minimum ratio of background in transfer dataset
    parser.add_argument('--min_ratio', type = float, default = 2)
    parser.add_argument('--seiz_classes', nargs = '+', default=['fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
    parser.add_argument('--window_length', type=float, default = 2)
    parser.add_argument('--bckg_stride', type=eval, default=None)
    parser.add_argument('--seiz_stride', type=eval, default=None)
    parser.add_argument('--bckg_rate_val', type=eval, default=20) # None or value
    parser.add_argument('--bckg_rate_train', type=eval, default=1)
    parser.add_argument('--anno_based_seg', type=eval, default=True)
    parser.add_argument('--train_val_test', type=eval, default=False)
    parser.add_argument('--batch_size', type=eval, default=64)
    # protocol(s) to use for training
    parser.add_argument('--protocol', type=str, default= 'all')

    # model
    parser.add_argument('--model_type', type=str, default='BaselineCNN')
    parser.add_argument('--glob_avg_pool', type=eval, default=False)
    parser.add_argument('--dropoutprob', type=float, default=0.4)
    parser.add_argument('--padding', type=eval, default=True)     
    parser.add_argument('--model_path', type=str)    

    # Training parameters
    parser.add_argument('--optimizer', type = str, default = 'RMSprop')
    parser.add_argument('--scheduler', type = str, default = None)
    parser.add_argument('--milestones', type = eval, default = [50, 130, 150])
    parser.add_argument('--use_weighted_loss', type=eval, default=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type = float, default=1e-3)

    args = parser.parse_args()
    main(args)