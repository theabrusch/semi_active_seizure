from main import main
import argparse
parser = argparse.ArgumentParser()
# job name
parser.add_argument('--job_name', type = str, default='nojobname')
parser.add_argument('--run_folder', type = str, default='notspec')
# datagen
parser.add_argument('--seed', type = int, default = 20)
parser.add_argument('--seiz_classes', nargs = '+', default=['seiz','fnsz', 'gnsz', 'cpsz', 'spsz', 'tcsz', 'seiz', 'absz', 'tnsz', 'mysz'])
parser.add_argument('--eval_seiz_classes', nargs = '+', default=None)
parser.add_argument('--file_path', type = str)
parser.add_argument('--window_length', type=float, default = 2)
parser.add_argument('--bckg_stride', type=eval, default=None)
parser.add_argument('--seiz_stride', type=eval, default=None)
parser.add_argument('--bckg_rate_val', type=eval, default=None) # None or value
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
parser.add_argument('--cnn_dropoutprob', type=float, default=0.4)
parser.add_argument('--lstm_units', type=eval, default=140)
parser.add_argument('--dense_units', type=eval, default=70)
parser.add_argument('--padding', type=eval, default=False)       

# Training parameters
parser.add_argument('--optimizer', type = str, default = 'RMSprop')
parser.add_argument('--scheduler', type = str, default = None)
parser.add_argument('--milestones', type = eval, default = [50, 130, 150])
parser.add_argument('--use_weighted_loss', type=eval, default=True)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type = float, default=1e-3)  

args = parser.parse_args(['--file_path','/Volumes/GoogleDrive/Mit drev/Matematisk modellering/Speciale/semi_active_seizure/data/hdf5/temple_seiz_small_1.hdf5', 
                          '--model_type', 'BaselineCNNV2','--test_subj', '[0]' ])

                          #'--model_path', '/Users/theabrusch/Desktop/Speciale_data/final_model.pt'])
                          #'--test_subj', '/train/00012742', '/train/00006546', '/train/00003760', '/test/00005625', '/train/00007095', '/train/00002445', '/train/00005371', '/train/00000529', '/train/00004569', '/train/00008345', '/train/00006811', '/train/00005044', '/train/00005347', '/test/00008544', '/train/00010861', '/train/00009994', '/train/00008204', '/train/00008480', '/train/00001891', '/test/00006546', '/train/00000630', '/train/00007296', '/train/00005411', '/train/00012695', '/train/00004657' ,'/train/00011684', '/train/00004842' ,'/train/00004126', '/train/00009245', '/train/00011981', '/train/00000216', '/train/00010678', '/train/00002348', '/train/00006774', '/train/00007313', '/train/00012786', '/train/00010061', '/train/00000107', '/train/00002365', '/train/00011596', '/train/00007795', '/train/00010480', '/train/00010412', '/train/00008188', '/train/00008487', '/train/00005526', '/train/00009881', '/train/00003061','/train/00002044', '/train/00007279', '/train/00010639', '/train/00002235', '/train/00009297', '/train/00003036' ,'/train/00004030' ,'/train/00004045','/train/00004047', '/train/00008835', '/train/00003282', '/train/00010209', '/train/00009880', '/test/00003281',
                          #'--val_subj',  '/train/00001052', '/train/00011927', '/train/00009104', '/train/00008552', '/train/00009866', '/train/00011562', '/train/00007446', '/test/00005213', '/train/00012966', '/train/00008615', '/train/00013407', '/test/00005479', '/train/00001981', '/train/00011999', '/train/00012707', '/test/00006059', '/train/00007936', '/train/00008527', '/train/00006520', '/train/00007929', '/train/00003849', '/train/00008164', '/train/00000244', '/train/00011455', '/train/00000604', '/train/00004799', '/test/00000795', '/train/00008440', '/train/00009158', '/train/00012700', '/train/00007431', '/train/00001605', '/train/00003623', '/test/00000629', '/train/00007264', '/train/00004526', '/train/00000032', '/train/00008823', '/train/00011902', '/train/00009902', '/train/00011454', '/train/00006134', '/train/00000077', '/train/00005765', '/train/00004746', '/train/00010035', '/train/00000924', '/train/00001349', '/train/00009762', '/train/00008971', '/train/00005660', '/train/00000598', '/train/00005169', '/train/00004774', '/train/00007555', '/train/00002521', '/train/00003437', '/train/00011618', '/train/00001878', '/train/00009694'])
main(args)