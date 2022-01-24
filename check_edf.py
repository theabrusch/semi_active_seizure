from src.data import train_val_split

hdf5_path = '/Users/theabrusch/Desktop/Speciale_data/hdf5/temple_seiz_full.hdf5'
split = 2
seiz_classes = ['mysz', 'absz', 'spsz', 'tnsz', 'tcsz', 'cpsz', 'gnsz', 'fnsz']
train, val, test = train_val_split.get_kfold(hdf5_path = hdf5_path, 
                                            split = split,
                                            seiz_classes = seiz_classes,
                                            val_split = 0,
                                            n_splits = 5,
                                            n_val_splits = 5)

transfer_subjects, transfer_records, test_records = train_val_split.get_transfer_subjects(hdf5_path, test, seiz_classes, seed = 20)