from dataapi import data_collection as dc

f = dc.File('/Users/theabrusch/Desktop/Speciale_data/boston_scalp_new.hdf5', 'r')

seizures = dict()
for subj in f['train'].keys():
    subject = f['train'][subj]
    seizures[subj] = []
    for rec in subject:
        record = subject[rec]
        if 'Annotations' in record.keys():
            for anno in record['Annotations']:
                if anno['Name'] == 'Seiz':
                    seizures[subj].append(anno)
        else:
            print('No annotations for subject', subj, 'record', rec)


f.close()

print('succesfully loaded file')
