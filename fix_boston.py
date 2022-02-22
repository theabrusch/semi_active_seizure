from dataapi import data_collection as dc

F = dc.File('/work3/theb/boston_scalp_subj24.hdf5', 'r+')
records = F.get_children(dc.Record, get_obj=False)
i=1
for rec in records:
    print(i, 'of', len(records))
    i+=1
    record = F[rec]
    annotations = record['Annotations']
    if len(record['Annotations']) > 0:
        print('hej')
    bckg = annotations._get_unlabeled_data(annotations[()], record.start_time, 
                                            record.start_time + record.duration,
                                            unlabled_name="bckg")
    annotations.append(bckg)
    annotations.remove_dublicates()

    F.flush()
F.close()

F = dc.File('/work3/theb/boston_scalp_18ch.hdf5', 'r+')
f_subj24 = dc.File('/work3/theb/boston_scalp_subj24.hdf5', 'r+')

for subj in f_subj24['train'].keys():
    subject = f_subj24['train'][subj]
    if subj not in F.keys():
        train_subj = F.create_subject(subj)
    else: 
        train_subj = F[subj]

    for att in subject.attrs.keys():
        train_subj.attrs[att] = subject.attrs[att]
    
    for rec in subject.keys():
        record = subject[rec]
        if rec not in train_subj.keys():
            train_record = train_subj.create_record(rec)
        else:
            train_record = train_subj[rec]
        for att in record.attrs.keys():
            train_record.attrs[att]= record.attrs[att]

        if 'EEG' in train_record.keys():
            newsig = train_record['EEG']
        else:
            newsig = train_record.create_signal('EEG', data = record['EEG'][()])

        for att in record['EEG'].attrs.keys():
            newsig.attrs[att] = record['EEG'].attrs[att]
        if 'Annotations' in train_record.keys():
            annotations = train_record['Annotations']
        else:
            annotations = train_record.create_annotations()
        if len(record['Annotations']) > 0:
            annotations.append(record['Annotations'][()])
            annotations.remove_dublicates()

        F.flush()
F.close()

