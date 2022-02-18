from dataapi import data_collection as dc

F = dc.File('/work3/theb/boston_scalp_newnew.hdf5', 'r+')
records = F.get_children(dc.Record, get_obj=False)
for rec in records:
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