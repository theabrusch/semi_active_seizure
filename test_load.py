from dataapi import data_collection as dc

f = dc.File('/work3/theb/boston_scalp_new.hdf5', 'r')
f.close()

print('succesfully loaded file')
