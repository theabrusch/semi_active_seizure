import mne
import re
seizures = open('seizures.txt', 'r')
lines = seizures.readlines()
files = dict()

result = re.match('\A[SZ]', 'ABSZ')

for line in lines:
    if 'SZ' in line.strip():
        seiz = line.strip()
        files[seiz] = dict()
        files[seiz]['file'] = []
        files[seiz]['start'] = []
        files[seiz]['end'] = []
    else:
        parts = line.strip().split(', ')
        file_name = parts[0].split('/')[-1].split('.')[0]
        edf_path = 'data/dataplot/' + file_name + '.edf'
        files[seiz]['file'].append(edf_path)
        files[seiz]['start'].append(float(parts[1]))
        files[seiz]['end'].append(float(parts[2]))

mne_file = mne.io.read_raw_edf(files['FNSZ']['file'][0])

dur = files['FNSZ']['end'][0] - files['FNSZ']['start'][0]
mne.viz.plot_raw(mne_file, duration = dur - 60, start = files['FNSZ']['start'][0]-10,
                 n_channels = 10, clipping=None)