import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import os

csv_files = glob.glob('src/visualization/tensorboard_new/*.csv')

cd = os.getcwd()
run_names = []

runs = dict()

for file in csv_files:
    file_name = file.split('/')[-1]
    run_name = file_name.split('-')[-3]
    run = '_'.join(run_name.split('_')[3:])
    subj = run.split('_')[-2]
    exp = run.split('_')[-1]
    measure = file_name.split('-')[-1].split('.')[0]
    if not subj in runs.keys():
        runs[subj] = dict()
    df = pd.read_csv(file)
    if exp in ['lowstride', 'stride01', 'stride05', 'stride1']:
        experiment = 'stride'
    elif exp in ['unbalst05', 'unbalst1']:
        experiment = 'balance'
    elif exp in ['dropout01', 'dropout02', 'dropout04']:
        experiment = 'dropout'
    if not experiment in runs[subj].keys():
        runs[subj][experiment] = dict()
    if not measure in runs[subj][experiment].keys():
        runs[subj][experiment][measure] = pd.DataFrame(df['Value'].values[:149], columns = [exp])
    else:
        runs[subj][experiment][measure][exp] = df['Value'].values[:149]

# plot sensitivity stride subject 14
plt.plot(runs['14']['stride']['test_sens'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'center left')
plt.ylabel('Sensitivity')
plt.xlabel('Epoch')
plt.show()

# plot specificity stride subject 14
plt.plot(runs['14']['stride']['test_spec'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'center left')
plt.ylabel('Specificity')
plt.xlabel('Epoch')
plt.show()


# plot sensitivity stride subject 14
plt.plot(runs['14']['stride']['test_sens'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'center left')
plt.ylabel('Sensitivity')
plt.xlabel('Epoch')
plt.show()

# plot specificity stride subject 7
plt.plot(runs['7']['stride']['test_sens'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'center left')
plt.ylabel('Sensitivity')
plt.xlabel('Epoch')
plt.show()


# plot specificity stride subject 7
plt.plot(runs['7']['stride']['test_spec'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'center left')
plt.ylabel('Specificity')
plt.xlabel('Epoch')
plt.show()

# add balance
runs['14']['balance']['test_sens']['bal05'] = runs['14']['stride']['test_sens']['stride05']
runs['14']['balance']['test_sens']['bal1'] = runs['14']['stride']['test_sens']['stride1']
runs['7']['balance']['test_sens']['bal05'] = runs['7']['stride']['test_sens']['stride05']
runs['7']['balance']['test_sens']['bal1'] = runs['7']['stride']['test_sens']['stride1']

runs['14']['balance']['test_spec']['bal05'] = runs['14']['stride']['test_spec']['stride05']
runs['14']['balance']['test_spec']['bal1'] = runs['14']['stride']['test_spec']['stride1']
runs['7']['balance']['test_spec']['bal05'] = runs['7']['stride']['test_spec']['stride05']
runs['7']['balance']['test_spec']['bal1'] = runs['7']['stride']['test_spec']['stride1']
# add dropout
runs['14']['dropout']['test_sens']['dropout06'] = runs['14']['stride']['test_sens']['stride01']
runs['7']['dropout']['test_sens']['dropout06'] = runs['7']['stride']['test_sens']['stride01']
runs['14']['dropout']['test_spec']['dropout06'] = runs['14']['stride']['test_spec']['stride01']
runs['7']['dropout']['test_spec']['dropout06'] = runs['7']['stride']['test_spec']['stride01']

# plot balance subject 14
plt.plot(runs['14']['balance']['test_sens'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.5s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.5s, balanced', 'Stride 1s, balanced'], 
           loc = 'upper left')
plt.ylabel('Sensitivity')
plt.xlabel('Epoch')
plt.show()

# plot balance stride subject 14
plt.plot(runs['14']['balance']['test_spec'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.5s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.5s, balanced', 'Stride 1s, balanced'], 
           loc = 'upper right')
plt.ylabel('Specificity')
plt.xlabel('Epoch')
plt.show()


# plot balance subject 7
plt.plot(runs['7']['balance']['test_sens'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.5s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.5s, balanced', 'Stride 1s, balanced'], 
           loc = 'upper left')
plt.ylabel('Sensitivity')
plt.xlabel('Epoch')
plt.show()

# plot balance stride subject 7
plt.plot(runs['7']['balance']['test_spec'][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.5s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.5s, balanced', 'Stride 1s, balanced'], 
           loc = 'lower right')
plt.ylabel('Specificity')
plt.xlabel('Epoch')
plt.show()

# plot dropout subject 14
plt.plot(runs['14']['dropout']['test_sens'][['dropout01', 'dropout02', 'dropout04', 'dropout06']][0:100])
plt.ylim([0,1])
plt.legend(['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'], 
           loc = 'upper left')
plt.ylabel('Sensitivity')
plt.xlabel('Epoch')
plt.show()

plt.plot(runs['14']['dropout']['test_spec'][['dropout01', 'dropout02', 'dropout04', 'dropout06']][0:100])
plt.ylim([0,1])
plt.legend(['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'], 
           loc = 'upper right')
plt.ylabel('Specificity')
plt.xlabel('Epoch')
plt.show()


# plot dropout subject 7
plt.plot(runs['7']['dropout']['test_sens'][['dropout01', 'dropout02', 'dropout04', 'dropout06']][0:100])
plt.ylim([0,1])
plt.legend(['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'], 
           loc = 'lower right')
plt.ylabel('Sensitivity')
plt.xlabel('Epoch')
plt.show()

plt.plot(runs['7']['dropout']['test_spec'][['dropout01', 'dropout02', 'dropout04', 'dropout06']][0:100])
plt.ylim([0,1])
plt.legend(['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'], 
           loc = 'lower right')
plt.ylabel('Specificity')
plt.xlabel('Epoch')
plt.show()