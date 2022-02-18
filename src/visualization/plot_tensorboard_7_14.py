import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import os

csv_files = glob.glob('src/visualization/tensorboard_do/*.csv')

cd = os.getcwd()
run_names = []

runs = dict()

for file in csv_files:
    file_name = file.split('/')[-1]
    run_name = file_name.split('-')[-3]
    run = '_'.join(run_name.split('_')[3:])
    subj = run.split('_')[-3]
    exp = '_'.join(run.split('_')[-2:])
    measure = file_name.split('-')[-1].split('.')[0]
    df = pd.read_csv(file)
    if exp in ['stride_005', 'stride_01', 'stride_05', 'stride_1']:
        experiment = 'stride'
    elif exp in ['bckgrate_stride01', 'bckgrate_stride1']:
        experiment = 'balance'
    elif run.split('_')[-1] in ['dropout01', 'dropout02', 'dropout04']:
        experiment = 'dropout'
        subj = run.split('_')[-2]
    if not subj in runs.keys():
        runs[subj] = dict()
    if not experiment in runs[subj].keys():
        runs[subj][experiment] = dict()
    if not measure in runs[subj][experiment].keys():
        runs[subj][experiment][measure] = pd.DataFrame(df['Value'].values[:100], columns = [exp])
    else:
        runs[subj][experiment][measure][exp] = df['Value'].values[:100]

# plot sensitivity stride subject 14
plt.plot(runs['14']['stride']['test_sens'][['stride_005', 'stride_01', 'stride_05', 'stride_1']][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'upper right', fontsize =14)
plt.ylabel('Sensitivity', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.show()

# plot specificity stride subject 14
plt.plot(runs['14']['stride']['test_spec'][['stride_005', 'stride_01', 'stride_05', 'stride_1']][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'upper left', fontsize =14)
plt.ylabel('Specificity', fontsize =14)
plt.xlabel('Epoch', fontsize =14)
plt.show()



# plot specificity stride subject 7
plt.plot(runs['7']['stride']['test_sens'][['stride_005', 'stride_01', 'stride_05', 'stride_1']][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'center left', fontsize =14)
plt.ylabel('Sensitivity', fontsize =14)
plt.xlabel('Epoch',fontsize=14)
plt.show()


# plot specificity stride subject 7
plt.plot(runs['7']['stride']['test_spec'][['stride_005', 'stride_01', 'stride_05', 'stride_1']][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'], loc = 'center left', fontsize=14)
plt.ylabel('Specificity',fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.show()

# add balance
runs['14']['balance']['test_sens']['bal05'] = runs['14']['stride']['test_sens']['stride_05']
runs['14']['balance']['test_sens']['bal1'] = runs['14']['stride']['test_sens']['stride_1']
runs['7']['balance']['test_sens']['bal05'] = runs['7']['stride']['test_sens']['stride_05']
runs['7']['balance']['test_sens']['bal1'] = runs['7']['stride']['test_sens']['stride_1']

runs['14']['balance']['test_spec']['bal05'] = runs['14']['stride']['test_spec']['stride_01']
runs['14']['balance']['test_spec']['bal1'] = runs['14']['stride']['test_spec']['stride_1']
runs['7']['balance']['test_spec']['bal05'] = runs['7']['stride']['test_spec']['stride_01']
runs['7']['balance']['test_spec']['bal1'] = runs['7']['stride']['test_spec']['stride_1']
# add dropout
runs['14']['dropout']['test_sens']['dropout06'] = runs['14']['stride']['test_sens']['stride_01']
runs['7']['dropout']['test_sens']['dropout06'] = runs['7']['stride']['test_sens']['stride_01']
runs['14']['dropout']['test_spec']['dropout06'] = runs['14']['stride']['test_spec']['stride_01']
runs['7']['dropout']['test_spec']['dropout06'] = runs['7']['stride']['test_spec']['stride_01']

# plot balance subject 14
plt.plot(runs['14']['balance']['test_sens'][['bckgrate_stride01','bckgrate_stride1', 'bal05', 'bal1']][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.1s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.1s, balanced', 'Stride 1s, balanced'], 
           loc = 'upper right', fontsize =14)
plt.ylabel('Sensitivity', fontsize =14)
plt.xlabel('Epoch', fontsize =14)
plt.show()

# plot balance stride subject 14
plt.plot(runs['14']['balance']['test_spec'][['bckgrate_stride01','bckgrate_stride1', 'bal05', 'bal1']][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.1s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.1s, balanced', 'Stride 1s, balanced'], 
           loc = 'lower left', fontsize=14)
plt.ylabel('Specificity',fontsize=14)
plt.xlabel('Epoch',fontsize=14)
plt.show()


# plot balance subject 7
plt.plot(runs['7']['balance']['test_sens'][['bckgrate_stride01','bckgrate_stride1', 'bal05', 'bal1']][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.1s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.1s, balanced', 'Stride 1s, balanced'], 
           loc = 'lower left',fontsize=14)
plt.ylabel('Sensitivity',fontsize=14)
plt.xlabel('Epoch',fontsize=14)
plt.show()

# plot balance stride subject 7
plt.plot(runs['7']['balance']['test_spec'][['bckgrate_stride01','bckgrate_stride1', 'bal05', 'bal1']][0:100])
plt.ylim([0,1])
plt.legend(['Stride 0.1s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.1s, balanced', 'Stride 1s, balanced'], 
           loc = 'lower left', fontsize=14)
plt.ylabel('Specificity',fontsize=14)
plt.xlabel('Epoch',fontsize=14)
plt.show()

# plot dropout subject 14
plt.plot(runs['14']['dropout']['test_sens'][['14_dropout01', '14_dropout02', '14_dropout04', 'dropout06']][0:100])
plt.ylim([0,1])
plt.legend(['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'], 
           loc = 'upper left', fontsize=14)
plt.ylabel('Sensitivity',fontsize=14)
plt.xlabel('Epoch',fontsize=14)
plt.show()

plt.plot(runs['14']['dropout']['test_spec'][['14_dropout01', '14_dropout02', '14_dropout04', 'dropout06']][0:100])
plt.ylim([0,1])
plt.legend(['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'], 
           loc = 'upper left', fontsize=14)
plt.ylabel('Specificity',fontsize=14)
plt.xlabel('Epoch',fontsize=14)
plt.show()


# plot dropout subject 7
plt.plot(runs['7']['dropout']['test_sens'][['7_dropout01', '7_dropout02', '7_dropout04', 'dropout06']][0:100])
plt.ylim([0,1])
plt.legend(['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'], 
           loc = 'lower right', fontsize=14)
plt.ylabel('Sensitivity',fontsize=14)
plt.xlabel('Epoch',fontsize=14)
plt.show()

plt.plot(runs['7']['dropout']['test_spec'][['7_dropout01', '7_dropout02', '7_dropout04', 'dropout06']][0:100])
plt.ylim([0,1])
plt.legend(['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'], 
           loc = 'lower right',fontsize=14)
plt.ylabel('Specificity',fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.show()