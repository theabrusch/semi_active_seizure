import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np

csv_files = glob.glob('src/visualization/tensorboard/*.csv')

subj_14_files = []
subj_14_names = []
subj_7_files = []
subj_7_names = []

for file in csv_files:
    file_name = file.split('/')[-1]
    run = '_'.join(file_name.split('_')[4:])
    run_name = run.split('-')[0]
    metric = run.split('-')[-1].split('.')[0]
    if run_name == 'test_valset':
        subj_14_files.append(file)
        subj_14_names.append(metric)
    elif run_name == 'test_subj_7_valset_2':
        subj_7_files.append(file)
        subj_7_names.append(metric)

subj_7 = dict()
for name, file in zip(subj_7_names, subj_7_files):
    df = pd.read_csv(file)
    if name.split('_')[0] == 'val':
            i = 0 
    elif name.split('_')[0] == 'test':
        i = 1
    if name.split('_')[-1] in subj_7.keys():
        subj_7[name.split('_')[-1]][:,i] = df['Value'].values
    else:
        subj_7[name.split('_')[-1]] = np.zeros((150,2))
        subj_7[name.split('_')[-1]][:,i] = df['Value'].values

step_f1 = np.argmax(subj_7['f1'][:,0])
new_metric = 2*subj_7['sens']*subj_7['spec']/(subj_7['sens'] + subj_7['spec'])
step = np.argmax(new_metric[:,0])

plt.plot(subj_7['sens'])
plt.ylim([0,1])
plt.xlim([-2, 150])
plt.vlines(step, 0, 1, colors = 'black', zorder = 3)
plt.ylabel('Sensitivity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Validation score', 'Test score', 'Best val. F1 score'], loc = 'lower left', fontsize = 10)
plt.show()

plt.plot(subj_7['spec'])
plt.ylim([0,1])
plt.xlim([-2, 150])
plt.vlines(step, 0, 1, colors = 'black', zorder = 3)
plt.ylabel('Specificity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Validation score', 'Test score', 'Best val. F1 score'], loc = 'lower left', fontsize = 10)
plt.show()

plt.plot(new_metric)
plt.ylim([0,1])
plt.xlim([-2, 150])
plt.vlines(step, 0, 1, colors = 'black', zorder = 3)
plt.ylabel('F1 score', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Validation score', 'Test score', 'Best val. F1 score'], loc = 'lower left', fontsize = 10)
plt.show()

subj_14 = dict()
for name, file in zip(subj_14_names, subj_14_files):
    df = pd.read_csv(file)
    if name.split('_')[0] == 'val':
            i = 0 
    elif name.split('_')[0] == 'test':
        i = 1
    if name.split('_')[-1] in subj_14.keys():
        subj_14[name.split('_')[-1]][:,i] = df['Value'].values
    else:
        subj_14[name.split('_')[-1]] = np.zeros((150,2))
        subj_14[name.split('_')[-1]][:,i] = df['Value'].values

step_14 = np.argmax(subj_14['f1'][:,1])
new_met = 2*subj_14['sens']*subj_14['spec']/(subj_14['sens']+subj_14['spec'])
step14_newmet = np.argmax(new_met[:,1])

plt.plot(subj_14['sens'])
plt.ylim([0,1])
plt.xlim([-2, 150])
plt.vlines(step14_newmet, 0, 1, colors = 'black', zorder = 3)
plt.ylabel('Sensitivity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Validation score', 'Test score', 'Best val. F1 score'], loc = 'upper right', fontsize = 10)
plt.show()

plt.plot(subj_14['spec'])
plt.ylim([0,1])
plt.xlim([-2, 150])
plt.vlines(step_14, 0, 1, colors = 'black', zorder = 3)
plt.vlines(step14_newmet, 0, 1, colors = 'blue', zorder = 4)
plt.ylabel('Specificity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Validation score', 'Test score', 'Best val. F1 score'], loc = 'lower right', fontsize = 10)
plt.show()

plt.plot(new_met)
plt.ylim([0,1])
plt.xlim([-2, 150])
plt.vlines(step14_newmet, 0, 1, colors = 'black', zorder = 3)
plt.ylabel('F1 score', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Validation score', 'Test score', 'Best val. F1 score'], loc = 'upper right', fontsize = 10)
plt.show()


plt.plot(subj_14['f1'])
plt.plot(new_met)
plt.ylim([0,1])
plt.xlim([-2, 150])
plt.vlines(step_14, 0, 1, colors = 'black', zorder = 5)
#plt.vlines(step14_newmet, 0, 1, colors = 'blue', zorder = 6)
plt.ylabel('F1 score', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Test score', 'Best val. F1 score'], loc = 'upper right', fontsize = 10)
plt.show()