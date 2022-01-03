import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import os

csv_files = glob.glob('src/visualization/tensorboard/*.csv')

cd = os.getcwd()

run_names = []
subj4_6 = []

for file in csv_files:
    file_name = file.split('/')[-1]
    run_name = file_name.split('-')[-3]
    run = '_'.join(run_name.split('_')[3:])
    if not run in run_names:
        run_names.append(run)
        if len(run.split('_'))>2:
            subj = run.split('_')[2]
            if subj == '4' or subj == '6':
                subj4_6.append(run)

run_description = ['Stride 1s', 'Stride 0.05s', 'Stride 0.5s',
                   'Stride 0.1s', 'Stride 0.5s']

test_sens = np.zeros((60,12))
test_sens_names = []
test_spec_subjs = []
test_spec = np.zeros((60,12))
test_spec_names = []
test_sens_subjs = []
i=0
j=0
for file in csv_files:
    file_name = file.split('/')[-1]
    run_name = file_name.split('-')[-3]
    measure = file_name.split('-')[-1].split('.')[0]
    run = '_'.join(run_name.split('_')[3:])
    if run in subj4_6:
        df = pd.read_csv(file)
        subj = run.split('_')[2]
        try:
            if measure == 'test_sens':
                test_sens[:,i]=df['Value'].values[0:60]
                i+=1
                test_sens_names.append(file_name)
                test_sens_subjs.append(subj)
            elif measure == 'test_spec':
                test_spec[:,j]=df['Value'].values[0:60]
                j+=1
                test_spec_names.append(file_name)
                test_spec_subjs.append(subj)
        except:
            print('hej')

subj6_sens_idx = [i for i in range(len(test_sens_subjs)) if test_sens_subjs[i] == '6']
subj6_spec_idx = [ i for i in range(len(test_sens_subjs)) if test_spec_subjs[i] == '6']

sens_6 = test_sens[:,subj6_sens_idx[:-2]]
spec_6 = test_spec[:,subj6_spec_idx[:-2]]
plt.plot(sens_6)
plt.ylim([0,1])
plt.xlim([-2, 60])
plt.ylabel('Test sensitivity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(run_description[:-1], loc = 'upper left', fontsize = 10)
plt.show()

plt.plot(spec_6)
plt.ylim([0,1])
plt.xlim([-2, 60])
plt.ylabel('Test specificity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(run_description[:-1], loc = 'lower left', fontsize = 10)
plt.show()

# unbalanced 
sens_6_unbal = test_sens[:,[subj6_sens_idx[0], subj6_sens_idx[2], subj6_sens_idx[4], subj6_sens_idx[5]]]
spec_6_unbal = test_spec[:,[subj6_sens_idx[0], subj6_sens_idx[2], subj6_sens_idx[4], subj6_sens_idx[5]]]
# stride 1
plt.plot(sens_6_unbal[:,[0, 3, 1, 2]])
plt.ylim([0,1])
plt.ylabel('Test sensitivity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Stride 1s, balanced', 'Stride 1s, unbalanced',
            'Stride 0.5s, balanced', 'Stride 0.5s, unbalanced'], loc = 'upper left', fontsize = 10)
plt.show()

plt.plot(spec_6_unbal[:,[0, 3, 1, 2]])
plt.ylim([0,1])
plt.ylabel('Test specificity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Stride 1s, balanced', 'Stride 1s, unbalanced',
            'Stride 0.5s, balanced', 'Stride 0.5s, unbalanced'], loc = 'lower left', fontsize = 10)
plt.show()

subj4_sens_idx = [i for i in range(len(test_sens_subjs)) if test_sens_subjs[i] == '4']
subj4_spec_idx = [ i for i in range(len(test_sens_subjs)) if test_spec_subjs[i] == '4']

sens_4 = test_sens[:,subj4_sens_idx[:4]]
spec_4 = test_spec[:,subj4_spec_idx[:4]]
plt.plot(sens_4)
plt.ylim([0,1])
plt.xlim([-2, 60])
plt.ylabel('Test sensitivity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(run_description[:-1], loc = 'upper left', fontsize = 10)
plt.show()

plt.plot(spec_4)
plt.ylim([0,1])
plt.xlim([-2, 60])
plt.ylabel('Test specificity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(run_description[:-1], loc = 'lower left', fontsize = 10)
plt.show()

sens_4_unbal = test_sens[:,[subj4_sens_idx[0], subj4_sens_idx[-1], subj4_sens_idx[2], subj4_sens_idx[4]]]
spec_4_unbal = test_spec[:,[subj4_sens_idx[0], subj4_sens_idx[-1], subj4_sens_idx[2], subj4_sens_idx[4]]]

# stride 1
plt.plot(sens_4_unbal)
plt.ylim([0,1])
plt.xlim([-2, 60])
plt.ylabel('Test sensitivity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Stride 1s, balanced', 'Stride 1s, unbalanced',
            'Stride 0.5s, balanced', 'Stride 0.5s unbalanced'], loc = 'upper left', fontsize = 10)
plt.show()

plt.plot(spec_4_unbal)
plt.ylim([0,1])
plt.ylabel('Test specificity', fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.legend(['Stride 1s, balanced', 'Stride 1s, unbalanced',
            'Stride 0.5s, balanced', 'Stride 0.5s unbalanced'], loc = 'lower left', fontsize = 10)
plt.show()
