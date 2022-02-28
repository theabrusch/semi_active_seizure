import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import os
from matplotlib.gridspec import SubplotSpec

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



def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontsize = 16)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

# plot sensitivity stride subject 14
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (9,7))

ax[0,0].plot(runs['14']['stride']['test_sens'][['stride_005', 'stride_01', 'stride_05', 'stride_1']][0:100], label = ['Stride 0.05s', 'Stride 0.1s', 'Stride 0.5s', 'Stride 1s'])
ax[0,0].set_ylim([0,1])
ax[0,0].set_ylabel('Sensitivity', fontsize = 14)
ax[0,0].set_xlabel('Epoch', fontsize = 14)
ax[0,0].tick_params('x', labelsize=12)
ax[0,0].tick_params('y', labelsize=12)

# plot specificity stride subject 14
ax[0,1].plot(runs['14']['stride']['test_spec'][['stride_005', 'stride_01', 'stride_05', 'stride_1']][0:100])
ax[0,1].set_ylim([0,1])
ax[0,1].set_ylabel('Specificity', fontsize =14)
ax[0,1].set_xlabel('Epoch', fontsize =14)
ax[0,1].tick_params('x', labelsize=12)
ax[0,1].tick_params('y', labelsize=12)

# plot specificity stride subject 7
ax[1,0].plot(runs['7']['stride']['test_sens'][['stride_005', 'stride_01', 'stride_05', 'stride_1']][0:100])
ax[1,0].set_ylim([0,1])
ax[1,0].set_ylabel('Sensitivity', fontsize =14)
ax[1,0].set_xlabel('Epoch',fontsize=14)
ax[1,0].tick_params('x', labelsize=12)
ax[1,0].tick_params('y', labelsize=12)


# plot specificity stride subject 7
ax[1,1].plot(runs['7']['stride']['test_spec'][['stride_005', 'stride_01', 'stride_05', 'stride_1']][0:100])
ax[1,1].set_ylim([0,1])
ax[1,1].set_ylabel('Specificity',fontsize=14)
ax[1,1].set_xlabel('Epoch', fontsize=14)
ax[1,1].tick_params('x', labelsize=12)
ax[1,1].tick_params('y', labelsize=12)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.subplots_adjust(bottom = 0.15, hspace = 0.4, wspace = 0.3)
fig.legend(lines, labels, loc = 'lower center', bbox_to_anchor=(0.5, 0), borderaxespad=0.5,
            bbox_transform = plt.gcf().transFigure, ncol = 4, fontsize = 14)
grid = plt.GridSpec(2, 2)
create_subtitle(fig, grid[0, ::], 'Subject 14')
create_subtitle(fig, grid[1, ::], 'Subject 7')
#fig.tight_layout()
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

fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (9,7))

ax[0,0].plot(runs['14']['balance']['test_sens'][['bckgrate_stride01','bckgrate_stride1', 'bal05', 'bal1']][0:100], label = ['Stride 0.1s, unbalanced', 'Stride 1s, unbalanced', 'Stride 0.1s, balanced', 'Stride 1s, balanced'])
ax[0,0].set_ylim([0,1])
ax[0,0].set_ylabel('Sensitivity', fontsize = 14)
ax[0,0].set_xlabel('Epoch', fontsize = 14)
ax[0,0].tick_params('x', labelsize=12)
ax[0,0].tick_params('y', labelsize=12)

# plot specificity stride subject 14
ax[0,1].plot(runs['14']['balance']['test_spec'][['bckgrate_stride01','bckgrate_stride1', 'bal05', 'bal1']][0:100])
ax[0,1].set_ylim([0,1])
ax[0,1].set_ylabel('Specificity', fontsize =14)
ax[0,1].set_xlabel('Epoch', fontsize =14)
ax[0,1].tick_params('x', labelsize=12)
ax[0,1].tick_params('y', labelsize=12)

# plot specificity stride subject 7
ax[1,0].plot(runs['7']['balance']['test_sens'][['bckgrate_stride01','bckgrate_stride1', 'bal05', 'bal1']][0:100])
ax[1,0].set_ylim([0,1])
ax[1,0].set_ylabel('Sensitivity', fontsize =14)
ax[1,0].set_xlabel('Epoch',fontsize=14)
ax[1,0].tick_params('x', labelsize=12)
ax[1,0].tick_params('y', labelsize=12)


# plot specificity stride subject 7
ax[1,1].plot(runs['7']['balance']['test_spec'][['bckgrate_stride01','bckgrate_stride1', 'bal05', 'bal1']][0:100])
ax[1,1].set_ylim([0,1])
ax[1,1].set_ylabel('Specificity',fontsize=14)
ax[1,1].set_xlabel('Epoch', fontsize=14)
ax[1,1].tick_params('x', labelsize=12)
ax[1,1].tick_params('y', labelsize=12)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.subplots_adjust(bottom = 0.2, hspace = 0.4, wspace = 0.3)
fig.legend(lines, labels, loc = 'lower center', bbox_to_anchor=(0.5, 0), borderaxespad=0.5,
            bbox_transform = plt.gcf().transFigure, ncol = 2, fontsize = 14)
grid = plt.GridSpec(2, 2)
create_subtitle(fig, grid[0, ::], 'Subject 14')
create_subtitle(fig, grid[1, ::], 'Subject 7')
#fig.tight_layout()
plt.show()

# plot dropout
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (9,7))

ax[0,0].plot(runs['14']['dropout']['test_sens'][['14_dropout01', '14_dropout02', '14_dropout04', 'dropout06']][0:100], label = ['Dropout 0.1', 'Dropout 0.2', 'Dropout 0.4', 'Dropout 0.6'])
ax[0,0].set_ylim([0,1])
ax[0,0].set_ylabel('Sensitivity', fontsize = 14)
ax[0,0].set_xlabel('Epoch', fontsize = 14)
ax[0,0].tick_params('x', labelsize=12)
ax[0,0].tick_params('y', labelsize=12)

# plot specificity stride subject 14
ax[0,1].plot(runs['14']['dropout']['test_spec'][['14_dropout01', '14_dropout02', '14_dropout04', 'dropout06']][0:100])
ax[0,1].set_ylim([0,1])
ax[0,1].set_ylabel('Specificity', fontsize =14)
ax[0,1].set_xlabel('Epoch', fontsize =14)
ax[0,1].tick_params('x', labelsize=12)
ax[0,1].tick_params('y', labelsize=12)

# plot specificity stride subject 7
ax[1,0].plot(runs['7']['dropout']['test_sens'][['7_dropout01', '7_dropout02', '7_dropout04', 'dropout06']][0:100])
ax[1,0].set_ylim([0,1])
ax[1,0].set_ylabel('Sensitivity', fontsize =14)
ax[1,0].set_xlabel('Epoch',fontsize=14)
ax[1,0].tick_params('x', labelsize=12)
ax[1,0].tick_params('y', labelsize=12)


# plot specificity stride subject 7
ax[1,1].plot(runs['7']['dropout']['test_spec'][['7_dropout01', '7_dropout02', '7_dropout04', 'dropout06']][0:100])
ax[1,1].set_ylim([0,1])
ax[1,1].set_ylabel('Specificity',fontsize=14)
ax[1,1].set_xlabel('Epoch', fontsize=14)
ax[1,1].tick_params('x', labelsize=12)
ax[1,1].tick_params('y', labelsize=12)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.subplots_adjust(bottom = 0.2, hspace = 0.4, wspace = 0.3)
fig.legend(lines, labels, loc = 'lower center', bbox_to_anchor=(0.5, 0), borderaxespad=0.5,
            bbox_transform = plt.gcf().transFigure, ncol = 4, fontsize = 14)
grid = plt.GridSpec(2, 2)
create_subtitle(fig, grid[0, ::], 'Subject 14')
create_subtitle(fig, grid[1, ::], 'Subject 7')
#fig.tight_layout()
plt.show()


