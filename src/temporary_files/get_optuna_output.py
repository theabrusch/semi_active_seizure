import pandas as pd

df_1 = pd.read_csv('/Volumes/GoogleDrive/Mit drev/Matematisk modellering/Speciale/semi_active_seizure/data/optuna_final_valsplit_12022-02-05 04:20:52.397869.csv')
df_1 = df_1.drop(columns=['Unnamed: 0', 'number', 'datetime_start', 'datetime_complete', 'duration'])
df_1 = df_1.sort_values(by = 'value', ascending = False)

df_3 = pd.read_csv('/Volumes/GoogleDrive/Mit drev/Matematisk modellering/Speciale/semi_active_seizure/data/optuna_final_valsplit_32022-02-05 03:36:38.848985.csv')
df_3 = df_3.drop(columns=['Unnamed: 0', 'number', 'datetime_start', 'datetime_complete', 'duration'])
df_3 = df_3.sort_values(by='value', ascending = False)
