import pandas as pd

df_1 = pd.read_csv('/Users/theabrusch/Desktop/Speciale_data/optuna_fina/optuna_final_valsplit_f1val_0_newdo2022-02-19 16:55:41.270421.csv')
df_1 = df_1.drop(columns=['Unnamed: 0', 'number', 'datetime_start', 'datetime_complete', 'duration'])
df_1 = df_1.sort_values(by = 'value', ascending = False)

df_3 = pd.read_csv('/Users/theabrusch/Desktop/Speciale_data/optuna_fina/optuna_final_valsplit_f1val_2_newdo2022-02-20 15:39:43.071941.csv')
df_3 = df_3.drop(columns=['Unnamed: 0', 'number', 'datetime_start', 'datetime_complete', 'duration'])
df_3 = df_3.sort_values(by='value', ascending = False)
