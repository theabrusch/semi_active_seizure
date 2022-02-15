import pandas as pd

df_1 = pd.read_csv('/Users/theabrusch/Desktop/Speciale_data/optuna_final_valsplit_f1val_0_newdo2022-02-15 16:42:37.049930.csv')
df_1 = df_1.drop(columns=['Unnamed: 0', 'number', 'datetime_start', 'datetime_complete', 'duration'])
df_1 = df_1.sort_values(by = 'value', ascending = False)

df_3 = pd.read_csv('/Users/theabrusch/Desktop/Speciale_data/optuna_final_valsplit_f1val_2_newdo2022-02-15 13:45:26.952591.csv')
df_3 = df_3.drop(columns=['Unnamed: 0', 'number', 'datetime_start', 'datetime_complete', 'duration'])
df_3 = df_3.sort_values(by='value', ascending = False)
