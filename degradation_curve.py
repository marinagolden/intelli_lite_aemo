__author__ = "Mostafa Naemi"
__copyright__ =  "Cornwall Insight Australia"
#%%
import pandas as pd
import numpy as np

file_dir = "C:\\Users\\mostafa.naemi\\OneDrive - Cornwall Insight Ltd\\Documents\\Github\\BESS\\Inputs\\BNRG\\Bairnsdale\\"
file_name = 'degradation_assumption_tesla_2nd.csv'
file_out = 'degradation_Tesla_2nd.csv'
# %%
cyc_per_day = 1
max_cyc = 5000


df_  = pd.read_csv(file_dir+file_name)
df_['Num Cycles'] = (df_.index) * 365 * cyc_per_day 
df_['Num Cycles'] = np.round(df_['Num Cycles'])
df_['Deg per cycle'] = (df_['Capacity'].diff()) / (cyc_per_day * 365)

df = pd.DataFrame(range(max_cyc),columns = ['total cycles todate' ])

df = df.merge(df_ , right_on = 'Num Cycles' , left_on = 'total cycles todate' ,how='left')
df = df.fillna(method = 'bfill')
df = df.fillna(method = 'ffill')
df.loc[0,'Deg per cycle'] = 0

df['degradation_percentage'] = 1 +  df['Deg per cycle'].cumsum()
# %%
sel_col = ['total cycles todate', 'degradation_percentage']
df = df[sel_col]

df.to_csv(file_dir +file_out )

# %%
