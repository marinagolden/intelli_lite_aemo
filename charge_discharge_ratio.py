__author__ = "Mostafa Naemi"
__copyright__ =  "Cornwall Insight Australia"
#%%
import pandas as pd
import numpy as np

file_dir = "C:\\Users\mostafa.naemi\\OneDrive - Cornwall Insight Ltd\\Documents\\Github\\BESS\\Inputs\\BNRG\\"
file_name = 'charge_discharge_assumption_0.5C.csv'
file_out = 'BNRG_charge_discharge_0.5C_new.csv'
# %%
cyc_per_day = 2
max_cyc = 20000


df_  = pd.read_csv(file_dir+file_name)
df_['Num Cycles'] = (df_.index) * 365 * cyc_per_day 
df_['Num Cycles'] = np.round(df_['Num Cycles'])

df = pd.DataFrame(range(max_cyc),columns = ['total cycles' ])

df = df.merge(df_ , right_on = 'Num Cycles' , left_on = 'total cycles' ,how='left')
df = df.fillna(method = 'bfill')
df = df.fillna(method = 'ffill')
# %%
sel_col = ['total cycles', 'Discharge']
df = df[sel_col]

df.to_csv(file_dir +file_out )

# %%
