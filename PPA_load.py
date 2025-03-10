__author__ = 'Mostafa Naemi'
__copyright__ = 'Cornwall Insight Australia'
#%%
import pandas as pd
import numpy as np
import datetime as dt

dir_file = 'C:\\Users\\mostafa.naemi\\OneDrive - Cornwall Insight Ltd\\Documents\\Github\\BESS\\Inputs\\BNRG\\Bairnsdale\\'
ppa_daily_profile = 'PPA_load_daily_Bairnsdale.csv'
out_file = 'PPA_Load_BNRG_16flat.csv'


ppa_daily_load = pd.read_csv(dir_file + ppa_daily_profile)

start_date = '2025-01-01' #SIM start date
end_date = '2055-01-01'   #SIM end date

ppa_start_date = '2025-01-01' #PPA contract start date
ppa_end_date = '2034-12-31'   #PPA contract end date

start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')

ppa_start_date = dt.datetime.strptime(ppa_start_date, '%Y-%m-%d').date()
ppa_end_date = dt.datetime.strptime(ppa_end_date, '%Y-%m-%d').date()

# %%
time = pd.date_range(start_date,end_date,freq='30min')
df = pd.DataFrame()

df['Datetime'] = time
df['Date'] = df['Datetime'].dt.date
df['Period'] = df['Datetime'].dt.hour*2 + df['Datetime'].dt.minute / 30 + 1 
df['Year'] = df['Datetime'].dt.year

df = df.merge(ppa_daily_load,on='Period',how='left')

df.loc[(df.Date<ppa_start_date) | (df.Date>ppa_end_date),'Load_MW'] = 0

col = ['Date','Period','Load_MW']

df = df[col]
df.set_index('Date',inplace=True)

#%%
df.to_csv(dir_file+out_file)

# %%
