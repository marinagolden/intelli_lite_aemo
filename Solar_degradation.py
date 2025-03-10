__author__ = 'Mostafa Naemi'
__copyright__ = 'Cornwall Insight Australia'
__date__ = '2022/10/11'
#%%
import pandas as pd
import numpy as np

interp = True

file_dir = 'C:\\Users\\mostafa.naemi\\OneDrive - Cornwall Insight Ltd\\Documents\\Github\\BESS\\Inputs\\Keppel\\'
file_name = 'Harlin_Shifted_1hr_fwd.csv' #PVsyst file
deg_file_name = 'degradation_solar_Keppel_Part2.csv' #degradation curve
out_file = 'degraded_solar_profile_part2.csv'

solar = pd.read_csv(file_dir+file_name)
solar['Time'] = pd.to_datetime(solar['Time'])
solar['Month'] = solar.Time.dt.month
solar['Day'] = solar.Time.dt.day
solar['Time'] = solar.Time.dt.time


# %% create HH profile from hourly output of the PVsyst
time = pd.date_range('2025-10-01','2045-10-01',freq='30min') #simulation time for BESS
df = pd.DataFrame()

df['Datetime'] = time
df['Time'] = df['Datetime'].dt.time
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Interval'] = df['Datetime'].dt.hour*2 + df['Datetime'].dt.minute / 30 + 1 

df['Year'] = df['Datetime'].dt.year

df_new = df.merge(solar,on =['Month','Day','Time'],how='left')
df_new = df_new.interpolate()


#%% adjusting the curve based on the degradation of panels
solar_degradation = pd.read_csv(file_dir+deg_file_name) 
df_new = df_new.merge(solar_degradation , on='Year',how='left')
df_new = df_new.fillna(method ='bfill')
df_new['MW_DC_degraded'] = df_new['MW_DC'] * df_new['Capacity']
df_new['MW_AC_degraded'] = df_new['MW_AC'] * df_new['Capacity']

df_new.to_csv(file_dir+out_file)
