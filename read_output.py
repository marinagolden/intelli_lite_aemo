#%%
import pandas as pd
directory ='C:\\Users\\mostafa.naemi\\OneDrive - Cornwall Insight Ltd\\Documents\\Github\\BESS\\BayWa\\2021-11-30 18-24\\Scenario 3 - VIC - 90MW-180MWh (BtM).csv'
df = pd.read_csv(directory,header=[0,1],index_col=0)
# %%
regions = ['VIC','SA','QLD']
for r in regions:
    if r in directory:
        region = r+'1'



df[('Datetime','Unnamed: 1_level_1')] = pd.to_datetime(df[('Datetime','Unnamed: 1_level_1')] )
df[('Datetime','Day')] = df[('Datetime','Unnamed: 1_level_1')].dt.day
df[('Datetime','Quart')] = df[('Datetime','Unnamed: 1_level_1')].dt.quarter
df[('Datetime','Month')] = df[('Datetime','Unnamed: 1_level_1')].dt.month
df[('Datetime','Year')] = df[('Datetime','Unnamed: 1_level_1')].dt.year
df[('Datetime','Date')] = df[('Datetime','Unnamed: 1_level_1')].dt.date


svc_raise = ['energy_volumes_discharge','FCAS_rreg_volumes_discharge','FCAS_r6sec_volumes_discharge','FCAS_r60sec_volumes_discharge','FCAS_r5min_volumes_discharge']
df[('Throughput' , 'Period')] = df['Throughput' ][svc_raise].sum(axis=1)
df[('Throughput' , 'Total_day')] = df['Throughput' ]['Period'].cumsum()

#%%
df[('Datetime','Unnamed: 1_level_1')]  = pd.to_datetime(df[('Datetime','Unnamed: 1_level_1')])
df.set_index(('Datetime','Unnamed: 1_level_1'),inplace=True)
rev_cost_list = ['BTM_lreg_vol_charge_export_avail_gen','FCAS_l5min_volumes_charge','FCAS_l6sec_volumes_charge','FCAS_l60sec_volumes_charge','FCAS_lreg_volumes_charge',
            'FCAS_r5min_volumes_discharge','FCAS_r60sec_volumes_discharge','FCAS_r6sec_volumes_discharge',
            'FCAS_rreg_volumes_discharge','energy_volumes_charge','energy_volumes_discharge']

rev_list =  ['BTM_lreg_vol_charge_export_avail_gen','FCAS_l5min_volumes_charge','FCAS_l6sec_volumes_charge','FCAS_l60sec_volumes_charge','FCAS_lreg_volumes_charge',
            'FCAS_r5min_volumes_discharge','FCAS_r60sec_volumes_discharge','FCAS_r6sec_volumes_discharge',
            'FCAS_rreg_volumes_discharge','energy_volumes_discharge']

df_daily = df['Rev'][rev_cost_list].resample('D').sum()
hdr = pd.MultiIndex.from_product([['Daily_rev'],list(df_daily.columns)])
df_daily.columns = hdr

df_daily_fcas_energy = df['FCAS_Energy_Revenue'].resample('D').sum()
hdr = pd.MultiIndex.from_product([['Daily_FCAS_Energy'],list(df_daily_fcas_energy.columns)])
df_daily_fcas_energy.columns = hdr

df_daily_cost = df['Cost'].resample('D').sum()
hdr = pd.MultiIndex.from_product([['Daily_Cost'],list(df_daily_cost.columns)])
df_daily_cost.columns = hdr



df  = df.merge(df_daily,right_index=True,left_index=True , how='left')
df = df.merge(df_daily_fcas_energy,right_index=True,left_index=True,how='left')
df = df.merge(df_daily_cost,right_index=True,left_index=True,how='left')
df = df.fillna(method= 'ffill')


df[('Rev','Daily')] = df['Daily_rev'][rev_list].sum(axis=1) + df['Daily_FCAS_Energy']['FCAS_rreg_volumes_discharge'] 
df[('Daily_Cost','Daily_Energy')] = df['Daily_rev']['energy_volumes_charge']+df['Daily_FCAS_Energy']['FCAS_lreg_volumes_charge']

df_empty =pd.DataFrame(columns=list('ABCDEFGHIJKLMNO'))
hdr = pd.MultiIndex.from_product([['Empty'],list(df_empty.columns)])
df_empty.columns = hdr
df=df.join(df_empty)

# df[('Rev','energy_volumes_charge')] = -df[('Rev','energy_volumes_charge')] 
#%%
cols = df.columns
col = []
col.append(('Datetime','Day')) #
col.append(('Datetime','Quart')) #
col.append(('Datetime','Month')) #
col.append(('Datetime','Year')) #
col.append(('Datetime','Date')) #
col.append(('Period','Unnamed: 2_level_1'))


col.append(('Value','battery_current_cap'))
col.append(('Empty','A')) #charging status
col.append(('Empty','B')) #discharging status

col.append(('Value','energy_volumes_discharge'))
col.append(('Throughput','energy_volumes_discharge'))
col.append(('Value','energy_volumes_charge'))
col.append(('Throughput','energy_volumes_charge'))
col.append(('Value','FCAS_rreg_volumes_discharge'))
col.append(('Throughput','FCAS_rreg_volumes_discharge'))
col.append(('Value','FCAS_r6sec_volumes_discharge'))
col.append(('Throughput','FCAS_r6sec_volumes_discharge'))
col.append(('Value','FCAS_r60sec_volumes_discharge'))
col.append(('Throughput','FCAS_r60sec_volumes_discharge'))
col.append(('Value','FCAS_r5min_volumes_discharge'))
col.append(('Throughput','FCAS_r5min_volumes_discharge'))
col.append(('Value','FCAS_lreg_volumes_charge'))
col.append(('Throughput','FCAS_lreg_volumes_charge'))
col.append(('Value','FCAS_l6sec_volumes_charge'))
col.append(('Throughput','FCAS_l6sec_volumes_charge'))
col.append(('Value','FCAS_l60sec_volumes_charge'))
col.append(('Throughput','FCAS_l60sec_volumes_charge'))
col.append(('Value','FCAS_l5min_volumes_charge'))
col.append(('Throughput','FCAS_l5min_volumes_charge'))
col.append(('Value','soc_begin_period'))
col.append(('Value','soc_end_period'))
col.append(('Price',region))
col.append(('Throughput','Period'))
col.append(('Throughput','Total_day'))

col.append(('Empty','C')) #30day avg cycles
col.append(('Empty','D')) # max daily cycles
col.append(('Empty','E')) # 30 day avg $/MWh rev
col.append(('Empty','F')) # 30 day avg $/MWh spread
col.append(('Empty','G')) #min avg spread $/MWh

col.append(('Rev','Daily'))

col.append(('Empty','H')) #Cost
col.append(('Empty','I')) #Profit

col.append(('Daily_rev','energy_volumes_discharge'))
col.append(('Daily_rev','FCAS_rreg_volumes_discharge'))
col.append(('Daily_rev','FCAS_r6sec_volumes_discharge'))
col.append(('Daily_rev','FCAS_r60sec_volumes_discharge'))
col.append(('Daily_rev','FCAS_r5min_volumes_discharge'))
col.append(('Daily_rev','FCAS_lreg_volumes_charge'))
col.append(('Daily_rev','FCAS_l6sec_volumes_charge'))
col.append(('Daily_rev','FCAS_l60sec_volumes_charge'))
col.append(('Daily_rev','FCAS_l5min_volumes_charge'))
col.append(('Daily_rev','BTM_lreg_vol_charge_export_avail_gen'))

col.append(('Price','RAISEREGRRP'))
col.append(('Price','LOWERREGRRP'))
col.append(('Daily_Cost','Daily_Energy'))
col.append(('Daily_Cost','Network_cost'))
col.append(('Daily_Cost','LGC_STC_cost'))

col.append(('Empty','J')) #Wind period rev

col.append(('Rev','energy_volumes_discharge'))
col.append(('Rev','FCAS_rreg_volumes_discharge'))
col.append(('FCAS_Energy_Revenue','FCAS_rreg_volumes_discharge'))
col.append(('Rev','FCAS_r6sec_volumes_discharge'))
col.append(('Rev','FCAS_r60sec_volumes_discharge'))
col.append(('Rev','FCAS_r5min_volumes_discharge'))
col.append(('Rev','FCAS_lreg_volumes_charge'))
col.append(('Rev','FCAS_l6sec_volumes_charge'))
col.append(('Rev','FCAS_l60sec_volumes_charge'))
col.append(('Rev','FCAS_l5min_volumes_charge'))
col.append(('FCAS_Energy_Revenue','FCAS_lreg_volumes_charge'))
col.append(('Rev','energy_volumes_charge'))
col.append(('Cost','LGC_STC_cost'))
col.append(('Cost','Network_cost'))

col.append(('Empty','K')) #PPA purchase cost 
col.append(('Empty','L')) #Originak MWh load
col.append(('Empty','M')) #Total PPA load purchase
col.append(('Empty','N')) # Network demand


col.append(('Rev','BTM_co_loc_gen_energy_export'))
col.append(('Value','BTM_energy_vol_charge_curtail'))

col.append(('Empty','N')) # Avail curtail

col.append(('Value','BTM_co_loc_gen_energy_export'))
col.append(('Value','BTM_energy_vol_charge_export_avail_gen'))
col.append(('Value','BTM_lreg_vol_charge_export_avail_gen'))



col.append(('Price','LOWERREGRRP'))
col.append(('Price','RAISEREGRRP'))
col.append(('Price','LOWER5MINRRP'))
col.append(('Price','LOWER6SECRRP'))
col.append(('Price','LOWER60SECRRP'))
col.append(('Price','RAISE5MINRRP'))
col.append(('Price','RAISE6SECRRP'))
col.append(('Price','RAISE60SECRRP'))
								



df_ =df[col]

df_.to_csv('new_out.csv')



# %%
import numpy as np
cols_dem = []
charge_dem = 2.92 * 1000
power_factor = 0.93
cols_dem.append(('Datetime','Day')) #
cols_dem.append(('Datetime','Quart')) #
cols_dem.append(('Datetime','Year')) #
cols_dem.append(('Datetime','Date')) #
cols_dem.append(('Period','Unnamed: 2_level_1'))
cols_dem.append(('Value','energy_volumes_charge'))
cols_dem.append(('Value','FCAS_lreg_volumes_charge'))

df_dem = df[cols_dem]
df_dem[('Value','Total_charge')] = df_dem[('Value','energy_volumes_charge')] + df_dem[('Value','FCAS_lreg_volumes_charge')]
#%%
df_dem_peak = df_dem.loc[(df_dem[('Period','Unnamed: 2_level_1')] >= 15) & (df_dem[('Period','Unnamed: 2_level_1')] <=39) ] 
df_dem_peak[('Value','MVA')] = np.sqrt(((1- power_factor)*df_dem[('Value','Total_charge')])**2 +(df_dem[('Value','Total_charge')])**2 )
df_dem_peak=df_dem_peak.resample('A').max()
df_dem_peak[('Value','Demand_charge')] = df_dem_peak[('Value','MVA')] * charge_dem
# %%
