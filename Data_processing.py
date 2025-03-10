__author__      = "Mostafa Naemi"
__copyright__   = "Cornwall Insight Australia"
import pandas       as pd
import numpy        as np
import utilities    as ut
from   datetime         import date, timedelta

def select_DLF(df,current_year,DNSP_selection,DLF_DNSP_connection,DLF_cogen_connection=None):

        df        = df.loc[(df['Year'      ] == current_year) & (df['DNSP'] == DNSP_selection.iloc[0])]
        DLF_df    = df.loc[ df['Connection'] == DLF_DNSP_connection.iloc[0]]
        DLF_year  = DLF_df['DLF'].iloc[0]
        if DLF_cogen_connection.iloc[0] == 'None':
            DLF_co_gen= 1
        else:
            DLF_co_gen_df   = df.loc[df['Connection'] == DLF_cogen_connection.iloc[0]]
            DLF_co_gen      = DLF_co_gen_df['DLF'].iloc[0]
        return DLF_year,DLF_co_gen

def select_MLF(df,current_year):
        MLF_df = df.loc[(df['Year'] == current_year)]
        return MLF_df


def select_aemo_fees_scenario(df,selected_jurisdiction):
    df = df.loc[df.State == selected_jurisdiction.values[0]]
    df = df.pivot_table(columns = 'Fee or market charge', values='Rate')
    return df

def select_price_scenario(df,selected_jurisdiction,energy_price_forecast_scenario,FCAS_price_forecast_scenario):
    """ this function selects price scenarios from the price dataframe loaded 
    from the Prices.csv"""

    df_ = df.copy()
    if pd.isna(df_.loc[0, 'Datetime']):
        df_.loc[0,['Datetime','PERIODID']] = ''
    # Create column names as a combination of regions and scenario name
    df_.columns = df_.columns + '_' + df_.iloc[0].astype(str)
    cols = df_.columns#.levels[0]
    for c in cols:
        if type(c) != float: 
            if  c.find('.') != -1:
                first_half  = c.split('.')[0]
                second_half = c.split('.')[1][1:]
                new_c       = first_half  + second_half
                df_.rename(columns = {c:new_c},inplace=True)
    # Drop scenario name row
    df_.drop(0,inplace=True)      
    df_.rename(columns = {'Datetime_':'Datetime' , 'PERIODID_':'PERIODID' },inplace=True)      
    df_['Datetime'] = pd.to_datetime(df_['Datetime'], dayfirst=True, errors='coerce')
    df_.set_index('Datetime',inplace=True)
    
    sel_cols = []
    sel_cols = [(selected_jurisdiction+ '_'+ energy_price_forecast_scenario).values[0]]
    svc      = ['RAISE1SECRRP','RAISE6SECRRP',	'RAISE60SECRRP',	'RAISE5MINRRP',	'RAISEREGRRP',	'LOWER1SECRRP',	'LOWER6SECRRP',	'LOWER60SECRRP',	'LOWER5MINRRP',	'LOWERREGRRP']
    for s in svc:
        sel_col = s+'_'+FCAS_price_forecast_scenario.values[0]
        sel_cols.append(sel_col)
    sel_cols.append('PERIODID')
    df_ = df_[sel_cols]
    df_.reset_index(inplace=True)

    #remove scenario name from col names
    for c in df_.columns:
        if c in(['VIC1','TAS1','QLD1','NSW1','SA1']):
            df_.rename(columns= {c:'ENERGY'},inplace = True)
        c_ = c.split('_')[0]
        df_.rename(columns = {c:c_}, inplace=True )

    #convert object to numeric values
    for c in df_.columns:
        if c!= 'Datetime':
            df_[[c]] = df_[[c]].apply(pd.to_numeric)
    
    return df_

def get_daily_price(Foresight_period, df,date):
    df = df.loc[df.Datetime.dt.date >= date]
    df = df.head(Foresight_period)
    df['PERIODID']= pd.Series(range(1,len(df. axes[0])+1 )).values

    # for c in df.columns:
    #     if c!= 'Datetime':
    #         df[[c]] = df[[c]].apply(pd.to_numeric)
    return df

def select_LGC_price(df,current_year):
    df_ = df.copy()
    if np.isnan(df_.iloc[0]['Year']):
        df_.drop(0,inplace= True)
    df_ = df_.loc[df.Year == current_year]

    for c in df_.columns:
        df_[[c]] = df_[[c]].apply(pd.to_numeric)
    return df_
       
def degradation_battery_energy(df , num_cycles, battery_init_cap): ## got to set up the degradation number at the end of each loop  
    """ this will calculate the battery degradation
    ----
    df: is the degradation curve csv input
    num_cycles: num of cycles to date which is output of post-processing functions
    battery_init_cap: battery initial capacity
    """
    df                  = df.loc[df['total cycles todate']<=round(num_cycles)]
    degradation_factor  = df.iloc[-1]['degradation_percentage']
    battery_current_cap = battery_init_cap*degradation_factor
    return battery_current_cap.iloc[0]

       
def set_up_DNSP_tariff(Foresight_period, df, date, DNSP_tariff,DNSP_selection ,resolution):
    """ this function prepares the DNSP tariff  
    for the optimisation model and also the post-process functions"""

    peakday                 = date.weekday() < 8                                                            # If it is week day or weekend
    df                      = df.loc[(df['DNSP'] == DNSP_selection.values[0]) & (df['Tariff'] == DNSP_tariff.values[0]) & (df['Year'] == date.year)]  # Select row, corresponding to DNSP, Tariff, and Year for this iteration
    num_intervals           = int(Foresight_period)
    # Charge for each interval
    df_out                  = pd.DataFrame()                                                                # Define output dataframe   
    df_out['Period']        = range(1,num_intervals+1)                                                      # In output dataframe, define a column named 'Period'
    df_out                  = ut.postproc_period_to_datetime(df_out,date,resolution)                        # In output dataframe, define a column named 'Datetime'
    df_out['fixed_charge']  = df['Fixed ($/day)'].iloc[0]/num_intervals                                     # In output dataframe, define a column named 'Fixed ($/day)', calculate fixed cost for each interval

    # Months Peak & off-peak
    peak_demand_months      = check_for_input_error_split(df['Demand Peak months'])                          # Select 'Demand Peak months' from excel file 
    peak_demand_months      = check_for_input_error_map(peak_demand_months)                                            # Change 'Demand Peak months' to series

    off_peak_demand_months  = check_for_input_error_split(df['Demand Off-Peak months'])                               # Select 'Demand Off-Peak months' from excel file 
    off_peak_demand_months  = check_for_input_error_map(off_peak_demand_months)                                     # Change 'Demand Off-Peak months' to series

    # Peak dispatch intervals
    peak_demand_periods     = check_for_input_error_split(df['peak demand periods'])                                  # Select 'peak periods' from excel file 
    peak_demand_periods     = check_for_input_error_map(peak_demand_periods)                                           # Change 'peak periods' to series


    df_out['Peak_month'    ] = df_out.Datetime.dt.month.isin(peak_demand_months    ).astype(int)            # In output dataframe, define a column named 'Peak_month'     valued 0 or 1 if today is Peak_month, based on corresponding series calculated above
    df_out['Off_peak_month'] = df_out.Datetime.dt.month.isin(off_peak_demand_months).astype(int)            # In output dataframe, define a column named 'Off_peak_month' valued 0 or 1 if today is Peak_month, based on corresponding series calculated above
    df_out['Peak_period'   ] = df_out.Period.           isin(peak_demand_periods   ).astype(int)            # In output dataframe, define a column named 'Peak_period'    valued 0 or 1 if today is Peak_month, based on corresponding series calculated above
    df_out['Peak_day'      ] = int(peakday)

    df_out['Peak_month_charge'    ] = df_out['Peak_day'] * df_out['Peak_month'    ] * df_out['Peak_period'] * df['Demand Peak ($/kVA/month)'    ].values * 1000 # convert kW to MW
    df_out['Off_peak_month_charge'] = df_out['Peak_day'] * df_out['Off_peak_month'] * df_out['Peak_period'] * df['Demand Off-Peak ($/kVA/month)'].values * 1000 # convert Kw to MW
    # df_out['Peak_period_charge'] =  df_out['Peak_day'] * df_out['Peak_period'] * df['Demand Off-Peak ($/kVA/month)'].values / 1000 # convert Kw to MW

    df_out['DNSP_demand_tariff'] =  df_out['Peak_month_charge'] + df_out['Off_peak_month_charge'] #+ df_out['Peak_period_charge']
    df_out.drop(['Peak_month','Off_peak_month','Peak_period'],axis=1,inplace=True)
    
    
    #peak_volume_months      = df['peak volume months'].iloc[0].split(",")
    #peak_volume_months      = list(map(int, peak_volume_months))
    peak_volume_months      = check_for_input_error_split(df['peak volume months'])
    peak_volume_months  = check_for_input_error_map(peak_volume_months)

    #shoulder_volume_months  = df['shoulder volume months'].iloc[0].split(",")
    #shoulder_volume_months  = list(map(int, shoulder_volume_months))
    shoulder_volume_months  = check_for_input_error_split(df['shoulder volume months'])
    shoulder_volume_months = check_for_input_error_map(shoulder_volume_months)

    #peak_volume_periods     = df['volume peak periods'].iloc[0].split(",")
    #peak_volume_periods     = list(map(int, peak_volume_periods))
    peak_volume_periods     = check_for_input_error_split(df['volume peak periods'])
    peak_volume_periods     = check_for_input_error_map(peak_volume_periods)

    #shoulder_volume_periods = df['volume shoulder periods'].iloc[0].split(",")
    #shoulder_volume_periods = list(map(int, shoulder_volume_periods))
    shoulder_volume_periods = check_for_input_error_split(df['volume shoulder periods'])
    shoulder_volume_periods = check_for_input_error_map(shoulder_volume_periods)


    df_out['Peak_volume_months'     ] = df_out.Datetime.dt.month.isin(peak_volume_months     ).astype(int)
    df_out['shoulder_volume_months' ] = df_out.Datetime.dt.month.isin(shoulder_volume_months ).astype(int)
    df_out['peak_volume_periods'    ] = df_out.Period.           isin(peak_volume_periods    ).astype(int)
    df_out['shoulder_volume_periods'] = df_out.Period.           isin(shoulder_volume_periods).astype(int)
    df_out['shoulder_volume_charge' ] = df_out['Peak_day'] * df_out['shoulder_volume_periods'] * (1- df_out['peak_volume_periods']) * df_out['shoulder_volume_months'] * df['shoulder volume ($/kWh)'].values*1000
    df_out['peak_volume_charge'     ] = df_out['Peak_day'] * df_out['peak_volume_periods'    ] *     df_out['Peak_volume_months' ]  * df    ['peak volume ($/kWh)'   ].values*1000
    if peakday :
        df_out['off_peak_volume_charge'] = (1-df_out['shoulder_volume_periods']) * df['volume-off peak ($/kWh)'].values*1000
    else:
        df_out['off_peak_volume_charge'] = (1-df_out['Peak_day'               ]) * df['volume-off peak ($/kWh)'].values*1000

        
    df_out['DNSP_volume_tariff'] = df_out[['shoulder_volume_charge','peak_volume_charge','off_peak_volume_charge']].max(axis=1)#df_out['shoulder_volume_charge'] , df_out['peak_volume_charge'] , df_out['off_peak_volume_charge'])
    df_out.drop(['Peak_volume_months','shoulder_volume_months','peak_volume_periods','shoulder_volume_periods'],axis=1,inplace=True)

    return df_out ,peak_demand_months ,peak_demand_periods

def avg_spread_today(daily_price,location):
    df           = daily_price[[location]]
    spread_today = df[location].max()- df[location].mean()
    return spread_today

def select_min_spread(df , date , min_value = 15 , fraction = 0.75):
    min_spread = df.loc[(df.Year == date.year) & (df.Month == date.month),'Spread'].iloc[0] * fraction
    min_spread = max(min_value , min_spread)
    return min_spread

def get_coloc_gen_curtail(NumDay_Foresight, Foresight_period, df,date,con_size,curt_active,front_meter):

    if  not front_meter:
        df = df.loc[df.Date.dt.date >= date ]
        df = df.head(Foresight_period)
        df['Period'] = [p for p in range(1,Foresight_period+1)]

        df['default_connection'] = con_size
        df['max_connection'    ] = df[['cons_trans_MW', 'cons_dist_MW', 'cons_dynamic_MW', 'cons_inverter_MW','cons_other_MW','default_connection']].min(axis=1)
        df['net_co_locate_gen_avail_export'] = np.minimum(df['max_connection'],df['Solar_MW'])   
        if curt_active == 'Yes':     
            df = df.loc[df.Date.dt.date >= date ]
            df = df.head(Foresight_period)
            df['total_curt_avail'  ] = df[['curt_trans_MW', 'curt_dist_MW', 'curt_dynamic_MW']].sum(axis=1)
            df['total_unaviod_curt'] = df[['curt_inverter_MW', 'curt_other_MW']].sum(axis=1)
        else:
            df['total_curt_avail'  ] = 0
            df['total_unaviod_curt']  =0
    else: 

        df = pd.DataFrame(columns = ['Date' , 'Period' , 'max_connection', 'total_curt_avail','total_unaviod_curt', 'net_co_locate_gen_avail_export'])
        df['Period'             ] = [p for p in range(1,Foresight_period+1)]
        df['Date'               ] = date
        df['net_co_locate_gen_avail_export'] = 0
        df['max_connection'     ] = Foresight_period * 100
        df['total_curt_avail'   ] = 0
        df['total_unaviod_curt' ] = 0
    sel_col = ['Date' , 'Period' , 'max_connection', 'total_curt_avail','total_unaviod_curt', 'net_co_locate_gen_avail_export']
    df      = df[sel_col]
    return df


def get_res_load_curtail(NumDay_Foresight, Foresight_period, df,date,con_size,curt_active,front_meter):
    if not df.empty:
        if  not front_meter:
            df = df.loc[df.Date.dt.date >= date ]
            df = df.head(Foresight_period)
            df['Period'] = [p for p in range(1,Foresight_period+1)]

            df['default_connection'] = con_size
            df['max_connection'    ] = df[['cons_trans_MW', 'cons_dist_MW', 'cons_dynamic_MW', 'cons_inverter_MW','cons_other_MW','default_connection']].min(axis=1)
            df['net_co_locate_load_avail_export'] = np.minimum(df['max_connection'],df['Load_MW'])   
            if curt_active == 'Yes':     
                df = df.loc[df.Date.dt.date >= date ]
                df = df.head(Foresight_period)
                df['total_curt_avail'  ] = df[['curt_trans_MW', 'curt_dist_MW', 'curt_dynamic_MW']].sum(axis=1)
                df['total_unaviod_curt'] = df[['curt_inverter_MW', 'curt_other_MW']].sum(axis=1)
            else:
                df['total_curt_avail'  ] = 0
                df['total_unaviod_curt']  =0
        else: 

            df = pd.DataFrame(columns = ['Date' , 'Period' , 'max_connection', 'total_curt_avail','total_unaviod_curt', 'net_co_locate_load_avail_export'])
            df['Period'             ] = [p for p in range(1,Foresight_period+1)]
            df['Date'               ] = date
            df['net_co_locate_load_avail_export'] = 0
            df['max_connection'     ] = Foresight_period * 100
            df['total_curt_avail'   ] = 0
            df['total_unaviod_curt' ] = 0
        sel_col = ['Date' , 'Period' , 'max_connection', 'total_curt_avail','total_unaviod_curt', 'net_co_locate_load_avail_export']
        df      = df[sel_col]
    return df



def RTE_battery_curve(df , num_cycles): ## got to set up the degradation number at the end of each loop  
    """ this will calculate the battery degradation
    ----
    df: is the RTE curve csv input
    num_cycles: num of cycles to date which is output of post-processing functions
    battery_init_cap: battery initial capacity
    """
    df  = df.loc[df['total cycles todate']<=round(num_cycles)]
    RTE = df.iloc[-1]['RTE']
    return RTE


def dynamic_charging_NSP(Foresight_period, date, normal_ratio ,scale , months , periods=None, start_time=None , end_time=None , daylight_shift=0 ,resolution =30):
    """daylight_shift > 0 for ahead of market time"""
    num_periods = int(Foresight_period)
    m           = date.month
    if not periods: #check if list exist
        try:
            start_period    = ut.time_to_period(start_time) - daylight_shift
            end_period      = ut.time_to_period(end_time  ) - daylight_shift
            periods_range   = range(start_period,end_period+1) 
        except:
            print('dynamic charging needs periods or start and end time for peak hours')
    else:
        periods_range   = periods - daylight_shift

    charge_ratio = {}
    for p in range(1,num_periods+1):
        charge_ratio[p] = normal_ratio
        if m in months:
            if p in periods_range:
                charge_ratio[p] = scale
    return charge_ratio

def dynamic_charging(Foresight_period, curve , num_cycles,  date, normal_ratio ,scale , months , periods=None, start_time=None , end_time=None , daylight_shift=0 ,resolution =30):
    """daylight_shift > 0 for ahead of market time"""
    #first check if there is any curve
    if not curve is None:
        curve        = curve.loc[curve['total cycles']<=round(num_cycles)]
        normal_ratio = curve.iloc[-1]['Charge']

    num_periods = int(Foresight_period)
    m           = date.month
    if not periods: #check if list exist
        try:
            start_period    = ut.time_to_period(start_time) - daylight_shift
            end_period      = ut.time_to_period(end_time  ) - daylight_shift
            periods_range   = range(start_period,end_period+1) 
        except:
            print('dynamic charging needs periods or start and end time for peak hours')
    else:
        periods_range  = periods - daylight_shift
    charge_ratio = {}
    for p in range(1,num_periods+1):
        charge_ratio[p] = normal_ratio
        if m in months:
            if p in periods_range:
                charge_ratio[p] = scale
    return charge_ratio


def dynamic_discharging(Foresight_period, curve , num_cycles,  normal_ratio ,resolution =30):
    """daylight_shift > 0 for ahead of market time"""
    #first check if there is any curve
    if not curve is None:
        curve        = curve.loc [curve['total cycles']<=round(num_cycles)]
        normal_ratio = curve.iloc[-1]['Discharge']

    num_periods = int(Foresight_period)
    discharge_ratio = {}
    for p in range(1,num_periods+1):
        discharge_ratio[p] = normal_ratio

    return discharge_ratio

# %%
def DLF_for_summary(df,DNSP_selection,DLF_DNSP_connection,DLF_cogen_connection=None):
    DLF_all = df.loc[(df['DNSP'] == DNSP_selection.iloc[0]) &(df['Connection'].isin([DLF_DNSP_connection.iloc[0] ,DLF_cogen_connection.iloc[0]]))]     
    DLF_all = DLF_all.pivot_table(index='Year',columns='Connection',values='DLF')
    DLF_all.rename(columns={DLF_cogen_connection.iloc[0]:'DLF_cogen', DLF_DNSP_connection.iloc[0]:'DLF_BESS'}, inplace=True)
    if DLF_cogen_connection.iloc[0]==DLF_DNSP_connection.iloc[0]:
        DLF_all['DLF_cogen'] = DLF_all['DLF_BESS']
    return DLF_all

def DNSP_for_summary(df, DNSP_tariff,DNSP_selection):
    df = df.loc[(df['DNSP'] == DNSP_selection.values[0]) & (df['Tariff'] == DNSP_tariff.values[0])]
    return df

def curtail_for_summary(df,con_size,curt_active,front_meter):
    df['default_connection'            ] = con_size
    df['max_connection'                ] = df[['cons_trans_MW', 'cons_dist_MW', 'cons_dynamic_MW', 'cons_inverter_MW','cons_other_MW','default_connection']].min(axis=1)
    df['net_co_locate_gen_avail_export'] = np.minimum(df['max_connection'],df['Solar_MW'])

    if curt_active == 'Yes':
        df['total_curt_avail'  ] = df[['curt_trans_MW', 'curt_dist_MW', 'curt_dynamic_MW']].sum(axis=1)
        df['total_unaviod_curt'] = df[['curt_inverter_MW', 'curt_other_MW'               ]].sum(axis=1)
    else:
        df['total_curt_avail'  ] = 0
        df['total_unaviod_curt'] = 0
    if front_meter:
        df['net_co_locate_gen_avail_export'] = 0
        df['max_connection'                ] = 48 * con_size
        df['total_curt_avail'              ] = 0
        df['total_unaviod_curt'            ] = 0
    sel_col = ['Date' , 'Period' , 'max_connection', 'total_curt_avail','total_unaviod_curt', 'net_co_locate_gen_avail_export']
    df      = df[sel_col]
    return df

def residential_for_summary(df,con_size,curt_active,front_meter):
    if not df.empty:
        df['default_connection'            ] = con_size
        df['max_connection'                ] = df[['cons_trans_MW', 'cons_dist_MW', 'cons_dynamic_MW', 'cons_inverter_MW','cons_other_MW','default_connection']].min(axis=1)
        df['net_co_locate_load_avail_export'] = np.minimum(df['max_connection'],df['Load_MW'])

        if curt_active == 'Yes':
            df['total_curt_avail'  ] = df[['curt_trans_MW', 'curt_dist_MW', 'curt_dynamic_MW']].sum(axis=1)
            df['total_unaviod_curt'] = df[['curt_inverter_MW', 'curt_other_MW'               ]].sum(axis=1)
        else:
            df['total_curt_avail'  ] = 0
            df['total_unaviod_curt'] = 0
        if front_meter:
            df['net_co_locate_load_avail_export'] = 0
            df['max_connection'                ] = 48 * con_size
            df['total_curt_avail'              ] = 0
            df['total_unaviod_curt'            ] = 0
        sel_col = ['Date' , 'Period' , 'max_connection', 'total_curt_avail','total_unaviod_curt', 'net_co_locate_load_avail_export']
        df      = df[sel_col]
    return df

def PPA_Price_Diff(price_daily, jurisdiction, year , PPA_price):
    jurisdiction = jurisdiction.iloc[0]
    PPA_price    = PPA_price.loc[PPA_price['Year'] == year, 'PPA_Price'].iloc[0]
    df           = price_daily.copy()
    df.loc[df[jurisdiction]<0,jurisdiction] = 0 
    price_diff   = pd.DataFrame()
    price_diff['Price_Diff'] =  PPA_price - df[jurisdiction]
    price_diff['Period'    ] =  df['PERIODID'].values
    return price_diff

def PPA_price_update_q4_to_q1(price_scn,jurisdiction,scn_mgr , start_date):
    """this function updates the PPA price according to first & last quarter of each 5yr window 
    e.g. PPA Price new = PPA Price old * Avg Energy price Q4 / Avg Energy price Q1
    
    Inputs:
    start_date: start date for the PPA 
    update_period: the period (in years) after which the PPA price is updated

    """
    df            = price_scn[['Datetime',jurisdiction.iloc[0]]]
    df['Year']    = df['Datetime'].dt.year
    df['Quarter'] = df['Datetime'].dt.quarter
    
    df            = df.groupby(['Year','Quarter'],as_index=False).mean()
    start_year    = start_date.year
    df            = df.loc[df['Year'] >= start_year]
    update_period = scn_mgr['PPA Update Period'].iloc[0]

    df ['PPA_Update'] = np.floor((df['Year'] - start_year) / update_period).astype(int)
    PPA_price = scn_mgr['PPA Price'].iloc[0]
    
    for i in range(df['PPA_Update'].max()+1):
        if i == 0:
            df['PPA_Price'] = PPA_price
        else:
            #update the PPA price based on first & last quarter in 5yr window
            ix        = df.PPA_Update == i
            df_       = df[ix]
            PPA_price = PPA_price * (df_[jurisdiction].iloc[-1] / df_[jurisdiction].iloc[0]).values
            df.loc[ix,'PPA_Price'] = float(PPA_price)
    
    ppa_price  = df.groupby('Year',as_index=False).mean()
    ppa_price  = ppa_price[['Year','PPA_Price']]

    return ppa_price

def PPA_price_update_q1_to_q1(price_scn,jurisdiction,scn_mgr , start_date):
    """this function updates the PPA price according to first quarters of each 5yr window 
    e.g. PPA Price new = PPA Price old * Avg Energy price Q1 / Avg Energy price Q1
    
    Inputs:
    start_date: start date for the PPA 
    update_period: the period (in years) after which the PPA price is updated

    """
    df               = price_scn[['Datetime',jurisdiction.iloc[0]]]
    df['Year'   ]    = df['Datetime'].dt.year
    df['Quarter']    = df['Datetime'].dt.quarter
    df               = df.groupby(['Year','Quarter'],as_index=False).mean()
    start_year       = start_date.year
    df               = df.loc[df['Year'] >= start_year]
    update_period    = scn_mgr['PPA Update Period'].iloc[0]
    df['PPA_Update'] = np.floor((df['Year'] - start_year) / update_period).astype(int)
    PPA_price        = scn_mgr['PPA Price'].iloc[0]
    
    for i in range(df['PPA_Update'].max()+1):
        if i == 0:
            df['PPA_Price'] = PPA_price
        else:
            #update the PPA price based on first & last quarter in 5yr window
            ix        = df.PPA_Update == i
            Q1_last   = df [(df.PPA_Update == i-1) & (df.Quarter ==1)]
            Q1_new    = df [(df.PPA_Update == i  ) & (df.Quarter ==1)]
            PPA_price = PPA_price * (Q1_new[jurisdiction].iloc[0] / Q1_last[jurisdiction].iloc[0]).values
            df.loc[ix,'PPA_Price'] = float(PPA_price)
    ppa_price = df.groupby('Year',as_index=False).mean()
    ppa_price = ppa_price[['Year','PPA_Price']]

    return ppa_price

def PPA_price_update_yearly(price_scn,jurisdiction,scn_mgr , start_date):
    """this function updates the PPA price according to first quarters of each 5yr window 
    e.g. PPA Price new = PPA Price old * Avg Energy price Q1 / Avg Energy price Q1
    
    Inputs:
    start_date: start date for the PPA 
    update_period: the period (in years) after which the PPA price is updated

    """
    df         = price_scn[['Datetime',jurisdiction.iloc[0]]]
    df['Year'] = df['Datetime'].dt.year   
    df         = df.groupby(['Year'],as_index=False).mean()
    start_year = start_date.year
    df         = df.loc[df['Year'] >= start_year]
    update_period    = scn_mgr['PPA Update Period'].iloc[0]
    df['PPA_Update'] = np.floor((df['Year'] - start_year) / update_period).astype(int)
    PPA_price        = scn_mgr['PPA Price'].iloc[0]
    
    for i in range(df['PPA_Update'].max()+1):
        if i == 0:
            df['PPA_Price'] = PPA_price
        else:
            #update the PPA price based on first & last quarter in 5yr window
            ix          = df.PPA_Update == i
            last        = df[(df.PPA_Update == i-1) ]
            new         = df[(df.PPA_Update == i) ]
            PPA_price   = PPA_price * (new[jurisdiction].iloc[0] / last[jurisdiction].iloc[0]).values
            df.loc[ix,'PPA_Price'] = float(PPA_price)
    ppa_price  = df[['Year','PPA_Price']]

    return ppa_price


def PPA_load_daily(Foresight_period, ppa_load , date):
    date    = date.strftime('%Y-%m-%d')
    df      = ppa_load.loc[ppa_load['Date'] >= date]
    df      = df.head(Foresight_period)
    df['Period']= pd.Series(range(1,len(df. axes[0])+1 )).values

    df      = df[['Period', 'Load_MW']]
    df.set_index( 'Period', inplace = True)
    df_dict = df.to_dict()['Load_MW']
    return df_dict

def PPA_percent_contract(start_date,end_date,date):
    if (date>= start_date) & (date<=end_date):
        contract = True
    else:
        contract = False
    return contract

def get_max_cycle(curve , date , freq = 'yearly'):
    if freq == 'yearly':
        df = curve.loc[curve.Year == date.year]
    return df['Max_cycle']

def check_for_input_error_split(data):
    try:
        # Checks if the inputs generate an error - has been used to date in the DNSP tariffs function
        result = data.iloc[0].split(",")  
        return result  # If no error occurs, this line will be executed
    except Exception as e:
        return [0]
    
def check_for_input_error_map(data):
    try:
        # Checks if the inputs generate an error - has been used to date in the DNSP tariffs function
        result = list(map(int, data)) 
        return result  # If no error occurs, this line will be executed
    except Exception as e:
        return [0]    