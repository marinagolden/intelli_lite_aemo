"""
Created on 22/10/2021
"""
__author__ = "Mostafa Naemi"
__copyright__ = "Cornwall Insight Australia"

import pandas as pd
import numpy as np
import os
import yaml
import shutil
import datetime
import statistics
import time
import numpy
from   pathlib import Path
import os
import Data_import

class DataImport():

    def __init__(self,input_folder,G_drive,run_time):
        self.input_folder = input_folder
        self.G_drive      = G_drive
        self.G_drive_save = []
        self.run_time     = run_time
        
    def read_data_main(self):
        scn_manager_df, Scenario_Numbers    = self.read_scenario()
        self.G_drive                        = self.G_drive + scn_manager_df['G drive directory'].iloc[-1] + '\\' +self.run_time + '\\Inputs'
        self.G_drive_save                   = scn_manager_df['G drive save'].iloc[0] == 'Yes'
        if self.G_drive_save:
            if not os.path.exists(self.G_drive):
                os.makedirs(self.G_drive)
       
        data_main = {   'scn_mgr'          :scn_manager_df,
                        'Scenario_Numbers' :Scenario_Numbers,

                    }
        return data_main #scn_manager_df,price_data,fcas_req,carb_price,mlf_dlf,dnsp,demand,degrad,ppa_load,cpf,aemo_fees

# ========================================================================== 
    """ This function reads the scenario manager file and remove unwanted
        columns and convert numeric values from obj to float. The output 
        is a dataframe. 
    """
    def read_scenario(self):
        file_path               = os.path.join(self.input_folder , 'SIM Run.csv')
        SIM_RUN                 = pd.read_csv(file_path, index_col=None, sep=",", encoding='utf8')  
        
        # Iterate over the sim run file in case the import hasn;t worked properly
        for index, row in SIM_RUN.iterrows():
            # Check if the value is NaN in the specified column
            if pd.isna(row['Value']):
                # Extract the numbers between "" from a different column
                numbers_between_quotes = row['Parameter'].strip('""')
                numbers_between_quotes = row['Parameter'].split('"')
                # Replace NaN with the extracted numbers
                SIM_RUN.at[index, 'Parameter'] = 'Selected scenarios'
                SIM_RUN.at[index, 'Value'] = numbers_between_quotes[1]
        
        ScenarioManagerName     = SIM_RUN[SIM_RUN['Parameter']=='Scenario manager file']['Value'].iloc[0]
        Scenario_Numbers        = SIM_RUN[SIM_RUN['Parameter']=='Selected scenarios'   ]['Value'].iloc[0]

        file_path               = os.path.join(self.input_folder , ScenarioManagerName)
        scenario_manager_df     = pd.read_excel(file_path)
        scenario_manager_df.replace(np.nan, 'None', inplace=True)
        scenario_manager_df     = scenario_manager_df.drop(['#','Unit'],axis=1)
        scenario_manager_df.set_index('Assumption/input', inplace = True)
        scenario_manager_df     = scenario_manager_df.transpose()
        self.export_file(file_path)
        # convert obj to float
        cols = scenario_manager_df.columns

        for c in cols:
            try:
                scenario_manager_df[[c]] = scenario_manager_df[[c]].apply(pd.to_numeric)
            except:
                x=0
                # print("The column {} is string type".format(c))
        
        scenario_manager_df = scenario_manager_df.reset_index()
        scenario_manager_df.rename(columns={'index':'Scenario'},inplace=True)
        scenario_manager_df = self.LP_input_prepare(scenario_manager_df)
        return scenario_manager_df, Scenario_Numbers

# ========================================================================== 
    """ This function uses the mapping file of variables, and rename the 
        long named vars in Scenario manager file with a shorter name for 
        use in LP model formulations
    """
    def LP_input_prepare(self,df):
        mapping_file_dir    = os.getcwd()
        mapping_file_name   = "var_mapping.csv"
        file_path           = os.path.join(mapping_file_dir,mapping_file_name)
        mapping             = pd.read_csv(file_path)
        df                  = df.rename(columns = mapping.set_index('Scenario_manager')['model_variable'].to_dict())
        return df

# ========================================================================== 
    """ Import 'Price' excel file and export as a dataframe
    """
    def read_prices(inputs_path, Price_CV_Input, scn_mgr):
        file_path   = os.path.join(inputs_path,Price_CV_Input)
        price_data  = pd.read_csv (file_path,low_memory=False)
        
        # self.export_file(file_path)
        Shifting_Intervals =  scn_mgr['Dispatch interval (DI) shifting'].iloc[0]
        price_data         = Data_import.DataImport.Dispatch_Interval_Shifting (Shifting_Intervals, price_data  ,'PRICE', scn_mgr)

        return price_data

# ========================================================================== 
    """ Import 'FCAS Requirement Forecast' excel file and export as a dataframe,
        Currently seems not being used in the code as we limit it in the scenario manager file
    """ 
    # def read_fcas_req(self):
    #     file_path     = os.path.join(self.input_folder, 'FCAS Requirement Forecast.csv')
    #     fcas_demand   = pd.read_csv (file_path)     
    #     fcas_demand['Datetime'] = pd.to_datetime(fcas_demand['Datetime'], format = '%d/%m/%Y %H:%M')
        
    #     return fcas_demand

# ========================================================================== 
    """ Import 'Carbon Pricing' excel file and export as a dataframe
    """
    def read_carb_price(inputs_path, CarbonPricing_CV_Input):
        file_path   = os.path.join(inputs_path, CarbonPricing_CV_Input)
        carb_price  = pd.read_csv (file_path)
        # self.export_file(file_path)
        return carb_price

# ========================================================================== 
    """ Import 'DLF and MLF' excel file and export as a dataframe
    """
    def read_MLF_DLF(inputs_path, MLF_CV_Input, DLF_CV_Input):
        mlf_file_path   = os.path.join(inputs_path, MLF_CV_Input)
        MLF_df          = pd.read_csv (mlf_file_path)
        MLF_df.rename(columns = {'Battery Load MLF':'MLF_load' , 'Battery Generation MLF': 'MLF_generation' },inplace=True)
        # self.export_file(mlf_file_path)
        
        dlf_file_path   = os.path.join(inputs_path, DLF_CV_Input)
        DLF_df          = pd.read_csv(dlf_file_path)
        DLF_df.replace(np.nan, 'None', inplace=True)
        # self.export_file(dlf_file_path)
        
        MLF_df.dropna(inplace=True)
        DLF_df.dropna(inplace=True)
        return MLF_df , DLF_df

# ========================================================================== 
    """ Import 'DLF and MLF' excel file and export as a dataframe
    """
    def read_DNSP_tariff(inputs_path, DNSP_CV_Input, scn_mgr):
        file_path   = os.path.join(inputs_path, DNSP_CV_Input)
        DNSP_df     = pd.read_csv (file_path)
        # self.export_file(file_path)

        Shifting_Intervals =  scn_mgr['Dispatch interval (DI) shifting'].iloc[0]
        DNSP_df            = Data_import.DataImport.Dispatch_Interval_Shifting (Shifting_Intervals, DNSP_df, 'DNSP_TARIFF', scn_mgr)
        
        return DNSP_df

# ========================================================================== 
    """ Import 'Demand' excel file and export as a dataframe
    """
    def read_demand(inputs_path, Demand_Input):
        file_path   = os.path.join(inputs_path, Demand_Input)
        demand_df   = pd.read_csv (file_path)
        demand_df['Datetime'] = pd.to_datetime(demand_df[['Year', 'Month', 'Day']])
        # demand_df['DATE'] = demand_df['Datetime'].dt.date
        return demand_df

# ========================================================================== 
    """ Import 'Degradation' excel file and export as a dataframe
    """
    def read_degradation(self,file_path):  
        file_path       = os.path.join(file_path)
        degradation_df  = pd.read_csv (file_path)
        degradation_df  = degradation_df.iloc[:,0:3]
        self.export_file(file_path)
        return degradation_df

# ========================================================================== 
    """ Import 'Causer Pays Factor' excel file and export as a dataframe
    """
    def read_CPF(inputs_path, CauserPaysFcas_CV_Input ):
        try:
            file_path   = os.path.join(inputs_path, CauserPaysFcas_CV_Input )
            cpf         = pd.read_csv (file_path)
        except:
            cpf =[]
        return cpf

# ========================================================================== 
    """ Import 'AEMO and Market Fees and Charges' excel file and export as a dataframe
    """
    def read_aemo_fees(inputs_path, AEMO_charges_CV_Input):
        file_path           = os.path.join(inputs_path, AEMO_charges_CV_Input)
        market_charges_df   = pd.read_csv (file_path)
        market_charges_df   = market_charges_df.iloc[: , 0:4]
        # self.export_file(file_path)
        return market_charges_df

# ========================================================================== 
    """ Define model cofiguration and parameters from 'yaml_file'
    """
    def read_model_config(self,yaml_file):
        with open(yaml_file) as file:
            config   = yaml.full_load(file)
        parameters   = config ['parameters']
        model_config = config ['model'     ]
        return parameters , model_config

 # ========================================================================== 
    """ Import 'Curtailment profile' excel file and export as a dataframe
    """
    def read_curtailment(inputs_path, Curtailment_CV_Input, scn_mgr):

        behind_meter       = (scn_mgr['Behind_meter'] == 'Yes').sum() >0
        if behind_meter:
            file_path   = os.path.join(inputs_path, Curtailment_CV_Input)
            curtail_df  = pd.read_csv (file_path)
            curtail_df.insert(loc = 0, column = 'Date' , value = pd.to_datetime(curtail_df[['Year','Month','Day']]))
            curtail_df.rename(columns={'Interval':'Period'},inplace=True)

            Shifting_Intervals =  scn_mgr['Dispatch interval (DI) shifting'].iloc[0]
            curtail_df         = Data_import.DataImport.Dispatch_Interval_Shifting (Shifting_Intervals, curtail_df, 'SOLAR_PROFILE', scn_mgr)

        else:
            curtail_df = []
        # self.export_file(file_path)
        return curtail_df
   # ========================================================================== 
    """ Import 'Residential load profile' excel file and export as a dataframe ##copy of Curtailment profile
    """
    def read_residential(inputs_path, Residential_Load__CV_Input, scn_mgr):
    
        behind_meter       = (scn_mgr['Behind_meter'] == 'Yes').sum() >0 # Behind_meter input is True
        residential_load_exists = Residential_Load__CV_Input != "None" # is there a residential load csv input in scenario manager
        
        if behind_meter and residential_load_exists:
            file_path   = os.path.join(inputs_path, Residential_Load__CV_Input)
            residential_df  = pd.read_csv (file_path)
            residential_df.insert(loc = 0, column = 'Date' , value = pd.to_datetime(residential_df[['Year','Month','Day']]))
            residential_df.rename(columns={'Interval':'Period'},inplace=True)
    
            Shifting_Intervals =  scn_mgr['Dispatch interval (DI) shifting'].iloc[0]
            residential_df         = Data_import.DataImport.Dispatch_Interval_Shifting (Shifting_Intervals, residential_df, 'LOAD_PROFILE', scn_mgr)
    
        else:
            residential_df = pd.DataFrame()
        # self.export_file(file_path)
        return residential_df
  
  # ========================================================================== 
    """ Import 'RTE' excel file and export as a dataframe
    """       
    def read_RTE_curve(self,file_path):  
        # file_path = os.path.join(self.input_folder + '\\' + 'RTE_Aquila.csv')
        file_path   = os.path.join(file_path)
        RTE_df      = pd.read_csv (file_path)
        self.export_file(file_path)
        return RTE_df

 # ========================================================================== 
    """ Import 'PPA load' excel file and export as a dataframe
    """       
    def read_ppa_load(inputs_path, PPA_Load_CSV_Input, scn_mgr  ):  
        file_path   = os.path.join(inputs_path, PPA_Load_CSV_Input  )
        load_df     = pd.read_csv (file_path)
        # self.export_file(file_path)

        Shifting_Intervals =  scn_mgr['Dispatch interval (DI) shifting'].iloc[0]
        load_df    = Data_import.DataImport.Dispatch_Interval_Shifting (Shifting_Intervals, load_df, 'PPA_LOAD', scn_mgr)

        return load_df

 # ========================================================================== 
    """ Import 'Charge & discharge ratio' excel file and export as a dataframe
    """           
    def read_discharge_charge_ratio(self,file_path, flow):
        file_path   = os.path.join(file_path)
        ratios_df   = pd.read_csv (file_path)
        if flow =='Charge':
            df = ratios_df[['total cycles','Charge']] 
        else:
            df = ratios_df[['total cycles','Discharge']] 
        self.export_file(file_path)
        return df

 # ========================================================================== 
    """ Import 'Minimum average spread' excel file and export as a dataframe
    """           
    def read_min_avg_spread(self,file_path):
        file_path  = os.path.join(file_path)
        min_spread = pd.read_csv (file_path)
        self.export_file(file_path)
        return min_spread        

 # ========================================================================== 
    """ Import 'Minimum average spread' excel file and export as a dataframe
    """           
    def read_max_cycle(self,file_path):
        file_path = os.path.join(file_path)
        max_cycle = pd.read_csv (file_path)
        self.export_file(file_path)
        return max_cycle   

 # ========================================================================== 
    """ Copying the export files to the destination location
    """           
    def export_file(self,file_path):
        if self.G_drive_save:
            shutil.copy(file_path, self.G_drive)
 # ========================================================================== 
    """ Import 'BESS tolling factor' excel file and export as a dataframe
    """
    def read_BESS_Tolling(inputs_path, BESS_Tolling_Risk_Factor, scn_mgr ):  
        file_path        = os.path.join(inputs_path, BESS_Tolling_Risk_Factor )
        Toll_Risk_Factr  = pd.read_csv (file_path)
        # self.export_file(file_path)

        Shifting_Intervals = scn_mgr['Dispatch interval (DI) shifting'].iloc[0]
        Toll_Risk_Factr    = Data_import.DataImport.Dispatch_Interval_Shifting (Shifting_Intervals, Toll_Risk_Factr, 'TOLL_RISK_FACTOR', scn_mgr)

        return Toll_Risk_Factr    

 # ========================================================================== 
    """ Shifting Time Intervals 
    """ 
    def Dispatch_Interval_Shifting(Shifting_Intervals, DataFrame, Switch, scn_mgr):
        Num_DI = (scn_mgr['Run_Period']*(60/scn_mgr['Resolution'] )).iloc[0] # number of dispatch intervals

        if  Switch =='PRICE':
            #  ----- Shifting Time Intervals of Price datafram-------------------
            state = scn_mgr['Location'].iloc[0]
            price = DataFrame.copy()
            price.loc[1:, state:] = price.loc[1:, state:].shift(-Shifting_Intervals,fill_value=0)
            # price = price.fillna(0)        
            return price     

        if Switch =='SOLAR_PROFILE':
            curtail = DataFrame.copy()
            #  ----- Shifting Time Intervals of Solar profile and curtailment-----
            if len(curtail)!=0: # FTM->Empty list , BTM->Datafram
                curtail.loc[0:, 'Solar_MW':] = curtail.loc[0:, 'Solar_MW':].shift(-Shifting_Intervals, fill_value=0)
                return curtail
        
        if Switch =='LOAD_PROFILE':
            residential = DataFrame.copy()
            #  ----- Shifting Time Intervals of LOAD profile and residentialment-----
            if len(residential)!=0: # FTM->Empty list , BTM->Datafram
                residential.loc[0:, 'Load_MW':] = residential.loc[0:, 'Load_MW':].shift(-Shifting_Intervals, fill_value=0)
                return residential   
                
        if Switch =='PPA_LOAD':
            ppa_load = DataFrame.copy()
            ppa_load.loc[0:, 'Load_MW':] = ppa_load.loc[0:, 'Load_MW':].shift(-Shifting_Intervals, fill_value=0)
            return ppa_load    
                
        if Switch =='TOLL_RISK_FACTOR':
            Toll_Risk_Factr = DataFrame.copy()
            TOP             = Toll_Risk_Factr.loc[0:, 'Q1':] .head(Shifting_Intervals)                           # Get top rows
            Toll_Risk_Factr.loc[0:, 'Q1':] = Toll_Risk_Factr.loc[0:, 'Q1':].shift(-Shifting_Intervals)           # Shift Up
            NumRows         =Toll_Risk_Factr.shape[0]                                                            # Replace top rows with bottom rows
            Toll_Risk_Factr.loc[NumRows-Shifting_Intervals:,'Q1':]=TOP.values
            return Toll_Risk_Factr    
        
        if Switch == 'DNSP_TARIFF':
  
            DNSP_tariff = DataFrame.copy()
            # Define a function to shift dispatch intervals
            def Shift_Dispatch_Intervals(x):
                try:
                    nums = str(x).split(',')
                    if '0' in nums:
                        # Do not change row with 0 value
                        return x
                    else:
                        result = ','.join([str(int(i)-Shifting_Intervals) for i in nums])
                        result = ','.join([str(int(i)+int(Num_DI)) if int(i)<1 else i for i in result.split(',')])
                        # result = ','.join(sorted(result.split(','), key=lambda x: int(x)))
                        return result
                except ValueError:
                        return 0    

            # Apply the function to the column 'x' 
            DNSP_tariff['peak demand periods'    ] = DNSP_tariff['peak demand periods'    ].apply(Shift_Dispatch_Intervals)
            DNSP_tariff['volume peak periods'    ] = DNSP_tariff['volume peak periods'    ].apply(Shift_Dispatch_Intervals)
            DNSP_tariff['volume shoulder periods'] = DNSP_tariff['volume shoulder periods'].apply(Shift_Dispatch_Intervals)
            DNSP_tariff['off peak volume periods'] = DNSP_tariff['off peak volume periods'].apply(Shift_Dispatch_Intervals)

            z=0
            return DNSP_tariff