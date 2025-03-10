__author__ = 'Mostafa Naemi'
__date__ = '2022-03-24'
# this script is intended to use when the code does not ggenerate monthly output file for copying to excel sent to client file
#%%
import Data_import
import code_export
import Data_processing
import Post_processing
import pandas as pd
from dateutil.relativedelta import relativedelta
from pathlib import Path
import os
import turtle as tur

time_run = tur.textinput("Time of run","Enter time of your run based on Output folder created (yyyy-mm-dd HH-MM) ")
tur.Screen().bye()
print("your time of run is: {}".format(time_run))#'2022-03-23 17-50'
files_path = Path().absolute().parents[0]
inputs_path = str(files_path)+'\\Inputs'
G_drive = 'G:\\Work\\Australia\\Consulting\\Projects\\'
yaml_path = 'model_config.yaml'



# importing input data
DI = Data_import.DataImport(inputs_path,G_drive,str(''))
data = DI.read_data_main()

# reading model config file
param,model_config = DI.read_model_config(yaml_path)

scenario_manager = data['scn_mgr']
price = data['Price']
aemo_charges = data['AEMO']
DLF  = data['DLF']
MLF = data['MLF']
DNSP = data['DNSP']
Carb_price = data['carb_price']
curtail = data['Curtailment']

#saving on G drive/Locally
G_drive_save = scenario_manager['G drive save'].iloc[0] == 'Yes'
if G_drive_save:
    # copy code to G drive
    G_drive  = G_drive+scenario_manager['G drive directory'].iloc[0] + str(time_run)

    code_export.copy_code(os.getcwd(),G_drive)
    # ouput directory
    out_dir = os.path.join(G_drive, 'Outputs', str(time_run))
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
    #summary directory
    sum_dir = os.path.join(G_drive, 'Outputs', str(time_run))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
else:
    out_dir = os.path.join(files_path, 'Outputs', str(time_run))
    sum_dir = out_dir
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)



#%%

for s in scenario_manager.Scenario[[2]]:
    scn_mgr = scenario_manager.loc[scenario_manager.Scenario == s]
    selected_jurisdiction = scn_mgr['Location']
    energy_price_scenario  = scn_mgr['Energy price forecast']
    fcas_price_scenario = scn_mgr ['FCAS price forecast']
    DNSP_selection = scn_mgr['DNSP']
    DNSP_connection = scn_mgr['DLF']
    DNSP_tariff = scn_mgr['DNSP_tariff']
    cogen_connection = scn_mgr['DLF co-located generation']

    # behind meter 
    front_meter = scn_mgr['Behind_meter'].iloc[0] == 'No'
    BTM = not front_meter    

    price_scn = Data_processing.select_price_scenario(price,selected_jurisdiction,energy_price_scenario,fcas_price_scenario)
   
    DLF_all = Data_processing.DLF_for_summary(DLF, DNSP_selection , DNSP_connection,cogen_connection)
    DNSP_all = Data_processing.DNSP_for_summary(DNSP,DNSP_tariff,DNSP_selection)

    if BTM:
        curtail_all = Data_processing.curtail_for_summary(curtail,scn_mgr['Connection size'].iloc[0], scn_mgr['Curtailment'].iloc[0],front_meter)
    else:
        curtail_all = []
    
    start_date = pd.to_datetime(scn_mgr['start_date'].iloc[0],format = '%d/%m/%Y')

    ppa_sim = scn_mgr['PPA load'].iloc[0] == 'Yes'
    ppa_percent = scn_mgr['PPA load curve'].iloc[0] == 'None'
    if ppa_sim:
        PPA_price= Data_processing.PPA_price_update_yearly(price_scn,selected_jurisdiction, scn_mgr ,start_date)
        if not ppa_percent:
            file_path   = os.path.join(inputs_path, scn_mgr['PPA load curve'].iloc[0])
            ppa_load    = DI.read_ppa_load(file_path)

    else:
        PPA_price = None

    file_path   = os.path.join(out_dir, s, '.csv') 
    df_final = pd.read_csv(file_path ,header=[0,1],index_col=0)
    df_final[('Datetime','Unnamed: 1_level_1')] = pd.to_datetime(df_final[('Datetime','Unnamed: 1_level_1')],format='%d/%m/%Y %H:%M')
    df_final.rename(columns={'Unnamed: 1_level_1':'' , 'Unnamed: 2_level_1':''},inplace=True)

    print('Making summary file ...')
    Post_processing.make_summary(df_final,s,scn_mgr,BTM,curtail_all,aemo_charges, MLF,DLF_all, DNSP_all , Carb_price, selected_jurisdiction.iloc[0] ,param['resolution'],sum_dir,ppa_sim,PPA_price)





# %%
