__author__ = "Oliver Skelding"
__copyright__ = "Wartsila"

# modules required
#py -m pip install pandas, pyyaml, numpy, openpyxl, xlwings, pulp
#%%  
import Data_import
import code_export
import Data_processing
import Post_processing
import LP_Model
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import time
from pathlib import Path
import importlib
import datetime as dt
from datetime import date, timedelta
import os
import logging
import warnings

warnings.filterwarnings("ignore")
from itertools import islice

# ============================================================================
""" Getting the start time of the model, used to name output folder
"""
pd.options.mode.chained_assignment = None
time_run_start = dt.datetime.now()
time_run = time_run_start.strftime("%Y-%m-%d %H-%M")
model_start_time = time.time()
calc_start = time.time()

# ============================================================================ 
""" Define current path 'files_path', 'Input' folder containing all input CSV
    files, Location for potential output (Not Active), and 'yaml_path' as the 
    file includes modelling configuration
"""
files_path = Path().absolute().parents[0]
inputs_path = os.path.join(files_path, 'Inputs')

# inputs_path = str(files_path)+'\\Inputs'
G_drive = 'G:\\Work\\Australia\\Consulting\\Projects\\'
yaml_path = 'model_config.yaml'

# ============================================================================ 
""" The function to import all input CSV files (i.e., Data_import.py)
"""
DI = Data_import.DataImport(inputs_path, G_drive, str(time_run))
data = DI.read_data_main()

# ========================================================================== 
""" Reading model config file
"""

param, model_config = DI.read_model_config(yaml_path)

# ========================================================================== 
""" Getting start time of the model, used to name output folder
"""
scenario_manager = data['scn_mgr']
Scenario_Numbers = data['Scenario_Numbers']

# ==========================================================================
""" Saving on G drive/Locally
"""
G_drive_save = scenario_manager['G drive save'].iloc[0] == 'Yes'
if G_drive_save:
    # copy code to G drive
    G_drive = G_drive + scenario_manager['G drive directory'].iloc[0] + str(time_run)

    code_export.copy_code(os.getcwd(), G_drive)
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

logging.basicConfig(filename=out_dir + '/BESS_modl.log', level=logging.ERROR,
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)

# ========================================================================== 
""" Running optimisation algorithm for each scenario in 'scenario_manager' 
"""
obj_list = []
duration = []

if Scenario_Numbers == 'All':
    Scenarios_UniqueNumber = (scenario_manager['Unique Scenario Number'])
else:
    Scenarios_UniqueNumber = [int(item) if item.isdigit() else item for item in Scenario_Numbers.split(',')]
    # Scenarios_UniqueNumber = scenario_manager['Unique Scenario Number'][Scenarios_UniqueNumber]

Scenarios_List = scenario_manager[(scenario_manager['Unique Scenario Number'].isin(Scenarios_UniqueNumber))][
    'Scenario'].sort_index(ascending=False)

if len(Scenarios_List) != len(Scenarios_UniqueNumber):
    print('')
    print('')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Algorithm execution stopped!')
    print('At least one of selected scenarios are not available in Scenario Manager file')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('')
    exit()

for s in Scenarios_List:
    print('==========================================================================================================')
    print('Scenario is: ' + s)
    print('==========================================================================================================')

    time_run_start = dt.datetime.now()
    scn_mgr = scenario_manager.loc[scenario_manager.Scenario == s]
    selected_jurisdiction = scn_mgr['Location']
    energy_price_scenario = scn_mgr['Energy price forecast']
    fcas_price_scenario = scn_mgr['FCAS price forecast']
    DNSP_selection = scn_mgr['DNSP']
    DNSP_connection = scn_mgr['DLF']
    DNSP_tariff = scn_mgr['DNSP_tariff']
    cogen_connection = scn_mgr['DLF co-located generation']
    min_avg_spread = scn_mgr['min_avg_spread'].iloc[0]
    max_avg_spread = scn_mgr['max_avg_spread'].iloc[0]
    rreg_min_avg_spread = scn_mgr['rreg_min_avg_spread'].iloc[0]
    rreg_max_avg_spread = scn_mgr['rreg_max_avg_spread'].iloc[0]
    Price_CV_Input: object = scn_mgr['Price CSV File'].iloc[0]
    CarbonPricing_CV_Input = scn_mgr['Carbon pricing CSV File'].iloc[0]
    MLF_CV_Input = scn_mgr['Marginal loss factor (MLF) CSV file'].iloc[0]
    DLF_CV_Input = scn_mgr['Distribution loss factor (DLF) CSV file'].iloc[0]
    DNSP_CV_Input = scn_mgr['DNSP tarrif CSV file'].iloc[0]
    Demand_CV_Input = scn_mgr['Demand CSV file'].iloc[0]
    CauserPaysFcas_CV_Input = scn_mgr['Causer pays for FCAS CSV file'].iloc[0]
    Curtailment_CV_Input = scn_mgr['Curtailment profile CSV file'].iloc[0]
    Residential_Load__CV_Input = scn_mgr['Residential Load profile CSV file'].iloc[0]
    AEMO_charges_CV_Input = scn_mgr['AEMO and Market Fees and Charges CSV file'].iloc[0]
    PPA_Load_CSV_Input = scn_mgr['PPA load curve'].iloc[0]
    Tolling_Risk_Factr_Active = scn_mgr['Tolling Risk Factor Activation'].iloc[0]
    BESS_Tolling_Risk_Factor = scn_mgr['BESS Tolling Risk Factor CSV file'].iloc[0]
    Foresight_period = scn_mgr['Foresight Optimisation period'].iloc[0]

    scn_mgr['Run_Period'] = param['RunPeriod']
    scn_mgr['Resolution'] = param['resolution']

    Solve_status_dict = {}
    # file_path   = os.path.join()

    price = Data_import.DataImport.read_prices(inputs_path, Price_CV_Input, scn_mgr)
    Carb_price = Data_import.DataImport.read_carb_price(inputs_path, CarbonPricing_CV_Input)
    MLF, DLF = Data_import.DataImport.read_MLF_DLF(inputs_path, MLF_CV_Input, DLF_CV_Input)
    DNSP = Data_import.DataImport.read_DNSP_tariff(inputs_path, DNSP_CV_Input, scn_mgr)
    DNSP.replace(np.nan, 'None', inplace=True)
    demand = Data_import.DataImport.read_demand(inputs_path, Demand_CV_Input)
    CPF = Data_import.DataImport.read_CPF(inputs_path, CauserPaysFcas_CV_Input)
    curtail = Data_import.DataImport.read_curtailment(inputs_path, Curtailment_CV_Input, scn_mgr)
    residential = Data_import.DataImport.read_residential(inputs_path, Residential_Load__CV_Input, scn_mgr)
    aemo_charges = Data_import.DataImport.read_aemo_fees(inputs_path, AEMO_charges_CV_Input)
    #Toll_Risk_Factr = Data_import.DataImport.read_BESS_Tolling (inputs_path, BESS_Tolling_Risk_Factor    , scn_mgr)

    final_output = {}
    post_proc_dict = {'df': pd.DataFrame(), 'num_cycles': 0, 'end_of_loop_soc': param['SOC_init']}

    #  ----- Behind meter ------------------------------

    front_meter = scn_mgr['Behind_meter'].iloc[0] == 'No'
    BTM = not front_meter
    DLF_all = Data_processing.DLF_for_summary(DLF, DNSP_selection, DNSP_connection, cogen_connection)
    DNSP_all = Data_processing.DNSP_for_summary(DNSP, DNSP_tariff, DNSP_selection)

    No_Residential_load = scn_mgr['Residential Load'].iloc[0] == 'No'
    Res_Load = not No_Residential_load

    Tariff_constant = 1
    scn_mgr['Connection size'].iloc[0] = scn_mgr['Pwr_cap'].iloc[0]

    if BTM:
        residential_all = Data_processing.residential_for_summary(residential, scn_mgr['Connection size'].iloc[0],
                                                                  scn_mgr['Curtailment'].iloc[0], front_meter)
        if Res_Load:
            BESS_size = scn_mgr['Pwr_cap'].iloc[0]
            Max_load_value = residential['Load_MW'].max()
            if BESS_size > Max_load_value:

                Connection_size_required = BESS_size
            else:
                Connection_size_required = Max_load_value

            scn_mgr['Connection size'].iloc[0] = scn_mgr['Pwr_cap'].iloc[0]
            residential_all = Data_processing.residential_for_summary(residential, scn_mgr['Connection size'].iloc[0],
                                                                      scn_mgr['Curtailment'].iloc[0],
                                                                      front_meter)  ## curtailment is a yes/no in SM so adding it in here will just let our load profile not deal with the curtailment cols
            # read central or tariff, if tariff, t_c = 0, else =1 
            Tariff = scn_mgr['Energy price forecast'].iloc[0] == 'Tariff'
            Wholesale_price = not Tariff
            if Wholesale_price:
                Tariff_constant = 1  #1 for wholesale 0 for tariff
            else:
                Tariff_constant = 0

        curtail_all = Data_processing.curtail_for_summary(curtail, scn_mgr['Connection size'].iloc[0],
                                                          scn_mgr['Curtailment'].iloc[0], front_meter)
        #in future move csv that are only relevenat to Btm to be imported to here so the Ftm can run without a dummy file being added
        #e.g.     Curtailment_CV_Input        = scn_mgr['Curtailment profile CSV file'             ].iloc[0]




    else:
        curtail_all = []
        residential_all = []
    #  ----- Degradation ------------------------------
    files_path = os.path.join(inputs_path, scn_mgr['Degradation'].iloc[0])
    degradation_df = DI.read_degradation(files_path)

    #  ----- Charge & discharge ratio ------------------    
    charge_ratio = scn_mgr['charge_ratio'].iloc[0]
    discharge_ratio = scn_mgr['discharge_ratio'].iloc[0]
    max_grid_charge = scn_mgr['Max_grid_charge'].iloc[0]

    # ----- Check if min_avg_spread is file name ------
    if not isinstance(min_avg_spread, (int, np.integer)):
        if min_avg_spread.isdigit():
            min_avg_spread = float(min_avg_spread)
            min_avg_spread_curve = None
        else:
            files_path = os.path.join(inputs_path, min_avg_spread)
            min_avg_spread_curve = DI.read_min_avg_spread(files_path)
    else:
        min_avg_spread_curve = None

    #  ---- Check if charging ratio is a file name -----    
    if not (isinstance(charge_ratio, (float, float)) | isinstance(charge_ratio, (
    int, np.int64))):  #check if charging ratio is file name
        if charge_ratio.replace(".", "", 1).isdigit():
            charge_ratio = float(charge_ratio)
            charge_ratio_curve = None
        else:
            files_path = os.path.join(inputs_path, charge_ratio)
            charge_ratio_curve = DI.read_discharge_charge_ratio(files_path, flow='Charge')
    else:
        charge_ratio_curve = None

    if not (isinstance(discharge_ratio, (float, float)) | isinstance(discharge_ratio, (int, np.int64))):
        if discharge_ratio.replace(".", "", 1).isdigit():
            discharge_ratio = float(discharge_ratio)
            discharge_ratio_curve = None
        else:
            files_path = os.path.join(inputs_path, discharge_ratio)
            discharge_ratio_curve = DI.read_discharge_charge_ratio(files_path, flow='Discharge')
    else:
        discharge_ratio_curve = None

    #  --- Max cycle curve ------------------------------
    max_cycle = scn_mgr['Max_cyc_per_day'].iloc[0]
    cycle_curve = False

    #  --- Check if cycle is a file name ----------------    
    if not isinstance(max_cycle, (int, np.integer)):
        if max_cycle.isdigit():
            scn_mgr['Max_cyc_per_day'] = float(max_cycle)
        else:
            files_path = os.path.join(inputs_path, max_cycle)
            max_cycle_curve = DI.read_max_cycle(files_path)
            cycle_curve = True

    # ----- Get the RTE curve ----------------------------
    if scn_mgr['Battery RTE curve'].iloc[0] == 'Yes':
        files_path = os.path.join(inputs_path, scn_mgr['RTE curve dir'].iloc[0])
        RTE_df = DI.read_RTE_curve(files_path)

    price_scn = Data_processing.select_price_scenario(price, selected_jurisdiction, energy_price_scenario,
                                                      fcas_price_scenario)
    aemo_scn = Data_processing.select_aemo_fees_scenario(aemo_charges, selected_jurisdiction)

    start_date = pd.to_datetime(scn_mgr['start_date'].iloc[0], format='%d/%m/%Y')
    end_date = pd.to_datetime(scn_mgr['end_date'].iloc[0], format='%d/%m/%Y')
    dates = pd.date_range(start_date, end_date, freq='d')
    dates = dates.date

    ppa_sim = scn_mgr['PPA load'].iloc[0] == 'Yes'
    ppa_percent = scn_mgr['PPA load curve'].iloc[0] == 'None'
    if ppa_sim:
        PPA_price = Data_processing.PPA_price_update_yearly(price_scn, selected_jurisdiction, scn_mgr, start_date)
        if not ppa_percent:
            ppa_load = Data_import.DataImport.read_ppa_load(inputs_path, PPA_Load_CSV_Input, scn_mgr)

        else:
            ppa_start_date = pd.to_datetime(scn_mgr['PPA start date'].iloc[0], format='%d/%m/%Y')
            ppa_end_date = pd.to_datetime(scn_mgr['PPA end date'].iloc[0], format='%d/%m/%Y')
    else:
        PPA_price = None

    # dates = list(price_scn.Datetime.dt.date.unique())
    running_30days_cost = []
    final_output = {}
    year_change = True

    # ----- Pre-processing for each day ---------------------
    Foresight_hrs = (Foresight_period) * param['resolution'] / 60
    NumDay_Foresight = int(np.ceil(Foresight_hrs / 24))
    NumDays = len(dates) - NumDay_Foresight + 1
    dates = dates[:NumDays]

    for ind, d in enumerate(dates):
        the_date = d
        print(d)
        year = d.year
        curtail_scn = Data_processing.get_coloc_gen_curtail(NumDay_Foresight, Foresight_period, curtail, d,
                                                            scn_mgr['Connection size'].iloc[0],
                                                            scn_mgr['Curtailment'].iloc[0], front_meter)
        residential_load_scn = Data_processing.get_res_load_curtail(NumDay_Foresight, Foresight_period, residential, d,
                                                                    scn_mgr['Connection size'].iloc[0],
                                                                    scn_mgr['Curtailment'].iloc[0], front_meter)
        price_daily = Data_processing.get_daily_price(Foresight_period, price_scn, d)
        DNSP_scn, peak_demand_months, peak_demand_periods = Data_processing.set_up_DNSP_tariff(Foresight_period, DNSP,
                                                                                               d, DNSP_tariff,
                                                                                               DNSP_selection,
                                                                                               param['resolution'])
        spread_today = Data_processing.avg_spread_today(price_daily, selected_jurisdiction.iloc[0])
        if year_change:
            MLF_scn = Data_processing.select_MLF(MLF, year)
            DLF_year, DLF_cogen = Data_processing.select_DLF(DLF, year, DNSP_selection, DNSP_connection,
                                                             cogen_connection)
            LGC_STC_Scn = Data_processing.select_LGC_price(Carb_price, year)
        if ppa_sim:
            #PPA price diff
            PPA_Price_Diff = Data_processing.PPA_Price_Diff(price_daily, selected_jurisdiction, year, PPA_price)
            if not ppa_percent:
                PPA_load_daily = Data_processing.PPA_load_daily(Foresight_period, ppa_load, d)
            else:
                PPA_contract = Data_processing.PPA_percent_contract(ppa_start_date, ppa_end_date, d)

                # BESS_charging_ratio = Data_processing.dynamic_charging_NSP(Foresight_period, d , scn_mgr['charge_ratio'].iloc[0],scn_mgr['NSP peak charge ratio'].iloc[0] ,peak_demand_months ,periods = peak_demand_periods,daylight_shift=scn_mgr['Daylight shift'].iloc[0])
        if not min_avg_spread_curve is None:
            min_avg_spread = Data_processing.select_min_spread(min_avg_spread_curve, d)

        if ind == 0:
            # operation for first day 
            num_cycles = 0
            end_of_loop_soc = param['SOC_init'] * scn_mgr['Enrgy_init_cap']
            daily_spread, rreg_daily_spread, running_30days_cost = Post_processing.daily_spread_calc(pd.DataFrame(),
                                                                                                     spread_today,
                                                                                                     running_30days_cost,
                                                                                                     min_avg_spread,
                                                                                                     max_avg_spread,
                                                                                                     rreg_min_avg_spread,
                                                                                                     rreg_max_avg_spread)
            daily_spread = daily_spread + np.mean(price_daily[selected_jurisdiction].iloc[0])
            rreg_daily_spread = rreg_daily_spread + np.mean(price_daily[selected_jurisdiction].iloc[0])
        else:
            num_cycles = post_proc_dict['num_cycles']
            end_of_loop_soc = post_proc_dict['end_of_loop_soc']
            daily_spread, rreg_daily_spread, running_30days_cost = Post_processing.daily_spread_calc(
                post_proc_dict['df'], spread_today, running_30days_cost, min_avg_spread, max_avg_spread,
                rreg_min_avg_spread, rreg_max_avg_spread)

        battery_current_cap = Data_processing.degradation_battery_energy(degradation_df, num_cycles,
                                                                         scn_mgr['Enrgy_init_cap'])

        # dynamic charge/discharge based on SOH/cycles
        BESS_charging_ratio = Data_processing.dynamic_charging(Foresight_period, charge_ratio_curve, num_cycles, d,
                                                               charge_ratio, scn_mgr['NSP peak charge ratio'].iloc[0],
                                                               peak_demand_months, periods=peak_demand_periods,
                                                               daylight_shift=scn_mgr['Daylight shift'].iloc[0])
        BESS_discharging_ratio = Data_processing.dynamic_discharging(Foresight_period, discharge_ratio_curve,
                                                                     num_cycles, discharge_ratio)

        #battery replacement

        if battery_current_cap < (scn_mgr['Battery Replacement Threshold'] * scn_mgr['Enrgy_init_cap']).values:
            battery_current_cap = scn_mgr['Enrgy_init_cap'].iloc[0]
            num_cycles = 0

        # RTE curve
        if scn_mgr['Battery RTE curve'].iloc[0] == 'Yes':
            RTE = Data_processing.RTE_battery_curve(RTE_df, num_cycles)
            scn_mgr['Chrg_efncy'] = np.sqrt(RTE)
            scn_mgr['Disch_efncy'] = scn_mgr['Chrg_efncy']

        #Max cycle per day
        if (cycle_curve) & (year_change):
            max_cycle_today = Data_processing.get_max_cycle(max_cycle_curve, d)
            scn_mgr['Max_cyc_per_day'] = max_cycle_today.iloc[0]

        input = {'Price': price_daily,
                 'Scn_mgr': scn_mgr,
                 'DNSP_tariff': DNSP_scn,
                 'LGC_STC': LGC_STC_Scn,
                 'MLF': MLF_scn,
                 'DLF': DLF_year,
                 'Curtailment': curtail_scn,
                 'Residential_load': residential_load_scn,
                 'DLF_cogen': DLF_cogen,
                 'current_capacity': battery_current_cap,

                 'end_of_loop_soc': end_of_loop_soc,
                 'daily_spread': daily_spread,
                 'rreg_daily_spread': rreg_daily_spread,
                 'Max_grid_charge': max_grid_charge,
                 'charge_ratio': BESS_charging_ratio,
                 'discharge_ratio': BESS_discharging_ratio,
                 'BTM': BTM,
                 'Res_Load': Res_Load,
                 'Tariff_constant': Tariff_constant}

        if ppa_sim:
            input['price_diff'] = PPA_Price_Diff
            input['PPA_price'] = PPA_price.loc[PPA_price.Year == d.year]
            if not ppa_percent:  #pass load profile to model
                input['PPA load offtake'] = PPA_load_daily
            else:
                input['PPA contract'] = PPA_contract

        if ind == 0:
            prob = LP_Model.BESS_Model(input, param, model_config)

        else:
            prob.input_data = input
            prob.update_constraints()
            prob.set_objective()

        # ----- Solve the model ---------------------------------
        prob.solve_model()
        Solve_status_dict.update({the_date: prob.solve_status()})

        mdl_out = prob.model_output()

        # ----- Send the model result for post-processing---------

        post_proc_dict = Post_processing.post_process_battery_results(mdl_out, num_cycles, input, battery_current_cap,
                                                                      d, param['resolution'])

        input['Price'] = input['Price'].head(int(24 / (param['resolution'] / 60)))
        input['DNSP_tariff'] = input['DNSP_tariff'].head(int(24 / (param['resolution'] / 60)))
        input['Curtailment'] = input['Curtailment'].head(int(24 / (param['resolution'] / 60)))
        input['Residential_load'] = input['Residential_load'].head(int(24 / (param['resolution'] / 60)))
        input['charge_ratio'] = {k: input['charge_ratio'][k] for k in
                                 list(input['charge_ratio'])[:(int(24 / (param['resolution'] / 60)))]}
        input['discharge_ratio'] = {k: input['discharge_ratio'][k] for k in
                                    list(input['discharge_ratio'])[:(int(24 / (param['resolution'] / 60)))]}
        if ppa_sim:
            input['price_diff'] = input['price_diff'].head(int(24 / (param['resolution'] / 60)))
        post_proc_dict['df'] = Post_processing.revenue_calc_postproc(post_proc_dict['df'], input, param['resolution'])

        #check year change to update MLF DLF and LGC price in next step
        d_1 = d + relativedelta(days=1)  #to check year change
        if d_1.year == d.year:
            year_change = False
        else:
            year_change = True

        final_output[str(d.strftime(format='%Y-%m-%d'))] = post_proc_dict['df']

        # obj_list.append(prob.model.objective.value())
        # ds_list.append(daily_spread)

    time_run_End = dt.datetime.now()
    duration.append(time_run_End - time_run_start)

    file_path = os.path.join(out_dir, '{}.csv'.format(s))
    out_file = file_path

    df_final = Post_processing.combine_dfs_dict(final_output, param['resolution'])
    df_final = Post_processing.add_price_to_df_final(df_final, price_scn)
    if BTM:
        df_final = Post_processing.merge_curtail(df_final, curtail_all)
        if Res_Load:
            df_final = Post_processing.merge_residential(df_final, residential_all)
    # ----- Resetting the time interval shifting before sending out ---------------------
    ColumnsToShiftBack = df_final.columns[2:]
    Shifting_Intervals = scn_mgr['Dispatch interval (DI) shifting'].iloc[0]

    df_final.loc[0:, ColumnsToShiftBack] = df_final.loc[0:, ColumnsToShiftBack].shift(+Shifting_Intervals, fill_value=0)

    df_final.to_csv(out_file)
    print('')
    print("   ***: Run time is: {}".format(duration))
    # print('')

    print('   ***: Making summary file ...')
    # print('')

    Post_processing.make_summary(df_final, s, scn_mgr, BTM, curtail_all, residential_all, Res_Load, aemo_charges, MLF,
                                 DLF_all, DNSP_all, Carb_price, selected_jurisdiction.iloc[0], param['resolution'],
                                 sum_dir, ppa_sim, PPA_price, Solve_status_dict)
    print('   ***: Summary file for ' + s + ' created !!!')
    print('')

print('==========================================================================================================')
print('THIS IS THE END OF ALGORITHM EXECUTION FOR ALL SELECTED SCENARIOS')
print('==========================================================================================================')
print('')
print('')

# %%
