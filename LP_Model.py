"""
Created on 25/10/2021
"""
__author__      = "Mostafa Naemi"
__copyright__   = "Cornwall Insight Australia"


import  pandas    as pd
import  pulp
import  logging
import  numpy     as np
from    pathlib   import Path
from    sympy     import DiagonalMatrix


def add_constraint(model, constraint,name=None):
    """add the constraint with the name to the model"""
    model.addConstraint(constraint,name)
    return constraint

class BESS_Model():
    def __init__(self, input_data, input_params, model_config):
        self.input_data     = input_data
        self.input_params   = input_params
        self.model_config   = model_config
        self.model          = pulp.LpProblem("Battery_Profit_Max_ongoing", pulp.LpMaximize)
        self.contingency_raise_required_throughput = [0.00083,0.008333333, 0.040833333, 0.116666667]
        self.contingency_lower_required_throughput = [0.00083,0.008333333, 0.040833333, 0.116666667]
        # self.contingency_raise_required_throughput = [0,0.04583, 0.05, 0.0625]
        # self.contingency_lower_required_throughput = [0,0.04583, 0.05, 0.0625]
        
        self.Run_Period     = input_params['RunPeriod' ]
        self.resolution     = input_params['resolution']/60
        self.soc_start      = input_params['SOC_init'  ]
        Foresight_period    = input_data  ['Scn_mgr'   ]['Foresight Optimisation period'].iloc[0]
        daily_periods       = range(1,int(Foresight_period)+1)
        self.ppa_sim        = input_data['Scn_mgr']['PPA load'].iloc[0] == 'Yes'
        self.eps            = 0.01
        self.create_variables  (daily_periods)
        self.create_constraints(daily_periods)
        if self.ppa_sim:
            self.add_ppa_model(daily_periods)
        self.set_objective()

    # ============================================================================  
    """ This function creates decision vars in LP model 
    """
    def create_variables(self,daily_periods):
        battery_capacity        = self.input_data['Scn_mgr']['Pwr_cap'        ].iloc[0] # from scenario manager df
        battery_energy_current  = self.input_data['Scn_mgr']['Enrgy_init_cap' ].iloc[0]
        max_grid_charge       = self.input_data['Max_grid_charge'] if pd.notnull(self.input_data['Max_grid_charge']) else battery_capacity # if no value provided then assume can charge full battery capacity from grid
        max_grid_charge = max_grid_charge if max_grid_charge != 'None' else battery_capacity
        # charge_ratio = self.input_data['Scn_mgr']['charge_ratio'].iloc[0]
        discharge_ratio         = self.input_data['Scn_mgr']['discharge_ratio'].iloc[0]
        # self.battery_status = pulp.LpVariable.dicts('battery_status',[(period,state) for period in daily_periods
        #                                                                 for state in battery_state],
        #                                                                 cat=pulp.LpBinary)
        connection_size         = self.input_data['Scn_mgr']['Connection size'].iloc[0]     

        self.energy_volumes_discharge                   = pulp.LpVariable.dicts('energy_volumes_discharge',               [period for period in daily_periods],
                                                            0, battery_capacity,       cat=pulp.LpContinuous)
        
        self.energy_volumes_charge                      = pulp.LpVariable.dicts('energy_volumes_charge',                  [period for period in daily_periods],
                                                            -max_grid_charge , 0,     cat=pulp.LpContinuous)
        
        self.FCAS_rreg_volumes_discharge                = pulp.LpVariable.dicts('FCAS_rreg_volumes_discharge',            [period for period in daily_periods],
                                                            0, battery_capacity,       cat=pulp.LpContinuous)
  
        self.FCAS_r1sec_volumes_discharge               = pulp.LpVariable.dicts('FCAS_r1sec_volumes_discharge',           [period for period in daily_periods],
                                                            0, battery_capacity,       cat=pulp.LpContinuous)
                       
        self.FCAS_r6sec_volumes_discharge               = pulp.LpVariable.dicts('FCAS_r6sec_volumes_discharge',           [period for period in daily_periods],
                                                            0, battery_capacity,       cat=pulp.LpContinuous)
        
        self.FCAS_r60sec_volumes_discharge              = pulp.LpVariable.dicts('FCAS_r60sec_volumes_discharge',          [period for period in daily_periods],
                                                            0, battery_capacity,        cat=pulp.LpContinuous)
        
        self.FCAS_r5min_volumes_discharge               = pulp.LpVariable.dicts('FCAS_r5min_volumes_discharge',           [period for period in daily_periods],
                                                            0, battery_capacity,       cat=pulp.LpContinuous)
        
        self.FCAS_lreg_volumes_charge                   = pulp.LpVariable.dicts('FCAS_lreg_volumes_charge',               [period for period in daily_periods],
                                                            -battery_capacity, 0,      cat=pulp.LpContinuous)
        
        self.FCAS_l1sec_volumes_charge                  = pulp.LpVariable.dicts('FCAS_l1sec_volumes_charge',              [period for period in daily_periods],
                                                            -battery_capacity, 0,      cat=pulp.LpContinuous)
        
        self.FCAS_l6sec_volumes_charge                  = pulp.LpVariable.dicts('FCAS_l6sec_volumes_charge',              [period for period in daily_periods],
                                                            -battery_capacity, 0,      cat=pulp.LpContinuous)
        
        self.FCAS_l60sec_volumes_charge                 = pulp.LpVariable.dicts('FCAS_l60sec_volumes_charge',             [period for period in daily_periods],
                                                            -battery_capacity, 0,      cat=pulp.LpContinuous)
        
        self.FCAS_l5min_volumes_charge                  = pulp.LpVariable.dicts('FCAS_l5min_volumes_charge',              [period for period in daily_periods],
                                                            -battery_capacity, 0,      cat=pulp.LpContinuous)
        
        self.soc_begin_period                           = pulp.LpVariable.dicts('soc_begin_period',                       [period for period in daily_periods],
                                                            0, battery_energy_current, cat=pulp.LpContinuous)
        
        self.soc_end_period                             = pulp.LpVariable.dicts('soc_end_period',                         [period for period in daily_periods],
                                                            0, battery_energy_current, cat=pulp.LpContinuous)
        
        self.PPA_load_purchase                          = pulp.LpVariable.dicts('PPA_load_purchase',                      [period for period in daily_periods],
                                                            -10, 0,                    cat=pulp.LpContinuous)
        # ----- behind meter decision variables -----

        self.BTM_energy_vol_charge_from_curtailment     = pulp.LpVariable.dicts('BTM_energy_vol_charge_curtail',          [period for period in daily_periods],
                                                            -battery_capacity, 0,      cat=pulp.LpContinuous) 

        self.BTM_energy_vol_charge_from_export_avail_gen= pulp.LpVariable.dicts('BTM_energy_vol_charge_export_avail_gen', [period for period in daily_periods],
                                                            -battery_capacity, 0,       cat=pulp.LpContinuous) 

        # self.BTM_lreg_vol_charge_from_curtailment = pulp.LpVariable.dicts('BTM_lreg_vol_charge_curtail',[period for period in daily_periods],
        #                                  -battery_capacity, 0, cat=pulp.LpContinuous) 

        self.BTM_lreg_vol_charge_from_export_avail_gen  = pulp.LpVariable.dicts('BTM_lreg_vol_charge_export_avail_gen',   [period for period in daily_periods],
                                                            -battery_capacity, 0,      cat=pulp.LpContinuous)     

        self.BTM_co_loc_gen_energy_export_vol           = pulp.LpVariable.dicts('BTM_co_loc_gen_energy_export',           [period for period in daily_periods],
                                                            0, connection_size,        cat=pulp.LpContinuous) 

##battery can discharge itself (Raise_reg) to handle Residential_load. Solar can discharge to load, and finally the grid will be able to handle the Residential_load
        Res_Load              = self.input_data['Res_Load']
        upper_bound_solar = connection_size if Res_Load else 0
        upper_bound_battery = battery_capacity if Res_Load else 0
        lower_bound_grid = -connection_size if Res_Load else 0
        
        self.BTM_Solar_to_load= pulp.LpVariable.dicts('BTM_solar_to_load', [period for period in daily_periods],
                                                              0,upper_bound_solar,       cat=pulp.LpContinuous) 
         
        self.BTM_BESS_to_load  = pulp.LpVariable.dicts('BTM_BESS_to_load',   [period for period in daily_periods],
                                                              0,upper_bound_battery,      cat=pulp.LpContinuous)     

        self.BTM_Grid_to_load           = pulp.LpVariable.dicts('BTM_Grid_to_load',           [period for period in daily_periods],
                                                                lower_bound_grid, 0,        cat=pulp.LpContinuous)
        # Create a binary variable so we arent charging (energy_volumes_charge) from the grid and discharging (BTM_co_loc_gen_energy_export_vol) at the same time 
        self.BTM_export_binary =   pulp.LpVariable.dicts('BTM_export_binary',  [period for period in daily_periods], cat=pulp.LpBinary)   
    # ============================================================================  
    """ This function adds constraints to LP model object 
    """
    def create_constraints(self,daily_periods):
        scn_mgr               = self.input_data['Scn_mgr'           ]
        price                 = self.input_data['Price'             ]
        daily_spread          = self.input_data['daily_spread'      ]
        rreg_daily_spread     = self.input_data['rreg_daily_spread' ]
        curtailment           = self.input_data['Curtailment'       ]
        Residential_load      = self.input_data['Residential_load'   ]
        battery_capacity      = self.input_data['Scn_mgr']['Pwr_cap'].iloc[0] # from scenario manager df
        max_grid_charge       = self.input_data['Max_grid_charge']
        charge_ratio          = self.input_data['charge_ratio'      ]
        discharge_ratio       = self.input_data['discharge_ratio'   ]
        chrg_efficiency       = scn_mgr['Chrg_efncy'       ].iloc[0]
        disch_efficiency      = scn_mgr['Disch_efncy'      ].iloc[0]
        LReg_thrput           = scn_mgr['Svc_throuput_Lreg'].iloc[0]
        Rated_connection_size = self.input_data['Scn_mgr']['Connection size'].iloc[0] 
        M                     = Rated_connection_size + 1
        BTM                   = self.input_data['BTM']
        Res_Load              = self.input_data['Res_Load']
        
        max_grid_charge = max_grid_charge if max_grid_charge != 'None' else battery_capacity
        # cannot cycle more than three times a day
        add_constraint(self.model,pulp.lpSum( 
                [self.energy_volumes_discharge     [period]* self.resolution                                     / disch_efficiency  for period in daily_periods]
              + [self.FCAS_rreg_volumes_discharge  [period]* self.resolution*scn_mgr['Svc_throuput_Rreg'].iloc[0]/ disch_efficiency  for period in daily_periods]
              + [self.FCAS_r1sec_volumes_discharge [period]* self.resolution*scn_mgr['Svc_throuput_R1'  ].iloc[0]/ disch_efficiency  for period in daily_periods]
              + [self.FCAS_r6sec_volumes_discharge [period]* self.resolution*scn_mgr['Svc_throuput_R6'  ].iloc[0]/ disch_efficiency  for period in daily_periods]
              + [self.FCAS_r60sec_volumes_discharge[period]* self.resolution*scn_mgr['Svc_throuput_R60' ].iloc[0]/ disch_efficiency  for period in daily_periods] 
              + [self.FCAS_r5min_volumes_discharge [period]* self.resolution*scn_mgr['Svc_throuput_R5'  ].iloc[0]/ disch_efficiency  for period in daily_periods]
              + [self.BTM_BESS_to_load [period]* self.resolution for period in daily_periods] 
              )<= scn_mgr['Max_cyc_per_day'].iloc[0]*self.input_data['current_capacity'] ,'cycling_constraint')       
        
        for period in daily_periods:
            add_constraint(self.model , self.energy_volumes_discharge                   [period] 
                                      + self.BTM_BESS_to_load                           [period] <=  battery_capacity*discharge_ratio[period],name = str(period)+'_discharge_lowerbound')
            
            add_constraint(self.model , self.FCAS_rreg_volumes_discharge                [period] <=  battery_capacity*discharge_ratio[period],name = str(period)+'_rreg_lowerbound'     )
            add_constraint(self.model , self.FCAS_r1sec_volumes_discharge               [period] <=  battery_capacity*discharge_ratio[period],name = str(period)+'_r1_lowerbound'       )
            add_constraint(self.model , self.FCAS_r6sec_volumes_discharge               [period] <=  battery_capacity*discharge_ratio[period],name = str(period)+'_r6_lowerbound'       )
            add_constraint(self.model , self.FCAS_r60sec_volumes_discharge              [period] <=  battery_capacity*discharge_ratio[period],name = str(period)+'_r60_lowerbound'      )
            add_constraint(self.model , self.FCAS_r5min_volumes_discharge               [period] <=  battery_capacity*discharge_ratio[period],name = str(period)+'_r5_lowerbound'       )
            add_constraint(self.model , self.energy_volumes_charge                      [period] 
                                      + self.BTM_energy_vol_charge_from_curtailment     [period] 
                                      + self.BTM_energy_vol_charge_from_export_avail_gen[period] >= -battery_capacity*  charge_ratio [period],name = str(period)+'_charge_lowerbound'   )
            add_constraint(self.model , self.FCAS_lreg_volumes_charge                   [period] 
                                      + self.BTM_lreg_vol_charge_from_export_avail_gen  [period] >= -battery_capacity*  charge_ratio [period],name = str(period)+'_lreg_lowerbound'     )
            add_constraint(self.model , self.FCAS_l1sec_volumes_charge                  [period] >= -battery_capacity*  charge_ratio [period],name = str(period)+'_l1_lowerbound'       )
            add_constraint(self.model , self.FCAS_l6sec_volumes_charge                  [period] >= -battery_capacity*  charge_ratio [period],name = str(period)+'_l6_lowerbound'       )
            add_constraint(self.model , self.FCAS_l60sec_volumes_charge                 [period] >= -battery_capacity*  charge_ratio [period],name = str(period)+'_l60_lowerbound'      )
            add_constraint(self.model , self.FCAS_l5min_volumes_charge                  [period] >= -battery_capacity*  charge_ratio [period],name = str(period)+'_l5_lowerbound'       )
            add_constraint(self.model , self.energy_volumes_discharge                   [period] <=  scn_mgr['Energy_discharge_market'  ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_rreg_volumes_discharge                [period] <=  scn_mgr['Rreg_market'].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_r1sec_volumes_discharge               [period] <=  scn_mgr['R1_market'  ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_r6sec_volumes_discharge               [period] <=  scn_mgr['R6_market'  ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_r60sec_volumes_discharge              [period] <=  scn_mgr['R60_market' ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_r5min_volumes_discharge               [period] <=  scn_mgr['R5_market'  ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.energy_volumes_charge                      [period] <=  scn_mgr['Energy_charge_market'  ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_lreg_volumes_charge                   [period] 
                                      + self.BTM_lreg_vol_charge_from_export_avail_gen  [period] >= -scn_mgr['Lreg_market'].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_l1sec_volumes_charge                  [period] >= -scn_mgr['L1_market'  ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_l6sec_volumes_charge                  [period] >= -scn_mgr['L6_market'  ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_l60sec_volumes_charge                 [period] >= -scn_mgr['L60_market' ].iloc[0]*battery_capacity)
            add_constraint(self.model , self.FCAS_l5min_volumes_charge                  [period] >= -scn_mgr['L5_market'  ].iloc[0]*battery_capacity)

            if Res_Load:
                ix_load = Residential_load['Period'] == period
                add_constraint(self.model , self.BTM_Solar_to_load  [period]
                                          - self.BTM_Grid_to_load [period] 
                                          + self.BTM_BESS_to_load [period]
                                          ==   Residential_load.loc[ix_load,'net_co_locate_load_avail_export'].iloc[0],name = str(period)+'_residential_load') #Combination of solar/grid/bess must always meet the demand of load and never be below
    
                
            
            FCAS_discharge_contingency_volume_list = [self.FCAS_r1sec_volumes_discharge [period],
                                                      self.FCAS_r6sec_volumes_discharge [period],
                                                      self.FCAS_r60sec_volumes_discharge[period],
                                                      self.FCAS_r5min_volumes_discharge [period]]
            FCAS_charge_contingency_volume_list    = [self.FCAS_l1sec_volumes_charge    [period],
                                                      self.FCAS_l6sec_volumes_charge    [period],
                                                      self.FCAS_l60sec_volumes_charge   [period],
                                                      self.FCAS_l5min_volumes_charge    [period]]
            ix_curt = curtailment['Period'] == period
            for discharge_contingency_mkt, mkt_avail_througput_req in zip(FCAS_discharge_contingency_volume_list, self.contingency_raise_required_throughput):
                # FCAS_discharge: reg and enegy volumes and FCAS contingency must add to available battery capacity and available SOC
                add_constraint(self.model,pulp.lpSum( self.energy_volumes_discharge         [period] 
                                                    + self.FCAS_rreg_volumes_discharge      [period] 
                                                    + discharge_contingency_mkt                    ) 
                                                    <= discharge_ratio[period] * scn_mgr['Pwr_cap'].iloc[0],  str(period) + str(discharge_contingency_mkt) + 'Max_raise_service')

                add_constraint(self.model,pulp.lpSum( self.energy_volumes_discharge         [period]* self.resolution / disch_efficiency 
                                                    + self.FCAS_rreg_volumes_discharge      [period]* self.resolution / disch_efficiency 
                                                    + discharge_contingency_mkt * mkt_avail_througput_req / disch_efficiency           ) 
                                                    <= self.soc_begin_period[period])
                

                add_constraint(self.model,pulp.lpSum( self.energy_volumes_discharge         [period] 
                                                    + self.BTM_co_loc_gen_energy_export_vol [period] 
                                                    + self.FCAS_rreg_volumes_discharge      [period] 
                                                    + discharge_contingency_mkt                    ) 
                                                    <= curtailment.loc[ix_curt,'max_connection'].iloc[0],  str(period)+str(discharge_contingency_mkt) + '_max_connection'  )
 
            # if we have a constraint on what we can charge from grid this should be reflected in contingency lower
            lhs_max_grid_constraint = [
                    self.energy_volumes_discharge[period] +
                    self.BTM_co_loc_gen_energy_export_vol[period] +
                    max_grid_charge +
                    self.energy_volumes_charge[period] +
                    self.FCAS_lreg_volumes_charge[period]
            ]
            
            for charge_contingency_mkt, mkt_avail_througput_req in zip(FCAS_charge_contingency_volume_list, self.contingency_lower_required_throughput):
            # FCAS_charge: reg and enegy volumes and FCAS contingency must add to available battery capacity and available SOC
                add_constraint(self.model,pulp.lpSum( self.energy_volumes_charge                      [period] 
                                                    + self.BTM_energy_vol_charge_from_curtailment     [period] 
                                                    + self.BTM_energy_vol_charge_from_export_avail_gen[period] 
                                                    + self.BTM_lreg_vol_charge_from_export_avail_gen  [period] 
                                                    + self.FCAS_lreg_volumes_charge                   [period] 
                                                    + charge_contingency_mkt                                 ) 
                                                    >= -charge_ratio[period] * scn_mgr['Pwr_cap'].iloc[0],  str(period) + str(charge_contingency_mkt) + 'Max_lower_service')
                add_constraint(self.model,pulp.lpSum((self.energy_volumes_charge                      [period]
                                                    + self.FCAS_lreg_volumes_charge                   [period]
                                                    + self.BTM_energy_vol_charge_from_curtailment     [period]
                                                    + self.BTM_energy_vol_charge_from_export_avail_gen[period]
                                                    + self.BTM_lreg_vol_charge_from_export_avail_gen  [period])* chrg_efficiency* self.resolution 
                                                    + mkt_avail_througput_req* charge_contingency_mkt*chrg_efficiency ) 
                                                    >= -(self.input_data['current_capacity'] - self.soc_begin_period[period]),  str(period)+str(charge_contingency_mkt)+'_energy_const_cumulative_mkt')
                # Curtailment limitation is only considered for export, not for import, so BESS can fully charge at Rated_connection_size of the line. 
                add_constraint(self.model,pulp.lpSum( self.energy_volumes_charge            [period] 
                                                    + self.FCAS_lreg_volumes_charge         [period] 
                                                    + charge_contingency_mkt                       
                                                    + self.BTM_Grid_to_load                 [period]      )
                                                    >= -Rated_connection_size                                                 ,  str(period)+str(charge_contingency_mkt) + '_Rated_connection_size'  )
                add_constraint(self.model, pulp.lpSum(lhs_max_grid_constraint) >= -charge_contingency_mkt,  str(period) + str(charge_contingency_mkt) + 'Max_lower_service_grid_constraint')

             ### add rreg risk parameter
            ix_price = price['PERIODID'] == period
            add_constraint(self.model,( self.resolution
                                      * self.FCAS_rreg_volumes_discharge[period]
                                      * price.loc[ix_price,scn_mgr['Location']].values[0]
                                      * scn_mgr['Svc_throuput_Rreg'].iloc[0] 
                                      * self.input_data['DLF'] 
                                      * self.input_data['MLF']['MLF_generation'].iloc[0]) 
                                      * scn_mgr['rreg_risk_adj'].iloc[0] <= 
                                      (  (  self.resolution 
                                          * self.FCAS_rreg_volumes_discharge[period]
                                          * price.loc[ix_price,'RAISEREGRRP'].values[0]) 
                                        +(  self.resolution 
                                          * self.FCAS_rreg_volumes_discharge[period]
                                          * price.loc[ix_price,scn_mgr['Location']].values[0]
                                          * scn_mgr['Svc_throuput_Rreg'].iloc[0] 
                                          * self.input_data['DLF'] 
                                          * self.input_data['MLF']['MLF_generation'].iloc[0])    )  )

            if period == 1:
                add_constraint(self.model, self.soc_begin_period[1] == self.input_data['end_of_loop_soc'],'first_SOC_level')
            else:
                add_constraint(self.model, self.soc_begin_period[period] == self.soc_end_period[period-1])
            add_constraint(self.model,self.soc_end_period[period] ==  
                     (     self.soc_begin_period                            [period] 
                        - (self.energy_volumes_discharge                    [period]*self.resolution                                     / disch_efficiency ) 
                        - (self.FCAS_rreg_volumes_discharge                 [period]*self.resolution* scn_mgr['Svc_throuput_Rreg'].iloc[0]/ disch_efficiency)
                        - (self.FCAS_r1sec_volumes_discharge                [period]*self.resolution* scn_mgr['Svc_throuput_R1'  ].iloc[0]/ disch_efficiency)
                        - (self.FCAS_r6sec_volumes_discharge                [period]*self.resolution* scn_mgr['Svc_throuput_R6'  ].iloc[0]/ disch_efficiency)
                        - (self.FCAS_r60sec_volumes_discharge               [period]*self.resolution* scn_mgr['Svc_throuput_R60' ].iloc[0]/ disch_efficiency)
                        - (self.FCAS_r5min_volumes_discharge                [period]*self.resolution* scn_mgr['Svc_throuput_R5'  ].iloc[0]/ disch_efficiency) 
                        - (self.energy_volumes_charge                       [period]*self.resolution                                      * chrg_efficiency )
                        - (self.FCAS_lreg_volumes_charge                    [period]*self.resolution* LReg_thrput                         * chrg_efficiency )
                        - (self.FCAS_l1sec_volumes_charge                   [period]*self.resolution* scn_mgr['Svc_throuput_L1' ].iloc[0] * chrg_efficiency )
                        - (self.FCAS_l6sec_volumes_charge                   [period]*self.resolution* scn_mgr['Svc_throuput_L6' ].iloc[0] * chrg_efficiency )
                        - (self.FCAS_l60sec_volumes_charge                  [period]*self.resolution* scn_mgr['Svc_throuput_L60'].iloc[0] * chrg_efficiency )
                        - (self.FCAS_l5min_volumes_charge                   [period]*self.resolution* scn_mgr['Svc_throuput_L5' ].iloc[0] * chrg_efficiency )
                        - (self.BTM_energy_vol_charge_from_curtailment      [period]*self.resolution                                      * chrg_efficiency )
                        - (self.BTM_energy_vol_charge_from_export_avail_gen [period]*self.resolution                                      * chrg_efficiency )
                        - (self.BTM_lreg_vol_charge_from_export_avail_gen   [period]*self.resolution* LReg_thrput                         * chrg_efficiency )
                        - (self.BTM_BESS_to_load                            [period]*self.resolution                                                        )
                     )  ,'battery_soc_level_'+str(period))

            add_constraint(self.model,self.soc_end_period                              [period] <= scn_mgr['SOC_max'].iloc[0] * self.input_data['current_capacity'] ,  str(period)+'_Max_SOC'                )   
            add_constraint(self.model,self.soc_end_period                              [period] >= scn_mgr['SOC_min'].iloc[0] * self.input_data['current_capacity'] ,  str(period)+'_Min_SOC'                )


            add_constraint(self.model,self.BTM_co_loc_gen_energy_export_vol[period]  + self.BTM_Solar_to_load[period]   <=  self.BTM_energy_vol_charge_from_export_avail_gen[period]
                                                                                       +curtailment.loc[ix_curt,'net_co_locate_gen_avail_export'].iloc[0] , str(period)+'_max_co_loc_export'    )

            add_constraint(self.model,self.BTM_energy_vol_charge_from_curtailment      [period] >= -curtailment.loc[ix_curt,'total_curt_avail'              ].iloc[0], str(period)+'_max_curtail_charge'     )   
            add_constraint(self.model,self.BTM_energy_vol_charge_from_export_avail_gen [period] >= -curtailment.loc[ix_curt,'net_co_locate_gen_avail_export'].iloc[0], str(period)+'_max_export_avail_charge')
            add_constraint(self.model,self.BTM_lreg_vol_charge_from_export_avail_gen   [period] >= -self.BTM_co_loc_gen_energy_export_vol[period]                    , str(period)+'_max_Lreg_avail_export_charge')


            
            # Cant export BtM if we are charging at the same time, requires BigM constraints
            #BigM constraint (if else statement within daily period)
            #may have noticeable impact on performance so will be applied only when needed 
            if BTM:
                add_constraint(self.model,self.BTM_co_loc_gen_energy_export_vol           [period]* self.resolution  <=   M*1-self.BTM_export_binary[period]       , str(period)+'_BigM_1') 
                add_constraint(self.model,self.energy_volumes_charge                      [period]* self.resolution  >=  -M*(1-self.BTM_export_binary[period])    , str(period)+'_BigM_2')            
                if Res_Load:
                    add_constraint(self.model,self.BTM_Grid_to_load                           [period]* self.resolution  >=  -M*(1-self.BTM_export_binary[period])    , str(period)+'__BigM_3')

        ###### DAILY MIN SPREAD ########
            add_constraint(self.model,self.energy_volumes_discharge                    [period]
                                    * self.resolution
                                    * price.loc[ix_price,scn_mgr['Location']].iloc[0,0]         >=  self.energy_volumes_discharge[period]
                                                                                                  * self.resolution
                                                                                                  * daily_spread                                                     , str(period)+'_min_avg_spread'         )
            add_constraint(self.model,self.FCAS_rreg_volumes_discharge[period]
                                    * self.resolution 
                                    * (scn_mgr['Svc_throuput_Rreg'].iloc[0] 
                                    * price.loc[ix_price,scn_mgr['Location']].iloc[0,0] 
                                    + price.iloc[period-1]['RAISEREGRRP'])                       >=  self.FCAS_rreg_volumes_discharge[period] 
                                                                                                   * self.resolution 
                                                                                                   * scn_mgr['Svc_throuput_Rreg'].iloc[0] 
                                                                                                   * rreg_daily_spread                                               , str(period)+'_RREG_min_avg_spread'   ) 
        
        ### AC side limitation###
            # for charge_contingency_mkt, mkt_avail_througput_req in zip(FCAS_charge_contingency_volume_list, self.contingency_lower_required_throughput):
        
            # # reg and enegy volumes and FCAS contingency must add to available battery capacity and available SOC
            #     add_constraint(self.model,pulp.lpSum(self.energy_volumes_charge[period] + self.FCAS_lreg_volumes_charge[period] + charge_contingency_mkt) >= -2.5)

    # ============================================================================  
    """ This function updates the constraints in the LP model object
        for in each loop. adding this function enables us to prevent creating 
        a new LP model object in each loop and makes simulations faster 
    """

    def update_constraints(self):
        Foresight_period        = self.input_data['Scn_mgr'             ]['Foresight Optimisation period'].iloc[0]
        daily_periods           = range(1,int(Foresight_period)+1)
        scn_mgr                 = self.input_data['Scn_mgr'             ]
        price                   = self.input_data['Price'               ]
        daily_spread            = self.input_data['daily_spread'        ]
        rreg_daily_spread       = self.input_data['rreg_daily_spread'   ]
        curtailment             = self.input_data['Curtailment'         ]
        Residential_load        = self.input_data['Residential_load'    ]
        battery_energy_current  = self.input_data['Scn_mgr'             ]['Enrgy_init_cap'].iloc[0]
        battery_capacity        = self.input_data['Scn_mgr'             ]['Pwr_cap'       ].iloc[0] # from scenario manager df
        max_grid_charge         = self.input_data['Max_grid_charge']
        charge_ratio            = self.input_data['charge_ratio'        ]
        discharge_ratio         = self.input_data['discharge_ratio'     ]
        LReg_thrput             = scn_mgr['Svc_throuput_Lreg'].iloc[0]
        chrg_efficiency         = scn_mgr['Chrg_efncy'       ].iloc[0] 
        disch_efficiency        = scn_mgr['Disch_efncy'      ].iloc[0]         
        location                = scn_mgr['Location'         ].iloc[0]
        Rated_connection_size   = self.input_data['Scn_mgr']['Connection size'].iloc[0] 
        M                       = Rated_connection_size + 1   
        BTM                     = self.input_data['BTM']
        Res_Load                = self.input_data['Res_Load']
        
        max_grid_charge = max_grid_charge if max_grid_charge != 'None' else battery_capacity
        max_grid_charge = max_grid_charge

        self.model.constraints['cycling_constraint'] = pulp.lpSum(
              [self.energy_volumes_discharge      [period]* self.resolution                                       / disch_efficiency  for period in daily_periods] 
            + [self.FCAS_rreg_volumes_discharge   [period]* self.resolution *scn_mgr['Svc_throuput_Rreg'].iloc[0] / disch_efficiency  for period in daily_periods]
            + [self.FCAS_r1sec_volumes_discharge  [period]* self.resolution *scn_mgr['Svc_throuput_R1'  ].iloc[0] / disch_efficiency  for period in daily_periods]
            + [self.FCAS_r6sec_volumes_discharge  [period]* self.resolution *scn_mgr['Svc_throuput_R6'  ].iloc[0] / disch_efficiency  for period in daily_periods]
            + [self.FCAS_r60sec_volumes_discharge [period]* self.resolution *scn_mgr['Svc_throuput_R60' ].iloc[0] / disch_efficiency  for period in daily_periods] 
            + [self.FCAS_r5min_volumes_discharge  [period]* self.resolution *scn_mgr['Svc_throuput_R5'  ].iloc[0] / disch_efficiency  for period in daily_periods]
            + [self.BTM_BESS_to_load [period]* self.resolution for period in daily_periods]
            )<= scn_mgr['Max_cyc_per_day'].iloc[0] * self.input_data['current_capacity'] 

        for period in daily_periods:
            self.model.constraints[str(period)+'_charge_lowerbound'] = self.energy_volumes_charge      [period] >= -battery_capacity*charge_ratio[period]
            self.model.constraints[str(period)+'_lreg_lowerbound'  ] = self.FCAS_lreg_volumes_charge   [period] >= -battery_capacity*charge_ratio[period]
            self.model.constraints[str(period)+'_l1_lowerbound'    ] = self.FCAS_l1sec_volumes_charge  [period] >= -battery_capacity*charge_ratio[period]
            self.model.constraints[str(period)+'_l6_lowerbound'    ] = self.FCAS_l6sec_volumes_charge  [period] >= -battery_capacity*charge_ratio[period]
            self.model.constraints[str(period)+'_l60_lowerbound'   ] = self.FCAS_l60sec_volumes_charge [period] >= -battery_capacity*charge_ratio[period]
            self.model.constraints[str(period)+'_l5_lowerbound'    ] = self.FCAS_l5min_volumes_charge  [period] >= -battery_capacity*charge_ratio[period]          

            FCAS_charge_contingency_volume_list = [self.FCAS_l1sec_volumes_charge [period],
                                                   self.FCAS_l6sec_volumes_charge [period],
                                                   self.FCAS_l60sec_volumes_charge[period],
                                                   self.FCAS_l5min_volumes_charge [period]]
        
            # if we have a constraint on what we can charge from grid this should be reflected in contingency lower
            lhs_max_grid_constraint = [
                    self.energy_volumes_discharge[period] +
                    self.BTM_co_loc_gen_energy_export_vol[period] +
                    max_grid_charge +
                    self.energy_volumes_charge[period] +
                    self.FCAS_lreg_volumes_charge[period]
            ]

            for charge_contingency_mkt, mkt_avail_througput_req in zip(FCAS_charge_contingency_volume_list, self.contingency_lower_required_throughput):
                self.model.constraints[str(period)+str(charge_contingency_mkt)+'_energy_const_cumulative_mkt'] = pulp.lpSum(
                          ( self.energy_volumes_charge                       [period]
                          + self.FCAS_lreg_volumes_charge                    [period]
                          + self.BTM_energy_vol_charge_from_curtailment      [period]
                          + self.BTM_energy_vol_charge_from_export_avail_gen [period]
                          + self.BTM_lreg_vol_charge_from_export_avail_gen   [period] ) * chrg_efficiency* self.resolution 
                          + mkt_avail_througput_req* charge_contingency_mkt*(scn_mgr['Chrg_efncy'].iloc[0])                 ) >= -(   self.input_data['current_capacity'] 
                                                                                                                                    - self.soc_begin_period[period]     ) 
                self.model.constraints[str(period) + str(charge_contingency_mkt) + 'Max_lower_service']        = pulp.lpSum(
                            self.energy_volumes_charge                        [period] 
                          + self.BTM_energy_vol_charge_from_curtailment       [period] 
                          + self.BTM_energy_vol_charge_from_export_avail_gen  [period] 
                          + self.BTM_lreg_vol_charge_from_export_avail_gen    [period] 
                          + self.FCAS_lreg_volumes_charge                     [period] 
                          + charge_contingency_mkt                                                                          ) >= - charge_ratio[period] * scn_mgr['Pwr_cap'].iloc[0]
                
                self.model.constraints[str(period) + str(charge_contingency_mkt) + 'Max_lower_service_grid_constraint']        = pulp.lpSum(lhs_max_grid_constraint) >= -charge_contingency_mkt

            if period == 1:
                self.model.constraints['first_SOC_level'] = self.soc_begin_period[1] == self.input_data['end_of_loop_soc']
            
            self.model.constraints['battery_soc_level_'+str(period)] = self.soc_end_period[period] == (
                            self.soc_begin_period                           [period] 
                        - ( self.energy_volumes_discharge                   [period]*self.resolution/ disch_efficiency) 
                        - ( self.FCAS_rreg_volumes_discharge                [period]*self.resolution* scn_mgr['Svc_throuput_Rreg'].iloc[0]/ disch_efficiency)
                        - ( self.FCAS_r1sec_volumes_discharge               [period]*self.resolution* scn_mgr['Svc_throuput_R1'  ].iloc[0]/ disch_efficiency)
                        - ( self.FCAS_r6sec_volumes_discharge               [period]*self.resolution* scn_mgr['Svc_throuput_R6'  ].iloc[0]/ disch_efficiency)
                        - ( self.FCAS_r60sec_volumes_discharge              [period]*self.resolution* scn_mgr['Svc_throuput_R60' ].iloc[0]/ disch_efficiency)
                        - ( self.FCAS_r5min_volumes_discharge               [period]*self.resolution* scn_mgr['Svc_throuput_R5'  ].iloc[0]/ disch_efficiency) 
                        - ( self.energy_volumes_charge                      [period]*self.resolution                                      * chrg_efficiency )
                        - ( self.FCAS_lreg_volumes_charge                   [period]*self.resolution* LReg_thrput                         * chrg_efficiency )
                        - ( self.FCAS_l1sec_volumes_charge                  [period]*self.resolution* scn_mgr['Svc_throuput_L1'  ].iloc[0]* chrg_efficiency )
                        - ( self.FCAS_l6sec_volumes_charge                  [period]*self.resolution* scn_mgr['Svc_throuput_L6'  ].iloc[0]* chrg_efficiency )
                        - ( self.FCAS_l60sec_volumes_charge                 [period]*self.resolution* scn_mgr['Svc_throuput_L60' ].iloc[0]* chrg_efficiency )
                        - ( self.FCAS_l5min_volumes_charge                  [period]*self.resolution* scn_mgr['Svc_throuput_L5'  ].iloc[0]* chrg_efficiency )
                        - ( self.BTM_energy_vol_charge_from_curtailment     [period]*self.resolution                                      * chrg_efficiency )
                        - ( self.BTM_energy_vol_charge_from_export_avail_gen[period]*self.resolution                                      * chrg_efficiency )
                        - ( self.BTM_lreg_vol_charge_from_export_avail_gen  [period]*self.resolution* LReg_thrput                         * chrg_efficiency )
                        - ( self.BTM_BESS_to_load                           [period]*self.resolution                                                        )
                                                                                                       )


        
            if BTM:
                self.model.constraints[str(period) +'__BigM_1'] = self.BTM_co_loc_gen_energy_export_vol[period]* self.resolution  <=  M*self.BTM_export_binary[period]
        
                self.model.constraints[str(period) +'__BigM_2'] = self.energy_volumes_charge[period]* self.resolution  >=  -M*(1-self.BTM_export_binary[period])
                if Res_Load:
                    self.model.constraints[str(period) +'__BigM_3'] = self.BTM_Grid_to_load [period]* self.resolution  >=  -M*(1-self.BTM_export_binary[period])
                


            self.model.constraints[str(period)+'_min_avg_spread'          ] = ( 
                  self.energy_volumes_discharge[period]
                  * self.resolution
                  * price.iloc[period-1][location]                            ) >= (   self.energy_volumes_discharge[period]
                                                                                     * self.resolution
                                                                                     * daily_spread                          )
            
            self.model.constraints[str(period)+'_RREG_min_avg_spread'     ] = (  
                  self.FCAS_rreg_volumes_discharge[period]
                * self.resolution 
                * (scn_mgr['Svc_throuput_Rreg'].iloc[0] *price.iloc[period-1][location]+ price.iloc[period-1]['RAISEREGRRP'])  )>= (   self.FCAS_rreg_volumes_discharge[period] 
                                                                                                                                    * self.resolution 
                                                                                                                                    * scn_mgr['Svc_throuput_Rreg'].iloc[0] 
                                                                                                                                    * rreg_daily_spread                        )

            self.model.constraints[str(period) +'_max_co_loc_export'      ] = self.BTM_co_loc_gen_energy_export_vol           [period] + self.BTM_Solar_to_load[period]   <=   curtailment.iloc[period-1]['net_co_locate_gen_avail_export'] + self.BTM_energy_vol_charge_from_export_avail_gen[period] 
            self.model.constraints[ str(period)+'_max_curtail_charge'     ] = self.BTM_energy_vol_charge_from_curtailment     [period] >= - curtailment.iloc[period-1]['total_curt_avail'              ]   
            self.model.constraints[ str(period)+'_max_export_avail_charge'] = self.BTM_energy_vol_charge_from_export_avail_gen[period] >= - curtailment.iloc[period-1]['net_co_locate_gen_avail_export']
            self.model.constraints[ str(period)+'_Max_SOC'                ] = self.soc_end_period                             [period] <= scn_mgr['SOC_max'].iloc[0] * self.input_data['current_capacity']
            self.model.constraints[ str(period)+'_Min_SOC'                ] = self.soc_end_period                             [period] >= scn_mgr['SOC_min'].iloc[0] * self.input_data['current_capacity']
            if Res_Load:
                ix_load = Residential_load['Period'] == period
                self.model.constraints[str(period) +'_residential_load'] = self.BTM_Solar_to_load  [period] - self.BTM_Grid_to_load [period] + self.BTM_BESS_to_load [period] == Residential_load.loc[ix_load,'net_co_locate_load_avail_export'].iloc[0] #Combination of solar/grid/bess must always meet the demand of load and never be below

            FCAS_discharge_contingency_volume_list = [self.FCAS_r6sec_volumes_discharge [period],
                                                      self.FCAS_r60sec_volumes_discharge[period],
                                                      self.FCAS_r5min_volumes_discharge [period]]
            for discharge_contingency_mkt, mkt_avail_througput_req in zip(FCAS_discharge_contingency_volume_list, self.contingency_raise_required_throughput):
                self.model.constraints[ str(period)+ str(discharge_contingency_mkt) +'_max_connection'   ] = (   self.energy_volumes_discharge        [period] 
                                                                                                               + self.BTM_co_loc_gen_energy_export_vol[period] 
                                                                                                               + self.FCAS_rreg_volumes_discharge     [period] 
                                                                                                               + discharge_contingency_mkt                         ) <= curtailment.iloc[period-1]['max_connection']
                self.model.constraints[ str(period)+ str(discharge_contingency_mkt) + 'Max_raise_service'] = pulp.lpSum(  self.energy_volumes_discharge   [period] 
                                                                                                                        + self.FCAS_rreg_volumes_discharge[period] 
                                                                                                                        + discharge_contingency_mkt                ) <= discharge_ratio[period] * scn_mgr['Pwr_cap'].iloc[0]
                self.model.constraints[ str(period)+str(charge_contingency_mkt)+ '_Rated_connection_size'] = pulp.lpSum( self .energy_volumes_charge           [period] 
                                                                                                                        + self.FCAS_lreg_volumes_charge        [period] 
                                                                                                                        + charge_contingency_mkt                   
                                                                                                                        + self.BTM_Grid_to_load                [period])>= -Rated_connection_size                                                                                                         

        for item in self.soc_begin_period:
            self.soc_begin_period[item].upBound = battery_energy_current
        for item in self.soc_end_period:
            self.soc_end_period  [item].upBound = battery_energy_current
        
        if (self.ppa_sim):
            self.update_ppa_offtake(daily_periods,LReg_thrput)

    # ============================================================================  
        """ This function adds the PPA formulation to the model if there is PPA contract in scenario manager
        """

    def add_ppa_model(self,daily_periods):

        scn_mgr             = self.input_data['Scn_mgr']
        percent_method      = scn_mgr['PPA load curve'].iloc[0] == 'None'
        chrg_efficiency     = scn_mgr['Chrg_efncy'    ].iloc[0] 
        disch_efficiency    = scn_mgr['Disch_efncy'   ].iloc[0]

        if percent_method:
            PPA_contract        = self.input_data['PPA contract']
            PPA_percent_offtake = float(scn_mgr['PPA offtake'].iloc[0]) * PPA_contract
        connection_size = self.input_data['Scn_mgr']['Connection size'].iloc[0]
        LReg_thrput     = scn_mgr['Svc_throuput_Lreg'].iloc[0]

        self.PPA_offtake  = pulp.LpVariable.dicts ('PPA_offtake' ,[period for period in daily_periods], 0, connection_size, cat=pulp.LpContinuous )
        self.solar_export = pulp.LpVariable.dicts ('solar_export',[period for period in daily_periods], 0, 1              , cat=pulp.LpBinary     )

        for period in daily_periods:
            add_constraint(self.model, self.BTM_co_loc_gen_energy_export_vol[period]                                        <= 1000 * self.solar_export[period]    , str(period)+'_charge_abs' ) 
            add_constraint(self.model, self.energy_volumes_charge           [period] + self.FCAS_lreg_volumes_charge[period]>= 1000 *(self.solar_export[period]-1)                             )                          
        
        if percent_method:
        #### PPA % offtake #####
            for period in daily_periods:
                add_constraint(self.model, self.PPA_offtake[period] == PPA_percent_offtake * ( self.BTM_co_loc_gen_energy_export_vol                [period] 
                                                                                              -  ( self.BTM_lreg_vol_charge_from_export_avail_gen   [period] * LReg_thrput  # minus as charge value are negative
                                                                                                 + self.BTM_energy_vol_charge_from_export_avail_gen [period]
                                                                                                 + self.BTM_energy_vol_charge_from_curtailment      [period])*chrg_efficiency *disch_efficiency)         , str(period)+'_PPA_offtake'
                               )
        else:
        ### PPA  MWh offtake ####
            load_offtake    = self.input_data['PPA load offtake']
            price           = self.input_data['Price']
            location        = scn_mgr ['Location'           ].iloc[0]
            load_obligation = scn_mgr ['PPA Load Obligation'].iloc[0] == 'Strict'
            soc_end_of_day  = scn_mgr ['SOC End of Day'     ].iloc[0]

            self.charging_mode       = pulp.LpVariable.dicts('charging_mode'  ,[period for period in daily_periods], 0, 1               , cat=pulp.LpBinary    ) # when there is load , charge and discharge occurs at the same time
            if load_obligation:
                self.PPA_load_exceed = pulp.LpVariable.dicts('PPA_Load_Exceed',[period for period in daily_periods], 0, 0               , cat=pulp.LpContinuous) # does not allow the load exceed the generation
            else: 
                self.PPA_load_exceed = pulp.LpVariable.dicts('PPA_Load_Exceed',[period for period in daily_periods], 0, connection_size , cat=pulp.LpContinuous) # allows the PPA load exceeds the dispatch and penalise based on ppa price

            for period in daily_periods:
                add_constraint(self.model, self.PPA_offtake[period] == load_offtake[period], str(period)+'_ppa_load') #forces the system to meet the load   
                add_constraint(self.model, self.PPA_offtake[period] <= ( self.BTM_co_loc_gen_energy_export_vol          [period] 
                                                                        + self.BTM_lreg_vol_charge_from_export_avail_gen[period]*LReg_thrput 
                                                                        + self.energy_volumes_discharge                 [period] 
                                                                        + self.PPA_load_exceed                          [period])     , str(period)+'_PPA_offtake')

                add_constraint(self.model, (- self.energy_volumes_charge                        [period] 
                                            - self.BTM_energy_vol_charge_from_curtailment       [period] 
                                            - self.BTM_energy_vol_charge_from_export_avail_gen  [period])  <=     self.charging_mode[period] * 1000)
                add_constraint(self.model,    self.energy_volumes_discharge[period]                        <=  (1-self.charging_mode[period])* 1000)

                #if there is PPA load ignore the constraint on min avg spread
                if load_offtake[period] != 0:
                    daily_spread = 0
                    self.model.constraints[str(period)+'_min_avg_spread'] = (  self.energy_volumes_discharge[period]
                                                                             * self.resolution
                                                                             * price.iloc[period-1][location]        ) >= ( self.energy_volumes_discharge[period]
                                                                                                                           * self.resolution
                                                                                                                           * daily_spread                         )
                #if there is strict obligation for meeting PPA load, BESS need to have enough stored energy
                if (period == max(daily_periods)) & (sum(load_offtake.values())>0) & (not soc_end_of_day == 'None'):
                    add_constraint(self.model, self.soc_end_period[period] >= soc_end_of_day , str(period) + '_soc_PPA_load' )

    # ============================================================================  

    def update_ppa_offtake(self,daily_periods,LReg_thrput):
        scn_mgr             = self.input_data['Scn_mgr']
        percent_method      = scn_mgr['PPA load curve'].iloc[0] == 'None'
        chrg_efficiency         = scn_mgr['Chrg_efncy'       ].iloc[0] 
        disch_efficiency        = scn_mgr['Disch_efncy'      ].iloc[0] 

        if percent_method:
            PPA_contract        = self.input_data['PPA contract']
            PPA_percent_offtake = float(scn_mgr['PPA offtake'].iloc[0]) * PPA_contract

            for period in daily_periods:
                self.model.constraints[str(period)+'_PPA_offtake'] =  self.PPA_offtake[period] == PPA_percent_offtake * (   self.BTM_co_loc_gen_energy_export_vol               [period] 
                                                                                                                          - (  self.BTM_lreg_vol_charge_from_export_avail_gen   [period]*LReg_thrput # minus as charge value are negative
                                                                                                                             + self.BTM_energy_vol_charge_from_export_avail_gen [period]
                                                                                                                             + self.BTM_energy_vol_charge_from_curtailment      [period])*chrg_efficiency *disch_efficiency                                                                                                          
                                                                                                                        )
        else:  
            load_offtake   = self.input_data['PPA load offtake']
            price          = self.input_data['Price'           ]
            scn_mgr        = self.input_data['Scn_mgr'         ]      
            location       = scn_mgr['Location'      ].iloc[0]
            soc_end_of_day = scn_mgr['SOC End of Day'].iloc[0]

            for period in daily_periods:
                self.model.constraints[ str(period)+'_ppa_load'   ] = self.PPA_offtake[period] == load_offtake[period]
                self.model.constraints[ str(period)+'_PPA_offtake'] = self.PPA_offtake[period] <= (   self.BTM_co_loc_gen_energy_export_vol         [period] 
                                                                                                    + self.BTM_lreg_vol_charge_from_export_avail_gen[period]*LReg_thrput 
                                                                                                    + self.energy_volumes_discharge                 [period] 
                                                                                                    + self.PPA_load_exceed                          [period]              )               
                if load_offtake[period] != 0:
                    daily_spread = 0
                    self.model.constraints[str(period)+'_min_avg_spread'] = (   self.energy_volumes_discharge[period]
                                                                              * self.resolution
                                                                              * price.iloc[period-1][location]       ) >= (  self.energy_volumes_discharge[period]
                                                                                                                           * self.resolution
                                                                                                                           * daily_spread)

                if (period == max(daily_periods)) & (sum(load_offtake.values())>0) & (not soc_end_of_day == 'None'):
                    self.model.constraints[ str(period)+'_soc_PPA_load'] = self.soc_end_period[period] >= 7 * 8

        

    # ============================================================================  
        """ this function sets the objective for LP model
        """
 
    def set_objective(self):
        scn_mgr           = self.input_data['Scn_mgr'      ]
        price             = self.input_data['Price'        ]  
        DNSP_tariff       = self.input_data['DNSP_tariff'  ]      
        LGC_STC           = self.input_data['LGC_STC'      ]
        MLF               = self.input_data['MLF'          ]
        Foresight_period  = self.input_data['Scn_mgr'      ]['Foresight Optimisation period'].iloc[0]
        daily_periods     = range(1,int(Foresight_period)+1)
        LReg_thrput       = scn_mgr['Svc_throuput_Lreg'].iloc[0]
        location          = scn_mgr['Location'         ].iloc[0]
        retail_margin_rev = 1-scn_mgr['retail_margin_rev'].iloc[0]
        retail_margin     = 1+scn_mgr['retail_margin_rev'].iloc[0]
        chrg_efficiency   = scn_mgr['Chrg_efncy'       ].iloc[0] 
        disch_efficiency  = scn_mgr['Disch_efncy'      ].iloc[0] 
        Tariff_constant   = self.input_data['Tariff_constant']
        
        self.energy_revenue = pulp.lpSum(  
              pulp.LpAffineExpression([(self.energy_volumes_charge        [period], self.resolution * price.iloc[period-1][location] * Tariff_constant * self.input_data['DLF'] * MLF['MLF_load'      ].iloc[0] * retail_margin                                           )  for period in daily_periods])
            + pulp.LpAffineExpression([(self.energy_volumes_discharge     [period], self.resolution * price.iloc[period-1][location] * self.input_data['DLF'] * MLF['MLF_generation'].iloc[0] * retail_margin_rev                                       )  for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_rreg_volumes_discharge  [period], self.resolution * price.iloc[period-1][location] * self.input_data['DLF'] * MLF['MLF_generation'].iloc[0] * retail_margin_rev* scn_mgr['Svc_throuput_Rreg'].iloc[0] )  for period in daily_periods]) # only paid if registered as generator and not a MASP need a trigger for this
            + pulp.LpAffineExpression([(self.FCAS_lreg_volumes_charge     [period], self.resolution * price.iloc[period-1][location] * Tariff_constant * self.input_data['DLF'] * MLF['MLF_load'      ].iloc[0] * retail_margin    * LReg_thrput                          )  for period in daily_periods]) # only have to pay if registered as customer and not a MASP need a trigger for this
                                        )
                                
        self.LGC_revenue = pulp.lpSum(# LGC percentage to be set to 1 if BTM
              pulp.LpAffineExpression([(self.energy_volumes_charge        [period], self.resolution * LGC_STC['LGC price'        ].iloc[0] * LGC_STC['LGC percentage'].iloc[0]                                     * self.input_data['DLF'] * MLF['MLF_load'      ].iloc[0]) for period in daily_periods])
            + pulp.LpAffineExpression([(self.energy_volumes_discharge     [period], self.resolution * LGC_STC['LGC price'        ].iloc[0] * LGC_STC['LGC percentage'].iloc[0]                                     * self.input_data['DLF'] * MLF['MLF_generation'].iloc[0]) for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_lreg_volumes_charge     [period], self.resolution * LGC_STC['LGC price'        ].iloc[0] * LGC_STC['LGC percentage'].iloc[0] * LReg_thrput                       * self.input_data['DLF'] * MLF['MLF_load'      ].iloc[0]) for period in daily_periods])# only have to pay if registered as customer and not a MASP need a trigger for this
            + pulp.LpAffineExpression([(self.FCAS_rreg_volumes_discharge  [period], self.resolution * scn_mgr['Svc_throuput_Rreg'].iloc[0] * LGC_STC['LGC price'     ].iloc[0] * LGC_STC['LGC percentage'].iloc[0] * self.input_data['DLF'] * MLF['MLF_generation'].iloc[0]) for period in daily_periods])
                                )
        self.STC_revenue = pulp.lpSum(
              pulp.LpAffineExpression([(self.energy_volumes_charge        [period], self.resolution * LGC_STC['STC price'        ].iloc[0] * LGC_STC['STC percentage'].iloc[0]                                     * self.input_data['DLF'] * MLF['MLF_load'      ].iloc[0]) for period in daily_periods])
            + pulp.LpAffineExpression([(self.energy_volumes_discharge     [period], self.resolution * LGC_STC['STC price'        ].iloc[0] * LGC_STC['STC percentage'].iloc[0]                                     * self.input_data['DLF'] * MLF['MLF_generation'].iloc[0]) for period in daily_periods])#### ASSUMES NET LOAD LGC PAYMENTS
            + pulp.LpAffineExpression([(self.FCAS_lreg_volumes_charge     [period], self.resolution * LGC_STC['STC price'        ].iloc[0] * LGC_STC['STC percentage'].iloc[0] * LReg_thrput                       * self.input_data['DLF'] * MLF['MLF_load'      ].iloc[0]) for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_rreg_volumes_discharge  [period], self.resolution * scn_mgr['Svc_throuput_Rreg'].iloc[0] * LGC_STC['STC price'     ].iloc[0] * LGC_STC['STC percentage'].iloc[0] * self.input_data['DLF'] * MLF['MLF_generation'].iloc[0]) for period in daily_periods])#### ASSUMES NET LOAD LGC PAYMENTS
                                ) 
        self.FCAS_Reg_revenue = pulp.lpSum(
              pulp.LpAffineExpression([(self.FCAS_rreg_volumes_discharge              [period], self.resolution  * price.iloc[period-1]['RAISEREGRRP']) for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_lreg_volumes_charge                 [period],-self.resolution  * price.iloc[period-1]['LOWERREGRRP']* Tariff_constant ) for period in daily_periods]) # get paid for this so need negaitve sign to offset negative volume sign that would make this a cost
            + pulp.LpAffineExpression([(self.BTM_lreg_vol_charge_from_export_avail_gen[period],-self.resolution  * price.iloc[period-1]['LOWERREGRRP']* Tariff_constant ) for period in daily_periods])
                                          )
        self.FCAS_Cont_revenue =  pulp.lpSum(
              pulp.LpAffineExpression([(self.FCAS_r1sec_volumes_discharge [period],  self.resolution * price.iloc[period-1]['RAISE1SECRRP' ])  for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_r6sec_volumes_discharge [period],  self.resolution * price.iloc[period-1]['RAISE6SECRRP' ])  for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_r60sec_volumes_discharge[period],  self.resolution * price.iloc[period-1]['RAISE60SECRRP'])  for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_r5min_volumes_discharge [period],  self.resolution * price.iloc[period-1]['RAISE5MINRRP' ])  for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_l1sec_volumes_charge    [period], -self.resolution * price.iloc[period-1]['LOWER1SECRRP' ]* Tariff_constant )  for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_l6sec_volumes_charge    [period], -self.resolution * price.iloc[period-1]['LOWER6SECRRP' ]* Tariff_constant )  for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_l60sec_volumes_charge   [period], -self.resolution * price.iloc[period-1]['LOWER60SECRRP']* Tariff_constant )  for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_l5min_volumes_charge    [period], -self.resolution * price.iloc[period-1]['LOWER5MINRRP' ]* Tariff_constant )  for period in daily_periods])
                                            )
        self.DNSP_cost = pulp.lpSum(
              pulp.LpAffineExpression([(self.energy_volumes_charge        [period], self.resolution*DNSP_tariff.iloc[period-1]['DNSP_volume_tariff']            *self.input_data['DLF'] * MLF['MLF_load'].iloc[0]) for period in daily_periods])
            + pulp.LpAffineExpression([(self.FCAS_lreg_volumes_charge     [period], self.resolution*DNSP_tariff.iloc[period-1]['DNSP_volume_tariff']*LReg_thrput*self.input_data['DLF'] * MLF['MLF_load'].iloc[0]) for period in daily_periods])
            + pulp.LpAffineExpression([(self.BTM_Grid_to_load             [period], self.resolution*DNSP_tariff.iloc[period-1]['DNSP_volume_tariff']            *self.input_data['DLF'] * MLF['MLF_load'].iloc[0]) for period in daily_periods])                     
                        )

        if self.ppa_sim:
            price_diff = self.input_data['price_diff']
            PPA_price  = self.input_data['PPA_price' ]['PPA_Price'].iloc[0]
            percent_method = scn_mgr['PPA load curve'].iloc[0] == 'None'

            self.PPA_rev =  pulp.LpAffineExpression([(self.PPA_offtake    [period], self.resolution * price_diff.iloc[period-1]['Price_Diff']                      * self.input_data['DLF_cogen']* MLF['co gen'].iloc[0]) for period in daily_periods])
            if not percent_method:
                self.PPA_rev =  pulp.lpSum(
                       pulp.LpAffineExpression([(self.PPA_offtake         [period], self.resolution * price_diff.iloc[period-1]['Price_Diff']                      * self.input_data['DLF_cogen']* MLF['co gen'].iloc[0]) for period in daily_periods])
                     - pulp.LpAffineExpression([(self.PPA_load_exceed     [period], self.resolution * (PPA_price+self.eps-price_diff.iloc[period-1]['Price_Diff']) * self.input_data['DLF_cogen']* MLF['co gen'].iloc[0]) for period in daily_periods])
                                          )
        else:
            self.PPA_rev = 0
        
        self.BTM_rev_cost = pulp.lpSum(
              pulp.LpAffineExpression([(self.BTM_co_loc_gen_energy_export_vol         [period],self.resolution * price.iloc[period-1][location]  * self.input_data['DLF_cogen']*MLF['co gen'].iloc[0]*retail_margin_rev            ) for period in daily_periods])
            + pulp.LpAffineExpression([(self.BTM_co_loc_gen_energy_export_vol         [period],self.resolution * LGC_STC['LGC price'].iloc[0]* LGC_STC['LGC percentage'].iloc[0] * self.input_data['DLF_cogen']*MLF['co gen'].iloc[0]                              ) for period in daily_periods])
            )
            
           #+ pulp.LpAffineExpression([(self.BTM_lreg_vol_charge_from_export_avail_gen[period],self.resolution * price.iloc[period-1][location]  * self.input_data['DLF_cogen']*MLF['co gen'].iloc[0]*retail_margin_rev*LReg_thrput) for period in daily_periods])
            #+ pulp.LpAffineExpression([(self.BTM_lreg_vol_charge_from_export_avail_gen[period],self.resolution * -LGC_STC['LGC price'].iloc[0]    * self.input_data['DLF_cogen']*MLF['co gen'].iloc[0]*retail_margin_rev*LReg_thrput*chrg_efficiency*disch_efficiency) for period in daily_periods])
            #+ pulp.LpAffineExpression([(self.BTM_energy_vol_charge_from_curtailment[period],self.resolution * -LGC_STC['LGC price'].iloc[0] * self.input_data['DLF_cogen']*MLF['co gen'].iloc[0]*chrg_efficiency*disch_efficiency) for period in daily_periods]) # include RTE of battery to take account of RTE losses for LGC eligible MWh from curtailed
            #+ pulp.LpAffineExpression([(self.BTM_energy_vol_charge_from_export_avail_gen[period],self.resolution * -LGC_STC['LGC price'].iloc[0] * self.input_data['DLF_cogen']*MLF['co gen'].iloc[0]*chrg_efficiency*disch_efficiency) for period in daily_periods]) # include RTE of battery to take account of RTE losses for LGC eligible MWh from available solar gen
                                      
        
        self.BTM_Residential_load = pulp.lpSum(
            pulp.LpAffineExpression([(self.BTM_Grid_to_load  [period],self.resolution *price.iloc[period-1][location]* Tariff_constant  *MLF['MLF_load'].iloc[0]) for period in daily_periods])   )
        
        
        objective =  self.energy_revenue + self.LGC_revenue + self.STC_revenue + self.FCAS_Reg_revenue + self.FCAS_Cont_revenue + self.DNSP_cost+self.BTM_rev_cost + self.PPA_rev + self.BTM_Residential_load
        self.model.setObjective(objective)
    
    # ============================================================================  
        """ this function solves the LP model
        the solver and log is read from model_config.yaml 
        """
    def solve_model(self):
        _solver  = None
        s_name   = self.model_config['solver'     ]
        w_log    = self.model_config['write_log'  ]
        disp_log = self.model_config['display_log']
        # mip_gap = self.model_config['mip_gap']
        tl       = self.model_config ['time_limit']
        #gapRel (float) #– relative gap tolerance for the solver to stop (in fraction)
        #gapAbs (float) #– absolute gap tolerance for the solver to stop   gapAbs=(.01)
        #get_status(filename)
        #getOptions()

        if not s_name or s_name == 'cbc':
            _solver = pulp.PULP_CBC_CMD(keepFiles=w_log, msg=disp_log,  timeLimit=tl)
        elif s_name == 'gurobi':
            # One can use GUROBI_CMD like CPLEX_CMD and pass mip_gap and time_limit as options
            _solver = pulp.GUROBI      (                 msg=w_log,     timeLimit=tl)
        elif s_name == 'cplex':
            _solver = pulp.CPLEX_CMD   (keepFiles=w_log, msg=disp_log,  timelimit=tl)
        elif s_name == 'xpress':
            _solver = pulp.XPRESS      (keepFiles=w_log, msg=disp_log,  timeLimit=tl)
        self.model.solve(_solver)

    # ============================================================================  
        """ this function gets the solution of the LP model, including optimal
        solution and optimal values of the decision variables
        """           
    def model_output(self): 
        vars   = self.model.variables()
        obj    = self.model.objective.value()
        output = {'Objective':obj,'Variables': vars}
        return output

    def solve_status(self): 
        status   = self.model.status

        #self.model.writeLP(model_output_log, writeSOS=1, mip=1)
        if self.model.status == pulp.LpStatusOptimal:
            x="Optimal solution found."

            # Your solution is both feasible and optimal.
            # You can access variable values using variable.value()
            # For example, variable_value = variable.value()

        elif self.model.status  == pulp.LpStatusInfeasible:
            x="The problem is infeasible."

        elif self.model.status == pulp.LpStatusUnbounded:
            x="The problem is unbounded."

        else:
            x="The solver did not find an optimal solution."

        return x
