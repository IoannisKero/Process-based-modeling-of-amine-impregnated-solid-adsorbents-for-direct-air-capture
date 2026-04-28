# Copyright (c) 2026 Ioannis Keroglou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the LICENSE file for the specific language governing permissions
# and limitations under the License.
"""
Author: Ioannis Keroglou
Affiliation: School of Sustainability Engineering and Environmental Engineering,
             Purdue University, West Lafayette, IN, USA 
Date: April 2026
Project: Process-based modelling of direct air capture sorbent production:
         A comparative life cycle assessment of amine-impregnated solid adsorbents
"""

# ========================================================================================================================
# STEP 0: IMPORT FUNCTIONS AND LIBRARIES
# ========================================================================================================================

import numpy as np
import Mass_Function as mf
import Chemical_Properties as cp
import Physical_Properties as pp
import Chemical_Properties_Mixture as cpm
import Energy_Functions as ef
import Save_Results_Functions as srf
import Material_Input_Waste_Byproducts as im
import Monte_Carlo_Simulations as mcs
import Scenarios as sc
import Adsorption_Performance as ap
import Impact_Assessment as ia
from collections import OrderedDict
from tqdm import tqdm

# ========================================================================================================================
# STEP 1: CALL SCENARIOS FUNCTION & STORE MASS ANALYSIS RESULTS FOR ALL SCENARIOS
# ========================================================================================================================

# Call Scenarios Function
all_scenarios = sc.generate_scenarios()

# Filter scenarios to a subset for this run (scenario IDs 1 to 10)
scenarios = {k: v for k, v in all_scenarios.items() if   1 <= k <= 10}
print(f"Running model for scenarios: {list(scenarios.keys())}")

# Create dictionaries and lists to store results for all scenarios

# Sorbent Mass 
Sorbent_mass_Dic = {}

# Aziridine Synthesis
Aziridine_results_Lit_Dfs = []
Aziridine_results_Dic = {}
Aziridine_results_Dfs = []

# PEI Synthesis
PEI_results_Lit_Dfs = []
PEI_results_Dic = {}
PEI_results_Dfs = []

# Support Synthesis
Support_results_Lit_Dfs = []
Support_results_Dic = {}
Support_results_Dfs = []

# Sorbent Synthesis
Sorbent_results_Dic = {}
Sorbent_results_Dfs = []

# Run mass analysis for each scenario (Monte Carlo size = 5000)

mcs_number = 5000

print("🔄 Running Mass Analysis for all scenarios...")
for scenario_id, scenario_data in tqdm(scenarios.items(), desc="Mass Analysis", unit="scenario"):

    # Stage-wise mass balance outputs for each scenario
    
    Sorbent_mass, Sorbent_results_dic, Sorbent_results_df, Support_results_dic, Support_results_Df, PEI_results_dic, PEI_results_Df, Aziridine_results_dic, Aziridine_results_Df, Support_lit_Df, PEI_lit_Df, Aziridine_lit_Df = mf.mass_analysis(scenario_id, mcs_number)

# Store scenario outputs in dictionaries

    # Sorbent mass (final product basis)
    Sorbent_mass_Dic[scenario_id] = Sorbent_mass
    
    # Aziridine Synthesis
    Aziridine_results_Dic[scenario_id] = Aziridine_results_dic
    
    # PEI Synthesis
    PEI_results_Dic[scenario_id] = PEI_results_dic

    # Support Synthesis
    Support_results_Dic[scenario_id] = Support_results_dic
    
    # PEI/Support Mixture 
    Sorbent_results_Dic[scenario_id] = Sorbent_results_dic
    
# Store per-scenario DataFrames for later export/post-processing
    
    # Aziridine Synthesis
    Aziridine_results_Lit_Dfs.append(Aziridine_lit_Df)
    Aziridine_results_Dfs.append(Aziridine_results_Df)

    # PEI Synthesis
    PEI_results_Lit_Dfs.append(PEI_lit_Df)
    PEI_results_Dfs.append(PEI_results_Df)

    # Support Synthesis
    Support_results_Lit_Dfs.append(Support_lit_Df)
    Support_results_Dfs.append(Support_results_Df)

    # Sorbent Synthesis 
    Sorbent_results_Dfs.append(Sorbent_results_df)

# ========================================================================================================================
# STEP 2: BUILD MATERIAL INPUTS AND SOLVENT-RECOVERY INVENTORIES
# ========================================================================================================================

print("\n🔄 Calculating Compound Input Mass for all stages...")
# Material input mass per stage
Aziridine_mass_input_dic = im.material_mass_input(Aziridine_results_Dic, scenarios, PEI_results_Dic, "Aziridine")
PEI_mass_input_dic = im.material_mass_input(PEI_results_Dic, scenarios, PEI_results_Dic, "PEI")
Support_mass_input_dic = im.material_mass_input(Support_results_Dic, scenarios, Support_results_Dic, "Support")
Sorbent_mass_input_dic = im.material_mass_input(Sorbent_results_Dic, scenarios, Sorbent_results_Dic, "Sorbent")

# Solvent-recovery and mixture-level solvent-loss dictionaries
Aziridine_solvents_recovery_dic, Aziridine_solvents_recovery_mixture_dic = im.calculate_mass_loss_per_step(Aziridine_results_Dic, scenarios, Aziridine_results_Dic, "Aziridine", return_mixture_loss=True)
Support_solvents_recovery_dic, Support_solvents_recovery_mixture_dic = im.calculate_mass_loss_per_step(Support_results_Dic, scenarios, Support_results_Dic, "Support", return_mixture_loss=True)
Sorbent_solvents_recovery_dic, Sorbent_solvents_recovery_mixture_dic = im.calculate_mass_loss_per_step(Sorbent_results_Dic, scenarios, Sorbent_results_Dic, "Sorbent", return_mixture_loss=True)

# ========================================================================================================================
# STEP 3: CALL CHEMICAL & PHYSICAL PROPERTIES FUNCTIONS 
# ========================================================================================================================

# Chemical properties by stage/component
Support_cp, Aziridine_cp, PEI_cp, Sorbent_cp = cp.heat_capacity() # Heat capacity
Support_density, Aziridine_density, PEI_density, Sorbent_density = cp.density() # Density
Support_ΔH, Aziridine_ΔH, PEI_ΔH, Sorbent_ΔΗ = cp.enthalpy() # Enthalpy

# Process operating conditions
Process_T = cp.process_temperature(mcs_number) # Temperature
Process_t = pp.process_time(mcs_number) # Time   

# Equipment performance assumptions
n_lab, n_ind = pp.efficiency() # Efficiency
P_lab, P_ind = pp.power_consumption() # Power consumption

# ========================================================================================================================
# STEP 4: CALL ENERGY FUNCTIONS FOR AZIRIDINE SYNTHESIS
# ========================================================================================================================

# Initialize empty dictionary to save energy results
Aziridine_Energy = {}
Aziridine_Energy_per_kg = {}

# Iterate through all scenarios
print("\n🔄 Calculating Aziridine Energy for all scenarios...")
for scenario_id, scenario_data in tqdm(scenarios.items(), desc="Aziridine Energy", unit="scenario"):
    
    # Initialize empty dictionary to save energy results
    Aziridine_Energy[scenario_id] = {}
    Aziridine_Energy_per_kg[scenario_id] = {}
    
# Iterate over all sorbent formulations in this scenario
    for sorbent_type in tqdm(Aziridine_results_Dic[scenario_id].keys(), desc=f"  {scenario_id} sorbents", leave=False):

        # Extract scenario/sorbent-specific mass inventory and conditions
        aziridine_mass_dict = Aziridine_results_Dic[scenario_id][sorbent_type]
        T = Process_T[sorbent_type]
        t = Process_t[sorbent_type]

        # Solvent recovery: {sorbent_type: {scenario_id: {solvent: {"mass_kg", "phase"}}}}
        M_solvent_recovery = Aziridine_solvents_recovery_dic.get(sorbent_type, {}).get(scenario_id, {})
        M_solvent_recovery_mixture = Aziridine_solvents_recovery_mixture_dic.get(sorbent_type, {}).get(scenario_id, {})

        # Final sorbent mass is used to normalize kWh/kg-sorbent metrics
        M_sorbent = Sorbent_mass_Dic[scenario_id]

        # Call Heat Capacity - Mixture Function
        cp, M = cpm.heat_capacity_sum(aziridine_mass_dict, Aziridine_cp)

        # Call Heat Capacity - Solvents Function
        cp_solvents, M_solvents = cpm.heat_capacity_solvents(aziridine_mass_dict, Aziridine_cp)

        # Call Density - Mixture Function
        Aziridine_Density, Mixture_Volume, Mixture_Mass = cpm.density_sum(aziridine_mass_dict, Aziridine_density)

        # Distillation enthalpy includes Aziridine and solvents
        DH_vap_dist, M_dist = cpm.enthalpy_sum(aziridine_mass_dict, Aziridine_ΔH)

        # Call Enthalpy for Drying (Only solvents)
        Aziridine_ΔH_drying = Aziridine_ΔH.copy()
        Aziridine_ΔH_drying.pop("Aziridine")
        DH_vap_drying, M_drying = cpm.enthalpy_sum(aziridine_mass_dict, Aziridine_ΔH_drying)   
        
        # Calculation of Stirring Parameters
        N, D, R, Q_loss = pp.modeling_parameters(Mixture_Volume)

        # Recovered byproduct mass (Sodium Sulfate) with uncertainty sampling
        M_byproduct_recovered = aziridine_mass_dict["Sodium Sulfate"][:, [6], :] * np.random.triangular(0.95, 0.97, 0.99, mcs_number)
  
    # Call Energy Functions     
        
        # Stirring
        E_stir, E_stir_kg, E_kWh_stir_matrix, E_per_kg_stir_matrix = ef.stirring(M, cp, T, t, Aziridine_Density, N, D, R, n_lab['Stirring'] ,n_ind['Stirring'], n_lab['Stirring'] ,n_ind['Stirring'], scenario_data, Mixture_Volume, M_sorbent, mcs_number, Q_loss)
                
        # Heating
        E_heat, E_heat_kg, E_kWh_heat_matrix, E_per_kg_heat_matrix = ef.heating(M, cp, T, n_lab['Heating'], n_ind['Heating'], scenario_data, M_sorbent, 1, mcs_number, Q_loss)
        
        # Grinding
        E_grind, E_grind_kg, E_kWh_grind_matrix, E_per_kg_grind_matrix = ef.grinding(M, n_lab['Grinding'], P_lab['Grinding'], t, scenario_data, M_sorbent, Mixture_Volume, Mixture_Mass, M_solvents, mcs_number)
        
        # Vacuum Filtration
        E_filt, E_filt_kg, E_kWh_filt_matrix, E_per_kg_filt_matrix = ef.sol_liq_sep(M, n_lab['Filtration'], n_ind['Filtration'], P_lab['Filtration'], P_ind['Filtration'], t, 3, scenario_data,M_solvents, M_sorbent, Mixture_Mass, Mixture_Volume, mcs_number)         
        
        # Washing
        E_wash, E_wash_kg, E_kWh_wash_matrix, E_per_kg_wash_matrix = ef.sol_liq_sep(M, n_lab['Washing'] ,n_ind['Washing'] , P_lab['Washing'], P_ind['Washing'], t, 4, scenario_data, M_solvents, M_sorbent, Mixture_Mass, Mixture_Volume, mcs_number)
        
        # Drying
        E_dry, E_dry_kg, E_kWh_dry_matrix, E_per_kg_dry_matrix = ef.dry_evap(cp_solvents, T, n_lab['Drying'] ,n_ind['Drying'] , P_lab['Drying'], P_ind['Drying'], DH_vap_drying, M_drying, t, scenario_data, M_sorbent, Mixture_Volume, 7, mcs_number)

        # Distillation
        E_dis, E_dis_kg, E_kWh_dis_matrix, E_per_kg_dis_matrix = ef.distillation(M, cp, T, n_lab['Distillation'] ,n_ind['Distillation'] , DH_vap_dist, M_dist, scenario_data, M_sorbent, mcs_number, Q_loss)

        # Solvent Recovery 
        E_solvent_recovery_heat, E_solvent_recovery_electricity, E_solvent_recovery_heat_kg, E_solvent_recovery_electricity_kg = ef.solvent_recovery(M_solvent_recovery, M_solvent_recovery_mixture, n_lab['Distillation'] ,n_ind['Distillation'] , T, scenario_data, M_sorbent, mcs_number)

        # Byproduct Recovery (Sodium Sulfate) - Electricity Consumption
        E_byproduct = M_byproduct_recovered/1000 * 1.4 # kWh
        E_byproduct_kg = ((M_byproduct_recovered/1000) * 1.4 / M_sorbent) * 1000 # kWh/kg sorbent produced

        # Store the computed data in the results dictionary

        Aziridine_Energy[scenario_id][sorbent_type] = OrderedDict({
            'Stirring': E_stir,
            'Heating': E_heat,
            'Grinding': E_grind,
            'Filtration': E_filt,
            'Washing': E_wash,
            'Drying': E_dry,
            'Distillation': E_dis,
            'Solvent Recovery - Electricity': E_solvent_recovery_electricity,
            'Solvent Recovery - Heat': E_solvent_recovery_heat,
            'Byproduct Recovery': E_byproduct
        })

        Aziridine_Energy_per_kg[scenario_id][sorbent_type] = OrderedDict({
            'Stirring': E_stir_kg,
            'Heating': E_heat_kg,
            'Grinding': E_grind_kg,
            'Filtration': E_filt_kg,
            'Washing': E_wash_kg,
            'Drying': E_dry_kg,
            'Distillation': E_dis_kg,
            'Solvent Recovery - Electricity': E_solvent_recovery_electricity_kg,
            'Solvent Recovery - Heat': E_solvent_recovery_heat_kg,
            'Byproduct Recovery': E_byproduct_kg
        })

# ========================================================================================================================
# STEP 5: CALL ENERGY FUNCTIONS FOR PEI SYNTHESIS
# ========================================================================================================================

# Initialize empty dictionary to save energy results
PEI_Energy = {}
PEI_Energy_per_kg = {}

# Initialize empty dictionary to save final PEI properties
Final_PEI_Cp = {}
Final_PEI_Density = {}

# Iterate through all scenarios
print("\n🔄 Calculating PEI Energy for all scenarios...")
for scenario_id, scenario_data in tqdm(scenarios.items(), desc="PEI Energy", unit="scenario"):
    
    # Initialize empty dictionary to save energy results
    PEI_Energy[scenario_id] = {}
    PEI_Energy_per_kg[scenario_id] = {}
    
    # Iterate over all sorbent formulations in this scenario
    for sorbent_type in tqdm(PEI_results_Dic[scenario_id].keys(), desc=f"  {scenario_id} sorbents", leave=False):
        
        # Extract scenario/sorbent-specific mass inventory and conditions
        PEI_mass_dict = PEI_results_Dic[scenario_id][sorbent_type]
        T = Process_T[sorbent_type]
        t = Process_t[sorbent_type]
        
        # Extract final sorbent mass for the current scenario
        M_sorbent = Sorbent_mass_Dic[scenario_id]

        # Call Heat Capacity - Mixture Function
        cp, M = cpm.heat_capacity_sum(PEI_mass_dict, PEI_cp)

        
        # Call Heat Capacity - Solvents Function
        cp_solvents, M_solvents = cpm.heat_capacity_solvents(PEI_mass_dict, PEI_cp)

        # Call Density - Mixture Function
        PEI_Density, Mixture_Volume, Mixture_Mass = cpm.density_sum(PEI_mass_dict, PEI_density)

        # Call Enthalpy - Solvents Function
        DH_vap, M_s = cpm.enthalpy_sum(PEI_mass_dict, PEI_ΔH)
        
        # Persist final PEI properties for PEI-support blending stage
        Final_PEI_cp = cpm.final_product_property(Final_PEI_Cp, cp, sorbent_type, scenario_id)
        
        # Persist final PEI density for PEI-support blending stage
        Final_PEI_density = cpm.final_product_property(Final_PEI_Density, PEI_Density, sorbent_type, scenario_id)
        
        # Calculation of Stirring Parameters
        N, D, R, Q_loss = pp.modeling_parameters(Mixture_Volume)
        
    # Call Energy Functions

        # Stirring
        E_stir, E_stir_kg, E_kWh_stir_matrix, E_per_kg_stir_matrix = ef.stirring(M, cp, T, t, PEI_Density, N, D, R, n_lab['Stirring'] ,n_ind['Stirring'], n_lab['Stirring'] ,n_ind['Stirring'], scenario_data, Mixture_Volume, M_sorbent, mcs_number, Q_loss)
        
        # Oven
        E_aging, E_aging_kg, E_kWh_aging_matrix, E_per_kg_aging_matrix = ef.heating(M, cp, T, n_lab['Heating'], n_ind['Heating'], scenario_data, M_sorbent, 9, mcs_number, Q_loss)
        
        # Rotary Evaporation
        E_evap, E_evap_kg, E_kWh_evap_matrix, E_per_kg_evap_matrix = ef.dry_evap(cp_solvents, T, n_lab['Evaporation'] ,n_ind['Evaporation'] , P_lab['Drying'], P_ind['Drying'], DH_vap, M_solvents, t, scenario_data, M_sorbent, Mixture_Volume, 10, mcs_number)
        
        # Drying
        E_dry, E_dry_kg, E_kWh_dry_matrix, E_per_kg_dry_matrix = ef.dry_evap(cp_solvents, T, n_lab['Drying'] ,n_ind['Drying'] , P_lab['Drying'], P_ind['Drying'], DH_vap, M_solvents, t, scenario_data, M_sorbent, Mixture_Volume, 11, mcs_number)

        # Store the computed data in the results dictionary

        PEI_Energy[scenario_id][sorbent_type] = {
            'Stirring': E_stir,
            'Aging': E_aging,
            'Evaporation': E_evap,
            'Drying': E_dry,
            }
        
        PEI_Energy_per_kg[scenario_id][sorbent_type] = {
            'Stirring': E_stir_kg,
            'Aging': E_aging_kg,
            'Evaporation': E_evap_kg,
            'Drying': E_dry_kg,
            }

# ========================================================================================================================
# STEP 6: CALL ENERGY FUNCTIONS FOR SUPPORT SYNTHESIS
# ========================================================================================================================

# Initialize empty dictionary to save energy results
Support_Energy = {}
Support_Energy_per_kg = {}

# Initialize empty dictionary to save final support properties
Final_Support_Cp = {}
Final_Support_Density = {}

# Initialize a list to store reactor parameters
Reactor_params = []

# Iterate through all scenarios 
print("\n🔄 Calculating Support Energy for all scenarios...")
for scenario_id, scenario_data in tqdm(scenarios.items(), desc="Support Energy", unit="scenario"):
    
    # Initialize empty dictionary to save energy results
    Support_Energy[scenario_id] = {}
    Support_Energy_per_kg[scenario_id] = {}
    
    # Iterate over all sorbent formulations in this scenario
    for sorbent_type in tqdm(Support_results_Dic[scenario_id].keys(), desc=f"  {scenario_id} sorbents", leave=False):
         
        # Extract scenario/sorbent-specific mass inventory and properties
        support_mass_dict = Support_results_Dic[scenario_id][sorbent_type]
        support_cp_dict = Support_cp.get(sorbent_type, {})
        support_density_dict = Support_density.get(sorbent_type, {})
        T = Process_T[sorbent_type]
        t = Process_t[sorbent_type]
        
        # Extract final sorbent mass for the current scenario
        M_sorbent = Sorbent_mass_Dic[scenario_id]

         # Solvent recovery: {sorbent_type: {scenario_id: {solvent: {"mass_kg", "phase"}}}}
        M_solvent_recovery = Support_solvents_recovery_dic.get(sorbent_type, {}).get(scenario_id, {})
        M_solvent_recovery_mixture = Support_solvents_recovery_mixture_dic.get(sorbent_type, {}).get(scenario_id, {})
                    
        # Heat Capacity - Mixture
        cp, M = cpm.heat_capacity_sum(support_mass_dict, support_cp_dict)

        # Call Heat Capacity - Solvents Function
        cp_solvents, M_solvents = cpm.heat_capacity_solvents(support_mass_dict, support_cp_dict)
        
        # Density -Mixture
        Support_Density, Mixture_Volume, Mixture_Mass = cpm.density_sum(support_mass_dict, support_density_dict)
        
        # Persist final support properties for PEI-support blending stage
        Final_support_cp = cpm.final_product_property(Final_Support_Cp, cp, sorbent_type, scenario_id)
        
        # Persist final support density for PEI-support blending stage
        Final_support_density = cpm.final_product_property(Final_Support_Density, Support_Density, sorbent_type, scenario_id)
        
        support_enthalpy = Support_ΔH[sorbent_type]
        DH_vap, M_s = cpm.enthalpy_sum(support_mass_dict, support_enthalpy)
        
        # Calculation of Stirring Parameters
        N, D, R, Q_loss = pp.modeling_parameters(Mixture_Volume)

        # Input Mass of Byproduct Recovered (Sodium Sulfate)
        if sorbent_type == "PEI_SG":
            M_byproduct_recovered_support = support_mass_dict["Sodium Sulfate"][:, [17], :] * np.random.triangular(0.95, 0.97, 0.99, mcs_number)

    # Call Energy Functions

        # Stirring
        E_stir, E_stir_kg, E_kWh_stir_matrix, E_per_kg_stir_matrix = ef.stirring(M, cp, T, t, Support_Density, N, D, R, n_lab['Stirring'] ,n_ind['Stirring'], n_lab['Stirring'] ,n_ind['Stirring'], scenario_data, Mixture_Volume, M_sorbent, mcs_number, Q_loss)
        
        # Sonication
        E_son, E_son_kg, E_kWh_son_matrix, E_per_kg_son_matrix = ef.sonication(M, cp, T, t, Support_Density, N, D, R, n_lab['Sonication'], n_ind['Sonication'], P_lab['Sonication'], scenario_data, M_sorbent, Mixture_Volume, mcs_number)
        
        # Aging/Polymerization
        E_aging, E_aging_kg, E_kWh_aging_matrix, E_per_kg_aging_matrix = ef.heating(M, cp, T, n_lab['Aging'] ,n_ind['Aging'] , scenario_data, M_sorbent, np.r_[18,19], mcs_number, Q_loss)        

        # Centrifugation
        E_centr, E_centr_kg, E_kWh_centr_matrix, E_per_kg_centr_matrix = ef.sol_liq_sep(M, n_lab['Centrifugation'] ,n_ind['Centrifugation'] , P_lab['Centrifugation'], P_ind['Centrifugation'], t, 23, scenario_data, M_solvents, M_sorbent, Mixture_Mass, Mixture_Volume, mcs_number)
        
        # Vacuum Filtration
        E_filt, E_filt_kg, E_kWh_filt_matrix, E_per_kg_filt_matrix = ef.sol_liq_sep(M, n_lab['Filtration'], n_ind['Filtration'], P_lab['Filtration'], P_ind['Filtration'], t, 21, scenario_data, M_solvents, M_sorbent, Mixture_Mass, Mixture_Volume, mcs_number)         
        
        # Washing
        E_wash, E_wash_kg, E_kWh_wash_matrix, E_per_kg_wash_matrix = ef.sol_liq_sep(M, n_lab['Washing'] ,n_ind['Washing'] , P_lab['Washing'], P_ind['Washing'], t, 25, scenario_data, M_solvents, M_sorbent, Mixture_Mass, Mixture_Volume, mcs_number)
        
        # Evaporation
        E_evap, E_evap_kg, E_kWh_evap_matrix, E_per_kg_evap_matrix = ef.dry_evap(cp_solvents, T, n_lab['Evaporation'] ,n_ind['Evaporation'] , P_lab['Drying'], P_ind['Drying'], DH_vap, M_solvents, t, scenario_data, M_sorbent, Mixture_Volume, 24, mcs_number)
        
        # Drying
        E_dry, E_dry_kg, E_kWh_dry_matrix, E_per_kg_dry_matrix = ef.dry_evap(cp_solvents, T, n_lab['Drying'] ,n_ind['Drying'] , P_lab['Drying'], P_ind['Drying'], DH_vap, M_solvents, t, scenario_data, M_sorbent, Mixture_Volume, 26, mcs_number)

        # Calcination
        E_calc, E_calc_kg, E_kWh_calc_matrix, E_per_kg_calc_matrix = ef.heating(M, cp, T, n_lab['Calcination'] ,n_ind['Calcination'] , scenario_data, M_sorbent, 27, mcs_number, Q_loss)        
        
        # Solvent Recovery 
        E_recovery_heat, E_recovery_electricity, E_recovery_heat_kg, E_recovery_electricity_kg = ef.solvent_recovery(M_solvent_recovery, M_solvent_recovery_mixture, n_lab['Distillation'] ,n_ind['Distillation'] , T, scenario_data, M_sorbent, mcs_number)

        # Byproduct Recovery (Sodium Sulfate) - only for PEI_SG.
        if sorbent_type == "PEI_SG":
            E_byproduct = M_byproduct_recovered_support / 1000 * 1.4  # kWh
            E_byproduct_kg = ((M_byproduct_recovered_support / 1000) * 1.4 / M_sorbent) * 1000  # kWh/kg sorbent
        else:
            E_byproduct = np.zeros_like(E_stir, dtype=float)
            E_byproduct_kg = np.zeros_like(E_stir_kg, dtype=float)

        # Store the computed data in the results dictionary
        
        Support_Energy[scenario_id][sorbent_type] = {
            'Stirring': E_stir,
            'Sonication': E_son,
            'Aging/Polymerization': E_aging,
            'Centrifugation': E_centr,
            'Filtration': E_filt,
            'Washing': E_wash,
            'Evaporation': E_evap,
            'Drying': E_dry,
            'Calcination': E_calc,
            'Solvent Recovery - Electricity': E_recovery_electricity,
            'Solvent Recovery - Heat': E_recovery_heat,
            'Byproduct Recovery': E_byproduct}
        
        Support_Energy_per_kg[scenario_id][sorbent_type] = {
            'Stirring': E_stir_kg,
            'Sonication': E_son_kg,
            'Aging/Polymerization': E_aging_kg,
            'Centrifugation': E_centr_kg,
            'Filtration': E_filt_kg,
            'Washing': E_wash_kg,
            'Evaporation': E_evap_kg,
            'Drying': E_dry_kg,
            'Calcination': E_calc_kg,
            'Solvent Recovery - Electricity': E_recovery_electricity_kg,
            'Solvent Recovery - Heat': E_recovery_heat_kg,
            'Byproduct Recovery': E_byproduct_kg}
  

# ========================================================================================================================
# STEP 7: CALL ENERGY FUNCTIONS FOR SORBENT SYNTHESIS
# ========================================================================================================================

# Initialize empty dictionary to save energy results
Sorbent_Energy = {}
Sorbent_Energy_per_kg = {}

# Iterate through all scenarios
print("\n🔄 Calculating Sorbent Energy for all scenarios...")
for scenario_id, scenario_data in tqdm(scenarios.items(), desc="Sorbent Energy", unit="scenario"):
    
    # Initialize empty dictionary to save energy results
    Sorbent_Energy[scenario_id] = {}
    Sorbent_Energy_per_kg[scenario_id] = {}
    
    # Iterate over all available sorbents
    for sorbent_type in tqdm(Sorbent_results_Dic[scenario_id].keys(), desc=f"  {scenario_id} sorbents", leave=False):
        
        Sorbent_mass_dict = Sorbent_results_Dic[scenario_id][sorbent_type] # Mass dictionary for each support
        Sorbent_cp_dict = Sorbent_cp[sorbent_type] # Cp dictionary for each support
        Sorbent_density_dict = Sorbent_density[sorbent_type] # Density dictionary for each support
        T = Process_T[sorbent_type]
        t = Process_t[sorbent_type]
        
        # Extract final sorbent mass for the current scenario
        M_sorbent = Sorbent_mass_Dic[scenario_id]

        # Solvent recovery
        M_solvent_recovery = Sorbent_solvents_recovery_dic.get(sorbent_type, {}).get(scenario_id, {})
        M_solvent_recovery_mixture = Sorbent_solvents_recovery_mixture_dic.get(sorbent_type, {}).get(scenario_id, {})
        
        # Heat Capacity - Mixture
        cp, M = cpm.heat_capacity_sum(Sorbent_mass_dict, Sorbent_cp_dict)
        
        # Call Heat Capacity - Solvents Function
        cp_solvents, M_solvents = cpm.heat_capacity_solvents(Sorbent_mass_dict, Sorbent_cp_dict)
        
        # Density - Mixture
        Sorbent_Density, Mixture_Volume, Mixture_Mass = cpm.density_sum(Sorbent_mass_dict, Sorbent_density_dict)

        # Retrieve enthalpy values for each process using the proper lookup order:
        Sorbent_Drying_DH = Sorbent_ΔΗ['Drying'][sorbent_type]
        Sorbent_Evaporation_DH = Sorbent_ΔΗ['Evaporation'][sorbent_type]

        # Enthalpy - Solvents
        
        DH_vap_drying, M_solvents_drying = cpm.enthalpy_sum(Sorbent_mass_dict, Sorbent_Drying_DH) # Solvents enthalpy for drying process
        DH_vap_evaporation, M_solvents_evaporation = cpm.enthalpy_sum(Sorbent_mass_dict, Sorbent_Evaporation_DH) # Solvents enthalpy for evaporation process

        # Calculation of Stirring Parameters
        N, D, R, Q_loss = pp.modeling_parameters(Mixture_Volume)

    # Call Energy Functions

        # Activation
        E_activ, E_activ_kg, E_kWh_activ_matrix, E_per_kg_activ_matrix = ef.heating(M, cp, T, n_lab['Activation'] ,n_ind['Activation'] , scenario_data, M_sorbent, 29, mcs_number, Q_loss)    
        
        # Sonication
        E_son, E_son_kg, E_kWh_son_matrix, E_per_kg_son_matrix = ef.sonication(M, cp, T, t, Sorbent_Density, N, D, R, n_lab['Sonication'], n_ind['Sonication'], P_lab['Sonication'], scenario_data, M_sorbent, Mixture_Volume, mcs_number)
 
        # Stirring
        E_stir, E_stir_kg, E_kWh_stir_matrix, E_per_kg_stir_matrix = ef.stirring(M, cp, T, t, Sorbent_Density, N, D, R, n_lab['Stirring'] ,n_ind['Stirring'], n_lab['Stirring'] ,n_ind['Stirring'], scenario_data, Mixture_Volume, M_sorbent, mcs_number, Q_loss)

        # Evaporation
        E_evap, E_evap_kg, E_kWh_evap_matrix, E_per_kg_evap_matrix = ef.dry_evap(cp_solvents, T, n_lab['Evaporation'] ,n_ind['Evaporation'] , P_lab['Drying'], P_ind['Drying'], DH_vap_evaporation, M_solvents_evaporation, t, scenario_data, M_sorbent, Mixture_Volume, 33, mcs_number)

        # Drying
        E_dry, E_dry_kg, E_kWh_dry_matrix, E_per_kg_dry_matrix = ef.dry_evap(cp_solvents, T, n_lab['Drying'] ,n_ind['Drying'] , P_lab['Drying'], P_ind['Drying'], DH_vap_drying, M_solvents_drying, t, scenario_data, M_sorbent, Mixture_Volume, np.r_[28, 34], mcs_number)

        # Solvent Recovery 
        E_recovery_heat, E_recovery_electricity, E_recovery_heat_kg, E_recovery_electricity_kg = ef.solvent_recovery(M_solvent_recovery, M_solvent_recovery_mixture, n_lab['Distillation'] ,n_ind['Distillation'] , T, scenario_data, M_sorbent, mcs_number)

        # Store the computed data in the results dictionary
        Sorbent_Energy[scenario_id][sorbent_type] = {
            'Activation': E_activ,
            'Sonication': E_son,
            'Stirring': E_stir,
            'Evaporation': E_evap,
            'Drying': E_dry,
            'Solvent Recovery - Electricity': E_recovery_electricity,
            'Solvent Recovery - Heat': E_recovery_heat
            }

        Sorbent_Energy_per_kg[scenario_id][sorbent_type] = {
            'Activation': E_activ_kg,
            'Sonication': E_son_kg,
            'Stirring': E_stir_kg,
            'Evaporation': E_evap_kg,
            'Drying': E_dry_kg,
            'Solvent Recovery - Electricity': E_recovery_electricity_kg,
            'Solvent Recovery - Heat': E_recovery_heat_kg
            }

# ========================================================================================================================
# STEP 8: SAVE ENERGY RESULTS TO DATAFRAMES AND EXCEL DATABASE
# ========================================================================================================================

# Save stage-wise energy results to DataFrames
print("\n🔄 Saving Energy Results to DataFrames...")
Aziridine_Energy_Df, Aziridine_Energy_per_kg_Df = srf.energy_save_to_df(Aziridine_Energy, Aziridine_Energy_per_kg, mcs_number=5)
PEI_Energy_Df, PEI_Energy_per_kg_Df = srf.energy_save_to_df(PEI_Energy, PEI_Energy_per_kg, mcs_number=5)
Support_Energy_Df, Support_Energy_per_kg_Df = srf.energy_save_to_df(Support_Energy, Support_Energy_per_kg, mcs_number=5)
Sorbent_Energy_Df, Sorbent_Energy_per_kg_Df = srf.energy_save_to_df(Sorbent_Energy, Sorbent_Energy_per_kg, mcs_number=5)

# Calculate total energy consumption (kWh/kg) per synthesis process per sorbent & scenario
print("🔄 Calculating Total Energy...")

# Dataframes
Total_Energy_per_kg_per_stage_Df = ef.calculate_total_energy(
    [Aziridine_Energy_per_kg_Df, PEI_Energy_per_kg_Df, Support_Energy_per_kg_Df, Sorbent_Energy_per_kg_Df],
    ["Aziridine", "PEI", "Support", "Sorbent"])

Total_Energy_per_kg_Df = ef.calculate_total_energy([Total_Energy_per_kg_per_stage_Df], ["All stages"])

# Combined dictionary structure for downstream reorganization
Total_Energy_per_kg_Dic = ef.combine_energy_per_kg_dictionaries(
    Aziridine_Energy_per_kg, 
    PEI_Energy_per_kg, 
    Support_Energy_per_kg, 
    Sorbent_Energy_per_kg,
    ["Aziridine", "PEI", "Support", "Sorbent"]
)

# ========================================================================================================================
#  STEP 9: REORGANIZE ENERGY DICTIONARY BY ENERGY TYPE (ELECTRICITY OR HEAT)
# ========================================================================================================================

# Map each process step to primary energy carrier (electricity/heat)

process_energy_type = {

    # Aziridine processes
    "Aziridine": {
        "Stirring": "electricity",
        "Heating": "heat",
        "Grinding": "electricity",
        "Filtration": "electricity",
        "Washing": "electricity",
        "Drying": "electricity",
        "Distillation": "heat",
        "Solvent Recovery - Electricity": "electricity",
        "Solvent Recovery - Heat": "heat",
        "Byproduct Recovery": "electricity",
    },
    
    # Polymerization processes
    "PEI": {
        "Stirring": "electricity",
        "Aging": "heat",
        "Evaporation": "heat",
        "Drying": "heat",
    },

    # Support processes
    "Support": {
        "Stirring": {
            "PEI_γ_alumina": "electricity",
            "PEI_SG": "heat",
            "MCM_41_PEI": "electricity",
            "MMSV_PEI": "heat",
            "MIL_101_Cr_PEI": "electricity",
            "SBA_15_PEI": "heat",
            "KIT6_PEI": "heat",
            "HMS_PEI": "heat",
            "polyHIPE_PEI": "electricity",
        },
        "Sonication": "electricity",
        "Aging/Polymerization": "heat",
        "Extraction": "heat", 
        "Centrifugation": "electricity",
        "Filtration": "electricity",
        "Washing": "electricity",
        "Evaporation": "heat",
        "Drying": "heat",
        "Calcination": "heat",
        "Solvent Recovery - Electricity": "electricity",
        "Solvent Recovery - Heat": "heat",
        "Byproduct Recovery": "electricity",
    },
    
    # Impregnation processes
    "Sorbent": {
        "Activation": "heat",
        "Sonication": "electricity",
        "Stirring": "electricity",
        "Evaporation": "heat",
        "Drying": {
            "PEI_γ_alumina": "heat",
            "PEI_SG": "heat",
            "MCM_41_PEI": "heat",
            "MMSV_PEI": "heat",
            "MIL_101_Cr_PEI": "electricity",
            "SBA_15_PEI": "electricity",
            "KIT6_PEI": "heat",
            "HMS_PEI": "heat",
            "polyHIPE_PEI": "heat"
        },
        "Solvent Recovery - Electricity": "electricity",
        "Solvent Recovery - Heat": "heat",
        "Byproduct Recovery": "electricity"
    }

}

# Reorganize energy results by carrier type
print("\n🔄 Reorganizing energy dictionary by energy type (electricity/heat)...")
Total_Energy_per_kg_Dic_final = ef.reorganize_energy_by_type(Total_Energy_per_kg_Dic, process_energy_type)

# Remove zero-energy processes to reduce dictionary size
Total_Energy_per_kg_sorbent_Dic = ef.remove_zero_energy_processes(Total_Energy_per_kg_Dic_final)

# Normalize stage energy per kg of stage product (as configured below)
print("  Normalizing  energy per kg of stage product...")
Total_Energy_per_kg_product_Dic = ef.normalize_energy_per_kg_product(Total_Energy_per_kg_Dic_final, PEI_results_Dic, Support_results_Dic,scenarios, stages_to_normalize=("PEI", "Support", "Sorbent"))
Total_Energy_per_kg_product_Dic = ef.remove_zero_energy_processes(Total_Energy_per_kg_product_Dic)

# ========================================================================================================================
# STEP 10: SORBENT LIFETIME ADSORPTION CAPACITY 
# ========================================================================================================================

# Call Adsorption Performance Function
print("\n🔄 Calculating Adsorption Performance...")
Adsorption_Performance_Dic = ap.adsorption_performance(mcs_number, use_fixed_CC=False, fixed_CC_value=None)

# ========================================================================================================================
# STEP 11: Life Cycle Impact Assessment 
# ========================================================================================================================

# Impact assessment inputs and calculations
print("\n🔄 Calculating Impact Assessment...")
# Retrieve Impact Factors Dictionary
Impact_Factors_Dic = ia.environmental_impact_factors()

# Calculate Electricity and Heat Impacts by process, stage and total
Energy_LC_Impacts_Dic = ia.energy_impacts(Total_Energy_per_kg_sorbent_Dic, Impact_Factors_Dic)

# Calculate Chemical Input Impacts
print("  Calculating chemical input impacts per kg...")
Material_LC_Impacts_Dic = ia.chemical_impacts(Aziridine_mass_input_dic, PEI_mass_input_dic, Support_mass_input_dic, Sorbent_mass_input_dic)

# ========================================================================================================================
# STEP 12: Monte Carlo Statistics (n=5000) - Normalized per kg of product, kg of sorbent and per kg of CO2 captured
# ========================================================================================================================  

# Calculate Monte Carlo Statistics for mass dictionaries
print("\n🔄 Calculating Monte Carlo Statistics for LCI dictionaries...")

# Mass flow per step for each stage
Aziridine_MC_mass_df = mcs.monte_carlo_statistics_mass(Aziridine_results_Dic, Adsorption_Performance_Dic, "Aziridine", scenarios)
PEI_MC_mass_df = mcs.monte_carlo_statistics_mass(PEI_results_Dic, Adsorption_Performance_Dic, "PEI", scenarios)
Support_MC_mass_df = mcs.monte_carlo_statistics_mass(Support_results_Dic, Adsorption_Performance_Dic, "Support", scenarios)
Sorbent_MC_mass_df = mcs.monte_carlo_statistics_mass(Sorbent_results_Dic, Adsorption_Performance_Dic, "Sorbent", scenarios)

# Solvent Recovery Mass Loss per step
Aziridine_MC_solvent_recovery_df = mcs.monte_carlo_statistics_mass_emission(Aziridine_solvents_recovery_dic, Adsorption_Performance_Dic, scenarios, "Aziridine")
Support_MC_solvent_recovery_df = mcs.monte_carlo_statistics_mass_emission(Support_solvents_recovery_dic, Adsorption_Performance_Dic, scenarios, "Support")
Sorbent_MC_solvent_recovery_df = mcs.monte_carlo_statistics_mass_emission(Sorbent_solvents_recovery_dic, Adsorption_Performance_Dic, scenarios, "Sorbent")

# Life Cycle Inventory (LCI) per stage
Aziridine_MC_input_mass_df = mcs.monte_carlo_statistics_input_mass(Aziridine_mass_input_dic, Adsorption_Performance_Dic, scenarios, "Aziridine")
PEI_MC_input_mass_df = mcs.monte_carlo_statistics_input_mass(PEI_mass_input_dic, Adsorption_Performance_Dic, scenarios, "PEI")
Support_MC_input_mass_df = mcs.monte_carlo_statistics_input_mass(Support_mass_input_dic, Adsorption_Performance_Dic, scenarios, "Support")
Sorbent_MC_input_mass_df = mcs.monte_carlo_statistics_input_mass(Sorbent_mass_input_dic, Adsorption_Performance_Dic, scenarios, "Sorbent")

Direct_Energy_MC_df_product = mcs.monte_carlo_statistics_direct_energy(Total_Energy_per_kg_product_Dic, Adsorption_Performance_Dic, "Direct", scenarios, unit_label="per kg product", pei_mass_dic=PEI_results_Dic, support_mass_dic=Support_results_Dic)
Direct_Energy_MC_df_sorbent = mcs.monte_carlo_statistics_direct_energy(Total_Energy_per_kg_sorbent_Dic, Adsorption_Performance_Dic, "Direct", scenarios, unit_label="per kg sorbent")

# Life Cycle Impacts (Upstream + Direct)
print("\n🔄 Calculating Monte Carlo Statistics for Total Impacts dictionaries...")

Energy_LC_Impacts_MC_df = mcs.monte_carlo_statistics_energy_impacts(Energy_LC_Impacts_Dic, Adsorption_Performance_Dic, scenarios)
Material_LC_Impacts_MC_df = mcs.monte_carlo_statistics_material_impacts(Material_LC_Impacts_Dic, Adsorption_Performance_Dic, scenarios)
Total_LC_Impacts_MC_df = mcs.monte_carlo_statistics_total_impacts(Energy_LC_Impacts_Dic, Material_LC_Impacts_Dic, Adsorption_Performance_Dic, scenarios)

# Calculate Monte Carlo Statistics for adsorption performance dictionary
print("\n🔄 Calculating Monte Carlo Statistics for adsorption performance dictionary...")

Adsorption_Performance_MC_df = mcs.monte_carlo_statistics_adsorption_performance(Adsorption_Performance_Dic, mcs_number)
