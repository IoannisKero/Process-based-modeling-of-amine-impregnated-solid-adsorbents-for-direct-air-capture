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
Project: Process-based lifecycle climate, energy, water and material constraints of solid adsorbent production for gigaton-scale direct air capture.
"""

import pandas as pd
import numpy as np

# ========================================================================================================================
# LCA IMPACT FACTORS DATABASE
# ========================================================================================================================

def environmental_impact_factors():
    """
    Returns a dictionary containing LCA impact factors for all inputs, retrieved from Ecoinvent v3.11 and LCIA per kg of input conducted by openLCA (v2.5).
    
    Impact Categories:
    - Climate Change (kg CO2-Eq) - TRACI (v2.1)
    - Fossil resource scarcity (kg oil-Eq) - ReCiPe 2016 (Hierarchical)
    - Water use (m3) - ReCiPe 2016 (Hierarchical)
    
    Returns:
        dict: Dictionary with structure {material_name: {impact_category: value}}
    """
    
    impact_factors = {
        # Energy Sources
        "US Grid - Average": {
            "unit": "1 kWh",
            "Climate Change (kg CO2-Eq)": 0.390, # US Grid - Average
            "Fossil resource scarcity (kg oil-Eq)": 0.14,
            "Water Use (m3)": 0.0026
        },
        "Heat": {
            "unit": "1 MJ",
            "Climate Change (kg CO2-Eq)": 0.059, # US Heat production average (matural gas)
            "Fossil resource scarcity (kg oil-Eq)": 0.0143,
            "Water Use (m3)": 0.0001
        },
        
        # Chemical Inputs - Chemicals

        # Precursors
        "Monoethanolamine": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 3.74,
            "Fossil resource scarcity (kg oil-Eq)": 1.51,
            "Water Use (m3)": 0.0442
        },
        "Sodium Silicate": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 0.85,
            "Fossil resource scarcity (kg oil-Eq)": 0.165,
            "Water Use (m3)": 0.0096
        },
        "Pseudo": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 1.57,
            "Fossil resource scarcity (kg oil-Eq)": 0.359,
            "Water Use (m3)": 0.0059
        },
        "TEOS": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 3.08,
            "Fossil resource scarcity (kg oil-Eq)": 1.40,
            "Water Use (m3)": 0.1162
        },
        "CrNO3": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 3.35,
            "Fossil resource scarcity (kg oil-Eq)": 0.574,
            "Water Use (m3)": 0.0352
        },
        "H2BDC": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 2.47,
            "Fossil resource scarcity (kg oil-Eq)": 1.34,
            "Water Use (m3)": 0.0125
        },
        "Silica": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 0.04,
            "Fossil resource scarcity (kg oil-Eq)": 0.0106,
            "Water Use (m3)": 0.0011
        },
        
        # Surfactants
        "Pluronic P123": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 5.44,
            "Fossil resource scarcity (kg oil-Eq)": 2.166,
            "Water Use (m3)": 0.0467
        },
        "CTMA-Br": { # Esterquat Surfactant (v3.11 Ecoinvent) is used as proxy
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 3.06,
            "Fossil resource scarcity (kg oil-Eq)": 0.735,
            "Water Use (m3)": 0.2157
        },
        "TMA": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 4.37,
            "Fossil resource scarcity (kg oil-Eq)": 1.914,
            "Water Use (m3)": 0.0278
        },
        
        # Solvents
        "Water": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 0.000312,
            "Fossil resource scarcity (kg oil-Eq)": 0.0000794,
            "Water Use (m3)": 0.000014
        },
        "Ethanol": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 2.17,
            "Fossil resource scarcity (kg oil-Eq)": 1.183,
            "Water Use (m3)": 0.0114
        },
        "Methanol": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 0.85,
            "Fossil resource scarcity (kg oil-Eq)": 0.856,
            "Water Use (m3)": 0.0023
        },
        "DMF": {  
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 3.83,
            "Fossil resource scarcity (kg oil-Eq)": 1.58,
            "Water Use (m3)": 0.0248
        },
        "N-Butanol": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 3.78,
            "Fossil resource scarcity (kg oil-Eq)": 1.752,
            "Water Use (m3)": 0.0112
        },
        
        # Acids, Bases, and Salts
        "Sulfuric Acid": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 0.16,
            "Fossil resource scarcity (kg oil-Eq)": 0.0449,
            "Water Use (m3)": 0.0132
        },
        "Sodium Hydroxide": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 1.40,
            "Fossil resource scarcity (kg oil-Eq)": 0.367,
            "Water Use (m3)": 0.0174
        },
        "Nitric Acid": {  
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 2.10,
            "Fossil resource scarcity (kg oil-Eq)": 0.248,
            "Water Use (m3)": 0.0161
        },
        "Sodium Oxide": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 6.46,
            "Fossil resource scarcity (kg oil-Eq)": 1.734,
            "Water Use (m3)": 0.0671
        },
        "Acetic Acid": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 2.56,
            "Fossil resource scarcity (kg oil-Eq)": 1.08,
            "Water Use (m3)": 0.0212
        },
        "Hydrochloric acid": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 0.86,
            "Fossil resource scarcity (kg oil-Eq)": 0.242,
            "Water Use (m3)": 0.0100
        },
        "Sodium Sulfate": {
            "unit": "1 kg",
            "Climate Change (kg CO2-Eq)": 0.695,
            "Fossil resource scarcity (kg oil-Eq)": 0.178,
            "Water Use (m3)": 0.020
        }
    }
    
    return impact_factors


# ========================================================================================================================
# ENERGY IMPACT ASSESSMENT 
# ========================================================================================================================

def energy_impacts(energy_dict, impact_factors):
    """
    Calculate environmental impacts from energy consumption by process, stage, and total.
    Keeps only:
    1) US grid average electricity impacts (all categories)
    2) Heat impacts (all categories, separate from electricity)

    Parameters:
        energy_dict (dict): {scenario: {sorbent: {energy_type: {stage: {process: matrix}}}}}
            - energy_type: "electricity" (kWh) or "heat" (MJ)
            - matrix: NumPy array (typically 3D: 1 x 1 x mcs_number)
        impact_factors (dict): Impact factors from environmental_impact_factors()

    Returns:
        dict: {
            scenario: {
                sorbent: {
                    "US_Grid_Average": {..., "Total": {category: matrix}},
                    "Heat": {..., "Total": {category: matrix}},
                    "Total": {category: matrix}  # electricity + heat
                }
            }
        }
    """
    impact_results = {}

    us_grid_factors = impact_factors.get("US Grid - Average", {})
    heat_factors = impact_factors.get("Heat", {})

    base_stage_order = ["Aziridine", "PEI", "Support", "Sorbent", "Recovery"]
    impact_categories = ["Climate Change (kg CO2-Eq)", "Fossil resource scarcity (kg oil-Eq)", "Water Use (m3)"]
    
    for scenario_id, scenario_data in energy_dict.items():
        impact_results[scenario_id] = {}
        
        for sorbent_type, sorbent_data in scenario_data.items():
            impact_results[scenario_id][sorbent_type] = {}
            
            us_grid_name = "US_Grid_Average"

            # Build stage list dynamically so extra stages (e.g., Recovery) are included.
            stages_found = set()
            if "electricity" in sorbent_data:
                stages_found.update(sorbent_data["electricity"].keys())
            if "heat" in sorbent_data:
                stages_found.update(sorbent_data["heat"].keys())
            stages = [s for s in base_stage_order if s in stages_found]
            for s in sorted(stages_found):
                if s not in stages:
                    stages.append(s)

            # Initialize US grid electricity structure
            impact_results[scenario_id][sorbent_type][us_grid_name] = {}
            for stage in stages:
                impact_results[scenario_id][sorbent_type][us_grid_name][stage] = {}
            impact_results[scenario_id][sorbent_type][us_grid_name]["Total"] = {
                category: None for category in impact_categories
            }

            # Initialize heat structure
            impact_results[scenario_id][sorbent_type]["Heat"] = {}
            for stage in stages:
                impact_results[scenario_id][sorbent_type]["Heat"][stage] = {}
            impact_results[scenario_id][sorbent_type]["Heat"]["Total"] = {
                category: None for category in impact_categories
            }
            impact_results[scenario_id][sorbent_type]["Total"] = {
                category: None for category in impact_categories
            }

            # Process electricity (kWh) with US grid factors
            if "electricity" in sorbent_data:
                electricity_factors_us = us_grid_factors

                for stage in sorbent_data["electricity"]:
                    if stage not in stages:
                        continue

                    impact_results[scenario_id][sorbent_type][us_grid_name][stage]["Stage_Total"] = {
                        category: None for category in impact_categories
                    }

                    for process in sorbent_data["electricity"][stage]:
                        # Stage_Total is already an aggregate; do not treat it as a regular process.
                        if process == "Stage_Total":
                            continue
                        energy_matrix = sorbent_data["electricity"][stage][process]

                        impact_results[scenario_id][sorbent_type][us_grid_name][stage][process] = {}

                        for category in impact_categories:
                            impact_value = electricity_factors_us.get(category, 0.0)
                            impact_matrix = energy_matrix * impact_value
                            impact_results[scenario_id][sorbent_type][us_grid_name][stage][process][category] = impact_matrix

                            if impact_results[scenario_id][sorbent_type][us_grid_name][stage]["Stage_Total"][category] is None:
                                impact_results[scenario_id][sorbent_type][us_grid_name][stage]["Stage_Total"][category] = impact_matrix.copy()
                            else:
                                if impact_matrix.shape == impact_results[scenario_id][sorbent_type][us_grid_name][stage]["Stage_Total"][category].shape:
                                    impact_results[scenario_id][sorbent_type][us_grid_name][stage]["Stage_Total"][category] += impact_matrix
                                else:
                                    impact_results[scenario_id][sorbent_type][us_grid_name][stage]["Stage_Total"][category] += np.broadcast_to(
                                        impact_matrix, impact_results[scenario_id][sorbent_type][us_grid_name][stage]["Stage_Total"][category].shape)

            # Process heat (MJ)
            if "heat" in sorbent_data:
                for stage in sorbent_data["heat"]:
                    if stage not in stages:
                        continue

                    impact_results[scenario_id][sorbent_type]["Heat"][stage]["Stage_Total"] = {
                        category: None for category in impact_categories
                    }

                    for process in sorbent_data["heat"][stage]:
                        # Stage_Total is already an aggregate; do not treat it as a regular process.
                        if process == "Stage_Total":
                            continue
                        energy_matrix = sorbent_data["heat"][stage][process]

                        impact_results[scenario_id][sorbent_type]["Heat"][stage][process] = {}

                        for category in impact_categories:
                            impact_value = heat_factors.get(category, 0.0)
                            impact_matrix = energy_matrix * impact_value
                            impact_results[scenario_id][sorbent_type]["Heat"][stage][process][category] = impact_matrix

                            if impact_results[scenario_id][sorbent_type]["Heat"][stage]["Stage_Total"][category] is None:
                                impact_results[scenario_id][sorbent_type]["Heat"][stage]["Stage_Total"][category] = impact_matrix.copy()
                            else:
                                if impact_matrix.shape == impact_results[scenario_id][sorbent_type]["Heat"][stage]["Stage_Total"][category].shape:
                                    impact_results[scenario_id][sorbent_type]["Heat"][stage]["Stage_Total"][category] += impact_matrix
                                else:
                                    impact_results[scenario_id][sorbent_type]["Heat"][stage]["Stage_Total"][category] += np.broadcast_to(
                                        impact_matrix, impact_results[scenario_id][sorbent_type]["Heat"][stage]["Stage_Total"][category].shape)

            # Totals: US grid electricity impacts
            # Prefer precomputed total matrix if available, otherwise sum stage totals.
            if "Total_Electricity_per_kg" in sorbent_data:
                total_electricity_matrix = sorbent_data["Total_Electricity_per_kg"]
                for category in impact_categories:
                    impact_value = us_grid_factors.get(category, 0.0)
                    impact_results[scenario_id][sorbent_type][us_grid_name]["Total"][category] = (
                        total_electricity_matrix * impact_value
                    )
            else:
                for stage in stages:
                    if "Stage_Total" in impact_results[scenario_id][sorbent_type][us_grid_name][stage]:
                        for category in impact_categories:
                            stage_total_matrix = impact_results[scenario_id][sorbent_type][us_grid_name][stage]["Stage_Total"][category]
                            if stage_total_matrix is not None:
                                if impact_results[scenario_id][sorbent_type][us_grid_name]["Total"][category] is None:
                                    impact_results[scenario_id][sorbent_type][us_grid_name]["Total"][category] = stage_total_matrix.copy()
                                else:
                                    if stage_total_matrix.shape == impact_results[scenario_id][sorbent_type][us_grid_name]["Total"][category].shape:
                                        impact_results[scenario_id][sorbent_type][us_grid_name]["Total"][category] += stage_total_matrix
                                    else:
                                        impact_results[scenario_id][sorbent_type][us_grid_name]["Total"][category] += np.broadcast_to(
                                            stage_total_matrix, impact_results[scenario_id][sorbent_type][us_grid_name]["Total"][category].shape)

            # Totals: heat impacts
            # Prefer precomputed total matrix if available, otherwise sum stage totals.
            if "Total_Heat_per_kg" in sorbent_data:
                total_heat_matrix = sorbent_data["Total_Heat_per_kg"]
                for category in impact_categories:
                    impact_value = heat_factors.get(category, 0.0)
                    impact_results[scenario_id][sorbent_type]["Heat"]["Total"][category] = (
                        total_heat_matrix * impact_value
                    )
            else:
                for stage in stages:
                    if "Stage_Total" in impact_results[scenario_id][sorbent_type]["Heat"][stage]:
                        for category in impact_categories:
                            stage_total_matrix = impact_results[scenario_id][sorbent_type]["Heat"][stage]["Stage_Total"][category]
                            if stage_total_matrix is not None:
                                if impact_results[scenario_id][sorbent_type]["Heat"]["Total"][category] is None:
                                    impact_results[scenario_id][sorbent_type]["Heat"]["Total"][category] = stage_total_matrix.copy()
                                else:
                                    if stage_total_matrix.shape == impact_results[scenario_id][sorbent_type]["Heat"]["Total"][category].shape:
                                        impact_results[scenario_id][sorbent_type]["Heat"]["Total"][category] += stage_total_matrix
                                    else:
                                        impact_results[scenario_id][sorbent_type]["Heat"]["Total"][category] += np.broadcast_to(
                                            stage_total_matrix, impact_results[scenario_id][sorbent_type]["Heat"]["Total"][category].shape)

            # Combined total impacts = electricity total + heat total
            for category in impact_categories:
                elec_total = impact_results[scenario_id][sorbent_type][us_grid_name]["Total"].get(category)
                heat_total = impact_results[scenario_id][sorbent_type]["Heat"]["Total"].get(category)
                if elec_total is None and heat_total is None:
                    continue
                if elec_total is None:
                    impact_results[scenario_id][sorbent_type]["Total"][category] = heat_total.copy()
                elif heat_total is None:
                    impact_results[scenario_id][sorbent_type]["Total"][category] = elec_total.copy()
                else:
                    if elec_total.shape == heat_total.shape:
                        impact_results[scenario_id][sorbent_type]["Total"][category] = elec_total + heat_total
                    else:
                        impact_results[scenario_id][sorbent_type]["Total"][category] = elec_total + np.broadcast_to(
                            heat_total, elec_total.shape
                        )
    
    return impact_results

# ========================================================================================================================
# CHEMICAL IMPACT ASSESSMENT 
# ========================================================================================================================

# Chemical Impact Factors Database

def chemical_impact_factors():

    # Environmental Impact Factors
    impact_factors = environmental_impact_factors()
    
    # Support Synthesis
    Support_Impacts = {
        'PEI_SG': {'DI Water': impact_factors['Water'], 
                   'Sodium Silicate': impact_factors['Sodium Silicate'],  
                   'Water Washing': impact_factors['Water'],
                   'Water Sulfuric Acid':impact_factors['Water'],
                   'Sulphuric Acid': impact_factors['Sulfuric Acid'],
                   'Sodium Sulfate': impact_factors['Sodium Sulfate']},
        'MCM_41_PEI': {'DI Water': impact_factors['Water'], 
                   'Sodium Oxide': impact_factors['Sodium Oxide'],  
                   'Silica': impact_factors['Silica'], 
                   'Water Washing': impact_factors['Water'], 
                   'TMA': impact_factors['TMA'],
                   'CTMA-Br': impact_factors['CTMA-Br']},
        'PEI_γ_alumina': {'DI Water': impact_factors['Water'], 
                          'Pluronic P123': impact_factors['Pluronic P123'],  
                          'Pseudo': impact_factors['Pseudo'], 
                          'Ethanol Washing': impact_factors['Ethanol'], 
                          'Nitric Acid': impact_factors['Nitric Acid'],
                          'Water Nitric Acid': impact_factors['Water']},
        'MIL_101_Cr_PEI': {'DI Water': impact_factors['Water'],
                           'CrNO3': impact_factors['CrNO3'], 
                           'H2BDC': impact_factors['H2BDC'], 
                           'Acetic Acid': impact_factors['Acetic Acid'],
                           'Methanol Washing': impact_factors['Methanol'],
                           'DMF Washing': impact_factors['DMF'],
                           'Water Acetic Acid': impact_factors['Water']},
        'SBA_15_PEI': {'HCl': impact_factors['Hydrochloric acid'] , 
                    'Pluronic P123': impact_factors['Pluronic P123'] , 
                    'TEOS': impact_factors['TEOS'],
                    'DI Water': impact_factors['Water'],
                    'Water Washing': impact_factors['Water'],
                    'Water HCl': impact_factors['Water'],
                    'Ethanol Byproduct': impact_factors['Ethanol']},
        'KIT6_PEI': {'DI Water': impact_factors['Water'], 
                    'HCl': impact_factors['Hydrochloric acid'], 
                    'Pluronic P123': impact_factors['Pluronic P123'] , 
                    'N-Butanol': impact_factors['N-Butanol'] , 
                    'TEOS': impact_factors['TEOS'],
                    'Water HCl': impact_factors['Water'],
                    'Ethanol Byproduct': impact_factors['Ethanol']},
        'HMS_PEI': {'DI Water': impact_factors['Water'] , 
                    'HCl': impact_factors['Hydrochloric acid'] , 
                    'Pluronic P123': impact_factors['Pluronic P123'] , 
                    'TEOS': impact_factors['TEOS'] ,
                    'Water Washing': impact_factors['Water'],
                    'Water HCl': impact_factors['Water'],
                    'Ethanol Byproduct': impact_factors['Ethanol']},        
        }
    
    
    # PEI Synthesis
    PEI_Impacts = {
                    'Monoethanolamine': impact_factors['Monoethanolamine'],
                    'Ethanol': impact_factors['Ethanol'],
                    'Ethanol Washing': impact_factors['Ethanol'],
                    'Water Sulfuric Acid': impact_factors['Water'],
                    'Water Ethanol': impact_factors['Water'],
                    'Water NaOH': impact_factors['Water'],
                    "DI Water": impact_factors['Water'],
                    'Sulfuric Acid': impact_factors['Sulfuric Acid'],
                    'Sodium Hydroxide': impact_factors['Sodium Hydroxide'],
                    'Sodium Hydroxide Washing': impact_factors['Sodium Hydroxide'],
                    'Sodium Sulfate': impact_factors['Sodium Sulfate'],
        }
    
    # Sorbent Synthesis 
    Sorbent_Impacts = {
            "PEI_γ_alumina": {"Methanol": impact_factors['Methanol']},
            "PEI_SG": {"Methanol": impact_factors['Methanol']},
            "MIL_101_Cr_PEI": {"Methanol": impact_factors['Methanol']},
            "MCM_41_PEI":  {"Methanol": impact_factors['Methanol']},
            "SBA_15_PEI":  {"Methanol": impact_factors['Methanol']},
            "KIT6_PEI":  {"Methanol": impact_factors['Methanol']},
            "HMS_PEI":  {"Ethanol": impact_factors['Ethanol']},
        }
    

    return Support_Impacts, PEI_Impacts, Sorbent_Impacts


def chemical_impacts(
    Aziridine_mass_input_dic,
    PEI_mass_input_dic,
    Support_mass_input_dic,
    Sorbent_mass_input_dic,
    mass_key="Input Mass (kg)"
):
    """
    Multiply mass input dictionaries by their corresponding chemical impact factors.
    
    Rules:
    - Aziridine and PEI stages use PEI_Impacts dictionary
    - Support stage uses Support_Impacts dictionary (sorbent-specific)
    - Sorbent stage uses Sorbent_Impacts dictionary (sorbent-specific)
    - Components without impact factors are skipped (intermediate products)
    
    Parameters:
        Aziridine_mass_input_dic, PEI_mass_input_dic, Support_mass_input_dic, Sorbent_mass_input_dic:
            Structured dicts from material_mass_input
        mass_key (str): Key to use for normalization, default "Input Mass (kg)"
    
    Returns:
        dict: {scenario: {sorbent: {stage: {component: {impact_category: matrix}, "Stage_Total": {...}}, "Total": {...}}}}
    """
    impact_categories = [
        "Climate Change (kg CO2-Eq)",
        "Fossil resource scarcity (kg oil-Eq)",
        "Water Use (m3)",
    ]

    Support_Impacts, PEI_Impacts, Sorbent_Impacts = chemical_impact_factors()
    
    # Define solvent keywords for identification
    solvent_keywords = [
        "ethanol", "methanol", "dmf", "butanol"
    ]
    # Recovered byproducts credited as avoided burden (replace virgin raw materials)
    sodium_sulfate_names = {"sodium sulfate", "sodium sulphate"}
    ethanol_byproduct_name = "ethanol byproduct"

    def compute_stage_impacts(stage_dic, impact_map, stage_name, sorbent_specific=False):
        stage_results = {}
        missing_components = set()
        missing_sorbents = set()

        for scenario_id, sorbents in stage_dic.items():
            stage_results.setdefault(scenario_id, {})
            for sorbent_type, stages in sorbents.items():
                stage_results[scenario_id].setdefault(sorbent_type, {})
                components = stages.get(stage_name, {})

                if sorbent_specific:
                    if sorbent_type not in impact_map:
                        missing_sorbents.add(sorbent_type)
                        continue
                    sorbent_impacts = impact_map[sorbent_type]
                else:
                    sorbent_impacts = impact_map

                stage_impacts = {}
                stage_total = {cat: None for cat in impact_categories}
                
                # Determine number of MCS iterations and generate recovery percentages once per scenario/sorbent/stage
                # (to ensure consistency across all solvents in the same MCS iteration)
                num_mcs_iterations = None
                recovery_percentages = None
                for component, mcs_map in components.items():
                    if not isinstance(mcs_map, dict):
                        continue
                    mcs_indices = sorted([k for k in mcs_map.keys() if isinstance(k, int)])
                    if mcs_indices:
                        num_mcs_iterations = len(mcs_indices)
                        # Triangular distribution for recovery: min=0.95, mode=0.97, max=0.99
                        recovery_percentages = np.random.triangular(0.95, 0.97, 0.99, num_mcs_iterations)
                        break

                for component, mcs_map in components.items():
                    if component not in sorbent_impacts:
                        missing_components.add(component)
                        continue
                    if not isinstance(mcs_map, dict):
                        continue

                    mcs_indices = sorted([k for k in mcs_map.keys() if isinstance(k, int)])
                    if not mcs_indices:
                        continue

                    values = []
                    for k in mcs_indices:
                        entry = mcs_map.get(k, {})
                        values.append(entry.get(mass_key, np.nan))
                    values = np.asarray(values, dtype=float)
                    
                    # Check if component is a solvent
                    component_lower = component.lower()
                    component_norm = component_lower.replace("_", " ").replace("-", " ")
                    component_tokens = set(component_norm.split())
                    is_solvent = any(keyword in component_lower for keyword in solvent_keywords)
                    # Credit only explicit sodium sulfate (not sodium hydroxide/oxide/other sodium compounds).
                    is_sodium_sulfate = (
                        component_norm in sodium_sulfate_names
                        or (
                            "sodium" in component_tokens
                            and ("sulfate" in component_tokens or "sulphate" in component_tokens)
                            and "hydroxide" not in component_tokens
                            and "oxide" not in component_tokens
                        )
                    )
                    # Credit only explicit ethanol byproduct (never plain ethanol or methanol).
                    is_ethanol_byproduct = (
                        component_norm == ethanol_byproduct_name
                        or (
                            "ethanol" in component_tokens
                            and "byproduct" in component_tokens
                            and "methanol" not in component_tokens
                        )
                    )
                    # Exclude Monoethanolamine from solvents
                    if "monoethanolamine" in component_lower:
                        is_solvent = False
                    
                    # Apply solvent recovery percentage if component is a solvent
                    # Use the same recovery percentages for all solvents in this scenario/sorbent/stage
                    if is_solvent and recovery_percentages is not None:
                        # Ensure recovery_percentages length matches values length
                        if len(recovery_percentages) == len(values):
                            # Calculate loss percentage (1 - recovery) and apply to mass
                            loss_percentages = 1.0 - recovery_percentages
                            values = values * loss_percentages

                    comp_factors = sorbent_impacts[component]
                    stage_impacts[component] = {}

                    for category in impact_categories:
                        impact_value = comp_factors.get(category, 0.0)
                        impact_matrix = values.reshape(1, 1, -1) * impact_value
                        # Credit recovered byproducts as avoided burden.
                        if is_sodium_sulfate or is_ethanol_byproduct:
                            impact_matrix = -impact_matrix
                        stage_impacts[component][category] = impact_matrix

                        if stage_total[category] is None:
                            stage_total[category] = impact_matrix.copy()
                        else:
                            if impact_matrix.shape == stage_total[category].shape:
                                stage_total[category] += impact_matrix
                            else:
                                stage_total[category] += np.broadcast_to(impact_matrix, stage_total[category].shape)

                stage_impacts["Stage_Total"] = stage_total
                stage_results[scenario_id][sorbent_type][stage_name] = stage_impacts

        return stage_results, missing_components, missing_sorbents

    # Compute stage-specific impacts
    aziridine_results, az_missing_comp, _ = compute_stage_impacts(
        Aziridine_mass_input_dic, PEI_Impacts, "Aziridine", sorbent_specific=False
    )
    pei_results, pei_missing_comp, _ = compute_stage_impacts(
        PEI_mass_input_dic, PEI_Impacts, "PEI", sorbent_specific=False
    )
    support_results, sup_missing_comp, sup_missing_sorb = compute_stage_impacts(
        Support_mass_input_dic, Support_Impacts, "Support", sorbent_specific=True
    )
    sorbent_results, sorb_missing_comp, sorb_missing_sorb = compute_stage_impacts(
        Sorbent_mass_input_dic, Sorbent_Impacts, "Sorbent", sorbent_specific=True
    )

    # Combine all stages into one dictionary
    combined = {}
    for stage_results in [aziridine_results, pei_results, support_results, sorbent_results]:
        for scenario_id, sorbents in stage_results.items():
            combined.setdefault(scenario_id, {})
            for sorbent_type, stages in sorbents.items():
                combined.setdefault(scenario_id, {}).setdefault(sorbent_type, {})
                for stage_name, stage_data in stages.items():
                    combined[scenario_id][sorbent_type][stage_name] = stage_data

    # Add totals across stages and expose stage totals at sorbent level
    for scenario_id, sorbents in combined.items():
        for sorbent_type, stages in sorbents.items():
            total_impacts = {cat: None for cat in impact_categories}
            stage_totals = {}
            for stage_name, stage_data in stages.items():
                stage_total = stage_data.get("Stage_Total", {})
                if stage_total:
                    stage_totals[stage_name] = stage_total
                for category in impact_categories:
                    stage_matrix = stage_total.get(category)
                    if stage_matrix is not None:
                        if total_impacts[category] is None:
                            total_impacts[category] = stage_matrix.copy()
                        else:
                            if stage_matrix.shape == total_impacts[category].shape:
                                total_impacts[category] += stage_matrix
                            else:
                                total_impacts[category] += np.broadcast_to(stage_matrix, total_impacts[category].shape)
            combined[scenario_id][sorbent_type]["Stage_Totals"] = stage_totals
            combined[scenario_id][sorbent_type]["Total"] = total_impacts

    missing_components = az_missing_comp | pei_missing_comp | sup_missing_comp | sorb_missing_comp
    missing_sorbents = sup_missing_sorb | sorb_missing_sorb

    return combined
