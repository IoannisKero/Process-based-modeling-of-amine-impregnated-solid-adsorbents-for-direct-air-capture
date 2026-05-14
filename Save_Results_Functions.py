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

# Import Libraries

import pandas as pd
import numpy as np

# ========================================================================================================================
# STEP 0: SAVE ENERGY RESUTLS IN DATAFRAME
# ========================================================================================================================

def energy_save_to_df(Material_Energy, Material_Energy_per_kg, mcs_number):
    """
    Converts nested energy result dictionaries (sorbent-level & scenario-level) into clean DataFrames.
    Now includes MCS iteration column for Monte Carlo simulation results.

    Parameters:
    - Material_Energy (dict): {scenario: {sorbent_type: {process: energy_kWh}}}
    - Material_Energy_per_kg (dict): {scenario: {sorbent_type: {process: energy_kWh_per_kg}}}
    - mcs_number (int): Number of Monte Carlo iterations (default: 10)

    Returns:
    - Material_Energy_Df (pd.DataFrame): Total energy (kWh) per process, scenario, sorbent and MCS iteration
    - Material_Energy_per_kg_Df (pd.DataFrame): Energy per kg (kWh/kg) per process, scenario, sorbent and MCS iteration
    """

    Material_Energy_Consumption = []
    Material_Energy_Cons_per_kg = []

    def _to_mcs_vector(x):
        """
        Normalize scalars/arrays to a 1D vector over MCS iterations.
        Accepts scalar, 1D, 2D, or 3D inputs.
        """
        arr = np.asarray(x, dtype=float).squeeze()
        if arr.ndim == 0:
            return np.repeat(float(arr), mcs_number)
        if arr.ndim == 1:
            return arr
        return arr.reshape(-1)

    # ===================== TOTAL ENERGY (kWh) =====================
    for scenario, sorbent_dict in Material_Energy.items():
        for sorbent_type, process_dict in sorbent_dict.items():
            for process, energy_matrix in process_dict.items():
                energy_vec = _to_mcs_vector(energy_matrix)
                for mcs_iter in range(min(mcs_number, energy_vec.shape[0])):
                    val = float(energy_vec[mcs_iter])
                    if process != "Washing" and abs(val) == 0:
                        continue

                    Material_Energy_Consumption.append({
                        "Scenario": scenario,
                        "Sorbent": sorbent_type,
                        "Process": process,
                        "MCS_Iteration": mcs_iter,
                        "Energy (kWh)": round(val, 6)
                    })

    # ===================== ENERGY PER KG (kWh/kg) =====================
    for scenario, sorbent_dict in Material_Energy_per_kg.items():
        for sorbent_type, process_dict in sorbent_dict.items():
            for process, energy_matrix in process_dict.items():
                energy_vec = _to_mcs_vector(energy_matrix)
                for mcs_iter in range(min(mcs_number, energy_vec.shape[0])):
                    val = float(energy_vec[mcs_iter])
                    if process != 'Washing' and abs(val) == 0:
                        continue

                    Material_Energy_Cons_per_kg.append({
                        "Scenario": scenario,
                        "Sorbent": sorbent_type,
                        "Process": process,
                        "MCS_Iteration": mcs_iter,
                        "Energy (kWh/kg)": round(val, 6)
                        })
                    
    # ===================== TO DATAFRAMES =====================
    Material_Energy_Df = pd.DataFrame(Material_Energy_Consumption)
    Material_Energy_per_kg_Df = pd.DataFrame(Material_Energy_Cons_per_kg)

    return Material_Energy_Df, Material_Energy_per_kg_Df



