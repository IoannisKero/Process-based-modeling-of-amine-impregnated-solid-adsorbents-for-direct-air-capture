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
import numpy as np

def adsorption_performance(mcs_number, use_fixed_CC=False, fixed_CC_value=None):
    """
    For each adsorbent, PEI loading, and dry or humid case, this function draws random plant operating
    hours, then computes how many adsorption cycles fit in the assumed sorbent lifetime,
    a decay rate so capacity fades over those cycles, and a lifetime cumulative CO2 captured value
    for every Monte Carlo sample.

    Normally that cumulative value comes from the built-in capacity table and the decay math in code.

    If you turn on fixed CC by passing use_fixed_CC True and a number for fixed_CC_value, the function
    skips the table and the decay formula for the cumulative CO2 value only. It still computes N and k
    the same way, but every sorbent, loading, and dry or humid branch gets that one number repeated for
    all Monte Carlo draws. If fixed CC is on but you omit the number, behavior stays the normal one.

    Parameters:
    - mcs_number (int): Number of Monte Carlo iterations
    - use_fixed_CC (bool): Whether to use a fixed CC value (default: False)
    - fixed_CC_value (float): Fixed CC value (default: None)

    Returns:
    - Performance_results (dict): Dictionary containing the performance results
    """

    # Adsorption capacity data
    CO2_adsorption_capacity ={
        "PEI_γ_alumina": {
            20: {"Dry": 0.60, "Humid": 0.87},
            30: {"Dry": 0.89, "Humid": 1.42},
            40: {"Dry": 1.26, "Humid": 2.00},
            50: {"Dry": 1.55, "Humid": 2.47},
            60: {"Dry": 1.55, "Humid": 2.47},
        },
        "MIL_101_Cr_PEI": {
            20: {"Dry": 0.24, "Humid": 0.31},
            30: {"Dry": 0.24, "Humid": 0.31},
            40: {"Dry": 1.13, "Humid": 1.47},
            50: {"Dry": 1.81, "Humid": 2.35},
            60: {"Dry": 1.81, "Humid": 2.35},
        },
        "MCM_41_PEI": {
            20: {"Dry": 0.22, "Humid": 0.39},
            30: {"Dry": 0.26, "Humid": 0.46},
            40: {"Dry": 0.31, "Humid": 0.55},
            50: {"Dry": 0.37, "Humid": 0.66},
            60: {"Dry": 0.29, "Humid": 0.52},
        },
        "PEI_SG": {
            20: {"Dry": 0.16, "Humid": 0.43},
            30: {"Dry": 0.39, "Humid": 1.04},
            40: {"Dry": 0.67, "Humid": 1.79},
            50: {"Dry": 0.81, "Humid": 2.16},
            60: {"Dry": 1.14, "Humid": 3.04},
        },
        "SBA_15_PEI": {
            20: {"Dry": 0.16, "Humid": 0.43},
            30: {"Dry": 0.39, "Humid": 1.04},
            40: {"Dry": 0.67, "Humid": 1.79},
            50: {"Dry": 0.81, "Humid": 2.16},
            60: {"Dry": 1.14, "Humid": 3.04},
        },
        "KIT6_PEI": {
            20: {"Dry": 0.16, "Humid": 0.43},
            30: {"Dry": 0.39, "Humid": 1.04},
            40: {"Dry": 0.67, "Humid": 1.79},
            50: {"Dry": 0.81, "Humid": 2.16},
            60: {"Dry": 1.14, "Humid": 3.04},
        },
        "HMS_PEI": {
            20: {"Dry": 0.16, "Humid": 0.43},
            30: {"Dry": 0.39, "Humid": 1.04},
            40: {"Dry": 0.67, "Humid": 1.79},
            50: {"Dry": 0.81, "Humid": 2.16},
            60: {"Dry": 1.14, "Humid": 3.04},
        },
    }

    # Parameters MCS

    uncertainty_parameters = {
        "plant_operation": {
            "distribution_type": "triangular",
            "pessimistic": 7300,  # h/year
            "reference": 8000,    # h/year
            "optimistic": 8700    # h/year
        },
        "cycle_time": {
            "distribution_type": "uniform", # Uniform distribution of cycle times for each sorbent (optional if range is given)
            "sorbents": {
                "PEI_γ_alumina": {"min": 14.0, "max": 14.0},    # h
                "MIL_101_Cr_PEI": {"min": 4.0, "max": 4.0},    # h
                "MCM_41_PEI": {"min": 3.5, "max": 3.5},        # h 
                "PEI_SG": {"min": 3.5, "max": 3.5},            # h 
                "SBA_15_PEI": {"min": 3.5, "max": 3.5},         # h
                "KIT6_PEI": {"min": 3.5, "max": 3.5},          # h 
                "HMS_PEI": {"min": 3.5, "max": 3.5},           # h 
            }
        }
    }

    # Sample plant operation time (triangular distribution)
    plant_operation = uncertainty_parameters["plant_operation"]
    U = np.random.triangular(
        left=plant_operation["pessimistic"],
        mode=plant_operation["reference"], 
        right=plant_operation["optimistic"],
        size=mcs_number
    )

    # Sample sorbent lifetime 
    # Adjust sorbent lifetime from 0.5 years (worst case) to 1 year (base case) to 2 years (best case) to obtain sensitivity analysis results
    s =  1 

    # Sample cycle times (uniform distribution for each sorbent)
    cycle_times = uncertainty_parameters["cycle_time"]["sorbents"]
    Cycle_Times = {}
    
    for sorbent_name, bounds in cycle_times.items():
        if bounds["min"] == bounds["max"]:
            # Constant value (no uncertainty)
            Cycle_Times[sorbent_name] = np.full(mcs_number, bounds["min"])
        else:
            # Uniform distribution
            Cycle_Times[sorbent_name] = np.random.uniform(
                low=bounds["min"],
                high=bounds["max"],
                size=mcs_number
            )

    Performance_results = {}

    for sorbent, loads in CO2_adsorption_capacity.items():
        tcycle = Cycle_Times[sorbent]  # hours
        # total number of cycles over lifetime (Eq S14)
        N = (U / tcycle) * s

        # per-cycle degradation rate so that end-of-life capacity is 0.5*C0
        k = np.log(2) / N

        # store N and k (same for all loadings/conditions for a given sorbent)
        Performance_results[sorbent] = {"N": N, "k": k, "loadings": {}}

        for loading, conds in loads.items():
            Performance_results[sorbent]["loadings"][loading] = {}

            for cond in ("Dry", "Humid"):
                if use_fixed_CC and fixed_CC_value is not None:
                    # Use fixed CC value directly (bypass C0 calculation)
                    CC = np.full(mcs_number, float(fixed_CC_value))
                else:
                    # Calculate CC from C0 (original method)
                    C0 = float(conds[cond])  # mmol/g
                    # Eq S15: cumulative captured CO2 over lifetime
                    CC = (C0 / k) * (1.0 - np.exp(-k * N)) * 0.044

                Performance_results[sorbent]["loadings"][loading][cond] = CC
    
    
    return Performance_results

# Test the function
if __name__ == "__main__":
    adsorption_performance(5)
