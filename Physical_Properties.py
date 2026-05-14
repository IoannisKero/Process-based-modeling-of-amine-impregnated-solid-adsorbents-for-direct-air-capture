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

# ========================================================================================================================
# STEP 0: EQUIPMENT EFFICIENCY
# ========================================================================================================================

def efficiency():
    """Return lab-scale and industrial-scale equipment efficiencies [%] by unit operation."""

    # Lab scale
    
    n_heat = 70
    n_stir = 90
    n_dry = 80
    n_pump = 80 
    
    # Industrial scale
    
    n_heat_ind = 80
    n_stir_ind = 90
    n_dry_ind = 80
    n_pump_ind = 80 
    
    Efficiency_Lab = {
        'Activation': n_heat,
        'Stirring': n_stir,
        'Sonication': n_stir,
        'Extraction': n_heat,
        'Aging': n_heat,
        'Evaporation': n_dry, 
        'Distillation': n_heat,
        'Filtration': n_pump,
        'Centrifugation': n_pump,
        'Grinding':  n_stir,
        'Washing': n_pump,
        'Drying': n_dry,
        'Calcination': n_heat,
        'Heating': n_heat,
        }
    
    
    Efficiency_Industrial = {
        'Activation': n_heat_ind,
        'Stirring': n_stir_ind,
        'Sonication': n_stir_ind,
        'Extraction': n_heat_ind,
        'Aging': n_heat_ind,
        'Evaporation': n_dry_ind,
        'Distillation': n_heat_ind,
        'Filtration': n_pump_ind,
        'Centrifugation': n_pump_ind,
        'Grinding':  n_stir_ind,
        'Washing': n_pump_ind,
        'Drying': n_dry_ind,
        'Calcination': n_heat_ind,
        'Heating': n_heat_ind
        }
    
    return Efficiency_Lab, Efficiency_Industrial

# ========================================================================================================================
# STEP 1: EQUIPMENT POWER CONSUMPTION 
# ========================================================================================================================

def power_consumption():
    """Return lab-scale and industrial-scale equipment power draws [W] where applicable."""

    # Lab scale
    
    P_sonic = 250
    P_extrac = 220 
    P_filt = 180 
    P_wash = 180 
    P_centr = 190 
    P_grind = 290 
    P_dry = 400 
    
    # Industrial scale
    
    P_filt_ind = 3300 
    P_dry_ind = 2200 

    Power_Consumption_Lab = {
        'Sonication': P_sonic,
        'Filtration': P_filt,
        'Washing': P_wash,
        'Centrifugation': P_centr,
        'Grinding':  P_grind,
        'Drying': P_dry
        }
    
    
    Power_Consumption_Industrial = {
        'Filtration': P_filt_ind,
        'Centrifugation': P_filt_ind,
        'Washing': P_filt_ind,
        'Drying': P_dry_ind
        }
    

    return Power_Consumption_Lab, Power_Consumption_Industrial

# ========================================================================================================================
# STEP 2: STIRRER PARAMETERS
# ========================================================================================================================

def modeling_parameters(mixture_volume):
    """
    Vectorized stirring / reactor scaling from mixture volume.

    Parameters:
        mixture_volume: Volume [L], shape (1, n_steps, mcs_number).

    Returns:
        N: Rotational speed [1/s].
        D: Impeller diameter [m].
        R: Number of parallel reactors (ceil when volume > 10_000 L).
        Q_loss: Heat loss fraction [%] used in energy submodels.
    """
    
    # Preserve zero/negative volumes (no stirring / no heat-loss term)
    zero_mask = mixture_volume <= 0
    
    # Avoid log(0) inside correlations
    mixture_volume = np.where(zero_mask, 1e-10, mixture_volume)

    # Rotational Speed (N)
    N = np.where(mixture_volume < 100, 3.052, np.where(mixture_volume > 10000, 0.658, - 0.506 * np.log(mixture_volume) + 5.1366))

    # Impeller Diameter (m)
    D = np.where(mixture_volume < 100, 0.173, np.where(mixture_volume > 10000, 0.803, 0.1373 * np.log(mixture_volume) - 0.5167))

    # Number of reactors (scale-up beyond 10_000 L nominal)
    R = np.where(mixture_volume > 10000, np.ceil(mixture_volume / 10000).astype(int), 1)

    # Heat loss [%]
    Q_loss = np.where(mixture_volume < 100, 1.22, np.where(mixture_volume > 10000, 0.26, - 0.203 * np.log(mixture_volume) + 2.0524))
    
    # Zero out outputs where volume was invalid
    N = np.where(zero_mask, 0, N)
    D = np.where(zero_mask, 0, D)
    R = np.where(zero_mask, 0, R)
    Q_loss = np.where(zero_mask, 0, Q_loss)

    return N, D, R, Q_loss
    
# ========================================================================================================================
# STEP 3: PROCESS TIME FUNCTION (t)
# ========================================================================================================================

def process_time(mcs_number):
    """Per-sorbent process duration [h] per step; shape (1, n_steps, mcs_number)."""

    n = 36  # Must match Mass_Function / Chemical_Properties step layout

    # PEI_γ_alumina

    PEI_γ_alumina_Time = np.zeros((1, n, mcs_number))
    PEI_γ_alumina_Time[:, np.r_[0:12, 16, 18, 20, 24, 26:28, 29:36], :] = np.array([
        0.5, 1, 1/30, 0.5, 0.5, 0.5, 2.5, 8, 0.5, 24, 1, 24, 1.5, 17, 24, 48, 24, 4, 8, 0.25, 0.33, 8, 1, 8, 0]).reshape(1, -1, 1)
        
    # PEI_SG
    
    PEI_SG_Time = np.zeros((1, n, mcs_number))
    PEI_SG_Time[:, np.r_[0:12, 12, 17:19, 25:27, 28, 30, 32, 34:36], :] = np.array([
        0.5, 1, 1/30, 0.5, 0.5, 0.5, 2.5, 8, 0.5, 24, 1, 24, 0.5, 0.5, 8, 0.5, 15, 12, 1, 5, 12, 0]).reshape(1, -1, 1)

    # MIL_101_Cr_PEI
    
    MIL_101_Cr_PEI_Time = np.zeros((1, n, mcs_number))
    MIL_101_Cr_PEI_Time[:, np.r_[0:12, 12, 18, 23, 25:27, 29:36], :] = np.array([
        0.5, 1, 1/30, 0.5, 0.5, 0.5, 2.5, 8, 0.5, 24, 1, 24, 1, 12, 0.5, 1, 8, 24, 1/12, 0.33, 24, 1, 24, 0]).reshape(1, -1, 1)

    # MCM_41_PEI
    
    MCM_41_PEI_Time = np.zeros((1, n, mcs_number))
    MCM_41_PEI_Time[:, np.r_[0:12, 12, 18, 21, 25:28, 30, 32, 34:36], :] = np.array([
        0.5, 1, 1/30, 0.5, 0.5, 0.5, 2.5, 8, 0.5, 24, 1, 24, 1.05, 24, 0.5, 0.5, 8, 5, 0.25, 0.5, 16, 0]).reshape(1, -1, 1)

    # SBA_15_PEI
    
    SBA_15_PEI_Time = np.zeros((1, n, mcs_number))
    SBA_15_PEI_Time[:, np.r_[0:12, 12, 18, 21, 25:28, 30, 32, 34:36], :] = np.array([
        0.5, 1, 1/30, 0.5, 0.5, 0.5, 2.5, 8, 0.5, 24, 1, 24, 20, 24, 0.5, 0.5, 8, 8, 0.5, 4, 8, 0]).reshape(1, -1, 1)

    # KIT6_PEI
    
    KIT6_PEI_Time = np.zeros((1, n, mcs_number))
    KIT6_PEI_Time[:, np.r_[0:12, 12:15, 18, 21, 26:29, 30, 32, 34:36], :] = np.array([
        0.5, 1, 1/30, 0.5, 0.5, 0.5, 2.5, 8, 0.5, 24, 1, 24, 0.5, 24, 24, 24, 0.5, 24, 5, 12, 0.5, 6, 12, 0]).reshape(1, -1, 1)

    # HMS_PEI
    
    HMS_PEI_Time = np.zeros((1, n, mcs_number))
    HMS_PEI_Time[:, np.r_[0:12, 12:15, 18, 21, 25:28, 30, 32, 34:36], :] = np.array([
        0.5, 1, 1/30, 0.5, 0.5, 0.5, 2.5, 8, 0.5, 24, 1, 24, 0.5, 4.5, 24, 24, 0.5, 0.5, 8, 4, 0.5, 6, 12, 0]).reshape(1, -1, 1)


    Process_Time = {
        "PEI_γ_alumina": PEI_γ_alumina_Time,
        "PEI_SG": PEI_SG_Time,
        "MIL_101_Cr_PEI":  MIL_101_Cr_PEI_Time,
        "MCM_41_PEI":  MCM_41_PEI_Time,
        "SBA_15_PEI":  SBA_15_PEI_Time,
        "KIT6_PEI":  KIT6_PEI_Time,
        "HMS_PEI":  HMS_PEI_Time
    }


    return Process_Time
