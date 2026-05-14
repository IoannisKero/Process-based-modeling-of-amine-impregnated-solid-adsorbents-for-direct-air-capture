# Copyright (c) 2026 Ioannis Keroglou
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
import pandas as pd
import CoolProp.CoolProp as CP

# ========================================================================================================================
# STEP 0: ENERGY CONSUMPTION DURING DRYING/EVAPORATION
# ========================================================================================================================

def dry_evap(cp, T, n_dry_lab ,n_dry_ind , P_lab, P_ind, DH_vap, M_solvents, t, scenario_data, M_sorbent, Volume, x, mcs_number):
    
    """
    Computes the energy consumption for drying and evaporation in both lab and industrial scales.

    Parameters:
        - cp (numpy.ndarray): Mixture heat capacity [J/(g·°C)] per step.
        - T (numpy.ndarray): Process temperature [°C] per step.
        - n_dry_lab (float): Lab dryer efficiency [%].
        - n_dry_ind (float): Industrial dryer efficiency [%].
        - P_lab (float): Lab dryer power [W] for isothermal (T ≤ 25 °C) drying.
        - P_ind (float): Industrial dryer power [W] for isothermal drying.
        - DH_vap (numpy.ndarray): Mixture enthalpy of vaporization [J/g] per step.
        - M_solvents (numpy.ndarray): Mass to heat / evaporate [g] per step.
        - t (numpy.ndarray): Duration [h] per step.
        - scenario_data (dict): Scenario profile; uses ``profile['scale']`` (``'lab'`` vs industrial).
        - M_sorbent (numpy.ndarray or float): Sorbent mass [g] for kWh/kg normalization (broadcast as provided).
        - Volume (numpy.ndarray): Working volume [L] per step.
        - x (int or sequence): Step indices where this drying or evaporation model applies.
        - mcs_number (int): Monte Carlo third-axis length.

    Returns:
        - E_dry (numpy.ndarray): Total drying energy [kWh], shape ``(1, 1, mcs_number)``.
        - E_dry_kg (numpy.ndarray): Total per kg sorbent [kWh/kg], shape ``(1, 1, mcs_number)``.
        - E_kWh_dry_matrix (numpy.ndarray): Step-wise energy [kWh], shape ``(1, 36, mcs_number)``.
        - E_per_kg_dry_matrix (numpy.ndarray): Step-wise specific energy [kWh/kg sorbent], shape ``(1, 36, mcs_number)``.
    """
    
    # Define the chemical steps that drying is present
    
    Drying_steps = np.zeros((1,36, mcs_number))
    Drying_steps[:, x, :] = 1 
    
    # Parameters

    sorbent_steps = np.where(M_solvents != 0, 1, 0) 
    Cp_solv = cp
    Q_heat = Cp_solv * M_solvents * (T - 25)  * Drying_steps * sorbent_steps
    T_final = T * Drying_steps * sorbent_steps
    t_dry = t * Drying_steps * sorbent_steps
    Q_vap = DH_vap *  M_solvents * Drying_steps * sorbent_steps
    Volume = Volume * Drying_steps * sorbent_steps
    P_freezer_lab = 85 # Power consumption of freezer used in aziridine synthesis drying
    P_lab_matrix = np.zeros((1,36, mcs_number))
    P_lab_matrix[:, :, :] = P_lab * sorbent_steps
    P_lab_matrix[:, 7, :] = P_freezer_lab
    P_freezer_ind = 560 # Power consumption of freezer used in aziridine synthesis drying
    P_ind_matrix = np.zeros((1,36, mcs_number))
    P_ind_matrix[:, :, :] = P_ind * sorbent_steps
    P_ind_matrix[:, 7, :] = P_freezer_ind
    
    # Energy Equation for Lab & Industrial Scale

    scale = scenario_data["profile"]["scale"]   
    if scale == "lab": # Lab Scale
        R = np.where(Volume > 20,np.ceil(Volume / 20).astype(int),1)  # Literature-based (25 L)
        # Special case for step 7 (freezer)
        R[:, 7, :] = np.where(Volume[:, 7, :] > 50, np.ceil(Volume[:, 7, :] / 50).astype(int), 1) # For volume more than 50 L, extra freezer is needed
        E = np.where(T_final > 25, (Q_heat + Q_vap) / (n_dry_lab / 100), ((P_lab_matrix * R * t_dry) / (n_dry_lab / 100)) * 3600)
            
    else: # Industrial Scale
        R = np.where(Volume > 4000,np.ceil(Volume / 4000).astype(int),1)  # Literature-based Assumption (4000 L)
        # Special case for step 7 (freezer)
        R[:, 7, :] = np.where(Volume[:, 7, :] > 1300, np.ceil(Volume[:, 7, :] / 1300).astype(int), 1) # For volume more than 1300 L, extra freezer is needed
        E = np.where(T_final > 25, (Q_heat + Q_vap) / (n_dry_ind / 100),((P_ind_matrix * R * t_dry) / (n_dry_ind / 100)) * 3600)

    #Convert Joules to kWh
    E_kWh_dry_matrix = E  / 3.6e6

    # Energy consumption per kg of sorbent
    E_per_kg_dry_matrix = np.divide(E_kWh_dry_matrix, M_sorbent/1000, out=np.zeros_like(M_solvents), where=M_solvents > 0)
    
    # Conversion of matrix to scalar value
    E_dry = np.sum(E_kWh_dry_matrix, axis=1, keepdims=True)
    E_dry_kg = E_dry/(M_sorbent/1000)

    return E_dry, E_dry_kg, E_kWh_dry_matrix, E_per_kg_dry_matrix


# ========================================================================================================================
# STEP 1: ENERGY CONSUMPTION DURING HEATING
# ========================================================================================================================

def heating(M, cp, T, n_heat_lab, n_heat_ind, scenario_data, M_sorbent, x, mcs_number, Q_loss):
    """
    Computes the energy consumption for the heating process in both lab-scale and industrial-scale operations.

    Parameters:
        - M (numpy.ndarray): Mixture mass [g], shape ``(1, 36, mcs_number)``.
        - cp (numpy.ndarray): Heat capacity [J/(g·°C)], same shape as ``M``.
        - T (numpy.ndarray): Final temperature [°C], same shape as ``M``.
        - n_heat_lab (float): Lab heating efficiency [%].
        - n_heat_ind (float): Industrial heating efficiency [%].
        - scenario_data (dict): ``profile['scale']`` selects lab vs industrial.
        - M_sorbent (numpy.ndarray or float): Sorbent mass [g] for per-kg normalization.
        - x (list or numpy.ndarray): Step indices where heating is present.
        - mcs_number (int): Monte Carlo third-axis length (must match ``M``).
        - Q_loss (numpy.ndarray): Heat-loss fraction [%] per step (divided by 100 inside the function).

    Returns:
        - E_heat (numpy.ndarray): Total heating energy [kWh], shape ``(1, 1, mcs_number)``.
        - E_heat_kg (numpy.ndarray): Total per kg sorbent [kWh/kg], shape ``(1, 1, mcs_number)``.
        - E_kWh_heat_matrix (numpy.ndarray): Step-wise energy [kWh], shape ``(1, 36, mcs_number)``.
        - E_per_kg_heat_matrix (numpy.ndarray): Step-wise [kWh/kg sorbent], shape ``(1, 36, mcs_number)``.

    Applied to processes:
        - Heating
        - Activation
        - Aging
        - Polymerization
        - Calcination
    """
    
    # Define the chemical steps where heating is present
    heating_steps = np.zeros((1, 36, mcs_number))
    heating_steps[:, x, :] = 1  
    
    # Parameters
    M_mix = M
    sorbent_steps = np.where(M != 0, 1, 0) 
    Cp_mix = cp
    T_final = T
    Q_heat = Cp_mix * M_mix * (T_final - 25) * heating_steps * sorbent_steps
    Q_loss = Q_loss/100
    
    # Energy Equation for Lab & Industrial Scale
    # Automatically detect scale from scenario data
    scale = scenario_data["profile"]["scale"]
    if scale == "lab":  # Lab Scale
        E = Q_heat * (1 + Q_loss) / (n_heat_lab / 100)
    else:  # Industrial Scale
        E = Q_heat * (1 + Q_loss) / (n_heat_ind / 100)

    # Convert Joules to kWh
    E_kWh_heat_matrix = E / 3.6e6

    # Energy consumption per kg of sorbent
    E_per_kg_heat_matrix = np.divide(
        E_kWh_heat_matrix, M_sorbent / 1000, 
        out=np.zeros_like(M), where=M > 0
    )
    
    # Conversion of matrix to scalar value
    E_heat = np.sum(E_kWh_heat_matrix, axis=1, keepdims=True)
    E_heat_kg = E_heat / (M_sorbent / 1000)

    return E_heat, E_heat_kg, E_kWh_heat_matrix, E_per_kg_heat_matrix

# ========================================================================================================================
# STEP 2: ENERGY CONSUMPTION DURING DISTILLATION
# ========================================================================================================================

def distillation(M, cp, T, n_dis_lab ,n_dis_ind , DH_vap, M_solvents, scenario_data, M_sorbent, mcs_number, Q_loss):
    
    """
    Computes the energy consumption for the distillation process in both lab-scale and industrial-scale operations.

    Parameters:
        - M (numpy.ndarray): Mixture mass [g], shape ``(1, 36, mcs_number)``.
        - cp (numpy.ndarray): Heat capacity [J/(g·°C)], same shape as ``M``.
        - T (numpy.ndarray): Temperature [°C], same shape as ``M``.
        - n_dis_lab (float): Lab distillation / still efficiency [%].
        - n_dis_ind (float): Industrial distillation efficiency [%].
        - DH_vap (numpy.ndarray): Enthalpy of vaporization [J/g] for the solvent portion, same shape as ``M``.
        - M_solvents (numpy.ndarray): Solvent mass [g] used in the vaporization term, same shape as ``M``.
        - scenario_data (dict): ``profile['scale']`` selects ``n_dis_lab`` vs ``n_dis_ind``.
        - M_sorbent (numpy.ndarray or float): Sorbent mass [g] for per-kg normalization.
        - mcs_number (int): Monte Carlo third-axis length.
        - Q_loss (numpy.ndarray): Present for API consistency; the function replaces it with a fixed internal heat-loss fraction for this correlation.

    Returns:
        - E_dis (numpy.ndarray): Total distillation energy [kWh], shape ``(1, 1, mcs_number)``.
        - E_dis_kg (numpy.ndarray): Total per kg sorbent [kWh/kg], shape ``(1, 1, mcs_number)``.
        - E_kWh_dis_matrix (numpy.ndarray): Step-wise energy [kWh], shape ``(1, 36, mcs_number)`` (step 6 active by default).
        - E_per_kg_dis_matrix (numpy.ndarray): Step-wise [kWh/kg sorbent], shape ``(1, 36, mcs_number)``.
    """
    
    # Define the chemical steps that distillation is present
    
    Distillation_steps = np.zeros((1,36, mcs_number))
    Distillation_steps[:, 6, :] = 1
    
    # Parameters
    
    M_mix = M
    sorbent_steps = np.where(M != 0, 1, 0) 
    Cp_mix = cp
    T_final = T 
    Q_heat = Cp_mix * M_mix * (T_final - 25) * Distillation_steps * sorbent_steps
    Q_vap = DH_vap *  M_solvents * Distillation_steps * sorbent_steps
    R_min = 0.6 # Literature-based Assumption
    Q_loss = 0.05 # Heat loss %
    
    # Energy Equation for Lab & Industrial Scale
    # Automatically detect scale from scenario data
    scale = scenario_data["profile"]["scale"]
    if scale == "lab": # Lab Scale
        E = (Q_heat + Q_vap * (1.2 * R_min + 1)) * (1 + Q_loss) / (n_dis_lab /100)

    else: # Industrial Scale
        E = (Q_heat + Q_vap * (1.2 * R_min + 1)) * (1 + Q_loss) / (n_dis_ind /100)

    #Convert Joules to kWh
    E_kWh_dis_matrix = E  / 3.6e6

    # Energy consumption per kg of sorbent
    E_per_kg_dis_matrix = np.divide(E_kWh_dis_matrix, M_sorbent/1000, out=np.zeros_like(M), where=M > 0)
    
    # Conversion of matrix to scalar value
    E_dis = np.sum(E_kWh_dis_matrix, axis=1, keepdims=True)
    E_dis_kg = E_dis/(M_sorbent/1000)

    return E_dis, E_dis_kg, E_kWh_dis_matrix, E_per_kg_dis_matrix


# ========================================================================================================================
# STEP 3: ENERGY CONSUMPTION DURING STIRRING
# ========================================================================================================================

def stirring(M, cp, T, t, d, N, D, R, n_stir_lab, n_stir_ind, n_heat_lab, n_heat_ind, scenario_data, mixture_volume, M_sorbent, mcs_number, Q_loss):

    """
    Computes the energy consumption during stirring for both lab-scale and industrial-scale operations.

    Parameters:
        - M (numpy.ndarray): Mixture mass [g], shape ``(1, 36, mcs_number)``.
        - cp (numpy.ndarray): Heat capacity [J/(g·°C)], same shape as ``M``.
        - T (numpy.ndarray): Temperature [°C], same shape as ``M``.
        - t (numpy.ndarray): Stirring time [h], same shape as ``M``.
        - d (numpy.ndarray): Mixture density field (same tensor as upstream ``density_sum``), same shape as ``M``.
        - N (numpy.ndarray): Agitator speed [1/s], same shape as ``M``.
        - D (numpy.ndarray): Impeller diameter [m], same shape as ``M``.
        - R (numpy.ndarray): Number of reactors / parallel units, same shape as ``M``.
        - n_stir_lab (float): Lab stirrer efficiency [%].
        - n_stir_ind (float): Industrial stirrer efficiency [%].
        - n_heat_lab (float): Lab heating efficiency [%] when heating is added with stirring.
        - n_heat_ind (float): Industrial heating efficiency [%] when heating is added with stirring.
        - scenario_data (dict): ``profile['scale']`` selects lab vs industrial correlation.
        - mixture_volume (numpy.ndarray): Volume [L], same shape as ``M`` (lab uses it for small-stirrer count).
        - M_sorbent (numpy.ndarray or float): Sorbent mass [g] for per-kg normalization.
        - mcs_number (int): Monte Carlo third-axis length.
        - Q_loss (numpy.ndarray): Heat-loss fraction [%] on the heating add-on (divided by 100 inside the function).

    Returns:
        - E_stir (numpy.ndarray): Total stirring energy [kWh], shape ``(1, 1, mcs_number)``.
        - E_stir_kg (numpy.ndarray): Total per kg sorbent [kWh/kg], shape ``(1, 1, mcs_number)``.
        - E_kWh_stir_matrix (numpy.ndarray): Step-wise energy [kWh], shape ``(1, 36, mcs_number)``.
        - E_per_kg_stir_matrix (numpy.ndarray): Step-wise [kWh/kg sorbent], shape ``(1, 36, mcs_number)``.
    """
    
    # Define the chemical steps that stirring is present
    
    Stirring_steps = np.zeros((1,36, mcs_number))
    Stirring_steps[:, np.r_[0, 5, 8, 12:16, 20, 30, 32], :] = 1
    
    # Parameters 
    
    M_mix = M
    sorbent_steps = np.where(M != 0, 1, 0) 
    Cp_mix = cp
    d_mix = d
    T_final = T * Stirring_steps * sorbent_steps
    Q_heat = Cp_mix * M_mix * (T - 25) * Stirring_steps * sorbent_steps
    t_stir = t * Stirring_steps * sorbent_steps
    Np = 0.79 # Literature-based Assumption
    Q_loss = Q_loss/100
    
    # Apply stirring parameters only to stirring steps
    N_stir = N * Stirring_steps
    D_stir = D * Stirring_steps
    R_stir = R * Stirring_steps
    R_lab = np.where(mixture_volume > 100,np.ceil(mixture_volume / 100).astype(int),1)  # Number of stirrers for lab scale

    # Energy Equation for Lab & Industrial Scale
    # Automatically detect scale from scenario data
    scale = scenario_data["profile"]["scale"]
    
    if scale == "lab":  # Lab Scale
        E = np.where(
        T_final > 25,
        (Np * d_mix * (0.658**3) * R_lab * (0.173**5) * t_stir) / (n_stir_lab / 100) + Q_heat * (1 + Q_loss) / (n_heat_lab/ 100),
        (Np * d_mix * (0.658**3) * R_lab * (0.173**5) * t_stir) / (n_stir_lab / 100))
    else:  # Industrial Scale
        E = np.where(
        T_final > 25,
        (Np * d_mix * (N_stir**3) * R_stir * (D_stir**5) * t_stir) / (n_stir_ind / 100) + Q_heat * (1 + Q_loss) / (n_heat_ind / 100),
        (Np * d_mix * (N_stir**3) * R_stir * (D_stir**5) * t_stir) / (n_stir_ind / 100))

    #Convert Joules to kWh
    E_kWh_stir_matrix = E  / 3.6e6

    # Energy consumption per kg of sorbent
    E_per_kg_stir_matrix = np.divide(E_kWh_stir_matrix, M_sorbent/1000, out=np.zeros_like(M), where=M > 0)
    
    # Conversion of matrix to scalar value
    E_stir = np.sum(E_kWh_stir_matrix, axis=1, keepdims=True)
    E_stir_kg = E_stir/(M_sorbent/1000)

    return E_stir, E_stir_kg, E_kWh_stir_matrix, E_per_kg_stir_matrix


# ========================================================================================================================
# STEP 4: ENERGY CONSUMPTION DURING SOLID-LIQUID SEPARATION PROCESSES
# ========================================================================================================================

def sol_liq_sep(M, n_process_lab ,n_process_ind , P_lab, P_ind, t, x, scenario_data, M_solvents, M_sorbent, Mixture_Mass, Volume, mcs_number):
    """
    Computes the energy consumption for solid–liquid separation (vacuum filtration, centrifugation, washing)
    in lab and industrial scales. Lab uses power, time, and unit count R(Volume); industrial uses kWh/ton-style factors.

    Parameters:
        - M (numpy.ndarray): Mixture mass [g], shape ``(1, 36, mcs_number)`` (gates active steps).
        - n_process_lab (float): Lab process efficiency [%].
        - n_process_ind (float): Industrial efficiency [%] (not used in the current industrial mass-factor branch).
        - P_lab (float): Lab equipment power [W].
        - P_ind (float): Industrial equipment power [W] (not used in the current industrial branch).
        - t (numpy.ndarray): Process duration [h], shape ``(1, 36, mcs_number)``.
        - x (int or numpy.ndarray): Step index(ices); ``23`` = centrifugation; ``3, 4, 21, 25`` = filtration / washing.
        - scenario_data (dict): ``profile['scale']`` selects lab vs industrial.
        - M_solvents (numpy.ndarray): Accepted for API symmetry; not used in the current function body.
        - M_sorbent (numpy.ndarray or float): Sorbent mass [g] for per-kg normalization.
        - Mixture_Mass (numpy.ndarray): Mixture mass [g], shape ``(1, 36, mcs_number)`` (industrial correlations).
        - Volume (numpy.ndarray): Volume [L], shape ``(1, 36, mcs_number)``.
        - mcs_number (int): Monte Carlo third-axis length.

    Returns:
        - E_proc (numpy.ndarray): Total process energy summed over steps, shape ``(1, 1, mcs_number)`` (same units as ``E``).
        - E_proc_kg (numpy.ndarray): Total per kg sorbent, shape ``(1, 1, mcs_number)``.
        - E (numpy.ndarray): Step-wise energy matrix, shape ``(1, 36, mcs_number)`` (lab: from P, t, R; industrial: kWh/ton × mass).
        - E_per_kg_proc_matrix (numpy.ndarray): ``E`` divided by sorbent mass [kg] where ``M > 0``, shape ``(1, 36, mcs_number)``.
    """

    # Define the chemical steps that the process is present
    
    process_steps = np.zeros((1,36, mcs_number))
    process_steps[:, x, :] = 1 
    
    # Parameters
    
    sorbent_steps = np.where(M != 0, 1, 0) 
    t_process = t * process_steps * sorbent_steps
    Volume = Volume * process_steps * sorbent_steps
    Mixture_Mass = Mixture_Mass * sorbent_steps
    process_active_steps = process_steps * sorbent_steps

    # Energy Equation for Lab & Industrial Scale
    # Automatically detect scale from scenario data
    scale = scenario_data["profile"]["scale"]
    
    if scale == "lab": # Lab Scale
    
        if x == 23:
            R = np.where(Volume > 0.5,np.ceil(Volume / 0.5).astype(int),1)  # Number of centrifuges
        elif x in (3,4,21,25):
            R = np.where(Volume > 5,np.ceil(Volume / 5).astype(int),1)  # Number of filtration/washing devices

        E = (R * (P_lab * t_process)/ (n_process_lab/100)) /1000

    else: # Industrial Scale

        if x == 23: # Centrifugation
            Mixture_Mass_centrifugation = Mixture_Mass[:,[25],:]
            # Keep centrifugation only when its x-step is active in sorbent_steps.
            E = 10 * (Mixture_Mass_centrifugation * process_active_steps) * 10**-6 # 10 kWh/ton product assumed

        elif x in (3,4,21,25): # Filtration/Washing
            # Use the first non-zero mixture mass after step x (per MCS iteration).
            Mixture_Mass_filtration_washing = np.zeros_like(Mixture_Mass)
            x_indices = np.atleast_1d(x).astype(int)
            for x_idx in x_indices:
                for k in range(Mixture_Mass.shape[2]):
                    # If filtration/washing step itself is inactive, do not assign energy.
                    if Mixture_Mass[0, x_idx, k] <= 0:
                        continue
                    next_vals = Mixture_Mass[0, x_idx + 1 :, k]
                    nz_rel = np.flatnonzero(next_vals > 0)
                    if nz_rel.size > 0:
                        Mixture_Mass_filtration_washing[0, x_idx, k] = next_vals[nz_rel[0]]

            E = 10 * Mixture_Mass_filtration_washing * 10**-6 # 10 kWh/ton product assumed
        
     
    # Energy consumption per kg of sorbent
    E_per_kg_proc_matrix = np.divide(E, M_sorbent/1000, out=np.zeros_like(M), where=M > 0)
    
    # Conversion of matrix to scalar value
    E_proc = np.sum(E, axis=1, keepdims=True)
    E_proc_kg = E_proc/(M_sorbent/1000)

    return E_proc, E_proc_kg, E, E_per_kg_proc_matrix

# ========================================================================================================================
# STEP 6: ENERGY CONSUMPTION DURING SONICATION
# ========================================================================================================================

def sonication(M, cp, T, t, d, N, D, R, n_sonic_lab, n_sonic_ind, P_lab, scenario_data, M_sorbent, Volume, mcs_number):
    """
    Computes the energy consumption during sonication for both lab-scale and industrial-scale operations.

    Parameters:
        - M (numpy.ndarray): Mixture mass [g], shape ``(1, 36, mcs_number)`` (gates active steps).
        - cp (numpy.ndarray): Heat capacity (same shape as ``M``); present for API symmetry with ``stirring``, not used in the current body.
        - T (numpy.ndarray): Temperature (same shape as ``M``); present for API symmetry, not used in the current body.
        - t (numpy.ndarray): Sonication time [h], shape ``(1, 36, mcs_number)``.
        - d (numpy.ndarray): Mixture density tensor, same shape as ``M``.
        - N (numpy.ndarray): Agitator / sonic parameter [1/s], same shape as ``M``.
        - D (numpy.ndarray): Characteristic diameter [m], same shape as ``M``.
        - R (numpy.ndarray): Parallel units, same shape as ``M``.
        - n_sonic_lab (float): Lab sonication efficiency [%].
        - n_sonic_ind (float): Industrial sonication efficiency [%].
        - P_lab (float): Lab sonicator electrical power [W].
        - scenario_data (dict): ``profile['scale']`` selects lab vs industrial.
        - M_sorbent (numpy.ndarray or float): Sorbent mass [g] for per-kg normalization.
        - Volume (numpy.ndarray): Volume [L], shape ``(1, 36, mcs_number)``.
        - mcs_number (int): Monte Carlo third-axis length.

    Returns:
        - E_sonic (numpy.ndarray): Total sonication energy [kWh], shape ``(1, 1, mcs_number)``.
        - E_sonic_kg (numpy.ndarray): Total per kg sorbent [kWh/kg], shape ``(1, 1, mcs_number)``.
        - E_kWh_sonic_matrix (numpy.ndarray): Step-wise energy [kWh], shape ``(1, 36, mcs_number)``.
        - E_per_kg_sonic_matrix (numpy.ndarray): Step-wise [kWh/kg sorbent], shape ``(1, 36, mcs_number)``.
    """
    
    # Define the chemical steps where sonication is present
    Sonication_steps = np.zeros((1, 36, mcs_number))
    Sonication_steps[:, np.r_[16, 31], :] = 1
    
    # Parameters 
    sorbent_steps = np.where(M != 0, 1, 0) 
    d_mix = d
    t_sonic = t * Sonication_steps * sorbent_steps
    Np = 0.79  # Literature-based Assumption
    Volume = Volume * Sonication_steps * sorbent_steps
   
    # Energy Equation for Lab & Industrial Scale
    # Automatically detect scale from scenario data
    scale = scenario_data["profile"]["scale"]
    if scale == "lab":  # Lab Scale
        R = np.where(Volume > 0.5, np.ceil(Volume / 0.5).astype(int), 1)  # Number of sonication reactors
        E = (R * (P_lab * t_sonic) / (n_sonic_lab / 100)) * 3600
        
    else:  # Industrial Scale
        R = np.where(Volume > 10000, np.ceil(Volume / 10000).astype(int), 1)  # Number of sonication reactors
        E = (Np * d_mix * (N**3) * R * (D**5) * t_sonic) / (n_sonic_ind / 100)

    # Convert Joules to kWh
    E_kWh_sonic_matrix = E / 3.6e6

    # Energy consumption per kg of sorbent
    E_per_kg_sonic_matrix = np.divide(E_kWh_sonic_matrix, M_sorbent / 1000, out=np.zeros_like(M), where=M > 0)
    
    # Conversion of matrix to scalar value
    E_sonic = np.sum(E_kWh_sonic_matrix, axis=1, keepdims=True)
    E_sonic_kg = E_sonic / (M_sorbent / 1000)

    return E_sonic, E_sonic_kg, E_kWh_sonic_matrix, E_per_kg_sonic_matrix

# ========================================================================================================================
# STEP 7: ENERGY CONSUMPTION DURING GRINDING
# ========================================================================================================================

def grinding(M, n_process_lab, P_lab, t, scenario_data, M_sorbent, Volume, Mixture_Mass, M_solvents, mcs_number):
    
    """
    Computes the energy consumption for grinding in both lab-scale and industrial-scale operations.

    Parameters:
        - M (numpy.ndarray): Mixture mass [g] per step, shape ``(1, 36, mcs_number)``.
        - n_process_lab (float): Lab grinding efficiency [%].
        - P_lab (float): Lab equipment power [W].
        - t (numpy.ndarray): Process duration [h] per step, shape ``(1, 36, mcs_number)``.
        - scenario_data (dict): Scenario profile; uses ``profile['scale']`` (``'lab'`` vs industrial).
        - M_sorbent (numpy.ndarray or float): Sorbent mass [g] for per-kg normalization.
        - Volume (numpy.ndarray): Mixture volume [L] per step, shape ``(1, 36, mcs_number)``.
        - Mixture_Mass (numpy.ndarray): Mixture mass [g] on active grinding steps (industrial branch uses ``15 * Mixture_Mass * 1e-6`` kWh from g-scale mass).
        - M_solvents (numpy.ndarray): Solvent mass tensor (same shape as ``M``); masked with active steps in the function body.
        - mcs_number (int): Monte Carlo third-axis length.

    Returns:
        - E_proc (numpy.ndarray): Total grinding energy [kWh], shape ``(1, 1, mcs_number)``.
        - E_proc_kg (numpy.ndarray): Total per kg sorbent [kWh/kg], shape ``(1, 1, mcs_number)``.
        - E (numpy.ndarray): Step-wise energy [kWh], shape ``(1, 36, mcs_number)``.
        - E_per_kg_proc_matrix (numpy.ndarray): Step-wise [kWh/kg sorbent], shape ``(1, 36, mcs_number)``.

    Notes:
        - Lab scale: energy from equipment power, operation time, parallel units R(Volume), and lab efficiency.
        - Industrial scale: fixed factor 15 kWh per metric ton of mixture mass on the active step (mass in g).
    """

    # Define the chemical steps that the process is present
    process_steps = np.zeros((1,36, mcs_number))
    process_steps[:, 2, :] = 1
    
    # Parameters
    
    sorbent_steps = np.where(M != 0, 1, 0) 
    t_process = t * process_steps * sorbent_steps
    Volume = Volume * process_steps * sorbent_steps
    Mixture_Mass = M * process_steps * sorbent_steps
    M_solvents = M_solvents * process_steps * sorbent_steps

    # Energy Equation for Lab & Industrial Scale
    # Automatically detect scale from scenario data
    scale = scenario_data["profile"]["scale"]
    
    if scale == "lab": # Lab Scale
        R = np.where(Volume > 0.7,np.ceil(Volume / 0.7).astype(int),1)  # Number of reactors
        E = (R * (P_lab * t_process)/ (n_process_lab/100))/1000
            
    else: # Industrial Scale
        E = 15 * (Mixture_Mass) * 10**-6 # 15 kWh/ton product assumed

    # Energy consumption per kg of sorbent
    E_per_kg_proc_matrix = np.divide(E, M_sorbent/1000, out=np.zeros_like(M), where=M > 0)
    
    # Conversion of matrix to scalar value
    E_proc = np.sum(E, axis=1, keepdims=True)
    E_proc_kg = E_proc/(M_sorbent/1000)

    return E_proc, E_proc_kg, E, E_per_kg_proc_matrix

# ========================================================================================================================
# STEP 8: Solvent recovery
# ========================================================================================================================

def solvent_recovery(M_solvent_recovered, M_mixture_loss, n_lab, n_ind, T, scenario_data, M_sorbent, mcs_number):
    """
    Computes solvent recovery energy (liquid distillation + condenser; vapor condensation).

    Parameters:
        - M_solvent_recovered (dict): Per fluid among Ethanol, Methanol, DMF, n-Butanol: values are dicts with
          ``mass_kg`` and ``phase`` arrays of shape ``(1, steps, mcs_number)``; ``phase`` uses the strings Liquid and Vapor.
        - M_mixture_loss (dict or ndarray): Mixture loss co-shaped with recovered mass (see ``build_mass_matrix`` in function body).
        - n_lab (float): Reserved for lab-side efficiency (not used in the current loop).
        - n_ind (float or ndarray): Industrial heating efficiency [%]; broadcast to ``(1, 1, mcs_number)`` as ``eta_heat``.
        - T (float or ndarray): Temperature field for liquid heating and vapor sub-cooling (see ``build_T_matrix``).
        - scenario_data (dict): Accepted for API consistency with other energy functions (not referenced in the current body).
        - M_sorbent (float or ndarray): Sorbent mass [g] for per-kg outputs.
        - mcs_number (int): Monte Carlo third-axis length.

    Returns:
        - E_distillation (numpy.ndarray): Heat-side recovery energy [kWh], shape ``(1, 1, mcs_number)``.
        - E_condensation (numpy.ndarray): Electricity for condensation [kWh], shape ``(1, 1, mcs_number)``.
        - E_distillation_kg (numpy.ndarray): Heat per kg sorbent [kWh/kg], shape ``(1, 1, mcs_number)``.
        - E_condensation_kg (numpy.ndarray): Electricity per kg sorbent [kWh/kg], shape ``(1, 1, mcs_number)``.
    """

    P_cond = 101325  # Pa
    n = np.random.triangular(0.95, 0.97, 0.99, mcs_number)  # recovery fraction

    # outputs stored first in kJ
    E_distillation = np.zeros((1, 1, mcs_number))   # heat
    E_condensation = np.zeros((1, 1, mcs_number))   # electricity

    Fluid_List = ["Ethanol", "Methanol", "DMF", "n-Butanol"]

    R_min = 0.6
    Q_loss = 0.05
    COP = 3.0

    # heating efficiency
    n_ind_arr = np.asarray(n_ind, dtype=float)
    if n_ind_arr.ndim == 0:
        n_ind_arr = np.full(mcs_number, float(n_ind_arr))
    eta_heat = (n_ind_arr / 100.0).reshape(1, 1, -1)

    # make T compatible with (1, steps, mcs)
    def build_T_matrix(T, target_shape):
        if np.isscalar(T):
            return np.full(target_shape, float(T))
        T_arr = np.asarray(T, dtype=float)
        if T_arr.ndim == 1:   # (steps,)
            return T_arr.reshape(1, -1, 1) + np.zeros(target_shape)
        if T_arr.shape == target_shape:
            return T_arr
        raise ValueError(f"T shape {T_arr.shape} is incompatible with mass shape {target_shape}")

    def build_mass_matrix(x, target_shape, name="matrix"):
        """
        Accept either:
          - ndarray/scalar compatible with target_shape, or
          - dict with key 'mass_kg' that stores the ndarray.
        """
        if isinstance(x, dict):
            x = x.get("mass_kg", x)
        if np.isscalar(x):
            return np.full(target_shape, float(x))
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2:  # (1,steps) -> broadcast over mcs
            arr = arr[:, :, None] + np.zeros(target_shape)
        if arr.ndim == 1:  # (steps,) -> broadcast
            arr = arr.reshape(1, -1, 1) + np.zeros(target_shape)
        if arr.shape == target_shape:
            return arr
        raise ValueError(f"{name} shape {arr.shape} is incompatible with target shape {target_shape}")

    for Fluid in Fluid_List:
        if Fluid not in M_solvent_recovered:
            continue

        mass = np.asarray(M_solvent_recovered[Fluid]["mass_kg"], dtype=float)   # (1,steps,mcs)
        phase = np.asarray(M_solvent_recovered[Fluid]["phase"], dtype=object)

        if mass.shape != phase.shape:
            raise ValueError(f"{Fluid}: mass_kg and phase shapes do not match")

        T_matrix = build_T_matrix(T, mass.shape)
        mixture_matrix = build_mass_matrix(M_mixture_loss, mass.shape, name="M_mixture_loss")

        # solvent properties
        if Fluid == "Ethanol":
            Cp_liquid = 2.44
            T_saturation = 78.5
            T_sat_K = T_saturation + 273.15
            Cp_vapor = CP.PropsSI('C', 'T', T_sat_K, 'P', P_cond, Fluid) / 1000.0  # kJ/kg/K
            h_vap_sat = CP.PropsSI('H', 'P', P_cond, 'Q', 1, Fluid)
            h_liq_sat = CP.PropsSI('H', 'P', P_cond, 'Q', 0, Fluid)
            dH_vap = (h_vap_sat - h_liq_sat) / 1000.0  # kJ/kg

        elif Fluid == "Methanol":
            Cp_liquid = 2.53
            T_saturation = 64.7
            T_sat_K = T_saturation + 273.15
            Cp_vapor = CP.PropsSI('C', 'T', T_sat_K, 'P', P_cond, Fluid) / 1000.0  # kJ/kg/K
            h_vap_sat = CP.PropsSI('H', 'P', P_cond, 'Q', 1, Fluid)
            h_liq_sat = CP.PropsSI('H', 'P', P_cond, 'Q', 0, Fluid)
            dH_vap = (h_vap_sat - h_liq_sat) / 1000.0  # kJ/kg

        elif Fluid == "DMF":
            Cp_liquid = 2.00
            Cp_vapor = 1.78
            T_saturation = 153.0
            dH_vap = 650.0

        elif Fluid == "n-Butanol":
            Cp_liquid = 2.39
            Cp_vapor = 1.86
            T_saturation = 117.7
            dH_vap = 621.0

        else:
            continue

        # recovered mass
        n_matrix = n.reshape(1, 1, -1)
        recovered_mass = mass * n_matrix

        liquid_mask = (phase == "Liquid")
        vapor_mask = (phase == "Vapor")

        M_liquid = np.where(liquid_mask, recovered_mass, 0.0)
        # Consider mixture loss ONLY at steps/MCS where liquid solvent loss is present.
        active_liquid_mask = liquid_mask & (M_liquid > 0)
        M_liquid_mixture = np.where(active_liquid_mask, mixture_matrix, 0.0)
        M_vapor = np.where(vapor_mask, recovered_mass, 0.0)

        # -------------------------
        # 1. Liquid solvent recovery
        # heat for distillation + electricity for condenser

        if np.any(M_liquid > 0):
            Q_heat = Cp_liquid * M_liquid * np.maximum(T_saturation - 25.0, 0.0)  # kJ
            Q_heat_mixture = Cp_liquid * M_liquid_mixture * np.maximum(T_saturation - 25.0, 0.0)  # kJ
            Q_vap_liquid = M_liquid * dH_vap                                        # kJ

            # Distillation heating uses mixture liquid mass up to Tsat.
            # Vaporization enthalpy remains solvent-only.
            Q_dist_kJ = ((Q_heat_mixture + Q_vap_liquid * (1.2 * R_min + 1.0)) * (1.0 + Q_loss)) / eta_heat
            Q_cond_liquid_kJ = Q_vap_liquid / COP

            E_distillation += np.sum(Q_dist_kJ, axis=1, keepdims=True)
            E_condensation += np.sum(Q_cond_liquid_kJ, axis=1, keepdims=True)

        # -------------------------
        # 2. Vapor solvent recovery
        # condenser only
        # -------------------------
        if np.any(M_vapor > 0):
            deltaT_vapor = np.maximum(T_matrix - T_saturation, 0.0)
            Q_cond_vapor_kJ = M_vapor * (dH_vap + Cp_vapor * deltaT_vapor)
            E_condensation += np.sum(Q_cond_vapor_kJ / COP, axis=1, keepdims=True)

    # -------------------------
    # Convert kJ -> kWh
    # -------------------------
    E_distillation = E_distillation / 1000
    E_condensation = E_condensation / 3600.0

    # -------------------------
    # Normalize by sorbent mass
    # -------------------------
    M_sorbent_kg = np.asarray(M_sorbent, dtype=float) / 1000.0
    E_distillation_kg = E_distillation / M_sorbent_kg
    E_condensation_kg = E_condensation / M_sorbent_kg

    return E_distillation, E_condensation, E_distillation_kg, E_condensation_kg
    
# ========================================================================================================================
# STEP 9: TOTAL ENERGY CONSUMPTION PER PROCESS (AZIRIDINE, PEI, SUPPORT, SORBENT SYNTHESIS)
# ========================================================================================================================

def calculate_total_energy(dfs, df_names):
    """
    Calculate total energy per sorbent per scenario across multiple energy DataFrames.
    Now includes MCS iterations for Monte Carlo simulation results.

    Parameters:
        dfs (list): List of DataFrames with columns ["Scenario", "Sorbent", "Process", "Energy (kWh/kg)", "MCS_Iteration"]
        df_names (list): List of names corresponding to each DataFrame

    Returns:
        pd.DataFrame: Combined DataFrame with total energy per scenario, sorbent, and MCS iteration, including source name.
    """
    combined_results = []

    for df, name in zip(dfs, df_names):
        # Check if MCS_Iteration column exists
        if 'MCS_Iteration' in df.columns:
            # Group by Scenario, Sorbent, and MCS_Iteration
            total_energy = df.groupby(["Scenario", "Sorbent", "MCS_Iteration"])["Energy (kWh/kg)"].sum().reset_index()
            total_energy["Process"] = name
            combined_results.append(total_energy)
        else:
            # Fallback to original behavior for DataFrames without MCS_Iteration
            total_energy = df.groupby(["Scenario", "Sorbent"])["Energy (kWh/kg)"].sum().reset_index()
            total_energy["Process"] = name
            combined_results.append(total_energy)

    total_energy_df = pd.concat(combined_results, ignore_index=True)
    
    total_energy_df["Energy (kWh/kg)"] = total_energy_df["Energy (kWh/kg)"].round(0)

    # Select columns based on whether MCS_Iteration exists
    if 'MCS_Iteration' in total_energy_df.columns:
        summary_df = total_energy_df[["Scenario", "Sorbent", "MCS_Iteration", "Process", "Energy (kWh/kg)"]]
        # Sort by Sorbent, Scenario, and MCS_Iteration
        summary_df = summary_df.sort_values(by=["Sorbent", "Scenario", "MCS_Iteration"]).reset_index(drop=True)
    else:
        summary_df = total_energy_df[["Scenario", "Sorbent", "Process", "Energy (kWh/kg)"]]
        # Sort by Sorbent and then Scenario
        summary_df = summary_df.sort_values(by=["Sorbent", "Scenario"]).reset_index(drop=True)

    return summary_df



def combine_energy_per_kg_dictionaries(aziridine_energy_per_kg, pei_energy_per_kg, support_energy_per_kg, sorbent_energy_per_kg, process_names):
    """
    Combine multiple energy per kg dictionaries into a single total energy per kg dictionary.
    Sums all 3D matrices for each sorbent type to get total energy per kg.

    Parameters:
        aziridine_energy_per_kg (dict): {scenario: {sorbent: {process: energy_per_kg_3d_matrix}}}
        pei_energy_per_kg (dict): {scenario: {sorbent: {process: energy_per_kg_3d_matrix}}}
        support_energy_per_kg (dict): {scenario: {sorbent: {process: energy_per_kg_3d_matrix}}}
        sorbent_energy_per_kg (dict): {scenario: {sorbent: {process: energy_per_kg_3d_matrix}}}
        process_names (list): Names for each energy dictionary ['Aziridine', 'PEI', 'Support', 'Sorbent']

    Returns:
        dict: Combined energy per kg dictionary with total energy per kg for each sorbent
    """
    energy_per_kg_dicts = [aziridine_energy_per_kg, pei_energy_per_kg, support_energy_per_kg, sorbent_energy_per_kg]
    combined_result = {}
    
    # Get all scenarios and sorbents from the first dictionary
    if not energy_per_kg_dicts[0]:
        return combined_result
    
    for scenario in energy_per_kg_dicts[0]:
        combined_result[scenario] = {}
        
        for sorbent in energy_per_kg_dicts[0][scenario]:
            combined_result[scenario][sorbent] = {}
            
            # Collect all energy per kg matrices for this sorbent
            all_energy_per_kg_matrices = []
            
            # Add each process from each dictionary
            for energy_per_kg_dict, process_name in zip(energy_per_kg_dicts, process_names):
                if scenario in energy_per_kg_dict and sorbent in energy_per_kg_dict[scenario]:
                    for process, energy_per_kg_matrix in energy_per_kg_dict[scenario][sorbent].items():
                        # Store individual process
                        combined_result[scenario][sorbent][f"{process_name}_{process}"] = energy_per_kg_matrix
                        
                        # Collect for total calculation
                        if hasattr(energy_per_kg_matrix, 'shape') and energy_per_kg_matrix.ndim == 3:
                            all_energy_per_kg_matrices.append(energy_per_kg_matrix)
                        elif hasattr(energy_per_kg_matrix, 'shape') and energy_per_kg_matrix.ndim == 2:
                            # Convert 2D to 3D if needed
                            energy_per_kg_3d = energy_per_kg_matrix.reshape(1, 1, -1)
                            all_energy_per_kg_matrices.append(energy_per_kg_3d)
                        else:
                            # Convert scalar to 3D array
                            energy_per_kg_3d = np.full((1, 1, 10), energy_per_kg_matrix)  # Assuming 10 MCS iterations
                            all_energy_per_kg_matrices.append(energy_per_kg_3d)
            
            # Calculate total energy per kg by summing all 3D matrices
            if all_energy_per_kg_matrices:
                # Get the shape from the first matrix
                reference_shape = all_energy_per_kg_matrices[0].shape
                total_energy_per_kg_3d = np.zeros(reference_shape)
                
                # Sum all energy per kg matrices element-wise
                for energy_per_kg_matrix in all_energy_per_kg_matrices:
                    # Ensure shapes match
                    if energy_per_kg_matrix.shape == reference_shape:
                        total_energy_per_kg_3d += energy_per_kg_matrix
                    else:
                        # Handle shape mismatch by broadcasting
                        total_energy_per_kg_3d += np.broadcast_to(energy_per_kg_matrix, reference_shape)
                
                # Store total energy per kg
                combined_result[scenario][sorbent]["Total_Energy_per_kg"] = total_energy_per_kg_3d
    
    return combined_result

# ========================================================================================================================
# STEP 10: REORGANIZE ENERGY DICTIONARY BY ENERGY TYPE (ELECTRICITY OR HEAT)
# ========================================================================================================================

def reorganize_energy_by_type(energy_dict, process_energy_type_mapping):
    """
    Reorganize energy dictionary to group processes by energy type (electricity or heat).
    
    New structure: energy_dict[scenario_id][sorbent_type][process_type][stage][process] = matrix
    Where process_type is either "electricity" or "heat", and stage/process are separate keys
    
    Parameters:
        energy_dict (dict): Original structure {scenario: {sorbent: {process: matrix}}}
            Process names are like "Aziridine_Stirring" (stage_process)
        process_energy_type_mapping (dict): Mapping defining energy type for each process.
            Format: {stage: {process: "electricity" or "heat"}} or 
            {stage: {process: {sorbent_type: "electricity" or "heat"}}} for sorbent-specific
            Example: {"Aziridine": {"Stirring": "electricity", "Heating": "heat"}}
    
    Returns:
        dict: Reorganized structure {scenario: {sorbent: {process_type: {stage: {process: matrix}}}}}
    """
    reorganized_dict = {}
    
    for scenario_id in energy_dict:
        reorganized_dict[scenario_id] = {}
        
        for sorbent_type in energy_dict[scenario_id]:
            reorganized_dict[scenario_id][sorbent_type] = {
                "electricity": {},
                "heat": {}
            }
            
            for process_name, energy_matrix in energy_dict[scenario_id][sorbent_type].items():
                # Skip Total_Energy_per_kg - we'll handle it separately
                if process_name == "Total_Energy_per_kg":
                    continue
                
                # Parse process name to extract stage and process (e.g., "Aziridine_Stirring" -> stage="Aziridine", process="Stirring")
                if '_' not in process_name:
                    print(f"Warning: Process name '{process_name}' doesn't contain stage prefix. Skipping.")
                    continue
                
                # Split on first underscore to get stage and process
                parts = process_name.split('_', 1)
                stage = parts[0]
                process = parts[1]
                
                # Determine energy type for this process
                energy_type = None
                
                # Look up in nested structure: process_energy_type_mapping[stage][process]
                if stage in process_energy_type_mapping:
                    stage_mapping = process_energy_type_mapping[stage]
                    if process in stage_mapping:
                        mapping_value = stage_mapping[process]
                        
                        # Check if it's sorbent-specific mapping
                        if isinstance(mapping_value, dict):
                            # Sorbent-specific: check if this sorbent has a specific mapping
                            if sorbent_type in mapping_value:
                                energy_type = mapping_value[sorbent_type]
                            else:
                                # If sorbent not in dict, skip or use default
                                print(f"Warning: No energy type mapping for process '{process}' in stage '{stage}' and sorbent '{sorbent_type}'. Skipping.")
                                continue
                        else:
                            # Simple mapping: applies to all sorbents
                            energy_type = mapping_value
                
                if energy_type is None:
                    print(f"Warning: No energy type mapping found for process '{process}' in stage '{stage}'. Skipping.")
                    continue
                
                # Validate energy type
                if energy_type not in ["electricity", "heat"]:
                    print(f"Warning: Invalid energy type '{energy_type}' for process '{process}' in stage '{stage}'. Must be 'electricity' or 'heat'. Skipping.")
                    continue
                
                # Keep each process under its original stage (including recovery processes).
                target_stage = stage
                process_key = process

                # Initialize stage dictionary if it doesn't exist
                if target_stage not in reorganized_dict[scenario_id][sorbent_type][energy_type]:
                    reorganized_dict[scenario_id][sorbent_type][energy_type][target_stage] = {}
                
                # Store in reorganized structure with separate stage and process keys
                # Convert heat from kWh to MJ (1 kWh = 3.6 MJ), keep electricity in kWh.
                # Exception: "Solvent Recovery - Heat" is already in MJ in the model.
                if energy_type == "heat":
                    if process == "Solvent Recovery - Heat":
                        energy_matrix_MJ = energy_matrix
                    else:
                        energy_matrix_MJ = energy_matrix * 3.6

                    # Keep unique stage/process entries; overwrite if duplicated upstream.
                    reorganized_dict[scenario_id][sorbent_type][energy_type][target_stage][process_key] = energy_matrix_MJ
                else:
                    # Keep electricity in kWh
                    reorganized_dict[scenario_id][sorbent_type][energy_type][target_stage][process_key] = energy_matrix
            
            # Calculate Stage_Total for each stage (in MJ: electricity converted + heat)
            # Get all unique stages
            all_stages = set()
            if reorganized_dict[scenario_id][sorbent_type]["electricity"]:
                all_stages.update(reorganized_dict[scenario_id][sorbent_type]["electricity"].keys())
            if reorganized_dict[scenario_id][sorbent_type]["heat"]:
                all_stages.update(reorganized_dict[scenario_id][sorbent_type]["heat"].keys())
            
            # Calculate Stage_Total for each stage
            for stage in all_stages:
                stage_electricity_matrices = []
                stage_heat_matrices = []
                reference_shape = None
                
                # Collect electricity matrices for this stage
                if stage in reorganized_dict[scenario_id][sorbent_type]["electricity"]:
                    for process, matrix in reorganized_dict[scenario_id][sorbent_type]["electricity"][stage].items():
                        if process != "Stage_Total":  # Skip if Stage_Total already exists
                            stage_electricity_matrices.append(matrix)
                            if reference_shape is None:
                                reference_shape = matrix.shape
                
                # Collect heat matrices for this stage
                if stage in reorganized_dict[scenario_id][sorbent_type]["heat"]:
                    for process, matrix in reorganized_dict[scenario_id][sorbent_type]["heat"][stage].items():
                        if process != "Stage_Total":  # Skip if Stage_Total already exists
                            stage_heat_matrices.append(matrix)
                            if reference_shape is None:
                                reference_shape = matrix.shape
                
                # Calculate Stage_Total: convert electricity to MJ and sum with heat
                if reference_shape is not None:
                    stage_total_MJ = np.zeros(reference_shape)
                    
                    # Sum electricity processes (convert from kWh to MJ)
                    for matrix in stage_electricity_matrices:
                        if matrix.shape == reference_shape:
                            stage_total_MJ += matrix * 3.6  # Convert kWh to MJ
                        else:
                            stage_total_MJ += np.broadcast_to(matrix * 3.6, reference_shape)
                    
                    # Sum heat processes (already in MJ)
                    for matrix in stage_heat_matrices:
                        if matrix.shape == reference_shape:
                            stage_total_MJ += matrix
                        else:
                            stage_total_MJ += np.broadcast_to(matrix, reference_shape)
                    
                    # Store Stage_Total once (electricity tree, in MJ) to avoid downstream double counting.
                    if stage_electricity_matrices or stage_heat_matrices:
                        # Ensure stage dictionaries exist
                        if stage not in reorganized_dict[scenario_id][sorbent_type]["electricity"]:
                            reorganized_dict[scenario_id][sorbent_type]["electricity"][stage] = {}
                        
                        # Store Stage_Total in MJ
                        reorganized_dict[scenario_id][sorbent_type]["electricity"][stage]["Stage_Total"] = stage_total_MJ
            
            # Calculate separate totals for electricity and heat (across all stages)
            # Get reference shape from any available matrix (for zero matrices if needed)
            reference_shape = None
            
            # Collect all electricity matrices from nested structure [stage][process]
            electricity_matrices = []
            if reorganized_dict[scenario_id][sorbent_type]["electricity"]:
                for stage in reorganized_dict[scenario_id][sorbent_type]["electricity"]:
                    for process in reorganized_dict[scenario_id][sorbent_type]["electricity"][stage]:
                        if process != "Stage_Total":  # Skip Stage_Total when calculating overall totals
                            matrix = reorganized_dict[scenario_id][sorbent_type]["electricity"][stage][process]
                            electricity_matrices.append(matrix)
                            if reference_shape is None:
                                reference_shape = matrix.shape
            
            # Collect all heat matrices from nested structure [stage][process]
            heat_matrices = []
            if reorganized_dict[scenario_id][sorbent_type]["heat"]:
                for stage in reorganized_dict[scenario_id][sorbent_type]["heat"]:
                    for process in reorganized_dict[scenario_id][sorbent_type]["heat"][stage]:
                        if process != "Stage_Total":  # Skip Stage_Total when calculating overall totals
                            matrix = reorganized_dict[scenario_id][sorbent_type]["heat"][stage][process]
                            heat_matrices.append(matrix)
                            if reference_shape is None:
                                reference_shape = matrix.shape
            
            # Fallback to original Total_Energy_per_kg if no processes found
            if reference_shape is None and "Total_Energy_per_kg" in energy_dict[scenario_id][sorbent_type]:
                reference_shape = energy_dict[scenario_id][sorbent_type]["Total_Energy_per_kg"].shape
            
            # Total_Electricity_per_kg: sum of all electricity processes (in kWh)
            if electricity_matrices:
                # Get reference shape from first matrix
                reference_shape = electricity_matrices[0].shape
                total_electricity = np.zeros(reference_shape)
                
                # Sum all electricity matrices
                for matrix in electricity_matrices:
                    if matrix.shape == reference_shape:
                        total_electricity += matrix
                    else:
                        total_electricity += np.broadcast_to(matrix, reference_shape)
                
                reorganized_dict[scenario_id][sorbent_type]["Total_Electricity_per_kg"] = total_electricity
            else:
                # No electricity processes, create zero matrix
                if reference_shape is not None:
                    reorganized_dict[scenario_id][sorbent_type]["Total_Electricity_per_kg"] = np.zeros(reference_shape)
            
            # Total_Heat_per_kg: sum of all heat processes (already in MJ from conversion above)
            if heat_matrices:
                # Get reference shape from first matrix
                reference_shape = heat_matrices[0].shape
                total_heat = np.zeros(reference_shape)
                
                # Sum all heat matrices (already in MJ)
                for matrix in heat_matrices:
                    if matrix.shape == reference_shape:
                        total_heat += matrix
                    else:
                        total_heat += np.broadcast_to(matrix, reference_shape)
                
                reorganized_dict[scenario_id][sorbent_type]["Total_Heat_per_kg"] = total_heat
            else:
                # No heat processes, create zero matrix
                if reference_shape is not None:
                    reorganized_dict[scenario_id][sorbent_type]["Total_Heat_per_kg"] = np.zeros(reference_shape)
            
            # Calculate Total_Energy_per_kg in MJ: convert electricity (kWh) to MJ and add heat (already in MJ)
            total_energy_MJ = None
            if "Total_Electricity_per_kg" in reorganized_dict[scenario_id][sorbent_type] and "Total_Heat_per_kg" in reorganized_dict[scenario_id][sorbent_type]:
                # Convert electricity from kWh to MJ (1 kWh = 3.6 MJ) and add heat (already in MJ)
                electricity_MJ = reorganized_dict[scenario_id][sorbent_type]["Total_Electricity_per_kg"] * 3.6
                total_energy_MJ = electricity_MJ + reorganized_dict[scenario_id][sorbent_type]["Total_Heat_per_kg"]
            elif "Total_Electricity_per_kg" in reorganized_dict[scenario_id][sorbent_type]:
                # Only electricity: convert to MJ
                total_energy_MJ = reorganized_dict[scenario_id][sorbent_type]["Total_Electricity_per_kg"] * 3.6
            elif "Total_Heat_per_kg" in reorganized_dict[scenario_id][sorbent_type]:
                # Only heat: already in MJ
                total_energy_MJ = reorganized_dict[scenario_id][sorbent_type]["Total_Heat_per_kg"]
            
            if total_energy_MJ is not None:
                reorganized_dict[scenario_id][sorbent_type]["Total_Energy_per_kg"] = total_energy_MJ
    
    return reorganized_dict


def remove_zero_energy_processes(energy_dict, tolerance=1e-20):
    """
    Remove processes with zero (or near-zero) energy consumption from the energy dictionary.
    This reduces dictionary size by removing processes that are not applicable to certain sorbents.
    
    Parameters:
        energy_dict (dict): Dictionary structure {scenario: {sorbent: {energy_type: {stage: {process: matrix}}}}}
        tolerance (float): Tolerance for considering a value as zero (default: 1e-10)
    
    Returns:
        dict: Same structure with zero-energy processes removed
    """
    filtered_dict = {}
    
    for scenario_id, scenario_data in energy_dict.items():
        filtered_dict[scenario_id] = {}
        
        for sorbent_type, sorbent_data in scenario_data.items():
            filtered_dict[scenario_id][sorbent_type] = {}
            
            for energy_type, energy_type_data in sorbent_data.items():
                # Keep non-stage keys (like "Total_Energy_per_kg") as-is
                if energy_type not in ["electricity", "heat"]:
                    filtered_dict[scenario_id][sorbent_type][energy_type] = energy_type_data
                    continue
                
                filtered_dict[scenario_id][sorbent_type][energy_type] = {}
                
                for stage, stage_data in energy_type_data.items():
                    # Keep "Stage_Total" and "Total" keys
                    if stage in ["Stage_Total", "Total"]:
                        filtered_dict[scenario_id][sorbent_type][energy_type][stage] = stage_data
                        continue
                    
                    filtered_dict[scenario_id][sorbent_type][energy_type][stage] = {}
                    
                    for process, energy_matrix in stage_data.items():
                        # Check if matrix has any non-zero values
                        if energy_matrix is not None and hasattr(energy_matrix, 'shape'):
                            # Check if all values are effectively zero
                            max_value = np.max(np.abs(energy_matrix))
                            if max_value > tolerance:
                                # Keep this process - it has non-zero energy
                                filtered_dict[scenario_id][sorbent_type][energy_type][stage][process] = energy_matrix
                            # If max_value <= tolerance, skip this process (it's effectively zero)
                        else:
                            # If it's not a matrix or is None, keep it
                            filtered_dict[scenario_id][sorbent_type][energy_type][stage][process] = energy_matrix
    
    return filtered_dict


def normalize_energy_per_kg_product(
    energy_dict,
    pei_mass_dic,
    support_mass_dic,
    scenarios,
    stages_to_normalize=("PEI", "Support"),
):
    """
    Convert energy dictionary from per kg sorbent to per kg product normalization.
    
    Steps:
    1. Multiply by batch size to get total energy demand (kWh or MJ)
    2. Divide by product mass:
       - Aziridine and PEI stages: divide by PEI produced (step 30)
       - Support stage: divide by Support produced
       - Sorbent stage: keep as per kg sorbent (divide by batch size, which cancels out)
    
    Parameters:
        energy_dict (dict): {scenario: {sorbent: {energy_type: {stage: {process: matrix}}}}}
            Values are normalized per kg of final sorbent produced
        pei_mass_dic (dict): PEI results dict used to get PEI produced (step 30)
        support_mass_dic (dict): Support results dict used to get Support produced
        scenarios (dict): Scenarios dictionary (for batch size)
        stages_to_normalize (tuple): stages to normalize (default: ("PEI","Support"))
    
    Returns:
        dict: same structure as energy_dict, with values normalized per kg product
    """
    def _get_pei_produced(scenario, sorbent, mcs_len):
        """Get PEI produced at step 30 (kg)"""
        pei_produced = np.ones(mcs_len, dtype=float)  # Default: 1 kg
        if scenario in pei_mass_dic and sorbent in pei_mass_dic[scenario]:
            pei_masses = pei_mass_dic[scenario][sorbent]
            if "Polyethyleneimine" in pei_masses:
                arr = np.asarray(pei_masses["Polyethyleneimine"], dtype=float)
                if arr.ndim == 3 and arr.shape[1] > 30:
                    pei_produced = arr[0, 30, :mcs_len] / 1000.0  # g -> kg
        return pei_produced
    
    def _get_support_produced(scenario, sorbent, mcs_len):
        """Support product mass [kg] per MCS draw at the final support step (keys/steps aligned with ``Mass_Function``)."""
        ref_vec = np.ones(mcs_len, dtype=float)
        if support_mass_dic and scenario in support_mass_dic and sorbent in support_mass_dic[scenario]:
            support_masses = support_mass_dic[scenario][sorbent]
            sorbent_lower = sorbent.lower()
            
            # Map sorbent type to support name and step number
            support_name = None
            step_num = None
            
            if 'pei_sg' in sorbent_lower or 'sg' in sorbent_lower:
                # SG (Silica Gel): "Silica" at step 28
                support_name = 'Silica'
                step_num = 28
            elif 'alumina' in sorbent_lower:
                # γ-alumina: "γ-Alumina" at step 29
                if 'γ-Alumina' in support_masses:
                    support_name = 'γ-Alumina'
                elif 'Alumina' in support_masses:
                    support_name = 'Alumina'
                step_num = 29
            elif 'mcm_41' in sorbent_lower or 'mcm-41' in sorbent_lower:
                # MCM-41: "Silica" at step 32
                support_name = 'Silica'
                step_num = 32
            elif 'mil' in sorbent_lower:
                # MIL-101-Cr: "MIL-101 (Cr)" at step 29
                if 'MIL-101 (Cr)' in support_masses:
                    support_name = 'MIL-101 (Cr)'
                elif 'MIL-101(Cr)' in support_masses:
                    support_name = 'MIL-101(Cr)'
                step_num = 29
            elif 'sba_15' in sorbent_lower or 'sba-15' in sorbent_lower:
                # SBA-15: "Silica" at step 32
                support_name = 'Silica'
                step_num = 32
            elif 'kit6' in sorbent_lower or 'kit-6' in sorbent_lower:
                # KIT6: "Silica" at step 28
                support_name = 'Silica'
                step_num = 28
            elif 'hms' in sorbent_lower:
                # HMS: "Silica" at step 32
                support_name = 'Silica'
                step_num = 32
            
            if support_name and support_name in support_masses and step_num is not None:
                arr = np.asarray(support_masses[support_name], dtype=float)
                if arr.ndim == 3 and arr.shape[1] > step_num and arr.shape[2] >= mcs_len:
                    # Get mass at specific step for each MCS iteration
                    ref_vec = arr[0, step_num, :mcs_len] / 1000.0  # g -> kg
        return ref_vec
    
    normalized = {}
    for scenario_id, sorbents in energy_dict.items():
        normalized[scenario_id] = {}
        batch_size_kg = scenarios[scenario_id]["profile"]["batch_size"] / 1000.0  # Convert to kg
        
        for sorbent_type, sorbent_data in sorbents.items():
            normalized[scenario_id][sorbent_type] = {}
            
            for energy_type, stages in sorbent_data.items():
                if energy_type not in ["electricity", "heat"]:
                    normalized[scenario_id][sorbent_type][energy_type] = stages
                    continue
                
                normalized[scenario_id][sorbent_type][energy_type] = {}
                
                for stage_name, processes in stages.items():
                    if not isinstance(processes, dict):
                        normalized[scenario_id][sorbent_type][energy_type][stage_name] = processes
                        continue

                    # Sorbent stage is already normalized per kg sorbent.
                    # Keep as-is to avoid any unintended rescaling/broadcasting artifacts.
                    if stage_name == "Sorbent":
                        normalized[scenario_id][sorbent_type][energy_type][stage_name] = {}
                        for proc_name, proc_matrix in processes.items():
                            if isinstance(proc_matrix, np.ndarray):
                                normalized[scenario_id][sorbent_type][energy_type][stage_name][proc_name] = proc_matrix.copy()
                            else:
                                normalized[scenario_id][sorbent_type][energy_type][stage_name][proc_name] = proc_matrix
                        continue
                    
                    # Get MCS length from first matrix
                    mcs_len = None
                    for _, matrix in processes.items():
                        if isinstance(matrix, np.ndarray) and matrix.ndim == 3:
                            mcs_len = matrix.shape[2]
                            break
                    
                    if mcs_len is None:
                        # Copy processes if no matrices found
                        normalized[scenario_id][sorbent_type][energy_type][stage_name] = {}
                        for proc_name, proc_matrix in processes.items():
                            if isinstance(proc_matrix, np.ndarray) and proc_matrix.ndim == 3:
                                normalized[scenario_id][sorbent_type][energy_type][stage_name][proc_name] = proc_matrix.copy()
                            else:
                                normalized[scenario_id][sorbent_type][energy_type][stage_name][proc_name] = proc_matrix
                        continue
                    
                    # Determine product mass for normalization
                    if stage_name in ["Aziridine", "PEI"]:
                        # Normalize by PEI produced (step 30)
                        product_mass = _get_pei_produced(scenario_id, sorbent_type, mcs_len)
                    elif stage_name == "Support":
                        # Normalize by Support produced (using same logic as material_mass_input)
                        product_mass = _get_support_produced(scenario_id, sorbent_type, mcs_len)
                        # Check if support mass was found (if all values are 1.0, it means default was used)
                        if np.allclose(product_mass, 1.0):
                            print(f"ERROR: Support mass not found for scenario={scenario_id}, sorbent={sorbent_type}. Using default value of 1.0 kg. Cannot properly normalize Support stage energy per kg product.")
                            # Print debug info
                            if support_mass_dic and scenario_id in support_mass_dic:
                                if sorbent_type in support_mass_dic[scenario_id]:
                                    print(f"  DEBUG: Available support_masses keys: {list(support_mass_dic[scenario_id][sorbent_type].keys())}")
                                else:
                                    print(f"  DEBUG: Sorbent {sorbent_type} not in support_mass_dic[{scenario_id}]. Available sorbents: {list(support_mass_dic[scenario_id].keys())}")
                            else:
                                print(f"  DEBUG: Scenario {scenario_id} not in support_mass_dic. Available scenarios: {list(support_mass_dic.keys()) if support_mass_dic else 'None'}")
                    elif stage_name == "Sorbent":
                        # Handled above as no-op copy.
                        product_mass = np.full(mcs_len, batch_size_kg, dtype=float)
                    else:
                        # Other stages: copy without normalization
                        normalized[scenario_id][sorbent_type][energy_type][stage_name] = {}
                        for proc_name, proc_matrix in processes.items():
                            if isinstance(proc_matrix, np.ndarray) and proc_matrix.ndim == 3:
                                normalized[scenario_id][sorbent_type][energy_type][stage_name][proc_name] = proc_matrix.copy()
                            else:
                                normalized[scenario_id][sorbent_type][energy_type][stage_name][proc_name] = proc_matrix
                        continue
                    
                    # Reshape product_mass for broadcasting: (1, 1, mcs_len)
                    product_mass_3d = product_mass.reshape(1, 1, -1)
                    batch_size_3d = np.full((1, 1, mcs_len), batch_size_kg, dtype=float)
                    
                    # Normalize: (energy_per_kg_sorbent * batch_size_kg) / product_mass_kg
                    norm_stage = {}
                    stage_total_matrix = None
                    
                    for process_name, matrix in processes.items():
                        if isinstance(matrix, np.ndarray) and matrix.ndim == 3:
                            # Create a copy to avoid modifying original
                            matrix_copy = matrix.copy()
                            
                            # Step 1: Multiply by batch size to get total energy
                            total_energy = matrix_copy * batch_size_3d
                            
                            # Step 2: Divide by product mass to get per kg product
                            with np.errstate(divide="ignore", invalid="ignore"):
                                norm_matrix = np.divide(
                                    total_energy, product_mass_3d, 
                                    out=np.full_like(total_energy, np.nan), 
                                    where=product_mass_3d > 0
                                )
                            
                            if process_name == "Stage_Total":
                                # Stage_Total is the same MJ total in both electricity and heat trees.
                                # Keep it only once (under electricity) to avoid downstream double counting.
                                if energy_type == "electricity":
                                    stage_total_matrix = norm_matrix
                            else:
                                norm_stage[process_name] = norm_matrix
                        else:
                            if process_name == "Stage_Total":
                                if energy_type == "electricity":
                                    stage_total_matrix = matrix
                            else:
                                norm_stage[process_name] = matrix
                    
                    # Add normalized Stage_Total if it exists (stored once under electricity).
                    if stage_total_matrix is not None:
                        norm_stage["Stage_Total"] = stage_total_matrix
                    
                    normalized[scenario_id][sorbent_type][energy_type][stage_name] = norm_stage
    
    return normalized

