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

# Import Functions

import numpy as np

# ========================================================================================================================
# STEP 0: CHEMICAL PROPERTIES - HEAT CAPACITY (cp)
# ========================================================================================================================

def heat_capacity_sum(mass_dict, cp_dict):
    
    """
    Compute the step-wise effective heat capacity (Cp) matrix for components.

    Formula:
        Mixture_cp = ( Σ mass_component * Cp_component ) / ( Σ mass_component )

    Parameters:
    - mass_dict: Dictionary {component: mass_matrix (1x n_steps x mcs_number)}
    - cp_dict: Dictionary {component: Cp value (float)}

    Returns:
    - NumPy array (1x n_steps x mcs_number) with the mixture heat capacity
    """
    
    # Initialize mixture mass and cp-mass factor matrices
    
    mixture_mass = np.zeros_like(next(iter(mass_dict.values())))  
    cp_mass_factor = np.zeros_like(mixture_mass)

    for component, mass_matrix in mass_dict.items():
        if component in cp_dict:  # Ensure Cp data exists
            cp_value = cp_dict[component]  # Extract scalar Cp
            mixture_mass += mass_matrix  # Σ mass_component
            cp_mass_factor += mass_matrix * cp_value  # Σ (mass_component * Cp_component)

    # Mixture heat capacity 
    
    Mixture_cp = np.divide(cp_mass_factor, mixture_mass, out=np.zeros_like(mixture_mass), where=mixture_mass > 0)

    return Mixture_cp, mixture_mass

# ========================================================================================================================
# STEP 1: CHEMICAL PROPERTIES - HEAT CAPACITY (cp)
# ========================================================================================================================

def heat_capacity_solvents(mass_dict, cp_dict):
    """
    Compute the step-wise effective heat capacity (Cp) matrix for solvents only.

    Formula:
        Mixture_cp = ( Σ mass_solvent * Cp_solvent ) / ( Σ mass_solvent )

    Parameters:
    - mass_dict: Dictionary {component: mass_matrix (1x n_steps x mcs_number)}
    - cp_dict: Dictionary {component: Cp value (float)}

    Returns:
    - NumPy array (1x n_steps x mcs_number) with the mixture heat capacity of solvents only
    - NumPy array (1x n_steps x mcs_number) with the total solvent mass
    """

    solvent_keywords = [
        "water", "ethanol", "methanol", "dmf", "acetone",
        "toluene", "butanol"]

    

    mixture_mass = np.zeros_like(next(iter(mass_dict.values())))
    cp_mass_factor = np.zeros_like(mixture_mass)

    for component, mass_matrix in mass_dict.items():
        if component in cp_dict:
            name_lower = component.lower()
            if any(keyword in name_lower for keyword in solvent_keywords):
                if component == "Monoethanolamine":
                    continue
                cp_value = cp_dict[component]
                mixture_mass += mass_matrix
                cp_mass_factor += mass_matrix * cp_value

    Mixture_cp = np.divide(
        cp_mass_factor, mixture_mass,
        out=np.zeros_like(mixture_mass),
        where=mixture_mass > 0
    )

    return Mixture_cp, mixture_mass

            
# ========================================================================================================================
# STEP 2: CHEMICAL PROPERTIES - DENSITY (cp)
# ========================================================================================================================

def density_sum(mass_dict, density_dict):
    
    """
    Compute the step-wise effective density (ρ) matrix for components.

    Formula:
        Product_Volume = ( Σ mass_component / Density_component )
        Product_Density = ( Σ mass_component ) / Product_Volume

    Parameters:
    - mass_dict: Dictionary {component: mass_matrix (1x n_steps x mcs_number)}
    - density_dict: Dictionary {component: density value (float) in g/cm³}

    Returns:
    - NumPy array (1x n_steps x mcs_number) with the step-wise effective density (g/cm³).
    """

    # Initialize mixture mass and volume matrices
    
    mixture_mass = np.zeros_like(next(iter(mass_dict.values())))  
    mixture_volume = np.zeros_like(mixture_mass)

    for component, mass_matrix in mass_dict.items():
        if component in density_dict:  # Ensure density data exists
            density_value = density_dict[component]  # Extract scalar density
            mixture_mass += mass_matrix  # Σ mass_component
            mixture_volume += np.nan_to_num( mass_matrix / density_value, nan=0.0) / 1000
            

    # Mixture density
    
    Mixture_density = np.divide(mixture_mass, mixture_volume, out=np.zeros_like(mixture_volume), where=mixture_volume > 0)

    return Mixture_density, mixture_volume, mixture_mass 

# ========================================================================================================================
# STEP 3: CHEMICAL PROPERTIES - ENTHALPY OF VAPORIZATION (ΔΗvap)
# ========================================================================================================================

def enthalpy_sum(mass_dict, DHvap_dict):
    """
    Compute the step-wise effective enthalpy of vaporization (ΔHvap) matrix for SOLVENTS ONLY.

    Formula:
        Solvents_DHvap = ( Σ mass_solvent * DHvap_solvent ) / ( Σ mass_solvent )

    Parameters:
    - mass_dict: Dictionary {component: mass_matrix (1x n_steps x mcs_number)}
    - DHvap_dict: Dictionary {component: ΔHvap value (float, J/kg)}

    Returns:
    - NumPy array (1x n_steps x mcs_number) with the solvent mixture enthalpy of vaporization
    - NumPy array (1x n_steps x mcs_number) with the SOLVENT-ONLY mass matrix (zeros for non-solvents)
    """

    # Define solvent keywords (same as heat_capacity_solvents)
    
    # Initialize mixture mass and DHvap-mass factor matrices
    Solvents_mass = np.zeros_like(next(iter(mass_dict.values())))  
    DHvap_mass_factor = np.zeros_like(Solvents_mass)

    # If DHvap_data is a dictionary, iterate over the mass_dict items
    for solvent, mass_matrix in mass_dict.items():
        if solvent in DHvap_dict:
            DHvap_value = DHvap_dict[solvent]  # Extract scalar DHvap for this solvent
            Solvents_mass += mass_matrix        # Sum the mass components
            DHvap_mass_factor += mass_matrix * DHvap_value  # Sum weighted by DHvap value

    # Compute Mixture ΔHvap with element-wise division, replacing NaN values with 0
    Mixture_DHvap = np.divide(DHvap_mass_factor, Solvents_mass, out=np.zeros_like(Solvents_mass), where=Solvents_mass > 0)

    return Mixture_DHvap, Solvents_mass

# ========================================================================================================================
# STEP 4: CHEMICAL PROPERTIES - Final Product Parameters
# ========================================================================================================================

def final_product_property(Final_Product_Property, prop, sorbent_type, scenario):
    
    """
    Extracts the final nonzero value of a given property (e.g., heat capacity, density) 
    for a specific sorbent type across different scenarios. This function identifies the last 
    nonzero value in the provided parameter matrix and stores it for further calculations, such 
    as PEI-support mixture properties.
    """

    # Extract the final product parameter to use later in PEI-support mixture
    nonzero_property = np.where(prop != 0)[1]  # Get indices where cp is nonzero 
    
    if nonzero_property.size > 0:
        last_index = nonzero_property[-1] # Last nonzero column index
        product_property = prop[0, last_index] # Extract the corresponding Cp value
            
        if sorbent_type not in Final_Product_Property :
            Final_Product_Property [sorbent_type] = {}
        Final_Product_Property[sorbent_type][scenario] = product_property
        
    return Final_Product_Property
