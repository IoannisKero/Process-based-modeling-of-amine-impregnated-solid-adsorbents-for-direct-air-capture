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

# Import Libraries

import numpy as np
from CoolProp.CoolProp import PropsSI

# ========================================================================================================================
# STEP 0: PROCESS TEMPERATURE FUNCTION (T)
# ========================================================================================================================

def process_temperature(mcs_number):
    """Return process temperature (°C) per step for each sorbent in the study (shape 1 × n × mcs)."""

    n = 36  # Number of process steps (matches Mass_Function / LCA_Model routing)

    # PEI_γ_alumina

    PEI_γ_alumina_Temp = np.zeros((1, n, mcs_number))
    PEI_γ_alumina_Temp[:, np.r_[0:12, 16, 18, 20, 24, 26:28, 29:36], :] = np.array([
        25, 160, 25, 25, 25, 25, 100, 25, 25, 100, 50, 50, 25, 60, 25, 60, 100, 700, 80, 25, 25, 25, 60, 80, 0]).reshape(1, -1, 1)
        
    # PEI_SG
    
    PEI_SG_Temp = np.zeros((1, n, mcs_number))
    PEI_SG_Temp[:, np.r_[0:12, 12, 17:19, 25:27, 28, 30, 32, 34:36], :] = np.array([
        25, 160, 25, 25, 25, 25, 100,25, 25, 100, 50, 50, 40, 25, 25, 60, 140, 108, 25, 25, 40, 0]).reshape(1, -1, 1)

    # MIL_101_Cr_PEI
    
    MIL_101_Cr_PEI_Temp = np.zeros((1, n, mcs_number))
    MIL_101_Cr_PEI_Temp[:, np.r_[0:12, 12, 18, 23, 25:27, 29:36], :] = np.array([
        25, 160, 25, 25, 25, 25, 100, 25, 25, 100, 50, 50, 25, 200, 25, 25, 150, 110, 25, 25, 25, 60, 25, 0]).reshape(1, -1, 1)

    # MCM_41_PEI
    
    MCM_41_PEI_Temp = np.zeros((1, n, mcs_number))
    MCM_41_PEI_Temp[:, np.r_[0:12, 12, 18, 21, 25:28, 30, 32, 34:36], :] = np.array([
        25, 160, 25, 25, 25, 25, 100, 25, 25, 100, 50, 50, 25, 100, 25, 25, 100, 550, 25, 25, 70, 0]).reshape(1, -1, 1)

    # SBA_15_PEI
    
    SBA_15_PEI_Temp = np.zeros((1, n, mcs_number))
    SBA_15_PEI_Temp[:, np.r_[0:12, 12, 18, 21, 25:28, 30, 32, 34:36], :] = np.array([
        25, 160, 25, 25, 25, 25, 100, 25, 25, 100, 50, 50, 40, 80, 25, 25, 80, 540, 25, 25, 25, 0]).reshape(1, -1, 1)

    # KIT6_PEI
    
    KIT6_PEI_Temp = np.zeros((1, n, mcs_number))
    KIT6_PEI_Temp[:, np.r_[0:12, 12:15, 18, 21, 26:29, 30, 32, 34:36], :] = np.array([
        25, 160, 25, 25, 25, 25, 100, 25, 25, 100, 50, 50, 25, 25, 40, 100, 25, 100, 550, 150, 25, 25, 70, 0]).reshape(1, -1, 1)

    # HMS_PEI
    
    HMS_PEI_Temp = np.zeros((1, n, mcs_number))
    HMS_PEI_Temp[:, np.r_[0:12, 12:15, 18, 21, 25:28, 30, 32, 34:36], :] = np.array([
        25, 160, 25, 25, 25, 25, 100, 25, 25, 100, 50, 50, 25, 40, 40, 100, 25, 25, 60, 550, 40, 25, 60, 0]).reshape(1, -1, 1)


    Process_Temperature = {
        "PEI_γ_alumina": PEI_γ_alumina_Temp,
        "PEI_SG": PEI_SG_Temp,
        "MIL_101_Cr_PEI":  MIL_101_Cr_PEI_Temp,
        "MCM_41_PEI":  MCM_41_PEI_Temp,
        "SBA_15_PEI":  SBA_15_PEI_Temp,
        "KIT6_PEI":  KIT6_PEI_Temp,
        "HMS_PEI":  HMS_PEI_Temp
    }


    return Process_Temperature

# ========================================================================================================================
# STEP 1: CHEMICAL PROPERTIES - DENSITY (d)
# ========================================================================================================================

def density():
    """Component densities [g/cm³] at 298.15 K, 1 atm (CoolProp where noted)."""

    # Monomers

    Monoethanolamine_density = 1.020 # Source: NIH
    Sodium_Silicate_density = 1.400 # Source: NIH
    Pseudoboehmite_density = 3.010 # Source: Wikipedia
    TEOS_density = 0.933 # Source: Sigma-Aldrich
    CrNO3_density = 1.80 # Source: Sigma-Aldrich
    H2BDC_density = 1.58 # Source: Sigma-Aldrich
    MIL_101_crystals_density = 0.58 # Source: ChemBook

    # Surfactants/Templating Agents

    Pluronic_density = 1.018 # Source: Sigma-Aldrich
    CTAB_density = 2.30
    TMA_density = 0.630
    
    # Solvents
    
    Water_density = PropsSI('D', 'T', 298.153, 'P', 101325, 'Water')/1000
    Ethanol_density = PropsSI('D', 'T', 298.153, 'P', 101325, 'Ethanol')/1000
    Methanol_density = PropsSI('D', 'T', 298.153, 'P', 101325, 'Methanol')/1000
    DMF_density = 0.944 # Source: Sigma-Aldrich
    n_Butanol_density = 0.81 # Source: Sigma-Aldrich
    
    # Other
    
    Sulphuric_Acid_density = 1.840
    Sodium_Hydroxide_pellet_density = 2.130
    Sodium_Hydroxide_density = 1.434
    Ammonia_density = 0.900
    Nitric_Acid_density = 1.420
    Sodium_Oxide_density = 2.300
    Acetic_Acid_density = 1.049
    HCl_density = 1.180
    
    # Products/Byproducts
    
    Aminoethyl_Hydrogen_Sulfate_density = 1.344
    Aziridine_density = 0.830
    Polyethyleneimine_density = 1.050
    Silica_density = 2.300
    γ_alumina_density = 0.933
    MIL_density = 0.580
    Sodium_Sulfate_density = 2.680

    # Support Synthesis
    
    Support_Density = {
        'PEI_SG': {'DI Water': Water_density, 
                   'Sodium Silicate': Sodium_Silicate_density,  
                   'Silica': Silica_density, 
                   'Water Washing': Water_density,
                   'Water Sulfuric Acid':Sulphuric_Acid_density,
                   'Water Byproduct': Water_density,
                   'Sulphuric Acid': Sulphuric_Acid_density,
                   'Sodium Sulfate': Sodium_Sulfate_density},
        'MCM_41_PEI': {'DI Water': Water_density, 
                   'Sodium Oxide': Sodium_Oxide_density,  
                   'Silica': Silica_density, 
                   'Water Washing': Water_density, 
                   'TMA': TMA_density,
                   'CTMA-Br': CTAB_density},
        'PEI_γ_alumina': {'DI Water': Water_density, 
                          'Pluronic P123': Pluronic_density,  
                          'Pseudo': Pseudoboehmite_density, 
                          'Ethanol': Ethanol_density, 
                          'Nitric Acid': Nitric_Acid_density,
                          'γ-Alumina': γ_alumina_density,
                          'Water Byproduct': Water_density,
                          'Water Nitric Acid': Nitric_Acid_density},
        'MIL_101_Cr_PEI': {'DI Water': Water_density,
                           'CrNO3': CrNO3_density, 
                           'H2BDC': H2BDC_density, 
                           'Acetic Acid': Acetic_Acid_density, 
                           'MIL-101 (Cr) crystals': MIL_101_crystals_density,
                           'Methanol Washing': Methanol_density,
                           'DMF Washing': DMF_density,
                           'MIL-101 (Cr)': MIL_density,
                           'Water Acetic Acid': Acetic_Acid_density},
        'SBA_15_PEI': {'HCl': HCl_density , 
                    'Pluronic P123': Pluronic_density , 
                    'TEOS': TEOS_density,
                    'DI Water': Water_density,
                    'Silica': Silica_density,
                    'Ethanol Byproduct': Ethanol_density,
                    'Water Washing': Water_density,
                    'Water HCl': HCl_density},
        'KIT6_PEI': {'DI Water': Water_density, 
                    'HCl': HCl_density, 
                    'Pluronic P123': Pluronic_density , 
                    'N-Butanol': n_Butanol_density , 
                    'TEOS': TEOS_density,
                    'Silica': Silica_density,
                    'Ethanol Byproduct': Ethanol_density,
                    'Water HCl': HCl_density},
        'HMS_PEI': {'DI Water': Water_density , 
                    'HCl': HCl_density , 
                    'Pluronic P123': Pluronic_density , 
                    'TEOS': TEOS_density ,
                    'Water Washing': Water_density,
                    'Silica': Silica_density,
                    'Ethanol Byproduct': Ethanol_density,
                    'Water HCl': HCl_density}
        }
    
    
    # Aziridine Synthesis
    
    Aziridine_Density = {
                    'Monoethanolamine': Monoethanolamine_density,
                    'Aminoethyl Hydrogen Sulfate': Aminoethyl_Hydrogen_Sulfate_density,
                    'Aziridine': Aziridine_density,
                    'Ethanol': Ethanol_density,
                    'Ethanol Washing': Ethanol_density,
                    'Water Sulfuric Acid': Sulphuric_Acid_density,
                    'Water Ethanol': Water_density,
                    'Water Byproduct (1)': Water_density,
                    'Water Byproduct (2)': Water_density,
                    'Water NaOH': Sodium_Hydroxide_density,
                    'Sulfuric Acid': Sulphuric_Acid_density,
                    'Sodium Hydroxide': Sodium_Hydroxide_density,
                    'Sodium Sulfate': Sodium_Sulfate_density,
                    'Sodium Hydroxide Washing': Sodium_Hydroxide_pellet_density
        }
    # PEI Synthesis
    
    PEI_Density = {'Aziridine': Aziridine_density,
                   'DI Water': Water_density,
                   'Polyethyleneimine': Polyethyleneimine_density}
    
    # Sorbent Synthesis 
    
    Sorbent_Density = {
            "PEI_γ_alumina": {"Polyethyleneimine": Polyethyleneimine_density,
                              "Methanol": Methanol_density,
                              "PEI_γ_alumina": γ_alumina_density},
            "PEI_SG": {"Polyethyleneimine": Polyethyleneimine_density,
                       "Methanol": Methanol_density,
                       "PEI_SG": Silica_density},
            "MIL_101_Cr_PEI": {"Polyethyleneimine": Polyethyleneimine_density,
                               "Methanol": Methanol_density,
                               "MIL_101_Cr_PEI": MIL_density},
            "MCM_41_PEI":  {"Polyethyleneimine": Polyethyleneimine_density,
                            "Methanol": Methanol_density,
                            "MCM_41_PEI": Silica_density},
            "SBA_15_PEI":  {"Polyethyleneimine": Polyethyleneimine_density,
                            "Methanol": Methanol_density,
                            "SBA_15_PEI": Silica_density},
            "KIT6_PEI":  {"Polyethyleneimine": Polyethyleneimine_density,
                          "Methanol": Methanol_density,
                          "KIT6_PEI": Silica_density},
            "HMS_PEI":  {"Polyethyleneimine": Polyethyleneimine_density,
                         "Ethanol": Ethanol_density,
                         "HMS_PEI": Silica_density}
        }
    
    
    return Support_Density, Aziridine_Density, PEI_Density, Sorbent_Density


# ========================================================================================================================
# STEP 2: CHEMICAL PROPERTIES - HEAT CAPACITY (cp)
# ========================================================================================================================

def heat_capacity():
    """Component heat capacities [J/(g·°C)] near 298.15 K, 1 atm (CoolProp where noted)."""

    # Monomers

    Monoethanolamine_cp = 2.30 # Source: NIH
    Sodium_Silicate_cp = 0.92 # Source: NIH
    Pseudoboehmite_cp = 1.09 # Source: Wikipedia
    TEOS_cp = 1.69 # Source: Sigma-Aldrich
    CrNO3_cp = 1.14 
    H2BDC_cp = 1.20 
    MIL_101_crystals_cp = 0.49 # Source: ChemBook

    # Surfactants/Templating Agents

    Pluronic_cp = 1.83
    CTAB_cp = 1.20
    TMA_cp = 2.14
    
    # Solvents
    
    Water_cp = PropsSI('C', 'T', 298.15, 'P', 101325, 'Water')/1000
    Ethanol_cp = PropsSI('C', 'T', 298.15, 'P', 101325, 'Ethanol')/1000
    Methanol_cp = PropsSI('C', 'T', 298.15, 'P', 101325, 'Methanol')/1000
    DMF_cp = 2 # Source: NIST
    n_Butanol_cp = 2.39 # Source: Sigma-Aldrich & NIST

    # Other
    
    Sulphuric_Acid_cp = 1.40
    Sodium_Hydroxide_pellet_cp = 1.49
    Sodium_Hydroxide_cp = 3.39
    Ammonia_cp = 4.50
    Nitric_Acid_cp = 1.72
    Sodium_Oxide_cp = 1.12
    Acetic_Acid_cp = 2.05
    HCl_cp = 2.15

    # Products/Byproducts
    
    Aminoethyl_Hydrogen_Sulfate_cp = 1.42
    Aziridine_cp = 2.48
    Polyethyleneimine_cp = 1.60
    Silica_cp = 0.74
    γ_alumina_cp = 0.82
    MIL_cp = 0.49
    Sodium_Sulfate_cp = 0.90

    # Support Synthesis
    
    Support_Cp = {
        'PEI_SG': {'DI Water': Water_cp, 
                   'Sodium Silicate': Sodium_Silicate_cp,  
                   'Silica': Silica_cp, 
                   'Water Washing': Water_cp,
                   'Water Sulfuric Acid': Sulphuric_Acid_cp,
                   'Water Byproduct': Water_cp,
                   'Sulphuric Acid': Sulphuric_Acid_cp,
                   'Sodium Sulfate': Sodium_Sulfate_cp},
        'MCM_41_PEI': {'DI Water': Water_cp, 
                   'Sodium Oxide': Sodium_Oxide_cp,  
                   'Silica': Silica_cp, 
                   'Water Washing': Water_cp, 
                   'TMA': TMA_cp,
                   'CTMA-Br': CTAB_cp},
        'PEI_γ_alumina': {'DI Water': Water_cp, 
                          'Pluronic P123': Pluronic_cp,  
                          'Pseudo': Pseudoboehmite_cp, 
                          'Ethanol': Ethanol_cp, 
                          'Nitric Acid': Nitric_Acid_cp,
                          'γ-Alumina': γ_alumina_cp,
                          'Water Byproduct': Water_cp,
                          'Water Nitric Acid': Nitric_Acid_cp},
        'MIL_101_Cr_PEI': {'DI Water': Water_cp,
                           'CrNO3': CrNO3_cp, 
                           'H2BDC': H2BDC_cp, 
                           'Acetic Acid': Acetic_Acid_cp, 
                           'MIL-101 (Cr) crystals': MIL_101_crystals_cp,
                           'Methanol Washing': Methanol_cp,
                           'DMF Washing': DMF_cp,
                           'MIL-101 (Cr)': MIL_cp,
                           'Water Acetic Acid': Acetic_Acid_cp},
        'SBA_15_PEI': {'HCl': HCl_cp , 
                    'Pluronic P123': Pluronic_cp , 
                    'TEOS': TEOS_cp,
                    'DI Water': Water_cp,
                    'Silica': Silica_cp,
                    'Ethanol Byproduct': Ethanol_cp,
                    'Water Washing': Water_cp,
                    'Water HCl': HCl_cp},
        'KIT6_PEI': {'DI Water': Water_cp, 
                    'HCl': HCl_cp, 
                    'Pluronic P123': Pluronic_cp , 
                    'N-Butanol': n_Butanol_cp ,
                    'TEOS': TEOS_cp,
                    'Silica': Silica_cp,
                    'Ethanol Byproduct': Ethanol_cp,
                    'Water HCl': HCl_cp},
        'HMS_PEI': {'DI Water': Water_cp , 
                    'HCl': HCl_cp , 
                    'Pluronic P123': Pluronic_cp , 
                    'TEOS': TEOS_cp ,
                    'Water Washing': Water_cp,
                    'Silica': Silica_cp,
                    'Ethanol Byproduct': Ethanol_cp,
                    'Water HCl': HCl_cp}
        }
    
    # Aziridine Synthesis
    
    Aziridine_Cp = {
                    'Monoethanolamine': Monoethanolamine_cp,
                    'Aminoethyl Hydrogen Sulfate': Aminoethyl_Hydrogen_Sulfate_cp,
                    'Aziridine': Aziridine_cp,
                    'Ethanol': Ethanol_cp,
                    'Ethanol Washing': Ethanol_cp,
                    'Water Sulfuric Acid': Sulphuric_Acid_cp,
                    'Water Ethanol': Water_cp,
                    'Water Byproduct (1)': Water_cp,
                    'Water Byproduct (2)': Water_cp,
                    'Water NaOH': Sodium_Hydroxide_cp,
                    'Sulfuric Acid': Sulphuric_Acid_cp,
                    'Sodium Hydroxide': Sodium_Hydroxide_cp,
                    'Sodium Sulfate': Sodium_Sulfate_cp,
                    'Sodium Hydroxide Washing': Sodium_Hydroxide_pellet_cp
        }
    
    
    # PEI Synthesis

    PEI_Cp = {'Aziridine': Aziridine_cp,
              'DI Water': Water_cp,
              'Polyethyleneimine': Polyethyleneimine_cp}
    
    # Sorbent Synthesis 
    
    Sorbent_Cp = {
            "PEI_γ_alumina": {"Polyethyleneimine": Polyethyleneimine_cp,
                              "Methanol": Methanol_cp,
                              "PEI_γ_alumina": γ_alumina_cp},
            "PEI_SG": {"Polyethyleneimine": Polyethyleneimine_cp,
                       "Methanol": Methanol_cp,
                       "PEI_SG": Silica_cp},
            "MIL_101_Cr_PEI": {"Polyethyleneimine": Polyethyleneimine_cp,
                               "Methanol": Methanol_cp,
                               "MIL_101_Cr_PEI": MIL_cp},
            "MCM_41_PEI":  {"Polyethyleneimine": Polyethyleneimine_cp,
                            "Methanol": Methanol_cp,
                            "MCM_41_PEI": Silica_cp},
            "SBA_15_PEI":  {"Polyethyleneimine": Polyethyleneimine_cp,
                            "Methanol": Methanol_cp,
                            "SBA_15_PEI": Silica_cp},
            "KIT6_PEI":  {"Polyethyleneimine": Polyethyleneimine_cp,
                          "Methanol": Methanol_cp,
                          "KIT6_PEI": Silica_cp},
            "HMS_PEI":  {"Polyethyleneimine": Polyethyleneimine_cp,
                         "Ethanol": Ethanol_cp,
                         "HMS_PEI": Silica_cp}
        }
    
    return Support_Cp, Aziridine_Cp, PEI_Cp, Sorbent_Cp


# ========================================================================================================================
# STEP 3: CHEMICAL PROPERTIES - ENTHALPY OF VAPORIZATION (ΔΗvap)
# ========================================================================================================================

def enthalpy():
    """Enthalpy of vaporization ΔHvap [J/g] for solvents at process-relevant temperatures."""

    # Water: indices match temperature levels [50, 55, 60, 65, 75, 80, 100, 140, 150] °C
    Water_ΔH = np.array([2382, 2370, 2358, 2345, 2321, 2308, 2256, 2144, 2114]) # Source: The Engineering Toolbox
    # Ethanol: [60, 75, 100] °C
    Ethanol_ΔH = np.array([919, 842, 870])  # Source: The Engineering Toolbox
    # Methanol: [40, 50, 60, 70, 80, 90, 100, 150] °C
    Methanol_ΔH = (np.array([36.7, 36.2, 35.6, 34.7, 36.9, 36.3, 32.7, 28.1])/32.042)*1000 # Source: NIST & PubChem
    # DMF at 150 °C
    DMF_ΔH = 572 # Source: NIST & PubChem
    # n-butanol at 100 °C
    n_Butanol_ΔH = 621 # Source: Sigma-Aldrich & NIST
    # Ethyleneimine (aziridine) at 55 °C
    Aziridine_ΔΗ = 1323 # Source: Chemeo

    # Support Synthesis
    
    Support_ΔH = {
        "PEI_γ_alumina": {'DI Water': Water_ΔH[2],  
                          'Ethanol': Ethanol_ΔH[2],
                          'Water Byproduct': Water_ΔH[2]},
        "PEI_SG": {'DI Water': Water_ΔH[6], 
                   'Water Washing':  Water_ΔH[7],
                   'Water Sulphuric':  Water_ΔH[7],
                   'Water Byproduct':  Water_ΔH[7]},
        "MCM_41_PEI": {'DI Water':  Water_ΔH[6], 
                       'Water Washing':  Water_ΔH[6]},
        'MIL_101_Cr_PEI': {'DI Water': Water_ΔH[8],
                           'Methanol Washing': Methanol_ΔH[7],
                           'DMF Washing': DMF_ΔH},
        'SBA_15_PEI': {'DI Water': Water_ΔH[5], 
                       'Water Washing': Water_ΔH[5],
                       'Ethanol Byproduct': Ethanol_ΔH[2]},
        'KIT6_PEI': {'DI Water': Water_ΔH[6], 
                     'N-Butanol': n_Butanol_ΔH,
                     'Ethanol Byproduct': Ethanol_ΔH[2]}, 
        'HMS_PEI': {'DI Water': Water_ΔH[2], 
                    'Water Washing': Water_ΔH[2],
                    'Ethanol Byproduct': Ethanol_ΔH[0]}
    }

    # Aziridine Synthesis - Distillation
    
    Aziridine_ΔH = {'Ethanol': Ethanol_ΔH[2],
                    'Ethanol Washing': Ethanol_ΔH[2],
                    'Water Sulfuric': Water_ΔH[6],
                    'Water Ethanol': Water_ΔH[6],
                    'Water Byproduct (1)': Water_ΔH[6],
                    'Water Byproduct (2)': Water_ΔH[6],
                    'Water NaOH': Water_ΔH[6],
                    'Aziridine': Aziridine_ΔΗ}
    
    # PEI Synthesis
    
    PEI_ΔH = {'DI Water': Water_ΔH[0]}
    
    # Sorbent Synthesis

    Sorbent_ΔΗ = {
        'Drying':
            {"PEI_γ_alumina": { "Methanol": Methanol_ΔH[4]},
             "PEI_SG": { "Methanol": Methanol_ΔH[0]},
             "MCM_41_PEI":  { "Methanol": Methanol_ΔH[3]},
             "KIT6_PEI":  { "Methanol": Methanol_ΔH[3]},
             "HMS_PEI":  { "Ethanol": Ethanol_ΔH[0]},
             "MIL_101_Cr_PEI":  { "Methanol": 0},
             'SBA_15_PEI': { "Methanol": 0}},
        'Evaporation':
            {"PEI_γ_alumina": { "Methanol": Methanol_ΔH[2]},
             "MIL_101_Cr_PEI":  { "Methanol": Methanol_ΔH[2]},
             "PEI_SG": { "Methanol": 0},
             "MCM_41_PEI":  { "Methanol": 0},
             "KIT6_PEI":  { "Methanol": 0},
             "HMS_PEI":  { "Ethanol": 0},
             'SBA_15_PEI': { "Methanol": 0}},
            }

    return Support_ΔH, Aziridine_ΔH, PEI_ΔH, Sorbent_ΔΗ
