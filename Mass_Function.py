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

# Import Libraries & Functions
import pandas as pd
import numpy as np
import Propagation_Functions as pf
import Scenarios as sc

# ========================================================================================================================
# STEP 0: EXTRACT PARAMETER VALUES FROM EXCEL DATABASE
# ========================================================================================================================

def mass_analysis(scenario, mcs_number): 

    # Call Scenarios Function & Retrieve Yields
    Scenarios = sc.generate_scenarios()
    Yields = Scenarios[scenario]['yields']

    # Assign Sorbent Mass & PEI/Support Mass
    
    Sorbent_mass = Scenarios[scenario]['profile']['batch_size'] 
    PEI_sorbent = Sorbent_mass * Scenarios[scenario]['profile']['pei_composition'] / 100
    Support_sorbent = Sorbent_mass - PEI_sorbent

    def sample_yield_triangular(worst, base, high, size):
        """
        Robust triangular sampler for yield triplets.
        Ensures left <= mode <= right by rotating high/worst when needed and clipping base if out of bounds.
        """
        left = min(worst, high)
        right = max(worst, high)
        mode = float(np.clip(base, left, right))
        return np.random.triangular(left, mode, right, size)

    # Step Yields for Precursors

    Precursor_Yields = Yields['Precursors']
    High_Yields = Precursor_Yields['High']
    Base_Yields = Precursor_Yields['Base']
    Worst_Yields = Precursor_Yields['Worst']

    Activation_Yield = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Stirring_Yield = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Sonication_Yield = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Heating_Yield = sample_yield_triangular(Worst_Yields["Heating"], Base_Yields["Heating"], High_Yields["Heating"], mcs_number)
    Filtration_Yield = sample_yield_triangular(Worst_Yields["Filtration"], Base_Yields["Filtration"], High_Yields["Filtration"], mcs_number)
    Aging_Yield = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Evaporation_Yield = sample_yield_triangular(Worst_Yields["Evaporation"], Base_Yields["Evaporation"], High_Yields["Evaporation"], mcs_number)
    Distillation_Yield = sample_yield_triangular(Worst_Yields["Disitillation"], Base_Yields["Disitillation"], High_Yields["Disitillation"], mcs_number)
    Polymerization_Aziridine_Yield = sample_yield_triangular(Worst_Yields["Polymerization (Aziridine)"], Base_Yields["Polymerization (Aziridine)"], High_Yields["Polymerization (Aziridine)"], mcs_number)
    Centrifugation_Yield = sample_yield_triangular(Worst_Yields["Centrifugation"], Base_Yields["Centrifugation"], High_Yields["Centrifugation"], mcs_number)
    Grinding_Yield = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Washing_Yield = sample_yield_triangular(Worst_Yields["Washing"], Base_Yields["Washing"], High_Yields["Washing"], mcs_number)
    Drying_Yield = sample_yield_triangular(Worst_Yields["Drying"], Base_Yields["Drying"], High_Yields["Drying"], mcs_number)
    Calcination_Yield = sample_yield_triangular(Worst_Yields["Calcination"], Base_Yields["Calcination"], High_Yields["Calcination"], mcs_number)

    # Step Yields for Solvents 

    Solvent_Yields = Yields['Solvents']
    High_Yields = Solvent_Yields['High']
    Base_Yields = Solvent_Yields['Base']
    Worst_Yields = Solvent_Yields['Worst']

    Activation_Yield_s = 0
    Stirring_Yield_s = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Sonication_Yield_s = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Heating_Yield_s = sample_yield_triangular(Worst_Yields["Heating"], Base_Yields["Heating"], High_Yields["Heating"], mcs_number)
    Aging_Yield_s = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Evaporation_Yield_s = sample_yield_triangular(Worst_Yields["Evaporation"], Base_Yields["Evaporation"], High_Yields["Evaporation"], mcs_number)
    Distillation_Yield_s = sample_yield_triangular(Worst_Yields["Disitillation"], Base_Yields["Disitillation"], High_Yields["Disitillation"], mcs_number)
    Polymerization_Aziridine_Yield_s = sample_yield_triangular(Worst_Yields["Polymerization (Aziridine)"], Base_Yields["Polymerization (Aziridine)"], High_Yields["Polymerization (Aziridine)"], mcs_number)
    Filtration_Yield_s = sample_yield_triangular(Worst_Yields["Filtration"], Base_Yields["Filtration"], High_Yields["Filtration"], mcs_number)
    Centrifugation_Yield_s = sample_yield_triangular(Worst_Yields["Centrifugation"], Base_Yields["Centrifugation"], High_Yields["Centrifugation"], mcs_number)
    Grinding_Yield_s = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Washing_Yield_s = sample_yield_triangular(Worst_Yields["Washing"], Base_Yields["Washing"], High_Yields["Washing"], mcs_number)
    Drying_Yield_s = sample_yield_triangular(Worst_Yields["Drying"], Base_Yields["Drying"], High_Yields["Drying"], mcs_number)
    Calcination_Yield_s = sample_yield_triangular(Worst_Yields["Calcination"], Base_Yields["Calcination"], High_Yields["Calcination"], mcs_number)

    # Step Yield for Surfactants

    Surfactant_Yields = Yields['Surfactants']
    High_Yields = Surfactant_Yields['High']
    Base_Yields = Surfactant_Yields['Base']
    Worst_Yields = Surfactant_Yields['Worst']
    
    Activation_Yield_sur = 0
    Stirring_Yield_sur = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Sonication_Yield_sur = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Heating_Yield_sur = 0
    Aging_Yield_sur = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Evaporation_Yield_sur = sample_yield_triangular(Worst_Yields["Evaporation"], Base_Yields["Evaporation"], High_Yields["Evaporation"], mcs_number)
    Distillation_Yield_sur = 0
    Polymerization_Aziridine_Yield_sur = 0
    Filtration_Yield_sur = sample_yield_triangular(Worst_Yields["Filtration"], Base_Yields["Filtration"], High_Yields["Filtration"], mcs_number)
    Centrifugation_Yield_sur = sample_yield_triangular(Worst_Yields["Centrifugation"], Base_Yields["Centrifugation"], High_Yields["Centrifugation"], mcs_number)
    Grinding_Yield_sur = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Washing_Yield_sur = sample_yield_triangular(Worst_Yields["Washing"], Base_Yields["Washing"], High_Yields["Washing"], mcs_number)
    Drying_Yield_sur = sample_yield_triangular(Worst_Yields["Drying"], Base_Yields["Drying"], High_Yields["Drying"], mcs_number)
    Calcination_Yield_sur = sample_yield_triangular(Worst_Yields["Calcination"], Base_Yields["Calcination"], High_Yields["Calcination"], mcs_number)

    # Step Yields for Other

    Other_Yields = Yields['Other']
    High_Yields = Other_Yields['High']
    Base_Yields = Other_Yields['Base']
    Worst_Yields = Other_Yields['Worst']

    Activation_Yield_other = 0
    Stirring_Yield_other = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Sonication_Yield_other = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Heating_Yield_other = 0
    Aging_Yield_other = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Evaporation_Yield_other = sample_yield_triangular(Worst_Yields["Evaporation"], Base_Yields["Evaporation"], High_Yields["Evaporation"], mcs_number)
    Distillation_Yield_other = sample_yield_triangular(Worst_Yields["Disitillation"], Base_Yields["Disitillation"], High_Yields["Disitillation"], mcs_number)
    Polymerization_Aziridine_Yield_other = 0
    Filtration_Yield_other = sample_yield_triangular(Worst_Yields["Filtration"], Base_Yields["Filtration"], High_Yields["Filtration"], mcs_number)
    Centrifugation_Yield_other = sample_yield_triangular(Worst_Yields["Centrifugation"], Base_Yields["Centrifugation"], High_Yields["Centrifugation"], mcs_number)
    Grinding_Yield_other = sample_yield_triangular(Worst_Yields["Stirring/Sonication/Grinding/Aging/Activation"], Base_Yields["Stirring/Sonication/Grinding/Aging/Activation"], High_Yields["Stirring/Sonication/Grinding/Aging/Activation"], mcs_number)
    Washing_Yield_other = sample_yield_triangular(Worst_Yields["Washing"], Base_Yields["Washing"], High_Yields["Washing"], mcs_number)
    Drying_Yield_other = sample_yield_triangular(Worst_Yields["Drying"], Base_Yields["Drying"], High_Yields["Drying"], mcs_number)
    Calcination_Yield_other = sample_yield_triangular(Worst_Yields["Calcination"], Base_Yields["Calcination"], High_Yields["Calcination"], mcs_number)

    
    """
    n = 36, total manufacturing process steps across the selected sorbents.
    
    (1) - Aziridine Synthesis (Same for all sorbents)
    (2) - PEI Polymerization (Same for all sorbents)
    (3) - Support Preparation (includes all processes present among the selected supports)
    (4) - PEI Impregnation (includes all processes present among the selected sorbents)

    0. Stirring (1)            
    1. Heating (1)         
    2. Grinding (1)            
    3. Filtration (1)           
    4. Washing (1)        
    5. Stirring (1)          
    6. Distillation (1)   
    7. Drying (1) 
    8. Stirring (2)                 
    9. Aziridine Polymerization (2)
    10. Evaporation (2)
    11. Drying (2)
    12. Stirring (3) 
    13. Stirring (3) 
    14. Stirring (3)
    15. Stirring (3)
    16. Sonication (3)
    17. Gelation (3)
    18. Aging/Hydrothermal Treatment (3) 
    19. Polymerization (3) (not applicable)
    20. Stirring (3)
    21. Filtration (3)
    22. Extraction (3) (not applicable)
    23. Centrifugation (3)
    24. Evaporation (3)
    25. Washing (3)
    26. Drying (3)
    27. Calcination (3)
    28. Drying (4) - support drying before impregnation (when applicable)
    29. Activation (4) - support activation before impregnation (when applicable)
    30. Stirring (4) - PEI Dissolution (only PEI and solvent)
    31. Sonication (4) - support sonication before impregnation (when applicable)
    32. Stirring (4) - PEI/support mixture
    33. Evaporation (4)
    34. Drying (4)
    35. Final Sorbent

    """

# ========================================================================================================================
# STEP 1: DEFINE SORBENT TYPES & CHEMICAL PROCESSES FOR EACH TYPE
# ========================================================================================================================

# 0 -> step does not exist for this sorbent
# 1 -> step exists for this sorbent

    n = 36 # Number of process steps

    # PEI_γ_alumina

    PEI_γ_alumina_steps = np.zeros((1, n, mcs_number))  # , number of MCS
    PEI_γ_alumina_steps[:, np.r_[0:12, 16, 18, 20, 24, 26:28, 29:36], :] = 1

    # PEI_SG
    
    PEI_SG_steps = np.zeros((1, n, mcs_number))  # 3D array for MCS
    PEI_SG_steps[:, np.r_[0:12, 12, 17:19, 25:27, 28, 30, 32, 34:36], :] = 1

    # MIL_101_Cr_PEI
    
    MIL_101_Cr_PEI_steps = np.zeros((1, n, mcs_number))  # 3D array for MCS
    MIL_101_Cr_PEI_steps[:, np.r_[0:12, 12, 18, 23, 25:27, 29:36], :] = 1

    # MCM_41_PEI
    
    MCM_41_PEI_steps = np.zeros((1, n, mcs_number))  # 3D array for MCS
    MCM_41_PEI_steps[:, np.r_[0:12, 12, 18, 21, 25:28, 30, 32, 34:36], :] = 1

    # SBA_15_PEI
    
    SBA_15_PEI_steps = np.zeros((1, n, mcs_number))  # 3D array for MCS
    SBA_15_PEI_steps[:, np.r_[0:12, 12, 18, 21, 25:28, 30, 32, 34:36], :] = 1

    # KIT6_PEI
    
    KIT6_PEI_steps = np.zeros((1, n, mcs_number))  # 3D array for MCS
    KIT6_PEI_steps[:, np.r_[0:12, 12:15, 18, 21, 26:29, 30, 32, 34:36], :] = 1

    # HMS_PEI
    
    HMS_PEI_steps = np.zeros((1, n, mcs_number))  # 3D array for MCS
    HMS_PEI_steps[:, np.r_[0:12, 12:15, 18, 21, 25:28, 30, 32, 34:36], :] = 1


    # Create a dictionary for the final sorbents included in the study

    sorbent_steps = {
        "PEI_γ_alumina": PEI_γ_alumina_steps,
        "PEI_SG": PEI_SG_steps,
        "MIL_101_Cr_PEI":  MIL_101_Cr_PEI_steps,
        "MCM_41_PEI":  MCM_41_PEI_steps,
        "SBA_15_PEI":  SBA_15_PEI_steps,
        "KIT6_PEI":  KIT6_PEI_steps,
        "HMS_PEI":  HMS_PEI_steps
    }

# ========================================================================================================================
# STEP 2: STEP YIELDS MATRICES
# ========================================================================================================================
    
    # Step yields for precursors
    
    step_yields = np.zeros((1, n, mcs_number))  # 3D array for MCS
    step_yields[:, np.r_[0, 5, 8, 12:16, 20, 30, 32], :] = Stirring_Yield
    step_yields[:, np.r_[10, 24, 33], :] = Evaporation_Yield
    step_yields[:, np.r_[1], :] = Heating_Yield
    step_yields[:, np.r_[2], :] = Grinding_Yield
    step_yields[:, np.r_[3, 21], :] = Filtration_Yield
    step_yields[:, np.r_[4, 25], :] = Washing_Yield 
    step_yields[:, np.r_[6], :] = Distillation_Yield
    step_yields[:, np.r_[7, 11, 26, 28, 34], :] = Drying_Yield
    step_yields[:, np.r_[18], :] = Aging_Yield
    step_yields[:, np.r_[19], :] = 0
    step_yields[:, np.r_[9], :] = Polymerization_Aziridine_Yield
    step_yields[:, np.r_[22], :] = 0
    step_yields[:, np.r_[23], :] = Centrifugation_Yield
    step_yields[:, np.r_[27], :] = Calcination_Yield
    step_yields[:, np.r_[29], :] = Activation_Yield
    step_yields[:, np.r_[16, 31], :] = Sonication_Yield
    step_yields[:, np.r_[17], :] = Aging_Yield  # Gelation step - should have high yield similar to aging
    step_yields[:, n-1, :] = 1 # Final Sorbent -> no yield apply 

    # Step yields for Solvents
    
    step_yields_s = np.zeros((1, n, mcs_number))  # 3D array for MCS
    step_yields_s[:, np.r_[0, 5, 8, 12:16, 20, 30, 32], :] = Stirring_Yield_s
    step_yields_s[:, np.r_[10, 24, 33], :] = Evaporation_Yield_s
    step_yields_s[:, np.r_[1], :] = Heating_Yield_s
    step_yields_s[:, np.r_[2], :] = Grinding_Yield_s
    step_yields_s[:, np.r_[3, 21], :] = Filtration_Yield_s
    step_yields_s[:, np.r_[4, 25], :] = Washing_Yield_s 
    step_yields_s[:, np.r_[6], :] = Distillation_Yield_s
    step_yields_s[:, np.r_[7, 11, 26, 28, 34], :] = Drying_Yield_s
    step_yields_s[:, np.r_[18], :] = Aging_Yield_s
    step_yields_s[:, np.r_[19], :] = 0
    step_yields_s[:, np.r_[9], :] = Polymerization_Aziridine_Yield_s
    step_yields_s[:, np.r_[22], :] = 0
    step_yields_s[:, np.r_[23], :] = Centrifugation_Yield_s
    step_yields_s[:, np.r_[27], :] = Calcination_Yield_s
    step_yields_s[:, np.r_[29], :] = Activation_Yield_s
    step_yields_s[:, np.r_[16, 31], :] = Sonication_Yield_s
    step_yields_s[:, np.r_[17], :] = Aging_Yield_s  # Gelation step - should have reasonable yield for solvents
    step_yields_s[:, n-1, :] = 1 # Final Sorbent -> no yield apply 
    
    # Step yields for Surfactants
    
    step_yields_sur = np.zeros((1, n, mcs_number))  # 3D array for MCS
    step_yields_sur[:, np.r_[0, 5, 8, 12:16, 20, 30, 32], :] = Stirring_Yield_sur
    step_yields_sur[:, np.r_[10, 24, 33], :] = Evaporation_Yield_sur
    step_yields_sur[:, np.r_[1], :] = Heating_Yield_sur
    step_yields_sur[:, np.r_[2], :] = Grinding_Yield_sur
    step_yields_sur[:, np.r_[3, 21], :] = Filtration_Yield_sur
    step_yields_sur[:, np.r_[4, 25], :] = Washing_Yield_sur 
    step_yields_sur[:, np.r_[6], :] = Distillation_Yield_sur
    step_yields_sur[:, np.r_[7, 11, 26, 28, 34], :] = Drying_Yield_sur
    step_yields_sur[:, np.r_[18], :] = Aging_Yield_sur
    step_yields_sur[:, np.r_[19], :] = 0
    step_yields_sur[:, np.r_[9], :] = Polymerization_Aziridine_Yield_sur
    step_yields_sur[:, np.r_[22], :] = 0
    step_yields_sur[:, np.r_[23], :] = Centrifugation_Yield_sur
    step_yields_sur[:, np.r_[27], :] = Calcination_Yield_sur
    step_yields_sur[:, np.r_[29], :] = Activation_Yield_sur
    step_yields_sur[:, np.r_[16, 31], :] = Sonication_Yield_sur
    step_yields_sur[:, np.r_[17], :] = Aging_Yield_sur  # Gelation step - should have reasonable yield for surfactants
    step_yields_sur[:, n-1, :] = 1 # Final Sorbent -> no yield apply 

    # Step yields for Other
    
    step_yields_other = np.zeros((1, n, mcs_number))  # 3D array for MCS
    step_yields_other[:, np.r_[0, 5, 8, 12:16, 20, 30, 32], :] = Stirring_Yield_other
    step_yields_other[:, np.r_[10, 24, 33], :] = Evaporation_Yield_other
    step_yields_other[:, np.r_[1], :] = Heating_Yield_other
    step_yields_other[:, np.r_[2], :] = Grinding_Yield_other
    step_yields_other[:, np.r_[3, 21], :] = Filtration_Yield_other
    step_yields_other[:, np.r_[4, 25], :] = Washing_Yield_other   
    step_yields_other[:, np.r_[6], :] = 1e-10  # Set to very small value instead of Distillation_Yield_other
    step_yields_other[:, np.r_[7, 11, 26, 28, 34], :] = Drying_Yield_other
    step_yields_other[:, np.r_[18], :] = Aging_Yield_other
    step_yields_other[:, np.r_[19], :] = 0
    step_yields_other[:, np.r_[9], :] = Polymerization_Aziridine_Yield_other
    step_yields_other[:, np.r_[22], :] = 0
    step_yields_other[:, np.r_[23], :] = Centrifugation_Yield_other
    step_yields_other[:, np.r_[27], :] = Calcination_Yield_other
    step_yields_other[:, np.r_[29], :] = Activation_Yield_other
    step_yields_other[:, np.r_[16, 31], :] = Sonication_Yield_other
    step_yields_other[:, np.r_[17], :] = Aging_Yield_other  # Gelation step - should have reasonable yield for other components
    step_yields_other[:, n-1, :] = 1 # Final Sorbent -> no yield apply 
    
# ========================================================================================================================
# STEP 3: SORBENT SYNTHESIS - MATERIAL FLOW ANALYSIS
# ========================================================================================================================

# PEI Mass Flow for all sorbent types

    PEI_mass = np.zeros((1, n, mcs_number)) # All steps
    PEI_mass[:, np.r_[30, 32:36], :] = 1 # All steps
    PEI_mass[:, n-1, :] = PEI_sorbent # PEI mass in final sorbent - mass scenarios 

    # Initialize dictionary to store results
    PEI_sorbent_dic = {}

    for sorbent_type , processes in sorbent_steps.items(): # iterate through all sorbent types
        PEI_Mass = processes * PEI_mass # Chemical processes for each sorbent type
        PEI_Mass = pf.back_prop(PEI_Mass, step_yields)  # Call back propagation function to calculate PEI mass in each step
        PEI_sorbent_dic[sorbent_type] = PEI_Mass # Store results

# Support Mass Flow for all sorbent types

    Support_mass = np.zeros((1, n, mcs_number)) # All steps
    Support_mass[:, np.r_[28:30, 31:36], :] = 1 # All steps
    Support_mass[:, n-1, :] = Support_sorbent # Support mass in final sorbent - mass scenarios
    
    # For each sorbent find the support mass in each step
    Support_sorbent_dic = {}
    
    for sorbent_type , processes in sorbent_steps.items(): # iterate through all sorbent types
        Support_Mass = processes * Support_mass # Chemical processes for each sorbent type
        Support_Mass = pf.back_prop(Support_Mass, step_yields)  # Call back propagation function to calculate PEI mass in each step
        Support_sorbent_dic[sorbent_type] = Support_Mass # Store results

# Solvent Mass Flow
    
    Solvent_mass = np.zeros((1,n, mcs_number)) # All steps
    Solvent_mass[:, np.r_[30:36], :] = 1 # Steps that is present
    
    # Dictionary with Solvent - PEI:solvent mass ratio in PEI dissolution for each sorbent type
   
    Solvent_dis_ratio = {
        "PEI_γ_alumina": 27.7,
        "PEI_SG": 15.8, 
        "MIL_101_Cr_PEI":  28, 
        "MCM_41_PEI":  2.5, 
        "SBA_15_PEI":  16.7, 
        "KIT6_PEI":  39.6, 
        "HMS_PEI":  31.5 
    }

    # Dictionary with Solvent - PEI:solvent mass ratio in PEI-Support mixture for each sorbent type
    
    Solvent_mix_ratio = {
        "PEI_γ_alumina": 0, 
        "PEI_SG": 0, 
        "MIL_101_Cr_PEI":  0, 
        "MCM_41_PEI":  0, 
        "SBA_15_PEI":  0, 
        "KIT6_PEI":  0, 
        "HMS_PEI":  0 
    }
    
    
    # For each sorbent find the solvent mass in each step
    Solvent_sorbent_dic = {}

    for sorbent_type, processes in sorbent_steps.items(): # iterate through all sorbent types
        Solvent_sorbent_mass = processes * Solvent_mass  # Chemical processes for each sorbent type

        # Extract required ratios and mass matrices
        ratio_dis = Solvent_dis_ratio[sorbent_type]
        ratio_mix = Solvent_mix_ratio[sorbent_type]
        PEI_sorbent_mass = PEI_sorbent_dic[sorbent_type]
        Support_sorbent_mass = Support_sorbent_dic[sorbent_type]
        Solvent_sorbent_mass[:, 30, :] = ratio_dis * PEI_sorbent_mass[:, 30, :] # Solvent in PEI dissolution

        # Find first non-zero index in support mass
        first_non_zero_support = np.where(Support_sorbent_mass[:, 31:, :] != 0)[0]
        non_zero = 31 + first_non_zero_support[0]

        # Find first non zero cell after cell 30
        Solvent_sorbent_mass[:, non_zero, :] = (Solvent_sorbent_mass[:, 30, :] * (step_yields_s[:, 30, :] / 100)) + ratio_mix * PEI_sorbent_mass[:, 30, :]
        Solvent_Mass = pf.front_prop(Solvent_sorbent_mass, step_yields_s) # Call front propagation function to calculate solvent mass in each step
        Solvent_sorbent_dic[sorbent_type] = Solvent_Mass # Store results

# Combined Sorbent Synthesis Results 
    
    Solvent_type = {
        "PEI_γ_alumina": "Methanol",
        "PEI_SG": "Methanol",
        "MIL_101_Cr_PEI":  "Methanol",
        "MCM_41_PEI":  "Methanol",
        "SBA_15_PEI":  "Methanol",
        "KIT6_PEI":  "Methanol",
        "HMS_PEI":  "Ethanol"
    }
    
    # Mapping of sorbent types to specific support names
    Support_names = {
        "PEI_γ_alumina": "PEI_γ_alumina",
        "PEI_SG": "PEI_SG",
        "MIL_101_Cr_PEI": "MIL_101_Cr_PEI",
        "MCM_41_PEI": "MCM_41_PEI",
        "SBA_15_PEI": "SBA_15_PEI",
        "KIT6_PEI": "KIT6_PEI",
        "HMS_PEI": "HMS_PEI"
    }
    
    Sorbent_results_dic = {}
    
    for sorbent_type in sorbent_steps.keys():  
        support_name = Support_names[sorbent_type]
        Sorbent_results_dic[sorbent_type] = {
        "Polyethyleneimine": PEI_sorbent_dic[sorbent_type],  
        support_name: Support_sorbent_dic[sorbent_type],  
        }
        
        solvent_name = Solvent_type[sorbent_type]
        Sorbent_results_dic[sorbent_type][solvent_name] = Solvent_sorbent_dic[sorbent_type]
    
    # Save results in dataframe

    Sorbent_results = []

    # Iterate over sorbent types and extract data
    for sorbent_type, component_matrices in Sorbent_results_dic.items():
        for component, mass_matrix in component_matrices.items():  
            for step in range(mass_matrix.shape[1]):  # Iterate over each step
                # Store results for first 10 MCS iterations only
                for mcs_iter in range(min(5, mass_matrix.shape[2])):  # Iterate over first 10 MCS iterations
                    mass = float(mass_matrix[:, step, mcs_iter])  # Extract mass for specific MCS iteration and convert to scalar
                    
                    Sorbent_results.append({
                        "Scenario": scenario,
                        "Sorbent": sorbent_type,
                        "Component": component,
                        "Step": step,
                        "MCS_Iteration": mcs_iter,
                        "Mass (g)": round(mass, 3)
                    })

    Sorbent_results_df = pd.DataFrame(Sorbent_results)
    
# ========================================================================================================================
# STEP 4: SUPPORT SYNTHESIS: Fraction of each chemical component in final support (Literature-based Data)
# ========================================================================================================================

# Creation of matrices for all chemical compounds 

    # Silica Gel (SG)
    
    Silica_mass_SG = np.zeros((1,n, mcs_number))
    Sodium_Silicate_mass_SG = np.zeros((1,n, mcs_number))
    Water_mass_SG = np.zeros((1,n, mcs_number))
    Sulphuric_Acid_mass_SG = np.zeros((1,n, mcs_number))
    Water_sulphuric_acid_mass_SG = np.zeros((1,n, mcs_number))
    Water_washing_mass_SG = np.zeros((1,n, mcs_number))
    Water_byproduct_mass_SG = np.zeros((1,n, mcs_number))
    Sodium_sulfate_byproduct_mass_SG = np.zeros((1,n, mcs_number))
    
    Silica_mass_SG[:, np.r_[17:19, 25:27, 28], :]=1
    Sodium_Silicate_mass_SG[:, np.r_[12], :]=1
    Water_mass_SG[:, np.r_[12, 17:19, 25:27, 28], :]=1
    Sulphuric_Acid_mass_SG[:, np.r_[12], :]=1
    Water_sulphuric_acid_mass_SG[:, np.r_[12, 17:19, 25:27, 28], :]=1
    Water_washing_mass_SG[:, np.r_[25:27, 28], :]=1
    Water_byproduct_mass_SG[:, np.r_[17:19, 25:27, 28], :]=1
    Sodium_sulfate_byproduct_mass_SG[:, np.r_[17:19, 25:27, 28], :]=1
    
    # γ - alumina (γ-Al2O3)
    
    Pseudoboehmite_mass_alumina = np.zeros((1,n, mcs_number))
    Nitric_Acid_mass_alumina = np.zeros((1,n, mcs_number))
    Water_nitric_acid_mass_alumina = np.zeros((1,n, mcs_number))
    Water_mass_alumina = np.zeros((1,n, mcs_number))
    Pluronic123_mass_alumina = np.zeros((1,n, mcs_number))
    Ethanol_mass_alumina = np.zeros((1,n, mcs_number))
    Nitrate_Ions_byproduct_mass_alumina = np.zeros((1,n, mcs_number))
    γ_alumina_mass_alumina = np.zeros((1,n, mcs_number))
    Water_byproduct_mass_alumina = np.zeros((1,n, mcs_number))
    
    Pseudoboehmite_mass_alumina[:, np.r_[16, 18, 20, 24, 26:28], :]=1
    Nitric_Acid_mass_alumina[:, np.r_[16], :]=1
    Water_nitric_acid_mass_alumina[:, np.r_[16, 18, 20, 24, 26:28, 29], :]=1
    Water_mass_alumina[:, np.r_[16, 18, 20, 24, 26:28, 29], :]=1
    Pluronic123_mass_alumina[:, np.r_[20, 24, 26:28, 29], :]=1
    Ethanol_mass_alumina[:, np.r_[20, 24, 26:28, 29], :]=1
    Nitrate_Ions_byproduct_mass_alumina[:, np.r_[18, 20, 24, 26:28, 29], :]=1
    γ_alumina_mass_alumina[:, np.r_[29], :]=1
    Water_byproduct_mass_alumina[:, np.r_[29], :]=1
    
    # MCM-41
    
    Silica_mass_MCM = np.zeros((1,n, mcs_number))
    Sodium_Oxide_mass_MCM = np.zeros((1,n, mcs_number))
    Water_mass_MCM = np.zeros((1,n, mcs_number))
    TMA_mass_MCM = np.zeros((1,n, mcs_number))
    CTMA_Br_mass_MCM = np.zeros((1,n, mcs_number))
    Water_washing_mass_MCM = np.zeros((1,n, mcs_number))
    
    Silica_mass_MCM[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    Sodium_Oxide_mass_MCM[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    Water_mass_MCM[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    TMA_mass_MCM[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    CTMA_Br_mass_MCM[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    Water_washing_mass_MCM[:, np.r_[25:28, 32], :]=1
  
    # MIL-101-Cr
    
    Water_mass_MIL_101_Cr = np.zeros((1,n, mcs_number))
    CrNO3_mass_MIL_101_Cr = np.zeros((1,n, mcs_number))
    H2BDC_mass_MIL_101_Cr = np.zeros((1,n, mcs_number))
    Acetic_Acid_mass_MIL_101_Cr = np.zeros((1,n, mcs_number))
    Water_acetic_acid_mass_MIL_101_Cr = np.zeros((1,n, mcs_number))
    Crystals_mass_MIL_101_Cr = np.zeros((1,n, mcs_number))
    MIL_101_Cr_mass = np.zeros((1,n, mcs_number))
    Methanol_Washing_mass_MIL_101_Cr = np.zeros((1,n, mcs_number))
    DMF_Washing_mass_MIL_101_Cr = np.zeros((1,n, mcs_number))
    
    Water_mass_MIL_101_Cr[:, np.r_[12, 18, 23, 25:27, 29], :]=1
    CrNO3_mass_MIL_101_Cr[:, np.r_[12, 18], :]=1
    H2BDC_mass_MIL_101_Cr[:, np.r_[12, 18], :]=1
    Acetic_Acid_mass_MIL_101_Cr[:, np.r_[18, 23, 25:27, 29], :]=1
    Water_acetic_acid_mass_MIL_101_Cr[:, np.r_[18, 23, 25:27, 29], :]=1
    Crystals_mass_MIL_101_Cr[:, np.r_[18], :]=1
    MIL_101_Cr_mass[:, np.r_[23, 25:27, 29], :]=1
    Methanol_Washing_mass_MIL_101_Cr[:, np.r_[25:27, 29], :]=1
    DMF_Washing_mass_MIL_101_Cr[:, np.r_[25:27, 29], :]=1
    
    # SBA-15
    
    HCl_mass_SBA = np.zeros((1,n, mcs_number))
    Water_HCl_mass_SBA = np.zeros((1,n, mcs_number))
    Water_mass_SBA = np.zeros((1,n, mcs_number))
    Pluronic123_mass_SBA = np.zeros((1,n, mcs_number))
    TEOS_mass_SBA = np.zeros((1,n, mcs_number))
    Water_washing_mass_SBA = np.zeros((1,n, mcs_number))
    Silica_mass_SBA = np.zeros((1,n, mcs_number))
    Ethanol_byproduct_mass_SBA = np.zeros((1,n, mcs_number))
    
    HCl_mass_SBA[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    Water_HCl_mass_SBA[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    Pluronic123_mass_SBA[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    TEOS_mass_SBA[:, np.r_[12], :]=1
    Water_mass_SBA[:, np.r_[12, 18, 21, 25:28, 32], :]=1
    Water_washing_mass_SBA[:, np.r_[25:28, 32], :]=1
    Silica_mass_SBA[:, np.r_[18, 21, 25:28, 32], :]=1
    Ethanol_byproduct_mass_SBA[:, np.r_[18, 21, 25:28, 32], :]=1
    
    # KIT6 
    
    Water_mass_KIT6 = np.zeros((1,n, mcs_number))
    HCl_mass_KIT6 = np.zeros((1,n, mcs_number))
    Water_HCl_mass_KIT6 = np.zeros((1,n, mcs_number))
    Pluronic123_mass_KIT6 = np.zeros((1,n, mcs_number))
    N_Butanol_mass_KIT6 = np.zeros((1,n, mcs_number))
    TEOS_mass_KIT6 = np.zeros((1,n, mcs_number))
    Silica_mass_KIT6 = np.zeros((1,n, mcs_number))
    Ethanol_byproduct_mass_KIT6 = np.zeros((1,n, mcs_number))
    
    Water_mass_KIT6[:, np.r_[12:15, 18, 21, 26:29], :]=1
    HCl_mass_KIT6[:, np.r_[12:15, 18, 21, 26:29], :]=1
    Water_HCl_mass_KIT6[:, np.r_[12:15, 18, 21, 26:29], :]=1
    Pluronic123_mass_KIT6[:, np.r_[12:15, 18, 21, 26:29], :]=1
    N_Butanol_mass_KIT6[:, np.r_[13:15, 18, 21, 26:29], :]=1
    TEOS_mass_KIT6[:, np.r_[14], :]=1
    Silica_mass_KIT6[:, np.r_[18, 21, 26:29], :]=1
    Ethanol_byproduct_mass_KIT6[:, np.r_[18, 21, 26:29], :]=1
    
    # HMS
    
    Water_mass_HMS = np.zeros((1,n, mcs_number))
    HCl_mass_HMS = np.zeros((1,n, mcs_number))
    Water_HCl_mass_HMS = np.zeros((1,n, mcs_number))
    Pluronic123_mass_HMS = np.zeros((1,n, mcs_number))
    TEOS_mass_HMS = np.zeros((1,n, mcs_number))
    Water_Washing_mass_HMS = np.zeros((1,n, mcs_number))
    Silica_mass_HMS = np.zeros((1,n, mcs_number))
    Ethanol_byproduct_mass_HMS = np.zeros((1,n, mcs_number))
    
    Water_mass_HMS[:, np.r_[12:14, 18, 21, 25:28, 32], :]=1
    HCl_mass_HMS[:, np.r_[12:14, 18, 21, 25:28, 32], :]=1
    Water_HCl_mass_HMS[:, np.r_[12:14, 18, 21, 25:28, 32], :]=1
    Pluronic123_mass_HMS[:, np.r_[12:14, 18, 21, 25:28, 32], :]=1
    TEOS_mass_HMS[:, np.r_[13], :]=1
    Water_Washing_mass_HMS[:, np.r_[25:28, 32], :]=1
    Silica_mass_HMS[:, np.r_[18, 21, 25:28, 32], :]=1
    Ethanol_byproduct_mass_HMS[:, np.r_[18, 21, 25:28, 32], :]=1
    
# Input Mass Data from literature for each chemical compound (in grams)

    # Silica Gel
    
    Silica_mass_SG[:, 17, :] = 1.011 # From stoichiometry, with ratio based on literature ratio of SiO2:Na2O equal to 2.5:1
    Sodium_Silicate_mass_SG[:, 12, :]=1.474
    Water_mass_SG[:, 12, :]=2.423
    Sulphuric_Acid_mass_SG[:, 12, :]=0.66 * 0.96 # 96% Sulphuric Acid
    Water_washing_mass_SG[:, 25, :]= 40
    Water_sulphuric_acid_mass_SG[:, 12, :] = 0.66 * 0.04
    Water_byproduct_mass_SG[:, 17, :] = 0.1212
    Sodium_sulfate_byproduct_mass_SG[:, 17, :] = 0.9559
    
    # γ - alumina (γ-Al2O3)
    
    Pseudoboehmite_mass_alumina[:, 16, :]=13.75
    Nitric_Acid_mass_alumina[:, 16, :]=1.27 * 0.7
    Water_nitric_acid_mass_alumina[:, 16, :]=1.27 * 0.3
    Water_mass_alumina[:, 16, :]=200
    Pluronic123_mass_alumina[:, 20, :]=15.3
    Ethanol_mass_alumina[:, 20, :]=157.8
    γ_alumina_mass_alumina[:, 29, :] = (Pseudoboehmite_mass_alumina[:, 16, :] * np.prod(step_yields[:,[16, 18, 20, 24, 26, 27, 28, 29], :]/(100), axis=1))*0.743
    Water_byproduct_mass_alumina[:, 29, :] = (Pseudoboehmite_mass_alumina[:, 16, :] * np.prod(step_yields[:,[16, 18, 20, 24, 26, 27, 28, 29], :]/(100), axis=1))*0.257
  
    # MCM-41
    
    Silica_mass_MCM[:, 12, :]=3.004
    Sodium_Oxide_mass_MCM[:, 12, :]=0.2678
    Water_mass_MCM[:, 12, :]=57.03
    TMA_mass_MCM[:, 12, :]=0.3507
    CTMA_Br_mass_MCM[:, 12, :]=5.69
    Water_washing_mass_MCM[:, 25, :] =  (
                                        Silica_mass_MCM[:,12, :] * np.prod(step_yields[:,[12, 18, 21], :]/(100), axis=1) + 
                                        Sodium_Oxide_mass_MCM[:, 12, :]*np.prod(step_yields_other[:,[12, 18, 21], :]/(100), axis=1) + 
                                        Water_mass_MCM[:, 12, :]*np.prod(step_yields_s[:,[12, 18, 21], :]/(100), axis=1) + 
                                        TMA_mass_MCM[:, 12, :] * np.prod(step_yields_sur[:,[12, 18, 21], :]/(100), axis=1) + 
                                        CTMA_Br_mass_MCM[:, 12, :]*np.prod(step_yields_sur[:,[12, 18, 21], :]/(100), axis=1)
                                        )  # 10 ml / g mixture - 10 g / g mixture
    
    # MIL-101-Cr

    Water_mass_MIL_101_Cr[:, 12, :] = 10
    CrNO3_mass_MIL_101_Cr[:, 12, :] = 0.8
    H2BDC_mass_MIL_101_Cr[:, 12, :] = 0.332
    Acetic_Acid_mass_MIL_101_Cr[:, 18, :] = 1.569*0.36 
    Water_acetic_acid_mass_MIL_101_Cr[:, 18, :] = 1.569*0.64 
    Crystals_mass_MIL_101_Cr[:, 18, :] = 0.005
    MIL_101_Cr_mass[:, 23, :] = ((CrNO3_mass_MIL_101_Cr[:, 12, :] /400.15)/3) * 717.4
    Methanol_Washing_mass_MIL_101_Cr[:, 25, :] = (
                                                Water_mass_MIL_101_Cr[:, 12, :]*np.prod(step_yields_s[:,[12, 18, 23], :]/(100), axis=1) + 
                                                MIL_101_Cr_mass[:, 23, :] +
                                                Acetic_Acid_mass_MIL_101_Cr[:, 18, :]*np.prod(step_yields_other[:,[18, 23], :]/(100), axis=1) +
                                                Water_acetic_acid_mass_MIL_101_Cr[:, 18, :]*np.prod(step_yields_s[:,[18, 23], :]/(100), axis=1)
                                                ) # 5 ml / g mixture - 3.955 g / g mixture
    DMF_Washing_mass_MIL_101_Cr[:, 25, :] = (Water_mass_MIL_101_Cr[:, 12, :]*np.prod(step_yields_s[:,[12, 18, 23], :]/(100), axis=1) + 
                                                MIL_101_Cr_mass[:, 23, :] +
                                                Acetic_Acid_mass_MIL_101_Cr[:, 18, :]*np.prod(step_yields_other[:,[18, 23], :]/(100), axis=1) +
                                                Water_acetic_acid_mass_MIL_101_Cr[:, 18, :]*np.prod(step_yields_s[:,[18, 23], :]/(100), axis=1)
                                                )  # 5 ml / g mixture - 4.72 g / g mixture
    
    # SBA-15

    HCl_mass_SBA[:, 12, :] = 207.8 *0.37 
    Water_HCl_mass_SBA[:, 12, :] = 207.8 *0.63 
    Pluronic123_mass_SBA[:, 12, :] = 98.6
    TEOS_mass_SBA[:, 12, :] = 60.08
    Water_mass_SBA[:, 12, :] = 3451.8
    Silica_mass_SBA[:, 18, :] = 17.42
    Ethanol_byproduct_mass_SBA[:, 18, :] = 53.44
    Water_washing_mass_SBA[:, 25, :] = (
                                        HCl_mass_SBA[:, 12, :]*np.prod(step_yields_other[:,[12, 18, 21], :]/(100), axis=1) + 
                                        Water_HCl_mass_SBA[:, 12, :]*np.prod(step_yields_s[:,[12, 18, 21], :]/(100), axis=1) + 
                                        Pluronic123_mass_SBA[:, 12, :] * np.prod(step_yields_sur[:,[12, 18, 21], :]/(100), axis=1) + 
                                        Silica_mass_SBA[:, 18, :] *  np.prod(step_yields[:,[18, 21], :]/(100), axis=1) + 
                                        Ethanol_byproduct_mass_SBA[:, 18, :] * np.prod(step_yields_s[:,[18, 21], :]/(100), axis=1) +
                                        Water_mass_SBA[:, 12, :]*np.prod(step_yields_s[:,[12, 18, 21], :]/(100), axis=1)
                                        ) # 10 ml / g mixture - 10 g / g mixture
    
    # KIT6 
    
    Water_mass_KIT6[:, 12, :] = 144
    HCl_mass_KIT6[:, 12, :] = 7.9*0.37
    Water_HCl_mass_KIT6[:, 12, :] = 7.9*0.63
    Pluronic123_mass_KIT6[:, 12, :] = 4
    N_Butanol_mass_KIT6[:, 13, :] = 4
    TEOS_mass_KIT6[:, 14, :] = 8.6
    Silica_mass_KIT6[:, 18, :] = 2.78
    Ethanol_byproduct_mass_KIT6[:, 18, :] = 7.61
    
    # HMS

    Water_mass_HMS[:, 12, :] = 38
    HCl_mass_HMS[:, 12, :] = 6.15*0.37
    Water_HCl_mass_HMS[:, 12, :] = 6.15*0.63
    Pluronic123_mass_HMS[:, 12, :] = 2
    TEOS_mass_HMS[:, 13, :] = 4.28
    Silica_mass_HMS[:, 18, :] = 1.23
    Ethanol_byproduct_mass_HMS[:, 18, :] = 3.78
    Water_Washing_mass_HMS[:, 25, :] = (
                                        Water_mass_HMS[:, 12, :]*np.prod(step_yields_s[:,[12, 13, 18, 21], :]/(100), axis=1) + 
                                        HCl_mass_HMS[:, 12, :]*np.prod(step_yields_other[:,[12, 13, 18, 21], :]/(100), axis=1) + 
                                        Water_HCl_mass_HMS[:, 12, :]*np.prod(step_yields_s[:,[12, 13, 18, 21], :]/(100), axis=1) + 
                                        Pluronic123_mass_HMS[:, 12, :]*np.prod(step_yields_sur[:,[12, 13, 18, 21], :]/(100), axis=1)+ 
                                        Silica_mass_HMS[:, 18, :]*np.prod(step_yields[:,[18, 21], :]/(100), axis=1) +
                                        Ethanol_byproduct_mass_HMS[:, 18, :]*np.prod(step_yields[:,[18, 21], :]/(100), axis=1)
                                        ) # 10 ml / g mixture - 10 g / g mixture
    
# Grouping of chemical components based on their type/action
    
    Support_Monomers = {
        "PEI_γ_alumina": {'Pseudo': Pseudoboehmite_mass_alumina,
                          'γ-Alumina': γ_alumina_mass_alumina},
        "PEI_SG": { 'Sodium Silicate': Sodium_Silicate_mass_SG,
                    'Silica': Silica_mass_SG},
        "MCM_41_PEI":  {'Silica': Silica_mass_MCM},
        'MIL_101_Cr_PEI': {'CrNO3': CrNO3_mass_MIL_101_Cr, 
                           'H2BDC': H2BDC_mass_MIL_101_Cr,
                           'MIL-101 (Cr) crystals': Crystals_mass_MIL_101_Cr,
                           'MIL-101 (Cr)': MIL_101_Cr_mass},
        'SBA_15_PEI': {'TEOS': TEOS_mass_SBA,
                       'Silica': Silica_mass_SBA},      
        'KIT6_PEI': {'TEOS': TEOS_mass_KIT6,
                     'Silica': Silica_mass_KIT6},
        'HMS_PEI': {'TEOS': TEOS_mass_HMS,
                    'Silica': Silica_mass_HMS}
        }
    
    Support_Surfactants = {
        "PEI_γ_alumina": {'Pluronic P123': Pluronic123_mass_alumina},
        "MCM_41_PEI":  {'CTMA-Br': CTMA_Br_mass_MCM},
        'SBA_15_PEI': {'Pluronic P123': Pluronic123_mass_SBA},      
        'KIT6_PEI': {'Pluronic P123': Pluronic123_mass_KIT6},
        'HMS_PEI': {'Pluronic P123': Pluronic123_mass_HMS}
        }
    
    Support_Solvents = {
        "PEI_γ_alumina": {'DI Water': Water_mass_alumina,  
                     'Ethanol': Ethanol_mass_alumina,
                     'Water Byproduct': Water_byproduct_mass_alumina,
                     'Water Nitric Acid': Water_nitric_acid_mass_alumina
                     },
        "PEI_SG": {'DI Water': Water_mass_SG, 
                     'Water Washing': Water_washing_mass_SG,
                     'Water Byproduct': Water_byproduct_mass_SG,
                     'Water Sulfuric Acid': Water_sulphuric_acid_mass_SG},
        "MCM_41_PEI":  {'DI Water': Water_mass_MCM, 
                     'Water Washing': Water_washing_mass_MCM},
        'MIL_101_Cr_PEI': {'DI Water': Water_mass_MIL_101_Cr,
                           'Methanol Washing': Methanol_Washing_mass_MIL_101_Cr,
                           'DMF Washing': DMF_Washing_mass_MIL_101_Cr,
                           'Water Acetic Acid': Water_acetic_acid_mass_MIL_101_Cr},
        'SBA_15_PEI': {'DI Water': Water_mass_SBA, 
                       'Water Washing': Water_washing_mass_SBA,
                       'Ethanol Byproduct': Ethanol_byproduct_mass_SBA,
                       'Water HCl': Water_HCl_mass_SBA},
        'KIT6_PEI': {'DI Water': Water_mass_KIT6, 
                    'N-Butanol': N_Butanol_mass_KIT6,
                    'Ethanol Byproduct': Ethanol_byproduct_mass_KIT6,
                    'Water HCl': Water_HCl_mass_KIT6}, 
        'HMS_PEI': {'DI Water': Water_mass_HMS, 
                    'Water Washing': Water_Washing_mass_HMS,
                    'Ethanol Byproduct': Ethanol_byproduct_mass_HMS,
                    'Water HCl': Water_HCl_mass_HMS}
        }
    
    Support_Other = {
        "PEI_γ_alumina": {'Nitric Acid': Nitric_Acid_mass_alumina},
        "PEI_SG": {'Sodium Sulfate': Sodium_sulfate_byproduct_mass_SG,
                   'Sulfuric Acid': Sulphuric_Acid_mass_SG},
        "MCM_41_PEI":  {'Sodium Oxide': Sodium_Oxide_mass_MCM,
                        'TMA': TMA_mass_MCM},
        'MIL_101_Cr_PEI': {'Acetic Acid': Acetic_Acid_mass_MIL_101_Cr}, 
        'SBA_15_PEI': {'HCl': HCl_mass_SBA}, 
        'KIT6_PEI': {'HCl': HCl_mass_KIT6},
        'HMS_PEI': {'HCl': HCl_mass_HMS}
        }


# Literature - based mass flow analysis 
   
    Support_results_lit = [] # store results in list to save to df
    Support_results_lit_dic = {}  # store results in dictionary to access later

    # Iterate over each sorbent type
    for sorbent_type in Support_Monomers.keys():

        # Collect grouped component dictionaries
        monomers = Support_Monomers.get(sorbent_type, {})
        solvents = Support_Solvents.get(sorbent_type, {})
        surfactants_salts = Support_Surfactants.get(sorbent_type, {})
        other = Support_Other.get(sorbent_type, {})

        chemical_groups = {
        "Monomers": monomers,
        "Solvents": solvents,
        "Surfactants": surfactants_salts,
        "Other": other
        }

        # Initialize dictionary for current sorbent
        Support_results_lit_dic[sorbent_type] = {}

        # Iterate across all groups and components
        for group_name, components in chemical_groups.items():
            for comp, mass_matrix in components.items():

                # Apply front propagation - call front_prop function for each chemical component
                if group_name == "Monomers":
                    updated_matrix = pf.front_prop(mass_matrix, step_yields)
                elif group_name == "Solvents":
                    updated_matrix = pf.front_prop(mass_matrix, step_yields_s)
                elif group_name == "Surfactants":
                    updated_matrix = pf.front_prop(mass_matrix, step_yields_sur)
                elif group_name == "Other":
                    updated_matrix = pf.front_prop(mass_matrix, step_yields_other)

                # Store results in dictionary
                Support_results_lit_dic[sorbent_type][comp] = updated_matrix

                # Store results in dataframe
                for step in range(updated_matrix.shape[1]):
                    # Store results for first 10 MCS iterations only
                    for mcs_iter in range(min(5, updated_matrix.shape[2])):
                        mass = float(updated_matrix[:, step, mcs_iter])
                        Support_results_lit.append({
                "Scenario": scenario,
                "Component": comp,
                "Type": group_name,
                "Step": step,
                "MCS_Iteration": mcs_iter,
                "Mass (g)": round(mass, 3)
                    })

        # Create final DataFrame
        Support_lit_Df = pd.DataFrame(Support_results_lit)

# ========================================================================================================================
# STEP 5: SUPPORT SYNTHESIS: SCENARIO-BASED MASS FLOW ANALYSIS 
# ========================================================================================================================

# Find the fraction of each chemical compound in the final support

# Silica Gel
    
    # Mass of each component in final support
    
    Silica_SG = Support_results_lit_dic['PEI_SG']['Silica']
    Silica_SG_mass = Silica_SG[:, 28, :]
    Water_SG = Support_results_lit_dic['PEI_SG']['DI Water']
    Water_SG_mass = Water_SG[:, 28, :]
    Water_Washing_SG = Support_results_lit_dic['PEI_SG']['Water Washing']
    Water_Washing_SG_mass = Water_Washing_SG[:, 28, :]
    Water_Sulfuric_SG = Support_results_lit_dic['PEI_SG']['Water Sulfuric Acid']
    Water_Sulfuric_SG_mass = Water_Sulfuric_SG[:, 28, :]
    Water_byproduct_SG = Support_results_lit_dic['PEI_SG']['Water Byproduct']
    Water_byproduct_SG_mass = Water_byproduct_SG[:, 28, :]
    Sodium_Silicate_SG = Support_results_lit_dic['PEI_SG']['Sodium Silicate']
    Sodium_Silicate_SG_mass = Sodium_Silicate_SG[:, 12, :]
    Sulfuric_Acid_SG = Support_results_lit_dic['PEI_SG']['Sulfuric Acid']
    Sulfuric_Acid_SG_mass = Sulfuric_Acid_SG[:, 12, :]
    Sodium_sulfate_SG = Support_results_lit_dic['PEI_SG']['Sodium Sulfate']
    Sodium_sulfate_SG_mass = Sodium_sulfate_SG[:, 28, :]
    
    # Mass of final support
    
    SG_Total_mass = Silica_SG_mass + Water_SG_mass + Water_Washing_SG_mass + Water_Sulfuric_SG_mass + Water_byproduct_SG_mass + Sodium_sulfate_SG_mass
    
    # Mass Fraction of each component in final support
    
    Silica_SG_fraction = Silica_SG_mass/SG_Total_mass
    Water_SG_fraction = Water_SG_mass/SG_Total_mass
    Water_Sulfuric_SG_fraction = Water_Sulfuric_SG_mass/SG_Total_mass
    Water_Washing_SG_fraction = Water_Washing_SG_mass/SG_Total_mass
    Water_byproduct_SG_fraction = Water_byproduct_SG_mass/SG_Total_mass
    Sodium_Silicate_SG_fraction = Sodium_Silicate_SG_mass/SG_Total_mass
    Sodium_Sulfate_SG_fraction = Sodium_sulfate_SG_mass/SG_Total_mass
    Sulfuric_Acid_SG_fraction = Sulfuric_Acid_SG_mass/SG_Total_mass
  
# γ - Alumina
    
    # Mass of each component in final support
    
    Pseudo_alumina = Support_results_lit_dic['PEI_γ_alumina']['Pseudo']
    Pseudo_alumina_mass = Pseudo_alumina[:, 27, :]
    Water_alumina = Support_results_lit_dic['PEI_γ_alumina']['DI Water']
    Water_alumina_mass = Water_alumina[:, 29, :]
    Ethanol_alumina = Support_results_lit_dic['PEI_γ_alumina']['Ethanol']
    Ethanol_alumina_mass = Ethanol_alumina[:, 29, :]
    Pluronic_alumina = Support_results_lit_dic['PEI_γ_alumina']['Pluronic P123']
    Pluronic_alumina_mass = Pluronic_alumina[:, 29, :]
    Nitric_Acid_alumina = Support_results_lit_dic['PEI_γ_alumina']['Nitric Acid']
    Nitric_Acid_alumina_mass = Nitric_Acid_alumina[:, 16, :]
    Water_nitric_acid_alumina = Support_results_lit_dic['PEI_γ_alumina']['Water Nitric Acid']
    Water_nitric_acid_alumina_mass = Water_nitric_acid_alumina[:, 29, :]
    γ_alumina_alumina = Support_results_lit_dic['PEI_γ_alumina']['γ-Alumina']
    γ_alumina_alumina_mass = γ_alumina_alumina[:,29, :]
    Water_byproduct_alumina = Support_results_lit_dic['PEI_γ_alumina']['Water Byproduct']
    Water_byproduct_alumina_mass = Water_byproduct_alumina[:, 29, :]
    
    # Mass of final support
    
    Alumina_Total_mass = γ_alumina_alumina_mass + Water_alumina_mass + Ethanol_alumina_mass + Pluronic_alumina_mass + Water_nitric_acid_alumina_mass
    
    # Mass Fraction of each component in final support
    
    Pseudo_alumina_fraction = Pseudo_alumina_mass/Alumina_Total_mass
    Water_alumina_fraction = Water_alumina_mass/Alumina_Total_mass
    Ethanol_alumina_fraction = Ethanol_alumina_mass/Alumina_Total_mass
    Pluronic_alumina_fraction = Pluronic_alumina_mass/Alumina_Total_mass
    Nitric_Acid_alumina_fraction = Nitric_Acid_alumina_mass/Alumina_Total_mass
    Water_nitric_acid_alumina_fraction = Water_nitric_acid_alumina_mass/Alumina_Total_mass
    Water_byproduct_alumina_fraction = Water_byproduct_alumina_mass/Alumina_Total_mass
    γ_alumina_alumina_fraction = γ_alumina_alumina_mass/Alumina_Total_mass
    
# MCM-41
    
    # Mass of each component in final support
    
    Silica_MCM = Support_results_lit_dic['MCM_41_PEI']['Silica']
    Silica_MCM_mass = Silica_MCM[:, 32, :]
    Water_MCM = Support_results_lit_dic['MCM_41_PEI']['DI Water']
    Water_MCM_mass = Water_MCM[:, 32, :]
    Water_Washing_MCM = Support_results_lit_dic['MCM_41_PEI']['Water Washing']
    Water_Washing_MCM_mass = Water_Washing_MCM[:, 32, :]
    Sodium_Oxide_MCM = Support_results_lit_dic['MCM_41_PEI']['Sodium Oxide']
    Sodium_Oxide_MCM_mass = Sodium_Oxide_MCM[:, 32, :]
    TMA_MCM = Support_results_lit_dic['MCM_41_PEI']['TMA']
    TMA_MCM_mass = TMA_MCM[:, 32, :]
    CTMA_MCM = Support_results_lit_dic['MCM_41_PEI']['CTMA-Br']
    CTMA_MCM_mass = CTMA_MCM[:, 32, :]
    
    # Mass of final support
    
    MCM_Total_mass = Silica_MCM_mass + Water_MCM_mass + Water_Washing_MCM_mass + Sodium_Oxide_MCM_mass + TMA_MCM_mass + CTMA_MCM_mass
    
    # Mass Fraction of each component in final support
    
    Silica_MCM_fraction = Silica_MCM_mass/MCM_Total_mass
    Water_MCM_fraction = Water_MCM_mass/MCM_Total_mass
    Water_Washing_MCM_fraction = Water_Washing_MCM_mass/MCM_Total_mass
    Sodium_Oxide_MCM_fraction = Sodium_Oxide_MCM_mass/MCM_Total_mass
    TMA_MCM_fraction = TMA_MCM_mass/MCM_Total_mass
    CTMA_MCM_fraction = CTMA_MCM_mass/MCM_Total_mass
        

# MIL-101-Cr
    
    # Mass of each component in final support
    
    CrNO3_MIL_101_Cr = Support_results_lit_dic['MIL_101_Cr_PEI']['CrNO3']
    CrNO3_MIL_101_Cr_mass = CrNO3_MIL_101_Cr[:, 18, :]
    H2BDC_MIL_101_Cr = Support_results_lit_dic['MIL_101_Cr_PEI']['H2BDC']
    H2BDC_MIL_101_Cr_mass = H2BDC_MIL_101_Cr[:, 18, :]
    Crystals_MIL_101_Cr = Support_results_lit_dic['MIL_101_Cr_PEI']['MIL-101 (Cr) crystals']
    Crystals_MIL_101_Cr_mass = Crystals_MIL_101_Cr[:, 18, :]
    Water_MIL_101_Cr = Support_results_lit_dic['MIL_101_Cr_PEI']['DI Water']
    Water_MIL_101_Cr_mass = Water_MIL_101_Cr[:, 29, :]
    Methanol_Washing_MIL_101_Cr = Support_results_lit_dic['MIL_101_Cr_PEI']['Methanol Washing']
    Methanol_Washing_MIL_101_Cr_mass = Methanol_Washing_MIL_101_Cr[:, 29, :]
    DMF_Washing_MIL_101_Cr = Support_results_lit_dic['MIL_101_Cr_PEI']['DMF Washing']
    DMF_Washing_MIL_101_Cr_mass =DMF_Washing_MIL_101_Cr[:, 29, :] 
    Acetic_Acid_MIL_101_Cr = Support_results_lit_dic['MIL_101_Cr_PEI']['Acetic Acid']
    Acetic_Acid_MIL_101_Cr_mass = Acetic_Acid_MIL_101_Cr[:, 29, :]
    Water_Acetic_Acid_MIL_101_Cr = Support_results_lit_dic['MIL_101_Cr_PEI']['Water Acetic Acid']
    Water_Acetic_Acid_MIL_101_Cr_mass = Water_Acetic_Acid_MIL_101_Cr[:, 29, :]
    MIL_101_Cr_mass = Support_results_lit_dic['MIL_101_Cr_PEI']['MIL-101 (Cr)']
    MIL_101_Cr_mass_mass = MIL_101_Cr_mass[:, 29, :]

    # Mass of final support
    
    MIL_101_Cr_Total_mass = MIL_101_Cr_mass_mass + Water_MIL_101_Cr_mass + Methanol_Washing_MIL_101_Cr_mass + DMF_Washing_MIL_101_Cr_mass + Acetic_Acid_MIL_101_Cr_mass + Water_Acetic_Acid_MIL_101_Cr_mass
  
    CrNO3_MIL_101_Cr_fraction = CrNO3_MIL_101_Cr_mass/MIL_101_Cr_Total_mass
    H2BDC_MIL_101_Cr_fraction = H2BDC_MIL_101_Cr_mass/MIL_101_Cr_Total_mass
    Crystals_MIL_101_Cr_fraction = Crystals_MIL_101_Cr_mass/MIL_101_Cr_Total_mass
    Water_MIL_101_Cr_fraction = Water_MIL_101_Cr_mass/MIL_101_Cr_Total_mass
    Methanol_Washing_MIL_101_Cr_fraction = Methanol_Washing_MIL_101_Cr_mass/MIL_101_Cr_Total_mass
    DMF_Washing_MIL_101_Cr_fraction = DMF_Washing_MIL_101_Cr_mass/MIL_101_Cr_Total_mass
    Acetic_Acid_MIL_101_Cr_fraction = Acetic_Acid_MIL_101_Cr_mass/MIL_101_Cr_Total_mass
    MIL_101_Cr_fraction = MIL_101_Cr_mass_mass/MIL_101_Cr_Total_mass
    Water_Acetic_Acid_MIL_101_Cr_fraction = Water_Acetic_Acid_MIL_101_Cr_mass/MIL_101_Cr_Total_mass

# SBA-15

    # Mass of each component in final support
    
    TEOS_SBA_15 = Support_results_lit_dic['SBA_15_PEI']['TEOS']
    TEOS_SBA_15_mass = TEOS_SBA_15[:, 12, :]
    Pluronic_SBA_15 = Support_results_lit_dic['SBA_15_PEI']['Pluronic P123']
    Pluronic_SBA_15_mass = Pluronic_SBA_15[:, 32, :]
    HCl_SBA_15 = Support_results_lit_dic['SBA_15_PEI']['HCl']
    HCl_SBA_15_mass = HCl_SBA_15[:, 32, :]
    Water_HCl_SBA_15 = Support_results_lit_dic['SBA_15_PEI']['Water HCl']
    Water_HCl_SBA_15_mass = Water_HCl_SBA_15[:, 32, :]
    Water_SBA_15 = Support_results_lit_dic['SBA_15_PEI']['DI Water']
    Water_SBA_15_mass = Water_SBA_15[:, 32, :]
    Silica_SBA_15 = Support_results_lit_dic['SBA_15_PEI']['Silica']
    Silica_SBA_15_mass = Silica_SBA_15[:, 32, :]
    Ethanol_byproduct_SBA_15 = Support_results_lit_dic['SBA_15_PEI']['Ethanol Byproduct']
    Ethanol_byproduct_SBA_15_mass = Ethanol_byproduct_SBA_15[:, 32, :]
    Water_washing_SBA_15 = Support_results_lit_dic['SBA_15_PEI']['Water Washing']
    Water_washing_SBA_15_mass = Water_washing_SBA_15[:, 32, :]

    SBA_15_Total_mass = Silica_SBA_15_mass + Pluronic_SBA_15_mass + HCl_SBA_15_mass + Water_SBA_15_mass + Ethanol_byproduct_SBA_15_mass + Water_washing_SBA_15_mass + Water_HCl_SBA_15_mass
    
    TEOS_SBA_15_fraction = TEOS_SBA_15_mass/SBA_15_Total_mass
    Pluronic_SBA_15_fraction = Pluronic_SBA_15_mass/SBA_15_Total_mass
    HCl_SBA_15_fraction = HCl_SBA_15_mass/SBA_15_Total_mass
    Water_SBA_15_fraction = Water_SBA_15_mass/SBA_15_Total_mass
    Silica_SBA_15_fraction = Silica_SBA_15_mass/SBA_15_Total_mass
    Ethanol_byproduct_SBA_15_fraction = Ethanol_byproduct_SBA_15_mass/SBA_15_Total_mass
    Water_washing_SBA_15_fraction = Water_washing_SBA_15_mass/SBA_15_Total_mass
    Water_HCl_SBA_15_fraction = Water_HCl_SBA_15_mass/SBA_15_Total_mass

# KIT6

    # Mass of each component in final support
    
    TEOS_KIT6 = Support_results_lit_dic['KIT6_PEI']['TEOS']
    TEOS_KIT6_mass = TEOS_KIT6[:, 14, :]
    Pluronic_KIT6 = Support_results_lit_dic['KIT6_PEI']['Pluronic P123']
    Pluronic_KIT6_mass = Pluronic_KIT6[:, 28, :]
    HCl_KIT6 = Support_results_lit_dic['KIT6_PEI']['HCl']
    HCl_KIT6_mass = HCl_KIT6[:, 28, :]
    Water_HCl_KIT6 = Support_results_lit_dic['KIT6_PEI']['Water HCl']
    Water_HCl_KIT6_mass = Water_HCl_KIT6[:, 28, :]
    Water_KIT6 = Support_results_lit_dic['KIT6_PEI']['DI Water']
    Water_KIT6_mass = Water_KIT6[:, 28, :]
    N_butanol_KIT6 = Support_results_lit_dic['KIT6_PEI']['N-Butanol']
    N_butanol_KIT6_mass = N_butanol_KIT6[:, 28, :]
    Silica_KIT6 = Support_results_lit_dic['KIT6_PEI']['Silica']
    Silica_KIT6_mass = Silica_KIT6[:, 28, :]
    Ethanol_byproduct_KIT6 = Support_results_lit_dic['KIT6_PEI']['Ethanol Byproduct']
    Ethanol_byproduct_KIT6_mass = Ethanol_byproduct_KIT6[:, 28, :]
   
    # Mass of final support
    
    KIT6_Total_mass = Silica_KIT6_mass + Pluronic_KIT6_mass + HCl_KIT6_mass + Water_KIT6_mass + N_butanol_KIT6_mass + Ethanol_byproduct_KIT6_mass + Water_HCl_KIT6_mass

    # Mass of each component in final support
    
    TEOS_KIT6_fraction =  TEOS_KIT6_mass/KIT6_Total_mass
    Pluronic_KIT6_fraction =  Pluronic_KIT6_mass/KIT6_Total_mass
    HCl_KIT6_fraction =  HCl_KIT6_mass/KIT6_Total_mass
    Water_KIT6_fraction =  Water_KIT6_mass/KIT6_Total_mass
    N_butanol_KIT6_fraction =  N_butanol_KIT6_mass/KIT6_Total_mass
    Silica_KIT6_fraction = Silica_KIT6_mass/KIT6_Total_mass
    Ethanol_byproduct_KIT6_fraction = Ethanol_byproduct_KIT6_mass/KIT6_Total_mass
    Water_HCl_KIT6_fraction = Water_HCl_KIT6_mass/KIT6_Total_mass

# HMS

    # Mass of each component in final support
    
    TEOS_HMS = Support_results_lit_dic['HMS_PEI']['TEOS']
    TEOS_HMS_mass = TEOS_HMS[:, 13, :]
    Pluronic_HMS = Support_results_lit_dic['HMS_PEI']['Pluronic P123']
    Pluronic_HMS_mass = Pluronic_HMS[:, 32, :]
    Water_HMS = Support_results_lit_dic['HMS_PEI']['DI Water']
    Water_HMS_mass = Water_HMS[:, 32, :]
    Water_Washing_HMS = Support_results_lit_dic['HMS_PEI']['Water Washing']
    Water_Washing_HMS_mass = Water_Washing_HMS[:, 32, :]
    HCl_HMS = Support_results_lit_dic['HMS_PEI']['HCl']
    HCl_HMS_mass = HCl_HMS[:, 32, :]
    Silica_HMS = Support_results_lit_dic['HMS_PEI']['Silica']
    Silica_HMS_mass = Silica_HMS[:, 32, :]
    Ethanol_byproduct_HMS = Support_results_lit_dic['HMS_PEI']['Ethanol Byproduct']
    Ethanol_byproduct_HMS_mass = Ethanol_byproduct_HMS[:, 32, :]
    Water_HCl_HMS = Support_results_lit_dic['HMS_PEI']['Water HCl']
    Water_HCl_HMS_mass = Water_HCl_HMS[:, 32, :]

    # Mass of final support
    
    HMS_Total_mass = Silica_HMS_mass + Pluronic_HMS_mass + Water_HMS_mass + Water_Washing_HMS_mass + HCl_HMS_mass + Ethanol_byproduct_HMS_mass + Water_HCl_HMS_mass

    # Mass of each component in final support
    
    TEOS_HMS_fraction = TEOS_HMS_mass/HMS_Total_mass
    Pluronic_HMS_fraction = Pluronic_HMS_mass/HMS_Total_mass
    Water_HMS_fraction = Water_HMS_mass/HMS_Total_mass
    Water_Washing_HMS_fraction = Water_Washing_HMS_mass/HMS_Total_mass
    HCl_HMS_fraction = HCl_HMS_mass/HMS_Total_mass
    Silica_HMS_fraction = Silica_HMS_mass/HMS_Total_mass
    Ethanol_byproduct_HMS_fraction = Ethanol_byproduct_HMS_mass/HMS_Total_mass
    Water_HCl_HMS_fraction = Water_HCl_HMS_mass/HMS_Total_mass

# Dictionary with chemical component fractions
    
    Support_fractions = {
        'PEI_SG': {'DI Water': Water_SG_fraction, 
                   'Sodium Silicate': Sodium_Silicate_SG_fraction,  
                   'Silica': Silica_SG_fraction, 
                   'Water Washing': Water_Washing_SG_fraction,
                   'Water Sulfuric Acid':Water_Sulfuric_SG_fraction,
                   'Water Byproduct': Water_byproduct_SG_fraction,
                   'Sulfuric Acid': Sulfuric_Acid_SG_fraction,
                   'Sodium Sulfate': Sodium_Sulfate_SG_fraction},
        'MCM_41_PEI': {'DI Water': Water_MCM_fraction, 
                   'Sodium Oxide': Sodium_Oxide_MCM_fraction,  
                   'Silica': Silica_MCM_fraction, 
                   'Water Washing': Water_Washing_MCM_fraction, 
                   'TMA': TMA_MCM_fraction,
                   'CTMA-Br': CTMA_MCM_fraction},
        'PEI_γ_alumina': {'DI Water': Water_alumina_fraction, 
                          'Pluronic P123': Pluronic_alumina_fraction,  
                          'Pseudo': Pseudo_alumina_fraction, 
                          'Ethanol': Ethanol_alumina_fraction, 
                          'Nitric Acid': Nitric_Acid_alumina_fraction,
                          'γ-Alumina': γ_alumina_alumina_fraction,
                          'Water Byproduct': Water_byproduct_alumina_fraction,
                          'Water Nitric Acid': Water_nitric_acid_alumina_fraction},
        'MIL_101_Cr_PEI': {'DI Water': Water_MIL_101_Cr_fraction,
                           'CrNO3': CrNO3_MIL_101_Cr_fraction, 
                           'H2BDC': H2BDC_MIL_101_Cr_fraction, 
                           'Acetic Acid': Acetic_Acid_MIL_101_Cr_fraction, 
                           'MIL-101 (Cr) crystals': Crystals_MIL_101_Cr_fraction,
                           'Methanol Washing': Methanol_Washing_MIL_101_Cr_fraction,
                           'DMF Washing': DMF_Washing_MIL_101_Cr_fraction,
                           'MIL-101 (Cr)': MIL_101_Cr_fraction,
                           'Water Acetic Acid': Water_Acetic_Acid_MIL_101_Cr_fraction},
        'SBA_15_PEI': {'HCl': HCl_SBA_15_fraction , 
                    'Pluronic P123': Pluronic_SBA_15_fraction , 
                    'TEOS': TEOS_SBA_15_fraction,
                    'DI Water': Water_SBA_15_fraction,
                    'Silica': Silica_SBA_15_fraction,
                    'Ethanol Byproduct': Ethanol_byproduct_SBA_15_fraction,
                    'Water Washing': Water_washing_SBA_15_fraction,
                    'Water HCl': Water_HCl_SBA_15_fraction},
        'KIT6_PEI': {'TEOS': TEOS_KIT6_fraction,
                      'DI Water': Water_KIT6_fraction,
                      'Silica': Silica_KIT6_fraction,
                      'Ethanol Byproduct': Ethanol_byproduct_KIT6_fraction,
                      'Pluronic P123': Pluronic_KIT6_fraction,
                      'HCl': HCl_KIT6_fraction,
                      'N-Butanol': N_butanol_KIT6_fraction,
                      'Water HCl': Water_HCl_KIT6_fraction},
        'HMS_PEI': {'DI Water': Water_HMS_fraction , 
                    'HCl': HCl_HMS_fraction , 
                    'Pluronic P123': Pluronic_HMS_fraction , 
                    'TEOS': TEOS_HMS_fraction ,
                    'Water Washing': Water_Washing_HMS_fraction,
                    'Silica': Silica_HMS_fraction,
                    'Ethanol Byproduct': Ethanol_byproduct_HMS_fraction,
                    'Water HCl': Water_HCl_HMS_fraction}
        }
    

    
# Scenario - based mass flow analysis

    Support_results = [] # Store results in list
    Support_results_dic = {} # Store matrices for each component

    # Iterate over each sorbent
    for sorbent_type in Support_fractions.keys():
        if sorbent_type not in Support_sorbent_dic:
            continue

        support_mass = Support_sorbent_dic[sorbent_type]

        # Find the first non-zero step in the support mass starting from the end
        non_zero_support = None
        for i in range(support_mass.shape[1] - 1):
            if np.any(support_mass[:, i, :] != 0):
                non_zero_support = i
                break

        # Retrieve component groups
        monomers = Support_Monomers.get(sorbent_type, {})
        solvents = Support_Solvents.get(sorbent_type, {})
        surfactants = Support_Surfactants.get(sorbent_type, {})
        other = Support_Other.get(sorbent_type, {})

        # Initialize dictionary for this sorbent
        Support_results_dic[sorbent_type] = {}

        # Retrieve mass ratios for the current sorbent
        fractions = Support_fractions[sorbent_type]

        # Iterate through each component in the final support
        for component, fraction in fractions.items():
            
            # Identify the correct group & apply back propagation
            if component in monomers:
                matrix = monomers[component].copy()
                # Find the last nonzero index
                non_zero_index = None
                for i in range(matrix.shape[1] - 1, -1, -1):
                    if np.any(matrix[:, i, :] != 0):
                        non_zero_index = i
                        break
                matrix[:, non_zero_index] = support_mass[:, non_zero_support] * fraction # Mass of component in final support
                comp_matrix = pf.back_prop(matrix, step_yields) # Call back prop function
                
            elif component in solvents:
                matrix = solvents[component].copy()
                # Find the last nonzero index
                non_zero_index = None
                for i in range(matrix.shape[1] - 1, -1, -1):
                    if np.any(matrix[:, i, :] != 0):
                        non_zero_index = i
                        break
                matrix[:, non_zero_index] = support_mass[:, non_zero_support] * fraction # Mass of component in final support
                comp_matrix = pf.back_prop(matrix, step_yields_s) # Call back prop function
                
            elif component in surfactants:
                matrix = surfactants[component].copy()
                # Find the last nonzero index
                non_zero_index = None
                for i in range(matrix.shape[1] - 1, -1, -1):
                    if np.any(matrix[:, i, :] != 0):
                        non_zero_index = i
                        break
                matrix[:, non_zero_index] = support_mass[:, non_zero_support] * fraction # Mass of component in final support
                comp_matrix = pf.back_prop(matrix, step_yields_sur) # Call back prop function
                
            elif component in other:
                matrix = other[component].copy()
                # Find the last nonzero index
                non_zero_index = None
                for i in range(matrix.shape[1] - 1, -1, -1):
                    if np.any(matrix[:, i, :] != 0):
                        non_zero_index = i
                        break
                matrix[:, non_zero_index] = support_mass[:, non_zero_support] * fraction # Mass of component in final support
                comp_matrix = pf.back_prop(matrix, step_yields_other) # Call back prop function


            # Store results in dictionary
            Support_results_dic[sorbent_type][component] = comp_matrix.copy()

            # Store results in dataframe
            for step in range(comp_matrix.shape[1]):
                # Store results for first 10 MCS iterations only
                for mcs_iter in range(min(5, comp_matrix.shape[2])):
                    mass = float(comp_matrix[:, step, mcs_iter])
                    Support_results.append({
                    "Scenario": scenario,
                    "Sorbent": sorbent_type,
                    "Component": component,
                    "Step": step,
                    "MCS_Iteration": mcs_iter,
                    "Mass (g)": round(mass, 3)
                })

    Support_results_Df = pd.DataFrame(Support_results)
    
    
# ========================================================================================================================
# STEP 6: PEI SYNTHESIS: Fraction of each chemical component in final PEI (Literature-based Data)
# ========================================================================================================================

# Creation of matrices for all chemical compounds 

    Aziridine_mass = np.zeros((1,n, mcs_number))
    Water_mass = np.zeros((1,n, mcs_number))
    Polyethyleneimine_mass = np.zeros((1,n, mcs_number))
    
    Aziridine_mass[:, np.r_[8:10], :] = 1
    Water_mass[:, np.r_[8:12, 30], :] = 1
    Polyethyleneimine_mass[:, np.r_[10:12, 30], :] = 1
    
# Input data - Calculation of fraction in final PEI
    
    Aziridine_mass[:, 8, :] = 8.3
    Water_mass[:, 8, :] = 20 
    Polyethyleneimine_mass[:, 10, :] = Aziridine_mass[:, 8, :] * np.prod(step_yields[:, [8, 9], :]/(100), axis=1)

# Store chemical components in dictionary       
    PEI_components = {
    "Aziridine": Aziridine_mass,
    "DI Water": Water_mass,
    "Polyethyleneimine": Polyethyleneimine_mass
    }

# Literature - based mass flow analysis 
    
    PEI_results_lit = [] # store results in list to save to df
    PEI_results_lit_dic = {}  # store results in dictionary to access later

    # Iterate across all components
    for component, mass_matrix in PEI_components.items():
    
        # Apply appropriate front propagation based on component type
        if component == "Aziridine":
            updated_matrix = pf.front_prop(mass_matrix, step_yields)
        elif component == "DI Water":
            updated_matrix = pf.front_prop(mass_matrix, step_yields_s)
        elif component == "Polyethyleneimine":
            updated_matrix = pf.front_prop(mass_matrix, step_yields)
    
        # Store results in dictionary
        PEI_results_lit_dic[component] = updated_matrix
        
        # Store results in dataframe
        for step in range(updated_matrix.shape[1]):
            # Store results for first 10 MCS iterations only
            for mcs_iter in range(min(5, updated_matrix.shape[2])):
                mass = float(updated_matrix[:, step, mcs_iter])
                PEI_results_lit.append({
            "Scenario": scenario,
            "Component": component,
            "Step": step,
            "MCS_Iteration": mcs_iter,
            "Mass (g)": round(mass, 3)
        })

    # Create final DataFrame
    PEI_lit_Df = pd.DataFrame(PEI_results_lit)
    
# ========================================================================================================================
# STEP 7: PEI SYNTHESIS: SCENARIO-BASED MASS FLOW ANALYSIS 
# ========================================================================================================================
    
    # Mass of each component in PEI
    
    Aziridine = PEI_results_lit_dic['Aziridine']
    Aziridine_PEI = Aziridine[:, 9, :]
    Water = PEI_results_lit_dic['DI Water']
    Water_PEI = Water[:, 30, :]
    Polyethyleneimine = PEI_results_lit_dic['Polyethyleneimine']
    Polyethyleneimine_PEI = Polyethyleneimine[:, 30, :]
    
    # Mass of PEI after polymerization
    
    PEI_Total_mass = Polyethyleneimine_PEI + Water_PEI
    
    # Mass Fraction of each component in PEI
    
    Aziridine_PEI_fraction = Aziridine_PEI/PEI_Total_mass
    Water_PEI_fraction = Water_PEI/PEI_Total_mass
    Polyethyleneimine_PEI_fraction = Polyethyleneimine_PEI/PEI_Total_mass

    
    # Dictionary with chemical compound fractions
    
    PEI_fractions = {'Aziridine': Aziridine_PEI_fraction,
                     'DI Water': Water_PEI_fraction,
                     'Polyethyleneimine': Polyethyleneimine_PEI_fraction}
    
# Scenario - based mass flow analysis

    PEI_results_dic = {} # Store matrices for each component
    PEI_results = [] # Store results in list

    # Loop over sorbent types
    for sorbent_type, PEI_mass in PEI_sorbent_dic.items():

        PEI_results_dic[sorbent_type] = {}

        for component, fraction in PEI_fractions.items():

            matrix = PEI_components[component].copy()

            # Identify the correct component & apply back propagation
            if component == "Aziridine":
                matrix[:, 9, :] = PEI_mass[:, 30, :] * fraction # Assign mass of component in final PEI (step 30 - before PEI dissolution)
                comp_matrix = pf.back_prop(matrix, step_yields)
            elif component == "DI Water":
                matrix[:, 30, :] = PEI_mass[:, 30, :] * fraction # Assign mass of component in final PEI (step 30 - before PEI dissolution)
                comp_matrix = pf.back_prop(matrix, step_yields_s)
            elif component == "Polyethyleneimine":
                matrix[:, 30, :] = PEI_mass[:, 30, :] * fraction # Assign mass of component in final PEI (step 30 - before PEI dissolution)
                comp_matrix = pf.back_prop(matrix, step_yields)

            # Store result in dict
            PEI_results_dic[sorbent_type][component] = comp_matrix.copy()

            # Store result in long-format list
            for step in range(comp_matrix.shape[1]):
                # Store results for first 10 MCS iterations only
                for mcs_iter in range(min(5, comp_matrix.shape[2])):
                    mass = float(comp_matrix[:, step, mcs_iter])
                    PEI_results.append({
                    "Scenario": scenario,
                    "Sorbent Type": sorbent_type,
                    "Component": component,
                    "Step": step,
                    "MCS_Iteration": mcs_iter,
                    "Mass (g)": round(mass, 3),
                    })

    # Create final DataFrame
    PEI_results_Df = pd.DataFrame(PEI_results)
    

# ========================================================================================================================
# STEP 8: AZIRIDINE SYNTHESIS: Fraction of each chemical compound in final aziridine (Literature-based Data)
# ========================================================================================================================

# Creation of matrices for all chemical components - Definition of steps that each chemical is present 
    
    # Production of  aminoethyl sulfate
    Monoethanolamine_mass = np.zeros((1,n, mcs_number))
    Sulfuric_Acid_mass = np.zeros((1,n, mcs_number))
    Water_Sulfuric_mass = np.zeros((1,n, mcs_number))
    Aminoethyl_sulfate_mass = np.zeros((1,n, mcs_number))
    Water_byproduct_mass = np.zeros((1,n, mcs_number))
    Ethanol_mass = np.zeros((1,n, mcs_number))
    Water_Ethanol_mass = np.zeros((1,n, mcs_number))
    Ethanol_Washing_mass = np.zeros((1,n, mcs_number))
    
    # Production of aziridine
    Sodium_Hydroxide_mass = np.zeros((1,n, mcs_number))
    Water_NaOH_mass = np.zeros((1,n, mcs_number))
    Aziridine_mass = np.zeros((1,n, mcs_number))
    Sodium_sulfate_byproduct_mass = np.zeros((1,n, mcs_number))
    Water_byproduct_2_mass = np.zeros((1,n, mcs_number))
    Sodium_Hydroxide_washing = np.zeros((1,n, mcs_number))

    # Production of  aminoethyl sulfate
    Monoethanolamine_mass[:, np.r_[0:2], :] = 1
    Sulfuric_Acid_mass[:, np.r_[0:2], :] = 1
    Water_Sulfuric_mass[:, np.r_[0:2], :] = 1
    Aminoethyl_sulfate_mass[:, np.r_[1:6], :] = 1
    Water_byproduct_mass[:, np.r_[1:6], :] = 1
    Ethanol_mass[:, np.r_[2:6], :] = 1
    Water_Ethanol_mass[:, np.r_[2:6], :] = 1
    Ethanol_Washing_mass[:, np.r_[4:6], :] = 1
    
    # Production of aziridine
    Sodium_Hydroxide_mass[:, np.r_[5:6], :] = 1
    Water_NaOH_mass[:, np.r_[5:6], :] = 1
    Aziridine_mass[:, np.r_[6:9], :] = 1
    Sodium_sulfate_byproduct_mass[:, np.r_[6:9], :] = 1
    Water_byproduct_2_mass[:, np.r_[6:9], :] = 1
    Sodium_Hydroxide_washing[:, np.r_[7:9], :] = 1


# Input Mass Data from literature for each chemical compound (in grams)
    
    # Production of  aminoethyl sulfate
    Monoethanolamine_mass[:, 0, :] = 610.8
    Sulfuric_Acid_mass[:, 0, :] = 980.79
    Water_Sulfuric_mass[:, 0, :] = 40.8
    Aminoethyl_sulfate_mass[:, 1, :] = 1411.5 # Based on reaction stoichiometry
    Water_byproduct_mass[:, 1, :] = 180.15 # Based on reaction stoichiometry
    Ethanol_mass[:, 2, :] = (1411.5/2 * step_yields[:,1, :]/100 + 180/2 * step_yields_s[:,1, :]/100) * 0.60  # Mass of 60% ethanol - equal to half of mixture mass 
    Water_Ethanol_mass[:, 2, :] = (1411.5/2 * step_yields[:,1, :]/100 + 180/2 * step_yields_s[:,1, :]/100) * 0.40 # Mass of water in ethanol solution - equal to half of mixture mass
    Ethanol_Washing_mass[:, 4, :] = 0.789*(Ethanol_mass[:, 2, :]* np.prod(step_yields_s[:, [2,3], :]/(100), axis=1) + Water_Ethanol_mass[:, 2, :]* np.prod(step_yields_s[:,[2,3], :]/(100), axis=1)+ Aminoethyl_sulfate_mass[:, 1, :] * np.prod(step_yields[:, [1,2,3], :]/(100), axis=1) + Water_byproduct_mass[:, 1, :] * np.prod(step_yields_s[:, [1,2,3], :]/(100), axis=1)) # Mass of ethanol for washing 
    
    # Production of aziridine
    Sodium_Hydroxide_mass[:, 5, :] = 0.567 * 1411.5 * np.prod(step_yields[:, [2, 3, 4], :]/(100), axis=1)
    Water_NaOH_mass[:, 5, :] = 1411.5 * 0.85 * np.prod(step_yields[:, [2, 3, 4], :]/(100), axis=1)
    Aziridine_mass[:, 6, :] = 86/282 * 1411.5 * np.prod(step_yields[:, [2, 3, 4, 5], :]/(100), axis=1)
    Sodium_sulfate_byproduct_mass[:, 6, :] = 142.04/141.15 * 1411.5 * np.prod(step_yields[:, [2, 3, 4, 5], :]/(100), axis=1)
    Water_byproduct_2_mass[:, 6, :] = 72.06/282 * 1411.5 * np.prod(step_yields[:, [2, 3, 4, 5], :]/(100), axis=1)
    Sodium_Hydroxide_washing[:, 7, :] = 0.7 * 1.515 * (86/282 * 1411.5 * np.prod(step_yields[:, [2,3,4,5,6], :]/(100), axis=1) + 142.04/141.15 * 1411.5 * np.prod(step_yields[:, [2,3,4,5], :]/(100), axis=1) * step_yields_other[:, 6, :]/100 + 72.06/282 * 1411.5 * np.prod(step_yields[:, [2,3,4,5], :]/(100), axis=1) * step_yields_s[:, 6, :]/100)
    
# Grouping of chemical components based on their type/action
    
    Aziridine_components = {
        'Monomers':{'Monoethanolamine': Monoethanolamine_mass,
                    'Aminoethyl Sulfate': Aminoethyl_sulfate_mass,
                    'Aziridine': Aziridine_mass},
        'Solvents':{'Ethanol': Ethanol_mass,
                    'Ethanol Washing': Ethanol_Washing_mass,
                    'Water Sulfuric Acid': Water_Sulfuric_mass,
                    'Water Ethanol': Water_Ethanol_mass,
                    'Water Byproduct (1)': Water_byproduct_mass,
                    'Water Byproduct (2)': Water_byproduct_2_mass,
                    'Water NaOH': Water_NaOH_mass},
        'Other':{'Sulfuric Acid': Sulfuric_Acid_mass,
                 'Sodium Hydroxide': Sodium_Hydroxide_mass,
                 'Sodium Sulfate': Sodium_sulfate_byproduct_mass,
                 'Sodium Hydroxide Washing': Sodium_Hydroxide_washing}
        }


# Literature - based mass flow analysis 
   
    Aziridine_results_lit = [] # store results in list to save to df
    Aziridine_results_lit_dic = {}  # store results in dictionary to access later

    # Iterate across all groups and components
    for group_name, components in Aziridine_components.items():
        for comp, mass_matrix in components.items():

            # Apply front propagation - call front_prop function for each chemical component
            if group_name == "Monomers":
                updated_matrix = pf.front_prop(mass_matrix, step_yields)
            elif group_name == "Solvents":
                updated_matrix = pf.front_prop(mass_matrix, step_yields_s)
            elif group_name == "Other":
                updated_matrix = pf.front_prop(mass_matrix, step_yields_other)

            # Store results in dictionary
            Aziridine_results_lit_dic[comp] = updated_matrix

            # Store results in dataframe
            for step in range(updated_matrix.shape[1]):
                # Store results for first 10 MCS iterations only
                for mcs_iter in range(min(5, updated_matrix.shape[2])):
                    mass = float(updated_matrix[:, step, mcs_iter])
                    Aziridine_results_lit.append({
                "Scenario": scenario,
                "Component": comp,
                "Type": group_name,
                "Step": step,
                "MCS_Iteration": mcs_iter,
                "Mass (g)": round(mass, 3)
                })

    # Create final DataFrame
    Aziridine_lit_Df = pd.DataFrame(Aziridine_results_lit)
    
# ========================================================================================================================
# STEP 9: AZIRIDINE SYNTHESIS: SCENARIO-BASED MASS FLOW ANALYSIS  
# ========================================================================================================================

# Find the fraction of each chemical compound in the final aziridine

    # Mass of each component in aminoethyl sulfate synthesis
    
    Monoethanolamine = Aziridine_results_lit_dic['Monoethanolamine']
    Monoethanolamine_aziridine = Monoethanolamine[:, 1, :]
    Sulfuric_Acid = Aziridine_results_lit_dic['Sulfuric Acid']
    Sulfuric_Acid_aziridine = Sulfuric_Acid[:, 1, :]
    Water_Sulfuric = Aziridine_results_lit_dic['Water Sulfuric Acid']
    Water_Sulfuric_aziridine = Water_Sulfuric[:, 1, :]
    Aminoethyl_sulfate = Aziridine_results_lit_dic['Aminoethyl Sulfate']
    Aminoethyl_sulfate_aziridine = Aminoethyl_sulfate[:, 5, :]   
    Ethanol = Aziridine_results_lit_dic['Ethanol']
    Ethanol_aziridine = Ethanol[:, 5, :]
    Water_Ethanol = Aziridine_results_lit_dic['Water Ethanol']
    Water_Ethanol_aziridine = Water_Ethanol[:, 5, :]
    Ethanol_Washing = Aziridine_results_lit_dic['Ethanol Washing']
    Ethanol_Washing_aziridine = Ethanol_Washing[:, 5, :]
    Water_byproduct_1 = Aziridine_results_lit_dic['Water Byproduct (1)']
    Water_byproduct_1_aziridine = Water_byproduct_1[:, 5, :]

    # Mass of each component in aziridine synthesis

    Sodium_Hydroxide = Aziridine_results_lit_dic['Sodium Hydroxide']
    Sodium_Hydroxide_aziridine = Sodium_Hydroxide[:, 5, :]
    Water_NaOH = Aziridine_results_lit_dic['Water NaOH']
    Water_NaOH_aziridine = Water_NaOH[:, 5, :] 
    Aziridine = Aziridine_results_lit_dic['Aziridine']
    Aziridine_aziridine = Aziridine[:, 8, :]
    Water_byproduct_2 = Aziridine_results_lit_dic['Water Byproduct (2)']
    Water_byproduct_2_aziridine = Water_byproduct_2[:, 8, :]
    Sodium_Sulfate_byproduct = Aziridine_results_lit_dic['Sodium Sulfate']
    Sodium_Sulfate_byproduct_aziridine = Sodium_Sulfate_byproduct[:, 8, :]
    Sodium_Hydroxide_washing = Aziridine_results_lit_dic['Sodium Hydroxide Washing']
    Sodium_Hydroxide_washing_aziridine = Sodium_Hydroxide_washing[:, 8, :]

    # Mass of final aziridine
    Aziridine_Total_mass = Aziridine_aziridine + Water_byproduct_2_aziridine + Sodium_Sulfate_byproduct_aziridine 

    # Mass Fraction of each component in final aziridine
    
    Monoethanolamine_fraction = Monoethanolamine_aziridine/Aziridine_Total_mass
    Ethanol_Washing_fraction = Ethanol_Washing_aziridine/Aziridine_Total_mass
    Water_Sulfuric_fraction = Water_Sulfuric_aziridine/Aziridine_Total_mass
    Water_Ethanol_fraction = Water_Ethanol_aziridine/Aziridine_Total_mass
    Water_NaOH_fraction = Water_NaOH_aziridine/Aziridine_Total_mass
    Sulfuric_Acid_fraction = Sulfuric_Acid_aziridine/Aziridine_Total_mass
    Sodium_Hydroxide_fraction = Sodium_Hydroxide_aziridine/Aziridine_Total_mass
    Ethanol_fraction = Ethanol_aziridine/Aziridine_Total_mass
    Water_byproduct_1_fraction = Water_byproduct_1_aziridine/Aziridine_Total_mass
    Aziridine_fraction = Aziridine_aziridine/Aziridine_Total_mass
    Water_byproduct_2_fraction = Water_byproduct_2_aziridine/Aziridine_Total_mass
    Sodium_Sulfate_byproduct_fraction = Sodium_Sulfate_byproduct_aziridine/Aziridine_Total_mass
    Sodium_Hydroxide_washing_fraction = Sodium_Hydroxide_washing_aziridine/Aziridine_Total_mass
    Aminoethyl_sulfate_fraction = Aminoethyl_sulfate_aziridine/Aziridine_Total_mass
    
    
    # Dictionary with chemical compound fractions
    
    Aziridine_fractions = {
                    'Monoethanolamine': Monoethanolamine_fraction,
                    'Aminoethyl Hydrogen Sulfate': Aminoethyl_sulfate_fraction,
                    'Aziridine': Aziridine_fraction,
                    'Ethanol': Ethanol_fraction,
                    'Ethanol Washing': Ethanol_Washing_fraction,
                    'Water Sulfuric Acid': Water_Sulfuric_fraction,
                    'Water Ethanol': Water_Ethanol_fraction,
                    'Water Byproduct (1)': Water_byproduct_1_fraction,
                    'Water Byproduct (2)': Water_byproduct_2_fraction,
                    'Water NaOH': Water_NaOH_fraction,
                    'Sulfuric Acid': Sulfuric_Acid_fraction,
                    'Sodium Hydroxide': Sodium_Hydroxide_fraction,
                    'Sodium Sulfate': Sodium_Sulfate_byproduct_fraction,
                    'Sodium Hydroxide Washing': Sodium_Hydroxide_washing_fraction
        }
    

    
# Scenario - based mass flow analysis

    Aziridine_results = [] # Store results in list
    Aziridine_results_dic = {} # Store matrices for each component

    # Extract Aziridine matrices from PEI_results_dic
    Aziridine_matrices = {
        sorbent_type: matrix_mass["Aziridine"]
        for sorbent_type, matrix_mass in PEI_results_dic.items()
        if "Aziridine" in matrix_mass
        }

    # Iterate over each sorbent
    for sorbent_type, aziridine_matrix in Aziridine_matrices.items():

        # Extract Aziridine mass at step 8
        Pure_Aziridine = aziridine_matrix[:, 8, :]

        # Extract chemical groups
        monomers = Aziridine_components.get("Monomers", {})
        solvents = Aziridine_components.get("Solvents", {})
        other = Aziridine_components.get("Other", {})

        # Initialize dict for current sorbent
        Aziridine_results_dic[sorbent_type] = {}

        # Iterate through all components in the 
        for component, fraction in Aziridine_fractions.items():
        
            # Identify the correct group & apply front propagation
            if component in monomers:
                matrix = monomers[component].copy()
                yields = step_yields
        
            elif component in solvents:
                matrix = solvents[component].copy()
                yields = step_yields_s
        
            elif component in other:
                matrix = other[component].copy()
                yields = step_yields_other
        
            # Find the last nonzero index
            non_zero_index = None
            for i in range(matrix.shape[1] - 1, -1, -1):
                if np.any(matrix[:, i, :] != 0):
                    non_zero_index = i
                    break
        
            if non_zero_index is not None:
                matrix[:, non_zero_index, :] = Pure_Aziridine * fraction
        
            comp_matrix = pf.back_prop(matrix, yields)  # Call back prop function

            # Store results in dictionary
            Aziridine_results_dic[sorbent_type][component] = comp_matrix.copy()

            # Store results in long-format dataframe list
            for step in range(comp_matrix.shape[1]):
                # Store results for first 10 MCS iterations only
                for mcs_iter in range(min(10, comp_matrix.shape[2])):
                    mass_value = float(comp_matrix[:, step, mcs_iter])
                    Aziridine_results.append({
                "Scenario": scenario,
                "Sorbent Type": sorbent_type,
                "Component": component,
                "Step": step,
                "MCS_Iteration": mcs_iter,
                "Mass (g)": round(mass_value, 3),
                })

    # Convert to DataFrame
    Aziridine_results_Df = pd.DataFrame(Aziridine_results)

    return Sorbent_mass, Sorbent_results_dic, Sorbent_results_df, Support_results_dic, Support_results_Df, PEI_results_dic, PEI_results_Df, Aziridine_results_dic, Aziridine_results_Df, Support_lit_Df, PEI_lit_Df, Aziridine_lit_Df 

