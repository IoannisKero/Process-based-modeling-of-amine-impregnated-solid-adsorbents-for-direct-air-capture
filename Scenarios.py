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

import numpy as np

# ========================================================================================================================
# STEP 1: YIELD DEFINITIONS
# ========================================================================================================================

# Define the yield data 

Yields = {
    "Precursors": {
        "High": {
            "Stirring/Sonication/Grinding/Aging/Activation": 100.0,
            "Heating": 100,
            "Disitillation": 100,
            "Polymerization (Aziridine)": 90.0,
            "Centrifugation": 75.0,
            "Calcination": 100.0,
            "Filtration": 100.0,
            "Washing": 100.0,
            "Evaporation": 100.0,
            "Drying": 100.0,
        },
        "Base": {
            "Stirring/Sonication/Grinding/Aging/Activation": 99.0,
            "Heating": 98.0,
            "Disitillation": 98.0,
            "Polymerization (Aziridine)":80.0,
            "Centrifugation": 70.0,
            "Calcination": 98.0,
            "Filtration": 98.0,
            "Washing": 98.0,
            "Evaporation": 99.0,
            "Drying": 99.0,
        },
        "Worst": {
            "Stirring/Sonication/Grinding/Aging/Activation": 98.0,
            "Heating": 95.0,
            "Disitillation": 95.0,
            "Polymerization (Aziridine)":70.0,
            "Centrifugation": 65.0,
            "Calcination": 95.0,
            "Filtration": 95.0,
            "Washing": 95.0,
            "Evaporation": 98.0,
            "Drying": 98.0,
        }
    },
    "Surfactants": {
        "High": {
            "Stirring/Sonication/Grinding/Aging/Activation": 100.0,
            "Heating": 100,
            "Disitillation": 100,
            "Polymerization (Aziridine)": 100.0,
            "Centrifugation": 75.0,
            "Calcination": 0,
            "Filtration": 100.0,
            "Washing": 100.0,
            "Evaporation": 100.0,
            "Drying": 100.0,
        },
        "Base": {
            "Stirring/Sonication/Grinding/Aging/Activation": 99.0,
            "Heating": 98.0,
            "Disitillation": 98.0,
            "Polymerization (Aziridine)":98.0,
            "Centrifugation": 70.0,
            "Calcination": 10**-10,
            "Filtration": 98.0,
            "Washing": 5.0,
            "Evaporation": 99.0,
            "Drying": 99.0,
        },
        "Worst": {
            "Stirring/Sonication/Grinding/Aging/Activation": 98.0,
            "Heating": 95.0,
            "Disitillation": 95.0,
            "Polymerization (Aziridine)":95.0,
            "Centrifugation": 65.0,
            "Calcination": 10**-12,
            "Filtration": 95.0,
            "Washing": 10.0,
            "Evaporation": 98.0,
            "Drying": 98.0,
        }
    },
    "Solvents": {
        "High": {
            "Stirring/Sonication/Grinding/Aging/Activation": 100.0,
            "Heating": 0,
            "Disitillation": 0,
            "Polymerization (Aziridine)": 100.0,
            "Centrifugation": 0,
            "Calcination": 0,
            "Filtration": 0,
            "Washing": 0,
            "Evaporation": 0,
            "Drying": 0,
        },
        "Base": {
            "Stirring/Sonication/Grinding/Aging/Activation": 99.0,
            "Heating": 10**-10,
            "Disitillation": 1,
            "Polymerization (Aziridine)":98.0,
            "Centrifugation": 5,
            "Calcination": 10**-10,
            "Filtration": 5,
            "Washing": 5,
            "Evaporation": 2,
            "Drying": 10**-10,
        },
        "Worst": {
            "Stirring/Sonication/Grinding/Aging/Activation": 98.0,
            "Heating": 10**-12,
            "Disitillation": 2,
            "Polymerization (Aziridine)":95.0,
            "Centrifugation": 10,
            "Calcination": 10**-12,
            "Filtration": 10,
            "Washing": 10,
            "Evaporation": 5,
            "Drying": 10**-12,
        }
    },
    "Other": {
        "High": {
            "Stirring/Sonication/Grinding/Aging/Activation": 100.0,
            "Heating": 0,
            "Disitillation": 0,
            "Polymerization (Aziridine)": 100.0,
            "Centrifugation": 0,
            "Calcination": 0,
            "Filtration": 0,
            "Washing": 0,
            "Evaporation": 100.0,
            "Drying": 100.0,
        },
        "Base": {
            "Stirring/Sonication/Grinding/Aging/Activation": 99.0,
            "Heating": 10**-10,
            "Disitillation": 1,
            "Polymerization (Aziridine)":98.0,
            "Centrifugation": 5,
            "Calcination": 10**-10,
            "Filtration": 5,
            "Washing": 5,
            "Evaporation": 99.0,
            "Drying": 99.0,
        },
        "Worst": {
            "Stirring/Sonication/Grinding/Aging/Activation": 98.0,
            "Heating": 10**-12,
            "Disitillation": 2,
            "Polymerization (Aziridine)":95.0,
            "Centrifugation": 10,
            "Calcination": 10**-12,
            "Filtration": 10,
            "Washing": 10,
            "Evaporation": 98.0,
            "Drying": 98.0,
        }
    }
}


# ========================================================================================================================
# STEP 2: SCENARIO DEFINITIONS
# ========================================================================================================================

# Scenario configurations
scenario_configs = [
    
    # Lab to Pilot Scale - 1 kg batch (values are in grams)
    {"id": 1, "name": "Lab_1kg_20%_wt%", "batch_size": 1000, "pei_composition": 20, "scale": "lab"},
    {"id": 2, "name": "Lab_1kg_30%_wt%", "batch_size": 1000, "pei_composition": 30, "scale": "lab"},
    {"id": 3, "name": "Lab_1kg_40%_wt%", "batch_size": 1000, "pei_composition": 40, "scale": "lab"},
    {"id": 4, "name": "Lab_1kg_50%_wt%", "batch_size": 1000, "pei_composition": 50, "scale": "lab"},
    {"id": 5, "name": "Lab_1kg_60%_wt%", "batch_size": 1000, "pei_composition": 60, "scale": "lab"},
    
    # Industrial Scale - 1000 kg batch (values are in grams)
    {"id": 6, "name": "Industrial_1000kg_20%_wt%", "batch_size": 1000000, "pei_composition": 20, "scale": "industrial"},
    {"id": 7, "name": "Industrial_1000kg_30%_wt%", "batch_size": 1000000, "pei_composition": 30, "scale": "industrial"},
    {"id": 8, "name": "Industrial_1000kg_40%_wt%", "batch_size": 1000000, "pei_composition": 40, "scale": "industrial"},
    {"id": 9, "name": "Industrial_1000kg_50%_wt%", "batch_size": 1000000, "pei_composition": 50, "scale": "industrial"},
    {"id": 10, "name": "Industrial_1000kg_60%_wt%", "batch_size": 1000000, "pei_composition": 60, "scale": "industrial"}

]

# ========================================================================================================================
# STEP 3: SCENARIO GENERATION
# ========================================================================================================================

def generate_scenarios():

    scenarios = {}
    
    for config in scenario_configs:
        scenario_id = config["id"]
        scale = config["scale"]
        
        # Select the appropriate yield dictionary
        if scale == "lab":
            yields_dict = Yields
        else:
            yields_dict = Yields
        
        # Create the scenario structure
        scenarios[scenario_id] = {
            "profile": {
                "name": config["name"],
                "batch_size": config["batch_size"],
                "pei_composition": config["pei_composition"],
                "scale": config["scale"],
                "description": f"{config['scale'].title()} scale, {config['batch_size']}g batch, {config['pei_composition']}% PEI"
            },
            "yields": yields_dict
        }
    
    return scenarios


