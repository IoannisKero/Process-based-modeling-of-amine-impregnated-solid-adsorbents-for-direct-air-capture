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

# ========================================================================================================================
# STEP 0: REACTION 1 - SODIUM CHROMATE PRODUCTION
# ========================================================================================================================

# Molar mass (kg/mol)
MW_soda_ash = 105.99/1000
MW_chromium_trioxide = 100.00/1000  # CrO3

# Mass (kg) per 1 kg Cr(NO3)3·9H2O produced 
M_soda_ash = 0.441
M_chromium_trioxide = 0.441

# Heat Capacity (J/kg*K)
Cp_soda_ash_25 = 111 / MW_soda_ash
Cp_chromium_trioxide_25 = 55.98 / MW_chromium_trioxide

Cp_soda_ash_900 = 166.2 / MW_soda_ash
Cp_chromium_trioxide_900 = 79.94 / MW_chromium_trioxide

# Average Heat Capacity (J/kg*K)
Cp_soda_ash = 0.5 * (Cp_soda_ash_25 + Cp_soda_ash_900)
Cp_chromium_trioxide = 0.5 * (Cp_chromium_trioxide_25 + Cp_chromium_trioxide_900)

# Temperature (K)
T_initial = 298.15
T_final = 900 + 273.15

# Efficiency factor (Heating element)
efficiency = 0.8

# Heat-up energy (MJ), sensible heat only 
Q_step_1 = (Cp_soda_ash * M_soda_ash + Cp_chromium_trioxide * M_chromium_trioxide) * (T_final - T_initial) / (efficiency*1e6)

print(Q_step_1)

# ========================================================================================================================
# STEP 0: REACTION 2 - SODIUM NITRATE NONAHYDRATE PRODUCTION
# ========================================================================================================================

# Molar mass (kg/mol)
MW_sodium_chromate = 161.97/1000
MW_nitric_acid = 63.01/1000  
MW_methanol = 32.04/1000

# Mass (kg) per 1 kg Cr(NO3)3·9H2O produced 
M_sodium_chromate = 0.539
M_nitric_acid = 0.340 * 0.23
M_water = 0.340 * 0.77
M_methanol = 1.08*10**-4

# Heat Capacity (J/kg*K)
Cp_sodium_chromate_25 = 142.1 / MW_sodium_chromate
Cp_nitric_acid_25 = 1.72 * 1000
Cp_methanol_25 = 2.53 * 1000
Cp_water_25 = 4.18 * 1000

# Temperature (K)
T_initial = 25 + 273.15
T_mixing = T_initial - 15
T_concentration = T_mixing + 70
T_final = T_concentration - 50 

# Efficiency factor
efficiency = 0.8

# COP
COP = 4

# Enthalpy of vaporization (J/kg)
DH_vap_water = 2297*1000

# Solvents evaporation energy (MJ)
Q_evaporation = 0.5*(DH_vap_water * M_water)/(1e6*efficiency)

#print(Q_evaporation)

# Heat - mixing (MJ)
Q_mixing = -(Cp_sodium_chromate_25 * M_sodium_chromate + Cp_nitric_acid_25 * M_nitric_acid + Cp_water_25 * M_water + Cp_methanol_25 * M_methanol) * (T_mixing - T_initial)/1e6
E_mixing = Q_mixing / COP

#print(E_mixing)

# Heat - concentration (MJ)
Q_concentration = (Cp_sodium_chromate_25 * M_sodium_chromate + Cp_nitric_acid_25 * M_nitric_acid + Cp_water_25 * M_water + Cp_methanol_25 * M_methanol) * (T_concentration - T_mixing)/(efficiency*1e6) + Q_evaporation

##print(Q_concentration)

# Heat - cooling (MJ)
Q_cooling = -(Cp_sodium_chromate_25 * M_sodium_chromate + Cp_nitric_acid_25 * M_nitric_acid + Cp_water_25 * M_water + Cp_methanol_25 * M_methanol) * (T_final - T_concentration)/1e6
E_cooling = Q_cooling / COP

#print(Q_cooling)

Q_step_2 = Q_concentration 
E_step_2 = E_mixing + E_cooling

print(Q_step_2)
#print(E_step_2)

# Total energy consumption (MJ)
Q_total = Q_step_1 + Q_step_2
E_total = E_step_2 * 0.277778 # Convert MJ to kWh

print(Q_total)
print(E_total)
