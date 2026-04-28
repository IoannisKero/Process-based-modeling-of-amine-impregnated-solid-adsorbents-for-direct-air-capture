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

# ========================================================================================================================
# BACK PROPAGATION FUNCTION
# ======================================================================================================================== 

def back_prop(mass_matrix, step_yields):
    
    """
    Applies backward propagation to calculate mass across process steps 
    based on final product mass and step-wise yields.

    Parameters:
    - mass_matrix (np.ndarray): 1xNxMCS array with known mass at downstream step(s) and non-zero placeholders for active prior steps.
    - step_yields (np.ndarray): 1xNxMCS array with step-wise yields in percentage (e.g. 90 for 90%).

    Returns:
    - np.ndarray: Updated mass matrix after backward propagation.
    """
    
    mass_matrix = mass_matrix.copy()  # avoid modifying in-place
    for i in range(mass_matrix.shape[1] - 2, -1, -1):  # iterate from second-to-last to first
        if np.any(mass_matrix[:, i, :] != 0):
            next_cell = i + 1
            while next_cell < mass_matrix.shape[1] and np.all(mass_matrix[:, next_cell, :] == 0):
                next_cell += 1
            if next_cell < mass_matrix.shape[1] and np.any(mass_matrix[:, next_cell, :] != 0):
                mass_matrix[:, i, :] = mass_matrix[:, next_cell, :] / (step_yields[:, i, :] / 100)
    return mass_matrix

# ========================================================================================================================
# FRONT PROPAGATION FUNCTION
# ========================================================================================================================

def front_prop(mass_matrix, step_yields):
    """
    Applies forward propagation to a mass matrix using step yields.

    - Finds the first non-zero entry in the matrix.
    - Propagates values forward only for cells equal to 1 (active-step placeholder).
    - Uses the nearest previous non-zero cell and its yield to compute the propagated value.

    Parameters:
        mass_matrix (np.ndarray): Matrix of mass per step (1 × n × mcs).
        step_yields (np.ndarray): Yield per step (1 × n × mcs), in percentage.

    Returns:
        np.ndarray: Updated mass matrix after forward propagation.
        Note: this function updates `mass_matrix` in-place.
    """
    first_non_zero_found = False  # To track the first non-zero value
    non_zero = None

    for i in range(mass_matrix.shape[1]):
        if np.any(mass_matrix[:, i, :] != 0):
            if not first_non_zero_found:
                first_non_zero_found = True
                non_zero = i
                break

    if non_zero is not None:
        for j in range(non_zero + 1, mass_matrix.shape[1]):
            if np.all(mass_matrix[:, j, :] == 1):
                previous_cell = j - 1
                while previous_cell >= 0 and np.all(mass_matrix[:, previous_cell, :] == 0):
                    previous_cell -= 1
                if previous_cell >= 0 and np.any(mass_matrix[:, previous_cell, :] != 0):
                    mass_matrix[:, j, :] = (
                        (mass_matrix[:, previous_cell, :] * step_yields[:, previous_cell, :] )/ 100
                    )

    return mass_matrix