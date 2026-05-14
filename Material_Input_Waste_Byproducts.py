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
# STEP 0: MATERIAL MASS INPUT 
# ========================================================================================================================

def material_mass_input(Mass_Dic, scenarios=None, reference_mass_dic=None, stage_name=None):
    """
    Extracts input mass (first non-zero step per MCS) of each component from 3D matrices,

    Supports flexible normalization based on stage type.

    Parameters:
        Mass_Dic (dict): Nested dictionary [scenario][sorbent][component] = mass matrix (1xN)
        scenarios (dict): Scenarios dictionary to get PEI wt% and batch size info
        reference_mass_dic (dict): Reference mass dictionary for normalization
        stage_name (str): "Aziridine", "PEI", "Support", or "Sorbent" for normalization strategy

    Returns:
        Input_Mass_Df (pd.DataFrame): Long-format DataFrame of per-MCS input masses (kg)
        Input_Mass_Dic (dict): dict[scenario][sorbent][component] = 3D array (1,1,mcs) of first-nonzero per MCS
    """

  
    def get_reference_mass_vector(scenario, sorbent, reference_mass_dic, stage_name, mcs_len):
        """
        Return per-MCS reference mass vector in kg (length mcs_len) with no averaging.
        """
        ref_vec = np.ones(mcs_len, dtype=float)

        if stage_name in ["Aziridine", "PEI"]:
            if reference_mass_dic and scenario in reference_mass_dic and sorbent in reference_mass_dic[scenario]:
                pei_masses = reference_mass_dic[scenario][sorbent]
                if 'Polyethyleneimine' in pei_masses:
                    arr = np.asarray(pei_masses['Polyethyleneimine'], dtype=float)
                    if arr.ndim == 3 and arr.shape[1] > 30:
                        ref_vec = arr[0, 30, :mcs_len] / 1000.0  # g → kg per MCS
            return ref_vec

        if stage_name == "Support":
            if reference_mass_dic and scenario in reference_mass_dic and sorbent in reference_mass_dic[scenario]:
                support_masses = reference_mass_dic[scenario][sorbent]
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

        if stage_name == "Sorbent":
            batch_size_kg = 1.0
            if scenarios and scenario in scenarios:
                batch_size_kg = scenarios[scenario]['profile']['batch_size'] / 1000.0
            return np.full(mcs_len, batch_size_kg, dtype=float)

        return ref_vec

    def get_sorbent_mass_vector(scenario, scenarios, mcs_len):
        """
        Return per-MCS sorbent mass vector in kg (length mcs_len).
        Uses scenario batch size when available; defaults to 1.0 kg.
        """
        batch_size_kg = 1.0
        if scenarios and scenario in scenarios:
            batch_size_kg = scenarios[scenario]['profile']['batch_size'] / 1000.0
        return np.full(mcs_len, batch_size_kg, dtype=float)

    Input_Mass_Dic = {}
    results = []  # will be slim rows only
    structured_dic = {}

    for scenario, sorbents in Mass_Dic.items():
        Input_Mass_Dic[scenario] = {}
        structured_dic.setdefault(scenario, {})
        for sorbent, components in sorbents.items():
            Input_Mass_Dic[scenario][sorbent] = {}
            structured_dic[scenario].setdefault(sorbent, {})
            structured_dic[scenario][sorbent].setdefault(stage_name, {})

            mcs_len = None
            total_water_kg = None
            total_ethanol_kg = None
            has_water_input = False
            has_ethanol_input = False

            # Process each component separately (no grouping)
            for component, mass_matrix in components.items():
                if not isinstance(mass_matrix, np.ndarray):
                    continue

                comp_name = component
                a = np.asarray(mass_matrix, dtype=float)
                if a.ndim == 2:
                    a = a[:, :, None]
                steps = a.shape[1]
                mcs = a.shape[2]
                if mcs_len is None:
                    mcs_len = mcs

                # First non-zero per MCS (grams)
                first_vals = np.zeros((mcs,), dtype=float)
                for k in range(mcs):
                    col = a[0, :, k]
                    nz = np.flatnonzero(col != 0)
                    if nz.size > 0:
                        first_vals[k] = col[nz[0]]

                # Per-MCS reference mass (kg)
                ref_vec_kg = get_reference_mass_vector(scenario, sorbent, reference_mass_dic, stage_name, mcs)
                sorbent_ref_vec_kg = get_sorbent_mass_vector(scenario, scenarios, mcs)

                # Convert to kg and normalize element-wise
                input_mass_kg = first_vals / 1000.0
                with np.errstate(divide='ignore', invalid='ignore'):
                    norm_vec = np.divide(input_mass_kg, ref_vec_kg, out=np.full_like(input_mass_kg, np.nan), where=ref_vec_kg>0)
                    norm_vec_sorbent = np.divide(input_mass_kg, sorbent_ref_vec_kg, out=np.full_like(input_mass_kg, np.nan), where=sorbent_ref_vec_kg>0)

                # Save arrays
                Input_Mass_Dic[scenario][sorbent][comp_name] = first_vals.reshape(1, 1, -1)
                Input_Mass_Dic[scenario][sorbent][f"{comp_name} (kg)"] = input_mass_kg.reshape(1, 1, -1)
                if stage_name != "Sorbent":
                    Input_Mass_Dic[scenario][sorbent][f"{comp_name} (kg/kg product)"] = norm_vec.reshape(1, 1, -1)
                Input_Mass_Dic[scenario][sorbent][f"{comp_name} (kg/kg sorbent)"] = norm_vec_sorbent.reshape(1, 1, -1)

                # Aggregate totals:
                # - Total Water: all water inputs except byproduct-related water
                # - Total Ethanol: all ethanol inputs except byproduct-related ethanol
                #                  and excluding "water ethanol" entries
                comp_norm = str(comp_name).lower().replace("_", " ").replace("-", " ")
                comp_tokens = set(comp_norm.split())
                is_byproduct = "byproduct" in comp_tokens
                is_water_component = "water" in comp_tokens
                is_ethanol_component = ("ethanol" in comp_tokens) and ("water" not in comp_tokens)

                if is_water_component and not is_byproduct:
                    if total_water_kg is None:
                        total_water_kg = np.zeros_like(input_mass_kg, dtype=float)
                    total_water_kg += input_mass_kg
                    if np.any(input_mass_kg > 0):
                        has_water_input = True

                if is_ethanol_component and not is_byproduct:
                    if total_ethanol_kg is None:
                        total_ethanol_kg = np.zeros_like(input_mass_kg, dtype=float)
                    total_ethanol_kg += input_mass_kg
                    if np.any(input_mass_kg > 0):
                        has_ethanol_input = True

            # Add aggregated totals as extra input components
            if mcs_len is not None:
                ref_vec_kg = get_reference_mass_vector(scenario, sorbent, reference_mass_dic, stage_name, mcs_len)
                sorbent_ref_vec_kg = get_sorbent_mass_vector(scenario, scenarios, mcs_len)

                def _add_total_component(total_name, total_kg_vec):
                    if total_kg_vec is None:
                        return
                    with np.errstate(divide='ignore', invalid='ignore'):
                        total_norm_vec = np.divide(
                            total_kg_vec, ref_vec_kg,
                            out=np.full_like(total_kg_vec, np.nan),
                            where=ref_vec_kg > 0
                        )
                        total_norm_sorbent_vec = np.divide(
                            total_kg_vec, sorbent_ref_vec_kg,
                            out=np.full_like(total_kg_vec, np.nan),
                            where=sorbent_ref_vec_kg > 0
                        )
                    Input_Mass_Dic[scenario][sorbent][total_name] = (total_kg_vec * 1000.0).reshape(1, 1, -1)
                    Input_Mass_Dic[scenario][sorbent][f"{total_name} (kg)"] = total_kg_vec.reshape(1, 1, -1)
                    if stage_name != "Sorbent":
                        Input_Mass_Dic[scenario][sorbent][f"{total_name} (kg/kg product)"] = total_norm_vec.reshape(1, 1, -1)
                    Input_Mass_Dic[scenario][sorbent][f"{total_name} (kg/kg sorbent)"] = total_norm_sorbent_vec.reshape(1, 1, -1)

                if has_water_input:
                    _add_total_component("Total Water", total_water_kg)
                if has_ethanol_input:
                    _add_total_component("Total Ethanol", total_ethanol_kg)

            # Emit per-MCS rows (no aggregation)
            for comp_name, arr in Input_Mass_Dic[scenario][sorbent].items():
                if not comp_name.endswith(" (kg)"):
                    continue
                base = comp_name[:-5]
                kg_vec = arr[0, 0, :]
                norm_vec = None
                if stage_name != "Sorbent":
                    norm_key = f"{base} (kg/kg product)"
                    norm_arr = Input_Mass_Dic[scenario][sorbent].get(norm_key)
                    norm_vec = norm_arr[0, 0, :] if isinstance(norm_arr, np.ndarray) else np.full_like(kg_vec, np.nan)
                norm_sorbent_key = f"{base} (kg/kg sorbent)"
                norm_sorbent_arr = Input_Mass_Dic[scenario][sorbent].get(norm_sorbent_key)
                norm_sorbent_vec = norm_sorbent_arr[0, 0, :] if isinstance(norm_sorbent_arr, np.ndarray) else np.full_like(kg_vec, np.nan)
                # Ensure nested structure: Scenario -> Sorbent -> Stage -> Component -> MCS -> {values}
                stage_map = structured_dic[scenario][sorbent][stage_name]
                comp_map = stage_map.setdefault(base, {})
                for k in range(kg_vec.shape[0]):
                    row = {
                        "Scenario": scenario,
                        "Sorbent": sorbent,
                        "Stage": stage_name,
                        "MCS": k,
                        "Component": base,
                        "Input Mass (kg)": float(kg_vec[k]),
                    }
                    if stage_name != "Sorbent":
                        row["Input Mass (kg/kg product)"] = float(norm_vec[k])
                    row["Input Mass (kg/kg sorbent)"] = float(norm_sorbent_vec[k])
                    results.append(row)
                    comp_map[k] = {
                        "Input Mass (kg)": float(kg_vec[k]),
                    }
                    if stage_name != "Sorbent":
                        comp_map[k]["Input Mass (kg/kg product)"] = float(norm_vec[k])
                    comp_map[k]["Input Mass (kg/kg sorbent)"] = float(norm_sorbent_vec[k])

    
    return structured_dic

# ========================================================================================================================
# STEP 1: MASS LOSS CALCULATION PER STEP
# ========================================================================================================================

def calculate_mass_loss_per_step(
    Mass_Dic,
    scenarios=None,
    reference_mass_dic=None,
    stage_name=None,
    return_mixture_loss: bool = False,
):
    """
    Build stepwise solvent-removal losses from component mass matrices.

    Solvents tracked (merged by canonical name):
    - Ethanol
    - Methanol
    - DMF
    - n-Butanol
    Inventory rows such as ``Ethanol Washing``, ``Methanol Washing``, ``DMF Washing``,
    ``n-Butanol`` washing streams, and ``Ethanol Byproduct`` (and ``by-product`` spellings)
    are included: they map to the same canonical solvent and are summed on one matrix.
    Pure water streams are excluded. 

    For each MCS iteration, positive mass loss is assigned to the step where mass drops
    between successive non-zero inventory steps. Phase is inferred from the step name.

    Returns:
        dict:
        {sorbent_type: {scenario_id: {solvent_name: {
            "mass_kg": ndarray (1, steps, mcs),   # kg
            "phase": ndarray (1, steps, mcs),     # object dtype: "", "Liquid", or "Vapor"
        }}}}
        Only solvents with at least one positive loss appear under ``solvent_name``.
        ``scenarios``, ``reference_mass_dic``, and ``stage_name`` are accepted for API
        compatibility; they are not used in this calculation.
        If ``return_mixture_loss=True``, returns a tuple:
        ``(structured_dic, mixture_mass_loss_dic)`` where ``mixture_mass_loss_dic`` is:
        ``{sorbent_type: {scenario_id: {"mass_kg": ndarray(1,steps,mcs), "phase": ndarray}}}``
        built from total mixture mass (sum of all valid component mass matrices) losses.
    """

    step_definitions = {
        0: "Stirring (1)",
        1: "Heating (1)",
        2: "Grinding (1)",
        3: "Filtration (1)",
        4: "Washing (1)",
        5: "Stirring (1)",
        6: "Distillation (1)",
        7: "Drying (1)",
        8: "Stirring (2)",
        9: "Aziridine Polymerization (2)",
        10: "Evaporation (2)",
        11: "Drying (2)",
        12: "Stirring (3)",
        13: "Stirring (3)",
        14: "Stirring (3)",
        15: "Stirring (3)",
        16: "Sonication (3)",
        17: "Gelation (3)",
        18: "Aging/Hydrothermal Treatment (3)",
        19: "Polymerization (3)",
        20: "Stirring (3)",
        21: "Filtration (3)",
        22: "Extraction (3)",
        23: "Centrifugation (3)",
        24: "Evaporation (3)",
        25: "Washing (3)",
        26: "Drying (3)",
        27: "Calcination (3)",
        28: "Drying (4)",
        29: "Activation (4)",
        30: "Stirring (4)",
        31: "Sonication (4)",
        32: "Stirring (4)",
        33: "Evaporation (4)",
        34: "Drying (4)",
        35: "Final Sorbent",
    }

    vapor_keywords = (
        "heating", "distillation", "drying", "evaporation", "aging",
        "polymerization", "calcination", "activation", "extraction",
    )
    solvent_tokens = (
        "ethanol", "methanol", "dmf", "n-butanol", "n butanol", "butanol",
        "meoh", "etoh",  # common abbreviations in tables
    )

    def _canonical_solvent(component_name: str) -> str | None:
        """Map a component label to a canonical solvent key, or None if not tracked."""
        raw = str(component_name).lower().strip()
        # Strip process-role suffixes so washing / byproduct streams match the base solvent
        # (e.g. "Ethanol Washing", "Ethanol Byproduct", "DMF Wash").
        n = raw
        for suf in (" washing", " wash", " by-product", " byproduct"):
            if n.endswith(suf):
                n = n[: -len(suf)].strip()
                break
        if "ethanolamine" in raw:  # MEA, DEA, TEA, etc.; not bulk ethanol solvent
            return None
        if "water" in raw and not any(
            tok in raw
            for tok in ("ethanol", "methanol", "dmf", "butanol", "meoh", "etoh")
        ):
            return None
        if not any(tok in n for tok in solvent_tokens):
            return None
        if "dmf" in n:
            return "DMF"
        if "meoh" in n or "methanol" in n:
            return "Methanol"
        if "etoh" in n or "ethanol" in n:
            return "Ethanol"
        if "n-butanol" in n or "n butanol" in n or "butanol" in n:
            return "n-Butanol"
        return None

    def _phase_for_step(step_number: int) -> str | None:
        pname = step_definitions.get(step_number, f"Unknown Step {step_number}").lower()
        if "final sorbent" in pname:
            return None
        return "Vapor" if any(k in pname for k in vapor_keywords) else "Liquid"

    structured_dic = {}
    mixture_mass_loss_dic = {}

    for scenario, sorbents in Mass_Dic.items():
        for sorbent, components in sorbents.items():
            structured_dic.setdefault(sorbent, {})
            scenario_map = structured_dic[sorbent].setdefault(scenario, {})
            solvent_acc: dict[str, np.ndarray] = {}
            mixture_mass = None

            for comp, mass_matrix in components.items():
                canon = _canonical_solvent(comp)
                if canon is None:
                    continue
                if not isinstance(mass_matrix, np.ndarray):
                    continue

                mass_array = np.asarray(mass_matrix, dtype=float)
                if mass_array.ndim == 2:
                    mass_array = mass_array[:, :, None]
                if mass_array.ndim != 3 or mass_array.shape[0] != 1:
                    continue

                # Accumulate total mixture mass (all valid components)
                if mixture_mass is None:
                    mixture_mass = mass_array.copy()
                elif mixture_mass.shape == mass_array.shape:
                    mixture_mass = mixture_mass + mass_array

                mcs = mass_array.shape[2]
                mass_loss_matrix = np.zeros_like(mass_array)
                for mcs_iter in range(mcs):
                    mass_series = mass_array[0, :, mcs_iter]
                    non_zero_indices = np.where(mass_series != 0)[0]
                    if len(non_zero_indices) < 2:
                        continue
                    for i in range(len(non_zero_indices) - 1):
                        current_step = non_zero_indices[i]
                        next_step = non_zero_indices[i + 1]
                        loss = mass_series[current_step] - mass_series[next_step]
                        if loss > 0:
                            mass_loss_matrix[0, current_step, mcs_iter] = loss

                if canon in solvent_acc:
                    if solvent_acc[canon].shape != mass_loss_matrix.shape:
                        continue
                    solvent_acc[canon] = solvent_acc[canon] + mass_loss_matrix
                else:
                    solvent_acc[canon] = mass_loss_matrix.copy()

            for canon, loss_g in solvent_acc.items():
                if not np.any(loss_g > 0):
                    continue
                mass_kg = loss_g / 1000.0
                phase = np.full(loss_g.shape, "", dtype=object)
                for st in range(loss_g.shape[1]):
                    for k in range(loss_g.shape[2]):
                        if mass_kg[0, st, k] <= 0:
                            continue
                        p = _phase_for_step(st)
                        phase[0, st, k] = p if p is not None else ""
                scenario_map[canon] = {"mass_kg": mass_kg, "phase": phase}

            # Extra dictionary: mass loss of the whole mixture (all components summed)
            if isinstance(mixture_mass, np.ndarray):
                mix_loss = np.zeros_like(mixture_mass)
                mix_steps = mixture_mass.shape[1]
                mix_mcs = mixture_mass.shape[2]
                for mcs_iter in range(mix_mcs):
                    mass_series = mixture_mass[0, :, mcs_iter]
                    non_zero_indices = np.where(mass_series != 0)[0]
                    if len(non_zero_indices) < 2:
                        continue
                    for i in range(len(non_zero_indices) - 1):
                        current_step = non_zero_indices[i]
                        next_step = non_zero_indices[i + 1]
                        loss = mass_series[current_step] - mass_series[next_step]
                        if loss > 0:
                            mix_loss[0, current_step, mcs_iter] = loss

                if np.any(mix_loss > 0):
                    mix_mass_kg = mix_loss / 1000.0
                    mix_phase = np.full(mix_loss.shape, "", dtype=object)
                    for st in range(mix_steps):
                        for k in range(mix_mcs):
                            if mix_mass_kg[0, st, k] <= 0:
                                continue
                            p = _phase_for_step(st)
                            mix_phase[0, st, k] = p if p is not None else ""
                    mixture_mass_loss_dic.setdefault(sorbent, {})[scenario] = {
                        "mass_kg": mix_mass_kg,
                        "phase": mix_phase,
                    }

    if return_mixture_loss:
        return structured_dic, mixture_mass_loss_dic
    return structured_dic



                    
    
    
    
    
    
    
    
