# DAC Sorbent LCA Model 

These files represent the full process-based model for direct air capture (DAC) sorbent production and life-cycle assessment.

## 1) Python Version

- Recommended: `Python 3.10+`
- Good target: `Python 3.11` or `Python 3.12`

## 2) Required Libraries

Install these before running:

```bash
pip install numpy pandas CoolProp tqdm openpyxl
```

### Why these are needed

- `numpy`: array math and Monte Carlo calculations
- `pandas`: tabular outputs and Excel writing
- `CoolProp`: thermophysical properties used in chemical and energy calculations
- `tqdm`: progress bars while scenarios run
- `openpyxl`: Excel engine used by Monte Carlo output writers

## 3) Files 

- `LCA_Model.py`  
  Main script. Runs the full workflow end-to-end (mass, properties, energy, impacts, MC summaries, and exports).

- `Scenarios.py`  
  Builds the scenario dictionary used by the model, including process yields and batch size selection per scale.

- `Mass_Function.py`  
  Core stage-by-stage and stepwise mass balance model (Aziridine, PEI, Support, Sorbent-Impregnation), including yield uncertainty.

- `Propagation_Functions.py`  
  Helper propagation functions for applying step yields through process chains.

- `Material_Input_Waste_Byproducts.py`  
  Converts mass results into material input inventories and solvent recovery/loss outputs used downstream.

- `Chemical_Properties.py`  
  Returns process temperatures and component properties (density, heat capacity, enthalpy) used in energy modeling.

- `Chemical_Properties_Mixture.py`  
  Builds mixture-level properties from component-level data (e.g., summed heat capacity, density, enthalpy).

- `Physical_Properties.py`  
  Provides equipment efficiencies, power assumptions, process times, and modeling parameters.

- `Energy_Functions.py`  
  Unit-operation energy models (heating, stirring, drying, distillation, filtration, sonication, grinding, solvent recovery), plus aggregation/normalization utilities.

- `Impact_Assessment.py`  
  Environmental impact factor dictionaries, as obtained by openLCA and Ecoinvent and impact calculations for energy and material inventories.

- `Adsorption_Performance.py`  
  Monte Carlo adsorption performance model (lifetime cycles, degradation, cumulative capture), used to normalize outputs to functional performance.

- `Monte_Carlo_Simulations.py`  
  Post-processing and statistical summaries; writes key outputs to Excel files.

- `Save_Results_Functions.py`  
  Utility conversion/saving helpers for energy outputs and tidy result tables.

- `Chromium_Nitrate_Production.py`  
  Standalone calculation script for chromium nitrate production energy assumptions (supporting analysis script, not the main pipeline entrypoint).

## 4) Adsorption Performance Sensitivity Note

In `Adsorption_Performance.py`, sorbent lifetime is controlled by:

```
s = 1
```
To run sensitivity analysis for lifetime assumptions, adjust this value (as noted in the script comments), for example:

- `s = 0.5` for worst-case lifetime
- `s = 1` for base case
- `s = 2` for best-case lifetime

After changing `s`, rerun `LCA_Model.py` to propagate the change through all outputs.

## 5) Typical Workflow (How to Run)

1. Download all scripts.
2. Place all `.py` files together in the same directory-folder.
3. Create and activate a Python environment (recommended).
4. Install required libraries:
   - `pip install numpy pandas CoolProp tqdm openpyxl`
5. (Optional) Adjust model settings before running:
   - scenario subset in `LCA_Model.py`
   - `mcs_number` in `LCA_Model.py`
   - lifetime factor `s` in `Adsorption_Performance.py`
6. Run the main model:
   - `python LCA_Model.py`
7. Collect generated Excel outputs in the same folder (for example):
   - `Life_Cycle_Inventory.xlsx`
   - `Total_Environmental_Impacts.xlsx`
   - `Adsorption Performance.xlsx`


## 6) Notes

- Keep file names unchanged unless you also update import lines.
- Run from inside the folder you saved the files so relative output file paths are created there.
- If you change core assumptions, rerun the full pipeline to keep outputs consistent.

## 7) For LCA modellers:

This model is easy to adapt for alternative assumptions and comparative LCA studies. The most practical updates are:

- **Scenario design (`Scenarios.py`)**: add or edit scenarios for batch size, scale, sorbent loading, and pathway definitions.
- **Monte Carlo scope (`LCA_Model.py`)**: change `mcs_number` and scenario filters to balance runtime vs uncertainty resolution.
- **Mass/yield assumptions (`Mass_Function.py`)**: update step yields, recoveries, and conversion logic to represent new process data.
- **Adsorption performance assumptions (`Adsorption_Performance.py`)**: adjust sorbent lifetime factor `s`, cycle times, and adsorption capacity values for sensitivity cases.
- **Process operating conditions (`Chemical_Properties.py`, `Physical_Properties.py`)**: revise temperatures, times, efficiencies, and equipment power assumptions.
- **Energy model assumptions (`Energy_Functions.py`)**: tune unit-operation constants and correlations (for example loss factors or recovery assumptions).
- **LCA factor sets (`Impact_Assessment.py`)**: replace environmental impact factor dictionaries with region-specific or database-updated factors.

## 8) Citation

If you use this code to generate results for a publication (journal article, conference paper, thesis, report, or other public dissemination), please cite the corresponding paper from this work.
