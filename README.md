# Data

This folder contains the training/evaluation data and the fitted scalers used by the multi-task PyTorch models in the `Code/` directory. Each model takes a input position and jointly learns to predict:

1. 159 bits (multi-label binary classification, output via `Sigmoid` + `BCELoss`)
2. 3 real-valued outputs `z_1, z_2, z_3` (regression, `MSELoss`)

## CSV data files

|File|Scenario|Input columns (X)|Data rows|Total columns|
|-|-|-|-|-|
|`scaled_output_159bits_PC2R.csv`|P_C2R|`P_C2R_1, PC2R_2, P_C2R_3`|8000|166|
|`scaled_output_159bits_PMBS.csv`|PMBS|`P_MBS_1, P_MBS_2, P_MBS_3`|8000|166|
|`scaled_output_159bits_PUAV.csv`|PUAV|`P_UAV_1, P_UAV_2, P_UAV_3`|8000|166|
|`scaled_output_159bits_Kx.csv`|Kx/Ky|`Kx, Ky`|500|165|

Column layout for each file (in order):

```
[input X] , z_1, z_2, z_3 , bit_1 ... bit_159 
```

* Input columns (X): the scaled position/coordinates used as model input. 3-dimensional for `PC2R`, `PMBS`, `PUAV`; 2-dimensional (`Kx`, `Ky`) for the Kx dataset.
* `z_1, z_2, z_3`:** the 3 real-valued regression targets, already scaled in the CSV — the corresponding `scaler_y_*` is needed to recover the original values.
* `bit_1` through `bit_159`: 159 binary (0/1) labels used for the multi-label classification head.

All X and Y values in the CSV files are already scaled; original values can only be recovered using the matching `scaler_*.save` files below.

## Scaler files (`.save`, scikit-learn / `joblib`)

|Scaler file|Used for|Purpose|
|-|-|-|
|`scaler_x_input_min_PC2R.save`|PCR|Inverse-transform of input X (min-max)|
|`scaler_y_output_min_PC2R.save`|PCR|Inverse-transform of output `z_1..z_3`|
|`scaler_x_input_min_PUAV.save`|PUAV|Inverse-transform of input X|
|`scaler_y_output_min_PUAV.save`|PUAV|Inverse-transform of output `z_1..z_3`|
|`scaler_x_input_min_Kx.save`|Kx/Ky|Inverse-transform of input X|
|`scaler_y_output_min_PKx.save`|Kx/Ky|Inverse-transform of output `z_1..z_3`|
|`scaler_x_input_min.save`|PMBS|Inverse-transform of input X|
|`scaler_y_output_min.save`|PMBS|Inverse-transform of output `z_1..z_3`|

> Naming note: the **PMBS** dataset uses scaler files *without* a suffix (`scaler_x_input_min.save`, `scaler_y_output_min.save`), unlike PCR/PUAV/Kx, which use the `_PCR`, `_PUAV`, and `_Kx`/`_PKx` suffixes.

## Mapping to training scripts (`Code/`)

|Script|CSV used|Scaler X|Scaler Y|Input dim|Saved model|
|-|-|-|-|-|-|
|`Main_PC2R.py`|`scaled_output_159bits_PC2R.csv`|`scaler_x_input_min_PC2R.save`|`scaler_y_output_min_PC2R.save`|3|`multitask_159bits_3real_PCR.pth`|
|`Main_PMBS.py`|`scaled_output_159bits_PMBS.csv`|`scaler_x_input_min.save`|`scaler_y_output_min.save`|3|`multitask_159bits_3real_PMBS.pth`|
|`Main_PUAV.py`|`scaled_output_159bits_PUAV.csv`|`scaler_x_input_min_PUAV.save`|`scaler_y_output_min_PUAV.save`|3|`multitask_199bits_3real_PUAV.pth`|
|`MainKx_Ky.py`|`scaled_output_159bits_Kx.csv`|`scaler_x_input_min_Kx.save`|`scaler_y_output_min_PKx.save`|2|*(see script)*|

The corresponding `Predict_*.py` scripts load the matching CSV + scalers + saved model to run inference over the full dataset and write predictions to a `prediction_results_*.csv` file.

## Quick usage

```python
import pandas as pd
import joblib

df = pd.read_csv("Data/scaled_output_159bits_PC2R.csv")
scaler_x = joblib.load("Data/scaler_x_input_min_PCR.save")
scaler_y = joblib.load("Data/scaler_y_output_min_PCR.save")

X_original = scaler_x.inverse_transform(df[["P_C2R_1", "P_C2R_2", "P_C2R_3"]].values)
Y_original = scaler_y.inverse_transform(df[["z_1", "z_2", "z_3"]].values)
```

## File sizes

|File|Size|
|-|-|
|`scaled_output_159bits_PC2R.csv`|~3.0 MB|
|`scaled_output_159bits_PMBS.csv`|~3.2 MB|
|`scaled_output_159bits_PUAV.csv`|~3.2 MB|
|`scaled_output_159bits_Kx.csv`|~192 KB|
|Each `.save` file|~1 KB|



