# Data

This folder contains the training/evaluation data and the fitted scalers used by the multi-task PyTorch models in the `Code/` directory. Each model takes a input position and jointly learns to predict:

1. 159 bits (multi-label binary classification, output via `Sigmoid` + `BCELoss`)
2. 3 real-valued outputs `z\_1, z\_2, z\_3` (regression, `MSELoss`)

## CSV data files

|File|Scenario|Input columns (X)|Data rows|Total columns|
|-|-|-|-|-|
|`scaled\_output\_159bits\_PCR.csv`|PCR (P\_C2R)|`P\_C2R\_1, P\_C2R\_2, P\_C2R\_3`|8000|166|
|`scaled\_output\_159bits\_PMBS.csv`|PMBS|`P\_MBS\_1, P\_MBS\_2, P\_MBS\_3`|8000|166|
|`scaled\_output\_159bits\_PUAV.csv`|PUAV|`P\_UAV\_1, P\_UAV\_2, P\_UAV\_3`|8000|166|
|`scaled\_output\_159bits\_Kx.csv`|Kx/Ky|`Kx, Ky`|500|165|

Column layout for each file (in order):

```
\[input X] , z\_1, z\_2, z\_3 , bit\_1 ... bit\_159 
```

* Input columns (X): the scaled position/coordinates used as model input. 3-dimensional for `PCR`, `PMBS`, `PUAV`; 2-dimensional (`Kx`, `Ky`) for the Kx dataset.
* `z\_1, z\_2, z\_3`:** the 3 real-valued regression targets, already scaled in the CSV — the corresponding `scaler\_y\_\*` is needed to recover the original values.
* `bit\_1` through `bit\_159`: 159 binary (0/1) labels used for the multi-label classification head.

All X and Y values in the CSV files are already scaled; original values can only be recovered using the matching `scaler\_\*.save` files below.

## Scaler files (`.save`, scikit-learn / `joblib`)

|Scaler file|Used for|Purpose|
|-|-|-|
|`scaler\_x\_input\_min\_PCR.save`|PCR|Inverse-transform of input X (min-max)|
|`scaler\_y\_output\_min\_PCR.save`|PCR|Inverse-transform of output `z\_1..z\_3`|
|`scaler\_x\_input\_min\_PUAV.save`|PUAV|Inverse-transform of input X|
|`scaler\_y\_output\_min\_PUAV.save`|PUAV|Inverse-transform of output `z\_1..z\_3`|
|`scaler\_x\_input\_min\_Kx.save`|Kx/Ky|Inverse-transform of input X|
|`scaler\_y\_output\_min\_PKx.save`|Kx/Ky|Inverse-transform of output `z\_1..z\_3`|
|`scaler\_x\_input\_min.save`|PMBS|Inverse-transform of input X|
|`scaler\_y\_output\_min.save`|PMBS|Inverse-transform of output `z\_1..z\_3`|
|`scaler\_x\_input\_standard.save`|—|Standard-scaler version for X (not referenced by the current scripts in `Code/`)|
|`scaler\_y\_output\_standard.save`|—|Standard-scaler version for Y (not referenced by the current scripts in `Code/`)|

> Naming note: the \*\*PMBS\*\* dataset uses scaler files \*without\* a suffix (`scaler\_x\_input\_min.save`, `scaler\_y\_output\_min.save`), unlike PCR/PUAV/Kx, which use the `\_PCR`, `\_PUAV`, and `\_Kx`/`\_PKx` suffixes.

## Mapping to training scripts (`Code/`)

|Script|CSV used|Scaler X|Scaler Y|Input dim|Saved model|
|-|-|-|-|-|-|
|`Main\_PC2R.py`|`scaled\_output\_159bits\_PCR.csv`|`scaler\_x\_input\_min\_PCR.save`|`scaler\_y\_output\_min\_PCR.save`|3|`multitask\_159bits\_3real\_PCR.pth`|
|`Main\_PMBS.py`|`scaled\_output\_159bits\_PMBS.csv`|`scaler\_x\_input\_min.save`|`scaler\_y\_output\_min.save`|3|`multitask\_159bits\_3real\_PMBS.pth`|
|`Main\_PUAV.py`|`scaled\_output\_159bits\_PUAV.csv`|`scaler\_x\_input\_min\_PUAV.save`|`scaler\_y\_output\_min\_PUAV.save`|3|`multitask\_199bits\_3real\_PUAV.pth`|
|`Main\_Kx\_Ky.py`|`scaled\_output\_159bits\_Kx.csv`|`scaler\_x\_input\_min\_Kx.save`|`scaler\_y\_output\_min\_PKx.save`|2|*(see script)*|

The corresponding `Predict\_\*.py` scripts load the matching CSV + scalers + saved model to run inference over the full dataset and write predictions to a `prediction\_results\_\*.csv` file.

## Quick usage

```python
import pandas as pd
import joblib

df = pd.read\_csv("Data/scaled\_output\_159bits\_PCR.csv")
scaler\_x = joblib.load("Data/scaler\_x\_input\_min\_PCR.save")
scaler\_y = joblib.load("Data/scaler\_y\_output\_min\_PCR.save")

X\_original = scaler\_x.inverse\_transform(df\[\["P\_C2R\_1", "P\_C2R\_2", "P\_C2R\_3"]].values)
Y\_original = scaler\_y.inverse\_transform(df\[\["z\_1", "z\_2", "z\_3"]].values)
```

## File sizes

|File|Size|
|-|-|
|`scaled_output_159bits_PCR.csv`|\~3.0 MB|
|`scaled_output_159bits_PMBS.csv`|\~3.2 MB|
|`scaled_output_159bits_PUAV.csv`|\~3.2 MB|
|`scaled_output_159bits\_Kx.csv`|\~192 KB|
|Each `.save` file|\~1 KB|



