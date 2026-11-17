# Python API reference

These examples describe the functions in this repository. Run Python from the
repository root so that imports such as `from bagging import fit_bagging` work.
The project uses local modules rather than an installed Python package.

## Environment

Install the dependencies in [requirements.txt](../requirements.txt) into a
virtual environment. On Windows PowerShell, activation is optional:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On macOS or Linux, use `python3 -m venv .venv` and
`.venv/bin/python` for the subsequent commands. The examples below assume
that the selected interpreter has these dependencies installed.

## load_iris_split(test_size=0.2, random_state=42)

Source: [iris_baseline.py](../iris_baseline.py).

Returns `(X_train, X_test, y_train, y_test, encoder)`. Features are floating
NumPy arrays in this order: SepalLengthCm, SepalWidthCm, PetalLengthCm,
PetalWidthCm. The Id column is excluded. Targets are integer codes; use the
returned encoder to recover Species names.

```python
from iris_baseline import load_iris_split
X_train, X_test, y_train, y_test, encoder = load_iris_split()
print(encoder.inverse_transform(y_test))
```

The default split is stratified and yields 120 training and 30 test rows for
the bundled Iris CSV. Arrays are unscaled: fit preprocessing using training
data only. The CSV path is relative to the module, not the working directory.
Changing random_state changes the split.
