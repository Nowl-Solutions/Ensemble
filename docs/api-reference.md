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
