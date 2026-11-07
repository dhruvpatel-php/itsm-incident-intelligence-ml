# PowerShell Task Runner for ABC Tech ITSM ML Pipeline
param (
    [Parameter(Position=0)]
    [ValidateSet("setup", "run", "test", "app", "notebook", "report", "lint", "hashes")]
    [string]$Task = "test"
)

$ErrorActionPreference = "Stop"
$VenvPython = ".\.venv\Scripts\python.exe"

switch ($Task) {
    "setup" {
        Write-Host ">>> Setting up Python virtual environment..." -ForegroundColor Cyan
        python -m venv .venv
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r requirements.txt -r requirements-dev.txt
        Write-Host ">>> Environment setup complete!" -ForegroundColor Green
    }
    "run" {
        Write-Host ">>> Running end-to-end ITSM Machine Learning pipeline..." -ForegroundColor Cyan
        & $VenvPython -m src.run_all
        & $VenvPython -m src.verify_metrics
    }
    "test" {
        Write-Host ">>> Running Pytest test suite..." -ForegroundColor Cyan
        & $VenvPython -m pytest -q
    }
    "app" {
        Write-Host ">>> Launching Streamlit web application..." -ForegroundColor Cyan
        & ".\.venv\Scripts\streamlit.exe" run app.py
    }
    "notebook" {
        Write-Host ">>> Executing final delivery notebook in-place..." -ForegroundColor Cyan
        & ".\.venv\Scripts\jupyter.exe" nbconvert --to notebook --execute --inplace PRCL-0012_ITSM_Machine_Learning.ipynb
    }
    "report" {
        Write-Host ">>> Compiling REPORT.md and updating model cards..." -ForegroundColor Cyan
        & $VenvPython scripts/build_report.py
    }
    "lint" {
        Write-Host ">>> Running Ruff linter..." -ForegroundColor Cyan
        & $VenvPython -m ruff check .
    }
    "hashes" {
        Write-Host ">>> Generating artifact hashes..." -ForegroundColor Cyan
        & $VenvPython scripts/generate_hashes.py
    }
}
