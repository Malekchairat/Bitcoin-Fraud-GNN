param(
    [switch]$TrainNormal,
    [switch]$ExecuteNotebook
)
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing $python. Create the project environment first."
}

if ($TrainNormal) {
    Write-Host "1/4 Running the real normal experiment..." -ForegroundColor Cyan
    & $python "train_models.py" "--normal"
    if ($LASTEXITCODE -ne 0) { throw "Normal training failed." }
} else {
    Write-Host "1/4 Using the saved normal experiment (no retraining)" -ForegroundColor Cyan
}

Write-Host "2/4 Exporting verified results to React..." -ForegroundColor Cyan
& $python "export_react_results.py"
if ($LASTEXITCODE -ne 0) { throw "React result export failed." }

if ($ExecuteNotebook) {
    Write-Host "3/4 Executing the notebook and saving outputs..." -ForegroundColor Cyan
    & $python "-m" "jupyter" "nbconvert" "--execute" "--to" "notebook" "--inplace" "--ExecutePreprocessor.timeout=14400" "Bitcoin_Fraud_Detection_GNN.ipynb"
    if ($LASTEXITCODE -ne 0) { throw "Notebook execution failed." }
} else {
    Write-Host "3/4 Notebook execution skipped (saved notebook remains unchanged)" -ForegroundColor Cyan
}

Write-Host "4/4 Starting the React platform..." -ForegroundColor Cyan
Set-Location (Join-Path $PSScriptRoot "web")
if (-not (Test-Path "node_modules")) {
    npm.cmd install
}
npm.cmd run dev -- --host 0.0.0.0
