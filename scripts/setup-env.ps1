# generated from causaliq@27a2f560f378 src/causaliq/devtools/templates.py - do not edit
# Usage: .\scripts\setup-env.ps1 [-InstallOnly] [-Help]
param(
    [switch]$InstallOnly,
    [switch]$Help
)

$Environments = [ordered]@{
    "310" = "3.10"
    "311" = "3.11"
    "312" = "3.12"
    "313" = "3.13"
    "314" = "3.14"
}

if ($Help) {
    Write-Host "Usage: .\scripts\setup-env.ps1 [-InstallOnly] [-Help]"
    Write-Host "Creates one virtual environment per supported Python version."
    exit 0
}

foreach ($key in $Environments.Keys) {
    $version = $Environments[$key]
    $envPath = "venv\py$key"
    if ($InstallOnly -and -not (Test-Path $envPath)) {
        Write-Host "Skipping $envPath (missing)"
        continue
    }
    if (-not (Test-Path $envPath)) {
        Write-Host "Creating $envPath (Python $version)"
        py -$version -m venv $envPath
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python $version is not available; skipped"
        continue
    }
    $python = "$envPath\Scripts\python.exe"
    & $python -m pip install --upgrade pip setuptools wheel --quiet
    & $python -m pip install -e ".[dev,test,docs]"
}

Write-Host "Environment setup complete."
