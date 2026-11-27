# generated from causaliq@c7917d9e7fa8466b07548514cdf37d494b861321 src/causaliq/devtools/templates.py - do not edit
# Usage: .\scripts\setup-env.ps1 [-ExistingOnly] [-Help]
param(
    [switch]$ExistingOnly,
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
    Write-Host "Usage: .\scripts\setup-env.ps1 [-ExistingOnly] [-Help]"
    Write-Host "Creates one virtual environment per supported Python version."
    exit 0
}

foreach ($key in $Environments.Keys) {
    $version = $Environments[$key]
    $envPath = "venv\py$key"
    if ($ExistingOnly -and -not (Test-Path $envPath)) {
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
    $index = @("--extra-index-url", "https://test.pypi.org/simple/")
    & $python -m pip install --upgrade pip setuptools wheel --quiet
    & $python -m pip install @index -e ".[dev,test,docs]"
}

Write-Host "Environment setup complete."
