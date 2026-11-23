# generated from causaliq@27a2f560f378 src/causaliq/devtools/templates.py - do not edit
# Usage: .\scripts\activate.ps1 [310|311|312|313|314]
# Defaults to Python 3.12
param(
    [string]$Version = "312"
)

$Environments = [ordered]@{
    "310" = "3.10"
    "311" = "3.11"
    "312" = "3.12"
    "313" = "3.13"
    "314" = "3.14"
}

if ($Version -in @("help", "?", "-h")) {
    Write-Host "Usage: .\scripts\activate.ps1 [310|311|312|313|314]"
    Write-Host "Defaults to Python 3.12"
    foreach ($key in $Environments.Keys) {
        Write-Host ("  {0} - Python {1}" -f $key, $Environments[$key])
    }
    exit 0
}

if (-not $Environments.Contains($Version)) {
    Write-Host "Error: unknown Python version $Version"
    exit 1
}

$EnvPath = "venv\py$Version"
if (-not (Test-Path $EnvPath)) {
    Write-Host "Error: $EnvPath does not exist"
    Write-Host "Run .\scripts\setup-env.ps1 first"
    exit 1
}

Write-Host "Activating Python $($Environments[$Version]) environment..."
& "$EnvPath\Scripts\Activate.ps1"
