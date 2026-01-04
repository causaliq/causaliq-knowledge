# Activate a specific Python environment
# Usage: .\scripts\activate.ps1 [39|310|311|312]
# Defaults to Python 3.11 if no version specified

param(
    [string]$Version = "311"  # Default to Python 3.11
)

if ($Version -eq "help" -or $Version -eq "?" -or $Version -eq "-h") {
    Write-Host "Usage: .\scripts\activate.ps1 [39|310|311|312]" -ForegroundColor Yellow
    Write-Host "Defaults to Python 3.11 if no version specified" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Available environments:" -ForegroundColor Cyan
    
    if (Test-Path "venv\py39") { Write-Host "  39  - Python 3.9  (venv\py39)" -ForegroundColor Green }
    if (Test-Path "venv\py310") { Write-Host "  310 - Python 3.10 (venv\py310)" -ForegroundColor Green }
    if (Test-Path "venv\py311") { Write-Host "  311 - Python 3.11 (venv\py311) [DEFAULT]" -ForegroundColor Green }
    if (Test-Path "venv\py312") { Write-Host "  312 - Python 3.12 (venv\py312)" -ForegroundColor Green }
    
    Write-Host ""
    exit 0
}

$EnvPath = "venv\py$Version"

if (-not (Test-Path $EnvPath)) {
    Write-Host "Error: Environment $EnvPath does not exist" -ForegroundColor Red
    Write-Host "Run .\scripts\setup-env.ps1 first to create environments" -ForegroundColor Yellow
    exit 1
}

# Convert version number to proper display format
$DisplayVersion = switch ($Version) {
    "39" { "3.9" }
    "310" { "3.10" }
    "311" { "3.11" }
    "312" { "3.12" }
    default { "3.$Version" }
}

Write-Host "Activating Python $DisplayVersion environment..." -ForegroundColor Blue
& "$EnvPath\Scripts\Activate.ps1"

if ($Version -eq "311") {
    Write-Host ""
    Write-Host "$([char]0x2713) causaliq-knowledge CLI is now available!" -ForegroundColor Green
    Write-Host "Test it: causaliq-knowledge --help" -ForegroundColor Cyan
    Write-Host ""
}