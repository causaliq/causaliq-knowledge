# Comprehensive Python environment management for causaliq-repo-template
# Usage: .\scripts\setup-env.ps1 [options]

param(
    [switch]$Install,           # Install causaliq-repo-template package after creating environments
    [switch]$InstallOnly,       # Only install packages (skip environment creation)
    [switch]$Help               # Show help information
)

function Show-Help {
    Write-Host ""
    Write-Host "=== Python Environment Management Script ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\scripts\setup-env.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  (no args)        Create Python 3.9-3.12 virtual environments" -ForegroundColor White
    Write-Host "  -Install         Create environments AND install causaliq-repo-template package" -ForegroundColor White
    Write-Host "  -InstallOnly     Install causaliq-repo-template package in existing environments" -ForegroundColor White
    Write-Host "  -Help            Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\scripts\setup-env.ps1                    # Create environments only" -ForegroundColor Gray
    Write-Host "  .\scripts\setup-env.ps1 -Install           # Create + install causaliq-repo-template" -ForegroundColor Gray
    Write-Host "  .\scripts\setup-env.ps1 -InstallOnly       # Install causaliq-repo-template in existing envs" -ForegroundColor Gray
    Write-Host ""
    Write-Host "WORKFLOW:" -ForegroundColor Yellow
    Write-Host "  1. First run: .\scripts\setup-env.ps1 -Install" -ForegroundColor Gray
    Write-Host "  2. Add dependency to pyproject.toml" -ForegroundColor Gray
    Write-Host "  3. Update all environments: .\scripts\setup-env.ps1 -InstallOnly" -ForegroundColor Gray
    Write-Host ""
    exit 0
}

if ($Help) { Show-Help }

$BaseDir = Get-Location
$VenvDir = "venv"

# Check if venv directory exists, create if not
if (-not (Test-Path $VenvDir)) {
    New-Item -ItemType Directory -Path $VenvDir | Out-Null
}

# Function to install packages in an environment
function Install-InEnvironment {
    param(
        [string]$EnvName,
        [string]$DisplayName
    )
    
    if (-not (Test-Path "$VenvDir\$EnvName")) {
        Write-Host "Warning: $DisplayName environment not found, skipping..." -ForegroundColor Yellow
        return
    }
    
    Write-Host "Installing in $DisplayName..." -ForegroundColor Cyan
    
    # Define the Python executable path
    $PythonExe = "$VenvDir\$EnvName\Scripts\python.exe"

    try {
        & "$VenvDir\$EnvName\Scripts\Activate.ps1"

        # Upgrade core build tools first
        Write-Host "  Upgrading pip, setuptools, and wheel..." -ForegroundColor Gray
        & $PythonExe -m pip install  --upgrade pip setuptools wheel --quiet
        
        # Install the package with dependencies
        Write-Host "  Installing causaliq-repo-template with dev dependencies..." -ForegroundColor Gray
        & $PythonExe -m pip install --force-reinstall -e ".[dev,test,docs]"
        
        deactivate
        Write-Host "  $DisplayName installation complete!" -ForegroundColor Green
    }
    catch {
        Write-Host "  Error installing in $DisplayName" -ForegroundColor Red
        deactivate
    }
}

# Function to setup environment
function Setup-PythonEnv {
    param(
        [string]$Version,
        [string]$DisplayName,
        [string]$EnvName
    )
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Setting up $DisplayName environment..." -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    
    try {
        $null = py -$Version --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            if (Test-Path "$VenvDir\$EnvName") {
                Write-Host "$DisplayName environment already exists" -ForegroundColor Yellow
            } else {
                py -$Version -m venv "$VenvDir\$EnvName"
                Write-Host "$DisplayName environment created in $VenvDir\$EnvName" -ForegroundColor Green
            }
            
            # Install packages if requested
            if ($Install -or $InstallOnly) {
                Install-InEnvironment -EnvName $EnvName -DisplayName $DisplayName
            }
        } else {
            Write-Host "Warning: $DisplayName not found, skipping..." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "Warning: $DisplayName not found, skipping..." -ForegroundColor Yellow
    }
}

# Handle different modes
if ($InstallOnly) {
    # Only install packages, don't create environments
    Write-Host "Installing causaliq-repo-template in all environments..." -ForegroundColor Blue
    
    Install-InEnvironment -EnvName "py39" -DisplayName "Python 3.9"
    Install-InEnvironment -EnvName "py310" -DisplayName "Python 3.10"
    Install-InEnvironment -EnvName "py311" -DisplayName "Python 3.11"
    Install-InEnvironment -EnvName "py312" -DisplayName "Python 3.12"
} else {
    # Create environments (and optionally install)
    Write-Host "Setting up Python virtual environments..." -ForegroundColor Blue
    if ($Install) {
        Write-Host "(Will also install causaliq-repo-template package)" -ForegroundColor Gray
    }
    
    # Setup all Python environments
    Setup-PythonEnv -Version "3.9" -DisplayName "Python 3.9" -EnvName "py39"
    Setup-PythonEnv -Version "3.10" -DisplayName "Python 3.10" -EnvName "py310"
    Setup-PythonEnv -Version "3.11" -DisplayName "Python 3.11" -EnvName "py311"
    Setup-PythonEnv -Version "3.12" -DisplayName "Python 3.12" -EnvName "py312"
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Operation complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

if (-not $InstallOnly) {
    Write-Host "To activate an environment:" -ForegroundColor White
    Write-Host "  .\scripts\activate.ps1 39     (Python 3.9)" -ForegroundColor Gray
    Write-Host "  .\scripts\activate.ps1 310    (Python 3.10)" -ForegroundColor Gray
    Write-Host "  .\scripts\activate.ps1 311    (Python 3.11)" -ForegroundColor Gray
    Write-Host "  .\scripts\activate.ps1 312    (Python 3.12)" -ForegroundColor Gray
    Write-Host ""
}

if (-not ($Install -or $InstallOnly)) {
    Write-Host "To install causaliq-repo-template in all environments:" -ForegroundColor White
    Write-Host "  .\scripts\setup-env.ps1 -InstallOnly" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "For help: .\scripts\setup-env.ps1 -Help" -ForegroundColor Cyan
Write-Host ""