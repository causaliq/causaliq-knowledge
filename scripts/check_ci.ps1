# Pre-commit CI checks script
# Runs the same checks as GitHub Actions locally
#
# Parameters:
#   -Fast: Skip type checking and run only unit/functional tests (fastest)
#   -Fix: Auto-fix formatting issues instead of just checking
#   -IncludeSlow: Include slow tests (system resource tests, disabled by default)
#
# Default behavior: Full checks with coverage but excluding slow tests for faster development

param(
    [switch]$Fast,
    [switch]$Fix,
    [switch]$IncludeSlow
)

# Ensure we're using a Python virtualenvironment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating Python 3.11 environment..." -ForegroundColor Yellow
    if (Test-Path ".\venv\py311\Scripts\Activate.ps1") {
        & ".\venv\py311\Scripts\Activate.ps1"
    } else {
        Write-Error "Python 3.11 virtual environment not found!"
        exit 1
    }
}

Write-Host "Running CI checks locally..." -ForegroundColor Green
Write-Host "==================================================="

$allPassed = $true

function Test-Command {
    param(
        [string]$Command,
        [string]$Description
    )
    
    Write-Host ""
    Write-Host "Checking: $Description..." -ForegroundColor Cyan
    Write-Host "Running: $Command" -ForegroundColor Gray
    
    try {
        # Reset LASTEXITCODE to ensure clean state
        $global:LASTEXITCODE = 0
        
        # Execute command and capture exit code, but let output go to console
        Invoke-Expression $Command | Out-Host
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host "PASSED: $Description" -ForegroundColor Green
            return $true
        } else {
            Write-Host "FAILED: $Description (Exit Code: $exitCode)" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "ERROR: $Description - $_" -ForegroundColor Red
        return $false
    }
}

# 1. Black formatting
if ($Fix) {
    $blackResult = Test-Command "python -m black src/ tests/" "Auto-formatting with Black"
} else {
    $blackResult = Test-Command "python -m black --check src/ tests/" "Black code formatting check"
}
$allPassed = $allPassed -and $blackResult

# 2. Import sorting
if ($Fix) {
    $isortResult = Test-Command "python -m isort src/ tests/" "Auto-sorting imports with isort"
} else {
    $isortResult = Test-Command "python -m isort --check-only src/ tests/" "isort import sorting check"
}
$allPassed = $allPassed -and $isortResult

# 3. Flake8 style checking
$flake8Result = Test-Command "python -m flake8 src/ tests/" "Flake8 style checking"
$allPassed = $allPassed -and $flake8Result

# 4. Type checking (unless Fast)
$mypyResult = $true  # Default to true for Fast mode
if (-not $Fast) {
    $mypyResult = Test-Command "python -m mypy src/" "MyPy type checking"
    $allPassed = $allPassed -and $mypyResult
}

# 5. Tests
if ($Fast) {
    $testResult = Test-Command "python -m pytest -v tests/unit/ tests/functional/" "Fast tests"
} elseif ($IncludeSlow) {
    $testResult = Test-Command "python -m pytest -v --cov=src/causaliq_knowledge --cov-report=term-missing" "Full test suite with slow tests"
} else {
    $testResult = Test-Command "python -m pytest -v --cov=src/causaliq_knowledge --cov-report=term-missing -m 'not slow'" "Full test suite (excluding slow tests)"
}
$allPassed = $allPassed -and $testResult

# Summary
Write-Host ""
Write-Host "==================================================="
Write-Host "CI Check Results:" -ForegroundColor Yellow

# Show detailed results
Write-Host ""
Write-Host "Black formatting:       " -NoNewline
if ($blackResult) { Write-Host "PASSED" -ForegroundColor Green } else { Write-Host "FAILED" -ForegroundColor Red }

Write-Host "Import sorting:         " -NoNewline  
if ($isortResult) { Write-Host "PASSED" -ForegroundColor Green } else { Write-Host "FAILED" -ForegroundColor Red }

Write-Host "Flake8 style:           " -NoNewline
if ($flake8Result) { Write-Host "PASSED" -ForegroundColor Green } else { Write-Host "FAILED" -ForegroundColor Red }

if (-not $Fast) {
    Write-Host "MyPy type checking:     " -NoNewline
    if ($mypyResult) { Write-Host "PASSED" -ForegroundColor Green } else { Write-Host "FAILED" -ForegroundColor Red }
}

Write-Host "Tests:                  " -NoNewline
if ($testResult) { Write-Host "PASSED" -ForegroundColor Green } else { Write-Host "FAILED" -ForegroundColor Red }

if ($allPassed) {
    Write-Host ""
    Write-Host "$([char]0x2713) All checks passed! GitHub Actions should succeed." -ForegroundColor Green
    Write-Host "Safe to commit and push." -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "$([char]0x2717) Some checks failed. Fix issues before committing." -ForegroundColor Red
    if (-not $Fix) {
        Write-Host "Try running with -Fix to auto-fix formatting issues." -ForegroundColor Yellow
    }
    exit 1
}