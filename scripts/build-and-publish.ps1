# Build and publish script for Windows
# This script builds the package and optionally publishes to PyPI

$CurrentVersion = "0.1.0"
$Version = Read-Host "Enter version number (current: $CurrentVersion)"
if (-not $Version) { $Version = $CurrentVersion }

Write-Host "Building causaliq-knowledge version $Version..." -ForegroundColor Blue

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "src\causaliq-knowledge.egg-info") { Remove-Item -Recurse -Force "src\causaliq-knowledge.egg-info" }

# Build package
Write-Host "Building package..." -ForegroundColor Blue
python -m build

# Check package
Write-Host "Checking package..." -ForegroundColor Blue
twine check dist\*

Write-Host ""
Write-Host "Package built successfully!" -ForegroundColor Green
Write-Host "Files in dist/:" -ForegroundColor White
Get-ChildItem dist

Write-Host ""
$Publish = Read-Host "Publish to PyPI? (test/prod/no)"

switch ($Publish.ToLower()) {
    "test" {
        Write-Host "Publishing to Test PyPI..." -ForegroundColor Blue
        twine upload --repository testpypi dist\*
    }
    "prod" {
        Write-Host "Publishing to Production PyPI..." -ForegroundColor Blue
        twine upload dist\*
    }
    default {
        Write-Host "Package ready for manual upload." -ForegroundColor Yellow
        Write-Host "To upload to Test PyPI: twine upload --repository testpypi dist\*" -ForegroundColor Gray
        Write-Host "To upload to Production PyPI: twine upload dist\*" -ForegroundColor Gray
    }
}