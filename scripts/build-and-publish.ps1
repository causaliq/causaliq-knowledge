# generated from causaliq@27a2f560f378 src/causaliq/devtools/templates.py - do not edit
# Usage: .\scripts\build-and-publish.ps1
$InitFile = Join-Path $PSScriptRoot "..\src\causaliq_knowledge\__init__.py"
$VersionLine = Get-Content $InitFile |
    Select-String -Pattern '__version__\s*=\s*"([^"]+)"'
if (-not $VersionLine) {
    Write-Host "Could not extract version from __init__.py"
    exit 1
}

$CurrentVersion = $VersionLine.Matches[0].Groups[1].Value
$Version = Read-Host "Enter version number (current: $CurrentVersion)"
if (-not $Version) { $Version = $CurrentVersion }

Write-Host "Building causaliq-knowledge version $Version..."
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

python -m build
twine check dist\*

$Publish = Read-Host "Publish to PyPI? (test/prod/no)"
switch ($Publish.ToLower()) {
    "test" { twine upload --repository testpypi dist\* }
    "prod" { twine upload dist\* }
    default { Write-Host "Package ready for manual upload." }
}
