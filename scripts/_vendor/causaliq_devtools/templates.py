# generated from causaliq@838155024dcf src/causaliq/devtools/templates.py - do not edit
"""Text templates for the generated scripts.

Rendering is a plain placeholder replacement rather than ``str.format`` because
the PowerShell templates contain braces of their own. ``sync`` prepends the
provenance header to every rendered file.
"""

from __future__ import annotations

from typing import Dict

from .config import CausaliqConfig

ACTIVATE = """# Usage: .\\scripts\\activate.ps1 [@@KEYS@@]
# Defaults to Python @@DEFAULT_PYTHON@@
param(
    [string]$Version = "@@DEFAULT_KEY@@"
)

$Environments = [ordered]@{
@@ENVIRONMENTS@@
}

if ($Version -in @("help", "?", "-h")) {
    Write-Host "Usage: .\\scripts\\activate.ps1 [@@KEYS@@]"
    Write-Host "Defaults to Python @@DEFAULT_PYTHON@@"
    foreach ($key in $Environments.Keys) {
        Write-Host ("  {0} - Python {1}" -f $key, $Environments[$key])
    }
    exit 0
}

if (-not $Environments.Contains($Version)) {
    Write-Host "Error: unknown Python version $Version"
    exit 1
}

$EnvPath = "venv\\py$Version"
if (-not (Test-Path $EnvPath)) {
    Write-Host "Error: $EnvPath does not exist"
    Write-Host "Run .\\scripts\\setup-env.ps1 first"
    exit 1
}

Write-Host "Activating Python $($Environments[$Version]) environment..."
& "$EnvPath\\Scripts\\Activate.ps1"
"""

SETUP_ENV = """# Usage: .\\scripts\\setup-env.ps1 [-InstallOnly] [-Help]
param(
    [switch]$InstallOnly,
    [switch]$Help
)

$Environments = [ordered]@{
@@ENVIRONMENTS@@
}

if ($Help) {
    Write-Host "Usage: .\\scripts\\setup-env.ps1 [-InstallOnly] [-Help]"
    Write-Host "Creates one virtual environment per supported Python version."
    exit 0
}

foreach ($key in $Environments.Keys) {
    $version = $Environments[$key]
    $envPath = "venv\\py$key"
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
    $python = "$envPath\\Scripts\\python.exe"
    & $python -m pip install --upgrade pip setuptools wheel --quiet
    & $python -m pip install -e ".[@@EXTRAS@@]"
}

Write-Host "Environment setup complete."
"""

PUBLISH = """# Usage: .\\scripts\\build-and-publish.ps1
$InitFile = Join-Path $PSScriptRoot "..\\src\\@@PACKAGE@@\\__init__.py"
$VersionLine = Get-Content $InitFile |
    Select-String -Pattern '__version__\\s*=\\s*"([^"]+)"'
if (-not $VersionLine) {
    Write-Host "Could not extract version from __init__.py"
    exit 1
}

$CurrentVersion = $VersionLine.Matches[0].Groups[1].Value
$Version = Read-Host "Enter version number (current: $CurrentVersion)"
if (-not $Version) { $Version = $CurrentVersion }

Write-Host "Building @@DISTRIBUTION@@ version $Version..."
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

python -m build
twine check dist\\*

$Publish = Read-Host "Publish to PyPI? (test/prod/no)"
switch ($Publish.ToLower()) {
    "test" { twine upload --repository testpypi dist\\* }
    "prod" { twine upload dist\\* }
    default { Write-Host "Package ready for manual upload." }
}
"""

WRAPPER = """param(
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
)

& python "$PSScriptRoot\\check_ci.py" @Args
exit $LASTEXITCODE
"""

ENTRY_POINT = '''"""@@DOC@@"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

from causaliq_devtools.@@MODULE@@ import @@FUNCTION@@  # noqa: E402

if __name__ == "__main__":
    sys.exit(@@FUNCTION@@())
'''


def activate_ps1(config: CausaliqConfig) -> str:
    """Return the generated environment activation script."""
    return _fill(ACTIVATE, _values(config))


def setup_env_ps1(config: CausaliqConfig) -> str:
    """Return the generated environment setup script."""
    return _fill(SETUP_ENV, _values(config))


def build_and_publish_ps1(config: CausaliqConfig) -> str:
    """Return the generated build and publish script."""
    return _fill(PUBLISH, _values(config))


def check_ci_wrapper() -> str:
    """Return the PowerShell wrapper that forwards to ``check_ci.py``."""
    return WRAPPER


def entry_point(module: str, function: str, doc: str) -> str:
    """Return a thin Python entry point for a vendored devtools module."""
    return _fill(
        ENTRY_POINT,
        {"MODULE": module, "FUNCTION": function, "DOC": doc},
    )


def _values(config: CausaliqConfig) -> Dict[str, str]:
    """Return the placeholder values shared by the script templates."""
    pairs = [
        (version.replace(".", ""), version)
        for version in config.python_versions
    ]
    lines = "\n".join(f'    "{key}" = "{version}"' for key, version in pairs)
    return {
        "ENVIRONMENTS": lines,
        "KEYS": "|".join(key for key, _ in pairs),
        "DEFAULT_KEY": config.default_python.replace(".", ""),
        "DEFAULT_PYTHON": config.default_python,
        "EXTRAS": config.extras_spec,
        "PACKAGE": config.package,
        "DISTRIBUTION": config.distribution,
    }


def _fill(template: str, values: Dict[str, str]) -> str:
    """Replace every ``@@NAME@@`` placeholder in ``template``."""
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"@@{name}@@", value)
    return rendered
