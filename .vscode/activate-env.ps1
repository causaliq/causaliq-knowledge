# Auto-activate Python 3.11 virtual environment for zenodo-sync
if (Test-Path ".\venv\py311\Scripts\Activate.ps1") {
    Write-Host "Activating Python 3.11 virtual environment..." -ForegroundColor Green
    & ".\venv\py311\Scripts\Activate.ps1"
    Write-Host "$([char]0x2713) causaliq-repo-template CLI is now available!" -ForegroundColor Green
} else {
    Write-Host "$([char]0x2717) Virtual environment not found. Run: scripts\setup-env -Install" -ForegroundColor Yellow
}