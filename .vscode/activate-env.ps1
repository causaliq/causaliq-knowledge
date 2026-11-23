# Auto-activate Python 3.12 virtual environment for causaliq-knowledge
if (Test-Path ".\venv\py312\Scripts\Activate.ps1") {
    Write-Host "Activating Python 3.12 virtual environment..." -ForegroundColor Green
    & ".\venv\py312\Scripts\Activate.ps1"
    Write-Host "$([char]0x2713) causaliq-knowledge CLI is now available!" -ForegroundColor Green
} else {
    Write-Host "$([char]0x2717) Virtual environment not found. Run: scripts\setup-env -Install" -ForegroundColor Yellow
}