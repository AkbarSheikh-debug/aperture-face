# One command to run the face-recognition web app on Windows:   .\run_app.ps1   (optional: -Port 8080, -NoBrowser)
# Creates a private virtual environment (.venv-app), installs the 6 runtime packages, starts the app and opens the browser.
param([int]$Port = 8000, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$py = $null
foreach ($c in @('python', 'py')) { if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break } }
if (-not $py) { Write-Host 'Python 3.11+ was not found. Install it from https://www.python.org/downloads/ (tick "Add to PATH") and run this again.' -ForegroundColor Red; exit 1 }

if (-not (Test-Path '.venv-app\Scripts\python.exe')) {
    Write-Host 'Creating virtual environment (first run only)...'
    & $py -m venv .venv-app
}
$venvPy = '.venv-app\Scripts\python.exe'
Write-Host 'Installing packages (first run takes about a minute)...'
& $venvPy -m pip install --quiet --disable-pip-version-check -r requirements-app.txt
if ($LASTEXITCODE -ne 0) { Write-Host 'Package install failed. Python 3.12 is the tested version.' -ForegroundColor Red; exit 1 }

$url = "http://localhost:$Port"
Write-Host "Starting Aperture at $url  (press Ctrl+C to stop)" -ForegroundColor Green
if (-not $NoBrowser) { Start-Job -ScriptBlock { param($u) Start-Sleep -Seconds 4; Start-Process $u } -ArgumentList $url | Out-Null }
& $venvPy -m uvicorn api.app:app --host 127.0.0.1 --port $Port
