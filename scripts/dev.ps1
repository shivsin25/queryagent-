# Starts the API (port 8000) and the Streamlit UI (port 8501) in separate windows.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - add an LLM API key to it (see the comments inside), then run this script again." -ForegroundColor Yellow
    exit 1
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; uv run qt serve --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; uv run streamlit run frontend/app.py"
Write-Host "API:  http://localhost:8000/docs"
Write-Host "UI:   http://localhost:8501"
