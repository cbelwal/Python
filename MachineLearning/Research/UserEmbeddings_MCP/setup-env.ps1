[CmdletBinding()]
param(
    [string]$PythonCommand = "py"
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$activatePath = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
    throw "Python command '$PythonCommand' was not found. Install Python or pass -PythonCommand with a valid executable."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating virtual environment at '$venvPath'..."
    & $PythonCommand -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }
}
else {
    Write-Host "Using existing virtual environment at '$venvPath'."
}

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

if (-not (Test-Path -LiteralPath $requirementsPath)) {
    throw "Requirements file not found at '$requirementsPath'."
}

Write-Host "Installing project dependencies..."
& $venvPython -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install project dependencies."
}

Write-Host ""
Write-Host "Virtual environment setup completed."
Write-Host "Activate it with:"
Write-Host "  & `"$activatePath`""
