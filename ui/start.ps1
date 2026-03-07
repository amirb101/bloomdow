# Start both the FastAPI backend and Vite dev server (Windows)
# Run from bloomdow/ui/ or bloomdow/

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bloomdowRoot = Split-Path -Parent $scriptDir
$venvPath = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"
$frontendDir = Join-Path $scriptDir "frontend"
$nodeModules = Join-Path $frontendDir "node_modules"

function Write-Step { param($msg) Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Ok { param($msg) Write-Host "  $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "  $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "  Bloomdow UI" -ForegroundColor White
Write-Host ""

# --- Dependency checks ---
Write-Step "Checking dependencies..."

# Python 3.11+
$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $v = & $cmd --version 2>&1
        if ($v -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 11) {
                $pythonCmd = $cmd
                break
            }
        }
    } catch { continue }
}
if (-not $pythonCmd) {
    Write-Fail "Python 3.11+ required. Install from https://www.python.org/downloads/"
    exit 1
}
Write-Ok "Python: found"

# Node.js 18+
try {
    $nodeVer = node --version 2>&1
    if ($nodeVer -match "v(\d+)") {
        $major = [int]$Matches[1]
        if ($major -lt 18) {
            Write-Fail "Node.js 18+ required (found v$major). Install from https://nodejs.org/"
            exit 1
        }
    }
} catch {
    Write-Fail "Node.js required. Install from https://nodejs.org/"
    exit 1
}
Write-Ok "Node.js: $nodeVer"

# npm
try {
    $npmVer = npm --version 2>&1
} catch {
    Write-Fail "npm required (usually bundled with Node.js)"
    exit 1
}
Write-Ok "npm: $npmVer"

# --- Venv setup ---
if (-not (Test-Path $venvPython)) {
    Write-Step "Creating Python virtualenv..."
    Push-Location $bloomdowRoot
    try {
        & $pythonCmd -m venv (Join-Path $scriptDir ".venv")
        if (-not (Test-Path $venvPip)) {
            Write-Fail "Failed to create virtualenv"
            exit 1
        }
        Write-Ok "Virtualenv created"
        Write-Step "Installing backend dependencies..."
        & $venvPip install -e . -r (Join-Path $scriptDir "backend\requirements.txt")
        Write-Ok "Backend deps installed"
    } finally {
        Pop-Location
    }
} else {
    Write-Ok "Virtualenv: exists"
}

# --- Frontend deps ---
if (-not (Test-Path $nodeModules)) {
    Write-Step "Installing frontend dependencies..."
    Push-Location $frontendDir
    try {
        npm install
        Write-Ok "Frontend deps installed"
    } finally {
        Pop-Location
    }
} else {
    Write-Ok "Frontend deps: exists"
}

# --- Start servers ---
Write-Host ""
Write-Host "  Starting servers..." -ForegroundColor Cyan
Write-Host ""

$backendJob = $null
try {
    $backendDir = Join-Path $scriptDir "backend"
    $backendJob = Start-Job -ScriptBlock {
        param($venvPy, $backendDir)
        Set-Location $backendDir
        & $venvPy -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    } -ArgumentList $venvPython, $backendDir

    Write-Host "  Backend  -> http://localhost:8000" -ForegroundColor Green
    Write-Host "  Frontend -> http://localhost:5173" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Press Ctrl-C to stop both servers." -ForegroundColor Gray
    Write-Host ""

    Start-Sleep -Seconds 2

    Push-Location $frontendDir
    npm run dev
} finally {
    if ($backendJob) {
        Stop-Job $backendJob -ErrorAction SilentlyContinue
        Remove-Job $backendJob -Force -ErrorAction SilentlyContinue
    }
    Pop-Location -ErrorAction SilentlyContinue
}
