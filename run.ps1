# MuJoCo 环境检测入口（Windows PowerShell）。
# 依次探测 py -3 / python / python3，做最基本的版本预检后交给 main.py，并透传退出码。
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MainPy = Join-Path $ScriptDir "main.py"

function Test-UsablePython {
    param([string]$Exe, [string[]]$PrefixArgs)
    try {
        & $Exe @PrefixArgs -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)" 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$python = $null
$prefixArgs = @()

if ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-UsablePython -Exe "py" -PrefixArgs @("-3"))) {
    $python = "py"; $prefixArgs = @("-3")
} elseif ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-UsablePython -Exe "python" -PrefixArgs @())) {
    $python = "python"; $prefixArgs = @()
} elseif ((Get-Command python3 -ErrorAction SilentlyContinue) -and (Test-UsablePython -Exe "python3" -PrefixArgs @())) {
    $python = "python3"; $prefixArgs = @()
}

if (-not $python) {
    Write-Host "ERROR: no usable Python found (the checker itself needs Python >= 3.8)." -ForegroundColor Red
    Write-Host "       Install Python 3.10+ (required by mujoco), then re-run .\run.ps1"
    exit 2
}

& $python @prefixArgs $MainPy @args
exit $LASTEXITCODE
