# migrate_1x.ps1 - copy a 1.x tree's settings, models, caches and logs into
# the 2.x folders by kind. Never moves or changes the 1.x tree.
#
#   powershell -ExecutionPolicy Bypass -File tools\setup\migrate_1x.ps1             asks, shows the plan, asks before copying
#   powershell -ExecutionPolicy Bypass -File tools\setup\migrate_1x.ps1 --dry-run   the plan only
#
# Safe to run again: only what is missing is copied. Log:
# <state folder>\logs\migrate_1x.log, every level; --console-level (info)
# filters the terminal only. Every argument goes to migrate_1x.py.

$ErrorActionPreference = 'Stop'
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { Join-Path $RepoDir '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "no Python at $Python; run setup.ps1 first, or set PYTHON_BIN"
    exit 1
}
Set-Location $RepoDir
& $Python (Join-Path $RepoDir 'tools\setup\migrate_1x.py') @args
exit $LASTEXITCODE
