# catalog_rebuild.ps1 - build the library catalog again from the disk.
#
#   powershell -ExecutionPolicy Bypass -File tools\setup\catalog_rebuild.ps1         says what it will do, asks
#   powershell -ExecutionPolicy Bypass -File tools\setup\catalog_rebuild.ps1 --yes   rebuilds without asking
#
# Close the app first. Folders, pins, history and feedback are carried from the old
# file while it can be read; the old file is kept beside the new one. Log:
# <state folder>\logs\catalog_rebuild.log; --console-level (info) filters
# the terminal only. Every argument goes to catalog_rebuild.py.

$ErrorActionPreference = 'Stop'
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { Join-Path $RepoDir '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "no Python at $Python; run setup.ps1 first, or set PYTHON_BIN"
    exit 1
}
Set-Location $RepoDir
& $Python (Join-Path $RepoDir 'tools\setup\catalog_rebuild.py') @args
exit $LASTEXITCODE
