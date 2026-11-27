# catalog_backup.ps1 - back up the library catalog, or put a backup back.
#
#   powershell -ExecutionPolicy Bypass -File tools\setup\catalog_backup.ps1                       asks which, and where
#   powershell -ExecutionPolicy Bypass -File tools\setup\catalog_backup.ps1 backup [--to PATH]    copies the catalog
#   powershell -ExecutionPolicy Bypass -File tools\setup\catalog_backup.ps1 restore FILE [--yes]  puts FILE back
#
# A backup carries the watched folders, type pins, run history and
# feedback, and is safe while the app runs. Restore needs the app closed
# and keeps the current catalog beside the new one. Log:
# <state folder>\logs\catalog_backup.log; --console-level (info) filters
# the terminal only. Every argument goes to catalog_backup.py.

$ErrorActionPreference = 'Stop'
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { Join-Path $RepoDir '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "no Python at $Python; run setup.ps1 first, or set PYTHON_BIN"
    exit 1
}
Set-Location $RepoDir
& $Python (Join-Path $RepoDir 'tools\setup\catalog_backup.py') @args
exit $LASTEXITCODE
