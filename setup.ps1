# setup.ps1 - one-shot BetaSuite environment setup for Windows.
# Double-click setup.cmd, or run: powershell -ExecutionPolicy Bypass -File setup.ps1
# Safe to re-run: every step checks before it acts. See SETUP.md.

[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$Cpu,
    [string]$Python = '',
    [switch]$Vision,               # also install betavision deps (mss, pywin32)
    [switch]$Tests,
    [switch]$NoTests,
    [switch]$Yes,
    [ValidateSet('trace','debug','info','warn','error')]
    [string]$LogLevel = 'info',
    [switch]$Help
)

if ($Help) {
    @'
Usage: setup.cmd [options]   (or: powershell -ExecutionPolicy Bypass -File setup.ps1 [options])

  -Gpu | -Cpu            CUDA build or CPU build (default: ask; GPU suggested when nvidia-smi finds one)
  -Python VERSION        Python version (default: from .python-version)
  -Vision                also install betavision-* dependencies (mss, pywin32)
  -Tests | -NoTests      run the unit tests at the end (default: ask, no)
  -Yes                   accept the default answer to every prompt
  -LogLevel LEVEL        console level: trace|debug|info|warn|error (default info); setup.log gets everything
'@ | Write-Host
    exit 0
}

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$RepoDir = $PSScriptRoot
Set-Location $RepoDir
$LogFile = Join-Path $RepoDir 'setup.log'
$VenvDir = Join-Path $RepoDir '.venv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
$MinDriverMajor = 528
if (-not $Python) {
    $pin = Join-Path $RepoDir '.python-version'
    $Python = if (Test-Path $pin) { (Get-Content $pin -Raw).Trim() } else { '3.12' }
}

# ---------------------------------------------------------------- logging
$LevelRank = @{ trace = 0; debug = 1; info = 2; warn = 3; error = 4 }
function Write-Log([string]$Level, [string]$Message) {
    $line = "[{0}] [{1}] {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level.ToUpper(), $Message
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    if ($LevelRank[$Level] -ge $LevelRank[$LogLevel]) {
        $color = @{ trace = 'DarkGray'; debug = 'Gray'; info = 'White'; warn = 'Yellow'; error = 'Red' }[$Level]
        Write-Host $line -ForegroundColor $color
    }
}
function Stop-Setup([string]$Message) {
    Write-Log error $Message
    Write-Log error "setup stopped. full log: $LogFile"
    exit 1
}

# Runs a native command with all of its output appended to the log only.
function Invoke-Logged([string]$Description, [string]$Exe, [string[]]$Arguments) {
    Write-Log info "=== starting: $Description ==="
    Write-Log debug "command: $Exe $($Arguments -join ' ')"
    Add-Content -Path $LogFile -Value "----- begin output: $Description -----" -Encoding UTF8
    $start = Get-Date
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } | Add-Content -Path $LogFile -Encoding UTF8
        $code = $LASTEXITCODE
    } catch {
        Add-Content -Path $LogFile -Value "$_" -Encoding UTF8
        $code = 1
    } finally {
        $ErrorActionPreference = $prev
    }
    $secs = [int]((Get-Date) - $start).TotalSeconds
    Add-Content -Path $LogFile -Value "----- end output: $Description (exit $code) -----" -Encoding UTF8
    if ($code -eq 0) { Write-Log info "=== finished: $Description (${secs}s) ===" }
    else { Write-Log error "=== failed: $Description (exit $code, ${secs}s) - see $LogFile ===" }
    return ($code -eq 0)
}

function Ask-YesNo([string]$Prompt, [bool]$Default) {
    $d = if ($Default) { 'y' } else { 'n' }
    if ($Yes -or -not [Environment]::UserInteractive) {
        Write-Log info "auto-answer '$d': $Prompt"
        return $Default
    }
    $hint = if ($Default) { 'Y/n' } else { 'y/N' }
    Write-Log debug "AWAITING INPUT: $Prompt"
    $reply = Read-Host "$Prompt [$hint]"
    if (-not $reply) { $reply = $d }
    Write-Log debug "input received: $reply"
    return ($reply -match '^(y|yes)$')
}

function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
}

# ---------------------------------------------------------------- preflight
Write-Log info "BetaSuite setup starting in $RepoDir (log: $LogFile)"
Write-Log info ("platform: Windows {0} {1}, PowerShell {2}" -f [Environment]::OSVersion.Version, $env:PROCESSOR_ARCHITECTURE, $PSVersionTable.PSVersion)

$HaveGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpus = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpus) {
        $HaveGpu = $true
        foreach ($g in @($gpus)) { Write-Log info "GPU: $g" }
        $major = [int](($gpus | Select-Object -First 1).Split(',')[1].Trim().Split('.')[0])
        if ($major -lt $MinDriverMajor) {
            Write-Log warn "NVIDIA driver $major is older than $MinDriverMajor; CUDA 12 needs $MinDriverMajor+. Update the driver or use -Cpu."
        }
    }
}
if (-not $HaveGpu) { Write-Log info 'no working nvidia-smi found; GPU build not suggested' }

if ($Gpu -and $Cpu) { Stop-Setup 'pass -Gpu or -Cpu, not both' }
if ($Gpu) { $Mode = 'gpu' } elseif ($Cpu) { $Mode = 'cpu' }
else { $Mode = if (Ask-YesNo "Install the NVIDIA GPU (CUDA) build? 'n' installs the CPU build" $HaveGpu) { 'gpu' } else { 'cpu' } }
if ($Mode -eq 'gpu' -and -not $HaveGpu) { Write-Log warn 'GPU build requested but no NVIDIA GPU detected; it will fall back to CPU at runtime' }
Write-Log info "install mode: $Mode, python $Python"

$HaveWinget = [bool](Get-Command winget -ErrorAction SilentlyContinue)

# ---------------------------------------------------------------- ffmpeg
if ((Get-Command ffmpeg -ErrorAction SilentlyContinue) -and (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Log info ("ffmpeg found: " + (& ffmpeg -version 2>$null | Select-Object -First 1))
} elseif (Ask-YesNo 'ffmpeg/ffprobe not found on PATH. Install with winget (Gyan.FFmpeg)?' $true) {
    if (-not $HaveWinget) { Stop-Setup 'winget is not available. Install ffmpeg by hand (see SETUP.md) and re-run.' }
    if (-not (Invoke-Logged 'install ffmpeg' 'winget' @('install','--id','Gyan.FFmpeg','-e','--accept-source-agreements','--accept-package-agreements'))) {
        Stop-Setup 'ffmpeg install failed'
    }
    Update-SessionPath
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Write-Log warn 'ffmpeg installed but not on PATH in this window yet; open a new terminal before running BetaSuite'
    }
} else {
    Write-Log warn 'continuing without ffmpeg; betatv.py cannot render until it is installed'
}

# ---------------------------------------------------------------- uv
function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}
$Uv = Find-Uv
if (-not $Uv) {
    if (-not (Ask-YesNo 'uv (Python installer and venv tool from astral.sh) is not installed. Install it?' $true)) {
        Stop-Setup 'uv is required by setup.ps1. Install it, or follow the manual steps in SETUP.md.'
    }
    $ok = Invoke-Logged 'install uv' 'powershell' @('-NoProfile','-ExecutionPolicy','Bypass','-Command','irm https://astral.sh/uv/install.ps1 | iex')
    Update-SessionPath
    $Uv = Find-Uv
    if (-not $ok -or -not $Uv) { Stop-Setup 'uv install failed' }
}
Write-Log info "uv: $Uv"

# ---------------------------------------------------------------- python + venv
$existing = $null
if (Test-Path $VenvPy) {
    $existing = (& $VenvPy -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
}
if ($existing -and $existing -ne $Python) {
    if (Ask-YesNo ".venv uses Python $existing, not $Python. Delete and recreate it?" $true) {
        Remove-Item -Recurse -Force $VenvDir
        Write-Log info 'removed old .venv'
        $existing = $null
    } else {
        Write-Log warn "keeping .venv on Python $existing"
    }
}
if ($existing) {
    Write-Log info "reusing .venv (Python $existing)"
} else {
    if (-not (Invoke-Logged "uv python install $Python" $Uv @('python','install',$Python))) { Stop-Setup "could not install Python $Python" }
    if (-not (Invoke-Logged 'create .venv' $Uv @('venv','--python',$Python,$VenvDir))) { Stop-Setup 'could not create .venv' }
}

# onnxruntime and onnxruntime-gpu share one import directory: switching
# flavours must remove both first or the survivor ends up half-deleted.
$other = if ($Mode -eq 'gpu') { 'onnxruntime' } else { 'onnxruntime-gpu' }
& $VenvPy -c "import importlib.metadata as m, sys; m.version(sys.argv[1])" $other 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Log info "switching onnxruntime flavour: removing $other and any existing onnxruntime"
    Invoke-Logged 'uninstall onnxruntime packages' $Uv @('pip','uninstall','--python',$VenvPy,'onnxruntime','onnxruntime-gpu') | Out-Null
}

if (-not (Invoke-Logged "install requirements-$Mode.txt" $Uv @('pip','install','--python',$VenvPy,'-r',"requirements-$Mode.txt"))) {
    Stop-Setup 'package install failed'
}
if ($Vision -or (Ask-YesNo 'Install betavision-* live screen censoring dependencies (mss, pywin32)?' $false)) {
    Invoke-Logged 'install requirements-vision-windows.txt' $Uv @('pip','install','--python',$VenvPy,'-r','requirements-vision-windows.txt') | Out-Null
}

# ---------------------------------------------------------------- folders
$Parent = Split-Path $RepoDir -Parent
foreach ($d in @('resources\model','resources\uncensored_vids','resources\uncensored_pics','resources\source',
                 'resources\stickers\breasts','resources\stickers\vulva','output')) {
    $full = Join-Path $Parent $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Log info "created $full"
    }
}

# ---------------------------------------------------------------- models
# 320n downloads automatically; 640m and RetinaNet are manual (GitHub only
# serves them to signed-in users). fetch_models.py checks whatever is there
# and moves bad files (e.g. a saved sign-in page) aside.
& $VenvPy 'tools\setup\fetch_models.py' '--log-level' $LogLevel '--log-file' $LogFile
if ($LASTEXITCODE -ne 0) { Write-Log warn 'model check reported a problem; see the messages above' }

# ---------------------------------------------------------------- betaconfig.gpu_enabled
$wantFlag = if ($Mode -eq 'gpu') { '1' } else { '0' }
$cfgPath = Join-Path $RepoDir 'betaconfig.py'
$cfg = [IO.File]::ReadAllText($cfgPath)
$match = [regex]::Match($cfg, '(?m)^gpu_enabled(\s*=\s*)([01])')
if ($match.Success -and $match.Groups[2].Value -ne $wantFlag) {
    if (Ask-YesNo "betaconfig.py has gpu_enabled = $($match.Groups[2].Value) but this is a $Mode install. Set it to $wantFlag?" $true) {
        $cfg = [regex]::Replace($cfg, '(?m)^gpu_enabled(\s*=\s*)[01]', "gpu_enabled`${1}$wantFlag")
        [IO.File]::WriteAllText($cfgPath, $cfg, (New-Object Text.UTF8Encoding $false))
        Write-Log info "betaconfig.py: gpu_enabled = $wantFlag"
    } else {
        Write-Log warn "left gpu_enabled = $($match.Groups[2].Value) in betaconfig.py"
    }
}

# ---------------------------------------------------------------- verify
Write-Log info '=== verifying environment ==='
& $VenvPy 'tools\setup\verify_env.py' "--$Mode" '--log-level' $LogLevel '--log-file' $LogFile
$verifyStatus = $LASTEXITCODE

$runTests = if ($Tests) { $true } elseif ($NoTests) { $false } else { Ask-YesNo 'Run the unit tests now (about 15 seconds)?' $false }
if ($runTests) {
    if (-not (Invoke-Logged 'unit tests' $VenvPy @('-m','unittest','discover','-s','tests','-t','.','-p','test_*.py'))) {
        Write-Log warn "some tests failed; see $LogFile"
    }
}

# ---------------------------------------------------------------- done
if ($verifyStatus -ne 0) { Write-Log error "setup finished with problems (see FAIL lines above and $LogFile)" }
else { Write-Log info 'setup complete' }
Write-Host @"

Next steps:
  .venv\Scripts\Activate.ps1          # PowerShell, once per terminal (cmd: .venv\Scripts\activate.bat)
  # put videos in $(Join-Path $Parent 'resources\uncensored_vids')
  python betatv.py --preview on --preview-seconds 20

A script cannot activate a venv in the window that launched it, so the
activate line above is yours to run. Alternatively call .venv\Scripts\python.exe directly.
"@
exit $verifyStatus
