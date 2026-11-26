"""
betautils_hardware.py - what this machine can run, read without loading a model.

Small and stdlib-only at import time (onnxruntime is imported inside the
call that needs it), so the engine and the setup tools can ask cheaply.

  detect()          the GPU facts the model choice needs
  machine()         the whole picture for hw.detect: CPU, memory, GPU,
                    battery, free disk space per kind of folder
  safe_settings()   settings picked from those facts instantly, each
                    with its reason, for a machine nobody has measured

A usable NVIDIA GPU needs both halves: onnxruntime-gpu lists
CUDAExecutionProvider even on a machine with no GPU, and nvidia-smi can
see a GPU that a CPU-only onnxruntime cannot use.

Every reader takes its source as an argument (a /proc or /sys root, a
command runner) so tests feed it synthetic machines; the defaults read
the real one.
"""

import os
import platform
import subprocess
import sys

CUDA_PROVIDER = 'CUDAExecutionProvider'

GPU_FIELDS = ( 'name', 'driver_version', 'memory.total' )


def _run_quiet( command, timeout=30 ):
    """( exit code, stdout + stderr ), or ( None, why ) when it could not run."""
    try:
        done = subprocess.run( command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=timeout )
    except ( OSError, subprocess.SubprocessError ) as err:
        return None, str( err )
    return done.returncode, ( done.stdout or '' ) + ( done.stderr or '' )


def nvidia_smi( fields, run=_run_quiet ):
    """{ field: value } for the first GPU, or {} without nvidia-smi."""
    code, text = run( [ 'nvidia-smi', '--query-gpu=%s'%( ','.join( fields ), ),
                        '--format=csv,noheader,nounits' ] )
    if code != 0 or not text:
        return {}
    values = [ part.strip() for part in text.strip().splitlines()[0].split( ',' ) ]
    return dict( zip( fields, values ) )


def onnx_providers():
    """onnxruntime's available execution providers, or () when it is not installed."""
    try:
        import onnxruntime
    except ImportError:
        return ()
    try:
        return tuple( onnxruntime.get_available_providers() )
    except Exception:
        return ()


def cuda_usable( providers, gpu ):
    """True when onnxruntime can use CUDA and nvidia-smi sees a GPU."""
    return CUDA_PROVIDER in ( providers or () ) and bool( gpu )


def detect():
    """
    This machine's GPU facts, as plain data:
    { 'providers': [...], 'gpu': { 'name': ... } or None, 'cuda_usable': bool }.
    """
    providers = onnx_providers()
    gpu = nvidia_smi( [ 'name' ] ) or None
    return { 'providers': list( providers ), 'gpu': gpu, 'cuda_usable': cuda_usable( providers, gpu ) }


# ---------------------------------------------------------------------------
# The whole machine
# ---------------------------------------------------------------------------

def _int_or_none( value ):
    try:
        return int( float( value ) )
    except ( TypeError, ValueError ):
        return None


def meminfo_mb( proc_root='/proc' ):
    """{ 'MemTotal': MB, 'MemAvailable': MB, 'SwapTotal': MB, ... } from /proc/meminfo, or {}."""
    found = {}
    try:
        with open( os.path.join( proc_root, 'meminfo' ), 'r', encoding='ascii', errors='replace' ) as handle:
            for line in handle:
                name, _, rest = line.partition( ':' )
                parts = rest.split()
                if parts and parts[0].isdigit():
                    found[name.strip()] = int( parts[0] ) // 1024
    except OSError:
        return {}
    return found


def cpu_model_linux( proc_root='/proc' ):
    try:
        with open( os.path.join( proc_root, 'cpuinfo' ), 'r', encoding='ascii', errors='replace' ) as handle:
            for line in handle:
                if line.startswith( 'model name' ):
                    return line.split( ':', 1 )[1].strip()
    except OSError:
        pass
    return None


def battery_present_linux( sys_root='/sys' ):
    """True when /sys/class/power_supply lists a battery (type Battery)."""
    base = os.path.join( sys_root, 'class', 'power_supply' )
    try:
        names = os.listdir( base )
    except OSError:
        return False
    for name in names:
        try:
            with open( os.path.join( base, name, 'type' ), 'r', encoding='ascii' ) as handle:
                if handle.read().strip() == 'Battery':
                    return True
        except OSError:
            continue
    return False


def windows_memory_status():
    """GlobalMemoryStatusEx as { MemTotal, MemAvailable, SwapTotal } MB plus 'load' (percent), or {}."""
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX( ctypes.Structure ):
        _fields_ = [ ( 'dwLength', wintypes.DWORD ), ( 'dwMemoryLoad', wintypes.DWORD ),
                     ( 'ullTotalPhys', ctypes.c_ulonglong ), ( 'ullAvailPhys', ctypes.c_ulonglong ),
                     ( 'ullTotalPageFile', ctypes.c_ulonglong ), ( 'ullAvailPageFile', ctypes.c_ulonglong ),
                     ( 'ullTotalVirtual', ctypes.c_ulonglong ), ( 'ullAvailVirtual', ctypes.c_ulonglong ),
                     ( 'ullAvailExtendedVirtual', ctypes.c_ulonglong ) ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof( MEMORYSTATUSEX )
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx( ctypes.byref( status ) ):
        return {}
    return { 'MemTotal': status.ullTotalPhys // 2**20, 'MemAvailable': status.ullAvailPhys // 2**20,
             'SwapTotal': max( 0, status.ullTotalPageFile - status.ullTotalPhys ) // 2**20,
             'load': int( status.dwMemoryLoad ) }


def windows_power_status():
    """GetSystemPowerStatus as { battery, on_battery, percent }, or {}."""
    import ctypes
    from ctypes import wintypes

    class SYSTEM_POWER_STATUS( ctypes.Structure ):
        _fields_ = [ ( 'ACLineStatus', ctypes.c_ubyte ), ( 'BatteryFlag', ctypes.c_ubyte ),
                     ( 'BatteryLifePercent', ctypes.c_ubyte ), ( 'SystemStatusFlag', ctypes.c_ubyte ),
                     ( 'BatteryLifeTime', wintypes.DWORD ), ( 'BatteryFullLifeTime', wintypes.DWORD ) ]

    power = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus( ctypes.byref( power ) ):
        return {}
    battery = power.BatteryFlag not in ( 128, 255 )   # 128 no battery, 255 unknown
    return { 'battery': battery, 'on_battery': battery and power.ACLineStatus == 0,
             'percent': power.BatteryLifePercent if power.BatteryLifePercent <= 100 else None }


def _windows_cpu_model():
    try:
        import winreg
        with winreg.OpenKey( winreg.HKEY_LOCAL_MACHINE,
                             r'HARDWARE\DESCRIPTION\System\CentralProcessor\0' ) as key:
            return str( winreg.QueryValueEx( key, 'ProcessorNameString' )[0] ).strip()
    except OSError:
        return None


def platform_facts( platform_name, proc_root, sys_root ):
    """( cpu model, meminfo-style MB dict, battery present )."""
    if platform_name.startswith( 'linux' ):
        return cpu_model_linux( proc_root ), meminfo_mb( proc_root ), battery_present_linux( sys_root )
    if platform_name == 'win32':
        try:
            return ( _windows_cpu_model(), windows_memory_status(),
                     bool( windows_power_status().get( 'battery' ) ) )
        except Exception:
            return None, {}, False
    return None, {}, False


def disks( kinds=None, usage=None, stat=os.stat ):
    """
    [ { kind, volume, free_bytes, total_bytes } ] for each kind of folder.
    volume numbers the distinct file systems (1, 2, ...) so kinds on the
    same disk share one. A folder that does not exist yet is read at its
    nearest existing parent. Paths are not reported.
    """
    import shutil
    usage = usage or shutil.disk_usage
    if kinds is None:
        import betautils_paths as bu_paths
        kinds = { kind: bu_paths.kind_dir( kind ) for kind in bu_paths.KINDS }
    volumes = {}
    found = []
    for kind, folder in kinds.items():
        probe = os.path.abspath( folder )
        while not os.path.exists( probe ) and os.path.dirname( probe ) != probe:
            probe = os.path.dirname( probe )
        try:
            device = stat( probe ).st_dev
            total, _used, free = usage( probe )
        except OSError:
            continue
        volume = volumes.setdefault( device, len( volumes ) + 1 )
        found.append( { 'kind': kind, 'volume': volume, 'free_bytes': int( free ), 'total_bytes': int( total ) } )
    return found


def machine( platform_name=None, proc_root='/proc', sys_root='/sys', run=_run_quiet,
             providers=None, cpu_count=None, disk_kinds=None, usage=None ):
    """
    hw.detect's facts:
      os           { platform: linux|win32|darwin|..., release }
      cpu          { model, threads }
      memory       { total_mb, available_mb, swap_mb }
      gpu          { name, driver, vram_mb } or None
      cuda_usable, providers
      battery      True when the machine has a battery
      disks        see disks()
    Unknown values are None.
    """
    platform_name = platform_name or sys.platform
    model, memory, battery = platform_facts( platform_name, proc_root, sys_root )
    providers = list( onnx_providers() if providers is None else providers )
    raw_gpu = nvidia_smi( list( GPU_FIELDS ), run=run )
    gpu = None
    if raw_gpu:
        gpu = { 'name': raw_gpu.get( 'name' ) or None, 'driver': raw_gpu.get( 'driver_version' ) or None,
                'vram_mb': _int_or_none( raw_gpu.get( 'memory.total' ) ) }
    try:
        disk_list = disks( disk_kinds, usage=usage )
    except Exception:
        disk_list = []
    return {
        'os': { 'platform': platform_name, 'release': platform.release() or None },
        'cpu': { 'model': model, 'threads': cpu_count if cpu_count is not None else os.cpu_count() },
        'memory': { 'total_mb': memory.get( 'MemTotal' ), 'available_mb': memory.get( 'MemAvailable' ),
                    'swap_mb': memory.get( 'SwapTotal' ) },
        'gpu': gpu,
        'cuda_usable': cuda_usable( providers, gpu ),
        'providers': providers,
        'battery': bool( battery ),
        'disks': disk_list,
    }


# ---------------------------------------------------------------------------
# Safe settings, picked instantly
# ---------------------------------------------------------------------------

def safe_settings( facts, config ):
    """
    [ { setting, value, current, code, params } ] for each setting this
    machine's facts settle without measuring, where the value in effect
    differs. config is the betaconfig module (or a stand-in).

    Only settings that cannot change what is detected are picked here.
    gpu_enabled moves the same model between providers; the detection
    batch size is left to Optimize, which checks that the detections
    stay the same at the size it proposes. Render workers need nothing:
    render_workers = 0 already follows the cores and the free memory at
    every render.
    """
    picks = []
    usable = bool( facts.get( 'cuda_usable' ) )
    current = getattr( config, 'gpu_enabled', 0 )
    if usable and current != 1:
        picks.append( { 'setting': 'gpu_enabled', 'value': 1, 'current': current,
                        'code': 'hw.gpu_usable',
                        'params': { 'gpu': ( facts.get( 'gpu' ) or {} ).get( 'name' ) } } )
    elif not usable and current == 1:
        picks.append( { 'setting': 'gpu_enabled', 'value': 0, 'current': current,
                        'code': 'hw.gpu_not_usable',
                        'params': { 'provider': CUDA_PROVIDER in ( facts.get( 'providers' ) or () ),
                                    'gpu': bool( facts.get( 'gpu' ) ) } } )
    return picks
