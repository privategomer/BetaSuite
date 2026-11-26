"""
betautils_resources.py - what the machine is doing right now, read about
once a second for the governor and the Performance page.

  Signals         one snapshot: memory pressure and free memory, battery,
                  other programs' CPU use, input idle time, the GPU, this
                  engine's own CPU and memory, free disk space
  LinuxReader     /proc, /proc/pressure/memory and /sys/class/power_supply
  WindowsReader   the Win32 API through ctypes
  NullReader      every value unknown (other platforms)
  GpuStream       one long-lived nvidia-smi that prints a line a second,
                  so the GPU is read without starting a process each time
  Sampler         a thread that reads a snapshot every interval and hands
                  it to its listeners (the governor, the event stream)

Unknown values are None, never a guess: a reader that cannot see a
signal leaves it out and the governor ignores it.

Linux has no portable way to see user input under Wayland, so it leaves
input_idle_seconds unknown and the governor uses other programs' CPU use
instead (spec: Linux may rely on memory pressure and battery).

Readers take their sources as arguments (a /proc and /sys root, a clock,
a pid) so tests feed them synthetic machines.
"""

import os
import shutil
import subprocess
import sys
import threading
import time

FIELDS = ( 'time', 'memory_pressure', 'memory_available_mb', 'memory_total_mb', 'on_battery',
           'battery_percent', 'cpu_percent', 'other_cpu_percent', 'own_cpu_percent',
           'input_idle_seconds', 'gpu_percent', 'vram_used_mb', 'vram_total_mb', 'own_rss_mb',
           'cache_free_bytes', 'outputs_free_bytes' )


def signals( **values ):
    """A snapshot with every field present, unknown ones None."""
    snapshot = { name: None for name in FIELDS }
    unknown = set( values ) - set( FIELDS )
    if unknown:
        raise KeyError( 'unknown signal(s): %s'%( ', '.join( sorted( unknown ) ), ) )
    snapshot.update( values )
    return snapshot


def _read( path ):
    try:
        with open( path, 'r', encoding='ascii', errors='replace' ) as handle:
            return handle.read()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# CPU accounting shared by the readers
# ---------------------------------------------------------------------------

class CpuMeter:
    """
    Percent of all cores busy, split into this engine's process tree and
    everything else, from cumulative counters: ( busy, total ) for the
    machine and own busy time, all in the same unit.
    """

    def __init__( self ):
        self._last = None

    def update( self, busy, total, own ):
        last, self._last = self._last, ( busy, total, own )
        if last is None or busy is None or total is None:
            return None, None, None
        d_total = total - last[1]
        if d_total <= 0:
            return None, None, None
        cpu = max( 0.0, min( 100.0, 100.0 * ( busy - last[0] ) / d_total ) )
        if own is None or last[2] is None:
            return round( cpu, 1 ), None, None
        own_pct = max( 0.0, min( cpu, 100.0 * ( own - last[2] ) / d_total ) )
        return round( cpu, 1 ), round( max( 0.0, cpu - own_pct ), 1 ), round( own_pct, 1 )


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------

def psi_some_avg10( proc_root='/proc' ):
    """/proc/pressure/memory 'some avg10': percent of the last 10 s some task waited for memory."""
    text = _read( os.path.join( proc_root, 'pressure', 'memory' ) )
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith( 'some ' ):
            for part in line.split()[1:]:
                key, _, value = part.partition( '=' )
                if key == 'avg10':
                    try:
                        return float( value )
                    except ValueError:
                        return None
    return None


def power_linux( sys_root='/sys' ):
    """
    ( on_battery, battery_percent ). A mains supply's 'online' decides when
    there is one: a battery held at a charge limit reports Discharging
    while plugged in. Without a mains entry, the battery's own status.
    ( False, None ) on a machine with no battery.
    """
    base = os.path.join( sys_root, 'class', 'power_supply' )
    try:
        names = sorted( os.listdir( base ) )
    except OSError:
        return False, None
    mains_online = []
    discharging = False
    percents = []
    for name in names:
        kind = ( _read( os.path.join( base, name, 'type' ) ) or '' ).strip()
        if kind == 'Mains':
            mains_online.append( ( _read( os.path.join( base, name, 'online' ) ) or '' ).strip() == '1' )
        elif kind == 'Battery':
            if ( _read( os.path.join( base, name, 'present' ) ) or '1' ).strip() == '0':
                continue
            status = ( _read( os.path.join( base, name, 'status' ) ) or '' ).strip()
            discharging = discharging or status == 'Discharging'
            try:
                percents.append( int( ( _read( os.path.join( base, name, 'capacity' ) ) or '' ).strip() ) )
            except ValueError:
                pass
    if not percents and not discharging:
        return False, None
    on_battery = ( not any( mains_online ) ) if mains_online else discharging
    return on_battery, ( min( percents ) if percents else None )


def _proc_stat_cpu( proc_root ):
    """( busy, total ) jiffies from /proc/stat's cpu line."""
    text = _read( os.path.join( proc_root, 'stat' ) )
    if not text:
        return None, None
    for line in text.splitlines():
        if line.startswith( 'cpu ' ):
            values = [ int( v ) for v in line.split()[1:] ]
            idle = values[3] + ( values[4] if len( values ) > 4 else 0 )   # idle + iowait
            total = sum( values[:8] )   # guest time is already counted in user
            return total - idle, total
    return None, None


def _pid_stat( proc_root, pid ):
    """( ppid, utime + stime, cutime + cstime (reaped children), rss pages ) of a pid, or None."""
    text = _read( os.path.join( proc_root, str( pid ), 'stat' ) )
    if not text:
        return None
    # The command name is in parentheses and may hold spaces.
    rest = text.rsplit( ')', 1 )[-1].split()
    try:
        return int( rest[1] ), int( rest[11] ) + int( rest[12] ), int( rest[13] ) + int( rest[14] ), int( rest[21] )
    except ( IndexError, ValueError ):
        return None


def _children_files( proc_root, pid ):
    """pid's children from /proc/<pid>/task/*/children, or None when the kernel lacks them."""
    task_dir = os.path.join( proc_root, str( pid ), 'task' )
    try:
        tids = os.listdir( task_dir )
    except OSError:
        return None
    children = []
    for tid in tids:
        text = _read( os.path.join( task_dir, tid, 'children' ) )
        if text is None:
            return None
        children.extend( int( part ) for part in text.split() if part.isdigit() )
    return children


def _process_tree_walk( proc_root, root_pid ):
    """process_tree through the children files: only our own processes are read."""
    root = _pid_stat( proc_root, root_pid )
    if root is None:
        return None
    cpu, rss = root[2], 0
    stack, seen = [ root_pid ], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add( pid )
        found = root if pid == root_pid else _pid_stat( proc_root, pid )
        if found is None:
            continue
        cpu += found[1]
        rss += found[3]
        children = _children_files( proc_root, pid )
        if children is None:
            return None
        stack.extend( children )
    return cpu, rss


def process_tree( proc_root, root_pid ):
    """( cpu jiffies, rss pages ) of root_pid and every live descendant (the render's ffmpeg)."""
    walked = _process_tree_walk( proc_root, root_pid )
    if walked is not None:
        return walked
    try:
        pids = [ int( name ) for name in os.listdir( proc_root ) if name.isdigit() ]
    except OSError:
        return None, None
    stats = {}
    for pid in pids:
        found = _pid_stat( proc_root, pid )
        if found is not None:
            stats[pid] = found
    if root_pid not in stats:
        return None, None
    children = {}
    for pid, found in stats.items():
        children.setdefault( found[0], [] ).append( pid )
    # Children that ended and were waited for are in the root's own
    # cutime, so a finished render chunk keeps counting.
    cpu = stats[root_pid][2]
    rss = 0
    stack = [ root_pid ]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add( pid )
        _ppid, pid_cpu, _reaped, pid_rss = stats[pid]
        cpu += pid_cpu
        rss += pid_rss
        stack.extend( children.get( pid, () ) )
    return cpu, rss


class LinuxReader:

    def __init__( self, proc_root='/proc', sys_root='/sys', pid=None, page_size=None ):
        self.proc_root = proc_root
        self.sys_root = sys_root
        self.pid = pid or os.getpid()
        self.page_size = page_size or ( os.sysconf( 'SC_PAGE_SIZE' ) if hasattr( os, 'sysconf' ) else 4096 )
        self.cpu = CpuMeter()

    def read( self ):
        import betautils_hardware as bu_hardware
        memory = bu_hardware.meminfo_mb( self.proc_root )
        on_battery, percent = power_linux( self.sys_root )
        busy, total = _proc_stat_cpu( self.proc_root )
        own_cpu, own_rss = process_tree( self.proc_root, self.pid )
        cpu, other, own = self.cpu.update( busy, total, own_cpu )
        return {
            'memory_pressure': psi_some_avg10( self.proc_root ),
            'memory_available_mb': memory.get( 'MemAvailable' ),
            'memory_total_mb': memory.get( 'MemTotal' ),
            'on_battery': on_battery,
            'battery_percent': percent,
            'cpu_percent': cpu,
            'other_cpu_percent': other,
            'own_cpu_percent': own,
            'own_rss_mb': ( own_rss * self.page_size // 2**20 ) if own_rss is not None else None,
        }


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

class WindowsReader:
    """
    GlobalMemoryStatusEx, GetSystemPowerStatus, GetSystemTimes and
    GetLastInputInfo. This process's CPU and memory come from its own
    counters plus its live children's (the render's ffmpeg), found with
    a Toolhelp snapshot. Windows has no memory pressure stall figure, so
    memory_pressure is the memory load percent only when commit is near
    its limit; the governor also watches free memory.
    """

    def __init__( self ):
        self.cpu = CpuMeter()

    def read( self ):
        import betautils_hardware as bu_hardware
        memory = bu_hardware.windows_memory_status()
        power = bu_hardware.windows_power_status()
        busy, total = _windows_system_times()
        own_cpu, own_rss = _windows_tree( os.getpid() )
        cpu, other, own = self.cpu.update( busy, total, own_cpu )
        return {
            'memory_available_mb': memory.get( 'MemAvailable' ),
            'memory_total_mb': memory.get( 'MemTotal' ),
            'on_battery': power.get( 'on_battery' ),
            'battery_percent': power.get( 'percent' ),
            'cpu_percent': cpu,
            'other_cpu_percent': other,
            'own_cpu_percent': own,
            'input_idle_seconds': _windows_input_idle(),
            'own_rss_mb': ( own_rss // 2**20 ) if own_rss is not None else None,
        }


def _filetime( value ):
    return ( value.dwHighDateTime << 32 ) | value.dwLowDateTime


def _windows_system_times():
    """( busy, total ) in 100 ns units, adjusted for all cores (kernel time includes idle)."""
    import ctypes
    from ctypes import wintypes
    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes( ctypes.byref( idle ), ctypes.byref( kernel ),
                                                  ctypes.byref( user ) ):
        return None, None
    total = _filetime( kernel ) + _filetime( user )
    return total - _filetime( idle ), total


def _windows_tree( root_pid ):
    """( cpu in 100 ns units, working set bytes ) of root_pid and its live descendants."""
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32( ctypes.Structure ):
        _fields_ = [ ( 'dwSize', wintypes.DWORD ), ( 'cntUsage', wintypes.DWORD ),
                     ( 'th32ProcessID', wintypes.DWORD ), ( 'th32DefaultHeapID', ctypes.c_void_p ),
                     ( 'th32ModuleID', wintypes.DWORD ), ( 'cntThreads', wintypes.DWORD ),
                     ( 'th32ParentProcessID', wintypes.DWORD ), ( 'pcPriClassBase', ctypes.c_long ),
                     ( 'dwFlags', wintypes.DWORD ), ( 'szExeFile', ctypes.c_char * 260 ) ]

    class PROCESS_MEMORY_COUNTERS( ctypes.Structure ):
        _fields_ = [ ( 'cb', wintypes.DWORD ), ( 'PageFaultCount', wintypes.DWORD ),
                     ( 'PeakWorkingSetSize', ctypes.c_size_t ), ( 'WorkingSetSize', ctypes.c_size_t ),
                     ( 'QuotaPeakPagedPoolUsage', ctypes.c_size_t ), ( 'QuotaPagedPoolUsage', ctypes.c_size_t ),
                     ( 'QuotaPeakNonPagedPoolUsage', ctypes.c_size_t ),
                     ( 'QuotaNonPagedPoolUsage', ctypes.c_size_t ),
                     ( 'PagefileUsage', ctypes.c_size_t ), ( 'PeakPagefileUsage', ctypes.c_size_t ) ]

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE
    snapshot = kernel32.CreateToolhelp32Snapshot( 0x2, 0 )   # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == wintypes.HANDLE( -1 ).value:
        return None, None
    parents = {}
    try:
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof( PROCESSENTRY32 )
        ok = kernel32.Process32First( snapshot, ctypes.byref( entry ) )
        while ok:
            parents.setdefault( entry.th32ParentProcessID, [] ).append( entry.th32ProcessID )
            ok = kernel32.Process32Next( snapshot, ctypes.byref( entry ) )
    finally:
        kernel32.CloseHandle( snapshot )
    cpu = rss = 0
    stack, seen = [ root_pid ], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add( pid )
        stack.extend( parents.get( pid, () ) )
        handle = kernel32.OpenProcess( 0x1000 | 0x0010, False, pid )   # QUERY_LIMITED_INFORMATION | VM_READ
        if not handle:
            continue
        try:
            created, exited, kernel, user = ( wintypes.FILETIME() for _ in range( 4 ) )
            if kernel32.GetProcessTimes( handle, ctypes.byref( created ), ctypes.byref( exited ),
                                         ctypes.byref( kernel ), ctypes.byref( user ) ):
                cpu += _filetime( kernel ) + _filetime( user )
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof( PROCESS_MEMORY_COUNTERS )
            if ctypes.windll.psapi.GetProcessMemoryInfo( handle, ctypes.byref( counters ), counters.cb ):
                rss += counters.WorkingSetSize
        finally:
            kernel32.CloseHandle( handle )
    return cpu, rss


def _windows_input_idle():
    import ctypes
    from ctypes import wintypes

    class LASTINPUTINFO( ctypes.Structure ):
        _fields_ = [ ( 'cbSize', wintypes.UINT ), ( 'dwTime', wintypes.DWORD ) ]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof( LASTINPUTINFO )
    if not ctypes.windll.user32.GetLastInputInfo( ctypes.byref( info ) ):
        return None
    now = ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF
    return round( ( ( now - info.dwTime ) & 0xFFFFFFFF ) / 1000.0, 1 )


class NullReader:

    def read( self ):
        return {}


def platform_reader( platform_name=None ):
    platform_name = platform_name or sys.platform
    if platform_name.startswith( 'linux' ):
        return LinuxReader()
    if platform_name == 'win32':
        return WindowsReader()
    return NullReader()


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

GPU_QUERY = ( 'utilization.gpu', 'memory.used', 'memory.total' )


def parse_gpu_line( line ):
    """{ gpu_percent, vram_used_mb, vram_total_mb } from one nvidia-smi CSV line, or {}."""
    parts = [ part.strip() for part in line.split( ',' ) ]
    if len( parts ) < len( GPU_QUERY ):
        return {}
    found = {}
    for name, value in zip( ( 'gpu_percent', 'vram_used_mb', 'vram_total_mb' ), parts ):
        try:
            found[name] = int( float( value ) )
        except ValueError:
            found[name] = None
    return found


class GpuStream:
    """
    One nvidia-smi printing the first GPU's use every interval. latest()
    is the last line read, {} before the first or after nvidia-smi ends;
    a value older than a few intervals counts as unknown.
    """

    def __init__( self, interval_ms=1000, popen=subprocess.Popen, clock=time.monotonic ):
        self.interval = interval_ms / 1000.0
        self._popen = popen
        self._clock = clock
        self._proc = None
        self._thread = None
        self._latest = {}
        self._at = None
        self._lock = threading.Lock()

    def start( self ):
        if shutil.which( 'nvidia-smi' ) is None and self._popen is subprocess.Popen:
            return False
        try:
            self._proc = self._popen(
                [ 'nvidia-smi', '--query-gpu=%s'%( ','.join( GPU_QUERY ), ),
                  '--format=csv,noheader,nounits', '-lms', str( int( self.interval * 1000 ) ) ],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1 )
        except OSError:
            self._proc = None
            return False
        self._thread = threading.Thread( target=self._pump, name='betagpu', daemon=True )
        self._thread.start()
        return True

    def _pump( self ):
        for line in self._proc.stdout:
            found = parse_gpu_line( line )
            if found:
                with self._lock:
                    self._latest = found
                    self._at = self._clock()

    def latest( self ):
        with self._lock:
            if self._at is None or self._clock() - self._at > 5 * max( self.interval, 0.2 ):
                return {}
            return dict( self._latest )

    def stop( self ):
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait( timeout=2 )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join( timeout=2 )


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

def disk_free( folders, usage=shutil.disk_usage ):
    """{ 'cache_free_bytes': n, 'outputs_free_bytes': n } for the folders that can be read."""
    found = {}
    for kind, folder in folders.items():
        probe = os.path.abspath( folder )
        while not os.path.exists( probe ) and os.path.dirname( probe ) != probe:
            probe = os.path.dirname( probe )
        try:
            found['%s_free_bytes'%( kind, )] = int( usage( probe ).free )
        except OSError:
            pass
    return found


def _default_folders():
    try:
        import betautils_paths as bu_paths
        return { 'cache': bu_paths.kind_dir( 'cache' ), 'outputs': bu_paths.kind_dir( 'outputs' ) }
    except Exception:
        return {}


class Sampler:
    """
    Reads a snapshot every `interval` seconds on its own thread and calls
    each listener( snapshot ). A listener's error is logged, never fatal.
    sample() takes one snapshot on the caller's thread (tests, and the
    first answer before the thread has run).
    """

    def __init__( self, reader=None, gpu=None, interval=1.0, clock=time.time, folders=None,
                  usage=shutil.disk_usage, logger=None, disk_every=5 ):
        self.reader = reader if reader is not None else platform_reader()
        self.gpu = gpu
        self.interval = interval
        self.clock = clock
        self.folders = _default_folders() if folders is None else folders
        self.usage = usage
        self.logger = logger
        self.disk_every = max( 1, int( disk_every ) )
        self.listeners = []
        self.latest = None
        self._ticks = 0
        self._disk = {}
        self._stop = threading.Event()
        self._thread = None

    def add_listener( self, listener ):
        self.listeners.append( listener )

    def remove_listener( self, listener ):
        if listener in self.listeners:
            self.listeners.remove( listener )

    def sample( self ):
        values = {}
        try:
            values.update( self.reader.read() or {} )
        except Exception:
            if self.logger is not None:
                self.logger.debug( "resource reader failed", exc_info=True )
        if self.gpu is not None:
            values.update( self.gpu.latest() )
        if self._ticks % self.disk_every == 0:
            self._disk = disk_free( self.folders, self.usage )
        self._ticks += 1
        values.update( self._disk )
        values['time'] = self.clock()
        snapshot = signals( **{ k: v for k, v in values.items() if k in FIELDS } )
        self.latest = snapshot
        for listener in list( self.listeners ):
            try:
                listener( snapshot )
            except Exception:
                if self.logger is not None:
                    self.logger.warning( "resource listener failed", exc_info=True )
        return snapshot

    def start( self ):
        if self._thread is not None:
            return
        if self.gpu is not None:
            self.gpu.start()
        self._thread = threading.Thread( target=self._run, name='betasampler', daemon=True )
        self._thread.start()

    def _run( self ):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait( self.interval )

    def stop( self ):
        self._stop.set()
        if self._thread is not None:
            self._thread.join( timeout=5 )
            self._thread = None
        if self.gpu is not None:
            self.gpu.stop()
