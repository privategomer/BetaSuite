"""
verify_env.py - check that a BetaSuite environment is ready to run.

Run from anywhere; it changes to the repository root first so the
'../resources' and '../output' paths resolve the way the app sees them.

    python tools/setup/verify_env.py            # uses betaconfig.gpu_enabled
    python tools/setup/verify_env.py --cpu      # check the CPU path only
    python tools/setup/verify_env.py --gpu      # require CUDA to be active

Exit status is 0 when nothing is at ERROR level, 1 otherwise. Every
level goes to the log file; --log-level filters stdout only.
"""

import argparse
import importlib
import importlib.metadata
import logging
import os
import platform
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', '..' ) )

def relaunch_in_venv():
    """Re-run this script with the repo's .venv Python if started outside it."""
    if sys.prefix != sys.base_prefix or os.environ.get( 'BETASUITE_NO_RELAUNCH' ):
        return
    for rel in ( ( '.venv', 'bin', 'python' ), ( '.venv', 'Scripts', 'python.exe' ) ):
        venv_python = os.path.join( REPO_ROOT, *rel )
        if os.path.exists( venv_python ):
            print( 'not in the virtual environment; re-running with %s'%(venv_python), flush=True )
            os.environ['BETASUITE_NO_RELAUNCH'] = '1'
            args = [ venv_python, os.path.abspath( __file__ ) ] + sys.argv[1:]
            if os.name == 'nt':
                import subprocess
                sys.exit( subprocess.call( args ) )
            os.execv( venv_python, args )


TRACE = 5
logging.addLevelName( TRACE, 'TRACE' )
LEVELS = { 'trace': TRACE, 'debug': logging.DEBUG, 'info': logging.INFO,
           'warn': logging.WARNING, 'error': logging.ERROR }

# Model files the detector adapters look for. Only the selected backend's
# model is required to run; the rest are optional.
MODELS = {
    'nudenet_v3 / 320n': '../resources/model/v3.4-320n.onnx',
    'nudenet_v3 / 640m': '../resources/model/v3.4-640m.onnx',
    'retinanet_v2':      '../resources/model/detector_v2_default_checkpoint.onnx',
}

DIRS = [
    '../resources/model',
    '../resources/uncensored_vids',
    '../resources/uncensored_pics',
    '../resources/stickers/breasts',
    '../resources/stickers/vulva',
    '../output',
]


class Checker:
    def __init__( self, log ):
        self.log = log
        self.errors = 0
        self.warnings = 0

    def ok( self, msg ):
        self.log.info( 'OK    %s'%(msg) )

    def warn( self, msg ):
        self.warnings += 1
        self.log.warning( 'WARN  %s'%(msg) )

    def fail( self, msg ):
        self.errors += 1
        self.log.error( 'FAIL  %s'%(msg) )


def build_logger( log_file, stdout_level ):
    log = logging.getLogger( 'verify_env' )
    log.setLevel( TRACE )
    fmt = logging.Formatter( '[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S' )

    fh = logging.FileHandler( log_file, encoding='utf-8' )
    fh.setLevel( TRACE )
    fh.setFormatter( fmt )
    log.addHandler( fh )

    sh = logging.StreamHandler( sys.stdout )
    sh.setLevel( LEVELS[stdout_level] )
    sh.setFormatter( fmt )
    log.addHandler( sh )
    return log


def installed_version( dist ):
    try:
        return importlib.metadata.version( dist )
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python( c ):
    want = None
    pin_file = os.path.join( REPO_ROOT, '.python-version' )
    if os.path.exists( pin_file ):
        with open( pin_file ) as f:
            want = f.read().strip()
    have = '%d.%d'%( sys.version_info[0], sys.version_info[1] )
    c.log.debug( 'interpreter: %s'%(sys.executable) )
    if sys.prefix == sys.base_prefix:
        c.warn( 'not running inside a virtual environment (%s)'%(sys.executable) )
    if want and not have.startswith( want ) and not want.startswith( have ):
        c.warn( 'python %s, but .python-version pins %s'%( platform.python_version(), want ) )
    else:
        c.ok( 'python %s'%( platform.python_version() ) )


def check_packages( c ):
    for module, dist in ( ( 'numpy', 'numpy' ), ( 'cv2', 'opencv-python' ) ):
        try:
            importlib.import_module( module )
            c.ok( '%s %s'%( dist, installed_version( dist ) or '(version unknown)' ) )
        except Exception as e:
            c.fail( 'cannot import %s (%s): %s'%( module, dist, e ) )

    cpu_ver = installed_version( 'onnxruntime' )
    gpu_ver = installed_version( 'onnxruntime-gpu' )
    if cpu_ver and gpu_ver:
        c.fail( 'both onnxruntime %s and onnxruntime-gpu %s are installed; they overwrite each '
                'other. Uninstall both, then install only one.'%( cpu_ver, gpu_ver ) )
    elif not cpu_ver and not gpu_ver:
        c.fail( 'neither onnxruntime nor onnxruntime-gpu is installed' )
    else:
        c.ok( 'onnxruntime%s %s'%( '-gpu' if gpu_ver else '', gpu_ver or cpu_ver ) )

    if platform.system() == 'Windows':
        for module in ( 'mss', 'win32gui' ):
            try:
                importlib.import_module( module )
                c.ok( '%s (betavision)'%(module) )
            except Exception:
                c.log.info( 'skip  %s not installed; only needed for betavision-*'%(module) )


def check_ffmpeg( c ):
    for tool in ( 'ffmpeg', 'ffprobe' ):
        path = shutil.which( tool )
        if not path:
            c.fail( '%s not found on PATH'%(tool) )
            continue
        try:
            first = subprocess.run( [ tool, '-version' ], capture_output=True, text=True,
                                    timeout=20 ).stdout.splitlines()[0]
        except Exception as e:
            first = 'version check failed: %s'%(e)
        c.ok( '%s: %s'%( tool, first ) )
        c.log.debug( '%s path: %s'%( tool, path ) )


def check_dirs( c ):
    for d in DIRS:
        if os.path.isdir( d ):
            c.log.debug( 'dir present: %s'%( os.path.abspath( d ) ) )
        else:
            c.warn( 'missing directory %s (setup creates it)'%(d) )
    for label in ( 'breasts', 'vulva' ):
        d = '../resources/stickers/%s'%(label)
        n = len( [ f for f in os.listdir( d ) if f.lower().endswith( '.png' ) ] ) if os.path.isdir( d ) else 0
        if n:
            c.ok( 'stickers/%s: %d png'%( label, n ) )
        else:
            c.log.info( 'note  stickers/%s is empty; sticker styles render as solid bars until '
                        'you add .png files'%(label) )


def gpu_requested( args, c ):
    if args.gpu:
        return True
    if args.cpu:
        return False
    try:
        import betaconfig
        value = bool( getattr( betaconfig, 'gpu_enabled', 0 ) )
        c.log.info( 'betaconfig.gpu_enabled = %d'%( int( value ) ) )
        return value
    except Exception as e:
        c.warn( 'could not import betaconfig.py to read gpu_enabled (%s); assuming CPU'%(e) )
        return False


def check_models( c, want_gpu ):
    try:
        import onnxruntime
    except Exception:
        return  # already reported

    available = onnxruntime.get_available_providers()
    c.log.info( 'onnxruntime available providers: %s'%( ', '.join( available ) ) )
    if want_gpu and 'CUDAExecutionProvider' not in available:
        c.fail( 'GPU requested but this onnxruntime build has no CUDAExecutionProvider '
                '(CPU package installed, or wrong build)' )

    present = { name: path for name, path in MODELS.items() if os.path.exists( path ) }
    for name, path in MODELS.items():
        if name not in present:
            c.log.info( 'skip  model %s not present (%s)'%( name, path ) )
    if not present:
        c.fail( 'no model files in ../resources/model; the default backend needs v3.4-320n.onnx' )
        return

    if want_gpu:
        providers = [ 'CUDAExecutionProvider', 'CPUExecutionProvider' ]
    else:
        providers = [ 'CPUExecutionProvider' ]

    if want_gpu and hasattr( onnxruntime, 'preload_dlls' ):
        # Same as the app (betautils_detector.preload_cuda_libraries): load the
        # CUDA/cuDNN libraries from the nvidia-* wheels before any session.
        try:
            onnxruntime.preload_dlls()
        except Exception as e:
            c.warn( 'onnxruntime.preload_dlls() failed: %s'%(e) )

    for name, path in present.items():
        try:
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            session = onnxruntime.InferenceSession( path, sess_options=options, providers=providers )
            active = session.get_providers()
            if want_gpu and 'CUDAExecutionProvider' not in active:
                c.fail( 'model %s loaded on %s, not CUDA'%( name, ', '.join( active ) ) )
            else:
                c.ok( 'model %s loads (%s, %.1f MB)'%( name, active[0],
                                                       os.path.getsize( path ) / 1e6 ) )
        except Exception as e:
            with open( path, 'rb' ) as f:
                head = f.read( 512 ).lstrip().lower()
            if head.startswith( b'<' ):
                c.fail( 'model %s is an HTML page, not a model (a redirected download). Re-run '
                        'setup or tools/setup/fetch_models.py to replace it'%(name) )
            else:
                c.fail( 'model %s failed to load: %s'%( name, e ) )


def main():
    relaunch_in_venv()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    group = parser.add_mutually_exclusive_group()
    group.add_argument( '--gpu', action='store_true', help='require CUDA regardless of betaconfig' )
    group.add_argument( '--cpu', action='store_true', help='check the CPU path regardless of betaconfig' )
    parser.add_argument( '--log-level', default='info', choices=list( LEVELS ),
                         help='stdout level (the log file always gets everything)' )
    parser.add_argument( '--log-file', default=os.path.join( REPO_ROOT, 'verify_env.log' ) )
    args = parser.parse_args()

    log = build_logger( os.path.abspath( args.log_file ), args.log_level )
    os.chdir( REPO_ROOT )
    sys.path.insert( 0, REPO_ROOT )
    c = Checker( log )

    log.info( 'verifying BetaSuite environment in %s (%s %s)'%(
        REPO_ROOT, platform.system(), platform.machine() ) )
    check_python( c )
    check_packages( c )
    check_ffmpeg( c )
    check_dirs( c )
    check_models( c, gpu_requested( args, c ) )

    summary = 'verify finished: %d error(s), %d warning(s)'%( c.errors, c.warnings )
    if c.errors:
        log.error( summary )
    else:
        log.info( summary )
    return 1 if c.errors else 0


if __name__ == '__main__':
    sys.exit( main() )
