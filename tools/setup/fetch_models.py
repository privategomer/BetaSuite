"""
fetch_models.py - put BetaSuite's detector models in place and validate them.

Run from anywhere; models live in ../resources/model relative to the
repository root.

  320n       downloaded automatically from the nudenet package on PyPI
             (pinned by hash)
  640m       manual: GitHub only serves NudeNet's release files to
  RetinaNet  signed-in users, so download them yourself (links below and
             in SETUP.md) and save them under the names shown
  yolo11*-seg  person masks for reverse mode (n, s, m): Ultralytics' .pt
             releases (pinned by hash) exported to ONNX in a throwaway
             environment (ultralytics + CPU-only PyTorch) that is deleted
             afterwards, so the runtime .venv never gets PyTorch

Anything already present is checked. A file that isn't a real ONNX model
(for example a GitHub sign-in page saved under a .onnx name) is moved
aside to <name>.bad instead of being trusted.

    python tools/setup/fetch_models.py           # fetch/export what is missing, check all
    python tools/setup/fetch_models.py --no-download   # check only
    python tools/setup/fetch_models.py --no-export     # skip the person-model export

Exit status is 1 if 320n is missing, a person-model export failed, or any
model present is unusable.
"""

import argparse
import hashlib
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', '..' ) )
MODEL_DIR = os.path.normpath( os.path.join( REPO_ROOT, '..', 'resources', 'model' ) )


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

NUDENET_RELEASES = 'https://github.com/notAI-tech/NudeNet/releases'
ULTRALYTICS_ASSETS = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/'

# The throwaway export environment. CPU-only PyTorch: export needs no GPU,
# and the CPU wheels are a fraction of the CUDA ones. The environment
# variable points it at a mirror.
EXPORT_TORCH_INDEX = os.environ.get( 'BETASUITE_EXPORT_TORCH_INDEX', 'https://download.pytorch.org/whl/cpu' )
EXPORT_PACKAGES = [ 'ultralytics==8.4.198', 'onnx', 'onnxslim' ]
EXPORT_SCRIPT = (
    "import shutil, sys\n"
    "from ultralytics import YOLO\n"
    "pt, out, size = sys.argv[1], sys.argv[2], int( sys.argv[3] )\n"
    "exported = YOLO( pt ).export( format='onnx', imgsz=size, dynamic=True, simplify=True )\n"
    "shutil.move( exported, out )\n" )

MODELS = {
    '320n': {
        'file': 'v3.4-320n.onnx',
        'desc': 'NudeNet v3.4 320n (default backend)',
        'wheel_url': 'https://files.pythonhosted.org/packages/1c/ee/1aa02d44ba958cc77e16ff1e41a0aac5e721037db7bf62b9c9d124917f87/nudenet-3.4.2-py3-none-any.whl',
        'wheel_sha256': '5937dbd84e5d8e5de038f08ffea5a1bb50a08475776bf2b4795914ce0eaf0331',
        'wheel_member': 'nudenet/320n.onnx',
        'sha256': 'c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f',
    },
    '640m': {
        'file': 'v3.4-640m.onnx',
        'desc': 'NudeNet v3.4 640m (optional, slower, finds more)',
        'manual_url': NUDENET_RELEASES + '/download/v3.4-weights/640m.onnx',
    },
    'retinanet': {
        'file': 'detector_v2_default_checkpoint.onnx',
        'desc': 'NudeNet v2 RetinaNet (optional, retinanet_v2 backend)',
        'manual_url': NUDENET_RELEASES + '/download/v0/detector_v2_default_checkpoint.onnx',
    },
    'yolo11n-seg': {
        'file': 'yolo11n-seg.onnx',
        'desc': 'YOLO11n-seg person masks (reverse mode person windows)',
        'export': { 'pt_url': ULTRALYTICS_ASSETS + 'yolo11n-seg.pt', 'imgsz': 640,
                    'pt_sha256': '55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152' },
    },
    'yolo11s-seg': {
        'file': 'yolo11s-seg.onnx',
        'desc': 'YOLO11s-seg person masks (reverse mode person windows, default)',
        'export': { 'pt_url': ULTRALYTICS_ASSETS + 'yolo11s-seg.pt', 'imgsz': 640,
                    'pt_sha256': '1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3' },
    },
    'yolo11m-seg': {
        'file': 'yolo11m-seg.onnx',
        'desc': 'YOLO11m-seg person masks (reverse mode person windows)',
        'export': { 'pt_url': ULTRALYTICS_ASSETS + 'yolo11m-seg.pt', 'imgsz': 640,
                    'pt_sha256': 'eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf' },
    },
}


class NotAModel( Exception ):
    pass


def build_logger( log_file, stdout_level ):
    log = logging.getLogger( 'fetch_models' )
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


def onnx_problem( data_or_path ):
    """None if this looks like an ONNX model, otherwise why not."""
    if isinstance( data_or_path, bytes ):
        head, size = data_or_path[:512], len( data_or_path )
    else:
        size = os.path.getsize( data_or_path )
        with open( data_or_path, 'rb' ) as f:
            head = f.read( 512 )
    stripped = head.lstrip().lower()
    if stripped.startswith( b'<' ) or b'<html' in stripped:
        return 'it is an HTML web page, not a model (usually a GitHub sign-in page from a logged-out download)'
    if size < 1_000_000:
        return 'it is only %d bytes'%(size)
    # An ONNX ModelProto starts with field 1 (ir_version), a varint: tag byte 0x08.
    if head[:1] != b'\x08':
        return 'it does not start like an ONNX file'
    return None


def sha256_of( data ):
    return hashlib.sha256( data ).hexdigest()


def fetch_320n( log ):
    spec = MODELS['320n']
    log.debug( 'GET %s'%( spec['wheel_url'] ) )
    req = urllib.request.Request( spec['wheel_url'], headers={ 'User-Agent': 'BetaSuite-setup' } )
    with urllib.request.urlopen( req, timeout=120 ) as resp:
        wheel = resp.read()
    if sha256_of( wheel ) != spec['wheel_sha256']:
        raise NotAModel( 'nudenet wheel checksum mismatch' )
    data = zipfile.ZipFile( io.BytesIO( wheel ) ).read( spec['wheel_member'] )
    if sha256_of( data ) != spec['sha256']:
        raise NotAModel( 'model checksum mismatch' )
    return data


def download_checked( url, sha256, dest, log ):
    """Download url to dest, refusing it unless its sha256 matches."""
    log.debug( 'GET %s'%( url ) )
    req = urllib.request.Request( url, headers={ 'User-Agent': 'BetaSuite-setup' } )
    with urllib.request.urlopen( req, timeout=300 ) as resp:
        data = resp.read()
    if sha256_of( data ) != sha256:
        raise NotAModel( '%s checksum mismatch'%( os.path.basename( dest ) ) )
    with open( dest, 'wb' ) as f:
        f.write( data )


def find_uv( explicit=None ):
    """uv from --uv, PATH or its usual install folders, else None."""
    candidates = [ explicit ] if explicit else []
    candidates += [ shutil.which( 'uv' ),
                    os.path.join( os.path.expanduser( '~' ), '.local', 'bin', 'uv' ),
                    os.path.join( os.path.expanduser( '~' ), '.cargo', 'bin', 'uv' ),
                    os.path.join( os.path.expanduser( '~' ), '.local', 'bin', 'uv.exe' ),
                    os.path.join( os.path.expanduser( '~' ), '.cargo', 'bin', 'uv.exe' ) ]
    for candidate in candidates:
        if candidate and os.path.isfile( candidate ):
            return candidate
    return None


def run_logged( description, command, log ):
    """Run a command, its output to the log at debug; raise on failure."""
    log.info( '=== starting: %s ==='%( description ) )
    log.debug( 'command: %s'%( ' '.join( command ) ) )
    result = subprocess.run( command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors='replace' )
    for line in result.stdout.splitlines():
        log.debug( '  %s'%( line ) )
    if result.returncode != 0:
        for line in result.stdout.splitlines()[-15:]:
            log.error( '  %s'%( line ) )
        raise RuntimeError( '%s failed (exit %d)'%( description, result.returncode ) )
    log.info( '=== finished: %s ==='%( description ) )


def build_export_env( workdir, uv, log ):
    """A throwaway venv with ultralytics and CPU-only PyTorch. Returns its python."""
    venv = os.path.join( workdir, 'export-venv' )
    version = '%d.%d'%( sys.version_info[0], sys.version_info[1] )
    if uv:
        run_logged( 'create export environment', [ uv, 'venv', '--python', version, venv ], log )
    else:
        run_logged( 'create export environment', [ sys.executable, '-m', 'venv', venv ], log )
    python = os.path.join( venv, 'Scripts', 'python.exe' ) if os.name == 'nt' else os.path.join( venv, 'bin', 'python' )
    if uv:
        install = [ uv, 'pip', 'install', '--python', python ]
    else:
        install = [ python, '-m', 'pip', 'install', '--disable-pip-version-check' ]
    run_logged( 'install CPU-only PyTorch (export only)',
                install + [ '--index-url', EXPORT_TORCH_INDEX, 'torch', 'torchvision' ], log )
    run_logged( 'install ultralytics (export only)', install + EXPORT_PACKAGES, log )
    return python


def export_person_models( missing, log, uv=None ):
    """
    Export each missing person model to ONNX. One throwaway environment
    for all of them, removed whatever happens.

    Args:
        missing: [(spec, dest), ...] for specs with an 'export' entry.

    Returns:
        The number of models that failed.
    """
    workdir = tempfile.mkdtemp( prefix='betasuite-export-' )
    failures = 0
    try:
        try:
            python = build_export_env( workdir, uv, log )
        except Exception as e:
            log.error( 'could not build the export environment: %s'%( e ) )
            return len( missing )
        env = dict( os.environ, YOLO_OFFLINE='1' )
        for spec, dest in missing:
            export = spec['export']
            pt = os.path.join( workdir, os.path.basename( export['pt_url'] ) )
            out = os.path.join( workdir, spec['file'] )
            try:
                download_checked( export['pt_url'], export['pt_sha256'], pt, log )
                log.info( '=== starting: export %s ==='%( spec['file'] ) )
                result = subprocess.run( [ python, '-c', EXPORT_SCRIPT, pt, out, str( export['imgsz'] ) ],
                                         cwd=workdir, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, errors='replace' )
                for line in result.stdout.splitlines():
                    log.debug( '  %s'%( line ) )
                if result.returncode != 0:
                    for line in result.stdout.splitlines()[-15:]:
                        log.error( '  %s'%( line ) )
                    raise RuntimeError( 'export exited %d'%( result.returncode ) )
                problem = onnx_problem( out )
                if problem:
                    raise NotAModel( problem )
                shutil.move( out, dest )
                log.info( '=== finished: export %s (%.1f MB) ==='%(
                    spec['file'], os.path.getsize( dest ) / 1e6 ) )
            except Exception as e:
                failures += 1
                log.error( 'could not export %s: %s'%( spec['file'], e ) )
    finally:
        shutil.rmtree( workdir, ignore_errors=True )
        log.info( 'removed the export environment' )
    return failures


def main():
    relaunch_in_venv()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--no-download', action='store_true', help='only check what is already there' )
    parser.add_argument( '--no-export', action='store_true',
                         help='skip exporting the person models (reverse mode person windows)' )
    parser.add_argument( '--uv', help='path to uv, for the throwaway export environment' )
    parser.add_argument( '--log-level', default='info', choices=list( LEVELS ) )
    parser.add_argument( '--log-file', default=os.path.join( REPO_ROOT, 'fetch_models.log' ) )
    args = parser.parse_args()

    log = build_logger( os.path.abspath( args.log_file ), args.log_level )
    os.makedirs( MODEL_DIR, exist_ok=True )
    failures = 0
    manual_missing = []
    export_missing = []

    for key, spec in MODELS.items():
        dest = os.path.join( MODEL_DIR, spec['file'] )
        if os.path.exists( dest ):
            problem = onnx_problem( dest )
            if not problem:
                log.info( 'model present: %s'%(dest) )
                continue
            shutil.move( dest, dest + '.bad' )
            log.warning( '%s is not usable: %s. Moved it to %s.bad'%( spec['file'], problem, dest ) )

        if 'manual_url' in spec:
            manual_missing.append( ( spec, dest ) )
            continue
        if 'export' in spec:
            if args.no_download or args.no_export:
                log.warning( '%s is missing (person windows need it; run without --no-export '
                             'to create it)'%( spec['file'] ) )
            else:
                export_missing.append( ( spec, dest ) )
            continue
        if args.no_download:
            failures += 1
            log.error( '%s is missing'%( spec['file'] ) )
            continue

        log.info( '=== starting: fetch %s from PyPI ==='%( spec['file'] ) )
        try:
            data = fetch_320n( log )
        except Exception as e:
            failures += 1
            log.error( 'could not fetch %s: %s'%( spec['file'], e ) )
            continue
        tmp = dest + '.part'
        with open( tmp, 'wb' ) as f:
            f.write( data )
        os.replace( tmp, dest )
        log.info( '=== finished: fetch %s (%.1f MB) ==='%( spec['file'], len( data ) / 1e6 ) )

    if export_missing:
        log.info( 'exporting %d person model(s): %s'%(
            len( export_missing ), ', '.join( spec['file'] for spec, _dest in export_missing ) ) )
        failures += export_person_models( export_missing, log, find_uv( args.uv ) )

    if manual_missing:
        log.info( 'optional models not installed (download them yourself, signed in to GitHub):' )
        for spec, dest in manual_missing:
            log.info( '  %s' % ( spec['desc'] ) )
            log.info( '    download  %s'%( spec['manual_url'] ) )
            log.info( '    save as   %s'%(dest) )
        log.info( 'then re-run this script (or setup) to check them. See SETUP.md, Models.' )

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit( main() )
