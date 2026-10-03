"""
fetch_models.py - put BetaSuite's detector models in place and validate them.

Run from anywhere; models live in ../resources/model relative to the
repository root.

  320n       downloaded automatically from the nudenet package on PyPI
             (pinned by hash)
  640m       manual: GitHub only serves NudeNet's release files to
  RetinaNet  signed-in users, so download them yourself (links below and
             in SETUP.md) and save them under the names shown

Anything already present is checked. A file that isn't a real ONNX model
(for example a GitHub sign-in page saved under a .onnx name) is moved
aside to <name>.bad instead of being trusted.

    python tools/setup/fetch_models.py           # fetch 320n if missing, check all
    python tools/setup/fetch_models.py --no-download   # check only

Exit status is 1 if 320n is missing or any model present is unusable.
"""

import argparse
import hashlib
import io
import logging
import os
import shutil
import sys
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


def main():
    relaunch_in_venv()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--no-download', action='store_true', help='only check what is already there' )
    parser.add_argument( '--log-level', default='info', choices=list( LEVELS ) )
    parser.add_argument( '--log-file', default=os.path.join( REPO_ROOT, 'fetch_models.log' ) )
    args = parser.parse_args()

    log = build_logger( os.path.abspath( args.log_file ), args.log_level )
    os.makedirs( MODEL_DIR, exist_ok=True )
    failures = 0
    manual_missing = []

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
