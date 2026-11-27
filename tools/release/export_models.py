"""
export_models.py - export the person models to ONNX from their pinned
upstream weights (maintainers only).

The app and setup download the project's hosted .onnx files
(assets/models/catalog.json). This makes them, when a model is added or
re-exported: each catalog row with an 'export' entry is exported from its
first source file (Ultralytics' YOLO11-seg .pt, Roboflow's RF-DETR-Seg
.pth), downloaded and checked against its pinned SHA-256, in a throwaway
environment (CPU-only PyTorch plus the exporting package) that is deleted
afterwards. The export runs offline.

An export is not byte-for-byte reproducible across package versions, so
it prints each file's SHA-256 and size: put them in the catalog when the
new files are what gets hosted.

    python3 tools/release/export_models.py                    # asks for the folder
    python3 tools/release/export_models.py --models yolo11s-seg --out /some/folder
    python3 tools/release/export_models.py --models all --out /some/folder --yes

Every level goes to the log file (default export_models.log in the state
folder's logs); --log-level filters the terminal. Exit status 1 if any
export failed.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )
import betautils_models as bu_models  # noqa: E402  (stdlib only)
import betautils_paths as bu_paths  # noqa: E402
import betautils_log as bu_log  # noqa: E402

# CPU-only PyTorch: export needs no GPU, and the CPU wheels are a fraction
# of the CUDA ones. The environment variable points it at a mirror.
EXPORT_TORCH_INDEX = os.environ.get( 'BETASUITE_EXPORT_TORCH_INDEX', 'https://download.pytorch.org/whl/cpu' )
EXPORT_PACKAGES_COMMON = [ 'onnx', 'onnxslim' ]
EXPORT_PACKAGES = { 'ultralytics': 'ultralytics==8.4.198', 'rfdetr': 'rfdetr==1.12.0' }
# One script per exporting package, each run as: script <weights> <out.onnx> <arg>.
EXPORT_SCRIPTS = {
    # arg: the picture size. Dynamic batch and size.
    'ultralytics': (
        "import shutil, sys\n"
        "from ultralytics import YOLO\n"
        "pt, out, size = sys.argv[1], sys.argv[2], int( sys.argv[3] )\n"
        "exported = YOLO( pt ).export( format='onnx', imgsz=size, dynamic=True, simplify=True )\n"
        "shutil.move( exported, out )\n" ),
    # arg: the rfdetr model class. Dynamic batch; the picture size is the
    # class's own and is fixed in the graph (rfdetr exports no other way).
    'rfdetr': (
        "import glob, os, shutil, sys, tempfile\n"
        "import rfdetr\n"
        "pt, out, name = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "folder = tempfile.mkdtemp( dir=os.getcwd() )\n"
        "getattr( rfdetr, name )( pretrain_weights=pt ).export( output_dir=folder, dynamic_batch=True,\n"
        "                                                        verbose=False )\n"
        "exported = glob.glob( os.path.join( folder, '*.onnx' ) )\n"
        "if len( exported ) != 1:\n"
        "    sys.exit( 'expected one .onnx from the export, got %r'%( exported, ) )\n"
        "shutil.move( exported[0], out )\n" ),
}


class NotAModel( Exception ):
    pass


def exportable( catalog=None ):
    """The catalog rows this tool can export."""
    return [ row for row in bu_models.entries( catalog ) if row.get( 'export' ) and row.get( 'sources' ) ]


def onnx_problem( path ):
    """None if this looks like an ONNX model, otherwise why not."""
    size = os.path.getsize( path )
    with open( path, 'rb' ) as handle:
        head = handle.read( 512 )
    stripped = head.lstrip().lower()
    if stripped.startswith( b'<' ) or b'<html' in stripped:
        return 'it is an HTML web page, not a model'
    if size < 1_000_000:
        return 'it is only %d bytes'%( size, )
    # An ONNX ModelProto starts with field 1 (ir_version), a varint: tag byte 0x08.
    if head[:1] != b'\x08':
        return 'it does not start like an ONNX file'
    return None


def sha256_file( path ):
    digest = hashlib.sha256()
    with open( path, 'rb' ) as handle:
        for chunk in iter( lambda: handle.read( 4 * 1024 * 1024 ), b'' ):
            digest.update( chunk )
    return digest.hexdigest()


def download_checked( url, sha256, dest, log ):
    """Stream url to dest, keeping it only when its SHA-256 matches."""
    log.debug( 'GET %s'%( url, ) )
    request = urllib.request.Request( url, headers={ 'User-Agent': 'BetaSuite-export' } )
    part = dest + '.part'
    with urllib.request.urlopen( request, timeout=300 ) as response, open( part, 'wb' ) as out:
        shutil.copyfileobj( response, out, 4 * 1024 * 1024 )
    if sha256_file( part ) != sha256:
        os.remove( part )
        raise NotAModel( '%s checksum mismatch'%( os.path.basename( dest ), ) )
    os.replace( part, dest )


def find_uv( explicit=None ):
    """uv from --uv, PATH or its usual install folders, else None."""
    candidates = [ explicit ] if explicit else []
    home = os.path.expanduser( '~' )
    candidates += [ shutil.which( 'uv' ) ] + [ os.path.join( home, folder, 'bin', name )
                                                for folder in ( '.local', '.cargo' ) for name in ( 'uv', 'uv.exe' ) ]
    for candidate in candidates:
        if candidate and os.path.isfile( candidate ):
            return candidate
    return None


def run_logged( description, command, log ):
    """Run a command, its output to the log at debug; raise on failure."""
    log.info( '=== starting: %s ==='%( description, ) )
    log.debug( 'command: %s'%( ' '.join( command ), ) )
    result = subprocess.run( command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors='replace' )
    for line in result.stdout.splitlines():
        log.debug( '  %s'%( line, ) )
    if result.returncode != 0:
        for line in result.stdout.splitlines()[-15:]:
            log.error( '  %s'%( line, ) )
        raise RuntimeError( '%s failed (exit %d)'%( description, result.returncode ) )
    log.info( '=== finished: %s ==='%( description, ) )


def build_export_env( workdir, uv, log, tools ):
    """
    A throwaway venv with CPU-only PyTorch and the packages of the given
    export tools only (each a key of EXPORT_PACKAGES). Returns its python.
    """
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
    packages = [ EXPORT_PACKAGES[tool] for tool in EXPORT_PACKAGES if tool in tools ]
    run_logged( 'install %s (export only)'%( ' and '.join( tool for tool in EXPORT_PACKAGES if tool in tools ), ),
                install + packages + EXPORT_PACKAGES_COMMON, log )
    return python


def export_command( python, row, pt, out ):
    """The command that exports one model: python -c <script> <weights> <out> <arg>."""
    export = row['export']
    return [ python, '-c', EXPORT_SCRIPTS[ export['tool'] ], pt, out, str( export['arg'] ) ]


def export_models( rows, out_dir, log, uv=None ):
    """
    Export each row to out_dir/<file>. One throwaway environment for all
    of them, removed whatever happens.

    Returns { model id: ( sha256, size ) } for the exports that worked,
    and the number that failed.
    """
    workdir = tempfile.mkdtemp( prefix='betasuite-export-' )
    done, failures = {}, 0
    try:
        try:
            python = build_export_env( workdir, uv, log, { row['export']['tool'] for row in rows } )
        except Exception as err:
            log.error( 'could not build the export environment: %s'%( err, ) )
            return done, len( rows )
        # The weights are already here and checked; nothing else may be fetched.
        env = dict( os.environ, YOLO_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1' )
        for row in rows:
            source = row['sources'][0]
            pt = os.path.join( workdir, source['file'] )
            out = os.path.join( workdir, row['file'] )
            try:
                download_checked( source['upstream'], source['sha256'], pt, log )
                log.info( '=== starting: export %s ==='%( row['file'], ) )
                result = subprocess.run( export_command( python, row, pt, out ), cwd=workdir, env=env,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, errors='replace' )
                for line in result.stdout.splitlines():
                    log.debug( '  %s'%( line, ) )
                if result.returncode != 0:
                    for line in result.stdout.splitlines()[-15:]:
                        log.error( '  %s'%( line, ) )
                    raise RuntimeError( 'export exited %d'%( result.returncode, ) )
                problem = onnx_problem( out )
                if problem:
                    raise NotAModel( problem )
                dest = os.path.join( out_dir, row['file'] )
                shutil.move( out, dest )
                done[ row['id'] ] = ( sha256_file( dest ), os.path.getsize( dest ) )
                log.info( '=== finished: export %s (%d bytes, sha256 %s) ==='%(
                    row['file'], done[row['id']][1], done[row['id']][0] ) )
            except Exception as err:
                failures += 1
                log.error( 'could not export %s: %s'%( row['file'], err ) )
    finally:
        shutil.rmtree( workdir, ignore_errors=True )
        log.info( 'removed the export environment' )
    return done, failures


def _ask( prompt, default, log ):
    log.debug( 'AWAITING INPUT: %s'%( prompt, ) )
    try:
        answer = input( '%s [%s] '%( prompt, default ) ).strip()
    except EOFError:
        answer = ''
    log.debug( 'input received: %s'%( answer, ) )
    return answer or default


def main( argv=None ):
    rows_all = exportable()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--models', help='comma separated ids, or all (%s)'%(
        ', '.join( row['id'] for row in rows_all ), ) )
    parser.add_argument( '--out', help='folder to write the .onnx files to' )
    parser.add_argument( '--yes', action='store_true', help='take the defaults instead of asking' )
    parser.add_argument( '--uv', help='path to uv, for the throwaway export environment' )
    parser.add_argument( '--log-level', default='info', choices=list( bu_log.LEVEL_NAMES ) )
    parser.add_argument( '--log-file', default=os.path.join( bu_paths.legacy_path( 'output', 'logs' ),
                                                             'export_models.log' ) )
    args = parser.parse_args( argv )
    log = bu_log.tool_logger( 'export_models', args.log_file, file_level='trace', console_level=args.log_level )
    log.info( 'log: %s'%( os.path.abspath( args.log_file ), ) )

    ids = args.models or ( 'all' if args.yes else _ask( 'Models to export (ids, comma separated, or all)', 'all', log ) )
    wanted = [ row for row in rows_all if ids == 'all' or row['id'] in ids.replace( ',', ' ' ).split() ]
    unknown = set( ids.replace( ',', ' ' ).split() ) - { row['id'] for row in rows_all } - { 'all' }
    if unknown or not wanted:
        log.error( 'unknown or no models: %s; choices: %s'%( ', '.join( sorted( unknown ) ) or ids,
                                                            ', '.join( row['id'] for row in rows_all ) ) )
        return 2
    out_dir = args.out or _ask( 'Folder to write the .onnx files to', os.path.abspath( 'exported-models' ), log )
    os.makedirs( out_dir, exist_ok=True )
    log.info( 'exporting %s to %s'%( ', '.join( row['id'] for row in wanted ), out_dir ) )
    done, failures = export_models( wanted, out_dir, log, find_uv( args.uv ) )
    for model_id, ( sha, size ) in sorted( done.items() ):
        pinned = bu_models.entry( model_id )
        same = 'matches the catalog' if pinned['sha256'] == sha else 'differs from the catalog'
        log.info( '%s: sha256 %s, size %d (%s)'%( model_id, sha, size, same ) )
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit( main() )
