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
  rf-detr-seg-*  person masks for reverse mode (nano, small, medium):
             Roboflow's .pth releases (pinned by hash) exported to
             ONNX the same way, in the same environment (rfdetr added)

The person models are grouped by person backend (yolo11_seg, rf_detr_seg).
Only the chosen backends are exported, and the export environment gets
only their packages. Unset, it asks at a terminal, defaulting to
yolo11_seg plus any backend already partly installed; with --yes or no
terminal it takes that default.

Anything already present is checked. A file that isn't a real ONNX model
(for example a GitHub sign-in page saved under a .onnx name) is moved
aside to <name>.bad instead of being trusted.

    python tools/setup/fetch_models.py           # fetch/export what is missing, check all
    python tools/setup/fetch_models.py --no-download   # check only
    python tools/setup/fetch_models.py --no-export     # skip the person-model export
    python tools/setup/fetch_models.py --person-backends yolo11_seg,rf_detr_seg
    python tools/setup/fetch_models.py --person-backends all --yes

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
ROBOFLOW_ASSETS = 'https://repo.roboflow.com/rfdetr/'

# The throwaway export environment. CPU-only PyTorch: export needs no GPU,
# and the CPU wheels are a fraction of the CUDA ones. The environment
# variable points it at a mirror.
EXPORT_TORCH_INDEX = os.environ.get( 'BETASUITE_EXPORT_TORCH_INDEX', 'https://download.pytorch.org/whl/cpu' )
# Installed for every export, then the package of each export tool in use.
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
        'person_backend': 'yolo11_seg', 'approx_mb': 12,
        'file': 'yolo11n-seg.onnx',
        'desc': 'YOLO11n-seg person masks (reverse mode person windows)',
        'export': { 'pt_url': ULTRALYTICS_ASSETS + 'yolo11n-seg.pt', 'imgsz': 640,
                    'pt_sha256': '55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152' },
    },
    'yolo11s-seg': {
        'person_backend': 'yolo11_seg', 'approx_mb': 41,
        'file': 'yolo11s-seg.onnx',
        'desc': 'YOLO11s-seg person masks (reverse mode person windows, default)',
        'export': { 'pt_url': ULTRALYTICS_ASSETS + 'yolo11s-seg.pt', 'imgsz': 640,
                    'pt_sha256': '1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3' },
    },
    'yolo11m-seg': {
        'person_backend': 'yolo11_seg', 'approx_mb': 90,
        'file': 'yolo11m-seg.onnx',
        'desc': 'YOLO11m-seg person masks (reverse mode person windows)',
        'export': { 'pt_url': ULTRALYTICS_ASSETS + 'yolo11m-seg.pt', 'imgsz': 640,
                    'pt_sha256': 'eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf' },
    },
    # Roboflow's own md5s for these files (rfdetr's model registry) match
    # the downloads the sha256s below were taken from.
    'rf-detr-seg-nano': {
        'person_backend': 'rf_detr_seg', 'approx_mb': 125,
        'file': 'rf-detr-seg-nano.onnx',
        'desc': 'RF-DETR-Seg nano person masks (reverse mode person windows)',
        'export': { 'tool': 'rfdetr', 'pt_url': ROBOFLOW_ASSETS + 'rf-detr-seg-n-ft.pth', 'arg': 'RFDETRSegNano',
                    'pt_sha256': 'a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b' },
    },
    'rf-detr-seg-small': {
        'person_backend': 'rf_detr_seg', 'approx_mb': 125,
        'file': 'rf-detr-seg-small.onnx',
        'desc': 'RF-DETR-Seg small person masks (reverse mode person windows)',
        'export': { 'tool': 'rfdetr', 'pt_url': ROBOFLOW_ASSETS + 'rf-detr-seg-s-ft.pth', 'arg': 'RFDETRSegSmall',
                    'pt_sha256': '6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3' },
    },
    'rf-detr-seg-medium': {
        'person_backend': 'rf_detr_seg', 'approx_mb': 125,
        'file': 'rf-detr-seg-medium.onnx',
        'desc': 'RF-DETR-Seg medium person masks (reverse mode person windows)',
        'export': { 'tool': 'rfdetr', 'pt_url': ROBOFLOW_ASSETS + 'rf-detr-seg-m-ft.pth', 'arg': 'RFDETRSegMedium',
                    'pt_sha256': '3ad325094735f431aee9962a8d204d68eb5bfc393d53e7e836e70998fef5ea58' },
    },
}

# Person backends in the order the prompt lists them, and the default
# choice: the shipped person_backend['selected'].
PERSON_BACKENDS = list( dict.fromkeys( spec['person_backend'] for spec in MODELS.values()
                                       if 'person_backend' in spec ) )
DEFAULT_PERSON_BACKENDS = [ 'yolo11_seg' ]


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


def parse_person_backends( text ):
    """
    The person backends named in text: comma or space separated names,
    'all' or 'none'. Raises ValueError naming the bad value and the choices.
    """
    words = text.replace( ',', ' ' ).split()
    if words == [ 'all' ]:
        return list( PERSON_BACKENDS )
    if words == [ 'none' ]:
        return []
    unknown = [ word for word in words if word not in PERSON_BACKENDS ]
    if unknown or not words:
        raise ValueError( 'person backends %r: %s - valid choices: %s, or all, or none'%(
            text, 'unknown %s'%( ', '.join( unknown ) ) if unknown else 'none named',
            ', '.join( PERSON_BACKENDS ) ) )
    return [ backend for backend in PERSON_BACKENDS if backend in words ]


def default_person_backends( present_files ):
    """
    DEFAULT_PERSON_BACKENDS plus every backend with at least one model
    file present, so a backend chosen before (and then half exported) is
    finished rather than left broken.
    """
    present = { spec['person_backend'] for spec in MODELS.values()
                if 'person_backend' in spec and spec['file'] in present_files }
    return [ backend for backend in PERSON_BACKENDS
             if backend in DEFAULT_PERSON_BACKENDS or backend in present ]


def person_backend_menu( present_files ):
    """The prompt's lines: each backend, its sizes, size on disk, what is there."""
    lines = []
    for number, backend in enumerate( PERSON_BACKENDS, 1 ):
        specs = [ spec for spec in MODELS.values() if spec.get( 'person_backend' ) == backend ]
        have = [ spec for spec in specs if spec['file'] in present_files ]
        lines.append( '  %d) %-12s %d models, about %d MB%s: %s'%(
            number, backend, len( specs ), sum( spec['approx_mb'] for spec in specs ),
            ', %d of %d present'%( len( have ), len( specs ) ) if have else '',
            ', '.join( spec['file'] for spec in specs ) ) )
    return lines


def choose_person_backends( answer, present_files ):
    """
    The backends for a typed answer: blank takes the default, numbers
    pick from the menu, names and all/none as parse_person_backends.
    """
    if not answer.strip():
        return default_person_backends( present_files )
    words = answer.replace( ',', ' ' ).split()
    names = [ PERSON_BACKENDS[ int( word ) - 1 ] if word.isdigit() and 1 <= int( word ) <= len( PERSON_BACKENDS )
              else word for word in words ]
    return parse_person_backends( ' '.join( names ) )


def ask_person_backends( present_files, log, ask=None, interactive=None ):
    """
    Ask which person backends to export; the default without a terminal.
    A bad answer is asked again.
    """
    # ask is looked up per call, not bound as a default argument: a default
    # of input froze the builtin, so a patched input never applied and the
    # real one blocked the test suite on stdin.
    ask = ask or input
    default = default_person_backends( present_files )
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        log.info( "auto-answer '%s': person backends to export"%( ','.join( default ) or 'none' ) )
        return default
    log.info( 'Person models (reverse mode person windows only). Choose one or more to export:' )
    for line in person_backend_menu( present_files ):
        log.info( line )
    prompt = 'Person backends: numbers or names, all, none [%s] '%( ','.join( default ) or 'none' )
    while True:
        log.debug( 'AWAITING INPUT: %s'%( prompt.strip() ) )
        try:
            answer = ask( prompt )
        except EOFError:
            answer = ''
        log.debug( 'input received: %s'%( answer ) )
        try:
            return choose_person_backends( answer, present_files )
        except ValueError as e:
            log.warning( str( e ) )


def build_export_env( workdir, uv, log, tools ):
    """
    A throwaway venv with CPU-only PyTorch and the packages of the given
    export tools only (each tool a key of EXPORT_PACKAGES). Returns its python.
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
    run_logged( 'install %s (export only)'%( ' and '.join( tool for tool in EXPORT_PACKAGES if tool in tools ) ),
                install + packages + EXPORT_PACKAGES_COMMON, log )
    return python


def export_tool( export ):
    return export.get( 'tool', 'ultralytics' )


def export_command( python, export, pt, out ):
    """The command that exports one model: python -c <script> <weights> <out> <arg>."""
    tool = export_tool( export )
    arg = export['arg'] if tool == 'rfdetr' else export['imgsz']
    return [ python, '-c', EXPORT_SCRIPTS[tool], pt, out, str( arg ) ]


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
            python = build_export_env( workdir, uv, log, { export_tool( spec['export'] ) for spec, _dest in missing } )
        except Exception as e:
            log.error( 'could not build the export environment: %s'%( e ) )
            return len( missing )
        # The weights are already here and checked; nothing else may be fetched.
        env = dict( os.environ, YOLO_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1' )
        for spec, dest in missing:
            export = spec['export']
            pt = os.path.join( workdir, os.path.basename( export['pt_url'] ) )
            out = os.path.join( workdir, spec['file'] )
            try:
                download_checked( export['pt_url'], export['pt_sha256'], pt, log )
                log.info( '=== starting: export %s ==='%( spec['file'] ) )
                result = subprocess.run( export_command( python, export, pt, out ),
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
    parser.add_argument( '--person-backends', metavar='LIST',
                         help='person backends to export, comma separated: %s, all or none '
                              '(default: ask at a terminal, else %s plus any already partly present)'%(
                              ', '.join( PERSON_BACKENDS ), ','.join( DEFAULT_PERSON_BACKENDS ) ) )
    parser.add_argument( '--yes', action='store_true', help='take the default answer instead of asking' )
    parser.add_argument( '--uv', help='path to uv, for the throwaway export environment' )
    parser.add_argument( '--log-level', default='info', choices=list( LEVELS ) )
    parser.add_argument( '--log-file', default=os.path.join( REPO_ROOT, 'fetch_models.log' ) )
    args = parser.parse_args()

    if args.person_backends is not None:
        try:
            chosen_backends = parse_person_backends( args.person_backends )
        except ValueError as e:
            parser.error( '--person-backends: %s'%( e ) )
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
        present_files = { spec['file'] for spec in MODELS.values() if 'export' in spec
                          and os.path.exists( os.path.join( MODEL_DIR, spec['file'] ) ) }
        if args.person_backends is not None:
            log.info( 'person backends to export: %s (from --person-backends)'%(
                ', '.join( chosen_backends ) or 'none' ) )
        elif args.yes:
            chosen_backends = default_person_backends( present_files )
            log.info( "auto-answer '%s': person backends to export (--yes)"%(
                ','.join( chosen_backends ) or 'none' ) )
        else:
            chosen_backends = ask_person_backends( present_files, log )
            log.info( 'person backends to export: %s'%( ', '.join( chosen_backends ) or 'none' ) )
        skipped = [ spec for spec, _dest in export_missing if spec['person_backend'] not in chosen_backends ]
        for spec in skipped:
            log.info( 'not exporting %s: person backend %s not chosen (fetch_models.py --person-backends %s '
                      'adds it)'%( spec['file'], spec['person_backend'], spec['person_backend'] ) )
        export_missing = [ ( spec, dest ) for spec, dest in export_missing
                           if spec['person_backend'] in chosen_backends ]

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
