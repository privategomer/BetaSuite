"""
fetch_models.py - put BetaSuite's models in place and check them.

Downloads the project's hosted model files (assets/models/catalog.json),
each checked against its pinned SHA-256 before it is used, into the model
folder (tools/lib/where.py resources/model prints it). A download that
stops partway is resumed the next time. The desktop app's Models page
installs, switches and removes the same files.

  320n         the default detector: always installed
  person       the person models for reverse mode, by person backend
               (yolo11_seg: n, s, m; rf_detr_seg: nano, small, medium).
               Unset, it asks at a terminal, defaulting to yolo11_seg plus
               any backend already partly installed; with --yes or no
               terminal it takes that default
  extras       640m and RetinaNet, with --extra-models

A file at a model's place that is not the pinned one is moved aside to
<name>.bad and replaced.

    python tools/setup/fetch_models.py                 # install what is missing, check all
    python tools/setup/fetch_models.py --no-download   # check only
    python tools/setup/fetch_models.py --no-person     # skip the person models
    python tools/setup/fetch_models.py --person-backends yolo11_seg,rf_detr_seg
    python tools/setup/fetch_models.py --extra-models nudenet-640m,nudenet-retinanet --yes

Exit status is 1 if 320n is missing or any model failed to install.
"""

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )
import betautils_log as bu_log  # noqa: E402
import betautils_models as bu_models  # noqa: E402

# Where the models go: the app's model folder (under BETASUITE_DATA_ROOT
# when that names a 1.x tree).
MODEL_DIR = bu_models.model_dir()
REQUIRED = [ 'nudenet-320n' ]
EXTRAS = [ 'nudenet-640m', 'nudenet-retinanet' ]
PERSON_ROWS = bu_models.entries( kind='person' )
# Person backends in the order the prompt lists them, and the default
# choice: the shipped person_backend['selected'].
PERSON_BACKENDS = list( dict.fromkeys( row['backend'] for row in PERSON_ROWS ) )
DEFAULT_PERSON_BACKENDS = [ 'yolo11_seg' ]


def relaunch_in_venv():
    """Re-run this script with the repo's .venv Python if started outside it."""
    if sys.prefix != sys.base_prefix or os.environ.get( 'BETASUITE_NO_RELAUNCH' ):
        return
    for rel in ( ( '.venv', 'bin', 'python' ), ( '.venv', 'Scripts', 'python.exe' ) ):
        venv_python = os.path.join( REPO_ROOT, *rel )
        if os.path.exists( venv_python ):
            print( 'not in the virtual environment; re-running with %s'%( venv_python, ), flush=True )
            os.environ['BETASUITE_NO_RELAUNCH'] = '1'
            args = [ venv_python, os.path.abspath( __file__ ) ] + sys.argv[1:]
            if os.name == 'nt':
                import subprocess
                sys.exit( subprocess.call( args ) )
            os.execv( venv_python, args )


# ---------------------------------------------------------------------------
# Choosing the person backends
# ---------------------------------------------------------------------------

def rows_of( backend ):
    return [ row for row in PERSON_ROWS if row['backend'] == backend ]


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
            text, 'unknown %s'%( ', '.join( unknown ), ) if unknown else 'none named',
            ', '.join( PERSON_BACKENDS ) ) )
    return [ backend for backend in PERSON_BACKENDS if backend in words ]


def default_person_backends( present_files ):
    """
    DEFAULT_PERSON_BACKENDS plus every backend with at least one model
    file present, so a backend chosen before (and then half installed) is
    finished rather than left broken.
    """
    present = { row['backend'] for row in PERSON_ROWS if row['file'] in present_files }
    return [ backend for backend in PERSON_BACKENDS
             if backend in DEFAULT_PERSON_BACKENDS or backend in present ]


def person_backend_menu( present_files ):
    """The prompt's lines: each backend, its sizes, size on disk, what is there."""
    lines = []
    for number, backend in enumerate( PERSON_BACKENDS, 1 ):
        rows = rows_of( backend )
        have = [ row for row in rows if row['file'] in present_files ]
        lines.append( '  %d) %-12s %d models, about %d MB%s: %s'%(
            number, backend, len( rows ), round( sum( row['size'] for row in rows ) / 1e6 ),
            ', %d of %d present'%( len( have ), len( rows ) ) if have else '',
            ', '.join( row['file'] for row in rows ) ) )
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
    Ask which person backends to install; the default without a terminal.
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
        log.info( "auto-answer '%s': person backends to install"%( ','.join( default ) or 'none', ) )
        return default
    log.info( 'Person models (reverse mode person windows only). Choose one or more to install:' )
    for line in person_backend_menu( present_files ):
        log.info( line )
    prompt = 'Person backends: numbers or names, all, none [%s] '%( ','.join( default ) or 'none', )
    while True:
        log.debug( 'AWAITING INPUT: %s'%( prompt.strip(), ) )
        try:
            answer = ask( prompt )
        except EOFError:
            answer = ''
        log.debug( 'input received: %s'%( answer, ) )
        try:
            return choose_person_backends( answer, present_files )
        except ValueError as err:
            log.warning( str( err ) )


def parse_extras( text ):
    """Extra detector ids from a comma separated list, 'all' or 'none'."""
    words = text.replace( ',', ' ' ).split()
    if words == [ 'all' ]:
        return list( EXTRAS )
    if words == [ 'none' ]:
        return []
    unknown = [ word for word in words if word not in EXTRAS ]
    if unknown or not words:
        raise ValueError( 'extra models %r: valid choices: %s, or all, or none'%( text, ', '.join( EXTRAS ) ) )
    return [ model_id for model_id in EXTRAS if model_id in words ]


# ---------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------

def set_aside( row, log ):
    """Move a file that is not the pinned one to <name>.bad, so it is kept but never used."""
    path = bu_models.install_path( row )
    os.replace( path, path + '.bad' )
    log.warning( '%s is not the expected file. Moved it to %s.bad'%( row['file'], path ) )


def install( row, log ):
    """Install one model; True when it is in place afterwards."""
    log.info( '=== starting: download %s (%.1f MB) ==='%( row['file'], row['size'] / 1e6 ) )
    last = { 'step': -1 }

    def progress( done, total ):
        step = done * 10 // total
        if step != last['step']:
            last['step'] = step
            log.debug( '%s: %d of %d bytes'%( row['file'], done, total ) )

    try:
        bu_models.install( row['id'], progress=progress )
    except bu_models.ModelError as err:
        log.error( 'could not install %s: %s %s'%( row['file'], err.code, err.params ) )
        if err.code == 'model.download_failed' and err.params.get( 'reason' ) == bu_models.REASON_NETWORK:
            log.error( '  what arrived is kept; run this again to continue the download' )
        return False
    log.info( '=== finished: download %s ==='%( row['file'], ) )
    return True


def main():
    relaunch_in_venv()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--no-download', action='store_true', help='only check what is already there' )
    parser.add_argument( '--no-person', '--no-export', dest='no_person', action='store_true',
                         help='skip the person models (reverse mode person windows)' )
    parser.add_argument( '--person-backends', metavar='LIST',
                         help='person backends to install, comma separated: %s, all or none '
                              '(default: ask at a terminal, else %s plus any already partly present)'%(
                              ', '.join( PERSON_BACKENDS ), ','.join( DEFAULT_PERSON_BACKENDS ) ) )
    parser.add_argument( '--extra-models', metavar='LIST', default='none',
                         help='extra detectors to install: %s, all or none (default none)'%( ', '.join( EXTRAS ), ) )
    parser.add_argument( '--yes', action='store_true', help='take the default answer instead of asking' )
    parser.add_argument( '--uv', help=argparse.SUPPRESS )   # accepted for older setup scripts; unused
    parser.add_argument( '--log-level', default='info', choices=list( bu_log.LEVEL_NAMES ) )
    parser.add_argument( '--log-file', default=os.path.join( REPO_ROOT, 'fetch_models.log' ) )
    args = parser.parse_args()

    if args.person_backends is not None:
        try:
            chosen_backends = parse_person_backends( args.person_backends )
        except ValueError as err:
            parser.error( '--person-backends: %s'%( err, ) )
    try:
        extras = parse_extras( args.extra_models )
    except ValueError as err:
        parser.error( '--extra-models: %s'%( err, ) )
    log = bu_log.tool_logger( 'fetch_models', os.path.abspath( args.log_file ), file_level='trace',
                          console_level=args.log_level )
    log.info( 'model folder: %s'%( MODEL_DIR, ) )

    states = {}
    for row in bu_models.entries():
        state = bu_models.status( row )['state']
        if state == 'modified':
            set_aside( row, log )
            state = 'absent'
        states[row['id']] = state
        log.log( bu_log.TRACE if state == 'installed' else logging.DEBUG, '%s: %s'%( row['file'], state ) )

    wanted = list( REQUIRED ) + extras
    if not args.no_person:
        present_files = { row['file'] for row in PERSON_ROWS if states[row['id']] == 'installed' }
        if args.person_backends is not None:
            log.info( 'person backends: %s (from --person-backends)'%( ', '.join( chosen_backends ) or 'none', ) )
        elif args.yes:
            chosen_backends = default_person_backends( present_files )
            log.info( "auto-answer '%s': person backends to install (--yes)"%( ','.join( chosen_backends ) or 'none', ) )
        elif all( states[row['id']] == 'installed' for row in PERSON_ROWS ):
            chosen_backends = []   # nothing missing: no question
        else:
            chosen_backends = ask_person_backends( present_files, log )
            log.info( 'person backends: %s'%( ', '.join( chosen_backends ) or 'none', ) )
        wanted += [ row['id'] for backend in chosen_backends for row in rows_of( backend ) ]

    failures = 0
    for row in bu_models.entries():
        if states[row['id']] == 'installed':
            log.info( 'model present: %s'%( row['file'], ) )
            continue
        if row['id'] not in wanted:
            continue
        if args.no_download:
            log.error( '%s is missing'%( row['file'], ) )
            failures += 1
            continue
        if not install( row, log ):
            failures += 1

    not_wanted = [ row for row in bu_models.entries() if row['id'] not in wanted and states[row['id']] != 'installed' ]
    if not_wanted:
        log.info( 'not installed (the desktop app\'s Models page, or --extra-models and --person-backends, '
                  'add them): %s'%( ', '.join( row['file'] for row in not_wanted ), ) )
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit( main() )
