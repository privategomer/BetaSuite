#!/usr/bin/env python3
"""
catalog_backup.py - back up the library catalog, or put a backup back.

    python3 tools/setup/catalog_backup.py                       asks which, and where
    python3 tools/setup/catalog_backup.py backup [--to PATH]    copies the catalog (default: <data folder>/backups)
    python3 tools/setup/catalog_backup.py restore FILE [--yes]  puts FILE back as the catalog

The catalog holds what exists nowhere else: watched folders, type pins,
run history and feedback marks. A backup is the whole file, taken safely
while the app runs. Keep it somewhere other than this machine to carry
them to a reinstall or a new one.

Restore needs the app closed. The file is checked first (readable whole,
a catalog, not from a newer version), the current catalog is kept as
catalog.sqlite.pre-restore.bak, and a backup from an older version takes
the schema steps since. On a new machine, add the watched folders again:
items are known by their content, so pins, history and feedback come
back with each file a scan finds.

Everything goes to the log file (default <state folder>/logs/
catalog_backup.log); --console-level filters the terminal only. Without
a terminal, restore needs --yes and backup takes its defaults.

Exit codes: 0 done (or nothing asked), 1 refused or failed, 2 a bad argument.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )

import betautils_log as bu_log  # noqa: E402
import betautils_paths as bu_paths  # noqa: E402

LEVELS = ( 'trace', 'debug', 'info', 'warn', 'error' )


def ask( prompt, default, stdin, stdout ):
    stdout.write( "%s [%s]: "%( prompt, default ) )
    stdout.flush()
    return stdin.readline().strip() or default


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0],
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      epilog=__doc__.split( '\n\n', 1 )[1] )
    parser.add_argument( 'action', nargs='?', choices=( 'backup', 'restore' ), help="what to do (asked when left out)" )
    parser.add_argument( 'file', nargs='?', help="restore: the backup to put back" )
    parser.add_argument( '--to', help="backup: a folder, or a .sqlite file, to write (default: <data folder>/backups)" )
    parser.add_argument( '--catalog', help="the catalog file (default: <data folder>/catalog.sqlite)" )
    parser.add_argument( '--yes', action='store_true', help="restore without asking; backup over an existing file" )
    parser.add_argument( '--log-file', help="log file (default: <state folder>/logs/catalog_backup.log)" )
    parser.add_argument( '--log-level', default='info', choices=LEVELS, help="log file level" )
    parser.add_argument( '--console-level', default='info', choices=LEVELS, help="terminal level" )
    return parser


def refusal_text( err ):
    code, params = err.code, err.params
    if code == 'catalog.unavailable' and params.get( 'reason' ) == 'in_use':
        return "the app or a command-line run has the catalog open; close it and run this again"
    if code == 'catalog.too_new':
        return ( "that catalog was written by a newer version of BetaSuite (schema %d, this one reads %d); "
                 "update the app first"%( params['version'], params['supported'] ) )
    if code == 'backup.no_catalog':
        return "there is no catalog at %s yet, so there is nothing to back up"%( params['path'], )
    if code == 'backup.no_such_file':
        return "no file at %s"%( params['path'], )
    if code == 'backup.not_a_catalog':
        if params.get( 'reason' ) == 'damaged':
            return "%s cannot be read whole; it may be damaged"%( params['path'], )
        return "%s is not a BetaSuite catalog"%( params['path'], )
    if code == 'backup.same_file':
        return "%s is the catalog itself; choose another file"%( params['path'], )
    if code == 'backup.exists':
        return "%s exists; choose another name, or pass --yes to replace it"%( params['path'], )
    return "%s %s"%( code, params )


def main( argv=None, stdin=None, stdout=None ):
    parser = build_parser()
    args = parser.parse_args( argv )
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    try:
        state = bu_paths.kind_dir( 'state' )
    except bu_paths.PathsError as err:
        sys.stderr.write( "%s\n"%( err, ) )
        return 1
    log_path = args.log_file or os.path.join( state, 'logs', 'catalog_backup.log' )
    logger = bu_log.tool_logger( 'catalog_backup', log_path=log_path, file_level=args.log_level,
                                 console_level=args.console_level, console_stream=stdout )
    try:
        return _run( args, argv, stdin, stdout, logger )
    finally:
        bu_log.close_logger( logger )


def _run( args, argv, stdin, stdout, logger ):
    import betautils_catalog_backup as bu_backup
    import betautils_catalog_store as bu_store
    logger.debug( "command: %s %s"%( os.path.abspath( __file__ ), ' '.join( sys.argv[1:] if argv is None else argv ) ) )
    path = os.path.abspath( args.catalog ) if args.catalog else bu_store.default_path()
    logger.info( "catalog: %s"%( path, ) )
    interactive = stdin.isatty()
    action = args.action
    if action is None:
        if not interactive:
            logger.warning( "no terminal to ask on and no action given, so nothing is done" )
            return 0
        answer = ask( "back up the catalog, or restore a backup? (backup/restore)", 'backup', stdin, stdout ).lower()
        action = { 'b': 'backup', 'backup': 'backup', 'r': 'restore', 'restore': 'restore' }.get( answer )
        if action is None:
            logger.error( "not an answer: %s"%( answer, ) )
            return 2
    try:
        if action == 'backup':
            return _backup( args, path, interactive, stdin, stdout, logger, bu_backup )
        return _restore( args, path, interactive, stdin, stdout, logger, bu_backup )
    except ( bu_backup.BackupError, bu_store.CatalogError ) as err:
        logger.error( refusal_text( err ) )
        return 1
    except OSError as err:
        logger.error( "failed: %s"%( err, ) )
        return 1


def _backup( args, path, interactive, stdin, stdout, logger, bu_backup ):
    target = args.to
    if target is None:
        target = bu_backup.default_folder()
        if interactive and not args.yes:
            target = ask( "folder, or .sqlite file, to write the backup to", target, stdin, stdout )
    target = os.path.abspath( os.path.expanduser( target ) )
    summary = bu_backup.backup( target, path=path, overwrite=args.yes, logger=logger )
    logger.info( "keep a copy of %s off this machine to carry the library to a new one"%( summary['path'], ) )
    return 0


def _restore( args, path, interactive, stdin, stdout, logger, bu_backup ):
    source = args.file
    if source is None:
        if not interactive:
            logger.error( "restore needs the backup file to put back" )
            return 2
        source = ask( "backup file to restore", _newest( bu_backup.default_folder() ) or '', stdin, stdout )
        if not source:
            logger.info( "nothing restored" )
            return 0
    source = os.path.abspath( os.path.expanduser( source ) )
    found = bu_backup.describe( source )
    logger.info( "%s: schema %d, %d items, %d runs, %d feedback marks"%(
        source, found['version'], found['items'], found['runs'], found['feedback'] ) )
    if not args.yes:
        if not interactive:
            logger.warning( "no terminal to ask on and no --yes, so nothing is restored" )
            return 0
        if ask( "replace the current catalog with it? the current one is kept beside it (y/N)", 'N',
                stdin, stdout ).lower() not in ( 'y', 'yes' ):
            logger.info( "nothing restored" )
            return 0
    bu_backup.restore( source, path=path, logger=logger )
    return 0


def _newest( folder ):
    try:
        names = [ name for name in os.listdir( folder ) if name.endswith( '.sqlite' ) ]
    except OSError:
        return None
    if not names:
        return None
    return max( ( os.path.join( folder, name ) for name in names ), key=os.path.getmtime )


if __name__ == '__main__':
    raise SystemExit( main() )
