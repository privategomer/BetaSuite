#!/usr/bin/env python3
"""
catalog_rebuild.py - build the library catalog again from the disk.

    python3 tools/setup/catalog_rebuild.py           says what it will do, asks before rebuilding
    python3 tools/setup/catalog_rebuild.py --yes     rebuilds without asking

The items in the watched folders, where each one is, and the outputs in
the output folders are read from the disk (every new or changed file is
hashed, which reads it whole). The watched folders, type pins, run
history and items not on disk right now are carried from the old file
when it can still be read; when it cannot, they are lost, so keep a
copy of the catalog file as a backup. The old file stays beside the new
one as catalog.sqlite.pre-rebuild.bak.

Close the app first: the rebuild refuses while the app or a
command-line run has the catalog open. Without a terminal and without
--yes it only says what it would do. Everything goes to the log file
(default <state folder>/logs/catalog_rebuild.log); --console-level
filters the terminal only.

Exit codes: 0 rebuilt (or nothing asked), 1 refused or failed, 2 a bad argument.
"""

import argparse
import os
import sys
import time

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
    parser.add_argument( '--catalog', help="the catalog file (default: <data folder>/catalog.sqlite)" )
    parser.add_argument( '--yes', action='store_true', help="rebuild without asking" )
    parser.add_argument( '--log-file', help="log file (default: <state folder>/logs/catalog_rebuild.log)" )
    parser.add_argument( '--log-level', default='info', choices=LEVELS, help="log file level" )
    parser.add_argument( '--console-level', default='info', choices=LEVELS, help="terminal level" )
    return parser


def refusal_text( err ):
    if err.code == 'catalog.unavailable' and err.params.get( 'reason' ) == 'in_use':
        return "the app or a command-line run has the catalog open; close it and run this again"
    if err.code == 'catalog.too_new':
        return ( "the catalog was written by a newer version of BetaSuite (schema %d, this one reads %d); "
                 "update the app"%( err.params['version'], err.params['supported'] ) )
    return "%s %s"%( err.code, err.params )


def main( argv=None, stdin=None, stdout=None ):
    args = build_parser().parse_args( argv )
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    try:
        state = bu_paths.kind_dir( 'state' )
    except bu_paths.PathsError as err:
        sys.stderr.write( "%s\n"%( err, ) )
        return 1
    log_path = args.log_file or os.path.join( state, 'logs', 'catalog_rebuild.log' )
    logger = bu_log.tool_logger( 'catalog_rebuild', log_path=log_path, file_level=args.log_level,
                                 console_level=args.console_level, console_stream=stdout )
    try:
        return _run( args, argv, stdin, stdout, logger )
    finally:
        bu_log.close_logger( logger )


def _run( args, argv, stdin, stdout, logger ):
    import betautils_catalog_rebuild as bu_rebuild
    import betautils_catalog_store as bu_store
    logger.debug( "command: %s %s"%( os.path.abspath( __file__ ), ' '.join( sys.argv[1:] if argv is None else argv ) ) )
    path = os.path.abspath( args.catalog ) if args.catalog else bu_store.default_path()
    old = bu_rebuild.read_old( path, logger )
    logger.info( "catalog: %s"%( path, ) )
    if old.readable:
        logger.info( "the old file is readable: %d watched folder(s), %d item(s), %d run(s) are carried over"%(
            len( old.folders ), len( old.items ), len( old.runs ) ) )
    elif os.path.exists( path ):
        logger.warning( "the old file cannot be read: pins and history will be lost; the default folders are watched" )
    else:
        logger.info( "there is no catalog yet: the default folders are watched" )
    interactive = not args.yes and stdin.isatty()
    if not ( args.yes or interactive ):
        logger.warning( "no terminal to ask on and no --yes, so nothing is rebuilt" )
        return 0
    if interactive:
        if ask( "rebuild the catalog now? (y/N)", 'N', stdin, stdout ).lower() not in ( 'y', 'yes' ):
            logger.info( "nothing rebuilt" )
            return 0
    started = time.monotonic()
    try:
        bu_rebuild.rebuild( path, logger=logger )
    except bu_store.CatalogError as err:
        logger.error( refusal_text( err ) )
        return 1
    logger.info( "done in %s; the old file is %s"%(
        bu_log.format_duration( time.monotonic() - started ), path + bu_rebuild.OLD_SUFFIX ) )
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
