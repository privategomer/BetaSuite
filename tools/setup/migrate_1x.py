#!/usr/bin/env python3
"""
migrate_1x.py - copy a 1.x tree's settings, models, caches and logs into
the 2.x folders by kind.

    python3 tools/setup/migrate_1x.py                 asks for the 1.x folder, shows the plan, asks before copying
    python3 tools/setup/migrate_1x.py --dry-run       the plan only; copies nothing
    python3 tools/setup/migrate_1x.py --from /data/betasuite --yes
    python3 tools/setup/migrate_1x.py --no-cache      leave the (rebuildable, large) cache out

A 1.x tree is the folder holding resources/ and output/; its settings are
the betaconfig_local.py beside that checkout's betaconfig.py. The copy
never moves or changes anything in the 1.x tree, so 1.x keeps working on
it, and running again copies only what is missing. Your videos (the
input and output folders) are not copied: the summary names them and
the two variables that keep the 2.x layout using them where they are.

Without a terminal and without --yes it shows the plan and copies
nothing. Everything goes to the log file (default
<state folder>/logs/migrate_1x.log); --console-level filters the
terminal only. A run writes its full report, file by file, to
<state folder>/migration/report-<time>.json.

Exit codes: 0 done (or nothing to do, or a plan shown), 1 refused or
stopped, 2 a bad argument.
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )

import betautils_log as bu_log  # noqa: E402
import betautils_migrate as bu_migrate  # noqa: E402
import betautils_paths as bu_paths  # noqa: E402

LEVELS = ( 'trace', 'debug', 'info', 'warn', 'error' )


def default_source():
    """BETASUITE_DATA_ROOT when set, else the folder beside this checkout (where 1.x kept its tree)."""
    try:
        root = bu_paths.data_root()
    except bu_paths.PathsError:
        root = None
    return root or os.path.dirname( REPO_ROOT )


def default_local_config():
    value = os.environ.get( 'BETASUITE_LOCAL_CONFIG' )
    if value is not None and value.strip().lower() not in ( '', 'none', 'off' ):
        return os.path.abspath( os.path.expanduser( value ) )
    return os.path.join( REPO_ROOT, bu_migrate.LOCAL_CONFIG_NAME )


def refusal_text( refused ):
    p = refused.params
    if refused.code == 'not_a_1x_tree':
        return "%s holds neither resources/ nor output/, so it is not a 1.x tree"%( p['source'], )
    if refused.code == 'overlap':
        return ( "the 2.x %s folder %s and the 1.x tree %s overlap; copying would write into the 1.x "
                 "tree. Point the 2.x folder elsewhere (BETASUITE_CACHE_DIR, XDG_*_HOME)"%(
                     p['kind'], p['target'], p['source'] ) )
    if refused.code == 'run_in_progress':
        import betautils_runlock as bu_runlock
        holder = { key: p.get( key ) for key in bu_runlock.HOLDER_FIELDS }
        return ( "a run is in progress on the %s side: %s. Wait for it to finish, then migrate"%(
            '1.x' if p['side'] == 'source' else '2.x', bu_runlock.describe( holder ) ) )
    if refused.code == 'no_space':
        return ( "not enough free space in %s: the copy needs %s (with a 1 GiB margin), %s is free. "
                 "Free some, or leave the cache out with --no-cache"%(
                     p['folder'], bu_migrate.human_bytes( p['needed_bytes'] ),
                     bu_migrate.human_bytes( p['free_bytes'] ) ) )
    return refused.code


def ask( prompt, default, stdin, stdout ):
    stdout.write( "%s [%s]: "%( prompt, default ) )
    stdout.flush()
    answer = stdin.readline()
    return answer.strip() or default


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0],
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      epilog=__doc__.split( '\n\n', 1 )[1] )
    parser.add_argument( '--from', dest='source', help="the 1.x tree (default: $%s, or the folder beside "
                         "this checkout)"%( bu_paths.ENV_VAR, ) )
    parser.add_argument( '--local-config', help="the 1.x betaconfig_local.py (default: the one beside "
                         "betaconfig.py, or $BETASUITE_LOCAL_CONFIG); 'none' for no settings" )
    parser.add_argument( '--dry-run', action='store_true', help="show the plan and copy nothing" )
    parser.add_argument( '--yes', action='store_true', help="copy without asking" )
    parser.add_argument( '--no-cache', action='store_true', help="leave the cache out; it is rebuilt as runs need it" )
    parser.add_argument( '--report', help="also write the JSON report here (a dry run writes none otherwise)" )
    parser.add_argument( '--log-file', help="log file (default: <state folder>/logs/migrate_1x.log)" )
    parser.add_argument( '--log-level', default='info', choices=LEVELS, help="log file level" )
    parser.add_argument( '--console-level', default='info', choices=LEVELS, help="terminal level" )
    return parser


def main( argv=None, stdin=None, stdout=None, clock=time.time ):
    args = build_parser().parse_args( argv )
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    try:
        targets = bu_paths.os_dirs()
    except bu_paths.PathsError as err:
        sys.stderr.write( "%s\n"%( err, ) )
        return 1
    log_path = args.log_file or os.path.join( targets['state'], 'logs', 'migrate_1x.log' )
    logger = bu_log.tool_logger( 'migrate_1x', log_path=log_path, file_level=args.log_level,
                                 console_level=args.console_level, console_stream=stdout )
    logger.debug( "command: %s %s"%( os.path.abspath( __file__ ), ' '.join( sys.argv[1:] if argv is None else argv ) ) )
    interactive = not args.yes and not args.dry_run and stdin.isatty()

    source = args.source or default_source()
    local_config = args.local_config or default_local_config()
    if interactive and not args.source:
        source = ask( "1.x tree (the folder holding resources/ and output/)", source, stdin, stdout )
    if interactive and not args.local_config:
        local_config = ask( "1.x settings file ('none' for none)", local_config, stdin, stdout )
    if local_config.strip().lower() == 'none':
        local_config = None
    elif not os.path.isfile( local_config ):
        logger.info( "no settings file at %s; settings are not copied"%( local_config, ) )
        local_config = None
    if not os.path.isabs( source ):
        logger.error( "the 1.x tree must be an absolute path, got %r"%( source, ) )
        return 2

    try:
        plan = bu_migrate.make_plan( source, local_config, targets, include_cache=not args.no_cache )
    except bu_migrate.MigrationRefused as refused:
        logger.error( refusal_text( refused ) )
        return 1
    document = bu_migrate.report( plan, bu_migrate.DRY_RUN )
    for line in bu_migrate.summary_lines( document ):
        logger.info( line )
    to_copy = plan.to_copy()
    total = sum( item.size for item in to_copy )

    if args.dry_run or not to_copy or not ( args.yes or interactive ):
        if not to_copy:
            logger.info( "nothing to copy: every file is already in the 2.x folders" )
        elif not args.dry_run:
            logger.warning( "no terminal to ask on and no --yes, so nothing is copied" )
        if args.report:
            bu_migrate.write_json( args.report, document )
            logger.info( "report: %s"%( args.report, ) )
        return 0
    if interactive:
        answer = ask( "copy %d file(s), %s, into the 2.x folders? (y/N)"%(
            len( to_copy ), bu_migrate.human_bytes( total ) ), 'N', stdin, stdout )
        if answer.lower() not in ( 'y', 'yes' ):
            logger.info( "nothing copied" )
            return 0

    started = clock()
    try:
        outcome, errors = bu_migrate.execute( plan, logger, clock=clock )
    except bu_migrate.MigrationRefused as refused:
        logger.error( refusal_text( refused ) )
        return 1
    finished = clock()
    document = bu_migrate.report( plan, outcome, errors, started, finished )
    report_path, record_path = bu_migrate.record_paths( plan, time.strftime( '%Y%m%d_%H%M%S', time.gmtime( started ) ) )
    bu_migrate.write_json( report_path, document )
    if args.report:
        bu_migrate.write_json( args.report, document )
    if outcome in ( bu_migrate.COMPLETE, bu_migrate.FINISHED_WITH_ERRORS ):
        bu_migrate.write_json( record_path, bu_migrate.migrated_from( plan, finished ) )
    copied = [ item for item in plan.items if item.outcome == 'copied' ]
    logger.info( "copied %d file(s), %s, in %s; report: %s"%(
        len( copied ), bu_migrate.human_bytes( sum( item.size for item in copied ) ),
        bu_log.format_duration( finished - started ), report_path ) )
    logger.info( "outcome: %s"%( outcome, ) )
    return 0 if outcome == bu_migrate.COMPLETE else 1


if __name__ == '__main__':
    raise SystemExit( main() )
