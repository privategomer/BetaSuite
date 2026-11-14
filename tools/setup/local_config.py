#!/usr/bin/env python3
"""
local_config.py - read and edit your own settings, betaconfig_local.py
(beside betaconfig.py, ignored by git; CONFIG_REFERENCE.md, "Local config").

    python tools/setup/local_config.py                      show: every setting, its value and where it came from
    python tools/setup/local_config.py show --backend retinanet_v2 --nn-batch-size 8
                                                            the same, with betatv.py's flags applied on top
    python tools/setup/local_config.py init                 create an empty file with a short header
    python tools/setup/local_config.py set console_palette=deuteranopia encode_crf=21
    python tools/setup/local_config.py set detector_backend.nudenet_v3.nn_batch_size=8
    python tools/setup/local_config.py unset encode_crf     delete the line: betaconfig.py's value applies
    python tools/setup/local_config.py check                exit 1 if the file has a problem
    python tools/setup/local_config.py migrate              move ../resources/betaconfig_machine.py in
    python tools/setup/local_config.py setup --gpu-enabled 1    what setup.sh / setup.ps1 run

Precedence, lowest first: betaconfig.py < betaconfig_local.py < command-line flags.

`set` checks a change the way betatv.py would before writing it, and
writes nothing if it would fail validation. Values are Python literals;
a bare word is taken as a string (h264_nvenc), and None/True/False work.
A dotted name sets one key inside a dict setting.

Everything goes to the log file (default ../output/logs/local_config.log);
--log-level filters the terminal only.
"""

import argparse
import ast
import contextlib
import copy
import io
import os
import re
import sys
import tempfile

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )

import betautils_local as bu_local   # imports nothing of betaconfig's

LEVELS = ( 'trace', 'debug', 'info', 'warn', 'error' )
# How much of a value `show` prints on one row.
SHOW_VALUE_CHARS = 56


def build_logger( log_file, console_level ):
    import betautils_log as bu_log
    return bu_log.tool_logger( 'local_config', log_path=log_file, file_level='trace',
                               console_level=console_level, console_format='%(message)s' )


class Session( object ):
    """betaconfig with the shipped values, plus the local file's path and text."""

    def __init__( self, path, legacy_path=None ):
        self.path = path
        self.legacy_path = bu_local.legacy_machine_path() if legacy_path is None else legacy_path
        # betaconfig must load WITHOUT the local file, so its values are
        # the shipped ones: the file's path was resolved before this.
        os.environ[ bu_local.ENV_VAR ] = 'none'
        cwd = os.getcwd()
        os.chdir( REPO_ROOT )   # betaconfig's relative paths assume the repo root
        try:
            import betaconfig
            import betautils_config as bu_config
        finally:
            os.chdir( cwd )
        self.betaconfig = betaconfig
        self.bu_config = bu_config
        self.names = bu_local.known_names( vars( betaconfig ) )
        self.order = [ name for name in vars( betaconfig ) if name in set( self.names ) ]
        self.repo = { name: copy.deepcopy( getattr( betaconfig, name ) ) for name in self.names }

    def namespace( self ):
        return vars( self.betaconfig )

    def text( self ):
        if not os.path.exists( self.path ):
            return ''
        with open( self.path, 'r', encoding='utf-8' ) as handle:
            return handle.read()

    def restore( self ):
        for name, value in self.repo.items():
            self.namespace()[name] = copy.deepcopy( value )
        self.bu_config.invalidate_config_caches()

    def evaluate( self, text, cli_argv=() ):
        """
        What betatv.py would make of this text (and these flags).

        Returns:
            ( LocalConfig, new errors, { name: value after the local
            file }, { name: value after the flags too }, [ flag applied ] ).
            Only errors the shipped config does not already have count.
        """
        self.restore()
        cwd = os.getcwd()
        os.chdir( REPO_ROOT )
        try:
            baseline = set( self.bu_config._collect_validation_errors() )
            handle, candidate = tempfile.mkstemp( suffix='.py', prefix='betaconfig_local.' )
            try:
                with os.fdopen( handle, 'w', encoding='utf-8' ) as out:
                    out.write( text )
                loaded = bu_local.apply( self.namespace(), path=candidate, stderr=_Null(),
                                         legacy_path=self.legacy_path )
                self.bu_config.invalidate_config_caches()
                after_local = { name: copy.deepcopy( self.namespace().get( name ) ) for name in self.names }
                flags = apply_flags( self.betaconfig, cli_argv )
                after_flags = { name: copy.deepcopy( self.namespace().get( name ) ) for name in self.names }
                # The file's own problems are in validation's errors too; listed once.
                errors = [ error for error in self.bu_config._collect_validation_errors()
                           if error not in baseline and error not in loaded.problems ]
                loaded.path = self.path
                loaded.problems = [ problem.replace( os.path.basename( candidate ),
                                                     os.path.basename( self.path ) )
                                    for problem in loaded.problems ]
            finally:
                os.remove( candidate )
        finally:
            os.chdir( cwd )
            self.restore()
        return loaded, errors, after_local, after_flags, flags

    def write( self, text, logger ):
        bu_local.write_text( self.path, text )
        logger.debug( "wrote %s"%( self.path, ) )


class _Null( object ):
    def write( self, _text ):
        pass


def apply_flags( betaconfig, cli_argv ):
    """betatv.py's own flags applied to betaconfig, as betatv.py applies them; [ 'name=value' ]."""
    if not cli_argv:
        return []
    import betautils_cli as bu_cli
    parser = bu_cli.build_arg_parser( "betatv.py's flags", include_preview=True, include_logging=True )
    args = parser.parse_args( list( cli_argv ) )
    with contextlib.redirect_stdout( io.StringIO() ):   # its summary line; the rows say the same
        return bu_cli.apply_cli_overrides( betaconfig, args )


def parse_assignment( text ):
    """'detector_backend.nudenet_v3.nn_batch_size=8' -> ( ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ), 8 )."""
    if '=' not in text:
        raise ValueError( "%r is not NAME=VALUE"%( text, ) )
    name, raw = text.split( '=', 1 )
    path = parse_name( name.strip() )
    raw = raw.strip()
    try:
        value = ast.literal_eval( raw )
    except ( ValueError, SyntaxError ):
        if not re.match( r'^[A-Za-z_][\w.\-]*$', raw ):
            raise ValueError( "%s's value %r is not a Python literal (quote a string)"%( name, raw ) )
        value = raw
    return path, value


def parse_name( name ):
    parts = tuple( part for part in name.split( '.' ) if part )
    if not parts:
        raise ValueError( "empty setting name" )
    return parts


def check_path( path, repo ):
    """None, or why a dotted path cannot be set: it goes inside a value that is not a dict."""
    value = repo.get( path[0], bu_local.MISSING )
    for depth, part in enumerate( path[1:], start=1 ):
        if value is bu_local.MISSING:
            return None   # a new key; validation and the new-key warning judge it
        if not isinstance( value, dict ) or bu_local.is_censor_style( value ):
            return ( "%s is not a dict in betaconfig.py, so it has no keys to set one by one; "
                     "set it whole"%( bu_local.format_setting_path( path[:depth] ), ) )
        value = value.get( part, bu_local.MISSING )
    return None


def _report( logger, loaded, errors ):
    for problem in loaded.problems:
        logger.error( "  - %s"%( problem, ) )
    for error in errors:
        logger.error( "  - %s"%( error, ) )
    notes = [ bu_local.encoder_scale_warning( loaded ) ] + bu_local.new_key_warnings( loaded )
    for note in notes:
        if note:
            logger.warning( "  note: %s"%( note, ) )


def _warn_legacy( logger, loaded ):
    warning = bu_local.legacy_warning( loaded )
    if warning:
        logger.warning( "warning: %s"%( warning, ) )


def _clip( value ):
    text = repr( value )
    return text if len( text ) <= SHOW_VALUE_CHARS else text[ :SHOW_VALUE_CHARS - 3 ] + '...'


def show_rows( order, repo, after_local, after_flags, loaded ):
    """
    ( label, value, source, betaconfig.py's value or MISSING ) per row:
    one per setting, then one per key changed inside a dict setting.
    """
    rows = []
    for name in order:
        changed_by_flags = set( bu_local.changed_paths( after_local[name], after_flags[name], ( name, ) ) )
        local_paths = [ path for path in loaded.applied if path[0] == name ]
        if ( name, ) in changed_by_flags:
            rows.append( ( name, after_flags[name], 'CLI', repo[name] ) )
            continue
        if ( name, ) in loaded.applied:
            rows.append( ( name, after_flags[name], 'local', repo[name] ) )
            continue
        inner = sorted( set( local_paths ) | changed_by_flags, key=lambda path: [ str( part ) for part in path ] )
        # 'merged': betaconfig.py's dict with the keys on the rows below changed.
        rows.append( ( name, after_flags[name], 'merged' if inner else 'default', bu_local.MISSING ) )
        for path in inner:
            source = 'CLI' if path in changed_by_flags else 'local'
            rows.append( ( '  ' + ''.join( "[%r]"%( part, ) for part in path[1:] ),
                           bu_local.current_value( after_flags, path ), source,
                           bu_local.current_value( repo, path ) ) )
    return rows


def command_show( session, logger, args ):
    logger.info( "local config: %s (%s)"%( session.path,
                 'present' if os.path.exists( session.path ) else 'not created yet' ) )
    logger.info( "precedence, lowest first: %s"%( '  <  '.join(
        bu_local.LAYER_FILES[layer] for layer in bu_local.PRECEDENCE ), ) )
    try:
        loaded, errors, after_local, after_flags, flags = session.evaluate( session.text(), args.flags )
    except SystemExit:
        logger.error( "those are not betatv.py flags (see python betatv.py --help)" )
        return 2
    _warn_legacy( logger, loaded )
    logger.info( loaded.describe() )
    if flags:
        logger.info( "command-line overrides: %s"%( ', '.join( flags ), ) )
    rows = show_rows( session.order, session.repo, after_local, after_flags, loaded )
    width = max( len( row[0] ) for row in rows )
    for label, value, source, default in rows:
        detail = '' if source in ( 'default', 'merged' ) else '   (betaconfig.py: %s)'%( _clip( default ), )
        logger.info( ( "  %-*s  %-*s  %-7s%s"%( width, label, SHOW_VALUE_CHARS, _clip( value ), source,
                                                detail ) ).rstrip() )
    if loaded.problems or errors:
        logger.error( "problems (betatv.py would stop on these):" )
        _report( logger, loaded, errors )
        return 1
    _report( logger, loaded, [] )
    return 0


def command_init( session, logger, _args ):
    # Never rewrites an existing file: it holds only what a person chose.
    if os.path.exists( session.path ):
        logger.info( "%s already exists; left as it is"%( session.path, ) )
        return 0
    session.write( bu_local.template_text(), logger )
    logger.info( "created %s, empty: copy a setting from betaconfig.py into it to change it"%( session.path, ) )
    return 0


def _apply_edit( session, logger, updates, removals, comment, section=None ):
    text = session.text() or bu_local.template_text()
    try:
        new_text = bu_local.edit_text( text, updates, removals, comment, section )
    except ValueError as err:
        logger.error( "cannot edit %s: %s"%( session.path, err ) )
        return 1
    loaded, errors, _local, _flags, _applied = session.evaluate( new_text )
    if loaded.problems or errors:
        logger.error( "not written; betatv.py would stop on this:" )
        _report( logger, loaded, errors )
        return 1
    session.write( new_text, logger )
    for path, value in updates.items():
        logger.info( "set %s = %r in %s"%( bu_local.format_setting_path( path ), value, session.path ) )
    for path in removals:
        logger.info( "removed %s from %s; betaconfig.py's value applies"%(
            bu_local.format_setting_path( path ), session.path ) )
    _report( logger, loaded, [] )
    return 0


def command_set( session, logger, args ):
    try:
        updates = dict( parse_assignment( item ) for item in args.assignments )
    except ValueError as err:
        logger.error( str( err ) )
        return 2
    for path in updates:
        problem = check_path( path, session.repo )
        if problem:
            logger.error( problem )
            return 2
    section = bu_local.HARDWARE_SECTION if args.section == 'hardware' else None
    return _apply_edit( session, logger, updates, [], args.comment, section )


def command_unset( session, logger, args ):
    try:
        removals = [ parse_name( name ) for name in args.names ]
    except ValueError as err:
        logger.error( str( err ) )
        return 2
    if not os.path.exists( session.path ):
        logger.info( "%s does not exist; nothing to remove"%( session.path, ) )
        return 0
    return _apply_edit( session, logger, {}, removals, None )


def command_check( session, logger, _args ):
    loaded, errors, _local, _flags, _applied = session.evaluate( session.text() )
    _warn_legacy( logger, loaded )
    if not os.path.exists( session.path ):
        logger.info( "no local config at %s; betaconfig.py's defaults apply"%( session.path, ) )
        return 0
    if loaded.problems or errors:
        logger.error( "%s has problems:"%( session.path, ) )
        _report( logger, loaded, errors )
        return 1
    logger.info( loaded.describe() )
    _report( logger, loaded, [] )
    return 0


def _ask( question, default_yes, assume_yes, logger ):
    if assume_yes or not sys.stdin.isatty():
        logger.info( "auto-answer '%s': %s"%( 'y' if default_yes else 'n', question ) )
        return default_yes
    logger.debug( "AWAITING INPUT: %s"%( question, ) )
    reply = input( "%s [%s] "%( question, 'Y/n' if default_yes else 'y/N' ) ).strip().lower()
    logger.debug( "input received: %s"%( reply, ) )
    if not reply:
        return default_yes
    return reply in ( 'y', 'yes' )


def command_setup( session, logger, args ):
    """What setup runs: create the file, and match gpu_enabled to the install."""
    status = command_init( session, logger, args )
    if status or args.gpu_enabled is None:
        return status
    want = int( args.gpu_enabled )
    loaded = session.evaluate( session.text() )[0]
    _warn_legacy( logger, loaded )
    in_file = ( 'gpu_enabled', ) in loaded.applied
    have = loaded.applied[ ( 'gpu_enabled', ) ] if in_file else session.repo['gpu_enabled']
    install = 'GPU' if want else 'CPU'
    if have == want:
        logger.info( "gpu_enabled = %d matches this %s install (%s)"%(
            want, install, bu_local.FILE_NAME if in_file else 'betaconfig.py' ) )
        return 0
    if in_file and not _ask( "%s has gpu_enabled = %r but this is a %s install. Set it to %d?"%(
            os.path.basename( session.path ), have, install, want ), True, args.yes, logger ):
        logger.warning( "left gpu_enabled = %r in %s"%( have, session.path ) )
        return 0
    return _apply_edit( session, logger, { ( 'gpu_enabled', ): want }, [], 'set by setup',
                        bu_local.HARDWARE_SECTION )


def command_migrate( session, logger, _args ):
    """Copy the old machine config's settings in, then rename it so it is plainly done."""
    legacy = session.legacy_path
    done = legacy + bu_local.MIGRATED_SUFFIX
    if not os.path.isfile( legacy ):
        logger.info( "nothing to migrate: no %s%s"%( legacy, " (already migrated to %s)"%( done, )
                                                     if os.path.exists( done ) else '' ) )
        return 0
    with open( legacy, 'r', encoding='utf-8' ) as handle:
        legacy_text = handle.read()
    entries, problems = bu_local.parse( legacy_text, os.path.basename( legacy ) )
    if problems:
        logger.error( "not migrated, nothing changed: %s has lines that cannot be read; fix them first:"%( legacy, ) )
        for problem in problems:
            logger.error( "  - %s"%( problem, ) )
        return 1
    text = session.text() or bu_local.template_text()
    local_entries, local_problems = bu_local.parse( text, os.path.basename( session.path ) )
    if local_problems:
        logger.error( "not migrated, nothing changed: %s has problems; fix them first:"%( session.path, ) )
        for problem in local_problems:
            logger.error( "  - %s"%( problem, ) )
        return 1
    plan = bu_local.migration_plan( entries, { name: value for name, value, _line in local_entries },
                                    session.repo )
    comments = bu_local.trailing_comments( legacy_text )
    new_text = text
    for path, value in plan['copy']:
        note = '; '.join( part for part in ( comments.get( path[0] ), 'from %s'%( bu_local.LEGACY_FILE_NAME, ) )
                          if part )
        new_text = bu_local.edit_text( new_text, { path: value }, comment=note,
                                       section=bu_local.HARDWARE_SECTION )
    loaded, errors, _local, _flags, _applied = session.evaluate( new_text )
    if loaded.problems or errors:
        logger.error( "not migrated, nothing changed; betatv.py would stop on the result:" )
        _report( logger, loaded, errors )
        return 1
    if new_text != session.text():
        session.write( new_text, logger )
    for path, value in plan['copy']:
        logger.info( "moved %s = %r"%( bu_local.format_setting_path( path ), value ) )
    for path in plan['same_as_default']:
        logger.info( "not copied, same as betaconfig.py: %s"%( bu_local.format_setting_path( path ), ) )
    for path in plan['already_local']:
        logger.info( "already in %s: %s"%( bu_local.FILE_NAME, bu_local.format_setting_path( path ) ) )
    for path, old, local in plan['conflicts']:
        logger.warning( "kept %s's %s = %r; %s had %r"%( bu_local.FILE_NAME, bu_local.format_setting_path( path ),
                                                         local, bu_local.LEGACY_FILE_NAME, old ) )
    # mv -n: an earlier migration's file is never overwritten.
    if os.path.exists( done ):
        logger.error( "%s copied, but %s already exists, so %s was left where it is; "
                      "compare the two and rename one by hand"%( legacy, done, legacy ) )
        return 1
    os.rename( legacy, done )
    logger.info( "renamed %s to %s; it is no longer read"%( legacy, os.path.basename( done ) ) )
    return 0


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0],
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      epilog=__doc__.split( '\n\n', 1 )[1] )
    parser.add_argument( '--path', help='local config to use (default: %s, or $%s)'%(
        bu_local.default_path(), bu_local.ENV_VAR ) )
    parser.add_argument( '--log-level', default='info', choices=LEVELS,
                         help='terminal level (the log file gets everything)' )
    parser.add_argument( '--log-file', default=os.path.join( REPO_ROOT, '..', 'output', 'logs',
                                                             'local_config.log' ) )
    commands = parser.add_subparsers( dest='command' )
    commands.add_parser( 'show', help="every setting, its value and where it came from; "
                                      "betatv.py's flags after it are applied on top" )
    commands.add_parser( 'init', help="create an empty file with a short header (never rewrites one)" )
    set_parser = commands.add_parser( 'set', help='set NAME=VALUE ...' )
    set_parser.add_argument( 'assignments', nargs='+', metavar='NAME=VALUE' )
    set_parser.add_argument( '--comment', help="trailing comment for the lines written" )
    set_parser.add_argument( '--section', choices=[ 'hardware' ],
                             help="put new lines under the '%s' header"%( bu_local.HARDWARE_SECTION, ) )
    unset = commands.add_parser( 'unset', help="delete NAME ...; betaconfig.py's value applies" )
    unset.add_argument( 'names', nargs='+', metavar='NAME' )
    commands.add_parser( 'check', help='exit 1 if the file has a problem' )
    commands.add_parser( 'migrate', help="move ../resources/%s's settings in, then rename it"%(
        bu_local.LEGACY_FILE_NAME, ) )
    setup = commands.add_parser( 'setup', help='what setup.sh / setup.ps1 run' )
    setup.add_argument( '--gpu-enabled', choices=[ '0', '1' ] )
    setup.add_argument( '--yes', action='store_true', help='accept the default answer' )
    return parser


COMMANDS = { 'show': command_show, 'init': command_init, 'set': command_set,
             'unset': command_unset, 'check': command_check, 'setup': command_setup,
             'migrate': command_migrate }


def main( argv=None, notice=None, legacy_path=None ):
    """
    Args:
        argv: The arguments; default sys.argv[1:].
        notice: A line to log at warn first (the old machine_config.py
            name passes its deprecation note).
        legacy_path: For tests: the old machine config to look for.
    """
    parser = build_parser()
    args, extra = parser.parse_known_args( argv )
    # Only `show` takes more: betatv.py's own flags, applied on top.
    if extra and ( args.command or 'show' ) != 'show':
        parser.error( "unrecognized arguments: %s"%( ' '.join( extra ), ) )
    args.flags = extra
    path = os.path.abspath( args.path ) if args.path else bu_local.configured_path()
    turned_off = os.environ.get( bu_local.ENV_VAR )
    # Before anything imports betaconfig (the logger module does), so it
    # loads with the shipped values; the file is applied by hand.
    os.environ[ bu_local.ENV_VAR ] = 'none'
    logger = build_logger( args.log_file, args.log_level )
    if notice:
        logger.warning( notice )
    if path is None:
        logger.error( "the local config is turned off ($%s=%s); unset it or pass --path"%(
            bu_local.ENV_VAR, turned_off ) )
        return 2
    logger.debug( "command: %s"%( ' '.join( sys.argv ), ) )
    session = Session( path, legacy_path )
    return COMMANDS[ args.command or 'show' ]( session, logger, args )


if __name__ == '__main__':
    sys.exit( main() )
