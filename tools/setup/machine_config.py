#!/usr/bin/env python3
"""
machine_config.py - read and edit this machine's hardware settings
(../resources/betaconfig_machine.py; CONFIG_REFERENCE.md -> "Machine config").

    python tools/setup/machine_config.py                    show: every setting, its value and where it comes from
    python tools/setup/machine_config.py init               create the file, every setting commented out
    python tools/setup/machine_config.py set encode_video_codec=h264_nvenc encode_preset=p5 encode_crf=21
    python tools/setup/machine_config.py set detector_backend.nudenet_v3.nn_batch_size=8
    python tools/setup/machine_config.py unset encode_crf
    python tools/setup/machine_config.py check              exit 1 if the file has a problem
    python tools/setup/machine_config.py setup --gpu-enabled 1    what setup.sh / setup.ps1 run

`set` checks a change the way betatv.py would before writing it, and
writes nothing if it would fail validation. Values are Python literals;
a bare word is taken as a string (h264_nvenc), and None/True/False work.

Everything goes to the log file (default ../output/logs/machine_config.log);
--log-level filters the terminal only.
"""

import argparse
import copy
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )

import betautils_machine as bu_machine   # imports nothing of betaconfig's

LEVELS = ( 'trace', 'debug', 'info', 'warn', 'error' )


def build_logger( log_file, console_level ):
    import betautils_log as bu_log
    return bu_log.tool_logger( 'machine_config', log_path=log_file, file_level='trace',
                               console_level=console_level, console_format='%(message)s' )


class Session( object ):
    """betaconfig with the repo's values, plus the file's path and text."""

    def __init__( self, path ):
        self.path = path
        # betaconfig must load WITHOUT the machine file, so its values are
        # the repo's own: the file's path was resolved above, before this.
        os.environ[ bu_machine.ENV_VAR ] = 'none'
        cwd = os.getcwd()
        os.chdir( REPO_ROOT )   # betaconfig's relative paths assume the repo root
        try:
            import betaconfig
            import betautils_config as bu_config
        finally:
            os.chdir( cwd )
        self.betaconfig = betaconfig
        self.bu_config = bu_config
        self.repo = { name: copy.deepcopy( getattr( betaconfig, name, bu_machine.MISSING ) )
                      for name in bu_machine.SETTING_NAMES }

    def namespace( self ):
        return vars( self.betaconfig )

    def text( self ):
        if not os.path.exists( self.path ):
            return ''
        with open( self.path, 'r', encoding='utf-8' ) as handle:
            return handle.read()

    def restore( self ):
        for name, value in self.repo.items():
            if value is bu_machine.MISSING:
                self.namespace().pop( name, None )
            else:
                self.namespace()[name] = copy.deepcopy( value )

    def evaluate( self, text ):
        """
        ( MachineConfig, new errors ): what betatv.py would make of this
        text. Only errors the repo's own config does not already have
        count against it.
        """
        self.restore()
        cwd = os.getcwd()
        os.chdir( REPO_ROOT )
        try:
            baseline = set( self.bu_config._collect_validation_errors() )
            handle, candidate = tempfile.mkstemp( suffix='.py', prefix='betaconfig_machine.' )
            try:
                with os.fdopen( handle, 'w', encoding='utf-8' ) as out:
                    out.write( text )
                loaded = bu_machine.apply( self.namespace(), path=candidate, stderr=_Null() )
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
        return loaded, errors

    def write( self, text, logger ):
        os.makedirs( os.path.dirname( self.path ), exist_ok=True )
        temporary = self.path + '.tmp'
        with open( temporary, 'w', encoding='utf-8' ) as out:
            out.write( text )
        os.replace( temporary, self.path )
        logger.debug( "wrote %s"%( self.path, ) )


class _Null( object ):
    def write( self, _text ):
        pass


def parse_assignment( text ):
    """'detector_backend.nudenet_v3.nn_batch_size=8' -> ( ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ), 8 )."""
    import ast
    import re
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
    if parts[0] in bu_machine.NESTED:
        if len( parts ) != 3:
            raise ValueError( "%s is set per backend: %s.BACKEND.%s"%(
                parts[0], parts[0], bu_machine.NESTED[ parts[0] ][0] ) )
    elif len( parts ) != 1:
        raise ValueError( "%s has no parts to set separately"%( parts[0], ) )
    return parts


def _report( logger, loaded, errors ):
    for problem in loaded.problems:
        logger.error( "  - %s"%( problem, ) )
    for error in errors:
        logger.error( "  - %s"%( error, ) )
    warning = bu_machine.encoder_scale_warning( loaded )
    if warning:
        logger.warning( "  note: %s"%( warning, ) )


def command_show( session, logger, _args ):
    logger.info( "machine config: %s (%s)"%( session.path,
                 'present' if os.path.exists( session.path ) else 'not created yet' ) )
    loaded, errors = session.evaluate( session.text() )
    rows = []
    for name in bu_machine.SETTING_NAMES:
        repo = session.repo[name]
        if name in bu_machine.NESTED:
            blocks = repo if isinstance( repo, dict ) else {}
            for block, settings in sorted( blocks.items() ):
                if not isinstance( settings, dict ):
                    continue
                for key in bu_machine.NESTED[name]:
                    path = ( name, block, key )
                    base = settings.get( key, bu_machine.MISSING )
                    if path in loaded.applied:
                        rows.append( ( bu_machine.format_setting_path( path ), loaded.applied[path],
                                       'machine', base ) )
                    elif base is not bu_machine.MISSING:
                        rows.append( ( bu_machine.format_setting_path( path ), base, 'betaconfig.py', base ) )
            continue
        path = ( name, )
        if path in loaded.applied:
            rows.append( ( name, loaded.applied[path], 'machine', repo ) )
        else:
            rows.append( ( name, repo, 'betaconfig.py', repo ) )
    width = max( len( row[0] ) for row in rows )
    for label, value, source, repo in rows:
        detail = '' if source == 'betaconfig.py' else '   (betaconfig.py: %r)'%( repo, )
        logger.info( "  %-*s  %-16r  %-13s%s"%( width, label, value, source, detail ) )
    if loaded.problems or errors:
        logger.error( "problems (betatv.py would stop on these):" )
        _report( logger, loaded, errors )
        return 1
    _report( logger, loaded, [] )
    return 0


def command_init( session, logger, args ):
    if os.path.exists( session.path ) and not args.force:
        logger.info( "%s already exists; left as it is (--force rewrites it)"%( session.path, ) )
        return 0
    session.write( bu_machine.template_text( session.repo ), logger )
    logger.info( "created %s, every setting commented out at betaconfig.py's value"%( session.path, ) )
    return 0


def _apply_edit( session, logger, updates, removals, comment ):
    text = session.text() or bu_machine.template_text( session.repo )
    try:
        new_text = bu_machine.edit_text( text, updates, removals, session.repo, comment )
    except ValueError as err:
        logger.error( "cannot edit %s: %s"%( session.path, err ) )
        return 1
    loaded, errors = session.evaluate( new_text )
    if loaded.problems or errors:
        logger.error( "not written; betatv.py would stop on this:" )
        _report( logger, loaded, errors )
        return 1
    session.write( new_text, logger )
    for path, value in updates.items():
        logger.info( "set %s = %r in %s"%( bu_machine.format_setting_path( path ), value, session.path ) )
    for path in removals:
        logger.info( "removed %s from %s; betaconfig.py's value applies"%(
            bu_machine.format_setting_path( path ), session.path ) )
    _report( logger, loaded, [] )
    return 0


def command_set( session, logger, args ):
    try:
        updates = dict( parse_assignment( item ) for item in args.assignments )
    except ValueError as err:
        logger.error( str( err ) )
        return 2
    return _apply_edit( session, logger, updates, [], args.comment )


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
    if not os.path.exists( session.path ):
        logger.info( "no machine config at %s; betaconfig.py's hardware settings apply"%( session.path, ) )
        return 0
    loaded, errors = session.evaluate( session.text() )
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
    status = command_init( session, logger, argparse.Namespace( force=False ) )
    if status or args.gpu_enabled is None:
        return status
    want = int( args.gpu_enabled )
    loaded, _errors = session.evaluate( session.text() )
    in_file = ( 'gpu_enabled', ) in loaded.applied
    have = loaded.applied[ ( 'gpu_enabled', ) ] if in_file else session.repo['gpu_enabled']
    install = 'GPU' if want else 'CPU'
    if have == want:
        logger.info( "gpu_enabled = %d matches this %s install (%s)"%(
            want, install, 'machine config' if in_file else 'betaconfig.py' ) )
        return 0
    if in_file and not _ask( "%s has gpu_enabled = %r but this is a %s install. Set it to %d?"%(
            os.path.basename( session.path ), have, install, want ), True, args.yes, logger ):
        logger.warning( "left gpu_enabled = %r in %s"%( have, session.path ) )
        return 0
    return _apply_edit( session, logger, { ( 'gpu_enabled', ): want }, [], 'set by setup' )


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0],
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      epilog=__doc__.split( '\n\n', 1 )[1] )
    parser.add_argument( '--path', help='machine config to use (default: %s, or $%s)'%(
        bu_machine.default_path(), bu_machine.ENV_VAR ) )
    parser.add_argument( '--log-level', default='info', choices=LEVELS,
                         help='terminal level (the log file gets everything)' )
    parser.add_argument( '--log-file', default=os.path.join( REPO_ROOT, '..', 'output', 'logs',
                                                             'machine_config.log' ) )
    commands = parser.add_subparsers( dest='command' )
    commands.add_parser( 'show', help='every setting, its value and where it comes from' )
    init = commands.add_parser( 'init', help="create the file, every setting commented out" )
    init.add_argument( '--force', action='store_true', help='rewrite an existing file' )
    set_parser = commands.add_parser( 'set', help='set NAME=VALUE ...' )
    set_parser.add_argument( 'assignments', nargs='+', metavar='NAME=VALUE' )
    set_parser.add_argument( '--comment', help="trailing comment for the lines written" )
    unset = commands.add_parser( 'unset', help="remove NAME ...; betaconfig.py's value applies" )
    unset.add_argument( 'names', nargs='+', metavar='NAME' )
    commands.add_parser( 'check', help='exit 1 if the file has a problem' )
    setup = commands.add_parser( 'setup', help='what setup.sh / setup.ps1 run' )
    setup.add_argument( '--gpu-enabled', choices=[ '0', '1' ] )
    setup.add_argument( '--yes', action='store_true', help='accept the default answer' )
    return parser


COMMANDS = { 'show': command_show, 'init': command_init, 'set': command_set,
             'unset': command_unset, 'check': command_check, 'setup': command_setup }


def main( argv=None ):
    args = build_parser().parse_args( argv )
    path = os.path.abspath( args.path ) if args.path else bu_machine.configured_path()
    turned_off = os.environ.get( bu_machine.ENV_VAR )
    # Before anything imports betaconfig (the logger module does), so it
    # loads with the repo's own values; the file is applied by hand.
    os.environ[ bu_machine.ENV_VAR ] = 'none'
    logger = build_logger( args.log_file, args.log_level )
    if path is None:
        logger.error( "the machine config is turned off ($%s=%s); unset it or pass --path"%(
            bu_machine.ENV_VAR, turned_off ) )
        return 2
    logger.debug( "command: %s"%( ' '.join( sys.argv ), ) )
    session = Session( path )
    return COMMANDS[ args.command or 'show' ]( session, logger, args )


if __name__ == '__main__':
    sys.exit( main() )
