#!/usr/bin/env python3
"""
dump_settings_fixture.py - write the settings registry, as the engine sends
it, for the app's checks that run without the engine.

    python tools/maintenance/dump_settings_fixture.py [--out PATH]
        [--log-file PATH] [--log-level LEVEL] [--console-level LEVEL]

No input needed. Writes ui/scripts/fixtures/settings.json:

    schema  what config.schema answers
    paths   every setting path betaconfig.py holds, so a check can fill in
            the keys a user names (structure profiles, profile folders)

Only betaconfig.py counts: the local config layer is never read, so the
file is the same on every machine. tests/test_settings_fixture.py fails
when the file is out of date; run this tool to refresh it.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
sys.path.insert( 0, REPO )

import betautils_config_store as bu_store  # noqa: E402
import betautils_log as bu_log  # noqa: E402
import betautils_settings as bu_settings  # noqa: E402

LEVELS = ( 'trace', 'debug', 'info', 'warn', 'error' )
DEFAULT_OUT = os.path.join( REPO, 'ui', 'scripts', 'fixtures', 'settings.json' )


def fixture():
    """The fixture's content, from betaconfig.py alone."""
    store = bu_store.ConfigStore( path=None )
    # Some choices follow other settings (encode_preset's follow the codec),
    # so the schema is taken while betaconfig.py's own values are live.
    evaluation = store.evaluate( '', extra=lambda _config: bu_settings.schema() )
    paths = sorted( '.'.join( str( part ) for part in path )
                    for path, _value, entry in bu_settings.concrete_leaves( evaluation.values )
                    if entry is not None and len( entry.path ) == len( path ) )
    return { 'schema': evaluation.extra, 'paths': paths }


def render( content ):
    return json.dumps( content, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False ) + '\n'


def main( argv=None ):
    parser = argparse.ArgumentParser( description="Write the settings registry fixture for the app's checks." )
    parser.add_argument( '--out', default=DEFAULT_OUT, help='where to write it (default: %(default)s)' )
    parser.add_argument( '--log-file', default=os.path.join( os.path.dirname( bu_log.configured_log_path() ),
                                                             'dump_settings_fixture.log' ) )
    parser.add_argument( '--log-level', choices=LEVELS, default='info' )
    parser.add_argument( '--console-level', choices=LEVELS, default='info' )
    args = parser.parse_args( argv )
    logger = bu_log.tool_logger( 'dump_settings_fixture', log_path=args.log_file, file_level=args.log_level,
                                 console_level=args.console_level )
    try:
        content = fixture()
        text = render( content )
        logger.debug( "%d settings, %d paths, %d bytes", len( content['schema']['settings'] ),
                      len( content['paths'] ), len( text ) )
        os.makedirs( os.path.dirname( args.out ), exist_ok=True )
        old = None
        if os.path.exists( args.out ):
            with open( args.out, encoding='utf-8' ) as handle:
                old = handle.read()
        if old == text:
            logger.info( "%s is up to date", args.out )
        else:
            with open( args.out, 'w', encoding='utf-8', newline='\n' ) as handle:
                handle.write( text )
            logger.info( "wrote %s (%d settings)", args.out, len( content['schema']['settings'] ) )
        logger.info( "log: %s", args.log_file )
        return 0
    except Exception:
        logger.exception( "could not write the fixture" )
        return 1
    finally:
        bu_log.close_logger( logger )


if __name__ == '__main__':
    sys.exit( main() )
