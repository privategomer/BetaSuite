"""
protocol_schemas.py - test support: the protocol/v1 schemas, loaded and
ready to validate messages against.

jsonschema is a developer dependency (requirements-dev.txt); the engine
itself never imports it. A checkout without it skips the schema tests,
except under CI, where a missing package must fail rather than skip.
"""

import json
import os
import unittest

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
SCHEMA_DIR = os.path.join( REPO, 'protocol', 'v1' )
BASE_ID = 'https://betasuite.invalid/protocol/v1/'

try:
    import jsonschema
    import referencing
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

needs_jsonschema = unittest.skipUnless(
    HAVE_JSONSCHEMA or os.environ.get( 'CI' ),
    'jsonschema is not installed (pip install -r requirements-dev.txt)' )


def schema_files():
    """{ path relative to protocol/v1 with '/' separators: parsed document }"""
    found = {}
    for root, _dirs, names in os.walk( SCHEMA_DIR ):
        for name in names:
            if name.endswith( '.json' ):
                path = os.path.join( root, name )
                relative = os.path.relpath( path, SCHEMA_DIR ).replace( os.sep, '/' )
                with open( path, encoding='utf-8' ) as handle:
                    found[relative] = json.load( handle )
    return found


def command_schema_names():
    """The command names that have a file under commands/."""
    return sorted( relative[len( 'commands/' ):-len( '.schema.json' )]
                   for relative in schema_files()
                   if relative.startswith( 'commands/' ) and relative.endswith( '.schema.json' ) )


def event_schema_names():
    """The event names that have a file under events/."""
    return sorted( relative[len( 'events/' ):-len( '.schema.json' )]
                   for relative in schema_files()
                   if relative.startswith( 'events/' ) and relative.endswith( '.schema.json' ) )


_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        resources = [ ( doc['$id'], referencing.Resource.from_contents( doc ) )
                      for doc in schema_files().values() ]
        _registry = referencing.Registry().with_resources( resources )
    return _registry


def validator_for( uri ):
    """A validator for the schema at uri (relative to BASE_ID, fragment allowed)."""
    return jsonschema.Draft202012Validator( { '$ref': BASE_ID + uri }, registry=_get_registry() )


def validate_message( message ):
    """Raise jsonschema.ValidationError unless message is a valid envelope."""
    validator_for( 'envelope.schema.json' ).validate( message )


def validate_command_data( name, data ):
    validator_for( 'commands/%s.schema.json#/$defs/data'%( name, ) ).validate( data )


def validate_command_result( name, data ):
    validator_for( 'commands/%s.schema.json#/$defs/result'%( name, ) ).validate( data )


def validate_event_data( name, data ):
    validator_for( 'events/%s.schema.json#/$defs/data'%( name, ) ).validate( data )
