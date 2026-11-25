"""
betautils_serve_config.py - the engine's config.* commands.

    config.schema   the settings registry, for building settings pages
    config.get      current values, their defaults and where each came from
    config.set      check a change as betatv.py would; write it to the
                    local config only if it passes (or just report, dry_run)
    config.reset    remove settings from the local config

Every change goes through betautils_config_store, the same path
tools/setup/local_config.py uses, so a value set here is what the CLI
reads on its next run. Errors are codes with params, never sentences
(protocol/README.md).

betaconfig and the modules that read it are imported on first use, not
when serve starts, so hello stays fast and nothing they print at import
can reach stdout before serve has taken it over.
"""

import betautils_protocol as bu_proto

_STORE = None


def store():
    """The process's one ConfigStore, made on first use."""
    global _STORE
    if _STORE is None:
        import betautils_config_store as bu_store
        _STORE = bu_store.ConfigStore()
    return _STORE


def reset_store_for_tests():
    global _STORE
    _STORE = None


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


def _jsonable( value ):
    import betautils_settings as bu_settings
    return bu_settings._jsonable( value )


def _path( text, field ):
    if not isinstance( text, str ) or not text:
        raise _bad_params( field, bu_proto.REASON_TYPE )
    parts = tuple( text.split( '.' ) )
    if any( not part for part in parts ):
        raise _bad_params( field, bu_proto.REASON_VALUE )
    return parts


def _settable( path ):
    """The registry entry a client may set at this path, or a ProtocolError."""
    import betautils_settings as bu_settings
    entry = bu_settings.find( path )
    dotted = '.'.join( path )
    if entry is None:
        raise bu_proto.ProtocolError( 'config.unknown_key',
                                      { 'path': dotted, 'suggestion': bu_settings.suggestion( path ) } )
    if len( entry.path ) != len( path ):
        # Inside a value that is set whole (a censor style, a rule list).
        whole = '.'.join( path[:len( entry.path )] )
        raise bu_proto.ProtocolError( 'config.not_settable', { 'path': dotted, 'setting': whole } )
    return entry


def _overrides( data ):
    overrides = data.get( 'overrides', {} )
    if not isinstance( overrides, dict ):
        raise _bad_params( 'overrides', bu_proto.REASON_TYPE )
    out = {}
    for dotted, value in overrides.items():
        path = _path( dotted, 'overrides' )
        _settable( path )
        out[path] = value
    return out


def _invalid( errors ):
    return bu_proto.ProtocolError( 'config.invalid',
                                   { 'problems': [ _problem( error ) for error in errors ] } )


# The rule id for a check that returned plain text. tests/test_settings_validation.py
# proves no shipped check does; the id is in errors.schema.json all the same.
UNCLASSIFIED_RULE = 'config.unclassified'


def _problem( error ):
    import betautils_settings as bu_settings
    if isinstance( error, bu_settings.Problem ):
        return bu_proto.error( error.code, **_jsonable( error.params ) )
    # A check that has not been given a code yet: still never a sentence.
    return bu_proto.error( 'config.rule', rule=UNCLASSIFIED_RULE, path='' )


def _local_on():
    if not store().path:
        raise bu_proto.ProtocolError( 'config.local_off', {} )


# --- commands -----------------------------------------------------------------

def config_schema( _server, _data ):
    import betautils_settings as bu_settings
    return bu_settings.schema()


def config_get( _server, data ):
    """
    data: { paths?: [dotted path prefix], overrides?: { dotted path: value } }
    result: { settings: [ { path, value, default?, source, changed } ] }

    overrides are per-run values, shown with source 'run' and never
    written. An override that would fail validation is config.invalid.
    """
    import betautils_config_store as bu_store
    import betautils_settings as bu_settings
    prefixes = data.get( 'paths' )
    if prefixes is not None and ( not isinstance( prefixes, list )
                                  or not all( isinstance( prefix, str ) for prefix in prefixes ) ):
        raise _bad_params( 'paths', bu_proto.REASON_TYPE )
    overrides = _overrides( data )
    current = store()
    evaluation = current.evaluate( current.text(), overrides )
    if overrides and not evaluation.ok:
        raise _invalid( evaluation.errors )
    sources = { tuple( path ): 'local' for path in evaluation.loaded.applied }
    sources.update( { path: 'run' for path in overrides } )
    settings = []
    for path, value, entry in bu_settings.concrete_leaves( evaluation.values ):
        if entry is None or len( entry.path ) != len( path ):
            continue
        dotted = '.'.join( str( part ) for part in path )
        if prefixes and not any( dotted == prefix or dotted.startswith( prefix + '.' )
                                 for prefix in prefixes ):
            continue
        row = { 'path': dotted, 'value': _jsonable( value ),
                'source': bu_store.source_of( path, sources ) }
        default = bu_store._get( current.base, path )
        if default is not bu_store.MISSING:
            row['default'] = _jsonable( default )
        row['changed'] = default is bu_store.MISSING or default != value
        settings.append( row )
    return { 'settings': settings }


def _edit( data, updates, removals ):
    import betautils_settings as bu_settings
    _local_on()
    dry_run = data.get( 'dry_run', False )
    if not isinstance( dry_run, bool ):
        raise _bad_params( 'dry_run', bu_proto.REASON_TYPE )
    try:
        evaluation, moves, written = store().edit( updates, removals, dry_run=dry_run )
    except ValueError:
        raise bu_proto.ProtocolError( 'config.local_unparsable', {} )
    except OSError as err:
        import errno
        raise bu_proto.ProtocolError( 'config.local_unwritable',
                                      { 'reason': errno.errorcode.get( err.errno, 'unknown' ) } )
    if not evaluation.ok:
        raise _invalid( evaluation.errors )
    # Keys the change can move for some files only (a file's profile or
    # preview slice), on top of the config's own, for each setting whose
    # value actually changes.
    import betautils_config_store as bu_store
    may_move = set()
    for path in list( updates ) + list( removals ):
        if bu_store._get( evaluation.before_values, path ) != bu_store._get( evaluation.values, path ):
            may_move |= bu_settings.find( path ).may_move
    return { 'moves': moves, 'may_move': [ kind for kind in bu_settings.CACHE_KEYS
                                           if kind in may_move and kind not in moves ],
             'written': written, 'dry_run': dry_run }


def config_set( _server, data ):
    """
    data: { values: { dotted path: value }, dry_run?: bool }
    result: { moves: [ cache key ], may_move: [ cache key ], written: bool, dry_run: bool }

    moves names the cache keys the change moves, worked out from the
    keys themselves: what the change costs (re-detect, re-render,
    re-encode) before it is made. may_move names keys it can move for
    some files only, by changing their structure profile or preview
    slice (betautils_settings.MAY_MOVE).
    """
    values = data.get( 'values' )
    if not isinstance( values, dict ):
        raise _bad_params( 'values', bu_proto.REASON_MISSING if values is None else bu_proto.REASON_TYPE )
    if not values:
        raise _bad_params( 'values', bu_proto.REASON_VALUE )
    updates = {}
    for dotted, value in values.items():
        path = _path( dotted, 'values' )
        _settable( path )
        updates[path] = value
    return _edit( data, updates, [] )


def config_reset( _server, data ):
    """
    data: { paths: [ dotted path ], dry_run?: bool }
    result: as config.set. A path the local config does not set changes nothing.
    """
    paths = data.get( 'paths' )
    if not isinstance( paths, list ):
        raise _bad_params( 'paths', bu_proto.REASON_MISSING if paths is None else bu_proto.REASON_TYPE )
    if not paths:
        raise _bad_params( 'paths', bu_proto.REASON_VALUE )
    removals = []
    for dotted in paths:
        path = _path( dotted, 'paths' )
        _settable( path )
        removals.append( path )
    return _edit( data, {}, removals )


HANDLERS = {
    'config.schema': config_schema,
    'config.get': config_get,
    'config.set': config_set,
    'config.reset': config_reset,
}
