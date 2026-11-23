"""
Shared helpers for tests that change betaconfig deeply and must put it
back exactly: a snapshot of every setting, path-based get and set, and
the three cache keys computed without writing run-key records.
"""

import contextlib
import copy
import os

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_hash as bu_hash
import betautils_local as bu_local


def setting_names():
    """Every setting betaconfig defines, in file order."""
    return bu_local.known_names( vars( betaconfig ) )


def snapshot():
    """A deep copy of every setting, to hand back to restore()."""
    return { name: copy.deepcopy( getattr( betaconfig, name ) ) for name in setting_names() }


def restore( saved ):
    """Put every setting back as snapshot() found it, dropping any added since."""
    for name in setting_names():
        if name not in saved:
            delattr( betaconfig, name )
    for name, value in saved.items():
        setattr( betaconfig, name, copy.deepcopy( value ) )
    bu_config.invalidate_config_caches()


@contextlib.contextmanager
def preserved():
    """Run a block that may change any setting; restore them all afterwards."""
    saved = snapshot()
    saved_loaded = bu_local.LOADED   # what the local layer last applied; apply() replaces it
    try:
        yield
    finally:
        restore( saved )
        bu_local.LOADED = saved_loaded


_SHIPPED = None


def shipped_values():
    """
    betaconfig.py's own values, read fresh from the file, with the
    suite's adjustments (tests/__init__.py) on top. Other tests change
    betaconfig and some do not put everything back, so a test that
    compares against a recorded baseline starts from this instead.
    """
    global _SHIPPED
    if _SHIPPED is None:
        import importlib.util
        import tests
        path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                             'betaconfig.py' )
        spec = importlib.util.spec_from_file_location( '_betaconfig_shipped', path )
        module = importlib.util.module_from_spec( spec )
        with environment( BETASUITE_LOCAL_CONFIG='none' ):
            spec.loader.exec_module( module )
        values = { name: getattr( module, name ) for name in bu_local.known_names( vars( module ) ) }
        values.update( tests.TEST_ENCODE )
        values['items_to_censor'] = list( tests.SUITE_ITEMS_TO_CENSOR )
        values['log_path'] = tests.TEST_LOG_PATH
        values['stats_path'] = tests.TEST_STATS_PATH
        _SHIPPED = values
    return copy.deepcopy( _SHIPPED )


@contextlib.contextmanager
def shipped():
    """Run a block against the shipped config, whatever earlier tests left; restore afterwards."""
    with preserved():
        with environment( BETASUITE_DETECTOR_BACKEND_OVERRIDE=None,
                          BETASUITE_DETECTOR_VARIANT_OVERRIDE=None ):
            values = shipped_values()
            for name in setting_names():
                if name not in values:
                    delattr( betaconfig, name )
            for name, value in values.items():
                setattr( betaconfig, name, value )
            bu_config.invalidate_config_caches()
            yield


def get_path( path ):
    """The value at a setting path (a tuple of names and list indices)."""
    value = getattr( betaconfig, path[0] )
    for part in path[1:]:
        value = value[part]
    return value


def set_path( path, value ):
    """Set the value at a setting path, creating dicts on the way down."""
    if len( path ) == 1:
        setattr( betaconfig, path[0], value )
    else:
        node = getattr( betaconfig, path[0] )
        for part in path[1:-1]:
            if isinstance( node, dict ) and part not in node:
                node[part] = {}
            node = node[part]
        node[path[-1]] = value
    bu_config.invalidate_config_caches()


def leaf_paths( value, path ):
    """
    Every leaf under a setting: dict keys and list-of-dict indices are
    walked, anything else (numbers, strings, None, tuples, plain lists)
    is a leaf.
    """
    if isinstance( value, dict ) and value:
        for key in value:
            yield from leaf_paths( value[key], path + ( key, ) )
    elif isinstance( value, list ) and value and all( isinstance( item, dict ) for item in value ):
        for index, item in enumerate( value ):
            yield from leaf_paths( item, path + ( index, ) )
    else:
        yield path


def shipped_leaf_paths():
    """Every leaf of every setting, in file order."""
    paths = []
    for name in setting_names():
        paths.extend( leaf_paths( getattr( betaconfig, name ), ( name, ) ) )
    return paths


def keys( backend_name=None ):
    """
    { 'detection', 'censor', 'encode' } keys for the current config,
    hashed from the identities directly so no run-key file is written.
    """
    bu_config.invalidate_config_caches()
    return {
        'detection': bu_hash.dictionary_hash(
            bu_cache.detection_identity( backend_name ), betaconst.detection_key_len ),
        'censor': bu_hash.dictionary_hash(
            bu_cache.censor_identity( backend_name ), betaconst.censor_key_len ),
        'encode': bu_hash.dictionary_hash(
            bu_cache.encode_identity(), betaconst.encode_key_len ),
    }


@contextlib.contextmanager
def environment( **values ):
    """Set environment variables for a block (None removes one)."""
    saved = { name: os.environ.get( name ) for name in values }
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop( name, None )
            else:
                os.environ[name] = value
        bu_config.invalidate_config_caches()
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop( name, None )
            else:
                os.environ[name] = value
        bu_config.invalidate_config_caches()
