"""
betautils_models.py - the models the app can install, and which one to use.

The catalog (assets/models/catalog.json) is data: one row per model file
with its backend and variant, its pinned SHA-256 and size, the mirrors it
is hosted on, its license and the upstream source files mirrored beside
it. A new model is a new row, plus its name in the message catalog.

Files keep the names the detector and segmenter adapters already use,
under the model folder ('../resources/model', wherever the layout puts
it), so a config written before the catalog existed still finds them.

Stdlib only at import time.
"""

import json
import os

import betautils_paths as bu_paths

REPO_ROOT = os.path.dirname( os.path.abspath( __file__ ) )
CATALOG_PATH = os.path.join( REPO_ROOT, 'assets', 'models', 'catalog.json' )
CATALOG_FORMAT = 1
MODEL_DIR_CONFIG = '../resources/model'
KINDS = ( 'detector', 'person' )

# Detector picked by hardware: 640m finds more but is several times
# slower than 320n, which only pays off on a GPU.
GPU_DETECTOR = 'nudenet-640m'
CPU_DETECTOR = 'nudenet-320n'

_catalog = None


class CatalogError( Exception ):
    """The catalog file is malformed. A packaging bug, never a user error."""


def model_dir():
    """The folder model files live in."""
    return bu_paths.legacy_path( 'resources', 'model' )


def config_path_for( entry ):
    """The '../resources/model/<file>' string adapters and configs use."""
    return '%s/%s'%( MODEL_DIR_CONFIG, entry['file'] )


def install_path( entry ):
    return os.path.join( model_dir(), entry['file'] )


def _check( condition, why ):
    if not condition:
        raise CatalogError( why )


def validate( catalog ):
    """Raise CatalogError for anything a consumer would trip on later."""
    _check( isinstance( catalog, dict ) and catalog.get( 'format' ) == CATALOG_FORMAT,
            'unknown catalog format' )
    mirrors = catalog.get( 'mirrors' )
    _check( isinstance( mirrors, list ) and mirrors and all(
        isinstance( m, str ) and m.startswith( 'https://' ) and m.endswith( '/' ) for m in mirrors ),
        'mirrors must be https URLs ending in /' )
    licenses = catalog.get( 'licenses' )
    _check( isinstance( licenses, dict ) and licenses, 'no licenses' )
    ids, files = set(), set()
    for entry in catalog.get( 'models', () ):
        mid = entry.get( 'id' )
        _check( isinstance( mid, str ) and mid and mid not in ids, 'missing or repeated id %r'%( mid, ) )
        ids.add( mid )
        _check( entry.get( 'kind' ) in KINDS, '%s: bad kind'%( mid, ) )
        _check( isinstance( entry.get( 'backend' ), str ), '%s: no backend'%( mid, ) )
        name = entry.get( 'file' )
        _check( isinstance( name, str ) and name and os.path.basename( name ) == name
                and name not in files, '%s: bad or repeated file name'%( mid, ) )
        files.add( name )
        _check( _is_sha256( entry.get( 'sha256' ) ), '%s: bad sha256'%( mid, ) )
        _check( isinstance( entry.get( 'size' ), int ) and entry['size'] > 0, '%s: bad size'%( mid, ) )
        _check( entry.get( 'license' ) in licenses, '%s: unknown license'%( mid, ) )
        for source in entry.get( 'sources', () ):
            sname = source.get( 'file' )
            _check( isinstance( sname, str ) and sname and os.path.basename( sname ) == sname
                    and sname not in files, '%s: bad or repeated source file name'%( mid, ) )
            files.add( sname )
            _check( _is_sha256( source.get( 'sha256' ) ), '%s: bad source sha256'%( mid, ) )
            _check( str( source.get( 'upstream', '' ) ).startswith( 'https://' ),
                    '%s: source without an upstream URL'%( mid, ) )
    for key, lic in licenses.items():
        _check( isinstance( lic.get( 'file' ), str ) and lic['file'] not in files,
                'license %s: bad or repeated file name'%( key, ) )
        files.add( lic['file'] )
    _check( ids, 'no models' )
    return catalog


def _is_sha256( value ):
    return isinstance( value, str ) and len( value ) == 64 and all( c in '0123456789abcdef' for c in value )


def load( path=None ):
    """The validated catalog. Cached for the shipped file."""
    global _catalog
    if path is None and _catalog is not None:
        return _catalog
    with open( path or CATALOG_PATH, 'r', encoding='UTF-8' ) as fin:
        catalog = validate( json.load( fin ) )
    if path is None:
        _catalog = catalog
    return catalog


def entries( catalog=None, kind=None ):
    catalog = catalog or load()
    return [ entry for entry in catalog['models'] if kind is None or entry['kind'] == kind ]


def entry( model_id, catalog=None ):
    """The row for an id, or None."""
    for row in entries( catalog ):
        if row['id'] == model_id:
            return row
    return None


def entry_for( backend, variant, catalog=None ):
    """The row for a backend and variant (variant None for single-model backends), or None."""
    for row in entries( catalog ):
        if row['backend'] == backend and row.get( 'variant' ) == variant:
            return row
    return None


def entry_for_path( path, catalog=None ):
    """The row whose installed file a config path points at, or None."""
    if not path:
        return None
    target = os.path.normcase( os.path.abspath( bu_paths.resolve( path ) ) )
    for row in entries( catalog ):
        if os.path.normcase( os.path.abspath( install_path( row ) ) ) == target:
            return row
    return None


def urls( row, catalog=None ):
    """Where a file can be downloaded from, in the order to try."""
    catalog = catalog or load()
    return [ mirror + row['file'] for mirror in catalog['mirrors'] ] + list( row.get( 'urls', () ) )


def license_for( row, catalog=None ):
    return ( catalog or load() )['licenses'][ row['license'] ]


# ---------------------------------------------------------------------------
# Hardware default
# ---------------------------------------------------------------------------

def recommended_detector( hardware ):
    """
    ( model id, reason code ) for this machine: 640m when a usable NVIDIA
    GPU is found, 320n otherwise. hardware is betautils_hardware.detect().
    """
    if hardware and hardware.get( 'cuda_usable' ):
        return GPU_DETECTOR, 'gpu'
    return CPU_DETECTOR, 'cpu'


# ---------------------------------------------------------------------------
# What the current config uses
# ---------------------------------------------------------------------------

def detector_path_in_use():
    """The model path the selected detector backend would load, or None."""
    import betautils_detector as bu_detector
    try:
        module = bu_detector.get_detector()
    except ValueError:
        return None
    if hasattr( module, 'resolve_config' ):
        try:
            return module.resolve_config().get( 'model_path' )
        except ValueError:
            return None
    return getattr( module, 'MODEL_PATH', None )


def person_path_in_use():
    """The model path the selected person backend would load, or None."""
    import betautils_person as bu_person
    try:
        return bu_person.resolve_config().get( 'model_path' )
    except ValueError:
        return None


def in_use( catalog=None ):
    """
    { 'detector': id or None, 'person': id or None, 'custom': {kind: path} }.
    A path outside the catalog (an advanced model_path) shows as custom.
    """
    result = { 'detector': None, 'person': None, 'custom': {} }
    for kind, path in ( ( 'detector', detector_path_in_use() ), ( 'person', person_path_in_use() ) ):
        row = entry_for_path( path, catalog )
        if row is not None:
            result[kind] = row['id']
        elif path:
            result['custom'][kind] = path
    return result
