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

import http.client
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


MISSING_FILE = 'missing'


def file_identity( config_path ):
    """
    The SHA-256 a model's detections and masks are keyed on.

    The file's own hash when it is there (memoised, so only the first
    look reads it), so a different file at the same path never reuses
    old detections. When it is not there yet, the catalog's pinned hash
    for that path, which the installed file will have; so keys can be
    worked out, and stay the same, before a model is installed. A path
    outside the catalog with no file keys as 'missing'.
    """
    path = bu_paths.resolve( config_path ) if config_path else None
    if path and os.path.isfile( path ):
        try:
            import betautils_hash as bu_hash
            return bu_hash.sha256_for_file( path )
        except OSError:
            pass
    row = entry_for_path( config_path )
    return row['sha256'] if row is not None else MISSING_FILE


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


# ---------------------------------------------------------------------------
# Install, import, remove
# ---------------------------------------------------------------------------

PART_SUFFIX = '.part'
CHUNK_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 30
USER_AGENT = 'BetaSuite-models'
# Room kept free beyond the file itself, so an install never fills a disk.
FREE_MARGIN_BYTES = 64 * 1024 * 1024

# model.download_failed reasons
REASON_NETWORK = 'network'
REASON_HTTP = 'http'
REASON_NOT_A_MODEL = 'not_a_model'


class ModelError( Exception ):
    """A refusal with a protocol error code and its params."""

    def __init__( self, code, **params ):
        super().__init__( code, params )
        self.code = code
        self.params = params


class Stopped( Exception ):
    """The caller asked the install to stop. The partial file is kept for a resume."""


def part_path( row ):
    return install_path( row ) + PART_SUFFIX


def _sha256( path, should_stop=None, use_memo=True ):
    import betautils_hash as bu_hash
    try:
        return bu_hash.sha256_for_file( path, use_memo=use_memo, should_stop=should_stop )
    except bu_hash.HashStopped:
        raise Stopped()


def _flush_memo():
    import betautils_hash as bu_hash
    bu_hash.flush_file_hash_memo()


def _size( path ):
    try:
        return os.path.getsize( path )
    except OSError:
        return None


def _discard( path ):
    try:
        os.remove( path )
    except FileNotFoundError:
        pass


def status( row, should_stop=None ):
    """
    { 'state': 'installed' | 'modified' | 'partial' | 'absent', 'bytes': n }.
    modified: a file is at the install path but it is not the pinned one
    (the user swapped it), so the page never calls it installed.
    """
    path = install_path( row )
    size = _size( path )
    if size is not None:
        if size == row['size'] and _sha256( path, should_stop ) == row['sha256']:
            return { 'state': 'installed', 'bytes': size }
        return { 'state': 'modified', 'bytes': size }
    partial = _size( part_path( row ) )
    if partial:
        return { 'state': 'partial', 'bytes': partial }
    return { 'state': 'absent', 'bytes': 0 }


def _check_space( folder, needed ):
    import shutil
    free = shutil.disk_usage( folder ).free
    if free < needed + FREE_MARGIN_BYTES:
        raise ModelError( 'disk.full', free_bytes=free )


def _looks_like_html( chunk ):
    return chunk.lstrip()[:1] == b'<'


def _range_start( response ):
    """The first byte a 206 answer starts at, or None."""
    headers = getattr( response, 'headers', None )
    value = headers.get( 'Content-Range', '' ) if headers is not None else ''
    try:
        unit, rest = value.split( ' ', 1 )
        return int( rest.split( '-', 1 )[0] ) if unit.strip() == 'bytes' else None
    except ValueError:
        return None


def _fetch( url, part, row, progress, should_stop, opener ):
    """
    Bring part up to the full file from url, resuming from its current
    length. Raises Stopped, ModelError, or an OSError for the network.
    """
    import urllib.request
    offset = _size( part ) or 0
    if offset > row['size']:
        os.remove( part )
        offset = 0
    if offset == row['size']:
        return
    headers = { 'User-Agent': USER_AGENT }
    if offset:
        headers['Range'] = 'bytes=%d-'%( offset, )
    request = urllib.request.Request( url, headers=headers )
    with opener( request, timeout=TIMEOUT_SECONDS ) as response:
        code = getattr( response, 'status', None ) or response.getcode()
        if offset and code == 206 and _range_start( response ) == offset:
            mode = 'ab'
        elif code == 200:
            # A server that ignores Range sends the whole file: start over.
            offset, mode = 0, 'wb'
        else:
            if code == 206:
                # A range other than the one asked for: the part cannot
                # be trusted to line up, so the next attempt starts clean.
                _discard( part )
            raise ModelError( 'model.download_failed', model=row['id'], reason=REASON_HTTP )
        if progress:
            progress( offset, row['size'] )
        # read1 returns what has arrived, so progress and the part file
        # move with the network instead of in whole chunks.
        read = getattr( response, 'read1', None ) or response.read
        with open( part, mode ) as out:
            first = mode == 'wb'
            while True:
                if should_stop is not None and should_stop():
                    raise Stopped()
                chunk = read( CHUNK_BYTES )
                if not chunk:
                    break
                if first:
                    first = False
                    if _looks_like_html( chunk ):
                        # A sign-in or warning page saved under a model's name.
                        raise ModelError( 'model.download_failed', model=row['id'], reason=REASON_NOT_A_MODEL )
                offset += len( chunk )
                if offset > row['size']:
                    raise ModelError( 'model.hash_mismatch', model=row['id'] )
                out.write( chunk )
                out.flush()
                if progress:
                    progress( offset, row['size'] )
            os.fsync( out.fileno() )
    if offset < row['size']:
        # The body ended early: a dropped connection, not a wrong file.
        # Keep what arrived so the next attempt resumes.
        raise ConnectionError( 'body ended at %d of %d bytes'%( offset, row['size'] ) )


def _place( part, row ):
    """Verify part against the pinned hash and move it into place, or delete it."""
    if _sha256( part, use_memo=False ) != row['sha256']:
        os.remove( part )
        raise ModelError( 'model.hash_mismatch', model=row['id'] )
    os.replace( part, install_path( row ) )
    _sha256( install_path( row ) )
    _flush_memo()


def install( model_id, progress=None, should_stop=None, opener=None, catalog=None ):
    """
    Download a catalog model into place, verified against its pinned hash.

    Resumes from a .part file left by an earlier stop, kill or lost
    connection, tries each URL in turn, and only renames a file into
    place once its SHA-256 matches. A wrong file is deleted, never kept.

    Returns 'installed' or 'already'. Raises ModelError or Stopped.
    """
    import urllib.request
    row = entry( model_id, catalog )
    if row is None:
        raise ModelError( 'model.unknown', model=model_id )
    if status( row )['state'] == 'installed':
        return 'already'
    folder = model_dir()
    os.makedirs( folder, exist_ok=True )
    part = part_path( row )
    _check_space( folder, row['size'] - ( _size( part ) or 0 ) )
    opener = opener or urllib.request.urlopen
    failure = None
    for url in urls( row, catalog ):
        try:
            _fetch( url, part, row, progress, should_stop, opener )
        except ModelError as err:
            if err.code == 'model.hash_mismatch':
                _discard( part )
                raise
            if err.params.get( 'reason' ) == REASON_NOT_A_MODEL:
                _discard( part )
            failure = err
            continue
        except ( OSError, http.client.HTTPException ):
            # Lost network, refused connection, HTTP error status, a body
            # cut short: the part file stays, and the next URL or install
            # resumes it.
            failure = ModelError( 'model.download_failed', model=row['id'], reason=REASON_NETWORK )
            continue
        _place( part, row )
        return 'installed'
    raise failure or ModelError( 'model.download_failed', model=row['id'], reason=REASON_NETWORK )


def import_file( path, should_stop=None, catalog=None ):
    """
    "I already have the file": install a file the user picked, when its
    SHA-256 is a catalog model's. Copied, never moved, through a temporary
    file, so the user's copy is untouched and a half copy is never used.

    Returns the model id. Raises ModelError('model.unknown_file').
    """
    import shutil
    digest = _sha256( path, should_stop, use_memo=False )
    row = next( ( r for r in entries( catalog ) if r['sha256'] == digest ), None )
    if row is None:
        raise ModelError( 'model.unknown_file' )
    if status( row )['state'] == 'installed':
        return row['id']
    folder = model_dir()
    os.makedirs( folder, exist_ok=True )
    _check_space( folder, row['size'] )
    temp = install_path( row ) + '.import'
    try:
        shutil.copyfile( path, temp )
        if _sha256( temp, use_memo=False ) != row['sha256']:
            raise ModelError( 'model.hash_mismatch', model=row['id'] )
        os.replace( temp, install_path( row ) )
    finally:
        _discard( temp )
    _discard( part_path( row ) )
    _sha256( install_path( row ) )
    _flush_memo()
    return row['id']


def remove( model_id, catalog=None ):
    """
    Delete an installed model and any partial download. Refuses the model
    the current config uses (model.in_use): the user switches first.
    """
    row = entry( model_id, catalog )
    if row is None:
        raise ModelError( 'model.unknown', model=model_id )
    if in_use( catalog ).get( row['kind'] ) == model_id:
        raise ModelError( 'model.in_use', model=model_id )
    _discard( install_path( row ) )
    _discard( part_path( row ) )
