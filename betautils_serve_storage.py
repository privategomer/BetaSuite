"""
betautils_serve_storage.py - the Storage view's commands.

  storage.usage   every category of the app's files (betautils_storage)
                  with its size, file count and what clearing it costs;
                  free space on each kind of folder's disk; where the
                  cache folder comes from (BETASUITE_CACHE_DIR, the
                  cache_dir setting, or the default)
  storage.clean   remove named categories; costly and resume ones need
                  confirm: true, and what a running job writes is refused
  storage.history what "Clear history" would remove for a time range, by
                  kind (betautils_history), and whether the dialog's
                  "Delete permanently" box starts ticked
  storage.clear_history  clear the named kinds for a range; refused while
                  a job holds the queue; to the trash unless permanent

A cache folder the app used before cache_dir moved it is listed as
previous_cache, so it can be deleted rather than left behind: the engine
remembers the folder it started with in state/previous_cache.json when
it sees the cache has moved. A uv download cache in uv's shared default
place, left by a setup older than the app's own download folder, is
listed as shared and never removed.
"""

import json
import os

import betautils_protocol as bu_proto

PREVIOUS_CACHE_FILE = 'previous_cache.json'
LAST_CACHE_FILE = 'cache_folder.json'


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


def _state_file( name ):
    import betautils_paths as bu_paths
    return os.path.join( bu_paths.kind_dir( 'state' ), name )


def _read_json( path ):
    try:
        with open( path, 'r', encoding='utf-8' ) as handle:
            return json.load( handle )
    except ( OSError, ValueError ):
        return None


def _write_json( path, value ):
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    temp = path + '.tmp'
    with open( temp, 'w', encoding='utf-8' ) as handle:
        json.dump( value, handle )
    os.replace( temp, path )


def note_cache_folder( current=None ):
    """
    Remember the cache folder in use; when it differs from the one last
    seen, keep the old one as the previous cache (if it still exists and
    is not inside the new one). Returns the previous cache or None.
    """
    import betautils_paths as bu_paths
    if current is None and bu_paths.cache_source() == 'variable':
        # BETASUITE_CACHE_DIR is a run's own choice (tests, spikes, one-off
        # runs), never a move of the user's cache: not remembered.
        return previous_cache()
    current = os.path.normpath( current or bu_paths.kind_dir( 'cache' ) )
    last = ( _read_json( _state_file( LAST_CACHE_FILE ) ) or {} ).get( 'folder' )
    if last and os.path.normpath( last ) != current:
        last = os.path.normpath( last )
        inside = current.startswith( last + os.sep ) or last.startswith( current + os.sep )
        if os.path.isdir( last ) and not inside:
            _write_json( _state_file( PREVIOUS_CACHE_FILE ), { 'folder': last } )
    try:
        _write_json( _state_file( LAST_CACHE_FILE ), { 'folder': current } )
    except OSError:
        pass
    return previous_cache()


def previous_cache():
    found = ( _read_json( _state_file( PREVIOUS_CACHE_FILE ) ) or {} ).get( 'folder' )
    return found if found and os.path.isdir( found ) else None


def shared_download_cache( environ=None ):
    """uv's shared default cache, when it exists and is not the app's own folder."""
    environ = os.environ if environ is None else environ
    if os.name == 'nt':
        base = environ.get( 'LOCALAPPDATA', '' )
        folder = os.path.join( base, 'uv', 'cache' ) if base else None
    else:
        base = environ.get( 'XDG_CACHE_HOME' ) or os.path.join( environ.get( 'HOME', '' ), '.cache' )
        folder = os.path.join( base, 'uv' ) if base else None
    return folder if folder and os.path.isdir( folder ) else None


def _categories():
    import betautils_storage as bu_storage
    return bu_storage.categories( previous_cache=previous_cache(), shared_download_cache=shared_download_cache() )


def _busy( server ):
    jobs = getattr( server, 'jobs', None )
    return jobs.busy_job() if jobs is not None else None


def storage_usage( server, _data ):
    import betautils_hardware as bu_hardware
    import betautils_paths as bu_paths
    import betautils_storage as bu_storage
    note_cache_folder()
    return {
        'categories': bu_storage.usage( _categories() ),
        'disks': bu_hardware.disks(),
        'cache_source': bu_paths.cache_source(),
        'busy': _busy( server ) is not None,
    }


def storage_clean( server, data ):
    import betautils_storage as bu_storage
    ids = data.get( 'categories' )
    if not isinstance( ids, list ) or not all( isinstance( item, str ) and item for item in ids ):
        raise _bad_params( 'categories', bu_proto.REASON_MISSING if ids is None else bu_proto.REASON_TYPE )
    if not ids:
        raise _bad_params( 'categories', bu_proto.REASON_VALUE )
    confirm = data.get( 'confirm', False )
    if not isinstance( confirm, bool ):
        raise _bad_params( 'confirm', bu_proto.REASON_TYPE )
    import betaconfig
    permanent = bool( getattr( betaconfig, 'storage_delete_permanently', False ) )
    try:
        freed = bu_storage.clean( _categories(), ids, confirm=confirm, busy=_busy( server ) is not None,
                                  permanent=permanent )
    except bu_storage.StorageError as err:
        raise bu_proto.ProtocolError( err.code, err.params )
    if 'previous_cache' in ids:
        try:
            os.remove( _state_file( PREVIOUS_CACHE_FILE ) )
        except OSError:
            pass
    server._logger.info( "storage: cleaned %s"%( ', '.join( '%s (%d bytes, %d files)'%(
        row['id'], row['bytes'], row['files'] ) for row in freed ), ) )
    return { 'freed': freed, 'trashed': not permanent }


# "Clear history" ranges: how far back, in seconds; 'all' has no limit.
RANGES = { 'hour': 3600, 'day': 86400, 'week': 7 * 86400, 'all': None }


def _since( data, clock ):
    name = data.get( 'range', 'all' )
    if not isinstance( name, str ) or name not in RANGES:
        raise _bad_params( 'range', bu_proto.REASON_TYPE if not isinstance( name, str ) else bu_proto.REASON_VALUE )
    seconds = RANGES[name]
    return None if seconds is None else clock() - seconds


def _store( server ):
    import betautils_serve_catalog as bu_serve_catalog
    return bu_serve_catalog.catalog_for( server ).store


def _clock( server ):
    import time
    return getattr( server, 'clock', None ) or time.time


def storage_history( server, data ):
    import betaconfig
    import betautils_history as bu_history
    since = _since( data, _clock( server ) )
    return {
        'kinds': bu_history.summary( bu_history.kinds(), since=since, store=_store( server ) ),
        'busy': _busy( server ) is not None,
        'permanent': bool( getattr( betaconfig, 'storage_delete_permanently', False ) ),
    }


def storage_clear_history( server, data ):
    import betaconfig
    import betautils_history as bu_history
    import betautils_trash as bu_trash
    ids = data.get( 'kinds' )
    if not isinstance( ids, list ) or not all( isinstance( item, str ) and item for item in ids ):
        raise _bad_params( 'kinds', bu_proto.REASON_MISSING if ids is None else bu_proto.REASON_TYPE )
    if not ids:
        raise _bad_params( 'kinds', bu_proto.REASON_VALUE )
    since = _since( data, _clock( server ) )
    confirm = data.get( 'confirm', False )
    if not isinstance( confirm, bool ):
        raise _bad_params( 'confirm', bu_proto.REASON_TYPE )
    permanent = data.get( 'permanent', bool( getattr( betaconfig, 'storage_delete_permanently', False ) ) )
    if not isinstance( permanent, bool ):
        raise _bad_params( 'permanent', bu_proto.REASON_TYPE )
    try:
        cleared = bu_history.clear( bu_history.kinds(), ids, since=since, confirm=confirm,
                                    busy=_busy( server ) is not None, permanent=permanent,
                                    store=_store( server ) )
    except bu_history.HistoryError as err:
        raise bu_proto.ProtocolError( err.code, err.params )
    except bu_trash.TrashError as err:
        server._logger.warning( "storage: clear history stopped, the trash refused a file (%s)"%( err.reason, ) )
        raise bu_proto.ProtocolError( 'storage.trash_failed', {} )
    if 'run_history' in ids or 'library' in ids:
        jobs = getattr( server, 'jobs', None )
        if jobs is not None:
            jobs.forget_ended()
    server._logger.info( "storage: cleared history (%s, %s): %s"%(
        data.get( 'range', 'all' ), 'deleted' if permanent else 'to the trash',
        ', '.join( '%s (%d bytes, %d files, %d rows)'%( row['id'], row['bytes'], row['files'], row['rows'] )
                   for row in cleared ) ) )
    return { 'cleared': cleared, 'trashed': not permanent }


HANDLERS = {
    'storage.usage': storage_usage,
    'storage.clean': storage_clean,
    'storage.history': storage_history,
    'storage.clear_history': storage_clear_history,
}
