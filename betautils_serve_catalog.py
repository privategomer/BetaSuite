"""
betautils_serve_catalog.py - the catalog.* commands and the
catalog.changed event (protocol/README.md, "Catalog").

The catalog opens on the first catalog command, so `hello` stays fast
and a session that never asks for the library never opens it; it closes
when the server stops (Server.stop_catalog). A catalog that cannot be
opened answers catalog.unavailable or catalog.too_new, and the next
command tries again.

catalog.changed is coalesced but never dropped: changes that arrive
within CHANGE_INTERVAL of the last event are merged into the next one,
which lists every item touched (or says `all` past CHANGED_ITEMS_MAX).
"""

import threading
import time

import betautils_protocol as bu_proto

CHANGE_INTERVAL = 0.25
REASONS = ( 'scan', 'job', 'pin', 'external' )


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


def _protocol_error( err ):
    return bu_proto.ProtocolError( err.code, dict( err.params ) )


class ChangeEvents:
    """Merges catalog changes into catalog.changed events, a few a second."""

    def __init__( self, send_event, items_max, interval=CHANGE_INTERVAL, clock=time.monotonic ):
        self._send = send_event
        self._items_max = items_max
        self._interval = interval
        self._clock = clock
        self._cond = threading.Condition()
        self._items = set()
        self._all = False
        self._reasons = set()
        self._last_sent = None
        self._stopped = False
        self._thread = threading.Thread( target=self._run, name='catalog-events', daemon=True )
        self._thread.start()

    def __call__( self, items, reason ):
        with self._cond:
            if items is None:
                self._all = True
            else:
                self._items |= set( items )
            self._reasons.add( reason )
            self._cond.notify()

    def _pending_locked( self ):
        return self._all or bool( self._items )

    def _take_locked( self ):
        data = { 'reasons': sorted( self._reasons ) }
        if self._all or len( self._items ) > self._items_max:
            data['all'] = True
        else:
            data['items'] = sorted( self._items )
        self._items, self._all, self._reasons = set(), False, set()
        self._last_sent = self._clock()
        return data

    def flush( self ):
        """Send what is pending now (tests, and before the engine stops)."""
        with self._cond:
            data = self._take_locked() if self._pending_locked() else None
        if data is not None:
            self._send( 'catalog.changed', data )

    def _run( self ):
        while True:
            with self._cond:
                while not self._stopped and not self._pending_locked():
                    self._cond.wait()
                if self._stopped:
                    return
                wait = ( 0.0 if self._last_sent is None
                         else self._last_sent + self._interval - self._clock() )
                if wait > 0:
                    self._cond.wait( wait )
                    continue
                data = self._take_locked()
            self._send( 'catalog.changed', data )

    def stop( self ):
        with self._cond:
            self._stopped = True
            self._cond.notify()
        self._thread.join( 5.0 )
        self.flush()


class ServedCatalog:
    """The server's catalog and its event stream."""

    def __init__( self, server, catalog_factory=None ):
        import betautils_catalog as bu_catalog
        self._bu_catalog = bu_catalog
        self.events = ChangeEvents( server.send_event, bu_catalog.CHANGED_ITEMS_MAX )
        factory = catalog_factory or bu_catalog.Catalog

        def busy():
            # A job of this engine, or a command-line run on the library.
            jobs = getattr( server, 'jobs', None )
            if jobs is not None and jobs.busy:
                return True
            import betautils_runlock as bu_runlock
            return bu_runlock.library_held()

        try:
            self.catalog = factory( logger=server._logger, on_change=self.events, busy=busy,
                                    interval=_scan_interval() )
        except BaseException:
            self.events.stop()
            raise
        self.catalog.start()

    def close( self ):
        try:
            self.catalog.close()
        finally:
            self.events.stop()


def _scan_interval():
    import betaconfig
    import betautils_catalog_scan as bu_scan
    value = getattr( betaconfig, 'catalog_scan_interval_seconds', bu_scan.DEFAULT_INTERVAL_SECONDS )
    return value if isinstance( value, ( int, float ) ) and not isinstance( value, bool ) and value >= 0 \
        else bu_scan.DEFAULT_INTERVAL_SECONDS


def catalog_for( server ):
    """The server's catalog, opened on first use."""
    served = getattr( server, 'catalog', None )
    if served is None:
        import betautils_catalog_store as bu_store
        try:
            served = ServedCatalog( server, getattr( server, 'catalog_factory', None ) )
        except bu_store.CatalogError as err:
            raise _protocol_error( err )
        server.catalog = served
    return served.catalog


def _call( function, *args, **kwargs ):
    import betautils_catalog_store as bu_store
    try:
        return function( *args, **kwargs )
    except bu_store.CatalogError as err:
        raise _protocol_error( err )


# --- the job queue's side ---------------------------------------------------------

def _open_quietly( server ):
    """The server's catalog for a job, or None when it cannot be opened (logged once)."""
    try:
        return catalog_for( server )
    except bu_proto.ProtocolError as err:
        if not getattr( server, '_catalog_warned', False ):
            server._catalog_warned = True
            server._logger.warning( "the library catalog is not following jobs (%s)"%( err.code, ) )
        return None


def job_events_for( server ):
    """The job queue's send(): every event to the client, then to the catalog."""
    def send( name, data ):
        server.send_event( name, data )
        if name not in ( 'job.state', 'job.file' ):
            return
        try:
            catalog = _open_quietly( server )
            jobs = getattr( server, 'jobs', None )
            job = jobs.jobs.get( data.get( 'job' ) ) if jobs is not None else None
            if catalog is not None and job is not None:
                catalog.job_event( name, data, [ entry.path for entry in job.files ] )
        except Exception:   # the catalog never stops a job
            server._logger.warning( "the library catalog missed a job event", exc_info=True )
    return send


class _ServerHistorySink:
    """The catalog's history sink, found when a record arrives (best-effort)."""

    required = False

    def __init__( self, server ):
        self._server = server

    def record( self, run ):
        catalog = _open_quietly( self._server )
        if catalog is not None:
            catalog.sink.record( run )

    def moved( self, run ):
        catalog = _open_quietly( self._server )
        if catalog is not None:
            catalog.sink.moved( run )


def job_recorder_for( server ):
    """Each job file's record: the stats and failure logs, then the catalog's history."""
    import betautils_run_record as bu_run_record
    return bu_run_record.RunRecorder(
        [ bu_run_record.StatsLogSink(), bu_run_record.FailureLogSink(), _ServerHistorySink( server ) ],
        program='engine', logger=server._logger )


class _ServerRunTypes:
    """The catalog's pins and folder rules, found when a file starts."""

    def __init__( self, server ):
        self._server = server

    def declared( self, source_path, digest, backend_name=None ):
        catalog = _open_quietly( self._server )
        if catalog is None:
            import betautils_catalog_types as bu_types
            return bu_types.declared_for( source_path, None, (), backend_name )
        return catalog.run_types().declared( source_path, digest, backend_name )


def job_types_for( server ):
    return _ServerRunTypes( server )


# --- param checks ---------------------------------------------------------------

def _item_id( data ):
    item = data.get( 'item' )
    if not isinstance( item, str ) or not item:
        raise _bad_params( 'item', bu_proto.REASON_MISSING if item is None else bu_proto.REASON_TYPE )
    return item


def _folder_id( data ):
    folder = data.get( 'folder' )
    if isinstance( folder, bool ) or not isinstance( folder, int ):
        raise _bad_params( 'folder', bu_proto.REASON_MISSING if folder is None else bu_proto.REASON_TYPE )
    return folder


def _string_list( data, field, required=True, choices=None, label=None ):
    label = label or field
    value = data.get( field )
    if value is None and not required:
        return None
    if not isinstance( value, list ) or ( required and not value ):
        raise _bad_params( label, bu_proto.REASON_MISSING if value is None else bu_proto.REASON_TYPE )
    if not all( isinstance( entry, str ) and entry for entry in value ):
        raise _bad_params( label, bu_proto.REASON_TYPE )
    if choices is not None and not set( value ) <= set( choices ):
        raise _bad_params( label, bu_proto.REASON_VALUE )
    return value


def _filters( data ):
    import betautils_catalog_index as bu_index
    raw = data.get( 'filter', {} )
    if not isinstance( raw, dict ):
        raise _bad_params( 'filter', bu_proto.REASON_TYPE )
    filters = {}
    statuses = ( 'new', 'queued', 'processing', 'done', 'failed' )
    for field, choices in ( ( 'status', statuses ), ( 'availability', bu_index.AVAILABILITY ), ( 'type', None ),
                            ( 'output', ( 'present', 'missing', 'none' ) ) ):
        value = _string_list( raw, field, required=False, choices=choices, label='filter.' + field ) if field in raw else None
        if value is not None:
            filters[field] = value
    for field in ( 'duplicates', 'pinned' ):
        if field in raw:
            if not isinstance( raw[field], bool ):
                raise _bad_params( 'filter.' + field, bu_proto.REASON_TYPE )
            filters[field] = raw[field]
    if 'folder' in raw:
        if isinstance( raw['folder'], bool ) or not isinstance( raw['folder'], int ):
            raise _bad_params( 'filter.folder', bu_proto.REASON_TYPE )
        filters['folder'] = raw['folder']
    return filters


def _int( data, field, default, minimum, maximum ):
    value = data.get( field, default )
    if isinstance( value, bool ) or not isinstance( value, int ):
        raise _bad_params( field, bu_proto.REASON_TYPE )
    if not minimum <= value <= maximum:
        raise _bad_params( field, bu_proto.REASON_VALUE )
    return value


# --- commands -----------------------------------------------------------------------

def catalog_list( server, data ):
    """A page of items, filtered and sorted."""
    catalog = catalog_for( server )
    import betautils_catalog as bu_catalog
    filters = _filters( data )
    sort = data.get( 'sort', 'name' )
    if sort not in bu_catalog.SORT_KEYS:
        raise _bad_params( 'sort', bu_proto.REASON_VALUE if isinstance( sort, str ) else bu_proto.REASON_TYPE )
    descending = data.get( 'descending', False )
    if not isinstance( descending, bool ):
        raise _bad_params( 'descending', bu_proto.REASON_TYPE )
    offset = _int( data, 'offset', 0, 0, 2**31 )
    limit = _int( data, 'limit', bu_catalog.LIST_LIMIT_DEFAULT, 1, bu_catalog.LIST_LIMIT_MAX )
    return _call( catalog.list, filters, sort, descending, offset, limit )


def catalog_get( server, data ):
    """One item with its locations and outputs."""
    item = _item_id( data )
    return _call( catalog_for( server ).get, item )


def catalog_folders( server, data ):
    return { 'folders': _call( catalog_for( server ).folders ) }


def catalog_add_folder( server, data ):
    """Watch a folder; its files are indexed by the next pass, started now."""
    path = data.get( 'path' )
    if not isinstance( path, str ) or not path:
        raise _bad_params( 'path', bu_proto.REASON_MISSING if path is None else bu_proto.REASON_TYPE )
    type_from_folder = data.get( 'type_from_folder', False )
    if not isinstance( type_from_folder, bool ):
        raise _bad_params( 'type_from_folder', bu_proto.REASON_TYPE )
    import os
    if not os.path.isabs( path ):
        raise _bad_params( 'path', bu_proto.REASON_VALUE )
    folder, created = _call( catalog_for( server ).add_folder, path, type_from_folder )
    return { 'folder': folder, 'created': created }


def catalog_remove_folder( server, data ):
    folder = _folder_id( data )
    touched = _call( catalog_for( server ).remove_folder, folder )
    return { 'items': len( touched ) }


def catalog_add_files( server, data ):
    """Index chosen files, hashed ahead of everything else."""
    paths = _string_list( data, 'paths' )
    import os
    for position, path in enumerate( paths ):
        if not os.path.isabs( path ):
            raise _bad_params( 'paths.%d'%( position ), bu_proto.REASON_VALUE )
    accepted, refused = _call( catalog_for( server ).add_files, paths )
    return { 'accepted': len( accepted ),
             'refused': [ { 'index': position, 'reason': reason } for position, reason in refused ] }


def catalog_rescan( server, data ):
    _call( catalog_for( server ).rescan )
    return {}


def catalog_pin_type( server, data ):
    """Pin an item's video type, or unpin it (type null) for Auto."""
    item = _item_id( data )
    if 'type' not in data:
        raise _bad_params( 'type', bu_proto.REASON_MISSING )
    pin = data['type']
    if pin is not None and ( not isinstance( pin, str ) or not pin ):
        raise _bad_params( 'type', bu_proto.REASON_TYPE )
    catalog = catalog_for( server )
    _call( catalog.pin_type, item, pin )
    return _call( catalog.get, item )


def catalog_thumbnail( server, data ):
    """An item's thumbnail: blurred, or with reveal the clear frame (in memory only)."""
    item = _item_id( data )
    reveal = data.get( 'reveal', False )
    if not isinstance( reveal, bool ):
        raise _bad_params( 'reveal', bu_proto.REASON_TYPE )
    of = data.get( 'of', 'source' )
    if of not in ( 'source', 'output' ):
        raise _bad_params( 'of', bu_proto.REASON_VALUE if isinstance( of, str ) else bu_proto.REASON_TYPE )
    if reveal and of != 'source':
        raise _bad_params( 'reveal', bu_proto.REASON_VALUE )
    return _call( catalog_for( server ).thumbnail, item, reveal, of )


def catalog_forget( server, data ):
    item = _item_id( data )
    _call( catalog_for( server ).forget, item )
    return {}


HANDLERS = {
    'catalog.list':          catalog_list,
    'catalog.get':           catalog_get,
    'catalog.folders':       catalog_folders,
    'catalog.add_folder':    catalog_add_folder,
    'catalog.remove_folder': catalog_remove_folder,
    'catalog.add_files':     catalog_add_files,
    'catalog.rescan':        catalog_rescan,
    'catalog.forget':        catalog_forget,
    'catalog.pin_type':      catalog_pin_type,
    'catalog.thumbnail':     catalog_thumbnail,
}

EVENTS = ( 'catalog.changed', )
