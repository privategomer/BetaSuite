"""
betautils_serve_models.py - the models.* commands and the models.state and
models.progress events (protocol/README.md, "Models").

Downloads run one at a time on a worker thread, in the order asked for;
the command that asks returns at once. The network is used only by an
install the client asked for. A stopped, failed or interrupted download
keeps its partial file, so installing again resumes it, including after
the engine restarts.

models.use writes the same local config as config.set, and is refused
while a job holds the queue (config.busy), like config.set.
"""

import collections
import threading
import time

import betautils_models as bu_models
import betautils_protocol as bu_proto

PROGRESS_INTERVAL = 0.25

# models.state states
QUEUED = 'queued'
DOWNLOADING = 'downloading'
INSTALLED = 'installed'
STOPPED = 'stopped'
FAILED = 'failed'
REMOVED = 'removed'


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


def _protocol_error( err ):
    return bu_proto.ProtocolError( err.code, dict( err.params ) )


def _model_id( data ):
    value = data.get( 'id' )
    if value is None:
        raise _bad_params( 'id', bu_proto.REASON_MISSING )
    if not isinstance( value, str ):
        raise _bad_params( 'id', bu_proto.REASON_TYPE )
    row = bu_models.entry( value )
    if row is None:
        raise bu_proto.ProtocolError( 'model.unknown', { 'model': value } )
    return row


class Installer:
    """One download at a time, in order, with stop and progress."""

    def __init__( self, send_event, logger, install=None, clock=time.monotonic ):
        self._send = send_event
        self._logger = logger
        self._install = install or bu_models.install
        self._clock = clock
        self._cond = threading.Condition()
        self._queue = collections.deque()
        self._current = None
        self._stop_ids = set()
        self._closing = False
        self._thread = threading.Thread( target=self._run, name='models-install', daemon=True )
        self._thread.start()

    # --- what the commands see -------------------------------------------

    def busy_ids( self ):
        with self._cond:
            return ( [ self._current ] if self._current else [] ) + list( self._queue )

    def state_of( self, model_id ):
        with self._cond:
            if model_id == self._current:
                return DOWNLOADING
            if model_id in self._queue:
                return QUEUED
        return None

    def add( self, model_id ):
        """Queue an install. False when it is already queued or downloading."""
        with self._cond:
            if model_id == self._current or model_id in self._queue:
                return False
            self._stop_ids.discard( model_id )
            self._queue.append( model_id )
            self._cond.notify()
        self._send( 'models.state', { 'id': model_id, 'state': QUEUED } )
        return True

    def cancel( self, model_id ):
        """Stop a download or take it off the queue. False when it was neither."""
        with self._cond:
            if model_id in self._queue:
                self._queue.remove( model_id )
                queued = True
            elif model_id == self._current:
                self._stop_ids.add( model_id )
                return True
            else:
                return False
        if queued:
            self._send( 'models.state', { 'id': model_id, 'state': STOPPED } )
        return True

    def stop( self, timeout=10.0 ):
        """Stop at engine shutdown: the download in progress keeps its partial file."""
        with self._cond:
            self._closing = True
            self._queue.clear()
            if self._current:
                self._stop_ids.add( self._current )
            self._cond.notify()
        self._thread.join( timeout )

    # --- the worker ---------------------------------------------------------

    def _run( self ):
        while True:
            with self._cond:
                while not self._closing and not self._queue:
                    self._cond.wait()
                if self._closing:
                    return
                model_id = self._current = self._queue.popleft()
            try:
                self._one( model_id )
            finally:
                with self._cond:
                    self._current = None
                    self._stop_ids.discard( model_id )

    def _one( self, model_id ):
        self._send( 'models.state', { 'id': model_id, 'state': DOWNLOADING } )
        last = { 'at': None }

        def progress( done, total ):
            now = self._clock()
            if done < total and last['at'] is not None and now - last['at'] < PROGRESS_INTERVAL:
                return
            last['at'] = now
            self._send( 'models.progress', { 'id': model_id, 'bytes': done, 'total': total } )

        def should_stop():
            with self._cond:
                return model_id in self._stop_ids

        try:
            self._install( model_id, progress=progress, should_stop=should_stop )
        except bu_models.Stopped:
            self._send( 'models.state', { 'id': model_id, 'state': STOPPED } )
        except bu_models.ModelError as err:
            self._logger.warning( "model %s not installed: %s %s"%( model_id, err.code, err.params ) )
            self._send( 'models.state', { 'id': model_id, 'state': FAILED, 'code': err.code,
                                          'params': dict( err.params ) } )
        except Exception as err:
            self._logger.error( "model %s install failed"%( model_id, ), exc_info=True )
            self._send( 'models.state', { 'id': model_id, 'state': FAILED, 'code': 'engine.internal',
                                          'params': { 'exception_type': type( err ).__name__ } } )
        else:
            self._logger.info( "model %s installed"%( model_id, ) )
            self._send( 'models.state', { 'id': model_id, 'state': INSTALLED } )


def installer_for( server ):
    if getattr( server, 'models', None ) is None:
        server.models = Installer( server.send_event, server._logger )
    return server.models


def _busy_job( server ):
    jobs = getattr( server, 'jobs', None )
    return jobs.busy_job() if jobs is not None else None


def _refuse_while_busy( server ):
    busy = _busy_job( server )
    if busy is not None:
        raise bu_proto.ProtocolError( 'config.busy', { 'job': busy } )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _hardware():
    import betautils_hardware as bu_hardware
    return bu_hardware.detect()


def _gpu_enabled():
    import betaconfig
    return getattr( betaconfig, 'gpu_enabled', 0 ) == 1


def models_list( server, _data ):
    """
    result: { models: [ { id, kind, backend, variant, size, license, state, bytes } ],
              in_use: { detector, person }, custom: { kind: path },
              recommended: { detector, reason }, gpu: { usable, enabled, name } }
    """
    installer = getattr( server, 'models', None )
    rows = []
    for row in bu_models.entries():
        found = bu_models.status( row )
        busy = installer.state_of( row['id'] ) if installer is not None else None
        lic = bu_models.license_for( row )
        rows.append( {
            'id': row['id'], 'kind': row['kind'], 'backend': row['backend'], 'variant': row.get( 'variant' ),
            'size': row['size'], 'state': busy or found['state'], 'bytes': found['bytes'],
            'license': { 'spdx': lic['spdx'], 'project': lic['project'], 'home': lic['home'] },
        } )
    used = bu_models.in_use()
    hardware = _hardware()
    model_id, reason = bu_models.recommended_detector( hardware )
    return {
        'models': rows,
        'in_use': { 'detector': used['detector'], 'person': used['person'] },
        'custom': used['custom'],
        'recommended': { 'detector': model_id, 'reason': reason },
        'gpu': { 'usable': bool( hardware.get( 'cuda_usable' ) ), 'enabled': _gpu_enabled(),
                 'name': ( hardware.get( 'gpu' ) or {} ).get( 'name' ) },
    }


def models_install( server, data ):
    """data: { id }; result: { id, queued }. Progress follows as events."""
    row = _model_id( data )
    return { 'id': row['id'], 'queued': installer_for( server ).add( row['id'] ) }


def models_cancel( server, data ):
    """data: { id }; result: { id, cancelled }. The partial file is kept for a resume."""
    row = _model_id( data )
    installer = getattr( server, 'models', None )
    return { 'id': row['id'], 'cancelled': bool( installer and installer.cancel( row['id'] ) ) }


def models_import( _server, data ):
    """data: { path }; result: { id }. "I already have the file"."""
    import os
    path = data.get( 'path' )
    if path is None:
        raise _bad_params( 'path', bu_proto.REASON_MISSING )
    if not isinstance( path, str ) or not path:
        raise _bad_params( 'path', bu_proto.REASON_TYPE )
    if not os.path.isfile( path ):
        raise _bad_params( 'path', bu_proto.REASON_VALUE )
    try:
        return { 'id': bu_models.import_file( path ) }
    except bu_models.ModelError as err:
        raise _protocol_error( err )


def models_remove( server, data ):
    """data: { id }; result: { id }. Refused for the model in use, and while it downloads."""
    row = _model_id( data )
    _refuse_while_busy( server )
    installer = getattr( server, 'models', None )
    if installer is not None and installer.state_of( row['id'] ):
        raise bu_proto.ProtocolError( 'model.busy', { 'model': row['id'] } )
    try:
        bu_models.remove( row['id'] )
    except bu_models.ModelError as err:
        raise _protocol_error( err )
    server.send_event( 'models.state', { 'id': row['id'], 'state': REMOVED } )
    return { 'id': row['id'] }


def use_values( row, gpu=False ):
    """( { dotted path: value }, [ dotted paths to reset ] ) that make the config use this model."""
    values, resets = {}, []
    if row['kind'] == 'detector':
        values['detector_backend.selected'] = row['backend']
        if row.get( 'variant' ) is not None:
            values['detector_backend.%s.model_variant'%( row['backend'], )] = row['variant']
        # A model_path set by hand would win over the variant.
        resets.append( 'detector_backend.%s.model_path'%( row['backend'], ) )
    else:
        values['person_backend.selected'] = row['backend']
        values['person_backend.%s.model_variant'%( row['backend'], )] = row['variant']
        resets.append( 'person_backend.%s.model_path'%( row['backend'], ) )
    if gpu:
        values['gpu_enabled'] = 1
    import betautils_settings as bu_settings
    resets = [ path for path in resets if bu_settings.find( tuple( path.split( '.' ) ) ) is not None ]
    return values, resets


def models_use( server, data ):
    """
    data: { id, gpu?: bool, dry_run?: bool }
    result: as config.set. gpu: true also turns on GPU detection in the
    same write, which the recommended "install and use" sends when the
    machine has a usable GPU.
    """
    import betautils_serve_config as bu_serve_config
    row = _model_id( data )
    gpu = data.get( 'gpu', False )
    if not isinstance( gpu, bool ):
        raise _bad_params( 'gpu', bu_proto.REASON_TYPE )
    _refuse_while_busy( server )
    if bu_models.status( row )['state'] != 'installed':
        raise bu_proto.ProtocolError( 'model.not_installed', { 'model': row['id'] } )
    values, resets = use_values( row, gpu )
    updates = { tuple( path.split( '.' ) ): value for path, value in values.items() }
    removals = [ tuple( path.split( '.' ) ) for path in resets ]
    return bu_serve_config._edit( data, updates, removals )


HANDLERS = {
    'models.list':    models_list,
    'models.install': models_install,
    'models.cancel':  models_cancel,
    'models.import':  models_import,
    'models.remove':  models_remove,
    'models.use':     models_use,
}
