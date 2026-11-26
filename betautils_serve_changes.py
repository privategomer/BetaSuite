"""
betautils_serve_changes.py - settings changed while jobs run.

Settings stay as they were while a job holds the queue (config.set and
config.reset answer config.busy), because a run reads them as it goes.
Rather than lose what the user just chose, the app offers:

  apply after the jobs   config.set / config.reset with when: after_jobs.
                         The change is checked now (as config.set would,
                         against the local config file, not the job's
                         live settings) and kept in pending_config.json
                         in the state folder, so a restart keeps it. It
                         is written when the queue is next idle, or when
                         the engine next starts if it stopped first.
                         config.pending lists what waits;
                         config.pending_discard drops one or all
  pause, apply, go on    job.restart_with_changes mode resume: the
                         running job pauses at its next safe point, the
                         change is written, the job resumes. The file in
                         progress starts again from its checkpoints where
                         its keys did not move, and from the beginning
                         where they did
  start again with it    mode requeue: every job is cancelled, the change
                         is written, and each job's remaining files are
                         queued again with its own overrides

Events: config.applied { id, moves, may_move } when a waiting or
restart change is written; config.pending_failed { id, error } when a
waiting change no longer passes (the config changed under it).
"""

import json
import os
import threading
import time
import uuid

import betautils_protocol as bu_proto

FILE_NAME = 'pending_config.json'
RESUME = 'resume'
REQUEUE = 'requeue'


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


def parse_change( data ):
    """( updates { path: value }, removals [ path ] ) from { values?, resets? }, checked like config.set."""
    import betautils_serve_config as bu_serve_config
    values = data.get( 'values', {} )
    resets = data.get( 'resets', [] )
    if not isinstance( values, dict ):
        raise _bad_params( 'values', bu_proto.REASON_TYPE )
    if not isinstance( resets, list ):
        raise _bad_params( 'resets', bu_proto.REASON_TYPE )
    if not values and not resets:
        raise _bad_params( 'values', bu_proto.REASON_VALUE )
    updates = {}
    for dotted, value in values.items():
        path = bu_serve_config._path( dotted, 'values' )
        bu_serve_config._settable( path )
        updates[path] = value
    removals = []
    for dotted in resets:
        path = bu_serve_config._path( dotted, 'resets' )
        bu_serve_config._settable( path )
        removals.append( path )
    return updates, removals


def _dotted( updates, removals ):
    return { '.'.join( path ): value for path, value in updates.items() }, [ '.'.join( path ) for path in removals ]


def _paths( values, resets ):
    return { tuple( dotted.split( '.' ) ): value for dotted, value in values.items() }, \
           [ tuple( dotted.split( '.' ) ) for dotted in resets ]


def check( updates, removals ):
    """The change as config.set would answer it, dry: raises its ProtocolError when it would not pass."""
    import betautils_serve_config as bu_serve_config
    return bu_serve_config._edit( { 'dry_run': True }, updates, removals )


def write( updates, removals ):
    import betautils_serve_config as bu_serve_config
    return bu_serve_config._edit( {}, updates, removals )


class Pending:
    """The changes waiting for the queue to be idle, kept in a file in the state folder."""

    def __init__( self, path=None, clock=time.time ):
        if path is None:
            import betautils_paths as bu_paths
            path = os.path.join( bu_paths.kind_dir( 'state' ), FILE_NAME )
        self.path = path
        self.clock = clock
        self.lock = threading.Lock()

    def items( self ):
        try:
            with open( self.path, 'r', encoding='utf-8' ) as handle:
                found = json.load( handle )
        except ( OSError, ValueError ):
            return []
        return [ item for item in found if isinstance( item, dict ) and 'id' in item ] if isinstance( found, list ) else []

    def _save( self, items ):
        if not items:
            try:
                os.remove( self.path )
            except OSError:
                pass
            return
        os.makedirs( os.path.dirname( self.path ), exist_ok=True )
        temp = self.path + '.tmp'
        with open( temp, 'w', encoding='utf-8' ) as handle:
            json.dump( items, handle )
        os.replace( temp, self.path )

    def add( self, values, resets ):
        with self.lock:
            items = self.items()
            item = { 'id': uuid.uuid4().hex[:12], 'values': values, 'resets': resets, 'at': self.clock() }
            items.append( item )
            self._save( items )
            return item

    def discard( self, item_id=None ):
        """Drop one waiting change, or all of them. Returns how many were dropped."""
        with self.lock:
            items = self.items()
            kept = [ item for item in items if item_id is not None and item['id'] != item_id ]
            self._save( kept )
            return len( items ) - len( kept )

    def apply( self, send, logger=None ):
        """Write every waiting change, oldest first. Each that no longer passes is dropped and told."""
        with self.lock:
            items = self.items()
            self._save( [] )
        for item in items:
            updates, removals = _paths( item.get( 'values' ) or {}, item.get( 'resets' ) or [] )
            try:
                answer = write( updates, removals )
            except bu_proto.ProtocolError as refused:
                if logger is not None:
                    logger.warning( "a change waiting for the jobs no longer passes: %s"%( refused.error, ) )
                send( 'config.pending_failed', { 'id': item['id'], 'error': refused.error } )
                continue
            send( 'config.applied', { 'id': item['id'], 'moves': answer['moves'], 'may_move': answer['may_move'] } )
        return len( items )


def pending_for( server ):
    if getattr( server, 'pending', None ) is None:
        server.pending = Pending()
    return server.pending


def _busy_job( server ):
    jobs = getattr( server, 'jobs', None )
    return jobs.busy_job() if jobs is not None else None


def queue_after_jobs( server, data, updates, removals ):
    """config.set / config.reset with when: after_jobs: checked now, written when the queue is idle."""
    answer = check( updates, removals )
    if _busy_job( server ) is None:
        answer = write( updates, removals )
        answer['when'] = 'now'
        return answer
    values, resets = _dotted( updates, removals )
    item = pending_for( server ).add( values, resets )
    server.jobs.when_idle( 'config.pending', lambda: pending_for( server ).apply( server.send_event, server._logger ) )
    answer['when'] = 'after_jobs'
    answer['id'] = item['id']
    answer['written'] = False
    answer['dry_run'] = False
    return answer


def apply_at_start( server ):
    """Changes left waiting by an engine that stopped before its jobs did: written now, with nothing running."""
    if _busy_job( server ) is None and pending_for( server ).items():
        pending_for( server ).apply( server.send_event, server._logger )


def config_pending( server, _data ):
    return { 'pending': pending_for( server ).items() }


def config_pending_discard( server, data ):
    item_id = data.get( 'id' )
    if item_id is not None and ( not isinstance( item_id, str ) or not item_id ):
        raise _bad_params( 'id', bu_proto.REASON_TYPE )
    return { 'discarded': pending_for( server ).discard( item_id ) }


def job_restart_with_changes( server, data ):
    """
    data: { values?, resets?, mode: resume | requeue }. Checked first;
    then the jobs are paused or cancelled, the change written and the
    work carried on, in the background. Answers at once; config.applied
    says when the change is in.
    """
    mode = data.get( 'mode' )
    if mode not in ( RESUME, REQUEUE ):
        raise _bad_params( 'mode', bu_proto.REASON_MISSING if mode is None else bu_proto.REASON_VALUE )
    updates, removals = parse_change( data )
    check( updates, removals )
    jobs = getattr( server, 'jobs', None )
    if jobs is None or jobs.busy_job() is None:
        answer = write( updates, removals )
        server.send_event( 'config.applied', { 'id': None, 'moves': answer['moves'], 'may_move': answer['may_move'] } )
        return { 'mode': mode, 'jobs': [] }
    if mode == RESUME:
        touched = jobs.restart_paused( server, lambda: _write_and_tell( server, updates, removals ) )
    else:
        touched = jobs.requeue_with( server, lambda: _write_and_tell( server, updates, removals ) )
    return { 'mode': mode, 'jobs': touched }


def _write_and_tell( server, updates, removals ):
    try:
        answer = write( updates, removals )
    except bu_proto.ProtocolError as refused:
        server.send_event( 'config.pending_failed', { 'id': None, 'error': refused.error } )
        return False
    server.send_event( 'config.applied', { 'id': None, 'moves': answer['moves'], 'may_move': answer['may_move'] } )
    return True


HANDLERS = {
    'config.pending': config_pending,
    'config.pending_discard': config_pending_discard,
    'job.restart_with_changes': job_restart_with_changes,
}
