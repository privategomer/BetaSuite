"""
betautils_serve_strip.py - catalog.strip: the player's scrub strip for a
video, built in the background in two passes (betautils_catalog_strip).

catalog.strip answers at once with the best strip on disk (the even pass
over the keyframes pass) and whether a pass is still being built. A pass
being built is announced by catalog.strip_ready { path, density, ok } when
it is written, and asking again then gives it. The keyframes pass is
built at once, since the person is about to drag; the even pass waits
while a job runs or a command-line run holds the library, so it never
slows a run.

A video is a censored output, stored clear, only when the catalog lists
it as a present output. Any other file's strip is blurred like a
source's thumbnail.
"""

import collections
import os
import threading

import betautils_catalog_strip as bu_strip
import betautils_protocol as bu_proto

BUSY_POLL_SECONDS = 1.0


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


class StripService:
    """
    One worker thread building strips in the order asked.

    Args:
        send_event: send_event( name, data ).
        busy: busy() -> whether a run is on (the even pass waits).
        logger: The shared logger.
        build: bu_strip.build( path, density, blurred, facts ), for tests.
        facts: facts( path ) -> { duration, fps, width, height } or None.
    """

    def __init__( self, send_event, busy, logger, build=None, facts=None, poll=BUSY_POLL_SECONDS ):
        self.send_event = send_event
        self.busy = busy
        self.logger = logger
        self._build = build or bu_strip.build
        self._facts = facts or _stream_facts
        self._poll = poll
        self._cond = threading.Condition()
        self._tasks = collections.deque()
        self._pending = set()      # ( path, density, blurred ) queued or building
        self._failed = set()       # ( key ) that could not be built, not tried again
        self._stopping = False
        self._thread = threading.Thread( target=self._work, name='strip-worker', daemon=True )
        self._thread.start()

    def request( self, path, blurred ):
        """catalog.strip's answer for path, queuing whatever pass is missing."""
        even_key = bu_strip.key_for( path, bu_strip.EVEN, blurred )
        keyframes_key = bu_strip.key_for( path, bu_strip.KEYFRAMES, blurred )
        even = bu_strip.read( even_key )
        keyframes = None if even is not None else bu_strip.read( keyframes_key )
        best = even or keyframes
        with self._cond:
            if best is None and keyframes_key not in self._failed:
                self._queue_locked( path, bu_strip.KEYFRAMES, blurred )
            elif ( even is None and keyframes is not None and even_key not in self._failed
                   and bu_strip.needs_even_pass( keyframes['frames'], keyframes.get( 'duration', 0 ) ) ):
                self._queue_locked( path, bu_strip.EVEN, blurred )
            building = any( task[0] == path and task[2] == blurred for task in self._pending )
        out = { 'path': path, 'blurred': bool( blurred ), 'building': building, 'ready': best is not None }
        if best is not None:
            out.update( { key: best[key] for key in ( 'density', 'frames', 'tile', 'columns', 'duration', 'image' ) } )
        return out

    def _queue_locked( self, path, density, blurred ):
        task = ( path, density, blurred )
        if task in self._pending:
            return
        self._pending.add( task )
        self._tasks.append( task )
        self._cond.notify_all()

    def _next( self ):
        with self._cond:
            while not self._stopping:
                for task in list( self._tasks ):
                    if task[1] == bu_strip.KEYFRAMES or not self.busy():
                        self._tasks.remove( task )
                        return task
                # Only even passes are waiting, and a run is on.
                self._cond.wait( self._poll if self._tasks else None )
            return None

    def _work( self ):
        while True:
            task = self._next()
            if task is None:
                return
            path, density, blurred = task
            ok = False
            needs_even = False
            try:
                facts = self._facts( path )
                if facts is None:
                    raise bu_strip.StripError( 'not_video' )
                layout = self._build( path, density, blurred, facts )
                ok = layout is not None
                needs_even = ( ok and density == bu_strip.KEYFRAMES
                               and bu_strip.needs_even_pass( layout['frames'], facts['duration'] ) )
                self.logger.info( "strip: %s pass of %s, %d frame(s)%s"%(
                    density, os.path.basename( path ), len( layout['frames'] ) if ok else 0,
                    ', blurred' if blurred else '' ) )
            except bu_strip.StripError as err:
                self.logger.warning( "strip: no %s pass of %s (%s)"%( density, os.path.basename( path ), err.reason ) )
            except Exception:   # noqa: BLE001  the worker must outlive any file
                self.logger.warning( "strip: the %s pass of %s failed"%( density, os.path.basename( path ) ),
                                     exc_info=True )
            with self._cond:
                self._pending.discard( task )
                if not ok:
                    try:
                        self._failed.add( bu_strip.key_for( path, density, blurred ) )
                    except OSError:
                        pass
                if needs_even:
                    self._queue_locked( path, bu_strip.EVEN, blurred )
            self.send_event( 'catalog.strip_ready', { 'path': path, 'density': density, 'ok': ok } )

    def stop( self ):
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        self._thread.join( 5 )


def _stream_facts( path ):
    import betautils_serve_estimate as bu_serve_estimate
    return bu_serve_estimate.stream_facts( path )


def is_present_output( catalog, path ):
    import betautils_catalog_index as bu_index
    return bool( catalog.store.read( 'SELECT 1 FROM outputs WHERE path = ? AND present = 1 LIMIT 1',
                                     ( bu_index.norm( path ), ) ) )


def catalog_strip( server, data ):
    """
    data: { path }: an absolute path to a video file.
    result: { path, blurred, building, ready, density?, frames?, tile?, columns?, duration?, image? }
    """
    import betautils_serve_catalog as bu_serve_catalog
    path = data.get( 'path' )
    if not isinstance( path, str ) or not path:
        raise _bad_params( 'path', bu_proto.REASON_MISSING if path is None else bu_proto.REASON_TYPE )
    if not os.path.isabs( path ):
        raise _bad_params( 'path', bu_proto.REASON_VALUE )
    if not os.path.isfile( path ):
        raise bu_proto.ProtocolError( 'catalog.no_such_file', {} )
    catalog = bu_serve_catalog.catalog_for( server )
    return bu_serve_catalog.served_strips( server ).request( path, not is_present_output( catalog, path ) )


HANDLERS = {
    'catalog.strip': catalog_strip,
}

EVENTS = ( 'catalog.strip_ready', )
