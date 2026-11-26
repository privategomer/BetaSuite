"""
betautils_serve_resources.py - the resource commands of serve mode.

  hw.detect         this machine (betautils_hardware.machine) and the
                    settings its facts settle without measuring
  resources.watch   start or stop resource.sample events (the Performance
                    page and the status strip); answers the latest sample

The sampler (betautils_resources.Sampler) runs only while someone uses
it: the UI watching, or a running job's governor. Events are sent only
while the UI watches.
"""

import threading

import betautils_protocol as bu_proto

SAMPLE_EVENT = 'resource.sample'


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


class Resources:
    """
    The sampler, shared by everything that needs it. use( who ) and
    release( who ) count its users; it starts with the first and stops
    with the last. While 'watch' is a user, each sample is sent as a
    resource.sample event.
    """

    def __init__( self, send, logger, make_sampler=None ):
        self._send = send
        self._logger = logger
        self._make_sampler = make_sampler or self._default_sampler
        self._lock = threading.Lock()
        self._users = set()
        self.sampler = None

    def _default_sampler( self ):
        import betautils_resources as bu_resources
        return bu_resources.Sampler( gpu=bu_resources.GpuStream(), logger=self._logger )

    def _on_sample( self, snapshot ):
        if 'watch' in self._users:
            self._send( SAMPLE_EVENT, snapshot )

    def use( self, who ):
        with self._lock:
            self._users.add( who )
            if self.sampler is None:
                self.sampler = self._make_sampler()
                self.sampler.add_listener( self._on_sample )
                self.sampler.start()
            return self.sampler

    def release( self, who ):
        with self._lock:
            self._users.discard( who )
            sampler = self.sampler if not self._users else None
            if sampler is not None:
                self.sampler = None
        if sampler is not None:
            sampler.stop()

    def latest( self ):
        sampler = self.sampler
        if sampler is None:
            return None
        return sampler.latest or sampler.sample()

    def stop( self ):
        with self._lock:
            self._users.clear()
            sampler, self.sampler = self.sampler, None
        if sampler is not None:
            sampler.stop()


def resources_for( server ):
    if getattr( server, 'resources', None ) is None:
        server.resources = Resources( server.send_event, server._logger )
    return server.resources


def hw_detect( _server, _data ):
    import betaconfig
    import betautils_hardware as bu_hardware
    facts = bu_hardware.machine()
    facts['safe_settings'] = bu_hardware.safe_settings( facts, betaconfig )
    return facts


def resources_watch( server, data ):
    on = data.get( 'on' )
    if not isinstance( on, bool ):
        raise _bad_params( 'on', bu_proto.REASON_TYPE if on is not None else bu_proto.REASON_MISSING )
    resources = resources_for( server )
    if on:
        resources.use( 'watch' )
    else:
        resources.release( 'watch' )
    return { 'watching': on, 'sample': resources.latest() if on else None }


HANDLERS = {
    'hw.detect':       hw_detect,
    'resources.watch': resources_watch,
}
