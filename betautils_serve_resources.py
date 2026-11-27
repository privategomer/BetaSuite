"""
betautils_serve_resources.py - the resource commands of serve mode.

  hw.detect         this machine (betautils_hardware.machine) and the
                    settings its facts settle without measuring
  resources.watch   start or stop resource.sample events (the Performance
                    page and the status strip); answers the latest sample
  governor.get      the governor's decision, why, and its recent history
  governor.override "run at full speed anyway", for this run or the session;
                    memory_guard: false also turns off the memory guard
  governor.override_clear
  governor.enable   the on/off switch, live

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

    def __init__( self, send, logger, make_sampler=None, own_root=None ):
        self._send = send
        self._logger = logger
        self._make_sampler = make_sampler or self._default_sampler
        self.own_root = own_root      # the process whose tree is the run; None: this engine alone
        self._lock = threading.Lock()
        self._users = set()
        self.sampler = None
        self.observer = None    # the governor's observe(), fed every sample outside a run

    def _default_sampler( self ):
        import os
        import betautils_resources as bu_resources
        reader = bu_resources.platform_reader( own_root=self.own_root )
        return bu_resources.Sampler( reader=reader, gpu=bu_resources.GpuStream(), logger=self._logger )

    def _on_sample( self, snapshot ):
        if 'watch' in self._users:
            self._send( SAMPLE_EVENT, snapshot )
        observer = self.observer
        if observer is not None:
            try:
                observer( snapshot )
            except Exception:
                if self._logger is not None:
                    self._logger.warning( "the governor could not take a sample", exc_info=True )

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
        import os
        # The app starts the engine (python betaengine.py serve) itself, so
        # the parent's tree is the app: its window and web view count as
        # the run, never as other programs keeping the computer busy. The
        # command line (betatv) makes its own Resources, with the engine alone.
        server.resources = Resources( server.send_event, server._logger, own_root=os.getppid() )
    return server.resources


def runtime_for( server ):
    """The engine's governor, shared by every job and the governor.* commands."""
    if getattr( server, 'governor', None ) is None:
        import betaconfig
        import betautils_governor_run as bu_gov_run
        server.governor = bu_gov_run.Runtime( send=server.send_event, resources=resources_for( server ),
                                              logger=server._logger )
        # Outside a run the governor still follows the setting.
        import betautils_governor as bu_gov
        server.governor.governor.set_thresholds( bu_gov.Thresholds.from_config( betaconfig ),
                                                 server.governor.clock() )
        # And the machine, whenever the sampler runs (the Performance page).
        resources_for( server ).observer = server.governor.observe
    return server.governor


def _busy_job( server ):
    jobs = getattr( server, 'jobs', None )
    return jobs.busy_job() if jobs is not None else None


def _governor_data( runtime ):
    data = runtime.state_data()
    # The history in wall-clock time, oldest first.
    offset = runtime.wall() - runtime.clock()
    data['history'] = [ dict( decision.as_dict(), time=round( when + offset, 3 ) )
                        for when, decision in list( runtime.governor.history ) ]
    return data


def governor_get( server, _data ):
    return _governor_data( runtime_for( server ) )


def governor_override( server, data ):
    import betautils_governor as bu_gov
    scope = data.get( 'scope' )
    if scope not in ( bu_gov.RUN, bu_gov.SESSION ):
        raise _bad_params( 'scope', bu_proto.REASON_MISSING if scope is None else bu_proto.REASON_VALUE )
    memory_guard = data.get( 'memory_guard', True )
    if not isinstance( memory_guard, bool ):
        raise _bad_params( 'memory_guard', bu_proto.REASON_TYPE )
    runtime = runtime_for( server )
    runtime.set_override( scope, memory_guard=memory_guard )
    return _governor_data( runtime )


def governor_override_clear( server, _data ):
    runtime = runtime_for( server )
    runtime.clear_override()
    return _governor_data( runtime )


def _write_enabled( server, on ):
    import betautils_serve_config as bu_serve_config
    return bu_serve_config._edit( {}, { ( 'governor', 'enabled' ): on }, [] )


def governor_enable( server, data ):
    """
    The on/off switch. Takes effect at once, in a running job too. The
    setting is written now when no job holds the queue, else as soon as
    the queue is idle (settings stay as they were while a job runs).
    """
    on = data.get( 'on' )
    if not isinstance( on, bool ):
        raise _bad_params( 'on', bu_proto.REASON_TYPE if on is not None else bu_proto.REASON_MISSING )
    runtime = runtime_for( server )
    runtime.set_enabled( on )
    busy = _busy_job( server )
    if busy is None:
        _write_enabled( server, on )
        saved = 'now'
    else:
        def write_when_idle():
            try:
                _write_enabled( server, on )
            except bu_proto.ProtocolError as err:
                server._logger.warning( "governor.enabled could not be written: %s"%( err.error, ) )
        server.jobs.when_idle( 'governor.enabled', write_when_idle )
        saved = 'after_jobs'
    data = _governor_data( runtime )
    data['saved'] = saved
    return data


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
    runtime_for( server )     # so the governor follows what is watched
    if on:
        resources.use( 'watch' )
    else:
        resources.release( 'watch' )
    return { 'watching': on, 'sample': resources.latest() if on else None }


HANDLERS = {
    'hw.detect':               hw_detect,
    'resources.watch':         resources_watch,
    'governor.get':            governor_get,
    'governor.override':       governor_override,
    'governor.override_clear': governor_override_clear,
    'governor.enable':         governor_enable,
}
