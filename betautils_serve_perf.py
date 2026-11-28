"""
betautils_serve_perf.py - the performance index in serve mode and the
command line: one index per process, the sink every recorder gets, and
the perf.* commands.

  perf.evidence   what past runs on this machine measured at the
                  settings in use, how often memory capped the render,
                  and nudges (each with a one-click action)
  perf.rebuild    empty the index and import the stats log again
"""

import threading

_lock = threading.Lock()
_index = None
_hardware = None


def shared_index():
    """
    The process's index, opened (and on first use filled from the stats
    log) when first asked; opened again if the stats log has moved.
    """
    global _index
    import betautils_perf as bu_perf
    path = bu_perf.default_path()
    with _lock:
        if _index is not None and _index.path != path:
            _index.close()
            _index = None
        if _index is None:
            _index = bu_perf.Index.open( path )
        return _index


def close():
    global _index
    with _lock:
        if _index is not None:
            _index.close()
            _index = None


def machine():
    """( hw.detect facts, fingerprint ), read once per process."""
    global _hardware
    with _lock:
        if _hardware is None:
            import betautils_hardware as bu_hardware
            import betautils_perf as bu_perf
            facts = bu_hardware.machine( disk_kinds={} )
            _hardware = ( facts, bu_perf.hardware_fingerprint( facts ) )
        return _hardware


class _Sink:
    """PerfSink with the shared index and this machine's fingerprint, both found on first use."""

    required = False

    def record( self, run ):
        if run.stats is None:
            return
        import betautils_perf as bu_perf
        bu_perf.PerfSink( shared_index, hardware=machine()[1] ).record( run )


def sink():
    return _Sink()


def providers_now( config, facts ):
    """The execution providers a run would use with these settings, as the stats row writes them."""
    if getattr( config, 'gpu_enabled', 0 ) == 1 and facts.get( 'cuda_usable' ):
        return 'CUDAExecutionProvider,CPUExecutionProvider'
    return 'CPUExecutionProvider'


def current_settings( config ):
    import betautils_detector as bu_detector
    backend = bu_detector.selected_backend_name()
    return { 'variant': bu_detector.selected_variant_name( backend ),
             'batch': bu_detector.get_nn_batch_size( backend ) }


def perf_evidence( _server, _data ):
    import betaconfig
    facts, fingerprint = machine()
    return shared_index().evidence( providers_now( betaconfig, facts ), hardware=fingerprint,
                                    current=current_settings( betaconfig ), facts=facts,
                                    gpu_enabled=getattr( betaconfig, 'gpu_enabled', 0 ) )


def perf_rebuild( _server, _data ):
    index = shared_index()
    imported, read = index.rebuild()
    return { 'imported': imported, 'read': read, 'runs': index.count() }


# ---------------------------------------------------------------------------
# Optimize for this PC
# ---------------------------------------------------------------------------

def _bad_params( field, reason ):
    import betautils_protocol as bu_proto
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


def optimizer_for( server ):
    if getattr( server, 'optimizer', None ) is None:
        import betautils_optimize as bu_optimize
        server.optimizer = bu_optimize.Optimizer( server.send_event, server._logger )
    return server.optimizer


def _busy_job( server ):
    jobs = getattr( server, 'jobs', None )
    return jobs.busy_job() if jobs is not None else None


def machine_busy( run=None ):
    """True when tune_machine's preflight says the machine is busy (load, GPU use, a command-line run)."""
    import os
    import subprocess
    import sys
    script = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'tools', 'bench', 'tune_machine.py' )
    run = run or ( lambda command: subprocess.run( command, stdin=subprocess.DEVNULL, capture_output=True,
                                                   timeout=60 ).returncode )
    return run( [ sys.executable, script, 'preflight' ] ) == 3


def _settings_snapshot( config ):
    import betautils_optimize as bu_optimize
    snapshot = current_settings( config )
    snapshot.update( bu_optimize.encode_settings( config ) )
    return snapshot


def perf_optimize( server, data ):
    """
    data: { full?: bool, stages?: [ stage ], video?: absolute path, force?: bool }
    Starts the measurements; progress and the proposal come as
    perf.optimize_progress events. Answers the plan.
    """
    import os
    import time
    import betaconfig
    import betaconst
    import betautils_optimize as bu_optimize
    import betautils_protocol as bu_proto
    full = data.get( 'full', False )
    force = data.get( 'force', False )
    video = data.get( 'video' )
    stages = data.get( 'stages' )
    for name, value in ( ( 'full', full ), ( 'force', force ) ):
        if not isinstance( value, bool ):
            raise _bad_params( name, bu_proto.REASON_TYPE )
    if video is not None and ( not isinstance( video, str ) or not os.path.isabs( video ) ):
        raise _bad_params( 'video', bu_proto.REASON_VALUE )
    if video is not None and not os.path.isfile( video ):
        raise bu_proto.ProtocolError( 'job.no_such_file', { 'index': 0 } )
    if stages is not None and ( not isinstance( stages, list )
                                or not all( stage in bu_optimize.STAGES for stage in stages ) ):
        raise _bad_params( 'stages', bu_proto.REASON_VALUE )
    busy = _busy_job( server )
    if busy is not None:
        raise bu_proto.ProtocolError( 'perf.busy', { 'job': busy } )
    optimizer = optimizer_for( server )
    if optimizer.running:
        raise bu_proto.ProtocolError( 'perf.optimize_running', {} )
    if not force and machine_busy():
        raise bu_proto.ProtocolError( 'perf.machine_busy', {} )
    facts, fingerprint = machine()
    index = shared_index()
    optimized = index.last_optimization( fingerprint )
    evidence = perf_evidence( server, {} )
    planned = bu_optimize.plan( evidence, betaconfig, optimized['settings'] if optimized else None,
                                full=full, stages=stages )
    out_dir = os.path.join( betaconst.benchmark_dir, 'optimize-%s'%( time.strftime( '%Y%m%d-%H%M%S' ), ) )

    def finished( proposal ):
        import betaconfig as config
        index.record_optimization( optimizer.last['at'], fingerprint, _settings_snapshot( config ), proposal, False )
        return bu_optimize.proposal_rows( proposal, config )

    optimizer.start( planned, out_dir, video=video, on_finished=finished )
    return { 'plan': planned['stages'], 'narrow': planned['narrow'] }


def perf_optimize_cancel( server, _data ):
    return { 'cancelled': optimizer_for( server ).cancel() }


def perf_optimize_apply( server, _data ):
    """Write the last finished optimization's proposal through the settings registry."""
    import betaconfig
    import betautils_optimize as bu_optimize
    import betautils_protocol as bu_proto
    import betautils_serve_config as bu_serve_config
    optimizer = optimizer_for( server )
    last = optimizer.last
    if not last or last['state'] != bu_optimize.FINISHED:
        raise bu_proto.ProtocolError( 'perf.no_proposal', {} )
    busy = _busy_job( server )
    if busy is not None:
        raise bu_proto.ProtocolError( 'config.busy', { 'job': busy } )
    rows = bu_optimize.proposal_rows( last['proposal'], betaconfig )
    updates = { tuple( row['setting'].split( '.' ) ): row['value'] for row in rows }
    answer = bu_serve_config._edit( {}, updates, [] ) if updates else { 'moves': [], 'may_move': [],
                                                                       'written': False, 'dry_run': False }
    _facts, fingerprint = machine()
    shared_index().record_optimization( last['at'], fingerprint, _settings_snapshot( betaconfig ),
                                        last['proposal'], True )
    answer['applied'] = [ row['setting'] for row in rows ]
    return answer


HANDLERS = {
    'perf.evidence': perf_evidence,
    'perf.rebuild': perf_rebuild,
    'perf.optimize': perf_optimize,
    'perf.optimize_cancel': perf_optimize_cancel,
    'perf.optimize_apply': perf_optimize_apply,
}
