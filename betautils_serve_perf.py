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


HANDLERS = {
    'perf.evidence': perf_evidence,
    'perf.rebuild': perf_rebuild,
}
