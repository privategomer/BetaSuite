"""
betautils_serve_estimate.py - job.estimate: how long a job would take,
before it is submitted.

The estimate is made from the timings of earlier processed runs in the
catalog's history (betautils_estimate), matched to what this job would
run with: the settings with the job's overrides on top, read the way a
run reads them (ConfigStore.evaluate). Each file is probed for its
length, frame rate and size; a file the catalog knows is checked for
detections already cached at the job's settings, which cost no
detection time.

While a job is queued, running or paused, the live settings are that
job's, so an estimate cannot evaluate overrides. It then uses the
settings taken when the queue became busy, with only the preview
overrides (preview_mode_enabled, preview_max_seconds) applied on top.

Also answered: look_seen, whether any earlier run used the look this
job would render (its censor key), so the app can suggest a preview
the first time a look is used.

THE PROBE
    With no comparable run (a new machine, a new model), the estimate
    would be empty. So when a rate is missing and the queue is idle,
    job.estimate starts a probe (betautils_estimate_probe) on one of the
    files and answers probing: true; jobs submitted meanwhile wait for
    it, and the job.probed event says when it has ended. Its rates are
    kept for this engine's life, per settings, and fill in only where
    the history has none (basis probe). A file a probe could not measure
    is not tried again for the same settings.
"""

import os

import betautils_estimate as bu_estimate
import betautils_protocol as bu_proto
import betautils_serve_config as bu_serve_config


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


def checked_files( data ):
    """data['files'] as job.submit checks it: absolute paths to existing files."""
    files = data.get( 'files' )
    if not isinstance( files, list ):
        raise _bad_params( 'files', bu_proto.REASON_MISSING if files is None else bu_proto.REASON_TYPE )
    if not files:
        raise _bad_params( 'files', bu_proto.REASON_VALUE )
    for index, path in enumerate( files ):
        if not isinstance( path, str ) or not path:
            raise _bad_params( 'files', bu_proto.REASON_TYPE )
        if not os.path.isabs( path ):
            raise _bad_params( 'files', bu_proto.REASON_VALUE )
        if not os.path.isfile( path ):
            raise bu_proto.ProtocolError( 'job.no_such_file', { 'index': index } )
    return files


def context():
    """What an estimate needs from the live settings: read inside the run's view of them."""
    import betaconfig
    import betaconst
    import betautils_cache_paths as bu_cache
    import betautils_detector as bu_detector
    backend = bu_detector.selected_backend_name()
    return {
        'identity': bu_estimate.identity(),
        'preview': bool( getattr( betaconfig, 'preview_mode_enabled', False ) ),
        'preview_max': getattr( betaconfig, 'preview_max_seconds', 20 ),
        'sample_fps': getattr( betaconfig, 'video_censor_fps', None ),
        'sizes': list( bu_detector.get_picture_sizes( backend ) ),
        'backend': backend,
        'min_prob': getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob ),
        'censor_key': bu_cache.censor_key(),
    }


def _with_preview_overrides( base, overrides ):
    out = dict( base )
    if ( 'preview_mode_enabled', ) in overrides:
        out['preview'] = bool( overrides[( 'preview_mode_enabled', )] )
    if ( 'preview_max_seconds', ) in overrides:
        out['preview_max'] = overrides[( 'preview_max_seconds', )]
    identity = dict( out['identity'] )
    if out['preview'] != base['preview']:
        # The encode preset follows the preview switch, as betatv resolves it.
        import betaconfig
        identity['preset'] = ( getattr( betaconfig, 'preview_encode_preset', 'ultrafast' ) if out['preview']
                               else getattr( betaconfig, 'encode_preset', 'fast' ) )
    out['identity'] = identity
    return out


def _number( value ):
    try:
        number = float( value )
    except ( TypeError, ValueError ):
        return None
    return number if number > 0 else None


def _fps( text ):
    if not text:
        return None
    if '/' in text:
        top, _, bottom = text.partition( '/' )
        top, bottom = _number( top ), _number( bottom )
        return top / bottom if top and bottom else None
    return _number( text )


def stream_facts( path, probe=None ):
    """{ duration, fps, width, height } of a file's video, or None when it cannot be read."""
    import betautils_video as bu_video
    info = ( probe or bu_video.probe_stream_info )( path )
    fps = _fps( info.get( 'avg_frame_rate' ) )
    width, height = _number( info.get( 'width' ) ), _number( info.get( 'height' ) )
    duration = _number( info.get( 'duration' ) )
    if duration is None and fps and _number( info.get( 'nb_frames' ) ):
        duration = _number( info['nb_frames'] ) / fps
    if duration is None:
        duration = _number( bu_video.probe_duration_seconds( path ) )
    if not ( fps and width and height and duration ):
        return None
    return { 'duration': duration, 'fps': fps, 'width': width, 'height': height }


def cache_state( digest, facts, ctx ):
    """Whether a full run's detections are on disk already. A preview slice is taken as fresh."""
    if digest is None or ctx['preview'] or not ctx['sample_fps']:
        return bu_estimate.FRESH
    import betautils_cache_paths as bu_cache
    import betautils_hash as bu_hash
    suffix = bu_cache.preview_cache_suffix( False, 0.0, None, facts['duration'] )
    paths = [ bu_cache.box_hash_path_for( digest, size, ctx['sample_fps'], ctx['min_prob'], ctx['backend'], suffix )
              for size in ctx['sizes'] ]
    if paths and all( os.path.exists( path ) for path in paths ):
        return bu_estimate.CACHED
    if any( os.path.exists( path ) or bu_hash.has_checkpoint( path ) for path in paths ):
        return bu_estimate.PARTIAL
    return bu_estimate.FRESH


def answer( files, ctx, timings, hash_of, censor_seen, probe=None, probed=None ):
    """
    job.estimate's result, from a settings context and the history's
    timings. probed: rates a probe measured for this context, used only
    where the history has none.
    """
    measured = bu_estimate.rates( timings, ctx['identity'] )
    from_probe = False
    for rate in ( 'detect', 'render' ):
        if measured[rate] is None and probed and probed.get( rate ):
            measured[rate] = probed[rate]
            from_probe = True
    if measured['detect_runs'] or measured['render_runs']:
        basis = 'history'
    else:
        basis = 'probe' if from_probe else 'none'
    rows = []
    total = 0.0
    known = True
    for path in files:
        facts = stream_facts( path, probe )
        if facts is None:
            rows.append( { 'seconds': None, 'detect_cache': None } )
            known = False
            continue
        detect_cache = cache_state( hash_of( path ), facts, ctx )
        work = { 'seconds': bu_estimate.window_seconds( facts['duration'], ctx['preview'], ctx['preview_max'] ),
                 'fps': facts['fps'], 'width': facts['width'], 'height': facts['height'],
                 'sample_fps': ctx['sample_fps'] or 0, 'sizes': len( ctx['sizes'] ), 'detect_cache': detect_cache }
        seconds = bu_estimate.file_seconds( work, measured )
        rows.append( { 'seconds': round( seconds, 1 ) if seconds is not None else None,
                       'detect_cache': detect_cache } )
        if seconds is None:
            known = False
        else:
            total += seconds
    return {
        'files': rows,
        'seconds': round( total, 1 ) if known else None,
        'basis': basis,
        'runs': { 'detect': measured['detect_runs'], 'render': measured['render_runs'] },
        'look_seen': bool( censor_seen( ctx['censor_key'] ) ),
        'probing': False,
        'missing': [ rate for rate in ( 'detect', 'render' ) if measured[rate] is None ],
    }


# --- the probe -------------------------------------------------------------------

PROBED = {}           # identity key -> { detect, render }, measured this session
_TRIED = set()        # ( identity key, path ) a probe has been tried on
_PROBE_LOCK = None


def _key( ctx ):
    import json
    return json.dumps( ctx['identity'], sort_keys=True )


def _probe_lock():
    global _PROBE_LOCK
    if _PROBE_LOCK is None:
        import threading
        _PROBE_LOCK = threading.Lock()
    return _PROBE_LOCK


def probed_for( ctx ):
    with _probe_lock():
        return PROBED.get( _key( ctx ) )


def maybe_probe( server, jobs, files, ctx, hash_of, result, measure=None ):
    """
    Start a probe when the estimate lacks a rate no probe has given yet
    for these settings, the queue is idle and no command-line run holds
    the library. The first file not tried yet is measured. Sets
    result['probing']. job.probed { ok } says when it has ended; asking
    again then gives the measured estimate.
    """
    if jobs.probing:
        result['probing'] = True
        return
    if result['seconds'] is not None or not result['missing']:
        return
    key = _key( ctx )
    with _probe_lock():
        if key in PROBED:
            return
        path = next( ( path for path in files if ( key, path ) not in _TRIED ), None )
        if path is None:
            return
    import betautils_runlock as bu_runlock
    if bu_runlock.library_held():
        return
    measure = measure or _measure

    def work( session ):
        with _probe_lock():
            _TRIED.add( ( key, path ) )
        rates = measure( path, hash_of( path ), session, ctx, server._logger )
        if rates.get( 'detect' ) or rates.get( 'render' ):
            with _probe_lock():
                PROBED[key] = rates
            server._logger.info( "estimate probe: detect %s sampled frames/s, render %s pixel-frames/s"%(
                rates.get( 'detect' ), rates.get( 'render' ) ) )

    def done( ok ):
        server.send_event( 'job.probed', { 'ok': bool( ok and probed_for( ctx ) ) } )

    result['probing'] = jobs.start_probe( work, done )


def _measure( path, digest, session, ctx, logger ):
    import betautils_estimate_probe as bu_probe
    if digest is None:
        import betautils_hash as bu_hash
        digest = bu_hash.md5_for_file( path, 16 )
    return bu_probe.probe( path, digest, session, ctx, logger )


def job_estimate( server, data ):
    """
    data: { files: [ absolute path ], overrides?: { dotted path: value } }
    result: { files: [ { seconds, detect_cache } ], seconds, basis, runs, look_seen }
    """
    import betautils_serve_catalog as bu_serve_catalog
    import betautils_serve_job as bu_serve_job
    files = checked_files( data )
    overrides = bu_serve_config._overrides( data )
    jobs = bu_serve_job.queue_for( server )
    busy_context = jobs.estimate_context_if_busy()
    if busy_context is not None:
        ctx = _with_preview_overrides( busy_context, overrides )
    else:
        store = bu_serve_config.store()
        evaluation = store.evaluate( store.text(), overrides, extra=lambda module: context() )
        if not evaluation.ok:
            raise bu_serve_config._invalid( evaluation.errors )
        ctx = evaluation.extra

    catalog = bu_serve_catalog._open_quietly( server )
    if catalog is None:
        timings, hash_of, censor_seen = [], ( lambda path: None ), ( lambda key: False )
    else:
        timings = bu_estimate.comparable_rows( catalog.store )
        hash_of = catalog.index.location_hash

        def censor_seen( key ):
            return bool( key ) and bool( catalog.store.read( 'SELECT 1 FROM runs WHERE censor_key = ? LIMIT 1',
                                                             ( key, ) ) )
    result = answer( files, ctx, timings, hash_of, censor_seen, probed=probed_for( ctx ) )
    if busy_context is None:
        maybe_probe( server, jobs, files, ctx, hash_of, result )
    else:
        result['probing'] = jobs.probing
    return result


HANDLERS = {
    'job.estimate': job_estimate,
}
