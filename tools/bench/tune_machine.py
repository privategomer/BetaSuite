#!/usr/bin/env python3
"""
tune_machine.py - measure this machine's hardware settings and propose
values for its machine config (../resources/betaconfig_machine.py).

Run it through tools/bench/tune_machine.sh, which logs everything, asks
before it starts if the machine looks busy, and asks before writing
anything. Directly:

    python tools/bench/tune_machine.py preflight
    python tools/bench/tune_machine.py measure --out-dir DIR [--video PATH] [--skip encode scan]

STAGES (each one logged, timed, and skipped with a warning if it fails)
    inventory   CPU, memory, GPU and driver, ffmpeg's hardware encoders,
                onnxruntime's providers, OpenCV's decode support, and the
                settings in effect now
    detect      detection batch size: ms per frame at batch 1..16 on the
                same frames, and whether the detections stay the same
    prefetch    the real detection loop (decode, batch, model) over a
                stretch of video, with frames decoded 0, 1, 2 and 4
                batches ahead on a second thread
    render      the render's decode + encode (the real chunk encode
                command, the current encoder) at several worker counts,
                each with OpenCV's own decoder threads and with the cores
                shared out, sampling memory: frames per second, and memory
                per worker per megapixel
    encode      `betabench encode`: the current encoder against faster
                ones, quality checked frame by frame
    scan        `betabench concurrency`'s scan-worker measurement

Nothing real is read from or written to the caches; encodes go to a
scratch folder that is removed. The proposal is written as
proposal.json (values and reasons) and proposal.args (one NAME=VALUE per
line, what tools/setup/machine_config.py set takes).
"""

import argparse
import json
import os
import sys
import threading
import time

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )
sys.path.insert( 0, os.path.join( REPO_ROOT, 'tools', 'bench' ) )

SUMMARY_START = '=== tune_machine: summary ==='
SUMMARY_END = '=== end of tune_machine summary ==='
STAGES = ( 'inventory', 'detect', 'prefetch', 'render', 'encode', 'scan' )

# A setting is only proposed when it beats the current value by more
# than this; below it the difference is run-to-run noise.
NOISE = 0.03
# Measured memory per worker is the decode+encode floor; the real render
# also composites the censor, so the proposal adds this much.
MEMORY_MARGIN = 1.25


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------

def _status_kb( pid, field, proc_root='/proc' ):
    try:
        with open( os.path.join( proc_root, str( pid ), 'status' ), 'r', encoding='ascii' ) as handle:
            for line in handle:
                if line.startswith( field + ':' ):
                    return int( line.split()[1] )
    except ( OSError, ValueError, IndexError ):
        return 0
    return 0


def child_pids( pid, proc_root='/proc' ):
    """Direct children of pid (the render's ffmpeg encoders are ours)."""
    children = []
    try:
        names = os.listdir( proc_root )
    except OSError:
        return children
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open( os.path.join( proc_root, name, 'stat' ), 'r', encoding='ascii' ) as handle:
                stat = handle.read()
            # pid (comm) state ppid ...; comm may contain spaces, so split after ')'
            ppid = int( stat.rsplit( ')', 1 )[1].split()[1] )
        except ( OSError, ValueError, IndexError ):
            continue
        if ppid == pid:
            children.append( int( name ) )
    return children


def tree_rss_mb( pid=None, proc_root='/proc' ):
    """Resident memory of this process and its direct children, in MB."""
    pid = os.getpid() if pid is None else pid
    total = _status_kb( pid, 'VmRSS', proc_root )
    for child in child_pids( pid, proc_root ):
        total += _status_kb( child, 'VmRSS', proc_root )
    return total / 1024.0


class MemorySampler( object ):
    """Peak resident memory of this process tree, and the lowest MemAvailable, while it runs."""

    def __init__( self, interval=0.2, rss=tree_rss_mb, available=None ):
        self.interval = interval
        self._rss = rss
        self._available = available
        self._stop = threading.Event()
        self.baseline_mb = None
        self.peak_mb = None
        self.min_available_mb = None

    def _available_mb( self ):
        if self._available is not None:
            return self._available()
        import betautils_render as bu_render
        value = bu_render.available_memory_bytes()
        return None if value is None else value / 2**20

    def _sample( self ):
        self.peak_mb = max( self.peak_mb or 0.0, self._rss() )
        available = self._available_mb()
        if available is not None:
            self.min_available_mb = ( available if self.min_available_mb is None
                                      else min( self.min_available_mb, available ) )

    def _run( self ):
        while not self._stop.wait( self.interval ):
            self._sample()

    def __enter__( self ):
        self.baseline_mb = self._rss()
        self.peak_mb = self.baseline_mb
        self._sample()
        self._thread = threading.Thread( target=self._run, name='memory-sampler', daemon=True )
        self._thread.start()
        return self

    def __exit__( self, *exc ):
        self._stop.set()
        self._thread.join()
        self._sample()
        return False

    def growth_mb( self ):
        return max( 0.0, ( self.peak_mb or 0.0 ) - ( self.baseline_mb or 0.0 ) )


# ---------------------------------------------------------------------------
# Choosing
# ---------------------------------------------------------------------------

def fastest_within_noise( rows, speed, cost, noise=NOISE ):
    """
    The row with the lowest `cost` among those within `noise` of the
    best `speed` (higher is faster). A near-tie is noise, and the cheaper
    setting (fewer workers, smaller batch, less memory) leaves more for
    everything else.
    """
    rows = [ row for row in rows if speed( row ) ]
    if not rows:
        return None
    best = max( speed( row ) for row in rows )
    return min( ( row for row in rows if speed( row ) >= best * ( 1.0 - noise ) ), key=cost )


def decoder_mode_choice( rows, noise=NOISE ):
    """
    Of one worker count's two rows (OpenCV's own decoder threads, None,
    and the cores shared out, 0), the one to use: shared out when it is
    clearly faster, or as fast and clearly lighter on memory; otherwise
    OpenCV's own, which is what every render did before the setting.
    """
    default = next( ( r for r in rows if r['decode_threads_setting'] is None ), None )
    shared = next( ( r for r in rows if r['decode_threads_setting'] == 0 ), None )
    if shared is None or default is None:
        return shared or default
    if shared['fps'] > default['fps'] * ( 1.0 + noise ):
        return shared
    if shared['fps'] >= default['fps'] * ( 1.0 - noise ) \
            and shared['peak_growth_mb'] < default['peak_growth_mb'] * ( 1.0 - noise ):
        return shared
    return default


def mb_per_megapixel_proposal( rows, margin=MEMORY_MARGIN ):
    """The highest per-worker memory measured, per megapixel, with margin, rounded up to 10."""
    measured = [ row['mb_per_megapixel'] for row in rows if row.get( 'mb_per_megapixel' ) ]
    if not measured:
        return None
    value = max( measured ) * margin
    return int( max( 50, -( -value // 10 ) * 10 ) )


def auto_render_workers( cpu_count ):
    """render_workers = 0's count before the memory cap (betautils_render.resolve_worker_count)."""
    return max( 1, min( ( cpu_count or 1 ) // 2 or 1, 8 ) )


def render_worker_counts( cpu_count, nvenc_sessions=None ):
    """Worker counts worth timing: a spread up to the core count, within NVENC's session limit."""
    counts = { 1, 2, 4, 6, 8, 10, 12, 16, auto_render_workers( cpu_count ) }
    counts = [ n for n in sorted( counts ) if n <= max( 1, cpu_count or 1 ) ]
    if nvenc_sessions is not None:
        counts = [ n for n in counts if n <= max( 1, nvenc_sessions ) ]
    return counts or [ 1 ]


def _rate( milliseconds ):
    """Items per millisecond; a cost too small to time counts as very fast, not as a crash."""
    return 1.0 / max( milliseconds, 1e-9 )


def batch_sizes_to_try( current ):
    return sorted( { 1, 2, 4, 8, 16, int( current or 1 ) } )


def prefetch_depths_to_try( batch, current ):
    return sorted( { 0, batch, 2 * batch, 4 * batch, int( current or 0 ) } )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _run_quiet( command, timeout=30 ):
    import subprocess
    try:
        done = subprocess.run( command, capture_output=True, text=True, timeout=timeout )
    except ( OSError, subprocess.SubprocessError ) as err:
        return None, str( err )
    return done.returncode, ( done.stdout or '' ) + ( done.stderr or '' )


def nvidia_smi( fields ):
    """{ field: value } for the first GPU, or {} without nvidia-smi."""
    code, text = _run_quiet( [ 'nvidia-smi', '--query-gpu=%s'%( ','.join( fields ), ),
                               '--format=csv,noheader,nounits' ] )
    if code != 0 or not text:
        return {}
    values = [ part.strip() for part in text.strip().splitlines()[0].split( ',' ) ]
    return dict( zip( fields, values ) )


def stage_inventory( context, logger ):
    import platform
    import cv2
    import betaconfig
    import betautils_machine as bu_machine
    import betabench

    result = { 'platform': platform.platform(), 'python': platform.python_version(),
               'cpu_count': os.cpu_count() }
    try:
        with open( '/proc/cpuinfo', 'r', encoding='ascii', errors='replace' ) as handle:
            for line in handle:
                if line.startswith( 'model name' ):
                    result['cpu_model'] = line.split( ':', 1 )[1].strip()
                    break
    except OSError:
        pass
    result['memory_total_mb'] = _status_meminfo_mb( 'MemTotal' )
    result['memory_available_mb'] = _status_meminfo_mb( 'MemAvailable' )
    result['swap_total_mb'] = _status_meminfo_mb( 'SwapTotal' )
    result['gpu'] = nvidia_smi( [ 'name', 'driver_version', 'memory.total', 'memory.used',
                                  'utilization.gpu', 'encoder.stats.sessionCount' ] )
    code, text = _run_quiet( [ 'ffmpeg', '-hide_banner', '-version' ] )
    result['ffmpeg'] = text.splitlines()[0] if code == 0 and text else None
    result['ffmpeg_hardware_encoders'] = [
        name for name in ( 'h264_nvenc', 'hevc_nvenc', 'av1_nvenc', 'h264_qsv', 'h264_vaapi',
                           'h264_amf', 'h264_videotoolbox' ) if betabench._ffmpeg_lists_encoder( name ) ]
    try:
        import onnxruntime
        result['onnxruntime'] = onnxruntime.__version__
        result['onnxruntime_providers'] = onnxruntime.get_available_providers()
    except Exception as err:   # an install problem is a finding here, not a crash
        result['onnxruntime'] = 'not importable: %s'%( err, )
    build = cv2.getBuildInformation()
    result['opencv'] = cv2.__version__
    result['opencv_decoder_threads_settable'] = hasattr( cv2, 'CAP_PROP_N_THREADS' )
    result['opencv_video_io'] = [ line.strip() for line in build.splitlines()
                                  if any( key in line for key in ( 'FFMPEG:', 'GStreamer:', 'VA:', 'Media SDK' ) ) ]
    loaded = bu_machine.LOADED
    result['machine_config'] = loaded.describe() if loaded is not None else None
    result['in_effect'] = { name: getattr( betaconfig, name, None ) for name in (
        'gpu_enabled', 'detection_prefetch_frames', 'render_workers', 'render_decode_threads',
        'render_worker_mb_per_megapixel', 'render_memory_reserve_mb', 'render_chunk_seconds',
        'encode_video_codec', 'encode_preset', 'encode_crf', 'structure_scan_workers' ) }
    result['in_effect']['nn_batch_size'] = context['batch_now']

    logger.info( "machine: %s, %s core(s), %s MB memory (%s available), %s MB swap"%(
        result.get( 'cpu_model', '?' ), result['cpu_count'], result['memory_total_mb'],
        result['memory_available_mb'], result['swap_total_mb'] ) )
    gpu = result['gpu']
    if gpu:
        logger.info( "GPU: %s, driver %s, %s MB (%s MB in use, %s%% busy, %s NVENC session(s) open)"%(
            gpu.get( 'name' ), gpu.get( 'driver_version' ), gpu.get( 'memory.total' ),
            gpu.get( 'memory.used' ), gpu.get( 'utilization.gpu' ),
            gpu.get( 'encoder.stats.sessionCount' ) ) )
    else:
        logger.info( "GPU: no nvidia-smi, so no NVIDIA GPU details" )
    logger.info( "ffmpeg: %s; hardware encoders: %s"%(
        result['ffmpeg'], ', '.join( result['ffmpeg_hardware_encoders'] ) or 'none' ) )
    logger.info( "onnxruntime %s, providers %s"%( result.get( 'onnxruntime' ),
                                                  result.get( 'onnxruntime_providers' ) ) )
    logger.info( "OpenCV %s, decoder threads settable: %s; %s"%(
        result['opencv'], result['opencv_decoder_threads_settable'], '; '.join( result['opencv_video_io'] ) ) )
    logger.info( result['machine_config'] or 'machine config: unknown' )
    logger.info( "in effect: %s"%( ', '.join( "%s=%r"%( k, v ) for k, v in result['in_effect'].items() ), ) )
    return result


def _status_meminfo_mb( field ):
    try:
        with open( '/proc/meminfo', 'r', encoding='ascii' ) as handle:
            for line in handle:
                if line.startswith( field + ':' ):
                    return int( line.split()[1] ) // 1024
    except ( OSError, ValueError, IndexError ):
        return None
    return None


def _video_info( path ):
    import cv2
    capture = cv2.VideoCapture( path )
    try:
        return { 'path': path,
                 'width': int( capture.get( cv2.CAP_PROP_FRAME_WIDTH ) ),
                 'height': int( capture.get( cv2.CAP_PROP_FRAME_HEIGHT ) ),
                 'fps': capture.get( cv2.CAP_PROP_FPS ) or 30.0,
                 'frames': int( capture.get( cv2.CAP_PROP_FRAME_COUNT ) ) }
    finally:
        capture.release()


def _labels( boxes ):
    counts = {}
    for box in boxes:
        label = box.get( 'class_id', box.get( 'label' ) )
        counts[label] = counts.get( label, 0 ) + 1
    return counts


def _detector( context ):
    if 'detector' not in context:
        import betautils_detector as bu_detector
        backend = bu_detector.selected_backend_name()
        module = bu_detector.get_detector( backend )
        context['detector'] = ( backend, module, module.get_session(),
                                bu_detector.get_picture_sizes( backend )[0] )
    return context['detector']


def stage_detect( context, logger ):
    import cv2
    import betaconfig
    import betautils_detector as bu_detector
    import betautils_video as bu_video

    video = context['video']
    backend, module, session, size = _detector( context )
    capture = cv2.VideoCapture( video['path'] )
    start = video['frames'] / video['fps'] * 0.25
    frames, times = [], []
    try:
        for _index, timestamp, frame in bu_video.iter_sampled_frames(
                capture, video['fps'], betaconfig.video_censor_fps, offset_seconds=start ):
            frames.append( frame )
            times.append( timestamp )
            if len( frames ) >= context['args'].detect_frames:
                break
    finally:
        capture.release()
    if not frames:
        raise RuntimeError( "no frames decoded from %s"%( video['path'], ) )
    logger.info( "detect: %s at size %d, %d frames from %.0fs in, providers %s"%(
        backend, size, len( frames ), start, bu_detector.session_provider_summary( session ) ) )

    rows = []
    reference = None
    for batch in batch_sizes_to_try( context['batch_now'] ):
        module.raw_boxes_for_imgs( frames[ : batch ], size, session, times[ : batch ] )   # warm-up
        best = None
        for _repeat in range( 2 ):
            boxes = []
            started = time.perf_counter()
            for offset in range( 0, len( frames ), batch ):
                boxes += module.raw_boxes_for_imgs( frames[ offset : offset + batch ], size, session,
                                                    times[ offset : offset + batch ] )
            elapsed = time.perf_counter() - started
            best = elapsed if best is None else min( best, elapsed )
        labels = _labels( boxes )
        reference = labels if reference is None else reference
        gpu = nvidia_smi( [ 'memory.used' ] )
        row = { 'batch': batch, 'ms_per_frame': round( 1000.0 * best / len( frames ), 3 ),
                'detections': len( boxes ), 'labels': labels, 'same_as_batch_1': labels == reference,
                'gpu_memory_used_mb': gpu.get( 'memory.used' ) }
        rows.append( row )
        logger.info( "  batch %2d: %7.2f ms/frame, %d detection(s)%s, GPU memory in use %s MB"%(
            batch, row['ms_per_frame'], row['detections'],
            '' if row['same_as_batch_1'] else ' (DIFFERS from batch 1)', row['gpu_memory_used_mb'] ) )
    if any( not row['same_as_batch_1'] for row in rows ):
        logger.warning( "  detections differ between batch sizes on the same frames; batched GPU "
                        "kernels can round differently. Only matching sizes are proposed" )
    candidates = [ row for row in rows if row['same_as_batch_1'] ]
    choice = fastest_within_noise( candidates, lambda r: _rate( r['ms_per_frame'] ), lambda r: r['batch'] )
    current = next( ( row for row in rows if row['batch'] == context['batch_now'] ), None )
    result = { 'backend': backend, 'size': size, 'frames': len( frames ), 'rows': rows,
               'chosen': choice['batch'] if choice else None }
    if choice and current and choice['batch'] != current['batch'] \
            and choice['ms_per_frame'] < current['ms_per_frame'] * ( 1.0 - NOISE ):
        context['proposal'][ 'detector_backend.%s.nn_batch_size'%( backend, ) ] = {
            'value': choice['batch'],
            'reason': "%.2f ms/frame against %.2f at batch %d"%(
                choice['ms_per_frame'], current['ms_per_frame'], current['batch'] ) }
    context['batch_best'] = choice['batch'] if choice else context['batch_now']
    logger.info( "  best: batch %s"%( result['chosen'], ) )
    return result


def _detect_pass( context, depth, seconds ):
    """The real detection loop over `seconds` of video: decode, batch, model."""
    import cv2
    import betaconfig
    import betautils_video as bu_video
    _backend, module, session, size = _detector( context )
    video = context['video']
    batch = context.get( 'batch_best' ) or context['batch_now']
    start = video['frames'] / video['fps'] * 0.25
    capture = cv2.VideoCapture( video['path'] )
    samples = bu_video.prefetched( bu_video.iter_sampled_frames(
        capture, video['fps'], betaconfig.video_censor_fps, offset_seconds=start,
        max_seconds=seconds ), depth )
    frames, times, count = [], [], 0
    started = time.perf_counter()
    try:
        for _index, timestamp, frame in samples:
            frames.append( frame )
            times.append( timestamp )
            count += 1
            if len( frames ) >= batch:
                module.raw_boxes_for_imgs( frames, size, session, times )
                frames, times = [], []
        if frames:
            module.raw_boxes_for_imgs( frames, size, session, times )
    finally:
        samples.close()
        capture.release()
    return time.perf_counter() - started, count


def stage_prefetch( context, logger ):
    seconds = context['args'].prefetch_seconds
    batch = context.get( 'batch_best' ) or context['batch_now']
    depths = prefetch_depths_to_try( batch, context['prefetch_now'] )
    logger.info( "prefetch: the detection loop over %gs of video at batch %d, frames decoded ahead: %s"%(
        seconds, batch, depths ) )
    _detect_pass( context, 0, min( seconds, 5 ) )   # warm the page cache and the model
    rows = []
    # 0 first and last, so drift over the stage shows up as two different 0 rows.
    for depth in depths + [ 0 ]:
        wall, samples = _detect_pass( context, depth, seconds )
        rows.append( { 'depth': depth, 'seconds': round( wall, 3 ), 'samples': samples,
                       'ms_per_sample': round( 1000.0 * wall / max( 1, samples ), 3 ) } )
        logger.info( "  %3d ahead: %6.2fs for %d samples, %.2f ms/sample"%(
            depth, wall, samples, rows[-1]['ms_per_sample'] ) )
    by_depth = {}
    for row in rows:
        kept = by_depth.get( row['depth'] )
        if kept is None or row['ms_per_sample'] < kept['ms_per_sample']:
            by_depth[ row['depth'] ] = row
    choice = fastest_within_noise( list( by_depth.values() ), lambda r: _rate( r['ms_per_sample'] ),
                                   lambda r: r['depth'] )
    current = by_depth.get( context['prefetch_now'] )
    if choice and current and choice['depth'] != current['depth'] \
            and choice['ms_per_sample'] < current['ms_per_sample'] * ( 1.0 - NOISE ):
        context['proposal']['detection_prefetch_frames'] = {
            'value': choice['depth'],
            'reason': "%.2f ms/sample against %.2f with %d" %(
                choice['ms_per_sample'], current['ms_per_sample'], current['depth'] ) }
    off = by_depth.get( 0 )
    if off and choice:
        logger.info( "  best: %d ahead, %.0f%% less detection time than none"%(
            choice['depth'], 100.0 * ( 1.0 - choice['ms_per_sample'] / off['ms_per_sample'] ) ) )
    return { 'batch': batch, 'seconds': seconds, 'rows': rows,
             'chosen': choice['depth'] if choice else None }


def stage_render( context, logger ):
    import shutil
    import tempfile
    import betaconst
    import betautils_render as bu_render
    import betabench

    args = context['args']
    video = context['video']
    width, height, fps = video['width'], video['height'], video['fps']
    megapixels = width * height / 1e6
    candidate = betabench._baseline_encoder()
    cpu_count = os.cpu_count() or 1
    sessions = None
    if candidate['codec'] in bu_render.NVENC_CODECS:
        sessions, reason = betabench.probe_nvenc_sessions( max( render_worker_counts( cpu_count ) ) )
        logger.info( "render: NVENC opened %d session(s) at once%s"%(
            sessions, '' if not reason else ' (%s)'%( reason, ) ) )
        if sessions == 0:
            raise RuntimeError( "the configured encoder %s cannot open a session"%( candidate['codec'], ) )
    counts = render_worker_counts( cpu_count, sessions )
    reserve_mb = getattr( context['betaconfig'], 'render_memory_reserve_mb', 2048 )
    logger.info( "render: %s, %dx%d (%.2f MP), %s:%s:%s, %gs of video per worker, worker counts %s"%(
        os.path.basename( video['path'] ), width, height, megapixels, candidate['codec'],
        candidate['preset'], candidate['quality'], args.render_seconds, counts ) )

    scratch = tempfile.mkdtemp( prefix='tune-machine-render-',
                                dir=betaconst.benchmark_dir if os.path.isdir( betaconst.benchmark_dir ) else None )
    rows = []
    # The config's own estimate until something is measured, so the first
    # big step cannot run the machine out of memory before any data exists.
    model_mb = megapixels * ( getattr( context['betaconfig'], 'render_worker_mb_per_megapixel', None ) or 200 )
    per_worker_mb = None
    try:
        for workers in counts:
            available = bu_render.available_memory_bytes()
            estimate = per_worker_mb or model_mb
            if available is not None:
                needed = workers * estimate + reserve_mb
                if needed > available / 2**20:
                    logger.warning( "  %d worker(s) skipped: about %.0f MB needed (%s), %.0f MB available"%(
                        workers, needed, 'measured' if per_worker_mb else 'the config estimate',
                        available / 2**20 ) )
                    break
            stop = False
            for mode in ( None, 0 ):
                threads = None if mode is None else max( 1, cpu_count // workers )
                segments = betabench.encode_segments( video['frames'], fps, args.render_seconds, workers )
                out_dir = os.path.join( scratch, 'w%02d_%s'%( workers, 'default' if mode is None else 'shared' ) )
                os.makedirs( out_dir )
                with MemorySampler() as memory:
                    wall, results = betabench._run_segments(
                        video['path'], segments, width, height, fps, candidate,
                        bu_render._encoder_threads( workers ), out_dir, decoder_threads=threads )
                shutil.rmtree( out_dir, ignore_errors=True )
                frames = sum( r['frames'] for r in results )
                errors = [ r['error'] for r in results if r['error'] ]
                growth = memory.growth_mb()
                row = { 'workers': workers, 'decode_threads_setting': mode,
                        'decode_threads': threads, 'frames': frames, 'seconds': round( wall, 3 ),
                        'fps': round( frames / wall, 2 ) if wall and not errors else 0.0,
                        'peak_growth_mb': round( growth ), 'min_available_mb':
                            None if memory.min_available_mb is None else round( memory.min_available_mb ),
                        'mb_per_megapixel': round( growth / workers / megapixels, 1 ) if not errors else None,
                        'error': errors[0] if errors else None }
                rows.append( row )
                logger.info( "  %2d worker(s), decoder threads %-15s %6.1f fr/s   +%5.0f MB "
                             "(%4.0f MB/MP/worker)%s"%(
                                 workers, "OpenCV's own" if mode is None else '%d each'%( threads, ),
                                 row['fps'], growth, row['mb_per_megapixel'] or 0,
                                 '   ERROR %s'%( row['error'], ) if row['error'] else '' ) )
                if not errors:
                    per_worker_mb = max( per_worker_mb or 0, growth / workers )
                if memory.min_available_mb is not None and memory.min_available_mb < reserve_mb:
                    logger.warning( "  memory available fell to %.0f MB, under the %d MB reserve; "
                                    "no more workers tried"%( memory.min_available_mb, reserve_mb ) )
                    stop = True
            if stop:
                break
    finally:
        shutil.rmtree( scratch, ignore_errors=True )

    good = [ row for row in rows if not row['error'] ]
    choice = fastest_within_noise( good, lambda r: r['fps'],
                                   lambda r: ( r['workers'], r['peak_growth_mb'] ) )
    if choice:
        choice = decoder_mode_choice( [ r for r in good if r['workers'] == choice['workers'] ] ) or choice
    result = { 'video': video, 'encoder': candidate, 'rows': rows, 'nvenc_sessions': sessions,
               'chosen': choice }
    if not choice:
        return result
    now_workers = context['betaconfig'].render_workers or auto_render_workers( cpu_count )
    now_mode = getattr( context['betaconfig'], 'render_decode_threads', None )
    current = next( ( r for r in good if r['workers'] == now_workers
                      and r['decode_threads_setting'] == ( 0 if now_mode == 0 else None ) ), None )
    better = current is None or choice['fps'] > current['fps'] * ( 1.0 + NOISE )
    if better:
        wanted_workers = 0 if choice['workers'] == auto_render_workers( cpu_count ) else choice['workers']
        if wanted_workers != context['betaconfig'].render_workers:
            context['proposal']['render_workers'] = {
                'value': wanted_workers,
                'reason': "%.1f fr/s at %d worker(s)%s"%( choice['fps'], choice['workers'],
                          '' if current is None else ' against %.1f at %d'%( current['fps'], current['workers'] ) ) }
    if choice['decode_threads_setting'] != now_mode:
        other = next( ( r for r in good if r['workers'] == choice['workers']
                        and r['decode_threads_setting'] != choice['decode_threads_setting'] ), None )
        context['proposal']['render_decode_threads'] = {
            'value': choice['decode_threads_setting'],
            'reason': "at %d worker(s): %.1f fr/s and +%d MB, against %.1f fr/s and +%d MB with %s"%(
                choice['workers'], choice['fps'], choice['peak_growth_mb'],
                other['fps'] if other else 0, other['peak_growth_mb'] if other else 0,
                "OpenCV's own threads" if choice['decode_threads_setting'] == 0 else 'the cores shared out' ) }
    same_mode = [ r for r in good if r['decode_threads_setting'] == choice['decode_threads_setting'] ]
    per_megapixel = mb_per_megapixel_proposal( same_mode )
    now_per_megapixel = getattr( context['betaconfig'], 'render_worker_mb_per_megapixel', 200 )
    if per_megapixel and per_megapixel != now_per_megapixel:
        context['proposal']['render_worker_mb_per_megapixel'] = {
            'value': per_megapixel,
            'reason': "highest measured %.0f MB/MP per worker, plus %d%% for censoring"%(
                max( r['mb_per_megapixel'] for r in same_mode ), round( 100 * ( MEMORY_MARGIN - 1 ) ) ) }
    logger.info( "  best: %d worker(s), decoder threads %s, %.1f fr/s; memory per worker up to %s MB/MP"%(
        choice['workers'], "OpenCV's own" if choice['decode_threads_setting'] is None else 'shared out',
        choice['fps'], max( ( r['mb_per_megapixel'] for r in same_mode ), default=None ) ) )
    return result


def stage_encode( context, logger ):
    import betabench
    args = context['args']
    chosen = ( context['results'].get( 'render' ) or {} ).get( 'chosen' ) or {}
    namespace = argparse.Namespace( video=context['video']['path'], workers=chosen.get( 'workers' ),
                                    seconds=args.encode_seconds, quality_stride=args.quality_stride,
                                    encoders=None, first_videos=None, videos_per_folder=None )
    result = betabench.bench_encode( namespace, logger ).get( 'encode' ) or {}
    choice = result.get( 'recommended' )
    rows = result.get( 'rows' ) or []
    if choice and rows and choice is not rows[0]:
        codec_changes = choice['codec'] != rows[0]['codec']
        for name, key in ( ( 'encode_video_codec', 'codec' ), ( 'encode_preset', 'preset' ),
                           ( 'encode_crf', 'quality' ) ):
            # A new codec takes all three: preset and quality are on its own scale.
            if choice[key] == rows[0][key] and not codec_changes:
                continue
            context['proposal'][name] = {
                'value': choice[key],
                'reason': "%s at %.1f fr/s matched the current encoder's quality (%.1f fr/s)"%(
                    betabench._encoder_label( choice ), choice['fps'], rows[0]['fps'] ) }
    return result


def stage_scan( context, logger ):
    import betaconfig
    import betabench
    args = context['args']
    namespace = argparse.Namespace( seconds=args.scan_seconds, max_videos=None, video=None,
                                    first_videos=None, videos_per_folder=None )
    sources = [ source['path'] for source in betabench.selected_source_videos( namespace, logger ) ]
    if not sources:
        raise RuntimeError( "no source videos to scan" )
    wanted = betabench._scan_bench_file_count( namespace, os.cpu_count() or 1 )
    with betabench._ThrowawayCaches() as caches:
        result = betabench._measure_scan_workers( namespace, logger, sources[ : wanted ], caches )
    now = getattr( betaconfig, 'structure_scan_workers', 0 ) or betabench._default_scan_workers()
    best = result.get( 'recommended_workers' )
    rows = { row['workers']: row for row in result.get( 'rows', [] ) }
    if best and best != now and best in rows and now in rows \
            and rows[best]['wall_seconds'] < rows[now]['wall_seconds'] * ( 1.0 - NOISE ):
        context['proposal']['structure_scan_workers'] = {
            'value': best, 'reason': "%.1fs against %.1fs at %d"%(
                rows[best]['wall_seconds'], rows[now]['wall_seconds'], now ) }
    return result


STAGE_FUNCTIONS = { 'inventory': stage_inventory, 'detect': stage_detect, 'prefetch': stage_prefetch,
                    'render': stage_render, 'encode': stage_encode, 'scan': stage_scan }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _logger( out_dir, console_level ):
    import betautils_log as bu_log
    return bu_log.tool_logger( 'tune_machine', log_path=os.path.join( out_dir, 'tune_machine.log' ),
                               file_level='trace', console_level=console_level )


def pick_video( path, logger ):
    """--video, else the source with the most pixels per frame (the hardest one to render)."""
    if path:
        return _video_info( path )
    import betabench
    namespace = argparse.Namespace( video=None, first_videos=None, videos_per_folder=None )
    chosen = betabench._pick_encode_video( namespace, logger )
    return _video_info( chosen ) if chosen else None


def proposal_args( proposal ):
    """One NAME=VALUE per proposed setting, as tools/setup/machine_config.py set takes them."""
    return [ "%s=%r"%( name, entry['value'] ) for name, entry in sorted( proposal.items() ) ]


def command_measure( args ):
    os.makedirs( args.out_dir, exist_ok=True )
    os.chdir( REPO_ROOT )
    logger = _logger( args.out_dir, args.console_level )
    import betaconfig
    import betautils_detector as bu_detector

    context = { 'args': args, 'betaconfig': betaconfig, 'proposal': {}, 'results': {},
                'batch_now': bu_detector.get_nn_batch_size( bu_detector.selected_backend_name() ),
                'prefetch_now': getattr( betaconfig, 'detection_prefetch_frames', 0 ) }
    stages = [ stage for stage in STAGES if stage not in ( args.skip or [] ) ]
    logger.info( "tune_machine: stages %s, output in %s"%( ', '.join( stages ), args.out_dir ) )
    context['video'] = pick_video( args.video, logger )
    if context['video'] is None and any( stage != 'inventory' for stage in stages ):
        logger.error( "no source video found; put one in ../resources/uncensored_vids or pass --video" )
        return 1
    if context['video']:
        logger.info( "video: %s, %dx%d at %.3g fps, %d frames"%(
            context['video']['path'], context['video']['width'], context['video']['height'],
            context['video']['fps'], context['video']['frames'] ) )

    timings = {}
    for stage in stages:
        logger.info( "" )
        logger.info( "--- %s ---"%( stage, ) )
        started = time.perf_counter()
        try:
            context['results'][stage] = STAGE_FUNCTIONS[stage]( context, logger )
        except Exception as err:   # one stage failing must not lose the others' numbers
            logger.exception( "%s failed: %s"%( stage, err ) )
            context['results'][stage] = { 'error': repr( err ) }
        timings[stage] = round( time.perf_counter() - started, 1 )
        logger.info( "%s took %.0fs"%( stage, timings[stage] ) )
        _write_json( os.path.join( args.out_dir, 'results.json' ),
                     { 'results': context['results'], 'stage_seconds': timings,
                       'proposal': context['proposal'] } )

    proposal = context['proposal']
    _write_json( os.path.join( args.out_dir, 'proposal.json' ), proposal )
    with open( os.path.join( args.out_dir, 'proposal.args' ), 'w', encoding='utf-8' ) as handle:
        handle.write( ''.join( line + '\n' for line in proposal_args( proposal ) ) )

    logger.info( "" )
    logger.info( SUMMARY_START )
    for stage in stages:
        failed = 'error' in ( context['results'].get( stage ) or {} )
        logger.info( "  %-9s %5.0fs%s"%( stage, timings.get( stage, 0 ), '   FAILED, see the log' if failed else '' ) )
    if proposal:
        logger.info( "  proposed for the machine config:" )
        for name, entry in sorted( proposal.items() ):
            logger.info( "    %s = %r   (%s)"%( name, entry['value'], entry['reason'] ) )
    else:
        logger.info( "  nothing to change: the settings in effect measured as fast as anything tried" )
    logger.info( SUMMARY_END )
    return 0


def _write_json( path, data ):
    temporary = path + '.tmp'
    with open( temporary, 'w', encoding='utf-8' ) as handle:
        json.dump( data, handle, indent=2, default=repr )
    os.replace( temporary, path )


def busy_reasons( load_average=None, cpu_count=None, gpu=None, running_pid=None ):
    """Why measurements taken now would be skewed, if anything is running."""
    reasons = []
    if running_pid:
        reasons.append( "betatv.py is running (pid %s)"%( running_pid, ) )
    cpu_count = cpu_count or 1
    if load_average is not None and load_average > cpu_count * 0.25:
        reasons.append( "the 1-minute load average is %.1f on %d core(s)"%( load_average, cpu_count ) )
    try:
        busy = int( float( ( gpu or {} ).get( 'utilization.gpu', 0 ) ) )
    except ValueError:
        busy = 0
    if busy >= 20:
        reasons.append( "the GPU is %d%% busy"%( busy, ) )
    return reasons


def command_preflight( args ):
    """Exit 3 when the machine looks busy, so the shell can ask before measuring."""
    os.chdir( REPO_ROOT )
    import betaconfig
    import betautils_log as bu_log
    marker = bu_log.RunMarker( bu_log.run_marker_path_for(
        getattr( betaconfig, 'stats_path', None ) or bu_log.DEFAULT_STATS_PATH ), None,
        boot_id=bu_log.current_boot_id() )
    record = marker.read()
    running_pid = record.get( 'pid' ) if record and marker.previous_run_state( record ) == 'running' else None
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = None
    reasons = busy_reasons( load, os.cpu_count(), nvidia_smi( [ 'utilization.gpu' ] ), running_pid )
    for reason in reasons:
        print( "busy: %s"%( reason, ) )
    return 3 if reasons else 0


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0],
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      epilog=__doc__.split( '\n\n', 1 )[1] )
    commands = parser.add_subparsers( dest='command', required=True )
    commands.add_parser( 'preflight', help='exit 3 if the machine looks busy' )
    measure = commands.add_parser( 'measure', help='run the stages and write the proposal' )
    measure.add_argument( '--out-dir', required=True )
    measure.add_argument( '--video', help='the video to measure on (default: the one with the most pixels)' )
    measure.add_argument( '--skip', nargs='*', choices=STAGES, default=[] )
    measure.add_argument( '--console-level', default='info',
                          choices=( 'trace', 'debug', 'info', 'warn', 'error' ) )
    measure.add_argument( '--detect-frames', type=int, default=48 )
    measure.add_argument( '--prefetch-seconds', type=float, default=40.0 )
    measure.add_argument( '--render-seconds', type=float, default=8.0 )
    measure.add_argument( '--encode-seconds', type=float, default=10.0 )
    measure.add_argument( '--quality-stride', type=int, default=5 )
    measure.add_argument( '--scan-seconds', type=float, default=20.0 )
    return parser


def main( argv=None ):
    args = build_parser().parse_args( argv )
    if args.command == 'preflight':
        return command_preflight( args )
    return command_measure( args )


if __name__ == '__main__':
    sys.exit( main() )
