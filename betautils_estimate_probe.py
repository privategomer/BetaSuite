"""
betautils_estimate_probe.py - a short measurement of this machine's
detect and render speed, for the first estimate before any run.

With no comparable run in the history, job.estimate has nothing to go
on. The probe measures the two rates an estimate needs on a few seconds
of the chosen file, through the code a run uses:

  detect   the selected detector on the frames a run would sample, at
           every detection size, in its batch size; decode included,
           model loading not
  render   betautils_render.render_video on the same slice with no
           boxes, into a scratch folder that is removed afterwards

Nothing is written to the caches, the history, the stats or the output
folder. The rates are rougher than a run's (a short slice pays the
encoder's start-up cost, and censoring nothing skips drawing), so they
stand in only until the first real run gives the history its own.
"""

import os
import shutil
import tempfile
import time

PROBE_SECONDS = 3.0


def slice_for( duration, seconds=PROBE_SECONDS ):
    """( offset, length ) of the slice to probe: a tenth of the way in, never past the end."""
    duration = max( 0.0, float( duration or 0 ) )
    length = min( seconds, duration )
    offset = min( duration * 0.1, max( 0.0, duration - length ) )
    return offset, length


def detect_rate( source, session, sizes, sample_fps, batch_size, offset, length, detector, clock=time.perf_counter ):
    """Sampled frames per second over every size, or None when nothing was sampled."""
    import betautils_video as bu_video
    frames = 0
    started = clock()
    for size in sizes:
        batch, times = [], []
        for _index, timestamp, frame in bu_video.iter_sampled_frames(
                source.cap, source.vid_fps, sample_fps, offset_seconds=offset,
                num_frames=source.num_frames, max_seconds=length ):
            batch.append( frame )
            times.append( timestamp )
            if len( batch ) >= batch_size:
                detector.raw_boxes_for_imgs( batch, size, session, times )
                frames += len( batch )
                batch, times = [], []
        if batch:
            detector.raw_boxes_for_imgs( batch, size, session, times )
            frames += len( batch )
    elapsed = clock() - started
    return frames / elapsed if frames and elapsed > 0 else None


def render_rate( source, preset, offset, length, logger, clock=time.perf_counter, render=None ):
    """Pixel-frames per second for a render of the slice with nothing to censor, or None."""
    import betautils_render as bu_render
    render = render or bu_render.render_video
    folder = tempfile.mkdtemp( prefix='betasuite-probe-' )
    try:
        started = clock()
        render( source, [], os.path.join( folder, 'probe.mkv' ), os.path.join( folder, 'probe.mp4' ), preset,
                True, False, offset, length, logger )
        elapsed = clock() - started
    finally:
        shutil.rmtree( folder, ignore_errors=True )
    frames = int( round( length * float( source.vid_fps or 0 ) ) )
    pixels = int( source.width or 0 ) * int( source.height or 0 )
    return frames * pixels / elapsed if frames and pixels and elapsed > 0 else None


def probe( path, file_hash, session, ctx, logger ):
    """
    { detect, render } rates measured on path, with the settings context
    job.estimate uses (betautils_serve_estimate.context). Either is None
    when it could not be measured.
    """
    import betautils_detector as bu_detector
    import betautils_video as bu_video
    source = bu_video.open_video_source( path, file_hash, logger )
    try:
        duration = source.num_frames / source.vid_fps if source.vid_fps else 0
        offset, length = slice_for( duration )
        if length <= 0:
            return { 'detect': None, 'render': None }
        backend = ctx['backend']
        detect = detect_rate( source, session, ctx['sizes'], ctx['sample_fps'],
                              max( 1, bu_detector.get_nn_batch_size( backend ) ), offset, length,
                              bu_detector.get_detector( backend ) )
        render = render_rate( source, ctx['identity']['preset'], offset, length, logger )
        return { 'detect': detect, 'render': render }
    finally:
        source.release()
