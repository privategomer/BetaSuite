"""
betautils_render.py - Turning censorable boxes into a finished video.

The render is the slowest stage on a long video, so its shape matters:

  CHUNKED     The frame range is split into fixed-length chunks
              (betaconfig.render_chunk_seconds). A single continuous
              ffmpeg pipe cannot be resumed if it is killed - you cannot
              append more encoded frames onto a truncated file and get
              something valid - so each chunk is rendered to its own
              file and only promoted to its trusted name once ffmpeg
              exits 0, the output passes a container check, and the
              frame count matches. A restart then trusts "does this
              chunk exist" and picks up from the first missing one.

  PARALLEL    Chunks are independent by construction: each reconstructs
              its own live/pending box state, seeks its own start, and
              runs its own ffmpeg. The only thing they ever shared was
              the caller's single VideoCapture, so giving each worker
              its own capture is all it takes to run them concurrently.
              cv2 decoding, numpy compositing and ffmpeg encoding all
              release the GIL, so threads are the right primitive here -
              no pickling of box lists, and one shared sticker and
              shape-mask cache.

  ONE ENCODE  Chunks are encoded directly as H.264 and concatenated by
              stream copy; audio is muxed by stream copy too. Before 2.1
              the pipeline wrote mpeg4 -qscale 1 chunks, concatenated
              them, and then re-encoded the entire video to H.264 - a
              whole extra encode of every output, and two generations of
              lossy compression instead of one.

THREAD SAFETY REQUIREMENT
    Parallel workers share the box list. betautils_censor.censor_img_for_boxes
    therefore must not mutate it, and does not. tests/test_render_pipeline.py
    asserts that, because the moment it does, a parallel render starts
    producing chunk-order-dependent output.
"""

import math
import bisect
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_censor as bu_censor
import betautils_hash as bu_hash
import betautils_log as bu_log
import betautils_signals as bu_signals
import betautils_video as bu_video


class RenderError( RuntimeError ):
    """Raised when a chunk, a concat, or the final mux cannot be produced."""


class ChunkPlan:
    """One chunk's frame range and output paths."""

    __slots__ = ( 'index', 'start_frame', 'end_frame', 'path', 'meta_path', 'is_last' )

    def __init__( self, index, start_frame, end_frame, path, is_last ):
        self.index = index
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.path = path
        self.meta_path = path + '.meta.json'
        self.is_last = is_last

    @property
    def expected_frames( self ):
        return self.end_frame - self.start_frame

    def __repr__( self ):
        return "ChunkPlan(%d, frames %d..%d)"%(self.index, self.start_frame, self.end_frame)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_chunks( intermediate_path, vid_fps, num_frames, preview_mode_enabled,
                 preview_window_unlimited, preview_offset_seconds, preview_max_seconds ):
    """
    Work out the frame range to render and how it splits into chunks.

    Args:
        intermediate_path: The path chunks are named relative to.
        vid_fps: Frame rate of the file being decoded.
        num_frames: Frame count of the file being decoded. Must come
            from the file actually being decoded - after a transcode
            fallback that is the transcoded copy, whose count can differ
            from the original's by a frame or two. Planning past the real
            end is how a render silently truncates.
        preview_mode_enabled: Whether preview mode is on.
        preview_window_unlimited: Whether this file's preview is really
            the whole file.
        preview_offset_seconds: Where the preview slice starts.
        preview_max_seconds: Preview window length.

    Returns:
        A list of ChunkPlan, at least one, covering the whole range.
    """
    num_frames = int( num_frames )
    start_frame = 0
    end_frame = num_frames
    if preview_mode_enabled and not preview_window_unlimited:
        start_frame = int( round( preview_offset_seconds * vid_fps ) )
        end_frame = min( num_frames,
                         start_frame + int( math.ceil( preview_max_seconds * vid_fps ) ) )
    span_frames = max( 0, end_frame - start_frame )

    render_chunk_seconds = getattr( betaconfig, 'render_chunk_seconds', 600 )
    if preview_mode_enabled or not render_chunk_seconds or render_chunk_seconds <= 0:
        chunk_frames = max( 1, span_frames )
    else:
        chunk_frames = max( 1, int( round( render_chunk_seconds * vid_fps ) ) )
        chunk_frames = _chunk_frames_for_parallelism( chunk_frames, span_frames, vid_fps )
    chunk_count = max( 1, int( math.ceil( span_frames / chunk_frames ) ) ) if span_frames else 1

    plans = []
    for index in range( chunk_count ):
        chunk_start = start_frame + index * chunk_frames
        chunk_end = min( start_frame + ( index + 1 ) * chunk_frames, end_frame )
        plans.append( ChunkPlan(
            index, chunk_start, chunk_end,
            bu_cache.chunk_path_for( intermediate_path, index ),
            is_last=( index == chunk_count - 1 ) ) )
    return plans


def _available_workers():
    """
    The worker ceiling this machine would use given ENOUGH chunks.

    resolve_worker_count clamps to the chunk count, which is the right
    answer once chunks exist but useless for deciding how many to make.
    """
    configured = getattr( betaconfig, 'render_workers', 0 )
    if configured and configured > 0:
        return max( 1, int( configured ) )
    cpu_count = os.cpu_count() or 1
    return max( 1, min( cpu_count // 2 or 1, 8 ) )


def _chunk_frames_for_parallelism( chunk_frames, span_frames, vid_fps,
                                   min_chunk_seconds=15.0 ):
    """
    Shrink the chunk length when a fixed one would starve the workers.

    THE BUG THIS FIXES
    ------------------
    Chunk length was render_chunk_seconds flat, so chunk COUNT fell out
    of the video's duration alone - and workers are capped at the chunk
    count. On a 6-worker box with render_chunk_seconds=180 a 74s video
    got 1 chunk and therefore 1 worker; a 210s video got 2. Measured
    across a real 10-file run, 4 files rendered at 1-4 workers instead of
    6 for no reason but arithmetic: 22.9 minutes, 10.3% of render time
    and 7.1% of the whole run's wall clock, spent single-threading work
    the machine had cores for.

    Chunks are independent by construction (each seeks its own start and
    runs its own ffmpeg), so splitting further is free apart from one
    extra concat entry and one more ffmpeg startup each.

    min_chunk_seconds keeps that from going silly on a very short clip:
    below it, per-chunk ffmpeg startup and the concat would start to cost
    more than the parallelism wins.

    Args:
        chunk_frames: The length render_chunk_seconds asked for.
        span_frames: Total frames to render.
        vid_fps: Frame rate, to convert the floor into frames.
        min_chunk_seconds: Shortest chunk worth making.

    Returns:
        A chunk length in frames, never longer than asked for.
    """
    workers = _available_workers()
    if workers <= 1 or span_frames <= 0:
        return chunk_frames
    # Already enough chunks to fill every worker.
    if span_frames > chunk_frames * workers:
        return chunk_frames
    floor_frames = max( 1, int( round( min_chunk_seconds * vid_fps ) ) )
    wanted = max( 1, int( math.ceil( span_frames / workers ) ) )
    return max( floor_frames, min( chunk_frames, wanted ) )


def resolve_worker_count( chunk_count ):
    """
    How many chunks to render at once.

    betaconfig.render_workers of 0 means auto: one worker per two CPUs,
    clamped to [1, 8] and never more than there are chunks. Two CPUs per
    worker because each worker runs both a Python compositing loop and
    an x264 encoder, and oversubscribing them makes every chunk slower
    without finishing any sooner.

    Args:
        chunk_count: How many chunks this render has.

    Returns:
        A positive int.
    """
    configured = getattr( betaconfig, 'render_workers', 0 )
    if configured and configured > 0:
        return max( 1, min( int( configured ), chunk_count ) )
    cpu_count = os.cpu_count() or 1
    return max( 1, min( cpu_count // 2 or 1, 8, chunk_count ) )


def _encoder_threads( worker_count ):
    """x264 threads per worker, so N workers do not oversubscribe the box."""
    cpu_count = os.cpu_count() or 1
    return max( 1, cpu_count // max( 1, worker_count ) )


# ---------------------------------------------------------------------------
# Box windowing
# ---------------------------------------------------------------------------

def track_key( box ):
    """
    The key that identifies one tracked OBJECT across samples.

    Both parts are required. _track_id restarts from 0 for every label,
    so id 0 means "the first breast track" and "the first covered_vulva
    track" at the same time - on a real cache one id carried two boxes
    at 150 separate instants, a breast at (571,42) and a covered_vulva at
    (617,446). Interpolating between those two would slide a censor
    across the body.
    """
    return ( box.get( 'label' ), box.get( '_track_id' ) )


def build_motion_index( boxes ):
    """
    Per-object timelines, so a frame between two samples can be
    interpolated instead of holding the earlier sample's geometry.

    WHY THIS EXISTS
    ---------------
    Detection samples at video_censor_fps; the render writes every source
    frame. On a 60fps source sampled at 9fps that is 6.67 output frames
    per sample, so each box position is held for ~6.7 frames and then
    jumps. Measured on a real 640m cache (longest breast track, 152
    samples) the jump between consecutive samples was 5.1px median,
    10.4px p90, 14.1px max - an edge teleporting five pixels nine times a
    second, frozen in between.

    That is the flicker that survived every blur change, and it could
    not have been fixed by any of them: strength, method and the edge
    margin all change what FILLS the box, while this moves the box's
    boundary. position_smoothing could not fix it either, because it
    smooths between SAMPLES, not between output frames - it makes the
    jump smaller without making it less sudden.

    Args:
        boxes: Every renderable box, sorted by 'start'. Not mutated.

    Returns:
        {track_key: [box, ...]} with each timeline sorted by 't'. Objects
        with a single sample are omitted: there is nothing to interpolate
        between, and leaving them out keeps the per-frame lookup small.
    """
    grouped = {}
    for box in boxes:
        if box.get( 't' ) is None:
            continue
        grouped.setdefault( track_key( box ), [] ).append( box )

    timelines = {}
    for key, entries in grouped.items():
        if len( entries ) < 2:
            continue
        entries.sort( key=lambda entry: entry['t'] )
        # The timestamps are kept as their own list so the per-frame
        # lookup can bisect them directly. Rebuilding that list inside
        # the frame loop would make the render O(boxes) per box per
        # frame, which on a long video is the whole cost of the pass.
        timelines[key] = ( [ entry['t'] for entry in entries ], entries )
    return timelines


def _interpolated_geometry( earlier, later, current_time ):
    """
    Linearly blended x/y/w/h between two samples of one object.

    Returns None when the two samples are not adjacent in time in a way
    worth blending - see interpolate_boxes_for_frame for the rules.
    """
    span = later['t'] - earlier['t']
    if span <= 0:
        return None
    fraction = ( current_time - earlier['t'] ) / span
    if fraction <= 0:
        return None
    if fraction > 1:
        # Past the later sample there is nothing to blend toward, so the
        # caller holds what it has rather than extrapolating along a path
        # no detector confirmed.
        return None
    # Note the guard above is > 1, not >= 1. fraction == 1 means this frame
    # lands exactly on the later sample, and the right answer there is the
    # later sample's geometry. Bailing out instead snapped the rectangle
    # back to the EARLIER position on that one frame - a full-interval jump
    # backwards, once per sample, which is the flicker signature this
    # function exists to remove.
    blended = dict( earlier )
    for key in ( 'x', 'y', 'w', 'h' ):
        blended[key] = int( round( earlier[key]
                                   + fraction * ( later[key] - earlier[key] ) ) )
    # Clamp exactly as tracking does, so a blend can never push the box
    # past the frame and hand censor_image a mask wider than its region.
    width = earlier.get( '_vid_w' )
    height = earlier.get( '_vid_h' )
    if width and height:
        blended['x'] = max( 0, min( blended['x'], width - 1 ) )
        blended['y'] = max( 0, min( blended['y'], height - 1 ) )
        blended['w'] = max( 1, min( blended['w'], width - blended['x'] ) )
        blended['h'] = max( 1, min( blended['h'], height - blended['y'] ) )
    blended['_motion_interpolated'] = True
    return blended


def interpolate_boxes_for_frame( live_boxes, timelines, current_time, max_span,
                                 max_collapse_growth=1.15, size_window=None ):
    """
    Replace each live box with its geometry AT this frame's time.

    The style, seed and reference size all come from the EARLIER sample,
    so nothing about the censor's appearance changes mid-slide: only the
    rectangle moves. A style that changed here would be the flicker this
    is meant to remove.

    Rules, each of which exists to avoid inventing motion:
      - Only between two real samples of the SAME object, identified by
        (label, _track_id).
      - Only when the two samples are at most max_span apart. Across a
        longer hole the object genuinely was not seen, and sliding
        through it would draw a censor along a path nobody detected.
      - Never extrapolated. Before the first sample or after the last,
        the box holds, which is the old behaviour.

    Args:
        live_boxes: Boxes live at current_time.
        timelines: From build_motion_index.
        current_time: This frame's timestamp in seconds.
        max_span: Longest sample-to-sample gap to blend across, seconds.
        max_collapse_growth: Collapse two live samples of one object into
            one box when their union is no more than this multiple of the
            larger box. 1.0 disables the collapse.
        size_window: Hold each track's rectangle at the largest size it
            was sampled at within this many seconds either side, or for
            the whole track when 0. None disables size stabilisation.
            See _stabilise_sizes for what it costs and buys.

    Returns:
        A new list. NOT the same length as the input: overlapping samples
        of one object are collapsed to a single box when they are close
        enough. Boxes that cannot be interpolated are passed through
        unchanged (the same dict object), so nothing here is mutated.
    """
    if not timelines:
        return live_boxes

    # One box per tracked object per frame, and that one is the blend.
    #
    # This is the part that actually removes the pulsing. time_safety
    # makes each box live for LONGER than the sampling interval (0.17s
    # against 0.111s here, 153% coverage), so for a majority of output
    # frames TWO consecutive samples of the same object are live at once.
    # Measured on a real cache: 66 of 120 frames had two live. The blur
    # overlap strategy then unions them, and the rendered rectangle grows
    # and shrinks as the overlap comes and goes - 78x86, 78x87, 77x86,
    # 79x91 on consecutive frames, oscillating at the sample rate.
    #
    # That is the flicker. Drawing only the interpolated geometry of the
    # latest sample at or before this frame gives one rectangle that
    # moves smoothly instead of two that overlap intermittently.
    #
    # Coverage IS reduced, and the first version of this comment claimed
    # otherwise on the strength of a one-track slice. Measured properly,
    # with real pixel masks over a 120fps/9fps cache at growth 1.15: 561k
    # pixels the old behaviour censored go uncovered, 34% of them inside
    # the nearest real detection rather than union margin, up to 4012px on
    # one frame. The collapse is not free. render_motion_size_window is
    # what pays it back - see _stabilise_sizes.
    #
    # It is also worse while the object is moving fast.
    #
    # When two live samples nearly coincide, their union is a couple of
    # pixels wider than either and collapsing to one loses nothing. When
    # the object is moving fast they straddle the motion, and the union
    # covers the whole swept path. That extra area is a real safety
    # margin: if the blend is even slightly behind the subject, the union
    # still covered her. Measured worst case on a real cache, collapsing
    # unconditionally gave up 8265px on one fast-motion frame.
    #
    # So collapse only where it is free. Pulsing is what the eye notices
    # on a near-still subject; during fast motion the motion itself masks
    # it, which is exactly where the margin is kept instead.
    grouped = {}
    passthrough = []
    for box in live_boxes:
        key = track_key( box )
        if key not in timelines or box.get( 't' ) is None:
            passthrough.append( box )
            continue
        grouped.setdefault( key, [] ).append( box )

    collapsed = []
    for key, group in grouped.items():
        if len( group ) == 1:
            collapsed.append( group[0] )
            continue
        union_x0 = min( entry['x'] for entry in group )
        union_y0 = min( entry['y'] for entry in group )
        union_x1 = max( entry['x'] + entry['w'] for entry in group )
        union_y1 = max( entry['y'] + entry['h'] for entry in group )
        union_area = ( union_x1 - union_x0 ) * ( union_y1 - union_y0 )
        largest = max( entry['w'] * entry['h'] for entry in group )
        if largest > 0 and union_area <= largest * max_collapse_growth:
            # The survivor must be the latest sample AT OR BEFORE this
            # frame, not simply the latest in the group. time_safety makes
            # a sample live before its own timestamp, so the newest live
            # sample is usually still in the future, and the blend loop
            # below skips anything with t >= current_time. Keeping that one
            # therefore froze the rectangle at the future sample's
            # position: x went 100, 101, 105, 105, 105, 105 across six
            # frames - the area oscillation gone but a 4px positional jump
            # and a five-frame freeze put in its place, which is the
            # teleporting this function exists to remove.
            eligible = [ entry for entry in group if entry['t'] <= current_time ]
            collapsed.append( max( eligible or group,
                                   key=lambda entry: entry['t'] ) )
        else:
            collapsed.extend( group )
    live_boxes = passthrough + collapsed

    output = []
    for box in live_boxes:
        timeline = timelines.get( track_key( box ) )
        sample_time = box.get( 't' )
        if not timeline or sample_time is None or sample_time >= current_time:
            output.append( box )
            continue
        times, entries = timeline
        # The next real sample of this object after the one being shown.
        # bisect_right skips every entry sharing this timestamp, which is
        # what we want: two boxes at one instant are two objects, not a
        # step in time.
        index = bisect.bisect_right( times, sample_time )
        later = entries[index] if index < len( entries ) else None
        if later is None or later['t'] - sample_time > max_span:
            output.append( box )
            continue
        blended = _interpolated_geometry( box, later, current_time )
        output.append( blended if blended is not None else box )

    if size_window is not None:
        output = _stabilise_sizes( output, timelines, size_window )
    return output


def _stabilise_sizes( live_boxes, timelines, size_window ):
    """
    Hold each track's rectangle at a constant size, centred where the
    interpolation put it.

    WHY, AND WHAT IT COSTS
    ----------------------
    Sliding the box and collapsing overlaps removes most of the pulsing,
    but not all of it, and the collapse pays for what it removes: the
    union of two live samples covers area that one interpolated box does
    not. Measured on a real 120fps/9fps cache, at
    render_motion_collapse_max_growth 1.15, 561k pixels that the old
    behaviour censored went uncovered, and 34% of those were inside the
    nearest real detection rather than union margin - up to 4012px on one
    frame. That is a coverage regression, not a cosmetic change.

    Holding a track's size at the largest it was sampled at fixes both
    ends at once, because the box is then never smaller than any sample
    in range. size_window bounds how far in time that maximum is taken
    from, and it is a straight dial between coverage and censored area.
    Measured over the whole slice, against the old held behaviour:

        size_window   lost detected px   worst frame   area p99   area
        None (off)              531339          4055        151    -1.7%
        0.25s                   352260          2521        126    -0.0%
        0.5s                    298985          2521         80    +1.3%
        1.0s                    219475          2488         72    +4.0%
        2.0s                    153016          2111         68    +7.4%
        0 (whole track)          48652          2089          0   +20.2%

    0.25s is the default because it buys a third of the lost coverage
    back for no measurable extra area. The whole track is the best
    coverage on offer and the only setting that takes the residual size
    pulse to zero, but 20% more blurred area is a judgement about the
    footage rather than a number, so it is opt-in.

    A window also matters for a subject that genuinely changes size - one
    approaching the camera - where a whole-track maximum would hold the
    largest size for the entire track.
    """
    stabilised = []
    for box in live_boxes:
        key = track_key( box )
        timeline = timelines.get( key )
        sample_time = box.get( 't' )
        if timeline is None or sample_time is None:
            stabilised.append( box )
            continue
        # Every box gets its own size guarantee, including both boxes of a
        # track the collapse declined to merge. An earlier version skipped
        # a track after its first box, so on exactly the fast-motion frames
        # where two survive - where coverage matters most - the second one
        # kept whatever size its sample had.
        times, entries = timeline
        if size_window <= 0:
            selected = entries
        else:
            low = bisect.bisect_left( times, sample_time - size_window )
            high = bisect.bisect_right( times, sample_time + size_window )
            selected = entries[low:high] or entries
        width = max( entry['w'] for entry in selected )
        height = max( entry['h'] for entry in selected )
        if width <= box['w'] and height <= box['h']:
            stabilised.append( box )
            continue
        # Grow around the interpolated centre, so the box tracks the
        # subject's position while its size stays put.
        centre_x = box['x'] + box['w']/2.0
        centre_y = box['y'] + box['h']/2.0
        grown = dict( box )
        grown['w'] = width
        grown['h'] = height
        grown['x'] = int( round( centre_x - width/2.0 ) )
        grown['y'] = int( round( centre_y - height/2.0 ) )
        frame_w = box.get( '_vid_w' )
        frame_h = box.get( '_vid_h' )
        if frame_w and frame_h:
            grown['x'] = max( 0, min( grown['x'], frame_w - 1 ) )
            grown['y'] = max( 0, min( grown['y'], frame_h - 1 ) )
            grown['w'] = max( 1, min( grown['w'], frame_w - grown['x'] ) )
            grown['h'] = max( 1, min( grown['h'], frame_h - grown['y'] ) )
        grown['_motion_size_stabilised'] = True
        stabilised.append( grown )
    return stabilised


def boxes_for_chunk( boxes, chunk_start_time ):
    """
    Split the box list into "already live" and "starts later" at a time.

    A censored region with a long time_safety or interpolation tail can
    span a chunk boundary, so a chunk cannot assume it starts with
    nothing live. This reconstructs the state the single continuous pass
    used to build incrementally.

    Args:
        boxes: Every box for the video, sorted by 'start' ascending.
        chunk_start_time: The chunk's first frame, in seconds.

    Returns:
        A (live, pending) pair. Neither list is mutated by the renderer;
        both are views onto the caller's box dicts, shared read-only
        across workers.
    """
    live = []
    pending = []
    for box in boxes:
        if box['start'] > chunk_start_time:
            pending.append( box )
        elif box['end'] > chunk_start_time:
            live.append( box )
    return live, pending


# ---------------------------------------------------------------------------
# Chunk rendering
# ---------------------------------------------------------------------------

def _ffmpeg_base_command():
    """The ffmpeg prefix every render command shares."""
    if getattr( betaconfig, 'debug_mode', 0 ) & 1:
        return [ 'ffmpeg', '-y' ]
    return [ 'ffmpeg', '-y', '-loglevel', 'error' ]


def _chunk_encode_command( tmp_path, vid_w, vid_h, vid_fps, preset, encoder_threads ):
    """
    ffmpeg command that reads raw BGR frames on stdin and writes H.264.

    The even-dimension pad is applied here, at the chunk encode, rather
    than at a later re-encode: every chunk pads identically, so the
    concatenated stream has consistent parameters and can be copied
    rather than re-encoded.
    """
    return _ffmpeg_base_command() + [
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '{}x{}'.format( vid_w, vid_h ),
        '-pix_fmt', 'bgr24',
        '-r', '%.6f'%(vid_fps),
        '-i', '-',
        '-an',
        '-c:v', getattr( betaconfig, 'encode_video_codec', 'libx264' ),
        '-crf', str( getattr( betaconfig, 'encode_crf', 17 ) ),
        '-preset', preset,
        '-threads', str( encoder_threads ),
        '-pix_fmt', 'yuv420p',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        tmp_path,
    ]


def _write_chunk_metadata( plan, frames_written ):
    """Record a completed chunk's real frame count beside it."""
    bu_hash.write_json_plain( {
        'index': plan.index,
        'start_frame': plan.start_frame,
        'end_frame': plan.end_frame,
        'expected_frames': plan.expected_frames,
        'frames_written': frames_written,
    }, plan.meta_path )


def chunk_is_complete( plan, verify_frame_counts=True ):
    """
    Whether a previous run already produced a trustworthy chunk.

    Checks three things, because the container check alone cannot see
    the failure that matters most: a chunk whose decode died halfway is
    a perfectly valid video file that is simply too short, and before
    2.1 it was promoted, trusted on resume, and silently truncated the
    output.

    Args:
        plan: The ChunkPlan to check.
        verify_frame_counts: When False, only the container check runs.
            betaconfig.render_verify_frame_counts.

    Returns:
        True when the chunk can be reused as-is.
    """
    if not os.path.exists( plan.path ):
        return False
    if not bu_video.probe_video_ok( plan.path ):
        return False
    if not verify_frame_counts:
        return True
    if not os.path.exists( plan.meta_path ):
        # Written by an older build, or the metadata was lost. Re-render
        # rather than trust a count nobody recorded.
        return False
    try:
        metadata = bu_hash.read_json_plain( plan.meta_path )
    except Exception:
        return False
    return metadata.get( 'frames_written' ) == metadata.get( 'expected_frames' )


def _motion_settings( source ):
    """
    Whether to interpolate box geometry between samples, and how far.

    Default span is two sampling intervals: long enough to cover the
    normal sample-to-sample step plus one missed detection, short enough
    that a real absence is held rather than slid through.
    """
    enabled = getattr( betaconfig, 'render_motion_interpolation', True )
    sample_fps = getattr( betaconfig, 'video_censor_fps', 9 ) or 9
    span = getattr( betaconfig, 'render_motion_max_span_seconds', None )
    if span is None:
        span = 2.0 / sample_fps
    growth = getattr( betaconfig, 'render_motion_collapse_max_growth', 1.15 )
    # Default 0.25s, which on a real 120fps/9fps cache cut the detected
    # pixels the collapse gave up by 34% (528k -> 350k) and worst-frame
    # loss by 38% (4055 -> 2521px) for -0.1% area - free, within
    # measurement noise. Larger windows keep buying coverage but start
    # charging for it, up to the whole track (0) at 51k lost and +20.1%
    # area. That last one is the best coverage available here, but 20% more
    # blurred area is a judgement about the footage, not a number, so it is
    # a setting rather than the default. See _stabilise_sizes.
    window = getattr( betaconfig, 'render_motion_size_window_seconds', 0.25 )
    return ( bool( enabled ), float( span ), float( growth ),
             None if window is None else float( window ) )


def render_one_chunk( plan, source, boxes, preset, encoder_threads, max_retries,
                      retry_backoff, progress, logger, motion_timelines=None ):
    """
    Render one chunk to its own file, retrying on failure.

    Opens its own VideoCapture, so it is safe to run concurrently with
    other chunks of the same video.

    Args:
        plan: The ChunkPlan to render.
        source: A betautils_video.VideoSource.
        boxes: Every box for the video, sorted by 'start'. Read-only.
        preset: x264 preset for this run.
        encoder_threads: -threads for this chunk's encoder.
        max_retries: Additional attempts after the first.
        retry_backoff: Seconds between attempts.
        progress: A bu_log.ProgressReporter, or None.
        logger: The shared logger.
        motion_timelines: A prebuilt build_motion_index result covering
            the whole video, shared read-only across workers. None means
            build it here, which every worker would then do redundantly
            over the same box list - correct, just wasteful.

    Returns:
        The number of frames written.

    Raises:
        RenderError: every attempt failed, or the chunk came out short
            in a position where short means a decode failure.
    """
    tmp_path = bu_cache.temp_sibling( plan.path )
    chunk_start_time = plan.start_frame / source.vid_fps
    base_live, base_pending = boxes_for_chunk( boxes, chunk_start_time )
    ( motion_enabled, motion_max_span, motion_collapse_growth,
      motion_size_window ) = _motion_settings( source )
    # Outside the retry loop and outside the frame loop. Measured 35ms over
    # 65k boxes - small, but every chunk and every worker would otherwise
    # redo it over the same immutable list.
    if motion_timelines is None:
        motion_timelines = build_motion_index( boxes ) if motion_enabled else {}
    elif not motion_enabled:
        motion_timelines = {}

    last_error = None
    for attempt in range( 1 + max( 0, max_retries ) ):
        if attempt > 0:
            logger.warning( "render chunk %d: retry %d/%d after %s"%(
                plan.index, attempt, max_retries, last_error ) )
            time.sleep( retry_backoff )

        capture = source.open_capture()
        process = None
        frames_written = 0
        write_error = None
        try:
            capture.set( cv2.CAP_PROP_POS_FRAMES, plan.start_frame )
            command = _chunk_encode_command(
                tmp_path, source.width, source.height, source.vid_fps, preset, encoder_threads )
            logger.trace( "render chunk %d command: %s"%(plan.index, ' '.join( command )) )
            process = subprocess.Popen( command, stdin=subprocess.PIPE )

            live_boxes = list( base_live )
            pending_boxes = base_pending
            pending_cursor = 0
            frame_index = plan.start_frame

            while frame_index < plan.end_frame:
                bu_signals.check()
                retrieved, frame = capture.read()
                if not retrieved or frame is None:
                    break
                current_time = frame_index / source.vid_fps

                if live_boxes:
                    live_boxes = [ box for box in live_boxes if box['end'] >= current_time ]
                # An index cursor rather than pending.pop(0): popping the
                # head of a list is O(n), which on a long video with many
                # boxes made this loop quadratic in the box count.
                while ( pending_cursor < len( pending_boxes )
                        and pending_boxes[pending_cursor]['start'] <= current_time ):
                    live_boxes.append( pending_boxes[pending_cursor] )
                    pending_cursor += 1

                # Slide each box to where it actually is at THIS frame,
                # rather than holding the last sample's rectangle until
                # the next sample replaces it.
                frame_boxes = ( interpolate_boxes_for_frame(
                                    live_boxes, motion_timelines, current_time, motion_max_span,
                                    motion_collapse_growth, motion_size_window )
                                if motion_enabled else live_boxes )
                frame = bu_censor.censor_img_for_boxes( frame, frame_boxes )
                # memoryview avoids copying the whole frame just to hand
                # its bytes to the pipe. Frames from cv2 are contiguous.
                process.stdin.write( memoryview( frame ) if frame.flags['C_CONTIGUOUS']
                                     else frame.tobytes() )
                frames_written += 1
                frame_index += 1
                if progress is not None:
                    progress.advance()
        except bu_signals.Interrupted:
            # Never retried, and never reported as a chunk failure: the
            # user asked to stop, and the already-promoted chunks are
            # what the next run resumes from.
            capture.release()
            if process is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
                process.wait()
            _remove_quietly( tmp_path )
            raise
        except Exception as err:
            write_error = err
        finally:
            capture.release()
            if process is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
                process.wait()

        if write_error is not None:
            last_error = repr( write_error )
            continue
        if process.returncode != 0:
            last_error = "ffmpeg exited %d"%(process.returncode)
            continue
        if not bu_video.probe_video_ok( tmp_path ):
            last_error = "chunk failed the container sanity check"
            continue

        if frames_written < plan.expected_frames:
            shortfall = plan.expected_frames - frames_written
            if plan.is_last:
                # The very last chunk may legitimately come up short:
                # container metadata over-reports the frame count on some
                # files. Accept it, but say so, because the alternative -
                # silently accepting a short chunk anywhere - is the
                # truncation bug this check exists to catch.
                logger.warning( "render chunk %d (last): decoder ran out %d frame(s) before the planned "
                    "end (%d of %d). Container metadata over-reporting the frame count is the usual "
                    "cause; the output is complete to the real end of the file."%(
                        plan.index, shortfall, frames_written, plan.expected_frames ) )
            else:
                last_error = ( "chunk is %d frame(s) short (%d of %d) and is not the last chunk, so the "
                               "decoder failed mid-file rather than reaching EOF"%(
                                   shortfall, frames_written, plan.expected_frames ) )
                continue

        os.replace( tmp_path, plan.path )  # atomic promote: only now is it trusted
        _write_chunk_metadata( plan, frames_written )
        logger.debug( "render chunk %d complete: %d frame(s)"%(plan.index, frames_written) )
        return frames_written

    raise RenderError( "render chunk %d failed after %d attempt(s): %s"%(
        plan.index, max_retries+1, last_error ) )


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

def concat_chunks( plans, intermediate_path, max_retries, retry_backoff, logger ):
    """
    Concatenate every rendered chunk into the intermediate video.

    Stream copy via ffmpeg's concat demuxer - no re-encode, because the
    chunks are already the final codec at the final settings.

    Args:
        plans: Every ChunkPlan, in order.
        intermediate_path: Where the concatenated video goes.
        max_retries: Additional attempts after the first.
        retry_backoff: Seconds between attempts.
        logger: The shared logger.

    Side effects:
        On success, removes every chunk file, its metadata sidecar, and
        the temporary concat list.

    Raises:
        RenderError: every attempt failed.
    """
    if len( plans ) == 1:
        os.replace( plans[0].path, intermediate_path )
        _remove_quietly( plans[0].meta_path )
        return

    concat_list_path = intermediate_path + '.concat.txt'
    with open( concat_list_path, 'w', encoding='UTF-8' ) as list_file:
        for plan in plans:
            list_file.write( "file '%s'\n"%(os.path.abspath( plan.path )) )

    tmp_path = bu_cache.temp_sibling( intermediate_path )
    command = _ffmpeg_base_command() + [
        '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', tmp_path ]

    last_error = None
    for attempt in range( 1 + max( 0, max_retries ) ):
        if attempt > 0:
            logger.warning( "chunk concat: retry %d/%d after %s"%(attempt, max_retries, last_error) )
            time.sleep( retry_backoff )
        process = subprocess.Popen( command )
        process.wait()
        if process.returncode != 0:
            last_error = "ffmpeg exited %d"%(process.returncode)
            continue
        if not bu_video.probe_video_ok( tmp_path ):
            last_error = "concatenated output failed the container sanity check"
            continue
        os.replace( tmp_path, intermediate_path )
        _remove_quietly( concat_list_path )
        for plan in plans:
            _remove_quietly( plan.path )
            _remove_quietly( plan.meta_path )
        return

    raise RenderError( "chunk concatenation failed after %d attempt(s): %s"%(
        max_retries+1, last_error ) )


def _remove_quietly( path ):
    """Delete a path, ignoring the case where it is already gone."""
    try:
        if os.path.exists( path ):
            os.remove( path )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Audio mux
# ---------------------------------------------------------------------------

def mux_audio( intermediate_path, original_path, final_path, max_retries,
               retry_backoff, logger, preview_offset_seconds=0.0 ):
    """
    Add the source file's audio to the rendered video, without re-encoding.

    The video is already H.264 at the configured CRF and preset, so this
    is a pure container operation.

    Tries `-c:a copy` first. If the source's audio codec cannot live in
    an MP4 (Vorbis or Opus in a Matroska source, say), ffmpeg fails and
    this retries with AAC. Re-encoding the audio is a real quality
    decision, so it is logged rather than done silently.

    Args:
        intermediate_path: The rendered, audio-less video.
        original_path: The user's source file, the audio source. NEVER
            a transcoded copy - the transcode exists to work around a
            video-decode problem and has no bearing on the audio.
        final_path: The .mp4 to produce.
        max_retries: Additional attempts per audio strategy.
        retry_backoff: Seconds between attempts.
        logger: The shared logger.
        preview_offset_seconds: Where the rendered video starts within
            the source. MUST be seeked to on the audio input, or the
            audio starts from 0:00 while the video starts mid-file - a
            preview slice at 80s played the file's opening 30s of audio
            against video from 80s in, an 80-second desync. 0 for a
            normal full run, where the two already start together.

    Side effects:
        On success, removes intermediate_path.

    Raises:
        RenderError: no strategy produced a valid output.
    """
    has_audio = bu_video.video_file_has_audio( original_path )
    tmp_path = bu_cache.temp_sibling( final_path )

    if has_audio:
        audio_strategies = [
            ( 'copy', [ '-c:a', 'copy' ] ),
            ( 'aac',  [ '-c:a', 'aac', '-b:a', '192k' ] ),
        ]
    else:
        audio_strategies = [ ( 'none', [] ) ]

    last_error = None
    for strategy_name, audio_args in audio_strategies:
        if has_audio:
            # -ss BEFORE -i on the audio source: an input seek, so
            # ffmpeg starts decoding at that point and the first audio
            # sample it emits lines up with the first rendered frame.
            # An output seek (-ss after -i) would instead decode from
            # zero and discard, which is both slower and leaves the
            # timestamps offset.
            audio_seek = ( [ '-ss', '%.6f'%(preview_offset_seconds,) ]
                           if preview_offset_seconds > 0 else [] )
            command = _ffmpeg_base_command() + [
                '-i', intermediate_path ] + audio_seek + [ '-i', original_path,
                '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'copy' ] + audio_args + [ '-shortest', tmp_path ]
        else:
            command = _ffmpeg_base_command() + [
                '-i', intermediate_path, '-c:v', 'copy', tmp_path ]

        for attempt in range( 1 + max( 0, max_retries ) ):
            if attempt > 0:
                logger.warning( "final mux (%s): retry %d/%d after %s"%(
                    strategy_name, attempt, max_retries, last_error ) )
                time.sleep( retry_backoff )
            logger.trace( "final mux command: %s"%(' '.join( command )) )
            process = subprocess.Popen( command )
            process.wait()
            if process.returncode != 0:
                last_error = "ffmpeg exited %d"%(process.returncode)
                continue
            if not bu_video.probe_video_ok( tmp_path ):
                last_error = "final output failed the container sanity check"
                continue
            os.replace( tmp_path, final_path )
            _remove_quietly( intermediate_path )
            if strategy_name == 'aac':
                logger.warning( "source audio could not be copied into MP4 and was re-encoded to AAC "
                                "for %s"%(os.path.basename( final_path )) )
            return

        logger.warning( "final mux with '-c:a %s' failed: %s"%(strategy_name, last_error) )

    raise RenderError( "final mux failed for %s: %s"%(final_path, last_error) )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def render_video( source, boxes, intermediate_path, final_path, preset,
                  preview_mode_enabled, preview_window_unlimited,
                  preview_offset_seconds, preview_max_seconds, logger=None ):
    """
    Render, concatenate and mux one video, resuming whatever a previous
    run already finished.

    Args:
        source: A betautils_video.VideoSource.
        boxes: Every renderable box, sorted by 'start'. Not mutated.
        intermediate_path: Where the audio-less render goes.
        final_path: The .mp4 to produce.
        preset: x264 preset for this run.
        preview_mode_enabled: Whether preview mode is on.
        preview_window_unlimited: Whether this file's preview is really
            the whole file.
        preview_offset_seconds: Where the preview slice starts.
        preview_max_seconds: Preview window length.
        logger: Optional logger.

    Returns:
        A stats dict with 'chunks', 'chunks_rendered', 'chunks_reused',
        'frames_written' and 'render_workers'.

    Raises:
        RenderError: the render could not be completed.
    """
    logger = logger or bu_log.get_logger()
    max_retries = getattr( betaconfig, 'ffmpeg_max_retries', 2 )
    retry_backoff = getattr( betaconfig, 'ffmpeg_retry_backoff_seconds', 5 )
    verify_frame_counts = getattr( betaconfig, 'render_verify_frame_counts', True )

    if os.path.exists( final_path ) and bu_video.probe_video_ok( final_path ):
        logger.info( "final output already present and valid, nothing to render: %s"%(final_path) )
        return { 'chunks': 0, 'chunks_rendered': 0, 'chunks_reused': 0,
                 'frames_written': 0, 'render_workers': 0 }

    plans = plan_chunks( intermediate_path, source.vid_fps, source.num_frames,
                         preview_mode_enabled, preview_window_unlimited,
                         preview_offset_seconds, preview_max_seconds )

    if not ( os.path.exists( intermediate_path ) and bu_video.probe_video_ok( intermediate_path ) ):
        outstanding = [ plan for plan in plans
                        if not chunk_is_complete( plan, verify_frame_counts ) ]
        reused = len( plans ) - len( outstanding )
        if reused:
            logger.info( "render: reusing %d of %d chunk(s) from a previous run"%(reused, len( plans )) )

        worker_count = resolve_worker_count( max( 1, len( outstanding ) ) )
        encoder_threads = _encoder_threads( worker_count )
        total_frames = sum( plan.expected_frames for plan in outstanding )

        logger.info( "render: %d chunk(s) to build, %d worker(s), %d encoder thread(s) each, "
                     "%d frame(s) total"%(len( outstanding ), worker_count, encoder_threads, total_frames) )

        frames_written = 0
        if outstanding:
            progress = bu_log.ProgressReporter( "render", total=total_frames )
            failures = []
            failures_lock = threading.Lock()

            # One index for the whole video, shared read-only by every
            # worker. Safe for the same reason the box list itself is:
            # nothing in the render path mutates these dicts.
            shared_timelines = ( build_motion_index( boxes )
                                 if getattr( betaconfig, 'render_motion_interpolation', True )
                                 else {} )

            def render_plan( plan ):
                try:
                    return render_one_chunk( plan, source, boxes, preset, encoder_threads,
                                             max_retries, retry_backoff, progress, logger,
                                             motion_timelines=shared_timelines )
                except Exception as err:
                    with failures_lock:
                        failures.append( err )
                    return 0

            if worker_count == 1:
                results = [ render_plan( plan ) for plan in outstanding ]
            else:
                with ThreadPoolExecutor( max_workers=worker_count,
                                         thread_name_prefix='betarender' ) as pool:
                    results = list( pool.map( render_plan, outstanding ) )
            frames_written = sum( results )
            progress.finish( "%d chunk(s)"%(len( outstanding )) )

            if failures:
                raise RenderError( "%d render chunk(s) failed; first failure: %s"%(
                    len( failures ), failures[0] ) )

        concat_chunks( plans, intermediate_path, max_retries, retry_backoff, logger )
    else:
        logger.info( "render output already complete, skipping to the final mux" )
        reused = len( plans )
        worker_count = 0
        frames_written = 0

    mux_audio( intermediate_path, source.original_path, final_path,
               max_retries, retry_backoff, logger,
               preview_offset_seconds=preview_offset_seconds )

    return {
        'chunks': len( plans ),
        'chunks_rendered': len( plans ) - reused,
        'chunks_reused': reused,
        'frames_written': frames_written,
        'render_workers': worker_count,
    }
