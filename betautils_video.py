"""
betautils_video.py - Video decoding, frame sampling, shot-cut detection,
and the hardware-decode fallback ladder.

Everything that talks to a cv2.VideoCapture or to ffprobe lives here, so
betatv.py, betastare.py and the tooling all decode the same way.

THE SAMPLING RULE
-----------------
BetaSuite never processes every frame. It samples at
betaconfig.video_censor_fps and maps sample index -> video frame index
with sample_frame_index() below. That mapping is the contract: detection,
shot-cut detection, checkpoints and every analysis tool must agree on
which frames a sample corresponds to, or timestamps stop lining up.

HOW THOSE FRAMES ARE FETCHED
----------------------------
Sequentially, with cap.grab() for frames we skip and cap.retrieve() only
for frames we keep. Never with a seek per sample.

A seek is not a cheap operation on a long-GOP stream: ffmpeg seeks to the
nearest keyframe and decodes forward from there, so asking for every
third frame can cost more than decoding all of them. Measured on
1280x720 H.264 at the x264 default GOP of 250, sampling 9fps out of 25:

    seek per sample     173.7 ms/sample
    grab + retrieve       4.4 ms/sample

grab() also skips the colour conversion and the numpy allocation that
read() performs, so the frames we throw away cost almost nothing.

The earlier detection loop seeked once per sample. This module's
docstring said not to, and the shot-cut scanner said not to; the
detection loop did it anyway. Both now go through
iter_sampled_frames().
"""

import math
import os
import shutil
import subprocess

import cv2

import betaconst
import betautils_cache_paths as bu_cache
import betautils_hash as bu_hash
import betautils_log as bu_log


# ---------------------------------------------------------------------------
# Sample <-> frame mapping
# ---------------------------------------------------------------------------

def sample_time( sample_index, offset_seconds, sample_fps ):
    """
    Timestamp, in seconds, of one sample.

    Args:
        sample_index: 0-based sample number.
        offset_seconds: Where sampling starts (a preview slice offset;
            0.0 for a full run).
        sample_fps: betaconfig.video_censor_fps.

    Returns:
        The sample's timestamp in seconds.
    """
    return offset_seconds + sample_index / sample_fps


def sample_frame_index( sample_index, offset_seconds, vid_fps, sample_fps ):
    """
    Video frame index for one sample.

    Sample 0 rounds the offset to the nearest frame; every later sample
    floors its own timestamp. That asymmetry is inherited from the
    earlier loop and is preserved deliberately: changing it would shift
    every sampled frame by up to one frame, which would invalidate every
    cached detection and every tuning number derived from one. For an
    offset of 0 the two agree anyway.

    Args:
        sample_index: 0-based sample number.
        offset_seconds: Where sampling starts.
        vid_fps: The video's real frame rate.
        sample_fps: betaconfig.video_censor_fps.

    Returns:
        A 0-based video frame index.
    """
    if sample_index == 0:
        return int( round( offset_seconds * vid_fps ) )
    return int( math.floor( sample_time( sample_index, offset_seconds, sample_fps ) * vid_fps ) )


def iter_sampled_frames( cap, vid_fps, sample_fps, offset_seconds=0.0,
                         num_frames=None, max_seconds=None, start_sample_index=0 ):
    """
    Yield (sample_index, timestamp, frame) for each sampled frame.

    Decodes sequentially: cap.grab() advances past frames we do not want
    without decoding their contents, cap.retrieve() decodes only the ones
    we do. Performs at most ONE seek, to position the capture at the
    first sample (needed for a preview offset or a checkpoint resume).

    Stopping conditions, in the same order the earlier loop applied them:
      - the next sample's frame index reaches num_frames
      - the next sample's timestamp reaches offset_seconds + max_seconds
      - the decoder runs out of frames

    Args:
        cap: An open cv2.VideoCapture. Repositioned as a side effect.
        vid_fps: The video's real frame rate.
        sample_fps: Samples per second to take.
        offset_seconds: Where sampling starts.
        num_frames: Total frames in the file, or None for "until EOF".
        max_seconds: Stop this many seconds after offset_seconds, or None.
        start_sample_index: First sample to yield; > 0 resumes a
            checkpointed pass without re-detecting what it already has.

    Yields:
        (sample_index, timestamp_seconds, frame) with frame a BGR uint8
        numpy array.
    """
    if not vid_fps or vid_fps <= 0:
        raise ValueError( "vid_fps must be positive to sample a video, got %r"%(vid_fps,) )
    if not sample_fps or sample_fps <= 0:
        raise ValueError( "sample_fps must be positive, got %r"%(sample_fps,) )

    first_frame = sample_frame_index( start_sample_index, offset_seconds, vid_fps, sample_fps )
    cap.set( cv2.CAP_PROP_POS_FRAMES, first_frame )

    # next_position is the frame index the next grab() will fetch.
    next_position = first_frame
    sample_index = start_sample_index

    while True:
        target_frame = sample_frame_index( sample_index, offset_seconds, vid_fps, sample_fps )
        if num_frames is not None and target_frame >= num_frames:
            return
        timestamp = sample_time( sample_index, offset_seconds, sample_fps )
        if max_seconds is not None and timestamp >= offset_seconds + max_seconds:
            return

        # Advance to the target. When sample_fps exceeds the video's own
        # frame rate consecutive targets can repeat; the loop then does
        # not run and retrieve() re-decodes the already-buffered frame,
        # matching what a repeated seek used to do.
        while next_position <= target_frame:
            if not cap.grab():
                return
            next_position += 1

        retrieved, frame = cap.retrieve()
        if not retrieved or frame is None:
            return

        yield sample_index, timestamp, frame
        sample_index += 1


# ---------------------------------------------------------------------------
# Shot-cut detection
# ---------------------------------------------------------------------------

def detect_shot_cuts( cap, vid_fps, sample_fps, threshold=0.6, max_seconds=None ):
    """
    Find hard-cut timestamps, so tracking never smooths a box across a
    scene change.

    Why this exists: smooth_boxes assumes continuous single-scene
    footage. On quick-cut compilation material, without this, a tracked
    box can glide from one person in one shot to an unrelated person in
    the next, because they happened to be close in time and screen
    position.

    Method: a 3-channel BGR histogram per sampled frame, compared with
    the Bhattacharyya distance (0 identical, 1 maximally different).
    Histograms rather than pixel differences because ordinary camera
    motion inside one continuous shot moves pixels a lot and the colour
    distribution very little.

    Detection logic, each part of which fixes a real failure of a
    simpler earlier version:
      - A candidate is raised when a sample differs from BOTH of the two
        preceding samples. Comparing only against the immediate
        predecessor cannot catch a clean cut between two shots that each
        have zero internal variance, because a single transition
        produces only one flagged pair.
      - The candidate is then held one more sample and confirmed unless
        that next sample reverts to the PRE-CUT state, which is how a
        one-frame flash or compression spike is rejected.
      - Confirmation compares against the pre-cut state, not the
        candidate's own state. Comparing against the candidate scored
        zero cuts on real quick-cut footage: after one cut the next
        sample is very often the start of the NEXT cut rather than a
        continuation, so it differs from the candidate too and was
        wrongly discarded as a spike.

    A slow dissolve legitimately produces a short cluster of consecutive
    cut timestamps rather than one. That is fine: a track spanning the
    cluster is correctly reset either way.

    Frames are fetched with grab/retrieve, so the ~65% of frames between
    samples are never decoded or converted.

    Args:
        cap: An open cv2.VideoCapture positioned at the start of the
            range to scan. Consumed sequentially as a side effect, so
            callers that need their own read position preserved should
            pass a capture opened just for this.
        vid_fps: The video's real frame rate.
        sample_fps: Samples per second. Pass betaconfig.video_censor_fps
            so cut timestamps land on the same grid as detections.
        threshold: Bhattacharyya distance a pair must meet. Higher means
            fewer, more confident cuts.
        max_seconds: Stop scanning after this many seconds from the
            capture's current position, or None to scan to EOF. A
            preview run passes its window here; previously a 20-second
            preview of a two-hour file still paid for a two-hour scan.

    Returns:
        A sorted list of cut timestamps in seconds. Each is the
        timestamp of the sample that OPENS the new shot.
    """
    step_frames = max( 1, round( vid_fps / sample_fps ) )
    max_frames = None if max_seconds is None else int( math.ceil( max_seconds * vid_fps ) )

    def _bgr_histogram( frame ):
        hist = cv2.calcHist( [frame], [0, 1, 2], None, [32, 32, 32],
                             [0, 256, 0, 256, 0, 256] )
        cv2.normalize( hist, hist, 0, 1, cv2.NORM_MINMAX )
        return hist.flatten()

    cut_times = []
    prev_hist = None
    prev_prev_hist = None
    pending_cut_frame_idx = None
    pending_cut_pre_hist = None

    frame_idx = 0
    sample_idx = 0
    next_sample_frame = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        if not cap.grab():
            break
        if frame_idx == next_sample_frame:
            retrieved, frame = cap.retrieve()
            if not retrieved or frame is None:
                break
            hist = _bgr_histogram( frame )

            # Resolve a pending candidate against THIS sample. A cut is
            # confirmed unless this sample has reverted to the pre-cut
            # state, which is what a one-frame flash or compression
            # spike looks like.
            if pending_cut_frame_idx is not None:
                distance_to_pre_cut = cv2.compareHist(
                    pending_cut_pre_hist, hist, cv2.HISTCMP_BHATTACHARYYA )
                if distance_to_pre_cut >= threshold:
                    cut_times.append( round( pending_cut_frame_idx / vid_fps, 3 ) )
                pending_cut_frame_idx = None
                pending_cut_pre_hist = None

            # This sample is then evaluated as its own candidate,
            # REGARDLESS of whether it just resolved one.
            #
            # This used to be an `elif`, and that cost most of the cuts
            # on quick-cut footage. The sample after a cut is very often
            # the start of the NEXT cut rather than a continuation, so
            # consuming it purely as a confirmation threw the next cut
            # away. Simulated against alternating one-sample shots the
            # old logic found 5 of 11 real cuts, and reported each one a
            # sample late; on real footage two compilation files scored
            # ZERO cuts across 10 and 15 minutes while showing 4-5
            # simultaneous tracked subjects. Those files are exactly the
            # ones where a censor box was seen floating across a cut,
            # because with no cut timestamps nothing ever forced a track
            # reset.
            #
            # With the candidate evaluated every sample, the same
            # simulation finds 10 of 11, and slow-cut footage becomes
            # exact rather than one sample late. The flash rejection is
            # unchanged: it still compares against the pre-cut state.
            if prev_hist is not None and prev_prev_hist is not None:
                distance_prev = cv2.compareHist( prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA )
                distance_prev_prev = cv2.compareHist( prev_prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA )
                if distance_prev >= threshold and distance_prev_prev >= threshold:
                    pending_cut_frame_idx = frame_idx
                    pending_cut_pre_hist = prev_prev_hist

            prev_prev_hist = prev_hist
            prev_hist = hist
            sample_idx += 1
            next_sample_frame = sample_idx * step_frames
        frame_idx += 1

    # A candidate still pending at EOF has no following sample to
    # confirm it against. Accept it: the alternative is silently losing
    # a real cut that happens to land on the last sampled frame, and a
    # spurious reset at the very end of a file costs nothing.
    if pending_cut_frame_idx is not None:
        cut_times.append( round( pending_cut_frame_idx / vid_fps, 3 ) )

    return cut_times


# ---------------------------------------------------------------------------
# Preview slice resolution
# ---------------------------------------------------------------------------

def resolve_preview_slice( vid_fps, num_frames, preview_mode_enabled,
                           preview_max_seconds, explicit_start, random_slice=False ):
    """
    Where in one video a preview render starts.

    One shared implementation so betatv.py and every tool that needs to
    predict a preview run's cache/output filename compute the same
    offset. They each used to keep a copy, and the copies drifted: a
    video shorter than preview_start_seconds falls back to the whole
    file with a plain '-preview' suffix, which neither copy modelled, so
    short videos silently never matched the cache the tools expected.

    Args:
        vid_fps: The video's frame rate.
        num_frames: The video's total frame count.
        preview_mode_enabled: Whether preview mode is on at all.
        preview_max_seconds: Length of the preview window.
        explicit_start: betaconfig.preview_start_seconds, or None.
        random_slice: betaconfig.preview_random_slice. Only consulted
            when explicit_start is None.

    Returns:
        A (preview_offset_seconds, preview_window_unlimited) pair.
        preview_window_unlimited is True when the requested start is
        past the end of this video, meaning it is processed in full
        instead of as a slice.
    """
    preview_offset_seconds = 0.0
    preview_window_unlimited = False
    if preview_mode_enabled and vid_fps:
        video_duration_seconds = num_frames / vid_fps
        latest_possible_start = max( 0.0, video_duration_seconds - preview_max_seconds )
        if explicit_start is not None and explicit_start >= video_duration_seconds:
            preview_window_unlimited = True
            preview_offset_seconds = 0.0
        elif explicit_start is not None:
            preview_offset_seconds = min( max( 0.0, explicit_start ), latest_possible_start )
        elif random_slice and latest_possible_start > 0:
            import random
            preview_offset_seconds = random.uniform( 0.0, latest_possible_start )
    return preview_offset_seconds, preview_window_unlimited


def preview_cache_suffix( preview_mode_enabled, preview_offset_seconds ):
    """Deprecated alias. The real implementation lives in betautils_cache_paths."""
    return bu_cache.preview_cache_suffix( preview_mode_enabled, preview_offset_seconds )


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def probe_video_ok( filepath, timeout=120 ):
    """
    Cheap sanity check that a rendered file is a real, openable video.

    Reads the container header only. NOT a frame-accurate integrity
    check: that needs ffprobe -count_frames, which forces a full decode
    and is far too slow to run after every render chunk. It catches the
    failure this actually guards against - a file left truncated because
    the process writing it was killed - but it cannot notice a chunk
    that is valid and simply short. Frame-count verification for that
    case lives in betautils_render.

    Returns:
        True if ffprobe reports a video stream, False on any failure.
    """
    command = [ "ffprobe", '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', filepath ]
    try:
        result = subprocess.run( command, capture_output=True, timeout=timeout )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return b'video' in result.stdout


def video_file_has_audio( filepath, timeout=120 ):
    """
    Whether a file contains an audio stream.

    Returns False rather than raising when ffprobe fails, so a probe
    failure degrades to "render without audio" instead of aborting the
    file. The earlier version let the exception propagate.
    """
    command = [ "ffprobe", '-loglevel', 'error',
                '-show_entries', 'stream=index,codec_type', '-of', 'csv=p=0', filepath ]
    try:
        result = subprocess.run( command, capture_output=True, timeout=timeout )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return 'audio' in result.stdout.decode( 'utf-8', errors='replace' )


def probe_stream_info( filepath, timeout=120 ):
    """
    Codec, dimensions, frame rate and duration for a video's first stream.

    Used by the benchmark harness to report what a measurement was taken
    against, and by the render verifier.

    Returns:
        A dict with 'codec', 'width', 'height', 'avg_frame_rate',
        'nb_frames' and 'duration' where ffprobe reports them, or {} on
        any failure.
    """
    command = [ "ffprobe", '-v', 'error', '-select_streams', 'v:0',
                '-show_entries',
                'stream=codec_name,width,height,avg_frame_rate,nb_frames,duration',
                '-of', 'default=noprint_wrappers=1:nokey=0', filepath ]
    try:
        result = subprocess.run( command, capture_output=True, timeout=timeout, text=True )
    except Exception:
        return {}
    if result.returncode != 0:
        return {}
    info = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split( '=', 1 )
            info[key.strip()] = value.strip()
    return info


# ---------------------------------------------------------------------------
# Opening a capture, with the hardware-decode fallback ladder
# ---------------------------------------------------------------------------
#
# Rather than maintaining a list of "known bad" codecs, this detects the
# actual failure at runtime - a read that comes back with no frame - and
# retries that one file with progressively more invasive fallbacks. That
# is more robust than sniffing a codec name: the real problem is "this
# box's decoder cannot handle this particular stream", not the codec
# label, so it also catches a future codec/file combination without
# being told about it.
#
# It is deliberately narrow. Genuine bitstream corruption usually still
# produces *a* frame through ffmpeg's error concealment, so this ladder
# will not fire for - and is not meant to fix - that case.

def _hw_accel_prop_available():
    """Whether this cv2 build exposes the per-capture hwaccel override."""
    return hasattr( cv2, 'CAP_PROP_HW_ACCELERATION' ) and hasattr( cv2, 'VIDEO_ACCELERATION_NONE' )


def _reopen_with_hwaccel_prop_none( cap, path ):
    """
    Reopen with OpenCV's own per-capture hwaccel override forced off.

    Scoped to this one VideoCapture, so it cannot affect any other
    file's decode path. Returns False untouched when the build lacks the
    property, so the caller moves on rather than treating "not
    available" as "failed".
    """
    if not _hw_accel_prop_available():
        return False
    cap.release()
    cap.open( path, cv2.CAP_FFMPEG, [ cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE ] )
    return cap.isOpened()


def _reopen_with_hwaccel_env_none( cap, path ):
    """
    Reopen with hwaccel disabled through ffmpeg's own option parser.

    A different mechanism from the property above: this string is read
    by avformat directly rather than going through OpenCV's videoio
    abstraction, and the two do not behave identically on every
    platform/codec combination. The variable is set immediately before
    the reopen and restored immediately after, so it cannot leak into
    another file's open.
    """
    cap.release()
    previous = os.environ.get( 'OPENCV_FFMPEG_CAPTURE_OPTIONS' )
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'hwaccel;none'
    try:
        cap.open( path )
    finally:
        if previous is None:
            os.environ.pop( 'OPENCV_FFMPEG_CAPTURE_OPTIONS', None )
        else:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = previous
    return cap.isOpened()


def _transcode_to_h264( source_path, file_hash, logger, timeout=3600 ):
    """
    Transcode a file to H.264 with the system ffmpeg, and cache the result.

    The last-resort fallback, and the only one confirmed to work on real
    hardware: OpenCV here is linked against its own bundled ffmpeg whose
    AV1 hwaccel negotiation fails regardless of either hwaccel-disabling
    flag above (both produced byte-identical failures). The system ffmpeg
    binary decodes the same file cleanly, so this sidesteps OpenCV's
    decoder entirely rather than trying to persuade it.

    The transcoded copy is kept under
    ../output/cache/transcode_cache/<file_hash>.mp4 so resumes and
    re-runs reuse it. There is no automatic eviction.

    Returns:
        The transcoded path on success, or None.
    """
    ffmpeg_bin = shutil.which( 'ffmpeg' )
    if not ffmpeg_bin:
        logger.warning( "no system ffmpeg on PATH - cannot transcode %s as a decode fallback"%(
            source_path ) )
        return None

    transcoded_path = bu_cache.transcode_path_for( file_hash )
    if os.path.exists( transcoded_path ) and os.path.getsize( transcoded_path ) > 0:
        logger.info( "reusing cached transcode: %s"%(transcoded_path) )
        return transcoded_path

    logger.warning( "transcoding %s to H.264 via system ffmpeg as a decode fallback - "
                    "this can take a while on a long video"%(source_path) )
    try:
        result = subprocess.run(
            [ ffmpeg_bin, '-y', '-i', source_path, '-c:v', 'libx264',
              '-preset', 'veryfast', '-crf', '18', '-c:a', 'copy', transcoded_path ],
            capture_output=True, timeout=timeout, text=True )
    except subprocess.TimeoutExpired:
        logger.warning( "system ffmpeg transcode timed out after %ds for %s"%(timeout, source_path) )
        _remove_quietly( transcoded_path )
        return None

    if result.returncode != 0 or not os.path.exists( transcoded_path ) \
            or os.path.getsize( transcoded_path ) == 0:
        logger.warning( "system ffmpeg transcode failed for %s (exit %s): %s"%(
            source_path, result.returncode, ( result.stderr or '' )[-2000:] ) )
        _remove_quietly( transcoded_path )
        return None

    logger.info( "transcode complete: %s"%(transcoded_path) )
    return transcoded_path


def _remove_quietly( path ):
    """Delete a path, ignoring the case where it is already gone."""
    try:
        if os.path.exists( path ):
            os.remove( path )
    except OSError:
        pass


class VideoSource:
    """
    An openable, decodable view of one source video.

    Holds both paths on purpose. After a transcode fallback these differ,
    and confusing them is a real bug this class exists to prevent:
    previously the transcode repointed only the main capture, so the
    shot-cut scanner still opened the original AV1 file that OpenCV had
    just proven it could not decode, got no frames, and silently cached
    an empty cut list for the whole video.

    Attributes:
        original_path: The user's file. Use this, and only this, as the
            audio source for the final mux.
        decode_path: The file every cv2.VideoCapture should be opened
            on. Equal to original_path unless a transcode was needed.
        vid_fps: Frame rate of decode_path.
        num_frames: Frame count of decode_path. Re-read after a
            transcode, because a re-encode can shift the count by a
            frame or two and the render chunk planner needs the real
            number - planning past the end used to truncate the output
            silently.
        width, height: Frame dimensions, from a decoded frame rather
            than container metadata.
        transcoded: Whether a transcode fallback was used.
    """

    def __init__( self, original_path, decode_path, cap, vid_fps, num_frames,
                  width, height, transcoded ):
        self.original_path = original_path
        self.decode_path = decode_path
        self.cap = cap
        self.vid_fps = vid_fps
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.transcoded = transcoded

    def open_capture( self ):
        """
        A fresh, independent cv2.VideoCapture on decode_path.

        Used by anything that needs its own read position - the
        shot-cut scan, and each worker in a parallel render.
        """
        return cv2.VideoCapture( self.decode_path )

    def release( self ):
        """Release the primary capture."""
        if self.cap is not None:
            self.cap.release()

    def __enter__( self ):
        return self

    def __exit__( self, exc_type, exc, tb ):
        self.release()
        return False


class VideoOpenError( RuntimeError ):
    """Raised when no decode strategy can produce a frame from a file."""


def open_video_source( path, file_hash, logger=None ):
    """
    Open a video for decoding, applying the hardware-decode fallback
    ladder if the first read produces no frame.

    Args:
        path: The user's source video.
        file_hash: Content hash, used as the transcode cache key.
        logger: Optional logger.

    Returns:
        A VideoSource positioned at frame 0.

    Raises:
        VideoOpenError: the file is not an openable video, or no
            strategy could decode a frame from it.
    """
    logger = logger or bu_log.get_logger()

    cap = cv2.VideoCapture( path )
    if not cap.isOpened():
        cap.release()
        raise VideoOpenError( "not an openable video: %s"%(path) )

    # fps and frame count are container metadata; reading them does not
    # consume a frame, so it is safe before the first read.
    vid_fps = cap.get( cv2.CAP_PROP_FPS )
    num_frames = cap.get( cv2.CAP_PROP_FRAME_COUNT )

    retrieved, frame = cap.read()
    decode_path = path
    transcoded = False

    if not retrieved or frame is None:
        logger.warning( "hardware-accelerated read produced no frame for %s - "
                        "trying software-decode fallbacks"%(path) )
        strategies = [
            ( "CAP_PROP_HW_ACCELERATION=NONE",
              lambda: _reopen_with_hwaccel_prop_none( cap, path ) ),
            ( "OPENCV_FFMPEG_CAPTURE_OPTIONS=hwaccel;none",
              lambda: _reopen_with_hwaccel_env_none( cap, path ) ),
        ]
        for description, strategy in strategies:
            if not strategy():
                logger.debug( "decode fallback unusable: %s"%(description) )
                continue
            retrieved, frame = cap.read()
            if retrieved and frame is not None:
                logger.info( "decode succeeded for %s using %s"%(path, description) )
                break
            logger.debug( "decode fallback did not help: %s"%(description) )

        if not retrieved or frame is None:
            transcoded_path = _transcode_to_h264( path, file_hash, logger )
            if transcoded_path:
                cap.release()
                cap = cv2.VideoCapture( transcoded_path )
                retrieved, frame = cap.read()
                if retrieved and frame is not None:
                    decode_path = transcoded_path
                    transcoded = True
                    # The re-encode can shift the frame count; the chunk
                    # planner must use the real one.
                    vid_fps = cap.get( cv2.CAP_PROP_FPS ) or vid_fps
                    num_frames = cap.get( cv2.CAP_PROP_FRAME_COUNT ) or num_frames
                    logger.info( "decoding %s via transcoded copy %s (%.0f frames @ %.3f fps)"%(
                        path, transcoded_path, num_frames, vid_fps ) )

    if not retrieved or frame is None:
        cap.release()
        raise VideoOpenError(
            "could not decode any frame from %s - hardware decode, both software-decode "
            "fallbacks, and a system-ffmpeg transcode all failed. The file may be corrupt "
            "or in a genuinely unsupported format."%(path) )

    height, width = frame.shape[:2]
    return VideoSource( path, decode_path, cap, vid_fps, num_frames, width, height, transcoded )


def shot_cuts_for_source( source, file_hash, sample_fps, threshold,
                          preview_offset_seconds=0.0, preview_max_seconds=None,
                          preview_suffix='', enabled=True, logger=None ):
    """
    Shot-cut timestamps for one video, cached on disk.

    Scans with an independent capture on source.decode_path, so a full
    sequential scan never disturbs the caller's own read position, and
    so a transcoded file is scanned through the copy that actually
    decodes.

    Args:
        source: A VideoSource.
        file_hash: Content hash of the source video.
        sample_fps: betaconfig.video_censor_fps.
        threshold: betaconfig.shot_cut_threshold.
        preview_offset_seconds: Where a preview slice starts.
        preview_max_seconds: Preview window length, or None for the
            whole file.
        preview_suffix: Cache-key suffix from
            betautils_cache_paths.preview_cache_suffix.
        enabled: False returns [] with no scan and no cache read, so a
            project that never sees quick-cut footage pays nothing.
        logger: Optional logger.

    Returns:
        A sorted list of cut timestamps in seconds.
    """
    logger = logger or bu_log.get_logger()
    if not enabled:
        return []

    cache_path = bu_cache.shot_cut_path_for(
        file_hash, sample_fps, threshold, preview_suffix, preview_max_seconds )
    if os.path.exists( cache_path ):
        try:
            cuts = bu_hash.read_json( cache_path )
            logger.debug( "shot cuts for %s: %d from cache"%(file_hash, len( cuts )) )
            return cuts
        except Exception as err:
            logger.warning( "corrupt shot-cut cache %s (%s), rescanning"%(cache_path, err) )
            _remove_quietly( cache_path )

    scan_cap = source.open_capture()
    try:
        if preview_offset_seconds:
            scan_cap.set( cv2.CAP_PROP_POS_FRAMES,
                          int( round( preview_offset_seconds * source.vid_fps ) ) )
        cut_times = detect_shot_cuts( scan_cap, source.vid_fps, sample_fps,
                                      threshold, preview_max_seconds )
    finally:
        scan_cap.release()

    logger.info( "shot-cut scan for %s: %d cut(s)"%(
        os.path.basename( source.original_path ), len( cut_times ) ) )
    bu_hash.write_json( cut_times, cache_path )
    return cut_times
