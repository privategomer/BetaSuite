"""
betautils_tuning.py - the machinery behind automatic, data-derived tuning.

tools/tuning/auto_tune.py is the command; this module is what it is made
of, kept separate so the measurement, the decision rules and the config
writer can each be tested on their own.

THE MODEL
---------
A tuning run is a sequence of STAGES. Each stage owns exactly one
question ("how long should a track stay matchable?"), a list of
CANDIDATE values for it, and a DECISION RULE that reads measurements and
picks a winner. Stages run one at a time, and a stage's accepted value is
written to config before the next stage starts, so later stages measure
against earlier decisions rather than against the original config. That
ordering is the whole point: tuning several interacting knobs
simultaneously tells you about the combination, not about any one knob.

Analogy: this is a bracket, not a battle royale. One matchup at a time,
the winner advances, and you always know which change caused which
result.

MEASUREMENT
-----------
Every candidate is scored by REPLAYING the detection caches a real run
already wrote, through the real betautils_track pipeline, with the
candidate's config applied. No model runs, so a full sweep takes seconds
rather than hours, and every configuration on disk is measured, not just
the selected one.

Replay covers everything downstream of raw detections: suppression,
geometry filtering, cross-size dedup, tracking, smoothing, interpolation
and track confirmation. It does NOT cover anything that changes the raw
detections themselves - picture size, model variant, global_min_prob -
because those need the model re-run, and a tuner that silently replayed
stale detections under a changed detection setting would be measuring a
lie. Stages that would change detections are refused rather than
approximated.

THE SAFETY RULE
---------------
Performance work must never cost valid detections, and by extension must
never weaken censoring. That is not a guideline here, it is a guard every
candidate passes through: a candidate that reduces censor COVERAGE - the
total censored area-seconds, per label - by more than a small tolerance
is rejected no matter how good its other numbers look. A faster or
tidier configuration that leaves skin uncovered is not an improvement,
and no scoring function gets to trade that away.
"""

import contextlib
import copy
import json
import os
import re
import shutil
import time

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_censor as bu_censor
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_log as bu_log
import betautils_track as bu_track
import betautils_video as bu_video


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class ReplayObserver( bu_track.TrackingObserver ):
    """
    Counts the tracking decisions a replay makes.

    The three that matter for tuning:
      matches        a detection continued an existing track
      new_tracks     a detection started a fresh one, which on already
                     visible content is a visible snap
      risky          two candidate boxes competed for one track and the
                     closest two were within 25% of each other, so the
                     greedy assignment had real ambiguity to resolve

    Risky contention is the cost side of every loosening change. Without
    it, "fewer track resets" looks free, and it is not: past some point
    a looser threshold stops rescuing broken tracks and starts merging
    two different subjects into one.
    """

    RISKY_RATIO = 0.25

    def __init__( self ):
        self.matches = 0
        self.new_tracks = 0
        self.contention = 0
        self.risky_contention = 0

    def on_match( self, label, track_index, distance, box, tracks, settings ):
        self.matches += 1

    def on_new_track( self, label, tracks, track_index, box, settings ):
        self.new_tracks += 1

    def on_frame_candidates( self, label, candidates, tracks, settings ):
        by_track = {}
        for distance, _box, track_index in candidates:
            by_track.setdefault( track_index, [] ).append( distance )
        for distances in by_track.values():
            if len( distances ) < 2:
                continue
            self.contention += 1
            distances.sort()
            closest, second = distances[0], distances[1]
            if second and ( second - closest ) / second <= self.RISKY_RATIO:
                self.risky_contention += 1


REPLAY_SEED = 0


@contextlib.contextmanager
def _style_choice_held_fixed():
    """
    Make every censor style resolve to the same entry for this replay.

    WHY SEEDING WAS NOT ENOUGH
    --------------------------
    censor_style is a weighted list drawn from at random, once per
    tracked instance. Seeding the RNG makes a replay reproducible, but
    it does NOT make two replays comparable, because the knobs under
    test change how many draws happen and in what order: a different
    track_max_gap produces a different number of tracked instances, and
    paired_style re-resolves styles as tracking changes. One extra draw
    early shifts every style after it, so two candidates end up with
    different styles on the same detections.

    That matters because each style carries its own width_area_safety
    and height_area_safety - for exposed_breast those span -0.60 to
    +0.50 - and smooth_boxes recomputes box geometry from them. Coverage
    is an area measure, so the shifted styles showed up as coverage
    swinging 5-20% in both directions on changes that moved the rendered
    box count by a fraction of a percent. Those were false REJECTIONS:
    real candidates thrown out for a difference that was entirely the
    style lottery.

    Holding the choice fixed is ordinary experimental control. Style
    selection is orthogonal to every tracking knob this tool tunes, so
    freezing it removes the noise without removing any signal. The
    highest-weighted entry is chosen so the held style is the one the
    footage would most often see anyway.

    Analogy: you are timing runners, so you put them all on the same
    track. Not because the track does not matter, but because it is not
    what you are measuring today.
    """
    original = bu_censor.resolve_censor_style

    def held( style_config ):
        if isinstance( style_config, dict ):
            return style_config
        if not style_config:
            return original( style_config )
        return max( style_config, key=lambda entry: entry.get( 'weight', 1 ) )

    bu_censor.resolve_censor_style = held
    try:
        yield
    finally:
        bu_censor.resolve_censor_style = original


# Frame dimensions per source video, cached for the life of the process.
#
# measure_replay needs each video's width and height to turn a box area
# into a fraction of the frame. Those come from cv2.VideoCapture, which
# opens the actual video file - and they cannot change between replays of
# the same file, because the file is the same file.
#
# Every replay was re-opening all 7 source videos for two integers. One
# auto_tune run does 87 replays, so 609 file opens measured nothing. The
# cache makes that 7.
#
# Keyed on the resolved path, so two hashes pointing at one file share an
# entry and a moved file is looked up afresh. A video that cannot be
# opened caches (0, 0) so the failure is not retried 86 more times.
_FRAME_SIZE_CACHE = {}


def frame_size_for_video( video_path ):
    """
    (width, height) of a source video, read once per process.

    Args:
        video_path: Path to the source video.

    Returns:
        (width, height) as ints, or (0, 0) when the file cannot be read -
        callers treat that as 'skip this video', same as before.
    """
    import cv2

    if video_path in _FRAME_SIZE_CACHE:
        return _FRAME_SIZE_CACHE[video_path]
    capture = cv2.VideoCapture( video_path )
    try:
        size = ( int( capture.get( cv2.CAP_PROP_FRAME_WIDTH ) ),
                 int( capture.get( cv2.CAP_PROP_FRAME_HEIGHT ) ) )
    finally:
        capture.release()
    _FRAME_SIZE_CACHE[video_path] = size
    return size


def clear_replay_caches():
    """
    Drop the process-wide replay caches.

    Only the tests need this: within one tool run the cached values are
    properties of files on disk that the tool is not changing.
    """
    _FRAME_SIZE_CACHE.clear()
    _RAW_BOX_CACHE.clear()


# Parsed cache-file contents, keyed on (path, mtime, size).
#
# A replay reads every detection cache for every video, and those files
# are large - 182616 detections across 7 files for 640m. Re-parsing that
# JSON on all 87 replays of an auto_tune run is the single biggest
# non-tracking cost in the tool, and the parsed result is identical every
# time because auto_tune never writes detection caches.
#
# The mtime and size are in the key so an externally rewritten cache is
# re-read rather than served stale. Entries are deep-copied on the way
# out: prepare_boxes_for_render mutates the boxes it is handed, and a
# shared list would let one replay corrupt the next.
_RAW_BOX_CACHE = {}


def _parsed_caches( cache_paths ):
    """The cached parse of each readable cache file, shared, never copied."""
    parsed = []
    for path in cache_paths:
        if not os.path.exists( path ):
            continue
        try:
            stat = os.stat( path )
            key = ( path, stat.st_mtime_ns, stat.st_size )
            if key not in _RAW_BOX_CACHE:
                _RAW_BOX_CACHE[key] = bu_hash.read_json( path )
            parsed.append( _RAW_BOX_CACHE[key] )
        except Exception:
            continue
    return parsed


def _read_raw_boxes( cache_paths ):
    """
    Read and concatenate detection caches, reusing parsed results.

    Args:
        cache_paths: Paths to one video's detection cache files.

    Returns:
        A fresh list of box dicts, safe for the caller to mutate.
    """
    boxes = []
    for entries in _parsed_caches( cache_paths ):
        boxes.extend( copy.deepcopy( entries ) )
    return boxes


def raw_boxes_for_reading( cache_paths ):
    """
    The same detections without the deep copy, for a caller that only reads.

    The parse is shared with every later replay, so mutating these dicts
    corrupts them. On a large split-screen cache the copy is seconds.
    """
    boxes = []
    for entries in _parsed_caches( cache_paths ):
        boxes.extend( entries )
    return boxes


def forget_raw_boxes( cache_paths ):
    """Drop one video's parsed caches, for a caller that will not read them again."""
    wanted = set( cache_paths )
    for key in [ key for key in _RAW_BOX_CACHE if key[0] in wanted ]:
        del _RAW_BOX_CACHE[key]


def replay_tracking_context( file_hash, video_path, cache_paths, backend_name,
                             shot_cuts_by_hash=None ):
    """
    How a real run tracks this video: its cuts, profile and sample rate.

    Every replay (auto_tune, the track-break, jitter and flicker
    analysers) must track under the SAME structure profile a real run of
    the video selects, from the same measurement, through the same
    selector, including the folder convention, and must stop tracks at
    the same cuts. A replay that skips any of it measures a
    configuration that never runs: the profiles override track_max_gap,
    match distance and min_track_hits per label, so un-profiled replays
    swept values no real run reads.

    The scan window comes from the cut cache's own filename via
    scanned_window_seconds. It used to be the spread of the replayed
    detections, which measures the nudity rather than the footage.

    Args:
        file_hash: The video's hash.
        video_path: Its source path, for the folder rule and duration.
        cache_paths: Its detection cache paths; the first one's name
            gives the sample rate and preview suffix actually used.
        backend_name: The backend whose profiles apply.
        shot_cuts_by_hash: Optional {file_hash: cuts} already on the
            detection timeline, covering the whole file.

    Returns:
        A dict with shot_cut_times (absolute, or None when the video was
        never scanned), profile_name, sample_fps and signals.
    """
    detection_name = ( bu_cache.parse_any_cache_name( cache_paths[0] ) or {} ) if cache_paths else {}
    duration_seconds = bu_video.probe_duration_seconds( video_path )
    if shot_cuts_by_hash is not None and file_hash in shot_cuts_by_hash:
        tracking_cuts = shot_cuts_by_hash[file_hash]
        signals = bu_detector.structure_signals( tracking_cuts, duration_seconds )
    else:
        cached = bu_video.cached_shot_cuts(
            file_hash, betaconfig.video_censor_fps,
            getattr( betaconfig, 'shot_cut_threshold', 0.5 ), duration_seconds,
            prefer_preview_suffix=detection_name.get( 'preview_suffix', '' ) )
        if cached is None:
            tracking_cuts, signals = None, bu_detector.structure_signals( None, None )
        else:
            tracking_cuts = bu_video.absolute_cut_times( cached.cuts, cached.offset_seconds )
            signals = bu_detector.structure_signals( cached.cuts, cached.scanned_seconds )
    profile_name = bu_detector.select_profile_name(
        signals.cuts_per_min, backend_name, source_path=video_path,
        short_shot_fraction=signals.short_shot_fraction )
    # The rate the replayed detections were actually sampled at, read
    # from their cache name. The profile's own rate describes a run that
    # may never have happened on this data.
    sample_fps = ( detection_name.get( 'fps' )
                   or bu_detector.get_profile_sample_fps( profile_name, backend_name ) )
    return { 'shot_cut_times': tracking_cuts, 'profile_name': profile_name,
             'sample_fps': sample_fps, 'signals': signals }


_QUIET_LOGGER = bu_log.silent_logger( 'betasuite.replay.quiet' )


def replay_tracking_inputs( file_hash, video_path, cache_paths, raw_boxes, vid_w, vid_h,
                            backend_name, labels=None ):
    """
    (boxes, tracking) for an analyser that replays tracking itself.

    boxes are what a real run hands smooth_boxes (promotion, geometry,
    suppression and all, via bu_track.prepare_tracking_inputs), narrowed
    to `labels` when given; tracking is replay_tracking_context's dict,
    whose shot_cut_times, profile_name and sample_fps go straight into
    smooth_boxes. raw_boxes is consumed; pass a copy.
    """
    tracking = replay_tracking_context( file_hash, video_path, cache_paths, backend_name )
    boxes, _stats = bu_track.prepare_tracking_inputs( raw_boxes, vid_w, vid_h, backend_name,
                                                      logger=_QUIET_LOGGER )
    if labels:
        boxes = [ box for box in boxes if box['label'] in labels ]
    return boxes, tracking


def replay_rendered( file_hash, video_path, cache_paths, raw_boxes, vid_w, vid_h, backend_name,
                     logger=None, tracking=None ):
    """
    (rendered, stats, tracking) for one video, exactly as a run renders it.

    The whole production pipeline (bu_track.prepare_boxes_for_render)
    with the profile, shot cuts and sample rate a real run would use
    (replay_tracking_context). For the per-video replay tools, which
    used to rebuild suppression -> process_raw_box -> smooth_boxes by
    hand and so replayed without promotion, the geometry filter, cuts
    or the profile's overrides. raw_boxes is consumed; pass a copy.
    tracking: replay_tracking_context's dict when the caller already has it.
    """
    if tracking is None:
        tracking = replay_tracking_context( file_hash, video_path, cache_paths, backend_name )
    rendered, stats = bu_track.prepare_boxes_for_render(
        raw_boxes, vid_w, vid_h, shot_cut_times=tracking['shot_cut_times'],
        backend_name=backend_name, logger=logger or _QUIET_LOGGER,
        profile_name=tracking['profile_name'], sample_fps=tracking['sample_fps'] )
    return rendered, stats, tracking


@contextlib.contextmanager
def class_suppression_in_effect( backend_name, rules ):
    """The production pipeline reads `rules` as backend_name's class_suppression, then the old ones."""
    block = betaconfig.detector_backend.setdefault( backend_name, {} )
    had = 'class_suppression' in block
    saved = copy.deepcopy( block.get( 'class_suppression' ) )
    block['class_suppression'] = copy.deepcopy( rules )
    bu_config.invalidate_config_caches()
    try:
        yield
    finally:
        if had:
            block['class_suppression'] = saved
        else:
            block.pop( 'class_suppression', None )
        bu_config.invalidate_config_caches()


@contextlib.contextmanager
def item_overrides_in_effect( backend_name, patch, profile_name=None ):
    """
    The pipeline reads patch ({label: {key: value}}) whichever tier would shadow it.

    A key resolves shared tier < backend < active variant < profile, so
    writing only the shared betaconfig.item_overrides (what replay_tune
    used to do) is overridden by any backend, variant or profile that
    sets the same key, and the variant silently replays the baseline.
    Backend-tunable keys go to the backend tier, and also to the active
    variant's and `profile_name`'s tier where those set the key; the
    rest (censor_style, censor_shape) go to the shared tier, the only
    one that reads them. Everything is restored on exit.
    """
    saved_shared = copy.deepcopy( getattr( betaconfig, 'item_overrides', {} ) )
    had_backend = backend_name in betaconfig.detector_backend
    saved_backend = copy.deepcopy( betaconfig.detector_backend.get( backend_name ) )
    try:
        block = betaconfig.detector_backend.setdefault( backend_name, {} )
        variant_name = bu_detector.selected_variant_name( backend_name )
        shadowing = [
            ( block.get( 'variants' ) or {} ).get( variant_name ) if variant_name else None,
            ( ( block.get( 'profiles' ) or {} ).get( 'variants' ) or {} ).get( profile_name )
            if profile_name else None ]
        shared = betaconfig.item_overrides = copy.deepcopy( saved_shared )
        for label, keys in ( patch or {} ).items():
            for key, value in keys.items():
                if key not in bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS:
                    shared.setdefault( label, {} )[key] = copy.deepcopy( value )
                    continue
                block.setdefault( 'item_overrides', {} ).setdefault( label, {} )[key] = value
                for tier in shadowing:
                    overrides = ( tier or {} ).get( 'item_overrides' ) or {}
                    if key in overrides.get( label, {} ):
                        overrides[label][key] = value
        bu_config.invalidate_config_caches()
        yield
    finally:
        betaconfig.item_overrides = saved_shared
        if had_backend:
            betaconfig.detector_backend[backend_name] = saved_backend
        else:
            betaconfig.detector_backend.pop( backend_name, None )
        bu_config.invalidate_config_caches()


def profile_masking( profiles, label, key, backend_name ):
    """
    {profile_name: value} for profiles among `profiles` that set label's key.

    A profile's item_overrides layer on top of the backend's, so a file
    running one of these profiles never reads the backend value of key.
    profiles is {file_hash: profile_name} as measure_replay reports it.
    """
    masking = {}
    for profile_name in set( profiles.values() ):
        overrides = bu_detector.get_profile_item_overrides( profile_name, label, backend_name )
        if key in overrides:
            masking[profile_name] = overrides[key]
    return masking


# A detection at or above this score is treated as probably real when a
# replay asks how many of them a candidate leaves uncensored. Most
# detections of the censored labels clear it (exposed_breast's median is
# 0.76, exposed_vulva's 0.55); a short track of them being dropped is far
# more likely a brief real exposure than noise.
CONFIDENT_SCORE = 0.5


def confident_detections( raw_boxes, labels, confident_score ):
    """{label: [(t, centre_x, centre_y), ...]} for raw detections at or above confident_score."""
    found = {}
    for raw in raw_boxes:
        label = raw.get( 'class_id' )
        if label not in labels or raw.get( 'score', 0.0 ) < confident_score:
            continue
        found.setdefault( label, [] ).append(
            ( raw['t'], raw['x'] + raw['w'] / 2.0, raw['y'] + raw['h'] / 2.0 ) )
    return found


def label_content( raw_boxes, labels, sample_fps, confident_score=CONFIDENT_SCORE ):
    """
    How much of a video shows each label, from its raw detections.

    A sample counts for a label when at least one detection of it scores
    confident_score or more there. Seconds are samples over sample_fps,
    so a 9 fps cache and an 18 fps one measure the same footage alike.
    mixed_seconds: samples where two or more of `labels` are present at
    once, the footage where one label's suppression rule can remove
    another's censor. pair_seconds: the same per pair ('a+b', sorted).

    Returns:
        {'seconds': {label: s}, 'mixed_seconds': s, 'pair_seconds': {pair: s}}
        with every label present, zero when absent.
    """
    rate = float( sample_fps or 0 ) or 1.0
    times = { label: set() for label in labels }
    for raw in raw_boxes:
        label = raw.get( 'class_id' )
        if label in times and raw.get( 'score', 0.0 ) >= confident_score:
            # Sample times are k / fps as floats; rounding makes two
            # detections of one sample land on one key.
            times[label].add( round( raw['t'] * rate ) )
    present = {}
    for label, samples in times.items():
        for sample in samples:
            present.setdefault( sample, set() ).add( label )
    pairs = {}
    ordered = sorted( labels )
    for first_index, first in enumerate( ordered ):
        for second in ordered[ first_index + 1: ]:
            pairs['%s+%s'%( first, second )] = 0
    mixed = 0
    for found in present.values():
        if len( found ) < 2:
            continue
        mixed += 1
        found = sorted( found )
        for first_index, first in enumerate( found ):
            for second in found[ first_index + 1: ]:
                pairs['%s+%s'%( first, second )] += 1
    return {
        'seconds': { label: round( len( samples ) / rate, 1 ) for label, samples in sorted( times.items() ) },
        'mixed_seconds': round( mixed / rate, 1 ),
        'pair_seconds': { pair: round( count / rate, 1 ) for pair, count in sorted( pairs.items() ) },
    }


def content_by_video( videos, labels, confident_score=CONFIDENT_SCORE, forget=True ):
    """
    {file_hash: label_content(...)} for {file_hash: cache_paths}.

    Each video's rate comes from its own cache name, so quick-cut videos
    cached at their profile's rate count in seconds like the rest.
    forget drops each parsed cache once measured; a library's worth held
    at once runs to gigabytes.
    """
    found = {}
    for file_hash, cache_paths in sorted( videos.items() ):
        name = bu_cache.parse_any_cache_name( cache_paths[0] ) if cache_paths else None
        rate = ( name or {} ).get( 'fps' ) or betaconfig.video_censor_fps
        found[file_hash] = label_content( raw_boxes_for_reading( cache_paths ), labels, rate,
                                          confident_score )
        if forget:
            forget_raw_boxes( cache_paths )
    return found


def uncovered_detections( detections, rendered ):
    """
    {label: how many of detections no rendered box of that label covers}.

    Covered means a rendered box of the same label is live at the
    detection's time (start <= t <= end) and contains its centre. Centre
    rather than overlap, because rendered boxes are padded or shrunk by
    the style's area safety and a centre survives either.
    """
    import bisect

    by_label = {}
    for box in rendered:
        by_label.setdefault( box.get( 'label' ), [] ).append( box )
    uncovered = {}
    for label, points in detections.items():
        boxes = sorted( by_label.get( label, [] ), key=lambda box: box['start'] )
        starts = [ box['start'] for box in boxes ]
        longest = max( ( box['end'] - box['start'] for box in boxes ), default=0.0 )
        missed = 0
        for t, centre_x, centre_y in points:
            index = bisect.bisect_right( starts, t ) - 1
            covered = False
            while index >= 0 and starts[index] >= t - longest:
                box = boxes[index]
                if ( box['end'] >= t and box['x'] <= centre_x <= box['x'] + box['w']
                     and box['y'] <= centre_y <= box['y'] + box['h'] ):
                    covered = True
                    break
                index -= 1
            if not covered:
                missed += 1
        uncovered[label] = missed
    return uncovered


def measure_replay( videos, hash_to_path, backend_name, labels, shot_cuts_by_hash=None,
                    seed=REPLAY_SEED, confident_score=CONFIDENT_SCORE, cut_transform=None,
                    cut_bursts=None ):
    """
    Replay one configuration's caches and return everything a decision needs.

    WHY THIS SEEDS THE RNG, AND WHY IT IS NOT OPTIONAL
    --------------------------------------------------
    A label's censor_style is usually a weighted LIST, and
    betautils_censor.resolve_censor_style draws from it at random, once
    per box. Each style carries its own width_area_safety and
    height_area_safety - for exposed_breast those span -0.60 to +0.50 -
    and smooth_boxes recomputes each box's geometry from the style it
    drew. So the same detections, replayed twice, produce boxes of
    substantially different areas.

    Coverage is an area measure. Unseeded, two replays of the IDENTICAL
    configuration disagreed by 35-45% on coverage while their rendered
    box counts differed by 0.2%, and a baseline measured in three
    consecutive stages came back as 285, 265 and 268 track resets. Every
    comparison built on that is a comparison of two coin flips.

    Seeding identically before each replay makes the style draw the same
    sequence every time, so style contributes the same thing to every
    candidate and cancels out of the comparison. What is left moving is
    the knob under test. Interpolated boxes inherit their track's
    already-resolved style rather than drawing again, so they do not
    disturb the sequence; paired_style CAN re-resolve as tracking
    changes, and that is a real effect of the knob rather than noise.

    Analogy: this is a blind taste test. Both glasses have to come from
    the same bottle in the same order, or you are measuring the pour.

    Args:
        videos: {file_hash: [cache_path, ...]}, from discover_full_run_videos.
        hash_to_path: {file_hash: source video path}, for real frame sizes.
        backend_name: Which backend's settings apply. Pinned, never defaulted.
        labels: The censored labels to report coverage for.
        shot_cuts_by_hash: Optional {file_hash: [cut_time, ...]}, so a
            replay respects scene cuts exactly as a real run does.
        seed: The RNG seed every replay starts from. Change it to check
            that a result is not an artefact of one particular draw;
            never vary it WITHIN a comparison.
        confident_score: Detections at or above this score are counted
            in confident_by_label, and those no rendered box covers in
            confident_uncovered_by_label.
        cut_transform: Optional f(cuts) -> cuts applied to the cuts
            tracking stops at, after the profile is chosen from the raw
            ones. analyze_cut_bursts uses it to try merging cut bursts
            without changing which profile a file gets. Given, it
            replaces the configured burst merge rather than stacking on it.
        cut_bursts: The burst merge to replay with (see
            bu_track.tracking_cut_times): None follows the config, as a
            real run does; False replays the cuts as scanned.

    Returns:
        A metrics dict: rendered/interpolated/track counts, the tracking
        decision counts from ReplayObserver, and per-label coverage in
        censored area-seconds.
    """
    import random

    random.seed( seed )
    observer = ReplayObserver()
    metrics = {
        'rendered_boxes': 0,
        'interpolated': 0,
        'tracks': 0,
        'dropped_provisional': 0,
        'dropped_unconfirmed': 0,
        'raw_detections': 0,
        'suppressed': 0,
        'geometry_filtered': 0,
        'videos_measured': 0,
        'coverage_by_label': { label: 0.0 for label in labels },
        'rendered_by_label': { label: 0 for label in labels },
        'confident_by_label': { label: 0 for label in labels },
        'confident_uncovered_by_label': { label: 0 for label in labels },
        # {file_hash: structure profile the replay ran it under}, so a
        # caller can tell which configured values a file never reads.
        'profiles': {},
    }

    for file_hash, cache_paths in sorted( videos.items() ):
        video_path = hash_to_path.get( file_hash )
        if not video_path:
            continue

        raw_boxes = _read_raw_boxes( cache_paths )
        if not raw_boxes:
            continue

        vid_w, vid_h = frame_size_for_video( video_path )
        if not vid_w or not vid_h:
            continue

        metrics['videos_measured'] += 1
        metrics['raw_detections'] += len( raw_boxes )
        frame_area = float( vid_w * vid_h )

        tracking = replay_tracking_context( file_hash, video_path, cache_paths, backend_name,
                                            shot_cuts_by_hash )
        tracking_cuts = tracking['shot_cut_times']
        replay_bursts = cut_bursts
        if cut_transform is not None:
            # The transform owns the cuts: merging them again by the
            # config would measure a different variant than the one named.
            replay_bursts = False
            if tracking_cuts:
                tracking_cuts = cut_transform( tracking_cuts )
        confident = confident_detections( raw_boxes, labels, confident_score )
        profile_name = tracking['profile_name']
        sample_fps = tracking['sample_fps']
        metrics['profiles'][file_hash] = profile_name

        with _style_choice_held_fixed():
            with bu_track.tracking_observer( observer ):
                # raw_boxes is already a fresh deep copy from
                # _read_raw_boxes, so prepare_boxes_for_render may mutate
                # it without touching the cached parse.
                # Quiet: a replay is not a run, and its per-video lines
                # (3,800 in one analysis pass) buried the production log.
                rendered, stats = bu_track.prepare_boxes_for_render(
                    raw_boxes, vid_w, vid_h,
                    shot_cut_times=tracking_cuts, backend_name=backend_name,
                    logger=_QUIET_LOGGER, profile_name=profile_name,
                    sample_fps=sample_fps, cut_bursts=replay_bursts )

        metrics['rendered_boxes'] += len( rendered )
        metrics['interpolated'] += stats.get( 'interpolated', 0 )
        metrics['tracks'] += stats.get( 'tracks', 0 )
        metrics['dropped_provisional'] += stats.get( 'dropped_provisional', 0 )
        metrics['dropped_unconfirmed'] += stats.get( 'dropped_unconfirmed', 0 )
        metrics['suppressed'] += sum(
            entry['total'] if isinstance( entry, dict ) else entry
            for entry in ( stats.get( 'suppression_counts' ) or {} ).values() )
        metrics['geometry_filtered'] += stats.get( 'geometry_filtered', 0 )

        for label, missed in uncovered_detections( confident, rendered ).items():
            metrics['confident_by_label'][label] += len( confident[label] )
            metrics['confident_uncovered_by_label'][label] += missed

        for box in rendered:
            label = box.get( 'label' )
            if label not in metrics['coverage_by_label']:
                continue
            duration = max( 0.0, float( box.get( 'end', 0 ) ) - float( box.get( 'start', 0 ) ) )
            area_fraction = ( box.get( 'w', 0 ) * box.get( 'h', 0 ) ) / frame_area
            metrics['coverage_by_label'][label] += area_fraction * duration
            metrics['rendered_by_label'][label] += 1

    metrics['matches'] = observer.matches
    metrics['new_tracks'] = observer.new_tracks
    metrics['contention'] = observer.contention
    metrics['risky_contention'] = observer.risky_contention
    metrics['coverage_total'] = sum( metrics['coverage_by_label'].values() )
    return metrics


def coverage_change( baseline, candidate ):
    """
    Per-label fractional change in censored area-seconds, candidate vs baseline.

    Negative means the candidate covers LESS of the frame for that label
    over the same footage, which is the definition of weaker censoring
    and the thing no tuning change is allowed to buy speed with.

    A label with no baseline coverage is skipped rather than reported as
    an infinite change: there is nothing to lose.

    Returns:
        {label: fractional_change}, e.g. {'exposed_breast': -0.004}.
    """
    changes = {}
    for label, before in baseline.get( 'coverage_by_label', {} ).items():
        if before <= 0:
            continue
        after = candidate.get( 'coverage_by_label', {} ).get( label, 0.0 )
        changes[label] = ( after - before ) / before
    return changes


def worst_coverage_loss( baseline, candidate ):
    """The largest fractional coverage LOSS across labels, as a positive number."""
    changes = coverage_change( baseline, candidate )
    if not changes:
        return 0.0
    return max( 0.0, -min( changes.values() ) )


# ---------------------------------------------------------------------------
# Writing decisions back into betaconfig.py
# ---------------------------------------------------------------------------

class ConfigWriteError( Exception ):
    """A config edit could not be made unambiguously, so nothing was written."""


class ConfigWriter:
    """
    Edits betaconfig.py in place, surgically, and proves the edit took.

    Editing a live config file from a script is a thing to do carefully
    rather than cleverly. The rules here:

      - every write takes a timestamped backup first
      - an edit targets exactly one assignment, located by its enclosing
        block, and is refused if the pattern matches zero times or more
        than once, rather than picking one
      - after writing, the value is read back through the real resolver
        in a fresh interpreter. A config file that parses but resolves
        to the wrong value is worse than a failed write, because the
        next stage would measure against a setting nobody chose
      - a refused edit raises. It never falls back to "close enough"

    Formatting, comments and key order are preserved: only the value
    text between the ':' or '=' and the following ',' or newline moves.
    """

    def __init__( self, config_path=None, backup_dir=None, logger=None ):
        self.config_path = config_path or os.path.join(
            os.path.dirname( os.path.abspath( __file__ ) ), 'betaconfig.py' )
        self.backup_dir = backup_dir
        self.logger = logger
        self.backups = []

    def _backup( self ):
        stamp = time.strftime( '%Y%m%d_%H%M%S' )
        directory = self.backup_dir or os.path.dirname( self.config_path )
        os.makedirs( directory, exist_ok=True )
        target = os.path.join( directory, 'betaconfig.py.%s.bak'%(stamp) )
        suffix = 0
        while os.path.exists( target ):
            suffix += 1
            target = os.path.join( directory, 'betaconfig.py.%s.%d.bak'%(stamp, suffix) )
        shutil.copy2( self.config_path, target )
        self.backups.append( target )
        if self.logger:
            self.logger.debug( "config backed up to %s"%(target) )
        return target

    def read( self ):
        with open( self.config_path, 'r', encoding='UTF-8' ) as fin:
            return fin.read()

    def set_item_override( self, backend_name, label, key, value ):
        """
        Set detector_backend[<backend>]['item_overrides'][<label>][<key>],
        adding the key if that block does not have it yet.

        An UPSERT, not an update. The first version of this refused a
        missing key on the grounds that a tuner should not invent config
        structure. That was the wrong call in practice: the tuner's most
        useful decisions are about keys a label has never had set -
        min_track_hits and match_distance_multiplier both default
        implicitly and therefore appear nowhere - so refusing them meant
        the tool measured carefully, decided correctly, and then told
        you to go and type it in yourself.

        An inserted key is placed as the LAST entry of that label's
        block, indented to match its siblings, with a trailing comma,
        and with a short comment naming the run that put it there. That
        last part matters: a value that appeared in your config without
        you typing it should say where it came from.

        Args:
            backend_name, label, key: Which setting.
            value: The value to set.
        """
        source = self.read()
        start, end = self._locate_label_block( source, backend_name, label )
        pattern = re.compile( r"(^\s*'%s'\s*:\s*)([^,\n]+)(,?)"%( re.escape( key ) ), re.M )
        matches = list( pattern.finditer( source, start, end ) )

        if len( matches ) > 1:
            raise ConfigWriteError(
                "found %d entries for '%s' inside detector_backend[%r]['item_overrides'][%r] - "
                "a duplicated key is ambiguous, so nothing was written; remove the duplicate "
                "and re-run"%( len( matches ), key, backend_name, label ) )

        if matches:
            match = matches[0]
            updated = ( source[:match.start()] + match.group(1) + self._format( value )
                        + ( match.group(3) or ',' ) + source[match.end():] )
        else:
            updated = self._insert_item_override( source, start, end, key, value )

        self._write_and_verify(
            updated,
            lambda: bu_detector.get_item_overrides( label, backend_name ).get( key ),
            value,
            "detector_backend[%r]['item_overrides'][%r][%r]"%( backend_name, label, key ) )

    def set_variant_item_override( self, backend_name, variant_name, label, key, value ):
        """
        Set a value on ONE variant, creating whatever blocks it needs.

        detector_backend[<backend>]['variants'][<variant>]['item_overrides'][<label>][<key>]

        This is the escape hatch for variants that genuinely disagree.
        Two sets of weights detect differently, so their tracking wants
        different settings, and the first real tuning run showed it: at
        match_distance_multiplier 2.0, nudenet_v3's 640m was better on
        BOTH axes while 320n rejected every candidate. With only a
        backend tier, that value simply could not be written.

        Unlike the backend tier, this creates the intermediate
        structure - 'variants', the variant's own block, its
        'item_overrides', and the label - because a variant block is
        pure tuning output with no hand-maintained content to preserve.
        The backend tier is the one that is hand-written and therefore
        only ever edited in place.
        """
        source = self.read()
        backend_at = source.find( "'%s': {"%(backend_name) )
        if backend_at < 0:
            raise ConfigWriteError( "no detector_backend block for %r"%(backend_name) )
        backend_end = self._matching_brace( source, source.index( '{', backend_at ) )

        variants_at = source.find( "'variants'", backend_at )
        if variants_at < 0 or variants_at > backend_end:
            updated = self._insert_variants_block(
                source, backend_at, backend_end, variant_name, label, key, value )
        else:
            variants_end = self._matching_brace( source, source.index( '{', variants_at ) )
            variant_at = source.find( "'%s': {"%(variant_name), variants_at )
            if variant_at < 0 or variant_at > variants_end:
                updated = self._insert_into_variants(
                    source, variants_end, variant_name, label, key, value )
            else:
                updated = self._set_within_variant(
                    source, variant_at, variant_name, label, key, value )

        self._write_and_verify(
            updated,
            lambda: bu_detector.get_item_overrides( label, backend_name, variant_name ).get( key ),
            value,
            "detector_backend[%r]['variants'][%r]['item_overrides'][%r][%r]"%(
                backend_name, variant_name, label, key ) )

    @staticmethod
    def _matching_brace( source, open_at ):
        depth = 0
        for position in range( open_at, len( source ) ):
            if source[position] == '{':
                depth += 1
            elif source[position] == '}':
                depth -= 1
                if depth == 0:
                    return position
        raise ConfigWriteError( "unbalanced braces in betaconfig.py" )

    def _insert_variants_block( self, source, backend_at, backend_end,
                                variant_name, label, key, value ):
        indent = self._indent_of_line( source, backend_at ) + '    '
        block = (
            "%s'variants': {\n"
            "%s    # Per-variant overrides, written by auto_tune %s. These merge\n"
            "%s    # ON TOP of this backend's own item_overrides, so a variant only\n"
            "%s    # names what it genuinely wants to differ on.\n"
            "%s    '%s': {\n"
            "%s        'item_overrides': {\n"
            "%s            '%s': {\n"
            "%s                '%s': %s,\n"
            "%s            },\n"
            "%s        },\n"
            "%s    },\n"
            "%s},\n"
        )%( indent,
            indent, time.strftime( '%Y-%m-%d' ), indent, indent,
            indent, variant_name, indent, indent, label,
            indent, key, self._format( value ),
            indent, indent, indent, indent )
        insert_at = source.rfind( '\n', 0, backend_end ) + 1
        return source[:insert_at] + block + source[insert_at:]

    def _insert_into_variants( self, source, variants_end, variant_name, label, key, value ):
        indent = self._indent_of_line( source, variants_end ) + '    '
        block = (
            "%s'%s': {\n"
            "%s    'item_overrides': {\n"
            "%s        '%s': {\n"
            "%s            '%s': %s,  # set by auto_tune %s\n"
            "%s        },\n"
            "%s    },\n"
            "%s},\n"
        )%( indent, variant_name, indent, indent, label,
            indent, key, self._format( value ), time.strftime( '%Y-%m-%d' ),
            indent, indent, indent )
        insert_at = source.rfind( '\n', 0, variants_end ) + 1
        return source[:insert_at] + block + source[insert_at:]

    def _set_within_variant( self, source, variant_at, variant_name, label, key, value ):
        variant_end = self._matching_brace( source, source.index( '{', variant_at ) )
        overrides_at = source.find( "'item_overrides'", variant_at )
        if overrides_at < 0 or overrides_at > variant_end:
            indent = self._indent_of_line( source, variant_at ) + '    '
            block = (
                "%s'item_overrides': {\n"
                "%s    '%s': {\n"
                "%s        '%s': %s,  # set by auto_tune %s\n"
                "%s    },\n"
                "%s},\n"
            )%( indent, indent, label, indent, key, self._format( value ),
                time.strftime( '%Y-%m-%d' ), indent, indent )
            insert_at = source.rfind( '\n', 0, variant_end ) + 1
            return source[:insert_at] + block + source[insert_at:]

        overrides_end = self._matching_brace( source, source.index( '{', overrides_at ) )
        label_at = source.find( "'%s': {"%(label), overrides_at )
        if label_at < 0 or label_at > overrides_end:
            indent = self._indent_of_line( source, overrides_at ) + '    '
            block = (
                "%s'%s': {\n"
                "%s    '%s': %s,  # set by auto_tune %s\n"
                "%s},\n"
            )%( indent, label, indent, key, self._format( value ),
                time.strftime( '%Y-%m-%d' ), indent )
            insert_at = source.rfind( '\n', 0, overrides_end ) + 1
            return source[:insert_at] + block + source[insert_at:]

        label_end = self._matching_brace( source, source.index( '{', label_at ) )
        pattern = re.compile( r"(^\s*'%s'\s*:\s*)([^,\n]+)(,?)"%( re.escape( key ) ), re.M )
        matches = list( pattern.finditer( source, label_at, label_end ) )
        if len( matches ) > 1:
            raise ConfigWriteError(
                "found %d entries for %r inside %s's %r block"%(
                    len( matches ), key, variant_name, label ) )
        if matches:
            match = matches[0]
            return ( source[:match.start()] + match.group(1) + self._format( value )
                     + ( match.group(3) or ',' ) + source[match.end():] )
        return self._insert_item_override( source, label_at, label_end, key, value )

    @staticmethod
    def _indent_of_line( source, position ):
        line_start = source.rfind( '\n', 0, position ) + 1
        line = source[line_start:position]
        return line[ : len( line ) - len( line.lstrip() ) ]

    def _insert_item_override( self, source, start, end, key, value ):
        """
        Add a key as the last entry of a label's override block.

        `end` is the offset of that block's closing brace. The new line
        goes immediately before it, indented to match whatever the block
        already uses, so the result reads as if it had been typed there.

        Indentation is COPIED from an existing sibling rather than
        assumed, because this file is hand-maintained and a tuner that
        reformats it is a tuner people stop trusting with the file. A
        block with no siblings to copy from falls back to the closing
        brace's own indent plus four spaces.
        """
        block = source[start:end]
        sibling = re.search( r"^([ \t]+)'[^']+'\s*:", block, re.M )
        if sibling:
            indent = sibling.group(1)
        else:
            brace_line_start = source.rfind( '\n', 0, end ) + 1
            indent = source[brace_line_start:end] + '    '

        line = "%s'%s': %s,  # set by auto_tune %s\n"%(
            indent, key, self._format( value ), time.strftime( '%Y-%m-%d' ) )

        # Insert before the closing brace's own line, not before the
        # brace character, so a block written as "}" on its own line
        # keeps its shape.
        insert_at = source.rfind( '\n', 0, end ) + 1
        return source[:insert_at] + line + source[insert_at:]

    def set_module_scalar( self, name, value ):
        """Set a top-level `name = value` assignment in betaconfig.py."""
        source = self.read()
        pattern = re.compile( r"^(%s\s*=\s*)([^\n#]+)"%( re.escape( name ) ), re.M )
        matches = list( pattern.finditer( source ) )
        if len( matches ) != 1:
            raise ConfigWriteError(
                "expected exactly one top-level '%s =' in betaconfig.py, found %d"%(
                    name, len( matches ) ) )
        match = matches[0]
        trailing = match.group(2)
        comment = ''
        if '#' in trailing:
            comment = '  ' + trailing[ trailing.index( '#' ): ].strip()
        updated = ( source[:match.start()] + match.group(1) + self._format( value )
                    + comment + source[match.end():] )
        self._write_and_verify(
            updated, lambda: getattr( betaconfig, name, None ), value, name )

    def _locate_label_block( self, source, backend_name, label ):
        """
        The (start, end) offsets of one label's dict inside a backend's
        item_overrides, so a key edit cannot stray into another label.
        """
        backend_at = source.find( "'%s': {"%(backend_name) )
        if backend_at < 0:
            raise ConfigWriteError( "no detector_backend block for %r in betaconfig.py"%(backend_name) )
        overrides_at = source.find( "'item_overrides'", backend_at )
        if overrides_at < 0:
            raise ConfigWriteError(
                "detector_backend[%r] has no 'item_overrides' block"%(backend_name) )
        label_at = source.find( "'%s': {"%(label), overrides_at )
        if label_at < 0:
            raise ConfigWriteError(
                "detector_backend[%r]['item_overrides'] has no %r block"%(backend_name, label) )
        depth = 0
        index = source.index( '{', label_at )
        for position in range( index, len( source ) ):
            if source[position] == '{':
                depth += 1
            elif source[position] == '}':
                depth -= 1
                if depth == 0:
                    return ( label_at, position )
        raise ConfigWriteError( "unbalanced braces after %r in betaconfig.py"%(label) )

    @staticmethod
    def _format( value ):
        if isinstance( value, bool ):
            return 'True' if value else 'False'
        if isinstance( value, float ):
            text = ( '%.6f'%(value) ).rstrip( '0' ).rstrip( '.' )
            return text or '0'
        return repr( value )

    def _invalidate_bytecode( self ):
        """
        Drop any compiled copy of betaconfig before reloading it.

        Python decides a cached .pyc is still valid by comparing the
        source's mtime and SIZE. A config edit that replaces one value
        with another of the same length - 27.0 for 40.5, say - within
        the same second changes neither, so the interpreter serves the
        stale bytecode and the reloaded module still holds the old
        value. That is not a hypothetical: it is what this writer's own
        verification caught on its first live run, and without the
        verification it would have been a tuner that reported writing a
        value the next stage could not see.
        """
        import importlib
        importlib.invalidate_caches()
        cache_path = getattr( betaconfig, '__cached__', None )
        if cache_path and os.path.exists( cache_path ):
            try:
                os.remove( cache_path )
            except OSError:
                pass

    def _write_and_verify( self, updated_source, resolve, expected, description ):
        backup = self._backup()
        with open( self.config_path, 'w', encoding='UTF-8' ) as fout:
            fout.write( updated_source )

        # Reload in THIS process so later stages measure the new value,
        # and check the real resolver agrees. A file that parses but
        # resolves to something else is the failure mode worth catching,
        # and the only way to catch it is to ask the resolver rather
        # than to trust the write.
        import importlib
        self._invalidate_bytecode()
        try:
            importlib.reload( betaconfig )
        except Exception as err:
            shutil.copy2( backup, self.config_path )
            self._invalidate_bytecode()
            importlib.reload( betaconfig )
            bu_config.invalidate_config_caches()
            raise ConfigWriteError(
                "writing %s produced a betaconfig.py that will not import (%s); the original "
                "has been restored from %s"%( description, err, os.path.basename( backup ) ) )

        bu_config.invalidate_config_caches()
        actual = resolve()
        if actual is None or abs( float( actual ) - float( expected ) ) > 1e-9:
            shutil.copy2( backup, self.config_path )
            self._invalidate_bytecode()
            importlib.reload( betaconfig )
            bu_config.invalidate_config_caches()
            raise ConfigWriteError(
                "wrote %s = %r but the resolver reports %r; the original has been restored "
                "from %s"%( description, expected, actual, os.path.basename( backup ) ) )
        if self.logger:
            self.logger.info( "config: %s = %s"%( description, self._format( expected ) ) )


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

TIGHTENING_COVERAGE_TOLERANCE = 0.0001


def tightening_review( rows, baseline_value, coverage_tolerance=TIGHTENING_COVERAGE_TOLERANCE ):
    """
    A tighter value worth judging by eye, from a loosening stage's rows, or None.

    The loosening stages accept only candidates that cut track resets,
    so a tighter value never wins there, however much ambiguity it
    removes: on 2026-10-11, match_distance 0.75 for exposed_breast cut
    risky contention from 13,884 to 4,375 for 6,631 more resets at no
    coverage loss, and was rejected for the resets alone. Whether a
    snap (a reset) or a possible subject swap (risky contention) looks
    worse is for eyes, not this metric, so this only names the candidate.

    Eligible: below baseline_value, coverage loss within
    coverage_tolerance, and at least one fewer risky assignment for each
    reset it adds. Among those, the one removing the most risky ones.

    Returns:
        {'value', 'risky_removed', 'resets_added', 'why'} or None.
    """
    best = None
    for row in rows:
        value = row.get( 'value' )
        if value is None or baseline_value is None or value >= baseline_value:
            continue
        risky_removed = -( row.get( 'risky_added' ) or 0 )
        resets_added = -( row.get( 'resets_avoided' ) or 0 )
        if ( row.get( 'coverage_loss', 1.0 ) > coverage_tolerance or risky_removed <= 0
                or risky_removed < resets_added ):
            continue
        if best is None or risky_removed > best['risky_removed']:
            best = { 'value': value, 'risky_removed': risky_removed, 'resets_added': resets_added }
    if best:
        best['why'] = ( "%s removes %d ambiguous track decisions for %d more resets at no coverage "
                        "loss; whether that looks better is a judgement by eye"%(
                            best['value'], best['risky_removed'], best['resets_added'] ) )
    return best


class Decision:
    """
    One stage's outcome, with the reasoning that produced it.

    The reasoning is not decoration. A tuner that writes a value without
    saying why is indistinguishable from a tuner with a bug, and the
    only way to find out which you have is to be able to read the
    argument back.
    """

    def __init__( self, stage_name, chosen, accepted, reasoning, rows,
                  per_configuration=None, review=None, key=None, label=None ):
        self.stage_name = stage_name
        self.chosen = chosen
        self.accepted = accepted
        self.reasoning = reasoning
        self.rows = rows
        # A value the numbers cannot settle, for ab_compare.py to put in
        # front of a person: {'value', 'why', ...} or None.
        self.review = review
        self.key = key
        self.label = label
        # When the configurations sharing a key want DIFFERENT values,
        # this holds {configuration label: value} and `chosen` is None.
        # The write path then targets each variant's own block instead
        # of the shared one.
        self.per_configuration = per_configuration or {}

    @property
    def is_split( self ):
        return bool( self.per_configuration )

    def as_dict( self ):
        return {
            'stage': self.stage_name,
            'chosen': self.chosen,
            'per_configuration': self.per_configuration,
            'accepted': self.accepted,
            'reasoning': self.reasoning,
            'review': self.review,
            'key': self.key,
            'label': self.label,
            'rows': self.rows,
        }
