"""
betautils_cache_paths.py - THE single source of truth for every cache
path, output filename, and cache key in BetaSuite.

Nothing else in this codebase may construct one of these names. Not
betatv.py, not betastare.py, not a tool under tools/, not a test. They
all call in here.

WHY THIS RULE EXISTS
--------------------
Path formulas were once re-implemented independently in up to six
standalone tools. They drifted, and the drift caused real bugs twice:
once when the detector backend name was missing from one copy of a glob
(corrupting the file_hash it parsed back out, so every video looked
"not found"), and once when a video shorter than preview_start_seconds
fell back to a plain '-preview' suffix that two tools' hand-copied
formulas did not model. Both were invisible until someone noticed a
result that made no sense.

tests/test_cache_path_duplicates_stay_in_sync.py is the regression guard.

CACHE KEY MODEL
------------------------
Output is produced by three independent stages. Each has its own key, so
changing one stage's settings invalidates exactly that stage and no
more. All three appear in the final output filename.

  detection key  "d<6 hex>"   what raw detections come out of the model
      backend, model variant/path, detection size, sample fps,
      global_min_prob, and the backend's own detection-affecting
      tunables (nudenet_v3: candidate_floor / nms_iou / nms_mode).
      Deliberately NOT nn_batch_size: batching changes how many frames
      per inference call, never the detections themselves. That claim is
      enforced by tests/test_batch_size_invariance.py rather than
      asserted in a comment.

  censor key     "c<6 hex>"   what those detections turn into on screen
      parts_to_blur (styles, per-label min_prob and safety margins),
      every tracking/smoothing/suppression tunable resolved for the
      active backend, the censor-overlap and censor-scale strategies,
      the geometry sanity filter, cross-size dedup, and the debug
      overlay flag.

  encode key     "e<4 hex>"   how the finished frames are compressed
      codec, CRF, preset, container.

Previously the output filename embedded only a narrow "censor hash"
that deliberately excluded tracking settings, so re-running with a
changed track_max_gap silently overwrote the previous output. That was a
documented footgun; it is now simply fixed. Every setting that changes
the bytes of the output is in one of the three keys.

Each key's full expansion is written to
../output/cache/run_keys/<kind>-<key>.json the first time it is used, so
a filename can always be decoded back into the config that produced it.
"""

import glob
import os
import statistics

import betaconst
import betaconfig
import betautils_hash as bu_hash
import betautils_detector as bu_detector


# Legacy aliases, kept because tools import them by name.
VID_HASH_DIR = betaconst.vid_hash_dir
PIC_HASH_DIR = betaconst.pic_hash_dir


# ---------------------------------------------------------------------------
# Preview slice suffix
# ---------------------------------------------------------------------------

def preview_cache_suffix( preview_mode_enabled, preview_offset_seconds ):
    """
    Build the '' / '-preview' / '-preview@<offset>s' filename tail.

    An offset of exactly 0.0 lands on the plain '-preview' form, not
    '-preview@0.0s'. That covers both "no explicit slice requested" and
    the whole-file fallback (a preview run of a file shorter than
    preview_start_seconds), which are the same thing on disk: not a
    slice of anything, so no '@<offset>s' implying one.

    Args:
        preview_mode_enabled: Whether preview mode is on at all.
        preview_offset_seconds: The resolved slice offset in seconds.

    Returns:
        The suffix string.
    """
    if not preview_mode_enabled:
        return ''
    if preview_offset_seconds:
        return '-preview@%.1fs'%(preview_offset_seconds)
    return '-preview'


# ---------------------------------------------------------------------------
# Run-key manifests
# ---------------------------------------------------------------------------

_recorded_run_keys = set()


def record_run_key( kind, key, identity ):
    """
    Write a key's full expansion to ../output/cache/run_keys/ so an
    output filename can be decoded months later.

    Best-effort and idempotent per process: a failure here never affects
    a run. Costs one small file write the first time each key is seen.

    Args:
        kind: 'detection' | 'censor' | 'encode'.
        key: The short hex key as it appears in filenames.
        identity: The dict the key was hashed from.
    """
    path = os.path.join( betaconst.run_key_dir, '%s-%s.json'%(kind, key) )
    # Deduplicated on the resolved PATH, not on kind+key: the cache root
    # can be redirected (a test harness, a tool pointed at another tree),
    # and a key recorded into one directory says nothing about another.
    if path in _recorded_run_keys:
        return
    _recorded_run_keys.add( path )
    if os.path.exists( path ):
        return
    try:
        bu_hash.write_json_plain( { 'kind': kind, 'key': key, 'identity': identity }, path )
    except Exception:
        pass


def _keyed( kind, identity, length ):
    """Hash an identity dict, record its expansion, and return the key."""
    key = bu_hash.dictionary_hash( identity, length )
    record_run_key( kind, key, identity )
    return key


# ---------------------------------------------------------------------------
# Detection identity / key
# ---------------------------------------------------------------------------

def detection_identity( backend_name=None, size=None, fps=None, min_prob=None ):
    """
    Everything that determines which raw detections come out of the model.

    Args:
        backend_name: Backend to resolve for. Defaults to the selected one.
        size: Detection size. Defaults to the backend's first configured
            picture size.
        fps: Sample rate. Defaults to betaconfig.video_censor_fps.
        min_prob: Global confidence floor. Defaults to the configured one.

    Returns:
        A JSON-serialisable dict.
    """
    backend_name = backend_name or bu_detector.selected_backend_name()
    if size is None:
        sizes = bu_detector.get_picture_sizes( backend_name )
        size = sizes[0] if sizes else 0
    if fps is None:
        fps = getattr( betaconfig, 'video_censor_fps', None )
    if min_prob is None:
        min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    return {
        'box_version':     betaconst.picture_saved_box_version,
        'backend':         backend_name,
        'size':            size,
        'sample_fps':      fps,
        'global_min_prob': round( float( min_prob ), 6 ),
        'backend_tunables': bu_detector.get_detection_identity( backend_name ),
    }


def detection_key( backend_name=None, size=None, fps=None, min_prob=None ):
    """Short key for detection_identity(). See this module's docstring."""
    identity = detection_identity( backend_name, size, fps, min_prob )
    return _keyed( 'detection', identity, betaconst.detection_key_len )


# ---------------------------------------------------------------------------
# Censor identity / key
# ---------------------------------------------------------------------------

def _profiles_identity( backend_name ):
    """
    The structure-profile block for one backend, or None when it has none.

    Returned as-is rather than summarised. The values are already plain
    JSON types, and a summary would have to be kept in step with every
    future profile key by hand - the exact maintenance burden that let
    profiles go missing from the censor key in the first place.

    Args:
        backend_name: Backend to read the profiles block from.

    Returns:
        The profiles dict, or None when the backend defines none.
    """
    backend = getattr( betaconfig, 'detector_backend', {} ).get( backend_name, {} )
    return backend.get( 'profiles' ) or None


def censor_identity( backend_name=None ):
    """
    Everything that changes rendered output for a FIXED set of detections.

    Resolved against a specific backend AND variant, because per-label
    tracking settings (track_max_gap, min_prob, interpolation,
    hysteresis, ...) resolve through both tiers. The variant is named
    explicitly in the identity rather than left implicit in the resolved
    values: two variants that happen to agree on every setting today
    should still not share a censor key, or a later change to one would
    silently invalidate the other's rendered output too.

    Returns:
        A JSON-serialisable dict.
    """
    import betautils_config as bu_config  # deferred: bu_config imports bu_detector

    backend_name = backend_name or bu_detector.selected_backend_name()
    variant_name = bu_detector.selected_variant_name( backend_name )

    per_label = {}
    for label in sorted( bu_config.get_parts_to_blur( backend_name ).keys() ):
        per_label[label] = bu_config.get_label_settings( label, backend_name )

    return {
        'labels':                   per_label,
        'variant':                  variant_name,
        'class_suppression':        bu_detector.get_class_suppression( backend_name ),
        'class_promotion':          bu_detector.get_class_promotion( backend_name ),
        'censor_overlap_strategy':  getattr( betaconfig, 'censor_overlap_strategy', {} ),
        'censor_scale_strategy':    getattr( betaconfig, 'censor_scale_strategy', 'feature' ),
        'type_default_shapes':      getattr( betaconfig, 'type_default_shapes', {} ),
        'default_censor_shape':     getattr( betaconfig, 'default_censor_shape', 'box' ),
        'default_censor_style':     getattr( betaconfig, 'default_censor_style', {} ),
        'blur_fast_approximation':  getattr( betaconfig, 'blur_fast_approximation', True ),
        'blur_approximation_min_kernel': getattr( betaconfig, 'blur_approximation_min_kernel', 21 ),
        'blur_edge_margin':         bool( getattr( betaconfig, 'blur_edge_margin', True ) ),
        # Motion interpolation changes the rendered rectangle on most
        # frames, so it belongs here rather than in the encode key.
        'render_motion_interpolation': bool( getattr(
            betaconfig, 'render_motion_interpolation', True ) ),
        'render_motion_max_span_seconds': getattr(
            betaconfig, 'render_motion_max_span_seconds', None ),
        'render_motion_collapse_max_growth': getattr(
            betaconfig, 'render_motion_collapse_max_growth', 1.15 ),
        'render_motion_size_window_seconds': getattr(
            betaconfig, 'render_motion_size_window_seconds', 0.25 ),
        'cross_size_dedup':         getattr( betaconfig, 'cross_size_dedup', {} ),

        # The global default_* fallbacks, keyed DIRECTLY rather than only
        # through per_label above.
        #
        # per_label captures each censored label's RESOLVED value, which
        # covers a default only while some censored label still falls
        # through to it. Once every censored label sets the setting
        # explicitly - which a tuned config naturally ends up doing - the
        # default is shadowed everywhere and changing it moves nothing in
        # the key, so a run with a new default silently reuses the old
        # render and reports it as the new result. Caught by
        # tests/test_censor_key_coverage.py on a real tuned config, where
        # all three censored labels pinned position_smoothing.
        'global_defaults': {
            name: getattr( betaconfig, name )
            for name in sorted( dir( betaconfig ) )
            if name.startswith( 'default_' ) and not callable( getattr( betaconfig, name ) )
        },
        'watermark':                bool( getattr( betaconfig, 'enable_betasuite_watermark', False ) ),
        'debug_overlay':            bool( getattr( betaconfig, 'debug_mode', 0 ) & 1 ),

        # Shot cuts feed tracking, not detection, so they change rendered
        # output for a fixed set of detections and belong here rather
        # than in the detection key. They have their own separate cache
        # key for the scan results, which is why their absence here went
        # unnoticed: lowering shot_cut_threshold correctly rescanned and
        # found more cuts, then reused a rendered file produced from the
        # old ones.
        'shot_cut_detection_enabled': bool( getattr( betaconfig, 'shot_cut_detection_enabled', True ) ),
        'shot_cut_threshold':         getattr( betaconfig, 'shot_cut_threshold', 0.5 ),

        # Structure profiles override the per-label timing values above
        # AFTER they are resolved, so the 'labels' entry does not reflect
        # them. The whole profiles block is included rather than the
        # profile this run selects, because which one is selected depends
        # on the video and the key must describe the configuration, not
        # one file's outcome.
        'profiles':                  _profiles_identity( backend_name ),
        'default_style_min_dwell_seconds': getattr(
            betaconfig, 'default_style_min_dwell_seconds', 0.0 ),
        'default_profile_enabled':   bool( getattr( betaconfig, 'default_profile_enabled', True ) ),
        # Which profile is FORCED, if any. The whole profiles block above
        # covers what each profile does; this covers which one applies,
        # and --profile changes that without touching the block.
        'force_structure_profile':   getattr( betaconfig, 'force_structure_profile', None ),
        'profile_by_path_pattern':   getattr( betaconfig, 'profile_by_path_pattern', None ),
    }


def censor_key( backend_name=None ):
    """Short key for censor_identity(). See this module's docstring."""
    identity = censor_identity( backend_name )
    return _keyed( 'censor', identity, betaconst.censor_key_len )


def get_censor_hash():
    """
    Backwards-compatible alias for censor_key().

    Pre-2.1 this lived in betautils_hash and covered a deliberately
    narrow slice of settings. It now covers everything that changes
    rendered output; the name is kept so older tooling keeps resolving.
    """
    return censor_key()


# ---------------------------------------------------------------------------
# Encode identity / key
# ---------------------------------------------------------------------------

def encode_identity( preview_mode_enabled=None ):
    """
    Everything that changes the compression of the finished video.

    Separate from the censor key on purpose: re-encoding at a different
    preset produces different bytes but identical censoring, so it gets
    its own short key rather than invalidating detection or render work.
    """
    if preview_mode_enabled is None:
        preview_mode_enabled = bool( getattr( betaconfig, 'preview_mode_enabled', False ) )
    preset = ( getattr( betaconfig, 'preview_encode_preset', 'ultrafast' )
               if preview_mode_enabled
               else getattr( betaconfig, 'encode_preset', 'fast' ) )
    return {
        'codec':     getattr( betaconfig, 'encode_video_codec', 'libx264' ),
        'crf':       getattr( betaconfig, 'encode_crf', 17 ),
        'preset':    preset,
        'container': getattr( betaconfig, 'render_chunk_container', 'mkv' ),
    }


def encode_key( preview_mode_enabled=None ):
    """Short key for encode_identity()."""
    identity = encode_identity( preview_mode_enabled )
    return _keyed( 'encode', identity, betaconst.encode_key_len )


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def box_hash_path_for( file_hash, size, fps, min_prob, backend_name,
                       preview_suffix='', det_key=None ):
    """
    Detection cache path for one (file, size, fps, backend, settings).

    backend_name is a required positional arg and is never silently read
    from the current config, so a caller can always pin exactly which
    backend's cache it means - that was the bug in one of the old
    duplicate copies.

    size / fps / min_prob stay spelled out in the filename even though
    det_key also covers them, because humans and glob-based discovery
    both read them directly.

    Args:
        file_hash: Content hash of the source video.
        size: A picture_sizes entry.
        fps: betaconfig.video_censor_fps for the run that produced it.
        min_prob: global_min_prob for that run.
        backend_name: Detector backend that produced it.
        preview_suffix: From preview_cache_suffix(); '' for a real run.
        det_key: Detection key. None resolves it from the CURRENT config
            for backend_name, which is what every in-run caller wants.

    Returns:
        The cache path.
    """
    if det_key is None:
        det_key = detection_key( backend_name, size, fps, min_prob )
    return os.path.join( VID_HASH_DIR, '%s-%s-%s-%d-%g-%.3f-d%s%s.gz'%(
        file_hash, backend_name, betaconst.picture_saved_box_version,
        size, fps, min_prob, det_key, preview_suffix ) )


def pic_hash_path_for( image_hash, size, min_prob, backend_name, det_key=None ):
    """
    Still-image sibling of box_hash_path_for.

    No fps and no preview suffix: a still has neither a frame rate nor a
    slice.
    """
    if det_key is None:
        det_key = detection_key( backend_name, size, None, min_prob )
    return os.path.join( PIC_HASH_DIR, '%s-%s-%s-%d-%.3f-d%s.gz'%(
        image_hash, backend_name, betaconst.picture_saved_box_version,
        size, min_prob, det_key ) )


def shot_cut_path_for( file_hash, fps, threshold, preview_suffix='',
                       preview_max_seconds=None ):
    """
    Shot-cut cache path.

    Shot cuts depend on the sampling cadence and the Bhattacharyya
    threshold only, never on picture_sizes / global_min_prob / backend,
    so this is its own cache rather than folded into the detection one.

    preview_max_seconds is part of the key because a bounded preview
    scan genuinely sees less of the file than an unbounded one and must
    not be served back to a longer scan.
    """
    bound = '' if not preview_max_seconds else '-max%gs'%(preview_max_seconds)
    return os.path.join( betaconst.shot_cut_dir, '%s-%g-%.3f%s%s.gz'%(
        file_hash, fps, threshold, bound, preview_suffix ) )


def transcode_path_for( file_hash ):
    """
    Cache path for a system-ffmpeg transcode fallback (see betatv.py).

    Kept rather than deleted after use, same "disk is cheaper than
    compute" principle as the detection caches. There is no automatic
    eviction; clear the directory by hand if it grows.
    """
    os.makedirs( betaconst.transcode_cache_dir, exist_ok=True )
    return os.path.join( betaconst.transcode_cache_dir, '%s.mp4'%(file_hash) )


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def video_output_basename( stem, file_hash, sizes, fps, det_key, cen_key,
                           enc_key, preview_suffix='', output_label='' ):
    """
    Build the identifying stem of a censored video's filename.

    Format:
        <stem>-<file_hash>-<sizes>-<fps>-d<det>-c<cen>-e<enc><preview><label>

    Reading one of these tells you: which source file (hash), at what
    detection sizes and sample rate, under which detection settings
    (d), which censor/tracking settings (c), and which encode settings
    (e). Every one of those is a genuine input to the bytes on disk, so
    two runs that differ in any of them can never collide.

    output_label is a human-readable tail for a sweep that renders one
    source many times, e.g. '-gaussian-s120'. It is COSMETIC: the keys
    above already guarantee uniqueness, and the label exists only so a
    directory of sweep output can be read without decoding hex. It is
    appended to the OUTPUT name only and never reaches a cache path, so
    two labelled runs that differ in nothing else still share their
    cached detections.

    Labels are sanitised rather than trusted, because they end up in a
    filename: anything outside [A-Za-z0-9._-] becomes '-'.
    """
    sizes_text = "+".join( str( size ) for size in sizes )
    base = '%s-%s-%s-%g-d%s-c%s-e%s%s'%(
        stem, file_hash, sizes_text, fps, det_key, cen_key, enc_key, preview_suffix )
    return base + sanitise_output_label( output_label )


def sanitise_output_label( label ):
    """
    A filename-safe form of a cosmetic output label, or '' for none.

    Kept next to the basename builder rather than in the caller so every
    entry point that accepts a label agrees on what one may contain.
    """
    if not label:
        return ''
    cleaned = ''.join( character if ( character.isalnum() or character in '._-' )
                       else '-' for character in str( label ) )
    cleaned = cleaned.strip( '-' )
    if not cleaned:
        return ''
    return '-' + cleaned


def video_output_paths( censored_folder, stem, file_hash, sizes, fps,
                        det_key, cen_key, enc_key, preview_suffix='',
                        chunk_container='mkv', output_label='' ):
    """
    Final and intermediate paths for one censored video.

    Returns:
        A (final_mp4, intermediate_video) pair. The intermediate holds
        the already-censored, already-H.264 video with no audio; the
        final adds the source audio by stream copy. Both live in
        censored_folder.
    """
    base = video_output_basename( stem, file_hash, sizes, fps, det_key,
                                  cen_key, enc_key, preview_suffix, output_label )
    final_path = os.path.join( censored_folder, base + '.mp4' )
    intermediate_path = os.path.join( censored_folder, base + '.video.' + chunk_container )
    return final_path, intermediate_path


def picture_output_path( censored_folder, stem, suffix, image_hash, sizes,
                         det_key, cen_key ):
    """
    Output path for one censored still image (betastare.py).

    No encode key: a still is written by cv2.imwrite with no configurable
    codec settings.
    """
    sizes_text = "+".join( str( size ) for size in sizes )
    return os.path.join( censored_folder, '%s-%s-%s-d%s-c%s%s'%(
        stem, image_hash, sizes_text, det_key, cen_key, suffix ) )


def chunk_path_for( intermediate_path, chunk_index ):
    """Path for render chunk `chunk_index` of intermediate_path."""
    base, ext = os.path.splitext( intermediate_path )
    return '%s.part%04d%s'%(base, chunk_index, ext)


def temp_sibling( path ):
    """
    A '.tmp'-marked sibling of path that keeps path's extension.

    The marker goes BEFORE the extension, not after: ffmpeg picks its
    muxer from the extension, and a name ending '.mkv.tmp' makes it fail
    to pick one at all.
    """
    base, ext = os.path.splitext( path )
    return '%s.tmp%s'%(base, ext)


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------

def cache_name_stem( backend_name, size, fps, min_prob ):
    """
    The fixed middle of a detection cache filename, between the source
    file_hash and the detection key.

    Every discovery glob and every parse below is built from this one
    string, so a change to the filename format cannot leave a matcher
    behind. One of the real bugs this module exists to prevent was
    exactly that: a glob that had not been updated when the backend name
    joined the name.

    Returns:
        A string beginning and ending with '-'.
    """
    return '-%s-%s-%d-%g-%.3f-d'%(
        backend_name, betaconst.picture_saved_box_version, size, fps, min_prob )


def parse_cache_name( filename, backend_name, size, fps, min_prob ):
    """
    Pull a detection cache filename apart.

    Args:
        filename: A basename, with or without directories stripped.
        backend_name, size, fps, min_prob: What it should belong to.

    Returns:
        A (file_hash, detection_key, preview_suffix) tuple, or None when
        the name does not belong to this (backend, size, fps, min_prob).

    Note:
        The stem is located rather than merely tested for containment,
        so a file_hash that happens to contain the stem as a substring
        cannot cause a misparse.
    """
    name = os.path.basename( filename )
    if not name.endswith( '.gz' ):
        return None
    stem = cache_name_stem( backend_name, size, fps, min_prob )
    index = name.find( stem )
    if index <= 0:
        return None
    file_hash = name[:index]
    remainder = name[ index + len( stem ) : -len( '.gz' ) ]
    if not remainder:
        return None
    if '-' in remainder:
        detection_key, preview_suffix = remainder.split( '-', 1 )
        preview_suffix = '-' + preview_suffix
        if not preview_suffix.startswith( '-preview' ):
            return None
    else:
        detection_key, preview_suffix = remainder, ''
    if not detection_key:
        return None
    return file_hash, detection_key, preview_suffix


def _cache_glob( backend_name, size, fps, min_prob ):
    """Glob matching every detection cache for these parameters, any key."""
    return os.path.join( VID_HASH_DIR,
                         '*' + cache_name_stem( backend_name, size, fps, min_prob ) + '*.gz' )


def _file_hash_from_cache_path( path ):
    """
    Recover the source file_hash from a detection cache filename.

    Everything before the first '-'. Only unambiguous because the
    backend name is in the name; matching a looser pattern can return a
    filename whose leading segment is not a hash at all, which is
    precisely the bug this module was created to fix.
    """
    return os.path.basename( path ).split( '-', 1 )[0]


class CachedConfiguration:
    """
    One model setup that actually produced detections on this disk.

    A "configuration" is the unit every analysis tool should reason
    about: one backend, at one detection size, under one detection key.
    Two variants of the same backend (nudenet_v3 at 320n and at 640m)
    are two configurations and must never be pooled, because their
    detections come from different weights.

    Attributes:
        backend_name: The backend that wrote these caches.
        size: The detection size.
        detection_key: The short key covering every setting that
            changes raw detections, variant included.
        variant: The model variant name when the run-key manifest for
            this key is still on disk, otherwise None. Filenames carry
            the size but not the variant, so this is recovered rather
            than parsed.
        identity: The full detection identity dict from that manifest,
            or {} when it could not be read.
        preview_suffix: '' for a real run, or the preview suffix these
            caches were written under.
        file_hashes: The source videos covered.
    """

    __slots__ = ( 'backend_name', 'picture_sizes', 'detection_key', 'variant',
                  'identity', 'preview_suffix', 'file_hashes' )

    def __init__( self, backend_name, picture_sizes, detection_key=None, variant=None,
                  identity=None, preview_suffix='', file_hashes=() ):
        self.backend_name = backend_name
        self.picture_sizes = list( picture_sizes )
        self.detection_key = detection_key
        self.variant = variant
        self.identity = identity or {}
        self.preview_suffix = preview_suffix
        self.file_hashes = sorted( file_hashes )

    @property
    def size( self ):
        """The first (usually only) detection size."""
        return self.picture_sizes[0] if self.picture_sizes else 0

    @property
    def label( self ):
        """
        A short human label, e.g. 'nudenet_v3/640m @640'.

        The variant is included when known because "nudenet_v3" alone is
        ambiguous the moment a second variant has been run, and an
        ambiguous label on a comparison table is how two models' numbers
        get read as one model's.
        """
        parts = self.backend_name
        if self.variant:
            parts += '/' + self.variant
        parts += ' @%s'%( self.picture_sizes[0] if len( self.picture_sizes ) == 1
                          else '+'.join( str(s) for s in self.picture_sizes ) )
        if self.preview_suffix:
            parts += ' [PREVIEW]'
        return parts

    @property
    def sort_key( self ):
        return ( self.backend_name, self.picture_sizes, self.variant or '', self.preview_suffix )

    def __repr__( self ):
        return '<CachedConfiguration %s key=%s videos=%d>'%(
            self.label, self.detection_key, len( self.file_hashes ) )


def _identity_for_detection_key( detection_key ):
    """
    The recorded identity behind a detection key, or {}.

    betautils_cache_paths writes one of these per key the first time it
    is used (see record_run_key), precisely so a filename can be decoded
    later. Reading it back is how a tool learns which model VARIANT
    produced a cache, since the filename carries only the size.
    """
    path = os.path.join( betaconst.run_key_dir, 'detection-%s.json'%(detection_key) )
    try:
        payload = bu_hash.read_json_plain( path )
    except Exception:
        return {}
    if not isinstance( payload, dict ):
        return {}
    identity = payload.get( 'identity' )
    return identity if isinstance( identity, dict ) else {}


def _variant_from_identity( identity ):
    """Pull the model variant out of a detection identity, if it names one."""
    tunables = identity.get( 'backend_tunables' )
    if not isinstance( tunables, dict ):
        return None
    for key in ( 'model_variant', 'variant' ):
        value = tunables.get( key )
        if isinstance( value, str ) and value:
            return value
    return None


def discover_cached_configurations( fps=None, min_prob=None, include_preview=False,
                                    backend_names=None ):
    """
    Every model configuration that has actually written detections here.

    This is the answer to "which models and variants should I analyse?"
    and it is deliberately derived from DISK rather than from config.
    Asking betaconfig gives you the one configuration that happens to be
    selected right now, which is how an overnight run of three
    configurations got analysed as one: the 640m caches were sitting on
    disk, complete, and every tool looked past them because config said
    320n. Disk cannot forget a run that happened.

    Analogy: config is today's shopping list; the cache directory is the
    pantry. To find out what you have, look in the pantry.

    Grouping is by (backend, size, detection_key). Two entries with the
    same backend and different sizes are different variants or different
    settings, and are never merged. A backend genuinely configured with
    several sizes at once is the one case this splits where a caller may
    want them combined - pass explicit sizes to discover_full_run_videos
    for that, which every tool still exposes as --picture-sizes.

    Args:
        fps: Sample rate to match. Defaults to betaconfig.video_censor_fps.
        min_prob: Confidence floor to match. Defaults to the configured one.
        include_preview: Include preview-slice caches as their own
            configurations. Off by default: preview totals describe a
            few seconds and do not belong beside a real run's.
        backend_names: Restrict to these backends. Defaults to every
            registered one.

    Returns:
        A list of CachedConfiguration, sorted by backend then size.
    """
    if fps is None:
        fps = getattr( betaconfig, 'video_censor_fps', None )
    if min_prob is None:
        min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    names = backend_names or bu_detector.registered_backend_names()

    # {(backend, size, key, preview): set(file_hash)}
    found = {}
    for path in glob.glob( os.path.join( VID_HASH_DIR, '*.gz' ) ):
        name = os.path.basename( path )
        for backend_name in names:
            size = _size_in_cache_name( name, backend_name, fps, min_prob )
            if size is None:
                continue
            parsed = parse_cache_name( name, backend_name, size, fps, min_prob )
            if not parsed:
                continue
            file_hash, detection_key, preview_suffix = parsed
            if preview_suffix and not include_preview:
                continue
            bucket = ( backend_name, size, detection_key, preview_suffix )
            found.setdefault( bucket, set() ).add( file_hash )
            break

    configurations = []
    for ( backend_name, size, detection_key, preview_suffix ), hashes in found.items():
        identity = _identity_for_detection_key( detection_key )
        configurations.append( CachedConfiguration(
            backend_name=backend_name,
            picture_sizes=[ size ],
            detection_key=detection_key,
            variant=_variant_from_identity( identity ),
            identity=identity,
            preview_suffix=preview_suffix,
            file_hashes=hashes ) )

    configurations.sort( key=lambda config: config.sort_key )
    return configurations


def _size_in_cache_name( name, backend_name, fps, min_prob ):
    """
    Recover the detection size from a cache filename, or None.

    The size is the one field of the stem a discovery pass does not know
    in advance, so it is read back out rather than guessed at from
    config. Built from cache_name_stem's own format so a change to the
    filename layout cannot leave this behind.
    """
    prefix = '-%s-%s-'%( backend_name, betaconst.picture_saved_box_version )
    index = name.find( prefix )
    if index <= 0:
        return None
    remainder = name[ index + len( prefix ) : ]
    size_text = remainder.split( '-', 1 )[0]
    if not size_text.isdigit():
        return None
    size = int( size_text )
    # Confirm by rebuilding the full stem: this rejects a name whose fps
    # or min_prob belong to a different run, which a prefix match alone
    # would happily accept.
    if cache_name_stem( backend_name, size, fps, min_prob ) not in name:
        return None
    return size


def parse_any_cache_name( name ):
    """
    Pull a detection cache filename apart without knowing its run parameters.

    parse_cache_name is the strict form: you tell it which (backend,
    size, fps, min_prob) the file should belong to and it confirms.
    This is the loose form, for a caller walking a whole cache directory
    that does not know in advance what it will find - which is exactly
    the position every "report on everything on disk" tool is in.

    Args:
        name: A cache filename, with or without directories.

    Returns:
        A dict with file_hash, backend, box_version, size, fps,
        min_prob, detection_key and preview_suffix, or None when the
        name is not a detection cache this codebase wrote.
    """
    base = os.path.basename( name )
    if not base.endswith( '.gz' ):
        return None
    parts = base[ : -len( '.gz' ) ].split( '-' )
    if len( parts ) < 7:
        return None
    file_hash, backend, box_version, size, fps, min_prob, detection_key = parts[:7]
    if not size.isdigit() or not box_version.isdigit():
        return None
    if not detection_key.startswith( 'd' ) or len( detection_key ) < 2:
        return None
    try:
        fps_value = float( fps )
        min_prob_value = float( min_prob )
    except ValueError:
        return None
    preview_suffix = '-' + '-'.join( parts[7:] ) if len( parts ) > 7 else ''
    if preview_suffix and not preview_suffix.startswith( '-preview' ):
        return None
    return {
        'file_hash': file_hash,
        'backend': backend,
        'box_version': int( box_version ),
        'size': int( size ),
        'fps': fps_value,
        'min_prob': min_prob_value,
        'detection_key': detection_key[1:],
        'preview_suffix': preview_suffix,
    }


def configuration_label_of_cache_filename( name ):
    """
    Which model configuration wrote this cache file, as a short label.

    Why this exists, concretely: analyze_suppression_pairs and
    analyze_score_distribution grouped their reports by
    backend_of_cache_filename alone. The night nudenet_v3 was run at
    both 320n and 640m, that grouping put fourteen cache files from two
    different sets of model weights into one section headed
    "nudenet_v3", and every number under that heading described a blend
    belonging to neither variant. The size and the detection key are
    right there in the filename; ignoring them was the bug.

    Returns:
        A label like 'nudenet_v3/640m @640', or 'unknown/legacy' for a
        file written before this naming existed.
    """
    parsed = parse_any_cache_name( name )
    if not parsed or parsed['backend'] not in bu_detector.registered_backend_names():
        return 'unknown/legacy'
    identity = _identity_for_detection_key( parsed['detection_key'] )
    return CachedConfiguration(
        backend_name=parsed['backend'],
        picture_sizes=[ parsed['size'] ],
        detection_key=parsed['detection_key'],
        variant=_variant_from_identity( identity ),
        identity=identity,
        preview_suffix=parsed['preview_suffix'] ).label


def backend_of_cache_filename( name ):
    """
    Which backend produced a cache file, read from its own filename.

    The backend name is the second '-'-delimited field of every cache
    filename this module writes. It is matched against the registered
    backend list rather than trusted by position, so the answer stays
    correct if the filename format grows a field, and so some other
    field that happens to look like a name cannot be mistaken for one.

    Args:
        name: A cache file's basename.

    Returns:
        The backend name, or 'unknown/legacy' for a file written before
        backend tagging existed. Returning a sentinel rather than
        guessing matters: silently attributing an old cache to
        whichever backend happens to be selected is how a comparison
        ends up reporting one model's detections under another's name.
    """
    parts = os.path.basename( name ).split( '-' )
    if len( parts ) >= 2 and parts[1] in bu_detector.registered_backend_names():
        return parts[1]
    return 'unknown/legacy'


def is_preview_cache_filename( name ):
    """True for a cache written by a preview-slice run."""
    return '-preview' in os.path.basename( name )


def iter_cache_files( directory, include_preview=False ):
    """
    Yield (filename, raw_boxes) for every readable cache in a directory.

    Size- and backend-agnostic on purpose: the callers that use this
    (score distribution, suppression pairs) want everything on disk and
    group it themselves by backend_of_cache_filename. An unreadable or
    empty file is skipped rather than raised on, because one truncated
    cache should not cost you the report on the other two hundred.

    Args:
        directory: A cache directory (VID_HASH_DIR / PIC_HASH_DIR).
        include_preview: Whether to include preview-slice caches.

    Yields:
        (basename, list_of_raw_box_dicts) pairs, filename-sorted.
    """
    if not os.path.isdir( directory ):
        return
    for path in sorted( glob.glob( os.path.join( directory, '*.gz' ) ) ):
        name = os.path.basename( path )
        if not include_preview and is_preview_cache_filename( name ):
            continue
        try:
            boxes = bu_hash.read_json( path )
        except Exception:
            continue
        if boxes:
            yield ( name, boxes )


def configurations_to_analyse( requested_sizes=None, backend_names=None, fps=None,
                               min_prob=None, include_preview=False, logger=None,
                               variant_names=None ):
    """
    The configurations a tool should report on, in one place.

    Every analysis and tuning tool calls this instead of deciding for
    itself, so "which models and variants did this cover?" has exactly
    one answer across the suite and a new variant is picked up without
    anyone remembering to pass a flag.

    Three modes, in precedence order:

      1. Explicit sizes given (--picture-sizes): one configuration per
         requested backend at exactly those sizes. This is the escape
         hatch for inspecting a configuration that is not on disk in the
         expected shape, and for a backend deliberately run at several
         sizes at once.
      2. Caches found on disk: one configuration per (backend, size,
         detection key), labelled with the model variant recovered from
         its run-key manifest. This is the default and the reason a
         640m run cannot go unanalysed just because config now says
         320n.
      3. Nothing on disk: fall back to each requested backend's
         currently configured sizes, so a tool run before any detections
         exist still prints a sensible "no caches for X" rather than
         nothing at all.

    Args:
        requested_sizes: Explicit sizes, or None/empty to discover.
        backend_names: Restrict to these backends. Defaults to all registered.
        fps, min_prob: Run parameters to match. Default to config.
        include_preview: Whether preview-slice caches count.
        logger: Optional logger for a one-line summary of what was found.
        variant_names: Restrict to these model variants (e.g. ['640m']).
            Defaults to every variant found.

            Backend alone cannot express "just 640m": 320n and 640m are
            both nudenet_v3, so --backends nudenet_v3 still analyses
            both. Once a backend and variant are settled, re-analysing
            the ones you are no longer tuning costs real time - the
            retinanet replays alone were ~11 minutes of a 15-minute
            auto_tune run - and it pads every report with columns you
            have to read past. This is the narrowing knob; it changes
            nothing about what is on disk, so widening it again later
            needs no re-run.

    Returns:
        A list of CachedConfiguration.
    """
    names = list( backend_names or bu_detector.registered_backend_names() )
    wanted_variants = { str( v ) for v in variant_names } if variant_names else None

    def _keep( configuration ):
        if wanted_variants is None:
            return True
        # A backend with no variant concept (retinanet_v2) has
        # variant None. Naming variants is an explicit request for
        # specific ones, so an unnamed backend is filtered out rather
        # than silently kept.
        return str( configuration.variant ) in wanted_variants

    if requested_sizes:
        configurations = [
            CachedConfiguration( backend_name=name, picture_sizes=list( requested_sizes ) )
            for name in sorted( names ) ]
        # Explicit sizes name a shape rather than a cache on disk, so
        # there is no recovered variant to filter on.
        if logger:
            logger.debug( "picture sizes given explicitly; analysing %s"%(
                ", ".join( config.label for config in configurations ) ) )
        return configurations

    discovered = discover_cached_configurations(
        fps=fps, min_prob=min_prob, include_preview=include_preview, backend_names=names )
    if discovered:
        if wanted_variants is not None:
            kept = [ config for config in discovered if _keep( config ) ]
            if logger:
                skipped = [ config.label for config in discovered if config not in kept ]
                if skipped:
                    logger.debug( "variant filter %s skipped: %s"%(
                        sorted( wanted_variants ), ", ".join( skipped ) ) )
            if not kept:
                # Return nothing rather than falling through to the
                # "no caches on disk" fallback below. That fallback
                # lists every configured backend, so a typo in
                # --variants would silently analyse EVERYTHING while the
                # header claimed a filter was applied - output that
                # looks right and answers a different question. An empty
                # result is loud and correct: caches exist, none matched.
                if logger:
                    logger.warning( "no cached configuration matched variant(s) %s; "
                                    "found %s on disk"%(
                                        sorted( wanted_variants ),
                                        ", ".join( c.label for c in discovered ) ) )
                return []
            discovered = kept
        if logger:
            logger.debug( "discovered %d cached configuration(s): %s"%(
                len( discovered ), ", ".join( config.label for config in discovered ) ) )
        return discovered

    fallback = [
        CachedConfiguration( backend_name=name,
                             picture_sizes=list( bu_detector.get_picture_sizes( name ) ) )
        for name in sorted( names ) ]
    if logger:
        logger.debug( "no caches on disk; falling back to configured sizes: %s"%(
            ", ".join( config.label for config in fallback ) ) )
    return fallback


def describe_configurations( configurations ):
    """A one-line summary of a configuration list, for a log or header."""
    if not configurations:
        return "(none)"
    return ", ".join( config.label for config in configurations )


def resolve_picture_sizes( requested, backend_name ):
    """
    Decide which picture_sizes a tool should look for, per backend.

    Every analysis and tuning tool used to default its --picture-sizes
    flag to betaconfig.picture_sizes. That was correct when one
    shared list governed every backend. It is wrong now:
    picture_sizes resolves per backend (a backend's own block, then
    detector_backend['defaults'], then the adapter's native sizes, then
    the shared list), so nudenet_v3 runs at 320 or 640 while the shared
    list still reads [1280]. A tool defaulting to the shared list looks
    for caches at a size that backend never wrote and reports "no
    detection caches found" for a backend that has a full set of them.

    Analogy: the shared list is the address on an old envelope. The
    backend moved; mail still sent to the old address comes back marked
    "no such video", even though the backend is right there.

    Args:
        requested: What the user passed on the command line, or None /
            empty when they passed nothing. An explicit value always
            wins - a tool must stay able to inspect caches from a
            configuration that is no longer the current one.
        backend_name: The backend whose caches are being looked for.
            Never defaulted here; the caller is iterating backends and
            knows which one it is on.

    Returns:
        A list of ints.
    """
    if requested:
        return list( requested )
    return list( bu_detector.get_picture_sizes( backend_name ) )


def discover_full_run_videos( picture_sizes, fps, min_prob, backend_name,
                              include_preview=False ):
    """
    Find every file_hash with a complete set of detection caches.

    "Complete" means one cache per entry in picture_sizes at this fps
    and min_prob, produced by this backend: footage a real run actually
    finished, which is what the analysis tools want to reason about.

    backend_name is baked into the glob rather than left to a leading
    '*', so this never blends two backends' caches and never hands back
    a path that belongs to a different backend than the one asked for.

    Args:
        picture_sizes: Sizes that must ALL be present.
        fps: Sample rate to match.
        min_prob: global_min_prob to match.
        backend_name: Backend to match. Never defaulted - pin it.
        include_preview: When True, a file_hash with no real cache is
            still included if it has a complete set of preview caches at
            exactly ONE preview suffix. A real cache always wins over a
            preview one for the same file_hash, because a real run is
            strictly better data. A file_hash with preview caches at
            more than one offset is skipped rather than guessed at:
            blending two different slices into one "video" is worse than
            not reporting on it.

    Returns:
        A (result, preview_used_for) pair. result is
        {file_hash: [cache_path per size]}; preview_used_for is
        {file_hash: preview_suffix} for entries that came from preview
        caches, so a caller can print the caveat.
    """
    per_size_real = []
    per_size_preview = []   # [{file_hash: {preview_suffix: path}}, ...]

    for size in picture_sizes:
        real_paths = {}
        preview_paths = {}
        for path in glob.glob( _cache_glob( backend_name, size, fps, min_prob ) ):
            parsed = parse_cache_name( path, backend_name, size, fps, min_prob )
            if parsed is None:
                continue
            file_hash, _detection_key, preview_suffix = parsed
            if preview_suffix:
                if include_preview:
                    preview_paths.setdefault( file_hash, {} )[ preview_suffix ] = path
            else:
                real_paths[file_hash] = path
        per_size_real.append( real_paths )
        per_size_preview.append( preview_paths )

    if not per_size_real:
        return {}, {}

    result = {}
    common_real = set.intersection( *[ set( paths ) for paths in per_size_real ] )
    for file_hash in sorted( common_real ):
        result[file_hash] = [ paths[file_hash] for paths in per_size_real ]

    preview_used_for = {}
    if include_preview:
        candidates = set.intersection( *[ set( paths ) for paths in per_size_preview ] ) \
            if per_size_preview else set()
        candidates -= common_real
        for file_hash in sorted( candidates ):
            shared_suffixes = set.intersection(
                *[ set( paths[file_hash] ) for paths in per_size_preview ] )
            if len( shared_suffixes ) != 1:
                import betautils_log as bu_log
                if not shared_suffixes:
                    bu_log.get_logger().info(
                        "%s has preview caches for backend=%s but not at the same offset across "
                        "every picture size - skipping (re-run with one consistent "
                        "--preview-start-seconds)"%(file_hash, backend_name) )
                else:
                    bu_log.get_logger().info(
                        "%s has preview caches at more than one offset for backend=%s (%s) - "
                        "skipping rather than guessing which you meant"%(
                            file_hash, backend_name, sorted( shared_suffixes ) ) )
                continue
            preview_suffix = next( iter( shared_suffixes ) )
            result[file_hash] = [ paths[file_hash][preview_suffix] for paths in per_size_preview ]
            preview_used_for[file_hash] = preview_suffix

    return result, preview_used_for


def _walk_and_hash_for( target_hashes, remaining, found, search_dir ):
    """Hash every file under search_dir until every target hash is located."""
    if not os.path.isdir( search_dir ):
        return
    for root, _dirs, file_names in os.walk( search_dir ):
        for file_name in file_names:
            if not remaining:
                return
            path = os.path.join( root, file_name )
            try:
                file_hash = bu_hash.md5_for_file( path, 16 )
            except OSError:
                continue
            if file_hash in remaining:
                found[file_hash] = path
                remaining.discard( file_hash )


def build_hash_to_video_path( target_hashes ):
    """
    Map each target file_hash back to the source video that produced it.

    Searches betaconst.video_path_uncensored first, then falls back to
    betaconst.video_path_source_backup, so analysis still works against
    footage that has since been archived out of the working directory.

    Args:
        target_hashes: Iterable of file hashes to locate.

    Returns:
        {file_hash: path} for every hash found. Missing hashes are simply
        absent.
    """
    remaining = set( target_hashes )
    found = {}
    for search_dir in ( betaconst.video_path_uncensored, betaconst.video_path_source_backup ):
        if not remaining:
            break
        _walk_and_hash_for( target_hashes, remaining, found, search_dir )
    return found


def load_cached_raw_boxes( file_hash, picture_sizes, fps, min_prob, backend_name,
                           preview_suffix='' ):
    """
    Load and flatten every size's cached raw boxes for one video.

    Resolves the exact path for the CURRENT settings first, and falls
    back to any cache matching the same (backend, size, fps, min_prob)
    with a different detection key - which means an older run under
    different detection tunables. That fallback is deliberate for
    analysis tooling, which wants to read whatever is on disk, and is
    why missing_sizes is returned: a caller that needs to know it got a
    partial or mismatched picture can say so.

    Returns:
        A (raw_boxes, missing_sizes) pair.
    """
    raw_boxes = []
    missing_sizes = []
    for size in picture_sizes:
        path = box_hash_path_for( file_hash, size, fps, min_prob, backend_name, preview_suffix )
        if not os.path.exists( path ):
            path = None
            for candidate in sorted( glob.glob( _cache_glob( backend_name, size, fps, min_prob ) ) ):
                parsed = parse_cache_name( candidate, backend_name, size, fps, min_prob )
                if parsed and parsed[0] == file_hash and parsed[2] == preview_suffix:
                    path = candidate
                    break
        if path is None:
            missing_sizes.append( size )
            continue
        raw_boxes.extend( bu_hash.read_json( path ) )
    return raw_boxes, missing_sizes


# ---------------------------------------------------------------------------
# Shared stat formatting (used by every analysis tool's output)
# ---------------------------------------------------------------------------

def fmt_stats( values, fmt='%.1f', include_min=False ):
    """
    Format a list of numbers as every analysis tool's report line.

    One implementation, so two tools never disagree about what a
    "median" column means. Output shape:

        n=42  median=1.3  mean=1.6  p90=2.8  max=5.1

    Args:
        values: A list of numbers. Empty returns "n=0".
        fmt: Per-number format, e.g. '%.1fms' to bake in a unit.
        include_min: Also report the minimum, right after n. Some tools
            historically printed it and some did not; the flag preserves
            both rather than silently changing one tool's output.

    Returns:
        A single-line summary string.
    """
    if not values:
        return "n=0"
    values = sorted( values )
    count = len( values )
    mean = statistics.mean( values )
    median = statistics.median( values )
    p90 = values[ min( count-1, int( count*0.9 ) ) ]
    if include_min:
        template = "n=%%d  min=%s  median=%s  mean=%s  p90=%s  max=%s"%(fmt, fmt, fmt, fmt, fmt)
        return template%(count, values[0], median, mean, p90, values[-1])
    template = "n=%%d  median=%s  mean=%s  p90=%s  max=%s"%(fmt, fmt, fmt, fmt)
    return template%(count, median, mean, p90, values[-1])


def fmt_ms_stats( values_ms ):
    """fmt_stats for millisecond timings, with min included."""
    return fmt_stats( values_ms, fmt='%.1fms', include_min=True )


def percentile( values, fraction ):
    """
    Nearest-rank percentile of a list of numbers.

    Args:
        values: A non-empty list of numbers.
        fraction: 0.0 - 1.0.

    Returns:
        The value at that rank, or None for an empty list.
    """
    if not values:
        return None
    ordered = sorted( values )
    index = min( len( ordered ) - 1, max( 0, int( round( fraction * ( len( ordered ) - 1 ) ) ) ) )
    return ordered[index]
