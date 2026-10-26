"""
betautils_detector.py - The detector-adapter boundary.

Every detector backend implements the same three calls, so the rest of
BetaSuite never learns which concrete model is running:

    get_session()                             -> opaque backend handle
    raw_boxes_for_img( img, size, session, t )
    raw_boxes_for_imgs( imgs, size, session, ts )

Both raw_boxes_* return raw box dicts:

    {'x', 'y', 'w', 'h', 'class_id', 'score', 't'}

with 'class_id' being BetaSuite's own canonical string label from
betaconst.classes - never a model-native numeric index. Each adapter
owns translating its model's native classes into that vocabulary, and
owns all of its own preprocessing (normalisation and letterboxing are
not architecture-neutral, so they are not part of this interface).

Optional adapter hooks, all discovered with hasattr so an adapter that
does not need one simply omits it:

    validate_backend_config( config, errors )
        Sanity-check this backend's own config block at startup.
    detection_identity( config )
        The subset of this backend's settings that change which raw
        detections come out. Folded into the detection cache key, so
        changing one of them correctly invalidates cached detections.
        Anything omitted here is asserted not to affect detections.
    native_picture_sizes( config )
        The sizes this model was exported and validated at. Used as the
        convention-over-configuration default for picture_sizes, so a
        backend runs at its own correct resolution unless explicitly
        told otherwise.

CONFIGURATION SHAPE

    detector_backend = {
        'selected': '<backend name>',
        'defaults': { ... settings shared by every backend ... },
        '<backend name>': { ... that backend's own settings ... },
        ...
    }

Resolution order for any backend setting is always:
    1. detector_backend[<backend>][key]     this backend's own value
    2. detector_backend['defaults'][key]    the shared default
    3. the adapter's / BetaSuite's built-in default

Adding a backend is: write a module under detectors/ implementing the
three calls, register it in _BACKENDS, add its key to detector_backend.
"""

import fnmatch
import importlib
import os

import betaconfig
import betaconst


# Process-scoped backend override, read from the environment rather than
# betaconfig.py so an orchestrator can flip which backend a real
# subprocess run uses without ever writing betaconfig.py to disk (and so
# a crash mid-run can never leave it flipped).
_BACKEND_OVERRIDE_ENV_VAR = 'BETASUITE_DETECTOR_BACKEND_OVERRIDE'

# Same idea for the model variant, so a sweep across nudenet_v3's 320n
# and 640m exports needs no config edit.
_VARIANT_OVERRIDE_ENV_VAR = 'BETASUITE_DETECTOR_VARIANT_OVERRIDE'

# Registry of available backends. Imported lazily inside get_detector so
# an adapter's heavy imports are never paid for unless it is selected.
_BACKENDS = {
    'retinanet_v2': 'detectors.retinanet_v2',
    'nudenet_v3':   'detectors.nudenet_v3',
}

# Reserved keys inside detector_backend that are not backend names.
_RESERVED_BACKEND_KEYS = { 'selected', 'defaults' }


class Detector:
    """
    Reference shape every adapter module satisfies.

    Deliberately not an ABC: adapters are plain modules, which keeps
    them importable and testable without instantiating anything, and
    matches how the rest of BetaSuite is written.
    """

    def get_session( self ):
        """Load model state. The return value is opaque to every caller."""
        raise NotImplementedError

    def raw_boxes_for_img( self, img, size, session, t ):
        """Detect on one raw BGR uint8 image. Returns raw box dicts."""
        raise NotImplementedError

    def raw_boxes_for_imgs( self, imgs, size, session, ts ):
        """Batched sibling of raw_boxes_for_img. Returns one flat list."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def selected_backend_name():
    """
    The active backend's name.

    BETASUITE_DETECTOR_BACKEND_OVERRIDE wins over betaconfig.py for this
    process. Not validated here: get_detector() raises a clear error
    listing the valid choices.
    """
    env_override = os.environ.get( _BACKEND_OVERRIDE_ENV_VAR )
    if env_override:
        return env_override
    backend_setting = getattr( betaconfig, 'detector_backend', {} )
    return backend_setting.get( 'selected', 'retinanet_v2' )


def registered_backend_names():
    """Every backend name this build knows about, sorted."""
    return sorted( _BACKENDS.keys() )


def variant_names( backend_name=None ):
    """
    Every model variant a backend declares, sorted.

    Convention over configuration: an adapter that offers several sets
    of weights exposes them as a VARIANTS mapping, and this reads that
    mapping rather than asking config which one is selected. Tools use
    it so "measure this backend" means every variant it can run, not
    just the one currently switched on - which is how a 640m export sat
    unmeasured beside a 320n run for a day.

    Returns:
        A sorted list of variant names, or [] for a backend that has
        only one set of weights and therefore no variant axis.
    """
    try:
        module = get_detector( backend_name )
    except ValueError:
        return []
    variants = getattr( module, 'VARIANTS', None )
    if isinstance( variants, dict ):
        return sorted( variants )
    return []


def native_size_for_variant( backend_name, variant_name ):
    """
    The size a variant was exported and validated at, or None.

    A model run at a size it was not exported for is both slower and
    less accurate, so a measurement sweep that pairs every variant with
    every size mostly measures configurations nobody should use. This
    lets a sweep pair each variant with its own native size by default.
    """
    try:
        module = get_detector( backend_name )
    except ValueError:
        return None
    variants = getattr( module, 'VARIANTS', None )
    if not isinstance( variants, dict ):
        return None
    entry = variants.get( variant_name )
    if not isinstance( entry, dict ):
        return None
    size = entry.get( 'native_size' )
    return int( size ) if size else None


def get_detector( backend_name=None ):
    """
    Resolve a backend name to its adapter module.

    Raises:
        ValueError: unknown backend name, with the valid choices listed.
    """
    name = backend_name or selected_backend_name()
    if name not in _BACKENDS:
        raise ValueError(
            "unknown detector_backend %r - valid choices: %s"%(
                name, ', '.join( registered_backend_names() ) ) )
    return importlib.import_module( _BACKENDS[ name ] )


# ---------------------------------------------------------------------------
# Backend settings resolution
# ---------------------------------------------------------------------------

def get_backend_config( backend_name=None ):
    """
    One backend's own config block from detector_backend[<name>].

    Returns {} when detector_backend is unset or has no block for this
    backend; every adapter falls back to its own defaults for anything
    it reads.
    """
    name = backend_name or selected_backend_name()
    backend_setting = getattr( betaconfig, 'detector_backend', {} )
    block = backend_setting.get( name, {} )
    return block if isinstance( block, dict ) else {}


def get_backend_defaults():
    """
    detector_backend['defaults'] - settings shared by every backend.

    This is the "top level default in that section" tier: a value that
    is model-specific in principle, but for which one sensible value
    covers every backend that has not said otherwise.
    """
    backend_setting = getattr( betaconfig, 'detector_backend', {} )
    defaults = backend_setting.get( 'defaults', {} )
    return defaults if isinstance( defaults, dict ) else {}


def resolve_backend_setting( key, backend_name=None, fallback=None ):
    """
    Resolve one backend setting through the standard three tiers.

    Args:
        key: Setting name.
        backend_name: Backend to resolve for. Defaults to the selected one.
        fallback: Returned when neither tier defines the key.

    Returns:
        The resolved value.
    """
    backend_config = get_backend_config( backend_name )
    if key in backend_config:
        return backend_config[key]
    defaults = get_backend_defaults()
    if key in defaults:
        return defaults[key]
    return fallback


def selected_variant_name( backend_name=None ):
    """
    The active model variant for a backend, or None if it has no variants.

    BETASUITE_DETECTOR_VARIANT_OVERRIDE wins for this process, mirroring
    the backend override, so a 320-vs-640 sweep needs no config edit.
    """
    env_override = os.environ.get( _VARIANT_OVERRIDE_ENV_VAR )
    if env_override:
        return env_override
    return resolve_backend_setting( 'model_variant', backend_name, None )


def get_nn_batch_size( backend_name=None ):
    """
    Frames to batch into one inference call, for a given backend.

    Different backends genuinely want different values: retinanet_v2
    carries 146MB of weights and hits CUDA OOM at batch sizes that are
    comfortable for nudenet_v3's 12MB export.

    Resolution: backend block -> detector_backend['defaults'] -> 1.

    The module-level betaconfig.nn_batch_size tier was removed:
    batching is a VRAM/throughput property of a model, so it belongs in
    detector_backend. detector_backend['defaults'] is kept because a
    backend that has not stated a preference should still get a sane
    value rather than failing validation.

    Batching must never change WHICH detections come out, only how many
    frames are handed to the model per call. That is why nn_batch_size
    is excluded from the detection cache key, and it is enforced by
    tests/test_batch_size_invariance.py rather than assumed.
    """
    resolved = resolve_backend_setting( 'nn_batch_size', backend_name, None )
    if resolved is not None:
        return resolved
    return 1


def get_picture_sizes( backend_name=None ):
    """
    Detection sizes for a backend, as a list.

    Convention over configuration: a backend that declares
    native_picture_sizes() gets those by default, because a model run at
    a size it was neither exported nor validated at is both slower and
    less accurate. nudenet_v3's 320n export fed a 1280x1280 blob is 16x
    the anchors and a receptive field four times too small relative to
    object scale - it was the single largest cost in the earlier
    detection pass, and it silently biased every tuning number derived
    from that backend's output.

    Resolution:
        1. detector_backend[<backend>]['picture_sizes']
        2. detector_backend['defaults']['picture_sizes']
        3. the adapter's native_picture_sizes() for the active variant

    The module-level betaconfig.picture_sizes tier was removed:
    a detection size is a property of the model and its export, so a
    shared value could only ever be right for one backend at a time. A
    backend that declares neither its own picture_sizes nor a native
    size now fails validation instead of silently inheriting someone
    else's number.

    Returns:
        A list of ints, empty if the backend declares nothing (which
        _validate_picture_sizes reports as an error).
    """
    name = backend_name or selected_backend_name()
    resolved = resolve_backend_setting( 'picture_sizes', name, None )
    if resolved:
        return list( resolved )

    try:
        module = get_detector( name )
    except ValueError:
        module = None
    if module is not None and hasattr( module, 'native_picture_sizes' ):
        native = module.native_picture_sizes( get_backend_config( name ) )
        if native:
            return list( native )

    return []


def get_class_suppression( backend_name=None ):
    """
    The class_suppression ruleset for a backend.

    Per-backend rather than shared because two structurally different
    models produce different box geometry, and an IoU threshold
    calibrated against one model's boxes says nothing about the other's.

    Resolution: backend block -> detector_backend['defaults'] -> {}.

    The module-level betaconfig.class_suppression tier was removed for
    the same reason the rules are per-backend at all: the label
    vocabularies differ between models, so a shared ruleset could name
    classes one backend has never heard of.

    An empty result is legitimate - it means this backend arbitrates
    nothing and every detection stands on its own score.
    """
    resolved = resolve_backend_setting( 'class_suppression', backend_name, None )
    if resolved is not None:
        return resolved
    return {}


def get_class_suppression_refused( backend_name=None ):
    """
    Suppression pairs ('label<-suppressor') a person has decided against.

    The bench suggests a rule for every pair whose overlaps form a
    same-spot cluster, and it cannot tell a mislabel from two body parts
    that genuinely share a spot (exposed_vulva<-exposed_penis during
    penetration). A pair named here is never proposed as an addition
    again, so a decision made once stays made. It changes nothing a run
    renders.

    Resolution: backend block -> detector_backend['defaults'] -> [].
    """
    return list( resolve_backend_setting( 'class_suppression_refused', backend_name, None ) or [] )


def get_class_promotion( backend_name=None ):
    """
    The class_promotion ruleset for a backend: suppression's inverse.

    Suppression drops a detection when a competing label says it is
    something else. Promotion RELABELS a detection when corroborating
    evidence says it is something more specific - the motivating case
    being penetration, which NudeNet reports as covered_vulva because
    the penis is what is covering it.

    Per-backend for the same reason suppression is: label vocabularies
    and box geometry differ between models, so an overlap threshold
    calibrated on one says nothing about the other.

    Resolution: backend block -> detector_backend['defaults'] -> {}.
    Empty is legitimate and is the pre-promotion behaviour.
    """
    resolved = resolve_backend_setting( 'class_promotion', backend_name, None )
    return resolved if resolved else {}


def get_detection_identity( backend_name=None ):
    """
    The subset of a backend's settings that changes raw detections.

    Folded into the detection cache key by betautils_cache_paths, so
    changing (say) nudenet_v3's nms_iou correctly invalidates cached
    detections instead of silently serving back results produced under
    the old value - which it did previously.

    Returns:
        A JSON-serialisable dict, {} for an adapter with no such settings.
    """
    name = backend_name or selected_backend_name()
    try:
        module = get_detector( name )
    except ValueError:
        return {}
    if not hasattr( module, 'detection_identity' ):
        return {}
    identity = module.detection_identity( get_backend_config( name ) )
    return identity if isinstance( identity, dict ) else {}


# ---------------------------------------------------------------------------
# Per-label overrides
# ---------------------------------------------------------------------------

# item_overrides keys that are detection- or tracking-sensitive, and can
# therefore be set per backend under
# detector_backend[<name>]['item_overrides'][label].
#
# Deliberately NOT here: censor_style, censor_shape. Those are pure
# rendering choices, not derived from any model's behaviour, so they
# stay shared no matter which backend is selected.
ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS = {
    # confidence
    'min_prob', 'min_prob_continue',
    # geometry padding
    'width_area_safety', 'height_area_safety',
    # temporal padding
    'time_safety',
    # tracking / smoothing
    'position_smoothing', 'size_smoothing', 'style_min_dwell_seconds',
    'interpolation_enabled', 'interpolation_max_gap',
    'track_max_gap', 'match_distance_multiplier', 'min_track_hits', 'confirm_score',
    # paired style
    'paired_style', 'paired_style_max_distance', 'paired_style_tiebreak_margin',
    'paired_style_max_age',
    # geometry sanity filter
    'min_area_fraction', 'max_area_fraction',
    'min_aspect_ratio', 'max_aspect_ratio', 'geometry_action',
}


def get_variant_config( backend_name=None, variant_name=None ):
    """
    One variant's own config block, from
    detector_backend[<backend>]['variants'][<variant>].

    Returns {} for a backend with no variants, an unknown variant, or a
    variant that has not been given a block. A variant only needs one
    once it actually wants to differ.
    """
    name = backend_name or selected_backend_name()
    variant = variant_name or selected_variant_name( name )
    if not variant:
        return {}
    variants = get_backend_config( name ).get( 'variants', {} )
    if not isinstance( variants, dict ):
        return {}
    block = variants.get( variant, {} )
    return block if isinstance( block, dict ) else {}


def get_item_overrides( label, backend_name=None, variant_name=None ):
    """
    Effective item_overrides for one label, under one backend and variant.

    THREE TIERS, most general to most specific:

        betaconfig.item_overrides[label]
            Shared by everything: censor_style and censor_shape, which
            are about how censoring LOOKS and have nothing to do with
            which model produced the box.

        detector_backend[<backend>]['item_overrides'][label]
            What this backend wants, whichever variant is running.

        detector_backend[<backend>]['variants'][<variant>]['item_overrides'][label]
            What this specific set of weights wants.

    Each tier merges per KEY over the one above, so a variant that
    differs on one setting says so in one line and inherits everything
    else. A key set at a backend or variant tier that is not
    backend-tunable is ignored here and reported by betautils_config's
    validation rather than silently doing something unexpected.

    WHY THE VARIANT TIER EXISTS
    ---------------------------
    Two variants of one backend are two different sets of weights that
    detect differently, so their tracking wants different settings. The
    first real auto_tune run showed it directly: at
    match_distance_multiplier 2.0, nudenet_v3's 640m avoided 39 track
    resets AND had 51 fewer risky assignments - better on both axes -
    while 320n rejected every candidate. With only a backend tier, a
    shared key has to be a compromise that suits neither, or the better
    value goes unwritten. Which is what happened.

    Analogy: same recipe, two ovens. Most of it carries over; the bake
    time does not, and writing one bake time on the wall does not make
    the ovens agree.

    Args:
        label: The censored label.
        backend_name: Backend to resolve for. Defaults to the selected one.
        variant_name: Variant to resolve for. Defaults to the active one
            (which BETASUITE_DETECTOR_VARIANT_OVERRIDE can set), so a
            sweep needs no config edit.
    """
    name = backend_name or selected_backend_name()
    resolved = dict( getattr( betaconfig, 'item_overrides', {} ).get( label, {} ) )

    for block in ( get_backend_config( name ), get_variant_config( name, variant_name ) ):
        tier = block.get( 'item_overrides', {} )
        if not isinstance( tier, dict ):
            continue
        for key, value in tier.get( label, {} ).items():
            if key in ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS:
                resolved[key] = value
    return resolved


# ---------------------------------------------------------------------------
# Shared session construction
# ---------------------------------------------------------------------------

_cuda_libraries_preloaded = False


def preload_cuda_libraries( logger=None ):
    """
    Load the CUDA and cuDNN libraries that pip installed into the venv.

    requirements-gpu.txt ships CUDA 12 / cuDNN 9 as nvidia-* wheels so no
    system toolkit is needed, but onnxruntime does not search those
    site-packages folders on its own. Without this, the CUDA provider fails
    to load (libcublasLt.so.12 not found) and the run quietly lands on CPU.
    onnxruntime.preload_dlls() (1.21+) loads them from site-packages first,
    falling back to the system path. Safe to call more than once.
    """
    global _cuda_libraries_preloaded
    if _cuda_libraries_preloaded:
        return
    _cuda_libraries_preloaded = True

    import onnxruntime
    if not hasattr( onnxruntime, 'preload_dlls' ):
        return
    try:
        onnxruntime.preload_dlls()
    except Exception as e:
        import betautils_log as bu_log
        ( logger or bu_log.get_logger() ).warning(
            "onnxruntime.preload_dlls() failed (%s); CUDA and cuDNN must then be on the "
            "system library path"%(e) )


def build_onnx_session( model_path, logger=None, backend_name=None ):
    """
    Create an ONNX Runtime session and verify which provider it got.

    Both adapters used to request CUDA alone. When CUDA fails to
    initialise, onnxruntime falls back to CPU with a warning buried in
    stderr, and the run simply becomes 50-100x slower with nothing in
    the log to say why. This asks for a CPU fallback explicitly, then
    logs the provider actually in use so "is this on the GPU" is
    answerable from the log rather than from guesswork.

    Args:
        model_path: Path to the .onnx file.
        logger: Optional logger; betautils_log's shared one by default.
        backend_name: For log messages only.

    Returns:
        A ready onnxruntime.InferenceSession.

    Raises:
        FileNotFoundError: model_path does not exist. Fail fast with the
            path rather than surfacing an opaque ORT error mid-run.
    """
    import onnxruntime

    import betautils_log as bu_log
    logger = logger or bu_log.get_logger()

    if not os.path.exists( model_path ):
        raise FileNotFoundError(
            "detector model not found: %s (configure detector_backend[%r]'s "
            "model_path/model_variant, or place the file there)"%(
                model_path, backend_name or selected_backend_name() ) )

    if getattr( betaconfig, 'gpu_enabled', 0 ):
        preload_cuda_libraries( logger )
        providers = [
            ( 'CUDAExecutionProvider', { 'device_id': getattr( betaconfig, 'cuda_device_id', 0 ) } ),
            ( 'CPUExecutionProvider', {} ),
        ]
    else:
        providers = [ ( 'CPUExecutionProvider', {} ) ]

    session = onnxruntime.InferenceSession( model_path, providers=providers )
    active = session.get_providers()
    logger.info( "detector session: backend=%s model=%s providers=%s"%(
        backend_name or selected_backend_name(), os.path.basename( model_path ), active ) )

    if getattr( betaconfig, 'gpu_enabled', 0 ) and 'CUDAExecutionProvider' not in active:
        logger.warning(
            "gpu_enabled is set but CUDAExecutionProvider is NOT active (got %s) - "
            "this run is on CPU and will be dramatically slower. Check the onnxruntime-gpu "
            "install and CUDA/cuDNN versions."%(active) )

    return session


def session_provider_summary( session ):
    """
    Short description of a session's active execution providers.

    Recorded in the stats line so a slow run can be attributed to CPU
    fallback after the fact, without re-running anything.
    """
    try:
        return ','.join( session.get_providers() )
    except Exception:
        return 'unknown'


# ---------------------------------------------------------------------------
# Structure profiles
# ---------------------------------------------------------------------------
#
# One model's right timing values depend on how the FOOTAGE is cut, not
# only on the model. Measured on real files: a compilation ran 176.6
# shot cuts per minute with a 0.20s median shot, while ordinary
# scene-based footage runs 5-15 cuts/min with multi-second shots. A
# track_max_gap short enough to stop tracks welding across cuts in the
# first case resets tracks mid-scene in the second.
#
# Profiles live on the MODEL, not the variant: variants of one model
# share detection behaviour and box geometry (which is why they already
# share item_overrides), while different models genuinely differ.


def get_profiles( backend_name=None ):
    """This backend's profiles block, or {} when it has none."""
    block = get_backend_config( backend_name or selected_backend_name() )
    profiles = block.get( 'profiles' )
    return profiles if isinstance( profiles, dict ) else {}


def _dir_key_parts( key ):
    """
    A directory key split into comparable segments.

    Accepts either separator and tolerates doubled or trailing ones, so
    'comps', 'comps/', 'comps//' and 'comps\\' are one key. A LEADING
    separator is tolerated here but rejected by validation, because at
    that point the key is indistinguishable from a pasted absolute path,
    and an absolute path that silently matches nothing is worse than a
    rejected one.

    Segments are lowercased: a key that silently failed to match because
    the folder on disk is 'Comps' would hand that file to automatic
    selection while looking configured, and a quietly wrong profile is
    the failure this module works hardest to avoid.

    '..' is NOT filtered out here. Callers rely on seeing it to detect a
    path that climbed out of the tree.
    """
    text = str( key ).replace( '\\', '/' )
    return tuple( part.lower() for part in text.split( '/' )
                  if part not in ( '', '.' ) )


def source_dir_parts( source_path, uncensored_root=None ):
    """
    Where a source file sits relative to the uncensored-video root.

    Args:
        source_path: Path to the source video, absolute or relative.
        uncensored_root: Root of the source tree. Defaults to
            betaconst.video_path_uncensored.

    Returns:
        A tuple of lowercased directory segments between the root and
        the file, empty for a file sitting directly in the root, or None
        when the file is outside the tree or the root is unknown. None
        and () are different facts: () means "in the root", which takes
        automatic selection, while None means the question does not
        apply.
    """
    if not source_path:
        return None
    root = uncensored_root or getattr( betaconst, 'video_path_uncensored', None )
    if not root:
        return None
    try:
        relative = os.path.relpath( os.path.abspath( str( source_path ) ),
                                    os.path.abspath( str( root ) ) )
    except ValueError:
        # Windows: different drives have no relative path between them.
        return None
    parts = _dir_key_parts( os.path.dirname( relative ) )
    if parts and parts[0] == '..':
        return None
    return parts


def profile_for_source_dir( source_path, backend_name=None, uncensored_root=None ):
    """
    The profile a file's own folder asks for, or None.

    See source_dir_rule, which also reports WHICH key matched. This
    returns just the name.
    """
    return source_dir_rule( source_path, backend_name, uncensored_root )[0]


def source_dir_rule( source_path, backend_name=None, uncensored_root=None ):
    """
    The profile a file's own folder asks for, and the key that said so.

    betaconfig.profile_by_source_dir maps a folder, named relative to
    the uncensored-video root, to a profile:

        profile_by_source_dir = {
            'scene':        'scene',
            'compilation':  'compilation',
            'quick-cut':    'quick_cut',
            'split-screen': 'split_screen',
        }

    This is the folder convention: drop a video in 'split-screen/' and it
    is censored as a split-screen layout, with no measurement involved.
    It exists because the useful distinctions are not all measurable -
    a split-screen layout is invisible to every signal available before
    detection runs - and because sorting files into folders is something
    a person does once and can see at a glance.

    Add a folder by adding a line. The keys are ordinary folder names,
    not globs, so nothing has to be escaped and a name can never match
    part of a filename by accident.

    NESTING. A key may name a nested folder, and the DEEPEST matching
    key wins, so a child folder overrides its parent:

        profile_by_source_dir = {
            'compilation':              'compilation',
            'compilation/split-screen': 'split_screen',
        }

    That is how one broad folder holding mixed material gets a corner
    for the exception, rather than forcing a choice that is wrong for
    part of its contents.

    NOT MAPPED MEANS MEASURED. A file directly in the root, or in any
    folder with no key, falls through to automatic selection by measured
    shot structure. Only a folder named here opts out of measurement.

    An unknown profile name is ignored here and reported by validation.
    Selecting nothing silently would render the wrong thing and report
    success.

    Returns:
        (profile_name, matched_key) where matched_key is the folder key
        that decided, as a '/'-joined string, for the run log. Both are
        None when no key matches.
    """
    mapping = getattr( betaconfig, 'profile_by_source_dir', None )
    if not mapping:
        return ( None, None )
    parts = source_dir_parts( source_path, uncensored_root )
    if not parts:
        # None (outside the tree) and () (in the root) both mean no
        # folder rule applies.
        return ( None, None )
    variants = get_profiles( backend_name ).get( 'variants' ) or {}
    best_depth, best_name, best_key = 0, None, None
    for key, name in mapping.items():
        key_parts = _dir_key_parts( key )
        if not key_parts:
            # An empty key would match every file, which is what the
            # root fallthrough already means. Ignored, and reported by
            # validation.
            continue
        if parts[:len( key_parts )] == key_parts and len( key_parts ) > best_depth:
            best_depth = len( key_parts )
            best_name, best_key = name, '/'.join( key_parts )
    if best_name is None or best_name not in variants:
        return ( None, None )
    return ( best_name, best_key )


def profile_for_path( source_path, backend_name=None ):
    """
    The profile a file's own path asks for, or None.

    betaconfig.profile_by_path_pattern maps a glob to a profile name:

        profile_by_path_pattern = {
            '*/splitscreen/*': 'split_screen',
            '*multicam*':      'split_screen',
        }

    This is the "a specific video or directory" case, and the only route
    to a profile no measurement can select: a splitscreen/ folder whose
    contents are split-screen layouts, which cuts and shot lengths cannot
    distinguish from ordinary single-scene footage.
    Patterns are matched against the full path and against the bare
    filename, so '*reveal*' catches both a reveal/ directory and a file
    with reveal in its name.

    First match in declaration order wins, so the most specific pattern
    goes first. An unknown profile name is ignored here and reported by
    validation - silently selecting nothing would be the quiet-wrong-output
    failure this module keeps trying to avoid.
    """
    if not source_path:
        return None
    mapping = getattr( betaconfig, 'profile_by_path_pattern', None )
    if not mapping:
        return None
    variants = get_profiles( backend_name ).get( 'variants' ) or {}
    full = str( source_path )
    base = os.path.basename( full )
    for pattern, name in mapping.items():
        if fnmatch.fnmatch( full, pattern ) or fnmatch.fnmatch( base, pattern ):
            return name if name in variants else None
    return None


SHORT_SHOT_SECONDS = 1.0

# Which per-variant threshold key each match_on signal reads. Defined
# once here so the selector and config validation cannot disagree about
# what a signal means.
PROFILE_THRESHOLD_KEYS = {
    'cuts_per_min':        'min_cuts_per_min',
    'short_shot_fraction': 'min_short_shot_fraction',
}


def shot_spans_for( shot_cut_times, scanned_seconds=None ):
    """
    Shot lengths in seconds, from cut timestamps.

    The shots measured are the spans BETWEEN consecutive cuts, plus the
    opening span from 0 to the first cut and the closing span from the
    last cut to the end of the scanned window when that length is known.
    Leaving the end spans out biases every measure built on them: a
    40-minute file with two cuts a second apart would otherwise look
    entirely composed of 1-second shots.

    Args:
        shot_cut_times: Cut timestamps in seconds, any order.
        scanned_seconds: Length of the window scanned, or None. None
            drops the closing span rather than inventing an end time.

    Returns:
        A list of positive span lengths, empty when there is nothing to
        measure: cuts never scanned (None), or no cuts and no known
        window.
    """
    if shot_cut_times is None:
        return []
    times = sorted( float( t ) for t in shot_cut_times )
    if not times:
        # Scanned, and no cuts found: the whole window is one shot.
        # This used to return [] exactly as the unscanned case did, so a
        # scan that found zero cuts reported "no shot-cut data", fell
        # through to the configured default, and showed as n/a in
        # betabench. 13 of 56 files in one preview run took that path.
        # Today's default happens to be the profile zero cuts measure
        # as, which is the only reason it never changed an output.
        if scanned_seconds and scanned_seconds > 0:
            return [ float( scanned_seconds ) ]
        return []
    spans = [ times[0] ] if times[0] > 0 else []
    spans.extend( later - earlier
                  for earlier, later in zip( times, times[1:] ) )
    if scanned_seconds and scanned_seconds > times[-1]:
        spans.append( scanned_seconds - times[-1] )
    return [ span for span in spans if span > 0 ]


def short_shot_fraction_for( shot_cut_times, scanned_seconds=None,
                             short_seconds=None ):
    """
    Share of the scanned RUNTIME spent in shots shorter than a cutoff.

    Weighted by duration, not by count, and that is the whole point.
    Counting shots lets a brief flurry decide the answer for a whole
    file: on the tuning set a video whose cut detector fired on twenty
    consecutive samples inside one 2-second transition reported a 0.10s
    MEDIAN shot while spending 4.7% of its runtime in short shots, and a
    genuinely fast-cutting video reported a LONGER median of 0.50s while
    spending 23.6%. Ranked by median the two come out backwards; ranked
    by duration share they separate by 5x.

    The question a profile actually needs answered is how much of the
    footage is made of shots too short to track through, and that is a
    duration question.

    Args:
        shot_cut_times: Cut timestamps in seconds.
        scanned_seconds: Length of the window scanned, or None.
        short_seconds: Shots shorter than this count as short. Defaults
            to SHORT_SHOT_SECONDS.

    Returns:
        A fraction from 0 to 1, or None when there is nothing to
        measure. None means "not measured", never "no short shots": the
        caller must not read it as fast or slow.
    """
    spans = shot_spans_for( shot_cut_times, scanned_seconds )
    if not spans:
        return None
    cutoff = SHORT_SHOT_SECONDS if short_seconds is None else short_seconds
    total = sum( spans )
    if total <= 0:
        return None
    return sum( span for span in spans if span < cutoff ) / total


def median_shot_seconds_for( shot_cut_times, scanned_seconds=None ):
    """
    Median shot length in seconds, or None when nothing to measure.

    Reported in stats for context. NOT used to select a profile: see
    short_shot_fraction_for for why a median ranks flurries above
    genuinely fast-cut footage.
    """
    spans = sorted( shot_spans_for( shot_cut_times, scanned_seconds ) )
    if not spans:
        return None
    middle = len( spans ) // 2
    if len( spans ) % 2:
        return spans[middle]
    return 0.5 * ( spans[middle - 1] + spans[middle] )


def scanned_window_seconds( duration_seconds, offset_seconds=0.0, max_seconds=None ):
    """
    Length of footage a shot-cut scan actually covered, in seconds.

    The one definition of a structure measurement's denominator, shared
    by betatv, the tuning replay and betabench. Each used to derive it
    differently, and two of them derived it from the spread between the
    first and last DETECTION, which is the length of the nudity, not of
    the footage. betabench scored a file whose censored boxes fell in a
    single sampled frame at a 1-microsecond window, 600,000,000 cuts a
    minute and "100% of runtime in sub-1s shots", and recommended moving
    it to quick-cut.

    Args:
        duration_seconds: The whole video's length, or None if unknown.
        offset_seconds: Where the scan started (a preview slice offset).
        max_seconds: The scan's bound (a preview window), or None for
            "to the end of the file".

    Returns:
        Seconds scanned, or None when the duration is unknown. A bound
        past the end of the file is clamped to what was there to scan.
    """
    if not duration_seconds or duration_seconds <= 0:
        return None
    remaining = max( 0.0, float( duration_seconds ) - float( offset_seconds or 0.0 ) )
    if max_seconds:
        remaining = min( remaining, float( max_seconds ) )
    return remaining if remaining > 0 else None


class StructureSignals( object ):
    """
    What a shot-cut scan says about a video's structure.

    Every field is None when not measured, never 0: "cuts never scanned"
    and "scanned and found none" are different facts, and only the
    second is evidence of anything.
    """
    __slots__ = ( 'cut_count', 'scanned_seconds', 'cuts_per_min',
                  'short_shot_fraction', 'median_shot_seconds' )

    def __init__( self, cut_count=None, scanned_seconds=None, cuts_per_min=None,
                  short_shot_fraction=None, median_shot_seconds=None ):
        self.cut_count = cut_count
        self.scanned_seconds = scanned_seconds
        self.cuts_per_min = cuts_per_min
        self.short_shot_fraction = short_shot_fraction
        self.median_shot_seconds = median_shot_seconds

    @property
    def is_measured( self ):
        return self.short_shot_fraction is not None


def structure_signals( shot_cut_times, scanned_seconds ):
    """
    Every structure measure for one video, from its cuts and its window.

    One function for betatv's profile selection, the tuning replay and
    betabench, so the three cannot drift. They had: the replay passed a
    cut rate while profiles match on short_shot_fraction, and betabench
    divided by the spread of detections rather than the window scanned.

    Args:
        shot_cut_times: Cut timestamps in seconds, [] when scanned with
            none found, None when never scanned.
        scanned_seconds: From scanned_window_seconds, or None.

    Returns:
        A StructureSignals. Unmeasured unless both inputs are known.
    """
    if shot_cut_times is None or not scanned_seconds or scanned_seconds <= 0:
        return StructureSignals()
    return StructureSignals(
        cut_count=len( shot_cut_times ),
        scanned_seconds=float( scanned_seconds ),
        cuts_per_min=60.0 * len( shot_cut_times ) / float( scanned_seconds ),
        short_shot_fraction=short_shot_fraction_for( shot_cut_times, scanned_seconds ),
        median_shot_seconds=median_shot_seconds_for( shot_cut_times, scanned_seconds ) )


def select_profile_name( cuts_per_min, backend_name=None, source_path=None,
                         short_shot_fraction=None ):
    """
    Which profile this video should use. See profile_selection.

    Returns just the name. Callers that want to report WHY a profile was
    chosen use profile_selection instead.
    """
    return profile_selection( cuts_per_min, backend_name, source_path,
                              short_shot_fraction )[0]


def declared_profile_selection( source_path=None, backend_name=None ):
    """
    The profile someone DECLARED for this video, before anything is measured.

    Precedence steps 1 to 3 of profile_selection: --profile, a path
    pattern, the folder convention. Split out so a caller can know the
    profile, and therefore the sample rate and the output name, before
    the shot-cut scan runs. betatv uses that to start detection while
    the scan is still going, and to skip a finished file without
    scanning it at all. profile_selection calls this first, so the
    precedence ladder is still defined in exactly one place.

    Returns:
        (profile_name, reason) when a declaration decides, including the
        (None, reason) of a forced name that is not a variant; None when
        nothing was declared and the choice depends on measurement.
    """
    variants = get_profiles( backend_name ).get( 'variants' ) or {}

    # An explicit choice wins over everything, including the off switch:
    # asking for a profile by name and silently getting none because
    # profiles were disabled in config would be the same class of
    # confusing failure the off-switch comment in profile_selection
    # describes.
    forced = getattr( betaconfig, 'force_structure_profile', None )
    if forced:
        return ( ( forced, 'forced by --profile' ) if forced in variants
                 else ( None, 'forced profile %r is not a variant'%(forced,) ) )

    # A path rule is a standing explicit choice, so it outranks the cut
    # rate and the off switch for the same reason --profile does. It loses
    # to --profile, which is the one-off override.
    by_path = profile_for_path( source_path, backend_name )
    if by_path:
        return ( by_path, 'matched profile_by_path_pattern' )

    # The folder convention. Below path patterns, above measurement.
    by_dir, matched_key = source_dir_rule( source_path, backend_name )
    if by_dir:
        return ( by_dir, 'source folder %s/'%( matched_key, ) )

    return None


def profile_selection( cuts_per_min, backend_name=None, source_path=None,
                       short_shot_fraction=None ):
    """
    Which profile this video should use, and which rule decided.

    PRECEDENCE, most specific first:

        1. --profile / force_structure_profile   one run, one answer
        2. profile_by_path_pattern               a glob, can name one file
        3. profile_by_source_dir                 the folder convention
        4. measured shot structure               match_on + thresholds
        5. the configured default

    A path pattern outranks the folder convention because it can target
    a single file, and a rule about one file is a more specific statement
    than a rule about a folder of them. Both outrank measurement, because
    both are someone saying what the footage is rather than a proxy
    guessing.

    Returns:
        (profile_name, reason). profile_name is None when this backend
        has no profiles or profiles are switched off. reason is a short
        phrase naming the rule that decided, for the run log: a folder
        key that matches nothing on disk otherwise looks configured
        while quietly falling through to measurement.

    Two signals are available, both from the shot-cut scan, which is the
    only measurement that has run by the time a profile must be chosen.
    'match_on' in the profiles block picks which one decides, and each
    reads its own threshold key:

        cuts_per_min         min_cuts_per_min
        short_shot_fraction  min_short_shot_fraction

    Both run the same direction: higher means faster cutting, so the
    highest threshold cleared is the most specific match.

    Prefer short_shot_fraction. Cuts per minute divides cuts by the whole
    runtime, so a long file containing one rapid-fire stretch reads as
    slow, and a short flurry of detector false cuts reads as fast. The
    duration share asks how much of the footage is actually made of
    shots too short to track through, which is the thing a denser sample
    rate is being bought to fix.

    Args:
        cuts_per_min: Measured shot cuts per minute, or None when shot
            cuts were never scanned. None picks the configured default
            rather than guessing, because "no cuts looked for" and "no
            cuts in this footage" are different facts.
        backend_name: Backend to read profiles from.
        source_path: Full path, for profile_by_path_pattern rules.
        short_shot_fraction: Share of runtime in short shots, 0 to 1, or
            None when unavailable. Same None semantics.
    """
    declared = declared_profile_selection( source_path, backend_name )
    if declared is not None:
        return declared
    return measured_profile_selection( cuts_per_min, backend_name, short_shot_fraction )


def measured_profile_selection( cuts_per_min, backend_name=None, short_shot_fraction=None ):
    """
    The profile measurement alone would choose: steps 4 and 5 of
    profile_selection, ignoring every declaration.

    Exposed so betatv can say when a declared profile samples LESS
    densely than the footage measures for. A file sorted into a folder
    whose profile runs at 9 fps while its cutting measures as quick_cut
    (18 fps) loses detections on exactly the shots that most need them,
    and nothing else would ever point that out.

    Args and Returns: as profile_selection.
    """
    profiles = get_profiles( backend_name )
    variants = profiles.get( 'variants' ) or {}

    # The off switch. Without this check the setting existed in config
    # and in the censor key but was read by nothing, so turning profiles
    # off changed the output filename and not the output - the most
    # confusing possible failure, because it looks like it worked.
    if not getattr( betaconfig, 'default_profile_enabled', True ):
        return ( None, 'profiles disabled' )

    if not variants:
        return ( None, 'backend has no profiles' )

    # manual_only profiles are invisible to automatic selection. They
    # exist so a profile can be written for footage the cut rate cannot
    # identify - a compilation that happens to cut slowly, say - and
    # reached only by naming it.
    auto = { name: variant for name, variant in variants.items()
             if not variant.get( 'manual_only' ) }
    if not auto:
        return ( None, 'every profile is manual_only' )

    default_name = profiles.get( 'default' )
    fallback = ( ( default_name, 'the configured default' )
                 if default_name in auto
                 else ( None, 'the default names no auto-selectable variant' ) )

    match_on = profiles.get( 'match_on', 'cuts_per_min' )
    threshold_key = PROFILE_THRESHOLD_KEYS.get( match_on )
    if threshold_key is None:
        return fallback

    measured = ( short_shot_fraction if match_on == 'short_shot_fraction'
                 else cuts_per_min )
    if measured is None:
        return fallback

    # Highest threshold the measurement clears, so thresholds can be
    # written in any order and adding one cannot silently reorder the
    # others. A variant with no threshold defaults to 0 and so is only
    # ever beaten, never a winner over a tuned one, which is what lets
    # the catch-all profile sit in the same list as the tuned ones.
    best_name, best_threshold = None, None
    for name, variant in auto.items():
        threshold = variant.get( threshold_key, 0 )
        if measured >= threshold and ( best_threshold is None
                                       or threshold > best_threshold ):
            best_name, best_threshold = name, threshold

    if best_name is not None:
        return ( best_name, 'measured %s'%( match_on, ) )
    return fallback


def get_profile_sample_fps( profile_name, backend_name=None ):
    """
    The detection sample rate a profile asks for, or the global one.

    A profile may set 'video_censor_fps' so fast-cut footage is sampled
    more densely than single-scene footage. Shots in a compilation can
    be a fraction of a second long, and at the global rate a short shot
    yields too few samples to clear min_track_hits at all - its exposure
    is never censored. Denser sampling costs detection time roughly in
    proportion, so it is worth paying only where the footage needs it.

    Args:
        profile_name: The selected profile, or None.
        backend_name: Backend to read profiles from.

    Returns:
        A positive number of samples per second.
    """
    base = getattr( betaconfig, 'video_censor_fps', 9 )
    if not profile_name:
        return base
    variant = ( get_profiles( backend_name ).get( 'variants' ) or {} ).get( profile_name )
    if not isinstance( variant, dict ):
        return base
    return variant.get( 'video_censor_fps', base )


def profile_sample_rates( backend_name=None ):
    """
    Every sample rate any profile of this backend could select, plus the
    global rate.

    A single entry means the output filename can be resolved before the
    shot-cut scan runs (nothing about the footage can change it), which
    is what lets a repeat run skip without even reading the scan cache.
    """
    rates = { getattr( betaconfig, 'video_censor_fps', 9 ) }
    if getattr( betaconfig, 'default_profile_enabled', True ):
        for variant in ( get_profiles( backend_name ).get( 'variants' ) or {} ).values():
            if isinstance( variant, dict ) and 'video_censor_fps' in variant:
                rates.add( variant['video_censor_fps'] )
    return rates


def get_profile_item_overrides( profile_name, label, backend_name=None ):
    """
    One label's overrides from one profile, or {} when there are none.

    These layer ON TOP of the backend's own item_overrides, so a profile
    only has to state the keys it actually changes.
    """
    if not profile_name:
        return {}
    variant = ( get_profiles( backend_name ).get( 'variants' ) or {} ).get( profile_name )
    if not isinstance( variant, dict ):
        return {}
    overrides = variant.get( 'item_overrides' ) or {}
    found = overrides.get( label )
    return found if isinstance( found, dict ) else {}
