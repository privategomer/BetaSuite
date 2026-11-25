"""
betautils_config.py - Reads and validates betaconfig.py, and a few
config-driven runtime helpers (the input-delete-probability safety
prompt, and get_parts_to_blur which every entry point calls to build its
final per-item censoring settings).

Orchestration for the main entry point, validate_config():
    validate_config()
        -> _collect_validation_warnings()    # runs every non-fatal sub-check,
                                              # each appending to one shared
                                              # warnings list - printed but
                                              # never aborts the run
            -> _validate_class_suppression_zero_margin
        -> _collect_validation_errors()      # runs every check below,
                                              # each appending to one shared
                                              # errors list
            -> _validate_registry            # every registered value's kind,
                                              # range and choices, from
                                              # betautils_settings
            -> _validate_items_to_censor ... _validate_local_config
                                              # what the registry cannot say
                                              # about one value alone: rules
                                              # between values, unknown keys,
                                              # structures (styles, rules,
                                              # profiles), files on disk
        -> prints warnings (if any), then prints + raises SystemExit(1) if
           _collect_validation_errors returned anything, otherwise returns
           None

Each error is a betautils_settings.Problem: its English text for the CLI
and the log, plus a code and params for the engine protocol, which never
sends sentences. A registry check's code says what was wrong
(config.bad_type, config.out_of_range, config.bad_choice); every other
check's code is config.rule with a stable rule id ('area.what') given
where the error is raised.

A limit is stated once, in the registry. The style, rule and per-label
checks here read its field tables rather than keeping their own, so the
app's sliders and menus and the CLI's validation cannot disagree.
"""

import os
import random
import threading

import betaconfig
import betaconst
import betautils_detector as bu_detector
import betautils_local as bu_local
import betautils_paths as bu_paths
import betautils_reverse as bu_reverse
import betautils_settings as bu_settings


# ---------------------------------------------------------------------------
# One-shot warnings
# ---------------------------------------------------------------------------

_warned_messages = set()
_warn_lock = threading.Lock()


def warn_once( message ):
    """
    Log a warning the first time it is seen, and never again.

    Used for per-box conditions - an unreadable sticker PNG, a style
    with no usable images - where the underlying problem is a config
    mistake that would otherwise emit one identical line per box per
    frame for the length of the video.
    """
    with _warn_lock:
        if message in _warned_messages:
            return
        _warned_messages.add( message )
    import betautils_log as bu_log
    bu_log.get_logger().warning( message )


def reset_warn_once_for_tests():
    """Forget which warnings have been emitted. Test-support only."""
    with _warn_lock:
        _warned_messages.clear()


# ---------------------------------------------------------------------------
# Resolved per-label settings
# ---------------------------------------------------------------------------

# get_parts_to_blur is called once per video, but process_raw_box used to
# call it once per RAW DETECTION - rebuilding the whole table and
# re-resolving every label's per-backend overrides tens of thousands of
# times per file. The table is a pure function of betaconfig plus the
# backend name, so it is memoised here and invalidated whenever a CLI
# override or a test mutates betaconfig.
_parts_cache = {}
_parts_cache_lock = threading.Lock()


def invalidate_config_caches():
    """
    Drop memoised views of betaconfig.

    MUST be called after anything mutates the betaconfig module at
    runtime - CLI overrides, a tuning harness, a test. betautils_cli
    calls it for you.
    """
    with _parts_cache_lock:
        _parts_cache.clear()


def get_label_settings( label, backend_name=None ):
    """
    Every render-time setting for one label, fully resolved.

    Args:
        label: A class name from betaconst.classes.
        backend_name: Backend whose per-label overrides apply.

    Returns:
        A dict with every key the pipeline reads for this label. Keys
        whose feature is off are present and None, so the censor cache
        key hashes a stable shape rather than a varying one. The one
        exception is confirm_score, present only while set: it arrived
        after a library had been rendered, and keying it while off would
        have re-rendered all of it for no change.
    """
    override = bu_detector.get_item_overrides( label, backend_name )
    settings = {
        # confidence
        'min_prob':           override.get( 'min_prob', betaconfig.default_min_prob ),
        'min_prob_continue':  override.get( 'min_prob_continue',
                                            getattr( betaconfig, 'default_min_prob_continue', None ) ),
        # geometry padding
        'width_area_safety':  override.get( 'width_area_safety', betaconfig.default_area_safety ),
        'height_area_safety': override.get( 'height_area_safety', betaconfig.default_area_safety ),
        # temporal padding
        'time_safety':        override.get( 'time_safety', betaconfig.default_time_safety ),
        # rendering
        'censor_style':       override.get( 'censor_style', betaconfig.default_censor_style ),
        'censor_shape':       override.get( 'censor_shape',
                                            getattr( betaconfig, 'default_censor_shape', 'box' ) ),
        # tracking
        'position_smoothing': override.get( 'position_smoothing',
                                            betaconfig.default_position_smoothing ),
        'interpolation_enabled': override.get( 'interpolation_enabled',
                                               betaconfig.default_interpolation_enabled ),
        'interpolation_max_gap': override.get( 'interpolation_max_gap',
                                               betaconfig.default_interpolation_max_gap ),
        'track_max_gap':      override.get( 'track_max_gap', None ),
        'match_distance_multiplier': override.get( 'match_distance_multiplier', 1.0 ),
        'min_track_hits':     override.get( 'min_track_hits',
                                            getattr( betaconfig, 'default_min_track_hits', 1 ) ),
        # paired style
        'paired_style':       override.get( 'paired_style', False ),
        'paired_style_max_distance': override.get(
            'paired_style_max_distance', betaconfig.default_paired_style_max_distance ),
        'paired_style_tiebreak_margin': override.get(
            'paired_style_tiebreak_margin', betaconfig.default_paired_style_tiebreak_margin ),
        # geometry sanity filter
        'min_area_fraction':  override.get( 'min_area_fraction',
                                            getattr( betaconfig, 'default_min_area_fraction', None ) ),
        'max_area_fraction':  override.get( 'max_area_fraction',
                                            getattr( betaconfig, 'default_max_area_fraction', None ) ),
        'min_aspect_ratio':   override.get( 'min_aspect_ratio',
                                            getattr( betaconfig, 'default_min_aspect_ratio', None ) ),
        'max_aspect_ratio':   override.get( 'max_aspect_ratio',
                                            getattr( betaconfig, 'default_max_aspect_ratio', None ) ),
        'geometry_action':    override.get( 'geometry_action',
                                            getattr( betaconfig, 'default_geometry_action',
                                                     _geometry_action_default() ) ),
        # Tracking reads these three per label too (betautils_track's
        # per-label settings). They were missing here, so the censor key
        # did not move when one changed at the backend or model-variant
        # tier, and the old render was reused as the new result.
        'size_smoothing':     override.get( 'size_smoothing',
                                            getattr( betaconfig, 'default_size_smoothing', None ) ),
        'style_min_dwell_seconds': override.get(
            'style_min_dwell_seconds', getattr( betaconfig, 'default_style_min_dwell_seconds', 0.0 ) ),
        'paired_style_max_age': override.get(
            'paired_style_max_age', getattr( betaconfig, 'default_paired_style_max_age', 1.0 ) ),
    }
    # Present only while set, so adding the setting left every existing
    # render's censor key, and so every finished output, valid.
    confirm_score = override.get( 'confirm_score', getattr( betaconfig, 'default_confirm_score', None ) )
    if confirm_score is not None:
        settings['confirm_score'] = confirm_score
    return settings


def _default_setting_for( key ):
    """The default_* setting get_label_settings falls back to for key, or None."""
    name = 'default_area_safety' if key.endswith( '_area_safety' ) else 'default_' + key
    return name if hasattr( betaconfig, name ) else None


# Why a layer in a label's chain does not apply (label_view).
INACTIVE_BACKEND = 'backend_not_selected'
INACTIVE_VARIANT = 'variant_not_selected'
INACTIVE_NOT_TUNABLE = 'not_backend_tunable'
INACTIVE_WINDOW_LOOK = 'look_key_in_window'
INACTIVE_REVERSE = 'reverse_off'
INACTIVE_NOT_WINDOW = 'not_a_window'
INACTIVE_IS_WINDOW = 'is_a_window'
PER_PROFILE = 'per_profile'

_ABSENT = object()


def _held( block, *keys ):
    """block[k1][k2]..., through dicts only, or _ABSENT."""
    for key in keys:
        if not isinstance( block, dict ) or key not in block:
            return _ABSENT
        block = block[key]
    return block


def _mapping( value ):
    return value if isinstance( value, dict ) else {}


def _label_chain( label, key, backend, variant, window, reverse_censor ):
    """
    Every place one label's key is set, lowest precedence first, each
    saying why it does not apply when it does not. Profiles come last:
    they apply on top, per file.
    """
    chain = []
    default = _default_setting_for( key )
    if default is not None:
        chain.append( { 'source': default, 'value': getattr( betaconfig, default ) } )
    else:
        chain.append( { 'source': bu_detector.SOURCE_BUILTIN } )

    value = _held( getattr( betaconfig, 'item_overrides', {} ), label, key )
    if value is not _ABSENT:
        layer = { 'source': 'item_overrides.%s.%s'%( label, key ), 'value': value }
        if window and key in bu_reverse.LOOK_KEYS:
            layer['inactive'] = INACTIVE_WINDOW_LOOK
        chain.append( layer )

    tunable = key in bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
    blocks = _mapping( getattr( betaconfig, 'detector_backend', {} ) )
    names = [ name for name in bu_detector.registered_backend_names() if isinstance( blocks.get( name ), dict ) ]
    for name in names:
        layers = [ ( 'detector_backend.%s.item_overrides.%s.%s'%( name, label, key ),
                     _held( blocks[name], 'item_overrides', label, key ), None ) ]
        for other in sorted( _mapping( blocks[name].get( 'variants' ) ) ):
            layers.append( ( 'detector_backend.%s.variants.%s.item_overrides.%s.%s'%( name, other, label, key ),
                             _held( blocks[name], 'variants', other, 'item_overrides', label, key ), other ) )
        for source, value, its_variant in layers:
            if value is _ABSENT:
                continue
            layer = { 'source': source, 'value': value }
            if name != backend:
                layer['inactive'] = INACTIVE_BACKEND
            elif its_variant is not None and its_variant != variant:
                layer['inactive'] = INACTIVE_VARIANT
            elif not tunable:
                layer['inactive'] = INACTIVE_NOT_TUNABLE
            chain.append( layer )

    reverse = getattr( betaconfig, 'reverse', {} )
    for source, value, applies, reason in (
            ( 'reverse.windows.%s.%s'%( label, key ), _held( reverse, 'windows', label, key ),
              window, INACTIVE_NOT_WINDOW ),
            ( 'reverse.censor.overrides.%s.%s'%( label, key ),
              _held( reverse, 'censor', 'overrides', label, key ), reverse_censor, INACTIVE_IS_WINDOW ) ):
        if value is _ABSENT:
            continue
        layer = { 'source': source, 'value': value }
        if not applies:
            layer['inactive'] = INACTIVE_REVERSE if not bu_reverse.enabled() else reason
        chain.append( layer )
    if window and key == 'censor_style' and _held( reverse, 'windows', label, key ) is _ABSENT:
        chain.append( { 'source': bu_detector.SOURCE_BUILTIN, 'value': dict( bu_reverse.SHOW_STYLE ) } )

    for name in names:
        profiles = _mapping( _held( blocks[name], 'profiles', 'variants' ) )
        for profile in sorted( profiles ):
            value = _held( profiles, profile, 'item_overrides', label, key )
            if value is not _ABSENT:
                chain.append( {
                    'source': 'detector_backend.%s.profiles.variants.%s.item_overrides.%s.%s'%(
                        name, profile, label, key ),
                    'value': value, 'profile': profile,
                    'inactive': PER_PROFILE if name == backend else INACTIVE_BACKEND } )
    return chain


def label_view( label, backend_name=None ):
    """
    One label as a run resolves it, for the app's per-label rows.

    Returns { key: { 'value', 'source', 'chain', 'off'? } } for every key
    get_label_settings returns. source is the dotted settings path that
    supplied the value, or 'builtin'. chain lists every place the key is
    set, lowest precedence first, and marks the one in effect
    ('wins': True); a layer that does not apply says why ('inactive'),
    and a profile's layer ('per_profile') applies on top for files that
    get that profile. off, when present, names the per-label key whose
    resolved value turns this one's effect off
    (betautils_settings.LABEL_REQUIRES).
    """
    backend = backend_name or bu_detector.selected_backend_name()
    variant = bu_detector.selected_variant_name( backend )
    window = bu_reverse.window_entry( label ) is not None
    reverse_censor = not window and bu_reverse.censor_override( label ) is not None
    _overrides, sources = bu_detector.get_item_override_sources( label, backend )
    settings = get_label_settings( label, backend )
    fields = {}
    for key, value in settings.items():
        source = sources.get( key ) or _default_setting_for( key ) or bu_detector.SOURCE_BUILTIN
        chain = _label_chain( label, key, backend, variant, window, reverse_censor )
        winners = [ layer for layer in chain if layer['source'] == source and 'inactive' not in layer ]
        if winners:
            winners[-1]['wins'] = True
        fields[key] = { 'value': value, 'source': source, 'chain': chain }
    for key, parent in bu_settings.LABEL_REQUIRES.items():
        if key in fields and not settings.get( parent ):
            fields[key]['off'] = parent
    return fields


def label_views( backend_name=None ):
    """
    Every label: whether this run renders it, whether it is a reverse
    window, whether the selected backend can detect it at all, and
    label_view. A label the backend cannot detect says why, as a code
    with parameters ('unavailable').
    """
    name = backend_name or bu_detector.selected_backend_name()
    rendered = set( rendered_labels() )
    supported = bu_detector.supported_labels( name )
    views = []
    for label in bu_settings.label_names():
        view = { 'label': label, 'rendered': label in rendered,
                 'window': bu_reverse.window_entry( label ) is not None,
                 'available': label in supported,
                 'fields': label_view( label, name ) }
        if label not in supported:
            view['unavailable'] = { 'code': 'label.not_in_model', 'params': { 'backend': name } }
        views.append( view )
    return views


def _geometry_action_default():
    """DEFAULT_GEOMETRY_ACTION, imported late to avoid the import cycle."""
    import betautils_track as bu_track  # deferred: bu_track imports this module
    return bu_track.DEFAULT_GEOMETRY_ACTION


def rendered_labels():
    """
    The labels this run renders: reverse mode's windows and censored
    labels while it is on (censored labels default to items_to_censor),
    otherwise items_to_censor.
    """
    if bu_reverse.enabled():
        return bu_reverse.rendered_labels()
    return list( betaconfig.items_to_censor )


CENSOR_SOMETHING_HINT = ( "python tools/setup/local_config.py set "
                          "\"items_to_censor=['exposed_vulva', 'exposed_breast']\"" )


def nothing_to_render_reason():
    """
    Why this run would leave every frame as it is, or None when it would not.

    betaconfig.py ships items_to_censor = [] and reverse mode with no
    windows, so a fresh install censors nothing until the user says what.
    Run like that, detection and rendering take their full time and
    produce a copy of the input, which reads as a broken censor rather
    than a missing setting.
    """
    if bu_reverse.enabled():
        if ( bu_reverse.rendered_labels() or bu_reverse.trigger_labels()
                or bu_reverse.settings()['empty_frame'] == 'obscure' ):
            return None
        return ( "reverse mode is on with no windows, no censored labels and no trigger labels, "
                 "and empty_frame is 'passthrough', so no frame would change" )
    if betaconfig.items_to_censor:
        return None
    return "items_to_censor is empty, so nothing would be censored"


def exit_if_nothing_to_render( logger ):
    """Stop before any detection when the run would change no frame (exit 0: nothing failed)."""
    reason = nothing_to_render_reason()
    if reason is None:
        return
    logger.warning( "%s. Nothing was processed. Choose what to censor in betaconfig_local.py, for "
                    "example: %s (CONFIG_REFERENCE.md, \"Example local config\")"%(
                        reason, CENSOR_SOMETHING_HINT ) )
    raise SystemExit( 0 )


def get_parts_to_blur( backend_name=None ):
    """
    The per-label settings table the render pipeline actually uses.

    One entry per rendered label (rendered_labels: items_to_censor, or
    reverse mode's windows), resolved against that label's
    item_overrides with betaconfig's defaults filling in the rest.

    When betaconfig.debug_mode's bit 0 is set, every known class also
    gets a synthetic 'debug' entry, so detection quality can be eyeballed
    as labelled boxes without touching items_to_censor.

    Memoised per backend. Call invalidate_config_caches() after mutating
    betaconfig.

    Args:
        backend_name: Backend whose per-label overrides apply. Defaults
            to the selected one.

    Returns:
        {label: settings dict}. Never mutate the returned dict; it is
        shared.
    """
    cache_key = backend_name or bu_detector.selected_backend_name()
    with _parts_cache_lock:
        cached = _parts_cache.get( cache_key )
    if cached is not None:
        return cached

    parts_to_blur = {}
    for item in rendered_labels():
        parts_to_blur[item] = get_label_settings( item, backend_name )

    if betaconfig.debug_mode & 1:
        for class_name, debug_color in betaconst.classes.items():
            parts_to_blur[class_name] = {
                'min_prob': 0.5,
                'min_prob_continue': None,
                'width_area_safety': 0,
                'height_area_safety': 0,
                'time_safety': 0.4,
                'censor_style': { 'type': 'debug', 'color': debug_color },
                'censor_shape': 'box',
                'position_smoothing': betaconfig.default_position_smoothing,
                'interpolation_enabled': False,
                'interpolation_max_gap': betaconfig.default_interpolation_max_gap,
                'track_max_gap': None,
                'match_distance_multiplier': 1.0,
                'min_track_hits': 1,
                'paired_style': False,
                'paired_style_max_distance': betaconfig.default_paired_style_max_distance,
                'paired_style_tiebreak_margin': betaconfig.default_paired_style_tiebreak_margin,
                'min_area_fraction': None,
                'max_area_fraction': None,
                'min_aspect_ratio': None,
                'max_aspect_ratio': None,
            }

    with _parts_cache_lock:
        _parts_cache[cache_key] = parts_to_blur
    return parts_to_blur


def verify_input_delete_probability():
    """
    If betaconfig.input_delete_probability is non-zero, interactively
    prompt the user to type an exact confirmation phrase before
    continuing, since this setting can permanently delete input files.

    Side effects:
        Prints warnings and reads one line from stdin when
        input_delete_probability is non-zero. Calls quit() (exits the
        process) if the user doesn't type the exact required phrase.
        No-op entirely when input_delete_probability is 0.
    """
    delete_probability = betaconfig.input_delete_probability
    if delete_probability == 0:
        return()

    print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
    print( 'You have configured BetaSuite to delete input files with a %.2f%% chance.'%(100*delete_probability))
    print( 'If you are SURE this is what you want, enter "DELETE MY FILES" at the prompt' )
    print( 'All upper-case, without the quotes.' )
    print( 'Are you sure?' )
    confirmation = input()

    if confirmation == 'DELETE MY FILES':
        print( "Okay!  Proceeding with %.2f%% chance of deleting each input file."%(100*delete_probability) )
        return()
    else:
        print( 'You did not enter "DELETE MY FILES", all upper-case, with no quotes.')
        print( 'Aborting program.' )
        quit()


def delete_file_with_probability( delete_path, check_path ):
    """
    Randomly delete delete_path, at betaconfig.input_delete_probability
    odds, but only once a censored counterpart has actually been
    produced.

    Args:
        delete_path: The original (uncensored) input file that may be
            deleted.
        check_path: Path to the censored output that must already exist
            (and be non-trivially sized) before deletion is even
            considered - guards against deleting the only copy of a
            file whose censoring failed or never ran.

    Returns:
        A human-readable status suffix string describing what happened
        (deleted, not deleted, or skipped because the censored version
        wasn't found), suitable for appending to a log/print line. Never
        raises for the "nothing to report" case (probability 0).
    """
    delete_probability = betaconfig.input_delete_probability
    if delete_probability == 0:
        return('')

    if not os.path.exists( check_path ) or os.path.getsize( check_path ) < 1000:
        return( ' (Original NOT DELETED - censored version NOT FOUND)' )

    roll = random.random()
    if roll < delete_probability:
        os.remove( delete_path )
        return( ' (Original DELETED!!!!)' )
    else:
        return( ' (Original not deleted)' )


# --- Shared vocabulary for censor-style validation --------------------------

def _valid_class_names():
    """
    Returns:
        The set of every detection class name betaconst.py defines
        (regardless of whether it's actually in items_to_censor).
    """
    return( set( betaconst.classes.keys() ) )


# The registry's vocabulary, under the names the rest of the code uses.
VALID_CENSOR_STYLES = set( bu_settings.CENSOR_STYLE_TYPES )
VALID_CENSOR_SHAPES = set( bu_settings.CENSOR_SHAPES )
VALID_BLUR_METHODS = set( bu_settings.BLUR_METHODS )
VALID_PIXEL_PATTERNS = set( bu_settings.PIXEL_PATTERNS )
# same vocabulary as censor_overlap_strategy's values (betaconfig.py) - a
# style dict's own 'merge' key overrides the type-level default there for
# just this one style variant. See collapse_boxes_for_style in
# betautils_censor.py.
VALID_MERGE_STRATEGIES = set( bu_settings.MERGE_STRATEGIES )
VALID_ENCODE_PRESETS = { 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow' }
NVENC_PRESETS = frozenset( 'p%d'%( number, ) for number in range( 1, 8 ) )

# keys every style dict may carry, plus keys specific to each 'type'
# 'shape' is optional and overrides the item's default censor_shape for
# just this one style variant (e.g. exposed_breast's 'bar' entries wanting
# a rectangle while its 'pixel' entry stays a circle) - see smooth_boxes in
# betatv.py for where this actually gets applied.
# 'merge' is optional and overrides censor_overlap_strategy's type-level
# default for just this one style variant (e.g. exposed_breast's two 'bar'
# entries - one spans, one doesn't - despite both being type 'bar').
# 'width_area_safety'/'height_area_safety' are optional and override the
# item's own width_area_safety/height_area_safety (item_overrides, or
# default_area_safety) for just this one style variant - so a randomized
# style list can tune how generously each variant pads its box the same
# way it already tunes strength/feather. Applied in betatv.py's
# smooth_boxes once a track's style is resolved; see
# betautils_censor.compute_safe_geometry and README.md.
COMMON_STYLE_KEYS = { 'type', 'feather', 'weight', 'shape', 'merge', 'width_area_safety', 'height_area_safety' }
TYPE_SPECIFIC_KEYS = {
    'blur':    { 'method', 'strength' },
    'pixel':   { 'pattern', 'strength' },
    'bar':     { 'color', 'span_extend', 'thickness' },
    'sticker': { 'dir', 'images', 'scale' },
    'debug':   { 'color' },
}


# --- Per-style-type validation -----------------------------------------

def _check_style_common_fields( style, style_type, where, errors ):
    """
    The keys every censor style dict may carry, whatever its type
    (feather, weight, shape, merge, width/height_area_safety), against
    bu_settings.STYLE_COMMON_FIELDS; unknown keys for its type.

    Args:
        style: The style dict being validated.
        style_type: style['type'], already confirmed valid by the caller.
        where: Human-readable path describing style's location in
            betaconfig.py, used to prefix error messages.
        errors: Shared list of error strings to append to.
    """
    allowed_keys = COMMON_STYLE_KEYS | TYPE_SPECIFIC_KEYS.get( style_type, set() )
    unknown_keys = set(style.keys()) - allowed_keys
    if unknown_keys:
        errors.append( bu_settings.rule( 'style.unknown_keys', "%s has unrecognized key(s) %s for type '%s', likely a typo (valid keys: %s)"%(where, sorted(unknown_keys), style_type, sorted(allowed_keys)) ) )
    _check_fields( style, bu_settings.STYLE_COMMON_FIELDS, where, errors, skip=( 'type', ) )


def _check_type_fields( style, style_type, where, errors ):
    """The keys one style type adds, against bu_settings.STYLE_TYPE_FIELDS."""
    _check_fields( style, bu_settings.STYLE_TYPE_FIELDS.get( style_type, {} ), where, errors )
    if style_type == 'sticker' and 'dir' not in style and 'images' not in style:
        errors.append( bu_settings.rule( 'style.sticker_needs_images', "%s must set 'dir' and/or 'images' so there's something to composite"%(where) ) )


def _check_one_style_dict( style, where, errors ):
    """
    Validate a single censor style dict (one entry of a censor_style
    list, or a censor_style set to a single dict directly).

    Args:
        style: The value to validate - expected to be a dict like
            {'type': 'blur', ...}.
        where: Human-readable path describing style's location in
            betaconfig.py, used to prefix error messages.
        errors: Shared list of error strings to append to.
    """
    if not isinstance( style, dict ):
        errors.append( bu_settings.rule( 'style.not_a_mapping', "%s must be a dict like {'type': 'blur', ...}, got %r"%(where, style) ) )
        return
    if 'type' not in style:
        errors.append( bu_settings.rule( 'style.missing_type', "%s is missing required key 'type'"%(where) ) )
        return
    style_type = style['type']
    if style_type not in VALID_CENSOR_STYLES:
        errors.append( bu_settings.rule( 'style.unknown_type', "%s has unknown censor style type '%s' (valid: %s)"%(where, style_type, sorted(VALID_CENSOR_STYLES)) ) )
        return

    _check_style_common_fields( style, style_type, where, errors )

    _check_type_fields( style, style_type, where, errors )


def _check_censor_style( style, where, errors ):
    """
    Validate a censor_style value, which may be either a single style
    dict or a list of style dicts (one is picked at random per tracked
    detection, weighted by 'weight').

    Args:
        style: The censor_style value to validate.
        where: Human-readable path describing style's location in
            betaconfig.py, used to prefix error messages.
        errors: Shared list of error strings to append to.
    """
    if isinstance( style, list ):
        if not style:
            errors.append( bu_settings.rule( 'style.empty_list', "%s is an empty list - needs at least one style dict"%(where) ) )
        for idx, one_style in enumerate( style ):
            _check_one_style_dict( one_style, "%s[%d]"%(where, idx), errors )
    else:
        _check_one_style_dict( style, where, errors )


# --- Reverse mode -------------------------------------------------------------

REVERSE_KEYS = frozenset( bu_reverse.DEFAULTS )
REVERSE_TRIGGER_KEYS = frozenset( bu_reverse.DEFAULTS['trigger'] )
REVERSE_CENSOR_KEYS = frozenset( bu_reverse.DEFAULTS['censor'] )
REVERSE_PERSON_WINDOW_KEYS = frozenset( bu_reverse.DEFAULTS['person_windows'] )
REVERSE_FRAME_STYLE_TYPES = frozenset( { 'blur', 'pixel', 'bar' } )
# Measured with betabench reverse on a 12-thread laptop (3840x2160,
# one core): gaussian 4,765 ms and hex 930 ms per frame, against 78 ms
# for triple_box and 11 ms for mosaic. Whole-frame, those two would make
# every triggered frame cost seconds. With blur_edge_margin on, the
# gaussian method also skips the downscale approximation, which is why
# it is that slow.
REVERSE_FRAME_STYLES_TOO_SLOW = frozenset( { ( 'blur', 'gaussian' ), ( 'pixel', 'hex' ) } )
SHOW_STYLE_KEYS = frozenset( { 'type', 'feather', 'weight', 'shape',
                               'width_area_safety', 'height_area_safety' } )


def _check_window_style( style, where, errors ):
    """
    A reverse window's censor_style: {'type': 'show'} (the original
    pixels) with its optional keys, alone or in a weighted list. A window
    is uncensored by definition, so any other style is an error.
    """
    if isinstance( style, list ):
        if not style:
            errors.append( bu_settings.rule( 'reverse_window_style.empty_list', "%s is an empty list - needs at least one style dict"%( where, ) ) )
        entries = [ ( "%s[%d]"%( where, index ), one ) for index, one in enumerate( style ) ]
    else:
        entries = [ ( where, style ) ]
    for entry_where, one in entries:
        if isinstance( one, dict ) and one.get( 'type' ) == 'show':
            unknown = set( one ) - SHOW_STYLE_KEYS
            if unknown:
                errors.append( bu_settings.rule( 'reverse_window_style.unknown_keys', "%s has unrecognized key(s) %s for type 'show' (valid keys: %s)"%(
                    entry_where, sorted( unknown ), sorted( SHOW_STYLE_KEYS ) ) ) )
            known = { key: value for key, value in one.items() if key in SHOW_STYLE_KEYS }
            _check_style_common_fields( known, 'show', entry_where, errors )
        else:
            errors.append( bu_settings.rule( 'reverse_window_style.not_show', "%s is %r, but a window is always uncensored: its style can only be "
                           "{'type': 'show', ...}. To censor this label in reverse mode, add it to "
                           "reverse['censor'] instead"%( entry_where, one ) ) )


def _check_reverse_frame_style( frame_style, errors ):
    where = "reverse['frame_style']"
    if not isinstance( frame_style, dict ):
        errors.append( bu_settings.rule( 'reverse_frame_style.not_a_mapping', "%s must be one style dict like {'type': 'blur', ...}, got %r"%(
            where, frame_style ) ) )
        return
    style_type = frame_style.get( 'type' )
    if style_type not in REVERSE_FRAME_STYLE_TYPES:
        errors.append( bu_settings.rule( 'reverse_frame_style.bad_type', "%s['type'] = %r can't be a frame style (valid: %s); stickers, 'show' and "
                       "'debug' only make sense on a window"%(
                           where, style_type, sorted( REVERSE_FRAME_STYLE_TYPES ) ) ) )
        return
    variant = frame_style.get( 'method' ) or frame_style.get( 'pattern' )
    if ( style_type, variant ) in REVERSE_FRAME_STYLES_TOO_SLOW:
        errors.append( bu_settings.rule( 'reverse_frame_style.too_slow', "%s %s/%s is too slow over a whole frame (measured 0.9 to 4.8 s per 4K frame "
                       "on one core); use blur 'triple_box' or 'box', or pixel 'mosaic'"%(
                           where, style_type, variant ) ) )
    _check_fields( frame_style, bu_settings.FRAME_STYLE_EXTRA_FIELDS, where, errors )
    _check_one_style_dict( { key: value for key, value in frame_style.items() if key != 'dim' },
                           where, errors )


def _check_reverse_trigger( trigger, valid_classes, errors ):
    where = "reverse['trigger']"
    if not isinstance( trigger, dict ):
        errors.append( bu_settings.rule( 'reverse_trigger.not_a_mapping', "%s must be a dict, got %r"%( where, trigger ) ) )
        return
    unknown = set( trigger ) - REVERSE_TRIGGER_KEYS
    if unknown:
        errors.append( bu_settings.rule( 'reverse_trigger.unknown_keys', "%s has unrecognized key(s) %s (valid: %s)"%(
            where, sorted( unknown ), sorted( REVERSE_TRIGGER_KEYS ) ) ) )
    labels = trigger.get( 'labels' )
    if isinstance( labels, ( list, tuple ) ) and not labels:
        errors.append( bu_settings.rule( 'reverse_trigger.empty_list', "%s['labels'] must be None (every censored label) or a non-empty list of "
                       "class names, got %r"%( where, labels ) ) )
    if 'min_prob' in trigger:
        min_prob = trigger['min_prob']
        global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
        if ( not isinstance( min_prob, ( int, float ) ) or isinstance( min_prob, bool )
                or not ( 0 <= min_prob < 1 ) ):
            pass   # the registry reports it
        elif isinstance( global_min_prob, ( int, float ) ) and min_prob < global_min_prob:
            errors.append( bu_settings.rule( 'reverse_trigger.below_floor', "%s['min_prob'] (%r) is below global_min_prob (%r). Detections under the "
                           "floor are never stored, so the trigger would silently use the floor; set "
                           "it at or above global_min_prob"%( where, min_prob, global_min_prob ) ) )


def _check_reverse_censor( censor, windows, valid_classes, errors ):
    where = "reverse['censor']"
    if not isinstance( censor, dict ):
        errors.append( bu_settings.rule( 'reverse_censor.not_a_mapping', "%s must be a dict like {'labels': None, 'overrides': {}}, got %r"%(
            where, censor ) ) )
        return
    unknown = set( censor ) - REVERSE_CENSOR_KEYS
    if unknown:
        errors.append( bu_settings.rule( 'reverse_censor.unknown_keys', "%s has unrecognized key(s) %s (valid: %s)"%(
            where, sorted( unknown ), sorted( REVERSE_CENSOR_KEYS ) ) ) )
    labels = censor.get( 'labels' )
    if labels is None:
        labels = list( getattr( betaconfig, 'items_to_censor', [] ) )
    elif not isinstance( labels, ( list, tuple ) ):
        labels = []   # the registry reports it
    if isinstance( windows, dict ):
        both = sorted( set( labels ) & set( windows ) )
        if both:
            errors.append( bu_settings.rule( 'reverse_censor.window_and_censored', "%s %s: a label can't be both a window (shown) and censored. Remove it "
                           "from reverse['windows'] or from the censored labels%s"%(
                               'label' if len( both ) == 1 else 'labels', both,
                               " (None means items_to_censor)" if censor.get( 'labels' ) is None else '' ) ) )
    overrides = censor.get( 'overrides', {} )
    if not isinstance( overrides, dict ):
        errors.append( bu_settings.rule( 'reverse_censor.not_a_mapping', "%s['overrides'] must be a dict of {label: settings}, got %r"%( where, overrides ) ) )
        return
    allowed = bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS | bu_reverse.LOOK_KEYS
    for label, entry in overrides.items():
        entry_where = "%s['overrides']['%s']"%( where, label )
        if label not in labels:
            errors.append( bu_settings.rule( 'reverse_censor.override_not_censored', "%s: '%s' is not a censored label, so this override would never apply "
                           "(censored: %s)"%( entry_where, label, sorted( labels ) ) ) )
            continue
        if not isinstance( entry, dict ):
            errors.append( bu_settings.rule( 'reverse_censor.not_a_mapping', "%s must be a dict, got %r"%( entry_where, entry ) ) )
            continue
        unknown = set( entry ) - allowed
        if unknown:
            errors.append( bu_settings.rule( 'reverse_censor.unknown_keys', "%s has unrecognized key(s) %s (valid: censor_style, censor_shape and "
                           "the backend-tunable keys %s)"%(
                               entry_where, sorted( unknown ),
                               sorted( bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS ) ) ) )
        _check_item_override_values( { key: value for key, value in entry.items() if key in allowed },
                                     entry_where, errors )


def _check_reverse_person_windows( person_windows, valid_classes, errors ):
    where = "reverse['person_windows']"
    if not isinstance( person_windows, dict ):
        errors.append( bu_settings.rule( 'person_windows.not_a_mapping', "%s must be a dict, got %r"%( where, person_windows ) ) )
        return
    unknown = set( person_windows ) - REVERSE_PERSON_WINDOW_KEYS
    if unknown:
        errors.append( bu_settings.rule( 'person_windows.unknown_keys', "%s has unrecognized key(s) %s (valid: %s)"%(
            where, sorted( unknown ), sorted( REVERSE_PERSON_WINDOW_KEYS ) ) ) )
    label_sets = {}
    for key in ( 'male_labels', 'female_labels' ):
        labels = person_windows.get( key, bu_reverse.DEFAULTS['person_windows'][key] )
        if key not in person_windows:
            label_sets[key] = set( labels )
            continue
        if not isinstance( labels, ( list, tuple ) ):
            continue   # the registry reports it
        if not labels:
            errors.append( bu_settings.rule( 'person_windows.empty_list', "%s['%s'] must be a non-empty list of class names, got %r"%( where, key, labels ) ) )
            continue
        label_sets[key] = set( labels )
    both = label_sets.get( 'male_labels', set() ) & label_sets.get( 'female_labels', set() )
    if both:
        errors.append( bu_settings.rule( 'person_windows.male_and_female', "%s: %s can't be male and female evidence at once"%( where, sorted( both ) ) ) )
    # key: (low, low allowed, high, high allowed); None = no bound
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    floor = person_windows.get( 'evidence_min_prob' )
    if ( isinstance( floor, ( int, float ) ) and isinstance( global_min_prob, ( int, float ) )
            and floor < global_min_prob ):
        errors.append( bu_settings.rule( 'person_windows.below_floor', "%s['evidence_min_prob'] (%r) is below global_min_prob (%r): detections under the "
                       "floor are never stored"%( where, floor, global_min_prob ) ) )


def _validate_person_backend( errors ):
    """
    person_backend, when there is one, and (while reverse mode's person
    windows are on) that the selected model is implemented and on disk.
    """
    import betautils_person as bu_person   # deferred: keeps cv2 out of this import graph
    block = getattr( betaconfig, 'person_backend', None )
    if block is not None:
        if not isinstance( block, dict ):
            errors.append( bu_settings.rule( 'person_backend.not_a_mapping', "person_backend must be a dict, got %r"%( block, ) ) )
            return
        for key, entry in block.items():
            if key == 'selected':
                continue
            if key not in bu_person.registered_backend_names():
                errors.append( bu_settings.rule( 'person_backend.unknown_keys', "person_backend has unknown backend %r (valid: %s)"%(
                    key, bu_person.registered_backend_names() ) ) )
            elif not isinstance( entry, dict ):
                errors.append( bu_settings.rule( 'person_backend.not_a_mapping', "person_backend[%r] must be a dict, got %r"%( key, entry ) ) )
    selected = bu_person.selected_backend_name()
    try:
        segmenter = bu_person.get_segmenter( selected )
    except ValueError as err:
        errors.append( bu_settings.rule( 'person_backend.unknown_variant', str( err ) ) )
        return
    if not bu_reverse.person_windows_enabled():
        return
    segmenter.validate_config( bu_person.get_backend_config( selected ), errors )
    if not bu_person.is_implemented( selected ):
        return
    try:
        model_path = bu_paths.resolve( bu_person.resolve_config( selected )['model_path'] )
    except ValueError:
        return
    if not os.path.exists( model_path ):
        errors.append( bu_settings.rule( 'person_backend.model_missing', "reverse['person_windows'] is on but the person model %s is missing. Run setup "
                       "again (tools/setup/fetch_models.py exports it), or set "
                       "reverse['person_windows']['enabled'] = False"%( model_path, ) ) )


def _check_reverse_windows( windows, valid_classes, errors ):
    where_base = "reverse['windows']"
    if not isinstance( windows, dict ):
        errors.append( bu_settings.rule( 'reverse_windows.not_a_mapping', "%s must be a dict of {label: window settings}, got %r"%( where_base, windows ) ) )
        return
    allowed = bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS | bu_reverse.LOOK_KEYS
    for label, entry in windows.items():
        where = "%s['%s']"%( where_base, label )
        if label not in valid_classes:
            errors.append( bu_settings.rule( 'reverse_windows.unknown_label', "%s: '%s' is not a known class"%( where_base, label ) ) )
            continue
        if not isinstance( entry, dict ):
            errors.append( bu_settings.rule( 'reverse_windows.not_a_mapping', "%s must be a dict ({} inherits every tuning key), got %r"%( where, entry ) ) )
            continue
        unknown = set( entry ) - allowed
        if unknown:
            errors.append( bu_settings.rule( 'reverse_windows.unknown_keys', "%s has unrecognized key(s) %s (valid: censor_style (show only), "
                           "censor_shape and the backend-tunable keys %s)"%(
                               where, sorted( unknown ),
                               sorted( bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS ) ) ) )
        if 'censor_style' in entry:
            _check_window_style( entry['censor_style'], "%s['censor_style']"%( where, ), errors )
        tuning = { key: value for key, value in entry.items() if key not in bu_reverse.LOOK_KEYS }
        if 'censor_shape' in entry:
            tuning['censor_shape'] = entry['censor_shape']
        _check_item_override_values( tuning, where, errors )


def _validate_reverse( errors, valid_classes ):
    """
    The reverse block, when there is one. Checked whether or not it is
    enabled, so a mistake surfaces when it is written rather than on the
    first reverse run.
    """
    block = getattr( betaconfig, 'reverse', None )
    if block is None:
        return
    if not isinstance( block, dict ):
        errors.append( bu_settings.rule( 'reverse.not_a_mapping', "reverse must be a dict, got %r"%( block, ) ) )
        return
    unknown = set( block ) - REVERSE_KEYS
    if unknown:
        errors.append( bu_settings.rule( 'reverse.unknown_keys', "reverse has unrecognized key(s) %s (valid: %s)"%(
            sorted( unknown ), sorted( REVERSE_KEYS ) ) ) )
    if 'frame_style' in block:
        _check_reverse_frame_style( block['frame_style'], errors )
    if 'trigger' in block:
        _check_reverse_trigger( block['trigger'], valid_classes, errors )
    if 'windows' in block:
        _check_reverse_windows( block['windows'], valid_classes, errors )
    if 'censor' in block:
        _check_reverse_censor( block['censor'], block.get( 'windows', {} ), valid_classes, errors )
    if 'person_windows' in block:
        _check_reverse_person_windows( block['person_windows'], valid_classes, errors )


def _warn_reverse( warnings ):
    """Reverse mode set up so that it reveals or does nothing."""
    if not bu_reverse.enabled():
        return
    if not bu_reverse.trigger_labels():
        warnings.append( "reverse mode is on but nothing can trigger it (no censored labels and "
                         "trigger labels None): every frame passes through clear" )
    if not bu_reverse.window_labels():
        warnings.append( "reverse mode is on with no windows: a triggered frame is obscured "
                         "everywhere but its censored labels" )
    for label in bu_reverse.censor_labels():
        style = bu_detector.get_item_overrides( label ).get( 'censor_style' )
        for one in ( style if isinstance( style, list ) else [ style ] ):
            if not isinstance( one, dict ):
                continue
            see_through = ( one.get( 'type' ) == 'sticker'
                            or ( one.get( 'type' ) == 'bar' and one.get( 'thickness', 1.0 ) < 1.0 ) )
            if see_through:
                warnings.append( "reverse mode's censored %s has a %s style: its see-through parts "
                                 "show the ORIGINAL pixels, not the frame style (known limitation). "
                                 "If that reveals too much, give it a reverse-only censor_style "
                                 "without one in reverse['censor']['overrides']['%s']"%(
                                     label, one['type'], label ) )
                break


# --- Top-level betaconfig.py section validators -----------------------------

def _validate_items_to_censor( errors, valid_classes ):
    """items_to_censor is defined. Its type and labels are registry checks."""
    if getattr( betaconfig, 'items_to_censor', None ) is None:
        errors.append( bu_settings.rule( 'items_to_censor.missing', "items_to_censor is not defined" ) )


def _validate_class_suppression( errors, valid_classes ):
    """
    Check every registered backend's resolved class_suppression ruleset
    (see betautils_detector.get_class_suppression) - not just the
    currently selected backend's, so a typo in an unselected backend's
    ruleset is still caught before it's ever relied on (e.g. switching
    betaconfig.detector_backend['selected'] later shouldn't be the first
    time a bad rule there is noticed) - same reasoning and structure as
    _validate_nn_batch_size above. For each backend: every key is a real
    class, every rule (single dict or list of dicts) has a valid
    'suppressed_by' pointing at a real class with no duplicate
    suppressors, and both 'margin' and 'min_iou' are present and are
    numbers.

    margin is REQUIRED, not optional: apply_class_suppression (betatv.py)
    treats a missing margin as 0.0, the loosest possible gate (fires
    whenever the suppressing label's score is even just >= the
    suppressed label's) - not a neutral "no margin check" default. A
    rule author who omits margin is silently getting the loosest
    setting, not skipping a step, so this is caught here rather than
    left to that implicit behavior. If real data genuinely supports
    margin=0.0 for a pair (see derive_suppression_rules.py's
    propose_margin for when that's the honest answer, e.g. the
    suppressing label rarely being more confident), write it explicitly
    - that's a real, reviewed value, not the same thing as leaving the
    key out.
    """
    known_rule_keys = { 'suppressed_by', 'margin', 'min_iou' }

    if getattr( betaconfig, 'class_suppression', None ) is not None:
        errors.append( bu_settings.rule( 'class_suppression.removed_setting', "betaconfig.class_suppression is no longer read - label "
            "vocabularies differ between models, so a shared ruleset can name classes a backend has "
            "never heard of. Move it into detector_backend[<name>]['class_suppression'], or delete "
            "it. Leaving it here would silently do nothing." ) )

    for backend_name in bu_detector.registered_backend_names():
        class_suppression = bu_detector.get_class_suppression( backend_name )
        for label, rules in class_suppression.items():
            is_list = isinstance( rules, list )
            where_base = "detector_backend[%r]['class_suppression']['%s']"%(backend_name, label)
            if label not in valid_classes:
                errors.append( bu_settings.rule( 'class_suppression.unknown_label', "%s: '%s' is not a known class (valid: %s)"%(
                        "detector_backend[%r]['class_suppression']"%(backend_name), label, sorted(valid_classes)) ) )

            rule_list = rules if is_list else [ rules ]
            seen_suppressors = []
            for idx, rule in enumerate( rule_list ):
                where = "%s[%d]"%(where_base, idx) if is_list else where_base
                if not isinstance( rule, dict ):
                    errors.append( bu_settings.rule( 'class_suppression.not_a_mapping', "%s must be a dict, got %r"%(where, rule) ) )
                    continue
                unknown_keys = set(rule.keys()) - known_rule_keys
                if unknown_keys:
                    errors.append( bu_settings.rule( 'class_suppression.unknown_keys', "%s has unrecognized key(s) %s, likely a typo (valid keys: %s)"%(where, sorted(unknown_keys), sorted(known_rule_keys)) ) )
                if 'suppressed_by' not in rule:
                    errors.append( bu_settings.rule( 'class_suppression.missing_suppressed_by', "%s is missing required key 'suppressed_by'"%(where) ) )
                elif rule['suppressed_by'] not in valid_classes:
                    errors.append( bu_settings.rule( 'class_suppression.unknown_label', "%s['suppressed_by'] = '%s' is not a known class (valid: %s)"%(where, rule['suppressed_by'], sorted(valid_classes)) ) )
                else:
                    if rule['suppressed_by'] in seen_suppressors:
                        errors.append( bu_settings.rule( 'class_suppression.duplicate_suppressor', "%s has more than one rule with suppressed_by='%s' - only one will ever match, merge them"%(where_base, rule['suppressed_by']) ) )
                    seen_suppressors.append( rule['suppressed_by'] )
                if 'margin' not in rule:
                    errors.append( bu_settings.rule( 'class_suppression.missing_margin', "%s is missing required key 'margin' - a missing margin silently defaults to "
                        "0.0 (the loosest possible gate) in apply_class_suppression, not a neutral 'off' value, "
                        "so it must be set explicitly (see derive_suppression_rules.py's propose_margin if you "
                        "need to derive one from real data)"%(where) ) )
                if isinstance( rule.get( 'margin' ), (int, float) ) and rule['margin'] < 0:
                    # apply_class_suppression's check is 'other[score] - raw[score] >= margin' - a
                    # negative margin lets the suppressing label be LESS confident than the label
                    # it's overriding and still win, inverting the entire "require better evidence
                    # to override a real detection" premise the margin gate exists for. No known
                    # legitimate use case - always an error,
                    # unlike margin=0.0, which is only a warning (see _validate_class_suppression_zero_margin).
                    errors.append( bu_settings.rule( 'class_suppression.negative_margin', "%s['margin'] = %r is negative - the suppressing label would be allowed to be "
                        "LESS confident than the label it's overriding and still suppress it, which defeats the "
                        "point of the margin gate. Use 0.0 if you genuinely want IoU alone to gate this rule "
                        "(that's a warning, not an error), never a negative number."%(where, rule['margin']) ) )
                if 'min_iou' not in rule:
                    errors.append( bu_settings.rule( 'class_suppression.missing_min_iou', "%s is missing required key 'min_iou'"%(where) ) )
                margin_is_negative = isinstance( rule.get( 'margin' ), (int, float) ) and rule['margin'] < 0
                _check_fields( rule, bu_settings.RULE_FIELDS['suppression'], where, errors,
                               skip=( 'suppressed_by', ) + ( ( 'margin', ) if margin_is_negative else () ) )

        where = "detector_backend[%r]['class_suppression_refused']"%( backend_name, )
        refused = bu_detector.resolve_backend_setting(
            'class_suppression_refused', backend_name, None )
        if refused is None:
            continue
        if not isinstance( refused, ( list, tuple ) ):
            errors.append( bu_settings.rule( 'class_suppression.not_a_mapping', "%s must be a list of 'label<-suppressor' strings, got %r"%( where, refused ) ) )
            continue
        for pair in refused:
            # Without an arrow the suppressor comes out empty, so it fails the class check.
            label, _sep, suppressor = pair.partition( '<-' ) if isinstance( pair, str ) else ( '', '', '' )
            if label not in valid_classes or suppressor not in valid_classes:
                errors.append( bu_settings.rule( 'class_suppression.bad_refused_pair', "%s: %r is not 'label<-suppressor' with two known classes"%( where, pair ) ) )


def _validate_class_promotion( errors, valid_classes ):
    """
    Check every registered backend's class_promotion ruleset.

    Unknown keys are errors, not warnings, because every key in a rule
    is a gate: a misspelled 'min_prob' silently becomes "no floor", and a
    misspelled 'requires' silently becomes "no evidence needed", which
    promotes every detection of the source label. Both fail open, in the
    direction of censoring things the author never meant to.

    A target that is not rendered (items_to_censor, or reverse mode's
    windows and censored labels while it is on) is an error only while
    a source label IS rendered: the rule would relabel that label's
    boxes into one nothing renders, which DELETES their censor. With
    neither rendered nothing is lost, and the check used to stop every
    run of a config that censors nothing (items_to_censor = [], as
    betaconfig.py ships), because the shipped covered_vulva promotion
    targets a label that config does not render.
    """
    import betautils_track as bu_track  # deferred: bu_track imports this module
    censored = set( rendered_labels() )
    censored_name = ( "reverse mode's windows and censored labels" if bu_reverse.enabled()
                      else 'items_to_censor' )

    for backend_name in bu_detector.registered_backend_names():
        promotion = bu_detector.get_class_promotion( backend_name )
        if not isinstance( promotion, dict ):
            errors.append( bu_settings.rule( 'class_promotion.not_a_mapping', "detector_backend[%r]['class_promotion'] must be a dict, got %r"%(
                backend_name, promotion ) ) )
            continue
        for target, rules in promotion.items():
            where_base = "detector_backend[%r]['class_promotion']['%s']"%( backend_name, target )
            if target not in valid_classes:
                errors.append( bu_settings.rule( 'class_promotion.unknown_label', "%s: '%s' is not a known class"%( where_base, target ) ) )
            rule_list = rules if isinstance( rules, list ) else [ rules ]
            if target in valid_classes and target not in censored:
                losing = sorted( { rule.get( 'from' ) for rule in rule_list
                                   if isinstance( rule, dict ) and rule.get( 'from' ) in censored } )
                if losing:
                    errors.append( bu_settings.rule( 'class_promotion.target_not_rendered', "%s: '%s' is not in %s, so %s boxes promoted to it would lose "
                                   "their censor. Render '%s' too, or remove the rule"%(
                                       where_base, target, censored_name,
                                       ', '.join( repr( label ) for label in losing ), target ) ) )
            for index, rule in enumerate( rule_list ):
                where = "%s[%d]"%( where_base, index ) if isinstance( rules, list ) else where_base
                if not isinstance( rule, dict ):
                    errors.append( bu_settings.rule( 'class_promotion.not_a_mapping', "%s must be a dict, got %r"%( where, rule ) ) )
                    continue
                unknown = set( rule ) - bu_track.PROMOTION_RULE_KEYS
                if unknown:
                    errors.append( bu_settings.rule( 'class_promotion.unknown_keys', "%s has unrecognized key(s) %s (valid: %s)"%(
                        where, sorted( unknown ), sorted( bu_track.PROMOTION_RULE_KEYS ) ) ) )
                source = rule.get( 'from' )
                if source is None:
                    errors.append( bu_settings.rule( 'class_promotion.missing_from', "%s is missing required key 'from'"%( where ) ) )
                elif source not in valid_classes:
                    errors.append( bu_settings.rule( 'class_promotion.unknown_label', "%s: 'from' %r is not a known class"%( where, source ) ) )
                elif source == target:
                    errors.append( bu_settings.rule( 'class_promotion.from_is_target', "%s: 'from' is the same label as the target"%( where ) ) )
                _check_fields( rule, bu_settings.RULE_FIELDS['promotion'], where, errors,
                               skip=( 'from', 'requires' ) )
                requires = rule.get( 'requires', [] )
                if not isinstance( requires, list ):
                    errors.append( bu_settings.rule( 'class_promotion.not_a_mapping', "%s: requires must be a list of evidence dicts"%( where ) ) )
                    continue
                for req_index, requirement in enumerate( requires ):
                    req_where = "%s['requires'][%d]"%( where, req_index )
                    if not isinstance( requirement, dict ):
                        errors.append( bu_settings.rule( 'class_promotion.not_a_mapping', "%s must be a dict, got %r"%( req_where, requirement ) ) )
                        continue
                    unknown = set( requirement ) - bu_track.PROMOTION_EVIDENCE_KEYS
                    if unknown:
                        errors.append( bu_settings.rule( 'class_promotion.unknown_keys', "%s has unrecognized key(s) %s (valid: %s)"%(
                            req_where, sorted( unknown ),
                            sorted( bu_track.PROMOTION_EVIDENCE_KEYS ) ) ) )
                    if requirement.get( 'label' ) not in valid_classes:
                        errors.append( bu_settings.rule( 'class_promotion.unknown_label', "%s: label %r is not a known class"%(
                            req_where, requirement.get( 'label' ) ) ) )
                    _check_fields( requirement, bu_settings.RULE_FIELDS['promotion_requirement'],
                                   req_where, errors, skip=( 'label', ) )
                    if 'overlap' in requirement:
                        if requirement['overlap'] != 'anywhere':
                            pass   # the field check above reports it
                        elif ( 'min_iou' in requirement
                               or 'min_source_overlap' in requirement ):
                            # Silently ignoring one of two contradictory
                            # settings is how a rule ends up not doing what
                            # its config says.
                            errors.append( bu_settings.rule( 'class_promotion.anywhere_with_bounds', "%s: overlap 'anywhere' means no spatial test, so "
                                           "min_iou / min_source_overlap cannot also be set"%(
                                               req_where, ) ) )


def _warn_promotion_evidence_also_suppresses( warnings ):
    """
    Flag a promotion whose evidence label also suppresses its target.

    Promotion runs before suppression, so a box promoted to T because
    label E overlaps it then meets T's suppression rules - and if one of
    those is "T is suppressed by E", the same overlap that justified the
    promotion can delete the promoted box. The rule then does nothing on
    exactly the cases it was written for, with no error anywhere.

    Not an error: when E's score is low relative to the source, the
    suppression margin is not met and the promotion survives. But it is
    the first thing to check when a rule reports promotions and the
    render shows no change.
    """
    for backend_name in bu_detector.registered_backend_names():
        suppression = bu_detector.get_class_suppression( backend_name )
        for target, rules in bu_detector.get_class_promotion( backend_name ).items():
            target_rules = suppression.get( target ) or []
            target_rules = [ target_rules ] if isinstance( target_rules, dict ) else target_rules
            suppressors = { rule.get( 'suppressed_by' ) for rule in target_rules
                            if isinstance( rule, dict ) }
            for rule in ( [ rules ] if isinstance( rules, dict ) else rules ):
                if not isinstance( rule, dict ):
                    continue
                for requirement in rule.get( 'requires' ) or []:
                    label = requirement.get( 'label' ) if isinstance( requirement, dict ) else None
                    if label in suppressors:
                        warnings.append(
                            "detector_backend[%r]: promotion %s<-%s uses %r as evidence, but %r "
                            "is also a suppressor of %s. A box promoted on that evidence can be "
                            "suppressed by the same overlap one stage later."%(
                                backend_name, target, rule.get( 'from' ), label, label, target ) )


def _validate_profiles( errors ):
    """
    Check every backend's structure-profile block.

    Profiles were unvalidated when they shipped, so a typo in a variant
    key (min_cut_per_min, video_censor_fsp) was silently ignored and the
    profile quietly behaved like its neighbour.
    """
    import betautils_detector as detector
    # One shared definition of what each match_on signal reads, so this
    # check and the selector cannot drift apart.
    threshold_key_for = dict( detector.PROFILE_THRESHOLD_KEYS )
    known_match_signals = tuple( sorted( threshold_key_for ) )
    known_variant_keys = ( { 'item_overrides', 'video_censor_fps', 'manual_only' }
                           | set( threshold_key_for.values() ) )
    for backend_name in detector.registered_backend_names():
        profiles = detector.get_profiles( backend_name )
        if not profiles:
            continue
        where_base = "detector_backend[%r]['profiles']"%( backend_name, )
        unknown = set( profiles ) - { 'default', 'match_on', 'variants' }
        if unknown:
            errors.append( bu_settings.rule( 'profiles.unknown_keys', "%s has unrecognized key(s) %s"%( where_base, sorted( unknown ) ) ) )
        match_on = profiles.get( 'match_on', 'cuts_per_min' )
        if match_on not in known_match_signals:
            match_on = None   # the registry reports it
        variants = profiles.get( 'variants' ) or {}
        default = profiles.get( 'default' )
        if default is not None and default not in variants:
            errors.append( bu_settings.rule( 'profiles.default_unknown', "%s['default'] %r names no variant (have: %s)"%(
                where_base, default, sorted( variants ) ) ) )
        for name, variant in variants.items():
            where = "%s['variants'][%r]"%( where_base, name )
            if not isinstance( variant, dict ):
                errors.append( bu_settings.rule( 'profiles.not_a_mapping', "%s must be a dict"%( where, ) ) )
                continue
            unknown = set( variant ) - known_variant_keys
            if unknown:
                errors.append( bu_settings.rule( 'profiles.unknown_keys', "%s has unrecognized key(s) %s (valid: %s)"%(
                    where, sorted( unknown ), sorted( known_variant_keys ) ) ) )
            # A threshold for the signal that is NOT being matched on is
            # read by nothing. That is the silent-neighbour failure this
            # whole function exists to catch: the variant looks tuned and
            # behaves like the default. The variant that is only ever
            # reached as the default, or by name, legitimately carries no
            # threshold at all, so absence is fine and only the WRONG one
            # is an error.
            if match_on and not variant.get( 'manual_only' ):
                wrong = set( threshold_key_for.values() ) - { threshold_key_for[match_on] }
                present_wrong = sorted( wrong & set( variant ) )
                if present_wrong and threshold_key_for[match_on] not in variant:
                    errors.append( bu_settings.rule( 'profiles.threshold_for_other_signal',
                        "%s sets %s but %s['match_on'] is %r, which reads %s. "
                        "This variant would never match automatically."%(
                            where, present_wrong[0], where_base, match_on,
                            threshold_key_for[match_on] ) ) )
            for label, overrides in ( variant.get( 'item_overrides' ) or {} ).items():
                if not isinstance( overrides, dict ):
                    errors.append( bu_settings.rule( 'profiles.not_a_mapping', "%s['item_overrides'][%r] must be a dict"%( where, label ) ) )
                    continue
                unknown = set( overrides ) - detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
                if unknown:
                    errors.append( bu_settings.rule( 'profiles.unknown_keys', "%s['item_overrides'][%r] has unrecognized key(s) %s"%(
                        where, label, sorted( unknown ) ) ) )

    # Path rules must name real variants too, for the same reason.
    mapping = getattr( betaconfig, 'profile_by_path_pattern', None )
    if mapping is not None:
        if not isinstance( mapping, dict ):
            errors.append( bu_settings.rule( 'profiles.not_a_mapping', "profile_by_path_pattern must be a dict of "
                           "{glob: profile_name}, got %r"%( type( mapping ).__name__, ) ) )
        else:
            selected = detector.selected_backend_name()
            variants = detector.get_profiles( selected ).get( 'variants' ) or {}
            for pattern, name in mapping.items():
                if not isinstance( pattern, str ) or not pattern:
                    errors.append( bu_settings.rule( 'profiles.path_pattern_not_string', "profile_by_path_pattern has a non-string "
                                   "pattern %r"%( pattern, ) ) )
                if name not in variants:
                    errors.append( bu_settings.rule( 'profiles.path_pattern_unknown_profile',
                        "profile_by_path_pattern[%r] is %r, which names no variant for "
                        "backend %r (have: %s)"%( pattern, name, selected,
                                                  sorted( variants ) or 'none' ) ) )

    # The folder convention. Same reasoning as the path patterns above: a
    # key naming no variant would quietly fall through to measurement
    # while looking configured.
    folders = getattr( betaconfig, 'profile_by_source_dir', None )
    if folders is not None:
        if not isinstance( folders, dict ):
            errors.append( bu_settings.rule( 'profiles.not_a_mapping', "profile_by_source_dir must be a dict of "
                           "{folder: profile_name}, got %r"%(
                               type( folders ).__name__, ) ) )
        else:
            selected = detector.selected_backend_name()
            variants = detector.get_profiles( selected ).get( 'variants' ) or {}
            seen = {}
            for folder, name in folders.items():
                where = "profile_by_source_dir[%r]"%( folder, )
                if not isinstance( folder, str ) or not folder.strip():
                    errors.append( bu_settings.rule( 'profiles.source_dir_empty_key', "profile_by_source_dir has an empty or "
                                   "non-string folder key %r"%( folder, ) ) )
                    continue
                parts = detector._dir_key_parts( folder )
                if not parts:
                    # '/' or '.' normalise away to nothing, which would
                    # match every file and silently override the
                    # root-falls-through-to-measurement rule.
                    errors.append( bu_settings.rule( 'profiles.source_dir_matches_everything', "%s normalises to an empty folder, which would "
                                   "match every source file. Remove it, or name a "
                                   "real folder"%( where, ) ) )
                    continue
                if '..' in parts:
                    errors.append( bu_settings.rule( 'profiles.source_dir_climbs', "%s climbs above the uncensored root with '..', "
                                   "which can never match"%( where, ) ) )
                    continue
                if os.path.isabs( str( folder ).replace( '\\', '/' ) ):
                    errors.append( bu_settings.rule( 'profiles.source_dir_absolute', "%s is an absolute path. Keys are named relative "
                                   "to betaconst.video_path_uncensored"%( where, ) ) )
                    continue
                # Two spellings of one folder ('comps', '/comps/') would
                # both match, and which won would depend on dict order.
                if parts in seen:
                    errors.append( bu_settings.rule( 'profiles.source_dir_duplicate', "%s and profile_by_source_dir[%r] are the same "
                                   "folder spelled differently"%( where, seen[parts] ) ) )
                else:
                    seen[parts] = folder
                if name not in variants:
                    errors.append( bu_settings.rule( 'profiles.source_dir_unknown_profile',
                        "%s is %r, which names no variant for backend %r "
                        "(have: %s)"%( where, name, selected,
                                       sorted( variants ) or 'none' ) ) )

    # A forced profile that names nothing is an error, not a fallback.
    # Silently rendering unprofiled after --profile comp was typed is the
    # failure where the run looks like it worked and the output is wrong.
    forced = getattr( betaconfig, 'force_structure_profile', None )
    if forced:
        selected = detector.selected_backend_name()
        variants = detector.get_profiles( selected ).get( 'variants' ) or {}
        if forced not in variants:
            errors.append( bu_settings.rule( 'profiles.forced_unknown_profile',
                "force_structure_profile (--profile) is %r, which names no variant for "
                "backend %r (have: %s)"%( forced, selected, sorted( variants ) or 'none' ) ) )


def _validate_class_suppression_zero_margin( warnings, valid_classes ):
    """
    Warn (don't fail) on any class_suppression rule whose margin is
    exactly 0.0, across every registered backend - same per-backend
    resolution as _validate_class_suppression above.

    This is deliberately a warning, not an error: margin=0.0 is a
    real, sometimes-correct value (derive_suppression_rules.py's
    propose_margin legitimately proposes it when the suppressing
    label rarely beats the suppressed label's confidence at all - a
    positive margin there would gut the rule rather than refine it,
    see CONFIG_REFERENCE.md's nudenet_v3 margin section for several
    real examples), so this must never hard-block a run. But per
    explicit design decision: a bare tie (margin=0.0,
    "suppressing label's score just has to be >= the suppressed
    label's") is too permissive on its own - every nudenet_v3 rule that
    had been exactly 0.0 was raised to 0.10 by hand, and a future edit that reintroduces exactly 0.0 (most likely by
    copying an older rule, or by a future derive_suppression_rules.py
    run getting pasted back in unreviewed) flagged rather than
    silently shipped. This checks only for margin == 0.0 specifically,
    not "below some floor" - a rule someone deliberately sets to e.g.
    0.05 is a reviewed, in-between choice, not the bare-tie case this
    warning exists to catch. margin < 0 is a hard error instead (see
    _validate_class_suppression) - there's no legitimate reading of a
    negative margin, unlike 0.0.
    """
    for backend_name in bu_detector.registered_backend_names():
        class_suppression = bu_detector.get_class_suppression( backend_name )
        for label, rules in class_suppression.items():
            if label not in valid_classes:
                continue  # already flagged as an error by _validate_class_suppression
            is_list = isinstance( rules, list )
            where_base = "detector_backend[%r]['class_suppression']['%s']"%(backend_name, label)
            rule_list = rules if is_list else [ rules ]
            for idx, rule in enumerate( rule_list ):
                where = "%s[%d]"%(where_base, idx) if is_list else where_base
                if not isinstance( rule, dict ):
                    continue  # already flagged as an error
                margin = rule.get( 'margin' )
                if not isinstance( margin, (int, float) ):
                    continue  # missing/non-numeric margin already flagged as an error
                if margin == 0:
                    suppressed_by = rule.get( 'suppressed_by', '?' )
                    warnings.append( "%s['margin'] = 0.0 (suppressed_by='%s') - the suppressing label only needs "
                        "to tie the suppressed label's confidence to fire this rule. If this is intentional (see "
                        "derive_suppression_rules.py's propose_margin for when 0.0 is the honest data-driven "
                        "answer), no action needed - this is a warning, not an error."%(where, suppressed_by) )


def _warn_unsafe_suppression_rules( warnings ):
    """
    A hand-written rule for one of bu_detector.UNSAFE_SUPPRESSION_PAIRS.
    The tools never propose these, so one in the config was typed in;
    a warning rather than an error, because it is the person's call.
    """
    for backend_name in bu_detector.registered_backend_names():
        for label, rules in bu_detector.get_class_suppression( backend_name ).items():
            for rule in ( rules if isinstance( rules, list ) else [ rules ] ):
                if not isinstance( rule, dict ):
                    continue
                pair = "%s<-%s"%( label, rule.get( 'suppressed_by' ) )
                reason = bu_detector.UNSAFE_SUPPRESSION_PAIRS.get( pair )
                if reason:
                    warnings.append( "detector_backend[%r]['class_suppression'] has the rule %s, which "
                                     "removes the censor when both are really there: %s"%(
                                         backend_name, pair, reason ) )


def _collect_validation_warnings():
    """
    Non-fatal checks: things worth flagging but not worth aborting a
    run over (see _validate_class_suppression_zero_margin for why
    margin=0.0 specifically is a warning rather than an error). Mirrors
    _collect_validation_errors' structure/one-function-per-concern
    pattern, kept as a separate list/separate functions rather than
    reusing 'errors' so validate_config can print warnings without
    ever triggering the SystemExit(1) abort path errors does.
    """
    warnings = []
    valid_classes = _valid_class_names()
    _validate_class_suppression_zero_margin( warnings, valid_classes )
    _warn_promotion_evidence_also_suppresses( warnings )
    _warn_unsafe_suppression_rules( warnings )
    _warn_reverse( warnings )
    _warn_local_config( warnings )
    return warnings


def _check_item_override_values( override, where, errors ):
    """
    One item_overrides-shaped dict: the shared block, a backend's or
    variant's resolved result, a reverse window or override. Each value's
    kind and range is a registry check where it is set; what is left here
    is the censor style, geometry_action, and the checks that only make
    sense between values, which can fail only once tiers are merged (a
    variant lowering track_max_gap below the backend's
    interpolation_max_gap is fine in each block and wrong together).
    """
    if 'censor_style' in override:
        _check_censor_style( override['censor_style'], "%s['censor_style']"%(where), errors )

    track_max_gap = override.get( 'track_max_gap' )
    if _is_positive_number( track_max_gap ):
        # track_max_gap tighter than this label's own interpolation_max_gap
        # isn't invalid, but it silently makes that interpolation_max_gap
        # partly moot (a gap has to pass the track_max_gap test before
        # interpolation ever gets a chance to fill it). Flagged as a
        # likely-unintentional combination rather than allowed quietly.
        label_interp_max_gap = override.get( 'interpolation_max_gap', betaconfig.default_interpolation_max_gap )
        if isinstance( label_interp_max_gap, (int, float) ) and track_max_gap < label_interp_max_gap:
            errors.append( bu_settings.rule( 'label.track_gap_below_interpolation_gap', "%s['track_max_gap'] (%r) is less than this label's interpolation_max_gap (%r) - interpolation_max_gap will never get the chance to matter for gaps between track_max_gap and interpolation_max_gap, since the track hard-resets before interpolation is ever considered. This may be intentional, but if not, raise track_max_gap to at least interpolation_max_gap."%(where, track_max_gap, label_interp_max_gap) ) )

    # --- score hysteresis ---
    min_prob_continue = override.get( 'min_prob_continue' )
    if isinstance( min_prob_continue, (int, float) ) and not isinstance( min_prob_continue, bool ) \
            and 0 < min_prob_continue < 1:
        label_min_prob = override.get( 'min_prob', betaconfig.default_min_prob )
        if isinstance( label_min_prob, (int, float) ) and min_prob_continue > label_min_prob:
            errors.append( bu_settings.rule( 'label.hysteresis_inverted', "%s['min_prob_continue'] (%r) is ABOVE this label's min_prob (%r), which "
                "inverts hysteresis: the bar to keep tracking something would be higher than the bar "
                "to start. min_prob_continue is the lower, 'stay' threshold - set it below min_prob "
                "(or to None to disable hysteresis)."%(where, min_prob_continue, label_min_prob) ) )
        global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
        if isinstance( global_min_prob, (int, float) ) and min_prob_continue <= global_min_prob:
            errors.append( bu_settings.rule( 'label.hysteresis_below_floor', "%s['min_prob_continue'] (%r) is at or below global_min_prob (%r) - nothing "
                "that low ever reaches it, so hysteresis would extend no further than the global floor "
                "already does. Raise it, or lower global_min_prob."%(
                    where, min_prob_continue, global_min_prob ) ) )

    # --- geometry sanity filter ---
    min_area = override.get( 'min_area_fraction' )
    max_area = override.get( 'max_area_fraction' )
    if ( isinstance( min_area, (int, float) ) and isinstance( max_area, (int, float) )
            and min_area >= max_area ):
        errors.append( bu_settings.rule( 'label.area_bounds_crossed', "%s['min_area_fraction'] (%r) is not below ['max_area_fraction'] (%r) - no box can "
            "satisfy both, so every detection of this label violates a bound (what that then DOES depends "
            "on geometry_action)."%(where, min_area, max_area) ) )

    if 'geometry_action' in override:
        _check_geometry_action( override['geometry_action'], "%s['geometry_action']"%( where, ), errors )

    min_aspect = override.get( 'min_aspect_ratio' )
    max_aspect = override.get( 'max_aspect_ratio' )
    if ( isinstance( min_aspect, (int, float) ) and isinstance( max_aspect, (int, float) )
            and min_aspect >= max_aspect ):
        errors.append( bu_settings.rule( 'label.aspect_bounds_crossed', "%s['min_aspect_ratio'] (%r) is not below ['max_aspect_ratio'] (%r) - no box can satisfy "
            "both, so every detection of this label violates a bound (what that then DOES depends on "
            "geometry_action)."%(where, min_aspect, max_aspect) ) )


def _is_positive_number( value ):
    return isinstance( value, (int, float) ) and not isinstance( value, bool ) and value > 0


def _check_geometry_action( action, where, errors ):
    """One action for both bounds, or a {'min': .., 'max': ..} mapping of them."""
    valid = bu_settings.geometry_actions()
    if isinstance( action, dict ):
        unknown_bounds = sorted( set( action ) - { 'min', 'max' } )
        if unknown_bounds:
            errors.append( bu_settings.rule( 'geometry_action.unknown_bounds', "%s mapping keys must be 'min'/'max', got %s"%( where, unknown_bounds ) ) )
        bad = [ value for value in action.values() if value not in valid ]
        if bad:
            errors.append( bu_settings.rule( 'geometry_action.bad_value', "%s values must be one of %s, got %r"%( where, sorted( valid ), bad ) ) )
    elif action not in valid:
        errors.append( bu_settings.rule( 'geometry_action.bad_value', "%s must be one of %s, or a {'min':..,'max':..} mapping, got %r"%(
            where, sorted( valid ), action ) ) )


def _validate_item_overrides( errors, valid_classes ):
    """
    Check the SHARED item_overrides block: every key is a real class,
    every value is a dict containing ONLY censor_style/censor_shape -
    the sole keys that stay shared across backends (pure rendering
    choices, not detection-derived - see bu_detector.
    ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS's own comment).

    Every backend-tunable key (min_prob, width_area_safety,
    height_area_safety, time_safety, track_max_gap, interpolation_max_gap,
    paired_style, etc - the full ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS set)
    used to also be accepted here, since a config that never set a
    backend-specific override still put every key in this shared block
    and relied on it as the fallback. Those keys have since moved fully
    into each backend's own detector_backend[<name>]['item_overrides']
    [label] block (see betaconfig.py's own comment on item_overrides for
    why - no per-model default convention should be assumed, same
    reasoning as class_suppression's margin becoming required rather
    than defaulted). A backend-tunable key showing up in the SHARED
    block now is flagged as an error here - almost certainly a leftover
    from before that move (silently ignored by bu_detector.
    get_item_overrides, since it only reads backend-tunable keys from a
    backend's own block - see that function's resolution-order
    docstring), not a mistake to let slide quietly.

    Args:
        errors: fatal-error list, appended to in place.
        valid_classes: Set of every real class name (see
            betaconst.classes), used to catch a typo'd label key.

    Returns:
        The item_overrides dict itself (getattr'd with a {} fallback),
        so callers that need it again (censor_overlap_strategy coverage)
        don't have to re-fetch it.
    """
    item_overrides = getattr( betaconfig, 'item_overrides', {} )
    known_shared_keys = { 'censor_style', 'censor_shape' }
    for label, override in item_overrides.items():
        where = "item_overrides['%s']"%(label)
        if label not in valid_classes:
            errors.append( bu_settings.rule( 'item_overrides.unknown_label', "item_overrides key '%s' is not a known class (valid: %s)"%(label, sorted(valid_classes)) ) )
        if not isinstance( override, dict ):
            errors.append( bu_settings.rule( 'item_overrides.not_a_mapping', "%s must be a dict, got %r"%(where, override) ) )
            continue
        backend_tunable_keys_present = set(override.keys()) & bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
        if backend_tunable_keys_present:
            errors.append( bu_settings.rule( 'item_overrides.tunable_in_shared_block', "%s has backend-tunable key(s) %s in the SHARED item_overrides block - these moved to "
                "each backend's own detector_backend[<name>]['item_overrides']['%s'] block and are no "
                "longer read from here at all (bu_detector.get_item_overrides only reads backend-tunable keys from "
                "a backend's own block), so this is silently doing nothing - move %s into detector_backend['nudenet_v3']"
                "['item_overrides']['%s'] and detector_backend['retinanet_v2']['item_overrides']['%s'] (with "
                "whatever value is actually correct for each model - don't just copy the same number to both) "
                "instead."%(where, sorted(backend_tunable_keys_present), label, sorted(backend_tunable_keys_present), label, label) ) )
        unknown_keys = set(override.keys()) - known_shared_keys - bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
        if unknown_keys:
            errors.append( bu_settings.rule( 'item_overrides.unknown_keys', "%s has unrecognized key(s) %s, likely a typo (valid keys here: %s)"%(where, sorted(unknown_keys), sorted(known_shared_keys)) ) )
        _check_item_override_values( override, where, errors )

    return item_overrides


def _check_resolved_item_override( label, resolved, where, errors ):
    """
    One label's fully resolved settings (shared, backend and variant
    tiers merged), checked as _check_item_override_values does.

    While reverse mode is on, a window label's censor_style comes only
    from its reverse['windows'] entry, or defaults to 'show', never from
    these tiers. _validate_reverse already checks it there under the
    window rules ('show' is valid only in a window), so it is left out
    here rather than rejected as an unknown normal style.
    """
    if bu_reverse.window_entry( label ) is not None:
        resolved = { key: value for key, value in resolved.items() if key != 'censor_style' }
    _check_item_override_values( resolved, where, errors )


def _validate_backend_item_overrides( errors, valid_classes ):
    """
    Check every registered backend's per-backend item_overrides block
    (detector_backend[<name>]['item_overrides'][label]) - added
    alongside bu_detector.get_item_overrides, same reasoning
    and structure as _validate_class_suppression/_validate_nn_batch_size:
    checks every backend, not just the currently selected one, so a typo
    in an unselected backend's override is still caught before it's ever
    relied on.

    Two things checked here, distinct from _validate_item_overrides
    (which checks the shared block): (1) a backend's raw override block
    only contains keys in ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS - censor_
    style/censor_shape are deliberately not overridable per-backend (see
    that set's own comment), so setting them here is almost certainly a
    mistake, not silently ignored without a word; (2) the fully RESOLVED
    per-backend result (shared + this backend's override merged, i.e.
    what bu_detector.get_item_overrides actually returns and betatv.py/
    betautils_config.py actually use) is well-formed, using the same
    per-key checks _validate_item_overrides applies to the shared block -
    this catches e.g. a backend override that sets track_max_gap below
    the shared block's interpolation_max_gap, which neither block would
    show as invalid checked in isolation.
    """
    for backend_name in bu_detector.registered_backend_names():
        backend_config = bu_detector.get_backend_config( backend_name )
        backend_item_overrides = backend_config.get( 'item_overrides', {} )
        if not isinstance( backend_item_overrides, dict ):
            errors.append( bu_settings.rule( 'backend_item_overrides.not_a_mapping', "detector_backend[%r]['item_overrides'] must be a dict, got %r"%(backend_name, backend_item_overrides) ) )
            continue
        for label, override in backend_item_overrides.items():
            where = "detector_backend[%r]['item_overrides']['%s']"%(backend_name, label)
            if label not in valid_classes:
                errors.append( bu_settings.rule( 'backend_item_overrides.unknown_label', "detector_backend[%r]['item_overrides'] key '%s' is not a known class (valid: %s)"%(backend_name, label, sorted(valid_classes)) ) )
            if not isinstance( override, dict ):
                errors.append( bu_settings.rule( 'backend_item_overrides.not_a_mapping', "%s must be a dict, got %r"%(where, override) ) )
                continue
            unknown_keys = set(override.keys()) - bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
            if unknown_keys:
                errors.append( bu_settings.rule( 'backend_item_overrides.unknown_keys', "%s has key(s) %s that aren't backend-tunable (censor_style/censor_shape always come from the shared item_overrides block - see ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS) - valid keys here: %s"%(where, sorted(unknown_keys), sorted(bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS)) ) )

        # now check the fully resolved result for every label that has
        # EITHER a shared or a backend-specific override, not just the
        # labels present in this backend's own block - a backend with no
        # override at all for a label still resolves to the shared
        # block's values, and those need the same checks too (already
        # covered once by _validate_item_overrides for the shared block
        # alone, but re-checking the resolved result here is what catches
        # a shared/backend-override COMBINATION that's only invalid
        # together, like the track_max_gap/interpolation_max_gap example
        # in this function's own docstring).
        all_labels = set( getattr( betaconfig, 'item_overrides', {} ).keys() ) | set( backend_item_overrides.keys() )
        for label in all_labels:
            resolved = bu_detector.get_item_overrides( label, backend_name )
            where = "detector_backend[%r]'s resolved item_overrides['%s']"%(backend_name, label)
            _check_resolved_item_override( label, resolved, where, errors )

        _validate_variant_item_overrides( errors, valid_classes, backend_name, all_labels )


def _validate_variant_item_overrides( errors, valid_classes, backend_name, shared_labels ):
    """
    Check a backend's per-VARIANT override blocks, the third and most
    specific resolution tier.

    detector_backend[<backend>]['variants'][<variant>]['item_overrides']

    Same two checks as the backend tier, for the same reasons, plus one
    more that only applies here: a variant name that the adapter does
    not declare. A typo there is invisible at runtime - the block simply
    never resolves, and the variant quietly runs on the backend's values
    while config says otherwise - which is exactly the kind of silent
    wrongness worth failing loudly at startup.

    The fully resolved result is re-checked per variant, because a
    combination can be invalid only together: a variant that lowers
    track_max_gap below the backend tier's interpolation_max_gap is
    well-formed in isolation at both tiers and wrong when merged.
    """
    backend_config = bu_detector.get_backend_config( backend_name )
    variants_block = backend_config.get( 'variants', {} )
    if not variants_block:
        return
    if not isinstance( variants_block, dict ):
        errors.append( bu_settings.rule( 'variant_item_overrides.not_a_mapping', "detector_backend[%r]['variants'] must be a dict, got %r"%(
            backend_name, variants_block ) ) )
        return

    declared = set( bu_detector.variant_names( backend_name ) )
    for variant_name, variant_block in variants_block.items():
        where_variant = "detector_backend[%r]['variants'][%r]"%(backend_name, variant_name)
        if declared and variant_name not in declared:
            errors.append( bu_settings.rule( 'variant_item_overrides.unknown_variant', "%s is not a variant %s declares (valid: %s) - a block under an "
                           "unknown variant name never resolves, so this would silently do "
                           "nothing"%( where_variant, backend_name, sorted( declared ) ) ) )
            continue
        if not isinstance( variant_block, dict ):
            errors.append( bu_settings.rule( 'variant_item_overrides.not_a_mapping', "%s must be a dict, got %r"%(where_variant, variant_block) ) )
            continue

        variant_item_overrides = variant_block.get( 'item_overrides', {} )
        if not isinstance( variant_item_overrides, dict ):
            errors.append( bu_settings.rule( 'variant_item_overrides.not_a_mapping', "%s['item_overrides'] must be a dict, got %r"%(
                where_variant, variant_item_overrides ) ) )
            continue

        for label, override in variant_item_overrides.items():
            where = "%s['item_overrides']['%s']"%(where_variant, label)
            if label not in valid_classes:
                errors.append( bu_settings.rule( 'variant_item_overrides.unknown_label', "%s['item_overrides'] key '%s' is not a known class (valid: %s)"%(
                    where_variant, label, sorted( valid_classes ) ) ) )
            if not isinstance( override, dict ):
                errors.append( bu_settings.rule( 'variant_item_overrides.not_a_mapping', "%s must be a dict, got %r"%(where, override) ) )
                continue
            unknown_keys = set( override.keys() ) - bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
            if unknown_keys:
                errors.append( bu_settings.rule( 'variant_item_overrides.unknown_keys', "%s has key(s) %s that aren't backend-tunable - valid keys "
                               "here: %s"%( where, sorted( unknown_keys ),
                                            sorted( bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS ) ) ) )

        for label in shared_labels | set( variant_item_overrides.keys() ):
            resolved = bu_detector.get_item_overrides( label, backend_name, variant_name )
            where = "%s's resolved item_overrides['%s']"%(where_variant, label)
            _check_resolved_item_override( label, resolved, where, errors )


def _validate_default_censor_style( errors ):
    """
    Check that default_censor_style is defined and well-formed.

    Returns:
        default_censor_style itself (or None if undefined), so the
        censor_overlap_strategy coverage check below can reuse it
        without re-fetching.
    """
    default_censor_style = getattr( betaconfig, 'default_censor_style', None )
    if default_censor_style is None:
        errors.append( bu_settings.rule( 'default_censor_style.missing', "default_censor_style is not defined (or every option is commented out)" ) )
    else:
        _check_censor_style( default_censor_style, "default_censor_style", errors )
    return default_censor_style


def _validate_type_default_shapes( errors ):
    """
    Check type_default_shapes: optional structural default shape per
    censor style TYPE (e.g. 'bar': 'box' - a bar only makes sense as a
    rectangle). Only applies when a style doesn't set its own 'shape' -
    see resolve_censor_shape in betautils_censor.py for the full
    precedence order.
    """
    type_default_shapes = getattr( betaconfig, 'type_default_shapes', {} )
    if not isinstance( type_default_shapes, dict ):
        errors.append( bu_settings.rule( 'type_default_shapes.not_a_mapping', "type_default_shapes must be a dict of {style_type: shape}, got %r"%(type_default_shapes) ) )
        return
    for style_type, shape in type_default_shapes.items():
        if style_type not in VALID_CENSOR_STYLES:
            errors.append( bu_settings.rule( 'type_default_shapes.unknown_type', "type_default_shapes key '%s' is not a known censor style type (valid: %s)"%(style_type, sorted(VALID_CENSOR_STYLES)) ) )


def _collect_style_types_in_use( style_config, styles_in_use ):
    """
    Add every 'type' found in a censor_style value (single dict or list
    of dicts) into styles_in_use.

    Args:
        style_config: A censor_style value - a single style dict, a list
            of style dicts, or (if malformed) something else, which is
            silently ignored here since shape/type validity is already
            checked elsewhere.
        styles_in_use: A set to add discovered type strings into
            (mutated in place).
    """
    entries = style_config if isinstance( style_config, list ) else [ style_config ]
    for entry in entries:
        if isinstance( entry, dict ) and 'type' in entry:
            styles_in_use.add( entry['type'] )


def _validate_censor_overlap_strategy_coverage( errors, default_censor_style, item_overrides ):
    """
    Check that every censor style type actually referenced by
    default_censor_style or any item_overrides censor_style has a
    corresponding entry in censor_overlap_strategy - otherwise
    censor_img_for_boxes will KeyError on it mid-run.

    Args:
        errors: Shared list of error strings to append to.
        default_censor_style: The value returned by
            _validate_default_censor_style.
        item_overrides: The value returned by _validate_item_overrides.
    """
    censor_overlap_strategy = getattr( betaconfig, 'censor_overlap_strategy', {} )
    styles_in_use = set()
    _collect_style_types_in_use( default_censor_style, styles_in_use )
    for override in item_overrides.values():
        if isinstance( override, dict ) and 'censor_style' in override:
            _collect_style_types_in_use( override['censor_style'], styles_in_use )
    for style_type in styles_in_use:
        if style_type not in censor_overlap_strategy:
            errors.append( bu_settings.rule( 'censor_overlap_strategy.missing_type', "censor_overlap_strategy is missing an entry for '%s', which is used by default_censor_style/item_overrides"%(style_type) ) )


def _validate_picture_sizes( errors ):
    """
    Check every backend's resolved picture_sizes.

    picture_sizes became backend-resolvable (see
    betautils_detector.get_picture_sizes): a detection size that is
    correct for one model can be badly wrong for another, and the
    adapter knows its own native size. The shared
    betaconfig.picture_sizes fallback was later removed entirely, so a stale
    module-level value is reported here rather than silently ignored.
    Each backend's RESOLVED list is checked, so a bad value in an
    unselected backend is caught before it is relied on. Per-adapter
    constraints (nudenet_v3 rejecting 0 and non-multiples of 32) are
    enforced by that adapter's own validate_backend_config.
    """
    if getattr( betaconfig, 'picture_sizes', None ) is not None:
        errors.append( bu_settings.rule( 'picture_sizes.removed_setting', "betaconfig.picture_sizes is no longer read - a detection "
            "size is a property of the model, so move this value into "
            "detector_backend[<name>]['picture_sizes'], or delete it and let the adapter's native "
            "size apply. Leaving it here would silently do nothing." ) )

    for backend_name in bu_detector.registered_backend_names():
        resolved = bu_detector.get_picture_sizes( backend_name )
        where = "detector_backend[%r]'s resolved picture_sizes"%(backend_name)
        if not resolved:
            errors.append( bu_settings.rule( 'picture_sizes.empty', "%s is empty - set detector_backend[%r]['picture_sizes'] or "
                "detector_backend['defaults']['picture_sizes'], or give the adapter a "
                "native_picture_sizes()"%( where, backend_name ) ) )
            continue
        if not all( isinstance( size, int ) and size >= 0 for size in resolved ):
            errors.append( bu_settings.rule( 'picture_sizes.not_sizes', "%s must be non-negative ints, got %r"%(where, resolved) ) )
        if len( set( resolved ) ) != len( resolved ):
            errors.append( bu_settings.rule( 'picture_sizes.duplicates', "%s has duplicate entries (%r) - the model would run twice on identical "
                "input and every detection would be duplicated"%(where, resolved) ) )


def _validate_label_defaults( errors ):
    """
    default_geometry_action, the one label default with a structure of
    its own. Every other default's kind and range is a registry check.
    """
    import betautils_track as bu_track  # deferred: bu_track imports this module
    action = getattr( betaconfig, 'default_geometry_action', bu_track.DEFAULT_GEOMETRY_ACTION )
    _check_geometry_action( action, "default_geometry_action", errors )


def _validate_cross_size_dedup( errors ):
    """Check the cross-size dedup settings."""
    settings = getattr( betaconfig, 'cross_size_dedup', {} )
    if not isinstance( settings, dict ):
        errors.append( bu_settings.rule( 'cross_size_dedup.not_a_mapping', "cross_size_dedup must be a dict, got %r"%(settings,) ) )
        return
    unknown = set( settings.keys() ) - { 'enabled', 'iou_threshold' }
    if unknown:
        errors.append( bu_settings.rule( 'cross_size_dedup.unknown_keys', "cross_size_dedup has unrecognized key(s) %s (valid: enabled, iou_threshold)"%(
            sorted( unknown ),) ) )



def _validate_local_config( errors ):
    """Lines of betaconfig_local.py that could not be applied, each naming its file and line."""
    loaded = bu_local.LOADED
    if loaded is None:
        return
    # A line of the file that could not be applied. Its text names the
    # file and line; the path is the setting it tried to set, if any.
    errors.extend( problem if isinstance( problem, bu_settings.Problem )
                   else bu_settings.rule( 'local_config.line_not_applied', problem, path='' )
                   for problem in loaded.problems )


def _warn_local_config( warnings ):
    loaded = bu_local.LOADED
    for warning in [ bu_local.legacy_warning( loaded ), bu_local.ignored_repo_warning( loaded ),
                     bu_local.encoder_scale_warning( loaded ) ]:
        if warning:
            warnings.append( warning )
    warnings.extend( bu_local.new_key_warnings( loaded ) )


def valid_encode_presets( codec ):
    """
    The preset names an encode with `codec` accepts.

    x264's names always, since bu_render.encoder_options translates them
    for NVENC; NVENC's own p1 (fastest) to p7 (best) as well when the
    codec is an NVENC one. Checking only x264's names rejected the
    'p5' that `betabench encode` recommends for h264_nvenc.
    """
    import betautils_render as bu_render   # deferred: bu_render's imports reach back here
    if codec in bu_render.NVENC_CODECS:
        return VALID_ENCODE_PRESETS | NVENC_PRESETS
    return VALID_ENCODE_PRESETS


def _validate_logging_settings( errors ):
    """log_path is set while logging is on. Levels, colors and palettes are registry choices."""
    if getattr( betaconfig, 'logging_enabled', False ) and not getattr( betaconfig, 'log_path', None ):
        errors.append( bu_settings.rule( 'logging.log_path_required', "logging_enabled is True but log_path is not set" ) )


def _validate_stats_settings( errors ):
    """
    When stats_enabled is on, check stats_path is set. No-op entirely
    when stats are off.
    """
    if not getattr( betaconfig, 'stats_enabled', False ):
        return
    if not getattr( betaconfig, 'stats_path', None ):
        errors.append( bu_settings.rule( 'stats.stats_path_required', "stats_enabled is True but stats_path is not set" ) )


def _validate_nn_batch_size( errors ):
    """
    The module-level nn_batch_size is no longer read. Each backend's own
    and the shared detector_backend['defaults'] value are registry checks
    (a positive whole number).
    """
    if getattr( betaconfig, 'nn_batch_size', None ) is not None:
        errors.append( bu_settings.rule( 'nn_batch_size.removed_setting', "betaconfig.nn_batch_size is no longer read - move it to "
            "detector_backend['defaults']['nn_batch_size'] for a value shared by every backend, or "
            "to detector_backend[<name>]['nn_batch_size'] for one model. Leaving it here would "
            "silently do nothing." ) )


def _validate_min_prob_floor_consistency( errors ):
    """
    Check global_min_prob is a valid probability, and that neither
    default_min_prob nor any item_overrides min_prob sits at or below
    it - either would be a silent no-op rather than a deliberately
    permissive setting, since global_min_prob is applied to every raw
    detection BEFORE default_min_prob/item_overrides min_prob ever see
    it (applied inside each detector adapter - see betautils_detector.py).
    That's exactly
    the bug exposed_vulva's min_prob had (0.10, sitting below the
    then-hardcoded 0.30 floor) before global_min_prob was made
    configurable - caught here so a run doesn't get tuned/kicked off
    against a min_prob that was never actually going to do anything.
    """
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    if not isinstance( global_min_prob, (int, float) ) or not (0 <= global_min_prob < 1):
        return   # the registry reports it

    default_min_prob = getattr( betaconfig, 'default_min_prob', None )
    if isinstance( default_min_prob, (int, float) ) and default_min_prob <= global_min_prob:
        errors.append( bu_settings.rule( 'min_prob.default_below_floor', "default_min_prob (%r) is at or below global_min_prob (%r) - it can never reject anything, since nothing that low ever survives to be checked against it. Raise default_min_prob above global_min_prob, or lower global_min_prob."%(default_min_prob, global_min_prob) ) )

    # min_prob is backend-tunable (see bu_detector.ITEM_OVERRIDE_BACKEND_
    # TUNABLE_KEYS), so check every registered backend's RESOLVED value
    # per label, not just the shared item_overrides block - a backend
    # override could set a min_prob below the floor even if the shared
    # block's value for that label is fine, and vice versa.
    for backend_name in bu_detector.registered_backend_names():
        for label in set( getattr( betaconfig, 'item_overrides', {} ).keys() ):
            resolved = bu_detector.get_item_overrides( label, backend_name )
            if 'min_prob' in resolved:
                label_min_prob = resolved['min_prob']
                if isinstance( label_min_prob, (int, float) ) and label_min_prob <= global_min_prob:
                    errors.append( bu_settings.rule( 'min_prob.label_below_floor', "detector_backend[%r]'s resolved item_overrides['%s']['min_prob'] (%r) is at or below global_min_prob (%r) - it can never reject anything for this label under this backend, since nothing that low ever survives to be checked against it. Raise it above global_min_prob, or lower global_min_prob."%(backend_name, label, label_min_prob, global_min_prob) ) )


def _validate_detector_backend( errors ):
    """
    Check that detector_backend['selected'] names a real, registered
    adapter (see betautils_detector.py's _BACKENDS) - catches a typo
    here immediately instead of failing later, mid-run, the first time
    something actually tries to resolve it. If the resolved adapter
    module defines an optional validate_backend_config(config, errors)
    hook, also calls it with that backend's own block from
    detector_backend - this is how a model-specific setting (e.g.
    nudenet_v3's candidate_floor/nms_iou) gets validated without this
    generic function needing to know anything about any one adapter's
    own config keys.
    """
    backend_setting = getattr( betaconfig, 'detector_backend', {} )
    if not isinstance( backend_setting, dict ):
        errors.append( bu_settings.rule( 'detector_backend.not_a_mapping', "detector_backend must be a dict, got %r"%(backend_setting,) ) )
        return

    known_names = set( bu_detector.registered_backend_names() )
    unknown_sections = set( backend_setting.keys() ) - known_names - { 'selected', 'defaults' }
    if unknown_sections:
        errors.append( bu_settings.rule( 'detector_backend.unknown_sections', "detector_backend has section(s) %s that are neither a registered adapter nor "
            "'selected'/'defaults' - a typo here is silently ignored at runtime, which is why it is an "
            "error here (registered adapters: %s)"%(
                sorted( unknown_sections ), ', '.join( sorted( known_names ) ) ) ) )

    backend = bu_detector.selected_backend_name()
    if backend not in known_names:
        errors.append( bu_settings.rule( 'detector_backend.unknown_backend', "detector_backend['selected'] %r is not a registered detector adapter - valid "
            "choices: %s"%(backend, ', '.join( sorted( known_names ) )) ) )
        return

    # Validate EVERY registered backend's own block, not just the
    # selected one, so switching 'selected' is never the first time a
    # typo in the other block is noticed.
    for name in sorted( known_names ):
        module = bu_detector.get_detector( name )
        if hasattr( module, 'validate_backend_config' ):
            module.validate_backend_config( bu_detector.get_backend_config( name ), errors )


def _validate_registry( errors ):
    """
    Every registered setting's kind, range and choices, from
    betautils_settings. A structured setting (a censor style, a rule
    list) is checked by its own checker below, which reads the
    registry's field tables.
    """
    for path, value, entry in bu_settings.concrete_leaves( vars( betaconfig ) ):
        if entry is None or len( entry.path ) != len( path ):
            continue
        errors.extend( bu_settings.check_value(
            bu_local.format_setting_path( path ), value, entry,
            '.'.join( str( part ) for part in path ) ) )


def _check_fields( value, fields, where, errors, skip=() ):
    """A dict's keys against a registry field table (STYLE_FIELDS, RULE_FIELDS)."""
    for key, spec in fields.items():
        if key in value and key not in skip:
            errors.extend( bu_settings.check_value( "%s['%s']"%( where, key ), value[key], spec ) )


def _collect_validation_errors():
    """
    Run every betaconfig.py sanity check and collect the results.

    This is validate_config()'s actual work - split out as its own
    function so it can be called (and its return value inspected)
    without also triggering validate_config()'s print+SystemExit
    behavior, which is useful for tooling/tests that want the error list
    itself rather than a hard process exit.

    Returns:
        A list of human-readable error strings, in the same fixed order
        the checks below run in. Empty if betaconfig.py is entirely
        valid.
    """
    errors = []
    valid_classes = _valid_class_names()

    _validate_registry( errors )
    registry_found_problems = bool( errors )

    def run( check, *args ):
        # A wrong-typed value the registry already reported can make a
        # later check fail before it gets to its own test; that value's
        # error is already in the list. Without a registry error a crash
        # is a real bug and is raised.
        try:
            return check( errors, *args )
        except Exception:
            if not registry_found_problems:
                raise
            return None

    run( _validate_items_to_censor, valid_classes )
    run( _validate_class_suppression, valid_classes )
    run( _validate_class_promotion, valid_classes )
    run( _validate_profiles )
    item_overrides = run( _validate_item_overrides, valid_classes ) or {}
    run( _validate_backend_item_overrides, valid_classes )
    default_censor_style = run( _validate_default_censor_style )
    run( _validate_type_default_shapes )
    run( _validate_censor_overlap_strategy_coverage, default_censor_style, item_overrides )
    run( _validate_picture_sizes )
    run( _validate_logging_settings )
    run( _validate_stats_settings )
    run( _validate_nn_batch_size )
    run( _validate_min_prob_floor_consistency )
    run( _validate_label_defaults )
    run( _validate_cross_size_dedup )
    run( _validate_detector_backend )
    run( _validate_reverse, valid_classes )
    run( _validate_person_backend )
    run( _validate_local_config )

    return errors


def validate_config():
    """
    Sanity-check betaconfig.py against betaconst.py so typos and bad
    values (e.g. a misspelled dict key) are caught immediately with a
    clear message, instead of surfacing as a cryptic
    KeyError/AttributeError midway through a long detection/censor run.

    Side effects:
        If any check fails, prints every error found (prefixed with a
        summary line) and raises SystemExit(1), aborting the process.
        Non-fatal issues (see _collect_validation_warnings) are printed
        as warnings regardless of whether errors also fired, but never
        cause a SystemExit - config that's merely worth a second look,
        not broken, should never block a real run.

    Returns:
        None, always - either validation passed (with or without
        warnings), or the process has already exited via SystemExit(1).
    """
    import betautils_log as bu_log
    logger = bu_log.get_logger()

    # The run-start line that says which of this person's own settings
    # are in effect, so a result that differs from someone else's has
    # its reason in the log.
    if bu_local.LOADED is not None:
        logger.info( bu_local.LOADED.describe() )

    warnings = _collect_validation_warnings()
    if warnings:
        logger.warning( "betaconfig.py validation warnings (not fatal, review when convenient):" )
        for warn in warnings:
            logger.warning( "  - %s"%(warn) )

    errors = _collect_validation_errors()
    if errors:
        logger.error( "betaconfig.py failed validation, fix the following and re-run:" )
        for err in errors:
            logger.error( "  - %s"%(err) )
        loaded = bu_local.LOADED
        if loaded is not None and loaded.applied:
            logger.error( "  (%s sets %s; an error naming one of those is about its value there, "
                          "not betaconfig.py's)"%( loaded.path, ', '.join( loaded.set_names() ) ) )
        raise SystemExit(1)

    logger.debug( "betaconfig.py validated: %d warning(s), 0 errors"%(len( warnings )) )
