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
        -> _collect_validation_errors()      # runs every sub-validator below,
                                              # each appending to one shared
                                              # errors list
            -> _validate_items_to_censor
            -> _validate_class_suppression
            -> _validate_item_overrides
            -> _validate_backend_item_overrides
            -> _validate_default_censor_style
            -> _validate_default_censor_shape
            -> _validate_type_default_shapes
            -> _validate_censor_overlap_strategy_coverage
            -> _validate_video_censor_fps
            -> _validate_picture_sizes
            -> _validate_input_delete_probability_range
            -> _validate_encode_preset
            -> _validate_preview_settings
            -> _validate_logging_settings
            -> _validate_stats_settings
            -> _validate_nn_batch_size
            -> _validate_min_prob_floor_consistency
            -> _validate_checkpoint_and_chunk_settings
        -> prints warnings (if any), then prints + raises SystemExit(1) if
           _collect_validation_errors returned anything, otherwise returns
           None

Splitting validation into one function per concern (rather than one long
function) is meant to make it easy to find "the code that checks X" when
X turns out to be wrong, and to make it safe to add a new check without
wading through unrelated ones. Every sub-validator appends error strings
into a single shared `errors` list, mirroring exactly how the original
single-function implementation built up that same list - the point of
this file is to check that same set of things in the same order, not to
change what's checked.
"""

import os
import random
import threading

import betaconfig
import betaconst
import betautils_detector as bu_detector


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
        key hashes a stable shape rather than a varying one.
    """
    override = bu_detector.get_item_overrides( label, backend_name )
    return {
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
    }


def _geometry_action_default():
    """DEFAULT_GEOMETRY_ACTION, imported late to avoid the import cycle."""
    import betautils_track as bu_track  # deferred: bu_track imports this module
    return bu_track.DEFAULT_GEOMETRY_ACTION


def get_parts_to_blur( backend_name=None ):
    """
    The per-label settings table the render pipeline actually uses.

    One entry per betaconfig.items_to_censor label, resolved against
    that label's item_overrides with betaconfig's defaults filling in
    the rest.

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
    for item in betaconfig.items_to_censor:
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


VALID_CENSOR_STYLES = { 'blur', 'pixel', 'bar', 'sticker', 'debug' }
VALID_CENSOR_SHAPES = { 'box', 'circle', 'ellipse' }
VALID_BLUR_METHODS = { 'gaussian', 'box', 'triple_box' }
VALID_PIXEL_PATTERNS = { 'square', 'mosaic', 'hex' }
# same vocabulary as censor_overlap_strategy's values (betaconfig.py) - a
# style dict's own 'merge' key overrides the type-level default there for
# just this one style variant. See collapse_boxes_for_style in
# betautils_censor.py.
VALID_MERGE_STRATEGIES = { 'none', 'single-pass', 'span' }
VALID_ENCODE_PRESETS = { 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow' }

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
    Validate the fields every censor style dict may carry, regardless of
    its 'type' (feather, weight, shape, merge, width/height_area_safety).

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
        errors.append( "%s has unrecognized key(s) %s for type '%s', likely a typo (valid keys: %s)"%(where, sorted(unknown_keys), style_type, sorted(allowed_keys)) )

    if 'feather' in style:
        feather = style['feather']
        if not isinstance( feather, (int, float) ) or not (0 <= feather <= 1):
            errors.append( "%s['feather'] must be a number between 0 and 1, got %r"%(where, feather) )
    if 'weight' in style:
        weight = style['weight']
        if not isinstance( weight, (int, float) ) or weight < 0:
            errors.append( "%s['weight'] must be a non-negative number, got %r"%(where, weight) )
    if 'shape' in style and style['shape'] not in VALID_CENSOR_SHAPES:
        errors.append( "%s['shape'] = '%s' is not one of %s"%(where, style['shape'], sorted(VALID_CENSOR_SHAPES)) )
    if 'merge' in style and style['merge'] not in VALID_MERGE_STRATEGIES:
        errors.append( "%s['merge'] = '%s' is not one of %s"%(where, style['merge'], sorted(VALID_MERGE_STRATEGIES)) )
    if 'width_area_safety' in style and not isinstance( style['width_area_safety'], (int, float) ):
        errors.append( "%s['width_area_safety'] must be a number, got %r"%(where, style['width_area_safety']) )
    if 'height_area_safety' in style and not isinstance( style['height_area_safety'], (int, float) ):
        errors.append( "%s['height_area_safety'] must be a number, got %r"%(where, style['height_area_safety']) )


def _check_blur_style_fields( style, where, errors ):
    """Validate 'method'/'strength', the fields specific to type: 'blur'."""
    if 'method' in style and style['method'] not in VALID_BLUR_METHODS:
        errors.append( "%s['method'] = '%s' is not one of %s"%(where, style['method'], sorted(VALID_BLUR_METHODS)) )
    if 'strength' in style and not isinstance( style['strength'], (int, float) ):
        errors.append( "%s['strength'] must be a number, got %r"%(where, style['strength']) )


def _check_pixel_style_fields( style, where, errors ):
    """Validate 'pattern'/'strength', the fields specific to type: 'pixel'."""
    if 'pattern' in style and style['pattern'] not in VALID_PIXEL_PATTERNS:
        errors.append( "%s['pattern'] = '%s' is not one of %s"%(where, style['pattern'], sorted(VALID_PIXEL_PATTERNS)) )
    if 'strength' in style and not isinstance( style['strength'], (int, float) ):
        errors.append( "%s['strength'] must be a number, got %r"%(where, style['strength']) )


def _check_bar_style_fields( style, where, errors ):
    """Validate 'color'/'span_extend'/'thickness', specific to type: 'bar'."""
    if 'color' in style and not ( isinstance( style['color'], (list, tuple) ) and len(style['color']) == 3 ):
        errors.append( "%s['color'] must be an (r,g,b) tuple, got %r"%(where, style['color']) )
    if 'span_extend' in style:
        span_extend = style['span_extend']
        if not isinstance( span_extend, (int, float) ) or not (0 <= span_extend <= 1):
            errors.append( "%s['span_extend'] must be a number between 0 (tight) and 1 (full frame width), got %r"%(where, span_extend) )
    if 'thickness' in style:
        thickness = style['thickness']
        # >1 is legitimately allowed (grows the bar past the detection
        # box, clamped to the frame - see bar_image in
        # betautils_censor.py), but a very large value is almost always
        # a typo (e.g. 35 meant to be 0.35) rather than intentional, so
        # it's flagged rather than silently accepted.
        if not isinstance( thickness, (int, float) ) or not (0 < thickness <= 3):
            errors.append( "%s['thickness'] must be a number greater than 0 (fraction of the box height the bar covers; 1.0 = full box, <1 = thinner, up to 3 = allowed but almost certainly not what you meant), got %r"%(where, thickness) )


def _check_sticker_style_fields( style, where, errors ):
    """Validate 'dir'/'images'/'scale', the fields specific to type: 'sticker'."""
    if 'dir' not in style and 'images' not in style:
        errors.append( "%s must set 'dir' and/or 'images' so there's something to composite"%(where) )
    if 'scale' in style and not isinstance( style['scale'], (int, float) ):
        errors.append( "%s['scale'] must be a number, got %r"%(where, style['scale']) )


# Type-specific field checkers, dispatched by style['type'] - 'debug'
# intentionally has no entry here since its only type-specific key
# ('color') is a plain tuple with no further constraints worth checking.
_TYPE_SPECIFIC_CHECKERS = {
    'blur':    _check_blur_style_fields,
    'pixel':   _check_pixel_style_fields,
    'bar':     _check_bar_style_fields,
    'sticker': _check_sticker_style_fields,
}


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
        errors.append( "%s must be a dict like {'type': 'blur', ...}, got %r"%(where, style) )
        return
    if 'type' not in style:
        errors.append( "%s is missing required key 'type'"%(where) )
        return
    style_type = style['type']
    if style_type not in VALID_CENSOR_STYLES:
        errors.append( "%s has unknown censor style type '%s' (valid: %s)"%(where, style_type, sorted(VALID_CENSOR_STYLES)) )
        return

    _check_style_common_fields( style, style_type, where, errors )

    type_specific_checker = _TYPE_SPECIFIC_CHECKERS.get( style_type )
    if type_specific_checker is not None:
        type_specific_checker( style, where, errors )


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
            errors.append( "%s is an empty list - needs at least one style dict"%(where) )
        for idx, one_style in enumerate( style ):
            _check_one_style_dict( one_style, "%s[%d]"%(where, idx), errors )
    else:
        _check_one_style_dict( style, where, errors )


# --- Top-level betaconfig.py section validators -----------------------------

def _validate_items_to_censor( errors, valid_classes ):
    """Check that items_to_censor is defined and every entry is a real class."""
    items_to_censor = getattr( betaconfig, 'items_to_censor', None )
    if items_to_censor is None:
        errors.append( "items_to_censor is not defined" )
        return
    for item in items_to_censor:
        if item not in valid_classes:
            errors.append( "items_to_censor contains unknown class '%s' (valid: %s)"%(item, sorted(valid_classes)) )


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
        errors.append( "betaconfig.class_suppression is no longer read - label "
            "vocabularies differ between models, so a shared ruleset can name classes a backend has "
            "never heard of. Move it into detector_backend[<name>]['class_suppression'], or delete "
            "it. Leaving it here would silently do nothing." )

    for backend_name in bu_detector.registered_backend_names():
        class_suppression = bu_detector.get_class_suppression( backend_name )
        for label, rules in class_suppression.items():
            is_list = isinstance( rules, list )
            where_base = "detector_backend[%r]['class_suppression']['%s']"%(backend_name, label)
            if label not in valid_classes:
                errors.append( "%s: '%s' is not a known class (valid: %s)"%(
                        "detector_backend[%r]['class_suppression']"%(backend_name), label, sorted(valid_classes)) )

            rule_list = rules if is_list else [ rules ]
            seen_suppressors = []
            for idx, rule in enumerate( rule_list ):
                where = "%s[%d]"%(where_base, idx) if is_list else where_base
                if not isinstance( rule, dict ):
                    errors.append( "%s must be a dict, got %r"%(where, rule) )
                    continue
                unknown_keys = set(rule.keys()) - known_rule_keys
                if unknown_keys:
                    errors.append( "%s has unrecognized key(s) %s, likely a typo (valid keys: %s)"%(where, sorted(unknown_keys), sorted(known_rule_keys)) )
                if 'suppressed_by' not in rule:
                    errors.append( "%s is missing required key 'suppressed_by'"%(where) )
                elif rule['suppressed_by'] not in valid_classes:
                    errors.append( "%s['suppressed_by'] = '%s' is not a known class (valid: %s)"%(where, rule['suppressed_by'], sorted(valid_classes)) )
                else:
                    if rule['suppressed_by'] in seen_suppressors:
                        errors.append( "%s has more than one rule with suppressed_by='%s' - only one will ever match, merge them"%(where_base, rule['suppressed_by']) )
                    seen_suppressors.append( rule['suppressed_by'] )
                if 'margin' not in rule:
                    errors.append( "%s is missing required key 'margin' - a missing margin silently defaults to "
                        "0.0 (the loosest possible gate) in apply_class_suppression, not a neutral 'off' value, "
                        "so it must be set explicitly (see derive_suppression_rules.py's propose_margin if you "
                        "need to derive one from real data)"%(where) )
                elif not isinstance( rule['margin'], (int, float) ):
                    errors.append( "%s['margin'] must be a number, got %r"%(where, rule['margin']) )
                elif rule['margin'] < 0:
                    # apply_class_suppression's check is 'other[score] - raw[score] >= margin' - a
                    # negative margin lets the suppressing label be LESS confident than the label
                    # it's overriding and still win, inverting the entire "require better evidence
                    # to override a real detection" premise the margin gate exists for. No known
                    # legitimate use case (confirmed with the user 2026-09-16) - always an error,
                    # unlike margin=0.0, which is only a warning (see _validate_class_suppression_zero_margin).
                    errors.append( "%s['margin'] = %r is negative - the suppressing label would be allowed to be "
                        "LESS confident than the label it's overriding and still suppress it, which defeats the "
                        "point of the margin gate. Use 0.0 if you genuinely want IoU alone to gate this rule "
                        "(that's a warning, not an error), never a negative number."%(where, rule['margin']) )
                if 'min_iou' not in rule:
                    errors.append( "%s is missing required key 'min_iou'"%(where) )
                elif not isinstance( rule['min_iou'], (int, float) ):
                    errors.append( "%s['min_iou'] must be a number, got %r"%(where, rule['min_iou']) )


def _is_fraction( value ):
    return isinstance( value, (int, float) ) and not isinstance( value, bool ) and 0 <= value <= 1


def _validate_class_promotion( errors, valid_classes ):
    """
    Check every registered backend's class_promotion ruleset.

    Unknown keys are errors, not warnings, because every key in a rule
    is a gate: a misspelled 'min_prob' silently becomes "no floor", and a
    misspelled 'requires' silently becomes "no evidence needed", which
    promotes every detection of the source label. Both fail open, in the
    direction of censoring things the author never meant to.

    A target that is not in items_to_censor is also an error: the rule
    would relabel boxes into a label nothing renders, which DELETES the
    source detection's censor if the source label was being censored.
    """
    import betautils_track as bu_track  # deferred: bu_track imports this module
    censored = set( getattr( betaconfig, 'items_to_censor', [] ) )

    for backend_name in bu_detector.registered_backend_names():
        promotion = bu_detector.get_class_promotion( backend_name )
        if not isinstance( promotion, dict ):
            errors.append( "detector_backend[%r]['class_promotion'] must be a dict, got %r"%(
                backend_name, promotion ) )
            continue
        for target, rules in promotion.items():
            where_base = "detector_backend[%r]['class_promotion']['%s']"%( backend_name, target )
            if target not in valid_classes:
                errors.append( "%s: '%s' is not a known class"%( where_base, target ) )
            elif target not in censored:
                errors.append( "%s: '%s' is not in items_to_censor, so promoted boxes would "
                               "never be rendered - and a censored source label would lose its "
                               "censor"%( where_base, target ) )
            rule_list = rules if isinstance( rules, list ) else [ rules ]
            for index, rule in enumerate( rule_list ):
                where = "%s[%d]"%( where_base, index ) if isinstance( rules, list ) else where_base
                if not isinstance( rule, dict ):
                    errors.append( "%s must be a dict, got %r"%( where, rule ) )
                    continue
                unknown = set( rule ) - bu_track.PROMOTION_RULE_KEYS
                if unknown:
                    errors.append( "%s has unrecognized key(s) %s (valid: %s)"%(
                        where, sorted( unknown ), sorted( bu_track.PROMOTION_RULE_KEYS ) ) )
                source = rule.get( 'from' )
                if source is None:
                    errors.append( "%s is missing required key 'from'"%( where ) )
                elif source not in valid_classes:
                    errors.append( "%s: 'from' %r is not a known class"%( where, source ) )
                elif source == target:
                    errors.append( "%s: 'from' is the same label as the target"%( where ) )
                for key in ( 'min_prob', 'duplicate_iou' ):
                    if key in rule and not _is_fraction( rule[key] ):
                        errors.append( "%s: %s must be a number in [0, 1], got %r"%(
                            where, key, rule[key] ) )
                if rule.get( 'requires_mode', 'any' ) not in ( 'any', 'all' ):
                    errors.append( "%s: requires_mode must be 'any' or 'all', got %r"%(
                        where, rule.get( 'requires_mode' ) ) )
                requires = rule.get( 'requires', [] )
                if not isinstance( requires, list ):
                    errors.append( "%s: requires must be a list of evidence dicts"%( where ) )
                    continue
                for req_index, requirement in enumerate( requires ):
                    req_where = "%s['requires'][%d]"%( where, req_index )
                    if not isinstance( requirement, dict ):
                        errors.append( "%s must be a dict, got %r"%( req_where, requirement ) )
                        continue
                    unknown = set( requirement ) - bu_track.PROMOTION_EVIDENCE_KEYS
                    if unknown:
                        errors.append( "%s has unrecognized key(s) %s (valid: %s)"%(
                            req_where, sorted( unknown ),
                            sorted( bu_track.PROMOTION_EVIDENCE_KEYS ) ) )
                    if requirement.get( 'label' ) not in valid_classes:
                        errors.append( "%s: label %r is not a known class"%(
                            req_where, requirement.get( 'label' ) ) )
                    for key in ( 'min_prob', 'min_iou', 'min_source_overlap' ):
                        if key in requirement and not _is_fraction( requirement[key] ):
                            errors.append( "%s: %s must be a number in [0, 1], got %r"%(
                                req_where, key, requirement[key] ) )
                    if 'overlap' in requirement:
                        if requirement['overlap'] != 'anywhere':
                            errors.append( "%s: overlap must be 'anywhere' (the only mode), "
                                           "got %r"%( req_where, requirement['overlap'] ) )
                        elif ( 'min_iou' in requirement
                               or 'min_source_overlap' in requirement ):
                            # Silently ignoring one of two contradictory
                            # settings is how a rule ends up not doing what
                            # its config says.
                            errors.append( "%s: overlap 'anywhere' means no spatial test, so "
                                           "min_iou / min_source_overlap cannot also be set"%(
                                               req_where, ) )


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
            errors.append( "%s has unrecognized key(s) %s"%( where_base, sorted( unknown ) ) )
        match_on = profiles.get( 'match_on', 'cuts_per_min' )
        if match_on not in known_match_signals:
            errors.append( "%s['match_on'] must be one of %s, got %r"%(
                where_base, list( known_match_signals ), match_on ) )
            match_on = None
        variants = profiles.get( 'variants' ) or {}
        default = profiles.get( 'default' )
        if default is not None and default not in variants:
            errors.append( "%s['default'] %r names no variant (have: %s)"%(
                where_base, default, sorted( variants ) ) )
        for name, variant in variants.items():
            where = "%s['variants'][%r]"%( where_base, name )
            if not isinstance( variant, dict ):
                errors.append( "%s must be a dict"%( where, ) )
                continue
            unknown = set( variant ) - known_variant_keys
            if unknown:
                errors.append( "%s has unrecognized key(s) %s (valid: %s)"%(
                    where, sorted( unknown ), sorted( known_variant_keys ) ) )
            for key in sorted( threshold_key_for.values() ):
                if key not in variant:
                    continue
                threshold = variant[key]
                if not isinstance( threshold, (int, float) ) \
                   or isinstance( threshold, bool ) or threshold < 0:
                    errors.append( "%s: %s must be a number >= 0, got %r"%(
                        where, key, threshold ) )
                elif key == 'min_short_shot_fraction' and threshold > 1:
                    errors.append( "%s: min_short_shot_fraction is a fraction of "
                                   "runtime from 0 to 1, got %r"%( where, threshold ) )
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
                    errors.append(
                        "%s sets %s but %s['match_on'] is %r, which reads %s. "
                        "This variant would never match automatically."%(
                            where, present_wrong[0], where_base, match_on,
                            threshold_key_for[match_on] ) )
            if 'manual_only' in variant and not isinstance( variant['manual_only'], bool ):
                errors.append( "%s: manual_only must be True/False, got %r"%(
                    where, variant['manual_only'] ) )
            if 'video_censor_fps' in variant:
                fps = variant['video_censor_fps']
                if not isinstance( fps, (int, float) ) or isinstance( fps, bool ) or fps <= 0:
                    errors.append( "%s: video_censor_fps must be a positive number, got %r"%(
                        where, fps ) )
            for label, overrides in ( variant.get( 'item_overrides' ) or {} ).items():
                if not isinstance( overrides, dict ):
                    errors.append( "%s['item_overrides'][%r] must be a dict"%( where, label ) )
                    continue
                unknown = set( overrides ) - detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
                if unknown:
                    errors.append( "%s['item_overrides'][%r] has unrecognized key(s) %s"%(
                        where, label, sorted( unknown ) ) )

    # Path rules must name real variants too, for the same reason.
    mapping = getattr( betaconfig, 'profile_by_path_pattern', None )
    if mapping is not None:
        if not isinstance( mapping, dict ):
            errors.append( "profile_by_path_pattern must be a dict of "
                           "{glob: profile_name}, got %r"%( type( mapping ).__name__, ) )
        else:
            selected = detector.selected_backend_name()
            variants = detector.get_profiles( selected ).get( 'variants' ) or {}
            for pattern, name in mapping.items():
                if not isinstance( pattern, str ) or not pattern:
                    errors.append( "profile_by_path_pattern has a non-string "
                                   "pattern %r"%( pattern, ) )
                if name not in variants:
                    errors.append(
                        "profile_by_path_pattern[%r] is %r, which names no variant for "
                        "backend %r (have: %s)"%( pattern, name, selected,
                                                  sorted( variants ) or 'none' ) )

    # A forced profile that names nothing is an error, not a fallback.
    # Silently rendering unprofiled after --profile comp was typed is the
    # failure where the run looks like it worked and the output is wrong.
    forced = getattr( betaconfig, 'force_structure_profile', None )
    if forced:
        selected = detector.selected_backend_name()
        variants = detector.get_profiles( selected ).get( 'variants' ) or {}
        if forced not in variants:
            errors.append(
                "force_structure_profile (--profile) is %r, which names no variant for "
                "backend %r (have: %s)"%( forced, selected, sorted( variants ) or 'none' ) )


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
    explicit user direction (2026-09-16): a bare tie (margin=0.0,
    "suppressing label's score just has to be >= the suppressed
    label's") is too permissive on its own - the user hand-raised
    every nudenet_v3 rule that had been exactly 0.0 up to 0.10, and
    wants a future edit that reintroduces exactly 0.0 (most likely by
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
    return warnings


def _check_item_override_values( override, where, errors ):
    """
    Check one already-resolved item_overrides-shaped dict (either the
    shared betaconfig.item_overrides[label] block - which, as of
    2026-09-17, should only ever actually HAVE censor_style/censor_shape
    set, since every backend-tunable key now lives in each backend's own
    block instead - or the effective per-backend-merged result from
    bu_detector.get_item_overrides) for individually well-formed values -
    censor_style, censor_shape, interpolation_max_gap,
    interpolation_enabled, paired_style, track_max_gap and its
    relationship to interpolation_max_gap, match_distance_multiplier,
    paired_style_max_distance. Every check here is guarded by "if <key>
    in override", so calling this against the shared block is safe even
    though it will now typically only ever find censor_style/
    censor_shape present - shared between _validate_item_overrides (the
    shared block) and _validate_backend_item_overrides (each backend's
    resolved result, where the tunable keys actually do show up) so the
    same checks apply to both, not just whichever one happens to be
    selected.
    """
    if 'censor_style' in override:
        _check_censor_style( override['censor_style'], "%s['censor_style']"%(where), errors )
    if 'censor_shape' in override and override['censor_shape'] not in VALID_CENSOR_SHAPES:
        errors.append( "%s['censor_shape'] = '%s' is not one of %s"%(where, override['censor_shape'], sorted(VALID_CENSOR_SHAPES)) )
    if 'interpolation_max_gap' in override and not isinstance( override['interpolation_max_gap'], (int, float) ):
        errors.append( "%s['interpolation_max_gap'] must be a number, got %r"%(where, override['interpolation_max_gap']) )
    if 'interpolation_enabled' in override and not isinstance( override['interpolation_enabled'], bool ):
        errors.append( "%s['interpolation_enabled'] must be True/False, got %r"%(where, override['interpolation_enabled']) )
    if 'paired_style' in override and not isinstance( override['paired_style'], bool ):
        errors.append( "%s['paired_style'] must be True/False, got %r"%(where, override['paired_style']) )
    if 'track_max_gap' in override:
        track_max_gap = override['track_max_gap']
        if not isinstance( track_max_gap, (int, float) ) or track_max_gap <= 0:
            errors.append( "%s['track_max_gap'] must be a positive number (seconds), got %r"%(where, track_max_gap) )
        # track_max_gap tighter than this label's own interpolation_max_gap
        # isn't invalid, but it silently makes that interpolation_max_gap
        # partly moot (a gap has to pass the track_max_gap test before
        # interpolation ever gets a chance to fill it) - see the
        # track_max_gap comment in betatv.py's smooth_boxes for the real
        # bug this distinction fixed. Flagged as a likely-unintentional
        # combination rather than an error.
        elif isinstance( track_max_gap, (int, float) ) and track_max_gap > 0:
            label_interp_max_gap = override.get( 'interpolation_max_gap', betaconfig.default_interpolation_max_gap )
            if isinstance( label_interp_max_gap, (int, float) ) and track_max_gap < label_interp_max_gap:
                errors.append( "%s['track_max_gap'] (%r) is less than this label's interpolation_max_gap (%r) - interpolation_max_gap will never get the chance to matter for gaps between track_max_gap and interpolation_max_gap, since the track hard-resets before interpolation is ever considered. This may be intentional, but if not, raise track_max_gap to at least interpolation_max_gap."%(where, track_max_gap, label_interp_max_gap) )
    if 'match_distance_multiplier' in override:
        match_distance_multiplier = override['match_distance_multiplier']
        if not isinstance( match_distance_multiplier, (int, float) ) or match_distance_multiplier <= 0:
            errors.append( "%s['match_distance_multiplier'] must be a positive number, got %r"%(where, match_distance_multiplier) )
    if 'paired_style_max_distance' in override:
        paired_style_max_distance = override['paired_style_max_distance']
        if not isinstance( paired_style_max_distance, (int, float) ) or paired_style_max_distance <= 0:
            errors.append( "%s['paired_style_max_distance'] must be a positive number, got %r"%(where, paired_style_max_distance) )
    if 'paired_style_tiebreak_margin' in override:
        tiebreak = override['paired_style_tiebreak_margin']
        if not isinstance( tiebreak, (int, float) ) or not ( 0 <= tiebreak <= 1 ):
            errors.append( "%s['paired_style_tiebreak_margin'] must be between 0 and 1 (a fraction of the "
                "second-nearest candidate's distance), got %r"%(where, tiebreak) )

    # --- score hysteresis ---
    if 'min_prob_continue' in override:
        min_prob_continue = override['min_prob_continue']
        if min_prob_continue is not None:
            if not isinstance( min_prob_continue, (int, float) ) or not ( 0 < min_prob_continue < 1 ):
                errors.append( "%s['min_prob_continue'] must be None or a number strictly between 0 and 1, "
                    "got %r"%(where, min_prob_continue) )
            else:
                label_min_prob = override.get( 'min_prob', betaconfig.default_min_prob )
                if isinstance( label_min_prob, (int, float) ) and min_prob_continue > label_min_prob:
                    errors.append( "%s['min_prob_continue'] (%r) is ABOVE this label's min_prob (%r), which "
                        "inverts hysteresis: the bar to keep tracking something would be higher than the bar "
                        "to start. min_prob_continue is the lower, 'stay' threshold - set it below min_prob "
                        "(or to None to disable hysteresis)."%(where, min_prob_continue, label_min_prob) )
                global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
                if isinstance( global_min_prob, (int, float) ) and min_prob_continue <= global_min_prob:
                    errors.append( "%s['min_prob_continue'] (%r) is at or below global_min_prob (%r) - nothing "
                        "that low ever reaches it, so hysteresis would extend no further than the global floor "
                        "already does. Raise it, or lower global_min_prob."%(
                            where, min_prob_continue, global_min_prob ) )

    # --- track confirmation ---
    if 'min_track_hits' in override:
        min_track_hits = override['min_track_hits']
        if not isinstance( min_track_hits, int ) or isinstance( min_track_hits, bool ) or min_track_hits < 1:
            errors.append( "%s['min_track_hits'] must be an integer >= 1 (1 means 'render every track', the "
                "historical behaviour), got %r"%(where, min_track_hits) )
        elif min_track_hits > 10:
            errors.append( "%s['min_track_hits'] = %r would require %r consecutive-ish detections before ANY "
                "censoring appears for a new instance. At video_censor_fps=%r that is about %.2fs of visible, "
                "uncensored footage on every new appearance - almost certainly not intended."%(
                    where, min_track_hits, min_track_hits,
                    getattr( betaconfig, 'video_censor_fps', 9 ),
                    min_track_hits / max( getattr( betaconfig, 'video_censor_fps', 9 ), 1 ) ) )

    # --- geometry sanity filter ---
    for area_key in ( 'min_area_fraction', 'max_area_fraction' ):
        if area_key in override and override[area_key] is not None:
            value = override[area_key]
            if not isinstance( value, (int, float) ) or not ( 0 <= value <= 1 ):
                errors.append( "%s['%s'] must be None or a fraction of the frame's area between 0 and 1, "
                    "got %r"%(where, area_key, value) )
    min_area = override.get( 'min_area_fraction' )
    max_area = override.get( 'max_area_fraction' )
    if ( isinstance( min_area, (int, float) ) and isinstance( max_area, (int, float) )
            and min_area >= max_area ):
        errors.append( "%s['min_area_fraction'] (%r) is not below ['max_area_fraction'] (%r) - no box can "
            "satisfy both, so every detection of this label violates a bound (what that then DOES depends "
            "on geometry_action)."%(where, min_area, max_area) )

    if 'geometry_action' in override:
        import betautils_track as bu_track  # deferred: bu_track imports this module
        action = override['geometry_action']
        if isinstance( action, dict ):
            unknown_bounds = set( action ) - { 'min', 'max' }
            if unknown_bounds:
                errors.append( "%s['geometry_action'] mapping keys must be 'min'/'max', got %s"%(
                    where, sorted( unknown_bounds ) ) )
            bad = [ value for value in action.values()
                    if value not in bu_track.GEOMETRY_ACTIONS ]
            if bad:
                errors.append( "%s['geometry_action'] values must be one of %s, got %r"%(
                    where, sorted( bu_track.GEOMETRY_ACTIONS ), bad ) )
        elif action not in bu_track.GEOMETRY_ACTIONS:
            errors.append( "%s['geometry_action'] must be one of %s, or a {'min':..,'max':..} "
                           "mapping, got %r"%(
                               where, sorted( bu_track.GEOMETRY_ACTIONS ), action ) )

    for aspect_key in ( 'min_aspect_ratio', 'max_aspect_ratio' ):
        if aspect_key in override and override[aspect_key] is not None:
            value = override[aspect_key]
            if not isinstance( value, (int, float) ) or value <= 0:
                errors.append( "%s['%s'] must be None or a positive width/height ratio, got %r"%(
                    where, aspect_key, value ) )
    min_aspect = override.get( 'min_aspect_ratio' )
    max_aspect = override.get( 'max_aspect_ratio' )
    if ( isinstance( min_aspect, (int, float) ) and isinstance( max_aspect, (int, float) )
            and min_aspect >= max_aspect ):
        errors.append( "%s['min_aspect_ratio'] (%r) is not below ['max_aspect_ratio'] (%r) - no box can satisfy "
            "both, so every detection of this label violates a bound (what that then DOES depends on "
            "geometry_action)."%(where, min_aspect, max_aspect) )


def _validate_item_overrides( errors, valid_classes ):
    """
    Check the SHARED item_overrides block: every key is a real class,
    every value is a dict containing ONLY censor_style/censor_shape -
    the sole keys that stay shared across backends (pure rendering
    choices, not detection-derived - see bu_detector.
    ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS's own comment).

    2026-09-17: every backend-tunable key (min_prob, width_area_safety,
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
            errors.append( "item_overrides key '%s' is not a known class (valid: %s)"%(label, sorted(valid_classes)) )
        if not isinstance( override, dict ):
            errors.append( "%s must be a dict, got %r"%(where, override) )
            continue
        backend_tunable_keys_present = set(override.keys()) & bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
        if backend_tunable_keys_present:
            errors.append( "%s has backend-tunable key(s) %s in the SHARED item_overrides block - these moved to "
                "each backend's own detector_backend[<name>]['item_overrides']['%s'] block (2026-09-17) and are no "
                "longer read from here at all (bu_detector.get_item_overrides only reads backend-tunable keys from "
                "a backend's own block), so this is silently doing nothing - move %s into detector_backend['nudenet_v3']"
                "['item_overrides']['%s'] and detector_backend['retinanet_v2']['item_overrides']['%s'] (with "
                "whatever value is actually correct for each model - don't just copy the same number to both) "
                "instead."%(where, sorted(backend_tunable_keys_present), label, sorted(backend_tunable_keys_present), label, label) )
        unknown_keys = set(override.keys()) - known_shared_keys - bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
        if unknown_keys:
            errors.append( "%s has unrecognized key(s) %s, likely a typo (valid keys here: %s)"%(where, sorted(unknown_keys), sorted(known_shared_keys)) )
        _check_item_override_values( override, where, errors )

    return item_overrides


def _validate_backend_item_overrides( errors, valid_classes ):
    """
    Check every registered backend's per-backend item_overrides block
    (detector_backend[<name>]['item_overrides'][label]) - added
    2026-09-16 alongside bu_detector.get_item_overrides, same reasoning
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
            errors.append( "detector_backend[%r]['item_overrides'] must be a dict, got %r"%(backend_name, backend_item_overrides) )
            continue
        for label, override in backend_item_overrides.items():
            where = "detector_backend[%r]['item_overrides']['%s']"%(backend_name, label)
            if label not in valid_classes:
                errors.append( "detector_backend[%r]['item_overrides'] key '%s' is not a known class (valid: %s)"%(backend_name, label, sorted(valid_classes)) )
            if not isinstance( override, dict ):
                errors.append( "%s must be a dict, got %r"%(where, override) )
                continue
            unknown_keys = set(override.keys()) - bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
            if unknown_keys:
                errors.append( "%s has key(s) %s that aren't backend-tunable (censor_style/censor_shape always come from the shared item_overrides block - see ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS) - valid keys here: %s"%(where, sorted(unknown_keys), sorted(bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS)) )

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
            _check_item_override_values( resolved, where, errors )

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
        errors.append( "detector_backend[%r]['variants'] must be a dict, got %r"%(
            backend_name, variants_block ) )
        return

    declared = set( bu_detector.variant_names( backend_name ) )
    for variant_name, variant_block in variants_block.items():
        where_variant = "detector_backend[%r]['variants'][%r]"%(backend_name, variant_name)
        if declared and variant_name not in declared:
            errors.append( "%s is not a variant %s declares (valid: %s) - a block under an "
                           "unknown variant name never resolves, so this would silently do "
                           "nothing"%( where_variant, backend_name, sorted( declared ) ) )
            continue
        if not isinstance( variant_block, dict ):
            errors.append( "%s must be a dict, got %r"%(where_variant, variant_block) )
            continue

        variant_item_overrides = variant_block.get( 'item_overrides', {} )
        if not isinstance( variant_item_overrides, dict ):
            errors.append( "%s['item_overrides'] must be a dict, got %r"%(
                where_variant, variant_item_overrides ) )
            continue

        for label, override in variant_item_overrides.items():
            where = "%s['item_overrides']['%s']"%(where_variant, label)
            if label not in valid_classes:
                errors.append( "%s['item_overrides'] key '%s' is not a known class (valid: %s)"%(
                    where_variant, label, sorted( valid_classes ) ) )
            if not isinstance( override, dict ):
                errors.append( "%s must be a dict, got %r"%(where, override) )
                continue
            unknown_keys = set( override.keys() ) - bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
            if unknown_keys:
                errors.append( "%s has key(s) %s that aren't backend-tunable - valid keys "
                               "here: %s"%( where, sorted( unknown_keys ),
                                            sorted( bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS ) ) )

        for label in shared_labels | set( variant_item_overrides.keys() ):
            resolved = bu_detector.get_item_overrides( label, backend_name, variant_name )
            where = "%s's resolved item_overrides['%s']"%(where_variant, label)
            _check_item_override_values( resolved, where, errors )


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
        errors.append( "default_censor_style is not defined (or every option is commented out)" )
    else:
        _check_censor_style( default_censor_style, "default_censor_style", errors )
    return default_censor_style


def _validate_default_censor_shape( errors ):
    """Check that default_censor_shape (if set) is a recognized shape."""
    default_censor_shape = getattr( betaconfig, 'default_censor_shape', 'box' )
    if default_censor_shape not in VALID_CENSOR_SHAPES:
        errors.append( "default_censor_shape = '%s' is not one of %s"%(default_censor_shape, sorted(VALID_CENSOR_SHAPES)) )


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
        errors.append( "type_default_shapes must be a dict of {style_type: shape}, got %r"%(type_default_shapes) )
        return
    for style_type, shape in type_default_shapes.items():
        if style_type not in VALID_CENSOR_STYLES:
            errors.append( "type_default_shapes key '%s' is not a known censor style type (valid: %s)"%(style_type, sorted(VALID_CENSOR_STYLES)) )
        if shape not in VALID_CENSOR_SHAPES:
            errors.append( "type_default_shapes['%s'] = '%s' is not one of %s"%(style_type, shape, sorted(VALID_CENSOR_SHAPES)) )


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
            errors.append( "censor_overlap_strategy is missing an entry for '%s', which is used by default_censor_style/item_overrides"%(style_type) )


def _validate_video_censor_fps( errors ):
    """Check that video_censor_fps is a positive number."""
    video_censor_fps = getattr( betaconfig, 'video_censor_fps', None )
    if not isinstance( video_censor_fps, (int, float) ) or video_censor_fps <= 0:
        errors.append( "video_censor_fps must be a positive number, got %r"%(video_censor_fps) )


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
        errors.append( "betaconfig.picture_sizes is no longer read - a detection "
            "size is a property of the model, so move this value into "
            "detector_backend[<name>]['picture_sizes'], or delete it and let the adapter's native "
            "size apply. Leaving it here would silently do nothing." )

    for backend_name in bu_detector.registered_backend_names():
        resolved = bu_detector.get_picture_sizes( backend_name )
        where = "detector_backend[%r]'s resolved picture_sizes"%(backend_name)
        if not resolved:
            errors.append( "%s is empty - set detector_backend[%r]['picture_sizes'] or "
                "detector_backend['defaults']['picture_sizes'], or give the adapter a "
                "native_picture_sizes()"%( where, backend_name ) )
            continue
        if not all( isinstance( size, int ) and size >= 0 for size in resolved ):
            errors.append( "%s must be non-negative ints, got %r"%(where, resolved) )
        if len( set( resolved ) ) != len( resolved ):
            errors.append( "%s has duplicate entries (%r) - the model would run twice on identical "
                "input and every detection would be duplicated"%(where, resolved) )


def _validate_blur_approximation( errors ):
    """Check the fast-blur approximation settings."""
    approximate = getattr( betaconfig, 'blur_fast_approximation', True )
    if not isinstance( approximate, bool ):
        errors.append( "blur_fast_approximation must be True/False, got %r"%(approximate,) )
    min_kernel = getattr( betaconfig, 'blur_approximation_min_kernel', 21 )
    if not isinstance( min_kernel, int ) or min_kernel < 3 or min_kernel % 2 == 0:
        errors.append( "blur_approximation_min_kernel must be an odd integer >= 3 (a Gaussian kernel size), "
            "got %r"%(min_kernel,) )
    margin = getattr( betaconfig, 'blur_edge_margin', True )
    if not isinstance( margin, bool ):
        errors.append( "blur_edge_margin must be True/False, got %r"%(margin,) )
    motion = getattr( betaconfig, 'render_motion_interpolation', True )
    if not isinstance( motion, bool ):
        errors.append( "render_motion_interpolation must be True/False, got %r"%(motion,) )
    span = getattr( betaconfig, 'render_motion_max_span_seconds', None )
    if span is not None and ( not isinstance( span, (int, float) )
                              or isinstance( span, bool ) or span <= 0 ):
        errors.append( "render_motion_max_span_seconds must be a positive number of "
                       "seconds or None (meaning two sampling intervals), got %r"%(span,) )
    growth = getattr( betaconfig, 'render_motion_collapse_max_growth', 1.15 )
    if ( not isinstance( growth, (int, float) ) or isinstance( growth, bool )
         or growth < 1.0 ):
        # Below 1.0 no pair of samples could ever collapse (a union is never
        # smaller than the box it contains), so the flicker fix would be
        # silently off while still reading as enabled.
        errors.append( "render_motion_collapse_max_growth must be a number >= 1.0 "
                       "(the union of two samples is never smaller than either, "
                       "so anything below 1.0 disables the collapse entirely), "
                       "got %r"%(growth,) )
    window = getattr( betaconfig, 'render_motion_size_window_seconds', 0.25 )
    if window is not None and ( not isinstance( window, (int, float) )
                                or isinstance( window, bool ) or window < 0 ):
        errors.append( "render_motion_size_window_seconds must be a non-negative "
                       "number of seconds (0 means the whole track), or None to "
                       "leave box sizes alone, got %r"%(window,) )



def _validate_cross_size_dedup( errors ):
    """Check the cross-size dedup settings."""
    settings = getattr( betaconfig, 'cross_size_dedup', {} )
    if not isinstance( settings, dict ):
        errors.append( "cross_size_dedup must be a dict, got %r"%(settings,) )
        return
    unknown = set( settings.keys() ) - { 'enabled', 'iou_threshold' }
    if unknown:
        errors.append( "cross_size_dedup has unrecognized key(s) %s (valid: enabled, iou_threshold)"%(
            sorted( unknown ),) )
    if 'enabled' in settings and not isinstance( settings['enabled'], bool ):
        errors.append( "cross_size_dedup['enabled'] must be True/False, got %r"%(settings['enabled'],) )
    if 'iou_threshold' in settings:
        threshold = settings['iou_threshold']
        if not isinstance( threshold, (int, float) ) or not ( 0 < threshold <= 1 ):
            errors.append( "cross_size_dedup['iou_threshold'] must be between 0 (exclusive) and 1, "
                "got %r"%(threshold,) )


def _validate_render_settings( errors ):
    """Check the render/encode settings introduced with the parallel renderer."""
    workers = getattr( betaconfig, 'render_workers', 0 )
    if not isinstance( workers, int ) or isinstance( workers, bool ) or workers < 0:
        errors.append( "render_workers must be a non-negative integer (0 = auto-detect from CPU count), "
            "got %r"%(workers,) )

    container = getattr( betaconfig, 'render_chunk_container', 'mkv' )
    if container not in ( 'mkv', 'mp4' ):
        errors.append( "render_chunk_container must be 'mkv' or 'mp4', got %r - these are the two the "
            "concat demuxer handles reliably for stream-copied H.264"%(container,) )

    codec = getattr( betaconfig, 'encode_video_codec', 'libx264' )
    if not isinstance( codec, str ) or not codec:
        errors.append( "encode_video_codec must be a non-empty ffmpeg encoder name, got %r"%(codec,) )

    crf = getattr( betaconfig, 'encode_crf', 17 )
    if not isinstance( crf, int ) or isinstance( crf, bool ) or not ( 0 <= crf <= 51 ):
        errors.append( "encode_crf must be an integer between 0 (lossless) and 51 (worst), got %r"%(crf,) )

    verify = getattr( betaconfig, 'render_verify_frame_counts', True )
    if not isinstance( verify, bool ):
        errors.append( "render_verify_frame_counts must be True/False, got %r"%(verify,) )


def _validate_input_delete_probability_range( errors ):
    """Check that input_delete_probability is between 0 and 1."""
    if not ( 0 <= getattr( betaconfig, 'input_delete_probability', 0 ) <= 1 ):
        errors.append( "input_delete_probability must be between 0 and 1, got %r"%(betaconfig.input_delete_probability) )


def _validate_encode_preset( errors ):
    """Check that encode_preset is a known x264 preset name."""
    encode_preset = getattr( betaconfig, 'encode_preset', 'slow' )
    if encode_preset not in VALID_ENCODE_PRESETS:
        errors.append( "encode_preset = '%s' is not a known x264 preset (valid: %s)"%(encode_preset, sorted(VALID_ENCODE_PRESETS)) )


def _validate_preview_settings( errors ):
    """
    When preview_mode_enabled is on, check preview_max_seconds,
    preview_encode_preset, preview_start_seconds, and
    preview_random_slice are all well-formed. No-op entirely when
    preview mode is off.
    """
    if not getattr( betaconfig, 'preview_mode_enabled', False ):
        return
    preview_max_seconds = getattr( betaconfig, 'preview_max_seconds', 20 )
    if not isinstance( preview_max_seconds, (int, float) ) or preview_max_seconds <= 0:
        errors.append( "preview_max_seconds must be a positive number, got %r"%(preview_max_seconds) )
    preview_encode_preset = getattr( betaconfig, 'preview_encode_preset', 'ultrafast' )
    if preview_encode_preset not in VALID_ENCODE_PRESETS:
        errors.append( "preview_encode_preset = '%s' is not a known x264 preset (valid: %s)"%(preview_encode_preset, sorted(VALID_ENCODE_PRESETS)) )
    preview_start_seconds = getattr( betaconfig, 'preview_start_seconds', None )
    if preview_start_seconds is not None and ( not isinstance( preview_start_seconds, (int, float) ) or preview_start_seconds < 0 ):
        errors.append( "preview_start_seconds must be None or a non-negative number, got %r"%(preview_start_seconds) )
    preview_random_slice = getattr( betaconfig, 'preview_random_slice', False )
    if not isinstance( preview_random_slice, bool ):
        errors.append( "preview_random_slice must be True/False, got %r"%(preview_random_slice) )


def _validate_logging_settings( errors ):
    """
    Check console_level always (the console handler is attached whether
    or not file logging is on), and log_level/log_path only when
    logging_enabled is set.
    """
    import betautils_log as bu_log
    valid_levels = sorted( bu_log.LEVEL_NAMES.keys() )

    console_level = getattr( betaconfig, 'console_level', 'info' )
    if console_level not in bu_log.LEVEL_NAMES:
        errors.append( "console_level = %r is not one of %s"%(console_level, valid_levels) )

    if not getattr( betaconfig, 'logging_enabled', False ):
        return
    log_level = getattr( betaconfig, 'log_level', 'info' )
    if log_level not in bu_log.LEVEL_NAMES:
        errors.append( "log_level = %r is not one of %s"%(log_level, valid_levels) )
    if not getattr( betaconfig, 'log_path', None ):
        errors.append( "logging_enabled is True but log_path is not set" )


def _validate_stats_settings( errors ):
    """
    When stats_enabled is on, check stats_path is set. No-op entirely
    when stats are off.
    """
    if not getattr( betaconfig, 'stats_enabled', False ):
        return
    if not getattr( betaconfig, 'stats_path', None ):
        errors.append( "stats_enabled is True but stats_path is not set" )


def _validate_nn_batch_size( errors ):
    """
    Check that every registered backend's resolved nn_batch_size (see
    betautils_detector.get_nn_batch_size) is a positive integer - not
    just the currently selected backend's, so a typo in an unselected
    backend's override is still caught before it's ever relied on (e.g.
    switching betaconfig.detector_backend['selected'] later shouldn't be
    the first time a bad value there is noticed).

    Unlike picture_sizes and class_suppression, nn_batch_size keeps its
    detector_backend['defaults'] tier: it is a VRAM/throughput knob, not
    a statement about what the model means, so one sensible value can
    legitimately cover a backend that has not expressed a preference.
    Only the module-level betaconfig.nn_batch_size tier was removed.
    """
    if getattr( betaconfig, 'nn_batch_size', None ) is not None:
        errors.append( "betaconfig.nn_batch_size is no longer read - move it to "
            "detector_backend['defaults']['nn_batch_size'] for a value shared by every backend, or "
            "to detector_backend[<name>]['nn_batch_size'] for one model. Leaving it here would "
            "silently do nothing." )

    for name in bu_detector.registered_backend_names():
        nn_batch_size = bu_detector.get_nn_batch_size( name )
        if not isinstance( nn_batch_size, int ) or nn_batch_size < 1:
            errors.append( "detector_backend[%r]'s nn_batch_size must be a positive integer, got %r"%(
                    name, nn_batch_size ) )


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
        errors.append( "global_min_prob must be a number between 0 (inclusive) and 1 (exclusive), got %r"%(global_min_prob) )
        return

    default_min_prob = getattr( betaconfig, 'default_min_prob', None )
    if isinstance( default_min_prob, (int, float) ) and default_min_prob <= global_min_prob:
        errors.append( "default_min_prob (%r) is at or below global_min_prob (%r) - it can never reject anything, since nothing that low ever survives to be checked against it. Raise default_min_prob above global_min_prob, or lower global_min_prob."%(default_min_prob, global_min_prob) )

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
                    errors.append( "detector_backend[%r]'s resolved item_overrides['%s']['min_prob'] (%r) is at or below global_min_prob (%r) - it can never reject anything for this label under this backend, since nothing that low ever survives to be checked against it. Raise it above global_min_prob, or lower global_min_prob."%(backend_name, label, label_min_prob, global_min_prob) )


def _validate_checkpoint_and_chunk_settings( errors ):
    """
    Check detection_checkpoint_frames, render_chunk_seconds,
    ffmpeg_max_retries, and ffmpeg_retry_backoff_seconds are all
    well-formed (non-negative numbers of the right type).
    """
    detection_checkpoint_frames = getattr( betaconfig, 'detection_checkpoint_frames', 500 )
    if not isinstance( detection_checkpoint_frames, int ) or detection_checkpoint_frames < 0:
        errors.append( "detection_checkpoint_frames must be a non-negative integer (0 disables checkpointing), got %r"%(detection_checkpoint_frames) )

    render_chunk_seconds = getattr( betaconfig, 'render_chunk_seconds', 600 )
    if not isinstance( render_chunk_seconds, (int, float) ) or render_chunk_seconds < 0:
        errors.append( "render_chunk_seconds must be a non-negative number (0 disables chunking), got %r"%(render_chunk_seconds) )

    ffmpeg_max_retries = getattr( betaconfig, 'ffmpeg_max_retries', 2 )
    if not isinstance( ffmpeg_max_retries, int ) or ffmpeg_max_retries < 0:
        errors.append( "ffmpeg_max_retries must be a non-negative integer, got %r"%(ffmpeg_max_retries) )

    ffmpeg_retry_backoff_seconds = getattr( betaconfig, 'ffmpeg_retry_backoff_seconds', 5 )
    if not isinstance( ffmpeg_retry_backoff_seconds, (int, float) ) or ffmpeg_retry_backoff_seconds < 0:
        errors.append( "ffmpeg_retry_backoff_seconds must be a non-negative number, got %r"%(ffmpeg_retry_backoff_seconds) )


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
        errors.append( "detector_backend must be a dict, got %r"%(backend_setting,) )
        return

    known_names = set( bu_detector.registered_backend_names() )
    unknown_sections = set( backend_setting.keys() ) - known_names - { 'selected', 'defaults' }
    if unknown_sections:
        errors.append( "detector_backend has section(s) %s that are neither a registered adapter nor "
            "'selected'/'defaults' - a typo here is silently ignored at runtime, which is why it is an "
            "error here (registered adapters: %s)"%(
                sorted( unknown_sections ), ', '.join( sorted( known_names ) ) ) )

    backend = bu_detector.selected_backend_name()
    if backend not in known_names:
        errors.append( "detector_backend['selected'] %r is not a registered detector adapter - valid "
            "choices: %s"%(backend, ', '.join( sorted( known_names ) )) )
        return

    # Validate EVERY registered backend's own block, not just the
    # selected one, so switching 'selected' is never the first time a
    # typo in the other block is noticed.
    for name in sorted( known_names ):
        module = bu_detector.get_detector( name )
        if hasattr( module, 'validate_backend_config' ):
            module.validate_backend_config( bu_detector.get_backend_config( name ), errors )


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

    _validate_items_to_censor( errors, valid_classes )
    _validate_class_suppression( errors, valid_classes )
    _validate_class_promotion( errors, valid_classes )
    _validate_profiles( errors )
    item_overrides = _validate_item_overrides( errors, valid_classes )
    _validate_backend_item_overrides( errors, valid_classes )
    default_censor_style = _validate_default_censor_style( errors )
    _validate_default_censor_shape( errors )
    _validate_type_default_shapes( errors )
    _validate_censor_overlap_strategy_coverage( errors, default_censor_style, item_overrides )
    _validate_video_censor_fps( errors )
    _validate_picture_sizes( errors )
    _validate_input_delete_probability_range( errors )
    _validate_encode_preset( errors )
    _validate_preview_settings( errors )
    _validate_logging_settings( errors )
    _validate_stats_settings( errors )
    _validate_nn_batch_size( errors )
    _validate_min_prob_floor_consistency( errors )
    _validate_checkpoint_and_chunk_settings( errors )
    _validate_blur_approximation( errors )
    _validate_cross_size_dedup( errors )
    _validate_render_settings( errors )
    _validate_detector_backend( errors )

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
        raise SystemExit(1)

    logger.debug( "betaconfig.py validated: %d warning(s), 0 errors"%(len( warnings )) )
