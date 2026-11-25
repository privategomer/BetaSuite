"""
betautils_settings.py - the settings registry: one entry for every key
betaconfig.py has or the code reads.

Each entry says what a value may be (kind, range, choices, whether None
is allowed and what it means), where it may be set (scope, from the
placeholders in its path), whether the UI shows it by default (level)
and on which screen (SCREEN_RULES), which settings must be set for it
to have any effect (REQUIRES_RULES),
what changing it costs (the cache keys it moves, and under which
conditions; and the keys it can move for some files only, by changing
which structure profile or preview slice a file gets), and where its
words live (a catalog tip key and a
CONFIG_REFERENCE.md anchor).

Paths are tuples. A segment in angle brackets matches any key of that
map: '<backend>' a detector backend, '<label>' a detection class,
'<model_variant>' a model variant, '<profile>' a structure profile,
'<style_type>' a censor style type, '<person_backend>' a person
segmentation backend, '<folder>' and '<pattern>' a user-chosen key.

A structured kind (a censor style, a list of suppression rules) is one
entry: the parts inside it are checked by its own checker, which reads
STYLE_FIELDS and RULE_FIELDS below rather than its own constants.

Adding a setting: add its entry here, its tip to
i18n/en-US/settings.json, and its text to CONFIG_REFERENCE.md. The
registry tests then require its cache-key tags to be true.
"""

import difflib
import fnmatch
import functools
import re


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

DETECTION = 'detection'
CENSOR = 'censor'
ENCODE = 'encode'
CACHE_KEYS = ( DETECTION, CENSOR, ENCODE )

SIMPLE = 'simple'
ADVANCED = 'advanced'

PLACEHOLDERS = ( '<backend>', '<label>', '<model_variant>', '<profile>', '<style_type>',
                 '<person_backend>', '<folder>', '<pattern>' )

# Conditions under which an entry's cache keys move. The tests set each
# one up, check the tags hold with it, and check nothing moves without it.
CONDITIONS = {
    'backend_selected':       "the backend in the path is the selected one",
    'label_rendered':         "the label in the path is censored (or a reverse-mode window)",
    'model_variant_selected': "the model variant in the path is the selected one",
    'person_backend_selected': "the person backend in the path is the selected one",
    'reverse_on':             "reverse mode is on",
    'person_windows_on':      "reverse mode and its person windows are on",
    'preview_on':             "preview mode is on",
    'preview_off':            "preview mode is off",
    'cut_bursts_on':          "shot_cut_burst_window is above 0",
    'inherited_by_backend':   "the selected backend does not set this key itself",
}

KINDS = ( 'bool', 'int', 'number', 'choice', 'string', 'path', 'paths', 'labels', 'color', 'sizes',
          'censor_style', 'window_style', 'frame_style', 'geometry_action', 'suppression_rules',
          'promotion_rules', 'refused_pairs', 'profile_map' )

STRUCTURED_KINDS = frozenset( { 'censor_style', 'window_style', 'frame_style', 'geometry_action',
                                'suppression_rules', 'promotion_rules', 'refused_pairs',
                                'profile_map' } )


class Setting( object ):
    """One registry entry. See the module docstring."""

    __slots__ = ( 'path', 'kind', 'family', 'help', 'level', 'moves', 'when', 'minimum',
                  'maximum', 'min_open', 'max_open', 'choices', 'nullable', 'none_means',
                  'odd', 'cli', 'probe', 'note', 'may_move' )

    def __init__( self, path, kind, family, help, level=ADVANCED, moves=(), when=(),
                  minimum=None, maximum=None, min_open=False, max_open=False, choices=None,
                  nullable=False, none_means=None, odd=False, cli=None, probe=None, note=None,
                  may_move=() ):
        self.path = tuple( path.split( '.' ) ) if isinstance( path, str ) else tuple( path )
        self.kind = kind
        self.family = family
        self.help = help
        self.level = level
        self.moves = frozenset( moves )
        self.when = tuple( when )
        self.minimum = minimum
        self.maximum = maximum
        self.min_open = min_open
        self.max_open = max_open
        self.choices = choices
        self.nullable = nullable
        self.none_means = none_means
        self.odd = odd
        self.cli = cli
        self.probe = probe
        self.note = note
        self.may_move = frozenset( may_move )

    @property
    def dotted( self ):
        return '.'.join( self.path )

    @property
    def screen( self ):
        return screen_of( self )

    @property
    def requires( self ):
        return requires_of( self )

    @property
    def tip( self ):
        return 'setting.%s.tip'%( self.family, )

    @property
    def label_key( self ):
        return 'setting.%s.label'%( self.family, )

    @property
    def scope( self ):
        """Where the key may be set, from the placeholders in its path."""
        parts = set( part for part in self.path if part in PLACEHOLDERS )
        # A named block (detector_backend.nudenet_v3, person_backend.yolo11_seg)
        # is that backend's scope too; 'selected' is the one global key there.
        if self.path[0] == 'person_backend' and self.path[1] != 'selected':
            parts.add( '<person_backend>' )
        if self.path[0] == 'detector_backend' and self.path[1] != 'selected':
            parts.add( '<backend>' )
        if '<person_backend>' in parts:
            return 'person_backend'
        if '<profile>' in parts:
            return 'profile_label' if '<label>' in parts else 'profile'
        if '<model_variant>' in parts:
            return 'model_variant_label'
        if '<backend>' in parts:
            return 'backend_label' if '<label>' in parts else 'backend'
        if '<label>' in parts:
            return 'label'
        return 'global'

    @property
    def per_run( self ):
        """Whether a job may override it for one run (today: it has a betatv.py flag)."""
        return self.cli is not None

    def probe_for( self, path=None ):
        """A valid value for tests to change it to; some depend on the concrete path."""
        if callable( self.probe ):
            return self.probe( tuple( path ) if path is not None else self.path )
        return self.probe

    def choice_list( self ):
        """The allowed values, with dynamic choices resolved now."""
        choices = self.choices
        if callable( choices ):
            choices = choices()
        return list( choices ) if choices is not None else None


# ---------------------------------------------------------------------------
# Dynamic choices (imported late: these modules import betaconfig)
# ---------------------------------------------------------------------------

def label_names():
    import betaconst
    return sorted( betaconst.classes )


def backend_names():
    import betautils_detector as bu_detector
    return bu_detector.registered_backend_names()


def backend_labels( name ):
    """The labels one backend's adapter declares it can report."""
    import betautils_detector as bu_detector
    return bu_detector.supported_labels( name )


def label_table():
    """Each label's part, state (exposed, covered, none), sex (female, male, any) and figure view, in display order."""
    import betautils_labels as bu_labels
    return bu_labels.describe()


def person_backend_names():
    import betautils_person as bu_person
    return bu_person.registered_backend_names()


def model_variant_names( backend_name='nudenet_v3' ):
    import betautils_detector as bu_detector
    return bu_detector.variant_names( backend_name )


def person_model_variant_names( backend_name ):
    import betautils_person as bu_person
    return sorted( bu_person.get_segmenter( backend_name ).VARIANTS )


def log_levels():
    import betautils_log as bu_log
    return sorted( bu_log.LEVEL_NAMES )


def color_modes():
    import betautils_log as bu_log
    return list( bu_log.COLOR_MODES )


def palette_names():
    import betautils_log as bu_log
    return list( bu_log.PALETTE_NAMES )


def geometry_actions():
    import betautils_track as bu_track
    return list( bu_track.GEOMETRY_ACTIONS )


def profile_signals():
    import betautils_detector as bu_detector
    return sorted( bu_detector.PROFILE_THRESHOLD_KEYS )


def encode_presets( codec=None ):
    """Preset names encode_video_codec accepts (x264's, plus p1 to p7 for NVENC)."""
    import betautils_config as bu_config
    if codec is None:
        import betaconfig
        codec = getattr( betaconfig, 'encode_video_codec', 'libx264' )
    return sorted( bu_config.valid_encode_presets( codec ) )


def _codec_note( _value ):
    import betaconfig
    return "These are the presets %s accepts"%( getattr( betaconfig, 'encode_video_codec', 'libx264' ), )


def _track_hits_note( value ):
    import betaconfig
    fps = getattr( betaconfig, 'video_censor_fps', 10 ) or 10
    if not isinstance( value, ( int, float ) ) or isinstance( value, bool ) or value <= 10:
        return None
    return ( "%r detections before any censoring is about %.2f s of visible, uncensored footage on "
             "every new appearance at video_censor_fps=%r"%( value, value / max( fps, 1 ), fps ) )


CENSOR_STYLE_TYPES = ( 'bar', 'blur', 'debug', 'pixel', 'sticker' )
CENSOR_SHAPES = ( 'box', 'circle', 'ellipse' )
BLUR_METHODS = ( 'box', 'gaussian', 'triple_box' )
PIXEL_PATTERNS = ( 'hex', 'mosaic', 'square' )
MERGE_STRATEGIES = ( 'none', 'single-pass', 'span' )


# ---------------------------------------------------------------------------
# Censor style fields: read by betautils_config's style checks
# ---------------------------------------------------------------------------

def _field( kind, family, **limits ):
    limits.update( kind=kind, family=family )
    return limits


# Keys every style may carry, then the keys each type adds.
STYLE_COMMON_FIELDS = {
    'type':               _field( 'choice', 'style_type', choices=CENSOR_STYLE_TYPES ),
    'feather':            _field( 'number', 'style_feather', minimum=0, maximum=1 ),
    'weight':             _field( 'number', 'style_weight', minimum=0 ),
    'shape':              _field( 'choice', 'style_shape', choices=CENSOR_SHAPES ),
    'merge':              _field( 'choice', 'style_merge', choices=MERGE_STRATEGIES ),
    'width_area_safety':  _field( 'number', 'style_area_safety' ),
    'height_area_safety': _field( 'number', 'style_area_safety' ),
}

STYLE_TYPE_FIELDS = {
    'blur': {
        'method':   _field( 'choice', 'style_blur_method', choices=BLUR_METHODS ),
        'strength': _field( 'number', 'style_strength' ),
    },
    'pixel': {
        'pattern':  _field( 'choice', 'style_pixel_pattern', choices=PIXEL_PATTERNS ),
        'strength': _field( 'number', 'style_strength' ),
    },
    'bar': {
        'color':       _field( 'color', 'style_color' ),
        'span_extend': _field( 'number', 'style_bar_span_extend', minimum=0, maximum=1 ),
        'thickness':   _field( 'number', 'style_bar_thickness', minimum=0, min_open=True, maximum=3 ),
    },
    'sticker': {
        'dir':    _field( 'path', 'style_sticker_dir' ),
        'images': _field( 'paths', 'style_sticker_images' ),
        'scale':  _field( 'number', 'style_sticker_scale' ),
    },
    'debug': {
        'color': _field( 'color', 'style_color' ),
    },
}

# Reverse mode's frame style: blur, pixel or bar, plus how much it darkens.
FRAME_STYLE_EXTRA_FIELDS = {
    'dim': _field( 'number', 'reverse_frame_dim', minimum=0, maximum=1 ),
}

# Suppression and promotion rule fields.
RULE_FIELDS = {
    'suppression': {
        'suppressed_by': _field( 'choice', 'rule_suppressed_by', choices=label_names ),
        'margin':        _field( 'number', 'rule_margin', minimum=0, maximum=1 ),
        'min_iou':       _field( 'number', 'rule_min_iou', minimum=0, maximum=1 ),
    },
    'promotion': {
        'from':          _field( 'choice', 'rule_promote_from', choices=label_names ),
        'min_prob':      _field( 'number', 'rule_min_prob', minimum=0, maximum=1 ),
        'duplicate_iou': _field( 'number', 'rule_duplicate_iou', minimum=0, maximum=1 ),
        'requires_mode': _field( 'choice', 'rule_requires_mode', choices=( 'all', 'any' ) ),
        'requires':      _field( 'promotion_requirements', 'rule_requires' ),
    },
    'promotion_requirement': {
        'label':              _field( 'choice', 'rule_requires_label', choices=label_names ),
        'min_prob':           _field( 'number', 'rule_min_prob', minimum=0, maximum=1 ),
        'min_iou':            _field( 'number', 'rule_min_iou', minimum=0, maximum=1 ),
        'min_source_overlap': _field( 'number', 'rule_min_source_overlap', minimum=0, maximum=1 ),
        'overlap':            _field( 'choice', 'rule_overlap', choices=( 'anywhere', ) ),
    },
}


def style_fields_for( style_type ):
    """Every key a style of this type may carry, with its limits."""
    fields = dict( STYLE_COMMON_FIELDS )
    fields.update( STYLE_TYPE_FIELDS.get( style_type, {} ) )
    return fields


def style_field_families():
    """Every family a style field names, for the catalog checks."""
    families = { spec['family'] for spec in STYLE_COMMON_FIELDS.values() }
    for fields in STYLE_TYPE_FIELDS.values():
        families.update( spec['family'] for spec in fields.values() )
    families.update( spec['family'] for spec in FRAME_STYLE_EXTRA_FIELDS.values() )
    for fields in RULE_FIELDS.values():
        families.update( spec['family'] for spec in fields.values() )
    return families


# ---------------------------------------------------------------------------
# Per-label fields, and the tiers they may be set at
# ---------------------------------------------------------------------------

_LABEL_FIELD_HELP = {
    'min_prob':                     'default_min_prob-and-per-label-min_prob-per-backend',
    'min_prob_continue':            'min_prob_continue-per-label-per-backend-score-hysteresis',
    'area_safety':                  'default_area_safety-width_area_safety-height_area_safety',
    'time_safety':                  'default_time_safety-per-label-time_safety',
    'position_smoothing':           'default_position_smoothing-per-label-position_smoothing',
    'size_smoothing':               'default_size_smoothing-per-label-size_smoothing',
    'style_min_dwell_seconds':      'default_style_min_dwell_seconds-per-label-style_min_dwell_seconds',
    'interpolation_enabled':        'interpolation_max_gap-per-label-per-backend-and-default_interpolation_enabled',
    'interpolation_max_gap':        'interpolation_max_gap-per-label-per-backend-and-default_interpolation_enabled',
    'track_max_gap':                'track_max_gap-per-label-per-backend',
    'match_distance_multiplier':    'match_distance_multiplier-per-label-per-backend',
    'min_track_hits':               'default_min_track_hits-per-label-min_track_hits',
    'confirm_score':                'default_confirm_score-per-label-confirm_score',
    'paired_style':                 'paired_style-default_paired_style_max_distance-default_paired_style_tiebreak_margin',
    'paired_style_max_distance':    'paired_style-default_paired_style_max_distance-default_paired_style_tiebreak_margin',
    'paired_style_tiebreak_margin': 'paired_style-default_paired_style_max_distance-default_paired_style_tiebreak_margin',
    'paired_style_max_age':         'paired_style-default_paired_style_max_distance-default_paired_style_tiebreak_margin',
    'area_fraction':                'geometry-sanity-filter',
    'aspect_ratio':                 'geometry-sanity-filter',
    'geometry_action':              'geometry_action-what-a-violation-does',
}

# name: ( kind, family, limits, the default_* setting it falls back to or None )
LABEL_FIELDS = {
    'min_prob':           ( 'number', 'min_prob', dict( minimum=0, maximum=1 ), 'default_min_prob' ),
    'min_prob_continue':  ( 'number', 'min_prob_continue',
                            dict( minimum=0, maximum=1, min_open=True, max_open=True, nullable=True,
                                  none_means='no score hysteresis' ),
                            'default_min_prob_continue' ),
    'width_area_safety':  ( 'number', 'area_safety', {}, 'default_area_safety' ),
    'height_area_safety': ( 'number', 'area_safety', {}, 'default_area_safety' ),
    'time_safety':        ( 'number', 'time_safety', {}, 'default_time_safety' ),
    'position_smoothing': ( 'number', 'position_smoothing', dict( minimum=0, maximum=1, min_open=True ),
                            'default_position_smoothing' ),
    'size_smoothing':     ( 'number', 'size_smoothing',
                            dict( minimum=0, maximum=1, min_open=True, nullable=True,
                                  none_means='a third of position_smoothing' ),
                            'default_size_smoothing' ),
    'style_min_dwell_seconds': ( 'number', 'style_min_dwell_seconds', dict( minimum=0 ),
                                 'default_style_min_dwell_seconds' ),
    'interpolation_enabled': ( 'bool', 'interpolation_enabled', {}, 'default_interpolation_enabled' ),
    'interpolation_max_gap': ( 'number', 'interpolation_max_gap', dict( minimum=0 ),
                               'default_interpolation_max_gap' ),
    'track_max_gap':      ( 'number', 'track_max_gap', dict( minimum=0, min_open=True ), None ),
    'match_distance_multiplier': ( 'number', 'match_distance_multiplier',
                                   dict( minimum=0, min_open=True ), None ),
    'min_track_hits':     ( 'int', 'min_track_hits', dict( minimum=1, maximum=10, note=_track_hits_note ),
                            'default_min_track_hits' ),
    'confirm_score':      ( 'number', 'confirm_score',
                            dict( minimum=0, maximum=1, min_open=True, nullable=True,
                                  none_means='no score bar for confirming a track' ),
                            'default_confirm_score' ),
    'paired_style':       ( 'bool', 'paired_style', {}, None ),
    'paired_style_max_distance': ( 'number', 'paired_style_max_distance',
                                   dict( minimum=0, min_open=True ),
                                   'default_paired_style_max_distance' ),
    'paired_style_tiebreak_margin': ( 'number', 'paired_style_tiebreak_margin',
                                      dict( minimum=0, maximum=1 ),
                                      'default_paired_style_tiebreak_margin' ),
    'paired_style_max_age': ( 'number', 'paired_style_max_age', dict( minimum=0 ),
                              'default_paired_style_max_age' ),
    'min_area_fraction':  ( 'number', 'area_fraction',
                            dict( minimum=0, maximum=1, nullable=True, none_means='no lower bound' ),
                            'default_min_area_fraction' ),
    'max_area_fraction':  ( 'number', 'area_fraction',
                            dict( minimum=0, maximum=1, nullable=True, none_means='no upper bound' ),
                            'default_max_area_fraction' ),
    'min_aspect_ratio':   ( 'number', 'aspect_ratio',
                            dict( minimum=0, min_open=True, nullable=True, none_means='no lower bound' ),
                            'default_min_aspect_ratio' ),
    'max_aspect_ratio':   ( 'number', 'aspect_ratio',
                            dict( minimum=0, min_open=True, nullable=True, none_means='no upper bound' ),
                            'default_max_aspect_ratio' ),
    'geometry_action':    ( 'geometry_action', 'geometry_action', {}, 'default_geometry_action' ),
}

# The tiers a per-label field may be set at, most specific last, with
# what must hold for a change there to reach the render.
LABEL_TIERS = (
    ( 'detector_backend.<backend>.item_overrides.<label>',
      ( 'backend_selected', 'label_rendered' ) ),
    ( 'detector_backend.nudenet_v3.variants.<model_variant>.item_overrides.<label>',
      ( 'backend_selected', 'model_variant_selected', 'label_rendered' ) ),
    # The whole profiles block is in the censor key, rendered or not.
    ( 'detector_backend.<backend>.profiles.variants.<profile>.item_overrides.<label>',
      ( 'backend_selected', ) ),
    # Reverse mode's own per-label tuning; the whole reverse block is keyed while it is on.
    ( 'reverse.censor.overrides.<label>', ( 'reverse_on', ) ),
    ( 'reverse.windows.<label>', ( 'reverse_on', ) ),
)

# Probes for per-label fields: valid, and different from every shipped value.
_LABEL_PROBES = {
    'min_prob': 0.41, 'min_prob_continue': 0.2, 'width_area_safety': 0.03,
    'height_area_safety': 0.03, 'time_safety': 0.11, 'position_smoothing': 0.55,
    'size_smoothing': 0.4, 'style_min_dwell_seconds': 1.25, 'interpolation_enabled': False,
    'interpolation_max_gap': 0.35, 'track_max_gap': 50.0, 'match_distance_multiplier': 1.7,
    'min_track_hits': 3, 'confirm_score': 0.6, 'paired_style': True,
    'paired_style_max_distance': 1.3, 'paired_style_tiebreak_margin': 0.4,
    'paired_style_max_age': 0.45, 'min_area_fraction': 0.001, 'max_area_fraction': 0.9,
    'min_aspect_ratio': 0.2, 'max_aspect_ratio': 6.0, 'geometry_action': 'flag',
}


def _label_field_entries():
    entries = []
    for name, ( kind, family, limits, default_name ) in LABEL_FIELDS.items():
        for tier, when in LABEL_TIERS:
            entries.append( Setting( '%s.%s'%( tier, name ), kind, family,
                                     _LABEL_FIELD_HELP[family], moves=( CENSOR, ), when=when,
                                     probe=_LABEL_PROBES[name], **limits ) )
    return entries


# The global fallbacks. Every default_* is keyed directly, so each moves
# the censor key whatever is censored.
_DEFAULT_PROBES = {
    'default_min_prob': 0.4, 'default_min_prob_continue': 0.2, 'default_area_safety': 0.05,
    'default_time_safety': 0.2, 'default_position_smoothing': 0.5, 'default_size_smoothing': 0.3,
    'default_style_min_dwell_seconds': 1.0, 'default_interpolation_enabled': False,
    'default_interpolation_max_gap': 0.7, 'default_min_track_hits': 2,
    'default_confirm_score': 0.6, 'default_paired_style_max_distance': 1.5,
    'default_paired_style_tiebreak_margin': 0.3, 'default_paired_style_max_age': 0.5,
    'default_min_area_fraction': 0.001, 'default_max_area_fraction': 0.9,
    'default_min_aspect_ratio': 0.2, 'default_max_aspect_ratio': 6.0,
    'default_geometry_action': 'flag',
}


def _default_entries():
    entries = []
    seen = set()
    for name, ( kind, family, limits, default_name ) in LABEL_FIELDS.items():
        if default_name is None or default_name in seen:
            continue
        seen.add( default_name )
        if default_name == 'default_area_safety':
            limits = {}
        entries.append( Setting( default_name, kind, family, _LABEL_FIELD_HELP[family],
                                 moves=( CENSOR, ), probe=_DEFAULT_PROBES[default_name], **limits ) )
    return entries


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

S = Setting

_ENTRIES = [
    # --- detector backend ---
    S( 'detector_backend.selected', 'choice', 'detector_backend', 'selected', level=SIMPLE,
       choices=backend_names, moves=( DETECTION, CENSOR ), cli='--backend', probe='retinanet_v2' ),
    S( 'detector_backend.defaults.nn_batch_size', 'int', 'nn_batch_size',
       'nn_batch_size-per-backend-with-a-shared-default', minimum=1, probe=3 ),
    S( 'detector_backend.defaults.model_variant', 'choice', 'model_variant',
       'model_variant-nudenet_v3-per-backend', choices=model_variant_names,
       moves=( DETECTION, CENSOR ), when=( 'inherited_by_backend', ), probe='640m' ),
    S( 'detector_backend.defaults.picture_sizes', 'sizes', 'picture_sizes', 'picture_sizes-per-backend',
       moves=( DETECTION, ), when=( 'inherited_by_backend', ), probe=[ 416 ] ),
    S( 'detector_backend.<backend>.nn_batch_size', 'int', 'nn_batch_size',
       'nn_batch_size-per-backend-with-a-shared-default', minimum=1, cli='--nn-batch-size', probe=3 ),
    S( 'detector_backend.nudenet_v3.model_variant', 'choice', 'model_variant',
       'model_variant-nudenet_v3-per-backend', level=SIMPLE, choices=model_variant_names,
       moves=( DETECTION, CENSOR ), when=( 'backend_selected', ), cli='--variant', probe='640m' ),
    S( 'detector_backend.nudenet_v3.model_path', 'path', 'model_path', 'model_variant-nudenet_v3-per-backend',
       moves=( DETECTION, ), when=( 'backend_selected', ), probe='../resources/model/other.onnx' ),
    S( 'detector_backend.nudenet_v3.native_size', 'int', 'native_size', 'picture_sizes-per-backend',
       minimum=32, moves=( DETECTION, ), when=( 'backend_selected', ), probe=416 ),
    S( 'detector_backend.nudenet_v3.candidate_floor', 'number', 'candidate_floor',
       'candidate_floor-nudenet_v3-per-backend', minimum=0, maximum=1,
       moves=( DETECTION, ), when=( 'backend_selected', ), probe=0.18 ),
    S( 'detector_backend.nudenet_v3.nms_iou', 'number', 'nms_iou', 'nms_iou-nudenet_v3-per-backend',
       minimum=0, maximum=1, moves=( DETECTION, ), when=( 'backend_selected', ), probe=0.5 ),
    S( 'detector_backend.nudenet_v3.nms_mode', 'choice', 'nms_mode', 'nms_mode-nudenet_v3-per-backend',
       choices=( 'agnostic', 'per_class' ), moves=( DETECTION, ), when=( 'backend_selected', ),
       probe='agnostic' ),
    S( 'detector_backend.<backend>.picture_sizes', 'sizes', 'picture_sizes', 'picture_sizes-per-backend',
       moves=( DETECTION, ), when=( 'backend_selected', ), cli='--picture-sizes', probe=[ 416 ] ),
    S( 'detector_backend.<backend>.class_promotion', 'promotion_rules', 'class_promotion',
       'class-promotion', moves=( CENSOR, ), when=( 'backend_selected', ),
       probe={ 'exposed_vulva': [ { 'from': 'covered_vulva', 'min_prob': 0.3,
                                    'requires': [ { 'label': 'exposed_penis', 'min_prob': 0.3,
                                                    'overlap': 'anywhere' } ] } ] } ),
    S( 'detector_backend.<backend>.class_suppression', 'suppression_rules', 'class_suppression',
       'class-suppression', moves=( CENSOR, ), when=( 'backend_selected', ),
       probe={ 'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.09,
                                     'min_iou': 0.4 } ] } ),
    S( 'detector_backend.<backend>.class_suppression_refused', 'refused_pairs',
       'class_suppression_refused', 'pairs-never-proposed-built-in-and-class_suppression_refused',
       probe=[ 'exposed_breast<-face_femme' ] ),
    S( 'detector_backend.<backend>.profiles.default', 'string', 'profile_default', 'structure-profiles',
       moves=( CENSOR, ), when=( 'backend_selected', ), probe='quick_cut' ),
    S( 'detector_backend.<backend>.profiles.match_on', 'choice', 'profile_match_on',
       'why-the-signal-is-a-duration-share-and-not-a-median-shot-length', choices=profile_signals,
       moves=( CENSOR, ), when=( 'backend_selected', ), probe='cuts_per_min' ),
    S( 'detector_backend.<backend>.profiles.variants.<profile>.min_short_shot_fraction', 'number',
       'profile_min_short_shot_fraction', 'why-the-signal-is-a-duration-share-and-not-a-median-shot-length',
       minimum=0, maximum=1, moves=( CENSOR, ), when=( 'backend_selected', ), probe=0.2,
       note='It is a fraction of runtime' ),
    S( 'detector_backend.<backend>.profiles.variants.<profile>.min_cuts_per_min', 'number',
       'profile_min_cuts_per_min', 'why-the-signal-is-a-duration-share-and-not-a-median-shot-length',
       minimum=0, moves=( CENSOR, ), when=( 'backend_selected', ), probe=12 ),
    S( 'detector_backend.<backend>.profiles.variants.<profile>.video_censor_fps', 'number',
       'profile_video_censor_fps', 'structure-profiles', minimum=0, min_open=True,
       moves=( DETECTION, CENSOR ), when=( 'backend_selected', ), probe=14 ),
    S( 'detector_backend.<backend>.profiles.variants.<profile>.manual_only', 'bool', 'profile_manual_only',
       'manual_only', moves=( CENSOR, ), when=( 'backend_selected', ), probe=True ),

    # --- hardware ---
    S( 'gpu_enabled', 'choice', 'gpu_enabled', 'hardware', level=SIMPLE, choices=( 0, 1 ), probe=1 ),
    S( 'cuda_device_id', 'int', 'cuda_device_id', 'hardware', minimum=0, probe=1 ),
    S( 'detection_prefetch_frames', 'int', 'detection_prefetch_frames', 'detection_prefetch_frames',
       minimum=0, probe=8 ),
    S( 'structure_scan_workers', 'int', 'structure_scan_workers', 'hardware', minimum=0, probe=2 ),

    # --- detection sampling and confidence ---
    S( 'video_censor_fps', 'number', 'video_censor_fps', 'video_censor_fps', minimum=0, min_open=True,
       moves=( DETECTION, ), cli='--video-censor-fps', probe=12 ),
    S( 'global_min_prob', 'number', 'global_min_prob', 'global_min_prob', minimum=0, maximum=1,
       max_open=True, moves=( DETECTION, ), probe=0.1 ),

    # --- cross-size dedup ---
    S( 'cross_size_dedup.enabled', 'bool', 'cross_size_dedup_enabled', 'cross-size-dedup',
       moves=( CENSOR, ), probe=False ),
    S( 'cross_size_dedup.iou_threshold', 'number', 'cross_size_dedup_iou', 'cross-size-dedup',
       minimum=0, maximum=1, min_open=True, moves=( CENSOR, ), probe=0.5 ),

    # --- shot cuts ---
    S( 'shot_cut_detection_enabled', 'bool', 'shot_cut_detection_enabled',
       'shot_cut_detection_enabled-and-shot_cut_threshold', moves=( CENSOR, ), probe=False ),
    S( 'shot_cut_threshold', 'number', 'shot_cut_threshold',
       'shot_cut_detection_enabled-and-shot_cut_threshold', minimum=0, maximum=1,
       moves=( CENSOR, ), probe=0.3 ),
    S( 'shot_cut_burst_window', 'number', 'shot_cut_burst_window', 'shot-cut-detection', minimum=0,
       moves=( CENSOR, ), probe=0.5 ),
    S( 'shot_cut_burst_max_span', 'number', 'shot_cut_burst_max_span', 'shot-cut-detection',
       minimum=0, min_open=True, nullable=True, none_means='a burst of any length',
       moves=( CENSOR, ), when=( 'cut_bursts_on', ), probe=1.5 ),
    S( 'shot_cut_burst_keep', 'choice', 'shot_cut_burst_keep', 'shot-cut-detection',
       choices=( 'first', 'last' ), moves=( CENSOR, ), when=( 'cut_bursts_on', ), probe='last' ),

    # --- resumability, rendering, encoding ---
    S( 'detection_checkpoint_frames', 'int', 'detection_checkpoint_frames', 'detection_checkpoint_frames',
       minimum=0, probe=250 ),
    S( 'render_chunk_seconds', 'number', 'render_chunk_seconds', 'render_chunk_seconds', minimum=0,
       probe=120 ),
    S( 'render_workers', 'int', 'render_workers', 'render_workers', minimum=0, cli='--render-workers',
       probe=3 ),
    S( 'render_decode_threads', 'int', 'render_decode_threads', 'render_decode_threads', minimum=0,
       nullable=True, none_means="OpenCV's own thread count", probe=2 ),
    S( 'render_worker_mb_per_megapixel', 'number', 'render_worker_mb_per_megapixel',
       'render_worker_mb_per_megapixel', minimum=0, min_open=True, nullable=True,
       none_means='no memory cap on render workers', probe=150 ),
    S( 'render_memory_reserve_mb', 'number', 'render_memory_reserve_mb', 'render_memory_reserve_mb',
       minimum=0, probe=1024 ),
    S( 'render_chunk_container', 'choice', 'render_chunk_container',
       'encode_video_codec-encode_crf-encode_preset-render_chunk_container', choices=( 'mkv', 'mp4' ),
       moves=( ENCODE, ), probe='mp4' ),
    S( 'render_verify_frame_counts', 'bool', 'render_verify_frame_counts', 'render_verify_frame_counts',
       probe=False ),
    S( 'render_motion_interpolation', 'bool', 'render_motion_interpolation', 'render_motion_interpolation',
       moves=( CENSOR, ), cli='--motion-interpolation', probe=False ),
    S( 'render_motion_max_span_seconds', 'number', 'render_motion_max_span_seconds',
       'render_motion_max_span_seconds', minimum=0, min_open=True, nullable=True,
       none_means='two sampling intervals', moves=( CENSOR, ), cli='--motion-max-span', probe=0.5 ),
    S( 'render_motion_collapse_max_growth', 'number', 'render_motion_collapse_max_growth',
       'render_motion_collapse_max_growth', minimum=1, moves=( CENSOR, ),
       cli='--motion-collapse-growth', probe=1.3 ),
    S( 'render_motion_size_window_seconds', 'number', 'render_motion_size_window_seconds',
       'render_motion_size_window_seconds', minimum=0, nullable=True,
       none_means='box sizes left alone', moves=( CENSOR, ), cli='--motion-size-window', probe=0.5 ),
    S( 'ffmpeg_max_retries', 'int', 'ffmpeg_max_retries', 'ffmpeg_max_retries-ffmpeg_retry_backoff_seconds',
       minimum=0, probe=4 ),
    S( 'ffmpeg_retry_backoff_seconds', 'number', 'ffmpeg_retry_backoff_seconds',
       'ffmpeg_max_retries-ffmpeg_retry_backoff_seconds', minimum=0, probe=2 ),
    S( 'encode_video_codec', 'string', 'encode_video_codec',
       'encode_video_codec-encode_crf-encode_preset-render_chunk_container', level=SIMPLE,
       moves=( ENCODE, ), probe='libx265' ),
    S( 'encode_preset', 'choice', 'encode_preset',
       'encode_video_codec-encode_crf-encode_preset-render_chunk_container', choices=encode_presets,
       moves=( ENCODE, ), when=( 'preview_off', ), cli='--encode-preset', probe='medium',
       note=_codec_note ),
    S( 'encode_crf', 'int', 'encode_crf', 'encode_video_codec-encode_crf-encode_preset-render_chunk_container',
       level=SIMPLE, minimum=0, maximum=51, moves=( ENCODE, ), probe=21 ),

    # --- structure profile selection ---
    S( 'profile_by_source_dir.<folder>', 'string', 'profile_by_source_dir', 'the-source-folder-convention',
       moves=( CENSOR, ), probe='compilation' ),
    S( 'profile_by_path_pattern', 'profile_map', 'profile_by_path_pattern', 'the-source-folder-convention',
       nullable=True, none_means='no path patterns', moves=( CENSOR, ),
       probe={ '*compilation*': 'compilation' } ),
    S( 'force_structure_profile', 'string', 'force_structure_profile', 'forcing-a-profile---profile-name',
       nullable=True, none_means='each video picks its own profile', moves=( CENSOR, ), cli='--profile',
       probe='compilation' ),

    # --- what gets censored, and how ---
    S( 'items_to_censor', 'labels', 'items_to_censor', 'censor-styles', level=SIMPLE,
       choices=label_names, moves=( CENSOR, ), probe=[ 'exposed_vulva' ] ),
    S( 'item_overrides.<label>.censor_style', 'censor_style', 'censor_style', 'censor-styles',
       level=SIMPLE, moves=( CENSOR, ), when=( 'label_rendered', ),
       probe=[ { 'type': 'pixel', 'pattern': 'square', 'strength': 25 } ] ),
    S( 'item_overrides.<label>.censor_shape', 'choice', 'censor_shape', 'common-keys', level=SIMPLE,
       choices=CENSOR_SHAPES, moves=( CENSOR, ), when=( 'label_rendered', ), probe='box' ),
    S( 'default_censor_style', 'censor_style', 'censor_style', 'censor-styles', moves=( CENSOR, ),
       probe={ 'type': 'blur', 'method': 'triple_box', 'strength': 55 } ),
    S( 'default_censor_shape', 'choice', 'censor_shape', 'common-keys', choices=CENSOR_SHAPES,
       moves=( CENSOR, ), probe='ellipse' ),
    S( 'default_profile_enabled', 'bool', 'profile_enabled', 'structure-profiles',
       moves=( DETECTION, CENSOR ), probe=False ),
    S( 'type_default_shapes.<style_type>', 'choice', 'type_default_shapes', 'common-keys',
       choices=CENSOR_SHAPES, moves=( CENSOR, ), probe='ellipse' ),
    S( 'censor_overlap_strategy.<style_type>', 'choice', 'censor_overlap_strategy',
       'censor_overlap_strategy', choices=MERGE_STRATEGIES, moves=( CENSOR, ), probe='span' ),
    S( 'censor_scale_strategy', 'choice', 'censor_scale_strategy', 'censor_scale_strategy',
       choices=( 'feature', 'image', 'none' ), moves=( CENSOR, ), probe='image' ),
    S( 'blur_fast_approximation', 'bool', 'blur_fast_approximation',
       'blur_fast_approximation-and-blur_approximation_min_kernel', moves=( CENSOR, ), probe=False ),
    S( 'blur_approximation_min_kernel', 'int', 'blur_approximation_min_kernel',
       'blur_fast_approximation-and-blur_approximation_min_kernel', minimum=3, odd=True,
       moves=( CENSOR, ), probe=31 ),
    S( 'blur_edge_margin', 'bool', 'blur_edge_margin', 'blur_edge_margin', moves=( CENSOR, ), probe=False ),

    # --- reverse mode ---
    S( 'reverse.enabled', 'bool', 'reverse_enabled', 'reverse-mode', level=SIMPLE, moves=( CENSOR, ),
       cli='--reverse', probe=True ),
    S( 'reverse.frame_style', 'frame_style', 'reverse_frame_style', 'settings', moves=( CENSOR, ),
       when=( 'reverse_on', ), probe={ 'type': 'blur', 'method': 'triple_box', 'strength': 9, 'dim': 0.3 } ),
    S( 'reverse.empty_frame', 'choice', 'reverse_empty_frame', 'settings',
       choices=( 'obscure', 'passthrough' ), moves=( CENSOR, ), when=( 'reverse_on', ), probe='obscure' ),
    S( 'reverse.fade_seconds', 'number', 'reverse_fade_seconds', 'settings', minimum=0,
       moves=( CENSOR, ), when=( 'reverse_on', ), probe=0.4 ),
    S( 'reverse.trigger.labels', 'labels', 'reverse_trigger_labels', 'when-the-trigger-fires',
       choices=label_names, nullable=True, none_means='every censored label', moves=( CENSOR, ),
       when=( 'reverse_on', ), probe=[ 'exposed_breast' ] ),
    S( 'reverse.trigger.min_prob', 'number', 'reverse_trigger_min_prob', 'when-the-trigger-fires',
       minimum=0, maximum=1, max_open=True, moves=( CENSOR, ), when=( 'reverse_on', ),
       probe=0.45 ),
    S( 'reverse.trigger.hold_seconds', 'number', 'reverse_trigger_hold_seconds', 'when-the-trigger-fires',
       minimum=0, moves=( CENSOR, ), when=( 'reverse_on', ), probe=1.0 ),
    S( 'reverse.censor.labels', 'labels', 'reverse_censor_labels', 'what-each-kind-inherits',
       choices=label_names, nullable=True, none_means='items_to_censor', moves=( CENSOR, ),
       when=( 'reverse_on', ), probe=[ 'exposed_vulva' ] ),
    S( 'reverse.censor.overrides.<label>.censor_style', 'censor_style', 'censor_style',
       'what-each-kind-inherits', moves=( CENSOR, ), when=( 'reverse_on', ),
       probe=[ { 'type': 'pixel', 'pattern': 'square', 'strength': 25 } ] ),
    S( 'reverse.censor.overrides.<label>.censor_shape', 'choice', 'censor_shape', 'what-each-kind-inherits',
       choices=CENSOR_SHAPES, moves=( CENSOR, ), when=( 'reverse_on', ), probe='box' ),
    S( 'reverse.windows.<label>.censor_style', 'window_style', 'reverse_window_style',
       'what-each-kind-inherits', moves=( CENSOR, ), when=( 'reverse_on', ),
       probe={ 'type': 'show', 'feather': 0.2 } ),
    S( 'reverse.windows.<label>.censor_shape', 'choice', 'censor_shape', 'what-each-kind-inherits',
       choices=CENSOR_SHAPES, moves=( CENSOR, ), when=( 'reverse_on', ), probe='ellipse' ),
    S( 'reverse.person_windows.enabled', 'bool', 'person_windows_enabled', 'person-windows-whole-male-bodies',
       moves=( CENSOR, ), when=( 'reverse_on', ), probe=True ),
    S( 'reverse.person_windows.male_labels', 'labels', 'person_windows_male_labels',
       'person-windows-whole-male-bodies', choices=label_names, moves=( CENSOR, ),
       when=( 'reverse_on', ), probe=[ 'face_masc' ] ),
    S( 'reverse.person_windows.female_labels', 'labels', 'person_windows_female_labels',
       'person-windows-whole-male-bodies', choices=label_names, moves=( CENSOR, ),
       when=( 'reverse_on', ), probe=[ 'face_femme', 'exposed_breast' ] ),
    S( 'reverse.person_windows.evidence_min_prob', 'number', 'person_windows_evidence_min_prob',
       'person-windows-whole-male-bodies', minimum=0, maximum=1, max_open=True,
       moves=( CENSOR, ), when=( 'reverse_on', ), probe=0.5 ),
    S( 'reverse.person_windows.track_iou', 'number', 'person_windows_track_iou',
       'person-windows-whole-male-bodies', minimum=0, maximum=1, min_open=True,
       moves=( CENSOR, ), when=( 'reverse_on', ), probe=0.4 ),
    S( 'reverse.person_windows.track_max_gap_seconds', 'number', 'person_windows_track_max_gap_seconds',
       'person-windows-whole-male-bodies', minimum=0, moves=( CENSOR, ), when=( 'reverse_on', ),
       probe=1.5 ),
    S( 'reverse.person_windows.shrink', 'number', 'person_windows_shrink', 'person-windows-whole-male-bodies',
       minimum=0, maximum=0.5, max_open=True, moves=( CENSOR, ), when=( 'reverse_on', ),
       probe=0.05 ),

    # --- person segmentation ---
    S( 'person_backend.selected', 'choice', 'person_backend', 'person-segmentation',
       choices=person_backend_names, moves=( CENSOR, ), when=( 'person_windows_on', ),
       probe='rf_detr_seg' ),
    S( 'person_backend.<person_backend>.model_variant', 'string', 'person_model_variant',
       'person-segmentation', moves=( CENSOR, ), when=( 'person_windows_on', 'person_backend_selected' ),
       probe=lambda path: { 'yolo11_seg': 'm', 'rf_detr_seg': 'medium' }.get( path[1], 'm' ) ),
    S( 'person_backend.<person_backend>.model_path', 'path', 'person_model_path', 'person-segmentation',
       moves=( CENSOR, ), when=( 'person_windows_on', 'person_backend_selected' ),
       probe='../resources/model/other-seg.onnx' ),
    S( 'person_backend.yolo11_seg.picture_size', 'int', 'person_picture_size', 'person-segmentation',
       minimum=32, moves=( CENSOR, ), when=( 'person_windows_on', 'person_backend_selected' ), probe=480 ),
    S( 'person_backend.<person_backend>.min_prob', 'number', 'person_min_prob', 'person-segmentation',
       minimum=0, maximum=1, min_open=True, max_open=True, moves=( CENSOR, ),
       when=( 'person_windows_on', 'person_backend_selected' ), probe=0.4 ),
    S( 'person_backend.yolo11_seg.nms_iou', 'number', 'person_nms_iou', 'person-segmentation',
       minimum=0, maximum=1, min_open=True, max_open=True, moves=( CENSOR, ),
       when=( 'person_windows_on', 'person_backend_selected' ), probe=0.6 ),
    S( 'person_backend.<person_backend>.batch_size', 'int', 'person_batch_size', 'person-segmentation',
       minimum=1, probe=2 ),

    # --- input handling ---
    S( 'input_delete_probability', 'number', 'input_delete_probability', 'input_delete_probability',
       minimum=0, maximum=1, probe=0.5 ),
    S( 'move_completed_sources', 'bool', 'move_completed_sources', 'move_completed_sources',
       level=SIMPLE, probe=True ),

    # --- preview mode ---
    S( 'preview_mode_enabled', 'bool', 'preview_mode_enabled', 'preview-mode', level=SIMPLE,
       moves=( ENCODE, ), cli='--preview', probe=True ),
    S( 'preview_max_seconds', 'number', 'preview_max_seconds', 'preview-mode', level=SIMPLE,
       minimum=0, min_open=True, cli='--preview-seconds', probe=30 ),
    S( 'preview_encode_preset', 'choice', 'preview_encode_preset', 'preview-mode', choices=encode_presets,
       moves=( ENCODE, ), when=( 'preview_on', ), cli='--preview-preset', probe='veryfast',
       note=_codec_note ),
    S( 'preview_start_seconds', 'number', 'preview_start_seconds', 'preview-mode', minimum=0,
       nullable=True, none_means='the start of the video', cli='--preview-start-seconds', probe=5 ),
    S( 'preview_random_slice', 'bool', 'preview_random_slice', 'preview-mode', cli='--preview-random',
       probe=True ),

    # --- logging and stats ---
    S( 'logging_enabled', 'bool', 'logging_enabled', 'logging-and-stats', cli='--logging', probe=False ),
    S( 'log_path', 'path', 'log_path', 'logging-and-stats', nullable=True,
       none_means='no log file', probe='../output/logs/other.log' ),
    S( 'log_level', 'choice', 'log_level', 'logging-and-stats', choices=log_levels, cli='--log-level',
       probe='info' ),
    S( 'console_level', 'choice', 'console_level', 'logging-and-stats', choices=log_levels,
       cli='--console-level', probe='warn' ),
    S( 'console_color', 'choice', 'console_color', 'logging-and-stats', choices=color_modes, probe='never' ),
    S( 'console_palette', 'choice', 'console_palette', 'logging-and-stats', choices=palette_names,
       probe='monochrome' ),
    S( 'stats_enabled', 'bool', 'stats_enabled', 'stats', cli='--stats', probe=False ),
    S( 'stats_path', 'path', 'stats_path', 'stats', nullable=True,
       none_means='no stats file', probe='../output/stats/other.jsonl' ),

    # --- BetaVision screen capture ---
    S( 'vision_cap_monitor', 'int', 'vision_cap_monitor', 'betavision-screen-capture', minimum=0, probe=1 ),
    S( 'vision_cap_top', 'int', 'vision_cap_top', 'betavision-screen-capture', minimum=0, probe=10 ),
    S( 'vision_cap_left', 'int', 'vision_cap_left', 'betavision-screen-capture', minimum=0, probe=10 ),
    S( 'vision_cap_height', 'int', 'vision_cap_height', 'betavision-screen-capture', minimum=1, probe=720 ),
    S( 'vision_cap_width', 'int', 'vision_cap_width', 'betavision-screen-capture', minimum=1, probe=640 ),
    S( 'vision_cursor_color', 'color', 'vision_cursor_color', 'betavision-screen-capture',
       probe=( 255, 0, 0 ) ),
    S( 'betavision_delay', 'number', 'betavision_delay', 'betavision-screen-capture', minimum=0,
       probe=0.25 ),
    S( 'betavision_interpolate', 'bool', 'betavision_interpolate', 'betavision-screen-capture',
       probe=True ),

    # --- miscellaneous ---
    S( 'debug_mode', 'int', 'debug_mode', 'debug_mode', minimum=0, maximum=3, moves=( CENSOR, ),
       cli='--debug-mode', probe=1 ),
    S( 'enable_betasuite_watermark', 'bool', 'enable_betasuite_watermark', 'miscellaneous',
       moves=( CENSOR, ), probe=True ),
]

# Keys an entry can move for some files only. A config-level key cannot
# show these: what changes is which structure profile, and so which
# sample rate, a file gets, or which preview slice. The app says "may
# re-detect some files" for them.
_PROFILE_CHOICE = ( DETECTION, )   # the profile a file gets sets its sample rate
_PREVIEW_SLICE = ( DETECTION, CENSOR )   # a new slice is detected and rendered afresh
MAY_MOVE = {
    'detector_backend.<backend>.profiles.default': _PROFILE_CHOICE,
    'detector_backend.<backend>.profiles.match_on': _PROFILE_CHOICE,
    'detector_backend.<backend>.profiles.variants.<profile>.min_short_shot_fraction': _PROFILE_CHOICE,
    'detector_backend.<backend>.profiles.variants.<profile>.min_cuts_per_min': _PROFILE_CHOICE,
    'detector_backend.<backend>.profiles.variants.<profile>.manual_only': _PROFILE_CHOICE,
    'profile_by_source_dir.<folder>': _PROFILE_CHOICE,
    'profile_by_path_pattern': _PROFILE_CHOICE,
    'force_structure_profile': _PROFILE_CHOICE,
    # The shot cuts a file has decide which profile it measures as.
    'shot_cut_detection_enabled': _PROFILE_CHOICE,
    'shot_cut_threshold': _PROFILE_CHOICE,
    'shot_cut_burst_window': _PROFILE_CHOICE,
    'shot_cut_burst_max_span': _PROFILE_CHOICE,
    'shot_cut_burst_keep': _PROFILE_CHOICE,
    'preview_mode_enabled': _PREVIEW_SLICE,
    'preview_max_seconds': _PREVIEW_SLICE,
    'preview_start_seconds': _PREVIEW_SLICE,
    'preview_random_slice': _PREVIEW_SLICE,
}
for _entry in _ENTRIES:
    if _entry.dotted in MAY_MOVE:
        _entry.may_move = frozenset( MAY_MOVE[_entry.dotted] )

REGISTRY = tuple( _ENTRIES + _default_entries() + _label_field_entries() )


# ---------------------------------------------------------------------------
# Screens: where the app shows each setting
# ---------------------------------------------------------------------------
# A simple setting sits on its screen; an advanced one in that screen's
# "More options". SCREEN_OTHER is a tab of its own for tuning that no
# screen's task needs; it holds no simple setting, so it shows its
# settings directly, with no fold. Every setting is on All Settings too. First match wins; a path pattern is fnmatch-style
# over the dotted path, so '*' crosses dots.

SCREEN_RUN = 'run'
SCREEN_WHAT_TO_HIDE = 'what_to_hide'
SCREEN_MODELS = 'models'
SCREEN_LIBRARY = 'library'
SCREEN_REVERSE = 'reverse'
SCREEN_RESOURCES = 'resources'
SCREEN_OTHER = 'other'
SCREENS = ( SCREEN_RUN, SCREEN_WHAT_TO_HIDE, SCREEN_MODELS, SCREEN_LIBRARY, SCREEN_REVERSE,
            SCREEN_RESOURCES, SCREEN_OTHER )

# The look of a censor: the global fallbacks among the default_* settings.
_DEFAULT_LOOK = ( 'censor_style', 'censor_shape', 'area_safety', 'time_safety',
                  'paired_style_max_age', 'paired_style_max_distance', 'paired_style_tiebreak_margin',
                  'style_min_dwell_seconds' )

SCREEN_RULES = (
    # The models page: which detector and person model, and their files.
    ( 'detector_backend.selected', SCREEN_MODELS ),
    ( '*.model_variant', SCREEN_MODELS ),
    ( '*.model_path', SCREEN_MODELS ),
    ( 'detector_backend.nudenet_v3.native_size', SCREEN_MODELS ),
    ( 'person_backend.selected', SCREEN_MODELS ),
    # Resources: how hard the computer works, never what comes out.
    ( 'gpu_enabled', SCREEN_RESOURCES ),
    ( 'cuda_device_id', SCREEN_RESOURCES ),
    ( '*nn_batch_size', SCREEN_RESOURCES ),
    ( 'person_backend.*.batch_size', SCREEN_RESOURCES ),
    ( 'render_workers', SCREEN_RESOURCES ),
    ( 'render_decode_threads', SCREEN_RESOURCES ),
    ( 'render_memory_reserve_mb', SCREEN_RESOURCES ),
    ( 'render_worker_mb_per_megapixel', SCREEN_RESOURCES ),
    ( 'structure_scan_workers', SCREEN_RESOURCES ),
    ( 'detection_prefetch_frames', SCREEN_RESOURCES ),
    # Reverse mode, with the person model it uses.
    ( 'reverse.*', SCREEN_REVERSE ),
    ( 'person_backend.*', SCREEN_REVERSE ),
    # What to hide, and how it looks.
    ( 'items_to_censor', SCREEN_WHAT_TO_HIDE ),
    ( 'item_overrides.*', SCREEN_WHAT_TO_HIDE ),
    ( 'type_default_shapes.*', SCREEN_WHAT_TO_HIDE ),
    ( 'censor_overlap_strategy.*', SCREEN_WHAT_TO_HIDE ),
    ( 'censor_scale_strategy', SCREEN_WHAT_TO_HIDE ),
    ( 'blur_*', SCREEN_WHAT_TO_HIDE ),
) + tuple( ( 'default_' + field, SCREEN_WHAT_TO_HIDE ) for field in _DEFAULT_LOOK ) + (
    # The library: where videos come from and go, and their profile.
    ( 'move_completed_sources', SCREEN_LIBRARY ),
    ( 'input_delete_probability', SCREEN_LIBRARY ),
    ( 'profile_by_path_pattern', SCREEN_LIBRARY ),
    ( 'profile_by_source_dir.*', SCREEN_LIBRARY ),
    # A run: sampling, preview, output.
    ( 'video_censor_fps', SCREEN_RUN ),
    ( 'preview_*', SCREEN_RUN ),
    ( 'encode_*', SCREEN_RUN ),
    ( 'enable_betasuite_watermark', SCREEN_RUN ),
    ( 'force_structure_profile', SCREEN_RUN ),
    ( 'render_chunk_*', SCREEN_RUN ),
    ( 'render_verify_frame_counts', SCREEN_RUN ),
    ( 'ffmpeg_*', SCREEN_RUN ),
    ( 'detection_checkpoint_frames', SCREEN_RUN ),
    # Everything else is detection and tracking tuning, logging, capture.
    ( '*', SCREEN_OTHER ),
)


# ---------------------------------------------------------------------------
# Parent settings: a setting that has an effect only while another is set
# ---------------------------------------------------------------------------
# The app nests such a setting under its parent and greys it out, still
# visible and searchable, while the parent is not as required. Every
# matching rule adds its parents; a parent never requires itself. Each
# link is read from the code that reads the setting, cited per rule.
# Per-label tiers have none: whether a label interpolates or pairs its
# style resolves across tiers, so a sibling flag at one tier decides
# nothing on its own.

ON = 'on'
OFF = 'off'
ABOVE_ZERO = 'above_zero'
REQUIRE_TESTS = ( ON, OFF, ABOVE_ZERO )

REQUIRES_RULES = (
    # betatv.py: read only inside `if preview_mode_enabled`; the full run uses encode_preset.
    ( 'preview_max_seconds', ( ( 'preview_mode_enabled', ON ), ) ),
    ( 'preview_start_seconds', ( ( 'preview_mode_enabled', ON ), ) ),
    ( 'preview_random_slice', ( ( 'preview_mode_enabled', ON ), ) ),
    ( 'preview_encode_preset', ( ( 'preview_mode_enabled', ON ), ) ),
    ( 'encode_preset', ( ( 'preview_mode_enabled', OFF ), ) ),
    # betatv.py hands the threshold to shot-cut detection with its `enabled`
    # flag; betautils_track.py collapses bursts only when the window is above 0.
    ( 'shot_cut_threshold', ( ( 'shot_cut_detection_enabled', ON ), ) ),
    ( 'shot_cut_burst_*', ( ( 'shot_cut_detection_enabled', ON ), ) ),
    ( 'shot_cut_burst_keep', ( ( 'shot_cut_burst_window', ABOVE_ZERO ), ) ),
    ( 'shot_cut_burst_max_span', ( ( 'shot_cut_burst_window', ABOVE_ZERO ), ) ),
    # betautils_track.py: returns before reading iou_threshold when off.
    ( 'cross_size_dedup.iou_threshold', ( ( 'cross_size_dedup.enabled', ON ), ) ),
    # betautils_censor.py: the approximation, and so its threshold, is
    # skipped when off and whenever a blur edge margin is in use.
    ( 'blur_approximation_min_kernel', ( ( 'blur_fast_approximation', ON ), ( 'blur_edge_margin', OFF ) ) ),
    # betautils_detector.py: the device id goes to the CUDA provider only.
    ( 'cuda_device_id', ( ( 'gpu_enabled', ON ), ) ),
    # betautils_render.py: no motion index is built when interpolation is off.
    ( 'render_motion_max_span_seconds', ( ( 'render_motion_interpolation', ON ), ) ),
    ( 'render_motion_collapse_max_growth', ( ( 'render_motion_interpolation', ON ), ) ),
    ( 'render_motion_size_window_seconds', ( ( 'render_motion_interpolation', ON ), ) ),
    # betautils_reverse.py and betatv.py: reverse mode, and person windows
    # within it, which are the only users of the person model.
    ( 'reverse.*', ( ( 'reverse.enabled', ON ), ) ),
    ( 'reverse.person_windows.*', ( ( 'reverse.person_windows.enabled', ON ), ) ),
    ( 'person_backend.*', ( ( 'reverse.enabled', ON ), ( 'reverse.person_windows.enabled', ON ) ) ),
)


# Per-label keys that change nothing while the label's own resolved flag
# is off: betautils_track.py reads the three paired_style_* values only
# inside `if settings.paired_style` (and in a tuning observer that writes
# nothing). Not interpolation_max_gap: it also raises the label's default
# track_max_gap, so it acts with interpolation off.
LABEL_REQUIRES = {
    'paired_style_max_distance': 'paired_style',
    'paired_style_tiebreak_margin': 'paired_style',
    'paired_style_max_age': 'paired_style',
}


def requires_of( entry ):
    """( parent path, test ) pairs, in rule order, without itself."""
    found = []
    for pattern, parents in REQUIRES_RULES:
        if fnmatch.fnmatchcase( entry.dotted, pattern ):
            for parent in parents:
                if parent[0] != entry.dotted:
                    found.append( parent )
    return tuple( found )


def screen_of( entry ):
    for pattern, screen in SCREEN_RULES:
        if fnmatch.fnmatchcase( entry.dotted, pattern ):
            return screen
    raise AssertionError( 'no screen for ' + entry.dotted )


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _segment_matches( pattern, part ):
    return pattern == part or ( pattern in PLACEHOLDERS and isinstance( part, str ) )


def _matches( pattern, path ):
    return len( pattern ) == len( path ) and all(
        _segment_matches( want, got ) for want, got in zip( pattern, path ) )


def _specificity( entry ):
    """Literal segments beat placeholders, so 'detector_backend.defaults' wins over '<backend>'."""
    return sum( 1 for part in entry.path if part not in PLACEHOLDERS )


def find( path ):
    """
    The entry for a concrete setting path, or None.

    A path inside a structured entry (a field of a censor style, a rule)
    finds that entry.
    """
    path = tuple( path.split( '.' ) ) if isinstance( path, str ) else tuple( path )
    try:
        return _find( path )
    except TypeError:   # an unhashable key: not a setting path
        return None


@functools.lru_cache( maxsize=8192 )
def _find( path ):
    """find() for a tuple path; memoised, since validation walks the same paths every run."""
    candidates = []
    for entry in REGISTRY:
        if _matches( entry.path, path ):
            candidates.append( entry )
        elif ( entry.kind in STRUCTURED_KINDS and len( path ) > len( entry.path )
               and _matches( entry.path, path[:len( entry.path )] ) ):
            candidates.append( entry )
    if not candidates:
        return None
    return max( candidates, key=lambda entry: ( len( entry.path ), _specificity( entry ) ) )


def is_container( path ):
    """Whether a path is a block that registered settings live inside."""
    path = tuple( path.split( '.' ) ) if isinstance( path, str ) else tuple( path )
    return _is_container( path )


@functools.lru_cache( maxsize=1024 )
def _is_container( path ):
    return any( len( entry.path ) > len( path ) and _matches( entry.path[:len( path )], path )
                for entry in REGISTRY )


def families():
    """Every tip family the registry names, settings and style fields both."""
    return { entry.family for entry in REGISTRY } | style_field_families()


def suggestion( path ):
    """The closest real path to a mistyped one, or None."""
    path = tuple( path.split( '.' ) ) if isinstance( path, str ) else tuple( path )
    if not path:
        return None
    same_depth = [ entry for entry in REGISTRY if len( entry.path ) == len( path )
                   and all( _segment_matches( want, got )
                            for want, got in zip( entry.path[:-1], path[:-1] ) ) ]
    names = sorted( { entry.path[-1] for entry in same_depth if entry.path[-1] not in PLACEHOLDERS } )
    close = difflib.get_close_matches( str( path[-1] ), names, n=1 )
    if not close:
        return None
    return '.'.join( [ str( part ) for part in path[:-1] ] + close )


# ---------------------------------------------------------------------------
# Checking one value against its entry
# ---------------------------------------------------------------------------

class Problem( str ):
    """
    A validation error: English text for the CLI and the log, plus a
    code and params for the engine protocol, which never sends sentences.
    """

    def __new__( cls, text, code, **params ):
        problem = super().__new__( cls, text )
        problem.code = code
        problem.params = params
        return problem

    def as_error( self ):
        return { 'code': self.code, 'params': dict( self.params ) }


def rule( rule_id, text, **params ):
    """
    A cross-field check's error: code config.rule with a stable rule id,
    and the setting it is about as a dotted path (read from where the
    text names it first, unless given).
    """
    params['rule'] = rule_id
    params.setdefault( 'path', located( text ) )
    return Problem( text, 'config.rule', **params )


_LOCATION = re.compile( r"^(?:betaconfig\.)?([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\]]*\]|\.[A-Za-z_][A-Za-z0-9_]*)*)" )


def located( text ):
    """
    "detector_backend['nudenet_v3']['nms_iou'] must be ..." -> 'detector_backend.nudenet_v3.nms_iou'.
    The longest setting path the text starts with, or '' when it starts with none.
    """
    # "x['a']'s resolved item_overrides['b']" names x.a.item_overrides.b.
    match = _LOCATION.match( str( text ).replace( "]'s resolved ", "]." ) )
    if not match:
        return ''
    parts = [ match.group( 1 ) ]
    for subscript, name in re.findall( r"\[([^\]]*)\]|\.([A-Za-z_][A-Za-z0-9_]*)", match.group( 2 ) ):
        parts.append( name or subscript.strip( "'\"" ) )
    return '.'.join( parts )


def _is_number( value ):
    return isinstance( value, ( int, float ) ) and not isinstance( value, bool )


def _is_whole( value ):
    return isinstance( value, int ) and not isinstance( value, bool )


def _range_text( spec ):
    low, high = spec.get( 'minimum' ), spec.get( 'maximum' )
    parts = []
    if low is not None:
        parts.append( '%s %r'%( 'above' if spec.get( 'min_open' ) else 'at least', low ) )
    if high is not None:
        parts.append( '%s %r'%( 'below' if spec.get( 'max_open' ) else 'at most', high ) )
    return ' and '.join( parts )


def _range_params( spec ):
    """
    The bounds an out_of_range error carries: always the same six, as
    the protocol requires (None where there is no bound).
    """
    return { 'minimum': spec.get( 'minimum' ), 'maximum': spec.get( 'maximum' ),
             'min_open': bool( spec.get( 'min_open' ) ), 'max_open': bool( spec.get( 'max_open' ) ),
             'odd': bool( spec.get( 'odd' ) ) }


def _in_range( value, spec ):
    low, high = spec.get( 'minimum' ), spec.get( 'maximum' )
    if low is not None and ( value <= low if spec.get( 'min_open' ) else value < low ):
        return False
    if high is not None and ( value >= high if spec.get( 'max_open' ) else value > high ):
        return False
    return True


_EXPECTED = {
    'bool': 'True or False', 'int': 'a whole number', 'number': 'a number',
    'string': 'a non-empty string', 'path': 'a non-empty path', 'paths': 'a list of paths',
    'labels': 'a list of labels',
    'color': 'an (r, g, b) tuple of 0 to 255', 'sizes': 'a list of whole numbers',
}


def check_value( where, value, spec, params_path=None ):
    """
    Problems with one value against an entry's or a field's limits.

    Args:
        where: How the English text names the value.
        value: The value.
        spec: A Setting, or a field dict from STYLE_FIELDS / RULE_FIELDS.
        params_path: The dotted path the protocol error carries; where by default.

    Returns:
        A list of Problem, empty when the value is fine. Structured kinds
        return [] here; their own checkers handle them.
    """
    if isinstance( spec, Setting ):
        spec = { name: getattr( spec, name ) for name in Setting.__slots__ }
    path = params_path or where
    kind = spec['kind']
    note = spec.get( 'note' )
    note = note( value ) if callable( note ) else note
    suffix = ( '. %s'%( note, ) ) if note else ''
    if value is None and spec.get( 'nullable' ):
        return []
    if kind in STRUCTURED_KINDS or kind == 'promotion_requirements':
        return []
    expected = _EXPECTED.get( kind )

    def bad_type( what=None ):
        what = what or expected
        return [ Problem( "%s must be %s, got %r"%( where, what + ( ' or None' if spec.get( 'nullable' ) else '' ),
                                                    value ),
                          'config.bad_type', path=path, expected=kind ) ]

    if kind == 'bool':
        return [] if isinstance( value, bool ) else bad_type()
    if kind in ( 'int', 'number' ):
        if not ( _is_whole( value ) if kind == 'int' else _is_number( value ) ):
            return bad_type()
        if not _in_range( value, spec ):
            return [ Problem( "%s must be %s, got %r%s"%( where, _range_text( spec ), value, suffix ),
                              'config.out_of_range', path=path,
                              **_range_params( spec ) ) ]
        if spec.get( 'odd' ) and value % 2 == 0:
            return [ Problem( "%s must be an odd number, got %r"%( where, value ),
                              'config.out_of_range', path=path, **_range_params( spec ) ) ]
        return []
    if kind == 'choice':
        choices = spec['choices']() if callable( spec['choices'] ) else spec['choices']
        # True and False equal 1 and 0, so gpu_enabled = True is a valid 1.
        if value in choices:
            return []
        return [ Problem( "%s = %r is not one of %s%s"%( where, value, list( choices ), suffix ),
                          'config.bad_choice', path=path, choices=list( choices ) ) ]
    if kind in ( 'string', 'path' ):
        return [] if isinstance( value, str ) and value else bad_type()
    if kind == 'paths':
        if isinstance( value, ( list, tuple ) ) and all( isinstance( item, str ) and item for item in value ):
            return []
        return bad_type()
    if kind == 'labels':
        if not isinstance( value, ( list, tuple ) ):
            return bad_type()
        choices = spec['choices']() if callable( spec['choices'] ) else spec['choices']
        unknown = [ item for item in value if item not in choices ]
        if unknown:
            return [ Problem( "%s has unknown label(s) %s (valid: %s)"%( where, unknown, list( choices ) ),
                              'config.bad_choice', path=path, choices=list( choices ) ) ]
        return []
    if kind == 'color':
        if ( isinstance( value, ( list, tuple ) ) and len( value ) == 3
                and all( _is_whole( part ) and 0 <= part <= 255 for part in value ) ):
            return []
        return bad_type()
    if kind == 'sizes':
        if isinstance( value, ( list, tuple ) ) and value and all(
                _is_whole( size ) and size >= 0 for size in value ):
            return []
        return bad_type( 'a non-empty list of whole numbers >= 0' )
    raise ValueError( "unknown kind %r"%( kind, ) )


# ---------------------------------------------------------------------------
# Walking a config
# ---------------------------------------------------------------------------

def concrete_leaves( namespace ):
    """
    ( path, value, entry ) for every registered leaf the config sets.

    Walks dicts down to registered entries; a structured entry stops the
    walk and is yielded whole. A path with no entry is yielded with entry
    None, so callers can warn or reject.
    """
    import betautils_local as bu_local
    for name in bu_local.known_names( namespace ):
        yield from _walk( ( name, ), namespace[name] )


def _walk( path, value ):
    entry = find( path )
    if entry is not None and len( entry.path ) == len( path ):
        yield path, value, entry
        return
    if isinstance( value, dict ) and value:
        for key, child in value.items():
            yield from _walk( path + ( key, ), child )
        return
    if isinstance( value, dict ) and is_container( path ):
        return   # an empty block, such as a profile that changes nothing
    yield path, value, entry if ( entry is not None and len( entry.path ) <= len( path ) ) else None


# ---------------------------------------------------------------------------
# CONFIG_REFERENCE.md anchors
# ---------------------------------------------------------------------------

def heading_anchors( markdown ):
    """GitHub's anchor for every heading outside a code fence, duplicates numbered."""
    anchors = []
    seen = {}
    inside_fence = False
    for line in markdown.splitlines():
        if line.startswith( '```' ):
            inside_fence = not inside_fence
            continue
        if inside_fence:
            continue
        match = re.match( r'^#{1,6} (.*)', line )
        if not match:
            continue
        slug = re.sub( r'[^\w\- ]', '', match.group( 1 ).strip().lower() ).replace( ' ', '-' )
        count = seen.get( slug, 0 )
        seen[slug] = count + 1
        anchors.append( slug if not count else '%s-%d'%( slug, count ) )
    return anchors


# ---------------------------------------------------------------------------
# The schema the engine sends
# ---------------------------------------------------------------------------

def _jsonable( value ):
    if isinstance( value, tuple ):
        return [ _jsonable( item ) for item in value ]
    if isinstance( value, list ):
        return [ _jsonable( item ) for item in value ]
    if isinstance( value, dict ):
        return { str( key ): _jsonable( item ) for key, item in value.items() }
    return value


def _field_schema( spec ):
    out = { 'kind': spec['kind'], 'tip': 'setting.%s.tip'%( spec['family'], ),
            'label': 'setting.%s.label'%( spec['family'], ) }
    for key in ( 'minimum', 'maximum' ):
        if spec.get( key ) is not None:
            out[key] = spec[key]
    for key in ( 'min_open', 'max_open' ):
        if spec.get( key ):
            out[key] = True
    if spec.get( 'choices' ) is not None:
        choices = spec['choices']() if callable( spec['choices'] ) else spec['choices']
        out['choices'] = _jsonable( list( choices ) )
    return out


def schema():
    """The registry as plain JSON types, with dynamic choices resolved."""
    settings = []
    for entry in REGISTRY:
        item = {
            'path': entry.dotted, 'kind': entry.kind, 'scope': entry.scope, 'level': entry.level,
            'moves': sorted( entry.moves ), 'may_move': sorted( entry.may_move ),
            'when': list( entry.when ), 'per_run': entry.per_run,
            'tip': entry.tip, 'label': entry.label_key, 'help': entry.help,
            'screen': entry.screen,
            'requires': [ { 'path': path, 'test': test } for path, test in entry.requires ],
        }
        for key in ( 'minimum', 'maximum' ):
            if getattr( entry, key ) is not None:
                item[key] = getattr( entry, key )
        for key in ( 'min_open', 'max_open', 'nullable', 'odd' ):
            if getattr( entry, key ):
                item[key] = True
        choices = entry.choice_list()
        if choices is not None:
            item['choices'] = _jsonable( choices )
        if entry.none_means:
            item['none'] = 'setting.%s.none'%( entry.family, )
        settings.append( item )
    styles = { 'common': { name: _field_schema( spec ) for name, spec in STYLE_COMMON_FIELDS.items() },
               'types': { style_type: { name: _field_schema( spec ) for name, spec in fields.items() }
                          for style_type, fields in STYLE_TYPE_FIELDS.items() },
               'frame_extra': { name: _field_schema( spec )
                                for name, spec in FRAME_STYLE_EXTRA_FIELDS.items() } }
    rules = { kind: { name: _field_schema( spec ) for name, spec in fields.items() }
              for kind, fields in RULE_FIELDS.items() }
    return {
        'settings': settings,
        'styles': styles,
        'rules': rules,
        'placeholders': {
            '<backend>': backend_names(), '<label>': label_names(),
            '<model_variant>': model_variant_names(), '<person_backend>': person_backend_names(),
            '<style_type>': list( CENSOR_STYLE_TYPES ),
        },
        'labels': label_table(),
        'backends': { name: { 'labels': sorted( backend_labels( name ) ) } for name in backend_names() },
        'conditions': sorted( CONDITIONS ),
        'screens': list( SCREENS ),
        'geometry_actions': geometry_actions(),
        'catalog': 'i18n/en-US/settings.json',
        'help': 'CONFIG_REFERENCE.md',
    }
