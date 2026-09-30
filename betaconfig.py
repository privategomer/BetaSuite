# BetaSuite configuration.
#
# Every setting here is explained in CONFIG_REFERENCE.md, with what it
# does, what happens when you change it, and why the tuned values are
# what they are. README.md covers how a run works end to end.
#
# House rule: this file holds VALUES and one-line headers only. Rationale,
# tuning history and trade-offs live in CONFIG_REFERENCE.md.
#
# validate_config() (betautils_config.py) checks this file on every run
# and fails fast with a specific message before doing any real work.

# --- Detector backend ---------------------------------------------------
# Which model runs, and its per-model settings. Only 'selected' and the
# matching block are read; 'defaults' fills gaps for every backend.
# See CONFIG_REFERENCE.md -> "Detector backend".

detector_backend = {
    'selected': 'nudenet_v3',

    # Shared across every backend unless that backend overrides it.
    'defaults': {
        'nn_batch_size': 2,
    },

    'nudenet_v3': {
        # '320n' (12MB nano net, native 320) or '640m' (103MB medium
        # net, native 640). Letter = capacity, number = input size.
        'model_variant': '640m',
        'nn_batch_size': 4,
        'candidate_floor': 0.2,
        'nms_iou': 0.45,
        # 'per_class' keeps cross-label overlaps alive for
        # class_suppression to arbitrate. 'agnostic' is NudeNet's own
        # upstream behaviour.
        'nms_mode': 'per_class',

        # Timing values selected per video by measured shot-cut rate.
        # See CONFIG_REFERENCE.md -> "Structure profiles".
        'profiles': {
            'default': 'scene',
            'match_on': 'short_shot_fraction',
            'variants': {
                # Rapid cutting. Denser sampling, tight gaps so censoring
                # cannot bleed across a cut, longer style dwell so the
                # style does not re-roll on every shot.
                'quick_cut': {
                    'min_short_shot_fraction': 0.15,
                    'video_censor_fps': 18,
                    'item_overrides': {
                        'exposed_breast': { 'track_max_gap': 0.20,
                                            'interpolation_max_gap': 0.14,
                                            'time_safety': 0.06,
                                            'min_track_hits': 2,
                                            'paired_style_max_age': 0.25,
                                            'style_min_dwell_seconds': 2.5 },
                        'exposed_vulva':  { 'track_max_gap': 0.20,
                                            'interpolation_max_gap': 0.14,
                                            'time_safety': 0.06,
                                            'min_track_hits': 2,
                                            'style_min_dwell_seconds': 2.5 },
                        'covered_vulva':  { 'track_max_gap': 0.20,
                                            'interpolation_max_gap': 0.14,
                                            'time_safety': 0.06,
                                            'min_track_hits': 2,
                                            'style_min_dwell_seconds': 2.5 },
                    },
                },
                # Split-screen layouts (double, triple, quadrant). Not
                # auto-selected: no signal available before detection
                # separates it from 'scene'. Reach it with
                # --profile split_screen or profile_by_path_pattern.
                'split_screen': {
                    'manual_only': True,
                    'item_overrides': {
                        'exposed_breast': { 'track_max_gap': 2.0,
                                            'interpolation_max_gap': 0.8,
                                            'match_distance_multiplier': 0.8,
                                            'paired_style': True,
                                            'paired_style_max_distance': 0.9,
                                            'paired_style_tiebreak_margin': 0.12,
                                            'paired_style_max_age': 0.5,
                                            'style_min_dwell_seconds': 6.0 },
                        'exposed_vulva':  { 'track_max_gap': 4.0,
                                            'interpolation_max_gap': 4.0,
                                            'match_distance_multiplier': 1.0,
                                            'style_min_dwell_seconds': 6.0 },
                        'covered_vulva':  { 'track_max_gap': 4.0,
                                            'interpolation_max_gap': 4.0,
                                            'match_distance_multiplier': 1.0,
                                            'style_min_dwell_seconds': 6.0 },
                    },
                },
                # Single continuous scene. The default, and the catch-all
                # for anything the quick-cut ceiling does not claim, so it
                # carries no threshold of its own.
                'scene': {
                    'item_overrides': {
                        'exposed_breast': { 'track_max_gap': 3.0,
                                            'interpolation_max_gap': 1.0,
                                            'style_min_dwell_seconds': 3.0 },
                        # Vulva labels bridge much longer gaps than breast:
                        # see CONFIG_REFERENCE.md -> "Interpolating vulva
                        # dropouts within a scene".
                        'exposed_vulva':  { 'track_max_gap': 8.0,
                                            'interpolation_max_gap': 8.0,
                                            'style_min_dwell_seconds': 3.0 },
                        'covered_vulva':  { 'track_max_gap': 8.0,
                                            'interpolation_max_gap': 8.0,
                                            'style_min_dwell_seconds': 3.0 },
                    },
                },
            },
        },

        # Relabel a detection when corroborating evidence overlaps it.
        # See CONFIG_REFERENCE.md -> "Class promotion".
        'class_promotion': {
            'exposed_vulva': [
                { 'from': 'covered_vulva', 'min_prob': 0.30,
                  'requires': [
                      { 'label': 'exposed_penis', 'min_prob': 0.30, 'overlap': 'anywhere' },
                      { 'label': 'exposed_vulva', 'min_prob': 0.30, 'overlap': 'anywhere' },
                      { 'label': 'exposed_anus',  'min_prob': 0.30, 'overlap': 'anywhere' },
                  ] },
            ],
        },

        # See CONFIG_REFERENCE.md -> "Re-deriving suppression rules".
        'class_suppression': {
            'exposed_breast': [
                { 'suppressed_by': 'face_femme',       'margin': 0.105, 'min_iou': 0.010 },
                { 'suppressed_by': 'exposed_belly',    'margin': 0.069, 'min_iou': 0.010 },
                { 'suppressed_by': 'exposed_buttocks', 'margin': 0.119, 'min_iou': 0.010 },
                { 'suppressed_by': 'covered_breast',   'margin': 0.045, 'min_iou': 0.029 },
            ],
            'exposed_vulva': [
                { 'suppressed_by': 'face_femme',    'margin': 0.255, 'min_iou': 0.010 },
                { 'suppressed_by': 'exposed_breast','margin': 0.231, 'min_iou': 0.022 },
                { 'suppressed_by': 'exposed_belly', 'margin': 0.293, 'min_iou': 0.015 },
                { 'suppressed_by': 'covered_breast','margin': 0.295, 'min_iou': 0.016 },
                { 'suppressed_by': 'exposed_anus',  'margin': 0.113, 'min_iou': 0.080 },
            ],
        },

        # Detection- and tracking-sensitive per-label settings
        'item_overrides': {
            'exposed_vulva': {
                'min_prob': 0.18,
                'time_safety': 0.16,
                'height_area_safety': 0.00,
                'width_area_safety': 0.00,
                'interpolation_max_gap': 3.6,
                'track_max_gap': 43.2,  # set by auto_tune 2026-09-19
                'position_smoothing': 0.75,
                'paired_style': False,
                'min_track_hits': 3,  # set by auto_tune 2026-09-30
                'match_distance_multiplier': 3,  # set by auto_tune 2026-09-18
            },
            'exposed_breast': {
                'min_prob': 0.205,
                'time_safety': 0.17,
                'width_area_safety': 0.00,
                'height_area_safety': 0.00,
                'interpolation_max_gap': 0.15,
                'track_max_gap': 0.17,
                'paired_style': True,
                'paired_style_max_age': 0.8,
                'min_track_hits': 3,  # set by auto_tune 2026-09-30
                'position_smoothing': 0.60,
                'match_distance_multiplier': 2.0,  # set by auto_tune 2026-09-30
            },
            'covered_vulva': {
                'min_prob': 0.49,
                'time_safety': 0.16,
                'height_area_safety': 0.00,
                'width_area_safety': 0.00,
                'interpolation_max_gap': 3.6,
                'track_max_gap': 43.2,
                'paired_style': False,
                'min_track_hits': 3,  # set by auto_tune 2026-09-30
                'match_distance_multiplier': 3,
                'position_smoothing': 0.75,
            },
        },
    },

    'retinanet_v2': {
        'picture_sizes': [ 1280 ],
        'nn_batch_size': 2,

        'class_suppression': {
            'exposed_breast': [
                { 'suppressed_by': 'covered_breast',   'margin': 0.10, 'min_iou': 0.55 },
                { 'suppressed_by': 'face_femme',       'margin': 0.10, 'min_iou': 0.30 },
                { 'suppressed_by': 'face_masc',        'margin': 0.14, 'min_iou': 0.75 },
                { 'suppressed_by': 'exposed_belly',    'margin': 0.12, 'min_iou': 0.12 },
                { 'suppressed_by': 'exposed_buttocks', 'margin': 0.13, 'min_iou': 0.30 },
                { 'suppressed_by': 'exposed_chest',    'margin': 0.14, 'min_iou': 0.85 },
                { 'suppressed_by': 'exposed_penis',    'margin': 0.10, 'min_iou': 0.70 },
            ],
            'exposed_vulva': [
                { 'suppressed_by': 'exposed_anus',    'margin': 0.17, 'min_iou': 0.125 },
                { 'suppressed_by': 'covered_vulva',   'margin': 0.14, 'min_iou': 0.35 },
                { 'suppressed_by': 'face_femme',      'margin': 0.14, 'min_iou': 0.30 },
                { 'suppressed_by': 'face_masc',       'margin': 0.12, 'min_iou': 0.30 },
                { 'suppressed_by': 'exposed_penis',   'margin': 0.10, 'min_iou': 0.70 },
                { 'suppressed_by': 'exposed_armpits', 'margin': 0.14, 'min_iou': 0.60 },
            ],
        },

        'item_overrides': {
            'exposed_vulva': {
                'min_prob': 0.21,
                'time_safety': 0.27,
                'height_area_safety': 0.00,
                'width_area_safety': 0.00,
                'interpolation_max_gap': 1.2,
                'track_max_gap': 7.2,
                'paired_style': True,
                'match_distance_multiplier': 1.25,  # set by auto_tune 2026-09-18
                'min_track_hits': 2,  # set by auto_tune 2026-09-18
            },
            'exposed_breast': {
                'min_prob': 0.37,
                'time_safety': 0.35,
                'width_area_safety': 0.00,
                'height_area_safety': 0.00,
                'interpolation_max_gap': 0.5,
                'track_max_gap': 13.5,
                'paired_style': True,
                'min_track_hits': 3,  # set by auto_tune 2026-09-18
            },
        },
    },
}

# --- Per-item overrides -------------------------------------------------
# Shared across every backend. ONLY censor_shape and censor_style belong
# here; every detection- or tracking-sensitive key lives in that
# backend's own detector_backend[<name>]['item_overrides'] block.

item_overrides = {
    # covered_vulva is censored for penetrative content (see
    # CONFIG_REFERENCE.md). Blur only, no stickers: a sticker reads as a
    # deliberate playful cover-up, and this label also fires on ordinary
    # clothed crotches, where that would look like a mistake rather than
    # a censor.
    'covered_vulva': {
        'censor_shape': 'ellipse',
        'censor_style': [
            # Weights sum to 100, so each weight IS its percent of resolves.
            { 'type': 'blur', 'method': 'box', 'strength': 70, 'feather': 0.10, 'weight': 33.34, 'width_area_safety': 0.00, 'height_area_safety': 0.10 },
            { 'type': 'blur', 'method': 'box', 'strength': 85, 'feather': 0.15, 'weight': 33.33, 'width_area_safety': 0.00, 'height_area_safety': 0.10 },
            { 'type': 'blur', 'method': 'box', 'strength': 100, 'feather': 0.20, 'weight': 33.33, 'width_area_safety': 0.00, 'height_area_safety': 0.10 },
        ],
    },

    'exposed_vulva': {
        'censor_shape': 'ellipse',
        'censor_style': [
            # Weights sum to 100, so each weight IS its percent of resolves.
            # Family shares: blur 78, sticker 22.

            # Blur
            { 'type': 'blur', 'method': 'box', 'strength': 70, 'feather': 0.10, 'weight': 21.98, 'width_area_safety': 0.00, 'height_area_safety': 0.10 },
            { 'type': 'blur', 'method': 'box', 'strength': 85, 'feather': 0.15, 'weight': 21.98, 'width_area_safety': 0.00, 'height_area_safety': 0.10 },
            { 'type': 'blur', 'method': 'box', 'strength': 85, 'feather': 0.20, 'weight': 12.09, 'width_area_safety': -0.25, 'height_area_safety': 0.00 },
            { 'type': 'blur', 'method': 'box', 'strength': 100, 'feather': 0.20, 'weight': 21.98, 'width_area_safety': 0.00, 'height_area_safety': 0.10 },

            # Sticker
            { 'type': 'sticker', 'dir': '../resources/stickers/vulva/', 'weight': 7.25, 'feather': 0.15, 'width_area_safety': 0.10, 'height_area_safety': 0.10 },
            { 'type': 'sticker', 'dir': '../resources/stickers/vulva/', 'weight': 7.25, 'feather': 0.20, 'width_area_safety': 0.25, 'height_area_safety': 0.25 },
            { 'type': 'sticker', 'dir': '../resources/stickers/vulva/', 'weight': 7.47, 'feather': 0.25, 'width_area_safety': 0.50, 'height_area_safety': 0.50 },
        ],
    },

    'exposed_breast': {
        'censor_shape': 'circle',
        'censor_style': [
            # Weights sum to 100, so each weight IS its percent of resolves.
            # Family shares: blur 20, mosaic 25, hex 25, bar 12, sticker 18.

            # Blur, triple_box. Last two are the tight-crop variants.
            { 'type': 'blur',  'method': 'triple_box', 'shape': 'ellipse', 'strength': 40, 'feather': 0.25, 'weight': 2.12, 'width_area_safety':  0.00, 'height_area_safety':  0.00 },
            { 'type': 'blur',  'method': 'triple_box', 'shape': 'ellipse', 'strength': 50, 'feather': 0.27, 'weight': 4.54, 'width_area_safety':  0.00, 'height_area_safety':  0.00 },
            { 'type': 'blur',  'method': 'triple_box', 'shape': 'ellipse', 'strength': 60, 'feather': 0.45, 'weight': 4.54, 'width_area_safety':  0.00, 'height_area_safety':  0.00 },
            { 'type': 'blur',  'method': 'triple_box', 'shape': 'ellipse', 'strength': 70, 'feather': 0.55, 'weight': 4.54, 'width_area_safety': -0.05, 'height_area_safety': -0.05 },
            { 'type': 'blur',  'method': 'triple_box', 'shape': 'ellipse', 'strength': 50, 'feather': 0.27, 'weight': 2.12, 'width_area_safety': -0.10, 'height_area_safety': -0.10 },
            { 'type': 'blur',  'method': 'triple_box', 'shape': 'ellipse', 'strength': 60, 'feather': 0.45, 'weight': 2.12, 'width_area_safety': -0.10, 'height_area_safety': -0.10 },

            # Pixel, mosaic. Last two are the tight-crop variants.
            { 'type': 'pixel', 'pattern': 'mosaic',  'shape': 'ellipse', 'strength': 20, 'feather': 0.25, 'weight': 2.78, 'width_area_safety':  0.00, 'height_area_safety':  0.00 },
            { 'type': 'pixel', 'pattern': 'mosaic',  'shape': 'ellipse', 'strength': 30, 'feather': 0.27, 'weight': 5.56, 'width_area_safety':  0.10, 'height_area_safety':  0.10 },
            { 'type': 'pixel', 'pattern': 'mosaic',  'shape': 'ellipse', 'strength': 40, 'feather': 0.35, 'weight': 5.56, 'width_area_safety':  0.05, 'height_area_safety':  0.05 },
            { 'type': 'pixel', 'pattern': 'mosaic',  'shape': 'ellipse', 'strength': 50, 'feather': 0.45, 'weight': 5.56, 'width_area_safety':  0.00, 'height_area_safety':  0.00 },
            { 'type': 'pixel', 'pattern': 'mosaic',  'shape': 'ellipse', 'strength': 30, 'feather': 0.27, 'weight': 2.78, 'width_area_safety': -0.10, 'height_area_safety': -0.10 },
            { 'type': 'pixel', 'pattern': 'mosaic',  'shape': 'ellipse', 'strength': 40, 'feather': 0.35, 'weight': 2.78, 'width_area_safety': -0.10, 'height_area_safety': -0.10 },

            # Pixel, hex. Last two are the tight-crop variants.
            { 'type': 'pixel', 'pattern': 'hex',     'shape': 'ellipse', 'strength': 20, 'feather': 0.25, 'weight': 2.78, 'width_area_safety':  0.00, 'height_area_safety':  0.00 },
            { 'type': 'pixel', 'pattern': 'hex',     'shape': 'ellipse', 'strength': 30, 'feather': 0.27, 'weight': 5.56, 'width_area_safety':  0.10, 'height_area_safety':  0.10 },
            { 'type': 'pixel', 'pattern': 'hex',     'shape': 'ellipse', 'strength': 40, 'feather': 0.35, 'weight': 5.56, 'width_area_safety':  0.05, 'height_area_safety':  0.05 },
            { 'type': 'pixel', 'pattern': 'hex',     'shape': 'ellipse', 'strength': 50, 'feather': 0.45, 'weight': 5.56, 'width_area_safety':  0.00, 'height_area_safety':  0.00 },
            { 'type': 'pixel', 'pattern': 'hex',     'shape': 'ellipse', 'strength': 30, 'feather': 0.27, 'weight': 2.78, 'width_area_safety': -0.10, 'height_area_safety': -0.10 },
            { 'type': 'pixel', 'pattern': 'hex',     'shape': 'ellipse', 'strength': 40, 'feather': 0.35, 'weight': 2.78, 'width_area_safety': -0.10, 'height_area_safety': -0.10 },

            # Bar, Vertical
            { 'type': 'bar', 'shape': 'box', 'color': (0, 0, 0), 'merge': 'none', 'thickness': 1.0,   'feather': 0.15,  'weight': 4.00, 'width_area_safety': -0.35, 'height_area_safety': 0.00 },
            { 'type': 'bar', 'shape': 'box', 'color': (0, 0, 0), 'merge': 'none', 'thickness': 1.0,   'feather': 0.15,  'weight': 4.00, 'width_area_safety': -0.45, 'height_area_safety': 0.00 },
            { 'type': 'bar', 'shape': 'box', 'color': (0, 0, 0), 'merge': 'none', 'thickness': 1.0,   'feather': 0.15,  'weight': 4.00, 'width_area_safety': -0.55, 'height_area_safety': 0.00 },

            # Sticker
            { 'type': 'sticker', 'dir': '../resources/stickers/breasts/', 'weight': 2.81, 'feather': 0.25, 'width_area_safety': -0.10, 'height_area_safety': -0.10 },
            { 'type': 'sticker', 'dir': '../resources/stickers/breasts/', 'weight': 5.06, 'feather': 0.15, 'width_area_safety': 0.00, 'height_area_safety': 0.00 },
            { 'type': 'sticker', 'dir': '../resources/stickers/breasts/', 'weight': 5.06, 'feather': 0.20, 'width_area_safety': 0.13, 'height_area_safety': 0.13 },
            { 'type': 'sticker', 'dir': '../resources/stickers/breasts/', 'weight': 5.06, 'feather': 0.25, 'width_area_safety': 0.25, 'height_area_safety': 0.25 },
        ],
    },
}

# --- Hardware -----------------------------------------------------------

gpu_enabled = 1
cuda_device_id = 0

# Frames per second of video that get detection and tracking.
video_censor_fps = 9

# Hard floor applied to every raw detection before any per-label min_prob
# sees it. Part of the detection cache key.
global_min_prob = 0.12

# --- Cross-size dedup ---------------------------------------------------
# Only ever compares detections from DIFFERENT picture_sizes, so it is a
# no-op with a single size configured.

cross_size_dedup = {
    'enabled': True,
    'iou_threshold': 0.60,
}

# --- Shot-boundary (cut) detection --------------------------------------

shot_cut_detection_enabled = True
shot_cut_threshold = 0.40   # Bhattacharyya distance 0-1; higher = less sensitive

# --- Resumability and rendering -----------------------------------------

detection_checkpoint_frames = 500
render_chunk_seconds = 180
render_workers = 0          # 0 = auto (CPU count / 2, capped at 8)
render_chunk_container = 'mkv'
render_verify_frame_counts = True
ffmpeg_max_retries = 2
ffmpeg_retry_backoff_seconds = 5

# --- Encoding -----------------------------------------------------------

encode_video_codec = 'libx264'
encode_crf = 17
encode_preset = 'fast'

# --- What gets censored -------------------------------------------------

items_to_censor = [
    # 'exposed_anus',
    'exposed_vulva',
    'exposed_breast',
    # 'exposed_buttocks',
    # covered_vulva is NOT censored directly - it is promoted to
    # exposed_vulva in a penetration context instead. See
    # CONFIG_REFERENCE.md -> "covered_vulva and penetrative content".
    # 'covered_vulva',
    # 'covered_breast',
    # 'covered_buttocks',
    # 'face_femme',
    # 'exposed_belly',
    # 'covered_belly',
    # 'exposed_feet',
    # 'covered_feet',
    # 'exposed_armpits',
    # 'exposed_penis',
    # 'exposed_chest',
    # 'face_masc',
]

# --- Confidence and safety defaults -------------------------------------

default_min_prob = 0.55
default_min_prob_continue = None   # None disables score hysteresis
default_area_safety = 0
default_time_safety = 0.30

# --- Track confirmation -------------------------------------------------
# Real detections a track needs before any of its boxes render. 1 renders
# every track.

default_min_track_hits = 1

# --- Geometry sanity filter ---------------------------------------------
# All None = disabled. Derive values from your own footage with:
#   python3 tools/bench/betabench.py geometry

default_min_area_fraction = None
default_max_area_fraction = None
default_min_aspect_ratio = None
default_max_aspect_ratio = None

# --- Tracking and smoothing defaults ------------------------------------

default_position_smoothing = 0.65   # 0-1: lower = smoother, higher = snappier
default_interpolation_enabled = True
default_interpolation_max_gap = 0.5  # seconds
default_style_min_dwell_seconds = 0.0   # seconds a style is held before a cut may re-roll it
default_profile_enabled = True
default_paired_style_max_distance = 2.0
default_paired_style_tiebreak_margin = 0.25
default_paired_style_max_age = 1.0   # seconds a track may go undetected and still donate its style

# --- Censor style and shape defaults ------------------------------------

default_censor_style = { 'type': 'blur', 'method': 'triple_box', 'strength': 65 }
default_censor_shape = 'box'   # 'box' | 'circle' | 'ellipse'

type_default_shapes = {
    'bar': 'box',
}

censor_overlap_strategy = {
    'blur': 'single-pass',
    'bar': 'span',
    'pixel': 'single-pass',
    'sticker': 'none',
    'debug': 'none',
}

censor_scale_strategy = 'feature'   # 'feature' | 'image' | 'none'

# Compute heavy Gaussian blurs at reduced resolution. Visually equivalent
# at obscuring strengths, 30-60x faster.
blur_fast_approximation = True
blur_approximation_min_kernel = 21

# --- Input handling -----------------------------------------------------

input_delete_probability = 0

# --- Preview mode -------------------------------------------------------

preview_mode_enabled = False
preview_max_seconds = 20
preview_encode_preset = 'ultrafast'
preview_start_seconds = None
preview_random_slice = False

# --- Logging and stats --------------------------------------------------
# Levels: 'trace' | 'debug' | 'info' | 'warn' | 'error'

logging_enabled = True
log_path = '../output/logs/betasuite.log'
log_level = 'debug'      # the log FILE's level
console_level = 'info'   # the TERMINAL's level

stats_enabled = True
stats_path = '../output/stats/betasuite_stats.jsonl'

# --- BetaVision screen-capture region -----------------------------------

vision_cap_monitor = 0
vision_cap_top = 0
vision_cap_left = 0
vision_cap_height = 1080
vision_cap_width = 960
vision_cursor_color = (168, 93, 253)
betavision_delay = 0.5
betavision_interpolate = False

# --- Miscellaneous ------------------------------------------------------

debug_mode = 0   # 0 off | 1 debug overlay | 3 also save debug output
enable_betasuite_watermark = False
