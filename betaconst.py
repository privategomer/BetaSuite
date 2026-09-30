"""
betaconst.py - Structural constants. Nothing here is user-tunable.

These are facts about model I/O contracts, the on-disk layout, and the
BetaVision IPC convention. User-tunable settings live in betaconfig.py;
the reasoning behind every tunable lives in CONFIG_REFERENCE.md.

Do not rename or reorder anything here without checking every caller.
"""

# --- Detection confidence floor (fallback only) -----------------------------

# Fallback for betaconfig.global_min_prob. Only read if betaconfig.py
# does not define it. See CONFIG_REFERENCE.md -> global_min_prob.
global_min_prob = 0.20

# --- Cache / versioning ------------------------------------------------

# Bumped whenever the on-disk shape of a cached raw box record changes,
# so old caches are recognised as stale rather than misread.
#
#   1 -> 2 (v2.0.0)   class_id became a canonical string label
#   2 -> 3 (v2.1.0)   raw boxes carry 'size' (the picture_sizes entry
#                     that produced them) for cross-size dedup, and the
#                     cache path format changed to a content-addressed
#                     detection key (see betautils_cache_paths.py)
picture_saved_box_version = 3

# Hex characters kept from each short hash embedded in a filename.
ptb_hash_len = 4          # legacy censor hash width (betastare.py filenames)
detection_key_len = 6     # detection-identity key   -> "d<key>"
censor_key_len = 6        # censor/tracking key      -> "c<key>"
encode_key_len = 4        # final-encode key         -> "e<key>"

# --- Neural net input/output tensor names (retinanet_v2 only) ---------------

model_outputs = [
        'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',   # boxes
        'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0', # scores
        'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0', # classes
]

model_input = 'input_1:0'

# --- Default input/output directory layout ----------------------------------

picture_path_uncensored = '../resources/uncensored_pics/'
picture_path_censored   = '../output/censored_pics/'

video_path_uncensored = '../resources/uncensored_vids/'
video_path_censored   = '../output/censored_vids/'

# Secondary archive of source footage. Analysis tools fall back to this
# when a cached detection's source file is no longer under
# video_path_uncensored.
video_path_source_backup = '../resources/source/'

# --- Cache directory layout -------------------------------------------------
#
# Every path under here is built by betautils_cache_paths.py. Nothing
# else in the codebase may construct one of these paths itself; see that
# module's docstring for why.

cache_root            = '../output/cache'
vid_hash_dir          = '../output/cache/vid_hashes'
pic_hash_dir          = '../output/cache/pic_hashes'
shot_cut_dir          = '../output/cache/shot_cuts'
transcode_cache_dir   = '../output/cache/transcode_cache'
file_hash_cache_path  = '../output/cache/file_hashes.json'
run_key_dir           = '../output/cache/run_keys'
benchmark_dir         = '../output/benchmarks'

# --- Detection class labels --------------------------------------------
#
# BetaSuite's own canonical vocabulary, not any one model's class list.
# Each detector adapter translates its model's native classes into these
# labels. The RGB value is used only by the debug overlay.
#
# When a new adapter offers a distinction this vocabulary lacks, ADD a
# label rather than collapsing it into an existing one.

classes = {
    'exposed_anus':     ( 47,  79,  79), # darkslategray
    'exposed_armpits':  (139,  69,  19), # saddlebrown
    'covered_belly':    (  0, 100,   0), # darkgreen
    'exposed_belly':    (  0,   0, 139), # darkblue
    'covered_buttocks': (255,   0,   0), # red
    'exposed_buttocks': (255, 165,   0), # orange
    'face_femme':       (255, 255,   0), # yellow
    'face_masc':        (199,  21, 133), # mediumvioletred
    'covered_feet':     (  0, 255,   0), # lime
    'exposed_feet':     (  0, 250, 154), # mediumspringgreen
    'covered_breast':   (  0, 255, 255), # aqua
    'exposed_breast':   (  0,   0, 255), # blue
    'covered_vulva':    (216, 191, 216), # thistle
    'exposed_vulva':    (255,   0, 255), # fuchsia
    'exposed_chest':    ( 30, 144, 255), # dodgerblue
    'exposed_penis':    (240, 230, 140), # khaki
    'covered_armpits':  (160,  82,  45), # sienna
    'covered_anus':     (220,  20,  60), # crimson
}

# --- BetaVision shared-memory / IPC names -----------------------------------

bv_ss_timestamp1_name = 'bv_ss_timestamp1_name'
bv_ss_timestamp2_name = 'bv_ss_timestamp2_name'
bv_detect_timestamp1_name = 'bv_detect_timestamp1_name'
bv_detect_timestamp2_name = 'bv_detect_timestamp2_name'

bv_detect_shm_boxes_name = "bv_detect_shm_boxes_name"
bv_detect_shm_count_name = "bv_detect_shm_count_name"

# Preallocated box-record slots in the BetaVision shared-memory segment.
# Overflow beyond this is dropped (see betautils_vision.boxes_to_shared_array).
bv_detect_max_boxes = 2000
