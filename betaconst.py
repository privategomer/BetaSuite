"""
betaconst.py - Structural constants. Nothing here is user-tunable.

These are facts about model I/O contracts and the on-disk layout.
User-tunable settings live in betaconfig.py;
the reasoning behind every tunable lives in CONFIG_REFERENCE.md.

Do not rename or reorder anything here without checking every caller.
"""

import betautils_labels as _bu_labels
import betautils_paths as _bu_paths

# --- Detection confidence floor (fallback only) -----------------------------

# Fallback for betaconfig.global_min_prob. Only read if betaconfig.py
# does not define it. See CONFIG_REFERENCE.md -> global_min_prob.
global_min_prob = 0.20

# --- Cache / versioning ------------------------------------------------

# Bumped whenever the on-disk shape of a cached raw box record changes,
# so old caches are recognised as stale rather than misread.
#
#   1 -> 2   class_id became a canonical string label
#   2 -> 3   raw boxes carry 'size' (the picture_sizes entry
#                     that produced them) for cross-size dedup, and the
#                     cache path format changed to a content-addressed
#                     detection key (see betautils_cache_paths.py)
picture_saved_box_version = 3

# Hex characters kept from each short hash embedded in a filename.
ptb_hash_len = 4          # legacy censor hash width (betastare.py filenames)
detection_key_len = 6     # detection-identity key   -> "d<key>"
censor_key_len = 6        # censor/tracking key      -> "c<key>"
encode_key_len = 4        # final-encode key         -> "e<key>"

# --- RetinaNet model tensor names --------------------------------------------

# Fixed by the exported model file; used only by detectors/retinanet_v2.py.
# session.run returns the outputs in this order: per-image box corners,
# their confidence scores, then their class indexes.
model_outputs = [
    'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',
    'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',
    'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0',
]
# The batch of preprocessed images goes in here.
model_input = 'input_1:0'

# --- Default input/output directory layout ----------------------------------
#
# Named by their 1.x paths and resolved once, here: the OS folders by kind,
# or the 1.x tree under BETASUITE_DATA_ROOT (betautils_paths.py).

picture_path_uncensored = _bu_paths.resolve( '../resources/uncensored_pics/' )
picture_path_censored   = _bu_paths.resolve( '../output/censored_pics/' )

video_path_uncensored = _bu_paths.resolve( '../resources/uncensored_vids/' )
video_path_censored   = _bu_paths.resolve( '../output/censored_vids/' )

# Secondary archive of source footage. Analysis tools fall back to this
# when a cached detection's source file is no longer under
# video_path_uncensored.
video_path_source_backup = _bu_paths.resolve( '../resources/source/' )
# Finished sources, when move_completed_sources is on; profile folders mirror
# video_path_uncensored's.
video_path_completed = _bu_paths.resolve( '../resources/source/completed/' )

# --- Cache directory layout -------------------------------------------------
#
# Every path under here is built by betautils_cache_paths.py. Nothing
# else in the codebase may construct one of these paths itself; see that
# module's docstring for why.

cache_root            = _bu_paths.resolve( '../output/cache' )
vid_hash_dir          = _bu_paths.resolve( '../output/cache/vid_hashes' )
pic_hash_dir          = _bu_paths.resolve( '../output/cache/pic_hashes' )
shot_cut_dir          = _bu_paths.resolve( '../output/cache/shot_cuts' )
transcode_cache_dir   = _bu_paths.resolve( '../output/cache/transcode_cache' )
look_preview_dir      = _bu_paths.resolve( '../output/cache/look_previews' )
file_hash_cache_path  = _bu_paths.resolve( '../output/cache/file_hashes.json' )
run_key_dir           = _bu_paths.resolve( '../output/cache/run_keys' )
benchmark_dir         = _bu_paths.resolve( '../output/benchmarks' )

# --- Run lock ---------------------------------------------------------------
#
# One run at a time per library, CLI and desktop app alike
# (betautils_runlock.py). It sits with the data it protects.

run_lock_path         = _bu_paths.resolve( '../output/run.lock' )

# --- Detection class labels --------------------------------------------
#
# BetaSuite's own canonical vocabulary, not any one model's class list.
# Each detector adapter translates its model's native classes into these
# labels. The RGB value is used only by the debug overlay.
#
# When a new adapter offers a distinction this vocabulary lacks, ADD a
# label rather than collapsing it into an existing one. Labels are
# described in betautils_labels.py; this is their debug overlay colour.

classes = _bu_labels.debug_colors()
