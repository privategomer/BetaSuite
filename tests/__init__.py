"""
Test package setup: every output path points at a throwaway directory.

betaconst's paths are relative to the working directory, and the suite
runs from the repository root, so before this existed '../output' WAS
the user's real output folder. Each release.py test pass wrote ~230
lines of synthetic clip.mp4 runs into the real output/logs/betasuite.log,
interleaved with whatever production run was live, and ~48 run-key
records into output/cache/run_keys, the directory betabench reads to
discover which cached configurations exist.

This runs when unittest imports the package, which is before any test
module imports betautils_cache_paths (which copies two of these paths
into module constants at import time). Tests that set their own paths
still override these; this only changes what an unpatched test touches.
Input paths are redirected too, so no test can read the user's footage.
"""

import atexit
import os
import shutil
import tempfile

import betaconfig
import betaconst
import betautils_log

TEST_OUTPUT_ROOT = tempfile.mkdtemp( prefix='betasuite-tests-' )
atexit.register( shutil.rmtree, TEST_OUTPUT_ROOT, ignore_errors=True )

def _under_root( *parts ):
    return os.path.join( TEST_OUTPUT_ROOT, *parts )

# Every directory-valued path keeps its trailing separator where the
# original had one: some callers concatenate rather than join.
REDIRECTED_PATHS = {
    'picture_path_uncensored':  _under_root( 'resources', 'uncensored_pics' ) + os.sep,
    'picture_path_censored':    _under_root( 'output', 'censored_pics' ) + os.sep,
    'video_path_uncensored':    _under_root( 'resources', 'uncensored_vids' ) + os.sep,
    'video_path_censored':      _under_root( 'output', 'censored_vids' ) + os.sep,
    'video_path_source_backup': _under_root( 'resources', 'source' ) + os.sep,
    'cache_root':               _under_root( 'output', 'cache' ),
    'vid_hash_dir':             _under_root( 'output', 'cache', 'vid_hashes' ),
    'pic_hash_dir':             _under_root( 'output', 'cache', 'pic_hashes' ),
    'shot_cut_dir':             _under_root( 'output', 'cache', 'shot_cuts' ),
    'transcode_cache_dir':      _under_root( 'output', 'cache', 'transcode_cache' ),
    'file_hash_cache_path':     _under_root( 'output', 'cache', 'file_hashes.json' ),
    'run_key_dir':              _under_root( 'output', 'cache', 'run_keys' ),
    'benchmark_dir':            _under_root( 'output', 'benchmarks' ),
}

for _name, _path in REDIRECTED_PATHS.items():
    setattr( betaconst, _name, _path )

# betaconfig sets log_path and stats_path itself, and those win over
# betautils_log's defaults, so both layers are redirected.
TEST_LOG_PATH = _under_root( 'output', 'logs', 'betasuite.log' )
TEST_STATS_PATH = _under_root( 'output', 'stats', 'betasuite_stats.jsonl' )
betautils_log.DEFAULT_LOG_PATH = TEST_LOG_PATH
betautils_log.DEFAULT_STATS_PATH = TEST_STATS_PATH
betaconfig.log_path = TEST_LOG_PATH
betaconfig.stats_path = TEST_STATS_PATH

# The synthetic end-to-end renders encode with x264 whatever encoder the
# user's config names. Once that named h264_nvenc, every one of them
# needed a free GPU session (and failed outright without a GPU), though
# none of them is about the encoder. Tests that are about the encoder
# set their own values.
TEST_ENCODE = { 'encode_video_codec': 'libx264', 'encode_preset': 'fast', 'encode_crf': 17 }
for _name, _value in TEST_ENCODE.items():
    setattr( betaconfig, _name, _value )
