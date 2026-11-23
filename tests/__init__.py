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

# The shipped defaults, whatever betaconfig_local.py this checkout has:
# tests that check the shipped config, or count on its encoder or worker
# settings, must see the same values on every machine. Set before
# betaconfig is imported, and inherited by every subprocess a test
# starts. Tests of the local config pass their own paths.
os.environ['BETASUITE_LOCAL_CONFIG'] = 'none'

TEST_OUTPUT_ROOT = tempfile.mkdtemp( prefix='betasuite-tests-' )
atexit.register( shutil.rmtree, TEST_OUTPUT_ROOT, ignore_errors=True )

# A data root in the caller's environment points at real footage and
# caches. Dropped before betaconst resolves its paths, and so for every
# subprocess a test starts too; tests of the data root set their own.
os.environ.pop( 'BETASUITE_DATA_ROOT', None )

# Without a data root the app uses the OS folders by kind, which are the
# user's real ones (~/.cache/betasuite, %LOCALAPPDATA%\BetaSuite). Every
# variable those folders come from points under the temp root instead,
# before betaconst resolves anything and for every subprocess. The input
# and output folders follow the Videos folder: on XDG systems through a
# user-dirs.dirs in the temp config folder, on Windows through the
# profile folder. The folder overrides are dropped, not set, since they
# cannot be combined with a data root and tests of the root set one.
# Tests of the layout pass their own environment.
TEST_OS_ROOT = os.path.join( TEST_OUTPUT_ROOT, 'os' )
TEST_OS_ENV = {
    'XDG_CONFIG_HOME': os.path.join( TEST_OS_ROOT, 'config' ),
    'XDG_DATA_HOME':   os.path.join( TEST_OS_ROOT, 'data' ),
    'XDG_CACHE_HOME':  os.path.join( TEST_OS_ROOT, 'cache' ),
    'XDG_STATE_HOME':  os.path.join( TEST_OS_ROOT, 'state' ),
    'LOCALAPPDATA':    os.path.join( TEST_OS_ROOT, 'localappdata' ),
}
if os.name == 'nt':
    TEST_OS_ENV['USERPROFILE'] = os.path.join( TEST_OS_ROOT, 'profile' )
os.environ.update( TEST_OS_ENV )
for _name in ( 'BETASUITE_CACHE_DIR', 'BETASUITE_INPUT_DIR', 'BETASUITE_OUTPUT_DIR' ):
    os.environ.pop( _name, None )
os.makedirs( TEST_OS_ENV['XDG_CONFIG_HOME'] )
with open( os.path.join( TEST_OS_ENV['XDG_CONFIG_HOME'], 'user-dirs.dirs' ), 'w', encoding='utf-8' ) as _handle:
    _handle.write( 'XDG_VIDEOS_DIR="%s"\n'%( os.path.join( TEST_OS_ROOT, 'videos' ), ) )

# The data folder exists, as after a first run, so the warning about a 1.x
# tree beside the code (which a developer's own checkout has) never adds
# a line to what a test sees. Tests of that warning pass their own folders.
import betautils_paths
os.makedirs( betautils_paths.kind_dir( 'data' ), exist_ok=True )

import betaconfig
import betaconst
import betautils_log

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
    'video_path_completed':     _under_root( 'resources', 'source', 'completed' ) + os.sep,
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

# betaconfig.py ships items_to_censor = [] (a fresh install chooses what
# to censor), but most of the suite is about censoring: tracking, styles,
# shot cuts, the structure bench. Without labels every one of those
# would pass vacuously or fail on an empty render. The suite censors the
# two labels the shipped tuning covers. Tests of the shipped default read
# betaconfig.py's source (tests/test_shipped_defaults.py), and tests
# about an empty list set it themselves.
SUITE_ITEMS_TO_CENSOR = [ 'exposed_vulva', 'exposed_breast' ]
betaconfig.items_to_censor = list( SUITE_ITEMS_TO_CENSOR )
