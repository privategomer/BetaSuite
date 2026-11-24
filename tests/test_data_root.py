"""
test_data_root.py - BETASUITE_DATA_ROOT keeps the 1.x layout whole.

Set, every data path is the 1.x path under the root, byte for byte.
Unset, the same names resolve to the OS folders by kind (test_paths.py
checks those per OS). Either way the strings that feed cache keys
(model paths, sticker folders) stay as configured, so every layout
computes the same keys and finds the same caches. A relative or missing
root is refused.

The betaconst checks run in a subprocess: betaconst resolves its paths
once, at import, and the suite's own import has already been redirected
by tests/__init__.py.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import betaconfig
import betautils_detector as bu_detector
import betautils_local as bu_local
import betautils_log as bu_log
import betautils_paths as bu_paths
import tests


REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
ENV = bu_paths.ENV_VAR

# What betaconst and betautils_log said before the data root existed.
TODAY = {
    'picture_path_uncensored':  '../resources/uncensored_pics/',
    'picture_path_censored':    '../output/censored_pics/',
    'video_path_uncensored':    '../resources/uncensored_vids/',
    'video_path_censored':      '../output/censored_vids/',
    'video_path_source_backup': '../resources/source/',
    'video_path_completed':     '../resources/source/completed/',
    'cache_root':               '../output/cache',
    'vid_hash_dir':             '../output/cache/vid_hashes',
    'pic_hash_dir':             '../output/cache/pic_hashes',
    'shot_cut_dir':             '../output/cache/shot_cuts',
    'transcode_cache_dir':      '../output/cache/transcode_cache',
    'file_hash_cache_path':     '../output/cache/file_hashes.json',
    'run_key_dir':              '../output/cache/run_keys',
    'benchmark_dir':            '../output/benchmarks',
    'DEFAULT_LOG_PATH':         '../output/logs/betasuite.log',
    'DEFAULT_STATS_PATH':       '../output/stats/betasuite_stats.jsonl',
}

# Prints the live layout paths and the three cache keys as JSON.
_PROBE = r'''
import json, sys
import betaconst, betautils_log, betautils_cache_paths as c, betaconfig
paths = { name: getattr( betaconst, name ) for name in NAMES if hasattr( betaconst, name ) }
paths['DEFAULT_LOG_PATH'] = betautils_log.DEFAULT_LOG_PATH
paths['DEFAULT_STATS_PATH'] = betautils_log.DEFAULT_STATS_PATH
plain = c.censor_key()
# A sticker folder is part of the censor key, as a local config writes it.
betaconfig.default_probe_sticker = { 'type': 'sticker', 'dir': '../resources/stickers/probe/' }
sticker = c.censor_key()
betaconfig.default_probe_sticker = { 'type': 'sticker', 'dir': '/elsewhere/resources/stickers/probe/' }
moved = c.censor_key()
print( json.dumps( { 'paths': paths,
                     'detection': c.detection_key(), 'censor': plain, 'encode': c.encode_key(),
                     'sticker_censor': sticker, 'moved_sticker_censor': moved,
                     'identity_model_path': c.detection_identity()['backend_tunables'].get( 'model_path' ) } ) )
'''


def _probe( root=None ):
    """
    Run _PROBE with the data root set to `root` (None: unset).

    From a scratch folder, with the repo on PYTHONPATH. Computing a key
    writes its run-key record: unset, that goes to the OS data folder,
    which the suite's environment points under its temp root.
    """
    env = dict( os.environ, BETASUITE_LOCAL_CONFIG='none', PYTHONPATH=REPO_ROOT )
    env.pop( ENV, None )
    if root is not None:
        env[ENV] = root
    code = 'NAMES = %r\n%s'%( sorted( TODAY ), _PROBE )
    with tempfile.TemporaryDirectory( prefix='betasuite-probe-' ) as scratch:
        work = os.path.join( scratch, 'work' )
        os.mkdir( work )
        result = subprocess.run( [ sys.executable, '-c', code ], cwd=work, env=env,
                                 capture_output=True, text=True, timeout=120 )
    return result


class _RootCase( unittest.TestCase ):
    """A temp folder to use as the data root, and a clean variable."""

    def setUp( self ):
        self._tmp = tempfile.TemporaryDirectory( prefix='betasuite-data-root-' )
        self.root = self._tmp.name
        patcher = mock.patch.dict( os.environ )
        patcher.start()
        self.addCleanup( patcher.stop )
        self.addCleanup( self._tmp.cleanup )
        os.environ.pop( ENV, None )
        # Work from inside the temp folder, so a path that misses the root
        # ('../output/...' left relative) lands in it, not beside the repo.
        work = os.path.join( self.root, 'work' )
        os.mkdir( work )
        cwd = os.getcwd()
        os.chdir( work )
        self.addCleanup( os.chdir, cwd )

    def set_root( self, value=None ):
        os.environ[ENV] = self.root if value is None else value


class TestDataRoot( _RootCase ):

    def test_unset_and_empty_mean_no_root( self ):
        self.assertIsNone( bu_paths.data_root() )
        self.set_root( '' )
        self.assertIsNone( bu_paths.data_root() )

    def test_an_absolute_existing_folder_is_used_normalised( self ):
        os.mkdir( os.path.join( self.root, 'sub' ) )
        self.set_root( self.root + os.sep + 'sub' + os.sep + '..' + os.sep )
        self.assertEqual( bu_paths.data_root(), os.path.normpath( self.root ) )

    def test_a_relative_root_is_refused( self ):
        for value in ( 'data', './data', '../data', '~/data', '~' ):
            self.set_root( value )
            with self.assertRaises( bu_paths.DataRootError ) as raised:
                bu_paths.data_root()
            self.assertIn( ENV, str( raised.exception ) )

    def test_a_relative_root_is_refused_even_when_it_exists( self ):
        cwd = os.getcwd()
        os.chdir( self.root )
        self.addCleanup( os.chdir, cwd )
        os.mkdir( 'rel' )
        for value in ( 'rel', '.', '..' ):
            self.set_root( value )
            with self.assertRaises( bu_paths.DataRootError ):
                bu_paths.data_root()

    def test_a_missing_folder_is_refused( self ):
        self.set_root( os.path.join( self.root, 'not-there' ) )
        with self.assertRaises( bu_paths.DataRootError ):
            bu_paths.data_root()

    def test_a_file_is_not_a_folder( self ):
        path = os.path.join( self.root, 'file' )
        open( path, 'w' ).close()
        self.set_root( path )
        with self.assertRaises( bu_paths.DataRootError ):
            bu_paths.data_root()

    def test_the_error_is_a_value_error( self ):
        self.assertTrue( issubclass( bu_paths.DataRootError, ValueError ) )


class TestResolve( _RootCase ):

    MOVED = ( '../resources', '../output', '../resources/', '../output/',
              '../resources/model/v3.4-320n.onnx', '../output/logs/betasuite.log',
              '../resources/stickers/vulva/' )
    LEFT_ALONE = ( '/abs/output/x', 'output/x', './output/x', '../elsewhere/x',
                   '../outputs/x', '../output_old/x', '../resourcesx', '..', '../',
                   '../../output/x', 'x/../output/y', 'ab/output/x', '.x/resources', '' )

    def test_unset_leaves_other_paths_alone( self ):
        for path in self.LEFT_ALONE:
            self.assertIs( bu_paths.resolve( path ), path )

    def test_unset_moves_data_paths_into_the_os_folders( self ):
        for path in self.MOVED:
            moved = bu_paths.resolve( path )
            self.assertTrue( moved.startswith( tests.TEST_OS_ROOT + os.sep ), ( path, moved ) )
            self.assertEqual( moved.endswith( '/' ), path.endswith( '/' ) )

    def test_set_moves_data_paths_under_the_root_keeping_trailing_separators( self ):
        self.set_root()
        for path in self.MOVED:
            moved = bu_paths.resolve( path )
            self.assertEqual( moved, os.path.join( self.root, path[3:] ) )
            self.assertEqual( moved.endswith( '/' ), path.endswith( '/' ) )
        self.assertEqual( bu_paths.resolve( '../output/cache' ),
                          os.path.join( self.root, 'output', 'cache' ) )

    def test_set_leaves_other_paths_alone( self ):
        self.set_root()
        for path in self.LEFT_ALONE:
            self.assertIs( bu_paths.resolve( path ), path )

    def test_non_strings_pass_through( self ):
        self.set_root()
        for value in ( None, 3, ( '../output', ) ):
            self.assertIs( bu_paths.resolve( value ), value )

    def test_a_bad_root_is_refused_when_a_data_path_is_resolved( self ):
        self.set_root( 'relative' )
        with self.assertRaises( bu_paths.DataRootError ):
            bu_paths.resolve( '../output/logs' )
        # Paths it would not move never consult the root.
        self.assertEqual( bu_paths.resolve( '/abs/x' ), '/abs/x' )

    def test_backslash_form_follows_the_os( self ):
        windows = bu_paths._prefix_pattern( '\\/' )
        for path in ( '..\\output\\logs', '..\\resources', '../output\\x', '..\\output/x' ):
            self.assertTrue( windows.match( path ), path )
        self.assertFalse( windows.match( '..\\outputs\\x' ) )
        posix = bu_paths._prefix_pattern( '/' )
        self.assertFalse( posix.match( '..\\output\\logs' ) )
        if os.sep == '/' and not os.altsep:
            self.set_root()
            self.assertEqual( bu_paths.resolve( '..\\output\\logs' ), '..\\output\\logs' )


class TestLegacyPath( _RootCase ):

    def test_set_is_under_the_root( self ):
        self.set_root()
        self.assertEqual( bu_paths.legacy_path( 'output', 'logs' ),
                          os.path.join( self.root, 'output', 'logs' ) )
        self.assertEqual( bu_paths.legacy_path(), self.root )

    def test_legacy_machine_config_follows_the_root( self ):
        self.set_root()
        self.assertEqual( bu_local.legacy_machine_path(),
                          os.path.join( self.root, 'resources', 'betaconfig_machine.py' ) )


class TestLayoutAndKeys( unittest.TestCase ):
    """betaconst and the log defaults, in a fresh interpreter each way."""

    @classmethod
    def setUpClass( cls ):
        cls._tmp = tempfile.TemporaryDirectory( prefix='betasuite-data-root-' )
        cls.root = cls._tmp.name
        unset = _probe()
        other = tempfile.TemporaryDirectory( prefix='betasuite-data-root-' )
        try:
            cls.results = { 'unset': unset, 'set': _probe( cls.root ), 'empty': _probe( '' ),
                            'other': _probe( other.name ) }
        finally:
            other.cleanup()
        for name, result in cls.results.items():
            assert result.returncode == 0, ( name, result.stderr )
        cls.out = { name: json.loads( result.stdout.strip().splitlines()[-1] )
                    for name, result in cls.results.items() }

    @classmethod
    def tearDownClass( cls ):
        cls._tmp.cleanup()

    def test_unset_and_empty_are_the_os_folders( self ):
        expected = { name: bu_paths.resolve( today ) for name, today in TODAY.items() }
        self.assertEqual( self.out['unset']['paths'], expected )
        self.assertEqual( self.out['empty']['paths'], expected )
        for name, path in expected.items():
            self.assertTrue( path.startswith( tests.TEST_OS_ROOT + os.sep ), ( name, path ) )

    def test_set_is_byte_identical_to_1x_under_the_root( self ):
        paths = self.out['set']['paths']
        self.assertEqual( set( paths ), set( TODAY ) )
        for name, today in TODAY.items():
            self.assertEqual( paths[name], os.path.join( self.root, today[3:] ), name )
            self.assertTrue( paths[name].startswith( self.root + os.sep ), name )

    def test_cache_keys_do_not_depend_on_the_root( self ):
        for key in ( 'detection', 'censor', 'encode', 'sticker_censor', 'identity_model_path' ):
            values = { name: self.out[name][key] for name in self.out }
            self.assertEqual( len( set( values.values() ) ), 1, ( key, values ) )
        self.assertEqual( self.out['set']['identity_model_path'], '../resources/model/v3.4-320n.onnx' )

    def test_rewriting_a_sticker_folder_would_have_moved_the_censor_key( self ):
        # Why config-valued paths are resolved only at the filesystem.
        self.assertNotEqual( self.out['unset']['sticker_censor'], self.out['unset']['moved_sticker_censor'] )

    def test_a_relative_root_stops_the_import( self ):
        result = _probe( 'relative/root' )
        self.assertNotEqual( result.returncode, 0 )
        self.assertIn( 'DataRootError', result.stderr )
        self.assertIn( ENV, result.stderr )


class TestFilesystemBoundary( _RootCase ):
    """Config-valued paths are found under the root where they are opened."""

    def test_a_missing_model_is_reported_at_its_resolved_place( self ):
        self.set_root()
        with self.assertRaises( FileNotFoundError ) as raised:
            bu_detector.build_onnx_session( '../resources/model/absent.onnx', backend_name='nudenet_v3' )
        self.assertIn( os.path.join( self.root, 'resources', 'model', 'absent.onnx' ), str( raised.exception ) )

    def test_the_legacy_retinanet_session_opens_under_the_root( self ):
        import betautils_model as bu_model
        self.set_root()
        with mock.patch.object( bu_model.onnxruntime, 'InferenceSession' ) as session, \
             mock.patch.object( betaconfig, 'gpu_enabled', 0 ):
            bu_model.get_session()
        self.assertEqual( session.call_args[0][0], os.path.join(
            self.root, 'resources', 'model', 'detector_v2_default_checkpoint.onnx' ) )

    def test_stickers_load_from_under_the_root( self ):
        import cv2
        import betautils_censor as bu_censor
        # Unique per process: the loader caches by the configured string.
        logical = '../resources/stickers/root-test-%d/'%( os.getpid(), )
        folder = os.path.join( self.root, logical[3:] )
        os.makedirs( folder )
        cv2.imwrite( os.path.join( folder, 'a.png' ), np.zeros( ( 4, 4, 4 ), dtype=np.uint8 ) )
        cv2.imwrite( os.path.join( folder, 'b.png' ), np.zeros( ( 4, 4, 3 ), dtype=np.uint8 ) )
        self.set_root()
        self.assertEqual( len( bu_censor._load_stickers( { 'type': 'sticker', 'dir': logical } ) ), 2 )
        image = logical + 'a.png'
        self.assertEqual( len( bu_censor._load_stickers( { 'type': 'sticker', 'images': [ image ] } ) ), 1 )

    def test_log_and_stats_paths_resolve_under_the_root( self ):
        with mock.patch.object( betaconfig, 'log_path', '../output/logs/x.log' ), \
             mock.patch.object( betaconfig, 'stats_path', '../output/stats/x.jsonl' ):
            self.assertEqual( bu_log.configured_log_path(), bu_paths.resolve( '../output/logs/x.log' ) )
            self.assertEqual( bu_log.configured_stats_path(), bu_paths.resolve( '../output/stats/x.jsonl' ) )
            self.assertTrue( bu_log.configured_log_path().startswith( tests.TEST_OS_ROOT + os.sep ) )
            self.set_root()
            self.assertEqual( bu_log.configured_log_path(), os.path.join( self.root, 'output', 'logs', 'x.log' ) )
            self.assertEqual( bu_log.configured_stats_path(),
                              os.path.join( self.root, 'output', 'stats', 'x.jsonl' ) )

    def test_unset_log_and_stats_fall_back_to_the_defaults( self ):
        with mock.patch.object( betaconfig, 'log_path', None ), \
             mock.patch.object( betaconfig, 'stats_path', None ):
            self.assertEqual( bu_log.configured_log_path(), bu_log.DEFAULT_LOG_PATH )
            self.assertEqual( bu_log.configured_stats_path(), bu_log.DEFAULT_STATS_PATH )

    def test_stats_are_written_under_the_root( self ):
        self.set_root()
        with mock.patch.object( betaconfig, 'stats_enabled', True ), \
             mock.patch.object( betaconfig, 'stats_path', '../output/stats/w.jsonl' ):
            bu_log.write_stats( { 'probe': 1 } )
            bu_log.write_failure( { 'probe': 2 } )
        self.assertTrue( os.path.isfile( os.path.join( self.root, 'output', 'stats', 'w.jsonl' ) ) )
        self.assertTrue( os.path.isfile( os.path.join( self.root, 'output', 'stats', 'w.failures.jsonl' ) ) )


if __name__ == '__main__':
    unittest.main()
