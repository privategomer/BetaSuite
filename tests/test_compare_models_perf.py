"""
test_compare_models_perf.py - regression tests for the non-inference
parts of tools/analysis/compare_models_perf.py (fmt_ms_stats, time_calls,
find_real_frame's uncensored_vids/ -> source_backup fallback search
order). Does NOT exercise real ONNX inference (time_one_backend_size /
main()) - that needs a real vendored model file and was verified
manually against the real v3.4-320n.onnx during development (both the
single-image and batched paths ran end-to-end, and a missing-model-file
failure was confirmed to be isolated per (backend, size) rather than
fatal to the whole sweep - see main()'s try/except per combo).
"""

import importlib.util
import os
import tempfile
import unittest

import cv2
import numpy as np

import betaconst

_HERE = os.path.dirname( os.path.abspath( __file__ ) )
_REPO_ROOT = os.path.dirname( _HERE )


def _load_module():
    full_path = os.path.join( _REPO_ROOT, 'tools/analysis/compare_models_perf.py' )
    spec = importlib.util.spec_from_file_location( 'compare_models_perf_under_test', full_path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestFmtMsStats( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.mod = _load_module()

    def test_empty_list( self ):
        self.assertEqual( self.mod.fmt_ms_stats( [] ), "n=0" )

    def test_single_value( self ):
        result = self.mod.fmt_ms_stats( [ 10.0 ] )
        self.assertIn( "n=1", result )
        self.assertIn( "min=10.0ms", result )
        self.assertIn( "max=10.0ms", result )

    def test_multiple_values_min_max_correct( self ):
        result = self.mod.fmt_ms_stats( [ 5.0, 20.0, 10.0 ] )
        self.assertIn( "min=5.0ms", result )
        self.assertIn( "max=20.0ms", result )
        self.assertIn( "median=10.0ms", result )


class TestTimeCalls( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.mod = _load_module()

    def test_warmup_calls_not_included_in_returned_durations( self ):
        call_count = { 'n': 0 }
        def fn():
            call_count['n'] += 1
        durations = self.mod.time_calls( fn, n_warmup=3, n_timed=5 )
        self.assertEqual( call_count['n'], 8 )  # 3 warmup + 5 timed
        self.assertEqual( len( durations ), 5 )  # only timed calls reported

    def test_zero_warmup( self ):
        call_count = { 'n': 0 }
        def fn():
            call_count['n'] += 1
        durations = self.mod.time_calls( fn, n_warmup=0, n_timed=4 )
        self.assertEqual( call_count['n'], 4 )
        self.assertEqual( len( durations ), 4 )

    def test_durations_are_non_negative( self ):
        durations = self.mod.time_calls( lambda: None, n_warmup=1, n_timed=3 )
        for d in durations:
            self.assertGreaterEqual( d, 0.0 )


class TestFindRealFrameSearchOrder( unittest.TestCase ):
    """
    find_real_frame checks betaconst.video_path_uncensored first, then
    falls back to betaconst.video_path_source_backup - same fallback
    order as build_hash_to_video_path in the analyze_*.py tools (see
    betaconst.py's own comment on video_path_source_backup).
    """

    @classmethod
    def setUpClass( cls ):
        cls.mod = _load_module()

    def setUp( self ):
        self._tmpdir = tempfile.mkdtemp()
        self._uncensored_dir = os.path.join( self._tmpdir, 'uncensored_vids' )
        self._source_dir = os.path.join( self._tmpdir, 'source' )
        os.makedirs( self._uncensored_dir )
        os.makedirs( self._source_dir )

        self._orig_uncensored = betaconst.video_path_uncensored
        self._orig_source = betaconst.video_path_source_backup
        betaconst.video_path_uncensored = self._uncensored_dir + '/'
        betaconst.video_path_source_backup = self._source_dir + '/'

    def tearDown( self ):
        import shutil
        shutil.rmtree( self._tmpdir, ignore_errors=True )
        betaconst.video_path_uncensored = self._orig_uncensored
        betaconst.video_path_source_backup = self._orig_source

    def _write_test_video( self, path, width=64, height=48 ):
        frame = ( np.random.rand( height, width, 3 ) * 255 ).astype( np.uint8 )
        writer = cv2.VideoWriter( path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height) )
        writer.write( frame )
        writer.release()

    def test_finds_file_in_uncensored_vids_first( self ):
        self._write_test_video( os.path.join( self._uncensored_dir, 'test.mp4' ) )
        frame = self.mod.find_real_frame( 'test.mp4' )
        self.assertEqual( frame.shape[:2], (48, 64) )

    def test_falls_back_to_source_backup( self ):
        self._write_test_video( os.path.join( self._source_dir, 'test.mp4' ), width=100, height=80 )
        frame = self.mod.find_real_frame( 'test.mp4' )
        self.assertEqual( frame.shape[:2], (80, 100) )

    def test_missing_everywhere_exits( self ):
        with self.assertRaises( SystemExit ):
            self.mod.find_real_frame( 'does-not-exist.mp4' )


if __name__ == '__main__':
    unittest.main()
