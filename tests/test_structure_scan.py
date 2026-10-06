"""
test_structure_scan.py - betabench structure-scan.

WHAT IT PROMISES
    - a whole-file shot-cut scan for every source file not yet scanned,
      written to EXACTLY the cache betatv's full run reads, so the real
      run finds its scan done and the work is never paid twice
    - re-running does nothing for files already scanned
    - --geometry-slices N runs detection, never render, on N slices
      spread through each file, writing the caches a --preview run at
      that offset would
"""

import os
import shutil
import types
import unittest
from unittest import mock

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_hash as bu_hash
import betautils_log as bu_log
import betautils_video as bu_video

from tests.test_scan_overlap import _build_clip_with_cuts
from tests.test_betatv_end_to_end import BetaTvHarness, _ffmpeg_available
from tests.test_structure_bench import betabench, _quiet_logger


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestStructureScan( BetaTvHarness ):

    def setUp( self ):
        super().setUp()
        self.staging = os.path.join( self.tmpdir, 'source' )
        self._saved_staging = betaconst.video_path_source_backup
        betaconst.video_path_source_backup = self.staging + os.sep
        _build_clip_with_cuts( os.path.join( self.uncensored, 'scene', 'a.mp4' ) )
        _build_clip_with_cuts( os.path.join( self.staging, 'compilation', 'b.mp4' ), shots=2 )
        self.args = types.SimpleNamespace( workers=2, geometry_slices=0, slice_seconds=2.0 )
        self.logger = _quiet_logger()
        self.fps = betaconfig.video_censor_fps
        self.threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )

    def tearDown( self ):
        betaconst.video_path_source_backup = self._saved_staging
        super().tearDown()

    def hash_of( self, path ):
        return bu_hash.md5_for_file( path, 16 )

    def test_it_writes_the_cache_betatv_reads_for_every_tree( self ):
        betabench.bench_structure_scan( self.args, self.logger )
        for path in ( os.path.join( self.uncensored, 'scene', 'a.mp4' ),
                      os.path.join( self.staging, 'compilation', 'b.mp4' ) ):
            cache = bu_cache.shot_cut_path_for( self.hash_of( path ), self.fps, self.threshold )
            self.assertTrue( os.path.exists( cache ), path )

    def test_the_cached_cuts_match_a_direct_scan( self ):
        betabench.bench_structure_scan( self.args, self.logger )
        path = os.path.join( self.uncensored, 'scene', 'a.mp4' )
        file_hash = self.hash_of( path )
        cached = bu_hash.read_json( bu_cache.shot_cut_path_for( file_hash, self.fps, self.threshold ) )
        source = bu_video.open_video_source( path, file_hash, self.logger )
        try:
            direct = bu_video.detect_shot_cuts( source.open_capture(), source.vid_fps, self.fps,
                                                self.threshold )
        finally:
            source.cap.release()
        self.assertEqual( cached, direct )
        self.assertGreaterEqual( len( direct ), 2, "precondition: the clip has cuts" )

    def test_a_second_run_scans_nothing( self ):
        betabench.bench_structure_scan( self.args, self.logger )
        result = betabench.bench_structure_scan( self.args, self.logger )
        self.assertEqual( result['structure_scan'], [] )

    def test_the_next_real_run_reuses_the_scan( self ):
        betabench.bench_structure_scan( self.args, self.logger )
        with mock.patch.object( bu_video, 'detect_shot_cuts',
                                side_effect=AssertionError( 'scanned again' ) ):
            self.assertEqual( self.run_one( os.path.join( 'scene', 'a.mp4' ) ), 'processed' )

    def run_one( self, relative ):
        import betatv
        root, fname = os.path.split( os.path.join( self.uncensored, relative ) )
        censored = os.path.join( self.censored, os.path.dirname( relative ) )
        os.makedirs( censored, exist_ok=True )
        # The app's own logger: render calls logger.trace, which a plain
        # logging.Logger does not have.
        return betatv.process_one_video(
            root, fname, censored, 0, 1, self.detector.get_session(),
            False, 20, 'ultrafast', bu_log.get_logger() )

    def test_geometry_slices_detect_without_rendering( self ):
        self.args.geometry_slices = 2
        betabench.bench_structure_scan( self.args, self.logger )
        path = os.path.join( self.uncensored, 'scene', 'a.mp4' )
        file_hash = self.hash_of( path )
        names = os.listdir( betaconst.vid_hash_dir )
        offsets = betabench.geometry_slice_offsets( 6.0, 2.0, 2 )
        self.assertEqual( len( offsets ), 2 )
        for offset in offsets:
            suffix = bu_cache.preview_cache_suffix( True, offset )
            self.assertTrue( any( name.startswith( file_hash ) and name.endswith( suffix + '.gz' )
                                  for name in names ), ( suffix, names ) )
        self.assertEqual( self.outputs(), [], "geometry slices must never render" )


if __name__ == '__main__':
    unittest.main()


class TestBetabenchExitStatus( unittest.TestCase ):
    """A failed subcommand must fail the process, or a harness calls it done."""

    def run_main( self, behaviour ):
        import sys
        import tempfile
        out_dir = tempfile.mkdtemp( prefix='betabench-exit-' )
        try:
            with mock.patch.dict( betabench.SUBCOMMANDS, { 'structure': behaviour } ), \
                 mock.patch.object( sys, 'argv', [ 'betabench.py', 'structure', '--yes',
                                                   '--out-dir', out_dir,
                                                   '--console-level', 'error' ] ):
                return betabench.main()
        finally:
            shutil.rmtree( out_dir, ignore_errors=True )

    def test_a_failing_subcommand_exits_non_zero( self ):
        def boom( args, logger ):
            raise RuntimeError( 'no model' )
        self.assertEqual( self.run_main( boom ), 1 )

    def test_a_clean_run_exits_zero( self ):
        self.assertEqual( self.run_main( lambda args, logger: {} ), 0 )
