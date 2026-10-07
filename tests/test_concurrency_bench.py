"""
test_concurrency_bench.py - betabench concurrency measures without
touching anything real.

It times the shot-cut scan, detection, and both at once, then scan
throughput at several worker counts. A timing that hits a warm cache
measures a file read, and a bench that writes the real caches changes
what the next run does, so both are pinned here.
"""

import os
import types
import unittest

import betaconst
import betautils_cache_paths as bu_cache

from tests.test_betatv_end_to_end import BetaTvHarness, _ffmpeg_available
from tests.test_scan_overlap import _build_clip_with_cuts
from tests.test_structure_bench import betabench, _quiet_logger


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestConcurrencyBench( BetaTvHarness ):

    def setUp( self ):
        super().setUp()
        self._saved_staging = betaconst.video_path_source_backup
        betaconst.video_path_source_backup = os.path.join( self.tmpdir, 'none' ) + os.sep
        _build_clip_with_cuts( os.path.join( self.uncensored, 'a.mp4' ) )
        _build_clip_with_cuts( os.path.join( self.uncensored, 'b.mp4' ), shots=2 )
        self.args = types.SimpleNamespace( video=None, seconds=4.0, trials=1, max_videos=2 )

    def tearDown( self ):
        betaconst.video_path_source_backup = self._saved_staging
        super().tearDown()

    def real_cache_files( self ):
        found = []
        for directory in ( betaconst.shot_cut_dir, betaconst.vid_hash_dir ):
            if os.path.isdir( directory ):
                found += os.listdir( directory )
        return found

    def test_it_reports_both_measurements( self ):
        result = betabench.bench_concurrency( self.args, _quiet_logger() )['concurrency']
        overlap = result['overlap']
        for key in ( 'scan_seconds', 'detect_seconds', 'serial_seconds',
                     'overlapped_seconds', 'saved_seconds' ):
            self.assertIn( key, overlap )
        self.assertAlmostEqual( overlap['serial_seconds'],
                                overlap['scan_seconds'] + overlap['detect_seconds'], places=2 )
        workers = result['scan_workers']
        self.assertEqual( workers['files'], 2 )
        self.assertIn( 1, [ row['workers'] for row in workers['rows'] ] )
        self.assertTrue( all( not row['errors'] for row in workers['rows'] ) )
        self.assertIn( workers['recommended_workers'], [ row['workers'] for row in workers['rows'] ] )

    def test_nothing_real_is_read_or_written( self ):
        saved = ( betaconst.shot_cut_dir, betaconst.vid_hash_dir, bu_cache.VID_HASH_DIR )
        before = self.real_cache_files()
        betabench.bench_concurrency( self.args, _quiet_logger() )
        self.assertEqual( ( betaconst.shot_cut_dir, betaconst.vid_hash_dir, bu_cache.VID_HASH_DIR ),
                          saved, "cache paths must be restored" )
        self.assertEqual( self.real_cache_files(), before )

    def test_each_timing_starts_from_an_empty_cache( self ):
        # The scan must actually run in every timed pass: count real scans.
        calls = []
        real = betabench.bu_video.detect_shot_cuts

        def counting( *args, **kwargs ):
            calls.append( 1 )
            return real( *args, **kwargs )

        betabench.bu_video.detect_shot_cuts = counting
        try:
            self._measure_overlap_on_a()
        finally:
            betabench.bu_video.detect_shot_cuts = real
        # warm-up + 1 trial, each with a scan-alone pass and a both pass
        self.assertEqual( len( calls ), 4 )

    def test_detection_is_never_served_from_a_warm_cache( self ):
        # The detect-alone pass writes a detection cache; the both pass
        # must not read it, or "at the same time" times a file read.
        frames_before = self.detector.frames_seen
        self._measure_overlap_on_a()
        per_pass = 4 * 5   # 4s window at the harness's 5 fps
        # warm-up + 1 trial, each with a detect-alone pass and a both pass
        self.assertEqual( self.detector.frames_seen - frames_before, 4 * per_pass )

    def test_the_both_timing_waits_for_the_scan( self ):
        import time as _time
        real = betabench.bu_video.detect_shot_cuts

        def slow_scan( *args, **kwargs ):
            _time.sleep( 0.4 )
            return real( *args, **kwargs )

        betabench.bu_video.detect_shot_cuts = slow_scan
        try:
            source_path = os.path.join( self.uncensored, 'a.mp4' )
            import betautils_hash as bu_hash
            import betautils_video as bu_video
            file_hash = bu_hash.md5_for_file( source_path, 16 )
            source = bu_video.open_video_source( source_path, file_hash, _quiet_logger() )
            try:
                with betabench._ThrowawayCaches() as caches:
                    _scan, _detect, both = betabench._time_overlap_once(
                        source, 'a.mp4', file_hash, self.detector.get_session(), [ 320 ], 5,
                        0.4, 4.0, caches, _quiet_logger() )
            finally:
                source.cap.release()
        finally:
            betabench.bu_video.detect_shot_cuts = real
        self.assertGreaterEqual( both, 0.4 )


    def _measure_overlap_on_a( self ):
        # A with-block, never addCleanup: cleanups run AFTER tearDown, so
        # exiting there would restore the harness's temp paths on top of
        # the real ones and leak them into every later test (the suite
        # isolation test caught exactly that).
        with betabench._ThrowawayCaches() as caches:
            return betabench._measure_overlap(
                self.args, _quiet_logger(), os.path.join( self.uncensored, 'a.mp4' ), caches )


class TestScanWorkerWindow( unittest.TestCase ):

    @unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clip" )
    def test_a_bounded_worker_scan_writes_the_bounded_cache( self ):
        import shutil
        import tempfile
        import betautils_hash as bu_hash
        import betautils_video as bu_video
        tmp = tempfile.mkdtemp( prefix='betasuite-window-' )
        self.addCleanup( shutil.rmtree, tmp, True )
        clip = os.path.join( tmp, 'a.mp4' )
        _build_clip_with_cuts( clip )
        file_hash = bu_hash.md5_for_file( clip, 16 )
        paths = { 'shot_cut_dir': os.path.join( tmp, 'cuts' ),
                  'transcode_cache_dir': os.path.join( tmp, 'transcode' ),
                  'file_hash_cache_path': os.path.join( tmp, 'hashes.json' ) }
        saved = { name: getattr( betaconst, name ) for name in paths }
        try:
            _path, _cuts, _seconds, error = bu_video.scan_whole_file_in_worker(
                clip, file_hash, 5, 0.4, paths, max_seconds=2.0 )
            self.assertIsNone( error )
            self.assertTrue( os.path.exists( bu_cache.shot_cut_path_for(
                file_hash, 5, 0.4, '', 2.0 ) ) )
        finally:
            for name, value in saved.items():
                setattr( betaconst, name, value )

if __name__ == '__main__':
    unittest.main()
