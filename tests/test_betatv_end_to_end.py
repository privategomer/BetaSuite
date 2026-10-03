"""
test_betatv_end_to_end.py - betatv.py against real video files, with a
stub detector.

WHAT THIS COVERS THAT UNIT TESTS CANNOT
    The wiring. Every piece of betatv has unit coverage elsewhere; what
    breaks in practice is the seams between them - a cache path built
    from one set of settings and looked up with another, a preview
    window applied to detection but not to the render, an output that
    exists but was produced under different settings.

    So these build real (tiny) videos with ffmpeg, run the real
    orchestration, and assert on what lands on disk. The only thing
    faked is the model: a stub adapter returns deterministic boxes, so
    there is no dependency on a 12MB or 146MB .onnx file and no
    dependence on what a real model happens to detect in a test pattern.

ALSO COVERED HERE
    Interrupt semantics. Ctrl-C must stop the run, not be swallowed by
    the per-file error handling that makes one bad file non-fatal. That
    distinction was a real defect: the handler caught BaseException, so
    an interrupt was reported as a failed file and the run moved on to
    the next one.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_log as bu_log
import betautils_signals as bu_signals
import betautils_video as bu_video
import betatv


def _ffmpeg_available():
    return shutil.which( 'ffmpeg' ) is not None


# Distinguishes "betaconfig had no such attribute" from "it was set to
# None", which the harness's save/restore has to tell apart.
_MISSING = object()


class _StubDetectorModule:
    """
    A detector adapter that reports a fixed box in the middle of every
    frame, so the pipeline has something real to track and render.
    """

    def __init__( self, label='exposed_breast', score=0.95 ):
        self.label = label
        self.score = score
        self.frames_seen = 0
        self.batch_sizes_seen = []

    def get_session( self ):
        return object()

    def raw_boxes_for_img( self, img, size, session, t ):
        return self.raw_boxes_for_imgs( [ img ], size, session, [ t ] )

    def raw_boxes_for_imgs( self, imgs, size, session, ts ):
        self.frames_seen += len( imgs )
        self.batch_sizes_seen.append( len( imgs ) )
        boxes = []
        for img, timestamp in zip( imgs, ts ):
            height, width = np.asarray( img ).shape[:2]
            boxes.append( {
                'x': float( width//4 ), 'y': float( height//4 ),
                'w': float( width//3 ), 'h': float( height//3 ),
                'class_id': self.label, 'score': self.score,
                't': timestamp, 'size': size,
            } )
        return boxes


def _build_clip( path, seconds=3, width=128, height=96, rate=10, with_audio=True ):
    """A small synthetic clip, with an audio track by default."""
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    command = [ 'ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'lavfi',
                '-i', 'testsrc2=size=%dx%d:rate=%d:duration=%d'%(width, height, rate, seconds) ]
    if with_audio:
        command += [ '-f', 'lavfi', '-i', 'sine=frequency=440:duration=%d'%(seconds),
                     '-c:a', 'aac', '-shortest' ]
    command += [ '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', path ]
    subprocess.run( command, check=True )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class BetaTvHarness( unittest.TestCase ):
    """
    Redirects every BetaSuite path into a temp tree and installs a stub
    detector, so a test run touches nothing real.
    """

    def setUp( self ):
        self.tmpdir = tempfile.mkdtemp( prefix='betasuite-e2e-' )
        self.uncensored = os.path.join( self.tmpdir, 'uncensored_vids' )
        self.censored = os.path.join( self.tmpdir, 'censored_vids' )
        os.makedirs( self.uncensored, exist_ok=True )
        os.makedirs( self.censored, exist_ok=True )

        self._saved_const = { name: getattr( betaconst, name ) for name in (
            'video_path_uncensored', 'video_path_censored', 'vid_hash_dir',
            'pic_hash_dir', 'shot_cut_dir', 'transcode_cache_dir',
            'file_hash_cache_path', 'run_key_dir' ) }
        betaconst.video_path_uncensored = self.uncensored + os.sep
        betaconst.video_path_censored = self.censored + os.sep
        betaconst.vid_hash_dir = os.path.join( self.tmpdir, 'cache', 'vid_hashes' )
        betaconst.pic_hash_dir = os.path.join( self.tmpdir, 'cache', 'pic_hashes' )
        betaconst.shot_cut_dir = os.path.join( self.tmpdir, 'cache', 'shot_cuts' )
        betaconst.transcode_cache_dir = os.path.join( self.tmpdir, 'cache', 'transcode' )
        betaconst.file_hash_cache_path = os.path.join( self.tmpdir, 'cache', 'file_hashes.json' )
        betaconst.run_key_dir = os.path.join( self.tmpdir, 'cache', 'run_keys' )
        bu_cache.VID_HASH_DIR = betaconst.vid_hash_dir
        bu_cache.PIC_HASH_DIR = betaconst.pic_hash_dir

        self._saved_config = {}
        for name, value in (
                ( 'render_chunk_seconds', 0 ),
                ( 'render_workers', 1 ),
                ( 'logging_enabled', False ),
                ( 'stats_enabled', True ),
                ( 'stats_path', os.path.join( self.tmpdir, 'stats.jsonl' ) ),
                ( 'preview_mode_enabled', False ),
                # Saved and pinned even though most tests never touch
                # them: a test that leaves a random slice or a pinned
                # start behind changes which output filename the NEXT
                # test resolves to, which surfaces as an unrelated test
                # mysteriously skipping its own file.
                ( 'preview_start_seconds', None ),
                ( 'preview_random_slice', False ),
                ( 'default_censor_shape',
                  getattr( betaconfig, 'default_censor_shape', 'box' ) ),
                ( 'shot_cut_detection_enabled', True ),
                ( 'encode_preset', 'ultrafast' ),
                ( 'items_to_censor', [ 'exposed_breast' ] ),
                ( 'video_censor_fps', 5 ) ):
            # _MISSING, not None, for "this attribute did not exist".
            # None is a legitimate configured value (preview_start_seconds
            # means "no pinned start"), and the restore loop used to skip
            # every None, so any test that set one leaked it into the
            # rest of the run.
            self._saved_config[name] = getattr( betaconfig, name, _MISSING )
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

        self.detector = _StubDetectorModule()
        self._detector_patch = mock.patch.object(
            betatv.bu_detector, 'get_detector', return_value=self.detector )
        self._detector_patch.start()
        self._sizes_patch = mock.patch.object(
            betatv.bu_detector, 'get_picture_sizes', return_value=[ 320 ] )
        self._sizes_patch.start()

        bu_signals.clear()
        self.logger = bu_log.get_logger()

    def tearDown( self ):
        self._detector_patch.stop()
        self._sizes_patch.stop()
        for name, value in self._saved_const.items():
            setattr( betaconst, name, value )
        bu_cache.VID_HASH_DIR = betaconst.vid_hash_dir
        bu_cache.PIC_HASH_DIR = betaconst.pic_hash_dir
        for name, value in self._saved_config.items():
            if value is _MISSING:
                if hasattr( betaconfig, name ):
                    delattr( betaconfig, name )
                continue
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()
        bu_signals.clear()
        shutil.rmtree( self.tmpdir, ignore_errors=True )

    def run_one( self, fname='clip.mp4' ):
        """Run process_one_video against one clip and return the outcome."""
        return betatv.process_one_video(
            self.uncensored, fname, self.censored, 0, 1,
            self.detector.get_session(),
            getattr( betaconfig, 'preview_mode_enabled', False ),
            getattr( betaconfig, 'preview_max_seconds', 20 ),
            'ultrafast', self.logger )

    def outputs( self ):
        return sorted( name for name in os.listdir( self.censored )
                       if name.endswith( '.mp4' ) )


class TestFullRun( BetaTvHarness ):

    def test_a_clip_is_detected_tracked_rendered_and_muxed( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        outcome = self.run_one()
        self.assertEqual( outcome, 'processed' )

        outputs = self.outputs()
        self.assertEqual( len( outputs ), 1 )
        final = os.path.join( self.censored, outputs[0] )
        self.assertTrue( bu_video.probe_video_ok( final ) )
        self.assertTrue( bu_video.video_file_has_audio( final ),
            "the source audio must survive the single-encode pipeline" )

    def test_the_output_name_carries_all_three_cache_keys( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.run_one()
        name = self.outputs()[0]
        self.assertRegex( name, r'-d[0-9a-f]{6}-c[0-9a-f]{6}-e[0-9a-f]{4}\.mp4$' )

    def test_no_intermediate_or_chunk_files_are_left_behind( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.run_one()
        leftovers = [ name for name in os.listdir( self.censored )
                      if not name.endswith( '.mp4' ) ]
        self.assertEqual( leftovers, [] )

    def test_detections_are_cached_and_reused_on_a_second_run( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.run_one()
        frames_first = self.detector.frames_seen
        self.assertGreater( frames_first, 0 )

        # Remove the output so the run is not skipped outright, and
        # confirm the detection cache - not the model - supplied the
        # boxes the second time.
        for name in self.outputs():
            os.remove( os.path.join( self.censored, name ) )
        self.run_one()
        self.assertEqual( self.detector.frames_seen, frames_first,
            "the second run re-ran the model instead of reading the detection cache" )

    def test_an_existing_output_is_skipped( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.assertEqual( self.run_one(), 'processed' )
        self.assertEqual( self.run_one(), 'skipped' )

    def test_a_non_video_file_is_skipped_not_failed( self ):
        with open( os.path.join( self.uncensored, 'notes.txt' ), 'w' ) as handle:
            handle.write( 'not a video' )
        self.assertEqual( self.run_one( 'notes.txt' ), 'skipped' )

    def test_a_clip_without_audio_still_renders( self ):
        _build_clip( os.path.join( self.uncensored, 'silent.mp4' ), with_audio=False )
        self.assertEqual( self.run_one( 'silent.mp4' ), 'processed' )
        self.assertTrue( bu_video.probe_video_ok(
            os.path.join( self.censored, self.outputs()[0] ) ) )

    def test_a_stats_row_is_written_with_the_run_identity( self ):
        import json
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.run_one()
        with open( betaconfig.stats_path, 'r', encoding='UTF-8' ) as handle:
            rows = [ json.loads( line ) for line in handle if line.strip() ]
        self.assertEqual( len( rows ), 1 )
        row = rows[0]
        for key in ( 'detector_backend', 'picture_sizes', 'detection_key',
                     'censor_key', 'encode_key', 'execution_providers',
                     'video_frames', 'track_rendered_boxes', 'render_chunks' ):
            self.assertIn( key, row )

    def test_changing_a_tracking_setting_changes_the_output_name( self ):
        # The earlier filename deliberately excluded tracking settings,
        # so re-running with a changed track_max_gap silently overwrote
        # the previous output and left nothing to compare.
        import copy
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.run_one()
        first = self.outputs()[0]

        original_backend = betaconfig.detector_backend
        try:
            mutated = copy.deepcopy( original_backend )
            backend = mutated['selected']
            mutated[backend].setdefault( 'item_overrides', {} ).setdefault(
                'exposed_breast', {} )['position_smoothing'] = 0.9
            betaconfig.detector_backend = mutated
            bu_config.invalidate_config_caches()
            self.run_one()
        finally:
            betaconfig.detector_backend = original_backend
            bu_config.invalidate_config_caches()

        outputs = self.outputs()
        self.assertEqual( len( outputs ), 2,
            "a changed tracking setting must not overwrite the previous output" )
        self.assertIn( first, outputs )

    def test_shot_cuts_are_cached( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.run_one()
        self.assertTrue( os.path.isdir( betaconst.shot_cut_dir ) )
        self.assertTrue( os.listdir( betaconst.shot_cut_dir ) )

    def test_run_key_manifests_are_written_so_a_filename_can_be_decoded( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.run_one()
        manifests = sorted( os.listdir( betaconst.run_key_dir ) )
        self.assertTrue( any( name.startswith( 'detection-' ) for name in manifests ) )
        self.assertTrue( any( name.startswith( 'censor-' ) for name in manifests ) )
        self.assertTrue( any( name.startswith( 'encode-' ) for name in manifests ) )


class TestPreviewMode( BetaTvHarness ):

    def test_a_preview_run_processes_only_its_window( self ):
        betaconfig.preview_mode_enabled = True
        betaconfig.preview_max_seconds = 1
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=4 )
        self.assertEqual( self.run_one(), 'processed' )

        # 1 second at 5 samples per second.
        self.assertLessEqual( self.detector.frames_seen, 6 )
        self.assertGreater( self.detector.frames_seen, 0 )

        name = self.outputs()[0]
        self.assertIn( '-preview', name )

    def test_a_preview_output_never_collides_with_a_real_one( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=2 )
        self.run_one()
        real_name = self.outputs()[0]

        betaconfig.preview_mode_enabled = True
        betaconfig.preview_max_seconds = 1
        self.run_one()

        outputs = self.outputs()
        self.assertEqual( len( outputs ), 2 )
        self.assertIn( real_name, outputs )

    def test_a_repeated_preview_run_reuses_its_identical_output( self ):
        # This test used to assert the opposite: that a preview run is
        # NEVER skipped. That was the right call when the output name
        # could not be trusted to describe the work - but the name now
        # carries all three cache keys plus the slice offset, so a repeat
        # run with unchanged settings can only reproduce the same bytes.
        #
        # Re-running it anyway cost a shot-cut scan and a full detection
        # pass per file (17-31s each in real use) to arrive at a file
        # that was already on disk. Preview mode is the tuning loop, so
        # that waste landed on the most repeated workflow in the app.
        #
        # The protection against reusing output for CHANGED settings is
        # the censor key, not this skip. See test_censor_key_coverage.py.
        betaconfig.preview_mode_enabled = True
        betaconfig.preview_max_seconds = 1
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=2 )
        self.assertEqual( self.run_one(), 'processed' )
        self.assertEqual( self.run_one(), 'skipped' )

    def test_a_changed_censor_setting_reprocesses_a_preview( self ):
        # The other half of the contract above, and the reason the skip
        # is safe: changing anything that alters rendered output must
        # produce a different filename, so the skip cannot hide it.
        betaconfig.preview_mode_enabled = True
        betaconfig.preview_max_seconds = 1
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=2 )
        self.assertEqual( self.run_one(), 'processed' )

        betaconfig.default_censor_shape = (
            'ellipse' if getattr( betaconfig, 'default_censor_shape', 'box' ) != 'ellipse'
            else 'box' )
        bu_config.invalidate_config_caches()
        self.assertEqual(
            self.run_one(), 'processed',
            "a changed censor setting reused an existing output; the "
            "early skip is only safe while the censor key covers "
            "everything that changes rendered bytes" )

    def test_a_random_preview_slice_always_reruns( self ):
        # A random slice is a request for a NEW sample. The offset is
        # written into the name at one-decimal precision, so a fresh draw
        # can collide with an existing file and would otherwise be
        # skipped - reporting a slice that was never rendered.
        betaconfig.preview_mode_enabled = True
        betaconfig.preview_max_seconds = 1
        betaconfig.preview_start_seconds = None
        betaconfig.preview_random_slice = True
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=4 )

        # Pin the draw so both runs resolve to the same filename; without
        # the random-slice guard the second would be skipped.
        with mock.patch( 'random.uniform', return_value=1.5 ):
            self.assertEqual( self.run_one(), 'processed' )
            self.assertEqual( self.run_one(), 'processed' )

    def test_a_preview_window_bounds_the_shot_cut_scan( self ):
        # Previously a 20-second preview of a two-hour file still paid
        # for a two-hour histogram scan.
        betaconfig.preview_mode_enabled = True
        betaconfig.preview_max_seconds = 1
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=4 )

        seen = {}
        original = bu_video.detect_shot_cuts

        def spy( capture, vid_fps, sample_fps, threshold=0.6, max_seconds=None ):
            seen['max_seconds'] = max_seconds
            return original( capture, vid_fps, sample_fps, threshold, max_seconds )

        with mock.patch.object( bu_video, 'detect_shot_cuts', spy ):
            self.run_one()
        self.assertEqual( seen.get( 'max_seconds' ), 1 )


class TestProfileSampleRate( BetaTvHarness ):
    """
    A profile's video_censor_fps must reach DETECTION, not just the
    filename. The unit tests cover resolution; this covers the wiring,
    which is where a rate change most easily applies to one stage and
    not another.
    """

    def _install_profiles( self, default_fps ):
        import copy
        saved = copy.deepcopy( betaconfig.detector_backend )
        self.addCleanup( setattr, betaconfig, 'detector_backend', saved )
        self.addCleanup( bu_config.invalidate_config_caches )
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend[backend]['profiles'] = {
            'default': 'dense', 'match_on': 'cuts_per_min',
            'variants': { 'dense': { 'min_cuts_per_min': 0,
                                     'video_censor_fps': default_fps } },
        }
        bu_config.invalidate_config_caches()

    def test_detection_samples_at_the_profiles_rate( self ):
        # Global rate is 5 in this harness. With shot cuts off the
        # footage has no measured rate and takes the default profile.
        betaconfig.shot_cut_detection_enabled = False
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=2 )
        self._install_profiles( 10 )
        self.assertEqual( self.run_one(), 'processed' )
        self.assertGreaterEqual(
            self.detector.frames_seen, 18,
            "2s at the profile's 10 fps should be ~20 samples; saw %d, which "
            "is the global 5 fps - the profile rate never reached detection"
            %( self.detector.frames_seen, ) )

    def test_a_different_rate_is_a_different_output( self ):
        betaconfig.shot_cut_detection_enabled = False
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=2 )
        self.assertEqual( self.run_one(), 'processed' )
        self._install_profiles( 10 )
        self.assertEqual(
            self.run_one(), 'processed',
            "raising the sample rate reused the output rendered at the old "
            "rate" )
        self.assertEqual( len( self.outputs() ), 2 )

    def test_a_repeat_run_at_a_profile_rate_still_skips( self ):
        betaconfig.shot_cut_detection_enabled = False
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=2 )
        self._install_profiles( 10 )
        self.assertEqual( self.run_one(), 'processed' )
        seen = self.detector.frames_seen
        self.assertEqual( self.run_one(), 'skipped' )
        self.assertEqual( self.detector.frames_seen, seen,
                          "a skipped file still ran detection" )


class TestInterruptHandling( BetaTvHarness ):
    """
    Ctrl-C must stop the run. It must NOT be absorbed by the per-file
    error handling that makes one bad file non-fatal.
    """

    def test_an_interrupt_propagates_through_the_per_file_handler( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )

        def interrupting_detect( *args, **kwargs ):
            raise bu_signals.Interrupted( 'stopped by user request' )

        with mock.patch.object( betatv, 'detect_boxes_for_video', interrupting_detect ):
            with self.assertRaises( bu_signals.Interrupted ):
                self.run_one()

    def test_an_ordinary_failure_is_still_absorbed( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )

        def failing_detect( *args, **kwargs ):
            raise RuntimeError( 'a bad file, not a user interrupt' )

        with mock.patch.object( betatv, 'detect_boxes_for_video', failing_detect ):
            self.assertEqual( self.run_one(), 'failed' )

    def test_detection_checkpoints_before_stopping( self ):
        # A stop must always be clean: whatever was detected so far is
        # on disk, so re-running resumes rather than restarting.
        import betautils_hash as bu_hash
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=4 )

        original_check = bu_signals.check
        state = { 'calls': 0 }

        def check_then_interrupt():
            state['calls'] += 1
            if state['calls'] > 3:
                raise bu_signals.Interrupted( 'stopped by user request' )

        with mock.patch.object( betatv.bu_signals, 'check', check_then_interrupt ):
            with self.assertRaises( bu_signals.Interrupted ):
                self.run_one()

        checkpoints = []
        for root, _dirs, names in os.walk( betaconst.vid_hash_dir ):
            checkpoints += [ name for name in names if name.endswith( '.checkpoint' ) ]
        self.assertTrue( checkpoints,
            "an interrupted detection pass must leave a checkpoint behind" )
        self.assertIs( bu_signals.check, original_check )


class TestSignalPlumbing( unittest.TestCase ):
    """betautils_signals on its own, with no video involved."""

    def setUp( self ):
        bu_signals.clear()

    def tearDown( self ):
        bu_signals.clear()

    def test_check_is_a_no_op_until_an_interrupt_is_requested( self ):
        bu_signals.check()   # must not raise

    def test_check_raises_once_requested( self ):
        bu_signals.request_interrupt()
        self.assertTrue( bu_signals.interrupted() )
        with self.assertRaises( bu_signals.Interrupted ):
            bu_signals.check()

    def test_interrupted_is_a_keyboard_interrupt_not_an_exception( self ):
        # This is the whole mechanism: `except Exception` handlers give
        # the pipeline its one-bad-file-is-not-fatal behaviour, and an
        # interrupt has to pass straight through them.
        self.assertTrue( issubclass( bu_signals.Interrupted, KeyboardInterrupt ) )
        self.assertFalse( issubclass( bu_signals.Interrupted, Exception ) )

        caught = False
        try:
            try:
                raise bu_signals.Interrupted( 'stop' )
            except Exception:
                caught = True
        except bu_signals.Interrupted:
            pass
        self.assertFalse( caught, "`except Exception` must not swallow an interrupt" )

    def test_clear_resets_the_flag( self ):
        bu_signals.request_interrupt()
        bu_signals.clear()
        self.assertFalse( bu_signals.interrupted() )
        bu_signals.check()


if __name__ == '__main__':
    unittest.main()
