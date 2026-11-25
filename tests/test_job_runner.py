"""
test_job_runner.py - betautils_job: the job runner betatv.py and the
engine share, against real clips with the stub detector.

Covers the event order of a normal job, a failed file and a cancel; that
one failed file does not stop the queue; pause then resume continuing
from the detection checkpoint and the finished render chunks, with the
same output as an uninterrupted run; cancel leaving no partial output; a
full disk stopping the job with its code; overrides applied for the job
only; and the state table.
"""

import hashlib
import os
import shutil
import threading
import unittest
from unittest import mock

import betaconfig
import betaconst
import betautils_config as bu_config
import betautils_job as bu_job
import betautils_signals as bu_signals
import betatv

from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip


def _sha256( path ):
    with open( path, 'rb' ) as handle:
        return hashlib.sha256( handle.read() ).hexdigest()


class Recorder:
    """Collects events; optionally acts on one (a pause or cancel mid-run)."""

    def __init__( self ):
        self.events = []
        self.on_event = None
        self.lock = threading.Lock()

    def __call__( self, name, data ):
        with self.lock:
            self.events.append( ( name, data ) )
        if self.on_event is not None:
            self.on_event( name, data )

    def names( self, skip_progress=True ):
        return [ self._short( name, data ) for name, data in self.events
                 if not ( skip_progress and name == 'job.progress' ) ]

    @staticmethod
    def _short( name, data ):
        if name == 'job.state':
            return 'state:%s'%( data['state'], )
        if name == 'job.file':
            return 'file%d:%s'%( data['file'], data['state'] )
        if name == 'job.decision':
            return 'decision%d:%s'%( data['file'], data['code'] )
        if name == 'job.error':
            return 'error%s:%s'%( data.get( 'file', '' ), data['code'] )
        return name

    def of( self, name ):
        return [ data for event, data in self.events if event == name ]


class JobHarness( BetaTvHarness ):

    def setUp( self ):
        super().setUp()
        self.recorder = Recorder()
        self.runner = bu_job.Runner( self.logger, emit=self.recorder )

    def job( self, *names, overrides=None ):
        files = [ bu_job.JobFile( os.path.join( self.uncensored, name ), self.censored ) for name in names ]
        return bu_job.Job( 'j1', files, overrides )

    def leftovers( self ):
        return sorted( name for name in os.listdir( self.censored ) if not name.endswith( '.mp4' ) )


class TestEventOrder( JobHarness ):

    def test_a_normal_job( self ):
        for name in ( 'a.mp4', 'b.mp4' ):
            _build_clip( os.path.join( self.uncensored, name ) )
        job = self.job( 'a.mp4', 'b.mp4' )
        self.assertEqual( self.runner.run( job ), bu_job.COMPLETED )
        self.assertEqual( self.recorder.names(), [
            'state:running',
            'file0:started', 'decision0:structure.profile', 'file0:processed',
            'file1:started', 'decision1:structure.profile', 'file1:processed',
            'state:completed' ] )
        final = self.recorder.of( 'job.state' )[-1]
        self.assertEqual( final['counts'], { 'total': 2, 'processed': 2, 'skipped': 0, 'failed': 0 } )
        outputs = [ data['output'] for data in self.recorder.of( 'job.file' ) if 'output' in data ]
        self.assertEqual( sorted( os.path.basename( path ) for path in outputs ), self.outputs() )

    def test_progress_sits_inside_its_file_and_names_its_stage( self ):
        for name in ( 'a.mp4', 'b.mp4' ):
            _build_clip( os.path.join( self.uncensored, name ) )
        self.runner.run( self.job( 'a.mp4', 'b.mp4' ) )
        current = None
        stages = set()
        for name, data in self.recorder.events:
            if name == 'job.file':
                current = data['file'] if data['state'] == 'started' else None
            elif name == 'job.progress':
                self.assertEqual( data['file'], current, "progress outside its file" )
                stages.add( data['stage'] )
                self.assertLessEqual( data['done'], data['total'] )
                if data['stage'] == 'detect':
                    self.assertEqual( data['size'], 320 )
        self.assertEqual( stages, { 'detect', 'render' } )

    def test_the_profile_decision_carries_its_rule_as_a_code( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )
        self.runner.run( self.job( 'a.mp4' ) )
        decision = self.recorder.of( 'job.decision' )[0]
        self.assertEqual( decision['code'], 'structure.profile' )
        self.assertIn( decision['params']['rule'], ( 'measured', 'default', 'disabled', 'no_profiles' ) )
        self.assertNotIn( ' ', decision['params']['rule'] )

    def test_a_failed_file_is_reported_and_the_queue_goes_on( self ):
        for name in ( 'a.mp4', 'b.mp4', 'c.mp4' ):
            _build_clip( os.path.join( self.uncensored, name ) )
        real = betatv.detect_boxes_for_video

        def fail_b( source, fname, *args, **kwargs ):
            if fname == 'b.mp4':
                raise RuntimeError( 'a bad file' )
            return real( source, fname, *args, **kwargs )

        with mock.patch.object( betatv, 'detect_boxes_for_video', fail_b ):
            state = self.runner.run( self.job( 'a.mp4', 'b.mp4', 'c.mp4' ) )
        self.assertEqual( state, bu_job.COMPLETED )
        self.assertEqual( self.recorder.names(), [
            'state:running',
            'file0:started', 'decision0:structure.profile', 'file0:processed',
            'file1:started', 'decision1:structure.profile', 'error1:file.failed', 'file1:failed',
            'file2:started', 'decision2:structure.profile', 'file2:processed',
            'state:completed' ] )
        error = self.recorder.of( 'job.error' )[0]
        self.assertEqual( error['params'], { 'exception_type': 'RuntimeError', 'stage': 'prepare' } )
        self.assertEqual( self.recorder.of( 'job.state' )[-1]['counts'],
                          { 'total': 3, 'processed': 2, 'skipped': 0, 'failed': 1 } )

    def test_skips_say_why( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )
        with open( os.path.join( self.uncensored, 'notes.txt' ), 'w' ) as handle:
            handle.write( 'not a video' )
        self.runner.run( self.job( 'a.mp4' ) )
        self.recorder.events.clear()
        self.runner.run( bu_job.Job( 'j2', [ bu_job.JobFile( os.path.join( self.uncensored, name ), self.censored )
                                              for name in ( 'a.mp4', 'notes.txt' ) ] ) )
        self.assertEqual( self.recorder.names(), [
            'state:running',
            # The measured profile sets the sample rate, which is in the
            # output name, so it is decided before the skip.
            'file0:started', 'decision0:structure.profile', 'decision0:output.present', 'file0:skipped',
            'file1:started', 'decision1:source.not_video', 'file1:skipped',
            'state:completed' ] )

    def test_a_full_disk_stops_the_job_with_its_code( self ):
        for name in ( 'a.mp4', 'b.mp4' ):
            _build_clip( os.path.join( self.uncensored, name ) )

        def full( *args, **kwargs ):
            raise OSError( 28, 'No space left on device' )

        with mock.patch.object( betatv, 'detect_boxes_for_video', full ):
            state = self.runner.run( self.job( 'a.mp4', 'b.mp4' ) )
        self.assertEqual( state, bu_job.STOPPED )
        self.assertEqual( self.recorder.names(), [
            'state:running',
            'file0:started', 'decision0:structure.profile', 'file0:failed',
            'error0:disk.full', 'state:stopped' ] )
        final = self.recorder.of( 'job.state' )[-1]
        self.assertEqual( final['code'], 'disk.full' )
        self.assertIn( 'free_bytes', final['params'] )

    def test_without_an_emitter_no_events_are_built_and_the_run_is_the_same( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )
        runner = bu_job.Runner( self.logger )
        with mock.patch.object( bu_job.FileReporter, 'progress' ) as progress:
            self.assertEqual( runner.run( self.job( 'a.mp4' ) ), bu_job.COMPLETED )
        progress.assert_not_called()
        self.assertEqual( len( self.outputs() ), 1 )


class TestPauseResumeCancel( JobHarness ):

    def setUp( self ):
        super().setUp()
        self._saved = { name: getattr( betaconfig, name, None ) for name in (
            'detection_checkpoint_frames', 'render_chunk_seconds' ) }
        betaconfig.detection_checkpoint_frames = 2
        betaconfig.render_chunk_seconds = 1
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        for name, value in self._saved.items():
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()
        super().tearDown()

    def _clean( self ):
        for folder in ( self.censored, os.path.join( self.tmpdir, 'cache' ) ):
            shutil.rmtree( folder, ignore_errors=True )
        os.makedirs( self.censored )
        os.remove( betaconfig.stats_path )

    def _reference( self ):
        """Output name and bytes of an uninterrupted run; then a clean slate."""
        job = self.job( 'a.mp4' )
        self.assertEqual( bu_job.Runner( self.logger ).run( job ), bu_job.COMPLETED )
        name = self.outputs()[0]
        digest = _sha256( os.path.join( self.censored, name ) )
        self.reference_detections = self.detections()
        self._clean()
        self.detector.frames_seen = 0
        return name, digest

    def detections( self ):
        """The finished detection caches, as sample times, per file name."""
        import betautils_hash as bu_hash
        found = {}
        for name in sorted( os.listdir( betaconst.vid_hash_dir ) ):
            if name.endswith( '.gz' ) and 'checkpoint' not in name:
                boxes = bu_hash.read_json( os.path.join( betaconst.vid_hash_dir, name ) )
                found[name] = [ round( box['t'], 4 ) for box in boxes ]
        self.assertTrue( found, "no finished detection cache was found" )
        return found

    def test_pause_in_detection_resumes_from_the_checkpoint( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ), seconds=4 )
        reference = self._reference()
        samples = 4 * betaconfig.video_censor_fps

        # Every frame batch is a progress update only every 0.25 s, so the
        # pause is asked for from the detector itself, part way through.
        stub = self.detector
        real = stub.raw_boxes_for_imgs

        asked = []

        def pausing( imgs, size, session, ts ):
            boxes = real( imgs, size, session, ts )
            if not asked and stub.frames_seen >= 7:
                asked.append( stub.frames_seen )
                bu_signals.request_stop( bu_signals.PAUSE )
            return boxes

        job = self.job( 'a.mp4' )
        with mock.patch.object( stub, 'raw_boxes_for_imgs', side_effect=pausing ):
            self.assertEqual( self.runner.run( job ), bu_job.PAUSED )
        self.assertEqual( self.recorder.names()[-1], 'state:paused' )
        self.assertFalse( bu_signals.interrupted(), "the handled stop is cleared" )
        self.assertEqual( job.next_index, 0 )
        self.assertEqual( self.outputs(), [] )
        seen_before = stub.frames_seen
        self.assertLess( seen_before, samples )
        self.assertEqual( [ thread.name for thread in threading.enumerate()
                            if thread.name in ( 'shot-cut-scan', 'person-pass' ) ], [] )

        self.recorder.events.clear()
        self.assertEqual( self.runner.run( job ), bu_job.COMPLETED )
        self.assertEqual( self.recorder.names()[0], 'state:running' )
        # The checkpoint covered the samples before the pause; only the
        # rest are detected again.
        self.assertLessEqual( stub.frames_seen, samples + 1 )
        self.assertEqual( self.detections(), self.reference_detections,
                          "every sample is detected exactly once across the pause" )
        self.assertEqual( ( self.outputs()[0], _sha256( os.path.join( self.censored, self.outputs()[0] ) ) ),
                          reference )

    def test_pause_in_render_reuses_the_finished_chunks( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ), seconds=4 )
        reference = self._reference()
        import betautils_render as bu_render
        real = bu_render.render_one_chunk
        calls = []

        def pause_after_first( plan, *args, **kwargs ):
            calls.append( plan.index )
            if len( calls ) == 2:
                bu_signals.request_stop( bu_signals.PAUSE )
            return real( plan, *args, **kwargs )

        job = self.job( 'a.mp4' )
        with mock.patch.object( bu_render, 'render_one_chunk', pause_after_first ):
            self.assertEqual( self.runner.run( job ), bu_job.PAUSED )
        chunks = [ name for name in self.leftovers() if '.part' in name and name.endswith( '.mkv' ) ]
        self.assertEqual( len( chunks ), 1, "the first chunk is kept for the resume" )
        self.assertEqual( self.runner.run( job ), bu_job.COMPLETED )
        import json
        with open( betaconfig.stats_path, encoding='UTF-8' ) as handle:
            row = json.loads( handle.readlines()[-1] )
        self.assertEqual( row['render_chunks_reused'], 1 )
        self.assertEqual( self.leftovers(), [] )
        self.assertEqual( ( self.outputs()[0], _sha256( os.path.join( self.censored, self.outputs()[0] ) ) ),
                          reference )

    def test_cancel_in_render_leaves_no_partial_output( self ):
        for name in ( 'a.mp4', 'b.mp4' ):
            _build_clip( os.path.join( self.uncensored, name ), seconds=4 )
        import betautils_render as bu_render
        real = bu_render.render_one_chunk
        calls = []

        def cancel_on_second( plan, *args, **kwargs ):
            calls.append( plan.index )
            if len( calls ) == 2:
                bu_signals.request_stop( bu_signals.CANCEL )
            return real( plan, *args, **kwargs )

        with mock.patch.object( bu_render, 'render_one_chunk', cancel_on_second ):
            state = self.runner.run( self.job( 'a.mp4', 'b.mp4' ) )
        self.assertEqual( state, bu_job.CANCELLED )
        self.assertEqual( self.recorder.names(), [
            'state:running', 'file0:started', 'decision0:structure.profile', 'state:cancelled' ] )
        self.assertEqual( os.listdir( self.censored ), [], "no chunk, intermediate or temp is left" )
        self.assertTrue( os.listdir( betaconst.vid_hash_dir ), "detections are cache and stay" )
        self.assertFalse( bu_signals.interrupted() )

    def test_cancel_while_paused_removes_the_partial_output( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ), seconds=4 )
        import betautils_render as bu_render
        real = bu_render.render_one_chunk
        calls = []

        def pause_on_second( plan, *args, **kwargs ):
            calls.append( plan.index )
            if len( calls ) == 2:
                bu_signals.request_stop( bu_signals.PAUSE )
            return real( plan, *args, **kwargs )

        job = self.job( 'a.mp4' )
        with mock.patch.object( bu_render, 'render_one_chunk', pause_on_second ):
            self.assertEqual( self.runner.run( job ), bu_job.PAUSED )
        self.assertTrue( self.leftovers() )
        self.runner.cancel_idle( job )
        self.assertEqual( job.state, bu_job.CANCELLED )
        self.assertEqual( os.listdir( self.censored ), [] )

    def test_pause_between_files_resumes_at_the_next_file( self ):
        for name in ( 'a.mp4', 'b.mp4' ):
            _build_clip( os.path.join( self.uncensored, name ) )

        def pause_after_first( name, data ):
            if name == 'job.file' and data['file'] == 0 and data['state'] == 'processed':
                bu_signals.request_stop( bu_signals.PAUSE )
        self.recorder.on_event = pause_after_first
        job = self.job( 'a.mp4', 'b.mp4' )
        self.assertEqual( self.runner.run( job ), bu_job.PAUSED )
        self.recorder.on_event = None
        self.recorder.events.clear()
        self.assertEqual( self.runner.run( job ), bu_job.COMPLETED )
        self.assertEqual( [ data['file'] for data in self.recorder.of( 'job.file' ) ], [ 1, 1 ] )
        self.assertEqual( self.recorder.of( 'job.state' )[-1]['counts']['processed'], 2 )

    def test_pause_waits_for_a_scan_running_beside_detection( self ):
        import betautils_detector as bu_detector
        variants = sorted( bu_detector.get_profiles().get( 'variants' ) or {} )
        if not variants:
            self.skipTest( "the selected backend has no profiles" )
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ), seconds=4 )
        saved = getattr( betaconfig, 'force_structure_profile', None )
        betaconfig.force_structure_profile = variants[0]
        bu_config.invalidate_config_caches()
        self.addCleanup( setattr, betaconfig, 'force_structure_profile', saved )
        import betautils_video as bu_video
        real_scan = bu_video.shot_cuts_for_source
        started = threading.Event()

        def slow_scan( *args, **kwargs ):
            started.set()
            import time
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                bu_signals.check()
                time.sleep( 0.01 )
            return real_scan( *args, **kwargs )

        stub = self.detector
        real = stub.raw_boxes_for_imgs

        def pausing( imgs, size, session, ts ):
            started.wait( 5 )
            bu_signals.request_stop( bu_signals.PAUSE )
            return real( imgs, size, session, ts )

        with mock.patch.object( bu_video, 'shot_cuts_for_source', slow_scan ), \
                mock.patch.object( stub, 'raw_boxes_for_imgs', side_effect=pausing ):
            self.assertEqual( self.runner.run( self.job( 'a.mp4' ) ), bu_job.PAUSED )
        self.assertEqual( [ thread.name for thread in threading.enumerate()
                            if thread.name == 'shot-cut-scan' ], [] )

    def test_a_stronger_stop_wins_over_a_pause( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )
        bu_signals.request_stop( bu_signals.PAUSE )
        bu_signals.request_stop( bu_signals.CANCEL )
        bu_signals.request_stop( bu_signals.PAUSE )
        self.assertEqual( self.runner.run( self.job( 'a.mp4' ) ), bu_job.CANCELLED )

    def test_shutdown_stops_with_its_code_and_keeps_the_checkpoint( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )
        bu_signals.request_stop( bu_signals.SHUTDOWN )
        job = self.job( 'a.mp4' )
        self.assertEqual( self.runner.run( job ), bu_job.STOPPED )
        self.assertEqual( job.code, 'engine.stopping' )

    def test_ctrl_c_still_raises_for_the_command_line( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )
        bu_signals.request_interrupt()
        with self.assertRaises( bu_signals.Interrupted ) as raised:
            self.runner.run( self.job( 'a.mp4' ) )
        self.assertEqual( raised.exception.reason, bu_signals.INTERRUPT )


class TestOverrides( JobHarness ):

    def test_overrides_apply_for_the_job_only( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )
        before = betaconfig.encode_preset
        seen = []
        real = betatv._build_paths

        def spy( *args, **kwargs ):
            seen.append( betaconfig.encode_crf )
            return real( *args, **kwargs )

        with mock.patch.object( betatv, '_build_paths', spy ):
            self.runner.run( self.job( 'a.mp4', overrides={ ( 'encode_crf', ): 31 } ) )
        self.assertEqual( set( seen ), { 31 } )
        self.assertNotEqual( betaconfig.encode_crf, 31 )
        self.assertEqual( betaconfig.encode_preset, before )
        plain = self.outputs()
        self.assertEqual( len( plain ), 1 )
        self.runner.run( bu_job.Job( 'j2', [ bu_job.JobFile( os.path.join( self.uncensored, 'a.mp4' ),
                                                             self.censored ) ] ) )
        self.assertEqual( len( self.outputs() ), 2, "the override moved the encode key" )

    def test_overrides_are_put_back_when_the_job_fails( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )
        original = betaconfig.encode_crf

        class Broken:
            def get_session( self ):
                raise RuntimeError( 'no model' )

        with mock.patch.object( betatv.bu_detector, 'get_detector', return_value=Broken() ):
            with self.assertRaises( RuntimeError ):
                self.runner.run( self.job( 'a.mp4', overrides={ ( 'encode_crf', ): 31 } ) )
        self.assertEqual( betaconfig.encode_crf, original )

    def test_a_load_hook_can_stop_the_job_with_a_code( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ) )

        def refuse():
            raise bu_job.JobStopped( 'model.load_failed', { 'exception_type': 'RuntimeError' } )

        runner = bu_job.Runner( self.logger, emit=self.recorder, load_session=refuse )
        self.assertEqual( runner.run( self.job( 'a.mp4' ) ), bu_job.STOPPED )
        self.assertEqual( self.recorder.names(),
                          [ 'state:running', 'error:model.load_failed', 'state:stopped' ] )


class TestStates( unittest.TestCase ):

    def test_the_table( self ):
        allowed = { ( a, b ) for a, targets in bu_job.TRANSITIONS.items() for b in targets }
        self.assertEqual( allowed, {
            ( 'queued', 'running' ), ( 'queued', 'cancelled' ), ( 'queued', 'stopped' ),
            ( 'running', 'paused' ), ( 'running', 'completed' ), ( 'running', 'cancelled' ),
            ( 'running', 'stopped' ),
            ( 'paused', 'running' ), ( 'paused', 'cancelled' ), ( 'paused', 'stopped' ) } )
        for state in bu_job.FINAL:
            self.assertEqual( bu_job.TRANSITIONS[state], frozenset() )

    def test_set_state_refuses_what_the_table_does_not_allow( self ):
        runner = bu_job.Runner( mock.Mock(), pipeline=mock.Mock() )
        job = bu_job.Job( 'j', [] )
        for state in ( bu_job.PAUSED, bu_job.COMPLETED, bu_job.QUEUED ):
            with self.assertRaises( bu_job.BadTransition ):
                runner.set_state( job, state )
        runner.set_state( job, bu_job.RUNNING )
        runner.set_state( job, bu_job.COMPLETED )
        with self.assertRaises( bu_job.BadTransition ):
            runner.set_state( job, bu_job.RUNNING )

    def test_a_finished_job_cannot_be_run_again( self ):
        runner = bu_job.Runner( mock.Mock(), pipeline=mock.Mock() )
        job = bu_job.Job( 'j', [] )
        job.state = bu_job.CANCELLED
        with self.assertRaises( bu_job.BadTransition ):
            runner.run( job )


class TestOutputFolder( unittest.TestCase ):

    def test_under_the_input_folder_it_mirrors_and_elsewhere_it_is_the_top( self ):
        inputs = os.path.abspath( betaconst.video_path_uncensored )
        outputs = betaconst.video_path_censored
        self.assertEqual( os.path.normpath( bu_job.output_folder_for( os.path.join( inputs, 'x.mp4' ) ) ),
                          os.path.normpath( outputs ) )
        self.assertEqual( os.path.normpath( bu_job.output_folder_for( os.path.join( inputs, 'a', 'b', 'x.mp4' ) ) ),
                          os.path.normpath( os.path.join( outputs, 'a', 'b' ) ) )
        elsewhere = os.path.join( os.path.dirname( inputs ), inputs + '-other', 'x.mp4' )
        self.assertEqual( bu_job.output_folder_for( elsewhere ), outputs )


class TestPartialOutputs( unittest.TestCase ):

    def test_everything_named_after_the_final_but_the_final_itself( self ):
        import tempfile
        import betautils_cache_paths as bu_cache
        folder = tempfile.mkdtemp()
        self.addCleanup( shutil.rmtree, folder, True )
        base = os.path.join( folder, 'a-0123-320-5-d1-c2-e3' )
        names = [ base + '.mp4', base + '.tmp.mp4', base + '.video.mkv', base + '.video.part0000.mkv',
                  base + '.video.part0000.mkv.meta.json', base + '.video.mkv.concat.txt',
                  base + '-preview@1.0s+1.0s.mp4', os.path.join( folder, 'other.mp4' ) ]
        for name in names:
            open( name, 'w' ).close()
        self.assertEqual( bu_cache.partial_outputs( base + '.mp4' ), sorted( names[1:6] ) )


class TestProgressSink( unittest.TestCase ):

    def test_the_sink_hears_counts_during_a_stage_and_at_its_end( self ):
        import betautils_log as bu_log
        heard = []
        previous = bu_log.set_progress_sink( lambda label, done, total, rate: heard.append( done ) )
        self.addCleanup( bu_log.set_progress_sink, previous )
        with bu_log.ProgressReporter( 'render', total=3, min_interval=0, logger=mock.Mock() ) as progress:
            for done in ( 1, 2, 3 ):
                progress.update( done )
            self.assertEqual( heard, [ 1, 2, 3 ] )
        self.assertEqual( heard, [ 1, 2, 3, 3 ] )

    def test_no_sink_no_calls( self ):
        import betautils_log as bu_log
        self.assertIsNone( bu_log.set_progress_sink( None ) )


class TestStopReasons( unittest.TestCase ):

    def tearDown( self ):
        bu_signals.clear()

    def test_strength_order( self ):
        for weaker, stronger in ( ( 'pause', 'cancel' ), ( 'cancel', 'interrupt' ), ( 'pause', 'shutdown' ) ):
            bu_signals.clear()
            bu_signals.request_stop( weaker )
            bu_signals.request_stop( stronger )
            self.assertEqual( bu_signals.stop_reason(), stronger )
            bu_signals.request_stop( weaker )
            self.assertEqual( bu_signals.stop_reason(), stronger )

    def test_check_raises_with_the_reason_and_clear_resets_it( self ):
        bu_signals.request_stop( 'pause' )
        with self.assertRaises( bu_signals.Interrupted ) as raised:
            bu_signals.check()
        self.assertEqual( raised.exception.reason, 'pause' )
        bu_signals.clear()
        self.assertIsNone( bu_signals.stop_reason() )
        bu_signals.check()

    def test_an_unknown_reason_is_refused( self ):
        with self.assertRaises( ValueError ):
            bu_signals.request_stop( 'nap' )
        self.assertFalse( bu_signals.interrupted() )


if __name__ == '__main__':
    unittest.main()
