"""
test_estimate_probe.py - the first estimate's probe of this machine.

Covers the probe's slice, its measurement on a real clip with the stub
detector (nothing kept on disk), when job.estimate starts it (rates
missing, idle, not already measured or tried, no command-line run),
probe rates filling in only where the history has none, and the queue
holding jobs while it runs.
"""

import os
import threading
import unittest
from unittest import mock

import betautils_estimate_probe as bu_probe
import betautils_serve_estimate as bu_serve_estimate

from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip
from tests import test_estimate as estimate_tests
from tests.test_estimate import IDENTITY, probe, timing
from tests.test_serve_job import _QueueTest


class TestSlice( unittest.TestCase ):

    def test_a_tenth_of_the_way_in( self ):
        self.assertEqual( bu_probe.slice_for( 100 ), ( 10.0, 3.0 ) )

    def test_never_past_the_end( self ):
        self.assertEqual( bu_probe.slice_for( 10 ), ( 1.0, 3.0 ) )
        offset, length = bu_probe.slice_for( 3.2 )
        self.assertAlmostEqual( offset, 0.2 )
        self.assertEqual( length, 3.0 )

    def test_a_clip_shorter_than_the_slice_is_all_of_it( self ):
        self.assertEqual( bu_probe.slice_for( 2 ), ( 0.0, 2.0 ) )
        self.assertEqual( bu_probe.slice_for( 0 ), ( 0.0, 0.0 ) )


class _Clock:
    def __init__( self, *times ):
        self.times = list( times )

    def __call__( self ):
        return self.times.pop( 0 )


class _Detector:
    def __init__( self ):
        self.batches = []

    def raw_boxes_for_imgs( self, imgs, size, session, ts ):
        self.batches.append( ( size, len( imgs ) ) )
        return []


class TestRates( unittest.TestCase ):

    def source( self, frames=10, fps=10.0, width=100, height=50 ):
        import types
        return types.SimpleNamespace( cap='cap', vid_fps=fps, num_frames=frames, width=width, height=height )

    def test_detect_rate_counts_every_size_in_batches( self ):
        detector = _Detector()
        frames = [ ( index, index / 5.0, 'frame' ) for index in range( 7 ) ]
        with mock.patch( 'betautils_video.iter_sampled_frames', side_effect=lambda *a, **k: iter( frames ) ):
            rate = bu_probe.detect_rate( self.source(), 's', [ 320, 640 ], 5, 3, 0.0, 3.0, detector,
                                         clock=_Clock( 10.0, 17.0 ) )
        self.assertEqual( detector.batches, [ ( 320, 3 ), ( 320, 3 ), ( 320, 1 ), ( 640, 3 ), ( 640, 3 ), ( 640, 1 ) ] )
        self.assertEqual( rate, 14 / 7.0 )

    def test_detect_rate_of_nothing_is_none( self ):
        with mock.patch( 'betautils_video.iter_sampled_frames', side_effect=lambda *a, **k: iter( [] ) ):
            self.assertIsNone( bu_probe.detect_rate( self.source(), 's', [ 320 ], 5, 3, 0.0, 3.0, _Detector(),
                                                     clock=_Clock( 1.0, 2.0 ) ) )

    def test_render_rate_is_pixel_frames_per_second_and_its_folder_is_removed( self ):
        seen = []

        def render( source, boxes, intermediate, final, preset, preview, unlimited, offset, length, logger ):
            seen.append( ( os.path.dirname( final ), boxes, preset, preview, offset, length ) )
            with open( final, 'w' ) as handle:
                handle.write( 'x' )

        rate = bu_probe.render_rate( self.source(), 'fast', 1.0, 2.0, None, clock=_Clock( 5.0, 9.0 ), render=render )
        self.assertEqual( rate, 20 * 100 * 50 / 4.0 )
        [ ( folder, boxes, preset, preview, offset, length ) ] = seen
        self.assertEqual( ( boxes, preset, preview, offset, length ), ( [], 'fast', True, 1.0, 2.0 ) )
        self.assertFalse( os.path.exists( folder ) )

    def test_a_video_with_no_length_measures_nothing( self ):
        import types
        source = types.SimpleNamespace( cap='cap', vid_fps=30.0, num_frames=0, width=10, height=10,
                                        release=lambda: None )
        with mock.patch( 'betautils_video.open_video_source', return_value=source ), \
                mock.patch.object( bu_probe, 'detect_rate' ) as detect, \
                mock.patch.object( bu_probe, 'render_rate' ) as render:
            self.assertEqual( bu_probe.probe( '/a.mp4', 'h', 's', estimate_tests.TestAnswer.CTX, None ),
                              { 'detect': None, 'render': None } )
        detect.assert_not_called()
        render.assert_not_called()


class TestProbeOnAClip( BetaTvHarness ):

    def test_measures_both_rates_and_keeps_nothing( self ):
        clip = os.path.join( self.uncensored, 'clip.mp4' )
        _build_clip( clip )
        before = sorted( os.listdir( self.tmpdir ) )
        ctx = dict( estimate_tests.TestAnswer.CTX, sizes=[ 320 ], sample_fps=5, identity=dict( IDENTITY, preset='ultrafast' ) )
        rates = bu_probe.probe( clip, 'probehash', object(), ctx, self.logger )
        self.assertGreater( rates['detect'], 0 )
        self.assertGreater( rates['render'], 0 )
        self.assertEqual( self.detector.frames_seen, 15, "3 s at 5 fps, one size" )
        self.assertEqual( sorted( os.listdir( self.tmpdir ) ), before )
        self.assertEqual( os.listdir( self.censored ), [] )


class _Jobs:
    """A queue stand-in: start_probe runs the work at once, on this thread."""

    def __init__( self, idle=True ):
        self.idle = idle
        self.probing = False
        self.started = []

    def start_probe( self, work, done ):
        if not self.idle:
            return False
        self.started.append( work )
        try:
            work( 'session' )
            ok = True
        except Exception:
            ok = False
        done( ok )
        return True


class _Server:
    def __init__( self ):
        import logging
        self._logger = logging.getLogger( 'test_estimate_probe' )
        self.events = []

    def send_event( self, name, data ):
        self.events.append( ( name, data ) )


class TestMaybeProbe( unittest.TestCase ):

    CTX = estimate_tests.TestAnswer.CTX

    def setUp( self ):
        patcher = mock.patch.object( bu_serve_estimate, 'PROBED', {} )
        patcher.start()
        self.addCleanup( patcher.stop )
        patcher = mock.patch.object( bu_serve_estimate, '_TRIED', set() )
        patcher.start()
        self.addCleanup( patcher.stop )
        patcher = mock.patch( 'betautils_runlock.library_held', return_value=False )
        self.held = patcher.start()
        self.addCleanup( patcher.stop )
        self.server = _Server()
        self.measured = []

    def measure( self, rates=None ):
        def run( path, digest, session, ctx, logger ):
            self.measured.append( ( path, digest, session ) )
            return rates if rates is not None else { 'detect': 40.0, 'render': 1e9 }
        return run

    def result( self, **fields ):
        out = { 'seconds': None, 'missing': [ 'detect', 'render' ], 'probing': False }
        out.update( fields )
        return out

    def maybe( self, jobs=None, result=None, files=( '/a.mp4', '/b.mp4' ), rates=None ):
        result = result or self.result()
        bu_serve_estimate.maybe_probe( self.server, jobs or _Jobs(), list( files ), self.CTX,
                                       lambda path: 'h-' + path, result, measure=self.measure( rates ) )
        return result

    def test_a_missing_rate_starts_a_probe_of_the_first_file_and_says_when_it_ends( self ):
        result = self.maybe()
        self.assertTrue( result['probing'] )
        self.assertEqual( self.measured, [ ( '/a.mp4', 'h-/a.mp4', 'session' ) ] )
        self.assertEqual( self.server.events, [ ( 'job.probed', { 'ok': True } ) ] )
        self.assertEqual( bu_serve_estimate.probed_for( self.CTX ), { 'detect': 40.0, 'render': 1e9 } )

    def test_measured_settings_are_not_probed_again( self ):
        self.maybe()
        self.assertFalse( self.maybe()['probing'] )
        self.assertEqual( len( self.measured ), 1 )

    def test_other_settings_are_probed_on_their_own( self ):
        self.maybe()
        bu_serve_estimate.maybe_probe( self.server, _Jobs(), [ '/a.mp4' ], dict( self.CTX, identity=dict(
            IDENTITY, variant='640m' ) ), lambda path: None, self.result(), measure=self.measure() )
        self.assertEqual( len( self.measured ), 2 )

    def test_a_probe_that_measured_nothing_tries_the_next_file_then_stops( self ):
        nothing = { 'detect': None, 'render': None }
        self.maybe( rates=nothing )
        self.assertEqual( self.server.events, [ ( 'job.probed', { 'ok': False } ) ] )
        self.maybe( rates=nothing )
        self.assertFalse( self.maybe( rates=nothing )['probing'] )
        self.assertEqual( [ path for path, _, _ in self.measured ], [ '/a.mp4', '/b.mp4' ] )

    def test_a_probe_that_raises_is_tried_and_reported( self ):
        def boom( *args ):
            raise RuntimeError( 'x' )
        result = self.result()
        bu_serve_estimate.maybe_probe( self.server, _Jobs(), [ '/a.mp4' ], self.CTX, lambda path: None, result,
                                       measure=boom )
        self.assertEqual( self.server.events, [ ( 'job.probed', { 'ok': False } ) ] )
        self.assertFalse( self.maybe( files=( '/a.mp4', ) )['probing'] )

    def test_no_probe_when_the_estimate_is_known_or_nothing_is_missing( self ):
        self.assertFalse( self.maybe( result=self.result( seconds=12.0 ) )['probing'] )
        self.assertFalse( self.maybe( result=self.result( missing=[] ) )['probing'] )
        self.assertEqual( self.measured, [] )

    def test_no_probe_while_busy_or_while_a_command_line_run_holds_the_library( self ):
        self.assertFalse( self.maybe( jobs=_Jobs( idle=False ) )['probing'] )
        self.held.return_value = True
        self.assertFalse( self.maybe()['probing'] )
        self.assertEqual( self.measured, [] )

    def test_a_probe_already_running_is_reported_and_not_started_again( self ):
        jobs = _Jobs()
        jobs.probing = True
        self.assertTrue( self.maybe( jobs=jobs )['probing'] )
        self.assertEqual( jobs.started, [] )


class TestProbedRates( unittest.TestCase ):

    def answer( self, timings, probed ):
        return bu_serve_estimate.answer( [ '/a.mp4' ], estimate_tests.TestAnswer.CTX, timings, lambda path: None,
                                         lambda key: False, probe(), probed=probed )

    def test_probe_rates_give_an_estimate_with_basis_probe( self ):
        out = self.answer( [], { 'detect': 50.0, 'render': 3000 * 1920 * 1080 / 20.0 } )
        self.assertEqual( ( out['seconds'], out['basis'], out['missing'] ), ( 30.0, 'probe', [] ) )

    def test_history_wins_where_it_has_a_rate( self ):
        out = self.answer( [ timing( cache='cached' ) ], { 'detect': 50.0, 'render': 1.0 } )
        self.assertEqual( out['basis'], 'history' )
        self.assertEqual( out['seconds'], 32.0, "render from history, detect from the probe" )

    def test_what_neither_gives_is_missing( self ):
        out = self.answer( [], { 'detect': 50.0, 'render': None } )
        self.assertEqual( ( out['seconds'], out['missing'] ), ( None, [ 'render' ] ) )


class TestQueueHoldsJobsWhileProbing( _QueueTest ):

    def test_a_job_submitted_during_a_probe_waits_for_it( self ):
        gate = threading.Event()
        ended = []
        self.assertTrue( self.server.jobs.start_probe( lambda session: gate.wait( 10 ), ended.append ) )
        self.assertTrue( self.server.jobs.probing )
        job = self.submit( 'a.mp4' )
        self.assertFalse( self.server.jobs.start_probe( lambda session: None, ended.append ), "one at a time" )
        import time
        time.sleep( 0.2 )
        self.assertEqual( self.pipeline.started, [], "the job waits for the probe" )
        gate.set()
        self.wait_state( job, 'completed' )
        self.assertEqual( ended, [ True ] )
        self.assertFalse( self.server.jobs.probing )

    def test_one_probe_at_a_time( self ):
        gate = threading.Event()
        ended = threading.Event()
        self.assertTrue( self.server.jobs.start_probe( lambda session: gate.wait( 10 ), lambda ok: ended.set() ) )
        self.assertFalse( self.server.jobs.start_probe( lambda session: None, lambda ok: None ) )
        gate.set()
        self.assertTrue( ended.wait( 10 ) )

    def test_no_probe_while_a_job_is_queued( self ):
        self.pipeline.auto = False
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        self.assertFalse( self.server.jobs.start_probe( lambda session: None, lambda ok: None ) )
        self.pipeline.auto = True
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( job, 'completed' )

    def test_done_is_called_when_the_model_does_not_load( self ):
        ended = threading.Event()
        result = []

        def done( ok ):
            result.append( ok )
            ended.set()

        with mock.patch.object( self.server.jobs, '_load_session', side_effect=RuntimeError( 'no model' ) ):
            self.assertTrue( self.server.jobs.start_probe( lambda session: None, done ) )
            self.assertTrue( ended.wait( 10 ) )
        self.assertEqual( result, [ False ] )
        self.assertFalse( self.server.jobs.probing )


if __name__ == '__main__':
    unittest.main()
