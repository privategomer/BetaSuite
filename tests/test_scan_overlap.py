"""
test_scan_overlap.py - the shot-cut scan running alongside detection.

WHAT CHANGED
    With a declared profile (--profile, a path pattern, a source folder)
    the sample rate is known before the scan, so betatv starts the scan
    on a background thread and detects while it runs. Detection never
    reads cuts; tracking does, after detection. The scan took 4m38s in
    front of detection on a 35-minute file.

WHAT MUST NOT CHANGE
    The cuts tracking receives, the sample rate, and therefore every
    censored pixel. These tests run the same clip both ways and compare
    what reaches tracking, rather than trusting that a thread "should"
    produce the same list.
"""

import os
import shutil
import subprocess
import unittest
from unittest import mock

import betaconfig
import betaconst
import betautils_config as bu_config
import betautils_video as bu_video
import betatv

from tests.test_betatv_end_to_end import BetaTvHarness, _ffmpeg_available


def _build_clip_with_cuts( path, seconds_per_shot=2, shots=3, width=128, height=96, rate=10 ):
    """A clip made of visibly different shots, so the scan finds real cuts."""
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    sources = [ 'testsrc2', 'smptebars', 'rgbtestsrc', 'testsrc' ]
    command = [ 'ffmpeg', '-y', '-loglevel', 'error' ]
    for index in range( shots ):
        command += [ '-f', 'lavfi', '-i', '%s=size=%dx%d:rate=%d:duration=%d'%(
            sources[index % len( sources )], width, height, rate, seconds_per_shot ) ]
    command += [ '-filter_complex',
                 ''.join( '[%d:v]'%( i, ) for i in range( shots ) )
                 + 'concat=n=%d:v=1:a=0[v]'%( shots, ),
                 '-map', '[v]', '-c:v', 'libx264', '-preset', 'ultrafast',
                 '-pix_fmt', 'yuv420p', path ]
    subprocess.run( command, check=True )


class _OverlapHarness( BetaTvHarness ):
    """A clip with real cuts, and a spy on what reaches tracking."""

    def setUp( self ):
        super().setUp()
        self._saved_forced = getattr( betaconfig, 'force_structure_profile', None )
        betaconfig.force_structure_profile = None
        bu_config.invalidate_config_caches()
        _build_clip_with_cuts( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.seen = []
        real_prepare = betatv.bu_track.prepare_boxes_for_render

        def spy_prepare( raw_boxes, vid_w, vid_h, shot_cut_times=None, **kwargs ):
            self.seen.append( {
                'cuts': list( shot_cut_times ) if shot_cut_times is not None else None,
                'sample_fps': kwargs.get( 'sample_fps' ),
                'profile': kwargs.get( 'profile_name' ),
                'raw': len( raw_boxes ),
            } )
            return real_prepare( raw_boxes, vid_w, vid_h, shot_cut_times, **kwargs )

        self._prepare_patch = mock.patch.object(
            betatv.bu_track, 'prepare_boxes_for_render', spy_prepare )
        self._prepare_patch.start()

    def tearDown( self ):
        self._prepare_patch.stop()
        betaconfig.force_structure_profile = self._saved_forced
        bu_config.invalidate_config_caches()
        super().tearDown()

    def _clear_outputs_and_cut_cache( self ):
        shutil.rmtree( betaconst.shot_cut_dir, ignore_errors=True )
        for name in os.listdir( self.censored ):
            os.remove( os.path.join( self.censored, name ) )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestScanOverlap( _OverlapHarness ):

    def test_precondition_the_clip_has_cuts( self ):
        # Comparing two empty cut lists would prove nothing.
        self.assertEqual( self.run_one(), 'processed' )
        self.assertGreaterEqual( len( self.seen[0]['cuts'] ), 2 )

    def test_a_declared_profile_scans_in_the_background_with_identical_results( self ):
        self.assertEqual( self.run_one(), 'processed' )
        measured = self.seen[-1]

        self._clear_outputs_and_cut_cache()
        betaconfig.force_structure_profile = measured['profile']
        bu_config.invalidate_config_caches()
        constructed = []
        real_background = bu_video.BackgroundShotCutScan

        def spy_background( *args, **kwargs ):
            constructed.append( args )
            return real_background( *args, **kwargs )

        with mock.patch.object( betatv.bu_video, 'BackgroundShotCutScan', spy_background ):
            self.assertEqual( self.run_one(), 'processed' )
        declared = self.seen[-1]

        self.assertEqual( len( constructed ), 1, "the declared path must use the background scan" )
        self.assertEqual( declared['cuts'], measured['cuts'] )
        self.assertEqual( declared['sample_fps'], measured['sample_fps'] )
        self.assertEqual( declared['raw'], measured['raw'] )

    def test_an_undeclared_profile_still_scans_before_detecting( self ):
        # The sample rate depends on the measurement, so there is nothing
        # to overlap with; the background path must not be taken.
        with mock.patch.object( betatv.bu_video, 'BackgroundShotCutScan',
                                side_effect=AssertionError( 'background scan used' ) ):
            self.assertEqual( self.run_one(), 'processed' )

    def test_a_finished_file_with_a_declared_profile_skips_without_scanning( self ):
        betaconfig.force_structure_profile = 'scene'
        bu_config.invalidate_config_caches()
        self.assertEqual( self.run_one(), 'processed' )

        with mock.patch.object( betatv.bu_video, 'shot_cuts_for_source',
                                side_effect=AssertionError( 'scanned a finished file' ) ), \
             mock.patch.object( betatv.bu_video, 'BackgroundShotCutScan',
                                side_effect=AssertionError( 'scanned a finished file' ) ):
            self.assertEqual( self.run_one(), 'skipped' )

    def test_a_scan_failure_on_the_background_thread_reaches_the_caller( self ):
        betaconfig.force_structure_profile = 'scene'
        bu_config.invalidate_config_caches()
        with mock.patch.object( bu_video, 'shot_cuts_for_source',
                                side_effect=RuntimeError( 'decoder fell over' ) ):
            # Absorbed by the per-file handler like any other failure,
            # not lost on the thread with tracking given no cuts.
            self.assertEqual( self.run_one(), 'failed' )
        self.assertEqual( self.seen, [] )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestOffsetSliceCutTimebase( _OverlapHarness ):
    """
    A preview slice at offset X must hand tracking cuts at X + t, the
    timeline detections are stamped on, not the scan's own 0-based t.
    """

    def test_an_offset_slice_reports_cuts_where_the_full_run_does( self ):
        self.assertEqual( self.run_one(), 'processed' )
        full_run_cuts = self.seen[-1]['cuts']
        self.assertTrue( any( abs( t - 4.0 ) < 0.25 for t in full_run_cuts ),
                         "precondition: the full run sees the cut at 4s, got %r"%( full_run_cuts, ) )

        self._clear_outputs_and_cut_cache()
        saved = { name: getattr( betaconfig, name, None ) for name in (
            'preview_mode_enabled', 'preview_max_seconds', 'preview_start_seconds' ) }
        try:
            betaconfig.preview_mode_enabled = True
            betaconfig.preview_max_seconds = 3.0
            betaconfig.preview_start_seconds = 2.5
            bu_config.invalidate_config_caches()
            self.assertEqual( self.run_one(), 'processed' )
        finally:
            for name, value in saved.items():
                setattr( betaconfig, name, value )
            bu_config.invalidate_config_caches()
        slice_cuts = self.seen[-1]['cuts']

        # The slice covers 2.5s to 5.5s, so the 4s cut is inside it. On
        # the scan's own timeline it would read 1.5s, before the slice's
        # first detection at 2.5s, where tracking can never see it.
        self.assertTrue( slice_cuts, "the slice must find the cut inside it" )
        self.assertTrue( all( t >= 2.5 for t in slice_cuts ), slice_cuts )
        self.assertTrue( any( abs( t - 4.0 ) < 0.25 for t in slice_cuts ), slice_cuts )


class TestDeclaredAgainstMeasured( unittest.TestCase ):
    """The warning that a folder is costing detections."""

    def setUp( self ):
        self.logger = mock.MagicMock()

    def report( self, declared_profile, declared_fps, short_fraction ):
        signals = betatv.bu_detector.structure_signals(
            [ 10.0 + 0.5 * n for n in range( 30 ) ] if short_fraction else [], 60.0 )
        with mock.patch.object( betatv.bu_detector, 'measured_profile_selection',
                                return_value=( 'quick_cut' if short_fraction else 'scene',
                                               'measured short_shot_fraction' ) ), \
             mock.patch.object( betatv.bu_detector, 'get_profile_sample_fps',
                                side_effect=lambda name, *a: 18 if name == 'quick_cut' else 9 ):
            betatv._report_declared_against_measured(
                'clip.mp4', declared_profile, 'source folder compilation/', declared_fps,
                signals, self.logger )

    def test_a_sparser_declared_profile_than_measured_warns( self ):
        self.report( 'compilation', 9, short_fraction=True )
        self.assertTrue( self.logger.warning.called )
        self.assertIn( 'quick_cut', self.logger.warning.call_args[0][0] )

    def test_a_matching_profile_is_only_information( self ):
        self.report( 'scene', 9, short_fraction=False )
        self.assertFalse( self.logger.warning.called )
        self.assertTrue( self.logger.info.called )

    def test_a_denser_declared_profile_is_only_information( self ):
        # Declaring quick_cut on slow footage costs time, never detections.
        self.report( 'quick_cut', 18, short_fraction=False )
        self.assertFalse( self.logger.warning.called )


if __name__ == '__main__':
    unittest.main()
