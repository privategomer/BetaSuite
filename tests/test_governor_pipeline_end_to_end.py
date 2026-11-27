"""
test_governor_pipeline_end_to_end.py - a real betatv file run under a
governor that is slowing it, with a stub detector.

What each test prevents:
  - detection never yielding when the governor says to, or yielding past
    its caps
  - a governed run whose stats row does not say what the governor did
  - a governed run that comes out different: the same output name (all
    three cache keys) and the same frames as an ungoverned run
"""

import json
import os
import shutil
import types
import unittest

import betaconfig
import betautils_governor as bu_gov
import betautils_governor_run as bu_gov_run
from betautils_resources import signals as S

from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip
from tests.test_governor_render_process import _framemd5


class TestGovernedFile( BetaTvHarness ):

    def setUp( self ):
        super().setUp()
        self._saved_governor = getattr( betaconfig, 'governor', None )
        self.addCleanup( setattr, betaconfig, 'governor', self._saved_governor )
        betaconfig.governor = dict( self._saved_governor or {}, enabled=True, detection_pace_max=0.10 )

    def _rows( self ):
        with open( betaconfig.stats_path, 'r', encoding='UTF-8' ) as handle:
            return [ json.loads( line ) for line in handle if line.strip() ]

    def _governed( self ):
        clock = [ 100.0 ]
        slept = []
        runtime = bu_gov_run.Runtime( clock=lambda: clock[0], sleep=slept.append )
        runtime.slept = slept
        return runtime, clock

    def test_a_slowed_file_yields_within_its_caps_and_comes_out_the_same( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=4 )
        self.assertEqual( self.run_one(), 'processed' )
        plain_name = self.outputs()[0]
        plain_frames = _framemd5( os.path.join( self.censored, plain_name ) )
        # Fresh start: no output, no cached detections.
        os.remove( os.path.join( self.censored, plain_name ) )
        shutil.rmtree( betaconfig_cache_dir(), ignore_errors=True )
        os.remove( betaconfig.stats_path )

        runtime, clock = self._governed()
        with bu_gov_run.attached( runtime, betaconfig ):
            for _ in range( 3 ):   # on battery: priority low, detection paced
                clock[0] += 1.0
                runtime.on_sample( S( on_battery=True, battery_percent=60, memory_pressure=0.0,
                                      memory_available_mb=8000, other_cpu_percent=1.0 ) )
            self.assertEqual( runtime.decision().reasons[0]['code'], bu_gov.BATTERY )
            runtime.start_file()
            frames_before = self.detector.frames_seen
            self.assertEqual( self.run_one(), 'processed' )
            batches = len( self.detector.batch_sizes_seen )
            runtime.end_file()

        self.assertGreater( self.detector.frames_seen, frames_before, "detection ran again" )
        self.assertTrue( runtime.slept, "detection yielded" )
        self.assertLessEqual( max( runtime.slept ), bu_gov_run.MAX_YIELD_SECONDS )
        self.assertLessEqual( len( runtime.slept ), batches )
        row = self._rows()[0]
        self.assertEqual( row['governor_reasons'], [ bu_gov.BATTERY ] )
        self.assertAlmostEqual( row['governor_paced_seconds'], round( sum( runtime.slept ), 3 ), places=3 )
        self.assertIn( 'governor_throttled_seconds', row )
        self.assertIn( 'governor_fewest_workers', row )
        # The same output, name and frames.
        self.assertEqual( self.outputs(), [ plain_name ] )
        self.assertEqual( _framemd5( os.path.join( self.censored, plain_name ) ), plain_frames )

    def test_an_ungoverned_row_says_nothing_was_done( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ), seconds=2 )
        self.run_one()
        row = self._rows()[0]
        self.assertEqual( row['governor_paced_seconds'], 0.0 )
        self.assertEqual( row['governor_reasons'], [] )


def betaconfig_cache_dir():
    import betaconst
    return betaconst.vid_hash_dir


if __name__ == '__main__':
    unittest.main()
