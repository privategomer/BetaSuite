"""
test_replay_structure.py - the tuning replay selects the profile a real
run would, from the same measurement.

THE BUG
    measure_replay passed select_profile_name a cut rate and nothing
    else. Profiles match on short_shot_fraction, so the rate was ignored
    and every replayed video fell through to the default; no source_path
    meant no folder rule ever applied; and auto_tune never supplied cuts,
    so replays also tracked across scene cuts that real runs block at.
    Every auto_tune verdict described a configuration that never runs.
"""

import copy
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_tuning as bu_tuning
import betautils_video as bu_video


PROFILES = {
    'default': 'scene',
    'match_on': 'short_shot_fraction',
    'variants': {
        'quick_cut': { 'min_short_shot_fraction': 0.15, 'video_censor_fps': 18 },
        'compilation': { 'manual_only': True },
        'scene': {},
    },
}


@unittest.skipUnless( shutil.which( 'ffmpeg' ), "ffmpeg is needed to build the test clip" )
class TestReplaySelectsLikeARealRun( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-replay-' )
        self.root = os.path.join( self.tmp, 'uncensored_vids' )
        self._saved_const = { name: getattr( betaconst, name ) for name in (
            'video_path_uncensored', 'shot_cut_dir', 'vid_hash_dir' ) }
        betaconst.video_path_uncensored = self.root + os.sep
        betaconst.shot_cut_dir = os.path.join( self.tmp, 'shot_cuts' )
        betaconst.vid_hash_dir = os.path.join( self.tmp, 'vid_hashes' )
        self._saved_vid_dir = bu_cache.VID_HASH_DIR
        bu_cache.VID_HASH_DIR = betaconst.vid_hash_dir

        self._saved_backend = copy.deepcopy( betaconfig.detector_backend )
        self._saved = { name: getattr( betaconfig, name, None ) for name in (
            'profile_by_source_dir', 'profile_by_path_pattern', 'force_structure_profile',
            'video_censor_fps', 'shot_cut_threshold' ) }
        self.backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.detector_backend[self.backend]['profiles'] = copy.deepcopy( PROFILES )
        betaconfig.profile_by_source_dir = {}
        betaconfig.profile_by_path_pattern = {}
        betaconfig.force_structure_profile = None
        betaconfig.video_censor_fps = 9
        betaconfig.shot_cut_threshold = 0.5
        bu_config.invalidate_config_caches()
        bu_tuning.clear_replay_caches()
        bu_video._DURATION_CACHE.clear()

        self.seen = []
        real_prepare = bu_tuning.bu_track.prepare_boxes_for_render

        def spy( raw_boxes, vid_w, vid_h, **kwargs ):
            self.seen.append( kwargs )
            return real_prepare( raw_boxes, vid_w, vid_h, **kwargs )

        self._patch = mock.patch.object( bu_tuning.bu_track, 'prepare_boxes_for_render', spy )
        self._patch.start()

    def tearDown( self ):
        self._patch.stop()
        for name, value in self._saved_const.items():
            setattr( betaconst, name, value )
        bu_cache.VID_HASH_DIR = self._saved_vid_dir
        betaconfig.detector_backend = self._saved_backend
        for name, value in self._saved.items():
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()
        bu_tuning.clear_replay_caches()
        shutil.rmtree( self.tmp, ignore_errors=True )

    def _clip( self, folder, seconds=60 ):
        path = os.path.join( self.root, folder, 'clip.mp4' ) if folder else os.path.join( self.root, 'clip.mp4' )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        subprocess.run( [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                          'color=c=gray:size=64x48:rate=10:duration=%d'%( seconds, ),
                          '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', path ],
                        check=True )
        return path

    def _caches( self, file_hash, cuts ):
        detection = bu_cache.box_hash_path_for( file_hash, 640, 9, 0.12, self.backend, det_key='abc123' )
        bu_hash.write_json( [ { 'x': 10.0, 'y': 10.0, 'w': 20.0, 'h': 20.0, 'score': 0.9,
                                'class_id': 'exposed_breast', 't': t / 9.0, 'size': 640 }
                              for t in range( 0, 540, 3 ) ], detection )
        bu_hash.write_json( cuts, bu_cache.shot_cut_path_for( file_hash, 9, 0.5 ) )
        return { file_hash: [ detection ] }

    def replay( self, path, cuts ):
        videos = self._caches( 'feedfacecafebeef', cuts )
        bu_tuning.measure_replay( videos, { 'feedfacecafebeef': path }, self.backend,
                                  [ 'exposed_breast' ] )
        return self.seen[-1]

    def test_fast_cutting_replays_under_quick_cut( self ):
        # 30 cuts half a second apart in 60s: 25% of runtime in sub-1s
        # shots. The old replay passed only a cut rate and got 'scene'.
        seen = self.replay( self._clip( None ), [ 10.0 + 0.5 * n for n in range( 30 ) ] )
        self.assertEqual( seen['profile_name'], 'quick_cut' )

    def test_the_folder_convention_applies_in_replay( self ):
        betaconfig.profile_by_source_dir = { 'comps': 'compilation' }
        bu_config.invalidate_config_caches()
        seen = self.replay( self._clip( 'comps' ), [] )
        self.assertEqual( seen['profile_name'], 'compilation' )

    def test_replay_tracking_receives_the_cuts( self ):
        cuts = [ 12.0, 30.0 ]
        seen = self.replay( self._clip( None ), cuts )
        self.assertEqual( seen['shot_cut_times'], cuts )

    def test_replay_uses_the_cached_detections_own_rate( self ):
        seen = self.replay( self._clip( None ), [ 10.0 + 0.5 * n for n in range( 30 ) ] )
        # quick_cut asks for 18, but these detections were sampled at 9.
        self.assertEqual( seen['sample_fps'], 9 )

    def test_no_cut_cache_means_unmeasured_not_zero( self ):
        betaconfig.detector_backend[self.backend]['profiles']['default'] = 'quick_cut'
        bu_config.invalidate_config_caches()
        path = self._clip( None )
        videos = self._caches( 'feedfacecafebeef', [] )
        os.remove( bu_cache.shot_cut_path_for( 'feedfacecafebeef', 9, 0.5 ) )
        bu_tuning.measure_replay( videos, { 'feedfacecafebeef': path }, self.backend,
                                  [ 'exposed_breast' ] )
        self.assertEqual( self.seen[-1]['profile_name'], 'quick_cut' )
        self.assertIsNone( self.seen[-1]['shot_cut_times'] )


if __name__ == '__main__':
    unittest.main()
