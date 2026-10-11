"""
test_suppression_ab.py - A/B two class_suppression rule sets on footage.

suppression_ab ranks every cached video by how differently the current
and proposed rules censor it, then renders the most-changed ones. These
tests pin the proposed set it derives (never adds a pair), that the
production pipeline sees the swapped rules and gets the old ones back,
the box matching behind "differs", the clip windows, the ranking, the
overlay colours and the renders, including a real one through ffmpeg.
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_detector as bu_detector
import betautils_tuning as bu_tuning
import betautils_video as bu_video


def _load_tool():
    path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                         'tools', 'tuning', 'suppression_ab.py' )
    spec = importlib.util.spec_from_file_location( '_suppression_ab_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


TOOL = _load_tool()


def _box( start, end, x=0, y=0, w=100, h=100, label='exposed_breast', track=1 ):
    return { 'start': start, 'end': end, 't': start, 'x': x, 'y': y, 'w': w, 'h': h,
             'label': label, 'score': 0.8, 'track_id': track,
             'censor_style': { 'type': 'blur', 'method': 'gaussian', 'strength': 10 },
             'censor_shape': 'box' }


class _Logger:
    def __init__( self ):
        self.lines = []

    def _note( self, message, *args ):
        self.lines.append( message )

    trace = debug = info = warning = error = _note


class TestProposedRules( unittest.TestCase ):

    CONFIGURED = {
        'exposed_breast': [ { 'suppressed_by': 'face_femme', 'margin': 0.0, 'min_iou': 0.3 },
                            { 'suppressed_by': 'covered_breast', 'margin': 0.0, 'min_iou': 0.3 },
                            { 'suppressed_by': 'belly', 'margin': 0.0, 'min_iou': 0.3 } ],
        'exposed_vulva': { 'suppressed_by': 'exposed_anus', 'margin': 0.1, 'min_iou': 0.2 },
    }

    def test_retunes_drops_and_keeps_unmeasured( self ):
        rows = [
            { 'pair': 'exposed_breast<-covered_breast', 'suggested_min_iou': 0.425,
              'suggested_margin': 0.081 },
            { 'pair': 'exposed_breast<-face_femme', 'suggested_min_iou': None,
              'suggested_margin': None },
            { 'pair': 'exposed_breast<-exposed_buttocks', 'suggested_min_iou': 0.175,
              'suggested_margin': 0.141 },
        ]
        proposed = TOOL.proposed_rules( self.CONFIGURED, rows )
        self.assertEqual( proposed['exposed_breast'], [
            { 'suppressed_by': 'covered_breast', 'margin': 0.081, 'min_iou': 0.425 },
            # belly: no row, too few overlaps to judge, kept as configured
            { 'suppressed_by': 'belly', 'margin': 0.0, 'min_iou': 0.3 } ] )
        # the dict-shaped rule is read too, and kept unmeasured
        self.assertEqual( proposed['exposed_vulva'],
                          [ { 'suppressed_by': 'exposed_anus', 'margin': 0.1, 'min_iou': 0.2 } ] )

    def test_never_adds_a_pair_that_is_not_configured( self ):
        rows = [ { 'pair': 'exposed_breast<-exposed_buttocks', 'suggested_min_iou': 0.175,
                   'suggested_margin': 0.141 } ]
        proposed = TOOL.proposed_rules( { 'exposed_breast': [] }, rows )
        self.assertEqual( proposed, {} )

    def test_a_label_whose_every_pair_is_dropped_disappears( self ):
        configured = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'margin': 0.0,
                                             'min_iou': 0.3 } ] }
        rows = [ { 'pair': 'exposed_breast<-face_femme', 'suggested_min_iou': None,
                   'suggested_margin': None } ]
        self.assertEqual( TOOL.proposed_rules( configured, rows ), {} )


class TestRuleChanges( unittest.TestCase ):

    def test_each_pair_is_named_with_what_happens_to_it( self ):
        current = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'margin': 0.1, 'min_iou': 0.01 },
                                        { 'suppressed_by': 'covered_breast', 'margin': 0.05, 'min_iou': 0.03 },
                                        { 'suppressed_by': 'belly', 'margin': 0.0, 'min_iou': 0.3 },
                                        { 'suppressed_by': 'buttocks', 'margin': 0.1, 'min_iou': 0.2 } ] }
        proposed = { 'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.081, 'min_iou': 0.425 },
                                         { 'suppressed_by': 'belly', 'margin': 0.0, 'min_iou': 0.3 },
                                         { 'suppressed_by': 'buttocks', 'margin': 0.14, 'min_iou': 0.2 } ],
                     'exposed_vulva': { 'suppressed_by': 'penis', 'margin': 0.2, 'min_iou': 0.5 } }
        self.assertEqual( TOOL.rule_changes( current, proposed ), [
            'exposed_breast<-belly: unchanged',
            'exposed_breast<-buttocks: retuned iou 0.2 -> 0.2, margin 0.1 -> 0.14',
            'exposed_breast<-covered_breast: retuned iou 0.03 -> 0.425, margin 0.05 -> 0.081',
            'exposed_breast<-face_femme: dropped',
            'exposed_vulva<-penis: added (iou 0.5, margin 0.2)' ] )


class TestRulesInEffect( unittest.TestCase ):

    def test_the_pipeline_reads_the_swapped_rules_and_gets_the_old_ones_back( self ):
        backend = 'nudenet_v3'
        before = bu_detector.get_class_suppression( backend )
        swapped = { 'exposed_breast': [ { 'suppressed_by': 'belly', 'margin': 0.5, 'min_iou': 0.9 } ] }
        with bu_tuning.class_suppression_in_effect( backend, swapped ):
            self.assertEqual( bu_detector.get_class_suppression( backend ), swapped )
        self.assertEqual( bu_detector.get_class_suppression( backend ), before )

    def test_restored_after_an_exception_and_absent_stays_absent( self ):
        backend = '_ab_test_backend'
        self.assertNotIn( backend, betaconfig.detector_backend )
        try:
            with self.assertRaises( RuntimeError ):
                with bu_tuning.class_suppression_in_effect( backend, { 'x': [] } ):
                    raise RuntimeError( 'boom' )
            self.assertNotIn( 'class_suppression', betaconfig.detector_backend[backend] )
        finally:
            betaconfig.detector_backend.pop( backend, None )

    def test_caller_mutation_does_not_leak_into_the_swap( self ):
        backend = 'nudenet_v3'
        rules = { 'exposed_breast': [ { 'suppressed_by': 'belly', 'margin': 0.5, 'min_iou': 0.9 } ] }
        with bu_tuning.class_suppression_in_effect( backend, rules ):
            rules['exposed_breast'].clear()
            self.assertEqual( len( bu_detector.get_class_suppression( backend )['exposed_breast'] ), 1 )


class TestDiffBoxes( unittest.TestCase ):

    def test_matches_need_label_time_and_overlap( self ):
        current = [ _box( 0.0, 1.0 ), _box( 2.0, 3.0 ), _box( 4.0, 5.0, label='exposed_vulva' ),
                    _box( 6.0, 7.0 ) ]
        proposed = [ _box( 0.5, 1.5 ),                         # same spot, overlapping time
                     _box( 3.5, 3.9 ),                         # same spot, after the box ended
                     _box( 4.0, 5.0 ),                         # same time, other label
                     _box( 6.0, 7.0, x=500 ) ]                 # same time, elsewhere
        both, current_only, proposed_only = TOOL.diff_boxes( current, proposed )
        self.assertEqual( both, [ proposed[0] ] )
        self.assertEqual( current_only, current[1:] )
        self.assertEqual( proposed_only, proposed[1:] )

    def test_another_label_on_the_same_spot_is_not_a_match( self ):
        both, current_only, proposed_only = TOOL.diff_boxes(
            [ _box( 0.0, 1.0, label='exposed_vulva' ) ], [ _box( 0.0, 1.0 ) ] )
        self.assertEqual( ( len( both ), len( current_only ), len( proposed_only ) ), ( 0, 1, 1 ) )

    def test_a_box_that_ended_earlier_is_not_a_match( self ):
        # The long box elsewhere widens the look-back to the earlier box.
        current = [ _box( 0.0, 50.0, x=900 ), _box( 2.0, 3.0 ) ]
        proposed = [ _box( 3.5, 3.9 ) ]
        both, current_only, proposed_only = TOOL.diff_boxes( current, proposed )
        self.assertEqual( both, [] )
        self.assertEqual( proposed_only, proposed )

    def test_a_spot_censored_twice_is_still_censored_under_both( self ):
        current = [ _box( 0.0, 1.0 ) ]
        proposed = [ _box( 0.0, 1.0 ), _box( 0.0, 1.0 ) ]
        both, current_only, proposed_only = TOOL.diff_boxes( current, proposed )
        self.assertEqual( len( both ), 2 )
        self.assertEqual( ( current_only, proposed_only ), ( [], [] ) )

    def test_identical_tracks_of_overlapping_samples_do_not_differ( self ):
        # Each sample's box overlaps its neighbours in time. Greedy pairing
        # took the next sample's box, leaving one spurious box per track on
        # each side: 9% of an identical render reported as different.
        track = [ _box( 10.0 + step / 9.0 - 0.15, 10.0 + step / 9.0 + 0.15, x=100 + step )
                  for step in range( 12 ) ]
        both, current_only, proposed_only = TOOL.diff_boxes( track, [ dict( box ) for box in track ] )
        self.assertEqual( len( both ), 12 )
        self.assertEqual( ( current_only, proposed_only ), ( [], [] ) )

    def test_boxes_in_one_time_bucket_must_still_overlap_in_time( self ):
        both, current_only, proposed_only = TOOL.diff_boxes( [ _box( 0.0, 0.1 ) ], [ _box( 0.3, 0.4 ) ] )
        self.assertEqual( ( len( both ), len( current_only ), len( proposed_only ) ), ( 0, 1, 1 ) )

    def test_a_long_box_is_found_from_any_bucket_it_spans( self ):
        current = [ _box( 0.0, 30.0 ) ]
        proposed = [ _box( 17.2, 17.3 ) ]
        both, current_only, _proposed_only = TOOL.diff_boxes( current, proposed )
        self.assertEqual( both, proposed )
        self.assertEqual( current_only, [] )

    def test_the_iou_floor( self ):
        # 100x100 against 100x100 shifted 60px: IoU 40/160 = 0.25
        current = [ _box( 0.0, 1.0 ) ]
        proposed = [ _box( 0.0, 1.0, x=60 ) ]
        self.assertEqual( len( TOOL.diff_boxes( current, proposed, min_iou=0.2 )[0] ), 1 )
        self.assertEqual( len( TOOL.diff_boxes( current, proposed, min_iou=0.3 )[0] ), 0 )

    def test_a_long_box_earlier_in_the_list_is_found( self ):
        current = [ _box( 0.0, 100.0 ), _box( 49.0, 50.0, x=900 ) ]
        proposed = [ _box( 50.0, 50.1 ) ]
        both, current_only, _proposed_only = TOOL.diff_boxes( current, proposed )
        self.assertEqual( both, proposed )
        self.assertEqual( current_only, [ current[1] ] )

    def test_box_seconds( self ):
        self.assertAlmostEqual( TOOL.box_seconds( [ _box( 0.0, 1.5 ), _box( 2.0, 2.25 ) ] ), 1.75 )
        self.assertEqual( TOOL.box_seconds( [] ), 0.0 )


class TestPickWindows( unittest.TestCase ):

    def test_the_densest_windows_first_never_overlapping( self ):
        events = [ _box( 10.0, 12.0 ), _box( 11.0, 13.0 ), _box( 50.0, 50.5 ), _box( 90.0, 99.0 ) ]
        windows = TOOL.pick_windows( events, 120.0, 10.0, 2 )
        self.assertEqual( len( windows ), 2 )
        self.assertTrue( any( start <= 94.5 < end for start, end in windows ) )
        self.assertTrue( any( start <= 11.5 < end for start, end in windows ) )
        for ( a_start, a_end ), ( b_start, b_end ) in zip( windows, windows[1:] ):
            self.assertLessEqual( a_end, b_start )

    def test_a_second_window_never_overlaps_the_first( self ):
        events = [ _box( 9.0, 11.0 ), _box( 11.0, 13.0 ), _box( 50.0, 50.5 ) ]
        windows = TOOL.pick_windows( events, 120.0, 10.0, 2 )
        self.assertEqual( len( windows ), 2 )
        self.assertTrue( any( start <= 50.25 < end for start, end in windows ) )

    def test_events_with_no_duration_choose_nothing( self ):
        self.assertEqual( TOOL.pick_windows( [ _box( 5.0, 5.0 ) ], 100.0, 10.0, 2 ), [] )

    def test_windows_stay_inside_the_video( self ):
        windows = TOOL.pick_windows( [ _box( 0.0, 0.5 ), _box( 119.5, 120.0 ) ], 120.0, 15.0, 2 )
        self.assertEqual( windows, [ ( 0.0, 15.0 ), ( 105.0, 120.0 ) ] )

    def test_short_video_and_nothing_to_show( self ):
        self.assertEqual( TOOL.pick_windows( [ _box( 1.0, 2.0 ) ], 5.0, 15.0, 3 ), [ ( 0.0, 5.0 ) ] )
        self.assertEqual( TOOL.pick_windows( [], 100.0, 15.0, 3 ), [] )
        self.assertEqual( TOOL.pick_windows( [ _box( 1.0, 2.0 ) ], 0.0, 15.0, 3 ), [] )


class TestRanking( unittest.TestCase ):

    def setUp( self ):
        self.renders = {
            # video a: proposed adds 3 box-seconds; b: identical; c: removes 10
            ( 'a', 'cur' ): [ _box( 0.0, 1.0 ) ],
            ( 'a', 'new' ): [ _box( 0.0, 1.0 ), _box( 5.0, 8.0, x=400 ) ],
            ( 'b', 'cur' ): [ _box( 0.0, 1.0 ) ],
            ( 'b', 'new' ): [ _box( 0.0, 1.0 ) ],
            ( 'c', 'cur' ): [ _box( 0.0, 10.0 ), _box( 20.0, 30.0, x=400 ) ],
            ( 'c', 'new' ): [ _box( 0.0, 10.0 ) ],
        }

    def fake_replay( self, video_path, cache_paths, backend_name, rules, tracking ):
        return self.renders[ ( os.path.basename( video_path )[0], rules['which'] ) ]

    def rank( self, forget=None ):
        context = { 'shot_cut_times': [], 'profile_name': 'scene', 'sample_fps': 9, 'signals': None }
        raw = [ { 'class_id': 'exposed_breast', 'score': 0.9, 't': 6.0,
                  'x': 400, 'y': 0, 'w': 100, 'h': 100 } ]
        with mock.patch.object( bu_tuning, 'replay_tracking_context', return_value=context ), \
             mock.patch.object( bu_tuning, 'raw_boxes_for_reading', return_value=raw ), \
             mock.patch.object( bu_tuning, 'forget_raw_boxes', forget or ( lambda paths: None ) ):
            return TOOL.rank_videos(
                { 'ha': [ 'a.gz' ], 'hb': [ 'b.gz' ], 'hc': [ 'c.gz' ], 'hd': [ 'd.gz' ] },
                { 'ha': '/v/a.mp4', 'hb': '/v/b.mp4', 'hc': '/v/c.mp4' },
                'nudenet_v3', { 'which': 'cur' }, { 'which': 'new' }, { 'exposed_breast' },
                _Logger(), clip_seconds=5.0, clips=2, replay=self.fake_replay,
                duration_of=lambda path: 60.0 )

    def test_most_changed_first_either_direction( self ):
        rows = self.rank()
        self.assertEqual( [ row['name'] for row in rows ], [ 'c.mp4', 'a.mp4', 'b.mp4' ] )
        self.assertEqual( rows[0]['current_only_seconds'], 10.0 )
        self.assertEqual( rows[1]['proposed_only_seconds'], 3.0 )
        self.assertEqual( rows[2]['differing_seconds'], 0.0 )
        self.assertEqual( rows[2]['windows'], [] )

    def test_rows_carry_share_restored_labels_and_windows( self ):
        rows = { row['name']: row for row in self.rank() }
        self.assertAlmostEqual( rows['c.mp4']['differing_share'], 0.5 )
        self.assertAlmostEqual( rows['a.mp4']['differing_share'], 3.0, msg="added seconds count too" )
        self.assertEqual( rows['c.mp4']['current_only_by_label'], { 'exposed_breast': 10.0 } )
        # the confident detection at t=6 x=400 is bare now and censored by a's proposed box
        self.assertEqual( rows['a.mp4']['confident_restored'], 1 )
        self.assertEqual( rows['c.mp4']['confident_restored'], 0 )
        self.assertEqual( rows['a.mp4']['windows'], [ ( 4.0, 9.0 ) ] )

    def test_each_video_cache_is_released_after_its_replay( self ):
        released = []
        self.rank( forget=released.append )
        self.assertEqual( sorted( released ), [ [ 'a.gz' ], [ 'b.gz' ], [ 'c.gz' ] ] )


class TestForgetRawBoxes( unittest.TestCase ):

    def test_only_the_named_paths_are_dropped( self ):
        saved = dict( bu_tuning._RAW_BOX_CACHE )
        try:
            bu_tuning._RAW_BOX_CACHE.clear()
            bu_tuning._RAW_BOX_CACHE.update( { ( 'a.gz', 1, 1 ): [], ( 'a.gz', 2, 2 ): [],
                                               ( 'b.gz', 1, 1 ): [] } )
            bu_tuning.forget_raw_boxes( [ 'a.gz' ] )
            self.assertEqual( list( bu_tuning._RAW_BOX_CACHE ), [ ( 'b.gz', 1, 1 ) ] )
        finally:
            bu_tuning._RAW_BOX_CACHE.clear()
            bu_tuning._RAW_BOX_CACHE.update( saved )


class TestVideosAcrossSampleRates( unittest.TestCase ):
    """Quick-cut videos are cached at their profile's rate and must be found too."""

    def setUp( self ):
        import betaconst
        self.betaconst = betaconst
        self.dir = tempfile.TemporaryDirectory()
        self.saved = ( bu_cache.VID_HASH_DIR, betaconst.run_key_dir )
        bu_cache.VID_HASH_DIR = os.path.join( self.dir.name, 'vid_hashes' )
        betaconst.run_key_dir = os.path.join( self.dir.name, 'run_keys' )
        os.makedirs( bu_cache.VID_HASH_DIR )
        os.makedirs( betaconst.run_key_dir )

    def tearDown( self ):
        bu_cache.VID_HASH_DIR, self.betaconst.run_key_dir = self.saved
        self.dir.cleanup()

    def write( self, file_hash, fps, key, variant='640m', size=640, age=0 ):
        import gzip, json
        stem = bu_cache.cache_name_stem( 'nudenet_v3', size, fps, 0.2 )
        path = os.path.join( bu_cache.VID_HASH_DIR, '%s%s%s.gz'%( file_hash, stem, key ) )
        with gzip.open( path, 'wt', encoding='UTF-8' ) as handle:
            json.dump( [], handle )
        os.utime( path, ( 1000000 - age, 1000000 - age ) )
        identity = { 'backend': 'nudenet_v3', 'size': size,
                     'backend_tunables': { 'model_variant': variant } if variant else {} }
        with open( os.path.join( self.betaconst.run_key_dir, 'detection-%s.json'%( key, ) ), 'w' ) as handle:
            json.dump( { 'kind': 'detection', 'key': key, 'identity': identity }, handle )
        return path

    def base( self ):
        return [ config for config in bu_cache.discover_cached_configurations( fps=9, min_prob=0.2 )
                 if config.variant == '640m' ][0]

    def test_videos_at_every_rate_of_the_same_model( self ):
        self.write( 'aaaa', 9, 'k9' )
        quick = self.write( 'bbbb', 18, 'k18' )
        self.write( 'cccc', 18, 'k18other', variant='320n', size=320 )
        self.write( 'dddd', 18, 'k18small', variant='640m', size=320 )      # same variant, other size
        self.write( 'eeee', 18, 'k18s', variant='640s', size=640 )          # same size, other variant
        videos, rates = bu_cache.videos_across_sample_rates( self.base(), 0.2, rates={ 9, 18 } )
        self.assertEqual( sorted( videos ), [ 'aaaa', 'bbbb' ] )
        self.assertEqual( videos['bbbb'], [ quick ] )
        self.assertEqual( rates, [ 9, 18 ] )

    def test_a_video_at_two_rates_keeps_its_newest_caches( self ):
        self.write( 'aaaa', 9, 'k9', age=100 )
        newer = self.write( 'aaaa', 18, 'k18', age=0 )
        videos, _rates = bu_cache.videos_across_sample_rates( self.base(), 0.2, rates={ 9, 18 } )
        self.assertEqual( videos['aaaa'], [ newer ] )

    def test_without_a_variant_two_models_at_a_rate_are_not_blended( self ):
        self.write( 'aaaa', 9, 'k9', variant=None )
        self.write( 'bbbb', 18, 'k18a', variant=None )
        self.write( 'cccc', 18, 'k18b', variant=None )
        base = bu_cache.discover_cached_configurations( fps=9, min_prob=0.2 )[0]
        videos, rates = bu_cache.videos_across_sample_rates( base, 0.2, rates={ 9, 18 } )
        self.assertEqual( sorted( videos ), [ 'aaaa' ] )
        self.assertEqual( rates, [ 9 ] )

    def test_default_rates_come_from_the_profiles( self ):
        self.write( 'aaaa', 9, 'k9' )
        self.write( 'bbbb', 18, 'k18' )
        with mock.patch.object( bu_cache.bu_detector, 'profile_sample_rates', return_value={ 9, 18 } ):
            videos, rates = bu_cache.videos_across_sample_rates( self.base(), 0.2 )
        self.assertEqual( sorted( videos ), [ 'aaaa', 'bbbb' ] )
        self.assertEqual( rates, [ 9, 18 ] )
        with mock.patch.object( bu_cache.bu_detector, 'profile_sample_rates', return_value={ 9 } ):
            videos, _rates = bu_cache.videos_across_sample_rates( self.base(), 0.2 )
        self.assertEqual( list( videos ), [ 'aaaa' ] )


class TestRawCacheReaders( unittest.TestCase ):

    def test_reading_shares_the_parse_and_replaying_copies_it( self ):
        import gzip, json
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join( tmp, 'c.gz' )
            with gzip.open( path, 'wt' ) as handle:
                json.dump( [ { 'class_id': 'exposed_breast', 't': 1.0 } ], handle )
            try:
                shared = bu_tuning.raw_boxes_for_reading( [ path ] )
                again = bu_tuning.raw_boxes_for_reading( [ path, os.path.join( tmp, 'missing.gz' ) ] )
                fresh = bu_tuning._read_raw_boxes( [ path ] )
                self.assertIs( shared[0], again[0] )
                self.assertIsNot( fresh[0], shared[0] )
                self.assertEqual( fresh, shared )
            finally:
                bu_tuning.forget_raw_boxes( [ path ] )


class TestBenchLogger( unittest.TestCase ):

    def test_the_bench_logger_can_trace( self ):
        with tempfile.TemporaryDirectory() as tmp:
            logger = TOOL._load_betabench().build_logger( tmp, 'trace', 'error' )
            logger.trace( "render chunk %d command: %s", 3, 'ffmpeg' )
            for handler in logger.handlers:
                handler.flush()
            with open( os.path.join( tmp, 'betabench.log' ) ) as handle:
                self.assertIn( 'render chunk 3 command: ffmpeg', handle.read() )
            for handler in list( logger.handlers ):
                handler.close()
                logger.removeHandler( handler )


class TestOverlay( unittest.TestCase ):

    def test_one_outline_colour_per_group_sorted_by_start( self ):
        both, current_only, proposed_only = [ _box( 3.0, 4.0 ) ], [ _box( 1.0, 2.0 ) ], [ _box( 2.0, 3.0 ) ]
        styled = TOOL.overlay_boxes( both, current_only, proposed_only )
        self.assertEqual( [ box['start'] for box in styled ], [ 1.0, 2.0, 3.0 ] )
        self.assertEqual( [ box['censor_style']['color'] for box in styled ],
                          [ TOOL.COLOUR_CURRENT_ONLY, TOOL.COLOUR_PROPOSED_ONLY, TOOL.COLOUR_BOTH ] )
        self.assertTrue( all( box['censor_style']['type'] == 'debug' for box in styled ) )
        self.assertEqual( both[0]['censor_style']['type'], 'blur', "inputs are not mutated" )

    def test_the_three_colours_are_distinct( self ):
        self.assertEqual( len( { TOOL.COLOUR_BOTH, TOOL.COLOUR_CURRENT_ONLY,
                                 TOOL.COLOUR_PROPOSED_ONLY } ), 3 )

    def test_motion_interpolation_is_off_inside_and_restored( self ):
        saved = getattr( betaconfig, 'render_motion_interpolation', True )
        betaconfig.render_motion_interpolation = True
        try:
            with TOOL.motion_interpolation_off():
                self.assertFalse( betaconfig.render_motion_interpolation )
            self.assertTrue( betaconfig.render_motion_interpolation )
        finally:
            betaconfig.render_motion_interpolation = saved


class TestRenderPlan( unittest.TestCase ):
    """What render_video_comparison asks to have rendered, with renders faked."""

    def setUp( self ):
        self.tmpdir = tempfile.mkdtemp( prefix='betasuite-ab-' )
        self.calls = []

    def tearDown( self ):
        shutil.rmtree( self.tmpdir, ignore_errors=True )

    def fake_render( self, video, file_hash, boxes, path, start, length, logger ):
        self.calls.append( ( os.path.basename( path ), start, length, boxes ) )
        open( path, 'w' ).close()
        return path

    def comparison( self, stack_ok=True, **kwargs ):
        current = [ _box( 0.0, 1.0 ) ]
        proposed = [ _box( 0.0, 1.0 ), _box( 2.0, 3.0, x=400 ) ]
        result = { 'current_boxes': current, 'proposed_boxes': proposed, 'both': [ proposed[0] ],
                   'current_only': [], 'proposed_only': [ proposed[1] ],
                   'current_only_causes': [], 'proposed_only_causes': [ 'exposed_breast<-face_femme' ] }
        row = { 'file_hash': 'h', 'video': '/v/My Video.mp4', 'name': 'My Video.mp4',
                'cache_paths': [ 'c.gz' ], 'windows': [ ( 10.0, 25.0 ) ] }
        ranking = { 'backend': 'nudenet_v3', 'current_rules': {}, 'proposed_rules': {},
                    'labels': [ 'exposed_breast' ] }
        return TOOL.render_video_comparison(
            row, ranking, _Logger(), self.tmpdir, render=self.fake_render,
            stack=lambda left, right, out, logger: stack_ok,
            compare=lambda *args: result, **kwargs ), current, proposed

    def test_clip_renders_outlines_then_current_then_proposed( self ):
        produced, current, proposed = self.comparison()
        names = [ call[0] for call in self.calls ]
        self.assertEqual( names, [ 'My_Video_clip01_10s_diff.mp4', 'My_Video_clip01_10s_current.mp4',
                                   'My_Video_clip01_10s_proposed.mp4' ] )
        self.assertTrue( all( call[1:3] == ( 10.0, 15.0 ) for call in self.calls ) )
        self.assertIs( self.calls[1][3], current )
        self.assertIs( self.calls[2][3], proposed )
        self.assertEqual( [ os.path.basename( path ) for path in produced ],
                          [ 'My_Video_clip01_10s_diff.mp4', 'My_Video_clip01_10s_ab.mp4' ] )
        self.assertFalse( os.path.exists( os.path.join( self.tmpdir, names[1] ) ),
                          "the halves are removed once stacked" )

    def test_outlines_carry_the_rule_that_made_the_difference( self ):
        self.comparison( censored=False )
        outlines = self.calls[0][3]
        notes = { box['censor_style']['color']: box.get( 'debug_note' ) for box in outlines }
        self.assertEqual( notes[TOOL.COLOUR_PROPOSED_ONLY], '<-face_femme' )
        self.assertIsNone( notes[TOOL.COLOUR_BOTH] )

    def test_a_failed_stack_keeps_both_halves( self ):
        produced, _current, _proposed = self.comparison( stack_ok=False, overlay=False )
        self.assertEqual( [ os.path.basename( path ) for path in produced ],
                          [ 'My_Video_clip01_10s_current.mp4', 'My_Video_clip01_10s_proposed.mp4' ] )
        self.assertTrue( all( os.path.exists( path ) for path in produced ) )

    def test_full_renders_the_whole_video_under_each_set( self ):
        produced, current, proposed = self.comparison( overlay=False, censored=False, full=True )
        self.assertEqual( [ call[0:3] for call in self.calls ],
                          [ ( 'My_Video_full_current.mp4', None, None ),
                            ( 'My_Video_full_proposed.mp4', None, None ) ] )
        self.assertIs( self.calls[0][3], current )
        self.assertEqual( len( produced ), 2 )


@unittest.skipUnless( shutil.which( 'ffmpeg' ), "ffmpeg is needed for a real render" )
class TestRealRender( unittest.TestCase ):
    """A window of a synthetic clip through the production renderer, then stacked."""

    @classmethod
    def setUpClass( cls ):
        cls.tmpdir = tempfile.mkdtemp( prefix='betasuite-ab-render-' )
        cls.source = os.path.join( cls.tmpdir, 'source.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error',
              '-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=10:duration=4',
              '-f', 'lavfi', '-i', 'sine=frequency=440:duration=4',
              '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
              '-c:a', 'aac', '-shortest', cls.source ],
            check=True )

    @classmethod
    def tearDownClass( cls ):
        shutil.rmtree( cls.tmpdir, ignore_errors=True )

    def test_window_render_and_side_by_side( self ):
        # The tool's own logger: a stand-in with trace() hid that the real
        # one lacked it, and every chunk of the first real run failed.
        logger = TOOL._load_betabench().build_logger( self.tmpdir, 'debug', 'error' )
        styled = TOOL.overlay_boxes( [ _box( 0.0, 4.0, w=40, h=40 ) ], [], [] )
        diff = TOOL.render_window( self.source, 'abhash', styled,
                                   os.path.join( self.tmpdir, 'diff.mp4' ), 1.0, 2.0, logger )
        self.assertTrue( bu_video.probe_video_ok( diff ) )
        self.assertAlmostEqual( bu_video.probe_duration_seconds( diff ), 2.0, delta=0.3 )
        self.assertFalse( os.path.exists( diff + '.video.mkv' ), "the intermediate is cleaned up" )

        whole = TOOL.render_window( self.source, 'abhash', [ _box( 0.0, 4.0, w=40, h=40 ) ],
                                    os.path.join( self.tmpdir, 'whole.mp4' ), None, None, logger )
        self.assertAlmostEqual( bu_video.probe_duration_seconds( whole ), 4.0, delta=0.3 )

        combined = os.path.join( self.tmpdir, 'ab.mp4' )
        self.assertTrue( TOOL.side_by_side( diff, diff, combined, logger ) )
        width = subprocess.run(
            [ 'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width',
              '-of', 'csv=p=0', combined ], capture_output=True, text=True ).stdout.strip()
        self.assertEqual( width, '256' )
        self.assertTrue( bu_video.video_file_has_audio( combined ) )

    def test_side_by_side_failure_is_reported_not_raised( self ):
        logger = _Logger()
        missing = os.path.join( self.tmpdir, 'missing.mp4' )
        self.assertFalse( TOOL.side_by_side( missing, missing,
                                             os.path.join( self.tmpdir, 'x.mp4' ), logger ) )
        self.assertTrue( any( 'side-by-side failed' in line for line in logger.lines ) )


if __name__ == '__main__':
    unittest.main()
