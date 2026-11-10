"""
Analysis covers the last real run, every profile, and previews as one group.

Three gaps these pin:

  - The replaying analysers and auto_tune looked for caches at the
    configured video_censor_fps only. Changing it after a run hid the
    whole run from them, and quick_cut files (sampled at their profile's
    own rate) were never read at all.
  - A profile's own tracking values win over the backend's for its
    files, so auto_tune could only report "tune it in the profiles" and
    never measure them.
  - betabench treated every preview slice as its own configuration and
    printed a section for each.
"""

import ast
import copy
import gzip
import importlib.util
import json
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_tuning as bu_tuning

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load( relative, name ):
    spec = importlib.util.spec_from_file_location( name, os.path.join( REPO, relative ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class CacheDir( unittest.TestCase ):
    """A private vid_hashes, run_keys and shot_cuts directory."""

    MIN_PROB = 0.12

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-rates-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        saved = ( bu_cache.VID_HASH_DIR, betaconst.run_key_dir, betaconst.shot_cut_dir )

        def restore():
            bu_cache.VID_HASH_DIR, betaconst.run_key_dir, betaconst.shot_cut_dir = saved
        self.addCleanup( restore )
        bu_cache.VID_HASH_DIR = os.path.join( self.dir, 'vid_hashes' )
        betaconst.run_key_dir = os.path.join( self.dir, 'run_keys' )
        betaconst.shot_cut_dir = os.path.join( self.dir, 'shot_cuts' )
        for path in ( bu_cache.VID_HASH_DIR, betaconst.run_key_dir, betaconst.shot_cut_dir ):
            os.makedirs( path )

    def write( self, file_hash, fps, key, variant='640m', size=640, age=0, preview='',
               min_prob=None, box_version=None ):
        stem = '-%s-%s-%d-%g-%.3f-d'%(
            'nudenet_v3', betaconst.picture_saved_box_version if box_version is None else box_version,
            size, fps, self.MIN_PROB if min_prob is None else min_prob )
        path = os.path.join( bu_cache.VID_HASH_DIR, '%s%s%s%s.gz'%( file_hash, stem, key, preview ) )
        with gzip.open( path, 'wt', encoding='UTF-8' ) as handle:
            json.dump( [], handle )
        os.utime( path, ( 1000000 - age, 1000000 - age ) )
        identity = { 'backend': 'nudenet_v3', 'size': size,
                     'backend_tunables': { 'model_variant': variant } if variant else {} }
        with open( os.path.join( betaconst.run_key_dir, 'detection-%s.json'%( key, ) ), 'w' ) as handle:
            json.dump( { 'kind': 'detection', 'key': key, 'identity': identity }, handle )
        return path

    def configurations( self, **kwargs ):
        kwargs.setdefault( 'min_prob', self.MIN_PROB )
        kwargs.setdefault( 'backend_names', [ 'nudenet_v3' ] )
        return bu_cache.configurations_to_analyse( None, **kwargs )


class TestLastRealRunIsFound( CacheDir ):

    def test_a_run_at_another_rate_is_one_configuration_with_every_rate( self ):
        self.write( 'aaaa', 9, 'k9' )
        self.write( 'bbbb', 18, 'k18' )          # a quick_cut file
        self.write( 'cccc', 18, 'k18new' )       # same profile, after a detection change
        with mock.patch.object( betaconfig, 'video_censor_fps', 10 ):
            found = self.configurations()
        self.assertEqual( [ config.label for config in found ], [ 'nudenet_v3/640m @640' ] )
        self.assertEqual( found[0].sample_rates, [ 9.0, 18.0 ] )
        self.assertEqual( found[0].file_hashes, [ 'aaaa', 'bbbb', 'cccc' ] )
        self.assertEqual( found[0].detection_keys, [ 'k18', 'k18new', 'k9' ] )
        self.assertIsNone( found[0].detection_key, "several keys: no single one to name" )

    def test_a_named_rate_reads_only_that_rate( self ):
        self.write( 'aaaa', 9, 'k9' )
        self.write( 'bbbb', 18, 'k18' )
        found = self.configurations( fps=9 )
        self.assertEqual( len( found ), 1 )
        self.assertEqual( found[0].file_hashes, [ 'aaaa' ] )
        videos, preview, rates = bu_cache.videos_for_configuration( found[0], 9, self.MIN_PROB )
        self.assertEqual( ( sorted( videos ), preview, rates ), ( [ 'aaaa' ], {}, [ 9.0 ] ) )

    def test_the_replay_set_is_the_newest_run_of_each_video( self ):
        self.write( 'aaaa', 9, 'k9', age=100 )
        newer = self.write( 'aaaa', 10, 'k10', age=0 )
        self.write( 'bbbb', 18, 'k18' )
        config = self.configurations()[0]
        videos, preview, rates = bu_cache.videos_for_configuration( config, None, self.MIN_PROB )
        self.assertEqual( videos['aaaa'], [ newer ] )
        self.assertEqual( sorted( videos ), [ 'aaaa', 'bbbb' ] )
        self.assertEqual( ( preview, rates ), ( {}, [ 10.0, 18.0 ] ) )

    def test_two_models_without_a_recorded_variant_stay_apart( self ):
        self.write( 'aaaa', 9, 'ka', variant=None )
        self.write( 'bbbb', 18, 'kb', variant=None )
        found = self.configurations()
        self.assertEqual( len( found ), 2, "nothing says ka and kb are one model" )
        videos, _rates = bu_cache.videos_across_sample_rates( found[0], self.MIN_PROB )
        self.assertEqual( len( videos ), 1 )

    def test_other_floors_box_versions_and_variants_are_not_read( self ):
        self.write( 'aaaa', 9, 'k9' )
        self.write( 'bbbb', 9, 'kfloor', min_prob=0.2 )
        self.write( 'cccc', 9, 'kbox', box_version=betaconst.picture_saved_box_version + 1 )
        self.write( 'dddd', 9, 'k640s', variant='640s' )
        config = [ config for config in self.configurations() if config.variant == '640m' ][0]
        videos, _rates = bu_cache.videos_across_sample_rates( config, self.MIN_PROB )
        self.assertEqual( list( videos ), [ 'aaaa' ] )
        self.assertEqual( bu_cache.sample_rates_on_disk( [ 'nudenet_v3' ], self.MIN_PROB ), [ 9.0 ] )

    def test_preview_rates_count_only_when_previews_are_asked_for( self ):
        self.write( 'aaaa', 9, 'k9' )
        self.write( 'bbbb', 10, 'k10', preview='-preview@1.0s+20.0s' )
        self.assertEqual( bu_cache.sample_rates_on_disk( [ 'nudenet_v3' ], self.MIN_PROB ), [ 9.0 ] )
        self.assertEqual( bu_cache.sample_rates_on_disk( [ 'nudenet_v3' ], self.MIN_PROB, True ), [ 9.0, 10.0 ] )

    def test_include_preview_keeps_the_per_rate_lookup_with_a_slice_fallback( self ):
        real = self.write( 'aaaa', 10, 'k10' )
        self.write( 'bbbb', 10, 'k10', preview='-preview@1.0s+20.0s' )
        self.write( 'cccc', 9, 'k9' )
        config = bu_cache.CachedConfiguration( 'nudenet_v3', [ 640 ], variant='640m' )
        with mock.patch.object( betaconfig, 'video_censor_fps', 10 ):
            videos, preview, rates = bu_cache.videos_for_configuration(
                config, None, self.MIN_PROB, include_preview=True )
        self.assertEqual( videos['aaaa'], [ real ] )
        self.assertEqual( sorted( videos ), [ 'aaaa', 'bbbb' ], "cccc is at another rate" )
        self.assertEqual( list( preview ), [ 'bbbb' ] )
        self.assertEqual( rates, [ 10.0 ] )

    def test_an_explicit_shape_reads_every_key( self ):
        # --picture-sizes names a shape, not a run: newest of anything there.
        self.write( 'aaaa', 9, 'ka', variant=None, age=50 )
        newest = self.write( 'aaaa', 18, 'kb', variant=None, age=0 )
        shape = bu_cache.CachedConfiguration( 'nudenet_v3', [ 640 ] )
        videos, rates = bu_cache.videos_across_sample_rates( shape, self.MIN_PROB )
        self.assertEqual( ( videos, rates ), ( { 'aaaa': [ newest ] }, [ 18.0 ] ) )

    def test_every_size_must_come_from_one_run( self ):
        self.write( 'aaaa', 9, 'k9', size=320 )
        self.write( 'aaaa', 18, 'k18', size=640 )
        both = bu_cache.CachedConfiguration( 'nudenet_v3', [ 320, 640 ] )
        videos, _rates = bu_cache.videos_across_sample_rates( both, self.MIN_PROB )
        self.assertEqual( videos, {}, "a 320 from one run and a 640 from another are not a set" )

    def test_nothing_on_disk_still_answers_with_the_configured_shape( self ):
        found = self.configurations()
        self.assertEqual( [ config.backend_name for config in found ], [ 'nudenet_v3' ] )


class TestPreviewsAreOneGroup( CacheDir ):

    def test_every_slice_lands_in_one_preview_configuration( self ):
        self.write( 'zzzz', 10, 'k10' )        # a real run: never read by the preview group
        self.write( 'aaaa', 10, 'k10', preview='-preview@100.0s+20.0s' )
        self.write( 'aaaa', 10, 'k10', preview='-preview@200.0s+20.0s', age=10 )
        self.write( 'bbbb', 9, 'k9', preview='-preview@50.0s+20.0s' )
        found = [ config for config in self.configurations( include_preview=True )
                  if config.preview_suffix ]
        self.assertEqual( [ config.label for config in found ], [ 'nudenet_v3/640m @640 [PREVIEW]' ] )
        self.assertEqual( found[0].preview_suffix, bu_cache.PREVIEW_ANY )
        videos, _rates = bu_cache.videos_across_sample_rates( found[0], self.MIN_PROB )
        self.assertEqual( sorted( videos ), [ 'aaaa', 'bbbb' ] )
        self.assertIn( 'preview@100.0s', videos['aaaa'][0], "one slice per video: the newest" )

    def test_fallback_drops_previews_where_a_real_run_exists( self ):
        self.write( 'aaaa', 9, 'k9' )
        self.write( 'bbbb', 10, 'k10', preview='-preview@100.0s+20.0s' )
        self.write( 'cccc', 9, 'k320', variant='320n', size=320, preview='-preview@1.0s+20.0s' )
        labels = [ config.label for config in self.configurations( include_preview='fallback' ) ]
        self.assertEqual( labels, [ 'nudenet_v3/320n @320 [PREVIEW]', 'nudenet_v3/640m @640' ] )

    def test_a_real_configuration_never_reads_a_slice( self ):
        self.write( 'aaaa', 9, 'k9', preview='-preview@1.0s+20.0s' )
        real = self.write( 'bbbb', 9, 'k9' )
        config = self.configurations()[0]
        videos, _rates = bu_cache.videos_across_sample_rates( config, self.MIN_PROB )
        self.assertEqual( videos, { 'bbbb': [ real ] } )

    def test_one_exact_slice_keeps_its_own_suffix( self ):
        self.write( 'aaaa', 9, 'k9', preview='-preview@1.0s+20.0s', age=0 )
        self.write( 'aaaa', 9, 'k9', preview='-preview@9.0s+20.0s', age=50 )
        exact = bu_cache.CachedConfiguration( 'nudenet_v3', [ 640 ], detection_key='k9',
                                              variant='640m', preview_suffix='-preview@9.0s+20.0s' )
        videos, _rates = bu_cache.videos_across_sample_rates( exact, self.MIN_PROB )
        self.assertIn( 'preview@9.0s', videos['aaaa'][0] )


class TestShotCutsFromAnEarlierRate( CacheDir ):

    @staticmethod
    def threshold():
        return getattr( betaconfig, 'shot_cut_threshold', 0.5 )

    def scan( self, fps, cuts, age=0, bound=None ):
        path = bu_cache.shot_cut_path_for( 'aaaa', fps, self.threshold(), preview_max_seconds=bound )
        bu_tuning.bu_hash.write_json( cuts, path )
        os.utime( path, ( 1000000 - age, 1000000 - age ) )
        return path

    def test_any_rate_takes_a_whole_file_scan_first_then_the_newest( self ):
        older = self.scan( 9, [ 1.0 ], age=100 )
        newer = self.scan( 12, [ 2.0 ], age=0 )
        bounded = self.scan( 9, [ 3.0 ], age=0, bound=20 )
        found = [ path for path, _parsed in bu_cache.shot_cut_caches_for( 'aaaa', None, self.threshold() ) ]
        self.assertEqual( found, [ newer, older, bounded ] )
        self.assertEqual( bu_cache.shot_cut_caches_for( 'aaaa', 10, self.threshold() ), [] )

    def test_a_replay_uses_the_cuts_its_run_scanned( self ):
        self.scan( 9, [ 4.0, 8.0 ] )
        detection = self.write( 'aaaa', 9, 'k9' )
        with mock.patch.object( betaconfig, 'video_censor_fps', 10 ), \
             mock.patch.object( bu_tuning.bu_video, 'probe_duration_seconds', return_value=12.0 ):
            context = bu_tuning.replay_tracking_context( 'aaaa', '/x/a.mp4', [ detection ], 'nudenet_v3' )
        self.assertEqual( context['shot_cut_times'], [ 4.0, 8.0 ] )
        self.assertEqual( context['sample_fps'], 9.0 )

    def test_the_configured_rate_wins_when_it_has_a_scan( self ):
        self.scan( 9, [ 4.0 ] )
        self.scan( 10, [ 5.0 ], age=500 )
        detection = self.write( 'aaaa', 10, 'k10' )
        with mock.patch.object( betaconfig, 'video_censor_fps', 10 ), \
             mock.patch.object( bu_tuning.bu_video, 'probe_duration_seconds', return_value=12.0 ):
            context = bu_tuning.replay_tracking_context( 'aaaa', '/x/a.mp4', [ detection ], 'nudenet_v3' )
        self.assertEqual( context['shot_cut_times'], [ 5.0 ] )


class TestMedianShotSeesMergedBursts( CacheDir ):

    def test_a_fade_counts_as_one_cut( self ):
        auto_tune = _load( 'tools/tuning/auto_tune.py', '_auto_tune_shots_under_test' )
        threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
        # Four 10-second shots, every cut but the first scanned as a fade:
        # a burst of cuts 0.1s apart.
        cuts = [ 10.0, 20.0, 20.1, 20.2, 30.0, 30.1, 30.2, 40.0 ]
        bu_tuning.bu_hash.write_json( cuts, bu_cache.shot_cut_path_for( 'aaaa', 9, threshold ) )
        entries = [ ( None, { 'videos': { 'aaaa': [] } } ) ]
        with mock.patch.object( betaconfig, 'video_censor_fps', 10 ), \
             mock.patch.object( betaconfig, 'shot_cut_burst_window', 0.35, create=True ), \
             mock.patch.object( betaconfig, 'shot_cut_burst_max_span', None, create=True ), \
             mock.patch.object( betaconfig, 'shot_cut_burst_keep', 'first', create=True ):
            median = auto_tune._median_shot_seconds_for( entries, mock.Mock() )
        self.assertGreater( median, 9.0, "raw slivers would put the median at 0.1s" )


class TestPerProfileHelpers( unittest.TestCase ):

    def test_merge_counts_adds_numbers_joins_lists_and_merges_dicts( self ):
        merged = bu_tuning.merge_counts( [
            { 'n': 1, 'l': [ 1 ], 'd': { 'x': { 'total': 2, 'risky': 1 } }, 'only_a': 5, 'f': 0.5 },
            { 'n': 2, 'l': [ 2, 3 ], 'd': { 'x': { 'total': 3, 'risky': 0 }, 'y': { 'total': 1 } },
              'f': 0.25 },
        ] )
        self.assertEqual( merged, { 'n': 3, 'l': [ 1, 2, 3 ], 'only_a': 5, 'f': 0.75,
                                    'd': { 'x': { 'total': 5, 'risky': 1 }, 'y': { 'total': 1 } } } )
        self.assertEqual( bu_tuning.merge_counts( [] ), {} )
        with self.assertRaises( TypeError ):
            bu_tuning.merge_counts( [ { 'a': 1 }, { 'a': 'text' } ] )

    def test_merge_counts_does_not_mutate_its_inputs( self ):
        first = { 'd': { 'x': 1 }, 'l': [ 1 ] }
        bu_tuning.merge_counts( [ first, { 'd': { 'x': 2 }, 'l': [ 2 ] } ] )
        self.assertEqual( first, { 'd': { 'x': 1 }, 'l': [ 1 ] } )

    def test_videos_group_by_the_profile_a_run_would_pick( self ):
        videos = { 'a': [ 'a.gz' ], 'b': [ 'b.gz' ], 'c': [ 'c.gz' ] }
        profile = { 'a': 'scene', 'b': 'quick_cut' }
        with mock.patch.object( bu_tuning, 'replay_tracking_context',
                                side_effect=lambda h, *a, **k: { 'profile_name': profile[h] } ):
            grouped = bu_tuning.videos_by_profile( videos, { 'a': '/a', 'b': '/b' }, 'nudenet_v3' )
        self.assertEqual( grouped, { 'scene': { 'a': [ 'a.gz' ] }, 'quick_cut': { 'b': [ 'b.gz' ] } },
                          "c has no source on disk, so no profile and no replay" )


class TestTrackBreaksPerProfile( unittest.TestCase ):

    def setUp( self ):
        self.module = _load( 'tools/analysis/analyze_track_breaks.py', '_track_breaks_under_test' )

    def test_pooled_rows_are_the_sum_and_missing_sources_still_count( self ):
        def combo( match, gap, videos, *rest ):
            return { 'match_dists_by_label': { 'x': [ 1.0 ] * len( videos ) },
                     'new_track_by_label': { 'x': len( videos ) },
                     'reason_counts_by_label': { 'x': { 'after-cut': 1 } },
                     'contention_by_label': { 'x': { 'total': 2, 'risky': 1 } },
                     'dims_missing': [] }
        videos = { 'a': [], 'b': [], 'c': [], 'z': [] }
        by_profile = { 'scene': { 'a': [], 'b': [] }, 'quick_cut': { 'c': [] } }
        with mock.patch.object( self.module, 'run_one_combo', side_effect=combo ) as run:
            pooled, per_profile = self.module.run_profiles( 1.0, 2.0, videos, by_profile, {}, {},
                                                            { 'x' }, 'nudenet_v3' )
        self.assertEqual( run.call_count, 2, "one replay per profile, no pooled re-run" )
        self.assertEqual( pooled['new_track_by_label'], { 'x': 3 } )
        self.assertEqual( pooled['contention_by_label'], { 'x': { 'total': 4, 'risky': 2 } } )
        self.assertEqual( pooled['dims_missing'], [ 'z' ] )
        self.assertEqual( sorted( per_profile ), [ 'quick_cut', 'scene' ] )

    def test_no_resolvable_video_falls_back_to_one_plain_run( self ):
        with mock.patch.object( self.module, 'run_one_combo', return_value={ 'dims_missing': [ 'a' ] } ) as run:
            pooled, per_profile = self.module.run_profiles( 1.0, 1.0, { 'a': [] }, {}, {}, {},
                                                            { 'x' }, 'nudenet_v3' )
        self.assertEqual( ( pooled, per_profile, run.call_count ), ( { 'dims_missing': [ 'a' ] }, {}, 1 ) )

    def test_a_single_profile_prints_no_breakdown( self ):
        result = { 'match_dists_by_label': { 'x': [ 1 ] }, 'new_track_by_label': { 'x': 1 },
                   'contention_by_label': {} }
        with mock.patch( 'builtins.print' ) as printed:
            self.module.print_profile_breakdown( { 'scene': result }, [ 'x' ] )
        printed.assert_not_called()
        with mock.patch( 'builtins.print' ) as printed:
            self.module.print_profile_breakdown( { 'scene': result, 'quick_cut': result }, [ 'x' ] )
        lines = [ call.args[0] for call in printed.call_args_list if call.args ]
        self.assertTrue( any( 'quick_cut' in line and 'new_track=1/2' in line for line in lines ), lines )


class TestProfileStages( unittest.TestCase ):

    def setUp( self ):
        self.auto_tune = _load( 'tools/tuning/auto_tune.py', '_auto_tune_profiles_under_test' )
        saved = copy.deepcopy( betaconfig.detector_backend )
        self.addCleanup( setattr, betaconfig, 'detector_backend', saved )
        self.addCleanup( bu_config.invalidate_config_caches )
        block = copy.deepcopy( saved )
        block['nudenet_v3'].setdefault( 'item_overrides', {} ).setdefault( 'exposed_breast', {} )[
            'track_max_gap'] = 0.17
        variants = block['nudenet_v3'].setdefault( 'profiles', {} ).setdefault( 'variants', {} )
        variants.setdefault( 'quick_cut', {} ).setdefault( 'item_overrides', {} )[
            'exposed_breast'] = { 'track_max_gap': 0.2 }
        variants['plain'] = { 'item_overrides': { 'exposed_breast': { 'time_safety': 0.1 } } }
        betaconfig.detector_backend = block
        bu_config.invalidate_config_caches()

    def stage( self, profile=None ):
        stage = self.auto_tune.TrackMaxGapStage( 'nudenet_v3', 'exposed_breast' )
        stage.profile_name = profile
        return stage

    def test_a_profile_stage_reads_and_sets_the_profile_value( self ):
        stage = self.stage( 'quick_cut' )
        self.assertEqual( stage.baseline_value(), 0.2 )
        self.assertEqual( stage.title, 'track_max_gap / nudenet_v3 / exposed_breast [quick_cut]' )
        saved = stage.snapshot()
        stage.apply( 0.4 )
        block = betaconfig.detector_backend['nudenet_v3']
        self.assertEqual( block['profiles']['variants']['quick_cut']['item_overrides'][
            'exposed_breast']['track_max_gap'], 0.4 )
        self.assertEqual( block['item_overrides']['exposed_breast']['track_max_gap'], 0.17,
                          "the backend value must not move" )
        stage.restore( saved )
        self.assertEqual( self.stage( 'quick_cut' ).baseline_value(), 0.2 )

    def test_a_profile_that_does_not_set_the_key_reads_the_backend_value( self ):
        self.assertEqual( self.stage( 'plain' ).baseline_value(), 0.17 )
        self.assertEqual( self.stage().baseline_value(), 0.17 )

    def test_restore_removes_a_profiles_block_that_apply_created( self ):
        del betaconfig.detector_backend['nudenet_v3']['profiles']
        stage = self.stage( 'quick_cut' )
        saved = stage.snapshot()
        stage.apply( 0.4 )
        stage.restore( saved )
        self.assertNotIn( 'profiles', betaconfig.detector_backend['nudenet_v3'] )

    def test_a_profile_stage_writes_into_the_profile( self ):
        writer = mock.Mock()
        self.stage( 'quick_cut' ).write( writer, 0.3 )
        writer.set_profile_item_override.assert_called_once_with(
            'nudenet_v3', 'quick_cut', 'exposed_breast', 'track_max_gap', 0.3 )
        writer.set_item_override.assert_not_called()
        self.stage().write( writer, 0.3 )
        writer.set_item_override.assert_called_once_with( 'nudenet_v3', 'exposed_breast', 'track_max_gap', 0.3 )

    def test_each_profile_is_measured_over_only_its_own_files( self ):
        out_dir = tempfile.mkdtemp( prefix='betasuite-autotune-' )
        self.addCleanup( shutil.rmtree, out_dir, True )
        config = types.SimpleNamespace( backend_name='nudenet_v3', picture_sizes=[ 640 ],
                                        label='nudenet_v3/640m @640', variant='640m' )
        seen = []

        def measure( stage, context, *rest ):
            seen.append( ( stage.profile_name, sorted( context['videos'] ),
                           context['configuration_label'] ) )
            return [ { 'value': stage.baseline_value(), 'label': 'baseline', 'verdict': 'baseline',
                       'coverage_loss': 0.0, 'metrics': {} } ]
        overrides = { 'scene': { 'track_max_gap': 3.0 }, 'quick_cut': { 'track_max_gap': 0.2 } }
        argv = [ 'auto_tune.py', '--stages', 'track_max_gap', '--labels', 'exposed_breast',
                 '--out-dir', out_dir, '--backends', 'nudenet_v3' ]
        with mock.patch( 'sys.argv', argv ), \
             mock.patch.object( self.auto_tune.bu_cache, 'configurations_to_analyse', return_value=[ config ] ), \
             mock.patch.object( self.auto_tune.bu_cache, 'videos_for_configuration',
                                return_value=( { 'h1': [ 'x' ], 'h2': [ 'y' ], 'h3': [ 'z' ] }, {}, [ 9.0, 18.0 ] ) ), \
             mock.patch.object( self.auto_tune.bu_cache, 'build_hash_to_video_path',
                                return_value={ 'h1': '/a', 'h2': '/b', 'h3': '/c' } ), \
             mock.patch.object( self.auto_tune.bu_tuning, 'replay_tracking_context',
                                side_effect=lambda h, *a, **k: { 'profile_name':
                                    { 'h1': 'scene', 'h2': 'quick_cut', 'h3': 'quick_cut' }[h] } ), \
             mock.patch.object( self.auto_tune.bu_tuning.bu_detector, 'get_profile_item_overrides',
                                side_effect=lambda profile, label, backend=None: overrides.get( profile, {} ) ), \
             mock.patch.object( self.auto_tune, 'measure_stage', side_effect=measure ), \
             mock.patch.object( self.auto_tune, 'print_stage_table' ), \
             mock.patch.object( self.auto_tune.TrackMaxGapStage, 'score_rows', create=True ), \
             mock.patch.object( self.auto_tune.TrackMaxGapStage, 'decide',
                                side_effect=lambda rows, *a: bu_tuning.Decision( 'x', None, False, 'kept', [] ) ), \
             mock.patch.object( self.auto_tune, '_median_shot_seconds_for', return_value=None ):
            self.auto_tune.main()
        self.assertEqual( seen, [
            ( 'quick_cut', [ 'h2', 'h3' ], 'nudenet_v3/640m @640 [quick_cut]' ),
            ( 'scene', [ 'h1' ], 'nudenet_v3/640m @640 [scene]' ) ] )

    def run_main_measuring( self, backend_value ):
        """auto_tune.main() over three files in two profiles; the stages it measured."""
        out_dir = tempfile.mkdtemp( prefix='betasuite-autotune-' )
        self.addCleanup( shutil.rmtree, out_dir, True )
        overrides = betaconfig.detector_backend['nudenet_v3']['item_overrides']['exposed_breast']
        if backend_value is None:
            overrides.pop( 'track_max_gap', None )
        else:
            overrides['track_max_gap'] = backend_value
        bu_config.invalidate_config_caches()
        config = types.SimpleNamespace( backend_name='nudenet_v3', picture_sizes=[ 640 ],
                                        label='nudenet_v3/640m @640', variant='640m' )
        seen = []

        def measure( stage, context, *rest ):
            seen.append( ( stage.profile_name, sorted( context['videos'] ),
                           context['configuration_label'] ) )
            return [ { 'value': stage.baseline_value(), 'label': 'baseline', 'verdict': 'baseline',
                       'coverage_loss': 0.0, 'metrics': {} } ]
        profile_values = { 'scene': { 'track_max_gap': 3.0 }, 'quick_cut': { 'track_max_gap': 0.2 } }
        argv = [ 'auto_tune.py', '--stages', 'track_max_gap', '--labels', 'exposed_breast',
                 '--out-dir', out_dir, '--backends', 'nudenet_v3' ]
        with mock.patch( 'sys.argv', argv ), \
             mock.patch.object( self.auto_tune.bu_cache, 'configurations_to_analyse', return_value=[ config ] ), \
             mock.patch.object( self.auto_tune.bu_cache, 'videos_for_configuration',
                                return_value=( { 'h1': [ 'x' ], 'h2': [ 'y' ], 'h3': [ 'z' ] }, {}, [ 9.0, 18.0 ] ) ), \
             mock.patch.object( self.auto_tune.bu_cache, 'build_hash_to_video_path',
                                return_value={ 'h1': '/a', 'h2': '/b', 'h3': '/c' } ), \
             mock.patch.object( self.auto_tune.bu_tuning, 'replay_tracking_context',
                                side_effect=lambda h, *a, **k: { 'profile_name':
                                    { 'h1': 'scene', 'h2': 'quick_cut', 'h3': 'quick_cut' }[h] } ), \
             mock.patch.object( self.auto_tune.bu_tuning.bu_detector, 'get_profile_item_overrides',
                                side_effect=lambda profile, label, backend=None: profile_values.get( profile, {} ) ), \
             mock.patch.object( self.auto_tune.bu_detector, 'get_profile_item_overrides',
                                side_effect=lambda profile, label, backend=None: profile_values.get( profile, {} ) ), \
             mock.patch.object( self.auto_tune, 'measure_stage', side_effect=measure ), \
             mock.patch.object( self.auto_tune, 'print_stage_table' ), \
             mock.patch.object( self.auto_tune.TrackMaxGapStage, 'decide',
                                side_effect=lambda rows, *a: bu_tuning.Decision( 'x', None, False, 'kept', [] ) ), \
             mock.patch.object( self.auto_tune, '_median_shot_seconds_for', return_value=None ):
            self.auto_tune.main()
        return seen

    def test_a_label_with_no_backend_value_still_tunes_its_profiles( self ):
        # exposed_vulva sets track_max_gap only in its profiles; the key
        # used to be skipped whole when the backend had no value.
        self.assertEqual( [ profile for profile, _videos, _label in self.run_main_measuring( None ) ],
                          [ 'quick_cut', 'scene' ] )

    def test_a_per_variant_split_is_never_written_into_a_profile( self ):
        source = open( os.path.join( REPO, 'tools/tuning/auto_tune.py' ), encoding='utf-8' ).read()
        self.assertIn( 'if decision.is_split and probe.profile_name:', source )


class TestProfileWriter( unittest.TestCase ):

    SAMPLE = '''detector_backend = {
    'nudenet_v3': {
        'variants': {
            '640m': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 9.0 } } },
        },
        'profiles': {
            'default': 'scene',
            'variants': {
                'quick_cut': {
                    'video_censor_fps': 18,
                    'item_overrides': {
                        'exposed_breast': { 'track_max_gap': 0.20,
                                            'time_safety': 0.06,
                                            'style_min_dwell_seconds': 2.5 },
                        'exposed_vulva':  { 'track_max_gap': 0.20 },
                    },
                },
                'scene': {
                    'item_overrides': {
                        'exposed_breast': { 'track_max_gap': 3.0,
                                            'style_min_dwell_seconds': 3.0 },
                    },
                },
                'bare': {
                    'manual_only': True,
                },
            },
        },
        'item_overrides': {
            'exposed_breast': {
                'track_max_gap': 0.17,
            },
        },
    },
}
'''

    def setUp( self ):
        directory = tempfile.mkdtemp( prefix='betasuite-profile-writer-' )
        self.addCleanup( shutil.rmtree, directory, True )
        self.path = os.path.join( directory, 'betaconfig.py' )
        with open( self.path, 'w', encoding='UTF-8' ) as handle:
            handle.write( self.SAMPLE )
        self.writer = bu_tuning.ConfigWriter( config_path=self.path, backup_dir=directory )
        self.writer._write_and_verify = lambda updated, resolve, expected, description: open(
            self.path, 'w', encoding='UTF-8' ).write( updated )

    def resolved( self ):
        source = self.writer.read()
        ast.parse( source )
        namespace = {}
        exec( compile( source, self.path, 'exec' ), namespace )
        return namespace['detector_backend']['nudenet_v3']

    def profile( self, name ):
        return self.resolved()['profiles']['variants'][name].get( 'item_overrides', {} )

    def test_the_first_key_on_the_brace_line_is_updated( self ):
        self.writer.set_profile_item_override( 'nudenet_v3', 'quick_cut', 'exposed_breast', 'track_max_gap', 0.3 )
        self.assertEqual( self.profile( 'quick_cut' )['exposed_breast'],
                          { 'track_max_gap': 0.3, 'time_safety': 0.06, 'style_min_dwell_seconds': 2.5 } )
        self.assertEqual( self.profile( 'quick_cut' )['exposed_vulva'], { 'track_max_gap': 0.2 } )

    def test_the_last_key_before_the_brace_is_updated_not_duplicated( self ):
        self.writer.set_profile_item_override( 'nudenet_v3', 'quick_cut', 'exposed_breast',
                                               'style_min_dwell_seconds', 4.0 )
        self.assertEqual( self.writer.read().count( "'style_min_dwell_seconds'" ), 2,
                          "one in quick_cut, one in scene" )
        self.assertEqual( self.profile( 'quick_cut' )['exposed_breast']['style_min_dwell_seconds'], 4.0 )

    def test_a_new_key_joins_a_compact_block( self ):
        self.writer.set_profile_item_override( 'nudenet_v3', 'scene', 'exposed_breast',
                                               'match_distance_multiplier', 1.5 )
        self.assertEqual( self.profile( 'scene' )['exposed_breast'],
                          { 'track_max_gap': 3.0, 'style_min_dwell_seconds': 3.0,
                            'match_distance_multiplier': 1.5 } )
        self.assertIn( 'set by auto_tune', self.writer.read() )
        self.writer.set_profile_item_override( 'nudenet_v3', 'scene', 'exposed_breast',
                                               'match_distance_multiplier', 2.0 )
        self.assertEqual( self.profile( 'scene' )['exposed_breast']['match_distance_multiplier'], 2.0 )

    def test_a_missing_label_and_a_missing_item_overrides_are_created( self ):
        self.writer.set_profile_item_override( 'nudenet_v3', 'scene', 'exposed_vulva', 'track_max_gap', 8.0 )
        self.writer.set_profile_item_override( 'nudenet_v3', 'bare', 'exposed_vulva', 'track_max_gap', 4.0 )
        self.assertEqual( self.profile( 'scene' )['exposed_vulva'], { 'track_max_gap': 8.0 } )
        self.assertEqual( self.profile( 'bare' ), { 'exposed_vulva': { 'track_max_gap': 4.0 } } )
        self.assertTrue( self.resolved()['profiles']['variants']['bare']['manual_only'] )

    def test_other_blocks_are_untouched( self ):
        before = self.resolved()
        self.writer.set_profile_item_override( 'nudenet_v3', 'quick_cut', 'exposed_breast', 'track_max_gap', 0.5 )
        after = self.resolved()
        self.assertEqual( after['item_overrides'], before['item_overrides'] )
        self.assertEqual( after['variants'], before['variants'],
                          "a model variants block is not the profiles' variants" )
        self.assertEqual( after['profiles']['variants']['scene'], before['profiles']['variants']['scene'] )

    def test_an_unknown_profile_or_backend_is_refused( self ):
        with self.assertRaises( bu_tuning.ConfigWriteError ):
            self.writer.set_profile_item_override( 'nudenet_v3', 'no_such', 'exposed_breast', 'track_max_gap', 1 )
        with self.assertRaises( bu_tuning.ConfigWriteError ):
            self.writer.set_profile_item_override( 'nope', 'scene', 'exposed_breast', 'track_max_gap', 1 )
        self.assertEqual( self.writer.read(), self.SAMPLE )

    def test_only_direct_children_are_matched( self ):
        nested = self.SAMPLE.replace(
            "                    'video_censor_fps': 18,\n",
            "                    'video_censor_fps': 18,\n"
            "                    'notes': { 'item_overrides': { 'exposed_vulva': { 'track_max_gap': 1.0 } } },\n" )
        with open( self.path, 'w', encoding='UTF-8' ) as handle:
            handle.write( nested )
        self.writer.set_profile_item_override( 'nudenet_v3', 'quick_cut', 'exposed_vulva', 'track_max_gap', 0.4 )
        quick = self.resolved()['profiles']['variants']['quick_cut']
        self.assertEqual( quick['item_overrides']['exposed_vulva']['track_max_gap'], 0.4 )
        self.assertEqual( quick['notes']['item_overrides']['exposed_vulva']['track_max_gap'], 1.0 )

    def test_a_label_joins_an_item_overrides_closed_on_the_same_line( self ):
        compact = self.SAMPLE.replace(
            "                'bare': {\n                    'manual_only': True,\n                },\n",
            "                'bare': {\n                    'item_overrides': { 'exposed_breast': { 'track_max_gap': 1.0 } } },\n" )
        with open( self.path, 'w', encoding='UTF-8' ) as handle:
            handle.write( compact )
        self.writer.set_profile_item_override( 'nudenet_v3', 'bare', 'exposed_vulva', 'track_max_gap', 2.0 )
        self.assertEqual( self.profile( 'bare' ), { 'exposed_breast': { 'track_max_gap': 1.0 },
                                                    'exposed_vulva': { 'track_max_gap': 2.0 } } )

    def profiles_first( self ):
        """His layout: the profiles block, with its own variants and item_overrides, comes first."""
        source = self.SAMPLE
        compact = (
            "        'variants': {\n"
            "            '640m': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 9.0 } } },\n"
            "        },\n" )
        assert compact in source
        source = source.replace( compact, '' )
        # The layout the variant writer itself produces.
        model_variants = (
            "        'variants': {\n"
            "            '640m': {\n"
            "                'item_overrides': {\n"
            "                    'exposed_breast': {\n"
            "                        'track_max_gap': 9.0,\n"
            "                    },\n"
            "                },\n"
            "            },\n"
            "        },\n" )
        source = source.replace( "        'item_overrides': {\n            'exposed_breast': {\n",
                                 model_variants + "        'item_overrides': {\n            'exposed_breast': {\n" )
        with open( self.path, 'w', encoding='UTF-8' ) as handle:
            handle.write( source )

    def test_a_backend_value_goes_to_the_backend_not_the_first_profile( self ):
        self.profiles_first()
        before = self.resolved()['profiles']
        self.writer.set_item_override( 'nudenet_v3', 'exposed_breast', 'confirm_score', 0.7 )
        self.writer.set_item_override( 'nudenet_v3', 'exposed_breast', 'track_max_gap', 0.25 )
        after = self.resolved()
        self.assertEqual( after['item_overrides']['exposed_breast'],
                          { 'track_max_gap': 0.25, 'confirm_score': 0.7 } )
        self.assertEqual( after['profiles'], before, "quick_cut's block comes first and must not move" )

    def test_a_model_variant_value_skips_the_profiles_variants( self ):
        self.profiles_first()
        before = self.resolved()['profiles']
        self.writer.set_variant_item_override( 'nudenet_v3', '640m', 'exposed_breast', 'track_max_gap', 5.0 )
        self.writer.set_variant_item_override( 'nudenet_v3', '320n', 'exposed_vulva', 'track_max_gap', 6.0 )
        after = self.resolved()
        self.assertEqual( after['variants']['640m']['item_overrides']['exposed_breast']['track_max_gap'], 5.0 )
        self.assertEqual( after['variants']['320n']['item_overrides']['exposed_vulva']['track_max_gap'], 6.0 )
        self.assertEqual( after['profiles'], before )

    def test_a_duplicated_key_is_refused( self ):
        doubled = self.SAMPLE.replace( "'time_safety': 0.06,", "'time_safety': 0.06, 'time_safety': 0.07," )
        with open( self.path, 'w', encoding='UTF-8' ) as handle:
            handle.write( doubled )
        with self.assertRaises( bu_tuning.ConfigWriteError ):
            self.writer.set_profile_item_override( 'nudenet_v3', 'quick_cut', 'exposed_breast', 'time_safety', 0.1 )

    def test_a_key_named_in_a_comment_or_string_is_not_matched( self ):
        commented = self.SAMPLE.replace(
            "                'scene': {\n",
            "                # 'scene': { old notes\n                'scene': {\n" )
        with open( self.path, 'w', encoding='UTF-8' ) as handle:
            handle.write( commented )
        self.writer.set_profile_item_override( 'nudenet_v3', 'scene', 'exposed_breast', 'track_max_gap', 5.0 )
        self.assertEqual( self.profile( 'scene' )['exposed_breast']['track_max_gap'], 5.0 )
        self.assertIn( "# 'scene': { old notes", self.writer.read() )


class TestSuppressionPairsCoverTheRules( unittest.TestCase ):

    def test_configured_rules_are_added_once( self ):
        module = _load( 'tools/analysis/analyze_suppression_pairs.py', '_pairs_under_test' )
        rules = { 'exposed_vulva': [ { 'suppressed_by': 'exposed_belly' }, { 'suppressed_by': 'face_femme' } ],
                  'exposed_breast': { 'suppressed_by': 'covered_breast' } }
        with mock.patch.object( module.bu_detector, 'registered_backend_names', return_value=[ 'a', 'b' ] ), \
             mock.patch.object( module.bu_detector, 'get_class_suppression', return_value=rules ):
            pairs = module.pairs_to_check()
        self.assertEqual( pairs[:len( module.PAIRS_OF_INTEREST )], module.PAIRS_OF_INTEREST )
        self.assertEqual( pairs.count( ( 'exposed_vulva', 'exposed_belly' ) ), 1 )
        self.assertEqual( pairs.count( ( 'exposed_vulva', 'face_femme' ) ), 1, "already listed" )
        self.assertEqual( pairs.count( ( 'exposed_breast', 'covered_breast' ) ), 1 )


class TestBetabenchReadsMergedConfigurations( CacheDir ):

    def test_a_configuration_reads_its_newest_run_at_every_rate( self ):
        betabench = _load( 'tools/bench/betabench.py', '_betabench_rates_under_test' )
        self.write( 'aaaa', 9, 'k9' )
        self.write( 'bbbb', 18, 'k18' )
        config = self.configurations( include_preview='fallback' )[0]
        logger = mock.Mock()
        with mock.patch.object( betaconfig, 'global_min_prob', self.MIN_PROB ), \
             mock.patch.object( betabench, '_read_cached_detections',
                                side_effect=lambda videos, *a: ( videos, len( videos ) ) ):
            videos, count = betabench.load_cached_detections( 'nudenet_v3', logger, configuration=config )
        self.assertEqual( ( sorted( videos ), count ), ( [ 'aaaa', 'bbbb' ], 2 ) )


if __name__ == '__main__':
    unittest.main()
