"""
test_analysis_tools_contract.py - the contract every analysis and tuning
tool depends on, tested directly so a refactor of the pipeline cannot
break the whole tuning suite silently again.

This exists because of a real, expensive failure. The v2.1.0 refactor
moved smooth_boxes and apply_class_suppression out of betatv.py into
betautils_track.py, and moved every tracking knob out of the shared
betaconfig.item_overrides into each backend's own
detector_backend[<name>]['item_overrides'] block. Both moves were
correct. Neither broke a single test. What they broke was every tool
that reads the cache and replays the pipeline, in three distinct ways:

  1. Three tools loaded the real functions by regex-extracting
     betatv.py's SOURCE TEXT and exec'ing it. The functions were no
     longer in that file, so the tools raised at startup. Loud, at
     least - it was found the morning after an overnight run.

  2. Every tool defaulted --picture-sizes to the shared
     betaconfig.picture_sizes, which since v2.1 is not the size any
     backend necessarily runs at. The tools looked for caches at a size
     nothing had written and reported "no detection caches found" for
     backends that had a complete set. SILENT: an empty report reads
     exactly like "you have not run anything yet".

  3. analyze_track_breaks's and analyze_jitter's parameter sweeps wrote
     the swept value into the shared betaconfig.item_overrides, which
     get_item_overrides then OVERWROTE with the backend block for
     exactly the keys being swept. Also silent, and worse than silent:
     the sweep printed a full table of numbers in which every row was
     the same run. Acting on that table would have been acting on
     noise.

The common shape is that all three failures produce plausible-looking
output. So this file tests the contract rather than the tools' prose:
the extension point exists and fires, size resolution follows the
backend, and a sweep's writes actually reach the resolver.
"""

import ast
import copy
import importlib
import os
import sys
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_track as bu_track


TOOL_DIRS = ( 'tools/analysis', 'tools/tuning' )

# Every tool that replays the real pipeline over cached detections.
# Listed explicitly rather than globbed: a new tool should have to be
# added here deliberately, so "it isn't covered" is a visible decision
# and not an oversight.
PIPELINE_REPLAY_TOOLS = (
    'tools/analysis/analyze_jitter.py',
    'tools/analysis/analyze_track_breaks.py',
    'tools/analysis/analyze_style_flicker.py',
    'tools/analysis/analyze_span_merge.py',
    'tools/tuning/replay_tune.py',
)

CACHE_READING_TOOLS = PIPELINE_REPLAY_TOOLS + (
    'tools/analysis/analyze_score_distribution.py',
    'tools/analysis/analyze_suppression_pairs.py',
    'tools/analysis/compare_models_perf.py',
    'tools/tuning/batch_ab_test.py',
)


def repo_path( *parts ):
    return os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), *parts )


class TestNoToolReadsPipelineSourceText( unittest.TestCase ):
    """
    Failure mode 1: loading pipeline functions by scraping source text.

    A tool that regex-extracts and exec's another module's source is
    coupled to that module's line-by-line text, not to its interface -
    so any refactor breaks it, and nothing in the test suite notices.
    betautils_track.TrackingObserver exists precisely so the tools can
    observe real decisions through a supported seam instead.
    """

    def test_no_tool_extracts_and_execs_pipeline_source( self ):
        offenders = []
        for tool in PIPELINE_REPLAY_TOOLS:
            with open( repo_path( tool ), encoding='UTF-8' ) as fin:
                source = fin.read()
            # The specific pattern that broke: pulling a function body
            # out of another file's text and exec'ing it.
            if 'exec(' in source and 're.search' in source:
                offenders.append( tool )
        self.assertEqual(
            offenders, [],
            "these tools reconstruct pipeline functions from source text instead of importing "
            "them and using bu_track.tracking_observer; a refactor will break them silently: %s"%(offenders) )


class TestTrackingObserverIsALiveExtensionPoint( unittest.TestCase ):
    """
    The seam the tools now depend on. If smooth_boxes ever stops
    notifying, every jitter/track-break/style-flicker number silently
    becomes an empty report rather than an error.
    """

    def _one_label_boxes( self ):
        # Same shape process_raw_box produces, which is what smooth_boxes
        # is contractually handed. Four consecutive, nearly-stationary
        # detections: the first starts a track, the rest continue it, so
        # both observer events are guaranteed to fire.
        boxes = []
        for index in range( 4 ):
            start = index * 0.1
            boxes.append( {
                'label': 'exposed_breast',
                'start': start, 'end': start + 0.1, 't': start,
                'x': 100 + index, 'y': 100, 'w': 40, 'h': 40,
                'score': 0.9,
                'censor_style': { 'type': 'blur', 'method': 'gaussian', 'strength': 16 },
                'censor_shape': 'box',
                'censor_sticker_seed': 0,
                '_raw_x': 100 + index, '_raw_y': 100, '_raw_w': 40, '_raw_h': 40,
                '_vid_w': 1920, '_vid_h': 1080,
            } )
        return boxes

    def test_observer_sees_matches_and_new_tracks( self ):
        seen = { 'match': 0, 'new_track': 0, 'candidates': 0 }

        class _Recorder( bu_track.TrackingObserver ):
            def on_match( self, label, track_index, distance, box, tracks, settings ):
                seen['match'] += 1

            def on_new_track( self, label, tracks, track_index, box, settings ):
                seen['new_track'] += 1

            def on_frame_candidates( self, label, candidates, tracks, settings ):
                seen['candidates'] += 1

        with bu_track.tracking_observer( _Recorder() ):
            bu_track.smooth_boxes( self._one_label_boxes() )

        self.assertGreaterEqual( seen['new_track'], 1,
            "smooth_boxes never reported starting a track - the observer seam the analysis "
            "tools rely on has stopped firing" )
        self.assertGreaterEqual( seen['match'], 1,
            "smooth_boxes never reported continuing a track, so no jitter/reset numbers can be measured" )

    def test_observer_is_removed_on_the_way_out( self ):
        class _Recorder( bu_track.TrackingObserver ):
            pass

        with bu_track.tracking_observer( _Recorder() ):
            pass
        after = { 'match': 0 }

        class _ShouldNotFire( bu_track.TrackingObserver ):
            def on_match( self, *args, **kwargs ):
                after['match'] += 1

        # No context manager: nothing should be observing now.
        bu_track.smooth_boxes( self._one_label_boxes() )
        self.assertEqual( after['match'], 0 )


class TestPictureSizeResolutionFollowsTheBackend( unittest.TestCase ):
    """
    Failure mode 2: looking for caches at a size nothing wrote.
    """

    def setUp( self ):
        self._orig = copy.deepcopy( betaconfig.detector_backend )

    def tearDown( self ):
        betaconfig.detector_backend = self._orig
        bu_config.invalidate_config_caches()

    def test_default_follows_the_backend_not_another_backends_size( self ):
        # retinanet_v2 runs at 1280; nudenet_v3 must still resolve to its
        # own 320 rather than picking up the other backend's size.
        betaconfig.detector_backend = copy.deepcopy( self._orig )
        betaconfig.detector_backend['nudenet_v3'] = dict(
            betaconfig.detector_backend.get( 'nudenet_v3', {} ) )
        betaconfig.detector_backend['nudenet_v3']['picture_sizes'] = [ 320 ]

        self.assertEqual( bu_cache.resolve_picture_sizes( None, 'nudenet_v3' ), [ 320 ],
            "a tool with no --picture-sizes must look where THIS backend's caches actually are" )

    def test_an_explicit_value_always_wins( self ):
        self.assertEqual( bu_cache.resolve_picture_sizes( [ 2000 ], 'nudenet_v3' ), [ 2000 ],
            "an explicit --picture-sizes must still be able to inspect an old configuration's caches" )

    def test_no_tool_reads_the_removed_shared_picture_sizes( self ):
        # betaconfig.picture_sizes was removed in 2.5. A tool still
        # READING it would now raise AttributeError, or resolve to
        # nothing and report 'no caches found' for a backend whose caches
        # are complete. Mentions in help strings and docstrings are fine
        # and common (they explain why the default is per-backend), so
        # this walks the AST and flags only real attribute loads.
        offenders = []
        for tool in CACHE_READING_TOOLS:
            with open( repo_path( tool ), encoding='UTF-8' ) as fin:
                source = fin.read()
            tree = ast.parse( source, filename=tool )
            for node in ast.walk( tree ):
                if ( isinstance( node, ast.Attribute )
                        and node.attr == 'picture_sizes'
                        and isinstance( node.value, ast.Name )
                        and node.value.id == 'betaconfig' ):
                    offenders.append( '%s:%d'%(tool, node.lineno) )
                # getattr( betaconfig, 'picture_sizes', ... ) reads it too
                if ( isinstance( node, ast.Call )
                        and isinstance( node.func, ast.Name )
                        and node.func.id == 'getattr'
                        and len( node.args ) >= 2
                        and isinstance( node.args[0], ast.Name )
                        and node.args[0].id == 'betaconfig'
                        and isinstance( node.args[1], ast.Constant )
                        and node.args[1].value == 'picture_sizes' ):
                    offenders.append( '%s:%d'%(tool, node.lineno) )
        self.assertEqual(
            sorted( offenders ), [],
            "these tools still read the removed shared betaconfig.picture_sizes instead of "
            "resolving the size per backend: %s"%(sorted( offenders )) )


class TestSweepsReachTheResolver( unittest.TestCase ):
    """
    Failure mode 3: a sweep whose value is overwritten before it is read.

    Both swept keys are backend-tunable, so a backend block containing
    them wins over anything written to the shared dict. These tests set
    up exactly that situation - a backend block that already pins the
    key - and assert the sweep's value is what smooth_boxes ends up
    seeing.
    """

    def setUp( self ):
        self._orig_backend = copy.deepcopy( betaconfig.detector_backend )
        self._orig_shared = copy.deepcopy( getattr( betaconfig, 'item_overrides', {} ) )
        self._orig_alpha = betaconfig.default_position_smoothing

    def tearDown( self ):
        betaconfig.detector_backend = self._orig_backend
        betaconfig.item_overrides = self._orig_shared
        betaconfig.default_position_smoothing = self._orig_alpha
        bu_config.invalidate_config_caches()

    def _pin_backend_key( self, backend_name, label, key, value ):
        block = betaconfig.detector_backend.setdefault( backend_name, {} )
        overrides = dict( block.get( 'item_overrides', {} ) )
        overrides[label] = dict( overrides.get( label, {} ) )
        overrides[label][key] = value
        block['item_overrides'] = overrides
        bu_config.invalidate_config_caches()

    def _load_tool( self, tool_path, module_name ):
        spec = importlib.util.spec_from_file_location( module_name, repo_path( tool_path ) )
        module = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( module )
        return module

    def test_track_break_sweep_overrides_a_pinned_backend_value( self ):
        backend, label = 'nudenet_v3', 'exposed_breast'
        self._pin_backend_key( backend, label, 'match_distance_multiplier', 1.0 )
        self._pin_backend_key( backend, label, 'track_max_gap', 0.5 )

        tool = self._load_tool( 'tools/analysis/analyze_track_breaks.py', '_tb_under_test' )
        baseline_gap = bu_detector.get_item_overrides( label, backend )['track_max_gap']

        with tool._swept_overrides( backend, { label }, match_mult=3.0, gap_mult=4.0 ):
            effective = bu_detector.get_item_overrides( label, backend )
            self.assertAlmostEqual( effective['match_distance_multiplier'], 3.0,
                msg="the match-distance sweep never reached the resolver - every row of the "
                    "sweep table would be the same run" )
            self.assertAlmostEqual( effective['track_max_gap'], baseline_gap * 4.0,
                msg="the track_max_gap sweep never reached the resolver" )

        restored = bu_detector.get_item_overrides( label, backend )
        self.assertAlmostEqual( restored['match_distance_multiplier'], 1.0 )
        self.assertAlmostEqual( restored['track_max_gap'], 0.5 )

    def test_jitter_alpha_sweep_overrides_a_pinned_backend_value( self ):
        backend, label = 'nudenet_v3', 'exposed_breast'
        self._pin_backend_key( backend, label, 'position_smoothing', 0.9 )

        tool = self._load_tool( 'tools/analysis/analyze_jitter.py', '_jit_under_test' )

        with tool._swept_alpha( backend, { label }, 0.05 ):
            effective = bu_detector.get_item_overrides( label, backend )
            self.assertAlmostEqual( effective['position_smoothing'], 0.05,
                msg="the alpha sweep never reached the resolver: a label with its own "
                    "position_smoothing would keep it and the sweep table would be flat" )

        self.assertAlmostEqual(
            bu_detector.get_item_overrides( label, backend )['position_smoothing'], 0.9 )
        self.assertAlmostEqual( betaconfig.default_position_smoothing, self._orig_alpha )



class TestConfigurationDiscovery( unittest.TestCase ):
    """
    Failure mode 2, solved properly: ask the disk, not the config.

    Resolving picture sizes per backend fixed the immediate bug, but it
    still answers "what is configured now". The question every analysis
    tool actually has is "what has been RUN", and those differ the
    moment a variant is swapped. These tests pin the difference.
    """

    def setUp( self ):
        import tempfile, gzip, json as json_module
        self._dir = tempfile.TemporaryDirectory()
        self._saved_vid = bu_cache.VID_HASH_DIR
        self._saved_runkeys = betaconst.run_key_dir
        self._cache = os.path.join( self._dir.name, 'vid_hashes' )
        self._runkeys = os.path.join( self._dir.name, 'run_keys' )
        os.makedirs( self._cache )
        os.makedirs( self._runkeys )
        bu_cache.VID_HASH_DIR = self._cache
        betaconst.run_key_dir = self._runkeys
        self._gzip = gzip
        self._json = json_module

    def tearDown( self ):
        bu_cache.VID_HASH_DIR = self._saved_vid
        betaconst.run_key_dir = self._saved_runkeys
        self._dir.cleanup()

    def _write_cache( self, file_hash, backend, size, key, variant=None,
                      fps=None, min_prob=None ):
        fps = betaconfig.video_censor_fps if fps is None else fps
        min_prob = ( getattr( betaconfig, 'global_min_prob', 0.2 )
                     if min_prob is None else min_prob )
        stem = bu_cache.cache_name_stem( backend, size, fps, min_prob )
        path = os.path.join( self._cache, '%s%s%s.gz'%(file_hash, stem, key) )
        with self._gzip.open( path, 'wt', encoding='UTF-8' ) as fout:
            self._json.dump( [], fout )
        identity = { 'backend': backend, 'size': size,
                     'backend_tunables': ( { 'model_variant': variant } if variant else {} ) }
        with open( os.path.join( self._runkeys, 'detection-%s.json'%(key) ), 'w',
                   encoding='UTF-8' ) as fout:
            self._json.dump( { 'kind': 'detection', 'key': key, 'identity': identity }, fout )

    def test_two_variants_of_one_backend_are_two_configurations( self ):
        self._write_cache( 'aaaa', 'nudenet_v3', 320, '320aaa', '320n' )
        self._write_cache( 'aaaa', 'nudenet_v3', 640, '640bbb', '640m' )
        configurations = bu_cache.discover_cached_configurations()
        self.assertEqual( len( configurations ), 2,
            "a 320n run and a 640m run are two sets of model weights and must never be "
            "reported as one backend's results" )
        self.assertEqual( [ config.variant for config in configurations ], [ '320n', '640m' ] )

    def test_the_variant_is_recovered_from_the_run_key_manifest( self ):
        self._write_cache( 'aaaa', 'nudenet_v3', 640, '640bbb', '640m' )
        config = bu_cache.discover_cached_configurations()[0]
        self.assertEqual( config.variant, '640m',
            "the filename carries the size but not the variant, so the label has to come "
            "from the run-key manifest or the report cannot name what it measured" )
        self.assertEqual( config.label, 'nudenet_v3/640m @640' )

    def test_a_cache_with_no_manifest_still_discovers( self ):
        stem = bu_cache.cache_name_stem(
            'nudenet_v3', 320, betaconfig.video_censor_fps,
            getattr( betaconfig, 'global_min_prob', 0.2 ) )
        path = os.path.join( self._cache, 'bbbb%sorphan.gz'%(stem) )
        with self._gzip.open( path, 'wt', encoding='UTF-8' ) as fout:
            self._json.dump( [], fout )
        configurations = bu_cache.discover_cached_configurations()
        self.assertEqual( len( configurations ), 1,
            "a missing manifest costs the variant NAME, never the configuration itself - "
            "silently dropping the data would be the worse failure" )
        self.assertIsNone( configurations[0].variant )

    def test_preview_caches_are_excluded_by_default( self ):
        self._write_cache( 'aaaa', 'nudenet_v3', 320, '320aaa', '320n' )
        stem = bu_cache.cache_name_stem(
            'nudenet_v3', 320, betaconfig.video_censor_fps,
            getattr( betaconfig, 'global_min_prob', 0.2 ) )
        path = os.path.join( self._cache, 'cccc%s320aaa-preview.gz'%(stem) )
        with self._gzip.open( path, 'wt', encoding='UTF-8' ) as fout:
            self._json.dump( [], fout )
        self.assertEqual( len( bu_cache.discover_cached_configurations() ), 1 )
        self.assertEqual( len( bu_cache.discover_cached_configurations( include_preview=True ) ), 2,
            "a preview slice describes a few seconds, so it is its own configuration rather "
            "than extra rows inside a real run's" )

    def test_a_different_fps_is_not_matched( self ):
        self._write_cache( 'aaaa', 'nudenet_v3', 320, '320aaa', '320n',
                           fps=betaconfig.video_censor_fps + 3 )
        self.assertEqual( bu_cache.discover_cached_configurations(), [],
            "caches from another sample rate are not this run's data, and quietly folding "
            "them in would mix two different samplings of the same footage" )

    def test_explicit_sizes_still_win( self ):
        self._write_cache( 'aaaa', 'nudenet_v3', 320, '320aaa', '320n' )
        configurations = bu_cache.configurations_to_analyse( [ 2000 ] )
        self.assertTrue( all( config.picture_sizes == [ 2000 ] for config in configurations ),
            "--picture-sizes must remain able to inspect a configuration that is not on disk "
            "in the shape discovery would produce" )

    def test_an_empty_cache_falls_back_to_configured_sizes( self ):
        configurations = bu_cache.configurations_to_analyse( None )
        self.assertTrue( configurations,
            "with nothing on disk a tool should still print a sensible 'no caches for X' per "
            "backend rather than reporting nothing at all" )


class TestCacheFilenameConfigurationLabel( unittest.TestCase ):
    """The grouping key that analyze_suppression_pairs got wrong."""

    def test_size_is_part_of_the_label( self ):
        first = bu_cache.configuration_label_of_cache_filename(
            'aaaa-nudenet_v3-%s-320-9-0.200-dabc123.gz'%(betaconst.picture_saved_box_version) )
        second = bu_cache.configuration_label_of_cache_filename(
            'aaaa-nudenet_v3-%s-640-9-0.200-ddef456.gz'%(betaconst.picture_saved_box_version) )
        self.assertNotEqual( first, second,
            "grouping by backend name alone put fourteen cache files from two different sets "
            "of model weights under one heading, and every number under it described neither" )

    def test_an_unparseable_name_is_labelled_not_guessed( self ):
        self.assertEqual(
            bu_cache.configuration_label_of_cache_filename( 'something-else.gz' ),
            'unknown/legacy' )

    def test_a_preview_cache_is_marked_as_one( self ):
        label = bu_cache.configuration_label_of_cache_filename(
            'aaaa-nudenet_v3-%s-320-9-0.200-dabc123-preview@120.0s.gz'%(
                betaconst.picture_saved_box_version ) )
        self.assertIn( 'PREVIEW', label )


class TestVariantEnumeration( unittest.TestCase ):

    def test_a_multi_variant_backend_lists_them_all( self ):
        self.assertEqual( bu_detector.variant_names( 'nudenet_v3' ), [ '320n', '640m' ] )

    def test_a_single_weight_backend_has_no_variant_axis( self ):
        self.assertEqual( bu_detector.variant_names( 'retinanet_v2' ), [] )

    def test_each_variant_reports_its_native_size( self ):
        self.assertEqual( bu_detector.native_size_for_variant( 'nudenet_v3', '320n' ), 320 )
        self.assertEqual( bu_detector.native_size_for_variant( 'nudenet_v3', '640m' ), 640 )

    def test_an_unknown_variant_reports_no_size_rather_than_guessing( self ):
        self.assertIsNone( bu_detector.native_size_for_variant( 'nudenet_v3', 'nope' ) )


class TestEveryToolStillImports( unittest.TestCase ):
    """
    The cheapest possible guard against the class of breakage above: a
    tool that no longer imports is a tool that will fail at 2am, after
    the overnight run it was supposed to analyse has already finished.
    """

    def test_every_tool_compiles( self ):
        import py_compile
        failures = []
        for directory in TOOL_DIRS:
            full = repo_path( directory )
            for name in sorted( os.listdir( full ) ):
                if not name.endswith( '.py' ):
                    continue
                try:
                    py_compile.compile( os.path.join( full, name ), doraise=True )
                except Exception as err:
                    failures.append( '%s/%s: %s'%(directory, name, err) )
        self.assertEqual( failures, [], "tool(s) failed to compile: %s"%(failures) )


class TestVariantFiltering( unittest.TestCase ):
    """
    --variants narrows which cached configurations a tool reads.

    WHY BACKEND ALONE IS NOT ENOUGH
    -------------------------------
    320n and 640m are both nudenet_v3, so --backends nudenet_v3 still
    analyses both. Once one variant is the one being tuned, replaying
    the others is pure cost: on the 2026-09-19 run the retinanet replays
    alone were roughly 11 minutes of a 15-minute auto_tune.

    The filter must only ever narrow. It reads nothing new, writes
    nothing, and leaves every cache on disk untouched, so widening it
    again later needs no re-run - that is the property these tests pin,
    because a filter that quietly changed what was measured would make
    two runs incomparable.
    """

    class _Configuration:
        """Minimal stand-in carrying the two fields the filter reads."""
        def __init__( self, label, variant ):
            self.label = label
            self.variant = variant
            self.backend_name = label.split( '/' )[0]
            self.picture_sizes = [ 640 ]
            self.file_hashes = ()

    def setUp( self ):
        self._discovered = [
            self._Configuration( 'nudenet_v3/320n @320', '320n' ),
            self._Configuration( 'nudenet_v3/640m @640', '640m' ),
            self._Configuration( 'retinanet_v2 @1280', None ),
        ]
        self._original = bu_cache.discover_cached_configurations
        bu_cache.discover_cached_configurations = (
            lambda **kwargs: list( self._discovered ) )

    def tearDown( self ):
        bu_cache.discover_cached_configurations = self._original

    def test_no_filter_returns_every_configuration( self ):
        got = bu_cache.configurations_to_analyse()
        self.assertEqual( [ c.label for c in got ],
                          [ c.label for c in self._discovered ] )

    def test_a_single_variant_is_isolated( self ):
        got = bu_cache.configurations_to_analyse( variant_names=[ '640m' ] )
        self.assertEqual( [ c.label for c in got ], [ 'nudenet_v3/640m @640' ] )

    def test_several_variants_can_be_named( self ):
        got = bu_cache.configurations_to_analyse( variant_names=[ '320n', '640m' ] )
        self.assertEqual( [ c.label for c in got ],
                          [ 'nudenet_v3/320n @320', 'nudenet_v3/640m @640' ] )

    def test_a_backend_with_no_variant_is_excluded_by_an_explicit_filter( self ):
        # retinanet_v2 has no variant concept, so its variant is None.
        # Naming variants is an explicit request for specific ones;
        # silently keeping an unnamed backend would defeat the point of
        # asking for 640m only.
        got = bu_cache.configurations_to_analyse( variant_names=[ '640m' ] )
        self.assertNotIn( 'retinanet_v2 @1280', [ c.label for c in got ] )

    def test_an_unmatched_filter_returns_nothing_rather_than_everything( self ):
        # Failing open here would be the dangerous behaviour: a typo in
        # --variants would silently analyse every configuration and the
        # numbers would look plausible.
        got = bu_cache.configurations_to_analyse( variant_names=[ 'no-such-variant' ] )
        self.assertEqual( [ c.label for c in got ], [] )

    def test_the_filter_does_not_mutate_what_was_discovered( self ):
        # It narrows a view; it must not edit the underlying list, or a
        # second call in the same process would see less than the first.
        bu_cache.configurations_to_analyse( variant_names=[ '640m' ] )
        self.assertEqual( len( self._discovered ), 3 )
        again = bu_cache.configurations_to_analyse()
        self.assertEqual( len( again ), 3 )


class TestEveryCacheReadingToolAcceptsVariants( unittest.TestCase ):
    """
    Every tool the harness runs with --variants has to accept it.

    The harness passes one flag to several tools. A tool that silently
    lacked it would fail with 'unrecognized arguments' mid-suite, and a
    tool that accepted it but never forwarded it would report on
    everything while claiming to be filtered - which is worse, because
    the output looks right.
    """

    TOOLS = (
        'tools/analysis/analyze_jitter.py',
        'tools/analysis/analyze_track_breaks.py',
        'tools/analysis/analyze_style_flicker.py',
        'tools/tuning/auto_tune.py',
        'tools/bench/betabench.py',
    )

    def _source( self, relative ):
        root = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
        with open( os.path.join( root, relative ), encoding='UTF-8' ) as handle:
            return handle.read()

    def test_each_tool_declares_the_flag( self ):
        missing = [ tool for tool in self.TOOLS
                    if "'--variants'" not in self._source( tool ) ]
        self.assertEqual( missing, [],
            "these tools do not accept --variants, so the harness would fail "
            "passing it: %s"%(missing) )

    def test_each_tool_forwards_the_flag_to_discovery( self ):
        # Declaring the flag and then not using it is the silent-wrong
        # case: the tool runs, reports on every configuration, and the
        # header claims a filter was applied.
        missing = [ tool for tool in self.TOOLS
                    if 'variant_names=' not in self._source( tool ) ]
        self.assertEqual( missing, [],
            "these tools accept --variants but never pass it to "
            "configurations_to_analyse, so it would be silently ignored: %s"%(missing) )


if __name__ == '__main__':
    unittest.main()
