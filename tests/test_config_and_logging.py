"""
test_config_and_logging.py - fail-fast validation, resolution order,
and the logging/progress plumbing.

Validation exists so a bad setting produces a specific message before
any real work, rather than a KeyError an hour into a render. Every check
added in 2.1 is covered here, along with the ones most likely to be
silently wrong:

  - a setting that resolves per backend, set in the wrong place
  - a combination that is only invalid TOGETHER (min_prob_continue above
    min_prob; a geometry filter whose bounds exclude everything)
  - a value that would make a feature a silent no-op rather than an
    error (a min_prob at or below the global floor)

The shipped betaconfig.py must itself validate clean, which is asserted
first: a repository whose own default config fails validation is a
broken repository.
"""

import copy
import logging
import os
import shutil
import tempfile
import unittest

import betaconfig
import betaconst
import betautils_cli as bu_cli
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_log as bu_log


class ConfigHarness( unittest.TestCase ):
    """Restores every betaconfig attribute a test touches."""

    def setUp( self ):
        self._saved = {}
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        for name, value in self._saved.items():
            if value is self._MISSING:
                delattr( betaconfig, name )
            else:
                setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    _MISSING = object()

    def set_config( self, name, value ):
        if name not in self._saved:
            self._saved[name] = getattr( betaconfig, name, self._MISSING )
        setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def errors( self ):
        return bu_config._collect_validation_errors()

    def assert_error_mentioning( self, *fragments ):
        errors = self.errors()
        joined = '\n'.join( errors )
        for fragment in fragments:
            self.assertIn( fragment, joined,
                "expected an error mentioning %r, got:\n%s"%(fragment, joined) )

    def assert_no_errors( self ):
        errors = self.errors()
        self.assertEqual( errors, [], 'unexpected validation errors:\n' + '\n'.join( errors ) )


class TestShippedConfigIsValid( ConfigHarness ):

    def test_the_default_config_validates_clean( self ):
        self.assert_no_errors()

    def test_every_censored_label_is_a_real_class( self ):
        for label in betaconfig.items_to_censor:
            self.assertIn( label, betaconst.classes )

    def test_the_shared_item_overrides_block_holds_only_rendering_keys( self ):
        # Detection- and tracking-sensitive keys belong in each backend's
        # own block. A stray one here is silently ignored at runtime,
        # which is exactly why it is an error at startup.
        for label, override in betaconfig.item_overrides.items():
            unexpected = set( override ) & bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS
            self.assertEqual( unexpected, set(),
                "item_overrides['%s'] has backend-tunable key(s) %s"%(label, sorted( unexpected )) )


class TestPictureSizeValidation( ConfigHarness ):

    def test_an_empty_resolved_size_list_is_an_error( self ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['retinanet_v2']['picture_sizes'] = []
        self.set_config( 'detector_backend', backend )
        self.set_config( 'picture_sizes', [] )
        self.assert_error_mentioning( 'picture_sizes' )

    def test_duplicate_sizes_are_an_error( self ):
        # The model would run twice on identical input and every
        # detection would be duplicated.
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['retinanet_v2']['picture_sizes'] = [ 640, 640 ]
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'duplicate' )

    def test_nudenet_rejects_a_size_of_zero( self ):
        # 'native resolution, no resize' is a retinanet-only concept;
        # this adapter needs a fixed square blob. Before 2.1 this was a
        # runtime ValueError mid-render, not a startup error.
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['nudenet_v3']['picture_sizes'] = [ 0 ]
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'nudenet_v3', 'positive picture sizes' )

    def test_nudenet_rejects_a_size_that_is_not_a_multiple_of_32( self ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['nudenet_v3']['picture_sizes'] = [ 300 ]
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'multiple of 32' )

    def test_a_backend_defaults_to_its_native_size( self ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['nudenet_v3'].pop( 'picture_sizes', None )
        backend['nudenet_v3']['model_variant'] = '640m'
        self.set_config( 'detector_backend', backend )
        self.assertEqual( bu_detector.get_picture_sizes( 'nudenet_v3' ), [ 640 ] )


class TestDetectorBackendValidation( ConfigHarness ):

    def test_an_unknown_selected_backend_is_an_error( self ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['selected'] = 'not_a_backend'
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'not_a_backend', 'registered detector adapter' )

    def test_an_unknown_section_is_an_error_not_a_silent_no_op( self ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['nudnet_v3'] = { 'nn_batch_size': 8 }   # typo
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'nudnet_v3' )

    def test_an_unknown_variant_is_an_error( self ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['nudenet_v3']['model_variant'] = '512x'
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'model_variant' )

    def test_an_invalid_nms_mode_is_an_error( self ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['nudenet_v3']['nms_mode'] = 'whatever'
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'nms_mode' )

    def test_an_unselected_backend_is_validated_too( self ):
        # Switching 'selected' later must not be the first time a typo
        # in the other block is noticed.
        backend = copy.deepcopy( betaconfig.detector_backend )
        backend['selected'] = 'retinanet_v2'
        backend['nudenet_v3']['nms_iou'] = 5.0
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'nms_iou' )


class TestBackendSettingResolution( ConfigHarness ):

    def test_a_backend_block_wins_over_the_shared_defaults( self ):
        self.set_config( 'detector_backend', {
            'selected': 'nudenet_v3',
            'defaults': { 'nn_batch_size': 2 },
            'nudenet_v3': { 'nn_batch_size': 7 },
            'retinanet_v2': {},
        } )
        self.assertEqual( bu_detector.get_nn_batch_size( 'nudenet_v3' ), 7 )

    def test_the_shared_defaults_fill_in_for_a_silent_backend( self ):
        # This is the "top level default in that section" tier: one
        # sensible value for every backend that has not said otherwise.
        self.set_config( 'detector_backend', {
            'selected': 'retinanet_v2',
            'defaults': { 'nn_batch_size': 3 },
            'nudenet_v3': {},
            'retinanet_v2': {},
        } )
        self.assertEqual( bu_detector.get_nn_batch_size( 'retinanet_v2' ), 3 )

    def test_the_removed_top_level_setting_no_longer_resolves( self ):
        # The module-level betaconfig.nn_batch_size tier was dropped in
        # 2.5. With no backend block and no 'defaults' entry, resolution
        # lands on the hardcoded 1 rather than the stale shared value.
        self.set_config( 'detector_backend', {
            'selected': 'retinanet_v2', 'nudenet_v3': {}, 'retinanet_v2': {} } )
        self.set_config( 'nn_batch_size', 5 )
        self.assertEqual( bu_detector.get_nn_batch_size( 'retinanet_v2' ), 1 )

    def test_item_overrides_merge_per_key_not_wholesale( self ):
        # Setting one tracking knob per backend must never force
        # censor_style to be duplicated into that backend's block.
        self.set_config( 'item_overrides', {
            'exposed_breast': { 'censor_shape': 'circle',
                                'censor_style': { 'type': 'blur' } } } )
        self.set_config( 'detector_backend', {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_breast': { 'min_prob': 0.42 } } },
            'retinanet_v2': {},
        } )
        resolved = bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3' )
        self.assertEqual( resolved['min_prob'], 0.42 )
        self.assertEqual( resolved['censor_shape'], 'circle' )
        self.assertEqual( resolved['censor_style'], { 'type': 'blur' } )

    def test_a_non_tunable_key_in_a_backend_block_is_ignored_and_flagged( self ):
        self.set_config( 'detector_backend', {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': {
                'exposed_breast': { 'censor_shape': 'ellipse' } } },
            'retinanet_v2': {},
        } )
        resolved = bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3' )
        self.assertNotEqual( resolved.get( 'censor_shape' ), 'ellipse' )
        self.assert_error_mentioning( "aren't backend-tunable" )


class TestHysteresisValidation( ConfigHarness ):

    def _with_label_override( self, **keys ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        name = backend['selected']
        backend[name].setdefault( 'item_overrides', {} ).setdefault(
            'exposed_breast', {} ).update( keys )
        self.set_config( 'detector_backend', backend )

    def test_min_prob_continue_above_min_prob_is_an_error( self ):
        # That inverts hysteresis: the bar to KEEP tracking would be
        # higher than the bar to start.
        self._with_label_override( min_prob=0.5, min_prob_continue=0.7 )
        self.assert_error_mentioning( 'min_prob_continue', 'inverts hysteresis' )

    def test_min_prob_continue_below_the_global_floor_is_an_error( self ):
        # Nothing that low ever reaches it, so hysteresis would extend no
        # further than global_min_prob already does - a silent no-op
        # rather than a deliberately permissive setting.
        self._with_label_override( min_prob=0.5,
                                   min_prob_continue=betaconfig.global_min_prob )
        self.assert_error_mentioning( 'min_prob_continue', 'global_min_prob' )

    def test_a_sensible_pair_validates( self ):
        self._with_label_override( min_prob=0.5, min_prob_continue=0.3 )
        self.assert_no_errors()

    def test_none_disables_hysteresis_without_complaint( self ):
        self._with_label_override( min_prob=0.5, min_prob_continue=None )
        self.assert_no_errors()


class TestTrackConfirmationValidation( ConfigHarness ):

    def _with_min_track_hits( self, value ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        name = backend['selected']
        backend[name].setdefault( 'item_overrides', {} ).setdefault(
            'exposed_breast', {} )['min_track_hits'] = value
        self.set_config( 'detector_backend', backend )

    def test_zero_is_an_error( self ):
        self._with_min_track_hits( 0 )
        self.assert_error_mentioning( 'min_track_hits' )

    def test_a_non_integer_is_an_error( self ):
        self._with_min_track_hits( 2.5 )
        self.assert_error_mentioning( 'min_track_hits' )

    def test_an_absurdly_large_value_is_flagged_with_its_real_cost( self ):
        self._with_min_track_hits( 40 )
        self.assert_error_mentioning( 'min_track_hits', 'uncensored footage' )

    def test_a_reasonable_value_validates( self ):
        self._with_min_track_hits( 3 )
        self.assert_no_errors()


class TestGeometryFilterValidation( ConfigHarness ):

    def _with_limits( self, **keys ):
        backend = copy.deepcopy( betaconfig.detector_backend )
        name = backend['selected']
        backend[name].setdefault( 'item_overrides', {} ).setdefault(
            'exposed_breast', {} ).update( keys )
        self.set_config( 'detector_backend', backend )

    def test_an_area_fraction_above_one_is_an_error( self ):
        self._with_limits( max_area_fraction=1.5 )
        self.assert_error_mentioning( 'max_area_fraction' )

    def test_inverted_area_bounds_are_an_error( self ):
        # No box can satisfy both, so every detection of this label would
        # be filtered out - silently, without the filter looking broken.
        self._with_limits( min_area_fraction=0.5, max_area_fraction=0.1 )
        self.assert_error_mentioning( 'min_area_fraction', 'max_area_fraction' )

    def test_inverted_aspect_bounds_are_an_error( self ):
        self._with_limits( min_aspect_ratio=3.0, max_aspect_ratio=0.5 )
        self.assert_error_mentioning( 'aspect' )

    def test_a_negative_aspect_ratio_is_an_error( self ):
        self._with_limits( min_aspect_ratio=-1 )
        self.assert_error_mentioning( 'min_aspect_ratio' )

    def test_sensible_bounds_validate( self ):
        self._with_limits( min_area_fraction=0.0005, max_area_fraction=0.2,
                           min_aspect_ratio=0.3, max_aspect_ratio=4.0 )
        self.assert_no_errors()


class TestRenderAndEncodeValidation( ConfigHarness ):

    def test_negative_render_workers_is_an_error( self ):
        self.set_config( 'render_workers', -1 )
        self.assert_error_mentioning( 'render_workers' )

    def test_zero_render_workers_means_auto_and_validates( self ):
        self.set_config( 'render_workers', 0 )
        self.assert_no_errors()

    def test_an_unsupported_chunk_container_is_an_error( self ):
        self.set_config( 'render_chunk_container', 'avi' )
        self.assert_error_mentioning( 'render_chunk_container' )

    def test_an_out_of_range_crf_is_an_error( self ):
        self.set_config( 'encode_crf', 99 )
        self.assert_error_mentioning( 'encode_crf' )

    def test_an_even_blur_approximation_kernel_is_an_error( self ):
        # Gaussian kernel sizes are odd by definition.
        self.set_config( 'blur_approximation_min_kernel', 20 )
        self.assert_error_mentioning( 'blur_approximation_min_kernel' )

    def test_an_unknown_encode_preset_is_an_error( self ):
        self.set_config( 'encode_preset', 'turbo' )
        self.assert_error_mentioning( 'encode_preset' )


class TestCrossSizeDedupValidation( ConfigHarness ):

    def test_an_out_of_range_threshold_is_an_error( self ):
        self.set_config( 'cross_size_dedup', { 'enabled': True, 'iou_threshold': 1.5 } )
        self.assert_error_mentioning( 'iou_threshold' )

    def test_an_unknown_key_is_an_error( self ):
        self.set_config( 'cross_size_dedup', { 'enabled': True, 'iou_treshold': 0.6 } )
        self.assert_error_mentioning( 'cross_size_dedup' )

    def test_the_default_validates( self ):
        self.assert_no_errors()


class TestLoggingValidation( ConfigHarness ):

    def test_an_unknown_console_level_is_an_error( self ):
        self.set_config( 'console_level', 'verbose' )
        self.assert_error_mentioning( 'console_level' )

    def test_console_level_is_checked_even_with_file_logging_off( self ):
        # The console handler is attached whether or not file logging is
        # enabled, so its level always matters.
        self.set_config( 'logging_enabled', False )
        self.set_config( 'console_level', 'verbose' )
        self.assert_error_mentioning( 'console_level' )

    def test_every_documented_level_is_accepted( self ):
        for level in ( 'trace', 'debug', 'info', 'warn', 'error' ):
            self.set_config( 'log_level', level )
            self.set_config( 'console_level', level )
            self.assert_no_errors()


class TestMinProbFloorConsistency( ConfigHarness ):

    def test_a_label_min_prob_at_the_global_floor_is_an_error( self ):
        # global_min_prob is applied inside the adapter, before any
        # per-label min_prob ever sees a detection, so a per-label value
        # at or below it can never reject anything.
        backend = copy.deepcopy( betaconfig.detector_backend )
        name = backend['selected']
        backend[name].setdefault( 'item_overrides', {} ).setdefault(
            'exposed_breast', {} )['min_prob'] = betaconfig.global_min_prob
        self.set_config( 'detector_backend', backend )
        self.assert_error_mentioning( 'min_prob', 'global_min_prob' )

    def test_default_min_prob_at_the_floor_is_an_error( self ):
        self.set_config( 'default_min_prob', betaconfig.global_min_prob )
        self.assert_error_mentioning( 'default_min_prob' )


class TestResolvedLabelSettings( ConfigHarness ):

    def test_every_documented_key_is_present( self ):
        settings = bu_config.get_label_settings( 'exposed_breast' )
        for key in ( 'min_prob', 'min_prob_continue', 'width_area_safety',
                     'height_area_safety', 'time_safety', 'censor_style',
                     'censor_shape', 'position_smoothing', 'interpolation_enabled',
                     'interpolation_max_gap', 'track_max_gap', 'min_track_hits',
                     'paired_style', 'min_area_fraction', 'max_area_fraction',
                     'min_aspect_ratio', 'max_aspect_ratio' ):
            self.assertIn( key, settings )

    def test_parts_to_blur_is_memoised_and_invalidated( self ):
        first = bu_config.get_parts_to_blur()
        self.assertIs( bu_config.get_parts_to_blur(), first )
        bu_config.invalidate_config_caches()
        self.assertIsNot( bu_config.get_parts_to_blur(), first )

    def test_parts_to_blur_covers_exactly_the_censored_labels( self ):
        self.assertEqual( set( bu_config.get_parts_to_blur() ),
                          set( betaconfig.items_to_censor ) )

    def test_debug_mode_adds_an_entry_for_every_class( self ):
        self.set_config( 'debug_mode', 1 )
        parts = bu_config.get_parts_to_blur()
        for label in betaconst.classes:
            self.assertIn( label, parts )
            self.assertEqual( parts[label]['censor_style']['type'], 'debug' )


class TestCliOverrides( ConfigHarness ):

    def _parse( self, argv ):
        parser = bu_cli.build_arg_parser( 'test', include_preview=True, include_logging=True )
        return parser.parse_args( argv )

    def test_picture_sizes_lands_in_the_selected_backends_block( self ):
        # A flat setattr would be silently ignored by any backend that
        # sets its own value, which both shipped backends do.
        self.set_config( 'detector_backend', copy.deepcopy( betaconfig.detector_backend ) )
        args = self._parse( [ '--backend', 'nudenet_v3', '--picture-sizes', '640' ] )
        bu_cli.apply_cli_overrides( betaconfig, args )
        self.assertEqual( bu_detector.get_picture_sizes( 'nudenet_v3' ), [ 640 ] )

    def test_variant_override_lands_in_the_backends_block( self ):
        self.set_config( 'detector_backend', copy.deepcopy( betaconfig.detector_backend ) )
        args = self._parse( [ '--backend', 'nudenet_v3', '--variant', '640m' ] )
        bu_cli.apply_cli_overrides( betaconfig, args )
        self.assertEqual( bu_detector.selected_variant_name( 'nudenet_v3' ), '640m' )

    def test_backend_is_applied_before_backend_scoped_settings( self ):
        # Otherwise --nn-batch-size lands in the OLD backend's block.
        self.set_config( 'detector_backend', copy.deepcopy( betaconfig.detector_backend ) )
        args = self._parse( [ '--backend', 'retinanet_v2', '--nn-batch-size', '9' ] )
        bu_cli.apply_cli_overrides( betaconfig, args )
        self.assertEqual( bu_detector.get_nn_batch_size( 'retinanet_v2' ), 9 )

    def test_overrides_invalidate_the_memoised_config( self ):
        self.set_config( 'detector_backend', copy.deepcopy( betaconfig.detector_backend ) )
        before = bu_config.get_parts_to_blur()
        args = self._parse( [ '--video-censor-fps', '12' ] )
        bu_cli.apply_cli_overrides( betaconfig, args )
        self.assertIsNot( bu_config.get_parts_to_blur(), before )

    def test_unset_flags_change_nothing( self ):
        before = copy.deepcopy( betaconfig.detector_backend )
        args = self._parse( [] )
        applied = bu_cli.apply_cli_overrides( betaconfig, args )
        self.assertEqual( applied, [] )
        self.assertEqual( betaconfig.detector_backend, before )


class TestLoggingPlumbing( unittest.TestCase ):

    def setUp( self ):
        self.tmpdir = tempfile.mkdtemp( prefix='betasuite-log-' )
        self._saved = { name: getattr( betaconfig, name, None ) for name in
                        ( 'logging_enabled', 'log_path', 'log_level', 'console_level' ) }
        bu_log.reset_logger_for_tests()

    def tearDown( self ):
        bu_log.reset_logger_for_tests()
        for name, value in self._saved.items():
            if value is not None:
                setattr( betaconfig, name, value )
        shutil.rmtree( self.tmpdir, ignore_errors=True )

    def _configure( self, log_level='debug', console_level='error' ):
        betaconfig.logging_enabled = True
        betaconfig.log_path = os.path.join( self.tmpdir, 'test.log' )
        betaconfig.log_level = log_level
        betaconfig.console_level = console_level
        return bu_log.get_logger()

    def test_the_file_captures_detail_the_console_filters_out( self ):
        # The whole point of two independent levels: a readable terminal
        # and a log file that still has what you need afterwards.
        logger = self._configure( log_level='trace', console_level='error' )
        logger.trace( 'trace-marker' )
        logger.debug( 'debug-marker' )
        logger.info( 'info-marker' )
        for handler in logger.handlers:
            handler.flush()
        with open( betaconfig.log_path, 'r', encoding='UTF-8' ) as handle:
            contents = handle.read()
        self.assertIn( 'trace-marker', contents )
        self.assertIn( 'debug-marker', contents )
        self.assertIn( 'info-marker', contents )

    def test_the_file_level_is_honoured( self ):
        logger = self._configure( log_level='warn', console_level='error' )
        logger.info( 'info-marker' )
        logger.warning( 'warn-marker' )
        for handler in logger.handlers:
            handler.flush()
        with open( betaconfig.log_path, 'r', encoding='UTF-8' ) as handle:
            contents = handle.read()
        self.assertNotIn( 'info-marker', contents )
        self.assertIn( 'warn-marker', contents )

    def test_trace_is_available_on_the_logger( self ):
        logger = self._configure()
        self.assertTrue( hasattr( logger, 'trace' ) )

    def test_repeated_get_logger_calls_do_not_stack_handlers( self ):
        first = self._configure()
        handler_count = len( first.handlers )
        for _ in range( 5 ):
            bu_log.get_logger()
        self.assertEqual( len( bu_log.get_logger().handlers ), handler_count )

    def test_every_documented_level_resolves( self ):
        for name, expected in bu_log.LEVEL_NAMES.items():
            self.assertEqual( bu_log.resolve_level( name ), expected )
        self.assertEqual( bu_log.resolve_level( 'nonsense' ), logging.INFO )


class TestProgressReporter( unittest.TestCase ):
    """
    The pre-2.1 loops printed once per frame. Piped to a log that is one
    more copy of the whole line per frame, which is how a 90-second
    video produced a 16KB single-line log.
    """

    class _CountingLogger:
        def __init__( self ):
            self.records = []

        def log( self, level, message ):
            self.records.append( ( level, message ) )

        def info( self, message ):
            self.records.append( ( logging.INFO, message ) )

    def test_a_thousand_updates_produce_at_most_a_handful_of_records( self ):
        logger = self._CountingLogger()
        progress = bu_log.ProgressReporter( 'test', total=1000, logger=logger,
                                            log_interval=3600 )
        for index in range( 1000 ):
            progress.update( index + 1 )
        progress.finish()
        self.assertLessEqual( len( logger.records ), 3,
            "the progress reporter emitted a record per update: %d"%(len( logger.records )) )

    def test_finish_emits_exactly_one_summary( self ):
        logger = self._CountingLogger()
        progress = bu_log.ProgressReporter( 'test', total=10, logger=logger,
                                            log_interval=3600 )
        progress.update( 10 )
        progress.finish()
        self.assertEqual( len( logger.records ), 1 )
        self.assertIn( 'done', logger.records[0][1] )

    def test_advance_accumulates_across_threads( self ):
        import threading
        logger = self._CountingLogger()
        progress = bu_log.ProgressReporter( 'test', total=400, logger=logger,
                                            log_interval=3600 )

        def worker():
            for _ in range( 100 ):
                progress.advance()

        threads = [ threading.Thread( target=worker ) for _ in range( 4 ) ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        progress.finish()
        self.assertIn( '400', logger.records[-1][1] )

    def test_it_works_as_a_context_manager( self ):
        logger = self._CountingLogger()
        with bu_log.ProgressReporter( 'test', total=5, logger=logger,
                                      log_interval=3600 ) as progress:
            progress.update( 5 )
        self.assertEqual( len( logger.records ), 1 )

    def test_duration_formatting_is_readable_at_every_scale( self ):
        self.assertEqual( bu_log.format_duration( 5 ), '5.0s' )
        self.assertEqual( bu_log.format_duration( 125 ), '2m05s' )
        self.assertEqual( bu_log.format_duration( 7300 ), '2h01m' )


class TestStatsWriting( unittest.TestCase ):

    def setUp( self ):
        self.tmpdir = tempfile.mkdtemp( prefix='betasuite-stats-' )
        self._saved = { name: getattr( betaconfig, name, None )
                        for name in ( 'stats_enabled', 'stats_path' ) }
        betaconfig.stats_enabled = True
        betaconfig.stats_path = os.path.join( self.tmpdir, 'stats.jsonl' )

    def tearDown( self ):
        for name, value in self._saved.items():
            if value is not None:
                setattr( betaconfig, name, value )
        shutil.rmtree( self.tmpdir, ignore_errors=True )

    def test_a_record_is_appended_as_one_json_line( self ):
        import json
        bu_log.write_stats( { 'file': 'a.mp4', 'total_seconds': 1.0 } )
        bu_log.write_stats( { 'file': 'b.mp4', 'total_seconds': 2.0 } )
        with open( betaconfig.stats_path, 'r', encoding='UTF-8' ) as handle:
            rows = [ json.loads( line ) for line in handle if line.strip() ]
        self.assertEqual( [ row['file'] for row in rows ], [ 'a.mp4', 'b.mp4' ] )
        self.assertIn( 'timestamp', rows[0] )

    def test_the_callers_dict_is_never_mutated( self ):
        record = { 'file': 'a.mp4' }
        bu_log.write_stats( record )
        self.assertEqual( record, { 'file': 'a.mp4' } )

    def test_disabled_stats_write_nothing( self ):
        betaconfig.stats_enabled = False
        bu_log.write_stats( { 'file': 'a.mp4' } )
        self.assertFalse( os.path.exists( betaconfig.stats_path ) )

    def test_a_write_failure_is_swallowed( self ):
        # Stats are a nice-to-have and must never abort a censoring run.
        betaconfig.stats_path = os.path.join( self.tmpdir, 'nope', '\0bad', 'x.jsonl' )
        bu_log.write_stats( { 'file': 'a.mp4' } )   # must not raise


if __name__ == '__main__':
    unittest.main()
