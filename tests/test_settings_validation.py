"""
test_settings_validation.py - validation reads the settings registry,
and every error it reports is a code with params.

The engine never sends sentences (protocol/README.md): config.set
reports a rejected value by its error's code and params, and the app
words it. So every error validation returns must carry a code, and a
cross-field rule a stable id. And the limits must come from the
registry itself, not from a copy that can drift from what the app shows
on a slider: changing an entry's limit here must change what
validation accepts.
"""

import copy
import re
import unittest

import betaconfig
import betautils_config as bu_config
import betautils_settings as bu_settings

from tests import config_harness as harness
from tests import test_settings_cache_tags as tags
from tests import test_validation_corpus as corpus

REGISTRY_CODES = { 'config.bad_type', 'config.out_of_range', 'config.bad_choice' }
CODES = REGISTRY_CODES | { 'config.rule' }
RULE_ID = re.compile( r'^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$' )


def _errors():
    bu_config.invalidate_config_caches()
    return bu_config._collect_validation_errors()


class TestEveryErrorIsACode( unittest.TestCase ):

    def assert_coded( self, errors, where ):
        for error in errors:
            self.assertIsInstance( error, bu_settings.Problem, ( where, error ) )
            self.assertIn( error.code, CODES, ( where, error ) )
            self.assertIn( 'path', error.params, ( where, error ) )
            if error.code == 'config.rule':
                self.assertRegex( error.params['rule'], RULE_ID, ( where, error ) )

    def test_every_corpus_error_has_a_code_and_a_path( self ):
        with harness.shipped():
            baseline = set( _errors() )
            for path in harness.shipped_leaf_paths():
                for bad in corpus.bad_values( harness.get_path( path ) ):
                    with harness.preserved():
                        harness.set_path( path, bad )
                        errors = [ error for error in _errors() if error not in baseline ]
                    self.assert_coded( errors, corpus.case_name( path, bad ) )

    def test_a_registry_error_names_the_dotted_path_it_is_about( self ):
        with harness.shipped():
            betaconfig.detector_backend['nudenet_v3']['item_overrides']['exposed_breast']['min_prob'] = 7
            errors = [ error for error in _errors() if error.code == 'config.out_of_range' ]
        self.assertEqual( [ 'detector_backend.nudenet_v3.item_overrides.exposed_breast.min_prob' ],
                          [ error.params['path'] for error in errors ] )

    def test_a_rule_error_names_its_rule_and_path( self ):
        with harness.shipped():
            rules = betaconfig.detector_backend['nudenet_v3']['class_suppression']['exposed_breast']
            rules.append( dict( rules[0] ) )
            errors = [ error for error in _errors() if error.code == 'config.rule' ]
        self.assertEqual( [ ( 'class_suppression.duplicate_suppressor',
                              'detector_backend.nudenet_v3.class_suppression.exposed_breast' ) ],
                          [ ( error.params['rule'], error.params['path'] ) for error in errors ] )

    def test_located_reads_the_path_a_message_starts_with( self ):
        self.assertEqual( 'detector_backend.nudenet_v3.nms_iou',
                          bu_settings.located( "detector_backend['nudenet_v3']['nms_iou'] must be ..." ) )
        self.assertEqual( 'nn_batch_size', bu_settings.located( "betaconfig.nn_batch_size is no longer read" ) )
        self.assertEqual( 'detector_backend.nudenet_v3.item_overrides.exposed_breast.min_prob_continue',
                          bu_settings.located( "detector_backend['nudenet_v3']'s resolved "
                                               "item_overrides['exposed_breast']['min_prob_continue'] (0.5) is ABOVE" ) )
        self.assertEqual( '', bu_settings.located( "%%s is wrong" ) )


class TestACrashIsNotHidden( unittest.TestCase ):
    """
    A check that crashes after the registry reported a wrong-typed value
    is skipped, since the value's error is already listed. Without any
    registry error a crash is a bug in validation and must surface.
    """

    def test_a_crash_with_a_clean_registry_pass_is_raised( self ):
        import unittest.mock as mock
        with harness.shipped(), mock.patch.object( bu_config, '_validate_stats_settings',
                                                   side_effect=RuntimeError( 'bug' ) ):
            with self.assertRaises( RuntimeError ):
                _errors()

    def test_a_crash_after_a_registry_error_is_skipped( self ):
        import unittest.mock as mock
        with harness.shipped(), mock.patch.object( bu_config, '_validate_stats_settings',
                                                   side_effect=RuntimeError( 'bug' ) ):
            betaconfig.encode_crf = 'x'
            self.assertEqual( [ 'config.bad_type' ], [ error.code for error in _errors() ] )


class TestValidationReadsTheRegistry( unittest.TestCase ):
    """Changing an entry changes what validation accepts: there is no second copy of the limit."""

    def entry( self, path ):
        return bu_settings.find( path )

    def test_a_range_comes_from_the_entry( self ):
        entry = self.entry( 'encode_crf' )
        saved = entry.maximum
        try:
            with harness.shipped():
                betaconfig.encode_crf = 40
                self.assertEqual( [], _errors() )
                entry.maximum = 30
                self.assertEqual( [ 'config.out_of_range' ], [ error.code for error in _errors() ] )
        finally:
            entry.maximum = saved

    def test_choices_come_from_the_entry( self ):
        entry = self.entry( 'render_chunk_container' )
        saved = entry.choices
        try:
            with harness.shipped():
                betaconfig.render_chunk_container = 'mp4'
                self.assertEqual( [], _errors() )
                entry.choices = ( 'mkv', )
                self.assertEqual( [ 'config.bad_choice' ], [ error.code for error in _errors() ] )
        finally:
            entry.choices = saved

    def test_a_per_label_range_comes_from_the_entry( self ):
        entry = self.entry( 'detector_backend.nudenet_v3.item_overrides.exposed_breast.min_track_hits' )
        saved = entry.maximum
        try:
            with harness.shipped():
                self.assertEqual( [], _errors() )
                entry.maximum = 1
                codes = { error.code for error in _errors() }
            self.assertEqual( { 'config.out_of_range' }, codes )
        finally:
            entry.maximum = saved

    def test_a_style_field_range_comes_from_the_table( self ):
        spec = bu_settings.STYLE_COMMON_FIELDS['feather']
        saved = copy.deepcopy( spec )
        try:
            with harness.shipped():
                self.assertEqual( [], _errors() )
                spec['maximum'] = 0.05
                errors = _errors()
            self.assertTrue( errors )
            self.assertEqual( { 'config.out_of_range' }, { error.code for error in errors } )
            self.assertTrue( all( "['feather']" in error for error in errors ) )
        finally:
            spec.clear()
            spec.update( saved )

    def test_a_rule_field_range_comes_from_the_table( self ):
        spec = bu_settings.RULE_FIELDS['suppression']['min_iou']
        saved = copy.deepcopy( spec )
        try:
            with harness.shipped():
                self.assertEqual( [], _errors() )
                spec['maximum'] = 0.1
                errors = _errors()
            self.assertTrue( errors )
            self.assertTrue( all( "['min_iou']" in error for error in errors ) )
        finally:
            spec.clear()
            spec.update( saved )

    def test_the_old_constants_are_the_registry_vocabulary( self ):
        self.assertEqual( set( bu_settings.CENSOR_SHAPES ), bu_config.VALID_CENSOR_SHAPES )
        self.assertEqual( set( bu_settings.CENSOR_STYLE_TYPES ), bu_config.VALID_CENSOR_STYLES )
        self.assertEqual( set( bu_settings.BLUR_METHODS ), bu_config.VALID_BLUR_METHODS )
        self.assertEqual( set( bu_settings.PIXEL_PATTERNS ), bu_config.VALID_PIXEL_PATTERNS )
        self.assertEqual( set( bu_settings.MERGE_STRATEGIES ), bu_config.VALID_MERGE_STRATEGIES )

    def test_nms_modes_match_the_adapter( self ):
        import detectors.nudenet_v3 as nudenet_v3
        self.assertEqual( set( nudenet_v3.VALID_NMS_MODES ),
                          set( self.entry( 'detector_backend.nudenet_v3.nms_mode' ).choice_list() ) )

    def test_label_vocabularies_match_the_code( self ):
        import betautils_detector as bu_detector
        import betautils_reverse as bu_reverse
        import betautils_track as bu_track
        self.assertEqual( set( bu_track.GEOMETRY_ACTIONS ), set( bu_settings.geometry_actions() ) )
        self.assertEqual( set( bu_reverse.EMPTY_FRAME_POLICIES ),
                          set( self.entry( 'reverse.empty_frame' ).choice_list() ) )
        self.assertEqual( set( bu_track.PROMOTION_RULE_KEYS ), set( bu_settings.RULE_FIELDS['promotion'] ) )
        self.assertEqual( set( bu_track.PROMOTION_EVIDENCE_KEYS ),
                          set( bu_settings.RULE_FIELDS['promotion_requirement'] ) )
        self.assertEqual( bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS, set( bu_settings.LABEL_FIELDS ) )
        self.assertEqual( bu_config.COMMON_STYLE_KEYS, set( bu_settings.STYLE_COMMON_FIELDS ) )
        for style_type, keys in bu_config.TYPE_SPECIFIC_KEYS.items():
            self.assertEqual( keys, set( bu_settings.STYLE_TYPE_FIELDS[style_type] ), style_type )


class TestEveryProbeValidates( unittest.TestCase ):
    """
    The probe each entry is changed to in the cache-tag test passes the
    registry's own checks in the context that test uses, so the tags are
    proven on a configuration a run would accept.
    """

    def test_no_probe_trips_a_registry_check( self ):
        wrong = []
        for entry in bu_settings.REGISTRY:
            for path, bindings in tags.instantiations( entry ):
                with harness.shipped():
                    for condition in entry.when:
                        tags.set_up( condition, bindings, True, path )
                    try:
                        current = harness.get_path( path )
                    except ( KeyError, IndexError, TypeError ):
                        current = tags._MISSING
                    harness.set_path( path, tags.probe_for( entry, path, current, bindings ) )
                    dotted = '.'.join( path )
                    errors = [ error for error in _errors()
                               if error.code in REGISTRY_CODES and error.params['path'].startswith( dotted ) ]
                if errors:
                    wrong.append( '%s: %s'%( dotted, errors[0] ) )
        self.assertEqual( [], wrong )


if __name__ == '__main__':
    unittest.main()
