"""
test_track_pipeline.py - the detection -> renderable-boxes stages.

Covers betautils_track's four filters and the tracking rules, each
of which can remove a detection and therefore has to be
provably conservative when it is off:

    apply_cross_size_dedup      no-op with one picture size
    apply_geometry_filter       no-op when no bounds are configured
    apply_class_suppression     unchanged behaviour, honest counts
    smooth_boxes hysteresis     no-op when min_prob_continue is unset
    smooth_boxes confirmation   no-op when min_track_hits is 1
    smooth_boxes track pruning  identical results to not pruning

The "no-op when off" cases matter as much as the feature cases: a
performance or tuning change that quietly drops a real detection is the
one failure mode this project cannot accept.
"""

import copy
import unittest

import betaconfig
import betautils_censor as bu_censor
import betautils_config as bu_config
import betautils_track as bu_track


def raw_box( label, t, x=100, y=100, w=50, h=50, score=0.9, size=320 ):
    """A raw detection, as a detector adapter would emit it."""
    return { 'x': float( x ), 'y': float( y ), 'w': float( w ), 'h': float( h ),
             'class_id': label, 'score': score, 't': t, 'size': size }


def censorable_box( label, t, x=100, y=100, w=50, h=50, score=0.9,
                    time_safety=0.3, provisional=False, style=None ):
    """A censorable box, as betautils_censor.process_raw_box would emit it."""
    return {
        'start': max( t - time_safety/2, 0 ), 'end': t + time_safety/2, 't': t,
        'x': x, 'y': y, 'w': w, 'h': h,
        '_raw_x': float( x ), '_raw_y': float( y ),
        '_raw_w': float( w ), '_raw_h': float( h ),
        '_vid_w': 1920, '_vid_h': 1080,
        'censor_style': style or { 'type': 'blur', 'method': 'gaussian', 'strength': 20 },
        'censor_shape': 'box',
        'censor_sticker_seed': 0.5,
        'label': label, 'score': score, 'size': 320,
        'provisional': provisional,
    }


class TestIntersectionOverUnion( unittest.TestCase ):

    def test_identical_boxes_have_iou_one( self ):
        box = { 'x': 0, 'y': 0, 'w': 10, 'h': 10 }
        self.assertAlmostEqual( bu_track.intersection_over_union( box, dict( box ) ), 1.0 )

    def test_disjoint_boxes_have_iou_zero( self ):
        self.assertEqual( bu_track.intersection_over_union(
            { 'x': 0, 'y': 0, 'w': 10, 'h': 10 },
            { 'x': 100, 'y': 100, 'w': 10, 'h': 10 } ), 0.0 )

    def test_half_overlap( self ):
        # Two 10x10 boxes sharing a 5x10 strip: intersection 50,
        # union 150, IoU 1/3.
        self.assertAlmostEqual( bu_track.intersection_over_union(
            { 'x': 0, 'y': 0, 'w': 10, 'h': 10 },
            { 'x': 5, 'y': 0, 'w': 10, 'h': 10 } ), 1/3, places=6 )

    def test_zero_area_box_has_iou_zero( self ):
        self.assertEqual( bu_track.intersection_over_union(
            { 'x': 0, 'y': 0, 'w': 0, 'h': 10 },
            { 'x': 0, 'y': 0, 'w': 10, 'h': 10 } ), 0.0 )


class TestCrossSizeDedup( unittest.TestCase ):

    def test_is_a_no_op_with_a_single_picture_size( self ):
        # The common configuration. Two same-label detections from ONE
        # size are what a detector legitimately produces for two real
        # instances, and must never be merged.
        boxes = [ raw_box( 'exposed_breast', 1.0, x=100, size=320 ),
                  raw_box( 'exposed_breast', 1.0, x=102, size=320 ) ]
        survivors, counts = bu_track.apply_cross_size_dedup( boxes )
        self.assertEqual( len( survivors ), 2 )
        self.assertEqual( counts, {} )

    def test_merges_the_same_object_seen_at_two_sizes( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, x=100, score=0.9, size=320 ),
                  raw_box( 'exposed_breast', 1.0, x=101, score=0.7, size=640 ) ]
        survivors, counts = bu_track.apply_cross_size_dedup( boxes )
        self.assertEqual( len( survivors ), 1 )
        self.assertEqual( survivors[0]['score'], 0.9, "the more confident detection must survive" )
        self.assertEqual( counts, { 'exposed_breast': 1 } )

    def test_keeps_distant_detections_from_different_sizes( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, x=100, size=320 ),
                  raw_box( 'exposed_breast', 1.0, x=900, size=640 ) ]
        survivors, _counts = bu_track.apply_cross_size_dedup( boxes )
        self.assertEqual( len( survivors ), 2 )

    def test_never_merges_different_labels( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, size=320 ),
                  raw_box( 'exposed_vulva', 1.0, size=640 ) ]
        survivors, _counts = bu_track.apply_cross_size_dedup( boxes )
        self.assertEqual( len( survivors ), 2 )

    def test_never_merges_across_timestamps( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, size=320 ),
                  raw_box( 'exposed_breast', 2.0, size=640 ) ]
        survivors, _counts = bu_track.apply_cross_size_dedup( boxes )
        self.assertEqual( len( survivors ), 2 )

    def test_can_be_disabled( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, size=320 ),
                  raw_box( 'exposed_breast', 1.0, size=640 ) ]
        survivors, counts = bu_track.apply_cross_size_dedup(
            boxes, { 'enabled': False } )
        self.assertEqual( len( survivors ), 2 )
        self.assertEqual( counts, {} )


class TestGeometryFilter( unittest.TestCase ):

    def setUp( self ):
        self.original = copy.deepcopy( betaconfig.detector_backend )
        self._clear_configured_limits()

    def tearDown( self ):
        betaconfig.detector_backend = self.original
        bu_config.invalidate_config_caches()

    GEOMETRY_KEYS = ( 'min_area_fraction', 'max_area_fraction',
                      'min_aspect_ratio', 'max_aspect_ratio' )

    def _clear_configured_limits( self ):
        """
        Start each test from "no geometry bounds anywhere".

        These tests assert what the filter does for a GIVEN set of
        bounds, including the empty set. Reading whatever the live
        config happens to have tuned made them pass only while the
        shipped config left the filter disabled - the moment real bounds
        were derived and applied, 'is a no-op when nothing is
        configured' started failing against bounds that were, in fact,
        configured. The test was right and its fixture was incomplete.
        """
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        for block in betaconfig.detector_backend.values():
            if not isinstance( block, dict ):
                continue
            for label_overrides in block.get( 'item_overrides', {} ).values():
                if isinstance( label_overrides, dict ):
                    for key in self.GEOMETRY_KEYS:
                        label_overrides.pop( key, None )
        bu_config.invalidate_config_caches()

    def _set_limits( self, label, **limits ):
        backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.detector_backend[backend].setdefault(
            'item_overrides', {} ).setdefault( label, {} ).update( limits )
        bu_config.invalidate_config_caches()

    def test_is_a_no_op_when_nothing_is_configured( self ):
        # Every bound defaults to None, so a config that has not opted in
        # can never lose a detection to this filter.
        boxes = [ raw_box( 'exposed_breast', 1.0, w=1, h=1 ),
                  raw_box( 'exposed_breast', 1.0, w=1900, h=1000 ),
                  raw_box( 'exposed_breast', 1.0, w=500, h=5 ) ]
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( len( survivors ), 3 )
        self.assertEqual( counts, {} )

    def test_drops_a_box_covering_too_much_of_the_frame( self ):
        # geometry_action pinned to drop: this test is about the BOUND, and
        # the shipped default for a max bound is now clamp (see
        # TestGeometryActions for why).
        self._set_limits( 'exposed_vulva', max_area_fraction=0.05,
                          geometry_action='drop' )
        boxes = [ raw_box( 'exposed_vulva', 1.0, w=100, h=100 ),      # 0.005
                  raw_box( 'exposed_vulva', 1.0, w=1200, h=800 ) ]    # 0.46
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( len( survivors ), 1 )
        self.assertEqual( survivors[0]['w'], 100 )
        self.assertEqual( counts, { 'exposed_vulva:too_large:drop': 1 } )

    def test_rejects_a_box_that_is_too_small( self ):
        # No pin needed: min bounds still default to drop, because a 4x4
        # box carries no coverage worth keeping.
        self._set_limits( 'exposed_vulva', min_area_fraction=0.001 )
        boxes = [ raw_box( 'exposed_vulva', 1.0, w=4, h=4 ),
                  raw_box( 'exposed_vulva', 1.0, w=200, h=200 ) ]
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( len( survivors ), 1 )
        self.assertEqual( counts, { 'exposed_vulva:too_small:drop': 1 } )

    def test_rejects_an_implausible_aspect_ratio( self ):
        self._set_limits( 'exposed_vulva', min_aspect_ratio=0.3, max_aspect_ratio=3.0,
                          geometry_action='drop' )
        boxes = [ raw_box( 'exposed_vulva', 1.0, w=100, h=100 ),   # 1.0
                  raw_box( 'exposed_vulva', 1.0, w=600, h=20 ),    # 30.0
                  raw_box( 'exposed_vulva', 1.0, w=20, h=600 ) ]   # 0.033
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( len( survivors ), 1 )
        self.assertEqual( counts, { 'exposed_vulva:too_wide:drop': 1,
                                    'exposed_vulva:too_tall:drop': 1 } )

    def test_limits_are_per_label( self ):
        self._set_limits( 'exposed_vulva', max_area_fraction=0.05,
                          geometry_action='drop' )
        boxes = [ raw_box( 'exposed_vulva', 1.0, w=1200, h=800 ),
                  raw_box( 'exposed_breast', 1.0, w=1200, h=800 ) ]
        survivors, _counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( [ box['class_id'] for box in survivors ], [ 'exposed_breast' ] )


class TestGeometryActions( unittest.TestCase ):
    """
    What a geometry violation DOES, not just which boxes violate.

    THE BUG THIS FIXES
    ------------------
    A max_area_fraction bound cannot distinguish a correct close-up from a
    torso misfire, because both are simply large. Dropping on violation
    therefore means a legitimate close-up gets NO CENSOR AT ALL - which is
    exactly why geometry was switched off and left off.

    Measured with exposed_breast's own suggested cap (0.123238, betabench
    p99 x1.25) on 1080p:

        normal breast   1.1% of frame   kept
        p99 breast     10.3%            kept
        close-up       32.4%            DROPPED  <- legitimate, uncensored
        torso misfire  40.0%            DROPPED  <- the intended catch

    And on a real 10-file run the largest detection of every censored
    label sat 2.0-4.1x above its suggested cap, so this is the common
    case, not an edge one.

    Clamping keeps coverage: a clamped misfire is a roughly correctly
    sized censor near the right place, a clamped close-up is censored
    slightly small. Both beat nothing.
    """

    def setUp( self ):
        self._saved = copy.deepcopy( betaconfig.detector_backend )

    def tearDown( self ):
        betaconfig.detector_backend = self._saved
        bu_config.invalidate_config_caches()

    def _set_limits( self, label, **limits ):
        backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.detector_backend[backend].setdefault(
            'item_overrides', {} ).setdefault( label, {} ).update( limits )
        bu_config.invalidate_config_caches()

    def test_a_max_area_violation_clamps_by_default( self ):
        self._set_limits( 'exposed_breast', max_area_fraction=0.123238 )
        # The legitimate close-up from the table above.
        boxes = [ raw_box( 'exposed_breast', 1.0, w=820, h=820 ) ]
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( len( survivors ), 1, "a close-up must still be censored" )
        self.assertEqual( counts, { 'exposed_breast:too_large:clamp': 1 } )
        self.assertTrue( survivors[0]['_geometry_clamped'] )

    def test_a_clamped_box_lands_exactly_on_the_limit( self ):
        self._set_limits( 'exposed_breast', max_area_fraction=0.05 )
        boxes = [ raw_box( 'exposed_breast', 1.0, w=1200, h=800 ) ]
        survivors, _counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        area = survivors[0]['w'] * survivors[0]['h']
        self.assertAlmostEqual( area / ( 1920*1080 ), 0.05, places=4 )

    def test_a_clamped_box_keeps_its_centre( self ):
        # The censor has to stay on the thing it was covering.
        self._set_limits( 'exposed_breast', max_area_fraction=0.05 )
        box = raw_box( 'exposed_breast', 1.0, w=1200, h=800 )
        box['x'], box['y'] = 300, 200
        centre = ( box['x'] + box['w']/2.0, box['y'] + box['h']/2.0 )
        survivors, _counts = bu_track.apply_geometry_filter( [ box ], 1920, 1080 )
        got = ( survivors[0]['x'] + survivors[0]['w']/2.0,
                survivors[0]['y'] + survivors[0]['h']/2.0 )
        self.assertAlmostEqual( got[0], centre[0], places=3 )
        self.assertAlmostEqual( got[1], centre[1], places=3 )

    def test_a_clamped_box_keeps_its_aspect_ratio( self ):
        # Scaling both sides by one factor, so the clamp does not reshape
        # what the detector found.
        self._set_limits( 'exposed_breast', max_area_fraction=0.05 )
        box = raw_box( 'exposed_breast', 1.0, w=1200, h=600 )
        survivors, _counts = bu_track.apply_geometry_filter( [ box ], 1920, 1080 )
        self.assertAlmostEqual( survivors[0]['w'] / survivors[0]['h'],
                                1200/600.0, places=3 )

    def test_a_clamped_box_stays_inside_the_frame( self ):
        self._set_limits( 'exposed_breast', max_area_fraction=0.05 )
        box = raw_box( 'exposed_breast', 1.0, w=1900, h=1000 )
        box['x'], box['y'] = 10, 10
        survivors, _counts = bu_track.apply_geometry_filter( [ box ], 1920, 1080 )
        got = survivors[0]
        self.assertGreaterEqual( got['x'], 0 )
        self.assertGreaterEqual( got['y'], 0 )
        self.assertLessEqual( got['x'] + got['w'], 1920 )
        self.assertLessEqual( got['y'] + got['h'], 1080 )

    def test_a_too_wide_box_clamps_its_width( self ):
        self._set_limits( 'exposed_breast', max_aspect_ratio=3.0 )
        boxes = [ raw_box( 'exposed_breast', 1.0, w=600, h=20 ) ]   # 30.0
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( counts, { 'exposed_breast:too_wide:clamp': 1 } )
        self.assertAlmostEqual( survivors[0]['w'] / survivors[0]['h'], 3.0, places=3 )

    def test_a_min_violation_still_drops_under_the_default( self ):
        # Min and max take different actions by default, from one setting.
        self._set_limits( 'exposed_breast', min_area_fraction=0.001,
                          max_area_fraction=0.05 )
        boxes = [ raw_box( 'exposed_breast', 1.0, w=4, h=4 ),
                  raw_box( 'exposed_breast', 1.0, w=1200, h=800 ) ]
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( len( survivors ), 1, "the tiny box drops, the big one clamps" )
        self.assertEqual( counts, { 'exposed_breast:too_small:drop': 1,
                                    'exposed_breast:too_large:clamp': 1 } )

    def test_flag_keeps_the_box_unchanged_and_counts_it( self ):
        # For measuring a candidate bound before trusting it.
        self._set_limits( 'exposed_breast', max_area_fraction=0.05,
                          geometry_action='flag' )
        boxes = [ raw_box( 'exposed_breast', 1.0, w=1200, h=800 ) ]
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( survivors[0]['w'], 1200 )
        self.assertEqual( survivors[0]['h'], 800 )
        self.assertEqual( counts, { 'exposed_breast:too_large:flag': 1 } )

    def test_a_per_direction_action_mapping_is_honoured( self ):
        self._set_limits( 'exposed_breast', min_area_fraction=0.001,
                          max_area_fraction=0.05,
                          geometry_action={ 'min': 'flag', 'max': 'drop' } )
        boxes = [ raw_box( 'exposed_breast', 1.0, w=4, h=4 ),
                  raw_box( 'exposed_breast', 1.0, w=1200, h=800 ) ]
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( [ box['w'] for box in survivors ], [ 4 ] )
        self.assertEqual( counts, { 'exposed_breast:too_small:flag': 1,
                                    'exposed_breast:too_large:drop': 1 } )

    def test_an_unclampable_violation_keeps_the_box_rather_than_deleting_it( self ):
        # 'too_small' cannot be fixed by shrinking. Under clamp - which
        # means "never delete coverage" - the honest answer is to leave the
        # box alone and let the count record it, not to silently drop.
        self._set_limits( 'exposed_breast', min_area_fraction=0.001,
                          geometry_action='clamp' )
        boxes = [ raw_box( 'exposed_breast', 1.0, w=4, h=4 ) ]
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( len( survivors ), 1 )
        self.assertEqual( survivors[0]['w'], 4 )
        self.assertEqual( counts, { 'exposed_breast:too_small:clamp': 1 } )

    def test_the_input_boxes_are_never_mutated( self ):
        # The caller's list is shared with the cache reader.
        self._set_limits( 'exposed_breast', max_area_fraction=0.05 )
        box = raw_box( 'exposed_breast', 1.0, w=1200, h=800 )
        before = dict( box )
        bu_track.apply_geometry_filter( [ box ], 1920, 1080 )
        self.assertEqual( box, before )

    def test_an_unknown_action_falls_back_to_drop( self ):
        self._set_limits( 'exposed_breast', max_area_fraction=0.05,
                          geometry_action='shrinkify' )
        boxes = [ raw_box( 'exposed_breast', 1.0, w=1200, h=800 ) ]
        survivors, counts = bu_track.apply_geometry_filter( boxes, 1920, 1080 )
        self.assertEqual( survivors, [] )
        self.assertEqual( counts, { 'exposed_breast:too_large:drop': 1 } )


class TestClassSuppression( unittest.TestCase ):

    RULES = { 'exposed_breast': [
        { 'suppressed_by': 'covered_breast', 'margin': 0.10, 'min_iou': 0.30 } ] }

    def test_suppresses_a_lower_scoring_overlapping_detection( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, x=100, y=100, w=50, h=50, score=0.5 ),
                  raw_box( 'covered_breast', 1.0, x=100, y=100, w=50, h=50, score=0.8 ) ]
        survivors, counts = bu_track.apply_class_suppression( boxes, self.RULES )
        self.assertEqual( [ box['class_id'] for box in survivors ], [ 'covered_breast' ] )
        self.assertEqual( counts['exposed_breast<-covered_breast']['total'], 1 )

    def test_respects_the_margin( self ):
        # 0.05 apart, margin is 0.10: not enough better evidence to win.
        boxes = [ raw_box( 'exposed_breast', 1.0, score=0.75 ),
                  raw_box( 'covered_breast', 1.0, score=0.80 ) ]
        survivors, counts = bu_track.apply_class_suppression( boxes, self.RULES )
        self.assertEqual( len( survivors ), 2 )
        self.assertEqual( counts, {} )

    def test_respects_min_iou( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, x=100, score=0.5 ),
                  raw_box( 'covered_breast', 1.0, x=900, score=0.9 ) ]
        survivors, _counts = bu_track.apply_class_suppression( boxes, self.RULES )
        self.assertEqual( len( survivors ), 2 )

    def test_never_compares_across_timestamps( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, score=0.5 ),
                  raw_box( 'covered_breast', 2.0, score=0.9 ) ]
        survivors, _counts = bu_track.apply_class_suppression( boxes, self.RULES )
        self.assertEqual( len( survivors ), 2 )

    def test_does_not_mutate_the_input( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0, score=0.5 ),
                  raw_box( 'covered_breast', 1.0, score=0.9 ) ]
        before = copy.deepcopy( boxes )
        bu_track.apply_class_suppression( boxes, self.RULES )
        self.assertEqual( boxes, before )

    def test_counts_separate_renderable_from_total( self ):
        # A suppression that removed something already below the label's
        # own min_prob did no real work. Reporting only the total
        # overstated how much each rule was contributing.
        parts = { 'exposed_breast': { 'min_prob': 0.60 } }
        boxes = [
            raw_box( 'exposed_breast', 1.0, score=0.30 ),   # below min_prob
            raw_box( 'covered_breast', 1.0, score=0.95 ),
            raw_box( 'exposed_breast', 2.0, score=0.70 ),   # above min_prob
            raw_box( 'covered_breast', 2.0, score=0.95 ),
        ]
        _survivors, counts = bu_track.apply_class_suppression( boxes, self.RULES, parts )
        entry = counts['exposed_breast<-covered_breast']
        self.assertEqual( entry['total'], 2 )
        self.assertEqual( entry['renderable'], 1 )

    def test_no_rules_is_a_pass_through( self ):
        boxes = [ raw_box( 'exposed_breast', 1.0 ) ]
        survivors, counts = bu_track.apply_class_suppression( boxes, {} )
        self.assertIs( survivors, boxes )
        self.assertEqual( counts, {} )


class TestRelevantLabels( unittest.TestCase ):

    def test_keeps_censored_labels_and_suppressors_only( self ):
        rules = { 'exposed_breast': [
            { 'suppressed_by': 'face_femme', 'margin': 0.1, 'min_iou': 0.3 } ] }
        relevant = bu_track.relevant_labels_for_suppression( rules, [ 'exposed_vulva' ] )
        self.assertEqual( relevant, { 'exposed_vulva', 'exposed_breast', 'face_femme' } )

    def test_handles_a_single_rule_dict( self ):
        rules = { 'exposed_breast': { 'suppressed_by': 'face_femme',
                                      'margin': 0.1, 'min_iou': 0.3 } }
        relevant = bu_track.relevant_labels_for_suppression( rules, [] )
        self.assertEqual( relevant, { 'exposed_breast', 'face_femme' } )


class _ConfirmationPinnedMixin:
    """
    Pin min_track_hits to 1 for tests that are not about confirmation.

    These fixtures are one or two boxes in one or two frames. The
    shipped config sets min_track_hits: 2, which drops any track with
    fewer real detections than that - so every such fixture renders
    nothing and the assertion fails for a reason unrelated to what the
    test is checking. Reading the live value made these tests silently
    couple to a tuning decision.

    Confirmation itself is covered by the tests in TestTrackConfirmation
    that set the value deliberately, and those override this pin.
    """

    def setUp( self ):
        self._pin_orig_default = betaconfig.default_min_track_hits
        self._pin_orig_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.default_min_track_hits = 1
        for block in betaconfig.detector_backend.values():
            if not isinstance( block, dict ):
                continue
            for label_overrides in block.get( 'item_overrides', {} ).values():
                if isinstance( label_overrides, dict ):
                    label_overrides.pop( 'min_track_hits', None )
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.default_min_track_hits = self._pin_orig_default
        betaconfig.detector_backend = self._pin_orig_backend
        bu_config.invalidate_config_caches()


class TestScoreHysteresis( _ConfirmationPinnedMixin, unittest.TestCase ):
    """
    A provisional detection may continue a track but never start one.

    The thermostat rule: one threshold to start, a lower one to keep
    going, so a detection hovering near the line does not make the
    censor blink on and off.
    """

    def test_a_provisional_box_alone_starts_no_track( self ):
        boxes = [ censorable_box( 'exposed_breast', 1.0, provisional=True ) ]
        rendered, stats = bu_track.smooth_boxes( boxes )
        self.assertEqual( rendered, [] )
        self.assertEqual( stats['dropped_provisional'], 1 )

    def test_a_provisional_box_continues_an_established_track( self ):
        step = 1.0 / betaconfig.video_censor_fps
        boxes = [ censorable_box( 'exposed_breast', 1.0, x=100 ),
                  censorable_box( 'exposed_breast', 1.0 + step, x=102, provisional=True ) ]
        rendered, stats = bu_track.smooth_boxes( boxes )
        self.assertEqual( stats['dropped_provisional'], 0 )
        self.assertEqual( len( [ b for b in rendered if not b.get( 'interpolated' ) ] ), 2 )

    def test_a_provisional_box_after_the_track_died_is_dropped( self ):
        # Far enough apart that track_max_gap cannot bridge them.
        boxes = [ censorable_box( 'exposed_breast', 1.0, x=100 ),
                  censorable_box( 'exposed_breast', 90.0, x=100, provisional=True ) ]
        rendered, stats = bu_track.smooth_boxes( boxes )
        self.assertEqual( stats['dropped_provisional'], 1 )
        self.assertEqual( len( rendered ), 1 )

    def test_nothing_is_provisional_when_hysteresis_is_off( self ):
        # process_raw_box only ever marks a box provisional when the
        # label sets min_prob_continue, so a config that has not opted in
        # behaves exactly as it did before hysteresis existed.
        parts = { 'exposed_breast': {
            'min_prob': 0.5, 'min_prob_continue': None,
            'width_area_safety': 0, 'height_area_safety': 0, 'time_safety': 0.3,
            'censor_style': { 'type': 'blur' }, 'censor_shape': 'box' } }
        box = bu_censor.process_raw_box( raw_box( 'exposed_breast', 1.0, score=0.9 ),
                                         1920, 1080, parts )
        self.assertFalse( box['provisional'] )
        self.assertIsNone( bu_censor.process_raw_box(
            raw_box( 'exposed_breast', 1.0, score=0.4 ), 1920, 1080, parts ) )

    def test_a_score_between_the_two_gates_is_admitted_as_provisional( self ):
        parts = { 'exposed_breast': {
            'min_prob': 0.5, 'min_prob_continue': 0.3,
            'width_area_safety': 0, 'height_area_safety': 0, 'time_safety': 0.3,
            'censor_style': { 'type': 'blur' }, 'censor_shape': 'box' } }
        provisional = bu_censor.process_raw_box(
            raw_box( 'exposed_breast', 1.0, score=0.4 ), 1920, 1080, parts )
        self.assertIsNotNone( provisional )
        self.assertTrue( provisional['provisional'] )

        confirmed = bu_censor.process_raw_box(
            raw_box( 'exposed_breast', 1.0, score=0.9 ), 1920, 1080, parts )
        self.assertFalse( confirmed['provisional'] )

        self.assertIsNone( bu_censor.process_raw_box(
            raw_box( 'exposed_breast', 1.0, score=0.2 ), 1920, 1080, parts ) )


class TestTrackConfirmation( _ConfirmationPinnedMixin, unittest.TestCase ):
    """
    min_track_hits removes a track that never accumulated enough real
    detections - the single-frame false positive that would otherwise
    paint a censor blob for time_safety seconds.
    """

    def setUp( self ):
        # The mixin pins min_track_hits to 1 so the "default renders
        # every track" case is testing the default rather than whatever
        # the live config happens to be tuned to. Tests that care about
        # a specific value call _set_min_track_hits, which wins.
        super().setUp()
        self.original = copy.deepcopy( betaconfig.detector_backend )
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self.original
        super().tearDown()
        bu_config.invalidate_config_caches()

    def _set_min_track_hits( self, label, value ):
        backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.detector_backend[backend].setdefault(
            'item_overrides', {} ).setdefault( label, {} )['min_track_hits'] = value
        bu_config.invalidate_config_caches()

    def test_default_of_one_renders_every_track( self ):
        boxes = [ censorable_box( 'exposed_breast', 1.0 ) ]
        rendered, stats = bu_track.smooth_boxes( boxes )
        self.assertEqual( len( rendered ), 1 )
        self.assertEqual( stats['dropped_unconfirmed'], 0 )

    def test_a_one_frame_track_is_dropped_when_two_hits_are_required( self ):
        self._set_min_track_hits( 'exposed_breast', 2 )
        boxes = [ censorable_box( 'exposed_breast', 1.0 ) ]
        rendered, stats = bu_track.smooth_boxes( boxes )
        self.assertEqual( rendered, [] )
        self.assertEqual( stats['dropped_unconfirmed'], 1 )

    def test_a_continuing_track_survives_confirmation( self ):
        self._set_min_track_hits( 'exposed_breast', 2 )
        step = 1.0 / betaconfig.video_censor_fps
        boxes = [ censorable_box( 'exposed_breast', 1.0, x=100 ),
                  censorable_box( 'exposed_breast', 1.0 + step, x=101 ) ]
        rendered, stats = bu_track.smooth_boxes( boxes )
        self.assertEqual( stats['dropped_unconfirmed'], 0 )
        self.assertEqual( len( rendered ), 2 )

    def test_dropping_a_track_also_drops_its_interpolated_boxes( self ):
        self._set_min_track_hits( 'exposed_breast', 5 )
        step = 1.0 / betaconfig.video_censor_fps
        boxes = [ censorable_box( 'exposed_breast', 1.0, x=100 ),
                  censorable_box( 'exposed_breast', 1.0 + 3*step, x=110 ) ]
        rendered, _stats = bu_track.smooth_boxes( boxes )
        self.assertEqual( rendered, [],
            "an unconfirmed track must not leave synthetic boxes behind" )


class TestTrackPruningIsBehaviourPreserving( unittest.TestCase ):
    """
    Dead tracks are pruned each frame so the candidate scan stays
    proportional to LIVE tracks rather than every track ever created.
    That is purely a speed change, and the results must be identical.
    """

    def _long_sequence( self ):
        # Alternating clusters far apart in space, so tracks are
        # constantly created and abandoned.
        step = 1.0 / betaconfig.video_censor_fps
        boxes = []
        for index in range( 60 ):
            x = 100 if ( index // 3 ) % 2 == 0 else 1500
            boxes.append( censorable_box( 'exposed_breast', index*step, x=x, y=100 ) )
        return boxes

    def test_results_match_an_unpruned_reference( self ):
        # The reference: run with a track_max_gap so enormous that no
        # track is ever prunable, then with the normal one. Geometry of
        # the surviving boxes must be identical for the boxes that
        # matched, because pruning only ever removes candidates that
        # could not have matched anyway.
        rendered, stats = bu_track.smooth_boxes( self._long_sequence() )
        self.assertGreater( stats['tracks'], 1 )
        self.assertEqual( len( rendered ),
                          60 + stats['interpolated'] - stats['dropped_unconfirmed'] )

    def test_every_rendered_box_carries_a_track_id( self ):
        rendered, _stats = bu_track.smooth_boxes( self._long_sequence() )
        for box in rendered:
            self.assertIn( '_track_id', box )


class TestPrepareBoxesForRender( unittest.TestCase ):
    """The whole pipeline, end to end, on synthetic detections."""

    def test_runs_every_stage_and_reports_counts( self ):
        step = 1.0 / betaconfig.video_censor_fps
        raw_boxes = []
        for index in range( 6 ):
            raw_boxes.append( raw_box( 'exposed_breast', index*step,
                                       x=200, y=200, w=80, h=80, score=0.9 ) )
            raw_boxes.append( raw_box( 'face_femme', index*step,
                                       x=800, y=100, w=60, h=60, score=0.95 ) )
        boxes, stats = bu_track.prepare_boxes_for_render( raw_boxes, 1920, 1080 )

        self.assertEqual( stats['raw_detections'], 12 )
        self.assertTrue( boxes )
        # face_femme is not censored, but it IS a suppressor in the
        # shipped rules, so it must survive the relevance filter and then
        # be dropped by process_raw_box rather than earlier.
        self.assertTrue( all( box['label'] == 'exposed_breast' for box in boxes ) )
        self.assertEqual( boxes, sorted( boxes, key=lambda box: box['start'] ) )

    def test_output_is_sorted_by_start_time( self ):
        step = 1.0 / betaconfig.video_censor_fps
        raw_boxes = [ raw_box( 'exposed_breast', index*step, score=0.9 )
                      for index in reversed( range( 8 ) ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render( raw_boxes, 1920, 1080 )
        starts = [ box['start'] for box in boxes ]
        self.assertEqual( starts, sorted( starts ) )

    def test_an_empty_detection_set_produces_no_boxes( self ):
        boxes, stats = bu_track.prepare_boxes_for_render( [], 1920, 1080 )
        self.assertEqual( boxes, [] )
        self.assertEqual( stats['rendered_boxes'], 0 )


if __name__ == '__main__':
    unittest.main()
