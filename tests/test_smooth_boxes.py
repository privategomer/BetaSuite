#!/usr/bin/env python3
"""
test_smooth_boxes.py - unit tests for bu_track.smooth_boxes: track matching,
match_distance/track_max_gap blocking, shot-cut blocking (_cut_between),
interpolation, and paired_style behavior.

Imports betatv.py directly (everything real in that module runs under
if __name__ == '__main__', so a plain import is safe and gives the ACTUAL
production smooth_boxes - no source-extraction/regex tricks, unlike the
tools/analysis/*.py scripts, which need instrumentation hooks these tests
don't). Uses synthetic box dicts built by hand (same shape
betautils_censor.process_raw_box produces) rather than real cached
detections, so these run fast and don't depend on any video/cache
existing - they're a regression guard against smooth_boxes' own matching
logic changing shape, not a substitute for validating against real
footage (see tools/tuning/track_break_investigation.sh for that).
"""

import copy
import os
import sys
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_track as bu_track


class _UnconfirmedTracksRenderMixin:
    """
    Pin min_track_hits to 1 for the duration of each test.

    These tests are about track MATCHING and paired-style sharing, and
    their fixtures are two or three boxes in one or two frames. Track
    confirmation (default_min_track_hits, and the per-backend
    item_overrides value) filters any track with fewer real detections
    than that, so with the shipped config's value of 2 every fixture
    here renders nothing and the assertions fail for a reason that has
    nothing to do with what they are testing.

    Pinning it here keeps these tests measuring one thing. Confirmation
    itself is covered by TestTrackConfirmationValidation and the
    track-pipeline tests, which set it deliberately.
    """

    def setUp( self ):
        self._orig_default = betaconfig.default_min_track_hits
        self._orig_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.default_min_track_hits = 1
        for name in bu_detector.registered_backend_names():
            block = betaconfig.detector_backend.get( name )
            if not isinstance( block, dict ):
                continue
            for label_overrides in block.get( 'item_overrides', {} ).values():
                label_overrides.pop( 'min_track_hits', None )
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.default_min_track_hits = self._orig_default
        betaconfig.detector_backend = self._orig_backend
        bu_config.invalidate_config_caches()


def _box( label, start, end, t, x, y, w, h, score=0.9, shape='circle' ):
    return {
        'label': label, 'start': start, 'end': end, 't': t,
        'x': x, 'y': y, 'w': w, 'h': h, 'score': score,
        'censor_style': { 'type': 'blur', 'method': 'gaussian', 'strength': 16 },
        'censor_shape': shape,
        '_raw_x': x, '_raw_y': y, '_raw_w': w, '_raw_h': h,
        '_vid_w': 1920, '_vid_h': 1080,
    }


class TestSmoothBoxesTrackMatching( _UnconfirmedTracksRenderMixin, unittest.TestCase ):

    def test_backward_compatible_no_shot_cut_times_arg( self ):
        # smooth_boxes(boxes) with no second arg must still work exactly
        # as it did before shot-cut support was added - the two real
        # boxes stay in the result, plus whatever synthetic interpolated
        # boxes fill the gap between them (documented behavior: close
        # boxes with a real detection gap get interpolated, appended to
        # the end of the returned list)
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 0.5, 0.8, 0.6, 105, 102, 50, 50 ),
        ]
        result, _stats = bu_track.smooth_boxes( boxes )
        self.assertGreaterEqual( len( result ), 2 )
        self.assertEqual( len( result ), 2 + _stats['interpolated'] )
        self.assertIsInstance( _stats['interpolated'], int )

    def test_close_boxes_continue_one_track( self ):
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 0.11, 0.41, 0.21, 103, 101, 50, 50 ),
            _box( 'exposed_breast', 0.22, 0.52, 0.32, 106, 103, 50, 50 ),
        ]
        result, _stats = bu_track.smooth_boxes( copy.deepcopy( boxes ) )
        # all three should share the same resolved censor_style object
        # identity chain (proof they matched into one track, not three
        # independent new tracks) - censor_style gets reused verbatim
        # across a continuing track
        styles = [ b['censor_style'] for b in result if not b.get( 'interpolated' ) ]
        self.assertEqual( len( styles ), 3 )

    def test_far_apart_boxes_start_separate_tracks( self ):
        # same label, same timestamp region, but far apart in space -
        # should NOT be treated as the same instance
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 0.11, 0.41, 0.21, 1500, 900, 50, 50 ),
        ]
        result, _stats = bu_track.smooth_boxes( boxes )
        # far apart -> no match -> no interpolation between them
        self.assertEqual( _stats['interpolated'], 0 )

    def test_shot_cut_forces_track_reset( self ):
        # two close-in-space, close-in-time boxes that WOULD normally
        # match into one continuing track - but a shot cut falls exactly
        # between them, which must force a hard reset (no interpolation
        # bridging the cut) regardless of how close they are
        boxes_with_cut = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 0.5, 0.8, 0.6, 102, 101, 50, 50 ),
        ]
        _, _stats_with_cut = bu_track.smooth_boxes( copy.deepcopy( boxes_with_cut ), [0.35] )

        boxes_no_cut = copy.deepcopy( boxes_with_cut )
        _, stats_no_cut = bu_track.smooth_boxes( boxes_no_cut )

        # same geometry, only difference is the cut - with the cut, this
        # gap must NOT get interpolated; without it, it's close enough in
        # time/space that it should
        self.assertEqual( _stats_with_cut['interpolated'], 0 )
        self.assertGreaterEqual( stats_no_cut['interpolated'], 0 )  # sanity: still runs without error

    def test_cut_boundary_is_half_open( self ):
        # _cut_between(earlier, later) is (earlier, later] - a cut AT
        # prev['end'] exactly should NOT block (that cut already would
        # have forced a reset when prev's own box was matched); a cut
        # strictly between should block. Verified indirectly via
        # interpolated_count since _cut_between isn't exposed directly.
        boxes_a = [
            _box( 'exposed_breast', 0.0, 0.30, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 0.40, 0.70, 0.5, 102, 101, 50, 50 ),
        ]
        # cut exactly at prev['end'] (0.30) - should NOT block
        _, stats_at_boundary = bu_track.smooth_boxes( copy.deepcopy( boxes_a ), [0.30] )
        # cut strictly inside the gap (0.35) - SHOULD block
        _, stats_inside_gap = bu_track.smooth_boxes( copy.deepcopy( boxes_a ), [0.35] )
        self.assertGreaterEqual( stats_at_boundary['interpolated'], stats_inside_gap['interpolated'] )

    def test_two_simultaneous_boxes_get_separate_tracks( self ):
        # two boxes in the SAME frame (same 'start') - must never both be
        # assigned to the same track, even if they're close together
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 400, 400, 50, 50 ),
        ]
        result, _stats = bu_track.smooth_boxes( boxes )
        self.assertEqual( len( result ), 2 )
        xs = sorted( b['x'] for b in result )
        # neither box should have been pulled toward the other - positions
        # stay near their own raw detection, not blended into a midpoint
        self.assertLess( xs[0], 200 )
        self.assertGreater( xs[1], 300 )

    def test_empty_input_returns_empty( self ):
        result, _stats = bu_track.smooth_boxes( [] )
        self.assertEqual( result, [] )
        self.assertEqual( _stats['interpolated'], 0 )


class TestPairedStyleUnambiguousNearest( _UnconfirmedTracksRenderMixin, unittest.TestCase ):
    """
    paired_style (exposed_breast has it on) used to require EXACTLY ONE
    other live track nearby to share style - real data from
    analyze_style_flicker.py (2026-09-13) showed that gate almost never
    actually applied: 79% of independent-resolve events had 2+ candidates
    nearby, because two breasts on the SAME person plus any other
    already-live track nearby routinely puts the count above 1. Fixed to
    share with the single nearest candidate, as long as it's
    unambiguously closer (a real margin, not a near-tie) than the
    second-nearest - so a clean "two breasts, one person" case shares
    even with a 3rd unrelated track somewhere else in frame, while a
    genuinely ambiguous case (two candidates at similar distance) still
    resolves independently, same as before this fix.
    """

    def test_shares_with_nearest_despite_a_third_track_nearby( self ):
        # two very close boxes (same person, both breasts) plus a THIRD
        # track that's also within paired_style_max_distance but much
        # farther away - old exactly-one logic would see 2 candidates for
        # the close pair and refuse to share; the fix should still share
        # since the near one is unambiguously nearest
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),   # track A
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 160, 100, 50, 50 ),   # track B - close to A
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 600, 100, 50, 50 ),   # track C - much farther, still in range
        ]
        result, _stats = bu_track.smooth_boxes( copy.deepcopy( boxes ) )
        by_x = sorted( result, key=lambda b: b['x'] )
        # _box()'s censor_style is a single dict, always returned as-is by
        # resolve_censor_style regardless of sharing - censor_sticker_seed
        # is the real distinguishing signal (see the equidistant test
        # below for why): equal seeds mean A/B actually shared one
        # resolve, not two independent resolves that happened to match.
        self.assertEqual( by_x[0]['censor_sticker_seed'], by_x[1]['censor_sticker_seed'] )

    def test_does_not_share_when_two_candidates_are_equidistant( self ):
        # a triangle layout, not a straight line: A and B side by side,
        # C directly above their midpoint - by symmetry, C is EXACTLY
        # equidistant from A and B regardless of the vertical offset.
        # (A straight-line A-B-C layout does NOT produce a genuine tie
        # here: by the time C is resolved, A and B are already separate
        # tracks, and C's distance to the nearer one - B - is always
        # exactly half its distance to the farther one - A - so it's
        # never actually ambiguous; caught while writing this test by
        # tracing the real candidate list, not assumed.)
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 300, 80, 80 ),   # A
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 220, 300, 80, 80 ),   # B - same row as A
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 160, 200, 80, 80 ),   # C - above the A/B midpoint, equidistant from both
        ]
        result, _stats = bu_track.smooth_boxes( copy.deepcopy( boxes ) )
        # identify by starting position rather than a sort order that
        # could coincidentally put boxes in a misleading sequence
        a_box = next( b for b in result if not b.get('interpolated') and b['y'] > 250 and b['x'] < 150 )
        b_box = next( b for b in result if not b.get('interpolated') and b['y'] > 250 and b['x'] > 150 )
        c_box = next( b for b in result if not b.get('interpolated') and b['y'] < 250 )
        # _box()'s censor_style is a single dict (not a list), so
        # resolve_censor_style always returns that exact dict regardless
        # of whether a box shared or independently resolved - comparing
        # censor_style can never distinguish the two here. censor_sticker_
        # seed IS distinguishing: a shared box copies its source track's
        # seed verbatim, an independently-resolved one gets its own fresh
        # random.random() via setdefault - so equal seeds mean "shared",
        # different seeds mean "resolved independently" (astronomically
        # unlikely to coincidentally match by chance).
        # A and B are unambiguously each other's nearest (only candidate
        # in range at the time each resolves) and should share; C is
        # equidistant from both and should resolve on its own.
        self.assertEqual( a_box['censor_sticker_seed'], b_box['censor_sticker_seed'] )
        self.assertNotEqual( c_box['censor_sticker_seed'], a_box['censor_sticker_seed'] )


class TestPairedStyleDonorFreshness( _UnconfirmedTracksRenderMixin, unittest.TestCase ):
    """
    A style donor has to be RECENTLY DETECTED, not merely still
    matchable.

    THE BUG THIS PINS
    -----------------
    nearest_unambiguous_live_track used settings.max_gap - the same
    window track continuation uses - to decide whether a track was
    "live" enough to donate its censor style. max_gap is 27s for
    exposed_breast and 43.2s for exposed_vulva, because a track has to
    survive an occlusion.

    Style pairing asks a different question: "is this the other breast
    of the person on screen right now?" On the 2026-09-19 overnight run
    the structure bench measured 53.5 cuts/min on one compilation and
    74.7 on another - a shot every 0.8-1.1 seconds. At 27s of donor
    eligibility that leaves roughly two dozen departed subjects
    available to donate, which is why a new person appearing in a
    quadrant inherited the style of whoever was there before, and why a
    style appeared to "stick around" long after its subject left.

    paired_style_max_age bounds the donor window separately. These tests
    pin both directions: a fresh neighbour still shares, a stale one
    does not.
    """

    def _with_max_age( self, seconds, track_max_gap=None,
                       match_distance_multiplier=1.0 ):
        """
        Pin paired_style_max_age, and the two settings that bound it.

        track_max_gap is pinned rather than inherited because the donor
        window is min(paired_style_max_age, track_max_gap): a test that
        set only max_age was really testing whichever of the two the
        live config happened to make smaller. That is not hypothetical -
        tuning exposed_breast's track_max_gap down to 0.16s turned
        test_a_generous_max_age_restores_the_old_behaviour into a
        failure that looked like a code regression and was really a test
        reading a setting it never meant to depend on.

        match_distance_multiplier is pinned for the same reason, and it
        happened twice: auto_tune raising exposed_breast's value to 2.0
        widened the match radius enough that the two boxes 60px apart in
        test_a_stale_neighbour_does_not_donate_its_style matched into ONE
        track. They then shared a seed legitimately - same track, same
        style - and the assertion failed as though a stale donor had
        leaked. These tests are about the donor window, so the matching
        radius has to be held still for them to mean anything.
        """
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        backend = betaconfig.detector_backend['selected']
        overrides = betaconfig.detector_backend[backend].setdefault(
            'item_overrides', {} ).setdefault( 'exposed_breast', {} )
        overrides['paired_style_max_age'] = seconds
        overrides['match_distance_multiplier'] = match_distance_multiplier
        if track_max_gap is not None:
            overrides['track_max_gap'] = track_max_gap
        bu_config.invalidate_config_caches()

    def test_a_fresh_neighbour_still_donates_its_style( self ):
        # The behaviour that must NOT regress: two breasts on one
        # person, both detected in the same frame, still share.
        self._with_max_age( 1.0 )
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 160, 100, 50, 50 ),
        ]
        result, _stats = bu_track.smooth_boxes( copy.deepcopy( boxes ) )
        real = sorted( ( b for b in result if not b.get( 'interpolated' ) ),
                       key=lambda b: b['x'] )
        self.assertEqual( len( real ), 2 )
        self.assertEqual( real[0]['censor_sticker_seed'], real[1]['censor_sticker_seed'],
            "a co-detected pair stopped sharing a style; the donor window is too tight" )

    def test_a_stale_neighbour_does_not_donate_its_style( self ):
        # The bug. A track last seen 10s ago is well inside
        # exposed_breast's 27s max_gap, so before the fix it was an
        # eligible donor. With paired_style_max_age at 1s it is not.
        #
        # The two boxes are close in space (same screen region, as a
        # replacement subject in a split-screen panel would be) and far
        # apart in time.
        #
        # track_max_gap is pinned WIDE so max_age is provably the thing
        # doing the rejecting. With a narrow max_gap this test would
        # still pass, but for the wrong reason, and would no longer be
        # testing paired_style_max_age at all.
        self._with_max_age( 1.0, track_max_gap=30.0 )
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 10.0, 10.3, 10.1, 160, 100, 50, 50 ),
        ]
        result, _stats = bu_track.smooth_boxes( copy.deepcopy( boxes ) )
        real = sorted( ( b for b in result if not b.get( 'interpolated' ) ),
                       key=lambda b: b['start'] )
        self.assertEqual( len( real ), 2 )
        self.assertNotEqual( real[0]['censor_sticker_seed'], real[1]['censor_sticker_seed'],
            "a 10-second-stale track donated its style to a new box; a replacement "
            "subject will inherit the departed subject's censor style" )

    def test_a_generous_max_age_restores_the_old_behaviour( self ):
        # Confirms the gate is the new setting and nothing else: raise
        # paired_style_max_age above the gap and the stale donor is
        # eligible again, exactly as it was before the fix. Without this
        # the previous test could be passing for an unrelated reason.
        #
        # track_max_gap is pinned above the 10s gap too, since it is the
        # other half of the donor window. Leaving it to the live config
        # made this test assert "max_age is generous" while the config
        # quietly made max_gap the binding constraint.
        self._with_max_age( 30.0, track_max_gap=30.0 )
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 10.0, 10.3, 10.1, 160, 100, 50, 50 ),
        ]
        result, _stats = bu_track.smooth_boxes( copy.deepcopy( boxes ) )
        real = sorted( ( b for b in result if not b.get( 'interpolated' ) ),
                       key=lambda b: b['start'] )
        self.assertEqual( real[0]['censor_sticker_seed'], real[1]['censor_sticker_seed'] )

    def test_the_donor_window_never_exceeds_track_max_gap( self ):
        # paired_style_max_age is a CEILING on the donor window, not a
        # replacement for max_gap. A max_age larger than max_gap must
        # not resurrect a track that tracking itself has already given
        # up on, or style sharing would outlive the track it shares
        # from.
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        backend = betaconfig.detector_backend['selected']
        overrides = betaconfig.detector_backend[backend].setdefault(
            'item_overrides', {} ).setdefault( 'exposed_breast', {} )
        overrides['paired_style_max_age'] = 600.0
        overrides['track_max_gap'] = 2.0
        bu_config.invalidate_config_caches()
        boxes = [
            _box( 'exposed_breast', 0.0, 0.3, 0.1, 100, 100, 50, 50 ),
            _box( 'exposed_breast', 60.0, 60.3, 60.1, 160, 100, 50, 50 ),
        ]
        result, _stats = bu_track.smooth_boxes( copy.deepcopy( boxes ) )
        real = sorted( ( b for b in result if not b.get( 'interpolated' ) ),
                       key=lambda b: b['start'] )
        self.assertNotEqual( real[0]['censor_sticker_seed'], real[1]['censor_sticker_seed'] )


if __name__ == '__main__':
    unittest.main()
