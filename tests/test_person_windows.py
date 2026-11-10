"""
test_person_windows.py - person segmentation and reverse mode's whole-body windows.

What each group prevents:
  - The YOLO11-seg decoder: a box or mask landing off the person once the
    letterbox padding is undone, another class or a weak score let in,
    duplicates NMS should have merged, and a batch of mixed sizes
    misaligned. Checked against hand-built model outputs, so no model
    file is needed; one smoke test runs the real model when it is on disk
  - The cache: masks or boxes changed by a save/load round trip
  - Attribution: a woman windowed as a man, a man lost the moment his
    face turns away, evidence counted outside the mask, an overlap with
    a woman left windowed, a still windowed on male evidence alone when
    female evidence is inside too
  - The timeline: a moving body dragging its window over what it left
    (between samples it must be the intersection), windows across a gap
    or past the last sample, the trim not trimming, frame coordinates off
  - The compositor and the run: person windows not shown, shown over a
    censored label, or shown on an untriggered frame; the pass not run or
    not handed to the renderer
  - Config: an unknown backend, a missing model, bad person_windows
    settings, and the censor key ignoring the person model
  - betabench persons: a backend or size left out, a size one backend
    does not have crashing the sweep instead of being skipped
  (RF-DETR-Seg's own decoder and settings: tests/test_rf_detr_seg.py)
"""

import copy
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_censor as bu_censor
import betautils_config as bu_config
import betautils_person as bu_person
import betautils_reverse as bu_reverse
from segmenters import yolo11_seg

_WATCHED = ( 'reverse', 'person_backend', 'items_to_censor', 'item_overrides', 'detector_backend',
             'global_min_prob', 'enable_betasuite_watermark' )
_MISSING = object()


class Harness( unittest.TestCase ):

    def setUp( self ):
        self._saved = { name: copy.deepcopy( getattr( betaconfig, name, _MISSING ) ) for name in _WATCHED }
        betaconfig.enable_betasuite_watermark = False
        self._env = { key: os.environ.pop( key, None ) for key in (
            bu_person.BACKEND_OVERRIDE_ENV_VAR, bu_person.VARIANT_OVERRIDE_ENV_VAR ) }
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        for name, value in self._saved.items():
            if value is _MISSING:
                if hasattr( betaconfig, name ):
                    delattr( betaconfig, name )
            else:
                setattr( betaconfig, name, value )
        for key, value in self._env.items():
            if value is not None:
                os.environ[key] = value
        bu_config.invalidate_config_caches()

    def set_reverse( self, person_windows=None, **settings ):
        block = { 'enabled': True,
                  'frame_style': { 'type': 'bar', 'color': ( 0, 0, 0 ) },
                  'censor': { 'labels': [ 'exposed_breast' ] },
                  'windows': { 'face_femme': {} },
                  'person_windows': dict( { 'enabled': True, 'evidence_min_prob': 0.4, 'shrink': 0.0,
                                            'track_iou': 0.3, 'track_max_gap_seconds': 1.0 },
                                          **( person_windows or {} ) ) }
        block.update( settings )
        betaconfig.reverse = block
        bu_config.invalidate_config_caches()


def _person( x, y, w, h, t=0.0, score=0.9, mask=None ):
    """A person whose mask covers exactly its box; mask defaults to all inside."""
    mask = np.full( ( 8, 8 ), 255, dtype=np.uint8 ) if mask is None else mask
    return { 't': t, 'x': float( x ), 'y': float( y ), 'w': float( w ), 'h': float( h ),
             'score': score, 'mask': mask, 'mask_box': ( float( x ), float( y ), float( w ), float( h ) ) }


def _raw( label, x, y, w, h, t=0.0, score=0.9 ):
    return { 'class_id': label, 'score': score, 't': t, 'x': x, 'y': y, 'w': w, 'h': h, 'size': 320 }


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def _outputs( candidates, in_w, in_h ):
    """
    Hand-built (predictions, prototypes) for one image. Each candidate is
    (cx, cy, w, h, class, score, mask_on) in INPUT pixels; mask_on picks
    prototype 0 (everything inside) or prototype 1 (nothing).
    """
    predictions = np.zeros( ( 4 + 80 + 32, max( 1, len( candidates ) ) ), dtype=np.float32 )
    for index, ( cx, cy, w, h, cls, score, mask_on ) in enumerate( candidates ):
        predictions[ 0:4, index ] = ( cx, cy, w, h )
        predictions[ 4 + cls, index ] = score
        predictions[ 4 + 80 + ( 0 if mask_on else 1 ), index ] = 10.0
    prototypes = np.zeros( ( 32, in_h // 4, in_w // 4 ), dtype=np.float32 )
    prototypes[0] = 1.0
    prototypes[1] = -1.0
    return predictions, prototypes


class TestDecoder( unittest.TestCase ):

    def test_the_batch_is_letterboxed_to_the_next_multiple_of_32( self ):
        frame = np.zeros( ( 720, 1280, 3 ), dtype=np.uint8 )
        blob, placements = yolo11_seg.prep_batch( [ frame ], 640 )
        self.assertEqual( blob.shape, ( 1, 3, 384, 640 ) )
        self.assertEqual( placements, [ ( 0.5, 0, 12 ) ] )
        # The padding is the letterbox grey, the image is centred in it.
        self.assertAlmostEqual( float( blob[0, 0, 0, 0] ), 114 / 255.0, places=4 )
        self.assertAlmostEqual( float( blob[0, 0, 12, 0] ), 0.0, places=4 )

    def test_a_mixed_batch_shares_the_largest_shape_and_keeps_each_placement( self ):
        blob, placements = yolo11_seg.prep_batch(
            [ np.zeros( ( 720, 1280, 3 ), np.uint8 ), np.zeros( ( 1280, 720, 3 ), np.uint8 ) ], 640 )
        self.assertEqual( blob.shape, ( 2, 3, 640, 640 ) )
        self.assertEqual( placements[0], ( 0.5, 0, 140 ) )
        self.assertEqual( placements[1], ( 0.5, 140, 0 ) )

    def test_a_box_and_its_mask_map_back_through_the_letterbox( self ):
        # 1280x720 at 640: scale 0.5, 12 px of padding on top. A person at
        # original (200, 100)-(400, 500) is at input (100, 62)-(200, 262).
        predictions, prototypes = _outputs( [ ( 150, 162, 100, 200, 0, 0.9, True ) ], 640, 384 )
        people = yolo11_seg.decode_one( predictions, prototypes, ( 0.5, 0, 12 ), 1280, 720, 640,
                                        0.35, 0.5, 2.5 )
        self.assertEqual( len( people ), 1 )
        person = people[0]
        self.assertEqual( person['t'], 2.5 )
        for key, want in zip( ( 'x', 'y', 'w', 'h' ), ( 200, 100, 200, 400 ) ):
            self.assertAlmostEqual( person[key], want, places=3 )
        mx, my, mw, mh = person['mask_box']
        self.assertLessEqual( mx, 200 )
        self.assertLessEqual( my, 100 )
        self.assertGreaterEqual( mx + mw, 400 )
        self.assertGreaterEqual( my + mh, 500 )
        self.assertLess( mw - 200, 8.01, "rounded out by at most one prototype cell each side" )
        self.assertTrue( np.all( person['mask'] > 250 ) )

    def test_a_portrait_frame_maps_back_through_its_side_padding( self ):
        # 720x1280 at 640: scale 0.5, 140 px of padding on the left. A
        # person at original (100, 200)-(300, 600) is at input (190, 100)-(290, 300).
        predictions, prototypes = _outputs( [ ( 240, 200, 100, 200, 0, 0.9, True ) ], 640, 640 )
        person = yolo11_seg.decode_one( predictions, prototypes, ( 0.5, 140, 0 ), 720, 1280, 640,
                                        0.35, 0.5, 0.0 )[0]
        for key, want in zip( ( 'x', 'y', 'w', 'h' ), ( 100, 200, 200, 400 ) ):
            self.assertAlmostEqual( person[key], want, places=3 )
        mx, my, mw, mh = person['mask_box']
        self.assertTrue( 92 <= mx <= 100 and 300 <= mx + mw <= 308, person['mask_box'] )

    def test_only_candidates_above_the_floor_reach_nms( self ):
        # 8,400 candidates per 640x640 frame; NMS over all of them would
        # dominate the decode. The floor has to come first.
        candidates = [ ( 50 + 60 * k, 162, 40, 80, 0, 0.9 if k < 2 else 0.1, True ) for k in range( 8 ) ]
        predictions, prototypes = _outputs( candidates, 640, 384 )
        with mock.patch.object( yolo11_seg.cv2.dnn, 'NMSBoxes', wraps=yolo11_seg.cv2.dnn.NMSBoxes ) as nms:
            yolo11_seg.decode_one( predictions, prototypes, ( 0.5, 0, 12 ), 1280, 720, 640, 0.35, 0.5, 0.0 )
        self.assertEqual( len( nms.call_args[0][0] ), 2 )

    def test_another_class_and_a_weak_person_are_not_people( self ):
        predictions, prototypes = _outputs( [ ( 150, 162, 100, 200, 2, 0.95, True ),
                                              ( 400, 162, 100, 200, 0, 0.2, True ) ], 640, 384 )
        self.assertEqual( yolo11_seg.decode_one( predictions, prototypes, ( 0.5, 0, 12 ), 1280, 720,
                                                 640, 0.35, 0.5, 0.0 ), [] )

    def test_duplicates_merge_and_separate_people_stay_separate( self ):
        predictions, prototypes = _outputs( [ ( 150, 162, 100, 200, 0, 0.9, True ),
                                              ( 152, 164, 100, 200, 0, 0.8, True ),
                                              ( 450, 162, 100, 200, 0, 0.7, True ) ], 640, 384 )
        people = yolo11_seg.decode_one( predictions, prototypes, ( 0.5, 0, 12 ), 1280, 720, 640,
                                        0.35, 0.5, 0.0 )
        self.assertEqual( sorted( round( p['score'], 2 ) for p in people ), [ 0.7, 0.9 ] )

    def test_the_mask_is_the_candidates_own( self ):
        predictions, prototypes = _outputs( [ ( 150, 162, 100, 200, 0, 0.9, False ) ], 640, 384 )
        person = yolo11_seg.decode_one( predictions, prototypes, ( 0.5, 0, 12 ), 1280, 720, 640,
                                        0.35, 0.5, 0.0 )[0]
        self.assertTrue( np.all( person['mask'] < 5 ) )

    def test_mask_probability_is_the_sigmoid_of_the_coefficients( self ):
        predictions, prototypes = _outputs( [ ( 150, 162, 100, 200, 0, 0.9, True ) ], 640, 384 )
        predictions[ 4 + 80, 0 ] = 0.0          # logit 0 everywhere: probability one half
        person = yolo11_seg.decode_one( predictions, prototypes, ( 0.5, 0, 12 ), 1280, 720, 640,
                                        0.35, 0.5, 0.0 )[0]
        self.assertTrue( np.all( np.abs( person['mask'].astype( int ) - 128 ) <= 1 ), person['mask'] )

    def test_a_box_off_the_frame_is_clamped( self ):
        predictions, prototypes = _outputs( [ ( 20, 162, 100, 200, 0, 0.9, True ) ], 640, 384 )
        person = yolo11_seg.decode_one( predictions, prototypes, ( 0.5, 0, 12 ), 1280, 720, 640,
                                        0.35, 0.5, 0.0 )[0]
        self.assertEqual( person['x'], 0.0 )
        self.assertAlmostEqual( person['w'], 140.0, places=3 )

    def test_the_real_model_runs_when_it_is_on_disk( self ):
        path = yolo11_seg.VARIANTS['n']['model_path']
        if not os.path.exists( path ):
            self.skipTest( 'yolo11n-seg.onnx not exported here (setup does it)' )
        import onnxruntime
        session = onnxruntime.InferenceSession( path, providers=[ 'CPUExecutionProvider' ] )
        people = yolo11_seg.persons_for_imgs( [ np.zeros( ( 360, 640, 3 ), np.uint8 ) ] * 2, session,
                                              [ 0.0, 0.1 ], { 'model_variant': 'n' } )
        self.assertEqual( people, [], "a black frame has nobody in it" )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache( unittest.TestCase ):

    def test_a_round_trip_keeps_every_field( self ):
        tmp = tempfile.mkdtemp( prefix='betasuite-person-' )
        self.addCleanup( shutil.rmtree, tmp, True )
        path = os.path.join( tmp, 'p.npz' )
        mask = np.arange( 30, dtype=np.uint8 ).reshape( 5, 6 )
        people = [ dict( _person( 10, 20, 30, 40, t=1.5, score=0.7, mask=mask ), mask_box=( 9.0, 19.0, 32.0, 44.0 ) ),
                   _person( 50, 60, 10, 10, t=1.6 ) ]
        bu_person.save_cache( path, [ 1.5, 1.6, 1.7 ], people )
        times, loaded = bu_person.load_cache( path )
        self.assertEqual( times, [ 1.5, 1.6, 1.7 ] )
        self.assertEqual( len( loaded ), 2 )
        np.testing.assert_array_equal( bu_person.mask_of( loaded[0] ), np.where( mask >= 128, 255, 0 ) )
        self.assertEqual( loaded[0]['mask_box'], ( 9.0, 19.0, 32.0, 44.0 ) )
        self.assertAlmostEqual( loaded[0]['score'], 0.7, places=5 )
        self.assertEqual( ( loaded[1]['t'], loaded[1]['x'], loaded[1]['h'] ), ( 1.6, 50.0, 10.0 ) )

    def test_packing_keeps_inside_and_outside_and_shrinks_the_mask( self ):
        mask = np.random.RandomState( 1 ).randint( 0, 256, ( 37, 23 ), dtype=np.uint8 )
        packed = bu_person.compact( _person( 0, 0, 10, 10, mask=mask ) )
        self.assertNotIn( 'mask', packed )
        self.assertLessEqual( packed['mask_bits'].nbytes, mask.size // 8 + 1 )
        np.testing.assert_array_equal( bu_person.mask_of( packed ), np.where( mask >= 128, 255, 0 ) )
        self.assertIs( bu_person.compact( packed ), packed )

    def test_attribution_and_windows_read_packed_masks( self ):
        self.assertTrue( bu_person._inside_mask( bu_person.compact( _person( 0, 0, 100, 200 ) ), 50, 50 ) )

    def test_an_empty_pass_round_trips( self ):
        tmp = tempfile.mkdtemp( prefix='betasuite-person-' )
        self.addCleanup( shutil.rmtree, tmp, True )
        path = os.path.join( tmp, 'p.npz' )
        bu_person.save_cache( path, [ 0.0, 0.1 ], [] )
        self.assertEqual( bu_person.load_cache( path ), ( [ 0.0, 0.1 ], [] ) )


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

class TestAttribution( Harness ):

    def setUp( self ):
        super().setUp()
        self.set_reverse()

    def windows( self, sample_times, people, raw ):
        evidence = {}
        for box in raw:
            evidence.setdefault( round( box['t'], 3 ), [] ).append( box )
        return bu_person.male_people_by_time( sample_times, people, evidence )

    def shown( self, sample_times, people, raw ):
        return { t: entry['show'] for t, entry in self.windows( sample_times, people, raw ).items() }

    def test_male_evidence_inside_the_mask_windows_the_person( self ):
        man, woman = _person( 0, 0, 100, 200 ), _person( 200, 0, 100, 200 )
        windows = self.windows( [ 0.0 ], [ man, woman ],
                                [ _raw( 'face_masc', 30, 10, 40, 40 ), _raw( 'face_femme', 230, 10, 40, 40 ) ] )
        self.assertEqual( windows[0.0], { 'show': [ man ], 'hide': [ woman ] } )

    def test_evidence_inside_the_box_but_outside_the_mask_does_not_count( self ):
        mask = np.zeros( ( 8, 8 ), np.uint8 )
        mask[ 4:, : ] = 255                      # only the bottom half of the box is the person
        person = _person( 0, 0, 100, 200, mask=mask )
        self.assertEqual( self.shown( [ 0.0 ], [ person ], [ _raw( 'face_masc', 30, 10, 40, 40 ) ] )[0.0], [] )
        self.assertEqual( self.shown( [ 0.0 ], [ person ], [ _raw( 'face_masc', 30, 140, 40, 40 ) ] )[0.0],
                          [ person ] )

    def test_weak_evidence_and_other_labels_are_not_evidence( self ):
        raw = [ _raw( 'face_masc', 30, 10, 40, 40, score=0.39 ), _raw( 'face_masc', 30, 10, 40, 40, score=0.4 ),
                _raw( 'exposed_belly', 30, 10, 40, 40, score=0.9 ) ]
        with mock.patch( 'betautils_track.filter_raw_detections', side_effect=lambda raw_boxes, *a, **k: ( raw_boxes, {} ) ):
            evidence = bu_person.evidence_detections( raw, 640, 360 )
        self.assertEqual( [ box['score'] for box in evidence[0.0] ], [ 0.4 ] )

    def test_a_man_stays_a_man_when_his_face_turns_away( self ):
        samples = [ _person( 10 * k, 0, 100, 200, t=round( 0.1 * k, 3 ) ) for k in range( 5 ) ]
        shown = self.shown( [ p['t'] for p in samples ], samples, [ _raw( 'face_masc', 30, 10, 40, 40, t=0.0 ) ] )
        self.assertTrue( all( shown[ p['t'] ] == [ p ] for p in samples ), shown )

    def test_a_different_person_in_the_next_sample_does_not_inherit_his_track( self ):
        man, stranger = _person( 0, 0, 100, 200, t=0.0 ), _person( 400, 0, 100, 200, t=0.1 )
        windows = self.windows( [ 0.0, 0.1 ], [ man, stranger ], [ _raw( 'face_masc', 30, 10, 40, 40, t=0.0 ) ] )
        self.assertEqual( windows[0.1], { 'show': [], 'hide': [ stranger ] } )

    def test_a_track_broken_by_a_long_gap_starts_over( self ):
        early, late = _person( 0, 0, 100, 200, t=0.0 ), _person( 0, 0, 100, 200, t=3.0 )
        shown = self.shown( [ 0.0, 3.0 ], [ early, late ], [ _raw( 'face_masc', 30, 10, 40, 40, t=0.0 ) ] )
        self.assertEqual( shown[3.0], [] )

    def test_more_female_than_male_evidence_is_not_a_man( self ):
        # A woman a man's face overlaps once: male evidence once, female twice.
        samples = [ _person( 0, 0, 100, 200, t=round( 0.1 * k, 3 ) ) for k in range( 3 ) ]
        raw = [ _raw( 'face_masc', 30, 10, 40, 40, t=0.0 ),
                _raw( 'exposed_breast', 30, 80, 30, 30, t=0.1 ), _raw( 'face_femme', 30, 10, 40, 40, t=0.2 ) ]
        shown = self.shown( [ p['t'] for p in samples ], samples, raw )
        self.assertTrue( all( not w for w in shown.values() ), shown )

    def test_a_woman_overlapping_a_man_is_cut_out_and_he_stays_shown( self ):
        man, woman = _person( 0, 0, 200, 300 ), _person( 120, 100, 200, 200 )
        raw = [ _raw( 'face_masc', 30, 10, 40, 40 ), _raw( 'face_femme', 250, 110, 40, 40 ),
                _raw( 'exposed_breast', 140, 150, 30, 30 ) ]          # inside both outlines
        self.assertEqual( self.windows( [ 0.0 ], [ man, woman ], raw )[0.0],
                          { 'show': [ man ], 'hide': [ woman ] } )

    def test_anyone_not_attributed_is_cut_out_too( self ):
        man, stranger = _person( 0, 0, 200, 300 ), _person( 120, 100, 200, 200 )
        self.assertEqual( self.windows( [ 0.0 ], [ man, stranger ], [ _raw( 'face_masc', 30, 10, 40, 40 ) ] )[0.0],
                          { 'show': [ man ], 'hide': [ stranger ] } )

    def test_evidence_inside_two_outlines_votes_for_neither( self ):
        # The chest lands where the two overlap: it cannot say which of
        # them it belongs to, so neither becomes a man from it.
        left, right = _person( 0, 0, 200, 300 ), _person( 120, 0, 200, 300 )
        self.assertEqual( self.windows( [ 0.0 ], [ left, right ], [ _raw( 'exposed_chest', 140, 150, 20, 20 ) ] )[0.0],
                          { 'show': [], 'hide': [ left, right ] } )

    def test_a_penis_inside_a_womans_outline_does_not_make_her_a_man( self ):
        # In penetration the man's penis sits inside his partner's
        # outline. Counted as male evidence it outvoted her own face, so
        # her whole body was shown wherever her face went undetected.
        samples = [ _person( 0, 0, 100, 200, t=round( 0.1 * k, 3 ) ) for k in range( 3 ) ]
        raw = [ _raw( 'face_femme', 30, 10, 40, 40, t=0.0 ), _raw( 'face_femme', 30, 10, 40, 40, t=0.1 ) ]
        raw += [ _raw( 'exposed_penis', 40, 120, 20, 20, t=round( 0.1 * k, 3 ) ) for k in range( 3 ) ]
        windows = self.windows( [ p['t'] for p in samples ], samples, raw )
        self.assertEqual( [ windows[t]['show'] for t in ( 0.0, 0.1, 0.2 ) ], [ [], [], [] ] )

    def test_one_outline_holding_a_woman_is_not_shown_at_that_sample( self ):
        # Two bodies returned as ONE outline: female evidence inside his
        # outline that no separate outline accounts for.
        samples = [ _person( 0, 0, 100, 200, t=round( 0.1 * k, 3 ) ) for k in range( 3 ) ]
        raw = [ _raw( 'face_masc', 30, 10, 40, 40, t=0.0 ), _raw( 'face_masc', 30, 10, 40, 40, t=0.1 ),
                _raw( 'exposed_vulva', 40, 120, 20, 20, t=0.2 ) ]
        windows = self.windows( [ p['t'] for p in samples ], samples, raw )
        self.assertEqual( [ bool( windows[t]['show'] ) for t in ( 0.0, 0.1, 0.2 ) ], [ True, True, False ] )
        self.assertEqual( windows[0.2]['hide'], [ samples[2] ], "and he is cut out of anyone else's" )

    def test_female_evidence_a_separate_man_accounts_for_still_hides_him( self ):
        # Only a NON-male outline explains female evidence; another man's does not.
        first, second = _person( 0, 0, 200, 300 ), _person( 120, 0, 200, 300 )
        raw = [ _raw( 'face_masc', 20, 10, 40, 40 ), _raw( 'face_masc', 260, 10, 40, 40 ),
                _raw( 'exposed_vulva', 150, 150, 20, 20 ) ]
        self.assertEqual( self.windows( [ 0.0 ], [ first, second ], raw )[0.0],
                          { 'show': [], 'hide': [ first, second ] } )

    def test_a_still_is_judged_on_its_own_votes( self ):
        person = _person( 0, 0, 100, 200 )
        self.assertEqual( bu_person.male_people_in_still( [ person ], [ _raw( 'face_masc', 30, 10, 40, 40 ) ] ),
                          { 'show': [ person ], 'hide': [] } )
        self.assertEqual( bu_person.male_people_in_still(
            [ person ], [ _raw( 'face_masc', 30, 10, 40, 40 ), _raw( 'exposed_breast', 30, 80, 30, 30 ) ] ),
            { 'show': [], 'hide': [ person ] } )
        # B has a man's face AND a woman's of his own: not a man, so the
        # vulva shared with A is accounted for by B and A stays shown.
        a, b = _person( 0, 0, 200, 300 ), _person( 120, 0, 200, 300 )
        self.assertEqual( bu_person.male_people_in_still(
            [ a, b ], [ _raw( 'face_masc', 20, 10, 40, 40 ), _raw( 'face_masc', 260, 10, 40, 40 ),
                        _raw( 'face_femme', 260, 100, 40, 40 ), _raw( 'exposed_vulva', 150, 150, 20, 20 ) ] ),
            { 'show': [ a ], 'hide': [ b ] } )
        woman = _person( 50, 100, 100, 100 )
        self.assertEqual( bu_person.male_people_in_still(
            [ person, woman ], [ _raw( 'face_masc', 30, 10, 40, 40 ), _raw( 'face_femme', 110, 160, 30, 30 ),
                                 _raw( 'exposed_breast', 60, 120, 20, 20 ) ] ),
            { 'show': [ person ], 'hide': [ woman ] } )

    def test_evidence_goes_through_suppression_like_everything_else( self ):
        with mock.patch( 'betautils_track.filter_raw_detections',
                         return_value=( [ _raw( 'face_masc', 0, 0, 9, 9, t=0.5 ) ], {} ) ) as stage:
            evidence = bu_person.evidence_detections( [ _raw( 'face_masc', 0, 0, 9, 9, t=0.5 ) ], 640, 360 )
        self.assertEqual( list( evidence ), [ 0.5 ] )
        self.assertEqual( set( stage.call_args[0][3] ),
                          set( bu_reverse.settings()['person_windows']['male_labels'] )
                          | set( bu_reverse.settings()['person_windows']['female_labels'] ) )


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def _shown( *people ):
    return { 'show': list( people ), 'hide': [] }


def _inside( mask, width=640, height=360 ):
    inside = np.zeros( ( height, width ), bool )
    if mask is not None:
        x, y, w, h, alpha = mask
        inside[ y:y+h, x:x+w ] = alpha > 0
    return inside


class TestTimeline( Harness ):

    def timeline( self, windows, fps=10, shrink=0.0, size=( 640, 360 ) ):
        return bu_person.PersonMaskTimeline( windows, size[0], size[1], fps, shrink )

    def test_at_a_sample_the_window_is_that_samples_mask( self ):
        timeline = self.timeline( { 0.0: _shown( _person( 100, 50, 200, 200 ) ), 0.1: _shown() } )
        inside = _inside( timeline.mask_at( 0.0, 640, 360 ) )
        self.assertTrue( inside[ 60:240, 110:290 ].all() )
        self.assertFalse( inside[ :, :90 ].any() or inside[ :, 310: ].any() )

    def test_between_samples_only_what_both_cover_is_shown( self ):
        timeline = self.timeline( { 0.0: _shown( _person( 100, 50, 200, 200 ) ),
                                    0.1: _shown( _person( 200, 50, 200, 200 ) ) } )
        inside = _inside( timeline.mask_at( 0.05, 640, 360 ) )
        self.assertTrue( inside[ 100:200, 210:290 ].all(), "the overlap is shown" )
        self.assertFalse( inside[ :, :190 ].any(), "where he was is not" )
        self.assertFalse( inside[ :, 310: ].any(), "where he is going is not" )

    def test_a_sample_with_nobody_closes_the_window_next_to_it( self ):
        timeline = self.timeline( { 0.0: _shown( _person( 100, 50, 200, 200 ) ), 0.1: _shown() } )
        self.assertIsNone( timeline.mask_at( 0.05, 640, 360 ) )

    def test_nothing_past_the_last_sample_or_across_a_wide_gap( self ):
        timeline = self.timeline( { 0.0: _shown( _person( 100, 50, 200, 200 ) ),
                                    1.0: _shown( _person( 100, 50, 200, 200 ) ) } )
        self.assertIsNone( timeline.mask_at( 0.5, 640, 360 ) )
        self.assertIsNone( timeline.mask_at( 1.5, 640, 360 ) )
        self.assertIsNone( timeline.mask_at( -0.5, 640, 360 ) )

    def test_shrink_trims_the_edge( self ):
        person = _person( 100, 50, 200, 200 )
        plain = self.timeline( { 0.0: _shown( person ) } ).mask_at( 0.0, 640, 360 )
        trimmed = self.timeline( { 0.0: _shown( person ) }, shrink=0.1 ).mask_at( 0.0, 640, 360 )
        self.assertLess( ( trimmed[4] > 0 ).sum(), ( plain[4] > 0 ).sum() * 0.85 )

    def test_frame_pixels_follow_the_frame_size_asked_for( self ):
        timeline = self.timeline( { 0.0: _shown( _person( 320, 0, 320, 360 ) ) } )
        x, y, w, h, alpha = timeline.mask_at( 0.0, 1280, 720 )
        self.assertGreaterEqual( x, 600 )
        self.assertEqual( x + w, 1280 )

    def test_a_hidden_outline_is_cut_out_of_a_shown_one( self ):
        man, woman = _person( 100, 50, 300, 300 ), _person( 250, 100, 200, 200 )
        inside = _inside( self.timeline( { 0.0: { 'show': [ man ], 'hide': [ woman ] } } ).mask_at( 0.0, 640, 360 ) )
        self.assertTrue( inside[ 60:340, 110:240 ].all(), "his body clear of her" )
        self.assertFalse( inside[ 100:300, 250:400 ].any(), "her overlap with him obscured" )

    def test_shrink_grows_what_is_cut_out( self ):
        man, woman = _person( 100, 50, 300, 300 ), _person( 250, 100, 200, 200 )
        entry = { 0.0: { 'show': [ man ], 'hide': [ woman ] } }
        plain = _inside( self.timeline( entry ).mask_at( 0.0, 640, 360 ) )
        grown = _inside( self.timeline( entry, shrink=0.1 ).mask_at( 0.0, 640, 360 ) )
        self.assertTrue( plain[ 150:250, 235:245 ].all() )
        self.assertFalse( grown[ 150:250, 235:245 ].any(), "the cut reaches past her outline" )


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

class TestCompositor( Harness ):

    def test_a_person_window_shows_the_source_and_a_censored_label_still_wins( self ):
        source = np.random.RandomState( 3 ).randint( 0, 256, ( 120, 160, 3 ), dtype=np.uint8 )
        alpha = np.zeros( ( 60, 80 ), np.uint8 )
        alpha[ 10:50, 10:70 ] = 255
        white = { 'x': 40, 'y': 30, 'w': 20, 'h': 20, 'censor_style': { 'type': 'bar', 'color': ( 255, 255, 255 ) },
                  'censor_shape': 'box', 'label': 'exposed_breast', 'censor_sticker_seed': 0.0,
                  'start': 0.0, 'end': 1.0 }
        out = bu_censor.censor_img_reverse( source.copy(), [ white ], 1.0, { 'type': 'bar', 'color': ( 0, 0, 0 ) },
                                            ( 20, 20, 80, 60, alpha ) )
        np.testing.assert_array_equal( out[ 30:40, 30:40 ], source[ 30:40, 30:40 ] )
        self.assertTrue( np.all( out[ 30:50, 40:60 ] == 255 ), "the censored label wins" )
        self.assertTrue( np.all( out[ 20:30, 20:30 ] == 0 ), "outside the mask, inside its rectangle" )
        self.assertTrue( np.all( out[ 100:, : ] == 0 ) )

    def test_render_plan_asks_for_masks_only_with_a_timeline( self ):
        plan = bu_reverse.ReverseRender( bu_reverse.TriggerTimeline( [ ( 0, 1 ) ], 0 ), { 'type': 'bar' } )
        self.assertIsNone( plan.window_mask( 0.5, 160, 120 ) )
        plan.window_masks = mock.Mock( **{ 'mask_at.return_value': 'mask' } )
        self.assertEqual( plan.window_mask( 0.5, 160, 120 ), 'mask' )
        plan.window_masks.mask_at.assert_called_once_with( 0.5, 160, 120 )


# ---------------------------------------------------------------------------
# Config, key
# ---------------------------------------------------------------------------

class TestConfig( Harness ):

    def errors( self ):
        bu_config.invalidate_config_caches()
        errors = []
        bu_config._validate_reverse( errors, bu_config._valid_class_names() )
        bu_config._validate_person_backend( errors )
        return errors

    def test_person_windows_mistakes_are_rejected( self ):
        for person_windows in ( { 'male_labels': [ 'not_a_class' ] }, { 'male_labels': [] },
                                { 'female_labels': [ 'face_masc' ] }, { 'evidence_min_prob': 1.0 },
                                { 'evidence_min_prob': 0.05 }, { 'track_iou': 0 }, { 'shrink': 0.5 },
                                { 'track_max_gap_seconds': -1 }, { 'enabled': 'yes' }, { 'shrnk': 0.1 } ):
            with mock.patch( 'os.path.exists', return_value=True ):
                self.set_reverse( person_windows=person_windows )
                self.assertTrue( self.errors(), person_windows )

    def test_the_shipped_person_settings_validate_when_the_model_is_there( self ):
        if self._saved['reverse'] is _MISSING or self._saved['person_backend'] is _MISSING:
            self.skipTest( 'no reverse or person_backend block in betaconfig.py' )
        betaconfig.reverse = dict( copy.deepcopy( self._saved['reverse'] ), enabled=True )
        betaconfig.person_backend = copy.deepcopy( self._saved['person_backend'] )
        with mock.patch( 'os.path.exists', return_value=True ):
            self.assertEqual( self.errors(), [] )

    def test_a_missing_model_stops_the_run_only_while_person_windows_are_on( self ):
        betaconfig.person_backend = { 'selected': 'yolo11_seg',
                                      'yolo11_seg': { 'model_path': '/nowhere/yolo.onnx' } }
        self.set_reverse()
        self.assertTrue( any( '/nowhere/yolo.onnx' in e and 'setup' in e for e in self.errors() ) )
        self.set_reverse( person_windows={ 'enabled': False } )
        self.assertEqual( self.errors(), [] )

    def test_an_unknown_backend_is_named( self ):
        betaconfig.person_backend = { 'selected': 'sam9' }
        self.set_reverse()
        self.assertTrue( any( 'sam9' in e for e in self.errors() ) )
        betaconfig.person_backend = { 'selected': 'yolo11_seg', 'yolo11_sg': {} }
        self.assertTrue( any( 'yolo11_sg' in e for e in self.errors() ) )

    def test_rf_detr_seg_is_a_finished_backend_that_needs_its_model( self ):
        betaconfig.person_backend = { 'selected': 'rf_detr_seg',
                                      'rf_detr_seg': { 'model_path': '/nowhere/rf.onnx' } }
        self.set_reverse()
        errors = self.errors()
        self.assertTrue( any( '/nowhere/rf.onnx' in e and 'setup' in e for e in errors ), errors )
        betaconfig.person_backend = { 'selected': 'rf_detr_seg', 'rf_detr_seg': {} }
        with mock.patch( 'os.path.exists', return_value=True ):
            self.assertEqual( self.errors(), [] )

    def test_the_shipped_rf_detr_settings_validate_when_selected( self ):
        if self._saved['person_backend'] is _MISSING:
            self.skipTest( 'no person_backend block in betaconfig.py' )
        betaconfig.person_backend = dict( copy.deepcopy( self._saved['person_backend'] ), selected='rf_detr_seg' )
        self.set_reverse()
        with mock.patch( 'os.path.exists', return_value=True ):
            self.assertEqual( self.errors(), [] )

    def test_bad_yolo_settings_are_rejected( self ):
        for settings in ( { 'model_variant': 'xl' }, { 'picture_size': 650 }, { 'min_prob': 0 },
                          { 'nms_iou': 1.5 }, { 'batch_size': 0 } ):
            betaconfig.person_backend = { 'selected': 'yolo11_seg', 'yolo11_seg': settings }
            self.set_reverse()
            with mock.patch( 'os.path.exists', return_value=True ):
                self.assertTrue( self.errors(), settings )

    def test_the_variant_override_picks_the_model( self ):
        betaconfig.person_backend = { 'selected': 'yolo11_seg', 'yolo11_seg': { 'model_variant': 's' } }
        self.assertTrue( bu_person.resolve_config()['model_path'].endswith( 'yolo11s-seg.onnx' ) )
        os.environ[ bu_person.VARIANT_OVERRIDE_ENV_VAR ] = 'm'
        try:
            self.assertTrue( bu_person.resolve_config()['model_path'].endswith( 'yolo11m-seg.onnx' ) )
        finally:
            del os.environ[ bu_person.VARIANT_OVERRIDE_ENV_VAR ]

    def key( self ):
        bu_config.invalidate_config_caches()
        return bu_cache.censor_key()

    def test_the_censor_key_covers_the_person_model_only_while_windows_are_on( self ):
        betaconfig.person_backend = { 'selected': 'yolo11_seg', 'yolo11_seg': { 'model_variant': 's' } }
        self.set_reverse()
        before = self.key()
        betaconfig.person_backend['yolo11_seg']['model_variant'] = 'm'
        self.assertNotEqual( self.key(), before )
        self.set_reverse( person_windows={ 'shrink': 0.1 } )
        self.assertNotEqual( self.key(), before )
        self.set_reverse( person_windows={ 'enabled': False } )
        off = self.key()
        betaconfig.person_backend['yolo11_seg']['model_variant'] = 's'
        self.assertEqual( self.key(), off )

    def test_the_person_cache_is_keyed_by_the_model( self ):
        betaconfig.person_backend = { 'selected': 'yolo11_seg', 'yolo11_seg': { 'model_variant': 's' } }
        first = bu_cache.person_cache_path_for( 'abc', 9, 'yolo11_seg', bu_person.identity() )
        betaconfig.person_backend['yolo11_seg']['min_prob'] = 0.5
        self.assertNotEqual( first, bu_cache.person_cache_path_for( 'abc', 9, 'yolo11_seg', bu_person.identity() ) )
        self.assertIn( 'abc-person-yolo11_seg-9-p', first )


# ---------------------------------------------------------------------------
# One real run, with the person model stubbed
# ---------------------------------------------------------------------------

from tests.test_betatv_end_to_end import BetaTvHarness, _StubDetectorModule  # noqa: E402


class _ManAndBreastStub( _StubDetectorModule ):
    """Between 1.0 s and 2.0 s: a breast box in the middle and a man's face top-left."""

    def raw_boxes_for_imgs( self, imgs, size, session, ts ):
        boxes = [ box for box in super().raw_boxes_for_imgs( imgs, size, session, ts )
                  if 1.0 <= box['t'] <= 2.0 ]
        faces = [ dict( box, x=10.0, y=10.0, w=20.0, h=20.0, class_id='face_masc', score=0.95 )
                  for box in boxes ]
        return boxes + faces


def _fake_people( imgs, ts, session=None, backend_name=None ):
    """A man filling x 0..38, y 0..80 in every sampled frame."""
    return [ _person( 0, 0, 38, 80, t=t ) for t in ts ]


@unittest.skipUnless( shutil.which( 'ffmpeg' ), "ffmpeg is needed to build the test clip" )
class TestPersonWindowRun( BetaTvHarness ):

    def setUp( self ):
        super().setUp()
        self._saved_reverse = copy.deepcopy( getattr( betaconfig, 'reverse', _MISSING ) )
        betaconfig.reverse = { 'enabled': True,
                               'frame_style': { 'type': 'blur', 'method': 'box', 'strength': 30 },
                               'fade_seconds': 0.0,
                               'trigger': { 'min_prob': 0.3, 'hold_seconds': 0.0 },
                               'censor': { 'labels': [ 'exposed_breast' ] },
                               'windows': {},
                               'person_windows': { 'enabled': True, 'shrink': 0.0 } }
        betaconfig.video_censor_fps = 10
        bu_config.invalidate_config_caches()
        self.detector = _ManAndBreastStub()
        self._detector_patch.stop()
        import betatv
        self._detector_patch = mock.patch.object( betatv.bu_detector, 'get_detector', return_value=self.detector )
        self._detector_patch.start()
        self._people_patch = mock.patch.object( bu_person, 'persons_for_imgs', side_effect=_fake_people )
        self._session_patch = mock.patch.object( bu_person, 'get_session', return_value=object() )
        self._people_patch.start()
        self._session_patch.start()

    def tearDown( self ):
        self._people_patch.stop()
        self._session_patch.stop()
        if self._saved_reverse is _MISSING:
            if hasattr( betaconfig, 'reverse' ):
                delattr( betaconfig, 'reverse' )
        else:
            betaconfig.reverse = self._saved_reverse
        super().tearDown()

    def test_a_mans_body_stays_clear_inside_a_triggered_frame( self ):
        source = os.path.join( self.uncensored, 'clip.mp4' )
        subprocess.run( [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                          'testsrc2=size=160x120:rate=10:duration=3', '-c:v', 'libx264',
                          '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', source ], check=True )
        self.assertEqual( self.run_one(), 'processed' )
        outputs = self.outputs()
        self.assertEqual( len( outputs ), 1 )
        from tests.test_reverse_mode import _decode
        original = _decode( source )
        rendered = _decode( os.path.join( self.censored, outputs[0] ) )

        def mean_diff( index, rows, cols ):
            return float( np.abs( rendered[index][rows, cols] - original[index][rows, cols] ).mean() )

        body = ( slice( 6, 74 ), slice( 4, 34 ) )
        elsewhere = ( slice( 90, 120 ), slice( 0, 160 ) )
        self.assertGreater( mean_diff( 15, *elsewhere ), 15.0, "the frame is triggered" )
        self.assertLess( mean_diff( 15, *body ), 6.0, "the man's body is shown" )
        self.assertLess( mean_diff( 3, *body ), 4.0, "untriggered frames pass through" )
        cache = [ name for name in os.listdir( bu_cache.VID_HASH_DIR ) if '-person-' in name ]
        self.assertEqual( len( cache ), 1, "the pass is cached" )

    def test_a_failed_file_stops_the_person_pass_so_it_can_checkpoint( self ):
        import betatv
        source = os.path.join( self.uncensored, 'clip.mp4' )
        subprocess.run( [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                          'testsrc2=size=160x120:rate=10:duration=1', '-c:v', 'libx264',
                          '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', source ], check=True )
        background = mock.Mock()
        with mock.patch.object( bu_person, 'BackgroundPersonPass', return_value=background ), \
             mock.patch.object( betatv, 'detect_boxes_for_video', side_effect=RuntimeError( 'boom' ) ):
            self.assertEqual( self.run_one(), 'failed' )
        background.stop.assert_called_once_with()

    def test_an_interrupted_reverse_render_resumes_to_the_same_frames( self ):
        # Sixteen looks for the censored label, so a style re-rolled on
        # the resumed run would almost surely differ from the clean one.
        import betautils_render as bu_render
        import betautils_signals as bu_signals
        from tests.test_reverse_mode import _decode
        betaconfig.reverse['censor'] = { 'labels': [ 'exposed_breast' ], 'overrides': { 'exposed_breast': {
            'censor_shape': 'box',
            'censor_style': [ { 'type': 'bar', 'color': ( 16 * k, 255 - 16 * k, ( 97 * k ) % 256 ), 'weight': 1 }
                              for k in range( 16 ) ] } } }
        betaconfig.render_chunk_seconds = 1
        bu_config.invalidate_config_caches()
        source = os.path.join( self.uncensored, 'clip.mp4' )
        subprocess.run( [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                          'testsrc2=size=160x120:rate=10:duration=3', '-c:v', 'libx264',
                          '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', source ], check=True )

        self.assertEqual( self.run_one(), 'processed' )
        output = os.path.join( self.censored, self.outputs()[0] )
        clean = _decode( output )
        os.remove( output )

        real_chunk = bu_render.render_one_chunk
        rendered = []

        def stop_after_the_first_chunk( plan, *args, **kwargs ):
            rendered.append( plan.index )
            result = real_chunk( plan, *args, **kwargs )
            bu_signals.request_interrupt()
            return result

        with mock.patch.object( bu_render, 'render_one_chunk', side_effect=stop_after_the_first_chunk ):
            with self.assertRaises( bu_signals.Interrupted ):
                self.run_one()
        bu_signals.clear()
        self.assertEqual( rendered[0], 0, "chunk 0 finished before the stop" )

        rendered.clear()
        with mock.patch.object( bu_render, 'render_one_chunk',
                                side_effect=lambda plan, *a, **k: rendered.append( plan.index ) or real_chunk( plan, *a, **k ) ):
            self.assertEqual( self.run_one(), 'processed' )
        self.assertNotIn( 0, rendered, "the finished chunk is reused" )
        resumed = _decode( os.path.join( self.censored, self.outputs()[0] ) )
        self.assertEqual( len( resumed ), len( clean ) )
        worst = max( float( np.abs( a - b ).mean() ) for a, b in zip( clean, resumed ) )
        self.assertLess( worst, 1.0, "a resumed render matches an uninterrupted one" )


if __name__ == '__main__':
    unittest.main()


# ---------------------------------------------------------------------------
# betabench persons
# ---------------------------------------------------------------------------

class TestPersonBench( Harness ):

    def test_every_backend_is_listed_timed_or_skipped_and_previews_are_written( self ):
        from tests.test_structure_bench import betabench, _quiet_logger
        tmp = tempfile.mkdtemp( prefix='betasuite-bench-' )
        self.addCleanup( shutil.rmtree, tmp, True )
        betaconfig.person_backend = { 'selected': 'yolo11_seg',
                                      'yolo11_seg': { 'model_path': os.path.join( tmp, 'here.onnx' ) } }
        open( os.path.join( tmp, 'here.onnx' ), 'w' ).close()
        args = betabench.build_parser().parse_args(
            [ 'persons', '--video', 'clip.mp4', '--person-variants', 's', '--batch-sizes', '1', '2',
              '--trials', '1', '--max-frames', '4' ] )
        args.run_dir = tmp
        frames = [ np.full( ( 90, 160, 3 ), 40 * k, np.uint8 ) for k in range( 4 ) ]
        fake = mock.Mock( wraps=yolo11_seg, VARIANTS=yolo11_seg.VARIANTS )
        fake.get_session.return_value = mock.Mock( **{ 'get_providers.return_value': [ 'CPUExecutionProvider' ] } )
        fake.persons_for_imgs.side_effect = lambda imgs, session, ts, config: [
            _person( 10, 10, 30, 60, t=t ) for t in ts ]
        real_get = bu_person.get_segmenter
        with mock.patch.object( betabench, '_sample_frames', return_value=frames ), \
             mock.patch.object( bu_person, 'get_segmenter',
                                side_effect=lambda name=None: fake if name == 'yolo11_seg' else real_get( name ) ):
            result = betabench.bench_persons( args, _quiet_logger() )
        rows = result['persons']['rows']
        timed = [ row for row in rows if 'ms_per_frame' in row ]
        self.assertEqual( [ ( row['variant'], row['batch_size'], row['people'] ) for row in timed ],
                          [ ( 's', 1, 4 ), ( 's', 2, 4 ) ] )
        self.assertIn( { 'backend': 'rf_detr_seg', 'variant': 's', 'skipped': 'unknown variant' }, rows,
                       "a size only YOLO has is skipped for RF-DETR, not a crash" )
        previews = os.listdir( os.path.join( tmp, 'person_previews', 'yolo11_seg-s' ) )
        self.assertEqual( len( previews ), 4 )
        self.assertNotIn( bu_person.VARIANT_OVERRIDE_ENV_VAR, os.environ, "the sweep's override is undone" )

    def test_a_missing_model_is_skipped_with_the_fix( self ):
        from tests.test_structure_bench import betabench, _quiet_logger
        betaconfig.person_backend = { 'selected': 'yolo11_seg', 'yolo11_seg': { 'model_path': '/nowhere.onnx' } }
        args = betabench.build_parser().parse_args( [ 'persons', '--video', 'clip.mp4', '--person-backends',
                                                      'yolo11_seg', '--person-variants', 'n' ] )
        args.run_dir = tempfile.mkdtemp( prefix='betasuite-bench-' )
        self.addCleanup( shutil.rmtree, args.run_dir, True )
        with mock.patch.object( betabench, '_sample_frames', return_value=[ np.zeros( ( 9, 16, 3 ), np.uint8 ) ] ):
            result = betabench.bench_persons( args, _quiet_logger() )
        self.assertEqual( result['persons']['rows'], [ { 'backend': 'yolo11_seg', 'variant': 'n',
                                                         'skipped': 'model missing' } ] )

    def test_rf_detr_seg_is_timed_beside_yolo_not_skipped( self ):
        from tests.test_structure_bench import betabench, _quiet_logger
        from segmenters import rf_detr_seg
        tmp = tempfile.mkdtemp( prefix='betasuite-bench-' )
        self.addCleanup( shutil.rmtree, tmp, True )
        here = os.path.join( tmp, 'here.onnx' )
        open( here, 'w' ).close()
        betaconfig.person_backend = { 'selected': 'yolo11_seg', 'yolo11_seg': { 'model_path': here },
                                      'rf_detr_seg': { 'model_path': here } }
        args = betabench.build_parser().parse_args(
            [ 'persons', '--video', 'clip.mp4', '--person-variants', 's', 'small', '--batch-sizes', '2',
              '--trials', '1', '--max-frames', '4' ] )
        args.run_dir = tmp
        fakes = {}
        for name, module in ( ( 'yolo11_seg', yolo11_seg ), ( 'rf_detr_seg', rf_detr_seg ) ):
            fake = mock.Mock( wraps=module, VARIANTS=module.VARIANTS )
            fake.get_session.return_value = mock.Mock( **{ 'get_providers.return_value': [ 'CPUExecutionProvider' ] } )
            fake.persons_for_imgs.side_effect = lambda imgs, session, ts, config: [
                _person( 10, 10, 30, 60, t=t ) for t in ts ]
            fakes[name] = fake
        frames = [ np.full( ( 90, 160, 3 ), 40 * k, np.uint8 ) for k in range( 4 ) ]
        with mock.patch.object( betabench, '_sample_frames', return_value=frames ), \
             mock.patch.object( bu_person, 'get_segmenter', side_effect=lambda name=None: fakes[name] ):
            rows = betabench.bench_persons( args, _quiet_logger() )['persons']['rows']
        self.assertEqual( sorted( ( row['backend'], row['variant'] ) for row in rows if 'ms_per_frame' in row ),
                          [ ( 'rf_detr_seg', 'small' ), ( 'yolo11_seg', 's' ) ] )
        self.assertEqual( sorted( ( row['backend'], row['variant'] ) for row in rows if 'skipped' in row ),
                          [ ( 'rf_detr_seg', 's' ), ( 'yolo11_seg', 'small' ) ] )
        self.assertEqual( len( os.listdir( os.path.join( tmp, 'person_previews', 'rf_detr_seg-small' ) ) ), 4 )


class TestBetastarePersonWindows( Harness ):

    def test_a_still_shows_a_mans_body_inside_the_obscured_frame( self ):
        import betastare
        tmp = tempfile.mkdtemp( prefix='betasuite-stare-' )
        self.addCleanup( shutil.rmtree, tmp, True )
        source_dir, out_dir = os.path.join( tmp, 'in' ), os.path.join( tmp, 'out' )
        os.makedirs( source_dir )
        os.makedirs( out_dir )
        source = np.random.RandomState( 5 ).randint( 0, 256, ( 120, 160, 3 ), dtype=np.uint8 )
        cv2.imwrite( os.path.join( source_dir, 'still.png' ), source )
        self.set_reverse( windows={}, censor={ 'labels': [ 'exposed_breast' ], 'overrides': {
            'exposed_breast': { 'min_prob': 0.2, 'censor_shape': 'box',
                                'censor_style': { 'type': 'bar', 'color': ( 255, 255, 255 ) } } } } )
        raw = [ _raw( 'face_masc', 10, 10, 20, 20 ), _raw( 'exposed_breast', 100, 60, 30, 30 ) ]
        with mock.patch.object( betastare, '_raw_boxes_for_all_sizes', return_value=( [ raw ], False ) ), \
             mock.patch.object( betastare.bu_detector, 'get_picture_sizes', return_value=[ 320 ] ), \
             mock.patch.object( bu_person, 'persons_for_imgs',
                                return_value=[ _person( 0, 0, 40, 100 ) ] ) as persons:
            betastare._process_one_file( source_dir, 'still.png', out_dir, 1, 1, None,
                                         betastare.bu_log.get_logger() )
        persons.assert_called_once()
        out = cv2.imread( os.path.join( out_dir, os.listdir( out_dir )[0] ) )
        np.testing.assert_array_equal( out[ 5:90, 3:35 ], source[ 5:90, 3:35 ] )
        self.assertTrue( np.all( out[ 105:, 50: ] == 0 ), "the rest is obscured" )
        self.assertTrue( np.all( out[ 62:88, 102:128 ] == 255 ) )


class TestPassMemory( Harness ):
    """The pass holds every mask of a video at once; these keep that small."""

    def test_masks_are_packed_as_they_are_decoded( self ):
        import betautils_video as bu_video
        source = mock.Mock( vid_fps=10.0, num_frames=30 )
        frames = [ ( index, index / 10.0, np.zeros( ( 9, 16, 3 ), np.uint8 ) ) for index in range( 3 ) ]
        saved = {}
        with mock.patch.object( bu_video, 'iter_sampled_frames', return_value=iter( frames ) ), \
             mock.patch.object( bu_person, 'get_session', return_value=object() ), \
             mock.patch.object( bu_person, 'persons_for_imgs',
                                side_effect=lambda imgs, ts, *a: [ _person( 0, 0, 5, 5, t=t ) for t in ts ] ), \
             mock.patch.object( bu_person, 'save_cache',
                                side_effect=lambda path, times, people: saved.update( people=people ) ), \
             mock.patch.object( bu_person.os.path, 'exists', return_value=False ):
            times, people = bu_person.persons_for_video( source, 'clip', 'abc', '', 0.0, None,
                                                         mock.Mock(), 10.0 )
        self.assertEqual( times, [ 0.0, 0.1, 0.2 ] )
        self.assertEqual( len( people ), 3 )
        self.assertTrue( all( 'mask' not in person and 'mask_bits' in person for person in people ) )

    def test_the_background_pass_hands_its_result_over_once( self ):
        with mock.patch.object( bu_person, 'persons_for_video', return_value=( [ 0.0 ], [ 'p' ] ) ):
            background = bu_person.BackgroundPersonPass( 1, 2 )
            self.assertEqual( background.result(), ( [ 0.0 ], [ 'p' ] ) )
        self.assertIsNone( background._result )

    def test_the_background_pass_raises_in_the_caller( self ):
        with mock.patch.object( bu_person, 'persons_for_video', side_effect=FileNotFoundError( 'model' ) ):
            background = bu_person.BackgroundPersonPass()
            with self.assertRaises( FileNotFoundError ):
                background.result()


# ---------------------------------------------------------------------------
# Resuming the person pass
# ---------------------------------------------------------------------------

class TestPassResume( Harness ):

    def setUp( self ):
        super().setUp()
        import betautils_signals as bu_signals
        self.signals = bu_signals
        bu_signals.clear()
        self.addCleanup( bu_signals.clear )
        self.tmp = tempfile.mkdtemp( prefix='betasuite-resume-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.cache_path = os.path.join( self.tmp, 'p.npz' )
        self.saved_interval = getattr( betaconfig, 'detection_checkpoint_frames', _MISSING )
        self.addCleanup( self._restore_interval )
        # frames() requests the interrupt from inside the frame generator,
        # which a prefetch thread runs ahead of the loop, so the interrupt
        # would land samples early. A real Ctrl-C is checked by the loop
        # itself either way; this pins the generator to the loop's pace.
        saved_prefetch = getattr( betaconfig, 'detection_prefetch_frames', _MISSING )
        betaconfig.detection_prefetch_frames = 0
        self.addCleanup( lambda: setattr( betaconfig, 'detection_prefetch_frames', saved_prefetch )
                         if saved_prefetch is not _MISSING
                         else betaconfig.__dict__.pop( 'detection_prefetch_frames', None ) )
        betaconfig.person_backend = { 'selected': 'yolo11_seg', 'yolo11_seg': { 'batch_size': 3 } }
        self.starts = []
        self.interrupt_at = None

    def _restore_interval( self ):
        if self.saved_interval is _MISSING:
            if hasattr( betaconfig, 'detection_checkpoint_frames' ):
                delattr( betaconfig, 'detection_checkpoint_frames' )
        else:
            betaconfig.detection_checkpoint_frames = self.saved_interval

    def frames( self, capture, vid_fps, sample_fps, start_sample_index=0, **kwargs ):
        self.starts.append( start_sample_index )
        for index in range( start_sample_index, 10 ):
            if index == self.interrupt_at:
                self.signals.request_interrupt()
            yield index, round( index / 10.0, 3 ), np.full( ( 9, 16, 3 ), index, np.uint8 )

    def run_pass( self, stop_event=None ):
        import betautils_video as bu_video
        source = mock.Mock( vid_fps=10.0, num_frames=10 )
        with mock.patch.object( bu_video, 'iter_sampled_frames', side_effect=self.frames ), \
             mock.patch.object( bu_person, 'get_session', return_value=object() ), \
             mock.patch.object( bu_person, 'persons_for_imgs',
                                side_effect=lambda imgs, ts, *a: [ _person( 0, 0, 5, 5, t=t, score=float( img[0, 0, 0] ) )
                                                                   for img, t in zip( imgs, ts ) ] ), \
             mock.patch( 'betautils_cache_paths.person_cache_path_for', return_value=self.cache_path ):
            return bu_person.persons_for_video( source, 'clip', 'abc', '', 0.0, None, mock.Mock(), 10.0,
                                                stop_event=stop_event )

    @staticmethod
    def summary( result ):
        times, people = result
        return times, [ ( p['t'], p['score'] ) for p in people ]

    def test_an_interrupted_pass_resumes_where_it_stopped_and_matches_a_clean_one( self ):
        clean = self.summary( self.run_pass() )
        os.remove( self.cache_path )
        self.starts.clear()
        self.interrupt_at = 5
        with self.assertRaises( self.signals.Interrupted ):
            self.run_pass()
        self.assertTrue( os.path.exists( bu_person.checkpoint_path_for( self.cache_path ) ) )
        self.assertFalse( os.path.exists( self.cache_path ) )
        self.signals.clear()
        self.interrupt_at = None
        resumed = self.summary( self.run_pass() )
        self.assertEqual( self.starts, [ 0, 5 ] )
        self.assertEqual( resumed, clean )
        self.assertFalse( os.path.exists( bu_person.checkpoint_path_for( self.cache_path ) ),
                          "the checkpoint goes once the cache is written" )

    def test_a_checkpoint_is_written_as_it_goes( self ):
        # Batches of 2 against checkpoints every 3: a checkpoint may only
        # land with nothing buffered, or it would claim samples whose
        # people it does not hold.
        betaconfig.detection_checkpoint_frames = 3
        betaconfig.person_backend['yolo11_seg']['batch_size'] = 2
        written = []
        real_save = bu_person.save_cache
        with mock.patch.object( bu_person, 'save_cache',
                                side_effect=lambda path, times, people, next_index=None:
                                written.append( ( path.endswith( '.checkpoint.npz' ), next_index,
                                                  len( times ), len( people ) ) )
                                or real_save( path, times, people, next_index ) ):
            self.run_pass()
        checkpoints = [ entry[1:] for entry in written if entry[0] ]
        self.assertEqual( [ next_index for next_index, _t, _p in checkpoints ], [ 4, 8 ] )
        self.assertTrue( all( next_index == times == people for next_index, times, people in checkpoints ),
                         checkpoints )

    def test_a_stopped_pass_checkpoints_and_ends_quietly( self ):
        import threading
        stop = threading.Event()
        stop.set()
        with self.assertRaises( bu_person.Stopped ):
            self.run_pass( stop )
        self.assertTrue( os.path.exists( bu_person.checkpoint_path_for( self.cache_path ) ) )

    def test_stop_waits_for_the_background_pass_to_checkpoint( self ):
        import threading
        started, release = threading.Event(), threading.Event()

        def slow_pass( *args, stop_event=None ):
            started.set()
            release.wait( 5 )
            if stop_event.is_set():
                raise bu_person.Stopped()
            return 'done'
        with mock.patch.object( bu_person, 'persons_for_video', side_effect=slow_pass ):
            background = bu_person.BackgroundPersonPass()
            started.wait( 5 )
            threading.Timer( 0.2, release.set ).start()
            background.stop()
        self.assertFalse( background._thread.is_alive() )
        self.assertIsNone( background._error, "being stopped is not a failure" )


class TestStyleSeed( unittest.TestCase ):

    def test_a_seeded_run_rolls_the_same_looks_and_leaves_random_alone( self ):
        import random
        import betautils_censor as censor
        styles = [ { 'type': 'bar', 'color': ( k, k, k ) } for k in range( 50 ) ]

        def fake_prepare( *args ):
            return [ censor.resolve_censor_style( styles )['color'] for _ in range( 5 ) ], {}
        import betautils_track as bu_track
        with mock.patch.object( bu_track, '_prepare_boxes_for_render', side_effect=fake_prepare ):
            random.seed( 1 )
            first = bu_track.prepare_boxes_for_render( [], 10, 10, style_seed='abc' )[0]
            random.seed( 2 )
            second = bu_track.prepare_boxes_for_render( [], 10, 10, style_seed='abc' )[0]
            other = bu_track.prepare_boxes_for_render( [], 10, 10, style_seed='xyz' )[0]
            random.seed( 1 )
            expected = random.random()
            random.seed( 1 )
            bu_track.prepare_boxes_for_render( [], 10, 10, style_seed='abc' )
            self.assertEqual( random.random(), expected, "the caller's random state is put back" )
        self.assertEqual( first, second )
        self.assertNotEqual( first, other )

