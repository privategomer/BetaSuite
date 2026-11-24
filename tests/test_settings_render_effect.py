"""
test_settings_render_effect.py - a setting that moves no censor key
does not change what gets rendered, and one that moves no detection key
does not change what gets detected.

test_settings_cache_tags.py proves each entry moves exactly the keys it
declares. That cannot catch a tag missing because the KEY is missing a
setting: a setting that changes the output but neither its tag nor the
key says so moves nothing, and passes. This file looks at the output.

Censor stage: fixed synthetic detections and a shot cut go through the
real pipeline (prepare_boxes_for_render, the motion index,
interpolate_boxes_for_frame, censor_img_for_boxes) onto synthetic
frames, and the pixels are hashed. Detection stage: a stub model with
overlapping candidates at many scores, through each adapter's real
decode, NMS and floors. For every entry not tagged censor (detection),
changing it to its probe must leave that hash unchanged.

This proves invariance on this input, not on all footage: a setting
whose effect this input never reaches would pass too. Entries tagged
censor whose change this input does not show are allowed, since keying
too much costs a re-render and never a wrong result.
"""

import copy
import hashlib
import unittest

import numpy as np

import betaconfig
import betautils_censor as bu_censor
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_render as bu_render
import betautils_settings as bu_settings
import betautils_track as bu_track
import detectors.nudenet_v3 as nudenet_v3
import detectors.retinanet_v2 as retinanet_v2

from tests import config_harness as harness
from tests import test_settings_cache_tags as tags

WIDTH, HEIGHT = 96, 64
SAMPLE_FPS = 10
RENDER_FPS = 15
SECONDS = 3.0
SHOT_CUTS = [ 1.5 ]


def synthetic_detections():
    """Two breasts that pair, a vulva with a gap, an overlapping weaker label, all moving."""
    raw = []
    for index in range( int( SECONDS*SAMPLE_FPS ) ):
        t = index / SAMPLE_FPS
        drift = index*0.6
        raw.append( { 'x': 8 + drift, 'y': 10, 'w': 16, 'h': 14, 'class_id': 'exposed_breast',
                      'score': 0.55 + 0.04*( index % 5 ), 't': t, 'size': 320 } )
        if index % 4:
            raw.append( { 'x': 30 + drift, 'y': 11, 'w': 15, 'h': 13, 'class_id': 'exposed_breast',
                          'score': 0.45 + 0.05*( index % 3 ), 't': t, 'size': 320 } )
        if not 12 <= index <= 15:
            raw.append( { 'x': 50, 'y': 36 + ( index % 3 ), 'w': 18, 'h': 16, 'class_id': 'exposed_vulva',
                          'score': 0.62, 't': t, 'size': 320 } )
        raw.append( { 'x': 9 + drift, 'y': 10, 'w': 15, 'h': 14, 'class_id': 'covered_breast',
                      'score': 0.3, 't': t, 'size': 320 } )
    return raw


def _frame():
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    image = np.stack( [ ( x*7 + y*3 ) % 256, ( x*2 + y*11 ) % 256, ( x*13 ) % 256 ], axis=-1 )
    return image.astype( np.uint8 )


def censor_fingerprint():
    """The rendered pixels of the synthetic clip under the current config."""
    bu_config.invalidate_config_caches()
    boxes, _stats = bu_track.prepare_boxes_for_render(
        copy.deepcopy( synthetic_detections() ), WIDTH, HEIGHT, shot_cut_times=list( SHOT_CUTS ),
        style_seed='render-effect' )
    enabled, span, growth, window = bu_render._motion_settings( None )
    timelines = bu_render.build_motion_index( boxes ) if enabled else {}
    base = _frame()
    digest = hashlib.sha256()
    for frame_index in range( int( SECONDS*RENDER_FPS ) ):
        t = frame_index / RENDER_FPS
        live = [ box for box in boxes if box['start'] <= t <= box['end'] ]
        frame_boxes = ( bu_render.interpolate_boxes_for_frame( live, timelines, t, span, growth, window )
                        if enabled else live )
        digest.update( bu_censor.censor_img_for_boxes( base.copy(), frame_boxes ).tobytes() )
    return digest.hexdigest()


class _NudeNetCandidates:
    """Several overlapping candidates per image at many scores, so floors and NMS matter."""

    def get_inputs( self ):
        class _Input:
            name = 'images'
        return [ _Input() ]

    def run( self, _output_names, feed ):
        blob = feed[ list( feed )[0] ]
        output = np.zeros( ( blob.shape[0], 22, 64 ), dtype=np.float32 )
        for index in range( blob.shape[0] ):
            seed = int( abs( float( blob[index].sum() ) ) ) % 7
            for anchor in range( 12 ):
                output[index, 0, anchor] = 100 + 6*( anchor % 4 ) + seed
                output[index, 1, anchor] = 120 + 5*( anchor // 4 )
                output[index, 2, anchor] = 40 + anchor
                output[index, 3, anchor] = 36 + anchor
                # Three overlapping candidates per class, so NMS has something to remove.
                output[index, 4 + ( anchor // 3 + seed ) % 18, anchor] = 0.08 + 0.075*anchor
        return [ output ]


class _RetinaNetCandidates:
    def get_inputs( self ):
        class _Input:
            name = 'input'
        return [ _Input() ]

    def run( self, _output_names, feed ):
        images = feed[ list( feed )[0] ]
        count = retinanet_v2.DETECTIONS_PER_IMAGE
        boxes = np.zeros( ( len( images ), count, 4 ), dtype=np.float32 )
        scores = np.zeros( ( len( images ), count ), dtype=np.float32 )
        classes = np.full( ( len( images ), count ), -1, dtype=np.int32 )
        for index, image in enumerate( images ):
            seed = int( abs( float( np.asarray( image ).sum() ) ) ) % 7
            for detection in range( 10 ):
                boxes[index][detection] = [ 10 + 3*detection + seed, 12 + detection, 50 + 3*detection, 60 ]
                scores[index][detection] = 0.08 + 0.09*detection
                classes[index][detection] = ( seed + detection ) % len( retinanet_v2._MODEL_CLASS_ORDER )
        return [ boxes, scores, classes ]


def detection_fingerprint():
    """Each backend's raw detections for fixed frames under the current config."""
    bu_config.invalidate_config_caches()
    frames = [ np.full( ( 48, 64, 3 ), value, dtype=np.uint8 ) for value in ( 10, 70, 140, 210 ) ]
    stamps = [ index / SAMPLE_FPS for index in range( len( frames ) ) ]
    out = {}
    for name, module, session in ( ( 'nudenet_v3', nudenet_v3, _NudeNetCandidates() ),
                                   ( 'retinanet_v2', retinanet_v2, _RetinaNetCandidates() ) ):
        sizes = bu_detector.get_picture_sizes( name ) or [ 320 ]
        boxes = module.raw_boxes_for_imgs( frames, sizes[0], session, stamps )
        out[name] = sorted( ( round( box['x'], 3 ), round( box['y'], 3 ), round( box['w'], 3 ),
                              round( box['h'], 3 ), box['class_id'], round( float( box['score'] ), 4 ) )
                            for box in boxes )
    return hashlib.sha256( repr( out ).encode() ).hexdigest()


def effect_of( entry, path, bindings, fingerprint ):
    """( before, after ) of a fingerprint when this entry changes to its probe."""
    with harness.shipped():
        for condition in entry.when:
            tags.set_up( condition, bindings, True, path )
        before = fingerprint()
        try:
            current = harness.get_path( path )
        except ( KeyError, IndexError, TypeError ):
            current = tags._MISSING
        harness.set_path( path, tags.probe_for( entry, path, current, bindings ) )
        after = fingerprint()
    return before, after


class TestFingerprintsSeeChanges( unittest.TestCase ):
    """The fingerprints are not blind: a change that must show on this input does."""

    def changed( self, fingerprint, change ):
        with harness.shipped():
            before = fingerprint()
            change()
            bu_config.invalidate_config_caches()
            after = fingerprint()
        return before != after

    def test_the_censor_fingerprint_sees_a_look_change( self ):
        self.assertTrue( self.changed( censor_fingerprint, lambda: betaconfig.item_overrides['exposed_breast'].__setitem__(
            'censor_style', { 'type': 'bar', 'color': ( 255, 0, 0 ) } ) ) )

    def test_the_censor_fingerprint_sees_a_confidence_change( self ):
        def raise_the_floor():
            overrides = betaconfig.detector_backend['nudenet_v3']['item_overrides']
            overrides['exposed_vulva']['min_prob'] = 0.9
        self.assertTrue( self.changed( censor_fingerprint, raise_the_floor ) )

    def test_the_censor_fingerprint_sees_a_motion_change( self ):
        self.assertTrue( self.changed( censor_fingerprint,
                                       lambda: setattr( betaconfig, 'render_motion_interpolation', False ) ) )

    def test_the_detection_fingerprint_sees_a_detection_change( self ):
        def backend():
            return betaconfig.detector_backend['nudenet_v3']
        for name, change in ( ( 'global_min_prob', lambda: setattr( betaconfig, 'global_min_prob', 0.5 ) ),
                              ( 'candidate_floor', lambda: backend().__setitem__( 'candidate_floor', 0.6 ) ),
                              ( 'nms_iou', lambda: backend().__setitem__( 'nms_iou', 0.05 ) ) ):
            self.assertTrue( self.changed( detection_fingerprint, change ), name )

    def test_the_fingerprints_are_repeatable( self ):
        with harness.shipped():
            self.assertEqual( censor_fingerprint(), censor_fingerprint() )
            self.assertEqual( detection_fingerprint(), detection_fingerprint() )


def one_per_backend( entry ):
    """One instantiation per backend (and person backend): the stage reads one label like another."""
    seen = set()
    for path, bindings in tags.instantiations( entry ):
        key = ( bindings.get( 'backend' ), bindings.get( 'person_backend' ) )
        if key not in seen:
            seen.add( key )
            yield path, bindings


class TestNoHiddenEffects( unittest.TestCase ):

    def check( self, stage, fingerprint ):
        changed = []
        for entry in bu_settings.REGISTRY:
            if stage in entry.moves:
                continue
            for path, bindings in one_per_backend( entry ):
                before, after = effect_of( entry, path, bindings, fingerprint )
                if before != after:
                    changed.append( '.'.join( path ) )
        return changed

    def test_no_setting_outside_the_censor_key_changes_the_render( self ):
        self.assertEqual( [], self.check( bu_settings.CENSOR, censor_fingerprint ) )

    def test_no_setting_outside_the_detection_key_changes_the_detections( self ):
        self.assertEqual( [], self.check( bu_settings.DETECTION, detection_fingerprint ) )


def keyed_but_not_shown():
    """Entries tagged censor whose probe does not change this render: listed in the evidence, not failed."""
    quiet = []
    for entry in bu_settings.REGISTRY:
        if bu_settings.CENSOR not in entry.moves:
            continue
        if { 'reverse_on', 'person_windows_on' } & set( entry.when ):
            quiet.append( entry.dotted + '  (reverse mode: its render path is not simulated here)' )
            continue
        # A label the synthetic clip has, and the selected backend, so a change can show.
        candidates = list( tags.instantiations( entry ) )
        path, bindings = next( ( ( p, b ) for p, b in candidates
                                 if b.get( 'label', 'exposed_breast' ) == 'exposed_breast'
                                 and b.get( 'backend', 'nudenet_v3' ) == 'nudenet_v3' ), candidates[0] )
        before, after = effect_of( entry, path, bindings, censor_fingerprint )
        if before == after:
            quiet.append( entry.dotted )
    return quiet


if __name__ == '__main__':
    unittest.main()
