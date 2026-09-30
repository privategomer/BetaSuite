"""
test_structure_bench.py - the per-video structure classifier, and the
render bench's style deduplication.

WHY THESE TWO TOGETHER
----------------------
Both are betabench changes whose failure mode is silence rather than an
error. The structure bench can report a misleading shape without
crashing; the render dedup can collapse two styles that do NOT cost the
same, which quietly removes a row from the cost table and hides an
expensive style.

WHAT STRUCTURE IS FOR
---------------------
Every other subcommand pools all footage into one distribution. That is
right for "what does this model do" and wrong for "does this file need
different settings from that one" - a compilation and a long
single-scene clip average into a description of neither.

The three structural shapes it has to tell apart, all readable from
caches a real run already wrote:

  compilation     many cuts per minute, very short median shot
  single-scene    few cuts, long shots, boxes on one side of frame
  split-screen    two boxes live at once, split evenly between halves

These are structural, not content labels: "solo" and "couple" cannot be
recovered from detections, but cut rate and spatial layout can, and those
are what actually want different tracking and timing values.
"""

import importlib.util
import logging
import os
import sys
import tempfile
import types
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_censor as bu_censor
import betautils_hash as bu_hash
import betautils_tuning as bu_tuning


def _load_betabench():
    path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                         'tools', 'bench', 'betabench.py' )
    spec = importlib.util.spec_from_file_location( 'betabench_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


betabench = _load_betabench()


def _quiet_logger():
    logger = logging.getLogger( 'structure_test' )
    logger.handlers = [ logging.NullHandler() ]
    logger.setLevel( logging.CRITICAL )
    return logger


def _box( file_hash, t, x, y, w=200, h=200, label='exposed_breast' ):
    return { '_file_hash': file_hash, 't': t, 'x': x, 'y': y, 'w': w, 'h': h,
             'class_id': label, 'score': 0.8 }


class TestStructureBench( unittest.TestCase ):

    def setUp( self ):
        self.logger = _quiet_logger()
        self.args = types.SimpleNamespace( sample_fps=None, frame_width=1920, frame_height=1080 )
        self.config = types.SimpleNamespace( label='test' )
        self._orig_hash_to_path = betabench.bu_cache.build_hash_to_video_path
        self._orig_frame_size = betabench.bu_tuning.frame_size_for_video
        betabench.bu_cache.build_hash_to_video_path = (
            lambda hashes: { h: '/fake/%s.mp4'%(h) for h in hashes } )
        betabench.bu_tuning.frame_size_for_video = lambda path: ( 1920, 1080 )

    def tearDown( self ):
        betabench.bu_cache.build_hash_to_video_path = self._orig_hash_to_path
        betabench.bu_tuning.frame_size_for_video = self._orig_frame_size

    def _rows( self, raw_boxes ):
        produced = betabench._bench_structure_one(
            self.args, self.logger, self.config, raw_boxes )
        return { row['file_hash']: row for row in produced.get( 'structure', [] ) }

    def test_split_screen_shows_two_simultaneous_boxes_split_evenly( self ):
        raw = []
        for i in range( 200 ):
            raw.append( _box( 'split', i / 9.0, 300, 400 ) )
            raw.append( _box( 'split', i / 9.0, 1300, 400 ) )
        row = self._rows( raw )['split']
        self.assertEqual( row['median_simultaneous'], 2 )
        self.assertAlmostEqual( row['left_half_fraction'], 0.5, places=2 )

    def test_one_subject_on_one_side_is_not_mistaken_for_split_screen( self ):
        raw = [ _box( 'solo', i / 9.0, 200 + i, 400 ) for i in range( 200 ) ]
        row = self._rows( raw )['solo']
        self.assertEqual( row['median_simultaneous'], 1 )
        self.assertGreater( row['left_half_fraction'], 0.9 )

    def test_cut_rate_separates_a_compilation_from_a_single_scene( self ):
        tmp = tempfile.mkdtemp()
        saved_dir = betaconst.shot_cut_dir
        bu_cache.betaconst.shot_cut_dir = tmp
        try:
            fps = betaconfig.video_censor_fps
            threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
            bu_hash.write_json( [ 5.0, 15.0 ],
                                bu_cache.shot_cut_path_for( 'slow', fps, threshold ) )
            bu_hash.write_json( [ round( i * 0.3, 2 ) for i in range( 1, 70 ) ],
                                bu_cache.shot_cut_path_for( 'fast', fps, threshold ) )
            raw = []
            for i in range( 200 ):
                raw.append( _box( 'slow', i / 9.0, 700, 400 ) )
                raw.append( _box( 'fast', i / 9.0, ( i * 211 ) % 1600, 400 ) )
            rows = self._rows( raw )
            self.assertGreater( rows['fast']['cuts_per_min'], rows['slow']['cuts_per_min'] * 5,
                "the compilation's cut rate did not separate from the single-scene file's" )
            self.assertLess( rows['fast']['median_shot_seconds'],
                             rows['slow']['median_shot_seconds'] )
        finally:
            bu_cache.betaconst.shot_cut_dir = saved_dir

    def test_a_file_with_no_shot_cut_cache_reports_unknown_not_zero( self ):
        # "no cuts detected" and "cuts never looked for" are different
        # facts, and reporting the second as the first would make a file
        # scanned without shot-cut detection look like a single long take.
        tmp = tempfile.mkdtemp()
        saved_dir = betaconst.shot_cut_dir
        bu_cache.betaconst.shot_cut_dir = tmp
        try:
            raw = [ _box( 'nocuts', i / 9.0, 700, 400 ) for i in range( 50 ) ]
            row = self._rows( raw )['nocuts']
            self.assertIsNone( row['cuts_per_min'] )
            self.assertIsNone( row['cuts'] )
        finally:
            bu_cache.betaconst.shot_cut_dir = saved_dir

    def test_only_censored_labels_are_counted( self ):
        # Structure describes what gets CENSORED. Counting face_femme
        # into 'simultaneous' would report every talking-head shot as
        # multi-subject.
        raw = [ _box( 'mixed', i / 9.0, 700, 400 ) for i in range( 50 ) ]
        raw += [ _box( 'mixed', i / 9.0, 100, 100, label='face_femme' ) for i in range( 50 ) ]
        row = self._rows( raw )['mixed']
        self.assertEqual( row['detections'], 50 )
        self.assertEqual( row['median_simultaneous'], 1 )

    def test_untagged_detections_produce_no_rows_rather_than_a_crash( self ):
        raw = [ { 't': 0.0, 'x': 1, 'y': 1, 'w': 10, 'h': 10, 'class_id': 'exposed_breast' } ]
        produced = betabench._bench_structure_one(
            self.args, self.logger, self.config, raw )
        self.assertEqual( produced, {} )


class TestRenderBenchStyleDedup( unittest.TestCase ):
    """
    The render bench times every configured style at every box size. When
    exposed_breast grew to 33 entries that became 17220 censor_image
    calls, each copying a frame.

    Entries that differ only in area safety or weight render identically
    - area safety resizes the box, and box size is already swept
    separately - so collapsing them removes duplicate measurements, not
    information. Collapsing two entries that DO cost differently would
    hide an expensive style, so the key must keep everything that drives
    pixel work.
    """

    def _key( self, owner, style ):
        # Mirrors _cost_key in bench_render.
        return (
            style.get( 'type' ),
            style.get( 'pattern' ) or style.get( 'method' ) or '',
            style.get( 'strength' ),
            round( float( style.get( 'feather', 0 ) ), 4 ),
            bu_censor.resolve_censor_shape(
                style, getattr( betaconfig, 'default_censor_shape', 'box' ) ),
            style.get( 'dir' ),
            style.get( 'thickness' ),
            owner,
        )

    def test_entries_differing_only_in_area_safety_collapse( self ):
        a = { 'type': 'pixel', 'pattern': 'hex', 'strength': 30, 'feather': 0.27,
              'width_area_safety': -0.10, 'height_area_safety': 0.00, 'weight': 1 }
        b = dict( a, width_area_safety=-0.50, height_area_safety=-0.25, weight=0.6364 )
        self.assertEqual( self._key( 'exposed_breast', a ), self._key( 'exposed_breast', b ) )

    def test_different_strength_does_not_collapse( self ):
        a = { 'type': 'blur', 'method': 'gaussian', 'strength': 30, 'feather': 0.27 }
        b = dict( a, strength=50 )
        self.assertNotEqual( self._key( 'exposed_breast', a ), self._key( 'exposed_breast', b ) )

    def test_different_feather_does_not_collapse( self ):
        a = { 'type': 'blur', 'method': 'gaussian', 'strength': 30, 'feather': 0.27 }
        b = dict( a, feather=0.55 )
        self.assertNotEqual( self._key( 'exposed_breast', a ), self._key( 'exposed_breast', b ) )

    def test_different_method_does_not_collapse( self ):
        a = { 'type': 'blur', 'method': 'gaussian', 'strength': 30, 'feather': 0.27 }
        b = dict( a, method='box' )
        self.assertNotEqual( self._key( 'exposed_breast', a ), self._key( 'exposed_breast', b ) )

    def test_different_sticker_directory_does_not_collapse( self ):
        # A sticker's cost is dominated by which images it loads.
        a = { 'type': 'sticker', 'dir': '../resources/stickers/breasts/', 'feather': 0.25 }
        b = dict( a, dir='../resources/stickers/vulva/' )
        self.assertNotEqual( self._key( 'exposed_breast', a ), self._key( 'exposed_breast', b ) )

    def test_different_bar_thickness_does_not_collapse( self ):
        a = { 'type': 'bar', 'shape': 'box', 'thickness': 0.35, 'feather': 0.15 }
        b = dict( a, thickness=1.0 )
        self.assertNotEqual( self._key( 'exposed_breast', a ), self._key( 'exposed_breast', b ) )

    def test_the_shipped_config_still_collapses_something( self ):
        # A regression pin. If this stops collapsing anything, either the
        # config lost its duplicate-cost entries (fine, delete this) or
        # the key picked up a field that makes every entry unique (not
        # fine - the bench is back to timing duplicates).
        styles = []
        for label, override in getattr( betaconfig, 'item_overrides', {} ).items():
            for entry in override.get( 'censor_style', [] ) or []:
                styles.append( ( label, entry ) )
        distinct = { self._key( owner, style ) for owner, style in styles }
        self.assertLess( len( distinct ), len( styles ),
            "no style entries collapsed; the render bench is timing duplicates again" )


if __name__ == '__main__':
    unittest.main()
