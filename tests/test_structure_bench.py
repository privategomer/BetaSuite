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
import shutil
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


def _write_text( path, text ):
    with open( path, 'w' ) as handle:
        handle.write( text )


def _quiet_logger():
    logger = logging.getLogger( 'structure_test' )
    logger.handlers = [ logging.NullHandler() ]
    logger.setLevel( logging.CRITICAL )
    return logger


def _box( file_hash, t, x, y, w=200, h=200, label='exposed_breast' ):
    return { '_file_hash': file_hash, 't': t, 'x': x, 'y': y, 'w': w, 'h': h,
             'class_id': label, 'score': 0.8 }


def _build_clip( path, seconds ):
    """A tiny still clip: structure reads its duration, nothing else."""
    import subprocess
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    subprocess.run( [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                      'color=c=gray:size=64x48:rate=10:duration=%d'%( seconds, ),
                      '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', path ],
                    check=True )


@unittest.skipUnless( shutil.which( 'ffmpeg' ), "ffmpeg is needed to build the test clips" )
class StructureHarness( unittest.TestCase ):
    """
    Real files in a real pair of source trees, real cache files on disk,
    and bench_structure run end to end. The previous tests handed
    _bench_structure_one a list of boxes and a fake path, which is how
    the window bug shipped: the window was derived from those same boxes,
    so a test that spread its boxes across the clip could never see it.
    """

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-structure-' )
        self.vids = os.path.join( self.tmp, 'uncensored_vids' )
        self.staging = os.path.join( self.tmp, 'source' )
        os.makedirs( self.vids )
        os.makedirs( self.staging )
        self._saved_const = { name: getattr( betaconst, name ) for name in (
            'video_path_uncensored', 'video_path_source_backup', 'shot_cut_dir',
            'vid_hash_dir', 'file_hash_cache_path' ) }
        betaconst.video_path_uncensored = self.vids + os.sep
        betaconst.video_path_source_backup = self.staging + os.sep
        betaconst.shot_cut_dir = os.path.join( self.tmp, 'shot_cuts' )
        betaconst.vid_hash_dir = os.path.join( self.tmp, 'vid_hashes' )
        betaconst.file_hash_cache_path = os.path.join( self.tmp, 'file_hashes.json' )
        self._saved_vid_dir = bu_cache.VID_HASH_DIR
        bu_cache.VID_HASH_DIR = betaconst.vid_hash_dir
        self._saved_memo = bu_hash._file_hash_memo
        bu_hash._file_hash_memo = None
        self._saved_map = getattr( betaconfig, 'profile_by_source_dir', None )
        betaconfig.profile_by_source_dir = {
            'scene': 'scene', 'compilation': 'compilation',
            'quick-cut': 'quick_cut', 'split-screen': 'split_screen' }
        self._saved_frame_size = betabench.bu_tuning.frame_size_for_video
        betabench.bu_tuning.frame_size_for_video = lambda path: ( 1920, 1080 )
        betabench.bu_video._DURATION_CACHE.clear()
        self.fps = betaconfig.video_censor_fps
        self.threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
        self.backend = betaconfig.detector_backend['selected']
        self.logger = _quiet_logger()
        self.args = types.SimpleNamespace( backend=None, write_move_script=None )

    def tearDown( self ):
        for name, value in self._saved_const.items():
            setattr( betaconst, name, value )
        bu_cache.VID_HASH_DIR = self._saved_vid_dir
        bu_hash._file_hash_memo = self._saved_memo
        betaconfig.profile_by_source_dir = self._saved_map
        betabench.bu_tuning.frame_size_for_video = self._saved_frame_size
        shutil.rmtree( self.tmp, ignore_errors=True )

    def clip( self, relative, seconds=60, tree=None ):
        path = os.path.join( tree or self.vids, relative )
        _build_clip( path, seconds )
        return bu_hash.md5_for_file( path, 16 )

    def cuts( self, file_hash, cuts, preview_suffix='', max_seconds=None ):
        bu_hash.write_json( cuts, bu_cache.shot_cut_path_for(
            file_hash, self.fps, self.threshold, preview_suffix, max_seconds ) )

    def boxes( self, file_hash, boxes, size=640, fps=None, preview_suffix='' ):
        bu_hash.write_json( boxes, bu_cache.box_hash_path_for(
            file_hash, size, fps or self.fps, 0.12, self.backend,
            preview_suffix, det_key='abc123' ) )

    def rows( self ):
        produced = betabench.bench_structure( self.args, self.logger )
        return { row['source']: row for row in produced.get( 'structure', [] ) }


class TestStructureWindow( StructureHarness ):
    """The denominator is the footage scanned, never the detections' spread."""

    def test_boxes_in_one_frame_do_not_shrink_the_window( self ):
        # The exact failure: censored boxes in ONE sampled frame gave a
        # 1-microsecond window, 600,000,000 cuts/min, "100% of runtime in
        # sub-1s shots", and a recommended move to quick-cut.
        file_hash = self.clip( 'scene/petite.mp4', 60 )
        self.cuts( file_hash, [ 30.0 ] )
        self.boxes( file_hash, [ _box( file_hash, 10.0, 700, 400 ) ] )
        row = self.rows()['petite.mp4']
        self.assertEqual( row['scanned_seconds'], 60.0 )
        self.assertAlmostEqual( row['cuts_per_min'], 1.0 )
        self.assertEqual( row['short_shot_fraction'], 0.0 )
        self.assertEqual( row['recommended_profile'], 'scene' )

    def test_a_slice_is_measured_over_the_slice_and_says_so( self ):
        file_hash = self.clip( 'clip.mp4', 60 )
        self.cuts( file_hash, [ 5.0, 15.0 ], '-preview', 20 )
        row = self.rows()['clip.mp4']
        self.assertEqual( row['scanned_seconds'], 20.0 )
        self.assertEqual( row['timing_evidence'], 'slice' )
        self.assertAlmostEqual( row['cuts_per_min'], 6.0 )

    def test_a_whole_file_scan_wins_over_a_slice( self ):
        file_hash = self.clip( 'clip.mp4', 60 )
        self.cuts( file_hash, [ 1.0, 1.5, 2.0 ], '-preview', 20 )
        self.cuts( file_hash, [ 30.0 ] )
        row = self.rows()['clip.mp4']
        self.assertEqual( row['timing_evidence'], 'whole file' )
        self.assertEqual( row['cuts'], 1 )

    def test_zero_cuts_in_a_whole_file_scan_is_measured_as_scene( self ):
        file_hash = self.clip( 'clip.mp4', 60 )
        self.cuts( file_hash, [] )
        row = self.rows()['clip.mp4']
        self.assertEqual( row['cuts'], 0 )
        self.assertEqual( row['short_shot_fraction'], 0.0 )
        self.assertEqual( row['recommended_profile'], 'scene' )

    def test_no_scan_at_all_recommends_nothing( self ):
        self.clip( 'scene/clip.mp4', 60 )
        row = self.rows()['clip.mp4']
        self.assertIsNone( row['cuts'] )
        self.assertIsNone( row['timing_evidence'] )
        self.assertIsNone( row['recommended_profile'] )


class TestEveryFileIsSurveyed( StructureHarness ):
    """Files come from disk, not from one detection configuration's caches."""

    def test_a_file_detected_at_another_rate_still_appears_with_geometry( self ):
        # The 8 files the app measured as quick_cut were detected at 18
        # fps and were invisible to a survey reading only the 9 fps caches.
        file_hash = self.clip( 'fast.mp4', 60 )
        self.boxes( file_hash, [ _box( file_hash, i / 18.0, 700, 400 ) for i in range( 60 ) ],
                    fps=18 )
        row = self.rows()['fast.mp4']
        self.assertEqual( row['frames_with_boxes'], 60 )

    def test_a_file_with_no_caches_at_all_still_has_a_row( self ):
        self.clip( 'unseen.mp4', 60 )
        self.assertIn( 'unseen.mp4', self.rows() )

    def test_staging_tree_files_are_surveyed( self ):
        self.clip( 'scene/waiting.mp4', 60, tree=self.staging )
        row = self.rows()['waiting.mp4']
        self.assertEqual( row['tree'], 'source' )
        self.assertEqual( row['current_folder'], 'scene' )

    def test_two_configurations_are_not_counted_as_two_subjects( self ):
        # 320 and 640 see the same frame; mixing them would double p90.
        file_hash = self.clip( 'solo.mp4', 60 )
        self.boxes( file_hash, [ _box( file_hash, i / 9.0, 700, 400 ) for i in range( 50 ) ], size=640 )
        self.boxes( file_hash, [ _box( file_hash, i / 9.0, 710, 400 ) for i in range( 50 ) ], size=320 )
        self.assertEqual( self.rows()['solo.mp4']['median_simultaneous'], 1 )

    def test_the_configuration_that_saw_most_is_chosen_whatever_the_order( self ):
        # Two sizes, one sparse, one dense. Directory order is not
        # sorted, so both assignments are tried.
        for dense_size, sparse_size in ( ( 640, 320 ), ( 320, 640 ) ):
            with self.subTest( dense=dense_size ):
                shutil.rmtree( betaconst.vid_hash_dir, ignore_errors=True )
                file_hash = self.clip( 'pick%d.mp4'%( dense_size, ), 60 )
                self.boxes( file_hash, [ _box( file_hash, i / 9.0, 700, 400 ) for i in range( 50 ) ],
                            size=dense_size )
                self.boxes( file_hash, [ _box( file_hash, i / 9.0, 700, 400 ) for i in range( 5 ) ],
                            size=sparse_size )
                self.assertEqual( self.rows()['pick%d.mp4'%( dense_size, )]['frames_with_boxes'], 50 )

    def test_slices_of_one_configuration_are_combined( self ):
        file_hash = self.clip( 'sliced.mp4', 120 )
        for offset in ( 25.0, 50.0, 75.0 ):
            self.boxes( file_hash, [ _box( file_hash, offset + i / 9.0, 700, 400 ) for i in range( 20 ) ],
                        preview_suffix=bu_cache.preview_cache_suffix( True, offset ) )
        self.assertEqual( self.rows()['sliced.mp4']['frames_with_boxes'], 60 )


class TestStructureGeometry( StructureHarness ):

    def test_split_screen_shows_two_simultaneous_boxes_split_evenly( self ):
        file_hash = self.clip( 'split.mp4', 60 )
        raw = []
        for i in range( 200 ):
            raw.append( _box( file_hash, i / 9.0, 300, 400 ) )
            raw.append( _box( file_hash, i / 9.0, 1300, 400 ) )
        self.boxes( file_hash, raw )
        row = self.rows()['split.mp4']
        self.assertEqual( row['median_simultaneous'], 2 )
        self.assertAlmostEqual( row['left_half_fraction'], 0.5, places=2 )

    def test_one_subject_on_one_side_is_not_mistaken_for_split_screen( self ):
        file_hash = self.clip( 'solo.mp4', 60 )
        self.boxes( file_hash, [ _box( file_hash, i / 9.0, 200 + i, 400 ) for i in range( 200 ) ] )
        row = self.rows()['solo.mp4']
        self.assertEqual( row['median_simultaneous'], 1 )
        self.assertGreater( row['left_half_fraction'], 0.9 )

    def test_only_censored_labels_are_counted( self ):
        file_hash = self.clip( 'mixed.mp4', 60 )
        raw = [ _box( file_hash, i / 9.0, 700, 400 ) for i in range( 50 ) ]
        raw += [ _box( file_hash, i / 9.0, 100, 100, label='face_femme' ) for i in range( 50 ) ]
        self.boxes( file_hash, raw )
        row = self.rows()['mixed.mp4']
        self.assertEqual( row['detections'], 50 )
        self.assertEqual( row['median_simultaneous'], 1 )


class TestStructureShares( StructureHarness ):
    """The duration-weighted shares, driven through the real arithmetic."""

    def test_the_shot_shares_are_weighted_by_duration_not_by_count( self ):
        # Nineteen 0.1s cuts in the first two seconds of a 22s clip: 19 of
        # 21 SHOTS are short, about 9% of RUNTIME is. Counting shots would
        # report ~90%.
        file_hash = self.clip( 'flurry.mp4', 22 )
        self.cuts( file_hash, [ round( 0.1 * n, 2 ) for n in range( 1, 20 ) ] )
        row = self.rows()['flurry.mp4']
        self.assertLess( row['short_shot_fraction'], 0.25 )
        self.assertLess( row['mid_shot_fraction'], 0.5 )
        self.assertEqual( row['recommended_profile'], 'scene' )
        # The median DOES read as rapid-fire, which is why it does not decide.
        self.assertLess( row['median_shot_seconds'], 0.2 )

    def test_sustained_short_shots_do_reach_the_compilation_share( self ):
        file_hash = self.clip( 'comp.mp4', 24 )
        self.cuts( file_hash, [ round( 3.0 * n, 2 ) for n in range( 1, 8 ) ] )
        row = self.rows()['comp.mp4']
        self.assertGreaterEqual( row['mid_shot_fraction'], 0.5 )
        self.assertEqual( row['recommended_profile'], 'compilation' )


class TestMovesStayInTheirTree( StructureHarness ):

    def test_a_staged_file_is_sorted_inside_the_staging_tree( self ):
        file_hash = self.clip( 'fast.mp4', 60, tree=self.staging )
        self.cuts( file_hash, [ 10.0 + 0.5 * n for n in range( 40 ) ] )
        script = os.path.join( self.tmp, 'moves.sh' )
        self.args.write_move_script = script
        self.rows()
        with open( script ) as handle:
            text = handle.read()
        self.assertIn( os.path.join( self.staging, 'quick-cut' ), text )
        self.assertNotIn( os.path.join( self.vids, 'quick-cut' ), text )


class TestMoveVerdict( StructureHarness ):
    """Which recommended moves go in the script. See _move_verdict."""

    def row( self, folder, recommended, evidence='whole file', tree=None ):
        tree = tree or self.vids
        path = os.path.join( tree, folder, 'x.mp4' ) if folder != '(root)' else os.path.join( tree, 'x.mp4' )
        return { 'source_path': path, 'tree_root': tree, 'current_folder': folder,
                 'recommended_profile': recommended, 'timing_evidence': evidence }

    def verdict( self, *args, **kwargs ):
        return betabench._move_verdict( self.row( *args, **kwargs ) )[0]

    def test_a_root_file_with_whole_file_evidence_is_scripted( self ):
        self.assertEqual( self.verdict( '(root)', 'quick_cut' ), 'script' )

    def test_slice_evidence_is_review_only( self ):
        self.assertEqual( self.verdict( '(root)', 'quick_cut', evidence='slice' ), 'review' )

    def test_split_screen_geometry_does_not_need_whole_file_timing( self ):
        self.assertEqual( self.verdict( 'scene', 'split_screen', evidence='slice' ), 'script' )

    def test_a_move_that_lowers_the_sample_rate_is_never_scripted( self ):
        # quick-cut/ samples at 18 fps; scene at the global rate.
        self.assertEqual( self.verdict( 'quick-cut', 'scene' ), 'review' )

    def test_a_manual_folder_is_not_overridden_by_timing( self ):
        self.assertEqual( self.verdict( 'compilation', 'scene' ), 'review' )
        self.assertEqual( self.verdict( 'split-screen', 'scene' ), 'review' )

    def test_a_manual_folder_is_overridden_when_sampling_goes_up( self ):
        # The direction that only ever adds detections.
        self.assertEqual( self.verdict( 'compilation', 'quick_cut' ), 'script' )

    def test_an_unmapped_folder_is_review_only( self ):
        self.assertEqual( self.verdict( 'completed', 'quick_cut' ), 'review' )
        self.assertEqual( self.verdict( 'titty_drops', 'scene' ), 'review' )

    def test_an_auto_folder_moves_on_whole_file_evidence( self ):
        self.assertEqual( self.verdict( 'scene', 'compilation' ), 'script' )


class TestGeometrySliceOffsets( unittest.TestCase ):

    def test_three_slices_sit_at_quarters_of_the_room( self ):
        self.assertEqual( betabench.geometry_slice_offsets( 120.0, 20.0, 3 ), [ 25.0, 50.0, 75.0 ] )

    def test_no_slice_starts_at_zero_on_a_long_file( self ):
        self.assertGreater( min( betabench.geometry_slice_offsets( 3600.0, 20.0, 5 ) ), 0.0 )

    def test_a_file_shorter_than_a_slice_is_processed_whole( self ):
        self.assertEqual( betabench.geometry_slice_offsets( 15.0, 20.0, 3 ), [ 0.0 ] )

    def test_every_slice_fits_inside_the_file( self ):
        for offset in betabench.geometry_slice_offsets( 100.0, 20.0, 7 ):
            self.assertLessEqual( offset + 20.0, 100.0 )


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


class TestFolderRecommendation( unittest.TestCase ):
    """
    Which source folder a measured file belongs in.

    profile_by_source_dir makes the folder decide how a file is
    censored, which moves the burden onto sorting correctly. These are
    the boundaries that turn the structure measurements above into a
    folder, so that sorting is measured rather than eyeballed.
    """

    def row( self, **over ):
        base = { 'p90_simultaneous': 1, 'left_half_fraction': 0.2,
                 'short_shot_fraction': 0.0, 'mid_shot_fraction': 0.0 }
        base.update( over )
        return base

    def pick( self, quick=0.15, quick_name='quick_cut', **over ):
        return betabench._recommend_profile( self.row( **over ), quick, quick_name )[0]

    # --- split-screen, which is the whole reason this lives in betabench

    def test_the_split_screen_signature_is_recognised( self ):
        # Several boxes live at once, split evenly between screen halves.
        # No timing measure can see this, which is why the old standalone
        # classifier could never recommend split-screen at all.
        self.assertEqual(
            self.pick( p90_simultaneous=8, left_half_fraction=0.50 ),
            'split_screen' )

    def test_several_subjects_all_on_one_side_is_not_split_screen( self ):
        # A couple in frame together, action on one side. High
        # simultaneity, lopsided placement.
        self.assertNotEqual(
            self.pick( p90_simultaneous=8, left_half_fraction=0.13 ),
            'split_screen' )

    def test_one_subject_centred_is_not_split_screen( self ):
        # Even left-half share but only one box live: a single subject
        # crossing the middle, not two panes.
        self.assertNotEqual(
            self.pick( p90_simultaneous=1, left_half_fraction=0.50 ),
            'split_screen' )

    def test_split_screen_is_tested_before_the_timing_boundaries( self ):
        # A split-screen compilation is a split-screen problem first: the
        # failure is spatial, a loose match radius dragging a censor
        # across the divider, and that happens whatever the cut rate.
        self.assertEqual(
            self.pick( p90_simultaneous=8, left_half_fraction=0.50,
                       short_shot_fraction=0.9, mid_shot_fraction=1.0 ),
            'split_screen' )

    def test_an_unknown_left_half_share_cannot_claim_split_screen( self ):
        # Frame size unavailable, so the spatial test has no input. It
        # must decline rather than guess.
        self.assertNotEqual(
            self.pick( p90_simultaneous=9, left_half_fraction=None ),
            'split_screen' )

    # --- the timing boundaries, weighted by duration

    def test_rapid_cutting_is_quick_cut( self ):
        self.assertEqual(
            self.pick( short_shot_fraction=0.30, mid_shot_fraction=1.0 ),
            'quick_cut' )

    def test_a_compilation_is_mostly_short_shots_by_runtime( self ):
        self.assertEqual(
            self.pick( short_shot_fraction=0.01, mid_shot_fraction=0.9 ),
            'compilation' )

    def test_a_single_scene_is_mostly_long_shots( self ):
        self.assertEqual(
            self.pick( short_shot_fraction=0.01, mid_shot_fraction=0.05 ),
            'scene' )

    def test_a_flurry_is_a_scene_not_a_compilation( self ):
        # Nineteen 0.1s false cuts then one 28s take: 7% of runtime in
        # short shots, so a scene. A median shot length would read 0.1s
        # and call it the fastest-cutting file in the corpus.
        self.assertEqual(
            self.pick( short_shot_fraction=0.067, mid_shot_fraction=0.067 ),
            'scene' )

    def test_the_quick_cut_boundary_is_inclusive( self ):
        self.assertEqual( self.pick( quick=0.15, short_shot_fraction=0.15 ),
                          'quick_cut' )
        self.assertNotEqual( self.pick( quick=0.15, short_shot_fraction=0.1499 ),
                             'quick_cut' )

    def test_the_compilation_boundary_is_inclusive( self ):
        exact = betabench.COMPILATION_MIN_RUNTIME_FRACTION
        self.assertEqual( self.pick( mid_shot_fraction=exact ), 'compilation' )
        self.assertEqual( self.pick( mid_shot_fraction=exact - 0.001 ), 'scene' )

    def test_never_scanned_recommends_nothing( self ):
        # It used to recommend 'scene' with "no shot cuts found", which
        # read an unscanned file as a single long take and could move it
        # out of its folder.
        profile, why = betabench._recommend_profile(
            self.row( short_shot_fraction=None, mid_shot_fraction=None ),
            0.15, 'quick_cut' )
        self.assertIsNone( profile )
        self.assertIn( 'never scanned', why )

    def test_zero_measured_cuts_is_a_scene( self ):
        self.assertEqual( self.pick( short_shot_fraction=0.0, mid_shot_fraction=0.0 ), 'scene' )

    def test_split_screen_needs_enough_frames( self ):
        # One sampled frame with boxes on both sides is not a layout.
        self.assertNotEqual(
            self.pick( p90_simultaneous=8, left_half_fraction=0.50, frames_with_boxes=1 ),
            'split_screen' )
        self.assertEqual(
            self.pick( p90_simultaneous=8, left_half_fraction=0.50,
                       frames_with_boxes=betabench.SPLIT_MIN_FRAMES_WITH_BOXES ),
            'split_screen' )

    def test_no_configured_quick_boundary_means_no_quick_recommendation( self ):
        # Rather than inventing one, fall through to the boundaries this
        # bench does own.
        self.assertEqual(
            self.pick( quick=None, quick_name=None,
                       short_shot_fraction=0.9, mid_shot_fraction=1.0 ),
            'compilation' )

    def test_every_recommendation_carries_a_reason( self ):
        for over in ( { 'p90_simultaneous': 8, 'left_half_fraction': 0.5 },
                      { 'short_shot_fraction': 0.5, 'mid_shot_fraction': 1.0 },
                      { 'mid_shot_fraction': 0.8 },
                      { 'mid_shot_fraction': 0.0 },
                      { 'short_shot_fraction': None, 'mid_shot_fraction': None } ):
            _profile, why = betabench._recommend_profile(
                self.row( **over ), 0.15, 'quick_cut' )
            self.assertTrue( why and why.strip(), over )


class TestQuickCutThresholdComesFromConfig( unittest.TestCase ):
    """
    The bench must not restate a boundary the app owns.

    A bench with its own copy would recommend a folder the app then
    disagrees with, and files would be sorted on advice the app ignores.
    """

    def setUp( self ):
        self._saved = betabench.betaconfig.detector_backend
        betabench.betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'profiles': {
                'default': 'scene',
                'match_on': 'short_shot_fraction',
                'variants': {
                    'quick_cut': { 'min_short_shot_fraction': 0.42 },
                    'scene': {},
                },
            } },
        }
        betabench.bu_config.invalidate_config_caches()

    def tearDown( self ):
        betabench.betaconfig.detector_backend = self._saved
        betabench.bu_config.invalidate_config_caches()

    def profiles( self ):
        return betabench.betaconfig.detector_backend['nudenet_v3']['profiles']

    def test_the_threshold_tracks_config( self ):
        threshold, name = betabench._quick_cut_threshold()
        self.assertAlmostEqual( threshold, 0.42 )
        self.assertEqual( name, 'quick_cut' )

    def test_a_different_match_on_disables_it( self ):
        # Reading a short-shot threshold while the app selects on cuts
        # per minute would be advice the app never acts on.
        self.profiles()['match_on'] = 'cuts_per_min'
        betabench.bu_config.invalidate_config_caches()
        self.assertEqual( betabench._quick_cut_threshold(), ( None, None ) )

    def test_a_manual_only_variant_is_not_offered( self ):
        self.profiles()['variants']['quick_cut']['manual_only'] = True
        betabench.bu_config.invalidate_config_caches()
        self.assertEqual( betabench._quick_cut_threshold(), ( None, None ) )


class TestFolderForProfile( unittest.TestCase ):

    def setUp( self ):
        self._saved = getattr( betabench.betaconfig, 'profile_by_source_dir', None )

    def tearDown( self ):
        betabench.betaconfig.profile_by_source_dir = self._saved

    def test_the_mapping_is_inverted( self ):
        betabench.betaconfig.profile_by_source_dir = { 'quick-cut': 'quick_cut' }
        self.assertEqual( betabench._folder_for_profile( 'quick_cut' ), 'quick-cut' )

    def test_an_unmapped_profile_has_no_folder( self ):
        betabench.betaconfig.profile_by_source_dir = { 'scene': 'scene' }
        self.assertIsNone( betabench._folder_for_profile( 'quick_cut' ) )

    def test_the_shallowest_folder_wins( self ):
        betabench.betaconfig.profile_by_source_dir = {
            'compilation/split': 'split_screen', 'split-screen': 'split_screen' }
        self.assertEqual( betabench._folder_for_profile( 'split_screen' ),
                          'split-screen' )


class TestMoveScript( unittest.TestCase ):
    """
    The generated sorting script.

    It exists instead of moving files here because the recommendations
    rest on boundaries read off small samples: someone should read the
    reasons before seventy files move.
    """

    MOVES = [ { 'source': 'a.mp4', 'source_path': '/vids/a.mp4',
                'dest_dir': '/vids/compilation', 'from': '(root)',
                'to': 'compilation', 'why': 'a.mp4: 90% of runtime in sub-10s shots' },
              { 'source': "b's film.mkv", 'source_path': "/vids/b's film.mkv",
                'dest_dir': '/vids/scene', 'from': 'comps',
                'to': 'scene', 'why': "b's film.mkv: only 5% in sub-10s shots" } ]

    def write( self, moves=None ):
        import tempfile
        handle = tempfile.NamedTemporaryFile( suffix='.sh', delete=False )
        handle.close()
        betabench._write_move_script(
            handle.name, self.MOVES if moves is None else moves, _quiet_logger() )
        with open( handle.name ) as f:
            return handle.name, f.read()

    def test_the_script_is_valid_bash( self ):
        path, _text = self.write()
        import subprocess
        result = subprocess.run( [ 'bash', '-n', path ], capture_output=True, text=True )
        self.assertEqual( result.returncode, 0, result.stderr )

    def test_it_never_overwrites( self ):
        # mv -n is the whole safety story: a name collision must leave
        # both files alone rather than destroying one.
        import re
        _path, text = self.write()
        self.assertIn( 'mv -n', text )
        self.assertNotIn( 'mv -f', text )
        # A destructive command as a COMMAND, not the letters 'rm'
        # appearing inside a word like "confirm". Checked per line, with
        # comments stripped, so prose cannot pass or fail this.
        for line in text.split( '\n' ):
            code = line.split( '#' )[0].strip() if not line.lstrip().startswith( '#' ) else ''
            self.assertIsNone(
                re.search( r'(^|[;&|]\s*)(rm|rmdir|unlink|shred|truncate)\s', code ),
                "destructive command in: %r"%(line,) )

    def test_paths_with_quotes_and_spaces_survive( self ):
        # A filename with an apostrophe is the classic way a generated
        # script turns into an arbitrary command.
        _path, text = self.write()
        self.assertIn( """'/vids/b'\\''s film.mkv'""", text )

    def test_every_move_carries_its_reason_as_a_comment( self ):
        _path, text = self.write()
        for move in self.MOVES:
            self.assertIn( move['why'], text )

    def test_it_creates_the_destination( self ):
        _path, text = self.write()
        self.assertIn( 'mkdir -p', text )

    def test_it_checks_the_source_still_exists( self ):
        # So a re-run after a partial move reports rather than errors.
        _path, text = self.write()
        self.assertIn( 'if [[ ! -f "$src" ]]', text )

    def test_it_is_executable( self ):
        path, _text = self.write()
        self.assertTrue( os.access( path, os.X_OK ) )

    def test_an_empty_move_list_still_writes_a_valid_script( self ):
        path, text = self.write( moves=[] )
        import subprocess
        self.assertEqual(
            subprocess.run( [ 'bash', '-n', path ] ).returncode, 0 )
        self.assertIn( 'moved $moved', text )

    def test_the_script_actually_moves_a_file( self ):
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as root:
            src = os.path.join( root, 'clip.mp4' )
            _write_text( src, 'x' )
            dest = os.path.join( root, 'compilation' )
            script = os.path.join( root, 'moves.sh' )
            betabench._write_move_script( script, [ {
                'source': 'clip.mp4', 'source_path': src, 'dest_dir': dest,
                'from': '(root)', 'to': 'compilation', 'why': 'test' } ],
                _quiet_logger() )
            subprocess.run( [ 'bash', script ], capture_output=True, text=True )
            self.assertFalse( os.path.exists( src ), "source should be gone" )
            self.assertTrue( os.path.exists( os.path.join( dest, 'clip.mp4' ) ) )

    def test_re_running_the_script_is_harmless( self ):
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as root:
            src = os.path.join( root, 'clip.mp4' )
            _write_text( src, 'original' )
            dest = os.path.join( root, 'scene' )
            script = os.path.join( root, 'moves.sh' )
            betabench._write_move_script( script, [ {
                'source': 'clip.mp4', 'source_path': src, 'dest_dir': dest,
                'from': '(root)', 'to': 'scene', 'why': 'test' } ],
                _quiet_logger() )
            subprocess.run( [ 'bash', script ], capture_output=True, text=True )
            # Put a DIFFERENT file back at the old name, then re-run: the
            # already-moved copy must not be clobbered.
            _write_text( src, 'different' )
            result = subprocess.run( [ 'bash', script ], capture_output=True, text=True )
            with open( os.path.join( dest, 'clip.mp4' ) ) as f:
                self.assertEqual( f.read(), 'original',
                                  "mv -n must not overwrite the moved file" )
            self.assertIn( 'skip', result.stdout )
