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

    def test_the_shot_shares_are_weighted_by_duration_not_by_count( self ):
        # The arithmetic, driven through the real function rather than
        # handed in. A mutation changing these sums to counts passed
        # every recommendation test, because those pass the fractions in
        # as row values and never reach this code.
        #
        # 'flurry' is nineteen 0.1s cuts in the first two seconds of a
        # ~22s span, then nothing: 19 of its 21 SHOTS are short, but only
        # about 9% of its RUNTIME is. Counting shots would report ~90%.
        tmp = tempfile.mkdtemp()
        saved_dir = betaconst.shot_cut_dir
        bu_cache.betaconst.shot_cut_dir = tmp
        try:
            fps = betaconfig.video_censor_fps
            threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
            bu_hash.write_json( [ round( 0.1 * n, 2 ) for n in range( 1, 20 ) ],
                                bu_cache.shot_cut_path_for( 'flurry', fps, threshold ) )
            raw = [ _box( 'flurry', i / 9.0, 700, 400 ) for i in range( 200 ) ]
            row = self._rows( raw )['flurry']

            self.assertLess( row['short_shot_fraction'], 0.25,
                "short-shot share must be a share of RUNTIME; counting shots "
                "would put this near 0.9" )
            self.assertLess( row['mid_shot_fraction'], 0.5,
                "so this file must not be recommended as a compilation" )
            self.assertEqual( row['recommended_profile'], 'scene' )
            # The median DOES read as rapid-fire here, which is exactly
            # why it is not what decides.
            self.assertLess( row['median_shot_seconds'], 0.2 )
        finally:
            bu_cache.betaconst.shot_cut_dir = saved_dir

    def test_sustained_short_shots_do_reach_the_compilation_share( self ):
        # The other side of the same coin: a file genuinely made of short
        # shots must report a high duration share, or the measure would
        # just always say "scene".
        tmp = tempfile.mkdtemp()
        saved_dir = betaconst.shot_cut_dir
        bu_cache.betaconst.shot_cut_dir = tmp
        try:
            fps = betaconfig.video_censor_fps
            threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
            bu_hash.write_json( [ round( 3.0 * n, 2 ) for n in range( 1, 8 ) ],
                                bu_cache.shot_cut_path_for( 'comp', fps, threshold ) )
            raw = [ _box( 'comp', i / 9.0, 700, 400 ) for i in range( 200 ) ]
            row = self._rows( raw )['comp']
            self.assertGreaterEqual( row['mid_shot_fraction'], 0.5 )
            self.assertEqual( row['recommended_profile'], 'compilation' )
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

    def test_no_cuts_found_is_a_scene_and_says_so( self ):
        profile, why = betabench._recommend_profile(
            self.row( short_shot_fraction=None, mid_shot_fraction=None ),
            0.15, 'quick_cut' )
        self.assertEqual( profile, 'scene' )
        self.assertIn( 'no shot cuts', why )

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
            open( src, 'w' ).write( 'x' )
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
            open( src, 'w' ).write( 'original' )
            dest = os.path.join( root, 'scene' )
            script = os.path.join( root, 'moves.sh' )
            betabench._write_move_script( script, [ {
                'source': 'clip.mp4', 'source_path': src, 'dest_dir': dest,
                'from': '(root)', 'to': 'scene', 'why': 'test' } ],
                _quiet_logger() )
            subprocess.run( [ 'bash', script ], capture_output=True, text=True )
            # Put a DIFFERENT file back at the old name, then re-run: the
            # already-moved copy must not be clobbered.
            open( src, 'w' ).write( 'different' )
            result = subprocess.run( [ 'bash', script ], capture_output=True, text=True )
            with open( os.path.join( dest, 'clip.mp4' ) ) as f:
                self.assertEqual( f.read(), 'original',
                                  "mv -n must not overwrite the moved file" )
            self.assertIn( 'skip', result.stdout )
