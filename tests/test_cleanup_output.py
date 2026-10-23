"""
test_cleanup_output.py - the output folder cleanup removes renders and nothing a later decision needs.

Each category is pinned on a built output tree: what it offers, what
--apply removes, and that stats, caches, analysis runs, benchmark
results and every round's record survive every category. Plus the
safety rails: no symlink is followed, nothing outside the output folder
is touched, a round still ranking is not "aborted", and a log a run is
writing is not rotated.
"""

import gzip
import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
import unittest

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load():
    path = os.path.join( REPO_ROOT, 'tools', 'maintenance', 'cleanup_output.py' )
    spec = importlib.util.spec_from_file_location( '_cleanup_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


TOOL = _load()
DAY = 86400


class _Tree( unittest.TestCase ):
    """An output folder with one of everything, dated relative to NOW."""

    NOW = 2000000000.0

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-cleanup-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.outside = tempfile.mkdtemp( prefix='betasuite-outside-' )
        self.addCleanup( shutil.rmtree, self.outside, ignore_errors=True )
        self.kept = []
        self.write( 'stats/betasuite_stats.jsonl', keep=True )
        self.write( 'cache/vid_hashes/abc-nudenet_v3-3-640-9-0.120-df43d00.gz', keep=True )
        self.write( 'cache/shot_cuts/abc-9-0.400.gz', keep=True )
        self.write( 'logs/analysis_runs/20261018_170253/auto_tune/decisions.json', keep=True )
        self.write( 'benchmarks/20261006_165708/results.json', keep=True )
        self.write( 'benchmarks/20261006_165708/betabench.log', keep=True )
        # Decided round: renders, a half-rendered whole video, its record.
        self.round( 'ab_suppression_20261019_100748', age=DAY, decided=True,
                    renders=[ 'a_clip01_ab.mp4', 'a_clip01_diff.mp4', 'a_full_current.tmp.mp4',
                              'a_full_current.chunk000.mkv', 'a_full_current.mp4.concat.txt' ] )
        # Undecided, stale and fresh.
        self.round( 'ab_suppression_20261019_092944', age=20 * DAY, renders=[ 'b_clip01_ab.mp4' ] )
        self.round( 'ab_min_track_hits_20261022_090000', age=DAY, renders=[ 'c_clip01_ab.mp4' ] )
        # Older name, decided by chosen_rules.json only.
        self.round( 'suppression_ab_20261010_145028', age=10 * DAY, decided='chosen_rules.json',
                    renders=[ 'd_full_proposed.mp4' ] )
        # Aborted starts: an old one, one still ranking, a start folder.
        self.write( 'benchmarks/ab_suppression_20261018_181042/ab_compare.log', age=5 * DAY )
        self.write( 'benchmarks/ab_suppression_20261023_093620_confirm/betabench.log', age=600, keep=True )
        self.write( 'benchmarks/.ab_start_20261020_101010/ab_compare.log', age=2 * DAY )
        # A round folder symlinked from elsewhere, and a symlinked render.
        self.write_outside( 'elsewhere/x_clip01_ab.mp4' )
        self.write_outside( 'elsewhere/ranking.json' )
        self.write_outside( 'elsewhere/chosen.json' )
        os.symlink( os.path.join( self.outside, 'elsewhere' ),
                    os.path.join( self.root, 'benchmarks', 'ab_linked_20261001_000000' ) )
        os.symlink( os.path.join( self.outside, 'elsewhere', 'x_clip01_ab.mp4' ),
                    os.path.join( self.root, 'benchmarks', 'ab_suppression_20261019_100748', 'linked.mp4' ) )

    def write( self, relative, size=10, age=0, keep=False ):
        path = os.path.join( self.root, relative )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'wb' ) as handle:
            handle.write( b'x' * size )
        os.utime( path, ( self.NOW - age, self.NOW - age ) )
        if keep:
            self.kept.append( path )
        return path

    def write_outside( self, relative ):
        path = os.path.join( self.outside, relative )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'wb' ) as handle:
            handle.write( b'y' * 100 )
        return path

    def round( self, name, age, decided=False, renders=() ):
        base = 'benchmarks/%s/'%( name, )
        for record in ( 'ranking.json', 'clips.md', 'feedback.csv', 'ab_compare.log', 'betabench.log' ):
            self.write( base + record, age=age, keep=True )
        if decided:
            self.write( base + ( decided if isinstance( decided, str ) else 'chosen.json' ), age=age, keep=True )
        for render in renders:
            self.write( base + render, size=1000, age=age )

    def plan( self, **kwargs ):
        return TOOL.plan( self.root, now=self.NOW, **kwargs )

    def by_category( self, **kwargs ):
        found = {}
        for item in self.plan( **kwargs ):
            found.setdefault( item.category, [] ).append( item.name )
        return found

    def cli( self, *argv ):
        lines = []
        TOOL.main( [ '--root', self.root ] + list( argv ), now=self.NOW, out=lines.append )
        return lines


class TestPlan( _Tree ):

    def test_each_round_lands_in_its_category( self ):
        self.assertEqual( self.by_category(), {
            'decided_renders': [ 'ab_suppression_20261019_100748', 'suppression_ab_20261010_145028' ],
            'stale_renders': [ 'ab_suppression_20261019_092944' ],
            'aborted_rounds': [ '.ab_start_20261020_101010', 'ab_suppression_20261018_181042' ] } )

    def test_renders_include_half_finished_leftovers_and_nothing_else( self ):
        decided = [ item for item in self.plan() if item.name == 'ab_suppression_20261019_100748' ][0]
        self.assertEqual( sorted( os.path.basename( path ) for path in decided.files ), [
            'a_clip01_ab.mp4', 'a_clip01_diff.mp4', 'a_full_current.chunk000.mkv',
            'a_full_current.mp4.concat.txt', 'a_full_current.tmp.mp4' ] )
        self.assertEqual( decided.size, 5000 )

    def test_stale_days_moves_the_line( self ):
        self.assertNotIn( 'stale_renders', self.by_category( stale_days=21 ) )
        self.assertEqual( self.by_category( stale_days=1 )['stale_renders'],
                          [ 'ab_min_track_hits_20261022_090000', 'ab_suppression_20261019_092944' ],
                          "a day old at stale_days=1 is stale" )

    def test_a_round_still_ranking_is_not_aborted( self ):
        names = [ item.name for item in self.plan() ]
        self.assertNotIn( 'ab_suppression_20261023_093620_confirm', names )

    def test_an_aborted_round_with_a_render_is_left_alone( self ):
        self.write( 'benchmarks/ab_cut_bursts_20261001_000000/odd.mp4', age=5 * DAY )
        self.assertNotIn( 'ab_cut_bursts_20261001_000000', [ item.name for item in self.plan() ] )

    def test_a_symlinked_round_is_never_planned( self ):
        self.assertNotIn( 'ab_linked_20261001_000000', [ item.name for item in self.plan() ] )


class TestApply( _Tree ):

    def survivors( self ):
        return [ path for path in self.kept if os.path.exists( path ) ]

    def test_every_category_keeps_the_record( self ):
        for category in TOOL.CATEGORIES:
            self.cli( '--apply', category )
        self.assertEqual( self.survivors(), self.kept )
        self.assertTrue( os.path.exists( os.path.join( self.outside, 'elsewhere', 'x_clip01_ab.mp4' ) ),
                         "a symlink's target is never removed" )

    def test_decided_renders_go_and_their_records_stay( self ):
        lines = self.cli( '--apply', 'decided_renders' )
        folder = os.path.join( self.root, 'benchmarks', 'ab_suppression_20261019_100748' )
        self.assertEqual( sorted( os.listdir( folder ) ), [
            'ab_compare.log', 'betabench.log', 'chosen.json', 'clips.md', 'feedback.csv', 'linked.mp4',
            'ranking.json' ] )
        self.assertTrue( os.path.exists( os.path.join(
            self.root, 'benchmarks', 'ab_suppression_20261019_092944', 'b_clip01_ab.mp4' ) ),
            "an undecided round is another category" )
        self.assertIn( 'INFO freed 5.9 KB (decided_renders)', lines )

    def test_only_removes_one_round( self ):
        self.cli( '--apply', 'decided_renders', '--only', 'suppression_ab_20261010_145028' )
        self.assertFalse( os.path.exists( os.path.join(
            self.root, 'benchmarks', 'suppression_ab_20261010_145028', 'd_full_proposed.mp4' ) ) )
        self.assertTrue( os.path.exists( os.path.join(
            self.root, 'benchmarks', 'ab_suppression_20261019_100748', 'a_clip01_ab.mp4' ) ) )

    def test_aborted_rounds_go_whole( self ):
        self.cli( '--apply', 'aborted_rounds' )
        benchmarks = os.path.join( self.root, 'benchmarks' )
        self.assertFalse( os.path.exists( os.path.join( benchmarks, 'ab_suppression_20261018_181042' ) ) )
        self.assertFalse( os.path.exists( os.path.join( benchmarks, '.ab_start_20261020_101010' ) ) )
        self.assertTrue( os.path.exists( os.path.join( benchmarks, 'ab_suppression_20261023_093620_confirm' ) ) )

    def test_nothing_outside_the_root_is_removed_even_if_planned( self ):
        stray = TOOL.Item( 'decided_renders', 'stray', self.outside,
                           [ os.path.join( self.outside, 'elsewhere', 'x_clip01_ab.mp4' ) ], 100, 'test' )
        lines = []
        self.assertEqual( TOOL.apply( [ stray ], self.root, 'decided_renders', out=lines.append ), 0 )
        self.assertTrue( lines[0].startswith( 'WARN refusing' ) )

    def test_a_sibling_folder_sharing_the_roots_name_is_outside_it( self ):
        sibling = self.root + '-sibling'
        os.makedirs( sibling )
        self.addCleanup( shutil.rmtree, sibling, ignore_errors=True )
        victim = os.path.join( sibling, 'v.mp4' )
        with open( victim, 'wb' ) as handle:
            handle.write( b'z' )
        stray = TOOL.Item( 'decided_renders', 'stray', sibling, [ victim ], 1, 'test' )
        TOOL.apply( [ stray ], self.root, 'decided_renders', out=lambda line: None )
        self.assertTrue( os.path.exists( victim ) )

    def test_a_file_outside_its_round_is_refused( self ):
        folder = os.path.join( self.root, 'benchmarks', 'ab_suppression_20261019_100748' )
        stray = TOOL.Item( 'decided_renders', 'r', folder, [ self.kept[0] ], 10, 'test' )
        lines = []
        TOOL.apply( [ stray ], self.root, 'decided_renders', out=lines.append )
        self.assertTrue( os.path.exists( self.kept[0] ) )
        self.assertTrue( any( line.startswith( 'WARN refusing' ) for line in lines ) )

    def test_without_apply_nothing_is_removed( self ):
        lines = self.cli()
        self.assertTrue( all( os.path.exists( item.files[0] ) for item in self.plan() if item.files ) )
        self.assertTrue( any( 'total that could be freed' in line for line in lines ) )

    def test_summary_and_list_feed_the_wrapper( self ):
        self.assertEqual( self.cli( '--summary', 'decided_renders' ), [ '2 6000' ] )
        self.assertEqual( self.cli( '--summary', 'decided_renders', '--only',
                                    'ab_suppression_20261019_100748' ), [ '1 5000' ] )
        self.assertEqual( self.cli( '--summary', 'stale_renders', '--only', 'nope' ), [ '0 0' ] )
        listed = self.cli( '--list', 'stale_renders' )
        self.assertEqual( len( listed ), 1 )
        self.assertTrue( listed[0].startswith( 'ab_suppression_20261019_092944\t1000 B\tno decision, '
                                               'untouched 20 day(s)' ), listed )


class TestLogRotation( unittest.TestCase ):

    NOW = 2000000000.0

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-cleanup-' )
        self.addCleanup( shutil.rmtree, self.root )
        self.log = os.path.join( self.root, 'logs', 'betasuite.log' )
        os.makedirs( os.path.dirname( self.log ) )

    def write_log( self, size, age ):
        with open( self.log, 'wb' ) as handle:
            handle.write( b'line\n' * ( size // 5 ) )
        os.utime( self.log, ( self.NOW - age, self.NOW - age ) )

    def item( self, max_mb=0.001 ):
        return TOOL.log_item( self.log, self.NOW, max_mb )

    def test_a_small_log_is_left( self ):
        self.write_log( 500, age=3600 )
        self.assertIsNone( self.item( max_mb=0.001 ) )

    def test_a_log_at_the_limit_rotates( self ):
        self.write_log( 1050, age=3600 )
        item = self.item( max_mb=1050 / ( 1024.0 * 1024 ) )
        self.assertEqual( item.files, [ self.log ] )

    def test_a_log_written_in_the_last_ten_minutes_is_not_rotated( self ):
        self.write_log( 5000, age=599 )
        item = self.item()
        self.assertEqual( item.files, [], "offered as busy, nothing to remove" )
        self.assertIn( 'last 10 minutes', item.why )
        self.write_log( 5000, age=600 )
        self.assertEqual( self.item().files, [ self.log ] )

    def test_rotation_compresses_shifts_and_keeps_n( self ):
        for number in ( 1, 2, 3 ):
            with gzip.open( '%s.%d.gz'%( self.log, number ), 'wb' ) as handle:
                handle.write( b'old %d'%( number, ) )
        self.write_log( 5000, age=3600 )
        TOOL.apply( [ self.item() ], self.root, 'log_rotation', keep_logs=3, out=lambda line: None )
        self.assertEqual( os.path.getsize( self.log ), 0, "emptied in place" )
        with gzip.open( self.log + '.1.gz' ) as handle:
            self.assertEqual( handle.read(), b'line\n' * 1000 )
        with gzip.open( self.log + '.2.gz' ) as handle:
            self.assertEqual( handle.read(), b'old 1' )
        with gzip.open( self.log + '.3.gz' ) as handle:
            self.assertEqual( handle.read(), b'old 2' )
        self.assertFalse( os.path.exists( self.log + '.4.gz' ), "the oldest past keep_logs is gone" )

    def test_a_writer_holding_the_log_open_keeps_writing_to_it( self ):
        self.write_log( 5000, age=3600 )
        writer = open( self.log, 'ab' )
        self.addCleanup( writer.close )
        TOOL.apply( [ self.item() ], self.root, 'log_rotation', out=lambda line: None )
        writer.write( b'after\n' )
        writer.flush()
        with open( self.log, 'rb' ) as handle:
            self.assertEqual( handle.read(), b'after\n' )


class TestWrapper( unittest.TestCase ):
    """Unattended, the wrapper shows the plan and removes nothing."""

    def test_unattended_removes_nothing( self ):
        tree = _Tree( 'run' )
        tree.setUp()
        self.addCleanup( tree.doCleanups )
        top = tempfile.mkdtemp( prefix='betasuite-checkout-' )
        self.addCleanup( shutil.rmtree, top, ignore_errors=True )
        # A checkout beside its output folder, as on disk: repo/ and output/.
        shutil.copytree( tree.root, os.path.join( top, 'output' ), symlinks=True )
        repo = os.path.join( top, 'repo' )
        for part in ( 'maintenance', 'tuning', 'lib' ):
            shutil.copytree( os.path.join( REPO_ROOT, 'tools', part ), os.path.join( repo, 'tools', part ),
                             ignore=shutil.ignore_patterns( '__pycache__', '*.log' ) )
        for name in os.listdir( REPO_ROOT ):
            if name.endswith( '.py' ):
                shutil.copy( os.path.join( REPO_ROOT, name ), repo )
        render = os.path.join( top, 'output', 'benchmarks', 'ab_suppression_20261019_100748',
                               'a_clip01_ab.mp4' )
        environment = dict( os.environ, PYTHON_BIN=shutil.which( 'python3' ), PYTHON_BIN_SOURCE='test' )
        environment.pop( 'HARNESS_LOG_FILE', None )
        result = subprocess.run( [ 'bash', 'tools/maintenance/cleanup_output.sh' ], cwd=repo,
                                 env=environment, stdin=subprocess.DEVNULL, capture_output=True,
                                 text=True, timeout=120 )
        self.assertEqual( result.returncode, 0, result.stdout + result.stderr )
        self.assertIn( 'nothing is removed', result.stdout )
        self.assertIn( 'decided_renders', result.stdout )
        self.assertTrue( os.path.exists( render ) )
        self.assertTrue( os.path.exists( os.path.join( top, 'output', 'logs', 'cleanup_output.log' ) ) )

if __name__ == '__main__':
    unittest.main()
