"""
test_bench_video_subset.py - measuring a predictable subset of the library.

--first-videos and --videos-per-folder narrow structure, structure-scan,
concurrency, encode and reverse to a few files, and the two harnesses
expose them as VIDEOS_FIRST / VIDEOS_PER_FOLDER. The point of the subset
is that it is the SAME files every time, so two runs (or a run and its
re-run after a fix) can be compared. These tests pin that ordering, the
grouping by directory, and that every subcommand reads the subset rather
than the whole library.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import betaconst

from tests.test_betatv_end_to_end import _ffmpeg_available
from tests.test_structure_bench import betabench, _quiet_logger

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _source( tree, folder, name ):
    return { 'path': '/%s/%s/%s'%( tree, folder, name ), 'name': name, 'tree': tree,
             'tree_root': '/%s'%( tree, ), 'current_folder': folder }


LIBRARY = [
    _source( 'uncensored_vids', '(root)', 'zeta.mp4' ),
    _source( 'uncensored_vids', 'compilation', 'Beta.mp4' ),
    _source( 'uncensored_vids', 'compilation', 'alpha.mp4' ),
    _source( 'uncensored_vids', 'scene', 'delta.mp4' ),
    _source( 'uncensored_vids', 'scene', 'Charlie.mp4' ),
    _source( 'source', 'scene', 'echo.mp4' ),
    _source( 'source', 'scene', 'alpha.mp4' ),
]


def _names( sources ):
    return [ '%s/%s/%s'%( s['tree'], s['current_folder'], s['name'] ) for s in sources ]


class TestSelectSourceVideos( unittest.TestCase ):

    def test_no_subset_is_every_file_in_order_as_a_new_list( self ):
        chosen = betabench.select_source_videos( LIBRARY )
        self.assertEqual( chosen, LIBRARY )
        self.assertIsNot( chosen, LIBRARY, "a caller sorting the result must not reorder the library" )

    def test_first_is_by_name_ignoring_case_across_every_folder_and_tree( self ):
        chosen = betabench.select_source_videos( LIBRARY, first=3 )
        # Two files are named alpha.mp4; the full path breaks the tie, so
        # the order never depends on how the directory walk happened to go.
        self.assertEqual( _names( chosen ), [ 'source/scene/alpha.mp4',
                                              'uncensored_vids/compilation/alpha.mp4',
                                              'uncensored_vids/compilation/Beta.mp4' ] )

    def test_first_does_not_depend_on_input_order( self ):
        self.assertEqual( betabench.select_source_videos( list( reversed( LIBRARY ) ), first=4 ),
                          betabench.select_source_videos( LIBRARY, first=4 ) )

    def test_first_beyond_the_library_is_the_whole_library( self ):
        self.assertEqual( len( betabench.select_source_videos( LIBRARY, first=50 ) ), len( LIBRARY ) )

    def test_per_folder_takes_the_first_by_name_from_each_directory( self ):
        chosen = betabench.select_source_videos( LIBRARY, per_folder=1 )
        # The same folder name in two trees is two directories.
        self.assertEqual( _names( chosen ), [ 'uncensored_vids/(root)/zeta.mp4',
                                              'uncensored_vids/compilation/alpha.mp4',
                                              'uncensored_vids/scene/Charlie.mp4',
                                              'source/scene/alpha.mp4' ] )

    def test_per_folder_n_caps_each_directory_not_the_total( self ):
        chosen = betabench.select_source_videos( LIBRARY, per_folder=2 )
        self.assertEqual( len( chosen ), 7 )
        self.assertEqual( _names( chosen )[1:3], [ 'uncensored_vids/compilation/alpha.mp4',
                                                   'uncensored_vids/compilation/Beta.mp4' ] )

    def test_both_at_once_is_refused( self ):
        with self.assertRaises( ValueError ):
            betabench.select_source_videos( LIBRARY, first=1, per_folder=1 )


class TestSubsetFlags( unittest.TestCase ):

    def test_the_two_flags_are_mutually_exclusive( self ):
        with mock.patch( 'sys.stderr' ), self.assertRaises( SystemExit ):
            betabench.build_parser().parse_args(
                [ 'structure', '--first-videos', '2', '--videos-per-folder', '1' ] )

    def test_zero_and_negative_are_rejected( self ):
        for value in ( '0', '-1' ):
            with mock.patch( 'sys.stderr' ), self.assertRaises( SystemExit ):
                betabench.build_parser().parse_args( [ 'structure', '--first-videos', value ] )

    def test_parsed_values( self ):
        args = betabench.build_parser().parse_args( [ 'reverse', '--videos-per-folder', '2' ] )
        self.assertEqual( ( args.first_videos, args.videos_per_folder ), ( None, 2 ) )


class TestSelectedSourceVideosOnDisk( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-subset-' )
        self.saved = ( betaconst.video_path_uncensored, betaconst.video_path_source_backup )
        betaconst.video_path_uncensored = os.path.join( self.root, 'vids' ) + os.sep
        betaconst.video_path_source_backup = os.path.join( self.root, 'staging' ) + os.sep
        for relative in ( 'vids/scene/b.mp4', 'vids/scene/a.mp4', 'vids/compilation/c.mp4',
                          'staging/scene/d.mp4', 'vids/notes.txt' ):
            path = os.path.join( self.root, relative )
            os.makedirs( os.path.dirname( path ), exist_ok=True )
            open( path, 'wb' ).close()
        betabench._LAST_LOGGED_SUBSET[0] = None

    def tearDown( self ):
        betaconst.video_path_uncensored, betaconst.video_path_source_backup = self.saved
        shutil.rmtree( self.root, ignore_errors=True )

    def test_the_subset_is_drawn_from_both_trees_and_listed_in_the_log( self ):
        lines = []

        class Capture( logging.Handler ):
            def emit( self, record ):
                lines.append( record.getMessage() )

        logger = logging.getLogger( 'subset_test' )
        logger.handlers = [ Capture() ]
        logger.setLevel( logging.INFO )
        logger.propagate = False
        args = types.SimpleNamespace( first_videos=None, videos_per_folder=1 )
        chosen = betabench.selected_source_videos( args, logger )
        self.assertEqual( [ s['name'] for s in chosen ], [ 'c.mp4', 'a.mp4', 'd.mp4' ] )
        self.assertIn( "video subset: the first 1 by name in each folder, 3 of 4 file(s):", lines )
        self.assertIn( "  uncensored_vids/scene/a.mp4", lines )
        self.assertIn( "  source/scene/d.mp4", lines )

        # The same subset again, from a second subcommand in the same run,
        # is not listed twice.
        betabench.selected_source_videos( args, logger )
        self.assertEqual( sum( 1 for line in lines if line.startswith( 'video subset:' ) ), 1 )

    def test_args_without_the_flags_mean_every_file( self ):
        # Older callers (and the other subcommands' tests) pass a bare
        # namespace; that must keep meaning "the whole library".
        chosen = betabench.selected_source_videos( types.SimpleNamespace(), None )
        self.assertEqual( len( chosen ), 4 )


class TestSubcommandsReadTheSubset( unittest.TestCase ):
    """Every subcommand that enumerates sources goes through the subset."""

    def setUp( self ):
        self.hashed = []

        def refuse( path, _length ):
            self.hashed.append( path )
            raise OSError( "test: not hashing" )

        self.patches = [
            mock.patch.object( betabench, '_source_videos', return_value=list( LIBRARY ) ),
            mock.patch.object( betabench.bu_hash, 'md5_for_file', side_effect=refuse ),
            mock.patch.object( betabench.bu_hash, 'flush_file_hash_memo' ),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown( self ):
        for patcher in self.patches:
            patcher.stop()

    def _args( self, **extra ):
        values = dict( first_videos=2, videos_per_folder=None, backend=None, variants=None,
                       workers=1, geometry_slices=0, write_move_script=None,
                       trigger_min_prob=0.3, trigger_hold=0.5, yes=True )
        values.update( extra )
        return types.SimpleNamespace( **values )

    def test_structure_hashes_only_the_subset( self ):
        betabench.bench_structure( self._args(), _quiet_logger() )
        self.assertEqual( sorted( self.hashed ), [ '/source/scene/alpha.mp4',
                                                   '/uncensored_vids/compilation/alpha.mp4' ] )

    def test_structure_scan_hashes_only_the_subset( self ):
        betabench.bench_structure_scan( self._args(), _quiet_logger() )
        self.assertEqual( sorted( self.hashed ), [ '/source/scene/alpha.mp4',
                                                   '/uncensored_vids/compilation/alpha.mp4' ] )

    def test_concurrency_measures_only_the_subset( self ):
        seen = {}

        def overlap( _args, _logger, path, _caches ):
            seen['overlap'] = path

        def scan_workers( _args, _logger, paths, _caches ):
            seen['scan'] = list( paths )

        with mock.patch.object( betabench, '_measure_overlap', side_effect=overlap ), \
             mock.patch.object( betabench, '_measure_scan_workers', side_effect=scan_workers ), \
             mock.patch.object( betabench, '_scan_bench_file_count', return_value=10 ):
            betabench.bench_concurrency( self._args( video=None, first_videos=2 ), _quiet_logger() )
        self.assertEqual( seen['overlap'], '/source/scene/alpha.mp4' )
        self.assertEqual( seen['scan'], [ '/source/scene/alpha.mp4',
                                          '/uncensored_vids/compilation/alpha.mp4' ] )

    def test_the_reverse_share_hashes_only_the_subset( self ):
        betabench._reverse_trigger_share( self._args(), _quiet_logger() )
        self.assertEqual( sorted( self.hashed ), [ '/source/scene/alpha.mp4',
                                                   '/uncensored_vids/compilation/alpha.mp4' ] )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the clips" )
class TestPickedVideoComesFromTheSubset( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-subset-pick-' )
        self.saved = ( betaconst.video_path_uncensored, betaconst.video_path_source_backup )
        betaconst.video_path_uncensored = os.path.join( self.root, 'vids' ) + os.sep
        betaconst.video_path_source_backup = betaconst.video_path_uncensored
        os.makedirs( betaconst.video_path_uncensored )
        for name, size in ( ( 'a_small.mp4', '64x48' ), ( 'b_large.mp4', '160x120' ) ):
            subprocess.run(
                [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                  '-i', 'testsrc2=size=%s:rate=10'%( size, ), '-t', '1', '-c:v', 'libx264',
                  '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
                  os.path.join( betaconst.video_path_uncensored, name ) ], check=True )

    def tearDown( self ):
        betaconst.video_path_uncensored, betaconst.video_path_source_backup = self.saved
        shutil.rmtree( self.root, ignore_errors=True )

    def test_the_largest_file_in_the_subset_not_in_the_library( self ):
        args = types.SimpleNamespace( video=None, first_videos=1, videos_per_folder=None )
        picked = betabench._pick_encode_video( args, _quiet_logger() )
        self.assertEqual( os.path.basename( picked ), 'a_small.mp4' )
        args.first_videos = None
        picked = betabench._pick_encode_video( args, _quiet_logger() )
        self.assertEqual( os.path.basename( picked ), 'b_large.mp4' )


class TestHarnessesPassTheSubsetToEveryBenchCall( unittest.TestCase ):
    """
    Runs the real launchers against a stand-in Python that records its
    arguments, so a phase that forgets the subset (and quietly measures
    the whole library) fails here. The scripts are copied into a scratch
    repo: they write to <repo>/../output, which for the real checkout is
    the user's own output folder.
    """

    SCRIPTS = ( 'tools/bench/structure_survey.sh', 'tools/bench/reverse_cost.sh',
                'tools/bench/video_subset.sh', 'tools/tuning/harness_lib.sh',
                'tools/lib/python_env.sh' )

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-harness-' )
        self.repo = os.path.join( self.root, 'repo' )
        for relative in self.SCRIPTS:
            target = os.path.join( self.repo, relative )
            os.makedirs( os.path.dirname( target ), exist_ok=True )
            shutil.copy( os.path.join( REPO_ROOT, relative ), target )
        self.calls = os.path.join( self.root, 'calls.txt' )
        self.fake_python = os.path.join( self.root, 'fake_python' )
        with open( self.fake_python, 'w' ) as handle:
            handle.write( '#!/bin/bash\n'
                          'if [ "${1:-}" = "-c" ]; then exit 0; fi\n'
                          'echo "$*" >> "%s"\n'%( self.calls, ) )
        os.chmod( self.fake_python, 0o755 )

    def tearDown( self ):
        shutil.rmtree( self.root, ignore_errors=True )

    def run_script( self, relative, **env ):
        environment = dict( os.environ, PYTHON_BIN=self.fake_python,
                            PYTHON_BIN_SOURCE='test stand-in', LOG_LEVEL='error' )
        environment.pop( 'VIDEOS_FIRST', None )
        environment.pop( 'VIDEOS_PER_FOLDER', None )
        environment.update( env )
        result = subprocess.run( [ 'bash', os.path.join( self.repo, relative ) ], cwd=self.repo,
                                 env=environment, stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, timeout=120 )
        calls = []
        if os.path.exists( self.calls ):
            with open( self.calls ) as handle:
                calls = [ line.split() for line in handle if 'betabench.py' in line ]
        return result, calls

    def test_the_survey_passes_the_subset_to_all_three_phases( self ):
        result, calls = self.run_script( 'tools/bench/structure_survey.sh', VIDEOS_FIRST='3' )
        self.assertEqual( result.returncode, 0, result.stdout + result.stderr )
        commands = [ call[1] for call in calls ]
        self.assertEqual( commands, [ 'structure-scan', 'structure-scan', 'structure' ] )
        for call in calls:
            self.assertIn( '--first-videos', call )
            self.assertEqual( call[ call.index( '--first-videos' ) + 1 ], '3' )

    @unittest.skipUnless( shutil.which( 'ffmpeg' ), "reverse_cost.sh checks for ffmpeg first" )
    def test_reverse_passes_the_subset( self ):
        result, calls = self.run_script( 'tools/bench/reverse_cost.sh', VIDEOS_PER_FOLDER='1' )
        self.assertEqual( result.returncode, 0, result.stdout + result.stderr )
        self.assertEqual( len( calls ), 1 )
        self.assertIn( '--videos-per-folder', calls[0] )

    def test_no_subset_passes_no_subset_flag( self ):
        result, calls = self.run_script( 'tools/bench/structure_survey.sh' )
        self.assertEqual( result.returncode, 0, result.stdout + result.stderr )
        self.assertEqual( len( calls ), 3 )
        for call in calls:
            self.assertNotIn( '--first-videos', call )
            self.assertNotIn( '--videos-per-folder', call )

    def test_both_set_stops_before_any_bench_runs( self ):
        result, calls = self.run_script( 'tools/bench/structure_survey.sh',
                                         VIDEOS_FIRST='1', VIDEOS_PER_FOLDER='1' )
        self.assertNotEqual( result.returncode, 0 )
        self.assertEqual( calls, [] )


class TestHarnessVariables( unittest.TestCase ):
    """tools/bench/video_subset.sh, shared by both bench harnesses."""

    def run_helper( self, **env ):
        script = ( 'set -uo pipefail; '
                   'log_info() { echo "INFO $*"; }; log_error() { echo "ERROR $*"; }; '
                   'source tools/bench/video_subset.sh; '
                   'video_subset_args; status=$?; '
                   'echo "ARGS ${VIDEO_SUBSET_ARGS[@]+${VIDEO_SUBSET_ARGS[@]}}"; exit $status' )
        environment = dict( os.environ )
        environment.pop( 'VIDEOS_FIRST', None )
        environment.pop( 'VIDEOS_PER_FOLDER', None )
        environment.update( env )
        return subprocess.run( [ 'bash', '-c', script ], cwd=REPO_ROOT, env=environment,
                               capture_output=True, text=True )

    def test_unset_means_every_file( self ):
        result = self.run_helper()
        self.assertEqual( result.returncode, 0, result.stdout + result.stderr )
        self.assertIn( "ARGS \n", result.stdout )

    def test_each_variable_becomes_its_flag( self ):
        self.assertIn( "ARGS --first-videos 3", self.run_helper( VIDEOS_FIRST='3' ).stdout )
        self.assertIn( "ARGS --videos-per-folder 1",
                       self.run_helper( VIDEOS_PER_FOLDER='1' ).stdout )

    def test_both_set_is_refused( self ):
        result = self.run_helper( VIDEOS_FIRST='3', VIDEOS_PER_FOLDER='1' )
        self.assertEqual( result.returncode, 1 )
        self.assertIn( "set one, not both", result.stdout )

    def replay( self, **env ):
        log = tempfile.NamedTemporaryFile( 'w', suffix='.log', delete=False )
        log.write( "2026-10-08 22:52:56,100 [INFO ] betabench: command=reverse\n"
                   "2026-10-08 22:52:56,177 [INFO ] video subset: the first 2 by name, 2 of 9 file(s):\n"
                   "2026-10-08 22:52:56,177 [INFO ]   uncensored_vids/scene/a.mp4\n"
                   "2026-10-08 22:52:56,177 [INFO ]   source/compilation/b.mp4\n"
                   "2026-10-08 22:52:56,597 [INFO ] reverse: a.mp4, 1920x1080 at 30 fps\n"
                   "2026-10-08 22:52:57,000 [INFO ]   part 1 detail that is not a file\n" )
        log.close()
        self.addCleanup( os.unlink, log.name )
        script = ( 'log_info() { echo "INFO $*"; }; log_error() { echo "ERROR $*"; }; '
                   'source tools/bench/video_subset.sh; video_subset_args >/dev/null; '
                   'video_subset_replay "%s"'%( log.name, ) )
        environment = dict( os.environ )
        environment.pop( 'VIDEOS_FIRST', None )
        environment.pop( 'VIDEOS_PER_FOLDER', None )
        environment.update( env )
        return subprocess.run( [ 'bash', '-c', script ], cwd=REPO_ROOT, env=environment,
                               capture_output=True, text=True ).stdout.splitlines()

    def test_the_chosen_files_are_replayed_to_the_terminal( self ):
        self.assertEqual( self.replay( VIDEOS_FIRST='2' ), [
            "INFO video subset: the first 2 by name, 2 of 9 file(s):",
            "INFO   uncensored_vids/scene/a.mp4",
            "INFO   source/compilation/b.mp4" ] )

    def test_nothing_is_replayed_without_a_subset( self ):
        self.assertEqual( self.replay(), [] )

    def test_a_value_that_is_not_a_positive_whole_number_is_refused( self ):
        for value in ( '0', 'three', '2.5', '-1' ):
            result = self.run_helper( VIDEOS_FIRST=value )
            self.assertEqual( result.returncode, 1, value )
            self.assertIn( "whole number", result.stdout )


if __name__ == '__main__':
    unittest.main()
