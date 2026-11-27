"""
test_upstream_check.py - tools/release/upstream_check.py finds code that
still comes from an upstream tree, and its reports never quote that tree.

Every case builds a small synthetic upstream tree and a synthetic repository
in a temporary folder. No network and no real history: the blame case makes
its own throwaway git repository.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, os.path.join( REPO, 'tools', 'release' ) )

import upstream_check as uc  # noqa: E402

UPSTREAM_PY = textwrap.dedent( '''\
    import os

    def overlap_of_two_boxes( first_box, second_box ):
        if first_box['x'] + first_box['w'] < second_box['x']:
            return False
        if first_box['y'] + first_box['h'] < second_box['y']:
            return False
        if first_box['x'] > second_box['x'] + second_box['w']:
            return False
        return first_box['y'] <= second_box['y'] + second_box['h']

    def greeting_for_the_person( person_name ):
        message = 'a distinctive greeting line for ' + person_name
        return message.upper()
    ''' )

SECRET = 'a distinctive greeting line for'


def _write( root, rel, text ):
    path = os.path.join( root, rel )
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    with open( path, 'w', encoding='utf-8' ) as handle:
        handle.write( text )


class Fixture( unittest.TestCase ):
    def setUp( self ):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup( self.tmp.cleanup )
        quiet = mock.patch.object( uc, 'log', lambda level, message: None )
        quiet.start()
        self.addCleanup( quiet.stop )
        self.upstream = os.path.join( self.tmp.name, 'upstream' )
        self.repo = os.path.join( self.tmp.name, 'repo' )
        self.out = os.path.join( self.tmp.name, 'out' )
        _write( self.upstream, 'tool.py', UPSTREAM_PY )
        _write( self.upstream, uc.MARKER, uc.DEFAULT_COMMIT + '\n' )

    def run_main( self, *argv ):
        return uc.main( list( argv ) )

    def report( self, name ):
        with open( os.path.join( self.out, f'{name}.json' ), encoding='utf-8' ) as handle:
            return json.load( handle )


class TestVerbatimAndRuns( Fixture ):
    def test_copied_line_is_found_with_its_line_number( self ):
        _write( self.repo, 'mine.py', 'x = 1\n    message   =  \'a distinctive greeting line for \' + person_name\n' )
        upstream = uc.Upstream( self.upstream )
        hits = uc.verbatim( self.repo, uc.list_files( self.repo ), upstream, 12 )
        self.assertEqual( [ 2 ], [ h[ 'line' ] for h in hits[ 'mine.py' ] ] )
        self.assertFalse( hits[ 'mine.py' ][ 0 ][ 'trivial' ] )

    def test_own_code_is_not_flagged( self ):
        _write( self.repo, 'mine.py', 'def unrelated_helper( value ):\n    return value * 2 + 7\n' )
        upstream = uc.Upstream( self.upstream )
        self.assertEqual( {}, uc.verbatim( self.repo, uc.list_files( self.repo ), upstream, 12 ) )

    def test_marker_file_is_not_upstream_text( self ):
        _write( self.repo, 'pins.py', f"PIN = '{uc.DEFAULT_COMMIT}'\n{uc.DEFAULT_COMMIT}\n" )
        upstream = uc.Upstream( self.upstream )
        self.assertEqual( {}, uc.verbatim( self.repo, uc.list_files( self.repo ), upstream, 12 ) )

    def test_run_of_three_copied_lines_is_one_range( self ):
        body = UPSTREAM_PY.splitlines()[ 2:7 ]
        _write( self.repo, 'mine.py', 'y = 2\n' + '\n'.join( body ) + '\nz = 3\n' )
        upstream = uc.Upstream( self.upstream )
        found = uc.runs( self.repo, uc.list_files( self.repo ), upstream, 3 )
        self.assertEqual( [ ( 2, 6, 5 ) ], [ ( r[ 'start' ], r[ 'end' ], r[ 'lines' ] ) for r in found[ 'mine.py' ] ] )

    def test_short_runs_are_ignored( self ):
        _write( self.repo, 'mine.py', 'import os\n\n' )
        upstream = uc.Upstream( self.upstream )
        self.assertEqual( {}, uc.runs( self.repo, uc.list_files( self.repo ), upstream, 3 ) )

    def test_trivial_lines( self ):
        self.assertTrue( uc.is_trivial( '' ) )
        self.assertTrue( uc.is_trivial( 'import betautils_config as bu_config' ) )
        self.assertTrue( uc.is_trivial( 'from os import path, sep' ) )
        self.assertTrue( uc.is_trivial( 'cap.release()' ) )
        self.assertFalse( uc.is_trivial( "message = 'a distinctive greeting'" ) )


class TestStructural( Fixture ):
    def test_renamed_copy_scores_high_and_names_its_source( self ):
        renamed = UPSTREAM_PY.replace( 'overlap_of_two_boxes', 'boxes_touch' )
        _write( self.repo, 'mine.py', renamed.replace( 'first_box', 'a' ).replace( 'second_box', 'b' ) )
        upstream = uc.Upstream( self.upstream )
        rows = { r[ 'unit' ]: r for r in uc.structural( self.repo, uc.list_files( self.repo ), upstream ) }
        self.assertGreaterEqual( rows[ 'boxes_touch' ][ 'norm_share' ], 0.9 )
        self.assertEqual( 'tool.py:overlap_of_two_boxes', rows[ 'boxes_touch' ][ 'best_upstream_unit' ] )

    def test_units_split_methods_from_class_body( self ):
        source = 'class A:\n    k = 1\n    def m( self ):\n        return 2\n\nx = 3\n'
        names = [ u[ 'name' ] for u in uc.python_units( source ) ]
        self.assertEqual( [ 'A', 'A.m', '<top>@L6' ], names )

    def test_line_hits_count_inside_their_unit_only( self ):
        _write( self.repo, 'mine.py', UPSTREAM_PY )
        upstream = uc.Upstream( self.upstream )
        hits = { 'mine.py': { 'blame': { 4, 13 }, 'substantive': { 13 } } }
        rows = { r[ 'unit' ]: r for r in uc.structural( self.repo, uc.list_files( self.repo ), upstream, hits ) }
        self.assertEqual( 1, rows[ 'overlap_of_two_boxes' ][ 'blame_lines' ] )
        self.assertEqual( 0, rows[ 'overlap_of_two_boxes' ][ 'substantive_lines' ] )
        self.assertEqual( 1, rows[ 'greeting_for_the_person' ][ 'substantive_lines' ] )


class TestCommands( Fixture ):
    def test_gate_fails_on_a_copy_and_reports_no_upstream_text( self ):
        _write( self.repo, 'mine.py', UPSTREAM_PY )
        self.assertEqual( 1, self.run_main( 'gate', '--upstream', self.upstream, '--repo', self.repo, '--out', self.out ) )
        for name in ( 'gate.md', 'gate.json' ):
            with open( os.path.join( self.out, name ), encoding='utf-8' ) as handle:
                self.assertNotIn( SECRET, handle.read() )
        self.assertTrue( self.report( 'gate' )[ 'failures' ] )

    def test_gate_passes_on_own_code( self ):
        _write( self.repo, 'mine.py', 'def unrelated_helper( value ):\n    return value * 2 + 7\n' )
        self.assertEqual( 0, self.run_main( 'gate', '--upstream', self.upstream, '--repo', self.repo, '--out', self.out ) )

    def test_verify_uses_the_allowlist_and_needs_reasons( self ):
        line = "        message = 'a distinctive greeting line for ' + person_name"
        _write( self.repo, 'mine.py', line + '\n' )
        args = ( 'verify', '--upstream', self.upstream, '--repo', self.repo, '--out', self.out )
        self.assertEqual( 1, self.run_main( *args ) )
        allow = os.path.join( self.tmp.name, 'allow.json' )
        entry = { 'path': 'mine.py', 'hash': uc.line_hash( uc.normalize( line ) ), 'reason': 'test' }
        with open( allow, 'w' ) as handle:
            json.dump( { 'entries': [ entry ] }, handle )
        self.assertEqual( 0, self.run_main( *args, '--allowlist', allow ) )
        entry[ 'reason' ] = ' '
        with open( allow, 'w' ) as handle:
            json.dump( { 'entries': [ entry ] }, handle )
        with self.assertRaises( ValueError ):
            self.run_main( *args, '--allowlist', allow )

    def test_verify_ignores_lines_under_its_threshold( self ):
        _write( self.repo, 'mine.py', 'import os\n' )
        self.assertEqual( 0, self.run_main( 'verify', '--upstream', self.upstream, '--repo', self.repo, '--out', self.out ) )

    def test_missing_upstream_copy_is_an_error( self ):
        _write( self.repo, 'mine.py', 'x = 1\n' )
        missing = os.path.join( self.tmp.name, 'nothing' )
        self.assertEqual( 1, self.run_main( 'gate', '--upstream', missing, '--repo', self.repo, '--out', self.out ) )


class TestBlame( Fixture ):
    def git( self, *args ):
        env = dict( os.environ, GIT_AUTHOR_NAME='a', GIT_AUTHOR_EMAIL='a@example.invalid',
                    GIT_COMMITTER_NAME='a', GIT_COMMITTER_EMAIL='a@example.invalid',
                    GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1' )
        return subprocess.run( [ 'git', '-C', self.repo, *args ], env=env, check=True,
                               capture_output=True, text=True ).stdout.strip()

    def test_lines_from_the_upstream_commit_are_blamed( self ):
        os.makedirs( self.repo )
        self.git( 'init', '-q' )
        _write( self.repo, 'mine.py', 'first_line_from_upstream = 1\nsecond = 2\n' )
        self.git( 'add', '.' )
        self.git( 'commit', '-q', '-m', 'upstream' )
        base = self.git( 'rev-parse', 'HEAD' )
        _write( self.repo, 'mine.py', 'first_line_from_upstream = 1\nsecond_line_rewritten_here = 3\n' )
        self.git( 'commit', '-q', '-am', 'fork' )
        hits = uc.blame( self.repo, [ 'mine.py' ], base )
        self.assertEqual( [ 1 ], [ h[ 'line' ] for h in hits[ 'mine.py' ] ] )

    def test_unknown_commit_skips_blame( self ):
        _write( self.repo, 'mine.py', 'x = 1\n' )
        self.assertIsNone( uc.blame( self.repo, [ 'mine.py' ], '0' * 40 ) )


if __name__ == '__main__':
    unittest.main()
