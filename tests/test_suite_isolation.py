"""
The suite must never write into the user's real output folder.

Before tests/__init__.py redirected them, every unpatched test resolved
betaconst's '../output/...' paths against the repository root, which is
the user's real output folder. A release.py test pass interleaved ~230
synthetic clip.mp4 lines into a live run's betasuite.log and left ~48
run-key records where betabench discovers cached configurations.

These tests read betaconst.py's SOURCE, so a path added there later
without a matching redirect fails here instead of quietly writing into
the real folder again.
"""

import ast
import os
import unittest

import betaconfig
import betaconst
import betautils_cache_paths
import betautils_log
import tests


def _relative_path_constants_in( module_path ):
    """
    Module-level NAME = '../...' string assignments in a source file.

    Read from the file rather than the live module, because the live
    module has already been redirected and would show the new values.
    """
    with open( module_path, 'r', encoding='UTF-8' ) as handle:
        tree = ast.parse( handle.read() )
    found = {}
    for node in tree.body:
        if not isinstance( node, ast.Assign ):
            continue
        if not ( isinstance( node.value, ast.Constant ) and isinstance( node.value.value, str ) ):
            continue
        if not node.value.value.startswith( '../' ):
            continue
        for target in node.targets:
            if isinstance( target, ast.Name ):
                found[target.id] = node.value.value
    return found


def _is_under_test_root( path ):
    root = os.path.realpath( tests.TEST_OUTPUT_ROOT )
    return os.path.realpath( path ).startswith( root + os.sep )


class TestSuiteIsolation( unittest.TestCase ):

    def test_the_source_scan_finds_the_paths_it_guards( self ):
        # Precondition: if the scan found nothing, every assertion below
        # would pass vacuously.
        found = _relative_path_constants_in( betaconst.__file__ )
        self.assertIn( 'run_key_dir', found )
        self.assertIn( 'video_path_censored', found )
        self.assertGreaterEqual( len( found ), 10 )

    def test_every_relative_betaconst_path_is_redirected( self ):
        # Model and sticker paths are read-only inputs the tests need,
        # and live in detectors/ and betaconfig, not betaconst.
        for name, original in _relative_path_constants_in( betaconst.__file__ ).items():
            live = getattr( betaconst, name )
            self.assertTrue( _is_under_test_root( live ),
                "betaconst.%s is %r, outside the test root (originally %r): a test "
                "touching it writes into the real folder. Add it to "
                "tests/__init__.py REDIRECTED_PATHS."%( name, live, original ) )

    def test_log_and_stats_paths_are_redirected_at_both_layers( self ):
        for label, live in ( ( 'betaconfig.log_path', betaconfig.log_path ),
                             ( 'betaconfig.stats_path', betaconfig.stats_path ),
                             ( 'betautils_log.DEFAULT_LOG_PATH', betautils_log.DEFAULT_LOG_PATH ),
                             ( 'betautils_log.DEFAULT_STATS_PATH', betautils_log.DEFAULT_STATS_PATH ) ):
            self.assertTrue( _is_under_test_root( live ), "%s is %r"%( label, live ) )

    def test_import_time_copies_were_taken_after_the_redirect( self ):
        # betautils_cache_paths copies these at import. If any test
        # module imported it before tests/__init__.py ran, the copies
        # would still name the real folder.
        self.assertTrue( _is_under_test_root( betautils_cache_paths.VID_HASH_DIR ) )
        self.assertTrue( _is_under_test_root( betautils_cache_paths.PIC_HASH_DIR ) )


if __name__ == '__main__':
    unittest.main()
