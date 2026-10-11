"""
test_betabench_empty_results.py - "no caches" only when there are none.

A cache-reading betabench subcommand runs once per cached configuration.
dedup with one picture size reads the caches, says it has nothing to
tune, and returns no rows; the wrapper then also warned "no cached
detections found for any configuration", and the 2026-10-11 hub run sent
people looking for caches that were there. The warning is now kept for
the case it describes.
"""

import argparse
import importlib.util
import os
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load_betabench():
    path = os.path.join( REPO_ROOT, 'tools', 'bench', 'betabench.py' )
    spec = importlib.util.spec_from_file_location( '_betabench_empty_results', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class _Logger( object ):

    def __init__( self ):
        self.warnings = []
        self.infos = []

    def info( self, text, *args ):
        self.infos.append( text%args if args else text )

    def warning( self, text, *args ):
        self.warnings.append( text%args if args else text )

    debug = error = info


class _Config( object ):
    label = 'nudenet_v3/640m @640'
    backend_name = 'nudenet_v3'
    variant = '640m'
    size = 640
    picture_sizes = [ 640 ]


class TestEmptyResults( unittest.TestCase ):

    def setUp( self ):
        self.bench = _load_betabench()
        self.args = argparse.Namespace( sizes=None, backend=None, variants=None, sample_fps=None )

    def run_with( self, loaded, body ):
        logger = _Logger()
        run = self.bench._per_configuration( 'dedup', body )
        with mock.patch.object( self.bench.bu_cache, 'configurations_to_analyse',
                                return_value=[ _Config() ] ), \
                mock.patch.object( self.bench, 'load_cached_detections', return_value=loaded ):
            result = run( self.args, logger )
        return result, logger

    def test_caches_read_but_nothing_to_report_is_not_missing_caches( self ):
        result, logger = self.run_with( ( [ { 't': 0 } ], 1 ), lambda *args: { 'dedup': [] } )
        self.assertEqual( result['dedup'], [] )
        self.assertFalse( [ line for line in logger.warnings if 'no cached detections' in line ] )
        self.assertTrue( [ line for line in logger.infos if 'nothing to report' in line ] )

    def test_no_caches_still_warns( self ):
        result, logger = self.run_with( ( [], 0 ), lambda *args: { 'dedup': [ { 'x': 1 } ] } )
        self.assertTrue( [ line for line in logger.warnings if 'no cached detections' in line ] )

    def test_rows_are_tagged_and_no_message( self ):
        result, logger = self.run_with( ( [ { 't': 0 } ], 1 ), lambda *args: { 'dedup': [ { 'x': 1 } ] } )
        self.assertEqual( result['dedup'][0]['configuration'], _Config.label )
        self.assertFalse( logger.warnings )


class TestGroups( unittest.TestCase ):
    """cache is what the hub runs every time; hardware is the opt-in timing."""

    def setUp( self ):
        self.bench = _load_betabench()

    def test_cache_and_hardware_split_all_with_no_timing_in_cache( self ):
        self.assertEqual( self.bench.GROUPS['all'], self.bench.GROUPS['cache'] + self.bench.GROUPS['hardware'] )
        for timing in ( 'render', 'decode', 'detect', 'encode', 'reverse' ):
            self.assertNotIn( timing, self.bench.GROUPS['cache'] )

    def test_the_parser_takes_each_group( self ):
        for group in ( 'all', 'cache', 'hardware' ):
            self.assertEqual( self.bench.build_parser().parse_args( [ group ] ).command, group )


if __name__ == '__main__':
    unittest.main()
