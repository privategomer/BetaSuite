"""
test_clean_run_both_backends.py - regression coverage for
tools/tuning/clean_run_both_backends.py's clear-path logic.

Covers a real bug caught by this script's own smoke test 2026-09-17:
an earlier version tried to keep only THIS run's own active log
subdirectory alive during the clear step, via a path-based "is this
entry an ancestor of the path to keep" check. That's fragile in a way
the smoke test caught directly: a top-level entry in a cleared
directory (e.g. '.../logs/clean_run_both_backends', the shared parent
every timestamped run directory lives under) is only ever an ANCESTOR
of the specific run directory to keep, never equal to it - so a naive
equality check deleted the active log directory anyway (breaking every
subsequent log call in that same process), and even the corrected
ancestor-aware check still couldn't distinguish the active run's
subdirectory from an OLDER SIBLING run subdirectory sharing the same
parent - both are "inside a kept ancestor" from the top level's point
of view, so a per-run keep-path was either too broad (spares stale
sibling runs too) or required descending into the tree with the same
fragile logic at every level.

Fixed by dropping the path-based approach in favor of a simple,
name-based whole-subtree exclusion (_EXCLUDED_NAMES_BY_CLEAR_DIR):
../output/logs/clean_run_both_backends/ is excluded by NAME, wholesale,
every run - not just today's - since analysis tools never read
../output/logs at all (only the cache directories), so a stale log
file here can never pollute a future run's conclusions the way a stale
CACHE file can (the actual problem this script exists to solve).
"""

import importlib.util
import os
import sys
import tempfile
import shutil
import unittest

_HERE = os.path.dirname( os.path.abspath( __file__ ) )
_REPO_ROOT = os.path.dirname( _HERE )


def _load_module():
    full_path = os.path.join( _REPO_ROOT, 'tools', 'tuning', 'clean_run_both_backends.py' )
    spec = importlib.util.spec_from_file_location( 'clean_run_both_backends_under_test', full_path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestClearEverythingExcludesOwnLogHistory( unittest.TestCase ):
    """
    End-to-end proof: clear_everything must never delete anything under
    ../output/logs/clean_run_both_backends/ - not the active run's own
    subdirectory, and not an older sibling run's subdirectory either -
    while still deleting everything ELSE under ../output/logs (a stray
    betasuite.log, or any other stale file/dir that isn't this script's
    own excluded name).
    """

    def setUp( self ):
        self.mod = _load_module()
        self._tmpdir = tempfile.mkdtemp()
        self._core_dir = os.path.join( self._tmpdir, 'core' )
        os.makedirs( self._core_dir, exist_ok=True )
        self.mod._CORE_DIR = self._core_dir
        self.mod._CLEAR_DIRS = [ '../output/logs' ]

        self._logs_dir = os.path.join( self._tmpdir, 'output', 'logs' )

        # this run's own active log directory
        self._active_run_dir = os.path.join( self._logs_dir, 'clean_run_both_backends', '20260917_201441' )
        os.makedirs( self._active_run_dir, exist_ok=True )
        with open( os.path.join( self._active_run_dir, 'clean_run_both_backends.log' ), 'w' ) as f:
            f.write( 'in-progress log content\n' )

        # an OLDER sibling run directory under the same shared parent -
        # the case the earlier path-based approach couldn't distinguish
        self._old_run_dir = os.path.join( self._logs_dir, 'clean_run_both_backends', '20260101_000000' )
        os.makedirs( self._old_run_dir, exist_ok=True )
        with open( os.path.join( self._old_run_dir, 'clean_run_both_backends.log' ), 'w' ) as f:
            f.write( 'old run\n' )

        # a stray unrelated log file that SHOULD be deleted
        with open( os.path.join( self._logs_dir, 'betasuite.log' ), 'w' ) as f:
            f.write( 'stale\n' )

        # a stray unrelated subdirectory that SHOULD be deleted
        other_stray_dir = os.path.join( self._logs_dir, 'analysis_runs', '20260101_000000' )
        os.makedirs( other_stray_dir, exist_ok=True )
        with open( os.path.join( other_stray_dir, 'analyze_jitter.log' ), 'w' ) as f:
            f.write( 'stale analysis log\n' )

    def tearDown( self ):
        shutil.rmtree( self._tmpdir, ignore_errors=True )

    def test_clear_everything_spares_whole_log_history_but_clears_everything_else( self ):
        import logging
        logger = logging.getLogger( 'test_clean_run_both_backends' )
        logger.addHandler( logging.NullHandler() )
        logger.setLevel( logging.CRITICAL + 1 )

        self.mod.clear_everything( logger )

        # active run's own log survives, content intact
        active_log_file = os.path.join( self._active_run_dir, 'clean_run_both_backends.log' )
        self.assertTrue( os.path.exists( active_log_file ), "active run's log directory was deleted" )
        with open( active_log_file ) as f:
            self.assertEqual( f.read(), 'in-progress log content\n' )

        # OLDER sibling run also survives - the case the earlier
        # approach got wrong
        old_log_file = os.path.join( self._old_run_dir, 'clean_run_both_backends.log' )
        self.assertTrue( os.path.exists( old_log_file ), "older sibling run's log directory was deleted" )

        # everything else under ../output/logs IS cleared
        self.assertFalse( os.path.exists( os.path.join( self._logs_dir, 'betasuite.log' ) ) )
        self.assertFalse( os.path.exists( os.path.join( self._logs_dir, 'analysis_runs' ) ) )

    def test_describe_clear_plan_count_excludes_log_history( self ):
        import logging
        logger = logging.getLogger( 'test_clean_run_both_backends_plan' )
        logger.addHandler( logging.NullHandler() )
        logger.setLevel( logging.CRITICAL + 1 )

        plan = self.mod._describe_clear_plan( logger )
        rel_dir, abs_dir, count, note = plan[0]
        self.assertEqual( rel_dir, '../output/logs' )
        # top level of ../output/logs has exactly 2 entries: betasuite.log
        # and analysis_runs/ (clean_run_both_backends/ is excluded by name,
        # so it must NOT be counted even though it exists)
        self.assertEqual( count, 2, "clean_run_both_backends/ should be excluded from the count" )


class TestExcludedNamesConfig( unittest.TestCase ):

    def setUp( self ):
        self.mod = _load_module()

    def test_logs_dir_excludes_clean_run_both_backends_by_name( self ):
        excluded = self.mod._EXCLUDED_NAMES_BY_CLEAR_DIR.get( '../output/logs', set() )
        self.assertIn( 'clean_run_both_backends', excluded )

    def test_cache_dirs_have_no_exclusions( self ):
        # nothing under cache/ should ever be spared - that's the whole
        # point of this script (no stale cache surviving a "clean" run)
        for cache_dir in ( '../output/cache/vid_hashes', '../output/cache/pic_hashes' ):
            self.assertEqual( self.mod._EXCLUDED_NAMES_BY_CLEAR_DIR.get( cache_dir, set() ), set() )


if __name__ == '__main__':
    unittest.main()
