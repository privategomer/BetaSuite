"""
test_data_root_tools.py - the tools follow BETASUITE_DATA_ROOT too.

Covers the shell helper (tools/lib/data_root.sh), which must apply the
same rule as betautils_paths, the Python tools' default folders, and a
source scan so a tool cannot reintroduce its own '../output' path.
"""

import importlib.util
import io
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import betautils_paths as bu_paths


REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
HELPER = os.path.join( REPO_ROOT, 'tools', 'lib', 'data_root.sh' )
ENV = bu_paths.ENV_VAR


def _shell_root( value, cwd=None ):
    """(exit status, stdout, stderr) of betasuite_data_root with the variable set to value (None: unset)."""
    env = dict( os.environ )
    env.pop( ENV, None )
    if value is not None:
        env[ENV] = value
    result = subprocess.run( [ 'bash', '-c', 'source "$1"; betasuite_data_root "$2"', '_', HELPER, REPO_ROOT ],
                             env=env, cwd=cwd or REPO_ROOT, capture_output=True, text=True, timeout=30 )
    return result.returncode, result.stdout.strip(), result.stderr


def _python_root( value, cwd=None ):
    """What betautils_paths says for the same value: the folder, or None when it refuses."""
    with mock.patch.dict( os.environ ):
        os.environ.pop( ENV, None )
        if value is not None:
            os.environ[ENV] = value
        cwd_before = os.getcwd()
        os.chdir( cwd or REPO_ROOT )
        try:
            return bu_paths.data_root() or os.path.dirname( REPO_ROOT )
        except bu_paths.DataRootError:
            return None
        finally:
            os.chdir( cwd_before )


def _load_tool( relative ):
    spec = importlib.util.spec_from_file_location(
        '_tool_' + os.path.basename( relative )[:-3], os.path.join( REPO_ROOT, relative ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestShellHelper( unittest.TestCase ):

    def setUp( self ):
        self._tmp = tempfile.TemporaryDirectory( prefix='betasuite-data-root-' )
        self.addCleanup( self._tmp.cleanup )
        self.root = self._tmp.name
        os.mkdir( os.path.join( self.root, 'sub' ) )

    def test_unset_and_empty_are_the_folder_beside_the_repo( self ):
        for value in ( None, '' ):
            status, out, _err = _shell_root( value )
            self.assertEqual( status, 0 )
            self.assertEqual( out, os.path.dirname( REPO_ROOT ) )

    def test_set_is_the_root_normalised( self ):
        for value in ( self.root, self.root + '/', self.root + '/sub/..' ):
            status, out, _err = _shell_root( value )
            self.assertEqual( ( status, out ), ( 0, self.root ), value )

    def test_relative_tilde_and_missing_are_refused_naming_the_variable( self ):
        for value in ( 'sub', '.', '~', '~/x', os.path.join( self.root, 'absent' ) ):
            status, out, err = _shell_root( value, cwd=self.root )
            self.assertNotEqual( status, 0, value )
            self.assertEqual( out, '' )
            self.assertIn( ENV, err )

    def test_the_shell_and_python_agree( self ):
        for value in ( None, '', self.root, self.root + '/sub/..', 'sub', '~',
                       os.path.join( self.root, 'absent' ) ):
            status, out, _err = _shell_root( value, cwd=self.root )
            expected = _python_root( value, cwd=self.root )
            self.assertEqual( out if status == 0 else None, expected, value )


class TestToolDefaults( unittest.TestCase ):

    def setUp( self ):
        self._tmp = tempfile.TemporaryDirectory( prefix='betasuite-data-root-' )
        self.addCleanup( self._tmp.cleanup )
        self.root = self._tmp.name
        patcher = mock.patch.dict( os.environ )
        patcher.start()
        self.addCleanup( patcher.stop )
        os.environ.pop( ENV, None )

    def test_fetch_models_puts_models_under_the_root( self ):
        self.assertEqual( _load_tool( 'tools/setup/fetch_models.py' ).MODEL_DIR,
                          os.path.normpath( os.path.join( REPO_ROOT, '..', 'resources', 'model' ) ) )
        os.environ[ENV] = self.root
        self.assertEqual( _load_tool( 'tools/setup/fetch_models.py' ).MODEL_DIR,
                          os.path.join( self.root, 'resources', 'model' ) )

    def test_bar_smoothing_results_go_under_the_root( self ):
        module = _load_tool( 'tools/tuning/run_bar_smoothing_test.py' )
        self.assertEqual( module.RESULTS_DIR, os.path.join( REPO_ROOT, '..', 'bar_smoothing_test_results' ) )
        os.environ[ENV] = self.root
        module = _load_tool( 'tools/tuning/run_bar_smoothing_test.py' )
        self.assertEqual( module.RESULTS_DIR, os.path.join( self.root, 'bar_smoothing_test_results' ) )
        self.assertEqual( module.QUARANTINE_DIR, os.path.join( self.root, 'bar_smoothing_test_quarantine' ) )

    def test_local_config_log_goes_under_the_root( self ):
        module = _load_tool( 'tools/setup/local_config.py' )
        default = module.build_parser().parse_args( [] ).log_file
        self.assertEqual( default, os.path.join( REPO_ROOT, '..', 'output', 'logs', 'local_config.log' ) )
        os.environ[ENV] = self.root
        module = _load_tool( 'tools/setup/local_config.py' )
        self.assertEqual( module.build_parser().parse_args( [] ).log_file,
                          os.path.join( self.root, 'output', 'logs', 'local_config.log' ) )

    def test_clean_run_clears_under_the_root( self ):
        module = _load_tool( 'tools/tuning/clean_run_both_backends.py' )
        os.environ[ENV] = self.root
        logger = logging.getLogger( 'test_data_root_tools.clean_run' )
        logger.addHandler( logging.NullHandler() )
        logger.propagate = False
        plan = module._describe_clear_plan( logger )
        self.assertEqual( { rel for rel, _abs, _n, _note in plan }, set( module._CLEAR_DIRS ) )
        for rel_dir, abs_dir, _count, _note in plan:
            self.assertEqual( abs_dir, os.path.join( self.root, rel_dir[3:] ) )

    def _quiet_logger( self, name ):
        logger = logging.getLogger( 'test_data_root_tools.' + name )
        logger.handlers[:] = [ logging.NullHandler() ]
        logger.propagate = False
        return logger

    def test_clean_run_deletes_under_the_root_only( self ):
        module = _load_tool( 'tools/tuning/clean_run_both_backends.py' )
        # A stand-in repo inside the temp folder, so a path that missed the
        # root would still land in the temp folder and not beside this repo.
        core = os.path.join( self.root, 'core' )
        data = os.path.join( self.root, 'data' )
        os.makedirs( os.path.join( data, 'output', 'cache', 'vid_hashes' ) )
        os.makedirs( core )
        stale = os.path.join( data, 'output', 'cache', 'vid_hashes', 'old.gz' )
        memo = os.path.join( data, 'output', 'cache', 'file_hashes.json' )
        for path in ( stale, memo ):
            open( path, 'w' ).close()
        os.environ[ENV] = data
        with mock.patch.object( module, '_CORE_DIR', core ):
            module.clear_everything( self._quiet_logger( 'clean_run_clear' ) )
        self.assertFalse( os.path.exists( stale ) )
        self.assertFalse( os.path.exists( memo ) )
        self.assertFalse( os.path.exists( os.path.join( self.root, 'output' ) ) )

    def test_verify_env_loads_models_from_under_the_root( self ):
        module = _load_tool( 'tools/setup/verify_env.py' )
        model_dir = os.path.join( self.root, 'resources', 'model' )
        os.makedirs( model_dir )
        with open( os.path.join( model_dir, 'v3.4-320n.onnx' ), 'w' ) as handle:
            handle.write( '<html>not a model</html>' )
        os.environ[ENV] = self.root
        log = logging.getLogger( 'test_data_root_tools.verify_models' )
        stream = io.StringIO()
        log.handlers[:] = [ logging.StreamHandler( stream ) ]
        log.setLevel( logging.DEBUG )
        log.propagate = False
        checker = module.Checker( log )
        cwd = os.getcwd()
        os.chdir( tempfile.gettempdir() )
        try:
            module.check_models( checker, False )
        finally:
            os.chdir( cwd )
        self.assertIn( 'HTML page', stream.getvalue() )

    def test_summarize_run_reads_the_stats_under_the_root( self ):
        import contextlib
        import betaconfig
        module = _load_tool( 'tools/analysis/summarize_run.py' )
        os.makedirs( os.path.join( self.root, 'output', 'stats' ) )
        stats = os.path.join( self.root, 'output', 'stats', 'betasuite_stats.jsonl' )
        open( stats, 'w' ).close()
        os.environ[ENV] = self.root
        output = io.StringIO()
        with mock.patch.object( betaconfig, 'stats_path', '../output/stats/betasuite_stats.jsonl' ), \
             mock.patch.object( sys, 'argv', [ 'summarize_run.py' ] ), contextlib.redirect_stdout( output ):
            try:
                module.main()
            except SystemExit:
                pass
        self.assertIn( stats, output.getvalue() )

    def test_cleanup_output_rotates_the_log_under_the_root( self ):
        import betaconfig
        module = _load_tool( 'tools/maintenance/cleanup_output.py' )
        os.environ[ENV] = self.root
        seen = {}

        def fake_plan( root, now=None, stale_days=14, log_path=None, log_max_mb=50 ):
            seen['log_path'] = log_path
            return []
        with mock.patch.object( betaconfig, 'log_path', '../output/logs/betasuite.log' ), \
             mock.patch.object( module, 'plan', fake_plan ):
            try:
                module.main( [], out=lambda *_: None )
            except SystemExit:
                pass
        self.assertEqual( seen['log_path'], os.path.join( self.root, 'output', 'logs', 'betasuite.log' ) )

    def test_verify_env_checks_folders_under_the_root( self ):
        module = _load_tool( 'tools/setup/verify_env.py' )
        for d in module.DIRS:
            os.makedirs( os.path.join( self.root, d[3:] ) )
        os.environ[ENV] = self.root
        log = logging.getLogger( 'test_data_root_tools.verify_env' )
        stream = io.StringIO()
        log.addHandler( logging.StreamHandler( stream ) )
        log.setLevel( logging.DEBUG )
        log.propagate = False
        checker = module.Checker( log )
        cwd = os.getcwd()
        os.chdir( tempfile.gettempdir() )
        try:
            module.check_dirs( checker )
        finally:
            os.chdir( cwd )
        self.assertEqual( checker.warnings, 0, stream.getvalue() )
        self.assertIn( self.root, stream.getvalue() )


class TestNoToolBuildsItsOwnDataPath( unittest.TestCase ):
    """
    Every way a tool used to reach the data folder, outside the resolver.
    A new script that writes one of these again would skip the data root.
    """

    SHELL_PATTERNS = [
        re.compile( r'cd "\$(REPO_ROOT|CORE_DIR|REPO_DIR)/\.\."' ),
        re.compile( r'\$(REPO_ROOT|CORE_DIR|REPO_DIR)/\.\./(output|resources)' ),
        re.compile( r'(^|[\s"=])\.\./(output|resources)\b' ),
        re.compile( r'(^|[\s"=])(output|resources)/(cache|logs|benchmarks|stats|model)\b' ),
    ]
    PYTHON_PATTERNS = [
        re.compile( r"""['"]\.\.['"]\s*,\s*['"](output|resources)['"]""" ),
        re.compile( r"""dirname\(\s*(CORE_DIR|REPO_ROOT|_CORE_DIR|BASE_DIR)\s*\)\s*,\s*['"](output|resources)['"]""" ),
    ]

    @staticmethod
    def _code_lines( path ):
        with open( path, encoding='UTF-8' ) as handle:
            for number, line in enumerate( handle, 1 ):
                stripped = line.strip()
                if stripped.startswith( '#' ):
                    continue
                yield number, line

    def _sources( self, suffix ):
        for top in ( os.path.join( REPO_ROOT, 'tools' ), ):
            for dirpath, dirnames, filenames in os.walk( top ):
                dirnames[:] = [ d for d in dirnames if d != '__pycache__' ]
                for name in sorted( filenames ):
                    if name.endswith( suffix ):
                        yield os.path.join( dirpath, name )
        for name in sorted( os.listdir( REPO_ROOT ) ):
            if name.endswith( suffix ) and name != 'betautils_paths.py':
                yield os.path.join( REPO_ROOT, name )

    def test_shell_scripts( self ):
        found = []
        for path in self._sources( '.sh' ):
            for number, line in self._code_lines( path ):
                # echo/printf text and heredoc help lines are messages, not paths.
                if re.match( r'\s*(echo|printf|log_\w+)\b', line ):
                    continue
                if any( p.search( line ) for p in self.SHELL_PATTERNS ):
                    found.append( '%s:%d: %s'%( os.path.relpath( path, REPO_ROOT ), number, line.strip() ) )
        self.assertEqual( found, [], 'use tools/lib/data_root.sh instead:\n' + '\n'.join( found ) )

    def test_every_script_stops_when_the_data_root_is_refused( self ):
        assignment = re.compile( r'^\s*DATA_ROOT="\$\(betasuite_data_root ' )
        unguarded = []
        for path in self._sources( '.sh' ):
            for number, line in self._code_lines( path ):
                if assignment.match( line ) and not re.search( r'\|\|\s*(exit 1|die\b)', line ):
                    unguarded.append( '%s:%d'%( os.path.relpath( path, REPO_ROOT ), number ) )
        self.assertEqual( unguarded, [], "add '|| exit 1' so a bad root stops the script" )

    def test_python_tools( self ):
        found = []
        for path in self._sources( '.py' ):
            for number, line in self._code_lines( path ):
                if 'bu_paths.resolve(' in line:
                    continue
                if any( p.search( line ) for p in self.PYTHON_PATTERNS ):
                    found.append( '%s:%d: %s'%( os.path.relpath( path, REPO_ROOT ), number, line.strip() ) )
        self.assertEqual( found, [], 'use betautils_paths.beside_repo instead:\n' + '\n'.join( found ) )

    def test_the_scan_sees_the_forms_it_guards( self ):
        self.assertTrue( any( p.search( 'RUN_DIR="$(cd "$REPO_ROOT/.." && pwd)/output/x"' ) for p in self.SHELL_PATTERNS ) )
        self.assertTrue( any( p.search( 'for f in output/cache/vid_hashes/*.gz; do' ) for p in self.SHELL_PATTERNS ) )
        self.assertTrue( any( p.search( 'mkdir -p ../output/logs' ) for p in self.SHELL_PATTERNS ) )
        self.assertTrue( any( p.search( "os.path.join( REPO_ROOT, '..', 'output', 'logs' )" ) for p in self.PYTHON_PATTERNS ) )
        self.assertTrue( any( p.search( "os.path.join( os.path.dirname( CORE_DIR ), 'output' )" ) for p in self.PYTHON_PATTERNS ) )


if __name__ == '__main__':
    unittest.main()
