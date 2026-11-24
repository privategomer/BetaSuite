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
import tests


REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
HELPER = os.path.join( REPO_ROOT, 'tools', 'lib', 'data_root.sh' )
ENV = bu_paths.ENV_VAR


def _shell_path( value, relative, cwd=None, extra=None ):
    """(exit status, stdout, stderr) of betasuite_path with the data root set to value (None: unset)."""
    env = dict( os.environ )
    env.pop( ENV, None )
    if value is not None:
        env[ENV] = value
    env.update( extra or {} )
    result = subprocess.run( [ 'bash', '-c', 'source "$1"; betasuite_path "$2" "$3"', '_', HELPER, REPO_ROOT, relative ],
                             env=env, cwd=cwd or REPO_ROOT, capture_output=True, text=True, timeout=30 )
    return result.returncode, result.stdout.strip(), result.stderr


def _python_path( value, relative, cwd=None, extra=None ):
    """What betautils_paths says for the same case: the folder, or None when it refuses."""
    with mock.patch.dict( os.environ ):
        os.environ.pop( ENV, None )
        if value is not None:
            os.environ[ENV] = value
        os.environ.update( extra or {} )
        cwd_before = os.getcwd()
        os.chdir( cwd or REPO_ROOT )
        try:
            return os.path.abspath( bu_paths.legacy_path( *relative.split( '/' ) ) )
        except bu_paths.PathsError:
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

    RELATIVES = ( 'output/logs', 'output/cache/vid_hashes', 'output/benchmarks', 'resources/model',
                  'resources/uncensored_vids', 'output', 'bar_smoothing_test_results' )

    def setUp( self ):
        self._tmp = tempfile.TemporaryDirectory( prefix='betasuite-data-root-' )
        self.addCleanup( self._tmp.cleanup )
        self.root = self._tmp.name
        os.mkdir( os.path.join( self.root, 'sub' ) )

    def test_unset_is_the_os_folders( self ):
        for relative in self.RELATIVES:
            status, out, err = _shell_path( None, relative )
            self.assertEqual( status, 0, err )
            self.assertTrue( out.startswith( tests.TEST_OS_ROOT + os.sep ), ( relative, out ) )
        self.assertEqual( _shell_path( None, 'output/logs' )[1], os.path.join( bu_paths.kind_dir( 'state' ), 'logs' ) )

    def test_set_is_the_1x_path_under_the_root_normalised( self ):
        for value in ( self.root, self.root + '/', self.root + '/sub/..' ):
            status, out, _err = _shell_path( value, 'output/logs' )
            self.assertEqual( ( status, out ), ( 0, os.path.join( self.root, 'output', 'logs' ) ), value )

    def test_relative_tilde_missing_and_clashing_are_refused_naming_the_variable( self ):
        for value in ( 'sub', '.', '~', '~/x', os.path.join( self.root, 'absent' ) ):
            status, out, err = _shell_path( value, 'output/logs', cwd=self.root )
            self.assertNotEqual( status, 0, value )
            self.assertEqual( out, '' )
            self.assertIn( ENV, err )
        status, out, err = _shell_path( self.root, 'output/logs', extra={ bu_paths.CACHE_ENV_VAR: self.root } )
        self.assertNotEqual( status, 0 )
        self.assertIn( bu_paths.CACHE_ENV_VAR, err )
        status, out, err = _shell_path( None, 'output/cache', extra={ bu_paths.CACHE_ENV_VAR: 'rel' } )
        self.assertNotEqual( status, 0 )
        self.assertIn( bu_paths.CACHE_ENV_VAR, err )

    def test_the_shell_and_python_agree( self ):
        for value in ( None, '', self.root, self.root + '/sub/..', 'sub', '~', os.path.join( self.root, 'absent' ) ):
            for relative in self.RELATIVES:
                status, out, _err = _shell_path( value, relative, cwd=self.root )
                expected = _python_path( value, relative, cwd=self.root )
                self.assertEqual( out if status == 0 else None, expected, ( value, relative ) )

    def test_where_prints_one_line_per_argument_and_lists_every_kind( self ):
        where = os.path.join( REPO_ROOT, 'tools', 'lib', 'where.py' )
        result = subprocess.run( [ sys.executable, where, 'output/logs', 'resources\\model' ],
                                 capture_output=True, text=True, timeout=30 )
        self.assertEqual( result.returncode, 0, result.stderr )
        self.assertEqual( result.stdout.splitlines(), [ os.path.join( bu_paths.kind_dir( 'state' ), 'logs' ),
                                                        os.path.join( bu_paths.kind_dir( 'data' ), 'model' ) ] )
        kinds = subprocess.run( [ sys.executable, where, 'inputs', 'outputs', 'cache' ], capture_output=True,
                                text=True, timeout=30 )
        self.assertEqual( kinds.stdout.splitlines(), [ bu_paths.kind_dir( 'inputs' ), bu_paths.kind_dir( 'outputs' ),
                                                       bu_paths.kind_dir( 'cache' ) ] )
        listing = subprocess.run( [ sys.executable, where ], capture_output=True, text=True, timeout=30 )
        self.assertEqual( listing.returncode, 0 )
        self.assertEqual( [ line.split( '\t' ) for line in listing.stdout.splitlines() ],
                          [ [ kind, bu_paths.kind_dir( kind ) ] for kind in bu_paths.KINDS ] )


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
                          os.path.join( bu_paths.kind_dir( 'data' ), 'model' ) )
        os.environ[ENV] = self.root
        self.assertEqual( _load_tool( 'tools/setup/fetch_models.py' ).MODEL_DIR,
                          os.path.join( self.root, 'resources', 'model' ) )

    def test_bar_smoothing_results_go_under_the_root( self ):
        module = _load_tool( 'tools/tuning/run_bar_smoothing_test.py' )
        self.assertEqual( module.RESULTS_DIR, os.path.join( bu_paths.kind_dir( 'state' ), 'bar_smoothing_test_results' ) )
        os.environ[ENV] = self.root
        module = _load_tool( 'tools/tuning/run_bar_smoothing_test.py' )
        self.assertEqual( module.RESULTS_DIR, os.path.join( self.root, 'bar_smoothing_test_results' ) )
        self.assertEqual( module.QUARANTINE_DIR, os.path.join( self.root, 'bar_smoothing_test_quarantine' ) )

    def test_local_config_log_goes_under_the_root( self ):
        module = _load_tool( 'tools/setup/local_config.py' )
        default = module.build_parser().parse_args( [] ).log_file
        self.assertEqual( default, os.path.join( bu_paths.kind_dir( 'state' ), 'logs', 'local_config.log' ) )
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
                # echo/printf text and heredoc help lines are messages, not paths;
                # a betasuite_path call names a 1.x path for the resolver to map.
                if re.match( r'\s*(echo|printf|log_\w+)\b', line ) or 'betasuite_path ' in line:
                    continue
                if any( p.search( line ) for p in self.SHELL_PATTERNS ):
                    found.append( '%s:%d: %s'%( os.path.relpath( path, REPO_ROOT ), number, line.strip() ) )
        self.assertEqual( found, [], 'use tools/lib/data_root.sh instead:\n' + '\n'.join( found ) )

    def test_every_script_stops_when_a_data_folder_is_refused( self ):
        assignment = re.compile( r'^\s*\w+="\$\(betasuite_path ' )
        unguarded = []
        for path in self._sources( '.sh' ):
            for number, line in self._code_lines( path ):
                if assignment.match( line ) and not re.search( r'\|\|\s*(exit 1|die\b)', line ):
                    unguarded.append( '%s:%d'%( os.path.relpath( path, REPO_ROOT ), number ) )
        self.assertEqual( unguarded, [], "add '|| exit 1' so a bad data folder variable stops the script" )

    def test_python_tools( self ):
        found = []
        for path in self._sources( '.py' ):
            for number, line in self._code_lines( path ):
                if 'bu_paths.resolve(' in line:
                    continue
                if any( p.search( line ) for p in self.PYTHON_PATTERNS ):
                    found.append( '%s:%d: %s'%( os.path.relpath( path, REPO_ROOT ), number, line.strip() ) )
        self.assertEqual( found, [], 'use betautils_paths.legacy_path instead:\n' + '\n'.join( found ) )

    def test_the_scan_sees_the_forms_it_guards( self ):
        self.assertTrue( any( p.search( 'RUN_DIR="$(cd "$REPO_ROOT/.." && pwd)/output/x"' ) for p in self.SHELL_PATTERNS ) )
        self.assertTrue( any( p.search( 'for f in output/cache/vid_hashes/*.gz; do' ) for p in self.SHELL_PATTERNS ) )
        self.assertTrue( any( p.search( 'mkdir -p ../output/logs' ) for p in self.SHELL_PATTERNS ) )
        self.assertTrue( any( p.search( "os.path.join( REPO_ROOT, '..', 'output', 'logs' )" ) for p in self.PYTHON_PATTERNS ) )
        self.assertTrue( any( p.search( "os.path.join( os.path.dirname( CORE_DIR ), 'output' )" ) for p in self.PYTHON_PATTERNS ) )


if __name__ == '__main__':
    unittest.main()
