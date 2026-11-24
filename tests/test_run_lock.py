"""
test_run_lock.py - one run at a time per library.

The lock is the kernel's, so these tests use real processes: a second
process is refused with the holder named, a holder killed outright
leaves the library free, and a child process does not keep its parent's
lock. They run on the Windows CI job too, where the lock is msvcrt's.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import betautils_protocol as bu_proto
import betautils_runlock as bu_runlock


REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

# Takes the lock, says so, and holds it until its stdin closes. With
# CHILD set it first starts a long-lived child of its own and prints its pid.
_HOLDER = r'''
import os, subprocess, sys
import betautils_runlock as lock
child = None
if os.environ.get( 'CHILD' ):
    child = subprocess.Popen( [ sys.executable, '-c', 'import time; time.sleep( 60 )' ] )
held = lock.acquire( 'holder', path=sys.argv[1] )
print( 'ready %d %s'%( os.getpid(), child.pid if child else '' ), flush=True )
sys.stdin.readline()
held.release()
'''

# Tries the lock once and prints what happened as JSON.
_TRY = r'''
import json, sys
import betautils_runlock as lock
try:
    held = lock.acquire( 'second', path=sys.argv[1] )
except lock.RunLocked as locked:
    print( json.dumps( { 'refused': locked.error() } ) )
else:
    print( json.dumps( { 'took': True, 'recovered': held.recovered } ) )
    held.release()
'''


def _env():
    return dict( os.environ, PYTHONPATH=REPO_ROOT )


class _LockCase( unittest.TestCase ):

    def setUp( self ):
        self._tmp = tempfile.TemporaryDirectory( prefix='betasuite-lock-' )
        self.addCleanup( self._tmp.cleanup )
        self.path = os.path.join( self._tmp.name, 'state', 'run.lock' )

    def start_holder( self, child=False ):
        env = _env()
        if child:
            env['CHILD'] = '1'
        proc = subprocess.Popen( [ sys.executable, '-c', _HOLDER, self.path ], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True, env=env )
        self.addCleanup( self._stop, proc )
        line = proc.stdout.readline().split()
        self.assertEqual( line[0], 'ready', line )
        return proc, int( line[1] ), ( int( line[2] ) if len( line ) > 2 else None )

    @staticmethod
    def _stop( proc ):
        if proc.poll() is None:
            proc.kill()
        proc.wait( timeout=30 )
        for stream in ( proc.stdin, proc.stdout ):
            if stream:
                stream.close()

    def try_in_subprocess( self ):
        result = subprocess.run( [ sys.executable, '-c', _TRY, self.path ], capture_output=True, text=True,
                                 env=_env(), timeout=60 )
        self.assertEqual( result.returncode, 0, result.stderr )
        return json.loads( result.stdout.strip().splitlines()[-1] )


class TestAcquireRelease( _LockCase ):

    def test_acquire_writes_the_holder_and_release_removes_only_that( self ):
        held = bu_runlock.acquire( 'betatv', path=self.path, clock=lambda: 1000.5 )
        self.assertTrue( held.held )
        self.assertIsNone( held.recovered )
        self.assertEqual( bu_runlock.read_holder( self.path ),
                          { 'program': 'betatv', 'pid': os.getpid(), 'started_at': 1000.5 } )
        held.release()
        self.assertFalse( held.held )
        self.assertFalse( os.path.exists( bu_runlock.holder_path( self.path ) ) )
        # Never deleted: a process waiting on it would otherwise lock a file nobody else sees.
        self.assertTrue( os.path.exists( self.path ) )
        held.release()

    def test_a_with_block_releases( self ):
        with bu_runlock.acquire( 'betatv', path=self.path ) as held:
            self.assertTrue( held.held )
        self.assertFalse( held.held )
        self.assertIsNone( bu_runlock.probe( self.path ) )
        with bu_runlock.acquire( 'betatv', path=self.path ):
            pass

    def test_a_second_descriptor_in_one_process_is_refused( self ):
        with bu_runlock.acquire( 'first', path=self.path ):
            with self.assertRaises( bu_runlock.RunLocked ) as raised:
                bu_runlock.acquire( 'second', path=self.path )
        self.assertEqual( raised.exception.holder['program'], 'first' )
        self.assertEqual( raised.exception.holder['pid'], os.getpid() )

    def test_release_leaves_another_runs_record_alone( self ):
        held = bu_runlock.acquire( 'betatv', path=self.path )
        with open( bu_runlock.holder_path( self.path ), 'w' ) as handle:
            json.dump( { 'program': 'other', 'pid': os.getpid() + 1, 'started_at': 1 }, handle )
        held.release()
        self.assertEqual( bu_runlock.read_holder( self.path )['program'], 'other' )

    def test_the_lock_descriptor_is_not_inherited( self ):
        with bu_runlock.acquire( 'betatv', path=self.path ) as held:
            self.assertFalse( os.get_inheritable( held._fd ) )


class TestAcrossProcesses( _LockCase ):

    def test_a_second_process_is_refused_with_the_holder_named( self ):
        _proc, pid, _child = self.start_holder()
        outcome = self.try_in_subprocess()
        self.assertEqual( outcome['refused']['code'], 'run.locked' )
        params = outcome['refused']['params']
        self.assertEqual( params['program'], 'holder' )
        self.assertEqual( params['pid'], pid )
        self.assertIsInstance( params['started_at'], float )
        self.assertEqual( bu_runlock.probe( self.path )['pid'], pid )

    def test_the_holder_letting_go_frees_it( self ):
        proc, _pid, _child = self.start_holder()
        proc.stdin.write( '\n' )
        proc.stdin.flush()
        proc.wait( timeout=30 )
        outcome = self.try_in_subprocess()
        self.assertEqual( outcome, { 'took': True, 'recovered': None } )

    def test_a_killed_holder_leaves_the_library_free_and_is_named_as_recovered( self ):
        proc, pid, _child = self.start_holder()
        proc.kill()
        proc.wait( timeout=30 )
        self.assertIsNone( bu_runlock.probe( self.path ) )
        logger = mock.Mock()
        with bu_runlock.acquire( 'betatv', path=self.path, logger=logger ) as held:
            self.assertEqual( held.recovered['program'], 'holder' )
            self.assertEqual( held.recovered['pid'], pid )
        warning = logger.warning.call_args[0][0]
        self.assertIn( 'pid %d'%( pid, ), warning )
        self.assertIn( 'ended without releasing', warning )

    def test_a_child_that_outlives_the_holder_does_not_keep_the_lock( self ):
        proc, _pid, child = self.start_holder( child=True )
        self.addCleanup( self._kill_pid, child )
        proc.kill()
        proc.wait( timeout=30 )
        self.assertTrue( self._alive( child ), 'the child should still be running for this to mean anything' )
        self.assertTrue( self.try_in_subprocess()['took'] )

    @staticmethod
    def _alive( pid ):
        if os.name == 'nt':
            result = subprocess.run( [ 'tasklist', '/FI', 'PID eq %d'%( pid, ) ], capture_output=True, text=True )
            return str( pid ) in result.stdout
        try:
            os.kill( pid, 0 )
        except ProcessLookupError:
            return False
        return True

    @staticmethod
    def _kill_pid( pid ):
        if os.name == 'nt':
            subprocess.run( [ 'taskkill', '/F', '/PID', str( pid ) ], capture_output=True )
            return
        try:
            os.kill( pid, 9 )
        except ProcessLookupError:
            pass


class TestLibraryLocks( _LockCase ):
    """A run takes the state lock, plus one in each media folder that is not its OS default."""

    def setUp( self ):
        super().setUp()
        import betautils_paths as bu_paths
        self.bu_paths = bu_paths
        patcher = mock.patch.dict( os.environ )
        patcher.start()
        self.addCleanup( patcher.stop )
        for name in ( bu_paths.ENV_VAR, ) + tuple( bu_paths.OVERRIDES.values() ):
            os.environ.pop( name, None )
        self.tree = os.path.join( self._tmp.name, 'tree' )
        os.makedirs( os.path.join( self.tree, 'resources' ) )
        os.makedirs( os.path.join( self.tree, 'output' ) )
        import betaconst
        self.state_lock = betaconst.run_lock_path

    def test_the_default_os_layout_locks_only_the_state_folder( self ):
        self.assertEqual( bu_runlock.library_paths(), [ self.state_lock ] )

    def test_media_folders_moved_onto_a_1x_tree_lock_where_that_tree_does( self ):
        os.environ[self.bu_paths.INPUT_ENV_VAR] = os.path.join( self.tree, 'resources' )
        os.environ[self.bu_paths.OUTPUT_ENV_VAR] = os.path.join( self.tree, 'output' )
        os_layout = bu_runlock.library_paths()
        self.assertEqual( os_layout, [ self.state_lock, os.path.join( self.tree, 'resources', 'run.lock' ),
                                       os.path.join( self.tree, 'output', 'run.lock' ) ] )
        for name in self.bu_paths.OVERRIDES.values():
            os.environ.pop( name, None )
        os.environ[self.bu_paths.ENV_VAR] = self.tree
        on_the_tree = bu_runlock.library_paths()
        self.assertIn( os.path.join( self.tree, 'output', 'run.lock' ), on_the_tree )
        self.assertIn( os.path.join( self.tree, 'resources', 'run.lock' ), on_the_tree )

    def test_the_two_layouts_on_one_tree_refuse_each_other( self ):
        os.environ[self.bu_paths.ENV_VAR] = self.tree
        on_the_tree = bu_runlock.library_paths()
        del os.environ[self.bu_paths.ENV_VAR]
        os.environ[self.bu_paths.OUTPUT_ENV_VAR] = os.path.join( self.tree, 'output' )
        os_layout = bu_runlock.library_paths()
        with bu_runlock.acquire_library( 'betatv', paths=on_the_tree ):
            with self.assertRaises( bu_runlock.RunLocked ):
                bu_runlock.acquire_library( 'betatv', paths=os_layout )
        # Refused part way, it gave back what it had taken.
        self.assertIsNone( bu_runlock.probe( self.state_lock ) )
        with bu_runlock.acquire_library( 'betatv', paths=os_layout ) as held:
            self.assertEqual( len( held.locks ), 2 )
        self.assertFalse( held.held )

    def test_refused_part_way_it_gives_back_what_it_took( self ):
        mine, other, shared = ( os.path.join( self.tree, name ) for name in ( 'mine.lock', 'other.lock', 'shared.lock' ) )
        with bu_runlock.acquire_library( 'engine', paths=[ other, shared ] ):
            with self.assertRaises( bu_runlock.RunLocked ):
                bu_runlock.acquire_library( 'betatv', paths=[ mine, shared ] )
            self.assertIsNone( bu_runlock.probe( mine ), "the lock taken before the refusal was released" )

    def test_a_lock_named_twice_is_taken_once( self ):
        # Under a data root the output folder's lock is the tree's own run lock.
        os.environ[self.bu_paths.ENV_VAR] = self.tree
        same = os.path.join( self.tree, 'output', 'run.lock' )
        with mock.patch.object( bu_runlock, 'default_path', return_value=same ):
            paths = bu_runlock.library_paths()
        self.assertEqual( paths.count( same ), 1 )
        self.assertEqual( len( paths ), 2 )

    def test_a_recovered_record_from_any_lock_is_reported( self ):
        paths = [ os.path.join( self.tree, 'a.lock' ), os.path.join( self.tree, 'b.lock' ) ]
        with open( bu_runlock.holder_path( paths[1] ), 'w' ) as handle:
            json.dump( { 'program': 'old', 'pid': 1, 'started_at': 1 }, handle )
        with bu_runlock.acquire_library( 'betatv', paths=paths ) as held:
            self.assertEqual( held.recovered['program'], 'old' )


class TestProbeAndHolder( _LockCase ):

    def test_probe_creates_nothing( self ):
        self.assertIsNone( bu_runlock.probe( self.path ) )
        self.assertFalse( os.path.exists( os.path.dirname( self.path ) ) )

    def test_probe_works_on_a_read_only_lock_file( self ):
        held = bu_runlock.acquire( 'holder', path=self.path )
        held.release()
        os.chmod( self.path, 0o400 )
        self.addCleanup( os.chmod, self.path, 0o600 )
        self.assertIsNone( bu_runlock.probe( self.path ) )

    def test_a_damaged_record_names_what_it_can( self ):
        os.makedirs( os.path.dirname( self.path ) )
        target = bu_runlock.holder_path( self.path )
        for body, expected in (
                ( 'not json', { 'program': None, 'pid': None, 'started_at': None } ),
                ( '[1, 2]', { 'program': None, 'pid': None, 'started_at': None } ),
                ( '{"program": "engine", "pid": "12", "started_at": true}',
                  { 'program': 'engine', 'pid': None, 'started_at': None } ),
                ( '{"program": 3, "pid": 12, "started_at": 5}', { 'program': None, 'pid': 12, 'started_at': 5 } ),
                ( '{"program": "x", "pid": true, "started_at": 5}', { 'program': 'x', 'pid': None, 'started_at': 5 } ) ):
            with open( target, 'w' ) as handle:
                handle.write( body )
            self.assertEqual( bu_runlock.read_holder( self.path ), expected, body )

    def test_a_refusal_with_no_record_still_gives_a_complete_error( self ):
        locked = bu_runlock.RunLocked( self.path, { 'program': None, 'pid': None, 'started_at': None } )
        self.assertEqual( locked.error(), { 'code': 'run.locked',
                                            'params': { 'program': None, 'pid': None, 'started_at': None } } )
        self.assertIn( 'an unknown program (pid unknown, started an unknown time)', locked.message() )

    def test_the_error_is_a_registered_protocol_error( self ):
        self.assertEqual( bu_proto.ERRORS['run.locked'], bu_runlock.HOLDER_FIELDS )
        locked = bu_runlock.RunLocked( self.path, { 'program': 'betatv', 'pid': 7, 'started_at': 0 } )
        self.assertEqual( locked.error()['params'], { 'program': 'betatv', 'pid': 7, 'started_at': 0 } )
        self.assertIn( 'betatv (pid 7, started ', locked.message() )
        self.assertIn( self.path, locked.message() )

    def test_a_refused_run_waits_briefly_for_the_holders_record( self ):
        os.makedirs( os.path.dirname( self.path ) )
        reads = []

        def late( path ):
            reads.append( path )
            return None if len( reads ) < 3 else { 'program': 'late', 'pid': 9, 'started_at': 1 }
        with mock.patch.object( bu_runlock, 'read_holder', late ), \
             mock.patch.object( bu_runlock, '_HOLDER_READ_GAP_SECONDS', 0 ):
            self.assertEqual( bu_runlock._read_holder_of_live_lock( self.path )['program'], 'late' )
        self.assertEqual( len( reads ), 3 )

    def test_a_record_without_a_pid_yet_is_read_again( self ):
        reads = []
        empty = { 'program': None, 'pid': None, 'started_at': None }

        def partial( path ):
            reads.append( path )
            return dict( empty ) if len( reads ) < 2 else { 'program': 'whole', 'pid': 9, 'started_at': 1 }
        with mock.patch.object( bu_runlock, 'read_holder', partial ), \
             mock.patch.object( bu_runlock, '_HOLDER_READ_GAP_SECONDS', 0 ):
            self.assertEqual( bu_runlock._read_holder_of_live_lock( self.path )['program'], 'whole' )


def _engine_packages_installed():
    import importlib.util
    return importlib.util.find_spec( 'cv2' ) is not None


@unittest.skipUnless( _engine_packages_installed(), "betatv and betastare need the engine's packages "
                      "(the Windows CI job installs only the protocol's)" )
class TestTheCommandLineTakesIt( unittest.TestCase ):
    """betatv and betastare refuse with exit 3 and give the lock back however they end."""

    def setUp( self ):
        import betaconst
        self.path = betaconst.run_lock_path
        self.logger = mock.Mock()

    def _held_elsewhere( self ):
        held = bu_runlock.acquire( 'engine', path=self.path )
        self.addCleanup( held.release )
        return held

    def test_betatv_refuses( self ):
        import betatv
        self._held_elsewhere()
        with mock.patch.object( betatv, '_startup', return_value=self.logger ), \
             mock.patch.object( betatv.bu_detector, 'get_detector' ) as detector:
            self.assertEqual( betatv.main(), bu_runlock.EXIT_LOCKED )
        detector.assert_not_called()
        message = self.logger.error.call_args[0][0]
        self.assertIn( 'engine (pid %d'%( os.getpid(), ), message )

    def test_betatv_gives_it_back_when_the_model_fails( self ):
        import betatv
        with mock.patch.object( betatv, '_startup', return_value=self.logger ), \
             mock.patch.object( betatv.bu_detector, 'get_detector', side_effect=RuntimeError( 'no model' ) ):
            with self.assertRaises( RuntimeError ):
                betatv.main()
        self.assertIsNone( bu_runlock.probe( self.path ) )
        self.assertIsNone( bu_runlock.read_holder( self.path ) )

    def test_betatv_gives_it_back_when_the_main_loop_fails( self ):
        import betatv
        with mock.patch.object( betatv, '_startup', return_value=self.logger ), \
             mock.patch.object( betatv, '_log_run_banner' ), \
             mock.patch.object( betatv, '_resolve_active_encode_preset', return_value='fast' ), \
             mock.patch.object( betatv.bu_detector, 'get_detector', side_effect=RuntimeError( 'no model' ) ):
            with self.assertRaises( RuntimeError ):
                betatv.main()
        self.assertIsNone( bu_runlock.probe( self.path ) )
        self.assertIsNone( bu_runlock.read_holder( self.path ) )

    def test_betastare_refuses( self ):
        import betastare
        self._held_elsewhere()
        with mock.patch.object( betastare, '_parse_cli_args_and_validate_config' ), \
             mock.patch.object( betastare.bu_log, 'get_logger', return_value=self.logger ), \
             mock.patch.object( betastare.bu_detector, 'get_detector' ) as detector:
            self.assertEqual( betastare.main(), bu_runlock.EXIT_LOCKED )
        detector.assert_not_called()
        self.assertIn( 'engine (pid', self.logger.error.call_args[0][0] )

    def test_betastare_gives_it_back_when_the_main_loop_fails( self ):
        import betastare
        with mock.patch.object( betastare, '_parse_cli_args_and_validate_config' ), \
             mock.patch.object( betastare.bu_log, 'get_logger', return_value=self.logger ), \
             mock.patch.object( betastare.bu_signals, 'install_handler' ), \
             mock.patch.object( betastare.bu_detector, 'get_detector' ), \
             mock.patch.object( betastare, '_count_total_files', return_value=1 ), \
             mock.patch.object( betastare.os, 'walk', side_effect=RuntimeError( 'disk gone' ) ):
            with self.assertRaises( RuntimeError ):
                betastare.main()
        self.assertIsNone( bu_runlock.probe( self.path ) )
        self.assertIsNone( bu_runlock.read_holder( self.path ) )

    def test_betastare_gives_it_back_when_the_model_fails( self ):
        import betastare
        with mock.patch.object( betastare, '_parse_cli_args_and_validate_config' ), \
             mock.patch.object( betastare.bu_log, 'get_logger', return_value=self.logger ), \
             mock.patch.object( betastare.bu_signals, 'install_handler' ), \
             mock.patch.object( betastare.bu_detector, 'get_detector', side_effect=RuntimeError( 'no model' ) ):
            with self.assertRaises( RuntimeError ):
                betastare.main()
        self.assertIsNone( bu_runlock.probe( self.path ) )
        self.assertIsNone( bu_runlock.read_holder( self.path ) )


if __name__ == '__main__':
    unittest.main()
