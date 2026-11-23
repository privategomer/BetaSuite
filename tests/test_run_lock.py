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


class TestProbeAndHolder( _LockCase ):

    def test_probe_creates_nothing( self ):
        self.assertIsNone( bu_runlock.probe( self.path ) )
        self.assertFalse( os.path.exists( os.path.dirname( self.path ) ) )

    def test_a_damaged_record_names_what_it_can( self ):
        os.makedirs( os.path.dirname( self.path ) )
        target = bu_runlock.holder_path( self.path )
        for body, expected in (
                ( 'not json', { 'program': None, 'pid': None, 'started_at': None } ),
                ( '[1, 2]', { 'program': None, 'pid': None, 'started_at': None } ),
                ( '{"program": "engine", "pid": "12", "started_at": true}',
                  { 'program': 'engine', 'pid': None, 'started_at': None } ),
                ( '{"program": 3, "pid": 12, "started_at": 5}', { 'program': None, 'pid': 12, 'started_at': 5 } ) ):
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

    def test_betastare_refuses( self ):
        import betastare
        self._held_elsewhere()
        with mock.patch.object( betastare, '_parse_cli_args_and_validate_config' ), \
             mock.patch.object( betastare.bu_log, 'get_logger', return_value=self.logger ), \
             mock.patch.object( betastare.bu_detector, 'get_detector' ) as detector:
            self.assertEqual( betastare.main(), bu_runlock.EXIT_LOCKED )
        detector.assert_not_called()
        self.assertIn( 'engine (pid', self.logger.error.call_args[0][0] )

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
