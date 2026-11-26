"""
test_governor_run.py - the governor in a running job: the render gate,
priority, detection pacing, the per-file account, and the serve
commands (governor.get, override, override_clear, enable).

What each test prevents:
  - a cap that stops running chunks, or lets more start than it allows,
    or a stop that hangs a render waiting at the gate
  - a lowered thread or ffmpeg child left lowered for work started after
    the decision returned to normal (Linux cannot raise it back)
  - detection yields longer than MAX_YIELD_SECONDS or past the file's
    budget, yields with nothing to slow for, or yields not counted
  - throttled time not counted, or counted while running at full speed
  - the on/off switch not taking effect in a running job, or not saved
  - answers and events drifting from protocol/v1's schemas
"""

import os
import subprocess
import sys
import threading
import time
import types
import unittest

import betautils_governor as bu_gov
import betautils_governor_run as bu_gov_run
import betautils_protocol as bu_proto
import betautils_render as bu_render
import betautils_serve_resources as bu_serve_resources
import betautils_signals as bu_signals
from betautils_resources import signals as S

from tests import protocol_schemas as schemas

IDLE = dict( memory_pressure=0.0, memory_available_mb=8000, other_cpu_percent=2.0, on_battery=False )
SEVERE = dict( IDLE, memory_pressure=80.0 )
BUSY = dict( IDLE, other_cpu_percent=90.0 )
BATTERY = dict( IDLE, on_battery=True, battery_percent=60 )


class _Clock:

    def __init__( self ):
        self.now = 100.0

    def __call__( self ):
        return self.now


def _runtime( **kw ):
    clock = _Clock()
    slept = []
    runtime = bu_gov_run.Runtime( clock=clock, wall=lambda: 5000.0 + clock.now, sleep=slept.append, **kw )
    runtime.slept = slept
    runtime.test_clock = clock
    runtime.start_run( types.SimpleNamespace() )
    return runtime


def _hold( runtime, values, seconds=40 ):
    """Feed the same signals once a second for `seconds` of the runtime's clock."""
    for _ in range( int( seconds ) ):
        runtime.test_clock.now += 1.0
        runtime.on_sample( S( **values ) )
    return runtime.decision()


def _nice( pid ):
    with open( '/proc/%d/stat'%( pid, ), 'r', encoding='ascii' ) as handle:
        return int( handle.read().rsplit( ')', 1 )[1].split()[16] )


def _thread_nice():
    return os.getpriority( os.PRIO_PROCESS, threading.get_native_id() )


class TestGate( unittest.TestCase ):

    def test_holds_at_the_limit_and_lets_through_on_release( self ):
        gate = bu_gov_run.Gate( 2, poll=0.01 )
        gate.acquire()
        gate.acquire()
        entered = threading.Event()
        thread = threading.Thread( target=lambda: ( gate.acquire(), entered.set() ) )
        thread.start()
        self.assertFalse( entered.wait( 0.1 ) )
        gate.release()
        self.assertTrue( entered.wait( 2 ) )
        thread.join()
        self.assertEqual( gate.peak, 2 )

    def test_raising_the_limit_lets_waiters_in( self ):
        gate = bu_gov_run.Gate( 1, poll=5 )
        gate.acquire()
        entered = threading.Event()
        thread = threading.Thread( target=lambda: ( gate.acquire(), entered.set() ) )
        thread.start()
        self.assertFalse( entered.wait( 0.1 ) )
        gate.set_limit( 2 )
        self.assertTrue( entered.wait( 2 ) )
        thread.join()

    def test_lowering_the_limit_never_evicts( self ):
        gate = bu_gov_run.Gate( 3 )
        for _ in range( 3 ):
            gate.acquire()
        gate.set_limit( 1 )
        self.assertEqual( gate.running, 3 )

    def test_never_below_one( self ):
        gate = bu_gov_run.Gate( 0 )
        self.assertEqual( gate.limit, 1 )
        gate.set_limit( -3 )
        self.assertEqual( gate.limit, 1 )

    def test_stop_while_waiting( self ):
        gate = bu_gov_run.Gate( 1, poll=0.01 )
        gate.acquire()

        def should_stop():
            raise bu_signals.Interrupted( 'stop', reason=bu_signals.PAUSE )
        with self.assertRaises( bu_signals.Interrupted ):
            gate.acquire( should_stop=should_stop )


class TestDecisionsReachTheGate( unittest.TestCase ):

    def test_cap_follows_the_decision_and_returns( self ):
        runtime = _runtime()
        gate = runtime.render_started( 6 )
        self.assertEqual( gate.limit, 6 )
        _hold( runtime, SEVERE, 2 )
        self.assertEqual( gate.limit, 1 )
        _hold( runtime, IDLE, 120 )
        self.assertEqual( gate.limit, 6 )
        runtime.render_ended()
        self.assertIsNone( runtime.gate )

    def test_override_opens_the_gate_but_the_guard_holds( self ):
        runtime = _runtime()
        gate = runtime.render_started( 4 )
        _hold( runtime, BATTERY, 2 )
        self.assertEqual( gate.limit, 2 )
        runtime.set_override( bu_gov.RUN )
        self.assertEqual( gate.limit, 4 )
        runtime.clear_override()
        _hold( runtime, SEVERE, 2 )
        runtime.set_override( bu_gov.RUN )
        self.assertEqual( gate.limit, 1 )
        runtime.set_override( bu_gov.RUN, memory_guard=False )
        self.assertEqual( gate.limit, 4 )

    def test_switching_off_mid_run_opens_the_gate( self ):
        runtime = _runtime()
        gate = runtime.render_started( 4 )
        _hold( runtime, BATTERY, 2 )
        self.assertEqual( gate.limit, 2 )
        decision = runtime.set_enabled( False )
        self.assertEqual( gate.limit, 4 )
        self.assertEqual( decision.reasons[0]['code'], bu_gov.OFF )
        self.assertFalse( _hold( runtime, SEVERE, 5 ).throttled )
        runtime.set_enabled( True )
        self.assertEqual( gate.limit, 1 )

    def test_switch_outlives_the_run( self ):
        runtime = _runtime()
        runtime.set_enabled( False )
        runtime.end_run()
        runtime.start_run( types.SimpleNamespace( governor={ 'enabled': True } ) )
        self.assertFalse( runtime.governor.thresholds.enabled )


class _Chunks:
    """A fake render_plan: records how many run at once and how each thread and child are prioritised."""

    def __init__( self, on_start=None, hold=0.05 ):
        self.lock = threading.Lock()
        self.running = 0
        self.peak_after = {}
        self.thread_nice = {}
        self.child_nice = {}
        self.on_start = on_start
        self.hold = hold
        self.overlaps = {}

    def __call__( self, plan ):
        with self.lock:
            self.running += 1
            self.overlaps[plan] = self.running
        if self.on_start is not None:
            self.on_start( plan )
        if sys.platform.startswith( 'linux' ):
            self.thread_nice[plan] = _thread_nice()
            child = subprocess.Popen( [ sys.executable, '-c', 'import time; time.sleep(0.2)' ],
                                      **bu_gov_run.child_popen_kwargs() )
            try:
                self.child_nice[plan] = _nice( child.pid )
            finally:
                child.wait()
        time.sleep( self.hold )
        with self.lock:
            self.running -= 1
        return plan


class TestRunChunks( unittest.TestCase ):

    def setUp( self ):
        bu_signals.clear()
        self.addCleanup( bu_signals.clear )

    def test_results_in_plan_order_at_full_width( self ):
        chunks = _Chunks( hold=0.2 )
        runtime = _runtime()
        self.assertEqual( bu_render.run_chunks( list( range( 6 ) ), chunks, 3, runtime=runtime ), list( range( 6 ) ) )
        self.assertEqual( max( chunks.overlaps.values() ), 3 )

    @unittest.skipUnless( sys.platform.startswith( 'linux' ), 'thread niceness is per thread on Linux' )
    def test_workers_and_priority_change_mid_run( self ):
        runtime = _runtime()
        base = _thread_nice()
        first = threading.Barrier( 4, timeout=10 )
        later = threading.Barrier( 2, timeout=3 )
        later_met = []

        def on_start( plan ):
            if plan < 4:
                first.wait()          # all four run at once before the throttle
                if plan == 3:
                    _hold( runtime, dict( SEVERE, other_cpu_percent=90.0 ), 6 )   # severe memory and busy
            if plan == 5:
                _hold( runtime, IDLE, 120 )                                        # all clear
            if plan in ( 6, 7 ):
                try:
                    later.wait()      # met only if two may run at once again
                    later_met.append( plan )
                except threading.BrokenBarrierError:
                    pass
        chunks = _Chunks( on_start=on_start, hold=0.15 )
        bu_render.run_chunks( list( range( 10 ) ), chunks, 4, runtime=runtime )
        self.assertEqual( max( chunks.overlaps[p] for p in range( 4 ) ), 4 )
        self.assertEqual( sorted( later_met ), [ 6, 7 ], "the gate opened again after the all clear" )
        for plan in ( 4, 5 ):
            self.assertEqual( chunks.overlaps[plan], 1, plan )
            self.assertEqual( chunks.thread_nice[plan], max( base, bu_gov_run.LINUX_LOW_NICE ), plan )
            self.assertEqual( chunks.child_nice[plan], max( base, bu_gov_run.LINUX_LOW_NICE ), plan )
        for plan in range( 6, 10 ):
            self.assertEqual( chunks.thread_nice[plan], base, plan )
            self.assertEqual( chunks.child_nice[plan], base, plan )
        for plan in range( 4 ):
            self.assertEqual( chunks.thread_nice[plan], base )
        self.assertEqual( _thread_nice(), base, "the job's own thread is never lowered" )

    def test_a_stop_starts_no_more_chunks_and_is_raised( self ):
        started = []

        def render_plan( plan ):
            started.append( plan )
            if plan == 1:
                bu_signals.request_stop( bu_signals.PAUSE )
                raise bu_signals.Interrupted( 'stop', reason=bu_signals.PAUSE )
            time.sleep( 0.05 )
            return plan
        with self.assertRaises( bu_signals.Interrupted ):
            bu_render.run_chunks( list( range( 8 ) ), render_plan, 1, runtime=_runtime() )
        self.assertEqual( started, [ 0, 1 ] )

    def test_a_chunk_that_stops_ends_the_dispatch_even_without_the_flag( self ):
        started = []

        def render_plan( plan ):
            started.append( plan )
            if plan == 0:
                raise bu_signals.Interrupted( 'stop', reason=bu_signals.CANCEL )
            return plan
        with self.assertRaises( bu_signals.Interrupted ):
            bu_render.run_chunks( list( range( 6 ) ), render_plan, 1, runtime=_runtime() )
        self.assertEqual( started, [ 0 ] )

    def test_render_ended_even_after_a_stop( self ):
        runtime = _runtime()

        def render_plan( plan ):
            raise bu_signals.Interrupted( 'stop', reason=bu_signals.CANCEL )
        with self.assertRaises( bu_signals.Interrupted ):
            bu_render.run_chunks( [ 0 ], render_plan, 2, runtime=runtime )
        self.assertIsNone( runtime.gate )
        self.assertIsNone( runtime.governor.planned )


class TestPacing( unittest.TestCase ):

    def test_no_yield_at_full_speed( self ):
        runtime = _runtime()
        runtime.start_file()
        self.assertEqual( runtime.pace( 0.5 ), 0.0 )
        self.assertEqual( runtime.slept, [] )

    def test_yield_is_short_and_counted( self ):
        runtime = _runtime()
        runtime.start_file()
        _hold( runtime, BATTERY, 2 )
        # Enough batch time first that the file's budget is not the limit.
        for _ in range( 20 ):
            runtime.pace( 0.01 )
        runtime.slept.clear()
        before = runtime.account.paced_seconds
        self.assertEqual( runtime.pace( 1.0 ), bu_gov_run.MAX_YIELD_SECONDS )
        self.assertEqual( runtime.slept, [ bu_gov_run.MAX_YIELD_SECONDS ] )
        self.assertAlmostEqual( runtime.account.paced_seconds - before, bu_gov_run.MAX_YIELD_SECONDS )

    def test_yield_scales_with_the_batch( self ):
        runtime = _runtime()
        runtime.start_file()
        _hold( runtime, BATTERY, 2 )
        runtime.account.batch_seconds = 100.0   # budget not the limit
        self.assertAlmostEqual( runtime.pace( 0.04 ), 0.04 * bu_gov.PACE[bu_gov.BATTERY] )

    def test_file_budget_caps_the_total( self ):
        runtime = _runtime()
        runtime.start_file()
        _hold( runtime, BATTERY, 2 )
        total = sum( runtime.pace( 0.1 ) for _ in range( 500 ) )
        batch_total = 0.1 * 500
        self.assertLessEqual( total, runtime.governor.thresholds.detection_pace_max * batch_total + 1e-9 )
        self.assertGreater( total, 0.5 * runtime.governor.thresholds.detection_pace_max * batch_total )
        stats = runtime.end_file()
        self.assertAlmostEqual( stats['governor_paced_seconds'], round( total, 3 ) )

    def test_budget_follows_the_setting( self ):
        runtime = _runtime()
        runtime.start_run( types.SimpleNamespace( governor={ 'detection_pace_max': 0.0 } ) )
        runtime.start_file()
        _hold( runtime, BATTERY, 2 )
        self.assertEqual( sum( runtime.pace( 0.1 ) for _ in range( 50 ) ), 0.0 )

    def test_default_sleep_wakes_for_a_stop( self ):
        bu_signals.clear()
        self.addCleanup( bu_signals.clear )
        bu_signals.request_stop( bu_signals.PAUSE )
        started = time.monotonic()
        with self.assertRaises( bu_signals.Interrupted ):
            bu_signals.wait( 5.0 )
        self.assertLess( time.monotonic() - started, 1.0 )


class TestAccount( unittest.TestCase ):

    def test_throttled_time_and_reasons( self ):
        runtime = _runtime()
        runtime.start_file()
        _hold( runtime, IDLE, 10 )
        _hold( runtime, BUSY, 20 )
        stats = runtime.file_stats()
        # busy acts on the sixth one-second sample (held 5 s), and is
        # throttled from there to the twentieth: 14 s.
        self.assertAlmostEqual( stats['governor_throttled_seconds'], 14.0 )
        self.assertEqual( stats['governor_reasons'], [ bu_gov.BUSY ] )
        self.assertIsNone( stats['governor_fewest_workers'] )

    def test_fewest_workers( self ):
        runtime = _runtime()
        runtime.start_file()
        runtime.render_started( 6 )
        _hold( runtime, BATTERY, 2 )
        _hold( runtime, IDLE, 60 )
        self.assertEqual( runtime.end_file()['governor_fewest_workers'], 3 )

    def test_a_small_plan_is_not_a_throttle( self ):
        runtime = _runtime()
        runtime.start_file()
        runtime.render_started( 1 )
        _hold( runtime, IDLE, 10 )
        self.assertIsNone( runtime.end_file()['governor_fewest_workers'] )

    def test_nothing_outside_a_file( self ):
        self.assertEqual( bu_gov_run.NEVER.file_stats(), { 'governor_throttled_seconds': 0.0,
                                                            'governor_paced_seconds': 0.0,
                                                            'governor_reasons': [],
                                                            'governor_fewest_workers': None } )


class TestAttached( unittest.TestCase ):

    def test_active_only_while_attached_and_samples_flow( self ):
        sampler = types.SimpleNamespace( listeners=[] )
        sampler.add_listener = sampler.listeners.append
        sampler.remove_listener = sampler.listeners.remove
        used = []
        resources = types.SimpleNamespace( use=lambda who: ( used.append( ( 'use', who ) ), sampler )[1],
                                           release=lambda who: used.append( ( 'release', who ) ) )
        runtime = bu_gov_run.Runtime( resources=resources )
        self.assertIs( bu_gov_run.active(), bu_gov_run.NEVER )
        with bu_gov_run.attached( runtime, types.SimpleNamespace() ):
            self.assertIs( bu_gov_run.active(), runtime )
            self.assertEqual( sampler.listeners, [ runtime.on_sample ] )
        self.assertIs( bu_gov_run.active(), bu_gov_run.NEVER )
        self.assertEqual( used, [ ( 'use', 'job' ), ( 'release', 'job' ) ] )
        self.assertEqual( sampler.listeners, [] )

    def test_none_is_full_speed( self ):
        with bu_gov_run.attached( None, types.SimpleNamespace() ) as runtime:
            self.assertIs( runtime, bu_gov_run.NEVER )
            self.assertFalse( runtime.decision().throttled )

    def test_run_override_ends_with_the_run( self ):
        runtime = _runtime()
        runtime.set_override( bu_gov.RUN )
        runtime.end_run()
        self.assertIsNone( runtime.governor.override )


class TestPriority( unittest.TestCase ):

    @unittest.skipUnless( sys.platform.startswith( 'linux' ), 'Linux niceness' )
    def test_low_lowers_only_this_thread( self ):
        base = _thread_nice()
        seen = {}

        def worker():
            seen['returned'] = bu_gov_run.set_thread_priority( bu_gov.LOW )
            seen['nice'] = _thread_nice()
        thread = threading.Thread( target=worker )
        thread.start()
        thread.join()
        self.assertEqual( seen['returned'], bu_gov.LOW )
        self.assertEqual( seen['nice'], max( base, bu_gov_run.LINUX_LOW_NICE ) )
        self.assertEqual( _thread_nice(), base )

    def test_normal_changes_nothing( self ):
        seen = {}

        def worker():
            seen['returned'] = bu_gov_run.set_thread_priority( bu_gov.NORMAL )
            seen['kwargs'] = bu_gov_run.child_popen_kwargs()
        thread = threading.Thread( target=worker )
        thread.start()
        thread.join()
        self.assertEqual( seen, { 'returned': bu_gov.NORMAL, 'kwargs': {} } )

    def test_windows_child_class( self ):
        seen = {}

        def worker():
            bu_gov_run._local.priority = bu_gov.LOW
            original = bu_gov_run.sys.platform
            bu_gov_run.sys.platform = 'win32'
            try:
                seen['kwargs'] = bu_gov_run.child_popen_kwargs()
            finally:
                bu_gov_run.sys.platform = original
        thread = threading.Thread( target=worker )
        thread.start()
        thread.join()
        self.assertEqual( seen['kwargs'], { 'creationflags': bu_gov_run.WINDOWS_BELOW_NORMAL_CLASS } )


class _Jobs:

    def __init__( self, busy ):
        self.busy = busy
        self.waiting = {}

    def busy_job( self ):
        return 'job-1' if self.busy else None

    def when_idle( self, key, callback ):
        self.waiting[key] = callback


class _Server:

    def __init__( self, busy=False ):
        self.sent = []
        self.send_event = lambda name, data: self.sent.append( ( name, data ) )
        self._logger = types.SimpleNamespace( info=lambda *a, **k: None, warning=lambda *a, **k: None )
        self.governor = None
        self.resources = None
        self.jobs = _Jobs( busy )


class TestCommands( unittest.TestCase ):

    def setUp( self ):
        self.writes = []
        original = bu_serve_resources._write_enabled
        bu_serve_resources._write_enabled = lambda server, on: self.writes.append( on )
        self.addCleanup( setattr, bu_serve_resources, '_write_enabled', original )

    def test_override_and_clear( self ):
        server = _Server()
        answer = bu_serve_resources.governor_override( server, { 'scope': 'session' } )
        self.assertEqual( answer['override'], { 'scope': 'session', 'memory_guard': True } )
        self.assertEqual( answer['reasons'][0]['code'], bu_gov.OVERRIDE )
        self.assertEqual( [ name for name, _ in server.sent ], [ 'governor.change' ] )
        self.assertEqual( len( answer['history'] ), 1 )
        self.assertIsNone( bu_serve_resources.governor_override_clear( server, {} )['override'] )

    def test_guard_override( self ):
        answer = bu_serve_resources.governor_override( _Server(), { 'scope': 'run', 'memory_guard': False } )
        self.assertEqual( answer['override'], { 'scope': 'run', 'memory_guard': False } )

    def test_bad_params( self ):
        for data, field, reason in ( ( {}, 'scope', 'missing' ), ( { 'scope': 'forever' }, 'scope', 'value' ),
                                     ( { 'scope': 'run', 'memory_guard': 1 }, 'memory_guard', 'type' ) ):
            with self.assertRaises( bu_proto.ProtocolError ) as caught:
                bu_serve_resources.governor_override( _Server(), data )
            self.assertEqual( caught.exception.error['params'], { 'field': field, 'reason': reason } )
        with self.assertRaises( bu_proto.ProtocolError ):
            bu_serve_resources.governor_enable( _Server(), { 'on': 'yes' } )

    def test_enable_when_idle_saves_now( self ):
        server = _Server()
        answer = bu_serve_resources.governor_enable( server, { 'on': False } )
        self.assertEqual( answer['saved'], 'now' )
        self.assertFalse( answer['enabled'] )
        self.assertEqual( self.writes, [ False ] )

    def test_enable_while_busy_applies_now_and_saves_after_the_jobs( self ):
        server = _Server( busy=True )
        answer = bu_serve_resources.governor_enable( server, { 'on': False } )
        self.assertEqual( answer['saved'], 'after_jobs' )
        self.assertFalse( server.governor.governor.thresholds.enabled )
        self.assertEqual( self.writes, [] )
        server.jobs.waiting['governor.enabled']()
        self.assertEqual( self.writes, [ False ] )

    def test_get( self ):
        answer = bu_serve_resources.governor_get( _Server(), {} )
        self.assertEqual( answer['reasons'][0]['code'], bu_gov.FULL )
        self.assertEqual( answer['history'], [] )


@schemas.needs_jsonschema
class TestSchemas( unittest.TestCase ):

    def test_change_events( self ):
        runtime = _runtime()
        sent = []
        runtime.send = lambda name, data: sent.append( data )
        runtime.render_started( 4 )
        for values in ( SEVERE, BATTERY, BUSY, IDLE ):
            _hold( runtime, values, 40 )
        runtime.set_override( bu_gov.RUN, memory_guard=False )
        runtime.set_enabled( False )
        self.assertGreater( len( sent ), 4 )
        for data in sent:
            schemas.validate_event_data( 'governor.change', data )

    def test_commands( self ):
        original = bu_serve_resources._write_enabled
        bu_serve_resources._write_enabled = lambda server, on: None
        self.addCleanup( setattr, bu_serve_resources, '_write_enabled', original )
        server = _Server()
        schemas.validate_command_data( 'governor.override', { 'scope': 'run', 'memory_guard': False } )
        schemas.validate_command_data( 'governor.enable', { 'on': True } )
        schemas.validate_command_result( 'governor.override',
                                         bu_serve_resources.governor_override( server, { 'scope': 'run' } ) )
        schemas.validate_command_result( 'governor.get', bu_serve_resources.governor_get( server, {} ) )
        schemas.validate_command_result( 'governor.override_clear',
                                         bu_serve_resources.governor_override_clear( server, {} ) )
        schemas.validate_command_result( 'governor.enable',
                                         bu_serve_resources.governor_enable( server, { 'on': False } ) )


if __name__ == '__main__':
    unittest.main()
