"""
test_governor.py - the governor's decisions on synthetic signal sets.

What each test prevents:
  - a throttle on an idle machine, or one that acts on a single noisy
    sample, or never lets go
  - a busy computer cutting render workers (a lower priority is free
    when nothing competes, cutting workers is not)
  - memory pressure cutting workers all at once, below one, or never
    giving them back; severe pressure not reaching one worker
  - battery not halving the workers, or low battery not reaching one
  - an unknown signal (None) read as a reason to throttle
  - the override not winning, or winning over the memory guard without
    being asked to; a run-scoped override outliving its run
  - the governor switched off still throttling
  - listeners told about decisions that did not change, or not told
    about ones that did
"""

import types
import unittest

import betautils_governor as bu_gov
from betautils_resources import signals as S


def _codes( decision ):
    return [ reason['code'] for reason in decision.reasons ]


class _Clock:

    def __init__( self ):
        self.now = 1000.0

    def tick( self, seconds ):
        self.now += seconds
        return self.now


class _Base( unittest.TestCase ):

    PLANNED = 8

    def setUp( self ):
        self.clock = _Clock()
        self.gov = bu_gov.Governor( bu_gov.Thresholds( reserve_mb=2048 ) )
        self.changes = []
        self.gov.listeners.append( lambda decision, now: self.changes.append( ( now, decision ) ) )
        self.gov.set_planned( self.PLANNED, self.clock.now )

    def feed( self, seconds, step=1.0, **values ):
        """The same signals once a second for `seconds`; returns the last decision."""
        decision = None
        elapsed = 0.0
        while elapsed < seconds:
            decision = self.gov.update( S( **values ), self.clock.tick( step ) )
            elapsed += step
        return decision


IDLE = dict( memory_pressure=0.0, memory_available_mb=8000, other_cpu_percent=2.0, on_battery=False,
             battery_percent=90, input_idle_seconds=None )


class TestIdle( _Base ):

    def test_idle_machine_runs_full( self ):
        decision = self.feed( 60, **IDLE )
        self.assertFalse( decision.throttled )
        self.assertEqual( _codes( decision ), [ bu_gov.FULL ] )
        self.assertIsNone( decision.worker_cap )
        self.assertEqual( decision.priority, bu_gov.NORMAL )
        self.assertEqual( decision.pace, 0.0 )
        self.assertEqual( self.changes, [] )

    def test_unknown_signals_never_throttle( self ):
        decision = self.feed( 60, **{ name: None for name in IDLE } )
        self.assertFalse( decision.throttled )


class TestBusy( _Base ):

    BUSY = dict( IDLE, other_cpu_percent=80.0 )   # over the default 67

    def test_busy_lowers_priority_but_keeps_workers( self ):
        decision = self.feed( 10, **self.BUSY )
        self.assertEqual( _codes( decision ), [ bu_gov.BUSY ] )
        self.assertEqual( decision.priority, bu_gov.LOW )
        self.assertIsNone( decision.worker_cap )
        self.assertEqual( decision.pace, bu_gov.PACE[bu_gov.BUSY] )
        self.assertEqual( decision.reasons[0]['params'], { 'other_cpu': 80.0 } )

    def test_an_ordinary_desktop_is_not_busy( self ):
        # A desktop at rest can show a fifth of the processor in use;
        # only a clearly loaded machine slows a run.
        self.assertFalse( self.feed( 30, **dict( IDLE, other_cpu_percent=60.0 ) ).throttled )
        self.assertTrue( self.feed( 10, **dict( IDLE, other_cpu_percent=70.0 ) ).throttled )

    def test_a_short_burst_is_ignored( self ):
        decision = self.feed( 3, **self.BUSY )
        self.assertFalse( decision.throttled )
        decision = self.feed( 30, **IDLE )
        self.assertEqual( self.changes, [] )

    def test_takes_its_time_to_let_go( self ):
        self.feed( 10, **self.BUSY )
        decision = self.feed( 10, **IDLE )
        self.assertTrue( decision.throttled )
        decision = self.feed( 10, **IDLE )
        self.assertFalse( decision.throttled )

    def test_two_short_bursts_do_not_add_up( self ):
        self.feed( 3, **self.BUSY )
        self.feed( 2, **IDLE )
        self.assertFalse( self.feed( 2, **self.BUSY ).throttled )

    def test_a_blip_restarts_the_wait_to_let_go( self ):
        self.feed( 10, **self.BUSY )
        self.feed( 10, **IDLE )
        self.feed( 1, **self.BUSY )
        self.assertTrue( self.feed( 10, **IDLE ).throttled )

    def test_a_boolean_is_not_a_load( self ):
        low = bu_gov.Thresholds( { 'busy_cpu_pct': 0.5, 'memory_pressure_pct': 0.5, 'memory_severe_pct': 0.9 } )
        found = bu_gov.conditions( S( other_cpu_percent=True, memory_pressure=True ), low )
        self.assertEqual( found, {} )

    def test_flapping_load_does_not_flap_the_decision( self ):
        self.feed( 10, **self.BUSY )
        for _ in range( 10 ):
            self.feed( 5, **IDLE )
            self.feed( 1, **self.BUSY )
        self.assertEqual( len( self.changes ), 1 )


class TestUserActive( _Base ):

    def test_recent_input_lowers_priority( self ):
        decision = self.feed( 5, **dict( IDLE, input_idle_seconds=3.0 ) )
        self.assertEqual( _codes( decision ), [ bu_gov.USER_ACTIVE ] )
        self.assertEqual( decision.priority, bu_gov.LOW )
        self.assertIsNone( decision.worker_cap )

    def test_away_user_is_idle( self ):
        decision = self.feed( 30, **dict( IDLE, input_idle_seconds=600.0 ) )
        self.assertFalse( decision.throttled )


class TestMemory( _Base ):

    PRESSURE = dict( IDLE, memory_pressure=15.0 )

    def test_steps_down_one_at_a_time_to_one( self ):
        caps = []
        for _ in range( 60 ):
            caps.append( self.feed( 1, **self.PRESSURE ).worker_cap )
        self.assertEqual( caps[:3], [ None, None, 7 ] )   # acts once it has held 2 s
        drops = [ b for a, b in zip( caps, caps[1:] ) if a != b ]
        self.assertEqual( drops, [ 7, 6, 5, 4, 3, 2, 1 ] )
        self.assertEqual( caps[-1], 1 )
        decision = self.gov.decision
        self.assertEqual( _codes( decision ), [ bu_gov.MEMORY ] )
        self.assertEqual( decision.priority, bu_gov.NORMAL )   # memory is not a priority problem
        self.assertFalse( decision.guard )

    def test_climbs_back_after_pressure( self ):
        self.feed( 20, **self.PRESSURE )
        low = self.gov.decision.worker_cap
        self.assertLess( low, self.PLANNED )
        caps = [ self.feed( 1, **IDLE ).worker_cap for _ in range( 120 ) ]
        self.assertIsNone( caps[-1] )
        rising = [ c for c in caps if c is not None ]
        self.assertEqual( rising, sorted( rising ) )
        self.assertEqual( set( rising ), set( range( low, self.PLANNED ) ) )

    def test_one_step_every_few_seconds( self ):
        caps = [ self.feed( 1, **self.PRESSURE ).worker_cap for _ in range( 20 ) ]
        first = caps.index( 7 )
        self.assertEqual( caps[first:first + int( bu_gov.MEMORY_STEP_SECONDS )], [ 7 ] * int( bu_gov.MEMORY_STEP_SECONDS ) )
        self.assertEqual( caps[first + int( bu_gov.MEMORY_STEP_SECONDS )], 6 )

    def test_recovered_means_full_again( self ):
        self.feed( 20, **self.PRESSURE )
        self.feed( 120, **IDLE )
        self.assertEqual( _codes( self.gov.decision ), [ bu_gov.FULL ] )
        self.assertIsNone( self.gov.memory_cap() )

    def test_a_new_render_under_pressure_starts_below_its_plan( self ):
        self.feed( 12, **self.PRESSURE )
        self.assertEqual( self.gov.set_planned( 4, self.clock.now ).worker_cap, 3 )

    def test_severe_counts_as_pressure_whatever_the_thresholds( self ):
        odd = bu_gov.Thresholds( { 'memory_pressure_pct': 30, 'memory_severe_pct': 5 } )
        found = bu_gov.conditions( S( memory_pressure=10.0 ), odd )
        self.assertIn( bu_gov.MEMORY_SEVERE, found )
        self.assertIn( bu_gov.MEMORY, found )

    def test_cap_holds_while_pressure_is_gone_but_not_let_go( self ):
        self.feed( 12, **self.PRESSURE )
        held = self.gov.decision.worker_cap
        caps = { self.feed( 1, **IDLE ).worker_cap for _ in range( int( bu_gov.EXIT_SECONDS[bu_gov.MEMORY] ) - 1 ) }
        self.assertEqual( caps, { held } )

    def test_low_free_memory_is_pressure_without_psi( self ):
        decision = self.feed( 5, **dict( IDLE, memory_pressure=None, memory_available_mb=1500 ) )
        self.assertEqual( _codes( decision ), [ bu_gov.MEMORY ] )
        self.assertEqual( decision.reasons[0]['params'], { 'pressure': None, 'available_mb': 1500 } )

    def test_severe_is_one_worker_at_once( self ):
        decision = self.feed( 1, **dict( IDLE, memory_pressure=55.0 ) )
        self.assertEqual( decision.worker_cap, 1 )
        self.assertTrue( decision.guard )
        self.assertEqual( _codes( decision ), [ bu_gov.MEMORY_SEVERE ] )

    def test_very_low_free_memory_is_severe( self ):
        decision = self.feed( 1, **dict( IDLE, memory_available_mb=900 ) )
        self.assertTrue( decision.guard )

    def test_no_cap_before_a_render_plans( self ):
        self.gov.set_planned( None, self.clock.now )
        decision = self.feed( 10, **self.PRESSURE )
        self.assertEqual( _codes( decision ), [ bu_gov.MEMORY ] )
        self.assertIsNone( decision.worker_cap )

    def test_single_worker_plan_has_nothing_to_cut( self ):
        self.gov.set_planned( 1, self.clock.now )
        self.assertIsNone( self.feed( 30, **self.PRESSURE ).worker_cap )


class TestBattery( _Base ):

    def test_on_battery_halves_workers( self ):
        decision = self.feed( 1, **dict( IDLE, on_battery=True, battery_percent=70 ) )
        self.assertEqual( decision.worker_cap, 4 )
        self.assertEqual( decision.priority, bu_gov.LOW )
        self.assertEqual( decision.pace, bu_gov.PACE[bu_gov.BATTERY] )
        self.assertEqual( _codes( decision ), [ bu_gov.BATTERY ] )

    def test_low_battery_is_one_worker( self ):
        decision = self.feed( 1, **dict( IDLE, on_battery=True, battery_percent=12 ) )
        self.assertEqual( decision.worker_cap, 1 )
        self.assertEqual( _codes( decision ), [ bu_gov.BATTERY_LOW ] )

    def test_plugging_in_lets_go( self ):
        self.feed( 5, **dict( IDLE, on_battery=True ) )
        decision = self.feed( 10, **IDLE )
        self.assertFalse( decision.throttled )

    def test_battery_with_unknown_charge( self ):
        decision = self.feed( 1, **dict( IDLE, on_battery=True, battery_percent=None ) )
        self.assertEqual( decision.worker_cap, 4 )


class TestCombined( _Base ):

    def test_the_lowest_cap_and_every_reason( self ):
        values = dict( IDLE, on_battery=True, battery_percent=60, memory_pressure=50.0, other_cpu_percent=80.0 )
        decision = self.feed( 10, **values )
        self.assertEqual( decision.worker_cap, 1 )
        self.assertEqual( decision.priority, bu_gov.LOW )
        self.assertEqual( _codes( decision ), [ bu_gov.MEMORY_SEVERE, bu_gov.BATTERY, bu_gov.BUSY ] )
        self.assertEqual( decision.pace, bu_gov.PACE[bu_gov.BATTERY] )


class TestOverride( _Base ):

    def test_override_wins_over_every_ordinary_throttle( self ):
        values = dict( IDLE, on_battery=True, battery_percent=10, memory_pressure=15.0, other_cpu_percent=90.0,
                       input_idle_seconds=1.0 )
        self.assertTrue( self.feed( 30, **values ).throttled )
        decision = self.gov.set_override( bu_gov.RUN, self.clock.now )
        self.assertFalse( decision.throttled )
        self.assertEqual( _codes( decision ), [ bu_gov.OVERRIDE ] )
        self.assertEqual( decision.override, ( bu_gov.RUN, True ) )
        self.assertFalse( self.feed( 30, **values ).throttled )

    def test_memory_guard_holds_under_the_ordinary_override( self ):
        severe = dict( IDLE, memory_pressure=70.0 )
        self.feed( 2, **severe )
        decision = self.gov.set_override( bu_gov.SESSION, self.clock.now )
        self.assertEqual( decision.worker_cap, 1 )
        self.assertTrue( decision.guard )
        self.assertEqual( _codes( decision ), [ bu_gov.MEMORY_SEVERE, bu_gov.OVERRIDE ] )
        self.assertEqual( decision.priority, bu_gov.NORMAL )
        self.assertEqual( decision.pace, 0.0 )

    def test_guard_override_releases_the_guard( self ):
        severe = dict( IDLE, memory_pressure=70.0 )
        self.feed( 2, **severe )
        decision = self.gov.set_override( bu_gov.RUN, self.clock.now, memory_guard=False )
        self.assertFalse( decision.throttled )
        self.assertFalse( decision.guard )
        self.assertEqual( decision.reasons[0]['params'], { 'scope': bu_gov.RUN, 'memory_guard': False } )

    def test_ordinary_memory_pressure_yields_to_the_ordinary_override( self ):
        self.feed( 20, **dict( IDLE, memory_pressure=15.0 ) )
        self.assertIsNotNone( self.gov.decision.worker_cap )
        self.assertIsNone( self.gov.set_override( bu_gov.RUN, self.clock.now ).worker_cap )

    def test_run_override_ends_with_the_run( self ):
        self.gov.set_override( bu_gov.RUN, self.clock.now )
        self.gov.end_run( self.clock.now )
        self.assertIsNone( self.gov.override )
        self.gov.set_override( bu_gov.SESSION, self.clock.now )
        self.gov.end_run( self.clock.now )
        self.assertEqual( self.gov.override, ( bu_gov.SESSION, True ) )

    def test_clear_override_throttles_again( self ):
        busy = dict( IDLE, other_cpu_percent=90.0 )
        self.feed( 10, **busy )
        self.gov.set_override( bu_gov.RUN, self.clock.now )
        self.assertEqual( _codes( self.gov.clear_override( self.clock.now ) ), [ bu_gov.BUSY ] )

    def test_bad_scope( self ):
        with self.assertRaises( ValueError ):
            self.gov.set_override( 'forever', self.clock.now )


class TestDisabled( _Base ):

    def test_switched_off_drops_every_throttle_but_the_memory_guard( self ):
        self.gov.set_thresholds( bu_gov.Thresholds( { 'enabled': False } ), self.clock.now )
        decision = self.feed( 30, **dict( IDLE, memory_pressure=15.0, on_battery=True, battery_percent=5,
                                          other_cpu_percent=99.0 ) )
        self.assertFalse( decision.throttled )
        self.assertEqual( _codes( decision ), [ bu_gov.OFF ] )
        # Severe memory pressure: one render worker, nothing else slowed.
        decision = self.feed( 2, **dict( IDLE, memory_pressure=90.0, on_battery=True, battery_percent=5,
                                         other_cpu_percent=99.0 ) )
        self.assertTrue( decision.guard )
        self.assertEqual( decision.worker_cap, 1 )
        self.assertEqual( ( decision.priority, decision.pace ), ( bu_gov.NORMAL, 0.0 ) )
        self.assertEqual( _codes( decision ), [ bu_gov.MEMORY_SEVERE, bu_gov.OFF ] )

    def test_switched_off_and_the_guard_released( self ):
        self.gov.set_thresholds( bu_gov.Thresholds( { 'enabled': False } ), self.clock.now )
        self.gov.set_override( bu_gov.SESSION, self.clock.now, memory_guard=False )
        decision = self.feed( 5, **dict( IDLE, memory_pressure=90.0 ) )
        self.assertFalse( decision.throttled )
        self.assertEqual( _codes( decision ), [ bu_gov.OFF ] )
        # The override still shows, so it can be cleared.
        self.assertEqual( decision.override, ( bu_gov.SESSION, False ) )

    def test_switching_on_mid_run_takes_effect( self ):
        self.gov.set_thresholds( bu_gov.Thresholds( { 'enabled': False } ), self.clock.now )
        busy = dict( IDLE, other_cpu_percent=90.0 )
        self.feed( 10, **busy )
        decision = self.gov.set_thresholds( bu_gov.Thresholds( { 'enabled': True } ), self.clock.now )
        self.assertEqual( _codes( decision ), [ bu_gov.BUSY ] )

    def test_from_config( self ):
        config = types.SimpleNamespace( governor={ 'enabled': False, 'busy_cpu_pct': 50 },
                                        render_memory_reserve_mb=1024 )
        thresholds = bu_gov.Thresholds.from_config( config )
        self.assertFalse( thresholds.enabled )
        self.assertEqual( thresholds.busy_cpu_pct, 50.0 )
        self.assertEqual( thresholds.memory_severe_pct, 40.0 )
        self.assertEqual( thresholds.reserve_mb, 1024.0 )
        self.assertTrue( bu_gov.Thresholds.from_config( types.SimpleNamespace() ).enabled )


class TestThresholdsAreRead( _Base ):

    def test_thresholds_move_the_decision( self ):
        self.gov.set_thresholds( bu_gov.Thresholds( { 'busy_cpu_pct': 70 } ), self.clock.now )
        self.assertFalse( self.feed( 10, **dict( IDLE, other_cpu_percent=60.0 ) ).throttled )
        self.gov.set_thresholds( bu_gov.Thresholds( { 'battery_low_pct': 50 } ), self.clock.now )
        self.assertEqual( self.feed( 2, **dict( IDLE, on_battery=True, battery_percent=40 ) ).worker_cap, 1 )

    def test_user_idle_threshold( self ):
        self.gov.set_thresholds( bu_gov.Thresholds( { 'user_idle_seconds': 5 } ), self.clock.now )
        self.assertFalse( self.feed( 10, **dict( IDLE, input_idle_seconds=10.0 ) ).throttled )

    def test_memory_thresholds( self ):
        self.gov.set_thresholds( bu_gov.Thresholds( { 'memory_pressure_pct': 30, 'memory_severe_pct': 80 } ),
                                 self.clock.now )
        self.assertFalse( self.feed( 10, **dict( IDLE, memory_pressure=20.0 ) ).throttled )
        self.assertFalse( self.feed( 1, **dict( IDLE, memory_pressure=50.0 ) ).guard )


class TestListenersAndHistory( _Base ):

    def test_told_once_per_change( self ):
        busy = dict( IDLE, other_cpu_percent=90.0 )
        self.feed( 30, **busy )
        self.feed( 60, **IDLE )
        self.assertEqual( [ _codes( d ) for _, d in self.changes ], [ [ bu_gov.BUSY ], [ bu_gov.FULL ] ] )
        self.assertEqual( len( self.gov.history ), 2 )

    def test_drifting_parameters_are_not_a_change( self ):
        for cpu in ( 70.0, 71.0, 80.0, 90.0, 75.0, 95.0, 85.0, 76.0 ):
            self.feed( 2, **dict( IDLE, other_cpu_percent=cpu ) )
        self.assertEqual( len( self.changes ), 1 )

    def test_any_lever_is_a_throttle( self ):
        self.assertTrue( bu_gov.Decision( pace=0.1 ).throttled )
        self.assertTrue( bu_gov.Decision( worker_cap=2 ).throttled )
        self.assertTrue( bu_gov.Decision( priority=bu_gov.LOW ).throttled )
        self.assertFalse( bu_gov.Decision().throttled )

    def test_as_dict( self ):
        self.feed( 1, **dict( IDLE, memory_pressure=70.0 ) )
        self.gov.set_override( bu_gov.RUN, self.clock.now )
        self.assertEqual( self.gov.decision.as_dict(), {
            'worker_cap': 1, 'priority': 'normal', 'pace': 0.0, 'guard': True, 'throttled': True,
            'override': { 'scope': 'run', 'memory_guard': True },
            'reasons': [ { 'code': 'governor.memory_severe', 'params': { 'pressure': 70.0, 'available_mb': 8000 } },
                         { 'code': 'governor.override', 'params': { 'scope': 'run', 'memory_guard': True } } ] } )


if __name__ == '__main__':
    unittest.main()
