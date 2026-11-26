"""
betautils_governor.py - how hard a run may work right now, and why.

The governor reads resource signals (betautils_resources) and decides
how many render workers may run, at what priority, and how much the
detection loop yields. Wall-clock speed comes first, so:

  - the cheapest throttle is tried first: a lower priority only matters
    when another program wants the CPU, so it costs nothing on an idle
    machine
  - workers are cut only for memory and battery
  - every decision carries its reasons as codes with parameters
  - the user can always say "run at full speed anyway" (an override)

  Thresholds     the settings the policy reads (betaconfig.governor)
  Decision       worker cap, priority, detection pace, reasons
  AutomaticPolicy  the policy that ships; manual modes later are more
                 policies behind the same decide()
  Governor       holds the policy, the hysteresis state, the override
                 and the history, and tells its listeners when the
                 decision changes

Automatic policy, per signal (each needs its condition to hold for a
short time before it acts, and to be gone for longer before it lets go):

  other programs busy, or     priority low, detection paced lightly;
  the user active (Windows)   workers unchanged
  memory pressure             render workers stepped down one at a time,
                              to 1 at the least
  severe memory pressure      one render worker (the memory guard)
  on battery                  half the workers, priority low, detection
                              paced
  battery low                 one worker

The override undoes every throttle but the memory guard: running out of
memory gets the run killed by the operating system, which loses more
time than any throttle. A second override, memory_guard: False, turns
the guard off too, and the UI warns before it is used.

Pure: no threads, no clock of its own (every call is given the time),
no I/O. The levers that apply a decision are in the render and detection
code; this module only decides.
"""

import collections

NORMAL = 'normal'
LOW = 'low'

# Reason codes (protocol/v1/events/governor.change.schema.json).
FULL = 'governor.full'
OFF = 'governor.off'
OVERRIDE = 'governor.override'
MEMORY = 'governor.memory'
MEMORY_SEVERE = 'governor.memory_severe'
BUSY = 'governor.busy'
USER_ACTIVE = 'governor.user_active'
BATTERY = 'governor.battery'
BATTERY_LOW = 'governor.battery_low'

REASONS = ( FULL, OFF, OVERRIDE, MEMORY, MEMORY_SEVERE, BUSY, USER_ACTIVE, BATTERY, BATTERY_LOW )
THROTTLES = ( MEMORY, MEMORY_SEVERE, BUSY, USER_ACTIVE, BATTERY, BATTERY_LOW )

RUN = 'run'
SESSION = 'session'

# How long a condition must hold before it acts, and be gone before it
# lets go, in seconds. Memory acts fast: an out-of-memory kill is the
# costliest outcome. Letting go is slower so a decision does not flap.
ENTER_SECONDS = { MEMORY: 2.0, MEMORY_SEVERE: 0.0, BUSY: 5.0, USER_ACTIVE: 2.0, BATTERY: 0.0,
                  BATTERY_LOW: 0.0 }
EXIT_SECONDS = { MEMORY: 20.0, MEMORY_SEVERE: 10.0, BUSY: 15.0, USER_ACTIVE: 30.0, BATTERY: 5.0,
                 BATTERY_LOW: 5.0 }
# Under memory pressure the worker cap drops by one this often, and
# climbs back by one this often once it has gone.
MEMORY_STEP_SECONDS = 5.0

# Detection pacing per reason: the fraction of each batch's own time
# the loop yields after it. The lever caps each yield and the total.
PACE = { BUSY: 0.05, USER_ACTIVE: 0.10, BATTERY: 0.25, BATTERY_LOW: 0.25 }

HISTORY = 200


class Thresholds:
    """The settings the policy reads, from betaconfig.governor (or defaults)."""

    DEFAULTS = {
        'enabled': True,
        'memory_pressure_pct': 10.0,
        'memory_severe_pct': 40.0,
        'busy_cpu_pct': 67.0,
        'user_idle_seconds': 60.0,
        'battery_low_pct': 20.0,
        'detection_pace_max': 0.10,
    }

    def __init__( self, values=None, reserve_mb=2048 ):
        merged = dict( self.DEFAULTS )
        merged.update( values or {} )
        self.enabled = bool( merged['enabled'] )
        self.memory_pressure_pct = float( merged['memory_pressure_pct'] )
        self.memory_severe_pct = float( merged['memory_severe_pct'] )
        self.busy_cpu_pct = float( merged['busy_cpu_pct'] )
        self.user_idle_seconds = float( merged['user_idle_seconds'] )
        self.battery_low_pct = float( merged['battery_low_pct'] )
        self.detection_pace_max = float( merged['detection_pace_max'] )
        # Free memory under the render's reserve is pressure; under half
        # of it is severe.
        self.reserve_mb = float( reserve_mb or 0 )

    @classmethod
    def from_config( cls, config ):
        return cls( getattr( config, 'governor', None ),
                    reserve_mb=getattr( config, 'render_memory_reserve_mb', 2048 ) )


class Decision:
    """
    worker_cap   most render workers running at once; None for no cap
    priority     NORMAL or LOW for work started from now on
    pace         fraction of a detection batch's time to yield after it
    reasons      [ { code, params } ], the first being the main one
    guard        True when the memory guard is what holds the cap
    override     the override in force ( scope, memory_guard ) or None
    """

    __slots__ = ( 'worker_cap', 'priority', 'pace', 'reasons', 'guard', 'override' )

    def __init__( self, worker_cap=None, priority=NORMAL, pace=0.0, reasons=None, guard=False, override=None ):
        self.worker_cap = worker_cap
        self.priority = priority
        self.pace = pace
        self.reasons = list( reasons or [ { 'code': FULL, 'params': {} } ] )
        self.guard = guard
        self.override = override

    @property
    def throttled( self ):
        return self.worker_cap is not None or self.priority != NORMAL or self.pace > 0

    def key( self ):
        """What a listener needs to know changed: everything but parameter values that drift."""
        return ( self.worker_cap, self.priority, round( self.pace, 3 ), self.guard,
                 tuple( r['code'] for r in self.reasons ), self.override )

    def as_dict( self ):
        override = None
        if self.override is not None:
            override = { 'scope': self.override[0], 'memory_guard': self.override[1] }
        return { 'worker_cap': self.worker_cap, 'priority': self.priority, 'pace': round( self.pace, 3 ),
                 'reasons': [ dict( r ) for r in self.reasons ], 'guard': self.guard,
                 'override': override, 'throttled': self.throttled }

    def __repr__( self ):
        return 'Decision(%r)'%( self.as_dict(), )


def _num( value ):
    return value if isinstance( value, ( int, float ) ) and not isinstance( value, bool ) else None


def conditions( signals, thresholds ):
    """
    { code: params } for every throttle condition the signals show right
    now, before any hysteresis. A signal that is None is not evidence of
    anything and never triggers a condition.
    """
    found = {}
    pressure = _num( signals.get( 'memory_pressure' ) )
    available = _num( signals.get( 'memory_available_mb' ) )
    reserve = thresholds.reserve_mb
    severe = ( ( pressure is not None and pressure >= thresholds.memory_severe_pct ) or
               ( available is not None and reserve > 0 and available < reserve / 2 ) )
    pressured = ( ( pressure is not None and pressure >= thresholds.memory_pressure_pct ) or
                  ( available is not None and reserve > 0 and available < reserve ) )
    memory_params = { 'pressure': pressure, 'available_mb': available }
    if severe:
        found[MEMORY_SEVERE] = memory_params
    if pressured or severe:
        found[MEMORY] = memory_params

    other = _num( signals.get( 'other_cpu_percent' ) )
    if other is not None and other >= thresholds.busy_cpu_pct:
        found[BUSY] = { 'other_cpu': other }
    idle = _num( signals.get( 'input_idle_seconds' ) )
    if idle is not None and idle < thresholds.user_idle_seconds:
        found[USER_ACTIVE] = { 'idle_seconds': idle }

    if signals.get( 'on_battery' ) is True:
        percent = _num( signals.get( 'battery_percent' ) )
        found[BATTERY] = { 'percent': percent }
        if percent is not None and percent < thresholds.battery_low_pct:
            found[BATTERY_LOW] = { 'percent': percent }
    return found


class AutomaticPolicy:
    """
    decide( active, planned, memory_cap ) -> Decision, from the throttle
    conditions that passed hysteresis ({ code: params }), the render's
    planned worker count (None before a render plans one) and the
    memory step cap the governor keeps.
    """

    name = 'automatic'

    def decide( self, active, planned, memory_cap ):
        reasons = []
        caps = []
        priority = NORMAL
        pace = 0.0
        guard = False
        if MEMORY_SEVERE in active:
            caps.append( 1 )
            guard = True
            reasons.append( { 'code': MEMORY_SEVERE, 'params': active[MEMORY_SEVERE] } )
        elif MEMORY in active:
            if memory_cap is not None:
                caps.append( memory_cap )
            reasons.append( { 'code': MEMORY, 'params': active[MEMORY] } )
        if BATTERY_LOW in active:
            caps.append( 1 )
            priority = LOW
            pace = max( pace, PACE[BATTERY_LOW] )
            reasons.append( { 'code': BATTERY_LOW, 'params': active[BATTERY_LOW] } )
        elif BATTERY in active:
            if planned:
                caps.append( max( 1, planned // 2 ) )
            priority = LOW
            pace = max( pace, PACE[BATTERY] )
            reasons.append( { 'code': BATTERY, 'params': active[BATTERY] } )
        for code in ( USER_ACTIVE, BUSY ):
            if code in active:
                priority = LOW
                pace = max( pace, PACE[code] )
                reasons.append( { 'code': code, 'params': active[code] } )
        cap = min( caps ) if caps else None
        if cap is not None and planned is not None and cap >= planned:
            cap = None   # a cap at or above the plan changes nothing
        if not reasons:
            return Decision()
        return Decision( worker_cap=cap, priority=priority, pace=pace, reasons=reasons, guard=guard )


class Governor:
    """
    update( signals, now ) -> Decision. Listeners get ( decision, now )
    each time the decision changes. Thread safety is the caller's: the
    serve wrapper calls it under one lock.
    """

    def __init__( self, thresholds=None, policy=None, history=HISTORY ):
        self.thresholds = thresholds or Thresholds()
        self.policy = policy or AutomaticPolicy()
        self.planned = None
        self.override = None              # ( scope, memory_guard ) or None
        self.listeners = []
        self.history = collections.deque( maxlen=history )
        self._seen = {}                   # code -> when the condition was first seen this spell
        self._gone = {}                   # code -> when an active condition was last seen gone
        self._active = {}                 # code -> params, past hysteresis
        self._memory_cap = None
        self._memory_step_at = None
        self.decision = Decision()
        self._last_signals = {}

    # --- inputs ---------------------------------------------------------

    def set_planned( self, workers, now ):
        """The render planned this many workers (None between renders)."""
        self.planned = workers
        if workers is None:
            self._memory_cap = None
        elif self._memory_cap is not None:
            # A new render under pressure starts below its own plan, as
            # the first one did; while climbing back, at most its plan.
            ceiling = workers - 1 if MEMORY in self._active else workers
            self._memory_cap = max( 1, min( self._memory_cap, ceiling ) )
        return self._decide( now )

    def set_thresholds( self, thresholds, now ):
        self.thresholds = thresholds
        return self.update( self._last_signals, now )

    def set_override( self, scope, now, memory_guard=True ):
        """
        "Run at full speed anyway". scope RUN ends with the run
        (end_run), SESSION with the engine. memory_guard False also turns
        off the memory guard.
        """
        if scope not in ( RUN, SESSION ):
            raise ValueError( scope )
        self.override = ( scope, bool( memory_guard ) )
        return self._decide( now )

    def clear_override( self, now ):
        self.override = None
        return self._decide( now )

    def end_run( self, now ):
        """A run ended: a run-scoped override ends with it."""
        if self.override is not None and self.override[0] == RUN:
            self.override = None
        return self.set_planned( None, now )

    def update( self, signals, now ):
        self._last_signals = dict( signals or {} )
        seen = conditions( self._last_signals, self.thresholds )
        for code in THROTTLES:
            if code in seen:
                self._gone.pop( code, None )
                self._seen.setdefault( code, now )
                if code in self._active or now - self._seen[code] >= ENTER_SECONDS[code]:
                    self._active[code] = seen[code]
            else:
                self._seen.pop( code, None )
                if code in self._active:
                    self._gone.setdefault( code, now )
                    if now - self._gone[code] >= EXIT_SECONDS[code]:
                        del self._active[code]
                        self._gone.pop( code, None )
        self._step_memory( now, MEMORY in seen )
        return self._decide( now )

    # --- the memory step ------------------------------------------------

    def _step_memory( self, now, pressure_now ):
        """
        While memory pressure is seen, the cap drops one worker per step.
        While it is gone but not yet let go, the cap holds. Once let go,
        it climbs one per step.
        """
        planned = self.planned
        if planned is None:
            return
        if MEMORY in self._active and not pressure_now:
            return
        if MEMORY in self._active:
            if self._memory_cap is None:
                self._memory_cap = max( 1, planned - 1 )
                self._memory_step_at = now
            elif now - self._memory_step_at >= MEMORY_STEP_SECONDS and self._memory_cap > 1:
                self._memory_cap -= 1
                self._memory_step_at = now
        elif self._memory_cap is not None:
            if now - self._memory_step_at >= MEMORY_STEP_SECONDS:
                self._memory_cap += 1
                self._memory_step_at = now
                if self._memory_cap >= planned:
                    self._memory_cap = None

    def memory_cap( self ):
        """The stepped cap, also while climbing back after pressure has gone."""
        return self._memory_cap

    # --- the decision ---------------------------------------------------

    def _decide( self, now ):
        if not self.thresholds.enabled:
            decision = Decision( reasons=[ { 'code': OFF, 'params': {} } ] )
        else:
            active = dict( self._active )
            if MEMORY not in active and self._memory_cap is not None:
                # Climbing back after pressure: still say why there is a cap.
                active[MEMORY] = { 'pressure': None, 'available_mb': None }
            decision = self.policy.decide( active, self.planned, self._memory_cap )
            if self.override is not None:
                decision = self._overridden( decision )
        if decision.key() != self.decision.key():
            self.decision = decision
            self.history.append( ( now, decision ) )
            for listener in list( self.listeners ):
                listener( decision, now )
        else:
            self.decision = decision
        return self.decision

    def _overridden( self, decision ):
        scope, memory_guard = self.override
        reasons = [ { 'code': OVERRIDE, 'params': { 'scope': scope, 'memory_guard': memory_guard } } ]
        if decision.guard and memory_guard:
            # The guard stays: one worker until memory recovers.
            guard_reason = [ r for r in decision.reasons if r['code'] == MEMORY_SEVERE ]
            return Decision( worker_cap=decision.worker_cap, reasons=guard_reason + reasons, guard=True,
                             override=self.override )
        return Decision( reasons=reasons, override=self.override )
