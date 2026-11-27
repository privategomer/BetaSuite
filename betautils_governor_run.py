"""
betautils_governor_run.py - the governor in a running job: what the
render and detection code ask, and the levers that apply a decision.

  Runtime          one per engine (serve) or per command-line run. Holds
                   the Governor, feeds it resource samples, tells
                   listeners (governor.change) when the decision changes,
                   and keeps the per-file account for the stats row
  attached()       installs a Runtime as the active one around a job
  active()         the Runtime in use, or one that never throttles
  Gate             how many render chunks may run at once; the render
                   waits on it before starting each chunk
  set_thread_priority()   lower the calling thread (a fresh render-chunk
                   thread) and so the ffmpeg it starts
  child_popen_kwargs()    what subprocess.Popen needs for the calling
                   thread's priority on Windows

Levers change when and how fast work happens, never what is computed:
chunk boundaries, encoder threads and frames are the render's own.

Priority. Linux lets an unprivileged process lower a thread's priority
but never raise it back, so nothing that lives across decisions is ever
lowered: each render chunk runs on a new thread, lowered at its start
when the decision says so, and the ffmpeg it starts inherits that. A
chunk started after the decision returns to normal runs at normal
priority. The detection loop runs on the job's own thread, which would
stay lowered for good, so for detection a low priority becomes pacing: a
short yield after each batch.

Pacing is capped twice, since wall-clock time comes first: each yield
is at most MAX_YIELD_SECONDS, and all of a file's yields together at
most governor.detection_pace_max of the file's detection time so far.
The time is counted (paced_seconds in the file's account and stats row).
"""

import contextlib
import os
import sys
import threading
import time

import betautils_governor as bu_gov

MAX_YIELD_SECONDS = 0.05
LINUX_LOW_NICE = 10
WINDOWS_BELOW_NORMAL_CLASS = 0x00004000
WINDOWS_THREAD_BELOW_NORMAL = -1

_local = threading.local()


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

def set_thread_priority( priority ):
    """
    Lower the calling thread when priority is LOW. Call it only on a
    thread that ends with its piece of work (a render chunk): on Linux a
    lowered thread can never be raised again. NORMAL leaves the thread
    as it started. Returns the priority now in force on the thread.
    """
    _local.priority = priority
    if priority != bu_gov.LOW:
        return bu_gov.NORMAL
    try:
        if sys.platform.startswith( 'linux' ):
            tid = threading.get_native_id()
            os.setpriority( os.PRIO_PROCESS, tid, max( os.getpriority( os.PRIO_PROCESS, tid ), LINUX_LOW_NICE ) )
        elif sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority( kernel32.GetCurrentThread(), WINDOWS_THREAD_BELOW_NORMAL )
        elif hasattr( os, 'setpriority' ):
            return bu_gov.NORMAL   # elsewhere niceness is per process; leave it
    except OSError:
        return bu_gov.NORMAL
    return bu_gov.LOW


def thread_priority():
    """The priority set_thread_priority gave this thread (NORMAL when never called)."""
    return getattr( _local, 'priority', bu_gov.NORMAL )


def child_popen_kwargs():
    """subprocess.Popen arguments for a child of this thread: Windows needs its class; Linux inherits."""
    if sys.platform == 'win32' and thread_priority() == bu_gov.LOW:
        return { 'creationflags': WINDOWS_BELOW_NORMAL_CLASS }
    return {}


# ---------------------------------------------------------------------------
# The render gate
# ---------------------------------------------------------------------------

class Gate:
    """
    At most `limit` holders at once. acquire() waits for room, checking
    should_stop() (which raises to abandon the wait) every `poll` seconds.
    set_limit() takes effect for the next acquire: holders that already
    run are never stopped.
    """

    def __init__( self, limit, poll=0.2 ):
        self._cond = threading.Condition()
        self._limit = max( 1, int( limit ) )
        self.running = 0
        self.peak = 0
        self.poll = poll

    @property
    def limit( self ):
        return self._limit

    def set_limit( self, limit ):
        with self._cond:
            self._limit = max( 1, int( limit ) )
            self._cond.notify_all()

    def acquire( self, should_stop=None ):
        with self._cond:
            while self.running >= self._limit:
                if should_stop is not None:
                    should_stop()
                self._cond.wait( self.poll )
            self.running += 1
            self.peak = max( self.peak, self.running )

    def release( self ):
        with self._cond:
            self.running -= 1
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------

class FileAccount:
    """What the governor did to one file, for its stats row."""

    def __init__( self ):
        self.throttled_seconds = 0.0
        self.paced_seconds = 0.0
        self.batch_seconds = 0.0
        self.reasons = []
        self.fewest_workers = None

    def note( self, decision ):
        for reason in decision.reasons:
            code = reason['code']
            if code in bu_gov.THROTTLES and code not in self.reasons:
                self.reasons.append( code )

    def stats( self ):
        return { 'governor_throttled_seconds': round( self.throttled_seconds, 2 ),
                 'governor_paced_seconds': round( self.paced_seconds, 3 ),
                 'governor_reasons': list( self.reasons ),
                 'governor_fewest_workers': self.fewest_workers }


class Runtime:
    """
    Thread-safe: samples arrive on the sampler's thread, the render asks
    from its own, commands come from the serve loop.

    Args:
        send: send( name, data ) for governor.change, or None.
        resources: what supplies samples while a run is on, with
            use( who ) returning a sampler (add_listener, remove_listener)
            and release( who ); None for no samples.
        clock: monotonic seconds; wall: time.time for event stamps.
        sleep: sleep( seconds ) for pacing; default waits on the stop
            flag, so a stop is never delayed by a yield.
    """

    def __init__( self, send=None, resources=None, logger=None, clock=time.monotonic, wall=time.time,
                  sleep=None ):
        import betautils_signals as bu_signals
        self.lock = threading.RLock()
        self.governor = bu_gov.Governor()
        self.governor.listeners.append( self._on_change )
        self.send = send
        self.resources = resources
        self.logger = logger
        self.clock = clock
        self.wall = wall
        self.sleep = sleep or bu_signals.wait
        self.enabled = None             # the live on/off switch; None follows the setting
        self.gate = None
        self.start_cap = None           # the render's own memory cap at its start, or None
        self.account = None
        self._last_accrual = None
        self._sampler = None
        self.active = False             # a run is attached (start_run .. end_run)

    # --- run lifecycle ----------------------------------------------------

    def start_run( self, config ):
        with self.lock:
            thresholds = bu_gov.Thresholds.from_config( config )
            if self.enabled is not None:
                thresholds.enabled = self.enabled
            self.governor.set_thresholds( thresholds, self.clock() )
            self._apply()
            self.active = True
        if self.resources is not None:
            self._sampler = self.resources.use( 'job' )
            self._sampler.add_listener( self.on_sample )

    def end_run( self ):
        if self._sampler is not None:
            self._sampler.remove_listener( self.on_sample )
            self._sampler = None
            self.resources.release( 'job' )
        with self.lock:
            self.gate = None
            self.start_cap = None
            self.active = False
            self.governor.end_run( self.clock() )

    def set_enabled( self, enabled ):
        """The on/off switch, live: takes effect at once, for this and later runs."""
        with self.lock:
            self.enabled = bool( enabled )
            thresholds = self.governor.thresholds
            thresholds.enabled = self.enabled
            decision = self.governor.set_thresholds( thresholds, self.clock() )
            self._apply()
            return decision

    # --- inputs -----------------------------------------------------------

    def observe( self, snapshot, config=None ):
        """
        A sample while no run is attached (the Performance page watching):
        the governor follows the machine, so the page can say what a run
        would meet now, with the settings as they are. During a run the
        run's own listener (on_sample) feeds it instead.
        """
        with self.lock:
            if self.active:
                return
            if config is None:
                import betaconfig as config
            thresholds = bu_gov.Thresholds.from_config( config )
            if self.enabled is not None:
                thresholds.enabled = self.enabled
            self.governor.thresholds = thresholds
            self.governor.update( snapshot, self.clock() )

    def on_sample( self, snapshot ):
        with self.lock:
            now = self.clock()
            self._accrue( now )
            self.governor.update( snapshot, now )
            self._apply()

    def set_override( self, scope, memory_guard=True ):
        with self.lock:
            decision = self.governor.set_override( scope, self.clock(), memory_guard=memory_guard )
            self._apply()
            return decision

    def clear_override( self ):
        with self.lock:
            decision = self.governor.clear_override( self.clock() )
            self._apply()
            return decision

    def decision( self ):
        with self.lock:
            return self.governor.decision

    # --- the render -------------------------------------------------------

    def render_started( self, planned, wanted=None, available_mb=None ):
        """
        A render planned `planned` workers: the gate it waits on. wanted:
        what the processor allowed, when the render's own memory check
        planned fewer (available_mb: the free memory it saw). That cap is
        the render's, fixed for its length, so it is reported
        (memory_start_cap) and never lifted here.
        """
        with self.lock:
            self.gate = Gate( planned )
            self.start_cap = ( { 'workers': planned, 'wanted': wanted, 'available_mb': available_mb }
                               if wanted is not None and wanted > planned else None )
            before = self.governor.decision.key()
            self.governor.set_planned( planned, self.clock() )
            self._apply()
            if self.start_cap is not None and self.governor.decision.key() == before:
                self._send_state()
            return self.gate

    def render_ended( self ):
        with self.lock:
            self.gate = None
            capped, self.start_cap = self.start_cap, None
            before = self.governor.decision.key()
            self.governor.set_planned( None, self.clock() )
            if capped is not None and self.governor.decision.key() == before:
                self._send_state()

    def _apply( self ):
        decision = self.governor.decision
        if self.gate is not None:
            planned = self.governor.planned or self.gate.limit
            cap = decision.worker_cap if decision.worker_cap is not None else planned
            self.gate.set_limit( min( planned, cap ) )
            # Only a cap the governor set counts: a render that planned
            # one worker was not slowed.
            if self.account is not None and decision.worker_cap is not None:
                limit = self.gate.limit
                if self.account.fewest_workers is None or limit < self.account.fewest_workers:
                    self.account.fewest_workers = limit
        if self.account is not None:
            self.account.note( decision )

    # --- detection --------------------------------------------------------

    def pace( self, batch_seconds ):
        """
        After a detection batch that took batch_seconds: yield as the
        decision says, within both caps. Returns the seconds yielded.
        """
        with self.lock:
            decision = self.governor.decision
            account = self.account
            if account is not None:
                account.batch_seconds += max( 0.0, batch_seconds )
            if batch_seconds <= 0:
                return 0.0
            wanted = min( decision.pace * batch_seconds, MAX_YIELD_SECONDS )
            if account is not None:
                budget = self.governor.thresholds.detection_pace_max * account.batch_seconds - account.paced_seconds
                wanted = min( wanted, max( 0.0, budget ) )
            if wanted <= 0:
                return 0.0
            if account is not None:
                account.paced_seconds += wanted
        self.sleep( wanted )
        return wanted

    # --- per-file account -------------------------------------------------

    def start_file( self ):
        with self.lock:
            self.account = FileAccount()
            self._last_accrual = self.clock()
            self.account.note( self.governor.decision )

    def end_file( self ):
        """The file's governor fields for its stats row."""
        with self.lock:
            if self.account is None:
                return FileAccount().stats()
            self._accrue( self.clock() )
            stats = self.account.stats()
            self.account = None
            return stats

    def file_stats( self ):
        with self.lock:
            if self.account is None:
                return FileAccount().stats()
            self._accrue( self.clock() )
            return self.account.stats()

    def _accrue( self, now ):
        if self.account is not None and self._last_accrual is not None and self.governor.decision.throttled:
            self.account.throttled_seconds += max( 0.0, now - self._last_accrual )
        self._last_accrual = now

    # --- events -----------------------------------------------------------

    def _send_state( self ):
        """The state changed with no new decision (a render capped at its start): say so."""
        if self.send is not None:
            self.send( 'governor.change', self.state_data() )

    def _on_change( self, decision, _now ):
        if self.logger is not None:
            self.logger.info( "governor: %s"%( ', '.join(
                '%s %s'%( r['code'], r['params'] ) if r['params'] else r['code'] for r in decision.reasons ), ) )
        if self.send is not None:
            self.send( 'governor.change', self.state_data( decision ) )

    def state_data( self, decision=None ):
        decision = decision or self.governor.decision
        data = decision.as_dict()
        data['planned'] = self.governor.planned
        data['running'] = self.gate.running if self.gate is not None else None
        data['enabled'] = self.governor.thresholds.enabled
        data['active'] = self.active
        data['memory_start_cap'] = self.start_cap
        data['time'] = self.wall()
        return data


class _Never( Runtime ):
    """The runtime when no governor is attached: full speed, no samples, no events."""

    def __init__( self ):
        super().__init__( sleep=lambda _s: None )

    def start_run( self, config ):
        pass

    def end_run( self ):
        pass


NEVER = _Never()
_active = NEVER


def active():
    return _active


@contextlib.contextmanager
def attached( runtime, config ):
    """runtime is the active one for the duration, its run started and ended around it."""
    global _active
    if runtime is None:
        yield NEVER
        return
    previous = _active
    runtime.start_run( config )
    _active = runtime
    try:
        yield runtime
    finally:
        try:
            runtime.end_run()
        finally:
            _active = previous
