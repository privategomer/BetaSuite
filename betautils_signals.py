"""
betautils_signals.py - Cooperative Ctrl-C handling.

THE PROBLEM THIS SOLVES
    Ctrl-C during a long run used to do almost nothing useful. betatv's
    per-file loop caught BaseException, so an interrupt was reported as
    "Skipping (failed)", logged as a warning, slept a second, and moved
    on to the next file - you had to hit Ctrl-C once per remaining file
    to actually stop. The render loop caught BaseException too and fed
    the interrupt into its retry-with-backoff path.

HOW IT WORKS NOW
    install_handler() replaces SIGINT with one that sets an Event and
    logs once. Long loops call check() - an Event lookup, cheap enough
    per frame - which raises Interrupted at a safe point. Interrupted
    subclasses KeyboardInterrupt, so it passes straight through the
    `except Exception` handlers that give the pipeline its
    one-bad-file-does-not-kill-the-run behaviour, and is caught only at
    the top level.

    A stop is always clean: detection checkpoints and completed render
    chunks are already on disk, so re-running resumes rather than
    restarting.

    A second Ctrl-C restores Python's default handler, so an
    unresponsive run can still be killed immediately.

STOP REASONS
    The same safe points serve the engine's job commands (betautils_job).
    request_stop( reason ) says why: 'interrupt' (Ctrl-C), 'pause',
    'cancel' or 'shutdown' (the engine is ending). Interrupted.reason
    carries it to whoever handles the stop. When several are asked for
    before a safe point is reached, the strongest wins: shutdown and
    interrupt over cancel, cancel over pause.
"""

import signal
import threading


INTERRUPT = 'interrupt'
PAUSE = 'pause'
CANCEL = 'cancel'
SHUTDOWN = 'shutdown'

# Higher wins when two stops are asked for before a safe point.
_STRENGTH = { PAUSE: 1, CANCEL: 2, INTERRUPT: 3, SHUTDOWN: 3 }


class Interrupted( KeyboardInterrupt ):
    """
    Raised at a safe point after SIGINT or a stop request.

    Subclasses KeyboardInterrupt on purpose: it must not be swallowed by
    the `except Exception` blocks that make a single bad file
    non-fatal. `reason` is one of the stop reasons above.
    """

    def __init__( self, message="stopped by user request", reason=INTERRUPT ):
        super().__init__( message )
        self.reason = reason


_interrupt_event = threading.Event()
# Re-entrant: the SIGINT handler takes it on the main thread, which may
# already hold it when the signal arrives.
_reason_lock = threading.RLock()
_reason = None
_previous_handler = None
_installed = False


def install_handler( logger=None ):
    """
    Install the cooperative SIGINT handler.

    Idempotent, and safe to call from any entry point. Only installs
    from the main thread, because that is the only place Python allows
    signal handlers.

    Args:
        logger: Optional logger for the "stopping" message.
    """
    global _previous_handler, _installed
    if _installed:
        return
    if threading.current_thread() is not threading.main_thread():
        return

    import betautils_log as bu_log
    resolved_logger = logger or bu_log.get_logger()

    def handle_sigint( _signum, _frame ):
        if _interrupt_event.is_set():
            # Second Ctrl-C: give the user their immediate exit back.
            signal.signal( signal.SIGINT, _previous_handler or signal.default_int_handler )
            raise KeyboardInterrupt( "interrupted twice - exiting immediately" )
        request_stop( INTERRUPT )
        resolved_logger.warning(
            "interrupt received - finishing the current step and stopping. Detection "
            "checkpoints and completed render chunks are already on disk, so re-running "
            "resumes from here. Press Ctrl-C again to exit immediately." )

    _previous_handler = signal.getsignal( signal.SIGINT )
    signal.signal( signal.SIGINT, handle_sigint )
    _installed = True


def restore_handler():
    """Put the previous SIGINT handler back. Mostly for tests."""
    global _previous_handler, _installed
    if not _installed:
        return
    if threading.current_thread() is threading.main_thread():
        signal.signal( signal.SIGINT, _previous_handler or signal.default_int_handler )
    _previous_handler = None
    _installed = False


def request_stop( reason ):
    """
    Ask the running work to stop at its next safe point, for `reason`.
    A weaker reason never replaces a stronger one already pending.
    """
    global _reason
    if reason not in _STRENGTH:
        raise ValueError( "unknown stop reason %r"%( reason, ) )
    with _reason_lock:
        if _reason is None or _STRENGTH[reason] > _STRENGTH[_reason]:
            _reason = reason
        _interrupt_event.set()


def request_interrupt():
    """Set the interrupt flag without a signal. For tests and tooling."""
    request_stop( INTERRUPT )


def clear():
    """Clear any pending stop: after one has been handled, and in tests."""
    global _reason
    with _reason_lock:
        _reason = None
        _interrupt_event.clear()


def take( reason=None ):
    """
    Clear the pending stop and return the stronger of it and `reason`
    (one already raised as Interrupted), in one step under the lock, so a
    stop asked for while another is being handled is never wiped unseen.
    """
    global _reason
    with _reason_lock:
        pending = _reason if _interrupt_event.is_set() else None
        _reason = None
        _interrupt_event.clear()
    if pending is None:
        return reason
    if reason is None or _STRENGTH[pending] > _STRENGTH[reason]:
        return pending
    return reason


def stronger( first, second ):
    """The stronger of two stop reasons (either may be None)."""
    if first is None:
        return second
    if second is None or _STRENGTH[first] >= _STRENGTH[second]:
        return first
    return second


def interrupted():
    """Whether a stop has been requested."""
    return _interrupt_event.is_set()


def stop_reason():
    """The pending stop's reason, or None."""
    with _reason_lock:
        return _reason if _interrupt_event.is_set() else None


def wait( seconds ):
    """
    Sleep up to `seconds`, waking at once for a stop, then check(). For
    the governor's short detection yields: a pause or cancel is never
    held up by one.
    """
    if seconds > 0:
        _interrupt_event.wait( seconds )
    check()


def check():
    """
    Raise Interrupted if a stop has been requested.

    Call this from any loop that can run for more than a second or two.
    The cost is one Event lookup, so per-frame is fine.
    """
    if _interrupt_event.is_set():
        raise Interrupted( "stopped by user request", reason=stop_reason() or INTERRUPT )
