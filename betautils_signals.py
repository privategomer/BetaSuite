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
"""

import signal
import threading


class Interrupted( KeyboardInterrupt ):
    """
    Raised at a safe point after SIGINT.

    Subclasses KeyboardInterrupt on purpose: it must not be swallowed by
    the `except Exception` blocks that make a single bad file
    non-fatal.
    """


_interrupt_event = threading.Event()
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
        _interrupt_event.set()
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


def request_interrupt():
    """Set the interrupt flag without a signal. For tests and tooling."""
    _interrupt_event.set()


def clear():
    """Clear the interrupt flag. Test-support only."""
    _interrupt_event.clear()


def interrupted():
    """Whether a stop has been requested."""
    return _interrupt_event.is_set()


def check():
    """
    Raise Interrupted if a stop has been requested.

    Call this from any loop that can run for more than a second or two.
    The cost is one Event lookup, so per-frame is fine.
    """
    if _interrupt_event.is_set():
        raise Interrupted( "stopped by user request" )
