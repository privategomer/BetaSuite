#!/usr/bin/env python3
"""
step_meter.py - run one command and record its wall time and peak memory.

    step_meter.py --record PATH -- COMMAND [ARGS...]

Runs COMMAND, passes its exit status through (a signal death becomes
128 + the signal number, the shell convention, so a kernel OOM kill
still reads 137), and writes PATH as JSON:

    {"exit": 137, "signal": 9, "wall_seconds": 58.2, "peak_rss_mb": 6123.4}

peak_rss_mb is the largest resident set of the command or any process
it waited for (getrusage RUSAGE_CHILDREN), which is what the kernel's
OOM killer weighs; null on native Windows, which has no getrusage. The
analysis hub wraps every step in this, so a step that was killed says
how much memory it reached instead of only "137".

Standalone on purpose: no project imports, so it measures the step and
nothing of its own beyond the interpreter (about 10 MB).
"""

import json
import os
import signal
import subprocess
import sys
import time


def peak_rss_mb():
    """
    Peak RSS of waited-for children, in MB (Linux reports KB, macOS
    bytes), or None where getrusage does not exist (native Windows).
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage( resource.RUSAGE_CHILDREN ).ru_maxrss
    return peak / ( 1024.0 * 1024.0 ) if sys.platform == 'darwin' else peak / 1024.0


def main( argv ):
    if len( argv ) < 4 or argv[0] != '--record' or argv[2] != '--':
        sys.stderr.write( __doc__ )
        return 2
    record_path, command = argv[1], argv[3:]
    started = time.monotonic()
    try:
        child = subprocess.Popen( command )
    except OSError as err:
        sys.stderr.write( "step_meter: cannot run %s: %s\n"%( command[0], err ) )
        return 127
    try:
        returncode = child.wait()
    except KeyboardInterrupt:
        child.send_signal( signal.SIGINT )
        returncode = child.wait()
    wall = time.monotonic() - started
    died_of = -returncode if returncode < 0 else None
    status = 128 + died_of if died_of else returncode
    peak = peak_rss_mb()
    record = { 'exit': status, 'signal': died_of, 'wall_seconds': round( wall, 3 ),
               'peak_rss_mb': None if peak is None else round( peak, 1 ) }
    tmp = record_path + '.tmp'
    with open( tmp, 'w' ) as handle:
        json.dump( record, handle )
    os.replace( tmp, record_path )
    return status


if __name__ == '__main__':
    sys.exit( main( sys.argv[1:] ) )
