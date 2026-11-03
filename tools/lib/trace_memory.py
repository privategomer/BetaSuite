#!/usr/bin/env python3
"""
trace_memory.py - where a Python script's memory went, with the standard
library only.

    trace_memory.py --out REPORT -- SCRIPT [ARGS...]

The fallback for tools/bench/profile.sh where memray is not installed or
cannot run (native Windows). Runs SCRIPT as __main__ under tracemalloc
and writes REPORT: the traced peak, then the 30 source lines holding the
most memory at (close to) that peak.

tracemalloc has no "snapshot at the peak", so a background thread takes
a snapshot each time traced memory passes the last one by 25%, checking
every second. The report is from the last such snapshot: at most 25%
and a second short of the true peak. Only the allocating line is kept
(one frame), which is what the report groups by. Tracing every
allocation is slow: an allocation-heavy tool runs many times slower
than normal (one that parses JSON, tens of times).

Analogy: memray is a security camera that records every allocation;
this is a guard who photographs the room whenever it looks fuller than
the last photo.
"""

import os
import runpy
import sys
import threading
import time
import tracemalloc

FRAMES = 1
TOP_LINES = 30
GROWTH_FOR_SNAPSHOT = 1.25
CHECK_SECONDS = 1.0


class PeakWatcher( threading.Thread ):

    def __init__( self ):
        super().__init__( daemon=True )
        self.stop = threading.Event()
        self.snapshot = None
        self.snapshot_bytes = 0

    def run( self ):
        while not self.stop.wait( CHECK_SECONDS ):
            self.check()

    def check( self ):
        current, _peak = tracemalloc.get_traced_memory()
        if current > self.snapshot_bytes * GROWTH_FOR_SNAPSHOT:
            self.snapshot = tracemalloc.take_snapshot()
            self.snapshot_bytes = current


def write_report( path, script, peak, watcher, seconds, status ):
    lines = [ "script: %s"%( script, ),
              "exit: %s, %.1f s under tracemalloc"%( status, seconds ),
              "traced peak: %.1f MB (Python allocations only; RSS is higher)"%( peak / 1048576.0, ) ]
    if watcher.snapshot is None:
        lines.append( "no snapshot was taken (the script finished within %.1f s)"%( CHECK_SECONDS, ) )
    else:
        lines.append( "top %d lines by memory held at the last snapshot (%.1f MB traced):"%(
            TOP_LINES, watcher.snapshot_bytes / 1048576.0 ) )
        snapshot = watcher.snapshot.filter_traces( (
            tracemalloc.Filter( False, tracemalloc.__file__ ),
            tracemalloc.Filter( False, __file__ ) ) )
        for stat in snapshot.statistics( 'lineno' )[ :TOP_LINES ]:
            frame = stat.traceback[0]
            lines.append( "  %9.1f MB  %8d blocks  %s:%d"%(
                stat.size / 1048576.0, stat.count, frame.filename, frame.lineno ) )
    with open( path, 'w' ) as handle:
        handle.write( "\n".join( lines ) + "\n" )


def main( argv ):
    if len( argv ) < 4 or argv[0] != '--out' or argv[2] != '--':
        sys.stderr.write( __doc__ )
        return 2
    report_path, script, script_args = argv[1], argv[3], argv[4:]
    sys.argv = [ script ] + script_args
    sys.path.insert( 0, os.path.dirname( os.path.abspath( script ) ) )
    tracemalloc.start( FRAMES )
    watcher = PeakWatcher()
    watcher.start()
    started = time.monotonic()
    status = 0
    try:
        runpy.run_path( script, run_name='__main__' )
    except SystemExit as exit_:
        status = exit_.code if isinstance( exit_.code, int ) else ( 0 if exit_.code is None else 1 )
    except BaseException:
        status = 1
        raise
    finally:
        watcher.stop.set()
        watcher.join()
        watcher.check()
        _current, peak = tracemalloc.get_traced_memory()
        write_report( report_path, script, peak, watcher, time.monotonic() - started, status )
    return status


if __name__ == '__main__':
    sys.exit( main( sys.argv[1:] ) )
