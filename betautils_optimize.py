"""
betautils_optimize.py - "Optimize for this PC": which measurements to
run, from the evidence of past runs, and running them.

  plan()        the stages of tools/bench/tune_machine.py to run and
                whether to try only values near the ones in use, each
                choice with its reason as a code. full=True runs it all
  Optimizer     runs tune_machine as a child process, follows its
                results.json (written after every stage) for progress,
                and reads its proposal.json when it ends

Evidence shortens the run, never the checks: a stage the evidence
cannot settle runs in full, and every proposal tune_machine makes still
needs its own measured gain (above the noise) and, for the batch size,
identical detections.

  inventory   always: it reads the machine and turns the GPU on or off
  detect      narrowed to half, same and double the batch when past runs
              at the settings in use are steady (STEADY_RUNS full runs
              detected at the current model and batch)
  prefetch    narrowed the same way
  render      narrowed to the workers in use and two either side, unless
              memory has been capping the render (then the full spread,
              since the cap is what needs measuring)
  encode      skipped when an earlier optimization on this machine
              already measured the same encoder settings
  scan        skipped: structure_scan_workers is read by the bench tool
              only, never by a run
"""

import json
import os
import subprocess
import sys
import threading
import time

STAGES = ( 'inventory', 'detect', 'prefetch', 'render', 'encode', 'scan' )
STEADY_RUNS = 5

# Reason codes (protocol/v1/commands/perf.optimize.schema.json).
FULL = 'optimize.full'
STEADY = 'optimize.steady_evidence'
NO_EVIDENCE = 'optimize.no_evidence'
MEMORY_CAPPED = 'optimize.memory_capped'
ENCODE_SETTLED = 'optimize.encode_settled'
NOT_USED = 'optimize.not_used_by_runs'
ALWAYS = 'optimize.always'

ENCODE_SETTINGS = ( 'encode_video_codec', 'encode_preset', 'encode_crf' )


def encode_settings( config ):
    return { name: getattr( config, name, None ) for name in ENCODE_SETTINGS }


def plan( evidence, config, optimized_settings=None, full=False, stages=None ):
    """
    { stages: [ { stage, run, narrow, code } ], narrow: [ stage ], skip: [ stage ] }.

    evidence: perf.evidence's answer for this machine. optimized_settings:
    the settings recorded with the last optimization on this machine, or
    None. stages: only these (a nudge's action), the rest skipped.
    """
    steps = []
    detected = sum( band.get( 'detected_runs', 0 ) for band in ( evidence or {} ).get( 'bands', [] ) )
    steady = detected >= STEADY_RUNS
    capped = any( nudge['code'] == 'perf.memory_capped' for nudge in ( evidence or {} ).get( 'nudges', [] ) )
    encode_same = bool( optimized_settings ) and all(
        optimized_settings.get( name ) == value for name, value in encode_settings( config ).items() )
    for stage in STAGES:
        if stage == 'inventory':
            step = ( True, False, ALWAYS )
        elif full:
            step = ( True, False, FULL ) if stage != 'scan' else ( False, False, NOT_USED )
        elif stage == 'scan':
            step = ( False, False, NOT_USED )
        elif stage == 'encode':
            step = ( False, False, ENCODE_SETTLED ) if encode_same else ( True, False, NO_EVIDENCE )
        elif stage == 'render' and capped:
            step = ( True, False, MEMORY_CAPPED )
        elif steady:
            step = ( True, True, STEADY )
        else:
            step = ( True, False, NO_EVIDENCE )
        run, narrow, code = step
        if stages is not None and stage not in stages and stage != 'inventory':
            run, narrow = False, False
        steps.append( { 'stage': stage, 'run': run, 'narrow': narrow, 'code': code } )
    return {
        'stages': steps,
        'narrow': [ s['stage'] for s in steps if s['narrow'] ],   # a stage that does not run is never narrow
        'skip': [ s['stage'] for s in steps if not s['run'] ],
    }


def proposal_rows( proposal, config ):
    """[ { setting, value, current } ] from tune_machine's proposal.json, sorted by setting."""
    import betautils_config_store as bu_store
    rows = []
    for name, entry in sorted( ( proposal or {} ).items() ):
        if not isinstance( entry, dict ) or 'value' not in entry:
            continue
        path = tuple( name.split( '.' ) )
        try:
            current = bu_store._get( { path[0]: getattr( config, path[0], None ) }, path )
        except Exception:
            current = None
        rows.append( { 'setting': name, 'value': entry['value'], 'current': current } )
    return rows


QUEUED = 'queued'
RUNNING = 'running'
FINISHED = 'finished'
FAILED = 'failed'
CANCELLED = 'cancelled'


class Optimizer:
    """
    One optimization at a time, in a child process.

    send( name, data ) gets perf.optimize_progress events: state running
    with each stage as it starts and ends, then finished (with the
    proposal), failed (code optimize.failed, the exit code) or cancelled.
    """

    def __init__( self, send, logger, python=None, script=None, popen=subprocess.Popen, clock=time.time,
                  poll=1.0 ):
        self._send = send
        self._logger = logger
        self._python = python or sys.executable
        self._script = script or os.path.join( os.path.dirname( os.path.abspath( __file__ ) ),
                                               'tools', 'bench', 'tune_machine.py' )
        self._popen = popen
        self._clock = clock
        self._poll = poll
        self._lock = threading.Lock()
        self._proc = None
        self._thread = None
        self._cancelled = False
        self.last = None       # the last run's answer: { at, out_dir, state, proposal, plan }

    @property
    def running( self ):
        with self._lock:
            return self._proc is not None

    def start( self, planned, out_dir, video=None, on_finished=None ):
        with self._lock:
            if self._proc is not None:
                return False
            command = [ self._python, self._script, 'measure', '--out-dir', out_dir, '--console-level', 'warn' ]
            if planned['skip']:
                command += [ '--skip' ] + planned['skip']
            if planned['narrow']:
                command += [ '--narrow' ] + planned['narrow']
            if video:
                command += [ '--video', video ]
            os.makedirs( out_dir, exist_ok=True )
            self._cancelled = False
            self._proc = self._popen( command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL )
            self.last = { 'at': self._clock(), 'out_dir': out_dir, 'state': RUNNING, 'proposal': None,
                          'plan': planned }
            stages = [ s['stage'] for s in planned['stages'] if s['run'] ]
            self._thread = threading.Thread( target=self._follow, args=( self._proc, out_dir, stages, on_finished ),
                                             name='betaoptimize', daemon=True )
            self._thread.start()
        self._send( 'perf.optimize_progress', { 'state': RUNNING, 'stage': None, 'done': 0, 'total': len( stages ) } )
        return True

    def cancel( self ):
        with self._lock:
            proc = self._proc
            self._cancelled = proc is not None
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
        return proc is not None

    def wait( self, timeout=None ):
        thread = self._thread
        if thread is not None:
            thread.join( timeout )

    def _done_stages( self, out_dir ):
        try:
            with open( os.path.join( out_dir, 'results.json' ), 'r', encoding='utf-8' ) as handle:
                return list( ( json.load( handle ).get( 'stage_seconds' ) or {} ).keys() )
        except ( OSError, ValueError ):
            return []

    def _follow( self, proc, out_dir, stages, on_finished ):
        seen = 0
        while True:
            code = proc.poll()
            done = self._done_stages( out_dir )
            if len( done ) > seen:
                seen = len( done )
                self._send( 'perf.optimize_progress', { 'state': RUNNING, 'stage': done[-1], 'done': seen,
                                                        'total': len( stages ) } )
            if code is not None:
                break
            time.sleep( self._poll )
        with self._lock:
            self._proc = None
            cancelled = self._cancelled
        if cancelled:
            self.last['state'] = CANCELLED
            self._send( 'perf.optimize_progress', { 'state': CANCELLED, 'stage': None, 'done': seen,
                                                    'total': len( stages ) } )
            return
        try:
            with open( os.path.join( out_dir, 'proposal.json' ), 'r', encoding='utf-8' ) as handle:
                proposal = json.load( handle )
        except ( OSError, ValueError ):
            proposal = None
        if code != 0 or proposal is None:
            self.last['state'] = FAILED
            if self._logger is not None:
                self._logger.error( "optimize: tune_machine exited %s; its log is in %s"%( code, out_dir ) )
            self._send( 'perf.optimize_progress', { 'state': FAILED, 'stage': None, 'done': seen,
                                                    'total': len( stages ), 'code': 'optimize.failed',
                                                    'params': { 'exit_code': code } } )
            return
        self.last['state'] = FINISHED
        self.last['proposal'] = proposal
        rows = on_finished( proposal ) if on_finished is not None else []
        self._send( 'perf.optimize_progress', { 'state': FINISHED, 'stage': None, 'done': seen,
                                                'total': len( stages ), 'proposal': rows } )
