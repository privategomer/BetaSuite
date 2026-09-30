#!/usr/bin/env python3
"""
compare_backends_orchestrator.py - runs a REAL, full-length betatv.py
detection pass under every registered detector backend in turn (see
betautils_detector.py's _BACKENDS), then runs the full backend-aware
analysis suite against the resulting caches, and prints one unified
before/after comparison report - the single-command version of "flip
detector_backend['selected'], rerun betatv.py, rerun the analysis tools,
flip it again, rerun everything, then manually compare the two sets of
output".

Why a real full run, not a preview slice: jitter/track-break numbers are
sensitive to continuous footage length (see analyze_jitter.py's own
docstring on track continuity/gap thresholds) - a short preview slice
would understate exactly the track-continuity behavior this comparison
exists to surface. This orchestrator ALWAYS runs betatv.py's real,
current video_path_uncensored library end to end, under both backends,
however long that actually takes - there is no --preview/slice shortcut
here on purpose (explicit choice, see this tool's own design
discussion). If you want a fast look instead, run betatv.py once
yourself with --preview on and use analyze_jitter.py/analyze_track_breaks.py
directly against that single backend's preview cache.

How backend-switching works WITHOUT touching betaconfig.py: each
betatv.py subprocess is launched with the environment variable
BETASUITE_DETECTOR_BACKEND_OVERRIDE set to that pass's backend name -
betautils_detector.selected_backend_name() checks this env var first,
before ever reading betaconfig.detector_backend['selected'] (see that
function's own comment). betaconfig.py itself, on disk, is NEVER
written to by this tool, not even temporarily - so a crash mid-run
can't leave it in a flipped state. The parent orchestrator process's
own environment isn't touched either; the override is passed only to
each subprocess via subprocess.run(..., env=...).

What this produces, in order:
  1. A real betatv.py run under backend A (full video_path_uncensored
     library, full length, stdout+stderr teed to its own log file).
  2. The same under backend B.
  3. analyze_suppression_pairs.py, analyze_jitter.py, and
     analyze_track_breaks.py - each already backend-aware as of this
     same work, so a single invocation of each now reports both
     backends' sections back to back, straight from the caches these
     two real passes just populated.
  4. A short closing summary pointing at where everything landed.

This does NOT run compare_models_perf.py (that measures raw model
inference speed only, independent of real footage - run it directly,
see its own docstring) or analyze_style_flicker.py/analyze_span_merge.py
(not backend-specific concerns this orchestrator is about). Run
run_all_analysis.sh yourself afterward for the full suite including
those, now that real cache exists for both backends.

Usage:
    python3 compare_backends_orchestrator.py
    python3 compare_backends_orchestrator.py --backends retinanet_v2 nudenet_v3
    python3 compare_backends_orchestrator.py --labels exposed_breast exposed_vulva
"""

import argparse
import datetime
import os
import subprocess
import sys

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import betaconfig
import betautils_detector as bu_detector

_BACKEND_OVERRIDE_ENV_VAR = 'BETASUITE_DETECTOR_BACKEND_OVERRIDE'

_HERE = os.path.dirname( os.path.abspath( __file__ ) )
_CORE_DIR = os.path.dirname( os.path.dirname( _HERE ) )  # tools/analysis/../.. - matches run_all_analysis.sh's CORE_DIR


def run_betatv_pass( backend_name, log_dir, extra_env=None ):
    """
    Runs a REAL, full betatv.py pass (the whole current
    video_path_uncensored library, full length - no --preview) as a
    subprocess, with BETASUITE_DETECTOR_BACKEND_OVERRIDE set to
    backend_name so this one subprocess uses that backend regardless of
    betaconfig.detector_backend['selected'] - betaconfig.py itself is
    never touched (see this module's own docstring).

    Args:
        backend_name: Which registered backend (bu_detector._BACKENDS
            key) this pass should run under.
        log_dir: Directory to write this pass's combined stdout+stderr
            log file into (created if needed).
        extra_env: Optional dict of additional env vars to also set for
            this subprocess (merged on top of os.environ and the
            backend override) - not used by default, present so a
            future caller (or a human debugging) can pass through
            something like a custom video_censor_fps override without
            editing this function.

    Returns:
        (return_code, log_path) - return_code is betatv.py's real exit
        code (0 on success); log_path is where its full output was
        written. This does NOT raise on a non-zero exit - the caller
        decides whether a failed pass should abort the whole comparison
        (see main()), matching run_all_analysis.sh's own "one tool's
        failure doesn't necessarily block everything else" philosophy.
    """
    log_path = os.path.join( log_dir, 'betatv_%s.log'%(backend_name) )
    env = dict( os.environ )
    env[ _BACKEND_OVERRIDE_ENV_VAR ] = backend_name
    if extra_env:
        env.update( extra_env )

    print( "=== running a REAL, full betatv.py pass under backend=%s ==="%(backend_name) )
    print( "    (this processes your entire current video_path_uncensored library at full length -" )
    print( "     no preview slice, by design - see this tool's own docstring for why. This can take a while.)" )
    print( "    log: %s"%(log_path) )
    sys.stdout.flush()

    with open( log_path, 'w' ) as log_file:
        proc = subprocess.run(
            [ sys.executable, 'betatv.py' ],
            cwd=_CORE_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    status_word = "OK" if proc.returncode == 0 else "FAILED (exit %d)"%(proc.returncode)
    print( "    betatv.py under backend=%s: %s\n"%(backend_name, status_word) )
    return proc.returncode, log_path


def run_analysis_tool( tool_name, log_dir, extra_args=None ):
    """
    Runs one of the backend-aware analysis tools (tools/analysis/<tool_name>)
    as a subprocess against whatever real cache now exists (from
    run_betatv_pass's runs, plus anything already cached from before),
    tees its combined output to its own log file, AND prints it live -
    same "live view + saved copy without re-running" approach
    run_all_analysis.sh already uses.

    Args:
        tool_name: Filename of the tool under tools/analysis/, e.g.
            'analyze_jitter.py'.
        log_dir: Directory to write this tool's log file into.
        extra_args: Optional list of extra CLI args to pass through
            (e.g. ['--labels', 'exposed_breast']).

    Returns:
        The tool's real exit code (0 on success). Output is teed to
        both stdout and log_dir/<tool_name minus .py>.log; non-fatal to
        the rest of the orchestrator if this returns non-zero (matches
        run_all_analysis.sh's philosophy - see main()).
    """
    log_path = os.path.join( log_dir, '%s.log'%(tool_name[:-3] if tool_name.endswith('.py') else tool_name) )
    args = [ sys.executable, os.path.join( 'tools', 'analysis', tool_name ) ] + ( extra_args or [] )

    print( "############################################################" )
    print( "# %s"%(tool_name) )
    print( "############################################################" )
    sys.stdout.flush()

    with open( log_path, 'w' ) as log_file:
        proc = subprocess.Popen( args, cwd=_CORE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True )
        for line in proc.stdout:
            sys.stdout.write( line )
            log_file.write( line )
        proc.wait()

    sys.stdout.flush()
    print()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--backends', nargs='+', default=None, choices=sorted( bu_detector._BACKENDS.keys() ),
        help="which backends to run and compare (default: every registered backend, in sorted order)" )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="restrict analyze_jitter.py/analyze_track_breaks.py to these labels (passed through; default: everything currently in items_to_censor)" )
    parser.add_argument( '--skip-betatv-runs', action='store_true',
        help="skip the real betatv.py passes and go straight to the analysis suite against whatever cache already exists - useful for re-running just the report after you've already done the real passes once (e.g. while iterating on report formatting), NOT a substitute for actually running both backends for a first-time comparison" )
    args = parser.parse_args()

    backends = args.backends or sorted( bu_detector._BACKENDS.keys() )
    if len( backends ) < 2 and not args.skip_betatv_runs:
        print( "note: only one backend given (%s) - this will still work, but there's nothing to COMPARE it "
               "against. Pass --backends with two or more names (or omit --backends to use every registered one) "
               "for an actual before/after comparison."%(backends[0]) )

    ts = datetime.datetime.now().strftime( '%Y%m%d_%H%M%S' )
    log_dir = os.path.join( _CORE_DIR, '..', 'output', 'logs', 'backend_comparison_runs', ts )
    log_dir = os.path.abspath( log_dir )
    os.makedirs( log_dir, exist_ok=True )

    print( "=== compare_backends_orchestrator.py starting at %s ==="%(datetime.datetime.now().isoformat()) )
    print( "backends: %s"%(", ".join(backends)) )
    print( "logs for this run: %s\n"%(log_dir) )

    betatv_results = {}
    if args.skip_betatv_runs:
        print( "--skip-betatv-runs given: skipping the real betatv.py passes, going straight to the analysis suite "
               "against whatever cache already exists.\n" )
    else:
        for backend_name in backends:
            return_code, log_path = run_betatv_pass( backend_name, log_dir )
            betatv_results[ backend_name ] = ( return_code, log_path )

        failed = [ b for b, (rc, _) in betatv_results.items() if rc != 0 ]
        if failed:
            print( "WARNING: betatv.py failed for backend(s): %s - see their logs above. Continuing to the analysis "
                   "suite anyway; that backend's section may simply show no/incomplete cache below.\n"%(", ".join(failed)) )

    analysis_args = []
    if args.labels:
        analysis_args = [ '--labels' ] + args.labels

    print( "=== running the backend-aware analysis suite against the real cache now on disk ===\n" )
    analysis_results = {}
    for tool_name in ( 'analyze_suppression_pairs.py', 'analyze_jitter.py', 'analyze_track_breaks.py' ):
        analysis_results[ tool_name ] = run_analysis_tool( tool_name, log_dir, analysis_args )

    print( "=== compare_backends_orchestrator.py finished at %s ==="%(datetime.datetime.now().isoformat()) )
    print( "--- summary ---" )
    if not args.skip_betatv_runs:
        for backend_name in backends:
            rc, log_path = betatv_results[ backend_name ]
            print( "  betatv.py (%s): %s  (%s)"%(backend_name, "OK" if rc == 0 else "FAILED (exit %d)"%(rc), log_path) )
    for tool_name, rc in analysis_results.items():
        print( "  %s: %s"%(tool_name, "OK" if rc == 0 else "FAILED (exit %d)"%(rc)) )
    print()
    print( "full logs: %s"%(log_dir) )
    print( "each analysis tool's own printed report above already breaks results out per backend - scroll up (or " )
    print( "read the per-tool .log files) to compare %s section by section for each of analyze_suppression_pairs.py "
           "(detection-quality/confusion-pair signature), analyze_jitter.py (position smoothing/track-continuity "
           "jitter), and analyze_track_breaks.py (why tracks reset, and contention risk from loosening thresholds)."%(
               " vs ".join(backends)) )

    exit_codes = [ rc for rc in analysis_results.values() ]
    if not args.skip_betatv_runs:
        exit_codes += [ rc for rc, _ in betatv_results.values() ]
    sys.exit( 1 if any( code != 0 for code in exit_codes ) else 0 )


if __name__ == '__main__':
    main()
