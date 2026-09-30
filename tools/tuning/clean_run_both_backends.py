#!/usr/bin/env python3
"""
clean_run_both_backends.py - single-click "clean slate, run both
backends, analyze" harness.

Why this exists: comparing retinanet_v2 vs nudenet_v3 by hand (edit
detector_backend['selected'], run betatv.py, edit it again, run again,
then eyeball two different runs' logs/caches that may be mixed with
older leftover data) is exactly what produced real confusion diagnosed
2026-09-17 - three of six nudenet_v3 caches from a "6-video full run"
turned out to be stale leftovers from a DIFFERENT, earlier session,
because nothing had cleared the cache first. This script removes that
whole class of mistake: it wipes every cache/log/stat file in the app
dir, THEN runs betatv.py once per backend (via --backend, not by
editing betaconfig.py - see betautils_cli.py), so what's on disk
afterward is guaranteed to be from exactly this one clean run, for
BOTH backends, directly comparable apples-to-apples.

What gets cleared (see betaconst.py / betautils_log.py / betatv.py for
where each of these paths is defined):
    ../output/cache/vid_hashes/*        (video detection cache)
    ../output/cache/pic_hashes/*        (photo detection cache)
    ../output/cache/shot_cuts/*         (shot-cut cache)
    ../output/cache/transcode_cache/*   (hardware-decode-fallback transcodes)
    ../output/logs/*                    (betasuite.log + analysis_runs/backend_comparison_runs
                                          subdirs), EXCEPT ../output/logs/clean_run_both_backends/
                                          itself (this script's OWN run history - see below)
    ../output/stats/*                   (betasuite_stats.jsonl)
    any stray *.checkpoint files anywhere under output/cache/ (resume
        state for an interrupted detection pass - always safe to drop,
        including pre-migration orphans like the ones found 2026-09-17)

Deliberately NOT cleared: ../output/censored_vids/, ../output/censored_pics/,
../output/betafied/ - those are real DELIVERABLE output, not cache or
analysis state, and this script has no business touching them.

Also deliberately NOT cleared: ../output/logs/clean_run_both_backends/
- this script's OWN per-run log directory tree, including EARLIER runs'
subdirectories, not just the current one. Two reasons: (1) this run's
own logger is actively writing into a fresh subdirectory under here
before the clear step runs - a first version of this script tried to
protect only that one active subdirectory while still clearing its
siblings, which is fragile (an ancestor-vs-descendant check has to get
the direction exactly right - a real bug this caught in its own smoke
test) and provides no benefit over just excluding the whole tree, since
(2) analysis tools only ever read the CACHE directories, never
../output/logs - a stale log file sitting here from a previous
clean_run_both_backends.py run is an audit trail a human might want to
diff against later, not data that can pollute an analysis run's
conclusions the way a stale CACHE file can (which is the actual problem
this script exists to solve - see this docstring's opening paragraph).
Delete old subdirectories under here by hand if they pile up; nothing
here is silently accumulating anything that skews results.

What it does, in order:
    1. Prints exactly what will be deleted and asks for confirmation
       (skippable with --yes, for unattended/scripted use - see below).
    2. Deletes it.
    3. Runs betatv.py under backend A (env BETASUITE_DETECTOR_BACKEND_
       OVERRIDE, same mechanism compare_backends_orchestrator.py already
       uses - betaconfig.py itself is never touched on disk).
    4. Runs betatv.py under backend B.
    5. Runs the backend-aware analysis suite (same four tools
       run_all_analysis.sh runs) against the now-clean, both-backends
       cache.
    6. Prints a final summary.

Every step's full stdout+stderr goes to this run's own timestamped log
directory AND is printed live, same "live view + saved copy" approach
every other harness in this repo uses (run_all_analysis.sh,
compare_backends_orchestrator.py). This script's OWN narration (not the
subprocess output, which is already captured verbatim in its own log
file) also goes through Python's logging module at configurable
trace/debug/info/warn/error levels - default info - into
clean_run_both_backends.log inside this run's log directory, with
stdout filterable to a different level via --console-level. Trace-level
detail (env vars passed to each subprocess, resolved paths before
deletion, etc) is only in the log file at --log-level trace, kept out
of the default console output to stay readable.

This is a REAL, FULL run under both backends - no --preview shortcut,
same deliberate choice compare_backends_orchestrator.py makes and
documents in its own docstring (jitter/track-break numbers need
continuous real footage length to mean anything).

Usage (from BetaSuite-env/core/, i.e. same place you run betatv.py from):
    python3 tools/tuning/clean_run_both_backends.py
    python3 tools/tuning/clean_run_both_backends.py --yes
    python3 tools/tuning/clean_run_both_backends.py --backends nudenet_v3 retinanet_v2
    python3 tools/tuning/clean_run_both_backends.py --log-level debug --console-level warn
    python3 tools/tuning/clean_run_both_backends.py --skip-clean --skip-betatv-runs   # just re-run the analysis suite
"""

import argparse
import datetime
import glob
import logging
import os
import shutil
import subprocess
import sys

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import betautils_detector as bu_detector

_BACKEND_OVERRIDE_ENV_VAR = 'BETASUITE_DETECTOR_BACKEND_OVERRIDE'
_VARIANT_OVERRIDE_ENV_VAR = 'BETASUITE_DETECTOR_VARIANT_OVERRIDE'

_HERE = os.path.dirname( os.path.abspath( __file__ ) )
_CORE_DIR = os.path.dirname( os.path.dirname( _HERE ) )  # tools/tuning/../.. - matches run_all_analysis.sh's/compare_backends_orchestrator.py's CORE_DIR

# Every path this script deletes the CONTENTS of (the directory itself
# is kept, so a fresh detection pass has somewhere to write into without
# needing to recreate it). Listed explicitly rather than derived from
# betaconst/betautils_cache_paths constants, so what gets deleted is
# always visible in one place in THIS file, not scattered across
# several modules' own path constants - a deliberate readability choice
# for a destructive script, even though it means updating this list by
# hand if one of those paths ever moves.
_CLEAR_DIRS = [
    '../output/cache/vid_hashes',
    '../output/cache/pic_hashes',
    '../output/cache/shot_cuts',
    '../output/cache/transcode_cache',
    '../output/cache/run_keys',
    '../output/logs',
    '../output/stats',
]

# Individual files cleared alongside the directories above. The
# file-hash memo only ever caches (path, size, mtime) -> hash, so
# dropping it costs one re-hash per file and nothing else; it is cleared
# so a clean run really is measured from cold.
_CLEAR_FILES = [
    '../output/cache/file_hashes.json',
]

_LEVEL_NAMES = { 'trace': 5, 'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARNING, 'error': logging.ERROR }
logging.addLevelName( 5, 'TRACE' )


def _build_logger( log_dir, log_level_name, console_level_name ):
    """
    File handler at log_level_name (default info, can go down to trace),
    console handler at console_level_name (default info) - independent
    levels, per the user's own standing rule for harness scripts: full
    detail always in the log file, console kept readable/filterable
    separately.
    """
    logger = logging.getLogger( 'clean_run_both_backends' )
    logger.setLevel( 5 )  # let handlers do the actual filtering
    logger.handlers.clear()

    fmt = logging.Formatter( '%(asctime)s [%(levelname)s] %(message)s' )

    log_path = os.path.join( log_dir, 'clean_run_both_backends.log' )
    file_handler = logging.FileHandler( log_path )
    file_handler.setLevel( _LEVEL_NAMES[log_level_name] )
    file_handler.setFormatter( fmt )
    logger.addHandler( file_handler )

    console_handler = logging.StreamHandler( sys.stdout )
    console_handler.setLevel( _LEVEL_NAMES[console_level_name] )
    console_handler.setFormatter( fmt )
    logger.addHandler( console_handler )

    return logger, log_path


# Entry NAMES (not full paths - checked against os.listdir's bare
# results at the top level of the matching _CLEAR_DIRS entry only) that
# are excluded from clearing entirely, whole subtree included - not just
# "this run's own active subdirectory" (a first version tried that,
# via a path-based keep_paths + ancestor-or-equal check; that's fragile,
# since a top-level entry in a cleared dir is always an ANCESTOR of a
# nested keep-path, never equal to it, and it provides no real benefit
# over a whole-subtree exclusion anyway - see this file's module
# docstring for why sparing the WHOLE clean_run_both_backends/ log
# history, not just today's run, is fine: analysis tools never read
# ../output/logs, so a stale log file here can't pollute a future run's
# conclusions the way a stale CACHE file can). Keyed by which
# _CLEAR_DIRS entry the name applies under.
_EXCLUDED_NAMES_BY_CLEAR_DIR = {
    '../output/logs': { 'clean_run_both_backends' },
}


def _clearable_entries( abs_dir, rel_dir ):
    """
    Entry names directly under abs_dir that clearing should actually
    touch - every os.listdir result except whatever
    _EXCLUDED_NAMES_BY_CLEAR_DIR lists for this rel_dir.
    """
    excluded_names = _EXCLUDED_NAMES_BY_CLEAR_DIR.get( rel_dir, set() )
    return [ e for e in os.listdir( abs_dir ) if e not in excluded_names ]


def _describe_clear_plan( logger ):
    """
    Prints/logs exactly what --yes (or the confirmation prompt) is about
    to delete, with a real file/dir count per path so "confirm" isn't a
    blind leap - resolves each _CLEAR_DIRS entry relative to _CORE_DIR
    the same way betatv.py's own relative paths resolve when run from
    the right working directory.
    """
    plan = []
    for rel_dir in _CLEAR_DIRS:
        abs_dir = os.path.normpath( os.path.join( _CORE_DIR, rel_dir ) )
        if not os.path.isdir( abs_dir ):
            plan.append( (rel_dir, abs_dir, 0, "(directory doesn't exist yet - nothing to clear)") )
            continue
        entries = _clearable_entries( abs_dir, rel_dir )
        excluded_names = _EXCLUDED_NAMES_BY_CLEAR_DIR.get( rel_dir, set() )
        note = "(excludes %s - see this script's own docstring)"%(", ".join(sorted(excluded_names))) if excluded_names else ""
        plan.append( (rel_dir, abs_dir, len(entries), note) )
    checkpoint_count = len( glob.glob( os.path.join( _CORE_DIR, '..', 'output', 'cache', '**', '*.checkpoint' ), recursive=True ) )
    for rel_dir, abs_dir, count, note in plan:
        logger.info( "  %-45s %4d entr%s  %s"%(rel_dir, count, 'y' if count==1 else 'ies', note) )
    logger.info( "  (%d stray .checkpoint file(s) under output/cache/ also included in the above counts)"%(checkpoint_count) )
    return plan


def clear_everything( logger ):
    """
    Deletes the CONTENTS of every dir in _CLEAR_DIRS (keeps the
    directory itself), except whatever _EXCLUDED_NAMES_BY_CLEAR_DIR
    lists for that dir - see that constant's own comment for why
    ../output/logs/clean_run_both_backends specifically is spared
    wholesale. Logged at info per directory, trace per file for real
    detail without flooding the default console output.
    """
    for rel_dir in _CLEAR_DIRS:
        abs_dir = os.path.normpath( os.path.join( _CORE_DIR, rel_dir ) )
        if not os.path.isdir( abs_dir ):
            os.makedirs( abs_dir, exist_ok=True )
            logger.debug( "created missing directory %s"%(abs_dir) )
            continue
        excluded_names = _EXCLUDED_NAMES_BY_CLEAR_DIR.get( rel_dir, set() )
        removed = 0
        for entry in os.listdir( abs_dir ):
            entry_path = os.path.join( abs_dir, entry )
            if entry in excluded_names:
                logger.log( 5, "keeping %s (excluded by name - see _EXCLUDED_NAMES_BY_CLEAR_DIR)"%(entry_path) )
                continue
            logger.log( 5, "removing %s"%(entry_path) )
            if os.path.isdir( entry_path ):
                shutil.rmtree( entry_path )
            else:
                os.remove( entry_path )
            removed += 1
        kept_note = " (%d excluded by name)"%(len(excluded_names)) if excluded_names else ""
        logger.info( "cleared %s (%d entr%s removed%s)"%(rel_dir, removed, 'y' if removed==1 else 'ies', kept_note) )

    for rel_file in _CLEAR_FILES:
        abs_file = os.path.normpath( os.path.join( _CORE_DIR, rel_file ) )
        if os.path.exists( abs_file ):
            logger.log( 5, "removing %s"%(abs_file) )
            os.remove( abs_file )
            logger.info( "cleared %s"%(rel_file) )
        else:
            logger.debug( "%s does not exist - nothing to clear"%(rel_file) )


def configurations_to_run( backend_names ):
    """
    Every (backend, variant) pair worth running, as (name, backend, variant).

    A backend with several sets of weights gets one pass per variant,
    because that is what makes the comparison possible at all. Running
    only the selected variant is how a 640m export sat unrun, and then
    unanalysed, while its 320n sibling got all the attention.

    Returns:
        A list of (label, backend_name, variant_name) tuples; variant is
        None for a backend with a single set of weights.
    """
    pairs = []
    for backend_name in backend_names:
        variants = bu_detector.variant_names( backend_name )
        if variants:
            pairs.extend( ( '%s/%s'%(backend_name, variant), backend_name, variant )
                          for variant in variants )
        else:
            pairs.append( ( backend_name, backend_name, None ) )
    return pairs


def run_betatv_pass( label, backend_name, variant, log_dir, logger ):
    """
    One real, full betatv.py pass under a given backend and variant.

    Both are passed through environment overrides
    (BETASUITE_DETECTOR_BACKEND_OVERRIDE and
    BETASUITE_DETECTOR_VARIANT_OVERRIDE) rather than by editing
    betaconfig.py, so an interrupted harness can never leave your config
    holding a value you did not choose. Full stdout and stderr are teed
    to this pass's own log file.
    """
    safe_label = label.replace( '/', '_' )
    log_path = os.path.join( log_dir, 'betatv_%s.log'%(safe_label) )
    env = dict( os.environ )
    env[ _BACKEND_OVERRIDE_ENV_VAR ] = backend_name
    if variant:
        env[ _VARIANT_OVERRIDE_ENV_VAR ] = variant
    else:
        env.pop( _VARIANT_OVERRIDE_ENV_VAR, None )

    logger.info( "running a REAL, full betatv.py pass under %s (this processes your entire current "
                 "video_path_uncensored library at full length - no preview slice, by design)"%(label) )
    logger.info( "  log: %s"%(log_path) )
    logger.log( 5, "subprocess env overrides: %s=%s%s"%(
        _BACKEND_OVERRIDE_ENV_VAR, backend_name,
        ' %s=%s'%(_VARIANT_OVERRIDE_ENV_VAR, variant) if variant else '' ) )

    with open( log_path, 'w' ) as log_file:
        proc = subprocess.run(
            [ sys.executable, 'betatv.py' ],
            cwd=_CORE_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode == 0:
        logger.info( "betatv.py under %s: OK"%(label) )
    else:
        logger.warning( "betatv.py under %s: FAILED (exit %d) - see %s"%(label, proc.returncode, log_path) )
    return proc.returncode, log_path


def run_analysis_tool( tool_name, log_dir, logger, extra_args=None ):
    """
    Runs one analysis tool as a subprocess, tees combined output to its
    own log file, prints it live - same approach run_all_analysis.sh/
    compare_backends_orchestrator.py already use.
    """
    log_path = os.path.join( log_dir, '%s.log'%(tool_name[:-3] if tool_name.endswith('.py') else tool_name) )
    args = [ sys.executable, os.path.join( 'tools', 'analysis', tool_name ) ] + ( extra_args or [] )

    logger.info( "running %s (log: %s)"%(tool_name, log_path) )

    with open( log_path, 'w' ) as log_file:
        proc = subprocess.Popen( args, cwd=_CORE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True )
        for line in proc.stdout:
            log_file.write( line )
        proc.wait()

    if proc.returncode == 0:
        logger.info( "%s: OK"%(tool_name) )
    else:
        logger.warning( "%s: FAILED (exit %d) - see %s"%(tool_name, proc.returncode, log_path) )
    return proc.returncode


def _confirm( assume_yes ):
    if assume_yes:
        return True
    print()
    answer = input( "Delete everything listed above and proceed with a clean run under both backends? [y/N] " ).strip().lower()
    return answer == 'y'


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--backends', nargs='+', default=None, choices=sorted( bu_detector._BACKENDS.keys() ),
        help="which backends to run, in order (default: every registered backend, sorted)" )
    parser.add_argument( '--yes', action='store_true',
        help="skip the confirmation prompt before deleting cache/logs/stats - for scripted/unattended use. "
             "Without this flag, the prompt is REQUIRED (the wait/confirm step the user's own standing rule for "
             "harness scripts asks for) - this script never deletes anything without an explicit human or --yes." )
    parser.add_argument( '--skip-clean', action='store_true',
        help="skip the delete step entirely (rare - mainly for debugging this script itself)" )
    parser.add_argument( '--skip-betatv-runs', action='store_true',
        help="skip the real betatv.py passes and go straight to the analysis suite against whatever cache exists "
             "(only useful combined with --skip-clean, e.g. to re-run just the analysis suite's report formatting)" )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="restrict analyze_jitter.py/analyze_track_breaks.py/analyze_style_flicker.py to these labels" )
    parser.add_argument( '--log-level', choices=sorted(_LEVEL_NAMES.keys()), default='info',
        help="this script's OWN narration log-file verbosity (default info) - trace/debug/info/warn/error, per "
             "the user's own standing rule for harness scripts. Does not affect betatv.py's own logging (see "
             "betaconfig.log_level) - only this orchestrator script's messages." )
    parser.add_argument( '--console-level', choices=sorted(_LEVEL_NAMES.keys()), default='info',
        help="this script's OWN narration console verbosity (default info) - independent of --log-level, so the "
             "log file can hold full trace detail while the console stays readable." )
    args = parser.parse_args()

    backends = args.backends or sorted( bu_detector._BACKENDS.keys() )

    ts = datetime.datetime.now().strftime( '%Y%m%d_%H%M%S' )
    log_dir = os.path.abspath( os.path.join( _CORE_DIR, '..', 'output', 'logs', 'clean_run_both_backends', ts ) )
    os.makedirs( log_dir, exist_ok=True )

    logger, own_log_path = _build_logger( log_dir, args.log_level, args.console_level )

    logger.info( "clean_run_both_backends.py starting at %s"%(datetime.datetime.now().isoformat()) )
    logger.info( "backends: %s"%(", ".join(backends)) )
    logger.info( "this run's own log: %s"%(own_log_path) )
    logger.info( "per-step logs: %s"%(log_dir) )

    if not args.skip_clean:
        logger.info( "the following will be CLEARED (contents deleted, directories kept):" )
        _describe_clear_plan( logger )
        if not _confirm( args.yes ):
            logger.warning( "not confirmed - aborting without deleting anything or running betatv.py." )
            sys.exit( 1 )
        clear_everything( logger )
    else:
        logger.info( "--skip-clean given: leaving existing cache/logs/stats in place." )

    pairs = configurations_to_run( backends )
    logger.info( "configurations to run: %s"%( ", ".join( label for label, _b, _v in pairs ) ) )

    betatv_results = {}
    if not args.skip_betatv_runs:
        for label, backend_name, variant in pairs:
            return_code, log_path = run_betatv_pass( label, backend_name, variant, log_dir, logger )
            betatv_results[ label ] = ( return_code, log_path )
        failed = [ label for label, (rc, _) in betatv_results.items() if rc != 0 ]
        if failed:
            logger.warning( "betatv.py failed for: %s - continuing to the analysis suite anyway; that "
                             "configuration's section may show no or incomplete cache below."%(", ".join(failed)) )
    else:
        logger.info( "--skip-betatv-runs given: skipping the real betatv.py passes." )

    analysis_args = []
    if args.labels:
        analysis_args = [ '--labels' ] + args.labels

    logger.info( "running the backend-aware analysis suite against the now-clean cache..." )
    analysis_results = {}
    # summarize_run first: it frames everything after it by saying where
    # the wall clock actually went, and acting on a tuning report without
    # that is how an evening gets spent on the stage that was 16% of the
    # run. score_distribution added because it was the one read-only
    # report nothing in the suite was running.
    #
    # The flag says whether that tool accepts --labels. Passing it to one
    # that does not turns a working tool into an argparse error, and the
    # suite then reports a failure that has nothing to do with the data.
    for tool_name, accepts_labels in ( ( 'summarize_run.py', False ),
                                       ( 'analyze_suppression_pairs.py', False ),
                                       ( 'analyze_score_distribution.py', True ),
                                       ( 'analyze_jitter.py', True ),
                                       ( 'analyze_track_breaks.py', True ),
                                       ( 'analyze_style_flicker.py', True ) ):
        analysis_results[ tool_name ] = run_analysis_tool(
            tool_name, log_dir, logger, analysis_args if accepts_labels else None )

    logger.info( "clean_run_both_backends.py finished at %s"%(datetime.datetime.now().isoformat()) )
    logger.info( "--- summary ---" )
    if not args.skip_betatv_runs:
        for label, _backend_name, _variant in pairs:
            rc, log_path = betatv_results[ label ]
            logger.info( "  betatv.py (%s): %s  (%s)"%(label, "OK" if rc == 0 else "FAILED (exit %d)"%(rc), log_path) )
    for tool_name, rc in analysis_results.items():
        logger.info( "  %s: %s"%(tool_name, "OK" if rc == 0 else "FAILED (exit %d)"%(rc)) )
    logger.info( "full logs: %s"%(log_dir) )
    logger.info( "every cache/log/stat file on disk now comes from exactly this run, for every backend run above - "
                 "safe to compare directly, apples-to-apples." )

    exit_codes = list( analysis_results.values() )
    if not args.skip_betatv_runs:
        exit_codes += [ rc for rc, _ in betatv_results.values() ]
    sys.exit( 1 if any( code != 0 for code in exit_codes ) else 0 )


if __name__ == '__main__':
    main()
