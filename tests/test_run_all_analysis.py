"""
test_run_all_analysis.py - the one-click post-run suite's ordering.

The suite runs its middle steps several at a time. That is fine for
steps that only read caches, and wrong for steps that MEASURE time: a
timing step running beside three CPU-bound analyses reports their load
as its own cost. encode, reverse, persons and hardware are timing steps, so they
must start only after every parallel step has finished, run one at a
time, and finish before auto_tune. They are off unless asked for. These tests run the real script against a
stand-in Python that records when each step started and ended.

Also pinned: every betabench step writes to its own folder. Without
--out-dir, betabench names its folder after the current second, and two
steps launched together shared one folder and one results.json.

And the memory side: every step runs under tools/lib/step_meter.py, the
summary carries each step's time and peak, an OOM kill (exit 137) is
named as one, and a parallel step only starts when its expected peak
fits the memory budget beside the steps already running.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

# A stand-in for the project Python. Records "<start> <end> <argv>" per
# call, sleeping long enough that overlapping steps really do overlap.
# Called as step_meter.py it does what the meter does: runs the command
# after "--" and writes a record (FAKE_KILL names a tool whose step dies
# as an OOM kill would, exit 137).
FAKE_PYTHON = r'''#!/bin/bash
if [ "${1:-}" = "-" ]; then cat >/dev/null; exit 0; fi
if [ "${1:-}" = "-c" ]; then exit 0; fi
if [ "$(basename "${1:-}")" = "step_meter.py" ]; then
    record="$3"; shift 4
    "$@"; status=$?
    signal=null
    if [ -n "${FAKE_KILL:-}" ] && [ "$(basename "${2:-}")" = "$FAKE_KILL" ]; then status=137; signal=9; fi
    echo "{\"exit\": $status, \"signal\": $signal, \"wall_seconds\": 0.3, \"peak_rss_mb\": 12.5}" > "$record"
    exit $status
fi
start=$(date +%s.%N)
sleep 0.3
end=$(date +%s.%N)
echo "$start $end $*" >> "__CALLS__"
'''


class TestRunAllAnalysisOrdering( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-suite-' )
        self.repo = os.path.join( self.root, 'repo' )
        for relative in ( 'tools/analysis/run_all_analysis.sh', 'tools/lib/python_env.sh' ):
            target = os.path.join( self.repo, relative )
            os.makedirs( os.path.dirname( target ), exist_ok=True )
            shutil.copy( os.path.join( REPO_ROOT, relative ), target )
        self.calls = os.path.join( self.root, 'calls.txt' )
        self.fake_python = os.path.join( self.root, 'fake_python' )
        with open( self.fake_python, 'w' ) as handle:
            handle.write( FAKE_PYTHON.replace( '__CALLS__', self.calls ) )
        os.chmod( self.fake_python, 0o755 )

    def tearDown( self ):
        shutil.rmtree( self.root, ignore_errors=True )

    def run_suite( self, *flags, status=0, **environment_extra ):
        environment = dict( os.environ, PYTHON_BIN=self.fake_python,
                            PYTHON_BIN_SOURCE='test stand-in', BETASUITE_ANALYSIS_JOBS='3',
                            **environment_extra )
        environment.pop( 'BETASUITE_ANALYSIS_MEMORY_MB', None )
        result = subprocess.run( [ 'bash', 'tools/analysis/run_all_analysis.sh' ] + list( flags ),
                                 cwd=self.repo, env=environment, stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, timeout=120 )
        self.assertEqual( result.returncode, status, result.stdout[-2000:] + result.stderr[-2000:] )
        self.stdout = result.stdout
        steps = []
        with open( self.calls ) as handle:
            for line in handle:
                start, end, *argv = line.split()
                steps.append( { 'start': float( start ), 'end': float( end ), 'argv': argv,
                                'name': self._name( argv ) } )
        return steps

    @staticmethod
    def _name( argv ):
        script = os.path.basename( argv[0] )
        if script == 'betabench.py':
            return 'betabench ' + argv[1]
        return script

    TIMING = ( 'betabench encode', 'betabench reverse', 'betabench persons', 'betabench hardware',
               'betabench render' )

    def test_timing_steps_run_alone_after_the_parallel_batch( self ):
        steps = self.run_suite( '--with-timing' )
        by_name = { step['name']: step for step in steps }
        for name in self.TIMING:
            self.assertIn( name, by_name )
        timing = [ by_name[name] for name in self.TIMING ]
        for step in timing:
            for other in steps:
                if other is step:
                    continue
                overlaps = other['start'] < step['end'] and step['start'] < other['end']
                self.assertFalse( overlaps, "%s ran alongside %s"%( step['name'], other['name'] ) )
        batch = [ s for s in steps if s['name'] not in (
            ( 'summarize_run.py', 'auto_tune.py' ) + self.TIMING ) ]
        self.assertGreater( len( batch ), 3 )
        self.assertGreater( min( t['start'] for t in timing ), max( s['end'] for s in batch ) )
        self.assertGreater( by_name['auto_tune.py']['start'], max( t['end'] for t in timing ),
                            "auto_tune stays last" )

    def test_the_parallel_batch_really_is_parallel( self ):
        # Precondition for the test above: if nothing in the batch
        # overlapped, "the timing steps overlap nothing" would pass vacuously.
        steps = self.run_suite()
        batch = [ s for s in steps if s['name'].startswith( 'analyze_' ) ]
        overlapping = any( a['start'] < b['end'] and b['start'] < a['end']
                           for i, a in enumerate( batch ) for b in batch[ i + 1: ] )
        self.assertTrue( overlapping )

    def test_every_betabench_step_has_its_own_folder_under_this_run( self ):
        steps = self.run_suite( '--with-timing' )
        folders = []
        for step in steps:
            if step['name'].startswith( 'betabench' ):
                self.assertIn( '--out-dir', step['argv'], step['name'] )
                folder = step['argv'][ step['argv'].index( '--out-dir' ) + 1 ]
                self.assertIn( os.path.join( 'output', 'logs', 'analysis_runs' ), folder )
                folders.append( folder )
        self.assertEqual( len( folders ), len( set( folders ) ), folders )

    def test_timing_steps_are_off_unless_asked_for( self ):
        for flags in ( (), ( '--quick', ) ):
            names = [ step['name'] for step in self.run_suite( *flags ) ]
            for name in self.TIMING:
                self.assertNotIn( name, names, flags )
            self.assertIn( 'betabench cache', names, "the cache-only analyses always run" )
            self.assertEqual( names[-1], 'auto_tune.py' )
            os.remove( self.calls )

    def test_each_timing_step_has_its_own_switch( self ):
        for flag, wanted in ( ( '--with-encode', { 'betabench encode' } ),
                              ( '--with-reverse', { 'betabench reverse' } ),
                              ( '--with-persons', { 'betabench persons' } ),
                              ( '--with-hardware', { 'betabench hardware', 'betabench render' } ) ):
            names = set( step['name'] for step in self.run_suite( flag ) )
            self.assertEqual( names & set( self.TIMING ), wanted, flag )
            os.remove( self.calls )

    def test_the_structure_moves_land_in_this_run_as_a_script( self ):
        steps = self.run_suite()
        cache = [ step for step in steps if step['name'] == 'betabench cache' ][0]
        target = cache['argv'][ cache['argv'].index( '--write-move-script' ) + 1 ]
        self.assertTrue( target.endswith( 'moves.sh' ) )
        self.assertIn( os.path.join( 'output', 'logs', 'analysis_runs' ), target )

    @staticmethod
    def _overlapping( steps ):
        return any( a['start'] < b['end'] and b['start'] < a['end']
                    for i, a in enumerate( steps ) for b in steps[ i + 1: ] )

    def run_folder( self ):
        runs = os.path.join( self.root, 'output', 'logs', 'analysis_runs' )
        return os.path.join( runs, sorted( os.listdir( runs ) )[-1] )

    def test_every_step_is_metered_and_the_summary_says_so( self ):
        steps = self.run_suite()
        folder = self.run_folder()
        for label in ( 'summarize_run', 'betabench_cache', 'analyze_jitter', 'auto_tune_dry_run' ):
            self.assertTrue( os.path.isfile( os.path.join( folder, label + '.meter.json' ) ), label )
        self.assertIn( 'OK   analyze_jitter  (0s, peak 12 MB)', self.stdout )
        self.assertIn( '--- analyze_jitter: 0s, peak 12 MB', self.stdout )
        self.assertEqual( len( steps ), 12, "the meter runs each tool exactly once" )

    def test_an_oom_kill_is_named_with_the_memory_it_reached( self ):
        self.run_suite( status=1, FAKE_KILL='analyze_jitter.py' )
        self.assertIn( 'FAIL analyze_jitter (exit 137: killed by the system, almost always for running '
                       'out of memory; it had reached 12 MB)', self.stdout )
        self.assertIn( 'OK   analyze_track_breaks', self.stdout, "the suite carries on" )

    def test_a_budget_smaller_than_any_step_runs_the_batch_one_at_a_time( self ):
        steps = self.run_suite( '--memory-budget', '1' )
        batch = [ s for s in steps if s['name'] not in ( 'summarize_run.py', 'auto_tune.py' ) ]
        self.assertGreater( len( batch ), 3 )
        self.assertFalse( self._overlapping( batch ) )
        self.assertIn( 'more than the 1 MB budget: it will run alone', self.stdout )

    def test_a_budget_that_fits_two_steps_runs_two_at_a_time( self ):
        # No caches and no history: every step is expected at the 300 MB
        # floor, so 700 MB fits two of them and never three.
        steps = self.run_suite( '--memory-budget', '700' )
        batch = sorted( ( s for s in steps if s['name'] not in ( 'summarize_run.py', 'auto_tune.py' ) ),
                        key=lambda s: s['start'] )
        self.assertTrue( self._overlapping( batch ) )
        for step in batch:
            beside = [ o for o in batch if o is not step and o['start'] < step['end'] and step['start'] < o['end']
                       and o['start'] <= step['start'] ]
            self.assertLessEqual( len( beside ), 1, step['name'] )
        self.assertIn( 'holding ', self.stdout )

    def test_an_earlier_run_sizes_the_estimate( self ):
        earlier = os.path.join( self.root, 'output', 'logs', 'analysis_runs', '20000101_000000' )
        os.makedirs( earlier )
        with open( os.path.join( earlier, 'analyze_jitter.meter.json' ), 'w' ) as handle:
            handle.write( '{"exit": 0, "signal": null, "wall_seconds": 4.0, "peak_rss_mb": 1000.4}' )
        self.run_suite( '--memory-budget', '100000' )
        with open( os.path.join( self.run_folder(), 'admission.log' ) ) as handle:
            admission = handle.read()
        self.assertIn( 'analyze_jitter: expect about 1251 MB (run 20000101_000000 peaked at 1000 MB, '
                       'plus 25%)', admission )
        self.assertIn( 'analyze_track_breaks: expect about 300 MB (no earlier run recorded it', admission )

    def test_help_prints_the_whole_header( self ):
        result = subprocess.run( [ 'bash', 'tools/analysis/run_all_analysis.sh', '--help' ],
                                 cwd=REPO_ROOT, capture_output=True, text=True, timeout=30 )
        self.assertIn( 'betabench.py reverse', result.stdout )
        self.assertIn( 'safe to schedule', result.stdout, "the last header line" )
        self.assertNotIn( 'python_env.sh', result.stdout, "nothing past the header" )


if __name__ == '__main__':
    unittest.main()
