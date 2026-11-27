"""
test_optimize.py - "Optimize for this PC": the plan the evidence gives,
the narrowed measurements, the runner, and the perf.optimize commands.

What each test prevents:
  - evidence that shortens a stage it cannot settle, or a full run that
    skips anything a run uses
  - a memory-capped machine whose render spread is narrowed (the cap is
    what needs measuring)
  - narrowing that loses the value in use, or goes below one
  - an optimization whose progress, proposal or failure is never told,
    or a cancel that leaves the child running
  - Optimize starting beside a job, a second one, or on a busy machine
    without being asked to; a job starting while it measures
  - a proposal applied while a job runs, or not recorded as applied
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import betautils_optimize as bu_optimize
import betautils_protocol as bu_proto
import betautils_serve_perf as bu_serve_perf

from tests import protocol_schemas as schemas
from tests.test_tune_machine import tune


def _evidence( detected=10, capped=False ):
    nudges = [ { 'code': 'perf.memory_capped', 'params': {}, 'action': {} } ] if capped else []
    return { 'bands': [ { 'band': 'fhd', 'runs': detected, 'detected_runs': detected } ], 'nudges': nudges }


CONFIG = types.SimpleNamespace( encode_video_codec='h264_nvenc', encode_preset='p5', encode_crf=21 )


def _by_stage( planned ):
    return { step['stage']: step for step in planned['stages'] }


class TestPlan( unittest.TestCase ):

    def test_steady_evidence_narrows_detect_prefetch_and_render( self ):
        planned = bu_optimize.plan( _evidence(), CONFIG )
        steps = _by_stage( planned )
        self.assertEqual( planned['narrow'], [ 'detect', 'prefetch', 'render' ] )
        for stage in ( 'detect', 'prefetch', 'render' ):
            self.assertEqual( ( steps[stage]['run'], steps[stage]['code'] ), ( True, bu_optimize.STEADY ) )
        self.assertEqual( steps['inventory']['code'], bu_optimize.ALWAYS )
        self.assertEqual( planned['skip'], [ 'scan' ] )
        self.assertEqual( steps['encode']['code'], bu_optimize.NO_EVIDENCE )

    def test_too_little_evidence_measures_in_full( self ):
        planned = bu_optimize.plan( _evidence( detected=bu_optimize.STEADY_RUNS - 1 ), CONFIG )
        self.assertEqual( planned['narrow'], [] )
        self.assertEqual( _by_stage( planned )['detect']['code'], bu_optimize.NO_EVIDENCE )
        self.assertEqual( bu_optimize.plan( None, CONFIG )['narrow'], [] )

    def test_memory_capping_measures_the_render_in_full( self ):
        planned = bu_optimize.plan( _evidence( capped=True ), CONFIG )
        self.assertEqual( planned['narrow'], [ 'detect', 'prefetch' ] )
        self.assertEqual( _by_stage( planned )['render']['code'], bu_optimize.MEMORY_CAPPED )

    def test_encode_settled_by_an_earlier_optimization( self ):
        same = bu_optimize.encode_settings( CONFIG )
        planned = bu_optimize.plan( _evidence(), CONFIG, optimized_settings=same )
        self.assertIn( 'encode', planned['skip'] )
        self.assertEqual( _by_stage( planned )['encode']['code'], bu_optimize.ENCODE_SETTLED )
        changed = dict( same, encode_preset='p7' )
        self.assertNotIn( 'encode', bu_optimize.plan( _evidence(), CONFIG, optimized_settings=changed )['skip'] )

    def test_full_runs_everything_a_run_uses( self ):
        planned = bu_optimize.plan( _evidence(), CONFIG, optimized_settings=bu_optimize.encode_settings( CONFIG ),
                                    full=True )
        self.assertEqual( planned['narrow'], [] )
        self.assertEqual( planned['skip'], [ 'scan' ] )
        self.assertEqual( _by_stage( planned )['scan']['code'], bu_optimize.NOT_USED )

    def test_a_nudge_limits_the_stages( self ):
        planned = bu_optimize.plan( _evidence( capped=True ), CONFIG, stages=[ 'render' ] )
        self.assertEqual( planned['skip'], [ 'detect', 'prefetch', 'encode', 'scan' ] )
        self.assertTrue( _by_stage( planned )['inventory']['run'] )


class TestNarrowCandidates( unittest.TestCase ):

    def test_batch( self ):
        self.assertEqual( tune.batch_sizes_to_try( 4, narrow=True ), [ 2, 4, 8 ] )
        self.assertEqual( tune.batch_sizes_to_try( 1, narrow=True ), [ 1, 2 ] )

    def test_prefetch( self ):
        self.assertEqual( tune.prefetch_depths_to_try( 4, 0, narrow=True ), [ 0, 4 ] )
        self.assertEqual( tune.prefetch_depths_to_try( 4, 12, narrow=True ), [ 0, 4, 12 ] )

    def test_render_workers( self ):
        self.assertEqual( tune.render_worker_counts( 16, narrow=True ), [ 6, 8, 10 ] )
        self.assertEqual( tune.render_worker_counts( 16, narrow=True, current=2 ), [ 2, 4 ] )
        self.assertEqual( tune.render_worker_counts( 16, nvenc_sessions=7, narrow=True ), [ 6 ] )
        self.assertEqual( tune.render_worker_counts( 4, narrow=True, current=4 ), [ 2, 4 ] )

    def test_narrow_names_its_stages( self ):
        args = tune.build_parser().parse_args( [ 'measure', '--out-dir', '/x', '--narrow', 'detect', 'render' ] )
        self.assertTrue( tune._narrow( args, 'detect' ) )
        self.assertFalse( tune._narrow( args, 'prefetch' ) )
        self.assertFalse( tune._narrow( tune.build_parser().parse_args( [ 'measure', '--out-dir', '/x' ] ), 'render' ) )


FAKE_TUNE = r'''
import json, os, sys, time
args = sys.argv[1:]
out = args[ args.index( '--out-dir' ) + 1 ]
mode = os.environ.get( 'FAKE_TUNE_MODE', 'ok' )
with open( os.path.join( out, 'argv.json' ), 'w' ) as handle:
    json.dump( args, handle )
done = {}
for stage in ( 'inventory', 'detect' ):
    done[stage] = 0.1
    with open( os.path.join( out, 'results.json' ), 'w' ) as handle:
        json.dump( { 'stage_seconds': done }, handle )
    time.sleep( 0.05 )
if mode == 'sleep':
    time.sleep( 30 )
if mode == 'fail':
    sys.exit( 2 )
with open( os.path.join( out, 'proposal.json' ), 'w' ) as handle:
    json.dump( { 'render_workers': { 'value': 6, 'reason': 'x' },
                 'detector_backend.nudenet_v3.nn_batch_size': { 'value': 8, 'reason': 'y' } }, handle )
if mode == 'fail_after_proposal':
    sys.exit( 1 )
'''


class _Events:

    def __init__( self ):
        self.items = []
        self.cond = threading.Condition()

    def __call__( self, name, data ):
        with self.cond:
            self.items.append( ( name, data ) )
            self.cond.notify_all()

    def states( self ):
        return [ data['state'] for _name, data in self.items ]


class TestOptimizer( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='optimize-' )
        self.addCleanup( shutil.rmtree, self.root, True )
        self.script = os.path.join( self.root, 'fake_tune.py' )
        with open( self.script, 'w', encoding='utf-8' ) as handle:
            handle.write( FAKE_TUNE )
        self.events = _Events()
        self.optimizer = bu_optimize.Optimizer( self.events, mock.Mock(), python=sys.executable, script=self.script,
                                                poll=0.02 )

    def run_mode( self, mode, planned=None, finished=None ):
        planned = planned or bu_optimize.plan( _evidence(), CONFIG )
        out = os.path.join( self.root, 'out' )
        with mock.patch.dict( os.environ, { 'FAKE_TUNE_MODE': mode } ):
            self.assertTrue( self.optimizer.start( planned, out, video='/v.mp4', on_finished=finished ) )
        return out

    def test_progress_then_the_proposal( self ):
        out = self.run_mode( 'ok', finished=lambda proposal: [ { 'setting': name } for name in sorted( proposal ) ] )
        self.optimizer.wait( 20 )
        self.assertEqual( self.events.states()[0], 'running' )
        self.assertEqual( self.events.states()[-1], 'finished' )
        final = self.events.items[-1][1]
        self.assertEqual( [ row['setting'] for row in final['proposal'] ],
                          [ 'detector_backend.nudenet_v3.nn_batch_size', 'render_workers' ] )
        self.assertIn( 'detect', [ data['stage'] for _n, data in self.events.items ] )
        with open( os.path.join( out, 'argv.json' ), encoding='utf-8' ) as handle:
            argv = json.load( handle )
        self.assertEqual( argv[ argv.index( '--skip' ) + 1 ], 'scan' )
        self.assertEqual( argv[ argv.index( '--narrow' ) + 1: argv.index( '--narrow' ) + 4 ],
                          [ 'detect', 'prefetch', 'render' ] )
        self.assertEqual( argv[ argv.index( '--video' ) + 1 ], '/v.mp4' )
        self.assertFalse( self.optimizer.running )
        self.assertEqual( self.optimizer.last['state'], bu_optimize.FINISHED )

    def test_a_failure_is_told_with_its_code( self ):
        self.run_mode( 'fail' )
        self.optimizer.wait( 20 )
        final = self.events.items[-1][1]
        self.assertEqual( final['state'], 'failed' )
        self.assertEqual( ( final['code'], final['params'] ), ( 'optimize.failed', { 'exit_code': 2 } ) )

    def test_a_proposal_from_a_run_that_failed_is_not_offered( self ):
        self.run_mode( 'fail_after_proposal' )
        self.optimizer.wait( 20 )
        self.assertEqual( self.events.states()[-1], 'failed' )
        self.assertEqual( self.optimizer.last['state'], bu_optimize.FAILED )

    def test_cancel_stops_the_child( self ):
        self.run_mode( 'sleep' )
        self.assertFalse( self.optimizer.start( bu_optimize.plan( None, CONFIG ), self.root ), 'one at a time' )
        self.assertTrue( self.optimizer.cancel() )
        self.optimizer.wait( 20 )
        self.assertEqual( self.events.states()[-1], 'cancelled' )
        self.assertFalse( self.optimizer.running )
        self.assertFalse( self.optimizer.cancel() )

    @schemas.needs_jsonschema
    def test_events_match_the_schema( self ):
        self.run_mode( 'ok', finished=lambda proposal: [ { 'setting': 'render_workers', 'value': 6, 'current': 0 } ] )
        self.optimizer.wait( 20 )
        self.run_mode( 'fail' )
        self.optimizer.wait( 20 )
        for name, data in self.events.items:
            schemas.validate_event_data( name, data )


class TestProposalRows( unittest.TestCase ):

    def test_rows_carry_the_value_in_effect( self ):
        config = types.SimpleNamespace( render_workers=0,
                                        detector_backend={ 'nudenet_v3': { 'nn_batch_size': 4 } } )
        rows = bu_optimize.proposal_rows( { 'render_workers': { 'value': 6, 'reason': 'x' },
                                            'detector_backend.nudenet_v3.nn_batch_size': { 'value': 8 },
                                            'broken': 'not an entry' }, config )
        self.assertEqual( rows, [ { 'setting': 'detector_backend.nudenet_v3.nn_batch_size', 'value': 8, 'current': 4 },
                                  { 'setting': 'render_workers', 'value': 6, 'current': 0 } ] )


class _Server:

    def __init__( self, busy=None ):
        self.sent = []
        self.send_event = lambda name, data: self.sent.append( ( name, data ) )
        self._logger = mock.Mock()
        self.jobs = types.SimpleNamespace( busy_job=lambda: busy )
        self.optimizer = None


class TestCommands( unittest.TestCase ):

    def setUp( self ):
        self.index = mock.Mock()
        self.index.last_optimization.return_value = None
        patches = [
            mock.patch.object( bu_serve_perf, 'shared_index', return_value=self.index ),
            mock.patch.object( bu_serve_perf, 'machine', return_value=( {}, 'fp' ) ),
            mock.patch.object( bu_serve_perf, 'perf_evidence', return_value=_evidence() ),
            mock.patch.object( bu_serve_perf, 'machine_busy', return_value=False ),
            mock.patch.object( bu_serve_perf, 'current_settings', return_value={ 'variant': '640m', 'batch': 4 } ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup( patch.stop )

    def error( self, call ):
        with self.assertRaises( bu_proto.ProtocolError ) as caught:
            call()
        return caught.exception.error['code']

    def test_refusals( self ):
        self.assertEqual( self.error( lambda: bu_serve_perf.perf_optimize( _Server( busy='j1' ), {} ) ), 'perf.busy' )
        server = _Server()
        server.optimizer = types.SimpleNamespace( running=True )
        self.assertEqual( self.error( lambda: bu_serve_perf.perf_optimize( server, {} ) ), 'perf.optimize_running' )
        with mock.patch.object( bu_serve_perf, 'machine_busy', return_value=True ):
            self.assertEqual( self.error( lambda: bu_serve_perf.perf_optimize( _Server(), {} ) ), 'perf.machine_busy' )
        for data in ( { 'full': 1 }, { 'video': 'relative.mp4' }, { 'stages': [ 'everything' ] } ):
            self.assertEqual( self.error( lambda: bu_serve_perf.perf_optimize( _Server(), data ) ), 'command.bad_params' )
        self.assertEqual( self.error( lambda: bu_serve_perf.perf_optimize( _Server(), { 'video': '/no/such.mp4' } ) ),
                          'job.no_such_file' )

    def test_force_runs_on_a_busy_machine_and_answers_the_plan( self ):
        server = _Server()
        started = []
        server.optimizer = types.SimpleNamespace( running=False, last=None,
                                                  start=lambda planned, out, video=None, on_finished=None:
                                                  started.append( ( planned, out ) ) )
        with mock.patch.object( bu_serve_perf, 'machine_busy', return_value=True ):
            answer = bu_serve_perf.perf_optimize( server, { 'force': True } )
        self.assertEqual( answer['narrow'], [ 'detect', 'prefetch', 'render' ] )
        self.assertEqual( len( started ), 1 )
        self.assertIn( 'optimize-', started[0][1] )

    def test_apply( self ):
        server = _Server()
        server.optimizer = types.SimpleNamespace( running=False, last=None )
        self.assertEqual( self.error( lambda: bu_serve_perf.perf_optimize_apply( server, {} ) ), 'perf.no_proposal' )
        server.optimizer.last = { 'at': 6.0, 'state': bu_optimize.FAILED, 'proposal': None }
        self.assertEqual( self.error( lambda: bu_serve_perf.perf_optimize_apply( server, {} ) ), 'perf.no_proposal' )
        server.optimizer.last = { 'at': 7.0, 'state': bu_optimize.FINISHED,
                                  'proposal': { 'render_workers': { 'value': 6 } } }
        busy = _Server( busy='j2' )
        busy.optimizer = server.optimizer
        self.assertEqual( self.error( lambda: bu_serve_perf.perf_optimize_apply( busy, {} ) ), 'config.busy' )
        with mock.patch( 'betautils_serve_config._edit', return_value={ 'moves': [], 'may_move': [], 'written': True,
                                                                         'dry_run': False } ) as edit:
            answer = bu_serve_perf.perf_optimize_apply( server, {} )
        edit.assert_called_once_with( {}, { ( 'render_workers', ): 6 }, [] )
        self.assertEqual( answer['applied'], [ 'render_workers' ] )
        at, fingerprint, _settings, _proposal, applied = self.index.record_optimization.call_args[0]
        self.assertEqual( ( at, fingerprint, applied ), ( 7.0, 'fp', True ) )

    def test_no_job_while_optimizing( self ):
        import betautils_serve_job as bu_serve_job
        server = _Server()
        server.optimizer = types.SimpleNamespace( running=True )
        self.assertEqual( self.error( lambda: bu_serve_job.job_submit( server, { 'files': [ '/x.mp4' ] } ) ),
                          'perf.optimize_running' )

    @schemas.needs_jsonschema
    def test_schemas( self ):
        server = _Server()
        server.optimizer = types.SimpleNamespace( running=False, last=None, start=lambda *a, **k: True )
        schemas.validate_command_result( 'perf.optimize', bu_serve_perf.perf_optimize( server, {} ) )
        schemas.validate_command_data( 'perf.optimize', { 'full': True, 'stages': [ 'render' ], 'force': False } )
        schemas.validate_command_result( 'perf.optimize_cancel', { 'cancelled': False } )
        schemas.validate_command_result( 'perf.optimize_apply', { 'moves': [], 'may_move': [], 'written': True,
                                                                  'dry_run': False, 'applied': [ 'render_workers' ] } )


if __name__ == '__main__':
    unittest.main()
