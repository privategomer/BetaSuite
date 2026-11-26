"""
test_perf.py - the performance index (betautils_perf), its sink and
perf.evidence, on synthetic stats rows.

What each test prevents:
  - past runs missing on the first day (the one-time import), or counted
    twice (an import after the sink, or a second import)
  - a preview, a run read from the detection cache, or a run the
    governor slowed, skewing the speeds
  - CUDA and CPU runs, or two machines, pooled into one speed
  - speeds measured at another model or batch size shown as this one's
  - a memory cap that did not bind counted as capping the render
  - a nudge that fires without its cause, or not with it
  - a broken index failing a run
  - answers drifting from protocol/v1's schemas
"""

import json
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import betautils_perf as bu_perf
import betautils_run_record as bu_run_record

from tests import protocol_schemas as schemas

CUDA = 'CUDAExecutionProvider,CPUExecutionProvider'
CPU = 'CPUExecutionProvider'


def _row( n, height=1080, detect=100.0, encode=60.0, frames=9000, fps=30.0, preview=False, providers=CUDA,
          variant='640m', batch=4, workers=8, cap=None, throttled=None, paced=None ):
    row = {
        'file': 'clip%03d.mp4'%( n, ), 'output': 'out%03d.mp4'%( n, ), 'timestamp': 1000.0 + n,
        'execution_providers': providers, 'detector_backend': 'nudenet_v3', 'detector_variant': variant,
        'nn_batch_size': batch, 'video_censor_fps': 9, 'video_width': height * 16 // 9, 'video_height': height,
        'video_frames': frames, 'video_fps': fps, 'preview_mode': preview, 'detection_seconds': detect,
        'encode_seconds': encode, 'total_seconds': detect + encode, 'render_frames_written': frames,
        'render_render_workers': workers, 'render_memory_worker_cap': cap,
    }
    if throttled is not None:
        row['governor_throttled_seconds'] = throttled
    if paced is not None:
        row['governor_paced_seconds'] = paced
    return row


class _Base( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='perf-' )
        self.addCleanup( shutil.rmtree, self.root, True )
        self.stats = os.path.join( self.root, 'stats.jsonl' )
        self.db = os.path.join( self.root, 'perf.sqlite' )

    def write_stats( self, rows, tail='' ):
        with open( self.stats, 'w', encoding='utf-8' ) as handle:
            for row in rows:
                handle.write( json.dumps( row ) + '\n' )
            handle.write( tail )

    def index( self ):
        index = bu_perf.Index.open( self.db, stats_path=self.stats )
        self.addCleanup( index.close )
        return index


class TestImport( _Base ):

    def test_first_open_imports_once_and_never_twice( self ):
        self.write_stats( [ _row( i ) for i in range( 5 ) ], tail='{"file": "trunc' )
        index = self.index()
        self.assertEqual( index.count(), 5 )
        self.assertEqual( index.import_stats( self.stats ), ( 0, 6 ) )
        index.close()
        self.assertEqual( bu_perf.Index.open( self.db, stats_path=self.stats ).count(), 5 )

    def test_an_emptied_index_is_not_refilled_behind_your_back( self ):
        self.write_stats( [ _row( 0 ) ] )
        index = self.index()
        index.db.execute( "DELETE FROM runs" )
        index.db.commit()
        index.close()
        self.assertEqual( bu_perf.Index.open( self.db, stats_path=self.stats ).count(), 0 )

    def test_rebuild( self ):
        self.write_stats( [ _row( i ) for i in range( 3 ) ] )
        index = self.index()
        self.write_stats( [ _row( i ) for i in range( 4 ) ] )
        self.assertEqual( index.rebuild( self.stats ), ( 4, 4 ) )
        self.assertEqual( index.count(), 4 )

    def test_missing_log( self ):
        self.assertEqual( self.index().count(), 0 )

    def test_sink_and_import_do_not_double_count( self ):
        self.write_stats( [] )
        index = self.index()
        run = types.SimpleNamespace( stats=_row( 7 ) )
        bu_perf.PerfSink( lambda: index, hardware='abc' ).record( run )
        bu_perf.PerfSink( lambda: index, hardware='abc' ).record( types.SimpleNamespace( stats=None ) )
        self.write_stats( [ _row( 7 ) ] )
        index.import_stats( self.stats )
        self.assertEqual( index.count(), 1 )
        self.assertEqual( index.rows( CUDA, 'abc' )[0]['origin'], 'run' )

    def test_a_row_seen_again_keeps_the_first( self ):
        index = self.index()
        self.assertTrue( index.add( bu_perf.row_from_stats( _row( 3 ), 'run', hardware='first' ) ) )
        self.assertFalse( index.add( bu_perf.row_from_stats( _row( 3 ), 'run', hardware='second' ) ) )
        self.assertEqual( [ r['hardware'] for r in index.rows( CUDA, 'first' ) ], [ 'first' ] )

    def test_a_broken_index_never_fails_the_run( self ):
        def broken():
            raise OSError( 'disk' )
        recorder = bu_run_record.RunRecorder( [ bu_perf.PerfSink( broken ) ], logger=mock.Mock() )
        run = recorder.start( '/x.mp4' )
        run.stats = _row( 1 )
        recorder.record( run )


class TestEvidence( _Base ):

    def evidence( self, rows, **kw ):
        self.write_stats( rows )
        kw.setdefault( 'current', { 'variant': '640m', 'batch': 4 } )
        return self.index().evidence( kw.pop( 'providers', CUDA ), **kw )

    def test_speeds_per_band( self ):
        rows = [ _row( 0, detect=100.0, encode=50.0 ), _row( 1, detect=150.0, encode=50.0 ),
                 _row( 2, height=2160, detect=300.0, encode=200.0 ) ]
        ev = self.evidence( rows )
        bands = { band['band']: band for band in ev['bands'] }
        # 9000 frames at 30 fps: 300 s of video.
        self.assertEqual( bands['fhd']['runs'], 2 )
        self.assertAlmostEqual( bands['fhd']['detect_speed'], ( 3.0 + 2.0 ) / 2 )
        self.assertAlmostEqual( bands['fhd']['render_fps'], 180.0 )
        self.assertAlmostEqual( bands['uhd']['total_speed'], 0.6 )
        self.assertNotIn( 'sd', bands )

    def test_previews_cached_and_slowed_runs_are_left_out( self ):
        rows = [ _row( 0, detect=100.0 ),
                 _row( 1, detect=5.0, preview=True ),
                 _row( 2, detect=2.0 ),                         # cached: 150x real time
                 _row( 3, detect=40.0, throttled=30.0 ) ]       # slowed for 30 of 100 s
        ev = self.evidence( rows )
        fhd = ev['bands'][0]
        self.assertEqual( fhd['runs'], 2 )                     # the cached run still renders
        self.assertEqual( fhd['detected_runs'], 1 )
        self.assertAlmostEqual( fhd['detect_speed'], 3.0 )
        self.assertEqual( ev['measured'], 2 )

    def test_detection_yields_count_as_slowing( self ):
        self.assertEqual( self.evidence( [ _row( 0, paced=30.0 ) ] )['measured'], 0 )

    def test_band_edges( self ):
        self.assertEqual( [ bu_perf.band_of( h ) for h in ( 599, 600, 899, 900, 1299, 1300, 1799, 1800, None ) ],
                          [ 'sd', 'hd', 'hd', 'fhd', 'fhd', 'qhd', 'qhd', 'uhd', None ] )

    def test_render_speed_counts_the_frames_written( self ):
        row = _row( 0, encode=50.0 )
        row['render_frames_written'] = 4500    # a render that wrote half the frames (a skip)
        self.assertAlmostEqual( self.evidence( [ row ] )['bands'][0]['render_fps'], 90.0 )

    def test_a_little_slowing_is_kept( self ):
        ev = self.evidence( [ _row( 0, throttled=2.0, paced=1.0 ) ] )
        self.assertEqual( ev['measured'], 1 )

    def test_other_providers_and_other_machines_are_not_pooled( self ):
        rows = [ _row( 0 ), _row( 1, providers=CPU, detect=900.0 ) ]
        self.assertEqual( self.evidence( rows )['measured'], 1 )
        self.assertEqual( self.evidence( rows, providers=CPU )['bands'][0]['detect_speed'], round( 300 / 900, 3 ) )
        index = self.index()
        index.add( bu_perf.row_from_stats( _row( 50, detect=10.0 ), 'run', hardware='other' ) )
        # This machine: the imported CUDA row only (the CPU row and the
        # other machine's row are not its). The other machine: its own
        # row and the imported CUDA row.
        mine = index.evidence( CUDA, hardware='mine', current={} )
        self.assertEqual( mine['runs'], 1 )
        theirs = index.evidence( CUDA, hardware='other', current={} )
        self.assertEqual( theirs['runs'], 2 )

    def test_only_the_settings_in_use( self ):
        rows = [ _row( 0 ), _row( 1, batch=8, detect=10.0 ), _row( 2, variant='320n', detect=20.0 ) ]
        ev = self.evidence( rows )
        self.assertEqual( ev['measured'], 1 )
        self.assertEqual( ev['bands'][0]['detect_speed'], 3.0 )

    def test_memory_capped_counts_only_a_cap_that_set_the_workers( self ):
        rows = [ _row( i, workers=8, cap=10 ) for i in range( 4 ) ] + [ _row( 10 + i, workers=5, cap=5 ) for i in range( 3 ) ]
        ev = self.evidence( rows )
        self.assertEqual( ev['memory_capped'], { 'runs': 3, 'of': 7 } )
        self.assertIn( bu_perf.MEMORY_CAPPED, [ n['code'] for n in ev['nudges'] ] )

    def test_too_few_renders_say_nothing_about_memory( self ):
        ev = self.evidence( [ _row( i, workers=4, cap=4 ) for i in range( 4 ) ] )
        self.assertNotIn( bu_perf.MEMORY_CAPPED, [ n['code'] for n in ev['nudges'] ] )


class TestNudges( _Base ):

    def codes( self, **kw ):
        if os.path.exists( self.db ):
            os.remove( self.db )
        self.write_stats( kw.pop( 'rows', [ _row( 0 ) ] ) )
        index = self.index()
        for at, settings in kw.pop( 'optimized', [] ):
            index.record_optimization( at, None, settings, {}, True )
        ev = index.evidence( CUDA, current=kw.pop( 'current', { 'variant': '640m', 'batch': 4 } ), **kw )
        return [ nudge['code'] for nudge in ev['nudges'] ], ev

    def test_gpu_off( self ):
        codes, ev = self.codes( facts={ 'cuda_usable': True, 'gpu': { 'name': 'G' } }, gpu_enabled=0 )
        self.assertIn( bu_perf.GPU_OFF, codes )
        nudge = [ n for n in ev['nudges'] if n['code'] == bu_perf.GPU_OFF ][0]
        self.assertEqual( nudge['action'], { 'kind': 'set', 'values': { 'gpu_enabled': 1 } } )
        self.assertNotIn( bu_perf.GPU_OFF, self.codes( facts={ 'cuda_usable': True }, gpu_enabled=1 )[0] )
        self.assertNotIn( bu_perf.GPU_OFF, self.codes( facts={ 'cuda_usable': False }, gpu_enabled=0 )[0] )

    def test_not_optimized_then_optimized_then_changed( self ):
        self.assertIn( bu_perf.NOT_OPTIMIZED, self.codes()[0] )
        codes, ev = self.codes( optimized=[ ( 5.0, { 'variant': '640m', 'batch': 4 } ) ] )
        self.assertNotIn( bu_perf.NOT_OPTIMIZED, codes )
        self.assertNotIn( bu_perf.SETTINGS_CHANGED, codes )
        self.assertEqual( ev['optimized'], { 'at': 5.0, 'applied': True } )
        codes, _ = self.codes( optimized=[ ( 5.0, { 'variant': '640m', 'batch': 4 } ) ],
                               current={ 'variant': '320n', 'batch': 4 } )
        self.assertIn( bu_perf.SETTINGS_CHANGED, codes )

    def test_slower_than_usual( self ):
        steady = [ _row( i, detect=100.0 ) for i in range( 8 ) ]
        self.assertNotIn( bu_perf.SLOWER, self.codes( rows=steady )[0] )
        slowed = steady + [ _row( 20 + i, detect=400.0 ) for i in range( 3 ) ]
        codes, ev = self.codes( rows=slowed )
        self.assertIn( bu_perf.SLOWER, codes )

    def test_fingerprint( self ):
        a = { 'cpu': { 'model': 'C', 'threads': 16 }, 'gpu': { 'name': 'G' }, 'memory': { 'total_mb': 15384 } }
        b = dict( a, memory={ 'total_mb': 15000 } )
        c = dict( a, gpu={ 'name': 'Other' } )
        self.assertEqual( bu_perf.hardware_fingerprint( a ), bu_perf.hardware_fingerprint( b ) )
        self.assertNotEqual( bu_perf.hardware_fingerprint( a ), bu_perf.hardware_fingerprint( c ) )
        self.assertIsNone( bu_perf.hardware_fingerprint( None ) )


class TestServe( _Base ):

    def test_shared_index_follows_the_stats_log( self ):
        import betautils_serve_perf as bu_serve_perf
        self.addCleanup( bu_serve_perf.close )
        other = os.path.join( self.root, 'other', 'stats.jsonl' )
        with mock.patch( 'betautils_log.configured_stats_path', return_value=self.stats ):
            first = bu_serve_perf.shared_index()
            self.assertIs( bu_serve_perf.shared_index(), first )
        with mock.patch( 'betautils_log.configured_stats_path', return_value=other ):
            self.assertEqual( bu_serve_perf.shared_index().path, os.path.join( self.root, 'other', 'perf.sqlite' ) )

    def test_providers_now( self ):
        import betautils_serve_perf as bu_serve_perf
        on = types.SimpleNamespace( gpu_enabled=1 )
        self.assertEqual( bu_serve_perf.providers_now( on, { 'cuda_usable': True } ), CUDA )
        self.assertEqual( bu_serve_perf.providers_now( on, { 'cuda_usable': False } ), CPU )
        self.assertEqual( bu_serve_perf.providers_now( types.SimpleNamespace( gpu_enabled=0 ),
                                                       { 'cuda_usable': True } ), CPU )

    @schemas.needs_jsonschema
    def test_schemas( self ):
        import betautils_serve_perf as bu_serve_perf
        self.addCleanup( bu_serve_perf.close )
        self.write_stats( [ _row( i, workers=5, cap=5 ) for i in range( 9 ) ] + [ _row( 30, height=2160 ) ] )
        facts = { 'cuda_usable': True, 'gpu': { 'name': 'G' } }
        with mock.patch( 'betautils_log.configured_stats_path', return_value=self.stats ), \
             mock.patch.object( bu_serve_perf, 'machine', return_value=( facts, 'fp' ) ), \
             mock.patch.object( bu_serve_perf, 'current_settings', return_value={ 'variant': '640m', 'batch': 4 } ):
            for gpu, expected in ( ( 1, { bu_perf.MEMORY_CAPPED, bu_perf.NOT_OPTIMIZED } ),
                                   ( 0, { bu_perf.GPU_OFF, bu_perf.NOT_OPTIMIZED } ) ):
                with mock.patch( 'betaconfig.gpu_enabled', gpu ):
                    answer = bu_serve_perf.perf_evidence( None, {} )
                self.assertEqual( { n['code'] for n in answer['nudges'] }, expected, gpu )
                schemas.validate_command_result( 'perf.evidence', answer )
            schemas.validate_command_result( 'perf.rebuild', bu_serve_perf.perf_rebuild( None, {} ) )


if __name__ == '__main__':
    unittest.main()
