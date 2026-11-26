"""
test_resources.py - the resource signals the governor reads, on
synthetic machines, and resources.watch.

What each test prevents:
  - memory pressure, free memory or battery read wrong, or "on battery"
    reported for a laptop held at a charge limit while plugged in
  - this engine's CPU (with its ffmpeg children, live or finished)
    counted as other programs' load, which would throttle the run for
    its own work
  - a GPU reading kept after nvidia-smi stopped printing
  - a failing reader or listener stopping the sampler
  - the sampler left running with nobody using it, or events sent while
    the UI is not watching
  - samples and answers drifting from protocol/v1's schemas
"""

import io
import os
import shutil
import tempfile
import threading
import types
import unittest

import betautils_protocol as bu_proto
import betautils_resources as bu_resources
import betautils_serve_resources as bu_serve_resources

from tests import protocol_schemas as schemas


def _write( path, text ):
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    with open( path, 'w', encoding='ascii' ) as handle:
        handle.write( text )


def _stat_line( pid, name, ppid, utime, stime, cutime=0, cstime=0, rss=100 ):
    # pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt
    # utime stime cutime cstime priority nice threads itreal starttime vsize rss
    return '%d (%s) S %d 1 1 0 -1 0 0 0 0 0 %d %d %d %d 20 0 1 0 100 1000 %d\n'%(
        pid, name, ppid, utime, stime, cutime, cstime, rss )


class _Proc:

    def __init__( self, children_files=True ):
        self.root = tempfile.mkdtemp( prefix='res-' )
        self.proc = os.path.join( self.root, 'proc' )
        self.sys = os.path.join( self.root, 'sys' )
        self.children_files = children_files
        os.makedirs( self.proc )

    def close( self ):
        shutil.rmtree( self.root, ignore_errors=True )

    def cpu( self, busy, idle ):
        _write( os.path.join( self.proc, 'stat' ), 'cpu  %d 0 0 %d 0 0 0 0 0 0\ncpu0 1 0 0 1\n'%( busy, idle ) )

    def process( self, pid, ppid, utime, stime=0, cutime=0, children=(), name='python3 x', rss=100 ):
        _write( os.path.join( self.proc, str( pid ), 'stat' ),
                _stat_line( pid, name, ppid, utime, stime, cutime, 0, rss ) )
        if self.children_files:
            _write( os.path.join( self.proc, str( pid ), 'task', str( pid ), 'children' ),
                    ' '.join( str( c ) for c in children ) + ( ' ' if children else '' ) )

    def psi( self, avg10 ):
        _write( os.path.join( self.proc, 'pressure', 'memory' ),
                'some avg10=%.2f avg60=0.00 avg300=0.00 total=1\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=1\n'%(
                    avg10, ) )

    def meminfo( self, total_kb, available_kb ):
        _write( os.path.join( self.proc, 'meminfo' ),
                'MemTotal: %d kB\nMemAvailable: %d kB\nSwapTotal: 0 kB\n'%( total_kb, available_kb ) )

    def supply( self, name, kind, **files ):
        base = os.path.join( self.sys, 'class', 'power_supply', name )
        _write( os.path.join( base, 'type' ), kind + '\n' )
        for key, value in files.items():
            _write( os.path.join( base, key ), '%s\n'%( value, ) )


class TestMemoryAndPower( unittest.TestCase ):

    def setUp( self ):
        self.m = _Proc()
        self.addCleanup( self.m.close )

    def test_psi( self ):
        self.m.psi( 12.5 )
        self.assertEqual( bu_resources.psi_some_avg10( self.m.proc ), 12.5 )

    def test_psi_missing( self ):
        self.assertIsNone( bu_resources.psi_some_avg10( self.m.proc ) )

    def test_plugged_in_at_a_charge_limit_is_not_on_battery( self ):
        self.m.supply( 'AC0', 'Mains', online=1 )
        self.m.supply( 'BAT0', 'Battery', status='Discharging', capacity=86 )
        self.assertEqual( bu_resources.power_linux( self.m.sys ), ( False, 86 ) )

    def test_unplugged_is_on_battery( self ):
        self.m.supply( 'AC0', 'Mains', online=0 )
        self.m.supply( 'BAT0', 'Battery', status='Discharging', capacity=40 )
        self.assertEqual( bu_resources.power_linux( self.m.sys ), ( True, 40 ) )

    def test_without_a_mains_entry_the_battery_status_decides( self ):
        self.m.supply( 'BAT0', 'Battery', status='Discharging', capacity=55 )
        self.assertEqual( bu_resources.power_linux( self.m.sys ), ( True, 55 ) )
        self.m.supply( 'BAT0', 'Battery', status='Charging', capacity=55 )
        self.assertEqual( bu_resources.power_linux( self.m.sys ), ( False, 55 ) )

    def test_lowest_of_two_batteries( self ):
        self.m.supply( 'AC0', 'Mains', online=0 )
        self.m.supply( 'BAT0', 'Battery', status='Discharging', capacity=70 )
        self.m.supply( 'BAT1', 'Battery', status='Discharging', capacity=20 )
        self.assertEqual( bu_resources.power_linux( self.m.sys ), ( True, 20 ) )

    def test_absent_battery_is_ignored( self ):
        self.m.supply( 'AC0', 'Mains', online=0 )
        self.m.supply( 'BAT0', 'Battery', present=0, status='Discharging', capacity=5 )
        self.assertEqual( bu_resources.power_linux( self.m.sys ), ( False, None ) )

    def test_desktop_has_no_battery( self ):
        self.m.supply( 'AC0', 'Mains', online=1 )
        self.assertEqual( bu_resources.power_linux( self.m.sys ), ( False, None ) )
        self.assertEqual( bu_resources.power_linux( os.path.join( self.m.root, 'none' ) ), ( False, None ) )


class TestCpu( unittest.TestCase ):

    def test_meter_splits_own_from_other( self ):
        meter = bu_resources.CpuMeter()
        self.assertEqual( meter.update( 100, 1000, 10 ), ( None, None, None ) )
        # 1000 more jiffies, 600 busy, 200 of them ours.
        self.assertEqual( meter.update( 700, 2000, 210 ), ( 60.0, 40.0, 20.0 ) )

    def test_waiting_for_disk_is_not_busy( self ):
        m = _Proc()
        self.addCleanup( m.close )
        # user 300, idle 500, iowait 200: busy is 300 of 1000.
        _write( os.path.join( m.proc, 'stat' ), 'cpu  300 0 0 500 200 0 0 0 0 0\n' )
        self.assertEqual( bu_resources._proc_stat_cpu( m.proc ), ( 300, 1000 ) )

    def test_meter_without_own_counter( self ):
        meter = bu_resources.CpuMeter()
        meter.update( 0, 0, None )
        self.assertEqual( meter.update( 500, 1000, None ), ( 50.0, None, None ) )

    def test_meter_never_negative( self ):
        meter = bu_resources.CpuMeter()
        meter.update( 100, 1000, 0 )
        cpu, other, own = meter.update( 200, 2000, 500 )   # own grew more than the machine (rounding)
        self.assertEqual( ( cpu, other, own ), ( 10.0, 0.0, 10.0 ) )
        self.assertEqual( meter.update( 200, 2000, 500 ), ( None, None, None ) )   # no time passed

    def _tree( self, children_files ):
        m = _Proc( children_files=children_files )
        self.addCleanup( m.close )
        m.process( 10, 1, utime=100, stime=50, cutime=30, children=( 11, 12 ) )
        m.process( 11, 10, utime=200, children=( 13, ) )
        m.process( 12, 10, utime=5 )
        m.process( 13, 11, utime=7 )
        m.process( 99, 1, utime=100000 )   # someone else
        return m

    def test_tree_counts_live_children_and_reaped_ones( self ):
        for files in ( True, False ):
            with self.subTest( children_files=files ):
                m = self._tree( files )
                cpu, rss = bu_resources.process_tree( m.proc, 10 )
                self.assertEqual( cpu, 100 + 50 + 30 + 200 + 5 + 7 )
                self.assertEqual( rss, 400 )

    def test_name_with_spaces_and_parens( self ):
        m = _Proc()
        self.addCleanup( m.close )
        m.process( 10, 1, utime=3, name='a ) b' )
        self.assertEqual( bu_resources.process_tree( m.proc, 10 ), ( 3, 100 ) )

    def test_unknown_pid( self ):
        m = _Proc()
        self.addCleanup( m.close )
        self.assertEqual( bu_resources.process_tree( m.proc, 10 ), ( None, None ) )

    def test_linux_reader_does_not_count_our_render_as_load( self ):
        m = _Proc()
        self.addCleanup( m.close )
        m.meminfo( 16 * 2**20, 4 * 2**20 )
        m.psi( 3.0 )
        m.cpu( busy=1000, idle=9000 )
        m.process( 10, 1, utime=500, children=( 11, ) )
        m.process( 11, 10, utime=0 )
        reader = bu_resources.LinuxReader( m.proc, m.sys, pid=10, page_size=4096 )
        reader.read()
        # Next second: the machine did 800 more busy jiffies of 1000, our ffmpeg child 700 of them.
        m.cpu( busy=1800, idle=9200 )
        m.process( 11, 10, utime=700 )
        found = reader.read()
        self.assertEqual( found['cpu_percent'], 80.0 )
        self.assertEqual( found['own_cpu_percent'], 70.0 )
        self.assertEqual( found['other_cpu_percent'], 10.0 )
        self.assertEqual( found['memory_available_mb'], 4096 )
        self.assertEqual( found['memory_pressure'], 3.0 )
        self.assertFalse( found['on_battery'] )


class _FakeProc:

    def __init__( self, lines ):
        self.stdout = io.StringIO( ''.join( lines ) )
        self.terminated = False

    def terminate( self ):
        self.terminated = True

    def wait( self, timeout=None ):
        return 0


class TestGpu( unittest.TestCase ):

    def test_parse( self ):
        self.assertEqual( bu_resources.parse_gpu_line( '67, 1209, 8192\n' ),
                          { 'gpu_percent': 67, 'vram_used_mb': 1209, 'vram_total_mb': 8192 } )
        self.assertEqual( bu_resources.parse_gpu_line( '[N/A], 10, 20' ),
                          { 'gpu_percent': None, 'vram_used_mb': 10, 'vram_total_mb': 20 } )
        self.assertEqual( bu_resources.parse_gpu_line( 'garbage' ), {} )

    def test_stream_keeps_the_last_line_while_fresh( self ):
        now = [ 100.0 ]
        proc = _FakeProc( [ '10, 1, 8\n', '20, 2, 8\n' ] )
        stream = bu_resources.GpuStream( interval_ms=1000, popen=lambda *a, **k: proc, clock=lambda: now[0] )
        self.assertTrue( stream.start() )
        stream._thread.join( timeout=2 )
        self.assertEqual( stream.latest(), { 'gpu_percent': 20, 'vram_used_mb': 2, 'vram_total_mb': 8 } )
        now[0] += 6.0   # nvidia-smi stopped printing
        self.assertEqual( stream.latest(), {} )
        stream.stop()
        self.assertTrue( proc.terminated )

    def test_stream_that_cannot_start( self ):
        def popen( *_a, **_k ):
            raise OSError( 'no such file' )
        stream = bu_resources.GpuStream( popen=popen )
        self.assertFalse( stream.start() )
        self.assertEqual( stream.latest(), {} )
        stream.stop()


class _Reader:

    def __init__( self, values=None, fail=False ):
        self.values = values or {}
        self.fail = fail

    def read( self ):
        if self.fail:
            raise RuntimeError( 'broken' )
        return dict( self.values )


class TestSampler( unittest.TestCase ):

    def test_snapshot_has_every_field( self ):
        sampler = bu_resources.Sampler( reader=_Reader( { 'memory_pressure': 4.0, 'not_a_field': 1 } ),
                                        clock=lambda: 5.0, folders={} )
        snapshot = sampler.sample()
        self.assertEqual( set( snapshot ), set( bu_resources.FIELDS ) )
        self.assertEqual( snapshot['memory_pressure'], 4.0 )
        self.assertEqual( snapshot['time'], 5.0 )
        self.assertIsNone( snapshot['on_battery'] )

    def test_failing_reader_and_listener_do_not_stop_it( self ):
        got = []
        sampler = bu_resources.Sampler( reader=_Reader( fail=True ), folders={}, logger=types.SimpleNamespace(
            debug=lambda *a, **k: None, warning=lambda *a, **k: None ) )
        sampler.add_listener( lambda s: 1 / 0 )
        sampler.add_listener( got.append )
        sampler.sample()
        self.assertEqual( len( got ), 1 )

    def test_disk_read_every_few_samples( self ):
        calls = []

        def usage( path ):
            calls.append( path )
            return types.SimpleNamespace( free=123 )
        sampler = bu_resources.Sampler( reader=_Reader(), folders={ 'cache': '/' }, usage=usage, disk_every=3 )
        values = [ sampler.sample()['cache_free_bytes'] for _ in range( 4 ) ]
        self.assertEqual( values, [ 123 ] * 4 )
        self.assertEqual( len( calls ), 2 )

    def test_signals_rejects_unknown_names( self ):
        with self.assertRaises( KeyError ):
            bu_resources.signals( nonsense=1 )

    def test_thread_samples_until_stopped( self ):
        seen = threading.Event()
        sampler = bu_resources.Sampler( reader=_Reader(), folders={}, interval=0.01 )
        sampler.add_listener( lambda s: seen.set() )
        sampler.start()
        self.assertTrue( seen.wait( 2 ) )
        sampler.stop()
        self.assertIsNone( sampler._thread )


class _FakeSampler:

    def __init__( self ):
        self.listeners = []
        self.started = self.stopped = False
        self.latest = None

    def add_listener( self, listener ):
        self.listeners.append( listener )

    def start( self ):
        self.started = True

    def stop( self ):
        self.stopped = True

    def sample( self ):
        self.latest = bu_resources.signals( time=1.0 )
        for listener in self.listeners:
            listener( self.latest )
        return self.latest


class TestResources( unittest.TestCase ):

    def setUp( self ):
        self.sent = []
        self.made = []

        def make():
            self.made.append( _FakeSampler() )
            return self.made[-1]
        self.resources = bu_serve_resources.Resources( lambda n, d: self.sent.append( ( n, d ) ), None,
                                                       make_sampler=make )

    def test_runs_while_used_and_stops_with_the_last_user( self ):
        self.resources.use( 'job' )
        self.resources.use( 'watch' )
        self.assertEqual( len( self.made ), 1 )
        self.resources.release( 'watch' )
        self.assertFalse( self.made[0].stopped )
        self.resources.release( 'job' )
        self.assertTrue( self.made[0].stopped )
        self.assertIsNone( self.resources.sampler )

    def test_events_only_while_watched( self ):
        sampler = self.resources.use( 'job' )
        sampler.sample()
        self.assertEqual( self.sent, [] )
        self.resources.use( 'watch' )
        sampler.sample()
        self.assertEqual( [ name for name, _ in self.sent ], [ 'resource.sample' ] )

    def test_watch_command( self ):
        server = types.SimpleNamespace( resources=self.resources, send_event=None, _logger=None )
        answer = bu_serve_resources.resources_watch( server, { 'on': True } )
        self.assertTrue( answer['watching'] )
        self.assertEqual( answer['sample']['time'], 1.0 )
        self.assertEqual( bu_serve_resources.resources_watch( server, { 'on': False } ),
                          { 'watching': False, 'sample': None } )
        self.assertTrue( self.made[0].stopped )

    def test_watch_needs_a_boolean( self ):
        server = types.SimpleNamespace( resources=self.resources )
        for data, reason in ( ( {}, 'missing' ), ( { 'on': 1 }, 'type' ) ):
            with self.assertRaises( bu_proto.ProtocolError ) as caught:
                bu_serve_resources.resources_watch( server, data )
            self.assertEqual( caught.exception.error, { 'code': 'command.bad_params',
                                                        'params': { 'field': 'on', 'reason': reason } } )


@schemas.needs_jsonschema
class TestSchemas( unittest.TestCase ):

    def test_sample_event( self ):
        m = _Proc()
        self.addCleanup( m.close )
        m.meminfo( 16 * 2**20, 4 * 2**20 )
        m.cpu( 1, 1 )
        m.process( 10, 1, utime=1 )
        sampler = bu_resources.Sampler( reader=bu_resources.LinuxReader( m.proc, m.sys, pid=10 ),
                                        folders={ 'cache': m.root },
                                        gpu=types.SimpleNamespace( latest=lambda: { 'gpu_percent': 5,
                                                                                    'vram_used_mb': 1,
                                                                                    'vram_total_mb': 2 } ) )
        schemas.validate_event_data( 'resource.sample', sampler.sample() )
        schemas.validate_event_data( 'resource.sample', bu_resources.signals( time=0.0 ) )

    def test_watch( self ):
        schemas.validate_command_data( 'resources.watch', { 'on': True } )
        schemas.validate_command_result( 'resources.watch', { 'watching': False, 'sample': None } )
        schemas.validate_command_result( 'resources.watch', { 'watching': True,
                                                              'sample': bu_resources.signals( time=2.0 ) } )


if __name__ == '__main__':
    unittest.main()
