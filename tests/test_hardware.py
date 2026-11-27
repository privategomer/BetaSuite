"""
test_hardware.py - hardware detection (betautils_hardware.machine) and
the settings it picks without measuring (safe_settings), on synthetic
machines.

What each test prevents:
  - CPU, memory or battery facts read wrong from /proc and /sys, or a
    crash on a machine where they are missing
  - the GPU reported from nvidia-smi alone, or its memory as text
  - disks reported per folder when two kinds share one disk, or a path
    leaking into the answer
  - a safe setting that turns the GPU on without a usable one, leaves
    it on when nothing can use it, or proposes a value already in effect
  - hw.detect's answer drifting from protocol/v1's schema
"""

import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import betautils_hardware as bu_hardware
import betautils_serve_resources as bu_serve_resources

from tests import protocol_schemas as schemas

MEMINFO = """MemTotal:       16000000 kB
MemFree:         1000000 kB
MemAvailable:    6000000 kB
SwapTotal:       4096000 kB
"""

CPUINFO = """processor\t: 0
vendor_id\t: Example
model name\t: Example CPU 8-Core
processor\t: 1
model name\t: Example CPU 8-Core
"""


def _write( path, text ):
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    with open( path, 'w', encoding='ascii' ) as handle:
        handle.write( text )


class _Machine:
    """A temporary /proc and /sys."""

    def __init__( self, battery=True ):
        self.root = tempfile.mkdtemp( prefix='hw-' )
        self.proc = os.path.join( self.root, 'proc' )
        self.sys = os.path.join( self.root, 'sys' )
        _write( os.path.join( self.proc, 'meminfo' ), MEMINFO )
        _write( os.path.join( self.proc, 'cpuinfo' ), CPUINFO )
        _write( os.path.join( self.sys, 'class', 'power_supply', 'AC0', 'type' ), 'Mains\n' )
        if battery:
            _write( os.path.join( self.sys, 'class', 'power_supply', 'BAT0', 'type' ), 'Battery\n' )

    def close( self ):
        shutil.rmtree( self.root, ignore_errors=True )


def _smi( text, code=0 ):
    return lambda _command: ( code, text )


class TestLinuxReaders( unittest.TestCase ):

    def setUp( self ):
        self.machine = _Machine()
        self.addCleanup( self.machine.close )

    def test_meminfo_in_mb( self ):
        found = bu_hardware.meminfo_mb( self.machine.proc )
        self.assertEqual( found['MemTotal'], 15625 )
        self.assertEqual( found['MemAvailable'], 5859 )
        self.assertEqual( found['SwapTotal'], 4000 )

    def test_cpu_model( self ):
        self.assertEqual( bu_hardware.cpu_model_linux( self.machine.proc ), 'Example CPU 8-Core' )

    def test_battery( self ):
        self.assertTrue( bu_hardware.battery_present_linux( self.machine.sys ) )
        mains_only = _Machine( battery=False )
        self.addCleanup( mains_only.close )
        self.assertFalse( bu_hardware.battery_present_linux( mains_only.sys ) )

    def test_missing_sources_are_unknown_not_a_crash( self ):
        nowhere = os.path.join( self.machine.root, 'nowhere' )
        self.assertEqual( bu_hardware.meminfo_mb( nowhere ), {} )
        self.assertIsNone( bu_hardware.cpu_model_linux( nowhere ) )
        self.assertFalse( bu_hardware.battery_present_linux( nowhere ) )


class TestMachine( unittest.TestCase ):

    def setUp( self ):
        self.machine = _Machine()
        self.addCleanup( self.machine.close )

    def _facts( self, smi, providers, **kw ):
        return bu_hardware.machine( platform_name='linux', proc_root=self.machine.proc, sys_root=self.machine.sys,
                                    run=smi, providers=providers, cpu_count=16, disk_kinds={}, **kw )

    def test_gpu_machine( self ):
        facts = self._facts( _smi( 'Example GPU, 580.10, 8192\n' ),
                             [ 'CUDAExecutionProvider', 'CPUExecutionProvider' ] )
        self.assertEqual( facts['gpu'], { 'name': 'Example GPU', 'driver': '580.10', 'vram_mb': 8192 } )
        self.assertTrue( facts['cuda_usable'] )
        self.assertEqual( facts['cpu'], { 'model': 'Example CPU 8-Core', 'threads': 16 } )
        self.assertEqual( facts['memory'], { 'total_mb': 15625, 'available_mb': 5859, 'swap_mb': 4000 } )
        self.assertTrue( facts['battery'] )

    def test_gpu_without_cuda_provider_is_not_usable( self ):
        facts = self._facts( _smi( 'Example GPU, 580.10, 8192\n' ), [ 'CPUExecutionProvider' ] )
        self.assertIsNotNone( facts['gpu'] )
        self.assertFalse( facts['cuda_usable'] )

    def test_no_nvidia_smi( self ):
        facts = self._facts( lambda _c: ( None, 'not found' ), [ 'CUDAExecutionProvider' ] )
        self.assertIsNone( facts['gpu'] )
        self.assertFalse( facts['cuda_usable'] )

    def test_unparsable_vram_is_unknown( self ):
        facts = self._facts( _smi( 'Example GPU, 580.10, [N/A]\n' ), [] )
        self.assertIsNone( facts['gpu']['vram_mb'] )

    def test_unknown_platform_has_no_facts_but_answers( self ):
        facts = bu_hardware.machine( platform_name='plan9', run=_smi( '', 9 ), providers=[], disk_kinds={} )
        self.assertIsNone( facts['cpu']['model'] )
        self.assertIsNone( facts['memory']['total_mb'] )
        self.assertFalse( facts['battery'] )


class TestDisks( unittest.TestCase ):

    def test_kinds_on_one_disk_share_a_volume_and_no_paths_leak( self ):
        root = tempfile.mkdtemp( prefix='hw-disk-' )
        self.addCleanup( shutil.rmtree, root, True )
        os.makedirs( os.path.join( root, 'a' ) )
        os.makedirs( os.path.join( root, 'b' ) )
        devices = { os.path.join( root, 'a' ): 1, os.path.join( root, 'b' ): 2, root: 1 }
        stat = lambda path: types.SimpleNamespace( st_dev=devices[path] )
        usage = lambda path: ( 1000, 400, 600 ) if devices[path] == 1 else ( 50, 40, 10 )
        kinds = { 'cache': os.path.join( root, 'a' ), 'data': os.path.join( root, 'not-yet', 'deeper' ),
                  'outputs': os.path.join( root, 'b' ) }
        found = bu_hardware.disks( kinds, usage=usage, stat=stat )
        by_kind = { row['kind']: row for row in found }
        self.assertEqual( by_kind['cache']['volume'], by_kind['data']['volume'] )
        self.assertNotEqual( by_kind['cache']['volume'], by_kind['outputs']['volume'] )
        self.assertEqual( by_kind['outputs']['free_bytes'], 10 )
        self.assertEqual( by_kind['cache']['total_bytes'], 1000 )
        for row in found:
            self.assertEqual( set( row ), { 'kind', 'volume', 'free_bytes', 'total_bytes' } )

    def test_unreadable_folder_is_left_out( self ):
        def stat( _path ):
            raise OSError( 'denied' )
        self.assertEqual( bu_hardware.disks( { 'cache': '/' }, usage=lambda p: ( 1, 0, 1 ), stat=stat ), [] )


class TestSafeSettings( unittest.TestCase ):

    def _config( self, gpu ):
        return types.SimpleNamespace( gpu_enabled=gpu )

    def test_usable_gpu_turned_on( self ):
        facts = { 'cuda_usable': True, 'gpu': { 'name': 'Example GPU' }, 'providers': [] }
        picks = bu_hardware.safe_settings( facts, self._config( 0 ) )
        self.assertEqual( picks, [ { 'setting': 'gpu_enabled', 'value': 1, 'current': 0,
                                     'code': 'hw.gpu_usable', 'params': { 'gpu': 'Example GPU' } } ] )

    def test_unusable_gpu_turned_off( self ):
        facts = { 'cuda_usable': False, 'gpu': { 'name': 'Example GPU' }, 'providers': [ 'CPUExecutionProvider' ] }
        picks = bu_hardware.safe_settings( facts, self._config( 1 ) )
        self.assertEqual( [ ( p['setting'], p['value'], p['code'] ) for p in picks ],
                          [ ( 'gpu_enabled', 0, 'hw.gpu_not_usable' ) ] )
        self.assertEqual( picks[0]['params'], { 'provider': False, 'gpu': True } )

    def test_nothing_when_already_right( self ):
        self.assertEqual( bu_hardware.safe_settings( { 'cuda_usable': True }, self._config( 1 ) ), [] )
        self.assertEqual( bu_hardware.safe_settings( { 'cuda_usable': False }, self._config( 0 ) ), [] )

    def test_batch_size_is_never_picked( self ):
        facts = { 'cuda_usable': True, 'gpu': { 'name': 'G', 'vram_mb': 24000 } }
        settings = { pick['setting'] for pick in bu_hardware.safe_settings( facts, self._config( 0 ) ) }
        self.assertEqual( settings, { 'gpu_enabled' } )


@schemas.needs_jsonschema
class TestSchema( unittest.TestCase ):

    def test_hw_detect_result( self ):
        machine = _Machine()
        self.addCleanup( machine.close )
        facts = bu_hardware.machine( platform_name='linux', proc_root=machine.proc, sys_root=machine.sys,
                                     run=_smi( 'Example GPU, 580.10, 8192\n' ),
                                     providers=[ 'CUDAExecutionProvider' ], cpu_count=16 )
        with mock.patch.object( bu_hardware, 'machine', return_value=facts ):
            result = bu_serve_resources.hw_detect( None, {} )
        schemas.validate_command_result( 'hw.detect', result )
        schemas.validate_command_data( 'hw.detect', {} )

    def test_no_gpu_result( self ):
        facts = bu_hardware.machine( platform_name='plan9', run=_smi( '', 9 ), providers=[], disk_kinds={} )
        facts['safe_settings'] = []
        schemas.validate_command_result( 'hw.detect', facts )


if __name__ == '__main__':
    unittest.main()
