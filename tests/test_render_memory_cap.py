"""
tests/test_render_memory_cap.py - render workers sized by memory as well as CPUs.

On 2026-10-26 a 16-CPU laptop with 15 GiB of RAM got 8 render workers
for a 4096x2160 file and the out-of-memory killer killed betatv twice;
the 1080p file before it rendered fine with the same 8. These tests pin
that the worker count now falls with frame size and available memory,
never below one, that it says why, and that a real render obeys it.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betautils_log as bu_log
import betautils_render as bu_render

GIB = 2**30


class _Config:
    """Pin the memory-cap settings for one test, restoring them after."""

    def __init__( self, **values ):
        self.values = values
        self.patches = []

    def __enter__( self ):
        for name, value in self.values.items():
            patch = mock.patch.object( betaconfig, name, value, create=True )
            patch.start()
            self.patches.append( patch )
        return self

    def __exit__( self, *exc ):
        for patch in reversed( self.patches ):
            patch.stop()
        return False


SHIPPED = dict( render_worker_mb_per_megapixel=200, render_memory_reserve_mb=2048, render_workers=0 )


class TestAvailableMemory( unittest.TestCase ):

    def _read( self, text ):
        with tempfile.NamedTemporaryFile( 'w', delete=False, suffix='meminfo' ) as handle:
            handle.write( text )
        try:
            return bu_render.available_memory_bytes( handle.name )
        finally:
            os.remove( handle.name )

    def test_reads_mem_available_in_bytes( self ):
        text = "MemTotal:       16000000 kB\nMemFree:  100 kB\nMemAvailable:   10485760 kB\nSwapFree: 9 kB\n"
        self.assertEqual( self._read( text ), 10 * GIB )

    def test_unknown_when_the_field_or_file_is_missing_or_garbled( self ):
        self.assertIsNone( self._read( "MemTotal: 1 kB\nMemFree: 1 kB\n" ) )
        self.assertIsNone( self._read( "MemAvailable: lots kB\n" ) )
        self.assertIsNone( bu_render.available_memory_bytes( '/no/such/meminfo' ) )


class TestMemoryWorkerCap( unittest.TestCase ):

    def test_the_2026_10_26_case_caps_4k_and_leaves_1080p_alone( self ):
        # Precondition: 8 CPU workers is what both files got that night.
        with _Config( **SHIPPED ):
            cap_4k, per_worker_4k = bu_render.memory_worker_cap( 4096, 2160, 10 * GIB )
            cap_1080, _ = bu_render.memory_worker_cap( 1920, 1080, 10 * GIB )
        self.assertLess( cap_4k, 8, "8 workers at 4K is what ran out of memory" )
        self.assertGreaterEqual( cap_1080, 8, "8 workers at 1080p rendered fine" )
        self.assertEqual( cap_4k, 4 )
        self.assertEqual( per_worker_4k, int( 200 * 4096 * 2160 / 1e6 * 2**20 ) )

    def test_the_reserve_comes_off_the_top( self ):
        with _Config( render_worker_mb_per_megapixel=100, render_memory_reserve_mb=1024 ):
            # 10 MP at 100 MB/MP is 1000 MB each; 5 GiB less 1 GiB leaves 4096 MB.
            self.assertEqual( bu_render.memory_worker_cap( 5000, 2000, 5 * GIB )[0], 4 )
        with _Config( render_worker_mb_per_megapixel=100, render_memory_reserve_mb=0 ):
            self.assertEqual( bu_render.memory_worker_cap( 5000, 2000, 5 * GIB )[0], 5 )

    def test_never_fewer_than_one_worker( self ):
        with _Config( **SHIPPED ):
            self.assertEqual( bu_render.memory_worker_cap( 4096, 2160, 1 * GIB )[0], 1 )
            self.assertEqual( bu_render.memory_worker_cap( 4096, 2160, 0 )[0], 1 )

    def test_no_cap_when_off_unknown_or_sizeless( self ):
        with _Config( render_worker_mb_per_megapixel=None, render_memory_reserve_mb=2048 ):
            self.assertEqual( bu_render.memory_worker_cap( 4096, 2160, 10 * GIB ), ( None, None ) )
        with _Config( **SHIPPED ):
            self.assertEqual( bu_render.memory_worker_cap( 4096, 2160, None ), ( None, None ) )
            self.assertEqual( bu_render.memory_worker_cap( 0, 2160, 10 * GIB ), ( None, None ) )


class TestPlanWorkerCount( unittest.TestCase ):

    def _plan( self, by_cpu, available, render_workers=0 ):
        logger = bu_log.get_logger()
        config = dict( SHIPPED, render_workers=render_workers )
        with _Config( **config ), \
             mock.patch.object( bu_render, 'resolve_worker_count', return_value=by_cpu ):
            with self.assertLogs( logger, level='DEBUG' ) as logged:
                result = bu_render.plan_worker_count( 8, 4096, 2160, available, logger )
        return result, [ ( record.levelname, record.getMessage() ) for record in logged.records ]

    def test_memory_lowers_the_auto_count_and_says_so( self ):
        ( count, cap ), records = self._plan( 8, 10 * GIB )
        self.assertEqual( ( count, cap ), ( 4, 4 ) )
        self.assertEqual( records[0][0], 'INFO' )
        self.assertIn( "memory caps workers at 4 of 8", records[0][1] )
        self.assertIn( "4096x2160", records[0][1] )

    def test_an_explicit_render_workers_is_capped_with_a_warning( self ):
        ( count, _cap ), records = self._plan( 8, 10 * GIB, render_workers=8 )
        self.assertEqual( count, 4 )
        self.assertEqual( records[0][0], 'WARNING' )
        self.assertIn( "render_workers asks for 8 but memory holds 4", records[0][1] )

    def test_plenty_of_memory_changes_nothing( self ):
        ( count, cap ), records = self._plan( 8, 64 * GIB )
        self.assertEqual( count, 8 )
        self.assertGreater( cap, 8 )
        self.assertEqual( [ level for level, _message in records ], [ 'DEBUG' ] )

    def test_exactly_enough_memory_keeps_the_cpu_count( self ):
        per_worker = int( 200 * 4096 * 2160 / 1e6 * 2**20 )
        ( count, _cap ), records = self._plan( 4, 2048 * 2**20 + 4 * per_worker )
        self.assertEqual( count, 4 )
        self.assertEqual( [ level for level, _message in records ], [ 'DEBUG' ],
            "a cap that only equals the CPU count lowers nothing and must not say it did" )

    def test_unknown_memory_falls_back_to_the_cpu_count( self ):
        ( count, cap ), records = self._plan( 8, None )
        self.assertEqual( ( count, cap ), ( 8, None ) )
        self.assertIn( "MemAvailable unknown", records[0][1] )


@unittest.skipUnless( shutil.which( 'ffmpeg' ), "needs ffmpeg" )
class TestRenderObeysTheCap( unittest.TestCase ):
    """A real multi-chunk render, with memory set so the cap binds at one worker."""

    def setUp( self ):
        import betautils_video as bu_video
        self.tmpdir = tempfile.mkdtemp( prefix='betasuite-memcap-' )
        source_path = os.path.join( self.tmpdir, 'source.mp4' )
        subprocess.run( [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                          '-i', 'testsrc2=size=64x48:rate=10:duration=40',
                          '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
                          source_path ], check=True )
        self.source = bu_video.open_video_source( source_path, 'memcaphash' )

    def tearDown( self ):
        self.source.release()
        shutil.rmtree( self.tmpdir, ignore_errors=True )

    def _render( self, available ):
        config = dict( render_worker_mb_per_megapixel=1_000_000, render_memory_reserve_mb=0,
                       render_workers=4, render_chunk_seconds=15, encode_video_codec='libx264' )
        with _Config( **config ), \
             mock.patch.object( bu_render, 'available_memory_bytes', return_value=available ):
            return bu_render.render_video(
                self.source, [], os.path.join( self.tmpdir, 'v.mkv' ),
                os.path.join( self.tmpdir, 'v.mp4' ), 'ultrafast', False, False, 0.0, 20,
                bu_log.get_logger() )

    def test_the_render_uses_the_capped_count_and_records_it( self ):
        # 64x48 at 1,000,000 MB/MP is about 2.9 GiB a worker; 4 GiB holds one.
        stats = self._render( 4 * GIB )
        self.assertGreaterEqual( stats['chunks'], 2, "precondition: several chunks to share out" )
        self.assertEqual( stats['render_workers'], 1 )
        self.assertEqual( stats['memory_worker_cap'], 1 )
        self.assertEqual( stats['memory_available_mb'], 4096 )

    def test_without_a_reading_the_render_is_uncapped( self ):
        stats = self._render( None )
        self.assertEqual( stats['render_workers'], min( 4, stats['chunks'] ) )
        self.assertIsNone( stats['memory_worker_cap'] )
        self.assertIsNone( stats['memory_available_mb'] )


if __name__ == '__main__':
    unittest.main()
