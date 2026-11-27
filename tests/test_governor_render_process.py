"""
test_governor_render_process.py - the governor acting on a real render
(ffmpeg, a synthetic clip) mid-run, and the output it leaves.

What each test prevents:
  - a governor that decides but does not act: render workers that do not
    drop when memory runs short and come back after, or chunk encoders
    (ffmpeg) that keep their priority when the decision lowers it, or
    stay lowered after it returns to normal
  - a throttle that changes what is rendered: the governed output must
    decode to the same frames as an ungoverned render of the same input
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import betaconfig
import betautils_governor_run as bu_gov_run
import betautils_render as bu_render
import betautils_signals as bu_signals
import betautils_video as bu_video
from betautils_resources import signals as S

IDLE = dict( memory_pressure=0.0, memory_available_mb=8000, other_cpu_percent=2.0, on_battery=False )
SEVERE_AND_BUSY = dict( IDLE, memory_pressure=80.0, other_cpu_percent=90.0 )


def _nice( pid ):
    try:
        with open( '/proc/%d/stat'%( pid, ), 'r', encoding='ascii' ) as handle:
            return int( handle.read().rsplit( ')', 1 )[1].split()[16] )
    except OSError:
        return None


def _framemd5( path ):
    done = subprocess.run( [ 'ffmpeg', '-loglevel', 'error', '-i', path, '-map', '0:v', '-f', 'framemd5', '-' ],
                           check=True, capture_output=True, text=True )
    return [ line.split( ',' )[-1].strip() for line in done.stdout.splitlines() if not line.startswith( '#' ) ]


@unittest.skipUnless( shutil.which( 'ffmpeg' ) and sys.platform.startswith( 'linux' ),
                      'needs ffmpeg, and Linux to read the encoders\' priority' )
class TestGovernedRender( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.tmpdir = tempfile.mkdtemp( prefix='betasuite-governor-' )
        cls.source_path = os.path.join( cls.tmpdir, 'source.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error',
              '-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=10:duration=12',
              '-f', 'lavfi', '-i', 'sine=frequency=440:duration=12',
              '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
              '-c:a', 'aac', '-shortest', cls.source_path ],
            check=True )

    @classmethod
    def tearDownClass( cls ):
        shutil.rmtree( cls.tmpdir, ignore_errors=True )

    def setUp( self ):
        bu_signals.clear()
        saved = { name: getattr( betaconfig, name ) for name in ( 'render_workers', 'render_chunk_seconds' ) }
        self.addCleanup( lambda: [ setattr( betaconfig, k, v ) for k, v in saved.items() ] )
        betaconfig.render_workers = 4
        betaconfig.render_chunk_seconds = 1

    def _render( self, tag, runtime=None, wrap=None ):
        source = bu_video.open_video_source( self.source_path, 'governor-test' )
        boxes = [ { 'start': 0.0, 'end': 20.0, 't': 10.0, 'x': 20, 'y': 20, 'w': 40, 'h': 30,
                    'censor_style': { 'type': 'blur', 'method': 'gaussian', 'strength': 10 },
                    'censor_shape': 'box', 'censor_sticker_seed': 0.5,
                    'label': 'exposed_breast', 'score': 0.9 } ]
        final = os.path.join( self.tmpdir, '%s.mp4'%( tag, ) )
        try:
            with bu_gov_run.attached( runtime, types.SimpleNamespace() ):
                if wrap is None:
                    stats = bu_render.render_video( source, boxes, os.path.join( self.tmpdir, '%s.mkv'%( tag, ) ),
                                                    final, 'ultrafast', False, False, 0.0, 20 )
                else:
                    with wrap():
                        stats = bu_render.render_video( source, boxes,
                                                        os.path.join( self.tmpdir, '%s.mkv'%( tag, ) ),
                                                        final, 'ultrafast', False, False, 0.0, 20 )
        finally:
            source.release()
        return final, stats

    def test_workers_and_encoder_priority_change_mid_render_and_output_is_unchanged( self ):
        clock = [ 100.0 ]
        runtime = bu_gov_run.Runtime( clock=lambda: clock[0], sleep=lambda _s: None )

        def hold( values, seconds ):
            for _ in range( seconds ):
                clock[0] += 1.0
                runtime.on_sample( S( **values ) )

        lock = threading.Lock()
        running = [ 0 ]
        started = {}          # chunk index -> chunks running when it started (itself included)
        encoder_nice = {}     # chunk index -> its ffmpeg encoder's nice value
        first = threading.Barrier( 4, timeout=30 )
        later = threading.Barrier( 2, timeout=10 )
        later_met = []
        real_chunk = bu_render.render_one_chunk
        real_popen = subprocess.Popen
        chunk_of_thread = {}

        def chunk( plan, *args, **kwargs ):
            with lock:
                running[0] += 1
                started[plan.index] = running[0]
                chunk_of_thread[threading.get_ident()] = plan.index
            try:
                if plan.index < 4:
                    first.wait()
                    if plan.index == 3:
                        hold( SEVERE_AND_BUSY, 6 )
                if plan.index == 6:
                    hold( IDLE, 120 )
                if plan.index in ( 7, 8 ):
                    try:
                        later.wait()          # met only if two may run at once again
                        later_met.append( plan.index )
                    except threading.BrokenBarrierError:
                        pass
                return real_chunk( plan, *args, **kwargs )
            finally:
                with lock:
                    running[0] -= 1

        def popen( command, *args, **kwargs ):
            process = real_popen( command, *args, **kwargs )
            index = chunk_of_thread.get( threading.get_ident() )
            if index is not None and '-f' in command and 'rawvideo' in command:
                encoder_nice[index] = _nice( process.pid )
            return process

        def wrap():
            import contextlib
            stack = contextlib.ExitStack()
            stack.enter_context( mock.patch.object( bu_render, 'render_one_chunk', chunk ) )
            stack.enter_context( mock.patch.object( bu_render.subprocess, 'Popen', popen ) )
            return stack

        base = os.getpriority( os.PRIO_PROCESS, 0 )
        low = max( base, bu_gov_run.LINUX_LOW_NICE )
        governed, stats = self._render( 'governed', runtime, wrap )
        self.assertGreaterEqual( stats['chunks'], 10 )
        self.assertEqual( stats['render_workers'], 4 )

        self.assertEqual( max( started[i] for i in range( 4 ) ), 4, "four chunks at once before the throttle" )
        for index in ( 4, 5, 6 ):
            self.assertEqual( started[index], 1, "chunk %d ran alone under the memory guard"%( index, ) )
            self.assertEqual( encoder_nice.get( index ), low, "chunk %d's encoder was lowered"%( index, ) )
        for index in range( 4 ):
            self.assertEqual( encoder_nice.get( index ), base )
        self.assertEqual( sorted( later_met ), [ 7, 8 ], "several chunks at once again after the all-clear" )
        later = [ i for i in started if i > 6 ]
        self.assertTrue( later )
        for index in later:
            self.assertEqual( encoder_nice.get( index ), base, "chunk %d's encoder back at normal"%( index, ) )

        plain, _ = self._render( 'plain' )
        self.assertEqual( _framemd5( governed ), _framemd5( plain ),
                          "the governed render decodes to the same frames" )

    def test_each_chunk_encoder_gets_the_priority_arguments( self ):
        seen = []
        real_popen = subprocess.Popen

        def popen( command, *args, **kwargs ):
            if '-f' in command and 'rawvideo' in command:
                seen.append( kwargs.get( 'start_new_session' ) )
            return real_popen( command, *args, **kwargs )

        betaconfig.render_chunk_seconds = 4
        with mock.patch.object( bu_gov_run, 'child_popen_kwargs', return_value={ 'start_new_session': False } ), \
             mock.patch.object( bu_render.subprocess, 'Popen', popen ):
            self._render( 'kwargs' )
        self.assertTrue( seen )
        self.assertEqual( set( seen ), { False } )


if __name__ == '__main__':
    unittest.main()
