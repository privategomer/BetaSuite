"""
test_catalog_strip.py - the player's scrub strip.

Covers the two passes on synthetic clips (keyframes only from a
short-GOP clip, an even pass after a long-GOP one), the frame cap and
its spacing, the privacy rule (a source's strip is only ever written
blurred, at the source thumbnail's size), the cache key following the
file, the shell's path check, and the service: an answer at once,
passes built in the background and announced, the even pass waiting
while a run is on, a failed pass not retried, and catalog.strip through
the server (clear only for a present output).
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

import betautils_catalog_strip as bu_strip
import betautils_catalog_thumbs as bu_thumbs
import betautils_serve_strip as bu_serve_strip

WAIT = 30


def _ffmpeg_available():
    return shutil.which( 'ffmpeg' ) is not None and shutil.which( 'ffprobe' ) is not None


def make_clip( path, seconds=12, gop=300, size='320x180', rate=30 ):
    subprocess.run( [ 'ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=%s:rate=%d'%( size, rate ),
                      '-t', str( seconds ), '-c:v', 'libx264', '-preset', 'ultrafast', '-g', str( gop ),
                      '-sc_threshold', '0', path ], check=True )


class _CacheTest( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-strip-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.cache = os.path.join( self.dir, 'cache', 'strips' )
        patcher = mock.patch.object( bu_strip, 'folder', return_value=self.cache )
        patcher.start()
        self.addCleanup( patcher.stop )

    def facts( self, seconds=12, width=320, height=180 ):
        return { 'duration': float( seconds ), 'fps': 30.0, 'width': width, 'height': height }


class TestPureParts( unittest.TestCase ):

    def test_tile_size_keeps_the_aspect_and_is_even( self ):
        self.assertEqual( bu_strip.tile_size( 3840, 2160 ), ( 160, 90 ) )
        self.assertEqual( bu_strip.tile_size( 1080, 1920 ), ( 160, 284 ) )
        self.assertEqual( bu_strip.tile_size( 100, 1 ), ( 160, 2 ) )
        self.assertEqual( bu_strip.tile_size( 160, 91 ), ( 160, 92 ), "an odd height is made even, as ffmpeg needs" )

    def test_the_selector_keeps_one_frame_per_slot_and_at_most_the_limit( self ):
        selector = bu_strip.Selector( 10.0, limit=5 )
        for tenth in range( 100 ):
            selector.offer( tenth / 10.0, tenth )
        self.assertEqual( [ t for t, _ in selector.kept ], [ 0.0, 2.0, 4.0, 6.0, 8.0 ] )

    def test_the_selector_keeps_everything_sparse( self ):
        selector = bu_strip.Selector( 600.0, limit=600 )
        for seconds in ( 0.0, 4.0, 8.0 ):
            selector.offer( seconds, None )
        self.assertEqual( len( selector.kept ), 3 )

    def test_the_selector_stops_at_the_limit_even_with_no_duration( self ):
        selector = bu_strip.Selector( 0.0, limit=3 )
        for seconds in range( 10 ):
            selector.offer( float( seconds ), None )
        self.assertEqual( len( selector.kept ), 3 )

    def test_an_even_pass_is_needed_only_when_keyframes_are_sparse( self ):
        self.assertTrue( bu_strip.needs_even_pass( [ 0.0, 10.0 ], 12.0 ) )
        self.assertFalse( bu_strip.needs_even_pass( [ float( s ) for s in range( 0, 12, 2 ) ], 12.0 ) )
        self.assertTrue( bu_strip.needs_even_pass( [ float( s ) for s in range( 0, 12, 3 ) ], 12.0 ) )
        self.assertFalse( bu_strip.needs_even_pass( [], 0.0 ) )

    def test_the_even_interval_caps_the_frames( self ):
        self.assertEqual( bu_strip.even_interval( 60 ), 1.0 )
        self.assertEqual( bu_strip.even_interval( 3600 ), 6.0 )

    def test_the_commands( self ):
        keyframes = bu_strip.command( '/v.mp4', bu_strip.KEYFRAMES, ( 160, 90 ) )
        self.assertEqual( keyframes[keyframes.index( '-skip_frame' ) + 1], 'nokey' )
        self.assertLess( keyframes.index( '-skip_frame' ), keyframes.index( '-i' ), "an input option" )
        self.assertIn( 'scale=160:90,showinfo', keyframes )
        even = bu_strip.command( '/v.mp4', bu_strip.EVEN, ( 160, 90 ), interval=2.5 )
        self.assertNotIn( '-skip_frame', even )
        self.assertIn( 'fps=1/2.500000,scale=160:90,showinfo', even )
        for argv in ( keyframes, even ):
            self.assertEqual( argv[-1], 'pipe:1', "frames go to a pipe, never a file" )


class TestShellPathCheck( _CacheTest ):

    def test_only_a_strip_image_in_the_cache_folder( self ):
        good = os.path.join( self.cache, 'a' * 32 + '.jpg' )
        self.assertTrue( bu_strip.is_strip_image( good ) )
        for bad in ( os.path.join( self.cache, 'a' * 32 + '.json' ), os.path.join( self.cache, 'x.jpg' ),
                     os.path.join( self.dir, 'a' * 32 + '.jpg' ),
                     os.path.join( self.cache, '..', 'a' * 32 + '.jpg' ), '/etc/' + 'a' * 32 + '.jpg' ):
            self.assertFalse( bu_strip.is_strip_image( bad ), bad )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestBuild( _CacheTest ):

    def clip( self, name='clip.mp4', **options ):
        path = os.path.join( self.dir, name )
        make_clip( path, **options )
        return path

    def test_keyframes_of_a_short_gop_clip_are_enough( self ):
        path = self.clip( gop=30 )
        layout = bu_strip.build( path, bu_strip.KEYFRAMES, False, self.facts() )
        self.assertEqual( layout['frames'], [ float( s ) for s in range( 12 ) ] )
        self.assertFalse( bu_strip.needs_even_pass( layout['frames'], 12.0 ) )
        self.assertEqual( layout['tile'], { 'width': 160, 'height': 90 } )
        self.assertEqual( ( layout['density'], layout['blurred'], layout['columns'] ), ( 'keyframes', False, 20 ) )
        import cv2
        sheet = cv2.imread( layout['image'] )
        self.assertEqual( sheet.shape[:2], ( 90, 12 * 160 ), "one row of twelve tiles" )
        with open( bu_strip.layout_path( bu_strip.key_for( path, 'keyframes', False ) ) ) as handle:
            self.assertEqual( json.load( handle )['frames'], layout['frames'] )

    def test_a_long_gop_clip_needs_the_even_pass( self ):
        path = self.clip( gop=300 )
        keyframes = bu_strip.build( path, bu_strip.KEYFRAMES, False, self.facts() )
        self.assertEqual( keyframes['frames'], [ 0.0, 10.0 ] )
        self.assertTrue( bu_strip.needs_even_pass( keyframes['frames'], 12.0 ) )
        even = bu_strip.build( path, bu_strip.EVEN, False, self.facts() )
        self.assertEqual( even['frames'], [ float( s ) for s in range( 12 ) ] )

    def test_the_frame_cap_holds( self ):
        path = self.clip( gop=1, seconds=4 )
        with mock.patch.object( bu_strip, 'MAX_FRAMES', 10 ):
            layout = bu_strip.build( path, bu_strip.KEYFRAMES, False, self.facts( seconds=4 ) )
        self.assertEqual( len( layout['frames'] ), 10 )
        gaps = [ later - earlier for earlier, later in zip( layout['frames'], layout['frames'][1:] ) ]
        self.assertTrue( all( gap >= 0.4 - 1e-6 for gap in gaps ), gaps )

    def test_a_source_strip_is_only_ever_written_blurred( self ):
        path = self.clip( gop=30 )
        layout = bu_strip.build( path, bu_strip.KEYFRAMES, True, self.facts() )
        self.assertTrue( layout['blurred'] )
        self.assertEqual( max( layout['tile']['width'], layout['tile']['height'] ), bu_thumbs.BLUR_SIZE )
        import cv2
        for root, _dirs, names in os.walk( self.dir ):
            for name in names:
                if name.endswith( ( '.jpg', '.png' ) ):
                    image = cv2.imread( os.path.join( root, name ) )
                    self.assertLessEqual( image.shape[0], bu_thumbs.BLUR_SIZE,
                                          "%s holds a frame taller than a blurred tile"%( name, ) )

    def test_a_changed_file_gets_a_new_key_and_the_passes_and_privacy_differ( self ):
        path = self.clip( gop=30 )
        keys = { bu_strip.key_for( path, density, blurred )
                 for density in ( 'keyframes', 'even' ) for blurred in ( False, True ) }
        self.assertEqual( len( keys ), 4 )
        before = bu_strip.key_for( path, 'keyframes', False )
        os.utime( path, ns=( 1, 1 ) )
        self.assertNotEqual( bu_strip.key_for( path, 'keyframes', False ), before )

    def test_not_a_video( self ):
        path = os.path.join( self.dir, 'notes.mp4' )
        with open( path, 'w' ) as handle:
            handle.write( 'not a video' )
        with self.assertRaises( bu_strip.StripError ) as raised:
            bu_strip.build( path, bu_strip.KEYFRAMES, True, self.facts() )
        self.assertEqual( raised.exception.reason, 'not_video' )
        self.assertFalse( os.path.exists( self.cache ) and os.listdir( self.cache ) )

    def test_read_needs_both_files( self ):
        path = self.clip( gop=30 )
        layout = bu_strip.build( path, bu_strip.KEYFRAMES, False, self.facts() )
        key = bu_strip.key_for( path, 'keyframes', False )
        self.assertEqual( bu_strip.read( key )['frames'], layout['frames'] )
        os.remove( layout['image'] )
        self.assertIsNone( bu_strip.read( key ) )
        self.assertIsNone( bu_strip.read( 'f' * 32 ) )


class TestService( _CacheTest ):

    def setUp( self ):
        super().setUp()
        self.events = []
        self.event_cond = threading.Condition()
        self.run_on = False
        self.builds = []
        self.gate = threading.Event()
        self.gate.set()
        self.layouts = {}
        self.video = os.path.join( self.dir, 'v.mp4' )
        with open( self.video, 'w' ) as handle:
            handle.write( 'x' )
        self.facts_for = { self.video: self.facts() }
        self.service = bu_serve_strip.StripService(
            self.send, lambda: self.run_on, logging.getLogger( 'test_catalog_strip' ), build=self.build,
            facts=lambda path: self.facts_for.get( path ), poll=0.05 )
        self.addCleanup( self.service.stop )

    def send( self, name, data ):
        with self.event_cond:
            self.events.append( ( name, data ) )
            self.event_cond.notify_all()

    def wait_events( self, count ):
        with self.event_cond:
            self.assertTrue( self.event_cond.wait_for( lambda: len( self.events ) >= count, WAIT ), self.events )

    def build( self, path, density, blurred, facts ):
        self.gate.wait( WAIT )
        self.builds.append( ( density, blurred ) )
        frames = self.layouts.get( density, [ 0.0, 10.0 ] if density == 'keyframes' else [ 0.0, 1.0 ] )
        key = bu_strip.key_for( path, density, blurred )
        os.makedirs( self.cache, exist_ok=True )
        with open( bu_strip.image_path( key ), 'wb' ) as handle:
            handle.write( b'jpg' )
        with open( bu_strip.layout_path( key ), 'w' ) as handle:
            json.dump( { 'density': density, 'blurred': blurred, 'frames': frames, 'tile': { 'width': 160,
                                                                                            'height': 90 },
                         'columns': 20, 'duration': facts['duration'] }, handle )
        return bu_strip.read( key )

    def test_answers_at_once_then_builds_both_passes_in_turn( self ):
        self.gate.clear()
        first = self.service.request( self.video, True )
        self.assertEqual( first, { 'path': self.video, 'blurred': True, 'building': True, 'ready': False } )
        self.gate.set()
        self.wait_events( 2 )
        self.assertEqual( self.events, [
            ( 'catalog.strip_ready', { 'path': self.video, 'density': 'keyframes', 'ok': True } ),
            ( 'catalog.strip_ready', { 'path': self.video, 'density': 'even', 'ok': True } ) ] )
        self.assertEqual( self.builds, [ ( 'keyframes', True ), ( 'even', True ) ] )
        done = self.service.request( self.video, True )
        self.assertEqual( ( done['ready'], done['building'], done['density'], done['frames'] ),
                          ( True, False, 'even', [ 0.0, 1.0 ] ) )
        self.assertTrue( bu_strip.is_strip_image( done['image'] ) )

    def test_dense_keyframes_need_no_even_pass( self ):
        self.layouts['keyframes'] = [ float( s ) for s in range( 12 ) ]
        self.service.request( self.video, False )
        self.wait_events( 1 )
        answer = self.service.request( self.video, False )
        self.assertEqual( ( answer['density'], answer['building'] ), ( 'keyframes', False ) )
        self.assertEqual( self.builds, [ ( 'keyframes', False ) ] )

    def test_the_keyframes_strip_is_answered_while_the_even_pass_waits_for_a_run_to_end( self ):
        self.run_on = True
        self.service.request( self.video, True )
        self.wait_events( 1 )
        answer = self.service.request( self.video, True )
        self.assertEqual( ( answer['density'], answer['building'] ), ( 'keyframes', True ) )
        import time
        time.sleep( 0.3 )
        self.assertEqual( self.builds, [ ( 'keyframes', True ) ], "the even pass waits while a run is on" )
        self.run_on = False
        self.wait_events( 2 )
        self.assertEqual( self.builds[-1], ( 'even', True ) )

    def test_asking_twice_queues_once( self ):
        self.gate.clear()
        self.service.request( self.video, True )
        self.service.request( self.video, True )
        self.gate.set()
        self.wait_events( 2 )
        self.assertEqual( self.builds, [ ( 'keyframes', True ), ( 'even', True ) ] )

    def test_clear_and_blurred_are_separate_strips( self ):
        self.layouts['keyframes'] = [ float( s ) for s in range( 12 ) ]
        self.service.request( self.video, True )
        self.wait_events( 1 )
        self.assertFalse( self.service.request( self.video, False )['ready'] )

    def test_a_pass_that_fails_is_announced_and_not_tried_again( self ):
        self.facts_for[self.video] = None
        self.service.request( self.video, True )
        self.wait_events( 1 )
        self.assertEqual( self.events, [ ( 'catalog.strip_ready', { 'path': self.video, 'density': 'keyframes',
                                                                    'ok': False } ) ] )
        answer = self.service.request( self.video, True )
        self.assertEqual( ( answer['ready'], answer['building'] ), ( False, False ) )
        self.assertEqual( self.builds, [] )

    def test_a_build_that_raises_is_announced_and_the_worker_goes_on( self ):
        other = os.path.join( self.dir, 'w.mp4' )
        with open( other, 'w' ) as handle:
            handle.write( 'y' )
        self.facts_for[other] = self.facts()
        real = self.build

        def flaky( path, density, blurred, facts ):
            if path == self.video:
                raise RuntimeError( 'boom' )
            return real( path, density, blurred, facts )

        self.service._build = flaky
        self.service.request( self.video, True )
        self.wait_events( 1 )
        self.service.request( other, True )
        self.wait_events( 2 )
        self.assertEqual( self.events[0][1]['ok'], False )
        self.assertEqual( self.events[1][1], { 'path': other, 'density': 'keyframes', 'ok': True } )


class TestCommand( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-strip-cmd-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.video = os.path.join( self.dir, 'v.mp4' )
        with open( self.video, 'w' ) as handle:
            handle.write( 'x' )
        self.asked = []
        service = mock.Mock()
        service.request.side_effect = lambda path, blurred: self.asked.append( ( path, blurred ) ) or {}
        catalog = mock.Mock()
        self.outputs = set()
        catalog.store.read.side_effect = lambda sql, params: [ ( 1, ) ] if params[0] in self.outputs else []
        for target, value in ( ( 'catalog_for', catalog ), ( 'served_strips', service ) ):
            patcher = mock.patch( 'betautils_serve_catalog.%s'%( target, ), return_value=value )
            patcher.start()
            self.addCleanup( patcher.stop )

    def call( self, data ):
        return bu_serve_strip.catalog_strip( object(), data )

    def test_any_file_is_blurred_and_a_present_output_is_clear( self ):
        self.call( { 'path': self.video } )
        import betautils_catalog_index as bu_index
        self.outputs.add( bu_index.norm( self.video ) )
        self.call( { 'path': self.video } )
        self.assertEqual( self.asked, [ ( self.video, True ), ( self.video, False ) ] )

    def test_bad_requests( self ):
        import betautils_protocol as bu_proto
        for data, code, params in ( ( {}, 'command.bad_params', { 'field': 'path', 'reason': 'missing' } ),
                                    ( { 'path': 3 }, 'command.bad_params', { 'field': 'path', 'reason': 'type' } ),
                                    ( { 'path': 'v.mp4' }, 'command.bad_params', { 'field': 'path', 'reason': 'value' } ),
                                    ( { 'path': self.dir }, 'catalog.no_such_file', {} ) ):
            with self.assertRaises( bu_proto.ProtocolError ) as raised:
                self.call( data )
            self.assertEqual( ( raised.exception.code, raised.exception.error['params'] ), ( code, params ) )
        self.assertEqual( self.asked, [] )


if __name__ == '__main__':
    unittest.main()
