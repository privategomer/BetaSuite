"""
test_serve_job_process.py - real jobs through a real engine process
(tests/serve_with_stub.py: everything real but the model).

Covers a job from submit to completed over the pipes, every message
checked against the schemas; that a running job's ffmpeg children hold
no descriptor to the engine's command or result pipe (Linux); and a
shutdown while a job runs stopping it with engine.stopping and exiting 0.
"""

import json
import os
import shutil
import sys
import unittest

from tests.test_betatv_end_to_end import _build_clip, _ffmpeg_available
from tests.test_serve import REPO, Engine, _EngineTest

WRAPPER = os.path.join( REPO, 'tests', 'serve_with_stub.py' )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestJobsThroughTheEngine( _EngineTest ):

    def setUp( self ):
        super().setUp()
        self.record = os.path.join( self.tmp, 'children.jsonl' )
        self.clip = os.path.join( self.tmp, 'clip.mp4' )
        _build_clip( self.clip, seconds=3 )

    def engine( self ):
        log_path = os.path.join( self.tmp, 'engine.log' )
        engine = Engine( [ sys.executable, WRAPPER, log_path, self.record ], log_path )
        self.addCleanup( engine.close )
        self.hello( engine )
        return engine

    def until( self, engine, predicate ):
        seen = []
        while True:
            message = engine.recv()
            seen.append( message )
            if predicate( message ):
                return seen

    @staticmethod
    def final( message, job ):
        return ( message['kind'] == 'event' and message['name'] == 'job.state'
                 and message['data']['job'] == job
                 and message['data']['state'] in ( 'completed', 'cancelled', 'stopped' ) )

    def test_a_job_runs_to_completion_over_the_pipes( self ):
        engine = self.engine()
        result = engine.request( 'job.submit', { 'files': [ self.clip ] } )
        self.assertTrue( result['ok'], result )
        job = result['data']['job']
        seen = self.until( engine, lambda m: self.final( m, job ) )
        self.assertEqual( seen[-1]['data']['state'], 'completed', seen[-1] )
        names = [ m['name'] for m in seen ]
        self.assertEqual( names[0], 'job.state' )
        self.assertIn( 'job.progress', names )
        outputs = [ m['data']['output'] for m in seen if m['name'] == 'job.file' and 'output' in m['data'] ]
        self.assertEqual( len( outputs ), 1 )
        self.assertTrue( os.path.getsize( outputs[0] ) > 0 )
        self.assertEqual( engine.request( 'shutdown' )['ok'], True )
        self.assertEqual( engine.wait(), 0 )
        self.assertStdoutOnlyProtocol( engine )
        transcript = os.environ.get( 'BETASUITE_JOB_TRANSCRIPT' )
        if transcript:
            shutil.copyfile( self.record, transcript + '.children.jsonl' ) if os.path.exists( self.record ) else None
            with open( transcript, 'wb' ) as handle:
                handle.write( bytes( engine.raw ) )

    @unittest.skipUnless( sys.platform.startswith( 'linux' ), "reads /proc" )
    def test_ffmpeg_children_hold_no_protocol_pipe( self ):
        engine = self.engine()
        protocol_pipes = { 'pipe:[%d]'%( os.fstat( stream.fileno() ).st_ino, )
                           for stream in ( engine.proc.stdin, engine.proc.stdout ) }
        job = engine.request( 'job.submit', { 'files': [ self.clip ] } )['data']['job']
        self.until( engine, lambda m: self.final( m, job ) )
        with open( self.record, encoding='utf-8' ) as handle:
            children = [ json.loads( line ) for line in handle ]
        self.assertTrue( children, "no ffmpeg child was seen running" )
        for child in children:
            self.assertEqual( set( child['links'] ) & protocol_pipes, set(), child )

    def test_shutdown_during_a_job_stops_it_and_exits_cleanly( self ):
        engine = self.engine()
        job = engine.request( 'job.submit', { 'files': [ self.clip, self.clip ] } )['data']['job']
        self.until( engine, lambda m: m['name'] == 'job.file' and m['data']['state'] == 'started' )
        engine.send( 'shutdown' )
        seen = self.until( engine, lambda m: self.final( m, job ) )
        final = seen[-1]['data']
        self.assertEqual( ( final['state'], final.get( 'code' ) ), ( 'stopped', 'engine.stopping' ) )
        self.assertEqual( engine.wait(), 0 )
        self.assertStdoutOnlyProtocol( engine )


if __name__ == '__main__':
    unittest.main()
