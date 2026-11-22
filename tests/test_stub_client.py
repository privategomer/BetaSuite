"""
test_stub_client.py - tools/engine/stub_client.py drives a real engine
from start to exit and records the session it saw.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
TOOL = os.path.join( REPO, 'tools', 'engine', 'stub_client.py' )


class TestStubClient( unittest.TestCase ):

    def test_a_session_from_hello_to_shutdown_is_recorded( self ):
        with tempfile.TemporaryDirectory() as tmp:
            paths = { name: os.path.join( tmp, name ) for name in ( 't.jsonl', 'stub.log', 'engine.log' ) }
            result = subprocess.run(
                [ sys.executable, TOOL, '--transcript', paths['t.jsonl'], '--log-file', paths['stub.log'],
                  '--engine-log', paths['engine.log'], '--console-level', 'error', '--log-level', 'debug' ],
                capture_output=True, cwd=REPO, timeout=120, stdin=subprocess.DEVNULL )
            self.assertEqual( result.returncode, 0, result.stderr.decode( 'utf-8', 'replace' ) )
            self.assertEqual( result.stdout, b'' )
            with open( paths['t.jsonl'], encoding='utf-8' ) as handle:
                records = [ json.loads( line ) for line in handle ]
            with open( paths['stub.log'], encoding='utf-8' ) as handle:
                log = handle.read()
            with open( paths['engine.log'], encoding='utf-8' ) as handle:
                engine_log = handle.read()
        self.assertEqual( [ ( r['dir'], r['message']['name'] ) for r in records ], [
            ( 'send', 'hello' ), ( 'recv', 'hello' ), ( 'send', 'ping' ), ( 'recv', 'ping' ),
            ( 'send', 'shutdown' ), ( 'recv', 'shutdown' ) ] )
        times = [ r['ms'] for r in records ]
        self.assertEqual( times, sorted( times ) )
        for sent, received in zip( records[::2], records[1::2] ):
            self.assertEqual( received['message']['id'], sent['message']['id'] )
            self.assertTrue( received['message']['ok'] )
        self.assertIn( 'ready in', log )
        self.assertIn( 'engine stderr: engine serving', log )
        self.assertIn( 'engine exited 0', log )
        self.assertIn( 'hello from stub_client', engine_log )

    def test_an_engine_that_fails_is_reported_with_exit_1( self ):
        with tempfile.TemporaryDirectory() as tmp:
            # A log path inside a file cannot be created, so the engine fails before serving.
            blocker = os.path.join( tmp, 'file' )
            open( blocker, 'w' ).close()
            result = subprocess.run(
                [ sys.executable, TOOL, '--transcript', os.path.join( tmp, 't.jsonl' ),
                  '--log-file', os.path.join( tmp, 'stub.log' ), '--engine-log', os.path.join( blocker, 'x.log' ),
                  '--console-level', 'error', '--timeout', '30' ],
                capture_output=True, cwd=REPO, timeout=120, stdin=subprocess.DEVNULL )
            self.assertEqual( result.returncode, 1 )
            self.assertIn( b'closed stdout before answering hello', result.stdout )


if __name__ == '__main__':
    unittest.main()
