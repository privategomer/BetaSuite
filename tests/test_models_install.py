"""
test_models_install.py - downloading, importing and removing models.

A local HTTP server on 127.0.0.1 stands in for the release mirrors, with
Range support and switches for the failures that matter. Fixtures are
small synthetic files; nothing touches the real model folder or the real
hash memo, and nothing leaves the machine.

What each test prevents:
  - a tampered or wrong file installed: it is rejected and deleted
  - a download killed partway (the engine process dies) starting over,
    or never finishing: the next install resumes with a Range request
  - a lost connection leaving a broken model in place, or losing what
    was already downloaded
  - a server that ignores Range corrupting the file by appending a
    whole copy
  - a GitHub sign-in or warning page saved as a model
  - one dead mirror failing the install while another works
  - "I already have the file" installing an unknown file, or moving the
    user's own copy
  - removing the model the config is using
"""

import hashlib
import http.server
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import betautils_models as bu_models

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
PAYLOAD = bytes( ( i * 7 + 3 ) % 251 for i in range( 300 * 1024 ) )


class _Server( http.server.ThreadingHTTPServer ):
    daemon_threads = True

    def __init__( self ):
        super().__init__( ( '127.0.0.1', 0 ), _Handler )
        self.files = {}
        self.ranges = []
        self.ignore_range = False
        self.drop_after = None
        self.drop_reset = False
        self.range_shift = 0
        self.delay = 0.0
        self.status_override = None

    @property
    def base( self ):
        return 'http://127.0.0.1:%d/'%( self.server_address[1], )


class _Handler( http.server.BaseHTTPRequestHandler ):

    def log_message( self, *_args ):
        pass

    def do_GET( self ):
        server = self.server
        name = self.path.lstrip( '/' )
        if server.status_override:
            self.send_response( server.status_override )
            self.end_headers()
            return
        if name not in server.files:
            self.send_response( 404 )
            self.end_headers()
            return
        body = server.files[name]
        header = self.headers.get( 'Range' )
        server.ranges.append( header )
        start = 0
        if header and not server.ignore_range:
            start = int( header.split( '=', 1 )[1].split( '-', 1 )[0] )
            # range_shift: a server answering a different range than asked.
            start = min( len( body ) - 1, start + server.range_shift )
            self.send_response( 206 )
            self.send_header( 'Content-Range', 'bytes %d-%d/%d'%( start, len( body ) - 1, len( body ) ) )
        else:
            self.send_response( 200 )
        self.send_header( 'Content-Length', str( len( body ) - start ) )
        self.end_headers()
        sent, piece = start, 16 * 1024
        try:
            while sent < len( body ):
                if server.drop_after is not None and sent >= server.drop_after:
                    # Lost network: the body stops short. A reset when
                    # drop_reset, otherwise a clean close (the client
                    # sees the end of the stream early).
                    if server.drop_reset:
                        self.connection.setsockopt( socket.SOL_SOCKET, socket.SO_LINGER,
                                                    struct.pack( 'HH' if os.name == 'nt' else 'ii', 1, 0 ) )
                    self.close_connection = True
                    return
                end = min( len( body ), sent + piece )
                self.wfile.write( body[sent:end] )
                self.wfile.flush()
                sent = end
                if server.delay:
                    time.sleep( server.delay )
        except ( BrokenPipeError, ConnectionResetError ):
            pass


def _catalog( mirrors, payload=PAYLOAD, model_id='m-test' ):
    return {
        'format': 1, 'mirrors': mirrors,
        'licenses': { 'l': { 'file': 'LICENSE-x' } },
        'models': [ { 'id': model_id, 'kind': 'detector', 'backend': 'x', 'variant': None,
                      'file': 'test-model.onnx', 'size': len( payload ),
                      'sha256': hashlib.sha256( payload ).hexdigest(), 'license': 'l', 'sources': [] } ],
    }


class _ModelDirTest( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='bs-models-' )
        self.dir = os.path.join( self.tmp, 'model' )
        patches = [
            mock.patch.object( bu_models, 'model_dir', return_value=self.dir ),
            mock.patch.object( bu_models, '_flush_memo' ),
            mock.patch.object( bu_models, 'in_use', return_value={ 'detector': None, 'person': None, 'custom': {} } ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup( patch.stop )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.server = _Server()
        self.server.files['test-model.onnx'] = PAYLOAD
        threading.Thread( target=self.server.serve_forever, kwargs={ 'poll_interval': 0.02 }, daemon=True ).start()
        self.addCleanup( self.server.server_close )
        self.addCleanup( self.server.shutdown )
        self.catalog = _catalog( [ self.server.base ] )

    def installed( self ):
        return os.path.join( self.dir, 'test-model.onnx' )

    def part( self ):
        return self.installed() + bu_models.PART_SUFFIX

    def install( self, **kw ):
        return bu_models.install( 'm-test', catalog=self.catalog, **kw )


class TestDownload( _ModelDirTest ):

    def test_fresh_install_is_verified_and_placed( self ):
        seen = []
        self.assertEqual( self.install( progress=lambda done, total: seen.append( ( done, total ) ) ), 'installed' )
        with open( self.installed(), 'rb' ) as fin:
            self.assertEqual( fin.read(), PAYLOAD )
        self.assertFalse( os.path.exists( self.part() ) )
        self.assertEqual( seen[-1], ( len( PAYLOAD ), len( PAYLOAD ) ) )
        self.assertEqual( self.server.ranges, [ None ] )
        self.assertEqual( bu_models.status( self.catalog['models'][0] )['state'], 'installed' )

    def test_second_install_is_a_no_op( self ):
        self.install()
        self.assertEqual( self.install(), 'already' )
        self.assertEqual( len( self.server.ranges ), 1 )

    def test_tampered_file_is_rejected_and_deleted( self ):
        tampered = bytearray( PAYLOAD )
        tampered[1000] ^= 0xFF
        self.server.files['test-model.onnx'] = bytes( tampered )
        with self.assertRaises( bu_models.ModelError ) as caught:
            self.install()
        self.assertEqual( caught.exception.code, 'model.hash_mismatch' )
        self.assertFalse( os.path.exists( self.installed() ) )
        self.assertFalse( os.path.exists( self.part() ) )

    def test_oversized_answer_is_rejected( self ):
        self.server.files['test-model.onnx'] = PAYLOAD + b'extra'
        with self.assertRaises( bu_models.ModelError ) as caught:
            self.install()
        self.assertEqual( caught.exception.code, 'model.hash_mismatch' )
        self.assertFalse( os.path.exists( self.part() ) )

    def test_lost_network_keeps_progress_then_resumes( self ):
        self.server.drop_after = 128 * 1024
        with self.assertRaises( bu_models.ModelError ) as caught:
            self.install()
        self.assertEqual( caught.exception.code, 'model.download_failed' )
        self.assertEqual( caught.exception.params['reason'], bu_models.REASON_NETWORK )
        self.assertFalse( os.path.exists( self.installed() ) )
        kept = os.path.getsize( self.part() )
        self.assertGreaterEqual( kept, 128 * 1024 )
        self.assertEqual( bu_models.status( self.catalog['models'][0] ), { 'state': 'partial', 'bytes': kept } )

        self.server.drop_after = None
        self.assertEqual( self.install(), 'installed' )
        self.assertEqual( self.server.ranges[-1], 'bytes=%d-'%( kept, ) )
        with open( self.installed(), 'rb' ) as fin:
            self.assertEqual( fin.read(), PAYLOAD )

    def test_connection_reset_keeps_progress_then_resumes( self ):
        self.server.drop_after = 128 * 1024
        self.server.drop_reset = True
        with self.assertRaises( bu_models.ModelError ) as caught:
            self.install()
        self.assertEqual( caught.exception.params['reason'], bu_models.REASON_NETWORK )
        self.assertTrue( os.path.exists( self.part() ) )
        self.server.drop_after = None
        self.assertEqual( self.install(), 'installed' )
        self.assertTrue( self.server.ranges[-1].startswith( 'bytes=' ) )

    def test_progress_never_passes_the_size( self ):
        self.server.files['test-model.onnx'] = PAYLOAD + b'x' * 100000
        seen = []
        with self.assertRaises( bu_models.ModelError ):
            self.install( progress=lambda done, total: seen.append( done ) )
        self.assertLessEqual( max( seen ), len( PAYLOAD ) )

    def test_wrong_range_answer_starts_over( self ):
        with open( self.part_dir_ready(), 'wb' ) as out:
            out.write( PAYLOAD[:50000] )
        self.server.range_shift = 10
        with self.assertRaises( bu_models.ModelError ) as caught:
            self.install()
        self.assertEqual( caught.exception.params['reason'], bu_models.REASON_HTTP )
        self.assertFalse( os.path.exists( self.part() ) )
        self.server.range_shift = 0
        self.assertEqual( self.install(), 'installed' )
        self.assertEqual( self.server.ranges[-1], None )

    def test_server_ignoring_range_restarts_cleanly( self ):
        with open( self.part_dir_ready(), 'wb' ) as out:
            out.write( PAYLOAD[:50000] )
        self.server.ignore_range = True
        self.assertEqual( self.install(), 'installed' )
        self.assertEqual( self.server.ranges, [ 'bytes=50000-' ] )
        with open( self.installed(), 'rb' ) as fin:
            self.assertEqual( fin.read(), PAYLOAD )

    def test_part_longer_than_the_file_starts_over( self ):
        with open( self.part_dir_ready(), 'wb' ) as out:
            out.write( PAYLOAD + b'junk' )
        self.assertEqual( self.install(), 'installed' )
        self.assertEqual( self.server.ranges, [ None ] )

    def test_complete_part_is_verified_without_a_request( self ):
        with open( self.part_dir_ready(), 'wb' ) as out:
            out.write( PAYLOAD )
        self.assertEqual( self.install(), 'installed' )
        self.assertEqual( self.server.ranges, [] )

    def test_sign_in_page_is_not_kept( self ):
        page = b'<!DOCTYPE html><html>Sign in</html>'
        self.server.files['test-model.onnx'] = page + b' ' * ( len( PAYLOAD ) - len( page ) )
        with self.assertRaises( bu_models.ModelError ) as caught:
            self.install()
        self.assertEqual( caught.exception.params['reason'], bu_models.REASON_NOT_A_MODEL )
        self.assertFalse( os.path.exists( self.part() ) )

    def test_http_error_is_reported( self ):
        self.server.status_override = 503
        with self.assertRaises( bu_models.ModelError ) as caught:
            self.install()
        self.assertEqual( caught.exception.code, 'model.download_failed' )

    def test_next_mirror_is_tried( self ):
        dead = _Server()
        dead_base = dead.base
        dead.server_close()
        self.catalog['mirrors'] = [ dead_base, self.server.base ]
        self.assertEqual( self.install(), 'installed' )

    def test_stop_keeps_the_part_for_a_resume( self ):
        calls = { 'n': 0 }

        def stop():
            calls['n'] += 1
            return calls['n'] > 3

        self.server.delay = 0.01
        with self.assertRaises( bu_models.Stopped ):
            self.install( should_stop=stop )
        self.assertTrue( os.path.exists( self.part() ) )
        self.assertFalse( os.path.exists( self.installed() ) )
        self.server.delay = 0.0
        self.assertEqual( self.install(), 'installed' )
        self.assertTrue( self.server.ranges[-1].startswith( 'bytes=' ) )

    def test_full_disk_is_refused_before_downloading( self ):
        usage = shutil.disk_usage( self.tmp )._replace( free=10 )
        with mock.patch( 'shutil.disk_usage', return_value=usage ):
            with self.assertRaises( bu_models.ModelError ) as caught:
                self.install()
        self.assertEqual( caught.exception.code, 'disk.full' )
        self.assertEqual( self.server.ranges, [] )

    def test_unknown_id( self ):
        with self.assertRaises( bu_models.ModelError ) as caught:
            bu_models.install( 'nope', catalog=self.catalog )
        self.assertEqual( caught.exception.code, 'model.unknown' )

    def part_dir_ready( self ):
        os.makedirs( self.dir, exist_ok=True )
        return self.part()


_KILL_SCRIPT = r'''
import json, sys
sys.path.insert( 0, sys.argv[1] )
import betautils_models as bu_models
bu_models.model_dir = lambda: sys.argv[2]
bu_models._flush_memo = lambda: None
bu_models.CHUNK_BYTES = 4096
catalog = json.loads( sys.argv[3] )
bu_models.install( 'm-test', catalog=catalog )
'''


class TestKilledDownload( _ModelDirTest ):

    def test_process_killed_mid_download_resumes( self ):
        self.server.delay = 0.02
        child = subprocess.Popen( [ sys.executable, '-c', _KILL_SCRIPT, REPO_ROOT, self.dir,
                                    json.dumps( self.catalog ) ] )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if ( bu_models._size( self.part() ) or 0 ) >= 64 * 1024:
                    break
                time.sleep( 0.01 )
            child.kill()
        finally:
            child.wait( 10 )
        kept = os.path.getsize( self.part() )
        self.assertGreater( kept, 0 )
        self.assertLess( kept, len( PAYLOAD ) )
        self.assertFalse( os.path.exists( self.installed() ) )

        self.server.delay = 0.0
        self.assertEqual( self.install(), 'installed' )
        self.assertEqual( self.server.ranges[-1], 'bytes=%d-'%( kept, ) )
        with open( self.installed(), 'rb' ) as fin:
            self.assertEqual( hashlib.sha256( fin.read() ).hexdigest(), hashlib.sha256( PAYLOAD ).hexdigest() )


class TestImportAndRemove( _ModelDirTest ):

    def _user_file( self, data ):
        path = os.path.join( self.tmp, 'downloads', 'whatever-name.onnx' )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'wb' ) as out:
            out.write( data )
        return path

    def test_import_of_a_known_file_installs_it( self ):
        mine = self._user_file( PAYLOAD )
        self.assertEqual( bu_models.import_file( mine, catalog=self.catalog ), 'm-test' )
        with open( self.installed(), 'rb' ) as fin:
            self.assertEqual( fin.read(), PAYLOAD )
        self.assertTrue( os.path.exists( mine ), "the user's own copy must stay" )
        self.assertEqual( os.listdir( self.dir ), [ 'test-model.onnx' ] )

    def test_import_replaces_a_partial_download( self ):
        with open( self.part_dir_ready(), 'wb' ) as out:
            out.write( PAYLOAD[:1000] )
        bu_models.import_file( self._user_file( PAYLOAD ), catalog=self.catalog )
        self.assertFalse( os.path.exists( self.part() ) )

    def test_import_of_an_unknown_file_is_refused( self ):
        with self.assertRaises( bu_models.ModelError ) as caught:
            bu_models.import_file( self._user_file( PAYLOAD[:-1] ), catalog=self.catalog )
        self.assertEqual( caught.exception.code, 'model.unknown_file' )
        self.assertFalse( os.path.exists( self.installed() ) )

    def test_modified_file_is_not_installed( self ):
        os.makedirs( self.dir )
        with open( self.installed(), 'wb' ) as out:
            out.write( PAYLOAD[:-1] + b'x' )
        self.assertEqual( bu_models.status( self.catalog['models'][0] )['state'], 'modified' )
        self.assertEqual( self.install(), 'installed' )
        with open( self.installed() + '.bad', 'rb' ) as kept:
            self.assertEqual( kept.read(), PAYLOAD[:-1] + b'x', "the swapped-in file is kept aside" )

    def test_remove_deletes_file_and_part( self ):
        self.install()
        with open( self.part(), 'wb' ) as out:
            out.write( b'x' )
        bu_models.remove( 'm-test', catalog=self.catalog )
        self.assertEqual( os.listdir( self.dir ), [] )

    def test_remove_refuses_the_model_in_use( self ):
        self.install()
        with mock.patch.object( bu_models, 'in_use', return_value={ 'detector': 'm-test', 'person': None, 'custom': {} } ):
            with self.assertRaises( bu_models.ModelError ) as caught:
                bu_models.remove( 'm-test', catalog=self.catalog )
        self.assertEqual( caught.exception.code, 'model.in_use' )
        self.assertTrue( os.path.exists( self.installed() ) )

    part_dir_ready = TestDownload.part_dir_ready


class TestSha256ForFile( unittest.TestCase ):

    def test_matches_hashlib_and_stops_on_request( self ):
        import betautils_hash as bu_hash
        with tempfile.NamedTemporaryFile( delete=False ) as out:
            out.write( PAYLOAD )
        self.addCleanup( os.remove, out.name )
        self.assertEqual( bu_hash.sha256_for_file( out.name, use_memo=False ), hashlib.sha256( PAYLOAD ).hexdigest() )
        with self.assertRaises( bu_hash.HashStopped ):
            bu_hash.sha256_for_file( out.name, use_memo=False, should_stop=lambda: True )


if __name__ == '__main__':
    unittest.main()
