"""
The catalog.* commands and the catalog.changed event, in process: every
message checked against its schema (tests.test_serve_job.Wire).
"""

import io
import json
import logging
import os
import shutil
import tempfile
import threading
import unittest

import betautils_catalog as bu_catalog
import betautils_catalog_store as bu_store
import betautils_hash as bu_hash
import betautils_serve as bu_serve
import betautils_serve_catalog as bu_serve_catalog

from tests.test_serve_job import Wire


def _logger():
    logger = logging.getLogger( 'test_serve_catalog' )
    logger.propagate = False
    if not logger.handlers:
        quiet = logging.NullHandler()
        quiet.setLevel( logging.CRITICAL )
        logger.addHandler( quiet )
    return logger


class _ServeTest( unittest.TestCase ):
    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-serve-catalog-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.db_path = os.path.join( self.tmp, 'catalog.sqlite' )
        self.watched = os.path.join( self.tmp, 'videos' )
        os.makedirs( self.watched )
        self.wire = Wire()
        self.server = bu_serve.Server( io.BytesIO(), self.wire, _logger() )
        self.server.protocol = 1
        self.server.catalog_factory = self.factory
        self.addCleanup( self.server.stop_catalog )
        self._ids = iter( range( 1, 10000 ) )

    def factory( self, **kwargs ):
        return bu_catalog.Catalog( path=self.db_path, hasher=lambda path: bu_hash.md5_for_file( path, 16, use_memo=False ),
                                   settle=0.0, seed_folders=[], **dict( kwargs, interval=0 ) )

    def command( self, name, data=None ):
        request_id = 'c%d'%( next( self._ids ), )
        self.server.handle_line( json.dumps( { 'v': 1, 'kind': 'cmd', 'id': request_id, 'name': name,
                                               'data': data or {} } ).encode() )
        return self.wire.wait_for( lambda m: m.get( 'id' ) == request_id, name )

    def ok( self, name, data=None ):
        result = self.command( name, data )
        self.assertTrue( result['ok'], result )
        return result['data']

    def refused( self, name, data, code ):
        result = self.command( name, data )
        self.assertFalse( result['ok'], result )
        self.assertEqual( result['error']['code'], code, result )
        return result['error']['params']

    def write( self, name, size=2048, folder=None ):
        path = os.path.join( folder or self.watched, name )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'wb' ) as handle:
            handle.write( os.urandom( size ) )
        return path

    def changed_with( self, digest ):
        return self.wire.wait_for( lambda m: m.get( 'name' ) == 'catalog.changed' and digest in m['data'].get( 'items', () ),
                                   'catalog.changed for %s'%( digest, ) )


class TestCommands( _ServeTest ):
    def test_a_watched_folder_reports_its_files_and_lists_them( self ):
        path = self.write( 'a.mp4' )
        added = self.ok( 'catalog.add_folder', { 'path': self.watched, 'type_from_folder': True } )
        self.assertTrue( added['created'] )
        self.assertEqual( ( added['folder']['path'], added['folder']['type_from_folder'] ),
                          ( os.path.normpath( self.watched ), True ) )
        digest = bu_hash.md5_for_file( path, 16, use_memo=False )
        self.changed_with( digest )
        listed = self.ok( 'catalog.list' )
        self.assertEqual( [ item['item'] for item in listed['items'] ], [ digest ] )
        self.assertEqual( listed['total'], 1 )
        got = self.ok( 'catalog.get', { 'item': digest } )
        self.assertEqual( [ location['path'] for location in got['locations'] ], [ os.path.normpath( path ) ] )
        folders = self.ok( 'catalog.folders' )['folders']
        self.assertEqual( [ ( folder['items'], folder['available'] ) for folder in folders ], [ ( 1, True ) ] )

    def test_add_files_answers_what_it_took_and_what_it_refused( self ):
        good = self.write( 'g.mkv', folder=os.path.join( self.tmp, 'loose' ) )
        result = self.ok( 'catalog.add_files', { 'paths': [ good, os.path.join( self.tmp, 'none.mp4' ) ] } )
        self.assertEqual( result, { 'accepted': 1, 'refused': [ { 'index': 1, 'reason': 'not_found' } ] } )
        self.changed_with( bu_hash.md5_for_file( good, 16, use_memo=False ) )

    def test_rescan_finds_a_new_file( self ):
        self.ok( 'catalog.add_folder', { 'path': self.watched } )
        path = self.write( 'later.mp4' )
        self.ok( 'catalog.rescan' )
        self.changed_with( bu_hash.md5_for_file( path, 16, use_memo=False ) )

    def test_remove_folder_and_forget( self ):
        path = self.write( 'a.mp4' )
        folder = self.ok( 'catalog.add_folder', { 'path': self.watched } )['folder']['folder']
        digest = bu_hash.md5_for_file( path, 16, use_memo=False )
        self.changed_with( digest )
        self.assertEqual( self.ok( 'catalog.remove_folder', { 'folder': folder } ), { 'items': 1 } )
        self.assertEqual( self.ok( 'catalog.list' )['items'], [] )
        self.assertEqual( self.ok( 'catalog.forget', { 'item': digest } ), {} )
        self.assertEqual( self.refused( 'catalog.get', { 'item': digest }, 'catalog.no_such_item' ),
                          { 'item': digest } )


class TestRefusals( _ServeTest ):
    def test_unknown_ids( self ):
        self.assertEqual( self.refused( 'catalog.get', { 'item': 'f' * 16 }, 'catalog.no_such_item' ),
                          { 'item': 'f' * 16 } )
        self.assertEqual( self.refused( 'catalog.remove_folder', { 'folder': 9 }, 'catalog.no_such_folder' ),
                          { 'folder': 9 } )
        self.assertEqual( self.refused( 'catalog.forget', { 'item': 'e' * 16 }, 'catalog.no_such_item' ),
                          { 'item': 'e' * 16 } )

    def test_bad_folders( self ):
        self.assertEqual( self.refused( 'catalog.add_folder', { 'path': os.path.join( self.tmp, 'gone' ) },
                                        'catalog.bad_folder' ), { 'reason': 'not_found' } )

    def test_bad_params( self ):
        cases = (
            ( 'catalog.get', {}, { 'field': 'item', 'reason': 'missing' } ),
            ( 'catalog.get', { 'item': 3 }, { 'field': 'item', 'reason': 'type' } ),
            ( 'catalog.add_folder', {}, { 'field': 'path', 'reason': 'missing' } ),
            ( 'catalog.add_folder', { 'path': 'relative/dir' }, { 'field': 'path', 'reason': 'value' } ),
            ( 'catalog.add_folder', { 'path': self.watched, 'type_from_folder': 'yes' },
              { 'field': 'type_from_folder', 'reason': 'type' } ),
            ( 'catalog.remove_folder', { 'folder': '1' }, { 'field': 'folder', 'reason': 'type' } ),
            ( 'catalog.remove_folder', { 'folder': True }, { 'field': 'folder', 'reason': 'type' } ),
            ( 'catalog.add_files', { 'paths': [] }, { 'field': 'paths', 'reason': 'type' } ),
            ( 'catalog.add_files', { 'paths': [ 'x.mp4' ] }, { 'field': 'paths.0', 'reason': 'value' } ),
            ( 'catalog.list', { 'sort': 'color' }, { 'field': 'sort', 'reason': 'value' } ),
            ( 'catalog.list', { 'limit': 501 }, { 'field': 'limit', 'reason': 'value' } ),
            ( 'catalog.list', { 'limit': 0 }, { 'field': 'limit', 'reason': 'value' } ),
            ( 'catalog.list', { 'offset': -1 }, { 'field': 'offset', 'reason': 'value' } ),
            ( 'catalog.list', { 'filter': { 'status': [ 'odd' ] } }, { 'field': 'filter.status', 'reason': 'value' } ),
            ( 'catalog.list', { 'filter': { 'duplicates': 1 } }, { 'field': 'filter.duplicates', 'reason': 'type' } ),
            ( 'catalog.list', { 'filter': [] }, { 'field': 'filter', 'reason': 'type' } ),
        )
        for name, data, params in cases:
            with self.subTest( name=name, data=data ):
                self.assertEqual( self.refused( name, data, 'command.bad_params' ), params )


class TestOpening( _ServeTest ):
    def test_the_catalog_opens_on_the_first_catalog_command_only( self ):
        self.assertIsNone( self.server.catalog )
        self.ok( 'catalog.list' )
        self.assertIsNotNone( self.server.catalog )

    def test_a_catalog_that_cannot_open_is_refused_with_its_code_and_tried_again( self ):
        calls = []

        def failing( **kwargs ):
            calls.append( 1 )
            if len( calls ) == 1:
                raise bu_store.CatalogError( 'catalog.too_new', version=9, supported=1 )
            return self.factory( **kwargs )

        self.server.catalog_factory = failing
        self.assertEqual( self.refused( 'catalog.list', {}, 'catalog.too_new' ), { 'version': 9, 'supported': 1 } )
        self.ok( 'catalog.list' )
        self.assertEqual( len( calls ), 2 )

    def test_stopping_the_server_closes_the_catalog_and_its_lock( self ):
        self.ok( 'catalog.list' )
        self.server.stop_catalog()
        self.assertIsNone( self.server.catalog )
        bu_store.Store( self.db_path, exclusive=True ).close()

    def test_hashing_counts_a_command_line_run_as_busy( self ):
        import betautils_runlock as bu_runlock
        seen = {}

        def capture( **kwargs ):
            seen['busy'] = kwargs['busy']
            return self.factory( **kwargs )

        self.server.catalog_factory = capture
        self.ok( 'catalog.list' )
        self.assertFalse( seen['busy']() )
        held = bu_runlock.acquire_library( 'betatv' )
        try:
            self.assertTrue( seen['busy'](), 'a run holding the library lock' )
        finally:
            held.release()
        self.assertFalse( seen['busy']() )

    def test_hello_offers_the_catalog_commands_and_event( self ):
        hello = bu_serve._hello( self.server, { 'protocol': [ 1 ] } )
        for name in bu_serve_catalog.HANDLERS:
            self.assertIn( name, hello['commands'] )
        self.assertIn( 'catalog.changed', hello['events'] )


class TestChangeEvents( unittest.TestCase ):
    def setUp( self ):
        self.sent = []
        self.cond = threading.Condition()

    def send( self, name, data ):
        with self.cond:
            self.sent.append( ( name, data ) )
            self.cond.notify_all()

    def wait_sent( self, count ):
        with self.cond:
            self.assertTrue( self.cond.wait_for( lambda: len( self.sent ) >= count, 10 ), self.sent )

    def test_changes_in_one_burst_are_merged_never_dropped( self ):
        events = bu_serve_catalog.ChangeEvents( self.send, items_max=1000, interval=3600 )
        self.addCleanup( events.stop )
        events( { 'a' }, 'scan' )
        self.wait_sent( 1 )
        events( { 'b' }, 'scan' )
        events( { 'c', 'b' }, 'pin' )
        events.flush()
        self.assertEqual( self.sent, [ ( 'catalog.changed', { 'reasons': [ 'scan' ], 'items': [ 'a' ] } ),
                                       ( 'catalog.changed', { 'reasons': [ 'pin', 'scan' ], 'items': [ 'b', 'c' ] } ) ] )

    def test_too_many_items_or_an_unknown_change_say_all( self ):
        events = bu_serve_catalog.ChangeEvents( self.send, items_max=2, interval=0 )
        self.addCleanup( events.stop )
        events( { 'a', 'b', 'c' }, 'scan' )
        self.wait_sent( 1 )
        events( None, 'external' )
        self.wait_sent( 2 )
        self.assertEqual( [ data for _name, data in self.sent ],
                          [ { 'reasons': [ 'scan' ], 'all': True }, { 'reasons': [ 'external' ], 'all': True } ] )

    def test_stop_sends_what_is_pending( self ):
        events = bu_serve_catalog.ChangeEvents( self.send, items_max=10, interval=3600 )
        events( { 'a' }, 'scan' )
        self.wait_sent( 1 )
        events( { 'z' }, 'job' )
        events.stop()
        self.assertEqual( self.sent[-1], ( 'catalog.changed', { 'reasons': [ 'job' ], 'items': [ 'z' ] } ) )


if __name__ == '__main__':
    unittest.main()
