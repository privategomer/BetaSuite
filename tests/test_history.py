"""
test_history.py - "Clear history" (betautils_history, storage.history and
storage.clear_history), on scratch trees.

What each test prevents:
  - a default that clears work to redo (detections, the library) or
    leaves a private trace (transcodes, previews) unticked
  - costly kinds cleared without confirm, anything cleared while a job
    runs, a refusal that comes after removing some of it
  - a time range that clears older files, rows or log lines, or keeps
    newer ones
  - a log the app is writing moved away, so later lines land in the trash
  - cleared catalog rows still readable in the file's free pages or WAL
  - settings, models or outputs touched (end to end, on the real kinds)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import betautils_catalog_store as bu_store
import betautils_history as bu_history
from tests import protocol_schemas as schemas

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
NOW = 1_800_000_000.0
HOUR = 3600.0


def _write( path, data=b'x' * 10, mtime=None ):
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    with open( path, 'wb' ) as handle:
        handle.write( data if isinstance( data, bytes ) else data.encode( 'utf-8' ) )
    if mtime is not None:
        os.utime( path, ( mtime, mtime ) )


def _stamp( t ):
    return time.strftime( '%Y-%m-%d %H:%M:%S', time.localtime( t ) )


class _Trash:

    def __init__( self, root ):
        self.root = root
        self.moved = []

    def __call__( self, path ):
        target = os.path.join( self.root, '%d-%s'%( len( self.moved ), os.path.basename( path ) ) )
        os.makedirs( self.root, exist_ok=True )
        shutil.move( path, target )
        self.moved.append( ( path, target ) )


class _Tree( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='history-' )
        self.addCleanup( shutil.rmtree, self.root, True )
        self.trash = _Trash( os.path.join( self.root, 'trash' ) )
        r = self.root
        self.previews = os.path.join( r, 'cache', 'thumbs' )
        self.transcodes = os.path.join( r, 'cache', 'transcode' )
        self.detections = os.path.join( r, 'cache', 'vid_hashes' )
        self.logs = os.path.join( r, 'state', 'logs' )
        _write( os.path.join( self.previews, 'old.jpg' ), mtime=NOW - 3 * HOUR )
        _write( os.path.join( self.previews, 'new.jpg' ), mtime=NOW - 0.5 * HOUR )
        _write( os.path.join( self.transcodes, 'a.mkv' ), b'y' * 30, mtime=NOW - 0.5 * HOUR )
        _write( os.path.join( self.detections, 'd.json' ), mtime=NOW - 0.5 * HOUR )
        K = bu_history.Kind
        self.kinds = [
            K( 'previews', default=True, paths=[ self.previews ] ),
            K( 'transcodes', default=True, paths=[ self.transcodes ] ),
            K( 'logs', default=True, paths=[ bu_history._Logs( self.logs ) ] ),
            K( 'detections', confirm=True, paths=[ self.detections ] ),
        ]

    def clear( self, ids, **kw ):
        kw.setdefault( 'trash', self.trash )
        return bu_history.clear( self.kinds, ids, **kw )


class KindsTest( unittest.TestCase ):

    def test_defaults_are_cheap_traces_and_costly_kinds_need_confirm( self ):
        kinds = { kind.id: kind for kind in bu_history.kinds() }
        self.assertEqual( list( kinds ), list( bu_history.KIND_IDS ) )
        self.assertEqual( sorted( k for k, kind in kinds.items() if kind.default ),
                          [ 'logs', 'previews', 'run_history', 'transcodes' ] )
        self.assertEqual( sorted( k for k, kind in kinds.items() if kind.confirm ),
                          [ 'detections', 'library', 'run_keys', 'stats' ] )
        self.assertEqual( sorted( k for k, kind in kinds.items() if not kind.ranged ), [ 'library', 'run_keys' ] )
        self.assertFalse( kinds[ 'feedback' ].default )   # the user's own notes are never ticked for them
        for kind in kinds.values():
            self.assertFalse( kind.default and kind.confirm, kind.id )


class RefusalTest( _Tree ):

    def test_unknown_kind_clears_nothing( self ):
        with self.assertRaises( bu_history.HistoryError ) as ctx:
            self.clear( [ 'previews', 'nope' ] )
        self.assertEqual( ctx.exception.code, 'storage.unknown_category' )
        self.assertEqual( self.trash.moved, [] )

    def test_costly_kind_needs_confirm_and_nothing_goes_first( self ):
        with self.assertRaises( bu_history.HistoryError ) as ctx:
            self.clear( [ 'previews', 'detections' ] )
        self.assertEqual( ctx.exception.code, 'storage.needs_confirm' )
        self.assertEqual( self.trash.moved, [] )
        self.clear( [ 'detections' ], confirm=True )
        self.assertFalse( os.path.exists( os.path.join( self.detections, 'd.json' ) ) )

    def test_refused_while_a_job_holds_the_queue( self ):
        with self.assertRaises( bu_history.HistoryError ) as ctx:
            self.clear( [ 'previews' ], busy=True )
        self.assertEqual( ctx.exception.code, 'storage.busy' )
        self.assertEqual( self.trash.moved, [] )


class RangeTest( _Tree ):

    def test_range_clears_only_newer_files( self ):
        result = self.clear( [ 'previews' ], since=NOW - HOUR )
        self.assertEqual( result, [ { 'id': 'previews', 'bytes': 10, 'files': 1, 'rows': 0 } ] )
        self.assertTrue( os.path.exists( os.path.join( self.previews, 'old.jpg' ) ) )
        self.assertFalse( os.path.exists( os.path.join( self.previews, 'new.jpg' ) ) )

    def test_unranged_kind_ignores_the_range( self ):
        keys = os.path.join( self.root, 'cache', 'run_keys' )
        _write( os.path.join( keys, 'old.json' ), mtime=NOW - 30 * 24 * HOUR )
        self.kinds.append( bu_history.Kind( 'run_keys', confirm=True, ranged=False, paths=[ keys ] ) )
        found = { row[ 'id' ]: row for row in bu_history.summary( self.kinds, since=NOW - HOUR ) }
        self.assertEqual( found[ 'run_keys' ][ 'files' ], 1 )
        self.clear( [ 'run_keys' ], since=NOW - HOUR, confirm=True )
        self.assertEqual( os.listdir( keys ), [] )

    def test_all_time_clears_everything_to_the_trash( self ):
        self.clear( [ 'previews', 'transcodes' ] )
        self.assertEqual( sorted( os.path.basename( src ) for src, _ in self.trash.moved ),
                          [ 'a.mkv', 'new.jpg', 'old.jpg' ] )
        for _src, target in self.trash.moved:
            self.assertTrue( os.path.exists( target ) )

    def test_permanent_deletes_and_uses_no_trash( self ):
        self.clear( [ 'transcodes' ], permanent=True )
        self.assertEqual( self.trash.moved, [] )
        self.assertEqual( os.listdir( self.transcodes ), [] )

    def test_summary_matches_what_clear_removes( self ):
        found = { row[ 'id' ]: row for row in bu_history.summary( self.kinds, since=NOW - HOUR ) }
        self.assertEqual( ( found[ 'previews' ][ 'bytes' ], found[ 'previews' ][ 'files' ] ), ( 10, 1 ) )
        self.assertEqual( ( found[ 'transcodes' ][ 'bytes' ], found[ 'transcodes' ][ 'files' ] ), ( 30, 1 ) )
        cleared = { row[ 'id' ]: row for row in self.clear( [ 'previews', 'transcodes' ], since=NOW - HOUR ) }
        for kind_id in ( 'previews', 'transcodes' ):
            self.assertEqual( cleared[ kind_id ][ 'bytes' ], found[ kind_id ][ 'bytes' ] )
            self.assertEqual( cleared[ kind_id ][ 'files' ], found[ kind_id ][ 'files' ] )


class LogTest( _Tree ):

    def test_open_log_is_cut_back_in_place( self ):
        path = os.path.join( self.logs, 'ui.log' )
        os.makedirs( self.logs )
        writer = open( path, 'a', encoding='utf-8' )
        self.addCleanup( writer.close )
        writer.write( '%s [INFO ] played secret-name.mp4\n'%( _stamp( NOW - 60 ), ) )
        writer.flush()
        inode = os.stat( path ).st_ino
        self.clear( [ 'logs' ] )
        self.assertEqual( os.path.getsize( path ), 0 )
        self.assertEqual( os.stat( path ).st_ino, inode )
        writer.write( '%s [INFO ] after\n'%( _stamp( NOW ), ) )
        writer.flush()
        with open( path, encoding='utf-8' ) as handle:
            self.assertEqual( handle.read().count( '\n' ), 1 )
        ( _src, copy ), = self.trash.moved
        with open( copy, encoding='utf-8' ) as handle:
            self.assertIn( 'secret-name.mp4', handle.read() )

    def test_range_keeps_older_lines_and_their_continuations( self ):
        path = os.path.join( self.logs, 'engine.log' )
        lines = [ 'before the first stamp\n',
                  '%s [INFO ] old run\n'%( _stamp( NOW - 3 * HOUR ), ),
                  'Traceback continuation of the old line\n',
                  '[%s] [INFO] new run of private.mp4\n'%( _stamp( NOW - 60 ), ),
                  'continuation of the new line\n' ]
        _write( path, ''.join( lines ), mtime=NOW - 30 )
        self.clear( [ 'logs' ], since=NOW - HOUR )
        with open( path, encoding='utf-8' ) as handle:
            self.assertEqual( handle.read(), ''.join( lines[:3] ) )
        ( _src, copy ), = self.trash.moved
        with open( copy, encoding='utf-8' ) as handle:
            self.assertEqual( handle.read(), ''.join( lines[3:] ) )

    def test_untimed_log_goes_whole_when_written_in_range( self ):
        old = os.path.join( self.logs, 'old-transcript.jsonl' )
        new = os.path.join( self.logs, 'engine-transcript.jsonl' )
        _write( old, '{"cmd": "a"}\n', mtime=NOW - 3 * HOUR )
        _write( new, '{"cmd": "b"}\n', mtime=NOW - 60 )
        self.clear( [ 'logs' ], since=NOW - HOUR )
        self.assertEqual( os.path.getsize( old ), 13 )
        self.assertEqual( os.path.getsize( new ), 0 )


class CatalogTest( unittest.TestCase ):

    MARK = 'zqx-private-title-marker'

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='history-db-' )
        self.addCleanup( shutil.rmtree, self.root, True )
        self.path = os.path.join( self.root, 'catalog.sqlite' )
        self.store = bu_store.Store( self.path )
        self.addCleanup( self.store.close )
        with self.store.write() as db:
            db.execute( "INSERT INTO folders ( path, added_at ) VALUES ( ?, ? )", ( '/videos', NOW ) )
            db.execute( "INSERT INTO items ( hash, size, name, added_at ) VALUES ( 'h1', 1, ?, ? )",
                        ( self.MARK + '.mp4', NOW ) )
            for n, ended in enumerate( ( NOW - 3 * HOUR, NOW - 60 ) ):
                db.execute( "INSERT INTO runs ( hash, path, started_at, ended_at, outcome ) "
                            "VALUES ( 'h1', ?, ?, ?, 'processed' )",
                            ( '/videos/%s-%d.mp4'%( self.MARK, n ), ended - 10, ended ) )
            for n, made in enumerate( ( NOW - 3 * HOUR, NOW - 60 ) ):
                db.execute( "INSERT INTO feedback ( hash, at_seconds, verdict, detection_key, censor_key, encode_key, "
                            "created_at, label ) VALUES ( 'h1', 1, 'missed', 'd', 'c', 'e', ?, ? )",
                            ( made, '%s-fb%d'%( self.MARK, n ) ) )
        self.backup = self.path + '.v2.bak'
        _write( self.backup, self.MARK, mtime=NOW - 60 )
        self.pre_restore = self.path + '.pre-restore.bak'
        _write( self.pre_restore, self.MARK, mtime=NOW - 60 )
        self.saved = os.path.join( self.root, 'backups', 'catalog-mine.sqlite' )
        _write( self.saved, self.MARK, mtime=NOW - 60 )
        self.kinds = bu_history.kinds( catalog_path=self.path )

    def _clear( self, ids, **kw ):
        trash = _Trash( os.path.join( self.root, 'trash' ) )
        return bu_history.clear( self.kinds, ids, store=self.store, trash=trash, **kw )

    def _raw( self ):
        data = b''
        for suffix in ( '', '-wal' ):
            if os.path.exists( self.path + suffix ):
                with open( self.path + suffix, 'rb' ) as handle:
                    data += handle.read()
        return data

    def test_run_history_range_keeps_older_runs( self ):
        result = self._clear( [ 'run_history' ], since=NOW - HOUR )
        self.assertEqual( result[0][ 'rows' ], 1 )
        left = [ row[0] for row in self.store.read( 'SELECT path FROM runs' ) ]
        self.assertEqual( left, [ '/videos/%s-0.mp4'%( self.MARK, ) ] )   # the older run stays
        self.assertEqual( self.store.read( 'SELECT COUNT(*) FROM items' )[0][0], 1 )
        self.assertFalse( os.path.exists( self.backup ) )
        self.assertFalse( os.path.exists( self.pre_restore ) )
        self.assertTrue( os.path.exists( self.saved ) )   # a backup the user saved is kept
        self.assertEqual( self.store.read( 'SELECT COUNT(*) FROM feedback' )[0][0], 2 )

    def test_feedback_range_keeps_older_marks( self ):
        result = self._clear( [ 'feedback' ], since=NOW - HOUR )
        self.assertEqual( result[0][ 'rows' ], 1 )
        left = [ row[0] for row in self.store.read( 'SELECT label FROM feedback' ) ]
        self.assertEqual( left, [ '%s-fb0'%( self.MARK, ) ] )
        self.assertEqual( self.store.read( 'SELECT COUNT(*) FROM runs' )[0][0], 2 )

    def test_cleared_rows_do_not_linger_in_the_file( self ):
        self.assertIn( ( self.MARK + '-1' ).encode(), self._raw() )
        self._clear( [ 'run_history' ] )
        raw = self._raw()
        self.assertNotIn( ( self.MARK + '-0' ).encode(), raw )
        self.assertNotIn( ( self.MARK + '-1' ).encode(), raw )
        self.assertIn( ( self.MARK + '.mp4' ).encode(), raw )   # the library entry is kept

    def test_library_is_all_time_and_takes_its_runs( self ):
        rows = { row[ 'id' ]: row for row in bu_history.summary( self.kinds, since=NOW - HOUR, store=self.store ) }
        self.assertEqual( rows[ 'library' ][ 'rows' ], 2 )   # one item and one folder, whatever the range
        self._clear( [ 'library' ], since=NOW - HOUR, confirm=True )
        for table in ( 'items', 'folders', 'runs' ):
            self.assertEqual( self.store.read( 'SELECT COUNT(*) FROM %s'%( table, ) )[0][0], 0, table )
        # Feedback marks have their own box: the library list leaves them.
        self.assertEqual( self.store.read( 'SELECT COUNT(*) FROM feedback' )[0][0], 2 )
        self._clear( [ 'feedback' ] )
        self.assertNotIn( self.MARK.encode(), self._raw() )


_END_TO_END = r'''
import hashlib, json, os, sys
import betaconst
import betautils_catalog_store as bu_store
import betautils_catalog_strip as bu_strip
import betautils_catalog_thumbs as bu_thumbs
import betautils_history as bu_history
import betautils_paths as bu_paths
state = bu_paths.kind_dir( 'state' )
catalog = bu_store.default_path()
# Where traces live, named here on their own (not from kinds()), so a kind
# that loses a path leaves its trace behind and the test sees it.
folders = [ bu_thumbs.folder(), bu_strip.folder(), betaconst.look_preview_dir, betaconst.transcode_cache_dir,
            os.path.join( state, 'logs' ), betaconst.vid_hash_dir, betaconst.pic_hash_dir, betaconst.shot_cut_dir,
            betaconst.run_key_dir, os.path.join( state, 'stats' ), betaconst.benchmark_dir ]
files = [ betaconst.file_hash_cache_path, catalog + '.v3.bak', catalog + '.pre-restore.bak' ]
written = [ os.path.join( folder, 'sub', 'trace.log' ) for folder in folders ] + files
for target in written:
    os.makedirs( os.path.dirname( target ), exist_ok=True )
    with open( target, 'w' ) as handle:
        handle.write( '2030-01-01 00:00:00 [INFO ] trace of %s\\n'%( target, ) )
kept = [ os.path.join( bu_paths.kind_dir( 'data' ), 'backups', 'catalog-saved.sqlite' ),
         os.path.join( bu_paths.kind_dir( 'config' ), 'betaconfig_local.py' ),
         os.path.join( bu_paths.kind_dir( 'data' ), 'model', 'm.onnx' ),
         os.path.join( sys.argv[1], 'out', 'result.mp4' ) ]
for path in kept:
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    with open( path, 'w' ) as handle:
        handle.write( 'keep me' )
digest = lambda p: hashlib.sha256( open( p, 'rb' ).read() ).hexdigest()
before = { p: digest( p ) for p in kept }
store = bu_store.Store()
bu_history.clear( bu_history.kinds(), list( bu_history.KIND_IDS ), confirm=True, permanent=True, store=store )
store.close()
left = []
for kind_dir in ( 'data', 'cache', 'state' ):
    for root, _dirs, names in os.walk( bu_paths.kind_dir( kind_dir ) ):
        for name in names:
            full = os.path.join( root, name )
            if os.path.getsize( full ) and b'trace of' in open( full, 'rb' ).read():
                left.append( full )
print( json.dumps( { 'written': len( written ), 'left': left,
                     'kept': all( os.path.exists( p ) and digest( p ) == before[p] for p in kept ) } ) )
'''


class EndToEndTest( unittest.TestCase ):

    def test_every_kind_cleared_and_settings_models_outputs_untouched( self ):
        root = tempfile.mkdtemp( prefix='history-e2e-' )
        self.addCleanup( shutil.rmtree, root, True )
        env = { k: v for k, v in os.environ.items() if not k.startswith( 'BETASUITE_' ) }
        for kind in ( 'CONFIG', 'DATA', 'CACHE', 'STATE' ):
            env[ 'XDG_%s_HOME'%( kind, ) ] = os.path.join( root, kind.lower() )
        env[ 'BETASUITE_OUTPUT_DIR' ] = os.path.join( root, 'out' )
        env[ 'BETASUITE_INPUT_DIR' ] = os.path.join( root, 'in' )
        env[ 'PYTHONPATH' ] = REPO
        done = subprocess.run( [ sys.executable, '-c', _END_TO_END, root ], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=120 )
        self.assertEqual( done.returncode, 0, done.stderr[-2000:] )
        found = json.loads( done.stdout.strip().splitlines()[-1] )
        self.assertEqual( found[ 'written' ], 14 )
        self.assertEqual( found[ 'left' ], [] )
        self.assertTrue( found[ 'kept' ] )


class _Jobs:

    def __init__( self ):
        self.forgot = 0

    def busy_job( self ):
        return None

    def forget_ended( self ):
        self.forgot += 1


class _Server:

    def __init__( self, store ):
        import logging
        self._logger = logging.getLogger( 'test_history' )
        self.jobs = _Jobs()
        self.clock = lambda: NOW
        self._store = store


class ServeTest( unittest.TestCase ):

    def setUp( self ):
        from unittest import mock
        import betautils_serve_storage as bu_serve_storage
        self.serve = bu_serve_storage
        root = tempfile.mkdtemp( prefix='history-serve-' )
        self.addCleanup( shutil.rmtree, root, True )
        self.store = bu_store.Store( os.path.join( root, 'catalog.sqlite' ) )
        self.addCleanup( self.store.close )
        trees = _Tree( 'run' )
        trees.setUp()
        self.addCleanup( trees.doCleanups )
        self.kinds = trees.kinds
        patches = [ mock.patch.object( bu_serve_storage, '_store', lambda server: self.store ),
                    mock.patch.object( bu_history, 'kinds', lambda: self.kinds ) ]
        for patch in patches:
            patch.start()
            self.addCleanup( patch.stop )
        self.server = _Server( self.store )

    def test_history_answer_matches_its_schema( self ):
        answer = self.serve.storage_history( self.server, { 'range': 'hour' } )
        schemas.validate_command_result( 'storage.history', answer )
        self.assertEqual( [ row[ 'id' ] for row in answer[ 'kinds' ] ], [ 'previews', 'transcodes', 'logs', 'detections' ] )

    def test_clear_answer_and_ended_jobs_forgotten( self ):
        from unittest import mock
        with mock.patch.object( bu_history, 'clear', return_value=[ { 'id': 'run_history', 'bytes': 0, 'files': 0, 'rows': 3 } ] ):
            answer = self.serve.storage_clear_history( self.server, { 'kinds': [ 'run_history' ], 'permanent': True } )
        schemas.validate_command_result( 'storage.clear_history', answer )
        self.assertFalse( answer[ 'trashed' ] )
        self.assertEqual( self.server.jobs.forgot, 1 )

    def test_bad_range_and_kinds_are_refused( self ):
        import betautils_protocol as bu_proto
        for data in ( { 'kinds': [ 'previews' ], 'range': 'year' }, { 'kinds': [] }, { 'kinds': 'previews' },
                      { 'kinds': [ 'previews' ], 'permanent': 'yes' } ):
            with self.subTest( data=data ), self.assertRaises( bu_proto.ProtocolError ) as ctx:
                self.serve.storage_clear_history( self.server, data )
            self.assertEqual( ctx.exception.code, 'command.bad_params' )

    def test_range_names_map_to_since( self ):
        self.assertIsNone( self.serve._since( {}, lambda: NOW ) )
        self.assertEqual( self.serve._since( { 'range': 'day' }, lambda: NOW ), NOW - 86400 )


if __name__ == '__main__':
    unittest.main()
