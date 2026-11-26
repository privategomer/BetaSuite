"""
Run history in the catalog: the schema step from version 1, what
recording a run does to its item, sources the engine moves, queued and
processing while a job runs, and the command line's record.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest

import betaconfig
import betatv
import betautils_catalog as bu_catalog
import betautils_catalog_history as bu_history
import betautils_catalog_index as bu_index
import betautils_catalog_status as bu_status
import betautils_catalog_store as bu_store
import betautils_hash as bu_hash
import betautils_job as bu_job
import betautils_run_record as bu_run_record

from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip

FIXTURE_V1 = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'golden', 'catalog_v1.sql' )


class _Tmp( unittest.TestCase ):
    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-history-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.db_path = os.path.join( self.root, 'catalog.sqlite' )


class TestFromVersionOne( _Tmp ):
    def load_fixture( self ):
        db = sqlite3.connect( self.db_path )
        with open( FIXTURE_V1 ) as handle:
            db.executescript( handle.read() )
        db.close()

    def test_a_version_one_file_moves_to_the_current_version_keeping_everything( self ):
        self.load_fixture()
        store = bu_store.Store( self.db_path )
        self.addCleanup( store.close )
        self.assertEqual( store.read( 'PRAGMA user_version' )[0][0], bu_store.LATEST_VERSION )
        self.assertEqual( [ ( row['hash'], row['status'], row['type_pin'] ) for row in store.read(
            'SELECT hash, status, type_pin FROM items ORDER BY hash' ) ],
            [ ( '0123456789abcdef', 'done', 'quick_cut' ), ( 'fedcba9876543210', 'new', None ) ] )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM locations' )[0][0], 2 )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM outputs' )[0][0], 1 )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM runs' )[0][0], 0 )
        self.assertEqual( [ row[0] for row in store.read( 'SELECT version FROM schema_migrations ORDER BY version' ) ],
                          list( range( 1, bu_store.LATEST_VERSION + 1 ) ) )

    def test_the_version_one_file_is_kept_as_a_backup( self ):
        self.load_fixture()
        bu_store.Store( self.db_path ).close()
        backup = sqlite3.connect( self.db_path + '.v1.bak' )
        self.addCleanup( backup.close )
        self.assertEqual( backup.execute( 'PRAGMA user_version' ).fetchone()[0], 1 )
        self.assertEqual( backup.execute( 'SELECT COUNT(*) FROM items' ).fetchone()[0], 2 )

    def test_the_fixture_is_what_version_one_wrote( self ):
        with open( FIXTURE_V1 ) as handle:
            text = handle.read()
        self.assertIn( bu_store.MIGRATIONS[0][2].strip().splitlines()[0].rstrip( ' (' ), text )
        self.assertNotIn( 'CREATE TABLE runs', text )


class _HistoryTest( _Tmp ):
    def setUp( self ):
        super().setUp()
        self.store = bu_store.Store( self.db_path )
        self.addCleanup( self.store.close )
        self.index = bu_index.Index( self.store )
        self.history = bu_history.History( self.store, self.index )
        self.folder = os.path.join( self.root, 'videos' )
        os.makedirs( self.folder )

    def make( self, name, size=3000 ):
        path = os.path.join( self.folder, name )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'wb' ) as handle:
            handle.write( os.urandom( size ) )
        return path, bu_hash.md5_for_file( path, 16, use_memo=False )

    def run_of( self, path, digest, outcome, reason=None, **fields ):
        run = bu_run_record.RunRecord( path, program='engine', job='j1' )
        run.hash = digest
        for name, value in fields.items():
            setattr( run, name, value )
        return run.end( outcome, reason )

    def item( self, digest ):
        return self.index.item_row( digest )


class TestRecording( _HistoryTest ):
    def test_a_run_on_a_file_the_catalog_does_not_know_adds_it( self ):
        path, digest = self.make( 'new.mp4' )
        touched = self.history.record( self.run_of( path, digest, 'processed' ) )
        self.assertEqual( touched, { digest } )
        self.assertEqual( self.item( digest )['status'], 'done' )
        self.assertEqual( [ location['path'] for location in self.index.locations_of( digest ) ],
                          [ bu_index.norm( path ) ] )

    def test_each_outcome_leaves_its_status( self ):
        cases = ( ( 'processed', None, 'done' ), ( 'skipped', 'output.present', 'done' ),
                  ( 'skipped', 'source.not_video', 'failed' ), ( 'failed', 'file.failed', 'failed' ),
                  ( 'out_of_space', 'disk.full', 'failed' ) )
        for position, ( outcome, reason, status ) in enumerate( cases ):
            with self.subTest( outcome=outcome, reason=reason ):
                path, digest = self.make( 'f%d.mp4'%( position ) )
                self.history.record( self.run_of( path, digest, outcome, reason ) )
                self.assertEqual( self.item( digest )['status'], status )

    def test_a_preview_is_history_but_changes_no_status_or_type( self ):
        path, digest = self.make( 'p.mp4' )
        self.history.record( self.run_of( path, digest, 'processed', preview=True, profile='scene', rule='default' ) )
        row = self.item( digest )
        self.assertEqual( ( row['status'], row['last_profile'] ), ( 'new', None ) )
        [ run ] = self.history.runs_of( digest )
        self.assertTrue( run['preview'] )

    def test_the_run_is_kept_with_everything_the_record_said( self ):
        path, digest = self.make( 'k.mp4' )
        self.history.record( self.run_of(
            path, digest, 'failed', 'file.failed', profile='quick_cut', rule='measured',
            rule_params={ 'signal': 'short_shot_fraction' }, keys={ 'detection': 'd1', 'censor': 'c2', 'encode': 'e3' },
            error_type='RenderError', stage='render', sample_fps=18 ) )
        [ run ] = self.history.runs_of( digest )
        self.assertEqual( ( run['program'], run['job'], run['outcome'], run['reason'], run['profile'], run['rule'],
                            run['rule_params'], run['keys'], run['error_type'], run['stage'] ),
                          ( 'engine', 'j1', 'failed', 'file.failed', 'quick_cut', 'measured',
                            { 'signal': 'short_shot_fraction' }, { 'detection': 'd1', 'censor': 'c2', 'encode': 'e3' },
                            'RenderError', 'render' ) )
        row = self.item( digest )
        self.assertEqual( ( row['last_profile'], row['last_rule'] ), ( 'quick_cut', 'measured' ) )

    def test_a_full_disk_is_kept_as_a_failure( self ):
        path, digest = self.make( 'd.mp4' )
        self.history.record( self.run_of( path, digest, 'out_of_space', 'disk.full' ) )
        self.assertEqual( self.history.runs_of( digest )[0]['outcome'], 'failed' )

    def test_a_file_never_read_leaves_no_history( self ):
        run = bu_run_record.RunRecord( os.path.join( self.folder, 'gone.mp4' ) ).end( 'skipped', 'source.unreadable' )
        self.assertEqual( self.history.record( run ), set() )
        self.assertEqual( self.store.read( 'SELECT COUNT(*) FROM runs' )[0][0], 0 )

    def test_history_is_newest_first( self ):
        path, digest = self.make( 'n.mp4' )
        for outcome, reason in ( ( 'failed', 'file.failed' ), ( 'processed', None ) ):
            self.history.record( self.run_of( path, digest, outcome, reason ) )
        self.assertEqual( [ run['outcome'] for run in self.history.runs_of( digest ) ], [ 'processed', 'failed' ] )

    def test_the_output_made_is_the_items_output( self ):
        path, digest = self.make( 'o.mp4' )
        output = os.path.join( self.root, 'out', 'o-%s-640-9-d1-c2-e3.mp4'%( digest ) )
        os.makedirs( os.path.dirname( output ) )
        open( output, 'wb' ).close()
        self.history.record( self.run_of( path, digest, 'processed', output=output ) )
        self.assertEqual( [ found['path'] for found in self.index.outputs_of( digest ) ], [ bu_index.norm( output ) ] )


class TestMoves( _HistoryTest ):
    def test_a_moved_source_is_the_same_item_at_its_new_path( self ):
        path, digest = self.make( 'uncensored/scene/x.mp4' )
        run = self.run_of( path, digest, 'processed', profile='scene', rule='source_folder' )
        self.history.record( run )
        new = os.path.join( self.folder, 'completed', 'scene', 'x.mp4' )
        os.makedirs( os.path.dirname( new ) )
        os.rename( path, new )
        run.moved_to = new
        self.history.moved( run )
        self.assertEqual( [ location['path'] for location in self.index.locations_of( digest ) ], [ bu_index.norm( new ) ] )
        row = self.item( digest )
        self.assertEqual( ( row['status'], row['last_profile'] ), ( 'done', 'scene' ) )

    def test_a_move_to_another_disk_keeps_one_location( self ):
        path, digest = self.make( 'y.mp4' )
        run = self.run_of( path, digest, 'processed' )
        self.history.record( run )
        new = os.path.join( self.folder, 'elsewhere', 'y.mp4' )
        os.makedirs( os.path.dirname( new ) )
        shutil.copyfile( path, new )   # a new inode, as across disks
        os.remove( path )
        run.moved_to = new
        self.history.moved( run )
        self.assertEqual( [ location['path'] for location in self.index.locations_of( digest ) ], [ bu_index.norm( new ) ] )


class TestItemStatus( _HistoryTest ):
    def setUp( self ):
        super().setUp()
        self.status = bu_status.ItemStatus( self.index )
        self.paths, self.digests = [], []
        for name in ( 'a.mp4', 'b.mp4', 'c.mp4' ):
            path, digest = self.make( name )
            self.index.record_hash( path, bu_index.Stat.of( os.stat( path ) ), digest )
            self.paths.append( path )
            self.digests.append( digest )

    def event( self, name, **data ):
        return self.status.job_event( name, dict( data, job='j1' ), self.paths )

    def shown( self ):
        return [ ( self.status.status_of( digest ) or ( None, ) )[0] for digest in self.digests ]

    def test_a_job_moves_its_files_through_queued_and_processing_and_back( self ):
        self.assertEqual( self.event( 'job.state', state='queued' ), set( self.digests ) )
        self.assertEqual( self.shown(), [ 'queued' ] * 3 )
        self.event( 'job.state', state='running' )
        self.event( 'job.file', file=0, state='started' )
        self.assertEqual( self.shown(), [ 'processing', 'queued', 'queued' ] )
        self.event( 'job.file', file=0, state='processed' )
        self.assertEqual( self.shown(), [ None, 'queued', 'queued' ] )
        self.event( 'job.file', file=1, state='started' )
        self.event( 'job.state', state='paused' )
        self.assertEqual( self.shown(), [ None, 'queued', 'queued' ] )
        self.event( 'job.state', state='cancelled' )
        self.assertEqual( self.shown(), [ None, None, None ] )

    def test_files_the_catalog_does_not_know_are_left_out( self ):
        unknown = os.path.join( self.folder, 'unknown.mp4' )
        touched = self.status.job_event( 'job.state', { 'job': 'j2', 'state': 'queued' }, [ unknown, self.paths[0] ] )
        self.assertEqual( touched, { self.digests[0] } )

    def test_a_stopped_job_lets_go_of_its_files( self ):
        self.event( 'job.state', state='queued' )
        self.event( 'job.file', file=0, state='started' )
        self.event( 'job.state', state='stopped' )
        self.assertEqual( self.shown(), [ None, None, None ] )


class TestCatalogShowsHistory( _Tmp ):
    def test_get_answers_the_history_and_list_the_live_status( self ):
        changes = []
        catalog = bu_catalog.Catalog( path=self.db_path, seed_folders=[],
                                      on_change=lambda items, reason: changes.append( reason ) )
        self.addCleanup( catalog.store.close )
        path = os.path.join( self.root, 'v.mp4' )
        with open( path, 'wb' ) as handle:
            handle.write( os.urandom( 1000 ) )
        digest = bu_hash.md5_for_file( path, 16, use_memo=False )
        run = bu_run_record.RunRecord( path, program='betatv' )
        run.hash = digest
        catalog.sink.record( run.end( 'processed' ) )
        self.assertIn( 'job', changes )
        self.assertEqual( [ entry['program'] for entry in catalog.get( digest )['history'] ], [ 'betatv' ] )
        catalog.job_event( 'job.state', { 'job': 'j', 'state': 'queued' }, [ path ] )
        self.assertEqual( catalog.list()['items'][0]['status'], 'queued' )
        catalog.job_event( 'job.state', { 'job': 'j', 'state': 'completed' }, [ path ] )
        self.assertEqual( catalog.list()['items'][0]['status'], 'done' )


class TestCommandLineRecords( BetaTvHarness ):
    """The CLI's history sink, on real runs with the stub detector."""

    def setUp( self ):
        super().setUp()
        self.db_path = os.path.join( self.tmpdir, 'catalog.sqlite' )
        self.cli = bu_history.CliHistory.open( self.logger, path=self.db_path )
        self.addCleanup( self.cli.close )
        self.recorder = bu_run_record.RunRecorder(
            [ bu_run_record.StatsLogSink(), bu_run_record.FailureLogSink(), self.cli.sink ],
            program='betatv', logger=self.logger )

    def run_file( self, folder, fname='clip.mp4' ):
        reporter = bu_job.FileReporter( None, 'cli', 0, self.recorder )
        return betatv.process_one_video( folder, fname, self.censored, 0, 1, self.detector.get_session(),
                                         False, 20, 'ultrafast', self.logger, reporter=reporter )

    def test_a_command_line_run_lands_in_the_history( self ):
        clip = os.path.join( self.uncensored, 'clip.mp4' )
        _build_clip( clip )
        self.assertEqual( self.run_file( self.uncensored ), 'processed' )
        digest = bu_hash.md5_for_file( clip, 16, use_memo=False )
        [ run ] = self.cli.history.runs_of( digest )
        self.assertEqual( ( run['program'], run['outcome'] ), ( 'betatv', 'processed' ) )
        self.assertEqual( self.cli.index.item_row( digest )['status'], 'done' )
        self.assertEqual( len( self.cli.index.outputs_of( digest ) ), 1 )

    def test_a_processed_run_keeps_its_timing_and_a_rerun_knows_detection_was_cached( self ):
        import json
        clip = os.path.join( self.uncensored, 'clip.mp4' )
        _build_clip( clip )
        self.assertEqual( self.run_file( self.uncensored ), 'processed' )
        for name in os.listdir( self.censored ):
            os.remove( os.path.join( self.censored, name ) )
        self.assertEqual( self.run_file( self.uncensored ), 'processed' )
        rows = self.cli.store.read( 'SELECT timing FROM runs ORDER BY id' )
        first, second = [ json.loads( row['timing'] ) for row in rows ]
        self.assertEqual( ( first['detect_cache'], second['detect_cache'] ), ( 'fresh', 'cached' ) )
        self.assertEqual( ( first['width'], first['height'] ), ( 128, 96 ) )
        self.assertGreater( first['rendered_frames'], 0 )
        self.assertGreater( first['sampled_frames'], 0 )
        self.assertGreater( first['render_seconds'], 0 )
        self.assertGreaterEqual( first['total_seconds'], first['detect_seconds'] + first['render_seconds'] - 0.01 )
        import betautils_estimate as bu_estimate
        self.assertEqual( first['identity'], bu_estimate.identity() )

    def test_a_source_moved_by_the_engine_is_one_item_done_with_its_type( self ):
        # Carried from the completed-source move: uncensored_vids/scene/ to
        # source/completed/scene/ keeps the item, its status and its type.
        betaconfig.move_completed_sources = True
        folder = os.path.join( self.uncensored, 'scene' )
        os.makedirs( folder )
        clip = os.path.join( folder, 'clip.mp4' )
        _build_clip( clip )
        digest = bu_hash.md5_for_file( clip, 16, use_memo=False )
        self.cli.index.record_hash( clip, bu_index.Stat.of( os.stat( clip ) ), digest )
        self.assertEqual( self.run_file( folder ), 'processed' )
        moved = os.path.join( self.completed, 'scene', 'clip.mp4' )
        self.assertTrue( os.path.exists( moved ) )
        self.assertEqual( [ row[0] for row in self.cli.store.read( 'SELECT hash FROM items' ) ], [ digest ] )
        self.assertEqual( [ location['path'] for location in self.cli.index.locations_of( digest ) ],
                          [ bu_index.norm( moved ) ] )
        row = self.cli.index.item_row( digest )
        self.assertEqual( ( row['status'], row['last_profile'], row['last_rule'] ), ( 'done', 'scene', 'source_folder' ) )
        import betautils_catalog_types as bu_types
        self.assertEqual( bu_types.shown_type( moved )['profile'], 'scene' )

    def test_a_catalog_that_cannot_open_is_logged_and_the_run_goes_on( self ):
        with open( os.path.join( self.tmpdir, 'bad.sqlite' ), 'wb' ) as handle:
            handle.write( b'not a database' * 100 )
        with self.assertLogs( self.logger, level='WARNING' ):
            self.assertIsNone( bu_history.CliHistory.open( self.logger, path=os.path.join( self.tmpdir, 'bad.sqlite' ) ) )


if __name__ == '__main__':
    unittest.main()
