"""
The per-video type pin: it beats every other rule, a run uses it (the
app's jobs and the command line alike), it is keyed only while set, and
the catalog refuses a pin it cannot honor.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import betaconfig
import betaconst
import betatv
import betautils_cache_paths as bu_cache
import betautils_catalog as bu_catalog
import betautils_catalog_history as bu_history
import betautils_catalog_index as bu_index
import betautils_catalog_store as bu_store
import betautils_catalog_types as bu_types
import betautils_config as bu_config
import betautils_hash as bu_hash
import betautils_job as bu_job
import betautils_run_record as bu_run_record

from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip


class _Config( unittest.TestCase ):
    """Sets betaconfig values for one test and puts them back."""

    def setUp( self ):
        self._saved = {}

    def tearDown( self ):
        for name, value in self._saved.items():
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def config( self, **values ):
        for name, value in values.items():
            self._saved.setdefault( name, getattr( betaconfig, name, None ) )
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()


class TestPrecedence( _Config ):
    """pin > forced > path pattern > folder rule (1.x roots, then the watched root) > measured."""

    def setUp( self ):
        super().setUp()
        self.config( force_structure_profile=None, profile_by_path_pattern=None,
                     profile_by_source_dir={ 'scene': 'scene', 'quick-cut': 'quick_cut' } )
        self.under_1x = os.path.join( betaconst.video_path_uncensored, 'scene', 'clip.mp4' )
        self.watched = os.path.join( tempfile.gettempdir(), 'watched-root' )
        self.under_watched = os.path.join( self.watched, 'quick-cut', 'clip.mp4' )

    def chosen( self, path, pin=None, roots=() ):
        declared = bu_types.declared_for( path, pin, roots )
        return None if declared is None else ( declared[0], declared[1].code )

    def test_with_nothing_declared_the_scan_decides( self ):
        self.assertIsNone( self.chosen( '/elsewhere/clip.mp4' ) )

    def test_a_watched_folders_rule_decides_when_nothing_above_does( self ):
        self.assertEqual( self.chosen( self.under_watched, roots=( self.watched, ) ), ( 'quick_cut', 'source_folder' ) )

    def test_a_1x_folder_beats_the_watched_folders_rule( self ):
        self.assertEqual( self.chosen( self.under_1x, roots=( os.path.dirname( self.under_1x ), ) ),
                          ( 'scene', 'source_folder' ) )

    def test_a_path_pattern_beats_the_folder_rule( self ):
        self.config( profile_by_path_pattern={ '*clip*': 'quick_cut' } )
        self.assertEqual( self.chosen( self.under_1x ), ( 'quick_cut', 'path_pattern' ) )

    def test_forced_beats_the_path_pattern( self ):
        self.config( profile_by_path_pattern={ '*clip*': 'quick_cut' }, force_structure_profile='scene' )
        self.assertEqual( self.chosen( self.under_1x ), ( 'scene', 'forced' ) )

    def test_the_pin_beats_everything( self ):
        self.config( profile_by_path_pattern={ '*clip*': 'quick_cut' }, force_structure_profile='scene' )
        for path, roots in ( ( self.under_1x, () ), ( self.under_watched, ( self.watched, ) ),
                             ( '/elsewhere/clip.mp4', () ) ):
            self.assertEqual( self.chosen( path, pin='split_screen', roots=roots ), ( 'split_screen', 'pinned' ) )

    def test_a_pin_may_name_a_type_no_measurement_picks( self ):
        names = bu_types.profile_names()
        self.assertIn( 'compilation', names )
        self.assertEqual( self.chosen( '/x.mp4', pin='compilation' ), ( 'compilation', 'pinned' ) )

    def test_a_pin_to_a_type_that_is_gone_falls_through_to_the_rules( self ):
        self.assertEqual( self.chosen( self.under_1x, pin='no_such_type' ), ( 'scene', 'source_folder' ) )

    def test_the_library_shows_the_pin_and_why( self ):
        self.assertEqual( bu_types.shown_type( self.under_1x, pin='quick_cut' ),
                          { 'profile': 'quick_cut', 'rule': 'pinned', 'params': {},
                            'when': 'declared' } )

    def test_the_run_types_lookup_failing_falls_back_to_the_rules( self ):
        def broken( digest ):
            raise RuntimeError( 'catalog gone' )
        types = bu_types.RunTypes( broken, lambda path: () )
        self.assertEqual( types.declared( self.under_1x, 'a' * 16 )[0], 'scene' )


class TestCensorKey( _Config ):
    def test_an_unpinned_file_keeps_its_key( self ):
        self.assertEqual( bu_cache.censor_key( catalog_profile=None ), bu_cache.censor_key() )
        self.assertNotIn( 'catalog_profile', bu_cache.censor_identity() )

    def test_a_pin_moves_the_key_and_each_pin_has_its_own( self ):
        keys = { bu_cache.censor_key(), bu_cache.censor_key( catalog_profile='scene' ),
                 bu_cache.censor_key( catalog_profile='compilation' ) }
        self.assertEqual( len( keys ), 3 )
        self.assertEqual( bu_cache.censor_identity( catalog_profile='scene' )['catalog_profile'], 'scene' )


class TestPinning( unittest.TestCase ):
    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-pin-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.changes = []
        self.catalog = bu_catalog.Catalog( path=os.path.join( self.root, 'c.sqlite' ), seed_folders=[],
                                           on_change=lambda items, reason: self.changes.append( reason ) )
        self.addCleanup( self.catalog.store.close )
        self.path = os.path.join( self.root, 'v.mp4' )
        with open( self.path, 'wb' ) as handle:
            handle.write( os.urandom( 500 ) )
        self.digest = bu_hash.md5_for_file( self.path, 16, use_memo=False )
        self.catalog.index.record_hash( self.path, bu_index.Stat.of( os.stat( self.path ) ), self.digest )

    def refused( self, code, *args ):
        with self.assertRaises( bu_store.CatalogError ) as caught:
            self.catalog.pin_type( *args )
        self.assertEqual( caught.exception.code, code )
        return caught.exception.params

    def test_pin_and_unpin( self ):
        self.catalog.pin_type( self.digest, 'split_screen' )
        self.assertEqual( self.catalog.get( self.digest )['pin'], 'split_screen' )
        self.assertEqual( self.catalog.get( self.digest )['type']['rule'], 'pinned' )
        self.assertIn( 'pin', self.changes )
        self.catalog.pin_type( self.digest, None )
        self.assertIsNone( self.catalog.get( self.digest )['pin'] )

    def test_a_name_that_is_not_a_type_is_refused( self ):
        self.assertEqual( self.refused( 'catalog.bad_type', self.digest, 'slow_motion' ), { 'type': 'slow_motion' } )

    def test_an_unknown_item_is_refused( self ):
        self.assertEqual( self.refused( 'catalog.no_such_item', 'f' * 16, 'scene' ), { 'item': 'f' * 16 } )

    def test_an_item_a_job_holds_is_refused_until_the_job_ends( self ):
        self.catalog.job_event( 'job.state', { 'job': 'j9', 'state': 'queued' }, [ self.path ] )
        self.assertEqual( self.refused( 'catalog.item_busy', self.digest, 'scene' ),
                          { 'item': self.digest, 'job': 'j9' } )
        self.catalog.job_event( 'job.state', { 'job': 'j9', 'state': 'completed' }, [ self.path ] )
        self.catalog.pin_type( self.digest, 'scene' )

    def test_the_run_types_give_the_pin( self ):
        self.catalog.pin_type( self.digest, 'quick_cut' )
        self.assertEqual( self.catalog.run_types().declared( self.path, self.digest )[0], 'quick_cut' )


class TestARunUsesThePin( BetaTvHarness ):
    """A pinned file runs under its pin, from the app's jobs or the command line."""

    def setUp( self ):
        super().setUp()
        self.cli = bu_history.CliHistory.open( self.logger, path=os.path.join( self.tmpdir, 'catalog.sqlite' ) )
        self.addCleanup( self.cli.close )
        self.clip = os.path.join( self.uncensored, 'clip.mp4' )
        _build_clip( self.clip )
        self.digest = bu_hash.md5_for_file( self.clip, 16, use_memo=False )
        self.cli.index.record_hash( self.clip, bu_index.Stat.of( os.stat( self.clip ) ), self.digest )
        self.keep = []

        class Keep:
            required = False

            def record( inner, run ):
                self.keep.append( run )

        self.recorder = bu_run_record.RunRecorder(
            [ bu_run_record.StatsLogSink(), bu_run_record.FailureLogSink(), Keep(), self.cli.sink ],
            program='betatv', logger=self.logger )

    def run_file( self ):
        decisions = []
        reporter = bu_job.FileReporter( lambda name, data: decisions.append( data ), 'cli', 0,
                                        self.recorder, self.cli.run_types() )
        outcome = betatv.process_one_video( self.uncensored, 'clip.mp4', self.censored, 0, 1,
                                            self.detector.get_session(), False, 20, 'ultrafast',
                                            self.logger, reporter=reporter )
        return outcome, decisions

    def test_a_pin_chooses_the_profile_and_names_itself_as_the_rule( self ):
        self.cli.index.set_pin( self.digest, 'compilation' )
        outcome, decisions = self.run_file()
        self.assertEqual( outcome, 'processed' )
        [ decision ] = [ data for data in decisions if data['code'] == 'structure.profile' ]
        self.assertEqual( ( decision['params']['profile'], decision['params']['rule'] ), ( 'compilation', 'pinned' ) )
        self.assertEqual( ( self.keep[0].profile, self.keep[0].rule ), ( 'compilation', 'pinned' ) )
        self.assertEqual( self.cli.index.item_row( self.digest )['last_rule'], 'pinned' )

    def test_a_pinned_file_renders_under_its_own_key_and_unpinned_it_is_as_before( self ):
        _outcome, _ = self.run_file()
        unpinned = self.outputs()
        self.assertEqual( len( unpinned ), 1 )
        self.assertIn( '-c%s-'%( bu_cache.censor_key() ), unpinned[0] )
        self.cli.index.set_pin( self.digest, 'scene' )
        outcome, _ = self.run_file()
        self.assertEqual( outcome, 'processed', 'a pin is a new censor key: rendered again' )
        pinned = sorted( set( self.outputs() ) - set( unpinned ) )
        self.assertEqual( len( pinned ), 1 )
        self.assertIn( '-c%s-'%( bu_cache.censor_key( catalog_profile='scene' ) ), pinned[0] )

    def test_a_watched_folders_rule_is_keyed_and_a_1x_folder_is_not( self ):
        # Turning a watched folder's rule on picks another profile; the
        # output must be its own, never one rendered under the other.
        watched = os.path.join( self.tmpdir, 'watched' )
        clip = os.path.join( watched, 'compilation', 'clip.mp4' )
        os.makedirs( os.path.dirname( clip ) )
        shutil.copyfile( self.clip, clip )
        import betautils_catalog_types as bu_types

        def run_with( roots ):
            decisions = []
            reporter = bu_job.FileReporter( lambda name, data: decisions.append( data ), 'cli', 0, self.recorder,
                                            bu_types.RunTypes( lambda digest: None, lambda path: roots ) )
            outcome = betatv.process_one_video( os.path.dirname( clip ), 'clip.mp4', self.censored, 0, 1,
                                                self.detector.get_session(), False, 20, 'ultrafast', self.logger,
                                                reporter=reporter )
            [ decision ] = [ data for data in decisions if data['code'] == 'structure.profile' ]
            return outcome, decision['params']
        _, plain = run_with( () )
        before = set( self.outputs() )
        outcome, ruled = run_with( ( watched, ) )
        self.assertEqual( ( ruled['profile'], ruled['rule'] ), ( 'compilation', 'source_folder' ) )
        self.assertEqual( outcome, 'processed', 'a new key: rendered, never the old output reused' )
        [ made ] = sorted( set( self.outputs() ) - before )
        self.assertIn( '-c%s-'%( bu_cache.censor_key( catalog_profile='compilation' ) ), made )
        # The 1.x tree's own folder rule is keyed through the config, as before.
        decisions = []
        folder = os.path.join( self.uncensored, 'scene' )
        os.makedirs( folder )
        shutil.copyfile( self.clip, os.path.join( folder, 'clip.mp4' ) )
        reporter = bu_job.FileReporter( lambda name, data: decisions.append( data ), 'cli', 0, self.recorder,
                                        bu_types.RunTypes( lambda digest: None, lambda path: () ) )
        os.makedirs( os.path.join( self.censored, 'scene' ) )
        betatv.process_one_video( folder, 'clip.mp4', os.path.join( self.censored, 'scene' ), 0, 1,
                                  self.detector.get_session(), False, 20, 'ultrafast', self.logger, reporter=reporter )
        names = os.listdir( os.path.join( self.censored, 'scene' ) )
        self.assertTrue( any( '-c%s-'%( bu_cache.censor_key() ) in name for name in names ), names )

    def test_without_a_pin_the_types_lookup_changes_nothing( self ):
        outcome, decisions = self.run_file()
        self.assertEqual( outcome, 'processed' )
        [ decision ] = [ data for data in decisions if data['code'] == 'structure.profile' ]
        self.assertNotEqual( decision['params']['rule'], 'pinned' )


if __name__ == '__main__':
    unittest.main()
