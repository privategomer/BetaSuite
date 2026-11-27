"""
test_catalog_censors.py - the library's censors applied, and the filter
by them.

Covers reading a censor key's record (labels and look kinds, weighted
style lists, a missing or broken record), merging several outputs, the
filter's meaning (one output must hide every label and use every look),
and catalog.list through real outputs on disk, with records written to
the test run-key folder.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import betaconst
import betautils_cache_paths as bu_cache
import betautils_catalog_censors as bu_censors
import betautils_hash as bu_hash

from tests.test_catalog_scan import _CatalogTest, _remove

BREAST_MIXED = { 'exposed_breast': { 'censor_style': [ { 'type': 'blur', 'weight': 1 }, { 'type': 'pixel', 'weight': 1 },
                                                       { 'type': 'blur', 'weight': 2 } ] } }
VULVA_BAR = { 'exposed_vulva': { 'censor_style': { 'type': 'bar' } } }


class _Records( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-censors-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        patcher = mock.patch.object( betaconst, 'run_key_dir', self.dir )
        patcher.start()
        self.addCleanup( patcher.stop )
        bu_censors.forget_cache()
        self.addCleanup( bu_censors.forget_cache )

    def record( self, key, labels ):
        with open( os.path.join( self.dir, 'censor-%s.json'%( key, ) ), 'w' ) as handle:
            json.dump( { 'kind': 'censor', 'key': key, 'identity': { 'labels': labels, 'variant': 'x' } }, handle )


class TestRecords( _Records ):

    def test_labels_and_their_look_kinds( self ):
        self.record( 'k1', dict( BREAST_MIXED, **VULVA_BAR ) )
        self.assertEqual( bu_censors.censors_of( 'k1' ), { 'exposed_breast': [ 'blur', 'pixel' ],
                                                           'exposed_vulva': [ 'bar' ] } )

    def test_a_missing_or_broken_record_is_unknown( self ):
        self.assertIsNone( bu_censors.censors_of( 'nope' ) )
        with open( os.path.join( self.dir, 'censor-bad.json' ), 'w' ) as handle:
            handle.write( '{' )
        self.assertIsNone( bu_censors.censors_of( 'bad' ) )
        with open( os.path.join( self.dir, 'censor-odd.json' ), 'w' ) as handle:
            json.dump( { 'identity': {} }, handle )
        self.assertIsNone( bu_censors.censors_of( 'odd' ) )
        self.assertIsNone( bu_censors.censors_of( None ) )

    def test_a_record_is_read_once( self ):
        self.record( 'k1', VULVA_BAR )
        self.assertEqual( bu_censors.censors_of( 'k1' ), { 'exposed_vulva': [ 'bar' ] } )
        self.record( 'k1', BREAST_MIXED )
        self.assertEqual( bu_censors.censors_of( 'k1' ), { 'exposed_vulva': [ 'bar' ] }, "records never change" )

    def test_entries_without_a_type_are_left_out( self ):
        self.record( 'k1', { 'exposed_feet': { 'censor_style': [ { 'weight': 1 }, 'x' ] }, 'exposed_belly': 'x' } )
        self.assertEqual( bu_censors.censors_of( 'k1' ), { 'exposed_feet': [] } )


class TestSummaryAndMatch( unittest.TestCase ):

    A = { 'exposed_breast': [ 'blur', 'pixel' ] }
    B = { 'exposed_vulva': [ 'bar' ], 'exposed_breast': [ 'sticker' ] }

    def test_summary_merges_outputs( self ):
        self.assertEqual( bu_censors.summary( [ self.A, None, self.B ] ), [
            { 'label': 'exposed_breast', 'looks': [ 'blur', 'pixel', 'sticker' ] },
            { 'label': 'exposed_vulva', 'looks': [ 'bar' ] } ] )
        self.assertEqual( bu_censors.summary( [] ), [] )

    def test_one_output_must_hide_every_label( self ):
        self.assertTrue( bu_censors.matches( [ self.A, self.B ], labels=[ 'exposed_vulva', 'exposed_breast' ] ) )
        self.assertFalse( bu_censors.matches( [ self.A, { 'exposed_vulva': [ 'bar' ] } ],
                                              labels=[ 'exposed_vulva', 'exposed_breast' ] ),
                          "two outputs that each hid one label are not one that hid both" )

    def test_looks_are_those_used_on_the_labels_asked_for( self ):
        self.assertTrue( bu_censors.matches( [ self.B ], labels=[ 'exposed_vulva' ], looks=[ 'bar' ] ) )
        self.assertFalse( bu_censors.matches( [ self.B ], labels=[ 'exposed_vulva' ], looks=[ 'sticker' ] ) )
        self.assertTrue( bu_censors.matches( [ self.B ], looks=[ 'sticker', 'bar' ] ), "any label when none asked" )
        self.assertFalse( bu_censors.matches( [ self.B ], looks=[ 'bar', 'pixel' ] ), "every look asked for" )

    def test_an_unreadable_output_does_not_hide_a_matching_one( self ):
        self.assertTrue( bu_censors.matches( [ None, self.A ], labels=[ 'exposed_breast' ] ) )

    def test_no_criteria_matches_any_readable_output_and_unknown_matches_nothing( self ):
        self.assertTrue( bu_censors.matches( [ self.A ] ) )
        self.assertFalse( bu_censors.matches( [ None ] ) )
        self.assertFalse( bu_censors.matches( [] ) )


class TestRequest( unittest.TestCase ):

    def filters( self, censored ):
        import betautils_serve_catalog as bu_serve_catalog
        return bu_serve_catalog._filters( { 'filter': { 'censored': censored } } )

    def test_labels_and_looks_are_read_as_given( self ):
        self.assertEqual( self.filters( { 'labels': [ 'exposed_breast' ], 'looks': [ 'blur' ] } ),
                          { 'censored': { 'labels': [ 'exposed_breast' ], 'looks': [ 'blur' ] } } )
        self.assertEqual( self.filters( {} ), { 'censored': {} } )

    def test_bad_requests( self ):
        import betautils_protocol as bu_proto
        for censored, field in ( ( [ 'x' ], 'filter.censored' ), ( { 'labels': 'x' }, 'filter.censored.labels' ),
                                 ( { 'looks': [ 3 ] }, 'filter.censored.looks' ) ):
            with self.assertRaises( bu_proto.ProtocolError ) as raised:
                self.filters( censored )
            self.assertEqual( raised.exception.error['params'], { 'field': field, 'reason': 'type' } )


class TestLibrary( _CatalogTest ):

    def setUp( self ):
        super().setUp()
        self.keys = tempfile.mkdtemp( prefix='betasuite-censor-keys-' )
        self.addCleanup( shutil.rmtree, self.keys, True )
        patcher = mock.patch.object( betaconst, 'run_key_dir', self.keys )
        patcher.start()
        self.addCleanup( patcher.stop )
        bu_censors.forget_cache()
        self.addCleanup( bu_censors.forget_cache )
        self.watch()
        os.makedirs( betaconst.video_path_censored, exist_ok=True )

    def record( self, key, labels ):
        with open( os.path.join( self.keys, 'censor-%s.json'%( key, ) ), 'w' ) as handle:
            json.dump( { 'identity': { 'labels': labels } }, handle )

    def source_with_output( self, name, censor_key ):
        path = self.write( name )
        self.settle_and_hash()
        digest = bu_hash.md5_for_file( path, 16, use_memo=False )
        base = bu_cache.video_output_basename( os.path.splitext( name )[0], digest, [ 640 ], 9, 'd1', censor_key, 'e3' )
        output = self.write( base + '.mp4', where=betaconst.video_path_censored )
        self.addCleanup( _remove, output )
        self.catalog.scanner.pass_now()
        return digest, output

    def names( self, **censored ):
        return sorted( item['name'] for item in self.items( censored=censored ) )

    def test_items_show_and_filter_by_the_censors_of_their_outputs( self ):
        self.record( 'c1', BREAST_MIXED )
        self.record( 'c2', VULVA_BAR )
        breast, _ = self.source_with_output( 'a.mp4', 'c1' )
        self.source_with_output( 'b.mp4', 'c2' )
        self.write( 'never.mp4' )
        self.settle_and_hash()
        shown = { item['name']: item['censors'] for item in self.items() }
        self.assertEqual( shown, { 'a.mp4': [ { 'label': 'exposed_breast', 'looks': [ 'blur', 'pixel' ] } ],
                                   'b.mp4': [ { 'label': 'exposed_vulva', 'looks': [ 'bar' ] } ],
                                   'never.mp4': [] } )
        self.assertEqual( self.catalog.get( breast )['censors'], shown['a.mp4'] )
        self.assertEqual( self.names( labels=[ 'exposed_breast' ] ), [ 'a.mp4' ] )
        self.assertEqual( self.names( looks=[ 'bar' ] ), [ 'b.mp4' ] )
        self.assertEqual( self.names( labels=[ 'exposed_breast' ], looks=[ 'bar' ] ), [] )
        self.assertEqual( self.names(), [ 'a.mp4', 'b.mp4' ], "an empty filter: anything censored" )

    def test_a_deleted_output_no_longer_counts( self ):
        self.record( 'c1', BREAST_MIXED )
        _digest, output = self.source_with_output( 'a.mp4', 'c1' )
        os.remove( output )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.items()[0]['censors'], [] )
        self.assertEqual( self.names( labels=[ 'exposed_breast' ] ), [] )

    def test_an_output_with_no_record_shows_nothing_and_matches_nothing( self ):
        self.source_with_output( 'a.mp4', 'gone' )
        self.assertEqual( self.items()[0]['censors'], [] )
        self.assertEqual( self.names(), [] )


if __name__ == '__main__':
    unittest.main()
