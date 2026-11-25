"""
Watched folders, adding files, re-finding moved and renamed files by
inode and by hash, missing and offline, copies, settling, jobs first,
outputs, and the folders a new catalog watches.

Synthetic files of seeded random bytes; a fake clock; no wall-clock
waits. Everything under the test OS root (tests/__init__.py).
"""

import json
import os
import random
import shutil
import sqlite3
import tempfile
import unittest

import betaconst
import betautils_cache_paths as bu_cache
import betautils_catalog as bu_catalog
import betautils_catalog_index as bu_index
import betautils_catalog_scan as bu_scan
import betautils_catalog_store as bu_store
import betautils_hash as bu_hash


class Clock:
    def __init__( self ):
        self.now = 1000.0

    def __call__( self ):
        return self.now


class _CatalogTest( unittest.TestCase ):
    SETTLE = 5.0

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-scan-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.watched = os.path.join( self.root, 'watched' )
        os.makedirs( self.watched )
        self.clock = Clock()
        self.hashed = []
        self.busy = False
        self.changes = []
        self.random = random.Random( 5 )
        self.catalog = self.open_catalog()

    def open_catalog( self, seed_folders=(), path=None ):
        catalog = bu_catalog.Catalog(
            path=path or os.path.join( self.root, 'catalog.sqlite' ), clock=self.clock,
            on_change=lambda items, reason: self.changes.append( ( items, reason ) ),
            busy=lambda: self.busy, hasher=self.hasher, settle=self.SETTLE,
            seed_folders=list( seed_folders ) )
        self.addCleanup( catalog.store.close )
        return catalog

    def hasher( self, path ):
        self.hashed.append( os.path.basename( path ) )
        return bu_hash.md5_for_file( path, 16, use_memo=False )

    def write( self, relative, size=4096, content=None, where=None ):
        path = os.path.join( where or self.watched, relative )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'wb' ) as handle:
            handle.write( content if content is not None else self.random.randbytes( size ) )
        return path

    def watch( self, folder=None, type_from_folder=False ):
        return self.catalog.add_folder( folder or self.watched, type_from_folder )[0]

    def settle_and_hash( self ):
        """A pass, then the settle time, then hashing: a new file is in."""
        self.catalog.scanner.pass_now()
        self.clock.now += self.SETTLE
        return self.catalog.scanner.hash_ready()

    def items( self, **filters ):
        return self.catalog.list( filters )['items']

    def item( self, digest ):
        return self.catalog.get( digest )


class TestWatchedFolder( _CatalogTest ):
    def test_a_new_file_waits_to_settle_then_is_indexed( self ):
        self.watch()
        path = self.write( 'a.mp4' )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.items(), [] )
        self.assertEqual( self.catalog.list()['pending'], 1 )
        self.catalog.scanner.hash_ready()
        self.assertEqual( self.hashed, [], 'hashed before it settled' )
        self.clock.now += self.SETTLE
        self.catalog.scanner.hash_ready()
        [ item ] = self.items()
        self.assertEqual( item['item'], bu_hash.md5_for_file( path, 16, use_memo=False ) )
        self.assertEqual( ( item['name'], item['status'], item['availability'], item['copies'] ),
                          ( 'a.mp4', 'new', 'present', 1 ) )
        self.assertEqual( self.catalog.list()['pending'], 0 )

    def test_a_file_still_growing_is_not_hashed_until_it_holds_still( self ):
        self.watch()
        path = self.write( 'grow.mkv' )
        self.catalog.scanner.pass_now()
        self.clock.now += self.SETTLE
        with open( path, 'ab' ) as handle:
            handle.write( b'more' )
        self.catalog.scanner.hash_ready()
        self.assertEqual( self.hashed, [] )
        self.clock.now += self.SETTLE
        self.catalog.scanner.hash_ready()
        self.assertEqual( self.hashed, [ 'grow.mkv' ] )

    def test_an_unchanged_file_is_not_hashed_again( self ):
        self.watch()
        self.write( 'a.mp4' )
        self.settle_and_hash()
        self.catalog.scanner.pass_now()
        self.clock.now += self.SETTLE
        self.catalog.scanner.hash_ready()
        self.assertEqual( self.hashed, [ 'a.mp4' ] )

    def test_only_videos_are_indexed( self ):
        self.watch()
        for name in ( 'a.MP4', 'b.txt', '.hidden.mp4', 'c.tmp.mp4', 'd.part0003.mkv', '.dir/e.mp4', 'sub/f.webm' ):
            self.write( name )
        self.write( 'empty.mp4', content=b'' )
        self.settle_and_hash()
        self.assertEqual( sorted( item['name'] for item in self.items() ), [ 'a.MP4', 'f.webm' ] )

    def test_output_folders_inside_a_watched_folder_are_skipped( self ):
        outputs = os.path.normpath( betaconst.video_path_censored )
        parent = os.path.dirname( outputs )
        inside = self.write( 'made.mp4', where=outputs )
        beside = self.write( 'beside.mp4', where=os.path.join( parent, 'other' ) )
        self.addCleanup( os.remove, inside )
        self.addCleanup( shutil.rmtree, os.path.dirname( beside ), ignore_errors=True )
        self.watch( parent )
        self.settle_and_hash()
        self.assertEqual( [ item['name'] for item in self.items() ], [ 'beside.mp4' ] )

    def test_jobs_come_first_hashing_waits_while_busy( self ):
        self.watch()
        self.write( 'a.mp4' )
        self.busy = True
        self.settle_and_hash()
        self.assertEqual( self.hashed, [] )
        self.assertEqual( self.catalog.list()['pending'], 1 )
        self.busy = False
        self.catalog.scanner.hash_ready()
        self.assertEqual( self.hashed, [ 'a.mp4' ] )

    def test_changes_are_reported_with_the_items( self ):
        self.watch()
        path = self.write( 'a.mp4' )
        self.settle_and_hash()
        digest = bu_hash.md5_for_file( path, 16, use_memo=False )
        self.assertIn( ( { digest }, 'scan' ), self.changes )


class TestRefind( _CatalogTest ):
    def setUp( self ):
        super().setUp()
        self.watch()
        self.path = self.write( 'one/clip.mp4', size=8192 )
        self.settle_and_hash()
        [ item ] = self.items()
        self.digest = item['item']
        with self.catalog.store.write() as db:
            db.execute( "UPDATE items SET status = 'done', type_pin = 'scene' WHERE hash = ?", ( self.digest, ) )
        self.hashed.clear()

    def only( self ):
        items = self.items()
        self.assertEqual( len( items ), 1, items )
        return items[0]

    def test_a_rename_is_followed_without_hashing( self ):
        new = os.path.join( self.watched, 'one', 'renamed.mp4' )
        os.rename( self.path, new )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.hashed, [] )
        item = self.only()
        self.assertEqual( ( item['item'], item['name'], item['availability'] ), ( self.digest, 'renamed.mp4', 'present' ) )
        self.assertEqual( [ location['path'] for location in self.item( self.digest )['locations'] ],
                          [ bu_index.norm( new ) ] )

    def test_a_move_to_another_folder_of_the_same_disk_is_followed_without_hashing( self ):
        new = os.path.join( self.watched, 'two', 'deeper', 'clip.mp4' )
        os.makedirs( os.path.dirname( new ) )
        os.rename( self.path, new )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.hashed, [] )
        self.assertEqual( self.only()['item'], self.digest )

    def test_a_renamed_folder_is_followed_without_hashing( self ):
        os.rename( os.path.join( self.watched, 'one' ), os.path.join( self.watched, 'uno' ) )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.hashed, [] )
        self.assertEqual( self.only()['availability'], 'present' )

    def test_a_copy_then_delete_is_refound_by_hash_keeping_status_and_pin( self ):
        new = os.path.join( self.watched, 'elsewhere.mkv' )
        shutil.copyfile( self.path, new )
        os.remove( self.path )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.only()['availability'], 'missing' )
        self.clock.now += self.SETTLE
        self.catalog.scanner.hash_ready()
        self.assertEqual( self.hashed, [ 'elsewhere.mkv' ] )
        item = self.only()
        self.assertEqual( ( item['item'], item['status'], item['pin'], item['availability'], item['copies'] ),
                          ( self.digest, 'done', 'scene', 'present', 1 ) )

    def test_a_file_that_comes_back_is_present_again( self ):
        content = open( self.path, 'rb' ).read()
        os.remove( self.path )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.only()['availability'], 'missing' )
        self.write( 'one/clip.mp4', content=content )
        self.settle_and_hash()
        self.assertEqual( self.only()['availability'], 'present' )

    def test_a_missing_item_keeps_its_place_in_the_library( self ):
        os.remove( self.path )
        self.catalog.scanner.pass_now()
        item = self.only()
        self.assertEqual( ( item['availability'], item['status'], item['pin'] ), ( 'missing', 'done', 'scene' ) )
        self.assertEqual( [ found['item'] for found in self.items( availability=[ 'missing' ] ) ], [ self.digest ] )

    def test_copies_are_one_item_with_its_reclaimable_bytes( self ):
        shutil.copyfile( self.path, os.path.join( self.watched, 'copy.mp4' ) )
        self.settle_and_hash()
        item = self.only()
        self.assertEqual( item['copies'], 2 )
        listed = self.catalog.list( { 'duplicates': True } )
        self.assertEqual( [ found['item'] for found in listed['items'] ], [ self.digest ] )
        self.assertEqual( listed['reclaimable_bytes'], 8192 )

    def test_a_hard_link_is_another_location_without_hashing( self ):
        link = os.path.join( self.watched, 'link.mp4' )
        try:
            os.link( self.path, link )
        except ( OSError, AttributeError ):
            self.skipTest( 'no hard links here' )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.hashed, [] )
        self.assertEqual( self.only()['copies'], 2 )

    def test_new_bytes_at_a_known_path_leave_the_old_item( self ):
        os.remove( self.path )
        self.write( 'one/clip.mp4', size=9000 )
        self.settle_and_hash()
        self.assertEqual( self.hashed, [ 'clip.mp4' ] )
        items = { item['item']: item for item in self.items() }
        self.assertEqual( len( items ), 1, 'the old item has no location left, so it is not listed' )
        self.assertNotIn( self.digest, items )
        self.assertEqual( self.item( self.digest )['locations'], [] )

    def test_files_the_size_of_a_missing_item_are_hashed_first( self ):
        content = open( self.path, 'rb' ).read()
        os.remove( self.path )
        self.write( 'aaa-other.mp4', size=100 )
        self.write( 'zzz-moved.mp4', content=content )
        self.catalog.scanner.pass_now()
        self.clock.now += self.SETTLE
        self.catalog.scanner.hash_ready( limit=1 )
        self.assertEqual( self.hashed, [ 'zzz-moved.mp4' ] )


class TestOffline( _CatalogTest ):
    def test_an_unreachable_folder_is_offline_never_missing( self ):
        self.watch()
        self.write( 'a.mp4' )
        self.settle_and_hash()
        away = self.watched + '-unmounted'
        os.rename( self.watched, away )
        self.catalog.scanner.pass_now()
        [ item ] = self.items()
        self.assertEqual( item['availability'], 'offline' )
        self.assertEqual( item['status'], 'new' )
        self.assertFalse( self.catalog.folders()[0]['available'] )
        os.rename( away, self.watched )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.items()[0]['availability'], 'present' )
        self.assertEqual( len( self.hashed ), 1 )

    def test_an_added_file_whose_folder_is_gone_is_offline( self ):
        elsewhere = os.path.join( self.root, 'drive' )
        path = self.write( 'b.mp4', where=elsewhere )
        self.catalog.add_files( [ path ] )
        self.catalog.scanner.hash_ready()
        shutil.rmtree( elsewhere )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.items()[0]['availability'], 'offline' )

    def test_an_added_file_removed_from_a_folder_that_is_still_there_is_missing( self ):
        elsewhere = os.path.join( self.root, 'drive' )
        path = self.write( 'b.mp4', where=elsewhere )
        self.catalog.add_files( [ path ] )
        self.catalog.scanner.hash_ready()
        os.remove( path )
        self.catalog.scanner.pass_now()
        self.assertEqual( self.items()[0]['availability'], 'missing' )


class TestAddFiles( _CatalogTest ):
    def test_chosen_files_are_hashed_at_once( self ):
        path = self.write( 'chosen.mp4', where=os.path.join( self.root, 'loose' ) )
        accepted, refused = self.catalog.add_files( [ path ] )
        self.assertEqual( ( accepted, refused ), ( [ bu_index.norm( path ) ], [] ) )
        self.catalog.scanner.hash_ready()
        self.assertEqual( [ item['name'] for item in self.items() ], [ 'chosen.mp4' ] )

    def test_chosen_files_go_before_settling_ones( self ):
        self.watch()
        self.write( 'aaa.mp4' )
        self.catalog.scanner.pass_now()
        self.clock.now += self.SETTLE
        chosen = self.write( 'zzz.mp4', where=os.path.join( self.root, 'loose' ) )
        self.catalog.add_files( [ chosen ] )
        self.catalog.scanner.hash_ready( limit=1 )
        self.assertEqual( self.hashed, [ 'zzz.mp4' ] )

    def test_paths_that_are_not_files_are_refused_by_position( self ):
        good = self.write( 'g.mp4', where=os.path.join( self.root, 'loose' ) )
        output = self.write( 'o.mp4', where=betaconst.video_path_censored )
        self.addCleanup( os.remove, output )
        accepted, refused = self.catalog.add_files(
            [ os.path.join( self.root, 'nope.mp4' ), self.root, good, output ] )
        self.assertEqual( accepted, [ bu_index.norm( good ) ] )
        self.assertEqual( refused, [ ( 0, 'not_found' ), ( 1, 'not_a_file' ), ( 3, 'output_folder' ) ] )

    def test_a_chosen_file_inside_a_watched_folder_belongs_to_it( self ):
        folder = self.watch()
        path = self.write( 'in.mp4' )
        self.catalog.add_files( [ path ] )
        self.catalog.scanner.hash_ready()
        [ item ] = self.items()
        self.assertEqual( self.item( item['item'] )['locations'][0]['folder'], folder['folder'] )


class TestFolders( _CatalogTest ):
    def test_a_folder_that_does_not_exist_or_is_a_file_or_an_output_is_refused( self ):
        cases = ( ( os.path.join( self.root, 'nope' ), 'not_found' ),
                  ( self.write( 'f.mp4', where=self.root ), 'not_a_folder' ),
                  ( betaconst.video_path_censored, 'output_folder' ) )
        os.makedirs( betaconst.video_path_censored, exist_ok=True )
        for path, reason in cases:
            with self.assertRaises( bu_store.CatalogError ) as caught:
                self.catalog.add_folder( path )
            self.assertEqual( ( caught.exception.code, caught.exception.params ),
                              ( 'catalog.bad_folder', { 'reason': reason } ) )

    def test_adding_a_folder_twice_keeps_one_and_updates_its_rule( self ):
        first = self.watch()
        again, created = self.catalog.add_folder( self.watched + os.sep, True )
        self.assertFalse( created )
        self.assertEqual( again['folder'], first['folder'] )
        self.assertEqual( [ folder['type_from_folder'] for folder in self.catalog.folders() ], [ True ] )

    def test_removing_a_folder_keeps_the_items_and_adding_it_back_restores_them( self ):
        folder = self.watch()
        self.write( 'a.mp4' )
        self.settle_and_hash()
        [ item ] = self.items()
        with self.catalog.store.write() as db:
            db.execute( "UPDATE items SET type_pin = 'quick_cut' WHERE hash = ?", ( item['item'], ) )
        self.catalog.remove_folder( folder['folder'] )
        self.assertEqual( self.items(), [] )
        self.assertEqual( self.catalog.get( item['item'] )['pin'], 'quick_cut' )
        self.watch()
        self.settle_and_hash()
        [ back ] = self.items()
        self.assertEqual( ( back['item'], back['pin'] ), ( item['item'], 'quick_cut' ) )

    def test_a_location_belongs_to_the_deepest_watched_folder( self ):
        outer = self.watch()
        inner_path = os.path.join( self.watched, 'inner' )
        os.makedirs( inner_path )
        inner = self.watch( inner_path )
        self.write( 'inner/x.mp4' )
        self.settle_and_hash()
        [ item ] = self.items()
        self.assertEqual( self.item( item['item'] )['locations'][0]['folder'], inner['folder'] )
        self.assertEqual( [ found['item'] for found in self.items( folder=inner['folder'] ) ], [ item['item'] ] )
        self.assertEqual( self.items( folder=outer['folder'] ), [] )

    def test_forget_removes_the_item( self ):
        self.watch()
        self.write( 'a.mp4' )
        self.settle_and_hash()
        [ item ] = self.items()
        self.catalog.forget( item['item'] )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            self.catalog.get( item['item'] )
        self.assertEqual( caught.exception.code, 'catalog.no_such_item' )


class TestFolderRule( _CatalogTest ):
    def test_a_watched_1x_tree_picks_the_type_from_its_folder_names( self ):
        self.watch( type_from_folder=True )
        for folder, expected in ( ( 'scene', 'scene' ), ( 'quick-cut', 'quick_cut' ),
                                  ( 'compilation', 'compilation' ), ( 'split-screen', 'split_screen' ) ):
            self.write( '%s/%s.mp4'%( folder, folder ) )
        self.write( 'loose.mp4' )
        self.settle_and_hash()
        shown = { item['name']: item['type'] for item in self.items() }
        for folder, expected in ( ( 'scene', 'scene' ), ( 'quick-cut', 'quick_cut' ),
                                  ( 'compilation', 'compilation' ), ( 'split-screen', 'split_screen' ) ):
            self.assertEqual( shown[folder + '.mp4'], { 'profile': expected, 'rule': 'source_folder',
                                                        'params': { 'folder': folder }, 'when': 'declared' } )
        self.assertEqual( shown['loose.mp4']['when'], 'at_run' )

    def test_without_the_folder_rule_the_folder_name_means_nothing( self ):
        self.watch( type_from_folder=False )
        self.write( 'scene/x.mp4' )
        self.settle_and_hash()
        self.assertEqual( self.items()[0]['type']['when'], 'at_run' )

    def test_type_filter( self ):
        self.watch( type_from_folder=True )
        self.write( 'scene/a.mp4' )
        self.write( 'b.mp4' )
        self.settle_and_hash()
        self.assertEqual( [ item['name'] for item in self.items( type=[ 'scene' ] ) ], [ 'a.mp4' ] )


class TestOutputs( _CatalogTest ):
    def test_an_output_on_disk_marks_its_source_done( self ):
        self.watch()
        path = self.write( 'src.mp4' )
        self.settle_and_hash()
        digest = bu_hash.md5_for_file( path, 16, use_memo=False )
        base = bu_cache.video_output_basename( 'src', digest, [ 640 ], 9, 'd1', 'c2', 'e3' )
        preview = bu_cache.video_output_basename( 'src', digest, [ 640 ], 9, 'd1', 'c2', 'e3', '-preview+20.0s' )
        os.makedirs( betaconst.video_path_censored, exist_ok=True )
        for name in ( base + '.mp4', preview + '.mp4', base + '.video.mkv' ):
            self.write( name, where=betaconst.video_path_censored )
            self.addCleanup( _remove, os.path.join( betaconst.video_path_censored, name ) )
        self.catalog.scanner.pass_now()
        item = self.items()[0]
        self.assertEqual( ( item['status'], item['outputs'] ), ( 'done', 1 ) )
        [ output ] = self.item( digest )['output_files']
        self.assertEqual( ( output['detection_key'], output['censor_key'], output['encode_key'], output['present'] ),
                          ( 'd1', 'c2', 'e3', True ) )
        content = open( os.path.join( betaconst.video_path_censored, base + '.mp4' ), 'rb' ).read()
        os.remove( os.path.join( betaconst.video_path_censored, base + '.mp4' ) )
        self.catalog.scanner.pass_now()
        # Status says what happened; the output's availability says it is gone.
        item = self.items()[0]
        self.assertEqual( ( item['status'], item['outputs'], item['outputs_missing'] ), ( 'done', 0, 1 ) )
        self.assertEqual( [ found['item'] for found in self.items( output=[ 'missing' ] ) ], [ digest ] )
        self.assertEqual( self.items( output=[ 'present' ] ), [] )
        self.write( base + '.mp4', content=content, where=betaconst.video_path_censored )
        self.catalog.scanner.pass_now()
        item = self.items()[0]
        self.assertEqual( ( item['status'], item['outputs'], item['outputs_missing'] ), ( 'done', 1, 0 ) )

    def test_an_item_never_run_has_no_outputs( self ):
        self.watch()
        self.write( 'src.mp4' )
        self.settle_and_hash()
        self.assertEqual( len( self.items( output=[ 'none' ] ) ), 1 )

    def test_an_output_moved_into_a_watched_folder_is_its_output_not_a_new_source( self ):
        self.watch()
        path = self.write( 'src.mp4' )
        self.settle_and_hash()
        digest = bu_hash.md5_for_file( path, 16, use_memo=False )
        base = bu_cache.video_output_basename( 'src', digest, [ 640 ], 9, 'd1', 'c2', 'e3' )
        self.write( 'kept/' + base + '.mp4' )
        self.hashed.clear()
        self.settle_and_hash()
        self.assertEqual( self.hashed, [], 'an output is never hashed as a source' )
        [ item ] = self.items()
        self.assertEqual( ( item['item'], item['outputs'], item['status'] ), ( digest, 1, 'done' ) )
        self.assertEqual( [ output['path'] for output in self.item( digest )['output_files'] ],
                          [ bu_index.norm( os.path.join( self.watched, 'kept', base + '.mp4' ) ) ] )


def _remove( path ):
    try:
        os.remove( path )
    except FileNotFoundError:
        pass


class TestSeeding( _CatalogTest ):
    def test_a_new_catalog_watches_the_folders_it_is_given_that_exist( self ):
        exists = os.path.join( self.root, 'uncensored_vids' )
        os.makedirs( exists )
        catalog = self.open_catalog( seed_folders=[ exists, os.path.join( self.root, 'gone' ) ],
                                     path=os.path.join( self.root, 'seeded.sqlite' ) )
        self.assertEqual( [ ( folder['path'], folder['type_from_folder'] ) for folder in catalog.folders() ],
                          [ ( bu_index.norm( exists ), True ) ] )

    def test_an_existing_catalog_is_never_seeded_again( self ):
        exists = os.path.join( self.root, 'source' )
        os.makedirs( exists )
        path = os.path.join( self.root, 'seeded.sqlite' )
        catalog = self.open_catalog( seed_folders=[ exists ], path=path )
        catalog.remove_folder( catalog.folders()[0]['folder'] )
        catalog.store.close()
        again = self.open_catalog( seed_folders=[ exists ], path=path )
        self.assertEqual( again.folders(), [] )

    def test_the_default_seeds_are_the_video_inputs_and_the_1x_video_folders( self ):
        import betautils_paths as bu_paths
        record = os.path.join( bu_paths.kind_dir( 'data' ), bu_catalog.MIGRATION_RECORD )
        self.assertFalse( os.path.exists( record ) )
        old = os.path.join( self.root, 'old-tree', 'resources' )
        with open( record, 'w', encoding='utf-8' ) as handle:
            json.dump( { 'version': 1, 'media_folders': [ os.path.join( old, name ) for name in (
                'censored_pics', 'source', 'uncensored_pics', 'uncensored_vids' ) ] }, handle )
        self.addCleanup( os.remove, record )
        self.assertEqual( bu_catalog.default_seed_folders(), [
            bu_index.norm( betaconst.video_path_source_backup ), bu_index.norm( betaconst.video_path_uncensored ),
            bu_index.norm( os.path.join( old, 'source' ) ), bu_index.norm( os.path.join( old, 'uncensored_vids' ) ) ] )


class TestStatOnWindows( unittest.TestCase ):
    """Windows file ids are unsigned 64-bit, ReFS ones 128-bit; SQLite stores signed 64."""

    def test_every_64_bit_id_fits_sqlite_one_to_one( self ):
        # NTFS ids are unsigned 64-bit: all of them map one to one.
        values = [ 0, 1, ( 1 << 63 ) - 1, 1 << 63, ( 1 << 63 ) + 1, ( 1 << 64 ) - 2, ( 1 << 64 ) - 1 ]
        folded = [ bu_index.signed64( value ) for value in values ]
        for value in folded:
            self.assertTrue( -( 1 << 63 ) <= value < 1 << 63 )
        self.assertEqual( len( set( folded ) ), len( values ) )
        self.assertEqual( folded[:3], values[:3], 'ids that fit are stored as they are' )

    def test_a_wider_id_still_fits( self ):
        # ReFS ids are 128-bit: folded into range (the size and mtime are
        # compared with it, so a shared fold alone never joins two files).
        for value in ( 1 << 64, ( 1 << 127 ) + 12345, ( 1 << 128 ) - 1 ):
            self.assertTrue( -( 1 << 63 ) <= bu_index.signed64( value ) < 1 << 63 )

    def test_a_location_with_a_large_file_id_is_stored( self ):
        root = tempfile.mkdtemp( prefix='catalog-ids-' )
        self.addCleanup( shutil.rmtree, root, ignore_errors=True )
        store = bu_store.Store( os.path.join( root, 'c.sqlite' ) )
        self.addCleanup( store.close )
        index = bu_index.Index( store )
        path = os.path.join( root, 'x.mp4' )
        open( path, 'wb' ).close()

        class Result:
            st_size, st_mtime_ns, st_dev, st_ino = 5, 7, ( 1 << 64 ) - 3, ( 1 << 64 ) - 1

        file_stat = bu_index.Stat.of( Result )
        index.record_hash( path, file_stat, 'a' * 16 )
        os.remove( path )
        moved = os.path.join( root, 'y.mp4' )
        self.assertEqual( index.observe( moved, file_stat )[0], 'moved' )


class TestSchedule( unittest.TestCase ):
    def scanner( self, interval ):
        class _Index:
            clock = staticmethod( lambda: 0.0 )
        return bu_scan.Scanner( _Index(), hasher=lambda path: 'x', interval=interval )

    def test_the_first_pass_is_due_at_once( self ):
        self.assertEqual( self.scanner( 300 ).next_pass_due( 50.0 ), 50.0 )

    def test_the_next_pass_waits_the_interval_or_ten_times_the_last_pass( self ):
        scanner = self.scanner( 300 )
        scanner._last_pass_end, scanner._last_pass_seconds = 100.0, 1.0
        self.assertEqual( scanner.next_pass_due( 101.0 ), 400.0 )
        scanner._last_pass_seconds = 45.0
        self.assertEqual( scanner.next_pass_due( 101.0 ), 100.0 + 450.0 )

    def test_interval_zero_means_only_when_asked( self ):
        scanner = self.scanner( 0 )
        scanner._last_pass_end = 100.0
        self.assertIsNone( scanner.next_pass_due( 10_000.0 ) )


class TestExternalWrites( _CatalogTest ):
    def test_a_write_by_another_process_is_reported_as_external( self ):
        self.catalog.check_external()
        self.assertEqual( self.changes, [] )
        other = sqlite3.connect( os.path.join( self.root, 'catalog.sqlite' ), isolation_level=None )
        self.addCleanup( other.close )
        other.execute( "INSERT INTO items ( hash, size, name, added_at ) VALUES ( 'x', 1, 'x', 1 )" )
        self.catalog.check_external()
        self.assertEqual( self.changes, [ ( None, 'external' ) ] )

    def test_its_own_writes_are_not_external( self ):
        self.watch()
        self.write( 'a.mp4' )
        self.settle_and_hash()
        self.changes.clear()
        self.catalog.check_external()
        self.assertEqual( self.changes, [] )


if __name__ == '__main__':
    unittest.main()
