"""
test_trash.py - moving cleared files to the freedesktop trash
(betautils_trash), on scratch folders only.

What each test prevents:
  - a file "trashed" without the info file the file manager needs to
    put it back, or with its path written wrong
  - a second item of the same name overwriting the first in the trash
  - a file on another disk copied across into the home trash, or
    deleted when the trash cannot take it
  - an info file left behind when the move fails
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.parse
from unittest import mock

import betautils_trash as bu_trash

NOW = lambda: time.struct_time( ( 2000, 1, 2, 3, 4, 5, 0, 2, -1 ) )


@unittest.skipIf( sys.platform in ( 'win32', 'darwin' ), 'the freedesktop trash' )
class TestFreedesktop( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='trash-' )
        self.addCleanup( shutil.rmtree, self.root, True )
        self.env = { 'XDG_DATA_HOME': os.path.join( self.root, 'data' ), 'HOME': self.root }
        self.trash = os.path.join( self.root, 'data', 'Trash' )

    def make( self, *parts, text='x' ):
        path = os.path.join( self.root, *parts )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'w', encoding='utf-8' ) as handle:
            handle.write( text )
        return path

    def test_a_file_goes_with_its_info( self ):
        path = self.make( 'cache', 'a b.json', text='hello' )
        bu_trash.trash( path, environ=self.env, now=NOW )
        self.assertFalse( os.path.exists( path ) )
        with open( os.path.join( self.trash, 'files', 'a b.json' ), encoding='utf-8' ) as handle:
            self.assertEqual( handle.read(), 'hello' )
        with open( os.path.join( self.trash, 'info', 'a b.json.trashinfo' ), encoding='utf-8' ) as handle:
            info = handle.read()
        self.assertEqual( info, '[Trash Info]\nPath=%s\nDeletionDate=2000-01-02T03:04:05\n'%( urllib.parse.quote( path ), ) )

    def test_a_folder_goes_whole( self ):
        self.make( 'cache', 'vid_hashes', 'one.json' )
        self.make( 'cache', 'vid_hashes', 'sub', 'two.json' )
        bu_trash.trash( os.path.join( self.root, 'cache', 'vid_hashes' ), environ=self.env, now=NOW )
        self.assertTrue( os.path.exists( os.path.join( self.trash, 'files', 'vid_hashes', 'sub', 'two.json' ) ) )

    def test_the_same_name_twice_keeps_both( self ):
        first = self.make( 'a', 'thumb.png', text='1' )
        second = self.make( 'b', 'thumb.png', text='2' )
        bu_trash.trash( first, environ=self.env, now=NOW )
        bu_trash.trash( second, environ=self.env, now=NOW )
        names = sorted( os.listdir( os.path.join( self.trash, 'files' ) ) )
        self.assertEqual( names, [ 'thumb.png', 'thumb.png.1' ] )
        self.assertTrue( os.path.exists( os.path.join( self.trash, 'info', 'thumb.png.1.trashinfo' ) ) )

    def test_an_orphan_in_the_trash_is_never_overwritten( self ):
        orphan = os.path.join( self.trash, 'files', 'thumb.png' )      # no info file beside it
        os.makedirs( os.path.dirname( orphan ) )
        with open( orphan, 'w', encoding='utf-8' ) as handle:
            handle.write( 'old' )
        bu_trash.trash( self.make( 'a', 'thumb.png', text='new' ), environ=self.env, now=NOW )
        with open( orphan, encoding='utf-8' ) as handle:
            self.assertEqual( handle.read(), 'old' )
        self.assertTrue( os.path.exists( os.path.join( self.trash, 'files', 'thumb.png.1' ) ) )

    def test_missing_path_is_nothing_to_do( self ):
        bu_trash.trash( os.path.join( self.root, 'nowhere' ), environ=self.env, now=NOW )
        self.assertFalse( os.path.exists( self.trash ) )

    def test_another_disk_uses_its_own_trash_with_a_relative_path( self ):
        path = self.make( 'disk', 'cache', 'x.json' )
        disk = os.path.join( self.root, 'disk' )
        home_device = os.lstat( self.root ).st_dev
        real_device = bu_trash._device

        def device( probe ):
            # The home trash is on the home disk; everything under disk/ is another one.
            return home_device + 1 if probe.startswith( disk ) else real_device( probe )
        with mock.patch.object( bu_trash, '_device', side_effect=device ), \
             mock.patch.object( bu_trash, 'mount_point', return_value=disk ):
            bu_trash.trash( path, environ=self.env, now=NOW )
        own = os.path.join( disk, '.Trash-%d'%( os.getuid(), ) )
        self.assertTrue( os.path.exists( os.path.join( own, 'files', 'x.json' ) ) )
        with open( os.path.join( own, 'info', 'x.json.trashinfo' ), encoding='utf-8' ) as handle:
            self.assertIn( 'Path=cache/x.json\n', handle.read() )
        self.assertFalse( os.path.exists( self.trash ), 'nothing copied into the home trash' )

    def test_no_copy_across_disks_and_nothing_deleted( self ):
        path = self.make( 'disk', 'x.json' )
        with mock.patch.object( bu_trash, '_device', side_effect=lambda probe: 1 if 'disk' in probe else 2 ), \
             mock.patch.object( bu_trash, '_topdir_trash', return_value=( self.trash, self.root ) ):
            with self.assertRaises( bu_trash.TrashError ) as caught:
                bu_trash.trash( path, environ=self.env, now=NOW )
        self.assertEqual( caught.exception.reason, 'other_disk' )
        self.assertTrue( os.path.exists( path ) )

    def test_a_failed_move_leaves_no_info_behind( self ):
        path = self.make( 'cache', 'y.json' )
        with mock.patch( 'os.rename', side_effect=OSError( 'busy' ) ):
            with self.assertRaises( bu_trash.TrashError ):
                bu_trash.trash( path, environ=self.env, now=NOW )
        self.assertEqual( os.listdir( os.path.join( self.trash, 'info' ) ), [] )
        self.assertTrue( os.path.exists( path ) )


if __name__ == '__main__':
    unittest.main()
