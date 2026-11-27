"""
test_icons_setting.py - the app_icon and tray_icon_shown settings, and the
icon folders their choices come from (betautils_icons.py).
"""

import json
import os
import tempfile
import unittest

import betaconfig
import betautils_icons as bu_icons
import betautils_settings as bu_settings


def _folder( root, name, order=None, sizes=( 16, ) ):
    os.makedirs( os.path.join( root, name ) )
    meta = { 'id': name, 'sizes': { str( n ): { 'png': 'icon-%d.png'%( n, ) } for n in sizes } }
    if order is not None:
        meta[ 'order' ] = order
    with open( os.path.join( root, name, 'icon.json' ), 'w', encoding='utf-8' ) as handle:
        json.dump( meta, handle )


class IconFoldersTest( unittest.TestCase ):

    def test_shipped_icons_in_picker_order( self ):
        self.assertEqual( bu_icons.app_icons(),
                          [ 'censored_play', 'bars_shield', 'focus_blur', 'pixel_play', 'pixel_b' ] )

    def test_order_then_name_and_unordered_last( self ):
        with tempfile.TemporaryDirectory() as root:
            _folder( root, 'zeta', order=0 )
            _folder( root, 'alpha', order=1 )
            _folder( root, 'beta', order=1 )
            _folder( root, 'loose' )
            os.makedirs( os.path.join( root, 'no_meta' ) )
            self.assertEqual( bu_icons.app_icons( root ), [ 'zeta', 'alpha', 'beta', 'loose' ] )

    def test_bad_order_sorts_last( self ):
        with tempfile.TemporaryDirectory() as root:
            _folder( root, 'b', order=0 )
            _folder( root, 'a', order='first' )
            self.assertEqual( bu_icons.app_icons( root ), [ 'b', 'a' ] )

    def test_missing_root_has_none( self ):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual( bu_icons.app_icons( os.path.join( root, 'gone' ) ), [] )

    def test_icon_png( self ):
        path = bu_icons.icon_png( 'censored_play', 16 )
        self.assertEqual( path, os.path.join( bu_icons.ICON_ROOT, 'censored_play', 'icon-16.png' ) )
        self.assertTrue( os.path.isfile( path ) )

    def test_icon_png_refuses_unknown_icon_and_size( self ):
        with self.assertRaises( bu_icons.UnknownIcon ):
            bu_icons.icon_png( 'no_such_icon', 16 )
        with self.assertRaises( bu_icons.UnknownIcon ):
            bu_icons.icon_png( 'censored_play', 17 )
        with self.assertRaises( bu_icons.UnknownIcon ):
            bu_icons.icon_png( '../art', 16 )


class SettingsTest( unittest.TestCase ):

    def test_app_icon_choices_are_the_folders( self ):
        entry = bu_settings.find( ( 'app_icon', ) )
        self.assertEqual( entry.kind, 'choice' )
        self.assertEqual( list( entry.choices() ), bu_icons.app_icons() )
        self.assertEqual( betaconfig.app_icon, 'censored_play' )
        self.assertEqual( bu_icons.app_icons()[0], betaconfig.app_icon )

    def test_in_no_cache_key( self ):
        for path in ( 'app_icon', 'tray_icon_shown' ):
            entry = bu_settings.find( ( path, ) )
            with self.subTest( path=path ):
                self.assertEqual( entry.moves, frozenset() )
                self.assertEqual( entry.may_move, frozenset() )
                self.assertEqual( entry.level, bu_settings.ADVANCED )
                self.assertEqual( entry.help, path )
                self.assertEqual( entry.tip, 'setting.%s.tip'%( path, ) )

    def test_app_only_settings_move_no_key( self ):
        app_only = sorted( entry.dotted for entry in bu_settings.REGISTRY if entry.app_only )
        self.assertEqual( app_only, [ 'app_icon', 'art_style', 'tray_icon_shown' ] )
        for entry in bu_settings.REGISTRY:
            if entry.app_only:
                with self.subTest( path=entry.dotted ):
                    self.assertEqual( entry.moves, frozenset() )
                    self.assertEqual( entry.may_move, frozenset() )
        sent = { item[ 'path' ]: item for item in bu_settings.schema()[ 'settings' ] }
        self.assertIs( sent[ 'app_icon' ][ 'app_only' ], True )
        self.assertNotIn( 'app_only', sent[ 'debug_mode' ] )

    def test_tray_shown_by_default( self ):
        self.assertIs( betaconfig.tray_icon_shown, True )
        self.assertEqual( bu_settings.find( ( 'tray_icon_shown', ) ).kind, 'bool' )


if __name__ == '__main__':
    unittest.main()
