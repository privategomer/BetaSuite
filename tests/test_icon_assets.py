"""
test_icon_assets.py - the committed app icon set matches its generator.

assets/icons/<icon>/ holds SVG sources and icon.json written by
tools/art/icons.py, and PNGs and icon.ico drawn from them by
tools/art/rasterize_icons.py, which also copies the default icon into
ui/src-tauri/icons. These tests need no rasterizer:

- rerunning the generator writes byte-identical sources
- every icon has a PNG of the right pixel size at every size
- icon.ico holds every size up to 256, each entry the PNG of that size
- manifest.json still matches every source, PNG and icon file
- the shell's bundle icons are the default icon's files

After changing the generator, run tools/art/make_icons.sh and commit the
result.
"""

import hashlib
import json
import os
import struct
import sys
import unittest

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
ICONS = os.path.join( REPO, 'assets', 'icons' )
TAURI_ICONS = os.path.join( REPO, 'ui', 'src-tauri', 'icons' )
sys.path.insert( 0, os.path.join( REPO, 'tools', 'art' ) )

import icons  # noqa: E402
import rasterize_icons  # noqa: E402

EXPECTED_IDS = [ 'censored_play', 'bars_shield', 'focus_blur', 'pixel_play', 'pixel_b' ]
EXPECTED_SIZES = [ 16, 22, 24, 32, 48, 64, 128, 256, 512 ]


def _sha( data ):
    return hashlib.sha256( data ).hexdigest()


def _read( *parts ):
    with open( os.path.join( *parts ), 'rb' ) as handle:
        return handle.read()


def _png_size( data ):
    if data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        raise ValueError( 'not a PNG' )
    return struct.unpack( '>II', data[16:24] )


def _meta( key ):
    return json.loads( _read( ICONS, key, 'icon.json' ).decode( 'utf-8' ) )


def _manifest():
    return json.loads( _read( ICONS, 'manifest.json' ).decode( 'utf-8' ) )


class GeneratorTest( unittest.TestCase ):

    def test_five_icons_ship_default_first( self ):
        self.assertEqual( list( icons.ICONS ), EXPECTED_IDS )
        self.assertEqual( rasterize_icons.icons_in( ICONS ), EXPECTED_IDS )

    def test_rerun_is_byte_identical( self ):
        for key in icons.ICONS:
            for rel, text in icons.files( key ):
                with self.subTest( icon=key, file=rel ):
                    self.assertEqual( _read( ICONS, key, rel ).decode( 'utf-8' ), text )

    def test_sources_carry_no_metadata( self ):
        for key in icons.ICONS:
            for rel, text in icons.files( key ):
                with self.subTest( icon=key, file=rel ):
                    self.assertNotIn( 'metadata', text )

    def test_catalog_key_and_order( self ):
        for order, key in enumerate( EXPECTED_IDS ):
            meta = _meta( key )
            self.assertEqual( meta[ 'id' ], key )
            self.assertEqual( meta[ 'name_key' ], 'app_icon.' + key )
            self.assertEqual( meta[ 'order' ], order )


class RasterTest( unittest.TestCase ):

    def test_every_icon_has_every_size( self ):
        for key in EXPECTED_IDS:
            meta = _meta( key )
            self.assertEqual( sorted( int( n ) for n in meta[ 'sizes' ] ), EXPECTED_SIZES )
            for size in EXPECTED_SIZES:
                with self.subTest( icon=key, size=size ):
                    png = _read( ICONS, key, 'icon-%d.png' % size )
                    self.assertEqual( _png_size( png ), ( size, size ) )

    def test_ico_holds_each_size_up_to_256( self ):
        for key in EXPECTED_IDS:
            data = _read( ICONS, key, 'icon.ico' )
            with self.subTest( icon=key ):
                self.assertEqual( rasterize_icons.ico_sizes( data ), [ n for n in EXPECTED_SIZES if n <= 256 ] )
                count = struct.unpack_from( '<H', data, 4 )[0]
                for i in range( count ):
                    dim, _, _, _, _, _, length, offset = struct.unpack_from( '<BBBBHHII', data, 6 + 16 * i )
                    size = dim or 256
                    self.assertEqual( data[ offset:offset + length ], _read( ICONS, key, 'icon-%d.png' % size ) )

    def test_manifest_matches_sources_and_pngs( self ):
        manifest = _manifest()
        seen = set()
        for key in EXPECTED_IDS:
            meta = _meta( key )
            for size, entry in meta[ 'sizes' ].items():
                rel = '%s/%s' % ( key, entry[ 'png' ] )
                seen.add( rel )
                with self.subTest( png=rel ):
                    record = manifest[ 'png' ][ rel ]
                    self.assertEqual( record[ 'svg_sha256' ], _sha( _read( ICONS, key, entry[ 'source' ] ) ) )
                    self.assertEqual( record[ 'png_sha256' ], _sha( _read( ICONS, key, entry[ 'png' ] ) ) )
            ico = '%s/%s' % ( key, meta[ 'ico' ] )
            self.assertEqual( manifest[ 'ico' ][ ico ], _sha( _read( ICONS, key, meta[ 'ico' ] ) ) )
        self.assertEqual( set( manifest[ 'png' ] ), seen )

    def test_shell_bundle_icons_are_the_default( self ):
        default = icons.ICONS[0]
        for name, size in rasterize_icons.TAURI_PNGS.items():
            with self.subTest( file=name ):
                self.assertEqual( _read( TAURI_ICONS, name ), _read( ICONS, default, 'icon-%d.png' % size ) )
        self.assertEqual( _read( TAURI_ICONS, 'icon.ico' ), _read( ICONS, default, 'icon.ico' ) )
        self.assertEqual( _read( TAURI_ICONS, 'source.svg' ), _read( ICONS, default, 'icon.svg' ) )
        pngs = { size: _read( ICONS, default, 'icon-%d.png' % size ) for size, _ in rasterize_icons.ICNS_TYPES }
        self.assertEqual( _read( TAURI_ICONS, 'icon.icns' ), rasterize_icons.icns_bytes( pngs ) )


if __name__ == '__main__':
    unittest.main()
