"""
test_art_assets.py - the committed figure art matches its generator.

assets/art/<style>/ holds SVG figures and a style.json written by
tools/art/make_art.py, and PNGs drawn from them by tools/art/rasterize.py.
The engine reads only the PNGs and style.json. These tests keep the three
in step without needing the rasterizer:

- rerunning the generator writes byte-identical SVGs and style.json
- manifest.json records, for every PNG, the sha256 of the SVG it was drawn
  from (a figure, or a figure cropped to a tile's window) and of the PNG
  itself, and both still match
- every box sits inside its canvas, and every label the engine knows has
  art in every style

After changing the generator, run tools/art/make_art.sh and commit the
result.
"""

import hashlib
import json
import os
import re
import sys
import tempfile
import unittest

import betaconst

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
ART = os.path.join( REPO, 'assets', 'art' )
sys.path.insert( 0, os.path.join( REPO, 'tools', 'art' ) )

import make_art  # noqa: E402
import rasterize  # noqa: E402


def _sha( data ):
    return hashlib.sha256( data ).hexdigest()


def _styles():
    return rasterize.styles_in( ART )


def _meta( key ):
    with open( os.path.join( ART, key, 'style.json' ), encoding = 'utf-8' ) as f:
        return json.load( f )


class GeneratorTest( unittest.TestCase ):

    def test_four_styles_ship( self ):
        self.assertEqual( _styles(), [ 'angular', 'cel', 'flat', 'ink' ] )

    def test_rerun_is_byte_identical( self ):
        with tempfile.TemporaryDirectory() as out:
            with open( os.devnull, 'w' ) as quiet:
                saved, sys.stdout = sys.stdout, quiet
                try:
                    make_art.main( [ '--out', out ] )
                finally:
                    sys.stdout = saved
            for key in _styles():
                fresh = os.path.join( out, key )
                for dirpath, _dirs, files in os.walk( fresh ):
                    for name in files:
                        rel = os.path.relpath( os.path.join( dirpath, name ), fresh )
                        with open( os.path.join( fresh, rel ), 'rb' ) as f:
                            want = f.read()
                        committed = os.path.join( ART, key, rel )
                        self.assertTrue( os.path.exists( committed ), '%s/%s is not committed' % ( key, rel ) )
                        with open( committed, 'rb' ) as f:
                            self.assertEqual( f.read(), want, '%s/%s differs from the generator; run tools/art/make_art.sh' % ( key, rel ) )

    def test_svgs_carry_no_text( self ):
        for key in _styles():
            for fig in _meta( key )[ 'figures' ].values():
                with open( os.path.join( ART, key, fig[ 'source' ] ), encoding = 'utf-8' ) as f:
                    self.assertNotIn( '<text', f.read() )


class ManifestTest( unittest.TestCase ):

    def setUp( self ):
        with open( os.path.join( ART, 'manifest.json' ), encoding = 'utf-8' ) as f:
            self.manifest = json.load( f )

    def test_every_png_matches_its_svg( self ):
        listed = set()
        for key in _styles():
            for rel, svg in rasterize.sources( ART, key ):
                path = '%s/%s' % ( key, rel )
                listed.add( path )
                entry = self.manifest[ 'png' ].get( path )
                self.assertIsNotNone( entry, '%s is not in manifest.json' % path )
                self.assertEqual( entry[ 'svg_sha256' ], _sha( svg.encode( 'utf-8' ) ),
                                  '%s was drawn from an older SVG; run tools/art/make_art.sh' % path )
                with open( os.path.join( ART, key, rel ), 'rb' ) as f:
                    self.assertEqual( entry[ 'png_sha256' ], _sha( f.read() ), '%s changed after it was drawn' % path )
        self.assertEqual( set( self.manifest[ 'png' ] ), listed, 'manifest.json lists PNGs no style has' )

    def test_tile_svg_crops_the_figure( self ):
        fig = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 1200" width="600" height="1200"><g/></svg>\n'
        out = rasterize.tile_svg( fig, [ 10, 20, 300 ], 384 )
        self.assertTrue( out.startswith( '<svg xmlns="http://www.w3.org/2000/svg" viewBox="10 20 300 300" width="384" height="384">' ) )
        self.assertTrue( out.endswith( '<g/></svg>\n' ) )
        with self.assertRaises( ValueError ):
            rasterize.tile_svg( '<svg>', [ 0, 0, 1 ], 1 )


class BoxesTest( unittest.TestCase ):

    def _items( self, meta ):
        for fid, fig in meta[ 'figures' ].items():
            yield 'figure ' + fid, fig
        for tid, tile in meta[ 'tiles' ].items():
            yield 'tile ' + tid, tile

    def test_boxes_inside_canvas( self ):
        for key in _styles():
            for name, item in self._items( _meta( key ) ):
                cw, ch = item[ 'canvas' ]
                self.assertTrue( item[ 'boxes' ], '%s %s has no box' % ( key, name ) )
                for b in item[ 'boxes' ]:
                    where = '%s %s %s' % ( key, name, b )
                    self.assertIn( b[ 'label' ], betaconst.classes, where )
                    self.assertGreater( b[ 'w' ], 0, where )
                    self.assertGreater( b[ 'h' ], 0, where )
                    self.assertGreaterEqual( b[ 'x' ], 0, where )
                    self.assertGreaterEqual( b[ 'y' ], 0, where )
                    self.assertLessEqual( b[ 'x' ] + b[ 'w' ], cw, where )
                    self.assertLessEqual( b[ 'y' ] + b[ 'h' ], ch, where )

    def test_tiles_show_one_label_from_a_figure( self ):
        for key in _styles():
            meta = _meta( key )
            for tid, tile in meta[ 'tiles' ].items():
                self.assertEqual( tid, '%s.%s' % ( tile[ 'label' ], tile[ 'sex' ] ) )
                self.assertIn( tile[ 'figure' ], meta[ 'figures' ] )
                self.assertTrue( tile[ 'figure' ].startswith( tile[ 'sex' ] + '.' ) )
                self.assertEqual( { b[ 'label' ] for b in tile[ 'boxes' ] }, { tile[ 'label' ] } )
                x, y, s = tile[ 'window' ]
                self.assertTrue( 0 <= x and 0 <= y and x + s <= 600 and y + s <= 1200, tid )

    def test_every_label_has_art_in_every_style( self ):
        for key in _styles():
            labels = { t[ 'label' ] for t in _meta( key )[ 'tiles' ].values() }
            self.assertEqual( labels, set( betaconst.classes ), key )

    def test_styles_have_the_same_items( self ):
        metas = [ _meta( key ) for key in _styles() ]
        for meta in metas[ 1: ]:
            self.assertEqual( set( meta[ 'figures' ] ), set( metas[ 0 ][ 'figures' ] ) )
            self.assertEqual( set( meta[ 'tiles' ] ), set( metas[ 0 ][ 'tiles' ] ) )

    def test_style_ids_are_names( self ):
        for key in _styles():
            meta = _meta( key )
            self.assertEqual( meta[ 'id' ], key )
            self.assertTrue( re.fullmatch( r'[a-z][a-z0-9_]*', key ) )
            self.assertEqual( meta[ 'name_key' ], 'art_style.' + key )


if __name__ == '__main__':
    unittest.main()
