"""
betautils_art.py - the bundled figure art, as data.

assets/art/<style>/style.json lists each figure and tile of one art style
with its PNG and the boxes authored on it (tools/art/make_art.py writes
them). A style is a folder: adding one needs no code, and the art_style
setting's choices are whatever folders are present.

Stdlib only, so the settings registry can list the styles cheaply.
"""

import json
import os

ART_ROOT = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'assets', 'art' )

_STYLE_CACHE = {}


class UnknownArt( KeyError ):
    """No such art style or item."""


def art_styles( root=None ):
    """The art styles present, by folder name, sorted."""
    root = root or ART_ROOT
    if not os.path.isdir( root ):
        return []
    return sorted( name for name in os.listdir( root )
                   if os.path.isfile( os.path.join( root, name, 'style.json' ) ) )


def style_meta( style, root=None ):
    """One style's style.json, read again only when the file changes."""
    root = root or ART_ROOT
    if style not in art_styles( root ):
        raise UnknownArt( style )
    path = os.path.join( root, style, 'style.json' )
    stamp = os.stat( path ).st_mtime_ns
    cached = _STYLE_CACHE.get( path )
    if cached is None or cached[0] != stamp:
        with open( path, 'r', encoding='utf-8' ) as handle:
            cached = ( stamp, json.load( handle ) )
        _STYLE_CACHE[path] = cached
    return cached[1]


def art_item( style, item, root=None ):
    """
    One figure or tile of a style.

    Args:
        item: 'figure.<sex>.<view>.<dress>' or 'tile.<label>.<sex>', as
            style.json names them.

    Returns:
        { 'path': PNG path, 'boxes': [ {label, x, y, w, h} ], 'canvas': [w, h] }
    """
    root = root or ART_ROOT
    meta = style_meta( style, root )
    kind, _, name = item.partition( '.' )
    table = { 'figure': meta.get( 'figures', {} ), 'tile': meta.get( 'tiles', {} ) }.get( kind, {} )
    if name not in table:
        raise UnknownArt( item )
    entry = table[name]
    return { 'path': os.path.join( root, style, entry['file'] ),
             'boxes': [ dict( box ) for box in entry['boxes'] ],
             'canvas': list( entry['canvas'] ) }
