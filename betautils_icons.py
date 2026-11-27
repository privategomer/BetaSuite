"""
betautils_icons.py - the bundled app icons, as data.

assets/icons/<icon>/icon.json names one icon, its catalog key, its place in
the picker and the PNG for each size (tools/art/icons.py writes them). An
icon is a folder: adding one needs no code, and the app_icon setting's
choices are whatever folders are present.

Stdlib only, so the settings registry can list the icons cheaply.
"""

import json
import os

ICON_ROOT = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'assets', 'icons' )


class UnknownIcon( KeyError ):
    """No such icon, or no PNG of that size."""


def _meta( root, name ):
    with open( os.path.join( root, name, 'icon.json' ), 'r', encoding='utf-8' ) as handle:
        return json.load( handle )


def app_icons( root=None ):
    """The icons present, by folder name, in picker order (then by name)."""
    root = root or ICON_ROOT
    if not os.path.isdir( root ):
        return []
    names = [ name for name in os.listdir( root )
              if os.path.isfile( os.path.join( root, name, 'icon.json' ) ) ]

    def order( name ):
        try:
            place = int( _meta( root, name ).get( 'order', len( names ) ) )
        except ( ValueError, TypeError, OSError, json.JSONDecodeError ):
            place = len( names )
        return ( place, name )
    return sorted( names, key=order )


def icon_png( icon, size, root=None ):
    """The path of one icon's PNG at one pixel size."""
    root = root or ICON_ROOT
    if icon not in app_icons( root ):
        raise UnknownIcon( icon )
    entry = _meta( root, icon ).get( 'sizes', {} ).get( str( size ) )
    if entry is None:
        raise UnknownIcon( '%s at %s px'%( icon, size ) )
    return os.path.join( root, icon, entry[ 'png' ] )
