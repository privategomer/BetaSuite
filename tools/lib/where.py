#!/usr/bin/env python3
"""
where.py - where 1.x data folders are, in the layout in use, for shell tools.

    python3 tools/lib/where.py output/logs resources/model

Prints one absolute path per argument, in order: the 1.x path under
BETASUITE_DATA_ROOT when that is set, else its OS folder by kind
(betautils_paths.legacy_path). Exits 1 with the reason on stderr when a
data location variable is unusable, so a script can stop on it.

Stdlib only, so it runs before the venv exists (setup.sh).
"""

import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

import betautils_paths as bu_paths  # noqa: E402  (stdlib only)


def main( argv ):
    if not argv:
        sys.stderr.write( "usage: where.py <1.x path under the data folder> ...\n" )
        return 2
    try:
        paths = [ bu_paths.legacy_path( *[ part for part in arg.replace( '\\', '/' ).split( '/' ) if part ] )
                  for arg in argv ]
    except bu_paths.PathsError as err:
        sys.stderr.write( "%s\n"%( err, ) )
        return 1
    for path in paths:
        print( os.path.abspath( path ) )
    return 0


if __name__ == '__main__':
    raise SystemExit( main( sys.argv[1:] ) )
