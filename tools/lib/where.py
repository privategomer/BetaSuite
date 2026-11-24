#!/usr/bin/env python3
"""
where.py - where BetaSuite keeps its files, in the layout in use.

    python3 tools/lib/where.py                    every kind of folder, one per line
    python3 tools/lib/where.py inputs outputs     where to put videos, where results go
    python3 tools/lib/where.py output/logs resources/model

An argument is either a kind (config, data, cache, state, inputs,
outputs) or a 1.x path under the data folder (output/logs). Prints one
absolute path per argument, in order: under BETASUITE_DATA_ROOT the 1.x
tree's folder, otherwise the OS folder by kind (betautils_paths). With
no argument, prints every kind as "kind<TAB>folder". Exits 1 with the
reason on stderr when a data location variable is unusable, so a script
can stop on it.

Stdlib only, so it runs before the venv exists (setup.sh).
"""

import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

import betautils_paths as bu_paths  # noqa: E402  (stdlib only)


def where( arg ):
    if arg in bu_paths.KINDS:
        return bu_paths.kind_dir( arg )
    return bu_paths.legacy_path( *[ part for part in arg.replace( '\\', '/' ).split( '/' ) if part ] )


def main( argv ):
    try:
        if not argv:
            for kind in bu_paths.KINDS:
                print( '%s\t%s'%( kind, os.path.abspath( bu_paths.kind_dir( kind ) ) ) )
            return 0
        paths = [ where( arg ) for arg in argv ]
    except bu_paths.PathsError as err:
        sys.stderr.write( "%s\n"%( err, ) )
        return 1
    for path in paths:
        print( os.path.abspath( path ) )
    return 0


if __name__ == '__main__':
    raise SystemExit( main( sys.argv[1:] ) )
