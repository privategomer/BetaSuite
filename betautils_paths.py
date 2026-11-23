"""
betautils_paths.py - where resources/ and output/ live.

By default the app finds its data beside the repository: '../resources'
and '../output', relative to the folder it runs in (the repo root). The
BETASUITE_DATA_ROOT environment variable names the folder that holds
resources/ and output/ instead, so a second checkout (a git worktree, a
copy for testing) can work on the same library and caches.

Two kinds of path go through here, handled differently on purpose:

  Layout paths (betaconst's folders, the default log and stats files)
      are never hashed, so they are resolved once, at import.

  Config-valued paths (model_path, sticker 'dir' and 'images',
      log_path, stats_path) keep the string the config wrote. A model
      path is part of the detection key and a sticker folder is part of
      the censor key, so rewriting them would give a different key per
      checkout and miss every cache. Callers resolve them only where
      they touch the filesystem.

Unset (or empty), every function here returns exactly what the code
used before it existed. Set, the value must be an absolute path to an
existing folder: a relative one would depend on the folder each tool
happens to run from, and '~' is not expanded, so both are refused.

Only paths that start with '../resources' or '../output' are moved
under the root. Absolute paths, and any other relative path, are the
user's own choice and pass through untouched.
"""

import os
import re

ENV_VAR = 'BETASUITE_DATA_ROOT'

def _prefix_pattern( separators ):
    """'..' + a separator + 'resources' or 'output', then a separator or the end."""
    sep = '[%s]'%( re.escape( separators ), )
    return re.compile( r'^\.\.%s(?:resources|output)(?:%s|$)'%( sep, sep ) )


# This OS's separators only: on Windows '..\\output\\logs' maps too, while
# on Linux that string is one oddly named file and is left alone.
_DATA_PREFIX = _prefix_pattern( os.sep + ( os.altsep or '' ) )


class DataRootError( ValueError ):
    """BETASUITE_DATA_ROOT is set to something that cannot be used."""


def data_root():
    """
    The data root from BETASUITE_DATA_ROOT, or None when it is unset.

    Read on every call, so a test (or a launcher) can change it without
    re-importing this module. Empty counts as unset.

    Returns:
        The normalised absolute folder, or None.

    Raises:
        DataRootError: the value is relative, or is not an existing folder.
    """
    value = os.environ.get( ENV_VAR, '' )
    if not value:
        return None
    if not os.path.isabs( value ):
        raise DataRootError( "%s must be an absolute path, got %r ('~' is not expanded)"%(
            ENV_VAR, value ) )
    if not os.path.isdir( value ):
        raise DataRootError( "%s is %r, which is not an existing folder"%( ENV_VAR, value ) )
    return os.path.normpath( value )


def resolve( path ):
    """
    Where a '../resources/...' or '../output/...' path really is.

    Args:
        path: Any path string, as betaconst or a config wrote it.

    Returns:
        path itself when BETASUITE_DATA_ROOT is unset or path does not
        start with '../resources' or '../output'; otherwise the same path
        under the data root, keeping any trailing separator.
    """
    if not isinstance( path, str ) or not _DATA_PREFIX.match( path ):
        return path
    root = data_root()
    if root is None:
        return path
    return os.path.join( root, path[3:] )


def beside_repo( repo_root, *parts ):
    """
    A path in the folder that holds resources/ and output/, for tools
    that build it from their own location rather than the working folder.

    Args:
        repo_root: The repository root the caller computed.
        *parts: Path parts under the data folder, e.g. 'output', 'logs'.

    Returns:
        os.path.join( repo_root, '..', *parts ) when the variable is unset
        (exactly what those tools built before), otherwise the same parts
        under the data root.
    """
    root = data_root()
    if root is None:
        return os.path.join( repo_root, '..', *parts )
    return os.path.join( root, *parts )
