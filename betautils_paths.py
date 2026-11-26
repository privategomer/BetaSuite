"""
betautils_paths.py - where each kind of data lives.

BetaSuite keeps six kinds of data, split the way the operating system
expects, per user and without admin rights:

  config    settings (betaconfig_local.py)
  data      durable files: models, stickers, run-key manifests, the
            library catalog (catalog.sqlite)
  cache     files that can be rebuilt: detections, shot cuts, transcodes,
            the file-hash memo, blurred library thumbnails. The large
            one, so it can be moved
  state     logs, stats, benchmarks, the run lock
  inputs    the folders runs read from (uncensored_vids, uncensored_pics,
  outputs   source) and write to (censored_vids, censored_pics)

Default folders:

  kind      Linux and other XDG systems        Windows                            macOS
  config    ~/.config/betasuite                %LOCALAPPDATA%\\BetaSuite\\config    ~/Library/Application Support/BetaSuite/config
  data      ~/.local/share/betasuite/data      %LOCALAPPDATA%\\BetaSuite\\data      ~/Library/Application Support/BetaSuite/data
  cache     ~/.cache/betasuite                 %LOCALAPPDATA%\\BetaSuite\\cache     ~/Library/Caches/BetaSuite
  state     ~/.local/state/betasuite           %LOCALAPPDATA%\\BetaSuite\\state     ~/Library/Application Support/BetaSuite/state
  inputs    ~/.local/share/betasuite/library   %LOCALAPPDATA%\\BetaSuite\\library   ~/Library/Application Support/BetaSuite/library.noindex
  outputs   (the same folder as inputs)

WHY THE VIDEOS SIT IN APP DATA. They are mostly explicit, and the
usual video folders are indexed and shown off by the desktop: Baloo and
Tracker search ~/Videos, media apps list it, Windows Search indexes the
profile's libraries. App data is not: Baloo skips hidden folders, Tracker
indexes only the XDG user folders, Windows Search skips AppData, and
Spotlight skips a folder named *.noindex. The folders stay their own
kinds, never inside data or cache, so clearing app data or the cache
can never take them: no kind's folder contains another's
(test_paths.py checks it). BETASUITE_INPUT_DIR and BETASUITE_OUTPUT_DIR
put them anywhere else; `tools/lib/where.py inputs outputs` prints them.

On XDG systems XDG_CONFIG_HOME, XDG_DATA_HOME, XDG_CACHE_HOME and
XDG_STATE_HOME are honored when absolute (the XDG spec says to ignore a
relative one). BETASUITE_CACHE_DIR, BETASUITE_INPUT_DIR and
BETASUITE_OUTPUT_DIR move one kind each; they must be absolute.

THE 1.x SHAPE. BETASUITE_DATA_ROOT names a folder that holds resources/
and output/ the way 1.x laid them out, and uses it whole: every path is
exactly what 1.x used, under that folder. It is for checkouts that work
on a 1.x tree (a git worktree, a test copy) and for going back to 1.x.
It cannot be combined with the three folder overrides, which only move
the OS folders.

1.x PATHS STAY THE NAMES. Code and config address everything 1.x had
by its 1.x path ('../resources/model/x.onnx', '../output/cache'), and
resolve() says where that is. Config-valued paths (model_path, sticker
'dir' and 'images', log_path, stats_path) are never rewritten: a model
path is part of the detection key and a sticker folder part of the
censor key, so rewriting them would give a different key per layout and
miss every cache. Callers resolve them only where they touch the
filesystem. betaconst's layout paths are never hashed and are resolved
once, at import. LEGACY_LAYOUT is the one table from 1.x paths to
kinds; the 1.x to 2.x migration reads the same table, so the two cannot
disagree. Files 2.x adds use kind_dir().

Only paths that start with '../resources' or '../output' are mapped.
Absolute paths, and any other relative path, are the user's own choice
and pass through untouched.
"""

import ntpath
import os
import posixpath
import re
import sys

ENV_VAR = 'BETASUITE_DATA_ROOT'
CACHE_ENV_VAR = 'BETASUITE_CACHE_DIR'
INPUT_ENV_VAR = 'BETASUITE_INPUT_DIR'
OUTPUT_ENV_VAR = 'BETASUITE_OUTPUT_DIR'

KINDS = ( 'config', 'data', 'cache', 'state', 'inputs', 'outputs' )

# What "remove app data" covers. Inputs and outputs are the user's files.
APP_DATA_KINDS = ( 'config', 'data', 'cache', 'state' )

# kind -> the variable that moves it.
OVERRIDES = { 'cache': CACHE_ENV_VAR, 'inputs': INPUT_ENV_VAR, 'outputs': OUTPUT_ENV_VAR }

# A 1.x path under the data folder -> ( kind, path under that kind's folder ).
# The longest matching prefix wins. The two bare rows catch anything else
# under resources/ (durable, like models) and output/ (tool results and
# logs, like the A/B rounds), keeping its name.
LEGACY_LAYOUT = (
    ( 'resources/model',                 'data',    'model' ),
    ( 'resources/stickers',              'data',    'stickers' ),
    ( 'resources/betaconfig_machine.py', 'config',  'betaconfig_machine.py' ),
    ( 'resources/uncensored_vids',       'inputs',  'uncensored_vids' ),
    ( 'resources/uncensored_pics',       'inputs',  'uncensored_pics' ),
    ( 'resources/source',                'inputs',  'source' ),
    ( 'resources',                       'data',    '' ),
    ( 'output/censored_vids',            'outputs', 'censored_vids' ),
    ( 'output/censored_pics',            'outputs', 'censored_pics' ),
    ( 'output/betafied',                 'outputs', 'betafied' ),
    ( 'output/cache/run_keys',           'data',    'run_keys' ),
    ( 'output/cache',                    'cache',   '' ),
    ( 'output/logs',                     'state',   'logs' ),
    ( 'output/stats',                    'state',   'stats' ),
    ( 'output/benchmarks',               'state',   'benchmarks' ),
    ( 'output/run.lock',                 'state',   'run.lock' ),
    ( 'output',                          'state',   '' ),
)

# Where each kind sits inside a 1.x data folder, for files 2.x adds.
# Things 1.x already had are found through LEGACY_LAYOUT instead (run
# keys, for one, sit in output/cache in 1.x and in data in 2.x).
LEGACY_KIND_DIRS = {
    'config':  'resources',
    'data':    'resources',
    'cache':   'output/cache',
    'state':   'output',
    'inputs':  'resources',
    'outputs': 'output',
}

APP_NAME = 'BetaSuite'      # Windows and macOS folder names
XDG_APP_NAME = 'betasuite'  # XDG folder names are lower case


class PathsError( ValueError ):
    """A data location variable is set to something that cannot be used."""


class DataRootError( PathsError ):
    """BETASUITE_DATA_ROOT is set to something that cannot be used."""


def _prefix_pattern( separators ):
    """'..' + a separator + 'resources' or 'output', then a separator or the end."""
    sep = '[%s]'%( re.escape( separators ), )
    return re.compile( r'^\.\.%s(?:resources|output)(?:%s|$)'%( sep, sep ) )


_WINDOWS_SEPARATORS = '\\/'
_POSIX_SEPARATORS = '/'

# This OS's separators only: on Windows '..\\output\\logs' maps too, while
# on Linux that string is one oddly named file and is left alone.
_DATA_PREFIX = _prefix_pattern( os.sep + ( os.altsep or '' ) )


def _family( platform ):
    """'windows', 'macos' or 'xdg' for a sys.platform value."""
    if platform == 'win32':
        return 'windows'
    if platform == 'darwin':
        return 'macos'
    return 'xdg'


def _pathmod( family ):
    return ntpath if family == 'windows' else posixpath


def _is_absolute( value, family ):
    """Absolute for that OS. On Windows that needs a drive or a share: '\\x' hangs off whichever drive is current."""
    if family == 'windows':
        return ntpath.isabs( value ) and bool( ntpath.splitdrive( value )[0] )
    return posixpath.isabs( value )


def data_root( environ=None ):
    """
    The 1.x data folder from BETASUITE_DATA_ROOT, or None when it is unset.

    Read on every call, so a test (or a launcher) can change it without
    re-importing this module. Empty counts as unset.

    Returns:
        The normalised absolute folder, or None.

    Raises:
        DataRootError: the value is relative, or is not an existing folder.
    """
    environ = os.environ if environ is None else environ
    value = environ.get( ENV_VAR, '' )
    if not value:
        return None
    if not os.path.isabs( value ):
        raise DataRootError( "%s must be an absolute path, got %r ('~' is not expanded)"%(
            ENV_VAR, value ) )
    if not os.path.isdir( value ):
        raise DataRootError( "%s is %r, which is not an existing folder"%( ENV_VAR, value ) )
    return os.path.normpath( value )


def _checked_root( environ ):
    """data_root(), refusing the folder overrides beside it."""
    root = data_root( environ )
    if root is not None:
        clashing = sorted( name for name in OVERRIDES.values() if environ.get( name ) )
        if clashing:
            raise PathsError( "%s keeps the 1.x layout whole, so %s cannot be set with it; "
                              "unset one or the other"%( ENV_VAR, ' and '.join( clashing ) ) )
    return root


def _override( kind, environ, family ):
    name = OVERRIDES.get( kind )
    value = environ.get( name, '' ) if name else ''
    if not value:
        return None
    if not _is_absolute( value, family ):
        raise PathsError( "%s must be an absolute path, got %r ('~' is not expanded)"%( name, value ) )
    return _pathmod( family ).normpath( value )


def _home( environ, family, platform ):
    if family == 'windows':
        value = environ.get( 'USERPROFILE', '' ) or (
            environ.get( 'HOMEDRIVE', '' ) + environ.get( 'HOMEPATH', '' ) )
    else:
        value = environ.get( 'HOME', '' )
        if not value and platform == sys.platform and environ is os.environ:
            value = os.path.expanduser( '~' )
    if not value or not _is_absolute( value, family ):
        raise PathsError( "cannot find the home folder: %s is not set to an absolute path"%(
            'USERPROFILE' if family == 'windows' else 'HOME', ) )
    return _pathmod( family ).normpath( value )


def _xdg_base( environ, name, default ):
    """An XDG base folder: the variable when absolute, else the default."""
    value = environ.get( name, '' )
    return posixpath.normpath( value ) if value and posixpath.isabs( value ) else default


# The cache folder setting (cache_dir in betaconfig_local.py), read here
# because the paths are needed before betaconfig is loaded (setup, the
# launchers, where.py). The file is read as Python literals, never run,
# as betautils_local reads it; the same file BETASUITE_LOCAL_CONFIG names.
LOCAL_CONFIG_ENV_VAR = 'BETASUITE_LOCAL_CONFIG'
LOCAL_CONFIG_FILE = 'betaconfig_local.py'
CACHE_SETTING = 'cache_dir'
_cache_setting_memo = {}


def _local_config_file( platform, environ ):
    value = environ.get( LOCAL_CONFIG_ENV_VAR )
    if value is not None:
        if value.strip().lower() in ( '', 'none', 'off' ):
            return None
        return os.path.abspath( os.path.expanduser( value ) )
    return _pathmod( _family( platform ) ).join( _os_kind_dir( 'config', platform, environ, setting=False ),
                                                 LOCAL_CONFIG_FILE )


def read_cache_setting( config_file ):
    """
    cache_dir from a local config file: an absolute folder, or None when
    the file, the line or a usable value is missing. Remembered per file
    and modification time, since kind_dir is asked often.
    """
    if config_file is None:
        return None
    try:
        stamp = os.stat( config_file ).st_mtime_ns
    except OSError:
        return None
    memo = _cache_setting_memo.get( config_file )
    if memo is not None and memo[0] == stamp:
        return memo[1]
    import ast
    value = None
    try:
        with open( config_file, 'r', encoding='utf-8' ) as handle:
            tree = ast.parse( handle.read() )
        for node in tree.body:
            if ( isinstance( node, ast.Assign ) and len( node.targets ) == 1
                    and isinstance( node.targets[0], ast.Name ) and node.targets[0].id == CACHE_SETTING ):
                value = ast.literal_eval( node.value )
    except ( OSError, SyntaxError, ValueError, UnicodeDecodeError ):
        value = None
    if not isinstance( value, str ) or not os.path.isabs( value ):
        value = None
    else:
        value = os.path.normpath( value )
    _cache_setting_memo[config_file] = ( stamp, value )
    return value


def cache_source( platform=None, environ=None ):
    """Where the cache folder comes from: 'variable' (BETASUITE_CACHE_DIR), 'setting' (cache_dir) or 'default'."""
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    if _override( 'cache', environ, _family( platform ) ) is not None:
        return 'variable'
    if read_cache_setting( _local_config_file( platform, environ ) ) is not None:
        return 'setting'
    return 'default'


def _os_kind_dir( kind, platform, environ, setting=True ):
    family = _family( platform )
    path = _pathmod( family )
    override = _override( kind, environ, family )
    if override is not None:
        return override
    if kind == 'cache' and setting:
        configured = read_cache_setting( _local_config_file( platform, environ ) )
        if configured is not None:
            return configured
    home = _home( environ, family, platform )
    media = kind in ( 'inputs', 'outputs' )
    if family == 'windows':
        local = environ.get( 'LOCALAPPDATA', '' )
        if not ( local and _is_absolute( local, family ) ):
            local = path.join( home, 'AppData', 'Local' )
        return path.join( path.normpath( local ), APP_NAME, 'library' if media else kind )
    if family == 'macos':
        if kind == 'cache':
            return path.join( home, 'Library', 'Caches', APP_NAME )
        return path.join( home, 'Library', 'Application Support', APP_NAME,
                          'library.noindex' if media else kind )
    if kind in ( 'data', 'inputs', 'outputs' ):
        share = _xdg_base( environ, 'XDG_DATA_HOME', path.join( home, '.local', 'share' ) )
        return path.join( share, XDG_APP_NAME, 'library' if media else 'data' )
    variable, default = {
        'config': ( 'XDG_CONFIG_HOME', ( '.config', ) ),
        'cache':  ( 'XDG_CACHE_HOME', ( '.cache', ) ),
        'state':  ( 'XDG_STATE_HOME', ( '.local', 'state' ) ),
    }[kind]
    return path.join( _xdg_base( environ, variable, path.join( home, *default ) ), XDG_APP_NAME )


def _check_kind( kind ):
    if kind not in KINDS:
        raise ValueError( "unknown kind of data %r; one of %s"%( kind, ', '.join( KINDS ) ) )


def os_dirs( platform=None, environ=None ):
    """
    The OS folder for every kind, whatever BETASUITE_DATA_ROOT says.

    The migration's targets, and what kind_dir() returns when the root
    is unset. Nothing here touches the filesystem.

    Args:
        platform: A sys.platform value; default this one. 'win32' gives
            Windows paths on any OS, so tests can check them anywhere.
        environ: The environment to read; default os.environ.

    Returns:
        { kind: absolute folder }.

    Raises:
        PathsError: an override is relative, or there is no home folder.
    """
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    return { kind: _os_kind_dir( kind, platform, environ ) for kind in KINDS }


def kind_dir( kind, platform=None, environ=None ):
    """
    The folder for one kind of data in the layout in use.

    Under BETASUITE_DATA_ROOT that is the kind's place inside the 1.x
    tree (LEGACY_KIND_DIRS); otherwise its OS folder. Not created here.

    Raises:
        ValueError: kind is not one of KINDS.
        PathsError: a variable is set to something unusable.
    """
    _check_kind( kind )
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    root = _checked_root( environ )
    if root is not None:
        return os.path.join( root, *LEGACY_KIND_DIRS[kind].split( '/' ) )
    return _os_kind_dir( kind, platform, environ )


def legacy_entry( parts ):
    """
    The LEGACY_LAYOUT row for a 1.x path given as parts under the data
    folder ( 'output', 'cache', 'vid_hashes' ), and the parts below it.

    Returns:
        ( legacy prefix, kind, sub, remaining parts ), or None when parts
        does not start with 'resources' or 'output'.
    """
    parts = tuple( parts )
    best = None
    for prefix, kind, sub in LEGACY_LAYOUT:
        prefix_parts = tuple( prefix.split( '/' ) )
        if parts[:len( prefix_parts )] == prefix_parts and (
                best is None or len( prefix_parts ) > len( best[0].split( '/' ) ) ):
            best = ( prefix, kind, sub, parts[len( prefix_parts ):] )
    return best


def _os_location( parts, trailing, platform, environ ):
    """Where a 1.x path, as parts under the data folder, is in the OS layout."""
    prefix, kind, sub, rest = legacy_entry( parts )
    path = _pathmod( _family( platform ) )
    pieces = [ piece for piece in ( sub, ) + tuple( rest ) if piece ]
    location = path.join( _os_kind_dir( kind, platform, environ ), *pieces )
    return location + path.sep if trailing else location


def resolve( path, platform=None, environ=None ):
    """
    Where a '../resources/...' or '../output/...' path really is.

    Args:
        path: Any path string, as betaconst or a config wrote it.
        platform, environ: For tests; see os_dirs().

    Returns:
        path itself when it does not start with '../resources' or
        '../output'; otherwise its place under BETASUITE_DATA_ROOT when
        that is set, or in the OS layout. A trailing separator is kept.

    Raises:
        PathsError: a variable is set to something unusable. Paths that
            are not mapped never consult them.
    """
    platform = sys.platform if platform is None else platform
    windows = _family( platform ) == 'windows'
    if platform == sys.platform:
        pattern, separators = _DATA_PREFIX, os.sep + ( os.altsep or '' )
    elif windows:
        pattern, separators = _prefix_pattern( _WINDOWS_SEPARATORS ), _WINDOWS_SEPARATORS
    else:
        pattern, separators = _prefix_pattern( _POSIX_SEPARATORS ), _POSIX_SEPARATORS
    if not isinstance( path, str ) or not pattern.match( path ):
        return path
    environ = os.environ if environ is None else environ
    root = _checked_root( environ )
    if root is not None:
        return os.path.join( root, path[3:] )
    rest = path[3:]
    parts = [ part for part in re.split( '[%s]'%( re.escape( separators ), ), rest ) if part ]
    return _os_location( parts, rest[-1:] in separators, platform, environ )


def legacy_path( *parts, platform=None, environ=None ):
    """
    A path inside the 1.x data folder, given as parts ( 'output', 'logs' ),
    wherever the layout in use puts it. For tools that name folders
    directly rather than through a '../output/...' string.

    Under BETASUITE_DATA_ROOT it is the same parts under the root. In the
    OS layout, parts under resources/ or output/ go through
    LEGACY_LAYOUT; anything else 1.x kept beside them (a tool's results
    folder) goes in the state folder.
    """
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    root = _checked_root( environ )
    if root is not None:
        return os.path.join( root, *parts )
    if parts and parts[0] in ( 'resources', 'output' ):
        return _os_location( parts, False, platform, environ )
    return _pathmod( _family( platform ) ).join(
        _os_kind_dir( 'state', platform, environ ), *parts )


def legacy_tree_hint( repo_dir, environ=None ):
    """
    A line for the run log when the OS layout is in use, its data folder
    does not exist yet, and a 1.x tree sits beside the code: that tree is
    not read any more. None otherwise, and None for any problem finding
    out, since this is only a hint.
    """
    environ = os.environ if environ is None else environ
    try:
        if _checked_root( environ ) is not None:
            return None
        data = _os_kind_dir( 'data', sys.platform, environ )
    except PathsError:
        return None
    beside = os.path.normpath( os.path.join( repo_dir, '..' ) )
    if os.path.exists( data ):
        return None
    # resources/ is what makes a 1.x tree: a stray output/ (a tool run from
    # an odd folder) is not one.
    if not os.path.isdir( os.path.join( beside, 'resources' ) ):
        return None
    return ( "a 1.x layout (resources/, output/) is in %s, and this run does not read it: it uses the "
             "2.x folders by kind (data in %s). Copy it over with tools/setup/migrate_1x.sh (copies, never "
             "moves), or set %s=%s to keep using it"%( beside, data, ENV_VAR, beside ) )
