"""
betautils_trash.py - move files and folders to the system's trash, so
clearing something in the app can be undone from the file manager.

  trash( path )   move one file or folder to the trash; TrashError when
                  it cannot be done (nothing is deleted instead)

Where it goes:
  Linux and other freedesktop systems: the Trash specification. The home
      trash ($XDG_DATA_HOME/Trash, ~/.local/share/Trash) when the path is
      on the same disk; otherwise that disk's $topdir/.Trash/$uid (when
      it is a sticky folder, not a link) or $topdir/.Trash-$uid. Each item
      gets an info/<name>.trashinfo with its original path and the time,
      so the file manager can put it back
  Windows: the Recycle Bin, through the shell (SHFileOperationW with undo)
  macOS: ~/.Trash

Stdlib only. Never copies across disks: a move that would need a copy
is refused, so clearing a cache never doubles its size first.
"""

import os
import stat
import sys
import time
import urllib.parse


class TrashError( Exception ):
    """The path could not be moved to the trash (reason: a short word for the log)."""

    def __init__( self, reason ):
        super().__init__( reason )
        self.reason = reason


def trash( path, environ=None, now=time.localtime ):
    path = os.path.abspath( path )
    if not os.path.lexists( path ):
        return
    if sys.platform == 'win32':
        _trash_windows( path )
    elif sys.platform == 'darwin':
        _move_into( path, os.path.expanduser( '~/.Trash' ), None, now )
    else:
        _trash_freedesktop( path, os.environ if environ is None else environ, now )


# ---------------------------------------------------------------------------
# freedesktop
# ---------------------------------------------------------------------------

def home_trash( environ ):
    data = environ.get( 'XDG_DATA_HOME' ) or os.path.join( environ.get( 'HOME' ) or os.path.expanduser( '~' ),
                                                           '.local', 'share' )
    return os.path.join( data, 'Trash' )


def _device( path ):
    probe = path
    while not os.path.exists( probe ) and os.path.dirname( probe ) != probe:
        probe = os.path.dirname( probe )
    return os.lstat( probe ).st_dev


def mount_point( path ):
    path = os.path.realpath( path )
    while not os.path.ismount( path ):
        parent = os.path.dirname( path )
        if parent == path:
            break
        path = parent
    return path


def _topdir_trash( path ):
    top = mount_point( os.path.dirname( path ) )
    uid = os.getuid()
    shared = os.path.join( top, '.Trash' )
    try:
        info = os.lstat( shared )
        if stat.S_ISDIR( info.st_mode ) and not stat.S_ISLNK( info.st_mode ) and info.st_mode & stat.S_ISVTX:
            mine = os.path.join( shared, str( uid ) )
            os.makedirs( mine, mode=0o700, exist_ok=True )
            return mine, top
    except OSError:
        pass
    own = os.path.join( top, '.Trash-%d'%( uid, ) )
    try:
        os.makedirs( own, mode=0o700, exist_ok=True )
    except OSError as err:
        raise TrashError( 'no_trash_on_disk' ) from err
    return own, top


def _trash_freedesktop( path, environ, now ):
    home = home_trash( environ )
    if _device( home ) == _device( path ):
        _move_into( path, home, None, now )
        return
    trash_dir, top = _topdir_trash( path )
    if _device( trash_dir ) != _device( path ):
        raise TrashError( 'other_disk' )
    _move_into( path, trash_dir, top, now )


def _move_into( path, trash_dir, top, now ):
    """Move path into trash_dir/files with its trashinfo (freedesktop and, without info, macOS)."""
    freedesktop = sys.platform != 'darwin'
    files = os.path.join( trash_dir, 'files' ) if freedesktop else trash_dir
    info_dir = os.path.join( trash_dir, 'info' )
    try:
        os.makedirs( files, mode=0o700, exist_ok=True )
        if freedesktop:
            os.makedirs( info_dir, mode=0o700, exist_ok=True )
    except OSError as err:
        raise TrashError( 'cannot_make_trash' ) from err
    base = os.path.basename( path.rstrip( os.sep ) ) or 'item'
    for attempt in range( 1000 ):
        name = base if attempt == 0 else '%s.%d'%( base, attempt )
        if os.path.lexists( os.path.join( files, name ) ):
            continue
        info_path = os.path.join( info_dir, name + '.trashinfo' )
        if freedesktop:
            # The info file first, created exclusively: it reserves the name.
            original = os.path.relpath( path, top ) if top is not None else path
            text = '[Trash Info]\nPath=%s\nDeletionDate=%s\n'%(
                urllib.parse.quote( original ), time.strftime( '%Y-%m-%dT%H:%M:%S', now() ) )
            try:
                handle = os.open( info_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 )
            except FileExistsError:
                continue
            except OSError as err:
                raise TrashError( 'cannot_write_info' ) from err
            with os.fdopen( handle, 'w', encoding='utf-8' ) as stream:
                stream.write( text )
        try:
            os.rename( path, os.path.join( files, name ) )
        except OSError as err:
            if freedesktop:
                try:
                    os.remove( info_path )
                except OSError:
                    pass
            raise TrashError( 'cannot_move' ) from err
        return
    raise TrashError( 'no_free_name' )


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _trash_windows( path ):
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW( ctypes.Structure ):
        _fields_ = [ ( 'hwnd', wintypes.HWND ), ( 'wFunc', wintypes.UINT ), ( 'pFrom', wintypes.LPCWSTR ),
                     ( 'pTo', wintypes.LPCWSTR ), ( 'fFlags', ctypes.c_ushort ), ( 'fAnyOperationsAborted', wintypes.BOOL ),
                     ( 'hNameMappings', ctypes.c_void_p ), ( 'lpszProgressTitle', wintypes.LPCWSTR ) ]

    FO_DELETE = 3
    FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 0x4, 0x10, 0x40, 0x400
    operation = SHFILEOPSTRUCTW( None, FO_DELETE, path + '\0', None,
                                 FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI, False, None, None )
    result = ctypes.windll.shell32.SHFileOperationW( ctypes.byref( operation ) )
    if result != 0 or operation.fAnyOperationsAborted or os.path.lexists( path ):
        raise TrashError( 'recycle_bin_refused' )
