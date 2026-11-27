"""
betautils_history.py - "Clear history": remove the traces of what was
watched and censored, by kind, like a browser's "clear browsing data".

Each kind says what it holds, whether the dialog ticks it by default
(traces that are cheap to get back), whether clearing it needs confirm
(work to redo), and whether it can be limited to a time range. A range is
"since" a time: files written since then, catalog rows that ended since
then, log lines stamped since then.

  run_history   the catalog's run records (when each video ran, with
                what), the catalog's backup copies, which hold them too,
                and the engine's list of ended jobs (serve layer)
  previews      thumbnails, scrub strips and look previews
  transcodes    transcoded copies of videos
  logs          the app's and tools' logs and the engine transcript. A
                log the app is writing stays open: the cleared part is
                copied out and that copy is discarded, then the log is
                cut back in place, so later lines never land in the trash
  detections    detections and shot cuts (confirm: every video is
                detected again)
  run_keys      run keys and the file-hash cache (confirm, all time only:
                every video is hashed again and outputs count as stale)
  stats         run statistics and benchmark results (confirm)
  library       the catalog's videos, watched folders and pins (confirm,
                all time only); their run records go with them

Settings, models and outputs are never touched. Files go to the system
trash (betautils_trash) unless permanent; catalog rows are deleted from
the database for good either way, and the file is vacuumed so the rows
do not linger in free pages. Stdlib only.
"""

import os
import re

KIND_IDS = ( 'run_history', 'previews', 'transcodes', 'logs', 'detections', 'run_keys', 'stats', 'library' )

# A log line's time: 2026-01-02 03:04:05 anywhere near its start, with or
# without brackets (Python logging, the shell harness, the UI log).
_STAMP = re.compile( r'(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})' )
_STAMP_WITHIN = 40


class Kind:

    __slots__ = ( 'id', 'default', 'confirm', 'ranged', 'paths', 'rows' )

    def __init__( self, id, default=False, confirm=False, ranged=True, paths=(), rows=None ):
        self.id = id
        self.default = default
        self.confirm = confirm
        self.ranged = ranged
        self.paths = tuple( paths )
        # 'runs' or 'library': which catalog rows the kind clears.
        self.rows = rows


class HistoryError( Exception ):

    def __init__( self, code, params ):
        super().__init__( code )
        self.code = code
        self.params = params


def kinds( catalog_path=None ):
    """Every kind, in the order the dialog lists them."""
    import betaconst
    import betautils_catalog_store as bu_store
    import betautils_catalog_strip as bu_strip
    import betautils_catalog_thumbs as bu_thumbs
    import betautils_paths as bu_paths
    state = bu_paths.kind_dir( 'state' )
    catalog_path = catalog_path or bu_store.default_path()
    backups = _Backups( catalog_path )
    return [
        Kind( 'run_history', default=True, paths=[ backups ], rows='runs' ),
        Kind( 'previews', default=True, paths=[ bu_thumbs.folder(), bu_strip.folder(), betaconst.look_preview_dir ] ),
        Kind( 'transcodes', default=True, paths=[ betaconst.transcode_cache_dir ] ),
        Kind( 'logs', default=True, paths=[ _Logs( os.path.join( state, 'logs' ) ) ] ),
        Kind( 'detections', confirm=True,
              paths=[ betaconst.vid_hash_dir, betaconst.pic_hash_dir, betaconst.shot_cut_dir ] ),
        Kind( 'run_keys', confirm=True, ranged=False,
              paths=[ betaconst.run_key_dir, betaconst.file_hash_cache_path ] ),
        Kind( 'stats', confirm=True, paths=[ os.path.join( state, 'stats' ), betaconst.benchmark_dir ] ),
        Kind( 'library', confirm=True, ranged=False, paths=[ backups ], rows='library' ),
    ]


class _Backups:
    """The catalog's backup copies (catalog.sqlite.v<N>.bak), beside it."""

    def __init__( self, catalog_path ):
        self.folder = os.path.dirname( os.path.abspath( catalog_path ) )
        self.prefix = os.path.basename( catalog_path ) + '.v'

    def files( self ):
        if not os.path.isdir( self.folder ):
            return []
        return sorted( os.path.join( self.folder, name ) for name in os.listdir( self.folder )
                       if name.startswith( self.prefix ) and name.endswith( '.bak' ) )


class _Logs:
    """A logs folder, cleared line by line (see _clear_log)."""

    def __init__( self, folder ):
        self.folder = folder


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def _files_under( path ):
    if isinstance( path, _Backups ):
        yield from path.files()
        return
    folder = path.folder if isinstance( path, _Logs ) else path
    if not os.path.lexists( folder ):
        return
    if os.path.isfile( folder ) or os.path.islink( folder ):
        yield folder
        return
    for root, _dirs, files in os.walk( folder ):
        for name in files:
            yield os.path.join( root, name )


def _in_range( path, since ):
    if since is None:
        return True
    try:
        return os.lstat( path ).st_mtime >= since
    except OSError:
        return False


def _stamp( line ):
    found = _STAMP.search( line[:_STAMP_WITHIN] )
    if found is None:
        return None
    import time
    try:
        return time.mktime( tuple( int( part ) for part in found.groups() ) + ( 0, 0, -1 ) )
    except ( OverflowError, ValueError ):
        return None


def _split_log( lines, since ):
    """( kept, cleared ): lines stamped before since stay, with the unstamped lines that follow them."""
    if since is None:
        return [], list( lines )
    kept, cleared = [], []
    current_in = None
    for line in lines:
        stamp = _stamp( line )
        if stamp is not None:
            current_in = stamp >= since
        elif current_in is None:
            current_in = False   # lines before the first stamp are older than it
        ( cleared if current_in else kept ).append( line )
    return kept, cleared


def _log_lines( path ):
    with open( path, 'r', encoding='utf-8', errors='surrogateescape', newline='' ) as handle:
        return handle.readlines()


def _log_measure( path, since ):
    """( bytes, cleared ) a log would lose."""
    if not _in_range( path, since ):
        return 0, False
    try:
        lines = _log_lines( path )
    except OSError:
        return 0, False
    if not any( _stamp( line ) is not None for line in lines ):
        # No times in it at all (a transcript): it goes whole when written in the range.
        return os.lstat( path ).st_size, True
    _kept, cleared = _split_log( lines, since )
    size = len( ''.join( cleared ).encode( 'utf-8', errors='surrogateescape' ) )
    return size, bool( cleared )


def _clear_log( path, since, discard ):
    """Copy the cleared lines out and discard the copy; cut the log back in place. ( bytes, cleared )."""
    if not _in_range( path, since ):
        return 0, False
    lines = _log_lines( path )
    if not any( _stamp( line ) is not None for line in lines ):
        kept, cleared = [], lines
    else:
        kept, cleared = _split_log( lines, since )
    if not cleared:
        return 0, False
    data = ''.join( cleared ).encode( 'utf-8', errors='surrogateescape' )
    copy = _free_name( path + '.cleared' )
    with open( copy, 'wb' ) as handle:
        handle.write( data )
    try:
        discard( copy )
    except BaseException:
        os.remove( copy )
        raise
    # In place, never replaced: a writer holding it open keeps writing here.
    with open( path, 'r+b' ) as handle:
        handle.write( ''.join( kept ).encode( 'utf-8', errors='surrogateescape' ) )
        handle.truncate()
    return len( data ), True


def _free_name( base ):
    name, n = base, 1
    while os.path.lexists( name ):
        n += 1
        name = '%s%d'%( base, n )
    return name


# ---------------------------------------------------------------------------
# Catalog rows
# ---------------------------------------------------------------------------

def _row_count( store, rows, since ):
    if store is None or rows is None:
        return 0
    if rows == 'runs':
        if since is None:
            return store.read( 'SELECT COUNT(*) FROM runs' )[0][0]
        return store.read( 'SELECT COUNT(*) FROM runs WHERE ended_at >= ?', ( since, ) )[0][0]
    return ( store.read( 'SELECT COUNT(*) FROM items' )[0][0]
             + store.read( 'SELECT COUNT(*) FROM folders' )[0][0] )


def _clear_rows( store, rows, since ):
    if store is None or rows is None:
        return 0
    count = _row_count( store, rows, since )
    with store.write() as db:
        if rows == 'runs':
            if since is None:
                db.execute( 'DELETE FROM runs' )
            else:
                db.execute( 'DELETE FROM runs WHERE ended_at >= ?', ( since, ) )
        else:
            # Locations, outputs and runs go with their items (ON DELETE CASCADE).
            db.execute( 'DELETE FROM items' )
            db.execute( 'DELETE FROM folders' )
    return count


# ---------------------------------------------------------------------------
# The two calls
# ---------------------------------------------------------------------------

def _since_for( kind, since ):
    return since if kind.ranged else None


def summary( all_kinds, since=None, store=None ):
    """[ { id, default, confirm, ranged, bytes, files, rows } ]: what clearing each would remove."""
    out = []
    for kind in all_kinds:
        when = _since_for( kind, since )
        size = files = 0
        for path in kind.paths:
            for full in _files_under( path ):
                if isinstance( path, _Logs ):
                    lost, hit = _log_measure( full, when )
                    size += lost
                    files += int( hit )
                elif _in_range( full, when ):
                    try:
                        size += os.lstat( full ).st_size
                        files += 1
                    except OSError:
                        continue
        out.append( { 'id': kind.id, 'default': kind.default, 'confirm': kind.confirm,
                      'ranged': kind.ranged, 'bytes': size, 'files': files,
                      'rows': _row_count( store, kind.rows, when ) } )
    return out


def clear( all_kinds, ids, since=None, confirm=False, busy=False, permanent=False, trash=None, store=None ):
    """
    Clear the named kinds. Returns [ { id, bytes, files, rows } ].

    Raises HistoryError (codes shared with storage.clean):
      storage.unknown_category   an id no kind has
      storage.needs_confirm      a confirm kind without confirm
      storage.busy               any kind while a job holds the queue
    and betautils_trash.TrashError when the trash refuses a file (what was
    already moved stays in the trash; nothing is removed for good instead).
    Nothing is cleared unless every named kind may be.
    """
    by_id = { kind.id: kind for kind in all_kinds }
    chosen = []
    for kind_id in ids:
        kind = by_id.get( kind_id )
        if kind is None:
            raise HistoryError( 'storage.unknown_category', { 'category': kind_id } )
        if kind.confirm and not confirm:
            raise HistoryError( 'storage.needs_confirm', { 'category': kind_id, 'policy': 'costly' } )
        if busy:
            raise HistoryError( 'storage.busy', { 'category': kind_id } )
        if kind not in chosen:
            chosen.append( kind )
    if permanent:
        discard = _permanently
    else:
        if trash is None:
            import betautils_trash as bu_trash
            trash = bu_trash.trash
        discard = trash
    results = []
    touched_rows = False
    for kind in chosen:
        when = _since_for( kind, since )
        size = files = 0
        for path in kind.paths:
            for full in list( _files_under( path ) ):
                if isinstance( path, _Logs ):
                    lost, hit = _clear_log( full, when, discard )
                    size += lost
                    files += int( hit )
                    continue
                if not _in_range( full, when ):
                    continue
                try:
                    length = os.lstat( full ).st_size
                except OSError:
                    continue
                discard( full )
                size += length
                files += 1
        rows = _clear_rows( store, kind.rows, when )
        touched_rows = touched_rows or bool( rows )
        results.append( { 'id': kind.id, 'bytes': size, 'files': files, 'rows': rows } )
    if touched_rows:
        store.vacuum()
    return results


def _permanently( path ):
    os.remove( path )
