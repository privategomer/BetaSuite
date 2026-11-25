"""
betautils_catalog_index.py - what the catalog knows about files on disk:
watched folders, items keyed by content hash, where each item is, and
which outputs each has.

An item is one piece of content, identified by the same 16-character
hash the engine puts in its cache keys and output names
(betautils_hash.md5_for_file). A location is one path that content was
found at. Copies of one file are one item with several locations.

RE-FINDING. Every pass looks at each file's size, mtime, device and
inode, never its bytes, unless something changed:

  - the path is known and nothing changed: nothing to do
  - a known location's device, inode, size and mtime now sit at a new
    path: a rename or a move within one disk. The location follows it,
    no hashing. (A second path for the same inode is a hard link: the
    same item again)
  - anything else: the file is hashed, by the caller, later
    (needs_hash). A hash that names an existing item adds or re-finds a
    location of it, keeping its status, history and pin

Availability is per location: present, missing (its folder was read and
the file was not there) or offline (its folder could not be read: an
unmounted share, an unplugged drive). Nothing is marked missing while
its folder is offline, and nothing is removed because it went missing.

This module does no hashing and starts no threads; the scanner
(betautils_catalog_scan) decides when to hash and calls back in.
"""

import json
import os
import stat as stat_module

import betautils_catalog_store as bu_store

PRESENT = 'present'
MISSING = 'missing'
OFFLINE = 'offline'
AVAILABILITY = ( PRESENT, MISSING, OFFLINE )

# What the catalog indexes as a video. betatv itself tries to decode any
# file; a library scan cannot afford that on every file of every pass.
VIDEO_EXTENSIONS = frozenset( (
    '.mp4', '.mkv', '.webm', '.avi', '.mov', '.m4v', '.wmv', '.flv', '.ts', '.m2ts',
    '.mts', '.mpg', '.mpeg', '.3gp', '.ogv',
) )

# Stored statuses. queued and processing exist only while a job runs
# (betautils_catalog_status), so a crash never leaves an item stuck.
NEW = 'new'
DONE = 'done'
FAILED = 'failed'


def is_video_name( name ):
    if name.startswith( '.' ):
        return False
    base, extension = os.path.splitext( name )
    if base.endswith( '.tmp' ) or '.part' in base[-9:]:
        return False
    return extension.lower() in VIDEO_EXTENSIONS


def norm( path ):
    """The form paths are stored and compared in: absolute, normalized."""
    return os.path.normpath( os.path.abspath( path ) )


def is_within( path, folder ):
    path, folder = os.path.normcase( norm( path ) ), os.path.normcase( norm( folder ) )
    try:
        return os.path.commonpath( [ path, folder ] ) == folder
    except ValueError:   # Windows: different drives
        return False


def output_roots():
    """The folders the engine writes to. Never watched or indexed."""
    import betaconst
    return [ norm( betaconst.video_path_censored ), norm( betaconst.picture_path_censored ) ]


class Stat:
    """The four facts a pass compares, from os.stat."""

    __slots__ = ( 'size', 'mtime_ns', 'dev', 'ino' )

    def __init__( self, size, mtime_ns, dev, ino ):
        self.size, self.mtime_ns, self.dev, self.ino = size, mtime_ns, dev, ino

    @classmethod
    def of( cls, result ):
        return cls( result.st_size, result.st_mtime_ns, signed64( result.st_dev ), signed64( result.st_ino ) )

    def same_file( self, other ):
        return ( self.size, self.mtime_ns ) == ( other.size, other.mtime_ns )

    def same_inode( self, other ):
        # A zero inode (some network filesystems) identifies nothing.
        return ( self.ino != 0 and ( self.dev, self.ino, self.size, self.mtime_ns )
                 == ( other.dev, other.ino, other.size, other.mtime_ns ) )

    def __eq__( self, other ):
        return isinstance( other, Stat ) and ( self.size, self.mtime_ns, self.dev, self.ino ) == (
            other.size, other.mtime_ns, other.dev, other.ino )

    def __repr__( self ):
        return 'Stat(%d, %d, %d, %d)'%( self.size, self.mtime_ns, self.dev, self.ino )


def signed64( value ):
    """
    value as a signed 64-bit integer, which is what SQLite stores. Windows
    reports file ids as unsigned 64-bit (and ReFS as 128-bit), so a large
    one would not fit. The same file always folds to the same number, and
    the four facts are compared together, so a fold shared by two files
    would also need their size and mtime to the nanosecond to match.
    """
    value = int( value )
    while value >= 1 << 64:
        value = ( value & ( ( 1 << 64 ) - 1 ) ) ^ ( value >> 64 )
    return value - ( 1 << 64 ) if value >= 1 << 63 else value


def _row_stat( row ):
    return Stat( row['size'], row['mtime_ns'], row['dev'], row['ino'] )


class Index:
    """
    The catalog's tables of folders, items, locations and outputs.

    Every method that changes something returns the set of item hashes
    it touched, so the caller can say what changed.
    """

    def __init__( self, store, clock=None, logger=None ):
        self.store = store
        self.clock = clock or store.clock
        self.logger = logger

    # --- folders --------------------------------------------------------

    def folders( self ):
        return [ dict( row ) for row in self.store.read(
            'SELECT id, path, type_from_folder, added_at FROM folders ORDER BY id' ) ]

    def folder( self, folder_id ):
        rows = self.store.read( 'SELECT id, path, type_from_folder, added_at FROM folders WHERE id = ?',
                                ( folder_id, ) )
        if not rows:
            raise bu_store.CatalogError( 'catalog.no_such_folder', folder=folder_id )
        return dict( rows[0] )

    def add_folder( self, path, type_from_folder=False ):
        """
        Watch a folder. Returns ( folder dict, created ). Adding a folder
        already watched changes nothing but its folder rule.
        """
        path = norm( path )
        if not os.path.exists( path ):
            raise bu_store.CatalogError( 'catalog.bad_folder', reason='not_found' )
        if not os.path.isdir( path ):
            raise bu_store.CatalogError( 'catalog.bad_folder', reason='not_a_folder' )
        if any( is_within( path, root ) for root in output_roots() ):
            raise bu_store.CatalogError( 'catalog.bad_folder', reason='output_folder' )
        with self.store.write() as db:
            row = db.execute( 'SELECT id FROM folders WHERE path = ?', ( path, ) ).fetchone()
            if row is not None:
                db.execute( 'UPDATE folders SET type_from_folder = ? WHERE id = ?',
                            ( int( bool( type_from_folder ) ), row['id'] ) )
                created, folder_id = False, row['id']
            else:
                cursor = db.execute( 'INSERT INTO folders ( path, type_from_folder, added_at ) VALUES ( ?, ?, ? )',
                                     ( path, int( bool( type_from_folder ) ), self.clock() ) )
                created, folder_id = True, cursor.lastrowid
                self._reassign_folders( db )
        return self.folder( folder_id ), created

    def remove_folder( self, folder_id ):
        """
        Stop watching a folder. Its locations go; its items stay, with
        their history and pins, and come back if the folder is added
        again or the files turn up elsewhere. Returns the items touched.
        """
        self.folder( folder_id )
        with self.store.write() as db:
            touched = { row['hash'] for row in db.execute(
                'SELECT hash FROM locations WHERE folder_id = ?', ( folder_id, ) ) }
            db.execute( 'DELETE FROM locations WHERE folder_id = ?', ( folder_id, ) )
            db.execute( 'DELETE FROM folders WHERE id = ?', ( folder_id, ) )
            self._reassign_folders( db )
        return touched

    def folder_for( self, path, folders=None ):
        """The deepest watched folder holding path, or None."""
        best = None
        for folder in ( folders if folders is not None else self.folders() ):
            if is_within( path, folder['path'] ) and (
                    best is None or len( folder['path'] ) > len( best['path'] ) ):
                best = folder
        return best

    def _reassign_folders( self, db ):
        """Each location belongs to the deepest folder that holds it."""
        folders = [ dict( row ) for row in db.execute( 'SELECT id, path, type_from_folder FROM folders' ) ]
        for row in db.execute( 'SELECT id, path, folder_id FROM locations' ).fetchall():
            folder = self.folder_for( row['path'], folders )
            folder_id = folder['id'] if folder else None
            if folder_id != row['folder_id']:
                db.execute( 'UPDATE locations SET folder_id = ? WHERE id = ?', ( folder_id, row['id'] ) )

    # --- looking at one file ----------------------------------------------

    def observe( self, path, file_stat, folder_id=None ):
        """
        What a pass saw at path. Returns ( outcome, touched ): outcome is
        'same', 'moved', 'linked' (a hard link of a known item), or
        'needs_hash' when only the file's bytes can tell.
        """
        path = norm( path )
        now = self.clock()
        with self.store.write() as db:
            row = db.execute( 'SELECT * FROM locations WHERE path = ?', ( path, ) ).fetchone()
            if row is not None and _row_stat( row ).same_file( file_stat ):
                db.execute( 'UPDATE locations SET state = ?, seen_at = ?, dev = ?, ino = ?, folder_id = ?'
                            ' WHERE id = ?', ( PRESENT, now, file_stat.dev, file_stat.ino, folder_id, row['id'] ) )
                return 'same', ( { row['hash'] } if row['state'] != PRESENT else set() )
            if row is not None:
                # Same path, other bytes: only a hash can say what it is now.
                return 'needs_hash', set()
            if file_stat.ino != 0:
                for other in db.execute( 'SELECT * FROM locations WHERE dev = ? AND ino = ? AND size = ?'
                                         ' AND mtime_ns = ?', ( file_stat.dev, file_stat.ino, file_stat.size,
                                                                file_stat.mtime_ns ) ).fetchall():
                    if not os.path.lexists( other['path'] ):
                        db.execute( 'UPDATE locations SET path = ?, folder_id = ?, state = ?, seen_at = ?'
                                    ' WHERE id = ?', ( path, folder_id, PRESENT, now, other['id'] ) )
                        return 'moved', { other['hash'] }
                    self._insert_location( db, other['hash'], path, file_stat, folder_id, now )
                    return 'linked', { other['hash'] }
        return 'needs_hash', set()

    def record_hash( self, path, file_stat, digest, folder_id=None, kind='video' ):
        """
        The hash of the file at path, taken while it had file_stat.
        Adds the item if it is new, and the location to it. Returns
        ( outcome, touched ), outcome one of:

          added      content the catalog did not know: a new item
          refound    an item that had no present location: it is back
          copy       another location of an item present elsewhere
          unchanged  a known path whose mtime moved but bytes did not
        """
        path = norm( path )
        now = self.clock()
        touched = { digest }
        with self.store.write() as db:
            item = db.execute( 'SELECT hash FROM items WHERE hash = ?', ( digest, ) ).fetchone()
            row = db.execute( 'SELECT id, hash FROM locations WHERE path = ?', ( path, ) ).fetchone()
            present_elsewhere = item is not None and db.execute(
                'SELECT 1 FROM locations WHERE hash = ? AND state = ? AND path != ?',
                ( digest, PRESENT, path ) ).fetchone() is not None
            if item is None:
                db.execute( 'INSERT INTO items ( hash, kind, size, name, added_at, status ) VALUES ( ?, ?, ?, ?, ?, ? )',
                            ( digest, kind, file_stat.size, os.path.basename( path ), now, NEW ) )
                outcome = 'added'
            elif row is not None and row['hash'] == digest:
                outcome = 'unchanged'
            elif present_elsewhere:
                outcome = 'copy'
            else:
                outcome = 'refound'
            if row is not None:
                if row['hash'] != digest:
                    # The path now holds other content: it leaves its old item.
                    touched.add( row['hash'] )
                db.execute( 'UPDATE locations SET hash = ?, size = ?, mtime_ns = ?, dev = ?, ino = ?,'
                            ' folder_id = ?, state = ?, seen_at = ? WHERE id = ?',
                            ( digest, file_stat.size, file_stat.mtime_ns, file_stat.dev, file_stat.ino,
                              folder_id, PRESENT, now, row['id'] ) )
            else:
                self._insert_location( db, digest, path, file_stat, folder_id, now )
            if outcome in ( 'refound', 'copy' ) and self.logger is not None:
                self.logger.debug( "catalog: %s found again at %s"%( digest, path ) )
        return outcome, touched

    def _insert_location( self, db, digest, path, file_stat, folder_id, now ):
        db.execute( 'INSERT INTO locations ( hash, path, folder_id, size, mtime_ns, dev, ino, state,'
                    ' added_at, seen_at ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )',
                    ( digest, path, folder_id, file_stat.size, file_stat.mtime_ns, file_stat.dev,
                      file_stat.ino, PRESENT, now, now ) )

    def mark( self, paths, state ):
        """Set the availability of known locations. Returns items touched."""
        touched = set()
        with self.store.write() as db:
            for path in paths:
                row = db.execute( 'SELECT id, hash, state FROM locations WHERE path = ?', ( norm( path ), ) ).fetchone()
                if row is not None and row['state'] != state:
                    db.execute( 'UPDATE locations SET state = ? WHERE id = ?', ( state, row['id'] ) )
                    touched.add( row['hash'] )
        return touched

    def drop_location( self, path ):
        """Forget one path (its item stays). Returns the items touched."""
        with self.store.write() as db:
            row = db.execute( 'SELECT id, hash FROM locations WHERE path = ?', ( norm( path ), ) ).fetchone()
            if row is None:
                return set()
            db.execute( 'DELETE FROM locations WHERE id = ?', ( row['id'], ) )
        return { row['hash'] }

    def location_hash( self, path ):
        """The item at a known path, or None."""
        rows = self.store.read( 'SELECT hash FROM locations WHERE path = ?', ( norm( path ), ) )
        return rows[0][0] if rows else None

    def locations_in( self, folder_id ):
        return [ dict( row ) for row in self.store.read(
            'SELECT * FROM locations WHERE folder_id IS ?', ( folder_id, ) ) ]

    def missing_sizes( self ):
        """Sizes of items with no present location, to hash those first."""
        return { row[0] for row in self.store.read(
            'SELECT i.size FROM items i WHERE NOT EXISTS ( SELECT 1 FROM locations l'
            ' WHERE l.hash = i.hash AND l.state = ? )', ( PRESENT, ) ) }

    # --- items --------------------------------------------------------------

    def has_item( self, digest ):
        return bool( self.store.read( 'SELECT 1 FROM items WHERE hash = ?', ( digest, ) ) )

    def item_row( self, digest ):
        rows = self.store.read( 'SELECT * FROM items WHERE hash = ?', ( digest, ) )
        if not rows:
            raise bu_store.CatalogError( 'catalog.no_such_item', item=digest )
        return dict( rows[0] )

    def locations_of( self, digest ):
        return [ dict( row ) for row in self.store.read(
            'SELECT * FROM locations WHERE hash = ? ORDER BY added_at, id', ( digest, ) ) ]

    def forget( self, digest ):
        """Remove an item, its locations, outputs and history."""
        self.item_row( digest )
        with self.store.write() as db:
            db.execute( 'DELETE FROM items WHERE hash = ?', ( digest, ) )
        return { digest }

    def set_status( self, digest, status ):
        with self.store.write() as db:
            db.execute( 'UPDATE items SET status = ? WHERE hash = ?', ( status, digest ) )

    def pin_of( self, digest ):
        """The type pinned to an item, or None (also for an unknown item)."""
        rows = self.store.read( 'SELECT type_pin FROM items WHERE hash = ?', ( digest, ) )
        return rows[0][0] if rows else None

    def set_pin( self, digest, pin ):
        self.item_row( digest )
        with self.store.write() as db:
            db.execute( 'UPDATE items SET type_pin = ? WHERE hash = ?', ( pin, digest ) )

    def set_last_profile( self, digest, profile, rule, params ):
        with self.store.write() as db:
            db.execute( 'UPDATE items SET last_profile = ?, last_rule = ?, last_rule_params = ? WHERE hash = ?',
                        ( profile, rule, bu_store.dumps_params( params ), digest ) )

    # --- outputs ------------------------------------------------------------

    def record_outputs( self, found ):
        """
        Make the outputs table say what one walk of the output folders
        found: found is { path: parsed name } for every final output on
        disk. Outputs of hashes the catalog does not know are left out.
        An item with a finished (non-preview) output that reads new is
        done. Returns items touched.
        """
        now = self.clock()
        touched = set()
        with self.store.write() as db:
            known = { row['path']: dict( row ) for row in db.execute( 'SELECT * FROM outputs' ) }
            items = { row['hash'] for row in db.execute( 'SELECT hash FROM items' ) }
            for path, fields in found.items():
                if fields['hash'] not in items or fields['preview']:
                    continue
                row = known.pop( path, None )
                if row is None:
                    db.execute( 'INSERT INTO outputs ( hash, path, detection_key, censor_key, encode_key,'
                                ' present, seen_at ) VALUES ( ?, ?, ?, ?, ?, 1, ? )',
                                ( fields['hash'], path, fields['detection_key'], fields['censor_key'],
                                  fields['encode_key'], now ) )
                    touched.add( fields['hash'] )
                else:
                    db.execute( 'UPDATE outputs SET present = 1, seen_at = ? WHERE id = ?', ( now, row['id'] ) )
                    if not row['present']:
                        touched.add( row['hash'] )
            for row in known.values():
                if row['present']:
                    db.execute( 'UPDATE outputs SET present = 0 WHERE id = ?', ( row['id'], ) )
                    touched.add( row['hash'] )
            for digest in touched:
                db.execute( "UPDATE items SET status = ? WHERE hash = ? AND status = ? AND EXISTS ("
                            " SELECT 1 FROM outputs o WHERE o.hash = items.hash AND o.present = 1 )",
                            ( DONE, digest, NEW ) )
        return touched

    def add_output( self, digest, path, fields ):
        """One output the engine just wrote. Returns items touched."""
        if fields is None or fields['preview'] or not self.has_item( digest ):
            return set()
        with self.store.write() as db:
            db.execute( 'INSERT INTO outputs ( hash, path, detection_key, censor_key, encode_key, present, seen_at )'
                        ' VALUES ( ?, ?, ?, ?, ?, 1, ? ) ON CONFLICT( path ) DO UPDATE SET present = 1,'
                        ' seen_at = excluded.seen_at, hash = excluded.hash',
                        ( digest, norm( path ), fields['detection_key'], fields['censor_key'],
                          fields['encode_key'], self.clock() ) )
        return { digest }

    def outputs_of( self, digest ):
        return [ dict( row ) for row in self.store.read(
            'SELECT path, detection_key, censor_key, encode_key, present, seen_at FROM outputs'
            ' WHERE hash = ? ORDER BY seen_at, id', ( digest, ) ) ]


def availability( states ):
    """An item's availability from its locations' states."""
    if PRESENT in states:
        return PRESENT
    if OFFLINE in states:
        return OFFLINE
    return MISSING


def walk_videos( folder, exclude=() ):
    """
    ( files, unreadable ) under folder: files is { path: Stat } for every
    video found, unreadable the subfolders that could not be listed
    (their locations are left as they are). Hidden folders, links and
    the excluded folders (outputs) are skipped.
    """
    files, unreadable = {}, []
    excluded = [ os.path.normcase( norm( path ) ) for path in exclude ]
    stack = [ norm( folder ) ]
    while stack:
        current = stack.pop()
        try:
            with os.scandir( current ) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir( follow_symlinks=False ):
                            if not entry.name.startswith( '.' ) and os.path.normcase(
                                    entry.path ) not in excluded:
                                stack.append( entry.path )
                        elif entry.is_file( follow_symlinks=False ) and is_video_name( entry.name ):
                            result = entry.stat( follow_symlinks=False )
                            if stat_module.S_ISREG( result.st_mode ) and result.st_size > 0:
                                # scandir's inode on Windows can be 0; os.stat fills it.
                                if result.st_ino == 0:
                                    result = os.stat( entry.path )
                                files[ entry.path ] = Stat.of( result )
                    except OSError:
                        continue
        except OSError:
            if current == norm( folder ):
                raise
            unreadable.append( current )
    return files, unreadable


def loads_params( text ):
    return json.loads( text ) if text else {}
