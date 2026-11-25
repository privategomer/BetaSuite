"""
betautils_catalog_rebuild.py - a new catalog file, built from the disk.

What the engine's own files cover is rebuilt from them (spec section 8):
the items in the watched folders, keyed by their content hash, where
each one is, and the outputs in the output folders (so which items are
done). What exists nowhere else is carried from the old file while it
can still be read: the watched folders and their rules, type pins, run
history (and with it a failed status), and items that are not on disk
right now (a share not mounted), with their paths marked missing or
offline. When the old file cannot be read, pins and history are lost:
that is what backing up the catalog file is for.

The rebuild holds the catalog's lock exclusive the whole time, so it
never runs beside an engine or a command-line run that has the catalog
open. It writes the new file beside the old one, keeps the old one as
catalog.sqlite.pre-rebuild.bak, and only then puts the new one in place.
"""

import os
import shutil
import sqlite3

import betautils_catalog as bu_catalog
import betautils_catalog_history as bu_history
import betautils_catalog_index as bu_index
import betautils_catalog_scan as bu_scan
import betautils_catalog_store as bu_store

NEW_SUFFIX = '.rebuild'
OLD_SUFFIX = '.pre-rebuild.bak'

_RUN_COLUMNS = ( 'hash', 'program', 'job', 'path', 'started_at', 'ended_at', 'outcome', 'reason', 'output',
                 'detection_key', 'censor_key', 'encode_key', 'profile', 'rule', 'rule_params', 'sample_fps',
                 'preview', 'error_type', 'stage' )


class Carried:
    """What the old file holds that the disk does not."""

    def __init__( self ):
        self.readable = False
        self.folders = []        # ( path, type_from_folder )
        self.items = {}          # hash -> item row
        self.locations = {}      # hash -> [ location rows ]
        self.runs = []           # run rows, oldest first


def read_old( path, logger=None ):
    """Everything worth carrying from the old file, or an empty Carried."""
    carried = Carried()
    if not os.path.exists( path ):
        return carried
    try:
        db = sqlite3.connect( 'file:%s?mode=ro'%( path, ), uri=True )
        db.row_factory = sqlite3.Row
        try:
            version = db.execute( 'PRAGMA user_version' ).fetchone()[0]
            if version > bu_store.LATEST_VERSION:
                raise bu_store.CatalogError( 'catalog.too_new', version=version, supported=bu_store.LATEST_VERSION )
            carried.folders = [ ( row['path'], bool( row['type_from_folder'] ) ) for row in
                                db.execute( 'SELECT path, type_from_folder FROM folders ORDER BY id' ) ]
            carried.items = { row['hash']: dict( row ) for row in db.execute( 'SELECT * FROM items' ) }
            for row in db.execute( 'SELECT * FROM locations ORDER BY added_at, id' ):
                carried.locations.setdefault( row['hash'], [] ).append( dict( row ) )
            if version >= 2:
                carried.runs = [ dict( row ) for row in db.execute( 'SELECT * FROM runs ORDER BY ended_at, id' ) ]
            carried.readable = True
        finally:
            db.close()
    except sqlite3.DatabaseError as err:
        if logger is not None:
            logger.warning( "the old catalog could not be read (%s); pins and history are not carried"%(
                type( err ).__name__, ) )
        return Carried()
    return carried


def rebuild( path=None, seed_folders=None, hasher=None, logger=None, clock=None ):
    """
    Build the catalog again. Returns a summary dict: items, locations,
    outputs, carried_pins, carried_runs, carried_missing, old_readable.
    Raises CatalogError catalog.unavailable {in_use} when anything has
    the catalog open, catalog.too_new for a file this code cannot read.
    """
    import time
    path = path or bu_store.default_path()
    clock = clock or time.time
    lock = bu_store.FileLock( os.path.join( os.path.dirname( os.path.abspath( path ) ), bu_store.LOCK_NAME ),
                              exclusive=True )
    try:
        carried = read_old( path, logger )
        new_path = path + NEW_SUFFIX
        for leftover in ( new_path, new_path + '-wal', new_path + '-shm' ):
            if os.path.exists( leftover ):
                os.remove( leftover )
        summary = _build( new_path, carried, seed_folders, hasher, logger, clock )
        if os.path.exists( path ):
            keep_old( path, path + OLD_SUFFIX )
            for side in ( path + '-wal', path + '-shm' ):
                if os.path.exists( side ):
                    os.remove( side )
        os.replace( new_path, path )
        if logger is not None:
            logger.info( "catalog rebuilt: %(items)d items, %(locations)d paths, %(outputs)d outputs; "
                         "carried %(carried_pins)d pins, %(carried_runs)d runs, %(carried_missing)d items "
                         "not on disk"%summary )
        return summary
    finally:
        lock.release()


def keep_old( path, target ):
    """
    The old file, whole, as target: through SQLite's backup API, so what
    is still in its write-ahead log (an engine that was killed never
    checkpointed) is in the copy. A file SQLite cannot read is copied as
    it is, with its log beside it.
    """
    for leftover in ( target, target + '-wal', target + '-shm' ):
        if os.path.exists( leftover ):
            os.remove( leftover )
    try:
        source = sqlite3.connect( 'file:%s?mode=ro'%( path, ), uri=True )
        try:
            copy = sqlite3.connect( target )
            try:
                source.backup( copy )
            finally:
                copy.close()
        finally:
            source.close()
        return
    except sqlite3.DatabaseError:
        pass
    shutil.copy2( path, target )
    if os.path.exists( path + '-wal' ):
        shutil.copy2( path + '-wal', target + '-wal' )


def _build( new_path, carried, seed_folders, hasher, logger, clock ):
    store = bu_store.Store( new_path, clock=clock, logger=logger, lock_path=new_path + '.lock' )
    try:
        index = bu_index.Index( store )
        folders = ( carried.folders if carried.readable
                    else [ ( folder, True ) for folder in ( bu_catalog.default_seed_folders()
                                                             if seed_folders is None else seed_folders ) ] )
        for folder, type_from_folder in folders:
            if os.path.isdir( folder ):
                try:
                    index.add_folder( folder, type_from_folder )
                except bu_store.CatalogError:
                    continue
            elif carried.readable:
                # An unreachable folder stays watched; its items come over below.
                with store.write() as db:
                    db.execute( 'INSERT OR IGNORE INTO folders ( path, type_from_folder, added_at ) VALUES ( ?, ?, ? )',
                                ( bu_index.norm( folder ), int( type_from_folder ), clock() ) )
        scanner = bu_scan.Scanner( index, hasher=hasher, settle=0.0, interval=0 )
        scanner.pass_now()
        for item in carried.items.values():
            for location in carried.locations.get( item['hash'], () ):
                _carry_known_location( index, location, item['hash'] )
        while scanner.pending_count():
            before = scanner.pending_count()
            scanner.hash_ready()
            if scanner.pending_count() >= before:
                break
        scanner.pass_now()   # outputs of items the hashing added
        summary = _carry( store, index, carried, clock )
        rows = lambda sql: store.read( sql )[0][0]
        summary.update( items=rows( 'SELECT COUNT(*) FROM items' ),
                        locations=rows( 'SELECT COUNT(*) FROM locations' ),
                        outputs=rows( 'SELECT COUNT(*) FROM outputs WHERE present = 1' ),
                        old_readable=carried.readable )
        return summary
    finally:
        store.close()
        lock_file = new_path + '.lock'
        if os.path.exists( lock_file ):
            os.remove( lock_file )


def _carry_known_location( index, location, digest ):
    """A path the old file knew, unchanged on disk: its hash is known, no re-read."""
    try:
        file_stat = bu_index.Stat.of( os.stat( location['path'] ) )
    except OSError:
        return
    if ( file_stat.size, file_stat.mtime_ns ) != ( location['size'], location['mtime_ns'] ):
        return
    if index.location_hash( location['path'] ) is None:
        folder = index.folder_for( location['path'] )
        index.record_hash( location['path'], file_stat, digest, folder['id'] if folder else None )


def _carry( store, index, carried, clock ):
    pins = runs = missing = 0
    folders = index.folders()
    with store.write() as db:
        for digest, item in carried.items.items():
            exists = db.execute( 'SELECT 1 FROM items WHERE hash = ?', ( digest, ) ).fetchone() is not None
            if not exists:
                if not ( item['type_pin'] or any( run['hash'] == digest for run in carried.runs )
                         or carried.locations.get( digest ) ):
                    continue
                # Not on disk now: kept, with its paths missing or offline.
                db.execute( 'INSERT INTO items ( hash, kind, size, name, added_at, status ) VALUES ( ?, ?, ?, ?, ?, ? )',
                            ( digest, item['kind'], item['size'], item['name'], item['added_at'], bu_index.NEW ) )
                for location in carried.locations.get( digest, () ):
                    folder = index.folder_for( location['path'], folders )
                    reachable = folder is None or os.path.isdir( folder['path'] )
                    state = bu_index.MISSING if reachable else bu_index.OFFLINE
                    db.execute( 'INSERT OR IGNORE INTO locations ( hash, path, folder_id, size, mtime_ns, dev, ino,'
                                ' state, added_at, seen_at ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )',
                                ( digest, location['path'], folder['id'] if folder else None, location['size'],
                                  location['mtime_ns'], location['dev'], location['ino'], state,
                                  location['added_at'], location['seen_at'] ) )
                missing += 1
            else:
                db.execute( 'UPDATE items SET added_at = ? WHERE hash = ?', ( item['added_at'], digest ) )
            if item['type_pin']:
                db.execute( 'UPDATE items SET type_pin = ? WHERE hash = ?', ( item['type_pin'], digest ) )
                pins += 1
        for run in carried.runs:
            if db.execute( 'SELECT 1 FROM items WHERE hash = ?', ( run['hash'], ) ).fetchone() is None:
                continue
            db.execute( 'INSERT INTO runs ( %s ) VALUES ( %s )'%( ', '.join( _RUN_COLUMNS ),
                                                                 ', '.join( '?' for _ in _RUN_COLUMNS ) ),
                        tuple( run[column] for column in _RUN_COLUMNS ) )
            runs += 1
    _settle_from_runs( store, index, carried )
    return { 'carried_pins': pins, 'carried_runs': runs, 'carried_missing': missing }


def _settle_from_runs( store, index, carried ):
    """Status and last type from the newest run that set them, as History did."""
    import betautils_run_record as bu_run_record
    for run in carried.runs:   # oldest first, so the newest wins
        if not index.has_item( run['hash'] ):
            continue
        record = bu_run_record.RunRecord( run['path'] )
        record.outcome = run['outcome']
        record.reason = run['reason']
        record.preview = bool( run['preview'] )
        status = bu_history.status_after( record )
        if status is not None:
            index.set_status( run['hash'], status )
        if run['profile'] and not run['preview']:
            index.set_last_profile( run['hash'], run['profile'], run['rule'], bu_index.loads_params( run['rule_params'] ) )
