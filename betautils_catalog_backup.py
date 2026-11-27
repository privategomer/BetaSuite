"""
betautils_catalog_backup.py - a copy of the catalog file, and putting
one back.

The catalog holds what exists nowhere else: watched folders, type pins,
run history and feedback. A backup is that file, whole, so it carries
all of them to a reinstall or a new machine. There, paths the old
machine knew read missing until the watched folders are set again; an
item is keyed by its content hash, so once a scan finds the same file,
its pins, history and feedback are back with it.

BACKUP copies through SQLite's backup API under the catalog's shared
lock, so it is safe while the app or a command-line run has the catalog
open, and what is still in the write-ahead log is in the copy. It is
written beside its target first and renamed into place, and checked
before it is kept.

RESTORE refuses while anything has the catalog open (it needs the lock
exclusive, as the rebuild does). The file to restore is checked first:
SQLite must read it whole (quick_check), it must be a catalog (its
schema steps and tables), and not from a newer version than this code.
The current file is kept as catalog.sqlite.pre-restore.bak, then the
copy is put in place and opened once, so a backup from an older version
takes the schema steps since (with their usual copy beforehand).

Stdlib only.
"""

import os
import sqlite3
import time

import betautils_catalog_store as bu_store

PARTIAL_SUFFIX = '.partial'
RESTORE_SUFFIX = '.restore'
OLD_SUFFIX = '.pre-restore.bak'
BACKUP_FOLDER = 'backups'
FILE_SUFFIX = '.sqlite'
_TABLES = ( 'schema_migrations', 'folders', 'items', 'locations', 'outputs' )


class BackupError( Exception ):
    """A backup or restore refused, with a code and its params (the tool words them)."""

    def __init__( self, code, **params ):
        super().__init__( code )
        self.code = code
        self.params = params


def default_folder():
    import betautils_paths as bu_paths
    return os.path.join( bu_paths.kind_dir( 'data' ), BACKUP_FOLDER )


def default_name( clock=time.time ):
    return 'catalog-%s.sqlite'%( time.strftime( '%Y%m%d-%H%M%S', time.localtime( clock() ) ), )


def _lock_path( path ):
    return os.path.join( os.path.dirname( os.path.abspath( path ) ), bu_store.LOCK_NAME )


def _remove( *paths ):
    for path in paths:
        if os.path.exists( path ):
            os.remove( path )


def _same( first, second ):
    try:
        return os.path.samefile( first, second )
    except OSError:
        return os.path.normcase( os.path.abspath( first ) ) == os.path.normcase( os.path.abspath( second ) )


def describe( path ):
    """
    What a catalog file holds: { version, items, runs, feedback }.
    Raises BackupError backup.not_a_catalog for a file SQLite cannot
    read whole or that is not a catalog, CatalogError catalog.too_new
    for one from a newer version.
    """
    if not os.path.isfile( path ):
        raise BackupError( 'backup.no_such_file', path=path )
    try:
        db = sqlite3.connect( 'file:%s?mode=ro'%( path, ), uri=True )
        try:
            if db.execute( 'PRAGMA quick_check' ).fetchone()[0] != 'ok':
                raise BackupError( 'backup.not_a_catalog', path=path, reason='damaged' )
            version = db.execute( 'PRAGMA user_version' ).fetchone()[0]
            if version > bu_store.LATEST_VERSION:
                raise bu_store.CatalogError( 'catalog.too_new', version=version, supported=bu_store.LATEST_VERSION )
            tables = { row[0] for row in db.execute( "SELECT name FROM sqlite_master WHERE type = 'table'" ) }
            if version < 1 or not set( _TABLES ) <= tables:
                raise BackupError( 'backup.not_a_catalog', path=path, reason='not_a_catalog' )
            count = lambda table: ( db.execute( 'SELECT COUNT(*) FROM %s'%( table, ) ).fetchone()[0]
                                    if table in tables else 0 )
            return { 'version': version, 'items': count( 'items' ), 'runs': count( 'runs' ),
                     'feedback': count( 'feedback' ) }
        finally:
            db.close()
    except sqlite3.DatabaseError as err:
        raise BackupError( 'backup.not_a_catalog', path=path, reason='damaged' ) from err


def backup( target, path=None, overwrite=False, logger=None ):
    """
    Copy the catalog to target: a .sqlite file, or a folder (made if
    need be) to name one in.
    Returns { path, version, items, runs, feedback }. Safe beside a
    running app.
    """
    path = path or bu_store.default_path()
    if not os.path.isfile( path ):
        raise BackupError( 'backup.no_catalog', path=path )
    # A folder (made if need be) unless it names a .sqlite file.
    if os.path.isdir( target ) or not target.lower().endswith( FILE_SUFFIX ):
        target = os.path.join( target, default_name() )
    target = os.path.abspath( target )
    if _same( target, path ):
        raise BackupError( 'backup.same_file', path=target )
    if os.path.exists( target ) and not overwrite:
        raise BackupError( 'backup.exists', path=target )
    os.makedirs( os.path.dirname( target ), exist_ok=True )
    partial = target + PARTIAL_SUFFIX
    lock = bu_store.FileLock( _lock_path( path ), exclusive=False )
    try:
        _remove( partial )
        try:
            source = sqlite3.connect( 'file:%s?mode=ro'%( path, ), uri=True )
            try:
                copy = sqlite3.connect( partial )
                try:
                    source.backup( copy )
                    # One file, nothing beside it: the copy is what is kept.
                    copy.execute( 'PRAGMA journal_mode = DELETE' )
                finally:
                    copy.close()
            finally:
                source.close()
        except sqlite3.DatabaseError as err:
            _remove( partial )
            raise BackupError( 'backup.not_a_catalog', path=path, reason='damaged' ) from err
    finally:
        lock.release()
    try:
        summary = describe( partial )
    except Exception:
        _remove( partial )
        raise
    os.replace( partial, target )
    summary['path'] = target
    if logger is not None:
        logger.info( "catalog backed up to %s: schema %d, %d items, %d runs, %d feedback marks"%(
            target, summary['version'], summary['items'], summary['runs'], summary['feedback'] ) )
    return summary


def restore( source, path=None, logger=None, clock=time.time ):
    """
    Put the backup at source in place of the catalog. Returns
    { restored: describe(source), version, kept }, kept being where the
    replaced file went (None when there was none). Raises CatalogError
    catalog.unavailable {in_use} while anything has the catalog open.
    """
    path = path or bu_store.default_path()
    source = os.path.abspath( source )
    if os.path.exists( path ) and _same( source, path ):
        raise BackupError( 'backup.same_file', path=source )
    os.makedirs( os.path.dirname( os.path.abspath( path ) ), exist_ok=True )
    lock = bu_store.FileLock( _lock_path( path ), exclusive=True )
    kept = None
    try:
        restored = describe( source )
        staged = path + RESTORE_SUFFIX
        _remove( staged, staged + '-wal', staged + '-shm' )
        try:
            origin = sqlite3.connect( 'file:%s?mode=ro'%( source, ), uri=True )
            try:
                copy = sqlite3.connect( staged )
                try:
                    origin.backup( copy )
                finally:
                    copy.close()
            finally:
                origin.close()
        except sqlite3.DatabaseError as err:
            _remove( staged )
            raise BackupError( 'backup.not_a_catalog', path=source, reason='damaged' ) from err
        if os.path.exists( path ):
            import betautils_catalog_rebuild as bu_rebuild
            kept = path + OLD_SUFFIX
            bu_rebuild.keep_old( path, kept )
        _remove( path + '-wal', path + '-shm' )
        os.replace( staged, path )
    finally:
        lock.release()
    # Opened once, so an older backup takes the schema steps since now.
    store = bu_store.Store( path, clock=clock, logger=logger )
    try:
        version = bu_store.schema_version( store._db )
    finally:
        store.close()
    if logger is not None:
        logger.info( "catalog restored from %s: %d items, %d runs, %d feedback marks; schema %d, now %d%s"%(
            source, restored['items'], restored['runs'], restored['feedback'], restored['version'], version,
            '; the replaced file is %s'%( kept, ) if kept else '' ) )
    return { 'restored': restored, 'version': version, 'kept': kept }
