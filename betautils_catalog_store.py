"""
betautils_catalog_store.py - the catalog's database file: where it is,
how it is opened, and how its schema moves forward.

The catalog is one SQLite file in the data folder
(kind_dir('data')/catalog.sqlite). It holds what exists nowhere else
(type pins, run history, later feedback) next to what it indexes from
disk (items, locations, outputs), so it is durable data, not cache.
Backing it up is backing up the library.

SCHEMA VERSIONS. Each step in MIGRATIONS is SQL that takes the schema
from version N-1 to N. The version is SQLite's own `PRAGMA user_version`,
an integer in the file header kept for the application. A step runs in
one transaction that also sets the version and adds its row to
schema_migrations (version, applied_at, app version, sha256 of the step),
so a step is applied whole or not at all. The checksums make an edited
shipped step visible: a step is never changed once shipped, a new one is
added instead (tests pin the checksums). A file newer than this code is
refused, never opened. Before the first step runs on an existing file,
a copy is taken with SQLite's backup API (catalog.sqlite.v<N>.bak, the
last BACKUPS_KEPT kept).

Why not a migration tool: Flyway and Liquibase need a JVM, Alembic needs
SQLAlchemy, yoyo-migrations is one more package; each is a dependency
for one embedded single-user file, and serve mode starts with the
standard library only. This is their versioned-script model in a page.

CONCURRENCY. The engine and the CLI open the catalog at the same time;
WAL lets readers and one writer work together, and writers wait up to
BUSY_TIMEOUT_SECONDS for each other. Each opener holds a shared lock on
catalog.lock beside the file for as long as it has the catalog open; the
rebuild tool takes it exclusive, so it never swaps the file under a
running engine or CLI. The kernel drops the lock when a process ends.

Stdlib only.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time

if os.name == 'nt':
    import ctypes
    import ctypes.wintypes
    import msvcrt
else:
    import fcntl

FILE_NAME = 'catalog.sqlite'
LOCK_NAME = 'catalog.lock'
BACKUP_SUFFIX = '.bak'
BACKUPS_KEPT = 3
BUSY_TIMEOUT_SECONDS = 5.0

# Each entry: ( version, name, sql ). Never edit a shipped entry; append.
MIGRATIONS = (
    ( 1, 'items_locations_folders_outputs', """
CREATE TABLE folders (
    id               INTEGER PRIMARY KEY,
    path             TEXT    NOT NULL UNIQUE,
    type_from_folder INTEGER NOT NULL DEFAULT 0,
    added_at         REAL    NOT NULL
);
CREATE TABLE items (
    hash             TEXT    PRIMARY KEY,
    kind             TEXT    NOT NULL DEFAULT 'video',
    size             INTEGER NOT NULL,
    name             TEXT    NOT NULL,
    added_at         REAL    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'new',
    type_pin         TEXT,
    last_profile     TEXT,
    last_rule        TEXT,
    last_rule_params TEXT
);
CREATE TABLE locations (
    id        INTEGER PRIMARY KEY,
    hash      TEXT    NOT NULL REFERENCES items( hash ) ON DELETE CASCADE,
    path      TEXT    NOT NULL UNIQUE,
    folder_id INTEGER REFERENCES folders( id ) ON DELETE SET NULL,
    size      INTEGER NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    dev       INTEGER NOT NULL,
    ino       INTEGER NOT NULL,
    state     TEXT    NOT NULL DEFAULT 'present',
    added_at  REAL    NOT NULL,
    seen_at   REAL    NOT NULL
);
CREATE INDEX locations_hash  ON locations( hash );
CREATE INDEX locations_inode ON locations( dev, ino );
CREATE TABLE outputs (
    id            INTEGER PRIMARY KEY,
    hash          TEXT    NOT NULL REFERENCES items( hash ) ON DELETE CASCADE,
    path          TEXT    NOT NULL UNIQUE,
    detection_key TEXT    NOT NULL,
    censor_key    TEXT    NOT NULL,
    encode_key    TEXT    NOT NULL,
    present       INTEGER NOT NULL DEFAULT 1,
    seen_at       REAL    NOT NULL
);
CREATE INDEX outputs_hash ON outputs( hash );
""" ),
    ( 2, 'run_history', """
CREATE TABLE runs (
    id            INTEGER PRIMARY KEY,
    hash          TEXT    NOT NULL REFERENCES items( hash ) ON DELETE CASCADE,
    program       TEXT,
    job           TEXT,
    path          TEXT    NOT NULL,
    started_at    REAL    NOT NULL,
    ended_at      REAL    NOT NULL,
    outcome       TEXT    NOT NULL,
    reason        TEXT,
    output        TEXT,
    detection_key TEXT,
    censor_key    TEXT,
    encode_key    TEXT,
    profile       TEXT,
    rule          TEXT,
    rule_params   TEXT,
    sample_fps    REAL,
    preview       INTEGER NOT NULL DEFAULT 0,
    error_type    TEXT,
    stage         TEXT
);
CREATE INDEX runs_hash ON runs( hash, ended_at );
""" ),
)

LATEST_VERSION = MIGRATIONS[-1][0]


class CatalogError( Exception ):
    """A catalog failure with a protocol error code and its params."""

    def __init__( self, code, **params ):
        super().__init__( code )
        self.code = code
        self.params = params


def default_path():
    import betautils_paths as bu_paths
    return os.path.join( bu_paths.kind_dir( 'data' ), FILE_NAME )


def step_checksum( sql ):
    return hashlib.sha256( sql.encode( 'utf-8' ) ).hexdigest()


def _app_version():
    try:
        import betautils_version
        return betautils_version.get_version()
    except Exception:   # a version lookup never stops the catalog opening
        return None


# ---------------------------------------------------------------------------
# The lock beside the file
# ---------------------------------------------------------------------------

if os.name == 'nt':
    _LOCKFILE_FAIL_IMMEDIATELY = 0x1
    _LOCKFILE_EXCLUSIVE_LOCK = 0x2

    class _Overlapped( ctypes.Structure ):
        _fields_ = [ ( 'Internal', ctypes.c_void_p ), ( 'InternalHigh', ctypes.c_void_p ),
                     ( 'Offset', ctypes.wintypes.DWORD ), ( 'OffsetHigh', ctypes.wintypes.DWORD ),
                     ( 'hEvent', ctypes.wintypes.HANDLE ) ]

    _kernel32 = ctypes.WinDLL( 'kernel32', use_last_error=True )
    _kernel32.LockFileEx.argtypes = [ ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD,
                                      ctypes.wintypes.DWORD, ctypes.wintypes.DWORD,
                                      ctypes.wintypes.DWORD, ctypes.POINTER( _Overlapped ) ]
    _kernel32.LockFileEx.restype = ctypes.wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [ ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD,
                                        ctypes.wintypes.DWORD, ctypes.wintypes.DWORD,
                                        ctypes.POINTER( _Overlapped ) ]
    _kernel32.UnlockFileEx.restype = ctypes.wintypes.BOOL


def _try_lock( fd, exclusive ):
    if os.name == 'nt':
        flags = _LOCKFILE_FAIL_IMMEDIATELY | ( _LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0 )
        handle = msvcrt.get_osfhandle( fd )
        return bool( _kernel32.LockFileEx( handle, flags, 0, 1, 0, ctypes.byref( _Overlapped() ) ) )
    try:
        fcntl.flock( fd, ( fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH ) | fcntl.LOCK_NB )
    except ( BlockingIOError, PermissionError ):
        return False
    return True


def _unlock( fd ):
    if os.name == 'nt':
        _kernel32.UnlockFileEx( msvcrt.get_osfhandle( fd ), 0, 1, 0, ctypes.byref( _Overlapped() ) )
    else:
        fcntl.flock( fd, fcntl.LOCK_UN )


class FileLock:
    """A shared or exclusive lock on catalog.lock, held until release()."""

    def __init__( self, path, exclusive ):
        self.path = path
        self.exclusive = exclusive
        os.makedirs( os.path.dirname( os.path.abspath( path ) ), exist_ok=True )
        flags = os.O_RDWR | os.O_CREAT | getattr( os, 'O_BINARY', 0 ) | getattr( os, 'O_CLOEXEC', 0 )
        self._fd = os.open( path, flags, 0o600 )
        if not _try_lock( self._fd, exclusive ):
            os.close( self._fd )
            self._fd = None
            raise CatalogError( 'catalog.unavailable', reason='in_use' )

    def release( self ):
        if self._fd is not None:
            try:
                _unlock( self._fd )
            finally:
                os.close( self._fd )
                self._fd = None


# ---------------------------------------------------------------------------
# Opening and migrating
# ---------------------------------------------------------------------------

def _connect( path ):
    connection = sqlite3.connect( path, timeout=BUSY_TIMEOUT_SECONDS,
                                  isolation_level=None, check_same_thread=False )
    connection.row_factory = sqlite3.Row
    connection.execute( 'PRAGMA foreign_keys = ON' )
    return connection


def schema_version( connection ):
    return connection.execute( 'PRAGMA user_version' ).fetchone()[0]


def _backup( connection, path, version, logger=None ):
    """Copy the file as it is before a migration; keep the newest few."""
    target = '%s.v%d%s'%( path, version, BACKUP_SUFFIX )
    copy = sqlite3.connect( target )
    try:
        connection.backup( copy )
    finally:
        copy.close()
    folder, base = os.path.split( path )
    prefix = base + '.v'
    olds = sorted( ( name for name in os.listdir( folder or '.' )
                     if name.startswith( prefix ) and name.endswith( BACKUP_SUFFIX ) ),
                   key=lambda name: os.path.getmtime( os.path.join( folder, name ) ) )
    for name in olds[:-BACKUPS_KEPT]:
        try:
            os.remove( os.path.join( folder, name ) )
        except OSError:
            pass
    if logger is not None:
        logger.info( "catalog: copied schema version %d to %s before migrating"%( version, target ) )
    return target


def _apply( connection, version, name, sql, clock ):
    """One step, whole or not at all."""
    connection.execute( 'BEGIN IMMEDIATE' )
    try:
        if version == 1:
            connection.execute( """CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL,
                app_version TEXT, checksum TEXT NOT NULL )""" )
        for statement in _statements( sql ):
            connection.execute( statement )
        connection.execute( 'INSERT INTO schema_migrations VALUES ( ?, ?, ?, ?, ? )',
                            ( version, name, clock(), _app_version(), step_checksum( sql ) ) )
        connection.execute( 'PRAGMA user_version = %d'%( version ) )
        connection.execute( 'COMMIT' )
    except BaseException:
        connection.execute( 'ROLLBACK' )
        raise


def _statements( sql ):
    """The step's statements, one by one (executescript would commit)."""
    pending = ''
    for line in sql.splitlines( keepends=True ):
        pending += line
        if sqlite3.complete_statement( pending ):
            if pending.strip():
                yield pending.strip()
            pending = ''
    if pending.strip():
        raise ValueError( 'incomplete statement in a migration step' )


def migrate( connection, path=None, migrations=MIGRATIONS, clock=time.time, logger=None ):
    """
    Bring the open database up to the last step in `migrations`.

    Returns the list of versions applied. Raises CatalogError
    catalog.too_new when the file is newer than the code.
    """
    latest = migrations[-1][0] if migrations else 0
    current = schema_version( connection )
    if current > latest:
        raise CatalogError( 'catalog.too_new', version=current, supported=latest )
    applied = []
    pending = [ step for step in migrations if step[0] > current ]
    if pending and current > 0 and path is not None:
        _backup( connection, path, current, logger )
    for version, name, sql in pending:
        _apply( connection, version, name, sql, clock )
        applied.append( version )
        if logger is not None:
            logger.info( "catalog: schema now at version %d (%s)"%( version, name ) )
    _check_checksums( connection, migrations, logger )
    return applied


def _check_checksums( connection, migrations, logger ):
    """Warn (never refuse) when a step applied here differs from the code's."""
    try:
        rows = connection.execute( 'SELECT version, checksum FROM schema_migrations' ).fetchall()
    except sqlite3.DatabaseError:
        return
    known = { version: step_checksum( sql ) for version, _name, sql in migrations }
    for version, checksum in rows:
        if version in known and known[version] != checksum and logger is not None:
            logger.warning( "catalog: schema step %d differs from the one applied to this file"%(
                version ) )


class Store:
    """
    The open catalog database: one connection shared by the engine's
    threads behind a lock, plus the shared file lock.

    Use `with store.write() as db:` for a transaction and
    `store.read()` for a query. Tables belong to the modules that use
    them; this class owns only the file.
    """

    def __init__( self, path=None, clock=time.time, logger=None, exclusive=False, lock_path=None ):
        self.path = path or default_path()
        self.clock = clock
        self.logger = logger
        os.makedirs( os.path.dirname( os.path.abspath( self.path ) ), exist_ok=True )
        # lock_path: another lock file, for a file being built beside the
        # catalog (the rebuild, which holds catalog.lock itself).
        self._lock = FileLock( lock_path or os.path.join( os.path.dirname( os.path.abspath( self.path ) ),
                                                          LOCK_NAME ), exclusive )
        self._mutex = threading.RLock()
        try:
            self._db = _connect( self.path )
            try:
                # Refused before anything writes: switching to WAL alone
                # rewrites the header of a file this code must not touch.
                current = schema_version( self._db )
                if current > LATEST_VERSION:
                    raise CatalogError( 'catalog.too_new', version=current, supported=LATEST_VERSION )
                mode = self._db.execute( 'PRAGMA journal_mode = WAL' ).fetchone()[0]
                if mode.lower() != 'wal' and logger is not None:
                    logger.warning( "catalog: journal mode is %s, not wal"%( mode ) )
                applied = migrate( self._db, self.path, clock=clock, logger=logger )
                # True for a file this open made: the caller seeds it.
                self.created = bool( applied ) and applied[0] == 1
            except BaseException:
                self._db.close()
                raise
        except sqlite3.DatabaseError as err:
            self._lock.release()
            raise CatalogError( 'catalog.unavailable', reason=_reason( err ) ) from err
        except BaseException:
            self._lock.release()
            raise

    def read( self, sql, args=() ):
        with self._mutex:
            return self._db.execute( sql, args ).fetchall()

    def write( self ):
        return _Transaction( self )

    def data_version( self ):
        """Changes when another connection commits (PRAGMA data_version)."""
        with self._mutex:
            return self._db.execute( 'PRAGMA data_version' ).fetchone()[0]

    def close( self ):
        with self._mutex:
            if self._db is not None:
                try:
                    self._db.close()
                finally:
                    self._db = None
                    self._lock.release()


class _Transaction:
    def __init__( self, store ):
        self._store = store

    def __enter__( self ):
        self._store._mutex.acquire()
        try:
            self._store._db.execute( 'BEGIN IMMEDIATE' )
        except BaseException:
            self._store._mutex.release()
            raise
        return self._store._db

    def __exit__( self, kind, value, traceback ):
        try:
            self._store._db.execute( 'COMMIT' if kind is None else 'ROLLBACK' )
        finally:
            self._store._mutex.release()
        return False


def _reason( err ):
    text = str( err ).lower()
    if 'not a database' in text or 'malformed' in text:
        return 'corrupt'
    if 'locked' in text or 'busy' in text:
        return 'busy'
    if 'readonly' in text or 'read-only' in text or 'unable to open' in text:
        return 'not_writable'
    return 'error'


def dumps_params( params ):
    """JSON text for a params column, keys sorted, or None for none."""
    return json.dumps( params, sort_keys=True ) if params else None
