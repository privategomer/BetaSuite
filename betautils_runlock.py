"""
betautils_runlock.py - one run at a time per library, CLI and app alike.

Two runs over one library write the same outputs, caches and file-hash
memo, and fight over one GPU. betatv, betastare and the engine's jobs
take this lock before they start work; a second one is refused with the
holder named, as the protocol error run.locked {program, pid,
started_at} or as a message on the command line.

HOW. The lock is the operating system's own lock on one file
(betaconst.run_lock_path, in the state folder, or output/ of a 1.x tree):
fcntl.flock on POSIX, msvcrt.locking on Windows. The kernel drops it the
moment the holding process ends, however it ends, so a run killed by the
out-of-memory killer or a power cut never leaves the library locked, and
no pid has to be checked (pids are reused). The descriptor is not
inherited, so an ffmpeg child that outlives its parent does not keep it.
The lock file itself is never deleted: removing it while another process
waits on it would let two runs each lock a different file.

Who holds it is in a small JSON file beside it (HOLDER_SUFFIX), written
after the lock is taken and removed on release. One found when the lock
is free belongs to a run that ended without releasing: the new run says
so (`recovered`). It names the program, pid and start time only, never a
path or a file name.

Stdlib only, so the Windows CI job runs it without the engine's packages.
"""

import json
import os
import time

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

# Exit code of betatv and betastare when another run holds the lock.
EXIT_LOCKED = 3

ERROR_CODE = 'run.locked'
HOLDER_FIELDS = ( 'program', 'pid', 'started_at' )
HOLDER_SUFFIX = '.json'

# A refused run reads the holder this many times, this far apart, since
# the holder writes its record just after taking the lock.
_HOLDER_READS = 10
_HOLDER_READ_GAP_SECONDS = 0.05


def default_path():
    import betaconst
    return betaconst.run_lock_path


def holder_path( lock_path ):
    return lock_path + HOLDER_SUFFIX


def _try_lock( fd ):
    """True if this descriptor now holds the lock, False if another does."""
    try:
        if os.name == 'nt':
            os.lseek( fd, 0, os.SEEK_SET )
            msvcrt.locking( fd, msvcrt.LK_NBLCK, 1 )
        else:
            fcntl.flock( fd, fcntl.LOCK_EX | fcntl.LOCK_NB )
    except ( BlockingIOError, PermissionError ):
        return False
    except OSError as err:
        # msvcrt reports a held lock as EACCES or EDEADLOCK.
        if os.name == 'nt' and err.errno in ( 13, 36 ):
            return False
        raise
    return True


def _unlock( fd ):
    if os.name == 'nt':
        os.lseek( fd, 0, os.SEEK_SET )
        msvcrt.locking( fd, msvcrt.LK_UNLCK, 1 )
    else:
        fcntl.flock( fd, fcntl.LOCK_UN )


def _empty_holder():
    return { field: None for field in HOLDER_FIELDS }


def read_holder( lock_path ):
    """
    The holder record beside a lock, or None when there is none.

    Fields that are missing or of the wrong type come back as None, so
    a damaged record still names what it can.
    """
    try:
        with open( holder_path( lock_path ), 'r', encoding='utf-8' ) as handle:
            record = json.load( handle )
    except FileNotFoundError:
        return None
    except ( OSError, ValueError ):
        return _empty_holder()
    holder = _empty_holder()
    if isinstance( record, dict ):
        if isinstance( record.get( 'program' ), str ):
            holder['program'] = record['program']
        if isinstance( record.get( 'pid' ), int ) and not isinstance( record.get( 'pid' ), bool ):
            holder['pid'] = record['pid']
        started = record.get( 'started_at' )
        if isinstance( started, ( int, float ) ) and not isinstance( started, bool ):
            holder['started_at'] = started
    return holder


def _read_holder_of_live_lock( lock_path ):
    """The record of a lock someone else holds: retried briefly, since it is written just after locking."""
    for attempt in range( _HOLDER_READS ):
        holder = read_holder( lock_path )
        if holder is not None and holder['pid'] is not None:
            return holder
        if attempt + 1 < _HOLDER_READS:
            time.sleep( _HOLDER_READ_GAP_SECONDS )
    return holder or _empty_holder()


def describe( holder ):
    """'betatv (pid 4242, started 2001-09-17 20:14:41)', with 'unknown' for what the record lacks."""
    holder = holder or _empty_holder()
    started = holder.get( 'started_at' )
    when = ( time.strftime( '%Y-%m-%d %H:%M:%S', time.localtime( started ) )
             if isinstance( started, ( int, float ) ) else 'an unknown time' )
    return "%s (pid %s, started %s)"%( holder.get( 'program' ) or 'an unknown program',
                                      holder.get( 'pid' ) if holder.get( 'pid' ) is not None else 'unknown',
                                      when )


class RunLocked( Exception ):
    """Another run holds the lock. `holder` is its record (fields may be None)."""

    def __init__( self, lock_path, holder ):
        self.lock_path = lock_path
        self.holder = dict( holder )
        super().__init__( ERROR_CODE )

    def error( self ):
        """The protocol error object, {code, params}."""
        import betautils_protocol as bu_proto
        return bu_proto.error( ERROR_CODE, **{ field: self.holder.get( field ) for field in HOLDER_FIELDS } )

    def message( self ):
        """The command line's sentence."""
        return ( "another BetaSuite run is working on this library: %s. Only one run at a time can use "
                 "it (lock %s). Wait for that run to finish or stop it, then run again"%(
                     describe( self.holder ), self.lock_path ) )


class RunLock:
    """A held lock. release() (or leaving a with block) gives it up."""

    def __init__( self, path, fd, record, recovered ):
        self.path = path
        self.record = record
        self.recovered = recovered
        self._fd = fd

    @property
    def held( self ):
        return self._fd is not None

    def release( self ):
        """Remove this run's holder record and unlock. Safe to call twice."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            on_disk = read_holder( self.path )
            if on_disk is not None and on_disk.get( 'pid' ) == self.record['pid']:
                try:
                    os.remove( holder_path( self.path ) )
                except FileNotFoundError:
                    pass
        finally:
            try:
                _unlock( fd )
            finally:
                os.close( fd )

    def __enter__( self ):
        return self

    def __exit__( self, *exc ):
        self.release()
        return False


def _write_holder( lock_path, record ):
    target = holder_path( lock_path )
    temporary = '%s.%d.tmp'%( target, os.getpid() )
    with open( temporary, 'w', encoding='utf-8' ) as handle:
        json.dump( record, handle )
    for attempt in range( 5 ):
        try:
            os.replace( temporary, target )
            return
        except PermissionError:
            # Windows: a reader has the old record open for a moment.
            if attempt == 4:
                raise
            time.sleep( 0.05 )


def acquire( program, path=None, clock=time.time, logger=None ):
    """
    Take the run lock or raise RunLocked naming who has it.

    Args:
        program: Who is asking: 'betatv', 'betastare', 'engine'.
        path: The lock file; default betaconst.run_lock_path.
        clock: Epoch seconds, for the record.
        logger: Told when a holder record from a run that ended without
            releasing was found and replaced.

    Returns:
        A RunLock. Its `recovered` is the replaced record, or None.

    Raises:
        RunLocked: another process holds it.
        OSError: the lock file cannot be created or opened.
    """
    path = default_path() if path is None else path
    folder = os.path.dirname( path )
    if folder:
        os.makedirs( folder, exist_ok=True )
    fd = os.open( path, os.O_RDWR | os.O_CREAT, 0o600 )
    try:
        if not _try_lock( fd ):
            raise RunLocked( path, _read_holder_of_live_lock( path ) )
        recovered = read_holder( path )
        record = { 'program': program, 'pid': os.getpid(), 'started_at': clock() }
        _write_holder( path, record )
    except BaseException:
        os.close( fd )
        raise
    if recovered is not None and logger is not None:
        logger.warning( "the run lock was left by %s, which ended without releasing it; taken over"%(
            describe( recovered ), ) )
    return RunLock( path, fd, record, recovered )


def probe( path=None ):
    """
    Who holds the lock right now, without taking it or creating anything.

    Returns:
        The holder record (fields may be None), or None when the lock is
        free or has never been made.
    """
    path = default_path() if path is None else path
    try:
        # Read-only: probing another tree (a 1.x one being migrated) must not
        # open anything in it for writing. Both locks accept such a descriptor.
        fd = os.open( path, os.O_RDONLY )
    except FileNotFoundError:
        return None
    try:
        if not _try_lock( fd ):
            return _read_holder_of_live_lock( path )
        _unlock( fd )
        return None
    finally:
        os.close( fd )
