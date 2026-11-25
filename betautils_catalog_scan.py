"""
betautils_catalog_scan.py - keeping the catalog in step with the disk.

A pass lists every watched folder and stats what it finds (readdir and
stat only; bytes are read only to hash a file the index cannot place
any other way). It also stats files added on their own, and lists the
output folders to see which outputs exist.

WHEN. One pass when the catalog starts, then every
`catalog_scan_interval_seconds` (0: only when asked). A pass never
starts sooner than PASS_GAP_FACTOR times the last pass's own length
after it ended, so a folder on a slow network share is not listed back
to back. `rescan` asks for one now.

NEW FILES SETTLE FIRST. A file being copied in grows as it goes, so a
file the index does not know is hashed only once its size and mtime
have held for settle_seconds, re-checked with a stat on a short timer
rather than at the next pass. Files given to add_files are hashed at
once (someone chose them), ahead of the rest.

JOBS COME FIRST. Hashing reads whole files, which competes with a
running job for the disk, so it waits while busy() says a run is on:
the engine passes one that is true for its own jobs and for any run
holding the library's run lock (a command-line run).
Listing and stat-ing do not wait. Files whose size matches an item
that went missing are hashed first: they are the likeliest re-finds.

The scanner is a small task queue on one thread, below jobs: a later
kind of background work (a shot-cut pre-scan, a scheduled scan) is one
more task kind here.
"""

import os
import threading
import time

import betautils_catalog_index as bu_index

PASS_GAP_FACTOR = 10
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_SETTLE_SECONDS = 5.0
BUSY_RECHECK_SECONDS = 2.0
EXTERNAL_CHECK_SECONDS = 2.0


def default_hasher( path ):
    import betautils_hash as bu_hash
    return bu_hash.md5_for_file( path, 16 )


def output_fields( name ):
    """The parsed name of a finished censored output, or None for any other file."""
    if name.startswith( '.' ) or not name.endswith( '.mp4' ):
        return None
    import betautils_cache_paths as bu_cache
    fields = bu_cache.parse_video_output_basename( name[:-len( '.mp4' )] )
    if fields is None or '.tmp' in fields['tail']:
        return None
    return fields


def flush_hashes():
    import betautils_hash as bu_hash
    bu_hash.flush_file_hash_memo()


class Scanner:
    """
    One pass at a time over the index. pass_now() and hash_ready() do
    the work synchronously (the tests drive them with a fake clock);
    start() runs them on a thread on the schedule above.
    """

    def __init__( self, index, hasher=None, clock=None, busy=None, on_change=None, logger=None,
                  interval=DEFAULT_INTERVAL_SECONDS, settle=DEFAULT_SETTLE_SECONDS, flush=None ):
        self.index = index
        self.hasher = hasher or default_hasher
        self.flush = flush if flush is not None else ( flush_hashes if hasher is None else ( lambda: None ) )
        self.clock = clock or index.clock
        self.busy = busy or ( lambda: False )
        self.on_change = on_change or ( lambda items, reason: None )
        self.logger = logger
        self.interval = interval
        self.settle = settle
        # path -> ( Stat, first seen at, folder_id, chosen ). Files waiting
        # to be hashed: settling, or settled and waiting for the disk.
        self._pending = {}
        self._mutex = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._pass_requested = False
        self._last_pass_end = None
        self._last_pass_seconds = 0.0
        self._watched_outputs = {}
        self.passes = 0

    # --- what the catalog shows ---------------------------------------------

    def pending_count( self ):
        with self._mutex:
            return len( self._pending )

    # --- a pass ---------------------------------------------------------------

    def pass_now( self ):
        """List and stat everything once. Returns the items touched."""
        started = time.monotonic()
        touched = set()
        folders = self.index.folders()
        self._watched_outputs = {}
        for folder in folders:
            touched |= self._pass_folder( folder, folders )
        touched |= self._pass_loose()
        touched |= self._pass_outputs()
        self._last_pass_seconds = time.monotonic() - started
        self._last_pass_end = time.monotonic()
        self.passes += 1
        if self.logger is not None:
            self.logger.debug( "catalog: pass over %d folder(s) in %.3f s, %d waiting to hash"%(
                len( folders ), self._last_pass_seconds, self.pending_count() ) )
        self._changed( touched, 'scan' )
        return touched

    def _pass_folder( self, folder, folders ):
        touched = set()
        # A watched folder inside this one has its own pass.
        nested = [ other['path'] for other in folders
                   if other['id'] != folder['id'] and bu_index.is_within( other['path'], folder['path'] ) ]
        try:
            found, unreadable = bu_index.walk_videos( folder['path'], exclude=bu_index.output_roots() + nested )
        except OSError:
            # The folder itself cannot be read: offline, never missing.
            paths = [ row['path'] for row in self.index.locations_in( folder['id'] ) ]
            return self.index.mark( paths, bu_index.OFFLINE )
        for path, file_stat in list( found.items() ):
            fields = output_fields( os.path.basename( path ) )
            if fields is not None:
                # A censored output, wherever it was put: never a source.
                self._watched_outputs[ bu_index.norm( path ) ] = fields
                continue
            touched |= self._observe( path, file_stat, folder['id'], chosen=False )
        gone = []
        for row in self.index.locations_in( folder['id'] ):
            if bu_index.norm( row['path'] ) in { bu_index.norm( path ) for path in found }:
                continue
            if any( bu_index.is_within( row['path'], skipped ) for skipped in unreadable ):
                continue
            if not os.path.lexists( row['path'] ):
                gone.append( row['path'] )
        touched |= self.index.mark( gone, bu_index.MISSING )
        return touched

    def _pass_loose( self ):
        """Files added on their own, outside every watched folder."""
        touched, gone, offline = set(), [], []
        for row in self.index.locations_in( None ):
            try:
                result = os.stat( row['path'] )
            except FileNotFoundError:
                ( gone if os.path.isdir( os.path.dirname( row['path'] ) ) else offline ).append( row['path'] )
                continue
            except OSError:
                offline.append( row['path'] )
                continue
            touched |= self._observe( row['path'], bu_index.Stat.of( result ), None, chosen=True )
        touched |= self.index.mark( gone, bu_index.MISSING )
        touched |= self.index.mark( offline, bu_index.OFFLINE )
        return touched

    def _pass_outputs( self ):
        """Outputs in the output folders, and any the last pass met in a watched one."""
        found = dict( self._watched_outputs )
        for root in bu_index.output_roots():
            for folder, _dirs, names in os.walk( root ):
                for name in names:
                    fields = output_fields( name )
                    if fields is not None:
                        found[ bu_index.norm( os.path.join( folder, name ) ) ] = fields
        return self.index.record_outputs( found )

    def _observe( self, path, file_stat, folder_id, chosen ):
        outcome, touched = self.index.observe( path, file_stat, folder_id )
        key = bu_index.norm( path )
        with self._mutex:
            if outcome != 'needs_hash':
                self._pending.pop( key, None )
                return touched
            waiting = self._pending.get( key )
            if waiting is None or waiting[0] != file_stat:
                self._pending[ key ] = ( file_stat, self.clock(), folder_id, chosen )
        return touched

    # --- files chosen by someone ------------------------------------------------

    def add_files( self, paths ):
        """
        Index chosen files. Returns ( accepted, refused ): refused is a
        list of ( index, reason ) for paths that are not files.
        """
        accepted, refused = [], []
        folders = self.index.folders()
        for position, path in enumerate( paths ):
            try:
                result = os.stat( path )
            except OSError:
                refused.append( ( position, 'not_found' ) )
                continue
            if not os.path.isfile( path ):
                refused.append( ( position, 'not_a_file' ) )
                continue
            if any( bu_index.is_within( path, root ) for root in bu_index.output_roots() ):
                refused.append( ( position, 'output_folder' ) )
                continue
            folder = self.index.folder_for( path, folders )
            self._observe( path, bu_index.Stat.of( result ), folder['id'] if folder else None, chosen=True )
            accepted.append( bu_index.norm( path ) )
        self._wake.set()
        return accepted, refused

    # --- hashing ----------------------------------------------------------------

    def ready( self ):
        """Pending files due for hashing, best first."""
        now = self.clock()
        with self._mutex:
            entries = list( self._pending.items() )
        missing = self.index.missing_sizes() if entries else set()
        due = [ ( path, entry ) for path, entry in entries
                if entry[3] or now - entry[1] >= self.settle ]
        due.sort( key=lambda pair: ( not pair[1][3], pair[1][0].size not in missing, pair[0] ) )
        return due

    def hash_ready( self, limit=None ):
        """
        Hash what is due, unless a job runs. Returns items touched.
        A file that changed since it was seen starts settling again.
        """
        touched = set()
        hashed = 0
        for path, ( seen, _since, folder_id, chosen ) in self.ready():
            if self.busy() or self._stop.is_set():
                break
            if limit is not None and hashed >= limit:
                break
            try:
                before = bu_index.Stat.of( os.stat( path ) )
            except OSError:
                with self._mutex:
                    self._pending.pop( path, None )
                continue
            if not before.same_file( seen ):
                with self._mutex:
                    self._pending[ path ] = ( before, self.clock(), folder_id, chosen )
                continue
            try:
                digest = self.hasher( path )
                after = bu_index.Stat.of( os.stat( path ) )
            except OSError as err:
                if self.logger is not None:
                    self.logger.warning( "catalog: could not hash %s (%s)"%( path, type( err ).__name__ ) )
                with self._mutex:
                    self._pending.pop( path, None )
                continue
            if not after.same_file( before ):
                with self._mutex:
                    self._pending[ path ] = ( after, self.clock(), folder_id, chosen )
                continue
            _outcome, changed = self.index.record_hash( path, after, digest, folder_id )
            touched |= changed
            hashed += 1
            with self._mutex:
                if self._pending.get( path, ( None, ) )[0] == seen:
                    self._pending.pop( path, None )
        if hashed:
            self.flush()
            touched |= self._pass_outputs()
        self._changed( touched, 'scan' )
        return touched

    def _changed( self, touched, reason ):
        if touched:
            self.on_change( set( touched ), reason )

    # --- the thread ---------------------------------------------------------------

    def rescan( self ):
        with self._mutex:
            self._pass_requested = True
        self._wake.set()

    def next_pass_due( self, now ):
        """When the next scheduled pass may start (monotonic seconds), or None."""
        if self._last_pass_end is None:
            return now
        if not self.interval:
            return None
        return self._last_pass_end + max( self.interval, PASS_GAP_FACTOR * self._last_pass_seconds )

    def start( self, watch_external=None ):
        self._thread = threading.Thread( target=self._run, args=( watch_external, ),
                                         name='catalog-scan', daemon=True )
        self._thread.start()

    def stop( self, timeout=10.0 ):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join( timeout )

    def _run( self, watch_external ):
        last_external = time.monotonic()
        while not self._stop.is_set():
            try:
                now = time.monotonic()
                due = self.next_pass_due( now )
                with self._mutex:
                    requested, self._pass_requested = self._pass_requested, False
                if requested or ( due is not None and now >= due ):
                    self.pass_now()
                if self.pending_count() and not self.busy():
                    self.hash_ready()
                if watch_external is not None and now - last_external >= EXTERNAL_CHECK_SECONDS:
                    last_external = now
                    watch_external()
            except Exception:   # the scanner never takes the engine down
                if self.logger is not None:
                    self.logger.exception( "catalog: scan failed" )
            self._wake.wait( self._sleep_for() )
            self._wake.clear()

    def _sleep_for( self ):
        now = time.monotonic()
        waits = [ EXTERNAL_CHECK_SECONDS ]
        due = self.next_pass_due( now )
        if due is not None:
            waits.append( max( 0.0, due - now ) )
        if self.pending_count():
            waits.append( BUSY_RECHECK_SECONDS if self.busy() else min( self.settle, BUSY_RECHECK_SECONDS ) )
        return max( 0.05, min( waits ) )
