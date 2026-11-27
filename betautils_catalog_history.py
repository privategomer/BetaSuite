"""
betautils_catalog_history.py - every run of every item, kept in the
catalog as a record of its own.

History is written from the RunRecord the pipeline hands its recorder
(betautils_run_record), through HistorySink, beside the stats and
failure logs and never from them. It is not rebuilt from disk: like
pins (and feedback later) it exists only in the catalog, so the
catalog's file is what to back up.

Recording a run also keeps the item in step:

  - a run on a file the catalog does not know yet adds it, keyed by
    the hash the run computed (a command-line run on a new file)
  - its stored status follows the outcome: processed, or skipped
    because the output is there, is done; failed, a full disk, or a
    file that is not a video is failed. A preview changes no status
  - the video type and the rule that chose it are kept as the item's
    last, which the library shows once nothing declares a type
  - the output it made, or found, is recorded as the item's output
  - a source the engine moved after its run (move_completed_sources)
    is followed to where it went: the same item, same status and type
"""

import json
import os

import betautils_catalog_index as bu_index
import betautils_catalog_store as bu_store

HISTORY_LIMIT_DEFAULT = 50

_STATUS_FOR = {
    ( 'processed', None ): bu_index.DONE,
    ( 'skipped', 'output.present' ): bu_index.DONE,
    ( 'skipped', 'source.not_video' ): bu_index.FAILED,
    ( 'failed', 'file.failed' ): bu_index.FAILED,
    ( 'out_of_space', 'disk.full' ): bu_index.FAILED,
    ( 'failed', 'disk.full' ): bu_index.FAILED,     # as the runs table keeps it
}


def status_after( run ):
    """The stored status a run leaves its item in, or None for no change."""
    if run.preview:
        return None
    return _STATUS_FOR.get( ( run.outcome, run.reason ) )


class History:
    """The runs table, and what recording a run does to its item."""

    def __init__( self, store, index ):
        self.store = store
        self.index = index

    def record( self, run ):
        """Keep one RunRecord. Returns the items touched."""
        if run.hash is None or run.path is None:
            return set()   # nothing was read, so nothing to key it by
        digest = run.hash
        touched = self._ensure_located( digest, run.path )
        if not self.index.has_item( digest ):
            return touched
        outcome = 'failed' if run.outcome == 'out_of_space' else run.outcome
        keys = run.keys or {}
        with self.store.write() as db:
            db.execute( 'INSERT INTO runs ( hash, program, job, path, started_at, ended_at, outcome, reason,'
                        ' output, detection_key, censor_key, encode_key, profile, rule, rule_params, sample_fps,'
                        ' preview, error_type, stage, timing ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,'
                        ' ?, ?, ?, ?, ?, ? )',
                        ( digest, run.program, run.job, bu_index.norm( run.path ), run.started_at,
                          run.ended_at if run.ended_at is not None else run.started_at, outcome, run.reason,
                          bu_index.norm( run.output ) if run.output else None, keys.get( 'detection' ),
                          keys.get( 'censor' ), keys.get( 'encode' ), run.profile, run.rule,
                          bu_store.dumps_params( run.rule_params ), run.sample_fps, int( bool( run.preview ) ),
                          run.error_type, run.stage,
                          json.dumps( run.timing, sort_keys=True ) if getattr( run, 'timing', None ) else None ) )
        status = status_after( run )
        if status is not None:
            self.index.set_status( digest, status )
        if run.profile and not run.preview:
            self.index.set_last_profile( digest, run.profile, run.rule, run.rule_params )
        if run.output and status == bu_index.DONE and os.path.exists( run.output ):
            import betautils_catalog_scan as bu_scan
            self.index.add_output( digest, run.output, bu_scan.output_fields( os.path.basename( run.output ) ) )
        return touched | { digest }

    def _ensure_located( self, digest, path ):
        """The run's file as a location of its item, adding the item if new."""
        path = bu_index.norm( path )
        known = { location['path'] for location in self.index.locations_of( digest ) } \
            if self.index.has_item( digest ) else set()
        if path in known:
            return set()
        try:
            file_stat = bu_index.Stat.of( os.stat( path ) )
        except OSError:
            return set()
        folder = self.index.folder_for( path )
        _outcome, touched = self.index.record_hash( path, file_stat, digest, folder['id'] if folder else None )
        return touched

    def moved( self, run ):
        """The run's source now lives at run.moved_to. Returns the items touched."""
        if run.hash is None or not run.moved_to:
            return set()
        old, new = bu_index.norm( run.path ), bu_index.norm( run.moved_to )
        try:
            file_stat = bu_index.Stat.of( os.stat( new ) )
        except OSError:
            return set()
        folder = self.index.folder_for( new )
        folder_id = folder['id'] if folder else None
        outcome, touched = self.index.observe( new, file_stat, folder_id )
        if outcome == 'needs_hash':
            # Another disk: the bytes are the ones just run, so the hash is known.
            _outcome, touched = self.index.record_hash( new, file_stat, run.hash, folder_id )
            touched |= self.index.drop_location( old )
        return touched | { run.hash }

    def runs_of( self, digest, limit=HISTORY_LIMIT_DEFAULT ):
        """The item's runs, newest first."""
        rows = self.store.read( 'SELECT * FROM runs WHERE hash = ? ORDER BY ended_at DESC, id DESC LIMIT ?',
                                ( digest, limit ) )
        return [ _run_out( row ) for row in rows ]


def _run_out( row ):
    return {
        'program': row['program'], 'job': row['job'], 'path': row['path'],
        'started_at': row['started_at'], 'ended_at': row['ended_at'],
        'outcome': row['outcome'], 'reason': row['reason'], 'output': row['output'],
        'keys': { 'detection': row['detection_key'], 'censor': row['censor_key'], 'encode': row['encode_key'] },
        'profile': row['profile'], 'rule': row['rule'], 'rule_params': bu_index.loads_params( row['rule_params'] ),
        'preview': bool( row['preview'] ), 'error_type': row['error_type'], 'stage': row['stage'],
    }


class HistorySink:
    """
    The run recorder's sink for the catalog: best-effort, so the catalog
    never fails a run. on_change( items, reason ) hears what it touched.
    """

    required = False

    def __init__( self, history, on_change=None ):
        self.history = history
        self.on_change = on_change

    def record( self, run ):
        self._tell( self.history.record( run ) )

    def moved( self, run ):
        self._tell( self.history.moved( run ) )

    def _tell( self, touched ):
        if touched and self.on_change is not None:
            self.on_change( touched, 'job' )


class CliHistory:
    """
    The command line's way into the catalog: the file opened shared
    beside a running engine, its history sink, and its pins. Opening
    never stops a run: a catalog that cannot be opened is logged and
    the run goes on without it (open() returns None).
    """

    def __init__( self, store ):
        self.store = store
        self.index = bu_index.Index( store )
        self.history = History( store, self.index )
        self.sink = HistorySink( self.history )

    def run_types( self ):
        """The pins and folder rules a command-line run uses, as the app's jobs do."""
        import betautils_catalog as bu_catalog
        import betautils_catalog_types as bu_types

        def roots_for( path ):
            return bu_catalog.Catalog._folder_roots( path, self.index.folders() )

        return bu_types.RunTypes( self.index.pin_of, roots_for )

    @classmethod
    def open( cls, logger=None, path=None ):
        try:
            store = bu_store.Store( path, logger=logger )
        except bu_store.CatalogError as err:
            if logger is not None:
                logger.warning( "the library catalog is not recording this run (%s %s)"%(
                    err.code, err.params ) )
            return None
        except Exception:   # whatever it is, the run goes on without the catalog
            if logger is not None:
                logger.warning( "the library catalog is not recording this run", exc_info=True )
            return None
        try:
            if store.created:
                import betautils_catalog as bu_catalog
                bu_catalog.seed( bu_index.Index( store ), bu_catalog.default_seed_folders(), logger )
            return cls( store )
        except Exception:
            store.close()
            if logger is not None:
                logger.warning( "the library catalog is not recording this run", exc_info=True )
            return None

    def close( self ):
        self.store.close()
