"""
betautils_catalog_status.py - Queued and Processing, while a job runs.

The stored status (new, done, failed) is what runs left behind
(betautils_catalog_history). While the engine's queue holds a file, the
library shows it queued, and processing while it runs; that lives here,
in memory, driven by the job events, so an engine that stops in the
middle of a file never leaves an item stuck as processing.

  job.state queued        every file of the job the catalog knows: queued
  job.file started        that file: processing
  job.file ended          that file: back to its stored status
  job.state paused        files not yet ended: queued
  job.state completed, cancelled, stopped: every file of the job: back
"""

import threading

QUEUED = 'queued'
PROCESSING = 'processing'
FINAL_STATES = ( 'completed', 'cancelled', 'stopped' )
FILE_ENDINGS = ( 'processed', 'skipped', 'failed' )


class ItemStatus:
    def __init__( self, index ):
        self.index = index
        self._mutex = threading.Lock()
        self._jobs = {}        # job id -> { file index: item hash }
        self._items = {}       # item hash -> ( status, job id )

    def status_of( self, digest ):
        """( status, job ) while a job holds the item, else None."""
        with self._mutex:
            return self._items.get( digest )

    def job_event( self, name, data, files ):
        """
        One job event. files are the job's source paths, by index.
        Returns the items whose shown status changed.
        """
        job = data.get( 'job' )
        if name == 'job.state':
            state = data.get( 'state' )
            if state == 'queued':
                return self._queue( job, files )
            if state == 'paused':
                return self._set_all( job, QUEUED )
            if state in FINAL_STATES:
                return self._clear( job )
            return set()
        if name == 'job.file':
            position, state = data.get( 'file' ), data.get( 'state' )
            if state == 'started':
                return self._set_one( job, position, files, PROCESSING )
            if state in FILE_ENDINGS:
                return self._end_one( job, position )
        return set()

    def _queue( self, job, files ):
        mapping = {}
        for position, path in enumerate( files ):
            digest = self.index.location_hash( path )
            if digest is not None:
                mapping[position] = digest
        with self._mutex:
            self._jobs[job] = mapping
            for digest in mapping.values():
                self._items[digest] = ( QUEUED, job )
        return set( mapping.values() )

    def _set_one( self, job, position, files, status ):
        with self._mutex:
            mapping = self._jobs.setdefault( job, {} )
            digest = mapping.get( position )
        if digest is None and position is not None and 0 <= position < len( files ):
            digest = self.index.location_hash( files[position] )
        if digest is None:
            return set()
        with self._mutex:
            self._jobs.setdefault( job, {} )[position] = digest
            self._items[digest] = ( status, job )
        return { digest }

    def _end_one( self, job, position ):
        with self._mutex:
            mapping = self._jobs.get( job, {} )
            digest = mapping.pop( position, None )
            if digest is not None and self._items.get( digest, ( None, None ) )[1] == job:
                del self._items[digest]
                return { digest }
        return set()

    def _set_all( self, job, status ):
        with self._mutex:
            touched = set()
            for digest in self._jobs.get( job, {} ).values():
                if self._items.get( digest, ( None, None ) )[1] == job:
                    self._items[digest] = ( status, job )
                    touched.add( digest )
            return touched

    def _clear( self, job ):
        with self._mutex:
            touched = set()
            for digest in self._jobs.pop( job, {} ).values():
                if self._items.get( digest, ( None, None ) )[1] == job:
                    del self._items[digest]
                    touched.add( digest )
            return touched
