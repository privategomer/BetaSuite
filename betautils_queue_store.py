"""
betautils_queue_store.py - the engine's job queue, kept across restarts.

The queue lives in the engine's memory. So that closing the app (pause
and close, the tray's Quit, a shutdown) does not make the person queue
everything again, every job that has not ended is written to
queue.json in the state folder each time the queue changes, and read
back when the next engine makes its queue. A job stopped only because
the engine shut down is kept too: it ended for no reason of its own.

What is kept per job: its id, files, overrides (dotted paths), the next
file to run and its counts so far. Detection checkpoints and finished
render chunks are on disk already, so a restored job continues where it
was. Writing is best-effort: a queue that cannot be written is logged
and never stops a job.
"""

import json
import os

FILE_NAME = 'queue.json'
VERSION = 1


def default_path():
    import betautils_paths as bu_paths
    return os.path.join( bu_paths.kind_dir( 'state' ), FILE_NAME )


def row_for( job ):
    """The saved form of one job."""
    return {
        'job': job.id,
        'files': [ entry.path for entry in job.files ],
        'overrides': { '.'.join( path ): value for path, value in job.overrides.items() },
        'next_index': job.next_index,
        'counts': dict( job.counts ),
    }


def save( rows, path=None ):
    """Writes the queue's jobs, in order, replacing the file whole."""
    path = path or default_path()
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    temporary = '%s.%d.tmp'%( path, os.getpid() )
    try:
        with open( temporary, 'w', encoding='utf-8' ) as handle:
            json.dump( { 'version': VERSION, 'jobs': list( rows ) }, handle )
        os.replace( temporary, path )
    except BaseException:
        try:
            os.remove( temporary )
        except OSError:
            pass
        raise


def load( path=None, logger=None ):
    """The saved jobs, in order; [] when there is no readable queue."""
    path = path or default_path()
    try:
        with open( path, encoding='utf-8' ) as handle:
            data = json.load( handle )
    except FileNotFoundError:
        return []
    except ( OSError, ValueError ) as err:
        if logger is not None:
            logger.warning( "the saved job queue could not be read (%s); starting with an empty queue"%( err, ) )
        return []
    if not isinstance( data, dict ) or data.get( 'version' ) != VERSION or not isinstance( data.get( 'jobs' ), list ):
        if logger is not None:
            logger.warning( "the saved job queue is not one this engine reads; starting with an empty queue" )
        return []
    return [ row for row in data['jobs'] if _valid( row ) ]


def _valid( row ):
    return ( isinstance( row, dict ) and isinstance( row.get( 'job' ), str ) and row['job']
             and isinstance( row.get( 'files' ), list ) and row['files']
             and all( isinstance( path, str ) and path for path in row['files'] )
             and isinstance( row.get( 'overrides', {} ), dict )
             and isinstance( row.get( 'next_index', 0 ), int ) and not isinstance( row.get( 'next_index', 0 ), bool )
             and 0 <= row.get( 'next_index', 0 ) <= len( row['files'] )
             and isinstance( row.get( 'counts', {} ), dict ) )
