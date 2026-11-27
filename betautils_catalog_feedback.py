"""
betautils_catalog_feedback.py - "missed something" and "censored too
much", marked at a point in a result, kept in the catalog.

A record is the source's content hash, the time in the source, the label
it is about (or none, for "not sure"), the verdict, and the detection,
censor and encode keys of the result it was made on, so later tuning can
tell which settings it judged. Capture only: nothing here sends or acts
on a record.

The keys never come from the caller. The result being played is looked
up by its path: a finished output the catalog knows, then a run that
wrote that file (previews are kept only there), then the file's own
name for an item the catalog knows. Anything else is not a result
(feedback.not_a_result).

Times are on the source's timeline: a preview slice starts at an offset
its name records (to a tenth of a second), and that offset is added, so
a mark made on a preview and one made on the full output line up.

Feedback exists nowhere else, so it is not tied to the items table: no
rebuild, missing share or new machine can cascade it away. It is keyed
by hash and joins its item again when the file is found. Forgetting an
item removes its feedback with the rest of what the catalog knew of it
(betautils_catalog_index.Index.forget). No path is stored.
"""

import math
import os
import re

import betautils_catalog_index as bu_index
import betautils_catalog_store as bu_store

VERDICTS = ( 'missed', 'too_much' )

# '-preview@600.0s+20.0s' (betautils_cache_paths.preview_cache_suffix);
# no '@' is a slice from the start.
_PREVIEW_OFFSET = re.compile( r'^-preview@([0-9]+(?:\.[0-9]+)?)s' )


def preview_offset( name ):
    """The slice offset in seconds a preview output's file name carries, or 0.0."""
    fields = _parse_name( name )
    if fields is None:
        return 0.0
    match = _PREVIEW_OFFSET.match( fields['tail'] )
    return float( match.group( 1 ) ) if match else 0.0


def _parse_name( name ):
    if not name.endswith( '.mp4' ):
        return None
    import betautils_cache_paths as bu_cache
    return bu_cache.parse_video_output_basename( name[:-len( '.mp4' )] )


def _labels():
    import betautils_labels as bu_labels
    return bu_labels.LABELS


class Feedback:
    """The feedback table, and the result a path is."""

    def __init__( self, store, clock=None ):
        self.store = store
        self.clock = clock or store.clock

    def resolve( self, path ):
        """
        The result at path: { hash, keys: {detection, censor, encode},
        preview, offset }. Raises CatalogError feedback.not_a_result.
        """
        path = bu_index.norm( path )
        name = os.path.basename( path )
        rows = self.store.read( 'SELECT hash, detection_key, censor_key, encode_key FROM outputs'
                                ' WHERE path = ? AND present = 1 ORDER BY seen_at DESC, id DESC LIMIT 1', ( path, ) )
        if rows:
            return _result( rows[0], False, 0.0 )
        rows = self.store.read( "SELECT hash, detection_key, censor_key, encode_key, preview FROM runs"
                                " WHERE output = ? AND outcome IN ( 'processed', 'skipped' )"
                                " AND detection_key IS NOT NULL AND censor_key IS NOT NULL AND encode_key IS NOT NULL"
                                " ORDER BY ended_at DESC, id DESC LIMIT 1", ( path, ) )
        if rows and os.path.isfile( path ):
            preview = bool( rows[0]['preview'] )
            return _result( rows[0], preview, preview_offset( name ) if preview else 0.0 )
        fields = _parse_name( name )
        if fields is not None and os.path.isfile( path ) and self.store.read(
                'SELECT 1 FROM items WHERE hash = ?', ( fields['hash'], ) ):
            return _result( fields, fields['preview'], preview_offset( name ) )
        raise bu_store.CatalogError( 'feedback.not_a_result', path=path )

    def add( self, path, time_seconds, verdict, label=None ):
        """Keep one mark made at time_seconds into the result at path. Returns the record."""
        if verdict not in VERDICTS:
            raise ValueError( 'verdict' )
        if label is not None and label not in _labels():
            raise ValueError( 'label' )
        if ( isinstance( time_seconds, bool ) or not isinstance( time_seconds, ( int, float ) )
                or not math.isfinite( time_seconds ) or time_seconds < 0 ):
            raise ValueError( 'time' )
        result = self.resolve( path )
        keys = result['keys']
        at = round( float( time_seconds ) + result['offset'], 3 )
        with self.store.write() as db:
            cursor = db.execute( 'INSERT INTO feedback ( hash, at_seconds, label, verdict, detection_key, censor_key,'
                                 ' encode_key, preview, created_at, app_version ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )',
                                 ( result['hash'], at, label, verdict, keys['detection'], keys['censor'],
                                   keys['encode'], int( result['preview'] ), self.clock(), bu_store._app_version() ) )
            record_id = cursor.lastrowid
        return self.record( record_id )

    def record( self, record_id ):
        rows = self.store.read( 'SELECT * FROM feedback WHERE id = ?', ( record_id, ) )
        if not rows:
            raise bu_store.CatalogError( 'feedback.no_such_record', id=record_id )
        return record_out( rows[0] )

    def of_item( self, digest ):
        """Every mark on the item, in the order of its time in the source."""
        return [ record_out( row ) for row in self.store.read(
            'SELECT * FROM feedback WHERE hash = ? ORDER BY at_seconds, id', ( digest, ) ) ]

    def of_result( self, path ):
        """
        ( result, records ): the result at path and the marks made on it
        (same item and keys), times on the result's own timeline. result is
        None, and records empty, for a file that is not a result.
        """
        try:
            result = self.resolve( path )
        except bu_store.CatalogError as err:
            if err.code == 'feedback.not_a_result':
                return None, []
            raise
        keys = result['keys']
        rows = self.store.read( 'SELECT * FROM feedback WHERE hash = ? AND detection_key = ? AND censor_key = ?'
                                ' AND encode_key = ? ORDER BY at_seconds, id',
                                ( result['hash'], keys['detection'], keys['censor'], keys['encode'] ) )
        records = []
        for row in rows:
            record = record_out( row )
            record['time'] = round( record['at_seconds'] - result['offset'], 3 )
            records.append( record )
        return result, records

    def remove( self, record_id ):
        """Drop one mark. Returns the hash it was on."""
        with self.store.write() as db:
            row = db.execute( 'SELECT hash FROM feedback WHERE id = ?', ( record_id, ) ).fetchone()
            if row is None:
                raise bu_store.CatalogError( 'feedback.no_such_record', id=record_id )
            db.execute( 'DELETE FROM feedback WHERE id = ?', ( record_id, ) )
        return row['hash']


def _result( row, preview, offset ):
    return { 'hash': row['hash'],
             'keys': { 'detection': row['detection_key'], 'censor': row['censor_key'], 'encode': row['encode_key'] },
             'preview': bool( preview ), 'offset': offset }


def record_out( row ):
    return {
        'id': row['id'], 'hash': row['hash'], 'at_seconds': row['at_seconds'], 'label': row['label'],
        'verdict': row['verdict'],
        'keys': { 'detection': row['detection_key'], 'censor': row['censor_key'], 'encode': row['encode_key'] },
        'preview': bool( row['preview'] ), 'created_at': row['created_at'], 'app_version': row['app_version'],
    }
