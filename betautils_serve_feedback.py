"""
betautils_serve_feedback.py - the feedback.* commands (protocol/README.md,
"Feedback"): marks made in the player on a result, kept in the catalog.

The client names the result by its path and never sends keys: the engine
looks them up (betautils_catalog_feedback). Times on the wire are on the
result's own timeline, as the player shows it; stored times are on the
source's (a preview's offset added).
"""

import math
import os

import betautils_protocol as bu_proto
from betautils_serve_catalog import _bad_params, _call, catalog_for


def _feedback( server ):
    import betautils_catalog_feedback as bu_feedback
    return bu_feedback.Feedback( catalog_for( server ).store )


def _path( data ):
    path = data.get( 'path' )
    if not isinstance( path, str ) or not path:
        raise _bad_params( 'path', bu_proto.REASON_MISSING if path is None else bu_proto.REASON_TYPE )
    if not os.path.isabs( path ):
        raise _bad_params( 'path', bu_proto.REASON_VALUE )
    return path


def _record_wire( record ):
    return { 'id': record['id'], 'time': record['time'], 'at_seconds': record['at_seconds'],
             'label': record['label'], 'verdict': record['verdict'], 'keys': record['keys'],
             'preview': record['preview'], 'created_at': record['created_at'] }


def _result_wire( result ):
    return { 'item': result['hash'], 'keys': result['keys'], 'preview': result['preview'], 'offset': result['offset'] }


def feedback_list( server, data ):
    """
    data: { path }. result: { result: {item, keys, preview, offset} | null,
    records: [...], labels }: whether the file is a result, the marks on
    it, and the labels a mark can be about.
    """
    import betautils_labels as bu_labels
    path = _path( data )
    result, records = _call( _feedback( server ).of_result, path )
    labels = list( bu_labels.LABELS )
    if result is None:
        return { 'result': None, 'records': [], 'labels': labels }
    return { 'result': _result_wire( result ),
             'records': [ _record_wire( record ) for record in records ], 'labels': labels }


def feedback_add( server, data ):
    """data: { path, time, verdict, label }. result: the record kept."""
    import betautils_catalog_feedback as bu_feedback
    import betautils_labels as bu_labels
    path = _path( data )
    time_seconds = data.get( 'time' )
    if isinstance( time_seconds, bool ) or not isinstance( time_seconds, ( int, float ) ):
        raise _bad_params( 'time', bu_proto.REASON_MISSING if time_seconds is None else bu_proto.REASON_TYPE )
    if not math.isfinite( time_seconds ) or time_seconds < 0:
        raise _bad_params( 'time', bu_proto.REASON_VALUE )
    verdict = data.get( 'verdict' )
    if verdict not in bu_feedback.VERDICTS:
        raise _bad_params( 'verdict', bu_proto.REASON_MISSING if verdict is None else
                           bu_proto.REASON_VALUE if isinstance( verdict, str ) else bu_proto.REASON_TYPE )
    if 'label' not in data:
        raise _bad_params( 'label', bu_proto.REASON_MISSING )
    label = data['label']
    if label is not None and label not in bu_labels.LABELS:
        raise _bad_params( 'label', bu_proto.REASON_VALUE if isinstance( label, str ) else bu_proto.REASON_TYPE )
    return _record_wire( _call( _feedback( server ).add, path, time_seconds, verdict, label ) )


def feedback_remove( server, data ):
    """data: { id }. result: {}."""
    record_id = data.get( 'id' )
    if isinstance( record_id, bool ) or not isinstance( record_id, int ):
        raise _bad_params( 'id', bu_proto.REASON_MISSING if record_id is None else bu_proto.REASON_TYPE )
    _call( _feedback( server ).remove, record_id )
    return {}


HANDLERS = {
    'feedback.list':   feedback_list,
    'feedback.add':    feedback_add,
    'feedback.remove': feedback_remove,
}
