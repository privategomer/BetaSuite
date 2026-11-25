"""
betautils_serve_preview.py - preview.look over the engine protocol.

    data:   { previews: [ { art: { style, item } } | { image, boxes }, plus
                          optional labels and seed per entry ],
              overrides?: { dotted path: value } }
    result: { previews: [ { path, key, cached, width, height } ] }

Each entry is drawn by the real censor code (betautils_preview). labels
are the labels to censor (default: items_to_censor), so a tile can show a
label's look while it is switched off. overrides are per-run values, the
same as config.get's, applied for this command only and never written.
While a job is queued, running or paused the live config belongs to it,
so preview.look answers config.busy, as config.get does with overrides.
"""

import betautils_protocol as bu_proto
import betautils_serve_config as bu_serve_config

MAX_PREVIEWS = 64
MAX_BOXES = 256


def _bad( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


def _int( value ):
    return isinstance( value, int ) and not isinstance( value, bool )


def _boxes( value, field ):
    if not isinstance( value, list ):
        raise _bad( field, bu_proto.REASON_TYPE )
    if len( value ) > MAX_BOXES:
        raise _bad( field, bu_proto.REASON_VALUE )
    import betaconst
    out = []
    for box in value:
        if not isinstance( box, dict ) or not all( _int( box.get( key ) ) for key in ( 'x', 'y', 'w', 'h' ) ):
            raise _bad( field, bu_proto.REASON_TYPE )
        if box.get( 'label' ) not in betaconst.classes or box['w'] <= 0 or box['h'] <= 0:
            raise _bad( field, bu_proto.REASON_VALUE )
        out.append( { key: box[key] for key in ( 'label', 'x', 'y', 'w', 'h' ) } )
    return out


def _entry( entry, index ):
    """One preview request, checked: ( art, image, boxes, labels, seed )."""
    import betaconst
    where = 'previews.%d'%( index, )
    if not isinstance( entry, dict ):
        raise _bad( where, bu_proto.REASON_TYPE )
    art = entry.get( 'art' )
    image = entry.get( 'image' )
    if ( art is None ) == ( image is None ):
        raise _bad( where, bu_proto.REASON_MISSING if art is None else bu_proto.REASON_VALUE )
    boxes = None
    if art is not None:
        if ( not isinstance( art, dict ) or not isinstance( art.get( 'style' ), str )
                or not isinstance( art.get( 'item' ), str ) ):
            raise _bad( where + '.art', bu_proto.REASON_TYPE )
        if 'boxes' in entry:
            boxes = _boxes( entry['boxes'], where + '.boxes' )
    else:
        if not isinstance( image, str ) or not image:
            raise _bad( where + '.image', bu_proto.REASON_TYPE )
        if 'boxes' not in entry:
            raise _bad( where + '.boxes', bu_proto.REASON_MISSING )
        boxes = _boxes( entry['boxes'], where + '.boxes' )
    labels = entry.get( 'labels' )
    if labels is not None:
        if not isinstance( labels, list ) or not all( isinstance( label, str ) for label in labels ):
            raise _bad( where + '.labels', bu_proto.REASON_TYPE )
        if any( label not in betaconst.classes for label in labels ):
            raise _bad( where + '.labels', bu_proto.REASON_VALUE )
    seed = entry.get( 'seed', 0 )
    if not _int( seed ) or seed < 0:
        raise _bad( where + '.seed', bu_proto.REASON_TYPE )
    return art, image, boxes, labels, seed


def _render_all( entries ):
    import betautils_preview as bu_preview
    out = []
    for art, image_path, boxes, labels, seed in entries:
        if art is not None:
            try:
                item = bu_preview.art_item( art['style'], art['item'] )
            except bu_preview.UnknownArt:
                raise bu_proto.ProtocolError( 'preview.unknown_art', { 'style': art['style'], 'item': art['item'] } )
            image_path = item['path']
            boxes = item['boxes'] if boxes is None else boxes
        try:
            image = bu_preview.read_image( image_path )
        except bu_preview.BadImage:
            raise bu_proto.ProtocolError( 'preview.bad_image', {} )
        height, width = image.shape[:2]
        raw = bu_preview.raw_detections( boxes, width, height )
        chosen = bu_preview.default_labels() if labels is None else labels
        out.append( bu_preview.preview( image, raw, chosen, seed ) )
    return out


def preview_look( server, data ):
    previews = data.get( 'previews' )
    if not isinstance( previews, list ):
        raise _bad( 'previews', bu_proto.REASON_MISSING if previews is None else bu_proto.REASON_TYPE )
    if not previews or len( previews ) > MAX_PREVIEWS:
        raise _bad( 'previews', bu_proto.REASON_VALUE )
    entries = [ _entry( entry, index ) for index, entry in enumerate( previews ) ]
    overrides = bu_serve_config._overrides( data )
    jobs = getattr( server, 'jobs', None )
    busy = jobs.busy_job() if jobs is not None else None
    if busy is not None:
        raise bu_proto.ProtocolError( 'config.busy', { 'job': busy } )
    if not overrides:
        return { 'previews': _render_all( entries ) }
    # Checked first, so nothing is drawn from a config a run would refuse.
    current = bu_serve_config.store()
    checked = current.evaluate( current.text(), overrides )
    if not checked.ok:
        raise bu_serve_config._invalid( checked.errors )
    evaluation = current.evaluate( current.text(), overrides,
                                   extra=lambda _config: _render_all( entries ) )
    return { 'previews': evaluation.extra }


HANDLERS = {
    'preview.look': preview_look,
}
