"""
betautils_person.py - person segmentation, and reverse mode's whole-body windows.

The nudity detectors report body PARTS. Reverse mode's "leave men
uncensored" needs whole PEOPLE, so a second model (person_backend,
YOLO11-seg by default) masks every person in each sampled frame. A
person counts as male when male-specific detections (a masculine face,
a penis, a male chest) land inside their mask and outweigh female ones.
Their mask becomes a window: the original pixels, inside a frame the
frame style otherwise obscures.

Three parts:

  backends    person_backend config, adapters in segmenters/, sessions.
              Same shape as detector_backend: 'selected', one block per
              backend, a model_variant per backend, env overrides for
              sweeps (BETASUITE_PERSON_BACKEND_OVERRIDE / _VARIANT_)
  the pass    persons_for_video: samples the video at the detection
              rate, runs the model, caches the masks (.npz beside the
              detection caches). BackgroundPersonPass runs it in a
              thread alongside detection, the way the shot-cut scan runs
  windows     attribute people to male from detections, then
              PersonMaskTimeline answers "which pixels are male bodies
              at time t" for the renderer

Errors point one way. Between two samples a window is the INTERSECTION
of the two samples' masks, so a moving body loses its edges to the
frame style rather than dragging a window over whatever it just left.
Every outline that is not a man's (a woman's, or anyone not attributed)
is cut out of the men's, so where a woman overlaps a man she stays
obscured and he stays clear around her. Anyone not attributed stays
obscured.

Analogy: the detections are name tags, the masks are silhouettes. A
silhouette wearing a "male" tag gets cut out of the blur; no tag, no
cut.
"""

import bisect
import collections
import importlib
import os
import threading

import cv2
import numpy as np

import betaconfig
import betautils_reverse as bu_reverse


BACKEND_OVERRIDE_ENV_VAR = 'BETASUITE_PERSON_BACKEND_OVERRIDE'
VARIANT_OVERRIDE_ENV_VAR = 'BETASUITE_PERSON_VARIANT_OVERRIDE'

_BACKENDS = {
    'yolo11_seg':  'segmenters.yolo11_seg',
    'rf_detr_seg': 'segmenters.rf_detr_seg',
}
_RESERVED_KEYS = { 'selected' }
DEFAULT_BACKEND = 'yolo11_seg'

# The person windows are composed on a grid this many cells along the
# frame's longer side, then scaled up per frame. 384 is 10 px per cell
# at 4K and keeps one sample's grid around 80 KB.
GRID_LONG_SIDE = 384
# A mask probability at or above this is inside the person.
INSIDE = 128
# Per-sample grids kept per video; render chunks walk forward in time.
_GRID_CACHE_SIZE = 64


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def selected_backend_name():
    override = os.environ.get( BACKEND_OVERRIDE_ENV_VAR )
    if override:
        return override
    block = getattr( betaconfig, 'person_backend', None )
    if isinstance( block, dict ):
        return block.get( 'selected', DEFAULT_BACKEND )
    return DEFAULT_BACKEND


def registered_backend_names():
    return sorted( _BACKENDS )


def get_segmenter( backend_name=None ):
    """
    The adapter module for a backend.

    Raises:
        ValueError: unknown backend, with the valid choices.
    """
    name = backend_name or selected_backend_name()
    if name not in _BACKENDS:
        raise ValueError( "unknown person_backend %r - valid choices: %s"%(
            name, ', '.join( registered_backend_names() ) ) )
    return importlib.import_module( _BACKENDS[name] )


def get_backend_config( backend_name=None ):
    block = getattr( betaconfig, 'person_backend', None )
    if not isinstance( block, dict ):
        return {}
    entry = block.get( backend_name or selected_backend_name(), {} )
    return entry if isinstance( entry, dict ) else {}


def is_implemented( backend_name=None ):
    try:
        return bool( getattr( get_segmenter( backend_name ), 'IMPLEMENTED', False ) )
    except ValueError:
        return False


def variant_names( backend_name=None ):
    try:
        return sorted( getattr( get_segmenter( backend_name ), 'VARIANTS', {} ) )
    except ValueError:
        return []


def resolve_config( backend_name=None ):
    name = backend_name or selected_backend_name()
    return get_segmenter( name ).resolve_config( get_backend_config( name ) )


def identity( backend_name=None ):
    """What changes the masks: backend plus the adapter's own settings."""
    name = backend_name or selected_backend_name()
    return dict( get_segmenter( name ).identity( get_backend_config( name ) ), backend=name )


def batch_size( backend_name=None ):
    return max( 1, int( resolve_config( backend_name ).get( 'batch_size', 1 ) ) )


_sessions = {}
_sessions_lock = threading.Lock()


def get_session( backend_name=None, logger=None ):
    """One session per backend, variant and model file for the process."""
    name = backend_name or selected_backend_name()
    resolved = resolve_config( name )
    key = ( name, resolved.get( 'model_variant' ), resolved.get( 'model_path' ) )
    with _sessions_lock:
        if key not in _sessions:
            _sessions[key] = get_segmenter( name ).get_session( get_backend_config( name ), logger )
        return _sessions[key]


def persons_for_imgs( imgs, ts, session=None, backend_name=None ):
    """Person dicts for a batch of BGR images (see segmenters/yolo11_seg.decode_one)."""
    name = backend_name or selected_backend_name()
    session = session if session is not None else get_session( name )
    return get_segmenter( name ).persons_for_imgs( imgs, session, ts, get_backend_config( name ) )


# ---------------------------------------------------------------------------
# The pass, and its cache
# ---------------------------------------------------------------------------
#
# A pass keeps every person's mask for the whole video until attribution,
# which needs the detections. As uint8 probabilities that is roughly
# 500 MB per hour of video with two or three people on screen, so each
# mask is packed to one bit per cell (inside or not) as soon as it is
# decoded: about 60 MB per hour. Everything downstream thresholds at
# INSIDE anyway.

def compact( person ):
    """The person with its mask packed to bits ('mask_bits', 'mask_shape')."""
    if 'mask' not in person:
        return person
    packed = dict( person )
    mask = packed.pop( 'mask' )
    packed['mask_shape'] = tuple( int( v ) for v in mask.shape )
    packed['mask_bits'] = np.packbits( mask >= INSIDE, axis=None )
    return packed


def mask_of( person ):
    """A person's mask as uint8, 255 inside and 0 outside, packed or not."""
    if 'mask' in person:
        return person['mask']
    height, width = person['mask_shape']
    bits = np.unpackbits( person['mask_bits'], count=height * width )
    return ( bits.reshape( height, width ) * 255 ).astype( np.uint8 )


class Stopped( Exception ):
    """The pass was asked to stop (BackgroundPersonPass.stop); its progress is checkpointed."""


def checkpoint_path_for( cache_path ):
    return cache_path + '.checkpoint.npz'


def save_cache( path, sample_times, people, next_index=None ):
    """
    Write a pass's results as compressed arrays, atomically; masks as
    bits. With next_index it is a checkpoint: everything up to that
    sample index is in it.
    """
    people = [ compact( person ) for person in people ]
    masks = [ person['mask_bits'] for person in people ]
    offsets = np.zeros( len( masks ) + 1, dtype=np.int64 )
    if masks:
        offsets[1:] = np.cumsum( [ mask.size for mask in masks ] )
    os.makedirs( os.path.dirname( path ) or '.', exist_ok=True )
    tmp = path + '.tmp.npz'
    np.savez_compressed(
        tmp,
        sample_times=np.asarray( sample_times, dtype=np.float64 ),
        t=np.asarray( [ person['t'] for person in people ], dtype=np.float64 ),
        box=np.asarray( [ ( person['x'], person['y'], person['w'], person['h'] ) for person in people ],
                        dtype=np.float32 ).reshape( -1, 4 ),
        mask_box=np.asarray( [ person['mask_box'] for person in people ], dtype=np.float32 ).reshape( -1, 4 ),
        score=np.asarray( [ person['score'] for person in people ], dtype=np.float32 ),
        mask_shape=np.asarray( [ person['mask_shape'] for person in people ], dtype=np.int32 ).reshape( -1, 2 ),
        mask_offset=offsets,
        mask_data=( np.concatenate( masks ) if masks else np.zeros( 0, dtype=np.uint8 ) ),
        next_index=np.asarray( [ -1 if next_index is None else next_index ], dtype=np.int64 ) )
    os.replace( tmp, path )


def load_cache( path ):
    """(sample_times, people) as save_cache wrote them, masks packed."""
    sample_times, people, _next_index = load_checkpoint( path )
    return sample_times, people


def load_checkpoint( path ):
    """(sample_times, people, next_index); next_index is None for a finished pass."""
    with np.load( path ) as data:
        next_index = int( data['next_index'][0] ) if 'next_index' in data.files else -1
        offsets = data['mask_offset']
        mask_data = data['mask_data']
        people = []
        for index in range( len( data['t'] ) ):
            height, width = ( int( v ) for v in data['mask_shape'][index] )
            x, y, w, h = ( float( v ) for v in data['box'][index] )
            people.append( {
                't': float( data['t'][index] ), 'x': x, 'y': y, 'w': w, 'h': h,
                'score': float( data['score'][index] ),
                'mask_box': tuple( float( v ) for v in data['mask_box'][index] ),
                'mask_shape': ( height, width ),
                'mask_bits': mask_data[ offsets[index]:offsets[index + 1] ].copy(),
            } )
        return ( [ float( t ) for t in data['sample_times'] ], people,
                 None if next_index < 0 else next_index )


def persons_for_video( source, fname, file_hash, preview_suffix, preview_offset_seconds,
                       preview_max_seconds, logger, sample_fps, stop_event=None ):
    """
    Every person in every sampled frame of one video (or preview slice).

    Samples exactly the frames detection samples (same rate, offset and
    iterator), so a person and the detections that attribute them share
    a timestamp. Served from cache when present.

    Resumable the way detection is: a checkpoint every
    detection_checkpoint_frames samples, and one on Ctrl-C or when
    stop_event is set, so a re-run picks up where this one stopped. The
    checkpoint sits beside the cache under the same key, so a changed
    person model starts over instead of mixing two models' masks.

    Raises:
        betautils_signals.Interrupted: Ctrl-C, after checkpointing.
        Stopped: stop_event was set, after checkpointing.

    Returns:
        (sample_times, people): every sampled time, and the flat list of
        person dicts across them.
    """
    import betautils_cache_paths as bu_cache
    import betautils_log as bu_log
    import betautils_signals as bu_signals
    import betautils_video as bu_video

    backend_name = selected_backend_name()
    cache_path = bu_cache.person_cache_path_for( file_hash, sample_fps, backend_name,
                                                 identity( backend_name ), preview_suffix )
    if os.path.exists( cache_path ):
        try:
            sample_times, people = load_cache( cache_path )
            logger.info( "person pass: %d cached person mask(s) over %d sample(s) for %s"%(
                len( people ), len( sample_times ), fname ) )
            return sample_times, people
        except Exception as err:
            logger.warning( "corrupt person cache %s (%s), regenerating"%( cache_path, err ) )

    checkpoint_path = checkpoint_path_for( cache_path )
    sample_times, people, start_index = [], [], 0
    if os.path.exists( checkpoint_path ):
        try:
            sample_times, people, start_index = load_checkpoint( checkpoint_path )
            start_index = start_index or 0
            logger.info( "person pass: resuming %s at sample %d (%d person mask(s) so far)"%(
                fname, start_index, len( people ) ) )
        except Exception as err:
            logger.warning( "corrupt person checkpoint %s (%s), starting over"%( checkpoint_path, err ) )
            sample_times, people, start_index = [], [], 0

    session = get_session( backend_name, logger )
    batch = batch_size( backend_name )
    interval = getattr( betaconfig, 'detection_checkpoint_frames', 500 )
    frames, times = [], []
    next_index = last_checkpoint = start_index

    def flush():
        if frames:
            people.extend( compact( person )
                           for person in persons_for_imgs( frames, times, session, backend_name ) )
            frames.clear()
            times.clear()

    total = None
    if source.vid_fps:
        span = ( preview_max_seconds if preview_max_seconds is not None
                 else source.num_frames / source.vid_fps - preview_offset_seconds )
        total = max( 1, int( span * sample_fps ) )
    capture = source.open_capture()
    samples = bu_video.prefetched( bu_video.iter_sampled_frames(
                                       capture, source.vid_fps, sample_fps,
                                       offset_seconds=preview_offset_seconds,
                                       num_frames=source.num_frames, max_seconds=preview_max_seconds,
                                       start_sample_index=start_index ),
                                   getattr( betaconfig, 'detection_prefetch_frames', 0 ) )
    try:
        with bu_log.ProgressReporter( "persons", total=total ) as progress:
            for sample_index, timestamp, frame in samples:
                if stop_event is not None and stop_event.is_set():
                    raise Stopped()
                bu_signals.check()
                sample_times.append( timestamp )
                frames.append( frame )
                times.append( timestamp )
                if len( frames ) >= batch:
                    flush()
                next_index = sample_index + 1
                progress.update( next_index )
                # Only with an empty buffer, so the checkpoint holds every
                # sample before next_index and nothing after it.
                if not frames and interval > 0 and next_index - last_checkpoint >= interval:
                    save_cache( checkpoint_path, sample_times, people, next_index )
                    last_checkpoint = next_index
            flush()
    except ( bu_signals.Interrupted, Stopped ):
        flush()
        save_cache( checkpoint_path, sample_times, people, next_index )
        logger.info( "person pass: checkpointed %s at sample %d before stopping"%( fname, next_index ) )
        raise
    finally:
        samples.close()
        capture.release()
    save_cache( cache_path, sample_times, people )
    try:
        os.remove( checkpoint_path )
    except OSError:
        pass
    logger.info( "person pass: %d person mask(s) over %d sample(s) for %s"%(
        len( people ), len( sample_times ), fname ) )
    return sample_times, people


class BackgroundPersonPass:
    """persons_for_video in a thread, so it overlaps detection's decode and GPU work."""

    def __init__( self, *args ):
        self._result = None
        self._error = None
        self._stop = threading.Event()
        self._thread = threading.Thread( target=self._run, args=args, name='person-pass', daemon=True )
        self._thread.start()

    def _run( self, *args ):
        try:
            self._result = persons_for_video( *args, stop_event=self._stop )
        except Stopped:
            pass
        except BaseException as err:   # re-raised in the caller's thread by result()
            self._error = err

    def stop( self ):
        """
        Ask the pass to checkpoint and end, and wait until it has. A no-op
        once it has finished. The caller's way out on any error or
        interrupt: the thread is a daemon, and exiting without this would
        kill it mid-pass and lose everything since its last checkpoint.
        """
        self._stop.set()
        self._thread.join()

    def result( self ):
        """The pass's (sample_times, people), once. Not kept: they are the run's largest object."""
        self._thread.join()
        if self._error is not None:
            raise self._error
        result, self._result = self._result, None
        return result


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def _time_key( t ):
    return round( float( t ), 3 )


def _inside_mask( person, x, y ):
    """Whether image point (x, y) falls inside the person's mask."""
    mx, my, mw, mh = person['mask_box']
    if not ( mx <= x < mx + mw and my <= y < my + mh ):
        return False
    mask = mask_of( person )
    column = min( mask.shape[1] - 1, int( ( x - mx ) / mw * mask.shape[1] ) )
    row = min( mask.shape[0] - 1, int( ( y - my ) / mh * mask.shape[0] ) )
    return mask[ row, column ] >= INSIDE


def _iou( a, b ):
    ax1, ay1, bx1, by1 = a['x'] + a['w'], a['y'] + a['h'], b['x'] + b['w'], b['y'] + b['h']
    inter_w = min( ax1, bx1 ) - max( a['x'], b['x'] )
    inter_h = min( ay1, by1 ) - max( a['y'], b['y'] )
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    return inter / ( a['w'] * a['h'] + b['w'] * b['h'] - inter )


def evidence_detections( raw_boxes, vid_w, vid_h, backend_name=None ):
    """
    The detections that can attribute a person, by time: male and female
    labels at or above evidence_min_prob, after the same dedup,
    promotion, geometry and suppression stages tracking sees.
    """
    import betautils_log as bu_log
    import betautils_track as bu_track
    settings = bu_reverse.settings()['person_windows']
    labels = set( settings['male_labels'] ) | set( settings['female_labels'] )
    floor = settings['evidence_min_prob']
    with bu_track.tracking_observer( None ):
        survivors, _stats = bu_track.filter_raw_detections(
            raw_boxes, vid_w, vid_h, sorted( labels ), backend_name,
            bu_log.silent_logger( 'betasuite.person_evidence' ) )
    by_time = collections.defaultdict( list )
    for raw in survivors:
        if raw['class_id'] in labels and raw['score'] >= floor:
            by_time[ _time_key( raw['t'] ) ].append( raw )
    return by_time


def _centre( raw ):
    return raw['x'] + raw['w']/2.0, raw['y'] + raw['h']/2.0


def _sample_evidence( people, evidence, male_labels, female_labels ):
    """
    For each person at one sample: (male vote, female vote, female points).

    A detection votes only for the ONE person whose outline holds its
    centre. One inside two overlapping outlines says nothing about which
    of them it belongs to, so it votes for neither. Female points are
    every female detection centre inside the person, shared or not: the
    backstop below needs them all.
    """
    holders = []
    for raw in evidence:
        x, y = _centre( raw )
        holders.append( ( raw['class_id'], ( x, y ),
                          [ index for index, person in enumerate( people ) if _inside_mask( person, x, y ) ] ) )
    result = []
    for index in range( len( people ) ):
        male = any( label in male_labels and inside == [ index ] for label, _point, inside in holders )
        female = any( label in female_labels and inside == [ index ] for label, _point, inside in holders )
        points = [ point for label, point, inside in holders if label in female_labels and index in inside ]
        result.append( ( male, female, points ) )
    return result


def _show_and_hide( people, is_male, female_points ):
    """
    One sample's windows: {'show': male outlines, 'hide': everyone else}.

    Everyone not shown is cut OUT of the shown outlines, so where a
    woman, or anyone not attributed, overlaps a man, she stays under the
    frame style and he stays clear around her.

    Backstop for two bodies the model returned as ONE outline: a female
    detection inside a man's outline that no separate non-male outline
    accounts for means part of a woman is inside his. That man is not
    shown at this sample.
    """
    show, hide = [], []
    for index, person in enumerate( people ):
        if not is_male[index]:
            hide.append( person )
            continue
        unexplained = [ point for point in female_points[index]
                        if not any( not is_male[other] and _inside_mask( people[other], *point )
                                    for other in range( len( people ) ) if other != index ) ]
        ( hide if unexplained else show ).append( person )
    return { 'show': show, 'hide': hide }


def male_people_by_time( sample_times, people, evidence_by_time ):
    """
    The outlines to show and to cut out at each sample time.

    People are linked across samples by box overlap (track_iou, within
    track_max_gap_seconds), so a man whose face shows in one sample is
    still a man in the next one where it doesn't. A track is male when
    it has a male vote in at least one sample and more male than female
    votes (see _sample_evidence for what votes). Then, per sample, see
    _show_and_hide.

    Returns:
        {time_key: {'show': [person, ...], 'hide': [person, ...]}} with
        every sample time present.
    """
    settings = bu_reverse.settings()['person_windows']
    male_labels = set( settings['male_labels'] )
    female_labels = set( settings['female_labels'] )
    track_iou = settings['track_iou']
    max_gap = settings['track_max_gap_seconds']

    by_time = collections.defaultdict( list )
    for person in people:
        by_time[ _time_key( person['t'] ) ].append( person )

    tracks = []
    samples = []
    for t in sorted( sample_times ):
        key = _time_key( t )
        here = by_time.get( key, [] )
        flags = _sample_evidence( here, evidence_by_time.get( key, [] ), male_labels, female_labels )
        live = [ track for track in tracks if t - track['last_t'] <= max_gap ]
        pairs = sorted( ( ( _iou( track['last'], person ), track_index, person_index )
                          for track_index, track in enumerate( live )
                          for person_index, person in enumerate( here ) ), reverse=True )
        taken_tracks, assigned = set(), {}
        for overlap, track_index, person_index in pairs:
            if overlap < track_iou:
                break
            if track_index in taken_tracks or person_index in assigned:
                continue
            taken_tracks.add( track_index )
            assigned[person_index] = live[track_index]
        sample_tracks = []
        for person_index, person in enumerate( here ):
            track = assigned.get( person_index )
            if track is None:
                track = { 'male': 0, 'female': 0 }
                tracks.append( track )
            male_vote, female_vote, _points = flags[person_index]
            track['male'] += int( male_vote )
            track['female'] += int( female_vote )
            track['last'], track['last_t'] = person, t
            sample_tracks.append( track )
        samples.append( ( key, here, sample_tracks, [ points for _m, _f, points in flags ] ) )

    windows = {}
    for key, here, sample_tracks, female_points in samples:
        is_male = [ track['male'] >= 1 and track['male'] > track['female'] for track in sample_tracks ]
        windows[key] = _show_and_hide( here, is_male, female_points )
    return windows


def male_people_in_still( people, evidence ):
    """
    A still's {'show': ..., 'hide': ...}. No time axis, so no tracks: a
    person is male with a male vote and no female vote of their own.
    """
    settings = bu_reverse.settings()['person_windows']
    flags = _sample_evidence( people, evidence, set( settings['male_labels'] ),
                              set( settings['female_labels'] ) )
    return _show_and_hide( people, [ male and not female for male, female, _points in flags ],
                           [ points for _m, _f, points in flags ] )


# ---------------------------------------------------------------------------
# Windows on the time axis
# ---------------------------------------------------------------------------

class PersonMaskTimeline:
    """
    Which pixels are male bodies at any render time.

    Each sample's shown outlines are drawn onto a coarse grid (built on
    first use, a few dozen kept), and every outline in its 'hide' list is
    cut back out. A frame between two samples gets the cell-wise MINIMUM
    of both grids, so only what was a male body, uncovered, at both ends
    is shown. Past the first or last sample, or across a gap
    wider than 1.5 sample intervals, nothing is shown.

    Thread-safe: render chunks run in parallel threads and share it.
    """

    def __init__( self, windows_by_time, vid_w, vid_h, sample_fps, shrink=0.0 ):
        self.times = sorted( windows_by_time )
        self.windows = windows_by_time
        self.vid_w, self.vid_h = int( vid_w ), int( vid_h )
        self.scale = GRID_LONG_SIDE / float( max( self.vid_w, self.vid_h ) )
        self.grid_w = max( 1, int( round( self.vid_w * self.scale ) ) )
        self.grid_h = max( 1, int( round( self.vid_h * self.scale ) ) )
        interval = 1.0 / sample_fps if sample_fps else 0.0
        self.max_gap = 1.5 * interval
        self.at_sample = min( 0.02, 0.25 * interval ) if interval else 0.02
        self.shrink = float( shrink or 0.0 )
        self._grids = collections.OrderedDict()
        self._lock = threading.Lock()

    def windowed_samples( self ):
        return sum( 1 for key in self.times if self.windows[key]['show'] )

    def _paint( self, people, morph ):
        """
        Every outline onto one grid. Each is trimmed (cv2.erode) or grown
        (cv2.dilate) by shrink times its own shorter side: shown outlines
        shrink and cut-out outlines grow, so at a boundary between a man
        and a woman the doubt goes to the frame style.
        """
        grid = np.zeros( ( self.grid_h, self.grid_w ), dtype=np.uint8 )
        for person in people:
            mx, my, mw, mh = person['mask_box']
            gx0, gy0 = int( round( mx * self.scale ) ), int( round( my * self.scale ) )
            gw = max( 1, int( round( mw * self.scale ) ) )
            gh = max( 1, int( round( mh * self.scale ) ) )
            cell = cv2.resize( mask_of( person ), ( gw, gh ), interpolation=cv2.INTER_LINEAR )
            radius = int( round( self.shrink * min( person['w'], person['h'] ) * self.scale ) )
            if radius >= 1:
                kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, ( 2*radius + 1, 2*radius + 1 ) )
                if morph == 'grow':
                    cell = cv2.copyMakeBorder( cell, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0 )
                    cell = cv2.dilate( cell, kernel )
                    gx0, gy0, gw, gh = gx0 - radius, gy0 - radius, gw + 2*radius, gh + 2*radius
                else:
                    cell = cv2.erode( cell, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0 )
            x0, y0 = max( 0, gx0 ), max( 0, gy0 )
            x1, y1 = min( self.grid_w, gx0 + gw ), min( self.grid_h, gy0 + gh )
            if x1 <= x0 or y1 <= y0:
                continue
            region = grid[ y0:y1, x0:x1 ]
            np.maximum( region, cell[ y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0 ], out=region )
        return grid

    def _draw( self, entry ):
        grid = self._paint( entry['show'], 'shrink' )
        if entry['hide'] and grid.any():
            grid[ self._paint( entry['hide'], 'grow' ) >= INSIDE ] = 0
        return grid

    def _grid( self, key ):
        with self._lock:
            grid = self._grids.get( key )
            if grid is not None:
                self._grids.move_to_end( key )
                return grid
        grid = self._draw( self.windows[key] )
        with self._lock:
            self._grids[key] = grid
            while len( self._grids ) > _GRID_CACHE_SIZE:
                self._grids.popitem( last=False )
        return grid

    def grid_at( self, t ):
        """The coarse window grid at time t, or None when nothing is shown."""
        if not self.times:
            return None
        index = bisect.bisect_right( self.times, _time_key( t ) ) - 1
        before = self.times[index] if index >= 0 else None
        after = self.times[index + 1] if index + 1 < len( self.times ) else None
        if before is not None and t - before <= self.at_sample:
            if not self.windows[before]['show']:
                return None
            return self._grid( before )
        if after is not None and after - t <= self.at_sample:
            if not self.windows[after]['show']:
                return None
            return self._grid( after )
        if before is None or after is None or after - before > self.max_gap:
            return None
        if not self.windows[before]['show'] or not self.windows[after]['show']:
            return None
        return np.minimum( self._grid( before ), self._grid( after ) )

    def mask_at( self, t, frame_w, frame_h ):
        """
        (x, y, w, h, alpha) in frame pixels, alpha uint8 0 or 255 over
        that rectangle, or None when no male body is shown at t.
        """
        grid = self.grid_at( t )
        if grid is None:
            return None
        rows = np.flatnonzero( grid.max( axis=1 ) >= INSIDE )
        if not rows.size:
            return None
        columns = np.flatnonzero( grid.max( axis=0 ) >= INSIDE )
        # One cell of margin so the upscale has something to fall off into.
        gy0, gy1 = max( 0, rows[0] - 1 ), min( self.grid_h, rows[-1] + 2 )
        gx0, gx1 = max( 0, columns[0] - 1 ), min( self.grid_w, columns[-1] + 2 )
        scale_x, scale_y = frame_w / float( self.grid_w ), frame_h / float( self.grid_h )
        x0, y0 = int( gx0 * scale_x ), int( gy0 * scale_y )
        x1, y1 = min( frame_w, int( np.ceil( gx1 * scale_x ) ) ), min( frame_h, int( np.ceil( gy1 * scale_y ) ) )
        if x1 <= x0 or y1 <= y0:
            return None
        crop = grid[ gy0:gy1, gx0:gx1 ]
        scaled = cv2.resize( crop, ( x1 - x0, y1 - y0 ), interpolation=cv2.INTER_LINEAR )
        _threshold, alpha = cv2.threshold( scaled, INSIDE - 1, 255, cv2.THRESH_BINARY )
        return x0, y0, x1 - x0, y1 - y0, alpha


def window_masks_for_video( sample_times, people, raw_boxes, vid_w, vid_h, sample_fps,
                            backend_name=None ):
    """The PersonMaskTimeline for one video."""
    evidence = evidence_detections( raw_boxes, vid_w, vid_h, backend_name )
    windows = male_people_by_time( sample_times, people, evidence )
    return PersonMaskTimeline( windows, vid_w, vid_h, sample_fps,
                               bu_reverse.settings()['person_windows']['shrink'] )


def window_mask_for_still( people, raw_boxes, img_w, img_h, backend_name=None ):
    """(x, y, w, h, alpha) of male bodies in a still, or None."""
    evidence = evidence_detections( raw_boxes, img_w, img_h, backend_name )
    flat = [ raw for group in evidence.values() for raw in group ]
    timeline = PersonMaskTimeline( { 0.0: male_people_in_still( people, flat ) }, img_w, img_h, None,
                                   bu_reverse.settings()['person_windows']['shrink'] )
    return timeline.mask_at( 0.0, img_w, img_h )
