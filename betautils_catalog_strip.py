"""
betautils_catalog_strip.py - the player's scrub strip: small frames of a
video at known times, shown while the pointer drags along the timeline.

Seeking a long-GOP video while dragging is decode-bound in any webview
(seconds behind the pointer at 4K), so the player shows a strip frame
while dragging and seeks exactly only on release. The strip is built
once per video, in two passes:

  keyframes  only the stream's keyframes are decoded (ffmpeg
             -skip_frame nokey). Cheap even at 4K, so the player has
             pictures within seconds
  even       when keyframes are sparser than one per MIN_SPACING
             seconds, a second pass decodes the whole stream and keeps a
             frame at an even interval, at most MAX_FRAMES of them

Both passes keep at most MAX_FRAMES frames, chosen as they decode, so a
long file never holds more than that in memory.

PRIVACY (spec sections 3 and 10). Only a censored output's strip is
stored clear. Any other video's strip is a source's: each frame is
shrunk and blurred exactly as the source thumbnail is
(betautils_catalog_thumbs.blur) before anything is written, and frames
reach this process through a pipe, never a file.

Files, in cache/strips/: <key>.jpg (the frames in rows of COLUMNS) and
<key>.json (the layout: times, tile size, columns, pass, blurred). The
key covers the file's path, size and modification time, the pass and
whether it is blurred, so a changed file gets a new strip.
"""

import hashlib
import json
import os
import re
import subprocess
import threading

FOLDER_NAME = 'strips'
TILE_WIDTH = 160
COLUMNS = 20
MAX_FRAMES = 600
MIN_SPACING = 2.0
QUALITY = 75
TIME_WAIT_SECONDS = 10
KEYFRAMES = 'keyframes'
EVEN = 'even'

_PTS_TIME = re.compile( rb'pts_time:\s*(-?[0-9.]+)' )


class StripError( Exception ):
    """reason: not_video (nothing could be decoded) or error."""

    def __init__( self, reason ):
        super().__init__( reason )
        self.reason = reason


def folder():
    import betautils_paths as bu_paths
    return os.path.join( bu_paths.kind_dir( 'cache' ), FOLDER_NAME )


def key_for( path, density, blurred ):
    stat = os.stat( path )
    identity = '\0'.join( ( os.path.realpath( path ), str( stat.st_size ), str( stat.st_mtime_ns ), density,
                            'blurred' if blurred else 'clear' ) )
    return hashlib.sha256( identity.encode( 'utf-8' ) ).hexdigest()[:32]


def image_path( key ):
    return os.path.join( folder(), key + '.jpg' )


def layout_path( key ):
    return os.path.join( folder(), key + '.json' )


def is_strip_image( path ):
    """Whether path is a strip image in the cache folder (for the shell's reader)."""
    name = os.path.basename( path )
    return ( os.path.dirname( os.path.abspath( path ) ) == os.path.abspath( folder() )
             and re.fullmatch( r'[0-9a-f]{32}\.jpg', name ) is not None )


def tile_size( width, height, tile_width=TILE_WIDTH ):
    """( width, height ) of a frame scaled to tile_width, both even, as ffmpeg's scale is told."""
    tile_height = max( 2, int( round( height * tile_width / float( width ) / 2.0 ) ) * 2 )
    return tile_width, tile_height


def command( path, density, tile, interval=None ):
    """ffmpeg decoding the pass's frames, scaled, to raw BGR on its stdout, with their times on stderr."""
    scale = 'scale=%d:%d'%tile
    head = [ 'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'info' ]
    if density == KEYFRAMES:
        return head + [ '-skip_frame', 'nokey', '-i', path, '-an', '-sn', '-dn', '-vf', scale + ',showinfo',
                        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1' ]
    return head + [ '-i', path, '-an', '-sn', '-dn', '-vf', 'fps=1/%.6f,%s,showinfo'%( interval, scale ),
                    '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1' ]


class Selector:
    """Keeps at most `limit` frames over `duration`: the first frame at or after each slot."""

    def __init__( self, duration, limit=None ):
        limit = MAX_FRAMES if limit is None else limit
        self.slot = ( duration / float( limit ) ) if duration > 0 else 0.0
        self.limit = limit
        self.next_at = None
        self.kept = []

    def offer( self, seconds, frame ):
        if len( self.kept ) >= self.limit:
            return
        if self.next_at is not None and seconds < self.next_at:
            return
        self.kept.append( ( seconds, frame ) )
        self.next_at = seconds + self.slot if self.slot else None


def decode( path, density, tile, duration, interval=None, run=None ):
    """[ ( seconds, BGR frame ) ] of the pass, at most MAX_FRAMES, in order."""
    import numpy
    popen = run or subprocess.Popen
    try:
        process = popen( command( path, density, tile, interval ), stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    except OSError as err:
        raise StripError( 'error' ) from err
    times = []
    ended = []
    cond = threading.Condition()

    def drain():
        # showinfo logs each frame's time as the filter passes it, so a
        # frame's time arrives before or soon after its pixels.
        for line in process.stderr:
            found = _PTS_TIME.search( line )
            if found:
                with cond:
                    times.append( float( found.group( 1 ) ) )
                    cond.notify_all()
        with cond:
            ended.append( True )
            cond.notify_all()

    def time_of( index ):
        with cond:
            cond.wait_for( lambda: len( times ) > index or ended, timeout=TIME_WAIT_SECONDS )
            return times[index] if len( times ) > index else None

    reader = threading.Thread( target=drain, name='strip-stderr', daemon=True )
    reader.start()
    size = tile[0] * tile[1] * 3
    selector = Selector( duration )
    index = 0
    try:
        while True:
            data = process.stdout.read( size )
            if not data or len( data ) < size:
                break
            seconds = time_of( index )
            index += 1
            if seconds is None:
                break
            # A kept frame holds its own bytes; the rest are dropped at once.
            selector.offer( seconds, numpy.frombuffer( data, dtype=numpy.uint8 ).reshape( tile[1], tile[0], 3 ) )
    finally:
        process.stdout.close()
        process.wait()
        reader.join( 5 )
    if not selector.kept:
        raise StripError( 'not_video' )
    return selector.kept


def needs_even_pass( times, duration ):
    """Whether keyframes alone leave gaps wider than MIN_SPACING on average."""
    if duration <= 0:
        return False
    return duration / max( 1, len( times ) ) > MIN_SPACING


def even_interval( duration ):
    return max( 1.0, duration / float( MAX_FRAMES ) )


def sprite( frames, blurred ):
    """( JPEG bytes, tile ( width, height ) ) of the frames in rows of COLUMNS."""
    import cv2
    import numpy
    import betautils_catalog_thumbs as bu_thumbs
    tiles = [ bu_thumbs.blur( frame ) if blurred else frame for frame in frames ]
    height, width = tiles[0].shape[:2]
    rows = ( len( tiles ) + COLUMNS - 1 ) // COLUMNS
    sheet = numpy.zeros( ( rows * height, min( len( tiles ), COLUMNS ) * width, 3 ), dtype=numpy.uint8 )
    for index, tile in enumerate( tiles ):
        row, column = divmod( index, COLUMNS )
        sheet[row * height:( row + 1 ) * height, column * width:( column + 1 ) * width] = tile[:height, :width]
    ok, data = cv2.imencode( '.jpg', sheet, [ cv2.IMWRITE_JPEG_QUALITY, QUALITY ] )
    if not ok:
        raise StripError( 'error' )
    return data.tobytes(), ( width, height )


def read( key ):
    """The layout of a strip on disk, with its image path, or None."""
    try:
        with open( layout_path( key ), encoding='utf-8' ) as handle:
            layout = json.load( handle )
    except ( OSError, ValueError ):
        return None
    if not os.path.exists( image_path( key ) ):
        return None
    layout['image'] = image_path( key )
    return layout


def build( path, density, blurred, facts, run=None ):
    """Make one pass's strip, write it, and return its layout (as read gives it)."""
    import betautils_catalog_thumbs as bu_thumbs
    tile = tile_size( facts['width'], facts['height'] )
    interval = even_interval( facts['duration'] ) if density == EVEN else None
    kept = decode( path, density, tile, facts['duration'], interval, run )
    data, shown = sprite( [ frame for _, frame in kept ], blurred )
    key = key_for( path, density, blurred )
    layout = { 'density': density, 'blurred': bool( blurred ), 'frames': [ round( t, 3 ) for t, _ in kept ],
               'tile': { 'width': shown[0], 'height': shown[1] }, 'columns': COLUMNS,
               'duration': round( facts['duration'], 3 ) }
    bu_thumbs._write_atomic( image_path( key ), data )
    bu_thumbs._write_atomic( layout_path( key ), json.dumps( layout ).encode( 'utf-8' ) )
    return read( key )
