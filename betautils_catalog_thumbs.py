"""
betautils_catalog_thumbs.py - library thumbnails under the privacy rule.

THE RULE (spec section 3). Only the blurred thumbnail of a source is ever
written to disk. A folder of clear stills would be a folder of explicit
images of every source, there for file managers, desktop search and
backups to find. So:

  - a source's frame is decoded by ffmpeg into a pipe, never a file
  - in memory it is shrunk to BLUR_SIZE pixels on its long side and
    blurred; only that is written (cache/thumbnails/<hash>.png)
  - reveal (hover) decodes the frame again and hands it back in memory,
    as JPEG bytes, and nothing of it is written anywhere. The cost is a
    short delay on each reveal
  - a censored output's thumbnail is an ordinary thumbnail, stored clear
    (cache/thumbnails/output-<name hash>.jpg), as the spec allows

The frame is the one at FRAME_AT of the video's length, past any intro.
"""

import base64
import hashlib
import os
import subprocess

FRAME_AT = 0.10
BLUR_SIZE = 24
REVEAL_SIZE = 320
REVEAL_QUALITY = 85
FOLDER_NAME = 'thumbnails'
FFMPEG_TIMEOUT_SECONDS = 60


class ThumbnailError( Exception ):
    """reason: not_video (no frame could be decoded) or error."""

    def __init__( self, reason ):
        super().__init__( reason )
        self.reason = reason


def folder():
    import betautils_paths as bu_paths
    return os.path.join( bu_paths.kind_dir( 'cache' ), FOLDER_NAME )


def blurred_path( digest ):
    return os.path.join( folder(), '%s.png'%( digest ) )


def output_path( output ):
    name = hashlib.sha256( os.path.basename( output ).encode( 'utf-8' ) ).hexdigest()[:16]
    return os.path.join( folder(), 'output-%s.jpg'%( name ) )


def frame_command( path, seconds, width ):
    """ffmpeg decoding one frame, scaled, to PNG on its stdout (a pipe)."""
    return [ 'ffmpeg', '-nostdin', '-loglevel', 'error', '-ss', '%.3f'%( seconds ), '-i', path,
             '-frames:v', '1', '-vf', 'scale=%d:-2'%( width ), '-f', 'image2pipe', '-c:v', 'png', 'pipe:1' ]


def grab_frame( path, width=REVEAL_SIZE ):
    """One frame of the video as a BGR array, in memory only."""
    import cv2
    import numpy
    import betautils_video as bu_video
    try:
        duration = bu_video.probe_duration_seconds( path ) or 0.0
    except Exception:
        duration = 0.0
    for seconds in ( duration * FRAME_AT, 0.0 ):
        try:
            done = subprocess.run( frame_command( path, seconds, width ), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=FFMPEG_TIMEOUT_SECONDS )
        except ( OSError, subprocess.TimeoutExpired ) as err:
            raise ThumbnailError( 'error' ) from err
        if done.returncode == 0 and done.stdout:
            frame = cv2.imdecode( numpy.frombuffer( done.stdout, dtype=numpy.uint8 ), cv2.IMREAD_COLOR )
            if frame is not None:
                return frame
    raise ThumbnailError( 'not_video' )


def blur( frame ):
    """The frame shrunk to BLUR_SIZE on its long side, then blurred."""
    import cv2
    height, width = frame.shape[:2]
    scale = BLUR_SIZE / float( max( height, width ) )
    small = cv2.resize( frame, ( max( 1, round( width * scale ) ), max( 1, round( height * scale ) ) ),
                        interpolation=cv2.INTER_AREA )
    return cv2.GaussianBlur( small, ( 3, 3 ), 1.0 )


def _encode( image, extension, params=() ):
    import cv2
    ok, data = cv2.imencode( extension, image, list( params ) )
    if not ok:
        raise ThumbnailError( 'error' )
    return data.tobytes()


def _write_atomic( path, data ):
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    temporary = '%s.%d.tmp'%( path, os.getpid() )
    try:
        with open( temporary, 'wb' ) as handle:
            handle.write( data )
        os.replace( temporary, path )
    except BaseException:
        try:
            os.remove( temporary )
        except OSError:
            pass
        raise


def _answer( data, mime, cached ):
    import cv2
    import numpy
    image = cv2.imdecode( numpy.frombuffer( data, dtype=numpy.uint8 ), cv2.IMREAD_COLOR )
    height, width = image.shape[:2] if image is not None else ( 0, 0 )
    return { 'mime': mime, 'data': base64.b64encode( data ).decode( 'ascii' ),
             'width': int( width ), 'height': int( height ), 'cached': cached }


def blurred( digest, source ):
    """The source's blurred thumbnail: from the cache, or made and cached."""
    path = blurred_path( digest )
    try:
        with open( path, 'rb' ) as handle:
            return _answer( handle.read(), 'image/png', True )
    except OSError:
        pass
    data = _encode( blur( grab_frame( source ) ), '.png' )
    _write_atomic( path, data )
    return _answer( data, 'image/png', False )


def revealed( source ):
    """The source's frame, clear, in memory only. Nothing is written."""
    import cv2
    frame = grab_frame( source, REVEAL_SIZE )
    return _answer( _encode( frame, '.jpg', ( cv2.IMWRITE_JPEG_QUALITY, REVEAL_QUALITY ) ), 'image/jpeg', False )


def output_thumbnail( output ):
    """A censored output's thumbnail, stored clear like any thumbnail."""
    import cv2
    path = output_path( output )
    try:
        with open( path, 'rb' ) as handle:
            return _answer( handle.read(), 'image/jpeg', True )
    except OSError:
        pass
    data = _encode( grab_frame( output, REVEAL_SIZE ), '.jpg', ( cv2.IMWRITE_JPEG_QUALITY, REVEAL_QUALITY ) )
    _write_atomic( path, data )
    return _answer( data, 'image/jpeg', False )


def forget( digest ):
    try:
        os.remove( blurred_path( digest ) )
    except OSError:
        pass
