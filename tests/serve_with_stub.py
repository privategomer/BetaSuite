"""
serve_with_stub.py - the engine's serve mode with the stub detector, for
tests that drive real jobs through a real engine process.

    python tests/serve_with_stub.py <log file> <child fd record file>

The data, cache and state folders go under the record file's folder.

Everything but the model is real: the job queue, the pipeline, render's
ffmpeg children. Each ffmpeg child render starts is looked at once it is
running ffmpeg, and its open descriptors are appended to the record
file as JSON lines, so a test can check that none of them is the
engine's command or result pipe. Not a test module (no test_ prefix).
"""

import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, REPO )

# Each engine its own data, cache and state folders under the record
# file's folder, set before anything resolves a path.
_ROOT = os.path.join( os.path.dirname( os.path.abspath( sys.argv[2] ) ), 'os' )
for _kind in ( 'DATA', 'CACHE', 'STATE' ):
    os.environ['XDG_%s_HOME'%( _kind, )] = os.path.join( _ROOT, _kind.lower() )

import numpy as np                       # noqa: E402

import betaconfig                        # noqa: E402
import betautils_config as bu_config     # noqa: E402
import betautils_detector as bu_detector  # noqa: E402
import betautils_render as bu_render     # noqa: E402
import betautils_serve as bu_serve       # noqa: E402


class StubDetector:
    """A box in the middle of every frame, as in test_betatv_end_to_end."""

    def get_session( self ):
        return object()

    def raw_boxes_for_imgs( self, imgs, size, session, ts ):
        boxes = []
        for img, timestamp in zip( imgs, ts ):
            height, width = np.asarray( img ).shape[:2]
            boxes.append( { 'x': float( width//4 ), 'y': float( height//4 ), 'w': float( width//3 ),
                            'h': float( height//3 ), 'class_id': 'exposed_breast', 'score': 0.95,
                            't': timestamp, 'size': size } )
        return boxes


def record_children( record_path ):
    real_popen = subprocess.Popen

    class RecordingPopen( real_popen ):
        def __init__( self, *args, **kwargs ):
            super().__init__( *args, **kwargs )
            links = _links_once_ffmpeg( self.pid )
            if links is not None:
                with open( record_path, 'a', encoding='utf-8' ) as handle:
                    handle.write( json.dumps( { 'pid': self.pid, 'links': links } ) + '\n' )

    bu_render.subprocess = type( 'subprocess_with_record', (), dict(
        vars( subprocess ), Popen=RecordingPopen ) )


def _links_once_ffmpeg( pid, timeout=5.0 ):
    """The child's descriptor links, read once it runs ffmpeg; None if it ended first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if 'ffmpeg' in os.path.basename( os.readlink( '/proc/%d/exe'%( pid, ) ) ):
                folder = '/proc/%d/fd'%( pid, )
                return sorted( os.readlink( os.path.join( folder, name ) ) for name in os.listdir( folder ) )
        except OSError:
            return None
        time.sleep( 0.001 )
    return None


def main():
    log_path, record_path = sys.argv[1], sys.argv[2]
    stub = StubDetector()
    bu_detector.get_detector = lambda *args, **kwargs: stub
    bu_detector.get_picture_sizes = lambda *args, **kwargs: [ 320 ]
    for name, value in ( ( 'video_censor_fps', 5 ), ( 'encode_preset', 'ultrafast' ),
                         ( 'render_workers', 1 ), ( 'render_chunk_seconds', 1 ),
                         ( 'items_to_censor', [ 'exposed_breast' ] ), ( 'stats_enabled', True ),
                         ( 'move_completed_sources', False ), ( 'input_delete_probability', 0 ) ):
        setattr( betaconfig, name, value )
    bu_config.invalidate_config_caches()
    if sys.platform.startswith( 'linux' ):
        record_children( record_path )
    return bu_serve.run( log_path=log_path, log_level='debug' )


if __name__ == '__main__':
    sys.exit( main() )
