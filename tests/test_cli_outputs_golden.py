"""
test_cli_outputs_golden.py - what betatv.py writes for a fixed set of
clips is pinned.

betatv.main() runs over a small tree of synthetic clips (two folders, a
clip without audio, a file that is not a video, an empty folder) with the
stub detector and the shipped config, then once more to skip everything.
The golden file holds, per scenario: the exit code, every path under the
output folder (folders included, since the run mirrors the input tree),
the stats and failure rows without their timings, and the sha256 of each
output file.

It was recorded before the CLI ran through the job runner
(betautils_job), so it proves that routing changed no output name, key
or byte. The clips are built with ffmpeg, so their bytes, and the
source hash in each output name, depend on the ffmpeg version; each
source hash is recorded as a token naming its clip ({a.mp4}). Under the
ffmpeg that recorded the golden everything is compared, output bytes
included. Under another one the exit codes, the output tree and each
stats row's identity (file, output name, the three keys, sizes, rates,
video dimensions and frames) are compared, since detection counts and
shot cuts follow the clip's pixels.

A change that is meant to change outputs updates the golden file on
purpose, and says so in CHANGELOG:

    python -m tests.test_cli_outputs_golden write
"""

import hashlib
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

import betaconfig
import betatv

from tests import config_harness as harness
from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip, _ffmpeg_available

GOLDEN_PATH = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ),
                            'golden', 'cli_fixture_outputs.json' )

# Fields of a stats or failure row that vary from run to run, or with
# the machine, rather than with what was produced.
VOLATILE_FIELDS = ( 'timestamp', 'detection_seconds', 'encode_seconds', 'total_seconds',
                    'elapsed_seconds', 'betasuite_version', 'local_overrides',
                    'render_memory_available_mb', 'render_memory_worker_cap',
                    'render_render_workers', 'execution_providers' )

# The settings the end-to-end harness pins, on top of the shipped config.
PINNED = (
    ( 'render_chunk_seconds', 1 ),
    ( 'render_workers', 1 ),
    ( 'logging_enabled', False ),
    ( 'stats_enabled', True ),
    ( 'preview_start_seconds', None ),
    ( 'preview_random_slice', False ),
    ( 'shot_cut_detection_enabled', True ),
    ( 'encode_preset', 'ultrafast' ),
    ( 'items_to_censor', [ 'exposed_breast' ] ),
    ( 'video_censor_fps', 5 ),
    ( 'move_completed_sources', False ),
    ( 'input_delete_probability', 0 ),
)

# Stats fields compared under any ffmpeg.
IDENTITY_FIELDS = ( 'file', 'output', 'detection_key', 'censor_key', 'encode_key',
                    'detector_backend', 'detector_variant', 'picture_sizes', 'video_censor_fps',
                    'nn_batch_size', 'global_min_prob', 'preview_mode', 'preview_offset_seconds',
                    'video_width', 'video_height', 'video_fps', 'video_frames', 'transcoded_source' )

SCENARIOS = {
    'full': {},
    'preview': { 'preview_mode_enabled': True, 'preview_max_seconds': 1,
                 'preview_start_seconds': 1 },
}


def ffmpeg_version():
    try:
        line = subprocess.run( [ 'ffmpeg', '-version' ], capture_output=True, text=True,
                               stdin=subprocess.DEVNULL ).stdout.splitlines()[0]
    except ( OSError, IndexError ):
        return None
    return line.split( ' Copyright' )[0]


def _sha256( path ):
    digest = hashlib.sha256()
    with open( path, 'rb' ) as handle:
        for block in iter( lambda: handle.read( 1 << 20 ), b'' ):
            digest.update( block )
    return digest.hexdigest()


def _tokens( value, names ):
    """value with each source hash replaced by its clip's token."""
    if isinstance( value, str ):
        for file_hash, token in names.items():
            value = value.replace( file_hash, token )
        return value
    if isinstance( value, list ):
        return [ _tokens( item, names ) for item in value ]
    if isinstance( value, dict ):
        return { _tokens( key, names ): _tokens( item, names ) for key, item in value.items() }
    return value


def _rows( path, names ):
    if not os.path.exists( path ):
        return []
    with open( path, 'r', encoding='UTF-8' ) as handle:
        rows = [ json.loads( line ) for line in handle if line.strip() ]
    return [ _tokens( { key: value for key, value in sorted( row.items() ) if key not in VOLATILE_FIELDS },
                      names ) for row in rows ]


def identity( produced ):
    """What is compared under any ffmpeg."""
    return {
        'exit_codes': produced['exit_codes'],
        'tree': produced['tree'],
        'tree_after_second_run': produced['tree_after_second_run'],
        'stats_rows_after_second_run': produced['stats_rows_after_second_run'],
        'failures': [ row.get( 'file' ) for row in produced['failures'] ],
        'stats': [ { key: row.get( key ) for key in IDENTITY_FIELDS } for row in produced['stats'] ],
        'outputs': sorted( produced['sha256'] ),
    }


class GoldenRun( BetaTvHarness ):
    """Builds the fixture tree and runs betatv.main() over it."""

    def build_tree( self ):
        _build_clip( os.path.join( self.uncensored, 'a.mp4' ), seconds=3 )
        _build_clip( os.path.join( self.uncensored, 'b_silent.mp4' ), seconds=2, with_audio=False )
        _build_clip( os.path.join( self.uncensored, 'sub', 'c.mp4' ), seconds=3,
                     width=160, height=120 )
        os.makedirs( os.path.join( self.uncensored, 'empty' ) )
        with open( os.path.join( self.uncensored, 'sub', 'notes.txt' ), 'w' ) as handle:
            handle.write( 'not a video' )

    def run_main( self ):
        with mock.patch.object( betatv, '_startup', return_value=self.logger ):
            return betatv.main()

    def source_tokens( self ):
        import betautils_hash as bu_hash
        names = {}
        for root, _dirs, files in os.walk( self.uncensored ):
            for name in files:
                if name.endswith( '.mp4' ):
                    names[bu_hash.md5_for_file( os.path.join( root, name ), 16 )] = '{%s}'%( name, )
        return names

    def tree( self ):
        entries = []
        for root, dirs, files in os.walk( self.censored ):
            relative = os.path.relpath( root, self.censored )
            for name in sorted( dirs ) + sorted( files ):
                entries.append( os.path.normpath( os.path.join( relative, name ) ).replace( os.sep, '/' ) )
        return sorted( entries )

    def hashes( self ):
        found = {}
        for root, _dirs, files in os.walk( self.censored ):
            for name in files:
                path = os.path.join( root, name )
                found[os.path.relpath( path, self.censored ).replace( os.sep, '/' )] = _sha256( path )
        return dict( sorted( found.items() ) )

    def record( self, scenario ):
        """Run one scenario twice from a clean tree; what it produced."""
        with harness.shipped():
            for name, value in PINNED:
                setattr( betaconfig, name, value )
            betaconfig.stats_path = os.path.join( self.tmpdir, 'stats.jsonl' )
            for name, value in SCENARIOS[scenario].items():
                setattr( betaconfig, name, value )
            harness.bu_config.invalidate_config_caches()
            self.build_tree()
            names = self.source_tokens()
            first = self.run_main()
            produced = {
                'exit_codes': [ first ],
                'tree': _tokens( self.tree(), names ),
                'stats': _rows( betaconfig.stats_path, names ),
                'failures': _rows( os.path.splitext( betaconfig.stats_path )[0] + '.failures.jsonl', names ),
                'sha256': _tokens( self.hashes(), names ),
            }
            second = self.run_main()
            produced['exit_codes'].append( second )
            produced['tree_after_second_run'] = _tokens( self.tree(), names )
            produced['stats_rows_after_second_run'] = len( _rows( betaconfig.stats_path, names ) )
        return produced


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestCliOutputsMatchTheGolden( GoldenRun ):

    def _check( self, scenario ):
        with open( GOLDEN_PATH, 'r', encoding='UTF-8' ) as handle:
            golden = json.load( handle )
        expected = golden['scenarios'][scenario]
        produced = self.record( scenario )
        self.assertEqual( identity( produced ), identity( expected ) )
        if ffmpeg_version() == golden['ffmpeg']:
            self.assertEqual( produced, expected )

    def test_full_run( self ):
        self._check( 'full' )

    def test_preview_run( self ):
        self._check( 'preview' )


def write():
    class Writer( GoldenRun ):
        def runTest( self ):
            pass

    golden = { 'ffmpeg': ffmpeg_version(), 'scenarios': {} }
    for scenario in SCENARIOS:
        case = Writer()
        case.setUp()
        try:
            golden['scenarios'][scenario] = case.record( scenario )
        finally:
            case.tearDown()
    with open( GOLDEN_PATH, 'w', encoding='UTF-8' ) as handle:
        json.dump( golden, handle, indent=1, sort_keys=True )
        handle.write( '\n' )
    print( "wrote %s"%( GOLDEN_PATH, ) )


if __name__ == '__main__':
    if sys.argv[1:] == [ 'write' ]:
        write()
    else:
        unittest.main()
