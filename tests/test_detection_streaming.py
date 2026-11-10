"""
test_detection_streaming.py - reading detection caches without holding
the library as dicts, and the meter the analysis hub runs every step in.

The analysis hub was killed by the kernel (exit 137) because each
cache-reading step parsed every detection in the library into a dict,
about 500 bytes each, and three ran at once. The fix keeps one video's
dicts at a time and holds the rest as columns. None of that may change
a number: these tests pin that the columns give back exactly what the
file held, that anything not the standard shape is left alone, and that
betabench's measurements read the same through either path.
"""

import gzip
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, REPO_ROOT )
sys.path.insert( 0, os.path.join( REPO_ROOT, 'tools', 'bench' ) )

import betautils_detections as bu_detections  # noqa: E402
import betautils_hash as bu_hash  # noqa: E402
import betautils_tuning as bu_tuning  # noqa: E402

LABELS = ( 'exposed_breast', 'exposed_vulva', 'face_femme', 'exposed_belly' )


def records( count, seed=1, t_step=0.111 ):
    rng = random.Random( seed )
    out = []
    for index in range( count ):
        out.append( { 'x': rng.uniform( -5, 1900 ), 'y': rng.uniform( 0, 1080 ), 'w': rng.uniform( 1, 600 ),
                      'h': rng.uniform( 1, 600 ), 'class_id': rng.choice( LABELS ),
                      'score': rng.random(), 't': round( ( index // 3 ) * t_step, 6 ),
                      'size': rng.choice( ( 320, 640 ) ) } )
    return out


def write_cache( directory, name, content ):
    path = os.path.join( directory, name )
    bu_hash.write_json( content, path )
    return path


class TestColumns( unittest.TestCase ):

    def test_round_trip_is_exact_through_json( self ):
        original = records( 500 )
        original.append( { 'x': 0.1, 'y': 1e-300, 'w': 5e-324, 'h': 1.7976931348623157e308,
                           'class_id': 'exposed_vulva', 'score': 0.30000000000000004, 't': -0.0,
                           'size': 2**40 } )
        parsed = json.loads( json.dumps( original ) )
        columns = bu_detections.to_columns( parsed )
        self.assertIsNotNone( columns )
        back = columns.to_dicts()
        self.assertEqual( back, parsed )
        for got, want in zip( back, parsed ):
            self.assertEqual( list( got ), [ 'x', 'y', 'w', 'h', 'class_id', 'score', 't', 'size' ] )
            for key in want:
                self.assertIs( type( got[key] ), type( want[key] ), key )
                if isinstance( want[key], float ):
                    self.assertEqual( repr( got[key] ), repr( want[key] ), key )

    def test_a_tenth_of_the_memory( self ):
        columns = bu_detections.to_columns( records( 1000 ) )
        self.assertLessEqual( columns.nbytes, 1000 * 57 )
        self.assertEqual( len( columns ), 1000 )

    def test_extra_keys_are_added_to_every_record_and_each_call_is_fresh( self ):
        columns = bu_detections.to_columns( records( 4 ) )
        first = columns.to_dicts( _file_hash='abc' )
        self.assertTrue( all( record['_file_hash'] == 'abc' for record in first ) )
        first[0]['x'] = 'poisoned'
        self.assertNotEqual( columns.to_dicts()[0]['x'], 'poisoned' )
        self.assertNotIn( '_file_hash', columns.to_dicts()[0] )

    def test_more_than_256_labels_still_round_trip( self ):
        many = [ dict( record, class_id='label_%d'%( index, ) ) for index, record in enumerate( records( 300 ) ) ]
        columns = bu_detections.to_columns( many )
        self.assertEqual( columns.label.dtype.itemsize, 2 )
        self.assertEqual( columns.to_dicts(), many )
        self.assertEqual( columns.class_ids(), [ record['class_id'] for record in many ] )

    def test_anything_not_the_standard_shape_is_left_as_dicts( self ):
        base = records( 3 )
        cases = {
            'extra key': [ dict( base[0], person=True ) ] + base[1:],
            'missing key': [ { key: value for key, value in base[0].items() if key != 'size' } ] + base[1:],
            'int coordinate': [ dict( base[0], x=12 ) ] + base[1:],
            'bool size': [ dict( base[0], size=True ) ] + base[1:],
            'float size': [ dict( base[0], size=320.0 ) ] + base[1:],
            'non-str label': [ dict( base[0], class_id=3 ) ] + base[1:],
            'last record differs': base[:-1] + [ dict( base[-1], x=None ) ],
        }
        for name, content in cases.items():
            self.assertIsNone( bu_detections.to_columns( content ), name )

    def test_read_compacts_only_what_it_can( self ):
        with tempfile.TemporaryDirectory() as tmp:
            standard = write_cache( tmp, 'a.gz', records( 10 ) )
            odd = write_cache( tmp, 'b.gz', [ { 'class_id': 'x', 'x': 1 } ] )
            empty = write_cache( tmp, 'c.gz', [] )
            self.assertIsInstance( bu_detections.read( standard ), bu_detections.Columns )
            self.assertEqual( bu_detections.read( odd ), [ { 'class_id': 'x', 'x': 1 } ] )
            self.assertEqual( bu_detections.read( empty ), [] )

    def test_as_dicts_copies_a_kept_list( self ):
        held = [ { 'class_id': 'x', 'nested': [ 1 ] } ]
        copied = bu_detections.as_dicts( held, _file_hash='f' )
        copied[0]['nested'].append( 2 )
        self.assertEqual( held, [ { 'class_id': 'x', 'nested': [ 1 ] } ] )
        self.assertEqual( copied[0]['_file_hash'], 'f' )
        self.assertNotIn( '_file_hash', bu_detections.as_dicts( held )[0] )


class TestParseCache( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp()
        bu_detections.clear()

    def tearDown( self ):
        bu_detections.clear()
        import shutil
        shutil.rmtree( self.tmp, ignore_errors=True )

    def test_parsed_once_and_re_read_when_rewritten( self ):
        path = write_cache( self.tmp, 'a.gz', records( 5 ) )
        first = bu_detections.cached_read( path )
        self.assertIs( bu_detections.cached_read( path ), first )
        os.remove( path )
        bu_hash.write_json( records( 7, seed=2 ), path )
        stat = os.stat( path )
        os.utime( path, ( stat.st_atime, stat.st_mtime + 5 ) )
        again = bu_detections.cached_read( path )
        self.assertEqual( len( again ), 7 )
        self.assertEqual( again.to_dicts(), records( 7, seed=2 ) )

    def test_a_touched_file_is_re_read_even_at_the_same_size( self ):
        path = write_cache( self.tmp, 'a.gz', records( 5 ) )
        first = bu_detections.cached_read( path )
        stat = os.stat( path )
        os.utime( path, ( stat.st_atime, stat.st_mtime + 5 ) )
        self.assertIsNot( bu_detections.cached_read( path ), first )

    def test_read_once_uses_a_kept_parse_and_keeps_nothing_new( self ):
        kept = write_cache( self.tmp, 'kept.gz', records( 5 ) )
        fresh = write_cache( self.tmp, 'fresh.gz', records( 3, seed=4 ) )
        held = bu_detections.cached_read( kept )
        self.assertIs( bu_detections.read_once( kept ), held )
        once = bu_detections.read_once( fresh )
        self.assertEqual( once, records( 3, seed=4 ) )
        self.assertIsInstance( once, list )
        self.assertEqual( [ key[0] for key in bu_detections._PARSED ], [ kept ] )

    def test_forget_drops_only_the_named_files( self ):
        a = write_cache( self.tmp, 'a.gz', records( 2 ) )
        b = write_cache( self.tmp, 'b.gz', records( 2 ) )
        bu_detections.cached_read( a )
        bu_detections.cached_read( b )
        bu_detections.forget( [ a ] )
        self.assertEqual( [ key[0] for key in bu_detections._PARSED ], [ b ] )

    def test_an_unreadable_file_raises( self ):
        with self.assertRaises( Exception ):
            bu_detections.cached_read( os.path.join( self.tmp, 'missing.gz' ) )

    def test_replays_share_the_same_parse_and_keep_it_compact( self ):
        path = write_cache( self.tmp, 'a.gz', records( 20 ) )
        self.assertIs( bu_tuning._RAW_BOX_CACHE, bu_detections._PARSED )
        first = bu_tuning._read_raw_boxes( [ path ] )
        self.assertEqual( first, records( 20 ) )
        first[0]['x'] = 'poisoned'
        self.assertEqual( bu_tuning._read_raw_boxes( [ path ] ), records( 20 ) )
        self.assertEqual( bu_tuning.raw_boxes_for_reading( [ path ] ), records( 20 ) )
        held = list( bu_detections._PARSED.values() )
        self.assertEqual( len( held ), 1 )
        self.assertIsInstance( held[0], bu_detections.Columns )


class TestOneShotReaders( unittest.TestCase ):

    def test_label_content_reads_each_video_once_without_compacting( self ):
        # It reads every video once and forgets it; compacting first
        # costs about as much as parsing and buys nothing.
        from unittest import mock
        bu_detections.clear()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cache( tmp, 'a.gz', records( 30 ) )
            with mock.patch.object( bu_detections, 'to_columns', wraps=bu_detections.to_columns ) as compact:
                found = bu_tuning.content_by_video( { 'aaa': [ path ] }, set( LABELS ) )
            self.assertEqual( compact.call_count, 0 )
            self.assertTrue( found['aaa']['seconds'], "the video's detections were counted" )
            self.assertEqual( dict( bu_detections._PARSED ), {} )


class TestVideoCache( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp()
        bu_detections.clear()

    def tearDown( self ):
        bu_detections.clear()
        import shutil
        shutil.rmtree( self.tmp, ignore_errors=True )

    def test_one_video_at_a_time_tagged_and_concatenated( self ):
        a1 = write_cache( self.tmp, 'a1.gz', records( 4, seed=1 ) )
        a2 = write_cache( self.tmp, 'a2.gz', records( 3, seed=2 ) )
        b = write_cache( self.tmp, 'b.gz', records( 5, seed=3 ) )
        videos = bu_detections.VideoCache( { 'aaa': [ a1, a2 ], 'bbb': [ b ] } )
        self.assertEqual( videos.prime(), 12 )
        self.assertEqual( videos.detections, 12 )
        self.assertEqual( videos.file_hashes, [ 'aaa', 'bbb' ] )
        passes = []
        for _ in range( 2 ):
            seen = []
            for file_hash, boxes in videos:
                seen.append( ( file_hash, boxes ) )
                for box in boxes:
                    box['x'] = 'mutated'
            passes.append( seen )
        want_a = [ dict( r, _file_hash='aaa' ) for r in records( 4, seed=1 ) + records( 3, seed=2 ) ]
        want_b = [ dict( r, _file_hash='bbb' ) for r in records( 5, seed=3 ) ]
        self.assertEqual( [ fh for fh, _ in passes[1] ], [ 'aaa', 'bbb' ] )
        # Each pass gets fresh dicts, so the first pass's mutation never shows.
        fresh = list( videos )
        self.assertEqual( fresh, [ ( 'aaa', want_a ), ( 'bbb', want_b ) ] )

    def test_an_unreadable_cache_is_warned_once_and_skipped( self ):
        good = write_cache( self.tmp, 'good.gz', records( 3 ) )
        bad = os.path.join( self.tmp, 'bad.gz' )
        with open( bad, 'wb' ) as handle:
            handle.write( b'not gzip' )
        logger = logging.getLogger( 'test_video_cache' )
        videos = bu_detections.VideoCache( { 'aaa': [ good, bad ] }, logger )
        with self.assertLogs( logger, 'WARNING' ) as logged:
            self.assertEqual( videos.prime(), 3 )
            self.assertEqual( videos.prime(), 3 )
            self.assertEqual( len( list( videos )[0][1] ), 3 )
            list( videos )
        self.assertEqual( len( logged.records ), 1 )


class TestBetabenchReadsTheSameEitherWay( unittest.TestCase ):
    """A VideoCache and a plain list of tagged dicts give identical measurements."""

    @classmethod
    def setUpClass( cls ):
        import betabench
        cls.bench = betabench

    def setUp( self ):
        self.tmp = tempfile.mkdtemp()
        bu_detections.clear()

    def tearDown( self ):
        bu_detections.clear()
        import shutil
        shutil.rmtree( self.tmp, ignore_errors=True )

    def both( self ):
        discovered = {}
        flat = []
        for index, file_hash in enumerate( ( 'aaa', 'bbb', 'ccc' ) ):
            content = records( 600, seed=index + 10 )
            # Overlapping same-instant pairs, so suppression and dedup have work.
            for record in content[ ::7 ]:
                twin = dict( record, class_id='exposed_vulva' if record['class_id'] != 'exposed_vulva'
                             else 'exposed_breast', score=min( 1.0, record['score'] + 0.05 ),
                             x=record['x'] + 3.0, size=640 if record['size'] == 320 else 320 )
                content.append( twin )
            discovered[file_hash] = [ write_cache( self.tmp, file_hash + '.gz', content ) ]
            flat.extend( dict( record, _file_hash=file_hash ) for record in content )
        videos = bu_detections.VideoCache( discovered )
        videos.prime()
        return videos, flat

    def config( self ):
        import types
        import betaconfig
        backend = betaconfig.detector_backend['selected']
        return types.SimpleNamespace( backend_name=backend, label=backend, picture_sizes=[ 320, 640 ] )

    def test_each_cache_body( self ):
        args = self.bench.build_parser().parse_args( [ 'cache', '--min-samples', '5' ] )
        args.frame_width, args.frame_height = 1920, 1080
        logger = logging.getLogger( 'test_bench_equivalence' )
        logger.setLevel( logging.CRITICAL )
        videos, flat = self.both()
        for name in ( '_bench_suppression_one', '_bench_dedup_one', '_bench_hysteresis_one',
                      '_bench_geometry_one' ):
            body = getattr( self.bench, name )
            try:
                from_videos = body( args, logger, self.config(), videos )
                from_list = body( args, logger, self.config(), flat )
            except AttributeError as err:
                self.fail( "%s needs an argument this test does not give: %s"%( name, err ) )
            self.assertEqual( json.dumps( from_videos, sort_keys=True, default=str ),
                              json.dumps( from_list, sort_keys=True, default=str ), name )
            self.assertTrue( from_videos, name )


class TestStepMeter( unittest.TestCase ):

    METER = os.path.join( REPO_ROOT, 'tools', 'lib', 'step_meter.py' )

    def meter( self, *command ):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = os.path.join( tmp, 'step.meter.json' )
            result = subprocess.run( [ sys.executable, self.METER, '--record', record_path, '--' ] + list( command ),
                                     capture_output=True, text=True, timeout=60 )
            record = None
            if os.path.exists( record_path ):
                with open( record_path ) as handle:
                    record = json.load( handle )
            return result.returncode, record

    def test_exit_status_passes_through( self ):
        status, record = self.meter( sys.executable, '-c', 'import sys; sys.exit( 3 )' )
        self.assertEqual( status, 3 )
        self.assertEqual( record['exit'], 3 )
        self.assertIsNone( record['signal'] )
        self.assertGreater( record['wall_seconds'], 0 )

    def test_a_kill_reads_as_the_shell_does( self ):
        status, record = self.meter( sys.executable, '-c', 'import os, signal; os.kill( os.getpid(), signal.SIGKILL )' )
        self.assertEqual( status, 137 )
        self.assertEqual( ( record['exit'], record['signal'] ), ( 137, 9 ) )

    @unittest.skipIf( sys.platform == 'win32', 'no getrusage' )
    def test_the_peak_is_the_child_s( self ):
        _status, record = self.meter( sys.executable, '-c', 'x = bytearray( 200 * 1024 * 1024 ); x[::4096] = b"1" * len( x[::4096] )' )
        self.assertGreater( record['peak_rss_mb'], 200 )
        self.assertLess( record['peak_rss_mb'], 400 )

    def test_usage_and_missing_command( self ):
        for argv in ( [ 'oops' ], [ '--record', 'x', 'true', 'now' ], [ '--out', 'x', '--', 'true' ] ):
            result = subprocess.run( [ sys.executable, self.METER ] + argv, capture_output=True, text=True )
            self.assertEqual( result.returncode, 2, argv )
        status, record = self.meter( '/nonexistent/command' )
        self.assertEqual( status, 127 )
        self.assertIsNone( record )


if __name__ == '__main__':
    unittest.main()
