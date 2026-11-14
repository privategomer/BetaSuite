"""
test_person_window_analysis.py - the person-window attribution tally and
tools/analysis/analyze_person_windows.py.

The tool answers "how often does reverse mode find a man, and why not
the rest", which decides whether evidence attribution is enough or a
person model that tells men from women itself is worth trying. Its
numbers come from the render's own attribution
(male_people_by_time) through an optional tally, so these pin that the
tally counts what the windows actually did, that passing it changes no
window, and that the tool reads real cache files end to end.
"""

import collections
import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import betautils_hash as bu_hash
import betautils_person as bu_person
from tests.test_person_windows import Harness, _person, _raw

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        '_person_windows_tool', os.path.join( REPO_ROOT, 'tools', 'analysis', 'analyze_person_windows.py' ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


def _evidence( raw ):
    found = {}
    for box in raw:
        found.setdefault( round( box['t'], 3 ), [] ).append( box )
    return found


class TestTally( Harness ):

    def setUp( self ):
        super().setUp()
        self.set_reverse()

    def run_tally( self, sample_times, people, raw ):
        tally = collections.Counter()
        windows = bu_person.male_people_by_time( sample_times, people, _evidence( raw ), tally )
        self.assertEqual( windows, bu_person.male_people_by_time( sample_times, people, _evidence( raw ) ),
                          "the tally must not change any window" )
        return windows, tally

    def test_one_sample_split_by_attribution_and_where_evidence_fell( self ):
        man = _person( 0, 0, 100, 200 )
        woman = _person( 200, 0, 100, 200 )
        nobody = _person( 400, 0, 100, 200 )
        overlap_a, overlap_b = _person( 0, 300, 100, 50 ), _person( 50, 300, 100, 50 )
        raw = [ _raw( 'face_masc', 30, 10, 40, 40 ),          # inside the man only: votes
                _raw( 'face_femme', 230, 10, 40, 40 ),        # inside the woman only: votes
                _raw( 'exposed_chest', 70, 310, 20, 20 ),     # inside both overlapping outlines: votes for nobody
                _raw( 'exposed_chest', 600, 300, 20, 20 ) ]   # outside every outline
        windows, tally = self.run_tally( [ 0.0, 0.1 ], [ man, woman, nobody, overlap_a, overlap_b ], raw )
        self.assertEqual( windows[0.0]['show'], [ man ] )
        self.assertEqual( tally['samples'], 2 )
        self.assertEqual( tally['samples_with_people'], 1 )
        self.assertEqual( tally['samples_with_man_shown'], 1 )
        self.assertEqual( tally['outlines'], 5 )
        self.assertEqual( tally['outlines_shown'], 1 )
        self.assertEqual( tally['outlines_male'], 1 )
        self.assertEqual( tally['outlines_female'], 1 )
        self.assertEqual( tally['outlines_no_evidence'], 3 )
        self.assertEqual( tally['outlines_backstop_hidden'], 0 )
        self.assertEqual( ( tally['evidence_male_one'], tally['evidence_male_shared'],
                            tally['evidence_male_outside'] ), ( 1, 1, 1 ) )
        self.assertEqual( ( tally['evidence_female_one'], tally['evidence_female_shared'],
                            tally['evidence_female_outside'] ), ( 1, 0, 0 ) )
        self.assertEqual( ( tally['tracks'], tally['tracks_male'], tally['tracks_female'],
                            tally['tracks_no_evidence'], tally['tracks_mixed'] ), ( 5, 1, 1, 3, 0 ) )

    def test_the_backstop_and_mixed_votes_are_counted_apart( self ):
        # A man tracked over three samples, his face seen in two; in the
        # third a breast is inside his outline and no separate outline
        # holds it: a woman merged into his outline, so the backstop
        # hides him there. Elsewhere, one person with a male and a female
        # vote is mixed, which is not male.
        merged = [ _person( 0, 0, 200, 200, t=t ) for t in ( 0.0, 0.1, 0.2 ) ]
        mixed = _person( 400, 0, 100, 200, t=0.0 )
        raw = [ _raw( 'face_masc', 20, 10, 40, 40, t=0.0 ), _raw( 'face_masc', 20, 10, 40, 40, t=0.1 ),
                _raw( 'exposed_breast', 130, 100, 30, 30, t=0.2 ),
                _raw( 'face_masc', 420, 10, 40, 40, t=0.0 ), _raw( 'face_femme', 430, 100, 40, 40, t=0.0 ) ]
        windows, tally = self.run_tally( [ 0.0, 0.1, 0.2 ], merged + [ mixed ], raw )
        self.assertEqual( [ windows[t]['show'] for t in ( 0.0, 0.1, 0.2 ) ], [ [ merged[0] ], [ merged[1] ], [] ] )
        self.assertEqual( tally['outlines_male'], 3 )
        self.assertEqual( tally['outlines_backstop_hidden'], 1 )
        self.assertEqual( tally['outlines_shown'], 2 )
        self.assertEqual( tally['samples_with_man_shown'], 2 )
        self.assertEqual( tally['outlines_mixed'], 1 )
        self.assertEqual( ( tally['tracks'], tally['tracks_male'], tally['tracks_mixed'] ), ( 2, 1, 1 ) )

    def test_attribution_follows_the_whole_track( self ):
        samples = [ _person( 10 * k, 0, 100, 200, t=round( 0.1 * k, 3 ) ) for k in range( 3 ) ]
        _windows, tally = self.run_tally( [ p['t'] for p in samples ], samples,
                                          [ _raw( 'face_masc', 30, 10, 40, 40, t=0.0 ) ] )
        self.assertEqual( tally['tracks'], 1 )
        self.assertEqual( tally['outlines_male'], 3, "a man stays a man where his face is not seen" )
        self.assertEqual( tally['outlines_shown'], 3 )
        self.assertEqual( tally['samples_with_man_shown'], 3 )


class TestTool( Harness ):

    def setUp( self ):
        super().setUp()
        self.set_reverse()
        self.tool = _load_tool()
        self.tmp = tempfile.mkdtemp()

    def tearDown( self ):
        shutil.rmtree( self.tmp, ignore_errors=True )
        super().tearDown()

    def test_the_cache_name_is_parsed_with_and_without_a_preview_tail( self ):
        parse = self.tool.parse_person_cache_name
        self.assertEqual( parse( 'abc123-person-yolo11_seg-9-p0f0f0f.npz' ),
                          { 'hash': 'abc123', 'backend': 'yolo11_seg', 'fps': 9.0, 'key': '0f0f0f', 'suffix': '' } )
        self.assertEqual( parse( 'abc123-person-yolo11_seg-18-p0f0f0f-preview@600.0s.npz' )['suffix'],
                          '-preview@600.0s' )
        self.assertEqual( parse( 'abc123-person-yolo11_seg-9-p0f0f0f-preview.npz' )['suffix'], '-preview' )
        self.assertEqual( parse( 'abc123-person-yolo11_seg-9-p0f0f0f-preview@600.0s+20.0s.npz' )['suffix'],
                          '-preview@600.0s+20.0s' )
        self.assertEqual( parse( 'abc123-person-yolo11_seg-9-p0f0f0f-preview+all.npz' )['suffix'],
                          '-preview+all' )
        self.assertIsNone( parse( 'abc123-person-yolo11_seg-9-p0f0f0f.npz.checkpoint.npz' ) )
        self.assertIsNone( parse( 'abc123-nudenet_v3-3-640-9-0.120-df43d00.gz' ) )

    def test_a_legacy_preview_name_with_a_named_twin_is_read_once( self ):
        names = [ 'abc123-person-yolo11_seg-9-p0f0f0f-preview.npz',
                  'abc123-person-yolo11_seg-9-p0f0f0f-preview+20.0s.npz',
                  'abc123-person-yolo11_seg-9-p0f0f0f-preview@600.0s.npz',
                  'def456-person-yolo11_seg-9-p0f0f0f-preview.npz' ]
        for name in names:
            open( os.path.join( self.tmp, name ), 'w' ).close()
        kept = sorted( os.path.basename( path ) for path, _parsed in self.tool.person_caches( self.tmp ) )
        self.assertEqual( kept, sorted( names[ 1: ] ),
                          "only the lengthless name at the offset a named cache covers goes" )

    def test_end_to_end_from_cache_files( self ):
        man = _person( 0, 0, 100, 200, t=0.0 )
        nobody = _person( 300, 0, 100, 200, t=0.0 )
        person_path = os.path.join( self.tmp, 'aaaa-person-yolo11_seg-9-p0f0f0f.npz' )
        bu_person.save_cache( person_path, [ 0.0, 1 / 9.0 ], [ man, nobody ] )
        with open( os.path.join( self.tmp, 'aaaa-person-yolo11_seg-9-p0f0f0f.npz.checkpoint.npz' ), 'wb' ) as handle:
            handle.write( b'not a finished cache' )
        detections = [ _raw( 'face_masc', 30, 10, 40, 40 ), _raw( 'exposed_chest', 700, 300, 20, 20 ) ]
        caches = self.tool.person_caches( self.tmp )
        self.assertEqual( [ os.path.basename( path ) for path, _parsed in caches ],
                          [ 'aaaa-person-yolo11_seg-9-p0f0f0f.npz' ] )
        warnings = []
        with mock.patch.object( self.tool.bu_cache, 'load_cached_raw_boxes', return_value=( detections, [] ) ), \
             mock.patch( 'betautils_track.filter_raw_detections',
                         side_effect=lambda raw_boxes, *a, **k: ( raw_boxes, {} ) ):
            results = self.tool.analyse( caches, 'nudenet_v3', [ 640 ], 0.12, { 'aaaa': '/v/clip.mp4' },
                                         lambda path: ( 1280, 720 ), warnings.append )
        self.assertEqual( warnings, [] )
        ( label, group ), = results.items()
        self.assertTrue( label.startswith( 'yolo11_seg' ) )
        tally = group['tally']
        self.assertEqual( ( tally['outlines'], tally['outlines_shown'], tally['outlines_no_evidence'] ), ( 2, 1, 1 ) )
        self.assertEqual( tally['evidence_male_outside'], 1 )
        row, = group['videos']
        self.assertEqual( row['name'], 'clip.mp4' )
        self.assertAlmostEqual( row['unattributed_person_seconds'], round( 1 / 9.0, 1 ) )
        self.assertEqual( ( row['male_evidence_outside'], row['male_evidence'] ), ( 1, 2 ) )
        text = '\n'.join( self.tool.report( results, 5, bu_person.bu_reverse.settings()['person_windows'] ) )
        self.assertIn( 'frames where a man is shown        1  (100% of frames with people)', text )
        self.assertIn( 'hidden: no evidence at all', text )
        self.assertIn( '50% of male detections fell outside every outline', text )

    def test_a_cache_without_its_video_or_detections_is_skipped_and_said( self ):
        path = os.path.join( self.tmp, 'bbbb-person-yolo11_seg-9-p0f0f0f.npz' )
        bu_person.save_cache( path, [ 0.0 ], [ _person( 0, 0, 10, 10 ) ] )
        caches = self.tool.person_caches( self.tmp )
        warnings = []
        self.assertEqual( self.tool.analyse( caches, 'nudenet_v3', [ 640 ], 0.12, {}, None, warnings.append ), {} )
        self.assertIn( 'source video not found', warnings[0] )
        warnings = []
        with mock.patch.object( self.tool.bu_cache, 'load_cached_raw_boxes', return_value=( [], [ 640 ] ) ):
            self.assertEqual( self.tool.analyse( caches, 'nudenet_v3', [ 640 ], 0.12, { 'bbbb': '/v/x.mp4' },
                                                 lambda path: ( 640, 360 ), warnings.append ), {} )
        self.assertIn( 'no nudenet_v3 detection cache at size(s) [640]', warnings[0] )

    def test_a_preview_cache_reads_its_own_slice_and_names_its_model( self ):
        path = os.path.join( self.tmp, 'cccc-person-yolo11_seg-9-p0f0f0f-preview@600.0s.npz' )
        bu_person.save_cache( path, [ 0.0 ], [ _person( 0, 0, 100, 200 ) ] )
        manifests = os.path.join( self.tmp, 'run_keys' )
        os.makedirs( manifests )
        bu_hash.write_json_plain( { 'kind': 'person', 'key': '0f0f0f', 'identity': { 'model_variant': 's' } },
                                  os.path.join( manifests, 'person-0f0f0f.json' ) )
        with mock.patch.object( self.tool.betaconst, 'run_key_dir', manifests ), \
             mock.patch.object( self.tool.bu_cache, 'load_cached_raw_boxes',
                                return_value=( [ _raw( 'face_masc', 30, 10, 40, 40 ) ], [] ) ) as load, \
             mock.patch( 'betautils_track.filter_raw_detections',
                         side_effect=lambda raw_boxes, *a, **k: ( raw_boxes, {} ) ):
            results = self.tool.analyse( self.tool.person_caches( self.tmp ), 'nudenet_v3', [ 640 ], 0.12,
                                         { 'cccc': '/v/clip.mp4' }, lambda path: ( 1280, 720 ), self.fail )
        self.assertEqual( load.call_args.args[5], '-preview@600.0s', "the person cache's own slice" )
        self.assertEqual( list( results ), [ 'yolo11_seg s' ] )
        self.assertEqual( results['yolo11_seg s']['videos'][0]['name'], 'clip.mp4-preview@600.0s' )

    def test_backstop_hidden_men_are_not_counted_as_shown( self ):
        tally = collections.Counter( outlines_male=3, outlines_shown=2, outlines_backstop_hidden=1 )
        self.assertEqual( self.tool.shares( tally )['shown'], 2 )
        self.assertEqual( self.tool.shares( tally )['backstop_hidden'], 1 )

    def test_no_caches_says_how_to_make_them( self ):
        text = '\n'.join( self.tool.report( {}, 5, {} ) )
        self.assertIn( '--reverse on', text )

    def test_the_hub_runs_it( self ):
        with open( os.path.join( REPO_ROOT, 'tools', 'analysis', 'run_all_analysis.sh' ) ) as handle:
            hub = handle.read()
        self.assertIn( 'analyze_person_windows.py --json $LOG_DIR/person_windows.json', hub )


if __name__ == '__main__':
    unittest.main()
