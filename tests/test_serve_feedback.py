"""
The feedback.* commands in process, every message checked against its
schema (tests.test_serve_job.Wire): a mark made on a result and read
back with all its fields, a preview's times on both timelines, a source
refused, bad params, removal; and the label enum in the schemas is the
engine's label list.
"""

import json
import os
import unittest

import betaconst
import betautils_cache_paths as bu_cache
import betautils_hash as bu_hash
import betautils_labels as bu_labels
import betautils_run_record as bu_run_record
import betautils_serve_catalog as bu_serve_catalog

from tests.test_serve_catalog import _ServeTest

KEYS = { 'detection': 'd1a', 'censor': 'c2b', 'encode': 'e3c' }
PREVIEW_KEYS = { 'detection': 'aa1', 'censor': 'bb2', 'encode': 'ff3' }
COMMANDS = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'protocol', 'v1', 'commands' )


class _Results( _ServeTest ):
    def setUp( self ):
        super().setUp()
        self.source = self.write( 'clip.mp4' )
        self.digest = bu_hash.md5_for_file( self.source, 16, use_memo=False )
        self.ok( 'catalog.add_folder', { 'path': self.watched } )
        self.changed_with( self.digest )
        os.makedirs( betaconst.video_path_censored, exist_ok=True )
        self.output = self.result( bu_cache.video_output_basename(
            'clip', self.digest, [ 640 ], 30, KEYS['detection'], KEYS['censor'], KEYS['encode'] ) )
        self.preview = self.result( bu_cache.video_output_basename(
            'clip', self.digest, [ 640 ], 30, PREVIEW_KEYS['detection'], PREVIEW_KEYS['censor'],
            PREVIEW_KEYS['encode'], preview_suffix=bu_cache.preview_cache_suffix( True, 90.0, 20, None ) ) )
        history = bu_serve_catalog.catalog_for( self.server ).history
        for path, keys, preview in ( ( self.output, KEYS, False ), ( self.preview, PREVIEW_KEYS, True ) ):
            record = bu_run_record.RunRecord( self.source, program='engine', job='j' )
            record.hash, record.output, record.keys, record.preview = self.digest, path, keys, preview
            history.record( record.end( 'processed', None ) )

    def result( self, base ):
        path = os.path.join( betaconst.video_path_censored, base + '.mp4' )
        with open( path, 'wb' ) as handle:
            handle.write( b'censored' )
        self.addCleanup( lambda: os.path.exists( path ) and os.remove( path ) )
        return path


class TestFeedback( _Results ):
    def test_a_mark_is_kept_with_every_field_and_read_back( self ):
        self.assertEqual( self.ok( 'feedback.list', { 'path': self.output } ),
                          { 'result': { 'item': self.digest, 'keys': KEYS, 'preview': False, 'offset': 0.0 },
                            'records': [], 'labels': list( bu_labels.LABELS ) } )
        made = self.ok( 'feedback.add', { 'path': self.output, 'time': 12.25, 'verdict': 'missed',
                                          'label': 'exposed_breast' } )
        self.assertEqual( ( made['time'], made['at_seconds'], made['label'], made['verdict'], made['keys'],
                            made['preview'] ), ( 12.25, 12.25, 'exposed_breast', 'missed', KEYS, False ) )
        listed = self.ok( 'feedback.list', { 'path': self.output } )['records']
        self.assertEqual( listed, [ made ] )

    def test_a_preview_mark_shows_on_the_preview_and_is_kept_on_the_source_timeline( self ):
        made = self.ok( 'feedback.add', { 'path': self.preview, 'time': 5, 'verdict': 'too_much', 'label': None } )
        self.assertEqual( ( made['time'], made['at_seconds'], made['preview'], made['keys'], made['label'] ),
                          ( 5.0, 95.0, True, PREVIEW_KEYS, None ) )
        listed = self.ok( 'feedback.list', { 'path': self.preview } )
        self.assertEqual( ( listed['result']['offset'], [ row['time'] for row in listed['records'] ] ), ( 90.0, [ 5.0 ] ) )
        self.assertEqual( self.ok( 'feedback.list', { 'path': self.output } )['records'], [] )

    def test_a_source_is_not_a_result( self ):
        self.assertEqual( self.ok( 'feedback.list', { 'path': self.source } ),
                          { 'result': None, 'records': [], 'labels': list( bu_labels.LABELS ) } )
        self.assertEqual( self.refused( 'feedback.add', { 'path': self.source, 'time': 1, 'verdict': 'missed',
                                                          'label': None }, 'feedback.not_a_result' ), {} )

    def test_remove( self ):
        made = self.ok( 'feedback.add', { 'path': self.output, 'time': 1, 'verdict': 'missed', 'label': None } )
        self.assertEqual( self.ok( 'feedback.remove', { 'id': made['id'] } ), {} )
        self.assertEqual( self.ok( 'feedback.list', { 'path': self.output } )['records'], [] )
        self.assertEqual( self.refused( 'feedback.remove', { 'id': made['id'] }, 'feedback.no_such_record' ),
                          { 'id': made['id'] } )

    def test_bad_params( self ):
        good = { 'path': self.output, 'time': 1, 'verdict': 'missed', 'label': None }
        cases = (
            ( 'feedback.list', {}, 'path', 'missing' ),
            ( 'feedback.list', { 'path': 3 }, 'path', 'type' ),
            ( 'feedback.list', { 'path': 'relative.mp4' }, 'path', 'value' ),
            ( 'feedback.add', dict( good, path='relative.mp4' ), 'path', 'value' ),
            ( 'feedback.add', { k: v for k, v in good.items() if k != 'time' }, 'time', 'missing' ),
            ( 'feedback.add', dict( good, time='1' ), 'time', 'type' ),
            ( 'feedback.add', dict( good, time=True ), 'time', 'type' ),
            ( 'feedback.add', dict( good, time=-1 ), 'time', 'value' ),
            ( 'feedback.add', { k: v for k, v in good.items() if k != 'verdict' }, 'verdict', 'missing' ),
            ( 'feedback.add', dict( good, verdict='fine' ), 'verdict', 'value' ),
            ( 'feedback.add', dict( good, verdict=1 ), 'verdict', 'type' ),
            ( 'feedback.add', { k: v for k, v in good.items() if k != 'label' }, 'label', 'missing' ),
            ( 'feedback.add', dict( good, label='exposed_elbow' ), 'label', 'value' ),
            ( 'feedback.add', dict( good, label=3 ), 'label', 'type' ),
            ( 'feedback.remove', {}, 'id', 'missing' ),
            ( 'feedback.remove', { 'id': '3' }, 'id', 'type' ),
            ( 'feedback.remove', { 'id': True }, 'id', 'type' ),
        )
        for name, data, field, reason in cases:
            with self.subTest( name=name, data=data ):
                self.assertEqual( self.refused( name, data, 'command.bad_params' ), { 'field': field, 'reason': reason } )
        self.assertEqual( self.ok( 'feedback.list', { 'path': self.output } )['records'], [] )


class TestSchemaLabels( unittest.TestCase ):
    def test_the_label_enum_is_the_engine_label_list_plus_not_sure( self ):
        for name in ( 'feedback.add', 'feedback.list' ):
            with open( os.path.join( COMMANDS, name + '.schema.json' ) ) as handle:
                text = handle.read()
            for enum in _label_enums( json.loads( text ) ):
                self.assertEqual( enum, [ None ] + list( bu_labels.LABELS ) )


def _label_enums( node ):
    if isinstance( node, dict ):
        if 'label' in node.get( 'properties', {} ):
            yield node['properties']['label']['enum']
        for value in node.values():
            yield from _label_enums( value )
    elif isinstance( node, list ):
        for value in node:
            yield from _label_enums( value )


if __name__ == '__main__':
    unittest.main()
