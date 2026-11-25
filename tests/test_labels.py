"""
test_labels.py - the label table, what each backend declares, and what the app is told.

betautils_labels describes every label once; betaconst.classes, the
settings registry and the app read it. Each detector adapter declares the
labels it can report (SUPPORTED_LABELS), and a label the selected backend
cannot report reaches the app as unavailable, with a code.
"""

import os
import unittest
from unittest import mock

import betaconst
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_labels as bu_labels
import betautils_settings as bu_settings
from detectors import nudenet_v3, retinanet_v2

# betaconst.classes' order before the table existed; the debug overlay iterates it
LEGACY_ORDER = [
    'exposed_anus', 'exposed_armpits', 'covered_belly', 'exposed_belly', 'covered_buttocks',
    'exposed_buttocks', 'face_femme', 'face_masc', 'covered_feet', 'exposed_feet', 'covered_breast',
    'exposed_breast', 'covered_vulva', 'exposed_vulva', 'exposed_chest', 'exposed_penis',
    'covered_armpits', 'covered_anus',
]
NUDENET_ONLY = { 'covered_armpits', 'covered_anus' }


def _backend( name ):
    return mock.patch.dict( os.environ, { 'BETASUITE_DETECTOR_BACKEND_OVERRIDE': name } )


class TableTest( unittest.TestCase ):

    def test_betaconst_classes_come_from_the_table_unchanged( self ):
        self.assertEqual( list( betaconst.classes ), LEGACY_ORDER )
        self.assertEqual( betaconst.classes, bu_labels.debug_colors() )
        self.assertEqual( betaconst.classes['exposed_vulva'], ( 255, 0, 255 ) )

    def test_rows_use_known_values( self ):
        for label, row in bu_labels.LABELS.items():
            self.assertEqual( row['id'], label )
            self.assertIn( row['part'], bu_labels.PARTS, label )
            self.assertIn( row['state'], bu_labels.STATES, label )
            self.assertIn( row['sex'], bu_labels.SEXES, label )
            self.assertIn( row['view'], bu_labels.VIEWS, label )
            if row['state'] != 'none':
                self.assertTrue( label.startswith( row['state'] + '_' ), label )

    def test_display_order_follows_the_body( self ):
        order = [ row['id'] for row in bu_labels.describe() ]
        self.assertEqual( sorted( order ), sorted( LEGACY_ORDER ) )
        self.assertEqual( order[:2], [ 'face_femme', 'face_masc' ] )
        self.assertLess( order.index( 'exposed_breast' ), order.index( 'covered_breast' ) )
        self.assertLess( order.index( 'exposed_belly' ), order.index( 'exposed_feet' ) )
        self.assertEqual( order[-2:], [ 'exposed_feet', 'covered_feet' ] )

    def test_describe_returns_copies( self ):
        bu_labels.describe()[0]['part'] = 'changed'
        self.assertNotEqual( bu_labels.describe()[0]['part'], 'changed' )

    def test_registry_label_choices_are_the_table( self ):
        self.assertEqual( bu_settings.label_names(), sorted( bu_labels.LABELS ) )


class BackendLabelsTest( unittest.TestCase ):

    def test_adapters_declare_known_labels( self ):
        for name in bu_detector.registered_backend_names():
            declared = bu_detector.supported_labels( name )
            self.assertTrue( declared, name )
            self.assertLessEqual( declared, set( bu_labels.LABELS ), name )

    def test_nudenet_reports_every_label( self ):
        self.assertEqual( bu_detector.supported_labels( 'nudenet_v3' ), set( bu_labels.LABELS ) )
        self.assertEqual( nudenet_v3.SUPPORTED_LABELS, set( bu_labels.LABELS ) )

    def test_retinanet_lacks_the_two_covered_labels( self ):
        self.assertEqual( set( bu_labels.LABELS ) - bu_detector.supported_labels( 'retinanet_v2' ), NUDENET_ONLY )
        self.assertEqual( retinanet_v2.SUPPORTED_LABELS, set( retinanet_v2._MODEL_CLASS_ORDER ) )

    def test_unknown_backend_supports_nothing( self ):
        self.assertEqual( bu_detector.supported_labels( 'no_such_backend' ), frozenset() )

    def test_default_is_the_selected_backend( self ):
        with _backend( 'retinanet_v2' ):
            self.assertEqual( bu_detector.supported_labels(), bu_detector.supported_labels( 'retinanet_v2' ) )
        with _backend( 'nudenet_v3' ):
            self.assertEqual( bu_detector.supported_labels(), bu_detector.supported_labels( 'nudenet_v3' ) )


class AppViewTest( unittest.TestCase ):

    def _views( self, name ):
        with _backend( name ):
            return { view['label']: view for view in bu_config.label_views() }

    def test_retinanet_marks_two_labels_unavailable_with_a_code( self ):
        views = self._views( 'retinanet_v2' )
        for label, view in views.items():
            if label in NUDENET_ONLY:
                self.assertFalse( view['available'], label )
                self.assertEqual( view['unavailable'], { 'code': 'label.not_in_model',
                                                         'params': { 'backend': 'retinanet_v2' } } )
            else:
                self.assertTrue( view['available'], label )
                self.assertNotIn( 'unavailable', view, label )

    def test_nudenet_has_every_label_available( self ):
        views = self._views( 'nudenet_v3' )
        self.assertTrue( all( view['available'] for view in views.values() ) )
        self.assertFalse( any( 'unavailable' in view for view in views.values() ) )

    def test_explicit_backend_wins_over_selected( self ):
        with _backend( 'nudenet_v3' ):
            views = { v['label']: v for v in bu_config.label_views( 'retinanet_v2' ) }
        self.assertFalse( views['covered_anus']['available'] )

    def test_schema_lists_labels_and_backends( self ):
        schema = bu_settings.schema()
        self.assertEqual( schema['labels'], bu_labels.describe() )
        self.assertEqual( set( schema['backends'] ), set( bu_detector.registered_backend_names() ) )
        self.assertEqual( set( schema['backends']['nudenet_v3']['labels'] ) - set( schema['backends']['retinanet_v2']['labels'] ),
                          NUDENET_ONLY )
        self.assertEqual( schema['backends']['nudenet_v3']['labels'], sorted( schema['backends']['nudenet_v3']['labels'] ) )


if __name__ == '__main__':
    unittest.main()
