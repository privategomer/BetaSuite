"""
test_model_key.py - the model file's content is part of the cache keys.

What each test prevents:
  - a different file at the same path (an import, a hand-copied export)
    reusing detections or person masks made by the old file
  - keys that change when a genuine model is installed, or that cannot
    be worked out before it is (CI, the Models page before a download)
  - a backend whose model is not in its key at all (RetinaNet's identity
    used to be empty)
"""

import os
import unittest
from unittest import mock

import betautils_cache_paths as bu_cache_paths
import betautils_models as bu_models


class TestFileIdentity( unittest.TestCase ):

    def setUp( self ):
        self.row = bu_models.entry( 'nudenet-320n' )
        self.path = bu_models.install_path( self.row )
        os.makedirs( os.path.dirname( self.path ), exist_ok=True )
        self.assertFalse( os.path.exists( self.path ), 'the suite must not see real models' )
        self.addCleanup( lambda: os.path.exists( self.path ) and os.remove( self.path ) )

    def _write( self, data, mtime ):
        with open( self.path, 'wb' ) as out:
            out.write( data )
        os.utime( self.path, ns=( mtime, mtime ) )

    def test_missing_catalog_file_keys_on_the_pinned_hash( self ):
        self.assertEqual( bu_models.file_identity( bu_models.config_path_for( self.row ) ), self.row['sha256'] )

    def test_missing_custom_file( self ):
        self.assertEqual( bu_models.file_identity( '/nowhere/custom.onnx' ), bu_models.MISSING_FILE )
        self.assertEqual( bu_models.file_identity( None ), bu_models.MISSING_FILE )

    def test_a_file_present_keys_on_its_content( self ):
        self._write( b'one', 1_000_000_000 )
        first = bu_models.file_identity( bu_models.config_path_for( self.row ) )
        self.assertNotEqual( first, self.row['sha256'] )
        self._write( b'two', 2_000_000_000 )
        self.assertNotEqual( bu_models.file_identity( bu_models.config_path_for( self.row ) ), first )

    def test_detection_key_moves_when_the_file_is_swapped( self ):
        env = { 'BETASUITE_DETECTOR_BACKEND_OVERRIDE': 'nudenet_v3', 'BETASUITE_DETECTOR_VARIANT_OVERRIDE': '320n' }
        with mock.patch.dict( os.environ, env ):
            missing = bu_cache_paths.detection_key()
            self._write( b'not the real model', 3_000_000_000 )
            swapped = bu_cache_paths.detection_key()
        self.assertNotEqual( missing, swapped )

    def test_retinanet_key_carries_its_model( self ):
        import detectors.retinanet_v2 as retinanet_v2
        self.assertEqual( retinanet_v2.detection_identity( {} ),
                          { 'model_sha256': bu_models.entry( 'nudenet-retinanet' )['sha256'] } )

    def test_person_identities_carry_their_model( self ):
        import segmenters.rf_detr_seg as rf_detr_seg
        import segmenters.yolo11_seg as yolo11_seg
        self.assertEqual( yolo11_seg.identity( { 'model_variant': 'm' } )['model_sha256'],
                          bu_models.entry( 'yolo11m-seg' )['sha256'] )
        self.assertEqual( rf_detr_seg.identity( { 'model_variant': 'nano' } )['model_sha256'],
                          bu_models.entry( 'rf-detr-seg-nano' )['sha256'] )


if __name__ == '__main__':
    unittest.main()
