"""
test_models_catalog.py - the model catalog and the hardware default.

What each test prevents:
  - a shipped catalog that a consumer would trip on (bad hash, repeated
    file name, path in a file name, unknown license, plain-http mirror)
  - a model an adapter can load that the Models page cannot install, or
    a catalog row no adapter would ever load
  - the hardware default picking 640m on a CPU-only machine, or 320n
    when a usable GPU is there; a GPU counted as usable from only one of
    the two signals (CUDA provider, nvidia-smi)
  - the page showing the wrong model as in use
"""

import copy
import os
import unittest
from unittest import mock

import betautils_hardware as bu_hardware
import betautils_models as bu_models


def _shipped():
    return copy.deepcopy( bu_models.load() )


class TestCatalogShape( unittest.TestCase ):

    def test_shipped_catalog_validates( self ):
        bu_models.validate( _shipped() )

    def _rejects( self, change ):
        catalog = _shipped()
        change( catalog )
        with self.assertRaises( bu_models.CatalogError ):
            bu_models.validate( catalog )

    def test_rejects_repeated_id( self ):
        self._rejects( lambda c: c['models'].append( dict( c['models'][0], file='other.onnx', sources=[] ) ) )

    def test_rejects_bad_hash( self ):
        self._rejects( lambda c: c['models'][0].update( sha256='ABC' ) )

    def test_rejects_path_in_file_name( self ):
        self._rejects( lambda c: c['models'][0].update( file='../evil.onnx' ) )

    def test_rejects_repeated_file_name( self ):
        self._rejects( lambda c: c['models'][1].update( file=c['models'][0]['file'] ) )

    def test_rejects_source_named_like_a_model( self ):
        self._rejects( lambda c: c['models'][0]['sources'][0].update( file=c['models'][1]['file'] ) )

    def test_rejects_repeated_source_file_name( self ):
        self._rejects( lambda c: c['models'][1]['sources'][0].update( file=c['models'][0]['sources'][0]['file'] ) )

    def test_rejects_short_hash( self ):
        self._rejects( lambda c: c['models'][0].update( sha256='a' * 63 ) )

    def test_rejects_unknown_license( self ):
        self._rejects( lambda c: c['models'][0].update( license='nope' ) )

    def test_rejects_plain_http_mirror( self ):
        self._rejects( lambda c: c.update( mirrors=[ 'http://example.invalid/' ] ) )

    def test_rejects_mirror_without_trailing_slash( self ):
        self._rejects( lambda c: c.update( mirrors=[ 'https://example.invalid/models' ] ) )

    def test_rejects_unknown_format( self ):
        self._rejects( lambda c: c.update( format=99 ) )

    def test_urls_try_each_mirror_then_extras( self ):
        catalog = _shipped()
        catalog['mirrors'] = [ 'https://a.invalid/x/', 'https://b.invalid/' ]
        row = dict( catalog['models'][0], urls=[ 'https://c.invalid/file' ] )
        self.assertEqual( bu_models.urls( row, catalog ),
                          [ 'https://a.invalid/x/' + row['file'], 'https://b.invalid/' + row['file'],
                            'https://c.invalid/file' ] )

    def test_every_model_mirrors_its_source( self ):
        # AGPL 6d: the preferred form for modification is hosted beside
        # the binary, so every row names at least one source file.
        for row in bu_models.entries():
            self.assertTrue( row.get( 'sources' ), row['id'] )


class TestCatalogMatchesAdapters( unittest.TestCase ):
    """Every file an adapter can load is installable, and nothing else is listed."""

    def _adapter_paths( self ):
        import detectors.nudenet_v3 as nudenet_v3
        import detectors.retinanet_v2 as retinanet_v2
        import segmenters.rf_detr_seg as rf_detr_seg
        import segmenters.yolo11_seg as yolo11_seg
        paths = { ( 'retinanet_v2', None ): retinanet_v2.MODEL_PATH }
        for backend, module in ( ( 'nudenet_v3', nudenet_v3 ), ( 'yolo11_seg', yolo11_seg ),
                                 ( 'rf_detr_seg', rf_detr_seg ) ):
            for variant, spec in module.VARIANTS.items():
                paths[( backend, variant )] = spec['model_path']
        return paths

    def test_one_row_per_adapter_model( self ):
        paths = self._adapter_paths()
        rows = { ( row['backend'], row.get( 'variant' ) ): row for row in bu_models.entries() }
        self.assertEqual( set( rows ), set( paths ) )
        for key, path in paths.items():
            self.assertEqual( bu_models.config_path_for( rows[key] ), path, key )

    def test_kinds_follow_backends( self ):
        for row in bu_models.entries():
            expected = 'person' if row['backend'] in ( 'yolo11_seg', 'rf_detr_seg' ) else 'detector'
            self.assertEqual( row['kind'], expected, row['id'] )


class TestHardwareDefault( unittest.TestCase ):

    def test_gpu_machine_gets_640m( self ):
        self.assertEqual( bu_models.recommended_detector( { 'cuda_usable': True } ), ( 'nudenet-640m', 'gpu' ) )

    def test_cpu_machine_gets_320n( self ):
        self.assertEqual( bu_models.recommended_detector( { 'cuda_usable': False } ), ( 'nudenet-320n', 'cpu' ) )
        self.assertEqual( bu_models.recommended_detector( None ), ( 'nudenet-320n', 'cpu' ) )

    def test_recommendations_are_catalog_detectors( self ):
        for model_id in ( bu_models.GPU_DETECTOR, bu_models.CPU_DETECTOR ):
            self.assertEqual( bu_models.entry( model_id )['kind'], 'detector' )

    def test_cuda_usable_needs_both_signals( self ):
        cuda = [ 'CUDAExecutionProvider', 'CPUExecutionProvider' ]
        self.assertTrue( bu_hardware.cuda_usable( cuda, { 'name': 'GPU' } ) )
        self.assertFalse( bu_hardware.cuda_usable( cuda, {} ) )
        self.assertFalse( bu_hardware.cuda_usable( [ 'CPUExecutionProvider' ], { 'name': 'GPU' } ) )
        self.assertFalse( bu_hardware.cuda_usable( None, None ) )

    def test_nvidia_smi_parses_first_gpu( self ):
        run = mock.Mock( return_value=( 0, 'Card A, 550.1\nCard B, 550.1\n' ) )
        self.assertEqual( bu_hardware.nvidia_smi( [ 'name', 'driver_version' ], run=run ),
                          { 'name': 'Card A', 'driver_version': '550.1' } )

    def test_nvidia_smi_missing_is_empty( self ):
        self.assertEqual( bu_hardware.nvidia_smi( [ 'name' ], run=lambda _c: ( None, 'not found' ) ), {} )
        self.assertEqual( bu_hardware.nvidia_smi( [ 'name' ], run=lambda _c: ( 9, 'error' ) ), {} )

    def test_detect_combines_signals( self ):
        with mock.patch.object( bu_hardware, 'onnx_providers', return_value=( 'CUDAExecutionProvider', ) ), \
             mock.patch.object( bu_hardware, 'nvidia_smi', return_value={ 'name': 'GPU' } ):
            self.assertTrue( bu_hardware.detect()['cuda_usable'] )
        with mock.patch.object( bu_hardware, 'onnx_providers', return_value=( 'CUDAExecutionProvider', ) ), \
             mock.patch.object( bu_hardware, 'nvidia_smi', return_value={} ):
            found = bu_hardware.detect()
            self.assertFalse( found['cuda_usable'] )
            self.assertIsNone( found['gpu'] )


class TestInUse( unittest.TestCase ):

    def test_catalog_paths_map_to_rows( self ):
        with mock.patch.object( bu_models, 'detector_path_in_use', return_value='../resources/model/v3.4-640m.onnx' ), \
             mock.patch.object( bu_models, 'person_path_in_use', return_value='../resources/model/yolo11s-seg.onnx' ):
            self.assertEqual( bu_models.in_use(), { 'detector': 'nudenet-640m', 'person': 'yolo11s-seg', 'custom': {} } )

    def test_custom_path_is_reported( self ):
        with mock.patch.object( bu_models, 'detector_path_in_use', return_value='/somewhere/mine.onnx' ), \
             mock.patch.object( bu_models, 'person_path_in_use', return_value=None ):
            used = bu_models.in_use()
        self.assertIsNone( used['detector'] )
        self.assertEqual( used['custom'], { 'detector': '/somewhere/mine.onnx' } )

    def test_follows_the_engine_variant_resolution( self ):
        env = { 'BETASUITE_DETECTOR_BACKEND_OVERRIDE': 'nudenet_v3',
                'BETASUITE_DETECTOR_VARIANT_OVERRIDE': '640m',
                'BETASUITE_PERSON_BACKEND_OVERRIDE': 'rf_detr_seg',
                'BETASUITE_PERSON_VARIANT_OVERRIDE': 'medium' }
        import betautils_detector as bu_detector
        with mock.patch.dict( os.environ, env ), \
             mock.patch.object( bu_detector, 'get_backend_config', return_value={} ), \
             mock.patch( 'betautils_person.get_backend_config', return_value={} ):
            used = bu_models.in_use()
        self.assertEqual( ( used['detector'], used['person'] ), ( 'nudenet-640m', 'rf-detr-seg-medium' ) )

    def test_retinanet_has_no_variant( self ):
        env = { 'BETASUITE_DETECTOR_BACKEND_OVERRIDE': 'retinanet_v2' }
        with mock.patch.dict( os.environ, env ):
            self.assertEqual( bu_models.in_use()['detector'], 'nudenet-retinanet' )


if __name__ == '__main__':
    unittest.main()
