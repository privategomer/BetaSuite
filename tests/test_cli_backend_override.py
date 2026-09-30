"""
test_cli_backend_override.py - regression coverage for the --backend
CLI flag (added 2026-09-17, per-user request: "the model backend should
be configurable via config file, env var, or cli, with normal
precedence rules"). BETASUITE_DETECTOR_BACKEND_OVERRIDE (env var) and
betaconfig.detector_backend['selected'] (config file) already existed;
this covers the new CLI tier and its precedence against both.

Precedence, highest wins: env var > --backend CLI flag > betaconfig.py.
See betautils_detector.selected_backend_name()'s own docstring for the
env-var-over-config-file half (unchanged by this work) and
betautils_cli.apply_cli_overrides' --backend special case for how the
CLI flag writes into betaconfig.detector_backend['selected'] so
selected_backend_name()'s existing env-var-first check naturally wins
over it without selected_backend_name() itself needing to change.
"""

import os
import unittest
from unittest import mock

import betaconfig
import betautils_cli as bu_cli
import betautils_detector as bu_detector


class TestBackendCliFlag( unittest.TestCase ):

    def setUp( self ):
        self._had_attr = hasattr( betaconfig, 'detector_backend' )
        if self._had_attr:
            self._original = betaconfig.detector_backend
        betaconfig.detector_backend = { 'selected': 'retinanet_v2', 'retinanet_v2': {}, 'nudenet_v3': {} }

    def tearDown( self ):
        if self._had_attr:
            betaconfig.detector_backend = self._original
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend

    def _parse_and_apply( self, cli_args ):
        parser = bu_cli.build_arg_parser( 'test', include_preview=True, include_logging=True )
        args = parser.parse_args( cli_args )
        return bu_cli.apply_cli_overrides( betaconfig, args )

    def test_backend_flag_overrides_config_file( self ):
        self._parse_and_apply( [ '--backend', 'nudenet_v3' ] )
        self.assertEqual( betaconfig.detector_backend['selected'], 'nudenet_v3' )
        self.assertEqual( bu_detector.selected_backend_name(), 'nudenet_v3' )

    def test_no_backend_flag_leaves_config_file_value( self ):
        self._parse_and_apply( [] )
        self.assertEqual( betaconfig.detector_backend['selected'], 'retinanet_v2' )

    def test_unknown_backend_choice_is_rejected_by_argparse( self ):
        parser = bu_cli.build_arg_parser( 'test', include_preview=True, include_logging=True )
        with self.assertRaises( SystemExit ):
            parser.parse_args( [ '--backend', 'not_a_real_backend' ] )

    @mock.patch.dict( os.environ, { 'BETASUITE_DETECTOR_BACKEND_OVERRIDE': 'retinanet_v2' } )
    def test_env_var_still_wins_over_backend_flag( self ):
        self._parse_and_apply( [ '--backend', 'nudenet_v3' ] )
        # betaconfig itself reflects the CLI override...
        self.assertEqual( betaconfig.detector_backend['selected'], 'nudenet_v3' )
        # ...but the env var still wins for what actually gets used
        self.assertEqual( bu_detector.selected_backend_name(), 'retinanet_v2' )

    def test_backend_flag_and_nn_batch_size_flag_together_land_in_the_new_backend( self ):
        # ordering regression: --backend must be applied BEFORE
        # --nn-batch-size's own special case resolves "the currently
        # selected backend" (see apply_cli_overrides' own comment) - if
        # applied in the wrong order, --nn-batch-size would land in the
        # OLD backend's block instead of the one this run actually uses.
        self._parse_and_apply( [ '--backend', 'nudenet_v3', '--nn-batch-size', '9' ] )
        self.assertEqual( betaconfig.detector_backend['nudenet_v3']['nn_batch_size'], 9 )
        self.assertNotIn( 'nn_batch_size', betaconfig.detector_backend['retinanet_v2'] )

    def test_backend_flag_applied_returns_in_applied_overrides_list( self ):
        applied = self._parse_and_apply( [ '--backend', 'nudenet_v3' ] )
        self.assertTrue( any( "detector_backend['selected']='nudenet_v3'" in a for a in applied ), applied )

    def test_detector_backend_entirely_unset_still_works( self ):
        del betaconfig.detector_backend
        self._parse_and_apply( [ '--backend', 'nudenet_v3' ] )
        self.assertEqual( betaconfig.detector_backend['selected'], 'nudenet_v3' )


if __name__ == '__main__':
    unittest.main()
