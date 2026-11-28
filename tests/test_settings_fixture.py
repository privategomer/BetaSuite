"""
test_settings_fixture.py - the registry fixture the app's checks read is
what the engine sends today.

ui/scripts/check-settings-reach.mjs proves every setting is reachable in
All Settings without starting the engine, from
ui/scripts/fixtures/settings.json. A stale fixture would prove it for a
registry that no longer exists, so this fails until
tools/maintenance/dump_settings_fixture.py is run again.
"""

import importlib.util
import io
import os
import tempfile
import unittest

import betaconfig
import betautils_local as bu_local

from tests import config_harness as harness

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
TOOL = os.path.join( REPO, 'tools', 'maintenance', 'dump_settings_fixture.py' )


def _tool():
    spec = importlib.util.spec_from_file_location( 'dump_settings_fixture', TOOL )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestFixtureIsCurrent( unittest.TestCase ):

    def test_the_fixture_equals_what_the_engine_sends( self ):
        tool = _tool()
        with harness.shipped():
            expected = tool.render( tool.fixture() )
        with open( tool.DEFAULT_OUT, encoding='utf-8' ) as handle:
            actual = handle.read()
        self.assertTrue( actual == expected,
                         'ui/scripts/fixtures/settings.json is stale: run '
                         'python tools/maintenance/dump_settings_fixture.py' )

    def test_the_fixture_is_the_same_whatever_the_live_config_holds( self ):
        # The live config carries the local layer; a choice list that
        # follows another setting (encode_preset follows the codec) must
        # still come out as betaconfig.py's.
        tool = _tool()
        saved_loaded = bu_local.LOADED
        self.addCleanup( setattr, bu_local, 'LOADED', saved_loaded )
        with harness.shipped(), tempfile.TemporaryDirectory() as folder:
            bu_local.LOADED = None
            plain = tool.fixture()
            shipped_codec = harness.get_path( ( 'encode_video_codec', ) )
            other = 'h264_nvenc' if shipped_codec != 'h264_nvenc' else 'libx264'
            self.assertNotEqual( tool.bu_settings.encode_presets( shipped_codec ),
                                 tool.bu_settings.encode_presets( other ) )
            local = os.path.join( folder, 'betaconfig_local.py' )
            with open( local, 'w', encoding='utf-8' ) as handle:
                handle.write( "encode_video_codec = %r\nencode_crf = 23\n"%( other, ) )
            # The way a run applies a local file, so the layer is recorded.
            bu_local.LOADED = bu_local.apply( vars( betaconfig ), path=local, stderr=io.StringIO() )
            self.assertEqual( other, betaconfig.encode_video_codec )
            changed = tool.fixture()
        self.assertEqual( plain, changed )


if __name__ == '__main__':
    unittest.main()
