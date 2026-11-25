"""
test_ffmpeg_pin.py - tools/lib/ffmpeg_pin: the pin file's shape, and the
installer against a fake archive served from a file:// URL (nothing is
downloaded or run).
"""

import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, os.path.join( REPO, 'tools', 'lib' ) )

import ffmpeg_pin  # noqa: E402


def _tar( path, files ):
    with tarfile.open( path, 'w:xz' ) as bundle:
        for name, data in files.items():
            info = tarfile.TarInfo( name )
            info.size = len( data )
            bundle.addfile( info, io.BytesIO( data ) )


def _zip( path, files ):
    with zipfile.ZipFile( path, 'w' ) as bundle:
        for name, data in files.items():
            bundle.writestr( name, data )


class TestThePinFile( unittest.TestCase ):

    def test_each_build_has_an_https_url_and_a_sha256( self ):
        pin = ffmpeg_pin.load_pin()
        self.assertIn( 'linux64', pin['builds'] )
        self.assertIn( 'win64', pin['builds'] )
        for build, entry in pin['builds'].items():
            with self.subTest( build=build ):
                self.assertTrue( entry['url'].startswith( 'https://' ) )
                self.assertIn( pin['version'], entry['url'], "every build is the one pinned version" )
                self.assertRegex( entry['sha256'], r'^[0-9a-f]{64}$' )


class TestInstall( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-ffmpeg-pin-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.logger = mock.Mock()
        self.target = os.path.join( self.tmp, 'target' )

    def pin_for( self, archive, sha=None ):
        with open( archive, 'rb' ) as handle:
            digest = hashlib.sha256( handle.read() ).hexdigest()
        return { 'version': 'n9.9', 'builds': { 'test': {
            'url': 'file://' + archive, 'sha256': sha or digest } } }

    def test_a_tar_build_is_checked_and_its_two_programs_unpacked( self ):
        archive = os.path.join( self.tmp, 'ffmpeg-n9.9-linux64.tar.xz' )
        _tar( archive, { 'ffmpeg-n9.9/bin/ffmpeg': b'one', 'ffmpeg-n9.9/bin/ffprobe': b'two',
                         'ffmpeg-n9.9/bin/ffplay': b'three', 'ffmpeg-n9.9/LICENSE.txt': b'gpl' } )
        bin_dir = ffmpeg_pin.install( self.target, 'test', self.logger, self.pin_for( archive ) )
        self.assertEqual( sorted( os.listdir( bin_dir ) ), [ 'ffmpeg', 'ffprobe' ] )
        self.assertEqual( open( os.path.join( bin_dir, 'ffmpeg' ), 'rb' ).read(), b'one' )
        self.assertTrue( os.access( os.path.join( bin_dir, 'ffprobe' ), os.X_OK ) )
        self.assertEqual( open( os.path.join( self.target, 'version.txt' ) ).read(), 'n9.9 test\n' )

    def test_a_zip_build_too( self ):
        archive = os.path.join( self.tmp, 'ffmpeg-n9.9-win64.zip' )
        _zip( archive, { 'ffmpeg-n9.9/bin/ffmpeg.exe': b'one', 'ffmpeg-n9.9/bin/ffprobe.exe': b'two' } )
        bin_dir = ffmpeg_pin.install( self.target, 'test', self.logger, self.pin_for( archive ) )
        self.assertEqual( sorted( os.listdir( bin_dir ) ), [ 'ffmpeg.exe', 'ffprobe.exe' ] )

    def test_a_wrong_checksum_installs_nothing( self ):
        archive = os.path.join( self.tmp, 'ffmpeg-n9.9-linux64.tar.xz' )
        _tar( archive, { 'x/bin/ffmpeg': b'one' } )
        with self.assertRaises( ValueError ):
            ffmpeg_pin.install( self.target, 'test', self.logger, self.pin_for( archive, sha='0' * 64 ) )
        self.assertFalse( os.path.exists( os.path.join( self.target, 'bin' ) ) )
        self.assertFalse( os.path.exists( os.path.join( self.target, 'version.txt' ) ) )
        self.assertEqual( [ name for name in os.listdir( self.target ) if name != 'ffmpeg_pin.log' ], [],
                          "the download is not left behind" )

    def test_an_installed_version_is_not_fetched_again( self ):
        archive = os.path.join( self.tmp, 'ffmpeg-n9.9-linux64.tar.xz' )
        _tar( archive, { 'x/bin/ffmpeg': b'one', 'x/bin/ffprobe': b'two' } )
        pin = self.pin_for( archive )
        ffmpeg_pin.install( self.target, 'test', self.logger, pin )
        os.remove( archive )
        ffmpeg_pin.install( self.target, 'test', self.logger, pin )   # would fail if it downloaded

    def test_members_outside_bin_or_with_other_names_are_never_written( self ):
        self.assertEqual( ffmpeg_pin._members( [ 'a/bin/ffmpeg', '../bin/ffprobe', 'a/ffmpeg', 'a/bin/sh',
                                                 'a/bin/../../ffmpeg' ] ),
                          { 'a/bin/ffmpeg': 'ffmpeg', '../bin/ffprobe': 'ffprobe' } )
        archive = os.path.join( self.tmp, 'evil.tar.xz' )
        _tar( archive, { '../bin/ffmpeg': b'one', 'x/bin/ffprobe': b'two' } )
        bin_dir = ffmpeg_pin.install( self.target, 'test', self.logger, self.pin_for( archive ) )
        self.assertEqual( sorted( os.listdir( bin_dir ) ), [ 'ffmpeg', 'ffprobe' ] )
        self.assertFalse( os.path.exists( os.path.join( self.tmp, 'bin' ) ) )


class TestOpenCvIsPinnedExactly( unittest.TestCase ):

    def test_requirements_pin_one_opencv_version( self ):
        with open( os.path.join( REPO, 'requirements.txt' ), encoding='utf-8' ) as handle:
            lines = [ line.strip() for line in handle if line.strip().startswith( 'opencv' ) ]
        self.assertEqual( len( lines ), 1 )
        self.assertRegex( lines[0], r'^opencv-python==\d+(\.\d+)+$' )


if __name__ == '__main__':
    unittest.main()
