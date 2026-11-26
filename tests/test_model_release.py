"""
test_model_release.py - building and publishing the hosted model files
(tools/release/prepare_model_assets.py and upload_model_assets.py).

Synthetic catalog and files; gh and the network are stubbed.

What each test prevents:
  - a release folder with a model, source or license file that is not
    the pinned one, or SHA256SUMS written over a folder with a problem
  - a model hosted without its upstream source beside it, or without its
    license and NOTICE
  - a GitHub sign-in page mirrored as a source file
  - publishing a folder that does not match its sums or the catalog
  - publishing without the maintainer typing the tag, a re-run uploading
    again what is already there, or replacing an asset already there
  - the release becoming the repository's latest release
  - calling the release done when a file cannot be downloaded signed out
"""

import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
RELEASE = os.path.join( REPO_ROOT, 'tools', 'release' )


def _load( name ):
    if RELEASE not in sys.path:
        sys.path.insert( 0, RELEASE )
    spec = importlib.util.spec_from_file_location( name + '_under_test', os.path.join( RELEASE, name + '.py' ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


def _sha( data ):
    return hashlib.sha256( data ).hexdigest()


MODEL = b'\x08model-bytes' * 100
SOURCE = b'PKsource-bytes' * 100
LICENSE_TEXT = 'GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3\n'


def _catalog():
    return {
        'format': 1, 'release': 'models-v9',
        'mirrors': [ 'https://github.com/owner/repo/releases/download/models-v9/' ],
        'licenses': { 'up': { 'spdx': 'AGPL-3.0', 'file': 'LICENSE-Up', 'url': 'https://licenses.invalid/up',
                              'project': 'Upstream', 'home': 'https://upstream.invalid' } },
        'models': [ { 'id': 'm1', 'kind': 'detector', 'backend': 'b', 'variant': 'v', 'file': 'm1.onnx',
                      'size': len( MODEL ), 'sha256': _sha( MODEL ), 'license': 'up',
                      'upstream': 'https://upstream.invalid/m1.onnx',
                      'sources': [ { 'file': 'm1.pt', 'size': len( SOURCE ), 'sha256': _sha( SOURCE ),
                                     'upstream': 'https://github.com/up/stream/releases/download/w/m1.pt' } ] } ],
    }


class _Case( unittest.TestCase ):

    def setUp( self ):
        self.prep = _load( 'prepare_model_assets' )
        self.tmp = tempfile.mkdtemp( prefix='betasuite-release-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.model_dir = os.path.join( self.tmp, 'model' )
        self.out = os.path.join( self.tmp, 'out' )
        os.makedirs( self.model_dir )
        with open( os.path.join( self.model_dir, 'm1.onnx' ), 'wb' ) as handle:
            handle.write( MODEL )
        self.log = mock.Mock()
        self.served = { 'https://licenses.invalid/up': LICENSE_TEXT.encode(),
                        'https://github.com/up/stream/releases/download/w/m1.pt': SOURCE }

    def fake_download( self, url, dest, log ):
        if url not in self.served:
            raise OSError( 'unreachable' )
        with open( dest, 'wb' ) as handle:
            handle.write( self.served[url] )
        if self.served[url].startswith( b'<!DOCTYPE html' ):
            raise self.prep.Mismatch( 'a web page came back instead of the file' )

    def build( self, catalog=None ):
        with mock.patch.object( self.prep, 'download', side_effect=self.fake_download ), \
             mock.patch.object( self.prep, 'gh_download', side_effect=self.prep.Mismatch( 'no gh' ) ):
            return self.prep.prepare( self.out, self.model_dir, None, self.log, catalog or _catalog() )


class TestPrepare( _Case ):

    def test_builds_every_file_with_sums( self ):
        self.assertEqual( self.build(), 0 )
        names = sorted( os.listdir( self.out ) )
        self.assertEqual( names, sorted( [ 'm1.onnx', 'm1.pt', 'LICENSE-Up', 'NOTICE-models.txt',
                                           'RELEASE_NOTES.md', 'SHA256SUMS' ] ) )
        sums = _load( 'upload_model_assets' ).read_sums( self.out )
        for name, digest in sums.items():
            with open( os.path.join( self.out, name ), 'rb' ) as handle:
                self.assertEqual( _sha( handle.read() ), digest, name )
        with open( os.path.join( self.out, 'NOTICE-models.txt' ) ) as handle:
            notice = handle.read()
        self.assertIn( 'm1.onnx: converted from m1.pt', notice )
        self.assertIn( 'AGPL-3.0', notice )

    def test_rerun_keeps_what_is_there( self ):
        self.build()
        self.served.clear()
        self.served['https://licenses.invalid/up'] = LICENSE_TEXT.encode()
        self.assertEqual( self.build(), 0, "the source is already there with the right hash" )

    def test_a_missing_model_writes_no_sums( self ):
        os.remove( os.path.join( self.model_dir, 'm1.onnx' ) )
        self.assertEqual( self.build(), 1 )
        self.assertFalse( os.path.exists( os.path.join( self.out, 'SHA256SUMS' ) ) )

    def test_a_tampered_source_is_refused( self ):
        self.served['https://github.com/up/stream/releases/download/w/m1.pt'] = SOURCE + b'x'
        self.assertEqual( self.build(), 1 )
        self.assertFalse( os.path.exists( os.path.join( self.out, 'm1.pt' ) ) )

    def test_a_sign_in_page_is_not_mirrored_and_gh_is_tried( self ):
        self.served['https://github.com/up/stream/releases/download/w/m1.pt'] = b'<!DOCTYPE html><html>sign in'
        tried = []

        def gh_fallback( url, dest, log ):
            tried.append( url )
            with open( dest, 'wb' ) as handle:
                handle.write( SOURCE )
        with mock.patch.object( self.prep, 'download', side_effect=self.fake_download ), \
             mock.patch.object( self.prep, 'gh_download', side_effect=gh_fallback ):
            self.assertEqual( self.prep.prepare( self.out, self.model_dir, None, self.log, _catalog() ), 0 )
        self.assertEqual( tried, [ 'https://github.com/up/stream/releases/download/w/m1.pt' ] )

    def test_a_wrong_license_text_is_a_problem( self ):
        self.served['https://licenses.invalid/up'] = b'MIT License\n'
        self.assertEqual( self.build(), 1 )
        self.assertFalse( os.path.exists( os.path.join( self.out, 'LICENSE-Up' ) ) )

    def test_sources_folder_is_used_when_given( self ):
        sources = os.path.join( self.tmp, 'sources' )
        os.makedirs( sources )
        with open( os.path.join( sources, 'm1.pt' ), 'wb' ) as handle:
            handle.write( SOURCE )
        self.served.pop( 'https://github.com/up/stream/releases/download/w/m1.pt' )
        with mock.patch.object( self.prep, 'download', side_effect=self.fake_download ):
            self.assertEqual( self.prep.prepare( self.out, self.model_dir, sources, self.log, _catalog() ), 0 )

    def test_the_shipped_catalog_names_every_file_once( self ):
        import betautils_models as bu_models
        names = self.prep.expected_files( bu_models.load() )
        self.assertEqual( len( names ), len( set( names ) ) )


class _Run:
    """A fake gh: answers by subcommand, records every call."""

    def __init__( self, assets=None ):
        self.calls = []
        self.assets = assets

    def __call__( self, command, **_kw ):
        self.calls.append( command[1:] )
        if command[1:3] == [ 'api', 'user' ]:
            return mock.Mock( returncode=0, stdout='maintainer\n' )
        if command[1:3] == [ 'release', 'view' ]:
            if self.assets is None:
                return mock.Mock( returncode=1, stdout='release not found\n' )
            return mock.Mock( returncode=0, stdout=json.dumps(
                { 'assets': [ { 'name': n, 'size': s } for n, s in self.assets.items() ] } ) )
        return mock.Mock( returncode=0, stdout='' )

    def uploads( self ):
        return [ os.path.basename( c[-1] ) for c in self.calls if c[:2] == [ 'release', 'upload' ] ]


class TestUpload( _Case ):

    def setUp( self ):
        super().setUp()
        self.up = _load( 'upload_model_assets' )
        self.assertEqual( self.build(), 0 )
        self.files = {}
        for name in os.listdir( self.out ):
            with open( os.path.join( self.out, name ), 'rb' ) as handle:
                self.files[name] = handle.read()
        patch = mock.patch( 'betautils_models.load', return_value=_catalog() )
        patch.start()
        self.addCleanup( patch.stop )

    def opener( self, request, timeout=None ):
        name = request.full_url.rsplit( '/', 1 )[1]
        self.assertNotIn( 'Authorization', request.headers, "the check downloads signed out" )
        if name not in self.files:
            raise OSError( '404' )
        return io.BytesIO( self.files[name] )

    def main( self, *argv, run, answers=() ):
        replies = iter( answers )
        with mock.patch( 'builtins.input', side_effect=lambda prompt: next( replies ) ):
            return self.up.main( [ '--folder', self.out, '--log-file', os.path.join( self.tmp, 'u.log' ), '--log-level', 'error' ] + list( argv ),
                                 run=run, opener=self.opener )

    def test_new_release_after_typing_the_tag( self ):
        run = _Run()
        self.assertEqual( self.main( run=run, answers=[ 'models-v9' ] ), 0 )
        create = [ c for c in run.calls if c[:2] == [ 'release', 'create' ] ]
        self.assertEqual( len( create ), 1 )
        self.assertIn( '--latest=false', create[0] )
        self.assertEqual( create[0][ create[0].index( '-R' ) + 1 ], 'owner/repo' )
        self.assertEqual( sorted( run.uploads() ), sorted( self.files ), "every file, SHA256SUMS included" )

    def test_a_wrong_answer_changes_nothing( self ):
        run = _Run()
        self.assertEqual( self.main( run=run, answers=[ 'yes' ] ), 1 )
        self.assertFalse( any( c[:2] in ( [ 'release', 'create' ], [ 'release', 'upload' ] ) for c in run.calls ) )

    def test_a_rerun_uploads_only_what_is_missing( self ):
        sizes = { n: len( d ) for n, d in self.files.items() if n != 'm1.pt' }
        run = _Run( assets=sizes )
        self.assertEqual( self.main( '--yes', run=run ), 0 )
        self.assertEqual( run.uploads(), [ 'm1.pt' ] )
        self.assertFalse( any( c[:2] == [ 'release', 'create' ] for c in run.calls ) )

    def test_a_wrong_size_already_there_stops( self ):
        sizes = { n: len( d ) for n, d in self.files.items() }
        sizes['m1.onnx'] += 1
        run = _Run( assets=sizes )
        self.assertEqual( self.main( '--yes', run=run ), 1 )
        self.assertEqual( run.uploads(), [] )

    def test_a_tampered_folder_is_not_published( self ):
        with open( os.path.join( self.out, 'm1.pt' ), 'ab' ) as handle:
            handle.write( b'x' )
        run = _Run()
        self.assertEqual( self.main( '--yes', run=run ), 1 )
        self.assertEqual( run.calls, [] )

    def test_a_file_not_public_fails_the_run( self ):
        del self.files['m1.pt']
        self.assertEqual( self.main( '--verify-only', run=_Run() ), 1 )

    def test_verify_only_never_calls_gh( self ):
        run = _Run()
        self.assertEqual( self.main( '--verify-only', run=run ), 0 )
        self.assertEqual( run.calls, [] )


if __name__ == '__main__':
    unittest.main()
