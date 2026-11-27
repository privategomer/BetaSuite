"""
test_third_party_notices.py - the committed third-party notices match the
lockfiles, pass the license policy, and render to the committed markdown.

tools/release/third_party_notices.sh writes assets/notices/third_party.json
and THIRD_PARTY_NOTICES.md (the About screen shows it). It needs cargo, the
UI's node_modules and the engine's packages, so these tests check the
committed files instead, with the standard library only:

- every crate in Cargo.lock and every runtime package in the UI's
  package-lock.json is listed, at its locked version, and nothing else
- every package the requirement files name is listed
- every hosted model is listed with its catalog license
- no license is outside the policy or the reviewed exceptions
- THIRD_PARTY_NOTICES.md is exactly what the data renders to

After changing a dependency, run tools/release/third_party_notices.sh and
commit the result.
"""

import json
import os
import sys
import tomllib
import unittest

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, os.path.join( REPO, 'tools', 'release' ) )

import third_party_notices as notices  # noqa: E402


def _data():
    with open( notices.DATA, encoding='utf-8' ) as handle:
        return json.load( handle )


def _pairs( rows ):
    return sorted( ( row[ 'name' ], row[ 'version' ] ) for row in rows )


class CurrentTest( unittest.TestCase ):

    def test_rust_crates_are_the_lockfile( self ):
        with open( os.path.join( REPO, 'ui', 'src-tauri', 'Cargo.lock' ), 'rb' ) as handle:
            lock = tomllib.load( handle )
        locked = sorted( ( pkg[ 'name' ], pkg[ 'version' ] ) for pkg in lock[ 'package' ] if 'source' in pkg )
        self.assertEqual( _pairs( _data()[ 'rust' ] ), locked )

    def test_npm_packages_are_the_runtime_lockfile( self ):
        with open( os.path.join( REPO, 'ui', 'package-lock.json' ), encoding='utf-8' ) as handle:
            lock = json.load( handle )
        locked = sorted( ( path.split( 'node_modules/' )[-1], version )
                         for path, version in notices.npm_runtime( lock ).items() )
        self.assertEqual( _pairs( _data()[ 'npm' ] ), locked )
        self.assertNotIn( 'eslint', [ name for name, _ in locked ] )   # build tools stay out

    def test_every_required_python_package_is_listed( self ):
        listed = { notices.canonical( row[ 'name' ] ) for row in _data()[ 'python' ] }
        self.assertEqual( sorted( set( notices.requirement_names() ) - listed ), [] )

    def test_every_model_is_listed_with_its_license( self ):
        with open( os.path.join( REPO, 'assets', 'models', 'catalog.json' ), encoding='utf-8' ) as handle:
            cat = json.load( handle )
        want = sorted( ( m[ 'id' ], cat[ 'licenses' ][ m[ 'license' ] ][ 'spdx' ] ) for m in cat[ 'models' ] )
        self.assertEqual( sorted( ( row[ 'name' ], row[ 'license' ] ) for row in _data()[ 'models' ] ), want )

    def test_every_license_passes_the_policy( self ):
        self.assertEqual( notices.audit( _data() ), [] )

    def test_markdown_is_rendered_from_the_data( self ):
        with open( notices.MARKDOWN, encoding='utf-8' ) as handle:
            self.assertEqual( handle.read(), notices.render( _data() ) )


class PolicyTest( unittest.TestCase ):

    def test_expressions( self ):
        cases = {
            'MIT': True, 'MIT OR Apache-2.0': True, '(MIT OR Apache-2.0) AND Unicode-3.0': True,
            'Apache-2.0 WITH LLVM-exception OR MIT': True, 'GPL-2.0-only': False,
            'MIT OR GPL-2.0-only': True, 'MIT AND GPL-2.0-only': False, 'SSPL-1.0': False, '': False,
            'LicenseRef-NVIDIA-Proprietary': False, 'Proprietary': False,
        }
        for expr, want in cases.items():
            with self.subTest( expr=expr ):
                self.assertIs( notices.allowed( expr ), want )

    def test_normalize( self ):
        self.assertEqual( notices.normalize( 'MIT/Apache-2.0' ), 'MIT OR Apache-2.0' )
        self.assertEqual( notices.normalize( 'Apache-2.0 / MIT' ), 'Apache-2.0 OR MIT' )
        self.assertEqual( notices.normalize( 'MIT License' ), 'MIT' )
        self.assertEqual( notices.normalize( 'Permission is hereby granted\n' * 3 ), '' )

    def test_review_never_waves_through_a_whole_section( self ):
        for rule in notices.reviewed()[ 'allow' ]:
            with self.subTest( rule=rule[ 'name' ] ):
                self.assertNotIn( rule[ 'name' ], ( '.*', '.+' ) )
                self.assertTrue( rule[ 'reason' ] )

    def test_an_unreviewed_license_is_reported( self ):
        data = { 'rust': [ { 'name': 'x', 'version': '1', 'license': 'SSPL-1.0' } ],
                 'npm': [], 'python': [], 'downloads': [], 'models': [] }
        self.assertEqual( notices.audit( data ), [ ( 'rust', 'x', 'SSPL-1.0' ) ] )


if __name__ == '__main__':
    unittest.main()
