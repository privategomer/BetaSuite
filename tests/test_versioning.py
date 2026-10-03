"""
test_versioning.py - VERSION, the runtime version string, and the pure
parts of tools/release/release.py.

The git-backed cases build a throwaway repository and skip themselves if
git is unavailable. The bump test skips if the semver package (a
developer-only dependency, requirements-dev.txt) isn't installed.
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest

import betautils_version as bu_version

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..' ) )


def _load_release():
    path = os.path.join( REPO_ROOT, 'tools', 'release', 'release.py' )
    spec = importlib.util.spec_from_file_location( 'release_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class VersionFileTest( unittest.TestCase ):

    def test_repo_version_file_is_semver( self ):
        with open( os.path.join( REPO_ROOT, 'VERSION' ) ) as f:
            self.assertRegex( f.read().strip(), bu_version.SEMVER_RE )

    def test_missing_or_malformed_file_reads_as_zero( self ):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join( tmp, 'VERSION' )
            self.assertEqual( bu_version.read_version_file( path ), '0.0.0' )
            with open( path, 'w' ) as f:
                f.write( 'v1.2\n' )
            self.assertEqual( bu_version.read_version_file( path ), '0.0.0' )
            with open( path, 'w' ) as f:
                f.write( '2.3.4-rc.1\n' )
            self.assertEqual( bu_version.read_version_file( path ), '2.3.4-rc.1' )


@unittest.skipUnless( shutil.which( 'git' ), 'git not installed' )
class DescribeTest( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup( shutil.rmtree, self.tmp, ignore_errors=True )
        self.git( 'init', '-q' )
        self.git( 'config', 'user.email', 'test@example.com' )
        self.git( 'config', 'user.name', 'test' )
        self.write( 'VERSION', '1.4.0\n' )
        self.git( 'add', 'VERSION' )
        self.git( 'commit', '-q', '-m', 'first' )

    def git( self, *args ):
        return subprocess.run( [ 'git', '-C', self.tmp ] + list( args ), check=True,
                               capture_output=True, text=True ).stdout.strip()

    def write( self, name, text ):
        with open( os.path.join( self.tmp, name ), 'w' ) as f:
            f.write( text )

    def test_untagged_checkout_gets_build_metadata( self ):
        sha = self.git( 'rev-parse', '--short', 'HEAD' )
        self.assertEqual( bu_version.describe( self.tmp ), '1.4.0+1.g%s'%(sha) )

    def test_exact_release_tag_is_plain( self ):
        self.git( 'tag', '-a', 'v1.4.0', '-m', 'v1.4.0' )
        self.assertEqual( bu_version.describe( self.tmp ), '1.4.0' )

    def test_commits_past_tag_and_dirty_tree( self ):
        self.git( 'tag', '-a', 'v1.4.0', '-m', 'v1.4.0' )
        self.write( 'VERSION', '1.4.0\n\n' )
        self.git( 'commit', '-q', '-am', 'second' )
        sha = self.git( 'rev-parse', '--short', 'HEAD' )
        self.assertEqual( bu_version.describe( self.tmp ), '1.4.0+1.g%s'%(sha) )
        self.write( 'VERSION', '1.4.0\n' )
        self.assertEqual( bu_version.describe( self.tmp ), '1.4.0+1.g%s.dirty'%(sha) )

    def test_not_a_checkout_is_plain( self ):
        shutil.rmtree( os.path.join( self.tmp, '.git' ) )
        self.assertEqual( bu_version.describe( self.tmp ), '1.4.0' )


class ReleaseHelpersTest( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.release = _load_release()

    def test_suggested_bump( self ):
        s = self.release.suggest_bump
        self.assertEqual( s( [ 'anything' ], version_is_tagged=False ), 'none' )
        self.assertEqual( s( [ 'fix crash', 'feat!: drop old config keys' ], True ), 'major' )
        self.assertEqual( s( [ 'tidy', 'BREAKING: cache format' ], True ), 'major' )
        self.assertEqual( s( [ 'fix crash', 'Add sticker shapes' ], True ), 'minor' )
        self.assertEqual( s( [ 'feat(render): mosaic' ], True ), 'minor' )
        self.assertEqual( s( [ 'fix crash', 'docs' ], True ), 'patch' )
        self.assertEqual( s( [ 'Address review comments' ], True ), 'patch' )

    def test_bump_uses_semver( self ):
        try:
            import semver
        except ImportError:
            self.skipTest( 'semver not installed (requirements-dev.txt)' )
        b = self.release.bump_version
        self.assertEqual( b( semver, '1.1.0', 'minor' ), '1.2.0' )
        self.assertEqual( b( semver, '1.1.3', 'major' ), '2.0.0' )
        self.assertEqual( b( semver, '1.1.3', 'patch' ), '1.1.4' )
        self.assertEqual( b( semver, '1.1.3', 'none' ), '1.1.3' )

    def test_changelog_unreleased_section_is_dated( self ):
        text = '# Changelog\n\n## Unreleased\n\n- thing\n\n## 1.0.0 (2026-09-30)\n\n- first\n'
        out, notes = self.release.date_changelog( text, '1.1.0', '2026-10-03', [ 'ignored' ] )
        self.assertIn( '## 1.1.0 (2026-10-03)\n\n- thing', out )
        self.assertNotIn( 'Unreleased', out )
        self.assertEqual( notes, '- thing' )

    def test_changelog_existing_version_is_redated( self ):
        text = '# Changelog\n\n## 1.1.0 (2026-10-01)\n\n- setup\n\n## 1.0.0 (2026-09-30)\n'
        out, notes = self.release.date_changelog( text, '1.1.0', '2026-10-03', [] )
        self.assertIn( '## 1.1.0 (2026-10-03)', out )
        self.assertEqual( out.count( '## 1.1.0' ), 1 )
        self.assertEqual( notes, '- setup' )

    def test_changelog_drafted_from_commits_above_latest( self ):
        text = '# Changelog\n\nIntro.\n\n## 1.1.0 (2026-10-01)\n\n- setup\n'
        out, notes = self.release.date_changelog( text, '1.1.1', '2026-10-03', [ 'fix a', 'fix b' ] )
        self.assertLess( out.index( '## 1.1.1' ), out.index( '## 1.1.0' ) )
        self.assertEqual( notes, '- fix a\n- fix b' )


if __name__ == '__main__':
    unittest.main()
