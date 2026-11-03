"""
Every file a shell script sources must exist and must not be git-ignored.

v1.2.0 shipped ten scripts that source tools/lib/python_env.sh while the
template .gitignore's unanchored 'lib/' rule kept that file out of git.
On the machine that wrote it everything worked; on any fresh clone every
script failed on its first line. Nothing local could notice, because the
file was right there on disk.
"""

import os
import re
import shutil
import subprocess
import unittest

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

# source "$(dirname "$0")/../lib/python_env.sh"  and the BASH_SOURCE form.
_SOURCE_LINE = re.compile(
    r'^\s*(?:source|\.)\s+"\$\(dirname\s+"\$(?:0|\{BASH_SOURCE\[0\]\})"\)/([^"]+)"', re.M )


def _shell_scripts():
    for root, dirs, names in os.walk( REPO_ROOT ):
        dirs[:] = [ d for d in dirs if d not in ( '.git', '.venv', '__pycache__' ) ]
        for name in names:
            if name.endswith( '.sh' ):
                yield os.path.join( root, name )


# source "$REPO_ROOT/tools/tuning/harness_lib.sh", the form the bench
# harnesses use. Missed by the pattern above, so a harness sourcing a
# file that was never shipped passed this test.
_REPO_ROOT_SOURCE_LINE = re.compile(
    r'^\s*(?:source|\.)\s+"\$(?:REPO_ROOT|\{REPO_ROOT\})/([^"]+)"', re.M )


def _sourced_files():
    for script in _shell_scripts():
        with open( script, 'r', encoding='UTF-8' ) as handle:
            text = handle.read()
        for relative in _SOURCE_LINE.findall( text ):
            yield script, os.path.normpath( os.path.join( os.path.dirname( script ), relative ) )
        for relative in _REPO_ROOT_SOURCE_LINE.findall( text ):
            yield script, os.path.normpath( os.path.join( REPO_ROOT, relative ) )


class TestShellSourcesAreShipped( unittest.TestCase ):

    def test_the_scan_finds_the_known_sources( self ):
        # Precondition: a regex that matched nothing would pass vacuously.
        targets = { os.path.relpath( target, REPO_ROOT ) for _s, target in _sourced_files() }
        self.assertIn( os.path.join( 'tools', 'lib', 'python_env.sh' ), targets )
        self.assertIn( os.path.join( 'tools', 'tuning', 'harness_lib.sh' ), targets,
                       "the $REPO_ROOT form the bench harnesses use is scanned too" )

    def test_every_sourced_file_exists( self ):
        for script, target in _sourced_files():
            self.assertTrue( os.path.isfile( target ),
                             "%s sources %s, which does not exist"%( script, target ) )

    @unittest.skipUnless( shutil.which( 'git' ) and os.path.isdir( os.path.join( REPO_ROOT, '.git' ) ),
                          "needs a git checkout" )
    def test_no_sourced_file_is_git_ignored( self ):
        for script, target in _sourced_files():
            result = subprocess.run( [ 'git', 'check-ignore', '-q', target ],
                                     cwd=REPO_ROOT, capture_output=True )
            self.assertNotEqual( result.returncode, 0,
                "%s sources %s, but .gitignore excludes it, so a fresh clone "
                "will not have it"%( os.path.relpath( script, REPO_ROOT ),
                                     os.path.relpath( target, REPO_ROOT ) ) )


if __name__ == '__main__':
    unittest.main()
