"""
test_harness_prompts.py - the menu and the wait-for-a-person prompts.

ab_compare.sh asks which experiment to run and waits while a person
fills in verdicts. Unattended (no terminal), neither may hang: the menu
takes its default and the wait gives up, both logged. These pin that,
since a harness that hangs at an invisible prompt looks like a crash.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


class TestPromptsWithoutATerminal( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-prompts-' )
        self.log = os.path.join( self.root, 'harness.log' )

    def tearDown( self ):
        shutil.rmtree( self.root, ignore_errors=True )

    def bash( self, body ):
        script = 'source "%s/tools/tuning/harness_lib.sh"\n%s\n'%( REPO_ROOT, body )
        environment = dict( os.environ, HARNESS_LOG_FILE=self.log, PYTHON_BIN='python3',
                            PYTHON_BIN_SOURCE='test' )
        return subprocess.run( [ 'bash', '-c', script ], env=environment, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=30 )

    def test_the_menu_takes_its_default_and_says_so( self ):
        result = self.bash( 'choose_option "Pick" 2 "a - first" "b - second" "c - third"\n'
                            'echo "CHOSE:$CHOSEN_OPTION"' )
        self.assertIn( 'CHOSE:b - second', result.stdout )
        with open( self.log ) as handle:
            self.assertIn( 'using the default: b - second', handle.read() )

    def test_the_wait_gives_up_at_once( self ):
        result = self.bash( 'if wait_for_enter "Fill it in"; then echo WAITED; else echo GAVE_UP; fi' )
        self.assertIn( 'GAVE_UP', result.stdout )
        with open( self.log ) as handle:
            self.assertIn( "AWAITING INPUT skipped: 'Fill it in'", handle.read() )


if __name__ == '__main__':
    unittest.main()
