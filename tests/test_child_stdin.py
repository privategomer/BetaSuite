"""
test_child_stdin.py - every child process the engine starts says what
its stdin is.

In serve mode the engine's stdin is the command stream. isolate_stdio
points fd 0 at the null device, so a child that inherits stdin reads end
of input rather than commands, but that is one safety net. Each call
also passes stdin explicitly (DEVNULL, or PIPE where the child is fed
frames), so no child depends on what the parent's fd 0 is. This test
reads the engine modules and fails on a subprocess call without it.
"""

import ast
import glob
import os
import unittest

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

CALLS = ( 'run', 'Popen', 'call', 'check_call', 'check_output' )


def engine_sources():
    names = sorted( glob.glob( os.path.join( REPO, 'betautils_*.py' ) ) )
    names += [ os.path.join( REPO, name ) for name in ( 'betatv.py', 'betastare.py', 'betaengine.py' ) ]
    names += sorted( glob.glob( os.path.join( REPO, 'detectors', '*.py' ) ) )
    names += sorted( glob.glob( os.path.join( REPO, 'segmenters', '*.py' ) ) )
    return [ name for name in names if os.path.exists( name ) ]


def calls_without_stdin( path ):
    with open( path, 'r', encoding='utf-8' ) as handle:
        tree = ast.parse( handle.read(), filename=path )
    found = []
    for node in ast.walk( tree ):
        if not isinstance( node, ast.Call ) or not isinstance( node.func, ast.Attribute ):
            continue
        target = node.func.value
        if not ( isinstance( target, ast.Name ) and target.id == 'subprocess'
                 and node.func.attr in CALLS ):
            continue
        if any( keyword.arg is None for keyword in node.keywords ):
            continue   # **kwargs: cannot tell, and none exist today
        if not any( keyword.arg in ( 'stdin', 'input' ) for keyword in node.keywords ):
            found.append( '%s:%d'%( os.path.relpath( path, REPO ), node.lineno ) )
    return found


class TestChildStdin( unittest.TestCase ):

    def test_every_engine_child_process_has_an_explicit_stdin( self ):
        missing = []
        for path in engine_sources():
            missing.extend( calls_without_stdin( path ) )
        self.assertEqual( missing, [], "pass stdin=subprocess.DEVNULL (or PIPE) to these" )

    def test_the_check_sees_a_call_without_it( self ):
        import tempfile
        with tempfile.NamedTemporaryFile( 'w', suffix='.py', delete=False ) as handle:
            handle.write( "import subprocess\nsubprocess.run( [ 'x' ], capture_output=True )\n"
                          "subprocess.Popen( [ 'y' ], stdin=subprocess.DEVNULL )\n" )
        self.addCleanup( os.remove, handle.name )
        self.assertEqual( [ line.rsplit( ':', 1 )[1] for line in calls_without_stdin( handle.name ) ],
                          [ '2' ] )


if __name__ == '__main__':
    unittest.main()
