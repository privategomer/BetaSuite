"""
test_validation_corpus.py - a bad value that validation caught once is
still caught.

For every leaf of the shipped config, a few bad values are tried: a
wrong type, a negative number, a huge number, a made-up choice. The
golden file records, from before the settings registry existed, which
of them validation rejected and whether the error named the setting.
Validation may learn to catch more (a case that passed may now fail),
never fewer: a case recorded as rejected must still be rejected, and
one recorded as named must still be named.

Cases that made validation itself crash are recorded too, so a change
that turns a crash into an error is visible.

    python -m tests.test_validation_corpus write

regenerates the file, which is only right when a check is removed on
purpose.
"""

import json
import os
import sys
import unittest

import betautils_config as bu_config

from tests import config_harness as harness

GOLDEN_PATH = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ),
                            'golden', 'validation_corpus.json' )

NOT_A_CHOICE = 'zz_not_a_choice'


def bad_values( value ):
    """Bad candidates for a leaf, chosen by the type of its shipped value."""
    if isinstance( value, bool ):
        return [ 'yes' ]
    if isinstance( value, ( int, float ) ):
        return [ 'x', -1, 10**9 ]
    if isinstance( value, str ):
        return [ 123, NOT_A_CHOICE ]
    if value is None:
        return [ 'x', -1 ]
    return [ 'x', 123 ]


def case_name( path, bad ):
    return '%s = %r'%( '.'.join( str( part ) for part in path ), bad )


def _named( path, errors ):
    words = [ part for part in path if isinstance( part, str ) ]
    return any( words[-1] in error for error in errors )


def corpus():
    """{ case: { 'fires': bool, 'named': bool, 'raises': exception type or None } }"""
    results = {}
    with harness.preserved():
        baseline = set( bu_config._collect_validation_errors() )
    for path in harness.shipped_leaf_paths():
        for bad in bad_values( harness.get_path( path ) ):
            with harness.preserved():
                harness.set_path( path, bad )
                raises = None
                try:
                    errors = [ error for error in bu_config._collect_validation_errors()
                               if error not in baseline ]
                except Exception as err:
                    raises = type( err ).__name__
                    errors = []
            results[case_name( path, bad )] = {
                'fires': bool( errors ), 'named': _named( path, errors ), 'raises': raises }
    return results


class TestValidationStillCatchesWhatItCaught( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        with open( GOLDEN_PATH, 'r', encoding='utf-8' ) as handle:
            cls.golden = json.load( handle )
        cls.current = corpus()

    def test_every_recorded_case_is_still_tried( self ):
        self.assertEqual( [], sorted( set( self.golden ) - set( self.current ) ) )

    def test_a_rejected_value_is_still_rejected( self ):
        lost = [ name for name, was in self.golden.items()
                 if was['fires'] and not self.current.get( name, {} ).get( 'fires' ) ]
        self.assertEqual( [], lost )

    def test_a_named_setting_is_still_named( self ):
        lost = [ name for name, was in self.golden.items()
                 if was['named'] and not self.current.get( name, {} ).get( 'named' ) ]
        self.assertEqual( [], lost )


def write_golden():
    os.makedirs( os.path.dirname( GOLDEN_PATH ), exist_ok=True )
    with open( GOLDEN_PATH, 'w', encoding='utf-8' ) as handle:
        json.dump( corpus(), handle, indent=1, sort_keys=True )
        handle.write( '\n' )
    print( "wrote %s"%( GOLDEN_PATH, ) )


if __name__ == '__main__':
    if sys.argv[1:] == [ 'write' ]:
        write_golden()
    else:
        unittest.main()
