"""
test_shipped_cache_keys.py - the shipped config's cache keys are pinned.

A finished output is found again by the detection, censor and encode
keys in its filename. If a change to how settings are read, checked or
stored moves any of them for the shipped config, every finished output
is re-rendered on upgrade with nothing about it changed. The keys here
were recorded before the settings registry existed, for both backends,
every model variant, preview on and off, reverse mode (with and without
person windows) on and off, and both the suite's censored labels and
the shipped empty list.

A change that is MEANT to re-render (a new setting that changes output,
a renderer fix) updates the golden file on purpose:

    python -m tests.test_shipped_cache_keys write

and says so in CHANGELOG.
"""

import json
import os
import sys
import unittest

import betaconfig
import betautils_detector as bu_detector

from tests import config_harness as harness

GOLDEN_PATH = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ),
                            'golden', 'shipped_cache_keys.json' )

LABEL_SETS = {
    'suite': [ 'exposed_vulva', 'exposed_breast' ],
    'none': [],
}

REVERSE_MODES = {
    'off': {},
    'on': { 'enabled': True },
    'on_person_windows': { 'enabled': True, 'person_windows': { 'enabled': True } },
}


def _variants( backend_name ):
    return bu_detector.variant_names( backend_name ) or [ None ]


def matrix():
    """{ case name: { 'detection', 'censor', 'encode' } } for the shipped config."""
    results = {}
    with harness.shipped():
        _fill( results )
    return results


def _fill( results ):
    for backend_name in bu_detector.registered_backend_names():
        for variant in _variants( backend_name ):
            for labels_name, labels in LABEL_SETS.items():
                for reverse_name, reverse in REVERSE_MODES.items():
                    for preview in ( False, True ):
                        name = '%s/%s/%s/reverse_%s/preview_%s'%(
                            backend_name, variant, labels_name, reverse_name,
                            'on' if preview else 'off' )
                        with harness.preserved():
                            betaconfig.detector_backend['selected'] = backend_name
                            if variant is not None:
                                betaconfig.detector_backend[backend_name]['model_variant'] = variant
                            betaconfig.items_to_censor = list( labels )
                            for key, value in reverse.items():
                                if isinstance( value, dict ):
                                    betaconfig.reverse[key].update( value )
                                else:
                                    betaconfig.reverse[key] = value
                            betaconfig.preview_mode_enabled = preview
                            results[name] = harness.keys( backend_name )


class TestShippedCacheKeysAreUnchanged( unittest.TestCase ):

    def test_every_case_matches_the_recorded_keys( self ):
        with open( GOLDEN_PATH, 'r', encoding='utf-8' ) as handle:
            golden = json.load( handle )
        current = matrix()
        self.assertEqual( sorted( golden ), sorted( current ),
                          "the case list changed; regenerate only on purpose" )
        moved = { name: { kind: ( golden[name][kind], current[name][kind] )
                          for kind in golden[name] if golden[name][kind] != current[name][kind] }
                  for name in golden }
        moved = { name: kinds for name, kinds in moved.items() if kinds }
        self.assertEqual( {}, moved, "the shipped config's keys moved, so upgrading re-renders" )

    def test_the_matrix_covers_both_backends_and_every_variant( self ):
        names = matrix().keys()
        for backend_name in bu_detector.registered_backend_names():
            for variant in _variants( backend_name ):
                self.assertTrue( any( name.startswith( '%s/%s/'%( backend_name, variant ) )
                                      for name in names ) )


def write_golden():
    os.makedirs( os.path.dirname( GOLDEN_PATH ), exist_ok=True )
    with open( GOLDEN_PATH, 'w', encoding='utf-8' ) as handle:
        json.dump( matrix(), handle, indent=1, sort_keys=True )
        handle.write( '\n' )
    print( "wrote %s"%( GOLDEN_PATH, ) )


if __name__ == '__main__':
    if sys.argv[1:] == [ 'write' ]:
        write_golden()
    else:
        unittest.main()
