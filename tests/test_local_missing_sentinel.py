"""
test_local_missing_sentinel.py - a key the local config adds is removed,
not left as a stray sentinel, when the shipped values are rebuilt.

A local config that gives a label item_overrides it has none of in
betaconfig.py (the desktop app writes one whenever a look is chosen for
such a label) recorded the label's old value as a deep copy of MISSING.
The copy was not MISSING, so rebuilding betaconfig.py's own values put it
in as a value, and the engine's next config.get failed on it with a
TypeError. MISSING now survives copying.
"""

import copy
import os
import pickle
import tempfile
import unittest

import betaconfig
import betautils_config as bu_config
import betautils_config_store as bu_store
import betautils_local as bu_local

from tests import config_harness as harness

NEW_LABEL = 'covered_anus'   # no item_overrides entry in betaconfig.py
TEXT = ( "item_overrides = { '%s': { 'censor_style': { 'type': 'bar', 'color': ( 0, 0, 0 ), 'thickness': 1.0 } } }\n"
         "items_to_censor = [ '%s' ]\n" )%( NEW_LABEL, NEW_LABEL )


def _missing_anywhere( value ):
    if value is bu_local.MISSING or type( value ).__name__ == '_Missing':
        return True
    if isinstance( value, dict ):
        return any( _missing_anywhere( item ) for item in value.values() )
    if isinstance( value, ( list, tuple ) ):
        return any( _missing_anywhere( item ) for item in value )
    return False


class SentinelTest( unittest.TestCase ):

    def test_copies_are_the_same_object( self ):
        self.assertIs( copy.copy( bu_local.MISSING ), bu_local.MISSING )
        self.assertIs( copy.deepcopy( bu_local.MISSING ), bu_local.MISSING )
        self.assertIs( copy.deepcopy( { 'a': [ bu_local.MISSING ] } )['a'][0], bu_local.MISSING )
        self.assertIs( pickle.loads( pickle.dumps( bu_local.MISSING ) ), bu_local.MISSING )


class AddedKeyTest( unittest.TestCase ):

    def test_the_shipped_values_do_not_keep_an_added_label( self ):
        self.assertNotIn( NEW_LABEL, getattr( betaconfig, 'item_overrides', {} ) )
        with tempfile.TemporaryDirectory() as folder, harness.preserved():
            path = os.path.join( folder, 'betaconfig_local.py' )
            with open( path, 'w', encoding='utf-8' ) as handle:
                handle.write( TEXT )
            # The engine starts with the file already applied, as a run does.
            bu_local.apply( vars( betaconfig ), path=path, stderr=bu_store._Null() )
            bu_config.invalidate_config_caches()
            self.assertIn( NEW_LABEL, betaconfig.item_overrides )

            store = bu_store.ConfigStore( path=path )
            self.assertFalse( _missing_anywhere( store.base ) )
            self.assertNotIn( NEW_LABEL, store.base['item_overrides'] )

            evaluation = store.evaluate( store.text(), {}, extra=lambda _config: bu_config.label_views() )
            self.assertTrue( evaluation.ok, evaluation.errors )
            views = { view['label']: view for view in evaluation.extra }
            self.assertTrue( views[NEW_LABEL]['rendered'] )
            self.assertEqual( views[NEW_LABEL]['fields']['censor_style']['value']['type'], 'bar' )


if __name__ == '__main__':
    unittest.main()
