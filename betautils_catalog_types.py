"""
betautils_catalog_types.py - which video type (structure profile) a
catalog item gets, and why, before and after it runs.

PRECEDENCE, highest first (spec section 3):

  1. pinned      the type the user chose for this one video
  2. forced      --profile / force_structure_profile
  3. path        profile_by_path_pattern
  4. folder      profile_by_source_dir, read under the 1.x roots and,
                 for a watched folder with its folder rule on, under
                 that folder
  5. measured    the shot-cut scan, at run time
  6. default

The pin is the user's word about one video, so it outranks every
standing rule, --profile included. Steps 2 to 6 are the engine's own
ladder (betautils_detector.profile_selection); this module adds the
pin above it and the watched folder's root beside the 1.x ones, and
the run uses the same function (declared_for), so what the library
shows is what a run does.
"""

import betautils_detector as bu_detector

PINNED = 'pinned'
# Before a run, nothing declared: the shot-cut scan decides at run time.
AT_RUN = 'at_run'


def profile_names( backend_name=None ):
    """Every structure profile a pin may name, manual-only ones included."""
    return sorted( ( bu_detector.get_profiles( backend_name ).get( 'variants' ) or {} ).keys() )


def declared_for( source_path, pin=None, folder_roots=(), backend_name=None ):
    """
    ( profile, Reason ) when a pin or a declaration decides, else None.

    folder_roots are the watched folders with the folder rule on that
    hold source_path, deepest first; each is tried after the 1.x roots.
    """
    if pin:
        if pin in profile_names( backend_name ):
            return ( pin, bu_detector.Reason( 'pinned in the library', PINNED ) )
        # A pin to a profile that is gone (config changed) is ignored, and
        # the library shows it: falling back silently would be wrong output.
    declared = bu_detector.declared_profile_selection( source_path, backend_name )
    if declared is not None:
        return declared
    for root in folder_roots:
        name, key = bu_detector.source_dir_rule( source_path, backend_name, uncensored_root=root )
        if name:
            return ( name, bu_detector.Reason( 'source folder %s/'%( key, ), 'source_folder', folder=key ) )
    return None


def shown_type( source_path, pin=None, folder_roots=(), last=None, backend_name=None ):
    """
    What the library shows: { profile, rule, params, when }.

    when is 'declared' (a pin or a rule decides before the run),
    'last_run' (measured on the item's last run) or 'at_run' (nothing
    declared and never run: the scan will decide).
    """
    declared = declared_for( source_path, pin, folder_roots, backend_name )
    if declared is not None:
        profile, reason = declared
        return { 'profile': profile, 'rule': reason.code, 'params': dict( reason.params ),
                 'when': 'declared' }
    if last and last.get( 'profile' ):
        return { 'profile': last['profile'], 'rule': last.get( 'rule' ),
                 'params': dict( last.get( 'params' ) or {} ), 'when': 'last_run' }
    return { 'profile': None, 'rule': None, 'params': {}, 'when': AT_RUN }


class RunTypes:
    """
    What a run asks the catalog before it picks a file's video type: the
    pin for that content, and the watched folders whose folder rule holds
    the file. pin_of( hash ) and roots_for( path ) are the catalog's; a
    lookup that fails is logged and the run goes on with the rules alone.
    """

    def __init__( self, pin_of, roots_for, logger=None ):
        self._pin_of = pin_of
        self._roots_for = roots_for
        self._logger = logger

    def declared( self, source_path, digest, backend_name=None ):
        try:
            pin = self._pin_of( digest )
            roots = self._roots_for( source_path )
        except Exception:
            if self._logger is not None:
                self._logger.warning( "the library catalog could not say this file's type; using the rules",
                                      exc_info=True )
            pin, roots = None, ()
        return declared_for( source_path, pin, roots, backend_name )


def pin_problem( pin, backend_name=None ):
    """None when pin may be stored, else the reason it may not."""
    if pin is None:
        return None
    if not isinstance( pin, str ) or pin not in profile_names( backend_name ):
        return 'unknown_type'
    return None
