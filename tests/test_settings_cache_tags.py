"""
test_settings_cache_tags.py - every registry entry moves exactly the
cache keys it says it moves.

The app warns before a change ("this re-detects", "this re-renders",
"this re-encodes") from each entry's tags, and a finished output is
reused when its keys match. A tag that is too narrow means a changed
setting reuses an old output and reports it as new; one that is too
wide re-renders for nothing and tells the user so.

Key side, for every entry and every instantiation of its placeholders:
set up the conditions its tags depend on, change it to its probe value,
and require the set of keys that moved to EQUAL its tags. Then, for each
condition, the same change with only that condition off must move
nothing, which proves the condition is real.

The effect side (a setting not tagged censor leaves a test render
unchanged) is in test_settings_render_effect.py.
"""

import copy
import unittest

import betaconfig
import betautils_detector as bu_detector
import betautils_person as bu_person
import betautils_settings as bu_settings

from tests import config_harness as harness

# Kept rendered whatever label is under test, so the run always renders something.
ANCHOR_LABELS = ( 'exposed_vulva', 'exposed_breast' )


def _variants( backend_name ):
    return bu_detector.variant_names( backend_name )


def instantiations( entry ):
    """
    ( concrete path, bindings ) for every useful filling of an entry's
    placeholders: each backend, its selected model variant, every label,
    a profile the backend has, each person backend.
    """
    combos = [ {} ]
    for part in entry.path:
        if part == '<backend>':
            combos = [ dict( c, backend=b ) for c in combos for b in bu_detector.registered_backend_names() ]
        elif part == '<model_variant>':
            expanded = []
            for c in combos:
                for variant in _variants( _backend( c ) )[:1]:
                    expanded.append( dict( c, model_variant=variant ) )
            combos = expanded
        elif part == '<label>':
            combos = [ dict( c, label=label ) for c in combos for label in bu_settings.label_names() ]
        elif part == '<profile>':
            combos = [ dict( c, profile='scene' ) for c in combos ]
        elif part == '<style_type>':
            combos = [ dict( c, style_type='blur' ) for c in combos ]
        elif part == '<person_backend>':
            combos = [ dict( c, person_backend=p ) for c in combos
                       for p in bu_person.registered_backend_names() ]
        elif part == '<folder>':
            combos = [ dict( c, folder='scene' ) for c in combos ]
    for bindings in combos:
        path = tuple( bindings.get( part.strip( '<>' ), part ) if part in bu_settings.PLACEHOLDERS
                      else part for part in entry.path )
        yield path, bindings


def _backend( bindings ):
    return bindings.get( 'backend', 'nudenet_v3' )


def _other_backend( name ):
    return [ b for b in bu_detector.registered_backend_names() if b != name ][0]


def set_up( condition, bindings, on, path ):
    """Make one condition hold (on) or fail (off) for this instantiation."""
    if condition == 'backend_selected':
        name = _backend( bindings )
        betaconfig.detector_backend['selected'] = name if on else _other_backend( name )
    elif condition == 'label_rendered':
        label = bindings['label']
        others = [ anchor for anchor in ANCHOR_LABELS if anchor != label ]
        betaconfig.items_to_censor = others + ( [ label ] if on else [] )
    elif condition == 'model_variant_selected':
        variant = bindings['model_variant']
        others = [ v for v in _variants( _backend( bindings ) ) if v != variant ]
        betaconfig.detector_backend[_backend( bindings )]['model_variant'] = variant if on else others[0]
    elif condition == 'person_backend_selected':
        name = bindings.get( 'person_backend', path[1] )
        other = [ p for p in bu_person.registered_backend_names() if p != name ][0]
        betaconfig.person_backend['selected'] = name if on else other
    elif condition == 'reverse_on':
        betaconfig.reverse['enabled'] = on
    elif condition == 'person_windows_on':
        betaconfig.reverse['enabled'] = True
        betaconfig.reverse['person_windows']['enabled'] = on
    elif condition == 'preview_on':
        betaconfig.preview_mode_enabled = on
    elif condition == 'preview_off':
        betaconfig.preview_mode_enabled = not on
    elif condition == 'cut_bursts_on':
        betaconfig.shot_cut_burst_window = 0.35 if on else 0
    elif condition == 'inherited_by_backend':
        # The selected backend must not set the key itself for the shared one to apply.
        name = 'nudenet_v3'
        betaconfig.detector_backend['selected'] = name
        if on:
            betaconfig.detector_backend[name].pop( path[-1], None )
        else:
            betaconfig.detector_backend[name][path[-1]] = copy.deepcopy( probe_for_own( path[-1] ) )
    else:
        raise AssertionError( "no set-up for condition %r"%( condition, ) )
    return None


def probe_for_own( key ):
    """A backend's own value for a key the shared defaults also set, for inherited_by_backend off."""
    return { 'model_variant': '320n', 'picture_sizes': [ 320 ] }[key]


def moved_by( entry, path, bindings, conditions_on, condition_off=None ):
    """The set of cache keys the probe moves under these conditions."""
    with harness.shipped():
        for condition in entry.when:
            set_up( condition, bindings, condition != condition_off, path )
        backend_name = bu_detector.selected_backend_name()
        before = harness.keys( backend_name )
        try:
            current = harness.get_path( path )
        except ( KeyError, IndexError, TypeError ):
            current = _MISSING
        probe = probe_for( entry, path, current, bindings )
        if current == probe:
            raise AssertionError( "%s: the probe equals the current value %r"%(
                '.'.join( path ), current ) )
        harness.set_path( path, probe )
        after = harness.keys( bu_detector.selected_backend_name() )
    return { kind for kind in before if before[kind] != after[kind] }


_MISSING = object()


def effective( path, current, bindings ):
    """What a setting resolves to now, through the per-label tiers when it is per-label."""
    if current is not _MISSING:
        return current
    if 'label' in bindings:
        import betautils_config as bu_config
        return bu_config.get_label_settings( bindings['label'], _backend( bindings ) ).get( path[-1] )
    return None


def probe_for( entry, path, current, bindings ):
    """
    The entry's probe, unless it equals what the setting already resolves
    to: then the opposite bool, or another valid choice.
    """
    probe = copy.deepcopy( entry.probe_for( path ) )
    now = effective( path, current, bindings )
    if entry.kind == 'bool':
        return not bool( now )
    if probe == now and entry.kind == 'choice':
        return [ choice for choice in entry.choice_list() if choice != now ][0]
    return probe


class TestCacheTagsAreExact( unittest.TestCase ):

    maxDiff = None

    def test_each_entry_moves_exactly_its_tags( self ):
        wrong = []
        for entry in bu_settings.REGISTRY:
            for path, bindings in instantiations( entry ):
                moved = moved_by( entry, path, bindings, entry.when )
                if moved != set( entry.moves ):
                    wrong.append( '%s: declares %s, moves %s'%(
                        '.'.join( path ), sorted( entry.moves ), sorted( moved ) ) )
        self.assertEqual( [], wrong )

    def test_each_condition_is_needed( self ):
        wrong = []
        for entry in bu_settings.REGISTRY:
            for condition in entry.when:
                for path, bindings in instantiations( entry ):
                    if condition == 'model_variant_selected' and len( _variants( _backend( bindings ) ) ) < 2:
                        continue
                    moved = moved_by( entry, path, bindings, entry.when, condition_off=condition )
                    if moved:
                        wrong.append( '%s with %s off: still moves %s'%(
                            '.'.join( path ), condition, sorted( moved ) ) )
        self.assertEqual( [], wrong )

    def test_every_entry_is_instantiated( self ):
        for entry in bu_settings.REGISTRY:
            with self.subTest( entry=entry.dotted ):
                self.assertTrue( list( instantiations( entry ) ) )


if __name__ == '__main__':
    unittest.main()
