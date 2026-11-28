"""
betautils_looks.py - named look presets.

A preset is a censor_style value with a name: choosing it sets a label's
item_overrides.<label>.censor_style to exactly that value, and a label
whose value equals a preset's shows that preset's name. Presets are data,
served to the app by config.schema, so a new one is a new entry here, and
a preset shared later as a file can use the same shape with its own
written name in place of name_key.

    id        stable name, [a-z][a-z0-9_]*
    version   bumped when the style changes, so a saved choice can tell
    name_key  message catalog key of its name
    style     a censor_style value: one style dict, or a weighted list

Pixelate's strength (20) was picked on the figure art in every style: it
reads clearly as a mosaic and leaves some tone, the middle ground between
14 (detail survives) and 28 or more (a few large blocks). Strength scales
with box size, so it reads the same on video.
"""

import copy

PRESETS = (
    { 'id': 'blur', 'version': 1, 'name_key': 'look.preset.blur',
      'style': { 'type': 'blur', 'method': 'triple_box', 'strength': 65 } },
    { 'id': 'pixelate', 'version': 1, 'name_key': 'look.preset.pixelate',
      'style': { 'type': 'pixel', 'pattern': 'square', 'strength': 20 } },
    { 'id': 'bars', 'version': 1, 'name_key': 'look.preset.bars',
      'style': { 'type': 'bar', 'color': [ 0, 0, 0 ], 'thickness': 1.0 } },
    { 'id': 'mixed', 'version': 1, 'name_key': 'look.preset.mixed',
      'style': [ { 'type': 'blur', 'method': 'triple_box', 'strength': 65, 'weight': 1 },
                 { 'type': 'pixel', 'pattern': 'square', 'strength': 20, 'weight': 1 },
                 { 'type': 'bar', 'color': [ 0, 0, 0 ], 'thickness': 1.0, 'weight': 1 } ] },
)


def presets():
    """Every preset, as plain JSON values (copies)."""
    return copy.deepcopy( list( PRESETS ) )


def _plain( value ):
    if isinstance( value, ( list, tuple ) ):
        return [ _plain( item ) for item in value ]
    if isinstance( value, dict ):
        return { key: _plain( item ) for key, item in value.items() }
    return value


def matching( style ):
    """The id of the preset whose style equals this censor_style value, or None."""
    plain = _plain( style )
    for preset in PRESETS:
        if _plain( preset['style'] ) == plain:
            return preset['id']
    return None
