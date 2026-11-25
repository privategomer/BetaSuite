"""
betautils_labels.py - BetaSuite's detection labels, described once.

Every label a detector adapter can report is a row here: its id (the
name used in settings and caches), the body part it covers, whether that
part is exposed or covered, which sex the model ties it to ('any' when the
model does not tell), which side of the figure shows it, and the colour
the debug overlay draws it in. betaconst.classes, the settings registry's
label choices and the desktop app's what-to-hide screen all read this
table, so a new label (one a newer model offers) is one new row, plus its
name in the message catalog and its boxes in the figure art.

No imports: betaconst reads this at import time.
"""

PARTS = ( 'face', 'breast', 'chest', 'armpits', 'belly', 'genitals', 'buttocks', 'anus', 'feet' )
STATES = ( 'exposed', 'covered', 'none' )
SEXES = ( 'female', 'male', 'any' )
VIEWS = ( 'front', 'back' )

# id, part, state, sex, view, debug overlay RGB. Order is the debug
# overlay's legacy order; the app orders by PARTS.
_ROWS = (
    ( 'exposed_anus',     'anus',     'exposed', 'any',    'back',  (  47,  79,  79 ) ),
    ( 'exposed_armpits',  'armpits',  'exposed', 'any',    'front', ( 139,  69,  19 ) ),
    ( 'covered_belly',    'belly',    'covered', 'any',    'front', (   0, 100,   0 ) ),
    ( 'exposed_belly',    'belly',    'exposed', 'any',    'front', (   0,   0, 139 ) ),
    ( 'covered_buttocks', 'buttocks', 'covered', 'any',    'back',  ( 255,   0,   0 ) ),
    ( 'exposed_buttocks', 'buttocks', 'exposed', 'any',    'back',  ( 255, 165,   0 ) ),
    ( 'face_femme',       'face',     'none',    'female', 'front', ( 255, 255,   0 ) ),
    ( 'face_masc',        'face',     'none',    'male',   'front', ( 199,  21, 133 ) ),
    ( 'covered_feet',     'feet',     'covered', 'any',    'front', (   0, 255,   0 ) ),
    ( 'exposed_feet',     'feet',     'exposed', 'any',    'front', (   0, 250, 154 ) ),
    ( 'covered_breast',   'breast',   'covered', 'female', 'front', (   0, 255, 255 ) ),
    ( 'exposed_breast',   'breast',   'exposed', 'female', 'front', (   0,   0, 255 ) ),
    ( 'covered_vulva',    'genitals', 'covered', 'female', 'front', ( 216, 191, 216 ) ),
    ( 'exposed_vulva',    'genitals', 'exposed', 'female', 'front', ( 255,   0, 255 ) ),
    ( 'exposed_chest',    'chest',    'exposed', 'male',   'front', (  30, 144, 255 ) ),
    ( 'exposed_penis',    'genitals', 'exposed', 'male',   'front', ( 240, 230, 140 ) ),
    ( 'covered_armpits',  'armpits',  'covered', 'any',    'front', ( 160,  82,  45 ) ),
    ( 'covered_anus',     'anus',     'covered', 'any',    'back',  ( 220,  20,  60 ) ),
)

LABELS = { row[0]: { 'id': row[0], 'part': row[1], 'state': row[2], 'sex': row[3], 'view': row[4] }
           for row in _ROWS }


def debug_colors():
    """{label: (r, g, b)} in the legacy order, for betaconst.classes."""
    return { row[0]: row[5] for row in _ROWS }


def describe():
    """Every label's description, ordered for display: by part, then exposed before covered, then sex."""
    def key( row ):
        return ( PARTS.index( row['part'] ), STATES.index( row['state'] ), SEXES.index( row['sex'] ), row['id'] )
    return [ dict( row ) for row in sorted( LABELS.values(), key = key ) ]
