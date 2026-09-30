"""
bar_smoothing_variants.py - the variant matrix for run_bar_smoothing_test.py.

Edit this file directly to add/remove/adjust variants for a re-run - the
harness re-reads it fresh every time it starts (it's imported, not copied),
nothing here is cached anywhere else.

Each variant is one dict:
  name               - short, filesystem-safe label (used in the output
                        filename and in the log)
  description         - one line, printed in the log so you know what
                        you're looking at when you review results later
  censor_style        - None to leave betaconfig.py's current
                        item_overrides['exposed_breast']['censor_style']
                        untouched for this run, or a list of style dicts
                        (Python literal - same schema as betaconfig.py
                        itself, see README.md's "Censor styles" section)
                        to temporarily replace it with, just for this one
                        run
  position_smoothing  - None to leave betaconfig.py's current
                        default_position_smoothing untouched for this run,
                        or a number to temporarily replace it with, just
                        for this one run

Nothing here is ever written back to the real betaconfig.py permanently -
run_bar_smoothing_test.py always restores the original file's exact
content when it's done (or if it's interrupted).
"""

def _bar( thickness, feather, merge='none', span_extend=None, width_area_safety=0.00, height_area_safety=0.00 ):
    d = { 'type': 'bar', 'shape': 'box', 'color': (0, 0, 0), 'merge': merge,
          'thickness': thickness, 'feather': feather, 'weight': 1,
          'width_area_safety': width_area_safety, 'height_area_safety': height_area_safety }
    if span_extend is not None:
        d['span_extend'] = span_extend
    return [d]


# a single fixed bar used for every default_position_smoothing variant below,
# so the only thing visibly changing across those 5 renders is the
# smoothing/tracking behavior itself, not a randomized style pick too
_SMOOTHING_TEST_BAR = _bar( 0.35, 0.15 )


VARIANTS = [
    # --- standalone (merge: 'none') width sweep - your 3 currently-live
    #     thickness values, isolated one at a time instead of randomized
    #     across a weighted list, so each preview shows exactly one style
    #     consistently rather than a random per-track pick ---
    { 'name': 'standalone_thin_0.35', 'position_smoothing': None,
      'description': "standalone bar, thickness 0.35 (your current narrowest entry)",
      'censor_style': _bar( 0.35, 0.15 ) },
    { 'name': 'standalone_mid_0.525', 'position_smoothing': None,
      'description': "standalone bar, thickness 0.525 (your current mid entry)",
      'censor_style': _bar( 0.525, 0.225 ) },
    { 'name': 'standalone_wide_0.60', 'position_smoothing': None,
      'description': "standalone bar, thickness 0.60 (your current widest entry)",
      'censor_style': _bar( 0.60, 0.35 ) },

    # --- paired (merge: 'span', one bar drawn across both breasts) - same
    #     3 widths, span_extend 0.35 matching the existing covered_breast
    #     example in betaconfig.py (an already-used, reasonable value -
    #     not independently tuned for exposed_breast specifically) ---
    { 'name': 'paired_thin_0.35', 'position_smoothing': None,
      'description': "paired/span bar across both breasts, thickness 0.35, span_extend 0.35",
      'censor_style': _bar( 0.35, 0.15, merge='span', span_extend=0.35 ) },
    { 'name': 'paired_mid_0.525', 'position_smoothing': None,
      'description': "paired/span bar across both breasts, thickness 0.525, span_extend 0.35",
      'censor_style': _bar( 0.525, 0.225, merge='span', span_extend=0.35 ) },
    { 'name': 'paired_wide_0.60', 'position_smoothing': None,
      'description': "paired/span bar across both breasts, thickness 0.60, span_extend 0.35",
      'censor_style': _bar( 0.60, 0.35, merge='span', span_extend=0.35 ) },

    # --- tall & skinny - there's no direct "bar width" field (thickness
    #     only controls the box's HEIGHT extent); a tall/skinny look is
    #     approximated by thickness near 1.0 (full box height) combined
    #     with a negative width_area_safety (shrinks the box narrower than
    #     the raw detection). feather held at a fixed, modest 0.15 rather
    #     than scaled with thickness (a proportional feather at
    #     thickness=1.0 would be large enough to eat into real coverage -
    #     see README.md's feather guidance) ---
    { 'name': 'tall_skinny_moderate', 'position_smoothing': None,
      'description': "tall/skinny bar: thickness 1.0 (full height), width_area_safety -0.30",
      'censor_style': _bar( 1.0, 0.15, width_area_safety=-0.30 ) },
    { 'name': 'tall_skinny_narrow', 'position_smoothing': None,
      'description': "tall/skinny bar: thickness 1.0 (full height), width_area_safety -0.50",
      'censor_style': _bar( 1.0, 0.15, width_area_safety=-0.50 ) },

    # --- default_position_smoothing sweep - censor_style pinned to the
    #     standalone 0.35 bar for all 5, so the only thing visibly
    #     changing across these renders is the smoothing/tracking
    #     behavior, not the bar style too ---
    { 'name': 'smoothing_0.05', 'position_smoothing': 0.05,
      'description': "default_position_smoothing=0.05 (censor_style pinned to standalone 0.35 bar)",
      'censor_style': _SMOOTHING_TEST_BAR },
    { 'name': 'smoothing_0.10', 'position_smoothing': 0.10,
      'description': "default_position_smoothing=0.10 (censor_style pinned to standalone 0.35 bar)",
      'censor_style': _SMOOTHING_TEST_BAR },
    { 'name': 'smoothing_0.20', 'position_smoothing': 0.20,
      'description': "default_position_smoothing=0.20 - your current live value (censor_style pinned to standalone 0.35 bar)",
      'censor_style': _SMOOTHING_TEST_BAR },
    { 'name': 'smoothing_0.35', 'position_smoothing': 0.35,
      'description': "default_position_smoothing=0.35 (censor_style pinned to standalone 0.35 bar)",
      'censor_style': _SMOOTHING_TEST_BAR },
    { 'name': 'smoothing_0.50', 'position_smoothing': 0.50,
      'description': "default_position_smoothing=0.50 (censor_style pinned to standalone 0.35 bar)",
      'censor_style': _SMOOTHING_TEST_BAR },
]
