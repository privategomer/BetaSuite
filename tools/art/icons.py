#!/usr/bin/env python3
"""Write the app icon set's SVG sources: one folder per icon under assets/icons/.

    python3 tools/art/icons.py [--out assets/icons]

Stdlib only. Each icon is a navy tile with a white and amber mark, drawn
here from coordinates. The marks differ by shape, never by color alone.
Every folder gets:

- icon.svg: the master on a 64-unit grid, used for 48px and up
- icon-16.svg, icon-22.svg, icon-24.svg, icon-32.svg: small masters on the pixel grid
  (viewBox = the size), so tray-sized icons stay crisp. Marks built from
  cells or bars are snapped to whole pixels; the others are the master's
  mark scaled onto a pixel-snapped tile
- icon.json: the id, its catalog key, its place in the picker and which
  source each size is drawn from

The tile has a lighter rim, since the navy alone almost vanishes on a dark
panel. tools/art/rasterize_icons.py draws the PNGs and icon.ico from these.
Lines on stdout start with a level (INFO, DEBUG) for the wrapper's log.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_OUT = os.path.join(REPO, 'assets', 'icons')
FORMAT = 1

NAVY = '#1B2A3A'
RIM = '#5B738C'
WHITE = '#FFFFFF'
AMBER = '#F4A300'

# Picker order; the first is the default.
ICONS = ('censored_play', 'bars_shield', 'focus_blur', 'pixel_play', 'pixel_b')
SMALL = (16, 22, 24, 32)
SIZES = (16, 22, 24, 32, 48, 64, 128, 256, 512)

PLAY = ((22, 17), (22, 47), (48, 32))
PIXEL_PLAY = ((19, 15), (19, 49), (51, 32))
B_MASK = ('11110', '10001', '11110', '10001', '11110')
# The play mark on the small sizes' mosaic: 5 rows at 16px, 7 rows above.
PLAY_MASK_5 = ('10000', '11100', '11111', '11100', '10000')
PLAY_MASK_7 = ('1000000', '1110000', '1111100', '1111111', '1111100', '1110000', '1000000')
# The second shade of the small mosaics' checker: texture, not meaning.
SHADE = '#C9D4DF'


def _num(value):
    """A coordinate written without trailing zeros, so output is stable."""
    text = ('%.3f' % value).rstrip('0').rstrip('.')
    return '0' if text == '-0' else text


def _svg(size, body):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">\n%s\n</svg>\n'
            % (size, size, size, size, body))


def _tile_master():
    return ('<rect x="2.75" y="2.75" width="58.5" height="58.5" rx="12.5" fill="%s" stroke="%s" stroke-width="1.5"/>'
            % (NAVY, RIM))


def _tile_small(n):
    """The tile snapped to the pixel grid: a 1px rim on the outermost pixels."""
    r = max(2, round(n * 0.2))
    return ('<rect x="0.5" y="0.5" width="%d" height="%d" rx="%d" fill="%s" stroke="%s" stroke-width="1"/>'
            % (n - 1, n - 1, r, NAVY, RIM))


def _in_tri(px, py, tri):
    def side(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    a, b, c = tri
    p = (px, py)
    d1, d2, d3 = side(p, a, b), side(p, b, c), side(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def _snap(value):
    return int(math.floor(value + 0.5))


def _runs(pixels, fill):
    """Whole-pixel rects, one per horizontal run of set pixels."""
    out = []
    for y in sorted({p[1] for p in pixels}):
        xs = sorted(p[0] for p in pixels if p[1] == y)
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            out.append('<rect x="%d" y="%d" width="%d" height="1" fill="%s"/>' % (start, y, prev - start + 1, fill))
            if x is not None:
                start = prev = x
    return out


def _cells(mask, cell, gap, ox, oy, fill):
    out = []
    for r, row in enumerate(mask):
        for c, v in enumerate(row):
            if v == '1':
                out.append('<rect x="%s" y="%s" width="%s" height="%s" fill="%s"/>'
                           % (_num(ox + c * (cell + gap)), _num(oy + r * (cell + gap)), _num(cell), _num(cell), fill))
    return out


def _tri_mask(tri, grid, ox, oy, pitch):
    """Which cells of a grid have their centre inside a triangle, as mask rows."""
    return tuple(''.join('1' if _in_tri(ox + pitch * (c + 0.5), oy + pitch * (r + 0.5), tri) else '0'
                         for c in range(grid)) for r in range(grid))


# --- marks on the 64-unit master grid --------------------------------------

def _mark_master(key):
    if key == 'censored_play':
        return ['<path d="M22 17 L22 47 L48 32 Z" fill="%s"/>' % WHITE,
                '<rect x="14" y="28" width="36" height="8" fill="%s"/>' % AMBER]
    if key == 'bars_shield':
        return ['<path d="M32 11 L49 17 V31 C49 42 42 50 32 54 C22 50 15 42 15 31 V17 Z" fill="%s"/>' % WHITE,
                '<rect x="12" y="25" width="40" height="5" fill="%s"/>' % NAVY,
                '<rect x="12" y="34" width="40" height="5" fill="%s"/>' % NAVY]
    if key == 'focus_blur':
        # A radial gradient, not a blur filter: rasterizers differ on filters.
        return ['<defs><radialGradient id="halo" cx="32" cy="32" r="24" gradientUnits="userSpaceOnUse">'
                '<stop offset="0.3" stop-color="%s" stop-opacity="1"/>'
                '<stop offset="0.65" stop-color="%s" stop-opacity="0.55"/>'
                '<stop offset="1" stop-color="%s" stop-opacity="0"/></radialGradient></defs>' % (AMBER, AMBER, AMBER),
                '<circle cx="32" cy="32" r="24" fill="url(#halo)"/>',
                '<circle cx="32" cy="32" r="8" fill="%s"/>' % WHITE]
    if key == 'pixel_play':
        n, pitch, ox, oy = 9, 3.9, 16.0, 15.0
        return _cells(_tri_mask(PIXEL_PLAY, n, ox, oy, pitch), pitch - 0.8, 0.8, ox, oy, WHITE)
    if key == 'pixel_b':
        return _cells(B_MASK, 6, 1, 14.5, 14.5, WHITE)
    raise KeyError(key)


# --- marks on the pixel grid ------------------------------------------------

def _mosaic_small(n, mask, cell, nudge=0):
    """A cell mask as solid cells checkered in two light shades, centred on an n-pixel tile, moved nudge px right."""
    span = len(mask[0]) * cell
    ox, oy = (n - span) // 2 + nudge, (n - len(mask) * cell) // 2
    out = []
    for r, row in enumerate(mask):
        for c, v in enumerate(row):
            if v == '1':
                out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>'
                           % (ox + c * cell, oy + r * cell, cell, cell, WHITE if (r + c) % 2 == 0 else SHADE))
    return out


def _mark_small(key, n):
    s = n / 64.0
    if key == 'censored_play':
        tri = tuple((x * s, y * s) for x, y in PLAY)
        pix = {(x, y) for y in range(n) for x in range(n) if _in_tri(x + 0.5, y + 0.5, tri)}
        y0, y1 = _snap(28 * s), _snap(36 * s)
        x0, x1 = _snap(14 * s), _snap(50 * s)
        bar = '<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>' % (x0, y0, x1 - x0, max(2, y1 - y0), AMBER)
        return _runs(pix, WHITE) + [bar]
    if key == 'pixel_play':
        # A play mark is heavier on its left, so it sits 1px right of centre.
        if n < 20:
            return _mosaic_small(n, PLAY_MASK_5, 2, 1)
        return _mosaic_small(n, PLAY_MASK_7, 3 if n >= 32 else 2, 1)
    if key == 'pixel_b':
        return _mosaic_small(n, B_MASK, 2 if n < 20 else 4 if n >= 32 else 3)
    # Shapes that hold at small sizes: the master's mark, scaled.
    return ['<g transform="scale(%s)">' % _num(s)] + _mark_master(key) + ['</g>']


def master_svg(key):
    return _svg(64, '\n'.join([_tile_master()] + _mark_master(key)))


def small_svg(key, n):
    body = [_tile_small(n)]
    mark = _mark_small(key, n)
    if key in ('censored_play', 'pixel_play', 'pixel_b'):
        mark = ['<g shape-rendering="crispEdges">'] + mark + ['</g>']
    return _svg(n, '\n'.join(body + mark))


def source_for(size):
    return 'icon-%d.svg' % size if size in SMALL else 'icon.svg'


def files(key):
    """(relative path, text) for every file this script writes for one icon."""
    meta = {'format': FORMAT, 'id': key, 'name_key': 'app_icon.%s' % key, 'order': ICONS.index(key),
            'sizes': {str(n): {'png': 'icon-%d.png' % n, 'source': source_for(n)} for n in SIZES},
            'ico': 'icon.ico'}
    out = [('icon.svg', master_svg(key))]
    out += [('icon-%d.svg' % n, small_svg(key, n)) for n in SMALL]
    out.append(('icon.json', json.dumps(meta, indent=1, sort_keys=True) + '\n'))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    for key in ICONS:
        folder = os.path.join(args.out, key)
        os.makedirs(folder, exist_ok=True)
        for rel, text in files(key):
            with open(os.path.join(folder, rel), 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
            print('DEBUG wrote %s/%s' % (key, rel))
        print('INFO %s sources written' % key)
    return 0


if __name__ == '__main__':
    sys.exit(main())
