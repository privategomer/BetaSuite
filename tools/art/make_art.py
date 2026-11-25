#!/usr/bin/env python3
"""Generate the figure art: SVG figures and a style.json per art style.

    python3 tools/art/make_art.py [--out assets/art] [--style flat ...]

Stdlib only and deterministic: the same code writes the same bytes, which
the test suite checks. Rasterizing is a separate step (rasterize.py), since
it needs a rasterizer the app does not ship.

Per style folder (assets/art/<style>/):
  figures/<sex>_<view>_<dress>.svg   600 x 1200 figures
  style.json                         figures and tiles, each with its
                                     authored boxes standing in for
                                     detections. A tile is a square crop
                                     of a figure (window) showing one label.
Lines on stdout start with a level (INFO, DEBUG) for the wrapper's log.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compose import DRESSES, SEXES, VIEWS, figure_boxes, figure_svg  # noqa: E402
from geom import H, W  # noqa: E402
from styles import STYLES  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_OUT = os.path.join(REPO, 'assets', 'art')
FORMAT = 1
TILE_SIZE = 384
TILE_ZOOM = 1.8        # window side over the label's box side
TILE_MIN = 220         # smallest window, in figure pixels


def figure_id(sex, view, dress):
    return '%s.%s.%s' % (sex, view, dress)


def figure_file(sex, view, dress):
    return 'figures/%s_%s_%s' % (sex, view, dress)


def tile_window(boxes):
    x0 = min(b['x'] for b in boxes)
    y0 = min(b['y'] for b in boxes)
    x1 = max(b['x'] + b['w'] for b in boxes)
    y1 = max(b['y'] + b['h'] for b in boxes)
    side = int(round(max(TILE_MIN, TILE_ZOOM * max(x1 - x0, y1 - y0))))
    side = min(side, W)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x = int(round(min(max(cx - side / 2, 0), W - side)))
    y = int(round(min(max(cy - side / 2, 0), H - side)))
    return [x, y, side]


def tile_boxes(boxes, window, size=TILE_SIZE):
    wx, wy, ws = window
    k = size / ws
    out = []
    for b in boxes:
        x = max(0, round((b['x'] - wx) * k))
        y = max(0, round((b['y'] - wy) * k))
        x1 = min(size, round((b['x'] + b['w'] - wx) * k))
        y1 = min(size, round((b['y'] + b['h'] - wy) * k))
        out.append({'label': b['label'], 'x': x, 'y': y, 'w': x1 - x, 'h': y1 - y})
    return out


def build_style(key):
    st = STYLES[key]()
    figures, tiles, svgs = {}, {}, {}
    for sex in SEXES:
        for view in VIEWS:
            for dress in DRESSES:
                fid = figure_id(sex, view, dress)
                stem = figure_file(sex, view, dress)
                svgs[stem + '.svg'] = figure_svg(st, sex, view, dress)
                boxes = figure_boxes(st.build, sex, view, dress)
                figures[fid] = {'file': stem + '.png', 'source': stem + '.svg',
                                'canvas': [W, H], 'boxes': boxes}
                for label in sorted({b['label'] for b in boxes}):
                    own = [b for b in boxes if b['label'] == label]
                    window = tile_window(own)
                    tid = '%s.%s' % (label, sex)
                    if tid in tiles:
                        continue    # faces show on two figures; the first one wins
                    tiles[tid] = {'file': 'tiles/%s_%s.png' % (label, sex), 'figure': fid,
                                  'label': label, 'sex': sex, 'window': window,
                                  'canvas': [TILE_SIZE, TILE_SIZE],
                                  'boxes': tile_boxes(own, window)}
    meta = {'format': FORMAT, 'id': key, 'name_key': 'art_style.%s' % key,
            'figures': figures, 'tiles': dict(sorted(tiles.items()))}
    return meta, svgs


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--style', action='append', choices=sorted(STYLES))
    args = ap.parse_args(argv)
    for key in args.style or sorted(STYLES):
        meta, svgs = build_style(key)
        root = os.path.join(args.out, key)
        for rel, text in sorted(svgs.items()):
            write(os.path.join(root, rel), text)
            print('DEBUG wrote %s/%s (%d bytes)' % (key, rel, len(text)))
        write(os.path.join(root, 'style.json'), json.dumps(meta, indent=1, sort_keys=True) + '\n')
        print('INFO %s: %d figures, %d tiles' % (key, len(meta['figures']), len(meta['tiles'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
