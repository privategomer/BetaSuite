#!/usr/bin/env python3
"""Contact sheets of the rasterized figure art, optionally with a censor look.

    python tools/art/contact_sheet.py --out DIR [--style flat] [--look blur ...]

One PNG per style (and per look): the eight figures on top, the tiles below,
each with its authored boxes. With --look, the look is drawn by the engine's
own censor code (betautils_censor.censor_img_for_boxes) onto every box;
without it, boxes are outlined. Looks: bar, blur, pixel or pixel:<strength>.
Captions are drawn here for review only; the art itself carries no text.
Lines on stdout start with a level for the wrapper's log.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ART = os.path.join(REPO, 'assets', 'art')
FIG_H, TILE = 360, 180
LOOKS = {
    'bar': {'type': 'bar', 'color': (0, 0, 0), 'thickness': 1.0},
    'blur': {'type': 'blur', 'method': 'triple_box', 'strength': 65},
    'pixel': {'type': 'pixel', 'pattern': 'square', 'strength': 20},
}


def look_style(name):
    if name.startswith('pixel:'):
        return dict(LOOKS['pixel'], strength=float(name.split(':', 1)[1]))
    return dict(LOOKS[name])


def draw(img, boxes, look):
    if look is None:
        for b in boxes:
            cv2.rectangle(img, (b['x'], b['y']), (b['x'] + b['w'], b['y'] + b['h']), (0, 200, 0), 2)
        return img
    import betautils_censor
    style = look_style(look)
    return betautils_censor.censor_img_for_boxes(
        img, [dict(b, censor_style=dict(style), censor_shape='box') for b in boxes])


def caption(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (40, 30, 24), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def sheet(key, look):
    root = os.path.join(ART, key)
    with open(os.path.join(root, 'style.json'), encoding='utf-8') as f:
        meta = json.load(f)
    figs = []
    for fid in sorted(meta['figures']):
        fig = meta['figures'][fid]
        img = draw(cv2.imread(os.path.join(root, fig['file'])), fig['boxes'], look)
        img = cv2.resize(img, (FIG_H // 2, FIG_H), interpolation=cv2.INTER_AREA)
        figs.append(caption(img, fid))
    top = np.hstack(figs)
    cols = top.shape[1] // TILE
    tiles = []
    for tid in sorted(meta['tiles']):
        t = meta['tiles'][tid]
        img = draw(cv2.imread(os.path.join(root, t['file'])), t['boxes'], look)
        tiles.append(caption(cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA), tid))
    while len(tiles) % cols:
        tiles.append(np.full((TILE, TILE, 3), 40, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    grid = np.vstack(rows)
    pad = top.shape[1] - grid.shape[1]
    if pad > 0:
        grid = np.hstack([grid, np.full((grid.shape[0], pad, 3), 40, np.uint8)])
    return np.vstack([top, grid])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--style', action='append')
    ap.add_argument('--look', action='append')
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    keys = args.style or sorted(d for d in os.listdir(ART)
                                if os.path.isfile(os.path.join(ART, d, 'style.json')))
    for key in keys:
        for look in args.look or [None]:
            name = '%s_%s.png' % (key, (look or 'boxes').replace(':', ''))
            cv2.imwrite(os.path.join(args.out, name), sheet(key, look))
            print('INFO wrote %s' % os.path.join(args.out, name))
    return 0


if __name__ == '__main__':
    sys.exit(main())
