#!/usr/bin/env python3
"""Rasterize the figure art to the PNGs the engine reads.

    <art venv>/bin/python tools/art/rasterize.py [--art assets/art] [--style flat ...]

Needs cairosvg (tools/art/requirements-art.txt), which the app never uses.
Writes each figure's PNG and each tile's PNG (a square crop of a figure),
then manifest.json, which records for every PNG the sha256 of the exact SVG
it came from and of the PNG itself. The test suite checks those hashes, so a
PNG that no longer matches its SVG fails without needing a rasterizer.
Lines on stdout start with a level (INFO, DEBUG) for the wrapper's log.
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_ART = os.path.join(REPO, 'assets', 'art')
MANIFEST_FORMAT = 1

_HEAD = re.compile(r'^<svg xmlns="http://www.w3.org/2000/svg" viewBox="[^"]*" width="\d+" height="\d+">')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tile_svg(figure_text, window, size):
    """The figure SVG cropped to a square window and drawn at size x size."""
    x, y, s = window
    head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="%d %d %d %d" width="%d" height="%d">'
            % (x, y, s, s, size, size))
    out, n = _HEAD.subn(head, figure_text, count=1)
    if n != 1:
        raise ValueError('unexpected SVG header')
    return out


def sources(art_root, key):
    """(png relative path, svg text) for every PNG of one style, in a fixed order."""
    root = os.path.join(art_root, key)
    with open(os.path.join(root, 'style.json'), encoding='utf-8') as f:
        meta = json.load(f)
    texts = {}
    out = []
    for fid in sorted(meta['figures']):
        fig = meta['figures'][fid]
        with open(os.path.join(root, fig['source']), encoding='utf-8') as f:
            texts[fid] = f.read()
        out.append((fig['file'], texts[fid]))
    for tid in sorted(meta['tiles']):
        t = meta['tiles'][tid]
        out.append((t['file'], tile_svg(texts[t['figure']], t['window'], t['canvas'][0])))
    return out


def styles_in(art_root):
    return sorted(d for d in os.listdir(art_root)
                  if os.path.isfile(os.path.join(art_root, d, 'style.json')))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--art', default=DEFAULT_ART)
    ap.add_argument('--style', action='append')
    args = ap.parse_args(argv)
    import cairosvg

    manifest_path = os.path.join(args.art, 'manifest.json')
    manifest = {'format': MANIFEST_FORMAT, 'rasterizer': 'cairosvg ' + cairosvg.__version__,
                'png': {}}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            old = json.load(f)
        manifest['png'] = old.get('png', {})
    for key in args.style or styles_in(args.art):
        manifest['png'] = {k: v for k, v in manifest['png'].items() if not k.startswith(key + '/')}
        for rel, svg in sources(args.art, key):
            png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
            dest = os.path.join(args.art, key, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, 'wb') as f:
                f.write(png)
            manifest['png']['%s/%s' % (key, rel)] = {'svg_sha256': sha256(svg.encode('utf-8')),
                                                     'png_sha256': sha256(png)}
            print('DEBUG wrote %s/%s (%d bytes)' % (key, rel, len(png)))
        print('INFO %s rasterized' % key)
    manifest['png'] = dict(sorted(manifest['png'].items()))
    with open(manifest_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print('INFO wrote %s (%d PNGs)' % (os.path.relpath(manifest_path, REPO), len(manifest['png'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
