#!/usr/bin/env python3
"""Rasterize the app icon set: PNGs, icon.ico, the shell's bundle icons, a check sheet.

    <art venv>/bin/python tools/art/rasterize_icons.py [--icons assets/icons] [--sheet DIR]

Needs cairosvg and Pillow (tools/art/requirements-art.txt), which the app
never uses. For every icon folder tools/art/icons.py wrote, draws a PNG at
each size from the source icon.json names, and icon.ico holding every size
up to 256 (the format's limit) as PNG entries. Writes manifest.json with the
sha256 of each PNG's SVG source and of the PNG, so the test suite can check
them without a rasterizer. Copies the default icon (icons.py ICONS[0]) into
ui/src-tauri/icons, which the shell's build compiles in as its window icon.
With --sheet, draws a legibility sheet there: every icon at every size on a
light, a dark and a grayscale panel, the master scaled down beside the
shipped small masters, and 16px magnified.
Lines on stdout start with a level (INFO, DEBUG) for the wrapper's log.
"""
import argparse
import hashlib
import io
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import icons  # noqa: E402

REPO = icons.REPO
DEFAULT_ICONS = icons.DEFAULT_OUT
TAURI_ICONS = os.path.join(REPO, 'ui', 'src-tauri', 'icons')
MANIFEST_FORMAT = 1
ICO_SIZES = tuple(n for n in icons.SIZES if n <= 256)
# The shell's bundle files: name -> pixel size (tauri.conf.json bundle.icon).
TAURI_PNGS = {'32x32.png': 32, '128x128.png': 128, '128x128@2x.png': 256, 'icon.png': 512}
# ICNS chunk types that hold a PNG, by pixel size.
ICNS_TYPES = ((16, b'icp4'), (32, b'icp5'), (64, b'icp6'), (128, b'ic07'), (256, b'ic08'), (512, b'ic09'))

LIGHT = (246, 246, 244)
DARK = (32, 33, 36)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def render(svg_text, size):
    import cairosvg
    return cairosvg.svg2png(bytestring=svg_text.encode('utf-8'), output_width=size, output_height=size)


def ico_bytes(pngs):
    """An ICO file whose entries are the given PNGs: [(size, png bytes)], size <= 256."""
    head = struct.pack('<HHH', 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, blobs = b'', b''
    for size, png in pngs:
        dim = 0 if size == 256 else size
        entries += struct.pack('<BBBBHHII', dim, dim, 0, 0, 1, 32, len(png), offset)
        blobs += png
        offset += len(png)
    return head + entries + blobs


def ico_sizes(data):
    """The pixel sizes an ICO file declares, in order."""
    _, kind, count = struct.unpack_from('<HHH', data, 0)
    if kind != 1:
        raise ValueError('not an icon file')
    return [(struct.unpack_from('<B', data, 6 + 16 * i)[0] or 256) for i in range(count)]


def icns_bytes(pngs):
    """An ICNS file of PNG chunks: {size: png bytes}."""
    body = b''.join(kind + struct.pack('>I', 8 + len(pngs[size])) + pngs[size]
                    for size, kind in ICNS_TYPES if size in pngs)
    return b'icns' + struct.pack('>I', 8 + len(body)) + body


def icons_in(root):
    found = [d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, 'icon.json'))]
    return sorted(found, key=lambda d: (icons.ICONS.index(d) if d in icons.ICONS else len(icons.ICONS), d))


def sources(root, key):
    """(size, png name, svg text of its source) for every size of one icon."""
    folder = os.path.join(root, key)
    with open(os.path.join(folder, 'icon.json'), encoding='utf-8') as f:
        meta = json.load(f)
    out = []
    for size in sorted(int(n) for n in meta['sizes']):
        entry = meta['sizes'][str(size)]
        with open(os.path.join(folder, entry['source']), encoding='utf-8') as f:
            out.append((size, entry['png'], f.read()))
    return meta, out


def write(path, data):
    with open(path, 'wb') as f:
        f.write(data)


def rasterize(root):
    manifest = {'format': MANIFEST_FORMAT, 'png': {}, 'ico': {}}
    for key in icons_in(root):
        meta, items = sources(root, key)
        pngs = {}
        for size, name, svg in items:
            png = render(svg, size)
            write(os.path.join(root, key, name), png)
            pngs[size] = png
            manifest['png']['%s/%s' % (key, name)] = {'svg_sha256': sha256(svg.encode('utf-8')),
                                                      'png_sha256': sha256(png)}
            print('DEBUG wrote %s/%s (%d bytes)' % (key, name, len(png)))
        ico = ico_bytes([(n, pngs[n]) for n in ICO_SIZES])
        write(os.path.join(root, key, meta['ico']), ico)
        manifest['ico']['%s/%s' % (key, meta['ico'])] = sha256(ico)
        print('INFO %s rasterized: %d PNGs and %s' % (key, len(pngs), meta['ico']))
    manifest['png'] = dict(sorted(manifest['png'].items()))
    manifest['ico'] = dict(sorted(manifest['ico'].items()))
    import cairosvg
    manifest['rasterizer'] = 'cairosvg ' + cairosvg.__version__
    with open(os.path.join(root, 'manifest.json'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print('INFO wrote manifest.json (%d PNGs, %d icon files)' % (len(manifest['png']), len(manifest['ico'])))


def copy_default(root, dest):
    """The default icon as the shell's bundle icons."""
    key = icons.ICONS[0]
    folder = os.path.join(root, key)

    def png(size):
        with open(os.path.join(folder, 'icon-%d.png' % size), 'rb') as f:
            return f.read()
    for name, size in TAURI_PNGS.items():
        write(os.path.join(dest, name), png(size))
    with open(os.path.join(folder, 'icon.ico'), 'rb') as f:
        write(os.path.join(dest, 'icon.ico'), f.read())
    write(os.path.join(dest, 'icon.icns'), icns_bytes({size: png(size) for size, _ in ICNS_TYPES}))
    with open(os.path.join(folder, 'icon.svg'), 'rb') as f:
        write(os.path.join(dest, 'source.svg'), f.read())
    print('INFO shell bundle icons drawn from %s' % key)


def sheet(root, out_dir):
    """The legibility sheet: for each icon, the master scaled and the shipped set, on light, dark and gray."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default()

    def on_panel(svg, n, bg, gray):
        icon = Image.open(io.BytesIO(render(svg, n))).convert('RGBA')
        base = Image.new('RGBA', icon.size, bg + (255,))
        base.alpha_composite(icon)
        tile = base.convert('RGB')
        return tile.convert('L').convert('RGB') if gray else tile
    sizes = (16, 22, 24, 32, 48, 64)
    keys = icons_in(root)
    pad, label_w, mag = 10, 170, 6
    strip_w = sum(sizes) + pad * (len(sizes) + 1) + 16 * mag + pad
    row_h = max(max(sizes) + 30, 16 * mag + 10)
    panels = (('light', LIGHT), ('dark', DARK), ('gray on dark', DARK))
    width = label_w + len(panels) * (strip_w + pad)
    height = 40 + len(keys) * 2 * row_h
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((pad, 12), 'App icons at real size. Per icon: the 64-unit master scaled (top), the shipped set (bottom). '
              'Gray panel: no meaning carried by color alone.', fill=(0, 0, 0), font=font)
    for p, (name, _) in enumerate(panels):
        draw.text((label_w + p * (strip_w + pad) + pad, 26), name, fill=(0, 0, 0), font=font)
    for i, key in enumerate(keys):
        meta, items = sources(root, key)
        with open(os.path.join(root, key, 'icon.svg'), encoding='utf-8') as f:
            master = f.read()
        shipped = {size: svg for size, _, svg in items}
        for j, (title, svg_for) in enumerate((('master scaled', lambda n: master), ('shipped', lambda n: shipped[n]))):
            y = 40 + (i * 2 + j) * row_h
            draw.text((pad, y + 10), '%s: %s' % (key, title), fill=(0, 0, 0), font=font)
            for p, (pname, bg) in enumerate(panels):
                x0 = label_w + p * (strip_w + pad)
                draw.rectangle([x0, y, x0 + strip_w, y + row_h - 4], fill=bg)
                x = x0 + pad
                for n in sizes:
                    tile = on_panel(svg_for(n), n, bg, pname.startswith('gray'))
                    img.paste(tile, (x, y + 6))
                    draw.text((x, y + 8 + max(sizes)), str(n), fill=(140, 140, 140), font=font)
                    x += n + pad
                big = on_panel(svg_for(16), 16, bg, pname.startswith('gray')).resize((16 * mag, 16 * mag), Image.NEAREST)
                img.paste(big.crop((0, 0, 16 * mag, row_h - 8)), (x, y + 2))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'icon_sheet.png')
    img.save(path)
    print('INFO legibility sheet: %s' % path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--icons', default=DEFAULT_ICONS)
    ap.add_argument('--tauri-icons', default=TAURI_ICONS)
    ap.add_argument('--sheet', help='folder for the legibility sheet (none drawn without it)')
    args = ap.parse_args(argv)
    rasterize(args.icons)
    copy_default(args.icons, args.tauri_icons)
    if args.sheet:
        sheet(args.icons, args.sheet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
