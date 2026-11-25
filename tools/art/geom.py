"""Geometry and SVG helpers for the figure art generator. Stdlib only.

Points are (x, y) tuples in the 600 x 1200 figure canvas. Every number that
reaches an SVG file goes through a fixed format, so output is byte-identical
across runs and platforms.
"""
import math

W, H = 600, 1200


def mir(p):
    """Mirror points across the figure's vertical centre line."""
    return [(W - x, y) for x, y in p]


def shift(p, dx=0.0, dy=0.0):
    return [(x + dx, y + dy) for x, y in p]


def scale(p, cx, cy, k):
    return [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in p]


def spline(p, closed=True):
    """Catmull-Rom through the points, as cubic Bezier path data."""
    n = len(p)

    def at(i):
        return p[i % n] if closed else p[max(0, min(n - 1, i))]
    d = 'M%.1f %.1f' % p[0]
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = at(i - 1), at(i), at(i + 1), at(i + 2)
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += ' C%.1f %.1f %.1f %.1f %.1f %.1f' % (c1 + c2 + p2)
    return d + (' Z' if closed else '')


def poly(p, closed=True):
    d = 'M%.1f %.1f' % p[0] + ''.join(' L%.1f %.1f' % q for q in p[1:])
    return d + (' Z' if closed else '')


def ring(cx, cy, r, n=12, amp=0.0):
    return [(cx + r * math.cos(6.2832 * i / n) * (1 + amp * ((i * 7) % 3 - 1)),
             cy + r * 0.86 * math.sin(6.2832 * i / n)) for i in range(n)]


def wob(p, amp=2.4):
    """Deterministic hand-drawn wobble."""
    return [(x + amp * math.sin(y * 0.05 + i * 1.3), y + amp * math.cos(x * 0.045 + i * 0.9))
            for i, (x, y) in enumerate(p)]


def clip(p, a, b, c):
    """Keep the part of closed polygon p where a*x + b*y <= c (Sutherland-Hodgman)."""
    out = []
    n = len(p)
    for i in range(n):
        cur, nxt = p[i], p[(i + 1) % n]
        fc = a * cur[0] + b * cur[1] - c
        fn = a * nxt[0] + b * nxt[1] - c
        if fc <= 0:
            out.append(cur)
        if (fc < 0 < fn) or (fn < 0 < fc):
            t = fc / (fc - fn)
            out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
    return out


def above(p, y):
    """Part of polygon p with y <= the given value."""
    return clip(p, 0, 1, y)


def below(p, y):
    """Part of polygon p with y >= the given value."""
    return clip(p, 0, -1, -y)


def bbox(p):
    xs = [x for x, _ in p]
    ys = [y for _, y in p]
    return min(xs), min(ys), max(xs), max(ys)


def attrs(kw):
    return ' '.join('%s="%s"' % (k.replace('_', '-'), v) for k, v in kw.items())


def path(d, **kw):
    return '<path d="%s" %s/>' % (d, attrs(kw))


def mirror_g(inner):
    return '<g transform="translate(%d 0) scale(-1 1)">%s</g>' % (W, inner)


def box(label, x, y, w, h):
    return {'label': label, 'x': int(round(x)), 'y': int(round(y)),
            'w': int(round(w)), 'h': int(round(h))}


def mir_box(b):
    return box(b['label'], W - b['x'] - b['w'], b['y'], b['w'], b['h'])
