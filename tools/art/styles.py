"""The four art styles: how a body part, a line, a face and hair are drawn.

flat    flat 90s cartoon: clean uniform outline, flat fills
angular heavy angular: polygons, thick-thin outline, hard shadow planes
ink     ink and wash comic: wobbly double brush line, off-register wash
cel     hand-painted cel: thin brown line, soft gradient volumes

Every style offers the same primitives (shape, line, hatch) so one composer
draws all of them. Faces and hair are style-specific SVG. No text anywhere.
"""
from geom import H, W, mir, mirror_g, path, poly, ring, spline, wob

ROLES_SKIN = ('limb', 'torso', 'vol', 'hand', 'pit')

# shared hair shapes (flat, ink, cel)
HAIR_BACK_L = [(260, 62), (236, 100), (228, 160), (226, 230), (232, 290), (252, 296),
               (262, 250), (266, 200)]
HAIR_BACK = [(300, 54)] + HAIR_BACK_L + mir(HAIR_BACK_L)[::-1]
HAIR_CAP = [(250, 150), (246, 110), (262, 80), (300, 66), (338, 80), (354, 110),
            (350, 150), (338, 124), (318, 110), (300, 106), (282, 110), (262, 124)]
HAIR_FULL_BACK = [(300, 56), (344, 64), (368, 104), (376, 170), (380, 250), (372, 300),
                  (344, 312), (322, 290), (300, 300), (278, 290), (256, 312), (228, 300),
                  (220, 250), (224, 170), (232, 104), (256, 64)]
M_CAP = [(250, 146), (246, 106), (258, 80), (300, 62), (342, 80), (354, 106), (350, 146),
         (340, 118), (320, 102), (300, 98), (280, 102), (260, 118)]
M_BACK = [(300, 60), (340, 70), (356, 108), (358, 160), (350, 196), (300, 204), (250, 196),
          (242, 160), (244, 108), (260, 70)]


class Style:
    """Flat, ink and cel share this renderer (the shared-build styles)."""
    key = ''
    build = 'shared'
    ow, oc = 5, '#1a1214'
    skin = '#f3c9a8'
    hairc = '#5a2a1a'
    pit = '#e8b58f'
    shade = None            # (colour, opacity) of the hard skin shade, or None
    cloth = {}              # role -> (fill, shade colour)

    def __init__(s):
        s.n = 0

    def begin(s):
        s.n = 0

    # ---- primitives
    def pts(s, p):
        return p

    def d(s, p, closed=True):
        return spline(s.pts(p), closed)

    def fill(s, role):
        if role in s.cloth:
            return s.cloth[role][0]
        if role == 'hair':
            return s.hairc
        if role == 'pit':
            return s.pit
        return s.skin

    def shade_of(s, role):
        if role in s.cloth:
            return (s.cloth[role][1], 1.0)
        if role == 'hair':
            return None
        return s.shade

    def stroke(s, d, w=1.0):
        return path(d, fill='none', stroke=s.oc, stroke_width='%.1f' % (s.ow * w),
                    stroke_linecap='round', stroke_linejoin='round')

    def shape(s, p, role='limb', shade=None, outline=True, wid=1.0):
        d = s.d(p)
        out = [path(d, fill=s.fill(role))]
        sh = s.shade_of(role)
        if shade and sh:
            s.n += 1
            col, op = sh
            dx, dy = shade
            out.append('<clipPath id="c%d"><path d="%s"/></clipPath><g clip-path="url(#c%d)">'
                       '<path d="%s" fill="%s" fill-opacity="%s"/>'
                       '<path d="%s" transform="translate(%s %s)" fill="%s"/></g>'
                       % (s.n, d, s.n, d, col, op, d, dx, dy, s.fill(role)))
        if outline:
            out.append(s.stroke(d, wid))
        return ''.join(out)

    def line(s, p, w=0.6):
        return s.stroke(s.d(p, False), w)

    def hatch(s, pairs):
        return ''

    # ---- page
    def defs(s):
        return ''

    def bg(s):
        return ''

    # ---- head
    def hair_behind(s, sex):
        return s.shape(HAIR_BACK, 'hair') if sex == 'female' else ''

    def hair_cap(s, sex):
        return s.shape(HAIR_CAP if sex == 'female' else M_CAP, 'hair')

    def hair_back_view(s, sex):
        if sex == 'female':
            return s.shape(HAIR_FULL_BACK, 'hair') + s.line([(300, 70), (300, 150), (300, 230)], 0.4)
        return s.shape(M_BACK, 'hair')

    def face(s, sex):
        return ''

    def nipple(s, cx, cy):
        return ''


# =============================================================== flat
class Flat(Style):
    key = 'flat'
    ow, oc, skin, hairc = 5, '#1a1214', '#f3c9a8', '#5a2a1a'
    pit = '#e8b58f'
    cloth = {'tee': ('#3f8f8a', '#2f6f6b'), 'shorts': ('#2d3a66', '#222c4e'),
             'shoe': ('#f2efe8', '#c9c2b4'), 'sole': ('#c9c2b4', '#a8a092')}

    def bg(s):
        return ('<rect width="%d" height="%d" fill="#d98f6b"/>'
                '<rect y="1100" width="600" height="100" fill="#c47c59"/>' % (W, H))

    def face(s, sex):
        ol = '#1c1210'

        def ln(p, sw=3):
            return path(spline(p, False), fill='none', stroke=ol, stroke_width=sw,
                        stroke_linecap='round', stroke_linejoin='round')
        o = ''
        for cx in (283, 317):
            o += path('M%d 130 Q%d 123 %d 130 Q%d 137 %d 130 Z' % (cx - 11, cx, cx + 11, cx, cx - 11),
                      fill='#fff', stroke=ol, stroke_width=2.5)
            o += '<circle cx="%d" cy="131" r="3.6" fill="#4a6a9a"/>' % cx
            if sex == 'female':
                o += ln([(cx - 13, 127), (cx, 122), (cx + 13, 127)], 4)
            else:
                o += ln([(cx - 14, 116), (cx, 112), (cx + 14, 115)], 5)
        o += ln([(301, 138), (297, 158), (306, 160)], 3)
        if sex == 'female':
            o += path('M288 176 C295 171 305 171 312 176 C305 183 295 183 288 176 Z', fill='#b8433a',
                      stroke=ol, stroke_width=2.5, stroke_linejoin='round')
        else:
            o += ln([(288, 178), (300, 180), (312, 177)], 3.5)
        return '<g transform="translate(0 22)">%s</g>' % o

    def nipple(s, cx, cy):
        return (path(spline(ring(cx, cy, 9)), fill='#d98579', stroke=s.oc, stroke_width=2.5)
                + '<circle cx="%s" cy="%s" r="3.2" fill="#a8524c"/>' % (cx - 1, cy))


# =============================================================== ink
class Ink(Style):
    key = 'ink'
    ow, oc, skin, hairc = 6, '#1d1612', '#f7d6bc', '#d9a441'
    shade = ('#e59a78', 0.5)
    pit = '#eeb898'
    cloth = {'tee': ('#8db8dc', '#5f8fbf'), 'shorts': ('#e3a24a', '#b97a2a'),
             'shoe': ('#ffffff', '#c8c0b0'), 'sole': ('#b9b09e', '#8f8574')}

    def pts(s, p):
        return wob(p)

    def stroke(s, d, w=1.0):
        sw = s.ow * w
        return (path(d, fill='none', stroke=s.oc, stroke_width='%.1f' % sw, stroke_linecap='round',
                     stroke_linejoin='round')
                + path(d, transform='translate(1.8 1.4)', fill='none', stroke=s.oc,
                       stroke_width='%.1f' % (sw * 0.45), stroke_linecap='round',
                       stroke_linejoin='round', opacity='0.8'))

    def hatch(s, pairs):
        return ''.join(s.stroke(poly(list(q), False), 0.4) for q in pairs)

    def bg(s):
        return ('<rect width="%d" height="%d" fill="#fbf7ee"/>'
                '<path d="M120 1130 C200 1106 300 1140 400 1116 C470 1106 500 1130 520 1144 '
                'L520 1170 L110 1170 Z" fill="#a6b85a" opacity="0.6"/>' % (W, H))

    def face(s, sex):
        o = (s.line([(268, 152), (280, 144), (292, 152)], 0.8)
             + s.line([(308, 152), (320, 144), (332, 152)], 0.8)
             + s.line([(298, 160), (294, 174), (304, 174)], 0.6)
             + s.line([(282, 190), (300, 204), (320, 188)], 0.8))
        if sex == 'female':
            o += ('<circle cx="272" cy="176" r="9" fill="#e87c6a" opacity="0.5"/>'
                  '<circle cx="328" cy="176" r="9" fill="#e87c6a" opacity="0.5"/>')
        else:
            o += s.line([(266, 136), (292, 138)], 0.9) + s.line([(308, 138), (334, 136)], 0.9)
        return o

    def nipple(s, cx, cy):
        d = spline(wob(ring(cx, cy, 10, n=16), 0.8))
        return (path(d, transform='translate(2 1.5)', fill='#e0806e', opacity='0.7')
                + path(spline(wob(ring(cx, cy, 10, n=16), 0.8)[3:13], False), fill='none',
                       stroke=s.oc, stroke_width=3, stroke_linecap='round')
                + '<circle cx="%s" cy="%s" r="3" fill="%s"/>' % (cx - 1, cy + 1, s.oc))


# =============================================================== cel
_CEL_DEFS = (
    '<linearGradient id="limb" x1="0" x2="1" y1="0" y2="0">'
    '<stop offset="0" stop-color="#d79470"/><stop offset="0.35" stop-color="#f6c9a8"/>'
    '<stop offset="0.65" stop-color="#f8d3b6"/><stop offset="1" stop-color="#d28c68"/>'
    '</linearGradient>'
    '<linearGradient id="torso" x1="0" x2="1" y1="0" y2="0">'
    '<stop offset="0" stop-color="#d28c68"/><stop offset="0.3" stop-color="#f6c9a8"/>'
    '<stop offset="0.7" stop-color="#f8d3b6"/><stop offset="1" stop-color="#cf8864"/>'
    '</linearGradient>'
    '<radialGradient id="vol" cx="0.38" cy="0.35" r="0.7">'
    '<stop offset="0" stop-color="#fcdfc8"/><stop offset="0.55" stop-color="#f3c3a0"/>'
    '<stop offset="1" stop-color="#c0805c"/></radialGradient>'
    '<radialGradient id="pit" cx="0.5" cy="0.5" r="0.6">'
    '<stop offset="0" stop-color="#c98a68"/><stop offset="1" stop-color="#ecb894"/></radialGradient>'
    '<linearGradient id="hair" x1="0" x2="1" y1="0" y2="1">'
    '<stop offset="0" stop-color="#6e3a22"/><stop offset="1" stop-color="#3a1c12"/>'
    '</linearGradient>'
    '<linearGradient id="bg" x1="0" x2="0" y1="0" y2="1">'
    '<stop offset="0" stop-color="#2d2f5e"/><stop offset="1" stop-color="#6b3f56"/>'
    '</linearGradient>'
    '<radialGradient id="bl"><stop offset="0" stop-color="#e8707a" stop-opacity="0.55"/>'
    '<stop offset="1" stop-color="#e8707a" stop-opacity="0"/></radialGradient>'
    '<radialGradient id="ar"><stop offset="0" stop-color="#c8666a" stop-opacity="0.85"/>'
    '<stop offset="0.7" stop-color="#d47e78" stop-opacity="0.6"/>'
    '<stop offset="1" stop-color="#d98a7c" stop-opacity="0"/></radialGradient>')


def _grad(gid, edge, mid, light):
    return ('<linearGradient id="%s" x1="0" x2="1" y1="0" y2="0">'
            '<stop offset="0" stop-color="%s"/><stop offset="0.35" stop-color="%s"/>'
            '<stop offset="0.65" stop-color="%s"/><stop offset="1" stop-color="%s"/>'
            '</linearGradient>' % (gid, edge, mid, light, edge))


class Cel(Style):
    key = 'cel'
    ow, oc, skin, hairc = 3, '#5a2c20', '#f6c9a8', 'url(#hair)'
    cloth = {'tee': ('url(#tee)', None), 'shorts': ('url(#shorts)', None),
             'shoe': ('url(#shoe)', None), 'sole': ('#b8ae9c', None)}

    def fill(s, role):
        if role in s.cloth:
            return s.cloth[role][0]
        return {'limb': 'url(#limb)', 'torso': 'url(#torso)', 'vol': 'url(#vol)',
                'hand': '#f3c6a5', 'hair': 'url(#hair)', 'pit': 'url(#pit)'}[role]

    def shade_of(s, role):
        return None

    def defs(s):
        return ('<defs>' + _CEL_DEFS
                + _grad('tee', '#8a73b4', '#b7a3d8', '#c6b5e2')
                + _grad('shorts', '#3e5480', '#5b77a8', '#6a86b6')
                + _grad('shoe', '#c9c1b2', '#f4f1ea', '#fbf9f4')
                + '</defs>')

    def bg(s):
        return ('<rect width="%d" height="%d" fill="url(#bg)"/>'
                '<ellipse cx="300" cy="1130" rx="170" ry="20" fill="#000" opacity="0.25"/>' % (W, H))

    def face(s, sex):
        def eye():
            o = ('<path d="M269 158 C275 150 291 150 298 157 C291 163 276 163 269 158 Z" fill="#fff" '
                 'stroke="#3a1c12" stroke-width="1.8"/>'
                 '<circle cx="284" cy="157" r="6.5" fill="#7a4a2a"/>'
                 '<circle cx="284" cy="157" r="3.3" fill="#2a140c"/>'
                 '<circle cx="286.5" cy="154.5" r="2" fill="#fff"/>')
            if sex == 'female':
                o += ('<path d="M268 158 C273 148 292 147 299 156" fill="none" stroke="#2a140c" '
                      'stroke-width="3.2" stroke-linecap="round"/>'
                      '<path d="M269 158 l-6 -4 M272 152 l-5 -6" stroke="#2a140c" stroke-width="2" '
                      'stroke-linecap="round"/>'
                      '<path d="M266 144 C276 135 292 136 298 144" fill="none" stroke="#5a2c20" '
                      'stroke-width="2.4" stroke-linecap="round"/>')
            else:
                o += ('<path d="M268 157 C273 149 292 148 299 155" fill="none" stroke="#2a140c" '
                      'stroke-width="2.2" stroke-linecap="round"/>'
                      '<path d="M264 142 C276 136 292 136 299 140" fill="none" stroke="#3a1c12" '
                      'stroke-width="4.5" stroke-linecap="round"/>')
            return o
        o = eye() + mirror_g(eye())
        if sex == 'female':
            o += ('<ellipse cx="272" cy="186" rx="15" ry="10" fill="url(#bl)"/>'
                  '<ellipse cx="328" cy="186" rx="15" ry="10" fill="url(#bl)"/>')
        o += path(spline([(300, 166), (297, 182), (303, 184)], False), fill='none', stroke='#4a261c',
                  stroke_width=1.4, stroke_linecap='round', opacity='0.85')
        if sex == 'female':
            o += ('<path d="M286 202 C294 196 306 196 314 202 C306 212 294 212 286 202 Z" '
                  'fill="#c8506a" stroke="#5a2c20" stroke-width="1.8"/>'
                  '<ellipse cx="298" cy="201" rx="5" ry="1.8" fill="#fff" opacity="0.5"/>')
        else:
            o += ('<path d="M287 200 C295 197 305 197 313 200 C305 205 295 205 287 200 Z" '
                  'fill="#b8766a" stroke="#5a2c20" stroke-width="1.6"/>')
        return o

    def nipple(s, cx, cy):
        return ('<ellipse cx="%s" cy="%s" rx="12" ry="10.5" fill="url(#ar)"/>' % (cx, cy)
                + '<circle cx="%s" cy="%s" r="3.4" fill="#b85a5e" stroke="#8a3c3e" '
                  'stroke-width="0.8"/>' % (cx - 1, cy)
                + '<circle cx="%s" cy="%s" r="1.2" fill="#fff" opacity="0.7"/>' % (cx - 2, cy - 1.5))


# =============================================================== angular
B_HAIR = [(240, 170), (228, 130), (238, 96), (226, 70), (254, 76), (262, 48), (284, 66), (300, 40),
          (318, 66), (342, 48), (350, 76), (376, 70), (364, 98), (374, 132), (362, 170), (352, 130),
          (330, 108), (300, 102), (270, 108), (250, 130)]
B_HB_L = [(240, 130), (222, 180), (214, 250), (224, 304), (246, 262), (254, 200)]
B_HAIR_BACK = [(300, 60)] + B_HB_L + mir(B_HB_L)[::-1]
B_HAIR_FULL = [(300, 52), (346, 64), (372, 110), (380, 190), (376, 270), (352, 312), (330, 280),
               (300, 300), (270, 280), (248, 312), (224, 270), (220, 190), (228, 110), (254, 64)]
B_M_BACK = [(240, 170), (228, 130), (238, 96), (226, 70), (254, 76), (262, 48), (284, 66),
            (300, 40), (318, 66), (342, 48), (350, 76), (376, 70), (364, 98), (374, 132),
            (362, 170), (352, 196), (300, 206), (248, 196)]


class Angular(Style):
    key = 'angular'
    build = 'angular'
    ow, oc = 7, '#120d10'
    hairc = '#17121c'
    # role -> (base, mid shadow, deep shadow)
    tones = {'skin': ('#c4846a', '#8f4e43', '#5f3238'),
             'pit': ('#a8695a', '#8f4e43', '#5f3238'),
             'tee': ('#8a3036', '#6a2329', '#4a171c'),
             'shorts': ('#3a4560', '#2b344a', '#1e2536'),
             'shoe': ('#e6dccb', '#bfb3a0', '#8f8574'),
             'sole': ('#8f8574', '#6f6656', '#4f4838')}

    def d(s, p, closed=True):
        return poly(p, closed)

    def _tones(s, role):
        if role == 'hair':
            return None
        return s.tones.get(role, s.tones['skin'])

    def fill(s, role):
        if role == 'hair':
            return s.hairc
        return s._tones(role)[0]

    def shape(s, p, role='limb', shade=None, outline=True, wid=1.0):
        d = poly(p)
        base = s.fill(role)
        o = path(d, fill=base)
        t = s._tones(role)
        if shade and t:
            dx, dy = shade
            s.n += 1
            o += ('<clipPath id="k%d"><path d="%s"/></clipPath><g clip-path="url(#k%d)">'
                  % (s.n, d, s.n))
            o += path(d, fill=t[2])
            o += path(d, transform='translate(%.1f %.1f)' % (dx * 1.2, dy * 1.2), fill=t[1])
            o += path(d, transform='translate(%.1f %.1f)' % (dx * 2.5, dy * 2.5), fill=base)
            o += '</g>'
        if outline:
            o += path(d, fill='none', stroke=s.oc, stroke_width='%.1f' % (s.ow * wid),
                      stroke_linejoin='round')
            o += path(d, transform='translate(3 3)', fill='none', stroke=s.oc,
                      stroke_width='%.1f' % (s.ow * wid * 0.6), stroke_linejoin='round',
                      stroke_linecap='round')
        return o

    def line(s, p, w=0.6):
        return path(poly(p, False), fill='none', stroke=s.oc, stroke_width='%.1f' % (8 * w),
                    stroke_linecap='round', stroke_linejoin='round')

    def hatch(s, pairs):
        return ''.join(s.line(list(q), 0.4) for q in pairs)

    def bg(s):
        return ('<rect width="%d" height="%d" fill="#1f3b36"/>'
                '<polygon points="60,0 130,0 150,1200 40,1200" fill="#17302b"/>'
                '<polygon points="470,0 560,0 580,1200 450,1200" fill="#17302b"/>'
                '<polygon points="0,1110 600,1080 600,1200 0,1200" fill="#122621"/>' % (W, H))

    def hair_behind(s, sex):
        return s.shape(B_HAIR_BACK, 'hair', outline=False) if sex == 'female' else ''

    def hair_cap(s, sex):
        return path(poly(B_HAIR), fill=s.hairc, stroke=s.oc, stroke_width=5, stroke_linejoin='round')

    def hair_back_view(s, sex):
        if sex == 'female':
            return (path(poly(B_HAIR_FULL), fill=s.hairc)
                    + s.line([(300, 80), (300, 170), (300, 280)], 0.4))
        return path(poly(B_M_BACK), fill=s.hairc, stroke=s.oc, stroke_width=5,
                    stroke_linejoin='round')

    def face(s, sex):
        o = s.line([(266, 146), (292, 154)], 0.9) + s.line([(334, 146), (308, 154)], 0.9)
        o += ('<circle cx="282" cy="160" r="3.5" fill="#f4e6d8"/>'
              '<circle cx="318" cy="160" r="3.5" fill="#f4e6d8"/>')
        o += s.line([(300, 164), (296, 184), (306, 186)], 0.5)
        o += s.line([(284, 204), (316, 204)], 0.6)
        if sex == 'male':
            o += s.line([(262, 196), (276, 212)], 0.4) + s.line([(338, 196), (324, 212)], 0.4)
        return o

    def nipple(s, cx, cy):
        return (path(poly(ring(cx, cy, 9.5, n=6)), fill='#6e3236', stroke=s.oc, stroke_width=3.5,
                     stroke_linejoin='round')
                + path(poly(ring(cx - 2, cy - 2, 5, n=6)), fill='#8f4e43')
                + '<circle cx="%s" cy="%s" r="3" fill="%s"/>' % (cx - 1, cy, s.oc))


STYLES = {cls.key: cls for cls in (Flat, Angular, Ink, Cel)}
