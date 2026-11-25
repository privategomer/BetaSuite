"""Compose figures from a body and a style, and author the label boxes.

A figure is (sex, view, dress): female or male, front or back, unclothed or
clothed. The figure's boxes stand in for detections; each lists the label a
detector would report there. Boxes are derived from the drawn shapes so they
follow each build's proportions.
"""
from anatomy import body, torso
from geom import H, W, above, bbox, below, box, clip, mir, mir_box, scale

SEXES = ('female', 'male')
VIEWS = ('front', 'back')
DRESSES = ('unclothed', 'clothed')


def _shoe(b, foot):
    x0, y0, x1, y1 = bbox(foot)
    return scale(foot, (x0 + x1) / 2, (y0 + y1) / 2 - 2, 1.16)


def _legs_feet(st, b, dress, back):
    o = ''
    legs = (b['leg_l'], mir(b['leg_l']))
    feet = (b['foot_l'], mir(b['foot_l']))
    if dress == 'unclothed':
        for f in feet:
            o += st.shape(f, 'hand')
    for lg in legs:
        o += st.shape(lg, 'limb', shade=(-10, -4))
    for side in (1, -1):
        kx = 270 if side == 1 else 330
        o += st.line([(kx - 14, 900), (kx, 910 if not back else 892), (kx + 14, 900)], 0.5)
    if dress == 'clothed':
        for lg, f in zip(legs, feet):
            o += st.shape(below(lg, 1052), 'shoe', shade=(-4, -3))
            shoe = _shoe(b, f)
            o += st.shape(shoe, 'shoe', shade=(-6, -4))
            y1 = bbox(shoe)[3]
            o += st.shape(below(shoe, y1 - 9), 'sole', outline=False)
    return o


def _torso(st, b):
    o = st.shape(torso(b), 'torso', shade=(-12, -4), outline=False)
    o += st.line(b['torso_l'], 1.0) + st.line(mir(b['torso_l']), 1.0)
    return o


def _shorts(st, b, back):
    o = ''
    for lg in (b['leg_l'], mir(b['leg_l'])):
        o += st.shape(above(lg, b['shorts_bottom']), 'shorts', shade=(-8, -3))
    o += st.shape(below(torso(b), b['shorts_top']), 'shorts', shade=(-10, -4))
    top = b['shorts_top']
    if back:
        for g, side in ((b['glute_l'], 1), (mir(b['glute_l']), -1)):
            o += st.shape(g, 'shorts', shade=(-8 * side, -10), outline=False)
        o += st.line(b['glute_edge_l'], 0.8) + st.line(mir(b['glute_edge_l']), 0.8)
        o += st.line(b['cleft'], 0.6)
    else:
        o += st.line([(300, top + 14), (300, b['torso_c'][1] - 6)], 0.4)
    # waistband
    xs = [x for x, y in torso(b) if abs(y - top) < 40]
    o += st.line([(min(xs) + 6, top + 14), (300, top + 18), (max(xs) - 6, top + 14)], 0.4)
    return o


def _tee(st, b, back):
    o = st.shape(above(torso(b), b['tee_hem']), 'tee', shade=(-12, -4))
    if not back:
        if b['sex'] == 'female':
            for p, e, side in ((b['breast_l'], b['breast_edge_l'], 1),
                               (mir(b['breast_l']), mir(b['breast_edge_l']), -1)):
                o += st.shape(p, 'tee', shade=(-7 * side, -12), outline=False)
                o += st.line(e, 0.8)
        else:
            o += st.line(b['pec_l'], 0.4) + st.line(mir(b['pec_l']), 0.4)
        nx = 278 if b['sex'] == 'female' else 274
        o += st.shape([(nx, 250), (288, 274), (300, 280), (312, 274), (600 - nx, 250), (300, 246)],
                      'limb', outline=False)
        o += st.line([(nx, 252), (288, 274), (300, 280), (312, 274), (600 - nx, 252)], 0.6)
    else:
        o += st.line([(276, 250), (300, 256), (324, 250)], 0.6)
    return o


def _raised_arm(st, b, dress, side):
    """side 1: viewer's left (front view); -1: viewer's right (back view)."""
    m = (lambda p: p) if side == 1 else mir
    o = ''
    if side == 1:
        o += st.shape(m(b['fore']), 'limb', shade=(-6, -4))
    o += st.shape(m(b['upper']), 'limb', shade=(9 * side, -3))
    if side == 1 and dress == 'unclothed':
        o += st.shape(m(b['pit']), 'pit', outline=False)
        o += st.line(m(b['pit_crease']), 0.45)
    if dress == 'clothed':
        a, bb, c = b['sleeve_cut']
        sleeve = clip(b['upper'], a, bb, c)
        o += st.shape(m(sleeve), 'tee', shade=(9 * side, -3))
        if side == 1:
            o += st.line(m(b['pit_crease']), 0.4)
    return o


def _hanging_arm(st, b, dress, side):
    arm = mir(b['arm_l']) if side == -1 else b['arm_l']
    o = st.shape(arm, 'limb', shade=(-9, -3))
    if dress == 'clothed':
        o += st.shape(above(arm, b['sleeve_y']), 'tee', shade=(-9, -3))
    return o


def front(st, b, dress):
    sex = b['sex']
    o = st.hair_behind(sex)
    o += _legs_feet(st, b, dress, back=False)
    o += st.shape(b['neck'], 'limb', shade=(0, -14))
    o += _torso(st, b)
    o += st.line(b['collar_l'], 0.5) + st.line(mir(b['collar_l']), 0.5)
    if dress == 'unclothed':
        if sex == 'female':
            for p, e, side in ((b['breast_l'], b['breast_edge_l'], 1),
                               (mir(b['breast_l']), mir(b['breast_edge_l']), -1)):
                o += st.shape(p, 'vol', shade=(-7 * side, -12), outline=False)
                o += st.line(e, 1.0)
            o += st.line([(300, 354), (300, 406)], 0.6)
            o += st.hatch([((262, 428), (258, 438)), ((278, 430), (276, 440)),
                           ((338, 428), (342, 438)), ((322, 430), (324, 440))])
        else:
            o += st.line(b['pec_l'], 0.7) + st.line(mir(b['pec_l']), 0.7)
            for ln in b['abs_lines']:
                o += st.line(ln, 0.4)
        cx, cy = b['nipple']
        one = st.nipple(cx, cy)
        o += one + '<g transform="translate(%d 0) scale(-1 1)">%s</g>' % (W, one)
        o += st.line(b['navel'], 0.5)
        o += st.shape(b['pelvis'], 'vol', shade=(-5, -9), outline=False)
        o += st.line(b['crease_l'], 0.8) + st.line(mir(b['crease_l']), 0.8)
        if sex == 'female':
            o += st.line([(300, 756), (300, 780)], 0.6)
        else:
            o += st.line(b['pelvis'][2:] + b['pelvis'][:1], 0.6)
        o += st.line(b['hip_line_l'], 0.45) + st.line(mir(b['hip_line_l']), 0.45)
    else:
        o += _shorts(st, b, back=False)
        o += _tee(st, b, back=False)
    o += _raised_arm(st, b, dress, 1)
    o += _hanging_arm(st, b, dress, -1)
    o += st.shape(b['head'], 'limb', shade=(-10, -6))
    o += st.hair_cap(sex)
    o += st.face(sex)
    return o


def back(st, b, dress):
    sex = b['sex']
    o = _legs_feet(st, b, dress, back=True)
    o += st.shape(b['neck'], 'limb', shade=(0, -14))
    o += _torso(st, b)
    o += st.line([(300, 262), (300, 330), (300, 450), (300, 570), (300, 690)], 0.5)
    sc = [(262, 318), (274, 336), (290, 346), (296, 338)]
    o += st.line(sc, 0.5) + st.line(mir(sc), 0.5)
    if dress == 'unclothed':
        o += st.line([(274, 676), (284, 680)], 0.4) + st.line([(326, 676), (316, 680)], 0.4)
        for g, e, side in ((b['glute_l'], b['glute_edge_l'], 1),
                           (mir(b['glute_l']), mir(b['glute_edge_l']), -1)):
            o += st.shape(g, 'vol', shade=(-8 * side, -12), outline=False)
            o += st.line(e, 1.0)
        o += st.line(b['cleft'], 0.7)
        o += st.hatch([((252, 814), (248, 824)), ((272, 816), (270, 826)),
                       ((348, 814), (352, 824)), ((328, 816), (330, 826))])
    else:
        o += _shorts(st, b, back=True)
        o += _tee(st, b, back=True)
    o += _hanging_arm(st, b, dress, 1)
    # the forearm passes behind the head, as on the front view
    o += st.shape(mir(b['fore']), 'limb', shade=(6, -4))
    o += _raised_arm(st, b, dress, -1)
    o += st.shape(b['head'], 'limb')
    o += st.hair_back_view(sex)
    return o


def figure_svg(st, sex, view, dress, viewbox=(0, 0, W, H), size=(W, H)):
    b = body(sex, st.build)
    st.begin()
    content = front(st, b, dress) if view == 'front' else back(st, b, dress)
    x, y, w, h = viewbox
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="%s %s %s %s" width="%d" height="%d">'
            '%s%s%s</svg>\n' % (x, y, w, h, size[0], size[1], st.defs(), st.bg(), content))


def _margin(p, m):
    x0, y0, x1, y1 = bbox(p)
    return x0 - m, y0 - m, x1 - x0 + 2 * m, y1 - y0 + 2 * m


def figure_boxes(build, sex, view, dress):
    """Authored boxes for one figure, in its 600 x 1200 canvas."""
    b = body(sex, build)
    cov = dress == 'clothed'

    def lab(exposed):
        return ('covered_' if cov else 'exposed_') + exposed
    out = []
    if view == 'front':
        x0, y0, x1, y1 = bbox(b['head'])
        out.append(box('face_femme' if sex == 'female' else 'face_masc',
                       x0 + 8, y0 + 22, x1 - x0 - 16, y1 - y0 - 26))
        if sex == 'female':
            br = box(lab('breast'), *_margin(b['breast_l'], 2))
            out += [br, mir_box(br)]
        elif not cov:
            ch = box('exposed_chest', *_margin(b['pec_l'] + [(b['pec_l'][0][0], 300)], 4))
            out += [ch, mir_box(ch)]
        out.append(box(lab('belly'), *b['belly_box']))
        out.append(box(lab('armpits'), *_margin(b['pit'], 8)))
        if sex == 'female':
            out.append(box(lab('vulva'), *_margin(b['pelvis'], 2)))
        elif not cov:
            out.append(box('exposed_penis', *_margin(b['pelvis'], 3)))
        for f in (b['foot_l'], mir(b['foot_l'])):
            out.append(box(lab('feet'), *_margin(_shoe(b, f) if cov else f, 3)))
    else:
        out.append(box(lab('buttocks'), *_margin(b['glute_l'] + mir(b['glute_l']), 2)))
        out.append(box(lab('anus'), *b['anus_box']))
    return out
