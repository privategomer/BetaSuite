"""
betautils_track.py - Everything between "the model said so" and "draw it".

The detection -> censoring pipeline, in order, with the function here
that owns each stage:

    1. apply_cross_size_dedup   collapse the same object detected at two
                                picture sizes into one detection
    2. apply_geometry_filter    drop detections whose box shape is
                                implausible for their label
    3. apply_class_suppression  drop detections that are more likely a
                                misclassification of a nearby, more
                                confident detection of a different label
    4. (betautils_censor.process_raw_box, per detection)
    5. smooth_boxes             match detections into per-instance
                                tracks, resolve one style per track,
                                smooth position, interpolate gaps,
                                confirm tracks, and drop what fails

These lived inside betatv.py before 2.1, which meant the replay/tuning
tools could not import them: tools/tuning/replay_tune.py literally read
betatv.py's source text, sliced the two function bodies out of it, and
exec'd them, so that it could be sure it was testing the real code. That
is the same "two copies drift apart" failure mode the cache-path module
exists to prevent. They are a module now.

BOX VOCABULARY
--------------
raw box       what a detector adapter returns
              {'x','y','w','h','class_id','score','t','size'}
censorable box  what process_raw_box returns: raw geometry padded by the
              label's area safety, clamped to the frame, with a
              start/end time window and an unresolved censor_style
rendered box  what smooth_boxes returns: the same dicts, smoothed, with
              one concrete censor_style/censor_shape per track, plus
              synthetic interpolated boxes
"""

import math
import bisect
import contextlib
import random

import betaconfig
import betautils_censor as bu_censor
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_log as bu_log


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def intersection_over_union( box_a, box_b ):
    """
    IoU of two boxes given as dicts with x/y/w/h.

    Returns:
        A float in [0, 1]. 0 when they do not overlap or either has no
        area.
    """
    a_x2, a_y2 = box_a['x'] + box_a['w'], box_a['y'] + box_a['h']
    b_x2, b_y2 = box_b['x'] + box_b['w'], box_b['y'] + box_b['h']
    intersect_w = min( a_x2, b_x2 ) - max( box_a['x'], box_b['x'] )
    intersect_h = min( a_y2, b_y2 ) - max( box_a['y'], box_b['y'] )
    if intersect_w <= 0 or intersect_h <= 0:
        return 0.0
    intersection = intersect_w * intersect_h
    union = box_a['w']*box_a['h'] + box_b['w']*box_b['h'] - intersection
    return intersection/union if union > 0 else 0.0


def _group_by_instant( raw_boxes ):
    """
    Index raw boxes by their exact timestamp.

    Grouping on the literal float value of 't' is safe, and equivalent
    to the tolerance comparison it replaced: every box from one sampled
    frame is stamped with the same t value, computed once, not
    recomputed per box.

    Returns:
        {t: [index, ...]} into raw_boxes.
    """
    by_t = {}
    for index, raw in enumerate( raw_boxes ):
        by_t.setdefault( raw['t'], [] ).append( index )
    return by_t


# ---------------------------------------------------------------------------
# 1. Cross-size dedup
# ---------------------------------------------------------------------------

def apply_cross_size_dedup( raw_boxes, settings=None ):
    """
    Collapse one real object detected at two different picture sizes.

    With more than one entry in picture_sizes the model runs once per
    size and the same object produces one detection per size, at the
    same timestamp with the same label. Nothing merged them before 2.1:
    class_suppression only ever compares DIFFERENT labels, so both
    survived, both became censorable boxes, and smooth_boxes built two
    separate tracks for one instance - which then independently resolved
    two different randomised styles and rendered on top of each other.

    Deliberately scoped to pairs from DIFFERENT sizes. Two same-label
    detections from the SAME size are what a detector legitimately
    produces for two real instances (two breasts, two people), and
    merging those would destroy real detections. Because of that scoping
    this function is provably a no-op when picture_sizes has one entry,
    which is the common configuration.

    Args:
        raw_boxes: Raw detections across every size.
        settings: betaconfig.cross_size_dedup. Defaults are read from
            betaconfig when omitted.

    Returns:
        A (surviving_boxes, dedup_counts) pair. dedup_counts maps label
        to how many detections were removed, so a multi-size run can
        report how much duplication there actually was.
    """
    if settings is None:
        settings = getattr( betaconfig, 'cross_size_dedup', {} ) or {}
    if not settings.get( 'enabled', True ):
        return raw_boxes, {}

    iou_threshold = settings.get( 'iou_threshold', 0.60 )

    distinct_sizes = { raw.get( 'size', 0 ) for raw in raw_boxes }
    if len( distinct_sizes ) < 2:
        return raw_boxes, {}

    removed = set()
    counts = {}
    for indices in _group_by_instant( raw_boxes ).values():
        if len( indices ) < 2:
            continue
        # Highest score first, so the survivor of any pair is the more
        # confident detection.
        ordered = sorted( indices, key=lambda i: -raw_boxes[i]['score'] )
        for position, keeper_index in enumerate( ordered ):
            if keeper_index in removed:
                continue
            keeper = raw_boxes[keeper_index]
            for candidate_index in ordered[ position+1 : ]:
                if candidate_index in removed:
                    continue
                candidate = raw_boxes[candidate_index]
                if candidate['class_id'] != keeper['class_id']:
                    continue
                if candidate.get( 'size', 0 ) == keeper.get( 'size', 0 ):
                    continue
                if intersection_over_union( keeper, candidate ) >= iou_threshold:
                    removed.add( candidate_index )
                    counts[ candidate['class_id'] ] = counts.get( candidate['class_id'], 0 ) + 1

    if not removed:
        return raw_boxes, {}
    return [ raw for index, raw in enumerate( raw_boxes ) if index not in removed ], counts


# ---------------------------------------------------------------------------
# 2. Geometry sanity filter
# ---------------------------------------------------------------------------

def apply_geometry_filter( raw_boxes, vid_w, vid_h, backend_name=None ):
    """
    Drop detections whose box shape is implausible for their label.

    A detector will occasionally latch onto the wrong thing at the wrong
    scale: an 'exposed_vulva' box covering 40% of a 1080p frame is the
    model having found a torso, not a vulva, and a 6-pixel box is noise.
    Neither is distinguishable by confidence alone - these misfires are
    often confident - but both are obvious by geometry.

    Four per-label limits, all optional and all disabled by default so
    this filter can never remove a valid detection until it is
    deliberately configured:

        min_area_fraction   box area / frame area, lower bound
        max_area_fraction   box area / frame area, upper bound
        min_aspect_ratio    box width / box height, lower bound
        max_aspect_ratio    box width / box height, upper bound

    There is no defensible universal default for any of them: the right
    numbers depend on the footage's framing and the label. Run
    tools/bench/betabench.py geometry to get the observed distribution
    per label from your own cached detections; it prints p1/p50/p99 and
    a suggested pair of bounds.

    Args:
        raw_boxes: Raw detections.
        vid_w, vid_h: Frame size in pixels.
        backend_name: Backend whose per-label settings apply.

    Returns:
        A (surviving_boxes, filter_counts) pair. filter_counts maps
        '<label>:<reason>' to a count.
    """
    frame_area = float( max( 1, vid_w * vid_h ) )
    limits_by_label = {}
    counts = {}

    def limits_for( label ):
        if label not in limits_by_label:
            overrides = bu_detector.get_item_overrides( label, backend_name )
            limits_by_label[label] = (
                overrides.get( 'min_area_fraction',
                               getattr( betaconfig, 'default_min_area_fraction', None ) ),
                overrides.get( 'max_area_fraction',
                               getattr( betaconfig, 'default_max_area_fraction', None ) ),
                overrides.get( 'min_aspect_ratio',
                               getattr( betaconfig, 'default_min_aspect_ratio', None ) ),
                overrides.get( 'max_aspect_ratio',
                               getattr( betaconfig, 'default_max_aspect_ratio', None ) ),
                overrides.get( 'geometry_action',
                               getattr( betaconfig, 'default_geometry_action',
                                        DEFAULT_GEOMETRY_ACTION ) ),
            )
        return limits_by_label[label]

    surviving = []
    for raw in raw_boxes:
        label = raw['class_id']
        ( min_area, max_area, min_aspect,
          max_aspect, action ) = limits_for( label )
        if min_area is None and max_area is None and min_aspect is None and max_aspect is None:
            surviving.append( raw )
            continue

        area_fraction = ( raw['w'] * raw['h'] ) / frame_area
        aspect_ratio = raw['w'] / raw['h'] if raw['h'] else 0.0

        reason = None
        if min_area is not None and area_fraction < min_area:
            reason = 'too_small'
        elif max_area is not None and area_fraction > max_area:
            reason = 'too_large'
        elif min_aspect is not None and aspect_ratio < min_aspect:
            reason = 'too_tall'
        elif max_aspect is not None and aspect_ratio > max_aspect:
            reason = 'too_wide'

        if reason is None:
            surviving.append( raw )
            continue

        taken = _geometry_action_for( action, reason )
        if taken == 'flag':
            surviving.append( raw )
        elif taken == 'clamp':
            surviving.append( _clamped_to_limits(
                raw, reason, frame_area, vid_w, vid_h,
                max_area, min_aspect, max_aspect ) )
        # 'drop' appends nothing.
        counts['%s:%s:%s'%( label, reason, taken )] = (
            counts.get( '%s:%s:%s'%( label, reason, taken ), 0 ) + 1 )

    return surviving, counts


# Per-violation default. A MIN violation is noise - a 6px box carries no
# coverage worth keeping, so dropping it removes nothing. A MAX violation
# is the dangerous one: area alone cannot tell a correct close-up from a
# torso misfire, and dropping means NO CENSOR AT ALL.
#
# Measured with exposed_breast's own suggested cap (max_area_fraction
# 0.123238, betabench p99 x1.25) on a 1080p frame:
#
#     normal breast   1.1% of frame   kept
#     p99 breast     10.3%            kept
#     close-up       32.4%            DROPPED  <- legitimate, uncensored
#     torso misfire  40.0%            DROPPED  <- the intended catch
#
# And it is not an edge case: on a real 10-file run the largest detection
# of every censored label sat 2.0-4.1x above its suggested cap.
#
# So max bounds clamp by default. A clamped torso misfire becomes a
# roughly breast-sized censor near the right place; a clamped close-up is
# censored slightly small. Both beat nothing.
DEFAULT_GEOMETRY_ACTION = { 'min': 'drop', 'max': 'clamp' }

GEOMETRY_ACTIONS = ( 'drop', 'clamp', 'flag' )

_GEOMETRY_REASON_BOUND = { 'too_small': 'min', 'too_large': 'max',
                           'too_tall':  'min', 'too_wide':  'max' }


def _geometry_action_for( action, reason ):
    """
    Which action a violation takes.

    `action` is either a single action name applied to every violation, or
    a {'min': ..., 'max': ...} mapping. The mapping form exists because
    the right answer genuinely differs by direction - see
    DEFAULT_GEOMETRY_ACTION.
    """
    if isinstance( action, dict ):
        bound = _GEOMETRY_REASON_BOUND.get( reason, 'max' )
        chosen = action.get( bound, DEFAULT_GEOMETRY_ACTION.get( bound, 'drop' ) )
    else:
        chosen = action
    return chosen if chosen in GEOMETRY_ACTIONS else 'drop'


def _clamped_to_limits( raw, reason, frame_area, vid_w, vid_h,
                        max_area, min_aspect, max_aspect ):
    """
    Shrink a box to its limit, keeping its centre and staying in frame.

    Only MAX violations are clampable. Growing a too-small box would
    invent coverage the detector never claimed, and stretching a box to
    satisfy an aspect bound would move its edges away from whatever the
    model actually found, so 'too_small' and 'too_tall' fall back to
    dropping.

    Returns a NEW dict; raw is never mutated (the caller's box list is
    shared with the cache reader).
    """
    width = float( raw['w'] )
    height = float( raw['h'] )

    if reason == 'too_large' and max_area is not None:
        # Scale both sides by the same factor, so the aspect ratio the
        # detector reported survives the clamp.
        target = max_area * frame_area
        current = width * height
        if current > 0 and target > 0:
            scale = math.sqrt( target / current )
            width *= scale
            height *= scale
    elif reason == 'too_wide' and max_aspect is not None:
        width = height * max_aspect
    else:
        return _unclampable( raw )

    centre_x = raw['x'] + raw['w'] / 2.0
    centre_y = raw['y'] + raw['h'] / 2.0
    width = max( 1.0, min( width, float( vid_w ) ) )
    height = max( 1.0, min( height, float( vid_h ) ) )
    x = max( 0.0, min( centre_x - width / 2.0, vid_w - width ) )
    y = max( 0.0, min( centre_y - height / 2.0, vid_h - height ) )

    clamped = dict( raw )
    clamped['x'] = x
    clamped['y'] = y
    clamped['w'] = width
    clamped['h'] = height
    clamped['_geometry_clamped'] = reason
    return clamped


def _unclampable( raw ):
    """
    A violation that shrinking cannot fix keeps its original geometry.

    Reached for 'too_small' and 'too_tall' under a clamp action. Returning
    the box unchanged (rather than dropping it) is deliberate: the caller
    asked for clamp, which means "never delete coverage", and the honest
    answer for a bound we cannot satisfy by shrinking is to leave the box
    alone and let the count record it.
    """
    return dict( raw )


# ---------------------------------------------------------------------------
# 3. Class suppression
# ---------------------------------------------------------------------------

def relevant_labels_for_suppression( suppression_rules, censorable_labels ):
    """
    Every label that can still matter once detection is done.

    A label matters if it is censored, or if it appears as a
    'suppressed_by' in some rule. Anything else is a detection nothing
    downstream will ever read, and dropping it early makes suppression
    cheaper with no change in behaviour.

    Args:
        suppression_rules: The resolved class_suppression dict.
        censorable_labels: The labels in items_to_censor.

    Returns:
        A set of label names.
    """
    relevant = set( censorable_labels )
    for label, rules in ( suppression_rules or {} ).items():
        relevant.add( label )
        rule_list = [ rules ] if isinstance( rules, dict ) else rules
        for rule in rule_list:
            relevant.add( rule['suppressed_by'] )
    return relevant


def apply_class_suppression( raw_boxes, suppression_rules=None, parts_to_blur=None ):
    """
    Drop detections that look like a misclassification of a nearby,
    more confident detection of a different label.

    A rule reads: label L is suppressed by label S when an S detection
    at the same instant overlaps it by at least min_iou AND outscores it
    by at least margin. Both thresholds are required by config
    validation; the .get() fallbacks below are a safety net, not the
    intended path.

    Only boxes from the exact same instant are ever compared. Grouping
    by timestamp up front turns what used to be an O(n^2) scan over every
    pair of detections in the whole video into an O(n) grouping pass plus
    a small O(k^2) comparison inside each instant, where k is a single
    digit. On a multi-hour video the old form could grind for tens of
    minutes with no progress output.

    Args:
        raw_boxes: Raw detections across every label and timestamp.
        suppression_rules: Resolved rules. Read from the active backend
            when omitted.
        parts_to_blur: Resolved per-label settings, used only to report
            how many suppressions affected a detection that would
            actually have been rendered. Optional.

    Returns:
        A (surviving_boxes, suppression_counts) pair. raw_boxes is not
        mutated. suppression_counts maps 'L<-S' to a
        {'total', 'renderable'} dict: 'total' is every time the rule
        fired, 'renderable' is the subset where the suppressed detection
        had cleared its own label's min_prob and so would really have
        been drawn. Before 2.1 only the total was reported, which
        overstated how much work a rule was doing, because most of what
        it suppressed was below min_prob and headed for the bin anyway.
    """
    if suppression_rules is None:
        suppression_rules = bu_detector.get_class_suppression()
    if not suppression_rules:
        return raw_boxes, {}

    def rules_for( label ):
        # A label's entry may be a single rule dict (old style) or a
        # list of them, each with its own suppressed_by/margin/min_iou,
        # so one label can be suppressed by several others, each judged
        # with its own bar.
        rules = suppression_rules.get( label )
        if not rules:
            return []
        return [ rules ] if isinstance( rules, dict ) else rules

    def is_renderable( raw ):
        if not parts_to_blur:
            return False
        settings = parts_to_blur.get( raw['class_id'] )
        return bool( settings ) and raw['score'] > settings['min_prob']

    suppressed = set()
    suppression_counts = {}

    for indices in _group_by_instant( raw_boxes ).values():
        if len( indices ) < 2:
            continue
        for index in indices:
            raw = raw_boxes[index]
            rules = rules_for( raw['class_id'] )
            if not rules:
                continue
            for other_index in indices:
                if other_index == index:
                    continue
                other = raw_boxes[other_index]
                matching_rules = [ r for r in rules if r['suppressed_by'] == other['class_id'] ]
                if not matching_rules:
                    continue
                box_iou = intersection_over_union( raw, other )
                for rule in matching_rules:
                    if box_iou < rule.get( 'min_iou', 0.3 ):
                        continue
                    if other['score'] - raw['score'] >= rule.get( 'margin', 0.0 ):
                        suppressed.add( index )
                        rule_key = '%s<-%s'%( raw['class_id'], other['class_id'] )
                        entry = suppression_counts.setdefault(
                            rule_key, { 'total': 0, 'renderable': 0 } )
                        entry['total'] += 1
                        if is_renderable( raw ):
                            entry['renderable'] += 1
                        break
                if index in suppressed:
                    break

    if not suppressed:
        return raw_boxes, suppression_counts
    surviving = [ raw for index, raw in enumerate( raw_boxes ) if index not in suppressed ]
    return surviving, suppression_counts


# ---------------------------------------------------------------------------
# 4b. Class promotion
# ---------------------------------------------------------------------------

PROMOTION_RULE_KEYS = { 'from', 'min_prob', 'requires', 'requires_mode', 'duplicate_iou' }
PROMOTION_EVIDENCE_KEYS = { 'label', 'min_prob', 'min_iou', 'min_source_overlap', 'overlap' }


def _intersection_over_source( source, other ):
    """Fraction of source's own area that other covers, in [0, 1]."""
    s_x2, s_y2 = source['x'] + source['w'], source['y'] + source['h']
    o_x2, o_y2 = other['x'] + other['w'], other['y'] + other['h']
    inter_w = min( s_x2, o_x2 ) - max( source['x'], other['x'] )
    inter_h = min( s_y2, o_y2 ) - max( source['y'], other['y'] )
    area = source['w'] * source['h']
    if inter_w <= 0 or inter_h <= 0 or area <= 0:
        return 0.0
    return ( inter_w * inter_h ) / area


def _evidence_matches( source, other, requirement ):
    """
    Whether one other detection satisfies one evidence requirement.

    Two overlap measures, because they answer different questions. IoU
    penalises a size mismatch: a long penis box fully covering a small
    vulva box scores a low IoU even though the vulva is entirely
    covered. min_source_overlap asks the question that case actually
    needs - how much of the SOURCE box the evidence lands on. When a
    requirement names neither, any overlap at all counts.
    """
    if other['class_id'] != requirement['label']:
        return False
    if other['score'] < requirement.get( 'min_prob', 0.0 ):
        return False
    has_iou = 'min_iou' in requirement
    # 'anywhere': co-presence in the frame IS the evidence, with no
    # spatial test at all.
    #
    # Measured on 9 real caches (488k detections, 9,302 covered_vulva
    # above 0.3): only 18.9% had any of exposed_penis / exposed_vulva /
    # exposed_anus live at the same instant, and of THOSE, 91% had zero
    # pixel overlap with it. So an overlap-based rule promoted 1
    # detection; requiring only co-presence promotes 1,452.
    #
    # That is the right question for this case. "Is this clothed crotch
    # in a penetration scene" is a fact about the SCENE, not about which
    # pixels touch which - the penis may be elsewhere in frame, or the
    # covered_vulva box may sit beside rather than under it. Overlap
    # answers "are these the same object", which is a different question
    # and the one min_iou / min_source_overlap exist for.
    if requirement.get( 'overlap' ) == 'anywhere':
        return True
    has_overlap = 'min_source_overlap' in requirement
    if not has_iou and not has_overlap:
        return _intersection_over_source( source, other ) > 0.0
    if has_iou and intersection_over_union( source, other ) < requirement['min_iou']:
        return False
    if has_overlap and _intersection_over_source( source, other ) < requirement['min_source_overlap']:
        return False
    return True


def _evidence_satisfied( source, others, rule ):
    """Whether a rule's 'requires' list is met by the same-instant boxes."""
    requirements = rule.get( 'requires' ) or []
    if not requirements:
        return True
    results = [ any( _evidence_matches( source, other, requirement ) for other in others )
                for requirement in requirements ]
    if rule.get( 'requires_mode', 'any' ) == 'all':
        return all( results )
    return any( results )


def apply_class_promotion( raw_boxes, promotion_rules=None ):
    """
    Relabel detections that corroborating evidence says are something
    more specific. Suppression's inverse, and additive by construction.

    A rule reads: a detection of label F becomes label T when it scores
    at least min_prob AND, at the same instant, the 'requires' evidence
    is present (any one of it by default, or all of it). The box keeps
    its geometry and score; only the label changes, so it is then
    filtered, suppressed, tracked and styled as T.

    Why relabel rather than add a copy: the source label may itself be
    censored (covered_vulva is), and a copy would draw two censors over
    one object with two independent styles.

    A detection is NOT promoted when a real T detection already overlaps
    it by duplicate_iou (default 0.5). The model has already found that
    object as T, and promoting too would give tracking two boxes for
    one thing - two tracks, two styles, and the flicker that brings.

    Evidence is judged against ORIGINAL labels only. A box promoted in
    this pass cannot serve as evidence for another promotion in the same
    pass, so rule order cannot change the outcome.

    Args:
        raw_boxes: Raw detections across every label and timestamp.
        promotion_rules: Resolved rules. Read from the active backend
            when omitted.

    Returns:
        A (boxes, promotion_counts) pair. raw_boxes is not mutated;
        promoted entries are fresh dicts carrying 'promoted_from'.
        promotion_counts maps 'T<-F' to {'promoted', 'already_covered',
        'no_evidence'}, counted only for F detections that cleared the
        rule's min_prob - the three outcomes a person tuning the rule
        needs to see side by side.
    """
    if promotion_rules is None:
        promotion_rules = bu_detector.get_class_promotion()
    if not promotion_rules:
        return raw_boxes, {}

    rules_by_source = {}
    for target, rules in promotion_rules.items():
        for rule in ( [ rules ] if isinstance( rules, dict ) else rules ):
            rules_by_source.setdefault( rule['from'], [] ).append( ( target, rule ) )

    promoted = {}
    counts = {}

    for indices in _group_by_instant( raw_boxes ).values():
        for index in indices:
            raw = raw_boxes[index]
            candidates = rules_by_source.get( raw['class_id'] )
            if not candidates:
                continue
            others = [ raw_boxes[other] for other in indices if other != index ]
            for target, rule in candidates:
                if raw['score'] < rule.get( 'min_prob', 0.0 ):
                    continue
                entry = counts.setdefault( '%s<-%s'%( target, raw['class_id'] ),
                                           { 'promoted': 0, 'already_covered': 0,
                                             'no_evidence': 0 } )
                duplicate_iou = rule.get( 'duplicate_iou', 0.5 )
                if any( other['class_id'] == target
                        and intersection_over_union( raw, other ) >= duplicate_iou
                        for other in others ):
                    entry['already_covered'] += 1
                    break
                if not _evidence_satisfied( raw, others, rule ):
                    entry['no_evidence'] += 1
                    continue
                promoted_box = dict( raw )
                promoted_box['class_id'] = target
                promoted_box['promoted_from'] = raw['class_id']
                promoted[index] = promoted_box
                entry['promoted'] += 1
                break

    if not promoted:
        return raw_boxes, counts
    return [ promoted.get( index, raw ) for index, raw in enumerate( raw_boxes ) ], counts


# ---------------------------------------------------------------------------
# 5. Tracking, smoothing, interpolation, confirmation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tracking observers
# ---------------------------------------------------------------------------
#
# The analysis tools under tools/analysis/ need to see INSIDE smooth_boxes:
# which detection matched which track and how far away it was, when a new
# track was started and what else was nearby, and the full candidate pool
# before the greedy assignment picks winners. None of that is in the
# return value, and none of it should be - it is diagnostic detail that
# would cost every real run to carry.
#
# Before 2.1 those tools got at it by reading betatv.py's SOURCE TEXT,
# regex-matching the smooth_boxes body out of it, string-replacing
# instrumentation calls into specific lines, and exec'ing the result.
# That worked until any of those lines moved, at which point the tools
# failed with "its structure has changed since this tool was written
# against it" - which is exactly what happened to three of them in the
# 2.1 refactor.
#
# This is the supported extension point instead. An observer is a plain
# object with any subset of the three methods below; missing ones are
# simply not called. When no observer is installed the cost is one
# `is not None` check per event.


class TrackingObserver:
    """
    Reference shape for a smooth_boxes observer.

    Subclass it or duck-type it - every method is optional. Observers
    are for diagnostics only: nothing they do or return can change a
    tracking decision, which is what makes it safe to run the real
    pipeline under instrumentation and trust the result.
    """

    def on_match( self, label, track_index, distance, box, tracks, settings ):
        """A detection was assigned to an existing track."""

    def on_new_track( self, label, tracks, track_index, box, settings ):
        """A detection started a brand new track, having matched nothing."""

    def on_frame_candidates( self, label, candidates, tracks, settings ):
        """
        Every plausible (box, track) pairing for one frame, before the
        greedy nearest-first assignment picks winners.

        Args:
            candidates: A list of (distance, box, track_index) tuples.
        """

    def on_independent_style_resolve( self, label, box, tracks, settings ):
        """A new track rolled its own style rather than sharing a pair's."""


_tracking_observer = None


def set_tracking_observer( observer ):
    """
    Install (or clear, with None) the tracking observer.

    Returns:
        The previously installed observer, so a caller can restore it.
    """
    global _tracking_observer
    previous = _tracking_observer
    _tracking_observer = observer
    return previous


@contextlib.contextmanager
def tracking_observer( observer ):
    """Install an observer for the duration of a block, then restore."""
    previous = set_tracking_observer( observer )
    try:
        yield observer
    finally:
        set_tracking_observer( previous )


def _notify( event, *args ):
    """Call one observer method if an observer is installed and has it."""
    observer = _tracking_observer
    if observer is None:
        return
    handler = getattr( observer, event, None )
    if handler is not None:
        handler( *args )


class _LabelSettings:
    """Every tracking knob for one label, resolved once per label."""

    __slots__ = ( 'label', 'alpha', 'interpolation_enabled', 'interpolation_max_gap',
                  'max_gap', 'match_distance_multiplier', 'time_safety', 'item_x_safety',
                  'item_y_safety', 'paired_style', 'paired_style_max_distance',
                  'paired_style_tiebreak_margin', 'paired_style_max_age',
                  'min_track_hits', 'unresolved_censor_style', 'size_alpha',
                  'style_min_dwell_seconds', 'profile_name' )

    def __init__( self, label, backend_name, frame_step, profile_name=None ):
        self.label = label
        overrides = dict( bu_detector.get_item_overrides( label, backend_name ) )
        # A profile layers ON TOP of the backend's own item_overrides, so
        # it only states the keys whose right value depends on how the
        # footage is cut. See betautils_detector.get_profiles.
        overrides.update(
            bu_detector.get_profile_item_overrides( profile_name, label, backend_name ) )
        self.profile_name = profile_name

        # The UNRESOLVED censor_style (the weighted list), kept so a shot
        # cut can re-roll a surviving track's look. censor_style always
        # comes from the shared item_overrides block, never a backend's -
        # see ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS.
        # How long a style is held before a shot cut may re-roll it.
        # Without it, footage cut faster than the dwell changes style
        # every shot - which on a dressing scene, where a subject is
        # briefly covered then re-exposed, reads as the censor flickering
        # between looks rather than as variety.
        self.style_min_dwell_seconds = overrides.get(
            'style_min_dwell_seconds',
            getattr( betaconfig, 'default_style_min_dwell_seconds', 0.0 ) )

        self.unresolved_censor_style = betaconfig.item_overrides.get(
            label, {} ).get( 'censor_style', betaconfig.default_censor_style )

        self.alpha = overrides.get(
            'position_smoothing', betaconfig.default_position_smoothing )
        # Defaults to a THIRD of the position alpha rather than to alpha
        # itself: size jitter is what flickers the censor's edge, and
        # nothing about following a moving subject requires the box to
        # resize as fast as it translates. An explicit per-label
        # size_smoothing overrides this.
        self.size_alpha = overrides.get(
            'size_smoothing',
            getattr( betaconfig, 'default_size_smoothing', self.alpha/3.0 ) )
        self.interpolation_enabled = overrides.get(
            'interpolation_enabled', betaconfig.default_interpolation_enabled )
        self.interpolation_max_gap = overrides.get(
            'interpolation_max_gap', betaconfig.default_interpolation_max_gap )

        # track_max_gap and interpolation_max_gap are NOT the same thing,
        # and conflating them was a real bug found with
        # analyze_track_breaks.py. track_max_gap decides whether a
        # detection continues an EXISTING track at all: miss it and the
        # track hard-resets to a brand new unsmoothed track at the raw
        # position, regardless of interpolation_max_gap.
        # interpolation_max_gap only ever applies to gaps that already
        # passed the track_max_gap test, filling between two real hits on
        # one continuing track. A track_max_gap tighter than a label's
        # interpolation_max_gap makes that label's interpolation tuning
        # partly moot - real footage showed 88-97% of resets were
        # gap-blocked rather than distance-blocked - so the default is
        # whichever is larger.
        default_track_max_gap = 2.0 * frame_step
        self.max_gap = overrides.get(
            'track_max_gap', max( default_track_max_gap, self.interpolation_max_gap ) )

        self.match_distance_multiplier = overrides.get( 'match_distance_multiplier', 1.0 )
        self.time_safety = overrides.get( 'time_safety', betaconfig.default_time_safety )
        self.item_x_safety = overrides.get( 'width_area_safety', betaconfig.default_area_safety )
        self.item_y_safety = overrides.get( 'height_area_safety', betaconfig.default_area_safety )

        self.paired_style = overrides.get( 'paired_style', False )
        self.paired_style_max_distance = overrides.get(
            'paired_style_max_distance', betaconfig.default_paired_style_max_distance )
        self.paired_style_tiebreak_margin = overrides.get(
            'paired_style_tiebreak_margin', betaconfig.default_paired_style_tiebreak_margin )
        # How stale a track may be and still donate its style. Separate
        # from track_max_gap on purpose - see
        # nearest_unambiguous_live_track for why the two questions are
        # not the same one.
        self.paired_style_max_age = overrides.get(
            'paired_style_max_age',
            getattr( betaconfig, 'default_paired_style_max_age', 1.0 ) )

        self.min_track_hits = max( 1, int( overrides.get(
            'min_track_hits', getattr( betaconfig, 'default_min_track_hits', 1 ) ) ) )


def _apply_style_area_safety( box, resolved_style, item_x_safety, item_y_safety ):
    """
    Recompute a box's padded geometry using its resolved style's own
    area safety, falling back to the item's.

    process_raw_box could only ever use the item default, because no
    style had been picked yet. This gives per-style width_area_safety /
    height_area_safety the same precedence strength and feather already
    have. When the style sets neither key this re-derives identical
    geometry, which is harmless and not worth special-casing.
    """
    style_x_safety = resolved_style.get( 'width_area_safety', item_x_safety )
    style_y_safety = resolved_style.get( 'height_area_safety', item_y_safety )
    box['x'], box['y'], box['w'], box['h'] = bu_censor.compute_safe_geometry(
        box['_raw_x'], box['_raw_y'], box['_raw_w'], box['_raw_h'],
        box['_vid_w'], box['_vid_h'], style_x_safety, style_y_safety )


def smooth_boxes( boxes, shot_cut_times=None, backend_name=None, profile_name=None,
                  sample_fps=None ):
    """
    Match detections into per-instance tracks, then smooth, style,
    interpolate and confirm them.

    WHY TRACKS, NOT JUST TIME ORDER
        The original implementation sorted every same-label box in the
        whole video by time and blended each one toward whichever box
        came immediately before it, with no notion of "this is a
        different instance from that one". On a frame with two boxes -
        both breasts on one person - one would routinely get blended
        toward the other and drift to a point between the two real
        positions instead of tracking either. Detections are now matched
        within each timestamp first, greedily nearest-pair first, so two
        simultaneous same-label detections always land on two different
        tracks.

    WHAT HAPPENS ONCE PER TRACK RATHER THAN ONCE PER FRAME
      - censor_style is resolved to ONE concrete style the first time a
        track is seen, and reused for its whole life, so a randomised
        multi-style item does not flicker between (say) pixel and bar
        frame to frame. If the resolved style carries its own 'shape',
        that overrides the label's default shape for the track too, for
        the same reason. censor_sticker_seed is carried forward
        identically: sticker_image runs once per RENDERED FRAME, so
        without this a sticker style flips through the whole folder
        instead of one sticker staying put.
      - paired_style: a newly-resolving box with an unambiguous nearest
        other live track of the same label shares that track's style
        instead of rolling its own.

    WHAT HAPPENS ACROSS GAPS
      - interpolation: when a track's next real hit arrives after a gap
        the model missed, synthetic boxes are generated by linearly
        interpolating position and size across it, instead of holding
        the last position static.

    SHOT CUTS
        shot_cut_times blocks BOTH kinds of cross-time matching -
        continuing a track, and paired_style's "nearby live track" test -
        across any recorded cut, regardless of every distance and gap
        setting. Those settings all assume continuous single-scene
        footage; a cut means whatever is on screen just became a
        different scene however close in time or position the next
        detection lands.

    HYSTERESIS AND CONFIRMATION
      - A box marked 'provisional' (score between min_prob_continue and
        min_prob - see process_raw_box) may only CONTINUE an existing
        track. If it matches nothing it is dropped rather than starting
        a track of its own.
      - A track must accumulate at least min_track_hits real detections
        before any of its boxes render. Every box of a track that never
        reaches that count is dropped, including its interpolated ones.
        This is what removes a single-frame false positive that would
        otherwise paint a censor blob for time_safety seconds.
        min_track_hits defaults to 1, which admits every track and is
        exactly the pre-2.1 behaviour.

    Args:
        boxes: Censorable boxes from process_raw_box, any order, all
            labels. Mutated in place (position, style, shape) - the
            returned list is a filtered view of the same dicts plus new
            interpolated ones.
        shot_cut_times: Sorted cut timestamps in seconds, or None.
        backend_name: Backend whose per-label settings apply.

    Returns:
        A (rendered_boxes, stats) pair. stats is a dict with
        'interpolated', 'dropped_provisional', 'dropped_unconfirmed' and
        'tracks'.
    """
    shot_cut_times = shot_cut_times or []
    logger = bu_log.get_logger()

    def cut_between( earlier_time, later_time ):
        """
        Whether a recorded cut falls in (earlier_time, later_time].

        A cut AT later_time counts: later_time is the first moment of
        the new shot. A cut AT earlier_time does not, because it already
        forced a reset when earlier_time's own box was matched. O(log n)
        via bisect, since this runs for every candidate pairing.
        """
        if not shot_cut_times:
            return False
        index = bisect.bisect_right( shot_cut_times, earlier_time )
        return index < len( shot_cut_times ) and shot_cut_times[index] <= later_time

    by_label = {}
    for box in boxes:
        by_label.setdefault( box['label'], [] ).append( box )

    # The rate the detections were actually SAMPLED at, which a profile
    # can raise above the global one. Every frame-count rule below
    # (interpolation steps, the default track_max_gap of two frames)
    # is expressed in this step, so reading the global rate here while
    # detection ran at a profile's rate would misjudge every gap.
    frame_step = 1.0 / ( sample_fps or betaconfig.video_censor_fps )

    output_boxes = []
    stats = { 'interpolated': 0, 'dropped_provisional': 0,
              'dropped_unconfirmed': 0, 'tracks': 0 }

    for label, boxes_for_label in by_label.items():
        settings = _LabelSettings( label, backend_name, frame_step, profile_name )

        # Every box from one detection pass shares an identical 'start'
        # (derived only from t and the label's fixed time_safety), so
        # grouping on 'start' reliably reconstructs "what was detected
        # together in this frame".
        frames = {}
        for box in boxes_for_label:
            frames.setdefault( box['start'], [] ).append( box )

        # The style this label last resolved, and when it first showed.
        # A shot cut ENDS a track, so without this the dwell rule would
        # never bite on cut-dense footage: every shot would open a brand
        # new track and roll a fresh style regardless of how recently the
        # last one appeared. Keyed per label because two labels' styles
        # are independent.
        last_style = { 'style': None, 'shape': None, 'seed': None, 'since': None }

        tracks = []              # live tracks only; pruned every frame
        track_hits = {}          # track id -> real (non-interpolated) hit count
        boxes_by_track = {}      # track id -> [box, ...] including interpolated
        next_track_id = 0

        for start in sorted( frames.keys() ):
            frame_boxes = frames[start]
            # This frame's true timestamp. Every box from one detection
            # pass shares it. Cut checks must use this, not the
            # time_safety-padded 'start'/'end' - see cut_between's callers.
            frame_t = frame_boxes[0].get( 't', start )

            # Prune tracks that can never match again. 'start' only
            # increases, and every liveness test below is
            # `start - track['end'] > max_gap`, so this is exactly
            # equivalent to leaving them in place - but it keeps the
            # candidate scan proportional to LIVE tracks rather than to
            # every track ever created. On a long video with frequent
            # resets the unpruned list made this loop quadratic.
            if tracks:
                tracks = [ track for track in tracks
                           if start - track['end'] <= settings.max_gap ]

            # Every (box, track) pairing close enough in time and space
            # to plausibly be the same instance, then greedily assign
            # the closest pairs first, so within one frame each box
            # claims the nearest track and no single track is pulled
            # toward two simultaneous detections.
            candidates = []
            for box in frame_boxes:
                centre_x = box['x'] + box['w']/2
                centre_y = box['y'] + box['h']/2
                for track_index, track in enumerate( tracks ):
                    if box['start'] - track['end'] > settings.max_gap:
                        continue
                    # Compare true frame TIMESTAMPS, never the
                    # time_safety-padded start/end. 'end' is t + safety/2
                    # and 'start' is t - safety/2, so whenever
                    # time_safety >= the frame step the interval
                    # (track.end, box.start] is INVERTED and empty, and
                    # cut_between can never fire. At fps=9 the step is
                    # 0.111s while exposed_breast's time_safety is 0.18,
                    # so every cut was invisible here: a 30s slice with 50
                    # detected cuts and 16 different women produced 3
                    # tracks and ONE style for the whole slice.
                    if cut_between( track['t'], box.get( 't', box['start'] ) ):
                        continue
                    track_cx = track['x'] + track['w']/2
                    track_cy = track['y'] + track['h']/2
                    distance = ( ( centre_x-track_cx )**2 + ( centre_y-track_cy )**2 ) ** 0.5
                    reach = settings.match_distance_multiplier * max(
                        track['w'], track['h'], box['w'], box['h'] )
                    if distance >= reach:
                        continue
                    candidates.append( ( distance, id( box ), box, track_index ) )
            if _tracking_observer is not None:
                _notify( 'on_frame_candidates', label,
                         [ ( distance, box, track_index )
                           for distance, _box_id, box, track_index in candidates ],
                         tracks, settings )
            candidates.sort( key=lambda candidate: ( candidate[0], candidate[1] ) )

            matched_box_ids = set()
            matched_track_indices = set()
            box_track_index = {}

            for distance, box_id, box, track_index in candidates:
                if box_id in matched_box_ids or track_index in matched_track_indices:
                    continue
                matched_box_ids.add( box_id )
                matched_track_indices.add( track_index )
                box_track_index[box_id] = track_index
                track = tracks[track_index]
                _notify( 'on_match', label, track_index, distance, box, tracks, settings )

                # Re-derive this box's padding from the track's resolved
                # style before interpolating or smoothing, so both use
                # the real, style-correct geometry.
                _apply_style_area_safety( box, track['censor_style'],
                                          settings.item_x_safety, settings.item_y_safety )

                # Interpolate whole skipped detection cycles BEFORE
                # smoothing this box, so the synthetic boxes run between
                # the track's actual last rendered position and this
                # box's actual new raw position.
                gap = box.get( 't', box['start'] ) - track['t']
                if ( settings.interpolation_enabled
                        and gap > 1.5*frame_step
                        and gap <= settings.interpolation_max_gap ):
                    steps = round( gap / frame_step )
                    for step in range( 1, steps ):
                        fraction = step / steps
                        synthetic_t = track['t'] + fraction*gap
                        synthetic = {
                            'start': max( synthetic_t - settings.time_safety/2, 0 ),
                            'end':   synthetic_t + settings.time_safety/2,
                            't': synthetic_t,
                            'x': int( track['x'] + fraction*( box['x']-track['x'] ) ),
                            'y': int( track['y'] + fraction*( box['y']-track['y'] ) ),
                            'w': int( track['w'] + fraction*( box['w']-track['w'] ) ),
                            'h': int( track['h'] + fraction*( box['h']-track['h'] ) ),
                            'censor_style': track['censor_style'],
                            'censor_shape': track['censor_shape'],
                            'censor_sticker_seed': track['censor_sticker_seed'],
                            'style_scale_w': track['style_scale_w'],
                            'style_scale_h': track['style_scale_h'],
                            'label': label,
                            'score': min( track['score'], box['score'] ),
                            'size': box.get( 'size', 0 ),
                            'interpolated': True,
                            '_track_id': track['id'],
                        }
                        boxes_by_track.setdefault( track['id'], [] ).append( synthetic )
                        stats['interpolated'] += 1

                # POSITION and SIZE are smoothed with different alphas,
                # because they are judged by the eye in different ways.
                #
                # Position wants to be responsive: a censor that lags the
                # subject exposes them, so alpha is tuned high (0.65) to
                # follow movement.
                #
                # Size does NOT want that. The rendered box's EDGE is
                # composited against the untouched frame every frame, so
                # any change in w/h moves that edge and repaints a ring
                # of real pixels that were censored last frame (or
                # censors a ring that was clear). Measured on a
                # structured scene, with NO blur or pixel maths involved
                # at all - a flat fill through apply_masked_region:
                #
                #     size sd  0 px  ->  edge brightness swing   0.0
                #     size sd  8 px  ->                         15.9
                #     size sd 23 px  ->                         62.0   (alpha 0.65)
                #
                # That is the flicker, and it is why it shows on EVERY
                # style - sticker and bar included - not just blur. Blur
                # merely makes it most visible, because a blurred edge
                # against a sharp background is a high-contrast boundary.
                #
                # Holding size steadier costs almost nothing: a censor
                # box that is a few pixels larger than it strictly needs
                # to be still covers the subject, while one that breathes
                # several pixels a frame reads as broken. size_alpha
                # therefore defaults well below alpha.
                alpha = settings.alpha
                size_alpha = settings.size_alpha
                box['x'] = int( alpha*box['x'] + ( 1-alpha )*track['x'] )
                box['y'] = int( alpha*box['y'] + ( 1-alpha )*track['y'] )
                box['w'] = int( size_alpha*box['w'] + ( 1-size_alpha )*track['w'] )
                box['h'] = int( size_alpha*box['h'] + ( 1-size_alpha )*track['h'] )

                # Re-clamp to the frame AFTER smoothing.
                #
                # Position and size now move at different rates, so they
                # can disagree about where the box ends even when every
                # raw detection was inside the frame. A box pinned to an
                # edge is the worst case: its raw y+h sits exactly at the
                # boundary, y follows the new detection quickly while h
                # still carries the old value, and y+h overshoots. With
                # alpha 0.65 against size_alpha 0.217 that overshoot
                # measured up to 17px.
                #
                # It surfaced as a render crash, not a visual glitch:
                # censor_image builds its feather mask from box w/h but
                # slices the region out of the frame, and numpy truncates
                # a slice at the array edge. The mask was then (79,94,1)
                # against a (77,94,3) region and the composite raised
                # ValueError, failing the whole chunk. Clamping here
                # keeps the two in agreement by construction.
                box['x'] = max( 0, min( box['x'], box['_vid_w'] - 1 ) )
                box['y'] = max( 0, min( box['y'], box['_vid_h'] - 1 ) )
                box['w'] = max( 1, min( box['w'], box['_vid_w'] - box['x'] ) )
                box['h'] = max( 1, min( box['h'], box['_vid_h'] - box['y'] ) )
                # A shot cut re-rolls the style WITHOUT ending the track.
                #
                # Style variety and censor continuity pull in opposite
                # directions, and separating them is the point here. A
                # style is resolved once per track and held for its
                # lifetime, which is right: re-rolling mid-shot is the
                # flicker this whole module exists to prevent. But a
                # long-lived track then wears one style for as long as it
                # survives, and on compilation footage that is the whole
                # file: measured on real caches, exposed_breast continued
                # 99.7% of its boxes (46315/46470) and opened only 155 new
                # tracks across 77 minutes - about ONE new track, and so
                # ONE style roll, per 30 seconds. That is why a 30s slice
                # comes out a single style even though the configured mix
                # is hex 33% / mosaic 33% / sticker 15% / blur 13% / bar 6%
                # and the resolver samples those weights correctly.
                #
                # The obvious lever, a shorter track_max_gap, is the wrong
                # one: auto_tune measured 27.0s -> 13.5s as +92 track
                # resets, and a reset mid-scene is a censor that blinks
                # off and back on. Never trade censor continuity for
                # cosmetic variety.
                #
                # A shot cut is the one moment where a new style costs
                # nothing, because the picture has already changed
                # completely - nobody reads it as the censor flickering.
                # The track itself continues, so coverage is untouched;
                # only the look is re-rolled.
                # Compare the two frames' real timestamps, NOT box['start']:
                # time_safety back-dates 'start' by half its value, so
                # box['start'] can land BEFORE track['t'] and the cut then
                # falls outside the interval and is never seen.
                # A cut re-rolls the style only once the current one has
                # been showing for style_min_dwell_seconds. Without that,
                # footage cut faster than the dwell re-rolls every shot.
                box_t = box.get( 't', box['start'] )
                style_age = box_t - track.get( 'style_since', track['t'] )
                crossed_cut = ( style_age >= settings.style_min_dwell_seconds
                                and cut_between( track['t'], box_t ) )
                if crossed_cut:
                    box['censor_style'] = bu_censor.resolve_censor_style(
                        settings.unresolved_censor_style )
                    box['censor_shape'] = bu_censor.resolve_censor_shape(
                        box['censor_style'], box.get( 'censor_shape', 'box' ) )
                    box['censor_sticker_seed'] = random.random()
                    box['_style_since'] = box_t
                    last_style.update( style=box['censor_style'],
                                       shape=box['censor_shape'],
                                       seed=box['censor_sticker_seed'],
                                       since=box_t )
                    _notify( 'on_independent_style_resolve', label, box, tracks, settings )
                else:
                    box['_style_since'] = track.get( 'style_since', track['t'] )
                    box['censor_style'] = track['censor_style']
                    box['censor_shape'] = track['censor_shape']
                    box['censor_sticker_seed'] = track['censor_sticker_seed']
                box['style_scale_w'] = track['style_scale_w']
                box['style_scale_h'] = track['style_scale_h']
                box['_track_id'] = track['id']

                boxes_by_track.setdefault( track['id'], [] ).append( box )
                if not box.get( 'provisional' ):
                    track_hits[ track['id'] ] = track_hits.get( track['id'], 0 ) + 1

                tracks[track_index] = {
                    'id': track['id'],
                    'x': box['x'], 'y': box['y'], 'w': box['w'], 'h': box['h'],
                    'end': box['end'], 't': box.get( 't', box['start'] ), 'score': box['score'],
                    # Taken from the BOX, not the track: a shot cut may
                    # have just re-rolled these, and the new style has to
                    # persist for the rest of the track or the next frame
                    # would revert to the old one - which would be a real
                    # one-frame flicker.
                    'censor_style': box['censor_style'],
                    'censor_shape': box['censor_shape'],
                    'censor_sticker_seed': box['censor_sticker_seed'],
                    'style_since': box.get( '_style_since', track.get( 'style_since', track['t'] ) ),
                    # Deliberately NOT updated from box: holding it is
                    # the point.
                    'style_scale_w': track['style_scale_w'],
                    'style_scale_h': track['style_scale_h'],
                }

            def nearest_unambiguous_live_track( x, y, w, h, reference_start,
                                                exclude_track_index=None,
                                                reference_t=None ):
                """
                The single nearest OTHER live track of this label, or None.

                "Live" means recently DETECTED, not merely still
                matchable, and not separated from this frame by a cut.

                WHY THIS IS NOT settings.max_gap
                --------------------------------
                Track continuation uses max_gap because a track has to
                survive an occlusion - exposed_breast allows 27s,
                exposed_vulva 43.2s. Style pairing is a different
                question: "is this the other breast of the person on
                screen right now?" A donor whose last real detection was
                twenty seconds ago cannot answer that, but under max_gap
                it was still an eligible donor.

                On quick-cut footage that is the whole bug. At 53 cuts
                per minute a shot lasts about 1.1s, so 27s of
                eligibility keeps roughly two dozen shots' worth of
                departed subjects available to donate a style. A new
                person appearing in the same screen region as someone
                who left inherits their style, which is exactly the
                "every ensuing person inherits the censor style on other
                people in other panels" report from split-screen
                compilations - and the same mechanism makes a style
                appear to "stick around" long after the subject it
                belonged to has gone.

                paired_style_max_age bounds it instead. Analogy: you
                will hand your umbrella to the person standing next to
                you, not to whoever was standing there a minute ago.

                Returns None when nothing is in range, and also when the
                nearest two candidates are too close in distance to
                confidently call one of them "the pair" - two different
                people at similar distance are genuinely ambiguous and
                are left to resolve independently.
                """
                centre_x, centre_y = x + w/2, y + h/2
                # Falls back to reference_start only when a caller has no
                # true timestamp; the cut check below wants a timestamp.
                if reference_t is None:
                    reference_t = reference_start
                nearby = []
                donor_max_age = min( settings.paired_style_max_age, settings.max_gap )
                for track_index, track in enumerate( tracks ):
                    if track_index == exclude_track_index:
                        continue
                    if reference_start - track['end'] > donor_max_age:
                        continue
                    # Timestamps, not padded bounds - see the note at
                    # the track-continuation check. A padded comparison
                    # here let a dead track donate its style across a cut.
                    if cut_between( track['t'], reference_t ):
                        continue
                    track_cx, track_cy = track['x'] + track['w']/2, track['y'] + track['h']/2
                    distance = ( ( centre_x-track_cx )**2 + ( centre_y-track_cy )**2 ) ** 0.5
                    scale = max( track['w'], track['h'], w, h )
                    if scale and distance < settings.paired_style_max_distance * scale:
                        nearby.append( ( distance, track_index ) )
                if not nearby:
                    return None
                nearby.sort()
                if len( nearby ) == 1:
                    return nearby[0][1]
                nearest_distance, nearest_index = nearby[0]
                second_distance = nearby[1][0]
                if nearest_distance == 0 or ( ( second_distance - nearest_distance )
                                              / max( second_distance, 1e-9 )
                                              >= settings.paired_style_tiebreak_margin ):
                    return nearest_index
                return None  # too close to call

            # paired_style, e.g. exposed_breast: when a box of this label
            # is near exactly one other live track of the same label -
            # the common "both breasts on one person" case - share ONE
            # resolved style between them. Without this, a span-bar pick
            # on one side lands next to a pixel pick on the other and the
            # "span both" style never gets to span anything.
            #
            # Proximity, not frame-exactness, is the test. Requiring both
            # to be detected in the same frame used to be the gate, and
            # analyze_style_flicker.py against real cached footage showed
            # it essentially never fired: 99.6% of independent style
            # resolves had another live box of the label already present,
            # and 99% of those visually mismatched it, because real
            # per-frame detection noise means two breasts very often do
            # not both get a fresh detection in the identical frame even
            # while both are continuously visible. The observed distances
            # (median 0.09x box size, 99% under 1.0x) confirmed these
            # were overwhelmingly the SAME pair rather than two people.
            if settings.paired_style:
                # Retroactive sync: two continuing tracks that each
                # resolved independently (one was visible alone first)
                # but are now each other's only unambiguous nearby live
                # track. This IS a visible style change for whichever
                # side loses, and rarer than the new-box case below, but
                # a persistently mismatched pair reads as more obviously
                # wrong than a one-time change.
                # MUTUAL pairing only, and each track syncs at most once.
                #
                # Both conditions exist to stop a style ratchet. This
                # loop copies from the lower track index to the higher,
                # which is stable but monotonic: styles only ever spread
                # from older tracks to newer ones, never back. Without a
                # bound, A donates to B, B later pairs with C and donates
                # the SAME style onward, and one early roll propagates
                # through every track in the file. Simulated over 12
                # tracks, that collapses a 6-style mix to an average of
                # 1.5 distinct styles per video, with the winner covering
                # 95% of boxes - which is the "it's almost all hex"
                # symptom, even though the configured weights (hex 32%,
                # mosaic 32%, sticker 17%, bar 7%, blur 12%) and the
                # resolver itself are both correct.
                #
                # Requiring the pairing to be MUTUAL - each track is the
                # other's nearest unambiguous neighbour - is what a real
                # left/right pair on one torso looks like, and a relay
                # link in a chain does not. already_synced then stops a
                # track that has taken a style this frame from donating
                # it onward in the same pass.
                already_synced = set()
                nearest_by_track = {}
                for track_index in sorted( matched_track_indices ):
                    track = tracks[track_index]
                    nearest_by_track[track_index] = nearest_unambiguous_live_track(
                        track['x'], track['y'], track['w'], track['h'], start,
                        exclude_track_index=track_index, reference_t=frame_t )

                for track_index in sorted( matched_track_indices ):
                    other_index = nearest_by_track.get( track_index )
                    if other_index is None:
                        continue
                    # Mutual: the neighbour must point back at this track.
                    if nearest_by_track.get( other_index ) != track_index:
                        continue
                    lower, higher = sorted( ( track_index, other_index ) )
                    if lower in already_synced or higher in already_synced:
                        continue
                    if tracks[higher]['censor_style'] == tracks[lower]['censor_style']:
                        continue
                    already_synced.add( lower )
                    already_synced.add( higher )
                    # style_scale_w/h is deliberately NOT copied: a
                    # shared style means the same look, not the same
                    # dimensions. Two breasts on one person are often
                    # different sizes on screen, and forcing the smaller
                    # one to scale its blur from the larger one's box
                    # would over-blur it.
                    for key in ( 'censor_style', 'censor_shape', 'censor_sticker_seed' ):
                        tracks[higher][key] = tracks[lower][key]
                    for box in frame_boxes:
                        if box_track_index.get( id( box ) ) == higher:
                            for key in ( 'censor_style', 'censor_shape', 'censor_sticker_seed' ):
                                box[key] = tracks[lower][key]

            # Anything unmatched starts a new track - first sighting, or
            # nothing nearby survived max_gap. This is also the only
            # place a censor_style is resolved.
            for box in frame_boxes:
                if id( box ) in matched_box_ids:
                    continue

                if box.get( 'provisional' ):
                    # Score hysteresis: a below-min_prob detection may
                    # continue a track but never begin one.
                    stats['dropped_provisional'] += 1
                    continue

                paired_index = ( nearest_unambiguous_live_track(
                    box['x'], box['y'], box['w'], box['h'], box['start'],
                    reference_t=box.get( 't', box['start'] ) )
                    if settings.paired_style else None )
                if paired_index is not None:
                    source = tracks[paired_index]
                    box['censor_style'] = source['censor_style']
                    box['censor_shape'] = source['censor_shape']
                    box['censor_sticker_seed'] = source['censor_sticker_seed']
                else:
                    new_t = box.get( 't', box['start'] )
                    reuse = ( last_style['style'] is not None
                              and last_style['since'] is not None
                              and new_t - last_style['since']
                                  < settings.style_min_dwell_seconds )
                    if reuse:
                        # Still inside the dwell window, so this keeps
                        # the look the viewer is already watching rather
                        # than rolling a new one. That is the whole point
                        # of the dwell on cut-dense footage.
                        box['censor_style'] = last_style['style']
                        box['censor_shape'] = last_style['shape']
                        box['censor_sticker_seed'] = last_style['seed']
                    else:
                        box['censor_style'] = bu_censor.resolve_censor_style( box['censor_style'] )
                        box['censor_shape'] = bu_censor.resolve_censor_shape(
                            box['censor_style'], box.get( 'censor_shape', 'box' ) )
                        # censor_sticker_seed keeps whatever
                        # process_raw_box already rolled: a fresh
                        # per-track pick is exactly what a genuinely new
                        # track should get.
                        box.setdefault( 'censor_sticker_seed', random.random() )
                        last_style.update( style=box['censor_style'],
                                           shape=box['censor_shape'],
                                           seed=box['censor_sticker_seed'],
                                           since=new_t )
                        _notify( 'on_independent_style_resolve', label, box, tracks, settings )

                _apply_style_area_safety( box, box['censor_style'],
                                          settings.item_x_safety, settings.item_y_safety )

                track_id = next_track_id
                next_track_id += 1
                box['_track_id'] = track_id
                boxes_by_track.setdefault( track_id, [] ).append( box )
                track_hits[track_id] = 1
                # The size blur/pixel STRENGTH is scaled from, fixed for
                # this track's lifetime. Position and extent still track
                # the subject frame by frame; only the strength multiplier
                # is held, so a continuous censor keeps one kernel instead
                # of recomputing it from a box that jitters several pixels
                # every frame. See censor_scale_for_image_box.
                box['style_scale_w'] = box['w']
                box['style_scale_h'] = box['h']
                tracks.append( {
                    'id': track_id,
                    'x': box['x'], 'y': box['y'], 'w': box['w'], 'h': box['h'],
                    'end': box['end'], 't': box.get( 't', box['start'] ), 'score': box['score'],
                    'censor_style': box['censor_style'],
                    'censor_shape': box['censor_shape'],
                    'censor_sticker_seed': box['censor_sticker_seed'],
                    'style_since': box.get( 't', box['start'] ),
                    'style_scale_w': box['w'],
                    'style_scale_h': box['h'],
                } )
                _notify( 'on_new_track', label, tracks, len( tracks )-1, box, settings )

        # Track confirmation: a track that never reached min_track_hits
        # real detections contributes nothing, including its synthetic
        # interpolated boxes.
        for track_id, track_boxes in boxes_by_track.items():
            if track_hits.get( track_id, 0 ) >= settings.min_track_hits:
                output_boxes.extend( track_boxes )
                stats['tracks'] += 1
            else:
                stats['dropped_unconfirmed'] += len( track_boxes )

        if settings.min_track_hits > 1:
            logger.debug( "%s: min_track_hits=%d dropped %d box(es) from unconfirmed tracks"%(
                label, settings.min_track_hits, stats['dropped_unconfirmed'] ) )

    return output_boxes, stats


def prepare_boxes_for_render( raw_boxes, vid_w, vid_h, shot_cut_times=None,
                             backend_name=None, logger=None, profile_name=None,
                             sample_fps=None ):
    """
    The whole detection -> renderable-boxes pipeline, in one call.

    One function so betatv.py, betastare.py and the replay tooling all
    run the same stages in the same order with the same settings. Before
    2.1 this sequence was open-coded in betatv.py and re-derived by
    exec-ing betatv.py's source in the replay tool.

    Args:
        raw_boxes: Raw detections across every configured picture size.
        vid_w, vid_h: Frame size in pixels.
        shot_cut_times: Sorted cut timestamps, or None.
        backend_name: Backend whose settings apply.
        logger: Optional logger.
        profile_name: Structure profile whose overrides apply, or None.
        sample_fps: Rate the detections were sampled at. None means the
            global video_censor_fps.

    Returns:
        A (boxes, stats) pair. boxes is sorted by 'start' ascending,
        ready for the renderer. stats carries per-stage counts for the
        run's stats line.
    """
    logger = logger or bu_log.get_logger()
    backend_name = backend_name or bu_detector.selected_backend_name()
    parts_to_blur = bu_config.get_parts_to_blur( backend_name )
    suppression_rules = bu_detector.get_class_suppression( backend_name )

    stats = { 'raw_detections': len( raw_boxes ) }

    raw_boxes, dedup_counts = apply_cross_size_dedup( raw_boxes )
    stats['cross_size_deduped'] = sum( dedup_counts.values() )
    stats['cross_size_dedup_counts'] = dedup_counts

    # Promotion runs BEFORE geometry, relevance and suppression so a
    # promoted box is judged entirely as its new label: the target's
    # geometry bounds, the target's suppression rules, the target's
    # min_prob. It also has to run before the relevance filter, which
    # would otherwise discard the evidence labels it reads.
    raw_boxes, promotion_counts = apply_class_promotion(
        raw_boxes, bu_detector.get_class_promotion( backend_name ) )
    stats['promotion_counts'] = promotion_counts
    stats['promoted'] = sum( entry['promoted'] for entry in promotion_counts.values() )
    for rule_key, entry in sorted( promotion_counts.items() ):
        logger.info( "promotion %s: %d promoted, %d already covered, %d lacked evidence"%(
            rule_key, entry['promoted'], entry['already_covered'], entry['no_evidence'] ) )

    raw_boxes, geometry_counts = apply_geometry_filter( raw_boxes, vid_w, vid_h, backend_name )
    stats['geometry_filtered'] = sum( geometry_counts.values() )
    stats['geometry_filter_counts'] = geometry_counts

    # Drop detections nothing downstream can read, before the O(k^2)
    # suppression comparison. A label that is neither censored nor a
    # suppressor is invisible to every later stage, so this is a pure
    # speedup with no behavioural change.
    relevant = relevant_labels_for_suppression( suppression_rules, parts_to_blur.keys() )
    before_relevance = len( raw_boxes )
    raw_boxes = [ raw for raw in raw_boxes if raw['class_id'] in relevant ]
    stats['irrelevant_labels_dropped'] = before_relevance - len( raw_boxes )

    raw_boxes, suppression_counts = apply_class_suppression(
        raw_boxes, suppression_rules, parts_to_blur )
    stats['suppression_counts'] = suppression_counts
    stats['after_suppression'] = len( raw_boxes )

    label_counts = {}
    for raw in raw_boxes:
        label_counts[ raw['class_id'] ] = label_counts.get( raw['class_id'], 0 ) + 1
    stats['label_counts'] = label_counts

    boxes = []
    for raw in raw_boxes:
        censorable = bu_censor.process_raw_box( raw, vid_w, vid_h, parts_to_blur )
        if censorable:
            boxes.append( censorable )
    stats['censorable_boxes'] = len( boxes )

    logger.debug( "tracking %d censorable box(es) from %d raw detection(s), %d shot cut(s)"%(
        len( boxes ), stats['raw_detections'], len( shot_cut_times or [] ) ) )

    boxes, smoothing_stats = smooth_boxes( boxes, shot_cut_times, backend_name, profile_name,
                                           sample_fps )
    stats.update( smoothing_stats )
    boxes.sort( key=lambda box: box['start'] )
    stats['rendered_boxes'] = len( boxes )

    return boxes, stats
