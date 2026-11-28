"""
betautils_reverse.py - reverse mode's settings and its trigger timeline.

Reverse mode obscures the whole frame with a frame style while a
trigger is live, except two kinds of region:

  - windows: labels left uncensored (style type 'show', the original
    pixels), e.g. faces
  - censored labels: items_to_censor by default, each keeping its
    normal-mode look (a breast's mosaic/hex list stays its own instead
    of becoming the frame style), with optional reverse-only overrides

The trigger is on wherever a trigger detection holds it, and wherever a
censored label's box is on screen, so a censored object never sits in a
clear frame.

Analogy: normal mode spot-paints the few patches you want gone. Reverse
mode spray-paints the whole wall, with masking tape over what you want
to keep (windows) and the patches that need their own paint job painted
separately (censored labels); the trigger decides when the spray can
comes out at all.

This module owns the settings and the time axis only. The compositor
lives in betautils_censor (censor_img_reverse) beside every other
renderer, and the per-label resolution of a window's settings lives in
betautils_detector.get_item_overrides, the one place every label's
settings are resolved. This module imports nothing from the project but
betaconfig, so any of those can import it without a cycle.

The design, its error economics and the phase 0 cost measurements are in
CONFIG_REFERENCE.md -> "Reverse mode".
"""

import bisect

import betaconfig


# Keys that decide how a region LOOKS. A window never inherits these from
# the label's normal-mode settings (a window is always shown, never a
# sticker), while a censored label inherits them on purpose. Every other
# key (confidence, padding, tracking) is inherited from wherever it is
# set, for both.
LOOK_KEYS = frozenset( { 'censor_style', 'censor_shape' } )

# What a window renders as: the original pixels.
SHOW_STYLE = { 'type': 'show' }

EMPTY_FRAME_POLICIES = ( 'passthrough', 'obscure' )

DEFAULTS = {
    'enabled': False,
    'frame_style': { 'type': 'blur', 'method': 'triple_box', 'strength': 7, 'dim': 0.4 },
    'empty_frame': 'passthrough',
    'fade_seconds': 0.25,
    'trigger': {
        'labels': None,
        'min_prob': 0.30,
        'hold_seconds': 0.5,
    },
    'censor': {
        'labels': None,
        'overrides': {},
    },
    'windows': {},
    'person_windows': {
        'enabled': False,
        'male_labels': [ 'face_masc', 'exposed_chest' ],
        'female_labels': [ 'face_femme', 'exposed_breast', 'exposed_vulva' ],
        'evidence_min_prob': 0.40,
        'track_iou': 0.30,
        'track_max_gap_seconds': 1.0,
        'shrink': 0.02,
    },
}

# The nested blocks settings() fills key by key, so a config that sets
# one key keeps the defaults of the rest.
_NESTED = ( 'trigger', 'censor', 'person_windows' )


def raw_block():
    """betaconfig.reverse as configured, or {} when it is absent or not a dict."""
    block = getattr( betaconfig, 'reverse', None )
    return block if isinstance( block, dict ) else {}


def settings():
    """
    The reverse block with every default filled in.

    A fresh dict on every call; mutating it changes nothing. 'trigger'
    and 'censor' are filled key by key, so a config that sets only the
    trigger's min_prob keeps the default hold.
    """
    block = raw_block()
    resolved = { key: block.get( key, default ) for key, default in DEFAULTS.items()
                 if key not in _NESTED }
    for name in _NESTED:
        nested = block.get( name )
        nested = nested if isinstance( nested, dict ) else {}
        resolved[name] = { key: nested.get( key, default )
                           for key, default in DEFAULTS[name].items() }
    windows = resolved.get( 'windows' )
    resolved['windows'] = dict( windows ) if isinstance( windows, dict ) else {}
    overrides = resolved['censor'].get( 'overrides' )
    resolved['censor']['overrides'] = dict( overrides ) if isinstance( overrides, dict ) else {}
    return resolved


def enabled():
    """Whether reverse mode is on for this run."""
    return bool( raw_block().get( 'enabled', False ) )


def person_windows_enabled():
    """Whether reverse mode is on AND whole male bodies are windowed from person masks."""
    return enabled() and bool( settings()['person_windows']['enabled'] )


def window_labels():
    """The labels that get a window, sorted, or [] when reverse mode is off."""
    if not enabled():
        return []
    return sorted( settings()['windows'] )


def window_entry( label ):
    """
    One label's window entry while reverse mode is on, else None.

    None means "no window for this label", which is different from {},
    a window with every setting inherited.
    """
    if not enabled():
        return None
    entry = settings()['windows'].get( label )
    if entry is None:
        return None
    return dict( entry ) if isinstance( entry, dict ) else {}


def censor_labels():
    """
    The labels censored inside a triggered frame, sorted, or [] when
    reverse mode is off. censor['labels'] None means items_to_censor.
    """
    if not enabled():
        return []
    configured = settings()['censor']['labels']
    if configured is None:
        configured = getattr( betaconfig, 'items_to_censor', [] )
    return sorted( set( configured ) )


def censor_override( label ):
    """
    One censored label's reverse-only overrides while reverse mode is on
    ({} when it has none), else None. None also for a label that is not
    censored in reverse mode.
    """
    if label not in censor_labels():
        return None
    entry = settings()['censor']['overrides'].get( label )
    return dict( entry ) if isinstance( entry, dict ) else {}


def rendered_labels():
    """Every label reverse mode renders, windows and censored, sorted; [] while off."""
    return sorted( set( window_labels() ) | set( censor_labels() ) )


def trigger_labels():
    """
    The labels whose detections switch the frame style on.

    labels: None means every censored label: the frame is obscured when
    something reverse mode censors is detected, not when a window (a
    face) is.
    """
    configured = settings()['trigger']['labels']
    if configured is None:
        return set( censor_labels() )
    return set( configured )


def is_show_style( style ):
    """True for a resolved style that shows the original pixels."""
    return isinstance( style, dict ) and style.get( 'type' ) == 'show'


# ---------------------------------------------------------------------------
# Trigger timeline
# ---------------------------------------------------------------------------

def merge_intervals( intervals ):
    """Sorted, non-overlapping (start, end) tuples, merged where they touch."""
    merged = []
    for start, end in sorted( intervals ):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append( [ start, end ] )
    return [ ( start, end ) for start, end in merged ]


def trigger_intervals( detection_times, lead_seconds, hold_seconds ):
    """
    Merge trigger detections into sorted, non-overlapping on-intervals.

    Each detection at time t holds the trigger on over
    [t - lead_seconds, t + hold_seconds].

    The lead is one sample interval, passed in by the caller. Detection
    only looks every 1/fps seconds, so the target can be on screen for
    up to one interval before the first detection that sees it; without
    the lead the frame style would come on (and its fade complete) only
    at that first detection, leaving the gap clear. A clear frame with
    the target in it is reverse mode's largest reveal, so the trigger
    errs early. The lead also joins consecutive samples into one
    interval whatever hold is set to.

    Args:
        detection_times: Times in seconds, any order.
        lead_seconds: How far before each detection the trigger starts.
        hold_seconds: How long after each detection it stays on.

    Returns:
        A list of (start, end) tuples, sorted, merged where they touch.
    """
    return merge_intervals( ( max( 0.0, time_seconds - lead_seconds ), time_seconds + hold_seconds )
                            for time_seconds in detection_times )


def censored_box_spans( boxes ):
    """
    (start, end) of every render box of a censored label.

    The trigger also holds on over each of these: a censored box can be
    on screen where no trigger detection is (it scored between its own
    min_prob and the trigger's, or tracking held it through a gap longer
    than the trigger's hold), and a censored object in a clear frame is
    exactly what reverse mode exists to prevent.
    """
    labels = set( censor_labels() )
    return [ ( box['start'], box['end'] ) for box in boxes if box.get( 'label' ) in labels ]


class TriggerTimeline:
    """
    The frame style's opacity at any time, 0 (clear) to 1 (obscured).

    Look-ahead fade: betatv renders offline and knows the future, so a
    fade-in FINISHES by an interval's start rather than beginning there,
    and a fade-out begins only once the interval ends. The trigger is
    therefore fully on for the whole of every interval, and the fades
    only ever add obscuring outside them, never remove it inside.

    Read with a bisect per frame rather than a stateful cursor: render
    chunks start anywhere and retry from their start, and a stateless
    lookup cannot go stale across either.
    """

    def __init__( self, intervals, fade_seconds, always_on=False ):
        self.intervals = list( intervals )
        self.fade_seconds = max( 0.0, float( fade_seconds or 0.0 ) )
        self.always_on = bool( always_on )
        self._starts = [ start for start, _end in self.intervals ]

    def alpha_at( self, time_seconds ):
        """The frame style's opacity at time_seconds."""
        if self.always_on:
            return 1.0
        if not self.intervals:
            return 0.0
        index = bisect.bisect_right( self._starts, time_seconds ) - 1
        if index >= 0 and time_seconds <= self.intervals[index][1]:
            return 1.0
        if self.fade_seconds <= 0:
            return 0.0
        alpha = 0.0
        if index >= 0:
            since_end = time_seconds - self.intervals[index][1]
            if since_end < self.fade_seconds:
                alpha = max( alpha, 1.0 - since_end / self.fade_seconds )
        if index + 1 < len( self.intervals ):
            until_start = self.intervals[index + 1][0] - time_seconds
            if until_start < self.fade_seconds:
                alpha = max( alpha, 1.0 - until_start / self.fade_seconds )
        return alpha

    def triggered_seconds( self ):
        """Total time the trigger holds fully on, fades excluded."""
        return sum( end - start for start, end in self.intervals )


def timeline_for_detections( detection_times, sample_fps, box_spans=() ):
    """
    The run's TriggerTimeline from its trigger detection times.

    Args:
        detection_times: Times of every detection that qualifies as a
            trigger (labels and min_prob already applied).
        sample_fps: The rate those detections were sampled at; sets the
            lead before each one.
        box_spans: (start, end) of every censored render box
            (censored_box_spans), held on as they are, no lead or hold.
    """
    resolved = settings()
    lead = 1.0 / sample_fps if sample_fps else 0.0
    intervals = merge_intervals(
        trigger_intervals( detection_times, lead, resolved['trigger']['hold_seconds'] )
        + list( box_spans ) )
    return TriggerTimeline( intervals, resolved['fade_seconds'],
                            always_on=( resolved['empty_frame'] == 'obscure' ) )


class ReverseRender:
    """
    What the renderer needs for a reverse-mode video: when, with what,
    and (with person windows on) which pixels belong to male bodies.
    """

    __slots__ = ( 'timeline', 'frame_style', 'window_masks' )

    def __init__( self, timeline, frame_style, window_masks=None ):
        self.timeline = timeline
        self.frame_style = dict( frame_style )
        # Anything with mask_at(t, frame_w, frame_h); see
        # betautils_person.PersonMaskTimeline. None: no person windows.
        self.window_masks = window_masks

    def frame_alpha( self, time_seconds ):
        return self.timeline.alpha_at( time_seconds )

    def window_mask( self, time_seconds, frame_w, frame_h ):
        """(x, y, w, h, alpha) of the person windows at this time, or None."""
        if self.window_masks is None:
            return None
        return self.window_masks.mask_at( time_seconds, frame_w, frame_h )


def render_plan_for_detections( detection_times, sample_fps, boxes=() ):
    """
    A ReverseRender for one video, from its trigger detection times and
    its render boxes (whose censored ones hold the trigger on too).
    """
    return ReverseRender( timeline_for_detections( detection_times, sample_fps,
                                                   censored_box_spans( boxes ) ),
                          settings()['frame_style'] )


def frame_alpha_for_still( raw_boxes, boxes=() ):
    """
    The frame style's opacity for a still image (betastare): on when any
    raw detection of a trigger label scores above the trigger's
    min_prob, or when any censored label has a box to render. No time
    axis, so no hold, no lead and no fade.
    """
    if settings()['empty_frame'] == 'obscure':
        return 1.0
    if censored_box_spans( boxes ):
        return 1.0
    labels = trigger_labels()
    min_prob = settings()['trigger']['min_prob']
    triggered = any( raw['class_id'] in labels and raw['score'] > min_prob for raw in raw_boxes )
    return 1.0 if triggered else 0.0


def identity():
    """
    Everything in the reverse block that changes rendered output, for
    the censor key. None while reverse mode is off, so turning the
    feature into the code left every existing render's key unchanged.
    """
    if not enabled():
        return None
    resolved = settings()
    resolved['trigger'] = dict( resolved['trigger'] )
    resolved['trigger']['resolved_labels'] = sorted( trigger_labels() )
    resolved['censor'] = dict( resolved['censor'] )
    resolved['censor']['resolved_labels'] = censor_labels()
    return resolved
