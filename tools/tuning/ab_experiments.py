"""
ab_experiments.py - what ab_compare.py can put side by side.

An experiment is a question of the shape "the config as it is, or this
change?", answered by eye on the footage where the two differ most.
ab_compare.py does the generic part (replay both sides through the
production pipeline, rank videos, pick clip windows, render, collect
verdicts); an experiment supplies the parts that depend on what changes:

  propose      the proposed side, from the bench, auto_tune or --set
  applied      a context in which the production pipeline reads a side
  attribute    which change caused each differing box
  choose       the chosen side, change by change, from the verdicts
  chosen_lines the config lines a person pastes

Experiments here:
  suppression     class_suppression rules (the bench's retune/drop/add proposal)
  min_track_hits  per-label item override, from auto_tune's review values
  confirm_score   per-label item override, from auto_tune's review values
                  or the confident score
  match_distance  per-label match_distance_multiplier, same source
  tracking        any per-label item override, from --set label.key=value
  cut_bursts      the shot-cut burst merge (shot_cut_burst_* globals)

A side is a JSON value: the current side is always the config as it is
({} for the override and global experiments, the configured rules for
suppression), so the comparison never assumes a value the config lacks.
"""

import contextlib
import copy
import glob
import json
import math
import os

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_track as bu_track
import betautils_tuning as bu_tuning


TRACKING_CAUSE = 'tracking'
BUCKET_SECONDS = 0.5
CORE_DIR = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
ANALYSIS_RUNS = os.path.join( os.path.dirname( CORE_DIR ), 'output', 'logs', 'analysis_runs' )


def parse_value( text ):
    """'1' -> 1, '0.75' -> 0.75, 'none' -> None, 'true' -> True, else the text."""
    lowered = str( text ).strip().lower()
    if lowered in ( 'none', 'null' ):
        return None
    if lowered in ( 'true', 'false' ):
        return lowered == 'true'
    try:
        return json.loads( text )
    except ( TypeError, ValueError ):
        return text


def _format( value ):
    return '%g'%( value, ) if isinstance( value, float ) else repr( value ) if isinstance( value, str ) else str( value )


class Experiment( object ):
    """The interface ab_compare.py drives. Subclasses fill in each part."""

    name = ''
    title = ''
    summary = ''
    chosen_file = 'chosen.json'

    def current( self, backend_name ):
        """The config as it is, as a side."""
        return {}

    def propose( self, args, logger, config, betabench ):
        """(proposed side, where it came from)."""
        raise NotImplementedError

    def change_lines( self, current, proposed, backend_name ):
        """One line per change: 'what: old -> new'."""
        raise NotImplementedError

    def changes_by_cause( self, current, proposed, backend_name ):
        """{cause: what changed}, the cause names attribute() returns."""
        raise NotImplementedError

    def is_noop( self, current, proposed, backend_name ):
        return not self.changes_by_cause( current, proposed, backend_name )

    def applied( self, side, backend_name, tracking ):
        """A context in which the production pipeline reads `side`."""
        raise NotImplementedError

    def make_observer( self ):
        """A bu_track.TrackingObserver to watch one side's replay, or None."""
        return None

    def attribute( self, boxes, other_observer, changes ):
        """The cause of each box censored on one side only."""
        raise NotImplementedError

    def choose( self, current, proposed, tally, backend_name ):
        """(chosen side, {cause: decision}) from the verdict tally."""
        raise NotImplementedError

    def chosen_lines( self, chosen, current, backend_name ):
        """The config lines that make `chosen` the config."""
        raise NotImplementedError

    def explain_intro( self ):
        return ( "LEFT is the config as it is, RIGHT the proposed change. Each row below is "
                 "one moment where the two censor differently, and the change that caused it." )

    def follow_up_lines( self, current, proposed, decisions, backend_name ):
        """Config lines worth pasting whatever was adopted (see SuppressionExperiment)."""
        return []

    def note( self, cause, proposed ):
        """The text drawn under a differing box's label in the outline render."""
        return cause


def _prefer( counts ):
    """True when the verdicts favour the proposed side: more 'right' than 'left'. A tie keeps the config."""
    return counts.get( 'proposed', 0 ) > counts.get( 'current', 0 )


def _decision( counts ):
    if _prefer( counts ):
        return 'proposed'
    return 'current' if counts.get( 'current', 0 ) else 'current (no verdict yet)'


# --- suppression rules -------------------------------------------------------

def _rule_list( rules ):
    return [ rules ] if isinstance( rules, dict ) else list( rules or [] )


def proposed_rules( configured, suggestions, refused=(), additions=True, min_win_rate=0.0 ):
    """
    The configured rules retuned or dropped per the bench, plus additions.

    suggestions is betabench suppression's rows ({'pair': 'L<-S',
    'suggested_min_iou', 'suggested_margin', 'same_spot_win_rate'}). A
    configured pair with a suggestion takes its min_iou and margin; one
    the bench measured and found no same-spot cluster for is dropped;
    one with too few overlaps to measure (no row) stays as configured,
    since there is no evidence either way.

    With additions, a pair that is not configured joins the proposed set
    when the bench found a same-spot cluster for it, unless it is in
    refused (class_suppression_refused: a person decided against it) or
    its suppressor wins under min_win_rate of those overlaps (the bench's
    ADDITION_MIN_WIN_RATE, below which a rule would almost never fire).
    Each addition is its own change in the round, judged on its own.
    """
    by_pair = { row['pair']: row for row in suggestions }
    proposed = {}
    configured_pairs = set()
    for label, rules in configured.items():
        for rule in _rule_list( rules ):
            pair = '%s<-%s'%( label, rule['suppressed_by'] )
            configured_pairs.add( pair )
            row = by_pair.get( pair )
            if row is None:
                proposed.setdefault( label, [] ).append( dict( rule ) )
                continue
            if row.get( 'suggested_min_iou' ) is None:
                continue
            proposed.setdefault( label, [] ).append( {
                'suppressed_by': rule['suppressed_by'],
                'margin': row['suggested_margin'],
                'min_iou': row['suggested_min_iou'],
            } )
    if additions:
        for row in suggestions:
            pair = row['pair']
            if ( pair in configured_pairs or pair in refused
                 or row.get( 'suggested_min_iou' ) is None
                 or row.get( 'same_spot_win_rate', 0.0 ) < min_win_rate ):
                continue
            label, _sep, suppressor = pair.partition( '<-' )
            proposed.setdefault( label, [] ).append( {
                'suppressed_by': suppressor,
                'margin': row['suggested_margin'],
                'min_iou': row['suggested_min_iou'],
            } )
    return proposed


def added_pairs( current, proposed ):
    """'L<-S' pairs the proposed side has and the current side does not."""
    def pairs( rules ):
        return { '%s<-%s'%( label, rule['suppressed_by'] )
                 for label, entries in rules.items() for rule in _rule_list( entries ) }
    return sorted( pairs( proposed ) - pairs( current ) )


def rule_changes( current, proposed ):
    """One line per pair: what the proposed set does to it."""
    def pairs( rules ):
        return { ( label, rule['suppressed_by'] ): rule
                 for label, entries in rules.items() for rule in _rule_list( entries ) }
    before, after = pairs( current ), pairs( proposed )
    lines = []
    for key in sorted( set( before ) | set( after ) ):
        name = '%s<-%s'%key
        old, new = before.get( key ), after.get( key )
        if new is None:
            lines.append( "%s: dropped"%( name, ) )
        elif old is None:
            lines.append( "%s: added (iou %s, margin %s)"%( name, new.get( 'min_iou' ), new.get( 'margin' ) ) )
        elif ( old.get( 'min_iou' ), old.get( 'margin' ) ) == ( new.get( 'min_iou' ), new.get( 'margin' ) ):
            lines.append( "%s: unchanged"%( name, ) )
        else:
            lines.append( "%s: retuned iou %s -> %s, margin %s -> %s"%(
                name, old.get( 'min_iou' ), new.get( 'min_iou' ), old.get( 'margin' ), new.get( 'margin' ) ) )
    return lines


def changes_by_rule( current, proposed ):
    """{'L<-S': 'dropped' / 'retuned ...' / 'added ...' / 'unchanged'} from rule_changes."""
    found = {}
    for line in rule_changes( current, proposed ):
        rule, _sep, what = line.partition( ': ' )
        found[rule] = what
    return found


class SuppressionLog( bu_track.TrackingObserver ):
    """Every detection a replay's suppression dropped: (t, label, centre x, centre y, rule)."""

    def __init__( self ):
        self.entries = []

    def on_suppressed( self, raw, rule_key, suppressor ):
        self.entries.append( ( raw['t'], raw['class_id'], raw['x'] + raw['w'] / 2.0,
                               raw['y'] + raw['h'] / 2.0, rule_key ) )


def attribute( boxes, suppressed ):
    """
    The cause of each box being censored under one set and not the other.

    A box the other set lacks is usually there because the other set's
    rules dropped a detection inside it: the rule that did it, when one
    of those detections sits inside the box while it is live. Otherwise
    the difference came from tracking (a dropped detection elsewhere
    changed which track confirmed or how far it carried): TRACKING_CAUSE.
    """
    by_bucket = {}
    for entry in suppressed:
        by_bucket.setdefault( ( entry[1], int( math.floor( entry[0] / BUCKET_SECONDS ) ) ), [] ).append( entry )
    causes = []
    for box in boxes:
        counts = {}
        for bucket in range( int( math.floor( box['start'] / BUCKET_SECONDS ) ),
                             int( math.floor( box['end'] / BUCKET_SECONDS ) ) + 1 ):
            for t, _label, centre_x, centre_y, rule in by_bucket.get( ( box.get( 'label' ), bucket ), () ):
                if ( box['start'] <= t <= box['end']
                     and box['x'] <= centre_x <= box['x'] + box['w']
                     and box['y'] <= centre_y <= box['y'] + box['h'] ):
                    counts[rule] = counts.get( rule, 0 ) + 1
        causes.append( max( sorted( counts ), key=counts.get ) if counts else TRACKING_CAUSE )
    return causes


def choose_rules( current, proposed, tally ):
    """
    (chosen rules, {rule: decision}): each changed rule takes the side its
    verdicts favour. A tie or no verdict keeps the current rule: no
    evidence, no change.
    """
    def pairs( rules ):
        return { '%s<-%s'%( label, rule['suppressed_by'] ): ( label, rule )
                 for label, entries in rules.items() for rule in _rule_list( entries ) }
    before, after = pairs( current ), pairs( proposed )
    chosen = { key: value for key, value in before.items() }
    decisions = {}
    for key in sorted( set( before ) | set( after ) ):
        if before.get( key, ( None, None ) )[1] == after.get( key, ( None, None ) )[1]:
            continue
        counts = tally.get( key, {} )
        decisions[key] = _decision( counts )
        if decisions[key] == 'proposed':
            if key not in after:
                chosen.pop( key, None )
            elif key in chosen:
                chosen[key] = after[key]
    # The current config's order, so the block reads like the one it replaces.
    for key, value in after.items():
        if decisions.get( key ) == 'proposed' and key not in before:
            chosen[key] = value
    rules = {}
    for _key, ( label, rule ) in chosen.items():
        rules.setdefault( label, [] ).append( dict( rule ) )
    return rules, decisions


class SuppressionExperiment( Experiment ):

    name = 'suppression'
    title = 'suppression rules'
    summary = "class_suppression: the bench's retune/drop/add proposal against the configured rules"
    chosen_file = 'chosen_rules.json'

    def current( self, backend_name ):
        return copy.deepcopy( bu_detector.get_class_suppression( backend_name ) or {} )

    def propose( self, args, logger, config, betabench ):
        import argparse
        current = self.current( config.backend_name )
        logger.info( "deriving the proposed rules from %s's caches (betabench suppression)"%(
            config.label, ) )
        raw_boxes, _count = betabench.load_cached_detections(
            config.backend_name, logger, picture_sizes=config.picture_sizes, configuration=config )
        bench_args = argparse.Namespace( min_samples=args.min_samples )
        rows = betabench._bench_suppression_one( bench_args, logger, config, raw_boxes )['suppression']
        del raw_boxes
        refused = bu_detector.get_class_suppression_refused( config.backend_name )
        additions = getattr( args, 'additions', True )
        proposed = proposed_rules( current, rows, refused=refused, additions=additions,
                                   min_win_rate=betabench.ADDITION_MIN_WIN_RATE )
        added = added_pairs( current, proposed )
        if refused:
            logger.info( "never proposed as additions (class_suppression_refused): %s"%(
                ", ".join( refused ), ) )
        source = 'betabench suppression suggestions'
        if added:
            source += ', with %d addition(s): %s'%( len( added ), ', '.join( added ) )
        elif not additions:
            source += ', additions off'
        return proposed, source

    def change_lines( self, current, proposed, backend_name ):
        return rule_changes( current, proposed )

    def changes_by_cause( self, current, proposed, backend_name ):
        return { rule: what for rule, what in changes_by_rule( current, proposed ).items()
                 if what != 'unchanged' }

    def applied( self, side, backend_name, tracking ):
        return bu_tuning.class_suppression_in_effect( backend_name, side )

    def make_observer( self ):
        return SuppressionLog()

    def attribute( self, boxes, other_observer, changes ):
        return attribute( boxes, other_observer.entries if other_observer else [] )

    def choose( self, current, proposed, tally, backend_name ):
        return choose_rules( current, proposed, tally )

    def chosen_lines( self, chosen, current, backend_name ):
        lines = [ "detector_backend[%r]['class_suppression'] = {"%( backend_name, ) ]
        for label, entries in sorted( chosen.items() ):
            lines.append( "    '%s': ["%( label, ) )
            for rule in entries:
                lines.append( "        { 'suppressed_by': '%s', 'margin': %s, 'min_iou': %s },"%(
                    rule['suppressed_by'], rule.get( 'margin' ), rule.get( 'min_iou' ) ) )
            lines.append( "    ]," )
        lines.append( "}" )
        return lines

    def follow_up_lines( self, current, proposed, decisions, backend_name ):
        """
        The refused list with every addition the verdicts turned down, so
        the next round does not propose it again. An addition with no
        verdict yet is not turned down: it was never judged.
        """
        turned_down = [ pair for pair in added_pairs( current, proposed )
                        if decisions.get( pair ) == 'current' ]
        if not turned_down:
            return []
        refused = bu_detector.get_class_suppression_refused( backend_name )
        merged = refused + [ pair for pair in turned_down if pair not in refused ]
        return [ "so the next round does not propose %s again:"%( ", ".join( turned_down ), ),
                 "detector_backend[%r]['class_suppression_refused'] = %r"%( backend_name, merged ) ]

    def note( self, cause, proposed ):
        # The label is drawn on the line above, so the suppressor alone.
        if cause == TRACKING_CAUSE or '<-' not in cause:
            return cause
        return '<-' + cause.split( '<-', 1 )[-1]

    def explain_intro( self ):
        return ( "LEFT is the current rules, RIGHT the proposed ones. Each row below is one "
                 "moment where the two censor differently, and the rule that caused it. RIGHT "
                 "censors: the CURRENT rule in the cause column removes that censor today, and the "
                 "proposed change (dropped or retuned) lets it through. LEFT censors: the PROPOSED "
                 "rule removes a censor the current rules keep; for a rule marked 'added' in the "
                 "change list that is the whole of what it does, so judge whether the censor it "
                 "takes away was on the right thing. In `_diff.mp4` the boxes only RIGHT "
                 "has are magenta and the ones only LEFT has are orange, with the cause written "
                 "under the label." )


# --- per-label item overrides ----------------------------------------------------

def latest_auto_tune_decisions( root=None ):
    """The newest analysis run's auto_tune decisions.json, or None."""
    paths = sorted( glob.glob( os.path.join( root or ANALYSIS_RUNS, '*', 'auto_tune', 'decisions.json' ) ) )
    return paths[-1] if paths else None


_STAGE_KEYS = { 'match_distance': 'match_distance_multiplier' }


def review_values( decisions, key ):
    """
    {label: value} auto_tune flagged for `key` as worth judging by eye.

    A decision's 'review' when it has one. Older decisions.json files
    predate it, so for those: the accepted value when it differs from
    the baseline, else a tightening candidate found the same way
    auto_tune finds one now (bu_tuning.tightening_review).
    """
    found = {}
    for decision in decisions:
        name, _sep, rest = decision.get( 'stage', '' ).partition( ' / ' )
        _backend, _sep, label = rest.partition( ' / ' )
        stage_key = decision.get( 'key' ) or _STAGE_KEYS.get( name, name )
        label = decision.get( 'label' ) or label
        if stage_key != key or not label:
            continue
        review = decision.get( 'review' )
        if review is None and 'review' not in decision:
            review = _review_from_rows( decision )
        if review is not None:
            found[label] = review['value']
    return found


def _review_from_rows( decision ):
    rows = decision.get( 'rows' ) or []
    baseline = [ row for row in rows if row.get( 'verdict' ) == 'baseline' ]
    baseline_value = baseline[0]['value'] if baseline else None
    if decision.get( 'accepted' ) and decision.get( 'chosen' ) not in ( None, baseline_value ):
        return { 'value': decision['chosen'] }
    if baseline_value is None:
        return None
    return bu_tuning.tightening_review( rows, baseline_value )


def _profiles_setting( label, key, backend_name ):
    """{profile: value} for profiles that set label's key themselves."""
    variants = ( bu_detector.get_profiles( backend_name ).get( 'variants' ) or {} )
    found = {}
    for name in sorted( variants ):
        overrides = bu_detector.get_profile_item_overrides( name, label, backend_name )
        if key in overrides:
            found[name] = overrides[key]
    return found


class OverrideExperiment( Experiment ):
    """
    Per-label item overrides: {label: {key: value}}. The current side is
    {} (the config as it is); the proposed side is applied through
    bu_tuning.item_overrides_in_effect, which also replaces the value in
    any profile that sets the key itself, so the change reaches every
    file. One change per label: verdicts cannot tell two keys of one
    label apart, so a label's keys are judged together.
    """

    def __init__( self, name, key=None, title=None, summary=None, default_value=None ):
        self.name = name
        self.key = key
        self.title = title or name
        self.summary = summary or "per-label item override"
        # The value every censored label is proposed when auto_tune has
        # nothing to review for this key, or None to require one.
        self.default_value = default_value

    def propose( self, args, logger, config, betabench ):
        patch = parse_sets( getattr( args, 'set', None ) or [] )
        if patch:
            return patch, '--set'
        if not self.key:
            raise SystemExit( "the %s experiment needs --set label.key=value (AB_SET in the "
                              "shell)"%( self.name, ) )
        path = getattr( args, 'auto_tune', None ) or latest_auto_tune_decisions()
        values = {}
        if path and os.path.exists( path ):
            with open( path ) as handle:
                values = review_values( json.load( handle ), self.key )
            logger.info( "auto_tune decisions: %s"%( path, ) )
        censored = set( betaconfig.items_to_censor )
        patch = { label: { self.key: value } for label, value in sorted( values.items() )
                  if label in censored }
        if patch:
            return patch, "auto_tune's review values (%s)"%( os.path.basename(
                os.path.dirname( os.path.dirname( path ) ) ), )
        if self.default_value is not None:
            logger.info( "auto_tune flagged nothing to review for %s (%s); proposing %s for every "
                         "censored label"%( self.key, path or 'no decisions.json found',
                                            _format( self.default_value ) ) )
            return ( { label: { self.key: self.default_value } for label in sorted( censored ) },
                     "the default %s of %s"%( self.key, _format( self.default_value ) ) )
        raise SystemExit(
            "auto_tune flagged nothing to review for %s (%s). Run the analysis hub again, or name "
            "the values: AB_SET='exposed_breast.%s=VALUE' (--set)"%(
                self.key, path or 'no decisions.json found', self.key ) )

    def _from_value( self, label, key, backend_name ):
        return bu_config.get_label_settings( label, backend_name ).get(
            key, bu_detector.get_item_overrides( label, backend_name ).get( key ) )

    def change_lines( self, current, proposed, backend_name ):
        lines = []
        for label in sorted( proposed ):
            for key, value in sorted( proposed[label].items() ):
                old = self._from_value( label, key, backend_name )
                profiles = _profiles_setting( label, key, backend_name )
                note = ( " (also replaces %s)"%( ", ".join( "%s's %s"%( name, _format( setting ) )
                                                          for name, setting in profiles.items() ), )
                         if profiles else "" )
                lines.append( "%s: %s %s -> %s%s"%( label, key, _format( old ), _format( value ), note ) )
        return lines

    def changes_by_cause( self, current, proposed, backend_name ):
        changed = {}
        for label in sorted( proposed ):
            parts = [ "%s %s -> %s"%( key, _format( self._from_value( label, key, backend_name ) ),
                                      _format( value ) )
                      for key, value in sorted( proposed[label].items() )
                      if value != self._from_value( label, key, backend_name )
                      or any( setting != value for setting in
                              _profiles_setting( label, key, backend_name ).values() ) ]
            if parts:
                changed[label] = ", ".join( parts )
        return changed

    def applied( self, side, backend_name, tracking ):
        if not side:
            return contextlib.nullcontext()
        return bu_tuning.item_overrides_in_effect( backend_name, side,
                                                  profile_name=tracking.get( 'profile_name' ) )

    def note( self, cause, proposed ):
        # The label is drawn above it; name the setting that changed.
        return '+'.join( sorted( proposed.get( cause, {} ) ) ) or cause

    def attribute( self, boxes, other_observer, changes ):
        # A label's tracking settings move only that label's boxes.
        return [ box.get( 'label' ) if box.get( 'label' ) in changes else TRACKING_CAUSE
                 for box in boxes ]

    def choose( self, current, proposed, tally, backend_name ):
        chosen, decisions = {}, {}
        for label in sorted( self.changes_by_cause( current, proposed, backend_name ) ):
            decisions[label] = _decision( tally.get( label, {} ) )
            if decisions[label] == 'proposed':
                chosen[label] = dict( proposed[label] )
        return chosen, decisions

    def chosen_lines( self, chosen, current, backend_name ):
        lines = []
        for label in sorted( chosen ):
            for key, value in sorted( chosen[label].items() ):
                lines.append( "detector_backend[%r]['item_overrides'][%r][%r] = %s"%(
                    backend_name, label, key, _format( value ) ) )
                for profile, setting in _profiles_setting( label, key, backend_name ).items():
                    lines.append( "    and in profiles['variants'][%r]['item_overrides'][%r][%r] "
                                  "(now %s), or remove it there"%( profile, label, key, _format( setting ) ) )
        return lines or [ "(nothing: every change kept the config as it is)" ]


def parse_sets( items ):
    """['exposed_breast.min_track_hits=1', ...] -> {'exposed_breast': {'min_track_hits': 1}}."""
    patch = {}
    for item in items:
        name, sep, value = item.partition( '=' )
        label, dot, key = name.partition( '.' )
        if not sep or not dot or not label or not key:
            raise SystemExit( "--set wants label.key=value, got %r"%( item, ) )
        patch.setdefault( label.strip(), {} )[key.strip()] = parse_value( value )
    return patch


# --- global config keys -------------------------------------------------------------

CUT_BURST_KEYS = ( 'shot_cut_burst_window', 'shot_cut_burst_max_span', 'shot_cut_burst_keep' )
# What the code reads when the keys are absent from betaconfig: off.
CUT_BURST_ABSENT = { 'shot_cut_burst_window': 0.0, 'shot_cut_burst_max_span': None,
                     'shot_cut_burst_keep': 'first' }
# The best variant of the 2026-10-11 analysis, for when no newer one is on disk.
CUT_BURST_DEFAULT = { 'shot_cut_burst_window': 0.35, 'shot_cut_burst_max_span': None,
                      'shot_cut_burst_keep': 'first' }


def latest_cut_burst_best( root=None ):
    """The newest analysis run's analyze_cut_bursts best variant as config keys, or None."""
    paths = sorted( glob.glob( os.path.join( root or ANALYSIS_RUNS, '*', 'analyze_cut_bursts.json' ) ) )
    for path in reversed( paths ):
        try:
            with open( path ) as handle:
                best = json.load( handle ).get( 'best_settings' )
        except ( OSError, ValueError ):
            continue
        if best:
            return { 'shot_cut_burst_window': best['window'],
                     'shot_cut_burst_max_span': best['max_span'],
                     'shot_cut_burst_keep': best['keep'] }, path
    return None


@contextlib.contextmanager
def globals_in_effect( values ):
    """betaconfig reads `values` ({name: value}) inside, its own afterwards."""
    missing = object()
    saved = { name: getattr( betaconfig, name, missing ) for name in values }
    try:
        for name, value in values.items():
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()
        yield
    finally:
        for name, value in saved.items():
            if value is missing:
                if hasattr( betaconfig, name ):
                    delattr( betaconfig, name )
            else:
                setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()


class GlobalsExperiment( Experiment ):
    """
    A group of global betaconfig keys judged as one change: {name: value}.
    Every differing box has the one cause, the experiment's name.
    """

    def __init__( self, name, keys, defaults, absent, title, summary ):
        self.name = name
        self.keys = keys
        self.defaults = defaults
        self.absent = absent
        self.title = title
        self.summary = summary

    def propose( self, args, logger, config, betabench ):
        sets = getattr( args, 'set', None ) or []
        if sets:
            values = {}
            for item in sets:
                name, sep, value = item.partition( '=' )
                if not sep or name.strip() not in self.keys:
                    raise SystemExit( "--set for %s wants one of %s as name=value, got %r"%(
                        self.name, ", ".join( self.keys ), item ) )
                values[name.strip()] = parse_value( value )
            return dict( self._now(), **values ), '--set'
        found = latest_cut_burst_best() if self.name == 'cut_bursts' else None
        if found:
            return found[0], "analyze_cut_bursts' best variant (%s)"%( os.path.basename(
                os.path.dirname( found[1] ) ), )
        return dict( self.defaults ), 'the measured default (window 0.35s, any span, keep first)'

    def _now( self ):
        return { name: getattr( betaconfig, name, self.absent.get( name ) ) for name in self.keys }

    def change_lines( self, current, proposed, backend_name ):
        now = self._now()
        return [ "%s: %s -> %s"%( name, _format( now.get( name ) ), _format( proposed[name] ) )
                 for name in self.keys if name in proposed and proposed[name] != now.get( name ) ]

    def changes_by_cause( self, current, proposed, backend_name ):
        lines = self.change_lines( current, proposed, backend_name )
        return { self.name: "; ".join( lines ) } if lines else {}

    def applied( self, side, backend_name, tracking ):
        return globals_in_effect( side ) if side else contextlib.nullcontext()

    def attribute( self, boxes, other_observer, changes ):
        return [ self.name if self.name in changes else TRACKING_CAUSE for _box in boxes ]

    def choose( self, current, proposed, tally, backend_name ):
        if not self.changes_by_cause( current, proposed, backend_name ):
            return {}, {}
        decision = _decision( tally.get( self.name, {} ) )
        return ( dict( proposed ) if decision == 'proposed' else {} ), { self.name: decision }

    def chosen_lines( self, chosen, current, backend_name ):
        if not chosen:
            return [ "(nothing: the verdicts kept the config as it is)" ]
        return [ "%s = %s"%( name, _format( chosen[name] ) ) for name in self.keys if name in chosen ]


# --- registry ---------------------------------------------------------------------------

def experiments():
    """Every experiment, in menu order."""
    return [
        SuppressionExperiment(),
        OverrideExperiment( 'min_track_hits', 'min_track_hits', 'min_track_hits',
                            "detections a track needs before it is drawn (auto_tune's value vs the config)" ),
        OverrideExperiment( 'confirm_score', 'confirm_score', 'confirm score',
                            "draw a track short of min_track_hits once one detection scores this "
                            "much (auto_tune's value, else %g)"%( bu_tuning.CONFIDENT_SCORE, ),
                            default_value=bu_tuning.CONFIDENT_SCORE ),
        OverrideExperiment( 'match_distance', 'match_distance_multiplier', 'match distance',
                            "how far a detection may jump and still continue a track "
                            "(auto_tune's tightening candidate vs the config)" ),
        GlobalsExperiment( 'cut_bursts', CUT_BURST_KEYS, CUT_BURST_DEFAULT, CUT_BURST_ABSENT,
                           'shot-cut burst merge',
                           "collapse bursts of shot cuts (fades, flashes) to one cut before tracking" ),
        OverrideExperiment( 'tracking', None, 'any per-label setting',
                            "any item override, named with --set label.key=value" ),
    ]


def experiment_named( name ):
    for experiment in experiments():
        if experiment.name == ( name or 'suppression' ):
            return experiment
    raise SystemExit( "no experiment %r; choose from %s"%(
        name, ", ".join( experiment.name for experiment in experiments() ) ) )
