#!/usr/bin/env python3
"""
auto_tune.py - derive config values from your own footage, one at a time.

One command. It measures, decides, writes the decision into
betaconfig.py, and moves to the next question. Nothing to remember, no
per-model or per-variant invocations to get right, and a full log of why
every value was chosen.

    python3 tools/tuning/auto_tune.py                # measure and report only
    python3 tools/tuning/auto_tune.py --apply        # also write the winners
    python3 tools/tuning/auto_tune.py --apply --yes  # unattended
    python3 tools/tuning/auto_tune.py --stages track_max_gap match_distance
    python3 tools/tuning/auto_tune.py --list-stages

HOW IT WORKS

Each STAGE owns one question, a set of candidate values, and a decision
rule. Stages run in order; an accepted value is written before the next
stage starts, so later stages measure against earlier decisions rather
than against the original config. Tuning several interacting knobs at
once tells you about the combination, never about a single knob.

Every candidate is scored by REPLAYING the detection caches your last
real run already wrote, through the real betautils_track pipeline. No
model runs, so a full sweep is seconds rather than hours, and it covers
every model configuration on disk - each backend, each variant - not
just whichever one betaconfig currently selects.

THE RULE NO STAGE CAN BREAK

A candidate that reduces censor coverage - total censored area-seconds,
per label, over the same footage - by more than --max-coverage-loss is
rejected, whatever else it improves. Performance is never bought with
censoring. The check is in betautils_tuning.worst_coverage_loss and it
runs on every candidate of every stage, including ones whose whole
purpose is to remove boxes.

WHAT IT WILL NOT DO

Anything that changes the raw detections - picture size, model variant,
global_min_prob, NMS settings - cannot be replayed from cache, because
the cache IS the detections. Those need a real run. This tool refuses
them rather than approximating, and tells you which command to run.

WHERE IT STOPS

When every stage has either accepted a value or reported that no
candidate beat the baseline, the numbers have said what they can. What
is left is visual: whether the censoring looks right. The final summary
says exactly that, and what to watch for.
"""

import argparse
import copy
import glob
import json
import logging
import os
import statistics
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname(
    os.path.abspath( __file__ ) ) ) ) )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_tuning as bu_tuning

TRACE = 5
LEVEL_NAMES = {
    'trace': TRACE, 'debug': logging.DEBUG, 'info': logging.INFO,
    'warn': logging.WARNING, 'error': logging.ERROR,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def build_logger( run_dir, log_level_name, console_level_name ):
    """Everything to run_dir/auto_tune.log; a filtered view to the terminal."""
    logging.addLevelName( TRACE, 'TRACE' )
    logger = logging.getLogger( 'auto_tune' )
    logger.propagate = False
    logger.setLevel( TRACE )
    for handler in list( logger.handlers ):
        logger.removeHandler( handler )

    os.makedirs( run_dir, exist_ok=True )
    file_handler = logging.FileHandler( os.path.join( run_dir, 'auto_tune.log' ), encoding='UTF-8' )
    file_handler.setLevel( LEVEL_NAMES[ log_level_name ] )
    file_handler.setFormatter( logging.Formatter( '%(asctime)s [%(levelname)-5s] %(message)s' ) )
    logger.addHandler( file_handler )

    console_handler = logging.StreamHandler( sys.stdout )
    console_handler.setLevel( LEVEL_NAMES[ console_level_name ] )
    console_handler.setFormatter( logging.Formatter( '%(message)s' ) )
    logger.addHandler( console_handler )
    return logger


def confirm( logger, question, assume_yes ):
    """Ask before acting, unless --yes. Every decision point routes through here."""
    if assume_yes:
        logger.info( "%s [auto-yes]"%(question) )
        return True
    logger.info( question )
    try:
        return input( "  [y/N] " ).strip().lower() in ( 'y', 'yes' )
    except EOFError:
        return False


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

class Stage:
    """
    One tuning question, its candidates, and the rule that picks a winner.

    A stage is deliberately small: one key, one label, one decision. The
    alternative - a stage that tunes several keys together - cannot
    attribute a result to a cause, which is the failure mode this whole
    design exists to avoid.
    """

    name = 'stage'
    key = None
    summary = ''

    def __init__( self, backend_name, label, configuration_label=None ):
        self.backend_name = backend_name
        self.label = label
        # The CONFIGURATION, not just the backend: two variants of one
        # backend produce two stages with the same name and the same
        # key, and a summary that cannot tell them apart is a summary
        # you cannot act on.
        self.configuration_label = configuration_label or backend_name
        # Median shot length for this configuration's footage, in
        # seconds, or None when no shot-cut cache was found. Set by the
        # caller after construction; only TrackMaxGapStage reads it, to
        # cap candidates that could only bridge across a cut.
        self.median_shot_seconds = None

    @property
    def title( self ):
        # Named by BACKEND, not by configuration: this key is shared by
        # every variant of the backend, so one title for one decision.
        return "%s / %s / %s"%( self.name, self.backend_name, self.label )

    def baseline_value( self ):
        return bu_detector.get_item_overrides( self.label, self.backend_name ).get( self.key )

    def candidates( self ):
        """[(value, human_label)], excluding the baseline."""
        raise NotImplementedError

    def apply( self, value ):
        """Set this candidate in memory for the duration of a measurement."""
        block = betaconfig.detector_backend.setdefault( self.backend_name, {} )
        overrides = copy.deepcopy( block.get( 'item_overrides', {} ) )
        overrides[self.label] = dict( overrides.get( self.label, {} ) )
        overrides[self.label][self.key] = value
        block['item_overrides'] = overrides
        bu_config.invalidate_config_caches()

    def restore( self, saved ):
        betaconfig.detector_backend[self.backend_name]['item_overrides'] = saved
        bu_config.invalidate_config_caches()

    def decide( self, rows, options, logger ):
        raise NotImplementedError

    def write( self, writer, value ):
        writer.set_item_override( self.backend_name, self.label, self.key, value )


class LooseningStage( Stage ):
    """
    The shape shared by track_max_gap and match_distance_multiplier.

    Both answer "should this threshold be looser?", both are paid for in
    the same currency, and both have repeatedly wanted opposite answers
    for different labels - which is exactly why they are tuned per label
    rather than globally.

    The decision rule is a PRICE, not a target. Loosening a threshold
    rescues broken tracks (fewer resets, so fewer visible snaps) and
    creates ambiguity (more risky contention, so more chance of a track
    following the wrong subject). The question is never "did resets go
    down" - a large enough threshold always drives them down - but "how
    many risky assignments did each rescued reset cost?". A candidate is
    accepted only when that price is under --max-risky-per-reset, and
    among those, the one that fixes the most resets wins.

    Real numbers from one night's footage, to show why the price matters:
    for exposed_breast, tripling match_distance fixed 130 resets and
    added 8874 risky assignments - 68 apiece. Tripling track_max_gap
    fixed 135 and added 536 - 4 apiece. Same improvement in the headline
    number, seventeen times the cost, and only the price tells them apart.
    """

    def score_rows( self, rows, options ):
        """Fill in each candidate row's verdict for ONE configuration."""
        baseline = rows[0]
        for row in rows[1:]:
            resets_avoided = baseline['metrics']['new_tracks'] - row['metrics']['new_tracks']
            risky_added = row['metrics']['risky_contention'] - baseline['metrics']['risky_contention']
            row['resets_avoided'] = resets_avoided
            row['risky_added'] = risky_added
            row['price'] = ( risky_added / resets_avoided ) if resets_avoided > 0 else None

            if row['coverage_loss'] > options.max_coverage_loss:
                row['eligible'] = False
                row['verdict'] = 'rejected: censor coverage fell %.2f%% (limit %.2f%%)'%(
                    100*row['coverage_loss'], 100*options.max_coverage_loss )
            elif resets_avoided <= 0:
                row['eligible'] = False
                row['verdict'] = 'rejected: no fewer track resets than baseline'
            elif row['price'] is not None and row['price'] > options.max_risky_per_reset:
                row['eligible'] = False
                row['verdict'] = 'rejected: %.1f risky assignments per reset avoided (limit %.1f)'%(
                    row['price'], options.max_risky_per_reset )
            else:
                row['eligible'] = True
                if risky_added < 0:
                    # Strictly better on both axes. Worth saying plainly
                    # rather than printing a negative price, which reads
                    # like an error.
                    row['verdict'] = ( 'eligible: %d resets avoided AND %d fewer risky '
                                       'assignments - better on both axes'%(
                                           resets_avoided, -risky_added ) )
                else:
                    row['verdict'] = 'eligible: %d resets avoided at %.1f risky each'%(
                        resets_avoided, row['price'] or 0.0 )

    def decide( self, rows_by_configuration, options, logger ):
        eligible = _values_eligible_everywhere( rows_by_configuration )
        baseline_value = _baseline_value_of( rows_by_configuration )
        if not eligible:
            # Nothing suits every configuration. Before giving up, ask
            # whether they simply WANT different things - two variants
            # of a backend are two sets of weights, and a value that is
            # right for one can be wrong for the other. When each has
            # its own eligible winner, write them per variant rather
            # than throwing away both.
            per_variant = _per_configuration_winners(
                rows_by_configuration, key=lambda row: row['resets_avoided'] )
            if per_variant:
                return bu_tuning.Decision(
                    self.title, None, True,
                    "the configurations want different values, so each gets its own: %s"%(
                        ", ".join( "%s -> %s"%( name, self._format_value( value ) )
                                   for name, value in sorted( per_variant.items() ) ) ),
                    _flatten_rows( rows_by_configuration ),
                    per_configuration=per_variant )
            return bu_tuning.Decision(
                self.title, baseline_value, False,
                "no candidate was eligible under any configuration this key governs "
                "(%s), within the limits of %.1f risky per reset avoided and %.2f%% coverage "
                "loss; keeping %s"%(
                    ", ".join( sorted( rows_by_configuration ) ),
                    options.max_risky_per_reset, 100*options.max_coverage_loss, baseline_value ),
                _flatten_rows( rows_by_configuration ) )

        totals = {}
        for value in eligible:
            totals[value] = sum( _row_for_value( rows, value )['resets_avoided']
                                 for rows in rows_by_configuration.values() )
        chosen = max( totals, key=lambda value: totals[value] )
        def _describe( name, rows ):
            row = _row_for_value( rows, chosen )
            if row['risky_added'] < 0:
                return "%s: %d resets and %d fewer risky"%(
                    name, row['resets_avoided'], -row['risky_added'] )
            return "%s: %d resets at %.1f risky each"%(
                name, row['resets_avoided'], row['price'] or 0.0 )

        per_config = ", ".join( _describe( name, rows )
                                for name, rows in sorted( rows_by_configuration.items() ) )
        return bu_tuning.Decision(
            self.title, chosen, True,
            "%s is the best value that holds up under every configuration this key governs "
            "(%s), avoiding %d track resets in total"%(
                self._format_value( chosen ), per_config, totals[chosen] ),
            _flatten_rows( rows_by_configuration ) )

    @staticmethod
    def _format_value( value ):
        return '%g'%(value) if isinstance( value, float ) else str( value )


class TrackMaxGapStage( LooseningStage ):
    """
    How long a track stays matchable after its last real detection.

    A CEILING THIS STAGE MUST RESPECT
    ---------------------------------
    The price rule above counts resets as the thing to minimise, and for
    a single continuous scene that is right: a reset there is a censor
    snapping off a subject who never left. On CUT-DENSE footage it is
    exactly backwards, and the metric cannot see the difference.

    A track that outlives a shot cut does not stop censoring - it keeps
    drawing, on whoever now occupies that part of the frame. So censor
    coverage stays high, risky contention stays low, and every candidate
    looks free. Measured on a 30s slice of a 16-woman compilation, all
    from one cache: track_max_gap 27.0s produced 3 tracks and 208
    interpolated boxes; 0.16s produced 31 tracks and 0. The 27.0s value
    came from this stage, scored as 10-11 resets avoided at 0.0 risky
    each. It was not free - it welded 16 women into 3 tracks, so all of
    them wore one censor style, and the EMA smoothed position ACROSS the
    cuts (median frame-to-frame jump 39.5px vs 48.1px at the tight
    value: lower, because it was averaging one woman's position into
    another's).

    Hence the ceiling: a gap longer than a typical shot can only ever
    bridge across a cut, so candidates above the median shot length are
    not offered at all. Where the shot length is unknown (no shot-cut
    cache), nothing is capped and the old behaviour stands - but the
    report says so, because "no cuts found" and "no cuts in this
    footage" are different facts.
    """

    name = 'track_max_gap'
    key = 'track_max_gap'
    summary = "how long a track stays matchable after its last real detection"

    MULTIPLIERS = ( 0.5, 0.75, 1.5, 2.0, 3.0 )

    def candidates( self ):
        baseline = self.baseline_value()
        if not baseline:
            return []
        ceiling = self._shot_length_ceiling()
        rows = []
        for multiplier in self.MULTIPLIERS:
            value = round( baseline * multiplier, 4 )
            if ceiling is not None and value > ceiling:
                continue
            rows.append( ( value, '%gx (%gs)'%( multiplier, round( value, 3 ) ) ) )
        return rows

    def _shot_length_ceiling( self ):
        """
        Median shot length across this configuration's footage, or None.

        None means "not knowable from what is on disk" - no shot-cut
        cache, or too few cuts to form a median - and the caller then
        applies no ceiling rather than guessing one.
        """
        median_shot = getattr( self, 'median_shot_seconds', None )
        if median_shot and median_shot > 0:
            return median_shot
        return None


class MatchDistanceStage( LooseningStage ):
    name = 'match_distance'
    key = 'match_distance_multiplier'
    summary = "how far a detection may be from a track and still continue it"

    VALUES = ( 0.75, 1.25, 1.5, 2.0, 3.0 )

    def candidates( self ):
        baseline = self.baseline_value() or 1.0
        return [ ( value, '%g'%(value) ) for value in self.VALUES
                 if abs( value - baseline ) > 1e-9 ]


class MinTrackHitsStage( Stage ):
    """
    How many real detections a track needs before anything is drawn for it.

    This one runs in the opposite direction to the loosening stages: it
    REMOVES boxes, so the coverage guard is not a formality here, it is
    the whole constraint. A one-frame track is often a false positive
    flickering for a ninth of a second, and dropping those is a real
    quality win - but a one-frame track is also what a genuine, brief
    exposure looks like, and dropping THOSE is a censoring failure.

    So the rule is strict: the highest value whose coverage loss stays
    inside the tolerance, and no value is accepted on the strength of
    its track-count reduction alone.
    """

    name = 'min_track_hits'
    key = 'min_track_hits'
    summary = "how many real detections a track needs before it is drawn"

    VALUES = ( 1, 2, 3 )

    def candidates( self ):
        baseline = self.baseline_value() or 1
        return [ ( value, '%d hit(s)'%(value) ) for value in self.VALUES if value != baseline ]

    def score_rows( self, rows, options ):
        baseline = rows[0]
        for row in rows[1:]:
            dropped = row['metrics']['dropped_unconfirmed'] - baseline['metrics']['dropped_unconfirmed']
            row['dropped_unconfirmed'] = dropped
            row['resets_avoided'] = dropped
            if row['coverage_loss'] > options.max_coverage_loss:
                row['eligible'] = False
                row['verdict'] = 'rejected: censor coverage fell %.2f%% (limit %.2f%%)'%(
                    100*row['coverage_loss'], 100*options.max_coverage_loss )
            elif dropped <= 0:
                row['eligible'] = False
                row['verdict'] = 'rejected: drops no unconfirmed tracks, so it changes nothing'
            else:
                row['eligible'] = True
                row['verdict'] = 'eligible: drops %d box(es) from unconfirmed tracks for %.3f%% coverage'%(
                    dropped, 100*row['coverage_loss'] )

    def decide( self, rows_by_configuration, options, logger ):
        eligible = _values_eligible_everywhere( rows_by_configuration )
        baseline_value = _baseline_value_of( rows_by_configuration )
        if not eligible:
            per_variant = _per_configuration_winners(
                rows_by_configuration, key=lambda row: -row['value'] )
            if per_variant:
                return bu_tuning.Decision(
                    self.title, None, True,
                    "the configurations want different values, so each gets its own: %s"%(
                        ", ".join( "%s -> %s"%( name, value )
                                   for name, value in sorted( per_variant.items() ) ) ),
                    _flatten_rows( rows_by_configuration ),
                    per_configuration=per_variant )
            return bu_tuning.Decision(
                self.title, baseline_value, False,
                "no value stayed inside the %.2f%% coverage tolerance under any "
                "configuration this key governs (%s); keeping %s"%(
                    100*options.max_coverage_loss, ", ".join( sorted( rows_by_configuration ) ),
                    baseline_value ),
                _flatten_rows( rows_by_configuration ) )

        # The STRICTEST eligible value, not the most aggressive: this
        # stage removes boxes, so among values that all passed the
        # coverage guard the conservative choice is the smallest one
        # that still does something.
        chosen = min( eligible )
        dropped_total = sum( _row_for_value( rows, chosen )['dropped_unconfirmed']
                             for rows in rows_by_configuration.values() )
        return bu_tuning.Decision(
            self.title, chosen, True,
            "%s removes %d box(es) in total from tracks that never reached that many real "
            "detections, inside the %.2f%% coverage tolerance under every configuration "
            "this key governs (%s)"%(
                chosen, dropped_total, 100*options.max_coverage_loss,
                ", ".join( sorted( rows_by_configuration ) ) ),
            _flatten_rows( rows_by_configuration ) )


# ---------------------------------------------------------------------------
# Combining evidence across the configurations that share a config key
# ---------------------------------------------------------------------------
#
# detector_backend[<backend>]['item_overrides'] is ONE block. Every
# variant of that backend reads the same track_max_gap. So a tuner that
# decides per configuration and writes after each one does not make two
# decisions - it makes one decision twice, each time compounding on the
# other's write. That is exactly what happened on the first real run:
# the 320n stage raised exposed_vulva's gap from 21.6 to 64.8, and the
# 640m stage then read 64.8 as its baseline and doubled it again to
# 129.6, a value no measurement ever proposed.
#
# A shared key gets a single decision, and a candidate has to hold up
# under every configuration that key governs. Requiring eligibility
# everywhere rather than on average is deliberate: "good for 320n,
# harmful for 640m" is not a value you want written to a setting both
# of them read.

def _per_configuration_winners( rows_by_configuration, key ):
    """
    Each configuration's own best eligible value, when they disagree.

    Only returned when MORE THAN ONE configuration has an eligible
    candidate. A single configuration finding a winner while its sibling
    finds none is not the configurations wanting different things - it
    is one of them having no opinion, and the right answer there is to
    leave the shared value alone rather than to split the setting.

    Args:
        rows_by_configuration: {configuration label: scored rows}.
        key: How to rank one configuration's eligible rows; the maximum wins.

    Returns:
        {configuration label: value}, or {} when splitting is not warranted.
    """
    winners = {}
    for name, rows in rows_by_configuration.items():
        eligible = [ row for row in rows[1:] if row.get( 'eligible' ) ]
        if eligible:
            winners[name] = max( eligible, key=key )['value']
    if len( winners ) < 2 or len( set( winners.values() ) ) < 2:
        return {}
    return winners


def _values_eligible_everywhere( rows_by_configuration ):
    """Candidate values marked eligible in every configuration measured."""
    eligible = None
    for rows in rows_by_configuration.values():
        here = { row['value'] for row in rows[1:] if row.get( 'eligible' ) }
        eligible = here if eligible is None else ( eligible & here )
    return eligible or set()


def _baseline_value_of( rows_by_configuration ):
    """The baseline value, which is shared by definition."""
    for rows in rows_by_configuration.values():
        return rows[0]['value']
    return None


def _row_for_value( rows, value ):
    for row in rows:
        if row['value'] == value:
            return row
    return None


def _flatten_rows( rows_by_configuration ):
    flattened = []
    for name, rows in sorted( rows_by_configuration.items() ):
        for row in rows:
            entry = dict( row )
            entry['configuration'] = name
            entry.pop( 'metrics', None )
            flattened.append( entry )
    return flattened


def _median_shot_seconds_for( entries, logger ):
    """
    Median shot length across the footage these entries cover, or None.

    Read from the shot-cut caches a real run already wrote, so this
    costs no decoding. Returns None when no cache is found or there are
    too few cuts to form a shot length - "cuts were never looked for"
    and "this footage has no cuts" are different facts, and only the
    second would justify an unbounded track_max_gap.
    """
    fps = betaconfig.video_censor_fps
    threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
    shots = []
    seen = set()
    for _config, context in entries:
        # context['videos'] is {file_hash: [cache_path, ...]}.
        for file_hash in ( context or {} ).get( 'videos', {} ):
            if file_hash in seen:
                continue
            seen.add( file_hash )
            # Glob rather than an exact path: the cache key carries a
            # preview suffix this caller does not know.
            prefix = os.path.join( betaconst.shot_cut_dir,
                                   '%s-%g-%.3f'%( file_hash, fps, threshold ) )
            for path in sorted( glob.glob( prefix + '*.gz' ),
                                key=os.path.getmtime, reverse=True )[:1]:
                try:
                    cuts = sorted( float( c ) for c in bu_hash.read_json( path ) )
                except Exception:                         # noqa: BLE001
                    continue
                shots.extend( b - a for a, b in zip( cuts, cuts[1:] ) if b > a )

    if len( shots ) < 2:
        logger.info( "no usable shot-cut cache: track_max_gap candidates are NOT "
                     "capped. If this footage is cut-dense, run betatv.py with "
                     "shot_cut_detection_enabled first so this stage can see the "
                     "shot length." )
        return None

    median_shot = statistics.median( shots )
    logger.info( "median shot length %.2fs across %d shot(s): track_max_gap "
                 "candidates above this are skipped, since a longer gap could "
                 "only bridge a cut.", median_shot, len( shots ) )
    return median_shot


STAGE_TYPES = {
    'track_max_gap': TrackMaxGapStage,
    'match_distance': MatchDistanceStage,
    'min_track_hits': MinTrackHitsStage,
}

# The order matters. track_max_gap first because it is the cheapest
# lever and it changes how many resets the later stages even see;
# match_distance second, judged against whatever gap won; min_track_hits
# last because it removes boxes and should judge the tracks the first
# two stages actually produce.
STAGE_ORDER = ( 'track_max_gap', 'match_distance', 'min_track_hits' )


# ---------------------------------------------------------------------------
# Running one stage
# ---------------------------------------------------------------------------

def measure_stage( stage, context, options, logger ):
    """
    Measure the baseline and every candidate for one stage.

    Returns:
        A list of rows, baseline first, each with value, label, metrics
        and coverage_loss.
    """
    saved = copy.deepcopy(
        betaconfig.detector_backend.get( stage.backend_name, {} ).get( 'item_overrides', {} ) )
    rows = []
    try:
        baseline_value = stage.baseline_value()

        # The baseline replay is reused across stages that share a
        # configuration AND an unchanged item_overrides block.
        #
        # Every stage measures its own baseline first, but a baseline is
        # a function of (configuration, current config state) and nothing
        # else - it does not depend on which knob the stage is about to
        # sweep. With 6 stages over 3 configurations that was 18 baseline
        # replays for 3 distinct results, and on retinanet a replay is
        # ~23s, so the redundant ones cost minutes per run.
        #
        # The cache key includes a digest of the backend's resolved
        # item_overrides, because an accepted value from an earlier stage
        # is written to config before the next stage runs: after such a
        # write the baseline genuinely HAS moved, and a key of
        # configuration alone would serve a stale one and silently
        # compare every later candidate against the wrong reference.
        # Keying on the config state means the cache holds exactly as
        # long as the thing it describes has not changed.
        baseline_key = (
            stage.backend_name,
            context.get( 'configuration_label' ),
            options.replay_seed,
            json.dumps(
                betaconfig.detector_backend.get( stage.backend_name, {} ).get(
                    'item_overrides', {} ),
                sort_keys=True, default=str ),
        )
        baseline_cache = context.setdefault( '_baseline_cache', {} )
        if baseline_key in baseline_cache:
            baseline_metrics = baseline_cache[baseline_key]
            logger.debug( "%s baseline: reusing the measurement from an earlier stage "
                          "(same configuration, config unchanged since)"%(stage.title) )
        else:
            baseline_metrics = bu_tuning.measure_replay(
                context['videos'], context['hash_to_path'], stage.backend_name,
                context['labels'], seed=options.replay_seed )
            baseline_cache[baseline_key] = baseline_metrics
        rows.append( {
            'value': baseline_value,
            'label': 'baseline (%s)'%(baseline_value),
            'metrics': baseline_metrics,
            'coverage_loss': 0.0,
            'verdict': 'baseline',
        } )
        logger.debug( "%s baseline: %s"%( stage.title, _row_summary( rows[0] ) ) )

        for value, label in stage.candidates():
            stage.apply( value )
            metrics = bu_tuning.measure_replay(
                context['videos'], context['hash_to_path'], stage.backend_name,
                context['labels'], seed=options.replay_seed )
            row = {
                'value': value,
                'label': label,
                'metrics': metrics,
                'coverage_loss': bu_tuning.worst_coverage_loss( baseline_metrics, metrics ),
                'verdict': '',
            }
            rows.append( row )
            logger.debug( "%s candidate %s: %s"%( stage.title, label, _row_summary( row ) ) )
    finally:
        stage.restore( saved )
    return rows


def _row_summary( row ):
    metrics = row['metrics']
    return ( "resets=%d risky=%d tracks=%d rendered=%d interpolated=%d coverage_loss=%.4f%%"%(
        metrics['new_tracks'], metrics['risky_contention'], metrics['tracks'],
        metrics['rendered_boxes'], metrics['interpolated'], 100*row['coverage_loss'] ) )


def print_stage_table( configuration_label, rows, logger ):
    logger.info( "" )
    logger.info( "  %s"%(configuration_label) )
    logger.info( "  %-22s %9s %9s %9s %11s   %s"%(
        'candidate', 'resets', 'risky', 'rendered', 'coverage', 'verdict' ) )
    for row in rows:
        metrics = row['metrics']
        logger.info( "  %-22s %9d %9d %9d %10.3f%%   %s"%(
            row['label'][:22], metrics['new_tracks'], metrics['risky_contention'],
            metrics['rendered_boxes'], -100*row['coverage_loss'], row['verdict'] ) )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--apply', action='store_true',
        help="write accepted values into betaconfig.py. Without this the run measures and "
             "reports and changes nothing, which is the right way to see what it would do" )
    parser.add_argument( '--yes', action='store_true',
        help="accept every prompt - what an unattended run wants" )
    parser.add_argument( '--stages', nargs='+', default=None,
        choices=sorted( STAGE_TYPES ),
        help="run only these stages (default: all, in dependency order)" )
    parser.add_argument( '--list-stages', action='store_true',
        help="print the stages and what each one decides, then exit" )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="tune only these labels (default: everything in items_to_censor)" )
    parser.add_argument( '--backends', nargs='+', default=None,
        help="tune only these backends (default: every one with caches on disk)" )
    # Every other tool in tools/analysis and tools/bench takes this, and
    # run_all_analysis.sh passes it to all of them uniformly. auto_tune
    # not accepting it made the whole harness exit 2 on any run that
    # narrowed to a variant - the tuning step died on argument parsing
    # before it read a single cache.
    parser.add_argument( '--variants', nargs='+', default=None,
                         help='only analyse these model variants (e.g. 640m). '
                              'Default: every variant with caches on disk.' )
    parser.add_argument( '--picture-sizes', type=int, nargs='+', default=None,
        help="analyse exactly these sizes instead of discovering what is on disk" )
    parser.add_argument( '--max-coverage-loss', type=float, default=0.005,
        help="the hard limit on how much censored area-seconds any candidate may cost, as a "
             "fraction. A candidate over this is rejected whatever else it improves "
             "(default: 0.005, i.e. half a percent)" )
    parser.add_argument( '--max-risky-per-reset', type=float, default=5.0,
        help="how many risky track assignments a loosening change may add per track reset it "
             "avoids, before it stops being worth it (default: 5)" )
    parser.add_argument( '--replay-seed', type=int, default=bu_tuning.REPLAY_SEED,
        help="the RNG seed every replay starts from. censor_style is drawn at random per box "
             "and each style carries its own area safety, so without a fixed seed two replays "
             "of the same configuration disagree on coverage by tens of percent. Change it to "
             "check a result is not an artefact of one draw; never vary it inside a comparison "
             "(default: %d)"%(bu_tuning.REPLAY_SEED) )
    parser.add_argument( '--include-preview', action='store_true',
        help="also tune against preview-slice caches when no real run exists" )
    parser.add_argument( '--out-dir', default=None,
        help="where this run's log and results go (default: ../output/tuning/<timestamp>)" )
    parser.add_argument( '--log-level', choices=sorted( LEVEL_NAMES ), default='debug',
        help="level for the LOG FILE (default: debug)" )
    parser.add_argument( '--console-level', choices=sorted( LEVEL_NAMES ), default='info',
        help="level for the TERMINAL (default: info)" )
    return parser


def main():
    args = build_parser().parse_args()

    if args.list_stages:
        print( "stages, in the order they run:\n" )
        for name in STAGE_ORDER:
            stage_type = STAGE_TYPES[name]
            print( "  %-16s %s"%( name, stage_type.summary ) )
        print( "\nEach runs per backend and per censored label. Use --stages to narrow." )
        return 0

    run_dir = args.out_dir or os.path.join(
        '..', 'output', 'tuning', time.strftime( '%Y%m%d_%H%M%S' ) )
    logger = build_logger( run_dir, args.log_level, args.console_level )

    logger.info( "auto_tune starting" )
    logger.info( "output directory: %s"%( os.path.abspath( run_dir ) ) )
    logger.info( "mode: %s"%( "APPLY - accepted values will be written to betaconfig.py"
                              if args.apply else "measure and report only (pass --apply to write)" ) )
    logger.info( "safety: reject any candidate costing more than %.2f%% censor coverage"%(
        100*args.max_coverage_loss ) )

    labels = args.labels or sorted( betaconfig.items_to_censor )
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )

    configurations = bu_cache.configurations_to_analyse(
        args.picture_sizes, backend_names=args.backends,
        min_prob=global_min_prob, include_preview=args.include_preview,
        variant_names=args.variants, logger=logger )
    logger.info( "configurations: %s"%( bu_cache.describe_configurations( configurations ) ) )

    writer = bu_tuning.ConfigWriter(
        config_path=os.path.join( os.path.dirname( os.path.dirname( os.path.dirname(
            os.path.abspath( __file__ ) ) ) ), 'betaconfig.py' ),
        backup_dir=run_dir, logger=logger )

    stage_names = args.stages or list( STAGE_ORDER )
    decisions = []

    # Gather each configuration's replay inputs once, grouped by the
    # BACKEND whose config block they share. The nesting is
    # backend -> stage -> label -> configuration, so every measurement
    # for one config key happens before that key is decided, and the
    # key is written exactly once.
    contexts_by_backend = {}
    for config in configurations:
        videos, _preview_used = bu_cache.discover_full_run_videos(
            config.picture_sizes, betaconfig.video_censor_fps, global_min_prob,
            config.backend_name, include_preview=args.include_preview )
        if not videos:
            logger.warning( "no caches for %s - skipping"%(config.label) )
            continue
        hash_to_path = bu_cache.build_hash_to_video_path( videos.keys() )
        if not hash_to_path:
            logger.warning( "none of %s's source videos could be found - skipping "
                            "(box geometry needs each video's real frame size)"%(config.label) )
            continue
        contexts_by_backend.setdefault( config.backend_name, [] ).append( (
            config, { 'videos': videos, 'hash_to_path': hash_to_path, 'labels': labels,
                      'configuration_label': config.label,
                      # Shared by reference across every stage that runs
                      # against this configuration, so measure_stage's
                      # baseline cache survives the per-stage dict() copy
                      # below. A fresh dict here would empty the cache on
                      # every stage and defeat the point.
                      '_baseline_cache': {} } ) )

    measured_anything = bool( contexts_by_backend )

    for backend_name in sorted( contexts_by_backend ):
        entries = contexts_by_backend[backend_name]
        logger.info( "" )
        logger.info( "=" * 78 )
        logger.info( "  %s   (%d configuration(s): %s)"%(
            backend_name, len( entries ),
            ", ".join( config.label for config, _context in entries ) ) )
        logger.info( "=" * 78 )
        logger.info( "  These share one detector_backend[%r]['item_overrides'] block, so each"%(backend_name) )
        logger.info( "  key below gets ONE decision, measured against all of them." )

        # Median shot length for this backend's footage, used only to cap
        # track_max_gap candidates that could not do anything except
        # bridge across a shot cut. None when no shot-cut cache exists,
        # and then no cap is applied - see TrackMaxGapStage.
        median_shot = _median_shot_seconds_for( entries, logger )

        for stage_name in stage_names:
            for label in labels:
                stages = [ ( config, STAGE_TYPES[stage_name]( backend_name, label, config.label ) )
                           for config, _context in entries ]
                for _config, stage in stages:
                    stage.median_shot_seconds = median_shot
                probe = stages[0][1]
                if probe.baseline_value() is None and stage_name == 'track_max_gap':
                    logger.debug( "%s/%s: no explicit baseline, skipping"%(stage_name, label) )
                    continue
                if not probe.candidates():
                    logger.debug( "%s/%s: no candidates to try"%(stage_name, label) )
                    continue

                logger.info( "" )
                logger.info( "-- %s / %s / %s  (%s)"%(
                    stage_name, backend_name, label, probe.summary ) )

                rows_by_configuration = {}
                for ( config, _context ), ( _same_config, stage ) in zip( entries, stages ):
                    context = dict( _context )
                    rows = measure_stage( stage, context, args, logger )
                    stage.score_rows( rows, args )
                    rows_by_configuration[config.label] = rows
                    print_stage_table( config.label, rows, logger )

                decision = probe.decide( rows_by_configuration, args, logger )
                logger.info( "  => %s"%( decision.reasoning ) )
                decisions.append( decision )

                if decision.accepted and args.apply:
                    variant_of = { config.label: config.variant for config, _c in entries }
                    if decision.is_split:
                        question = "  write %s per variant (%s) to betaconfig.py?"%(
                            probe.key,
                            ", ".join( "%s=%s"%( name, value )
                                       for name, value in sorted( decision.per_configuration.items() ) ) )
                    else:
                        question = "  write %s = %s to betaconfig.py (governs %s)?"%(
                            probe.key, decision.chosen,
                            ", ".join( config.label for config, _c in entries ) )
                    if confirm( logger, question, args.yes ):
                        try:
                            if decision.is_split:
                                for name, value in sorted( decision.per_configuration.items() ):
                                    variant = variant_of.get( name )
                                    if not variant:
                                        raise bu_tuning.ConfigWriteError(
                                            "%s has no variant name, so a per-variant value "
                                            "cannot be written for it"%(name) )
                                    writer.set_variant_item_override(
                                        backend_name, variant, label, probe.key, value )
                            else:
                                probe.write( writer, decision.chosen )
                        except bu_tuning.ConfigWriteError as err:
                            logger.error( "  could not write it: %s"%(err) )
                            decision.accepted = False
                            decision.reasoning += " (NOT written: %s)"%(err)
                    else:
                        decision.accepted = False
                        decision.reasoning += " (declined at the prompt)"

    if not measured_anything:
        logger.error( "nothing to tune: no detection caches found for any configuration. Run "
                      "betatv.py on real footage first, then re-run this." )
        return 1

    results_path = os.path.join( run_dir, 'decisions.json' )
    with open( results_path, 'w', encoding='UTF-8' ) as fout:
        json.dump( [ decision.as_dict() for decision in decisions ], fout, indent=2, default=str )

    accepted = [ decision for decision in decisions if decision.accepted ]
    logger.info( "" )
    logger.info( "=" * 78 )
    logger.info( "  SUMMARY" )
    logger.info( "=" * 78 )
    logger.info( "%d stage(s) measured, %d value(s) %s"%(
        len( decisions ), len( accepted ), "written" if args.apply else "would be written" ) )
    for decision in decisions:
        if not decision.accepted:
            outcome = "unchanged"
        elif decision.is_split:
            outcome = "-> per variant: %s"%(
                ", ".join( "%s=%s"%( name.split( '/' )[-1], value )
                           for name, value in sorted( decision.per_configuration.items() ) ) )
        else:
            outcome = "-> %s"%(decision.chosen)
        logger.info( "  %-52s %s"%( decision.stage_name, outcome ) )
    logger.info( "" )
    logger.info( "decisions written to %s"%( os.path.abspath( results_path ) ) )
    if writer.backups:
        logger.info( "config backups: %s"%( ", ".join( os.path.basename( b ) for b in writer.backups ) ) )

    logger.info( "" )
    if accepted and args.apply:
        logger.info( "The detection caches are untouched, so a re-run will not re-detect - but every" )
        logger.info( "accepted value changes the censor key, so rendered output will be rebuilt." )
    logger.info( "What the numbers cannot decide from here: whether the censoring LOOKS right." )
    logger.info( "Render a couple of files and watch for the two things this tool is blind to -" )
    logger.info( "a censor box that lags or overshoots the subject it is following, and a box that" )
    logger.info( "follows the wrong subject after two people cross. Both show up as good numbers." )
    return 0


if __name__ == '__main__':
    try:
        sys.exit( main() )
    except KeyboardInterrupt:
        print( "\ninterrupted - config was left as it was found at the last completed stage" )
        sys.exit( 130 )
