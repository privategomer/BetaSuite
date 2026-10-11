#!/usr/bin/env python3
"""
analyze_cut_bursts.py - would merging bursts of shot cuts recover censoring?

A fade, dissolve or white flash changes the picture over several
consecutive samples, and the shot-cut scan flags a cut at each step, so
one transition arrives as a burst: 10.0, 10.1, 10.2, 10.3s. Every cut
ends every live track. Each sliver between two cuts is too short for a
new track to collect min_track_hits detections, so nothing in the burst
is drawn, and the shot after it waits for its own confirmation. The
censor can blink off for half a second around every such transition.

Merging a burst into one cut (betautils_video.merge_cut_bursts) means one
restart per transition instead of four. The cost is on genuine rapid-fire
editing, where cuts really are 0.2s apart: merged, a track could carry
across a real cut onto a different person in the same part of the frame.
max_span limits merging to short bursts for exactly that reason.

WHAT THIS DOES
  1. Census: per file, how many cuts sit in bursts, and per profile.
  2. Replays every file's cached detections through the production
     pipeline, under its own profile, once with the cuts as scanned and
     once per merge variant. Only the cuts tracking stops at change; the
     profile each file gets is still chosen from the raw cuts.
  3. Per profile and in total: confident detections (score >= 0.5) left
     uncensored, rendered boxes, unconfirmed tracks dropped, track resets
     and risky assignments, against the unmerged baseline.

A variant helps when it leaves fewer confident detections uncensored
without paying more than --max-risky-per-restored risky assignments for
each one it recovers. Every variant is measured against the cuts as
scanned, so the numbers mean the same whether or not merging is on in
betaconfig (shot_cut_burst_window and friends). Nothing here changes a
real run; the best variant is printed as the config lines that would.

    python3 tools/analysis/analyze_cut_bursts.py
    python3 tools/analysis/analyze_cut_bursts.py --windows 0.25 0.4 --max-spans 0.6 1.0 none
"""

import argparse
import functools
import json
import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname(
    os.path.abspath( __file__ ) ) ) ) )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_track as bu_track
import betautils_tuning as bu_tuning
import betautils_video as bu_video


class Variant( object ):
    """One way of merging bursts: window, max_span, keep."""

    def __init__( self, window, max_span, keep ):
        self.window = window
        self.max_span = max_span
        self.keep = keep

    @property
    def label( self ):
        return 'window %gs, span %s, keep %s'%(
            self.window, 'any' if self.max_span is None else '%gs'%( self.max_span, ), self.keep )

    def transform( self ):
        return functools.partial( bu_video.merge_cut_bursts, window=self.window,
                                  max_span=self.max_span, keep=self.keep )

    def settings( self ):
        """The same merge as bu_track.tracking_cut_times takes it, and as the config spells it."""
        return { 'window': self.window, 'max_span': self.max_span, 'keep': self.keep }


def config_lines( variant ):
    """The betaconfig lines that turn `variant` on."""
    return "shot_cut_burst_window = %g; shot_cut_burst_max_span = %s; shot_cut_burst_keep = '%s'"%(
        variant.window, 'None' if variant.max_span is None else '%g'%( variant.max_span, ), variant.keep )


def describe_configured():
    """What the config does now, in this tool's variant words."""
    settings = bu_track.cut_burst_settings()
    if not settings:
        return "off (shot_cut_burst_window = 0)"
    return Variant( settings['window'], settings['max_span'], settings['keep'] ).label


def parse_span( text ):
    return None if str( text ).lower() in ( 'none', 'any', 'off' ) else float( text )


def variants_from( windows, spans, keeps ):
    return [ Variant( window, span, keep ) for window in windows for span in spans for keep in keeps ]


def burst_census( entries, window ):
    """
    Per-file burst counts, from each file's cached cuts.

    entries is [(file_hash, video_path, cache_paths, profile, cuts)].
    Returns a list of row dicts, worst first.
    """
    rows = []
    for file_hash, video_path, _cache_paths, profile, cuts in entries:
        bursts = bu_video.cut_bursts( cuts or [], window )
        rows.append( {
            'file': os.path.basename( video_path ), 'file_hash': file_hash, 'profile': profile,
            'cuts': len( cuts or [] ), 'bursts': len( bursts ),
            'cuts_in_bursts': sum( len( burst ) for burst in bursts ),
            'longest_burst_seconds': max( ( burst[-1] - burst[0] for burst in bursts ), default=0.0 ),
            'scanned': cuts is not None,
        } )
    rows.sort( key=lambda row: ( -row['cuts_in_bursts'], row['file'] ) )
    return rows


def _confident_missed( metrics ):
    return sum( metrics.get( 'confident_uncovered_by_label', {} ).values() )


def compare( baseline, candidate ):
    """Candidate minus baseline, for the numbers a merge decision rests on."""
    restored = _confident_missed( baseline ) - _confident_missed( candidate )
    risky_added = candidate['risky_contention'] - baseline['risky_contention']
    return {
        'confident_restored': restored,
        'confident_total': sum( baseline.get( 'confident_by_label', {} ).values() ),
        'rendered_boxes': candidate['rendered_boxes'] - baseline['rendered_boxes'],
        'dropped_unconfirmed': candidate['dropped_unconfirmed'] - baseline['dropped_unconfirmed'],
        'resets': candidate['new_tracks'] - baseline['new_tracks'],
        'risky_added': risky_added,
        'worst_coverage_loss': bu_tuning.worst_coverage_loss( baseline, candidate ),
    }


def verdict( change, max_risky_per_restored ):
    """(helps, text) for one variant's change against the baseline."""
    restored = change['confident_restored']
    if restored < 0:
        return False, 'worse: leaves %d more confident detection(s) uncensored'%( -restored, )
    if restored == 0:
        return False, 'no confident detection recovered'
    price = max( 0, change['risky_added'] ) / float( restored )
    if price > max_risky_per_restored:
        return False, '%d recovered at %.1f risky each (limit %.1f)'%(
            restored, price, max_risky_per_restored )
    return True, '%d recovered at %.1f risky each'%( restored, price )


def compare_variants( entries, backend_name, labels, variants, seed, confident_score, measure=None ):
    """
    {profile: {'baseline': metrics, variant.label: change}} plus a 'total' entry.

    Each profile's files are replayed together, so the per-profile split
    costs nothing extra over one pass.
    """
    measure = measure or bu_tuning.measure_replay
    groups = {}
    for file_hash, video_path, cache_paths, profile, _cuts in entries:
        group = groups.setdefault( profile or '(none)', ( {}, {} ) )
        group[0][file_hash] = cache_paths
        group[1][file_hash] = video_path
    results = {}
    totals = {}
    for profile, ( videos, hash_to_path ) in sorted( groups.items() ):
        # The cuts as scanned, whatever the config says: every variant,
        # the configured one included, is measured against no merging.
        baseline = measure( videos, hash_to_path, backend_name, labels, seed=seed,
                            confident_score=confident_score, cut_bursts=False )
        results[profile] = { 'files': len( videos ) }
        for variant in variants:
            candidate = measure( videos, hash_to_path, backend_name, labels, seed=seed,
                                 confident_score=confident_score,
                                 cut_bursts=variant.settings() )
            change = compare( baseline, candidate )
            results[profile][variant.label] = change
            total = totals.setdefault( variant.label, dict.fromkeys( change, 0 ) )
            for key, value in change.items():
                total[key] = ( max( total[key], value ) if key == 'worst_coverage_loss'
                               else total[key] + value )
    results['total'] = totals
    return results


def best_variant( totals, max_risky_per_restored ):
    """The variant recovering the most confident detections among those that help, or None."""
    helping = [ ( label, change ) for label, change in totals.items()
                if verdict( change, max_risky_per_restored )[0] ]
    if not helping:
        return None
    return max( helping, key=lambda item: ( item[1]['confident_restored'], -item[1]['risky_added'] ) )[0]


def _print_census( rows, window, top ):
    scanned = [ row for row in rows if row['scanned'] ]
    print( "=== cut bursts (cuts less than %gs apart) ==="%( window, ) )
    if not scanned:
        print( "  no shot-cut caches: run betatv.py (or betabench structure-scan) first" )
        return
    by_profile = {}
    for row in scanned:
        entry = by_profile.setdefault( row['profile'] or '(none)', [ 0, 0, 0, 0 ] )
        entry[0] += 1
        entry[1] += row['cuts']
        entry[2] += row['bursts']
        entry[3] += row['cuts_in_bursts']
    print( "  %-14s %5s %8s %8s %14s"%( 'profile', 'files', 'cuts', 'bursts', 'cuts in bursts' ) )
    for profile, ( files, cuts, bursts, in_bursts ) in sorted( by_profile.items() ):
        print( "  %-14s %5d %8d %8d %8d (%3.0f%%)"%(
            profile, files, cuts, bursts, in_bursts, 100.0 * in_bursts / cuts if cuts else 0.0 ) )
    print( "" )
    print( "  files with the most cuts in bursts:" )
    for row in scanned[ :top ]:
        print( "    %4d of %4d cuts in %3d burst(s), longest %.2fs  %-12s %s"%(
            row['cuts_in_bursts'], row['cuts'], row['bursts'], row['longest_burst_seconds'],
            row['profile'] or '', row['file'][ :60 ] ) )
    print( "" )


def _print_comparison( results, variants, max_risky_per_restored ):
    print( "=== merged vs as scanned (replayed through the production pipeline) ===" )
    print( "  confident = detections scoring >= the confident score; 'restored' is how many fewer" )
    print( "  of them a variant leaves uncensored" )
    for profile in sorted( key for key in results if key != 'total' ) + [ 'total' ]:
        block = results[profile]
        header = profile if profile == 'total' else '%s (%d file(s))'%( profile, block['files'] )
        print( "" )
        print( "  %s"%( header, ) )
        print( "    %-34s %9s %9s %10s %8s %8s %9s  %s"%(
            'variant', 'restored', 'boxes', 'unconfirm', 'resets', 'risky', 'coverage', 'verdict' ) )
        for variant in variants:
            change = block.get( variant.label )
            if change is None:
                continue
            _helps, text = verdict( change, max_risky_per_restored )
            print( "    %-34s %+9d %+9d %+10d %+8d %+8d %8.2f%%  %s"%(
                variant.label, change['confident_restored'], change['rendered_boxes'],
                change['dropped_unconfirmed'], change['resets'], change['risky_added'],
                -100.0 * change['worst_coverage_loss'], text ) )


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--windows', type=float, nargs='+', default=[ 0.35 ],
        help="largest gap, in seconds, that still joins two cuts into a burst (default: 0.35)" )
    parser.add_argument( '--max-spans', nargs='+', default=[ '0.6', 'none' ],
        help="longest burst merged, in seconds, or 'none' for any length (default: 0.6 none)" )
    parser.add_argument( '--keep', nargs='+', choices=( 'last', 'first' ),
                         default=[ 'last', 'first' ],
        help="which cut of a burst survives (default: both, compared)" )
    parser.add_argument( '--backends', nargs='+', default=None,
        help="backends to analyse (default: every one with caches)" )
    parser.add_argument( '--variants', nargs='+', default=None, dest='model_variants',
        help="model variants to analyse (default: every one with caches)" )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="censored labels to measure (default: items_to_censor)" )
    parser.add_argument( '--confident-score', type=float, default=bu_tuning.CONFIDENT_SCORE,
        help="a detection at or above this score counts as probably real (default: %(default)s)" )
    parser.add_argument( '--max-risky-per-restored', type=float, default=5.0,
        help="risky assignments a variant may add per confident detection it recovers "
             "(default: %(default)s)" )
    parser.add_argument( '--seed', type=int, default=bu_tuning.REPLAY_SEED,
        help="replay RNG seed (default: %(default)s)" )
    parser.add_argument( '--top', type=int, default=10,
        help="files listed in the census (default: %(default)s)" )
    parser.add_argument( '--json', default=None, metavar='FILE',
        help="also write the comparison and the best variant's settings here; "
             "ab_compare.sh's cut_bursts experiment proposes them" )
    return parser


def main( argv=None ):
    args = build_parser().parse_args( argv )
    variants = variants_from( args.windows, [ parse_span( span ) for span in args.max_spans ],
                              args.keep )
    labels = args.labels or sorted( betaconfig.items_to_censor )
    min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    configurations = bu_cache.configurations_to_analyse(
        None, backend_names=args.backends, variant_names=args.model_variants,
        min_prob=min_prob, include_preview=False )
    if not configurations:
        print( "no detection caches found: run betatv.py on real footage first" )
        return 1
    report = { 'configurations': {}, 'best_settings': None }
    for config in configurations:
        print( "" )
        print( "#" * 72 )
        print( "# %s"%( config.label, ) )
        print( "#" * 72 )
        # Every sample rate: quick-cut videos are detected at their
        # profile's rate, and they are where cut bursts matter most.
        videos, rates = bu_cache.videos_across_sample_rates( config, min_prob )
        print( "# %d video(s), sample rates %s"%(
            len( videos ), ", ".join( '%g'%( rate, ) for rate in rates ) or 'none' ) )
        print( "# burst merging in config now: %s"%( describe_configured(), ) )
        hash_to_path = bu_cache.build_hash_to_video_path( videos.keys() )
        entries = []
        for file_hash, cache_paths in sorted( videos.items() ):
            video_path = hash_to_path.get( file_hash )
            if not video_path:
                continue
            context = bu_tuning.replay_tracking_context( file_hash, video_path, cache_paths,
                                                         config.backend_name )
            entries.append( ( file_hash, video_path, cache_paths, context['profile_name'],
                              context['shot_cut_times'] ) )
        if not entries:
            print( "  none of this configuration's source videos could be found" )
            continue
        _print_census( burst_census( entries, min( args.windows ) ), min( args.windows ), args.top )
        results = compare_variants( entries, config.backend_name, labels, variants, args.seed,
                                    args.confident_score )
        report['configurations'][config.label] = results
        _print_comparison( results, variants, args.max_risky_per_restored )
        best = best_variant( results['total'], args.max_risky_per_restored )
        print( "" )
        if best:
            print( "  best: %s. Check its per-profile rows before turning it on: a quick_cut or "
                   "split_screen row that went worse is where merging hurts."%( best, ) )
            chosen = [ variant for variant in variants if variant.label == best ][0]
            print( "  as config: %s"%( config_lines( chosen ), ) )
            if report['best_settings'] is None:
                report['best_settings'] = chosen.settings()
            print( "  to see it before deciding: AB_EXPERIMENT=cut_bursts bash tools/tuning/ab_compare.sh" )
        else:
            print( "  no variant recovered confident detections at an acceptable price; merging "
                   "would not help this footage." )
    if args.json:
        with open( args.json, 'w', encoding='UTF-8' ) as handle:
            json.dump( report, handle, indent=2, default=str )
        print( "" )
        print( "comparison and best settings: %s"%( args.json, ) )
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
