#!/usr/bin/env python3
"""
analyze_person_windows.py - how often reverse mode's person windows find a man.

Reverse mode leaves male bodies uncensored by attributing each person
outline from the person model to a man or a woman, using the detections
inside it (face_masc, exposed_chest, exposed_penis against face_femme,
exposed_breast, exposed_vulva). Someone with no such detection inside
their outline stays under the frame style. This reads the person-mask
caches a reverse run wrote, replays that attribution exactly as the
render does (betautils_person.male_people_by_time), and reports:

  - how many sampled frames have anyone in them, and in how many a man
    is shown
  - every person outline by how it was attributed: shown as a man,
    hidden by female evidence, hidden with no evidence at all, mixed
    votes, or hidden by the merged-outline backstop
  - where each evidence detection's centre fell: inside one outline
    (it votes), inside two or more (it can't say whose, so it votes for
    neither), or outside every outline (the person model missed that
    body, or the box's centre is off it)
  - per video, the ones with the most unattributed person-seconds: where
    to look at a reverse preview by eye

What the numbers mean for the next step:

  - many outlines with no evidence, while male evidence mostly lands
    inside one outline: attribution is limited by how often a male
    label is detected, not by the outlines. A person model that labels
    men and women itself (an open-vocabulary one such as YOLOE) is the
    lever
  - much male evidence outside every outline: the person model misses
    bodies the detector sees. Try a larger person model size first
  - much evidence in shared outlines: overlapping people cost votes

Only videos run with reverse mode and person windows on have person
caches; with none, this says so and exits cleanly.

    python3 tools/analysis/analyze_person_windows.py
    python3 tools/analysis/analyze_person_windows.py --top 15 --json person_windows.json
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname(
    os.path.abspath( __file__ ) ) ) ) )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_person as bu_person
import betautils_reverse as bu_reverse
import betautils_tuning as bu_tuning


PERSON_CACHE = re.compile( r'^(?P<hash>[0-9a-f]+)-person-(?P<backend>[A-Za-z0-9_]+)-(?P<fps>[0-9.]+)'
                           r'-p(?P<key>[0-9a-f]+)(?P<suffix>(-preview(@[0-9.]+s)?)?)\.npz$' )

OUTLINE_KINDS = ( ( 'shown', 'shown as a man' ),
                  ( 'female', 'hidden: female evidence' ),
                  ( 'no_evidence', 'hidden: no evidence at all' ),
                  ( 'mixed', 'hidden: votes both ways' ),
                  ( 'backstop_hidden', 'hidden: merged outline (backstop)' ) )


def parse_person_cache_name( name ):
    """{'hash', 'backend', 'fps', 'key', 'suffix'} for a person cache filename, or None."""
    match = PERSON_CACHE.match( name )
    if not match:
        return None
    parsed = match.groupdict()
    parsed['fps'] = float( parsed['fps'] )
    return parsed


def person_caches( directory ):
    """Every finished person cache in a directory, as (path, parsed name), by name."""
    found = []
    for path in sorted( glob.glob( os.path.join( directory, '*-person-*.npz' ) ) ):
        parsed = parse_person_cache_name( os.path.basename( path ) )
        if parsed is not None:
            found.append( ( path, parsed ) )
    return found


def model_label( backend, key ):
    """'yolo11_seg s' from the person run-key manifest, else the backend and key."""
    try:
        payload = bu_hash.read_json_plain( os.path.join( betaconst.run_key_dir, 'person-%s.json'%( key, ) ) )
        variant = payload.get( 'identity', {} ).get( 'model_variant' )
    except Exception:
        variant = None
    return '%s %s'%( backend, variant ) if variant else '%s p%s'%( backend, key )


def shares( tally ):
    """The tally's outline split, with 'shown' and 'backstop_hidden' taken out of 'male'."""
    return {
        'shown': tally['outlines_shown'],
        'female': tally['outlines_female'],
        'no_evidence': tally['outlines_no_evidence'],
        'mixed': tally['outlines_mixed'],
        'backstop_hidden': tally['outlines_backstop_hidden'],
    }


def video_row( name, tally, interval ):
    split = shares( tally )
    outlines = tally['outlines']
    return {
        'name': name,
        'samples': tally['samples'],
        'seconds': round( tally['samples'] * interval, 1 ),
        'samples_with_people': tally['samples_with_people'],
        'samples_with_man_shown': tally['samples_with_man_shown'],
        'outlines': outlines,
        'outlines_by_attribution': split,
        'unattributed_person_seconds': round( split['no_evidence'] * interval, 1 ),
        'male_evidence_outside': tally['evidence_male_outside'],
        'male_evidence': sum( tally['evidence_male_' + where] for where in ( 'one', 'shared', 'outside' ) ),
        'tally': dict( tally ),
    }


def analyse( caches, detector_backend, picture_sizes, min_prob, hash_to_path, frame_size, warn ):
    """
    {model label: {'tally': Counter, 'videos': [row, ...], 'seconds': float}}.

    caches: [(path, parsed name)]. hash_to_path maps a file hash to its
    source video; frame_size(path) gives (width, height). warn(text) is
    told about every cache it has to skip.
    """
    results = {}
    for path, parsed in caches:
        name = os.path.basename( hash_to_path.get( parsed['hash'], parsed['hash'] ) )
        video_path = hash_to_path.get( parsed['hash'] )
        if not video_path:
            warn( "%s: source video not found, skipped (its frame size scales the evidence)"%( parsed['hash'], ) )
            continue
        vid_w, vid_h = frame_size( video_path )
        if not vid_w or not vid_h:
            warn( "%s: could not read its frame size, skipped"%( name, ) )
            continue
        raw_boxes, missing = bu_cache.load_cached_raw_boxes(
            parsed['hash'], picture_sizes, parsed['fps'], min_prob, detector_backend, parsed['suffix'] )
        if missing:
            warn( "%s%s: no %s detection cache at size(s) %s, skipped"%(
                name, parsed['suffix'], detector_backend, missing ) )
            continue
        try:
            sample_times, people = bu_person.load_cache( path )
        except Exception as err:
            warn( "%s: unreadable person cache (%s), skipped"%( path, err ) )
            continue
        evidence = bu_person.evidence_detections( raw_boxes, vid_w, vid_h, detector_backend )
        tally = collections.Counter()
        bu_person.male_people_by_time( sample_times, people, evidence, tally )
        interval = 1.0 / parsed['fps']
        label = model_label( parsed['backend'], parsed['key'] )
        group = results.setdefault( label, { 'tally': collections.Counter(), 'videos': [], 'seconds': 0.0 } )
        group['tally'].update( tally )
        group['seconds'] += tally['samples'] * interval
        group['videos'].append( video_row( name + parsed['suffix'], tally, interval ) )
    return results


def percent( part, whole ):
    return 100.0 * part / whole if whole else 0.0


def report( results, top, settings ):
    lines = []
    if not results:
        return [ "No usable person caches: run a video with reverse mode and person windows on first",
                 "(python3 betatv.py --reverse on --preview on), then run this again." ]
    lines.append( "Evidence: detections at or above evidence_min_prob %.2f after suppression; male labels %s, "
                  "female labels %s"%( settings['evidence_min_prob'], ', '.join( settings['male_labels'] ),
                                       ', '.join( settings['female_labels'] ) ) )
    for label in sorted( results ):
        group = results[label]
        tally = group['tally']
        split = shares( tally )
        outlines = tally['outlines']
        lines.append( "" )
        lines.append( "=== person model %s: %d video(s), %d sampled frame(s), %.1f min ==="%(
            label, len( group['videos'] ), tally['samples'], group['seconds'] / 60.0 ) )
        lines.append( "  frames with anyone in them   %7d  (%.0f%% of frames)"%(
            tally['samples_with_people'], percent( tally['samples_with_people'], tally['samples'] ) ) )
        lines.append( "  frames where a man is shown  %7d  (%.0f%% of frames with people)"%(
            tally['samples_with_man_shown'], percent( tally['samples_with_man_shown'], tally['samples_with_people'] ) ) )
        lines.append( "  person outlines              %7d  (%.2f per frame with people)"%(
            outlines, outlines / float( tally['samples_with_people'] ) if tally['samples_with_people'] else 0.0 ) )
        for key, text in OUTLINE_KINDS:
            lines.append( "    %-34s %7d  (%.0f%%)"%( text, split[key], percent( split[key], outlines ) ) )
        lines.append( "  people tracked               %7d: %d male, %d female, %d mixed, %d with no evidence"%(
            tally['tracks'], tally['tracks_male'], tally['tracks_female'], tally['tracks_mixed'],
            tally['tracks_no_evidence'] ) )
        for sex in ( 'male', 'female' ):
            counts = [ tally['evidence_%s_%s'%( sex, where )] for where in ( 'one', 'shared', 'outside' ) ]
            total = sum( counts )
            lines.append( "  %-6s evidence detections   %7d: %.0f%% inside one outline, %.0f%% inside two or more, "
                          "%.0f%% outside every outline"%(
                sex, total, percent( counts[0], total ), percent( counts[1], total ), percent( counts[2], total ) ) )
        lines.extend( readings( tally ) )
        ranked = sorted( group['videos'], key=lambda row: -row['unattributed_person_seconds'] )[ :top ]
        lines.append( "" )
        lines.append( "  most unattributed person-seconds (where to look at a reverse preview):" )
        lines.append( "    %8s %8s %8s %10s  %s"%( 'unattrib', 'man shown', 'seconds', 'male ev.', 'video' ) )
        for row in ranked:
            lines.append( "    %7.1fs %7.0f%% %7.1fs %10s  %s"%(
                row['unattributed_person_seconds'],
                percent( row['samples_with_man_shown'], row['samples_with_people'] ),
                row['seconds'], '%d/%d out'%( row['male_evidence_outside'], row['male_evidence'] ), row['name'] ) )
    return lines


def readings( tally ):
    """What the split points at, as plain sentences with the numbers that drive them."""
    split = shares( tally )
    outlines = tally['outlines']
    male_total = sum( tally['evidence_male_' + where] for where in ( 'one', 'shared', 'outside' ) )
    lines = [ "  reading:" ]
    lines.append( "    %.0f%% of outlines had no evidence at all; none of those can be shown, whoever they are. "
                  "Only a person model that tells men from women itself could decide them"%(
        percent( split['no_evidence'], outlines ) ) )
    if male_total:
        lines.append( "    %.0f%% of male detections fell outside every outline: bodies the person model missed "
                      "(a larger size may find them)"%( percent( tally['evidence_male_outside'], male_total ) ) )
        lines.append( "    %.0f%% of male detections sat in two or more outlines and voted for nobody"%(
            percent( tally['evidence_male_shared'], male_total ) ) )
    else:
        lines.append( "    no male detections at all: nobody can be shown on this footage" )
    return lines


def main( argv=None ):
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--top', type=int, default=10, help="videos to list per person model (default 10)" )
    parser.add_argument( '--json', default=None, metavar='PATH', help="also write every number here" )
    args = parser.parse_args( argv )

    caches = person_caches( bu_cache.VID_HASH_DIR )
    detector_backend = bu_detector.selected_backend_name()
    picture_sizes = bu_cache.resolve_picture_sizes( None, detector_backend )
    min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    print( "%d person cache(s) in %s; detections from %s at %s"%(
        len( caches ), bu_cache.VID_HASH_DIR, detector_backend, picture_sizes ) )
    hash_to_path = bu_cache.build_hash_to_video_path( { parsed['hash'] for _path, parsed in caches } ) if caches else {}
    results = analyse( caches, detector_backend, picture_sizes, min_prob, hash_to_path,
                       bu_tuning.frame_size_for_video, lambda text: print( "  skipped: " + text ) )
    settings = bu_reverse.settings()['person_windows']
    for line in report( results, args.top, settings ):
        print( line )
    if args.json:
        payload = { label: { 'tally': dict( group['tally'] ), 'seconds': round( group['seconds'], 1 ),
                             'videos': group['videos'] }
                    for label, group in results.items() }
        with open( args.json, 'w' ) as handle:
            json.dump( { 'settings': settings, 'models': payload }, handle, indent=2, sort_keys=True )
        print( "" )
        print( "every number: %s"%( args.json, ) )
    return 0


if __name__ == '__main__':
    sys.exit( main() )
