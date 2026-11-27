"""
betautils_catalog_censors.py - what a censored output hid, and how.

Every output's name carries its censor key, and every censor key's full
expansion is recorded once in the run-key folder
(betautils_cache_paths.record_run_key: censor-<key>.json). Its 'labels'
entry holds each label the run censored, with its resolved look. This
module reads that record, so the library can show and filter by the
censors applied: which labels, and which kinds of look (blur, pixel,
bar, sticker, ...). Nothing here is a fixed list; labels and look kinds
are whatever the records say, so a new label or look needs no change.

A record never changes once written, so each key is read at most once
per process. A key with no readable record has unknown censors (None),
which no censor filter matches.
"""

import json
import os
import threading

_CACHE = {}
_LOCK = threading.Lock()


def _record_path( censor_key ):
    import betaconst
    return os.path.join( betaconst.run_key_dir, 'censor-%s.json'%( censor_key, ) )


def _look_kinds( style ):
    styles = style if isinstance( style, list ) else [ style ]
    return sorted( { entry['type'] for entry in styles if isinstance( entry, dict ) and isinstance( entry.get( 'type' ), str ) } )


def censors_of( censor_key ):
    """{ label: [ look kind ] } for a censor key, or None when its record cannot be read."""
    if not censor_key:
        return None
    with _LOCK:
        if censor_key in _CACHE:
            return _CACHE[censor_key]
    try:
        with open( _record_path( censor_key ), encoding='utf-8' ) as handle:
            labels = json.load( handle )['identity']['labels']
        out = { label: _look_kinds( entry.get( 'censor_style' ) ) for label, entry in labels.items()
                if isinstance( entry, dict ) }
    except ( OSError, ValueError, KeyError, TypeError, AttributeError ):
        out = None
    with _LOCK:
        _CACHE[censor_key] = out
    return out


def summary( per_output ):
    """[ { label, looks } ] over several outputs' censors, sorted, looks merged."""
    merged = {}
    for censors in per_output:
        for label, looks in ( censors or {} ).items():
            merged.setdefault( label, set() ).update( looks )
    return [ { 'label': label, 'looks': sorted( merged[label] ) } for label in sorted( merged ) ]


def matches( per_output, labels=(), looks=() ):
    """Whether one output hid every label in labels, and used every look kind in looks."""
    wanted_labels, wanted_looks = set( labels ), set( looks )
    for censors in per_output:
        if censors is None:
            continue
        if not wanted_labels <= set( censors ):
            continue
        used = set()
        for label, kinds in censors.items():
            if not wanted_labels or label in wanted_labels:
                used.update( kinds )
        if wanted_looks <= used:
            return True
    return False


def forget_cache():
    """For tests: read every record again."""
    with _LOCK:
        _CACHE.clear()
