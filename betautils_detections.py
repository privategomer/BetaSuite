"""
betautils_detections.py - reading detection caches without holding them as dicts.

A detection cache is a gzipped JSON list of dicts, one per detection:
x, y, w, h, class_id, score, t, size. Parsed, each dict costs about
500 bytes of Python objects. A 56-video library holds 7.3
million detections for one configuration, so any tool that loads it
whole needs 3.7 GB before it computes anything, and the analysis hub ran
three or four such tools at once on a 15 GB machine. The kernel killed
them (exit 137).

Two ways out, both here:

  iter_videos   one video's detections at a time. A tool that only
                needs per-video work, or per-label aggregates, holds one
                video's dicts plus its aggregates: memory follows the
                largest video, not the library
  Columns       the same records as numpy columns, about 51 bytes each
                instead of 500. Exact: every value round-trips to the
                identical Python float, int or str. A cache whose records
                are not the standard shape is never compacted; it stays
                as dicts, so nothing is ever silently changed

Analogy: the dicts are a filing cabinet with one folder per detection.
Columns are the same facts as a spreadsheet: one sheet for every x, one
for every score. Same data, a fraction of the drawer space, and a
folder can still be printed back out exactly when someone needs one.
"""

import numpy as np

import betautils_hash as bu_hash


FLOAT_KEYS = ( 'x', 'y', 'w', 'h', 'score', 't' )
STANDARD_KEYS = frozenset( FLOAT_KEYS + ( 'class_id', 'size' ) )


class Columns:
    """
    One cache file's detections as columns. Build with to_columns; read
    back with to_dicts, which returns new dicts every call, so a caller
    may mutate what it gets.
    """

    __slots__ = ( 'x', 'y', 'w', 'h', 'score', 't', 'label', 'size', 'labels' )

    def __len__( self ):
        return len( self.t )

    @property
    def nbytes( self ):
        return sum( getattr( self, key ).nbytes for key in FLOAT_KEYS ) + self.label.nbytes + self.size.nbytes

    def class_ids( self ):
        """Every record's class_id, as a list of str."""
        return [ self.labels[index] for index in self.label.tolist() ]

    def to_dicts( self, **extra ):
        """
        The records as dicts, equal to what the cache file parses to.
        extra keys (e.g. _file_hash=...) are added to every one.
        """
        x, y, w, h, score, t = ( getattr( self, key ).tolist() for key in FLOAT_KEYS )
        rows = zip( x, y, w, h, self.class_ids(), score, t, self.size.tolist() )
        if extra:
            return [ dict( { 'x': a, 'y': b, 'w': c, 'h': d, 'class_id': e, 'score': f, 't': g,
                             'size': k }, **extra )
                     for a, b, c, d, e, f, g, k in rows ]
        return [ { 'x': a, 'y': b, 'w': c, 'h': d, 'class_id': e, 'score': f, 't': g, 'size': k }
                 for a, b, c, d, e, f, g, k in rows ]


def to_columns( records ):
    """
    Columns for a list of detection dicts, or None when any record is
    not exactly the standard shape (the eight keys, floats where floats
    belong, an int size, a str class_id). None means "keep the dicts":
    compacting a record with an extra key, or an int x, would change what
    a reader gets back.
    """
    for record in records:
        if record.keys() != STANDARD_KEYS:
            return None
        if type( record['class_id'] ) is not str or type( record['size'] ) is not int:
            return None
        for key in FLOAT_KEYS:
            if type( record[key] ) is not float:
                return None
    columns = Columns()
    for key in FLOAT_KEYS:
        setattr( columns, key, np.fromiter( ( record[key] for record in records ), dtype=np.float64,
                                            count=len( records ) ) )
    labels = sorted( { record['class_id'] for record in records } )
    index = { label: position for position, label in enumerate( labels ) }
    dtype = np.uint8 if len( labels ) <= 256 else np.uint16
    columns.label = np.fromiter( ( index[record['class_id']] for record in records ), dtype=dtype,
                                 count=len( records ) )
    columns.size = np.fromiter( ( record['size'] for record in records ), dtype=np.int64,
                                count=len( records ) )
    columns.labels = labels
    return columns


def read( path ):
    """
    One cache file, compacted when it can be: a Columns, or the parsed
    list when its records are not the standard shape.
    """
    records = bu_hash.read_json( path )
    columns = to_columns( records ) if records else None
    return columns if columns is not None else records


def as_dicts( held, **extra ):
    """Fresh dicts from what read() returned (Columns or a list), extra keys added."""
    if isinstance( held, Columns ):
        return held.to_dicts( **extra )
    import copy
    records = copy.deepcopy( held )
    if extra:
        for record in records:
            record.update( extra )
    return records


# ---------------------------------------------------------------------------
# Process-wide parse cache
# ---------------------------------------------------------------------------
#
# Several tools read the same caches more than once in one process:
# auto_tune replays every video for every candidate (87 replays in one
# run), and betabench's cache group runs four measurements over the same
# configuration. Parsing is the expensive part (about 2 to 5 us per
# detection), so each file is parsed once and kept, as Columns where it
# can be: 51 bytes a detection instead of 500, so keeping a whole library
# costs about a tenth of what holding its dicts did.
#
# Keyed on (path, mtime, size), so a cache rewritten on disk is re-read
# rather than served stale.

_PARSED = {}


def _key( path ):
    import os
    stat = os.stat( path )
    return ( path, stat.st_mtime_ns, stat.st_size )


def cached_read( path ):
    """read(path), parsed once per process and kept. Raises like read_json when unreadable."""
    key = _key( path )
    held = _PARSED.get( key )
    if held is None:
        held = read( path )
        _PARSED[key] = held
    return held


def read_once( path ):
    """
    For a caller that reads a file once and moves on: the kept parse when
    there is one, otherwise the plain parsed list, neither compacted nor
    kept. Compacting costs about as much as the parse itself, which only
    pays back on a second read.
    """
    held = _PARSED.get( _key( path ) )
    return held if held is not None else bu_hash.read_json( path )


def forget( paths ):
    """Drop these files from the parse cache."""
    wanted = set( paths )
    for key in [ key for key in _PARSED if key[0] in wanted ]:
        del _PARSED[key]


def clear():
    _PARSED.clear()


class VideoCache:
    """
    One configuration's detections, served one video at a time.

    Iterate it as often as needed: each pass yields ( file_hash, records )
    per video, records fresh dicts tagged '_file_hash' (as betabench's
    loader always did), so a caller may mutate them. The first pass
    parses each file; later passes rebuild the dicts from the kept
    Columns, roughly twice as fast as parsing. Only one video's dicts
    exist at a time.

    Args:
        discovered: {file_hash: [cache_path, ...]}.
        logger: For unreadable-cache warnings (each file warned once).
    """

    def __init__( self, discovered, logger=None ):
        self.discovered = dict( discovered )
        self.logger = logger
        self._unreadable = set()
        self.detections = 0

    def __len__( self ):
        return len( self.discovered )

    @property
    def file_hashes( self ):
        return list( self.discovered )

    def prime( self ):
        """
        Parse every file once (without building dicts) so the detection
        count is known before the first pass. Returns that count.
        """
        total = 0
        for cache_paths in self.discovered.values():
            for cache_path in cache_paths:
                if cache_path in self._unreadable:
                    continue
                try:
                    total += len( cached_read( cache_path ) )
                except Exception as err:
                    self._unreadable.add( cache_path )
                    if self.logger is not None:
                        self.logger.warning( "unreadable cache %s: %s"%( cache_path, err ) )
        self.detections = total
        return total

    def __iter__( self ):
        total = 0
        for file_hash, cache_paths in self.discovered.items():
            records = []
            for cache_path in cache_paths:
                if cache_path in self._unreadable:
                    continue
                try:
                    held = cached_read( cache_path )
                except Exception as err:
                    self._unreadable.add( cache_path )
                    if self.logger is not None:
                        self.logger.warning( "unreadable cache %s: %s"%( cache_path, err ) )
                    continue
                records.extend( as_dicts( held, _file_hash=file_hash ) )
            total += len( records )
            yield file_hash, records
        self.detections = total
