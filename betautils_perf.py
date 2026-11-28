"""
betautils_perf.py - the performance index: what past runs measured,
for the Performance page, the Optimize action and its nudges.

The stats log (betautils_log.write_stats) is the record: one row per
processed file, a fixed set of fields. This index is a SQLite copy of
the fields that say how fast a run went and with what, so the app can
ask without reading the whole log. It is rebuildable, and lives beside
the stats log (perf.sqlite in the same folder):

  PerfSink        a RunRecorder sink: each processed file's stats row,
                  as the run hands it over (betautils_run_record)
  Index.open()    opens the index; an empty one is filled once from the
                  stats log, so past runs count from the first day
  Index.rebuild() empties it and imports the stats log again
  Index.evidence( ... )   rates per resolution band at the settings in
                  use, how often memory capped the render, and nudges

A normal run measures only the settings it used (spec), so evidence
says how fast those settings were, never what another value would do;
that is Optimize's job. Rows from previews (a slice of the video) and
from runs the governor slowed for more than a twentieth of their time
are kept but left out of the rates.

Hardware. Rows from before this index carry no hardware facts, so an
imported row's machine is unknown; it counts as this machine's when the
execution providers match (a CUDA run and a CPU run are never pooled).
New rows carry a fingerprint (CPU model and threads, GPU name, memory
rounded to 4 GiB) so a moved library or a new GPU starts fresh.
"""

import hashlib
import json
import os
import sqlite3
import statistics
import threading
import time

FILE_NAME = 'perf.sqlite'
SCHEMA_VERSION = 1
THROTTLED_SHARE = 0.05

BANDS = ( ( 'sd', 0, 600 ), ( 'hd', 600, 900 ), ( 'fhd', 900, 1300 ), ( 'qhd', 1300, 1800 ), ( 'uhd', 1800, 10**6 ) )

# Nudge codes (protocol/v1/commands/perf.evidence.schema.json).
GPU_OFF = 'perf.gpu_off'
MEMORY_CAPPED = 'perf.memory_capped'
NOT_OPTIMIZED = 'perf.not_optimized'
SETTINGS_CHANGED = 'perf.settings_changed'
SLOWER = 'perf.slower_than_usual'

MEMORY_CAPPED_SHARE = 0.3
SLOWER_SHARE = 0.6
RECENT = 20
# Faster than this many seconds of video per second of detection, the
# detections came from the cache: the stats row has no field that says
# so, and a cached run only reads a file. In 421 full runs on an RTX
# 3070 Laptop GPU (640m, batch 4, 9 to 18 samples a second), real
# detection reached at most 33x real time (small, low-resolution files)
# and cached runs started at 358x, so the line sits well inside the gap.
CACHED_DETECT_SPEED = 50.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta( key TEXT PRIMARY KEY, value TEXT );
CREATE TABLE IF NOT EXISTS runs(
    row_key TEXT PRIMARY KEY,
    at REAL,
    origin TEXT,
    hardware TEXT,
    providers TEXT,
    backend TEXT,
    variant TEXT,
    batch INTEGER,
    sample_fps REAL,
    width INTEGER,
    height INTEGER,
    frames INTEGER,
    fps REAL,
    preview INTEGER,
    detection_seconds REAL,
    encode_seconds REAL,
    total_seconds REAL,
    frames_written INTEGER,
    render_workers INTEGER,
    memory_cap INTEGER,
    throttled_seconds REAL,
    paced_seconds REAL,
    reasons TEXT
);
CREATE INDEX IF NOT EXISTS runs_at ON runs( at );
CREATE TABLE IF NOT EXISTS optimizations(
    at REAL PRIMARY KEY,
    hardware TEXT,
    settings TEXT,
    proposal TEXT,
    applied INTEGER
);
"""


def default_path():
    """Beside the stats log it indexes (state/stats/perf.sqlite by default), so the two move together."""
    import betautils_log as bu_log
    return os.path.join( os.path.dirname( bu_log.configured_stats_path() ), FILE_NAME )


def hardware_fingerprint( facts ):
    """A short id for this machine's speed: CPU model and threads, GPU, memory in 4 GiB steps."""
    if not facts:
        return None
    gpu = ( facts.get( 'gpu' ) or {} ).get( 'name' )
    memory = ( facts.get( 'memory' ) or {} ).get( 'total_mb' )
    parts = [ ( facts.get( 'cpu' ) or {} ).get( 'model' ), ( facts.get( 'cpu' ) or {} ).get( 'threads' ), gpu,
              ( memory // 4096 ) if isinstance( memory, int ) else None ]
    return hashlib.sha256( json.dumps( parts ).encode( 'utf-8' ) ).hexdigest()[:16]


def band_of( height ):
    for name, low, high in BANDS:
        if height is not None and low <= height < high:
            return name
    return None


def _num( value ):
    return value if isinstance( value, ( int, float ) ) and not isinstance( value, bool ) else None


def row_from_stats( stats, origin, hardware=None ):
    """The index's row for one stats row, or None for a row that cannot be one."""
    if not isinstance( stats, dict ) or not stats.get( 'file' ):
        return None
    at = _num( stats.get( 'timestamp' ) )
    key = hashlib.sha256( json.dumps( [ stats.get( 'file' ), stats.get( 'output' ), at ],
                                      sort_keys=True ).encode( 'utf-8' ) ).hexdigest()
    return {
        'row_key': key,
        'at': at,
        'origin': origin,
        'hardware': hardware,
        'providers': stats.get( 'execution_providers' ),
        'backend': stats.get( 'detector_backend' ),
        'variant': stats.get( 'detector_variant' ),
        'batch': _num( stats.get( 'nn_batch_size' ) ),
        'sample_fps': _num( stats.get( 'video_censor_fps' ) ),
        'width': _num( stats.get( 'video_width' ) ),
        'height': _num( stats.get( 'video_height' ) ),
        'frames': _num( stats.get( 'video_frames' ) ),
        'fps': _num( stats.get( 'video_fps' ) ),
        'preview': 1 if stats.get( 'preview_mode' ) else 0,
        'detection_seconds': _num( stats.get( 'detection_seconds' ) ),
        'encode_seconds': _num( stats.get( 'encode_seconds' ) ),
        'total_seconds': _num( stats.get( 'total_seconds' ) ),
        'frames_written': _num( stats.get( 'render_frames_written' ) ),
        'render_workers': _num( stats.get( 'render_render_workers' ) ),
        'memory_cap': _num( stats.get( 'render_memory_worker_cap' ) ),
        'throttled_seconds': _num( stats.get( 'governor_throttled_seconds' ) ),
        'paced_seconds': _num( stats.get( 'governor_paced_seconds' ) ),
        'reasons': json.dumps( stats.get( 'governor_reasons' ) or [] ),
    }


_COLUMNS = ( 'row_key', 'at', 'origin', 'hardware', 'providers', 'backend', 'variant', 'batch', 'sample_fps',
             'width', 'height', 'frames', 'fps', 'preview', 'detection_seconds', 'encode_seconds',
             'total_seconds', 'frames_written', 'render_workers', 'memory_cap', 'throttled_seconds',
             'paced_seconds', 'reasons' )


class Index:

    def __init__( self, connection, path ):
        self.db = connection
        self.path = path
        self.lock = threading.Lock()

    @classmethod
    def open( cls, path=None, stats_path=None, import_empty=True ):
        path = path or default_path()
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        connection = sqlite3.connect( path, check_same_thread=False, timeout=10 )
        connection.executescript( _SCHEMA )
        connection.execute( "INSERT OR IGNORE INTO meta VALUES ( 'schema', ? )", ( str( SCHEMA_VERSION ), ) )
        connection.commit()
        index = cls( connection, path )
        # Once: an index emptied later is not filled again behind the user's back.
        if import_empty and not index._meta( 'imported' ):
            index.import_stats( stats_path )
        return index

    def close( self ):
        with self.lock:
            self.db.close()

    def _meta( self, key ):
        found = self.db.execute( "SELECT value FROM meta WHERE key = ?", ( key, ) ).fetchone()
        return found[0] if found else None

    def count( self ):
        return self.db.execute( "SELECT COUNT(*) FROM runs" ).fetchone()[0]

    def add( self, row ):
        """Add one row; a row already there (same file, output and time) is left as it was."""
        if row is None:
            return False
        with self.lock:
            done = self.db.execute( "INSERT OR IGNORE INTO runs( %s ) VALUES ( %s )"%(
                ', '.join( _COLUMNS ), ', '.join( '?' for _ in _COLUMNS ) ), [ row[c] for c in _COLUMNS ] )
            self.db.commit()
        return done.rowcount == 1

    def import_stats( self, stats_path=None ):
        """Import every row of the stats log not here yet. Returns ( imported, read )."""
        if stats_path is None:
            import betautils_log as bu_log
            stats_path = bu_log.configured_stats_path()
        imported = read = 0
        rows = []
        try:
            with open( stats_path, 'r', encoding='utf-8', errors='replace' ) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    read += 1
                    try:
                        row = row_from_stats( json.loads( line ), 'import' )
                    except ValueError:
                        continue      # a truncated last line, as the analysis tools allow
                    if row is not None:
                        rows.append( row )
        except OSError:
            pass
        with self.lock:
            before = self.count()
            self.db.executemany( "INSERT OR IGNORE INTO runs( %s ) VALUES ( %s )"%(
                ', '.join( _COLUMNS ), ', '.join( '?' for _ in _COLUMNS ) ), [ [ r[c] for c in _COLUMNS ] for r in rows ] )
            self.db.execute( "INSERT OR REPLACE INTO meta VALUES ( 'imported', ? )", ( str( time.time() ), ) )
            self.db.commit()
            imported = self.count() - before
        return imported, read

    def rebuild( self, stats_path=None ):
        """Empty the runs and import the stats log again. Optimization records are kept."""
        with self.lock:
            self.db.execute( "DELETE FROM runs" )
            self.db.commit()
        return self.import_stats( stats_path )

    def record_optimization( self, at, hardware, settings, proposal, applied ):
        with self.lock:
            self.db.execute( "INSERT OR REPLACE INTO optimizations VALUES ( ?, ?, ?, ?, ? )",
                             ( at, hardware, json.dumps( settings, sort_keys=True ),
                               json.dumps( proposal, sort_keys=True ), 1 if applied else 0 ) )
            self.db.commit()

    def last_optimization( self, hardware ):
        found = self.db.execute( "SELECT at, settings, applied FROM optimizations "
                                 "WHERE hardware IS ? ORDER BY at DESC LIMIT 1", ( hardware, ) ).fetchone()
        if found is None:
            return None
        return { 'at': found[0], 'settings': json.loads( found[1] ), 'applied': bool( found[2] ) }

    def rows( self, providers=None, hardware=None ):
        """Every row for this machine, oldest first: its fingerprint, or unknown with the same providers."""
        query = ( "SELECT %s FROM runs WHERE ( hardware IS NOT NULL AND hardware = ? ) "
                  "OR ( hardware IS NULL AND providers IS ? ) ORDER BY at" )%( ', '.join( _COLUMNS ), )
        return [ dict( zip( _COLUMNS, found ) ) for found in self.db.execute( query, ( hardware, providers ) ) ]

    def evidence( self, providers, hardware=None, current=None, facts=None, gpu_enabled=None ):
        """
        What past runs on this machine say, at the settings in use.

        providers: the execution providers a run would use now (the
            stats field's form). current: { variant, batch, sample_fps }
            in effect, to pick the rows measured with them. facts:
            hw.detect's facts, for the GPU nudge. gpu_enabled: the
            setting's value.
        """
        rows = self.rows( providers, hardware )
        current = current or {}
        usable = [ row for row in rows if _measurable( row ) ]
        same = [ row for row in usable if _same_settings( row, current ) ]
        bands = []
        for name, _low, _high in BANDS:
            in_band = [ row for row in same if band_of( row['height'] ) == name ]
            if not in_band:
                continue
            detected = [ row for row in in_band if not _cached( row ) ]
            bands.append( {
                'band': name,
                'runs': len( in_band ),
                'detected_runs': len( detected ),
                'detect_speed': _median( _detect_speed( row ) for row in detected ),
                'render_fps': _median( _render_fps( row ) for row in in_band ),
                'total_speed': _median( _total_speed( row ) for row in detected ),
            } )
        renders = [ row for row in rows if row['render_workers'] ]
        recent = renders[-RECENT:]
        capped = [ row for row in recent if _memory_capped( row ) ]
        optimized = self.last_optimization( hardware )
        nudges = _nudges( usable, same, recent, capped, optimized, current, facts, gpu_enabled )
        return {
            'runs': len( rows ),
            'measured': len( same ),
            'bands': bands,
            'memory_capped': { 'runs': len( capped ), 'of': len( recent ) },
            'optimized': ( { 'at': optimized['at'], 'applied': optimized['applied'] } if optimized else None ),
            'nudges': nudges,
        }


def _measurable( row ):
    """A full (not preview) run with its timings, that the governor did not slow much."""
    if row['preview'] or not row['detection_seconds'] or not row['encode_seconds'] or not row['frames'] or not row['fps']:
        return False
    total = row['total_seconds'] or ( row['detection_seconds'] + row['encode_seconds'] )
    throttled = ( row['throttled_seconds'] or 0 ) + ( row['paced_seconds'] or 0 )
    return throttled <= THROTTLED_SHARE * total


def _same_settings( row, current ):
    for name in ( 'variant', 'batch', 'sample_fps' ):
        if current.get( name ) is not None and row[name] != current[name]:
            return False
    return True


def _cached( row ):
    """Detections read from the cache rather than detected (see CACHED_DETECT_SPEED)."""
    return _detect_speed( row ) > CACHED_DETECT_SPEED


def _video_seconds( row ):
    return row['frames'] / row['fps']


def _detect_speed( row ):
    """Video seconds detected per second: 2.0 detects a minute of video in 30 s."""
    return _video_seconds( row ) / row['detection_seconds']


def _render_fps( row ):
    return ( row['frames_written'] or row['frames'] ) / row['encode_seconds']


def _total_speed( row ):
    total = row['total_seconds'] or ( row['detection_seconds'] + row['encode_seconds'] )
    return _video_seconds( row ) / total


def _median( values ):
    values = [ value for value in values if value is not None ]
    return round( statistics.median( values ), 3 ) if values else None


def _memory_capped( row ):
    """The render's workers were set by the memory cap (a cap above the workers used did not bind)."""
    return row['memory_cap'] is not None and row['render_workers'] is not None and row['memory_cap'] <= row['render_workers']


def _nudges( usable, same, recent, capped, optimized, current, facts, gpu_enabled ):
    nudges = []
    if facts is not None and facts.get( 'cuda_usable' ) and gpu_enabled != 1:
        nudges.append( { 'code': GPU_OFF, 'params': { 'gpu': ( facts.get( 'gpu' ) or {} ).get( 'name' ) },
                         'action': { 'kind': 'set', 'values': { 'gpu_enabled': 1 } } } )
    if len( recent ) >= 5 and len( capped ) >= MEMORY_CAPPED_SHARE * len( recent ):
        nudges.append( { 'code': MEMORY_CAPPED, 'params': { 'runs': len( capped ), 'of': len( recent ) },
                         'action': { 'kind': 'optimize', 'stages': [ 'render' ] } } )
    if optimized is None:
        nudges.append( { 'code': NOT_OPTIMIZED, 'params': {}, 'action': { 'kind': 'optimize', 'stages': None } } )
    elif any( optimized['settings'].get( name ) != current.get( name ) for name in ( 'variant', 'batch' )
              if name in optimized['settings'] ):
        nudges.append( { 'code': SETTINGS_CHANGED, 'params': {}, 'action': { 'kind': 'optimize', 'stages': None } } )
    detected = [ row for row in same if not _cached( row ) ]
    if len( detected ) >= 8:
        speeds = [ _total_speed( row ) for row in detected ]
        earlier, last = speeds[:-3], speeds[-3:]
        usual = statistics.median( earlier )
        now = statistics.median( last )
        if usual and now < SLOWER_SHARE * usual:
            nudges.append( { 'code': SLOWER, 'params': { 'usual': round( usual, 3 ), 'recent': round( now, 3 ) },
                             'action': { 'kind': 'optimize', 'stages': None } } )
    return nudges


class PerfSink:
    """
    The RunRecorder sink: each processed file's stats row into the index.
    Best effort: the index never fails a run.
    """

    required = False

    def __init__( self, index_for, hardware=None ):
        self._index_for = index_for       # () -> Index, opened on first use
        self.hardware = hardware

    def record( self, run ):
        # row_from_stats gives None for a file with no stats row (a skip or a failure).
        self._index_for().add( row_from_stats( run.stats, 'run', hardware=self.hardware ) )
