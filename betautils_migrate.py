"""
betautils_migrate.py - copy a 1.x tree's app data into the 2.x folders.

A 1.x tree is a folder holding resources/ and output/, plus the
betaconfig_local.py beside that checkout's betaconfig.py. The migration:

  COPIES, NEVER MOVES. Nothing in the 1.x tree is written, renamed or
      deleted, so 1.x (or BETASUITE_DATA_ROOT) keeps working on it.
  IS SAFE TO RE-RUN. A file already copied is left alone; a destination
      that differs from its source is never overwritten, only reported.
      A run stopped part way (a full disk, Ctrl-C) leaves only whole
      files behind, and the next run copies the rest.
  REPORTS WHAT IT DID, file by file, in a JSON report and a summary.

What goes where comes from betautils_paths.LEGACY_LAYOUT, the table the
resolver reads, so every copied file lands exactly where the app will
look for it. Copied, in this order so the small and irreplaceable go
first: settings (config), durable data (models, stickers, run keys),
state (logs, stats, benchmarks), cache (detections, shot cuts,
transcodes, file hashes; skippable, it can be rebuilt).

Not copied:
  media      the input and output video folders. They are the user's
             files: the report names each folder with its size (never
             the file names in it), and the 2.x layout can keep using
             them in place (BETASUITE_INPUT_DIR / BETASUITE_OUTPUT_DIR)
  unknown    anything under resources/ or output/ the table does not
             name (the bare catch-all rows), listed by top-level name
  transient  temp files, run markers, the run lock and its holder record,
             and symbolic links (never followed)

Config strings are copied byte for byte, so model and sticker paths keep
their 1.x spelling, and the detection, censor and encode keys after
migration are the ones before it: every cache copied is a cache hit.

Refused before anything is copied, with a code and params (the CLI words
them): not a 1.x tree, source and target overlap, a run in progress on
either side, not enough free space.
"""

import errno
import hashlib
import json
import os
import shutil
import sys
import time

import betautils_paths as bu_paths
import betautils_runlock as bu_runlock

REPORT_VERSION = 1

# Copied kinds, in copy order. 'cache' can be left out.
COPY_ORDER = ( 'config', 'data', 'state', 'cache' )
MEDIA_KINDS = ( 'inputs', 'outputs' )

# A destination being written, before it is renamed into place.
PARTIAL_SUFFIX = '.betasuite-partial'

TRANSIENT_SUFFIXES = ( '.tmp', PARTIAL_SUFFIX, '.running.json' )
TRANSIENT_NAMES = ( 'run.lock', 'run.lock' + bu_runlock.HOLDER_SUFFIX )

# Free space kept over what the copy needs, on each target filesystem.
SPACE_MARGIN_BYTES = 1 << 30

LOCAL_CONFIG_NAME = 'betaconfig_local.py'

# Actions in a plan.
COPY, SAME, CONFLICT = 'copy', 'same', 'conflict'
# Outcomes of a run.
COMPLETE, DRY_RUN, STOPPED_DISK_FULL, STOPPED_INTERRUPTED, FINISHED_WITH_ERRORS = (
    'complete', 'dry_run', 'stopped_disk_full', 'stopped_interrupted', 'finished_with_errors' )


class MigrationRefused( Exception ):
    """
    The migration will not start. `code` and `params` say why, for the
    caller to word (the engine never sends sentences).

    Codes: not_a_1x_tree {source}; overlap {source, target, kind};
    run_in_progress {side, program, pid, started_at}; no_space {folder,
    needed_bytes, free_bytes}.
    """

    def __init__( self, code, **params ):
        super().__init__( code )
        self.code = code
        self.params = params


class Item:
    """One file the migration considers copying."""

    __slots__ = ( 'source', 'relative', 'kind', 'dest', 'size', 'action', 'outcome' )

    def __init__( self, source, relative, kind, dest, size, action ):
        self.source = source
        self.relative = relative
        self.kind = kind
        self.dest = dest
        self.size = size
        self.action = action
        self.outcome = None

    def as_dict( self ):
        return { 'path': self.relative, 'kind': self.kind, 'dest': self.dest, 'bytes': self.size,
                 'action': self.action, 'outcome': self.outcome }


class Plan:
    """What a migration would do. Building one reads and never writes."""

    def __init__( self, source_root, local_config, targets, include_cache ):
        self.source_root = source_root
        self.local_config = local_config
        self.targets = targets
        self.include_cache = include_cache
        self.items = []
        self.media = {}      # 1.x folder -> [ files, bytes ]
        self.unknown = {}    # 1.x top-level entry -> [ files, bytes ]
        self.skipped = { 'transient': [ 0, 0 ], 'link': [ 0, 0 ], 'cache_left_out': [ 0, 0 ] }
        self.config_refs = []

    def to_copy( self ):
        return [ item for item in self.items if item.action == COPY ]

    def totals( self ):
        """{ kind: { action: [ files, bytes ] } }"""
        totals = {}
        for item in self.items:
            entry = totals.setdefault( item.kind, {} ).setdefault( item.action, [ 0, 0 ] )
            entry[0] += 1
            entry[1] += item.size
        return totals


def _same_contents( a, b, chunk=1 << 20 ):
    digests = []
    for path in ( a, b ):
        digest = hashlib.sha256()
        with open( path, 'rb' ) as handle:
            for block in iter( lambda: handle.read( chunk ), b'' ):
                digest.update( block )
        digests.append( digest.digest() )
    return digests[0] == digests[1]


def _action( source, dest, source_stat ):
    """copy, same or conflict for one file, reading only."""
    try:
        dest_stat = os.stat( dest )
    except FileNotFoundError:
        return COPY
    if dest_stat.st_size == source_stat.st_size and dest_stat.st_mtime_ns == source_stat.st_mtime_ns:
        return SAME
    if dest_stat.st_size == source_stat.st_size and _same_contents( source, dest ):
        return SAME
    return CONFLICT


def _is_transient( name ):
    return name in TRANSIENT_NAMES or name.endswith( TRANSIENT_SUFFIXES )


def _inside( path, folder ):
    path, folder = os.path.normcase( os.path.realpath( path ) ), os.path.normcase( os.path.realpath( folder ) )
    return path == folder or path.startswith( folder.rstrip( os.sep ) + os.sep )


def _config_references( local_config ):
    """1.x paths a local config names that the migration does not carry: (setting, value) pairs."""
    import betautils_local as bu_local
    if not local_config or not os.path.isfile( local_config ):
        return []
    with open( local_config, 'r', encoding='utf-8' ) as handle:
        entries, _problems = bu_local.parse( handle.read() )
    found = []

    def walk( name, value ):
        if isinstance( value, str ):
            parts = [ p for p in value.replace( '\\', '/' ).split( '/' ) if p ]
            if parts[:1] == [ '..' ]:
                entry = bu_paths.legacy_entry( parts[1:] )
                if entry is not None and entry[0] in ( 'resources', 'output' ) and len( parts ) > 2:
                    found.append( ( name, value ) )
        elif isinstance( value, dict ):
            for key, inner in value.items():
                walk( '%s.%s'%( name, key ), inner )
        elif isinstance( value, ( list, tuple ) ):
            for index, inner in enumerate( value ):
                walk( '%s[%d]'%( name, index ), inner )
    for name, value, _lineno in entries:
        walk( name, value )
    return found


def make_plan( source_root, local_config=None, targets=None, include_cache=True ):
    """
    Read a 1.x tree and say what migrating it would do. Writes nothing.

    Args:
        source_root: The folder holding resources/ and output/.
        local_config: The 1.x checkout's betaconfig_local.py, or None.
        targets: { kind: folder }; default betautils_paths.os_dirs().
        include_cache: False leaves the cache kind out.

    Raises:
        MigrationRefused: not_a_1x_tree, or overlap.
    """
    source_root = os.path.abspath( source_root )
    if not any( os.path.isdir( os.path.join( source_root, top ) ) for top in ( 'resources', 'output' ) ):
        raise MigrationRefused( 'not_a_1x_tree', source=source_root )
    targets = dict( bu_paths.os_dirs() if targets is None else targets )
    for kind in COPY_ORDER:
        target = targets[kind]
        if _inside( target, source_root ) or _inside( source_root, target ):
            raise MigrationRefused( 'overlap', source=source_root, target=target, kind=kind )
    plan = Plan( source_root, local_config, targets, include_cache )

    for top in ( 'resources', 'output' ):
        top_dir = os.path.join( source_root, top )
        for directory, subdirectories, names in os.walk( top_dir, followlinks=False ):
            subdirectories.sort()
            for sub in list( subdirectories ):
                if os.path.islink( os.path.join( directory, sub ) ):
                    plan.skipped['link'][0] += 1
                    subdirectories.remove( sub )
            for name in sorted( names ):
                path = os.path.join( directory, name )
                relative_parts = os.path.relpath( path, source_root ).split( os.sep )
                relative = '/'.join( relative_parts )
                if os.path.islink( path ):
                    plan.skipped['link'][0] += 1
                    continue
                stat = os.stat( path )
                if _is_transient( name ):
                    plan.skipped['transient'][0] += 1
                    plan.skipped['transient'][1] += stat.st_size
                    continue
                prefix, kind, sub, rest = bu_paths.legacy_entry( relative_parts )
                if kind in MEDIA_KINDS:
                    entry = plan.media.setdefault( prefix, [ 0, 0 ] )
                    entry[0] += 1
                    entry[1] += stat.st_size
                    continue
                if prefix in ( 'resources', 'output' ):
                    entry = plan.unknown.setdefault( '/'.join( relative_parts[:2] ), [ 0, 0 ] )
                    entry[0] += 1
                    entry[1] += stat.st_size
                    continue
                if kind == 'cache' and not include_cache:
                    plan.skipped['cache_left_out'][0] += 1
                    plan.skipped['cache_left_out'][1] += stat.st_size
                    continue
                dest = os.path.join( targets[kind], *[ p for p in ( sub, ) + tuple( rest ) if p ] )
                plan.items.append( Item( path, relative, kind, dest, stat.st_size, _action( path, dest, stat ) ) )

    if local_config and os.path.isfile( local_config ):
        stat = os.stat( local_config )
        dest = os.path.join( targets['config'], LOCAL_CONFIG_NAME )
        plan.items.append( Item( local_config, LOCAL_CONFIG_NAME, 'config', dest, stat.st_size,
                                 _action( local_config, dest, stat ) ) )
        plan.config_refs = _config_references( local_config )

    plan.items.sort( key=lambda item: ( COPY_ORDER.index( item.kind ), item.relative ) )
    return plan


def _live_run_on_source( source_root ):
    """The holder of the 1.x tree's run lock, or of a live run marker, else None. Creates nothing."""
    holder = bu_runlock.probe( os.path.join( source_root, 'output', 'run.lock' ) )
    if holder is not None:
        return holder
    import betautils_log as bu_log
    marker = os.path.join( source_root, 'output', 'stats', 'betasuite_stats.running.json' )
    try:
        with open( marker, 'r', encoding='utf-8' ) as handle:
            record = json.load( handle )
    except ( OSError, ValueError ):
        return None
    if not isinstance( record, dict ):
        return None
    boot = record.get( 'boot_id' )
    if boot and bu_log.current_boot_id() and boot != bu_log.current_boot_id():
        return None
    if bu_log.betatv_process_running( record.get( 'pid' ) ):
        return { 'program': 'betatv', 'pid': record.get( 'pid' ), 'started_at': record.get( 'started_at' ) }
    return None


def _existing_ancestor( path ):
    while not os.path.exists( path ):
        parent = os.path.dirname( path )
        if parent == path:
            break
        path = parent
    return path


def check_can_start( plan, disk_usage=shutil.disk_usage ):
    """
    Refuse a run that should not start: a run in progress on either side,
    or not enough room. Reads only.
    """
    holder = _live_run_on_source( plan.source_root )
    if holder is not None:
        raise MigrationRefused( 'run_in_progress', side='source', **holder )
    holder = bu_runlock.probe( os.path.join( plan.targets['state'], 'run.lock' ) )
    if holder is not None:
        raise MigrationRefused( 'run_in_progress', side='target', **holder )
    needed = {}
    for item in plan.to_copy():
        anchor = _existing_ancestor( os.path.dirname( item.dest ) )
        device = os.stat( anchor ).st_dev
        entry = needed.setdefault( device, [ anchor, 0 ] )
        entry[1] += item.size
    for anchor, total in needed.values():
        free = disk_usage( anchor ).free
        if free < total + SPACE_MARGIN_BYTES:
            raise MigrationRefused( 'no_space', folder=anchor, needed_bytes=total + SPACE_MARGIN_BYTES,
                                    free_bytes=free )


def _copy_one( item ):
    folder = os.path.dirname( item.dest )
    os.makedirs( folder, exist_ok=True )
    partial = item.dest + PARTIAL_SUFFIX
    try:
        shutil.copy2( item.source, partial )
        os.replace( partial, item.dest )
    except BaseException:
        try:
            os.remove( partial )
        except OSError:
            pass
        raise


def execute( plan, logger, clock=time.time ):
    """
    Copy what the plan says to copy, holding the 2.x run lock.

    Returns:
        ( outcome, errors ): one of COMPLETE, FINISHED_WITH_ERRORS,
        STOPPED_DISK_FULL, STOPPED_INTERRUPTED, and [ { path, error } ].
        Each item's `outcome` says what happened to it.
    """
    check_can_start( plan )
    errors = []
    outcome = COMPLETE
    lock = bu_runlock.acquire( 'migrate', path=os.path.join( plan.targets['state'], 'run.lock' ), logger=logger )
    try:
        for item in plan.items:
            if item.action != COPY:
                item.outcome = item.action
                continue
            try:
                _copy_one( item )
            except KeyboardInterrupt:
                item.outcome = 'not_copied'
                outcome = STOPPED_INTERRUPTED
                break
            except OSError as err:
                if err.errno == errno.ENOSPC:
                    item.outcome = 'not_copied'
                    outcome = STOPPED_DISK_FULL
                    logger.error( "a disk is full while copying %s; stopped. Free some space and run the "
                                  "migration again: what is copied stays, the rest follows"%( item.relative, ) )
                    break
                item.outcome = 'error'
                errors.append( { 'path': item.relative, 'error': '%s: %s'%( type( err ).__name__, err.strerror or err ) } )
                logger.error( "could not copy %s: %s"%( item.relative, err ) )
                continue
            item.outcome = 'copied'
            logger.debug( "copied %s -> %s"%( item.relative, item.dest ) )
        for item in plan.items:
            if item.outcome is None:
                item.outcome = 'not_copied'
    finally:
        lock.release()
    if outcome == COMPLETE and errors:
        outcome = FINISHED_WITH_ERRORS
    return outcome, errors


def media_hint( plan ):
    """The two variables that keep the 1.x video folders in use, or {} when it has none."""
    hint = {}
    tops = { prefix.split( '/' )[0] for prefix in plan.media }
    if 'resources' in tops:
        hint[bu_paths.INPUT_ENV_VAR] = os.path.join( plan.source_root, 'resources' )
    if 'output' in tops:
        hint[bu_paths.OUTPUT_ENV_VAR] = os.path.join( plan.source_root, 'output' )
    return hint


def report( plan, outcome, errors=(), started_at=None, finished_at=None ):
    """The JSON report of a plan, and of a run when outcome is not DRY_RUN."""
    return {
        'version': REPORT_VERSION,
        'outcome': outcome,
        'started_at': started_at,
        'finished_at': finished_at,
        'source': plan.source_root,
        'local_config': plan.local_config,
        'targets': plan.targets,
        'include_cache': plan.include_cache,
        'totals': plan.totals(),
        'files': [ item.as_dict() for item in plan.items ],
        'media_left_in_place': [ { 'folder': folder, 'files': n, 'bytes': size }
                                 for folder, ( n, size ) in sorted( plan.media.items() ) ],
        'media_hint': media_hint( plan ),
        'unknown_left_in_place': [ { 'path': path, 'files': n, 'bytes': size }
                                   for path, ( n, size ) in sorted( plan.unknown.items() ) ],
        'skipped': { reason: { 'files': n, 'bytes': size } for reason, ( n, size ) in plan.skipped.items() },
        'config_references_not_carried': [ { 'setting': name, 'value': value } for name, value in plan.config_refs ],
        'errors': list( errors ),
    }


def write_json( path, document ):
    folder = os.path.dirname( path )
    if folder:
        os.makedirs( folder, exist_ok=True )
    temporary = path + '.tmp'
    with open( temporary, 'w', encoding='utf-8' ) as handle:
        json.dump( document, handle, indent=1, sort_keys=True )
        handle.write( '\n' )
    os.replace( temporary, path )


def record_paths( plan, stamp ):
    """Where a real run writes its report and the record M5's catalog reads."""
    return ( os.path.join( plan.targets['state'], 'migration', 'report-%s.json'%( stamp, ) ),
             os.path.join( plan.targets['data'], 'migrated_from.json' ) )


def migrated_from( plan, finished_at ):
    """The record of where the 1.x tree and its media are, for the catalog's watched folders."""
    return { 'version': REPORT_VERSION, 'source': plan.source_root, 'finished_at': finished_at,
             'media_folders': [ os.path.join( plan.source_root, *folder.split( '/' ) ) for folder in sorted( plan.media ) ] }


def human_bytes( size ):
    for unit in ( 'B', 'KiB', 'MiB', 'GiB', 'TiB' ):
        if size < 1024 or unit == 'TiB':
            return '%d %s'%( size, unit ) if unit == 'B' else '%.1f %s'%( size, unit )
        size /= 1024.0


def summary_lines( document ):
    """The report as lines for the console and log."""
    lines = [ "1.x tree: %s"%( document['source'], ) ]
    for kind in COPY_ORDER:
        actions = document['totals'].get( kind, {} )
        if not actions and kind == 'cache' and not document['include_cache']:
            skipped = document['skipped']['cache_left_out']
            lines.append( "  %-7s left out (%d files, %s)"%( kind, skipped['files'], human_bytes( skipped['bytes'] ) ) )
            continue
        parts = [ "%s %d (%s)"%( action, n, human_bytes( size ) ) for action, ( n, size ) in sorted( actions.items() ) ]
        lines.append( "  %-7s -> %s: %s"%( kind, document['targets'][kind], ', '.join( parts ) or 'nothing' ) )
    for entry in document['media_left_in_place']:
        lines.append( "  media left in place: %s (%d files, %s)"%(
            entry['folder'], entry['files'], human_bytes( entry['bytes'] ) ) )
    for name, value in sorted( document['media_hint'].items() ):
        lines.append( "    to keep using it: %s=%s"%( name, value ) )
    for entry in document['unknown_left_in_place']:
        lines.append( "  not a 1.x app folder, left in place: %s (%d files, %s)"%(
            entry['path'], entry['files'], human_bytes( entry['bytes'] ) ) )
    for reason, entry in sorted( document['skipped'].items() ):
        if entry['files'] and reason != 'cache_left_out':
            lines.append( "  skipped (%s): %d"%( reason, entry['files'] ) )
    for ref in document['config_references_not_carried']:
        lines.append( "  note: %s = %r names a folder the migration does not copy"%( ref['setting'], ref['value'] ) )
    conflicts = [ f for f in document['files'] if f['action'] == CONFLICT ]
    for entry in conflicts:
        lines.append( "  conflict, left as it is: %s (the 2.x copy differs)"%( entry['dest'], ) )
    for entry in document['errors']:
        lines.append( "  error: %s: %s"%( entry['path'], entry['error'] ) )
    lines.append( "outcome: %s"%( document['outcome'], ) )
    return lines
