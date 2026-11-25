"""
betautils_catalog.py - the library catalog, composed from its parts.

  betautils_catalog_store   the SQLite file, schema steps, the lock
  betautils_catalog_index   folders, items, locations, outputs, re-finding
  betautils_catalog_scan    passes over the disk, settling, hashing
  betautils_catalog_types   which video type an item gets, and why

Catalog wires them together and answers in the shapes the engine
protocol sends (protocol/README.md, "Catalog"). Each part owns its own
logic and tables; this module only composes them, so a front end that
needs one part (the CLI reading pins, the rebuild tool) builds just
that.

A NEW CATALOG watches the library's video input folders (source/ and
uncensored_vids/, where they exist) and the video folders a 1.x
migration recorded (data/migrated_from.json), each with its folder rule
on: a 1.x tree's folder names pick the video type, as they always did.
"""

import json
import os

import betautils_catalog_index as bu_index
import betautils_catalog_scan as bu_scan
import betautils_catalog_store as bu_store
import betautils_catalog_types as bu_types

LIST_LIMIT_MAX = 500
LIST_LIMIT_DEFAULT = 200
CHANGED_ITEMS_MAX = 1000

SORT_KEYS = ( 'name', 'added_at', 'size', 'status' )
VIDEO_INPUT_FOLDER_NAMES = ( 'uncensored_vids', 'source' )
MIGRATION_RECORD = 'migrated_from.json'


def default_seed_folders():
    """The video input folders a new catalog watches, folder rule on."""
    import betaconst
    import betautils_paths as bu_paths
    seeds = [ betaconst.video_path_source_backup, betaconst.video_path_uncensored ]
    record = os.path.join( bu_paths.kind_dir( 'data' ), MIGRATION_RECORD )
    try:
        with open( record, 'r', encoding='utf-8' ) as handle:
            folders = json.load( handle ).get( 'media_folders' ) or []
    except ( OSError, ValueError, AttributeError ):
        folders = []
    for folder in folders:
        if isinstance( folder, str ) and os.path.basename( os.path.normpath( folder ) ) in VIDEO_INPUT_FOLDER_NAMES:
            seeds.append( folder )
    unique, seen = [], set()
    for folder in seeds:
        key = os.path.normcase( bu_index.norm( folder ) )
        if key not in seen:
            seen.add( key )
            unique.append( bu_index.norm( folder ) )
    return unique


class Catalog:
    """
    The open catalog.

    on_change( items, reason ) is told about every change: items is a
    set of item hashes (or None for "anything may have changed"), reason
    one of scan, job, pin, external. busy() says whether a job runs.
    """

    def __init__( self, path=None, clock=None, logger=None, on_change=None, busy=None,
                  hasher=None, interval=None, settle=None, seed_folders=None ):
        import time
        self.logger = logger
        self.store = bu_store.Store( path, clock=clock or time.time, logger=logger )
        try:
            self.index = bu_index.Index( self.store, logger=logger )
            self.on_change = on_change or ( lambda items, reason: None )
            self.scanner = bu_scan.Scanner(
                self.index, hasher=hasher, busy=busy, on_change=self._changed, logger=logger,
                interval=bu_scan.DEFAULT_INTERVAL_SECONDS if interval is None else interval,
                settle=bu_scan.DEFAULT_SETTLE_SECONDS if settle is None else settle )
            if self.store.created:
                self._seed( default_seed_folders() if seed_folders is None else seed_folders )
            self._data_version = self.store.data_version()
        except BaseException:
            self.store.close()
            raise

    def _seed( self, folders ):
        for folder in folders:
            if os.path.isdir( folder ):
                try:
                    self.index.add_folder( folder, type_from_folder=True )
                except bu_store.CatalogError as err:
                    if self.logger is not None:
                        self.logger.info( "catalog: not watching %s (%s)"%( folder, err.params.get( 'reason' ) ) )

    def _changed( self, items, reason ):
        self._data_version = self.store.data_version()
        self.on_change( items, reason )

    def check_external( self ):
        """Tell on_change when another process (the CLI) wrote the file."""
        version = self.store.data_version()
        if version != self._data_version:
            self._data_version = version
            self.on_change( None, 'external' )

    def start( self ):
        self.scanner.start( watch_external=self.check_external )

    def close( self ):
        self.scanner.stop()
        self.store.close()

    # --- folders ------------------------------------------------------------

    def folders( self ):
        counts = { row[0]: row[1] for row in self.store.read(
            'SELECT folder_id, COUNT( DISTINCT hash ) FROM locations GROUP BY folder_id' ) }
        return [ self._folder_out( folder, counts.get( folder['id'], 0 ) ) for folder in self.index.folders() ]

    @staticmethod
    def _folder_out( folder, items ):
        return { 'folder': folder['id'], 'path': folder['path'],
                 'type_from_folder': bool( folder['type_from_folder'] ),
                 'available': os.path.isdir( folder['path'] ), 'items': items }

    def add_folder( self, path, type_from_folder=False ):
        folder, created = self.index.add_folder( path, type_from_folder )
        self.scanner.rescan()
        return self._folder_out( folder, 0 ), created

    def remove_folder( self, folder_id ):
        touched = self.index.remove_folder( folder_id )
        self._changed( touched, 'scan' )
        return touched

    # --- files --------------------------------------------------------------

    def add_files( self, paths ):
        return self.scanner.add_files( paths )

    def rescan( self ):
        self.scanner.rescan()

    def forget( self, digest ):
        touched = self.index.forget( digest )
        self._changed( touched, 'scan' )

    # --- reading ------------------------------------------------------------

    def _items( self ):
        """Every item with its locations, in one pass over the tables."""
        items = { row['hash']: dict( row ) for row in self.store.read( 'SELECT * FROM items' ) }
        for item in items.values():
            item['locations'] = []
            item['outputs'] = 0
            item['outputs_missing'] = 0
        for row in self.store.read( 'SELECT hash, path, folder_id, state FROM locations ORDER BY added_at, id' ):
            if row['hash'] in items:
                items[row['hash']]['locations'].append( dict( row ) )
        for row in self.store.read( 'SELECT hash, present, COUNT(*) FROM outputs GROUP BY hash, present' ):
            if row[0] in items:
                items[row[0]]['outputs' if row[1] else 'outputs_missing'] = row[2]
        return items

    def _shape( self, item, folders ):
        """One item as catalog.list sends it."""
        present = [ location for location in item['locations'] if location['state'] == bu_index.PRESENT ]
        shown_at = ( present or item['locations'] or [ { 'path': None, 'folder_id': None } ] )[0]
        roots = self._folder_roots( shown_at['path'], folders ) if shown_at['path'] else ()
        last = { 'profile': item['last_profile'], 'rule': item['last_rule'],
                 'params': bu_index.loads_params( item['last_rule_params'] ) }
        return {
            'item': item['hash'],
            'name': os.path.basename( shown_at['path'] ) if shown_at['path'] else item['name'],
            'kind': item['kind'],
            'size': item['size'],
            'status': item['status'],
            'availability': bu_index.availability( { location['state'] for location in item['locations'] } ),
            'copies': len( present ),
            'pin': item['type_pin'],
            'type': bu_types.shown_type( shown_at['path'], item['type_pin'], roots, last ),
            'outputs': item['outputs'],
            'outputs_missing': item['outputs_missing'],
            'added_at': item['added_at'],
        }

    @staticmethod
    def _folder_roots( path, folders ):
        holding = [ folder for folder in folders
                    if folder['type_from_folder'] and bu_index.is_within( path, folder['path'] ) ]
        holding.sort( key=lambda folder: len( folder['path'] ), reverse=True )
        return tuple( folder['path'] for folder in holding )

    def folder_roots_for( self, path ):
        """The watched roots whose folder rule applies to path, deepest first."""
        return self._folder_roots( path, self.index.folders() )

    def list( self, filters=None, sort='name', descending=False, offset=0, limit=LIST_LIMIT_DEFAULT ):
        filters = filters or {}
        folders = self.index.folders()
        pairs = [ ( self._shape( item, folders ), { location['folder_id'] for location in item['locations'] } )
                  for item in self._items().values() if item['locations'] ]
        reclaimable = sum( item['size'] * ( item['copies'] - 1 ) for item, _ in pairs if item['copies'] > 1 )
        chosen = [ item for item, item_folders in pairs if _matches( item, filters, item_folders ) ]
        chosen.sort( key=lambda item: ( _sort_value( item, sort ), item['item'] ), reverse=descending )
        return { 'items': chosen[offset:offset + limit], 'total': len( chosen ),
                 'pending': self.scanner.pending_count(), 'reclaimable_bytes': reclaimable }

    def get( self, digest ):
        self.index.item_row( digest )   # raises catalog.no_such_item
        items = self._items()
        shaped = self._shape( items[digest], self.index.folders() )
        shaped['locations'] = [ { 'path': location['path'], 'folder': location['folder_id'],
                                  'availability': location['state'] }
                                for location in items[digest]['locations'] ]
        shaped['output_files'] = [ { 'path': output['path'], 'detection_key': output['detection_key'],
                                     'censor_key': output['censor_key'], 'encode_key': output['encode_key'],
                                     'present': bool( output['present'] ) }
                                   for output in self.index.outputs_of( digest ) ]
        return shaped


def _matches( item, filters, item_folders ):
    if filters.get( 'status' ) and item['status'] not in filters['status']:
        return False
    if filters.get( 'availability' ) and item['availability'] not in filters['availability']:
        return False
    if 'type' in filters and filters['type']:
        wanted = set( filters['type'] )
        if item['type']['profile'] not in wanted and not ( 'auto' in wanted and item['pin'] is None ):
            return False
    if filters.get( 'pinned' ) is not None and ( item['pin'] is not None ) != filters['pinned']:
        return False
    if filters.get( 'output' ) and _output_state( item ) not in filters['output']:
        return False
    if filters.get( 'duplicates' ) and item['copies'] < 2:
        return False
    if filters.get( 'folder' ) is not None and filters['folder'] not in item_folders:
        return False
    return True


def _output_state( item ):
    """present: a censored output is on disk; missing: every one was lost; none: never made."""
    if item['outputs']:
        return 'present'
    return 'missing' if item['outputs_missing'] else 'none'


def _sort_value( item, sort ):
    if sort == 'name':
        return item['name'].casefold()
    return item[sort]
