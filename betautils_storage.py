"""
betautils_storage.py - what the app's files use on disk, and safe
clean-up (the Storage view: storage.usage, storage.clean).

Every category is a kind of thing the app keeps, with where it is, how
much it uses, and what removing it costs:

  safe       made again by itself when needed, quickly: look previews,
             thumbnails, transcodes, partial or set-aside model files,
             the setup download cache, a previous cache folder
  costly     made again by itself when needed, but slowly: detections
             (every video is detected again, hours of GPU time for a
             library) and shot cuts
  resume     unfinished renders: removing them means those parts are
             rendered again
  shared     not the app's alone (a download cache other tools use too):
             listed, never removed
  kept       listed only: outputs, models in use, the catalog and
             feedback, settings, logs, stats, run keys, file hashes.
             Models are removed on the Models page; outputs are the
             user's to move or delete

storage.clean removes only safe, costly and resume categories, and only
when asked by name. costly and resume need confirm: true, so a client
cannot clear hours of work by accident. While a job holds the queue,
every category the running pipeline writes is refused (storage.busy).

Sizes are measured by walking the folders (no paths are reported, only
categories, bytes and file counts).
"""

import os
import shutil

SAFE = 'safe'
COSTLY = 'costly'
RESUME = 'resume'
SHARED = 'shared'
KEPT = 'kept'

POLICIES = ( SAFE, COSTLY, RESUME, SHARED, KEPT )
CLEANABLE = ( SAFE, COSTLY, RESUME )
NEEDS_CONFIRM = ( COSTLY, RESUME )

# The folder the first-run setup keeps its downloads in (uv's cache),
# under the cache kind so it is the app's own. setup.sh and setup.ps1
# point UV_CACHE_DIR here.
DOWNLOAD_CACHE = 'setup-downloads'

# Categories the running pipeline writes; refused while a job runs.
WRITTEN_BY_RUNS = ( 'detections', 'shot_cuts', 'transcodes', 'unfinished_renders', 'thumbnails' )


class Category:

    def __init__( self, id, kind, policy, paths, match=None ):
        self.id = id
        self.kind = kind            # the kind of folder it lives in (betautils_paths)
        self.policy = policy
        self.paths = list( paths )  # folders or files
        self.match = match          # match( name ) -> bool for files to count in a folder; None counts all


def _cache( *parts ):
    import betautils_paths as bu_paths
    return os.path.join( bu_paths.kind_dir( 'cache' ), *parts )


def _is_partial_output( name ):
    return '.video.' in name or '.tmp.' in name


def _is_model_leftover( name ):
    return name.endswith( '.part' ) or name.endswith( '.bad' )


def categories( previous_cache=None, shared_download_cache=None ):
    """
    Every category, in the order the Storage view lists them.

    previous_cache: a cache folder the app used before cache_dir moved
        it, or None. shared_download_cache: a download cache an older
        setup left in the shared default place (uv's own), or None.
    """
    import betaconst
    import betautils_catalog_store as bu_catalog_store
    import betautils_catalog_thumbs as bu_thumbs
    import betautils_paths as bu_paths
    found = [
        Category( 'detections', 'cache', COSTLY, [ betaconst.vid_hash_dir, betaconst.pic_hash_dir ] ),
        Category( 'shot_cuts', 'cache', COSTLY, [ betaconst.shot_cut_dir ] ),
        Category( 'transcodes', 'cache', SAFE, [ betaconst.transcode_cache_dir ] ),
        Category( 'look_previews', 'cache', SAFE, [ betaconst.look_preview_dir ] ),
        Category( 'thumbnails', 'cache', SAFE, [ bu_thumbs.folder() ] ),
        Category( 'download_cache', 'cache', SAFE, [ _cache( DOWNLOAD_CACHE ) ] ),
        Category( 'unfinished_renders', 'outputs', RESUME, [ betaconst.video_path_censored ],
                  match=_is_partial_output ),
        Category( 'model_leftovers', 'data', SAFE, [ _model_dir() ], match=_is_model_leftover ),
        Category( 'models', 'data', KEPT, [ _model_dir() ], match=lambda name: not _is_model_leftover( name ) ),
        Category( 'outputs', 'outputs', KEPT, [ betaconst.video_path_censored, betaconst.picture_path_censored ],
                  match=lambda name: not _is_partial_output( name ) ),
        Category( 'catalog', 'data', KEPT, [ bu_catalog_store.default_path() ] ),
        Category( 'logs', 'state', KEPT, [ os.path.join( bu_paths.kind_dir( 'state' ), 'logs' ) ] ),
        Category( 'stats', 'state', KEPT, [ os.path.join( bu_paths.kind_dir( 'state' ), 'stats' ) ] ),
        Category( 'benchmarks', 'state', SAFE, [ betaconst.benchmark_dir ] ),
        Category( 'run_keys', 'cache', KEPT, [ betaconst.run_key_dir, betaconst.file_hash_cache_path ] ),
    ]
    if previous_cache:
        found.append( Category( 'previous_cache', 'cache', SAFE, [ previous_cache ] ) )
    if shared_download_cache:
        found.append( Category( 'shared_download_cache', 'cache', SHARED, [ shared_download_cache ] ) )
    return found


def _model_dir():
    import betautils_models as bu_models
    return bu_models.model_dir()


def measure( path, match=None ):
    """( bytes, files ) under path (a file or a folder), counting files match() accepts."""
    if not os.path.lexists( path ):
        return 0, 0
    if os.path.isfile( path ) or os.path.islink( path ):
        if match is not None and not match( os.path.basename( path ) ):
            return 0, 0
        try:
            return os.lstat( path ).st_size, 1
        except OSError:
            return 0, 0
    total = count = 0
    # os.walk never follows a link to a folder (followlinks is off), so
    # nothing outside the folder is counted or removed.
    for root, _dirs, files in os.walk( path ):
        for name in files:
            if match is not None and not match( name ):
                continue
            try:
                total += os.lstat( os.path.join( root, name ) ).st_size
                count += 1
            except OSError:
                continue
    return total, count


def usage( cats ):
    """[ { id, kind, policy, bytes, files } ] for each category."""
    rows = []
    for cat in cats:
        size = files = 0
        for path in cat.paths:
            found_size, found_files = measure( path, cat.match )
            size += found_size
            files += found_files
        rows.append( { 'id': cat.id, 'kind': cat.kind, 'policy': cat.policy, 'bytes': size, 'files': files } )
    return rows


def _remove( path, match ):
    """Remove what path holds that match() accepts; ( bytes freed, files removed )."""
    freed = removed = 0
    if not os.path.lexists( path ):
        return 0, 0
    if os.path.isfile( path ) or os.path.islink( path ):
        if match is None or match( os.path.basename( path ) ):
            size = os.lstat( path ).st_size
            os.remove( path )
            return size, 1
        return 0, 0
    if match is None:
        size, files = measure( path )
        shutil.rmtree( path )
        return size, files
    for root, _dirs, files in os.walk( path ):
        for name in files:
            if not match( name ):
                continue
            full = os.path.join( root, name )
            try:
                size = os.lstat( full ).st_size
                os.remove( full )
            except OSError:
                continue
            freed += size
            removed += 1
    return freed, removed


class StorageError( Exception ):

    def __init__( self, code, params ):
        super().__init__( code )
        self.code = code
        self.params = params


def clean( cats, ids, confirm=False, busy=False ):
    """
    Remove the named categories. Returns [ { id, bytes, files } ] freed.

    Raises StorageError:
      storage.unknown_category   an id no category has
      storage.not_cleanable      a shared or kept category
      storage.needs_confirm      a costly or resume category without confirm
      storage.busy               a category the running pipeline writes, while a job runs
    Nothing is removed unless every named category may be.
    """
    by_id = { cat.id: cat for cat in cats }
    chosen = []
    for category_id in ids:
        cat = by_id.get( category_id )
        if cat is None:
            raise StorageError( 'storage.unknown_category', { 'category': category_id } )
        if cat.policy not in CLEANABLE:
            raise StorageError( 'storage.not_cleanable', { 'category': category_id, 'policy': cat.policy } )
        if cat.policy in NEEDS_CONFIRM and not confirm:
            raise StorageError( 'storage.needs_confirm', { 'category': category_id, 'policy': cat.policy } )
        if busy and cat.id in WRITTEN_BY_RUNS:
            raise StorageError( 'storage.busy', { 'category': category_id } )
        chosen.append( cat )
    freed = []
    for cat in chosen:
        size = files = 0
        for path in cat.paths:
            got_size, got_files = _remove( path, cat.match )
            size += got_size
            files += got_files
        freed.append( { 'id': cat.id, 'bytes': size, 'files': files } )
    return freed
