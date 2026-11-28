"""
betautils_hash.py - Content hashing, atomic JSON I/O, detection checkpoints.

Small filesystem primitives the rest of BetaSuite builds on:

  1. Hashing          dictionary_hash / md5_for_file - short, stable
                      identifiers derived from config or file content.
  2. Atomic JSON      write_json / read_json - crash-safe gzip caching.
  3. Checkpointing    write/read/clear_checkpoint - resumable detection.

NOTE ON NAMING: this module hashes things. It does NOT decide what a
cache file is called. Every cache/output path in BetaSuite is built by
betautils_cache_paths.py, which is the single source of truth for
naming. Adding a second place that knows a filename format is how the
"backend name missing from one copy of the glob" class of bug happens
(see betautils_cache_paths.py's docstring).
"""

import gzip
import hashlib
import json
import os
import threading

import betaconst


# --- Hashing ----------------------------------------------------------------

def dictionary_hash( to_hash, hash_len ):
    """
    Short, stable hash of any JSON-serialisable value.

    Detection and render cache keys are built from this, so its output
    for a given input must never change. Keys are serialised in sorted
    order, which is what makes two dicts holding the same content in a
    different insertion order hash alike. Values JSON cannot represent
    fall back to their str().

    Takes the FIRST `hash_len` hex characters of the MD5 digest (unlike
    md5_for_file, which takes the last).

    Args:
        to_hash: Any JSON-serialisable value.
        hash_len: Number of hex characters to keep.

    Returns:
        Lowercase hex string of length hash_len.
    """
    serialised = json.dumps( to_hash, sort_keys=True, ensure_ascii=True, default=str )
    return hashlib.md5( serialised.encode( 'UTF-8' ) ).hexdigest()[:hash_len]


# File-content hashes are memoised on (realpath, size, mtime_ns). Hashing
# a multi-GB source file costs real wall-clock and real IOPS, and it
# happens on EVERY run for EVERY file - including files whose censored
# output already exists and will be skipped, because the hash is part of
# the output filename and therefore has to be known before the skip
# check can run.
_file_hash_memo = None
_file_hash_memo_dirty = False
_file_hash_lock = threading.RLock()

# 4 MiB. The earlier value was 8192 bytes, which made hashing a 5GB file
# ~640k Python-level read calls; the hash itself was never the bottleneck,
# the call overhead was.
_FILE_HASH_CHUNK_BYTES = 4 * 1024 * 1024


def _load_file_hash_memo():
    """Load the on-disk file-hash memo, tolerating a missing/corrupt file."""
    global _file_hash_memo
    if _file_hash_memo is not None:
        return _file_hash_memo
    _file_hash_memo = {}
    path = betaconst.file_hash_cache_path
    if os.path.exists( path ):
        try:
            with open( path, 'r', encoding='UTF-8' ) as fin:
                loaded = json.load( fin )
            if isinstance( loaded, dict ):
                _file_hash_memo = loaded
        except Exception:
            # A corrupt memo is not worth salvaging or complaining about;
            # it only ever costs a re-hash.
            _file_hash_memo = {}
    return _file_hash_memo


def flush_file_hash_memo():
    """
    Persist the file-hash memo if anything new was added.

    Called once at the end of a run rather than on every insert, so a
    library scan does not rewrite the memo hundreds of times.

    Merges with what is on disk first: the engine's catalog scanner and
    a command-line run can each hold the memo, and a plain rewrite by
    one would drop the entries the other flushed meanwhile, costing a
    full re-read of those files. Each process writes through its own
    temporary file, so two flushes never write into one.
    """
    global _file_hash_memo_dirty
    with _file_hash_lock:
        if not _file_hash_memo_dirty or _file_hash_memo is None:
            return
        try:
            path = betaconst.file_hash_cache_path
            try:
                with open( path, 'r', encoding='UTF-8' ) as fin:
                    on_disk = json.load( fin )
            except Exception:
                on_disk = {}
            if isinstance( on_disk, dict ):
                for key, value in on_disk.items():
                    _file_hash_memo.setdefault( key, value )
            _write_json_plain_via( _file_hash_memo, path,
                                   '%s.%d.%d.tmp'%( path, os.getpid(), threading.get_ident() ) )
            _file_hash_memo_dirty = False
        except Exception:
            # Best-effort: losing the memo costs time on the next run and
            # nothing else.
            pass


class HashStopped( Exception ):
    """md5_for_file gave up because should_stop() said so; nothing was memoised."""


def md5_for_file( filename, length, use_memo=True, should_stop=None ):
    """
    Short content hash of a file, streamed so the file is never held in
    memory, and memoised on (path, size, mtime) so repeat runs are free.

    Takes the LAST `length` hex characters of the digest. That differs
    from dictionary_hash (which takes the first), and it is load-bearing:
    every cache key and output filename ever written depends on this
    exact behaviour. Do not "fix" the inconsistency.

    Args:
        filename: Path to hash.
        length: Trailing hex characters to keep (<= 32).
        use_memo: False forces a real re-read, for the memo's own tests
            and for verifying a suspected stale memo entry.
        should_stop: Optional callable checked before each chunk; when it
            answers true the read stops with HashStopped. The catalog's
            scanner passes one, so a file it is hashing in the background
            lets go of the disk as soon as a run starts.

    Returns:
        A `length`-character hex string.
    """
    global _file_hash_memo_dirty
    assert length <= 32, "md5_for_file length must be <= 32"

    memo_key = None
    identity_key = None
    if use_memo:
        try:
            stat = os.stat( filename )
            memo_key = "%s|%d|%d|%d"%(
                os.path.realpath( filename ), stat.st_size, stat.st_mtime_ns, length )
            # The same file after a move. mv within one filesystem keeps
            # the inode and the mtime, so this survives sorting a source
            # tree, staging files into uncensored_vids/ and archiving them
            # to completed/, each of which used to cost a full re-read:
            # 4m39s of one betabench run after ~70 files were sorted. A
            # copy gets a new inode and so is re-hashed, as it should be.
            # Kept alongside the path key, not instead of it, so memo
            # entries written before this existed still hit.
            identity_key = "inode:%d|%d|%d|%d|%d"%(
                stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, length )
        except OSError:
            memo_key = None

    if memo_key is not None:
        with _file_hash_lock:
            memo = _load_file_hash_memo()
            cached = memo.get( memo_key ) or memo.get( identity_key )
            if cached and memo_key not in memo:
                memo[ memo_key ] = cached
                _file_hash_memo_dirty = True
        if cached:
            return cached

    file_hash = hashlib.md5()
    with open( filename, "rb" ) as fin:
        while True:
            if should_stop is not None and should_stop():
                raise HashStopped( filename )
            chunk = fin.read( _FILE_HASH_CHUNK_BYTES )
            if not chunk:
                break
            file_hash.update( chunk )
    digest = file_hash.hexdigest()[32 - length:]

    if memo_key is not None:
        with _file_hash_lock:
            memo = _load_file_hash_memo()
            memo[ memo_key ] = digest
            if identity_key is not None:
                memo[ identity_key ] = digest
            _file_hash_memo_dirty = True

    return digest


def sha256_for_file( filename, use_memo=True, should_stop=None ):
    """
    Full SHA-256 hex digest of a file, streamed, memoised like
    md5_for_file on (path, size, mtime) in the same memo under its own
    'sha256|' keys. Used for model files, whose pinned hashes are SHA-256.
    No inode key: a model copied into place must be hashed again.

    Raises HashStopped when should_stop() answers true; OSError when the
    file cannot be read.
    """
    global _file_hash_memo_dirty
    memo_key = None
    if use_memo:
        try:
            stat = os.stat( filename )
            memo_key = "sha256|%s|%d|%d"%( os.path.realpath( filename ), stat.st_size, stat.st_mtime_ns )
        except OSError:
            memo_key = None
    if memo_key is not None:
        with _file_hash_lock:
            cached = _load_file_hash_memo().get( memo_key )
        if cached:
            return cached

    file_hash = hashlib.sha256()
    with open( filename, "rb" ) as fin:
        while True:
            if should_stop is not None and should_stop():
                raise HashStopped( filename )
            chunk = fin.read( _FILE_HASH_CHUNK_BYTES )
            if not chunk:
                break
            file_hash.update( chunk )
    digest = file_hash.hexdigest()

    if memo_key is not None:
        with _file_hash_lock:
            _load_file_hash_memo()[ memo_key ] = digest
            _file_hash_memo_dirty = True
    return digest


# --- Atomic JSON I/O --------------------------------------------------------

def write_json( variable, filename ):
    """
    Write a value as gzip-compressed JSON, atomically.

    The value goes to a temporary sibling first and is moved onto
    `filename` only once complete, so a reader can trust that a cache
    file which exists is whole, even if the writer was killed midway.
    No temporary file is left behind, on success or on failure.

    Args:
        variable: Any JSON-serialisable value.
        filename: Destination path; its folder is created if missing.
    """
    dest_dir = os.path.dirname( filename )
    if dest_dir:
        os.makedirs( dest_dir, exist_ok=True )
    tmp_filename = filename + '.tmp'
    try:
        with gzip.open( tmp_filename, 'wt', encoding='UTF-8' ) as fout:
            json.dump( variable, fout )
        os.replace( tmp_filename, filename )
    except BaseException:
        try:
            os.remove( tmp_filename )
        except OSError:
            pass
        raise


def write_json_plain( variable, filename ):
    """Uncompressed, atomic sibling of write_json, for small readable files."""
    _write_json_plain_via( variable, filename, filename + '.tmp' )


def _write_json_plain_via( variable, filename, tmp_filename ):
    dest_dir = os.path.dirname( filename )
    if dest_dir:
        os.makedirs( dest_dir, exist_ok=True )
    try:
        with open( tmp_filename, 'w', encoding='UTF-8' ) as fout:
            json.dump( variable, fout, indent=2, sort_keys=True, default=str )
        os.replace( tmp_filename, filename )
    except BaseException:
        try:
            os.remove( tmp_filename )
        except OSError:
            pass
        raise


def read_json( filename ):
    """Read a value previously written by write_json."""
    with gzip.open( filename, 'rt', encoding='UTF-8' ) as fin:
        return json.load( fin )


def read_json_plain( filename ):
    """Read a value previously written by write_json_plain."""
    with open( filename, 'r', encoding='UTF-8' ) as fin:
        return json.load( fin )


# --- Detection checkpointing -------------------------------------------
#
# A detection pass over a long video can take hours. Without this it
# wrote its cache exactly once, at the very end, so a kill at 99% lost
# everything. These write progress to a SEPARATE '<cache>.checkpoint'
# file; write_json's atomic rename still guarantees the trusted cache
# path only ever holds a complete result.

def _checkpoint_path( cache_filename ):
    """Derive a detection cache's in-progress checkpoint path."""
    return cache_filename + '.checkpoint'


def write_checkpoint( raw_boxes, next_sample_index, next_t, cache_filename ):
    """
    Persist in-progress detection results.

    Args:
        raw_boxes: Raw detections accumulated so far.
        next_sample_index: Sample index to resume at (NOT a video frame
            index - see betautils_video.sample_frame_index).
        next_t: Timestamp of that sample, in seconds.
        cache_filename: The final cache path this checkpoint stands in
            for. Only used to derive the checkpoint path.
    """
    write_json( {
        'boxes': raw_boxes,
        'next_frame': next_sample_index,   # historical key name, kept for compatibility
        'next_t': next_t,
    }, _checkpoint_path( cache_filename ) )


def read_checkpoint( cache_filename ):
    """
    Load a checkpoint for cache_filename, or None.

    A corrupt/truncated checkpoint is discarded and treated as "no
    checkpoint", so the caller simply restarts that size cleanly.
    """
    path = _checkpoint_path( cache_filename )
    if not os.path.exists( path ):
        return None
    try:
        return read_json( path )
    except Exception:
        try:
            os.remove( path )
        except OSError:
            pass
        return None


def has_checkpoint( cache_filename ):
    """Whether cache_filename has a checkpoint on disk, without reading it."""
    return os.path.exists( _checkpoint_path( cache_filename ) )


def clear_checkpoint( cache_filename ):
    """Delete cache_filename's checkpoint once the real cache is written."""
    path = _checkpoint_path( cache_filename )
    if os.path.exists( path ):
        try:
            os.remove( path )
        except OSError:
            pass
