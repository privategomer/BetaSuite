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
    Compute a short, stable hash of a JSON-serialisable value.

    Serialised with sorted keys, so two dicts with identical contents in
    a different insertion order hash the same. That is what makes it
    safe to key a cache on a config dict.

    Args:
        to_hash: Any JSON-serialisable value.
        hash_len: Hex characters to keep, from the FRONT of the digest.

    Returns:
        A hash_len-character lowercase hex string.
    """
    canonical = json.dumps( to_hash, sort_keys=True, ensure_ascii=True, default=str )
    return hashlib.md5( canonical.encode('utf-8') ).hexdigest()[:hash_len]


# File-content hashes are memoised on (realpath, size, mtime_ns). Hashing
# a multi-GB source file costs real wall-clock and real IOPS, and it
# happens on EVERY run for EVERY file - including files whose censored
# output already exists and will be skipped, because the hash is part of
# the output filename and therefore has to be known before the skip
# check can run.
_file_hash_memo = None
_file_hash_memo_dirty = False
_file_hash_lock = threading.RLock()

# 4 MiB. The pre-2.1 value was 8192 bytes, which made hashing a 5GB file
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
    """
    global _file_hash_memo_dirty
    with _file_hash_lock:
        if not _file_hash_memo_dirty or _file_hash_memo is None:
            return
        try:
            write_json_plain( _file_hash_memo, betaconst.file_hash_cache_path )
            _file_hash_memo_dirty = False
        except Exception:
            # Best-effort: losing the memo costs time on the next run and
            # nothing else.
            pass


def md5_for_file( filename, length, use_memo=True ):
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

    Returns:
        A `length`-character hex string.
    """
    assert length <= 32, "md5_for_file length must be <= 32"

    memo_key = None
    if use_memo:
        try:
            stat = os.stat( filename )
            memo_key = "%s|%d|%d|%d"%(
                os.path.realpath( filename ), stat.st_size, stat.st_mtime_ns, length )
        except OSError:
            memo_key = None

    if memo_key is not None:
        with _file_hash_lock:
            memo = _load_file_hash_memo()
            cached = memo.get( memo_key )
        if cached:
            return cached

    file_hash = hashlib.md5()
    with open( filename, "rb" ) as fin:
        while True:
            chunk = fin.read( _FILE_HASH_CHUNK_BYTES )
            if not chunk:
                break
            file_hash.update( chunk )
    digest = file_hash.hexdigest()[32 - length:]

    if memo_key is not None:
        global _file_hash_memo_dirty
        with _file_hash_lock:
            _load_file_hash_memo()[ memo_key ] = digest
            _file_hash_memo_dirty = True

    return digest


# --- Atomic JSON I/O --------------------------------------------------------

def write_json( variable, filename ):
    """
    Gzip-compress and write a JSON-serialisable value, atomically.

    Writes to '<filename>.tmp' then os.replace()s it into position.
    This one rule is what makes os.path.exists() a trustworthy "is this
    cache complete" signal everywhere else in the codebase: a reader can
    never observe a half-written file, even if this process is killed
    mid-write.
    """
    dest_dir = os.path.dirname( filename )
    if dest_dir:
        os.makedirs( dest_dir, exist_ok=True )
    tmp_filename = filename + '.tmp'
    with gzip.open( tmp_filename, 'wt', encoding='UTF-8' ) as fout:
        json.dump( variable, fout )
    os.replace( tmp_filename, filename )


def write_json_plain( variable, filename ):
    """Uncompressed, atomic sibling of write_json, for small readable files."""
    dest_dir = os.path.dirname( filename )
    if dest_dir:
        os.makedirs( dest_dir, exist_ok=True )
    tmp_filename = filename + '.tmp'
    with open( tmp_filename, 'w', encoding='UTF-8' ) as fout:
        json.dump( variable, fout, indent=2, sort_keys=True, default=str )
    os.replace( tmp_filename, filename )


def read_json( filename ):
    """Read and decompress a value previously written by write_json."""
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


def clear_checkpoint( cache_filename ):
    """Delete cache_filename's checkpoint once the real cache is written."""
    path = _checkpoint_path( cache_filename )
    if os.path.exists( path ):
        try:
            os.remove( path )
        except OSError:
            pass
