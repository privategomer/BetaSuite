"""
betautils_protocol.py - The engine's wire protocol: framing, envelope,
error codes.

The desktop UI starts the engine as a child process and talks to it in
JSON lines: one JSON object per line, commands on the engine's stdin,
results and events on its stdout. This module is that protocol and
nothing else. It does no I/O of its own (it reads and writes the binary
streams it is handed), imports only the standard library, and holds no
engine state, so a stub client or a test can use it as is.

    LineReader       bytes in, one line at a time, bounded memory
    parse_command()  one line -> Command, or ProtocolError with a code
    ok_result()      the three outgoing message shapes
    error_result()
    event()          ( rejected() for a line whose id could not be read )
    MessageWriter    one message -> one flushed line, thread-safe

Envelope, every message:
    {"v": 1, "kind": "cmd" | "result" | "event", "id": "...",
     "name": "dotted.name", "data": {...}}
    id appears on commands and results only, and a result's id is
    always the id of the command it answers. A result also carries
    "ok": true with "data", or "ok": false with
    "error": {"code": "...", "params": {...}}.

    A line too broken to read an id from (bad UTF-8, bad JSON, too long,
    or an envelope with no usable id) gets no result. The engine sends a
    protocol.rejected event instead, naming the line by its number: every
    line the engine has read counts, blank ones included, from 1. The
    client knows what it wrote on that line.

The engine never sends sentences. An error is a code from ERRORS with
exactly the parameters ERRORS lists for it; the UI owns the wording.
Both sides ignore fields they do not know, and v changes only on a
breaking change. The machine-readable schemas for all of this live in
protocol/v1/ and tests hold this module and those files in step.
"""

import collections
import json
import re
import threading


PROTOCOL_VERSION = 1
SUPPORTED_VERSIONS = ( 1, )

# Longest line accepted, in bytes, not counting its line ending. Commands are
# small; 1 MiB leaves room for a long list of file paths in one job.
MAX_LINE_BYTES = 1024*1024
MAX_ID_CHARS = 128

# Command and event names: lower-case words joined by dots, as in
# protocol/v1/envelope.schema.json.
NAME_PATTERN = re.compile( r'[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*' )

# An oversized line is thrown away in pieces this big, so it never sits
# in memory whole.
_DISCARD_CHUNK = 64*1024


# --- Error codes --------------------------------------------------------------

# code -> the parameter names it always carries, no more and no fewer.
# protocol/v1/errors.schema.json describes the same codes with types.
ERRORS = {
    'protocol.bad_encoding':        (),
    'protocol.bad_json':            (),
    'protocol.line_too_long':       ( 'limit_bytes', 'line_bytes' ),
    'protocol.bad_envelope':        ( 'field', 'reason' ),
    'protocol.unsupported_version': ( 'got', 'supported' ),
    'protocol.hello_required':      ( 'name', ),
    'command.unknown':              ( 'name', ),
    'command.bad_params':           ( 'field', 'reason' ),
    'engine.internal':              ( 'exception_type', ),
    # config.*: a value's problem (alone, or inside config.invalid)
    'config.bad_type':              ( 'path', 'expected' ),
    'config.out_of_range':          ( 'path', 'minimum', 'maximum', 'min_open', 'max_open', 'odd' ),
    'config.bad_choice':            ( 'path', 'choices' ),
    'config.rule':                  ( 'rule', 'path' ),
    # config.*: why a command changed nothing
    'config.invalid':               ( 'problems', ),
    'config.unknown_key':           ( 'path', 'suggestion' ),
    'config.not_settable':          ( 'path', 'setting' ),
    'config.local_off':             (),
    'config.local_unparsable':      (),
    'config.local_unwritable':      ( 'reason', ),
    'run.locked':                   ( 'program', 'pid', 'started_at' ),
    'config.busy':                  ( 'job', ),
    # job.*: why a job command was refused
    'job.unknown':                  ( 'job', ),
    'job.bad_state':                ( 'job', 'state' ),
    'job.no_such_file':             ( 'index', ),
    'job.nothing_to_render':        (),
    # preview.look
    'preview.unknown_art':          ( 'style', 'item' ),
    'preview.bad_image':            (),
    'engine.stopping':              (),
    # job.error events, and the code of a 'stopped' job.state
    'file.failed':                  ( 'exception_type', 'stage' ),
    'disk.full':                    ( 'free_bytes', ),
    'model.load_failed':            ( 'exception_type', ),
    # models.*
    'model.unknown':                ( 'model', ),
    'model.hash_mismatch':          ( 'model', ),
    'model.download_failed':        ( 'model', 'reason' ),
    'model.unknown_file':           (),
    'model.in_use':                 ( 'model', ),
    'model.busy':                   ( 'model', ),
    'model.not_installed':          ( 'model', ),
    # catalog.*
    'catalog.unavailable':          ( 'reason', ),
    'catalog.too_new':              ( 'version', 'supported' ),
    'catalog.no_such_item':         ( 'item', ),
    'catalog.no_such_folder':       ( 'folder', ),
    'catalog.bad_folder':           ( 'reason', ),
    'catalog.bad_type':             ( 'type', ),
    'catalog.item_busy':            ( 'item', 'job' ),
    'catalog.missing':              ( 'item', ),
    'catalog.thumbnail_failed':     ( 'item', 'reason' ),
    'catalog.no_such_file':         (),
    # feedback.*
    'feedback.not_a_result':        (),
    'feedback.no_such_record':      ( 'id', ),
    # perf.* and Optimize
    'perf.busy':                    ( 'job', ),
    'perf.optimize_running':        (),
    'perf.machine_busy':            (),
    'perf.no_proposal':             (),
    'optimize.failed':              ( 'exit_code', ),
    # storage.*
    'storage.unknown_category':     ( 'category', ),
    'storage.not_cleanable':        ( 'category', 'policy' ),
    'storage.needs_confirm':        ( 'category', 'policy' ),
    'storage.busy':                 ( 'category', ),
    'storage.trash_failed':         ( 'category', ),
}

# Values of the 'reason' parameter.
REASON_MISSING = 'missing'
REASON_TYPE = 'type'
REASON_VALUE = 'value'


def error( code, **params ):
    """
    The error object for a result: {"code": code, "params": params}.

    Raises ValueError for a code that is not in ERRORS or parameters
    that do not match its list exactly. That is a bug in the engine, not
    in the client, and failing loudly keeps the registry honest.
    """
    expected = ERRORS.get( code )
    if expected is None:
        raise ValueError( "unregistered error code %r"%( code, ) )
    if set( params ) != set( expected ):
        raise ValueError( "error %s takes params %s, got %s"%(
            code, sorted( expected ), sorted( params ) ) )
    return { 'code': code, 'params': dict( params ) }


class ProtocolError( Exception ):
    """
    A request the engine answers with an error result.

    Carries the error object and whatever of the request's id and name
    could be read, so the result can still be matched to its command.
    """

    def __init__( self, code, params=None, request_id=None, request_name=None ):
        self.error = error( code, **( params or {} ) )
        self.request_id = request_id
        self.request_name = request_name
        super().__init__( code )

    @property
    def code( self ):
        return self.error['code']


# --- Framing ------------------------------------------------------------------

class LineTooLong( Exception ):
    """A line went past the limit. It has been read and discarded."""

    def __init__( self, line_bytes, limit_bytes ):
        self.line_bytes = line_bytes
        self.limit_bytes = limit_bytes
        super().__init__( line_bytes )

    def as_protocol_error( self ):
        return ProtocolError( 'protocol.line_too_long',
                              { 'limit_bytes': self.limit_bytes, 'line_bytes': self.line_bytes } )


class LineReader:
    """
    Lines from a binary stream, one at a time.

    A message split across several writes and several messages in one
    write both come out as whole lines, because the stream is buffered
    and read up to each newline. A trailing carriage return is dropped
    (a Windows client may send CRLF) and blank lines are skipped. A last
    line with no newline before end of stream still counts.

    Memory is bounded: an oversized line is read and dropped in chunks
    up to its newline, then reported with LineTooLong, and the next line
    reads normally.
    """

    def __init__( self, stream, limit=MAX_LINE_BYTES ):
        self._stream = stream
        self._limit = limit
        # Lines read so far, blank and oversized ones included.
        self.line_number = 0

    def read_line( self ):
        """
        The next non-blank line without its line ending, or None at end
        of stream.

        Raises LineTooLong for a line longer than the limit.
        """
        while True:
            # Room for the limit plus CRLF, so the line ending never counts.
            line = self._stream.readline( self._limit + 2 )
            if not line:
                return None
            self.line_number += 1
            if line.endswith( b'\n' ):
                line = line[:-1]
            elif len( line ) == self._limit + 2:
                raise LineTooLong( self._discard_rest( line ), self._limit )
            if line.endswith( b'\r' ):
                line = line[:-1]
            if len( line ) > self._limit:
                raise LineTooLong( len( line ), self._limit )
            if line.strip():
                return line

    def _discard_rest( self, start ):
        # The size of the whole line, not counting its line ending.
        counted, tail = len( start ), start[-1:]
        while True:
            chunk = self._stream.readline( _DISCARD_CHUNK )
            if chunk.endswith( b'\n' ):
                chunk = chunk[:-1]
                counted += len( chunk )
                tail = chunk[-1:] or tail
                break
            if not chunk:
                break
            counted += len( chunk )
            tail = chunk[-1:]
        return counted - 1 if tail == b'\r' else counted


# --- Envelope -----------------------------------------------------------------

Command = collections.namedtuple( 'Command', 'id name data' )


def _reject_constant( name ):
    # json.loads accepts NaN and Infinity by default; JSON does not.
    raise ValueError( name )


def _readable_id( value ):
    if isinstance( value, str ) and 0 < len( value ) <= MAX_ID_CHARS:
        return value
    return None


def _readable_name( value ):
    return value if isinstance( value, str ) and value else None


def _is_int( value ):
    return isinstance( value, int ) and not isinstance( value, bool )


def parse_command( line ):
    """
    One line from the client -> Command( id, name, data ).

    Raises ProtocolError, carrying the id and name when they could be
    read, for anything that is not a well-formed v1 command. Fields the
    envelope does not define are ignored. A command with no data gets {}.
    """
    try:
        text = line.decode( 'utf-8' )
    except UnicodeDecodeError:
        raise ProtocolError( 'protocol.bad_encoding' ) from None
    try:
        message = json.loads( text, parse_constant=_reject_constant )
    except ( ValueError, RecursionError ):
        raise ProtocolError( 'protocol.bad_json' ) from None
    if not isinstance( message, dict ):
        raise ProtocolError( 'protocol.bad_envelope', { 'field': 'message', 'reason': REASON_TYPE } )

    request_id = _readable_id( message.get( 'id' ) )
    name = _readable_name( message.get( 'name' ) )

    def bad( field, reason ):
        return ProtocolError( 'protocol.bad_envelope', { 'field': field, 'reason': reason }, request_id, name )

    if 'v' not in message:
        raise bad( 'v', REASON_MISSING )
    if not _is_int( message['v'] ):
        raise bad( 'v', REASON_TYPE )
    if message['v'] not in SUPPORTED_VERSIONS:
        raise ProtocolError( 'protocol.unsupported_version',
                             { 'got': message['v'], 'supported': list( SUPPORTED_VERSIONS ) },
                             request_id, name )

    for field in ( 'kind', 'id', 'name' ):
        if field not in message:
            raise bad( field, REASON_MISSING )
        if not isinstance( message[field], str ):
            raise bad( field, REASON_TYPE )
    if message['kind'] != 'cmd':
        raise bad( 'kind', REASON_VALUE )
    if request_id is None:
        raise bad( 'id', REASON_VALUE )
    if name is None or not NAME_PATTERN.fullmatch( name ):
        raise bad( 'name', REASON_VALUE )

    data = message.get( 'data', {} )
    if not isinstance( data, dict ):
        raise bad( 'data', REASON_TYPE )
    return Command( request_id, name, data )


def command( request_id, name, data=None ):
    """A command message, as a client sends it."""
    return { 'v': PROTOCOL_VERSION, 'kind': 'cmd', 'id': request_id, 'name': name,
             'data': {} if data is None else data }


def ok_result( request_id, name, data ):
    """A successful result for the command with this id and name."""
    return { 'v': PROTOCOL_VERSION, 'kind': 'result', 'id': request_id, 'name': name,
             'ok': True, 'data': data }


def error_result( request_id, name, error_object ):
    """
    A failed result for the command with this id. name is None when the
    command's name could not be read. A line with no readable id gets
    rejected() instead, never a result.
    """
    if request_id is None:
        raise ValueError( "a result needs the id of the command it answers; use rejected()" )
    return { 'v': PROTOCOL_VERSION, 'kind': 'result', 'id': request_id, 'name': name,
             'ok': False, 'error': error_object }


def event( name, data ):
    """An event: engine-initiated, no id."""
    return { 'v': PROTOCOL_VERSION, 'kind': 'event', 'name': name, 'data': data }


REJECTED_EVENT = 'protocol.rejected'


def rejected( line_number, error_object ):
    """The event for a line whose id could not be read: which line, and why."""
    return event( REJECTED_EVENT, { 'line': line_number, 'error': error_object } )


def choose_version( client_versions ):
    """The highest protocol version both sides speak, or None."""
    common = [ v for v in client_versions if _is_int( v ) and v in SUPPORTED_VERSIONS ]
    return max( common ) if common else None


# --- Writing ------------------------------------------------------------------

def encode( message ):
    """
    One message -> one line of bytes, newline included.

    ASCII-only output, so a path holding bytes that are not UTF-8 still
    makes valid JSON, and NaN or infinity raise ValueError rather than
    produce JSON the client cannot parse. Anything that is not JSON data
    raises TypeError.
    """
    text = json.dumps( message, ensure_ascii=True, allow_nan=False, separators=( ',', ':' ) )
    return text.encode( 'ascii' ) + b'\n'


class MessageWriter:
    """
    Writes messages to a binary stream, one flushed line each.

    Thread-safe: a message is encoded first, then written and flushed
    under a lock, so lines from different threads never interleave.
    Encoding errors (ValueError, TypeError) are raised before anything
    is written.
    """

    def __init__( self, stream ):
        self._stream = stream
        self._lock = threading.Lock()

    def send( self, message ):
        """Write one message. Returns the bytes written."""
        line = encode( message )
        with self._lock:
            self._stream.write( line )
            self._stream.flush()
        return line
