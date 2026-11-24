"""
test_protocol.py - the engine's wire protocol (betautils_protocol):
framing, envelope parsing, error codes, outgoing messages, and the
protocol/v1 schemas kept in step with the code.

Everything here is in-process. test_serve.py drives the real engine in
a subprocess.
"""

import io
import json
import os
import threading
import time
import unittest
from unittest import mock

import betautils_protocol as bu_proto
from tests import protocol_schemas as schemas


def _code( line ):
    """The error code parse_command raises for line (bytes or str)."""
    if isinstance( line, str ):
        line = line.encode( 'utf-8' )
    with unittest.TestCase().assertRaises( bu_proto.ProtocolError ) as raised:
        bu_proto.parse_command( line )
    return raised.exception


def _cmd( **fields ):
    message = { 'v': 1, 'kind': 'cmd', 'id': 'a1', 'name': 'ping' }
    message.update( fields )
    return json.dumps( message ).encode( 'utf-8' )


def _without( field ):
    message = json.loads( _cmd() )
    del message[field]
    return json.dumps( message ).encode( 'utf-8' )


# One valid set of params for every code, for the schema round trip.
EXAMPLE_PARAMS = {
    'protocol.bad_encoding':        {},
    'protocol.bad_json':            {},
    'protocol.line_too_long':       { 'limit_bytes': 1048576, 'line_bytes': 2000000 },
    'protocol.bad_envelope':        { 'field': 'kind', 'reason': 'value' },
    'protocol.unsupported_version': { 'got': 2, 'supported': [ 1 ] },
    'protocol.hello_required':      { 'name': 'ping' },
    'command.unknown':              { 'name': 'no.such' },
    'command.bad_params':           { 'field': 'protocol', 'reason': 'type' },
    'engine.internal':              { 'exception_type': 'KeyError' },
    'config.bad_type':              { 'path': 'encode_crf', 'expected': 'int' },
    'config.out_of_range':          { 'path': 'encode_crf', 'minimum': 0, 'maximum': 51,
                                      'min_open': False, 'max_open': False, 'odd': False },
    'config.bad_choice':            { 'path': 'render_chunk_container', 'choices': [ 'mkv', 'mp4' ] },
    'config.rule':                  { 'rule': 'label.hysteresis_inverted',
                                      'path': 'detector_backend.nudenet_v3.item_overrides.exposed_breast' },
    'config.invalid':               { 'problems': [ { 'code': 'config.bad_type',
                                                      'params': { 'path': 'encode_crf', 'expected': 'int' } } ] },
    'config.unknown_key':           { 'path': 'render_worker', 'suggestion': 'render_workers' },
    'config.not_settable':          { 'path': 'item_overrides.exposed_breast.censor_style.0',
                                      'setting': 'item_overrides.exposed_breast.censor_style' },
    'config.local_off':             {},
    'config.local_unparsable':      {},
    'config.local_unwritable':      { 'reason': 'EACCES' },
}


class TestErrorRegistry( unittest.TestCase ):

    def test_a_registered_code_with_its_params_is_built( self ):
        self.assertEqual( bu_proto.error( 'command.unknown', name='x' ),
                          { 'code': 'command.unknown', 'params': { 'name': 'x' } } )

    def test_an_unregistered_code_is_refused( self ):
        with self.assertRaises( ValueError ):
            bu_proto.error( 'command.nope' )

    def test_missing_or_extra_params_are_refused( self ):
        with self.assertRaises( ValueError ):
            bu_proto.error( 'protocol.line_too_long', limit_bytes=1 )
        with self.assertRaises( ValueError ):
            bu_proto.error( 'command.unknown', name='x', message='a sentence' )

    def test_a_protocol_error_carries_the_request_it_answers( self ):
        err = bu_proto.ProtocolError( 'command.unknown', { 'name': 'no.such' }, 'a1', 'no.such' )
        self.assertEqual( ( err.request_id, err.request_name, err.code ), ( 'a1', 'no.such', 'command.unknown' ) )
        self.assertEqual( err.error['params'], { 'name': 'no.such' } )

    def test_the_example_params_cover_every_code( self ):
        self.assertEqual( set( EXAMPLE_PARAMS ), set( bu_proto.ERRORS ) )
        for code, params in EXAMPLE_PARAMS.items():
            bu_proto.error( code, **params )


class _RecordingStream( io.BytesIO ):
    """A BytesIO that records the length of every readline it returns."""

    def __init__( self, data ):
        super().__init__( data )
        self.returned = []

    def readline( self, size=-1 ):
        chunk = super().readline( size )
        self.returned.append( len( chunk ) )
        return chunk


def _lines( data, limit=bu_proto.MAX_LINE_BYTES ):
    reader = bu_proto.LineReader( io.BytesIO( data ), limit=limit )
    out = []
    while True:
        try:
            line = reader.read_line()
        except bu_proto.LineTooLong as err:
            out.append( ( 'too_long', err.line_bytes ) )
            continue
        if line is None:
            return out
        out.append( line )


class TestLineReader( unittest.TestCase ):

    def test_several_messages_in_one_write_come_out_one_per_line( self ):
        self.assertEqual( _lines( b'{"a":1}\n{"b":2}\n{"c":3}\n' ),
                          [ b'{"a":1}', b'{"b":2}', b'{"c":3}' ] )

    def test_a_message_split_across_writes_comes_out_whole( self ):
        read_fd, write_fd = os.pipe()
        message = b'{"v":1,"kind":"cmd","id":"a1","name":"ping"}\n'

        def trickle():
            with os.fdopen( write_fd, 'wb', buffering=0 ) as out:
                for i in range( len( message ) ):
                    out.write( message[i:i + 1] )
                out.write( message[:10] )
                out.write( message[10:] )

        writer = threading.Thread( target=trickle )
        writer.start()
        with os.fdopen( read_fd, 'rb' ) as stream:
            reader = bu_proto.LineReader( stream )
            got = [ reader.read_line(), reader.read_line(), reader.read_line() ]
        writer.join()
        self.assertEqual( got, [ message[:-1], message[:-1], None ] )

    def test_crlf_and_blank_lines( self ):
        self.assertEqual( _lines( b'a\r\n\n  \r\n\t\nb\n' ), [ b'a', b'b' ] )

    def test_only_the_line_ending_carriage_return_is_dropped( self ):
        self.assertEqual( _lines( b'a\r\r\n' ), [ b'a\r' ] )

    def test_a_last_line_without_newline_still_counts( self ):
        self.assertEqual( _lines( b'a\nb' ), [ b'a', b'b' ] )

    def test_end_of_stream_is_none( self ):
        self.assertEqual( _lines( b'' ), [] )

    def test_a_line_exactly_at_the_limit_is_accepted( self ):
        self.assertEqual( _lines( b'12345678\n', limit=8 ), [ b'12345678' ] )

    def test_a_crlf_line_exactly_at_the_limit_is_accepted( self ):
        self.assertEqual( _lines( b'12345678\r\nok\n', limit=8 ), [ b'12345678', b'ok' ] )

    def test_the_line_ending_never_counts_toward_the_size( self ):
        self.assertEqual( _lines( b'123456789\r\nok\n', limit=8 ), [ ( 'too_long', 9 ), b'ok' ] )
        self.assertEqual( _lines( b'x' * 30 + b'\r\nok\n', limit=8 ), [ ( 'too_long', 30 ), b'ok' ] )
        self.assertEqual( _lines( b'x' * 30 + b'\nok\n', limit=8 ), [ ( 'too_long', 30 ), b'ok' ] )
        with mock.patch.object( bu_proto, '_DISCARD_CHUNK', 4 ):
            # A CR that ends a discard chunk, then the newline on its own.
            self.assertEqual( _lines( b'x' * 13 + b'\r\nok\n', limit=8 ), [ ( 'too_long', 13 ), b'ok' ] )

    def test_one_byte_over_the_limit_is_refused_and_the_next_line_reads( self ):
        self.assertEqual( _lines( b'123456789\nok\n', limit=8 ), [ ( 'too_long', 9 ), b'ok' ] )

    def test_an_oversized_last_line_without_newline_is_refused( self ):
        self.assertEqual( _lines( b'ok\n' + b'x' * 20, limit=8 ), [ b'ok', ( 'too_long', 20 ) ] )

    def test_an_oversized_line_is_discarded_in_bounded_chunks( self ):
        stream = _RecordingStream( b'x' * 5000 + b'\nok\n' )
        with mock.patch.object( bu_proto, '_DISCARD_CHUNK', 100 ):
            reader = bu_proto.LineReader( stream, limit=64 )
            with self.assertRaises( bu_proto.LineTooLong ) as raised:
                reader.read_line()
            self.assertEqual( reader.read_line(), b'ok' )
        self.assertEqual( ( raised.exception.line_bytes, raised.exception.limit_bytes ), ( 5000, 64 ) )
        self.assertLessEqual( max( stream.returned ), 100 )

    def test_the_too_long_error_carries_both_sizes( self ):
        err = bu_proto.LineTooLong( 9, 8 ).as_protocol_error()
        self.assertEqual( err.error, { 'code': 'protocol.line_too_long',
                                       'params': { 'limit_bytes': 8, 'line_bytes': 9 } } )
        self.assertIsNone( err.request_id )

    def test_every_line_read_is_numbered_blank_and_oversized_ones_too( self ):
        reader = bu_proto.LineReader( io.BytesIO( b'a\n\n  \r\nb\n' + b'x' * 30 + b'\nc\r\nd' ), limit=8 )
        seen = []
        while True:
            try:
                line = reader.read_line()
            except bu_proto.LineTooLong:
                seen.append( ( 'too_long', reader.line_number ) )
                continue
            if line is None:
                break
            seen.append( ( line, reader.line_number ) )
        self.assertEqual( seen, [ ( b'a', 1 ), ( b'b', 4 ), ( 'too_long', 5 ), ( b'c', 6 ), ( b'd', 7 ) ] )
        self.assertEqual( reader.line_number, 7 )

    def test_the_default_limit_is_one_mebibyte( self ):
        self.assertEqual( bu_proto.MAX_LINE_BYTES, 1048576 )


class TestParseCommand( unittest.TestCase ):

    def test_a_minimal_command_gets_empty_data( self ):
        self.assertEqual( bu_proto.parse_command( _cmd() ), ( 'a1', 'ping', {} ) )

    def test_data_is_passed_through( self ):
        command = bu_proto.parse_command( _cmd( data={ 'protocol': [ 1 ], 'extra': { 'x': 1 } } ) )
        self.assertEqual( command.data, { 'protocol': [ 1 ], 'extra': { 'x': 1 } } )

    def test_unknown_fields_are_ignored( self ):
        self.assertEqual( bu_proto.parse_command( _cmd( trace='t', seq=4, future={ 'a': [] } ) ),
                          ( 'a1', 'ping', {} ) )

    def test_bad_utf8( self ):
        err = _code( b'{"v":1,"kind":"cmd","id":"a1","name":"\xff"}' )
        self.assertEqual( ( err.code, err.request_id ), ( 'protocol.bad_encoding', None ) )

    def test_bad_json( self ):
        for line in ( 'not json', '{"v":1', '{"v":1}}', '{"v":NaN}', 'Infinity', "{'v':1}",
                      '[' * 100000 + ']' * 100000 ):
            with self.subTest( line=line[:20] ):
                err = _code( line )
                self.assertEqual( ( err.code, err.request_id ), ( 'protocol.bad_json', None ) )

    def test_a_json_value_that_is_not_an_object( self ):
        for line in ( '[]', '1', '"cmd"', 'null', 'true' ):
            with self.subTest( line=line ):
                err = _code( line )
                self.assertEqual( err.error['params'], { 'field': 'message', 'reason': 'type' } )

    def test_version( self ):
        self.assertEqual( _code( _without( 'v' ) ).error['params'], { 'field': 'v', 'reason': 'missing' } )
        for bad in ( '1', True, 1.0, None ):
            with self.subTest( v=bad ):
                self.assertEqual( _code( _cmd( v=bad ) ).error['params'], { 'field': 'v', 'reason': 'type' } )
        for unsupported in ( 0, 2, 99 ):
            with self.subTest( v=unsupported ):
                err = _code( _cmd( v=unsupported ) )
                self.assertEqual( err.error, { 'code': 'protocol.unsupported_version',
                                               'params': { 'got': unsupported, 'supported': [ 1 ] } } )
                self.assertEqual( ( err.request_id, err.request_name ), ( 'a1', 'ping' ) )

    def test_kind( self ):
        self.assertEqual( _code( _without( 'kind' ) ).error['params'], { 'field': 'kind', 'reason': 'missing' } )
        self.assertEqual( _code( _cmd( kind=1 ) ).error['params'], { 'field': 'kind', 'reason': 'type' } )
        for other in ( 'result', 'event', 'CMD', '' ):
            with self.subTest( kind=other ):
                self.assertEqual( _code( _cmd( kind=other ) ).error['params'],
                                  { 'field': 'kind', 'reason': 'value' } )

    def test_id( self ):
        self.assertEqual( _code( _without( 'id' ) ).error['params'], { 'field': 'id', 'reason': 'missing' } )
        self.assertEqual( _code( _cmd( id=5 ) ).error['params'], { 'field': 'id', 'reason': 'type' } )
        for bad in ( '', 'x' * 129 ):
            with self.subTest( length=len( bad ) ):
                err = _code( _cmd( id=bad ) )
                self.assertEqual( err.error['params'], { 'field': 'id', 'reason': 'value' } )
                self.assertIsNone( err.request_id )
        self.assertEqual( bu_proto.parse_command( _cmd( id='x' * 128 ) ).id, 'x' * 128 )

    def test_name( self ):
        self.assertEqual( _code( _without( 'name' ) ).error['params'], { 'field': 'name', 'reason': 'missing' } )
        self.assertEqual( _code( _cmd( name=[ 'ping' ] ) ).error['params'], { 'field': 'name', 'reason': 'type' } )
        for bad in ( '', 'Ping', 'has space', 'job.', '.job', 'a..b', 'job.Submit', '1job', 'ping\n', 'job-submit' ):
            with self.subTest( name=bad ):
                self.assertEqual( _code( _cmd( name=bad ) ).error['params'], { 'field': 'name', 'reason': 'value' } )
        for good in ( 'ping', 'job.submit', 'config.get', 'a1.b_2', 'catalog.add_files' ):
            with self.subTest( name=good ):
                self.assertEqual( bu_proto.parse_command( _cmd( name=good ) ).name, good )

    def test_a_badly_formed_name_is_still_echoed( self ):
        self.assertEqual( _code( _cmd( name='Bad Name' ) ).request_name, 'Bad Name' )

    def test_data_that_is_not_an_object( self ):
        for bad in ( [], 'x', None, 3 ):
            with self.subTest( data=bad ):
                self.assertEqual( _code( _cmd( data=bad ) ).error['params'],
                                  { 'field': 'data', 'reason': 'type' } )

    def test_the_id_and_name_are_kept_when_another_field_is_bad( self ):
        err = _code( _cmd( kind='event', id='q7', name='hello' ) )
        self.assertEqual( ( err.request_id, err.request_name ), ( 'q7', 'hello' ) )
        err = _code( _cmd( v=None, id='q8' ) )
        self.assertEqual( err.request_id, 'q8' )

    def test_an_unreadable_id_is_never_echoed( self ):
        for bad in ( 5, '', 'x' * 129, None, [ 'a1' ] ):
            with self.subTest( id=bad ):
                self.assertIsNone( _code( _cmd( kind='event', id=bad ) ).request_id )


class TestChooseVersion( unittest.TestCase ):

    def test_the_highest_common_version_wins( self ):
        self.assertEqual( bu_proto.choose_version( [ 1 ] ), 1 )
        self.assertEqual( bu_proto.choose_version( [ 3, 2, 1 ] ), 1 )
        with mock.patch.object( bu_proto, 'SUPPORTED_VERSIONS', ( 1, 2 ) ):
            self.assertEqual( bu_proto.choose_version( [ 1, 2, 3 ] ), 2 )
            self.assertEqual( bu_proto.choose_version( [ 2, 1 ] ), 2 )

    def test_nothing_in_common( self ):
        for offered in ( [], [ 2 ], [ True ], [ '1' ], [ 1.0 ] ):
            with self.subTest( offered=offered ):
                self.assertIsNone( bu_proto.choose_version( offered ) )


class TestOutgoing( unittest.TestCase ):

    def test_shapes( self ):
        self.assertEqual( bu_proto.ok_result( 'a1', 'ping', {} ),
                          { 'v': 1, 'kind': 'result', 'id': 'a1', 'name': 'ping', 'ok': True, 'data': {} } )
        failure = bu_proto.error( 'command.unknown', name='x' )
        self.assertEqual( bu_proto.error_result( 'a1', None, failure ),
                          { 'v': 1, 'kind': 'result', 'id': 'a1', 'name': None, 'ok': False,
                            'error': failure } )
        self.assertEqual( bu_proto.rejected( 7, failure ),
                          { 'v': 1, 'kind': 'event', 'name': 'protocol.rejected',
                            'data': { 'line': 7, 'error': failure } } )
        self.assertEqual( bu_proto.event( 'job.state', { 'state': 'running' } ),
                          { 'v': 1, 'kind': 'event', 'name': 'job.state', 'data': { 'state': 'running' } } )

    def test_a_result_always_answers_an_id( self ):
        with self.assertRaises( ValueError ):
            bu_proto.error_result( None, 'ping', bu_proto.error( 'protocol.bad_json' ) )

    def test_a_built_command_parses_back( self ):
        line = bu_proto.encode( bu_proto.command( 'a1', 'hello', { 'protocol': [ 1 ] } ) )
        self.assertEqual( bu_proto.parse_command( line[:-1] ), ( 'a1', 'hello', { 'protocol': [ 1 ] } ) )

    def test_encode_is_one_ascii_line( self ):
        line = bu_proto.encode( bu_proto.event( 'x', { 'text': 'a\nb\ré☃', 'path': 'clip\udcff.mp4' } ) )
        self.assertTrue( line.endswith( b'\n' ) )
        self.assertEqual( line.count( b'\n' ), 1 )
        self.assertNotIn( b'\r', line )
        line.decode( 'ascii' )
        self.assertEqual( json.loads( line )['data']['text'], 'a\nb\ré☃' )

    def test_values_json_cannot_carry_are_refused( self ):
        for bad in ( float( 'nan' ), float( 'inf' ), -float( 'inf' ) ):
            with self.subTest( value=bad ), self.assertRaises( ValueError ):
                bu_proto.encode( { 'x': bad } )
        with self.assertRaises( TypeError ):
            bu_proto.encode( { 'x': { 1, 2 } } )


class _SlowStream:
    """
    A stream that writes a few bytes at a time and sleeps between, so
    writers that do not hold a lock across the whole line interleave.
    """

    def __init__( self ):
        self.data = bytearray()
        self.flushes = 0

    def write( self, line ):
        for i in range( len( line ) ):
            self.data += line[i:i + 1]
            if i % 7 == 0:
                time.sleep( 0.0001 )
        return len( line )

    def flush( self ):
        self.flushes += 1


class TestMessageWriter( unittest.TestCase ):

    def test_concurrent_senders_never_interleave( self ):
        stream = _SlowStream()
        writer = bu_proto.MessageWriter( stream )

        def send_many( tag ):
            for i in range( 40 ):
                writer.send( bu_proto.event( 'test.tick', { 'tag': tag, 'i': i, 'pad': tag * 50 } ) )

        threads = [ threading.Thread( target=send_many, args=( t, ) ) for t in 'abcdef' ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        lines = bytes( stream.data ).split( b'\n' )
        self.assertEqual( lines[-1], b'' )
        messages = [ json.loads( line ) for line in lines[:-1] ]
        self.assertEqual( len( messages ), 240 )
        for message in messages:
            self.assertEqual( message['data']['pad'], message['data']['tag'] * 50 )
        self.assertEqual( stream.flushes, 240 )

    def test_a_sent_message_reaches_the_reader_without_closing_the_stream( self ):
        read_fd, write_fd = os.pipe()
        got = []
        reader = threading.Thread( target=lambda: got.append( os.read( read_fd, 65536 ) ), daemon=True )
        reader.start()
        with os.fdopen( write_fd, 'wb', buffering=65536 ) as stream:
            line = bu_proto.MessageWriter( stream ).send( bu_proto.event( 'x', {} ) )
            reader.join( timeout=10 )
            self.assertEqual( got, [ line ] )
        os.close( read_fd )

    def test_an_unencodable_message_writes_nothing( self ):
        stream = io.BytesIO()
        with self.assertRaises( ValueError ):
            bu_proto.MessageWriter( stream ).send( bu_proto.event( 'x', { 'v': float( 'nan' ) } ) )
        self.assertEqual( stream.getvalue(), b'' )


@schemas.needs_jsonschema
class TestSchemas( unittest.TestCase ):

    def test_every_file_is_a_valid_schema_with_an_id_matching_its_path( self ):
        files = schemas.schema_files()
        self.assertIn( 'envelope.schema.json', files )
        self.assertIn( 'errors.schema.json', files )
        for relative, doc in files.items():
            with self.subTest( schema=relative ):
                schemas.jsonschema.Draft202012Validator.check_schema( doc )
                self.assertEqual( doc['$id'], schemas.BASE_ID + relative )

    def test_errors_schema_and_registry_agree_both_ways( self ):
        doc = schemas.schema_files()['errors.schema.json']
        codes = { name: definition for name, definition in doc['$defs'].items() if name != 'reason' }
        self.assertEqual( set( codes ), set( bu_proto.ERRORS ) )
        self.assertEqual( { entry['$ref'] for entry in doc['oneOf'] },
                          { '#/$defs/' + code for code in bu_proto.ERRORS } )
        for code, definition in codes.items():
            with self.subTest( code=code ):
                self.assertEqual( definition['properties']['code'], { 'const': code } )
                params = definition['properties']['params']
                self.assertEqual( set( params['required'] ), set( bu_proto.ERRORS[code] ) )
                self.assertEqual( set( params['properties'] ), set( bu_proto.ERRORS[code] ) )
                self.assertIs( params['additionalProperties'], False )

    def test_every_error_validates_as_a_result_and_as_a_rejection( self ):
        for code, params in EXAMPLE_PARAMS.items():
            with self.subTest( code=code ):
                failure = bu_proto.error( code, **params )
                schemas.validate_message( bu_proto.error_result( 'a1', 'ping', failure ) )
                schemas.validate_message( bu_proto.error_result( 'a1', None, failure ) )
                event = bu_proto.rejected( 3, failure )
                schemas.validate_message( event )
                schemas.validate_event_data( event['name'], event['data'] )

    def test_a_result_without_an_id_or_a_rejection_without_a_line_does_not_validate( self ):
        failure = bu_proto.error( 'protocol.bad_json' )
        result = dict( bu_proto.error_result( 'a1', None, failure ), id=None )
        with self.assertRaises( schemas.jsonschema.ValidationError ):
            schemas.validate_message( result )
        for data in ( { 'error': failure }, { 'line': 0, 'error': failure }, { 'line': 1 } ):
            with self.subTest( data=data ), self.assertRaises( schemas.jsonschema.ValidationError ):
                schemas.validate_event_data( 'protocol.rejected', data )

    def test_an_error_with_a_sentence_or_a_wrong_code_does_not_validate( self ):
        bad = [
            { 'code': 'command.unknown', 'params': { 'name': 'x', 'message': 'No such command.' } },
            { 'code': 'command.nope', 'params': {} },
            { 'code': 'command.unknown', 'params': {} },
        ]
        for error_object in bad:
            with self.subTest( error=error_object ), self.assertRaises( schemas.jsonschema.ValidationError ):
                schemas.validate_message( bu_proto.error_result( 'a1', 'x', error_object ) )

    def test_every_parse_failure_validates( self ):
        lines = [ b'\xff', b'nope', b'[]', _without( 'v' ), _cmd( v=2 ), _cmd( kind='event' ),
                  _cmd( id='' ), _cmd( name='' ), _cmd( data=[] ) ]
        for line in lines:
            with self.subTest( line=line[:30] ):
                err = _code( line )
                if err.request_id is None:
                    message = bu_proto.rejected( 1, err.error )
                    schemas.validate_event_data( message['name'], message['data'] )
                else:
                    message = bu_proto.error_result( err.request_id, err.request_name, err.error )
                schemas.validate_message( message )

    def test_outgoing_shapes_validate_and_wrong_ones_do_not( self ):
        schemas.validate_message( bu_proto.command( 'a1', 'hello', { 'protocol': [ 1 ] } ) )
        schemas.validate_message( bu_proto.ok_result( 'a1', 'ping', {} ) )
        schemas.validate_message( bu_proto.event( 'job.progress', { 'done': 1 } ) )
        wrong = [
            dict( bu_proto.event( 'x', {} ), id='a1' ),
            dict( bu_proto.ok_result( 'a1', 'ping', {} ), v=2 ),
            dict( bu_proto.ok_result( None, 'ping', {} ) ),
            { 'v': 1, 'kind': 'cmd', 'id': 'a1', 'name': 'Has Spaces' },
        ]
        for message in wrong:
            with self.subTest( message=message ), self.assertRaises( schemas.jsonschema.ValidationError ):
                schemas.validate_message( message )

    def test_unknown_fields_validate( self ):
        schemas.validate_message( dict( bu_proto.ok_result( 'a1', 'ping', {} ), future=True ) )


if __name__ == '__main__':
    unittest.main()
