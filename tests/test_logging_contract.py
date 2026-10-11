"""
test_logging_contract.py - every logger comes from betautils_log.

Pipeline code calls logger.trace(). A tool that built its own plain
logging.Logger and handed it to the render path failed every render
chunk with AttributeError (suppression_ab, 2026-10-10): betabench,
auto_tune and four other tools each re-declared TRACE and built their
own logger, and none of those loggers had trace(). These tests pin one
level vocabulary and one place loggers are built.
"""

import ast
import importlib.util
import io
import logging
import os
import tempfile
import unittest

import betautils_log as bu_log

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

# Standalone on purpose: they import no project module, so they run
# before the environment is set up (setup, release) or while
# betaconfig.py is being rewritten (the bar smoothing test), and they
# never hand a logger to pipeline code. The last test holds them to that.
# Value: the project modules each may import (none of which is handed
# the tool's logger).
STANDALONE = {
    'tools/release/release.py': set(),
    'tools/setup/verify_env.py': set(),
    'tools/setup/fetch_models.py': set(),
    'tools/tuning/run_bar_smoothing_test.py': { 'betautils_hash' },   # md5_for_file only
}


def _sources():
    for root, dirs, files in os.walk( REPO ):
        dirs[:] = [ d for d in dirs if d not in ( '.venv', '.git', '__pycache__', 'tests' ) ]
        for name in files:
            if name.endswith( '.py' ):
                path = os.path.join( root, name )
                yield os.path.relpath( path, REPO ).replace( os.sep, '/' ), path


def _load( relative, name ):
    spec = importlib.util.spec_from_file_location( name, os.path.join( REPO, relative ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestOneVocabularyOnePlace( unittest.TestCase ):

    def test_no_module_builds_its_own_logger_or_trace_level( self ):
        offenders = []
        for relative, path in _sources():
            if relative == 'betautils_log.py' or relative in STANDALONE:
                continue
            with open( path, encoding='UTF-8' ) as handle:
                source = handle.read()
            for needle in ( 'logging.getLogger(', 'addLevelName(', 'TRACE = 5' ):
                if needle in source:
                    offenders.append( '%s: %s'%( relative, needle ) )
        self.assertEqual( offenders, [], "build loggers with betautils_log.tool_logger / silent_logger" )

    def test_standalone_tools_import_no_project_module( self ):
        for relative, allowed in STANDALONE.items():
            with open( os.path.join( REPO, relative ), encoding='UTF-8' ) as handle:
                tree = ast.parse( handle.read() )
            imported = set()
            for node in ast.walk( tree ):
                if isinstance( node, ast.Import ):
                    imported.update( alias.name for alias in node.names )
                elif isinstance( node, ast.ImportFrom ) and node.module:
                    imported.add( node.module )
            self.assertEqual( sorted( name for name in imported
                                      if name.startswith( 'betautils' ) and name not in allowed ), [],
                              "%s hands loggers to nothing only while it imports no pipeline module"%(
                                  relative, ) )


class TestBuilders( unittest.TestCase ):

    def test_tool_logger_traces_to_its_file_and_filters_the_terminal( self ):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join( tmp, 'sub', 'tool.log' )
            stream = io.StringIO()
            logger = bu_log.tool_logger( '_contract_tool', path, 'trace', 'info', console_stream=stream )
            try:
                logger.trace( "chunk %d", 7 )
                logger.info( "done" )
                for handler in logger.handlers:
                    handler.flush()
                with open( path ) as handle:
                    text = handle.read()
                self.assertIn( '[TRACE] chunk 7', text )
                self.assertIn( 'done', text )
                self.assertEqual( stream.getvalue(), 'done\n' )
            finally:
                bu_log.tool_logger( '_contract_tool', None, console_level=None )

    def test_rebuilding_replaces_handlers_and_no_console_means_none( self ):
        stream = io.StringIO()
        bu_log.tool_logger( '_contract_rebuild', None, console_stream=stream )
        logger = bu_log.tool_logger( '_contract_rebuild', None, console_stream=stream )
        self.assertEqual( len( logger.handlers ), 1 )
        self.assertEqual( bu_log.tool_logger( '_contract_rebuild', None, console_level=None ).handlers, [] )

    def test_a_plain_logger_created_first_is_made_trace_capable( self ):
        plain = logging.getLogger( '_contract_plain_first' )
        self.assertFalse( hasattr( plain, 'trace' ) )
        logger = bu_log.silent_logger( '_contract_plain_first' )
        self.assertIs( logger, plain )
        logger.trace( "no error" )

    def test_silent_logger_discards( self ):
        logger = bu_log.silent_logger( '_contract_silent' )
        self.assertFalse( logger.propagate )
        self.assertTrue( all( isinstance( handler, logging.NullHandler ) for handler in logger.handlers ) )

    def test_every_tool_builder_hands_out_a_trace_capable_logger( self ):
        with tempfile.TemporaryDirectory() as tmp:
            for relative, name in ( ( 'tools/bench/betabench.py', '_c_betabench' ),
                                    ( 'tools/tuning/auto_tune.py', '_c_auto_tune' ) ):
                logger = _load( relative, name ).build_logger( tmp, 'trace', 'error' )
                logger.trace( "render chunk command" )
                self.assertTrue( callable( logger.trace ), relative )
                for handler in list( logger.handlers ):
                    handler.close()
                    logger.removeHandler( handler )
            import betautils_tuning as bu_tuning
            bu_tuning._QUIET_LOGGER.trace( "replay" )


if __name__ == '__main__':
    unittest.main()
