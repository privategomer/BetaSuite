"""
test_migrate.py - the 1.x to 2.x migration copies, never moves.

A synthetic 1.x tree with every kind of entry: app data the table maps,
media, entries the table does not name, temp files, a run marker, the
run lock. While a migration runs the tree is read-only, so any write to
it fails the test; afterwards its checksums and timestamps are compared
with a snapshot taken before. Every copied file must land exactly where
the resolver looks for it.
"""

import errno
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

import betautils_migrate as bu_migrate
import betautils_paths as bu_paths
import betautils_runlock as bu_runlock


# 1.x path -> contents. Where each one goes is the resolver's business.
APP_DATA = {
    'resources/model/v3.4-320n.onnx':               b'onnx bytes',
    'resources/stickers/vulva/a.png':               b'png bytes',
    'resources/betaconfig_machine.py':              b'gpu_enabled = 1\n',
    'output/cache/vid_hashes/abc-nudenet_v3-3-d123456.gz': b'detections',
    'output/cache/pic_hashes/def.gz':               b'pic detections',
    'output/cache/shot_cuts/abc-2-0.300.gz':        b'cuts',
    'output/cache/transcode_cache/abc.mp4':         b'transcode',
    'output/cache/file_hashes.json':                b'{"x": "abc"}',
    'output/cache/run_keys/detection-123456.json':  b'{"backend": "nudenet_v3"}',
    'output/logs/betasuite.log':                    b'log line\n',
    'output/logs/analysis_runs/20010917/x.log':     b'analysis\n',
    'output/stats/betasuite_stats.jsonl':           b'{"row": 1}\n',
    'output/stats/betasuite_stats.failures.jsonl':  b'{"row": 2}\n',
    'output/benchmarks/ab_round/ranking.json':      b'{}',
}
MEDIA = {
    'resources/uncensored_vids/folder/ünicode clip.mp4': b'video',
    'resources/uncensored_pics/p.jpg':              b'picture',
    'resources/source/completed/folder/done.mkv':   b'done video',
    'output/censored_vids/folder/out.mp4':          b'censored',
    'output/censored_pics/p.jpg':                   b'censored picture',
}
UNKNOWN = {
    'resources/custom/thing.bin':                   b'?',
    'output/github_settings/ruleset.json':          b'{}',
    'output/stray.json':                            b'{}',
}
TRANSIENT = {
    'output/cache/vid_hashes/abc.gz.tmp':           b'half',
    'output/stats/betasuite_stats.running.json':    b'{"pid": 2147483646, "boot_id": "another boot"}',
    'output/run.lock':                              b'',
}
LOCAL_CONFIG = b"detector_backend = { 'nudenet_v3': { 'model_path': '../resources/model/v3.4-320n.onnx' } }\n" \
               b"default_sticker = { 'type': 'sticker', 'dir': '../resources/custom/stickers/' }\n"


def _snapshot( root ):
    """{ relative path: ( sha256, size, mtime_ns ) } for every file, and the set of folders."""
    files, folders = {}, set()
    for directory, subdirectories, names in os.walk( root ):
        for sub in subdirectories:
            folders.add( os.path.relpath( os.path.join( directory, sub ), root ) )
        for name in names:
            path = os.path.join( directory, name )
            info = os.stat( path )
            with open( path, 'rb' ) as handle:
                digest = hashlib.sha256( handle.read() ).hexdigest()
            files[os.path.relpath( path, root )] = ( digest, info.st_size, info.st_mtime_ns )
    return files, folders


def _set_read_only( root, read_only ):
    for directory, subdirectories, names in os.walk( root ):
        for name in names:
            path = os.path.join( directory, name )
            os.chmod( path, stat.S_IREAD if read_only else stat.S_IREAD | stat.S_IWRITE )
        if os.name != 'nt':
            mode = stat.S_IREAD | stat.S_IEXEC
            os.chmod( directory, mode if read_only else mode | stat.S_IWRITE )


class _TreeCase( unittest.TestCase ):

    def setUp( self ):
        self._tmp = tempfile.TemporaryDirectory( prefix='betasuite-migrate-' )
        self.addCleanup( self._tmp.cleanup )
        self.addCleanup( lambda: os.path.isdir( self.source ) and _set_read_only( self.source, False ) )
        base = os.path.realpath( self._tmp.name )
        self.source = os.path.join( base, 'tree' )
        self.checkout = os.path.join( base, 'checkout' )
        self.env = { 'HOME': os.path.join( base, 'home' ), 'USERPROFILE': os.path.join( base, 'home' ),
                     'XDG_CONFIG_HOME': os.path.join( base, 'os', 'config' ),
                     'XDG_DATA_HOME': os.path.join( base, 'os', 'data' ),
                     'XDG_CACHE_HOME': os.path.join( base, 'os', 'cache' ),
                     'XDG_STATE_HOME': os.path.join( base, 'os', 'state' ),
                     'LOCALAPPDATA': os.path.join( base, 'os', 'local' ) }
        self.targets = bu_paths.os_dirs( environ=self.env )
        self.os_root = os.path.join( base, 'os' )
        for entries in ( APP_DATA, MEDIA, UNKNOWN, TRANSIENT ):
            for relative, body in entries.items():
                path = os.path.join( self.source, *relative.split( '/' ) )
                os.makedirs( os.path.dirname( path ), exist_ok=True )
                with open( path, 'wb' ) as handle:
                    handle.write( body )
        os.makedirs( self.checkout )
        self.local_config = os.path.join( self.checkout, 'betaconfig_local.py' )
        with open( self.local_config, 'wb' ) as handle:
            handle.write( LOCAL_CONFIG )

    def plan( self, **kwargs ):
        return bu_migrate.make_plan( self.source, self.local_config, self.targets, **kwargs )

    def run_migration( self, **kwargs ):
        _set_read_only( self.source, True )
        try:
            plan = self.plan( **kwargs )
            outcome, errors = bu_migrate.execute( plan, mock.Mock() )
        finally:
            _set_read_only( self.source, False )
        return plan, outcome, errors

    def resolved( self, relative ):
        """Where the app, in the OS layout of self.env, looks for a 1.x path."""
        return bu_paths.resolve( '../' + relative, environ=self.env )


class TestCopy( _TreeCase ):

    def test_every_app_data_file_lands_where_the_app_looks_byte_for_byte( self ):
        _plan, outcome, errors = self.run_migration()
        self.assertEqual( ( outcome, errors ), ( bu_migrate.COMPLETE, [] ) )
        for relative, body in APP_DATA.items():
            with open( self.resolved( relative ), 'rb' ) as handle:
                self.assertEqual( handle.read(), body, relative )
        with open( os.path.join( self.targets['config'], 'betaconfig_local.py' ), 'rb' ) as handle:
            self.assertEqual( handle.read(), LOCAL_CONFIG, "the settings are copied byte for byte" )

    def test_media_unknown_and_transient_entries_are_not_copied( self ):
        plan, _outcome, _errors = self.run_migration()
        copied = { item.relative for item in plan.items }
        self.assertEqual( copied, set( APP_DATA ) | { 'betaconfig_local.py' } )
        files, _folders = _snapshot( self.os_root )
        names = { os.path.basename( path ) for path in files }
        for relative in list( MEDIA ) + list( UNKNOWN ) + list( TRANSIENT ):
            if os.path.basename( relative ) not in ( 'run.lock', ):
                self.assertNotIn( os.path.basename( relative ), names, relative )

    def test_the_source_is_byte_identical_afterwards( self ):
        before = _snapshot( self.source )
        self.run_migration()
        self.run_migration()
        self.assertEqual( _snapshot( self.source ), before )

    def test_the_files_copied_keep_their_timestamps( self ):
        self.run_migration()
        source = os.stat( os.path.join( self.source, 'output', 'logs', 'betasuite.log' ) )
        copy = os.stat( self.resolved( 'output/logs/betasuite.log' ) )
        self.assertEqual( copy.st_mtime_ns, source.st_mtime_ns )


class TestRerun( _TreeCase ):

    def test_a_second_run_changes_nothing( self ):
        self.run_migration()
        after_first = _snapshot( self.os_root )
        plan, outcome, errors = self.run_migration()
        self.assertEqual( ( outcome, errors ), ( bu_migrate.COMPLETE, [] ) )
        self.assertEqual( { item.action for item in plan.items }, { bu_migrate.SAME } )
        self.assertEqual( { item.outcome for item in plan.items }, { bu_migrate.SAME } )
        self.assertEqual( _snapshot( self.os_root ), after_first )

    def test_same_bytes_with_another_timestamp_are_the_same( self ):
        self.run_migration()
        os.utime( self.resolved( 'output/logs/betasuite.log' ), ns=( 1, 1 ) )
        plan = self.plan()
        self.assertEqual( { item.relative: item.action for item in plan.items }['output/logs/betasuite.log'],
                          bu_migrate.SAME )

    def test_a_destination_that_differs_is_never_overwritten( self ):
        self.run_migration()
        changed = self.resolved( 'output/stats/betasuite_stats.jsonl' )
        _set_read_only( self.os_root, False )
        with open( changed, 'wb' ) as handle:
            handle.write( b'{"row": 1}\n{"row": "added by 2.x"}\n' )
        plan, outcome, _errors = self.run_migration()
        actions = { item.relative: item.action for item in plan.items }
        self.assertEqual( actions['output/stats/betasuite_stats.jsonl'], bu_migrate.CONFLICT )
        self.assertEqual( outcome, bu_migrate.COMPLETE )
        with open( changed, 'rb' ) as handle:
            self.assertIn( b'added by 2.x', handle.read() )
        lines = bu_migrate.summary_lines( bu_migrate.report( plan, outcome ) )
        self.assertTrue( any( 'conflict' in line and changed in line for line in lines ) )

    def test_same_size_different_bytes_is_a_conflict( self ):
        self.run_migration()
        changed = self.resolved( 'output/logs/betasuite.log' )
        _set_read_only( self.os_root, False )
        with open( changed, 'wb' ) as handle:
            handle.write( b'LOG LINE\n' )
        actions = { item.relative: item.action for item in self.plan().items }
        self.assertEqual( actions['output/logs/betasuite.log'], bu_migrate.CONFLICT )

    def test_a_full_disk_stops_cleanly_and_a_rerun_finishes( self ):
        real_copy = shutil.copy2
        calls = []

        def copy_until_full( source, dest ):
            calls.append( source )
            if len( calls ) == 4:
                with open( dest, 'wb' ) as handle:
                    handle.write( b'half' )
                raise OSError( errno.ENOSPC, 'No space left on device' )
            return real_copy( source, dest )
        with mock.patch.object( bu_migrate.shutil, 'copy2', copy_until_full ):
            plan, outcome, _errors = self.run_migration()
        self.assertEqual( outcome, bu_migrate.STOPPED_DISK_FULL )
        outcomes = [ item.outcome for item in plan.items ]
        self.assertEqual( outcomes[:3], [ 'copied' ] * 3 )
        self.assertTrue( all( o == 'not_copied' for o in outcomes[3:] ) )
        files, _folders = _snapshot( self.os_root )
        self.assertFalse( [ f for f in files if f.endswith( bu_migrate.PARTIAL_SUFFIX ) ], "no half file left" )
        _plan, outcome, _errors = self.run_migration()
        self.assertEqual( outcome, bu_migrate.COMPLETE )
        for relative, body in APP_DATA.items():
            with open( self.resolved( relative ), 'rb' ) as handle:
                self.assertEqual( handle.read(), body, relative )

    def test_other_copy_errors_are_reported_and_the_rest_continues( self ):
        real_copy = shutil.copy2

        def fail_one( source, dest ):
            if source.endswith( 'ranking.json' ):
                raise PermissionError( errno.EACCES, 'Permission denied' )
            return real_copy( source, dest )
        with mock.patch.object( bu_migrate.shutil, 'copy2', fail_one ):
            plan, outcome, errors = self.run_migration()
        self.assertEqual( outcome, bu_migrate.FINISHED_WITH_ERRORS )
        self.assertEqual( [ e['path'] for e in errors ], [ 'output/benchmarks/ab_round/ranking.json' ] )
        self.assertEqual( sum( 1 for item in plan.items if item.outcome == 'copied' ), len( plan.items ) - 1 )


class TestOrderAndOptions( _TreeCase ):

    def test_settings_and_data_go_first_cache_last( self ):
        kinds = [ item.kind for item in self.plan().items ]
        self.assertEqual( kinds, sorted( kinds, key=bu_migrate.COPY_ORDER.index ) )
        self.assertEqual( kinds[0], 'config' )
        self.assertEqual( kinds[-1], 'cache' )

    def test_the_cache_can_be_left_out( self ):
        plan = self.plan( include_cache=False )
        self.assertNotIn( 'cache', { item.kind for item in plan.items } )
        self.assertEqual( plan.skipped['cache_left_out'][0], 5 )

    def test_links_are_never_followed( self ):
        if not hasattr( os, 'symlink' ):
            self.skipTest( 'no symlinks' )
        outside = os.path.join( self.checkout, 'outside' )
        os.makedirs( outside )
        with open( os.path.join( outside, 'secret' ), 'wb' ) as handle:
            handle.write( b'x' )
        try:
            os.symlink( outside, os.path.join( self.source, 'output', 'logs', 'linked' ) )
            os.symlink( self.local_config, os.path.join( self.source, 'output', 'logs', 'linked.log' ) )
        except OSError:
            self.skipTest( 'symlinks not allowed here' )
        plan = self.plan()
        self.assertEqual( plan.skipped['link'][0], 2 )
        self.assertFalse( any( 'linked' in item.relative for item in plan.items ) )

    def test_without_a_settings_file_nothing_is_said_about_one( self ):
        plan = bu_migrate.make_plan( self.source, None, self.targets )
        self.assertNotIn( 'config', { item.kind for item in plan.items if item.relative == 'betaconfig_local.py' } )
        self.assertEqual( plan.config_refs, [] )


class TestReport( _TreeCase ):

    def test_report_names_media_folders_without_their_file_names( self ):
        plan, outcome, errors = self.run_migration()
        document = bu_migrate.report( plan, outcome, errors )
        text = json.dumps( document, ensure_ascii=False )
        for relative in MEDIA:
            self.assertNotIn( os.path.basename( relative ), text )
        self.assertEqual( { m['folder'] for m in document['media_left_in_place'] },
                          { 'resources/uncensored_vids', 'resources/uncensored_pics', 'resources/source',
                            'output/censored_vids', 'output/censored_pics' } )
        self.assertEqual( document['media_hint'], {
            bu_paths.INPUT_ENV_VAR: os.path.join( self.source, 'resources' ),
            bu_paths.OUTPUT_ENV_VAR: os.path.join( self.source, 'output' ) } )
        self.assertEqual( { u['path'] for u in document['unknown_left_in_place'] },
                          { 'resources/custom', 'output/github_settings', 'output/stray.json' } )
        self.assertEqual( document['skipped']['transient']['files'], 3 )

    def test_report_lists_every_file_and_what_happened( self ):
        plan, outcome, errors = self.run_migration()
        document = bu_migrate.report( plan, outcome, errors )
        self.assertEqual( { f['path'] for f in document['files'] }, set( APP_DATA ) | { 'betaconfig_local.py' } )
        self.assertEqual( { f['outcome'] for f in document['files'] }, { 'copied' } )
        self.assertEqual( document['totals']['cache']['copy'][0], 5 )

    def test_the_media_hint_keeps_the_1x_video_folders_in_use( self ):
        hint = bu_migrate.media_hint( self.plan() )
        env = dict( self.env, **hint )
        for relative in MEDIA:
            self.assertEqual( bu_paths.resolve( '../' + relative, environ=env ),
                              os.path.join( self.source, *relative.split( '/' ) ), relative )

    def test_a_setting_that_names_a_folder_not_copied_is_flagged( self ):
        refs = self.plan().config_refs
        self.assertEqual( refs, [ ( 'default_sticker.dir', '../resources/custom/stickers/' ) ] )

    def test_the_summary_is_plain_lines( self ):
        lines = bu_migrate.summary_lines( bu_migrate.report( self.plan(), bu_migrate.DRY_RUN ) )
        self.assertTrue( lines[0].startswith( '1.x tree: ' ) )
        self.assertTrue( any( 'media left in place: resources/uncensored_vids' in line for line in lines ) )
        self.assertTrue( any( 'default_sticker.dir' in line for line in lines ) )
        self.assertEqual( lines[-1], 'outcome: dry_run' )


class TestRefusals( _TreeCase ):

    def test_not_a_1x_tree( self ):
        with self.assertRaises( bu_migrate.MigrationRefused ) as raised:
            bu_migrate.make_plan( self.checkout, None, self.targets )
        self.assertEqual( raised.exception.code, 'not_a_1x_tree' )

    def test_target_and_source_overlap( self ):
        for kind, inside in ( ( 'cache', os.path.join( self.source, 'output', 'new-cache' ) ),
                              ( 'data', os.path.dirname( self.source ) ) ):
            targets = dict( self.targets, **{ kind: inside } )
            with self.assertRaises( bu_migrate.MigrationRefused ) as raised:
                bu_migrate.make_plan( self.source, None, targets )
            self.assertEqual( ( raised.exception.code, raised.exception.params['kind'] ), ( 'overlap', kind ) )

    def test_a_run_holding_either_lock( self ):
        plan = self.plan()
        for side, path in ( ( 'source', os.path.join( self.source, 'output', 'run.lock' ) ),
                            ( 'target', os.path.join( self.targets['state'], 'run.lock' ) ) ):
            with bu_runlock.acquire( 'betatv', path=path ):
                with self.assertRaises( bu_migrate.MigrationRefused ) as raised:
                    bu_migrate.execute( plan, mock.Mock() )
            self.assertEqual( raised.exception.code, 'run_in_progress' )
            self.assertEqual( ( raised.exception.params['side'], raised.exception.params['program'] ),
                              ( side, 'betatv' ) )
        self.assertFalse( os.path.exists( self.targets['config'] ), "refused before copying anything" )

    def test_a_live_1x_run_marker( self ):
        marker = os.path.join( self.source, 'output', 'stats', 'betasuite_stats.running.json' )
        with open( marker, 'w' ) as handle:
            json.dump( { 'pid': os.getpid(), 'boot_id': None, 'started_at': 5 }, handle )
        import betautils_log as bu_log
        with mock.patch.object( bu_log, 'betatv_process_running', return_value=True ):
            with self.assertRaises( bu_migrate.MigrationRefused ) as raised:
                bu_migrate.check_can_start( self.plan() )
        self.assertEqual( raised.exception.params['side'], 'source' )
        # A marker from a run that is gone (the default fixture) does not stop it.
        with mock.patch.object( bu_log, 'betatv_process_running', return_value=False ):
            bu_migrate.check_can_start( self.plan() )

    def test_not_enough_free_space( self ):
        plan = self.plan()
        need = sum( item.size for item in plan.to_copy() )
        small = mock.Mock( return_value=mock.Mock( free=need + bu_migrate.SPACE_MARGIN_BYTES - 1 ) )
        with self.assertRaises( bu_migrate.MigrationRefused ) as raised:
            bu_migrate.check_can_start( plan, disk_usage=small )
        self.assertEqual( raised.exception.code, 'no_space' )
        self.assertEqual( raised.exception.params['needed_bytes'], need + bu_migrate.SPACE_MARGIN_BYTES )
        enough = mock.Mock( return_value=mock.Mock( free=need + bu_migrate.SPACE_MARGIN_BYTES ) )
        bu_migrate.check_can_start( plan, disk_usage=enough )

    def test_the_lock_is_given_back( self ):
        self.run_migration()
        self.assertIsNone( bu_runlock.probe( os.path.join( self.targets['state'], 'run.lock' ) ) )


_CACHE_PATHS = r'''
import json, os, sys
import betautils_cache_paths as c
c.record_run_key = lambda *a, **k: None
det = c.detection_key()
paths = [ c.box_hash_path_for( 'f' * 16, 640, 2.0, 0.2, 'nudenet_v3', det_key=det ),
          c.pic_hash_path_for( 'e' * 16, 640, 0.2, 'nudenet_v3', det_key=det ),
          c.shot_cut_path_for( 'f' * 16, 2.0, 0.3 ),
          c.transcode_path_for( 'f' * 16 ),
          os.path.join( c.betaconst.run_key_dir, 'detection-%s.json'%( det, ) ) ]
if sys.argv[1] == 'write':
    for path in paths:
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'w' ) as handle:
            handle.write( 'cached' )
print( json.dumps( { 'paths': paths, 'found': [ os.path.isfile( p ) for p in paths ], 'det': det } ) )
'''


def _engine_packages_installed():
    import importlib.util
    return importlib.util.find_spec( 'numpy' ) is not None and importlib.util.find_spec( 'cv2' ) is not None


@unittest.skipUnless( _engine_packages_installed(), "needs the engine's packages" )
class TestCachesStayHits( _TreeCase ):
    """The app's own cache paths, written in the 1.x layout, are found in the 2.x one after migrating."""

    def probe( self, mode, env ):
        import subprocess
        import sys
        repo = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
        full = dict( os.environ, PYTHONPATH=repo, BETASUITE_LOCAL_CONFIG=self.local_config, **env )
        for name in bu_paths.OVERRIDES.values():
            full.pop( name, None )
        result = subprocess.run( [ sys.executable, '-c', _CACHE_PATHS, mode ], cwd=self.checkout, env=full,
                                 capture_output=True, text=True, timeout=120 )
        self.assertEqual( result.returncode, 0, result.stderr )
        return json.loads( result.stdout.strip().splitlines()[-1] )

    def test_every_cache_written_in_1x_is_found_in_2x( self ):
        written = self.probe( 'write', dict( self.env, BETASUITE_DATA_ROOT=self.source ) )
        self.assertTrue( all( p.startswith( self.source + os.sep ) for p in written['paths'] ) )
        before = self.probe( 'read', dict( self.env, BETASUITE_DATA_ROOT='' ) )
        self.assertEqual( before['found'], [ False ] * 5, "nothing there before the migration" )
        self.run_migration()
        after = self.probe( 'read', dict( self.env, BETASUITE_DATA_ROOT='' ) )
        self.assertEqual( after['det'], written['det'], "the same detection key in both layouts" )
        self.assertEqual( after['found'], [ True ] * 5, after['paths'] )
        self.assertTrue( all( p.startswith( self.os_root + os.sep ) for p in after['paths'] ) )


class TestCommandLine( _TreeCase ):

    def main( self, *argv, stdin=None ):
        import importlib.util
        tool = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                             'tools', 'setup', 'migrate_1x.py' )
        spec = importlib.util.spec_from_file_location( '_migrate_1x', tool )
        module = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( module )
        out = io.StringIO()
        log = os.path.join( self.checkout, 'migrate.log' )
        with mock.patch.dict( os.environ, self.env ), mock.patch.dict( os.environ, {}, clear=False ):
            for name in ( bu_paths.ENV_VAR, ) + tuple( bu_paths.OVERRIDES.values() ):
                os.environ.pop( name, None )
            code = module.main( [ '--from', self.source, '--local-config', self.local_config,
                                  '--log-file', log ] + list( argv ),
                                stdin=stdin or io.StringIO(), stdout=out )
        return code, out.getvalue()

    def test_a_dry_run_creates_nothing_anywhere( self ):
        before = _snapshot( self.source )
        report = os.path.join( self.checkout, 'plan.json' )
        code, out = self.main( '--dry-run', '--report', report )
        self.assertEqual( code, 0 )
        self.assertFalse( os.path.exists( self.os_root ) )
        self.assertEqual( _snapshot( self.source ), before )
        self.assertIn( 'outcome: dry_run', out )
        with open( report ) as handle:
            self.assertEqual( json.load( handle )['outcome'], 'dry_run' )

    def test_unattended_without_yes_copies_nothing( self ):
        code, out = self.main()
        self.assertEqual( code, 0 )
        self.assertIn( 'nothing is copied', out )
        self.assertFalse( os.path.exists( self.os_root ) )

    def test_yes_copies_and_writes_the_report_and_record( self ):
        code, out = self.main( '--yes' )
        self.assertEqual( code, 0, out )
        reports = os.listdir( os.path.join( self.targets['state'], 'migration' ) )
        self.assertEqual( len( reports ), 1 )
        with open( os.path.join( self.targets['state'], 'migration', reports[0] ) ) as handle:
            self.assertEqual( json.load( handle )['outcome'], 'complete' )
        with open( os.path.join( self.targets['data'], 'migrated_from.json' ) ) as handle:
            record = json.load( handle )
        self.assertEqual( record['source'], self.source )
        self.assertIn( os.path.join( self.source, 'resources', 'uncensored_vids' ), record['media_folders'] )
        code, out = self.main( '--yes' )
        self.assertEqual( code, 0 )
        self.assertIn( 'nothing to copy', out )

    def test_at_a_terminal_it_asks_and_no_copies_nothing( self ):
        class Terminal( io.StringIO ):
            def isatty( self ):
                return True
        code, out = self.main( stdin=Terminal( 'n\n' ) )
        self.assertEqual( code, 0 )
        self.assertIn( 'nothing copied', out )
        self.assertFalse( os.path.exists( self.targets['config'] ) )
        code, out = self.main( stdin=Terminal( 'y\n' ) )
        self.assertEqual( code, 0, out )
        self.assertTrue( os.path.isfile( os.path.join( self.targets['config'], 'betaconfig_local.py' ) ) )

    def test_a_refusal_exits_1_with_the_reason( self ):
        code, out = self.main( '--from', self.checkout, '--yes' )
        self.assertEqual( code, 1 )
        self.assertIn( 'not a 1.x tree', out )


if __name__ == '__main__':
    unittest.main()
