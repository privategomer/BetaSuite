"""
test_paths.py - where each kind of data lives, on every OS.

The OS folders are computed from an environment and a platform name
passed in, so Windows and macOS paths are checked on any machine (and
for real on the Windows CI job). Expected values are written out in
full rather than recomputed, so a change to a default shows up here.
"""

import ntpath
import os
import posixpath
import tempfile
import unittest
from unittest import mock

import betautils_local as bu_local
import betautils_paths as bu_paths


LINUX_HOME = { 'HOME': '/home/u' }
LINUX_DEFAULTS = {
    'config':  '/home/u/.config/betasuite',
    'data':    '/home/u/.local/share/betasuite/data',
    'cache':   '/home/u/.cache/betasuite',
    'state':   '/home/u/.local/state/betasuite',
    'inputs':  '/home/u/.local/share/betasuite/library',
    'outputs': '/home/u/.local/share/betasuite/library',
}

WINDOWS_ENV = { 'USERPROFILE': 'C:\\Users\\u', 'LOCALAPPDATA': 'C:\\Users\\u\\AppData\\Local' }
WINDOWS_DEFAULTS = {
    'config':  'C:\\Users\\u\\AppData\\Local\\BetaSuite\\config',
    'data':    'C:\\Users\\u\\AppData\\Local\\BetaSuite\\data',
    'cache':   'C:\\Users\\u\\AppData\\Local\\BetaSuite\\cache',
    'state':   'C:\\Users\\u\\AppData\\Local\\BetaSuite\\state',
    'inputs':  'C:\\Users\\u\\AppData\\Local\\BetaSuite\\library',
    'outputs': 'C:\\Users\\u\\AppData\\Local\\BetaSuite\\library',
}

MAC_HOME = { 'HOME': '/Users/u' }
MAC_DEFAULTS = {
    'config':  '/Users/u/Library/Application Support/BetaSuite/config',
    'data':    '/Users/u/Library/Application Support/BetaSuite/data',
    'cache':   '/Users/u/Library/Caches/BetaSuite',
    'state':   '/Users/u/Library/Application Support/BetaSuite/state',
    'inputs':  '/Users/u/Library/Application Support/BetaSuite/library.noindex',
    'outputs': '/Users/u/Library/Application Support/BetaSuite/library.noindex',
}

# Every path betaconst and betautils_log name, and where the OS layout
# puts it on Linux with LINUX_DEFAULTS.
LINUX_LAYOUT = {
    '../resources/uncensored_pics/':      '/home/u/.local/share/betasuite/library/uncensored_pics/',
    '../output/censored_pics/':           '/home/u/.local/share/betasuite/library/censored_pics/',
    '../resources/uncensored_vids/':      '/home/u/.local/share/betasuite/library/uncensored_vids/',
    '../output/censored_vids/':           '/home/u/.local/share/betasuite/library/censored_vids/',
    '../resources/source/':               '/home/u/.local/share/betasuite/library/source/',
    '../resources/source/completed/':     '/home/u/.local/share/betasuite/library/source/completed/',
    '../output/cache':                    '/home/u/.cache/betasuite',
    '../output/cache/vid_hashes':         '/home/u/.cache/betasuite/vid_hashes',
    '../output/cache/pic_hashes':         '/home/u/.cache/betasuite/pic_hashes',
    '../output/cache/shot_cuts':          '/home/u/.cache/betasuite/shot_cuts',
    '../output/cache/transcode_cache':    '/home/u/.cache/betasuite/transcode_cache',
    '../output/cache/file_hashes.json':   '/home/u/.cache/betasuite/file_hashes.json',
    '../output/cache/run_keys':           '/home/u/.local/share/betasuite/data/run_keys',
    '../output/benchmarks':               '/home/u/.local/state/betasuite/benchmarks',
    '../output/logs/betasuite.log':       '/home/u/.local/state/betasuite/logs/betasuite.log',
    '../output/stats/betasuite_stats.jsonl': '/home/u/.local/state/betasuite/stats/betasuite_stats.jsonl',
    '../output/run.lock':                 '/home/u/.local/state/betasuite/run.lock',
    '../resources/model/v3.4-320n.onnx':  '/home/u/.local/share/betasuite/data/model/v3.4-320n.onnx',
    '../resources/stickers/vulva/':       '/home/u/.local/share/betasuite/data/stickers/vulva/',
    '../resources/betaconfig_machine.py': '/home/u/.config/betasuite/betaconfig_machine.py',
    '../output/betafied/x.mp4':           '/home/u/.local/share/betasuite/library/betafied/x.mp4',
    # Anything else keeps its name in data (under resources/) or state (under output/).
    '../resources':                       '/home/u/.local/share/betasuite/data',
    '../resources/my_models/a.onnx':      '/home/u/.local/share/betasuite/data/my_models/a.onnx',
    '../output':                          '/home/u/.local/state/betasuite',
    '../output/':                         '/home/u/.local/state/betasuite/',
    '../output/ab_round_1/ranking.json':  '/home/u/.local/state/betasuite/ab_round_1/ranking.json',
}

LEFT_ALONE = ( '/abs/output/x', 'output/x', './output/x', '../elsewhere/x', '../outputs/x',
               '../output_old/x', '../resourcesx', '..', '../', '../../output/x', 'x/../output/y',
               'ab/output/x', '.x/resources', '' )


class TestOsDirs( unittest.TestCase ):

    def test_linux_defaults_hang_off_home( self ):
        self.assertEqual( bu_paths.os_dirs( 'linux', LINUX_HOME ), LINUX_DEFAULTS )

    def test_other_unix_systems_use_xdg_too( self ):
        self.assertEqual( bu_paths.os_dirs( 'freebsd14', LINUX_HOME ), LINUX_DEFAULTS )

    def test_xdg_base_folders_are_honored_when_absolute( self ):
        env = dict( LINUX_HOME, XDG_CONFIG_HOME='/x/conf', XDG_DATA_HOME='/x/data/',
                    XDG_CACHE_HOME='/x/cache', XDG_STATE_HOME='/x/state' )
        dirs = bu_paths.os_dirs( 'linux', env )
        self.assertEqual( dirs['config'], '/x/conf/betasuite' )
        self.assertEqual( dirs['data'], '/x/data/betasuite/data' )
        self.assertEqual( dirs['inputs'], '/x/data/betasuite/library' )
        self.assertEqual( dirs['cache'], '/x/cache/betasuite' )
        self.assertEqual( dirs['state'], '/x/state/betasuite' )

    def test_relative_xdg_values_are_ignored_as_the_spec_says( self ):
        env = dict( LINUX_HOME, XDG_CONFIG_HOME='conf', XDG_DATA_HOME='./data', XDG_CACHE_HOME='~/c',
                    XDG_STATE_HOME='' )
        self.assertEqual( bu_paths.os_dirs( 'linux', env ), LINUX_DEFAULTS )

    def test_windows_defaults( self ):
        self.assertEqual( bu_paths.os_dirs( 'win32', WINDOWS_ENV ), WINDOWS_DEFAULTS )

    def test_windows_without_localappdata_uses_the_profile( self ):
        for local in ( None, '', 'relative', '\\no\\drive' ):
            env = dict( WINDOWS_ENV )
            env.pop( 'LOCALAPPDATA' )
            if local is not None:
                env['LOCALAPPDATA'] = local
            self.assertEqual( bu_paths.os_dirs( 'win32', env ), WINDOWS_DEFAULTS, local )

    def test_windows_profile_from_homedrive_and_homepath( self ):
        env = { 'HOMEDRIVE': 'D:', 'HOMEPATH': '\\Users\\v' }
        dirs = bu_paths.os_dirs( 'win32', env )
        self.assertEqual( dirs['cache'], 'D:\\Users\\v\\AppData\\Local\\BetaSuite\\cache' )
        self.assertEqual( dirs['inputs'], 'D:\\Users\\v\\AppData\\Local\\BetaSuite\\library' )

    def test_macos_defaults( self ):
        self.assertEqual( bu_paths.os_dirs( 'darwin', MAC_HOME ), MAC_DEFAULTS )

    def test_no_home_folder_is_an_error_naming_the_variable( self ):
        for platform, name in ( ( 'linux', 'HOME' ), ( 'darwin', 'HOME' ), ( 'win32', 'USERPROFILE' ) ):
            with self.assertRaises( bu_paths.PathsError ) as raised:
                bu_paths.os_dirs( platform, {} )
            self.assertIn( name, str( raised.exception ) )
        with self.assertRaises( bu_paths.PathsError ):
            bu_paths.os_dirs( 'linux', { 'HOME': 'relative' } )

    def test_overrides_move_one_kind_each( self ):
        cases = (
            ( 'linux', LINUX_HOME, LINUX_DEFAULTS, '/big/cache/', '/big/cache', '/in', '/out' ),
            ( 'darwin', MAC_HOME, MAC_DEFAULTS, '/big/cache', '/big/cache', '/in', '/out' ),
            ( 'win32', WINDOWS_ENV, WINDOWS_DEFAULTS, 'E:\\cache\\', 'E:\\cache', 'E:\\in', '\\\\nas\\share\\out' ),
        )
        for platform, env, defaults, cache, cache_normal, inputs, outputs in cases:
            env = dict( env, BETASUITE_CACHE_DIR=cache, BETASUITE_INPUT_DIR=inputs, BETASUITE_OUTPUT_DIR=outputs )
            dirs = bu_paths.os_dirs( platform, env )
            self.assertEqual( dirs['cache'], cache_normal, platform )
            self.assertEqual( dirs['inputs'], inputs, platform )
            self.assertEqual( dirs['outputs'], outputs, platform )
            for kind in ( 'config', 'data', 'state' ):
                self.assertEqual( dirs[kind], defaults[kind], ( platform, kind ) )

    def test_an_override_needs_no_home( self ):
        env = { 'BETASUITE_CACHE_DIR': '/big/cache' }
        self.assertEqual( bu_paths.kind_dir( 'cache', 'linux', env ), '/big/cache' )

    def test_relative_overrides_are_refused_naming_the_variable( self ):
        for name in ( bu_paths.CACHE_ENV_VAR, bu_paths.INPUT_ENV_VAR, bu_paths.OUTPUT_ENV_VAR ):
            for platform, env, value in ( ( 'linux', LINUX_HOME, 'cache' ), ( 'linux', LINUX_HOME, './c' ),
                                          ( 'linux', LINUX_HOME, '~/c' ), ( 'win32', WINDOWS_ENV, 'cache' ),
                                          ( 'win32', WINDOWS_ENV, '\\cache' ), ( 'win32', WINDOWS_ENV, 'C:cache' ) ):
                with self.assertRaises( bu_paths.PathsError ) as raised:
                    bu_paths.os_dirs( platform, dict( env, **{ name: value } ) )
                self.assertIn( name, str( raised.exception ) )

    def test_an_empty_override_is_unset( self ):
        env = dict( LINUX_HOME, BETASUITE_CACHE_DIR='', BETASUITE_INPUT_DIR='', BETASUITE_OUTPUT_DIR='' )
        self.assertEqual( bu_paths.os_dirs( 'linux', env ), LINUX_DEFAULTS )

    def test_kind_dir_matches_os_dirs_and_rejects_unknown_kinds( self ):
        for kind in bu_paths.KINDS:
            self.assertEqual( bu_paths.kind_dir( kind, 'linux', LINUX_HOME ), LINUX_DEFAULTS[kind] )
        with self.assertRaises( ValueError ):
            bu_paths.kind_dir( 'logs', 'linux', LINUX_HOME )

    def test_no_kind_folder_contains_another( self ):
        # Clearing the cache or removing app data must never reach the
        # videos (or another kind). Inputs and outputs share one folder.
        cases = [ ( 'linux', LINUX_HOME ), ( 'darwin', MAC_HOME ), ( 'win32', WINDOWS_ENV ),
                  ( 'linux', dict( LINUX_HOME, XDG_DATA_HOME='/x', XDG_CACHE_HOME='/x/cache' ) ) ]
        for platform, env in cases:
            module = ntpath if platform == 'win32' else posixpath
            dirs = bu_paths.os_dirs( platform, env )
            for a in bu_paths.KINDS:
                for b in bu_paths.KINDS:
                    if a == b or { a, b } == { 'inputs', 'outputs' }:
                        continue
                    inner, outer = module.normcase( dirs[a] ), module.normcase( dirs[b] )
                    self.assertFalse( inner == outer or inner.startswith( outer.rstrip( module.sep ) + module.sep ),
                                      ( platform, a, 'inside', b, dirs ) )

    def test_videos_default_to_app_data_out_of_indexed_folders( self ):
        for platform, env, expected in ( ( 'linux', LINUX_HOME, '/home/u/.local/share/betasuite/library' ),
                                         ( 'win32', WINDOWS_ENV, 'C:\\Users\\u\\AppData\\Local\\BetaSuite\\library' ),
                                         ( 'darwin', MAC_HOME, '/Users/u/Library/Application Support/BetaSuite/library.noindex' ) ):
            dirs = bu_paths.os_dirs( platform, env )
            self.assertEqual( ( dirs['inputs'], dirs['outputs'] ), ( expected, expected ), platform )
            for indexed in ( 'Videos', 'Movies', 'Pictures', 'Documents' ):
                self.assertNotIn( indexed, expected.replace( 'AppData', '' ) )
        # Spotlight skips a folder whose name ends in .noindex.
        self.assertTrue( MAC_DEFAULTS['inputs'].endswith( '.noindex' ) )

    def test_app_data_is_everything_but_the_users_media( self ):
        self.assertEqual( set( bu_paths.APP_DATA_KINDS ), set( bu_paths.KINDS ) - { 'inputs', 'outputs' } )


class TestResolveOs( unittest.TestCase ):

    def resolve( self, path, platform='linux', env=LINUX_HOME ):
        return bu_paths.resolve( path, platform, env )

    def test_every_layout_path_lands_in_its_kind( self ):
        for logical, expected in LINUX_LAYOUT.items():
            self.assertEqual( self.resolve( logical ), expected, logical )

    def test_other_paths_are_left_alone( self ):
        for path in LEFT_ALONE:
            self.assertIs( self.resolve( path ), path )
        for value in ( None, 3, ( '../output', ) ):
            self.assertIs( self.resolve( value ), value )

    def test_windows_paths_use_windows_folders_and_either_separator( self ):
        expected = {
            '../output/cache/vid_hashes':        'C:\\Users\\u\\AppData\\Local\\BetaSuite\\cache\\vid_hashes',
            '..\\output\\cache\\vid_hashes':     'C:\\Users\\u\\AppData\\Local\\BetaSuite\\cache\\vid_hashes',
            '..\\resources/model\\v3.4-640m.onnx': 'C:\\Users\\u\\AppData\\Local\\BetaSuite\\data\\model\\v3.4-640m.onnx',
            '../output/censored_vids/':          'C:\\Users\\u\\AppData\\Local\\BetaSuite\\library\\censored_vids\\',
            '../output/logs/betasuite.log':      'C:\\Users\\u\\AppData\\Local\\BetaSuite\\state\\logs\\betasuite.log',
        }
        for logical, path in expected.items():
            self.assertEqual( self.resolve( logical, 'win32', WINDOWS_ENV ), path, logical )

    def test_backslashes_are_a_file_name_on_posix( self ):
        self.assertEqual( self.resolve( '..\\output\\logs' ), '..\\output\\logs' )

    def test_macos_paths( self ):
        self.assertEqual( self.resolve( '../output/cache/shot_cuts', 'darwin', MAC_HOME ),
                          '/Users/u/Library/Caches/BetaSuite/shot_cuts' )
        self.assertEqual( self.resolve( '../resources/stickers/', 'darwin', MAC_HOME ),
                          '/Users/u/Library/Application Support/BetaSuite/data/stickers/' )

    def test_the_cache_override_moves_every_cache_path_and_nothing_else( self ):
        env = dict( LINUX_HOME, BETASUITE_CACHE_DIR='/big' )
        self.assertEqual( self.resolve( '../output/cache/vid_hashes', env=env ), '/big/vid_hashes' )
        self.assertEqual( self.resolve( '../output/cache/file_hashes.json', env=env ), '/big/file_hashes.json' )
        self.assertEqual( self.resolve( '../output/cache/run_keys', env=env ),
                          '/home/u/.local/share/betasuite/data/run_keys' )

    def test_a_bad_override_is_refused_only_for_mapped_paths( self ):
        env = dict( LINUX_HOME, BETASUITE_CACHE_DIR='rel' )
        with self.assertRaises( bu_paths.PathsError ):
            self.resolve( '../output/cache', env=env )
        self.assertEqual( self.resolve( '/abs/x', env=env ), '/abs/x' )
        # A kind it does not move does not consult it.
        self.assertEqual( self.resolve( '../output/logs', env=env ), '/home/u/.local/state/betasuite/logs' )


class TestLegacyTable( unittest.TestCase ):

    def test_rows_are_well_formed_and_unique( self ):
        prefixes = [ row[0] for row in bu_paths.LEGACY_LAYOUT ]
        self.assertEqual( len( prefixes ), len( set( prefixes ) ) )
        for prefix, kind, sub in bu_paths.LEGACY_LAYOUT:
            self.assertIn( prefix.split( '/' )[0], ( 'resources', 'output' ) )
            self.assertIn( kind, bu_paths.KINDS )
            self.assertNotIn( '\\', prefix + sub )
        self.assertEqual( set( bu_paths.LEGACY_KIND_DIRS ), set( bu_paths.KINDS ) )

    def test_the_longest_prefix_wins( self ):
        self.assertEqual( bu_paths.legacy_entry( ( 'output', 'cache', 'run_keys', 'k.json' ) ),
                          ( 'output/cache/run_keys', 'data', 'run_keys', ( 'k.json', ) ) )
        self.assertEqual( bu_paths.legacy_entry( ( 'output', 'cache', 'vid_hashes' ) ),
                          ( 'output/cache', 'cache', '', ( 'vid_hashes', ) ) )
        self.assertEqual( bu_paths.legacy_entry( ( 'output', 'cachex' ) ), ( 'output', 'state', '', ( 'cachex', ) ) )
        self.assertIsNone( bu_paths.legacy_entry( ( 'elsewhere', 'x' ) ) )
        self.assertIsNone( bu_paths.legacy_entry( () ) )

    def test_resolve_agrees_with_the_table_for_every_row( self ):
        # What the migration relies on: a file at <1.x>/<prefix>/x is
        # copied to exactly where resolve('../<prefix>/x') looks.
        dirs = bu_paths.os_dirs( 'linux', LINUX_HOME )
        for prefix, kind, sub in bu_paths.LEGACY_LAYOUT:
            expected = posixpath.join( dirs[kind], *[ p for p in ( sub, 'x' ) if p ] )
            self.assertEqual( bu_paths.resolve( '../%s/x'%( prefix, ), 'linux', LINUX_HOME ),
                              expected, prefix )

    def test_in_the_1x_shape_only_run_keys_sit_outside_their_kind_folder( self ):
        # kind_dir() under a data root is for files 2.x adds; run keys are
        # the one 1.x folder that is not inside its kind's 1.x folder.
        outside = []
        for prefix, kind, sub in bu_paths.LEGACY_LAYOUT:
            inside = '/'.join( p for p in ( bu_paths.LEGACY_KIND_DIRS[kind], sub ) if p )
            if inside != prefix:
                outside.append( prefix )
        self.assertEqual( outside, [ 'output/cache/run_keys' ] )


class _EnvCase( unittest.TestCase ):

    def setUp( self ):
        patcher = mock.patch.dict( os.environ )
        patcher.start()
        self.addCleanup( patcher.stop )
        for name in ( bu_paths.ENV_VAR, ) + tuple( bu_paths.OVERRIDES.values() ):
            os.environ.pop( name, None )
        self._tmp = tempfile.TemporaryDirectory( prefix='betasuite-paths-' )
        self.addCleanup( self._tmp.cleanup )
        self.tmp = os.path.realpath( self._tmp.name )


class TestDataRootShape( _EnvCase ):

    def test_kind_dirs_inside_a_1x_tree( self ):
        os.environ[bu_paths.ENV_VAR] = self.tmp
        for kind, inside in bu_paths.LEGACY_KIND_DIRS.items():
            self.assertEqual( bu_paths.kind_dir( kind ), os.path.join( self.tmp, *inside.split( '/' ) ) )

    def test_overrides_cannot_be_combined_with_a_data_root( self ):
        os.environ[bu_paths.ENV_VAR] = self.tmp
        for name in bu_paths.OVERRIDES.values():
            with mock.patch.dict( os.environ, { name: self.tmp } ):
                for call in ( lambda: bu_paths.resolve( '../output/logs' ), lambda: bu_paths.kind_dir( 'state' ),
                              lambda: bu_paths.legacy_path( 'output' ) ):
                    with self.assertRaises( bu_paths.PathsError ) as raised:
                        call()
                    self.assertIn( name, str( raised.exception ) )
                    self.assertIn( bu_paths.ENV_VAR, str( raised.exception ) )

    def test_a_data_root_error_is_a_paths_error( self ):
        self.assertTrue( issubclass( bu_paths.DataRootError, bu_paths.PathsError ) )
        self.assertTrue( issubclass( bu_paths.PathsError, ValueError ) )


class TestLegacyPath( _EnvCase ):

    def test_os_layout( self ):
        cases = {
            ( 'output', 'logs' ):                    '/home/u/.local/state/betasuite/logs',
            ( 'resources', 'model' ):                '/home/u/.local/share/betasuite/data/model',
            ( 'output', 'cache', '**', '*.checkpoint' ): '/home/u/.cache/betasuite/**/*.checkpoint',
            ( 'bar_smoothing_test_results', ):       '/home/u/.local/state/betasuite/bar_smoothing_test_results',
            ():                                      '/home/u/.local/state/betasuite',
        }
        for parts, expected in cases.items():
            self.assertEqual( bu_paths.legacy_path( *parts, platform='linux', environ=LINUX_HOME ), expected, parts )

    def test_windows( self ):
        self.assertEqual( bu_paths.legacy_path( 'output', 'logs', platform='win32', environ=WINDOWS_ENV ),
                          'C:\\Users\\u\\AppData\\Local\\BetaSuite\\state\\logs' )

    def test_under_a_data_root_it_is_the_same_parts_under_the_root( self ):
        os.environ[bu_paths.ENV_VAR] = self.tmp
        self.assertEqual( bu_paths.legacy_path( 'output', 'logs' ), os.path.join( self.tmp, 'output', 'logs' ) )
        self.assertEqual( bu_paths.legacy_path( 'bar_smoothing_test_results' ),
                          os.path.join( self.tmp, 'bar_smoothing_test_results' ) )
        self.assertEqual( bu_paths.legacy_path(), self.tmp )


class TestLegacyTreeHint( _EnvCase ):

    def setUp( self ):
        super().setUp()
        self.repo = os.path.join( self.tmp, 'checkout', 'repo' )
        os.makedirs( self.repo )
        self.env = dict( HOME=os.path.join( self.tmp, 'home' ), USERPROFILE=os.path.join( self.tmp, 'home' ),
                         XDG_DATA_HOME=os.path.join( self.tmp, 'xdg-data' ),
                         LOCALAPPDATA=os.path.join( self.tmp, 'xdg-data' ) )
        self.data = bu_paths.kind_dir( 'data', environ=self.env )
        self.assertTrue( self.data.startswith( os.path.join( self.tmp, 'xdg-data' ) ), self.data )

    def test_a_1x_tree_beside_the_code_with_no_data_folder_yet( self ):
        self.assertIsNone( bu_paths.legacy_tree_hint( self.repo, self.env ) )
        os.mkdir( os.path.join( self.tmp, 'checkout', 'output' ) )
        self.assertIsNone( bu_paths.legacy_tree_hint( self.repo, self.env ), "a stray output/ is not a 1.x tree" )
        os.mkdir( os.path.join( self.tmp, 'checkout', 'resources' ) )
        hint = bu_paths.legacy_tree_hint( self.repo, self.env )
        self.assertIn( os.path.join( self.tmp, 'checkout' ), hint )
        self.assertIn( bu_paths.ENV_VAR, hint )

    def test_quiet_once_the_data_folder_exists_or_under_a_data_root( self ):
        os.mkdir( os.path.join( self.tmp, 'checkout', 'resources' ) )
        os.makedirs( self.data )
        self.assertIsNone( bu_paths.legacy_tree_hint( self.repo, self.env ) )
        os.rmdir( self.data )
        self.assertIsNotNone( bu_paths.legacy_tree_hint( self.repo, self.env ) )
        self.assertIsNone( bu_paths.legacy_tree_hint( self.repo, dict( self.env, **{ bu_paths.ENV_VAR: self.tmp } ) ) )

    def test_a_problem_is_no_hint( self ):
        os.mkdir( os.path.join( self.tmp, 'checkout', 'resources' ) )
        self.assertIsNone( bu_paths.legacy_tree_hint( self.repo, { bu_paths.ENV_VAR: 'relative' } ) )


class TestLocalConfigPlace( _EnvCase ):

    def test_os_layout_reads_the_config_folder( self ):
        self.assertEqual( bu_local.default_path(),
                          os.path.join( bu_paths.kind_dir( 'config' ), bu_local.FILE_NAME ) )
        self.assertNotEqual( bu_local.default_path(), bu_local.repo_path() )

    def test_a_data_root_keeps_it_beside_the_code( self ):
        os.environ[bu_paths.ENV_VAR] = self.tmp
        self.assertEqual( bu_local.default_path(), bu_local.repo_path() )
        self.assertEqual( bu_local.repo_path(),
                          os.path.join( os.path.dirname( os.path.abspath( bu_local.__file__ ) ), bu_local.FILE_NAME ) )

    def test_the_old_machine_config_follows_the_layout( self ):
        self.assertEqual( bu_local.legacy_machine_path(),
                          os.path.normpath( os.path.join( bu_paths.kind_dir( 'config' ), 'betaconfig_machine.py' ) ) )
        os.environ[bu_paths.ENV_VAR] = self.tmp
        self.assertEqual( bu_local.legacy_machine_path(),
                          os.path.join( self.tmp, 'resources', 'betaconfig_machine.py' ) )

    def _apply( self, path, repo_file ):
        return bu_local.apply( {}, path=path, legacy_path=os.path.join( self.tmp, 'absent.py' ),
                               repo_file=repo_file, stderr=mock.Mock() )

    def test_a_file_beside_the_code_that_is_not_read_is_warned_about( self ):
        used = os.path.join( self.tmp, 'config', bu_local.FILE_NAME )
        beside = os.path.join( self.tmp, 'repo', bu_local.FILE_NAME )
        loaded = self._apply( used, beside )
        self.assertIsNone( bu_local.ignored_repo_warning( loaded ), "no file beside the code" )
        os.makedirs( os.path.dirname( beside ) )
        with open( beside, 'w' ) as handle:
            handle.write( 'encode_crf = 20\n' )
        loaded = self._apply( used, beside )
        warning = bu_local.ignored_repo_warning( loaded )
        self.assertIn( beside, warning )
        self.assertIn( used, warning )
        self.assertIn( bu_paths.ENV_VAR, warning )
        self.assertIsNone( bu_local.ignored_repo_warning( self._apply( beside, beside ) ), "it is the file read" )
        self.assertIsNone( bu_local.ignored_repo_warning( None ) )

    def test_no_warning_when_the_layer_is_off( self ):
        beside = os.path.join( self.tmp, bu_local.FILE_NAME )
        open( beside, 'w' ).close()
        with mock.patch.dict( os.environ, { bu_local.ENV_VAR: 'none' } ):
            loaded = bu_local.apply( {}, repo_file=beside )
        self.assertIsNone( bu_local.ignored_repo_warning( loaded ) )


class TestRunStartLog( _EnvCase ):

    def _log( self, repo_dir ):
        import logging
        import betautils_log as bu_log
        logger = logging.getLogger( 'test_paths.run_start' )
        logger.propagate = False
        with self.assertLogs( logger, level='DEBUG' ) as logs:
            bu_log.log_data_layout( logger, repo_dir=repo_dir )
        return logs.output

    def test_names_the_layout_and_warns_about_a_1x_tree_left_behind( self ):
        repo = os.path.join( self.tmp, 'checkout', 'repo' )
        os.makedirs( repo )
        lines = self._log( repo )
        self.assertEqual( len( lines ), 1 )
        self.assertIn( 'OS folders by kind', lines[0] )
        os.mkdir( os.path.join( self.tmp, 'checkout', 'output' ) )
        with mock.patch.object( bu_paths, 'legacy_tree_hint', return_value='the hint' ):
            lines = self._log( repo )
        self.assertEqual( lines[-1], 'WARNING:test_paths.run_start:the hint' )
        os.environ[bu_paths.ENV_VAR] = self.tmp
        self.assertIn( 'the 1.x layout under %s'%( self.tmp, ), self._log( repo )[0] )


class TestThisPlatform( _EnvCase ):
    """The defaults with no platform or environment passed in."""

    def test_the_real_platform_and_environment_are_the_defaults( self ):
        self.assertEqual( bu_paths.os_dirs(), bu_paths.os_dirs( None, os.environ ) )
        for kind in bu_paths.KINDS:
            self.assertTrue( os.path.isabs( bu_paths.kind_dir( kind ) ), kind )

    def test_native_separators( self ):
        path = bu_paths.resolve( '../output/cache/vid_hashes' )
        module = ntpath if os.name == 'nt' else posixpath
        self.assertEqual( path, module.join( bu_paths.kind_dir( 'cache' ), 'vid_hashes' ) )


if __name__ == '__main__':
    unittest.main()
