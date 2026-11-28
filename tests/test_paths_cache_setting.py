"""
test_paths_cache_setting.py - the cache folder setting (cache_dir in
betaconfig_local.py) as the path resolver reads it.

What each test prevents:
  - the setting ignored, or winning over BETASUITE_CACHE_DIR
  - a relative or non-text value used as a folder
  - a file that is broken, or turned off with BETASUITE_LOCAL_CONFIG,
    moving the cache anyway
  - a change to the file not seen until the process restarts its memo
  - the setting moving any kind but the cache, or the 1.x layout
"""

import os
import shutil
import tempfile
import time
import unittest

import betautils_paths as bu_paths


class TestCacheSetting( unittest.TestCase ):

    def setUp( self ):
        self.home = tempfile.mkdtemp( prefix='paths-' )
        self.addCleanup( shutil.rmtree, self.home, True )
        self.config_dir = os.path.join( self.home, '.config', 'betasuite' )
        os.makedirs( self.config_dir )
        self.file = os.path.join( self.config_dir, 'betaconfig_local.py' )
        self.env = { 'HOME': self.home }
        self.default = os.path.join( self.home, '.cache', 'betasuite' )
        self.moved = os.path.join( self.home, 'elsewhere', 'cache' )

    def write( self, text ):
        with open( self.file, 'w', encoding='utf-8' ) as handle:
            handle.write( text )
        # A new modification time even on a coarse clock.
        stamp = time.time() + len( text )
        os.utime( self.file, ( stamp, stamp ) )

    def cache( self, env=None ):
        return bu_paths.kind_dir( 'cache', platform='linux', environ=env or self.env )

    def test_default_without_the_setting( self ):
        self.assertEqual( self.cache(), self.default )
        self.assertEqual( bu_paths.cache_source( 'linux', self.env ), 'default' )

    def test_setting_moves_the_cache( self ):
        self.write( "# mine\ncache_dir = %r\n"%( self.moved, ) )
        self.assertEqual( self.cache(), self.moved )
        self.assertEqual( bu_paths.cache_source( 'linux', self.env ), 'setting' )

    def test_only_the_cache_moves( self ):
        self.write( "cache_dir = %r\n"%( self.moved, ) )
        dirs = bu_paths.os_dirs( 'linux', self.env )
        self.assertEqual( dirs['cache'], self.moved )
        self.assertEqual( dirs['config'], self.config_dir )
        self.assertEqual( dirs['state'], os.path.join( self.home, '.local', 'state', 'betasuite' ) )

    def test_the_variable_wins( self ):
        self.write( "cache_dir = %r\n"%( self.moved, ) )
        env = dict( self.env, BETASUITE_CACHE_DIR=os.path.join( self.home, 'var' ) )
        self.assertEqual( self.cache( env ), os.path.join( self.home, 'var' ) )
        self.assertEqual( bu_paths.cache_source( 'linux', env ), 'variable' )

    def test_unusable_values_are_ignored( self ):
        for text in ( "cache_dir = 'relative/folder'\n", "cache_dir = None\n", "cache_dir = 3\n",
                      "cache_dir = (\n", "cache_dir = open('x')\n" ):
            with self.subTest( text=text ):
                self.write( text )
                self.assertEqual( self.cache(), self.default )

    def test_last_assignment_wins_and_other_names_are_ignored( self ):
        self.write( "gpu_enabled = 1\ncache_dir = '/first'\ncache_dir = %r\nsticker_dir = '/not/it'\n"%(
            self.moved, ) )
        self.assertEqual( self.cache(), self.moved )

    def test_local_config_turned_off( self ):
        self.write( "cache_dir = %r\n"%( self.moved, ) )
        self.assertEqual( self.cache( dict( self.env, BETASUITE_LOCAL_CONFIG='none' ) ), self.default )

    def test_turned_off_even_beside_a_file_named_none( self ):
        here = os.getcwd()
        os.chdir( self.home )
        self.addCleanup( os.chdir, here )
        with open( os.path.join( self.home, 'none' ), 'w', encoding='utf-8' ) as handle:
            handle.write( "cache_dir = %r\n"%( self.moved, ) )
        for value in ( 'none', 'off', '' ):
            self.assertEqual( self.cache( dict( self.env, BETASUITE_LOCAL_CONFIG=value ) ), self.default, value )

    def test_local_config_elsewhere( self ):
        other = os.path.join( self.home, 'other.py' )
        with open( other, 'w', encoding='utf-8' ) as handle:
            handle.write( "cache_dir = %r\n"%( self.moved, ) )
        self.assertEqual( self.cache( dict( self.env, BETASUITE_LOCAL_CONFIG=other ) ), self.moved )

    def test_a_changed_file_is_read_again( self ):
        self.write( "cache_dir = %r\n"%( self.moved, ) )
        self.assertEqual( self.cache(), self.moved )
        self.write( "# moved back\n" )
        self.assertEqual( self.cache(), self.default )

    def test_the_1x_layout_ignores_it( self ):
        self.write( "cache_dir = %r\n"%( self.moved, ) )
        root = os.path.join( self.home, 'tree' )
        os.makedirs( root )
        env = dict( self.env, BETASUITE_DATA_ROOT=root )
        self.assertEqual( self.cache( env ), os.path.join( root, 'output', 'cache' ) )


if __name__ == '__main__':
    unittest.main()
