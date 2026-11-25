"""
ffmpeg_pin.py - fetch the one pinned ffmpeg build (tools/lib/ffmpeg_pin.json).

CI installs ffmpeg this way, and the golden output test compares output
bytes only under this build. The desktop app's bundle is meant to read
the same file, so the app and CI never use different ffmpeg builds.

    python tools/lib/ffmpeg_pin.py <target folder> [--build linux64]
        [--log-file FILE] [--log-level info] [--console-level info]

Downloads the build's archive, checks its SHA-256 against the pin (a
mismatch stops with exit 1 and nothing installed), unpacks ffmpeg and
ffprobe into <target folder>/bin, writes <target folder>/version.txt,
and prints the bin folder. A target that already holds this version is
left as it is.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

HERE = os.path.dirname( os.path.abspath( __file__ ) )
PIN_PATH = os.path.join( HERE, 'ffmpeg_pin.json' )
sys.path.insert( 0, os.path.dirname( os.path.dirname( HERE ) ) )

LEVELS = ( 'trace', 'debug', 'info', 'warn', 'error' )


def load_pin( path=PIN_PATH ):
    with open( path, encoding='utf-8' ) as handle:
        return json.load( handle )


def sha256_of( path ):
    digest = hashlib.sha256()
    with open( path, 'rb' ) as handle:
        for block in iter( lambda: handle.read( 1 << 20 ), b'' ):
            digest.update( block )
    return digest.hexdigest()


def _members( names ):
    """ffmpeg and ffprobe inside an archive's bin folder: name in archive -> name on disk."""
    wanted = {}
    for name in names:
        parts = name.replace( '\\', '/' ).split( '/' )
        if len( parts ) >= 2 and parts[-2] == 'bin' and parts[-1] in ( 'ffmpeg', 'ffprobe', 'ffmpeg.exe', 'ffprobe.exe' ):
            wanted[name] = parts[-1]
    return wanted


def unpack( archive, bin_dir ):
    os.makedirs( bin_dir, exist_ok=True )
    if archive.endswith( '.zip' ):
        with zipfile.ZipFile( archive ) as bundle:
            members = _members( bundle.namelist() )
            for member, name in members.items():
                with bundle.open( member ) as source, open( os.path.join( bin_dir, name ), 'wb' ) as target:
                    shutil.copyfileobj( source, target )
    else:
        with tarfile.open( archive ) as bundle:
            members = _members( bundle.getnames() )
            for member, name in members.items():
                source = bundle.extractfile( member )
                with source, open( os.path.join( bin_dir, name ), 'wb' ) as target:
                    shutil.copyfileobj( source, target )
    for name in os.listdir( bin_dir ):
        os.chmod( os.path.join( bin_dir, name ), 0o755 )
    return sorted( set( members.values() ) )


def install( target, build, logger, pin=None ):
    """Install the pinned build into target. Returns the bin folder. Raises ValueError on a bad checksum."""
    pin = pin or load_pin()
    entry = pin['builds'][build]
    bin_dir = os.path.join( target, 'bin' )
    version_path = os.path.join( target, 'version.txt' )
    stamp = '%s %s\n'%( pin['version'], build )
    if os.path.exists( version_path ) and open( version_path, encoding='utf-8' ).read() == stamp:
        logger.info( "ffmpeg %s (%s) already in %s", pin['version'], build, bin_dir )
        return bin_dir
    os.makedirs( target, exist_ok=True )
    with tempfile.TemporaryDirectory( dir=target ) as scratch:
        archive = os.path.join( scratch, os.path.basename( entry['url'] ) )
        logger.info( "downloading %s", entry['url'] )
        with urllib.request.urlopen( entry['url'], timeout=120 ) as response, open( archive, 'wb' ) as out:
            shutil.copyfileobj( response, out )
        found = sha256_of( archive )
        logger.debug( "sha256 %s, pinned %s", found, entry['sha256'] )
        if found != entry['sha256']:
            raise ValueError( "checksum mismatch for %s: got %s, pinned %s"%( entry['url'], found, entry['sha256'] ) )
        shutil.rmtree( bin_dir, ignore_errors=True )
        names = unpack( archive, bin_dir )
        logger.info( "unpacked %s into %s", ', '.join( names ), bin_dir )
    with open( version_path, 'w', encoding='utf-8' ) as handle:
        handle.write( stamp )
    return bin_dir


def main( argv=None ):
    parser = argparse.ArgumentParser( description="Fetch the pinned ffmpeg build." )
    parser.add_argument( 'target', help="folder to install into (bin/ and version.txt go inside)" )
    parser.add_argument( '--build', default='linux64', help="build from the pin file (default linux64)" )
    parser.add_argument( '--log-file', default=None, help="log file (default <target>/ffmpeg_pin.log)" )
    parser.add_argument( '--log-level', default='info', choices=LEVELS )
    parser.add_argument( '--console-level', default='info', choices=LEVELS )
    args = parser.parse_args( argv )
    import betautils_log as bu_log
    os.makedirs( args.target, exist_ok=True )
    logger = bu_log.tool_logger( 'ffmpeg_pin', log_path=args.log_file or os.path.join( args.target, 'ffmpeg_pin.log' ),
                                 file_level=args.log_level, console_level=args.console_level,
                                 console_stream=sys.stderr )
    try:
        bin_dir = install( args.target, args.build, logger )
    except ( ValueError, OSError, KeyError ) as err:
        logger.error( "%s", err )
        return 1
    print( bin_dir )
    return 0


if __name__ == '__main__':
    sys.exit( main() )
