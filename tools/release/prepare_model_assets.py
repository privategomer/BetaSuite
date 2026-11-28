"""
prepare_model_assets.py - build the folder of model files the project hosts
as a GitHub release (maintainers only).

The folder holds, for every model in assets/models/catalog.json:
  - the .onnx file the app downloads, copied from the model folder and
    checked against its pinned SHA-256
  - its upstream source files (the .pt, .pth or training checkpoint), the
    preferred form for modification, mirrored beside it so it stays
    available as long as the binary is (AGPL-3.0 section 6). Each is
    fetched from upstream, or copied from --sources-dir, and checked
    against its pinned SHA-256
plus each upstream license file, taken at the release tag the files came
from, NOTICE-models.txt, RELEASE_NOTES.md (the release text) and
SHA256SUMS. Run upload_model_assets.py on the folder afterwards.

    python3 tools/release/prepare_model_assets.py                 # asks for the folder
    python3 tools/release/prepare_model_assets.py --out /some/folder --yes

Re-running is safe: files already in the folder with the right hash are
kept. A model missing from the model folder is an error; install it
first (the app's Models page, "I already have the file", or
export_models.py for a person model, whose new hash then goes in the
catalog). NudeNet's GitHub release files are served to signed-in users
only; when a plain download gets a web page, the GitHub CLI (gh, signed
in) is tried.

Every level goes to the log file; --log-level filters the terminal.
Exit status 1 if anything is missing or does not match.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )
import betautils_models as bu_models  # noqa: E402  (stdlib only)
import betautils_paths as bu_paths  # noqa: E402
import betautils_log as bu_log  # noqa: E402

SUMS = 'SHA256SUMS'
NOTICE = 'NOTICE-models.txt'
NOTES = 'RELEASE_NOTES.md'
LICENSE_TITLES = { 'AGPL-3.0': 'GNU AFFERO GENERAL PUBLIC LICENSE',
                   'GPL-3.0': 'GNU GENERAL PUBLIC LICENSE',
                   'Apache-2.0': 'Apache License' }
GITHUB_ASSET = re.compile( r'^https://github\.com/([^/]+/[^/]+)/releases/download/([^/]+)/([^/]+)$' )


class Mismatch( Exception ):
    pass


def sha256_file( path ):
    digest = hashlib.sha256()
    with open( path, 'rb' ) as handle:
        for chunk in iter( lambda: handle.read( 4 * 1024 * 1024 ), b'' ):
            digest.update( chunk )
    return digest.hexdigest()


def has( path, sha256 ):
    return os.path.isfile( path ) and sha256_file( path ) == sha256


def place_checked( source, dest, sha256 ):
    """Copy source to dest through a temporary file, only when its hash matches."""
    if sha256_file( source ) != sha256:
        raise Mismatch( '%s is not the pinned file'%( source, ) )
    part = dest + '.part'
    shutil.copyfile( source, part )
    os.replace( part, dest )


def download( url, dest, log ):
    """Stream url to dest. Raises on a network error or when a web page comes back."""
    log.debug( 'GET %s'%( url, ) )
    request = urllib.request.Request( url, headers={ 'User-Agent': 'BetaSuite-release' } )
    with urllib.request.urlopen( request, timeout=300 ) as response, open( dest, 'wb' ) as out:
        shutil.copyfileobj( response, out, 4 * 1024 * 1024 )
    with open( dest, 'rb' ) as handle:
        head = handle.read( 64 ).lstrip().lower()
    if head.startswith( b'<!doctype html' ) or head.startswith( b'<html' ):
        raise Mismatch( 'a web page came back instead of the file' )


def gh_download( url, dest, log, run=subprocess.run ):
    """Fetch a GitHub release asset with the signed-in GitHub CLI."""
    match = GITHUB_ASSET.match( url )
    if not match or not shutil.which( 'gh' ):
        raise Mismatch( 'no GitHub CLI fallback for %s'%( url, ) )
    repo, tag, name = match.groups()
    log.info( 'trying the GitHub CLI for %s'%( name, ) )
    result = run( [ 'gh', 'release', 'download', tag, '-R', repo, '-p', name, '-O', dest, '--clobber' ],
                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True )
    log.debug( result.stdout )
    if result.returncode != 0:
        raise Mismatch( 'gh release download failed (exit %d)'%( result.returncode, ) )


def fetch_source( source, out_dir, sources_dir, log ):
    """Put one upstream source file in out_dir, checked. Returns True when it is there."""
    dest = os.path.join( out_dir, source['file'] )
    if has( dest, source['sha256'] ):
        log.info( 'present: %s'%( source['file'], ) )
        return True
    if sources_dir and has( os.path.join( sources_dir, source['file'] ), source['sha256'] ):
        place_checked( os.path.join( sources_dir, source['file'] ), dest, source['sha256'] )
        log.info( 'copied from the sources folder: %s'%( source['file'], ) )
        return True
    with tempfile.TemporaryDirectory( dir=out_dir ) as work:
        part = os.path.join( work, source['file'] )
        for attempt in ( download, gh_download ):
            try:
                attempt( source['upstream'], part, log )
                place_checked( part, dest, source['sha256'] )
                log.info( 'downloaded: %s (%.1f MB)'%( source['file'], os.path.getsize( dest ) / 1e6 ) )
                return True
            except ( OSError, Mismatch ) as err:
                log.warning( '%s: %s'%( source['file'], err ) )
    log.error( 'could not get %s from %s'%( source['file'], source['upstream'] ) )
    return False


def fetch_license( key, lic, out_dir, log ):
    dest = os.path.join( out_dir, lic['file'] )
    part = dest + '.part'
    try:
        download( lic['url'], part, log )
        with open( part, 'r', encoding='utf-8', errors='replace' ) as handle:
            head = handle.read( 400 )
        title = LICENSE_TITLES.get( lic['spdx'] )
        if title and title not in head:
            raise Mismatch( '%s does not read as %s'%( lic['url'], lic['spdx'] ) )
        os.replace( part, dest )
    except ( OSError, Mismatch ) as err:
        if os.path.exists( part ):
            os.remove( part )
        log.error( 'license %s: %s'%( key, err ) )
        return False
    log.info( 'license: %s (%s)'%( lic['file'], lic['spdx'] ) )
    return True


def notice_text( catalog ):
    lines = [ 'Model files hosted by the BetaSuite project',
              '',
              'BetaSuite downloads these files for its detectors and person segmenters. Each keeps its own',
              'license, given beside it. The source each .onnx file was converted from (the upstream .pt,',
              '.pth or training checkpoint, the preferred form for modifying it) is hosted beside it, unchanged.',
              '' ]
    for key, lic in catalog['licenses'].items():
        rows = [ row for row in catalog['models'] if row['license'] == key ]
        if not rows:
            continue
        lines += [ '%s (%s)'%( lic['project'], lic['home'] ),
                   '  License: %s, in %s, from %s'%( lic['spdx'], lic['file'], lic['url'] ) ]
        for row in rows:
            lines.append( '  %s: converted from %s'%( row['file'], ', '.join( s['file'] for s in row['sources'] ) ) )
            for source in row['sources']:
                lines.append( '    %s: unchanged copy of %s'%( source['file'], source['upstream'] ) )
        lines.append( '' )
    return '\n'.join( lines )


def notes_text( catalog ):
    lines = [ 'Model files for BetaSuite\'s Models page and setup. Not an app release.', '',
              'The app checks every file against the SHA-256 in `assets/models/catalog.json` before it uses it.',
              'Licenses and sources: `%s`, and the `LICENSE-*` files.'%( NOTICE, ), '',
              '| File | Size | License | SHA-256 | Source |', '|---|---|---|---|---|' ]
    for row in catalog['models']:
        lic = catalog['licenses'][ row['license'] ]
        lines.append( '| %s | %.1f MB | %s | `%s` | %s |'%(
            row['file'], row['size'] / 1e6, lic['spdx'], row['sha256'],
            ', '.join( source['file'] for source in row['sources'] ) ) )
    return '\n'.join( lines ) + '\n'


def expected_files( catalog ):
    names = [ row['file'] for row in catalog['models'] ]
    names += [ source['file'] for row in catalog['models'] for source in row['sources'] ]
    names += [ lic['file'] for lic in catalog['licenses'].values() ]
    return names + [ NOTICE, NOTES ]


def write_sums( out_dir, names ):
    lines = [ '%s  %s'%( sha256_file( os.path.join( out_dir, name ) ), name ) for name in sorted( names ) ]
    with open( os.path.join( out_dir, SUMS ), 'w', encoding='utf-8', newline='\n' ) as out:
        out.write( '\n'.join( lines ) + '\n' )


def prepare( out_dir, model_dir, sources_dir, log, catalog=None ):
    """Build the folder. Returns the number of problems."""
    catalog = catalog or bu_models.load()
    os.makedirs( out_dir, exist_ok=True )
    problems = 0
    for row in catalog['models']:
        dest = os.path.join( out_dir, row['file'] )
        if has( dest, row['sha256'] ):
            log.info( 'present: %s'%( row['file'], ) )
        elif has( os.path.join( model_dir, row['file'] ), row['sha256'] ):
            place_checked( os.path.join( model_dir, row['file'] ), dest, row['sha256'] )
            log.info( 'copied: %s'%( row['file'], ) )
        else:
            log.error( '%s is missing from %s or is not the pinned file; install it first'%( row['file'], model_dir ) )
            problems += 1
        for source in row['sources']:
            if not fetch_source( source, out_dir, sources_dir, log ):
                problems += 1
    for key, lic in catalog['licenses'].items():
        if not fetch_license( key, lic, out_dir, log ):
            problems += 1
    for name, text in ( ( NOTICE, notice_text( catalog ) ), ( NOTES, notes_text( catalog ) ) ):
        with open( os.path.join( out_dir, name ), 'w', encoding='utf-8', newline='\n' ) as out:
            out.write( text )
    names = expected_files( catalog )
    extra = sorted( set( os.listdir( out_dir ) ) - set( names ) - { SUMS } )
    if extra:
        log.warning( 'not part of the release, left out of %s: %s'%( SUMS, ', '.join( extra ) ) )
    if problems:
        log.error( '%d problem(s); %s not written'%( problems, SUMS ) )
        return problems
    write_sums( out_dir, names )
    total = sum( os.path.getsize( os.path.join( out_dir, name ) ) for name in names )
    log.info( 'ready: %d files, %.0f MB, in %s'%( len( names ) + 1, total / 1e6, out_dir ) )
    return 0


def _ask( prompt, default, log ):
    log.debug( 'AWAITING INPUT: %s'%( prompt, ) )
    try:
        answer = input( '%s [%s] '%( prompt, default ) ).strip()
    except EOFError:
        answer = ''
    log.debug( 'input received: %s'%( answer, ) )
    return answer or default


def main( argv=None ):
    catalog = bu_models.load()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--out', help='the release folder to build' )
    parser.add_argument( '--model-dir', default=bu_models.model_dir(),
                         help='where the .onnx files are (default: the app\'s model folder)' )
    parser.add_argument( '--sources-dir', help='a folder of upstream source files already downloaded' )
    parser.add_argument( '--yes', action='store_true', help='take the defaults instead of asking' )
    parser.add_argument( '--log-level', default='info', choices=list( bu_log.LEVEL_NAMES ) )
    parser.add_argument( '--log-file', default=os.path.join( bu_paths.legacy_path( 'output', 'logs' ),
                                                             'prepare_model_assets.log' ) )
    args = parser.parse_args( argv )
    log = bu_log.tool_logger( 'prepare_model_assets', args.log_file, file_level='trace', console_level=args.log_level )
    log.info( 'log: %s'%( os.path.abspath( args.log_file ), ) )
    default_out = os.path.abspath( 'release-assets-%s'%( catalog['release'], ) )
    out_dir = args.out or ( default_out if args.yes else _ask( 'Folder to build the release in', default_out, log ) )
    log.info( 'building %s in %s from %s'%( catalog['release'], out_dir, args.model_dir ) )
    return 1 if prepare( out_dir, args.model_dir, args.sources_dir, log, catalog ) else 0


if __name__ == '__main__':
    sys.exit( main() )
