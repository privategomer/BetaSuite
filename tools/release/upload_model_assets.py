"""
upload_model_assets.py - publish the folder prepare_model_assets.py built
as the project's model release on GitHub (maintainers only).

    python3 tools/release/upload_model_assets.py                  # asks for the folder, confirms
    python3 tools/release/upload_model_assets.py --folder /some/folder
    python3 tools/release/upload_model_assets.py --folder /some/folder --verify-only

Steps, each logged:
  1. check every file in the folder against SHA256SUMS, and every model,
     source and license file the catalog lists against its pinned hash
  2. show the GitHub account (gh) and the release it will create or add
     to: the tag and repository of the catalog's first mirror
  3. ask before changing anything (type the tag to confirm; --yes skips)
  4. create the release if it is not there (never marked latest, so the
     app's own releases stay latest), then upload each file it does not
     have yet; re-running after a failure uploads only what is missing
  5. download every file again without signing in and check its hash, so
     the app's one-click download is known to work for anyone

A public release and tag are seen at once and cannot be quietly undone:
this script never deletes or replaces an asset that is already there
with the right size, and it stops when one is there with the wrong size.

Every level goes to the log file; --log-level filters the terminal. Exit
status 1 on any problem.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), '..', '..' ) )
sys.path.insert( 0, REPO_ROOT )
sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
import betautils_log as bu_log  # noqa: E402
import betautils_models as bu_models  # noqa: E402
import betautils_paths as bu_paths  # noqa: E402
import prepare_model_assets as prepare  # noqa: E402

MIRROR = re.compile( r'^https://github\.com/([^/]+/[^/]+)/releases/download/([^/]+)/$' )


class Problem( Exception ):
    pass


def release_target( catalog ):
    """( 'owner/repo', tag ) from the catalog's first mirror."""
    match = MIRROR.match( catalog['mirrors'][0] )
    if not match:
        raise Problem( 'the first mirror is not a GitHub release: %s'%( catalog['mirrors'][0], ) )
    return match.groups()


def read_sums( folder ):
    path = os.path.join( folder, prepare.SUMS )
    if not os.path.isfile( path ):
        raise Problem( '%s is missing; run prepare_model_assets.py first'%( path, ) )
    sums = {}
    with open( path, encoding='utf-8' ) as handle:
        for line in handle:
            if line.strip():
                digest, name = line.rstrip( '\n' ).split( '  ', 1 )
                sums[name] = digest
    return sums


def check_folder( folder, catalog, log ):
    """{ name: sha256 } for every file to publish, all checked. Raises Problem."""
    sums = read_sums( folder )
    pinned = { row['file']: row['sha256'] for row in catalog['models'] }
    pinned.update( { s['file']: s['sha256'] for row in catalog['models'] for s in row['sources'] } )
    missing = sorted( set( prepare.expected_files( catalog ) ) - set( sums ) )
    if missing:
        raise Problem( 'not in %s: %s'%( prepare.SUMS, ', '.join( missing ) ) )
    for name, digest in sorted( sums.items() ):
        if name in pinned and pinned[name] != digest:
            raise Problem( '%s: SHA256SUMS does not match the catalog'%( name, ) )
        actual = prepare.sha256_file( os.path.join( folder, name ) )
        if actual != digest:
            raise Problem( '%s does not match SHA256SUMS'%( name, ) )
        log.debug( 'checked %s'%( name, ) )
    log.info( 'folder checked: %d files match SHA256SUMS and the catalog'%( len( sums ), ) )
    return sums


def gh( args, log, run=subprocess.run ):
    """Run gh; ( exit code, output )."""
    log.debug( 'gh %s'%( ' '.join( args ), ) )
    result = run( [ 'gh' ] + args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  text=True )
    log.log( bu_log.TRACE, result.stdout )
    return result.returncode, result.stdout


def existing_assets( repo, tag, log, run=subprocess.run ):
    """{ name: size } of the release's assets, or None when there is no release."""
    code, out = gh( [ 'release', 'view', tag, '-R', repo, '--json', 'assets' ], log, run )
    if code != 0:
        if 'not found' in out.lower():
            return None
        raise Problem( 'gh release view failed: %s'%( out.strip(), ) )
    return { asset['name']: asset['size'] for asset in json.loads( out )['assets'] }


def plan( folder, names, assets ):
    """( names to upload, names already there ). Raises Problem for a wrong size already there."""
    upload, there = [], []
    for name in names:
        size = os.path.getsize( os.path.join( folder, name ) )
        if assets is not None and name in assets:
            if assets[name] != size:
                raise Problem( '%s is on the release with %d bytes, not %d; fix it by hand'%(
                    name, assets[name], size ) )
            there.append( name )
        else:
            upload.append( name )
    return upload, there


def publish( folder, repo, tag, upload, create, log, run=subprocess.run ):
    if create:
        code, out = gh( [ 'release', 'create', tag, '-R', repo, '--title', 'Model files (%s)'%( tag, ),
                          '--notes-file', os.path.join( folder, prepare.NOTES ), '--latest=false',
                          '--target', 'main' ], log, run )
        if code != 0:
            raise Problem( 'gh release create failed: %s'%( out.strip(), ) )
        log.info( 'created release %s on %s'%( tag, repo ) )
    for number, name in enumerate( upload, 1 ):
        log.info( 'uploading %d of %d: %s'%( number, len( upload ), name ) )
        code, out = gh( [ 'release', 'upload', tag, '-R', repo, os.path.join( folder, name ) ], log, run )
        if code != 0:
            raise Problem( 'upload of %s failed: %s (run again to continue)'%( name, out.strip() ) )


def verify_public( base, sums, log, opener=urllib.request.urlopen ):
    """Download each file without signing in and check its hash. Returns the names that failed."""
    failed = []
    for name, digest in sorted( sums.items() ):
        request = urllib.request.Request( base + name, headers={ 'User-Agent': 'BetaSuite-release-check' } )
        sha = hashlib.sha256()
        try:
            with opener( request, timeout=300 ) as response:
                for chunk in iter( lambda: response.read( 4 * 1024 * 1024 ), b'' ):
                    sha.update( chunk )
        except OSError as err:
            log.error( '%s: not downloadable without signing in (%s)'%( name, err ) )
            failed.append( name )
            continue
        if sha.hexdigest() != digest:
            log.error( '%s: the public download is not the file uploaded'%( name, ) )
            failed.append( name )
        else:
            log.info( 'public download checked: %s'%( name, ) )
    return failed


def _ask( prompt, log, default='' ):
    log.debug( 'AWAITING INPUT: %s'%( prompt, ) )
    try:
        answer = input( prompt ).strip()
    except EOFError:
        answer = ''
    log.debug( 'input received: %s'%( answer, ) )
    return answer or default


def main( argv=None, run=subprocess.run, opener=urllib.request.urlopen ):
    catalog = bu_models.load()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--folder', help='the folder prepare_model_assets.py built' )
    parser.add_argument( '--yes', action='store_true', help='do not ask before publishing' )
    parser.add_argument( '--verify-only', action='store_true', help='only check the folder and the public downloads' )
    parser.add_argument( '--log-level', default='info', choices=list( bu_log.LEVEL_NAMES ) )
    parser.add_argument( '--log-file', default=os.path.join( bu_paths.legacy_path( 'output', 'logs' ),
                                                             'upload_model_assets.log' ) )
    args = parser.parse_args( argv )
    log = bu_log.tool_logger( 'upload_model_assets', args.log_file, file_level='trace', console_level=args.log_level )
    log.info( 'log: %s'%( os.path.abspath( args.log_file ), ) )
    try:
        repo, tag = release_target( catalog )
        folder = args.folder or _ask( 'Folder to publish (built by prepare_model_assets.py): ', log )
        if not folder:
            raise Problem( 'no folder given' )
        sums = check_folder( folder, catalog, log )
        base = catalog['mirrors'][0]
        if not args.verify_only:
            code, account = gh( [ 'api', 'user', '-q', '.login' ], log, run )
            if code != 0:
                raise Problem( 'gh is not signed in: %s'%( account.strip(), ) )
            assets = existing_assets( repo, tag, log, run )
            # SHA256SUMS goes up too, for anyone downloading by hand.
            upload, there = plan( folder, sorted( sums ) + [ prepare.SUMS ], assets )
            size = sum( os.path.getsize( os.path.join( folder, name ) ) for name in upload ) / 1e6
            log.info( 'account: %s'%( account.strip(), ) )
            log.info( 'release: %s on %s (%s)'%( tag, repo, 'new' if assets is None else 'exists' ) )
            log.info( 'to upload: %d files, %.0f MB; already there: %d'%( len( upload ), size, len( there ) ) )
            if upload or assets is None:
                if not args.yes:
                    answer = _ask( 'This publishes them publicly. Type %s to go ahead: '%( tag, ), log )
                    if answer != tag:
                        log.info( 'nothing changed' )
                        return 1
                publish( folder, repo, tag, upload, assets is None, log, run )
        failed = verify_public( base, sums, log, opener )
        if failed:
            raise Problem( '%d file(s) failed the public download check'%( len( failed ), ) )
    except Problem as err:
        log.error( str( err ) )
        return 1
    log.info( 'done: %s is published and every file downloads without signing in'%( tag, ) )
    return 0


if __name__ == '__main__':
    sys.exit( main() )
