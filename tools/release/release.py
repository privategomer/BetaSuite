"""
release.py - cut a BetaSuite release: bump VERSION, date the CHANGELOG
entry, commit, tag, and optionally push and publish on GitHub.

    python3 tools/release/release.py              # suggest a bump from commits, ask
    python3 tools/release/release.py --bump minor
    python3 tools/release/release.py --dry-run    # show what would happen, change nothing

The bump is suggested from commit subjects since the last v* tag:

    major   "BREAKING" anywhere, or a conventional "type!:" subject
    minor   a subject starting with feat/add/new (or "feat(...):")
    patch   anything else
    none    VERSION isn't tagged yet: release it as-is (first use)

Release notes come from a "## Unreleased" section in CHANGELOG.md if you
keep one; otherwise one is drafted from the commit subjects and you get a
chance to edit it before anything is committed. Every step that changes
the repository or anything remote asks first. All output goes to
tools/release/release.log at every level; --log-level filters the console.

Needs the semver package (requirements-dev.txt); offers to install it.
"""

import argparse
import datetime
import logging
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', '..' ) )
VERSION_FILE = os.path.join( REPO_ROOT, 'VERSION' )
CHANGELOG = os.path.join( REPO_ROOT, 'CHANGELOG.md' )

TRACE = 5
logging.addLevelName( TRACE, 'TRACE' )
LEVELS = { 'trace': TRACE, 'debug': logging.DEBUG, 'info': logging.INFO,
           'warn': logging.WARNING, 'error': logging.ERROR }

log = logging.getLogger( 'release' )


class Abort( Exception ):
    pass


# ---------------------------------------------------------------- environment

def relaunch_in_venv():
    """Re-run with the repo's .venv Python if started outside it."""
    if sys.prefix != sys.base_prefix or os.environ.get( 'BETASUITE_NO_RELAUNCH' ):
        return
    for rel in ( ( '.venv', 'bin', 'python' ), ( '.venv', 'Scripts', 'python.exe' ) ):
        venv_python = os.path.join( REPO_ROOT, *rel )
        if os.path.exists( venv_python ):
            print( 'not in the virtual environment; re-running with %s'%(venv_python), flush=True )
            os.environ['BETASUITE_NO_RELAUNCH'] = '1'
            args = [ venv_python, os.path.abspath( __file__ ) ] + sys.argv[1:]
            if os.name == 'nt':
                sys.exit( subprocess.call( args ) )
            os.execv( venv_python, args )


def setup_logging( log_file, stdout_level ):
    log.setLevel( TRACE )
    fmt = logging.Formatter( '[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S' )
    fh = logging.FileHandler( log_file, encoding='utf-8' )
    fh.setLevel( TRACE )
    fh.setFormatter( fmt )
    log.addHandler( fh )
    sh = logging.StreamHandler( sys.stdout )
    sh.setLevel( LEVELS[stdout_level] )
    sh.setFormatter( fmt )
    log.addHandler( sh )


def ask( prompt, default, assume_yes ):
    """Explicit y/n prompt; --yes or no terminal takes the default (logged)."""
    d = 'y' if default else 'n'
    if assume_yes or not sys.stdin.isatty():
        log.info( "auto-answer '%s': %s"%( d, prompt ) )
        return default
    log.debug( 'AWAITING INPUT: %s'%(prompt) )
    reply = input( '%s [%s] '%( prompt, 'Y/n' if default else 'y/N' ) ).strip().lower() or d
    log.debug( 'input received: %s'%(reply) )
    return reply in ( 'y', 'yes' )


def run( cmd, check=True, capture=True ):
    """Run a command in the repo; everything it prints goes to the log."""
    log.debug( 'run: %s'%( ' '.join( cmd ) ) )
    result = subprocess.run( cmd, cwd=REPO_ROOT, capture_output=True, text=True )
    for line in ( result.stdout + result.stderr ).splitlines():
        log.log( TRACE, '  | %s'%(line) )
    if check and result.returncode != 0:
        raise Abort( '%s failed (exit %d): %s'%( ' '.join( cmd ), result.returncode,
                                                  result.stderr.strip() or result.stdout.strip() ) )
    return result.stdout.strip() if capture else result.returncode


def import_semver( assume_yes ):
    try:
        import semver
        return semver
    except ImportError:
        pass
    if not ask( 'The semver package is not installed. Install requirements-dev.txt into this environment?',
                True, assume_yes ):
        raise Abort( 'semver is required: pip install -r requirements-dev.txt' )
    uv = shutil.which( 'uv' )
    if uv:
        run( [ uv, 'pip', 'install', '--python', sys.executable, '-r', 'requirements-dev.txt' ] )
    else:
        run( [ sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt' ] )
    import semver
    return semver


# ---------------------------------------------------------------- pure helpers (tested)

def suggest_bump( subjects, version_is_tagged ):
    """'none', 'major', 'minor' or 'patch' for these commit subjects."""
    if not version_is_tagged:
        return 'none'
    if any( 'BREAKING' in s or re.match( r'^\w+(\([^)]*\))?!:', s ) for s in subjects ):
        return 'major'
    if any( re.match( r'^(feat|add|new)\b', s, re.IGNORECASE ) for s in subjects ):
        return 'minor'
    return 'patch'


def bump_version( semver, current, bump ):
    v = semver.Version.parse( current )
    return str( { 'none': v, 'major': v.bump_major(), 'minor': v.bump_minor(),
                  'patch': v.bump_patch() }[bump] )


def date_changelog( text, version, today, subjects ):
    """
    Return (new_text, notes) with a dated "## <version> (<today>)" section.

    Uses, in order: an existing "## <version>" section (re-dated), a
    "## Unreleased" section (renamed), or a new section drafted from the
    commit subjects, inserted above the first existing release.
    """
    heading = '## %s (%s)'%( version, today )
    lines = text.splitlines( keepends=True )

    def section_bounds( pattern ):
        for i, line in enumerate( lines ):
            if re.match( pattern, line ):
                end = next( ( j for j in range( i + 1, len( lines ) ) if lines[j].startswith( '## ' ) ),
                            len( lines ) )
                return i, end
        return None

    bounds = section_bounds( r'^## %s\b'%( re.escape( version ) ) ) or section_bounds( r'^## Unreleased\b' )
    if bounds:
        start, end = bounds
        lines[start] = heading + '\n'
        notes = ''.join( lines[start + 1:end] ).strip()
        return ''.join( lines ), notes

    notes = '\n'.join( '- %s'%(s) for s in subjects ) or '- Maintenance release'
    block = '%s\n\n%s\n\n'%( heading, notes )
    first = next( ( i for i, line in enumerate( lines ) if re.match( r'^## ', line ) ), len( lines ) )
    return ''.join( lines[:first] ) + block + ''.join( lines[first:] ), notes


# ---------------------------------------------------------------- release

def main():
    relaunch_in_venv()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--bump', choices=[ 'major', 'minor', 'patch', 'none' ],
                         help='skip the suggestion and use this bump' )
    parser.add_argument( '--dry-run', action='store_true', help='show the plan, change nothing' )
    parser.add_argument( '--yes', action='store_true', help='take the default answer to every question' )
    parser.add_argument( '--log-level', default='info', choices=list( LEVELS ) )
    parser.add_argument( '--log-file', default=os.path.join( os.path.dirname( os.path.abspath( __file__ ) ),
                                                             'release.log' ) )
    args = parser.parse_args()
    setup_logging( args.log_file, args.log_level )

    try:
        return release( args )
    except Abort as e:
        log.error( str( e ) )
        log.error( 'release stopped; nothing after the last "done" line was changed. Log: %s'%( args.log_file ) )
        return 1
    except KeyboardInterrupt:
        log.error( 'interrupted' )
        return 130


def release( args ):
    semver = import_semver( args.yes )

    # --- preflight
    if not os.path.isdir( os.path.join( REPO_ROOT, '.git' ) ):
        raise Abort( 'not a git checkout: %s'%(REPO_ROOT) )
    branch = run( [ 'git', 'rev-parse', '--abbrev-ref', 'HEAD' ] )
    if branch != 'main' and not ask( 'You are on branch %r, not main. Release from here anyway?'%(branch),
                                     False, args.yes ):
        raise Abort( 'switch to main first' )
    dirty = run( [ 'git', 'status', '--porcelain', '--untracked-files=no' ] )
    if dirty:
        raise Abort( 'uncommitted changes; commit or stash them first:\n%s'%(dirty) )

    with open( VERSION_FILE, encoding='utf-8' ) as f:
        current = f.read().strip()
    semver.Version.parse( current )   # raises on a malformed VERSION

    tags = set( run( [ 'git', 'tag', '--list', 'v*' ] ).split() )
    version_is_tagged = ( 'v' + current ) in tags
    last_tag = run( [ 'git', 'describe', '--tags', '--abbrev=0', '--match', 'v[0-9]*' ], check=False ) or None
    if last_tag and not last_tag.startswith( 'v' ):
        last_tag = None
    log_range = '%s..HEAD'%(last_tag) if last_tag else 'HEAD'
    subjects = [ s for s in run( [ 'git', 'log', '--format=%s', '--no-merges', log_range ] ).splitlines() if s ]

    log.info( 'VERSION is %s (%s); last tag %s; %d commit(s) since'%(
        current, 'tagged' if version_is_tagged else 'not tagged yet', last_tag or '(none)', len( subjects ) ) )
    for s in subjects:
        log.info( '  - %s'%(s) )
    if version_is_tagged and not subjects:
        raise Abort( 'nothing to release: no commits since %s'%(last_tag) )

    # --- choose the version
    suggestion = suggest_bump( subjects, version_is_tagged )
    bump = args.bump or suggestion
    if not args.bump and not args.yes and sys.stdin.isatty():
        log.debug( 'AWAITING INPUT: bump level' )
        reply = input( 'Bump [major/minor/patch/none] (suggested: %s): '%(suggestion) ).strip().lower()
        log.debug( 'input received: %s'%(reply) )
        bump = reply or suggestion
    if bump not in ( 'major', 'minor', 'patch', 'none' ):
        raise Abort( 'unknown bump %r'%(bump) )
    new = bump_version( semver, current, bump )
    if ( 'v' + new ) in tags:
        raise Abort( 'tag v%s already exists. Pick a bigger bump'%(new) )
    log.info( 'releasing %s -> %s (%s)'%( current, new, bump ) )

    # --- changelog
    today = datetime.date.today().isoformat()
    with open( CHANGELOG, encoding='utf-8' ) as f:
        changelog = f.read()
    new_changelog, notes = date_changelog( changelog, new, today, subjects )
    log.info( 'release notes:\n%s'%(notes) )

    if args.dry_run:
        log.info( 'dry run: would write VERSION=%s, update CHANGELOG.md, commit, and tag v%s'%( new, new ) )
        return 0
    if not ask( 'Write VERSION=%s and the CHANGELOG entry?'%(new), True, args.yes ):
        raise Abort( 'cancelled' )

    with open( VERSION_FILE, 'w', encoding='utf-8', newline='\n' ) as f:
        f.write( new + '\n' )
    with open( CHANGELOG, 'w', encoding='utf-8', newline='\n' ) as f:
        f.write( new_changelog )
    log.info( 'done: wrote VERSION and CHANGELOG.md' )

    if not args.yes and sys.stdin.isatty():
        log.debug( 'AWAITING INPUT: changelog review' )
        input( 'Review or edit CHANGELOG.md now, then press Enter to continue (Ctrl-C to stop)... ' )
        with open( CHANGELOG, encoding='utf-8' ) as f:
            _, notes = date_changelog( f.read(), new, today, subjects )

    # --- tests
    if ask( 'Run the unit tests before committing?', True, args.yes ):
        log.info( '=== starting: unit tests ===' )
        code = run( [ sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.', '-p', 'test_*.py' ],
                    check=False, capture=False )
        if code != 0:
            raise Abort( 'unit tests failed; VERSION and CHANGELOG.md are modified but nothing is committed' )
        log.info( '=== finished: unit tests ===' )

    # --- commit and tag
    if not ask( 'Commit and create tag v%s?'%(new), True, args.yes ):
        raise Abort( 'stopped before committing; VERSION and CHANGELOG.md are modified' )
    run( [ 'git', 'add', 'VERSION', 'CHANGELOG.md' ] )
    if run( [ 'git', 'diff', '--cached', '--name-only' ] ):
        run( [ 'git', 'commit', '-m', 'Release v%s'%(new) ] )
    else:
        log.info( 'VERSION and CHANGELOG.md already match; tagging the current commit' )
    run( [ 'git', 'tag', '-a', 'v%s'%(new), '-m', 'v%s'%(new) ] )
    log.info( 'done: committed and tagged v%s'%(new) )

    # --- publish
    if not ask( 'Push the commit and tag to origin?', False, args.yes ):
        log.info( 'not pushed. Later: git push origin %s --follow-tags'%(branch) )
        return 0
    run( [ 'git', 'push', 'origin', branch, '--follow-tags' ] )
    log.info( 'done: pushed %s and v%s'%( branch, new ) )

    gh = shutil.which( 'gh' )
    if gh and ask( 'Create the GitHub release v%s with these notes?'%(new), True, args.yes ):
        run( [ gh, 'release', 'create', 'v%s'%(new), '--title', 'v%s'%(new), '--notes', notes ] )
        log.info( 'done: GitHub release v%s created'%(new) )
    elif not gh:
        log.info( 'GitHub CLI not found; create the release on GitHub from tag v%s'%(new) )
    return 0


if __name__ == '__main__':
    sys.exit( main() )
