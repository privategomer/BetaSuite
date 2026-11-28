"""
release.py - cut a BetaSuite release through a pull request: bump VERSION,
date the CHANGELOG entry, open a release PR, wait for its checks, merge it,
then tag the merged commit on main and publish it on GitHub.

    python3 tools/release/release.py              # suggest a bump from commits, ask
    python3 tools/release/release.py --bump minor
    python3 tools/release/release.py --dry-run    # show what would happen, change nothing

Run it from an up-to-date, clean main. main is protected (changes only
through a PR with passing checks), so the release commit travels like any
other change:

    1. branch release/vX.Y.Z from main, write VERSION and the dated CHANGELOG
       entry, run the unit tests, commit, push, open the PR
    2. wait for the PR's checks to pass, then squash-merge it (an admin
       bypass of the approval rule; the checks rule has no bypass)
    3. pull main, tag the merged commit vX.Y.Z, push the tag, create the
       GitHub release

The tag is made after the merge, on the commit that is really on main, so a
squash merge never leaves it pointing at a commit main doesn't contain.
Re-running is safe: the stage is worked out from VERSION, the tags, the
CHANGELOG and any open release PR, and a run that stopped resumes there.

The bump is suggested from commit subjects since the last v* tag (with
squash merges, the PR titles):

    major   "BREAKING" anywhere, or a conventional "type!:" subject
    minor   a subject starting with feat/add/new (or "feat(...):")
    patch   anything else
    none    VERSION isn't tagged yet: release it as-is

Release notes come from a "## Unreleased" section in CHANGELOG.md if you
keep one; otherwise one is drafted from the commit subjects and you get a
chance to edit it before anything is committed. Every step that changes
the repository or anything remote asks first. All output goes to
tools/release/release.log at every level; --log-level filters the console.

Needs the semver package (requirements-dev.txt; offers to install it) and
the GitHub CLI (gh), logged in.
"""

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', '..' ) )
VERSION_FILE = os.path.join( REPO_ROOT, 'VERSION' )
CHANGELOG = os.path.join( REPO_ROOT, 'CHANGELOG.md' )

# Checks that must be reported and passing before the release PR is merged.
# The repository's ruleset enforces the same names; waiting here just avoids
# a rejected merge.
REQUIRED_CHECKS = ( 'tests', )
CHECKS_TIMEOUT_MINUTES = 45
CHECKS_POLL_SECONDS = 20

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
    heading = '## %s (%s)'%( version, today ) if today else None
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
        if heading:
            lines[start] = heading + '\n'
        notes = ''.join( lines[start + 1:end] ).strip()
        return ''.join( lines ), notes

    notes = '\n'.join( '- %s'%(s) for s in subjects ) or '- Maintenance release'
    block = '%s\n\n%s\n\n'%( heading, notes )
    first = next( ( i for i, line in enumerate( lines ) if re.match( r'^## ', line ) ), len( lines ) )
    return ''.join( lines[:first] ) + block + ''.join( lines[first:] ), notes


def release_branch( version ):
    return 'release/v%s'%(version)


def release_is_prepared( changelog, version ):
    """
    True if the CHANGELOG has a dated "## <version> (YYYY-MM-DD)" heading.

    That heading only exists once a release PR for this version has landed
    on main (an unreleased section is headed "## Unreleased").
    """
    return re.search( r'^## %s \(\d{4}-\d{2}-\d{2}\)[ \t]*$'%( re.escape( version ) ),
                      changelog, re.MULTILINE ) is not None


def choose_stage( version_is_tagged, head_is_tag, prepared ):
    """
    'publish': VERSION is tagged on HEAD; make sure it is on origin and on GitHub
    'tag':     a release PR for VERSION has merged but nothing is tagged yet
    'new':     start a release: bump, branch, PR
    """
    if version_is_tagged and head_is_tag:
        return 'publish'
    if not version_is_tagged and prepared:
        return 'tag'
    return 'new'


def sync_problem( ahead, behind ):
    """Why local main can't be released from, or None if it equals origin/main."""
    if ahead and behind:
        return 'main has diverged from origin/main (%d ahead, %d behind)'%( ahead, behind )
    if ahead:
        return ( 'main is %d commit(s) ahead of origin/main. main only takes changes through '
                 'a pull request, so those commits must go through one first'%(ahead) )
    if behind:
        return 'main is %d commit(s) behind origin/main: git pull --ff-only'%(behind)
    return None


def checks_state( checks, required=() ):
    """
    'pass', 'pending' or 'fail' for rows of {'name', 'bucket'} (see checks_from_rollup).

    No rows, or a required check that has not reported yet, is 'pending'.
    Any failed or cancelled check is 'fail', required or not.
    """
    if not checks:
        return 'pending'
    if any( c.get( 'bucket' ) in ( 'fail', 'cancel' ) for c in checks ):
        return 'fail'
    names = { c.get( 'name' ) for c in checks }
    if any( r not in names for r in required ):
        return 'pending'
    if any( c.get( 'bucket' ) == 'pending' for c in checks ):
        return 'pending'
    return 'pass'


def checks_from_rollup( text ):
    """
    Rows of {'name', 'bucket'} from `gh pr view --json statusCheckRollup`, or
    None if the text is not that JSON (an error, or nothing printed).

    This is read from `pr view` rather than `pr checks --json` because older
    GitHub CLI releases have no --json on `pr checks`. bucket is pass,
    pending, fail, cancel or skipping, as `gh pr checks` words it.
    """
    try:
        data = json.loads( text )
    except ( ValueError, TypeError ):
        return None
    if not isinstance( data, dict ) or not isinstance( data.get( 'statusCheckRollup' ), list ):
        return None
    rows = []
    for c in data['statusCheckRollup']:
        if c.get( '__typename' ) == 'StatusContext':
            name = c.get( 'context' )
            state = ( c.get( 'state' ) or '' ).upper()
            bucket = 'pass' if state == 'SUCCESS' else 'pending' if state in ( 'PENDING', 'EXPECTED' ) else 'fail'
        else:
            name = c.get( 'name' )
            if ( c.get( 'status' ) or '' ).upper() != 'COMPLETED':
                bucket = 'pending'
            else:
                conclusion = ( c.get( 'conclusion' ) or '' ).upper()
                bucket = ( 'pass' if conclusion in ( 'SUCCESS', 'NEUTRAL' ) else
                           'skipping' if conclusion == 'SKIPPED' else
                           'cancel' if conclusion == 'CANCELLED' else 'fail' )
        rows.append( { 'name': name, 'bucket': bucket } )
    return rows


# ---------------------------------------------------------------- release

def main():
    relaunch_in_venv()
    parser = argparse.ArgumentParser( description=__doc__.split( '\n\n' )[0] )
    parser.add_argument( '--bump', choices=[ 'major', 'minor', 'patch', 'none' ],
                         help='skip the suggestion and use this bump' )
    parser.add_argument( '--dry-run', action='store_true', help='show the plan, change nothing' )
    parser.add_argument( '--yes', action='store_true', help='take the default answer to every question' )
    parser.add_argument( '--checks-timeout', type=int, default=CHECKS_TIMEOUT_MINUTES, metavar='MINUTES',
                         help='how long to wait for the PR checks (default %(default)s)' )
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


def read_text( path ):
    with open( path, encoding='utf-8' ) as f:
        return f.read()


def release( args ):
    semver = import_semver( args.yes )

    # --- preflight
    if not os.path.isdir( os.path.join( REPO_ROOT, '.git' ) ):
        raise Abort( 'not a git checkout: %s'%(REPO_ROOT) )
    gh = shutil.which( 'gh' )
    if not gh:
        raise Abort( 'the GitHub CLI (gh) is required: https://cli.github.com, then gh auth login' )
    if run( [ gh, 'auth', 'status' ], check=False, capture=False ) != 0:
        raise Abort( 'gh is not logged in: gh auth login' )
    branch = run( [ 'git', 'rev-parse', '--abbrev-ref', 'HEAD' ] )
    if branch != 'main':
        raise Abort( 'you are on branch %r. Releases are cut from main (the release PR branches '
                     'from it): git switch main'%(branch) )
    dirty = run( [ 'git', 'status', '--porcelain', '--untracked-files=no' ] )
    if dirty:
        raise Abort( 'uncommitted changes; commit or stash them first:\n%s'%(dirty) )
    run( [ 'git', 'fetch', 'origin', '--tags' ] )
    ahead = int( run( [ 'git', 'rev-list', '--count', 'origin/main..HEAD' ] ) )
    behind = int( run( [ 'git', 'rev-list', '--count', 'HEAD..origin/main' ] ) )
    problem = sync_problem( ahead, behind )
    if problem:
        raise Abort( problem )

    current = read_text( VERSION_FILE ).strip()
    semver.Version.parse( current )   # raises on a malformed VERSION
    changelog = read_text( CHANGELOG )

    tags = set( run( [ 'git', 'tag', '--list', 'v*' ] ).split() )
    version_is_tagged = ( 'v' + current ) in tags
    head = run( [ 'git', 'rev-parse', 'HEAD' ] )
    head_is_tag = version_is_tagged and run( [ 'git', 'rev-list', '-n', '1', 'v' + current ] ) == head
    prepared = release_is_prepared( changelog, current )
    stage = choose_stage( version_is_tagged, head_is_tag, prepared )
    log.info( 'VERSION is %s (%s); stage: %s'%(
        current, 'tagged' if version_is_tagged else 'not tagged yet', stage ) )

    if stage == 'publish':
        log.info( 'v%s is already tagged on this commit; checking whether it still needs publishing'%(current) )
        _, notes = date_changelog( changelog, current, None, [] )
        return publish( args, gh, current, notes )
    if stage == 'tag':
        log.info( 'the release PR for %s has merged; tagging it'%(current) )
        _, notes = date_changelog( changelog, current, None, [] )
        return tag_and_publish( args, gh, current, notes )
    return new_release( args, gh, semver, current, tags, version_is_tagged, changelog )


def new_release( args, gh, semver, current, tags, version_is_tagged, changelog ):
    last_tag = run( [ 'git', 'describe', '--tags', '--abbrev=0', '--match', 'v[0-9]*' ], check=False ) or None
    if last_tag and not last_tag.startswith( 'v' ):
        last_tag = None
    log_range = '%s..HEAD'%(last_tag) if last_tag else 'HEAD'
    subjects = [ s for s in run( [ 'git', 'log', '--format=%s', '--no-merges', log_range ] ).splitlines() if s ]
    log.info( 'last tag %s; %d commit(s) since'%( last_tag or '(none)', len( subjects ) ) )
    for s in subjects:
        log.info( '  - %s'%(s) )

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
    rel = release_branch( new )
    log.info( 'releasing %s -> %s (%s) through %s'%( current, new, bump, rel ) )

    # --- a release PR that is already open (an earlier run stopped) resumes
    open_pr = find_open_pr( gh, rel )
    if open_pr is None:
        if run( [ 'git', 'ls-remote', '--heads', 'origin', rel ], check=False ):
            raise Abort( 'origin already has branch %s but it has no open PR. Delete it '
                         '(git push origin --delete %s) and re-run'%( rel, rel ) )
        if run( [ 'git', 'branch', '--list', rel ], check=False ):
            raise Abort( 'a local branch %s already exists. Delete it (git branch -D %s) and re-run'%( rel, rel ) )

    today = datetime.date.today().isoformat()
    new_changelog, notes = date_changelog( changelog, new, today, subjects )
    log.info( 'release notes:\n%s'%(notes) )

    if args.dry_run:
        if open_pr is not None:
            log.info( 'dry run: PR #%d for %s is open; would wait for its checks, squash-merge it, '
                      'tag v%s and publish'%( open_pr, rel, new ) )
        else:
            log.info( 'dry run: would branch %s, write VERSION=%s and the CHANGELOG entry, run the tests, '
                      'commit, push, open a PR, wait for checks, squash-merge, tag v%s and publish'%(
                          rel, new, new ) )
        return 0

    if open_pr is None:
        if not ask( 'Create branch %s with VERSION=%s and the CHANGELOG entry?'%( rel, new ), True, args.yes ):
            raise Abort( 'cancelled' )
        run( [ 'git', 'switch', '-c', rel ] )
        with open( VERSION_FILE, 'w', encoding='utf-8', newline='\n' ) as f:
            f.write( new + '\n' )
        with open( CHANGELOG, 'w', encoding='utf-8', newline='\n' ) as f:
            f.write( new_changelog )
        log.info( 'done: wrote VERSION and CHANGELOG.md on %s'%(rel) )
        undo = 'git restore VERSION CHANGELOG.md && git switch main && git branch -D %s'%(rel)

        if not args.yes and sys.stdin.isatty():
            log.debug( 'AWAITING INPUT: changelog review' )
            input( 'Review or edit CHANGELOG.md now, then press Enter to continue (Ctrl-C to stop)... ' )
            _, notes = date_changelog( read_text( CHANGELOG ), new, today, subjects )

        # --- tests
        if ask( 'Run the unit tests before committing?', True, args.yes ):
            log.info( '=== starting: unit tests ===' )
            code = run( [ sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.', '-p', 'test_*.py' ],
                        check=False, capture=False )
            if code != 0:
                raise Abort( 'unit tests failed; nothing is committed. To discard the release attempt: %s'%(undo) )
            log.info( '=== finished: unit tests ===' )

        # --- commit, push, open the PR
        if not ask( 'Commit, push %s and open the release PR?'%(rel), True, args.yes ):
            raise Abort( 'stopped before committing; VERSION and CHANGELOG.md are modified on %s. '
                         'To discard: %s'%( rel, undo ) )
        run( [ 'git', 'add', 'VERSION', 'CHANGELOG.md' ] )
        run( [ 'git', 'commit', '-m', 'Release v%s'%(new) ] )
        run( [ 'git', 'push', '-u', 'origin', rel ] )
        out = run( [ gh, 'pr', 'create', '--base', 'main', '--head', rel,
                     '--title', 'Release v%s'%(new), '--body', notes ] )
        found = re.search( r'/pull/(\d+)', out )
        if not found:
            raise Abort( 'could not read the new PR number from: %s'%(out) )
        open_pr = int( found.group( 1 ) )
        log.info( 'done: opened PR #%d'%(open_pr) )

    # --- checks, then merge
    wait_for_checks( gh, open_pr, args.checks_timeout )
    if not ask( 'Checks passed. Squash-merge PR #%d now? (no approving review, so this uses the admin bypass)'%(
            open_pr ), True, args.yes ):
        log.info( 'not merged. Re-run this script to resume at the merge' )
        return 0
    run( [ 'git', 'switch', 'main' ] )
    run( [ gh, 'pr', 'merge', str( open_pr ), '--squash', '--admin', '--delete-branch' ] )
    state = run( [ gh, 'pr', 'view', str( open_pr ), '--json', 'state', '--jq', '.state' ] )
    if state != 'MERGED':
        raise Abort( 'PR #%d is %s, not MERGED. Look at it on GitHub, then re-run to resume'%( open_pr, state ) )
    log.info( 'done: merged PR #%d'%(open_pr) )

    # --- main now has the release commit: tag it
    run( [ 'git', 'pull', '--ff-only', 'origin', 'main' ] )
    if read_text( VERSION_FILE ).strip() != new or not release_is_prepared( read_text( CHANGELOG ), new ):
        raise Abort( 'main does not show the release after the merge (VERSION or the CHANGELOG heading '
                     'for %s is missing). Check main, then re-run'%(new) )
    return tag_and_publish( args, gh, new, notes )


def find_open_pr( gh, rel ):
    """Number of the open PR from branch rel, or None."""
    out = run( [ gh, 'pr', 'list', '--head', rel, '--state', 'open', '--json', 'number' ], check=False )
    try:
        rows = json.loads( out ) if out else []
    except ValueError:
        rows = []
    return int( rows[0]['number'] ) if rows else None


def wait_for_checks( gh, number, timeout_minutes, sleep=None, clock=None ):
    """Block until the PR's checks pass; Abort on a failure or when time runs out."""
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    deadline = clock() + timeout_minutes * 60
    last = None
    unreadable = 0
    while True:
        try:
            rows = checks_from_rollup( run( [ gh, 'pr', 'view', str( number ), '--json', 'statusCheckRollup' ] ) )
            problem = None if rows is not None else 'gh printed something other than the checks JSON'
        except Abort as e:
            rows, problem = None, str( e )
        if rows is None:
            unreadable += 1
            if unreadable >= 3:
                raise Abort( 'could not read the checks on PR #%d, three polls in a row: %s'%( number, problem ) )
            log.warning( 'could not read the checks on PR #%d (%s); trying again'%( number, problem ) )
            sleep( CHECKS_POLL_SECONDS )
            continue
        unreadable = 0
        state = checks_state( rows, REQUIRED_CHECKS )
        if state != last:
            log.info( 'PR #%d checks: %s'%( number, state ) )
            last = state
        if state == 'pass':
            return
        if state == 'fail':
            bad = ', '.join( sorted( c.get( 'name', '?' ) for c in rows if c.get( 'bucket' ) in ( 'fail', 'cancel' ) ) )
            raise Abort( 'checks failed on PR #%d (%s). Fix it on the release branch or close the PR, '
                         'then re-run'%( number, bad ) )
        if clock() >= deadline:
            raise Abort( 'checks on PR #%d did not finish in %d minutes. The PR stays open; re-run '
                         'to resume'%( number, timeout_minutes ) )
        sleep( CHECKS_POLL_SECONDS )


def tag_and_publish( args, gh, version, notes ):
    """Tag the commit on main that carries the release, then publish."""
    tag = 'v%s'%(version)
    if args.dry_run:
        log.info( 'dry run: would tag %s on main, push the tag and create the GitHub release'%(tag) )
        return 0
    if not ask( 'Create tag %s on main?'%(tag), True, args.yes ):
        log.info( 'not tagged. Re-run this script later to tag and publish' )
        return 0
    run( [ 'git', 'tag', '-a', tag, '-m', tag ] )
    log.info( 'done: tagged %s'%(tag) )
    return publish( args, gh, version, notes )


def publish( args, gh, version, notes ):
    """Push the tag, then create the GitHub release. Safe to re-run."""
    tag = 'v%s'%(version)
    if args.dry_run:
        log.info( 'dry run: would push %s if origin lacks it and create the GitHub release if missing'%(tag) )
        return 0
    remote_tag = run( [ 'git', 'ls-remote', '--tags', 'origin', 'refs/tags/%s'%(tag) ], check=False )
    if remote_tag:
        log.info( '%s is already on origin'%(tag) )
    elif ask( 'Push tag %s to origin?'%(tag), True, args.yes ):
        run( [ 'git', 'push', 'origin', 'refs/tags/%s'%(tag) ] )
        log.info( 'done: pushed %s'%(tag) )
    else:
        log.info( 'not pushed. Later: git push origin refs/tags/%s, then re-run this script'%(tag) )
        return 0

    if run( [ gh, 'release', 'view', tag ], check=False, capture=False ) == 0:
        log.info( 'GitHub release %s already exists; nothing left to do'%(tag) )
        return 0
    if not ask( 'Create the GitHub release %s with these notes?'%(tag), True, args.yes ):
        log.info( 'no GitHub release created. Re-run this script later to create it' )
        return 0

    notes_file = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'release-notes.md' )
    with open( notes_file, 'w', encoding='utf-8', newline='\n' ) as f:
        f.write( notes + '\n' )
    cmd = [ gh, 'release', 'create', tag, '--title', tag, '--notes-file', notes_file, '--verify-tag' ]
    for attempt in range( 1, 4 ):
        if run( cmd, check=False, capture=False ) == 0:
            log.info( 'done: GitHub release %s created'%(tag) )
            os.remove( notes_file )
            return 0
        # GitHub sometimes answers 5xx; a release created despite the error must not be duplicated.
        if run( [ gh, 'release', 'view', tag ], check=False, capture=False ) == 0:
            log.info( 'done: GitHub release %s exists (created despite the error)'%(tag) )
            os.remove( notes_file )
            return 0
        if attempt < 3:
            log.warning( 'gh release create failed (attempt %d of 3; see the log); retrying in %ds'%(
                attempt, 10 * attempt ) )
            time.sleep( 10 * attempt )
    raise Abort( 'could not create the GitHub release %s. The tag is pushed; re-run this script '
                 'later to try again (notes kept in %s)'%( tag, notes_file ) )

if __name__ == '__main__':
    sys.exit( main() )
