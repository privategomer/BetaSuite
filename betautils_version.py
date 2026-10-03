"""
betautils_version.py - BetaSuite's version string.

The release version lives in the VERSION file at the repository root,
and tools/release/release.py is the only thing that changes it. At
runtime, a git checkout that isn't sitting exactly on that release's tag
gets semver build metadata appended, so a stats row or bug report says
precisely which code produced it:

    1.1.0                     a clean checkout of tag v1.1.0, or a zip download
    1.1.0+3.g1a2b3c4          3 commits past the last tag
    1.1.0+0.g1a2b3c4.dirty    on the tag, with uncommitted changes

Build metadata never changes semver ordering, so 1.1.0+3.g1a2b3c4 still
compares equal to 1.1.0. Standard library only: the app itself doesn't
depend on the semver package.
"""

import functools
import os
import re
import subprocess

REPO_DIR = os.path.dirname( os.path.abspath( __file__ ) )
VERSION_FILE = os.path.join( REPO_DIR, 'VERSION' )

# semver 2.0.0 core version with optional pre-release; build metadata is added here.
SEMVER_RE = re.compile(
    r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
    r'(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?$' )


def read_version_file( path=VERSION_FILE ):
    """The release version from VERSION, or '0.0.0' if it is missing or malformed."""
    try:
        with open( path, encoding='utf-8' ) as f:
            value = f.read().strip()
    except OSError:
        return '0.0.0'
    return value if SEMVER_RE.match( value ) else '0.0.0'


def _git( repo_dir, *args ):
    result = subprocess.run( [ 'git', '-C', repo_dir ] + list( args ), capture_output=True,
                             text=True, timeout=5 )
    if result.returncode != 0:
        raise RuntimeError( result.stderr.strip() )
    return result.stdout.strip()


def describe( repo_dir=REPO_DIR, base=None ):
    """
    The full version for the code in repo_dir.

    Returns the plain VERSION value when git is unavailable, repo_dir isn't
    a checkout, or the checkout is exactly on tag v<VERSION> with no
    local changes. Otherwise appends +<commits since tag>.g<sha>[.dirty].
    """
    base = base or read_version_file( os.path.join( repo_dir, 'VERSION' ) )
    if not os.path.exists( os.path.join( repo_dir, '.git' ) ):
        return base
    try:
        sha = _git( repo_dir, 'rev-parse', '--short', 'HEAD' )
        dirty = bool( _git( repo_dir, 'status', '--porcelain', '--untracked-files=no' ) )
        try:
            tag_desc = _git( repo_dir, 'describe', '--tags', '--long', '--match', 'v[0-9]*' )
            match = re.match( r'^(v.+)-(\d+)-g[0-9a-f]+$', tag_desc )
            tag, distance = match.group( 1 ), int( match.group( 2 ) )
        except Exception:
            tag, distance = None, int( _git( repo_dir, 'rev-list', '--count', 'HEAD' ) )
    except Exception:
        return base

    if tag == 'v' + base and distance == 0 and not dirty:
        return base
    return '%s+%d.g%s%s'%( base, distance, sha, '.dirty' if dirty else '' )


@functools.lru_cache( maxsize=1 )
def get_version():
    """This checkout's version, computed once per process."""
    return describe()
