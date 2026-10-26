"""
test_setup_release_check.py - setup's "switch to the latest release" step.

setup.sh and setup.ps1 move the user's git checkout: they fetch tags,
pick the newest release, check it out and restart from that release's
own copy of the script. Nothing else in the suite runs either script;
on 2026-10-26 the suite still passed all 1480 tests with setup.sh made
unparseable, and with the release step never called.

Each test builds a throwaway repository: a bare origin, and a clone
whose release tags carry a stub setup script that records the
arguments and environment it was restarted with. The branch carries
the real script, cut off just before its platform checks so it never
installs anything. The tags include the traps the step has to get
right: v1.10.0 above v1.2.0 (version order, not text order), a
pre-release v1.11.0-rc1, and a 'wip' tag that is not a release at all.

PowerShell tests run under pwsh, else Windows PowerShell, and skip when
neither is installed.
"""

import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import unittest

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
SETUP_SH = os.path.join( REPO, 'setup.sh' )
SETUP_PS1 = os.path.join( REPO, 'setup.ps1' )
HARNESS_LIB = os.path.join( REPO, 'tools', 'tuning', 'harness_lib.sh' )

GIT = shutil.which( 'git' )
BASH = shutil.which( 'bash' ) if os.name != 'nt' else None
POWERSHELL = shutil.which( 'pwsh' ) or shutil.which( 'powershell' )
# util-linux `script` gives the bash tests a terminal to answer prompts on;
# BSD/macOS `script` takes different arguments.
TTY_SCRIPT = shutil.which( 'script' ) if platform.system() == 'Linux' else None

# The script under test runs up to, not including, these lines: everything
# after them checks the machine and installs things.
SH_CUT_BEFORE = 'OS="$(uname -s)"'
PS1_CUT_BEFORE = 'Write-Log info ("platform:'

SH_STUB = r'''#!/bin/bash
{ printf 'STUB'; for a in "$@"; do printf ' [%s]' "$a"; done
  printf ' checked=%s\n' "${BETASUITE_RELEASE_CHECKED:-unset}"; } >> "$HARNESS_MARK"
exit 7
'''
SH_TAIL = '\necho "PREFLIGHT checked=${BETASUITE_RELEASE_CHECKED:-unset}" >> "$HARNESS_MARK"\nexit 0\n'

PS1_STUB = r'''$line = 'STUB'
foreach ($a in $args) { $line += " [$a]" }
$line += " checked=$($env:BETASUITE_RELEASE_CHECKED)"
Add-Content -Path $env:HARNESS_MARK -Value $line
exit 7
'''
PS1_TAIL = ( "Add-Content -Path $env:HARNESS_MARK -Value \"PREFLIGHT checked=$($env:BETASUITE_RELEASE_CHECKED)\"\n"
             "exit 0\n" )

# The stub's exit status, passed back through the restart.
STUB_EXIT = 7


def _cut( path, marker, tail ):
    """The script at path up to the line starting with marker, plus tail.
    A missing marker fails loudly: the tests would otherwise run setup for real."""
    with open( path, newline='' ) as handle:
        text = handle.read().replace( '\r\n', '\n' )
    kept = []
    for line in text.splitlines( keepends=True ):
        if line.startswith( marker ):
            return ''.join( kept ) + tail
        kept.append( line )
    raise AssertionError( '%s no longer has a line starting %r; update the cut point in %s'%(
        path, marker, os.path.basename( __file__ ) ) )


def _git_env():
    # The user's own git config (signing, hooks, default branch) must not
    # reach the fixture repositories.
    env = dict( os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1',
                GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t',
                GIT_COMMITTER_EMAIL='t@t', GIT_TERMINAL_PROMPT='0' )
    env.pop( 'BETASUITE_RELEASE_CHECKED', None )
    return env


def _git( cwd, *args ):
    result = subprocess.run( [ GIT, *args ], cwd=cwd, env=_git_env(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True )
    if result.returncode != 0:
        raise AssertionError( 'git %s failed in %s: %s'%( ' '.join( args ), cwd, result.stdout ) )
    return result.stdout.strip()


def _build_template( root, script_name, stub, under_test, with_harness_lib ):
    """
    origin.git and work/ under root.
      c1  v1.2.0, wip        stub script, only_in_release.txt
      c2  v1.10.0            stub script            <- the latest release
      c3  v1.11.0-rc1        stub script            (pre-release)
      c4  main               script under test, only_in_release.txt removed
    """
    work = os.path.join( root, 'work' )
    os.makedirs( work )

    def write( name, text ):
        with open( os.path.join( work, name ), 'w', newline='\n' ) as handle:
            handle.write( text )

    _git( work, 'init', '-q' )
    _git( work, 'checkout', '-q', '-b', 'main' )
    if with_harness_lib:
        os.makedirs( os.path.join( work, 'tools', 'tuning' ) )
        shutil.copy( HARNESS_LIB, os.path.join( work, 'tools', 'tuning', 'harness_lib.sh' ) )
    write( 'VERSION', '9.9.9\n' )
    write( '.python-version', '3.12\n' )
    write( '.gitignore', 'setup.log\n' )
    write( 'tracked.txt', 'a\n' )
    write( script_name, stub )
    write( 'only_in_release.txt', 'r\n' )
    _git( work, 'add', '-A' )
    _git( work, 'commit', '-qm', 'c1' )
    _git( work, 'tag', '-a', 'v1.2.0', '-m', 'v1.2.0' )
    _git( work, 'tag', 'wip' )
    write( 'tracked.txt', 'b\n' )
    _git( work, 'commit', '-qam', 'c2' )
    _git( work, 'tag', '-a', 'v1.10.0', '-m', 'v1.10.0' )
    write( 'tracked.txt', 'c\n' )
    _git( work, 'commit', '-qam', 'c3' )
    _git( work, 'tag', '-a', 'v1.11.0-rc1', '-m', 'rc' )
    write( script_name, under_test )
    os.remove( os.path.join( work, 'only_in_release.txt' ) )
    _git( work, 'add', '-A' )
    _git( work, 'commit', '-qm', 'c4 under test' )
    _git( root, 'clone', '-q', '--bare', work, os.path.join( root, 'origin.git' ) )
    _git( work, 'remote', 'add', 'origin', os.path.join( root, 'origin.git' ) )
    _git( work, 'fetch', '-q', 'origin' )


class _ReleaseCheckCase( unittest.TestCase ):
    """One fresh copy of a class-wide template repository per test."""

    SCRIPT = None
    STUB = None

    @classmethod
    def under_test( cls ):
        raise NotImplementedError

    @classmethod
    def setUpClass( cls ):
        cls._template = tempfile.mkdtemp( prefix='betasuite-relchk-' )
        _build_template( cls._template, cls.SCRIPT, cls.STUB, cls.under_test(),
                         with_harness_lib=( cls.SCRIPT == 'setup.sh' ) )

    @classmethod
    def tearDownClass( cls ):
        shutil.rmtree( cls._template, ignore_errors=True )

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-relchk-' )
        self.addCleanup( shutil.rmtree, self.root, True )
        shutil.copytree( os.path.join( self._template, 'work' ), os.path.join( self.root, 'work' ), symlinks=True )
        shutil.copytree( os.path.join( self._template, 'origin.git' ), os.path.join( self.root, 'origin.git' ), symlinks=True )
        self.work = os.path.join( self.root, 'work' )
        self.origin = os.path.join( self.root, 'origin.git' )
        self.mark = os.path.join( self.root, 'mark.txt' )
        self.git( 'remote', 'set-url', 'origin', self.origin )
        self.release_sha = self.git( 'rev-parse', 'v1.10.0^{commit}' )
        self.main_sha = self.git( 'rev-parse', 'HEAD' )

    def git( self, *args ):
        return _git( self.work, *args )

    def write( self, name, text ):
        with open( os.path.join( self.work, name ), 'w', newline='\n' ) as handle:
            handle.write( text )

    def marks( self ):
        if not os.path.exists( self.mark ):
            return []
        with open( self.mark ) as handle:
            return handle.read().splitlines()

    def log( self ):
        if not os.path.exists( self.log_path ):
            return ''
        with open( self.log_path, encoding='utf-8-sig', errors='replace' ) as handle:
            return handle.read()

    def head( self ):
        return self.git( 'rev-parse', 'HEAD' )

    def env( self, extra=None ):
        env = _git_env()
        env[ 'HARNESS_MARK' ] = self.mark
        env.update( extra or {} )
        return env

    # -- outcomes shared by both scripts

    def assert_restarted_from( self, sha, stub_args ):
        self.assertEqual( self.head(), sha, self.output )
        self.assertEqual( self.marks(), [ 'STUB%s checked=1'%( ''.join( ' [%s]'%a for a in stub_args ) ) ],
                          self.output )
        self.assertEqual( self.returncode, STUB_EXIT, self.output )

    def assert_stayed( self ):
        self.assertEqual( self.head(), self.main_sha, self.output )
        self.assertEqual( self.marks(), [ 'PREFLIGHT checked=1' ], self.output )
        self.assertEqual( self.returncode, 0, self.output )


class _SharedScenarios:
    """Behaviour both scripts promise. Subclasses supply run(), the flag
    spellings and the scenarios only one platform can drive."""

    def test_switches_to_the_newest_release_by_version_not_text_skipping_prerelease_and_non_release_tags( self ):
        # Text order would pick v1.2.0, a plain pattern 'wip', and no
        # pre-release filter v1.11.0-rc1.
        self.run_setup( [ self.YES ] )
        self.assert_restarted_from( self.release_sha, [ self.YES ] )
        self.assertIn( 'Latest release is v1.10.0;', self.log() )
        # git's own complaints (describe on an untagged HEAD) stay out of
        # the user's terminal on a normal run.
        self.assertNotIn( 'fatal:', self.output )

    def test_already_on_the_latest_release_carries_on_without_asking( self ):
        self.git( 'tag', '-a', 'v1.20.0', '-m', 'x' )
        self.run_setup( [ self.YES ] )
        self.assert_stayed()
        self.assertIn( 'on the latest release, v1.20.0', self.log() )
        self.assertNotIn( 'Switch to', self.log() )

    def test_the_setup_log_records_version_and_checkout( self ):
        self.git( 'tag', '-a', 'v1.20.0', '-m', 'x' )
        self.run_setup( [ self.YES ] )
        self.assertIn( 'version: 9.9.9 (v1.20.0)', self.log() )

    def test_uncommitted_tracked_changes_keep_the_checkout( self ):
        # Checking out over them would carry or lose the user's edits.
        self.write( 'tracked.txt', 'edited\n' )
        self.run_setup( [ self.YES ] )
        self.assert_stayed()
        self.assertIn( 'main has uncommitted changes; staying on main', self.log() )

    def test_untracked_files_alone_do_not_block_the_switch( self ):
        self.write( 'notes.txt', 'mine\n' )
        self.run_setup( [ self.YES ] )
        self.assert_restarted_from( self.release_sha, [ self.YES ] )

    def test_the_flag_skips_the_step_entirely( self ):
        self.run_setup( [ self.YES, self.NO_RELEASE_CHECK ] )
        self.assert_stayed()
        self.assertNotIn( 'fetch release tags', self.log() )

    def test_the_restarted_copy_is_not_asked_again( self ):
        # The env var, not a flag, because an older release's script
        # rejects flags it doesn't know.
        self.run_setup( [ self.YES ], env_extra={ 'BETASUITE_RELEASE_CHECKED': '1' } )
        self.assert_stayed()
        self.assertNotIn( 'fetch release tags', self.log() )

    def test_no_release_tags_installs_the_checkout_as_it_is( self ):
        for tag in self.git( 'tag' ).split():
            self.git( 'tag', '-d', tag )
            _git( self.origin, 'tag', '-d', tag )
        self.run_setup( [ self.YES ] )
        self.assert_stayed()
        self.assertIn( 'no release tags found', self.log() )

    def test_a_source_download_without_git_metadata_installs_as_it_is( self ):
        os.rename( os.path.join( self.work, '.git' ), os.path.join( self.root, 'dotgit' ) )
        try:
            self.run_setup( [ self.YES ] )
            self.assertEqual( self.marks(), [ 'PREFLIGHT checked=1' ], self.output )
            # The whole message: the version line also says "not a git checkout".
            self.assertIn( 'not a git checkout; using the files as they are', self.log() )
            self.assertNotIn( 'fetch release tags', self.log() )
        finally:
            os.rename( os.path.join( self.root, 'dotgit' ), os.path.join( self.work, '.git' ) )
        self.assertEqual( self.head(), self.main_sha )

    def test_a_release_published_since_the_clone_is_fetched_and_chosen( self ):
        other = os.path.join( self.root, 'other' )
        _git( self.root, 'clone', '-q', self.origin, other )
        _git( other, '-c', 'advice.detachedHead=false', 'checkout', '-q', 'v1.10.0' )
        _git( other, 'commit', '-q', '--allow-empty', '-m', 'r30' )
        _git( other, 'tag', '-a', 'v1.30.0', '-m', 'remote only' )
        _git( other, 'push', '-q', 'origin', 'v1.30.0' )
        self.run_setup( [ self.YES ] )
        self.assert_restarted_from( _git( other, 'rev-parse', 'HEAD' ), [ self.YES ] )

    def test_an_unreachable_remote_falls_back_to_local_tags( self ):
        self.git( 'remote', 'set-url', 'origin', os.path.join( self.root, 'missing.git' ) )
        self.run_setup( [ self.YES ] )
        self.assert_restarted_from( self.release_sha, [ self.YES ] )
        self.assertIn( 'could not fetch tags', self.log() )

    def test_a_detached_older_release_is_named_by_its_tag( self ):
        self.git( 'tag', '-a', 'v1.5.0', '-m', 'x' )
        self.git( '-c', 'advice.detachedHead=false', 'checkout', '-q', 'v1.5.0' )
        self.run_setup( [ self.YES ] )
        self.assert_restarted_from( self.release_sha, [ self.YES ] )
        self.assertIn( 'this checkout is on v1.5.0. Switch', self.log() )

    def test_a_detached_untagged_commit_is_named_by_its_hash( self ):
        self.git( 'checkout', '-q', '--detach' )
        short = self.git( 'rev-parse', '--short', 'HEAD' )
        self.run_setup( [ self.YES ] )
        self.assert_restarted_from( self.release_sha, [ self.YES ] )
        self.assertIn( 'this checkout is on commit %s. Switch'%( short, ), self.log() )

    def test_commits_past_the_release_are_reported( self ):
        self.run_setup( [ self.YES ] )
        self.assertIn( 'main is 2 commit(s) past v1.10.0', self.log() )

    def test_a_branch_that_never_contained_the_release_is_not_called_past_it( self ):
        self.git( 'checkout', '-q', '-b', 'side', 'v1.2.0' )
        self.git( 'checkout', '-q', 'main', '--', self.SCRIPT )
        self.git( 'rm', '-q', 'only_in_release.txt' )
        self.git( 'commit', '-qm', 'side' )
        self.run_setup( [ self.YES ] )
        self.assert_restarted_from( self.release_sha, [ self.YES ] )
        self.assertIn( 'this checkout is on side. Switch', self.log() )
        self.assertNotIn( 'past v1.10.0', self.log() )

    def test_a_failed_checkout_stops_setup_where_it_is( self ):
        # An untracked file the release would overwrite makes git refuse.
        self.write( 'only_in_release.txt', 'mine\n' )
        self.run_setup( [ self.YES ] )
        self.assertEqual( self.returncode, 1, self.output )
        self.assertEqual( self.marks(), [], self.output )
        self.assertEqual( self.head(), self.main_sha )
        self.assertIn( 'could not check out v1.10.0', self.log() )


@unittest.skipUnless( GIT and BASH, 'needs git and bash' )
class TestSetupShReleaseCheck( _SharedScenarios, _ReleaseCheckCase ):

    SCRIPT = 'setup.sh'
    STUB = SH_STUB
    YES = '--yes'
    NO_RELEASE_CHECK = '--no-release-check'

    @classmethod
    def under_test( cls ):
        return _cut( SETUP_SH, SH_CUT_BEFORE, SH_TAIL )

    def setUp( self ):
        super().setUp()
        self.log_path = os.path.join( self.root, 'setup.log' )

    def run_setup( self, args, env_extra=None, tty_input=None ):
        env = self.env( dict( HARNESS_LOG_FILE=self.log_path, LOG_LEVEL='error', **( env_extra or {} ) ) )
        command = [ BASH, os.path.join( self.work, 'setup.sh' ), *args ]
        if tty_input is None:
            result = subprocess.run( command, cwd=self.work, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, timeout=60 )
        else:
            result = subprocess.run( [ TTY_SCRIPT, '-qec', ' '.join( shlex.quote( c ) for c in command ), os.devnull ],
                                     cwd=self.work, env=env, input=tty_input, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, universal_newlines=True, timeout=60 )
        self.returncode, self.output = result.returncode, result.stdout

    def test_the_whole_script_parses( self ):
        result = subprocess.run( [ BASH, '-n', SETUP_SH ], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True )
        self.assertEqual( result.returncode, 0, result.stdout )

    def test_the_original_arguments_reach_the_release_intact( self ):
        # An argument with a space must arrive as one argument.
        self.run_setup( [ '--yes', '--cpu', '--python', '3 12' ] )
        self.assert_restarted_from( self.release_sha, [ '--yes', '--cpu', '--python', '3 12' ] )
        self.assertNotIn( 'unbound variable', self.output )

    def test_no_arguments_and_no_terminal_takes_the_default_yes( self ):
        # set -u in harness_lib: an empty argument array must not be fatal.
        self.run_setup( [] )
        self.assert_restarted_from( self.release_sha, [] )
        self.assertNotIn( 'unbound variable', self.output )
        self.assertIn( "auto-answer 'y'", self.log() )

    def test_any_other_value_of_the_env_var_still_checks( self ):
        self.run_setup( [ '--yes' ], env_extra={ 'BETASUITE_RELEASE_CHECKED': '0' } )
        self.assert_restarted_from( self.release_sha, [ '--yes' ] )

    @unittest.skipUnless( TTY_SCRIPT, 'needs util-linux script for a terminal' )
    def test_answering_no_keeps_the_checkout( self ):
        self.run_setup( [], tty_input='n\n' )
        self.assert_stayed()
        self.assertIn( 'staying on main', self.log() )

    @unittest.skipUnless( TTY_SCRIPT, 'needs util-linux script for a terminal' )
    def test_pressing_enter_takes_the_default_yes( self ):
        self.run_setup( [], tty_input='\n' )
        self.assert_restarted_from( self.release_sha, [] )


@unittest.skipUnless( GIT and POWERSHELL, 'needs git and PowerShell (pwsh or powershell)' )
class TestSetupPs1ReleaseCheck( _SharedScenarios, _ReleaseCheckCase ):

    SCRIPT = 'setup.ps1'
    STUB = PS1_STUB
    YES = '-Yes'
    NO_RELEASE_CHECK = '-NoReleaseCheck'

    @classmethod
    def under_test( cls ):
        return _cut( SETUP_PS1, PS1_CUT_BEFORE, PS1_TAIL )

    def setUp( self ):
        super().setUp()
        self.log_path = os.path.join( self.work, 'setup.log' )

    def run_setup( self, args, env_extra=None, stdin_text=None, script=None ):
        command = [ POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    script or os.path.join( self.work, 'setup.ps1' ), *args ]
        result = subprocess.run( command, cwd=self.work, env=self.env( env_extra ), input=stdin_text,
                                 stdin=None if stdin_text is not None else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, timeout=120 )
        self.returncode, self.output = result.returncode, result.stdout

    def test_the_whole_script_parses( self ):
        check = ( "$e = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile("
                  "'%s', [ref]$null, [ref]$e); exit $e.Count"%( SETUP_PS1.replace( "'", "''" ), ) )
        result = subprocess.run( [ POWERSHELL, '-NoProfile', '-Command', check ], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True )
        self.assertEqual( result.returncode, 0, result.stdout )

    def test_the_original_arguments_reach_the_release_intact( self ):
        self.run_setup( [ '-Yes', '-Cpu', '-Python', '3 12' ] )
        self.assertEqual( self.head(), self.release_sha, self.output )
        stub = self.marks()[0] if self.marks() else ''
        for expected in ( '[-Yes]', '[-Cpu]', '[-Python] [3 12]' ):
            self.assertIn( expected, stub )
        self.assertTrue( stub.endswith( 'checked=1' ), stub )
        self.assertEqual( self.returncode, STUB_EXIT, self.output )

    def test_answering_no_keeps_the_checkout( self ):
        self.run_setup( [], stdin_text='n\n' )
        self.assert_stayed()
        self.assertIn( 'staying on main', self.log() )

    def test_pressing_enter_takes_the_default_yes( self ):
        self.run_setup( [], stdin_text='\n' )
        self.assert_restarted_from( self.release_sha, [] )

    def test_git_writing_to_stderr_does_not_end_setup( self ):
        # Windows PowerShell 5.1 turns a native command's redirected stderr
        # into errors, which the script's 'Stop' preference makes fatal;
        # `git describe --exact-match` writes to stderr whenever HEAD isn't
        # on a tag, so a fresh clone of main stopped in the release check.
        # PowerShell 7 only does the same with this preference set.
        strict = os.path.join( self.root, 'strict.ps1' )
        text = _cut( SETUP_PS1, PS1_CUT_BEFORE, PS1_TAIL )
        stop = "$ErrorActionPreference = 'Stop'\n"
        self.assertEqual( text.count( stop ), 1 )
        with open( strict, 'w', newline='\n' ) as handle:
            handle.write( text.replace( stop, stop + '$PSNativeCommandUseErrorActionPreference = $true\n' ) )
        shutil.copy( strict, os.path.join( self.work, 'setup.ps1' ) )
        self.git( 'commit', '-qam', 'strict native errors' )
        self.main_sha = self.head()
        self.run_setup( [ '-Yes' ] )
        self.assert_restarted_from( self.release_sha, [ '-Yes' ] )


if __name__ == '__main__':
    unittest.main()
