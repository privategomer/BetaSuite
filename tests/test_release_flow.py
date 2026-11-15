"""
test_release_flow.py - the pull-request release flow in tools/release/release.py.

The pure helpers are tested directly. The flow itself runs against a fake
git and gh: release.run is replaced by a scripted command handler over a
throwaway VERSION and CHANGELOG, so no repository, network or GitHub is
touched. Each scenario asserts which commands ran and in what order, and
that nothing mutating ran where it must not.
"""

import importlib.util
import json
import logging
import os
import shutil
import tempfile
import unittest

REPO_ROOT = os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..' ) )

try:
    import semver   # noqa: F401
except ImportError:
    semver = None


def _load_release():
    path = os.path.join( REPO_ROOT, 'tools', 'release', 'release.py' )
    spec = importlib.util.spec_from_file_location( 'release_flow_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class HelpersTest( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.r = _load_release()

    def test_release_branch_name( self ):
        self.assertEqual( self.r.release_branch( '1.5.1' ), 'release/v1.5.1' )

    def test_prepared_needs_a_dated_heading_for_exactly_this_version( self ):
        p = self.r.release_is_prepared
        self.assertTrue( p( '# C\n\n## 1.5.1 (2000-01-03)\n\n- x\n', '1.5.1' ) )
        self.assertTrue( p( '## 1.5.1 (2000-01-03)  \n', '1.5.1' ) )
        self.assertFalse( p( '## Unreleased\n\n- x\n', '1.5.1' ) )
        self.assertFalse( p( '## 1.5.10 (2000-01-03)\n', '1.5.1' ) )
        self.assertFalse( p( '## 1.5.0 (2000-01-03)\n', '1.5.1' ) )
        self.assertFalse( p( '## 1.5.1\n\n- x\n', '1.5.1' ) )
        self.assertFalse( p( 'text ## 1.5.1 (2000-01-03)\n', '1.5.1' ) )
        self.assertFalse( p( '## 1.5.1 (soon)\n', '1.5.1' ) )

    def test_stage_choice( self ):
        c = self.r.choose_stage
        self.assertEqual( c( True, True, True ), 'publish' )
        self.assertEqual( c( True, True, False ), 'publish' )
        self.assertEqual( c( False, False, True ), 'tag' )
        self.assertEqual( c( False, False, False ), 'new' )
        self.assertEqual( c( True, False, True ), 'new' )
        self.assertEqual( c( True, False, False ), 'new' )

    def test_sync_problem_names_the_direction( self ):
        s = self.r.sync_problem
        self.assertIsNone( s( 0, 0 ) )
        self.assertIn( 'ahead', s( 2, 0 ) )
        self.assertIn( 'pull request', s( 2, 0 ) )
        self.assertIn( 'behind', s( 0, 3 ) )
        self.assertIn( 'diverged', s( 1, 1 ) )

    def test_checks_state( self ):
        c = lambda rows, req=( 'tests', ): self.r.checks_state( rows, req )
        ok = { 'name': 'tests', 'bucket': 'pass' }
        self.assertEqual( c( [] ), 'pending' )
        self.assertEqual( c( [ ok ] ), 'pass' )
        self.assertEqual( c( [ ok, { 'name': 'CodeQL', 'bucket': 'skipping' } ] ), 'pass' )
        self.assertEqual( c( [ ok, { 'name': 'CodeQL', 'bucket': 'pending' } ] ), 'pending' )
        self.assertEqual( c( [ { 'name': 'tests', 'bucket': 'pending' } ] ), 'pending' )
        self.assertEqual( c( [ { 'name': 'CodeQL', 'bucket': 'pass' } ] ), 'pending' )   # required one not reported
        self.assertEqual( c( [ { 'name': 'tests', 'bucket': 'fail' } ] ), 'fail' )
        self.assertEqual( c( [ ok, { 'name': 'CodeQL', 'bucket': 'cancel' } ] ), 'fail' )
        self.assertEqual( c( [ { 'name': 'x', 'bucket': 'fail' } ], () ), 'fail' )
        self.assertEqual( c( [ { 'name': 'x', 'bucket': 'pass' } ], () ), 'pass' )

    def test_rollup_rows_are_read_from_gh_pr_view_json( self ):
        f = self.r.checks_from_rollup
        run = lambda **kw: dict( { '__typename': 'CheckRun', 'name': 'tests', 'status': 'COMPLETED', 'conclusion': 'SUCCESS' }, **kw )
        def rows( *entries ):
            return f( json.dumps( { 'statusCheckRollup': list( entries ) } ) )
        self.assertEqual( rows( run() ), [ { 'name': 'tests', 'bucket': 'pass' } ] )
        self.assertEqual( rows( run( conclusion='NEUTRAL' ) )[0]['bucket'], 'pass' )
        self.assertEqual( rows( run( conclusion='SKIPPED' ) )[0]['bucket'], 'skipping' )
        self.assertEqual( rows( run( conclusion='FAILURE' ) )[0]['bucket'], 'fail' )
        self.assertEqual( rows( run( conclusion='TIMED_OUT' ) )[0]['bucket'], 'fail' )
        self.assertEqual( rows( run( conclusion='CANCELLED' ) )[0]['bucket'], 'cancel' )
        self.assertEqual( rows( run( status='IN_PROGRESS', conclusion='' ) )[0]['bucket'], 'pending' )
        self.assertEqual( rows( run( status='QUEUED', conclusion=None ) )[0]['bucket'], 'pending' )
        ctx = lambda state: { '__typename': 'StatusContext', 'context': 'ci/other', 'state': state }
        self.assertEqual( rows( ctx( 'SUCCESS' ) ), [ { 'name': 'ci/other', 'bucket': 'pass' } ] )
        self.assertEqual( rows( ctx( 'PENDING' ) )[0]['bucket'], 'pending' )
        self.assertEqual( rows( ctx( 'EXPECTED' ) )[0]['bucket'], 'pending' )
        self.assertEqual( rows( ctx( 'FAILURE' ) )[0]['bucket'], 'fail' )
        self.assertEqual( rows( ctx( 'ERROR' ) )[0]['bucket'], 'fail' )
        self.assertEqual( rows(), [] )

    def test_rollup_text_that_is_not_the_json_is_none_not_empty( self ):
        f = self.r.checks_from_rollup
        self.assertIsNone( f( '' ) )
        self.assertIsNone( f( 'unknown flag: --json' ) )
        self.assertIsNone( f( '[]' ) )
        self.assertIsNone( f( '{"statusCheckRollup": null}' ) )
        self.assertIsNone( f( '{"other": []}' ) )


CHANGELOG_UNRELEASED = '# Changelog\n\n## Unreleased\n\n- a new thing\n\n## 1.4.0 (2000-01-02)\n\n- old\n'
CHANGELOG_PREPARED = '# Changelog\n\n## 1.5.0 (2000-01-03)\n\n- a new thing\n\n## 1.4.0 (2000-01-02)\n\n- old\n'


class Args:
    bump = None
    dry_run = False
    yes = True
    checks_timeout = 5


@unittest.skipIf( semver is None, 'semver not installed (requirements-dev.txt)' )
class FlowTest( unittest.TestCase ):
    """release() against a scripted git and gh."""

    @classmethod
    def setUpClass( cls ):
        cls.r = _load_release()

    def setUp( self ):
        r = self.r
        self.tmp = tempfile.mkdtemp()
        self.addCleanup( shutil.rmtree, self.tmp, ignore_errors=True )
        os.makedirs( os.path.join( self.tmp, '.git' ) )
        self.calls = []
        # scripted world
        self.branch = 'main'
        self.dirty = ''
        self.ahead, self.behind = 0, 0
        self.tags = { 'v1.4.0' }
        self.head = 'aaa'
        self.tag_commit = { 'v1.4.0': 'old' }
        self.subjects = [ 'add a new thing' ]
        self.open_prs = []
        self.remote_branches = set()
        self.local_branches = set()
        self.remote_tags = { 'v1.4.0' }
        self.releases = set()
        self.checks_script = [ [ { 'name': 'tests', 'bucket': 'pending' } ],
                               [ { 'name': 'tests', 'bucket': 'pass' } ] ]
        self.unit_tests_exit = 0
        self.merge_state = 'MERGED'
        self.gh_logged_in = True
        self.on_pull = None
        self.view_errors = 0
        self.polls = 0
        self.fail_polls = set()
        self.have_gh = True
        self.write( 'VERSION', '1.4.0\n' )
        self.write( 'CHANGELOG.md', CHANGELOG_UNRELEASED )

        self.saved = ( r.REPO_ROOT, r.VERSION_FILE, r.CHANGELOG, r.run, r.shutil.which, r.time.sleep )
        r.REPO_ROOT = self.tmp
        r.VERSION_FILE = os.path.join( self.tmp, 'VERSION' )
        r.CHANGELOG = os.path.join( self.tmp, 'CHANGELOG.md' )
        r.run = self.fake_run
        r.shutil.which = lambda name: ( '/fake/gh' if self.have_gh else None ) if name == 'gh' else self.saved[4]( name )
        r.time.sleep = lambda s: None
        self.addCleanup( self.restore )
        logging.getLogger( 'release' ).handlers[:] = []
        logging.getLogger( 'release' ).addHandler( logging.NullHandler() )

    def restore( self ):
        r = self.r
        r.REPO_ROOT, r.VERSION_FILE, r.CHANGELOG, r.run, r.shutil.which, r.time.sleep = self.saved

    def write( self, name, text ):
        with open( os.path.join( self.tmp, name ), 'w', encoding='utf-8' ) as f:
            f.write( text )

    def read( self, name ):
        with open( os.path.join( self.tmp, name ), encoding='utf-8' ) as f:
            return f.read()

    def fake_run( self, cmd, check=True, capture=True ):
        self.calls.append( list( cmd ) )
        c = [ str( x ) for x in cmd ]
        out, code = self.respond( c )
        if check and code != 0:
            raise self.r.Abort( '%s failed'%( ' '.join( c ) ) )
        return out if capture else code

    def respond( self, c ):
        if c[0] == '/fake/gh':
            return self.respond_gh( c )
        if c[:2] == [ self.r.sys.executable, '-m' ]:
            return '', self.unit_tests_exit
        g = c[1:]
        if g[:2] == [ 'rev-parse', '--abbrev-ref' ]:
            return self.branch, 0
        if g[:1] == [ 'status' ]:
            return self.dirty, 0
        if g[:1] == [ 'fetch' ]:
            return '', 0
        if g[:2] == [ 'rev-list', '--count' ]:
            return str( self.ahead if g[2] == 'origin/main..HEAD' else self.behind ), 0
        if g[:1] == [ 'tag' ] and g[1:2] == [ '--list' ]:
            return '\n'.join( sorted( self.tags ) ), 0
        if g[:1] == [ 'tag' ] and g[1:2] == [ '-a' ]:
            self.tags.add( g[2] )
            self.tag_commit[g[2]] = self.head
            return '', 0
        if g == [ 'rev-parse', 'HEAD' ]:
            return self.head, 0
        if g[:2] == [ 'rev-list', '-n' ]:
            return self.tag_commit.get( g[3], '' ), 0
        if g[:1] == [ 'describe' ]:
            return ( 'v1.4.0' if 'v1.4.0' in self.tags else '' ), 0
        if g[:1] == [ 'log' ]:
            return '\n'.join( self.subjects ), 0
        if g[:2] == [ 'ls-remote', '--heads' ]:
            return ( 'sha\trefs/heads/%s'%( g[3] ) if g[3] in self.remote_branches else '' ), 0
        if g[:2] == [ 'ls-remote', '--tags' ]:
            tag = g[3].replace( 'refs/tags/', '' )
            return ( 'sha\t%s'%( g[3] ) if tag in self.remote_tags else '' ), 0
        if g[:1] == [ 'branch' ] and g[1:2] == [ '--list' ]:
            return ( g[2] if g[2] in self.local_branches else '' ), 0
        if g[:2] == [ 'switch', '-c' ]:
            self.branch = g[2]
            self.local_branches.add( g[2] )
            return '', 0
        if g[:1] == [ 'switch' ]:
            self.branch = g[1]
            return '', 0
        if g[:1] == [ 'push' ] and g[1:2] == [ '-u' ]:
            self.remote_branches.add( g[3] )
            return '', 0
        if g[:1] == [ 'push' ] and g[-1].startswith( 'refs/tags/' ):
            self.remote_tags.add( g[-1].replace( 'refs/tags/', '' ) )
            return '', 0
        if g[:1] == [ 'pull' ]:
            if self.on_pull:
                self.on_pull()
            return '', 0
        if g[:1] in ( [ 'add' ], [ 'commit' ] ):
            return '', 0
        raise AssertionError( 'unscripted command: %s'%( ' '.join( c ) ) )

    def respond_gh( self, c ):
        g = c[1:]
        if g[:2] == [ 'auth', 'status' ]:
            return '', 0 if self.gh_logged_in else 1
        if g[:2] == [ 'pr', 'list' ]:
            return json.dumps( [ { 'number': n } for n in self.open_prs ] ), 0
        if g[:2] == [ 'pr', 'create' ]:
            self.open_prs.append( 7 )
            return 'https://github.com/x/y/pull/7', 0
        if g[:2] == [ 'pr', 'merge' ]:
            return '', 0
        if g[:2] == [ 'pr', 'view' ] and g[4] == 'statusCheckRollup':
            self.polls += 1
            if self.view_errors or ( self.polls - 1 ) in self.fail_polls:
                self.view_errors = max( 0, self.view_errors - 1 )
                raise self.r.Abort( 'gh pr view failed: HTTP 502' )
            rows = self.checks_script[0] if len( self.checks_script ) == 1 else self.checks_script.pop( 0 )
            return json.dumps( { 'statusCheckRollup': [ self.rollup_entry( r ) for r in rows ] } ), 0
        if g[:2] == [ 'pr', 'view' ]:
            return self.merge_state, 0
        if g[:2] == [ 'release', 'view' ]:
            return '', ( 0 if g[2] in self.releases else 1 )
        if g[:2] == [ 'release', 'create' ]:
            self.releases.add( g[2] )
            return '', 0
        raise AssertionError( 'unscripted gh command: %s'%( ' '.join( c ) ) )

    @staticmethod
    def rollup_entry( row ):
        b = row['bucket']
        entry = { '__typename': 'CheckRun', 'name': row['name'], 'status': 'COMPLETED', 'conclusion': 'SUCCESS' }
        if b == 'pending':
            entry.update( status='IN_PROGRESS', conclusion='' )
        elif b == 'fail':
            entry.update( conclusion='FAILURE' )
        elif b == 'cancel':
            entry.update( conclusion='CANCELLED' )
        elif b == 'skipping':
            entry.update( conclusion='SKIPPED' )
        return entry

    def names( self ):
        """Compact names of what ran, in order: 'git switch -c', 'gh pr merge', 'git push refs/tags/v1.5.0'."""
        out = []
        for c in self.calls:
            if c[0] == '/fake/gh':
                out.append( 'gh ' + ' '.join( c[1:3] ) )
            elif c[0] == 'git':
                if c[1] == 'switch':
                    out.append( 'git switch -c' if c[2] == '-c' else 'git switch main' )
                elif c[1] == 'push':
                    out.append( 'git push -u' if '-u' in c else 'git push ' + c[-1] )
                elif c[1] == 'tag':
                    out.append( 'git tag -a' if c[2] == '-a' else 'git tag --list' )
                else:
                    out.append( 'git ' + c[1] )
            else:
                out.append( 'unit tests' )
        return out

    def assertOrder( self, *wanted ):
        names = self.names()
        pos = -1
        for w in wanted:
            try:
                pos = names.index( w, pos + 1 )
            except ValueError:
                self.fail( '%r missing (or out of order) in %s'%( w, names ) )

    def mutating( self ):
        bad = ( 'git switch -c', 'git add', 'git commit', 'git push -u', 'gh pr create', 'gh pr merge',
                'git tag -a', 'gh release create' )
        return [ n for n in self.names() if n in bad or n.startswith( 'git push' ) ]

    # ---- the new release

    def test_new_release_goes_through_a_pr_and_tags_after_the_merge( self ):
        self.assertEqual( self.r.release( Args() ), 0 )
        self.assertOrder( 'git fetch', 'git switch -c', 'unit tests', 'git add', 'git commit', 'git push -u',
                          'gh pr create', 'gh pr view', 'gh pr view', 'git switch main', 'gh pr merge',
                          'gh pr view', 'git pull', 'git tag -a', 'git push refs/tags/v1.5.0', 'gh release create' )
        self.assertEqual( self.read( 'VERSION' ), '1.5.0\n' )
        self.assertTrue( self.r.release_is_prepared( self.read( 'CHANGELOG.md' ), '1.5.0' ) )
        self.assertIn( 'v1.5.0', self.releases )

    def test_merge_is_a_squash_with_admin_bypass_and_branch_cleanup( self ):
        self.r.release( Args() )
        merge = next( c for c in self.calls if c[0] == '/fake/gh' and c[1:3] == [ 'pr', 'merge' ] )
        self.assertEqual( merge[3:], [ '7', '--squash', '--admin', '--delete-branch' ] )

    def test_pr_is_opened_against_main_from_the_release_branch_with_the_notes( self ):
        self.r.release( Args() )
        create = next( c for c in self.calls if c[0] == '/fake/gh' and c[1:3] == [ 'pr', 'create' ] )
        self.assertEqual( create[create.index( '--base' ) + 1], 'main' )
        self.assertEqual( create[create.index( '--head' ) + 1], 'release/v1.5.0' )
        self.assertEqual( create[create.index( '--title' ) + 1], 'Release v1.5.0' )
        self.assertIn( 'a new thing', create[create.index( '--body' ) + 1] )

    def test_nothing_is_pushed_to_main_directly( self ):
        self.r.release( Args() )
        for c in self.calls:
            if c[:2] == [ 'git', 'push' ]:
                self.assertNotIn( 'main', c )
                self.assertNotIn( '--follow-tags', c )

    def test_merge_waits_for_pending_checks( self ):
        self.checks_script = [ [], [ { 'name': 'tests', 'bucket': 'pending' } ], [ { 'name': 'tests', 'bucket': 'pass' } ] ]
        self.r.release( Args() )
        names = self.names()
        polls = [ n for n in names[:names.index( 'gh pr merge' )] if n == 'gh pr view' ]
        self.assertEqual( len( polls ), 3 )

    def test_failed_checks_stop_before_the_merge( self ):
        self.checks_script = [ [ { 'name': 'tests', 'bucket': 'fail' } ] ]
        with self.assertRaises( self.r.Abort ) as cm:
            self.r.release( Args() )
        self.assertIn( 'tests', str( cm.exception ) )
        self.assertNotIn( 'gh pr merge', self.names() )
        self.assertNotIn( 'git tag -a', self.names() )

    def test_checks_that_never_finish_time_out_without_merging( self ):
        self.checks_script = [ [ { 'name': 'tests', 'bucket': 'pending' } ] ]
        clock = iter( range( 0, 100000, 60 ) )
        with self.assertRaises( self.r.Abort ) as cm:
            self.r.wait_for_checks( '/fake/gh', 7, 5, sleep=lambda s: None, clock=lambda: next( clock ) )
        self.assertIn( 'did not finish', str( cm.exception ) )

    def test_checks_are_never_read_with_gh_pr_checks( self ):
        self.r.release( Args() )
        for c in self.calls:
            if c[0] == '/fake/gh':
                self.assertNotEqual( c[1:3], [ 'pr', 'checks' ] )   # older gh has no --json there

    def test_a_flaky_read_is_retried_but_a_broken_one_aborts( self ):
        self.view_errors = 2
        self.assertEqual( self.r.release( Args() ), 0 )
        self.assertIn( 'git tag -a', self.names() )

    def test_exactly_three_unreadable_polls_abort_but_two_do_not( self ):
        self.fail_polls = { 0, 1, 2 }
        with self.assertRaises( self.r.Abort ):
            self.r.release( Args() )

    def test_unreadable_polls_must_be_consecutive_to_abort( self ):
        self.checks_script = [ [ { 'name': 'tests', 'bucket': 'pending' } ] ] * 1 + [ [ { 'name': 'tests', 'bucket': 'pending' } ] ] * 4 \
            + [ [ { 'name': 'tests', 'bucket': 'pass' } ] ]
        self.fail_polls = { 0, 1, 3, 4 }
        self.assertEqual( self.r.release( Args() ), 0 )
        self.assertIn( 'gh release create', self.names() )

    def test_three_unreadable_polls_in_a_row_abort_instead_of_waiting_forever( self ):
        self.view_errors = 99
        with self.assertRaises( self.r.Abort ) as cm:
            self.r.release( Args() )
        self.assertIn( 'three polls in a row', str( cm.exception ) )
        self.assertNotIn( 'gh pr merge', self.names() )

    def test_failing_unit_tests_stop_before_any_commit( self ):
        self.unit_tests_exit = 1
        with self.assertRaises( self.r.Abort ) as cm:
            self.r.release( Args() )
        self.assertIn( 'git restore', str( cm.exception ) )
        self.assertNotIn( 'git commit', self.names() )
        self.assertNotIn( 'git push -u', self.names() )

    def test_unmerged_pr_after_the_merge_command_is_an_error( self ):
        self.merge_state = 'OPEN'
        with self.assertRaises( self.r.Abort ):
            self.r.release( Args() )
        self.assertNotIn( 'git tag -a', self.names() )

    def test_explicit_bump_is_used( self ):
        a = Args()
        a.bump = 'patch'
        self.r.release( a )
        self.assertEqual( self.read( 'VERSION' ), '1.4.1\n' )
        self.assertIn( 'v1.4.1', self.releases )

    # ---- resuming

    def test_open_release_pr_resumes_at_the_checks( self ):
        self.open_prs = [ 7 ]
        self.remote_branches = { 'release/v1.5.0' }

        def merged_into_main():
            self.write( 'VERSION', '1.5.0\n' )
            self.write( 'CHANGELOG.md', CHANGELOG_PREPARED )
        self.on_pull = merged_into_main
        self.assertEqual( self.r.release( Args() ), 0 )
        names = self.names()
        self.assertNotIn( 'git switch -c', names )
        self.assertNotIn( 'gh pr create', names )
        self.assertNotIn( 'git commit', names )
        self.assertOrder( 'gh pr view', 'gh pr merge', 'git pull', 'git tag -a',
                          'git push refs/tags/v1.5.0', 'gh release create' )

    def test_main_that_does_not_show_the_release_after_the_merge_is_an_error( self ):
        self.open_prs = [ 7 ]
        self.remote_branches = { 'release/v1.5.0' }
        with self.assertRaises( self.r.Abort ) as cm:
            self.r.release( Args() )   # the pull brings nothing, so VERSION is still 1.4.0
        self.assertIn( 'does not show the release', str( cm.exception ) )
        self.assertNotIn( 'git tag -a', self.names() )

    def test_stale_remote_branch_without_a_pr_is_refused( self ):
        self.remote_branches = { 'release/v1.5.0' }
        with self.assertRaises( self.r.Abort ) as cm:
            self.r.release( Args() )
        self.assertIn( 'git push origin --delete', str( cm.exception ) )
        self.assertEqual( self.mutating(), [] )

    def test_stale_local_branch_is_refused( self ):
        self.local_branches = { 'release/v1.5.0' }
        with self.assertRaises( self.r.Abort ) as cm:
            self.r.release( Args() )
        self.assertIn( 'git branch -D', str( cm.exception ) )
        self.assertEqual( self.mutating(), [] )

    def test_merged_release_pr_goes_straight_to_tagging( self ):
        self.write( 'VERSION', '1.5.0\n' )
        self.write( 'CHANGELOG.md', CHANGELOG_PREPARED )
        self.r.release( Args() )
        names = self.names()
        self.assertNotIn( 'git switch -c', names )
        self.assertNotIn( 'gh pr create', names )
        self.assertOrder( 'git tag -a', 'git push refs/tags/v1.5.0', 'gh release create' )

    def test_tagged_head_only_publishes( self ):
        self.write( 'VERSION', '1.5.0\n' )
        self.write( 'CHANGELOG.md', CHANGELOG_PREPARED )
        self.tags.add( 'v1.5.0' )
        self.tag_commit['v1.5.0'] = self.head
        self.remote_tags.add( 'v1.5.0' )
        self.r.release( Args() )
        self.assertEqual( self.mutating(), [ 'gh release create' ] )

    def test_tagged_head_with_a_release_already_published_does_nothing( self ):
        self.write( 'VERSION', '1.5.0\n' )
        self.write( 'CHANGELOG.md', CHANGELOG_PREPARED )
        self.tags.add( 'v1.5.0' )
        self.tag_commit['v1.5.0'] = self.head
        self.remote_tags.add( 'v1.5.0' )
        self.releases.add( 'v1.5.0' )
        self.r.release( Args() )
        self.assertEqual( self.mutating(), [] )

    def test_tagged_head_missing_from_origin_pushes_only_the_tag( self ):
        self.write( 'VERSION', '1.5.0\n' )
        self.write( 'CHANGELOG.md', CHANGELOG_PREPARED )
        self.tags.add( 'v1.5.0' )
        self.tag_commit['v1.5.0'] = self.head
        self.r.release( Args() )
        self.assertEqual( self.mutating(), [ 'git push refs/tags/v1.5.0', 'gh release create' ] )

    # ---- preflight refusals (each must leave everything untouched)

    def refused( self, text ):
        with self.assertRaises( self.r.Abort ) as cm:
            self.r.release( Args() )
        self.assertIn( text, str( cm.exception ) )
        self.assertEqual( self.mutating(), [] )

    def test_refuses_a_branch_other_than_main( self ):
        self.branch = 'feature-branch'
        self.refused( 'git switch main' )

    def test_refuses_a_dirty_tree( self ):
        self.dirty = ' M betaconfig.py'
        self.refused( 'uncommitted' )

    def test_refuses_main_ahead_of_origin( self ):
        self.ahead = 1
        self.refused( 'ahead' )

    def test_refuses_main_behind_origin( self ):
        self.behind = 2
        self.refused( 'behind' )

    def test_refuses_without_gh( self ):
        self.have_gh = False
        self.refused( 'GitHub CLI' )

    def test_refuses_when_gh_is_logged_out( self ):
        self.gh_logged_in = False
        self.refused( 'gh auth login' )

    def test_refuses_an_existing_tag_for_the_new_version( self ):
        self.tags.add( 'v1.5.0' )
        self.tag_commit['v1.5.0'] = 'elsewhere'
        self.refused( 'already exists' )

    # ---- dry run

    def test_dry_run_changes_nothing_for_a_new_release( self ):
        a = Args()
        a.dry_run = True
        self.assertEqual( self.r.release( a ), 0 )
        self.assertEqual( self.mutating(), [] )
        self.assertEqual( self.read( 'VERSION' ), '1.4.0\n' )
        self.assertEqual( self.read( 'CHANGELOG.md' ), CHANGELOG_UNRELEASED )

    def test_dry_run_changes_nothing_when_a_tag_is_due( self ):
        self.write( 'VERSION', '1.5.0\n' )
        self.write( 'CHANGELOG.md', CHANGELOG_PREPARED )
        a = Args()
        a.dry_run = True
        self.r.release( a )
        self.assertEqual( self.mutating(), [] )

    def test_dry_run_changes_nothing_when_a_release_is_due_on_github( self ):
        self.write( 'VERSION', '1.5.0\n' )
        self.write( 'CHANGELOG.md', CHANGELOG_PREPARED )
        self.tags.add( 'v1.5.0' )
        self.tag_commit['v1.5.0'] = self.head
        a = Args()
        a.dry_run = True
        self.r.release( a )
        self.assertEqual( self.mutating(), [] )


if __name__ == '__main__':
    unittest.main()
