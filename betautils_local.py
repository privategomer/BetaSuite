"""
betautils_local.py - The local config: a person's own settings, kept
out of git.

betaconfig.py holds the shipped defaults and is never edited by hand.
A person's own values (this computer's encoder, a terminal palette, a
batch size) go in betaconfig_local.py beside it, which git ignores and
which holds only the settings that person changed. The one rule: copy a
setting into betaconfig_local.py to change it, delete the line to go
back to the default.

This replaced ../resources/betaconfig_machine.py, which split settings
by "is it hardware". The split that matters is "is it mine or shipped":
console_palette is a personal taste, not a property of the machine, and
had nowhere to go but betaconfig.py, so it was committed as everyone's
default or cleaned out of every commit by hand.

betaconfig.py ends by calling apply() on its own namespace, so every
reader of betaconfig sees the merged values with no other change, the
cache keys and the stats record included. Precedence, lowest first
(PRECEDENCE, the one definition of it):

    betaconfig.py  <  betaconfig_local.py  <  command-line flags

WHAT IT MAY SET. Any setting betaconfig.py has. A name betaconfig.py
does not have is an error with the closest real name. A dict merges into
betaconfig.py's dict key by key, at every depth, so
`detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }` changes one
number and keeps the rest of that block. Everything else replaces
betaconfig.py's value whole: numbers, strings, lists, tuples, and a
complete censor style (a dict with a 'type' key), since merging two
styles key by key would carry a blur's 'method' into a pixel style.
A merged key betaconfig.py's dict does not have is kept and warned
about (new_keys): validation checks most blocks' keys but not every
one (a backend block's, for one), so a misspelt key there would
otherwise be silently ignored. Removing a key from betaconfig.py's dict
is not possible here; set it to its off value.

FORMAT. Plain `name = value` lines in Python syntax, read with
ast.literal_eval, never executed. The editor below (used by
tools/setup/local_config.py, setup and tools/bench/tune_machine.py)
keeps the comments a person added.

PROBLEMS. A line that cannot be applied, or a file that will not parse,
is skipped and recorded; nothing raises here, because betaconfig is
imported by every tool and a typo must not stop `local_config.py show`
from explaining it. validate_config turns the recorded problems into
errors that name the file and line, so a real run still stops on them.

BETASUITE_LOCAL_CONFIG overrides the path; set it to 'none' (or empty)
to ignore the file. The test suite does that, so tests see the shipped
defaults on every machine.
"""

import ast
import copy
import difflib
import os
import re
import sys
import types

ENV_VAR = 'BETASUITE_LOCAL_CONFIG'
FILE_NAME = 'betaconfig_local.py'
_REPO_DIR = os.path.dirname( os.path.abspath( __file__ ) )

# Where each layer's values come from, lowest precedence first. `show`
# labels every value with one of these.
PRECEDENCE = ( 'default', 'local', 'CLI' )
LAYER_FILES = { 'default': 'betaconfig.py', 'local': FILE_NAME, 'CLI': 'command line' }

# The settings tune_machine, setup and migrate write live under this
# header, so a person's own lines stay apart from the measured ones.
HARDWARE_SECTION = 'hardware (written by tune_machine)'

# The file this replaced. No longer read; `local_config.py migrate`
# moves its settings into the local config and renames it.
LEGACY_FILE_NAME = 'betaconfig_machine.py'
MIGRATED_SUFFIX = '.migrated'
MIGRATE_COMMAND = 'python tools/setup/local_config.py migrate'
SET_COMMAND = 'python tools/setup/local_config.py set'

# A value longer than this is logged by name only, so the run-start line
# stays one readable line when a whole style list is overridden.
SHORT_VALUE_CHARS = 40


class _Missing( object ):
    def __repr__( self ):
        return '<not set>'

MISSING = _Missing()


class LocalConfig( object ):
    """
    What apply() did.

    Attributes:
        path: The file looked for, or None when turned off.
        exists: Whether it was there.
        applied: { setting path: value } actually merged, in file order.
            A setting path is a tuple: ( 'render_workers', ) or
            ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ).
        replaced: { setting path: betaconfig.py's value, or MISSING }.
        new_keys: [ ( setting path, closest existing key or None, line ) ]
            for merged keys betaconfig.py's dict does not have.
        problems: Human-readable strings, one per skipped line.
        legacy_machine_path: The old machine config when it still exists
            and so still needs migrating, else None.
    """

    def __init__( self, path ):
        self.path = path
        self.exists = False
        self.applied = {}
        self.replaced = {}
        self.new_keys = []
        self.problems = []
        self.legacy_machine_path = None

    def describe( self ):
        """The one run-start line: where the file is and what it overrides."""
        if self.path is None:
            return "local config: off (%s=none)"%( ENV_VAR, )
        if not self.exists:
            return "local config: none at %s, so betaconfig.py's defaults apply"%( self.path, )
        if not self.applied:
            return "local config: %s overrides nothing"%( self.path, )
        return "local overrides active: %s (from %s)"%( ', '.join(
            "%s=%s"%( format_setting_path( path ), short_value( value ) )
            for path, value in self.applied.items() ), self.path )

    def set_names( self ):
        """The top-level setting names this file set, for error hints."""
        return sorted( { path[0] for path in self.applied } )

    def overridden_names( self ):
        """Every setting path this file set, as text, for the stats record."""
        return [ format_setting_path( path ) for path in self.applied ]


# The most recent apply()'s result, which is what betaconfig's import
# produced unless something has called apply() since.
LOADED = None


def default_path():
    """betaconfig_local.py beside betaconfig.py, from the repo, not the cwd."""
    return os.path.join( _REPO_DIR, FILE_NAME )


def legacy_machine_path():
    """../resources/betaconfig_machine.py, where the old machine config lived."""
    return os.path.normpath( os.path.join( _REPO_DIR, '..', 'resources', LEGACY_FILE_NAME ) )


def configured_path( environ=None ):
    """The file to read, or None when BETASUITE_LOCAL_CONFIG turns it off."""
    environ = os.environ if environ is None else environ
    value = environ.get( ENV_VAR )
    if value is None:
        return default_path()
    if value.strip().lower() in ( '', 'none', 'off' ):
        return None
    return os.path.abspath( os.path.expanduser( value ) )


def format_setting_path( path ):
    """( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ) -> "detector_backend['nudenet_v3']['nn_batch_size']"."""
    return path[0] + ''.join( "[%r]"%( part, ) for part in path[1:] )


def short_value( value ):
    """repr( value ) when it is short, else what kind of value it is."""
    text = repr( value )
    if len( text ) <= SHORT_VALUE_CHARS:
        return text
    if isinstance( value, ( list, tuple, dict ) ):
        return "<%s of %d>"%( type( value ).__name__, len( value ) )
    return "<%s>"%( type( value ).__name__, )


def known_names( namespace ):
    """The settings a namespace defines: its public names that are not modules or functions."""
    return sorted( name for name, value in namespace.items()
                   if not name.startswith( '_' ) and not isinstance( value, types.ModuleType )
                   and not callable( value ) )


def _unknown_name_message( name, known ):
    close = difflib.get_close_matches( name, known, n=1 )
    hint = " (did you mean %s?)"%( close[0], ) if close else ''
    return ( "%s is not a setting betaconfig.py has%s, so nothing reads it. Only settings that "
             "exist in betaconfig.py can be overridden here"%( name, hint ) )


def parse( text, filename=FILE_NAME ):
    """
    The assignments in a local config's text.

    Returns:
        ( [ ( name, value, lineno ) ... ], [ problem ... ] ). A line that
        is not a plain `name = value` is a problem, not an entry.
    """
    try:
        tree = ast.parse( text, filename=filename )
    except SyntaxError as err:
        return [], [ "%s line %s: not valid Python syntax (%s); nothing in the file was applied"%(
            filename, err.lineno, err.msg ) ]
    entries = []
    problems = []
    first_line = {}
    for node in tree.body:
        if isinstance( node, ast.Expr ) and isinstance( getattr( node, 'value', None ), ast.Constant ) \
                and isinstance( node.value.value, str ):
            continue   # a docstring or a bare string used as a comment
        if not ( isinstance( node, ast.Assign ) and len( node.targets ) == 1
                 and isinstance( node.targets[0], ast.Name ) ):
            problems.append( "%s line %d: only `name = value` lines are allowed here"%(
                filename, node.lineno ) )
            continue
        name = node.targets[0].id
        try:
            value = ast.literal_eval( node.value )
        except ( ValueError, TypeError, SyntaxError, MemoryError, RecursionError ):
            problems.append( "%s line %d: %s's value must be a plain value (a number, string, "
                             "True/False/None, list or dict), not an expression"%(
                                 filename, node.lineno, name ) )
            continue
        # Python would let the second line win without a word, and the
        # person editing the first would see no effect.
        if name in first_line:
            problems.append( "%s line %d: %s is already set on line %d; delete one of them"%(
                filename, node.lineno, name, first_line[name] ) )
        first_line.setdefault( name, node.lineno )
        entries.append( ( name, value, node.lineno ) )
    return entries, problems


def is_censor_style( value ):
    """A complete censor style: replaced whole, never merged key by key."""
    return isinstance( value, dict ) and 'type' in value


def merge_updates( base, value, path ):
    """
    [ ( setting path, value ) ... ]: the leaves `value` sets when merged
    over `base` at `path`. Dicts merge key by key at every depth; any
    other value, and a complete censor style, is one leaf.
    """
    if isinstance( value, dict ) and isinstance( base, dict ) and not is_censor_style( value ):
        updates = []
        for key, item in value.items():
            updates.extend( merge_updates( base.get( key, MISSING ), item, path + ( key, ) ) )
        return updates
    return [ ( path, value ) ]


def current_value( namespace, path ):
    """The namespace's value at a setting path, or MISSING."""
    value = namespace.get( path[0], MISSING )
    for part in path[1:]:
        if not isinstance( value, dict ) or part not in value:
            return MISSING
        value = value[part]
    return value


# Maps keyed by label whose entries validation checks key by key, so a
# misspelled key inside one is already an error. A new key under one is
# a new label or a new per-label setting, which is ordinary (the tuners
# write them), and warning about it on every run taught people to ignore
# the warning. tests/test_local_config.py checks each is validated.
VALIDATED_LABEL_MAPS = ( 'item_overrides', 'windows', 'overrides' )


def is_validated_new_key( path ):
    """Whether a new key at this setting path is, or sits inside, a map validation checks."""
    return any( part in VALIDATED_LABEL_MAPS for part in path )


def resolve( entries, namespace, filename=FILE_NAME ):
    """
    What the parsed entries would do to a namespace, without doing it.

    Returns:
        ( [ ( setting path, value, lineno ) ... ],
          [ ( setting path, closest key or None, lineno ) ... ] new keys,
          [ problem ... ] ).
    """
    known = known_names( namespace )
    known_set = set( known )
    updates = []
    new_keys = []
    problems = []
    for name, value, lineno in entries:
        if name not in known_set:
            problems.append( "%s line %d: %s"%( filename, lineno, _unknown_name_message( name, known ) ) )
            continue
        for path, leaf in merge_updates( namespace[name], value, ( name, ) ):
            updates.append( ( path, leaf, lineno ) )
            if ( len( path ) > 1 and current_value( namespace, path ) is MISSING
                    and not is_validated_new_key( path ) ):
                parent = current_value( namespace, path[:-1] )
                siblings = [ str( key ) for key in parent ] if isinstance( parent, dict ) else []
                close = difflib.get_close_matches( str( path[-1] ), siblings, n=1 )
                new_keys.append( ( path, close[0] if close else None, lineno ) )
    return updates, new_keys, problems


def _set( namespace, path, value ):
    if len( path ) == 1:
        namespace[ path[0] ] = value
        return
    container = namespace.setdefault( path[0], {} )
    for part in path[1:-1]:
        container = container.setdefault( part, {} )
    container[ path[-1] ] = value


def apply( namespace, path=None, environ=None, stderr=None, legacy_path=None ):
    """
    Merge the local config into a betaconfig namespace.

    Called by betaconfig.py's last line with its own globals(). Never
    raises for anything in the file; see the module docstring.

    Args:
        namespace: The dict to update (betaconfig's globals()).
        path: The file to read; default configured_path().
        environ: For tests; default os.environ.
        stderr: Where to say that lines were skipped; default
            sys.stderr. validate_config repeats them as errors.
        legacy_path: The old machine config to look for; default
            legacy_machine_path(). Not looked for when the layer is off.

    Returns:
        A LocalConfig, also stored as LOADED.
    """
    global LOADED
    if path is None:
        path = configured_path( environ )
    result = LocalConfig( path )
    LOADED = result
    if path is None:
        return result
    legacy = legacy_machine_path() if legacy_path is None else legacy_path
    if os.path.isfile( legacy ):
        result.legacy_machine_path = legacy
    if not os.path.isfile( path ):
        return result
    result.exists = True
    try:
        with open( path, 'r', encoding='utf-8' ) as handle:
            text = handle.read()
    except ( OSError, UnicodeDecodeError ) as err:
        result.problems.append( "%s could not be read (%s); nothing in it was applied"%( path, err ) )
    else:
        filename = os.path.basename( path )
        entries, problems = parse( text, filename )
        updates, new_keys, resolve_problems = resolve( entries, namespace, filename )
        result.problems.extend( problems + resolve_problems )
        result.new_keys = new_keys
        for setting_path, value, _lineno in updates:
            if setting_path not in result.replaced:
                result.replaced[ setting_path ] = copy.deepcopy( current_value( namespace, setting_path ) )
            _set( namespace, setting_path, value )
            result.applied[ setting_path ] = value
    if result.problems:
        stream = stderr if stderr is not None else sys.stderr
        try:
            stream.write( "%s: %d line(s) skipped:\n  %s\n"%(
                path, len( result.problems ), '\n  '.join( result.problems ) ) )
        except Exception:
            pass
    return result


def changed_paths( before, after, path=() ):
    """Setting paths whose value differs between two values, descending into dicts both sides have."""
    if isinstance( before, dict ) and isinstance( after, dict ):
        changed = []
        for key in list( before ) + [ key for key in after if key not in before ]:
            changed.extend( changed_paths( before.get( key, MISSING ), after.get( key, MISSING ),
                                           path + ( key, ) ) )
        return changed
    return [] if before == after else [ path ]


# ---------------------------------------------------------------------------
# Warnings validate_config and `local_config.py show` print
# ---------------------------------------------------------------------------

def legacy_warning( loaded ):
    """One line while the old machine config still waits to be migrated, else None."""
    if loaded is None or loaded.legacy_machine_path is None:
        return None
    return ( "%s is no longer read, so its settings are not in effect; run `%s` to move them "
             "into %s"%( loaded.legacy_machine_path, MIGRATE_COMMAND, FILE_NAME ) )


def new_key_warnings( loaded ):
    """One line per merged key betaconfig.py's dict does not have."""
    if loaded is None:
        return []
    filename = os.path.basename( loaded.path ) if loaded.path else FILE_NAME
    warnings = []
    for path, close, lineno in loaded.new_keys:
        hint = " (closest existing key: %r)"%( close, ) if close else ''
        warnings.append( "%s line %d: %s is a key betaconfig.py's %s does not have%s. It is merged in "
                         "as new; not every block's keys are validated, so a misspelling here can be "
                         "silently ignored"%( filename, lineno, format_setting_path( path ),
                                              format_setting_path( path[:-1] ), hint ) )
    return warnings


def nvenc_codec( codec ):
    return isinstance( codec, str ) and codec.endswith( '_nvenc' )


def encoder_scale_warning( loaded ):
    """
    A warning when the local config switches between x264 and NVENC
    without also setting encode_crf: the shipped value is then read on
    the other encoder's scale (x264 CRF 17 as NVENC cq 17), which runs
    and silently changes file size and quality. None otherwise.
    """
    if loaded is None:
        return None
    codec_path = ( 'encode_video_codec', )
    if codec_path not in loaded.applied or ( 'encode_crf', ) in loaded.applied:
        return None
    before = loaded.replaced.get( codec_path, MISSING )
    after = loaded.applied[ codec_path ]
    if before is MISSING or nvenc_codec( before ) == nvenc_codec( after ):
        return None
    return ( "%s sets encode_video_codec = %r but not encode_crf, so betaconfig.py's encode_crf "
             "(written for %r) is read as %s. The two scales do not match number for number; set "
             "encode_crf there too (`betabench encode` measures the match)"%(
                 os.path.basename( loaded.path ), after, before,
                 "NVENC's cq" if nvenc_codec( after ) else "x264's CRF" ) )


# ---------------------------------------------------------------------------
# Migrating the old machine config
# ---------------------------------------------------------------------------

def migration_plan( legacy_entries, local_namespace, defaults ):
    """
    What `migrate` copies from the old machine config, decided without
    touching a file.

    Args:
        legacy_entries: parse()'s entries for the old file.
        local_namespace: { name: value } the local config sets now.
        defaults: betaconfig.py's own values.

    Returns:
        { 'copy': [ ( setting path, value ) ],
          'same_as_default': [ setting path ],
          'already_local': [ setting path ],
          'conflicts': [ ( setting path, old value, local value ) ] }.
        A setting the local config already sets keeps its local value.
    """
    plan = { 'copy': [], 'same_as_default': [], 'already_local': [], 'conflicts': [] }
    for name, value, _lineno in legacy_entries:
        for path, leaf in merge_updates( defaults.get( name, MISSING ), value, ( name, ) ):
            local = current_value( local_namespace, path )
            if local is not MISSING:
                if local == leaf:
                    plan['already_local'].append( path )
                else:
                    plan['conflicts'].append( ( path, leaf, local ) )
            elif current_value( defaults, path ) == leaf:
                plan['same_as_default'].append( path )
            else:
                plan['copy'].append( ( path, leaf ) )
    return plan


# ---------------------------------------------------------------------------
# Writing the file: used by tools/setup/local_config.py (and through it
# setup and tools/bench/tune_machine.sh), so every writer edits it the same
# way and keeps the comments a person added.
# ---------------------------------------------------------------------------

TEMPLATE_HEADER = '''\
# BetaSuite local config: your own settings, kept out of git.
#
# Copy a setting from betaconfig.py into this file to change it; delete
# the line to go back to the default. Plain `name = value` lines only,
# nothing here is executed. Never edit betaconfig.py itself.
#
#   python tools/setup/local_config.py show    every setting, its value and where it came from
#
# Reference: CONFIG_REFERENCE.md, "Local config".
'''

_SECTION_PREFIX = '# --- '


def template_text():
    """A new local config: the header, and no settings."""
    return TEMPLATE_HEADER


def section_header( section ):
    return "%s%s %s"%( _SECTION_PREFIX, section, '-' * max( 3, 66 - len( section ) ) )


def format_value( value ):
    """A value as it is written in the file: betaconfig.py's dict spacing, repr otherwise."""
    if isinstance( value, dict ):
        if not value:
            return '{}'
        return '{ ' + ', '.join( "%r: %s"%( key, format_value( item ) )
                                 for key, item in value.items() ) + ' }'
    return repr( value )


def _shell_word( text ):
    """text as one shell word, in double quotes when it has single quotes a shell would not expand."""
    import shlex
    if "'" in text and not any( char in text for char in '"$`\\!' ):
        return '"%s"'%( text, )
    return shlex.quote( text )


def set_command( updates ):
    """
    The one local_config.py command that makes these changes, for a tool
    to print where it used to say "put this in betaconfig.py".

    Args:
        updates: [ ( setting path, value ) ... ], in the order to show.
    """
    return ' '.join( [ SET_COMMAND ] + [ _shell_word( "%s=%s"%( '.'.join( str( part ) for part in path ),
                                                                  format_value( value ) ) )
                                         for path, value in updates ] )


def write_text( path, text ):
    """
    Replace the file at path with text in one step (a temporary file
    beside it, then a rename), so a reader never sees half a file. The
    temporary name matches .gitignore's /.betaconfig_local.py.*.tmp.
    """
    import tempfile
    directory = os.path.dirname( os.path.abspath( path ) )
    os.makedirs( directory, exist_ok=True )
    handle, temporary = tempfile.mkstemp( dir=directory, prefix='.%s.'%( os.path.basename( path ), ),
                                          suffix='.tmp' )
    with os.fdopen( handle, 'w', encoding='utf-8' ) as out:
        out.write( text )
    os.replace( temporary, path )


def _assignments( text ):
    """{ name: ( first line, last line, end column, node ) } for each `name = value`, 1-based lines."""
    found = {}
    try:
        tree = ast.parse( text )
    except SyntaxError:
        return found
    for node in tree.body:
        if isinstance( node, ast.Assign ) and len( node.targets ) == 1 \
                and isinstance( node.targets[0], ast.Name ):
            found[ node.targets[0].id ] = ( node.lineno, node.end_lineno, node.end_col_offset, node )
    return found


def trailing_comments( text ):
    """{ name: the comment after its value, without the '#' } for each `name = value` that has one."""
    lines = text.split( '\n' )
    comments = {}
    for name, ( _first, last, end_col, _node ) in _assignments( text ).items():
        trailing = lines[ last - 1 ][ end_col: ].strip()
        if trailing.startswith( '#' ):
            comments[name] = trailing.lstrip( '#' ).strip()
    return comments


def _with_leaf( value, path, leaf ):
    """A copy of dict `value` with `leaf` set at `path` (a tuple of keys)."""
    merged = copy.deepcopy( value ) if isinstance( value, dict ) else {}
    container = merged
    for part in path[:-1]:
        if not isinstance( container.get( part ), dict ):
            container[part] = {}
        container = container[part]
    container[ path[-1] ] = leaf
    return merged


def _without_leaf( value, path ):
    """A copy of dict `value` with the key at `path` removed and any dict it empties pruned."""
    if not isinstance( value, dict ):
        return value
    merged = copy.deepcopy( value )
    chain = [ merged ]
    for part in path[:-1]:
        nxt = chain[-1].get( part )
        if not isinstance( nxt, dict ):
            return merged
        chain.append( nxt )
    chain[-1].pop( path[-1], None )
    for depth in range( len( path ) - 1, 0, -1 ):
        if not chain[depth]:
            chain[ depth - 1 ].pop( path[ depth - 1 ], None )
    return merged


def _insert_index( lines, section ):
    """Where a new setting's line goes: the end of its section, or above the first section."""
    if section is None:
        for index, line in enumerate( lines ):
            if line.startswith( _SECTION_PREFIX ):
                while index > 0 and not lines[ index - 1 ].strip():
                    index -= 1
                return index, False
        return len( lines ), False
    header = section_header( section )
    for index, line in enumerate( lines ):
        if line.rstrip() == header:
            end = len( lines )
            for later in range( index + 1, len( lines ) ):
                if lines[later].startswith( _SECTION_PREFIX ):
                    end = later
                    break
            while end > index + 1 and not lines[ end - 1 ].strip():
                end -= 1
            return end, False
    return len( lines ), True


def edit_text( text, updates=None, removals=None, comment=None, section=None ):
    """
    The local config text with settings set or removed, everything else
    (comments, order, other settings) left as it was.

    Args:
        text: The file's current text ('' for a new file).
        updates: { setting path: value }. A path longer than one name
            sets one key inside that setting's dict.
        removals: Setting paths to remove. A removed top-level setting's
            line is deleted, so betaconfig.py's value applies again.
        comment: Trailing comment for every line written, e.g.
            'tune_machine 20010101_120000'.
        section: The section header a new line goes under (created at
            the end if missing); None puts it above the first section.

    Raises:
        ValueError: the text does not parse, so it cannot be edited
            safely; nothing is written.
    """
    updates = dict( updates or {} )
    removals = list( removals or [] )
    if text.strip():
        try:
            ast.parse( text )
        except SyntaxError as err:
            raise ValueError( "the file does not parse (line %s: %s); fix it by hand first"%(
                err.lineno, err.msg ) )
    lines = text.split( '\n' ) if text else []
    if lines and lines[-1] == '':
        lines.pop()

    # The new value of every top-level name being touched; None = remove.
    targets = {}
    existing = _assignments( text )
    def current( name ):
        if name in targets:
            return targets[name][0] if targets[name] is not None else None
        if name in existing:
            try:
                return ast.literal_eval( existing[name][3].value )
            except ( ValueError, TypeError, SyntaxError ):
                return None
        return None
    for path, value in updates.items():
        name = path[0]
        targets[name] = ( value, ) if len( path ) == 1 else ( _with_leaf( current( name ), path[1:], value ), )
    for path in removals:
        name = path[0]
        if len( path ) == 1:
            targets[name] = None
            continue
        if name not in targets and name not in existing:
            continue
        merged = _without_leaf( current( name ), path[1:] )
        targets[name] = ( merged, ) if merged not in ( None, {} ) else None

    suffix = "  # %s"%( comment, ) if comment else ''
    # Edit from the bottom up so earlier line numbers stay valid.
    spans = sorted( ( ( existing[name][0], existing[name][1], existing[name][2], name )
                      for name in targets if name in existing ), reverse=True )
    for first, last, end_col, name in spans:
        target = targets[name]
        trailing = lines[ last - 1 ][ end_col: ]
        keep_comment = trailing if trailing.strip().startswith( '#' ) and not comment else ''
        if target is None:
            replacement = []
        else:
            replacement = [ "%s = %s%s%s"%( name, format_value( target[0] ), suffix, keep_comment ) ]
        lines[ first - 1 : last ] = replacement
    for name, target in targets.items():
        if name in existing or target is None:
            continue
        line = "%s = %s%s"%( name, format_value( target[0] ), suffix )
        pattern = re.compile( r'^#\s*%s\s*=' % re.escape( name ) )
        for index, existing_line in enumerate( lines ):
            if pattern.match( existing_line ):
                lines[index] = line
                break
        else:
            index, new_section = _insert_index( lines, section )
            if new_section:
                while lines and not lines[-1].strip():
                    lines.pop()
                lines += [ '', section_header( section ), line ]
            else:
                # The first setting under the file's header comment gets a
                # blank line, so the header does not read as its comment.
                if index > 0 and lines[ index - 1 ].startswith( '#' ) and all(
                        not part.strip() or part.startswith( '#' ) for part in lines[:index] ):
                    lines.insert( index, '' )
                    index += 1
                lines.insert( index, line )
    return '\n'.join( lines ) + '\n'
