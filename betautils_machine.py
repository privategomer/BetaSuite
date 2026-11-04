"""
betautils_machine.py - The machine config: this computer's hardware
settings, kept outside the repo.

betaconfig.py holds settings that describe what to censor and how. A
few of its settings describe the COMPUTER instead: which encoder its GPU
has, how many render workers its memory holds, how big a detection batch
its GPU likes. Those have no right answer that suits every machine, and
when they lived in betaconfig.py one machine's tuning (h264_nvenc, cq
21) became every checkout's default, and pulling a branch could undo a
machine's own tuning.

The machine config is ../resources/betaconfig_machine.py, next to the
models and footage, which the repo never touches. betaconfig.py ends by
calling apply() on its own namespace, so every reader of betaconfig sees
the merged values with no other change. Precedence, lowest first:

    betaconfig.py  <  ../resources/betaconfig_machine.py  <  command-line flags

WHAT IT MAY SET. Only the settings in SETTINGS. Tuning settings
(min_prob, tracking, styles) are rejected there with a message, on
purpose: a detection setting that differs per machine makes two
machines' output and A/B comparisons disagree for a reason no log
explains.

FORMAT. Plain `name = value` lines in Python syntax, read with
ast.literal_eval, never executed. Comments are kept by the editor in
tools/setup/machine_config.py, which setup and tools/bench/tune_machine.py
use to write it.

PROBLEMS. A line that is not allowed, or a file that will not parse, is
skipped and recorded; nothing raises here, because betaconfig is
imported by every tool and a typo in a hardware setting must not stop
`machine_config.py show` from explaining it. validate_config turns the
recorded problems into errors, so a real run still stops on them.

BETASUITE_MACHINE_CONFIG overrides the path; set it to 'none' (or empty)
to ignore the file. The test suite does that, so tests see the repo's
defaults on every machine.
"""

import ast
import difflib
import os
import re
import sys

ENV_VAR = 'BETASUITE_MACHINE_CONFIG'
FILE_NAME = 'betaconfig_machine.py'
_REPO_DIR = os.path.dirname( os.path.abspath( __file__ ) )

# ( name, section, one-line description ). Order is the template's order.
SETTINGS = [
    ( 'gpu_enabled', 'GPU',
      '1 runs detection on the GPU (CUDA), 0 on the CPU. Setup sets this' ),
    ( 'cuda_device_id', 'GPU',
      'which GPU, when there is more than one' ),
    ( 'detector_backend', 'Detection',
      "detection batch size per backend, e.g. { 'nudenet_v3': { 'nn_batch_size': 8 } }" ),
    ( 'person_backend', 'Detection',
      "person-segmentation batch size per backend, e.g. { 'yolo11_seg': { 'batch_size': 4 } }" ),
    ( 'detection_prefetch_frames', 'Detection',
      'frames decoded ahead of the model on a second thread; 0 = decode and detect in turn' ),
    ( 'render_workers', 'Render',
      'chunks rendered at once; 0 = auto (CPU count / 2, at most 8, then the memory cap)' ),
    ( 'render_decode_threads', 'Render',
      "decoder threads per render worker; None = OpenCV's own (one per core), 0 = cores / workers" ),
    ( 'render_worker_mb_per_megapixel', 'Render',
      'memory one render worker takes per megapixel of frame; None = no memory cap' ),
    ( 'render_memory_reserve_mb', 'Render',
      'memory left free for everything else while rendering' ),
    ( 'render_chunk_seconds', 'Render',
      'length of one render chunk; shorter chunks are split further to keep workers busy' ),
    ( 'encode_video_codec', 'Encode',
      "ffmpeg encoder, e.g. 'libx264' (CPU) or 'h264_nvenc' (NVIDIA)" ),
    ( 'encode_preset', 'Encode',
      "the encoder's speed/size preset: x264 'ultrafast'..'veryslow', NVENC 'p1'..'p7'" ),
    ( 'encode_crf', 'Encode',
      'quality: CRF for x264, cq for NVENC; lower is better and bigger. The scales differ' ),
    ( 'preview_encode_preset', 'Encode',
      'preset for preview runs' ),
    ( 'structure_scan_workers', 'Structure scan',
      '`betabench structure-scan` files scanned at once; 0 = auto (one per core, at most 4)' ),
    ( 'vision_cap_monitor', 'BetaVision screen region', 'monitor to capture' ),
    ( 'vision_cap_top', 'BetaVision screen region', 'top edge, pixels' ),
    ( 'vision_cap_left', 'BetaVision screen region', 'left edge, pixels' ),
    ( 'vision_cap_height', 'BetaVision screen region', 'height, pixels' ),
    ( 'vision_cap_width', 'BetaVision screen region', 'width, pixels' ),
]

SETTING_NAMES = [ name for name, _section, _description in SETTINGS ]

# Settings that are dicts in betaconfig.py, and the one key inside each
# backend's block that the machine config may set. The rest of those
# blocks (thresholds, suppression rules) is tuning.
NESTED = {
    'detector_backend': ( 'nn_batch_size', ),
    'person_backend': ( 'batch_size', ),
}

# Blocks inside a NESTED dict that are not backends but still accept the
# keys above.
NESTED_EXTRA_BLOCKS = {
    'detector_backend': ( 'defaults', ),
    'person_backend': (),
}


class MachineConfig( object ):
    """
    What apply() did.

    Attributes:
        path: The file looked for, or None when turned off.
        exists: Whether it was there.
        applied: { setting path: value } actually merged, in file order.
            A setting path is a tuple: ( 'render_workers', ) or
            ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ).
        replaced: { setting path: betaconfig.py's value, or MISSING }.
        problems: Human-readable strings, one per skipped line.
    """

    def __init__( self, path ):
        self.path = path
        self.exists = False
        self.applied = {}
        self.replaced = {}
        self.problems = []

    def describe( self ):
        """One line for the log: where the file is and what it set."""
        if self.path is None:
            return "machine config: off (%s=none)"%( ENV_VAR, )
        if not self.exists:
            return ( "machine config: none at %s, so betaconfig.py's hardware settings apply "
                     "(tools/setup/machine_config.py init creates one)"%( self.path, ) )
        if not self.applied:
            return "machine config: %s sets nothing"%( self.path, )
        return "machine config: %s sets %s"%( self.path, ', '.join(
            "%s=%r"%( format_setting_path( path ), value ) for path, value in self.applied.items() ) )

    def set_names( self ):
        """The top-level setting names this file set, for error hints."""
        return sorted( { path[0] for path in self.applied } )


class _Missing( object ):
    def __repr__( self ):
        return '<not set>'

MISSING = _Missing()

# The most recent apply()'s result, which is what betaconfig's import
# produced unless something has called apply() since.
LOADED = None


def default_path():
    """../resources/betaconfig_machine.py, from the repo, not the cwd."""
    return os.path.normpath( os.path.join( _REPO_DIR, '..', 'resources', FILE_NAME ) )


def configured_path( environ=None ):
    """The file to read, or None when BETASUITE_MACHINE_CONFIG turns it off."""
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


def _unknown_name_message( name ):
    close = difflib.get_close_matches( name, SETTING_NAMES, n=1 )
    hint = " (did you mean %s?)"%( close[0], ) if close else ''
    return ( "%s is not a machine setting%s. A machine config sets only hardware settings: %s. "
             "Everything else belongs in betaconfig.py"%( name, hint, ', '.join( SETTING_NAMES ) ) )


def parse( text, filename=FILE_NAME ):
    """
    The assignments in a machine config's text.

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
        entries.append( ( name, value, node.lineno ) )
    return entries, problems


def _nested_updates( name, value, namespace, filename, lineno ):
    """
    ( [ ( setting path, value ) ... ], [ problem ... ] ) for one NESTED
    setting, checked against the blocks betaconfig.py actually has.
    """
    where = "%s line %d: %s"%( filename, lineno, name )
    allowed_keys = NESTED[name]
    if not isinstance( value, dict ):
        return [], [ "%s must be a dict of { backend: { %s } }, got %r"%(
            where, ', '.join( "'%s': ..."%( key, ) for key in allowed_keys ), value ) ]
    base = namespace.get( name )
    known = sorted( key for key, block in ( base.items() if isinstance( base, dict ) else () )
                    if isinstance( block, dict ) )
    known = sorted( set( known ) | set( NESTED_EXTRA_BLOCKS[name] ) )
    updates = []
    problems = []
    for block_name, block in value.items():
        if block_name not in known:
            problems.append( "%s[%r] is not a backend betaconfig.py has (known: %s)"%(
                where, block_name, ', '.join( known ) or 'none' ) )
            continue
        if not isinstance( block, dict ):
            problems.append( "%s[%r] must be a dict, got %r"%( where, block_name, block ) )
            continue
        for key, setting in block.items():
            if key not in allowed_keys:
                problems.append( "%s[%r][%r] is not a machine setting; only %s may be set per "
                                 "backend here. Everything else belongs in betaconfig.py"%(
                                     where, block_name, key, ', '.join( allowed_keys ) ) )
                continue
            updates.append( ( ( name, block_name, key ), setting ) )
    return updates, problems


def resolve( entries, namespace, filename=FILE_NAME ):
    """
    ( [ ( setting path, value ) ... ], [ problem ... ] ): the parsed
    entries a namespace would take, with every disallowed one reported.
    """
    updates = []
    problems = []
    for name, value, lineno in entries:
        if name not in SETTING_NAMES:
            problems.append( "%s line %d: %s"%( filename, lineno, _unknown_name_message( name ) ) )
        elif name in NESTED:
            nested, nested_problems = _nested_updates( name, value, namespace, filename, lineno )
            updates.extend( nested )
            problems.extend( nested_problems )
        else:
            updates.append( ( ( name, ), value ) )
    return updates, problems


def current_value( namespace, path ):
    """The namespace's value at a setting path, or MISSING."""
    value = namespace.get( path[0], MISSING )
    for part in path[1:]:
        if not isinstance( value, dict ) or part not in value:
            return MISSING
        value = value[part]
    return value


def _set( namespace, path, value ):
    if len( path ) == 1:
        namespace[ path[0] ] = value
        return
    container = namespace.setdefault( path[0], {} )
    for part in path[1:-1]:
        container = container.setdefault( part, {} )
    container[ path[-1] ] = value


def apply( namespace, path=None, environ=None, stderr=None ):
    """
    Merge the machine config into a betaconfig namespace.

    Called by betaconfig.py's last line with its own globals(). Never
    raises for anything in the file; see the module docstring.

    Args:
        namespace: The dict to update (betaconfig's globals()).
        path: The file to read; default configured_path().
        environ: For tests; default os.environ.
        stderr: Where to say that lines were skipped; default
            sys.stderr. validate_config repeats them as errors.

    Returns:
        A MachineConfig, also stored as LOADED.
    """
    global LOADED
    if path is None:
        path = configured_path( environ )
    result = MachineConfig( path )
    LOADED = result
    if path is None or not os.path.isfile( path ):
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
        updates, resolve_problems = resolve( entries, namespace, filename )
        result.problems.extend( problems + resolve_problems )
        for setting_path, value in updates:
            if setting_path not in result.replaced:
                result.replaced[ setting_path ] = current_value( namespace, setting_path )
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


def nvenc_codec( codec ):
    return isinstance( codec, str ) and codec.endswith( '_nvenc' )


def encoder_scale_warning( loaded ):
    """
    A warning when the machine config switches between x264 and NVENC
    without also setting encode_crf: the repo's value is then read on
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
# Writing the file: used by tools/setup/machine_config.py and
# tools/bench/tune_machine.py, so setup and tuning edit it the same way and
# keep the comments a person added.
# ---------------------------------------------------------------------------

TEMPLATE_HEADER = '''\
# BetaSuite machine config: this computer's hardware settings.
#
# Lives outside the repo, so pulls and branch switches never touch it. A
# setting here replaces the same setting in betaconfig.py on this machine
# only; anything left commented out uses betaconfig.py's value (shown).
# Hardware settings only, plain values only, nothing here is executed.
#
#   python tools/setup/machine_config.py show          what is set, and from where
#   python tools/setup/machine_config.py set NAME=VALUE
#   tools/bench/tune_machine.sh                        measure this machine, then offer to fill this in
#
# Reference: CONFIG_REFERENCE.md -> "Machine config".
'''


def format_value( value ):
    """A value as it is written in the file: betaconfig.py's dict spacing, repr otherwise."""
    if isinstance( value, dict ):
        if not value:
            return '{}'
        return '{ ' + ', '.join( "%r: %s"%( key, format_value( item ) )
                                 for key, item in value.items() ) + ' }'
    return repr( value )


def _commented_default( name, namespace ):
    value = namespace.get( name, MISSING ) if namespace is not None else MISSING
    if name in NESTED and isinstance( value, dict ):
        value = { block: { key: settings[key] for key in NESTED[name] if key in settings }
                  for block, settings in value.items() if isinstance( settings, dict ) }
        value = { block: settings for block, settings in value.items() if settings }
    if value is MISSING:
        return "# %s = "%( name, )
    return "# %s = %s"%( name, format_value( value ) )


def template_text( namespace ):
    """A new machine config: every setting commented out at betaconfig.py's value."""
    lines = [ TEMPLATE_HEADER.rstrip( '\n' ) ]
    section = None
    for name, this_section, description in SETTINGS:
        if this_section != section:
            section = this_section
            lines += [ '', '# --- %s %s'%( section, '-' * max( 3, 66 - len( section ) ) ) ]
        lines += [ '', '# %s'%( description, ), _commented_default( name, namespace ) ]
    return '\n'.join( lines ) + '\n'


def _assignments( text ):
    """{ name: ( first line, last line, end column ) } for each `name = value`, 1-based lines."""
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


def edit_text( text, updates=None, removals=None, namespace=None, comment=None ):
    """
    The machine config text with settings set or removed, everything
    else (comments, order, other settings) left as it was.

    Args:
        text: The file's current text ('' for a new file).
        updates: { setting path: value }, a path being ( name, ) or
            ( name, backend, key ) for a NESTED setting.
        removals: Setting paths to remove. A removed top-level setting
            goes back to a commented-out line at betaconfig.py's value.
        namespace: betaconfig's values, for those commented-out lines.
        comment: Trailing comment for every line written, e.g.
            'measured by tune_machine'.

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
            return targets[name]
        if name in existing:
            try:
                return ast.literal_eval( existing[name][3].value )
            except ( ValueError, TypeError, SyntaxError ):
                return None
        return None
    for path, value in updates.items():
        name = path[0]
        if len( path ) == 1:
            targets[name] = ( value, )
        else:
            merged = current( name )
            merged = dict( merged[0] if isinstance( merged, tuple ) else ( merged or {} ) )
            block = dict( merged.get( path[1] ) or {} )
            block[ path[2] ] = value
            merged[ path[1] ] = block
            targets[name] = ( merged, )
    for path in removals:
        name = path[0]
        if len( path ) == 1:
            targets[name] = None
            continue
        merged = current( name )
        merged = dict( merged[0] if isinstance( merged, tuple ) else ( merged or {} ) )
        block = dict( merged.get( path[1] ) or {} )
        block.pop( path[2], None )
        if block:
            merged[ path[1] ] = block
        else:
            merged.pop( path[1], None )
        targets[name] = ( merged, ) if merged else None

    suffix = "  # %s"%( comment, ) if comment else ''
    # Edit from the bottom up so earlier line numbers stay valid.
    spans = sorted( ( ( existing[name][0], existing[name][1], existing[name][2], name )
                      for name in targets if name in existing ), reverse=True )
    for first, last, end_col, name in spans:
        target = targets[name]
        trailing = lines[ last - 1 ][ end_col: ]
        keep_comment = trailing if trailing.strip().startswith( '#' ) and not comment else ''
        if target is None:
            replacement = [ _commented_default( name, namespace ) ]
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
            lines.append( line )
    return '\n'.join( lines ) + '\n'
