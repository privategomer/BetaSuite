"""
betautils_config_store.py - read, check and change the local config
layer: the one write path for the engine's config.* commands and
tools/setup/local_config.py.

Precedence, lowest first: betaconfig.py < betaconfig_local.py < per-run
overrides (betatv.py's flags today, a job's overrides later). Writing
only ever changes the middle layer, and only once the change would pass
validation as betatv.py runs it.

Checking a candidate means changing the live betaconfig module, since
validation and the cache keys read it; every change is undone before a
call returns. Nothing else may read betaconfig meanwhile, so a config
change must not run while a job does.
"""

import contextlib
import copy
import os
import tempfile

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_local as bu_local
import betautils_settings as bu_settings

CACHE_KEYS = bu_settings.CACHE_KEYS
MISSING = bu_local.MISSING


def _get( values, path ):
    """The value at a path in a { setting name: value } dict, or MISSING."""
    node = values.get( path[0], MISSING )
    for part in path[1:]:
        if not isinstance( node, dict ) or part not in node:
            return MISSING
        node = node[part]
    return node


class _Null( object ):
    def write( self, _text ):
        pass


class Evaluation( object ):
    """
    What betatv.py would make of a local config text, plus overrides.

    Attributes:
        loaded: The LocalConfig apply() returned for the text.
        errors: Validation errors the baseline does not already have,
            the text's own unappliable lines included, each a Problem.
        values: { setting name: value } with the text and overrides applied.
        local_values: the same with the text only.
        extra: what the extra callable returned, if one was given.
        keys: { 'detection', 'censor', 'encode': key } for that config.
    """

    def __init__( self, loaded, errors, values, keys ):
        self.loaded = loaded
        self.errors = errors
        self.values = values
        self.keys = keys

    @property
    def ok( self ):
        return not self.errors


def baseline_values( module=None, loaded=None ):
    """
    { setting name: value } of betaconfig.py alone: the live module with
    each local-config value put back as it was before the file applied.
    """
    module = module if module is not None else betaconfig
    loaded = loaded if loaded is not None else bu_local.LOADED
    values = { name: copy.deepcopy( getattr( module, name ) )
               for name in bu_local.known_names( vars( module ) ) }
    replaced = loaded.replaced if loaded is not None else {}
    # Longest paths first, so a whole value put back is not then edited inside.
    for path, old in sorted( replaced.items(), key=lambda item: -len( item[0] ) ):
        _put( values, path, old )
    return values


def _put( values, path, value ):
    if len( path ) == 1:
        if value is bu_local.MISSING:
            values.pop( path[0], None )
        else:
            values[path[0]] = copy.deepcopy( value )
        return
    node = values.get( path[0] )
    for part in path[1:-1]:
        if not isinstance( node, dict ) or part not in node:
            return
        node = node[part]
    if not isinstance( node, dict ):
        return
    if value is bu_local.MISSING:
        node.pop( path[-1], None )
    else:
        node[path[-1]] = copy.deepcopy( value )


def detection_keys( backend_name ):
    """
    Every detection key a run of the live config can use: one per
    picture size and per sample rate a structure profile can select.
    A file's own key is one of these, so a change that moves none of
    them re-detects nothing.
    """
    sizes = bu_detector.get_picture_sizes( backend_name ) or [ 0 ]
    rates = sorted( bu_detector.profile_sample_rates( backend_name ), key=float )
    return [ bu_hash.dictionary_hash( bu_cache.detection_identity( backend_name, size, fps ),
                                      betaconst.detection_key_len )
             for size in sizes for fps in rates ]


def current_keys():
    """The cache keys of the live config, hashed without recording run keys."""
    backend_name = bu_detector.selected_backend_name()
    bu_config.invalidate_config_caches()
    return {
        'detection': '+'.join( detection_keys( backend_name ) ),
        'censor': bu_hash.dictionary_hash( bu_cache.censor_identity( backend_name ),
                                           betaconst.censor_key_len ),
        'encode': bu_hash.dictionary_hash( bu_cache.encode_identity(), betaconst.encode_key_len ),
    }


def moved( before, after ):
    """The cache keys that differ, in CACHE_KEYS order."""
    return [ kind for kind in CACHE_KEYS if before.get( kind ) != after.get( kind ) ]


def apply_overrides( namespace, overrides ):
    """Per-run overrides, { path tuple: value }, set into a betaconfig namespace."""
    for path, value in overrides.items():
        bu_local._set( namespace, tuple( path ), copy.deepcopy( value ) )


class ConfigStore( object ):
    """
    The local config file and the live betaconfig it feeds.

    Args:
        path: The local config file; default bu_local.configured_path().
            None when the layer is turned off.
        legacy_path: The old machine config to look for (tests).
    """

    def __init__( self, path=bu_local.MISSING, legacy_path=None ):
        self.path = bu_local.configured_path() if path is bu_local.MISSING else path
        self.legacy_path = legacy_path
        self.base = baseline_values()

    # --- the file ----------------------------------------------------------

    def text( self ):
        if not self.path or not os.path.exists( self.path ):
            return ''
        with open( self.path, 'r', encoding='utf-8' ) as handle:
            return handle.read()

    def write( self, text ):
        """Write the file (atomically) and make the live config what the next run reads."""
        bu_local.write_text( self.path, text )
        self.reload()

    def reload( self ):
        """The live betaconfig: betaconfig.py, then the local file as it is now."""
        namespace = vars( betaconfig )
        self._restore( namespace, self.base )
        bu_local.apply( namespace, path=self.path, stderr=_Null(), legacy_path=self.legacy_path )
        bu_config.invalidate_config_caches()

    # --- checking ----------------------------------------------------------

    @staticmethod
    def _restore( namespace, values ):
        for name in bu_local.known_names( namespace ):
            if name not in values:
                del namespace[name]
        for name, value in values.items():
            namespace[name] = copy.deepcopy( value )
        bu_config.invalidate_config_caches()

    @contextlib.contextmanager
    def _live( self ):
        """Run a block that changes the live config; put it back afterwards."""
        namespace = vars( betaconfig )
        saved = { name: copy.deepcopy( namespace[name] ) for name in bu_local.known_names( namespace ) }
        saved_loaded = bu_local.LOADED
        try:
            yield namespace
        finally:
            self._restore( namespace, saved )
            bu_local.LOADED = saved_loaded

    def evaluate( self, text, overrides=None, extra=None ):
        """
        What betatv.py would make of this local config text.

        Args:
            text: The candidate betaconfig_local.py text.
            overrides: { path tuple: value } per-run overrides on top.
            extra: Optional callable( betaconfig ) run after the overrides,
                such as applying betatv.py's flags; its return value is
                kept as Evaluation.extra.

        Returns:
            An Evaluation. Only errors betaconfig.py alone does not
            already have count.
        """
        with self._live() as namespace:
            self._restore( namespace, self.base )
            bu_local.LOADED = None
            baseline = set( bu_config._collect_validation_errors() )
            handle, candidate = tempfile.mkstemp( suffix='.py', prefix='betaconfig_local.' )
            try:
                with os.fdopen( handle, 'w', encoding='utf-8' ) as out:
                    out.write( text )
                loaded = bu_local.apply( namespace, path=candidate, stderr=_Null(),
                                         legacy_path=self.legacy_path )
            finally:
                os.remove( candidate )
            name = os.path.basename( self.path ) if self.path else bu_local.FILE_NAME
            loaded.path = self.path
            loaded.problems = [ bu_settings.rule( 'local_config.line_not_applied',
                                                  problem.replace( os.path.basename( candidate ), name ),
                                                  path='' )
                                for problem in loaded.problems ]
            local_values = { setting: copy.deepcopy( namespace[setting] )
                             for setting in bu_local.known_names( namespace ) }
            apply_overrides( namespace, overrides or {} )
            extra_result = extra( betaconfig ) if extra is not None else None
            bu_config.invalidate_config_caches()
            bu_local.LOADED = loaded
            errors = list( loaded.problems ) + [
                error for error in bu_config._collect_validation_errors()
                if error not in baseline and error not in loaded.problems ]
            values = { setting: copy.deepcopy( namespace[setting] )
                       for setting in bu_local.known_names( namespace ) }
            keys = current_keys()
        evaluation = Evaluation( loaded, errors, values, keys )
        evaluation.local_values = local_values
        evaluation.extra = extra_result
        return evaluation

    # --- the config.* commands -------------------------------------------------

    def edit( self, updates=None, removals=None, comment=None, section=None, dry_run=False ):
        """
        Set or remove local settings, if the result would pass validation.

        Args:
            updates: { path tuple: value }.
            removals: [ path tuple ].
            dry_run: Check and report, write nothing.

        Returns:
            ( Evaluation, moved cache keys, written ). Nothing is written
            when the evaluation has errors. The moved keys are those of
            the whole config; a change can also move keys for some files
            only (betautils_settings.MAY_MOVE).

        Raises:
            ValueError: the file does not parse, so it cannot be edited safely.
        """
        # From the file as it is now, not the engine's memory of it: the
        # tool or an editor may have changed it since.
        current = self.evaluate( self.text() )
        before = current.keys
        text = self.text() or bu_local.template_text()
        if removals and not self.text():
            new_text = self.text()
        else:
            new_text = bu_local.edit_text( text, updates or {}, removals or [], comment, section )
        evaluation = self.evaluate( new_text )
        evaluation.before_values = current.values
        changes = moved( before, evaluation.keys )
        written = False
        if evaluation.ok and not dry_run and new_text != self.text():
            self.write( new_text )
            written = True
        return evaluation, changes, written

    def sources( self, overrides=None ):
        """
        { path tuple: 'local' | 'run' } for the paths the local file and
        the overrides set; every other setting's source is 'default'.
        """
        loaded = bu_local.LOADED
        found = { tuple( path ): 'local' for path in ( loaded.applied if loaded is not None else {} ) }
        for path in ( overrides or {} ):
            found[tuple( path )] = 'run'
        return found


def source_of( path, sources ):
    """The source of a path: its own, or the nearest enclosing path's, or 'default'."""
    path = tuple( path )
    for length in range( len( path ), 0, -1 ):
        if path[:length] in sources:
            return sources[path[:length]]
    # A whole setting set by the file covers every key inside it; a key
    # set inside it does not make the whole setting local.
    return 'default'
