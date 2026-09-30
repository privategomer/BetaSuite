"""
betautils_cli.py - Shared command-line flag handling for BetaSuite's
entry points (betatv.py, betastare.py, and the tuning/analysis scripts).

Two-step orchestration, always used together:
    parser = build_arg_parser(...)      # define what flags this tool accepts
    args = parser.parse_args()
    apply_cli_overrides(betaconfig, args)  # apply any flags actually passed

Scope is deliberately narrow: only "testing type" settings - things that
change how THIS run behaves (speed, output verbosity, how much of a file
gets processed) rather than what gets detected/censored. Detection/
censoring tuning (items_to_censor, class_suppression, censor styles,
area/time safety, min_prob, etc.) stays config-file-only: those are
numerous, often nested dicts/lists, and are meant to be deliberated over
and committed to betaconfig.py, not casually flipped per invocation.

Any flag actually passed on the command line overrides betaconfig.py for
THIS RUN ONLY - nothing is ever written back to betaconfig.py. Any flag
left unset keeps whatever betaconfig.py already says.
"""

import argparse

import betautils_detector as bu_detector

# Declarative table describing every optional CLI override: which
# argparse destination it reads from (the attribute name argparse
# derives from the flag, e.g. '--nn-batch-size' -> 'nn_batch_size'),
# which betaconfig.py attribute it should overwrite, and how to convert
# the parsed CLI value into that attribute's value.
#
# `to_config_value` defaults to "use the CLI value as-is" (an identity
# function) for flags whose type already matches betaconfig.py's own
# type (ints, floats, strings, lists of ints). Flags backed by an
# argparse choices=['on', 'off'] pair use `_on_off_to_bool` instead,
# since betaconfig.py's corresponding setting is a plain bool.
#
# Order here is preserved exactly in apply_cli_overrides' processing
# order and therefore in the printed "CLI overrides in effect" message.
def _identity( value ):
    return value

def _on_off_to_bool( value ):
    return value == 'on'

# Settings that resolve PER BACKEND are not in this table: a flat
# setattr onto the module-level attribute would be silently ignored by
# any backend that sets its own value, which both registered backends
# do. apply_cli_overrides writes those into the selected backend's own
# config block instead - see _BACKEND_SCOPED_OVERRIDES.
_ALWAYS_AVAILABLE_OVERRIDES = [
    # (argparse_dest, betaconfig_attr, to_config_value)
    ( 'debug_mode',       'debug_mode',       _identity ),
    ( 'video_censor_fps', 'video_censor_fps', _identity ),
    ( 'render_workers',   'render_workers',   _identity ),
    ( 'motion_interpolation',    'render_motion_interpolation',        _on_off_to_bool ),
    ( 'motion_max_span',         'render_motion_max_span_seconds',     _identity ),
    ( 'motion_collapse_growth',  'render_motion_collapse_max_growth',  _identity ),
    ( 'motion_size_window',      'render_motion_size_window_seconds',  _identity ),
    ( 'profile',                 'force_structure_profile',            _identity ),
]

# (argparse_dest, backend config key). Written into
# detector_backend[<selected>][key], so the override lands where
# betautils_detector actually reads it.
#
# These used to be mirrored onto module-level betaconfig attributes as
# well, for a config with no detector_backend section. Those tiers were
# removed in 2.5, so the mirror would write an attribute nothing reads -
# and validate_config now reports a stale module-level nn_batch_size or
# picture_sizes as an error, which would make every run using one of
# these flags fail validation. The mirror is gone for that reason.
_BACKEND_SCOPED_OVERRIDES = [
    ( 'nn_batch_size', 'nn_batch_size' ),
    ( 'picture_sizes', 'picture_sizes' ),
    ( 'variant',       'model_variant' ),
]

_LOGGING_OVERRIDES = [
    ( 'log_level',     'log_level',       _identity ),
    ( 'console_level', 'console_level',   _identity ),
    ( 'logging',       'logging_enabled', _on_off_to_bool ),
    ( 'stats',         'stats_enabled',   _on_off_to_bool ),
]

_PREVIEW_OVERRIDES = [
    ( 'preview',               'preview_mode_enabled', _on_off_to_bool ),
    ( 'preview_seconds',       'preview_max_seconds',  _identity ),
    ( 'preview_start_seconds', 'preview_start_seconds', _identity ),
    ( 'preview_random',        'preview_random_slice',  _on_off_to_bool ),
    ( 'preview_preset',        'preview_encode_preset', _identity ),
    ( 'encode_preset',         'encode_preset',         _identity ),
]


def _add_always_available_flags( parser ):
    """
    Register the CLI flags every BetaSuite entry point accepts,
    regardless of include_preview/include_logging.

    Args:
        parser: The argparse.ArgumentParser to add flags to (mutated in
            place).
    """
    parser.add_argument( '--backend', choices=bu_detector.registered_backend_names(),
        help="override betaconfig.detector_backend['selected'] for this run only. Precedence: "
             "BETASUITE_DETECTOR_BACKEND_OVERRIDE env var (if set) wins over this flag, which wins "
             "over betaconfig.py's 'selected' - see betautils_detector.selected_backend_name()." )
    parser.add_argument( '--nn-batch-size', type=int, metavar='N',
        help="override betaconfig.py's nn_batch_size for this run only (frames batched per neural-net inference call)" )
    parser.add_argument( '--debug-mode', type=int, metavar='N',
        help="override betaconfig.py's debug_mode for this run only" )
    parser.add_argument( '--picture-sizes', type=int, nargs='+', metavar='SIZE',
        help="override betaconfig.py's picture_sizes for this run only (space-separated, e.g. --picture-sizes 320 640) - the neural net runs once per size listed, so more/larger sizes cost more time" )
    parser.add_argument( '--video-censor-fps', type=float, metavar='FPS',
        help="override betaconfig.py's video_censor_fps for this run only (how many frames per second of video get detection/tracking)" )
    parser.add_argument( '--variant', metavar='NAME',
        help="override the selected backend's model_variant for this run only (e.g. --variant 640m "
             "for nudenet_v3). Precedence: BETASUITE_DETECTOR_VARIANT_OVERRIDE env var wins over this "
             "flag, which wins over betaconfig.py" )
    parser.add_argument( '--render-workers', type=int, metavar='N',
        help="override betaconfig.py's render_workers for this run only (0 = auto-detect from CPU count). "
             "Render chunks are independent, so this is how many encode at the same time" )
    parser.add_argument( '--motion-interpolation', choices=['on', 'off'],
        help="override betaconfig.py's render_motion_interpolation for this run only (slide each "
             "censor box between its detection samples instead of holding each sample's rectangle "
             "until the next one replaces it)" )
    parser.add_argument( '--motion-max-span', type=float, metavar='SECONDS',
        help="override betaconfig.py's render_motion_max_span_seconds for this run only (longest "
             "sample-to-sample gap to slide across; a longer hole is held, not slid through)" )
    parser.add_argument( '--motion-collapse-growth', type=float, metavar='RATIO',
        help="override betaconfig.py's render_motion_collapse_max_growth for this run only (collapse "
             "two same-frame samples of one object when their union is no more than this multiple of "
             "the larger box; 1.0 disables the collapse, which is what removes the size pulsing)" )
    parser.add_argument( '--profile', metavar='NAME',
        help="force a structure profile for this run only instead of letting the measured shot "
             "structure pick one (e.g. --profile split_screen, which no pre-detection measurement "
             "can identify, so automatic selection never picks it). An explicit --profile also "
             "overrides default_profile_enabled=False, and can name a profile marked manual_only. "
             "An unknown name is rejected by validation rather than silently falling back" )
    parser.add_argument( '--motion-size-window', type=float, metavar='SECONDS',
        help="override betaconfig.py's render_motion_size_window_seconds for this run only (hold each "
             "track's box at the largest size it was sampled at within this many seconds either side; "
             "0 means the whole track, which maximises coverage at ~20%% more censored area)" )


def _add_logging_flags( parser ):
    """
    Register the logging/stats-related CLI flags, for entry points that
    opt in via build_arg_parser(include_logging=True).

    Args:
        parser: The argparse.ArgumentParser to add flags to (mutated in
            place).
    """
    parser.add_argument( '--log-level', choices=[ 'trace', 'debug', 'info', 'warn', 'error' ],
        help="override betaconfig.py's log_level (the LOG FILE's level) for this run only" )
    parser.add_argument( '--console-level', choices=[ 'trace', 'debug', 'info', 'warn', 'error' ],
        help="override betaconfig.py's console_level (the TERMINAL's level) for this run only - set it "
             "higher than --log-level to keep the console readable while the file keeps the detail" )
    parser.add_argument( '--logging', choices=['on', 'off'],
        help="override betaconfig.py's logging_enabled for this run only" )
    parser.add_argument( '--stats', choices=['on', 'off'],
        help="override betaconfig.py's stats_enabled for this run only" )


def _add_preview_flags( parser ):
    """
    Register the preview-mode-related CLI flags, for entry points that
    opt in via build_arg_parser(include_preview=True).

    Args:
        parser: The argparse.ArgumentParser to add flags to (mutated in
            place).
    """
    parser.add_argument( '--preview', choices=['on', 'off'],
        help="override betaconfig.py's preview_mode_enabled for this run only" )
    parser.add_argument( '--preview-seconds', type=float, metavar='SECONDS',
        help="override betaconfig.py's preview_max_seconds for this run only" )
    parser.add_argument( '--preview-start-seconds', type=float, metavar='SECONDS',
        help="override betaconfig.py's preview_start_seconds for this run only - use the SAME value across runs to keep comparing the same slice" )
    parser.add_argument( '--preview-random', choices=['on', 'off'],
        help="override betaconfig.py's preview_random_slice for this run only" )
    parser.add_argument( '--preview-preset', metavar='PRESET',
        help="override betaconfig.py's preview_encode_preset for this run only" )
    parser.add_argument( '--encode-preset', metavar='PRESET',
        help="override betaconfig.py's encode_preset (the full, non-preview run's x264 preset) for this run only" )


def build_arg_parser( description, include_preview=False, include_logging=False ):
    """
    Build an argparse.ArgumentParser with BetaSuite's standard set of
    "testing type" override flags.

    Args:
        description: Program description shown in --help, passed
            straight through to argparse.ArgumentParser.
        include_preview: If True, also register the preview-mode flags
            (--preview, --preview-seconds, etc.) - only meaningful for
            entry points that actually support preview mode (betatv.py).
        include_logging: If True, also register the logging/stats flags
            (--log-level, --logging, --stats).

    Returns:
        A configured argparse.ArgumentParser. Call .parse_args() on it,
        then pass the result to apply_cli_overrides().
    """
    parser = argparse.ArgumentParser( description=description )

    _add_always_available_flags( parser )

    if include_logging:
        _add_logging_flags( parser )

    if include_preview:
        _add_preview_flags( parser )

    return parser


def _overrides_to_apply():
    """
    Full flattened list of every possible (argparse_dest,
    betaconfig_attr, to_config_value) override entry, across all flag
    groups, in the same fixed order apply_cli_overrides has always used.

    Returns:
        A list of (argparse_dest, betaconfig_attr, to_config_value)
        tuples. Entries whose flag wasn't registered on the parser (e.g.
        preview flags when include_preview=False) simply won't be
        present as an attribute on `args` and are skipped harmlessly by
        apply_cli_overrides's getattr(..., None) check.
    """
    return _ALWAYS_AVAILABLE_OVERRIDES + _LOGGING_OVERRIDES + _PREVIEW_OVERRIDES


def apply_cli_overrides( betaconfig, args ):
    """
    Apply any CLI flags the caller actually passed onto the in-memory
    betaconfig module, so they take precedence over betaconfig.py for
    this run only.

    Must be called BEFORE bu_config.validate_config(), so validation -
    and everything downstream - sees the effective, CLI-overridden
    values rather than the file's raw values.

    Args:
        betaconfig: The betaconfig module object itself (mutated in
            place via setattr for each override actually applied).
        args: The argparse.Namespace returned by
            build_arg_parser(...).parse_args().

    Returns:
        The list of "attr=value" override description strings actually
        applied, in the same order they were applied - the caller can
        log/print these. Also prints a single summary line to stdout if
        any overrides were applied.
    """
    applied_overrides = []

    # --backend special-cased, and applied FIRST (before nn_batch_size
    # below): writes into betaconfig.detector_backend['selected'] rather
    # than a flat attribute, since that's what
    # bu_detector.selected_backend_name() actually reads as its
    # config-file tier. Applied before nn_batch_size's own special case
    # on purpose - that one resolves "the currently selected backend" via
    # selected_backend_name() to decide which backend's block to write
    # nn_batch_size into, so if both --backend and --nn-batch-size are
    # given together, the backend switch has to already be in effect by
    # the time nn_batch_size resolves it, or --nn-batch-size would land
    # in the OLD backend's block instead of the one this run actually
    # uses. Note betaconfig.py's own env var, BETASUITE_DETECTOR_BACKEND_
    # OVERRIDE, still wins over this flag if both are set - that check
    # lives in selected_backend_name() itself, so nothing here needs to
    # special-case it; this just sets the config-file-tier value the env
    # var would otherwise fall through to.
    backend_cli_value = getattr( args, 'backend', None )
    if backend_cli_value is not None:
        backend_setting = getattr( betaconfig, 'detector_backend', None )
        if backend_setting is None:
            backend_setting = {}
            setattr( betaconfig, 'detector_backend', backend_setting )
        backend_setting[ 'selected' ] = backend_cli_value
        applied_overrides.append( "detector_backend['selected']=%r"%(backend_cli_value) )

    # Backend-scoped overrides are written into the selected backend's
    # own block, because that is the tier betautils_detector reads first.
    # Applied AFTER --backend so "the selected backend" is already the one
    # this run will actually use; otherwise these would land in the old
    # backend's block. Nothing is mirrored onto a module-level attribute
    # any more - see _BACKEND_SCOPED_OVERRIDES for why.
    for argparse_dest, backend_key in _BACKEND_SCOPED_OVERRIDES:
        cli_value = getattr( args, argparse_dest, None )
        if cli_value is None:
            continue
        backend_name = bu_detector.selected_backend_name()
        backend_setting = getattr( betaconfig, 'detector_backend', None )
        if backend_setting is None:
            backend_setting = {}
            setattr( betaconfig, 'detector_backend', backend_setting )
        backend_setting.setdefault( backend_name, {} )[ backend_key ] = cli_value
        applied_overrides.append( "detector_backend[%r][%r]=%r"%(
            backend_name, backend_key, cli_value ) )

    for argparse_dest, betaconfig_attr, to_config_value in _overrides_to_apply():
        cli_value = getattr( args, argparse_dest, None )
        if cli_value is None:
            continue
        config_value = to_config_value( cli_value )
        applied_overrides.append( "%s=%r"%(betaconfig_attr, config_value) )
        setattr( betaconfig, betaconfig_attr, config_value )

    # Memoised views of betaconfig are now stale. Everything that caches
    # a resolved config value has to be told, or a CLI override would be
    # applied to betaconfig and then ignored by the table the pipeline
    # actually reads.
    import betautils_config as bu_config
    bu_config.invalidate_config_caches()

    if applied_overrides:
        import betautils_log as bu_log
        bu_log.get_logger().info(
            "CLI overrides in effect for this run (betaconfig.py itself is untouched): %s"%(
                ", ".join( applied_overrides ) ) )

    return applied_overrides
