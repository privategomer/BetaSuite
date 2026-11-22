"""
betaengine.py - The engine as a child process of the desktop UI.

    python betaengine.py serve [--log-file PATH] [--log-level LEVEL]
                               [--console-level LEVEL]

serve speaks the protocol described in protocol/README.md: JSON lines,
commands on stdin, results and events on stdout, logs on stderr and in
the log file. It runs until a shutdown command or the end of its input.
Exit codes: 0 for a clean stop, 2 for a usage error or a failure before
serving started, 130 for SIGINT, 143 for SIGTERM.

The command-line tools (betatv.py, betastare.py) do not go through
here and are unchanged by it.
"""

import argparse

LEVELS = ( 'trace', 'debug', 'info', 'warn', 'error' )


def build_parser():
    import betautils_version as bu_version
    parser = argparse.ArgumentParser( description="BetaSuite engine for the desktop UI" )
    parser.add_argument( '--version', action='version', version='BetaSuite %s'%( bu_version.get_version(), ) )
    commands = parser.add_subparsers( dest='command', required=True, metavar='COMMAND' )
    serve = commands.add_parser( 'serve', help="speak the engine protocol on stdin and stdout" )
    serve.add_argument( '--log-file', metavar='PATH',
                        help="log file (default: betaconfig's log_path)" )
    serve.add_argument( '--log-level', choices=LEVELS,
                        help="log file level (default: betaconfig's log_level)" )
    serve.add_argument( '--console-level', choices=LEVELS,
                        help="stderr level (default: betaconfig's console_level)" )
    return parser


def main( argv=None ):
    args = build_parser().parse_args( argv )
    if args.command == 'serve':
        import betautils_serve
        return betautils_serve.run( log_path=args.log_file, log_level=args.log_level,
                                    console_level=args.console_level )
    return 2


if __name__ == '__main__':
    raise SystemExit( main() )
