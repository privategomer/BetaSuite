"""
shell_harness.py - what a scratch copy of a shell launcher needs to find
its data folders.

The launchers ask tools/lib/where.py (through tools/lib/data_root.sh)
where each folder is, running it with $PYTHON_BIN. Tests that copy a
launcher into a scratch repo and stand a recording script in for
PYTHON_BIN copy WHERE_FILES too, and start the stand-in with
where_line(), so it hands where.py to the real interpreter and records
everything else.
"""

import sys

WHERE_FILES = ( 'tools/lib/data_root.sh', 'tools/lib/where.py', 'betautils_paths.py' )


def where_line():
    """A bash line for the top of a stand-in Python: where.py runs for real."""
    return 'if [ "$(basename "${1:-}")" = "where.py" ]; then exec "%s" "$@"; fi\n'%( sys.executable, )
