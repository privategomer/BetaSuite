#!/bin/bash
# catalog_rebuild.sh - build the library catalog again from the disk.
#
#   bash tools/setup/catalog_rebuild.sh          says what it will do, asks before rebuilding
#   bash tools/setup/catalog_rebuild.sh --yes    rebuilds without asking
#   bash tools/setup/catalog_rebuild.sh --help   every option
#
# Close the app first. Folders, pins and history are carried from the old
# file while it can be read; the old file is kept beside the new one. Log:
# <state folder>/logs/catalog_rebuild.log; --console-level (info) filters
# the terminal only.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/lib/python_env.sh"
exec "$PYTHON_BIN" "$REPO_ROOT/tools/setup/catalog_rebuild.py" "$@"
