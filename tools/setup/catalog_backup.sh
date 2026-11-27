#!/bin/bash
# catalog_backup.sh - back up the library catalog, or put a backup back.
#
#   bash tools/setup/catalog_backup.sh                         asks which, and where
#   bash tools/setup/catalog_backup.sh backup [--to PATH]      copies the catalog (default: <data folder>/backups)
#   bash tools/setup/catalog_backup.sh restore FILE [--yes]    puts FILE back as the catalog
#   bash tools/setup/catalog_backup.sh --help                  every option
#
# A backup carries the watched folders, type pins, run history and
# feedback, and is safe while the app runs. Restore needs the app closed
# and keeps the current catalog beside the new one. Log:
# <state folder>/logs/catalog_backup.log; --console-level (info) filters
# the terminal only.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/lib/python_env.sh"
exec "$PYTHON_BIN" "$REPO_ROOT/tools/setup/catalog_backup.py" "$@"
