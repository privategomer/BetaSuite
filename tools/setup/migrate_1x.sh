#!/bin/bash
# migrate_1x.sh - copy a 1.x tree's settings, models, caches and logs into
# the 2.x folders by kind. Never moves or changes the 1.x tree.
#
#   bash tools/setup/migrate_1x.sh               asks for the 1.x folder, shows the plan, asks before copying
#   bash tools/setup/migrate_1x.sh --dry-run     the plan only
#   bash tools/setup/migrate_1x.sh --help        every option
#
# Safe to run again: only what is missing is copied. Unattended (no
# terminal, no --yes) it shows the plan and copies nothing. Log:
# <state folder>/logs/migrate_1x.log, every level; --console-level
# (info) filters the terminal only.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/lib/python_env.sh"
exec "$PYTHON_BIN" "$REPO_ROOT/tools/setup/migrate_1x.py" "$@"
