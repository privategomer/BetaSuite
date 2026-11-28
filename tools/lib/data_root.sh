#!/bin/bash
# data_root.sh - where a 1.x data folder is, for shell tools.
#
# Asks betautils_paths through tools/lib/where.py, so the shell and Python
# always agree: under BETASUITE_DATA_ROOT the 1.x path below that folder,
# otherwise the OS folder for that kind of data (logs in the state folder,
# caches in the cache folder, ...). Fails, with the reason on stderr, when
# a data location variable is unusable.
#
#     source "$(dirname "${BASH_SOURCE[0]}")/../lib/data_root.sh"
#     LOG_DIR="$(betasuite_path "$REPO_ROOT" output/logs)" || exit 1

betasuite_path() {
    local repo_root="$1" relative="$2"
    "${PYTHON_BIN:-python3}" "$repo_root/tools/lib/where.py" "$relative"
}
