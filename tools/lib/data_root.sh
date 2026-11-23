#!/bin/bash
# data_root.sh - the folder that holds resources/ and output/, for shell tools.
#
# Same rule as betautils_paths.data_root(): BETASUITE_DATA_ROOT unset or
# empty means the folder beside the repo; otherwise it must be an absolute
# path to an existing folder ('~' is not expanded).
#
#     source "$(dirname "${BASH_SOURCE[0]}")/../lib/data_root.sh"
#     DATA_ROOT="$(betasuite_data_root "$REPO_ROOT")" || exit 1

betasuite_data_root() {
    local repo_root="$1"
    local value="${BETASUITE_DATA_ROOT:-}"
    if [ -z "$value" ]; then
        (cd "$repo_root/.." && pwd)
        return
    fi
    case "$value" in
        /*|[A-Za-z]:[/\\]*) ;;
        *)
            echo "BETASUITE_DATA_ROOT must be an absolute path, got '$value' ('~' is not expanded)" >&2
            return 1 ;;
    esac
    if [ ! -d "$value" ]; then
        echo "BETASUITE_DATA_ROOT is '$value', which is not an existing folder" >&2
        return 1
    fi
    (cd "$value" && pwd)
}
