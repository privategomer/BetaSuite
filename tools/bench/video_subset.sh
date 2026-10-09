#!/bin/bash
# video_subset.sh - turn VIDEOS_FIRST / VIDEOS_PER_FOLDER into betabench flags.
#
# Source it from a bench harness after harness_lib.sh, then:
#
#     video_subset_args || exit 1
#     "$PYTHON_BIN" betabench.py structure ${VIDEO_SUBSET_ARGS[@]+"${VIDEO_SUBSET_ARGS[@]}"}
#
# The ${...+...} form, not a bare "${VIDEO_SUBSET_ARGS[@]}": with set -u, bash
# before 4.4 treats an empty array as unbound and aborts the harness, which
# is exactly the no-subset default.
#
#   VIDEOS_FIRST=N        the first N source videos by file name, across
#                         every folder and both source trees
#   VIDEOS_PER_FOLDER=N   the first N by file name in each directory
#
# Neither set means every video, as before. Both set is refused rather
# than resolved silently, because either reading of it would surprise
# someone.
#
# One place for both harnesses, so the variable names and their checking
# cannot drift apart between scripts.

VIDEO_SUBSET_ARGS=()

video_subset_args() {
    local first="${VIDEOS_FIRST:-}"
    local per_folder="${VIDEOS_PER_FOLDER:-}"
    VIDEO_SUBSET_ARGS=()
    if [ -n "$first" ] && [ -n "$per_folder" ]; then
        log_error "VIDEOS_FIRST and VIDEOS_PER_FOLDER are both set; set one, not both"
        return 1
    fi
    if [ -n "$first" ]; then
        if ! [[ "$first" =~ ^[1-9][0-9]*$ ]]; then
            log_error "VIDEOS_FIRST must be a whole number of 1 or more, got '$first'"
            return 1
        fi
        VIDEO_SUBSET_ARGS=( --first-videos "$first" )
        log_info "video subset: the first $first source video(s) by name"
    elif [ -n "$per_folder" ]; then
        if ! [[ "$per_folder" =~ ^[1-9][0-9]*$ ]]; then
            log_error "VIDEOS_PER_FOLDER must be a whole number of 1 or more, got '$per_folder'"
            return 1
        fi
        VIDEO_SUBSET_ARGS=( --videos-per-folder "$per_folder" )
        log_info "video subset: the first $per_folder source video(s) by name in each folder"
    else
        log_info "video subset: none, every source video (set VIDEOS_FIRST or VIDEOS_PER_FOLDER to narrow it)"
    fi
    return 0
}

# Replay a betabench log's subset listing to the terminal, at info: the
# "video subset:" line and the file names under it. betabench writes the
# listing to its own log, which run_logged keeps off the terminal, so
# without this a subset run would not say which files it measured.
video_subset_replay() {
    local bench_log="$1"
    if [ "${#VIDEO_SUBSET_ARGS[@]}" -eq 0 ] || [ ! -f "$bench_log" ]; then
        return 0
    fi
    sed -E 's/^[0-9-]+ [0-9:,]+ \[[A-Z ]+\] //' "$bench_log" \
        | awk '/^video subset: /{ listing = 1; print; next }
               listing && /^  /{ print; next }
               listing{ exit }' \
        | while IFS= read -r line; do log_info "$line"; done
}
