#!/usr/bin/env bash
# run_tests.sh - run every automated test.
#
# These are fast, hermetic tests: no real video decoding except the few
# synthetic end-to-end cases, which build their own tiny clips with
# ffmpeg and skip themselves if ffmpeg is unavailable. Not to be
# confused with tools/analysis and tools/tuning, which are operational
# scripts you run against real footage and read the output of, or with
# tools/bench/betabench.py, which measures rather than asserts.
#
#   ./tests/run_tests.sh              # everything
#   ./tests/run_tests.sh -k suppression   # just the matching tests
#
# Discovery runs from the repository root (-t ..) so the test modules can
# import betaconfig/betaconst/betautils_* the same way the application
# does. Running discovery from inside tests/ silently failed to import
# eleven of these modules and still reported a pass for the rest.

set -uo pipefail
cd "$(dirname "$0")/.."

exec python3 -m unittest discover -s tests -t . -p 'test_*.py' -v "$@"
