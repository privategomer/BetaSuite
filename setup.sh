#!/usr/bin/env bash
# setup.sh - one-shot BetaSuite environment setup for Linux (macOS: CPU only).
# Safe to re-run: every step checks before it acts. See SETUP.md.
#
#   ./setup.sh                 # interactive, picks GPU if an NVIDIA GPU is found
#   ./setup.sh --yes --cpu     # unattended, CPU build, default answers
#   ./setup.sh --help
#
# In a git checkout it first offers to switch to the latest release tag, so
# a fresh clone of main installs the released code rather than work in progress.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR" || exit 1

ORIG_ARGS=("$@")
MODE=""                 # gpu | cpu | "" (ask)
ASSUME_YES=0
USE_UV=1
PY_VERSION="$(tr -d '[:space:]' < .python-version 2>/dev/null)"
RUN_TESTS=""            # 1 | 0 | "" (ask)
VENV_DIR="$REPO_DIR/.venv"
MIN_DRIVER_MAJOR=525
RELEASE_CHECK=1
# Set once a run has switched to (or chosen to stay off) the latest release, so
# the restarted script, which may be an older release's copy, doesn't ask again.
[ "${BETASUITE_RELEASE_CHECKED:-0}" = 1 ] && RELEASE_CHECK=0

usage() {
    cat <<EOF
Usage: ./setup.sh [options]

  --gpu | --cpu          install the CUDA build or the CPU build (default: ask,
                         suggesting GPU when nvidia-smi finds one)
  --python VERSION       Python version (default: ${PY_VERSION:-3.12}, from .python-version)
  --tests | --no-tests   run the unit tests at the end (default: ask, no)
  --no-uv                use an existing pythonX.Y on PATH and 'python -m venv'
                         instead of uv
  --no-release-check     stay on the current branch or commit instead of offering
                         to switch to the latest release (for development)
  --yes, -y              accept the default answer to every prompt
  --log-level LEVEL      stdout level: trace|debug|info|warn|error (default info).
                         setup.log always gets everything.
  -h, --help             this text
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) MODE=gpu ;;
        --cpu) MODE=cpu ;;
        --python) PY_VERSION="$2"; shift ;;
        --tests) RUN_TESTS=1 ;;
        --no-tests) RUN_TESTS=0 ;;
        --no-uv) USE_UV=0 ;;
        --no-release-check) RELEASE_CHECK=0 ;;
        -y|--yes) ASSUME_YES=1 ;;
        --log-level) export LOG_LEVEL="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1"; usage; exit 2 ;;
    esac
    shift
done
PY_VERSION="${PY_VERSION:-3.12}"

export HARNESS_LOG_FILE="${HARNESS_LOG_FILE:-$REPO_DIR/setup.log}"
# shellcheck source=tools/tuning/harness_lib.sh
source "$REPO_DIR/tools/tuning/harness_lib.sh"

die() { log_error "$*"; log_error "setup stopped. full log: $HARNESS_LOG_FILE"; exit 1; }

# ask_yes_no "question" y|n  -> 0 for yes. Uses the default with --yes or no terminal.
ask_yes_no() {
    local prompt="$1" default="$2" hint reply
    [ "$default" = y ] && hint="Y/n" || hint="y/N"
    if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then
        log_info "auto-answer '$default': $prompt"
        [ "$default" = y ]; return
    fi
    log_debug "AWAITING INPUT: $prompt"
    read -r -p "$prompt [$hint] " reply
    reply="${reply:-$default}"
    log_debug "input received: $reply"
    case "$reply" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

# ---------------------------------------------------------------- release
# Offer to move a git checkout onto the newest vX.Y.Z tag. Runs before
# anything else, and restarts setup from the release's own copy of this
# script after switching (exec, so this copy never reads another line).
release_check() {
    local git=(git -C "$REPO_DIR") latest here head_desc ahead
    if [ "$RELEASE_CHECK" != 1 ]; then
        log_debug "release check skipped"; return 0
    fi
    if [ ! -d "$REPO_DIR/.git" ] || ! command -v git >/dev/null 2>&1; then
        log_debug "not a git checkout; using the files as they are"; return 0
    fi
    run_logged "fetch release tags" "${git[@]}" fetch --tags --quiet \
        || log_warn "could not fetch tags; checking against the tags already here"
    latest="$("${git[@]}" tag --list 'v[0-9]*' --sort=-v:refname | grep -v -- '-' | head -n1)"
    if [ -z "$latest" ]; then
        log_warn "no release tags found; installing the current checkout"; return 0
    fi
    head_desc="$("${git[@]}" describe --tags --exact-match HEAD 2>/dev/null || true)"
    if [ "$head_desc" = "$latest" ]; then
        log_info "on the latest release, $latest"; return 0
    fi
    here="$("${git[@]}" rev-parse --abbrev-ref HEAD)"
    [ "$here" = HEAD ] && here="${head_desc:-commit $("${git[@]}" rev-parse --short HEAD)}"
    ahead="$("${git[@]}" rev-list --count "$latest..HEAD" 2>/dev/null || echo 0)"
    if [ -n "$("${git[@]}" status --porcelain --untracked-files=no)" ]; then
        log_warn "latest release is $latest, but $here has uncommitted changes; staying on $here"
        return 0
    fi
    if [ "$ahead" -gt 0 ] && "${git[@]}" merge-base --is-ancestor "$latest" HEAD; then
        log_info "$here is $ahead commit(s) past $latest (changes not released yet)"
    fi
    if ! ask_yes_no "Latest release is $latest; this checkout is on $here. Switch to $latest?" y; then
        log_info "staying on $here"; return 0
    fi
    run_logged "check out $latest" "${git[@]}" -c advice.detachedHead=false checkout --quiet "$latest" \
        || die "could not check out $latest"
    log_info "switched to $latest; restarting setup from that release"
    export BETASUITE_RELEASE_CHECKED=1
    exec bash "$REPO_DIR/setup.sh" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
}
release_check
export BETASUITE_RELEASE_CHECKED=1

# ---------------------------------------------------------------- preflight
log_info "BetaSuite setup starting in $REPO_DIR (log: $HARNESS_LOG_FILE)"
log_info "version: $(cat "$REPO_DIR/VERSION" 2>/dev/null || echo unknown) ($(git -C "$REPO_DIR" describe --tags --always --dirty 2>/dev/null || echo 'not a git checkout'))"
OS="$(uname -s)"
log_info "platform: $OS $(uname -m)"
case "$OS" in
    Linux) ;;
    Darwin) log_warn "macOS: no CUDA, so this will be a CPU install"; MODE=cpu ;;
    *) die "unsupported platform '$OS'. On Windows run setup.cmd instead." ;;
esac

HAVE_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAVE_GPU=1
    while IFS= read -r line; do log_info "GPU: $line"; done \
        < <(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)
    DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1)"
    if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -lt "$MIN_DRIVER_MAJOR" ] 2>/dev/null; then
        log_warn "NVIDIA driver $DRIVER_MAJOR is older than $MIN_DRIVER_MAJOR; CUDA 12 needs $MIN_DRIVER_MAJOR+. Update the driver or use --cpu."
    fi
else
    log_info "no working nvidia-smi found; GPU build not suggested"
fi

if [ -z "$MODE" ]; then
    if [ "$HAVE_GPU" = 1 ]; then default=y; else default=n; fi
    if ask_yes_no "Install the NVIDIA GPU (CUDA) build? 'n' installs the CPU build" "$default"; then
        MODE=gpu
    else
        MODE=cpu
    fi
fi
[ "$MODE" = gpu ] && [ "$HAVE_GPU" = 0 ] && log_warn "GPU build requested but no NVIDIA GPU detected; it will fall back to CPU at runtime"
log_info "install mode: $MODE, python $PY_VERSION"

# ---------------------------------------------------------------- ffmpeg
install_ffmpeg() {
    local pm_cmd=""
    if command -v apt-get >/dev/null; then pm_cmd="apt-get update && apt-get install -y ffmpeg"
    elif command -v dnf >/dev/null; then pm_cmd="dnf install -y ffmpeg"
    elif command -v pacman >/dev/null; then pm_cmd="pacman -S --noconfirm ffmpeg"
    elif command -v zypper >/dev/null; then pm_cmd="zypper --non-interactive install ffmpeg"
    elif command -v brew >/dev/null; then run_logged "brew install ffmpeg" brew install ffmpeg; return
    else
        log_error "no known package manager; install ffmpeg yourself so 'ffmpeg' and 'ffprobe' are on PATH"
        return 1
    fi
    log_info "installing ffmpeg needs sudo: $pm_cmd"
    if [ -t 0 ]; then sudo -v || return 1
    elif ! sudo -n true 2>/dev/null; then
        log_error "sudo needs a password and there is no terminal; run: sudo sh -c '$pm_cmd'"
        return 1
    fi
    run_logged "install ffmpeg" sudo sh -c "$pm_cmd"
}

if command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; then
    log_info "ffmpeg found: $(ffmpeg -version 2>/dev/null | head -n1)"
elif ask_yes_no "ffmpeg/ffprobe not found on PATH. Install with your package manager?" y; then
    install_ffmpeg || die "ffmpeg install failed"
else
    log_warn "continuing without ffmpeg; betatv.py cannot render until it is installed"
fi

# ---------------------------------------------------------------- python + venv
VENV_PY="$VENV_DIR/bin/python"

venv_python_version() {
    [ -x "$VENV_PY" ] && "$VENV_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

ensure_uv() {
    for candidate in uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if command -v "$candidate" >/dev/null 2>&1; then UV="$(command -v "$candidate")"; return 0; fi
    done
    ask_yes_no "uv (Python installer and venv tool from astral.sh) is not installed. Install it to ~/.local/bin?" y \
        || return 1
    if command -v curl >/dev/null; then
        run_logged "install uv" sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' || return 1
    elif command -v wget >/dev/null; then
        run_logged "install uv" sh -c 'wget -qO- https://astral.sh/uv/install.sh | sh' || return 1
    else
        log_error "need curl or wget to install uv"; return 1
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -x "$candidate" ] && { UV="$candidate"; return 0; }
    done
    return 1
}

if [ "$USE_UV" = 1 ] && ! ensure_uv; then
    log_warn "uv unavailable; falling back to a system python$PY_VERSION"
    USE_UV=0
fi

existing="$(venv_python_version)"
if [ -n "$existing" ] && [ "$existing" != "$PY_VERSION" ]; then
    if ask_yes_no ".venv uses Python $existing, not $PY_VERSION. Delete and recreate it?" y; then
        run_logged "remove old .venv" rm -rf "$VENV_DIR"
        existing=""
    else
        log_warn "keeping .venv on Python $existing"
    fi
fi

if [ -n "$existing" ]; then
    log_info "reusing .venv (Python $existing)"
elif [ "$USE_UV" = 1 ]; then
    run_logged "uv python install $PY_VERSION" "$UV" python install "$PY_VERSION" || die "could not install Python $PY_VERSION"
    run_logged "create .venv" "$UV" venv --python "$PY_VERSION" "$VENV_DIR" || die "could not create .venv"
else
    SYS_PY="$(command -v "python$PY_VERSION" || true)"
    [ -n "$SYS_PY" ] || die "python$PY_VERSION not on PATH. Install it (deadsnakes PPA, pyenv, or your distro) or drop --no-uv."
    run_logged "create .venv" "$SYS_PY" -m venv "$VENV_DIR" || die "could not create .venv (Debian/Ubuntu: apt install python$PY_VERSION-venv)"
    run_logged "upgrade pip" "$VENV_PY" -m pip install --upgrade pip || die "pip upgrade failed"
fi

pip_install() {
    if [ "$USE_UV" = 1 ]; then "$UV" pip install --python "$VENV_PY" "$@"
    else "$VENV_PY" -m pip install "$@"; fi
}
pip_uninstall() {
    if [ "$USE_UV" = 1 ]; then "$UV" pip uninstall --python "$VENV_PY" "$@"
    else "$VENV_PY" -m pip uninstall -y "$@"; fi
}

# onnxruntime and onnxruntime-gpu share one import directory, so switching
# flavours must remove both first or the survivor ends up half-deleted.
if [ "$MODE" = gpu ]; then other=onnxruntime; else other=onnxruntime-gpu; fi
if "$VENV_PY" -c "import importlib.metadata as m, sys; m.version(sys.argv[1])" "$other" >/dev/null 2>&1; then
    log_info "switching onnxruntime flavour: removing $other and any existing onnxruntime"
    run_logged "uninstall onnxruntime packages" pip_uninstall onnxruntime onnxruntime-gpu
fi

run_logged "install requirements-$MODE.txt" pip_install -r "requirements-$MODE.txt" \
    || die "package install failed"

# ---------------------------------------------------------------- folders
for d in ../resources/model ../resources/uncensored_vids ../resources/uncensored_pics \
         ../resources/source ../resources/stickers/breasts ../resources/stickers/vulva ../output; do
    if [ ! -d "$d" ]; then mkdir -p "$d" && log_info "created $(cd "$d" && pwd)"; fi
done

# ---------------------------------------------------------------- models
# 320n downloads automatically; 640m and RetinaNet are manual (GitHub only
# serves them to signed-in users). fetch_models.py checks whatever is there
# and moves bad files (e.g. a saved sign-in page) aside.
"$VENV_PY" tools/setup/fetch_models.py --log-level "${LOG_LEVEL:-info}" --log-file "$HARNESS_LOG_FILE" \
    || log_warn "model check reported a problem; see the messages above"

# ---------------------------------------------------------------- betaconfig.gpu_enabled
want_flag=0; [ "$MODE" = gpu ] && want_flag=1
current_flag="$(sed -n 's/^gpu_enabled[[:space:]]*=[[:space:]]*\([01]\).*/\1/p' betaconfig.py | head -n1)"
if [ -n "$current_flag" ] && [ "$current_flag" != "$want_flag" ]; then
    if ask_yes_no "betaconfig.py has gpu_enabled = $current_flag but this is a $MODE install. Set it to $want_flag?" y; then
        sed -i.bak "s/^gpu_enabled\([[:space:]]*=[[:space:]]*\)[01]/gpu_enabled\1$want_flag/" betaconfig.py \
            && rm -f betaconfig.py.bak
        log_info "betaconfig.py: gpu_enabled = $want_flag"
    else
        log_warn "left gpu_enabled = $current_flag in betaconfig.py"
    fi
fi

# ---------------------------------------------------------------- verify
log_info "=== verifying environment ==="
"$VENV_PY" tools/setup/verify_env.py "--$MODE" --log-level "${LOG_LEVEL:-info}" --log-file "$HARNESS_LOG_FILE"
VERIFY_STATUS=$?

if [ -z "$RUN_TESTS" ]; then
    ask_yes_no "Run the unit tests now (about 15 seconds)?" n && RUN_TESTS=1 || RUN_TESTS=0
fi
if [ "$RUN_TESTS" = 1 ]; then
    run_logged "unit tests" "$VENV_PY" -m unittest discover -s tests -t . -p 'test_*.py' \
        || log_warn "some tests failed; see $HARNESS_LOG_FILE"
fi

# ---------------------------------------------------------------- done
if [ "$VERIFY_STATUS" -ne 0 ]; then
    log_error "setup finished with problems (see FAIL lines above and $HARNESS_LOG_FILE)"
else
    log_info "setup complete"
fi
cat <<EOF

Next steps:
  source .venv/bin/activate          # once per terminal
  # put videos in $(cd ../resources/uncensored_vids && pwd)
  python3 betatv.py --preview on --preview-seconds 20

A script cannot activate a venv in the shell that launched it, so the
'source' line above is yours to run. Alternatively call .venv/bin/python
directly.
EOF
exit "$VERIFY_STATUS"
