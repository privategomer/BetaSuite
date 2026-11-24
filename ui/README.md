# BetaSuite desktop app

The desktop UI: a Tauri 2 shell (Rust, `src-tauri/`) around a React and TypeScript page (`src/`). Not shipped yet. These notes are for building and running it from source.

## Prerequisites

- Node 22.13 or later with npm. The Tauri CLI comes with the npm packages.
- Rust 1.90 or later. `src-tauri/rust-toolchain.toml` pins the version CI uses; rustup installs it on its own, and a distro toolchain of the same version works too. `clippy` and `rustfmt` for the checks.
- Linux: the WebKitGTK 4.1 and GTK 3 development packages (`libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libxdo-dev libssl-dev` on Ubuntu), plus GStreamer with an H.264 decoder (`gstreamer1.0-plugins-bad` for hardware decoding, `gstreamer1.0-libav` for software) to play video.
- Windows: the WebView2 runtime (part of Windows 11) and, to build, the MSVC toolchain.

## Build and run

```
cd ui
npm ci
npm run app:build      # release binary in src-tauri/target/release/
npm run app:dev        # debug build, started at once
```

App options go after three `--` in dev (npm, the Tauri CLI and cargo each take one): `npm run app:dev -- -- -- --log-level debug`.

There is no development server. Both commands build the page into `dist/` first, and the app always loads it through its own scheme, so no port is ever opened and video behaves the same in debug and release builds.

## Checks

```
npm run check                                                    # eslint, tsc, catalog check and its self-test
cd src-tauri && cargo fmt --check && cargo clippy --all-targets -- -D warnings
```

CI runs the same, and builds a Windows release binary kept as an artifact for a week.

## The engine

The shell starts the engine as a child process, `python betaengine.py serve`, and speaks the protocol in `protocol/README.md` with it (`src-tauri/src/engine.rs`). It looks for the engine in `BETASUITE_ENGINE_DIR` when that is set, else in the executable's folder or one of its parents, so a binary built in the repository, or copied into a checkout's root, finds it. Python is `PYTHON_BIN` when set, else the checkout's `.venv`, else `python3` or `python` on PATH. Serve mode needs only the standard library. The engine inherits the app's environment, `BETASUITE_DATA_ROOT` included, and runs in the engine's folder.

Closing the window hides it, sends `shutdown`, waits up to 5 s, kills the engine if it is still running, then exits. If the engine ends on its own, the page is told (`exited`, with the exit code or signal) and can restart it. Every message to and from the engine is recorded in `engine-transcript.jsonl` beside the log (the previous run's copy is kept as `.1`), and the engine's stderr goes into the log at debug.

`cargo test` (in `src-tauri/`) runs the supervisor against the real engine and a fake one: handshake, concurrent requests, rejected lines, the line limit, a stderr flood, kills, shutdown, a stuck engine, a missing engine or Python, and restart.

### Running a CI build on Windows

1. Download `betasuite-windows` from the pull request's checks (Actions run, Artifacts) and the branch's source as a zip.
2. Unzip the source, and in that folder run `py -3.12 -m venv .venv` (no packages needed).
3. Put `betasuite.exe` in that folder and start it.

## Text and the message catalog

Every string the app shows comes from the catalogs in the repository's `i18n/<locale>/` folders: flat JSON files of dotted keys to ICU MessageFormat messages, shared with the engine's settings registry. `ui.json` holds the app's own text and `errors.json` the wording for every error code the engine or the shell can send; the engine and shell send codes and parameters, never sentences. In code, `t('key', values)` or `<Msg id="key" values={...} />`; keys built at run time go through `msgKey('prefix.', part)`.

`npm run catalog:check` enforces this (the rules are listed at the top of `scripts/check-catalog.mjs`): no user-visible text in the source, every key used exists with matching values, every error code has a message, catalogs parse, CSS uses logical properties. A line that must hold text anyway takes `// catalog-ignore: <reason>`. `--locale qps-ploc` starts the pseudo-locale: accented, padded text that shows any string that skipped the catalog and any layout that cannot take longer translations.

## Options

```
betasuite [--log-level LEVEL] [--console-level LEVEL] [--log-file PATH] [--locale TAG]
          [--theme light|dark] [--open VIDEO] [--measure-startup FILE]
```

LEVEL is `trace`, `debug`, `info`, `warn` or `error`; both default to `info`. `--locale` picks the UI language (default: the system's; `qps-ploc` for the pseudo-locale). `--theme` forces the light or dark theme (default: follow the system). `--open` plays a video at once, muted. `--measure-startup` writes the wall-clock times of startup steps (process start, page script, window shown, engine ready, first video frame with `--open`) to FILE as JSON, then closes the app.

## Look and video

Light and dark themes follow the system (`src/theme/tokens.css`). Status colors come from the Okabe-Ito palette and each state also has its own icon shape and words. `npm run contrast:check` checks the theme's text and icon colors against WCAG AA in both themes. Layout uses logical CSS properties, so a right-to-left language mirrors it.

"Open video…" opens a native file dialog from the shell, which then allows that one file and no other. On Linux the video plays from a `file://` URL, since WebKitGTK cannot play the asset protocol; on Windows it uses the asset protocol. Linux needs a GStreamer H.264 decoder (see Prerequisites). Unknown options are logged and ignored.

## Logs

`logs/ui.log` in the app's state folder: `$XDG_STATE_HOME/betasuite` (by default `~/.local/state/betasuite`) on Linux, `%LOCALAPPDATA%\BetaSuite\state` on Windows. A log over 10 MiB moves to `ui.log.1` when the app starts. Page records carry the target `page`; the shell's own carry the module name. Other crates are logged at warn and above unless the level is trace.
