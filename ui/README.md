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

## Text and the message catalog

Every string the app shows comes from the catalogs in the repository's `i18n/<locale>/` folders: flat JSON files of dotted keys to ICU MessageFormat messages, shared with the engine's settings registry. `ui.json` holds the app's own text and `errors.json` the wording for every error code the engine or the shell can send; the engine and shell send codes and parameters, never sentences. In code, `t('key', values)` or `<Msg id="key" values={...} />`; keys built at run time go through `msgKey('prefix.', part)`.

`npm run catalog:check` enforces this (the rules are listed at the top of `scripts/check-catalog.mjs`): no user-visible text in the source, every key used exists with matching values, every error code has a message, catalogs parse, CSS uses logical properties. A line that must hold text anyway takes `// catalog-ignore: <reason>`. `--locale qps-ploc` starts the pseudo-locale: accented, padded text that shows any string that skipped the catalog and any layout that cannot take longer translations.

## Options

```
betasuite [--log-level LEVEL] [--console-level LEVEL] [--log-file PATH] [--locale TAG]
```

LEVEL is `trace`, `debug`, `info`, `warn` or `error`; both default to `info`. `--locale` picks the UI language (default: the system's; `qps-ploc` for the pseudo-locale). Unknown options are logged and ignored.

## Logs

`logs/ui.log` in the app's state folder: `$XDG_STATE_HOME/betasuite` (by default `~/.local/state/betasuite`) on Linux, `%LOCALAPPDATA%\BetaSuite\state` on Windows. A log over 10 MiB moves to `ui.log.1` when the app starts. Page records carry the target `page`; the shell's own carry the module name. Other crates are logged at warn and above unless the level is trace.
