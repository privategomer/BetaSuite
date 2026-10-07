# Changelog

This fork's releases. Versions follow [semantic versioning](https://semver.org/).

## 1.3.0 (2026-10-06)

### Added

- `tools/bench/structure_survey.sh`: one command that measures every source file in `resources/uncensored_vids/` and the staging tree `resources/source/` and writes a reviewable `mv -n` script sorting them into profile folders. Three phases: whole-file shot-cut scans on the CPU, detection-only geometry slices on the GPU (asks first; runs when unattended), then the table and the script. Logs everything to `../output/benchmarks/structure_survey_<timestamp>/`, terminal filtered by `LOG_LEVEL`
- `betabench structure-scan`: whole-file shot-cut scans for every source file not yet scanned, files in parallel (one worker per core, at most 4; 2 workers ran 1.36x faster than 1 on a 2-core box). Written to the exact cache betatv's full run reads, so the real run finds its scan done. `--geometry-slices N` then runs detection without rendering on N 20-second slices spread through each file (25/50/75% for 3, never the start), which is what split-screen is recognised from
- With a declared profile (`--profile`, a path pattern or a source folder) betatv runs the shot-cut scan on a background thread alongside detection instead of in front of it. Same function, same cache, same cuts reach tracking; only the timing changes. The scan took 4m38s ahead of detection on a 35-minute file. A finished file with a declared profile also skips without scanning
- `betabench concurrency`: measures on your own machine what running the shot-cut scan alongside detection saves (scan alone, detection alone, both at once, after a warm-up, median of `--trials`), and structure-scan throughput at 1, 2, 4 and more workers with a recommended `--workers`. Every timing starts from an empty throwaway cache; nothing real is read or written. Worker counts above the number of files are reported as unmeasured, not timed (a pool of 8 given 3 files runs 3 processes, so the first version recommended 8 on noise), and the file count defaults to two per worker at the largest count
- After that background scan the run log says what measurement alone would have chosen, as a warning when that profile samples more densely than the declared one (a file in `compilation/` measuring as `quick_cut`: 9 fps instead of 18 means fewer detections on the shortest shots)

### Changed

- `covered_vulva` is promoted to `exposed_vulva` only when an `exposed_penis` is in frame. `exposed_vulva` and `exposed_anus` were evidence too, and on a 35-minute multi-subject file they drove 6,465 of 6,534 promotions, median about 7 box-sizes from the box they promoted, 98.5% touching no evidence at all: an exposed vulva in another pane or on another person was promoting a clothed crotch. Vulva censor boxes on that file fall 30% and their area 37%, while the moments carrying a vulva censor barely move. The censor key changes, so outputs re-render once; detection caches are untouched
- `betabench structure` works per source file, enumerated from disk across both trees, instead of per cache of one detection configuration. New columns say how much footage the timing rests on and whether it was the whole file or a slice, and how many sampled frames the geometry rests on. A file never scanned shows `?` and is never moved
- `--write-move-script` only scripts moves that pass every safety rule: whole-file timing evidence, never lowering the sample rate, never overriding a `manual_only` folder on timing alone unless the move raises the sample rate, and only from the tree root or a mapped folder. The rest are listed with the rule that held them back. Moves stay inside the tree their file is in
- The tuning replay (`auto_tune`) selects each video's profile the way a real run does: from the cached shot-cut scan through the same selector, with the folder convention applied, cuts passed to tracking, and the replayed detections' own sample rate
- File hashes are also memoised by inode, so moving a file (sorting, staging into `uncensored_vids/`, archiving to `completed/`) no longer re-reads it in full. One survey spent 4m39s re-hashing ~70 files after they were sorted. Analysis tools now persist the memo too; only betatv and betastare did, so every analyzer re-hashed archived files on every run
- `betabench` exits non-zero when a subcommand fails, and its log file defaults to `trace`. A harness wrapping it reported a failed phase as finished
- `betaconfig.py` comments cut to one-line headers; the rationale they carried is in CONFIG_REFERENCE.md. Every value is unchanged, so no cache key moves
- The shot-cut scan responds to Ctrl-C

### Fixed

- `tools/lib/python_env.sh` was excluded from git by the template `.gitignore`'s unanchored `lib/` rule, so on a fresh clone of 1.2.0 every shell script failed on its first line. The rule is anchored to the repo root and a test fails if any sourced file is missing or ignored
- A shot-cut scan that found zero cuts was treated the same as no scan at all: it logged `configured default (no shot-cut data)` and fell through to the default profile (13 of 56 files in one preview run), and betabench showed `n/a`. Zero cuts now measures as one shot over the whole window. The outcome only matched before because the shipped default is `scene`
- `betabench structure` divided by the time between the first and last censored detection instead of the footage scanned. A file whose boxes fell in one sampled frame scored 600,000,000 cuts a minute and was recommended for `quick-cut/`. It now uses the same window betatv selects with
- `betabench structure` never saw files sampled at a non-global rate: 8 of 56 in one survey, exactly the files the app had measured as `quick_cut`
- Preview slices at a non-zero offset handed tracking cut times on the scan's own timeline, so every cut fell before the slice's first detection and tracking matched straight across cuts. Full runs and offset-0 previews were unaffected. Existing offset-preview outputs are not re-rendered automatically, since no key changed
- A preview window running past the end of a short file counted the missing seconds in the structure denominator
- The test suite wrote into the real `../output/`: ~230 lines per run into a live `betasuite.log` and ~48 run-key records where betabench discovers configurations. Tests now get a throwaway output root, and a test fails if a new output path is added without it

## 1.2.0 (2026-10-06)

### Added

- Source folder convention: `profile_by_source_dir` maps a folder under `resources/uncensored_vids/` to a structure profile, so the folder a video sits in decides how it is censored. Ships with `scene/`, `compilation/`, `quick-cut/` and `split-screen/`. Deepest match wins, so a child folder overrides its parent; files in the root or in an unmapped folder are selected by measurement as before, which keeps the convention additive. Adding a folder is adding a line. Validated strictly, since every mistake here is a silent one: a key naming no variant, an empty key, one that normalises away to nothing, a `..`, a pasted absolute path, or two spellings of the same folder are all hard errors
- `compilation` profile, a fourth variant for sequential compilations where the subject changes every few seconds. Between `scene` and `quick_cut` on every axis. `manual_only`, because `short_shot_fraction` only counts sub-second shots and so cannot tell a compilation from a single scene
- `betabench structure` now recommends which source folder each file belongs in, alongside the measurements it already reported. It can identify split-screen, which no timing measure can: the spatial signals it already computes (simultaneous boxes, left-half share) are what separate panes from a quiet single scene. The quick-cut boundary is read from `betaconfig` so the recommendation cannot contradict what the app would select; the compilation and split-screen boundaries belong to the bench and are advisory, both read off small samples
- `betabench structure --write-move-script PATH` writes a reviewable shell script that sorts the source tree. Every move is `mv -n`, which never overwrites; each carries its measured reason as a comment; paths are shell-quoted; and it is safe to re-run. betabench never moves a file itself
- Duration-weighted shot shares in `betabench structure`: the share of RUNTIME in sub-1s and sub-10s shots, which is what the folder recommendation uses. A median shot length ranks real footage backwards, reporting a scene with one flurry of false cuts as the fastest-cutting file in a corpus
- `tools/lib/python_env.sh`: resolves the Python this repo should run, preferring its own `.venv` over whatever is on PATH, with `python_env_require` for a readable message when a dependency is missing
- The run log now names the rule that chose a profile, not just the profile: `structure profile: split_screen via source folder split-screen/ (...)`. Without it a folder key matching nothing on disk looks configured while quietly falling through to measurement

### Changed

- `profile_by_source_dir` is part of the censor key, so retuning the map does not serve back a render made under the old one. Because the shipped config sets the map, the censor key changes on upgrade and existing censored output is re-rendered once. Detection caches are unaffected, so the re-render does not re-run the model
- Every shell script uses `#!/bin/bash` rather than `#!/usr/bin/env bash`. On a system where `/usr/bin/env` is a uutils-coreutils multicall binary, running one through `env` failed with "Security violation: Requested utility ... does not match executable name"
- Every shell script resolves Python through `tools/lib/python_env.sh` instead of calling bare `python3` and relying on an activated venv. Bare `python3` silently runs the system interpreter under `sudo` or from a shell that never activated the venv, and silently runs a DIFFERENT project's packages when another venv is active

### Fixed

- `tune_nudenet_suppression.sh` looked for a venv at `../bin/activate`, the layout from before the venv moved inside the repo, so on any current checkout it always took the "no venv found" branch and ran against whatever `python3` was on PATH
- Quoting in `run_all_analysis.sh`'s step list, where an interpolated interpreter path would have broken the command string if it contained a space

## 1.1.0 (2026-10-03)

### Added

- `SETUP.md`: full setup guide for Linux and Windows, scripted and manual, plus migration notes for upstream 0.2.4 users
- `setup.sh` (Linux) and `setup.cmd` / `setup.ps1` (Windows): one-shot, re-runnable setup. Installs the pinned Python with uv, creates `.venv`, installs the GPU or CPU build, creates the `resources/` and `output/` folders, downloads models, aligns `gpu_enabled`, and verifies the result. Logs everything to `setup.log`
- `tools/setup/fetch_models.py`: downloads the 320n model from the hash-pinned `nudenet` package on PyPI and checks every model present; anything that isn't a real ONNX file (such as a saved GitHub sign-in page) is moved aside instead of trusted. 640m and RetinaNet are manual downloads (GitHub serves them only to signed-in users), with direct links in `SETUP.md`
- `tools/setup/verify_env.py`: checks Python, packages, ffmpeg, folders, and that each model loads on the expected execution provider
- Versioning: a `VERSION` file, `--version` on every entry point, and the version in BetaTV's startup log line and in each stats row (`betasuite_version`). Checkouts that aren't exactly on a release tag report semver build metadata, e.g. `1.1.0+3.g1a2b3c4.dirty`
- `tools/release/release.py`: suggests a version bump from the commits since the last tag, bumps `VERSION` with the `semver` package, dates the CHANGELOG entry, runs the tests, commits, tags, and optionally pushes and creates the GitHub release
- `requirements.txt`, `requirements-cpu.txt`, `requirements-gpu.txt`, `requirements-vision-windows.txt`, `requirements-dev.txt`, and `.python-version` (3.12)

### Fixed

- GPU runs fell back to CPU when CUDA and cuDNN came from the pip `nvidia-*` packages instead of a system install (`libcublasLt.so.12` not found). Sessions now call `onnxruntime.preload_dlls()` first

### Changed

- `onnxruntime-gpu` is capped below 1.27 so installs stay on CUDA 12, matching `gpu-requirements.txt` and keeping Pascal-era GPUs supported
- Comments and docs no longer reference pre-release internal version numbers
- README points to `SETUP.md` for installation

## 1.0.0 (2026-09-30)

First public release of this fork of [solarorb93/BetaSuite](https://github.com/solarorb93/BetaSuite) 0.2.4, with performance and efficacy work throughout. See the README for what's included.
