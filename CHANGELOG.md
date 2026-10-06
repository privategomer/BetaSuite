# Changelog

This fork's releases. Versions follow [semantic versioning](https://semver.org/).

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
