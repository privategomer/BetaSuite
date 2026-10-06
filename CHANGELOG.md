# Changelog

This fork's releases. Versions follow [semantic versioning](https://semver.org/).

## Unreleased

### Added

- Source folder convention: `profile_by_source_dir` maps a folder under `resources/uncensored_vids/` to a structure profile, so the folder a video sits in decides how it is censored. Ships with `scene/`, `compilation/`, `quick-cut/` and `split-screen/`. Deepest match wins, so a child folder overrides its parent; files in the root or in an unmapped folder are selected by measurement as before, which keeps the convention additive. Adding a folder is adding a line. Validated strictly, since every mistake here is a silent one: a key naming no variant, an empty key, one that normalises away to nothing, a `..`, a pasted absolute path, or two spellings of the same folder are all hard errors
- `compilation` profile, a fourth variant for sequential compilations where the subject changes every few seconds. Between `scene` and `quick_cut` on every axis. `manual_only`, because `short_shot_fraction` only counts sub-second shots and so cannot tell a compilation from a single scene
- `tools/analysis/classify_sources.sh` and `classify_source_structure.py`: measure every source video's shot structure and report which folder it belongs in. Scan only, no detection or rendering, reuses the app's shot-cut cache, and moves nothing. Never recommends split-screen, which no pre-detection signal can identify; flags files whose names hint at one instead. Reads the quick-cut boundary from `betaconfig` so it cannot contradict the app
- The run log now names the rule that chose a profile, not just the profile: `structure profile: split_screen via source folder split-screen/ (...)`. Without it a folder key matching nothing on disk looks configured while quietly falling through to measurement

### Changed

- `profile_by_source_dir` is part of the censor key, so retuning the map does not serve back a render made under the old one
- `classify_sources.sh` resolves `.venv/bin/python` itself instead of calling bare `python3` and relying on an activated venv, so it works activated, unactivated, or under `sudo`. It also checks that `cv2` imports before starting a scan, and says what to do about it, rather than failing partway in with a bare `ModuleNotFoundError`
- `classify_sources.sh` uses `#!/bin/bash` rather than `#!/usr/bin/env bash`, which fails on a system where `/usr/bin/env` is a uutils-coreutils multicall binary. The repo's other nine shell scripts still use the `env` form

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
