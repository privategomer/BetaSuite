# Changelog

This fork's releases. Versions follow [semantic versioning](https://semver.org/).

## 1.1.0 (2026-10-01)

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
