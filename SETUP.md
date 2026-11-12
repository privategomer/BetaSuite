# Setting up BetaSuite

This covers getting a working BetaSuite environment on Linux or Windows,
either with the setup script or by hand. Once it's done, the
[README](README.md) covers running and tuning.

- [What you end up with](#what-you-end-up-with)
- [Requirements](#requirements)
- [Quick setup (script)](#quick-setup-script)
- [Manual setup](#manual-setup)
- [Verifying an install](#verifying-an-install)
- [Coming from upstream BetaSuite 0.2.4](#coming-from-upstream-betasuite-024)
- [Troubleshooting setup](#troubleshooting-setup)

---

## What you end up with

BetaSuite resolves its data folders relative to the folder you run it
from, so the repository sits **beside** a `resources/` and an `output/`
folder, not inside them:

```
BetaSuiteHome/                      any name, any location
├── BetaSuite/                      this repository; always run commands from here
│   ├── .venv/                      Python virtual environment (created by setup)
│   └── betaconfig_local.py         your own settings, ignored by git (created by setup)
├── resources/
│   ├── model/                      detector and person-segmentation .onnx files
│   │   ├── v3.4-320n.onnx          nudenet_v3 320n (default backend)
│   │   ├── v3.4-640m.onnx          nudenet_v3 640m (optional)
│   │   ├── detector_v2_default_checkpoint.onnx   retinanet_v2 (optional)
│   │   ├── yolo11s-seg.onnx        person masks, default size (reverse mode; exported by setup)
│   │   ├── yolo11n-seg.onnx        person masks, smaller (optional; exported by setup)
│   │   ├── yolo11m-seg.onnx        person masks, larger (optional; exported by setup)
│   │   └── rf-detr-seg-*.onnx      person masks, RF-DETR-Seg nano/small/medium
│   │                               (optional; exported by setup if you choose rf_detr_seg)
│   ├── uncensored_vids/            videos to censor (betatv.py input)
│   │   └── scene/, compilation/, quick-cut/, split-screen/
│   │                               optional: the folder a video sits in picks its tuned profile
│   ├── uncensored_pics/            images to censor (betastare.py input)
│   ├── source/                     optional archive of source footage
│   └── stickers/
│       ├── breasts/                .png files for sticker styles
│       └── vulva/
└── output/                         censored output, caches, stats, logs (created on first run)
```

Because the paths are relative to the working directory, running
`python3 BetaSuite/betatv.py` from `BetaSuiteHome/` will **not** work.
`cd` into the repository first.

The sticker folders ship empty, and the shipped styles use no stickers:
BetaSuite has no licence to distribute sticker images. To use them, add
your own `.png` files, then add sticker entries to a label's
`censor_style` in `betaconfig_local.py` (CONFIG_REFERENCE.md,
`type: 'sticker'`). A sticker style with an empty folder renders as a
solid bar.

The `yolo11*-seg.onnx` and `rf-detr-seg-*.onnx` files are
person-segmentation models. Only reverse mode uses them
(`reverse['enabled']`, off by default), for its `person_windows`, so a
normal run does not need them. With the shipped config only
`yolo11s-seg.onnx` is used. The other YOLO11 sizes (`n`, `m`) and the
RF-DETR-Seg backend (`person_backend['selected'] = 'rf_detr_seg'`, sizes
`nano`, `small`, `medium`) are alternatives you pick in
`person_backend`. Setup asks which person backends to export, one or
more, and exports every size of each one chosen: `yolo11_seg` (the
default) is about 12, 41 and 90 MB, `rf_detr_seg` about 125 MB per
size, so about 143 MB, 375 MB, or about 520 MB for both. Choosing only
`yolo11_seg` also leaves `rfdetr` out of the throwaway export
environment.

The subfolders under `uncensored_vids/` are optional. If you sort videos
into them, `profile_by_source_dir` gives each folder its own tuned
profile (see the README's "The folder convention").

## Requirements

| | Linux | Windows |
|---|---|---|
| OS | any recent x86_64 distro | Windows 10 or 11, x64 |
| Python | 3.12 (pinned in `.python-version`; the script installs it) | same |
| ffmpeg + ffprobe | on `PATH` | on `PATH` |
| GPU (optional, strongly recommended) | NVIDIA, GTX 10-series (Pascal) or newer, driver 525+ | same, driver 528+ |
| Disk | about 3 GB for the GPU environment, plus models and footage | same |

**No CUDA toolkit or cuDNN install is needed.** The GPU build pulls CUDA
12 and cuDNN 9 runtime libraries in as Python packages
(`gpu-requirements.txt`). All you need system-wide is the NVIDIA driver.
Check it with `nvidia-smi`.

CPU-only works, but expect detection to run something like 50-100x
slower than on a GPU.

`betavision-*.py` (live screen censoring) is Windows only, since it
uses `pywin32`. Everything else runs on both.

## Quick setup (script)

Clone the repository into the folder that will hold `resources/` and
`output/`, then run the script for your platform. Every step checks
before it acts, so it's safe to re-run to repair or update an install.

A clone starts on `main`, which can hold changes that aren't released
yet. The script's first step fetches the release tags and offers to
switch the checkout to the newest release (`vX.Y.Z`), then carries on
from that release's own copy of the script. Answer no, or pass
`--no-release-check` (`-NoReleaseCheck` on Windows), to install `main`
as it is. If you prefer no git at all, download the source zip from the
[latest release](https://github.com/privategomer/BetaSuite/releases/latest)
instead of cloning; setup works the same from it.

### Linux

```bash
mkdir -p ~/BetaSuiteHome && cd ~/BetaSuiteHome
git clone https://github.com/privategomer/BetaSuite.git
cd BetaSuite
./setup.sh
```

### Windows

```powershell
mkdir $HOME\BetaSuiteHome; cd $HOME\BetaSuiteHome
git clone https://github.com/privategomer/BetaSuite.git
cd BetaSuite
.\setup.cmd
```

Or double-click `setup.cmd` in Explorer. It launches `setup.ps1` with
the execution policy bypassed for that one run only.

### Updating

Run the setup script again. It fetches tags, offers to move to a newer
release if there is one, and brings the environment in line with that
release's requirements. Switching to a release tag leaves git in
"detached HEAD" state, which is normal for an install you don't edit.
To go back to development, `git switch main`.

### What the script does

1. In a git checkout, offers to switch to the latest release tag (and
   restarts from that release's script if you say yes)
2. Detects an NVIDIA GPU with `nvidia-smi` and asks whether to install
   the GPU or CPU build (suggesting GPU when one is found)
3. Checks for ffmpeg/ffprobe and offers to install them (`apt`, `dnf`,
   `pacman`, `zypper` or `brew` on Linux, `winget` on Windows)
4. Installs [uv](https://docs.astral.sh/uv/) if it's missing, then uses
   it to install the pinned Python version and create `.venv/` (no admin
   rights needed, and it doesn't touch your system Python)
5. Installs `requirements-gpu.txt` or `requirements-cpu.txt` into `.venv`
6. Creates the `resources/` and `output/` folders beside the repository
7. Downloads the 320n model, asks which person backends to export
   (`yolo11_seg` by default, `rf_detr_seg`, both or neither; for reverse
   mode's person windows) and exports every size of each, checks any
   models already present,
   and lists the two optional ones you download yourself (see
   [Models](#6-models))
8. Creates your local config, `betaconfig_local.py`, if it is missing
   (a header and nothing else), and sets `gpu_enabled` in it when the
   build does not match `betaconfig.py`'s value (asking first only if
   that file already says otherwise)
9. Runs `tools/setup/verify_env.py` and, optionally, the unit tests

After setup, choose what to censor. A fresh install censors nothing
(`items_to_censor` ships empty) and `betatv.py` stops before any
detection until you do:

```bash
python3 tools/setup/local_config.py set "items_to_censor=['exposed_vulva', 'exposed_breast']"
```

CONFIG_REFERENCE.md, "Example local config", has that line and a whole
local config to start from.

Optional, after setup: `tools/bench/tune_machine.sh` measures this
machine's hardware settings (detection batch size, render workers,
encoder and more, 15 to 30 minutes) and offers to write them to
`betaconfig_local.py`. See CONFIG_REFERENCE.md, "Local config".

Everything goes to `setup.log` in the repository at every log level.
The console shows `info` and above unless you pass a different level.

### Options

| Linux | Windows | Effect |
|---|---|---|
| `--gpu` / `--cpu` | `-Gpu` / `-Cpu` | skip the GPU/CPU question |
| `--python 3.12` | `-Python 3.12` | override `.python-version` |
| | `-Vision` | also install `betavision-*` dependencies |
| `--tests` / `--no-tests` | `-Tests` / `-NoTests` | run the unit tests at the end, or don't |
| `--person-backends yolo11_seg,rf_detr_seg` | `-PersonBackends yolo11_seg,rf_detr_seg` | skip the person-model question: comma separated backends, `all` or `none` |
| `--yes` | `-Yes` | take the default answer to every question (unattended) |
| `--log-level debug` | `-LogLevel debug` | console verbosity; `setup.log` always gets everything |
| `--no-release-check` | `-NoReleaseCheck` | stay on the current branch or commit instead of offering the latest release |
| `--no-uv` | | Linux only: use an existing `python3.12` and `python -m venv` instead of uv |

Example unattended GPU install that also runs the tests:

```bash
./setup.sh --yes --gpu --tests
```

### After the script

A script can't activate a virtual environment in the shell that
launched it, so activate it yourself, once per terminal:

```bash
source .venv/bin/activate          # Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
.venv\Scripts\activate.bat         # Windows cmd
```

Then drop a video into `../resources/uncensored_vids/` and do a
20-second preview:

```bash
python3 betatv.py --preview on --preview-seconds 20    # Windows: python, not python3
```

## Manual setup

The same steps the script takes. Run everything from inside the
repository folder.

### 0. Use the latest release

Skip this if you downloaded a release zip. In a clone, move to the
newest release tag:

```bash
git fetch --tags
git checkout "$(git tag --list 'v[0-9]*' --sort=-v:refname | grep -v -- - | head -n1)"
```

```powershell
git fetch --tags
git checkout (git tag --list 'v[0-9]*' --sort=-v:refname | Where-Object { $_ -notmatch '-' } | Select-Object -First 1)
```

### 1. NVIDIA driver (GPU only)

Install the current driver from your distro or from NVIDIA, then confirm:

```bash
nvidia-smi
```

The "Driver Version" must be 525+ on Linux or 528+ on Windows. Ignore
the "CUDA Version" it prints; that's the newest CUDA the driver
supports, not something that's installed.

### 2. ffmpeg

BetaSuite calls `ffmpeg` and `ffprobe` from `PATH`.

| Platform | Command |
|---|---|
| Debian / Ubuntu / Mint | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` (or `ffmpeg-free`; RPM Fusion's build has more codecs) |
| Arch | `sudo pacman -S ffmpeg` |
| openSUSE | `sudo zypper install ffmpeg` (Packman repo for full codecs) |
| Windows | `winget install --id Gyan.FFmpeg -e`, then open a **new** terminal |

Windows without winget: download a build from
[ffmpeg.org](https://ffmpeg.org/download.html#build-windows), extract
it, and add its `bin` folder to your user `PATH`. Unlike upstream
0.2.4, an `ffmpeg/` folder next to the repository is **not** picked up.

Check: `ffmpeg -version` and `ffprobe -version` both print a version.

### 3. Python 3.12 and a virtual environment

**With uv (recommended, both platforms):**

```bash
# install uv once
curl -LsSf https://astral.sh/uv/install.sh | sh                          # Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

uv python install 3.12
uv venv --python 3.12 .venv
```

**Without uv:** install Python 3.12 from python.org (Windows: tick "Add
python.exe to PATH") or your distro (Ubuntu may need the deadsnakes PPA
and `python3.12-venv`), then:

```bash
python3.12 -m venv .venv          # Linux
py -3.12 -m venv .venv            # Windows
```

Activate it:

```bash
source .venv/bin/activate          # Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
```

If PowerShell refuses with an execution-policy error, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### 4. Python packages

With the venv active, install **one** of these:

```bash
pip install -r requirements-gpu.txt     # NVIDIA GPU
pip install -r requirements-cpu.txt     # CPU only
```

(With uv, `uv pip install -r ...` does the same, much faster.)

Never install both. `onnxruntime` and `onnxruntime-gpu` write into the
same `onnxruntime/` package folder, so installing one over the other
leaves a broken mix. To switch, uninstall both first:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -r requirements-cpu.txt
```

Windows, for `betavision-*` only:

```powershell
pip install -r requirements-vision-windows.txt
```

| File | Contents |
|---|---|
| `requirements.txt` | numpy, opencv-python (shared) |
| `requirements-cpu.txt` | shared + `onnxruntime` |
| `requirements-gpu.txt` | shared + `onnxruntime-gpu` (CUDA 12 build) + `gpu-requirements.txt` |
| `gpu-requirements.txt` | CUDA 12.9 / cuDNN 9.10 runtime libraries as pip wheels |
| `requirements-vision-windows.txt` | mss, pywin32 |
| `requirements-dev.txt` | semver for `tools/release/release.py`; memray and py-spy for `tools/bench/profile.sh` (maintainers only) |

`onnxruntime-gpu` is capped below 1.27 on purpose. From 1.27 on, the
default PyPI build targets CUDA 13, which doesn't match the CUDA 12
libraries in `gpu-requirements.txt`, and the result is a run that
silently lands on the CPU. CUDA 13 also drops Maxwell, Pascal and Volta
GPUs (GTX 900 and 10-series, Titan V), so staying on CUDA 12 is what
keeps those cards working.

### 5. Folders

From the repository folder:

```bash
mkdir -p ../resources/{model,uncensored_vids,uncensored_pics,source,stickers/breasts,stickers/vulva} ../output
```

```powershell
'model','uncensored_vids','uncensored_pics','source','stickers\breasts','stickers\vulva' |
  ForEach-Object { New-Item -ItemType Directory -Force "..\resources\$_" } | Out-Null
New-Item -ItemType Directory -Force ..\output | Out-Null
```

### 6. Models

Models go in `../resources/model/` under the exact names below. The
detector adapters look for these names, so rename the downloads.

| Model | Backend | Download | Save as |
|---|---|---|---|
| 320n (default, required) | `nudenet_v3` | automatic: `python3 tools/setup/fetch_models.py` | `v3.4-320n.onnx` |
| 640m (optional) | `nudenet_v3`, `--variant 640m` | [640m.onnx](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx) | `v3.4-640m.onnx` |
| RetinaNet (optional) | `retinanet_v2` | [detector_v2_default_checkpoint.onnx](https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_checkpoint.onnx) | `detector_v2_default_checkpoint.onnx` |
| YOLO11n/s/m-seg (person windows) | `person_backend` `yolo11_seg` | automatic, the default choice: exported by `fetch_models.py` | `yolo11n-seg.onnx`, `yolo11s-seg.onnx`, `yolo11m-seg.onnx` |
| RF-DETR-Seg nano/small/medium (person windows) | `person_backend` `rf_detr_seg` | when chosen at setup: exported by `fetch_models.py --person-backends rf_detr_seg` | `rf-detr-seg-nano.onnx`, `rf-detr-seg-small.onnx`, `rf-detr-seg-medium.onnx` |

**320n** is bundled in the [`nudenet` package on PyPI](https://pypi.org/project/nudenet/3.4.2/#files),
so setup downloads it for you and checks it against a pinned hash. By
hand: download the 3.4.2 `.whl`, open it as a zip, and copy out
`nudenet/320n.onnx`.

**640m and RetinaNet you download yourself.** GitHub shows NudeNet's
repository behind a content warning ("This project may contain offensive
or objectionable content"), and only signed-in users can accept it:

1. Sign in to GitHub in your browser
2. Click the download link above. If the warning page appears, accept it
   and click the link again
3. Save the file into `../resources/model/` under the "Save as" name
4. Run `python3 tools/setup/fetch_models.py` (or re-run setup) to check it

With the [GitHub CLI](https://cli.github.com/) signed in (`gh auth login`)
you can skip the browser:

```bash
gh release download v3.4-weights -R notAI-tech/NudeNet -p 640m.onnx -O ../resources/model/v3.4-640m.onnx
gh release download v0 -R notAI-tech/NudeNet -p detector_v2_default_checkpoint.onnx -D ../resources/model/
```

**The person models are exported, not downloaded as ONNX.** They are for
reverse mode's person windows and not needed otherwise. Ultralytics
publishes YOLO11-seg (AGPL-3.0) as PyTorch `.pt` files and Roboflow
publishes RF-DETR-Seg (Apache-2.0, every size) as PyTorch `.pth` files;
BetaSuite redistributes neither. Setup downloads each file, checks it
against a pinned hash, and exports it to ONNX in one throwaway
environment (ultralytics, rfdetr and CPU-only PyTorch, a few hundred MB,
fetched once into uv's cache) that is deleted afterwards. The export
itself runs offline, from the checked file only. BetaSuite's own `.venv`
never gets PyTorch: the models run through onnxruntime like the
detectors. The exports take a few minutes once the packages are
downloaded, and need network access to github.com, repo.roboflow.com,
PyPI and download.pytorch.org (set `BETASUITE_EXPORT_TORCH_INDEX` to use
a mirror of the last).

An RF-DETR-Seg export has a fixed picture size, its own for each size
(312, 384 and 432 pixels square for nano, small and medium); only the
batch size is left free. Each frame is stretched to
that square, the way Roboflow trains and runs the model.

Setup exports only the person backends you choose. At the question,
type numbers or names (`1`, `2`, `1,2`, `rf_detr_seg`), `all` or
`none`; Enter takes the default, `yolo11_seg` plus any backend that
already has some of its files (a half-finished export is finished). With
`--yes` (`-Yes`), or without a terminal, setup takes that default
without asking. The question only appears when a model is missing, so
re-running setup after a complete install does not ask. To add a
backend later, run `python3 tools/setup/fetch_models.py --person-backends
rf_detr_seg` (or re-run setup and choose it); models already present are
only checked, whatever you choose, and nothing is deleted.
`python3 tools/setup/fetch_models.py --no-export` checks the models
without exporting. A failed export is
logged as an error and setup carries on, so the rest of the install
still completes; re-run setup or `fetch_models.py` to retry. If reverse
mode has `person_windows` on and the selected size is missing, the
config check refuses to start and names the missing file. To run reverse
mode without person windows, set `reverse['person_windows']['enabled'] = False`.

A logged-out download saves GitHub's sign-in page under the `.onnx` name,
which then fails to load with `INVALID_PROTOBUF`. `fetch_models.py`
spots that and moves the file to `<name>.bad`.

Only the model for the backend selected in `betaconfig.py`
(`detector_backend['selected']`) is required, plus the selected person
model (`person_backend['selected']`, size `model_variant`) if you use
reverse mode with person windows. The benchmarking and comparison tools
cover whichever ones are present: `python3 tools/bench/betabench.py
persons` times every person backend and size on your own footage.

### 7. Your local config

`betaconfig.py` holds the shipped defaults; never edit it. Your own
settings go in `betaconfig_local.py` beside it, which git ignores: copy
a setting in to change it, delete the line to go back. Setup creates it
and, on a GPU build, sets `gpu_enabled` (the shipped value is `0`). By
hand:

```bash
python3 tools/setup/local_config.py init
python3 tools/setup/local_config.py set gpu_enabled=1     # GPU build
python3 tools/setup/local_config.py                       # every setting, and where it came from
```

Leaving it at `1` on a CPU install still runs (a CPU fallback is always
added) but logs a warning on every run. `tools/bench/tune_machine.sh`
proposes `gpu_enabled` from what is installed and measures the rest of
this machine's hardware settings; see
CONFIG_REFERENCE.md, "Local config".

If you used an earlier version, your hardware settings may be in
`../resources/betaconfig_machine.py`, which is no longer read. Every
run warns until you move them:

```bash
python3 tools/setup/local_config.py migrate
```

### 8. Verify

See the next section.

## Verifying an install

```bash
python3 tools/setup/verify_env.py          # reads gpu_enabled (betaconfig_local.py, else betaconfig.py)
python3 tools/setup/verify_env.py --gpu    # insist on CUDA
python3 tools/setup/verify_env.py --cpu
```

It checks the Python version, that exactly one onnxruntime flavour is
installed, ffmpeg/ffprobe, the folder layout, and that each model
present (detectors and person models) actually loads, on CUDA when GPU
is requested. It exits
non-zero if anything fails. Full detail goes to `verify_env.log`.

A healthy GPU install ends with lines like:

```
OK    model nudenet_v3 / 320n loads (CUDAExecutionProvider, 12.2 MB)
OK    model person yolo11_seg / s loads (CUDAExecutionProvider, 41.4 MB)
verify finished: 0 error(s), 0 warning(s)
```

`python3 betatv.py --version` prints the version you installed.

Then the unit tests (a minute or two, no model or GPU needed):

```bash
./tests/run_tests.sh                                                   # Linux
python -m unittest discover -s tests -t . -p "test_*.py" -v           # Windows
```

## Coming from upstream BetaSuite 0.2.4

If you followed the [upstream instructions](https://github.com/solarorb93/BetaSuite),
these are the differences that matter:

| | Upstream 0.2.4 | This fork |
|---|---|---|
| Python | 3.9 | 3.12 |
| CUDA / cuDNN | 11.4 / 8.2.2, installed system-wide from NVIDIA | 12 / 9, installed into the venv by pip |
| Platforms | Windows | Linux and Windows (`betavision` still Windows only) |
| ffmpeg | `InstallFolder/ffmpeg/` | anywhere on `PATH` |
| Model folder | `InstallFolder/model/` | `resources/model/` |
| Default model | RetinaNet v2 | NudeNet v3.4 320n (RetinaNet still supported) |
| Input folders | `InstallFolder/uncensored_vids/` etc. | `resources/uncensored_vids/` etc. |
| Output and hashes | `censored_vids/`, `vid_hashes/` etc. | all under `output/` |

To migrate: move your old `model/detector_v2_default_checkpoint.onnx`
into `resources/model/`, move `uncensored_*` folders into `resources/`,
and run setup. Old hash/cache folders aren't reused; the cache format
changed.

You can uninstall CUDA 11.4 and cuDNN 8 afterwards if nothing else on
the machine needs them. They aren't used.

## Troubleshooting setup

**`libcublasLt.so.12` / `cublas64_12.dll` not found, model loads on
CPU.** The CUDA libraries come from the `nvidia-*` pip packages, and
onnxruntime only finds those after `onnxruntime.preload_dlls()`.
BetaSuite and `verify_env.py` call it on GPU installs, so if you still see
this, check that `gpu-requirements.txt` installed cleanly (re-run setup)
and that `onnxruntime-gpu` is 1.21 or newer.

**`GPU requested but this onnxruntime build has no CUDAExecutionProvider`.**
The CPU package is installed, or both are and the CPU one won. Uninstall
both and reinstall `requirements-gpu.txt`.

**`libcudnn*.so` / `cublas64_12.dll` not found, or a CUDA version
mismatch in the log.** Usually an `onnxruntime-gpu` that targets a
different CUDA than the `nvidia-*` wheels, or an old driver. Re-run
setup, which reinstalls from the pinned files, and check `nvidia-smi`.

**`ffmpeg` installed on Windows but still "not found".** winget updated
`PATH` for new terminals only. Close the window and open a new one.

**`python3` opens the Microsoft Store on Windows.** Use `python` inside
an activated venv, or `.venv\Scripts\python.exe` directly.

**`./setup.sh: Permission denied`.** The executable bit got lost in a
copy. Run `bash setup.sh`, or `chmod +x setup.sh`.

**`INVALID_PROTOBUF` / "Protobuf parsing failed" loading a model.** The
file is almost always a saved web page from a redirected download.
`python3 tools/setup/fetch_models.py` moves it aside and fetches it
again; for 640m or RetinaNet, download it in a browser if that fails too.

**`reverse['person_windows'] is on but the person model ... is missing`.**
The person-model export didn't finish or was skipped. Re-run setup (or
`python3 tools/setup/fetch_models.py`) and read `setup.log`, or
`fetch_models.log` when run by hand, for the step that failed. The
export needs network access to github.com, repo.roboflow.com, PyPI and
download.pytorch.org; behind a mirror, set `BETASUITE_EXPORT_TORCH_INDEX`. Reverse mode also
runs without person windows if you set `reverse['person_windows']['enabled'] = False`.

**Anything else.** `setup.log` has the full output of every command the
script ran. Re-run with `--log-level debug` (`-LogLevel debug`) to see
more on the console.
