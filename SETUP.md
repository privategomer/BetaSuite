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
├── Betasuite/                      this repository; always run commands from here
│   └── .venv/                      Python virtual environment (created by setup)
├── resources/
│   ├── model/                      detector .onnx files
│   │   ├── v3.4-320n.onnx          nudenet_v3 320n (default backend)
│   │   ├── v3.4-640m.onnx          nudenet_v3 640m (optional)
│   │   └── detector_v2_default_checkpoint.onnx   retinanet_v2 (optional)
│   ├── uncensored_vids/            videos to censor (betatv.py input)
│   ├── uncensored_pics/            images to censor (betastare.py input)
│   ├── source/                     optional archive of source footage
│   └── stickers/
│       ├── breasts/                .png files for sticker styles
│       └── vulva/
└── output/                         censored output, caches, stats, logs (created on first run)
```

Because the paths are relative to the working directory, running
`python3 Betasuite/betatv.py` from `BetaSuiteHome/` will **not** work.
`cd` into the repository first.

The sticker folders ship empty. Until you add `.png` files to them,
sticker styles render as solid bars.

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

### Linux

```bash
mkdir -p ~/BetaSuiteHome && cd ~/BetaSuiteHome
git clone https://github.com/privategomer/Betasuite.git
cd Betasuite
./setup.sh
```

### Windows

```powershell
mkdir $HOME\BetaSuiteHome; cd $HOME\BetaSuiteHome
git clone https://github.com/privategomer/Betasuite.git
cd Betasuite
.\setup.cmd
```

Or double-click `setup.cmd` in Explorer. It launches `setup.ps1` with
the execution policy bypassed for that one run only.

### What the script does

1. Detects an NVIDIA GPU with `nvidia-smi` and asks whether to install
   the GPU or CPU build (suggesting GPU when one is found)
2. Checks for ffmpeg/ffprobe and offers to install them (`apt`, `dnf`,
   `pacman`, `zypper` or `brew` on Linux, `winget` on Windows)
3. Installs [uv](https://docs.astral.sh/uv/) if it's missing, then uses
   it to install the pinned Python version and create `.venv/` (no admin
   rights needed, and it doesn't touch your system Python)
4. Installs `requirements-gpu.txt` or `requirements-cpu.txt` into `.venv`
5. Creates the `resources/` and `output/` folders beside the repository
6. Downloads the 320n model, checks any models already present, and
   lists the two optional ones you download yourself (see
   [Models](#6-models))
7. Offers to set `gpu_enabled` in `betaconfig.py` to match the build
8. Runs `tools/setup/verify_env.py` and, optionally, the unit tests

Everything goes to `setup.log` in the repository at every log level.
The console shows `info` and above unless you pass a different level.

### Options

| Linux | Windows | Effect |
|---|---|---|
| `--gpu` / `--cpu` | `-Gpu` / `-Cpu` | skip the GPU/CPU question |
| `--python 3.12` | `-Python 3.12` | override `.python-version` |
| | `-Vision` | also install `betavision-*` dependencies |
| `--tests` / `--no-tests` | `-Tests` / `-NoTests` | run the unit tests at the end, or don't |
| `--yes` | `-Yes` | take the default answer to every question (unattended) |
| `--log-level debug` | `-LogLevel debug` | console verbosity; `setup.log` always gets everything |
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
| `requirements-dev.txt` | semver, for `tools/release/release.py` (maintainers only) |

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

A logged-out download saves GitHub's sign-in page under the `.onnx` name,
which then fails to load with `INVALID_PROTOBUF`. `fetch_models.py`
spots that and moves the file to `<name>.bad`.

Only the model for the backend selected in `betaconfig.py`
(`detector_backend['selected']`) is required. The benchmarking and
comparison tools cover whichever ones are present.

### 7. betaconfig.py

Set `gpu_enabled` to match what you installed:

```python
gpu_enabled = 1     # GPU build
gpu_enabled = 0     # CPU build
```

Leaving it at `1` on a CPU install still runs (a CPU fallback is always
added) but logs a warning on every run.

### 8. Verify

See the next section.

## Verifying an install

```bash
python3 tools/setup/verify_env.py          # reads gpu_enabled from betaconfig.py
python3 tools/setup/verify_env.py --gpu    # insist on CUDA
python3 tools/setup/verify_env.py --cpu
```

It checks the Python version, that exactly one onnxruntime flavour is
installed, ffmpeg/ffprobe, the folder layout, and that each model
present actually loads, on CUDA when GPU is requested. It exits
non-zero if anything fails. Full detail goes to `verify_env.log`.

A healthy GPU install ends with lines like:

```
OK    model nudenet_v3 / 320n loads (CUDAExecutionProvider, 12.2 MB)
verify finished: 0 error(s), 0 warning(s)
```

`python3 betatv.py --version` prints the version you installed.

Then the unit tests (about 15 seconds, no model or GPU needed):

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

`betatest.py` is left over from upstream (Windows only, old model path)
and isn't part of setup or verification.

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

**Anything else.** `setup.log` has the full output of every command the
script ran. Re-run with `--log-level debug` (`-LogLevel debug`) to see
more on the console.
