# BetaSuite

BetaSuite detects and censors specific body parts in photos and video. A
neural net finds regions of interest; BetaSuite tracks them across
frames and renders an obscuring effect (blur, pixelation, a solid bar,
or a sticker) over each one.

Three entry points, one configuration file, one engine:

| Entry point | What it does |
|---|---|
| `betatv.py` | censors video files |
| `betastare.py` | censors still images |
| `betavision-*.py` | live screen-capture censoring (needs `gpu_enabled`) |

- **`betaconfig.py`** — every setting. Values and one-line headers only.
- **[`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md)** — what each setting
  does, with analogies and worked examples, and why the shipped values
  are what they are.
- **this file** — how to run it, what the architecture is, the complete
  settings catalogue, and how to test and measure.

This assumes BetaSuite is already installed and runnable. For
installation, see **[SETUP.md](SETUP.md)** (setup script or manual steps,
Linux and Windows).

**Original project by [solarorb93](https://github.com/solarorb93/BetaSuite).** This fork is optimized for performance and efficacy.

---

## Contents

- [Quick start](#quick-start)
- [How a run works](#how-a-run-works)
- [Architecture](#architecture)
- [Settings catalogue](#settings-catalogue)
- [Command-line overrides](#command-line-overrides)
- [Output filenames and cache keys](#output-filenames-and-cache-keys)
- [Resumability, interrupts and crash recovery](#resumability-interrupts-and-crash-recovery)
- [Measuring and tuning](#measuring-and-tuning)
- [Testing](#testing)
- [Versions and releases](#versions-and-releases)
- [Performance notes](#performance-notes)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
# censor everything under ../resources/uncensored_vids/
python3 betatv.py

# a fast 20-second preview of each file, to check settings
python3 betatv.py --preview on --preview-seconds 20

# the same, but with the other model
python3 betatv.py --preview on --backend retinanet_v2

# a clean, measured run under both backends, with analysis afterwards
python3 tools/tuning/clean_run_both_backends.py

# measure your own hardware and footage, and get suggested settings
python3 tools/bench/betabench.py all

# run the test suite
./tests/run_tests.sh
```

Ctrl-C stops a run cleanly at the next safe point. Detection
checkpoints and completed render chunks are already on disk, so
re-running resumes rather than restarting. A second Ctrl-C exits
immediately.

---

## How a run works

For each video file:

```
   source video
        │
        ▼
 1. DECODE            sample at video_censor_fps; frames between samples
        │             are skipped without decoding their contents
        ▼
 2. DETECT            run the model at every configured picture size
        │             → cached to ../output/cache/vid_hashes/
        ▼
 3. SHOT CUTS         histogram scan for scene changes
        │             → cached to ../output/cache/shot_cuts/
        ▼
 4. CROSS-SIZE DEDUP  collapse one object detected at two sizes
        ▼
 5. GEOMETRY FILTER   drop boxes whose shape is implausible for the label
        ▼
 6. SUPPRESSION       drop detections that are probably a misclassification
        ▼
 7. TRACK             match detections into per-instance tracks; resolve
        │             one censor style per track; interpolate gaps;
        │             confirm tracks
        ▼
 8. RENDER            draw the censor onto every frame, in parallel
        │             resumable chunks, encoded directly as H.264
        ▼
 9. MUX               add the original audio by stream copy
        ▼
   censored .mp4
```

Steps 4, 5 and the hysteresis/confirmation parts of 7 are all **off by
default** and cannot remove a detection until you deliberately configure
them.

Photos (`betastare.py`) work the same way minus steps 1, 3, 7 and 9 — a
photo has one instant, not a timeline.

---

## Architecture

```
betatv.py              orchestration only: walk files, call the stages
betastare.py           the same, for stills

betautils_video.py     decoding, frame sampling, shot cuts, the
                       hardware-decode fallback ladder
betautils_detector.py  the detector-adapter boundary and backend
                       settings resolution
  detectors/nudenet_v3.py     YOLOv8-family adapter
  detectors/retinanet_v2.py   RetinaNet adapter
betautils_track.py     dedup, geometry filter, suppression, tracking,
                       smoothing, interpolation, confirmation
betautils_censor.py    turning a box into pixels
betautils_render.py    chunked parallel rendering, concat, audio mux

betautils_cache_paths.py  THE single source of truth for every cache
                          path, output filename and cache key
betautils_config.py    config validation and resolved per-label settings
betautils_hash.py      hashing, atomic JSON, detection checkpoints
betautils_log.py       logging, progress, structured stats
betautils_signals.py   cooperative Ctrl-C
betautils_cli.py       shared command-line flags

tools/bench/           the measurement harness
tools/analysis/        read-only reports over cached detections
tools/tuning/          harnesses that run real passes and compare
tests/                 unit, integration and synthetic end-to-end tests
```

Two rules hold this together, both of which exist because breaking them
caused real bugs:

**Nothing but `betautils_cache_paths.py` builds a filename.** Path
formulas were once re-implemented in up to six tools. They drifted: one
copy's glob omitted the backend name, corrupting the file hash it parsed
back out so every video looked "not found"; another failed to model the
whole-file preview fallback, so short videos silently missed the cache.
`tests/test_cache_path_duplicates_stay_in_sync.py` scans the tree for a
second copy.

**The pipeline stages are importable modules, not code inside
`betatv.py`.** Tracking and suppression used to live in `betatv.py`,
which meant `tools/tuning/replay_tune.py` read `betatv.py`'s *source
text*, regex-matched the two function bodies out of it, and `exec`'d
them in order to be sure it was testing the real code.

---

## Settings catalogue

Every setting in `betaconfig.py`. The **Reference** column links to the
full explanation, with analogies and examples, in
[`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md).

**per-backend** means the setting lives inside
`detector_backend[<name>]`, resolved as: that backend's own block →
`detector_backend['defaults']` → a built-in default. Setting one of
these at the top level is silently ignored at runtime, so validation
rejects it at startup.

### Detector backend

| Setting | Default | What it does |
|---|---|---|
| `detector_backend['selected']` | `'nudenet_v3'` | which model runs |
| `detector_backend['defaults']` | `{'nn_batch_size': 2}` | values shared by every backend |
| `model_variant` *(per-backend)* | `'320n'` | which export: `320n` (12 MB, native 320) or `640m` (103 MB, native 640) |
| `nn_batch_size` *(per-backend)* | 4 / 2 | frames per inference call. Never changes which detections come out |
| `picture_sizes` *(per-backend)* | variant's native size | detection sizes. The model runs once per size |
| `candidate_floor` *(nudenet_v3)* | `0.2` | per-anchor score needed to reach NMS |
| `nms_iou` *(nudenet_v3)* | `0.45` | overlap at which NMS drops the lower score |
| `nms_mode` *(nudenet_v3)* | `'per_class'` | `'per_class'` leaves cross-label overlaps for `class_suppression`; `'agnostic'` is NudeNet's upstream behaviour |
| `class_suppression` *(per-backend)* | per model | which labels suppress which, and how strictly |
| `class_promotion` *(per-backend)* | `{}` | relabel a detection when corroborating evidence overlaps it, e.g. `covered_vulva` → `exposed_vulva` under an `exposed_penis`. See [Class promotion](CONFIG_REFERENCE.md#class-promotion) |
| `item_overrides` *(per-backend)* | per model | per-label detection and tracking settings |

Reference: [Detector backend](CONFIG_REFERENCE.md#detector-backend)

### Hardware

| Setting | Default | What it does |
|---|---|---|
| `gpu_enabled` | `1` | request CUDA. A CPU provider is always added as a fallback, and the provider actually in use is logged and recorded in the stats row |
| `cuda_device_id` | `0` | which GPU |

### Detection sampling

| Setting | Default | What it does |
|---|---|---|
| `video_censor_fps` | `9` | frames per second of video that get detection. The biggest single lever on detection time |
| `picture_sizes` | `[1280]` | shared fallback for a backend that declares neither its own nor a native size |
| `nn_batch_size` | `1` | last-resort fallback |
| `global_min_prob` | `0.20` | hard floor applied to every raw detection, before any per-label gate |

Reference: [Detection sampling](CONFIG_REFERENCE.md#detection-sampling),
[Confidence and the three gates](CONFIG_REFERENCE.md#confidence-and-the-three-gates)

### Confidence

| Setting | Default | What it does |
|---|---|---|
| `default_min_prob` | `0.55` | confidence needed to start a censor track |
| `default_min_prob_continue` | `None` | the lower "keep going" gate. `None` disables score hysteresis |
| `min_prob` *(per-label, per-backend)* | — | that label's entry gate |
| `min_prob_continue` *(per-label, per-backend)* | — | that label's continue gate |

Reference: [Confidence and the three gates](CONFIG_REFERENCE.md#confidence-and-the-three-gates)

### Geometry sanity filter

All default to `None`, which disables the filter entirely.

| Setting | What it does |
|---|---|
| `default_min_area_fraction`, `min_area_fraction` *(per-label)* | reject boxes smaller than this fraction of the frame |
| `default_max_area_fraction`, `max_area_fraction` *(per-label)* | reject boxes larger than this fraction of the frame |
| `default_min_aspect_ratio`, `min_aspect_ratio` *(per-label)* | reject boxes taller than this width/height ratio |
| `default_max_aspect_ratio`, `max_aspect_ratio` *(per-label)* | reject boxes wider than this ratio |

Derive them: `python3 tools/bench/betabench.py geometry`

Reference: [Geometry sanity filter](CONFIG_REFERENCE.md#geometry-sanity-filter)

### Cross-size dedup

| Setting | Default | What it does |
|---|---|---|
| `cross_size_dedup['enabled']` | `True` | collapse one object detected at two picture sizes |
| `cross_size_dedup['iou_threshold']` | `0.60` | overlap above which two different-size detections are one thing |

Only ever compares detections from **different** sizes, so it is a no-op
with one size configured.

Derive it: `python3 tools/bench/betabench.py dedup`

Reference: [Cross-size dedup](CONFIG_REFERENCE.md#cross-size-dedup)

### Tracking and smoothing

| Setting | Default | What it does |
|---|---|---|
| `default_position_smoothing`, `position_smoothing` *(per-label)* | `0.50` | 0–1: lower is smoother, higher is snappier. Moves where the box **is** |
| `default_size_smoothing`, `size_smoothing` *(per-label)* | `position_smoothing / 3` | 0–1, same scale, applied to the box's **width and height**. Lower than position on purpose: a breathing box reads as flicker |
| `default_interpolation_enabled`, `interpolation_enabled` *(per-label)* | `True` | fill detection gaps with synthetic boxes |
| `default_interpolation_max_gap`, `interpolation_max_gap` *(per-label)* | `0.5` s | longest gap that gets filled |
| `track_max_gap` *(per-label)* | auto | longest gap a track survives. Defaults to the larger of `2/video_censor_fps` and that label's `interpolation_max_gap` |
| `match_distance_multiplier` *(per-label)* | `1.0` | how far a detection can be from a track and still match |
| `paired_style` *(per-label)* | `False` | share one resolved style between a nearby pair |
| `default_paired_style_max_distance` | `2.0` | how near "nearby" is, in box widths |
| `default_paired_style_tiebreak_margin` | `0.25` | how much closer the nearest candidate must be than the second |
| `default_style_min_dwell_seconds`, `style_min_dwell_seconds` *(per-label)* | `0.0` | how long a censor style is held before a shot cut may re-roll it |

`track_max_gap` and `interpolation_max_gap` are **not** the same thing;
conflating them was a real bug. See
[Tracking and smoothing](CONFIG_REFERENCE.md#tracking-and-smoothing).

Smoothing position and size at the same rate is what made censors
flicker. Reference:
[Position and size smoothing](CONFIG_REFERENCE.md#position-and-size-smoothing).

### Structure profiles

One set of timing values cannot serve a fast-cut compilation, a
single-scene video and a split-screen layout. Profiles are a fourth
override tier, selected per video from its measured shot structure.

Three ship: `scene` (the default and catch-all), `quick_cut`
(auto-selected on fast-cut footage, samples at 18 fps), and
`split_screen` (`manual_only`, since no measurement available before
detection runs can identify a split-screen layout, so reach it with
`--profile split_screen` or a `profile_by_path_pattern` rule).

| Setting | Default | What it does |
|---|---|---|
| `default_profile_enabled` | `True` | master switch. `False` ignores every profile and uses the un-profiled settings |
| `detector_backend[<backend>]['profiles']['default']` | — | profile used when a video has no shot-cut data |
| `detector_backend[<backend>]['profiles']['match_on']` | `'cuts_per_min'` | the measurement profiles are selected by: `'short_shot_fraction'` (recommended) or `'cuts_per_min'` |
| `detector_backend[<backend>]['profiles']['variants'][<name>]['min_short_shot_fraction']` | — | least share of runtime in sub-1s shots this profile claims, 0 to 1. Read when `match_on` is `'short_shot_fraction'` |
| `detector_backend[<backend>]['profiles']['variants'][<name>]['min_cuts_per_min']` | — | lowest cut rate this profile claims. Read when `match_on` is `'cuts_per_min'`. Highest threshold cleared wins |
| `detector_backend[<backend>]['profiles']['variants'][<name>]['manual_only']` | `False` | `True` hides this profile from automatic selection entirely, including the `default` fallback |
| `detector_backend[<backend>]['profiles']['variants'][<name>]['item_overrides']` | — | per-label settings this profile changes. Partial: unnamed keys keep their resolved values |
| `detector_backend[<backend>]['profiles']['variants'][<name>]['video_censor_fps']` | global rate | detection sample rate for footage this profile matches. The shot-cut scan always uses the global rate |

Profiles are configured at the **model** level, not per variant: two
variants of one model see the same footage the same way.

Reference: [Structure profiles](CONFIG_REFERENCE.md#structure-profiles)

### Track confirmation

| Setting | Default | What it does |
|---|---|---|
| `default_min_track_hits`, `min_track_hits` *(per-label)* | `1` | real detections a track needs before any of its boxes render. `1` renders every track |

Reference: [Track confirmation](CONFIG_REFERENCE.md#track-confirmation)

### Shot-cut detection

| Setting | Default | What it does |
|---|---|---|
| `shot_cut_detection_enabled` | `True` | find scene changes so tracks never smooth across one |
| `shot_cut_threshold` | `0.5` | Bhattacharyya distance 0–1; higher is less sensitive |

Reference: [Shot-cut detection](CONFIG_REFERENCE.md#shot-cut-detection)

### What gets censored

| Setting | What it does |
|---|---|
| `items_to_censor` | the labels that produce a censor. Every other detected label is either a suppression signal or ignored |

Every label BetaSuite understands is listed in `betaconst.classes`.
Some (`exposed_anus`, `exposed_armpits`, `exposed_chest`) exist purely
as suppression signals and are not meant to be censored directly.

### Censor styles

| Setting | Default | What it does |
|---|---|---|
| `default_censor_style` | gaussian blur, strength 22 | the style for any label with no override |
| `default_censor_shape` | `'box'` | `'box'`, `'circle'` or `'ellipse'` |
| `type_default_shapes` | `{'bar': 'box'}` | structural shape per style type |
| `censor_overlap_strategy` | per type | `'none'`, `'single-pass'` or `'span'` when same-style boxes overlap |
| `censor_scale_strategy` | `'feature'` | how `strength` is interpreted: `'feature'`, `'image'` or `'none'` |
| `blur_fast_approximation` | `True` | compute heavy blurs at reduced resolution. 30–60x faster, visually equivalent |
| `blur_edge_margin` | `True` | blur reads real neighbouring pixels around the box, so the result does not change when the box resizes. Read-only; cannot change what is censored |
| `blur_approximation_min_kernel` | `21` | kernels below this are always exact |
| `item_overrides[<label>]['censor_style']` | — | one style dict, or a weighted list to pick from per track |
| `item_overrides[<label>]['censor_shape']` | — | that label's shape |

A style dict's keys: `type`, `feather`, `weight`, `shape`, `merge`,
`width_area_safety`, `height_area_safety`, plus per-type keys —
`method`/`strength` (blur), `pattern`/`strength` (pixel),
`color`/`thickness`/`span_extend` (bar), `dir`/`images`/`scale`
(sticker).

Reference: [Censor styles](CONFIG_REFERENCE.md#censor-styles)

### Area and time safety

| Setting | Default | What it does |
|---|---|---|
| `default_area_safety` | `0` | fractional padding around every box |
| `width_area_safety`, `height_area_safety` | — | settable per style variant, per label, or as the default |
| `default_time_safety`, `time_safety` *(per-label)* | `0.30` s | how long a detection's censor persists, centred on the detection |

Reference: [Area safety and time safety](CONFIG_REFERENCE.md#area-safety-and-time-safety)

### Rendering and encoding

| Setting | Default | What it does |
|---|---|---|
| `render_workers` | `0` | chunks encoded at once. `0` is auto (CPU/2, capped at 8) |
| `render_chunk_seconds` | `600` | seconds of video per resumable chunk. `0` disables chunking |
| `render_chunk_container` | `'mkv'` | container for chunks and the intermediate |
| `render_verify_frame_counts` | `True` | reject a chunk that is valid but short |
| `encode_video_codec` | `'libx264'` | video encoder |
| `encode_crf` | `17` | quality: lower is better and larger |
| `encode_preset` | `'fast'` | x264 speed/size trade for a real run |
| `detection_checkpoint_frames` | `500` | samples between detection checkpoints. `0` disables |
| `ffmpeg_max_retries` | `2` | additional attempts after a failure |
| `ffmpeg_retry_backoff_seconds` | `5` | pause between attempts |

Reference: [Rendering and encoding](CONFIG_REFERENCE.md#rendering-and-encoding)

### Preview mode

| Setting | Default | What it does |
|---|---|---|
| `preview_mode_enabled` | `False` | process a slice of each video instead of all of it |
| `preview_max_seconds` | `20` | how long the slice is |
| `preview_encode_preset` | `'ultrafast'` | preset for preview renders |
| `preview_start_seconds` | `None` | pin where the slice starts. Use the same value across runs |
| `preview_random_slice` | `False` | pick a random offset per file when no start is pinned |

Reference: [Preview mode](CONFIG_REFERENCE.md#preview-mode)

### Logging and stats

| Setting | Default | What it does |
|---|---|---|
| `logging_enabled` | `True` | write a log file |
| `log_path` | `../output/logs/betasuite.log` | where |
| `log_level` | `'debug'` | the **log file's** level |
| `console_level` | `'info'` | the **terminal's** level |
| `stats_enabled` | `True` | write one JSON-Lines row per processed file |
| `stats_path` | `../output/stats/betasuite_stats.jsonl` | where |

Levels: `trace` < `debug` < `info` < `warn` < `error`.

Reference: [Logging and stats](CONFIG_REFERENCE.md#logging-and-stats)

### Input handling and miscellaneous

| Setting | Default | What it does |
|---|---|---|
| `input_delete_probability` | `0` | chance of deleting each source file once its censored output exists. Non-zero requires typing a confirmation at startup |
| `debug_mode` | `0` | `1` draws labelled debug boxes for every class instead of censoring; `3` also saves debug output |
| `enable_betasuite_watermark` | `False` | draw a watermark on every frame |

### BetaVision screen capture

| Setting | Default |
|---|---|
| `vision_cap_monitor`, `vision_cap_top`, `vision_cap_left`, `vision_cap_height`, `vision_cap_width` | `0`, `0`, `0`, `1080`, `960` |
| `vision_cursor_color` | `(168, 93, 253)` |
| `betavision_delay` | `0.5` |
| `betavision_interpolate` | `False` |

The capture region is the rectangle read from your screen. The censored
copy is displayed elsewhere.

---

## Command-line overrides

A flag overrides `betaconfig.py` **for that run only**; nothing is ever
written back. Anything you do not pass falls through to the file.

Deliberately not covered: detection and censoring tuning
(`items_to_censor`, `class_suppression`, censor styles, safety margins,
`min_prob`). Those are nested structures meant to be deliberated over
and committed, not flipped per invocation.

| Flag | Overrides |
|---|---|
| `--backend {nudenet_v3,retinanet_v2}` | `detector_backend['selected']` |
| `--variant NAME` | the selected backend's `model_variant` |
| `--picture-sizes N [N ...]` | the selected backend's `picture_sizes` |
| `--nn-batch-size N` | the selected backend's `nn_batch_size` |
| `--video-censor-fps FPS` | `video_censor_fps` |
| `--render-workers N` | `render_workers` |
| `--debug-mode N` | `debug_mode` |
| `--log-level LEVEL` | `log_level` (the file) |
| `--console-level LEVEL` | `console_level` (the terminal) |
| `--logging {on,off}` | `logging_enabled` |
| `--stats {on,off}` | `stats_enabled` |
| `--preview {on,off}` | `preview_mode_enabled` |
| `--preview-seconds SECONDS` | `preview_max_seconds` |
| `--preview-start-seconds SECONDS` | `preview_start_seconds` |
| `--preview-random {on,off}` | `preview_random_slice` |
| `--preview-preset PRESET` | `preview_encode_preset` |
| `--encode-preset PRESET` | `encode_preset` |

```bash
# a pinned 10-second slice, so repeated runs compare the same footage
python3 betatv.py --preview on --preview-seconds 10 --preview-start-seconds 120

# the 640 variant, quiet terminal, full detail in the log
python3 betatv.py --backend nudenet_v3 --variant 640m \
                  --console-level warn --log-level trace
```

Backend-scoped flags are written into the selected backend's own block,
because that is the tier the code actually reads. `--backend` is applied
first, so combining it with `--nn-batch-size` lands the batch size in
the backend this run will really use.

Two environment variables take precedence over the flags, so an
orchestrator can sweep across subprocesses without ever touching
`betaconfig.py`:

```bash
BETASUITE_DETECTOR_BACKEND_OVERRIDE=retinanet_v2
BETASUITE_DETECTOR_VARIANT_OVERRIDE=640m
```

`betastare.py` exposes the same flags minus the preview group.

Every override is logged at startup, and a bad value fails immediately
with a specific message — overrides are applied before validation and
are validated the same way as the file.

---

## Output filenames and cache keys

```
clip-a1b2c3d4e5f6a7b8-320-9-d5418a7-c3bd954-e1638.mp4
     └── source hash ──┘ │  │ └ det ┘ └ cen ┘ └enc┘
                     sizes fps
```

| Key | Covers | Changing it means |
|---|---|---|
| `d<6>` detection | backend, variant, size, sample fps, `global_min_prob`, the backend's detection tunables | re-detect |
| `c<6>` censor | every per-label setting, tracking, suppression, styles, geometry filter, dedup, structure profiles, style dwell, shot-cut settings | re-render from cache |
| `e<4>` encode | codec, CRF, preset, container | re-encode only |

`../output/cache/run_keys/<kind>-<key>.json` holds each key's full
expansion, so a filename can always be decoded back into the settings
that produced it.

**Anything that changes the bytes on disk changes the filename.** Earlier
versions of the name carried a narrow hash that excluded every tracking
setting, so re-running with a changed `track_max_gap` silently
overwrote the previous output.

A file whose output already exists at the resolved name, and decodes, is
**skipped before any scanning or detection work** — so a repeated run
with unchanged settings costs seconds rather than minutes. That skip
leans entirely on the keys being complete, which is why
`tests/test_censor_key_coverage.py` exists. See
[The three cache keys](CONFIG_REFERENCE.md#the-three-cache-keys).

One exception: `preview_random_slice` picks a fresh offset every run, and
the offset is written into the name at one-decimal precision, so a new
draw can land on an existing file's name. A random slice is a request for
a new sample, so it always re-runs.

Cache layout under `../output/cache/`:

| Directory | Holds |
|---|---|
| `vid_hashes/` | per-video, per-size raw detections |
| `pic_hashes/` | per-image raw detections |
| `shot_cuts/` | per-video scene-change timestamps |
| `transcode_cache/` | H.264 copies of files OpenCV cannot decode |
| `run_keys/` | what each key in a filename expands to |
| `file_hashes.json` | memoised content hashes, keyed on path/size/mtime |

---

## Resumability, interrupts and crash recovery

One rule throughout: **a file is only trusted once it has been renamed
into its final name, and it is always written elsewhere first and
renamed atomically only after the write succeeds.** A `.tmp` or
`.checkpoint` file lying around always means "incomplete", never "done".

**Detection.** Progress is saved every `detection_checkpoint_frames`
samples to a separate `<cache>.checkpoint` file — never the trusted
cache path. On restart, a checkpoint with no completed cache resumes
from that sample rather than from zero.

**Rendering.** Each chunk is rendered to its own temp file and promoted
only once ffmpeg exits 0, `ffprobe` can read the container, **and** the
frame count matches what was planned. A restart trusts existing chunks
and resumes from the first missing one.

> The frame-count check is not redundant with the container check. A
> chunk whose decode died mid-file is a perfectly valid video that is
> simply too short: previously it passed the container check, got
> promoted, and a later resume skipped it — silently truncating the
> output.

**Failures and retries.** Every ffmpeg invocation has its exit code
checked and is retried up to `ffmpeg_max_retries` times with
`ffmpeg_retry_backoff_seconds` between attempts. A file that exhausts
its attempts is skipped; the batch continues.

**Interrupts.** Ctrl-C sets a flag; the run stops at the next safe point
— after checkpointing whatever detection is in flight, and without
promoting a partial chunk. A second Ctrl-C exits immediately. The
process exits 130.

> Previously the per-file handler caught `BaseException`, so Ctrl-C
> was reported as a failed file, logged, slept a second, and moved to
> the next one — you had to press it once per remaining file. The render
> loop caught it too and fed it into the retry-with-backoff path. Now
> the interrupt subclasses `KeyboardInterrupt` specifically so it passes
> through the `except Exception` handlers that give the pipeline its
> one-bad-file-is-not-fatal behaviour.

**Decode fallbacks.** A file OpenCV cannot decode is retried with
hardware acceleration disabled two different ways, then transcoded to
H.264 with the system ffmpeg and cached. Everything downstream —
including the shot-cut scan and the render — then reads the transcoded
copy, and the frame count is re-read from it.

> Previously the transcode repointed only the main capture. The
> shot-cut scanner still opened the original file that OpenCV had just
> proved it could not decode, got no frames, and silently cached an
> empty cut list for the whole video. The chunk planner also still used
> the original's frame count, which a re-encode can shift.

The original file is always the audio source for the final mux; the
transcode exists to work around a video-decode problem and has no
bearing on the audio.

---

## Measuring and tuning

### `tools/bench/betabench.py`

One harness. Each subcommand answers a specific configuration question
with numbers from **your** hardware and **your** footage, and prints the
setting it implies.

```bash
python3 tools/bench/betabench.py all
python3 tools/bench/betabench.py detect --variants 320n 640m --sizes 320 640
python3 tools/bench/betabench.py suppression --backend nudenet_v3
python3 tools/bench/betabench.py --help
```

| Command | Answers | Informs |
|---|---|---|
| `decode` | is sequential decoding really faster here? | confirms the sampling design |
| `detect` | what does one sampled frame cost, per stage? | `picture_sizes`, `model_variant`, `nn_batch_size` |
| `render` | what does one censored box cost per frame? | style weights, `blur_fast_approximation` |
| `geometry` | what box shapes does each label produce? | the geometry sanity filter |
| `suppression` | what do cross-label overlaps look like? | `class_suppression` |
| `dedup` | how much do picture sizes overlap? | `cross_size_dedup` |
| `hysteresis` | what are the score distributions? | `min_prob`, `min_prob_continue` |
| `structure` | per video: cut rate, subjects on screen, spatial layout | whether footage needs per-profile settings |

`structure` is the one to read per file rather than pooled: it reports
cut rate, how many subjects are live at once, and where they sit in
frame, which is what decides whether a file wants different timing and
tracking values from the rest. See CONFIG_REFERENCE.md ->
"Reading `structure`, and what profiles would need".

Everything goes to `../output/benchmarks/<timestamp>/` with a log at
`--log-level` (default `debug`), a terminal view at `--console-level`
(default `info`), and machine-readable `results.json`. Any step needing
a decision prompts; `--yes` accepts defaults and never blocks.

`geometry`, `suppression`, `dedup` and `hysteresis` read caches a real
run already produced, so they are fast and describe exactly what that
run saw. Run `betatv.py --preview on` first if the cache is empty.

### `tools/analysis/summarize_run.py`

The first thing to run after a real run finishes. It reads
`betasuite_stats.jsonl` and reports, per configuration, where the wall
clock went and what was detected.

```bash
python3 tools/analysis/summarize_run.py
python3 tools/analysis/summarize_run.py --last 7      # just the run that finished
python3 tools/analysis/summarize_run.py --since 2026-09-17
```

Rows are grouped by backend, variant, picture sizes, sample rate and
batch size, and never pooled across those: the difference between two
groups is the reason both were run, and a mean across them describes
neither. Two backends in one file therefore produce two reports plus a
per-source-minute comparison table that survives each configuration
having processed different files.

Read it before acting on anything else here. Tuning the stage that took
16% of a run caps the possible saving at 16%, whatever else the tuning
output says.

### Every tool covers every model and variant, automatically

No tool takes a backend or a variant to do its job. Each one asks the
**cache directory** what has actually been run and reports on all of it:

```bash
tools/analysis/run_all_analysis.sh        # the whole suite, every configuration
python3 tools/bench/betabench.py all      # same, for the benchmarks
python3 tools/tuning/auto_tune.py         # same, for the tuner
```

A *configuration* is one backend at one detection size under one
detection key — so `nudenet_v3/320n @320` and `nudenet_v3/640m @640` are
two, never one. The size comes from the cache filename; the variant name
is recovered from that key's run-key manifest, which is what those
manifests are for.

This is deliberately derived from disk rather than from config. Config
tells you what is selected *now*; the cache tells you what was *run*.
Those differ the moment a variant is swapped, and the difference is not
academic:

> Previously every tool defaulted to the shared
> `betaconfig.picture_sizes`. A full overnight run of nudenet at both
> 320n and 640m left three complete cache sets on disk, and every
> analysis tool reported on one of them, because config had been left at
> 320n. The output read "no detection caches found" — indistinguishable
> from having run nothing at all.

`--picture-sizes` (`--sizes` on betabench) still overrides discovery,
for inspecting a shape discovery would not produce:

```bash
python3 tools/analysis/analyze_jitter.py --picture-sizes 640
python3 tools/bench/betabench.py hysteresis --sizes 640
```

### `tools/tuning/auto_tune.py`

The staged tuner. It measures, decides, writes the decision into
`betaconfig.py`, and moves to the next question.

```bash
python3 tools/tuning/auto_tune.py            # measure and report, write nothing
python3 tools/tuning/auto_tune.py --apply    # also write the winners
python3 tools/tuning/auto_tune.py --list-stages
```

Each stage owns one knob, for one label, under one configuration. Stages
run in order and an accepted value is written before the next stage
starts, so later stages measure against earlier decisions. Every
candidate is scored by replaying the detection caches through the real
`betautils_track` pipeline, so a full sweep is seconds, not hours.

The decision rule is a **price**, not a target. Loosening a tracking
threshold always reduces track resets if you loosen it far enough; the
question is how many ambiguous track assignments each rescued reset
cost. From one night's real footage: tripling `match_distance` for
`exposed_breast` fixed 130 resets and added 8874 risky assignments, 68
apiece. Tripling `track_max_gap` fixed 135 and added 536, four apiece.
Same headline improvement, seventeen times the cost, and only the price
tells them apart.

**One rule no stage can break.** A candidate that reduces censor
coverage — total censored area-seconds, per label, over the same footage
— by more than `--max-coverage-loss` (default 0.5%) is rejected whatever
else it improves. Performance is never bought with censoring.

Every write takes a timestamped backup, and the value is read back
through the real resolver afterwards; a mismatch restores the backup and
stops. It will not touch anything that changes raw detections — picture
size, model variant, `global_min_prob` — because those cannot be
replayed from a cache of detections and would need a real run.

### The loop that works

1. **Run a pinned preview slice.**
   `python3 betatv.py --preview on --preview-start-seconds 120`
   The same slice every time, so differences are yours and not the
   footage's.
2. **Look at the distributions.**
   `python3 tools/bench/betabench.py suppression hysteresis geometry`
3. **Change one thing.** The output filename changes with it, so both
   versions survive to compare.
4. **Re-run the same slice and compare** — visually, and via
   `../output/stats/betasuite_stats.jsonl`.
5. **Widen to a real run** once a setting looks right on the slice.

Changing several settings between comparisons means you learn nothing
from the difference.

### Other tooling

| Tool | What it does |
|---|---|
| `tools/tuning/clean_run_both_backends.py` | clears every cache, runs a real full pass under each backend, then the analysis suite |
| `tools/tuning/replay_tune.py` | replays cached detections through the real tracking and suppression code under config variants, with no re-detection |
| `tools/tuning/batch_ab_test.py` | real `betatv.py` runs across batch sizes and backends, comparing detections and timing |
| `tools/tuning/derive_suppression_rules.py` | turns overlap statistics into proposed rules |
| `tools/analysis/analyze_*.py` | read-only reports over cached detections: jitter, track breaks, style flicker, span merges, score distributions, suppression pairs |
| `tools/analysis/summarize_run.py` | where a finished run's wall clock went, and what it detected, per configuration |
| `tools/tuning/auto_tune.py` | the staged tuner: measures candidates, prices them, writes the winners |
| `tools/analysis/run_all_analysis.sh` | runs `summarize_run.py` and the analysis suite in one go |

Every tool that replays the pipeline imports the real
`betautils_track` functions and observes their decisions through
`betautils_track.tracking_observer`, a supported extension point.
Previously several of them read `betatv.py`'s source text and
`exec`'d the extracted function bodies instead — which broke the
morning after an overnight run, when those functions moved to a module
of their own. `tests/test_analysis_tools_contract.py` now fails if any
tool goes back to reading source text.

---

## Testing

```bash
./tests/run_tests.sh                    # everything
./tests/run_tests.sh -k suppression     # just the matching tests
```

Discovery runs from the repository root so the test modules import the
same way the application does.

> Running discovery from inside `tests/` silently failed to import
> eleven of these modules while still reporting a pass for the rest.

What is covered, and why each exists:

| File | Covers |
|---|---|
| `test_frame_sampling.py` | the fast sampler reads **exactly** the frames the old seek-based loop did, pixel for pixel |
| `test_detector_invariants.py` | batch-size invariance; the vectorised anchor decode matches a scalar reference on randomised input; per-class vs agnostic NMS |
| `test_track_pipeline.py` | every filter is a no-op when off; hysteresis; track confirmation; suppression counts |
| `test_analysis_tools_contract.py` | the seam every analysis tool depends on: no tool reads pipeline source text, the tracking observer fires, picture sizes resolve per backend, and a parameter sweep's value actually reaches the resolver |
| `test_summarize_run.py` | stats rows from different configurations are never pooled; a truncated stats line degrades to a message |
| `test_auto_tuning.py` | the coverage guard cannot be fooled by a gain on another label; a decision rule rejects on price even when the headline number improves; the config writer edits the right key and proves it took |
| `test_censor_rendering.py` | the merge does not mutate its input; hex tessellation; mask caching; blur approximation |
| `test_render_pipeline.py` | chunk planning; the truncation guard; a real parallel render matching a serial one |
| `test_betatv_end_to_end.py` | real clips through real orchestration with a stub detector; preview bounds; interrupt semantics |
| `test_cache_path_duplicates_stay_in_sync.py` | no second copy of a path formula; the keys change exactly when they should |
| `test_config_and_logging.py` | every validator; resolution order; CLI overrides; log levels; progress throttling |
| `test_shot_cuts.py`, `test_smooth_boxes.py`, and the rest | the behaviours those modules already had |

The end-to-end tests build their own clips with ffmpeg and skip
themselves if ffmpeg is unavailable. No test needs a model file.

---

## Versions and releases

The release version lives in `VERSION` and follows
[semantic versioning](https://semver.org/). `--version` on any entry
point prints the version of the code you are running:

```bash
python3 betatv.py --version
# BetaSuite 1.1.0                    a clean checkout of tag v1.1.0, or a zip download
# BetaSuite 1.1.0+3.g1a2b3c4         3 commits past v1.1.0
# BetaSuite 1.1.0+3.g1a2b3c4.dirty   ...with uncommitted changes
```

The same string is in BetaTV's startup log line and in every stats row
(`betasuite_version`), so a timing can always be traced to the code that
produced it.

To cut a release, note your changes under a `## Unreleased` heading in
`CHANGELOG.md` as you go (optional), then run:

```bash
pip install -r requirements-dev.txt      # once: the semver package
python3 tools/release/release.py         # or --dry-run to see the plan
```

It lists the commits since the last tag, suggests a bump (`major` for
"BREAKING" or `type!:`, `minor` for subjects starting with feat/add/new,
`patch` otherwise), and asks before each step: bump `VERSION`, date the
CHANGELOG entry (drafting one from the commit subjects if there's no
`Unreleased` section), run the tests, commit and tag, push, and create the
GitHub release. Everything is logged to `tools/release/release.log`.

---

## Performance notes

The main performance changes in this fork, and what each was worth. Measured figures are
from the machine they were taken on; re-measure yours with
`betabench`.

| Change | Measured |
|---|---|
| Sequential `grab`/`retrieve` instead of a seek per sampled frame | 173.7 → 4.4 ms/sample on 1280x720 H.264 at GOP 250 |
| Vectorised nudenet anchor decode instead of a per-anchor Python loop | 96.8 → 0.52 ms/frame at a 1280 blob |
| Analytic hex tessellation instead of a mask-and-scan per cell | 21–175 → ~1 ms per box per frame |
| Cached shape masks and feather instead of rebuilding per frame | 0.7–17.6 ms per box per frame, eliminated |
| Approximate heavy Gaussian blur | 33–61x on kernels above ~100 |
| One encode instead of two (H.264 chunks, stream-copy concat and mux) | a whole extra encode of every output |
| Parallel render chunks | up to `render_workers` times on the render stage |
| `nn_batch_size` actually taking effect for `retinanet_v2` | it had been silently running at 1 |
| Memoised file hashes, 4 MiB reads instead of 8 KiB | re-runs no longer re-read every byte of every source |
| Pruned dead tracks, cached `parts_to_blur`, throttled progress output | removes quadratic and per-detection overheads |

None of these change which detections come out. The two that could have
— the sampler and the anchor decode — are both covered by equivalence
tests against reference implementations.

---

## Known limitations

- **Diffusion/inpainting censoring is not implemented.** It needs a
  generative model running locally; out of scope.
- **The `nudenet_v3` tuned values need re-deriving.** They were fitted
  at picture size 1280 with class-agnostic NMS; that backend now runs at
  its native 320 with per-class NMS. See
  [Class suppression](CONFIG_REFERENCE.md#class-suppression).
- **There is no `covered_penis` label.** Nothing to suppress a
  covered-penis false positive with.
- **Stream-copying untouched spans is not implemented.** On footage with
  sparse detections, copying the ranges that contain no boxes rather
  than re-encoding them would be a large further win. Parked pending
  real measurements from a few full runs.
- **Pixelation patterns beyond square/mosaic/hex** are not implemented.

---

## Troubleshooting

**"betaconfig.py failed validation"** — validation caught a problem
before any real work: a misspelled label, an unknown key, a
backend-tunable setting in the shared block, bounds that cannot both be
satisfied. Each line names the specific problem and, where relevant, the
valid options.

**A run is far slower than expected** — check the `execution_providers`
field in `../output/stats/betasuite_stats.jsonl`, or the startup line in
the log. If `gpu_enabled` is set but CUDA is not in the provider list,
the run silently fell back to CPU; the log carries a warning saying so.

**A censor flickers on and off** — the detection is crossing `min_prob`
back and forth. Set that label's `min_prob_continue` below its
`min_prob` to give it hysteresis. See
[Confidence and the three gates](CONFIG_REFERENCE.md#confidence-and-the-three-gates).

**A tracked object's censor style keeps changing** — style is resolved
once per track, so this means the *tracking* is losing and re-acquiring
the object, each time starting a new track with its own style.
`track_max_gap` is usually the culprit; `interpolation_max_gap` alone
cannot fix it, because it only applies once a match has already
succeeded. See
[Tracking and smoothing](CONFIG_REFERENCE.md#tracking-and-smoothing).

**Single-frame censor blobs on nothing** — raise that label's
`min_track_hits` to 2. See
[Track confirmation](CONFIG_REFERENCE.md#track-confirmation).

**A large spurious censor covering much of the frame** — the model found
a torso. Set that label's `max_area_fraction`; derive the value with
`betabench geometry`. See
[Geometry sanity filter](CONFIG_REFERENCE.md#geometry-sanity-filter).

**A `min_iou` never seems to fire** — check the achievable IoU for that
pair with `betabench suppression` before assuming the rule is wrong. A
size mismatch between the two labels' typical boxes can make a given
`min_iou` mathematically unreachable however well centred they are.

**A sticker style shows a plain black bar** — no usable PNGs were found
via `dir`/`images`. It falls back to a bar rather than failing the run,
and logs once. Check the path and that the folder has `.png` files.

**Skin briefly visible above or below a bar** — a box's rendered
position is static for its whole `time_safety` window, and a thin bar
covers only part of the box height. Raise `thickness` for that variant.
See [`type: 'bar'`](CONFIG_REFERENCE.md#type-bar).

**Preview output where a real run was expected** — check
`preview_mode_enabled`. Preview output has its own filename suffix and
its own cache, so it can never corrupt a real run either way, but it is
easy to leave on.

**The run seems stuck** — check the log at `debug`. Every long stage
emits a progress record on a bounded interval. If the terminal is quiet,
`console_level` is probably filtering it; run with
`--console-level debug`.

**Two output files for what looks like the same settings** — something
that changes the output changed. Compare the `d`/`c`/`e` keys in the two
names against `../output/cache/run_keys/` to see exactly which stage
differs.
