# BetaSuite configuration reference

Every setting in `betaconfig.py`: what it does, what actually changes on
screen when you change it, and why the shipped values are what they are.

`betaconfig.py` itself holds values and one-line headers only. This is
where the reasoning lives.

**Contents**

- [How to read this document](#how-to-read-this-document)
- [The three cache keys](#the-three-cache-keys)
- [Detector backend](#detector-backend)
- [Detection sampling](#detection-sampling)
- [Confidence and the three gates](#confidence-and-the-three-gates)
- [Geometry sanity filter](#geometry-sanity-filter)
- [Class suppression](#class-suppression)
- [Cross-size dedup](#cross-size-dedup)
- [Tracking and smoothing](#tracking-and-smoothing)
- [Track confirmation](#track-confirmation)
- [Shot-cut detection](#shot-cut-detection)
- [Censor styles](#censor-styles)
- [Area safety and time safety](#area-safety-and-time-safety)
- [Rendering and encoding](#rendering-and-encoding)
- [Preview mode](#preview-mode)
- [Logging and stats](#logging-and-stats)
- [Why the shipped values are what they are](#why-the-shipped-values-are-what-they-are)
- [Re-deriving values from your own footage](#re-deriving-values-from-your-own-footage)

---

## How to read this document

Each setting gets four things:

**What it is** — the mechanical definition.
**The analogy** — a way to hold the idea, when the mechanics alone are
hard to reason about.
**What you see** — what actually changes in the output, described
concretely.
**How to pick a value** — and, where one exists, which `betabench`
command derives it from your own footage.

A setting marked **per-backend** lives inside
`detector_backend[<name>]`, not at the top level. Setting it at the top
level is silently ignored at runtime, so validation treats it as an
error at startup instead.

### One caution about every number in this file

The values shipped in `betaconfig.py` were fitted to particular footage
under a particular configuration. Framing, subject distance, and the
detector variant all move these distributions. A number derived from one
library is a starting point for another, not a constant.

Where a value's evidence no longer describes the current configuration,
it says so.

---

## The three cache keys

Output filenames look like this:

```
clip-a1b2c3d4e5f6a7b8-320-9-d5418a7-c3bd954-e1638.mp4
     └── source hash ──┘ │  │ └ det ┘ └ cen ┘ └enc┘
                     sizes fps
```

Three independent stages produce the output, so each gets its own key.
Changing one invalidates exactly that stage:

| Key | Covers | Changing it means |
|---|---|---|
| `d<6>` **detection** | backend, model variant, detection size, sample fps, `global_min_prob`, the backend's own detection tunables | re-detect |
| `c<6>` **censor** | every per-label setting, tracking, suppression, styles, overlap strategy, geometry filter, dedup, structure profiles, style dwell, shot-cut settings | re-render from the cached detections |
| `e<4>` **encode** | codec, CRF, preset, container | re-encode only |

`../output/cache/run_keys/<kind>-<key>.json` holds each key's full
expansion, so a filename from months ago can be decoded back into the
settings that produced it.

**Why this matters.** Before v2.1.0 the filename carried a narrow hash
that deliberately excluded every tracking setting. Re-running with a
changed `track_max_gap` produced the same filename, silently overwrote
the previous output, and left nothing to compare. Now anything that
changes the bytes on disk changes the name.

`nn_batch_size` is deliberately **not** in any key: batching changes how
many frames go into one inference call, never which detections come out.
That claim is enforced by `tests/test_detector_invariants.py` rather
than assumed.

**The censor key is load-bearing, and it has been wrong before.** A run
whose output file already exists at the resolved name is skipped
outright, before any scanning or detection work. That skip is only
correct while the key covers *everything* that changes rendered bytes —
a setting that changes output but not the key means a tuning run quietly
reuses the previous render and reports it as the new result, which is
indistinguishable from the setting having no effect.

Shot-cut settings were missing for exactly this reason, and hid well:
they have their own separate scan cache, so lowering `shot_cut_threshold`
*did* correctly rescan and find more cuts, then reused a video rendered
from the old ones. Structure profiles hid differently — they override
per-label values *after* those values resolve, so the key's `labels`
entry looked complete while describing settings the render never used.

`tests/test_censor_key_coverage.py` now drives each setting the way a
person tuning would and asserts the key moves. Add a setting that
changes rendered output, add it to `censor_identity()` and to that file.

> **Changed in 2.5:** structure profiles, `default_style_min_dwell_seconds`,
> `default_profile_enabled`, `shot_cut_threshold` and
> `shot_cut_detection_enabled` joined the censor key. Existing rendered
> outputs get new filenames on the next run. Nothing is lost — the old
> files remain, under their old names — but a full re-render is a real
> cost, so expect it rather than being surprised by it.

---

## Detector backend

```python
detector_backend = {
    'selected': 'nudenet_v3',
    'defaults': { 'nn_batch_size': 2 },
    'nudenet_v3':   { ... },
    'retinanet_v2': { ... },
}
```

Settings resolve through three tiers, most to least specific:

```
detector_backend['nudenet_v3']['nn_batch_size']   ← this backend's own
detector_backend['defaults']['nn_batch_size']     ← shared default
the adapter's / BetaSuite's built-in default      ← last resort
```

`'defaults'` is the tier for a setting that is model-specific in
principle but for which one sensible value covers every backend that has
not said otherwise.

### `selected`

Which model runs. `'nudenet_v3'` or `'retinanet_v2'`.

Override for one run without editing the file:

```bash
python3 betatv.py --backend retinanet_v2
BETASUITE_DETECTOR_BACKEND_OVERRIDE=retinanet_v2 python3 betatv.py
```

The environment variable wins over the flag, which wins over the file.
That ordering exists so an orchestrator can flip backends across
subprocesses without ever writing `betaconfig.py` — a crash mid-run can
then never leave the file flipped.

### `model_variant` (nudenet_v3, per-backend)

Which export of the model to load.

| Variant | File | Size | Native input |
|---|---|---|---|
| `'320n'` | `v3.4-320n.onnx` | 12 MB | 320x320 |
| `'640m'` | `v3.4-640m.onnx` | 103 MB | 640x640 |

**The analogy.** A model is trained to recognise things at a particular
apparent size, the way you learn to read a page at arm's length. Hand it
a page at four times that size and you are looking at individual letter
strokes — each one perfectly sharp, none of them a word.

**What you see.** Running `320n` at a 1280 blob does not fail. It runs
16x the arithmetic, produces 16x the anchors, and reports boxes that are
systematically smaller and tighter than the same model at 320 — because
it is finding sub-features rather than whole objects. Every IoU-based
threshold derived from that output inherits the distortion.

**How to pick.** Start at `320n` because it is fast. Move to `640m` when
`320n` misses detections you care about, and measure both:

```bash
python3 tools/bench/betabench.py detect --variants 320n 640m
```

`picture_sizes` defaults to the selected variant's native size, so
switching variants moves the detection size with it automatically.

### `nn_batch_size` (per-backend, with a shared default)

Frames handed to the model in one inference call.

Resolution order:

```
detector_backend['nudenet_v3']['nn_batch_size']   ← explicit
detector_backend['defaults']['nn_batch_size']     ← shared default
1                                                 ← last resort
```

> **Changed in 2.5.** The module-level `betaconfig.nn_batch_size`
> fallback was removed, but the `'defaults'` tier was deliberately
> kept — unlike `picture_sizes` and `class_suppression`, this is a
> VRAM/throughput knob rather than a statement about what the model
> means, so one sensible value can legitimately cover a backend that has
> not expressed a preference. A stale top-level value is now reported by
> `validate_config()` instead of silently winning.

**What you see.** Nothing, in the output. Higher values trade memory for
throughput. `retinanet_v2` carries 146 MB of weights and hits CUDA
out-of-memory at batch sizes that are comfortable for `nudenet_v3`'s
12 MB export, which is why this is per-backend at all.

**How to pick.** Raise it until you see an out-of-memory error, then
back off one step:

```bash
python3 tools/bench/betabench.py detect --batch-sizes 1 2 4 8
```

> Before v2.1.0 the `retinanet_v2` adapter read the **top-level**
> `betaconfig.nn_batch_size` rather than the per-backend value. Setting
> it in the backend block did nothing: the frame buffer collected two
> frames, handed them over, and the adapter split them straight back
> into two single-image calls — while the output filename said
> `-batch2`. Fixed, with a test that asserts both the invariance and
> that the batching actually happens.

### `candidate_floor` (nudenet_v3, per-backend)

The per-anchor score an output has to reach to be considered for NMS at
all.

**The analogy.** A first-pass sift before the real sorting. Set too
high, you throw away things the second pass would have kept; set too
low, the second pass has to sort through a pile of gravel.

**What you see.** Below about 0.15 you get noticeably more work per
frame and more marginal detections reaching `global_min_prob`. Above
about 0.35 you start losing genuine low-confidence detections that
hysteresis could otherwise have rescued.

### `nms_iou` (nudenet_v3, per-backend)

How much two boxes must overlap before non-maximum suppression drops the
lower-scoring one. 0.45 is NudeNet's own default.

### `nms_mode` (nudenet_v3, per-backend)

**`'per_class'` (default)** — a detection only suppresses others of the
**same** label.

**`'agnostic'`** — the highest-scoring detection suppresses any
overlapping box regardless of label. NudeNet's own upstream behaviour.

**The analogy.** Two witnesses describing the same corner of a room.
Class-agnostic NMS says only one of them may speak. Per-class NMS lets
both speak and leaves it to a later, better-informed judge — which is
what `class_suppression` is, with per-pair thresholds and a confidence
margin.

**What you see.** Under `'agnostic'`, an `exposed_breast` overlapping an
`exposed_belly` loses whichever scored lower, before anything tunable
gets a say. Under `'per_class'`, both survive to `class_suppression`.

**Why the default changed.** Class-agnostic NMS deletes exactly the
high-IoU cross-label pairs that `class_suppression` exists to arbitrate,
using one global threshold and no margin logic. It is also the most
likely mechanical explanation for `nudenet_v3`'s measured cross-label
IoU distributions sitting an order of magnitude below `retinanet_v2`'s
on identical footage — the overlapping pairs were removed before
anything could measure them. See
[Why the shipped values are what they are](#why-the-shipped-values-are-what-they-are).

### `picture_sizes` (per-backend)

Detection sizes. The model runs **once per size listed**, so two sizes
costs roughly twice the detection time.

Resolution order:

```
detector_backend['nudenet_v3']['picture_sizes']   ← explicit
detector_backend['defaults']['picture_sizes']     ← shared default
the variant's native size                         ← convention
```

> **Changed in 2.5.** The module-level `betaconfig.picture_sizes`
> fallback was removed. A detection size is a property of the model and
> its export, so a single shared number could only ever be right for one
> backend at a time: `nudenet_v3/640m` wants 640, `retinanet_v2` wants
> 1280, and whichever one the shared value matched, the other was
> silently running at the wrong size. A backend that declares neither
> its own `picture_sizes` nor a native size now fails validation with a
> specific message instead of inheriting someone else's number. If you
> still have `picture_sizes` at the top of `betaconfig.py`,
> `validate_config()` will tell you to move or delete it rather than
> ignoring it quietly.

**What you see.** Larger sizes find smaller objects and cost more. For
`retinanet_v2` (trained on variable-size input), 1280 is a reasonable
working value. For `nudenet_v3`, leave it at the variant's native size
unless you are deliberately experimenting — see `model_variant` above.

Running multiple sizes makes [cross-size dedup](#cross-size-dedup)
meaningful; with one size it is a no-op by construction.

---

## Detection sampling

### `video_censor_fps`

How many frames per second of video get detection and tracking. **9** by
default.

**The analogy.** Sampling a conversation. At 9 samples a second you
catch every word; you do not need every phoneme. What you miss is
sub-100ms motion.

**What you see.** This is the single biggest lever on detection time, in
direct proportion: 9 → 18 doubles it. Lower it and fast motion gets
blurrier tracking, because there are fewer real positions between
interpolated ones.

```
video at 25fps, video_censor_fps = 9

frame:   0    1    2    3    4    5    6    7    8    9   10   11
sample:  ●         ●         ●         ●         ●         ●
         └─ detected ─┘  the frames between are decoded past, never
                         run through the model
```

Frames between samples are skipped with `grab()`, which does not decode
their contents — so raising `video_censor_fps` costs inference time, not
decode time.

### `global_min_prob`

A hard floor applied to **every** raw detection, inside the adapter,
before any per-label `min_prob` ever sees it. **0.12**.

**What you see.** Nothing below this can be censored, no matter what a
label's own `min_prob` says. A per-label `min_prob` at or below this
value can never reject anything, which validation treats as an error
rather than letting it sit there as a silent no-op.

Lowering it lets more marginal detections reach the per-label gates —
more work, and more for those gates to filter. Raising it is a blunt
instrument; prefer per-label `min_prob`.

**This is a real floor, and it is easy to mistake for a per-label one.**
If a label's `min_prob` already sits near `global_min_prob`, lowering
that label's value further changes nothing — everything between the two
was already discarded upstream. At `nudenet_v3/640m`, `exposed_breast`
had a `min_prob` of 0.37 against a score distribution whose 1st
percentile was 0.228: the label gate was doing essentially no work, and
the only setting that could admit more detections was this one.

`global_min_prob` is part of the **detection cache key**, so changing it
invalidates every cached detection and forces a full re-detect. That is
the honest cost of the change, and the reason to prefer per-label
`min_prob` whenever the label gate is the one actually binding.

> **Changed in 2.5:** lowered from 0.20 to 0.12, to admit the low-score
> detections behind "obvious content gets nothing at all". Watch the
> false-positive rate after this change; if it rises more than the
> recall gain is worth, raise it back toward 0.16 rather than putting
> the per-label gates back up.

---

## Confidence and the three gates

A detection passes through three separate confidence gates. Getting them
confused is the most common source of "why is my `min_prob` doing
nothing".

```
  raw model output
        │
        ▼
  candidate_floor      ← per anchor, nudenet_v3 only, before NMS
        │
        ▼
  global_min_prob      ← every detection, every label, inside the adapter
        │
        ▼
  min_prob             ← per label, when a box becomes censorable
        │  └── min_prob_continue: the lower "keep going" gate
        ▼
  censorable box
```

### `default_min_prob` and per-label `min_prob` (per-backend)

The confidence a detection needs before it can start a censor track.

**What you see.** Raising it removes marginal detections — fewer false
positives, more missed frames. Lowering it does the reverse. This is the
main efficacy dial.

**How to pick.**

```bash
python3 tools/bench/betabench.py hysteresis
```

reports each label's real score distribution from your cached
detections, and proposes a `min_prob` at the 25th percentile. Whether
that is right depends on whether the bottom quarter was noise or was
real — look at output at two or three values before committing.

### `min_prob_continue` (per-label, per-backend) — score hysteresis

The **lower** threshold a detection needs to **continue** an
already-established track. `None` disables hysteresis entirely, which is
the default and is exactly the pre-v2.1.0 behaviour.

**The analogy.** A thermostat. If it started and stopped heating at the
same temperature it would chatter on and off around the set point, so it
starts at one temperature and stops at a slightly lower one. Same
problem, same fix.

**What you see.** Without hysteresis, a detection hovering near
`min_prob` crosses it back and forth frame to frame and the censor
blinks, even though what is being censored never went anywhere:

```
score:   .62  .48  .57  .46  .61  .59  .44  .58        min_prob = 0.50
         ───  ···  ───  ···  ───  ───  ···  ───
one gate  ON  OFF   ON  OFF   ON   ON  OFF   ON        ← flicker

                                    min_prob = 0.50, continue = 0.35
two gates ON   ON   ON   ON   ON   ON   ON   ON        ← steady
```

A detection below `min_prob` but at or above `min_prob_continue` may
extend a track that already exists. It can never **start** one — so a
low-confidence false positive still cannot appear out of nowhere.

**How to pick.** Somewhere between `global_min_prob` and the label's
`min_prob`. Too close to `min_prob` and it does nothing; too close to
`global_min_prob` and a track can coast on noise for a long time.
`betabench hysteresis` proposes the midpoint. Validation rejects a value
above `min_prob` (that inverts the mechanism) or at or below
`global_min_prob` (nothing that low ever reaches it).

---

## Geometry sanity filter

Four optional per-label bounds. All default to `None`, which means the
filter is **off** and cannot remove anything.

| Setting | Bounds |
|---|---|
| `min_area_fraction` | box area / frame area, lower bound |
| `max_area_fraction` | box area / frame area, upper bound |
| `min_aspect_ratio` | box width / box height, lower bound |
| `max_aspect_ratio` | box width / box height, upper bound |

**The analogy.** A coin sorter. Confidence asks "does this look like a
coin?" The geometry filter asks "is it coin-sized?" — a different
question, a cheap one, and it catches things the first question waves
through.

**What it is for.** A detector occasionally latches onto the wrong thing
at the wrong scale, and does so confidently. Confidence cannot separate
these from real detections. Shape can:

```
1920 x 1080 frame

┌──────────────────────────────────────────┐
│                                          │
│   ┌──┐                                   │   ← 40x40 = 0.0008 of frame
│   └──┘    real detection                 │      plausible
│                                          │
│   ┌────────────────────────────────┐     │
│   │                                │     │   ← 1200x800 = 0.46 of frame
│   │   "exposed_vulva", score 0.81  │     │      the model found a torso
│   │                                │     │
│   └────────────────────────────────┘     │
│                                          │
│   ▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭             │   ← 600x20, aspect 30:1
│                                          │      not a body part
└──────────────────────────────────────────┘
```

`max_area_fraction: 0.05` removes the middle one.
`max_aspect_ratio: 3.0` removes the bottom one.

**What you see.** Fewer large spurious censor blobs. The risk is
symmetric: a bound that rejects something you wanted is worse than no
bound at all, because it fails silently.

**How to pick.** There is no defensible universal default — the right
numbers depend on your footage's framing. Derive them:

```bash
python3 tools/bench/betabench.py geometry --frame-width 1920 --frame-height 1080
```

It reports the observed p1/p50/p99 per label from your cached detections
and proposes bounds widened past the tails. Start with
`max_area_fraction` alone: the "model found a torso" case is the one
this reliably catches.

Validation rejects bounds that cannot both be satisfied, since that
would make every detection of the label violate a bound without looking
broken.

### `geometry_action` — what a violation DOES

**This is the setting that makes the filter safe to turn on**, and it
exists because of a specific failure that took geometry out of service.

A `max_area_fraction` bound **cannot distinguish a correct close-up from
a torso misfire.** Both are simply large. When a violation meant "drop
the detection", the close-up got **no censor at all** — a worse outcome
than the misfire it was trying to prevent.

Measured with `exposed_breast`'s own suggested cap (0.123238, betabench's
p99 × 1.25) on a 1080p frame:

| case | area of frame | old behaviour |
|---|---|---|
| normal breast | 1.1% | kept |
| p99 breast | 10.3% | kept |
| **close-up breast** | **32.4%** | **dropped → uncensored** |
| torso misfire | 40.0% | dropped |

And it is the common case, not an edge one: on a real 10-file run the
largest detection of **every** censored label sat 2.0–4.1× above its own
suggested cap.

| action | what happens |
|---|---|
| `drop` | remove the detection. Default for **min** bounds — a 6px box carries no coverage worth keeping. |
| `clamp` | **default for max bounds.** Shrink the box to the limit, keeping its centre and aspect ratio, and censor that. |
| `flag` | keep the box unchanged and count it. For measuring a candidate bound before trusting it. |

```python
'exposed_breast': {
    'max_area_fraction': 0.123238,
    # the shipped default; spelled out here for clarity
    'geometry_action': { 'min': 'drop', 'max': 'clamp' },
},
```

A single string applies to every violation; the `{'min':…, 'max':…}`
mapping form exists because the right answer genuinely differs by
direction.

**The analogy.** A smoke alarm that cuts power to the whole house when it
smells smoke. It technically responds, but the failure mode is worse than
the fault it detects. Clamping is the alarm that just makes noise.

Under clamp a misfire becomes a roughly correctly-sized censor near the
right place, and a close-up is censored slightly small. Both beat
nothing. `too_small` and `too_tall` cannot be fixed by shrinking, so
under `clamp` they keep the box unchanged and let the count record it —
`clamp` means "never delete coverage", and deleting anyway would be
dishonest.

Counts are keyed `<label>:<reason>:<action>`, so the stats line says what
was done, not just what was found.

### Measuring a bound before you trust it

```bash
python3 tools/analysis/analyze_geometry_impact.py
```

This is the tool whose absence let a bad bound ship unnoticed. For each
candidate value it reports how many real cached detections violate it and
**where those violations sit relative to the bound**, which is what
separates a discriminating bound from a blunt one:

- violations clustered **far** past the bound (median 2×+) → it is
  catching outliers, which is what a sanity filter is for
- violations starting **just** past it and running continuously upward →
  it is cutting through the middle of your real distribution, and no
  action setting makes that a good bound

On the current 10-file corpus every candidate lands in the second
category (median 1.1–1.5× past the bound), which is why **no geometry
bounds are set in the shipped config.** The filter is now safe to enable;
the data does not yet justify a bound.

---

## Class suppression

```python
'class_suppression': {
    'exposed_breast': [
        { 'suppressed_by': 'covered_breast', 'min_iou': 0.55, 'margin': 0.10 },
    ],
}
```

Reads as: **`exposed_breast` is suppressed by `covered_breast` when a
`covered_breast` detection at the same instant overlaps it by at least
0.55 IoU AND outscores it by at least 0.10.**

Both thresholds are required. A missing `margin` would fall back to 0.0
— the loosest possible gate, not a neutral "off" — so validation rejects
a rule without one.

Rules are **per-backend**, resolved as
`detector_backend[<name>]['class_suppression']` →
`detector_backend['defaults']['class_suppression']` → `{}`.

> **Changed in 2.5.** The module-level `betaconfig.class_suppression`
> fallback was removed. Label vocabularies differ between models, so a
> shared ruleset can name classes a backend has never heard of, and an
> IoU threshold calibrated against one model's box geometry says nothing
> about another's. A stale top-level ruleset is now reported by
> `validate_config()` rather than silently applying to whichever backend
> happened to leave the key out.

**The analogy.** Two witnesses describing one object. If they are
looking at clearly different places, both can be right. If they are
looking at the same place, the one who is markedly more certain wins —
and *markedly* is what `margin` sets. IoU decides "same place"; margin
decides "clearly more certain".

**What you see.**

```
IoU 0.03 — two real adjacent things          IoU 0.85 — one thing, two labels
┌────┐                                       ┌──────┐
│    │┌────┐                                 │┌─────┴┐
└────┘│    │                                 └┤      │
      └────┘                                  └──────┘
  suppression should NOT fire                suppression SHOULD fire
```

Real overlap data is usually bimodal: a low-IoU cluster of genuinely
adjacent objects and a high-IoU cluster of same-spot confusion.
`min_iou` goes in the valley between them.

**The asymmetry to keep in mind.** Some suppressor labels
(`exposed_chest`, `exposed_armpits`, `exposed_anus`) are not in
`items_to_censor`. Firing a rule against one of those leaves the spot
**completely uncensored** rather than censored under another label. That
is a deliberate, accepted trade, and it is why several rules are tuned
stricter than the raw data alone would support.

**How to pick.**

```bash
python3 tools/bench/betabench.py suppression
```

reports the per-pair IoU and score-margin distributions and proposes
both thresholds. Add one pair at a time and look at the output: a
suppression rule removes real detections when it is wrong.

`margin` semantics validation enforces:

- `margin < 0` — **error**. It would let the suppressing label be *less*
  confident and still win, which inverts the entire premise.
- `margin == 0.0` — **warning**, not an error. It is sometimes the
  honest data-driven answer (when the suppressing label rarely
  outscores the suppressed one, a positive margin guts the rule rather
  than refining it), so a reviewed 0.0 can ship, but a copied-in one
  gets flagged.

---

## Class promotion

Suppression's inverse. Suppression *drops* a detection when a competing
label says it is something else; promotion *relabels* a detection when
corroborating evidence says it is something more specific.

**The analogy.** A second opinion. A covered_vulva detection on its own
is a diagnosis of "probably clothed". The same detection with an
exposed_penis laid across it is a different diagnosis, and promotion is
the rule that says which second opinions change the answer.

**The motivating case.** NudeNet reports penetration as `covered_vulva`,
because the penis is what is covering it. On the 640m caches
`covered_vulva` outnumbered `exposed_vulva` 1.29:1, and a quarter of
`covered_vulva` detections score below the label's own `min_prob`.
Lowering that `min_prob` catches them and every clothed crotch with
them. Promotion asks for corroboration instead.

```python
'class_promotion': {
    'exposed_vulva': [                       # the label a box BECOMES
        { 'from': 'covered_vulva',           # the label it was
          'min_prob': 0.30,                  # source score floor
          'requires': [                      # corroborating evidence
              { 'label': 'exposed_penis',
                'min_prob': 0.30,            # evidence score floor
                'min_source_overlap': 0.10 },# share of the SOURCE box covered
          ],
          'requires_mode': 'any',            # 'any' (default) or 'all'
          'duplicate_iou': 0.5 },            # see below
    ],
},
```

Per-backend, like `class_suppression`, and for the same reason.

**What happens to a promoted box.** Its label changes; its position,
size and score do not. It is then filtered, suppressed, tracked and
styled entirely as the target label. Promotion runs first, before the
geometry filter and suppression, so the target's own rules all apply.

**Two overlap measures, and why.** IoU divides by the union, so a size
mismatch drags it down even when one box fully covers the other:

```
small vulva box, long penis box across it

      ┌──┐
      │▓▓│  ← penis (long)
    ┌─┼──┼─┐
    │ │▓▓│ │ ← vulva: 100% covered
    └─┼──┼─┘
      │▓▓│
      └──┘
IoU ≈ 0.15          min_source_overlap = 1.00
```

`min_iou` measures "are these the same size and place"; that is the
right question for suppression and the wrong one here.
`min_source_overlap` asks how much of the *source* box the evidence
lands on. Give either or both; give neither and any overlap at all
counts.

**`'overlap': 'anywhere'` — co-presence in the frame IS the evidence.**

Both spatial measures answer *"are these the same object?"* Sometimes the
question is *"what kind of scene is this?"*, and then any spatial test is
the wrong instrument.

Measured on 9 real caches (488k detections, 9,302 `covered_vulva` above
0.3): only **18.9%** had any of `exposed_penis` / `exposed_vulva` /
`exposed_anus` live at the same instant — and of those, **91% had zero
pixel overlap with it.** An overlap-based rule promoted **1** detection
across the entire corpus. Requiring only co-presence promotes **1,452**.

```python
'requires': [
    { 'label': 'exposed_penis', 'min_prob': 0.30, 'overlap': 'anywhere' },
    { 'label': 'exposed_vulva', 'min_prob': 0.30, 'overlap': 'anywhere' },
    { 'label': 'exposed_anus',  'min_prob': 0.30, 'overlap': 'anywhere' },
],
```

**The analogy.** Asking whether it is raining by checking whether a
raindrop has landed on *your own shoe*. The drop on the shoe proves it;
its absence proves nothing. Looking out the window — is there rain
anywhere in view — is the question you meant.

`'anywhere'` is anywhere in **space**, not in time: evidence is still
same-instant only, and the evidence `min_prob` and `duplicate_iou` still
apply. It cannot be combined with `min_iou` or `min_source_overlap` —
validation rejects that rather than silently ignoring one of two
contradictory settings.

It is not `'always'`. On the same corpus **82.8%** of `covered_vulva`
detections had no qualifying evidence anywhere in frame and were not
promoted, which is the intended outcome: see below.

**`duplicate_iou`.** If the model has already found the object as the
target label (a real `exposed_vulva` overlapping the `covered_vulva` by
at least this IoU), nothing is promoted. Two boxes for one object means
two tracks and two styles.

**Additive by construction.** Promotion never removes a detection and
never moves one. A box is relabelled rather than copied because the
source label may itself be censored, and a copy would draw two censors
over one object.

**The log line is the tuning feedback.** Every run prints, per rule:

```
promotion exposed_vulva<-covered_vulva: 34 promoted, 5 already covered, 120 lacked evidence
```

Only source boxes that cleared the rule's `min_prob` are counted. Lots
of "lacked evidence" with no visible change means the evidence gate is
doing its job or is too tight; loosen `min_source_overlap` first. Lots
of promotions over clothing means the evidence is too loose; raise the
evidence `min_prob`.

**Two interactions worth knowing before tuning.**

- **The target's `min_prob` still applies.** A promoted box keeps the
  source's score, so a rule `min_prob` below the target's own `min_prob`
  promotes boxes that are then dropped. The log will report promotions
  and the render will show nothing.
- **Evidence that also suppresses the target defeats itself.** If
  `exposed_vulva` is suppressed by `exposed_anus`, a rule using
  `exposed_anus` as evidence promotes a box that the same overlap then
  deletes. Validation warns about this pairing. It is why the shipped
  rule uses only `exposed_penis`, which was deliberately removed from the
  vulva suppression rules.

**The shipped rule is a starting point, not a measurement.** None of
the six current preview caches contain a `covered_vulva` overlapped by an
`exposed_penis`, so these values have not been checked against real
penetration footage. Run a file that has it and read the log line.

Validation rejects unknown keys in a rule. That is deliberate: a
misspelled `requires` would otherwise mean "no evidence needed" and
promote every `covered_vulva`.

### covered_vulva and penetrative content

The concrete case the whole promotion mechanism was built for, and a
worked example of picking the right lever.

**The problem.** Penetration often reads to the model as
`covered_vulva` — the penis *is* covering it. So censoring
`covered_vulva` catches penetration. It also catches every ordinary
clothed crotch in the corpus, which is over-censoring.

**Why `min_prob` cannot fix it.** On the real 10-file run
`covered_vulva`'s score distribution is broad and ordinary: p5 0.250,
p25 0.439, median 0.599, p95 0.734. There is no threshold anywhere in
that range that separates "clothed crotch, leave it" from "penetration,
censor it", because **the score does not encode the distinction.** The
context does.

> Turning up `min_prob` here is deciding whether to bring an umbrella
> from how confidently you can see the sky. The confidence is not the
> question; whether there are clouds is.

**The shipped answer.** `covered_vulva` is **not** in
`items_to_censor`. Instead it is promoted to `exposed_vulva` when a
penetration-context label is present anywhere in frame:

```python
'class_promotion': {
    'exposed_vulva': [
        { 'from': 'covered_vulva', 'min_prob': 0.30,
          'requires': [
              { 'label': 'exposed_penis', 'min_prob': 0.30, 'overlap': 'anywhere' },
              { 'label': 'exposed_vulva', 'min_prob': 0.30, 'overlap': 'anywhere' },
              { 'label': 'exposed_anus',  'min_prob': 0.30, 'overlap': 'anywhere' },
          ] },
    ],
},
```

Measured across 9 real caches, 9,302 `covered_vulva` above 0.30:

| outcome | count | what the viewer sees |
|---|---|---|
| promoted | **1,452** | censored as `exposed_vulva` |
| already covered | 146 | a real `exposed_vulva` was already there |
| no evidence | **7,704** | clothed crotch, **not censored** |

The previous overlap-based rule promoted **1** of those 9,302. See
`'overlap': 'anywhere'` above for why.

**The tradeoff, stated plainly.** 7,704 detections that used to be
censored no longer are. That is the point — they are the clothed crotches
— but it is also the risk: a penetration moment where none of the three
evidence labels fires anywhere in frame is now uncensored where it
previously was covered. Widen the evidence set, or put `covered_vulva`
back in `items_to_censor`, if that trade goes the wrong way on your
footage.

---

## Cross-size dedup

```python
cross_size_dedup = { 'enabled': True, 'iou_threshold': 0.60 }
```

Collapses one real object detected at two different `picture_sizes`.

**Only ever compares detections from DIFFERENT sizes**, which makes it
provably a no-op when `picture_sizes` has one entry — the common case.

**Why the scoping matters.** Two same-label detections from the *same*
size are what a detector legitimately produces for two real instances
(two breasts, two people). Merging those would destroy real detections.

**What you see.** With multiple sizes and no dedup, the same object
produces one detection per size, both become censorable boxes, tracking
builds two separate tracks for one instance, and each independently
rolls its own randomised style — so you get two overlapping censors in
different styles on one thing.

**How to pick.**

```bash
python3 tools/bench/betabench.py dedup
```

A clean bimodal split means any threshold in the valley works.
Overlapping clusters mean the sizes disagree about geometry and the
threshold matters.

---

## Tracking and smoothing

Detections become *tracks*: per-instance identities followed across
frames. Within each timestamp, detections are matched to tracks
nearest-pair-first, so two simultaneous same-label detections always
land on two different tracks.

> The original implementation sorted every same-label box in the video
> by time and blended each toward whichever came immediately before it,
> with no notion of instance identity. On a frame with two boxes — both
> breasts on one person — one would routinely blend toward the other and
> drift to a point between the two real positions, tracking neither.

### `default_position_smoothing` (per-label: `position_smoothing`)

How much of each new detection's raw position is used. **0.65**. Range
0–1.

**The analogy.** Steering. 1.0 is a direct linkage — every twitch of the
detector reaches the wheels. 0.1 is heavy power steering — smooth, and
slow to respond.

**What you see.**

```
raw detections:   ●   ●     ●   ●    ●      ●   ●
                  (the detector jitters a few pixels frame to frame)

alpha = 1.0 (snappy)     ●   ●     ●   ●    ●      ●   ●  ← jitters with it
alpha = 0.65 (default)   ●   ●    ●   ●    ●     ●   ●    ← tracks motion
alpha = 0.1 (very smooth)  ●  ●   ●   ●   ●    ●   ●      ← lags real motion
```

Curved shapes (circle, ellipse) make small jitter far more visible than
a rectangle does, which is what this exists to smooth.

**The direction of this knob is the opposite of what the name suggests.**
"More smoothing" means a *lower* number, not a higher one: the value is
how much of the **new** position is used, so 1.0 is no smoothing at all.
If censor blocks feel like they lag behind the body they are covering,
**raise** this.

Measured on the 640m caches (`analyze_jitter --sweep-alpha`), for
`exposed_breast`: alpha 0.15 gives a smoothed median frame-to-frame jump
of 1.1px, alpha 0.70 gives 2.7px. The jump size is the responsiveness —
a box that moves 1.1px per frame while the subject moves more than that
is being left behind.

> **Changed in 2.5:** raised from 0.50 to 0.65, for censor blocks that
> keep up with movement. The cost is a little more visible jitter on a
> static subject; if that reads worse than the lag did, 0.55 splits the
> difference.

### Position and size smoothing

`position_smoothing` moves where the box **is**. `size_smoothing` moves
how **big** it is. They are the same kind of number on the same 0–1
scale, and for a long time they were the same number, which is what made
censors flicker.

#### `default_size_smoothing` (per-label: `size_smoothing`)

How much of each new detection's raw width and height is used. **Defaults
to `position_smoothing / 3`.** Range 0–1.

**The analogy.** A camera with image stabilisation. Panning to follow a
person is what the stabiliser is *supposed* to pass through — that is
position. The lens breathing in and out while you hold still is what it
is supposed to absorb — that is size. A stabiliser that treated both the
same would either fight your panning or let the breathing through, and
the second one is what BetaSuite did.

**What you see.** A motionless subject, with a detector whose width
wobbles a few pixels each frame:

```
raw detection width:   300  340  298  342  301  339  300
                       (the subject has not moved at all)

size_alpha = 0.65      300  326  310  331  313  330  313   ← edge moves every frame
  rendered edge:       |████|  |█████| |████| |█████|      ← reads as FLICKER

size_alpha = 0.22      300  309  306  314  311  317  313   ← edge nearly still
  rendered edge:       |████| |████| |████| |████|         ← reads as solid
```

**Why this is the flicker, and why it looked like a blur bug.** The
censor's edge is a hard boundary between processed and untouched pixels.
Move that boundary two pixels every frame and the eye reads it as
pulsing, whatever fills the box. Blur shows it worst because a blurred
edge against a sharp background is the highest-contrast boundary of the
lot, but stickers and bars flicker identically — which is the observation
that finally located the real cause after three wrong ones.

**Jitter vs flicker.** Worth separating, because the words get used
interchangeably and they are cause and effect. *Jitter* is geometry
moving: the box's position or size changing frame to frame. *Flicker* is
appearance changing: brightness or texture pulsing. Here the jitter
**causes** the flicker. Measured with a flat fill and no blur at all: a
constant-size box scored 0.00 levels of frame-to-frame change, a box
wobbling ±8px scored 15.9, and ±23px scored 62.0.

**Why lower than position is safe.** A censor slightly larger than
strictly necessary still covers the subject completely. One that breathes
reads as broken. The asymmetry is free: there is no coverage cost to
holding size steady, only to holding *position* steady, and position
keeps its own faster alpha.

**Tuning.** Raise it toward `position_smoothing` if censors seem slow to
grow when a subject moves toward the camera. Lower it if edges still
pulse on a static subject.

#### Two different symptoms, two different knobs

Blur and stickers fail differently, and telling them apart points at the
right setting.

| What you see | What is moving | What to change |
|---|---|---|
| Blur **strength** pulsing weak/strong, subject barely moving | the blur kernel | already fixed in code; see the clamp note under `type: 'blur'` |
| A shape's **edge** shimmering, strength constant | the box's position | `position_smoothing` (lower it) |
| The censor **breathing** larger and smaller | the box's size | `size_smoothing` (lower it) |

A sticker or bar has a hard outline, so it shows *position* movement
that a blur hides: a blurred edge is soft and a few pixels of drift
disappears into the gradient, while a sticker's border is a crisp line
against the frame and the eye tracks it precisely. Seeing edge shimmer
on stickers while blur looks stable is therefore not a contradiction, it
is the expected ordering.

Measured on a real 640m cache after tracking and smoothing, per frame
pair on `exposed_breast`:

```
size change (w):  median 1px   p90  3px    ← well damped by size_smoothing
position change:  median 3px   p90 11px    ← this is what stickers show
```

At the shipped values (`position_smoothing` 0.65, `size_smoothing`
0.217) size is held four times tighter than position. That asymmetry is
deliberate, since a censor that lags the subject is a coverage risk
while one that is slightly too large is not. The consequence is that
position jitter is the remaining visible motion on hard-edged styles.

If sticker edges bother you more than lag does, lower
`position_smoothing` toward 0.45 and re-check that censors still keep up
with fast movement. That is a genuine tradeoff, not a free win, and it
is the reason the default is not lower already.

> Splitting these two alphas had one non-obvious consequence. Position
> and size now move at different rates, so they can disagree about where
> the box *ends* even when every raw detection was inside the frame — on
> an edge-pinned box the overshoot measured up to 17px. That surfaced as
> a render crash rather than a visual glitch, because the feather mask is
> built from the box's own w/h while the region is sliced out of the
> frame, and numpy silently truncates a slice at the array edge. Tracking
> now re-clamps to the frame *after* smoothing. Nothing to configure;
> noted because the symptom (a `ValueError` about broadcast shapes, or a
> `bincount` length complaint from the hex style) looks nothing like its
> cause.

### `default_style_min_dwell_seconds` (per-label: `style_min_dwell_seconds`)

How long a censor style must have been on screen before a shot cut is
allowed to re-roll it. **0.0** — no minimum.

**The analogy.** A minimum-stay rule at a hotel. Guests may check out
whenever they like, but not before one night. Without it, a burst of
rapid cuts checks everyone in and out so fast that nobody is ever really
staying anywhere.

**The problem it solves.** Shot cuts re-roll the censor style, which is
what gives a compilation visual variety instead of one style for sixteen
different women. But cut-dense footage can cut several times a second,
and re-rolling that fast is its own flicker — the censor changes from
blur to mosaic to sticker faster than the eye can settle on any of them.

```
cuts:           |    |  |     |  |  |        |    |
dwell = 0.0:    A    B  C     D  E  F        G    H   ← 8 styles, strobing
dwell = 1.5s:   A       A     B        B     C        ← 3 styles, legible
                        ↑ cut ignored, style too young
```

**Interaction worth knowing.** A shot cut usually *ends* a track rather
than continuing it across the cut, so dwell has to be honoured in two
places: when an existing track crosses a cut, and when a brand-new track
resolves its first style. Only fixing the first did nothing measurable
(37 style changes became 36). With both, the same clip went to 11.

**Tuning.** Start at roughly the shortest time you want to actually look
at one style. 1.5s on compilation footage, 3.0s on single-scene footage,
are the shipped profile values. `0.0` restores the previous
re-roll-on-every-cut behaviour.

### `track_max_gap` (per-label, per-backend)

How long a track can go without a detection and still be continued by
the next one.

**Defaults** to whichever is larger of `2 / video_censor_fps` and that
label's `interpolation_max_gap`.

### `interpolation_max_gap` (per-label, per-backend) and `default_interpolation_enabled`

The longest gap that gets filled with synthetic in-between boxes.

**These two are not the same thing, and conflating them was a real bug.**

```
                  ┌─ detection ─┐         gap          ┌─ detection ─┐
                  ●                                    ●

gap <= track_max_gap?
   no  → the track HARD RESETS. A brand new, unsmoothed track starts at
         the raw position. interpolation_max_gap never gets consulted.
   yes → the track continues, and THEN:
           gap <= interpolation_max_gap?
              yes → synthetic boxes fill the gap
              no  → the last position is held until the next real hit
```

A `track_max_gap` tighter than the label's `interpolation_max_gap` makes
that interpolation tuning partly moot, which validation flags. Real
footage showed 88–97% of track resets were gap-blocked rather than
distance-blocked, which is what motivated the auto-derivation.

**What you see.** Too small: the censor jumps to a new position and
un-smooths on every brief miss. Too large: a track survives across
something it should not have, and the censor slides from one subject to
another.

### `match_distance_multiplier` (per-label, per-backend)

How far, in multiples of the larger box dimension, a detection can be
from a track and still be matched to it. **1.0**.

**What you see.** Raising it lets tracks survive fast motion — and lets
two nearby instances get mis-merged. Real data showed almost no resets
were distance-blocked (8 of 659 for `exposed_vulva`), so there is little
to gain here and a real cross-instance risk, which is why it is left at
1.0 for every shipped label.

### `paired_style`, `default_paired_style_max_distance`, `default_paired_style_tiebreak_margin`

When on, a newly-resolving box that has one unambiguously-nearest other
live track of the same label **shares that track's resolved style**
rather than rolling its own.

**What you see.** Without it, on a randomised multi-style label, one
breast gets a span-bar and the other gets a pixel mosaic — and the "span
both" style never gets anything to span.

`paired_style_max_distance` (**2.0**) is in multiples of the larger box
dimension. `paired_style_tiebreak_margin` (**0.25**) is how much closer
the nearest candidate must be than the second-nearest, as a fraction of
the second's distance, before the pairing is trusted. Two similarly
distant candidates are genuinely ambiguous — probably two different
people — and resolve independently.

> Proximity, not frame-exactness, is the test. Requiring both to be
> detected in the *same frame* used to be the gate, and against real
> cached footage it essentially never fired: 99.6% of independent style
> resolves had another live box of the label already present, and 99% of
> those visually mismatched it, because per-frame detection noise means
> two breasts very often do not both get a fresh detection in the same
> frame while both are continuously visible. Observed distances (median
> 0.09x box size, 99% under 1.0x) confirmed these were overwhelmingly
> the same pair.

---

## Track confirmation

### `default_min_track_hits` (per-label: `min_track_hits`)

Real detections a track must accumulate before **any** of its boxes are
rendered. **1**, which renders every track and is the pre-v2.1.0
behaviour.

**The analogy.** Waiting for a second opinion before acting. One
confident-sounding report might be wrong; two in a row rarely are.

**What you see.** A single-frame false positive currently paints a
censor blob for the whole `time_safety` window — at `time_safety: 0.35`
that is a blob appearing on nothing for a third of a second. At
`min_track_hits: 2` it never appears at all.

The cost is latency on every genuinely new appearance:

```
min_track_hits = 3, video_censor_fps = 9

detections:  ●    ●    ●    ●    ●
rendered:    ·    ·    ●    ●    ●
             └─ 2/9s ─┘
             uncensored while the track is being confirmed
```

`time_safety`'s backward window absorbs part of that, because a box's
`start` is `t - time_safety/2`. Validation flags values above 10 with
the real cost in seconds spelled out.

**How to pick.** 2 is a cheap, low-risk improvement on most footage. 3+
only if single-frame noise is a visible problem. Drop the boxes of an
unconfirmed track and its interpolated boxes go with it — a track that
never confirmed contributes nothing at all.

> **Changed in 2.5.** Both `nudenet_v3` labels moved from 3 back to 2.
> At 3, a track needs a third of a second of continuous detection at
> `video_censor_fps = 9` before rendering anything — which looks exactly
> like a missed detection on brief or partially-occluded content, and is
> easy to misdiagnose as a threshold problem. This is a recall-side
> change: expect slightly more short-lived false positives in exchange.

---

## Shot-cut detection

### `shot_cut_detection_enabled` and `shot_cut_threshold`

Finds hard cuts so tracking never smooths a box across a scene change.
**Enabled**, threshold **0.5** (Bhattacharyya distance, 0–1; higher is
less sensitive).

**What you see.** On quick-cut compilation footage, without this, a
tracked box can glide from one person in one shot to an unrelated person
in the next, because they happened to be close in time and screen
position:

```
      shot A                    cut                shot B
  ●────────────●                 │            ●────────────●
  tracked here                   │            different person,
                                 │            similar position
  without cut detection: ────────┼──────────→ the censor slides across
  with cut detection:    ────────┤ ╳          the track resets
```

A cut blocks both kinds of cross-time matching — continuing a track, and
`paired_style`'s nearby-live-track test — regardless of every distance
and gap setting, because all of those assume continuous single-scene
footage.

**Tuning.** Lower catches subtler transitions at higher false-positive
risk. A slow dissolve legitimately produces a short cluster of
consecutive cut timestamps rather than one; that is fine, a track
spanning the cluster is correctly reset either way.

Turn it off entirely if your material never has cuts: the scan is then
skipped completely, cache read included.

---

## Structure profiles

One set of timing values cannot serve every kind of footage, and for a
long time BetaSuite pretended it could.

**The analogy.** Shutter speed. A photographer shooting a still portrait
and one shooting a sprint do not argue about which shutter speed is
"correct" — they are photographing different things. Asking for one
`track_max_gap` that suits both a two-hour single-scene video and a
compilation that changes women every four seconds is the same category
error.

**The three footage types, and why they conflict.**

```
QUICK CUT                 SINGLE SCENE              SPLIT SCREEN
cut cut cut cut cut       |                         ┌─────┬─────┐
 |   |   |   |   |        one woman, four minutes   │  A  │  B  │
 A   B   C   D   E        she turns away at 0:38    ├─────┼─────┤
 new subject each time    and back at 0:41          │  C  │  D  │
                                                    └─────┴─────┘
                                                    four at once,
                                                    all the way through

needs: give up FAST       needs: hold on LONG       needs: stay SEPARATE
a track outliving its     a track that gives up     a generous match radius
cut keeps drawing a       at 0.2s drops the          merges two panes into
censor on whoever         censor every time she      one track and drags the
replaced her             turns, leaving her         censor across the
                         exposed                    divider
```

A `track_max_gap` short enough for the left column drops censors in the
middle one. One long enough for the middle column paints censors on the
wrong people in the left. Both are censoring failures, in opposite
directions, so no compromise value is safe for both. The right column
fails on a different axis entirely, spatial rather than temporal, which
is why it needs its own profile rather than a point between the other
two.

**How a profile is chosen.** Every video is measured from its shot-cut
scan, and the profile with the **highest threshold the measurement
clears** wins. Thresholds can therefore be written in any order, and
adding a third profile cannot silently reorder the other two.

`match_on` picks which measurement decides, and each signal reads its own
per-variant threshold key:

| `match_on` | threshold key | what it measures |
|---|---|---|
| `short_shot_fraction` | `min_short_shot_fraction` | share of RUNTIME in shots under 1s |
| `cuts_per_min` | `min_cuts_per_min` | cuts divided by runtime |

```python
'profiles': {
    'default':  'scene',            # used when there is no shot-cut data
    'match_on': 'short_shot_fraction',
    'variants': {
        'quick_cut': {
            'min_short_shot_fraction': 0.15,
            'video_censor_fps': 18,
            'item_overrides': {
                'exposed_breast': { 'track_max_gap': 0.20,
                                    'time_safety': 0.06,
                                    'min_track_hits': 2,
                                    'style_min_dwell_seconds': 2.5 },
            },
        },
        'split_screen': {
            'manual_only': True,     # see "Split-screen layouts" below
            'item_overrides': { ... },
        },
        'scene': {                   # the default and the catch-all,
            'item_overrides': { ... }  # so it carries no threshold
        },
    },
},
```

```
short-shot share:  0 ───────────── 0.15 ─────────────→ 1
profile chosen:    [     scene     ][    quick_cut    ]

split_screen sits outside this line entirely: manual_only, so
reached only by --profile or profile_by_path_pattern.
```

### Why the signal is a duration share and not a median shot length

The obvious measure is "how long is a typical shot", and it ranks real
footage **backwards**. Measured on the tuning set:

| file | median shot | share of runtime in shots < 1s |
|---|---|---|
| a genuinely fast-cutting video | 0.50s | **23.6%** |
| a video with one 2-second transition | 0.100s | 4.7% |
| another with a brief flurry | 0.150s | 2.9% |

The second file's cut detector fired on twenty consecutive samples inside
a single transition. Twenty 0.1s spans against one 41.8s span makes the
*median* 0.1s, because a median counts shots and does not care that
nineteen of them together occupy two seconds of a forty-five second
window. The genuinely fast-cut file has a **longer** median than both
false positives.

A profile buys a denser sample rate to fix shots too short to track
through. How much of the footage is made of such shots is a duration
question, so the signal measures duration. The two separate by 5x that
way: 23.6% against 4.7%.

**The analogy.** Rainfall. "The median minute this month had no rain"
is true of a month with one catastrophic flood, and it tells you nothing
about whether to build a levee. Total inches does.

**Both end spans count.** A shot is the span between two cuts, plus the
opening span from 0 to the first cut and the closing span from the last
cut to the end of the scanned window. Leaving the end spans out was the
same bug in another dress: a forty-minute file with two cuts one second
apart would look entirely composed of one-second shots, because the forty
minutes either side were never counted as shots at all.

**Tuning the threshold.** `min_short_shot_fraction: 0.15` sits between
the measured 4.7% and 23.6%, with margin on both sides. Check where your
own footage lands before moving it: the per-file `short_shot_fraction`
is written to every stats record.

**Profiles are the fourth override tier.** Per-label settings already
resolve through three, and a profile layers on top of all of them:

```
betaconfig.item_overrides            (shared across backends)
  └→ detector_backend[<b>]['item_overrides']
       └→ variant overrides
            └→ profile overrides        ← highest precedence
```

A profile is a **partial** override. It states only the keys it changes;
every other setting keeps whatever the three lower tiers resolved. A
profile that mentions `track_max_gap` does not disturb `time_safety`.

**Configured at the model level, not per variant.** Two variants of one
model differ in speed and accuracy, not in how they see footage
structure, so a 320n and a 640m of the same model share one profiles
block.

**Per-profile sample rate: `video_censor_fps`.** A profile may set its
own detection rate, so cut-dense footage is sampled more densely without
paying for it on every file.

```
median shot 0.2s   (measured: a real preview at 172 cuts/min)

at  9 fps:   ●  ·          1.8 samples per shot  → min_track_hits 2
                                                  is rarely reached,
                                                  the shot is never censored
at 18 fps:   ● ● ● ●       3.6 samples per shot  → confirms
```

The shipped `quick_cut` profile uses **18**, and also drops
`min_track_hits` to **2** for its labels. At 18 fps a 0.2s shot yields
about 3.6 samples, so requiring 3 hits leaves no headroom: one missed
detection and that shot renders nothing at all. Two hits keeps a margin.
The cost is that a single-sample false positive can now render, but at
these shot lengths it lasts about 0.1s.

`quick_cut` also cuts `time_safety` to **0.06**, down from 0.17. Temporal
padding that outlives the shot paints the censor onto whoever the cut
replaced, which is a censoring error in its own right, so the padding has
to shrink with the shots.

Detection time scales roughly with the rate, so a quick-cut file takes
about 2x as long to detect as at 9 fps. Every other file is unaffected,
which is the reason the rate is per-profile rather than global.

Three things follow from how this has to work:

- **The shot-cut scan always runs at the global `video_censor_fps`.** The
  profile is chosen *from* the scan, so letting the profile set the scan
  rate would be circular. Cut timestamps are times, so they stay valid at
  whatever rate detection then runs.
- **Frame-count rules scale with the rate.** Tracking reads the rate
  detection actually ran at, so interpolation step counts and the
  two-frame default `track_max_gap` shrink in seconds at a higher rate.
  Values you set in seconds (`track_max_gap`, `interpolation_max_gap`,
  `time_safety`) keep their meaning; at 15 fps an `interpolation_max_gap`
  of 0.14 now spans two samples rather than one.
- **A different rate is a different output.** The rate is in the filename
  and the detection key, so each profile's files get their own caches.
  When any profile sets a rate, a finished file is recognised after the
  (cached) shot-cut scan rather than before it. Repeat runs still skip
  detection entirely.

The analysis tools read caches at the global rate. `betabench` accepts
`--sample-fps 18` to read caches sampled at a profile's rate; the other
analysis tools do not yet, so they will not see those files.

### Split-screen layouts

A split-screen compilation (double, triple, quadrant) needs different
settings from either of the other two, and **cannot be detected
automatically**. This is a hard constraint, not a missing feature.

Profile selection runs **before** detection, because the profile sets
`video_censor_fps` and so decides how detection runs. The only
measurement available at that moment is the shot-cut scan. The signals
that separate split-screen from single-scene footage all need boxes:

```
signal                        separates split?   available pre-detection?
p90 simultaneous detections   yes (5-9 vs 1-2)   no, needs boxes
left-half detection share     yes (~0.5 vs 0.1)  no, needs boxes
cuts per minute               no                 yes
short-shot fraction           no                 yes
```

Measured, the two available signals interleave completely: split files
ran 0.1-3.0 cuts/min against scene files at 0.8-3.3. There is no
threshold to place.

So `split_screen` ships `manual_only: True`, which makes it invisible to
automatic selection while still reachable by name:

```bash
betatv.py --profile split_screen        # one run
```
```python
profile_by_path_pattern = {             # standing rule
    '*/splitscreen/*': 'split_screen',
}
```

**What it tunes, and why.**

| setting | value | reason |
|---|---|---|
| `match_distance_multiplier` | 0.8 | Tighter than the 2.0 default. Subjects in separate panes are distinct people at similar screen positions; a generous match radius merges two panes' subjects into one track and drags a censor between them. |
| `paired_style_max_distance` | 0.9 | Same logic for style pairing: a breast pair belongs to one body in one pane, not across the divider. |
| `style_min_dwell_seconds` | 6.0 | The longest of the three. Several subjects censored at once means several styles on screen at once, and re-rolling them on every cut is visual noise. A long dwell keeps variety without churn. |
| `track_max_gap` | 2.0-4.0 | Between quick-cut and scene. Panes are stable, but a subject can leave one pane while others continue. |

If you want this auto-selected, the only route is running selection again
after detection and re-rendering when the profile changes, which costs a
full detection pass on every file guessed wrong. Not implemented, on the
grounds that wall-clock time takes precedence.

**`default_profile_enabled`.** The master switch. `False` ignores every
profile and uses the un-profiled resolved settings — the behaviour before
profiles existed. Useful for isolating whether a result came from a
profile or from the settings underneath it.

**No shot-cut data is not the same as no cuts.** A video whose shot-cut
scan never ran (detection disabled, or a cached run from before scanning
existed) reports *nothing*, not zero, and takes the configured `default`.
A video that was scanned and genuinely has no cuts reports 0.0 and
selects whichever profile covers 0. Collapsing those two would silently
apply a measured profile to an unmeasured video.

> **One measurement subtlety worth stating, because the first
> implementation got it backwards.** The cut *rate* is cuts divided by
> the length of the window that was **scanned** — not by the spread
> between the first and last cut. Dividing by the spread reads structure
> inside out: two cuts 0.1s apart in an otherwise still 30-second slice
> scored 1237 cuts/min and selected the fast-cut profile, while the same
> two cuts at opposite ends of that slice scored 4 and selected the slow
> one. Identical footage density, opposite answers, and the wrong one
> confidently wrong. A quiet clip with one brief flurry is exactly the
> case that breaks it, and that clip is a scene, not a compilation.

That flurry case is also what rules the cut *rate* out as the shipped
signal, and a median shot length with it: see "Why the signal is a
duration share" above. `cuts_per_min` remains a supported `match_on` for
footage where it fits your material better.

**Tuning.** Run a representative file and read the `structure profile:`
line in the log, which reports whichever signal `match_on` selects on.
Check it against what you would have said by eye. If a file you consider
fast-cut reports a 3% short-shot share, the threshold is wrong for your
material, not the file. Every stats record also carries
`short_shot_fraction`, `median_shot_seconds` and `cuts_per_min`, so you
can compare all three across a corpus before moving a threshold.

### Interpolating vulva dropouts within a scene

A worked example of the tier trap, and of why the symptom pointed at the
right setting in the wrong place.

**The symptom.** In a penetrative scene, censoring holds for most of it
but drops out for **0.5 to 1.5 seconds** at a time, then comes back.
Reads as if a detection fell below a threshold for a moment.

**What it is not.** Not confidence: `exposed_vulva`'s `min_prob` is 0.18,
below its own p5 of 0.247, so nearly nothing is rejected on score. Not
shot cuts: measured across nine 45s slices, **zero** of the 82 vulva
detection gaps had a cut inside them. Not track confirmation:
`min_track_hits` of 3 costs 6 boxes out of 3,160 (0.2%).

**What it is.** The gap between consecutive real detections exceeded
`interpolation_max_gap`, so nothing bridged it. Measured rendered
coverage holes for `exposed_vulva`, by hole length:

| `interpolation_max_gap` | holes | 0.5–1.5s | 1.5–3s | >3s |
|---|---|---|---|---|
| **1.0 (was)** | **15** | **8** | 4 | 3 |
| 2.0 | 7 | 1 | 3 | 3 |
| 4.0 | 2 | 0 | 0 | 2 |
| **8.0 (now)** | **0** | 0 | 0 | 0 |

The 8 holes in the 0.5–1.5s band are exactly the reported symptom.

**The tier trap.** The label-level `item_overrides` said 3.6s, so that
looked like the value to raise. It was not the value in force. The
**scene profile** overrode it to 1.0, and every file in the corpus
selects the scene profile, so 1.0 was the effective number all along.
Raising the label tier would have changed nothing.

> Four tiers resolve per-label settings, and a profile sits on top of all
> of them. When a setting does not seem to do anything, check which tier
> actually owns it before changing the one you found first.

**Why a large value is safe here.** `cut_between()` blocks matching
across any recorded shot cut regardless of every gap and distance
setting. So the real ceiling on interpolation is the scene, not the
number: at 8.0s, **0 of 1,674** interpolated vulva boxes cross a cut, and
raising to 12.0 or 15.0 adds no coverage at all because cuts already stop
it. The plateau is the evidence that the constraint binds.

An interpolated box is also not invented coverage. It sits between two
**real** detections of the same object, in the same scene, with no cut
between them. The object was seen before and after.

**Why breast keeps a small value.** `exposed_breast` stays at 1.0 in the
scene profile. Breasts are detected far more densely (209,977 vs 46,666
detections on the same corpus), so its gaps are short and a long bridge
would mostly paper over tracking mistakes rather than real dropouts.

### Forcing a profile: `--profile NAME`

Any structure measurement is a proxy, and a proxy can be wrong about a
specific file. A compilation shot as long takes reads as a scene; a
single scene with fast intercutting reads as a compilation. And some
distinctions no pre-detection signal captures at all, split-screen being
the worked example above. When the measurement is the thing misleading
you, no threshold fixes it — you have to say which profile you meant.

```
betatv.py --profile quick_cut
betatv.py --profile split_screen
```

An explicit `--profile`:

- **beats the measured signal.** Selection returns that profile whatever
  the structure measurement says.
- **beats `default_profile_enabled = False`.** Typing a profile name and
  silently getting no profile because config disabled them is the same
  looks-like-it-worked failure the off switch itself was added to fix.
- **can name a `manual_only` profile** (below), which automatic selection
  never picks.
- **must name a real variant.** An unknown name fails validation with the
  valid names listed, rather than falling back to the auto-selected
  profile. A quiet fallback would render the wrong thing and report
  success.

It is in the censor key, so forcing a profile does not collide with a
cached render made without it.

### `manual_only`

A variant marked `'manual_only': True` is invisible to automatic
selection and reachable only by `--profile` or `profile_by_path_pattern`.
This is for a profile that should exist but that no available measurement
can correctly trigger. The shipped `split_screen` is exactly that case:

```python
'variants': {
    'scene':        { ... },                     # default and catch-all
    'quick_cut':    { 'min_short_shot_fraction': 0.15, ... },
    'split_screen': { 'manual_only': True,       ... },
},
```

`manual_only` removes the variant from **both** automatic roles — the
threshold match and the `default` fallback. Marking the default
`manual_only` means a file that matches no threshold gets no profile at
all, which is deliberate: a profile you have declared unreachable
automatically should not become the silent catch-all.

---

## Censor styles

A `censor_style` is either one style dict, or a **list** of them, from
which one is chosen at random — weighted by each entry's `weight`,
default 1 — the first time a track is seen, and then held for that
track's whole life.

```python
'censor_style': [
    { 'type': 'blur',  'method': 'gaussian', 'strength': 40, 'feather': 0.45, 'weight': 0.25 },
    { 'type': 'pixel', 'pattern': 'hex',     'strength': 25, 'feather': 0.29, 'weight': 1 },
]
```

**Why once per track and not once per frame.** A style resolved per
frame flickers between pixel and bar frame to frame. The same applies to
a sticker style's choice of image: `sticker_image` runs once per
*rendered frame*, so without a stable per-track seed a sticker style
flips through the whole folder like a flipbook.

### Common keys

| Key | Meaning |
|---|---|
| `type` | `blur` / `pixel` / `bar` / `sticker` / `debug` |
| `feather` | 0–1. Soft-edge width as a fraction of the shape's shorter side |
| `weight` | Relative pick probability within a list. Default 1 |
| `shape` | Overrides the item's shape for this variant only |
| `merge` | Overrides `censor_overlap_strategy` for this variant only |
| `width_area_safety` / `height_area_safety` | Overrides the item's padding for this variant only |

### `feather`

```
feather = 0                    feather = 0.3               feather = 0.6
████████████                   ░▒▓██████▓▒░               ░░▒▒▓▓██▓▓▒▒░░
hard cutout                    soft edge                  very soft
reads as a sticker             reads as intentional       may show detail
```

Feathering blurs the shape mask itself, so edge pixels blend
proportionally instead of jumping. Too much and real detail shows
through the gradient.

### `type: 'blur'`

`method`: `'gaussian'` (default) or `'box'`. `strength`: kernel size,
scaled by `censor_scale_strategy`.

`'box'` is a flat average over a square window and tends to look smeary
at the edges; `'gaussian'` weights the window's centre more and reads
smoother at the same radius. `'box'` is an option for the look, not the
speed — see `blur_fast_approximation` for the speed.

**Strength does not increase forever, and this is not a bug.** Two
separate ceilings meet here.

*Perceptual saturation.* A Gaussian kernel already wider than the region
it is blurring has nothing left to mix in. Everything inside the box has
already been averaged with everything else, so raising `strength` past
that point changes nothing visible:

```
region 200px wide

strength → kernel 51    ████▓▓▒▒░░    recognisable shapes, softened
strength → kernel 191   ▒▒▒▒▒▒▒▒▒▒    featureless
strength → kernel 400   ▒▒▒▒▒▒▒▒▒▒    identical to the above
```

*Edge reflection.* Once the kernel is larger than the region, OpenCV's
default `BORDER_DEFAULT` fills the overhang by mirroring the region's
own edge pixels back inward. The result is dominated by reflected edge
content rather than the region's interior, which reads as a flat wash
that can *shift* frame to frame as the box moves — the opposite of what
a stronger blur is supposed to buy.

BetaSuite therefore clamps the kernel to a fraction of the region:
`1.0` of the shorter side for `gaussian`, `0.5` for `box` (box blur
degrades sooner, being a flat average). The clamp is per-region, so a
large censor still gets a large kernel; only kernels that would exceed
their own region are trimmed.

**If blur does not look strong enough,** raising `strength` past the
ceiling will not help. The things that do: `censor_scale_strategy`
(`'feature'` scales the kernel to the detection, `'frame'` to the video,
which is what you want if small detections look under-blurred), a
`pixel` style at a coarse block size, or an opaque `bar`. A blur that
looks weak on a *small* region is usually a scaling question, not a
strength question.

#### `method: 'triple_box'` — a cheaper near-Gaussian

Three passes of a box blur. Repeated box convolution converges to a
Gaussian (central limit theorem), and by the third pass the shape is
within a few percent of one. `cv2.blur` is a running-sum filter whose
cost per pixel does not grow with the window, so three box passes beat
one true Gaussian at the kernel sizes censoring uses.

Measured at matched obscuring power on a 400x400 region of synthetic
detail:

| method | kernel | time | mean abs. difference from exact |
|---|---|---|---|
| `gaussian` (exact) | 101 | 12.53 ms | — |
| `gaussian` (approximated) | 101 | 1.35 ms | 0.65 / 255 |
| `triple_box` | 101 | 1.02 ms | 0.19 / 255 |

So it is both closer to a true Gaussian than the downscale
approximation and slightly faster. Unlike the approximation it never
resamples, which is why it is the fast path that works with
`blur_edge_margin` below.

**"Slightly faster" understated it.** That table times the blur call on
one synthetic region. Measured through `blur_image` instead — the real
path, with the edge margin and the feather mask, at the box sizes and
strengths a tuned config actually uses (breast boxes: median 74x79, p90
177x176, max 187x210) — `triple_box` is **5.9x faster overall**, and the
gap widens with strength because the margin work scales with the kernel:

| box | strength | `gaussian` | `triple_box` | speedup |
|---|---|---|---|---|
| 74 | 30 | 0.496 ms | 0.118 ms | 4.2x |
| 74 | 60 | 2.746 ms | 0.236 ms | 11.7x |
| 177 | 40 | 4.509 ms | 1.025 ms | 4.4x |
| 177 | 60 | 8.741 ms | 1.080 ms | 8.1x |
| 210 | 60 | 13.624 ms | 1.349 ms | 10.1x |

And the outputs are indistinguishable at those settings: mean absolute
difference 0.05–0.50 levels, 99th percentile 1–2, **worst single pixel
anywhere 3 out of 255**. So for blur on large boxes at high strength,
`triple_box` is the default worth reaching for and `gaussian` is the
reference implementation to check it against.

#### Is `triple_box` redundant now?

A fair question once `blur_fast_approximation` is on, since that already
makes `gaussian` cheap. Three findings, measured rather than assumed:

- **Against `gaussian`, visually yes, but it is the faster one.** At
  matched kernels in the real range (k=11–149) the two differ by ≤3/255.
  Removing `triple_box` would not change how output looks; it would just
  make it ~6x more expensive to produce.
- **Against `box`, no.** Plain `box` differs from Gaussian by 0.6–2.2
  levels mean at the same kernel — a visibly flatter, boxier blur. It is
  the cheapest of the three and the right pick for small regions
  (the vulva labels, boxes around 23–40px), which is what it is used for.
- **`triple_box` is the only strictly monotonic one.** Averaged over
  eight fixtures across k=21–401, both `gaussian` and `box` show small
  non-monotonic steps (+2% to +16%) above k≈270, where residual detail is
  already down at 0.02–0.05 and the wiggle is invisible. Not the
  `stackBlur` failure mode (that reversed 11 → 72, a 6x rise in *visible*
  detail), so this is a curiosity rather than a reason to prefer one.

So all three earn their place: `box` for small regions, `triple_box` as
the working default for large ones, `gaussian` as the reference.

**Rejected alternative: `cv2.stackBlur`.** Fast, and available in
OpenCV 4.7+, but its blur strength is **not monotonic** in kernel size at
these sizes. Residual detail on the same fixture: 11.3 at k=31, 15.0 at
k=101, 72.8 at k=401 — past roughly k=31 a *larger* kernel obscures
*less*. A censor style whose strength silently reverses above a
threshold is the wrong shape of risk whatever it costs.
`tests/test_blur_methods_and_margin.py` re-measures that claim so it can
be revisited on a future OpenCV rather than taken on trust.

### `blur_edge_margin`

Whether a blur reads a margin of real neighbouring pixels around the box
before blurring. **True.**

**The analogy.** Asking about a neighbourhood. A blur averages a
pixel with its surroundings, and at the edge of the censor box those
surroundings run out. Without a margin OpenCV invents them by mirroring
the box's own edge back inward, so the answer depends on exactly where
the edge was drawn. With a margin it asks the actual neighbours, who do
not move when the box resizes.

**What it fixes.** The blur flicker that survived stabilising the
kernel. Measured with the box jittering +/-4px and the kernel held
perfectly constant, mean level swing inside a fixed crop:

| kernel / region | no margin | with margin |
|---|---|---|
| 0.30 | 1.31 | **0.00** |
| 0.51 | 2.19 | **0.00** |
| 0.76 | 2.74 | **0.00** |
| 0.99 | 3.04 | **0.00** |

Zero, not merely smaller. The size dependence is removed rather than
reduced.

**It cannot change what is censored.** The margin is read only; what
gets composited back is exactly the box. A test asserts every pixel
outside the box is byte-identical.

**Cost.** The blurred region grows by kernel/2 on each side. For a
200px box at kernel 101 that is 2.3x the pixels: exact Gaussian goes
1.25x slower, `triple_box` 2.4x. `triple_box` with a margin still costs
about 0.7ms against the exact Gaussian's 6.5ms without one.

**Interaction with `blur_fast_approximation`.** The margin wins. The
approximation resamples on a grid derived from the region's size, which
is the size dependence the margin exists to remove, so it is skipped
while the margin is on. `triple_box` is the fast path that is compatible
with a margin; choose it if the exact Gaussian is too slow for your box
sizes.

#### Would a weaker blur flicker less?

It depends which path is running, which is why this needs measuring
rather than a rule of thumb. Mean level swing, box jittering +/-4px:

| strength | kernel | `gaussian` exact | `gaussian` approximated | margin on |
|---|---|---|---|---|
| 20 | 41 | 0.64 | 0.64 | 0.00 |
| 60 | 121 | 2.46 | 1.55 | 0.00 |
| 120 | 199 | 3.04 | **0.35** | 0.00 |
| 200 | 199 | 3.04 | **0.35** | 0.00 |

On a true Gaussian, yes: weaker is steadier, monotonically. On the
**approximated** path the relationship inverts, because a bigger kernel
means a coarser downscale and the coarser resample averages away the
border sensitivity that causes the swing. Lowering strength to chase
flicker there makes it worse.

With `blur_edge_margin` on the question is moot at every strength, so
strength can be chosen purely for how much it obscures.

`tools/tuning/sweep_blur_strength.py` renders one preview slice at
several strengths so the obscuring judgement can be made against real
footage, and reports the predicted swing per strength alongside. Each
output filename carries a readable variant tail
(`-gaussian-s120-p065`) on top of the cache keys, and `--position-smoothing`
sweeps that setting too, because once the margin removes the size-driven
component, box MOVEMENT is what remains.

#### The clamp is measured against the track's reference size

The kernel is clamped against the size the track resolved once, not
against this frame's box. That distinction is the whole reason blur
stopped pulsing, and it is worth stating because the wrong version looks
more correct.

Both the strength scaling and the clamp read the stable reference. Only
the first half of that used to be true: the kernel was computed from the
stable size and then re-clamped against the live box, which jitters. The
clamp handed back exactly the per-frame variation the stable size had
just removed.

```
box jitters a few px
  └→ kernel re-clamped against the LIVE box
       └→ kernel changes frame to frame
            └→ sigma changes
                 └→ downscale = int(sigma // 4) STEPS, 10 → 9
                      └→ the region resamples on a different grid
                           └→ visibly weaker, then stronger, blur
```

The downscale factor is an integer, which is why this reads as a step
between strong and weak rather than a gentle drift.

**It was silent at low strength and bit hard at high strength**, because
the clamp only binds once the kernel approaches the region's size.
Measured on a real 640m cache, `exposed_breast`, 21 tracks:

| strength | median kernel swing | tracks that changed downscale |
|---|---|---|
| 60 | 0px | 0 of 21 |
| 100 | 4px | 8 of 21 |
| 140 | 18px | 16 of 21 |

Raising `strength` is precisely what someone does when they want a
stronger censor, so the defect punished the setting most likely to be
turned up.

**Why the safer-looking fix is not the fix.** `min(reference, live)`
seems more conservative and does not work: the live box is smaller than
its track's reference on 52% of boxes (median 0.89x), so the live term
keeps winning and keeps jittering, leaving 7 of 21 tracks unstable.
Clamping only ever *reduces* a kernel, so declining to clamp against the
live box can only blur more, never less. It cannot under-censor, which
is the direction that matters.

The cost is real but small: a box much smaller than its track's
reference gets a kernel larger than its own region, which reads as a
flat wash rather than as recoverable detail. That is a look tradeoff in
the safe direction.

`tests/test_blur_kernel_stability.py` pins this by observing the kernel
the real `blur_image` uses, rather than recomputing the arithmetic. An
earlier version of that file recomputed it, agreed with itself, and
passed cheerfully with the bug put back.

## Render-time motion interpolation

### The flicker that no blur setting could fix

Detection samples at `video_censor_fps`; the render writes **every**
source frame. At 9fps detection on a 120fps source that is 13.3 output
frames per sample. Two consequences, and the second is the one that
mattered:

1. **Held geometry.** Each box's rectangle was whatever its sample said,
   held until the next sample replaced it, so the box teleported. Measured
   on a real cache: 5.1px median jump, 14.1px max, nine times a second.

2. **Overlapping samples.** `time_safety` makes a box live for *longer*
   than the sampling interval (0.17s against 0.111s, 153% coverage), so on
   most output frames **two** consecutive samples of the same object are
   live at once. Measured on real footage: 54.5% of frames. The blur
   overlap strategy unions them, so the rendered rectangle grew and shrank
   as the overlap came and went: `78x86, 78x87, 77x86, 79x91` on
   consecutive frames, oscillating at the sample rate.

(2) is the pulsing, and it explains why strength, method, `blur_edge_margin`
and `position_smoothing` all failed to touch it. Those change what *fills*
the box; this moves the box's *boundary*. `position_smoothing` smooths
between **samples**, not between **output frames**, so it cannot see this
at all.

### `render_motion_interpolation`

`True` (default) slides each box's geometry linearly between its own
adjacent samples, so it moves at output frame rate instead of jumping at
sample rate. Never extrapolated: before a track's first sample or after
its last, the box holds, which is the old behaviour.

### `render_motion_max_span_seconds`

Longest sample-to-sample gap to blend across. `None` (default) means two
sampling intervals. Across a longer hole the object genuinely was not
seen, and sliding through it would draw a censor along a path nobody
detected, so the box holds instead.

### `render_motion_collapse_max_growth`

Default `1.15`. When two samples of one object are live on the same frame,
collapse them to one box if their union is no more than this multiple of
the larger box's area. `1.0` disables the collapse.

The collapse is what actually removes the pulsing — sliding alone does
not, because both boxes stay live and the union still oscillates. But it
is **not free**, and an earlier version of this document claimed it was on
the strength of a single-track measurement. Measured properly with pixel
masks over a real 120fps/9fps cache, collapsing gave up **531,339 pixels**
that the old behaviour censored, **34% of them inside the nearest real
detection** rather than union margin, up to 4,055px on one frame.

The gate exists because the cost is not uniform. When two live samples
nearly coincide their union is a couple of pixels wider than either, and
collapsing loses nothing. When the object is moving fast they straddle the
motion and the union covers the whole swept path — a real safety margin.
So the collapse applies only where it is nearly free, which is also
exactly the near-still case where the eye notices pulsing; during fast
motion the motion itself masks it.

### `render_motion_size_window_seconds`

Default `0.25`. Holds each track's rectangle at the largest size it was
sampled at within this many seconds either side, centred where the
interpolation put it. `0` means the whole track; `None` disables it.

This is what pays back the collapse's coverage cost, because a box sized
to its track's maximum is never smaller than any sample in range. It is a
straight dial, measured over the whole real slice against the old held
behaviour:

| `size_window` | lost detected px | worst frame | area step p99 | censored area |
|---|---|---|---|---|
| `None` (off) | 531,339 | 4,055 px | 151 | −1.7% |
| **`0.25`** | **352,260** | **2,521 px** | **126** | **−0.0%** |
| `0.5` | 298,985 | 2,521 px | 80 | +1.3% |
| `1.0` | 219,475 | 2,488 px | 72 | +4.0% |
| `2.0` | 153,016 | 2,111 px | 68 | +7.4% |
| `0` (whole track) | **48,652** | **2,089 px** | **0** | **+20.2%** |

`0.25` is the default because it recovers a third of the lost coverage for
no measurable extra area. The whole track is the best coverage available
and the only setting that takes the residual size pulse to **zero**, but
20% more blurred area is a judgement about the footage rather than a
number, so it is opt-in rather than the default.

The window matters for a second reason: a subject who genuinely changes
size — one approaching the camera — would have a whole-track maximum hold
the largest size for the entire track.

### Why these are in the censor key

All four change rendered pixels, so all four are in `censor_identity()`.
A censored video cached under one value is not served for another. Every
video rendered before this feature existed will correctly miss the cache
and re-render once.

### `type: 'pixel'`

`pattern`: `'square'` / `'mosaic'` (the same technique, two common
names) or `'hex'`. `strength`: block size in source pixels.

```
mosaic                          hex
┌──┬──┬──┬──┐                   ⬡ ⬡ ⬡ ⬡
├──┼──┼──┼──┤                    ⬡ ⬡ ⬡ ⬡
├──┼──┼──┼──┤                   ⬡ ⬡ ⬡ ⬡
└──┴──┴──┴──┘                    ⬡ ⬡ ⬡ ⬡
```

A **lower** `strength` means **less** obscuring (smaller blocks). Real
testing showed low values letting too much detail through.

> The hex grid was rewritten in v2.1.0 from a per-cell mask-and-scan
> loop into one analytic labelling pass: 21–175 ms per box per frame
> down to about a millisecond. It is also a correct non-overlapping
> tessellation now, where the old generator produced overlapping
> hexagons that partially overwrote each other. The rendered pattern is
> therefore slightly different: regular where it used to be subtly
> irregular.

### `type: 'bar'`

`color` as `(r, g, b)`. `thickness` as a fraction of the box height,
centred within it.

```
thickness = 1.0            thickness = 0.35          thickness = 0.175
┌──────────┐               ┌──────────┐              ┌──────────┐
│██████████│               │          │              │          │
│██████████│               │██████████│              │▬▬▬▬▬▬▬▬▬▬│
│██████████│               │          │              │          │
└──────────┘               └──────────┘              └──────────┘
fills the box              reads as a censor bar     risky (see below)
```

**The bouncing-exposure trap.** A box's rendered position is static for
its whole `time_safety` window. A thin bar covers only a fraction of the
box height, so real motion during that static window can briefly expose
skin above or below the bar. Confirmed against real footage at
`thickness: 0.3`. If bars start exposing skin during tuning, the thin
end of the range is the first thing to check.

`span_extend` (0–1) reaches past the tight two-box rectangle toward the
frame edges when `merge: 'span'` is active. 0.0 is detection-to-detection;
1.0 is edge to edge. Horizontal only.

### `type: 'sticker'`

`dir` (globbed for `*.png`) and/or `images`. `scale` (default 1.0).

PNGs with an alpha channel give a clean silhouette; without one the
sticker is composited as an opaque rectangle. An empty or unreadable
folder falls back to a plain black bar rather than failing the run, and
logs once.

OpenMoji and Twemoji both publish plain PNG emoji sets that drop
straight in.

### `censor_overlap_strategy`

What happens when two same-label, same-style boxes overlap. Per style
**type**; a style dict's own `merge` overrides it per variant.

| Strategy | Behaviour |
|---|---|
| `'none'` | Each box censored independently |
| `'single-pass'` | Overlapping boxes merged into one covering rectangle |
| `'span'` | Exactly two boxes bridged into one rectangle, overlapping or not |

```
'none'                'single-pass'            'span'
┌───┐ ┌───┐           ┌─────────┐              ┌───┐    ┌───┐
│ A │ │ B │           │  A + B  │              │ A │────│ B │
└───┘ └───┘           └─────────┘              └───┘    └───┘
                      (only if overlapping)    (as if a line were drawn)
```

`'span'` is restricted to exactly two boxes. Three or more same-label
boxes in one frame is far more likely two people than one person with an
unusual body-part count, and spanning a bar between two people reads as
one region — worse than doing nothing.

All three only apply when the shape is `'box'`.

### `censor_scale_strategy`

How `strength` is interpreted.

| Value | Multiplier | Meaning |
|---|---|---|
| `'feature'` (default) | `min(box_w, box_h) / 100` | scales with the censored thing |
| `'image'` | `max(img_h, img_w) / 1000` | scales with the frame |
| `'none'` | 1 | raw pixel count |

**What you see.** With `'feature'`, the same `strength` reads as
similarly strong whether the subject is close to camera or far away.
With `'none'`, a distant subject gets over-censored relative to its size
and a close one under-censored.

### `blur_fast_approximation` and `blur_approximation_min_kernel`

**True** and **21**. Heavy Gaussian blurs are computed at reduced
resolution: downsample, blur with the proportionally smaller sigma,
upsample.

**What you see.** Nothing, at obscuring strengths — the detail the
approximation loses is exactly the detail the blur exists to destroy.
Measured against the exact kernel:

| Region | Kernel | Exact | Approximate | Speedup |
|---|---|---|---|---|
| 200x200 | 101 | 6.73 ms | 0.11 ms | 61x |
| 300x300 | 121 | 13.52 ms | 0.41 ms | 33x |

Kernels below `blur_approximation_min_kernel` are never approximated:
the exact blur is already cheap there, and approximating a small kernel
is visible.

Turn it off to compare:

```bash
python3 tools/bench/betabench.py render --no-blur-approximation
```

It is part of the censor cache key, so switching it produces a distinct
output filename rather than overwriting.

---

## Area safety and time safety

### `default_area_safety`, `width_area_safety`, `height_area_safety`

Fractional padding around the detection box. Negative shrinks.

```
area_safety = 0          area_safety = 0.5        area_safety = -0.3
┌────────┐               ┌──────────────┐          ┌─────┐
│ detect │               │  ┌────────┐  │          │┌───┐│
│  -ion  │               │  │ detect │  │          ││det││
└────────┘               │  └────────┘  │          │└───┘│
                         └──────────────┘          └─────┘
exact box                25% wider each side       censors less than detected
```

Settable at three levels, most to least specific: the **style** variant,
the **item**, then `default_area_safety`. That lets a randomised style
list tune how generously each variant pads, the same way it tunes
strength and feather.

**What you see.** Positive values cover more, at the cost of censoring
things next to the detection. Negative values are used with elliptical
and bar styles where the box's corners are empty anyway.

### `default_time_safety`, per-label `time_safety`

How long a detection's censor persists, centred on the detection:

```
                    detection at t
                          │
   ├──────────────────────┼──────────────────────┤
   t - time_safety/2      t      t + time_safety/2
   └──────────── the box is rendered ────────────┘
```

**What you see.** Larger values cover motion between samples, at the
cost of the censor lingering after the thing has gone and appearing
before it arrives. **0.30** default.

The backward half is why track confirmation costs less latency than it
looks: by the time a track confirms, its first box's window has already
started.

**How to pick it: the floor is the sampling interval.** A box has to
stay up until the next sample, or consecutive detections leave an
uncensored flicker between them. At `video_censor_fps = 9` that interval
is 111ms, so `time_safety` below ~0.11 opens gaps. Everything above that
floor is buying motion tolerance and paying for it in bleed:

| `time_safety` | bleed past a true edge | overlap between samples |
|---|---|---|
| 0.35 | 175 ms each side | +239 ms |
| 0.30 | 150 ms | +189 ms |
| 0.22 | 110 ms | +109 ms |
| 0.18 |  90 ms |  +69 ms |
| 0.11 |  56 ms |    0 ms — no margin for a dropped frame |

> **Changed in 2.6:** `exposed_breast` 0.35 → 0.22 and `exposed_vulva`
> 0.27 → 0.20, reported as censoring that appears before an exposure and
> persists after it. Both keep a full sampling interval of overlap, so
> neither opens a gap; what they give up is tolerance for a subject that
> moves a long way between two samples. If censors start flickering
> mid-scene rather than bleeding past the edges, that is this change
> going too far and the values want to come back up, not down.

The window is symmetric, so it cannot be tuned for "appears too early"
and "persists too late" independently — one number moves both edges. If
only one side is wrong, the cause is more likely a track coasting past a
shot cut (see `shot_cut_detection_enabled`) than this setting.

---

## Rendering and encoding

### `render_workers`

How many render chunks encode at once. **0** means auto: CPU count / 2,
capped at 8, never more than the number of chunks.

**Why CPU/2.** Each worker runs both a Python compositing loop and an
x264 encoder. Oversubscribing makes every chunk slower without
finishing any sooner. Encoder threads are divided between workers
automatically.

**What you see.** On an 8-core box with `render_chunk_seconds: 180`, a
one-hour video splits into 20 chunks and finishes in roughly the
wall-clock of three, rather than of twenty.

The histogram `summarize_run.py` prints is how you check this is really
happening. "1 worker x5 file(s), 2 worker(s) x2 file(s)" means the
parallel renderer is idle, and no `render_workers` value fixes that —
the chunk size is what decides how many chunks exist to hand out.

### `render_chunk_seconds`

Seconds of video per render chunk. **180**. `0` disables chunking.

**Why 180 and not 600.** A file is split into
`ceil(duration / render_chunk_seconds)` chunks, and one worker renders
one chunk — so a file shorter than `2 x render_chunk_seconds` can never
use more than two workers, whatever `render_workers` says. At 600, a
library averaging six and a half minutes per file produced nine chunks
across seven files: five files ran on one worker, two on two, and the
rest of the cores sat idle through 84% of that run's wall clock.

180 gives a typical file three or four chunks, which is enough to keep
several workers busy without paying concat overhead on dozens of
fragments. The genuinely optimal value depends on your core count and
your typical file length, and it is a measurement rather than a
constant: `summarize_run.py` prints the worker histogram a run actually
achieved, which is the number to tune against.

**Why chunks exist.** A single continuous ffmpeg pipe cannot be resumed
if it is killed — you cannot append more encoded frames onto a truncated
file and get something valid. Each chunk is rendered to its own file and
only promoted to its trusted name once ffmpeg exits 0, the container is
readable, **and** the frame count matches. A restart picks up from the
first missing chunk.

Smaller chunks mean finer-grained resume and more parallelism, at the
cost of more concat work. Preview runs always use one chunk.

### `render_verify_frame_counts`

**True**. Compare frames actually written against frames planned.

> Before v2.1.0 a chunk whose decode died mid-file was a perfectly valid
> video file that was simply too short. It passed the container check,
> got promoted, and a later resume skipped it — silently truncating the
> output. The container check cannot see this; only the count can.

The final chunk is allowed to come up short, because container metadata
over-reports the frame count on some files. That is logged rather than
failed.

### `encode_video_codec`, `encode_crf`, `encode_preset`, `render_chunk_container`

`'libx264'`, **17**, `'fast'`, `'mkv'`.

CRF is quality: lower is better and larger, 17 is visually lossless for
most material, 0 is truly lossless, 51 is worst. Preset is the
speed/size trade: `ultrafast` through `veryslow`, same quality target,
bigger files at faster presets.

> v2.1.0 removed an entire encode. The pipeline used to write
> `mpeg4 -qscale 1` chunks, concatenate them, and then re-encode the
> whole video to H.264 — a full extra encode of every output and two
> generations of lossy compression. Chunks are now encoded directly as
> H.264 at these settings, concatenated by stream copy, and the audio is
> muxed by stream copy. One encode, one generation of loss.

If the source's audio codec cannot live in an MP4 (Vorbis or Opus in a
Matroska source), the copy fails and the mux retries with AAC. That is
logged, because re-encoding audio is a real quality decision.

### `detection_checkpoint_frames`

Samples between detection checkpoints. **500**. `0` disables.

A detection pass over a long video can take hours. Without
checkpointing it wrote its cache once, at the very end, so a kill at 99%
lost everything. Checkpoints go to a separate `.checkpoint` file, never
the trusted cache path.

### `ffmpeg_max_retries`, `ffmpeg_retry_backoff_seconds`

**2** and **5**. Additional attempts after the first, and the pause
between. A deliberate interrupt is never retried.

### `input_delete_probability`

Chance of deleting each **source** file after its censored counterpart
is confirmed on disk. **0**.

Non-zero requires typing `DELETE MY FILES` at startup. Deletion only
happens once the censored output exists and is non-trivially sized.

---

## Preview mode

```python
preview_mode_enabled = False
preview_max_seconds = 20
preview_encode_preset = 'ultrafast'
preview_start_seconds = None
preview_random_slice = False
```

Processes a slice of each video instead of the whole thing. Preview
output gets its own filename suffix, its own detection cache key, and is
never skipped for an existing output — you are expected to re-run it
every time you change a setting.

**`preview_start_seconds`** pins where the slice starts. Use the **same
value** across runs to keep comparing the same footage. The offset is
applied as a direct seek, so nothing before it is decoded.

Two edge cases, both handled per file because one setting is applied
across a directory of differently-sized videos:

- Start past the end of *this* video → that one file is processed **in
  full**, rather than clamped to a slice you never asked for.
- Start too late to fit the full window → backed up to the latest start
  that fits, with a warning.

**`preview_random_slice`** picks a random offset per file when
`preview_start_seconds` is unset, and logs the offset so you can pin it.

The shot-cut scan is bounded by the preview window. Before v2.1.0 a
20-second preview of a two-hour file still paid for a two-hour
histogram scan.

---

## Logging and stats

```python
logging_enabled = True
log_path = '../output/logs/betasuite.log'
log_level = 'debug'      # the log FILE
console_level = 'info'   # the TERMINAL
stats_enabled = True
stats_path = '../output/stats/betasuite_stats.jsonl'
```

Levels: `trace` < `debug` < `info` < `warn` < `error`.

Two independent levels so the file keeps the detail while the terminal
stays readable. `trace` is for per-frame and per-box detail that would
drown a debug log.

> Before v2.1.0 the detection and render loops each printed once per
> frame with a carriage return. On a terminal that is a redraw; piped to
> a log file it is one more copy of the whole line, which is how a
> 90-second video produced a 16 KB single-line log. Progress is now
> rate-limited: an in-place line on a tty, and a bounded number of real
> log records either way.

Set both to `trace` when something is going wrong and you want
everything:

```bash
python3 betatv.py --log-level trace --console-level warn
```

### Stats

One JSON-Lines row per processed file, carrying the run's identity as
well as its timing: backend, variant, picture sizes, batch size,
**execution providers**, all three cache keys, and per-stage counts.

`execution_providers` is there for a specific reason: if CUDA fails to
initialise, onnxruntime falls back to CPU with a warning buried in
stderr and the run simply becomes far slower with nothing to say why.
The provider list in the stats row makes that answerable after the fact.

### `debug_mode`

`0` off, `1` draws labelled debug boxes for **every** class instead of
censoring, `3` additionally saves debug output. Useful for eyeballing
detection quality without touching `items_to_censor`. It is part of the
censor cache key, so debug output never overwrites a real render.

---

## Why the shipped values are what they are

This section is the evidence trail. Where the evidence no longer
describes the current configuration, it says so.

### `retinanet_v2` — `class_suppression['exposed_breast']`

- **`covered_breast` (`min_iou: 0.55`, `margin: 0.10`)** — median IoU
  0.62 among overlapping pairs, 47% clear 0.70.
- **`face_femme` / `face_masc` (`min_iou: 0.30` / `0.75`)** — ears, eyes
  and mouths getting detected as `exposed_breast`. 256 measured
  overlapping instances for `face_femme` (46% clear IoU 0.20, 37% clear
  0.30); `face_masc` has almost no data (3 instances). Tuned stricter
  than the raw data alone suggests: protecting real detections takes
  priority over catching every face false positive.
- **`exposed_belly` (`min_iou: 0.12`)** — 465 same-instant overlapping
  pairs, median IoU only 0.028. Co-occurrence is usually a small edge
  overlap, not two boxes on one spot. Breast is typically the *more*
  confident label when they overlap (score delta median −0.079), so this
  fires rarely by design. It can only catch a breast misfire when belly
  is also detected in the same instant — if belly never fires, this rule
  cannot help, and that is a `min_prob` problem, not a suppression one.
- **`exposed_buttocks` (`min_iou: 0.30`)** — 116 overlapping pairs,
  median IoU 0.175. Tighter than the belly case, consistent with "same
  spot, model confused between two round body-part shapes". Scores are
  roughly equal when they overlap (+0.022), so IoU is the primary gate.
- **`exposed_chest` (`min_iou: 0.85`)** — the male-breast-equivalent
  label, detected alongside `exposed_breast` on androgynous bodies. 137
  overlapping pairs, cleanly bimodal (28 near-zero vs 60 in the 0.9–1.0
  bin); `min_iou` sits at the start of the high cluster. `exposed_chest`
  is not censored, so firing this leaves the spot fully uncensored — an
  accepted trade.
- **`covered_belly` has no rule** — only 2–4 same-instant pairs exist,
  median IoU ~0.022. Not enough signal.
- **No `exposed_anus` entry** — it is not in `items_to_censor` and exists
  purely as a suppression signal for the vulva rules, so a rule here
  would be a no-op.

### `retinanet_v2` — `class_suppression['exposed_vulva']`

- **`exposed_anus` (`min_iou: 0.125`)** — median IoU only 0.06; a
  stricter threshold would catch too few real overlaps to matter.
- **`covered_vulva` (`min_iou: 0.35`)** — `covered_vulva` was getting
  censored too often (misread as exposed, not suppressed). Loosened
  deliberately past what the raw overlap data alone suggests.
- **`face_femme` / `face_masc` (`min_iou: 0.30`)** — the more common of
  the two face false positives: 909 same-instant pairs and a higher
  overlap rate than breast's. Tightened so a real vulva detection near a
  real face is very unlikely to be suppressed, at the cost of missing
  more of the actual face false positives.
- **`exposed_penis` (`min_iou: 0.70`)** — the "genuinely adjacent real
  objects" case, not same-spot misclassification. Measured IoU across
  400+ overlapping instances is not a single blob: a large low cluster
  (0.0–0.2, ~46%, two real adjacent things) and a smaller high cluster
  (0.8–1.0, ~20%, same spot, model confused), with a flatter middle.
  0.70 sits at the start of the high cluster, deliberately sacrificing
  the ambiguous middle in exchange for very rarely suppressing real
  penetrative content.
- **`exposed_armpits` (`min_iou: 0.60`)** — n=15,533 same-instant
  overlapping pairs, median IoU 0.769, 71% clearing 0.20. A strong
  same-spot signal. Not censored, so firing leaves the spot uncensored —
  same accepted trade as `exposed_chest`. 0.60 sits solidly past the
  median, so it only fires on the closer-to-total-overlap half.

### `retinanet_v2` — `item_overrides`

**`exposed_vulva`: `track_max_gap: 3.6`, `interpolation_max_gap: 1.2`.**
3x the auto-derived baseline. `analyze_track_breaks.py` against real
cached footage showed this cuts track resets ~22% (659 → 513), with
contention barely moving (21% → 23%, already high on this label by
nature — vulva has no paired instance to confuse with, but the model
flickers between exposed and covered nearby).
`match_distance_multiplier` deliberately left at 1.0: only 8 of 659
resets were distance-blocked.

**`exposed_breast`: `track_max_gap: 4.5`, `interpolation_max_gap: 0.5`.**
Same 3x pattern off its own baseline, same gap-blocked-dominant
reset profile, `match_distance_multiplier` left at 1.0 for the same
reason.

`interpolation_max_gap` was raised in step with `track_max_gap` for both
labels, so the wider track-survival window actually gets synthetic
coverage instead of surviving as a longer blank gap.

**`min_prob: 0.21` for `exposed_vulva`** sits just above
`global_min_prob` — as permissive as it can usefully be set. Confirmed
working well for `retinanet_v2` specifically.

### Style weights and how often a style actually appears

A style's share of resolves is its `weight` divided by the sum of every
weight in that label's `censor_style` list — so weights are relative
within a label, and adding entries to one group dilutes every other
group. `analyze_style_flicker.py` prints the resulting expected share
per label, which is the number to check a change against rather than
counting entries by eye.

**Stickers were halved in v2.1.1.** They had drifted to 50% of
`exposed_vulva` resolves (three sticker entries at weight 1 against
three blur entries at weight 1) and 24% of `exposed_breast` resolves
(3.2 of 13.4 total). They are now 25% and 14%. The variety is
deliberately kept; the prevalence is not.

There is a performance argument in the same direction, which is why this
was a change worth making rather than purely a taste one. From
`betabench.py render` on real hardware, per box per rendered frame at
400px: bar 1.7-3.7ms, mosaic 2.6-3.4ms, gaussian blur 5.9-8.1ms, hex
6.9-15.1ms, **sticker 17.3-31.2ms**. Sticker is an order of magnitude
above the cheapest styles, so its weight is the single largest
style-side lever on render time — and render was 84% of that run's wall
clock.

> Read those render numbers with the spread in mind. Before v2.1.1 the
> bench took a single timed burst per style and reported 0.27ms and
> 1.12ms for two identical configurations in the same run. It now takes
> several trials, reports their median, and prints the spread; treat any
> difference smaller than the printed spread as noise.

**`exposed_breast` gained eight tight-crop variants in 2.5.** Each of
the four non-bar, non-sticker groups — gaussian blur, box blur, mosaic
pixel, hex pixel — got two extra entries replicating that group's two
median-strength configurations (strength 30 and 40) at
`width_area_safety: -0.50`, `height_area_safety: -0.25`.

The weights were then rescaled so that **each group's share of total
resolves is unchanged**, because adding entries to one group otherwise
dilutes every other group:

| group | share before | share after |
|---|---|---|
| blur / gaussian | 7.20% | 7.20% |
| blur / box | 7.20% | 7.20% |
| pixel / mosaic | 29.66% | 29.66% |
| pixel / hex | 29.66% | 29.66% |
| bar | 12.71% | 12.71% |
| sticker | 13.56% | 13.56% |

The arithmetic: a group's existing weights are multiplied by
`old_sum / expanded_sum`, so gaussian's four entries at 0.10/0.25/0.25/
0.25 (sum 0.85) become six entries summing to the same 0.85. Every
group keeps its slice; the slice is just cut into more pieces.

**What this costs.** These variants deliberately cover less: half the
box width and a quarter of its height is removed. Within each group the
tight-crop pair now takes a third of that group's probability mass, so
the *average* censored area for `exposed_breast` drops even though the
group shares held constant. That is the intended trade — more variety
in how tightly the censor hugs the detection — but it is a coverage
reduction, and worth re-checking visually rather than assuming the
unchanged share table means nothing changed.

### `retinanet_v2` — styles

**`exposed_vulva`** uses blur and sticker at equal weight, with light
feathering: a wider feather ring would let real detail show at the edge,
and this content should err toward covering rather than softening.

**`exposed_breast`** uses a randomised mix across all four types, each
weighted so blur / pixel / bar / sticker land with roughly equal
probability. Blur and pixel variants are elliptical; bars stay
rectangular with `merge: 'none'`, so each breast gets its own bar while
`paired_style` still shares which style got picked. Every style group
carries at least three strength and three feather values, paired
low/mid/high rather than a full cross product. More feathering than
vulva, because the opaque and blocky styles need a soft edge to read as
intentional censoring rather than a hard cutout.

### Per-variant overrides (v2.1.2)

`item_overrides` now resolves through **three** tiers, most general to
most specific:

```python
betaconfig.item_overrides[label]                     # how censoring LOOKS
detector_backend[<backend>]['item_overrides'][label] # what this model wants
detector_backend[<backend>]['variants'][<variant>]['item_overrides'][label]
```

Each merges per key over the one above, so a variant names only what it
genuinely differs on and inherits the rest.

**Why the third tier exists.** Two variants of one backend are two sets
of weights that detect differently, so their tracking wants different
settings. The first real `auto_tune` run showed it directly: at
`match_distance_multiplier` 2.0, nudenet_v3's 640m avoided 39 track
resets *and* had 51 fewer risky assignments — better on both axes —
while 320n rejected every candidate. With only a backend tier that value
could not be written at all; the two had to share a compromise that
suited neither.

The tuner writes here automatically when the configurations disagree,
and to the backend tier when they agree. It will not split a setting
just because one variant found a winner and the other found nothing:
that is silence, not disagreement.

The variant is now part of the **censor key**, so two variants with
different tracking settings no longer share rendered output.

### A note on the first live tuning run (v2.1.1)

The first `auto_tune.py --apply` run produced values that should not be
trusted, and both causes are fixed. They are recorded here because the
symptoms are worth recognising if they ever recur.

**Coverage was measured through a lottery.** `censor_style` is a
weighted list drawn from at random per tracked instance, and each style
carries its own `width_area_safety` / `height_area_safety` — for
`exposed_breast` those span -0.60 to +0.50. `smooth_boxes` recomputes
box geometry from whichever style was drawn, and coverage is an area
measure, so the draw dominated the signal. Seeding the RNG made each
replay reproducible but did NOT make two replays comparable: the knobs
under test change how many draws happen and in what order, so one extra
draw early shifts every style after it. Coverage swung 5-20% in both
directions on changes that moved the rendered box count by a fraction of
a percent, and those were false REJECTIONS — real candidates thrown out
for the style lottery. The replay now holds style selection fixed, which
is ordinary experimental control: style is orthogonal to every knob the
tuner touches. With it held, a `track_max_gap` sweep that drops resets
5 → 2 shows 0.0000% coverage change, which is what a working measurement
looks like.

**The replay was not deterministic.** A label's `censor_style` is a
weighted list, drawn from at random per box, and each style carries its
own `width_area_safety` / `height_area_safety` — for `exposed_breast`
those span -0.60 to +0.50. `smooth_boxes` recomputes each box's geometry
from the style it drew, so replaying the same detections twice produced
boxes of substantially different areas. Coverage is an area measure, so
two replays of the *identical* configuration disagreed by 35-45% while
their rendered box counts differed by 0.2%, and the same baseline came
back as 285, 265 and 268 track resets in three consecutive stages. The
tell: a coverage column that moves violently in both directions while
the rendered count barely moves. Replays are now seeded
(`--replay-seed`), so style contributes identically to every candidate
and cancels out of the comparison.

**Two variants compounded each other's writes.** Every variant of a
backend reads the same `item_overrides` block, so `track_max_gap` is one
setting, not two. The 320n stage raised `exposed_vulva` from 21.6 to
64.8; the 640m stage then read 64.8 as *its* baseline and doubled it
again to 129.6 — a value no measurement ever proposed. A shared key now
gets one decision, measured against every configuration it governs, and
a candidate has to be eligible under all of them. "Good for 320n,
harmful for 640m" is not a value you want written to a setting both
read.

`exposed_vulva`'s `track_max_gap` was reverted to 21.6 as a result.

### `nudenet_v3` — `track_max_gap`, doubled in v2.1.1

**`exposed_vulva`: 10.8 → 21.6. `exposed_breast`: 13.5 → 27.0.**

`analyze_track_breaks.py` against the first full two-backend run swept
both labels over 2x / 3x / 5x of their existing baselines. The deciding
number is risky contention added per track reset avoided, and for
`exposed_breast` the marginal price gets worse with every step past 2x:

| multiplier | resets avoided | risky added | price |
|---|---|---|---|
| 2x | 91 | 192 | 2.1 each |
| 3x | 135 | 536 | 4.0 each |
| 5x | 152 | 819 | 5.4 each |

2x is the best-priced point and the rest is diminishing returns, so 2x
is what was applied. The same sweep on `match_distance_multiplier` cost
68 risky assignments per reset avoided at 3x, which is why that knob was
left at 1.0 — the headline "fewer resets" was identical and the price
was seventeen times higher.

> Two caveats worth carrying. These baselines were themselves derived at
> picture size 1280 under the previous configuration, so doubling them
> compounds a number that has not been re-derived at 320. And for
> `exposed_vulva` the sweep showed essentially no contention at all (14
> events across 4919 decisions), which means the safety signal for that
> label has very little statistical power — "no measured downside" there
> is closer to "not enough events to measure" than to "safe". Both are
> reasons to re-run `auto_tune.py` once more footage exists rather than
> treating these as settled.

### `nudenet_v3` — fully data-derived at 640m (2026-09-19)

Every `margin` and `min_iou` in the shipped `nudenet_v3`
`class_suppression` block is now the value
`tools/bench/betabench.py suppression` proposed from the `640m` caches
(7 videos, 182,616 detections), used unmodified. Before this pass the
rules were **hand-set numbers that had never been derived from data** —
including the 2026-09-18 pass described below, which changed two rules
from the suggestions and left five at their 2026-09-16 hand-picked
values. That is worth stating plainly, because the file's comment said
"re-derived" and only part of it was.

**`min_iou` was 5-30x tighter than the data supports.** This was the
systematic error. 640m's boxes are tight, so cross-label IoU medians sit
at 0.02-0.08 — but the shipped gates were 0.05 and 0.30. A rule with
`min_iou: 0.30` against a median IoU of 0.018
(`exposed_breast<-exposed_buttocks`) essentially never fired. Several
"rules" in the config were decorative.

**The 2026-09-18 `covered_breast` change was wrong and is corrected
here.** It was set to `min_iou: 0.30` on the reasoning that "median IoU
0.815 means they are the same object". That 0.815 was the p75-weighted
figure; the p25 is 0.029 and the tool's suggestion is 0.029. Anchoring
on the wrong percentile made the rule fire far less than claimed, which
is consistent with covered breasts still being censored as exposed after
that change. It is now `margin: 0.113, min_iou: 0.029`.

**Pairs the tool flags "the suppressor rarely outscores this label" are
now absent rather than present with a hand-set margin.** A rule whose
suppressor wins 8-20% of the time needs a margin near zero to fire at
all, and a margin near zero is the loosest possible gate — the rule
either does nothing or fires indiscriminately. Removed on that basis:
`exposed_breast<-exposed_penis` (18%), `<-exposed_vulva` (20%),
`<-covered_belly` (8%), and `exposed_vulva<-exposed_anus` (17%).

**`exposed_vulva` gained three rules and deliberately refused three
more.** The label had two rules where the data supports eleven. Added:
`exposed_breast` (n=2388, suppressor wins 80%), `exposed_belly`
(n=1689, 72%), `covered_breast` (n=776, 65%) — all cases where the
suppressing label is usually the more confident one.

Not added, and the reasoning matters more than the list:

- **`exposed_penis`** — the penetrative case. Removing this rule is what
  fixed penetrative content being missed; adding it back under a
  different margin would undo that.
- **`exposed_buttocks`** — n=7721 against 8541 total `exposed_vulva`
  detections. Nearly every vulva detection co-occurs with a buttocks box
  at the same instant, which is anatomy, not a competing claim about the
  same object. A rule on that pair puts ~90% of the label's detections
  behind a suppression gate, and the tool's suggestion cannot see the
  difference between "these two labels describe one object" and "these
  two things are adjacent in every frame".
- **`exposed_feet`** — same shape at smaller scale: same-frame, not
  same-object.

That distinction is the standing limit on this tool. It measures
same-instant overlap and score margin; it cannot tell a misclassification
from two body parts that are genuinely both present. Pairs whose overlap
is anatomical get excluded by hand, and the exclusions are listed in
`betaconfig.py` so they are not silently re-added by the next pass.

**`time_safety` was lowered** to 0.22 (`exposed_breast`, from 0.35) and
0.20 (`exposed_vulva`, from 0.27), for censoring that appears before or
persists after the exposure. See `default_time_safety` below for the
arithmetic — the short version is that at `video_censor_fps = 9` the
sampling interval is 111ms, and 0.22 still leaves a full interval of
overlap between consecutive boxes while halving the bleed past a true
edge.

**`track_max_gap` for `exposed_vulva` went 21.6 → 43.2**, which is
`auto_tune`'s accepted decision from the 2026-09-19 run: 20 track resets
avoided across both nudenet configurations at ~0 risky assignments each.

`auto_tune` also proposed `min_track_hits: 3` for both labels again.
**Not applied.** It was lowered to 2 in the previous pass specifically to
fix missed detections on brief content, and the tool re-proposes 3 every
run because it scores only "boxes dropped from unconfirmed tracks
against coverage lost" — it has no way to see that those dropped boxes
were the reported symptom. This is a case where the operator's
observation outranks the metric.

---

### `nudenet_v3` — the 2026-09-18 pass (superseded above)

The shipped `nudenet_v3` values were re-derived on 2026-09-18 from the
`640m` caches: 7 videos, 182,616 detections, at the variant's native
640 with per-class NMS. They supersede the 2026-09-16 pass described
further down, which was fitted at 1280 with class-agnostic NMS and is
kept here because its reasoning about the confounds still applies.

**`exposed_vulva <- exposed_penis` was removed entirely.** This was the
largest single efficacy bug in the shipped ruleset. The pair has 2,062
overlapping same-instant detections; 25% clear the old `margin: 0.10`
and 40% clear the old `min_iou: 0.10`, so the rule was suppressing
roughly 500 vulva detections. A penis box overlapping a vulva box **is**
the penetrative case, not a misdetection of one as the other, so the
rule was deleting exactly the content it should have been keeping. The
IoU histogram is unimodal and decaying (1242 pairs in [0.0-0.1), 488 in
[0.1-0.2), 245 in [0.2-0.3)), with no bimodal "same spot" cluster to
separate a genuine confusion case from the real one, which is why the
rule was removed rather than retuned.

**`exposed_breast <- covered_breast` was tightened to
`margin: 0.05, min_iou: 0.30`** (from `0.10 / 0.05`). This targets
covered breasts being reported as exposed. At 640m the pair has 3,278
overlapping detections with median IoU 0.815 — the two labels really are
describing the same object, so the geometry gate can afford to be much
stricter than 0.05 without losing the cases that matter. Dropping the
margin from 0.10 to 0.05 takes the rule from firing on 21% of
overlapping pairs to 30%.

**`exposed_breast <- face_masc` was considered and deliberately not
added,** for male chests being reported as `exposed_breast`. The data
does not support it: only 30 overlapping pairs exist at 640m, and the
median score delta is **−0.070** — `face_masc` scores *below* the
`exposed_breast` box it overlaps, so a margin-based rule loses by
default and only 3 of 30 pairs clear a 0.20 margin. A face box near a
chest box is also a weak signal on its own, and actively wrong in a
two-person scene. NudeNet has no male-chest class that competes properly
(`exposed_chest` exists but is rare and weak: n=515, median score 0.427),
so this is a model limitation rather than a tuning one. Left alone by
explicit decision rather than oversight.

**`min_track_hits` was lowered from 3 to 2** for both labels. At
`video_censor_fps = 9`, requiring three real detections means roughly a
third of a second of continuous detection before anything renders at
all — a plausible cause of "obvious content gets nothing", independent
of any threshold. See `default_min_track_hits` above for the trade.

**`min_prob` was lowered** to 0.28 (`exposed_breast`) and 0.26
(`exposed_vulva`), alongside `global_min_prob` 0.20 → 0.12. Note which
of these was actually binding: at 640m, `exposed_breast`'s score
distribution has p1 = 0.228 and p25 = 0.690, so the old `min_prob` of
0.37 sat near the 1st percentile and was rejecting almost nothing. The
floor was the real gate. `betabench hysteresis` independently suggested
*raising* these to 0.69 and 0.393 (the p25 values); that recommendation
is about precision, and was not followed because the reported symptom
was missed detections, not false ones. If false positives become the
complaint, that suggestion is where to start.

---

The original 2026-09-16 findings, and why they were not simply trusted:

- Cross-label IoU distributions sat far below `retinanet_v2`'s for the
  same label pairs on the same footage — `exposed_breast<-covered_breast`
  median IoU 0.036 versus 0.475.
- `nudenet_v3`'s boxes ran structurally smaller and tighter across
  nearly every label (median `exposed_breast` box area 5994 versus 7052
  on the same footage).
- Track resets were far less frequent at the same thresholds
  (`exposed_vulva` baseline contention: 54 events versus 4403 on the
  same footage).

**All three are also what the two upstream confounds would produce
independently:**

1. A 320-trained export fed a 1280x1280 blob has a receptive field four
   times too small relative to object scale. It finds sub-features
   rather than whole objects, which produces exactly "smaller, tighter
   boxes" and therefore exactly "lower cross-label IoU".
2. Class-agnostic NMS at 0.45 deletes the high-IoU cross-label pairs
   *before* anything can measure them, which biases the measured
   distribution toward its low tail.

That does not mean the original conclusions were wrong. It means they
are not yet separable from the confounds, and the thresholds fitted to
them may be describing a misconfiguration rather than the model.

The cheap discriminating experiment:

```bash
# one preview pass at the native size, then look at the distributions
python3 betatv.py --backend nudenet_v3 --variant 320n --preview on
python3 tools/bench/betabench.py suppression --backend nudenet_v3
```

That experiment has since been run: the values at the top of this
section are its result. The confound analysis above is why they were
re-derived from scratch at the native size rather than adjusted from the
1280 numbers.

The `margin` policy from that pass still stands on its own, because it
is a judgment about how suppression should behave rather than a fitted
number: five `exposed_breast` rules whose data-derived margin was 0.00
were raised by hand to 0.10, on the grounds that a suppressing label
should have to be *measurably* more confident, not merely tie, before
overriding a real detection. `exposed_vulva<-exposed_anus` stays at its
derived 0.05 as a reviewed exception.

---

## Re-deriving values from your own footage

`tools/bench/betabench.py` is the one harness. Every subcommand reports
numbers from your hardware and your footage, and proposes the setting it
implies.

```bash
python3 tools/bench/betabench.py all          # everything, in order
python3 tools/bench/betabench.py --help       # what each one answers
```

| Command | Answers | Informs |
|---|---|---|
| `decode` | is sequential decoding really faster here? | confirms the sampling design on your box |
| `detect` | what does one sampled frame cost, per stage? | `picture_sizes`, `model_variant`, `nn_batch_size` |
| `render` | what does one censored box cost per frame? | `censor_style` weights, `blur_fast_approximation` |
| `geometry` | what box shapes does each label produce? | the geometry sanity filter |
| `suppression` | what do cross-label overlaps look like? | `class_suppression` |
| `dedup` | how much do sizes overlap? | `cross_size_dedup['iou_threshold']` |
| `hysteresis` | what does each label's score distribution look like? | `min_prob`, `min_prob_continue` |
| `structure` | per video: how fast does it cut, how many subjects, where in frame? | whether footage needs per-profile settings |

`structure`, `geometry`, `suppression`, `dedup` and `hysteresis` read
the detection caches a real run already produced — fast, and they
describe exactly what that run saw. Run `betatv.py` first if the cache
is empty; a `--preview` pass is enough to start.

### Reading `structure`, and what profiles would need

Every other subcommand pools all footage into one distribution. That is
right for "what does this model do" and wrong for "does this file need
different settings from that one": a quick-cut compilation and a long
single-scene clip average into a description of neither.

`structure` reports per video instead, and what it reports is
deliberately **structural rather than content-based**:

| Column | What it separates |
|---|---|
| `cuts/min`, `med shot` | compilation vs. single scene |
| `simul`, `p90` | one subject vs. several on screen at once |
| `L-half` | where the action sits; 0.50 with high `simul` is split-screen |

On synthetic footage those three shapes separate cleanly — a compilation
at 213 cuts/min with 0.28s shots, a single-scene file at 5.4 cuts/min
with 10s shots and `L-half` 1.00, a split-screen at `simul` 2.0 and
`L-half` exactly 0.50.

**Why structural and not "solo / couple / breast-focused".** Those are
content categories, and nothing in a detection cache can recover them. A
box does not say whose body it is on. What a cache *can* say is how
often the scene cuts, how many boxes are live at once, and where they
sit — and those are exactly the properties that want different settings:

- fast cuts want a **smaller `time_safety`** (less bleed across a cut)
  and a **smaller `track_max_gap`** (a track should not survive a scene
  change), and possibly a **higher `video_censor_fps`**
- several simultaneous subjects want a **tighter `match_distance`**, because
  that is precisely when a loose one merges two people into one track
- split-screen wants both, plus it makes `paired_style` sharing across
  the screen divide actively wrong

So content names can exist as *aliases* over structural profiles, but
the classifier has to run on structure.

**What this does not yet do.** `structure` is a read-only report. There
is no profile mechanism in the config, nothing selects a profile, and
nothing applies one. Building that means three separate pieces, and they
are worth keeping separate because each can be wrong on its own:

1. **Profile definitions** — named override blocks in `betaconfig.py`,
   resolved as a tier between a backend's `item_overrides` and the
   per-label values.
2. **A classifier** — the thresholds that turn `structure`'s numbers
   into a profile name. These have to be derived from footage that
   actually spans the range; on a set that all cuts at similar rates,
   any threshold is arbitrary.
3. **Selection** — how a run picks one. Two options with different
   risks: chosen per file at run time from a cheap pre-scan (shot-cut
   detection already runs, so the cut rate is nearly free), or set by
   hand per run. Automatic selection is the more useful and the more
   dangerous: a misclassified file gets settings tuned for footage it is
   not, silently, and the failure looks like a tuning regression rather
   than a misrouted profile.

The honest prerequisite for any of it is footage that spans the range.
`structure` prints the cut-rate spread across the files it found and
says so directly: below about 3x, one set of timing values suits
everything present, and a profile split would be answering a question
the footage is not asking.

Everything goes to `../output/benchmarks/<timestamp>/`, with a log at
`--log-level` (default `debug`) and a terminal view at
`--console-level` (default `info`), plus machine-readable
`results.json`.

**The order that works.** Detect first, look at the distributions, change
one thing, re-run the same preview slice, compare. Changing several
settings between comparisons means you learn nothing from the
difference.
