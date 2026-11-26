# Engine protocol

How a front end (the desktop UI, a stub client, a test) drives the engine. The schemas in `v1/` are the contract; `betautils_protocol.py` implements it, and the tests keep the two in step.

## Transport

- The client starts the engine as a child process (`python betaengine.py serve`) and owns it for its whole life. There is no network socket.
- One JSON object per line, UTF-8, `\n` ending (`\r\n` accepted). The engine reads commands on its stdin and writes results and events on its stdout, and nothing else ever reaches its stdout.
- Logs go to the engine's log file and its stderr. **The client must keep reading stderr** (or point it at a file): a pipe nobody drains fills after a few tens of KiB and then blocks the engine's next log write.
- A line longer than `limits.max_line_bytes` (reported by `hello`, 1 MiB in v1, line ending not counted) is discarded and reported with `protocol.line_too_long`. The limit is fixed; a client checks it before sending.
- Closing the engine's stdin ends the session as cleanly as `shutdown` does.

## Messages

Every message has `v` (the protocol version, 1), `kind`, `name` (dotted, lower case) and `data` (an object):

| kind | direction | carries |
|---|---|---|
| `cmd` | client to engine | `id`: a string of 1 to 128 characters, unique per session (a UUID is a good choice) |
| `result` | engine to client | the command's `id` and `name`; `"ok": true` with `data`, or `"ok": false` with `error` |
| `event` | engine to client | no `id` |

Each command gets exactly one result, and a result always carries the `id` of the command it answers.

A line too broken to read an `id` from (bad UTF-8, bad JSON, too long, or an envelope with no usable `id`) gets no result. The engine sends a `protocol.rejected` event instead, with `line` (which line) and `error`. Lines are numbered from 1 in the order the engine read them, counting every line, blank ones included, so a client that counts the lines it writes knows which command failed.

An error is `{"code": "...", "params": {...}}` with exactly the params `v1/errors.schema.json` lists for that code. The engine never sends sentences: the client owns wording and translation, and tracebacks stay in the engine log.

## Compatibility

- Both sides ignore fields they do not know.
- `v` changes only on a breaking change.
- `hello` must come first: the client sends the versions it speaks, the engine answers with the one it chose, its own version, and the commands and events it has. Any other command before a successful `hello` (except `shutdown`) gets `protocol.hello_required`.

## Settings

Four commands read and change the settings. They use the same local config file and the same checks as `tools/setup/local_config.py`, so a value set here is what the command-line tools read on their next run.

| command | does |
|---|---|
| `config.schema` | the settings registry: every setting's path, kind, limits, scope, level, screen, the settings it requires, the cache keys it moves, and its message and help keys |
| `config.get` | current values, with their defaults, where each came from (`default`, `local`, or `run` for an override sent with the request) and every layer that sets it; with `labels: true`, also every label as a run resolves it: each per-label value, the tier it comes from, every other tier that sets it and why it is not in effect, and a profile's value on top. A label the selected backend cannot report has `available: false` and `unavailable: {code: "label.not_in_model", params: {backend}}` |
| `config.set` | checks a change as a run would and writes it to the local config only if it passes; `dry_run` checks without writing |
| `config.reset` | removes settings from the local config, so the shipped values apply again |

A path is the setting's name, then the keys inside it, joined by dots: `encode_crf`, `detector_backend.nudenet_v3.nms_iou`. A value set whole, such as a censor style or a list of rules, is set at its own path, never inside.

`config.set` and `config.reset` answer `moves`, the cache keys the change moves: `detection` (re-detect), `censor` (re-render) or `encode` (re-encode). The engine works it out by comparing the keys before and after, which is how a client says what a change will cost before making it.

A change that would fail validation answers `config.invalid`, listing every problem as `config.bad_type`, `config.out_of_range`, `config.bad_choice`, or `config.rule` with a stable rule id. Nothing is written then. The rule ids are listed in `v1/errors.schema.json`; a test keeps that list equal to the ids the checks return.

Settings must not change while a job runs. While any job is queued, running or paused, `config.set` and `config.reset` answer `config.busy` and write nothing, and `config.get` answers from the values taken when the queue became busy (with `overrides`, `config.busy`). `config.schema` is always answered.

## Jobs

A job is a list of files run one after another through the same pipeline as `betatv.py`, with settings overrides for that job only. The command line runs everything in its input folder as one job, through the same code.

| command | does |
|---|---|
| `job.submit` | queues the chosen files (absolute paths) with optional `overrides` (dotted paths, as in `config.get`); answers the job's id and how many jobs are ahead of it |
| `job.pause` | the running job stops at its next safe point and the queue waits behind it; detection checkpoints and finished render chunks are kept |
| `job.resume` | the paused job runs again: the file it paused in continues from its checkpoints, finished files are not run again |
| `job.cancel` | a queued or paused job is cancelled at once, a running one at its next safe point; the partial outputs of the file it was on are removed, finished outputs and detection caches stay |
| `job.list` | the jobs the queue remembers: the active one at position 0, the waiting ones in run order, then the ended ones (position null), with their files, counts and overrides |
| `job.move` | moves a waiting job to another position; it cannot go ahead of the active job, and a position past either end is taken as that end |
| `job.estimate` | how long the same `files` and `overrides` would take, before submitting, from the timings of earlier comparable runs in the catalog's history; per file and in total, `null` where no comparable run gives a rate (`basis: none`), plus whether an earlier run used the same look (`look_seen`). With no comparable run and the queue idle, it starts a short probe of this machine (`probing`, then the `job.probed` event); jobs wait for it, and its rates fill in where the history has none (`basis: probe`) |

Jobs run one at a time, in queue order: the order submitted unless `job.move` changed it. A paused job holds the queue. While any job is queued, running or paused, the engine holds the library's run lock, so a command-line run is refused, and `job.submit` answers `run.locked` while a command-line run holds it. A file under the input folder writes its output to the same place under the output folder; any other file writes to the output folder itself.

A job's checks (`config.invalid`, `job.unsafe_setting` for `input_delete_probability` above 0, `job.nothing_to_render`) refuse `job.submit` when the queue is idle. A job submitted behind another is checked when it starts, and stops with that code if it fails.

### Job events

| event | when |
|---|---|
| `job.state` | the job is `queued`, `running` or `paused`, or ends `completed`, `cancelled` or `stopped`; a stopped job carries the `code` (`disk.full`, `model.load_failed`, `config.invalid`, `engine.stopping`, ...) |
| `job.file` | one file `started`, or ended `processed` (with its `output`), `skipped` or `failed` |
| `job.progress` | a file's stage (`detect` per size, `persons`, `render`): done, total, rate |
| `job.decision` | a choice the engine made about a file, as a code: the structure profile and the rule that chose it, an output already present, a source that is not a video |
| `job.error` | a file that failed (`file.failed`, the job goes on), or the code that stopped the job, just before its `stopped` state |
| `job.probed` | the estimate probe a `job.estimate` started has ended (`ok` when it measured a rate); asking `job.estimate` again gives the measured estimate |

Order: `job.submit`'s result comes before the job's first event. A file's events come between its `started` and its ending; a job's last event is its final `job.state`. When the last job in the queue sends its final `job.state`, the run lock is already given back and `config.set` is accepted, so a client can start a command-line run or change settings as soon as it reads it. `job.progress` is coalesced to a few per second per job (the latest of each stage), and is the only event that can be dropped. A full disk stops the job and every job queued behind it with `disk.full`. When the engine shuts down (`shutdown`, end of input, a signal), a running job stops at its next safe point with `engine.stopping` and its checkpoints kept, so submitting it again continues where it was.

The client must keep reading stdout while a job runs, as it does stderr: events are written as they happen, so a client that stops reading without closing the pipe stalls the job once the pipe is full. The engine also ends its jobs when the client closes stdin, so a job that keeps running after the window closes needs the client process to stay alive.

## Look previews

`preview.look` draws censor looks with the engine's own censor code and answers with PNG files in the cache folder. Each entry in `previews` is either a piece of the bundled figure art, `art: {style, item}` (items as `assets/art/<style>/style.json` names them: `figure.<sex>.<view>.<dress>`, `tile.<label>.<sex>`; their authored boxes stand in for detections), or any image file with `boxes`. A preview is exactly what a photo with those detections comes out as: the same steps (`betautils_still.py`) and the same censor. `labels` chooses which labels to censor (default `items_to_censor`), so a label switched off can still show its look; `seed` fixes every random pick; `overrides` are per-run settings, as in `config.get`, never written. Each result is `{path, key, cached, width, height, boxes}`, boxes being where each label is on the image (as given, before padding); `key` covers the pixels, the resolved looks, every render setting and the censor code, so an unchanged request answers `cached: true`. While a job holds the queue it answers `config.busy`. Errors: `preview.unknown_art {style, item}`, `preview.bad_image`.

`config.schema` also lists the look presets (`presets`: `{id, version, name_key, style}`), each a `censor_style` value with a name.

## Catalog

The library: every video the engine knows, identified by its content (the 16-character hash in its cache keys and output names), with where it is, what happened to it and what video type it gets. The engine keeps it in `catalog.sqlite` in the data folder, opened on the first catalog command. Files are never moved or copied; the catalog points at them where they are.

| command | does |
|---|---|
| `catalog.list` | a page of items (at most 500), filtered by status, availability, video type, output, duplicates, pinned, folder or censors applied (`censored`: labels hidden and kinds of look used, by one present output), sorted by name, added_at, size or status; with `total`, `pending` (files waiting to be hashed) and `reclaimable_bytes` (what removing every extra copy would free) |
| `catalog.get` | one item with every path it was found at, its outputs and their keys, and its last 50 runs |
| `catalog.folders` | the watched folders |
| `catalog.add_folder` | watch a folder; with `type_from_folder` its subfolder names pick the video type, as a 1.x `uncensored_vids` tree did |
| `catalog.remove_folder` | stop watching a folder; its items keep their history and pins and come back with it |
| `catalog.add_files` | index chosen files, ahead of everything else; a moved file is found again as the same item, so this is also how a client locates a missing one |
| `catalog.rescan` | a pass over every watched folder now |
| `catalog.pin_type` | pin an item's video type, or unpin it (Auto) |
| `catalog.thumbnail` | an item's blurred thumbnail, or with `reveal` its clear frame, made in memory and written nowhere; or its censored output's thumbnail |
| `catalog.forget` | remove an item and its history; its files are not touched |
| `catalog.strip` | a video's scrub strip for the player (by `path`): small frames at known times, in a JPEG in the cache folder; answered at once with what is on disk, while the missing pass is built in the background |

**Watching.** A pass lists each watched folder and reads each file's size, modification time and inode, never its bytes. A rename or a move on one disk is followed by inode; anything else is hashed once it has held still for a few seconds, and found again as the same item when its content is known. Hashing waits while a run is on (a job of this engine, or any run holding the library's run lock, such as the command line). Passes run at start, every `catalog_scan_interval_seconds` (0: only when asked), never sooner than ten times the last pass's length, and on `catalog.rescan`. A new catalog watches the library's `source` and `uncensored_vids` folders and the 1.x video folders a migration recorded, each with its folder rule. A file named as a censored output is an output, wherever it sits.

**Status and availability are separate.** `status` says what happened: `new`, `queued` and `processing` while a job holds it, `done` once a run finished (or found its output already there), `failed`. `availability` says whether its file is there: `present`, `missing` (its folder was read and the file was not in it) or `offline` (its folder could not be read: an unmounted share, an unplugged drive). A missing item stays in the library with its status, history and pin; nothing is removed because it went missing. `outputs` and `outputs_missing` count its censored outputs found and lost, so a done item whose output was deleted still says done, with its output missing.

**Video type.** Precedence, highest first: the item's pin, `--profile`, a path pattern, a folder rule, measurement, the default. An item's `type` says which one decides and when: `declared` (a pin or a rule, known before the run), `last_run` (measured on its last run) or `at_run` (the scan will decide). What the catalog chooses (a pin, or the folder rule of a watched folder outside the 1.x tree) is keyed in the censor key, so the next run renders again and never reuses an output made under another type. The command line honors pins and folder rules too.

**History** is kept from each run's own record, for the app's jobs and command-line runs alike: outcome and reason, output and keys, the video type and its rule, a failure's type and stage. It exists only in the catalog, like pins: back up the catalog file to keep them. `tools/setup/catalog_rebuild.sh` builds the catalog again from the disk and carries folders, pins and history from the old file while it can be read.

**Thumbnails.** Only a source's blurred thumbnail (24 pixels on its long side, blurred) is ever written to disk. `reveal` decodes the frame again into memory and answers it; it is written nowhere. A censored output's thumbnail is stored like any other.

**Scrub strips** are built in two passes. The keyframes pass decodes only keyframes, which is quick even on long-GOP 4K, so the player has pictures within seconds. When keyframes are sparser than one every 2 seconds, an even pass follows: the whole stream is decoded and a frame kept at an even interval, at most 600. The even pass waits while a run is on. Each pass's end is announced by `catalog.strip_ready`. A file the catalog lists as a present censored output gets a clear strip. Any other file's frames are shrunk and blurred as a source thumbnail is before anything is written.

`catalog.changed` says which items changed (`items`, or `all` when another process wrote the catalog or more than 1000 changed at once) and why (`reasons`: scan, job, pin, external). It is merged to a few a second and never dropped. `catalog.strip_ready` says a strip pass for a `path` was written (`ok`) or could not be made.

## Files

- `v1/envelope.schema.json`: the four message shapes
- `v1/errors.schema.json`: every error code and its params
- `v1/commands/<name>.schema.json`: per command, `$defs/data` (what the client sends) and `$defs/result` (the `data` of a successful result)
- `v1/events/<name>.schema.json`: per event, `$defs/data`

The schemas use JSON Schema draft 2020-12. Their `$id`s sit under `https://betasuite.invalid/`, a placeholder host that is never fetched.
