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
| `config.get` | current values, with their defaults and where each came from (`default`, `local`, or `run` for an override sent with the request) |
| `config.set` | checks a change as a run would and writes it to the local config only if it passes; `dry_run` checks without writing |
| `config.reset` | removes settings from the local config, so the shipped values apply again |

A path is the setting's name, then the keys inside it, joined by dots: `encode_crf`, `detector_backend.nudenet_v3.nms_iou`. A value set whole, such as a censor style or a list of rules, is set at its own path, never inside.

`config.set` and `config.reset` answer `moves`, the cache keys the change moves: `detection` (re-detect), `censor` (re-render) or `encode` (re-encode). The engine works it out by comparing the keys before and after, which is how a client says what a change will cost before making it.

A change that would fail validation answers `config.invalid`, listing every problem as `config.bad_type`, `config.out_of_range`, `config.bad_choice`, or `config.rule` with a stable rule id. Nothing is written then. The rule ids are listed in `v1/errors.schema.json`; a test keeps that list equal to the ids the checks return.

Settings must not change while a job runs.

## Files

- `v1/envelope.schema.json`: the four message shapes
- `v1/errors.schema.json`: every error code and its params
- `v1/commands/<name>.schema.json`: per command, `$defs/data` (what the client sends) and `$defs/result` (the `data` of a successful result)
- `v1/events/<name>.schema.json`: per event, `$defs/data`

The schemas use JSON Schema draft 2020-12. Their `$id`s sit under `https://betasuite.invalid/`, a placeholder host that is never fetched.
