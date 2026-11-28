//! The engine as a child process: `python betaengine.py serve`, speaking the
//! protocol in protocol/README.md over its stdin and stdout.
//!
//! An [`Engine`] runs at most one engine process at a time. It starts it,
//! sends `hello`, matches every result to the request that asked for it,
//! hands events to a callback, keeps reading stderr (into the log, target
//! `engine`) so the engine never blocks on a full pipe, notices when the
//! process ends, and shuts it down: `shutdown`, then a kill if it has not
//! exited in time. Nothing here depends on Tauri or on a window.
//!
//! What it reports are codes and parameters (`ui.engine.*`, or the engine's
//! own codes), never sentences; the page words them from the catalog.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Map, Value};

pub const PROTOCOL_VERSION: u64 = 1;

/// The line limit until `hello` reports the engine's own.
const DEFAULT_MAX_LINE: usize = 1 << 20;

/// How long the engine gets to exit after closing its stdout.
const EXIT_GRACE: Duration = Duration::from_secs(2);

/// An error as the page receives it: a code and its parameters.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ErrorCode {
    pub code: String,
    pub params: Map<String, Value>,
}

impl ErrorCode {
    pub fn new(code: &str, params: Value) -> Self {
        let params = match params {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        ErrorCode {
            code: code.to_owned(),
            params,
        }
    }

    /// An engine error from the wire (`{"code": ..., "params": {...}}`).
    fn from_wire(error: &Value) -> Self {
        match error.get("code").and_then(Value::as_str) {
            Some(code) => ErrorCode::new(code, error.get("params").cloned().unwrap_or(Value::Null)),
            None => ErrorCode::new("ui.engine.bad_message", json!({})),
        }
    }
}

/// Where the engine is in its life. Serialized for the page with a `state`
/// field: `{"state": "ready", "version": ..., "protocol": 1, "pid": ...}`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Status {
    /// Not started yet.
    Idle,
    /// Started, waiting for the `hello` result.
    Starting,
    Ready {
        version: String,
        protocol: u64,
        pid: u32,
    },
    /// Shutting down on request.
    Stopping,
    /// Stopped on request, with how the process ended: code 0 after a clean
    /// shutdown; a signal (POSIX) or another code when it had to be killed.
    Stopped {
        code: Option<i32>,
        signal: Option<i32>,
    },
    /// Ended without being asked to. `signal` is set on POSIX when a signal
    /// ended it.
    Exited {
        code: Option<i32>,
        signal: Option<i32>,
    },
    /// Could not be started, or failed its handshake.
    Failed { error: ErrorCode },
}

impl Status {
    fn can_start(&self) -> bool {
        matches!(
            self,
            Status::Idle | Status::Stopped { .. } | Status::Exited { .. } | Status::Failed { .. }
        )
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    /// The folder holding betaengine.py; found by [`locate_engine_dir`] when unset.
    pub engine_dir: Option<PathBuf>,
    /// The Python to run it with; found by [`locate_python`] when unset.
    pub python: Option<PathBuf>,
    /// Extra arguments after `serve`.
    pub args: Vec<String>,
    /// Sent in `hello` as the client's version.
    pub client_version: String,
    pub hello_timeout: Duration,
    pub request_timeout: Duration,
    pub shutdown_timeout: Duration,
    /// Every message in and out, one JSON object per line. An existing file
    /// is kept as `<name>.1` when the Engine is created.
    pub transcript: Option<PathBuf>,
}

impl Config {
    pub fn new(client_version: &str) -> Self {
        Config {
            engine_dir: None,
            python: None,
            args: Vec::new(),
            client_version: client_version.to_owned(),
            hello_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(5),
            transcript: None,
        }
    }
}

pub type StatusFn = Arc<dyn Fn(&Status) + Send + Sync>;
pub type EventFn = Arc<dyn Fn(&Value) + Send + Sync>;

type Reply = mpsc::Sender<Result<Value, ErrorCode>>;

/// One engine process and the requests waiting on it.
struct Session {
    generation: u64,
    pid: u32,
    stdin: Option<ChildStdin>,
    child: Arc<Mutex<Child>>,
    started: Instant,
    pending: HashMap<String, Reply>,
    /// Which request was written on which line, for `protocol.rejected`.
    id_at_line: HashMap<u64, String>,
    line_of_id: HashMap<String, u64>,
    lines_written: u64,
    next_id: u64,
    max_line: usize,
    /// Set when the shell itself asked the process to end.
    exit_expected: bool,
}

struct Shared {
    status: Status,
    session: Option<Session>,
    generation: u64,
}

struct Inner {
    config: Config,
    on_status: StatusFn,
    on_event: EventFn,
    shared: Mutex<Shared>,
    changed: Condvar,
    transcript: Mutex<Option<File>>,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Inner>,
}

/// The first folder holding betaengine.py: `BETASUITE_ENGINE_DIR` when set
/// (and only it), else the executable's folder or one of its parents, which
/// covers a binary built inside the repository and one placed in it.
pub fn locate_engine_dir() -> Option<PathBuf> {
    let has_engine = |dir: &Path| dir.join("betaengine.py").is_file();
    if let Some(dir) = env::var_os("BETASUITE_ENGINE_DIR").filter(|d| !d.is_empty()) {
        let dir = PathBuf::from(dir);
        return has_engine(&dir).then_some(dir);
    }
    let exe = env::current_exe().ok()?;
    exe.ancestors()
        .skip(1)
        .find(|dir| has_engine(dir))
        .map(Path::to_path_buf)
}

/// `PYTHON_BIN` when it names a file, else the engine folder's `.venv`,
/// else `python3` or `python` on PATH.
pub fn locate_python(engine_dir: &Path) -> Option<PathBuf> {
    if let Some(python) = env::var_os("PYTHON_BIN").map(PathBuf::from) {
        if python.is_file() {
            return Some(python);
        }
    }
    let venv = if cfg!(windows) {
        engine_dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        engine_dir.join(".venv").join("bin").join("python")
    };
    if venv.is_file() {
        return Some(venv);
    }
    let names: &[&str] = if cfg!(windows) {
        &["python.exe", "python3.exe"]
    } else {
        &["python3", "python"]
    };
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .find(|candidate| candidate.is_file())
}

fn exit_parts(status: ExitStatus) -> (Option<i32>, Option<i32>) {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        (status.code(), status.signal())
    }
    #[cfg(not(unix))]
    {
        (status.code(), None)
    }
}

impl Engine {
    pub fn new(config: Config, on_status: StatusFn, on_event: EventFn) -> Self {
        let transcript = config.transcript.as_ref().and_then(|path| {
            let open = || -> std::io::Result<File> {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                if path.exists() {
                    let mut old = path.clone().into_os_string();
                    old.push(".1");
                    fs::rename(path, old)?;
                }
                File::create(path)
            };
            open()
                .map_err(|error| {
                    log::warn!(
                        "cannot write the engine transcript {}: {error}",
                        path.display()
                    )
                })
                .ok()
        });
        Engine {
            inner: Arc::new(Inner {
                config,
                on_status,
                on_event,
                shared: Mutex::new(Shared {
                    status: Status::Idle,
                    session: None,
                    generation: 0,
                }),
                changed: Condvar::new(),
                transcript: Mutex::new(transcript),
            }),
        }
    }

    pub fn status(&self) -> Status {
        self.inner.lock().status.clone()
    }

    /// Starts the engine unless one is running. Returns at once; progress
    /// arrives through the status callback.
    pub fn start(&self) {
        let generation = {
            let mut shared = self.inner.lock();
            if !shared.status.can_start() {
                log::debug!("start ignored: the engine is {:?}", shared.status);
                return;
            }
            shared.generation += 1;
            self.inner.set_status(&mut shared, Status::Starting);
            shared.generation
        };
        let inner = Arc::clone(&self.inner);
        thread::spawn(move || inner.run_session(generation));
    }

    /// Starts it again after it stopped, exited or failed.
    pub fn restart(&self) {
        log::info!("engine restart requested");
        self.start();
    }

    /// Sends a command and waits for its result, up to the request timeout.
    pub fn request(&self, name: &str, data: Value) -> Result<Value, ErrorCode> {
        self.inner
            .request(name, data, self.inner.config.request_timeout, false)
    }

    /// Asks the engine to shut down and waits until it has, killing it after
    /// the shutdown timeout. Returns at once when no engine is running.
    pub fn shutdown(&self) {
        self.inner.shutdown();
    }

    /// Blocks until `ready` returns true for the status, or the timeout
    /// passes. Returns the last status seen.
    #[cfg(test)]
    pub fn wait_for(&self, timeout: Duration, ready: impl Fn(&Status) -> bool) -> Status {
        let deadline = Instant::now() + timeout;
        let mut shared = self.inner.lock();
        while !ready(&shared.status) {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            shared = self
                .inner
                .changed
                .wait_timeout(shared, left)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner().0);
        }
        shared.status.clone()
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Called with the lock held, so callbacks see statuses in order. The
    /// status callback must not call back into the Engine.
    fn set_status(&self, shared: &mut Shared, status: Status) {
        if shared.status != status {
            log::info!("engine status: {status:?}");
        }
        shared.status = status;
        (self.on_status)(&shared.status);
        self.changed.notify_all();
    }

    fn record(&self, started: Instant, direction: &str, message: &Value) {
        let redacted = redact(message);
        let message = redacted.as_ref().unwrap_or(message);
        log::trace!(target: "engine", "{direction} {message}");
        if let Ok(mut transcript) = self.transcript.lock() {
            if let Some(file) = transcript.as_mut() {
                let ms = (started.elapsed().as_secs_f64() * 10_000.0).round() / 10.0;
                let line = json!({ "ms": ms, "dir": direction, "message": message });
                let _ = writeln!(file, "{line}").and_then(|()| file.flush());
            }
        }
    }

    fn fail_start(&self, generation: u64, error: ErrorCode) {
        log::error!(
            "the engine could not start: {} {:?}",
            error.code,
            error.params
        );
        let mut shared = self.lock();
        if shared.generation == generation {
            self.set_status(&mut shared, Status::Failed { error });
        }
    }

    fn spawn(&self) -> Result<(Child, PathBuf), ErrorCode> {
        let engine_dir = match self.config.engine_dir.clone().or_else(locate_engine_dir) {
            Some(dir) if dir.join("betaengine.py").is_file() => dir,
            _ => return Err(ErrorCode::new("ui.engine.not_found", json!({}))),
        };
        let python = self
            .config
            .python
            .clone()
            .or_else(|| locate_python(&engine_dir))
            .ok_or_else(|| ErrorCode::new("ui.engine.python_not_found", json!({})))?;
        let mut command = Command::new(&python);
        command
            .arg(engine_dir.join("betaengine.py"))
            .arg("serve")
            .args(&self.config.args)
            .current_dir(&engine_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // No console window for the engine behind a GUI app.
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        log::info!(
            "starting the engine: {} {} serve {}",
            python.display(),
            engine_dir.join("betaengine.py").display(),
            self.config.args.join(" ")
        );
        match command.spawn() {
            Ok(child) => Ok((child, engine_dir)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(ErrorCode::new("ui.engine.python_not_found", json!({})))
            }
            Err(error) => Err(ErrorCode::new(
                "ui.engine.spawn_failed",
                json!({ "os_error": error.kind().to_string() }),
            )),
        }
    }

    fn run_session(self: Arc<Self>, generation: u64) {
        let (mut child, _engine_dir) = match self.spawn() {
            Ok(spawned) => spawned,
            Err(error) => return self.fail_start(generation, error),
        };
        let pid = child.id();
        let (stdin, stdout, stderr) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take());
        let (Some(stdin), Some(stdout), Some(stderr)) = (stdin, stdout, stderr) else {
            let _ = child.kill();
            return self.fail_start(
                generation,
                ErrorCode::new("ui.engine.spawn_failed", json!({ "os_error": "no pipes" })),
            );
        };
        let child = Arc::new(Mutex::new(child));
        let started = Instant::now();
        {
            let mut shared = self.lock();
            if shared.generation != generation {
                return;
            }
            shared.session = Some(Session {
                generation,
                pid,
                stdin: Some(stdin),
                child: Arc::clone(&child),
                started,
                pending: HashMap::new(),
                id_at_line: HashMap::new(),
                line_of_id: HashMap::new(),
                lines_written: 0,
                next_id: 1,
                max_line: DEFAULT_MAX_LINE,
                exit_expected: false,
            });
        }
        log::info!("engine process {pid} started");
        thread::spawn(move || drain_stderr(stderr));
        let reader = Arc::clone(&self);
        thread::spawn(move || reader.read_stdout(generation, stdout, child, started));

        let hello = json!({
            "protocol": [PROTOCOL_VERSION],
            "client": { "name": "betasuite-ui", "version": self.config.client_version },
        });
        match self.request("hello", hello, self.config.hello_timeout, true) {
            Ok(data) => self.on_hello(generation, pid, &data, started),
            Err(error) if error.code == "ui.engine.exited" => {}
            Err(error) => {
                let error = if error.code == "ui.engine.request_timeout" {
                    ErrorCode::new(
                        "ui.engine.hello_timeout",
                        json!({ "seconds": self.config.hello_timeout.as_secs() }),
                    )
                } else {
                    ErrorCode::new("ui.engine.hello_failed", json!({ "code": error.code }))
                };
                self.fail_start(generation, error);
                self.kill(generation);
            }
        }
    }

    fn on_hello(&self, generation: u64, pid: u32, data: &Value, started: Instant) {
        let version = data["engine"]["version"].as_str().unwrap_or("").to_owned();
        let protocol = data["protocol"].as_u64().unwrap_or(0);
        let max_line = data["limits"]["max_line_bytes"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(DEFAULT_MAX_LINE);
        log::info!(
            "engine {version} ready in {} ms, protocol {protocol}, pid {pid}, line limit {max_line} bytes",
            started.elapsed().as_millis()
        );
        let mut shared = self.lock();
        if shared.generation != generation || shared.status != Status::Starting {
            return;
        }
        if let Some(session) = shared.session.as_mut() {
            session.max_line = max_line;
        }
        self.set_status(
            &mut shared,
            Status::Ready {
                version,
                protocol,
                pid,
            },
        );
    }

    /// Writes one command line and waits for its answer. `during_start` lets
    /// `hello` through before the engine is ready.
    fn request(
        &self,
        name: &str,
        data: Value,
        timeout: Duration,
        during_start: bool,
    ) -> Result<Value, ErrorCode> {
        let (reply, answer) = mpsc::channel();
        let id = {
            let mut shared = self.lock();
            let allowed = match &shared.status {
                Status::Ready { .. } => true,
                Status::Starting => during_start,
                Status::Stopping => name == "shutdown",
                _ => false,
            };
            let Some(session) = shared.session.as_mut().filter(|_| allowed) else {
                return Err(ErrorCode::new(
                    "ui.engine.not_ready",
                    json!({ "name": name }),
                ));
            };
            let id = format!("ui-{}", session.next_id);
            session.next_id += 1;
            let message = json!({ "v": PROTOCOL_VERSION, "kind": "cmd", "id": id, "name": name, "data": data });
            let line = message.to_string();
            if line.len() > session.max_line {
                return Err(ErrorCode::new(
                    "ui.engine.line_too_long",
                    json!({ "limit_bytes": session.max_line, "line_bytes": line.len() }),
                ));
            }
            session.pending.insert(id.clone(), reply);
            let started = session.started;
            if let Err(error) = write_line(session, &id, line.as_bytes()) {
                session.pending.remove(&id);
                log::error!("cannot write to the engine: {error}");
                return Err(ErrorCode::new("ui.engine.write_failed", json!({})));
            }
            drop(shared);
            self.record(started, "send", &message);
            id
        };
        match answer.recv_timeout(timeout) {
            Ok(result) => result,
            Err(_) => {
                if let Some(session) = self.lock().session.as_mut() {
                    session.pending.remove(&id);
                }
                log::warn!(
                    "no answer to {name} ({id}) within {} s",
                    timeout.as_secs_f32()
                );
                Err(ErrorCode::new(
                    "ui.engine.request_timeout",
                    json!({ "name": name, "seconds": timeout.as_secs() }),
                ))
            }
        }
    }

    fn read_stdout(
        &self,
        generation: u64,
        stdout: ChildStdout,
        child: Arc<Mutex<Child>>,
        started: Instant,
    ) {
        let mut reader = BufReader::new(stdout);
        let mut line = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) => break,
                Ok(_) => self.handle_line(generation, &line, started),
                Err(error) => {
                    log::error!("reading the engine's stdout failed: {error}");
                    break;
                }
            }
        }
        self.handle_exit(generation, &child);
    }

    fn handle_line(&self, generation: u64, line: &[u8], started: Instant) {
        let message: Value = match serde_json::from_slice(line) {
            Ok(message @ Value::Object(_)) => message,
            _ => {
                log::error!(
                    "the engine's stdout carried a line that is not a protocol message: {:?}",
                    String::from_utf8_lossy(&line[..line.len().min(200)])
                );
                return;
            }
        };
        self.record(started, "recv", &message);
        match message["kind"].as_str() {
            Some("result") => {
                let id = message["id"].as_str().unwrap_or_default();
                let result = if message["ok"] == Value::Bool(true) {
                    Ok(message["data"].clone())
                } else {
                    Err(ErrorCode::from_wire(&message["error"]))
                };
                self.answer(generation, id, result);
            }
            Some("event") if message["name"] == "protocol.rejected" => {
                let line_number = message["data"]["line"].as_u64().unwrap_or(0);
                let error = ErrorCode::from_wire(&message["data"]["error"]);
                let id = self.lock().session.as_ref().and_then(|session| {
                    (session.generation == generation)
                        .then(|| session.id_at_line.get(&line_number).cloned())
                        .flatten()
                });
                match id {
                    Some(id) => self.answer(generation, &id, Err(error)),
                    None => log::warn!(
                        "the engine rejected line {line_number}, which no request is waiting on"
                    ),
                }
            }
            Some("event") => (self.on_event)(&message),
            _ => log::warn!("unexpected message from the engine: {message}"),
        }
    }

    fn answer(&self, generation: u64, id: &str, result: Result<Value, ErrorCode>) {
        let reply = {
            let mut shared = self.lock();
            let Some(session) = shared
                .session
                .as_mut()
                .filter(|s| s.generation == generation)
            else {
                return;
            };
            if let Some(line) = session.line_of_id.remove(id) {
                session.id_at_line.remove(&line);
            }
            session.pending.remove(id)
        };
        match reply {
            Some(reply) => {
                let _ = reply.send(result);
            }
            None => log::warn!(
                "a result for {id}, which no request is waiting on (it may have timed out)"
            ),
        }
    }

    fn handle_exit(&self, generation: u64, child: &Mutex<Child>) {
        let deadline = Instant::now() + EXIT_GRACE;
        let exit = loop {
            let mut child = child
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if Instant::now() < deadline => {
                    drop(child);
                    thread::sleep(Duration::from_millis(20));
                }
                Ok(None) => {
                    log::warn!("the engine closed its stdout but did not exit; killing it");
                    let _ = child.kill();
                    break child.wait().ok();
                }
                Err(error) => {
                    log::error!("cannot read the engine's exit status: {error}");
                    break None;
                }
            }
        };
        let (code, signal) = exit.map_or((None, None), exit_parts);
        let mut shared = self.lock();
        if shared.generation != generation {
            return;
        }
        let Some(session) = shared.session.take() else {
            return;
        };
        for (_, reply) in session.pending {
            let _ = reply.send(Err(ErrorCode::new("ui.engine.exited", json!({}))));
        }
        let status = match (&shared.status, session.exit_expected) {
            (Status::Failed { .. }, _) => shared.status.clone(),
            (_, true) => Status::Stopped { code, signal },
            (_, false) => Status::Exited { code, signal },
        };
        if session.exit_expected {
            log::info!(
                "engine process {} exited (code {code:?}, signal {signal:?})",
                session.pid
            );
        } else {
            log::error!(
                "engine process {} ended unexpectedly (code {code:?}, signal {signal:?})",
                session.pid
            );
        }
        self.set_status(&mut shared, status);
    }

    fn kill(&self, generation: u64) {
        let child = {
            let mut shared = self.lock();
            match shared
                .session
                .as_mut()
                .filter(|s| s.generation == generation)
            {
                Some(session) => {
                    session.exit_expected = true;
                    Arc::clone(&session.child)
                }
                None => return,
            }
        };
        let mut child = child
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Err(error) = child.kill() {
            log::warn!("cannot kill the engine: {error}");
        }
    }

    fn shutdown(&self) {
        let generation = {
            let mut shared = self.lock();
            let generation = shared.generation;
            let Some(session) = shared.session.as_mut() else {
                return;
            };
            session.exit_expected = true;
            if !matches!(shared.status, Status::Failed { .. }) {
                self.set_status(&mut shared, Status::Stopping);
            }
            generation
        };
        let started = Instant::now();
        let timeout = self.config.shutdown_timeout;
        match self.request("shutdown", json!({}), timeout, false) {
            Ok(_) => log::info!("the engine accepted shutdown"),
            Err(error) => log::warn!("shutdown was not answered: {}", error.code),
        }
        let left = timeout.saturating_sub(started.elapsed());
        if !self.wait_session_gone(generation, left) {
            log::warn!(
                "the engine did not exit within {} s of shutdown; killing it",
                timeout.as_secs()
            );
            self.kill(generation);
            self.wait_session_gone(generation, EXIT_GRACE + Duration::from_secs(1));
        }
    }

    fn wait_session_gone(&self, generation: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut shared = self.lock();
        loop {
            let gone = shared
                .session
                .as_ref()
                .is_none_or(|s| s.generation != generation);
            let left = deadline.saturating_duration_since(Instant::now());
            if gone || left.is_zero() {
                return gone;
            }
            shared = self
                .changed
                .wait_timeout(shared, left)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner().0);
        }
    }
}

/// Writes one line and remembers which request it carried, so a
/// `protocol.rejected` event naming that line fails that request.
fn write_line(session: &mut Session, id: &str, bytes: &[u8]) -> std::io::Result<()> {
    let stdin = session
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::BrokenPipe))?;
    stdin.write_all(bytes)?;
    stdin.write_all(b"\n")?;
    stdin.flush()?;
    session.lines_written += 1;
    session
        .id_at_line
        .insert(session.lines_written, id.to_owned());
    session
        .line_of_id
        .insert(id.to_owned(), session.lines_written);
    Ok(())
}

/// The engine logs to stderr as well as its own file. A pipe nobody reads
/// fills after a few tens of KiB and then blocks the engine, so this reads
/// it to the end, into the log at debug.
fn drain_stderr(stderr: ChildStderr) {
    let mut reader = BufReader::new(stderr);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => log::debug!(target: "engine", "{}", String::from_utf8_lossy(&line).trim_end()),
        }
    }
}

/// Result fields that never reach the transcript or the log, as the engine's
/// own `_NOT_LOGGED`: a source's clear frame (a revealed thumbnail) must not
/// be written anywhere, and the transcript is a file.
const NOT_RECORDED: &[(&str, &[&str])] = &[("catalog.thumbnail", &["data"])];

/// The message as it may be written: None when nothing in it is withheld,
/// else a copy with each withheld field replaced by its length.
fn redact(message: &Value) -> Option<Value> {
    if message["kind"] != "result" {
        return None;
    }
    let name = message["name"].as_str()?;
    let (_, fields) = NOT_RECORDED.iter().find(|(command, _)| *command == name)?;
    let mut copy = message.clone();
    let data = copy.get_mut("data")?.as_object_mut()?;
    for field in *fields {
        if let Some(value) = data.get_mut(*field) {
            let length = value.as_str().map_or(0, str::len);
            *value = json!(format!("<{length} bytes not recorded>"));
        }
    }
    Some(copy)
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
