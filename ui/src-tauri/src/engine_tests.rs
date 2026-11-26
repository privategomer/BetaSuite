//! The supervisor against the real `betaengine.py serve` (with its log in a
//! temporary folder), and against a fake engine for what the real one will
//! not do: ignore shutdown, or never answer. Python comes from PYTHON_BIN,
//! else PATH; serve needs only the standard library.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

const WAIT: Duration = Duration::from_secs(30);

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn python() -> PathBuf {
    locate_python(&repo()).expect("a Python 3 interpreter (PYTHON_BIN or python3 on PATH)")
}

/// A temporary folder, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let dir = env::temp_dir().join(format!(
            "betasuite-engine-test-{name}-{}-{}",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir_all(&dir).expect("scratch folder");
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn config(scratch: &Scratch) -> Config {
    let mut config = Config::new("test");
    config.engine_dir = Some(repo());
    config.python = Some(python());
    config.args = vec![
        "--log-file".into(),
        scratch.0.join("engine.log").to_string_lossy().into_owned(),
    ];
    config.request_timeout = Duration::from_secs(10);
    config.transcript = Some(scratch.0.join("transcript.jsonl"));
    config
}

/// An engine that answers hello, then ignores every line, shutdown and end
/// of input included.
/// Options after `serve`: --fake-limit N reports that line limit,
/// --fake-hello-delay S answers hello S seconds late, --fake-hello-error
/// answers it with an error.
const FAKE_ENGINE: &str = r#"
import json, sys, time
args = sys.argv[1:]
def option(name, default):
    return args[args.index(name) + 1] if name in args else default
limit = int(option("--fake-limit", 1048576))
delay = float(option("--fake-hello-delay", 0))
for line in sys.stdin:
    message = json.loads(line)
    if message.get("name") != "hello":
        continue
    time.sleep(delay)
    reply = {"v": 1, "kind": "result", "id": message["id"], "name": "hello"}
    if "--fake-hello-error" in args:
        reply.update(ok=False, error={"code": "protocol.unsupported_version", "params": {"got": [1], "supported": [9]}})
    else:
        reply.update(ok=True, data={"protocol": 1, "engine": {"version": "fake", "python": "3", "platform": "test"},
            "pid": 1, "commands": [], "events": [], "limits": {"max_line_bytes": limit}})
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
while True:
    time.sleep(1)
"#;

fn fake_config(scratch: &Scratch) -> Config {
    let dir = scratch.0.join("fake");
    fs::create_dir_all(&dir).expect("fake engine folder");
    fs::write(dir.join("betaengine.py"), FAKE_ENGINE).expect("fake engine");
    let mut config = config(scratch);
    config.engine_dir = Some(dir);
    config
}

fn engine(config: Config) -> Engine {
    Engine::new(config, Arc::new(|_: &Status| {}), Arc::new(|_: &Value| {}))
}

fn settled(status: &Status) -> bool {
    !matches!(status, Status::Idle | Status::Starting | Status::Stopping)
}

fn wait_ready(engine: &Engine) -> u32 {
    match engine.wait_for(WAIT, settled) {
        Status::Ready { pid, .. } => pid,
        other => panic!("expected ready, got {other:?}"),
    }
}

#[cfg(unix)]
fn kill9(pid: u32) {
    let status = Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .expect("kill");
    assert!(status.success());
}

#[cfg(unix)]
fn alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn lines_written(engine: &Engine) -> u64 {
    engine
        .inner
        .lock()
        .session
        .as_ref()
        .map_or(0, |s| s.lines_written)
}

#[test]
fn hello_reaches_ready_with_the_engines_version_and_limit() {
    let scratch = Scratch::new("hello");
    let engine = engine(config(&scratch));
    engine.start();
    match engine.wait_for(WAIT, settled) {
        Status::Ready {
            version,
            protocol,
            pid,
        } => {
            assert!(!version.is_empty());
            assert_eq!(protocol, 1);
            assert!(pid > 0);
        }
        other => panic!("expected ready, got {other:?}"),
    }
    assert_eq!(
        engine.inner.lock().session.as_ref().map(|s| s.max_line),
        Some(1 << 20)
    );
    engine.shutdown();
    assert_eq!(
        engine.status(),
        Status::Stopped {
            code: Some(0),
            signal: None
        }
    );
    let transcript = fs::read_to_string(scratch.0.join("transcript.jsonl")).expect("transcript");
    assert!(transcript.contains(r#""name":"hello""#), "{transcript}");
}

#[test]
fn concurrent_requests_each_get_their_own_result() {
    let scratch = Scratch::new("concurrent");
    let engine = engine(config(&scratch));
    engine.start();
    wait_ready(&engine);
    let workers: Vec<_> = (0..8)
        .map(|worker| {
            let engine = engine.clone();
            thread::spawn(move || {
                for n in 0..25 {
                    if (worker + n) % 2 == 0 {
                        assert_eq!(engine.request("ping", json!({})), Ok(json!({})));
                    } else {
                        // An unknown command answers with its own name, so a
                        // result handed to the wrong request shows up here.
                        let name = format!("nope.w{worker}_n{n}");
                        let error = engine
                            .request(&name, json!({}))
                            .expect_err("unknown command");
                        assert_eq!(error.code, "command.unknown");
                        assert_eq!(error.params["name"], json!(name));
                    }
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().expect("worker");
    }
    assert!(engine
        .inner
        .lock()
        .session
        .as_ref()
        .is_some_and(|s| s.pending.is_empty()));
    engine.shutdown();
}

#[test]
fn a_line_the_engine_cannot_read_fails_exactly_that_request() {
    let scratch = Scratch::new("rejected");
    let engine = engine(config(&scratch));
    engine.start();
    wait_ready(&engine);
    assert_eq!(engine.request("ping", json!({})), Ok(json!({})));
    let (reply, answer) = mpsc::channel();
    {
        let mut shared = engine.inner.lock();
        let session = shared.session.as_mut().expect("session");
        session.pending.insert("ui-raw".into(), reply);
        write_line(session, "ui-raw", b"{this is not json").expect("write");
    }
    let result = answer
        .recv_timeout(WAIT)
        .expect("an answer through protocol.rejected");
    assert_eq!(result.expect_err("rejected").code, "protocol.bad_json");
    assert_eq!(engine.request("ping", json!({})), Ok(json!({})));
    engine.shutdown();
}

#[test]
fn an_oversized_request_is_refused_before_it_is_written() {
    let scratch = Scratch::new("oversized");
    let engine = engine(config(&scratch));
    engine.start();
    wait_ready(&engine);
    let before = lines_written(&engine);
    let blob = "x".repeat((1 << 20) + 1);
    let error = engine
        .request("ping", json!({ "blob": blob }))
        .expect_err("too long");
    assert_eq!(error.code, "ui.engine.line_too_long");
    assert_eq!(error.params["limit_bytes"], json!(1 << 20));
    assert_eq!(lines_written(&engine), before);
    assert_eq!(engine.request("ping", json!({})), Ok(json!({})));
    engine.shutdown();
}

#[test]
fn a_flood_on_stderr_never_stalls_requests() {
    let scratch = Scratch::new("stderr");
    let mut config = config(&scratch);
    // Every message is logged at trace, so stderr carries two lines per
    // ping, far more than a pipe buffer holds.
    config
        .args
        .extend(["--console-level".into(), "trace".into()]);
    let engine = engine(config);
    engine.start();
    wait_ready(&engine);
    for _ in 0..3000 {
        assert_eq!(engine.request("ping", json!({})), Ok(json!({})));
    }
    engine.shutdown();
    assert_eq!(
        engine.status(),
        Status::Stopped {
            code: Some(0),
            signal: None
        }
    );
}

#[cfg(unix)]
#[test]
fn a_killed_engine_is_reported_with_its_signal() {
    let scratch = Scratch::new("killed");
    let engine = engine(config(&scratch));
    engine.start();
    let pid = wait_ready(&engine);
    kill9(pid);
    assert_eq!(
        engine.wait_for(WAIT, |s| matches!(s, Status::Exited { .. })),
        Status::Exited {
            code: None,
            signal: Some(9)
        }
    );
    assert_eq!(
        engine.request("ping", json!({})).expect_err("gone").code,
        "ui.engine.not_ready"
    );
}

#[cfg(unix)]
#[test]
fn requests_waiting_when_the_engine_dies_fail_with_exited() {
    let scratch = Scratch::new("pending");
    let engine = engine(fake_config(&scratch));
    engine.start();
    wait_ready(&engine);
    let pid = engine
        .inner
        .lock()
        .session
        .as_ref()
        .map(|s| s.pid)
        .expect("pid");
    let waiting = {
        let engine = engine.clone();
        thread::spawn(move || engine.request("ping", json!({})))
    };
    let deadline = Instant::now() + WAIT;
    while engine
        .inner
        .lock()
        .session
        .as_ref()
        .is_some_and(|s| s.pending.is_empty())
    {
        assert!(Instant::now() < deadline, "the ping never went out");
        thread::sleep(Duration::from_millis(10));
    }
    kill9(pid);
    let result = waiting.join().expect("request thread");
    assert_eq!(result.expect_err("engine died").code, "ui.engine.exited");
    assert!(matches!(engine.status(), Status::Exited { .. }));
}

#[cfg(unix)]
#[test]
fn shutdown_stops_the_engine_and_its_process() {
    let scratch = Scratch::new("shutdown");
    let engine = engine(config(&scratch));
    engine.start();
    let pid = wait_ready(&engine);
    engine.shutdown();
    assert_eq!(
        engine.status(),
        Status::Stopped {
            code: Some(0),
            signal: None
        }
    );
    assert!(!alive(pid), "process {pid} is still running");
    let transcript = fs::read_to_string(scratch.0.join("transcript.jsonl")).expect("transcript");
    let accepted = transcript.lines().any(|line| {
        let record: Value = serde_json::from_str(line).expect("transcript line");
        record["dir"] == "recv"
            && record["message"]["name"] == "shutdown"
            && record["message"]["ok"] == true
    });
    assert!(accepted, "{transcript}");
}

#[cfg(unix)]
#[test]
fn an_engine_that_ignores_shutdown_is_killed_after_the_timeout() {
    let scratch = Scratch::new("stubborn");
    let mut config = fake_config(&scratch);
    config.shutdown_timeout = Duration::from_secs(1);
    let engine = engine(config);
    engine.start();
    wait_ready(&engine);
    let pid = engine
        .inner
        .lock()
        .session
        .as_ref()
        .map(|s| s.pid)
        .expect("pid");
    let started = Instant::now();
    engine.shutdown();
    let took = started.elapsed();
    assert_eq!(
        engine.status(),
        Status::Stopped {
            code: None,
            signal: Some(9)
        }
    );
    assert!(
        took >= Duration::from_secs(1),
        "killed before the timeout: {took:?}"
    );
    assert!(took < Duration::from_secs(5), "took {took:?}");
    assert!(!alive(pid), "process {pid} is still running");
}

#[test]
fn a_missing_engine_or_python_fails_with_its_code() {
    let scratch = Scratch::new("missing");
    let mut no_engine = config(&scratch);
    no_engine.engine_dir = Some(scratch.0.join("nowhere"));
    let engine1 = engine(no_engine);
    engine1.start();
    assert_eq!(
        engine1.wait_for(WAIT, settled),
        Status::Failed {
            error: ErrorCode::new("ui.engine.not_found", json!({}))
        }
    );

    let mut no_python = config(&scratch);
    no_python.python = Some(scratch.0.join("no-such-python"));
    let engine2 = engine(no_python);
    engine2.start();
    assert_eq!(
        engine2.wait_for(WAIT, settled),
        Status::Failed {
            error: ErrorCode::new("ui.engine.python_not_found", json!({}))
        }
    );
}

#[cfg(unix)]
#[test]
fn restart_after_a_kill_reaches_ready_again() {
    let scratch = Scratch::new("restart");
    let engine = engine(config(&scratch));
    engine.start();
    let first = wait_ready(&engine);
    kill9(first);
    engine.wait_for(WAIT, |s| matches!(s, Status::Exited { .. }));
    engine.restart();
    let second = wait_ready(&engine);
    assert_ne!(first, second);
    assert_eq!(engine.request("ping", json!({})), Ok(json!({})));
    engine.shutdown();
    assert!(matches!(engine.status(), Status::Stopped { .. }));
}

#[test]
fn the_line_limit_comes_from_hello() {
    let scratch = Scratch::new("limit");
    let mut config = fake_config(&scratch);
    config.args.extend(["--fake-limit".into(), "200".into()]);
    let engine = engine(config);
    engine.start();
    wait_ready(&engine);
    let error = engine
        .request("ping", json!({ "blob": "x".repeat(300) }))
        .expect_err("over the engine's limit");
    assert_eq!(error.code, "ui.engine.line_too_long");
    assert_eq!(error.params["limit_bytes"], json!(200));
    engine.shutdown();
}

#[test]
fn requests_wait_for_the_handshake() {
    let scratch = Scratch::new("early");
    let mut config = fake_config(&scratch);
    config
        .args
        .extend(["--fake-hello-delay".into(), "1".into()]);
    let engine = engine(config);
    engine.start();
    let deadline = Instant::now() + WAIT;
    while engine.inner.lock().session.is_none() {
        assert!(Instant::now() < deadline, "the engine never started");
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(engine.status(), Status::Starting);
    let error = engine.request("ping", json!({})).expect_err("before hello");
    assert_eq!(error.code, "ui.engine.not_ready");
    wait_ready(&engine);
    engine.shutdown();
}

#[test]
fn a_refused_handshake_fails_and_stays_failed() {
    let scratch = Scratch::new("refused");
    let mut config = fake_config(&scratch);
    config.args.push("--fake-hello-error".into());
    let engine = engine(config);
    engine.start();
    let expected = Status::Failed {
        error: ErrorCode::new(
            "ui.engine.hello_failed",
            json!({ "code": "protocol.unsupported_version" }),
        ),
    };
    assert_eq!(engine.wait_for(WAIT, settled), expected);
    // The process is killed after the failure; its exit must not replace it.
    let deadline = Instant::now() + WAIT;
    while engine.inner.lock().session.is_some() {
        assert!(
            Instant::now() < deadline,
            "the refused engine was not stopped"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(engine.status(), expected);
}

#[test]
fn a_thumbnails_image_never_reaches_the_transcript() {
    let result = json!({ "v": 1, "kind": "result", "id": "ui-1", "name": "catalog.thumbnail", "ok": true,
                         "data": { "mime": "image/png", "data": "iVBORw0KGgoAAAA", "width": 320, "height": 180 } });
    let shown = redact(&result).expect("a thumbnail result is redacted");
    assert_eq!(shown["data"]["data"], "<15 bytes not recorded>");
    assert_eq!(shown["data"]["width"], 320);
    assert_eq!(shown["id"], "ui-1");
    // Other results, and the request itself, are written as they are.
    let other = json!({ "v": 1, "kind": "result", "id": "ui-2", "name": "catalog.get", "ok": true, "data": { "data": "x" } });
    assert!(redact(&other).is_none());
    let request = json!({ "v": 1, "kind": "cmd", "id": "ui-1", "name": "catalog.thumbnail", "data": { "item": "a", "reveal": true } });
    assert!(redact(&request).is_none());
}

#[test]
fn the_transcript_file_gets_the_redacted_thumbnail() {
    let scratch = Scratch::new("redact");
    let engine = engine(config(&scratch));
    let result = json!({ "v": 1, "kind": "result", "id": "ui-1", "name": "catalog.thumbnail", "ok": true,
                         "data": { "mime": "image/png", "data": "iVBORw0KGgoAAAA" } });
    engine.inner.record(Instant::now(), "recv", &result);
    let written = fs::read_to_string(scratch.0.join("transcript.jsonl")).expect("transcript");
    assert!(!written.contains("iVBORw0KGgo"), "{written}");
    assert!(written.contains("<15 bytes not recorded>"), "{written}");
}
