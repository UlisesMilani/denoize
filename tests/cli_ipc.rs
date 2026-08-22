use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

struct IpcFixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    state: PathBuf,
    discovery: PathBuf,
    admin: PathBuf,
    worker: PathBuf,
    server: Option<Child>,
}

impl IpcFixture {
    fn start() -> Self {
        let temp = tempfile::tempdir().expect("create IPC fixture root");
        let root = temp.path().to_path_buf();
        let state = root.join("state");
        let discovery = state.join("discovery.json");
        let admin = root.join("admin.json");
        let worker = root.join("worker.json");
        std::fs::create_dir(root.join("input")).expect("create input root");
        std::fs::create_dir(root.join("output")).expect("create output root");

        let initialized = run(&[
            "ipc",
            "init",
            "--state-dir",
            path(&state),
            "--admin-grant",
            path(&admin),
            "--request-timeout-ms",
            "5000",
            "--planning-timeout-ms",
            "20000",
            "--job-timeout-ms",
            "30000",
            "--max-memory",
            "512",
            "--max-temp-space",
            "512",
            "--max-history",
            "8",
        ]);
        assert_success(&initialized);

        let server = Command::new(env!("CARGO_BIN_EXE_denoize"))
            .args(["ipc", "serve", "--state-dir", path(&state)])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start IPC server");
        let mut fixture = Self {
            _temp: temp,
            root,
            state,
            discovery,
            admin,
            worker,
            server: Some(server),
        };
        fixture.wait_for_discovery();
        fixture
    }

    fn wait_for_discovery(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if self.discovery.is_file() {
                return;
            }
            if let Some(status) = self
                .server
                .as_mut()
                .expect("server exists")
                .try_wait()
                .expect("poll IPC server")
            {
                let output = self
                    .server
                    .take()
                    .expect("server exists")
                    .wait_with_output()
                    .expect("collect failed IPC server");
                panic!(
                    "IPC server exited early with {status}:\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            assert!(Instant::now() < deadline, "IPC discovery was not published");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn client_args<'a>(&'a self, grant: &'a Path) -> [&'a str; 4] {
        ["--discovery", path(&self.discovery), "--grant", path(grant)]
    }

    fn stop(&mut self) {
        if self.server.is_none() {
            return;
        }
        let mut arguments = vec!["ipc", "shutdown", "--force"];
        arguments.extend(self.client_args(&self.admin));
        let shutdown = run(&arguments);
        assert_success(&shutdown);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if self
                .server
                .as_mut()
                .expect("server exists")
                .try_wait()
                .expect("poll IPC server shutdown")
                .is_some()
            {
                let output = self
                    .server
                    .take()
                    .expect("server exists")
                    .wait_with_output()
                    .expect("collect IPC server");
                assert!(
                    output.status.success(),
                    "IPC server failed:\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                return;
            }
            assert!(Instant::now() < deadline, "IPC server did not stop");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for IpcFixture {
    fn drop(&mut self) {
        if let Some(mut server) = self.server.take() {
            let _ = server.kill();
            let _ = server.wait();
        }
    }
}

fn path(value: &Path) -> &str {
    value.to_str().expect("test path is UTF-8")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .expect("run denoize IPC command")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "command failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn result(output: &Output) -> Value {
    assert_success(output);
    serde_json::from_slice(&output.stdout).expect("parse IPC command result")
}

fn write_wav(path: &Path, frames: usize) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create IPC input WAV");
    for frame in 0..frames {
        let sample = if frame % 80 < 40 {
            2_000_i16
        } else {
            -2_000_i16
        };
        writer.write_sample(sample).expect("write IPC input WAV");
    }
    writer.finalize().expect("finalize IPC input WAV");
}

#[test]
fn authenticated_ipc_dry_run_job_history_revocation_and_shutdown() {
    let mut fixture = IpcFixture::start();
    let input = fixture.root.join("input/tone.wav");
    let output = fixture.root.join("output/clean.wav");
    write_wav(&input, 3_200);

    let policy = serde_json::json!({
        "label": "integration-worker",
        "capabilities": ["plan", "submit", "read-own", "control-own"],
        "input_roots": [path(&fixture.root.join("input"))],
        "output_roots": [path(&fixture.root.join("output"))],
        "max_priority": 10,
        "expires_at_unix_millis": null
    });
    let policy_path = fixture.root.join("worker-policy.json");
    std::fs::write(
        &policy_path,
        serde_json::to_vec_pretty(&policy).expect("serialize worker policy"),
    )
    .expect("write worker policy");
    let mut grant_arguments = vec![
        "ipc",
        "grant",
        "create",
        path(&policy_path),
        path(&fixture.worker),
    ];
    grant_arguments.extend(fixture.client_args(&fixture.admin));
    assert_success(&run(&grant_arguments));

    let mut ping_arguments = vec!["ipc", "ping"];
    ping_arguments.extend(fixture.client_args(&fixture.worker));
    assert_eq!(result(&run(&ping_arguments))["type"], "pong");

    let worker_bytes = std::fs::read(&fixture.worker).expect("read worker grant");
    let mut bad_grant: Value = serde_json::from_slice(&worker_bytes).expect("parse worker grant");
    bad_grant["token"] = Value::String("invalid-token-value-that-is-long-enough".into());
    std::fs::write(
        &fixture.worker,
        serde_json::to_vec_pretty(&bad_grant).expect("serialize bad grant"),
    )
    .expect("write bad grant");
    let denied = run(&ping_arguments);
    assert!(!denied.status.success());
    let denied_stderr = String::from_utf8_lossy(&denied.stderr);
    assert!(
        denied_stderr.contains("authentication failed"),
        "{denied_stderr}"
    );
    assert!(!denied_stderr.contains("invalid-token-value"));
    std::fs::write(&fixture.worker, worker_bytes).expect("restore worker grant");

    let mut dry_run_arguments = vec!["ipc", "dry-run", "file", path(&input), path(&output)];
    dry_run_arguments.extend(fixture.client_args(&fixture.worker));
    dry_run_arguments.extend(["--", "--no-metadata"]);
    let dry_run = result(&run(&dry_run_arguments));
    assert_eq!(dry_run["type"], "dry-run");
    assert_eq!(dry_run["value"]["schema"], "denoize-job-dry-run-v1");
    assert_eq!(dry_run["value"]["destinations"]["create"], 1);
    assert!(!output.exists(), "dry-run created an output");

    let mut submit_arguments = vec![
        "ipc",
        "submit",
        "file",
        path(&input),
        path(&output),
        "--priority",
        "5",
    ];
    submit_arguments.extend(fixture.client_args(&fixture.worker));
    submit_arguments.extend(["--", "--no-metadata"]);
    let submitted = result(&run(&submit_arguments));
    assert_eq!(submitted["type"], "submitted");
    let job_id = submitted["value"]["job_id"]
        .as_str()
        .expect("submitted job ID")
        .to_owned();

    let deadline = Instant::now() + Duration::from_secs(30);
    let completed = loop {
        let mut status_arguments = vec!["ipc", "status", &job_id];
        status_arguments.extend(fixture.client_args(&fixture.worker));
        let status = result(&run(&status_arguments));
        if status["value"]["state"] == "completed" {
            break status;
        }
        assert_ne!(status["value"]["state"], "failed", "{status}");
        assert!(
            Instant::now() < deadline,
            "IPC job did not complete: {status}"
        );
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(completed["value"]["receipt"].is_object());
    let mut reader = hound::WavReader::open(&output).expect("open IPC output WAV");
    assert!(reader.samples::<i16>().next().is_some());

    let mut history_arguments = vec!["ipc", "history", "--limit", "8"];
    history_arguments.extend(fixture.client_args(&fixture.worker));
    let history = result(&run(&history_arguments));
    assert_eq!(history["type"], "history");
    assert_eq!(history["value"]["entries"][0]["job_id"], job_id);
    assert_eq!(history["value"]["entries"][0]["destinations"]["create"], 1);
    let encoded_history = serde_json::to_string(&history).expect("serialize history");
    assert!(!encoded_history.contains(path(&input)));
    assert!(!encoded_history.contains(path(&output)));

    let worker_document: Value =
        serde_json::from_slice(&std::fs::read(&fixture.worker).expect("read worker grant"))
            .expect("parse worker grant");
    let grant_id = worker_document["grant_id"]
        .as_str()
        .expect("worker grant ID");
    let mut revoke_arguments = vec!["ipc", "grant", "revoke", grant_id];
    revoke_arguments.extend(fixture.client_args(&fixture.admin));
    assert_success(&run(&revoke_arguments));
    let revoked = run(&ping_arguments);
    assert!(!revoked.status.success());
    assert!(String::from_utf8_lossy(&revoked.stderr).contains("revoked"));

    fixture.stop();
    assert!(!fixture.discovery.exists());
    assert!(fixture.state.join("queue.json").is_file());
}
