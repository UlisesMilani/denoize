use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TestRoot(PathBuf);

impl TestRoot {
    fn new(label: &str) -> Self {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "denoize-watch-{label}-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn join(&self, path: impl AsRef<Path>) -> PathBuf {
        self.0.join(path)
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_wav(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in 0..640 {
        writer
            .write_sample(if frame % 40 < 20 {
                1_000_i16
            } else {
                -1_000_i16
            })
            .unwrap();
    }
    writer.finalize().unwrap();
}

fn run_watch(input: &Path, output: &Path, secret: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg("watch")
        .arg(input)
        .arg(output)
        .arg("--once")
        .arg("--settle-ms")
        .arg("0")
        .arg("--receipt-key")
        .arg(secret)
        .arg("--json")
        .args(extra)
        .output()
        .unwrap()
}

fn json_line(output: &Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let lines: Vec<_> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 1, "unexpected JSON lines: {lines:?}");
    lines.into_iter().next().unwrap()
}

#[test]
fn once_processes_a_settled_file_with_a_verifiable_receipt_and_restart_skip() {
    let root = TestRoot::new("success");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir(&input).unwrap();
    write_wav(&input.join("clip.wav"));
    let secret = root.join("receipt-secret.json");
    let public = root.join("receipt-public.json");
    denoize::write_new_receipt_keypair(&secret, &public).unwrap();

    let first = run_watch(&input, &output, &secret, &[]);
    let first_json = json_line(&first);
    assert_eq!(first_json["schema"], "denoize-watch-cycle-v1");
    assert_eq!(first_json["schema_version"], 1);
    assert!(first_json.get("schemaVersion").is_none());
    assert_eq!(first_json["attempted"], 1);
    assert_eq!(first_json["succeeded"], 1);
    let audio = output.join("clip.wav");
    let receipt = output
        .join(".denoize-receipts")
        .join("clip.wav.receipt.json");
    assert!(audio.is_file());
    assert!(receipt.is_file());

    let verify = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg("receipts")
        .arg("verify")
        .arg(&receipt)
        .arg("--key")
        .arg(&public)
        .arg("--output-root")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );

    let state_path = output.join(".denoize-watch-state.json");
    let mut state: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&state_path).unwrap()).unwrap();
    let jobs = state["jobs"].as_object_mut().unwrap();
    let (_, job) = jobs.iter_mut().next().unwrap();
    job["status"] = "processing".into();
    std::fs::write(&state_path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();

    let recovered = run_watch(&input, &output, &secret, &[]);
    let recovered_json = json_line(&recovered);
    assert_eq!(recovered_json["attempted"], 1);
    assert_eq!(recovered_json["succeeded"], 1);

    let skipped = run_watch(&input, &output, &secret, &[]);
    let skipped_json = json_line(&skipped);
    assert_eq!(skipped_json["attempted"], 0);
    assert_eq!(skipped_json["succeeded"], 0);

    let resource_policy_changed = run_watch(&input, &output, &secret, &["--max-memory", "64"]);
    let resource_policy_json = json_line(&resource_policy_changed);
    assert_eq!(resource_policy_json["attempted"], 0);
    assert_eq!(resource_policy_json["succeeded"], 0);

    let state_before_change = std::fs::read(&state_path).unwrap();
    let audio_before_change = denoize::batch_resume::fingerprint_file(&audio).unwrap();
    let changed = run_watch(&input, &output, &secret, &["--strength", "0.2"]);
    assert!(!changed.status.success());
    assert!(
        String::from_utf8_lossy(&changed.stderr).contains("different processing template"),
        "{}",
        String::from_utf8_lossy(&changed.stderr)
    );
    assert_eq!(std::fs::read(&state_path).unwrap(), state_before_change);
    assert_eq!(
        denoize::batch_resume::fingerprint_file(&audio).unwrap(),
        audio_before_change
    );
}

#[test]
fn permanent_failure_is_quarantined_with_a_reason_and_no_output() {
    let root = TestRoot::new("quarantine");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir(&input).unwrap();
    let source = input.join("broken.wav");
    std::fs::write(&source, b"not a wave file").unwrap();
    let secret = root.join("receipt-secret.json");
    let public = root.join("receipt-public.json");
    denoize::write_new_receipt_keypair(&secret, &public).unwrap();

    let result = run_watch(&input, &output, &secret, &["--max-attempts", "1"]);
    let json = json_line(&result);
    assert_eq!(json["attempted"], 1);
    assert_eq!(json["quarantined"], 1);
    assert!(!source.exists());
    assert!(!output.join("broken.wav").exists());
    let quarantined = output.join(".denoize-quarantine").join("broken.wav");
    assert_eq!(std::fs::read(&quarantined).unwrap(), b"not a wave file");
    let reason: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            output
                .join(".denoize-quarantine")
                .join("broken.wav.denoize-watch.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(reason["schema"], "denoize-watch-quarantine-v1");
    assert_eq!(reason["attempts"], 1);
}

#[test]
fn recursive_mode_preserves_relative_output_and_receipt_paths() {
    let root = TestRoot::new("recursive");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir_all(input.join("speaker/day")).unwrap();
    write_wav(&input.join("speaker/day/clip.wav"));
    let secret = root.join("receipt-secret.json");
    let public = root.join("receipt-public.json");
    denoize::write_new_receipt_keypair(&secret, &public).unwrap();

    let result = run_watch(&input, &output, &secret, &["--recursive"]);
    let json = json_line(&result);
    assert_eq!(json["succeeded"], 1);
    assert!(output.join("speaker/day/clip.wav").is_file());
    assert!(output
        .join(".denoize-receipts/speaker/day/clip.wav.receipt.json")
        .is_file());
}

#[test]
fn invalid_watch_configuration_does_not_create_output() {
    let root = TestRoot::new("invalid");
    let input = root.join("input");
    let output = input.join("nested-output");
    std::fs::create_dir(&input).unwrap();
    let secret = root.join("receipt-secret.json");
    let public = root.join("receipt-public.json");
    denoize::write_new_receipt_keypair(&secret, &public).unwrap();
    let result = run_watch(&input, &output, &secret, &[]);
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("must not overlap"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!output.exists());
}
