use serde_json::Value;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static NEXT_DIRECTORY_ID: AtomicUsize = AtomicUsize::new(0);

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn create() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "denoize-cli-execution-receipts-{}-{timestamp}-{}",
            std::process::id(),
            NEXT_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).expect("create unique test directory");
        Self { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn write_test_wav(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create test WAV");
    for frame in 0..1_600 {
        let sample = if frame % 80 < 40 {
            2_000_i16
        } else {
            -2_000_i16
        };
        writer.write_sample(sample).expect("write test sample");
    }
    writer.finalize().expect("finalize test WAV");
}

fn denoize(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .expect("run denoize")
}

fn denoize_with_stdin(args: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn denoize with stdin");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(input)
        .expect("write denoize stdin");
    child.wait_with_output().expect("collect denoize output")
}

fn denoize_with_timeout(args: &[&str], timeout: Duration) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn denoize");
    let started = Instant::now();
    loop {
        if child.try_wait().expect("poll denoize").is_some() {
            return child.wait_with_output().expect("collect denoize output");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child.wait_with_output().expect("collect timed-out denoize");
            panic!(
                "denoize did not finish within {timeout:?}:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "command failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn single_plan_receipt_and_offline_verification_match() {
    let root = TestDirectory::create();
    let input = root.join("input.wav");
    let output = root.join("output.wav");
    let plan = root.join("plan.json");
    let receipt = root.join("receipt.json");
    let secret = root.join("receipt-secret.json");
    let public = root.join("receipt-public.json");
    write_test_wav(&input);

    let generated = denoize(&[
        "receipts",
        "keygen",
        secret.to_str().unwrap(),
        public.to_str().unwrap(),
    ]);
    assert_success(&generated);

    let planned = denoize(&[
        "plan",
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--no-metadata",
        "--max-memory",
        "32",
    ]);
    assert_success(&planned);
    let plan_json: Value = serde_json::from_slice(&planned.stdout).expect("plan is JSON");
    assert_eq!(plan_json["schema"], "denoize-execution-plan-v1");
    assert_eq!(plan_json["kind"], "file");
    assert_eq!(plan_json["items"][0]["input"]["path"], "input.wav");
    assert_eq!(plan_json["items"][0]["output"]["path"], "output.wav");
    assert!(!output.exists(), "read-only plan created output");
    std::fs::write(&plan, &planned.stdout).expect("persist plan fixture");

    let executed = denoize(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--no-metadata",
        "--max-memory",
        "32",
        "--receipt",
        receipt.to_str().unwrap(),
        "--receipt-key",
        secret.to_str().unwrap(),
    ]);
    assert_success(&executed);
    assert!(output.is_file());
    assert!(receipt.is_file());

    let verified = denoize(&[
        "receipts",
        "verify",
        receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--plan",
        plan.to_str().unwrap(),
        "--output-root",
        root.path.to_str().unwrap(),
        "--json",
    ]);
    assert_success(&verified);
    let report: Value = serde_json::from_slice(&verified.stdout).expect("verification is JSON");
    assert_eq!(report["schema"], "denoize-receipt-verification-v1");
    assert_eq!(report["verified_items"].as_array().unwrap().len(), 1);

    std::fs::write(&output, b"changed").expect("tamper with output");
    let tampered = denoize(&[
        "receipts",
        "verify",
        receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--output-root",
        root.path.to_str().unwrap(),
    ]);
    assert!(!tampered.status.success());
    assert!(String::from_utf8_lossy(&tampered.stderr).contains("fingerprint mismatch"));
}

#[test]
fn bounded_stream_plans_use_v2_for_files_stdin_and_stdout_without_publication() {
    let root = TestDirectory::create();
    let input = root.join("input.wav");
    let file_output = root.join("output.flac");
    let stdin_output = root.join("stdin-output.wav");
    write_test_wav(&input);

    let file_plan = denoize(&[
        "plan",
        input.to_str().unwrap(),
        file_output.to_str().unwrap(),
        "--stream",
        "--stream-frames",
        "257",
        "--no-metadata",
    ]);
    assert_success(&file_plan);
    let file_json: Value = serde_json::from_slice(&file_plan.stdout).expect("stream plan JSON");
    assert_eq!(file_json["schema"], "denoize-execution-plan-v2");
    assert_eq!(file_json["schema_version"], 2);
    assert_eq!(file_json["kind"], "stream");
    assert_eq!(file_json["items"][0]["input"]["path"], "input.wav");
    assert_eq!(file_json["items"][0]["output"]["path"], "output.flac");
    assert_eq!(file_json["items"][0]["output"]["publication"], "no-clobber");
    assert_eq!(file_json["items"][0]["frames"], 1_600);
    assert!(!file_output.exists());

    let stdout_plan = denoize(&[
        "plan",
        input.to_str().unwrap(),
        "-",
        "--stream",
        "--output-format",
        "flac",
        "--no-metadata",
        "--max-temp-space",
        "64",
    ]);
    assert_success(&stdout_plan);
    let stdout_json: Value =
        serde_json::from_slice(&stdout_plan.stdout).expect("stdout stream plan JSON");
    assert_eq!(stdout_json["kind"], "stream");
    assert_eq!(stdout_json["items"][0]["output"]["path"], "-");
    assert_eq!(stdout_json["items"][0]["output"]["format"], "flac");
    assert_eq!(stdout_json["items"][0]["output"]["publication"], "stdout");
    assert_eq!(stdout_json["metadata_policy"], "drop");

    let wav = std::fs::read(&input).expect("read stdin fixture");
    let stdin_plan = denoize_with_stdin(
        &[
            "plan",
            "-",
            stdin_output.to_str().unwrap(),
            "--stream",
            "--no-metadata",
            "--max-temp-space",
            "64",
        ],
        &wav,
    );
    assert_success(&stdin_plan);
    let stdin_json: Value =
        serde_json::from_slice(&stdin_plan.stdout).expect("stdin stream plan JSON");
    assert_eq!(stdin_json["items"][0]["input"]["path"], "-");
    assert_eq!(stdin_json["items"][0]["output"]["path"], "stdin-output.wav");
    assert_eq!(stdin_json["items"][0]["frames"], 1_600);
    assert!(!stdin_output.exists());
    assert!(std::fs::read_dir(&root.path).unwrap().all(|entry| !entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains("denoize-stream")));
}

#[test]
fn bounded_stream_receipts_cover_files_spooled_stdin_and_captured_stdout() {
    let root = TestDirectory::create();
    let input = root.join("input.wav");
    let file_output = root.join("file-output.flac");
    let stdin_output = root.join("stdin-output.flac");
    let secret = root.join("secret.json");
    let public = root.join("public.json");
    write_test_wav(&input);
    let wav = std::fs::read(&input).unwrap();
    assert_success(&denoize(&[
        "receipts",
        "keygen",
        secret.to_str().unwrap(),
        public.to_str().unwrap(),
    ]));

    let file_plan_path = root.join("file-plan.json");
    let file_receipt = root.join("file-receipt.json");
    let file_plan = denoize(&[
        "plan",
        input.to_str().unwrap(),
        file_output.to_str().unwrap(),
        "--stream",
        "--stream-frames",
        "257",
        "--no-metadata",
    ]);
    assert_success(&file_plan);
    std::fs::write(&file_plan_path, &file_plan.stdout).unwrap();
    assert_success(&denoize(&[
        input.to_str().unwrap(),
        file_output.to_str().unwrap(),
        "--stream",
        "--stream-frames",
        "257",
        "--no-metadata",
        "--receipt",
        file_receipt.to_str().unwrap(),
        "--receipt-key",
        secret.to_str().unwrap(),
    ]));
    let file_receipt_json: Value =
        serde_json::from_slice(&std::fs::read(&file_receipt).unwrap()).unwrap();
    assert_eq!(file_receipt_json["schema"], "denoize-execution-receipt-v2");
    assert_eq!(file_receipt_json["payload"]["kind"], "stream");
    assert_success(&denoize(&[
        "receipts",
        "verify",
        file_receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--plan",
        file_plan_path.to_str().unwrap(),
        "--output-root",
        root.path.to_str().unwrap(),
    ]));

    let stdin_plan_path = root.join("stdin-plan.json");
    let stdin_receipt = root.join("stdin-receipt.json");
    let stdin_plan = denoize_with_stdin(
        &[
            "plan",
            "-",
            stdin_output.to_str().unwrap(),
            "--stream",
            "--stream-frames",
            "257",
            "--no-metadata",
            "--max-temp-space",
            "64",
        ],
        &wav,
    );
    assert_success(&stdin_plan);
    std::fs::write(&stdin_plan_path, &stdin_plan.stdout).unwrap();
    let stdin_run = denoize_with_stdin(
        &[
            "-",
            stdin_output.to_str().unwrap(),
            "--stream",
            "--stream-frames",
            "257",
            "--no-metadata",
            "--max-temp-space",
            "64",
            "--receipt",
            stdin_receipt.to_str().unwrap(),
            "--receipt-key",
            secret.to_str().unwrap(),
        ],
        &wav,
    );
    assert_success(&stdin_run);
    assert_success(&denoize(&[
        "receipts",
        "verify",
        stdin_receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--plan",
        stdin_plan_path.to_str().unwrap(),
        "--output-root",
        root.path.to_str().unwrap(),
    ]));

    let stdout_plan_path = root.join("stdout-plan.json");
    let stdout_receipt = root.join("stdout-receipt.json");
    let captured = root.join("captured.flac");
    let stdout_plan = denoize(&[
        "plan",
        input.to_str().unwrap(),
        "-",
        "--stream",
        "--stream-frames",
        "257",
        "--output-format",
        "flac",
        "--no-metadata",
        "--max-temp-space",
        "64",
    ]);
    assert_success(&stdout_plan);
    std::fs::write(&stdout_plan_path, &stdout_plan.stdout).unwrap();
    let stdout_run = denoize(&[
        input.to_str().unwrap(),
        "-",
        "--stream",
        "--stream-frames",
        "257",
        "--output-format",
        "flac",
        "--no-metadata",
        "--max-temp-space",
        "64",
        "--receipt",
        stdout_receipt.to_str().unwrap(),
        "--receipt-key",
        secret.to_str().unwrap(),
    ]);
    assert_success(&stdout_run);
    assert!(stdout_run.stdout.starts_with(b"fLaC"));
    std::fs::write(&captured, &stdout_run.stdout).unwrap();
    let verified = denoize(&[
        "receipts",
        "verify",
        stdout_receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--plan",
        stdout_plan_path.to_str().unwrap(),
        "--output",
        captured.to_str().unwrap(),
        "--json",
    ]);
    assert_success(&verified);
    let report: Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(report["schema"], "denoize-receipt-verification-v2");
    assert_eq!(report["verified_items"][0]["output_path"], "-");

    std::fs::write(&captured, b"tampered").unwrap();
    let tampered = denoize(&[
        "receipts",
        "verify",
        stdout_receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--output",
        captured.to_str().unwrap(),
    ]);
    assert!(!tampered.status.success());
    assert!(String::from_utf8_lossy(&tampered.stderr).contains("fingerprint mismatch"));
}

#[test]
fn invalid_receipt_preflight_never_creates_audio_output() {
    let root = TestDirectory::create();
    let input = root.join("input.wav");
    let output = root.join("output.wav");
    let receipt = root.join("receipt.json");
    let missing_key = root.join("missing-secret.json");
    write_test_wav(&input);

    let result = denoize(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--no-metadata",
        "--receipt",
        receipt.to_str().unwrap(),
        "--receipt-key",
        missing_key.to_str().unwrap(),
    ]);

    assert!(!result.status.success());
    assert!(!output.exists());
    assert!(!receipt.exists());
}

#[test]
fn unknown_and_future_execution_documents_fail_closed() {
    let root = TestDirectory::create();
    let input = root.join("input.wav");
    let output = root.join("output.wav");
    let receipt = root.join("receipt.json");
    let secret = root.join("secret.json");
    let public = root.join("public.json");
    let future_plan = root.join("future-plan.json");
    let unknown_receipt = root.join("unknown-receipt.json");
    let tampered_receipt = root.join("tampered-receipt.json");
    write_test_wav(&input);
    assert_success(&denoize(&[
        "receipts",
        "keygen",
        secret.to_str().unwrap(),
        public.to_str().unwrap(),
    ]));
    let planned = denoize(&[
        "plan",
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--no-metadata",
    ]);
    assert_success(&planned);
    let mut plan_json: Value = serde_json::from_slice(&planned.stdout).unwrap();
    plan_json["schema_version"] = Value::from(2);
    std::fs::write(&future_plan, serde_json::to_vec(&plan_json).unwrap()).unwrap();
    assert_success(&denoize(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--no-metadata",
        "--receipt",
        receipt.to_str().unwrap(),
        "--receipt-key",
        secret.to_str().unwrap(),
    ]));

    let mut receipt_json: Value =
        serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    let mut tampered_json = receipt_json.clone();
    tampered_json["payload"]["items"][0]["output"]["fingerprint"]["digest"] =
        Value::from("00".repeat(32));
    std::fs::write(
        &tampered_receipt,
        serde_json::to_vec(&tampered_json).unwrap(),
    )
    .unwrap();
    receipt_json["future_field"] = Value::from(true);
    std::fs::write(&unknown_receipt, serde_json::to_vec(&receipt_json).unwrap()).unwrap();
    let unknown = denoize(&[
        "receipts",
        "verify",
        unknown_receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--output-root",
        root.path.to_str().unwrap(),
    ]);
    assert!(!unknown.status.success());
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("unknown field"));

    let future = denoize(&[
        "receipts",
        "verify",
        receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--plan",
        future_plan.to_str().unwrap(),
        "--output-root",
        root.path.to_str().unwrap(),
    ]);
    assert!(!future.status.success());
    assert!(
        String::from_utf8_lossy(&future.stderr)
            .contains("unsupported execution plan schema version 2"),
        "{}",
        String::from_utf8_lossy(&future.stderr)
    );

    let unauthenticated = denoize(&[
        "receipts",
        "verify",
        tampered_receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--plan",
        future_plan.to_str().unwrap(),
        "--output-root",
        root.path.to_str().unwrap(),
    ]);
    assert!(!unauthenticated.status.success());
    assert!(
        String::from_utf8_lossy(&unauthenticated.stderr).contains("signature verification failed"),
        "{}",
        String::from_utf8_lossy(&unauthenticated.stderr)
    );
    assert!(output.is_file());
}

#[cfg(unix)]
#[test]
fn fifo_receipt_key_is_rejected_promptly_without_output() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt as _;

    let root = TestDirectory::create();
    let input = root.join("input.wav");
    let output = root.join("output.wav");
    let receipt = root.join("receipt.json");
    let fifo = root.join("secret.fifo");
    write_test_wav(&input);
    let fifo_path = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo_path.as_ptr(), 0o600) }, 0);

    let result = denoize_with_timeout(
        &[
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--no-metadata",
            "--receipt",
            receipt.to_str().unwrap(),
            "--receipt-key",
            fifo.to_str().unwrap(),
        ],
        Duration::from_secs(3),
    );

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("regular file"));
    assert!(!output.exists());
    assert!(!receipt.exists());
}

#[cfg(unix)]
#[test]
fn shared_secret_key_permissions_are_rejected_before_output() {
    use std::os::unix::fs::PermissionsExt as _;

    let root = TestDirectory::create();
    let input = root.join("input.wav");
    let output = root.join("output.wav");
    let receipt = root.join("receipt.json");
    let secret = root.join("secret.json");
    let public = root.join("public.json");
    write_test_wav(&input);
    assert_success(&denoize(&[
        "receipts",
        "keygen",
        secret.to_str().unwrap(),
        public.to_str().unwrap(),
    ]));
    std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o644)).unwrap();

    let result = denoize(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--no-metadata",
        "--receipt",
        receipt.to_str().unwrap(),
        "--receipt-key",
        secret.to_str().unwrap(),
    ]);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("permissions"));
    assert!(!output.exists());
    assert!(!receipt.exists());
}

#[cfg(unix)]
#[test]
fn receipt_verifier_rejects_symlink_escape_from_output_root() {
    use std::os::unix::fs::symlink;

    let root = TestDirectory::create();
    let outside = TestDirectory::create();
    let input = root.join("input.wav");
    let output = root.join("output.wav");
    let moved = outside.join("moved-output.wav");
    let receipt = root.join("receipt.json");
    let secret = root.join("secret.json");
    let public = root.join("public.json");
    write_test_wav(&input);
    assert_success(&denoize(&[
        "receipts",
        "keygen",
        secret.to_str().unwrap(),
        public.to_str().unwrap(),
    ]));
    assert_success(&denoize(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--no-metadata",
        "--receipt",
        receipt.to_str().unwrap(),
        "--receipt-key",
        secret.to_str().unwrap(),
    ]));
    std::fs::rename(&output, &moved).unwrap();
    symlink(&moved, &output).unwrap();

    let result = denoize(&[
        "receipts",
        "verify",
        receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--output-root",
        root.path.to_str().unwrap(),
    ]);

    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("escapes its verification root"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(moved.is_file());
}

#[test]
fn batch_plan_and_receipts_cover_processed_and_exactly_skipped_outputs() {
    let root = TestDirectory::create();
    let input = root.join("inputs");
    let output = root.join("outputs");
    let secret = root.join("secret.json");
    let public = root.join("public.json");
    std::fs::create_dir(&input).expect("create batch input");
    write_test_wav(&input.join("a.wav"));
    write_test_wav(&input.join("b.wav"));
    assert_success(&denoize(&[
        "receipts",
        "keygen",
        secret.to_str().unwrap(),
        public.to_str().unwrap(),
    ]));

    let first_plan_path = root.join("first-plan.json");
    let first_receipt = root.join("first-receipt.json");
    let first_plan = denoize(&[
        "plan",
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--batch",
        "--resume",
        "--jobs",
        "1",
        "--no-metadata",
    ]);
    assert_success(&first_plan);
    let first_json: Value = serde_json::from_slice(&first_plan.stdout).expect("batch plan JSON");
    assert_eq!(first_json["kind"], "batch");
    assert_eq!(first_json["items"].as_array().unwrap().len(), 2);
    assert!(first_json["items"]
        .as_array()
        .unwrap()
        .iter()
        .all(|item| item["output"]["action"] == "process"));
    assert!(first_json["items"]
        .as_array()
        .unwrap()
        .iter()
        .all(|item| item["output"]["existing_fingerprint"].is_null()));
    assert!(!output.exists(), "batch plan created output directory");
    std::fs::write(&first_plan_path, &first_plan.stdout).expect("write first plan");

    let first_run = denoize(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--batch",
        "--resume",
        "--jobs",
        "1",
        "--no-progress",
        "--no-metadata",
        "--receipt",
        first_receipt.to_str().unwrap(),
        "--receipt-key",
        secret.to_str().unwrap(),
    ]);
    assert_success(&first_run);
    let first_verify = denoize(&[
        "receipts",
        "verify",
        first_receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--plan",
        first_plan_path.to_str().unwrap(),
        "--output-root",
        output.to_str().unwrap(),
        "--json",
    ]);
    assert_success(&first_verify);

    let skip_plan_path = root.join("skip-plan.json");
    let skip_receipt = root.join("skip-receipt.json");
    let skip_plan = denoize(&[
        "plan",
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--batch",
        "--resume",
        "--jobs",
        "1",
        "--no-metadata",
    ]);
    assert_success(&skip_plan);
    let skip_json: Value = serde_json::from_slice(&skip_plan.stdout).expect("skip plan JSON");
    assert!(skip_json["items"]
        .as_array()
        .unwrap()
        .iter()
        .all(|item| item["output"]["action"] == "skip"));
    assert!(skip_json["items"].as_array().unwrap().iter().all(|item| {
        item["output"]["existing_fingerprint"]["len"]
            .as_u64()
            .is_some_and(|len| len > 0)
            && item["output"]["existing_fingerprint"]["digest"]
                .as_str()
                .is_some_and(|digest| digest.len() == 64)
    }));
    std::fs::write(&skip_plan_path, &skip_plan.stdout).expect("write skip plan");

    let skip_run = denoize(&[
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--batch",
        "--resume",
        "--jobs",
        "1",
        "--no-progress",
        "--no-metadata",
        "--receipt",
        skip_receipt.to_str().unwrap(),
        "--receipt-key",
        secret.to_str().unwrap(),
    ]);
    assert_success(&skip_run);
    let skip_verify = denoize(&[
        "receipts",
        "verify",
        skip_receipt.to_str().unwrap(),
        "--key",
        public.to_str().unwrap(),
        "--plan",
        skip_plan_path.to_str().unwrap(),
        "--output-root",
        output.to_str().unwrap(),
        "--json",
    ]);
    assert_success(&skip_verify);
    let verified: Value =
        serde_json::from_slice(&skip_verify.stdout).expect("skip verification JSON");
    assert!(verified["verified_items"]
        .as_array()
        .unwrap()
        .iter()
        .all(|item| item["outcome"] == "skipped"));
}

#[cfg(unix)]
#[test]
fn generated_secret_key_is_owner_only() {
    use std::os::unix::fs::PermissionsExt as _;

    let root = TestDirectory::create();
    let secret = root.join("secret.json");
    let public = root.join("public.json");
    let generated = denoize(&[
        "receipts",
        "keygen",
        secret.to_str().unwrap(),
        public.to_str().unwrap(),
    ]);
    assert_success(&generated);
    assert_eq!(
        std::fs::metadata(secret).unwrap().permissions().mode() & 0o777,
        0o600
    );
}
