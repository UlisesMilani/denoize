use serde_json::{json, Value};
use std::io::Read as _;
use std::path::Path;
use std::process::{Command, Stdio};

fn make_private_directory(path: &Path) {
    std::fs::create_dir(path).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
}

fn write_private(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
}

fn write_test_wav(path: &Path) {
    let sample_rate = 16_000_u32;
    let mut samples = Vec::with_capacity(sample_rate as usize / 2);
    for index in 0..(sample_rate / 4) {
        let sample = if index % 80 < 40 {
            1_000_i16
        } else {
            -1_000_i16
        };
        samples.extend(sample.to_le_bytes());
    }
    let mut wav = Vec::with_capacity(44 + samples.len());
    wav.extend(b"RIFF");
    wav.extend((36_u32 + samples.len() as u32).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16_u32.to_le_bytes());
    wav.extend(1_u16.to_le_bytes());
    wav.extend(1_u16.to_le_bytes());
    wav.extend(sample_rate.to_le_bytes());
    wav.extend((sample_rate * 2).to_le_bytes());
    wav.extend(2_u16.to_le_bytes());
    wav.extend(16_u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend((samples.len() as u32).to_le_bytes());
    wav.extend(samples);
    std::fs::write(path, wav).unwrap();
}

fn process_request(input: &Path, output: &Path) -> Value {
    json!({
        "input": input,
        "output": output,
        "expectedInputFingerprint": null,
        "expectedRecipe": null,
        "stream": false,
        "resume": false,
        "streamFrames": 8192,
        "receipt": null,
        "receiptKey": null,
        "options": {
            "backend": "classical",
            "preset": "hifi",
            "mode": "music",
            "strength": 0.4,
            "adaptiveNoise": false,
            "vad": false,
            "channelMode": "linked",
            "downmix": "preserve",
            "loudnessLufs": null,
            "truePeakDbtp": -1.0,
            "preserveMetadata": false,
            "force": false,
            "mp3BitrateKbps": 192,
            "aacBitrateKbps": 192,
            "aacEncoder": "oxide",
            "onnxModel": null,
            "onnxSampleRate": 16000,
            "sgmseProfile": "balanced",
            "accelerator": "cpu",
            "deterministic": false,
            "seed": null,
            "maxProcessMemoryMb": null,
            "maxTemporaryMb": null,
            "maxGpuMemoryMb": null,
            "maxGpuJobs": 1
        }
    })
}

fn execute_operation(
    directory: &Path,
    operation: Value,
    cancelled: bool,
) -> (std::process::ExitStatus, Vec<Value>, Value) {
    let recovery_root = directory.join("recovery");
    make_private_directory(&recovery_root);
    let recovery_id = "a".repeat(64);
    let recovery_path = recovery_root.join(format!("{recovery_id}.json"));
    let recovery = json!({
        "schema": "denoize-desktop-recovery-v1",
        "schema_version": 1,
        "recovery_id": recovery_id,
        "process_id": std::process::id(),
        "started_unix_seconds": 1,
        "state": "active",
        "operation": operation.clone(),
        "stages": []
    });
    write_private(&recovery_path, &serde_json::to_vec(&recovery).unwrap());

    let worker_root = directory.join("worker");
    make_private_directory(&worker_root);
    let request_path = worker_root.join("request.json");
    let cancel_marker = worker_root.join("cancel");
    let commit_fence = worker_root.join("commit.lock");
    let start_gate = worker_root.join("start.gate");
    write_private(&commit_fence, b"");
    write_private(&start_gate, b"");
    if cancelled {
        write_private(&cancel_marker, b"");
    }
    let request = json!({
        "schema": "denoize-desktop-job-worker-v1",
        "schema_version": 1,
        "nonce": "b".repeat(64),
        "parent_process_id": std::process::id(),
        "job_id": 7,
        "cancel_marker": cancel_marker,
        "commit_fence": commit_fence,
        "start_gate": start_gate,
        "recovery": {
            "path": recovery_path.clone(),
            "recovery_id": "a".repeat(64),
            "parent_process_id": std::process::id()
        },
        "operation": operation
    });
    write_private(&request_path, &serde_json::to_vec(&request).unwrap());

    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize-desktop"))
        .arg("--denoize-desktop-job-worker")
        .arg(&request_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::fs::remove_file(&start_gate).unwrap();
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    let status = child.wait().unwrap();
    let events = stdout
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let persisted: Value = serde_json::from_slice(&std::fs::read(&recovery_path).unwrap()).unwrap();
    (status, events, persisted)
}

fn execute_worker(
    cancelled: bool,
    create_input: bool,
) -> (std::process::ExitStatus, Vec<Value>, Value, bool) {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    let output = directory.path().join("output.wav");
    if create_input {
        write_test_wav(&input);
    }
    let process = process_request(&input, &output);
    let operation = json!({ "kind": "file", "request": process });
    let (status, events, persisted) = execute_operation(directory.path(), operation, cancelled);
    (status, events, persisted, output.exists())
}

#[test]
fn desktop_binary_isolates_a_final_file_and_streams_authenticated_progress() {
    let (status, events, recovery, output_exists) = execute_worker(false, true);
    assert!(status.success(), "worker status: {status}");
    assert!(output_exists);
    assert!(events.len() >= 3, "events: {events:#?}");
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["schema"], "denoize-desktop-job-worker-v1");
        assert_eq!(event["schema_version"], 1);
        assert_eq!(event["nonce"], "b".repeat(64));
        assert_eq!(event["sequence"], (index + 1) as u64);
        assert_eq!(event["progress"]["jobId"], 7);
        assert_eq!(event["progress"]["kind"], "file");
    }
    assert_eq!(events.last().unwrap()["progress"]["status"], "completed");
    assert_eq!(recovery["stages"], json!([]));
}

#[test]
fn preexisting_cancel_marker_prevents_final_output_publication() {
    let (status, events, recovery, output_exists) = execute_worker(true, true);
    assert!(status.success(), "worker status: {status}");
    assert!(!output_exists);
    assert_eq!(events.last().unwrap()["progress"]["status"], "cancelled");
    assert_eq!(recovery["stages"], json!([]));
}

#[test]
fn worker_failures_emit_a_localizable_structured_error() {
    let (status, events, recovery, output_exists) = execute_worker(false, false);
    assert!(status.success(), "worker status: {status}");
    assert!(!output_exists);
    let progress = &events.last().unwrap()["progress"];
    assert_eq!(progress["status"], "failed");
    assert_eq!(progress["error"]["code"], "input.not-found", "{progress:#}");
    assert_eq!(progress["error"]["parameters"], json!({}));
    assert!(!progress["error"]["technicalDetail"]
        .as_str()
        .unwrap()
        .is_empty());
    assert_eq!(recovery["stages"], json!([]));
}

#[test]
fn desktop_binary_isolates_a_batch_and_publishes_each_item() {
    let directory = tempfile::tempdir().unwrap();
    let input_dir = directory.path().join("input");
    let output_dir = directory.path().join("output");
    std::fs::create_dir(&input_dir).unwrap();
    std::fs::create_dir(&output_dir).unwrap();
    let input = input_dir.join("sample.wav");
    write_test_wav(&input);
    let options = process_request(&input, &output_dir.join("unused.wav"))["options"].clone();
    let operation = json!({
        "kind": "batch",
        "request": {
            "inputs": [],
            "inputDir": input_dir,
            "outputDir": output_dir,
            "outputFormat": "wav",
            "recursive": false,
            "jobs": 1,
            "resume": false,
            "receipt": null,
            "receiptKey": null,
            "options": options
        }
    });

    let (status, events, recovery) = execute_operation(directory.path(), operation, false);

    assert!(status.success(), "worker status: {status}");
    assert!(output_dir.join("sample.wav").is_file());
    assert_eq!(events.last().unwrap()["progress"]["status"], "completed");
    assert!(events
        .iter()
        .all(|event| event["progress"]["kind"] == "batch"));
    assert!(events
        .iter()
        .any(|event| event["progress"]["itemStatus"] == "completed"));
    assert_eq!(recovery["stages"], json!([]));
}
