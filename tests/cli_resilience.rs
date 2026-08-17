use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FAULT_ENV: &str = "DENOIZE_INTERNAL_FAULT_V1";
const FAULT_EXIT_CODE: i32 = 86;

struct TestDirectory(tempfile::TempDir);

impl TestDirectory {
    fn create() -> Self {
        Self(tempfile::tempdir().expect("create resilience test directory"))
    }

    fn path(&self) -> &Path {
        self.0.path()
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path().join(name)
    }
}

fn write_wav_frames(path: &Path, frames: usize) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create resilience WAV");
    for frame in 0..frames {
        let sample = if frame % 80 < 40 {
            2_000_i16
        } else {
            -2_000_i16
        };
        writer
            .write_sample(sample)
            .expect("write resilience sample");
    }
    writer.finalize().expect("finalize resilience WAV");
}

fn write_wav(path: &Path) {
    write_wav_frames(path, 3_200);
}

fn command(input: &Path, output: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_denoize"));
    command.arg(input).arg(output);
    command
}

fn fault(command: &mut Command, point: &str, action: &str) {
    command.env(FAULT_ENV, format!("v1|{point}|1|{action}"));
}

fn assert_injected_exit(output: &Output, point: &str) {
    assert_eq!(
        output.status.code(),
        Some(FAULT_EXIT_CODE),
        "unexpected status for {point}: {:?}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(point),
        "fault point was not reported: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|error| panic!("fault left partial automation JSON {line:?}: {error}"));
    }
}

fn assert_valid_wav_frames(path: &Path, expected_frames: usize) {
    let decoded = denoize::read_audio(path).expect("decode recovered WAV");
    assert_eq!(decoded.sample_rate, 16_000);
    assert_eq!(decoded.channels(), 1);
    assert_eq!(decoded.frames(), expected_frames);
}

fn assert_valid_wav(path: &Path) {
    assert_valid_wav_frames(path, 3_200);
}

fn stream_command(input: &Path, output: &Path) -> Command {
    let mut command = command(input, output);
    command.args([
        "--stream",
        "--resume",
        "--stream-frames",
        "73",
        "--no-metadata",
        "--no-progress",
    ]);
    command
}

fn assert_receipt_verifies(receipt: &Path, public_key: &Path) {
    let verified = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(["receipts", "verify"])
        .arg(receipt)
        .args(["--key"])
        .arg(public_key)
        .output()
        .expect("verify recovered receipt");
    assert!(
        verified.status.success(),
        "receipt verification failed: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
}

#[test]
fn injected_io_error_unwinds_without_publishing_or_leaking_a_stage() {
    for point in [
        "atomic-output.before-stage-sync",
        "atomic-output.after-stage-sync",
        "atomic-output.before-publish",
        "atomic-output.after-publish",
    ] {
        let directory = TestDirectory::create();
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_wav(&input);

        let mut child = command(&input, &output);
        child.args(["--no-metadata", "--no-progress"]);
        fault(&mut child, point, "error");
        let result = child.output().expect("run faulted file command");

        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        assert!(String::from_utf8_lossy(&result.stderr).contains("injected fault"));
        if point == "atomic-output.after-publish" {
            assert_valid_wav(&output);
        } else {
            assert!(!output.exists());
        }
        let stages: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(".denoize-"))
            .collect();
        assert!(
            stages.is_empty(),
            "recoverable error at {point} leaked stages: {stages:?}"
        );
    }
}

#[test]
fn abrupt_file_crashes_leave_only_an_absent_or_complete_output() {
    for point in [
        "atomic-output.before-stage-sync",
        "atomic-output.after-stage-sync",
        "atomic-output.before-publish",
        "atomic-output.after-publish",
    ] {
        let directory = TestDirectory::create();
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_wav(&input);

        let mut child = command(&input, &output);
        child.args(["--no-metadata", "--no-progress"]);
        fault(&mut child, point, "exit");
        let crashed = child
            .output()
            .expect("run abruptly terminated file command");
        assert_injected_exit(&crashed, point);

        let stages: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(".denoize-"))
            .collect();
        if point == "atomic-output.after-publish" {
            assert_valid_wav(&output);
            assert!(stages.is_empty(), "published output retained a stage");
        } else {
            assert!(!output.exists(), "{point} exposed a partial output");
            assert_eq!(
                stages.len(),
                1,
                "{point} did not retain exactly one private crash stage"
            );

            let resumed = command(&input, &output)
                .args(["--no-metadata", "--no-progress"])
                .output()
                .expect("restart abruptly terminated file command");
            assert!(
                resumed.status.success(),
                "file command did not recover from {point}: {}",
                String::from_utf8_lossy(&resumed.stderr)
            );
            assert_valid_wav(&output);
        }
    }
}

#[test]
fn batch_crash_matrix_recovers_every_journal_publication_prefix() {
    for point in [
        "batch-journal.after-prepare-sync",
        "batch-journal.after-output-publish",
        "batch-journal.after-complete-sync",
    ] {
        let directory = TestDirectory::create();
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).expect("create batch input");
        write_wav(&input.join("sample.wav"));

        let mut child = command(&input, &output);
        child.args([
            "--batch",
            "--resume",
            "--json",
            "--jobs",
            "1",
            "--no-metadata",
        ]);
        fault(&mut child, point, "exit");
        let crashed = child.output().expect("run faulted batch command");
        assert_injected_exit(&crashed, point);

        let resumed = command(&input, &output)
            .args([
                "--batch",
                "--resume",
                "--json",
                "--jobs",
                "1",
                "--no-metadata",
            ])
            .output()
            .expect("resume faulted batch command");
        assert!(
            resumed.status.success(),
            "batch did not recover from {point}: {}",
            String::from_utf8_lossy(&resumed.stderr)
        );
        for line in String::from_utf8_lossy(&resumed.stdout).lines() {
            serde_json::from_str::<serde_json::Value>(line)
                .expect("recovered batch emits complete JSON records");
        }
        assert_valid_wav(&output.join("sample.wav"));
        let state = std::fs::read_to_string(output.join(denoize::batch_resume::STATE_FILE_NAME))
            .expect("read recovered batch journal");
        assert_eq!(state.lines().count(), 2, "unexpected journal after {point}");
    }
}

#[test]
fn stream_crash_matrix_reconciles_prepared_and_committed_outputs() {
    for point in [
        "stream-checkpoint.after-prepare-publish-sync",
        "stream-checkpoint.after-output-publish",
        "stream-checkpoint.before-cleanup",
    ] {
        let directory = TestDirectory::create();
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_wav(&input);

        let mut child = stream_command(&input, &output);
        fault(&mut child, point, "exit");
        let crashed = child.output().expect("run faulted stream command");
        assert_injected_exit(&crashed, point);

        let (state, spool, lock) = denoize::batch_resume::stream_checkpoint_sidecar_paths(&output)
            .expect("resolve stream checkpoint paths");
        assert!(state.exists(), "{point} did not leave durable state");
        assert!(spool.exists(), "{point} did not leave a durable spool");

        let resumed = stream_command(&input, &output)
            .output()
            .expect("resume faulted stream command");
        assert!(
            resumed.status.success(),
            "stream did not recover from {point}: {}",
            String::from_utf8_lossy(&resumed.stderr)
        );
        assert!(resumed.stdout.is_empty());
        assert_valid_wav(&output);
        assert!(!state.exists(), "{point} recovery retained state");
        assert!(!spool.exists(), "{point} recovery retained spool");
        assert!(
            lock.exists(),
            "{point} recovery did not leave a reusable lock"
        );
    }
}

#[test]
fn periodic_stream_checkpoint_crash_resumes_from_the_durable_prefix() {
    const FRAMES: usize = 1_048_600;

    let directory = TestDirectory::create();
    let input = directory.join("input.wav");
    let output = directory.join("output.wav");
    write_wav_frames(&input, FRAMES);

    let mut child = command(&input, &output);
    child.args([
        "--stream",
        "--resume",
        "--stream-frames",
        "8192",
        "--no-metadata",
        "--no-progress",
    ]);
    fault(&mut child, "stream-checkpoint.after-periodic-sync", "exit");
    let crashed = child.output().expect("run periodic checkpoint crash");
    assert_injected_exit(&crashed, "stream-checkpoint.after-periodic-sync");
    assert!(!output.exists());

    let resumed = command(&input, &output)
        .args([
            "--stream",
            "--resume",
            "--stream-frames",
            "8192",
            "--no-metadata",
            "--no-progress",
        ])
        .output()
        .expect("resume periodic checkpoint crash");
    assert!(
        resumed.status.success(),
        "periodic checkpoint did not recover: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert_valid_wav_frames(&output, FRAMES);
}

#[test]
fn stream_receipt_crashes_leave_a_verifiable_or_recoverable_publication() {
    for point in [
        "stream-checkpoint.after-output-publish",
        "stream-checkpoint.after-receipt-publish",
    ] {
        let directory = TestDirectory::create();
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        let receipt = directory.join("receipt.json");
        let secret_key = directory.join("secret.json");
        let public_key = directory.join("public.json");
        write_wav(&input);

        let keygen = Command::new(env!("CARGO_BIN_EXE_denoize"))
            .args(["receipts", "keygen"])
            .arg(&secret_key)
            .arg(&public_key)
            .output()
            .expect("generate resilience receipt keypair");
        assert!(
            keygen.status.success(),
            "key generation failed: {}",
            String::from_utf8_lossy(&keygen.stderr)
        );

        let mut child = stream_command(&input, &output);
        child
            .args(["--receipt"])
            .arg(&receipt)
            .args(["--receipt-key"])
            .arg(&secret_key);
        fault(&mut child, point, "exit");
        let crashed = child.output().expect("run receipt publication crash");
        assert_injected_exit(&crashed, point);
        assert_valid_wav(&output);

        if point == "stream-checkpoint.after-output-publish" {
            assert!(
                !receipt.exists(),
                "receipt was published before its boundary"
            );
            let resumed = stream_command(&input, &output)
                .args(["--receipt"])
                .arg(&receipt)
                .args(["--receipt-key"])
                .arg(&secret_key)
                .output()
                .expect("resume before receipt publication");
            assert!(
                resumed.status.success(),
                "receipt recovery failed: {}",
                String::from_utf8_lossy(&resumed.stderr)
            );
        } else {
            assert!(receipt.is_file(), "committed receipt is missing");
            assert_receipt_verifies(&receipt, &public_key);
            let resumed = stream_command(&input, &output)
                .output()
                .expect("reconcile checkpoint after committed receipt");
            assert!(
                resumed.status.success(),
                "checkpoint cleanup failed after receipt commit: {}",
                String::from_utf8_lossy(&resumed.stderr)
            );
        }

        assert_receipt_verifies(&receipt, &public_key);
        let (state, spool, lock) = denoize::batch_resume::stream_checkpoint_sidecar_paths(&output)
            .expect("resolve receipt checkpoint paths");
        assert!(
            !state.exists(),
            "receipt recovery retained state at {point}"
        );
        assert!(
            !spool.exists(),
            "receipt recovery retained spool at {point}"
        );
        assert!(
            lock.exists(),
            "receipt recovery lost reusable lock at {point}"
        );
    }
}
