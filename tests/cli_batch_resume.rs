use serde_json::Value;
use std::path::Path;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_DIRECTORY_ID: AtomicUsize = AtomicUsize::new(0);

struct TestDirectory {
    path: std::path::PathBuf,
}

impl TestDirectory {
    fn create() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "denoize-cli-batch-resume-{}-{timestamp}-{}",
            std::process::id(),
            NEXT_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).expect("create unique test directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn write_silent_wav(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create test WAV");
    for _ in 0..1_600 {
        writer.write_sample(0_i16).expect("write test sample");
    }
    writer.finalize().expect("finalize test WAV");
}

fn write_tone_wav(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create test WAV");
    for index in 0..1_600 {
        let sample =
            ((index as f64 * 2.0 * std::f64::consts::PI * 440.0 / 16_000.0).sin() * 8_000.0) as i16;
        writer.write_sample(sample).expect("write test sample");
    }
    writer.finalize().expect("finalize test WAV");
}

fn run_batch_with_options(input: &Path, output: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(input)
        .arg(output)
        .args([
            "--batch",
            "--resume",
            "--json",
            "--jobs",
            "2",
            "--no-metadata",
        ])
        .args(extra)
        .output()
        .expect("run denoize batch command")
}

fn run_batch(input: &Path, output: &Path) -> Output {
    run_batch_with_options(input, output, &[])
}

fn summary(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "batch command failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = std::str::from_utf8(&output.stdout).expect("batch stdout is UTF-8");
    stdout
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("stdout line is valid JSON"))
        .find(|value| value["event"] == "summary")
        .expect("batch output contains a summary event")
}

fn assert_summary_counts(summary: &Value, succeeded: u64, skipped: u64, failed: u64) {
    assert_eq!(summary["event"], "summary");
    assert_eq!(summary["total"], 1);
    assert_eq!(summary["succeeded"], succeeded);
    assert_eq!(summary["skipped"], skipped);
    assert_eq!(summary["failed"], failed);
    assert_eq!(summary["cancelled"], false);
    assert_eq!(succeeded + skipped + failed, 1);
}

#[test]
fn resumed_batch_reports_skipped_file_exclusively() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_silent_wav(&input.join("sample.wav"));

    let first = run_batch(&input, &output);
    assert_summary_counts(&summary(&first), 1, 0, 0);

    let resumed = run_batch(&input, &output);
    assert_summary_counts(&summary(&resumed), 0, 1, 0);
}

#[test]
fn resume_identity_changes_with_the_planned_output_format() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_silent_wav(&input.join("sample.wav"));

    let first = run_batch_with_options(&input, &output, &["--output-format", "wav"]);
    assert_summary_counts(&summary(&first), 1, 0, 0);
    std::fs::write(output.join("sample.flac"), b"stale output").expect("write stale FLAC");

    let converted =
        run_batch_with_options(&input, &output, &["--output-format", "flac", "--force"]);

    assert_summary_counts(&summary(&converted), 1, 0, 0);
    let probe = denoize::decode::probe_file(&output.join("sample.flac"))
        .expect("probe refreshed FLAC output");
    assert_eq!(probe.format, denoize::AudioFormat::Flac);
    assert_eq!(probe.codec, denoize::AudioCodec::Flac);
}

#[test]
fn resume_identity_changes_with_the_canonical_input_root() {
    let root = TestDirectory::create();
    let input_a = root.path().join("input-a");
    let input_b = root.path().join("input-b");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input_a).expect("create first input directory");
    std::fs::create_dir_all(&input_b).expect("create second input directory");
    write_silent_wav(&input_a.join("sample.wav"));
    write_tone_wav(&input_b.join("sample.wav"));

    let first = run_batch(&input_a, &output);
    assert_summary_counts(&summary(&first), 1, 0, 0);
    let second = run_batch_with_options(&input_b, &output, &["--force"]);

    assert_summary_counts(&summary(&second), 1, 0, 0);
}
