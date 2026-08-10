use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

fn temp_root(label: &str) -> PathBuf {
    let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "denoize-cli-validation-{label}-{}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn run(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn run_with_stdin(args: &[String], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn write_wav(path: &Path, channels: u16, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for _ in 0..64 {
        for _ in 0..spec.channels {
            writer.write_sample(0_i16).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn assert_no_staged_output(root: &Path, output: &Path) {
    assert!(!output.exists(), "invalid config created output");
    let staged: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".part"))
        .collect();
    assert!(
        staged.is_empty(),
        "invalid config left staged files: {staged:?}"
    );
}

#[test]
fn invalid_config_wins_over_missing_input_without_output_side_effects() {
    let root = temp_root("missing-input");
    let input = root.join("missing.wav");
    let output = root.join("output.wav");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--stream".into(),
        "--profile".into(),
        "inf".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("denoize: error:"));
    assert!(stderr.contains("profile"), "unexpected error: {stderr}");
    assert!(!stderr.contains("open:"), "input I/O ran first: {stderr}");
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn overflowing_m4a_bitrate_wins_over_missing_input_without_output_side_effects() {
    let root = temp_root("m4a-bitrate-overflow");
    let input = root.join("missing.wav");
    let output = root.join("output.m4a");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--m4a-bitrate".into(),
        u32::MAX.to_string(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("--m4a-bitrate"),
        "unexpected error: {stderr}"
    );
    assert!(stderr.contains("kbps to bps"), "unexpected error: {stderr}");
    assert!(!stderr.contains("open:"), "input I/O ran first: {stderr}");
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "m4a-encode")]
#[test]
fn zero_m4a_bitrate_wins_over_missing_input_without_output_side_effects() {
    let root = temp_root("m4a-bitrate-zero");
    let input = root.join("missing.wav");
    let output = root.join("output.m4a");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--m4a-bitrate".into(),
        "0".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("M4A encode: bitrate must be greater than zero"),
        "unexpected error: {stderr}"
    );
    assert!(!stderr.contains("open:"), "input I/O ran first: {stderr}");
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "m4a-encode")]
#[test]
fn batch_validates_each_planned_aac_format_before_creating_output_directory() {
    let root = temp_root("batch-aac-bitrate-zero");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir_all(&input).unwrap();
    // A complete decode is intentionally unnecessary: batch planning only
    // probes the ADTS signature before the pure encoder-options preflight.
    std::fs::write(
        input.join("sample.aac"),
        b"\xff\xf1\x50\x80\x00\x1f\xfc\x00\x00\x00\x00\x00",
    )
    .unwrap();
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--batch".into(),
        "--m4a-bitrate".into(),
        "0".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("AAC encode: bitrate must be greater than zero"),
        "unexpected error: {stderr}"
    );
    assert!(!output.exists(), "batch preflight created output directory");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn batch_validates_every_decoded_codec_config_before_any_output() {
    let root = temp_root("batch-codec-preflight");
    let input = root.join("input");
    let output = root.join("output");
    std::fs::create_dir_all(&input).unwrap();
    write_wav(&input.join("valid.wav"), 1, 44_100);
    write_wav(&input.join("unsupported.wav"), 1, 12_345);

    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--batch".into(),
        "--output-format".into(),
        "mp3".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("MP3 encode: unsupported sample rate 12345 Hz"),
        "unexpected error: {stderr}"
    );
    assert!(!output.exists(), "batch preflight created output directory");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn piped_stdin_stops_at_the_checked_memory_limit() {
    let root = temp_root("stdin-memory");
    let output = root.join("output.wav");
    // One MiB of estimated working memory permits 128 KiB of encoded input.
    // The extra byte exercises the bounded-plus-one overflow probe.
    let input = vec![0_u8; 128 * 1024 + 1];
    let result = run_with_stdin(
        &[
            "-".into(),
            output.to_string_lossy().into_owned(),
            "--max-memory".into(),
            "1".into(),
            "--no-metadata".into(),
            "--json".into(),
        ],
        &input,
    );

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("stdin input preflight") && stderr.contains("--max-memory"),
        "unexpected error: {stderr}"
    );
    assert!(!stderr.contains("open:"), "WAV parsing ran first: {stderr}");
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn hostile_stream_plan_is_rejected_before_output_staging() {
    let root = temp_root("stream-plan");
    let input = root.join("input.wav");
    let output = root.join("output.wav");
    write_wav(&input, 8, 48_000);
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--stream".into(),
        "--profile".into(),
        "60000".into(),
        "--stream-frames".into(),
        "1048576".into(),
        "--frame".into(),
        "65536".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("denoize: error:"),
        "unexpected error: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn json_mode_preserves_the_existing_stderr_error_contract() {
    let root = temp_root("json-error");
    let output = root.join("output.wav");
    let result = run(&[
        root.join("missing.wav").to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--stream-frames".into(),
        "1048577".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.starts_with("denoize: error:"),
        "unexpected stderr: {stderr}"
    );
    assert!(stderr.contains("--stream-frames"));
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "onnx")]
#[test]
fn invalid_codec_config_precedes_backend_processing_and_output_staging() {
    let root = temp_root("codec-preflight");
    let input = root.join("input.wav");
    let output = root.join("output.mp3");
    let model = root.join("dummy.onnx");
    write_wav(&input, 1, 12_345);
    std::fs::write(&model, b"not an ONNX model").unwrap();
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--backend".into(),
        "onnx".into(),
        "--onnx-model".into(),
        model.to_string_lossy().into_owned(),
        "--onnx-rate".into(),
        "16000".into(),
        "--no-metadata".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("MP3 encode: unsupported sample rate 12345 Hz"),
        "unexpected error precedence: {stderr}"
    );
    assert!(
        !stderr.contains("failed to load ONNX model"),
        "backend model parsing ran before codec validation: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "onnx")]
#[test]
fn missing_backend_resource_precedes_missing_input() {
    let root = temp_root("backend-resource-preflight");
    let input = root.join("missing-input.wav");
    let output = root.join("output.wav");
    let model = root.join("missing-model.onnx");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--backend".into(),
        "onnx".into(),
        "--onnx-model".into(),
        model.to_string_lossy().into_owned(),
        "--onnx-rate".into(),
        "16000".into(),
        "--no-metadata".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("selected backend model does not exist or is not a file"),
        "unexpected error precedence: {stderr}"
    );
    assert!(
        !stderr.contains("read input metadata") && !stderr.contains("open:"),
        "input I/O ran before backend resource validation: {stderr}"
    );
    assert_no_staged_output(&root, &output);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "onnx")]
#[test]
fn batch_backend_resource_precedes_input_directory_scan() {
    let root = temp_root("batch-backend-resource-preflight");
    let input = root.join("missing-input-directory");
    let output = root.join("output");
    let model = root.join("missing-model.onnx");
    let result = run(&[
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "--batch".into(),
        "--backend".into(),
        "onnx".into(),
        "--onnx-model".into(),
        model.to_string_lossy().into_owned(),
        "--onnx-rate".into(),
        "16000".into(),
        "--json".into(),
    ]);

    assert!(!result.status.success());
    assert!(result.stdout.is_empty(), "error emitted partial JSON");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("selected backend model does not exist or is not a file"),
        "unexpected error precedence: {stderr}"
    );
    assert!(
        !stderr.contains("batch input is not a directory"),
        "batch input I/O ran before backend resource validation: {stderr}"
    );
    assert!(!output.exists(), "batch preflight created output directory");
    std::fs::remove_dir_all(root).unwrap();
}
