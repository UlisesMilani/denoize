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

#[cfg(unix)]
fn write_long_silent_wav(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 8_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create long test WAV");
    for _ in 0..2_400_000 {
        writer.write_sample(0_i16).expect("write long test sample");
    }
    writer.finalize().expect("finalize long test WAV");
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
    assert_eq!(summary["schema"], "denoize-cli-output-v1");
    assert_eq!(summary["schema_version"], 1);
    assert_eq!(summary["event"], "summary");
    assert_eq!(summary["recipe"]["domain"], "denoize-batch-recipe-v3");
    assert_eq!(summary["recipe"]["version"], 3);
    assert!(summary["recipe"]["digest"].is_null());
    assert_eq!(summary["total"], 1);
    assert_eq!(summary["succeeded"], succeeded);
    assert_eq!(summary["skipped"], skipped);
    assert_eq!(summary["failed"], failed);
    assert_eq!(summary["cancelled_count"], 0);
    assert_eq!(summary["cancelled"], false);
    assert_eq!(
        succeeded + skipped + failed + summary["cancelled_count"].as_u64().unwrap(),
        1
    );
}

fn assert_preflight_failure(output: &Output) -> String {
    assert!(!output.status.success(), "batch unexpectedly succeeded");
    assert!(
        output.stdout.is_empty(),
        "failed batch emitted stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn resumed_batch_reports_skipped_file_exclusively() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_silent_wav(&input.join("sample.wav"));

    let first = run_batch(&input, &output);
    let progress = std::str::from_utf8(&first.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .find(|value| value["event"] == "progress")
        .unwrap();
    assert_eq!(progress["schema"], "denoize-cli-output-v1");
    assert_eq!(progress["schema_version"], 1);
    assert_eq!(progress["recipe"]["domain"], "denoize-batch-recipe-v3");
    assert_eq!(progress["recipe"]["digest"].as_str().unwrap().len(), 64);
    assert_summary_counts(&summary(&first), 1, 0, 0);

    let resumed = run_batch(&input, &output);
    assert_summary_counts(&summary(&resumed), 0, 1, 0);
}

#[test]
fn exact_resume_match_skips_even_with_force() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));

    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);
    let destination = output.join("sample.wav");
    let before = std::fs::read(&destination).expect("read completed output");
    let before_modified = std::fs::metadata(&destination)
        .expect("inspect completed output")
        .modified()
        .expect("read output modification time");

    let forced = run_batch_with_options(&input, &output, &["--force"]);

    assert_summary_counts(&summary(&forced), 0, 1, 0);
    assert_eq!(std::fs::read(&destination).unwrap(), before);
    assert_eq!(
        std::fs::metadata(destination).unwrap().modified().unwrap(),
        before_modified
    );
}

#[cfg(unix)]
#[test]
fn cancellation_during_staging_reports_a_partition_and_publishes_nothing() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_long_silent_wav(&input.join("sample.wav"));

    let mut child = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(&input)
        .arg(&output)
        .args([
            "--batch",
            "--resume",
            "--json",
            "--jobs",
            "1",
            "--no-metadata",
            "--output-format",
            "mp3",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn cancellable batch");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let stage_exists = std::fs::read_dir(&output).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|entry| {
                entry.file_name().to_string_lossy().starts_with(".denoize-")
                    && entry
                        .path()
                        .extension()
                        .is_some_and(|value| value == "part")
            })
        });
        if stage_exists {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "batch did not begin staging before the test deadline"
        );
        assert!(
            child.try_wait().expect("poll cancellable batch").is_none(),
            "batch exited before cancellation could be delivered"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    // SAFETY: `child.id()` is the live child process polled above; SIGINT is
    // the public cancellation mechanism under test.
    let signal_result = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) };
    assert_eq!(signal_result, 0, "deliver SIGINT to batch child");
    let cancelled = child.wait_with_output().expect("wait for cancelled batch");

    assert!(!cancelled.status.success());
    let values = std::str::from_utf8(&cancelled.stdout)
        .expect("cancelled batch stdout is UTF-8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("stdout line is valid JSON"))
        .collect::<Vec<_>>();
    let progress = values
        .iter()
        .filter(|value| value["event"] == "progress")
        .collect::<Vec<_>>();
    assert_eq!(progress.len(), 1);
    assert_eq!(progress[0]["status"], "cancelled");
    let summary = values
        .iter()
        .find(|value| value["event"] == "summary")
        .expect("cancelled batch emitted summary");
    assert_eq!(summary["total"], 1);
    assert_eq!(summary["succeeded"], 0);
    assert_eq!(summary["skipped"], 0);
    assert_eq!(summary["failed"], 0);
    assert_eq!(summary["cancelled_count"], 1);
    assert_eq!(summary["cancelled"], true);
    assert!(!output.join("sample.mp3").exists());
    assert!(!output.join(".denoize-state").exists());
    assert!(
        std::fs::read_dir(&output)
            .unwrap()
            .filter_map(Result::ok)
            .all(|entry| entry.path().extension().is_none_or(|value| value != "part")),
        "cancelled staging file was not cleaned up"
    );
}

#[test]
fn input_content_change_is_detected_with_the_same_size_and_mtime() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    let source = input.join("sample.wav");
    write_silent_wav(&source);
    let original_metadata = std::fs::metadata(&source).expect("inspect original input");

    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);
    let destination = output.join("sample.wav");
    let completed = std::fs::read(&destination).expect("read completed output");
    let state = std::fs::read(output.join(".denoize-state")).expect("read v3 state");

    write_tone_wav(&source);
    assert_eq!(
        std::fs::metadata(&source).unwrap().len(),
        original_metadata.len(),
        "fixture rewrite must retain its length"
    );
    let input_file = std::fs::OpenOptions::new()
        .write(true)
        .open(&source)
        .expect("open rewritten input");
    input_file
        .set_times(
            std::fs::FileTimes::new().set_modified(
                original_metadata
                    .modified()
                    .expect("read original modification time"),
            ),
        )
        .expect("restore input modification time");

    let refused = run_batch(&input, &output);
    assert!(assert_preflight_failure(&refused).contains("--force"));
    assert_eq!(std::fs::read(&destination).unwrap(), completed);
    assert_eq!(std::fs::read(output.join(".denoize-state")).unwrap(), state);

    let replaced = run_batch_with_options(&input, &output, &["--force"]);
    assert_summary_counts(&summary(&replaced), 1, 0, 0);
    assert_ne!(std::fs::read(destination).unwrap(), completed);
}

#[test]
fn effective_strength_change_requires_regeneration() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));

    let first = run_batch_with_options(&input, &output, &["--strength", "0.2"]);
    assert_summary_counts(&summary(&first), 1, 0, 0);
    let before = std::fs::read(output.join("sample.wav")).unwrap();

    let refused = run_batch_with_options(&input, &output, &["--strength", "0.8"]);
    assert!(assert_preflight_failure(&refused).contains("--force"));
    assert_eq!(std::fs::read(output.join("sample.wav")).unwrap(), before);

    let migrated = run_batch_with_options(&input, &output, &["--strength", "0.8", "--force"]);
    assert_summary_counts(&summary(&migrated), 1, 0, 0);
    let exact = run_batch_with_options(&input, &output, &["--strength", "0.8", "--force"]);
    assert_summary_counts(&summary(&exact), 0, 1, 0);
}

#[cfg(not(any(feature = "rnnoise", feature = "deepfilter")))]
#[test]
fn backend_auto_and_explicit_share_the_same_resolved_recipe() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));

    let explicit = run_batch_with_options(&input, &output, &["--backend", "classical"]);
    assert_summary_counts(&summary(&explicit), 1, 0, 0);
    let automatic = run_batch_with_options(&input, &output, &["--backend", "auto"]);
    assert_summary_counts(&summary(&automatic), 0, 1, 0);
}

#[test]
fn metadata_policy_change_requires_regeneration() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_silent_wav(&input.join("sample.wav"));

    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);
    let preserving = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(&input)
        .arg(&output)
        .args(["--batch", "--resume", "--json", "--jobs", "2"])
        .output()
        .expect("run metadata-preserving batch");

    assert!(assert_preflight_failure(&preserving).contains("--force"));

    let migrated = Command::new(env!("CARGO_BIN_EXE_denoize"))
        .arg(&input)
        .arg(&output)
        .args(["--batch", "--resume", "--json", "--jobs", "2", "--force"])
        .output()
        .expect("force metadata-preserving batch");
    assert_summary_counts(&summary(&migrated), 1, 0, 0);
}

#[test]
fn codec_option_change_requires_regeneration() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));

    let first = run_batch_with_options(
        &input,
        &output,
        &["--output-format", "mp3", "--mp3-bitrate", "96"],
    );
    assert_summary_counts(&summary(&first), 1, 0, 0);

    let refused = run_batch_with_options(
        &input,
        &output,
        &["--output-format", "mp3", "--mp3-bitrate", "128"],
    );
    assert!(assert_preflight_failure(&refused).contains("--force"));

    let replaced = run_batch_with_options(
        &input,
        &output,
        &["--output-format", "mp3", "--mp3-bitrate", "128", "--force"],
    );
    assert_summary_counts(&summary(&replaced), 1, 0, 0);
}

#[cfg(feature = "onnx")]
fn identity_onnx_model(producer: &str) -> Vec<u8> {
    use prost::Message as _;
    use tract_onnx::pb::{
        tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };

    let dimension_value = |value| tensor_shape_proto::Dimension {
        value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
        denotation: String::new(),
    };
    let dimension_parameter = |name: &str| tensor_shape_proto::Dimension {
        value: Some(tensor_shape_proto::dimension::Value::DimParam(name.into())),
        denotation: String::new(),
    };
    let value_info = |name: &str| ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![dimension_value(1), dimension_parameter("samples")],
                }),
            })),
        }),
        doc_string: String::new(),
    };
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: producer.into(),
        graph: Some(GraphProto {
            name: "identity-waveform".into(),
            node: vec![NodeProto {
                input: vec!["input".into()],
                output: vec!["output".into()],
                name: "identity".into(),
                op_type: "Identity".into(),
                ..Default::default()
            }],
            input: vec![value_info("input")],
            output: vec![value_info("output")],
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

#[cfg(feature = "onnx")]
#[test]
fn model_content_change_requires_regeneration() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    let model = root.path().join("identity.onnx");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));
    std::fs::write(&model, identity_onnx_model("model-a")).expect("write first ONNX model");
    let model_path = model.to_str().expect("test model path is UTF-8");

    let first = run_batch_with_options(
        &input,
        &output,
        &[
            "--backend",
            "onnx",
            "--onnx-model",
            model_path,
            "--onnx-rate",
            "16000",
        ],
    );
    assert_summary_counts(&summary(&first), 1, 0, 0);
    let completed = std::fs::read(output.join("sample.wav")).unwrap();

    std::fs::write(&model, identity_onnx_model("model-b")).expect("replace ONNX model");
    let refused = run_batch_with_options(
        &input,
        &output,
        &[
            "--backend",
            "onnx",
            "--onnx-model",
            model_path,
            "--onnx-rate",
            "16000",
        ],
    );
    assert!(assert_preflight_failure(&refused).contains("--force"));
    assert_eq!(std::fs::read(output.join("sample.wav")).unwrap(), completed);

    let replaced = run_batch_with_options(
        &input,
        &output,
        &[
            "--backend",
            "onnx",
            "--onnx-model",
            model_path,
            "--onnx-rate",
            "16000",
            "--force",
        ],
    );
    assert_summary_counts(&summary(&replaced), 1, 0, 0);
}

#[cfg(feature = "rnnoise")]
#[test]
fn actual_backend_change_requires_regeneration() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));

    let first = run_batch_with_options(&input, &output, &["--backend", "classical"]);
    assert_summary_counts(&summary(&first), 1, 0, 0);
    let refused = run_batch_with_options(&input, &output, &["--backend", "rnnoise"]);
    assert!(assert_preflight_failure(&refused).contains("--force"));
    let replaced = run_batch_with_options(&input, &output, &["--backend", "rnnoise", "--force"]);
    assert_summary_counts(&summary(&replaced), 1, 0, 0);
}

#[test]
fn changed_or_truncated_output_is_never_skipped() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));

    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);
    let destination = output.join("sample.wav");
    std::fs::write(&destination, b"truncated output").expect("truncate completed output");
    let truncated = std::fs::read(&destination).unwrap();

    let refused = run_batch(&input, &output);
    assert!(assert_preflight_failure(&refused).contains("--force"));
    assert_eq!(std::fs::read(&destination).unwrap(), truncated);

    let replaced = run_batch_with_options(&input, &output, &["--force"]);
    assert_summary_counts(&summary(&replaced), 1, 0, 0);
    assert_ne!(std::fs::read(destination).unwrap(), truncated);
}

#[test]
fn replaced_output_is_never_skipped() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_silent_wav(&input.join("sample.wav"));

    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);
    let destination = output.join("sample.wav");
    std::fs::remove_file(&destination).expect("remove completed output");
    write_tone_wav(&destination);
    let replacement = std::fs::read(&destination).unwrap();

    let refused = run_batch(&input, &output);
    assert!(assert_preflight_failure(&refused).contains("--force"));
    assert_eq!(std::fs::read(&destination).unwrap(), replacement);

    let replaced = run_batch_with_options(&input, &output, &["--force"]);
    assert_summary_counts(&summary(&replaced), 1, 0, 0);
    assert_ne!(std::fs::read(destination).unwrap(), replacement);
}

#[test]
fn missing_recorded_output_is_processed_without_force() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_silent_wav(&input.join("sample.wav"));

    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);
    std::fs::remove_file(output.join("sample.wav")).expect("remove completed output");

    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);
    assert!(output.join("sample.wav").exists());
}

#[test]
fn legacy_v1_and_v2_entries_require_one_forced_regeneration() {
    for (index, legacy) in [
        "sample.wav\n".to_string(),
        format!("v2:{}\n", "00".repeat(32)),
    ]
    .into_iter()
    .enumerate()
    {
        let root = TestDirectory::create();
        let input = root.path().join(format!("input-{index}"));
        let output = root.path().join(format!("output-{index}"));
        std::fs::create_dir_all(&input).expect("create test input directory");
        std::fs::create_dir_all(&output).expect("create test output directory");
        write_tone_wav(&input.join("sample.wav"));
        let existing = b"legacy output must survive";
        std::fs::write(output.join("sample.wav"), existing).expect("write legacy output");
        std::fs::write(output.join(".denoize-state"), legacy).expect("write legacy state");

        let refused = run_batch(&input, &output);
        assert!(assert_preflight_failure(&refused).contains("--force"));
        assert_eq!(std::fs::read(output.join("sample.wav")).unwrap(), existing);

        let migrated = run_batch_with_options(&input, &output, &["--force"]);
        assert_summary_counts(&summary(&migrated), 1, 0, 0);
        let exact = run_batch(&input, &output);
        assert_summary_counts(&summary(&exact), 0, 1, 0);
    }
}

#[cfg(unix)]
#[test]
fn symlinked_output_is_not_trusted_and_force_replaces_only_the_link() {
    use std::os::unix::fs::symlink;

    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));
    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);

    let destination = output.join("sample.wav");
    std::fs::remove_file(&destination).expect("remove completed output");
    let target = root.path().join("symlink-target");
    let target_bytes = b"symlink target must survive";
    std::fs::write(&target, target_bytes).expect("write symlink target");
    symlink(&target, &destination).expect("replace output with symlink");

    let refused = run_batch(&input, &output);
    assert!(assert_preflight_failure(&refused).contains("--force"));
    assert_eq!(std::fs::read(&target).unwrap(), target_bytes);

    let replaced = run_batch_with_options(&input, &output, &["--force"]);
    assert_summary_counts(&summary(&replaced), 1, 0, 0);
    assert!(!std::fs::symlink_metadata(&destination)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(std::fs::read(target).unwrap(), target_bytes);
}

#[cfg(any(unix, windows))]
#[test]
fn multiply_linked_output_is_not_trusted_and_force_breaks_the_link() {
    let root = TestDirectory::create();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir_all(&input).expect("create test input directory");
    write_tone_wav(&input.join("sample.wav"));
    assert_summary_counts(&summary(&run_batch(&input, &output)), 1, 0, 0);

    let destination = output.join("sample.wav");
    let alias = root.path().join("output-alias.wav");
    std::fs::hard_link(&destination, &alias).expect("hard-link completed output");
    let old_bytes = std::fs::read(&alias).unwrap();

    let refused = run_batch(&input, &output);
    assert!(assert_preflight_failure(&refused).contains("--force"));
    assert_eq!(std::fs::read(&alias).unwrap(), old_bytes);

    let replaced = run_batch_with_options(&input, &output, &["--force"]);
    assert_summary_counts(&summary(&replaced), 1, 0, 0);
    assert_eq!(std::fs::read(&alias).unwrap(), old_bytes);
    std::fs::write(&destination, b"new independent destination")
        .expect("rewrite replaced destination");
    assert_eq!(std::fs::read(alias).unwrap(), old_bytes);
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
    write_silent_wav(&input_b.join("sample.wav"));

    let first = run_batch(&input_a, &output);
    assert_summary_counts(&summary(&first), 1, 0, 0);
    let second = run_batch_with_options(&input_b, &output, &["--force"]);

    assert_summary_counts(&summary(&second), 1, 0, 0);
}
