use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&str], model_dir: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .env("DENOIZE_MODEL_DIR", model_dir)
        .output()
        .unwrap()
}

fn write_wav(path: &Path, frames: usize) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in 0..frames {
        let time = frame as f64 / 48_000.0;
        let envelope = if frame % 9_600 < 7_200 { 1.0 } else { 0.02 };
        let sample =
            ((std::f64::consts::TAU * 180.0 * time).sin() * 0.32 * envelope * f64::from(i16::MAX))
                as i16;
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn recommendation_json_is_bounded_versioned_and_network_free() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    let models = directory.path().join("models");
    std::fs::create_dir(&models).unwrap();
    write_wav(&input, 48_000 * 3);

    let output = run(
        &[
            "recommend",
            input.to_str().unwrap(),
            "--goal",
            "speed",
            "--analysis-seconds",
            "1",
            "--accelerator",
            "cpu",
            "--json",
        ],
        &models,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1);
    assert!(
        !stdout.contains(directory.path().to_str().unwrap()),
        "report leaked a filesystem path: {stdout}"
    );
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["schema"], "denoize-recommendation-v1");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["network_accessed"], false);
    assert_eq!(value["goal"], "speed");
    assert_eq!(value["input"]["format"], "wav");
    assert_eq!(value["input"]["analysis_mode"], "bounded-stream");
    assert_eq!(value["input"]["total_frames"], 48_000 * 3);
    assert_eq!(value["input"]["analyzed_frames"], 48_000);
    assert_eq!(
        value["input"]["analysis_sha256"].as_str().unwrap().len(),
        64
    );
    assert!(value["calibration"].is_null());
    assert!(!value["decision"]["backend"].as_str().unwrap().is_empty());
    assert!(value["decision"]["strength"].as_f64().unwrap() >= 0.0);
    assert!(value["decision"]["adaptive_noise"].is_boolean());
    assert!(value["decision"]["vad"].is_boolean());
    assert!(!value["candidates"].as_array().unwrap().is_empty());
    assert!(value["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .all(|candidate| candidate
            .as_object()
            .unwrap()
            .contains_key("estimated_gpu_memory_bytes")));
    assert_eq!(
        value["decision"]["backend"],
        value["candidates"][0]["backend"]
    );

    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../schemas/denoize-recommendation-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(schema["properties"]["schema"]["const"], value["schema"]);
    assert_eq!(
        schema["properties"]["schema_version"]["const"],
        value["schema_version"]
    );
    for field in [
        "network_accessed",
        "goal",
        "input",
        "device",
        "calibration",
        "decision",
        "candidates",
    ] {
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|required| required == field));
    }
    #[cfg(feature = "gtcrn")]
    assert_eq!(
        std::fs::read_dir(&models).unwrap().count(),
        0,
        "recommendation must not create model provenance or catalog state"
    );
}

#[test]
fn recommendation_calibration_emits_reproducible_fixture_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    let models = directory.path().join("models");
    std::fs::create_dir(&models).unwrap();
    write_wav(&input, 4_800);
    let output = run(
        &[
            "recommend",
            input.to_str().unwrap(),
            "--calibration-runs",
            "1",
            "--accelerator",
            "cpu",
            "--json",
        ],
        &models,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["calibration"]["workload"], "classical-hifi-v1");
    assert_eq!(value["calibration"]["measured_runs"], 1);
    assert_eq!(
        value["calibration"]["fixture_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert!(value["calibration"]["median_elapsed_ms"].as_f64().unwrap() > 0.0);
    assert!(
        value["calibration"]["baseline_realtime_headroom"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn recommendation_human_output_and_help_expose_resource_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    write_wav(&input, 4_800);

    let help = run(&["recommend", "--help"], directory.path());
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("--max-gpu-memory"), "{help}");

    let output = run(
        &["recommend", input.to_str().unwrap(), "--accelerator", "cpu"],
        directory.path(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "recommendation:",
        "arguments:",
        "candidates:",
        "ram=",
        "gpu=n/a",
    ] {
        assert!(stdout.contains(expected), "missing {expected}: {stdout}");
    }
}

#[test]
fn recommendation_option_errors_precede_missing_input_io() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing.wav");
    let output = run(
        &[
            "recommend",
            missing.to_str().unwrap(),
            "--analysis-seconds",
            "0",
        ],
        directory.path(),
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("analysis duration"), "{stderr}");
    assert!(!stderr.contains("missing.wav"), "{stderr}");
}

#[cfg(unix)]
#[test]
fn recommendation_rejects_fifo_without_waiting_for_a_writer() {
    use std::os::unix::ffi::OsStrExt as _;

    let directory = tempfile::tempdir().unwrap();
    let fifo = directory.path().join("input.wav");
    let path = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let started = std::time::Instant::now();
    let output = run(
        &["recommend", fifo.to_str().unwrap(), "--json"],
        directory.path(),
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("regular file"));
}
