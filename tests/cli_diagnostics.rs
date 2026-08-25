use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn write_fixture(path: &Path, damaged: bool) {
    let sample_rate = 48_000_u32;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    let mut random = 0x1234_5678_u32;
    for frame in 0..sample_rate * 2 {
        let time = frame as f64 / f64::from(sample_rate);
        let envelope = if frame % 12_000 < 9_000 { 1.0 } else { 0.08 };
        let clean = envelope
            * (0.28 * (std::f64::consts::TAU * 180.0 * time).sin()
                + 0.14 * (std::f64::consts::TAU * 510.0 * time).sin()
                + 0.07 * (std::f64::consts::TAU * 2_300.0 * time).sin());
        random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let white = (f64::from(random >> 8) / f64::from(0x00ff_ffff_u32)) * 2.0 - 1.0;
        let value = if damaged {
            clean * 2.4
                + white * 0.16
                + 0.18 * (std::f64::consts::TAU * 60.0 * time).sin()
                + 0.09 * (std::f64::consts::TAU * 120.0 * time).sin()
                + 0.05 * (std::f64::consts::TAU * 180.0 * time).sin()
        } else {
            clean
        }
        .clamp(-1.0, 1.0);
        writer
            .write_sample((value * f64::from(i16::MAX)).round() as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn diagnose_emits_bounded_closed_network_free_json() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("damaged.wav");
    write_fixture(&input, true);
    let output = run(&[
        "diagnose",
        input.to_str().unwrap(),
        "--analysis-seconds",
        "1",
        "--json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1);
    assert!(!stdout.contains(directory.path().to_str().unwrap()));
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["schema"], "denoize-diagnostic-v1");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["network_accessed"], false);
    assert_eq!(value["input"]["analysis_mode"], "bounded-stream");
    assert_eq!(value["input"]["source_analyzed_frames"], 48_000);
    assert_eq!(
        value["input"]["analysis_sha256"].as_str().unwrap().len(),
        64
    );
    assert_eq!(value["quality"]["method"], "denoize-native-no-reference-v1");
    assert!(value["quality"]["score"].as_f64().unwrap() <= 100.0);
    assert_eq!(value["findings"].as_array().unwrap().len(), 9);
    assert!(value["findings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|finding| finding["kind"] == "clipping" && finding["detected"] == true));

    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../schemas/denoize-diagnostic-v1.schema.json")).unwrap();
    assert_eq!(schema["properties"]["schema"]["const"], value["schema"]);
    assert_eq!(
        schema["properties"]["schema_version"]["const"],
        value["schema_version"]
    );
}

#[test]
fn assess_before_after_reports_improvement_and_presentation_safety() {
    let directory = tempfile::tempdir().unwrap();
    let before = directory.path().join("before.wav");
    let after = directory.path().join("after.wav");
    write_fixture(&before, true);
    write_fixture(&after, false);
    let output = run(&[
        "assess",
        before.to_str().unwrap(),
        after.to_str().unwrap(),
        "--analysis-seconds",
        "2",
        "--pretty",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema"], "denoize-assessment-v1");
    assert_eq!(value["network_accessed"], false);
    assert_eq!(value["verdict"], "improved");
    assert!(value["comparison"]["quality_score_delta"].as_f64().unwrap() > 3.0);
    assert_eq!(value["comparison"]["sample_rate_equal"], true);
    assert_eq!(value["comparison"]["channel_count_equal"], true);
    assert_eq!(value["comparison"]["presentation_preserved"], true);
    assert_eq!(value["comparison"]["semantic_fidelity_assessed"], false);
    assert!(value["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|warning| warning.as_str().unwrap().contains("hallucinated")));
}

#[test]
fn single_assessment_and_human_diagnosis_are_explicit_about_proxy_limits() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    write_fixture(&input, false);
    let assessment = run(&["assess", input.to_str().unwrap(), "--json"]);
    assert!(assessment.status.success());
    let value: serde_json::Value = serde_json::from_slice(&assessment.stdout).unwrap();
    assert_eq!(value["verdict"], "single-input");
    assert!(value["baseline"].is_null());
    assert!(value["comparison"].is_null());

    let diagnosis = run(&["diagnose", input.to_str().unwrap()]);
    assert!(diagnosis.status.success());
    let stdout = String::from_utf8(diagnosis.stdout).unwrap();
    for expected in [
        "quality:",
        "findings:",
        "recommended pipeline:",
        "does not assess words, phonemes, speaker identity",
    ] {
        assert!(stdout.contains(expected), "missing {expected}: {stdout}");
    }
}

#[test]
fn diagnostic_option_errors_precede_missing_input_io() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing.wav");
    let output = run(&[
        "diagnose",
        missing.to_str().unwrap(),
        "--analysis-seconds",
        "0",
    ]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("between 1 and 60"), "{stderr}");
    assert!(!stderr.contains("missing.wav"), "{stderr}");
}

#[test]
fn diagnostic_memory_ceiling_accepts_small_plans_and_rejects_large_ones() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    write_fixture(&input, false);

    let small = run(&[
        "diagnose",
        input.to_str().unwrap(),
        "--analysis-seconds",
        "1",
        "--max-memory",
        "1",
        "--json",
    ]);
    assert!(
        small.status.success(),
        "{}",
        String::from_utf8_lossy(&small.stderr)
    );

    let large = run(&[
        "diagnose",
        input.to_str().unwrap(),
        "--analysis-seconds",
        "60",
        "--max-memory",
        "1",
        "--json",
    ]);
    assert!(!large.status.success());
    assert!(large.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&large.stderr).contains("working-set"),
        "{}",
        String::from_utf8_lossy(&large.stderr)
    );
}

#[cfg(unix)]
#[test]
fn diagnose_rejects_fifo_without_waiting_for_a_writer() {
    use std::os::unix::ffi::OsStrExt as _;

    let directory = tempfile::tempdir().unwrap();
    let fifo = directory.path().join("input.wav");
    let path = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let started = std::time::Instant::now();
    let output = run(&["diagnose", fifo.to_str().unwrap(), "--json"]);
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("regular file"));
}
