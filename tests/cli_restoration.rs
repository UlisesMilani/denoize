use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn write_click_fixture(path: &Path) {
    let sample_rate = 48_000u32;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in 0..4_800usize {
        let mut sample =
            0.08 * (std::f64::consts::TAU * 440.0 * frame as f64 / sample_rate as f64).sin();
        if frame == 2_400 {
            sample = 0.95;
        }
        writer
            .write_sample((sample * f64::from(i16::MAX)).round() as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
}

fn mask_has_exact_coverage(mask: &serde_json::Value) -> bool {
    let channels = mask["channels"].as_u64().unwrap() as usize;
    let frames = mask["frames"].as_u64().unwrap();
    let mut cursors = vec![0u64; channels];
    for run in mask["runs"].as_array().unwrap() {
        let channel = run["channel"].as_u64().unwrap() as usize;
        let start = run["start_frame"].as_u64().unwrap();
        let count = run["frame_count"].as_u64().unwrap();
        if channel >= channels || cursors[channel] != start || count == 0 {
            return false;
        }
        cursors[channel] += count;
    }
    cursors.into_iter().all(|cursor| cursor == frames)
}

#[test]
fn detect_only_exports_path_free_report_and_complete_mask() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("clicked.wav");
    let report = directory.path().join("report.json");
    let mask = directory.path().join("mask.json");
    write_click_fixture(&input);
    let output = run(&[
        "restore",
        input.to_str().unwrap(),
        "--detect-only",
        "--operations",
        "declick",
        "--report",
        report.to_str().unwrap(),
        "--mask",
        mask.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.contains(directory.path().to_str().unwrap()));
    let stdout_report: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let file_report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
    let file_mask: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&mask).unwrap()).unwrap();
    assert_eq!(stdout_report, file_report);
    assert_eq!(file_report["schema"], "denoize-restoration-report-v1");
    assert_eq!(file_report["mode"], "detect-only");
    assert_eq!(file_report["changed_samples"], 0);
    assert!(file_report["detected_samples"].as_u64().unwrap() > 0);
    assert_eq!(file_mask["schema"], "denoize-restoration-mask-v1");
    assert!(mask_has_exact_coverage(&file_mask));
    let detected_frames: u64 = file_mask["runs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|run| matches!(run["state"].as_str(), Some("detected" | "replaced")))
        .map(|run| run["frame_count"].as_u64().unwrap())
        .sum();
    assert_eq!(
        file_report["detected_samples"].as_u64(),
        Some(detected_frames)
    );
    assert_eq!(
        file_report["operations"][0]["detected_samples"].as_u64(),
        Some(detected_frames)
    );
    assert!(file_mask["runs"]
        .as_array()
        .unwrap()
        .iter()
        .all(|run| run["state"] != "replaced"));
}

#[test]
fn apply_is_same_length_deterministic_and_never_overwrites_by_default() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("clicked.wav");
    let first = directory.path().join("first.wav");
    let second = directory.path().join("second.wav");
    let first_report = directory.path().join("first-report.json");
    let second_report = directory.path().join("second-report.json");
    write_click_fixture(&input);
    for (output, report) in [(&first, &first_report), (&second, &second_report)] {
        let result = run(&[
            "restore",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--operations",
            "declick",
            "--report",
            report.to_str().unwrap(),
            "--max-memory",
            "1",
            "--json",
        ]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert!(report["changed_samples"].as_u64().unwrap() > 0);
    }
    let original = denoize::read_audio(&input).unwrap();
    let restored = denoize::read_audio(&first).unwrap();
    assert_eq!(restored.sample_rate, original.sample_rate);
    assert_eq!(restored.channels(), original.channels());
    assert_eq!(restored.frames(), original.frames());
    assert_ne!(restored.channels[0][2_400], original.channels[0][2_400]);
    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap()
    );
    assert_eq!(
        std::fs::read(&first_report).unwrap(),
        std::fs::read(&second_report).unwrap()
    );

    let existing = run(&[
        "restore",
        input.to_str().unwrap(),
        first.to_str().unwrap(),
        "--operations",
        "declick",
    ]);
    assert!(!existing.status.success());
    assert!(String::from_utf8_lossy(&existing.stderr).contains("exists"));

    let blocked_audio = directory.path().join("blocked.wav");
    let occupied_report = directory.path().join("occupied-report.json");
    std::fs::write(&occupied_report, b"keep me").unwrap();
    let blocked = run(&[
        "restore",
        input.to_str().unwrap(),
        blocked_audio.to_str().unwrap(),
        "--operations",
        "declick",
        "--report",
        occupied_report.to_str().unwrap(),
    ]);
    assert!(!blocked.status.success());
    assert!(!blocked_audio.exists());
    assert_eq!(std::fs::read(&occupied_report).unwrap(), b"keep me");
}

#[test]
fn invalid_options_and_input_alias_fail_before_publication() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("clicked.wav");
    write_click_fixture(&input);
    let missing = directory.path().join("missing.wav");
    let output = directory.path().join("output.wav");
    let invalid = run(&[
        "restore",
        missing.to_str().unwrap(),
        output.to_str().unwrap(),
        "--declip-iterations",
        "0",
    ]);
    assert!(!invalid.status.success());
    assert!(!output.exists());
    let stderr = String::from_utf8_lossy(&invalid.stderr);
    assert!(stderr.contains("1..=128"), "{stderr}");
    assert!(!stderr.contains("missing.wav"), "{stderr}");

    let alias = run(&[
        "restore",
        input.to_str().unwrap(),
        input.to_str().unwrap(),
        "--operations",
        "declick",
        "--replace",
    ]);
    assert!(!alias.status.success());
    assert!(String::from_utf8_lossy(&alias.stderr).contains("must not replace"));
    assert_eq!(denoize::read_audio(&input).unwrap().frames(), 4_800);
}
