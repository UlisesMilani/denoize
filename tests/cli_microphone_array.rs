use denoize::{
    sign_microphone_array_promotion_evidence, MicrophoneArrayConfig,
    MicrophoneArrayEvidenceStratum, MicrophoneArrayPromotionEvidencePayload,
};
use serde_json::Value;
use std::path::Path;
use std::process::{Command, Output};

const REQUIRED_STRATA: &[&str] = &[
    "bad-channel",
    "channel-permutation",
    "clock-skew",
    "diffuse-noise",
    "directional-noise",
    "gain-phase-mismatch",
    "moving-source",
    "program-stereo",
    "real-meeting",
    "simulated-rir",
    "two-microphone",
    "unseen-geometry",
];

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn payload(config: &MicrophoneArrayConfig) -> MicrophoneArrayPromotionEvidencePayload {
    MicrophoneArrayPromotionEvidencePayload {
        completed_at_unix_seconds: 1_800_000_000,
        implementation: "native-wpe-mask-mvdr-v1".into(),
        implementation_source_revision: "0123456789abcdef".into(),
        implementation_source_sha256: "11".repeat(32),
        configuration_sha256: config.digest().unwrap(),
        corpus_manifest_sha256: "22".repeat(32),
        evaluation_result_sha256: "33".repeat(32),
        listening_result_sha256: "44".repeat(32),
        strata: REQUIRED_STRATA
            .iter()
            .map(|id| MicrophoneArrayEvidenceStratum {
                id: (*id).into(),
                cases: 100,
                si_sdr_improvement_db: 1.0,
                wer_regression: 0.0,
                doa_error_degrees: 5.0,
                reference_coloration_db: 0.1,
                target_leakage_db: -6.0,
                non_finite_samples: 0,
                passed: true,
            })
            .collect(),
        real_meeting_cases: 100,
        unseen_geometry_cases: 100,
        permutation_cases: 100,
        paced_realtime_blocks: 10_000,
        worst_case_realtime_factor: 0.25,
        callback_allocations: 0,
        callback_locks: 0,
        callback_waits: 0,
        deadline_misses: 0,
        listener_count: 20,
        listener_preference: 0.6,
        accepted: true,
    }
}

fn write_array_wav(path: &Path, sample_rate: u32, frames: usize) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in 0..frames {
        let speech =
            (2.0 * std::f64::consts::PI * 440.0 * frame as f64 / sample_rate as f64).sin() * 0.2;
        let noise =
            (2.0 * std::f64::consts::PI * 173.0 * frame as f64 / sample_rate as f64).sin() * 0.03;
        writer.write_sample((speech + noise) as f32).unwrap();
        writer.write_sample((speech - noise) as f32).unwrap();
    }
    writer.finalize().unwrap();
}

struct Fixture {
    input: std::path::PathBuf,
    config: std::path::PathBuf,
    evidence: std::path::PathBuf,
    evidence_key: std::path::PathBuf,
    frames: usize,
}

fn fixture(root: &Path) -> Fixture {
    let config = MicrophoneArrayConfig::default();
    let (secret, public) = denoize::generate_receipt_keypair().unwrap();
    let signed = sign_microphone_array_promotion_evidence(payload(&config), &secret).unwrap();
    let config_path = root.join("array-config.json");
    let evidence = root.join("array-evidence.json");
    let evidence_key = root.join("array-public-key.json");
    std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    std::fs::write(&evidence, signed.to_pretty_json().unwrap()).unwrap();
    std::fs::write(&evidence_key, public.to_pretty_json().unwrap()).unwrap();
    let frames = 3_211;
    let input = root.join("microphone-array.wav");
    write_array_wav(&input, config.sample_rate, frames);
    Fixture {
        input,
        config: config_path,
        evidence,
        evidence_key,
        frames,
    }
}

#[test]
fn array_cli_authenticates_geometry_and_publishes_exact_mono_output() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = fixture(directory.path());
    let output_path = directory.path().join("output.wav");
    let report_path = directory.path().join("report.json");
    let output = run(&[
        "array",
        fixture.input.to_str().unwrap(),
        output_path.to_str().unwrap(),
        "--array-config",
        fixture.config.to_str().unwrap(),
        "--promotion-evidence",
        fixture.evidence.to_str().unwrap(),
        "--promotion-evidence-key",
        fixture.evidence_key.to_str().unwrap(),
        "--report",
        report_path.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema"], "denoize-microphone-array-report-v1");
    assert_eq!(report["input_channels"], 2);
    assert_eq!(report["output_channels"], 1);
    assert_eq!(report["input_frames"], fixture.frames);
    assert_eq!(report["output_frames"], fixture.frames);
    assert_eq!(report["reference_microphone_id"], "mic-0");
    assert_eq!(report["paths_recorded"], 0);
    assert_eq!(report["non_finite_samples"], 0);
    let serialized = serde_json::to_string(&report).unwrap();
    assert!(!serialized.contains(directory.path().to_str().unwrap()));
    let persisted: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(persisted, report);
    let reader = hound::WavReader::open(output_path).unwrap();
    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.duration() as usize, fixture.frames);
}

#[test]
fn tampered_array_evidence_fails_before_decode_or_publication() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = fixture(directory.path());
    let mut evidence: Value =
        serde_json::from_slice(&std::fs::read(&fixture.evidence).unwrap()).unwrap();
    evidence["payload"]["implementation_source_revision"] = Value::String("tampered".into());
    std::fs::write(
        &fixture.evidence,
        serde_json::to_vec_pretty(&evidence).unwrap(),
    )
    .unwrap();
    std::fs::write(&fixture.input, b"not audio").unwrap();
    let output_path = directory.path().join("output.wav");
    let report_path = directory.path().join("report.json");
    let output = run(&[
        "array-enhance",
        fixture.input.to_str().unwrap(),
        output_path.to_str().unwrap(),
        "--array-config",
        fixture.config.to_str().unwrap(),
        "--promotion-evidence",
        fixture.evidence.to_str().unwrap(),
        "--promotion-evidence-key",
        fixture.evidence_key.to_str().unwrap(),
        "--report",
        report_path.to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("signature verification failed"), "{stderr}");
    assert!(!stderr.contains("decode"), "{stderr}");
    assert!(!output_path.exists());
    assert!(!report_path.exists());
}
