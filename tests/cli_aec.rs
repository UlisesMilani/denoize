#![cfg(feature = "aec")]

use denoize::{
    sign_aec_promotion_evidence, AecConfig, AecEvidenceMetric, AecEvidenceMetricOperator,
    AecEvidenceStratum, AecEvidenceStratumKind, AecPromotionEvidencePayload,
};
use serde_json::Value;
use std::path::Path;
use std::process::{Command, Output};

const REQUIRED_STRATA: &[(&str, AecEvidenceStratumKind)] = &[
    ("background-noise", AecEvidenceStratumKind::Impairment),
    ("clipping", AecEvidenceStratumKind::Impairment),
    ("clock-drift-negative", AecEvidenceStratumKind::Transition),
    ("clock-drift-positive", AecEvidenceStratumKind::Transition),
    ("delay-jump", AecEvidenceStratumKind::Transition),
    ("delay-negative", AecEvidenceStratumKind::Transition),
    ("delay-positive", AecEvidenceStratumKind::Transition),
    ("double-talk", AecEvidenceStratumKind::DoubleTalk),
    ("far-end-clean", AecEvidenceStratumKind::FarEndOnly),
    ("linear-path", AecEvidenceStratumKind::FarEndOnly),
    ("music-playback", AecEvidenceStratumKind::Impairment),
    ("near-end-clean", AecEvidenceStratumKind::NearEndOnly),
    ("nonlinear-speaker", AecEvidenceStratumKind::Impairment),
    ("real-device", AecEvidenceStratumKind::Impairment),
    ("reference-loss", AecEvidenceStratumKind::Transition),
    ("room-change", AecEvidenceStratumKind::Transition),
    ("route-change", AecEvidenceStratumKind::Transition),
];

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn less(name: &str, limit: f64) -> AecEvidenceMetric {
    AecEvidenceMetric {
        metric: name.into(),
        value: limit,
        operator: AecEvidenceMetricOperator::LessOrEqual,
        limit,
        passed: true,
    }
}

fn greater(name: &str, limit: f64) -> AecEvidenceMetric {
    AecEvidenceMetric {
        metric: name.into(),
        value: limit,
        operator: AecEvidenceMetricOperator::GreaterOrEqual,
        limit,
        passed: true,
    }
}

fn metrics(kind: AecEvidenceStratumKind) -> Vec<AecEvidenceMetric> {
    let mut metrics = vec![
        less("latency.algorithmic-plus-buffering-ms", 20.0),
        less("output.duration-error-frames", 0.0),
        less("output.non-finite-samples", 0.0),
    ];
    match kind {
        AecEvidenceStratumKind::FarEndOnly => {
            metrics.push(greater("echo.erle-db", 10.0));
            metrics.push(greater("perceptual.aecmos-far-end", 3.5));
        }
        AecEvidenceStratumKind::NearEndOnly => {
            metrics.push(less("content.word-accuracy-regression", 0.02));
            metrics.push(less("near-end.attenuation-db", 1.0));
        }
        AecEvidenceStratumKind::DoubleTalk => {
            metrics.push(less("content.word-accuracy-regression", 0.02));
            metrics.push(less("near-end.attenuation-db", 1.5));
            metrics.push(greater("perceptual.aecmos-double-talk", 3.2));
        }
        AecEvidenceStratumKind::Transition => {
            metrics.push(less("near-end.attenuation-db", 1.5));
            metrics.push(less("reset.stale-output-frames", 0.0));
            metrics.push(less("transition.reconvergence-ms", 500.0));
        }
        AecEvidenceStratumKind::Impairment => {
            metrics.push(less("content.word-accuracy-regression", 0.02));
            metrics.push(greater("echo.erle-db", 6.0));
            metrics.push(less("near-end.attenuation-db", 1.5));
            metrics.push(greater("perceptual.aecmos", 3.0));
        }
    }
    metrics.sort_by(|left, right| left.metric.cmp(&right.metric));
    metrics
}

fn payload(config: &AecConfig) -> AecPromotionEvidencePayload {
    AecPromotionEvidencePayload {
        completed_at_unix_seconds: 1_700_000_000,
        implementation: "native-pfdnlms-v1".into(),
        implementation_source_revision: "0123456789abcdef".into(),
        implementation_source_sha256: "11".repeat(32),
        configuration_sha256: config.digest().unwrap(),
        corpus_manifest_sha256: "22".repeat(32),
        evaluation_result_sha256: "33".repeat(32),
        listening_result_sha256: "44".repeat(32),
        sample_rate: config.sample_rate,
        block_size_samples: config.block_size_samples,
        tail_samples: config.tail_samples,
        maximum_delay_samples: config.maximum_delay_samples,
        strata: REQUIRED_STRATA
            .iter()
            .map(|(id, kind)| AecEvidenceStratum {
                id: (*id).into(),
                kind: *kind,
                cases: 100,
                metrics: metrics(*kind),
            })
            .collect(),
        real_device_cases: 100,
        nonlinear_device_cases: 100,
        delay_transition_cases: 100,
        paced_realtime_blocks: 10_000,
        worst_case_realtime_factor: 0.5,
        callback_allocations: 0,
        callback_locks: 0,
        callback_waits: 0,
        callback_io_operations: 0,
        callback_log_operations: 0,
        deadline_misses: 0,
        stale_frames_after_reset: 0,
        minimum_listeners: 20,
        listener_count: 20,
        listener_preference: 0.5,
        listener_preference_limit: 0.5,
        accepted: true,
    }
}

fn pseudo_random(frames: usize) -> Vec<f32> {
    let mut state = 0x5eed_1234_u32;
    (0..frames)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f64 / u32::MAX as f64 * 1.2 - 0.6) as f32
        })
        .collect()
}

fn write_float_wav(path: &Path, sample_rate: u32, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for sample in samples {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
}

struct Fixture {
    microphone: std::path::PathBuf,
    reference: std::path::PathBuf,
    config: std::path::PathBuf,
    evidence: std::path::PathBuf,
    evidence_key: std::path::PathBuf,
    frames: usize,
    delay: i32,
}

fn fixture(root: &Path) -> Fixture {
    let config = AecConfig::default();
    let (secret, public) = denoize::generate_receipt_keypair().unwrap();
    let signed = sign_aec_promotion_evidence(payload(&config), &secret).unwrap();
    let config_path = root.join("aec-config.json");
    let evidence = root.join("aec-evidence.json");
    let evidence_key = root.join("aec-public-key.json");
    std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    std::fs::write(&evidence, signed.to_pretty_json().unwrap()).unwrap();
    std::fs::write(&evidence_key, public.to_pretty_json().unwrap()).unwrap();

    let frames = 1_603;
    let delay = 83_i32;
    let reference_samples = pseudo_random(frames + 256);
    let mut microphone_samples = vec![0.0_f32; frames];
    for (frame, sample) in microphone_samples.iter_mut().enumerate() {
        let reference_frame = frame as i64 - delay as i64;
        if reference_frame >= 0 {
            *sample = reference_samples[reference_frame as usize] * 0.4;
        }
    }
    let microphone = root.join("microphone.wav");
    let reference = root.join("reference.wav");
    write_float_wav(&microphone, config.sample_rate, &microphone_samples);
    write_float_wav(&reference, config.sample_rate, &reference_samples);
    Fixture {
        microphone,
        reference,
        config: config_path,
        evidence,
        evidence_key,
        frames,
        delay,
    }
}

#[test]
fn aec_cli_authenticates_evidence_and_preserves_exact_geometry() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = fixture(directory.path());
    let output_path = directory.path().join("output.wav");
    let report_path = directory.path().join("report.json");
    let output = run(&[
        "aec",
        fixture.microphone.to_str().unwrap(),
        fixture.reference.to_str().unwrap(),
        output_path.to_str().unwrap(),
        "--promotion-evidence",
        fixture.evidence.to_str().unwrap(),
        "--promotion-evidence-key",
        fixture.evidence_key.to_str().unwrap(),
        "--aec-config",
        fixture.config.to_str().unwrap(),
        "--route-generation",
        "7",
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
    assert_eq!(report["schema"], "denoize-aec-report-v1");
    assert_eq!(report["output_frames"], fixture.frames);
    assert_eq!(report["microphone_frames"], fixture.frames);
    assert_eq!(report["delay"]["signed_delay_samples"], fixture.delay);
    assert_eq!(report["route_generation"], 7);
    assert_eq!(report["reset_reasons"]["initial"], 1);
    assert_eq!(report["paths_recorded"], 0);
    assert_eq!(report["non_finite_output_samples"], 0);
    let serialized = serde_json::to_string(&report).unwrap();
    assert!(!serialized.contains(directory.path().to_str().unwrap()));
    let persisted: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(persisted, report);
    let reader = hound::WavReader::open(output_path).unwrap();
    assert_eq!(reader.duration() as usize, fixture.frames);
}

#[test]
fn modified_evidence_fails_before_audio_decode_or_publication() {
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
    std::fs::write(&fixture.microphone, b"not audio").unwrap();
    let output_path = directory.path().join("output.wav");
    let report_path = directory.path().join("report.json");
    let output = run(&[
        "aec",
        fixture.microphone.to_str().unwrap(),
        fixture.reference.to_str().unwrap(),
        output_path.to_str().unwrap(),
        "--promotion-evidence",
        fixture.evidence.to_str().unwrap(),
        "--promotion-evidence-key",
        fixture.evidence_key.to_str().unwrap(),
        "--aec-config",
        fixture.config.to_str().unwrap(),
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
