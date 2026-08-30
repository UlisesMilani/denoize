#![cfg(feature = "onnx")]

use prost::Message;
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tract_onnx::pb::{
    attribute_proto, tensor_proto, tensor_shape_proto, type_proto, AttributeProto, GraphProto,
    ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto,
    ValueInfoProto,
};

const MODEL_RATE: u32 = 16_000;
const WINDOW: usize = 256;
const REQUIRED_STRATA: &[(&str, denoize::TargetSoundStratumKind)] = &[
    (
        "binaural-spatial",
        denoize::TargetSoundStratumKind::BinauralSpatial,
    ),
    (
        "class-confusable",
        denoize::TargetSoundStratumKind::TargetPresent,
    ),
    (
        "clean-bypass",
        denoize::TargetSoundStratumKind::TargetAbsent,
    ),
    ("low-snr", denoize::TargetSoundStratumKind::TargetPresent),
    (
        "multi-instance",
        denoize::TargetSoundStratumKind::TargetPresent,
    ),
    (
        "music-foreground",
        denoize::TargetSoundStratumKind::ProtectedForeground,
    ),
    (
        "query-alias",
        denoize::TargetSoundStratumKind::TargetPresent,
    ),
    (
        "speech-foreground",
        denoize::TargetSoundStratumKind::ProtectedForeground,
    ),
    (
        "target-absent",
        denoize::TargetSoundStratumKind::TargetAbsent,
    ),
    (
        "target-present",
        denoize::TargetSoundStratumKind::TargetPresent,
    ),
    (
        "tonal-target",
        denoize::TargetSoundStratumKind::TargetPresent,
    ),
    (
        "transient-target",
        denoize::TargetSoundStratumKind::TargetPresent,
    ),
    (
        "unseen-domain",
        denoize::TargetSoundStratumKind::TargetPresent,
    ),
    (
        "unseen-interferer",
        denoize::TargetSoundStratumKind::TargetPresent,
    ),
];

#[derive(Clone, Copy)]
enum ResidualMode {
    Zero,
    Identity,
}

struct Fixture {
    package: PathBuf,
    package_key: PathBuf,
    query: PathBuf,
    evidence: PathBuf,
    evidence_key: PathBuf,
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn query() -> denoize::TargetSoundQuery {
    denoize::TargetSoundQuery {
        schema: denoize::TARGET_SOUND_QUERY_SCHEMA.into(),
        schema_version: denoize::TARGET_SOUND_SCHEMA_VERSION,
        catalog_revision: "catalog-1".into(),
        classes: vec![
            denoize::TargetSoundCatalogClass {
                id: "alarm".into(),
                canonical_label: "Alarm".into(),
            },
            denoize::TargetSoundCatalogClass {
                id: "baby-cry".into(),
                canonical_label: "Baby cry".into(),
            },
        ],
        selected_class_id: "baby-cry".into(),
    }
}

fn build_fixture(
    directory: &Path,
    id: &str,
    probabilities: [f32; 3],
    residual_mode: ResidualMode,
    channels: usize,
    mode: denoize::TargetSoundMode,
) -> Fixture {
    let root = directory.join(id);
    let components = root.join("components");
    std::fs::create_dir_all(&components).unwrap();
    let mut model = Vec::new();
    target_sound_model(probabilities, residual_mode, channels)
        .encode(&mut model)
        .unwrap();
    let license = b"MIT License\nfixture only\n".to_vec();
    let provenance = br#"{"schema":"denoize-test-provenance-v1"}"#.to_vec();
    let audio_values = vec![0.0_f64; WINDOW * channels];
    let target_values = audio_values.clone();
    let residual_values = match residual_mode {
        ResidualMode::Zero => vec![0.0_f64; WINDOW * channels],
        ResidualMode::Identity => audio_values.clone(),
    };
    let vectors = serde_json::to_vec(&json!({
        "schema": "denoize-runtime-model-numerical-vectors-v1",
        "profile_id": "fp32",
        "cases": [{
            "id": "target-sound-identity",
            "inputs": [
                {
                    "name": "audio",
                    "element_type": "float32",
                    "shape": [1, channels, WINDOW],
                    "values": audio_values
                },
                {
                    "name": "query",
                    "element_type": "float32",
                    "shape": [1, 2],
                    "values": [0.0, 1.0]
                }
            ],
            "outputs": [
                {
                    "name": "target",
                    "element_type": "float32",
                    "shape": [1, channels, WINDOW],
                    "values": target_values
                },
                {
                    "name": "residual",
                    "element_type": "float32",
                    "shape": [1, channels, WINDOW],
                    "values": residual_values
                },
                {
                    "name": "presence",
                    "element_type": "float32",
                    "shape": [1, 3],
                    "values": probabilities
                }
            ],
            "tolerance": { "absolute": 0.000001, "relative": 0.000001 }
        }]
    }))
    .unwrap();
    for (name, bytes) in [
        ("model.onnx", model.as_slice()),
        ("LICENSE.txt", license.as_slice()),
        ("provenance.json", provenance.as_slice()),
        ("vectors-fp32.json", vectors.as_slice()),
    ] {
        std::fs::write(components.join(name), bytes).unwrap();
    }

    let minisign::KeyPair { pk, sk } = minisign::KeyPair::generate_unencrypted_keypair().unwrap();
    let mut key_number = [0_u8; 8];
    key_number.copy_from_slice(pk.keynum());
    let key_id = format!("{:016X}", u64::from_le_bytes(key_number));
    let public_key = pk.to_box().unwrap().into_string();
    let audio_axes = json!([
        { "name": "batch", "kind": "batch", "fixed": 1 },
        { "name": "channels", "kind": "channel", "fixed": channels },
        { "name": "samples", "kind": "sample", "fixed": WINDOW }
    ]);
    let channel_roles = match channels {
        1 => json!([{ "channel_index": 0, "role": "program-center" }]),
        2 => json!([
            { "channel_index": 0, "role": "program-left" },
            { "channel_index": 1, "role": "program-right" }
        ]),
        _ => panic!("test fixture supports only mono/stereo"),
    };
    let manifest = json!({
        "schema": denoize::RUNTIME_MODEL_PACKAGE_SCHEMA_V2,
        "format_version": denoize::RUNTIME_MODEL_PACKAGE_VERSION_V2,
        "package_id": format!("denoize.test.target-sound.{id}"),
        "package_revision": "1",
        "signing_key_id": key_id,
        "runtime": {
            "kind": "onnx-audio-graph-v2",
            "sample_rate_hz": MODEL_RATE,
            "mode": "finite"
        },
        "frontend": {
            "normalization": "pcm-f32-minus-one-to-one-v1",
            "resampling": "bandlimited-waveform-v1",
            "duration": "preserve-input-frames-v1",
            "channels": {
                "policy": "program-multichannel",
                "roles": channel_roles,
                "geometry": null
            }
        },
        "tensors": {
            "inputs": [
                { "name": "audio", "role": "audio", "element_type": "float32", "axes": audio_axes, "optional": false, "state_id": null },
                {
                    "name": "query",
                    "role": "query",
                    "element_type": "float32",
                    "axes": [
                        { "name": "batch", "kind": "batch", "fixed": 1 },
                        { "name": "classes", "kind": "feature", "fixed": 2 }
                    ],
                    "optional": false,
                    "state_id": null
                }
            ],
            "outputs": [
                { "name": "target", "role": "audio", "element_type": "float32", "axes": audio_axes, "optional": false, "state_id": null },
                { "name": "residual", "role": "residual", "element_type": "float32", "axes": audio_axes, "optional": false, "state_id": null },
                {
                    "name": "presence",
                    "role": "diagnostic",
                    "element_type": "float32",
                    "axes": [
                        { "name": "batch", "kind": "batch", "fixed": 1 },
                        { "name": "classes", "kind": "feature", "fixed": 3 }
                    ],
                    "optional": false,
                    "state_id": null
                }
            ]
        },
        "state_pairs": [],
        "latency": {
            "frame_samples": WINDOW,
            "hop_samples": WINDOW,
            "left_context_samples": 0,
            "right_context_samples": 0,
            "lookahead_samples": 0,
            "algorithmic_latency_samples": 0,
            "flush_samples": 0
        },
        "components": [
            { "id": "model-fp32", "kind": "onnx-model", "file": file_contract("model.onnx", &model) },
            { "id": "license", "kind": "license-notice", "file": file_contract("LICENSE.txt", &license) },
            { "id": "provenance", "kind": "provenance-json", "file": file_contract("provenance.json", &provenance) },
            { "id": "vectors-fp32", "kind": "numerical-vectors-json", "file": file_contract("vectors-fp32.json", &vectors) }
        ],
        "precision_profiles": [{
            "id": "fp32",
            "element_type": "float32",
            "model_component": "model-fp32",
            "numerical_vectors_component": "vectors-fp32",
            "resources": {
                "max_session_memory_bytes": denoize::estimate_model_session_bytes(model.len() as u64).unwrap(),
                "max_worker_memory_bytes": 4096,
                "max_gpu_session_memory_bytes": 0,
                "max_gpu_worker_memory_bytes": 0,
                "accelerators": ["cpu"]
            }
        }],
        "default_precision_profile": "fp32",
        "license": { "spdx": "MIT", "notice_component": "license" },
        "provenance": {
            "component": "provenance",
            "source_repository": "https://example.invalid/target-sound",
            "source_revision": "0123456789abcdef",
            "source_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "source_license_spdx": "MIT",
            "checkpoint_source": "https://example.invalid/target-sound.ckpt",
            "checkpoint_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "checkpoint_license_spdx": "MIT",
            "conversion_tool": "denoize-test-fixture",
            "conversion_revision": "1",
            "training_datasets": [{
                "id": "synthetic",
                "source": "urn:denoize:test:synthetic",
                "revision": "1",
                "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
                "license_spdx": "CC0-1.0"
            }]
        }
    });
    let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
    let signature = minisign::sign(
        None,
        &sk,
        Cursor::new(&manifest_bytes),
        Some("denoize target-sound CLI fixture"),
        Some("untrusted comment: denoize test fixture"),
    )
    .unwrap()
    .into_string();
    let manifest_path = root.join("manifest.json");
    let signature_path = root.join("manifest.json.sig");
    let package_key = root.join("model.pub");
    std::fs::write(&manifest_path, manifest_bytes).unwrap();
    std::fs::write(&signature_path, signature).unwrap();
    std::fs::write(&package_key, public_key).unwrap();
    let package = root.join("model.dmp");
    denoize::build_runtime_model_package_v2(
        &package,
        manifest_path,
        signature_path,
        &package_key,
        components,
    )
    .unwrap();

    let query_document = query();
    let query_path = root.join("query.json");
    std::fs::write(
        &query_path,
        serde_json::to_vec_pretty(&query_document).unwrap(),
    )
    .unwrap();
    let opened = denoize::RuntimeModelPackage::open(&package, &package_key).unwrap();
    let mut config = denoize::TargetSoundConfig::default();
    config.mode = mode;
    let payload = promotion_payload(opened.package_sha256(), &query_document, &config);
    let (secret, public) = denoize::generate_receipt_keypair().unwrap();
    let signed = denoize::sign_target_sound_promotion_evidence(payload, &secret).unwrap();
    let evidence = root.join("evidence.json");
    let evidence_key = root.join("evidence-public.json");
    std::fs::write(&evidence, serde_json::to_vec_pretty(&signed).unwrap()).unwrap();
    std::fs::write(&evidence_key, serde_json::to_vec_pretty(&public).unwrap()).unwrap();
    Fixture {
        package,
        package_key,
        query: query_path,
        evidence,
        evidence_key,
    }
}

fn cli_args<'a>(
    fixture: &'a Fixture,
    input: &'a Path,
    target: &'a Path,
    residual: &'a Path,
    output: &'a Path,
    report: &'a Path,
    mode: &'a str,
) -> Vec<&'a str> {
    vec![
        "target-sound",
        input.to_str().unwrap(),
        "--query",
        fixture.query.to_str().unwrap(),
        "--target",
        target.to_str().unwrap(),
        "--residual",
        residual.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
        "--mode",
        mode,
        "--model-package",
        fixture.package.to_str().unwrap(),
        "--model-package-key",
        fixture.package_key.to_str().unwrap(),
        "--promotion-evidence",
        fixture.evidence.to_str().unwrap(),
        "--promotion-evidence-key",
        fixture.evidence_key.to_str().unwrap(),
        "--json",
    ]
}

#[test]
fn present_publishes_exact_target_residual_output_and_closed_report() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    write_wav(&input, WINDOW * 3, 1);
    let fixture = build_fixture(
        directory.path(),
        "present",
        [0.0, 0.0, 1.0],
        ResidualMode::Zero,
        1,
        denoize::TargetSoundMode::Preserve,
    );
    let target = directory.path().join("target.wav");
    let residual = directory.path().join("residual.wav");
    let output = directory.path().join("output.wav");
    let report_path = directory.path().join("report.json");
    let result = run(&cli_args(
        &fixture,
        &input,
        &target,
        &residual,
        &output,
        &report_path,
        "preserve",
    ));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for path in [&target, &residual, &output, &report_path] {
        assert!(path.is_file(), "missing {}", path.display());
    }
    let input_audio = denoize::read_audio(&input).unwrap();
    let target_audio = denoize::read_audio(&target).unwrap();
    let residual_audio = denoize::read_audio(&residual).unwrap();
    let selected_audio = denoize::read_audio(&output).unwrap();
    assert_eq!(target_audio.sample_rate, MODEL_RATE);
    assert_eq!(target_audio.frames(), input_audio.frames());
    assert_eq!(target_audio.sample_format, hound::SampleFormat::Float);
    for frame in 0..input_audio.frames() {
        assert!(
            (target_audio.channels[0][frame] + residual_audio.channels[0][frame]
                - input_audio.channels[0][frame])
                .abs()
                < 1.0e-6
        );
        assert!(
            (selected_audio.channels[0][frame] - target_audio.channels[0][frame]).abs() < 1.0e-6
        );
    }
    let report: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["decision"], "accepted-present");
    assert_eq!(report["query"]["class_id"], "baby-cry");
    assert_eq!(report["query"]["class_index"], 1);
    assert_eq!(report["query"]["encoding"], "one-hot-v1");
    assert_eq!(report["query"]["open_text_accepted"], false);
    assert_eq!(report["target_published"], true);
    assert_eq!(report["residual_published"], true);
    assert_eq!(report["output_published"], true);
    assert_eq!(report["path_fields_recorded"], 0);
    let text = serde_json::to_string(&report).unwrap();
    assert!(!text.contains(directory.path().to_str().unwrap()));
}

#[test]
fn absent_and_conservation_failure_publish_only_reports() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.wav");
    write_wav(&input, WINDOW * 3, 1);
    for (id, probabilities, residual_mode, expected) in [
        (
            "absent",
            [1.0, 0.0, 0.0],
            ResidualMode::Zero,
            "withheld-absent",
        ),
        (
            "nonconserving",
            [0.0, 0.0, 1.0],
            ResidualMode::Identity,
            "withheld-safety-gate",
        ),
    ] {
        let fixture = build_fixture(
            directory.path(),
            id,
            probabilities,
            residual_mode,
            1,
            denoize::TargetSoundMode::Preserve,
        );
        let target = directory.path().join(format!("{id}-target.wav"));
        let residual = directory.path().join(format!("{id}-residual.wav"));
        let output = directory.path().join(format!("{id}-output.wav"));
        let report_path = directory.path().join(format!("{id}-report.json"));
        let result = run(&cli_args(
            &fixture,
            &input,
            &target,
            &residual,
            &output,
            &report_path,
            "preserve",
        ));
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!target.exists());
        assert!(!residual.exists());
        assert!(!output.exists());
        let report: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(report["decision"], expected);
        assert_eq!(report["target_published"], false);
        assert_eq!(report["residual_published"], false);
        assert_eq!(report["output_published"], false);
        assert!(report["target_pcm_sha256"].is_null());
        assert!(report["residual_pcm_sha256"].is_null());
        assert!(report["output_pcm_sha256"].is_null());
    }
}

#[test]
fn stereo_remove_selects_the_exact_residual() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("stereo-input.wav");
    write_wav(&input, WINDOW * 3, 2);
    let fixture = build_fixture(
        directory.path(),
        "stereo-remove",
        [0.0, 0.0, 1.0],
        ResidualMode::Zero,
        2,
        denoize::TargetSoundMode::Remove,
    );
    let target = directory.path().join("stereo-target.wav");
    let residual = directory.path().join("stereo-residual.wav");
    let output = directory.path().join("stereo-output.wav");
    let report_path = directory.path().join("stereo-report.json");
    let result = run(&cli_args(
        &fixture,
        &input,
        &target,
        &residual,
        &output,
        &report_path,
        "remove",
    ));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let residual_audio = denoize::read_audio(&residual).unwrap();
    let selected_audio = denoize::read_audio(&output).unwrap();
    assert_eq!(residual_audio.channels(), 2);
    assert_eq!(selected_audio.channels(), 2);
    for (residual_channel, selected_channel) in
        residual_audio.channels.iter().zip(&selected_audio.channels)
    {
        for (&residual_sample, &selected_sample) in residual_channel.iter().zip(selected_channel) {
            assert!((residual_sample - selected_sample).abs() < 1.0e-6);
        }
    }
    let report: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["mode"], "remove");
    assert_eq!(report["source_channels"], 2);
    assert!(report["measurements"]["target_stereo_correlation_delta"].is_number());
    assert!(report["measurements"]["residual_mid_side_energy_ratio_delta_db"].is_number());
}

#[test]
fn changed_catalog_fails_before_audio_decode_or_publication() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = build_fixture(
        directory.path(),
        "catalog-tamper",
        [0.0, 0.0, 1.0],
        ResidualMode::Zero,
        1,
        denoize::TargetSoundMode::Preserve,
    );
    let input = directory.path().join("not-audio.wav");
    std::fs::write(&input, b"must not be decoded").unwrap();
    let mut changed: Value =
        serde_json::from_slice(&std::fs::read(&fixture.query).unwrap()).unwrap();
    changed["classes"][0]["canonical_label"] = json!("Changed label");
    let changed_query = directory.path().join("changed-query.json");
    std::fs::write(&changed_query, serde_json::to_vec(&changed).unwrap()).unwrap();
    let changed_fixture = Fixture {
        package: fixture.package,
        package_key: fixture.package_key,
        query: changed_query,
        evidence: fixture.evidence,
        evidence_key: fixture.evidence_key,
    };
    let target = directory.path().join("target.wav");
    let residual = directory.path().join("residual.wav");
    let output = directory.path().join("output.wav");
    let report = directory.path().join("report.json");
    let result = run(&cli_args(
        &changed_fixture,
        &input,
        &target,
        &residual,
        &output,
        &report,
        "preserve",
    ));
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("query catalog SHA-256"), "{stderr}");
    assert!(!stderr.contains("decode"), "{stderr}");
    assert!(!target.exists());
    assert!(!residual.exists());
    assert!(!output.exists());
    assert!(!report.exists());
}

fn promotion_payload(
    package_sha256: &str,
    query: &denoize::TargetSoundQuery,
    config: &denoize::TargetSoundConfig,
) -> denoize::TargetSoundPromotionEvidencePayload {
    denoize::TargetSoundPromotionEvidencePayload {
        completed_at_unix_seconds: 1,
        model_package_sha256: package_sha256.into(),
        source_revision: "0123456789abcdef".into(),
        source_sha256: "1".repeat(64),
        checkpoint_sha256: "2".repeat(64),
        configuration_sha256: config.digest().unwrap(),
        query_catalog_sha256: query.catalog_sha256().unwrap(),
        query_catalog_revision: query.catalog_revision.clone(),
        query_class_ids_sha256: query.class_ids_sha256().unwrap(),
        query_class_count: query.classes.len() as u32,
        class_coverage_manifest_sha256: "3".repeat(64),
        evaluated_class_count: query.classes.len() as u32,
        minimum_present_cases_per_class: 20,
        minimum_absent_cases_per_class: 20,
        worst_class_false_positive_rate: 0.01,
        worst_class_false_negative_rate: 0.05,
        artifact_bom_sha256: "4".repeat(64),
        training_dataset_license_manifest_sha256: "5".repeat(64),
        evaluation_corpus_manifest_sha256: "6".repeat(64),
        evaluation_corpus_license_manifest_sha256: "7".repeat(64),
        evaluation_result_sha256: "8".repeat(64),
        listening_result_sha256: "9".repeat(64),
        strata: REQUIRED_STRATA
            .iter()
            .map(|(id, kind)| denoize::TargetSoundEvidenceStratum {
                id: (*id).into(),
                kind: *kind,
                cases: 50,
                metrics: metrics(*kind),
            })
            .collect(),
        paired_cases: 1_000,
        target_absent_cases: 200,
        protected_foreground_cases: 200,
        binaural_cases: 200,
        listener_count: 20,
        listener_preference: 0.5,
        redistributed_restricted_artifacts: 0,
        unresolved_artifact_licenses: 0,
        unresolved_training_dataset_licenses: 0,
        unresolved_evaluation_dataset_licenses: 0,
        accepted: true,
    }
}

fn metrics(kind: denoize::TargetSoundStratumKind) -> Vec<denoize::TargetSoundMetricOutcome> {
    use denoize::TargetSoundMetricOperator::{GreaterOrEqual, LessOrEqual};
    let policies: &[(&str, denoize::TargetSoundMetricOperator, f64)] = match kind {
        denoize::TargetSoundStratumKind::TargetAbsent => &[
            ("output.clipped-samples", LessOrEqual, 0.0),
            ("output.duration-mismatch-samples", LessOrEqual, 0.0),
            ("output.non-finite-samples", LessOrEqual, 0.0),
            ("presence.expected-calibration-error", LessOrEqual, 0.05),
            ("presence.false-positive-rate", LessOrEqual, 0.01),
            ("recombination.maximum-absolute-error", LessOrEqual, 1.0e-5),
            ("target.output-rms-dbfs", LessOrEqual, -60.0),
        ],
        denoize::TargetSoundStratumKind::BinauralSpatial => &[
            (
                "extraction.target-si-sdr-improvement-db",
                GreaterOrEqual,
                3.0,
            ),
            ("output.clipped-samples", LessOrEqual, 0.0),
            ("output.duration-mismatch-samples", LessOrEqual, 0.0),
            ("output.non-finite-samples", LessOrEqual, 0.0),
            ("presence.expected-calibration-error", LessOrEqual, 0.05),
            ("presence.false-negative-rate", LessOrEqual, 0.05),
            ("recombination.maximum-absolute-error", LessOrEqual, 1.0e-5),
            ("residual.target-leakage-db", LessOrEqual, -20.0),
            ("spatial.ild-error-db", LessOrEqual, 1.0),
            ("spatial.itd-error-microseconds", LessOrEqual, 100.0),
        ],
        denoize::TargetSoundStratumKind::TargetPresent
        | denoize::TargetSoundStratumKind::ProtectedForeground => &[
            (
                "extraction.target-si-sdr-improvement-db",
                GreaterOrEqual,
                3.0,
            ),
            ("output.clipped-samples", LessOrEqual, 0.0),
            ("output.duration-mismatch-samples", LessOrEqual, 0.0),
            ("output.non-finite-samples", LessOrEqual, 0.0),
            ("output.protected-foreground-sdr-db", GreaterOrEqual, 20.0),
            ("presence.expected-calibration-error", LessOrEqual, 0.05),
            ("presence.false-negative-rate", LessOrEqual, 0.05),
            ("recombination.maximum-absolute-error", LessOrEqual, 1.0e-5),
            ("residual.target-leakage-db", LessOrEqual, -20.0),
        ],
    };
    policies
        .iter()
        .map(
            |(name, operator, limit)| denoize::TargetSoundMetricOutcome {
                metric: (*name).into(),
                value: *limit,
                operator: *operator,
                limit: *limit,
                passed: true,
            },
        )
        .collect()
}

fn file_contract(filename: &str, bytes: &[u8]) -> Value {
    json!({
        "filename": filename,
        "size_bytes": bytes.len(),
        "sha256": format!("{:x}", Sha256::digest(bytes))
    })
}

fn write_wav(path: &Path, frames: usize, channels: u16) {
    let spec = hound::WavSpec {
        channels,
        sample_rate: MODEL_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in 0..frames {
        for channel in 0..channels {
            let frequency = 220.0 + 110.0 * f64::from(channel);
            let value = (0.15
                * (std::f64::consts::TAU * frequency * frame as f64 / MODEL_RATE as f64).sin()
                * f64::from(i16::MAX))
            .round() as i16;
            writer.write_sample(value).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn target_sound_model(
    probabilities: [f32; 3],
    residual_mode: ResidualMode,
    channels: usize,
) -> ModelProto {
    let value_info = |name: &str, dims: Vec<tensor_shape_proto::Dimension>| ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(TensorShapeProto { dim: dims }),
            })),
        }),
        doc_string: String::new(),
    };
    let audio_shape = || {
        vec![
            dimension_value(1),
            dimension_value(channels as i64),
            dimension_value(WINDOW as i64),
        ]
    };
    let residual = match residual_mode {
        ResidualMode::Zero => NodeProto {
            input: vec!["audio".into(), "audio".into()],
            output: vec!["residual".into()],
            name: "zero-residual".into(),
            op_type: "Sub".into(),
            ..Default::default()
        },
        ResidualMode::Identity => NodeProto {
            input: vec!["audio".into()],
            output: vec!["residual".into()],
            name: "identity-residual".into(),
            op_type: "Identity".into(),
            ..Default::default()
        },
    };
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "denoize-test".into(),
        graph: Some(GraphProto {
            name: "target-sound-identity".into(),
            node: vec![
                NodeProto {
                    input: vec!["audio".into()],
                    output: vec!["target".into()],
                    name: "target".into(),
                    op_type: "Identity".into(),
                    ..Default::default()
                },
                residual,
                NodeProto {
                    output: vec!["presence".into()],
                    name: "presence".into(),
                    op_type: "Constant".into(),
                    attribute: vec![AttributeProto {
                        name: "value".into(),
                        r#type: attribute_proto::AttributeType::Tensor as i32,
                        t: Some(TensorProto {
                            dims: vec![1, 3],
                            data_type: tensor_proto::DataType::Float as i32,
                            float_data: probabilities.to_vec(),
                            ..Default::default()
                        }),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            input: vec![
                value_info("audio", audio_shape()),
                value_info("query", vec![dimension_value(1), dimension_value(2)]),
            ],
            output: vec![
                value_info("target", audio_shape()),
                value_info("residual", audio_shape()),
                value_info("presence", vec![dimension_value(1), dimension_value(3)]),
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn dimension_value(value: i64) -> tensor_shape_proto::Dimension {
    tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
    }
}
