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
const REQUIRED_STRATA: &[(&str, denoize::TargetSpeakerStratumKind)] = &[
    (
        "channel-mismatch",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "child-speaker",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "code-switching",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "codec-enrollment",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "different-sex",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "many-interferers",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "noisy-enrollment",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "one-interferer",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "real-t-conversation",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "reverberant-enrollment",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    ("same-sex", denoize::TargetSpeakerStratumKind::TargetPresent),
    (
        "same-words",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "similar-voices",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    ("singing", denoize::TargetSpeakerStratumKind::TargetPresent),
    (
        "speech-absent",
        denoize::TargetSpeakerStratumKind::TargetAbsent,
    ),
    (
        "target-absent",
        denoize::TargetSpeakerStratumKind::TargetAbsent,
    ),
    (
        "target-absent-same-words",
        denoize::TargetSpeakerStratumKind::TargetAbsent,
    ),
    (
        "target-absent-similar-interferer",
        denoize::TargetSpeakerStratumKind::TargetAbsent,
    ),
    (
        "target-present-clean",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "ts-superb",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "unseen-domain",
        denoize::TargetSpeakerStratumKind::TargetPresent,
    ),
    ("whisper", denoize::TargetSpeakerStratumKind::TargetPresent),
];

struct Fixture {
    package: PathBuf,
    package_key: PathBuf,
    evidence: PathBuf,
    evidence_key: PathBuf,
}

struct CausalFixture {
    package: PathBuf,
    package_key: PathBuf,
    offline_evidence: PathBuf,
    offline_evidence_key: PathBuf,
    causal_evidence: PathBuf,
    causal_evidence_key: PathBuf,
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn build_fixture(directory: &Path, id: &str, probabilities: [f32; 3]) -> Fixture {
    let root = directory.join(id);
    let components = root.join("components");
    std::fs::create_dir_all(&components).unwrap();
    let mut model = Vec::new();
    target_speaker_model(probabilities)
        .encode(&mut model)
        .unwrap();
    let license = b"MIT License\nfixture only\n".to_vec();
    let provenance = br#"{"schema":"denoize-test-provenance-v1"}"#.to_vec();
    let mixture_values = vec![0.0_f64, 0.1, -0.1, 0.2];
    let enrollment_values = vec![0.0_f64; 8];
    let vectors = serde_json::to_vec(&json!({
        "schema": "denoize-runtime-model-numerical-vectors-v1",
        "profile_id": "fp32",
        "cases": [{
            "id": "target-speaker-identity",
            "inputs": [
                {
                    "name": "mixture",
                    "element_type": "float32",
                    "shape": [1, 4],
                    "values": mixture_values
                },
                {
                    "name": "enrollment",
                    "element_type": "float32",
                    "shape": [1, 8],
                    "values": enrollment_values
                }
            ],
            "outputs": [
                {
                    "name": "extracted",
                    "element_type": "float32",
                    "shape": [1, 4],
                    "values": [0.0, 0.1, -0.1, 0.2]
                },
                {
                    "name": "target_presence_probabilities",
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
    let axes = json!([
        { "name": "batch", "kind": "batch", "fixed": 1 },
        { "name": "samples", "kind": "sample", "fixed": null }
    ]);
    let manifest = json!({
        "schema": denoize::RUNTIME_MODEL_PACKAGE_SCHEMA_V2,
        "format_version": denoize::RUNTIME_MODEL_PACKAGE_VERSION_V2,
        "package_id": format!("denoize.test.target-speaker.{id}"),
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
            "channels": { "policy": "independent-mono", "roles": [], "geometry": null }
        },
        "tensors": {
            "inputs": [
                { "name": "mixture", "role": "audio", "element_type": "float32", "axes": axes, "optional": false, "state_id": null },
                { "name": "enrollment", "role": "enrollment", "element_type": "float32", "axes": axes, "optional": false, "state_id": null }
            ],
            "outputs": [
                { "name": "extracted", "role": "audio", "element_type": "float32", "axes": axes, "optional": false, "state_id": null },
                {
                    "name": "target_presence_probabilities",
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
            "frame_samples": 1,
            "hop_samples": 1,
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
            "source_repository": "https://example.invalid/target-speaker",
            "source_revision": "0123456789abcdef",
            "source_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "source_license_spdx": "MIT",
            "checkpoint_source": "https://example.invalid/target-speaker.ckpt",
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
        Some("denoize target-speaker CLI fixture"),
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
    let opened = denoize::RuntimeModelPackage::open(&package, &package_key).unwrap();
    let payload = promotion_payload(opened.package_sha256());
    let (secret, public) = denoize::generate_receipt_keypair().unwrap();
    let signed = denoize::sign_target_speaker_promotion_evidence(payload, &secret).unwrap();
    let evidence = root.join("evidence.json");
    let evidence_key = root.join("evidence-public.json");
    std::fs::write(&evidence, serde_json::to_vec_pretty(&signed).unwrap()).unwrap();
    std::fs::write(&evidence_key, serde_json::to_vec_pretty(&public).unwrap()).unwrap();
    Fixture {
        package,
        package_key,
        evidence,
        evidence_key,
    }
}

fn build_causal_fixture(directory: &Path, id: &str) -> CausalFixture {
    let root = directory.join(id);
    let components = root.join("components");
    std::fs::create_dir_all(&components).unwrap();
    let mut model = Vec::new();
    causal_target_speaker_model().encode(&mut model).unwrap();
    let license = b"MIT License\ncausal fixture only\n".to_vec();
    let provenance = br#"{"schema":"denoize-test-provenance-v1"}"#.to_vec();
    let vector = |id: &str, mixture: Vec<f64>, state: Vec<f64>| {
        let output_mixture = mixture.clone();
        let output_state = state.clone();
        json!({
            "id": id,
            "inputs": [
                { "name": "mixture", "element_type": "float32", "shape": [1, 4], "values": mixture },
                { "name": "enrollment", "element_type": "float32", "shape": [1, 8], "values": vec![0.0; 8] },
                { "name": "state_in", "element_type": "float32", "shape": [1, 2], "values": state }
            ],
            "outputs": [
                { "name": "extracted", "element_type": "float32", "shape": [1, 4], "values": output_mixture },
                { "name": "target_presence_probabilities", "element_type": "float32", "shape": [1, 3], "values": [0.0, 0.0, 1.0] },
                { "name": "state_out", "element_type": "float32", "shape": [1, 2], "values": output_state }
            ],
            "tolerance": { "absolute": 0.000001, "relative": 0.000001 }
        })
    };
    let vectors = serde_json::to_vec(&json!({
        "schema": "denoize-runtime-model-numerical-vectors-v1",
        "profile_id": "fp32",
        "cases": [
            vector("causal-reset", vec![0.0, 0.1, -0.1, 0.2], vec![0.0, 0.0]),
            vector("causal-recurrent", vec![0.2, -0.2, 0.3, -0.3], vec![1.0, -1.0]),
            vector("causal-flush", vec![0.0; 4], vec![0.5, -0.5])
        ]
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
    let frame_axes = json!([
        { "name": "batch", "kind": "batch", "fixed": 1 },
        { "name": "samples", "kind": "sample", "fixed": 4 }
    ]);
    let enrollment_axes = json!([
        { "name": "batch", "kind": "batch", "fixed": 1 },
        { "name": "samples", "kind": "sample", "fixed": null }
    ]);
    let state_axes = json!([
        { "name": "batch", "kind": "batch", "fixed": 1 },
        { "name": "memory", "kind": "state", "fixed": 2 }
    ]);
    let manifest = json!({
        "schema": denoize::RUNTIME_MODEL_PACKAGE_SCHEMA_V2,
        "format_version": denoize::RUNTIME_MODEL_PACKAGE_VERSION_V2,
        "package_id": format!("denoize.test.causal-target-speaker.{id}"),
        "package_revision": "1",
        "signing_key_id": key_id,
        "runtime": { "kind": "onnx-audio-graph-v2", "sample_rate_hz": MODEL_RATE, "mode": "streaming" },
        "frontend": {
            "normalization": "pcm-f32-minus-one-to-one-v1",
            "resampling": "bandlimited-waveform-v1",
            "duration": "preserve-input-frames-v1",
            "channels": { "policy": "independent-mono", "roles": [], "geometry": null }
        },
        "tensors": {
            "inputs": [
                { "name": "mixture", "role": "audio", "element_type": "float32", "axes": frame_axes, "optional": false, "state_id": null },
                { "name": "enrollment", "role": "enrollment", "element_type": "float32", "axes": enrollment_axes, "optional": false, "state_id": null },
                { "name": "state_in", "role": "state", "element_type": "float32", "axes": state_axes, "optional": false, "state_id": "memory" }
            ],
            "outputs": [
                { "name": "extracted", "role": "audio", "element_type": "float32", "axes": frame_axes, "optional": false, "state_id": null },
                {
                    "name": "target_presence_probabilities",
                    "role": "diagnostic",
                    "element_type": "float32",
                    "axes": [
                        { "name": "batch", "kind": "batch", "fixed": 1 },
                        { "name": "classes", "kind": "feature", "fixed": 3 }
                    ],
                    "optional": false,
                    "state_id": null
                },
                { "name": "state_out", "role": "state", "element_type": "float32", "axes": state_axes, "optional": false, "state_id": "memory" }
            ]
        },
        "state_pairs": [{ "id": "memory", "input": "state_in", "output": "state_out", "initialization": "zeros" }],
        "latency": {
            "frame_samples": 4,
            "hop_samples": 4,
            "left_context_samples": 0,
            "right_context_samples": 0,
            "lookahead_samples": 0,
            "algorithmic_latency_samples": 4,
            "flush_samples": 4
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
            "source_repository": "https://example.invalid/causal-target-speaker",
            "source_revision": "0123456789abcdef",
            "source_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "source_license_spdx": "MIT",
            "checkpoint_source": "https://example.invalid/causal-target-speaker.ckpt",
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
        Some("denoize causal target-speaker CLI fixture"),
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
    let opened = denoize::RuntimeModelPackage::open(&package, &package_key).unwrap();
    let offline_payload = promotion_payload(opened.package_sha256());
    let (offline_secret, offline_public) = denoize::generate_receipt_keypair().unwrap();
    let offline_signed =
        denoize::sign_target_speaker_promotion_evidence(offline_payload.clone(), &offline_secret)
            .unwrap();
    let (causal_secret, causal_public) = denoize::generate_receipt_keypair().unwrap();
    let causal_signed = denoize::sign_causal_target_speaker_promotion_evidence(
        causal_promotion_payload(&offline_payload),
        &causal_secret,
    )
    .unwrap();
    let offline_evidence = root.join("offline-evidence.json");
    let offline_evidence_key = root.join("offline-evidence-public.json");
    let causal_evidence = root.join("causal-evidence.json");
    let causal_evidence_key = root.join("causal-evidence-public.json");
    std::fs::write(
        &offline_evidence,
        serde_json::to_vec_pretty(&offline_signed).unwrap(),
    )
    .unwrap();
    std::fs::write(
        &offline_evidence_key,
        serde_json::to_vec_pretty(&offline_public).unwrap(),
    )
    .unwrap();
    std::fs::write(
        &causal_evidence,
        serde_json::to_vec_pretty(&causal_signed).unwrap(),
    )
    .unwrap();
    std::fs::write(
        &causal_evidence_key,
        serde_json::to_vec_pretty(&causal_public).unwrap(),
    )
    .unwrap();
    CausalFixture {
        package,
        package_key,
        offline_evidence,
        offline_evidence_key,
        causal_evidence,
        causal_evidence_key,
    }
}

#[test]
fn present_publishes_but_uncertain_withholds_and_reports_no_enrollment_digest() {
    let directory = tempfile::tempdir().unwrap();
    let mixture = directory.path().join("mixture.wav");
    let enrollment = directory.path().join("enrollment.wav");
    write_wav(&mixture, 1_600);
    write_wav(&enrollment, 8_000);

    let present = build_fixture(directory.path(), "present", [0.0, 0.0, 1.0]);
    let present_output = directory.path().join("present.wav");
    let present_report = directory.path().join("present-report.json");
    let result = run(&[
        "target-speaker",
        mixture.to_str().unwrap(),
        enrollment.to_str().unwrap(),
        present_output.to_str().unwrap(),
        "--model-package",
        present.package.to_str().unwrap(),
        "--model-package-key",
        present.package_key.to_str().unwrap(),
        "--promotion-evidence",
        present.evidence.to_str().unwrap(),
        "--promotion-evidence-key",
        present.evidence_key.to_str().unwrap(),
        "--report",
        present_report.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(present_output.is_file());
    let report: Value = serde_json::from_slice(&std::fs::read(&present_report).unwrap()).unwrap();
    assert_eq!(report["decision"], "accepted-present");
    assert_eq!(report["output_published"], true);
    assert_eq!(report["enrollment"]["raw_audio_retained"], false);
    assert_eq!(report["enrollment"]["embedding_retained"], false);
    assert_eq!(report["enrollment"]["digest_recorded"], false);
    let report_text = serde_json::to_string(&report).unwrap();
    assert!(!report_text.contains(directory.path().to_str().unwrap()));
    assert!(!report_text.contains("enrollment_pcm_sha256"));

    let uncertain = build_fixture(directory.path(), "uncertain", [0.0, 1.0, 0.0]);
    let withheld_output = directory.path().join("withheld.wav");
    let withheld_report = directory.path().join("withheld-report.json");
    let result = run(&[
        "target-speaker",
        mixture.to_str().unwrap(),
        enrollment.to_str().unwrap(),
        withheld_output.to_str().unwrap(),
        "--model-package",
        uncertain.package.to_str().unwrap(),
        "--model-package-key",
        uncertain.package_key.to_str().unwrap(),
        "--promotion-evidence",
        uncertain.evidence.to_str().unwrap(),
        "--promotion-evidence-key",
        uncertain.evidence_key.to_str().unwrap(),
        "--report",
        withheld_report.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!withheld_output.exists());
    let report: Value = serde_json::from_slice(&std::fs::read(&withheld_report).unwrap()).unwrap();
    assert_eq!(report["decision"], "withheld-uncertain");
    assert_eq!(report["output_published"], false);
    assert!(report["candidate_pcm_sha256"].is_null());
    assert!(report["output_pcm_sha256"].is_null());
}

#[test]
fn causal_cli_authenticates_both_layers_and_preserves_exact_geometry() {
    let directory = tempfile::tempdir().unwrap();
    let mixture = directory.path().join("causal-mixture.wav");
    let enrollment = directory.path().join("causal-enrollment.wav");
    let output = directory.path().join("causal-output.wav");
    let report_path = directory.path().join("causal-report.json");
    write_wav(&mixture, 1_603);
    write_wav(&enrollment, 8_000);
    let fixture = build_causal_fixture(directory.path(), "causal");

    let result = run(&[
        "target-speaker",
        "causal",
        mixture.to_str().unwrap(),
        enrollment.to_str().unwrap(),
        output.to_str().unwrap(),
        "--model-package",
        fixture.package.to_str().unwrap(),
        "--model-package-key",
        fixture.package_key.to_str().unwrap(),
        "--offline-promotion-evidence",
        fixture.offline_evidence.to_str().unwrap(),
        "--offline-promotion-evidence-key",
        fixture.offline_evidence_key.to_str().unwrap(),
        "--causal-promotion-evidence",
        fixture.causal_evidence.to_str().unwrap(),
        "--causal-promotion-evidence-key",
        fixture.causal_evidence_key.to_str().unwrap(),
        "--present-hold-blocks",
        "1",
        "--report",
        report_path.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let rendered = denoize::read_audio(&output).unwrap();
    assert_eq!(rendered.sample_rate, MODEL_RATE);
    assert_eq!(rendered.channels(), 1);
    assert_eq!(rendered.frames(), 1_603);
    let report: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(
        report["schema"],
        denoize::CAUSAL_TARGET_SPEAKER_REPORT_SCHEMA
    );
    assert_eq!(report["source_frames"], 1_603);
    assert_eq!(report["output_frames"], 1_603);
    assert_eq!(report["algorithmic_latency_samples"], 4);
    assert_eq!(report["flush_samples"], 4);
    assert!(
        report["decision_counts"]["published_present_blocks"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(report["enrollment"]["raw_audio_retained"], false);
    assert_eq!(report["enrollment"]["embedding_retained"], false);
    assert_eq!(report["enrollment"]["digest_recorded"], false);
    assert_eq!(report["runtime_speaker_identity_verified"], false);
    assert_eq!(report["interferer_leakage_measured_at_runtime"], false);
    let report_text = serde_json::to_string(&report).unwrap();
    assert!(!report_text.contains(directory.path().to_str().unwrap()));
    assert!(!report_text.contains("enrollment_pcm_sha256"));
}

#[test]
fn modified_evidence_fails_before_audio_decode_or_publication() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = build_fixture(directory.path(), "tamper", [0.0, 0.0, 1.0]);
    let mixture = directory.path().join("not-audio.wav");
    let enrollment = directory.path().join("enrollment.wav");
    let output = directory.path().join("blocked.wav");
    std::fs::write(&mixture, b"must not be decoded").unwrap();
    write_wav(&enrollment, 8_000);
    let mut evidence: Value =
        serde_json::from_slice(&std::fs::read(&fixture.evidence).unwrap()).unwrap();
    evidence["payload"]["target_speaker_count"] = json!(101);
    let tampered = directory.path().join("tampered-evidence.json");
    std::fs::write(&tampered, serde_json::to_vec(&evidence).unwrap()).unwrap();
    let result = run(&[
        "target-speaker",
        mixture.to_str().unwrap(),
        enrollment.to_str().unwrap(),
        output.to_str().unwrap(),
        "--model-package",
        fixture.package.to_str().unwrap(),
        "--model-package-key",
        fixture.package_key.to_str().unwrap(),
        "--promotion-evidence",
        tampered.to_str().unwrap(),
        "--promotion-evidence-key",
        fixture.evidence_key.to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("signature verification failed"), "{stderr}");
    assert!(!stderr.contains("decode"), "{stderr}");
    assert!(!output.exists());
}

fn promotion_payload(package_sha256: &str) -> denoize::TargetSpeakerPromotionEvidencePayload {
    let strata = REQUIRED_STRATA
        .iter()
        .map(|(id, kind)| denoize::TargetSpeakerStratumEvidence {
            id: (*id).into(),
            kind: *kind,
            cases: 10,
            metrics: metrics(*kind),
        })
        .collect();
    denoize::TargetSpeakerPromotionEvidencePayload {
        completed_at_unix_seconds: 1_800_000_000,
        model_package_sha256: package_sha256.into(),
        source_revision: "0123456789abcdef".into(),
        source_sha256: "1".repeat(64),
        checkpoint_sha256: "2".repeat(64),
        corpus_manifest_sha256: "4".repeat(64),
        evaluation_result_sha256: "5".repeat(64),
        real_t_result_sha256: "6".repeat(64),
        ts_superb_result_sha256: "7".repeat(64),
        strata,
        target_speaker_count: 100,
        interferer_speaker_count: 100,
        language_count: 2,
        presence_expected_calibration_error: 0.05,
        presence_expected_calibration_error_limit: 0.05,
        minimum_listeners: 20,
        listener_count: 20,
        listener_preference: 0.5,
        listener_preference_limit: 0.5,
        accepted: true,
    }
}

fn causal_promotion_payload(
    offline: &denoize::TargetSpeakerPromotionEvidencePayload,
) -> denoize::CausalTargetSpeakerPromotionEvidencePayload {
    let strata = offline
        .strata
        .iter()
        .map(|stratum| denoize::CausalTargetSpeakerStratumEvidence {
            id: stratum.id.clone(),
            kind: stratum.kind,
            offline_cases: stratum.cases,
            causal_cases: stratum.cases,
            metrics: stratum
                .metrics
                .iter()
                .map(|metric| denoize::CausalTargetSpeakerMetricEvidence {
                    metric: metric.metric.clone(),
                    operator: metric.operator,
                    offline_value: metric.value,
                    causal_value: metric.value,
                    hard_limit: metric.limit,
                    maximum_regression: maximum_causal_regression(&metric.metric),
                    passed: true,
                })
                .collect(),
        })
        .collect();
    denoize::CausalTargetSpeakerPromotionEvidencePayload {
        completed_at_unix_seconds: 1_800_000_001,
        model_package_sha256: offline.model_package_sha256.clone(),
        source_revision: offline.source_revision.clone(),
        source_sha256: offline.source_sha256.clone(),
        checkpoint_sha256: offline.checkpoint_sha256.clone(),
        offline_evaluation_result_sha256: offline.evaluation_result_sha256.clone(),
        causal_evaluation_result_sha256: "8".repeat(64),
        state_reset_flush_result_sha256: "9".repeat(64),
        latency_result_sha256: "a".repeat(64),
        realtime_callback_result_sha256: "b".repeat(64),
        transition_result_sha256: "c".repeat(64),
        strata,
        model_sample_rate_hz: MODEL_RATE,
        frame_samples: 4,
        algorithmic_latency_samples: 4,
        flush_samples: 4,
        perturbation_latency_cases: 100,
        effective_latency_milliseconds: 0.25,
        effective_latency_limit_milliseconds: 100.0,
        realtime: denoize::CausalTargetSpeakerRealtimeAudit {
            paced_blocks: 10_000,
            deadline_misses: 0,
            overload_blocks: 0,
            queue_capacity_blocks: 16,
            maximum_queue_depth_blocks: 15,
            callback_allocations: 0,
            callback_locks: 0,
            callback_waits: 0,
            callback_file_io_operations: 0,
            callback_network_operations: 0,
            callback_log_operations: 0,
            callback_inference_calls: 0,
        },
        transitions: denoize::CausalTargetSpeakerTransitionAudit {
            absent_to_present_cases: 100,
            present_to_absent_cases: 100,
            uncertain_transition_cases: 100,
            enrollment_mismatch_cases: 100,
            reference_loss_cases: 100,
            late_results_injected: 100,
            late_results_discarded: 100,
            stale_generation_results_injected: 100,
            stale_generation_results_discarded: 100,
            false_attribution_publications: 0,
        },
        accepted: true,
    }
}

fn maximum_causal_regression(metric: &str) -> f64 {
    match metric {
        "content.target-word-error-rate" => 0.02,
        "extraction.si-sdr-improvement-db" => 0.5,
        "interferer.speaker-similarity" | "speaker.target-similarity" => 0.02,
        "interferer.word-leakage-rate" => 0.005,
        "perceptual.dnsmos-p808" => 0.1,
        "presence.recall" => 0.02,
        "output.rms-dbfs" => 3.0,
        "presence.false-positive-rate" => 0.005,
        "output.duration-error-frames" | "output.non-finite-samples" => 0.0,
        unknown => panic!("missing causal regression fixture for {unknown}"),
    }
}

fn metrics(kind: denoize::TargetSpeakerStratumKind) -> Vec<denoize::TargetSpeakerMetricOutcome> {
    let values: &[(&str, denoize::TargetSpeakerMetricOperator, f64)] = match kind {
        denoize::TargetSpeakerStratumKind::TargetPresent => &[
            (
                "content.target-word-error-rate",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.35,
            ),
            (
                "extraction.si-sdr-improvement-db",
                denoize::TargetSpeakerMetricOperator::GreaterOrEqual,
                3.0,
            ),
            (
                "interferer.speaker-similarity",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.30,
            ),
            (
                "interferer.word-leakage-rate",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.02,
            ),
            (
                "output.duration-error-frames",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.0,
            ),
            (
                "output.non-finite-samples",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.0,
            ),
            (
                "perceptual.dnsmos-p808",
                denoize::TargetSpeakerMetricOperator::GreaterOrEqual,
                3.0,
            ),
            (
                "presence.recall",
                denoize::TargetSpeakerMetricOperator::GreaterOrEqual,
                0.95,
            ),
            (
                "speaker.target-similarity",
                denoize::TargetSpeakerMetricOperator::GreaterOrEqual,
                0.70,
            ),
        ],
        denoize::TargetSpeakerStratumKind::TargetAbsent => &[
            (
                "interferer.speaker-similarity",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.30,
            ),
            (
                "interferer.word-leakage-rate",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.01,
            ),
            (
                "output.duration-error-frames",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.0,
            ),
            (
                "output.non-finite-samples",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.0,
            ),
            (
                "output.rms-dbfs",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                -60.0,
            ),
            (
                "presence.false-positive-rate",
                denoize::TargetSpeakerMetricOperator::LessOrEqual,
                0.01,
            ),
        ],
    };
    values
        .iter()
        .map(
            |(name, operator, limit)| denoize::TargetSpeakerMetricOutcome {
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

fn write_wav(path: &Path, frames: usize) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: MODEL_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in 0..frames {
        let value = (0.15
            * (std::f64::consts::TAU * 220.0 * frame as f64 / MODEL_RATE as f64).sin()
            * f64::from(i16::MAX))
        .round() as i16;
        writer.write_sample(value).unwrap();
    }
    writer.finalize().unwrap();
}

fn target_speaker_model(probabilities: [f32; 3]) -> ModelProto {
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
    let waveform = || vec![dimension_value(1), dimension_parameter("samples")];
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "denoize-test".into(),
        graph: Some(GraphProto {
            name: "target-speaker-identity".into(),
            node: vec![
                NodeProto {
                    input: vec!["mixture".into()],
                    output: vec!["extracted".into()],
                    name: "identity".into(),
                    op_type: "Identity".into(),
                    ..Default::default()
                },
                NodeProto {
                    output: vec!["target_presence_probabilities".into()],
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
                value_info("mixture", waveform()),
                value_info(
                    "enrollment",
                    vec![
                        dimension_value(1),
                        dimension_parameter("enrollment_samples"),
                    ],
                ),
            ],
            output: vec![
                value_info("extracted", waveform()),
                value_info(
                    "target_presence_probabilities",
                    vec![dimension_value(1), dimension_value(3)],
                ),
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn causal_target_speaker_model() -> ModelProto {
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
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "denoize-test".into(),
        graph: Some(GraphProto {
            name: "causal-target-speaker-identity".into(),
            node: vec![
                NodeProto {
                    input: vec!["mixture".into()],
                    output: vec!["extracted".into()],
                    name: "audio-identity".into(),
                    op_type: "Identity".into(),
                    ..Default::default()
                },
                NodeProto {
                    input: vec!["state_in".into()],
                    output: vec!["state_out".into()],
                    name: "state-identity".into(),
                    op_type: "Identity".into(),
                    ..Default::default()
                },
                NodeProto {
                    output: vec!["target_presence_probabilities".into()],
                    name: "presence".into(),
                    op_type: "Constant".into(),
                    attribute: vec![AttributeProto {
                        name: "value".into(),
                        r#type: attribute_proto::AttributeType::Tensor as i32,
                        t: Some(TensorProto {
                            dims: vec![1, 3],
                            data_type: tensor_proto::DataType::Float as i32,
                            float_data: vec![0.0, 0.0, 1.0],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            input: vec![
                value_info("mixture", vec![dimension_value(1), dimension_value(4)]),
                value_info(
                    "enrollment",
                    vec![
                        dimension_value(1),
                        dimension_parameter("enrollment_samples"),
                    ],
                ),
                value_info("state_in", vec![dimension_value(1), dimension_value(2)]),
            ],
            output: vec![
                value_info("extracted", vec![dimension_value(1), dimension_value(4)]),
                value_info(
                    "target_presence_probabilities",
                    vec![dimension_value(1), dimension_value(3)],
                ),
                value_info("state_out", vec![dimension_value(1), dimension_value(2)]),
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn dimension_value(value: i64) -> tensor_shape_proto::Dimension {
    tensor_shape_proto::Dimension {
        value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
        denotation: String::new(),
    }
}

fn dimension_parameter(name: &str) -> tensor_shape_proto::Dimension {
    tensor_shape_proto::Dimension {
        value: Some(tensor_shape_proto::dimension::Value::DimParam(name.into())),
        denotation: String::new(),
    }
}
