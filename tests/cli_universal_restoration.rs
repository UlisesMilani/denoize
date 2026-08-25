#![cfg(feature = "bsrnn")]

use prost::Message;
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tract_onnx::pb::{
    tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
};

const MODEL_RATE: u32 = 48_000;
const FFT_SIZE: usize = 960;
const HOP_SIZE: usize = 480;
const BINS: usize = FFT_SIZE / 2 + 1;

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denoize"))
        .args(args)
        .output()
        .unwrap()
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_contract(filename: &str, bytes: &[u8]) -> Value {
    json!({
        "filename": filename,
        "size_bytes": bytes.len(),
        "sha256": sha256(bytes),
    })
}

fn spectral_axes() -> Value {
    json!([
        { "name": "batch", "kind": "batch", "fixed": 1 },
        { "name": "frames", "kind": "frame", "fixed": null },
        { "name": "frequency", "kind": "frequency", "fixed": BINS },
        { "name": "complex", "kind": "coordinate", "fixed": 2 }
    ])
}

fn build_signed_identity_package(directory: &Path) -> (PathBuf, PathBuf) {
    let components = directory.join("components");
    std::fs::create_dir(&components).unwrap();

    let mut model = Vec::new();
    spectral_identity_model().encode(&mut model).unwrap();
    let license = b"MIT License\nfixture only\n".to_vec();
    let provenance = br#"{"schema":"denoize-test-provenance-v1"}"#.to_vec();
    let values = vec![0.0_f64; BINS * 2];
    let vectors = serde_json::to_vec(&json!({
        "schema": "denoize-runtime-model-numerical-vectors-v1",
        "profile_id": "fp32",
        "cases": [{
            "id": "spectral-identity",
            "inputs": [{
                "name": "spectrum",
                "element_type": "float32",
                "shape": [1, 1, BINS, 2],
                "values": values,
            }],
            "outputs": [{
                "name": "enhanced_spectrum",
                "element_type": "float32",
                "shape": [1, 1, BINS, 2],
                "values": vec![0.0_f64; BINS * 2],
            }],
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
    let manifest = json!({
        "schema": denoize::RUNTIME_MODEL_PACKAGE_SCHEMA_V2,
        "format_version": denoize::RUNTIME_MODEL_PACKAGE_VERSION_V2,
        "package_id": "denoize.test.universal-bsrnn",
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
                "policy": "independent-mono",
                "roles": [],
                "geometry": null
            }
        },
        "tensors": {
            "inputs": [{
                "name": "spectrum",
                "role": "audio",
                "element_type": "float32",
                "axes": spectral_axes(),
                "optional": false,
                "state_id": null
            }],
            "outputs": [{
                "name": "enhanced_spectrum",
                "role": "audio",
                "element_type": "float32",
                "axes": spectral_axes(),
                "optional": false,
                "state_id": null
            }]
        },
        "state_pairs": [],
        "latency": {
            "frame_samples": FFT_SIZE,
            "hop_samples": HOP_SIZE,
            "left_context_samples": FFT_SIZE / 2,
            "right_context_samples": FFT_SIZE / 2,
            "lookahead_samples": FFT_SIZE / 2,
            "algorithmic_latency_samples": FFT_SIZE / 2,
            "flush_samples": FFT_SIZE / 2
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
            "source_repository": "https://example.invalid/urgent",
            "source_revision": "b1dc3ad1e86419ff0bd666f455bda7936bff0e9a",
            "source_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "source_license_spdx": "Apache-2.0",
            "checkpoint_source": "https://example.invalid/bsrnn.ckpt",
            "checkpoint_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "checkpoint_license_spdx": "MIT",
            "conversion_tool": "denoize-test-fixture",
            "conversion_revision": "1",
            "training_datasets": [{
                "id": "synthetic",
                "source": "urn:denoize:test:synthetic",
                "revision": "1",
                "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
                "license_spdx": "CC0-1.0"
            }]
        }
    });
    let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
    let signature = minisign::sign(
        None,
        &sk,
        Cursor::new(&manifest_bytes),
        Some("denoize universal CLI fixture"),
        Some("untrusted comment: denoize test fixture"),
    )
    .unwrap()
    .into_string();

    let manifest_path = directory.join("manifest.json");
    let signature_path = directory.join("manifest.json.sig");
    let public_key_path = directory.join("model.pub");
    std::fs::write(&manifest_path, manifest_bytes).unwrap();
    std::fs::write(&signature_path, signature).unwrap();
    std::fs::write(&public_key_path, public_key).unwrap();
    let package_path = directory.join("model.dmp");
    denoize::build_runtime_model_package_v2(
        &package_path,
        manifest_path,
        signature_path,
        &public_key_path,
        components,
    )
    .unwrap();
    (package_path, public_key_path)
}

fn write_degraded_fixture(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: MODEL_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in 0..4_800usize {
        let value = (2.4
            * (std::f64::consts::TAU * 440.0 * frame as f64 / MODEL_RATE as f64).sin())
        .clamp(-1.0, 1.0);
        let quantized = if value >= 1.0 {
            i16::MAX
        } else if value <= -1.0 {
            i16::MIN
        } else {
            (value * f64::from(i16::MAX)).round() as i16
        };
        writer.write_sample(quantized).unwrap();
    }
    writer.finalize().unwrap();
}

fn mask_has_exact_coverage(mask: &Value) -> bool {
    let channels = mask["channels"].as_u64().unwrap() as usize;
    let frames = mask["frames"].as_u64().unwrap();
    let mut cursors = vec![0_u64; channels];
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
fn signed_package_runs_path_free_and_tampering_fails_closed() {
    let directory = tempfile::tempdir().unwrap();
    let (package, key) = build_signed_identity_package(directory.path());
    let input = directory.path().join("degraded.wav");
    let output = directory.path().join("restored.wav");
    let report = directory.path().join("report.json");
    let mask = directory.path().join("mask.json");
    write_degraded_fixture(&input);

    let result = run(&[
        "universal",
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--model-package",
        package.to_str().unwrap(),
        "--model-package-key",
        key.to_str().unwrap(),
        "--minimum-degradation-score",
        "0",
        "--report",
        report.to_str().unwrap(),
        "--mask",
        mask.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stderr.is_empty());
    assert_eq!(denoize::read_audio(&output).unwrap().frames(), 4_800);
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(!stdout.contains(directory.path().to_str().unwrap()));
    let stdout_report: Value = serde_json::from_str(&stdout).unwrap();
    let file_report: Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
    let file_mask: Value = serde_json::from_slice(&std::fs::read(&mask).unwrap()).unwrap();
    assert_eq!(stdout_report, file_report);
    assert_eq!(
        file_report["schema"],
        "denoize-universal-restoration-report-v1"
    );
    assert_eq!(file_report["network_accessed"], false);
    assert_eq!(file_report["model_invoked"], true);
    assert_eq!(file_report["deterministic"], true);
    assert_eq!(file_report["semantic_fidelity_assessed"], false);
    assert_eq!(file_report["speaker_identity_assessed"], false);
    assert_eq!(file_mask["schema"], "denoize-universal-restoration-mask-v1");
    assert!(mask_has_exact_coverage(&file_mask));

    let before = std::fs::read(&output).unwrap();
    let no_clobber = run(&[
        "universal",
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        "--model-package",
        package.to_str().unwrap(),
        "--model-package-key",
        key.to_str().unwrap(),
    ]);
    assert!(!no_clobber.status.success());
    assert!(String::from_utf8_lossy(&no_clobber.stderr).contains("exists"));
    assert_eq!(std::fs::read(&output).unwrap(), before);

    let collision_output = directory.path().join("collision.wav");
    let collision = run(&[
        "universal",
        package.to_str().unwrap(),
        collision_output.to_str().unwrap(),
        "--model-package",
        package.to_str().unwrap(),
        "--model-package-key",
        key.to_str().unwrap(),
    ]);
    assert!(!collision.status.success());
    assert!(String::from_utf8_lossy(&collision.stderr).contains("distinct source files"));
    assert!(!collision_output.exists());

    let tampered_package = directory.path().join("tampered.dmp");
    let mut tampered = std::fs::read(&package).unwrap();
    *tampered.last_mut().unwrap() ^= 0x01;
    std::fs::write(&tampered_package, tampered).unwrap();
    let invalid_audio = directory.path().join("invalid-audio.wav");
    std::fs::write(&invalid_audio, b"this must not be decoded").unwrap();
    let blocked_output = directory.path().join("blocked.wav");
    let blocked_report = directory.path().join("blocked-report.json");
    let blocked_mask = directory.path().join("blocked-mask.json");
    let blocked = run(&[
        "universal",
        invalid_audio.to_str().unwrap(),
        blocked_output.to_str().unwrap(),
        "--model-package",
        tampered_package.to_str().unwrap(),
        "--model-package-key",
        key.to_str().unwrap(),
        "--report",
        blocked_report.to_str().unwrap(),
        "--mask",
        blocked_mask.to_str().unwrap(),
    ]);
    assert!(!blocked.status.success());
    let blocked_stderr = String::from_utf8_lossy(&blocked.stderr);
    assert!(blocked_stderr.contains("runtime model"), "{blocked_stderr}");
    assert!(!blocked_stderr.contains("decode"), "{blocked_stderr}");
    assert!(!blocked_output.exists());
    assert!(!blocked_report.exists());
    assert!(!blocked_mask.exists());
}

fn spectral_identity_model() -> ModelProto {
    let value_info = |name: &str| ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        dimension_value(1),
                        dimension_parameter("frames"),
                        dimension_value(BINS as i64),
                        dimension_value(2),
                    ],
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
        producer_name: "denoize-test".into(),
        graph: Some(GraphProto {
            name: "bsrnn-spectral-identity".into(),
            node: vec![NodeProto {
                input: vec!["spectrum".into()],
                output: vec!["enhanced_spectrum".into()],
                name: "identity".into(),
                op_type: "Identity".into(),
                ..Default::default()
            }],
            input: vec![value_info("spectrum")],
            output: vec![value_info("enhanced_spectrum")],
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
