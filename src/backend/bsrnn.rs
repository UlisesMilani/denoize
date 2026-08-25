//! ESPnet BSRNN spectral adapter.
//!
//! The converted ONNX graph receives a real/imaginary spectrum shaped
//! `[1, frames, 481, 2]`. Rust reproduces ESPnet's variance normalization,
//! centered periodic-Hann 960-point STFT, whole-utterance inference, inverse
//! STFT, sample-rate conversion, and exact duration restoration. Raw ONNX files
//! remain supported for compatibility. Universal-restoration callers should
//! use a signed runtime-model package v2 so graph semantics, provenance,
//! resources, and numerical vectors are authenticated before source audio.

use super::tract_runtime::SharedRunnable;
use super::OnnxModelConfig;
use crate::AcceleratorRuntime;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::sync::{Arc, Mutex};
use tract_onnx::prelude::*;

const MODEL_RATE: u32 = 48_000;
const FFT_SIZE: usize = 960;
const HOP_SIZE: usize = 480;
const BINS: usize = FFT_SIZE / 2 + 1;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<Vec<f64>>, String> {
    BsrnnModel::load(config, AcceleratorRuntime::Cpu)?.process(channels, input_sample_rate)
}

struct CompiledBsrnnModel {
    frames: usize,
    model: SharedRunnable,
}

pub(crate) struct BsrnnModel {
    template: InferenceModel,
    runtime: AcceleratorRuntime,
    compiled: Mutex<Option<CompiledBsrnnModel>>,
}

impl BsrnnModel {
    pub(crate) fn load(
        config: &OnnxModelConfig,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        if config.sample_rate != MODEL_RATE {
            return Err(format!(
                "BSRNN expects a {MODEL_RATE} Hz model, got {} Hz",
                config.sample_rate
            ));
        }
        if !config.path.is_file() {
            return Err(format!(
                "BSRNN ONNX model does not exist or is not a file: {}",
                config.path.display()
            ));
        }
        Self::from_template(load_template(config)?, runtime)
    }

    pub(crate) fn load_runtime_package(
        package: &crate::RuntimeModelPackage,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let manifest = package.manifest_v2().ok_or(
            "BSRNN runtime packages must use format v2 with named spectral tensors and numerical vectors",
        )?;
        validate_runtime_package_contract(manifest)?;
        let profile = package
            .precision_profile_for(runtime)?
            .expect("a v2 package always selects a precision profile");
        let mut reader = package.open_model_reader_for(runtime)?;
        let template = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|error| {
                format!(
                    "failed to load BSRNN ONNX model from authenticated package {}: {error:#}",
                    package.package_path().display()
                )
            })?;
        reader.finish().map_err(|error| {
            format!(
                "failed to authenticate BSRNN model bytes from package {}: {error}",
                package.package_path().display()
            )
        })?;
        let mut inspected = template.clone();
        if inspected.analyse(false).is_ok() {
            super::onnx::validate_v2_graph_contract(&inspected, manifest)?;
        } else {
            super::onnx::validate_v2_graph_contract(&template, manifest)?;
        }
        let vectors = package
            .numerical_vectors_for(runtime)?
            .expect("a v2 precision profile always carries numerical vectors");
        super::onnx::validate_v2_numerical_vectors(
            &template, manifest, profile, &vectors, runtime,
        )?;
        Self::from_template(template, runtime)
    }

    fn from_template(
        template: InferenceModel,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        validate_template(&template)?;
        Ok(Self {
            template,
            runtime,
            compiled: Mutex::new(None),
        })
    }

    pub(crate) fn process(
        &self,
        channels: &[Vec<f64>],
        input_sample_rate: u32,
    ) -> Result<Vec<Vec<f64>>, String> {
        if channels.is_empty() {
            return Ok(Vec::new());
        }
        let model_samples =
            crate::resample::resample(&channels[0], input_sample_rate, MODEL_RATE)?.len();
        if model_samples == 0 {
            return Ok(channels.iter().map(|_| Vec::new()).collect());
        }
        let frames = model_samples / HOP_SIZE + 1;
        let model = self.compiled_model(frames)?;
        channels
            .iter()
            .map(|channel| process_channel(channel, input_sample_rate, frames, model.as_ref()))
            .collect()
    }

    fn compiled_model(&self, frames: usize) -> Result<SharedRunnable, String> {
        let mut compiled = self
            .compiled
            .lock()
            .map_err(|_| "BSRNN compiled-model cache lock was poisoned".to_string())?;
        if let Some(cached) = compiled.as_ref() {
            if cached.frames == frames {
                return Ok(Arc::clone(&cached.model));
            }
        }
        let model = compile_model(&self.template, frames, self.runtime)?;
        *compiled = Some(CompiledBsrnnModel {
            frames,
            model: Arc::clone(&model),
        });
        Ok(model)
    }
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    expected_frames: usize,
    model: &dyn tract_onnx::tract_core::runtime::Runnable,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let at_model_rate = crate::resample::resample(input, input_sample_rate, MODEL_RATE)?;
    let mean = at_model_rate.iter().sum::<f64>() / at_model_rate.len() as f64;
    let variance = if at_model_rate.len() > 1 {
        at_model_rate
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / (at_model_rate.len() - 1) as f64
    } else {
        0.0
    };
    let standard_deviation = variance.sqrt();
    if standard_deviation <= f64::EPSILON {
        return Ok(vec![0.0; input.len()]);
    }
    let normalized: Vec<f32> = at_model_rate
        .iter()
        .map(|sample| (*sample / standard_deviation) as f32)
        .collect();
    let spectrum = stft(&normalized);
    if spectrum.frames != expected_frames {
        return Err(format!(
            "BSRNN channel produced {} frames; expected {expected_frames}",
            spectrum.frames
        ));
    }
    let enhanced_spectrum = run_model(&spectrum.values, spectrum.frames, model)?;
    let reconstructed = istft(&enhanced_spectrum, spectrum.frames, normalized.len())?;
    let denormalized: Vec<f64> = reconstructed
        .iter()
        .map(|sample| *sample as f64 * standard_deviation)
        .collect();
    let mut output = crate::resample::resample(&denormalized, MODEL_RATE, input_sample_rate)?;
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

struct Spectrum {
    values: Vec<f32>,
    frames: usize,
}

fn stft(input: &[f32]) -> Spectrum {
    let pad = FFT_SIZE / 2;
    let padded: Vec<f32> = (0..input.len() + 2 * pad)
        .map(|index| input[reflect_index(index as isize - pad as isize, input.len())])
        .collect();
    let frames = 1 + (padded.len() - FFT_SIZE) / HOP_SIZE;
    let window = periodic_hann();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut values = vec![0.0; frames * BINS * 2];
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..frames {
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            buffer[index] = Complex32::new(padded[start + index] * window[index], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..BINS {
            let offset = (frame * BINS + bin) * 2;
            values[offset] = buffer[bin].re;
            values[offset + 1] = buffer[bin].im;
        }
    }
    Spectrum { values, frames }
}

fn istft(spectrum: &[f32], frames: usize, output_length: usize) -> Result<Vec<f32>, String> {
    if spectrum.len() != frames * BINS * 2 {
        return Err("BSRNN output tensor has an unexpected size".into());
    }
    let window = periodic_hann();
    let padded_length = (frames - 1) * HOP_SIZE + FFT_SIZE;
    let mut signal = vec![0.0f32; padded_length];
    let mut envelope = vec![0.0f32; padded_length];
    let mut planner = FftPlanner::new();
    let inverse = planner.plan_fft_inverse(FFT_SIZE);
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..frames {
        for bin in 0..BINS {
            let offset = (frame * BINS + bin) * 2;
            buffer[bin] = Complex32::new(spectrum[offset], spectrum[offset + 1]);
        }
        for bin in BINS..FFT_SIZE {
            buffer[bin] = buffer[FFT_SIZE - bin].conj();
        }
        inverse.process(&mut buffer);
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            signal[start + index] += buffer[index].re / FFT_SIZE as f32 * window[index];
            envelope[start + index] += window[index] * window[index];
        }
    }
    for (sample, weight) in signal.iter_mut().zip(envelope) {
        if weight > 1e-8 {
            *sample /= weight;
        }
    }
    let pad = FFT_SIZE / 2;
    // ESPnet/PyTorch iSTFT receives the original signal length explicitly. A
    // final partial hop is reconstructed from the last centered frame, so only
    // the leading center pad limits this crop.
    let available = signal.len().saturating_sub(pad);
    let copy_length = output_length.min(available);
    let mut output = signal[pad..pad + copy_length].to_vec();
    output.resize(output_length, 0.0);
    if output.iter().any(|sample| !sample.is_finite()) {
        return Err("BSRNN reconstruction produced a non-finite sample".into());
    }
    Ok(output)
}

fn load_template(config: &OnnxModelConfig) -> Result<InferenceModel, String> {
    let model = tract_onnx::onnx()
        .model_for_path(&config.path)
        .map_err(|error| model_error("load", error))?;
    validate_template(&model)?;
    Ok(model)
}

fn validate_template(model: &InferenceModel) -> Result<(), String> {
    if model
        .input_outlets()
        .map_err(|e| model_error("inspect", e))?
        .len()
        != 1
        || model
            .output_outlets()
            .map_err(|e| model_error("inspect", e))?
            .len()
            != 1
    {
        return Err("BSRNN ONNX model must have one input and one output".into());
    }
    Ok(())
}

fn validate_runtime_package_contract(
    manifest: &crate::RuntimeModelPackageManifestV2,
) -> Result<(), String> {
    let spectral_axes = |tensor: &crate::RuntimeModelTensorContractV2| {
        tensor.element_type == "float32"
            && tensor.role == "audio"
            && !tensor.optional
            && tensor.state_id.is_none()
            && tensor.axes.len() == 4
            && tensor.axes[0].kind == "batch"
            && tensor.axes[0].fixed == Some(1)
            && tensor.axes[1].kind == "frame"
            && tensor.axes[1].fixed.is_none()
            && tensor.axes[2].kind == "frequency"
            && tensor.axes[2].fixed == Some(BINS as u64)
            && tensor.axes[3].kind == "coordinate"
            && tensor.axes[3].fixed == Some(2)
    };
    let valid = manifest.runtime.sample_rate_hz == MODEL_RATE
        && manifest.runtime.mode == "finite"
        && manifest.frontend.channels.policy == "independent-mono"
        && manifest.frontend.channels.roles.is_empty()
        && manifest.frontend.channels.geometry.is_none()
        && manifest.tensors.inputs.len() == 1
        && manifest.tensors.outputs.len() == 1
        && spectral_axes(&manifest.tensors.inputs[0])
        && spectral_axes(&manifest.tensors.outputs[0])
        && manifest.tensors.inputs[0].axes == manifest.tensors.outputs[0].axes
        && manifest.state_pairs.is_empty()
        && manifest.latency.frame_samples == FFT_SIZE as u64
        && manifest.latency.hop_samples == HOP_SIZE as u64;
    if !valid {
        return Err(
            "authenticated BSRNN package must declare the finite independent-mono 48000 Hz [batch,frame,481,complex-coordinate] spectral contract with a 960-sample frame and 480-sample hop"
                .into(),
        );
    }
    Ok(())
}

fn compile_model(
    template: &InferenceModel,
    frames: usize,
    runtime: AcceleratorRuntime,
) -> Result<SharedRunnable, String> {
    let shape = tvec!(1, frames, BINS, 2);
    let mut model = template.clone();
    model
        .set_input_fact(0, f32::fact(shape.clone()).into())
        .map_err(|error| model_error("configure input", error))?;
    model
        .set_output_fact(0, f32::fact(shape).into())
        .map_err(|error| model_error("configure output", error))?;
    let model = model
        .into_typed()
        .map_err(|error| model_error("type", error))?;
    super::tract_runtime::prepare(model, runtime, "BSRNN model")
}

fn run_model(
    spectrum: &[f32],
    frames: usize,
    model: &dyn tract_onnx::tract_core::runtime::Runnable,
) -> Result<Vec<f32>, String> {
    let shape = tvec!(1, frames, BINS, 2);
    let tensor = Tensor::from_shape(&shape, spectrum)
        .map_err(|error| model_error("create spectrum tensor", error))?;
    let outputs = model
        .run(tvec!(tensor.into_tvalue()))
        .map_err(|error| model_error("run", error))?;
    let view = outputs[0]
        .to_plain_array_view::<f32>()
        .map_err(|error| model_error("read output", error))?;
    if view.len() != spectrum.len() {
        return Err(format!(
            "BSRNN output has {} values; expected {}",
            view.len(),
            spectrum.len()
        ));
    }
    let values: Vec<f32> = view.iter().copied().collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("BSRNN output contains a non-finite value".into());
    }
    Ok(values)
}

fn model_error(stage: &str, error: impl std::fmt::Display) -> String {
    format!("BSRNN ONNX {stage} failed: {error:#}")
}

fn periodic_hann() -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|index| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / FFT_SIZE as f32).cos()
        })
        .collect()
}

fn reflect_index(mut index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let last = length as isize - 1;
    while index < 0 || index > last {
        if index < 0 {
            index = -index;
        }
        if index > last {
            index = 2 * last - index;
        }
    }
    index as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use sha2::Digest as _;
    use tract_onnx::pb::{
        tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };

    #[test]
    fn stft_identity_reconstruction_is_transparent() {
        let input: Vec<f32> = (0..32_000)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 440.0 * index as f32 / MODEL_RATE as f32).sin() * 0.25
            })
            .collect();
        let spectrum = stft(&input);
        assert_eq!(spectrum.frames, 67);
        let output = istft(&spectrum.values, spectrum.frames, input.len()).unwrap();
        let mse = input
            .iter()
            .zip(&output)
            .map(|(expected, actual)| (expected - actual).powi(2))
            .sum::<f32>()
            / input.len() as f32;
        assert!(mse < 1e-8, "identity STFT MSE was {mse}");
    }

    #[test]
    fn torch_style_variance_uses_bessel_correction() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        let variance =
            input.iter().map(|x| (*x - mean).powi(2)).sum::<f64>() / (input.len() - 1) as f64;
        assert!((variance.sqrt() - 1.290_994_448_735_805_6).abs() < 1e-12);
    }

    #[test]
    fn spectral_identity_model_runs_end_to_end() {
        let mut bytes = Vec::new();
        spectral_identity_model().encode(&mut bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "denoize-bsrnn-identity-{}-{}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        let config = OnnxModelConfig {
            path: path.clone(),
            sample_rate: MODEL_RATE,
        };
        let input: Vec<f64> = (0..32_000)
            .map(|index| {
                0.1 * (2.0 * std::f64::consts::PI * 440.0 * index as f64 / MODEL_RATE as f64).sin()
                    + 0.02
            })
            .collect();
        let output = process(&[input.clone()], MODEL_RATE, &config).unwrap();
        std::fs::remove_file(path).unwrap();
        let mse = input
            .iter()
            .zip(&output[0])
            .map(|(expected, actual)| (expected - actual).powi(2))
            .sum::<f64>()
            / input.len() as f64;
        assert_eq!(output[0].len(), input.len());
        assert!(mse < 1e-10, "spectral identity model MSE was {mse}");
    }

    #[test]
    fn signed_v2_spectral_package_checks_contract_and_vectors_before_audio() {
        let directory = tempfile::tempdir().unwrap();
        let package_path = directory.path().join("bsrnn-v2.dmp");
        let mut model_bytes = Vec::new();
        spectral_identity_model().encode(&mut model_bytes).unwrap();
        let license = b"fixture license".to_vec();
        let provenance = br#"{"schema":"fixture-provenance-v1"}"#.to_vec();
        let values = vec![0.0_f64; BINS * 2];
        let vectors = serde_json::to_vec(&serde_json::json!({
            "schema": "denoize-runtime-model-numerical-vectors-v1",
            "profile_id": "fp32",
            "cases": [{
                "id": "spectral-identity",
                "inputs": [{
                    "name": "spectrum",
                    "element_type": "float32",
                    "shape": [1, 1, BINS, 2],
                    "values": values
                }],
                "outputs": [{
                    "name": "enhanced_spectrum",
                    "element_type": "float32",
                    "shape": [1, 1, BINS, 2],
                    "values": vec![0.0_f64; BINS * 2]
                }],
                "tolerance": { "absolute": 0.000001, "relative": 0.000001 }
            }]
        }))
        .unwrap();
        let file = |filename: &str, bytes: &[u8]| crate::RuntimeModelFileContract {
            filename: filename.into(),
            size_bytes: bytes.len() as u64,
            sha256: format!("{:x}", sha2::Sha256::digest(bytes)),
        };
        let axes = || {
            vec![
                crate::RuntimeModelAxisContractV2 {
                    name: "batch".into(),
                    kind: "batch".into(),
                    fixed: Some(1),
                },
                crate::RuntimeModelAxisContractV2 {
                    name: "frames".into(),
                    kind: "frame".into(),
                    fixed: None,
                },
                crate::RuntimeModelAxisContractV2 {
                    name: "frequency".into(),
                    kind: "frequency".into(),
                    fixed: Some(BINS as u64),
                },
                crate::RuntimeModelAxisContractV2 {
                    name: "complex".into(),
                    kind: "coordinate".into(),
                    fixed: Some(2),
                },
            ]
        };
        let resources = crate::RuntimeModelResourceContract {
            max_session_memory_bytes: crate::estimate_model_session_bytes(model_bytes.len() as u64)
                .unwrap(),
            max_worker_memory_bytes: 4096,
            max_gpu_session_memory_bytes: 0,
            max_gpu_worker_memory_bytes: 0,
            accelerators: vec!["cpu".into()],
        };
        let manifest = crate::RuntimeModelPackageManifestV2 {
            schema: crate::RUNTIME_MODEL_PACKAGE_SCHEMA_V2.into(),
            format_version: crate::RUNTIME_MODEL_PACKAGE_VERSION_V2,
            package_id: "denoize.test.bsrnn-v2".into(),
            package_revision: "1".into(),
            signing_key_id: "0000000000000001".into(),
            runtime: crate::RuntimeModelRuntimeContractV2 {
                kind: "onnx-audio-graph-v2".into(),
                sample_rate_hz: MODEL_RATE,
                mode: "finite".into(),
            },
            frontend: crate::RuntimeModelFrontendContractV2 {
                normalization: "pcm-f32-minus-one-to-one-v1".into(),
                resampling: "bandlimited-waveform-v1".into(),
                duration: "preserve-input-frames-v1".into(),
                channels: crate::RuntimeModelChannelContractV2 {
                    policy: "independent-mono".into(),
                    roles: vec![],
                    geometry: None,
                },
            },
            tensors: crate::RuntimeModelTensorSetContractV2 {
                inputs: vec![crate::RuntimeModelTensorContractV2 {
                    name: "spectrum".into(),
                    role: "audio".into(),
                    element_type: "float32".into(),
                    axes: axes(),
                    optional: false,
                    state_id: None,
                }],
                outputs: vec![crate::RuntimeModelTensorContractV2 {
                    name: "enhanced_spectrum".into(),
                    role: "audio".into(),
                    element_type: "float32".into(),
                    axes: axes(),
                    optional: false,
                    state_id: None,
                }],
            },
            state_pairs: vec![],
            latency: crate::RuntimeModelLatencyContractV2 {
                frame_samples: FFT_SIZE as u64,
                hop_samples: HOP_SIZE as u64,
                left_context_samples: (FFT_SIZE / 2) as u64,
                right_context_samples: (FFT_SIZE / 2) as u64,
                lookahead_samples: (FFT_SIZE / 2) as u64,
                algorithmic_latency_samples: (FFT_SIZE / 2) as u64,
                flush_samples: (FFT_SIZE / 2) as u64,
            },
            components: vec![
                crate::RuntimeModelComponentContractV2 {
                    id: "model-fp32".into(),
                    kind: "onnx-model".into(),
                    file: file("model.onnx", &model_bytes),
                },
                crate::RuntimeModelComponentContractV2 {
                    id: "license".into(),
                    kind: "license-notice".into(),
                    file: file("LICENSE.txt", &license),
                },
                crate::RuntimeModelComponentContractV2 {
                    id: "provenance".into(),
                    kind: "provenance-json".into(),
                    file: file("provenance.json", &provenance),
                },
                crate::RuntimeModelComponentContractV2 {
                    id: "vectors-fp32".into(),
                    kind: "numerical-vectors-json".into(),
                    file: file("vectors-fp32.json", &vectors),
                },
            ],
            precision_profiles: vec![crate::RuntimeModelPrecisionProfileContractV2 {
                id: "fp32".into(),
                element_type: "float32".into(),
                model_component: "model-fp32".into(),
                numerical_vectors_component: "vectors-fp32".into(),
                resources,
            }],
            default_precision_profile: "fp32".into(),
            license: crate::RuntimeModelLicenseContractV2 {
                spdx: "MIT".into(),
                notice_component: "license".into(),
            },
            provenance: crate::RuntimeModelProvenanceContractV2 {
                component: "provenance".into(),
                source_repository: "https://example.invalid/urgent".into(),
                source_revision: "b1dc3ad1e86419ff0bd666f455bda7936bff0e9a".into(),
                source_sha256: "0".repeat(64),
                source_license_spdx: "Apache-2.0".into(),
                checkpoint_source: "https://example.invalid/bsrnn.ckpt".into(),
                checkpoint_sha256: "1".repeat(64),
                checkpoint_license_spdx: "MIT".into(),
                conversion_tool: "fixture-converter".into(),
                conversion_revision: "1".into(),
                training_datasets: vec![crate::RuntimeModelTrainingDatasetContractV2 {
                    id: "synthetic".into(),
                    source: "urn:denoize:test:synthetic".into(),
                    revision: "1".into(),
                    sha256: Some("2".repeat(64)),
                    license_spdx: "CC0-1.0".into(),
                }],
            },
        };
        let mut invalid = manifest.clone();
        invalid.runtime.mode = "finite-and-streaming".into();
        assert!(validate_runtime_package_contract(&invalid).is_err());
        let package = crate::RuntimeModelPackage::for_onnx_v2_contract_test(
            package_path,
            manifest,
            vec![model_bytes, license, provenance, vectors],
        );
        let model = BsrnnModel::load_runtime_package(&package, AcceleratorRuntime::Cpu).unwrap();
        let input: Vec<f64> = (0..4_800)
            .map(|index| {
                0.1 * (2.0 * std::f64::consts::PI * 440.0 * index as f64 / MODEL_RATE as f64).sin()
            })
            .collect();
        let output = model
            .process(std::slice::from_ref(&input), MODEL_RATE)
            .unwrap();
        let mse = input
            .iter()
            .zip(&output[0])
            .map(|(expected, actual)| (expected - actual).powi(2))
            .sum::<f64>()
            / input.len() as f64;
        assert!(mse < 1e-10, "packaged spectral identity MSE was {mse}");
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
}
