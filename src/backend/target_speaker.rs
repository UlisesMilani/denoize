//! Dedicated finite target-speaker extraction adapter.
//!
//! This is deliberately separate from the generic waveform ONNX backend. A
//! target-speaker graph has two semantically different waveform inputs and a
//! calibrated presence output. Treating either auxiliary tensor as an
//! implementation detail would make it possible to run the wrong graph while
//! still producing plausible audio.

use super::tract_runtime::SharedRunnable;
use crate::{AcceleratorRuntime, RuntimeModelPackage, RuntimeModelPackageManifestV2};
use std::sync::{Arc, Mutex};
use tract_onnx::prelude::*;

const PRESENCE_CLASSES: usize = 3;
const PRESENCE_SUM_TOLERANCE: f32 = 0.001;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaveformLayout {
    BatchSamples,
    BatchChannelsSamples,
}

impl WaveformLayout {
    fn shape(self, samples: usize) -> TVec<usize> {
        match self {
            Self::BatchSamples => tvec!(1, samples),
            Self::BatchChannelsSamples => tvec!(1, 1, samples),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TargetSpeakerGraphContract {
    mixture_input: usize,
    enrollment_input: usize,
    audio_output: usize,
    presence_output: usize,
    mixture_layout: WaveformLayout,
    enrollment_layout: WaveformLayout,
    fixed_mixture_samples: Option<usize>,
    fixed_enrollment_samples: Option<usize>,
}

struct CompiledTargetSpeakerModel {
    mixture_samples: usize,
    enrollment_samples: usize,
    runnable: SharedRunnable,
}

/// One model-rate inference result. Presence probabilities are ordered as
/// absent, uncertain, and present by the dedicated adapter contract.
pub(crate) struct TargetSpeakerInference {
    pub audio: Vec<f32>,
    pub presence_probabilities: [f32; PRESENCE_CLASSES],
}

/// Parsed, authenticated, numerically checked target-speaker graph.
pub(crate) struct TargetSpeakerModel {
    template: InferenceModel,
    runtime: AcceleratorRuntime,
    contract: TargetSpeakerGraphContract,
    compiled: Mutex<Option<CompiledTargetSpeakerModel>>,
}

impl TargetSpeakerModel {
    pub(crate) fn load_runtime_package(
        package: &RuntimeModelPackage,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let manifest = package.manifest_v2().ok_or(
            "target-speaker extraction requires authenticated runtime model package v2",
        )?;
        let contract = validate_runtime_package_contract(manifest)?;
        let profile = package
            .precision_profile_for(runtime)?
            .expect("v2 packages always select a precision profile");
        let mut reader = package.open_model_reader_for(runtime)?;
        let template = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|error| {
                format!(
                    "failed to load target-speaker ONNX graph from authenticated package {}: {error:#}",
                    package.package_path().display()
                )
            })?;
        reader.finish().map_err(|error| {
            format!(
                "failed to authenticate target-speaker ONNX bytes from package {}: {error}",
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
            .expect("v2 precision profiles always carry numerical vectors");
        super::onnx::validate_v2_numerical_vectors(
            &template, manifest, profile, &vectors, runtime,
        )?;
        Ok(Self {
            template,
            runtime,
            contract,
            compiled: Mutex::new(None),
        })
    }

    pub(crate) fn fixed_enrollment_samples(&self) -> Option<usize> {
        self.contract.fixed_enrollment_samples
    }

    pub(crate) fn process(
        &self,
        mixture: &[f32],
        enrollment: &[f32],
    ) -> Result<TargetSpeakerInference, String> {
        if mixture.is_empty() {
            return Err("target-speaker mixture must not be empty".into());
        }
        if enrollment.is_empty() {
            return Err("target-speaker enrollment must not be empty".into());
        }
        let runnable = self.compiled_model(mixture.len(), enrollment.len())?;
        let mixture_tensor = Tensor::from_shape(
            &self.contract.mixture_layout.shape(mixture.len()),
            mixture,
        )
        .map_err(model_error)?;
        let enrollment_tensor = Tensor::from_shape(
            &self.contract.enrollment_layout.shape(enrollment.len()),
            enrollment,
        )
        .map_err(model_error)?;
        let mut inputs = TVec::new();
        for index in 0..2 {
            if index == self.contract.mixture_input {
                inputs.push(mixture_tensor.clone().into_tvalue());
            } else if index == self.contract.enrollment_input {
                inputs.push(enrollment_tensor.clone().into_tvalue());
            } else {
                unreachable!("the closed target-speaker contract has exactly two inputs");
            }
        }
        let outputs = runnable.run(inputs).map_err(model_error)?;
        if outputs.len() != 2 {
            return Err(format!(
                "target-speaker graph returned {} outputs; expected 2",
                outputs.len()
            ));
        }
        let audio_view = outputs[self.contract.audio_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        if audio_view.len() != mixture.len() {
            return Err(format!(
                "target-speaker graph returned {} audio samples for a {}-sample mixture",
                audio_view.len(),
                mixture.len()
            ));
        }
        let audio: Vec<f32> = audio_view.iter().copied().collect();
        let presence_view = outputs[self.contract.presence_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        if presence_view.len() != PRESENCE_CLASSES {
            return Err(format!(
                "target-speaker graph returned {} presence values; expected {PRESENCE_CLASSES}",
                presence_view.len()
            ));
        }
        let mut presence_values = presence_view.iter().copied();
        let presence_probabilities = [
            presence_values.next().expect("presence length was checked"),
            presence_values.next().expect("presence length was checked"),
            presence_values.next().expect("presence length was checked"),
        ];
        validate_presence_probabilities(presence_probabilities)?;
        Ok(TargetSpeakerInference {
            audio,
            presence_probabilities,
        })
    }

    fn compiled_model(
        &self,
        mixture_samples: usize,
        enrollment_samples: usize,
    ) -> Result<SharedRunnable, String> {
        if self
            .contract
            .fixed_mixture_samples
            .is_some_and(|required| required != mixture_samples)
        {
            return Err(format!(
                "target-speaker graph requires {} mixture samples, got {mixture_samples}",
                self.contract.fixed_mixture_samples.expect("checked Some")
            ));
        }
        if self
            .contract
            .fixed_enrollment_samples
            .is_some_and(|required| required != enrollment_samples)
        {
            return Err(format!(
                "target-speaker graph requires {} enrollment samples, got {enrollment_samples}",
                self.contract.fixed_enrollment_samples.expect("checked Some")
            ));
        }
        let mut compiled = self
            .compiled
            .lock()
            .map_err(|_| "target-speaker compiled-model cache lock was poisoned".to_string())?;
        if let Some(cached) = compiled.as_ref() {
            if cached.mixture_samples == mixture_samples
                && cached.enrollment_samples == enrollment_samples
            {
                return Ok(Arc::clone(&cached.runnable));
            }
        }
        *compiled = None;
        let mut model = self.template.clone();
        for index in 0..2 {
            let shape = if index == self.contract.mixture_input {
                self.contract.mixture_layout.shape(mixture_samples)
            } else {
                self.contract.enrollment_layout.shape(enrollment_samples)
            };
            model
                .set_input_fact(index, f32::fact(shape).into())
                .map_err(model_error)?;
        }
        model
            .set_output_fact(
                self.contract.audio_output,
                f32::fact(self.contract.mixture_layout.shape(mixture_samples)).into(),
            )
            .map_err(model_error)?;
        model
            .set_output_fact(
                self.contract.presence_output,
                f32::fact(tvec!(1, PRESENCE_CLASSES)).into(),
            )
            .map_err(model_error)?;
        let model = model.into_typed().map_err(model_error)?;
        let runnable = super::tract_runtime::prepare(
            model,
            self.runtime,
            "target-speaker extraction model",
        )?;
        *compiled = Some(CompiledTargetSpeakerModel {
            mixture_samples,
            enrollment_samples,
            runnable: Arc::clone(&runnable),
        });
        Ok(runnable)
    }
}

fn validate_runtime_package_contract(
    manifest: &RuntimeModelPackageManifestV2,
) -> Result<TargetSpeakerGraphContract, String> {
    let closed = manifest.runtime.mode == "finite"
        && manifest.frontend.channels.policy == "independent-mono"
        && manifest.frontend.channels.roles.is_empty()
        && manifest.frontend.channels.geometry.is_none()
        && manifest.tensors.inputs.len() == 2
        && manifest.tensors.outputs.len() == 2
        && manifest.state_pairs.is_empty();
    if !closed {
        return Err(
            "target-speaker package must declare a finite, independent-mono, stateless graph with exactly two inputs and two outputs"
                .into(),
        );
    }
    let mixture_input = unique_role(&manifest.tensors.inputs, "audio", "input")?;
    let enrollment_input = unique_role(&manifest.tensors.inputs, "enrollment", "input")?;
    let audio_output = unique_role(&manifest.tensors.outputs, "audio", "output")?;
    let presence_output = unique_role(&manifest.tensors.outputs, "diagnostic", "output")?;
    let mixture = &manifest.tensors.inputs[mixture_input];
    let enrollment = &manifest.tensors.inputs[enrollment_input];
    let output = &manifest.tensors.outputs[audio_output];
    let presence = &manifest.tensors.outputs[presence_output];
    let (mixture_layout, fixed_mixture_samples) = waveform_contract(mixture, "mixture")?;
    let (enrollment_layout, fixed_enrollment_samples) =
        waveform_contract(enrollment, "enrollment")?;
    let (output_layout, fixed_output_samples) = waveform_contract(output, "audio output")?;
    if output_layout != mixture_layout
        || output.axes != mixture.axes
        || fixed_output_samples != fixed_mixture_samples
    {
        return Err(
            "target-speaker audio output must exactly match the mixture tensor axes".into(),
        );
    }
    let presence_axes = presence
        .axes
        .iter()
        .map(|axis| (axis.kind.as_str(), axis.fixed))
        .collect::<Vec<_>>();
    if presence.element_type != "float32"
        || presence.optional
        || presence.state_id.is_some()
        || presence_axes.as_slice()
            != [("batch", Some(1)), ("feature", Some(PRESENCE_CLASSES as u64))]
    {
        return Err(
            "target-speaker diagnostic output must be float32 [batch=1,feature=3] probabilities ordered absent, uncertain, present"
                .into(),
        );
    }
    Ok(TargetSpeakerGraphContract {
        mixture_input,
        enrollment_input,
        audio_output,
        presence_output,
        mixture_layout,
        enrollment_layout,
        fixed_mixture_samples,
        fixed_enrollment_samples,
    })
}

fn unique_role(
    tensors: &[crate::RuntimeModelTensorContractV2],
    role: &str,
    kind: &str,
) -> Result<usize, String> {
    let indices = tensors
        .iter()
        .enumerate()
        .filter_map(|(index, tensor)| (tensor.role == role).then_some(index))
        .collect::<Vec<_>>();
    if indices.len() != 1 {
        return Err(format!(
            "target-speaker graph requires exactly one {kind} with role {role}"
        ));
    }
    Ok(indices[0])
}

fn waveform_contract(
    tensor: &crate::RuntimeModelTensorContractV2,
    description: &str,
) -> Result<(WaveformLayout, Option<usize>), String> {
    if tensor.element_type != "float32" || tensor.optional || tensor.state_id.is_some() {
        return Err(format!(
            "target-speaker {description} must be a required stateless float32 tensor"
        ));
    }
    let axes = tensor
        .axes
        .iter()
        .map(|axis| (axis.kind.as_str(), axis.fixed))
        .collect::<Vec<_>>();
    let (layout, samples) = match axes.as_slice() {
        [("batch", Some(1)), ("sample", samples)] => {
            (WaveformLayout::BatchSamples, *samples)
        }
        [("batch", Some(1)), ("channel", Some(1)), ("sample", samples)] => {
            (WaveformLayout::BatchChannelsSamples, *samples)
        }
        _ => {
            return Err(format!(
                "target-speaker {description} must use [batch=1,sample] or [batch=1,channel=1,sample] axes"
            ));
        }
    };
    let samples = samples
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| format!("target-speaker {description} sample axis is too large"))
        })
        .transpose()?;
    if samples == Some(0) {
        return Err(format!(
            "target-speaker {description} fixed sample axis must be positive"
        ));
    }
    Ok((layout, samples))
}

fn validate_presence_probabilities(values: [f32; PRESENCE_CLASSES]) -> Result<(), String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(
            "target-speaker presence output must contain finite probabilities in 0..=1".into(),
        );
    }
    let sum = values.iter().sum::<f32>();
    if (sum - 1.0).abs() > PRESENCE_SUM_TOLERANCE {
        return Err(format!(
            "target-speaker presence probabilities sum to {sum}; expected 1 within {PRESENCE_SUM_TOLERANCE}"
        ));
    }
    Ok(())
}

fn model_error(error: impl std::fmt::Display) -> String {
    format!("target-speaker ONNX inference failed: {error:#}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use sha2::Digest as _;
    use tract_onnx::pb::{
        attribute_proto, tensor_proto, tensor_shape_proto, type_proto, AttributeProto, GraphProto,
        ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto,
        ValueInfoProto,
    };

    #[test]
    fn signed_multi_input_graph_runs_presence_contract() {
        let directory = tempfile::tempdir().unwrap();
        let package_path = directory.path().join("target-speaker.dmp");
        let mut model_bytes = Vec::new();
        target_speaker_identity_model().encode(&mut model_bytes).unwrap();
        let license = b"fixture license".to_vec();
        let provenance = br#"{"schema":"fixture-provenance-v1"}"#.to_vec();
        let vectors = serde_json::to_vec(&serde_json::json!({
            "schema": "denoize-runtime-model-numerical-vectors-v1",
            "profile_id": "fp32",
            "cases": [{
                "id": "present-identity",
                "inputs": [
                    {
                        "name": "mixture",
                        "element_type": "float32",
                        "shape": [1, 4],
                        "values": [-0.5, 0.0, 0.25, 0.75]
                    },
                    {
                        "name": "enrollment",
                        "element_type": "float32",
                        "shape": [1, 3],
                        "values": [0.1, -0.1, 0.2]
                    }
                ],
                "outputs": [
                    {
                        "name": "extracted",
                        "element_type": "float32",
                        "shape": [1, 4],
                        "values": [-0.5, 0.0, 0.25, 0.75]
                    },
                    {
                        "name": "target_presence_probabilities",
                        "element_type": "float32",
                        "shape": [1, 3],
                        "values": [0.0, 0.0, 1.0]
                    }
                ],
                "tolerance": { "absolute": 0.000001, "relative": 0.000001 }
            }]
        }))
        .unwrap();
        let components = vec![
            model_bytes.clone(),
            license.clone(),
            provenance.clone(),
            vectors.clone(),
        ];
        let manifest = manifest(&components);
        let package = RuntimeModelPackage::for_onnx_v2_contract_test(
            package_path,
            manifest,
            components,
        );
        let model =
            TargetSpeakerModel::load_runtime_package(&package, AcceleratorRuntime::Cpu).unwrap();
        let result = model
            .process(&[-0.5, 0.0, 0.25, 0.75], &[0.1, -0.1, 0.2])
            .unwrap();
        assert_eq!(result.audio, [-0.5, 0.0, 0.25, 0.75]);
        assert_eq!(result.presence_probabilities, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn rejects_presence_logits_and_generic_waveform_use() {
        assert!(validate_presence_probabilities([0.0, 0.0, 3.0]).is_err());
        assert!(validate_presence_probabilities([0.3, 0.3, 0.3]).is_err());
        let components = vec![vec![0], vec![0], vec![0], vec![0]];
        let mut invalid = manifest(&components);
        invalid.tensors.inputs.retain(|tensor| tensor.role == "audio");
        assert!(validate_runtime_package_contract(&invalid)
            .unwrap_err()
            .contains("exactly two inputs"));
    }

    fn manifest(components: &[Vec<u8>]) -> RuntimeModelPackageManifestV2 {
        let file = |filename: &str, bytes: &[u8]| crate::RuntimeModelFileContract {
            filename: filename.into(),
            size_bytes: bytes.len() as u64,
            sha256: format!("{:x}", sha2::Sha256::digest(bytes)),
        };
        let waveform_axes = || {
            vec![
                crate::RuntimeModelAxisContractV2 {
                    name: "batch".into(),
                    kind: "batch".into(),
                    fixed: Some(1),
                },
                crate::RuntimeModelAxisContractV2 {
                    name: "samples".into(),
                    kind: "sample".into(),
                    fixed: None,
                },
            ]
        };
        let presence_axes = || {
            vec![
                crate::RuntimeModelAxisContractV2 {
                    name: "batch".into(),
                    kind: "batch".into(),
                    fixed: Some(1),
                },
                crate::RuntimeModelAxisContractV2 {
                    name: "classes".into(),
                    kind: "feature".into(),
                    fixed: Some(3),
                },
            ]
        };
        RuntimeModelPackageManifestV2 {
            schema: crate::RUNTIME_MODEL_PACKAGE_SCHEMA_V2.into(),
            format_version: crate::RUNTIME_MODEL_PACKAGE_VERSION_V2,
            package_id: "denoize.test.target-speaker".into(),
            package_revision: "1".into(),
            signing_key_id: "0000000000000001".into(),
            runtime: crate::RuntimeModelRuntimeContractV2 {
                kind: "onnx-audio-graph-v2".into(),
                sample_rate_hz: 16_000,
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
                inputs: vec![
                    crate::RuntimeModelTensorContractV2 {
                        name: "mixture".into(),
                        role: "audio".into(),
                        element_type: "float32".into(),
                        axes: waveform_axes(),
                        optional: false,
                        state_id: None,
                    },
                    crate::RuntimeModelTensorContractV2 {
                        name: "enrollment".into(),
                        role: "enrollment".into(),
                        element_type: "float32".into(),
                        axes: waveform_axes(),
                        optional: false,
                        state_id: None,
                    },
                ],
                outputs: vec![
                    crate::RuntimeModelTensorContractV2 {
                        name: "extracted".into(),
                        role: "audio".into(),
                        element_type: "float32".into(),
                        axes: waveform_axes(),
                        optional: false,
                        state_id: None,
                    },
                    crate::RuntimeModelTensorContractV2 {
                        name: "target_presence_probabilities".into(),
                        role: "diagnostic".into(),
                        element_type: "float32".into(),
                        axes: presence_axes(),
                        optional: false,
                        state_id: None,
                    },
                ],
            },
            state_pairs: vec![],
            latency: crate::RuntimeModelLatencyContractV2 {
                frame_samples: 1,
                hop_samples: 1,
                left_context_samples: 0,
                right_context_samples: 0,
                lookahead_samples: 0,
                algorithmic_latency_samples: 0,
                flush_samples: 0,
            },
            components: vec![
                crate::RuntimeModelComponentContractV2 {
                    id: "model-fp32".into(),
                    kind: "onnx-model".into(),
                    file: file("model.onnx", &components[0]),
                },
                crate::RuntimeModelComponentContractV2 {
                    id: "license".into(),
                    kind: "license-notice".into(),
                    file: file("LICENSE.txt", &components[1]),
                },
                crate::RuntimeModelComponentContractV2 {
                    id: "provenance".into(),
                    kind: "provenance-json".into(),
                    file: file("provenance.json", &components[2]),
                },
                crate::RuntimeModelComponentContractV2 {
                    id: "vectors-fp32".into(),
                    kind: "numerical-vectors-json".into(),
                    file: file("vectors.json", &components[3]),
                },
            ],
            precision_profiles: vec![crate::RuntimeModelPrecisionProfileContractV2 {
                id: "fp32".into(),
                element_type: "float32".into(),
                model_component: "model-fp32".into(),
                numerical_vectors_component: "vectors-fp32".into(),
                resources: crate::RuntimeModelResourceContract {
                    max_session_memory_bytes: crate::estimate_model_session_bytes(
                        components[0].len() as u64,
                    )
                    .unwrap(),
                    max_worker_memory_bytes: 4096,
                    max_gpu_session_memory_bytes: 0,
                    max_gpu_worker_memory_bytes: 0,
                    accelerators: vec!["cpu".into()],
                },
            }],
            default_precision_profile: "fp32".into(),
            license: crate::RuntimeModelLicenseContractV2 {
                spdx: "MIT".into(),
                notice_component: "license".into(),
            },
            provenance: crate::RuntimeModelProvenanceContractV2 {
                component: "provenance".into(),
                source_repository: "https://example.invalid/target-speaker".into(),
                source_revision: "0123456789abcdef".into(),
                source_sha256: "0".repeat(64),
                source_license_spdx: "MIT".into(),
                checkpoint_source: "https://example.invalid/target-speaker.ckpt".into(),
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
        }
    }

    fn target_speaker_identity_model() -> ModelProto {
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
                                float_data: vec![0.0, 0.0, 1.0],
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
                        vec![dimension_value(1), dimension_parameter("enrollment_samples")],
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
