//! Dedicated finite adapter for mixture-preserving music restoration.
//!
//! This graph is deliberately narrower than the generic waveform adapter. A
//! fixed mono or stereo program window produces one candidate program with the
//! exact same geometry plus a three-state `repair_state` head ordered bypass,
//! uncertain, apply. It cannot emit stems or silently reinterpret channels.

use super::tract_runtime::SharedRunnable;
use crate::{AcceleratorRuntime, RuntimeModelPackage, RuntimeModelPackageManifestV2};
use tract_onnx::prelude::*;

const REPAIR_STATE_CLASSES: usize = 3;
const PROBABILITY_SUM_TOLERANCE: f32 = 0.001;

#[derive(Clone, Copy, Debug)]
struct MusicRestorationGraphContract {
    channels: usize,
    window_samples: usize,
    state_frames: usize,
    audio_output: usize,
    state_output: usize,
}

/// One authenticated model-window result. State probabilities are ordered
/// bypass, uncertain, apply.
pub(crate) struct MusicRestorationInference {
    pub candidate: Vec<Vec<f32>>,
    pub state_probabilities: Vec<[f32; REPAIR_STATE_CLASSES]>,
}

/// Parsed, authenticated, numerically checked program-restoration graph.
pub(crate) struct MusicRestorationModel {
    runtime: AcceleratorRuntime,
    contract: MusicRestorationGraphContract,
    runnable: SharedRunnable,
}

impl MusicRestorationModel {
    pub(crate) fn load_runtime_package(
        package: &RuntimeModelPackage,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let manifest = package
            .manifest_v2()
            .ok_or("music restoration requires authenticated runtime model package v2")?;
        let contract = validate_runtime_package_contract(manifest)?;
        let profile = package
            .precision_profile_for(runtime)?
            .expect("v2 packages always select a precision profile");
        let mut reader = package.open_model_reader_for(runtime)?;
        let template = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|error| {
                format!(
                    "failed to load music-restoration ONNX graph from authenticated package {}: {error:#}",
                    package.package_path().display()
                )
            })?;
        reader.finish().map_err(|error| {
            format!(
                "failed to authenticate music-restoration ONNX bytes from package {}: {error}",
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

        let mut model = template;
        model
            .set_input_fact(
                0,
                f32::fact(tvec!(1, contract.channels, contract.window_samples)).into(),
            )
            .map_err(model_error)?;
        model
            .set_output_fact(
                contract.audio_output,
                f32::fact(tvec!(1, contract.channels, contract.window_samples)).into(),
            )
            .map_err(model_error)?;
        model
            .set_output_fact(
                contract.state_output,
                f32::fact(tvec!(1, contract.state_frames, REPAIR_STATE_CLASSES)).into(),
            )
            .map_err(model_error)?;
        let model = model.into_typed().map_err(model_error)?;
        let runnable =
            super::tract_runtime::prepare(model, runtime, "music program-restoration model")?;
        Ok(Self {
            runtime,
            contract,
            runnable,
        })
    }

    pub(crate) fn channels(&self) -> usize {
        self.contract.channels
    }

    pub(crate) fn window_samples(&self) -> usize {
        self.contract.window_samples
    }

    pub(crate) fn state_frames(&self) -> usize {
        self.contract.state_frames
    }

    pub(crate) fn process(
        &self,
        channels: &[Vec<f32>],
    ) -> Result<MusicRestorationInference, String> {
        if channels.len() != self.contract.channels
            || channels
                .iter()
                .any(|channel| channel.len() != self.contract.window_samples)
        {
            return Err(format!(
                "music-restoration graph requires {} channels by {} samples",
                self.contract.channels, self.contract.window_samples
            ));
        }
        if channels
            .iter()
            .flatten()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("music-restoration input contains an invalid normalized sample".into());
        }
        let input = channels.iter().flatten().copied().collect::<Vec<_>>();
        let tensor = Tensor::from_shape(
            &tvec!(1, self.contract.channels, self.contract.window_samples),
            &input,
        )
        .map_err(model_error)?;
        let outputs = self
            .runnable
            .run(tvec!(tensor.into_tvalue()))
            .map_err(model_error)?;
        if outputs.len() != 2 {
            return Err(format!(
                "music-restoration graph returned {} outputs; expected 2",
                outputs.len()
            ));
        }
        let audio = outputs[self.contract.audio_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        let expected_audio = self
            .contract
            .channels
            .checked_mul(self.contract.window_samples)
            .ok_or_else(|| "music-restoration audio shape overflow".to_string())?;
        if audio.len() != expected_audio {
            return Err(format!(
                "music-restoration graph returned {} audio samples; expected {expected_audio}",
                audio.len()
            ));
        }
        if audio
            .iter()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("music-restoration graph returned invalid normalized audio".into());
        }
        let flat_audio = audio.iter().copied().collect::<Vec<_>>();
        let mut candidate = Vec::new();
        candidate
            .try_reserve_exact(self.contract.channels)
            .map_err(|_| "unable to reserve music-restoration candidate channels".to_string())?;
        for channel in 0..self.contract.channels {
            let start = channel * self.contract.window_samples;
            candidate.push(flat_audio[start..start + self.contract.window_samples].to_vec());
        }

        let states = outputs[self.contract.state_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        let expected_states = self
            .contract
            .state_frames
            .checked_mul(REPAIR_STATE_CLASSES)
            .ok_or_else(|| "music-restoration state shape overflow".to_string())?;
        if states.len() != expected_states {
            return Err(format!(
                "music-restoration graph returned {} state values; expected {expected_states}",
                states.len()
            ));
        }
        let states = states.iter().copied().collect::<Vec<_>>();
        let mut state_probabilities = Vec::new();
        state_probabilities
            .try_reserve_exact(self.contract.state_frames)
            .map_err(|_| "unable to reserve music-restoration state frames".to_string())?;
        for frame in 0..self.contract.state_frames {
            let index = frame * REPAIR_STATE_CLASSES;
            let probabilities = [states[index], states[index + 1], states[index + 2]];
            validate_probabilities(probabilities)?;
            state_probabilities.push(probabilities);
        }
        Ok(MusicRestorationInference {
            candidate,
            state_probabilities,
        })
    }
}

impl std::fmt::Debug for MusicRestorationModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MusicRestorationModel")
            .field("runtime", &self.runtime)
            .field("contract", &self.contract)
            .finish_non_exhaustive()
    }
}

fn validate_runtime_package_contract(
    manifest: &RuntimeModelPackageManifestV2,
) -> Result<MusicRestorationGraphContract, String> {
    if manifest.runtime.mode != "finite"
        || !(8_000..=192_000).contains(&manifest.runtime.sample_rate_hz)
        || manifest.tensors.inputs.len() != 1
        || manifest.tensors.outputs.len() != 2
        || !manifest.state_pairs.is_empty()
        || manifest.frontend.channels.policy != "program-multichannel"
        || manifest.frontend.channels.geometry.is_some()
    {
        return Err(
            "music-restoration package must declare a finite, stateless program graph with one input and two outputs"
                .into(),
        );
    }
    let input = &manifest.tensors.inputs[0];
    if input.role != "audio" || input.element_type != "float32" || input.optional {
        return Err("music-restoration input must be required float32 audio".into());
    }
    let input_shape = exact_axes(
        input,
        &[("batch", Some(1)), ("channel", None), ("sample", None)],
        "audio input",
    )?;
    let channels = input_shape[1];
    let window_samples = input_shape[2];
    if !(1..=2).contains(&channels) || window_samples < 256 || window_samples > 16_777_216 {
        return Err("music-restoration input geometry is outside bounded limits".into());
    }
    validate_channel_roles(manifest, channels)?;
    if manifest.latency.frame_samples != window_samples as u64
        || manifest.latency.hop_samples == 0
        || manifest.latency.hop_samples > manifest.latency.frame_samples
    {
        return Err(
            "music-restoration package latency frame/hop must bind the fixed model window".into(),
        );
    }

    let audio_output = unique_role(&manifest.tensors.outputs, "audio")?;
    let state_output = manifest
        .tensors
        .outputs
        .iter()
        .enumerate()
        .find(|(_, tensor)| tensor.role == "diagnostic" && tensor.name == "repair_state")
        .map(|(index, _)| index)
        .ok_or("music-restoration package omits repair_state diagnostics")?;
    if manifest
        .tensors
        .outputs
        .iter()
        .filter(|tensor| tensor.role == "diagnostic")
        .count()
        != 1
    {
        return Err("music-restoration package requires exactly one diagnostic output".into());
    }
    let output_shape = exact_axes(
        &manifest.tensors.outputs[audio_output],
        &[("batch", Some(1)), ("channel", None), ("sample", None)],
        "audio output",
    )?;
    if output_shape != input_shape {
        return Err("music-restoration output must preserve exact input geometry".into());
    }
    let state_shape = exact_axes(
        &manifest.tensors.outputs[state_output],
        &[
            ("batch", Some(1)),
            ("frame", None),
            ("feature", Some(REPAIR_STATE_CLASSES)),
        ],
        "repair_state",
    )?;
    let state_frames = state_shape[1];
    if state_frames == 0
        || state_frames > window_samples
        || !window_samples.is_multiple_of(state_frames)
    {
        return Err(
            "music-restoration repair_state must evenly cover the authenticated window".into(),
        );
    }
    let state_hop = window_samples / state_frames;
    if !usize::try_from(manifest.latency.hop_samples)
        .map_err(|_| "music-restoration hop is too large".to_string())?
        .is_multiple_of(state_hop)
    {
        return Err(
            "music-restoration model hop must align to the repair-state frame clock".into(),
        );
    }
    Ok(MusicRestorationGraphContract {
        channels,
        window_samples,
        state_frames,
        audio_output,
        state_output,
    })
}

fn validate_channel_roles(
    manifest: &RuntimeModelPackageManifestV2,
    channels: usize,
) -> Result<(), String> {
    let roles = &manifest.frontend.channels.roles;
    let valid = match channels {
        1 => roles.len() == 1 && roles[0].channel_index == 0 && roles[0].role == "program-center",
        2 => {
            roles.len() == 2
                && roles[0].channel_index == 0
                && roles[0].role == "program-left"
                && roles[1].channel_index == 1
                && roles[1].role == "program-right"
        }
        _ => false,
    };
    if !valid {
        return Err(
            "music-restoration package requires exact mono-center or ordered stereo L/R roles"
                .into(),
        );
    }
    Ok(())
}

fn unique_role(
    tensors: &[crate::RuntimeModelTensorContractV2],
    role: &str,
) -> Result<usize, String> {
    let matches = tensors
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.role == role)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "music-restoration package requires exactly one {role} output"
        ));
    }
    Ok(matches[0])
}

fn exact_axes(
    tensor: &crate::RuntimeModelTensorContractV2,
    expected: &[(&str, Option<usize>)],
    label: &str,
) -> Result<Vec<usize>, String> {
    if tensor.element_type != "float32"
        || tensor.optional
        || tensor.state_id.is_some()
        || tensor.axes.len() != expected.len()
    {
        return Err(format!(
            "music-restoration {label} has the wrong element type or rank"
        ));
    }
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(expected.len())
        .map_err(|_| format!("unable to reserve music-restoration {label} shape"))?;
    for (axis, (kind, exact)) in tensor.axes.iter().zip(expected) {
        if axis.kind != *kind {
            return Err(format!(
                "music-restoration {label} axes do not match the closed contract"
            ));
        }
        let fixed = axis
            .fixed
            .ok_or_else(|| format!("music-restoration {label} requires fixed dimensions"))?;
        let fixed = usize::try_from(fixed)
            .map_err(|_| format!("music-restoration {label} dimension is too large"))?;
        if exact.is_some_and(|value| value != fixed) {
            return Err(format!(
                "music-restoration {label} dimension does not match the closed contract"
            ));
        }
        shape.push(fixed);
    }
    Ok(shape)
}

fn validate_probabilities(values: [f32; REPAIR_STATE_CLASSES]) -> Result<(), String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (values.iter().sum::<f32>() - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
    {
        return Err("music-restoration graph returned invalid state probabilities".into());
    }
    Ok(())
}

fn model_error(error: impl std::fmt::Display) -> String {
    format!("music-restoration ONNX inference failed: {error}")
}
