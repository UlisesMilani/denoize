//! Dedicated finite adapter for closed-class target-sound extraction.
//!
//! A graph accepted here has two explicit inputs (program audio and a one-hot
//! closed-catalog query) and three explicit outputs (target, residual, and
//! calibrated absent/uncertain/present probabilities). Target and residual
//! semantics are never inferred from tensor order or overloaded mask roles.

use super::tract_runtime::SharedRunnable;
use crate::{
    AcceleratorRuntime, RuntimeModelPackage, RuntimeModelPackageManifestV2,
    RuntimeModelTensorContractV2,
};
use tract_onnx::prelude::*;

const PRESENCE_CLASSES: usize = 3;
const PROBABILITY_SUM_TOLERANCE: f32 = 0.001;
const MAX_QUERY_CLASSES: usize = 4096;

#[derive(Clone, Copy, Debug)]
struct TargetSoundGraphContract {
    channels: usize,
    window_samples: usize,
    query_classes: usize,
    audio_input: usize,
    query_input: usize,
    target_output: usize,
    residual_output: usize,
    presence_output: usize,
}

/// One authenticated model-window result. Presence probabilities are ordered
/// absent, uncertain, present.
pub(crate) struct TargetSoundInference {
    pub target: Vec<Vec<f32>>,
    pub residual: Vec<Vec<f32>>,
    pub presence_probabilities: [f32; PRESENCE_CLASSES],
}

/// Parsed, authenticated, numerically checked target-sound graph.
pub(crate) struct TargetSoundModel {
    runtime: AcceleratorRuntime,
    contract: TargetSoundGraphContract,
    runnable: SharedRunnable,
}

impl TargetSoundModel {
    pub(crate) fn load_runtime_package(
        package: &RuntimeModelPackage,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let manifest = package
            .manifest_v2()
            .ok_or("target-sound extraction requires authenticated runtime model package v2")?;
        let contract = validate_runtime_package_contract(manifest)?;
        let profile = package
            .precision_profile_for(runtime)?
            .expect("v2 packages always select a precision profile");
        let mut reader = package.open_model_reader_for(runtime)?;
        let template = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|error| {
                format!(
                    "failed to load target-sound ONNX graph from authenticated package {}: {error:#}",
                    package.package_path().display()
                )
            })?;
        reader.finish().map_err(|error| {
            format!(
                "failed to authenticate target-sound ONNX bytes from package {}: {error}",
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
                contract.audio_input,
                f32::fact(tvec!(1, contract.channels, contract.window_samples)).into(),
            )
            .map_err(model_error)?;
        model
            .set_input_fact(
                contract.query_input,
                f32::fact(tvec!(1, contract.query_classes)).into(),
            )
            .map_err(model_error)?;
        for output in [contract.target_output, contract.residual_output] {
            model
                .set_output_fact(
                    output,
                    f32::fact(tvec!(1, contract.channels, contract.window_samples)).into(),
                )
                .map_err(model_error)?;
        }
        model
            .set_output_fact(
                contract.presence_output,
                f32::fact(tvec!(1, PRESENCE_CLASSES)).into(),
            )
            .map_err(model_error)?;
        let model = model.into_typed().map_err(model_error)?;
        let runnable =
            super::tract_runtime::prepare(model, runtime, "target-sound extraction model")?;
        Ok(Self {
            runtime,
            contract,
            runnable,
        })
    }

    pub(crate) const fn channels(&self) -> usize {
        self.contract.channels
    }

    pub(crate) const fn window_samples(&self) -> usize {
        self.contract.window_samples
    }

    pub(crate) const fn query_classes(&self) -> usize {
        self.contract.query_classes
    }

    pub(crate) fn process(
        &self,
        channels: &[Vec<f32>],
        class_index: usize,
    ) -> Result<TargetSoundInference, String> {
        if channels.len() != self.contract.channels
            || channels
                .iter()
                .any(|channel| channel.len() != self.contract.window_samples)
        {
            return Err(format!(
                "target-sound graph requires {} channels by {} samples",
                self.contract.channels, self.contract.window_samples
            ));
        }
        if channels
            .iter()
            .flatten()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("target-sound input contains an invalid normalized sample".into());
        }
        if class_index >= self.contract.query_classes {
            return Err(format!(
                "target-sound class index {class_index} is outside the authenticated {}-class query catalog",
                self.contract.query_classes
            ));
        }
        let audio = channels.iter().flatten().copied().collect::<Vec<_>>();
        let audio_tensor = Tensor::from_shape(
            &tvec!(1, self.contract.channels, self.contract.window_samples),
            &audio,
        )
        .map_err(model_error)?;
        let mut query = vec![0.0_f32; self.contract.query_classes];
        query[class_index] = 1.0;
        let query_tensor = Tensor::from_shape(&tvec!(1, self.contract.query_classes), &query)
            .map_err(model_error)?;
        let placeholder = Tensor::zero::<f32>(&[1])
            .map_err(model_error)?
            .into_tvalue();
        let mut inputs: TVec<TValue> = tvec!(placeholder.clone(), placeholder);
        inputs[self.contract.audio_input] = audio_tensor.into_tvalue();
        inputs[self.contract.query_input] = query_tensor.into_tvalue();
        let outputs = self.runnable.run(inputs).map_err(model_error)?;
        if outputs.len() != 3 {
            return Err(format!(
                "target-sound graph returned {} outputs; expected 3",
                outputs.len()
            ));
        }
        let target = decode_audio(
            &outputs[self.contract.target_output],
            self.contract.channels,
            self.contract.window_samples,
            "target",
        )?;
        let residual = decode_audio(
            &outputs[self.contract.residual_output],
            self.contract.channels,
            self.contract.window_samples,
            "residual",
        )?;
        let presence = outputs[self.contract.presence_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        if presence.len() != PRESENCE_CLASSES {
            return Err(format!(
                "target-sound graph returned {} presence values; expected {PRESENCE_CLASSES}",
                presence.len()
            ));
        }
        let mut values = presence.iter().copied();
        let presence_probabilities = [
            values.next().expect("presence length checked"),
            values.next().expect("presence length checked"),
            values.next().expect("presence length checked"),
        ];
        validate_probabilities(presence_probabilities)?;
        Ok(TargetSoundInference {
            target,
            residual,
            presence_probabilities,
        })
    }
}

impl std::fmt::Debug for TargetSoundModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TargetSoundModel")
            .field("runtime", &self.runtime)
            .field("contract", &self.contract)
            .finish_non_exhaustive()
    }
}

fn validate_runtime_package_contract(
    manifest: &RuntimeModelPackageManifestV2,
) -> Result<TargetSoundGraphContract, String> {
    if manifest.runtime.mode != "finite"
        || !(8_000..=192_000).contains(&manifest.runtime.sample_rate_hz)
        || manifest.tensors.inputs.len() != 2
        || manifest.tensors.outputs.len() != 3
        || !manifest.state_pairs.is_empty()
        || manifest.frontend.channels.policy != "program-multichannel"
        || manifest.frontend.channels.geometry.is_some()
    {
        return Err(
            "target-sound package must declare a finite stateless program graph with audio/query inputs and target/residual/presence outputs"
                .into(),
        );
    }
    if manifest
        .tensors
        .inputs
        .iter()
        .any(|tensor| !matches!(tensor.role.as_str(), "audio" | "query"))
        || manifest
            .tensors
            .outputs
            .iter()
            .any(|tensor| !matches!(tensor.role.as_str(), "audio" | "residual" | "diagnostic"))
    {
        return Err(
            "target-sound graph contains a tensor outside its closed semantic roles".into(),
        );
    }
    let audio_input = unique_role(&manifest.tensors.inputs, "audio", "input")?;
    let query_input = unique_role(&manifest.tensors.inputs, "query", "input")?;
    let target_output = unique_role(&manifest.tensors.outputs, "audio", "output")?;
    let residual_output = unique_role(&manifest.tensors.outputs, "residual", "output")?;
    let presence_output = unique_role(&manifest.tensors.outputs, "diagnostic", "output")?;
    let input_shape = exact_axes(
        &manifest.tensors.inputs[audio_input],
        &[("batch", Some(1)), ("channel", None), ("sample", None)],
        "audio input",
    )?;
    let channels = input_shape[1];
    let window_samples = input_shape[2];
    if !(1..=2).contains(&channels) || !(256..=16_777_216).contains(&window_samples) {
        return Err("target-sound input geometry is outside bounded limits".into());
    }
    validate_channel_roles(manifest, channels)?;
    if manifest.latency.frame_samples != window_samples as u64
        || manifest.latency.hop_samples == 0
        || manifest.latency.hop_samples > manifest.latency.frame_samples
    {
        return Err("target-sound frame/hop must bind the fixed authenticated window".into());
    }
    let query_shape = exact_axes(
        &manifest.tensors.inputs[query_input],
        &[("batch", Some(1)), ("feature", None)],
        "query input",
    )?;
    let query_classes = query_shape[1];
    if !(2..=MAX_QUERY_CLASSES).contains(&query_classes) {
        return Err("target-sound query catalog must contain 2..=4096 classes".into());
    }
    for (index, label) in [
        (target_output, "target output"),
        (residual_output, "residual output"),
    ] {
        if exact_axes(
            &manifest.tensors.outputs[index],
            &[("batch", Some(1)), ("channel", None), ("sample", None)],
            label,
        )? != input_shape
        {
            return Err(format!(
                "target-sound {label} must preserve exact input geometry"
            ));
        }
    }
    let presence = &manifest.tensors.outputs[presence_output];
    exact_axes(
        presence,
        &[("batch", Some(1)), ("feature", Some(PRESENCE_CLASSES))],
        "presence output",
    )?;
    if presence.name != "presence" {
        return Err(
            "target-sound diagnostic must be named presence with absent/uncertain/present values"
                .into(),
        );
    }
    if manifest.tensors.outputs[target_output].name != "target"
        || manifest.tensors.outputs[residual_output].name != "residual"
    {
        return Err("target-sound audio outputs must be named target and residual".into());
    }
    Ok(TargetSoundGraphContract {
        channels,
        window_samples,
        query_classes,
        audio_input,
        query_input,
        target_output,
        residual_output,
        presence_output,
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
    if valid {
        Ok(())
    } else {
        Err("target-sound package requires exact mono-center or ordered stereo L/R roles".into())
    }
}

fn unique_role(
    tensors: &[RuntimeModelTensorContractV2],
    role: &str,
    kind: &str,
) -> Result<usize, String> {
    let matches = tensors
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.role == role)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "target-sound package requires exactly one {role} {kind}"
        ));
    }
    Ok(matches[0])
}

fn exact_axes(
    tensor: &RuntimeModelTensorContractV2,
    expected: &[(&str, Option<usize>)],
    label: &str,
) -> Result<Vec<usize>, String> {
    if tensor.element_type != "float32" || tensor.optional || tensor.axes.len() != expected.len() {
        return Err(format!(
            "target-sound {label} must be required float32 with {} axes",
            expected.len()
        ));
    }
    tensor
        .axes
        .iter()
        .zip(expected)
        .map(|(axis, (kind, fixed))| {
            if axis.kind != *kind {
                return Err(format!("target-sound {label} has the wrong axis order"));
            }
            let value = axis
                .fixed
                .ok_or_else(|| format!("target-sound {label} axes must be fixed"))?;
            let value = usize::try_from(value)
                .map_err(|_| format!("target-sound {label} axis is too large"))?;
            if fixed.is_some_and(|required| required != value) {
                return Err(format!(
                    "target-sound {label} fixed axis differs from contract"
                ));
            }
            Ok(value)
        })
        .collect()
}

fn decode_audio(
    value: &TValue,
    channels: usize,
    samples: usize,
    label: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let view = value.to_plain_array_view::<f32>().map_err(model_error)?;
    let expected = channels
        .checked_mul(samples)
        .ok_or_else(|| format!("target-sound {label} shape overflow"))?;
    if view.len() != expected {
        return Err(format!(
            "target-sound graph returned {} {label} samples; expected {expected}",
            view.len()
        ));
    }
    if view
        .iter()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(format!(
            "target-sound graph returned invalid normalized {label} audio"
        ));
    }
    let flat = view.iter().copied().collect::<Vec<_>>();
    (0..channels)
        .map(|channel| {
            let start = channel * samples;
            Ok(flat[start..start + samples].to_vec())
        })
        .collect()
}

fn validate_probabilities(values: [f32; PRESENCE_CLASSES]) -> Result<(), String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (values.iter().sum::<f32>() - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
    {
        return Err(
            "target-sound presence output must contain normalized absent/uncertain/present probabilities"
                .into(),
        );
    }
    Ok(())
}

fn model_error(error: impl std::fmt::Display) -> String {
    format!("target-sound ONNX error: {error:#}")
}
