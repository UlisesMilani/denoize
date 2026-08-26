//! Authenticated recurrent target-speaker graph adapter.
//!
//! A causal target-speaker graph is intentionally not accepted by the generic
//! waveform adapter.  Its enrollment input, calibrated presence output, and
//! recurrent state edges all carry safety semantics that must remain explicit.

use super::tract_runtime::SharedRunnable;
use crate::{
    AcceleratorRuntime, RuntimeModelNumericalCaseV1, RuntimeModelPackage,
    RuntimeModelPackageManifestV2, RuntimeModelTensorContractV2,
};
use std::collections::HashMap;
use tract_onnx::prelude::*;

const PRESENCE_CLASSES: usize = 3;
const PRESENCE_SUM_TOLERANCE: f32 = 0.001;
const MAX_EFFECTIVE_LATENCY_MILLIS: u64 = 100;

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

#[derive(Clone, Debug)]
struct StateEdge {
    input: usize,
    output: usize,
    element_type: StateElementType,
    shape: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StateElementType {
    Float32,
    Int64,
}

#[derive(Clone, Debug)]
struct CausalTargetSpeakerGraphContract {
    mixture_input: usize,
    enrollment_input: usize,
    audio_output: usize,
    presence_output: usize,
    mixture_layout: WaveformLayout,
    enrollment_layout: WaveformLayout,
    frame_samples: usize,
    enrollment_samples: Option<usize>,
    state_edges: Vec<StateEdge>,
    sample_rate_hz: u32,
    algorithmic_latency_samples: usize,
    flush_samples: usize,
}

/// One model-rate causal inference result. Presence probabilities are ordered
/// absent, uncertain, and present.
pub(crate) struct CausalTargetSpeakerInference {
    pub audio: Vec<f32>,
    pub presence_probabilities: [f32; PRESENCE_CLASSES],
}

/// Parsed and authenticated graph template. Each active stream owns its
/// recurrent state and an independently prepared runnable.
pub(crate) struct CausalTargetSpeakerModel {
    template: InferenceModel,
    runtime: AcceleratorRuntime,
    contract: CausalTargetSpeakerGraphContract,
}

impl CausalTargetSpeakerModel {
    pub(crate) fn load_runtime_package(
        package: &RuntimeModelPackage,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let manifest = package.manifest_v2().ok_or(
            "causal target-speaker extraction requires authenticated runtime model package v2",
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
                    "failed to load causal target-speaker ONNX graph from authenticated package {}: {error:#}",
                    package.package_path().display()
                )
            })?;
        reader.finish().map_err(|error| {
            format!(
                "failed to authenticate causal target-speaker ONNX bytes from package {}: {error}",
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
        validate_sequence_vector_semantics(manifest, &vectors.cases)?;
        super::onnx::validate_v2_numerical_vectors(
            &template, manifest, profile, &vectors, runtime,
        )?;
        Ok(Self {
            template,
            runtime,
            contract,
        })
    }

    pub(crate) const fn sample_rate_hz(&self) -> u32 {
        self.contract.sample_rate_hz
    }

    pub(crate) const fn frame_samples(&self) -> usize {
        self.contract.frame_samples
    }

    pub(crate) const fn algorithmic_latency_samples(&self) -> usize {
        self.contract.algorithmic_latency_samples
    }

    pub(crate) const fn flush_samples(&self) -> usize {
        self.contract.flush_samples
    }

    pub(crate) const fn fixed_enrollment_samples(&self) -> Option<usize> {
        self.contract.enrollment_samples
    }

    pub(crate) fn start(&self, enrollment: Vec<f32>) -> Result<CausalTargetSpeakerRuntime, String> {
        if enrollment.is_empty() {
            return Err("causal target-speaker enrollment must not be empty".into());
        }
        if enrollment.iter().any(|sample| !sample.is_finite()) {
            return Err("causal target-speaker enrollment contains a non-finite sample".into());
        }
        if self
            .contract
            .enrollment_samples
            .is_some_and(|required| required != enrollment.len())
        {
            return Err(format!(
                "causal target-speaker graph requires {} enrollment samples, got {}",
                self.contract
                    .enrollment_samples
                    .expect("checked fixed enrollment"),
                enrollment.len()
            ));
        }

        let mut model = self.template.clone();
        for (index, tensor) in self
            .input_contracts(enrollment.len())?
            .into_iter()
            .enumerate()
        {
            model.set_input_fact(index, tensor).map_err(model_error)?;
        }
        for (index, tensor) in self.output_contracts()?.into_iter().enumerate() {
            model.set_output_fact(index, tensor).map_err(model_error)?;
        }
        let model = model.into_typed().map_err(model_error)?;
        let runnable = super::tract_runtime::prepare(
            model,
            self.runtime,
            "causal target-speaker extraction model",
        )?;
        let states = self
            .contract
            .state_edges
            .iter()
            .map(zero_state)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CausalTargetSpeakerRuntime {
            runnable,
            contract: self.contract.clone(),
            enrollment,
            states,
        })
    }

    fn input_contracts(&self, enrollment_samples: usize) -> Result<Vec<InferenceFact>, String> {
        let manifest_inputs = self.template.input_outlets().map_err(model_error)?.len();
        let mut facts = vec![None; manifest_inputs];
        facts[self.contract.mixture_input] = Some(
            f32::fact(
                self.contract
                    .mixture_layout
                    .shape(self.contract.frame_samples),
            )
            .into(),
        );
        facts[self.contract.enrollment_input] =
            Some(f32::fact(self.contract.enrollment_layout.shape(enrollment_samples)).into());
        for edge in &self.contract.state_edges {
            facts[edge.input] = Some(state_fact(edge));
        }
        facts
            .into_iter()
            .enumerate()
            .map(|(index, fact)| {
                fact.ok_or_else(|| {
                    format!("causal target-speaker input {index} has no closed contract")
                })
            })
            .collect()
    }

    fn output_contracts(&self) -> Result<Vec<InferenceFact>, String> {
        let manifest_outputs = self.template.output_outlets().map_err(model_error)?.len();
        let mut facts = vec![None; manifest_outputs];
        facts[self.contract.audio_output] = Some(
            f32::fact(
                self.contract
                    .mixture_layout
                    .shape(self.contract.frame_samples),
            )
            .into(),
        );
        facts[self.contract.presence_output] = Some(f32::fact(tvec!(1, PRESENCE_CLASSES)).into());
        for edge in &self.contract.state_edges {
            facts[edge.output] = Some(state_fact(edge));
        }
        facts
            .into_iter()
            .enumerate()
            .map(|(index, fact)| {
                fact.ok_or_else(|| {
                    format!("causal target-speaker output {index} has no closed contract")
                })
            })
            .collect()
    }
}

/// One active recurrent graph. The enrollment and states never leave this
/// object and are replaced by zeros on reset/drop where their concrete storage
/// permits it.
pub(crate) struct CausalTargetSpeakerRuntime {
    runnable: SharedRunnable,
    contract: CausalTargetSpeakerGraphContract,
    enrollment: Vec<f32>,
    states: Vec<Tensor>,
}

impl CausalTargetSpeakerRuntime {
    pub(crate) fn process(
        &mut self,
        mixture: &[f32],
    ) -> Result<CausalTargetSpeakerInference, String> {
        if mixture.len() != self.contract.frame_samples {
            return Err(format!(
                "causal target-speaker frame has {} samples; expected {}",
                mixture.len(),
                self.contract.frame_samples
            ));
        }
        if mixture
            .iter()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("causal target-speaker frame contains an invalid normalized sample".into());
        }
        let mixture_tensor = Tensor::from_shape(
            &self
                .contract
                .mixture_layout
                .shape(self.contract.frame_samples),
            mixture,
        )
        .map_err(model_error)?;
        let enrollment_tensor = Tensor::from_shape(
            &self.contract.enrollment_layout.shape(self.enrollment.len()),
            &self.enrollment,
        )
        .map_err(model_error)?;
        let mut inputs: TVec<TValue> = (0..(2 + self.contract.state_edges.len()))
            .map(|_| {
                Tensor::zero::<f32>(&[1])
                    .expect("one-element zero tensor")
                    .into_tvalue()
            })
            .collect();
        inputs[self.contract.mixture_input] = mixture_tensor.into_tvalue();
        inputs[self.contract.enrollment_input] = enrollment_tensor.into_tvalue();
        for (state, edge) in self.states.iter().zip(&self.contract.state_edges) {
            inputs[edge.input] = state.clone().into_tvalue();
        }
        let outputs = self.runnable.run(inputs).map_err(model_error)?;
        if outputs.len() != 2 + self.contract.state_edges.len() {
            return Err(format!(
                "causal target-speaker graph returned {} outputs; expected {}",
                outputs.len(),
                2 + self.contract.state_edges.len()
            ));
        }
        let audio_view = outputs[self.contract.audio_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        if audio_view.len() != self.contract.frame_samples {
            return Err(format!(
                "causal target-speaker graph returned {} audio samples; expected {}",
                audio_view.len(),
                self.contract.frame_samples
            ));
        }
        let audio: Vec<f32> = audio_view.iter().copied().collect();
        if audio.iter().any(|sample| !sample.is_finite()) {
            return Err("causal target-speaker graph returned non-finite audio".into());
        }
        let presence_view = outputs[self.contract.presence_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        if presence_view.len() != PRESENCE_CLASSES {
            return Err(format!(
                "causal target-speaker graph returned {} presence values; expected {PRESENCE_CLASSES}",
                presence_view.len()
            ));
        }
        let mut values = presence_view.iter().copied();
        let presence_probabilities = [
            values.next().expect("presence length checked"),
            values.next().expect("presence length checked"),
            values.next().expect("presence length checked"),
        ];
        validate_presence_probabilities(presence_probabilities)?;

        let mut next_states = Vec::with_capacity(self.states.len());
        for edge in &self.contract.state_edges {
            let mut tensor = outputs[edge.output].clone().into_tensor();
            if let Err(error) = validate_state_tensor(&tensor, edge) {
                let _ = zero_tensor(&mut tensor);
                for state in &mut next_states {
                    let _ = zero_tensor(state);
                }
                return Err(error);
            }
            next_states.push(tensor);
        }
        for state in &mut self.states {
            zero_tensor(state)?;
        }
        self.states = next_states;
        Ok(CausalTargetSpeakerInference {
            audio,
            presence_probabilities,
        })
    }

    pub(crate) fn reset(&mut self) -> Result<(), String> {
        for state in &mut self.states {
            zero_tensor(state)?;
        }
        Ok(())
    }
}

impl Drop for CausalTargetSpeakerRuntime {
    fn drop(&mut self) {
        self.enrollment.fill(0.0);
        for state in &mut self.states {
            let _ = zero_tensor(state);
        }
    }
}

fn validate_runtime_package_contract(
    manifest: &RuntimeModelPackageManifestV2,
) -> Result<CausalTargetSpeakerGraphContract, String> {
    if !matches!(
        manifest.runtime.mode.as_str(),
        "streaming" | "finite-and-streaming"
    ) || manifest.frontend.channels.policy != "independent-mono"
        || !manifest.frontend.channels.roles.is_empty()
        || manifest.frontend.channels.geometry.is_some()
        || manifest.state_pairs.is_empty()
        || manifest.tensors.inputs.len() != manifest.state_pairs.len() + 2
        || manifest.tensors.outputs.len() != manifest.state_pairs.len() + 2
    {
        return Err(
            "causal target-speaker package must declare an independent-mono streaming graph with audio, enrollment, diagnostic, and explicit recurrent-state pairs"
                .into(),
        );
    }
    if manifest.latency.frame_samples != manifest.latency.hop_samples {
        return Err("causal target-speaker v1 requires equal fixed frame and hop sizes".into());
    }
    let latency_millis = manifest
        .latency
        .algorithmic_latency_samples
        .saturating_mul(1000)
        .div_ceil(u64::from(manifest.runtime.sample_rate_hz));
    if latency_millis > MAX_EFFECTIVE_LATENCY_MILLIS {
        return Err(format!(
            "causal target-speaker signed algorithmic latency is {latency_millis} ms; maximum is {MAX_EFFECTIVE_LATENCY_MILLIS} ms"
        ));
    }

    let mixture_input = unique_role(&manifest.tensors.inputs, "audio", "input")?;
    let enrollment_input = unique_role(&manifest.tensors.inputs, "enrollment", "input")?;
    let audio_output = unique_role(&manifest.tensors.outputs, "audio", "output")?;
    let presence_output = unique_role(&manifest.tensors.outputs, "diagnostic", "output")?;
    if manifest
        .tensors
        .inputs
        .iter()
        .any(|tensor| !matches!(tensor.role.as_str(), "audio" | "enrollment" | "state"))
        || manifest
            .tensors
            .outputs
            .iter()
            .any(|tensor| !matches!(tensor.role.as_str(), "audio" | "diagnostic" | "state"))
        || manifest
            .tensors
            .inputs
            .iter()
            .chain(&manifest.tensors.outputs)
            .any(|tensor| tensor.optional)
    {
        return Err("causal target-speaker tensors must be required audio/enrollment/diagnostic/state tensors only".into());
    }

    let mixture = &manifest.tensors.inputs[mixture_input];
    let enrollment = &manifest.tensors.inputs[enrollment_input];
    let output = &manifest.tensors.outputs[audio_output];
    let presence = &manifest.tensors.outputs[presence_output];
    let mixture_layout = waveform_layout(mixture, "mixture")?;
    let enrollment_layout = waveform_layout(enrollment, "enrollment")?;
    if waveform_layout(output, "audio output")? != mixture_layout {
        return Err("causal target-speaker audio input and output layouts differ".into());
    }
    let frame_samples = usize::try_from(manifest.latency.frame_samples)
        .map_err(|_| "causal target-speaker frame size is too large".to_string())?;
    if fixed_samples(mixture) != Some(frame_samples) || fixed_samples(output) != Some(frame_samples)
    {
        return Err(
            "causal target-speaker audio input/output sample axes must equal the signed frame/hop size"
                .into(),
        );
    }
    if manifest.latency.flush_samples < manifest.latency.algorithmic_latency_samples {
        return Err("causal target-speaker flush must cover the signed algorithmic latency".into());
    }
    validate_presence_tensor(presence)?;

    let input_by_name: HashMap<_, _> = manifest
        .tensors
        .inputs
        .iter()
        .enumerate()
        .map(|(index, tensor)| (tensor.name.as_str(), (index, tensor)))
        .collect();
    let output_by_name: HashMap<_, _> = manifest
        .tensors
        .outputs
        .iter()
        .enumerate()
        .map(|(index, tensor)| (tensor.name.as_str(), (index, tensor)))
        .collect();
    let mut state_edges = Vec::with_capacity(manifest.state_pairs.len());
    for pair in &manifest.state_pairs {
        let (input, input_contract) =
            input_by_name
                .get(pair.input.as_str())
                .copied()
                .ok_or_else(|| {
                    format!(
                        "causal target-speaker state input {} is missing",
                        pair.input
                    )
                })?;
        let (output, output_contract) = output_by_name
            .get(pair.output.as_str())
            .copied()
            .ok_or_else(|| {
                format!(
                    "causal target-speaker state output {} is missing",
                    pair.output
                )
            })?;
        let shape = fixed_shape(input_contract, "state")?;
        if fixed_shape(output_contract, "state")? != shape {
            return Err("causal target-speaker paired state shapes differ".into());
        }
        let element_type = match input_contract.element_type.as_str() {
            "float32" => StateElementType::Float32,
            "int64" => StateElementType::Int64,
            _ => return Err("causal target-speaker state type is unsupported".into()),
        };
        state_edges.push(StateEdge {
            input,
            output,
            element_type,
            shape,
        });
    }
    state_edges.sort_by_key(|edge| edge.input);
    Ok(CausalTargetSpeakerGraphContract {
        mixture_input,
        enrollment_input,
        audio_output,
        presence_output,
        mixture_layout,
        enrollment_layout,
        frame_samples,
        enrollment_samples: fixed_samples(enrollment),
        state_edges,
        sample_rate_hz: manifest.runtime.sample_rate_hz,
        algorithmic_latency_samples: usize::try_from(manifest.latency.algorithmic_latency_samples)
            .map_err(|_| "causal target-speaker latency is too large".to_string())?,
        flush_samples: usize::try_from(manifest.latency.flush_samples)
            .map_err(|_| "causal target-speaker flush is too large".to_string())?,
    })
}

fn validate_sequence_vector_semantics(
    manifest: &RuntimeModelPackageManifestV2,
    cases: &[RuntimeModelNumericalCaseV1],
) -> Result<(), String> {
    let by_id: HashMap<_, _> = cases.iter().map(|case| (case.id.as_str(), case)).collect();
    let reset = by_id
        .get("causal-reset")
        .ok_or("causal target-speaker vectors omit causal-reset")?;
    let recurrent = by_id
        .get("causal-recurrent")
        .ok_or("causal target-speaker vectors omit causal-recurrent")?;
    let flush = by_id
        .get("causal-flush")
        .ok_or("causal target-speaker vectors omit causal-flush")?;
    let state_names: Vec<_> = manifest
        .tensors
        .inputs
        .iter()
        .filter(|tensor| tensor.role == "state")
        .map(|tensor| tensor.name.as_str())
        .collect();
    if !state_names.iter().all(|name| {
        reset
            .inputs
            .iter()
            .find(|tensor| tensor.name == *name)
            .is_some_and(|tensor| tensor.values.iter().all(|value| *value == 0.0))
    }) {
        return Err(
            "causal-reset must authenticate zero initialization for every recurrent state".into(),
        );
    }
    if !state_names.iter().any(|name| {
        recurrent
            .inputs
            .iter()
            .find(|tensor| tensor.name == *name)
            .is_some_and(|tensor| tensor.values.iter().any(|value| *value != 0.0))
    }) {
        return Err("causal-recurrent must exercise a non-zero recurrent state".into());
    }
    let mixture_name = manifest
        .tensors
        .inputs
        .iter()
        .find(|tensor| tensor.role == "audio")
        .expect("v2 manifest has audio input")
        .name
        .as_str();
    if !flush
        .inputs
        .iter()
        .find(|tensor| tensor.name == mixture_name)
        .is_some_and(|tensor| tensor.values.iter().all(|value| *value == 0.0))
    {
        return Err("causal-flush must authenticate a zero-audio flush frame".into());
    }
    Ok(())
}

fn unique_role(
    tensors: &[RuntimeModelTensorContractV2],
    role: &str,
    kind: &str,
) -> Result<usize, String> {
    let matches: Vec<_> = tensors
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.role == role)
        .map(|(index, _)| index)
        .collect();
    if matches.len() != 1 {
        return Err(format!(
            "causal target-speaker graph requires exactly one {role} {kind}"
        ));
    }
    Ok(matches[0])
}

fn waveform_layout(
    tensor: &RuntimeModelTensorContractV2,
    label: &str,
) -> Result<WaveformLayout, String> {
    if tensor.element_type != "float32" {
        return Err(format!("causal target-speaker {label} must be float32"));
    }
    let kinds: Vec<_> = tensor.axes.iter().map(|axis| axis.kind.as_str()).collect();
    match kinds.as_slice() {
        ["batch", "sample"] if tensor.axes[0].fixed == Some(1) => Ok(WaveformLayout::BatchSamples),
        ["batch", "channel", "sample"]
            if tensor.axes[0].fixed == Some(1) && tensor.axes[1].fixed == Some(1) =>
        {
            Ok(WaveformLayout::BatchChannelsSamples)
        }
        _ => Err(format!(
            "causal target-speaker {label} must use [batch=1,sample] or [batch=1,channel=1,sample]"
        )),
    }
}

fn fixed_samples(tensor: &RuntimeModelTensorContractV2) -> Option<usize> {
    tensor
        .axes
        .iter()
        .find(|axis| axis.kind == "sample")
        .and_then(|axis| axis.fixed)
        .and_then(|value| usize::try_from(value).ok())
}

fn fixed_shape(tensor: &RuntimeModelTensorContractV2, label: &str) -> Result<Vec<usize>, String> {
    tensor
        .axes
        .iter()
        .map(|axis| {
            axis.fixed
                .ok_or_else(|| format!("causal target-speaker {label} axes must be fixed"))
                .and_then(|value| {
                    usize::try_from(value)
                        .map_err(|_| format!("causal target-speaker {label} axis is too large"))
                })
        })
        .collect()
}

fn validate_presence_tensor(tensor: &RuntimeModelTensorContractV2) -> Result<(), String> {
    let valid = tensor.element_type == "float32"
        && tensor.axes.len() == 2
        && tensor.axes[0].kind == "batch"
        && tensor.axes[0].fixed == Some(1)
        && tensor.axes[1].kind == "feature"
        && tensor.axes[1].fixed == Some(PRESENCE_CLASSES as u64);
    if valid {
        Ok(())
    } else {
        Err("causal target-speaker diagnostic output must be float32 [batch=1,feature=3]".into())
    }
}

fn validate_presence_probabilities(values: [f32; PRESENCE_CLASSES]) -> Result<(), String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (values.iter().sum::<f32>() - 1.0).abs() > PRESENCE_SUM_TOLERANCE
    {
        return Err(
            "causal target-speaker presence output must contain normalized absent/uncertain/present probabilities"
                .into(),
        );
    }
    Ok(())
}

fn state_fact(edge: &StateEdge) -> InferenceFact {
    match edge.element_type {
        StateElementType::Float32 => f32::fact(edge.shape.clone()).into(),
        StateElementType::Int64 => i64::fact(edge.shape.clone()).into(),
    }
}

fn zero_state(edge: &StateEdge) -> Result<Tensor, String> {
    match edge.element_type {
        StateElementType::Float32 => Tensor::zero::<f32>(&edge.shape).map_err(model_error),
        StateElementType::Int64 => Tensor::zero::<i64>(&edge.shape).map_err(model_error),
    }
}

fn validate_state_tensor(tensor: &Tensor, edge: &StateEdge) -> Result<(), String> {
    if tensor.shape() != edge.shape.as_slice() {
        return Err(format!(
            "causal target-speaker recurrent state shape changed from {:?} to {:?}",
            edge.shape,
            tensor.shape()
        ));
    }
    match edge.element_type {
        StateElementType::Float32 => {
            let view = tensor.to_plain_array_view::<f32>().map_err(model_error)?;
            if view.iter().any(|value| !value.is_finite()) {
                return Err(
                    "causal target-speaker recurrent state contains a non-finite value".into(),
                );
            }
        }
        StateElementType::Int64 => {
            tensor.to_plain_array_view::<i64>().map_err(model_error)?;
        }
    }
    Ok(())
}

fn zero_tensor(tensor: &mut Tensor) -> Result<(), String> {
    match tensor.datum_type() {
        DatumType::F32 => tensor
            .to_plain_array_view_mut::<f32>()
            .map_err(model_error)?
            .fill(0.0),
        DatumType::I64 => tensor
            .to_plain_array_view_mut::<i64>()
            .map_err(model_error)?
            .fill(0),
        _ => return Err("causal target-speaker recurrent state type changed".into()),
    }
    Ok(())
}

fn model_error(error: impl std::fmt::Display) -> String {
    format!("causal target-speaker ONNX error: {error:#}")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use prost::Message;
    use sha2::Digest as _;
    use tract_onnx::pb::{
        attribute_proto, tensor_proto, tensor_shape_proto, type_proto, AttributeProto, GraphProto,
        ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto,
        ValueInfoProto,
    };

    #[test]
    fn signed_recurrent_graph_runs_and_resets() {
        let (_directory, package) = fixture_package();
        let model =
            CausalTargetSpeakerModel::load_runtime_package(&package, AcceleratorRuntime::Cpu)
                .unwrap();
        assert_eq!(model.sample_rate_hz(), 16_000);
        assert_eq!(model.frame_samples(), 4);
        assert_eq!(model.algorithmic_latency_samples(), 4);
        assert_eq!(model.flush_samples(), 4);
        let mut stream = model.start(vec![0.1, -0.1, 0.2]).unwrap();
        let result = stream.process(&[-0.5, 0.0, 0.25, 0.75]).unwrap();
        assert_eq!(result.audio, [-0.5, 0.0, 0.25, 0.75]);
        assert_eq!(result.presence_probabilities, [0.0, 0.0, 1.0]);
        stream.reset().unwrap();
        let reset = stream.process(&[0.0; 4]).unwrap();
        assert_eq!(reset.audio, [0.0; 4]);
    }

    pub(crate) fn fixture_package() -> (tempfile::TempDir, RuntimeModelPackage) {
        let directory = tempfile::tempdir().unwrap();
        let package_path = directory.path().join("causal-target-speaker.dmp");
        let mut model_bytes = Vec::new();
        causal_identity_model().encode(&mut model_bytes).unwrap();
        let components = fixture_components(model_bytes);
        let manifest = manifest(&components);
        let package =
            RuntimeModelPackage::for_onnx_v2_contract_test(package_path, manifest, components);
        (directory, package)
    }

    #[test]
    fn rejects_nonstreaming_missing_sequences_and_excess_latency() {
        let components = fixture_components(vec![0]);
        let mut invalid = manifest(&components);
        invalid.runtime.mode = "finite".into();
        assert!(validate_runtime_package_contract(&invalid)
            .unwrap_err()
            .contains("streaming graph"));

        let mut invalid = manifest(&components);
        invalid.latency.algorithmic_latency_samples = 1_601;
        assert!(validate_runtime_package_contract(&invalid)
            .unwrap_err()
            .contains("maximum is 100 ms"));

        let vectors: crate::RuntimeModelNumericalVectorsV1 =
            serde_json::from_slice(&components[3]).unwrap();
        assert!(validate_sequence_vector_semantics(&manifest(&components), &vectors.cases).is_ok());
        assert!(
            validate_sequence_vector_semantics(&manifest(&components), &vectors.cases[..2])
                .unwrap_err()
                .contains("causal-flush")
        );
    }

    fn fixture_components(model: Vec<u8>) -> Vec<Vec<u8>> {
        let vectors = serde_json::to_vec(&serde_json::json!({
            "schema": "denoize-runtime-model-numerical-vectors-v1",
            "profile_id": "fp32",
            "cases": [
                vector_case(
                    "causal-reset",
                    vec![-0.5, 0.0, 0.25, 0.75],
                    vec![0.0, 0.0]
                ),
                vector_case(
                    "causal-recurrent",
                    vec![0.25, -0.25, 0.5, -0.5],
                    vec![1.0, -1.0]
                ),
                vector_case(
                    "causal-flush",
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.5, -0.5]
                )
            ]
        }))
        .unwrap();
        vec![
            model,
            b"fixture license".to_vec(),
            br#"{"schema":"fixture-provenance-v1"}"#.to_vec(),
            vectors,
        ]
    }

    fn vector_case(id: &str, audio: Vec<f64>, state: Vec<f64>) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "inputs": [
                {
                    "name": "mixture",
                    "element_type": "float32",
                    "shape": [1, 4],
                    "values": audio
                },
                {
                    "name": "enrollment",
                    "element_type": "float32",
                    "shape": [1, 3],
                    "values": [0.1, -0.1, 0.2]
                },
                {
                    "name": "state_in",
                    "element_type": "float32",
                    "shape": [1, 2],
                    "values": state
                }
            ],
            "outputs": [
                {
                    "name": "extracted",
                    "element_type": "float32",
                    "shape": [1, 4],
                    "values": audio
                },
                {
                    "name": "target_presence_probabilities",
                    "element_type": "float32",
                    "shape": [1, 3],
                    "values": [0.0, 0.0, 1.0]
                },
                {
                    "name": "state_out",
                    "element_type": "float32",
                    "shape": [1, 2],
                    "values": state
                }
            ],
            "tolerance": { "absolute": 0.000001, "relative": 0.000001 }
        })
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
                    fixed: Some(4),
                },
            ]
        };
        let enrollment_axes = || {
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
        let state_axes = || {
            vec![
                crate::RuntimeModelAxisContractV2 {
                    name: "batch".into(),
                    kind: "batch".into(),
                    fixed: Some(1),
                },
                crate::RuntimeModelAxisContractV2 {
                    name: "memory".into(),
                    kind: "state".into(),
                    fixed: Some(2),
                },
            ]
        };
        RuntimeModelPackageManifestV2 {
            schema: crate::RUNTIME_MODEL_PACKAGE_SCHEMA_V2.into(),
            format_version: crate::RUNTIME_MODEL_PACKAGE_VERSION_V2,
            package_id: "denoize.test.causal-target-speaker".into(),
            package_revision: "1".into(),
            signing_key_id: "0000000000000001".into(),
            runtime: crate::RuntimeModelRuntimeContractV2 {
                kind: "onnx-audio-graph-v2".into(),
                sample_rate_hz: 16_000,
                mode: "streaming".into(),
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
                        axes: enrollment_axes(),
                        optional: false,
                        state_id: None,
                    },
                    crate::RuntimeModelTensorContractV2 {
                        name: "state_in".into(),
                        role: "state".into(),
                        element_type: "float32".into(),
                        axes: state_axes(),
                        optional: false,
                        state_id: Some("memory".into()),
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
                    crate::RuntimeModelTensorContractV2 {
                        name: "state_out".into(),
                        role: "state".into(),
                        element_type: "float32".into(),
                        axes: state_axes(),
                        optional: false,
                        state_id: Some("memory".into()),
                    },
                ],
            },
            state_pairs: vec![crate::RuntimeModelStatePairContractV2 {
                id: "memory".into(),
                input: "state_in".into(),
                output: "state_out".into(),
                initialization: "zeros".into(),
            }],
            latency: crate::RuntimeModelLatencyContractV2 {
                frame_samples: 4,
                hop_samples: 4,
                left_context_samples: 0,
                right_context_samples: 0,
                lookahead_samples: 0,
                algorithmic_latency_samples: 4,
                flush_samples: 4,
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
                source_repository: "https://example.invalid/causal-target-speaker".into(),
                source_revision: "0123456789abcdef".into(),
                source_sha256: "0".repeat(64),
                source_license_spdx: "MIT".into(),
                checkpoint_source: "https://example.invalid/causal-target-speaker.ckpt".into(),
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

    fn causal_identity_model() -> ModelProto {
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
                    value_info("mixture", vec![dimension(1), dimension(4)]),
                    value_info(
                        "enrollment",
                        vec![dimension(1), dimension_parameter("enrollment_samples")],
                    ),
                    value_info("state_in", vec![dimension(1), dimension(2)]),
                ],
                output: vec![
                    value_info("extracted", vec![dimension(1), dimension(4)]),
                    value_info(
                        "target_presence_probabilities",
                        vec![dimension(1), dimension(3)],
                    ),
                    value_info("state_out", vec![dimension(1), dimension(2)]),
                ],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn dimension(value: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
            denotation: String::new(),
        }
    }

    fn dimension_parameter(value: &str) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(value.into())),
            denotation: String::new(),
        }
    }
}
