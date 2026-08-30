//! Authenticated recurrent closed-catalog target-sound graph adapter.
//!
//! This adapter keeps the query, target, residual, calibrated presence, and
//! recurrent-state semantics explicit. It is deliberately separate from both
//! the generic waveform adapter and the finite target-sound adapter.

use super::tract_runtime::SharedRunnable;
use crate::{
    AcceleratorRuntime, RuntimeModelNumericalCaseV1, RuntimeModelPackage,
    RuntimeModelPackageManifestV2, RuntimeModelTensorContractV2,
};
use std::collections::HashMap;
use tract_onnx::prelude::*;

const PRESENCE_CLASSES: usize = 3;
const PRESENCE_SUM_TOLERANCE: f32 = 0.001;
const VECTOR_RECOMBINATION_TOLERANCE: f64 = 0.000_001;
const MAX_EFFECTIVE_LATENCY_MILLIS: u64 = 100;
const MAX_QUERY_CLASSES: usize = 4096;

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
struct CausalTargetSoundGraphContract {
    channels: usize,
    frame_samples: usize,
    query_classes: usize,
    audio_input: usize,
    query_input: usize,
    target_output: usize,
    residual_output: usize,
    presence_output: usize,
    state_edges: Vec<StateEdge>,
    sample_rate_hz: u32,
    algorithmic_latency_samples: usize,
    flush_samples: usize,
}

/// One model-rate causal inference result. Presence probabilities are ordered
/// absent, uncertain, and present.
pub(crate) struct CausalTargetSoundInference {
    pub target: Vec<Vec<f32>>,
    pub residual: Vec<Vec<f32>>,
    pub presence_probabilities: [f32; PRESENCE_CLASSES],
}

/// A typed recurrent-state value used by the portable stream snapshot layer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CausalTargetSoundStateValue {
    Float32 { shape: Vec<usize>, values: Vec<f32> },
    Int64 { shape: Vec<usize>, values: Vec<i64> },
}

/// A complete recurrent-state image. The public layer binds this image to the
/// authenticated model, catalog, selected class, and stream generation.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CausalTargetSoundBackendSnapshot {
    pub states: Vec<CausalTargetSoundStateValue>,
}

/// Parsed and authenticated graph template. Each active stream owns its
/// recurrent state and independently prepared runnable.
pub(crate) struct CausalTargetSoundModel {
    template: InferenceModel,
    runtime: AcceleratorRuntime,
    contract: CausalTargetSoundGraphContract,
}

impl CausalTargetSoundModel {
    pub(crate) fn load_runtime_package(
        package: &RuntimeModelPackage,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let manifest = package.manifest_v2().ok_or(
            "causal target-sound extraction requires authenticated runtime model package v2",
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
                    "failed to load causal target-sound ONNX graph from authenticated package {}: {error:#}",
                    package.package_path().display()
                )
            })?;
        reader.finish().map_err(|error| {
            format!(
                "failed to authenticate causal target-sound ONNX bytes from package {}: {error}",
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

    pub(crate) const fn channels(&self) -> usize {
        self.contract.channels
    }

    pub(crate) const fn frame_samples(&self) -> usize {
        self.contract.frame_samples
    }

    pub(crate) const fn query_classes(&self) -> usize {
        self.contract.query_classes
    }

    pub(crate) const fn algorithmic_latency_samples(&self) -> usize {
        self.contract.algorithmic_latency_samples
    }

    pub(crate) const fn flush_samples(&self) -> usize {
        self.contract.flush_samples
    }

    pub(crate) fn start(&self, class_index: usize) -> Result<CausalTargetSoundRuntime, String> {
        if class_index >= self.contract.query_classes {
            return Err(format!(
                "causal target-sound class index {class_index} is outside the authenticated {}-class catalog",
                self.contract.query_classes
            ));
        }
        let mut query = vec![0.0_f32; self.contract.query_classes];
        query[class_index] = 1.0;

        let mut model = self.template.clone();
        for (index, fact) in self.input_contracts()?.into_iter().enumerate() {
            model.set_input_fact(index, fact).map_err(model_error)?;
        }
        for (index, fact) in self.output_contracts()?.into_iter().enumerate() {
            model.set_output_fact(index, fact).map_err(model_error)?;
        }
        let model = model.into_typed().map_err(model_error)?;
        let runnable = super::tract_runtime::prepare(
            model,
            self.runtime,
            "causal target-sound extraction model",
        )?;
        let states = self
            .contract
            .state_edges
            .iter()
            .map(zero_state)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CausalTargetSoundRuntime {
            runnable,
            contract: self.contract.clone(),
            query,
            states,
        })
    }

    fn input_contracts(&self) -> Result<Vec<InferenceFact>, String> {
        let count = self.template.input_outlets().map_err(model_error)?.len();
        let mut facts = vec![None; count];
        facts[self.contract.audio_input] = Some(
            f32::fact(tvec!(
                1,
                self.contract.channels,
                self.contract.frame_samples
            ))
            .into(),
        );
        facts[self.contract.query_input] =
            Some(f32::fact(tvec!(1, self.contract.query_classes)).into());
        for edge in &self.contract.state_edges {
            facts[edge.input] = Some(state_fact(edge));
        }
        close_facts(facts, "input")
    }

    fn output_contracts(&self) -> Result<Vec<InferenceFact>, String> {
        let count = self.template.output_outlets().map_err(model_error)?.len();
        let mut facts = vec![None; count];
        let audio: InferenceFact = f32::fact(tvec!(
            1,
            self.contract.channels,
            self.contract.frame_samples
        ))
        .into();
        facts[self.contract.target_output] = Some(audio.clone());
        facts[self.contract.residual_output] = Some(audio);
        facts[self.contract.presence_output] = Some(f32::fact(tvec!(1, PRESENCE_CLASSES)).into());
        for edge in &self.contract.state_edges {
            facts[edge.output] = Some(state_fact(edge));
        }
        close_facts(facts, "output")
    }
}

impl std::fmt::Debug for CausalTargetSoundModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CausalTargetSoundModel")
            .field("runtime", &self.runtime)
            .field("contract", &self.contract)
            .finish_non_exhaustive()
    }
}

fn close_facts(
    facts: Vec<Option<InferenceFact>>,
    kind: &str,
) -> Result<Vec<InferenceFact>, String> {
    facts
        .into_iter()
        .enumerate()
        .map(|(index, fact)| {
            fact.ok_or_else(|| format!("causal target-sound {kind} {index} has no closed contract"))
        })
        .collect()
}

/// One active recurrent graph. Query and states are zeroed when dropped.
pub(crate) struct CausalTargetSoundRuntime {
    runnable: SharedRunnable,
    contract: CausalTargetSoundGraphContract,
    query: Vec<f32>,
    states: Vec<Tensor>,
}

impl CausalTargetSoundRuntime {
    pub(crate) fn process(
        &mut self,
        mixture: &[Vec<f32>],
    ) -> Result<CausalTargetSoundInference, String> {
        if mixture.len() != self.contract.channels
            || mixture
                .iter()
                .any(|channel| channel.len() != self.contract.frame_samples)
        {
            return Err(format!(
                "causal target-sound frame must contain {} channels by {} samples",
                self.contract.channels, self.contract.frame_samples
            ));
        }
        if mixture
            .iter()
            .flatten()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("causal target-sound frame contains an invalid normalized sample".into());
        }
        let audio = mixture.iter().flatten().copied().collect::<Vec<_>>();
        let mixture_tensor = Tensor::from_shape(
            &tvec!(1, self.contract.channels, self.contract.frame_samples),
            &audio,
        )
        .map_err(model_error)?;
        let query_tensor = Tensor::from_shape(&tvec!(1, self.contract.query_classes), &self.query)
            .map_err(model_error)?;
        let mut inputs: TVec<TValue> = (0..(2 + self.contract.state_edges.len()))
            .map(|_| {
                Tensor::zero::<f32>(&[1])
                    .expect("one-element zero tensor")
                    .into_tvalue()
            })
            .collect();
        inputs[self.contract.audio_input] = mixture_tensor.into_tvalue();
        inputs[self.contract.query_input] = query_tensor.into_tvalue();
        for (state, edge) in self.states.iter().zip(&self.contract.state_edges) {
            inputs[edge.input] = state.clone().into_tvalue();
        }

        let outputs = self.runnable.run(inputs).map_err(model_error)?;
        let expected_outputs = 3 + self.contract.state_edges.len();
        if outputs.len() != expected_outputs {
            return Err(format!(
                "causal target-sound graph returned {} outputs; expected {expected_outputs}",
                outputs.len()
            ));
        }
        let target = decode_audio(
            &outputs[self.contract.target_output],
            self.contract.channels,
            self.contract.frame_samples,
            "target",
        )?;
        let residual = decode_audio(
            &outputs[self.contract.residual_output],
            self.contract.channels,
            self.contract.frame_samples,
            "residual",
        )?;
        let presence = outputs[self.contract.presence_output]
            .to_plain_array_view::<f32>()
            .map_err(model_error)?;
        if presence.len() != PRESENCE_CLASSES {
            return Err(format!(
                "causal target-sound graph returned {} presence values; expected {PRESENCE_CLASSES}",
                presence.len()
            ));
        }
        let mut values = presence.iter().copied();
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
        Ok(CausalTargetSoundInference {
            target,
            residual,
            presence_probabilities,
        })
    }

    pub(crate) fn snapshot(&self) -> Result<CausalTargetSoundBackendSnapshot, String> {
        let states = self
            .states
            .iter()
            .zip(&self.contract.state_edges)
            .map(|(tensor, edge)| snapshot_state(tensor, edge))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CausalTargetSoundBackendSnapshot { states })
    }

    pub(crate) fn restore(
        &mut self,
        snapshot: &CausalTargetSoundBackendSnapshot,
    ) -> Result<(), String> {
        if snapshot.states.len() != self.contract.state_edges.len() {
            return Err(
                "causal target-sound snapshot state count differs from the model contract".into(),
            );
        }
        let replacement = snapshot
            .states
            .iter()
            .zip(&self.contract.state_edges)
            .map(|(value, edge)| restore_state(value, edge))
            .collect::<Result<Vec<_>, _>>()?;
        for state in &mut self.states {
            zero_tensor(state)?;
        }
        self.states = replacement;
        Ok(())
    }

    pub(crate) fn reset(&mut self) -> Result<(), String> {
        for state in &mut self.states {
            zero_tensor(state)?;
        }
        Ok(())
    }
}

impl Drop for CausalTargetSoundRuntime {
    fn drop(&mut self) {
        self.query.fill(0.0);
        for state in &mut self.states {
            let _ = zero_tensor(state);
        }
    }
}

fn validate_runtime_package_contract(
    manifest: &RuntimeModelPackageManifestV2,
) -> Result<CausalTargetSoundGraphContract, String> {
    if !matches!(
        manifest.runtime.mode.as_str(),
        "streaming" | "finite-and-streaming"
    ) || !(8_000..=192_000).contains(&manifest.runtime.sample_rate_hz)
        || manifest.frontend.channels.policy != "program-multichannel"
        || manifest.frontend.channels.geometry.is_some()
        || manifest.state_pairs.is_empty()
        || manifest.tensors.inputs.len() != manifest.state_pairs.len() + 2
        || manifest.tensors.outputs.len() != manifest.state_pairs.len() + 3
    {
        return Err(
            "causal target-sound package must declare a recurrent program-multichannel streaming graph with audio/query inputs, target/residual/presence outputs, and explicit state pairs"
                .into(),
        );
    }
    if manifest.latency.frame_samples == 0
        || manifest.latency.frame_samples != manifest.latency.hop_samples
    {
        return Err(
            "causal target-sound v1 requires equal non-zero fixed frame and hop sizes".into(),
        );
    }
    if manifest.latency.flush_samples < manifest.latency.algorithmic_latency_samples {
        return Err("causal target-sound flush must cover the signed algorithmic latency".into());
    }
    let latency_millis = manifest
        .latency
        .algorithmic_latency_samples
        .saturating_mul(1000)
        .div_ceil(u64::from(manifest.runtime.sample_rate_hz));
    if latency_millis > MAX_EFFECTIVE_LATENCY_MILLIS {
        return Err(format!(
            "causal target-sound signed algorithmic latency is {latency_millis} ms; maximum is {MAX_EFFECTIVE_LATENCY_MILLIS} ms"
        ));
    }

    let audio_input = unique_role(&manifest.tensors.inputs, "audio", "input")?;
    let query_input = unique_role(&manifest.tensors.inputs, "query", "input")?;
    let target_output = unique_role(&manifest.tensors.outputs, "audio", "output")?;
    let residual_output = unique_role(&manifest.tensors.outputs, "residual", "output")?;
    let presence_output = unique_role(&manifest.tensors.outputs, "diagnostic", "output")?;
    if manifest
        .tensors
        .inputs
        .iter()
        .any(|tensor| !matches!(tensor.role.as_str(), "audio" | "query" | "state"))
        || manifest.tensors.outputs.iter().any(|tensor| {
            !matches!(
                tensor.role.as_str(),
                "audio" | "residual" | "diagnostic" | "state"
            )
        })
        || manifest
            .tensors
            .inputs
            .iter()
            .chain(&manifest.tensors.outputs)
            .any(|tensor| tensor.optional)
    {
        return Err("causal target-sound graph contains a tensor outside its closed required semantic roles".into());
    }

    let input_shape = exact_axes(
        &manifest.tensors.inputs[audio_input],
        &[("batch", Some(1)), ("channel", None), ("sample", None)],
        "audio input",
    )?;
    let channels = input_shape[1];
    let frame_samples = input_shape[2];
    if !(1..=2).contains(&channels) || !(1..=262_144).contains(&frame_samples) {
        return Err("causal target-sound frame geometry is outside bounded limits".into());
    }
    validate_channel_roles(manifest, channels)?;
    if manifest.latency.frame_samples != frame_samples as u64 {
        return Err(
            "causal target-sound audio sample axis must equal the signed frame/hop size".into(),
        );
    }
    let query_shape = exact_axes(
        &manifest.tensors.inputs[query_input],
        &[("batch", Some(1)), ("feature", None)],
        "query input",
    )?;
    let query_classes = query_shape[1];
    if !(2..=MAX_QUERY_CLASSES).contains(&query_classes) {
        return Err("causal target-sound query catalog must contain 2..=4096 classes".into());
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
                "causal target-sound {label} must preserve exact input geometry"
            ));
        }
    }
    exact_axes(
        &manifest.tensors.outputs[presence_output],
        &[("batch", Some(1)), ("feature", Some(PRESENCE_CLASSES))],
        "presence output",
    )?;
    if manifest.tensors.outputs[target_output].name != "target"
        || manifest.tensors.outputs[residual_output].name != "residual"
        || manifest.tensors.outputs[presence_output].name != "presence"
    {
        return Err(
            "causal target-sound outputs must be named target, residual, and presence".into(),
        );
    }

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
        if pair.initialization != "zeros" {
            return Err("causal target-sound recurrent states must use zero initialization".into());
        }
        let (input, input_contract) = input_by_name
            .get(pair.input.as_str())
            .copied()
            .ok_or_else(|| format!("causal target-sound state input {} is missing", pair.input))?;
        let (output, output_contract) = output_by_name
            .get(pair.output.as_str())
            .copied()
            .ok_or_else(|| {
                format!(
                    "causal target-sound state output {} is missing",
                    pair.output
                )
            })?;
        let shape = fixed_shape(input_contract, "state")?;
        if fixed_shape(output_contract, "state")? != shape
            || input_contract.element_type != output_contract.element_type
        {
            return Err("causal target-sound paired state shape or type differs".into());
        }
        let element_type = match input_contract.element_type.as_str() {
            "float32" => StateElementType::Float32,
            "int64" => StateElementType::Int64,
            _ => return Err("causal target-sound state type is unsupported".into()),
        };
        state_edges.push(StateEdge {
            input,
            output,
            element_type,
            shape,
        });
    }
    state_edges.sort_by_key(|edge| edge.input);
    Ok(CausalTargetSoundGraphContract {
        channels,
        frame_samples,
        query_classes,
        audio_input,
        query_input,
        target_output,
        residual_output,
        presence_output,
        state_edges,
        sample_rate_hz: manifest.runtime.sample_rate_hz,
        algorithmic_latency_samples: usize::try_from(manifest.latency.algorithmic_latency_samples)
            .map_err(|_| "causal target-sound latency is too large".to_string())?,
        flush_samples: usize::try_from(manifest.latency.flush_samples)
            .map_err(|_| "causal target-sound flush is too large".to_string())?,
    })
}

fn validate_sequence_vector_semantics(
    manifest: &RuntimeModelPackageManifestV2,
    cases: &[RuntimeModelNumericalCaseV1],
) -> Result<(), String> {
    let by_id: HashMap<_, _> = cases.iter().map(|case| (case.id.as_str(), case)).collect();
    let reset = by_id
        .get("causal-reset")
        .ok_or("causal target-sound vectors omit causal-reset")?;
    let recurrent = by_id
        .get("causal-recurrent")
        .ok_or("causal target-sound vectors omit causal-recurrent")?;
    let flush = by_id
        .get("causal-flush")
        .ok_or("causal target-sound vectors omit causal-flush")?;
    let state_names = manifest
        .tensors
        .inputs
        .iter()
        .filter(|tensor| tensor.role == "state")
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
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

    let audio_name = role_name(&manifest.tensors.inputs, "audio")?;
    let query_name = role_name(&manifest.tensors.inputs, "query")?;
    let target_name = role_name(&manifest.tensors.outputs, "audio")?;
    let residual_name = role_name(&manifest.tensors.outputs, "residual")?;
    let query_classes = fixed_feature(
        manifest
            .tensors
            .inputs
            .iter()
            .find(|tensor| tensor.name == query_name)
            .expect("query role name came from manifest"),
    )?;
    let mut selected_query = None;
    for case in [reset, recurrent, flush] {
        let query = case
            .inputs
            .iter()
            .find(|tensor| tensor.name == query_name)
            .ok_or_else(|| format!("{} omits the closed-catalog query", case.id))?;
        if query.values.len() != query_classes
            || query
                .values
                .iter()
                .any(|value| *value != 0.0 && *value != 1.0)
            || query.values.iter().filter(|value| **value == 1.0).count() != 1
        {
            return Err(format!(
                "{} must authenticate one exact one-hot query",
                case.id
            ));
        }
        let index = query
            .values
            .iter()
            .position(|value| *value == 1.0)
            .expect("one-hot count checked");
        if selected_query
            .replace(index)
            .is_some_and(|prior| prior != index)
        {
            return Err("causal target-sound sequence vectors change query class".into());
        }
        validate_vector_recombination(case, audio_name, target_name, residual_name)?;
    }
    if !flush
        .inputs
        .iter()
        .find(|tensor| tensor.name == audio_name)
        .is_some_and(|tensor| tensor.values.iter().all(|value| *value == 0.0))
    {
        return Err("causal-flush must authenticate a zero-audio flush frame".into());
    }
    Ok(())
}

fn validate_vector_recombination(
    case: &RuntimeModelNumericalCaseV1,
    audio_name: &str,
    target_name: &str,
    residual_name: &str,
) -> Result<(), String> {
    let values = |name: &str, inputs: bool| {
        let tensors = if inputs { &case.inputs } else { &case.outputs };
        tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(|tensor| tensor.values.as_slice())
            .ok_or_else(|| format!("{} omits tensor {name}", case.id))
    };
    let audio = values(audio_name, true)?;
    let target = values(target_name, false)?;
    let residual = values(residual_name, false)?;
    if audio.len() != target.len()
        || audio.len() != residual.len()
        || audio
            .iter()
            .zip(target)
            .zip(residual)
            .any(|((&input, &target), &residual)| {
                (input - target - residual).abs() > VECTOR_RECOMBINATION_TOLERANCE
            })
    {
        return Err(format!(
            "{} does not authenticate target + residual = input",
            case.id
        ));
    }
    Ok(())
}

fn role_name<'a>(
    tensors: &'a [RuntimeModelTensorContractV2],
    role: &str,
) -> Result<&'a str, String> {
    let index = unique_role(tensors, role, "tensor")?;
    Ok(tensors[index].name.as_str())
}

fn fixed_feature(tensor: &RuntimeModelTensorContractV2) -> Result<usize, String> {
    tensor
        .axes
        .iter()
        .find(|axis| axis.kind == "feature")
        .and_then(|axis| axis.fixed)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "causal target-sound query feature axis must be fixed".into())
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
        Err(
            "causal target-sound package requires exact mono-center or ordered stereo L/R roles"
                .into(),
        )
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
            "causal target-sound graph requires exactly one {role} {kind}"
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
            "causal target-sound {label} must be required float32 with {} axes",
            expected.len()
        ));
    }
    tensor
        .axes
        .iter()
        .zip(expected)
        .map(|(axis, (kind, fixed))| {
            if axis.kind != *kind {
                return Err(format!(
                    "causal target-sound {label} has the wrong axis order"
                ));
            }
            let value = axis
                .fixed
                .ok_or_else(|| format!("causal target-sound {label} axes must be fixed"))?;
            let value = usize::try_from(value)
                .map_err(|_| format!("causal target-sound {label} axis is too large"))?;
            if fixed.is_some_and(|required| required != value) {
                return Err(format!(
                    "causal target-sound {label} fixed axis differs from contract"
                ));
            }
            Ok(value)
        })
        .collect()
}

fn fixed_shape(tensor: &RuntimeModelTensorContractV2, label: &str) -> Result<Vec<usize>, String> {
    tensor
        .axes
        .iter()
        .map(|axis| {
            axis.fixed
                .ok_or_else(|| format!("causal target-sound {label} axes must be fixed"))
                .and_then(|value| {
                    usize::try_from(value)
                        .map_err(|_| format!("causal target-sound {label} axis is too large"))
                })
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
        .ok_or_else(|| format!("causal target-sound {label} shape overflow"))?;
    if view.len() != expected {
        return Err(format!(
            "causal target-sound graph returned {} {label} samples; expected {expected}",
            view.len()
        ));
    }
    if view
        .iter()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(format!(
            "causal target-sound graph returned invalid normalized {label} audio"
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

fn validate_presence_probabilities(values: [f32; PRESENCE_CLASSES]) -> Result<(), String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (values.iter().sum::<f32>() - 1.0).abs() > PRESENCE_SUM_TOLERANCE
    {
        return Err(
            "causal target-sound presence output must contain normalized absent/uncertain/present probabilities"
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
            "causal target-sound recurrent state shape changed from {:?} to {:?}",
            edge.shape,
            tensor.shape()
        ));
    }
    match edge.element_type {
        StateElementType::Float32 => {
            let view = tensor.to_plain_array_view::<f32>().map_err(model_error)?;
            if view.iter().any(|value| !value.is_finite()) {
                return Err(
                    "causal target-sound recurrent state contains a non-finite value".into(),
                );
            }
        }
        StateElementType::Int64 => {
            tensor.to_plain_array_view::<i64>().map_err(model_error)?;
        }
    }
    Ok(())
}

fn snapshot_state(
    tensor: &Tensor,
    edge: &StateEdge,
) -> Result<CausalTargetSoundStateValue, String> {
    validate_state_tensor(tensor, edge)?;
    match edge.element_type {
        StateElementType::Float32 => Ok(CausalTargetSoundStateValue::Float32 {
            shape: edge.shape.clone(),
            values: tensor
                .to_plain_array_view::<f32>()
                .map_err(model_error)?
                .iter()
                .copied()
                .collect(),
        }),
        StateElementType::Int64 => Ok(CausalTargetSoundStateValue::Int64 {
            shape: edge.shape.clone(),
            values: tensor
                .to_plain_array_view::<i64>()
                .map_err(model_error)?
                .iter()
                .copied()
                .collect(),
        }),
    }
}

fn restore_state(value: &CausalTargetSoundStateValue, edge: &StateEdge) -> Result<Tensor, String> {
    let tensor = match (value, edge.element_type) {
        (
            CausalTargetSoundStateValue::Float32 { shape, values },
            StateElementType::Float32,
        ) if shape == &edge.shape && values.iter().all(|value| value.is_finite()) => {
            Tensor::from_shape(shape, values).map_err(model_error)?
        }
        (
            CausalTargetSoundStateValue::Int64 { shape, values },
            StateElementType::Int64,
        ) if shape == &edge.shape => Tensor::from_shape(shape, values).map_err(model_error)?,
        _ => {
            return Err(
                "causal target-sound snapshot state type, shape, or values differ from the model contract"
                    .into(),
            )
        }
    };
    validate_state_tensor(&tensor, edge)?;
    Ok(tensor)
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
        _ => return Err("causal target-sound recurrent state type changed".into()),
    }
    Ok(())
}

fn model_error(error: impl std::fmt::Display) -> String {
    format!("causal target-sound ONNX error: {error:#}")
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
    fn signed_recurrent_target_residual_graph_runs_snapshots_and_resets() {
        let (_directory, package) = fixture_package();
        let model = CausalTargetSoundModel::load_runtime_package(&package, AcceleratorRuntime::Cpu)
            .unwrap();
        assert_eq!(model.sample_rate_hz(), 16_000);
        assert_eq!(model.channels(), 1);
        assert_eq!(model.frame_samples(), 4);
        assert_eq!(model.query_classes(), 2);
        assert_eq!(model.algorithmic_latency_samples(), 4);
        assert_eq!(model.flush_samples(), 4);
        let mut stream = model.start(1).unwrap();
        let input = vec![vec![-0.5, 0.0, 0.25, 0.75]];
        let result = stream.process(&input).unwrap();
        assert_eq!(result.target, input);
        assert_eq!(result.residual, vec![vec![0.0; 4]]);
        assert_eq!(result.presence_probabilities, [0.0, 0.0, 1.0]);
        let snapshot = stream.snapshot().unwrap();
        stream.reset().unwrap();
        stream.restore(&snapshot).unwrap();
        assert_eq!(stream.snapshot().unwrap(), snapshot);
    }

    #[test]
    fn rejects_stateless_changed_query_and_excess_latency_contracts() {
        let components = fixture_components(vec![0]);
        let mut invalid = manifest(&components);
        invalid.state_pairs.clear();
        assert!(validate_runtime_package_contract(&invalid).is_err());

        let mut invalid = manifest(&components);
        invalid.latency.algorithmic_latency_samples = 1_601;
        invalid.latency.flush_samples = 1_601;
        assert!(validate_runtime_package_contract(&invalid)
            .unwrap_err()
            .contains("maximum is 100 ms"));

        let vectors: crate::RuntimeModelNumericalVectorsV1 =
            serde_json::from_slice(&components[3]).unwrap();
        assert!(validate_sequence_vector_semantics(&manifest(&components), &vectors.cases).is_ok());
        let mut changed = vectors.cases.clone();
        changed[1]
            .inputs
            .iter_mut()
            .find(|tensor| tensor.name == "query")
            .unwrap()
            .values = vec![1.0, 0.0];
        assert!(
            validate_sequence_vector_semantics(&manifest(&components), &changed)
                .unwrap_err()
                .contains("change query class")
        );
    }

    pub(crate) fn fixture_package() -> (tempfile::TempDir, RuntimeModelPackage) {
        let directory = tempfile::tempdir().unwrap();
        let package_path = directory.path().join("causal-target-sound.dmp");
        let mut model_bytes = Vec::new();
        causal_identity_model().encode(&mut model_bytes).unwrap();
        let components = fixture_components(model_bytes);
        let manifest = manifest(&components);
        let package =
            RuntimeModelPackage::for_onnx_v2_contract_test(package_path, manifest, components);
        (directory, package)
    }

    fn fixture_components(model: Vec<u8>) -> Vec<Vec<u8>> {
        let vectors = serde_json::to_vec(&serde_json::json!({
            "schema": "denoize-runtime-model-numerical-vectors-v1",
            "profile_id": "fp32",
            "cases": [
                vector_case("causal-reset", vec![-0.5, 0.0, 0.25, 0.75], vec![0.0, 0.0]),
                vector_case("causal-recurrent", vec![0.25, -0.25, 0.5, -0.5], vec![1.0, -1.0]),
                vector_case("causal-flush", vec![0.0, 0.0, 0.0, 0.0], vec![0.5, -0.5])
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
                {"name":"mixture","element_type":"float32","shape":[1,1,4],"values":audio},
                {"name":"query","element_type":"float32","shape":[1,2],"values":[0.0,1.0]},
                {"name":"state_in","element_type":"float32","shape":[1,2],"values":state}
            ],
            "outputs": [
                {"name":"target","element_type":"float32","shape":[1,1,4],"values":audio},
                {"name":"residual","element_type":"float32","shape":[1,1,4],"values":[0.0,0.0,0.0,0.0]},
                {"name":"presence","element_type":"float32","shape":[1,3],"values":[0.0,0.0,1.0]},
                {"name":"state_out","element_type":"float32","shape":[1,2],"values":state}
            ],
            "tolerance":{"absolute":0.000001,"relative":0.000001}
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
                axis("batch", "batch", 1),
                axis("channels", "channel", 1),
                axis("samples", "sample", 4),
            ]
        };
        let query_axes = || vec![axis("batch", "batch", 1), axis("classes", "feature", 2)];
        let presence_axes = || vec![axis("batch", "batch", 1), axis("classes", "feature", 3)];
        let state_axes = || vec![axis("batch", "batch", 1), axis("memory", "state", 2)];
        let tensor =
            |name: &str,
             role: &str,
             axes: Vec<crate::RuntimeModelAxisContractV2>,
             state_id: Option<&str>| crate::RuntimeModelTensorContractV2 {
                name: name.into(),
                role: role.into(),
                element_type: "float32".into(),
                axes,
                optional: false,
                state_id: state_id.map(str::to_string),
            };
        RuntimeModelPackageManifestV2 {
            schema: crate::RUNTIME_MODEL_PACKAGE_SCHEMA_V2.into(),
            format_version: crate::RUNTIME_MODEL_PACKAGE_VERSION_V2,
            package_id: "denoize.test.causal-target-sound".into(),
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
                    policy: "program-multichannel".into(),
                    roles: vec![crate::RuntimeModelChannelRoleContractV2 {
                        channel_index: 0,
                        role: "program-center".into(),
                    }],
                    geometry: None,
                },
            },
            tensors: crate::RuntimeModelTensorSetContractV2 {
                inputs: vec![
                    tensor("mixture", "audio", waveform_axes(), None),
                    tensor("query", "query", query_axes(), None),
                    tensor("state_in", "state", state_axes(), Some("memory")),
                ],
                outputs: vec![
                    tensor("target", "audio", waveform_axes(), None),
                    tensor("residual", "residual", waveform_axes(), None),
                    tensor("presence", "diagnostic", presence_axes(), None),
                    tensor("state_out", "state", state_axes(), Some("memory")),
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
                component(
                    "model-fp32",
                    "onnx-model",
                    file("model.onnx", &components[0]),
                ),
                component(
                    "license",
                    "license-notice",
                    file("LICENSE.txt", &components[1]),
                ),
                component(
                    "provenance",
                    "provenance-json",
                    file("provenance.json", &components[2]),
                ),
                component(
                    "vectors-fp32",
                    "numerical-vectors-json",
                    file("vectors.json", &components[3]),
                ),
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
                source_repository: "https://example.invalid/causal-target-sound".into(),
                source_revision: "0123456789abcdef".into(),
                source_sha256: "0".repeat(64),
                source_license_spdx: "MIT".into(),
                checkpoint_source: "https://example.invalid/causal-target-sound.ckpt".into(),
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

    fn axis(name: &str, kind: &str, fixed: u64) -> crate::RuntimeModelAxisContractV2 {
        crate::RuntimeModelAxisContractV2 {
            name: name.into(),
            kind: kind.into(),
            fixed: Some(fixed),
        }
    }

    fn component(
        id: &str,
        kind: &str,
        file: crate::RuntimeModelFileContract,
    ) -> crate::RuntimeModelComponentContractV2 {
        crate::RuntimeModelComponentContractV2 {
            id: id.into(),
            kind: kind.into(),
            file,
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
                name: "causal-target-sound-identity".into(),
                node: vec![
                    identity("mixture", "target", "target-identity"),
                    constant("residual", vec![1, 1, 4], vec![0.0; 4]),
                    constant("presence", vec![1, 3], vec![0.0, 0.0, 1.0]),
                    identity("state_in", "state_out", "state-identity"),
                ],
                input: vec![
                    value_info("mixture", vec![dimension(1), dimension(1), dimension(4)]),
                    value_info("query", vec![dimension(1), dimension(2)]),
                    value_info("state_in", vec![dimension(1), dimension(2)]),
                ],
                output: vec![
                    value_info("target", vec![dimension(1), dimension(1), dimension(4)]),
                    value_info("residual", vec![dimension(1), dimension(1), dimension(4)]),
                    value_info("presence", vec![dimension(1), dimension(3)]),
                    value_info("state_out", vec![dimension(1), dimension(2)]),
                ],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn identity(input: &str, output: &str, name: &str) -> NodeProto {
        NodeProto {
            input: vec![input.into()],
            output: vec![output.into()],
            name: name.into(),
            op_type: "Identity".into(),
            ..Default::default()
        }
    }

    fn constant(name: &str, dims: Vec<i64>, values: Vec<f32>) -> NodeProto {
        NodeProto {
            output: vec![name.into()],
            name: format!("{name}-constant"),
            op_type: "Constant".into(),
            attribute: vec![AttributeProto {
                name: "value".into(),
                r#type: attribute_proto::AttributeType::Tensor as i32,
                t: Some(TensorProto {
                    dims,
                    data_type: tensor_proto::DataType::Float as i32,
                    float_data: values,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn dimension(value: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
            denotation: String::new(),
        }
    }
}
