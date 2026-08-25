//! Generic waveform-to-waveform ONNX backend using the pure-Rust tract runtime.
//!
//! The model must have exactly one `float32` input and one `float32` output.
//! Accepted layouts are `[batch, samples]` and `[batch, channels, samples]`;
//! denoize supplies a batch and channel size of one and processes file channels
//! independently. The output must contain at least as many samples as the model
//! input. Models operating on spectra or requiring an iterative sampler need a
//! dedicated adapter and are deliberately rejected by this backend.
//!
//! [`OnnxWaveformModel`] parses and validates a graph once, then reuses the
//! optimized graph for repeated calls with the same model-rate input length.
//! The module-level [`process`] function remains as a one-call convenience
//! wrapper.

use super::tract_runtime::SharedRunnable;
use super::OnnxModelConfig;
use crate::AcceleratorRuntime;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;

/// Waveform tensor layout accepted by the generic ONNX backend.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnnxWaveformLayout {
    /// One mono waveform per batch item: `[batch, samples]`.
    BatchSamples,
    /// An explicit mono channel dimension: `[batch, channels, samples]`.
    BatchChannelsSamples,
}

impl OnnxWaveformLayout {
    fn shape(self, samples: usize) -> TVec<usize> {
        match self {
            Self::BatchSamples => tvec!(1, samples),
            Self::BatchChannelsSamples => tvec!(1, 1, samples),
        }
    }

    const fn sample_axis(self) -> usize {
        match self {
            Self::BatchSamples => 1,
            Self::BatchChannelsSamples => 2,
        }
    }
}

/// Validated tensor contract reported by a loaded waveform model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnnxWaveformContract {
    layout: OnnxWaveformLayout,
    fixed_input_samples: Option<usize>,
    fixed_output_samples: Option<usize>,
}

impl OnnxWaveformContract {
    /// Return the model's waveform tensor layout.
    #[must_use]
    pub const fn layout(self) -> OnnxWaveformLayout {
        self.layout
    }

    /// Return a fixed input length declared by the graph, if any.
    #[must_use]
    pub const fn fixed_input_samples(self) -> Option<usize> {
        self.fixed_input_samples
    }

    /// Return a fixed output length declared by the graph, if any.
    #[must_use]
    pub const fn fixed_output_samples(self) -> Option<usize> {
        self.fixed_output_samples
    }
}

struct CompiledWaveformModel {
    input_samples: usize,
    runnable: SharedRunnable,
}

/// A parsed waveform ONNX model that can be reused across inference calls.
///
/// Loading rejects known incompatibilities with the public one-input/one-output
/// `float32` contract before any audio is resampled. Output facts that require a
/// concrete sample length are enforced when that length is first compiled. The
/// most recently used input length is cached, so repeated calls do not parse or
/// optimize the graph again. Loading also detaches inference from later pathname
/// replacement: compiled graphs are cloned from the model parsed by
/// [`load`](Self::load), not reopened by path.
pub struct OnnxWaveformModel {
    config: OnnxModelConfig,
    contract: OnnxWaveformContract,
    template: InferenceModel,
    runtime: AcceleratorRuntime,
    compiled: Mutex<Option<CompiledWaveformModel>>,
}

impl OnnxWaveformModel {
    /// Parse and validate a waveform ONNX model.
    pub fn load(config: OnnxModelConfig) -> Result<Self, String> {
        Self::load_with_accelerator(config, AcceleratorRuntime::Cpu)
    }

    /// Parse and validate a waveform model for a selected runtime.
    pub fn load_with_accelerator(
        config: OnnxModelConfig,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        config
            .validate_config()
            .map_err(|error| error.to_string())?;
        if !config.path.is_file() {
            return Err(format!(
                "ONNX model does not exist or is not a file: {}",
                config.path.display()
            ));
        }
        let template = tract_onnx::onnx()
            .model_for_path(&config.path)
            .map_err(|error| {
                format!(
                    "failed to load ONNX model {}: {error}",
                    config.path.display()
                )
            })?;
        Self::from_template(config, runtime, template, None)
    }

    /// Parse a graph from an authenticated runtime package and require its
    /// signed tensor contract to match the graph before inference.
    pub(crate) fn load_runtime_package_with_accelerator(
        package: &crate::RuntimeModelPackage,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        let config = package.model_config();
        config
            .validate_config()
            .map_err(|error| error.to_string())?;
        let mut reader = package.open_model_reader_for(runtime)?;
        let template = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|error| {
                format!(
                    "failed to load ONNX model from authenticated package {}: {error:#}",
                    package.package_path().display()
                )
            })?;
        reader.finish().map_err(|error| {
            format!(
                "failed to authenticate ONNX model bytes from package {}: {error}",
                package.package_path().display()
            )
        })?;
        if let Some(manifest) = package.manifest_v2() {
            let profile = package
                .precision_profile_for(runtime)?
                .expect("v2 packages always select a precision profile");
            let mut inspected = template.clone();
            if inspected.analyse(false).is_ok() {
                validate_v2_graph_contract(&inspected, manifest)?;
            } else {
                validate_v2_graph_contract(&template, manifest)?;
            }
            validate_v2_generic_waveform_compatibility(manifest)?;
            let vectors = package
                .numerical_vectors_for(runtime)?
                .expect("v2 precision profiles always carry numerical vectors");
            validate_v2_numerical_vectors(&template, manifest, profile, &vectors, runtime)?;
        }
        Self::from_template(config, runtime, template, Some(&package.manifest().tensor))
    }

    fn from_template(
        config: OnnxModelConfig,
        runtime: AcceleratorRuntime,
        template: InferenceModel,
        expected: Option<&crate::RuntimeModelTensorContract>,
    ) -> Result<Self, String> {
        // Propagate declared facts through simple dynamic graphs before
        // inspecting their public output. Some operators need a concrete
        // sample length and are validated later during compilation, so an
        // incomplete analysis here is intentionally non-fatal.
        let mut inspected = template.clone();
        let contract = if inspected.analyse(false).is_ok() {
            validate_contract(&inspected)?
        } else {
            validate_contract(&template)?
        };
        if let Some(expected) = expected {
            validate_package_contract(contract, expected)?;
        }
        Ok(Self {
            config,
            contract,
            template,
            runtime,
            compiled: Mutex::new(None),
        })
    }

    /// Return the path and sample rate used when the model was loaded.
    #[must_use]
    pub fn config(&self) -> &OnnxModelConfig {
        &self.config
    }

    /// Return the graph contract established during loading.
    #[must_use]
    pub const fn contract(&self) -> OnnxWaveformContract {
        self.contract
    }

    /// Process planar audio while preserving its channel count and duration.
    ///
    /// File channels are presented to the mono model independently. Set
    /// `deterministic` to serialize channel inference in stable order.
    pub fn process(
        &self,
        channels: &[Vec<f64>],
        input_sample_rate: u32,
        deterministic: bool,
    ) -> Result<Vec<Vec<f64>>, String> {
        if channels.is_empty() {
            return Ok(Vec::new());
        }
        let input_frames = channels[0].len();
        if channels.iter().any(|channel| channel.len() != input_frames) {
            return Err("ONNX waveform input channels must have equal lengths".into());
        }
        let model_inputs = crate::resample::resample_channels(
            channels,
            input_sample_rate,
            self.config.sample_rate,
        )?;
        if model_inputs[0].is_empty() {
            return Ok(model_inputs);
        }
        let input_samples = model_inputs[0].len();
        let shape = self.contract.layout.shape(input_samples);
        let model = self.compiled_model(input_samples)?;
        let process_channel = |(model_input, original): (&Vec<f64>, &Vec<f64>)| {
            let model_output = run_model(model_input, &shape, model.as_ref())?;
            let mut output = crate::resample::resample(
                &model_output,
                self.config.sample_rate,
                input_sample_rate,
            )?;
            output.truncate(original.len());
            output.resize(original.len(), 0.0);
            Ok(output)
        };
        if deterministic {
            model_inputs
                .iter()
                .zip(channels.iter())
                .map(process_channel)
                .collect()
        } else {
            model_inputs
                .par_iter()
                .zip(channels.par_iter())
                .map(process_channel)
                .collect()
        }
    }

    fn compiled_model(&self, input_samples: usize) -> Result<SharedRunnable, String> {
        if let Some(required) = self.contract.fixed_input_samples {
            if input_samples != required {
                return Err(format!(
                    "ONNX model requires {required} input samples, got {input_samples}"
                ));
            }
        }
        if let Some(produced) = self.contract.fixed_output_samples {
            if produced < input_samples {
                return Err(format!(
                    "ONNX model declares {produced} output samples for an input of {input_samples}; output must not be shorter"
                ));
            }
        }

        let mut compiled = self
            .compiled
            .lock()
            .map_err(|_| "ONNX compiled-model cache lock was poisoned".to_string())?;
        if let Some(cached) = compiled.as_ref() {
            if cached.input_samples == input_samples {
                return Ok(Arc::clone(&cached.runnable));
            }
        }
        *compiled = None;

        let input_shape = self.contract.layout.shape(input_samples);
        let output_samples = self.contract.fixed_output_samples.unwrap_or(input_samples);
        let output_shape = self.contract.layout.shape(output_samples);
        let mut model = self.template.clone();
        model
            .set_input_fact(0, f32::fact(input_shape).into())
            .map_err(tract_error)?;
        model
            .set_output_fact(0, f32::fact(output_shape).into())
            .map_err(tract_error)?;
        let model = model.into_typed().map_err(tract_error)?;
        let runnable = super::tract_runtime::prepare(model, self.runtime, "ONNX waveform model")?;
        *compiled = Some(CompiledWaveformModel {
            input_samples,
            runnable: Arc::clone(&runnable),
        });
        Ok(runnable)
    }
}

pub(crate) fn validate_v2_graph_contract(
    model: &InferenceModel,
    manifest: &crate::RuntimeModelPackageManifestV2,
) -> Result<(), String> {
    let inputs = model.input_outlets().map_err(tract_error)?;
    let outputs = model.output_outlets().map_err(tract_error)?;
    if inputs.len() != manifest.tensors.inputs.len()
        || outputs.len() != manifest.tensors.outputs.len()
    {
        return Err(format!(
            "authenticated runtime model v2 tensor count does not match the ONNX graph: manifest inputs={} outputs={}, graph inputs={} outputs={}",
            manifest.tensors.inputs.len(),
            manifest.tensors.outputs.len(),
            inputs.len(),
            outputs.len()
        ));
    }
    for (index, (outlet, expected)) in inputs.iter().zip(&manifest.tensors.inputs).enumerate() {
        validate_v2_graph_tensor(
            model,
            *outlet,
            model.input_fact(index).map_err(tract_error)?,
            expected,
            "input",
        )?;
    }
    for (index, (outlet, expected)) in outputs.iter().zip(&manifest.tensors.outputs).enumerate() {
        validate_v2_graph_tensor(
            model,
            *outlet,
            model.output_fact(index).map_err(tract_error)?,
            expected,
            "output",
        )?;
    }
    Ok(())
}

fn validate_v2_graph_tensor(
    model: &InferenceModel,
    outlet: OutletId,
    fact: &InferenceFact,
    expected: &crate::RuntimeModelTensorContractV2,
    kind: &str,
) -> Result<(), String> {
    let actual_name = model
        .outlet_label(outlet)
        .unwrap_or_else(|| model.node(outlet.node).name.as_str());
    if actual_name != expected.name {
        return Err(format!(
            "authenticated runtime model v2 {kind} name does not match the ONNX graph: manifest={} graph={actual_name}",
            expected.name
        ));
    }
    let actual_type = match fact.datum_type.concretize() {
        Some(DatumType::F32) => "float32",
        Some(DatumType::I64) => "int64",
        Some(other) => {
            return Err(format!(
                "authenticated runtime model v2 {kind} {} has unsupported ONNX type {other:?}",
                expected.name
            ));
        }
        None => {
            return Err(format!(
                "authenticated runtime model v2 {kind} {} datum type is unknown",
                expected.name
            ));
        }
    };
    if actual_type != expected.element_type {
        return Err(format!(
            "authenticated runtime model v2 {kind} {} type mismatch: manifest={} graph={actual_type}",
            expected.name, expected.element_type
        ));
    }
    let rank = known_rank(kind, fact)?;
    if rank != expected.axes.len() {
        return Err(format!(
            "authenticated runtime model v2 {kind} {} rank mismatch: manifest={} graph={rank}",
            expected.name,
            expected.axes.len()
        ));
    }
    for (axis_index, axis) in expected.axes.iter().enumerate() {
        if let Some(expected_dimension) = axis.fixed {
            let actual = known_dimension(fact, axis_index)?;
            if actual.map(|value| value as u64) != Some(expected_dimension) {
                return Err(format!(
                    "authenticated runtime model v2 {kind} {} axis {} mismatch: manifest={expected_dimension} graph={actual:?}",
                    expected.name, axis.name
                ));
            }
        }
    }
    Ok(())
}

fn validate_v2_generic_waveform_compatibility(
    manifest: &crate::RuntimeModelPackageManifestV2,
) -> Result<(), String> {
    let generic = manifest.tensors.inputs.len() == 1
        && manifest.tensors.outputs.len() == 1
        && matches!(
            manifest.runtime.mode.as_str(),
            "finite" | "finite-and-streaming"
        )
        && manifest.frontend.channels.policy == "independent-mono"
        && manifest.tensors.inputs[0].role == "audio"
        && manifest.tensors.outputs[0].role == "audio"
        && !manifest.tensors.inputs[0].optional
        && manifest.state_pairs.is_empty()
        && matches!(
            manifest.tensors.inputs[0]
                .axes
                .iter()
                .map(|axis| axis.kind.as_str())
                .collect::<Vec<_>>()
                .as_slice(),
            ["batch", "sample"] | ["batch", "channel", "sample"]
        )
        && manifest.tensors.inputs[0].axes == manifest.tensors.outputs[0].axes;
    if !generic {
        return Err(
            "authenticated runtime model v2 graph requires a dedicated adapter for streaming-only, multichannel, optional, auxiliary/state, or non-waveform semantics"
                .into(),
        );
    }
    Ok(())
}

pub(crate) fn validate_v2_numerical_vectors(
    template: &InferenceModel,
    manifest: &crate::RuntimeModelPackageManifestV2,
    profile: &crate::RuntimeModelPrecisionProfileContractV2,
    vectors: &crate::RuntimeModelNumericalVectorsV1,
    runtime: AcceleratorRuntime,
) -> Result<(), String> {
    if vectors.profile_id != profile.id {
        return Err(
            "runtime model numerical vector profile selection changed after verification".into(),
        );
    }
    for case in &vectors.cases {
        let mut model = template.clone();
        let input_by_name: std::collections::HashMap<_, _> = case
            .inputs
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect();
        let output_by_name: std::collections::HashMap<_, _> = case
            .outputs
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect();
        for (index, contract) in manifest.tensors.inputs.iter().enumerate() {
            let tensor = input_by_name.get(contract.name.as_str()).ok_or_else(|| {
                format!(
                    "runtime model numerical case {} omits graph input {}",
                    case.id, contract.name
                )
            })?;
            model
                .set_input_fact(index, numerical_fact(tensor)?)
                .map_err(tract_error)?;
        }
        for (index, contract) in manifest.tensors.outputs.iter().enumerate() {
            let tensor = output_by_name.get(contract.name.as_str()).ok_or_else(|| {
                format!(
                    "runtime model numerical case {} omits graph output {}",
                    case.id, contract.name
                )
            })?;
            model
                .set_output_fact(index, numerical_fact(tensor)?)
                .map_err(tract_error)?;
        }
        let model = model.into_typed().map_err(tract_error)?;
        let runnable = super::tract_runtime::prepare(
            model,
            runtime,
            &format!("runtime model v2 numerical case {}", case.id),
        )?;
        let mut inputs = TVec::new();
        for contract in &manifest.tensors.inputs {
            let tensor = input_by_name
                .get(contract.name.as_str())
                .expect("all numerical inputs were checked");
            inputs.push(numerical_tensor(tensor)?.into_tvalue());
        }
        let actual = runnable.run(inputs).map_err(tract_error)?;
        if actual.len() != manifest.tensors.outputs.len() {
            return Err(format!(
                "runtime model numerical case {} returned {} outputs, expected {}",
                case.id,
                actual.len(),
                manifest.tensors.outputs.len()
            ));
        }
        for (index, contract) in manifest.tensors.outputs.iter().enumerate() {
            let expected = output_by_name
                .get(contract.name.as_str())
                .expect("all numerical outputs were checked");
            compare_numerical_output(&actual[index], expected, case.tolerance, &case.id)?;
        }
    }
    Ok(())
}

fn numerical_fact(tensor: &crate::RuntimeModelNumericalTensorV1) -> Result<InferenceFact, String> {
    let shape: TVec<usize> = tensor
        .shape
        .iter()
        .map(|value| {
            usize::try_from(*value)
                .map_err(|_| "runtime model numerical vector shape is too large".to_string())
        })
        .collect::<Result<_, _>>()?;
    match tensor.element_type.as_str() {
        "float32" => Ok(f32::fact(shape).into()),
        "int64" => Ok(i64::fact(shape).into()),
        other => Err(format!(
            "runtime model numerical vector element type {other} is unsupported"
        )),
    }
}

fn numerical_tensor(tensor: &crate::RuntimeModelNumericalTensorV1) -> Result<Tensor, String> {
    let shape: Vec<usize> = tensor
        .shape
        .iter()
        .map(|value| {
            usize::try_from(*value)
                .map_err(|_| "runtime model numerical vector shape is too large".to_string())
        })
        .collect::<Result<_, _>>()?;
    match tensor.element_type.as_str() {
        "float32" => {
            let values: Vec<f32> = tensor.values.iter().map(|value| *value as f32).collect();
            Tensor::from_shape(&shape, &values).map_err(tract_error)
        }
        "int64" => {
            let values: Vec<i64> = tensor.values.iter().map(|value| *value as i64).collect();
            Tensor::from_shape(&shape, &values).map_err(tract_error)
        }
        other => Err(format!(
            "runtime model numerical vector element type {other} is unsupported"
        )),
    }
}

fn compare_numerical_output(
    actual: &TValue,
    expected: &crate::RuntimeModelNumericalTensorV1,
    tolerance: crate::RuntimeModelNumericalToleranceV1,
    case_id: &str,
) -> Result<(), String> {
    match expected.element_type.as_str() {
        "float32" => {
            let values = actual.to_plain_array_view::<f32>().map_err(tract_error)?;
            let expected_shape = numerical_shape(expected)?;
            if values.shape() != expected_shape.as_slice() {
                return Err(format!(
                    "runtime model numerical case {case_id} output {} shape mismatch: expected {:?}, observed {:?}",
                    expected.name,
                    expected_shape,
                    values.shape()
                ));
            }
            if values.len() != expected.values.len() {
                return Err(format!(
                    "runtime model numerical case {case_id} output {} length mismatch",
                    expected.name
                ));
            }
            for (index, (observed, reference)) in values.iter().zip(&expected.values).enumerate() {
                let observed = f64::from(*observed);
                let allowed =
                    f64::from(tolerance.absolute) + f64::from(tolerance.relative) * reference.abs();
                if !observed.is_finite() || (observed - reference).abs() > allowed {
                    return Err(format!(
                        "runtime model numerical case {case_id} output {} differs at element {index}: expected {reference}, observed {observed}, tolerance {allowed}",
                        expected.name
                    ));
                }
            }
        }
        "int64" => {
            let values = actual.to_plain_array_view::<i64>().map_err(tract_error)?;
            let expected_shape = numerical_shape(expected)?;
            if values.shape() != expected_shape.as_slice() {
                return Err(format!(
                    "runtime model numerical case {case_id} output {} shape mismatch: expected {:?}, observed {:?}",
                    expected.name,
                    expected_shape,
                    values.shape()
                ));
            }
            if values.len() != expected.values.len()
                || values
                    .iter()
                    .zip(&expected.values)
                    .any(|(observed, reference)| *observed != *reference as i64)
            {
                return Err(format!(
                    "runtime model numerical case {case_id} integer output {} differs",
                    expected.name
                ));
            }
        }
        other => {
            return Err(format!(
                "runtime model numerical output type {other} is unsupported"
            ));
        }
    }
    Ok(())
}

fn numerical_shape(tensor: &crate::RuntimeModelNumericalTensorV1) -> Result<Vec<usize>, String> {
    tensor
        .shape
        .iter()
        .map(|value| {
            usize::try_from(*value)
                .map_err(|_| "runtime model numerical vector shape is too large".to_string())
        })
        .collect()
}

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
    deterministic: bool,
) -> Result<Vec<Vec<f64>>, String> {
    OnnxWaveformModel::load(config.clone())?.process(channels, input_sample_rate, deterministic)
}

fn validate_contract(model: &InferenceModel) -> Result<OnnxWaveformContract, String> {
    if model.input_outlets().map_err(tract_error)?.len() != 1
        || model.output_outlets().map_err(tract_error)?.len() != 1
    {
        return Err("ONNX waveform model must have exactly one input and one output".into());
    }
    let input = model.input_fact(0).map_err(tract_error)?;
    let output = model.output_fact(0).map_err(tract_error)?;
    validate_float32_input(input)?;
    validate_compatible_float32_output(output)?;

    let input_rank = known_rank("input", input)?;
    let layout = match input_rank {
        2 => OnnxWaveformLayout::BatchSamples,
        3 => OnnxWaveformLayout::BatchChannelsSamples,
        other => {
            return Err(format!(
                "unsupported ONNX input rank {other}; expected [batch, samples] or [batch, channels, samples]"
            ));
        }
    };
    validate_unit_dimension("input batch", input, 0)?;
    if layout == OnnxWaveformLayout::BatchChannelsSamples {
        validate_unit_dimension("input channel", input, 1)?;
    }
    let output_rank = optional_known_rank("output", output)?;
    if let Some(output_rank) = output_rank {
        if output_rank != input_rank {
            return Err(format!(
                "ONNX waveform output rank {output_rank} does not match input rank {input_rank}"
            ));
        }
        validate_unit_dimension("output batch", output, 0)?;
        if layout == OnnxWaveformLayout::BatchChannelsSamples {
            validate_unit_dimension("output channel", output, 1)?;
        }
    }
    let fixed_input_samples = known_dimension(input, layout.sample_axis())?;
    let fixed_output_samples = output_rank
        .map(|_| known_dimension(output, layout.sample_axis()))
        .transpose()?
        .flatten();
    if let (Some(input_samples), Some(output_samples)) = (fixed_input_samples, fixed_output_samples)
    {
        if output_samples < input_samples {
            return Err(format!(
                "ONNX model declares {output_samples} output samples for {input_samples} input samples; output must not be shorter"
            ));
        }
    }
    Ok(OnnxWaveformContract {
        layout,
        fixed_input_samples,
        fixed_output_samples,
    })
}

fn validate_package_contract(
    actual: OnnxWaveformContract,
    expected: &crate::RuntimeModelTensorContract,
) -> Result<(), String> {
    let actual_layout = match actual.layout {
        OnnxWaveformLayout::BatchSamples => "batch-samples",
        OnnxWaveformLayout::BatchChannelsSamples => "batch-channels-samples",
    };
    let actual_input = actual
        .fixed_input_samples
        .map(|value| u64::try_from(value).unwrap_or(u64::MAX));
    let actual_output = actual
        .fixed_output_samples
        .map(|value| u64::try_from(value).unwrap_or(u64::MAX));
    if expected.element_type != "float32"
        || expected.layout != actual_layout
        || expected.fixed_input_samples != actual_input
        || expected.fixed_output_samples != actual_output
    {
        return Err(format!(
            "authenticated runtime model tensor contract does not match the ONNX graph: manifest layout={} input={:?} output={:?}, graph layout={actual_layout} input={actual_input:?} output={actual_output:?}",
            expected.layout, expected.fixed_input_samples, expected.fixed_output_samples
        ));
    }
    Ok(())
}

fn validate_float32_input(fact: &InferenceFact) -> Result<(), String> {
    match fact.datum_type.concretize() {
        Some(DatumType::F32) => Ok(()),
        Some(other) => Err(format!(
            "ONNX waveform model input must be float32, got {other:?}"
        )),
        None => Err("ONNX waveform model input datum type must be known as float32".into()),
    }
}

fn validate_compatible_float32_output(fact: &InferenceFact) -> Result<(), String> {
    match fact.datum_type.concretize() {
        Some(DatumType::F32) | None => Ok(()),
        Some(other) => Err(format!(
            "ONNX waveform model output must be float32, got {other:?}"
        )),
    }
}

fn known_rank(kind: &str, fact: &InferenceFact) -> Result<usize, String> {
    optional_known_rank(kind, fact)?.ok_or_else(|| format!("ONNX model {kind} rank must be known"))
}

fn optional_known_rank(kind: &str, fact: &InferenceFact) -> Result<Option<usize>, String> {
    let Some(rank) = fact.shape.rank().concretize() else {
        return Ok(None);
    };
    usize::try_from(rank)
        .map(Some)
        .map_err(|_| format!("ONNX model {kind} rank is invalid: {rank}"))
}

fn known_dimension(fact: &InferenceFact, axis: usize) -> Result<Option<usize>, String> {
    let Some(dimension) = fact
        .shape
        .dim(axis)
        .and_then(|dimension| dimension.concretize())
    else {
        return Ok(None);
    };
    match dimension.to_i64() {
        Ok(value) if value >= 0 => usize::try_from(value)
            .map(Some)
            .map_err(|_| format!("ONNX tensor dimension {axis} is too large: {value}")),
        Ok(value) => Err(format!("ONNX tensor dimension {axis} is negative: {value}")),
        Err(_) => Ok(None),
    }
}

fn validate_unit_dimension(
    description: &str,
    fact: &InferenceFact,
    axis: usize,
) -> Result<(), String> {
    if let Some(value) = known_dimension(fact, axis)? {
        if value != 1 {
            return Err(format!(
                "ONNX waveform model {description} dimension must be one, got {value}"
            ));
        }
    }
    Ok(())
}

fn run_model(
    input: &[f64],
    shape: &[usize],
    runnable: &dyn tract_onnx::tract_core::runtime::Runnable,
) -> Result<Vec<f64>, String> {
    let samples: Vec<f32> = input.iter().map(|&sample| sample as f32).collect();
    let tensor = Tensor::from_shape(shape, &samples).map_err(tract_error)?;
    let outputs = runnable
        .run(tvec!(tensor.into_tvalue()))
        .map_err(tract_error)?;
    let output = outputs[0]
        .to_plain_array_view::<f32>()
        .map_err(tract_error)?;

    if output.len() < input.len() {
        return Err(format!(
            "ONNX model returned {} samples for an input of {}; output must not be shorter",
            output.len(),
            input.len()
        ));
    }
    let result: Vec<f64> = output
        .iter()
        .take(input.len())
        .map(|&sample| sample as f64)
        .collect();
    if result.iter().any(|sample| !sample.is_finite()) {
        return Err("ONNX model returned a non-finite audio sample".into());
    }
    Ok(result)
}

fn tract_error(error: impl std::fmt::Display) -> String {
    format!("ONNX inference failed: {error:#}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use sha2::Digest as _;
    use tract_onnx::pb::{
        attribute_proto, tensor_proto, tensor_shape_proto, type_proto, AttributeProto, GraphProto,
        ModelProto, NodeProto, OperatorSetIdProto, StringStringEntryProto, TensorProto,
        TensorShapeProto, TypeProto, ValueInfoProto,
    };

    #[test]
    fn rejects_missing_model() {
        let config = OnnxModelConfig {
            path: std::path::PathBuf::from("definitely-missing-model.onnx"),
            sample_rate: 16_000,
        };
        let error = process(&[vec![0.0]], 16_000, &config, false).unwrap_err();
        assert!(error.contains("does not exist"));
    }

    #[test]
    fn round_trip_resampling_preserves_requested_length() {
        let input: Vec<f64> = (0..441).map(|index| index as f64 / 441.0).collect();
        let at_16k = crate::resample::resample(&input, 44_100, 16_000).unwrap();
        let restored = crate::resample::resample(&at_16k, 16_000, 44_100).unwrap();
        assert_eq!(at_16k.len(), 160);
        assert_eq!(restored.len(), input.len());
    }

    #[test]
    fn identity_waveform_model_runs_end_to_end() {
        let model = identity_model();
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "denoize-identity-{}-{}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();

        let config = OnnxModelConfig {
            path: path.clone(),
            sample_rate: 16_000,
        };
        let input = vec![vec![-0.5, 0.0, 0.25, 0.75]];
        let output = process(&input, 16_000, &config, false).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), input[0].len());
        for (actual, expected) in output[0].iter().zip(&input[0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn authenticated_package_tensor_contract_is_checked_before_inference() {
        let (_directory, path) = write_model(identity_model());
        let tensor = crate::RuntimeModelTensorContract {
            element_type: "float32".into(),
            layout: "batch-samples".into(),
            fixed_input_samples: None,
            fixed_output_samples: None,
        };
        let package = crate::RuntimeModelPackage::for_onnx_contract_test(path, tensor.clone());
        let model = OnnxWaveformModel::load_runtime_package_with_accelerator(
            &package,
            AcceleratorRuntime::Cpu,
        )
        .unwrap();
        let input = vec![vec![-0.5, 0.0, 0.25, 0.75]];
        assert_eq!(model.process(&input, 16_000, false).unwrap(), input);

        let (_directory, path) = write_model(identity_model());
        let mut mismatched = tensor;
        mismatched.fixed_input_samples = Some(4);
        let package = crate::RuntimeModelPackage::for_onnx_contract_test(path, mismatched);
        let error = OnnxWaveformModel::load_runtime_package_with_accelerator(
            &package,
            AcceleratorRuntime::Cpu,
        )
        .err()
        .expect("mismatched signed tensor contract must fail");
        assert!(error.contains("does not match the ONNX graph"), "{error}");
    }

    #[test]
    fn authenticated_package_model_range_rejects_path_replacement() {
        let (_directory, path) = write_model(identity_model());
        let package = crate::RuntimeModelPackage::for_onnx_contract_test(
            path.clone(),
            crate::RuntimeModelTensorContract {
                element_type: "float32".into(),
                layout: "batch-samples".into(),
                fixed_input_samples: None,
                fixed_output_samples: None,
            },
        );
        std::fs::write(path, b"replaced after package verification").unwrap();
        let error = OnnxWaveformModel::load_runtime_package_with_accelerator(
            &package,
            AcceleratorRuntime::Cpu,
        )
        .err()
        .expect("replaced package path must fail");
        assert!(error.contains("changed after verification"), "{error}");
    }

    #[test]
    fn authenticated_package_rejects_external_tensor_sidecars() {
        let mut model = identity_model();
        let graph = model.graph.as_mut().unwrap();
        graph.node[0].op_type = "Add".into();
        graph.node[0].input.push("weights".into());
        graph.initializer.push(TensorProto {
            dims: vec![1],
            data_type: tensor_proto::DataType::Float as i32,
            name: "weights".into(),
            external_data: vec![StringStringEntryProto {
                key: "location".into(),
                value: "weights.bin".into(),
            }],
            data_location: Some(tensor_proto::DataLocation::External as i32),
            ..Default::default()
        });
        let (_directory, path) = write_model(model);
        let package = crate::RuntimeModelPackage::for_onnx_contract_test(
            path,
            crate::RuntimeModelTensorContract {
                element_type: "float32".into(),
                layout: "batch-samples".into(),
                fixed_input_samples: None,
                fixed_output_samples: None,
            },
        );
        let error = OnnxWaveformModel::load_runtime_package_with_accelerator(
            &package,
            AcceleratorRuntime::Cpu,
        )
        .err()
        .expect("package models must not resolve external tensor sidecars");
        assert!(error.to_ascii_lowercase().contains("external"), "{error}");
    }

    #[test]
    fn deterministic_mode_keeps_multichannel_output_byte_stable() {
        let model = identity_model();
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "denoize-deterministic-{}-{}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        let config = OnnxModelConfig {
            path: path.clone(),
            sample_rate: 16_000,
        };
        let input = vec![vec![-0.5, 0.0, 0.25, 0.75], vec![0.5, -0.25, 0.0, 0.125]];
        let first = process(&input, 16_000, &config, true).unwrap();
        let second = process(&input, 16_000, &config, true).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn loaded_rank_three_model_reports_contract_and_preserves_stereo_duration() {
        let (_directory, path) = write_model(identity_model_with_facts(
            vec![
                dimension_value(1),
                dimension_value(1),
                dimension_parameter("samples"),
            ],
            vec![
                dimension_value(1),
                dimension_value(1),
                dimension_parameter("samples"),
            ],
            tensor_proto::DataType::Float,
            tensor_proto::DataType::Float,
        ));
        let model = OnnxWaveformModel::load(OnnxModelConfig {
            path,
            sample_rate: 16_000,
        })
        .unwrap();
        assert_eq!(
            model.contract(),
            OnnxWaveformContract {
                layout: OnnxWaveformLayout::BatchChannelsSamples,
                fixed_input_samples: None,
                fixed_output_samples: None,
            }
        );

        let input = vec![
            (0..441).map(|index| index as f64 / 882.0).collect(),
            (0..441).map(|index| -(index as f64) / 882.0).collect(),
        ];
        let output = model.process(&input, 44_100, true).unwrap();
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), input[0].len());
        assert_eq!(output[1].len(), input[1].len());
        assert!(output.iter().flatten().all(|sample| sample.is_finite()));
    }

    #[test]
    fn loaded_model_reuses_one_compiled_graph_for_the_same_length() {
        let (_directory, path) = write_model(identity_model());
        let model = OnnxWaveformModel::load(OnnxModelConfig {
            path,
            sample_rate: 16_000,
        })
        .unwrap();
        let input = vec![vec![0.0; 32]];
        model.process(&input, 16_000, false).unwrap();
        let first = Arc::clone(&model.compiled.lock().unwrap().as_ref().unwrap().runnable);
        model.process(&input, 16_000, false).unwrap();
        let second = Arc::clone(&model.compiled.lock().unwrap().as_ref().unwrap().runnable);
        assert!(Arc::ptr_eq(&first, &second));

        model.process(&[vec![0.0; 64]], 16_000, false).unwrap();
        let third = Arc::clone(&model.compiled.lock().unwrap().as_ref().unwrap().runnable);
        assert!(!Arc::ptr_eq(&first, &third));
    }

    #[test]
    fn loaded_model_can_be_shared_between_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OnnxWaveformModel>();
    }

    #[test]
    fn loaded_model_does_not_reopen_a_replaced_path() {
        let (directory, path) = write_model(identity_model());
        let model = OnnxWaveformModel::load(OnnxModelConfig {
            path: path.clone(),
            sample_rate: 16_000,
        })
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"path was replaced after model loading").unwrap();

        let input = vec![vec![-0.5, 0.0, 0.25, 0.75]];
        let output = model.process(&input, 16_000, false).unwrap();
        assert_eq!(output, input);
        drop(directory);
    }

    #[test]
    fn fixed_sample_contract_is_enforced_before_compilation() {
        let (_directory, path) = write_model(identity_model_with_facts(
            vec![dimension_value(1), dimension_value(4)],
            vec![dimension_value(1), dimension_value(4)],
            tensor_proto::DataType::Float,
            tensor_proto::DataType::Float,
        ));
        let model = OnnxWaveformModel::load(OnnxModelConfig {
            path,
            sample_rate: 16_000,
        })
        .unwrap();
        assert_eq!(model.contract().fixed_input_samples(), Some(4));
        assert_eq!(model.contract().fixed_output_samples(), Some(4));
        let error = model.process(&[vec![0.0; 5]], 16_000, false).unwrap_err();
        assert!(error.contains("requires 4 input samples"), "{error}");
        assert!(model.compiled.lock().unwrap().is_none());
    }

    #[test]
    fn load_rejects_a_known_short_output_contract() {
        let (_directory, path) = write_model(constant_output_model(4, 3));
        let error = OnnxWaveformModel::load(OnnxModelConfig {
            path,
            sample_rate: 16_000,
        })
        .err()
        .expect("short fixed output must be rejected");
        assert!(error.contains("output must not be shorter"), "{error}");
    }

    #[test]
    fn load_rejects_non_float_and_non_mono_contracts() {
        let (_integer_directory, integer_path) = write_model(identity_model_with_facts(
            vec![dimension_value(1), dimension_parameter("samples")],
            vec![dimension_value(1), dimension_parameter("samples")],
            tensor_proto::DataType::Int32,
            tensor_proto::DataType::Int32,
        ));
        let error = OnnxWaveformModel::load(OnnxModelConfig {
            path: integer_path,
            sample_rate: 16_000,
        })
        .err()
        .expect("integer model must be rejected");
        assert!(error.contains("input must be float32"), "{error}");

        let (_integer_output_directory, integer_output_path) = write_model(cast_output_model());
        let error = OnnxWaveformModel::load(OnnxModelConfig {
            path: integer_output_path,
            sample_rate: 16_000,
        })
        .err()
        .expect("integer output model must be rejected");
        assert!(error.contains("output must be float32"), "{error}");

        let (_batch_directory, batch_path) = write_model(identity_model_with_facts(
            vec![dimension_value(2), dimension_parameter("samples")],
            vec![dimension_value(2), dimension_parameter("samples")],
            tensor_proto::DataType::Float,
            tensor_proto::DataType::Float,
        ));
        let error = OnnxWaveformModel::load(OnnxModelConfig {
            path: batch_path,
            sample_rate: 16_000,
        })
        .err()
        .expect("batch-two model must be rejected");
        assert!(
            error.contains("input batch dimension must be one"),
            "{error}"
        );

        let (_channel_directory, channel_path) = write_model(identity_model_with_facts(
            vec![
                dimension_value(1),
                dimension_value(2),
                dimension_parameter("samples"),
            ],
            vec![
                dimension_value(1),
                dimension_value(2),
                dimension_parameter("samples"),
            ],
            tensor_proto::DataType::Float,
            tensor_proto::DataType::Float,
        ));
        let error = OnnxWaveformModel::load(OnnxModelConfig {
            path: channel_path,
            sample_rate: 16_000,
        })
        .err()
        .expect("two-channel model must be rejected");
        assert!(
            error.contains("input channel dimension must be one"),
            "{error}"
        );
    }

    #[test]
    fn process_rejects_unequal_channel_lengths_before_resampling() {
        let (_directory, path) = write_model(identity_model());
        let model = OnnxWaveformModel::load(OnnxModelConfig {
            path,
            sample_rate: 16_000,
        })
        .unwrap();
        let error = model
            .process(&[vec![0.0; 4], vec![0.0; 3]], 48_000, false)
            .unwrap_err();
        assert!(error.contains("equal lengths"), "{error}");
        assert!(model.compiled.lock().unwrap().is_none());
    }

    fn identity_model() -> ModelProto {
        identity_model_with_facts(
            vec![dimension_value(1), dimension_parameter("samples")],
            vec![dimension_value(1), dimension_parameter("samples")],
            tensor_proto::DataType::Float,
            tensor_proto::DataType::Float,
        )
    }

    #[test]
    fn signed_runtime_package_fixture_loads_and_runs_end_to_end() {
        use base64::Engine as _;

        let directory = tempfile::tempdir().unwrap();
        let manifest_path = directory.path().join("manifest.json");
        let signature_path = directory.path().join("manifest.json.sig");
        let public_key_path = directory.path().join("minisign.pub");
        let model_path = directory.path().join("model.onnx");
        let license_path = directory.path().join("LICENSE.txt");
        let package_path = directory.path().join("model.dmp");
        std::fs::write(
            &manifest_path,
            include_bytes!("../model_package/testdata/manifest.json"),
        )
        .unwrap();
        std::fs::write(
            &signature_path,
            include_bytes!("../model_package/testdata/manifest.json.sig"),
        )
        .unwrap();
        std::fs::write(
            &public_key_path,
            include_bytes!("../model_package/testdata/minisign.pub"),
        )
        .unwrap();
        let model_bytes = base64::engine::general_purpose::STANDARD
            .decode(include_str!("../model_package/testdata/model.onnx.base64").trim())
            .unwrap();
        let mut expected_model_bytes = Vec::new();
        identity_model().encode(&mut expected_model_bytes).unwrap();
        assert_eq!(model_bytes, expected_model_bytes);
        std::fs::write(&model_path, model_bytes).unwrap();
        std::fs::write(&license_path, b"fixture license").unwrap();
        crate::build_runtime_model_package(
            &package_path,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &model_path,
            &license_path,
        )
        .unwrap();

        let package = crate::RuntimeModelPackage::open(&package_path, &public_key_path).unwrap();
        let model = OnnxWaveformModel::load_runtime_package_with_accelerator(
            &package,
            AcceleratorRuntime::Cpu,
        )
        .unwrap();
        let input = vec![vec![-0.5, 0.0, 0.25, 0.75]];
        assert_eq!(model.process(&input, 16_000, false).unwrap(), input);
    }

    #[test]
    fn v2_package_checks_named_graph_and_numerical_vector_before_audio() {
        let directory = tempfile::tempdir().unwrap();
        let package_path = directory.path().join("identity-v2.dmp");
        let mut model_bytes = Vec::new();
        identity_model().encode(&mut model_bytes).unwrap();
        let license = b"fixture license".to_vec();
        let provenance = br#"{"schema":"fixture-provenance-v1"}"#.to_vec();
        let vectors = serde_json::to_vec(&serde_json::json!({
            "schema": "denoize-runtime-model-numerical-vectors-v1",
            "profile_id": "fp32",
            "cases": [{
                "id": "identity",
                "inputs": [{
                    "name": "input",
                    "element_type": "float32",
                    "shape": [1, 4],
                    "values": [-0.5, 0.0, 0.25, 0.75]
                }],
                "outputs": [{
                    "name": "output",
                    "element_type": "float32",
                    "shape": [1, 4],
                    "values": [-0.5, 0.0, 0.25, 0.75]
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
                    name: "samples".into(),
                    kind: "sample".into(),
                    fixed: None,
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
            package_id: "denoize.test.identity-v2".into(),
            package_revision: "1".into(),
            signing_key_id: "0000000000000001".into(),
            runtime: crate::RuntimeModelRuntimeContractV2 {
                kind: "onnx-audio-graph-v2".into(),
                sample_rate_hz: 16_000,
                mode: "finite-and-streaming".into(),
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
                    name: "input".into(),
                    role: "audio".into(),
                    element_type: "float32".into(),
                    axes: axes(),
                    optional: false,
                    state_id: None,
                }],
                outputs: vec![crate::RuntimeModelTensorContractV2 {
                    name: "output".into(),
                    role: "audio".into(),
                    element_type: "float32".into(),
                    axes: axes(),
                    optional: false,
                    state_id: None,
                }],
            },
            state_pairs: vec![],
            latency: crate::RuntimeModelLatencyContractV2 {
                frame_samples: 4,
                hop_samples: 4,
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
                source_repository: "https://example.invalid/source".into(),
                source_revision: "0123456789abcdef".into(),
                source_sha256: "0".repeat(64),
                source_license_spdx: "MIT".into(),
                checkpoint_source: "https://example.invalid/checkpoint".into(),
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
        let mut streaming_only = manifest.clone();
        streaming_only.runtime.mode = "streaming".into();
        assert!(validate_v2_generic_waveform_compatibility(&streaming_only)
            .unwrap_err()
            .contains("dedicated adapter"));
        let mut program_audio = manifest.clone();
        program_audio.frontend.channels.policy = "program-multichannel".into();
        assert!(validate_v2_generic_waveform_compatibility(&program_audio)
            .unwrap_err()
            .contains("dedicated adapter"));
        let package = crate::RuntimeModelPackage::for_onnx_v2_contract_test(
            package_path,
            manifest,
            vec![model_bytes, license, provenance, vectors],
        );
        assert_eq!(package.info().format_version, 2);
        assert_eq!(package.info().v2.unwrap().numerical_vector_cases, 1);
        let model = OnnxWaveformModel::load_runtime_package_with_accelerator(
            &package,
            AcceleratorRuntime::Cpu,
        )
        .unwrap();
        let input = vec![vec![-0.5, 0.0, 0.25, 0.75]];
        assert_eq!(model.process(&input, 16_000, false).unwrap(), input);
    }

    fn cast_output_model() -> ModelProto {
        let mut model = identity_model_with_facts(
            vec![dimension_value(1), dimension_parameter("samples")],
            vec![dimension_value(1), dimension_parameter("samples")],
            tensor_proto::DataType::Float,
            tensor_proto::DataType::Int32,
        );
        model.graph.as_mut().unwrap().node[0] = NodeProto {
            input: vec!["input".into()],
            output: vec!["output".into()],
            name: "cast".into(),
            op_type: "Cast".into(),
            attribute: vec![AttributeProto {
                name: "to".into(),
                r#type: attribute_proto::AttributeType::Int as i32,
                i: tensor_proto::DataType::Int32 as i64,
                ..Default::default()
            }],
            ..Default::default()
        };
        model
    }

    fn constant_output_model(input_samples: i64, output_samples: i64) -> ModelProto {
        let mut model = identity_model_with_facts(
            vec![dimension_value(1), dimension_value(input_samples)],
            vec![dimension_value(1), dimension_value(output_samples)],
            tensor_proto::DataType::Float,
            tensor_proto::DataType::Float,
        );
        model.graph.as_mut().unwrap().node[0] = NodeProto {
            output: vec!["output".into()],
            name: "constant".into(),
            op_type: "Constant".into(),
            attribute: vec![AttributeProto {
                name: "value".into(),
                r#type: attribute_proto::AttributeType::Tensor as i32,
                t: Some(TensorProto {
                    dims: vec![1, output_samples],
                    data_type: tensor_proto::DataType::Float as i32,
                    float_data: vec![0.0; output_samples as usize],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        model
    }

    fn identity_model_with_facts(
        input_dimensions: Vec<tensor_shape_proto::Dimension>,
        output_dimensions: Vec<tensor_shape_proto::Dimension>,
        input_type: tensor_proto::DataType,
        output_type: tensor_proto::DataType,
    ) -> ModelProto {
        let value_info =
            |name: &str,
             dimensions: Vec<tensor_shape_proto::Dimension>,
             datum_type: tensor_proto::DataType| ValueInfoProto {
                name: name.into(),
                r#type: Some(TypeProto {
                    denotation: String::new(),
                    value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                        elem_type: datum_type as i32,
                        shape: Some(TensorShapeProto { dim: dimensions }),
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
                name: "identity-waveform".into(),
                node: vec![NodeProto {
                    input: vec!["input".into()],
                    output: vec!["output".into()],
                    name: "identity".into(),
                    op_type: "Identity".into(),
                    ..Default::default()
                }],
                input: vec![value_info("input", input_dimensions, input_type)],
                output: vec![value_info("output", output_dimensions, output_type)],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn write_model(model: ModelProto) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        std::fs::write(&path, bytes).unwrap();
        (directory, path)
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
