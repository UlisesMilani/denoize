//! Version 2 of the signed runtime-model package contract.
//!
//! The v2 container remains a length-delimited, non-extracting format.  It
//! adds an ordered component table so one signed package can carry multiple
//! ONNX precision profiles, one consolidated license notice, provenance, and
//! bounded numerical conformance vectors without permitting scripts or
//! archive paths.

use super::*;
use std::collections::{BTreeMap, HashMap};

pub const RUNTIME_MODEL_PACKAGE_SCHEMA_V2: &str = "denoize-runtime-model-package-v2";
pub const RUNTIME_MODEL_PACKAGE_VERSION_V2: u32 = 2;

pub(super) const PACKAGE_MAGIC_V2: &[u8] = b"denoize-runtime-model-package-v2\n";
const HEADER_FIXED_FIELDS: u64 = 3 * 8;
const MAX_COMPONENTS: u64 = 32;
const MAX_PROVENANCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_NUMERICAL_VECTORS_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSORS: usize = 32;
const MAX_AXES: usize = 8;
const MAX_PROFILES: usize = 8;
const MAX_STATE_PAIRS: usize = 16;
const MAX_CHANNELS: usize = 256;
const MAX_DATASETS: usize = 64;
const MAX_VECTOR_CASES: usize = 16;
const MAX_VECTOR_TENSOR_ELEMENTS: usize = 1 << 20;
const MAX_VECTOR_TOTAL_ELEMENTS: usize = 1 << 22;
const MAX_VECTOR_TOLERANCE: f32 = 0.01;

const RUNTIME_KIND_V2: &str = "onnx-audio-graph-v2";
const NORMALIZATION_V2: &str = "pcm-f32-minus-one-to-one-v1";
const RESAMPLING_V2: &str = "bandlimited-waveform-v1";
const DURATION_V2: &str = "preserve-input-frames-v1";

/// A named dimension in a v2 ONNX tensor contract.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelAxisContractV2 {
    pub name: String,
    /// `batch`, `channel`, `sample`, `frame`, `frequency`, `feature`,
    /// `coordinate`, or `state`.
    pub kind: String,
    pub fixed: Option<u64>,
}

/// One exact, named graph input or output.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelTensorContractV2 {
    pub name: String,
    /// Input roles are `audio`, `far-end-reference`, `enrollment`,
    /// `microphone-geometry`, `state`, and `control`. Output roles are
    /// `audio`, `state`, `mask`, and `diagnostic`.
    pub role: String,
    pub element_type: String,
    pub axes: Vec<RuntimeModelAxisContractV2>,
    pub optional: bool,
    pub state_id: Option<String>,
}

/// Closed graph input/output declaration.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelTensorSetContractV2 {
    pub inputs: Vec<RuntimeModelTensorContractV2>,
    pub outputs: Vec<RuntimeModelTensorContractV2>,
}

/// An explicit recurrent-state edge carried between inference calls.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelStatePairContractV2 {
    pub id: String,
    pub input: String,
    pub output: String,
    /// v2 intentionally permits only deterministic zero initialization.
    pub initialization: String,
}

/// A semantic role for a fixed audio channel.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelChannelRoleContractV2 {
    pub channel_index: u32,
    pub role: String,
}

/// Fixed microphone position in integer millimetres.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelMicrophonePositionV2 {
    pub channel_index: u32,
    pub x_mm: i32,
    pub y_mm: i32,
    pub z_mm: i32,
}

/// A fixed, right-handed microphone-array geometry.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelGeometryContractV2 {
    pub coordinate_system: String,
    pub units: String,
    pub microphones: Vec<RuntimeModelMicrophonePositionV2>,
}

/// Audio channel interpretation around the graph.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelChannelContractV2 {
    /// `independent-mono`, `program-multichannel`, or `microphone-array`.
    pub policy: String,
    pub roles: Vec<RuntimeModelChannelRoleContractV2>,
    pub geometry: Option<RuntimeModelGeometryContractV2>,
}

/// Runtime and streaming mode declared by a v2 package.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelRuntimeContractV2 {
    pub kind: String,
    pub sample_rate_hz: u32,
    /// `finite`, `streaming`, or `finite-and-streaming`.
    pub mode: String,
}

/// Signed audio-domain transformations around a v2 graph.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelFrontendContractV2 {
    pub normalization: String,
    pub resampling: String,
    pub duration: String,
    pub channels: RuntimeModelChannelContractV2,
}

/// Frame, context, and latency accounting in model-rate samples.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelLatencyContractV2 {
    pub frame_samples: u64,
    pub hop_samples: u64,
    pub left_context_samples: u64,
    pub right_context_samples: u64,
    pub lookahead_samples: u64,
    pub algorithmic_latency_samples: u64,
    pub flush_samples: u64,
}

/// One ordered, authenticated byte component in the v2 container.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelComponentContractV2 {
    pub id: String,
    /// `onnx-model`, `license-notice`, `provenance-json`, or
    /// `numerical-vectors-json`.
    pub kind: String,
    pub file: RuntimeModelFileContract,
}

/// One selectable model precision and its exact resource contract.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelPrecisionProfileContractV2 {
    pub id: String,
    pub element_type: String,
    pub model_component: String,
    pub numerical_vectors_component: String,
    pub resources: RuntimeModelResourceContract,
}

/// One training dataset disclosed by the signed provenance summary.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelTrainingDatasetContractV2 {
    pub id: String,
    pub source: String,
    pub revision: String,
    pub sha256: Option<String>,
    pub license_spdx: String,
}

/// Source, checkpoint, conversion, and training-data provenance.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelProvenanceContractV2 {
    pub component: String,
    pub source_repository: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub source_license_spdx: String,
    pub checkpoint_source: String,
    pub checkpoint_sha256: String,
    pub checkpoint_license_spdx: String,
    pub conversion_tool: String,
    pub conversion_revision: String,
    pub training_datasets: Vec<RuntimeModelTrainingDatasetContractV2>,
}

/// Consolidated notice and SPDX expression for every carried component.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelLicenseContractV2 {
    pub spdx: String,
    pub notice_component: String,
}

/// Signed v2 manifest embedded in a `.dmp` package.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelPackageManifestV2 {
    pub schema: String,
    pub format_version: u32,
    pub package_id: String,
    pub package_revision: String,
    pub signing_key_id: String,
    pub runtime: RuntimeModelRuntimeContractV2,
    pub frontend: RuntimeModelFrontendContractV2,
    pub tensors: RuntimeModelTensorSetContractV2,
    pub state_pairs: Vec<RuntimeModelStatePairContractV2>,
    pub latency: RuntimeModelLatencyContractV2,
    pub components: Vec<RuntimeModelComponentContractV2>,
    pub precision_profiles: Vec<RuntimeModelPrecisionProfileContractV2>,
    pub default_precision_profile: String,
    pub license: RuntimeModelLicenseContractV2,
    pub provenance: RuntimeModelProvenanceContractV2,
}

/// Path-free v2 details returned by package inspection.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeModelPackageV2Info {
    pub runtime_kind: String,
    pub runtime_mode: String,
    pub channel_policy: String,
    pub channel_roles: Vec<RuntimeModelChannelRoleContractV2>,
    pub geometry: Option<RuntimeModelGeometryContractV2>,
    pub inputs: Vec<RuntimeModelTensorContractV2>,
    pub outputs: Vec<RuntimeModelTensorContractV2>,
    pub state_pairs: Vec<RuntimeModelStatePairContractV2>,
    pub latency: RuntimeModelLatencyContractV2,
    pub default_precision_profile: String,
    pub precision_profiles: Vec<RuntimeModelPrecisionProfileContractV2>,
    pub provenance: RuntimeModelProvenanceContractV2,
    pub component_count: u32,
    pub numerical_vector_cases: u32,
}

/// One tensor in a bounded numerical conformance case.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelNumericalTensorV1 {
    pub name: String,
    pub element_type: String,
    pub shape: Vec<u64>,
    pub values: Vec<f64>,
}

/// Absolute and relative comparison tolerances for one case.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelNumericalToleranceV1 {
    pub absolute: f32,
    pub relative: f32,
}

/// One complete set of graph inputs and expected graph outputs.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelNumericalCaseV1 {
    pub id: String,
    pub inputs: Vec<RuntimeModelNumericalTensorV1>,
    pub outputs: Vec<RuntimeModelNumericalTensorV1>,
    pub tolerance: RuntimeModelNumericalToleranceV1,
}

/// Bounded numerical vectors authenticated as a v2 component.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelNumericalVectorsV1 {
    pub schema: String,
    pub profile_id: String,
    pub cases: Vec<RuntimeModelNumericalCaseV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ComponentRangeV2 {
    pub id: String,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OpenedRuntimeModelPackageV2 {
    pub manifest: RuntimeModelPackageManifestV2,
    pub components: Vec<ComponentRangeV2>,
    pub info: RuntimeModelPackageV2Info,
}

#[derive(Debug)]
pub(super) struct PreparedPackageV2 {
    pub info: RuntimeModelPackageInfo,
    pub compatibility_manifest: RuntimeModelPackageManifest,
    pub model_offset: u64,
    pub license_offset: u64,
    pub opened: OpenedRuntimeModelPackageV2,
}

pub(super) fn has_v2_magic(file: &mut File, path: &Path) -> Result<bool, String> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind runtime model package {}: {error}", path.display()))?;
    let mut magic = vec![0_u8; PACKAGE_MAGIC_V2.len()];
    let read = file.read(&mut magic).map_err(|error| {
        format!(
            "read runtime model package magic {}: {error}",
            path.display()
        )
    })?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind runtime model package {}: {error}", path.display()))?;
    Ok(read == PACKAGE_MAGIC_V2.len() && magic == PACKAGE_MAGIC_V2)
}

pub(super) fn prepare_open_package_v2<F>(
    file: &mut File,
    path: &Path,
    size: u64,
    public_key: &ParsedPublicKey,
    mut verify: F,
) -> Result<PreparedPackageV2, String>
where
    F: FnMut(&[u8], &[u8], &ParsedPublicKey) -> Result<(), String>,
{
    let base_minimum = (PACKAGE_MAGIC_V2.len() as u64)
        .checked_add(HEADER_FIXED_FIELDS)
        .ok_or_else(|| "runtime model package v2 size accounting overflow".to_string())?;
    if size < base_minimum {
        return Err("runtime model package v2 is truncated".into());
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind runtime model package {}: {error}", path.display()))?;
    let mut package_hasher = Sha256::new();
    let mut magic = vec![0_u8; PACKAGE_MAGIC_V2.len()];
    file.read_exact(&mut magic)
        .map_err(|error| format!("read runtime model package v2 magic: {error}"))?;
    if magic != PACKAGE_MAGIC_V2 {
        return Err("runtime model package has an unsupported magic/version".into());
    }
    package_hasher.update(&magic);
    let manifest_len = read_u64(file, "v2 manifest length", &mut package_hasher)?;
    let signature_len = read_u64(file, "v2 signature length", &mut package_hasher)?;
    let component_count = read_u64(file, "v2 component count", &mut package_hasher)?;
    require_bounded_length(manifest_len, 1, MAX_MANIFEST_BYTES, "v2 manifest")?;
    require_bounded_length(signature_len, 1, MAX_SIGNATURE_BYTES, "v2 signature")?;
    require_bounded_length(component_count, 1, MAX_COMPONENTS, "v2 component count")?;
    let mut component_lengths = Vec::with_capacity(component_count as usize);
    for index in 0..component_count {
        let length = read_u64(
            file,
            &format!("v2 component {index} length"),
            &mut package_hasher,
        )?;
        require_bounded_length(length, 1, MAX_MODEL_BYTES, "v2 component")?;
        component_lengths.push(length);
    }
    let header_size = base_minimum
        .checked_add(
            component_count
                .checked_mul(8)
                .ok_or_else(|| "runtime model package v2 header size overflow".to_string())?,
        )
        .ok_or_else(|| "runtime model package v2 header size overflow".to_string())?;
    let expected_size = component_lengths.iter().try_fold(
        header_size
            .checked_add(manifest_len)
            .and_then(|value| value.checked_add(signature_len))
            .ok_or_else(|| "runtime model package v2 size accounting overflow".to_string())?,
        |total, length| {
            total
                .checked_add(*length)
                .ok_or_else(|| "runtime model package v2 size accounting overflow".to_string())
        },
    )?;
    if expected_size != size {
        return Err(format!(
            "runtime model package v2 length mismatch: header declares {expected_size} bytes, file has {size}"
        ));
    }
    let manifest_bytes = read_exact_bounded(file, manifest_len, "runtime model v2 manifest")?;
    package_hasher.update(&manifest_bytes);
    let signature_bytes = read_exact_bounded(file, signature_len, "runtime model v2 signature")?;
    package_hasher.update(&signature_bytes);
    verify(&manifest_bytes, &signature_bytes, public_key)?;
    let manifest: RuntimeModelPackageManifestV2 = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("invalid runtime model v2 manifest JSON: {error}"))?;
    validate_manifest_v2(&manifest, &public_key.key_id)?;
    if manifest.components.len() != component_lengths.len() {
        return Err("runtime model v2 manifest component count does not match framing".into());
    }
    for (component, framed) in manifest.components.iter().zip(&component_lengths) {
        if component.file.size_bytes != *framed {
            return Err(format!(
                "runtime model v2 component {} size does not match package framing",
                component.id
            ));
        }
    }

    let mut offset = header_size
        .checked_add(manifest_len)
        .and_then(|value| value.checked_add(signature_len))
        .ok_or_else(|| "runtime model package v2 component offset overflow".to_string())?;
    let mut ranges = Vec::with_capacity(manifest.components.len());
    for component in &manifest.components {
        ranges.push(ComponentRangeV2 {
            id: component.id.clone(),
            offset,
            length: component.file.size_bytes,
            sha256: component.file.sha256.clone(),
        });
        require_stream_hash(
            file,
            component.file.size_bytes,
            &component.file.sha256,
            &format!("runtime model v2 component {}", component.id),
            &mut package_hasher,
        )?;
        offset = offset
            .checked_add(component.file.size_bytes)
            .ok_or_else(|| "runtime model package v2 component offset overflow".to_string())?;
    }
    let package_sha256 = format!("{:x}", package_hasher.finalize());
    validate_carried_provenance(file, &manifest, &ranges)?;
    let numerical_vector_cases = validate_carried_vectors(file, &manifest, &ranges)?;
    let compatibility_manifest = compatibility_manifest(&manifest)?;
    let default_profile = selected_profile(&manifest, AcceleratorRuntime::Cpu)?;
    let model_range = range_for_component(&ranges, &default_profile.model_component)?;
    let license_range = range_for_component(&ranges, &manifest.license.notice_component)?;
    let info = package_info_v2(
        &manifest,
        &compatibility_manifest,
        package_sha256,
        size,
        numerical_vector_cases,
    );
    let v2_info = info
        .v2
        .clone()
        .expect("v2 package inspection always contains v2 details");
    Ok(PreparedPackageV2 {
        info,
        compatibility_manifest,
        model_offset: model_range.offset,
        license_offset: license_range.offset,
        opened: OpenedRuntimeModelPackageV2 {
            manifest,
            components: ranges,
            info: v2_info,
        },
    })
}

/// Assemble a deterministic v2 package from an authenticated manifest and a
/// directory containing its exact component basenames.
pub fn build_runtime_model_package_v2(
    output: impl AsRef<Path>,
    manifest_path: impl AsRef<Path>,
    signature_path: impl AsRef<Path>,
    public_key_path: impl AsRef<Path>,
    components_directory: impl AsRef<Path>,
) -> Result<RuntimeModelPackageInfo, String> {
    let output = output.as_ref();
    let manifest_path = manifest_path.as_ref();
    let signature_path = signature_path.as_ref();
    let public_key_path = public_key_path.as_ref();
    let components_directory = components_directory.as_ref();
    let directory_metadata = std::fs::symlink_metadata(components_directory).map_err(|error| {
        format!(
            "inspect runtime model v2 component directory {}: {error}",
            components_directory.display()
        )
    })?;
    if directory_metadata.file_type().is_symlink() || !directory_metadata.is_dir() {
        return Err("runtime model v2 component directory must be a regular directory".into());
    }
    let (manifest_bytes, _) = read_bounded_regular_file(
        manifest_path,
        "runtime model v2 manifest",
        MAX_MANIFEST_BYTES,
    )?;
    let (signature_bytes, _) = read_bounded_regular_file(
        signature_path,
        "runtime model v2 manifest signature",
        MAX_SIGNATURE_BYTES,
    )?;
    let (public_key_bytes, _) = read_bounded_regular_file(
        public_key_path,
        "runtime model public key",
        MAX_PUBLIC_KEY_BYTES,
    )?;
    let public_key_text = std::str::from_utf8(&public_key_bytes)
        .map_err(|_| "runtime model public key is not UTF-8".to_string())?;
    let public_key = parse_public_key(public_key_text)?;
    verify_manifest_signature(&manifest_bytes, &signature_bytes, &public_key)?;
    let manifest: RuntimeModelPackageManifestV2 = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("invalid runtime model v2 manifest JSON: {error}"))?;
    validate_manifest_v2(&manifest, &public_key.key_id)?;

    let mut sources = Vec::with_capacity(manifest.components.len());
    for component in &manifest.components {
        let path = components_directory.join(&component.file.filename);
        require_filename(
            &path,
            &component.file.filename,
            &format!("runtime model v2 component {}", component.id),
        )?;
        let (mut file, length) = crate::input::open_regular_file(
            &path,
            &format!("runtime model v2 component {}", component.id),
        )?;
        require_component_identity(
            &mut file,
            &path,
            length,
            &component.file,
            &format!("runtime model v2 component {}", component.id),
        )?;
        sources.push((path, file));
    }

    let mut staged = AtomicOutput::new(output)?;
    staged
        .file_mut()
        .write_all(PACKAGE_MAGIC_V2)
        .and_then(|_| {
            staged
                .file_mut()
                .write_all(&(manifest_bytes.len() as u64).to_be_bytes())
        })
        .and_then(|_| {
            staged
                .file_mut()
                .write_all(&(signature_bytes.len() as u64).to_be_bytes())
        })
        .and_then(|_| {
            staged
                .file_mut()
                .write_all(&(manifest.components.len() as u64).to_be_bytes())
        })
        .map_err(|error| {
            format!(
                "write runtime model v2 package {}: {error}",
                output.display()
            )
        })?;
    for component in &manifest.components {
        staged
            .file_mut()
            .write_all(&component.file.size_bytes.to_be_bytes())
            .map_err(|error| {
                format!(
                    "write runtime model v2 package {}: {error}",
                    output.display()
                )
            })?;
    }
    staged
        .file_mut()
        .write_all(&manifest_bytes)
        .and_then(|_| staged.file_mut().write_all(&signature_bytes))
        .map_err(|error| {
            format!(
                "write runtime model v2 package {}: {error}",
                output.display()
            )
        })?;
    for ((path, source), component) in sources.iter_mut().zip(&manifest.components) {
        copy_exact_component(source, path, staged.file_mut(), &component.file)?;
    }
    staged.file_mut().flush().map_err(|error| {
        format!(
            "flush runtime model v2 package {}: {error}",
            output.display()
        )
    })?;
    let size = staged
        .file_mut()
        .metadata()
        .map_err(|error| format!("inspect staged runtime model v2 package: {error}"))?
        .len();
    let prepared = prepare_open_package_v2(
        staged.file_mut(),
        output,
        size,
        &public_key,
        verify_manifest_signature,
    )?;
    staged.commit(CommitMode::NoClobber)?;
    Ok(prepared.info)
}

pub(super) fn selected_profile(
    manifest: &RuntimeModelPackageManifestV2,
    runtime: AcceleratorRuntime,
) -> Result<&RuntimeModelPrecisionProfileContractV2, String> {
    let default = manifest
        .precision_profiles
        .iter()
        .find(|profile| profile.id == manifest.default_precision_profile)
        .ok_or_else(|| "runtime model v2 default precision profile is missing".to_string())?;
    if default
        .resources
        .accelerators
        .iter()
        .any(|accelerator| accelerator == runtime.name())
    {
        return Ok(default);
    }
    manifest
        .precision_profiles
        .iter()
        .find(|profile| {
            profile
                .resources
                .accelerators
                .iter()
                .any(|accelerator| accelerator == runtime.name())
        })
        .ok_or_else(|| {
            format!(
                "runtime model v2 package has no precision profile for the {} accelerator",
                runtime.name()
            )
        })
}

pub(super) fn range_for_component<'a>(
    ranges: &'a [ComponentRangeV2],
    id: &str,
) -> Result<&'a ComponentRangeV2, String> {
    ranges
        .iter()
        .find(|range| range.id == id)
        .ok_or_else(|| format!("runtime model v2 component {id} is missing"))
}

pub(crate) fn parse_numerical_vectors(
    bytes: &[u8],
    manifest: &RuntimeModelPackageManifestV2,
    profile: &RuntimeModelPrecisionProfileContractV2,
) -> Result<RuntimeModelNumericalVectorsV1, String> {
    let vectors: RuntimeModelNumericalVectorsV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid runtime model numerical vectors JSON: {error}"))?;
    validate_numerical_vectors(&vectors, manifest, profile)?;
    Ok(vectors)
}

fn validate_carried_vectors(
    file: &mut File,
    manifest: &RuntimeModelPackageManifestV2,
    ranges: &[ComponentRangeV2],
) -> Result<u32, String> {
    let mut total_cases = 0_u32;
    let mut seen = HashSet::new();
    for profile in &manifest.precision_profiles {
        if !seen.insert(profile.numerical_vectors_component.as_str()) {
            return Err(
                "runtime model v2 precision profiles must use distinct numerical vector components"
                    .into(),
            );
        }
        let range = range_for_component(ranges, &profile.numerical_vectors_component)?;
        file.seek(SeekFrom::Start(range.offset)).map_err(|error| {
            format!(
                "seek runtime model numerical vectors component {}: {error}",
                profile.numerical_vectors_component
            )
        })?;
        let bytes = read_exact_bounded(
            file,
            range.length,
            &format!(
                "runtime model numerical vectors component {}",
                profile.numerical_vectors_component
            ),
        )?;
        let vectors = parse_numerical_vectors(&bytes, manifest, profile)?;
        total_cases = total_cases
            .checked_add(vectors.cases.len() as u32)
            .ok_or_else(|| "runtime model numerical vector case count overflow".to_string())?;
    }
    Ok(total_cases)
}

fn validate_carried_provenance(
    file: &mut File,
    manifest: &RuntimeModelPackageManifestV2,
    ranges: &[ComponentRangeV2],
) -> Result<(), String> {
    let range = range_for_component(ranges, &manifest.provenance.component)?;
    file.seek(SeekFrom::Start(range.offset)).map_err(|error| {
        format!(
            "seek runtime model provenance component {}: {error}",
            manifest.provenance.component
        )
    })?;
    let bytes = read_exact_bounded(
        file,
        range.length,
        &format!(
            "runtime model provenance component {}",
            manifest.provenance.component
        ),
    )?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid runtime model provenance JSON: {error}"))?;
    if !value.is_object() {
        return Err("runtime model provenance JSON must be an object".into());
    }
    Ok(())
}

fn validate_manifest_v2(
    manifest: &RuntimeModelPackageManifestV2,
    expected_key_id: &str,
) -> Result<(), String> {
    if manifest.schema != RUNTIME_MODEL_PACKAGE_SCHEMA_V2
        || manifest.format_version != RUNTIME_MODEL_PACKAGE_VERSION_V2
    {
        return Err("runtime model v2 manifest has an unsupported schema/version".into());
    }
    validate_identifier(&manifest.package_id, "package id")?;
    validate_identifier(&manifest.package_revision, "package revision")?;
    if !valid_key_id(&manifest.signing_key_id) {
        return Err("runtime model v2 manifest has an invalid signing key id".into());
    }
    if manifest.signing_key_id != expected_key_id {
        return Err(
            "runtime model v2 manifest signing key id does not match the trusted key".into(),
        );
    }
    if manifest.runtime.kind != RUNTIME_KIND_V2 {
        return Err(format!(
            "unsupported runtime model v2 adapter: {}",
            manifest.runtime.kind
        ));
    }
    if manifest.runtime.sample_rate_hz == 0
        || manifest.runtime.sample_rate_hz > crate::config::MAX_SAMPLE_RATE
    {
        return Err("runtime model v2 sample rate is outside 1..=768000 Hz".into());
    }
    if !matches!(
        manifest.runtime.mode.as_str(),
        "finite" | "streaming" | "finite-and-streaming"
    ) {
        return Err("runtime model v2 mode is unsupported".into());
    }
    if manifest.frontend.normalization != NORMALIZATION_V2
        || manifest.frontend.resampling != RESAMPLING_V2
        || manifest.frontend.duration != DURATION_V2
    {
        return Err("runtime model v2 package declares an unsupported frontend contract".into());
    }
    validate_channels(&manifest.frontend.channels, &manifest.tensors)?;
    validate_tensors(&manifest.tensors, &manifest.state_pairs)?;
    validate_latency(&manifest.latency)?;
    validate_components_and_profiles(manifest)?;
    validate_provenance(&manifest.provenance)?;
    validate_spdx(&manifest.license.spdx)?;
    Ok(())
}

fn validate_channels(
    channels: &RuntimeModelChannelContractV2,
    tensors: &RuntimeModelTensorSetContractV2,
) -> Result<(), String> {
    if !matches!(
        channels.policy.as_str(),
        "independent-mono" | "program-multichannel" | "microphone-array"
    ) {
        return Err("runtime model v2 channel policy is unsupported".into());
    }
    if channels.roles.len() > MAX_CHANNELS {
        return Err("runtime model v2 declares too many channel roles".into());
    }
    let mut role_indices = HashSet::new();
    for role in &channels.roles {
        if usize::try_from(role.channel_index).unwrap_or(usize::MAX) >= MAX_CHANNELS
            || !matches!(
                role.role.as_str(),
                "program-left"
                    | "program-right"
                    | "program-center"
                    | "program-lfe"
                    | "program-surround"
                    | "microphone"
            )
            || !role_indices.insert(role.channel_index)
        {
            return Err("runtime model v2 channel role is invalid or repeated".into());
        }
    }
    let audio_input = tensors.inputs.iter().find(|tensor| tensor.role == "audio");
    let channel_axis = audio_input.and_then(|tensor| {
        tensor
            .axes
            .iter()
            .find(|axis| axis.kind.as_str() == "channel")
    });
    let fixed_audio_channels = channel_axis.and_then(|axis| axis.fixed);
    let geometry_inputs: Vec<&RuntimeModelTensorContractV2> = tensors
        .inputs
        .iter()
        .filter(|tensor| tensor.role == "microphone-geometry")
        .collect();
    if geometry_inputs.len() > 1 {
        return Err("runtime model v2 declares multiple microphone geometry inputs".into());
    }
    if let Some(geometry) = &channels.geometry {
        if channels.policy != "microphone-array"
            || geometry.coordinate_system != "right-handed-cartesian"
            || geometry.units != "millimeters"
            || geometry.microphones.is_empty()
            || geometry.microphones.len() > MAX_CHANNELS
        {
            return Err("runtime model v2 microphone geometry is invalid".into());
        }
        let mut indices = HashSet::new();
        for microphone in &geometry.microphones {
            if usize::try_from(microphone.channel_index).unwrap_or(usize::MAX) >= MAX_CHANNELS
                || !indices.insert(microphone.channel_index)
                || microphone.x_mm.unsigned_abs() > 100_000
                || microphone.y_mm.unsigned_abs() > 100_000
                || microphone.z_mm.unsigned_abs() > 100_000
            {
                return Err("runtime model v2 microphone position is invalid or repeated".into());
            }
        }
    }
    match channels.policy.as_str() {
        "independent-mono" => {
            if !channels.roles.is_empty()
                || channels.geometry.is_some()
                || !geometry_inputs.is_empty()
                || channel_axis.is_some_and(|axis| axis.fixed != Some(1))
            {
                return Err(
                    "independent-mono v2 packages require a unit channel axis and cannot declare roles or geometry"
                        .into(),
                );
            }
        }
        "program-multichannel" => {
            let count = fixed_audio_channels.ok_or_else(|| {
                "program-multichannel v2 packages require a fixed audio channel dimension"
                    .to_string()
            })?;
            if channels.geometry.is_some()
                || !geometry_inputs.is_empty()
                || channels.roles.len() as u64 != count
                || channels
                    .roles
                    .iter()
                    .any(|role| role.role == "microphone" || u64::from(role.channel_index) >= count)
                || (0..count).any(|index| !role_indices.contains(&(index as u32)))
            {
                return Err(
                    "program-multichannel v2 roles must cover every fixed program channel".into(),
                );
            }
        }
        "microphone-array" => {
            if channel_axis.is_none() {
                return Err(
                    "microphone-array v2 packages require an audio channel dimension".into(),
                );
            }
            if channels.geometry.is_some() == !geometry_inputs.is_empty() {
                return Err(
                    "microphone-array v2 packages require exactly one fixed or typed geometry source"
                        .into(),
                );
            }
            let fixed_geometry_channels = channels
                .geometry
                .as_ref()
                .map(|geometry| geometry.microphones.len() as u64);
            if fixed_audio_channels
                .zip(fixed_geometry_channels)
                .is_some_and(|(audio, geometry)| audio != geometry)
            {
                return Err(
                    "runtime model v2 microphone geometry does not match the audio channel count"
                        .into(),
                );
            }
            let expected_channels = fixed_audio_channels.or(fixed_geometry_channels);
            if channels.roles.iter().any(|role| role.role != "microphone")
                || expected_channels.is_some_and(|count| {
                    channels.roles.len() as u64 != count
                        || channels
                            .roles
                            .iter()
                            .any(|role| u64::from(role.channel_index) >= count)
                        || (0..count).any(|index| !role_indices.contains(&(index as u32)))
                })
                || (expected_channels.is_none() && !channels.roles.is_empty())
            {
                return Err(
                    "microphone-array v2 roles must cover every known microphone channel".into(),
                );
            }
            if let Some(geometry) = &channels.geometry {
                let count = geometry.microphones.len() as u32;
                if (0..count).any(|index| {
                    !geometry
                        .microphones
                        .iter()
                        .any(|microphone| microphone.channel_index == index)
                }) {
                    return Err(
                        "runtime model v2 microphone geometry indices must be contiguous".into(),
                    );
                }
            }
            if let Some(geometry_input) = geometry_inputs.first() {
                if geometry_input
                    .axes
                    .iter()
                    .filter(|axis| axis.kind == "channel")
                    .count()
                    != 1
                    || geometry_input
                        .axes
                        .iter()
                        .filter(|axis| axis.kind == "coordinate")
                        .count()
                        != 1
                {
                    return Err(
                        "runtime model v2 geometry input requires one channel and one coordinate axis"
                            .into(),
                    );
                }
                let input_channel = geometry_input
                    .axes
                    .iter()
                    .find(|axis| axis.kind == "channel")
                    .ok_or_else(|| {
                        "runtime model v2 geometry input requires a channel axis".to_string()
                    })?;
                let coordinate = geometry_input
                    .axes
                    .iter()
                    .find(|axis| axis.kind == "coordinate")
                    .ok_or_else(|| {
                        "runtime model v2 geometry input requires a coordinate axis".to_string()
                    })?;
                if coordinate.fixed != Some(3)
                    || fixed_audio_channels
                        .zip(input_channel.fixed)
                        .is_some_and(|(audio, geometry)| audio != geometry)
                {
                    return Err(
                        "runtime model v2 geometry tensor does not match three-dimensional audio channels"
                            .into(),
                    );
                }
            }
        }
        _ => unreachable!("channel policy was checked above"),
    }
    Ok(())
}

fn validate_tensors(
    tensors: &RuntimeModelTensorSetContractV2,
    state_pairs: &[RuntimeModelStatePairContractV2],
) -> Result<(), String> {
    if tensors.inputs.is_empty()
        || tensors.outputs.is_empty()
        || tensors.inputs.len() > MAX_TENSORS
        || tensors.outputs.len() > MAX_TENSORS
    {
        return Err("runtime model v2 tensor counts are outside supported bounds".into());
    }
    let mut all_names = HashSet::new();
    for tensor in tensors.inputs.iter().chain(&tensors.outputs) {
        validate_tensor(tensor)?;
        if !all_names.insert(tensor.name.as_str()) {
            return Err("runtime model v2 tensor names must be globally unique".into());
        }
    }
    for tensor in &tensors.inputs {
        if !matches!(
            tensor.role.as_str(),
            "audio"
                | "far-end-reference"
                | "enrollment"
                | "query"
                | "microphone-geometry"
                | "state"
                | "control"
        ) {
            return Err(format!(
                "unsupported runtime model v2 input role {}",
                tensor.role
            ));
        }
    }
    for tensor in &tensors.outputs {
        if tensor.optional
            || !matches!(
                tensor.role.as_str(),
                "audio" | "residual" | "state" | "mask" | "diagnostic"
            )
        {
            return Err(format!(
                "unsupported runtime model v2 output role {}",
                tensor.role
            ));
        }
    }
    if tensors
        .inputs
        .iter()
        .filter(|tensor| tensor.role == "audio")
        .count()
        != 1
        || tensors
            .outputs
            .iter()
            .filter(|tensor| tensor.role == "audio")
            .count()
            != 1
    {
        return Err("runtime model v2 requires exactly one primary audio input and output".into());
    }
    if state_pairs.len() > MAX_STATE_PAIRS {
        return Err("runtime model v2 declares too many recurrent state pairs".into());
    }
    let state_inputs: HashMap<&str, &RuntimeModelTensorContractV2> = tensors
        .inputs
        .iter()
        .filter(|tensor| tensor.role == "state")
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let state_outputs: HashMap<&str, &RuntimeModelTensorContractV2> = tensors
        .outputs
        .iter()
        .filter(|tensor| tensor.role == "state")
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    if state_inputs.len() != state_outputs.len() || state_inputs.len() != state_pairs.len() {
        return Err("runtime model v2 recurrent state tensors must form explicit pairs".into());
    }
    let mut pair_ids = HashSet::new();
    let mut paired_inputs = HashSet::new();
    let mut paired_outputs = HashSet::new();
    for pair in state_pairs {
        validate_identifier(&pair.id, "state pair id")?;
        if pair.initialization != "zeros"
            || !pair_ids.insert(pair.id.as_str())
            || !paired_inputs.insert(pair.input.as_str())
            || !paired_outputs.insert(pair.output.as_str())
        {
            return Err("runtime model v2 recurrent state pair is invalid or repeated".into());
        }
        let input = state_inputs
            .get(pair.input.as_str())
            .ok_or_else(|| "runtime model v2 state-pair input is not a state tensor".to_string())?;
        let output = state_outputs.get(pair.output.as_str()).ok_or_else(|| {
            "runtime model v2 state-pair output is not a state tensor".to_string()
        })?;
        if input.state_id.as_deref() != Some(pair.id.as_str())
            || output.state_id.as_deref() != Some(pair.id.as_str())
            || input.element_type != output.element_type
            || input.axes != output.axes
        {
            return Err("runtime model v2 recurrent state pair contracts do not match".into());
        }
    }
    Ok(())
}

fn validate_tensor(tensor: &RuntimeModelTensorContractV2) -> Result<(), String> {
    validate_tensor_name(&tensor.name)?;
    if !matches!(tensor.element_type.as_str(), "float32" | "int64") {
        return Err("runtime model v2 tensor element type is unsupported".into());
    }
    if tensor.axes.is_empty() || tensor.axes.len() > MAX_AXES {
        return Err("runtime model v2 tensor rank is outside 1..=8".into());
    }
    let mut names = HashSet::new();
    for axis in &tensor.axes {
        validate_identifier(&axis.name, "tensor axis name")?;
        if !names.insert(axis.name.as_str())
            || !matches!(
                axis.kind.as_str(),
                "batch"
                    | "channel"
                    | "sample"
                    | "frame"
                    | "frequency"
                    | "feature"
                    | "coordinate"
                    | "state"
            )
            || axis
                .fixed
                .is_some_and(|value| value == 0 || value > MAX_FIXED_TENSOR_SAMPLES)
        {
            return Err("runtime model v2 tensor axis is invalid or repeated".into());
        }
        if axis.kind == "batch" && axis.fixed.is_some_and(|value| value != 1) {
            return Err("runtime model v2 batch dimensions must be one when fixed".into());
        }
    }
    if tensor.role == "state" {
        let id = tensor
            .state_id
            .as_deref()
            .ok_or_else(|| "runtime model v2 state tensor requires state_id".to_string())?;
        validate_identifier(id, "state id")?;
    } else if tensor.state_id.is_some() {
        return Err("only runtime model v2 state tensors may declare state_id".into());
    }
    Ok(())
}

fn validate_tensor_name(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 255
        || value.chars().any(char::is_control)
        || value.contains(['/', '\\'])
    {
        return Err("runtime model v2 tensor name is invalid".into());
    }
    Ok(())
}

fn validate_latency(latency: &RuntimeModelLatencyContractV2) -> Result<(), String> {
    for value in [
        latency.frame_samples,
        latency.hop_samples,
        latency.left_context_samples,
        latency.right_context_samples,
        latency.lookahead_samples,
        latency.algorithmic_latency_samples,
        latency.flush_samples,
    ] {
        if value > MAX_FIXED_TENSOR_SAMPLES {
            return Err("runtime model v2 latency/context value is too large".into());
        }
    }
    if latency.frame_samples == 0
        || latency.hop_samples == 0
        || latency.hop_samples > latency.frame_samples
        || latency.lookahead_samples > latency.right_context_samples
        || latency.algorithmic_latency_samples < latency.lookahead_samples
    {
        return Err("runtime model v2 latency/context relationship is invalid".into());
    }
    Ok(())
}

fn validate_components_and_profiles(
    manifest: &RuntimeModelPackageManifestV2,
) -> Result<(), String> {
    if manifest.components.len() < 4 || manifest.components.len() > MAX_COMPONENTS as usize {
        return Err("runtime model v2 component count is outside supported bounds".into());
    }
    let mut ids = HashSet::new();
    let mut filenames = HashSet::new();
    let mut components = BTreeMap::new();
    for component in &manifest.components {
        validate_identifier(&component.id, "component id")?;
        if !ids.insert(component.id.as_str()) || !filenames.insert(component.file.filename.as_str())
        {
            return Err("runtime model v2 component id or filename is repeated".into());
        }
        let maximum = match component.kind.as_str() {
            "onnx-model" => MAX_MODEL_BYTES,
            "license-notice" => MAX_LICENSE_BYTES,
            "provenance-json" => MAX_PROVENANCE_BYTES,
            "numerical-vectors-json" => MAX_NUMERICAL_VECTORS_BYTES,
            other => {
                return Err(format!(
                    "unsupported runtime model v2 component kind {other}"
                ));
            }
        };
        validate_file_contract(
            &component.file,
            maximum,
            &format!("runtime model v2 component {}", component.id),
        )?;
        components.insert(component.id.as_str(), component);
    }
    let license = components
        .get(manifest.license.notice_component.as_str())
        .ok_or_else(|| "runtime model v2 license component is missing".to_string())?;
    if license.kind != "license-notice" {
        return Err("runtime model v2 license reference is not a license notice".into());
    }
    let provenance = components
        .get(manifest.provenance.component.as_str())
        .ok_or_else(|| "runtime model v2 provenance component is missing".to_string())?;
    if provenance.kind != "provenance-json" {
        return Err("runtime model v2 provenance reference is not provenance JSON".into());
    }
    if manifest.precision_profiles.is_empty() || manifest.precision_profiles.len() > MAX_PROFILES {
        return Err("runtime model v2 precision profile count is outside supported bounds".into());
    }
    let mut profile_ids = HashSet::new();
    let mut model_components = HashSet::new();
    let mut vector_components = HashSet::new();
    let mut referenced_components = HashSet::from([
        manifest.license.notice_component.as_str(),
        manifest.provenance.component.as_str(),
    ]);
    for profile in &manifest.precision_profiles {
        validate_identifier(&profile.id, "precision profile id")?;
        if !profile_ids.insert(profile.id.as_str()) {
            return Err("runtime model v2 precision profile id is repeated".into());
        }
        if !matches!(
            profile.element_type.as_str(),
            "float32" | "float16" | "int8"
        ) {
            return Err("runtime model v2 precision profile element type is unsupported".into());
        }
        let model = components
            .get(profile.model_component.as_str())
            .ok_or_else(|| "runtime model v2 precision model component is missing".to_string())?;
        if model.kind != "onnx-model" || !model_components.insert(profile.model_component.as_str())
        {
            return Err(
                "runtime model v2 precision profile model reference is invalid or repeated".into(),
            );
        }
        let vectors = components
            .get(profile.numerical_vectors_component.as_str())
            .ok_or_else(|| "runtime model v2 numerical vectors component is missing".to_string())?;
        if vectors.kind != "numerical-vectors-json" {
            return Err("runtime model v2 numerical vector reference has the wrong kind".into());
        }
        if !vector_components.insert(profile.numerical_vectors_component.as_str()) {
            return Err(
                "runtime model v2 precision profiles must use distinct numerical vector components"
                    .into(),
            );
        }
        referenced_components.insert(profile.model_component.as_str());
        referenced_components.insert(profile.numerical_vectors_component.as_str());
        validate_resource_contract(
            &profile.resources,
            model.file.size_bytes,
            profile.id == manifest.default_precision_profile,
        )?;
    }
    let default = manifest
        .precision_profiles
        .iter()
        .find(|profile| profile.id == manifest.default_precision_profile)
        .ok_or_else(|| "runtime model v2 default precision profile is missing".to_string())?;
    if default.element_type != "float32"
        || !default
            .resources
            .accelerators
            .iter()
            .any(|value| value == "cpu")
    {
        return Err("runtime model v2 default profile must be float32 and CPU compatible".into());
    }
    if referenced_components.len() != manifest.components.len()
        || manifest
            .components
            .iter()
            .any(|component| !referenced_components.contains(component.id.as_str()))
    {
        return Err("runtime model v2 package contains an unreferenced component".into());
    }
    Ok(())
}

fn validate_provenance(provenance: &RuntimeModelProvenanceContractV2) -> Result<(), String> {
    for (value, label) in [
        (&provenance.source_revision, "source revision"),
        (&provenance.conversion_tool, "conversion tool"),
        (&provenance.conversion_revision, "conversion revision"),
    ] {
        validate_bounded_text(value, label, 512)?;
    }
    validate_public_provenance_uri(&provenance.source_repository, "source repository")?;
    validate_public_provenance_uri(&provenance.checkpoint_source, "checkpoint source")?;
    if !valid_sha256(&provenance.source_sha256) || !valid_sha256(&provenance.checkpoint_sha256) {
        return Err("runtime model v2 provenance has an invalid SHA-256".into());
    }
    validate_spdx(&provenance.source_license_spdx)?;
    validate_spdx(&provenance.checkpoint_license_spdx)?;
    if provenance.training_datasets.is_empty() || provenance.training_datasets.len() > MAX_DATASETS
    {
        return Err("runtime model v2 training dataset provenance count is invalid".into());
    }
    let mut ids = HashSet::new();
    for dataset in &provenance.training_datasets {
        validate_identifier(&dataset.id, "training dataset id")?;
        if !ids.insert(dataset.id.as_str()) {
            return Err("runtime model v2 training dataset id is repeated".into());
        }
        validate_public_provenance_uri(&dataset.source, "training dataset source")?;
        validate_bounded_text(&dataset.revision, "training dataset revision", 256)?;
        if dataset
            .sha256
            .as_deref()
            .is_some_and(|digest| !valid_sha256(digest))
        {
            return Err("runtime model v2 training dataset has an invalid SHA-256".into());
        }
        validate_spdx(&dataset.license_spdx)?;
    }
    Ok(())
}

fn validate_public_provenance_uri(value: &str, label: &str) -> Result<(), String> {
    validate_bounded_text(value, label, 512)?;
    let uri = url::Url::parse(value)
        .map_err(|_| format!("runtime model v2 {label} must be an absolute public URI"))?;
    let allowed_scheme = match uri.scheme() {
        "https" => uri.host_str().is_some(),
        "urn" => !uri.path().is_empty(),
        _ => false,
    };
    if !allowed_scheme
        || !uri.username().is_empty()
        || uri.password().is_some()
        || uri.query().is_some()
        || uri.fragment().is_some()
    {
        return Err(format!(
            "runtime model v2 {label} must be a credential-free HTTPS URI or URN without query or fragment"
        ));
    }
    Ok(())
}

fn validate_bounded_text(value: &str, label: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(format!("runtime model v2 {label} is invalid"));
    }
    Ok(())
}

fn validate_numerical_vectors(
    vectors: &RuntimeModelNumericalVectorsV1,
    manifest: &RuntimeModelPackageManifestV2,
    profile: &RuntimeModelPrecisionProfileContractV2,
) -> Result<(), String> {
    if vectors.schema != "denoize-runtime-model-numerical-vectors-v1"
        || vectors.profile_id != profile.id
        || vectors.cases.is_empty()
        || vectors.cases.len() > MAX_VECTOR_CASES
    {
        return Err("runtime model numerical vector header is invalid".into());
    }
    let input_contracts: HashMap<&str, &RuntimeModelTensorContractV2> = manifest
        .tensors
        .inputs
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let output_contracts: HashMap<&str, &RuntimeModelTensorContractV2> = manifest
        .tensors
        .outputs
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let mut ids = HashSet::new();
    let mut aggregate_elements = 0_usize;
    for case in &vectors.cases {
        validate_identifier(&case.id, "numerical vector case id")?;
        if !ids.insert(case.id.as_str())
            || !case.tolerance.absolute.is_finite()
            || !case.tolerance.relative.is_finite()
            || case.tolerance.absolute < 0.0
            || case.tolerance.relative < 0.0
            || case.tolerance.absolute > MAX_VECTOR_TOLERANCE
            || case.tolerance.relative > MAX_VECTOR_TOLERANCE
        {
            return Err("runtime model numerical vector case metadata is invalid".into());
        }
        if case.inputs.len() > input_contracts.len() || case.outputs.len() != output_contracts.len()
        {
            return Err("runtime model numerical vector tensor count is invalid".into());
        }
        validate_vector_tensors(
            &case.inputs,
            &input_contracts,
            &mut aggregate_elements,
            "input",
        )?;
        validate_vector_tensors(
            &case.outputs,
            &output_contracts,
            &mut aggregate_elements,
            "output",
        )?;
        let supplied: HashSet<&str> = case
            .inputs
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        if manifest
            .tensors
            .inputs
            .iter()
            .any(|tensor| !supplied.contains(tensor.name.as_str()))
        {
            return Err("runtime model numerical vector omits a graph input".into());
        }
    }
    Ok(())
}

fn validate_vector_tensors(
    tensors: &[RuntimeModelNumericalTensorV1],
    contracts: &HashMap<&str, &RuntimeModelTensorContractV2>,
    aggregate_elements: &mut usize,
    kind: &str,
) -> Result<(), String> {
    let mut names = HashSet::new();
    for tensor in tensors {
        if !names.insert(tensor.name.as_str()) {
            return Err(format!(
                "runtime model numerical vector repeats {kind} tensor"
            ));
        }
        let contract = contracts
            .get(tensor.name.as_str())
            .ok_or_else(|| format!("runtime model numerical vector has unknown {kind} tensor"))?;
        if tensor.shape.len() != contract.axes.len()
            || tensor.element_type != contract.element_type
            || !matches!(tensor.element_type.as_str(), "float32" | "int64")
        {
            return Err(format!(
                "runtime model numerical vector {kind} type/rank is unsupported"
            ));
        }
        let mut elements = 1_usize;
        for (dimension, axis) in tensor.shape.iter().zip(&contract.axes) {
            if *dimension == 0
                || *dimension > MAX_FIXED_TENSOR_SAMPLES
                || axis.fixed.is_some_and(|fixed| fixed != *dimension)
            {
                return Err(format!(
                    "runtime model numerical vector {kind} shape is invalid"
                ));
            }
            elements = elements
                .checked_mul(usize::try_from(*dimension).map_err(|_| {
                    format!("runtime model numerical vector {kind} shape is too large")
                })?)
                .ok_or_else(|| format!("runtime model numerical vector {kind} size overflow"))?;
        }
        if elements > MAX_VECTOR_TENSOR_ELEMENTS
            || elements != tensor.values.len()
            || tensor.values.iter().any(|value| !value.is_finite())
            || (tensor.element_type == "float32"
                && tensor
                    .values
                    .iter()
                    .any(|value| !(*value as f32).is_finite()))
            || (tensor.element_type == "int64"
                && tensor
                    .values
                    .iter()
                    .any(|value| value.fract() != 0.0 || value.abs() > ((1_u64 << 53) as f64)))
        {
            return Err(format!(
                "runtime model numerical vector {kind} values are invalid"
            ));
        }
        *aggregate_elements = aggregate_elements
            .checked_add(elements)
            .ok_or_else(|| "runtime model numerical vector aggregate size overflow".to_string())?;
        if *aggregate_elements > MAX_VECTOR_TOTAL_ELEMENTS {
            return Err(
                "runtime model numerical vectors exceed the aggregate element limit".into(),
            );
        }
    }
    Ok(())
}

fn compatibility_manifest(
    manifest: &RuntimeModelPackageManifestV2,
) -> Result<RuntimeModelPackageManifest, String> {
    let default = selected_profile(manifest, AcceleratorRuntime::Cpu)?;
    let model = manifest
        .components
        .iter()
        .find(|component| component.id == default.model_component)
        .ok_or_else(|| "runtime model v2 default model component is missing".to_string())?;
    let license = manifest
        .components
        .iter()
        .find(|component| component.id == manifest.license.notice_component)
        .ok_or_else(|| "runtime model v2 license component is missing".to_string())?;
    let input = manifest
        .tensors
        .inputs
        .iter()
        .find(|tensor| tensor.role == "audio")
        .ok_or_else(|| "runtime model v2 audio input is missing".to_string())?;
    let output = manifest
        .tensors
        .outputs
        .iter()
        .find(|tensor| tensor.role == "audio")
        .ok_or_else(|| "runtime model v2 audio output is missing".to_string())?;
    let layout = legacy_layout(input).unwrap_or("named-audio-graph-v2");
    let fixed_input_samples = input
        .axes
        .iter()
        .find(|axis| axis.kind == "sample")
        .and_then(|axis| axis.fixed);
    let fixed_output_samples = output
        .axes
        .iter()
        .find(|axis| axis.kind == "sample")
        .and_then(|axis| axis.fixed);
    let mut resources = default.resources.clone();
    let mut accelerators = Vec::new();
    for profile in &manifest.precision_profiles {
        resources.max_session_memory_bytes = resources
            .max_session_memory_bytes
            .max(profile.resources.max_session_memory_bytes);
        resources.max_worker_memory_bytes = resources
            .max_worker_memory_bytes
            .max(profile.resources.max_worker_memory_bytes);
        resources.max_gpu_session_memory_bytes = resources
            .max_gpu_session_memory_bytes
            .max(profile.resources.max_gpu_session_memory_bytes);
        resources.max_gpu_worker_memory_bytes = resources
            .max_gpu_worker_memory_bytes
            .max(profile.resources.max_gpu_worker_memory_bytes);
        for accelerator in &profile.resources.accelerators {
            if !accelerators.contains(accelerator) {
                accelerators.push(accelerator.clone());
            }
        }
    }
    resources.accelerators = accelerators;
    Ok(RuntimeModelPackageManifest {
        schema: manifest.schema.clone(),
        format_version: manifest.format_version,
        package_id: manifest.package_id.clone(),
        package_revision: manifest.package_revision.clone(),
        signing_key_id: manifest.signing_key_id.clone(),
        runtime: RuntimeModelRuntimeContract {
            kind: manifest.runtime.kind.clone(),
            sample_rate_hz: manifest.runtime.sample_rate_hz,
        },
        frontend: RuntimeModelFrontendContract {
            channel_mapping: manifest.frontend.channels.policy.clone(),
            normalization: manifest.frontend.normalization.clone(),
            resampling: manifest.frontend.resampling.clone(),
            duration: manifest.frontend.duration.clone(),
        },
        tensor: RuntimeModelTensorContract {
            element_type: default.element_type.clone(),
            layout: layout.into(),
            fixed_input_samples,
            fixed_output_samples,
        },
        resources,
        model: model.file.clone(),
        license: RuntimeModelLicenseContract {
            spdx: manifest.license.spdx.clone(),
            file: license.file.clone(),
        },
    })
}

fn legacy_layout(tensor: &RuntimeModelTensorContractV2) -> Option<&'static str> {
    let kinds: Vec<&str> = tensor.axes.iter().map(|axis| axis.kind.as_str()).collect();
    match kinds.as_slice() {
        ["batch", "sample"] => Some("batch-samples"),
        ["batch", "channel", "sample"] => Some("batch-channels-samples"),
        _ => None,
    }
}

fn package_info_v2(
    manifest: &RuntimeModelPackageManifestV2,
    compatibility: &RuntimeModelPackageManifest,
    package_sha256: String,
    size_bytes: u64,
    numerical_vector_cases: u32,
) -> RuntimeModelPackageInfo {
    RuntimeModelPackageInfo {
        format_version: manifest.format_version,
        package_sha256,
        size_bytes,
        package_id: manifest.package_id.clone(),
        package_revision: manifest.package_revision.clone(),
        signing_key_id: manifest.signing_key_id.clone(),
        sample_rate_hz: manifest.runtime.sample_rate_hz,
        tensor_layout: compatibility.tensor.layout.clone(),
        fixed_input_samples: compatibility.tensor.fixed_input_samples,
        fixed_output_samples: compatibility.tensor.fixed_output_samples,
        model_filename: compatibility.model.filename.clone(),
        model_sha256: compatibility.model.sha256.clone(),
        model_size_bytes: compatibility.model.size_bytes,
        license_filename: compatibility.license.file.filename.clone(),
        license_sha256: compatibility.license.file.sha256.clone(),
        license_size_bytes: compatibility.license.file.size_bytes,
        license_spdx: compatibility.license.spdx.clone(),
        max_session_memory_bytes: compatibility.resources.max_session_memory_bytes,
        max_worker_memory_bytes: compatibility.resources.max_worker_memory_bytes,
        max_gpu_session_memory_bytes: compatibility.resources.max_gpu_session_memory_bytes,
        max_gpu_worker_memory_bytes: compatibility.resources.max_gpu_worker_memory_bytes,
        accelerators: compatibility.resources.accelerators.clone(),
        v2: Some(RuntimeModelPackageV2Info {
            runtime_kind: manifest.runtime.kind.clone(),
            runtime_mode: manifest.runtime.mode.clone(),
            channel_policy: manifest.frontend.channels.policy.clone(),
            channel_roles: manifest.frontend.channels.roles.clone(),
            geometry: manifest.frontend.channels.geometry.clone(),
            inputs: manifest.tensors.inputs.clone(),
            outputs: manifest.tensors.outputs.clone(),
            state_pairs: manifest.state_pairs.clone(),
            latency: manifest.latency.clone(),
            default_precision_profile: manifest.default_precision_profile.clone(),
            precision_profiles: manifest.precision_profiles.clone(),
            provenance: manifest.provenance.clone(),
            component_count: manifest.components.len() as u32,
            numerical_vector_cases,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_VECTORS: &[u8] = br#"{"schema":"denoize-runtime-model-numerical-vectors-v1","profile_id":"fp32","cases":[{"id":"identity","inputs":[{"name":"input","element_type":"float32","shape":[1,4],"values":[0.0,0.25,-0.25,0.5]}],"outputs":[{"name":"output","element_type":"float32","shape":[1,4],"values":[0.0,0.25,-0.25,0.5]}],"tolerance":{"absolute":0.000001,"relative":0.000001}}]}"#;

    fn file(filename: &str, bytes: &[u8]) -> RuntimeModelFileContract {
        RuntimeModelFileContract {
            filename: filename.into(),
            size_bytes: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    }

    fn tensor(name: &str, role: &str) -> RuntimeModelTensorContractV2 {
        RuntimeModelTensorContractV2 {
            name: name.into(),
            role: role.into(),
            element_type: "float32".into(),
            axes: vec![
                RuntimeModelAxisContractV2 {
                    name: "batch".into(),
                    kind: "batch".into(),
                    fixed: Some(1),
                },
                RuntimeModelAxisContractV2 {
                    name: "samples".into(),
                    kind: "sample".into(),
                    fixed: Some(4),
                },
            ],
            optional: false,
            state_id: None,
        }
    }

    fn manifest() -> RuntimeModelPackageManifestV2 {
        let model = b"model";
        let license = b"license";
        let provenance = b"{}";
        let resources = RuntimeModelResourceContract {
            max_session_memory_bytes: crate::estimate_model_session_bytes(model.len() as u64)
                .unwrap(),
            max_worker_memory_bytes: 4096,
            max_gpu_session_memory_bytes: 0,
            max_gpu_worker_memory_bytes: 0,
            accelerators: vec!["cpu".into()],
        };
        RuntimeModelPackageManifestV2 {
            schema: RUNTIME_MODEL_PACKAGE_SCHEMA_V2.into(),
            format_version: RUNTIME_MODEL_PACKAGE_VERSION_V2,
            package_id: "example.identity-v2".into(),
            package_revision: "1".into(),
            signing_key_id: "0000000000000001".into(),
            runtime: RuntimeModelRuntimeContractV2 {
                kind: RUNTIME_KIND_V2.into(),
                sample_rate_hz: 16_000,
                mode: "finite-and-streaming".into(),
            },
            frontend: RuntimeModelFrontendContractV2 {
                normalization: NORMALIZATION_V2.into(),
                resampling: RESAMPLING_V2.into(),
                duration: DURATION_V2.into(),
                channels: RuntimeModelChannelContractV2 {
                    policy: "independent-mono".into(),
                    roles: vec![],
                    geometry: None,
                },
            },
            tensors: RuntimeModelTensorSetContractV2 {
                inputs: vec![tensor("input", "audio")],
                outputs: vec![tensor("output", "audio")],
            },
            state_pairs: vec![],
            latency: RuntimeModelLatencyContractV2 {
                frame_samples: 4,
                hop_samples: 4,
                left_context_samples: 0,
                right_context_samples: 0,
                lookahead_samples: 0,
                algorithmic_latency_samples: 0,
                flush_samples: 0,
            },
            components: vec![
                RuntimeModelComponentContractV2 {
                    id: "model-fp32".into(),
                    kind: "onnx-model".into(),
                    file: file("model.onnx", model),
                },
                RuntimeModelComponentContractV2 {
                    id: "license".into(),
                    kind: "license-notice".into(),
                    file: file("LICENSE.txt", license),
                },
                RuntimeModelComponentContractV2 {
                    id: "provenance".into(),
                    kind: "provenance-json".into(),
                    file: file("provenance.json", provenance),
                },
                RuntimeModelComponentContractV2 {
                    id: "vectors-fp32".into(),
                    kind: "numerical-vectors-json".into(),
                    file: file("vectors-fp32.json", TEST_VECTORS),
                },
            ],
            precision_profiles: vec![RuntimeModelPrecisionProfileContractV2 {
                id: "fp32".into(),
                element_type: "float32".into(),
                model_component: "model-fp32".into(),
                numerical_vectors_component: "vectors-fp32".into(),
                resources,
            }],
            default_precision_profile: "fp32".into(),
            license: RuntimeModelLicenseContractV2 {
                spdx: "MIT".into(),
                notice_component: "license".into(),
            },
            provenance: RuntimeModelProvenanceContractV2 {
                component: "provenance".into(),
                source_repository: "https://example.invalid/source".into(),
                source_revision: "0123456789abcdef".into(),
                source_sha256: "0".repeat(64),
                source_license_spdx: "MIT".into(),
                checkpoint_source: "https://example.invalid/checkpoint".into(),
                checkpoint_sha256: "1".repeat(64),
                checkpoint_license_spdx: "MIT".into(),
                conversion_tool: "example-converter".into(),
                conversion_revision: "1".into(),
                training_datasets: vec![RuntimeModelTrainingDatasetContractV2 {
                    id: "synthetic".into(),
                    source: "urn:denoize:test:synthetic".into(),
                    revision: "1".into(),
                    sha256: Some("2".repeat(64)),
                    license_spdx: "CC0-1.0".into(),
                }],
            },
        }
    }

    fn test_key() -> ParsedPublicKey {
        parse_public_key("RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3").unwrap()
    }

    fn component_bytes() -> Vec<Vec<u8>> {
        vec![
            b"model".to_vec(),
            b"license".to_vec(),
            b"{}".to_vec(),
            TEST_VECTORS.to_vec(),
        ]
    }

    fn write_test_package(
        path: &Path,
        manifest: &RuntimeModelPackageManifestV2,
        components: &[Vec<u8>],
    ) {
        let manifest = serde_json::to_vec(manifest).unwrap();
        let signature = b"test signature bytes";
        let mut file = File::create(path).unwrap();
        file.write_all(PACKAGE_MAGIC_V2).unwrap();
        file.write_all(&(manifest.len() as u64).to_be_bytes())
            .unwrap();
        file.write_all(&(signature.len() as u64).to_be_bytes())
            .unwrap();
        file.write_all(&(components.len() as u64).to_be_bytes())
            .unwrap();
        for component in components {
            file.write_all(&(component.len() as u64).to_be_bytes())
                .unwrap();
        }
        file.write_all(&manifest).unwrap();
        file.write_all(signature).unwrap();
        for component in components {
            file.write_all(component).unwrap();
        }
    }

    #[test]
    fn manifest_and_vectors_are_closed_and_bounded() {
        let manifest = manifest();
        validate_manifest_v2(&manifest, "0000000000000001").unwrap();
        let vector_component = manifest
            .components
            .iter()
            .find(|component| component.id == "vectors-fp32")
            .unwrap();
        assert_eq!(vector_component.file.size_bytes, TEST_VECTORS.len() as u64);
        parse_numerical_vectors(TEST_VECTORS, &manifest, &manifest.precision_profiles[0]).unwrap();

        // Optional runtime inputs still need concrete values in every signed
        // conformance case so the selected graph can actually be executed.
        let mut optional_manifest = manifest.clone();
        let mut optional = tensor("conditioning", "control");
        optional.optional = true;
        optional_manifest.tensors.inputs.push(optional);
        assert!(parse_numerical_vectors(
            TEST_VECTORS,
            &optional_manifest,
            &optional_manifest.precision_profiles[0]
        )
        .unwrap_err()
        .contains("omits a graph input"));

        let mut loose_vectors: RuntimeModelNumericalVectorsV1 =
            serde_json::from_slice(TEST_VECTORS).unwrap();
        loose_vectors.cases[0].tolerance.absolute = 0.010_001;
        assert!(validate_numerical_vectors(
            &loose_vectors,
            &manifest,
            &manifest.precision_profiles[0]
        )
        .unwrap_err()
        .contains("metadata is invalid"));

        let mut secret_source = manifest.clone();
        secret_source.provenance.checkpoint_source =
            "https://user:secret@example.invalid/checkpoint?token=secret".into();
        assert!(validate_manifest_v2(&secret_source, "0000000000000001")
            .unwrap_err()
            .contains("credential-free HTTPS URI"));

        let mut unsafe_manifest = manifest.clone();
        unsafe_manifest
            .components
            .push(RuntimeModelComponentContractV2 {
                id: "script".into(),
                kind: "script".into(),
                file: file("run.sh", b"exit 0"),
            });
        assert!(validate_manifest_v2(&unsafe_manifest, "0000000000000001")
            .unwrap_err()
            .contains("unsupported"));
    }

    #[test]
    fn recurrent_states_must_pair_by_name_shape_and_id() {
        let mut manifest = manifest();
        let mut state_input = tensor("state_in", "state");
        state_input.state_id = Some("memory".into());
        state_input.axes[1].kind = "state".into();
        let mut state_output = state_input.clone();
        state_output.name = "state_out".into();
        state_output.optional = false;
        manifest.tensors.inputs.push(state_input);
        manifest.tensors.outputs.push(state_output);
        manifest.state_pairs.push(RuntimeModelStatePairContractV2 {
            id: "memory".into(),
            input: "state_in".into(),
            output: "state_out".into(),
            initialization: "zeros".into(),
        });
        validate_manifest_v2(&manifest, "0000000000000001").unwrap();
        manifest.tensors.outputs[1].axes[1].fixed = Some(5);
        assert!(validate_manifest_v2(&manifest, "0000000000000001")
            .unwrap_err()
            .contains("do not match"));
    }

    #[test]
    fn semantic_query_and_residual_roles_are_explicit_and_closed() {
        let mut manifest = manifest();
        let mut query = tensor("query", "query");
        query.axes[1].name = "classes".into();
        query.axes[1].kind = "feature".into();
        query.axes[1].fixed = Some(2);
        manifest.tensors.inputs.push(query);
        manifest
            .tensors
            .outputs
            .push(tensor("residual", "residual"));
        validate_manifest_v2(&manifest, "0000000000000001").unwrap();

        let mut overloaded = manifest.clone();
        overloaded.tensors.inputs[1].role = "natural-language".into();
        assert!(validate_manifest_v2(&overloaded, "0000000000000001")
            .unwrap_err()
            .contains("unsupported runtime model v2 input role"));
        let mut overloaded = manifest;
        overloaded.tensors.outputs[1].role = "target-or-residual".into();
        assert!(validate_manifest_v2(&overloaded, "0000000000000001")
            .unwrap_err()
            .contains("unsupported runtime model v2 output role"));
    }

    #[test]
    fn channel_roles_and_geometry_must_cover_the_declared_audio_channels() {
        let mut program = manifest();
        for tensor in program
            .tensors
            .inputs
            .iter_mut()
            .chain(program.tensors.outputs.iter_mut())
        {
            tensor.axes.insert(
                1,
                RuntimeModelAxisContractV2 {
                    name: "channels".into(),
                    kind: "channel".into(),
                    fixed: Some(2),
                },
            );
        }
        program.frontend.channels.policy = "program-multichannel".into();
        program.frontend.channels.roles = vec![
            RuntimeModelChannelRoleContractV2 {
                channel_index: 0,
                role: "program-left".into(),
            },
            RuntimeModelChannelRoleContractV2 {
                channel_index: 1,
                role: "program-right".into(),
            },
        ];
        validate_manifest_v2(&program, "0000000000000001").unwrap();
        program.frontend.channels.roles.pop();
        assert!(validate_manifest_v2(&program, "0000000000000001")
            .unwrap_err()
            .contains("cover every fixed program channel"));

        let mut array = program;
        array.frontend.channels.policy = "microphone-array".into();
        array.frontend.channels.roles = vec![
            RuntimeModelChannelRoleContractV2 {
                channel_index: 0,
                role: "microphone".into(),
            },
            RuntimeModelChannelRoleContractV2 {
                channel_index: 1,
                role: "microphone".into(),
            },
        ];
        array.frontend.channels.geometry = Some(RuntimeModelGeometryContractV2 {
            coordinate_system: "right-handed-cartesian".into(),
            units: "millimeters".into(),
            microphones: vec![
                RuntimeModelMicrophonePositionV2 {
                    channel_index: 0,
                    x_mm: -35,
                    y_mm: 0,
                    z_mm: 0,
                },
                RuntimeModelMicrophonePositionV2 {
                    channel_index: 1,
                    x_mm: 35,
                    y_mm: 0,
                    z_mm: 0,
                },
            ],
        });
        validate_manifest_v2(&array, "0000000000000001").unwrap();
        array
            .frontend
            .channels
            .geometry
            .as_mut()
            .unwrap()
            .microphones[1]
            .channel_index = 2;
        assert!(validate_manifest_v2(&array, "0000000000000001").is_err());
    }

    #[test]
    fn framing_authenticates_every_component_and_rejects_trailing_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model-v2.dmp");
        let key = test_key();
        let mut contract = manifest();
        contract.signing_key_id = key.key_id.clone();
        let components = component_bytes();
        write_test_package(&path, &contract, &components);

        let (mut package, size) =
            crate::input::open_regular_file(&path, "test v2 package").unwrap();
        let prepared =
            prepare_open_package_v2(&mut package, &path, size, &key, |_, _, _| Ok(())).unwrap();
        assert_eq!(prepared.info.format_version, 2);
        assert_eq!(prepared.info.package_id, "example.identity-v2");
        assert_eq!(prepared.opened.components.len(), components.len());
        assert_eq!(prepared.info.v2.unwrap().numerical_vector_cases, 1);

        let mut package = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        package.write_all(b"trailing").unwrap();
        drop(package);
        let (mut package, size) =
            crate::input::open_regular_file(&path, "test v2 package").unwrap();
        let error =
            prepare_open_package_v2(&mut package, &path, size, &key, |_, _, _| Ok(())).unwrap_err();
        assert!(error.contains("length mismatch"), "{error}");
    }

    #[test]
    fn framing_rejects_oversized_lengths_before_signature_verification() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized-v2.dmp");
        let mut package = File::create(&path).unwrap();
        package.write_all(PACKAGE_MAGIC_V2).unwrap();
        package
            .write_all(&(MAX_MANIFEST_BYTES + 1).to_be_bytes())
            .unwrap();
        package.write_all(&1_u64.to_be_bytes()).unwrap();
        package.write_all(&1_u64.to_be_bytes()).unwrap();
        package.write_all(&1_u64.to_be_bytes()).unwrap();
        drop(package);

        let key = test_key();
        let (mut package, size) =
            crate::input::open_regular_file(&path, "test v2 package").unwrap();
        let error = prepare_open_package_v2(&mut package, &path, size, &key, |_, _, _| {
            panic!("oversized framing must fail before signature verification")
        })
        .unwrap_err();
        assert!(error.contains("manifest length"), "{error}");
    }

    #[test]
    fn component_tampering_is_rejected_after_manifest_verification() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tampered-v2.dmp");
        let key = test_key();
        let mut contract = manifest();
        contract.signing_key_id = key.key_id.clone();
        let mut components = component_bytes();
        components[0] = b"modfl".to_vec();
        write_test_package(&path, &contract, &components);

        let (mut package, size) =
            crate::input::open_regular_file(&path, "test v2 package").unwrap();
        let error =
            prepare_open_package_v2(&mut package, &path, size, &key, |_, _, _| Ok(())).unwrap_err();
        assert!(error.contains("component model-fp32 SHA-256"), "{error}");
    }

    #[test]
    fn authenticated_provenance_component_must_be_a_json_object() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("invalid-provenance-v2.dmp");
        let key = test_key();
        let mut contract = manifest();
        contract.signing_key_id = key.key_id.clone();
        let mut components = component_bytes();
        components[2] = b"not-json".to_vec();
        contract.components[2].file = file("provenance.json", &components[2]);
        write_test_package(&path, &contract, &components);

        let (mut package, size) =
            crate::input::open_regular_file(&path, "test v2 package").unwrap();
        let error =
            prepare_open_package_v2(&mut package, &path, size, &key, |_, _, _| Ok(())).unwrap_err();
        assert!(
            error.contains("invalid runtime model provenance JSON"),
            "{error}"
        );
    }

    #[test]
    fn non_default_gpu_only_profile_keeps_the_default_cpu_fallback() {
        let mut contract = manifest();
        let model = b"cuda model";
        let vectors = TEST_VECTORS
            .windows(b"\"fp32\"".len())
            .position(|window| window == b"\"fp32\"")
            .map(|offset| {
                let mut bytes = TEST_VECTORS.to_vec();
                bytes.splice(
                    offset..offset + b"\"fp32\"".len(),
                    b"\"fp16\"".iter().copied(),
                );
                bytes
            })
            .unwrap();
        contract.components.push(RuntimeModelComponentContractV2 {
            id: "model-fp16".into(),
            kind: "onnx-model".into(),
            file: file("model-fp16.onnx", model),
        });
        contract.components.push(RuntimeModelComponentContractV2 {
            id: "vectors-fp16".into(),
            kind: "numerical-vectors-json".into(),
            file: file("vectors-fp16.json", &vectors),
        });
        contract
            .precision_profiles
            .push(RuntimeModelPrecisionProfileContractV2 {
                id: "fp16".into(),
                element_type: "float16".into(),
                model_component: "model-fp16".into(),
                numerical_vectors_component: "vectors-fp16".into(),
                resources: RuntimeModelResourceContract {
                    max_session_memory_bytes: crate::estimate_model_session_bytes(
                        model.len() as u64
                    )
                    .unwrap(),
                    max_worker_memory_bytes: 4096,
                    max_gpu_session_memory_bytes: crate::estimate_gpu_session_bytes(
                        model.len() as u64
                    )
                    .unwrap(),
                    max_gpu_worker_memory_bytes: 4096,
                    accelerators: vec!["cuda".into()],
                },
            });

        validate_manifest_v2(&contract, "0000000000000001").unwrap();
        assert_eq!(
            selected_profile(&contract, AcceleratorRuntime::Cpu)
                .unwrap()
                .id,
            "fp32"
        );
        assert_eq!(
            selected_profile(&contract, AcceleratorRuntime::Cuda)
                .unwrap()
                .id,
            "fp16"
        );
        parse_numerical_vectors(&vectors, &contract, &contract.precision_profiles[1]).unwrap();
    }

    #[test]
    fn signed_builder_round_trips_deterministically_and_never_clobbers() {
        let directory = tempfile::tempdir().unwrap();
        let components_directory = directory.path().join("components");
        std::fs::create_dir(&components_directory).unwrap();
        let components = component_bytes();
        for (contract, bytes) in manifest().components.iter().zip(&components) {
            std::fs::write(components_directory.join(&contract.file.filename), bytes).unwrap();
        }

        let minisign::KeyPair { pk, sk } =
            minisign::KeyPair::generate_unencrypted_keypair().unwrap();
        let public_key_text = pk.to_box().unwrap().into_string();
        let parsed_key = parse_public_key(&public_key_text).unwrap();
        let mut contract = manifest();
        contract.signing_key_id = parsed_key.key_id;
        let manifest_bytes = serde_json::to_vec(&contract).unwrap();
        let signature =
            minisign::sign(None, &sk, std::io::Cursor::new(&manifest_bytes), None, None)
                .unwrap()
                .into_string();
        let manifest_path = directory.path().join("manifest.json");
        let signature_path = directory.path().join("manifest.json.sig");
        let public_key_path = directory.path().join("minisign.pub");
        std::fs::write(&manifest_path, manifest_bytes).unwrap();
        std::fs::write(&signature_path, signature).unwrap();
        std::fs::write(&public_key_path, public_key_text).unwrap();

        let first = directory.path().join("first.dmp");
        let second = directory.path().join("second.dmp");
        let info = build_runtime_model_package_v2(
            &first,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &components_directory,
        )
        .unwrap();
        build_runtime_model_package_v2(
            &second,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &components_directory,
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&first).unwrap(),
            std::fs::read(&second).unwrap()
        );
        let original = std::fs::read(&first).unwrap();
        let error = build_runtime_model_package_v2(
            &first,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &components_directory,
        )
        .unwrap_err();
        assert!(error.contains("exists"), "{error}");
        assert_eq!(std::fs::read(&first).unwrap(), original);

        let opened = RuntimeModelPackage::open(&first, &public_key_path).unwrap();
        assert_eq!(opened.info(), info);
        assert_eq!(opened.manifest_v2().unwrap(), &contract);
        assert_eq!(opened.info().v2.unwrap().numerical_vector_cases, 1);
    }
}
