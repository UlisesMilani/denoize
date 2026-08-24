//! Deterministic, length-delimited offline bundles for portable projects.

use super::{
    canonical_project_root, inspect_project_source, normalized_project_output,
    read_bounded_regular, reject_project_output_collision, resolve_project_locator,
    validate_identifier, validate_locator, validate_project_files, ProjectExecutionPlan,
    ProjectManifest, ProjectModelReference, ProjectValidationReport, SignedProjectExecutionReceipt,
    PROJECT_MANIFEST_SCHEMA_VERSION, PROJECT_VALIDATION_SCHEMA,
};
use crate::batch_resume::{self, Digest, FileFingerprint};
use crate::{
    AtomicOutput, CommitMode, DecodeLimits, ExecutionPlan, RuntimeModelPackage,
    SignedExecutionReceipt,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const PROJECT_BUNDLE_SCHEMA: &str = "denoize-project-bundle-v1";
pub const PROJECT_BUNDLE_IMPORT_SCHEMA: &str = "denoize-project-bundle-import-v1";

const PROJECT_BUNDLE_HEADER_SCHEMA: &str = "denoize-project-bundle-header-v1";
const PROJECT_BUNDLE_MAGIC: &[u8] = b"denoize-project-bundle-v1\n";
const PROJECT_BUNDLE_VERSION: u32 = 1;
const IMPORTED_MANIFEST_LOCATOR: &str = "project.denoize.json";
const IMPORTED_VERIFICATION_LOCATOR: &str = ".denoize/project-verification-v1.json";
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_VERIFICATION_BYTES: u64 = 16 * 1024 * 1024;
const MAX_BUNDLE_FILES: usize = 32_768;
const MAX_DOCUMENT_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DOCUMENT_TOTAL_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_OPTIONAL_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_BUNDLE_BYTES: u64 = 128 * 1024 * 1024 * 1024;
const COPY_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectBundleBindingKind {
    Source,
    SourceLicense,
    Setting,
    Preset,
    ModelPackage,
    ModelPublicKey,
    Plan,
    Receipt,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectBundleBinding {
    pub kind: ProjectBundleBindingKind,
    pub id: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectBundleFileInfo {
    pub locator: String,
    pub fingerprint: FileFingerprint,
    pub bindings: Vec<ProjectBundleBinding>,
}

#[derive(Clone, Debug)]
pub struct ProjectBundleBuildOptions {
    pub include_sources: bool,
    pub source_payload_limit_bytes: u64,
    pub include_models: bool,
    pub model_payload_limit_bytes: u64,
    pub commit_mode: CommitMode,
}

impl Default for ProjectBundleBuildOptions {
    fn default() -> Self {
        Self {
            include_sources: false,
            source_payload_limit_bytes: 0,
            include_models: false,
            model_payload_limit_bytes: 0,
            commit_mode: CommitMode::NoClobber,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectBundleInfo {
    pub schema: String,
    pub schema_version: u32,
    pub project_id: String,
    pub manifest_digest: Digest,
    pub bundle: FileFingerprint,
    pub manifest: FileFingerprint,
    pub verification: FileFingerprint,
    pub source_payloads_included: bool,
    pub source_payload_limit_bytes: u64,
    pub source_payload_bytes: u64,
    pub model_payloads_included: bool,
    pub model_payload_limit_bytes: u64,
    pub model_payload_bytes: u64,
    pub document_bytes: u64,
    pub files: Vec<ProjectBundleFileInfo>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectBundleImportReport {
    pub schema: String,
    pub schema_version: u32,
    pub project_id: String,
    pub manifest_digest: Digest,
    pub bundle: FileFingerprint,
    pub destination: String,
    pub manifest_locator: String,
    pub verification_locator: String,
    pub files_imported: u64,
    pub omitted_sources: Vec<String>,
    pub omitted_models: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ProjectBundleHeader {
    schema: String,
    schema_version: u32,
    project_id: String,
    manifest_digest: Digest,
    manifest: BundleBlob,
    verification: BundleBlob,
    source_payloads_included: bool,
    source_payload_limit_bytes: u64,
    model_payloads_included: bool,
    model_payload_limit_bytes: u64,
    files: Vec<ProjectBundleFileInfo>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BundleBlob {
    name: String,
    fingerprint: FileFingerprint,
}

struct OpenBundleFile {
    info: ProjectBundleFileInfo,
    path: PathBuf,
    file: File,
}

struct PreparedProjectBundle {
    staging: tempfile::TempDir,
    info: ProjectBundleInfo,
    manifest: ProjectManifest,
}

#[derive(Default)]
struct BundleByteTotals {
    sources: u64,
    models: u64,
    documents: u64,
}

/// Build a deterministic project bundle. Source audio and model-package
/// payloads remain references unless their corresponding include flag and a
/// positive aggregate byte limit are both supplied.
pub fn build_project_bundle(
    manifest_path: impl AsRef<Path>,
    root: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: &ProjectBundleBuildOptions,
    decode_limits: DecodeLimits,
) -> Result<ProjectBundleInfo, String> {
    validate_build_options(options)?;
    let root = canonical_project_root(root.as_ref())?;
    let manifest = ProjectManifest::from_file(manifest_path.as_ref())?;
    reject_project_output_collision(&manifest, &root, output.as_ref())?;
    let manifest_identity = std::fs::canonicalize(manifest_path.as_ref()).map_err(|error| {
        format!(
            "re-resolve project bundle manifest {}: {error}",
            manifest_path.as_ref().display()
        )
    })?;
    let output_identity = normalized_project_output(output.as_ref())?;
    if output_identity == manifest_identity
        || std::fs::canonicalize(&output_identity).ok().as_ref() == Some(&manifest_identity)
    {
        return Err("project bundle output must not replace its project manifest".into());
    }
    validate_reserved_locators(&manifest)?;
    let verification = validate_project_files(&manifest, &root, decode_limits)?;
    validate_verification(&verification, &manifest)?;

    let manifest_bytes = serde_json::to_vec(&manifest)
        .map_err(|error| format!("serialize project bundle manifest: {error}"))?;
    let verification_bytes = serde_json::to_vec(&verification)
        .map_err(|error| format!("serialize project bundle verification: {error}"))?;
    let files = expected_bundle_files(&manifest, options.include_sources, options.include_models)?;
    let totals = validate_bundle_budgets(
        &files,
        options.include_sources,
        options.source_payload_limit_bytes,
        options.include_models,
        options.model_payload_limit_bytes,
    )?;

    let mut opened = Vec::new();
    opened
        .try_reserve_exact(files.len())
        .map_err(|error| format!("reserve project bundle file handles: {error}"))?;
    for info in &files {
        let path = resolve_project_locator(&root, &info.locator, "project bundle component")?;
        let (file, len) = crate::input::open_regular_file(&path, "project bundle component")?;
        if len != info.fingerprint.len {
            return Err(format!(
                "project bundle component {} length differs from the manifest",
                info.locator
            ));
        }
        let observed = batch_resume::fingerprint_open_file_at(&file, &path)?;
        if observed != info.fingerprint {
            return Err(format!(
                "project bundle component {} differs from the manifest",
                info.locator
            ));
        }
        opened.push(OpenBundleFile {
            info: info.clone(),
            path,
            file,
        });
    }

    let header = ProjectBundleHeader {
        schema: PROJECT_BUNDLE_HEADER_SCHEMA.into(),
        schema_version: PROJECT_BUNDLE_VERSION,
        project_id: manifest.project_id.clone(),
        manifest_digest: manifest.digest()?,
        manifest: BundleBlob {
            name: IMPORTED_MANIFEST_LOCATOR.into(),
            fingerprint: fingerprint_bytes(&manifest_bytes),
        },
        verification: BundleBlob {
            name: IMPORTED_VERIFICATION_LOCATOR.into(),
            fingerprint: fingerprint_bytes(&verification_bytes),
        },
        source_payloads_included: options.include_sources,
        source_payload_limit_bytes: options.source_payload_limit_bytes,
        model_payloads_included: options.include_models,
        model_payload_limit_bytes: options.model_payload_limit_bytes,
        files,
    };
    validate_header_shape(&header)?;
    let header_bytes = serde_json::to_vec(&header)
        .map_err(|error| format!("serialize project bundle header: {error}"))?;
    if header_bytes.is_empty() || header_bytes.len() as u64 > MAX_HEADER_BYTES {
        return Err("project bundle header exceeds the 16 MiB limit".into());
    }
    let expected_len = expected_bundle_len(&header, header_bytes.len() as u64)?;

    let mut transaction = AtomicOutput::new(output.as_ref())?;
    let mut digest = Sha256::new();
    write_hashed(transaction.file_mut(), &mut digest, PROJECT_BUNDLE_MAGIC)?;
    write_hashed(
        transaction.file_mut(),
        &mut digest,
        &(header_bytes.len() as u64).to_le_bytes(),
    )?;
    write_hashed(transaction.file_mut(), &mut digest, &header_bytes)?;
    write_hashed(transaction.file_mut(), &mut digest, &manifest_bytes)?;
    write_hashed(transaction.file_mut(), &mut digest, &verification_bytes)?;
    for component in &mut opened {
        copy_open_component(transaction.file_mut(), &mut digest, component)?;
    }
    let staged_len = transaction
        .file_mut()
        .stream_position()
        .map_err(|error| format!("inspect staged project bundle: {error}"))?;
    if staged_len != expected_len {
        return Err("staged project bundle length differs from its header".into());
    }
    let bundle = FileFingerprint {
        len: staged_len,
        digest: Digest::from_bytes(digest.finalize().into()),
    };
    let info = bundle_info(&header, bundle, totals);
    crate::fault_injection::hit("project-bundle.before-commit")?;
    transaction.commit(options.commit_mode)?;
    Ok(info)
}

/// Authenticate every byte and every embedded parseable contract without
/// changing project state.
pub fn inspect_project_bundle(path: impl AsRef<Path>) -> Result<ProjectBundleInfo, String> {
    Ok(prepare_project_bundle(path.as_ref(), None)?.info)
}

/// Authenticate a bundle completely, stage its portable tree in a private
/// sibling directory, and publish only when the destination does not exist.
pub fn import_project_bundle(
    path: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<ProjectBundleImportReport, String> {
    let destination = absolute_import_destination(destination.as_ref())?;
    ensure_destination_absent(&destination)?;
    let parent = destination
        .parent()
        .ok_or_else(|| "project bundle import destination has no parent".to_string())?;
    #[cfg(unix)]
    crate::atomic_output::validate_unix_staging_path(parent, &destination)?;

    let prepared = prepare_project_bundle(path.as_ref(), Some(parent))?;
    ensure_destination_absent(&destination)?;
    crate::fault_injection::hit("project-bundle-import.before-commit")?;
    rename_directory_noclobber(prepared.staging.path(), &destination)?;

    let omitted_sources = if prepared.info.source_payloads_included {
        Vec::new()
    } else {
        prepared
            .manifest
            .sources
            .iter()
            .map(|source| source.id.clone())
            .collect()
    };
    let omitted_models = if prepared.info.model_payloads_included {
        Vec::new()
    } else {
        prepared
            .manifest
            .models
            .iter()
            .map(|model| model.id.clone())
            .collect()
    };
    Ok(ProjectBundleImportReport {
        schema: PROJECT_BUNDLE_IMPORT_SCHEMA.into(),
        schema_version: PROJECT_BUNDLE_VERSION,
        project_id: prepared.info.project_id.clone(),
        manifest_digest: prepared.info.manifest_digest,
        bundle: prepared.info.bundle,
        destination: destination.to_string_lossy().into_owned(),
        manifest_locator: IMPORTED_MANIFEST_LOCATOR.into(),
        verification_locator: IMPORTED_VERIFICATION_LOCATOR.into(),
        files_imported: prepared.info.files.len() as u64 + 2,
        omitted_sources,
        omitted_models,
    })
}

fn validate_build_options(options: &ProjectBundleBuildOptions) -> Result<(), String> {
    for (label, included, limit) in [
        (
            "source",
            options.include_sources,
            options.source_payload_limit_bytes,
        ),
        (
            "model",
            options.include_models,
            options.model_payload_limit_bytes,
        ),
    ] {
        if included && !(1..=MAX_OPTIONAL_PAYLOAD_BYTES).contains(&limit) {
            return Err(format!(
                "included project {label} payloads require a byte limit in 1..={MAX_OPTIONAL_PAYLOAD_BYTES}"
            ));
        }
        if !included && limit != 0 {
            return Err(format!(
                "project {label} payload limit requires its include option"
            ));
        }
    }
    Ok(())
}

fn all_bundle_files(manifest: &ProjectManifest) -> Result<Vec<ProjectBundleFileInfo>, String> {
    let mut files = BTreeMap::<String, ProjectBundleFileInfo>::new();
    for source in &manifest.sources {
        add_bundle_binding(
            &mut files,
            &source.locator,
            source.fingerprint,
            ProjectBundleBindingKind::Source,
            &source.id,
        )?;
        if let Some(license) = &source.license {
            add_bundle_binding(
                &mut files,
                &license.locator,
                license.fingerprint,
                ProjectBundleBindingKind::SourceLicense,
                &license.id,
            )?;
        }
    }
    for (kind, references) in [
        (
            ProjectBundleBindingKind::Setting,
            manifest.settings.as_slice(),
        ),
        (
            ProjectBundleBindingKind::Preset,
            manifest.presets.as_slice(),
        ),
        (ProjectBundleBindingKind::Plan, manifest.plans.as_slice()),
        (
            ProjectBundleBindingKind::Receipt,
            manifest.receipts.as_slice(),
        ),
    ] {
        for reference in references {
            add_bundle_binding(
                &mut files,
                &reference.locator,
                reference.fingerprint,
                kind,
                &reference.id,
            )?;
        }
    }
    for model in &manifest.models {
        add_bundle_binding(
            &mut files,
            &model.package.locator,
            model.package.fingerprint,
            ProjectBundleBindingKind::ModelPackage,
            &model.id,
        )?;
        add_bundle_binding(
            &mut files,
            &model.public_key.locator,
            model.public_key.fingerprint,
            ProjectBundleBindingKind::ModelPublicKey,
            &model.id,
        )?;
    }
    if files.len() > MAX_BUNDLE_FILES {
        return Err(format!(
            "project bundle exceeds the {MAX_BUNDLE_FILES}-file limit"
        ));
    }
    let mut result = files.into_values().collect::<Vec<_>>();
    for file in &mut result {
        file.bindings.sort();
        file.bindings.dedup();
        let has_source = file
            .bindings
            .iter()
            .any(|binding| binding.kind == ProjectBundleBindingKind::Source);
        let has_model = file
            .bindings
            .iter()
            .any(|binding| binding.kind == ProjectBundleBindingKind::ModelPackage);
        if (has_source || has_model)
            && file.bindings.iter().any(|binding| {
                binding.kind
                    != if has_source {
                        ProjectBundleBindingKind::Source
                    } else {
                        ProjectBundleBindingKind::ModelPackage
                    }
            })
        {
            return Err(format!(
                "optional payload locator {} is reused by a different project artifact kind",
                file.locator
            ));
        }
    }
    Ok(result)
}

fn expected_bundle_files(
    manifest: &ProjectManifest,
    include_sources: bool,
    include_models: bool,
) -> Result<Vec<ProjectBundleFileInfo>, String> {
    let mut files = all_bundle_files(manifest)?;
    for file in &mut files {
        file.bindings.retain(|binding| match binding.kind {
            ProjectBundleBindingKind::Source => include_sources,
            ProjectBundleBindingKind::ModelPackage => include_models,
            _ => true,
        });
    }
    files.retain(|file| !file.bindings.is_empty());
    Ok(files)
}

fn add_bundle_binding(
    files: &mut BTreeMap<String, ProjectBundleFileInfo>,
    locator: &str,
    fingerprint: FileFingerprint,
    kind: ProjectBundleBindingKind,
    id: &str,
) -> Result<(), String> {
    validate_locator(locator)?;
    validate_identifier("project bundle binding ID", id)?;
    let file = files
        .entry(locator.to_string())
        .or_insert_with(|| ProjectBundleFileInfo {
            locator: locator.to_string(),
            fingerprint,
            bindings: Vec::new(),
        });
    if file.fingerprint != fingerprint {
        return Err(format!(
            "project bundle locator {locator} has conflicting fingerprints"
        ));
    }
    file.bindings.push(ProjectBundleBinding {
        kind,
        id: id.to_string(),
    });
    Ok(())
}

fn validate_reserved_locators(manifest: &ProjectManifest) -> Result<(), String> {
    for file in all_bundle_files(manifest)? {
        if file.locator == IMPORTED_MANIFEST_LOCATOR
            || file.locator == ".denoize"
            || file.locator.starts_with(".denoize/")
        {
            return Err(format!(
                "project locator {} is reserved for bundle import metadata",
                file.locator
            ));
        }
    }
    Ok(())
}

fn validate_bundle_budgets(
    files: &[ProjectBundleFileInfo],
    include_sources: bool,
    source_limit: u64,
    include_models: bool,
    model_limit: u64,
) -> Result<BundleByteTotals, String> {
    let mut totals = BundleByteTotals::default();
    for file in files {
        let source = file
            .bindings
            .iter()
            .any(|binding| binding.kind == ProjectBundleBindingKind::Source);
        let model = file
            .bindings
            .iter()
            .any(|binding| binding.kind == ProjectBundleBindingKind::ModelPackage);
        let target = if source {
            &mut totals.sources
        } else if model {
            &mut totals.models
        } else {
            if file.fingerprint.len > MAX_DOCUMENT_FILE_BYTES {
                return Err(format!(
                    "project bundle document {} exceeds the {MAX_DOCUMENT_FILE_BYTES}-byte limit",
                    file.locator
                ));
            }
            &mut totals.documents
        };
        *target = target
            .checked_add(file.fingerprint.len)
            .ok_or_else(|| "project bundle payload byte total overflows".to_string())?;
    }
    if totals.documents > MAX_DOCUMENT_TOTAL_BYTES {
        return Err(format!(
            "project bundle documents exceed the {MAX_DOCUMENT_TOTAL_BYTES}-byte aggregate limit"
        ));
    }
    if (!include_sources && totals.sources != 0) || totals.sources > source_limit {
        return Err(format!(
            "project source payloads require {0} bytes, exceeding the declared {source_limit}-byte limit",
            totals.sources
        ));
    }
    if (!include_models && totals.models != 0) || totals.models > model_limit {
        return Err(format!(
            "project model payloads require {0} bytes, exceeding the declared {model_limit}-byte limit",
            totals.models
        ));
    }
    Ok(totals)
}

fn validate_header_shape(header: &ProjectBundleHeader) -> Result<(), String> {
    if header.schema != PROJECT_BUNDLE_HEADER_SCHEMA
        || header.schema_version != PROJECT_BUNDLE_VERSION
    {
        return Err("unsupported project bundle header".into());
    }
    validate_identifier("project bundle project ID", &header.project_id)?;
    if header.manifest.name != IMPORTED_MANIFEST_LOCATOR
        || header.verification.name != IMPORTED_VERIFICATION_LOCATOR
    {
        return Err("project bundle reserved document names are invalid".into());
    }
    validate_blob(
        "project bundle manifest",
        &header.manifest,
        MAX_MANIFEST_BYTES,
    )?;
    validate_blob(
        "project bundle verification",
        &header.verification,
        MAX_VERIFICATION_BYTES,
    )?;
    if header.files.len() > MAX_BUNDLE_FILES {
        return Err(format!(
            "project bundle exceeds the {MAX_BUNDLE_FILES}-file limit"
        ));
    }
    let mut previous: Option<&str> = None;
    for file in &header.files {
        validate_locator(&file.locator)?;
        if previous.is_some_and(|value| value >= file.locator.as_str()) {
            return Err("project bundle files must be unique and sorted by locator".into());
        }
        previous = Some(&file.locator);
        if file.fingerprint.len == 0 || file.bindings.is_empty() {
            return Err("project bundle file fingerprint and bindings must be non-empty".into());
        }
        let mut previous_binding: Option<&ProjectBundleBinding> = None;
        for binding in &file.bindings {
            validate_identifier("project bundle binding ID", &binding.id)?;
            if previous_binding.is_some_and(|value| value >= binding) {
                return Err("project bundle bindings must be unique and sorted".into());
            }
            previous_binding = Some(binding);
        }
    }
    validate_build_options(&ProjectBundleBuildOptions {
        include_sources: header.source_payloads_included,
        source_payload_limit_bytes: header.source_payload_limit_bytes,
        include_models: header.model_payloads_included,
        model_payload_limit_bytes: header.model_payload_limit_bytes,
        commit_mode: CommitMode::NoClobber,
    })?;
    validate_bundle_budgets(
        &header.files,
        header.source_payloads_included,
        header.source_payload_limit_bytes,
        header.model_payloads_included,
        header.model_payload_limit_bytes,
    )?;
    Ok(())
}

fn validate_blob(context: &str, blob: &BundleBlob, maximum: u64) -> Result<(), String> {
    validate_locator(&blob.name)?;
    if blob.fingerprint.len == 0 || blob.fingerprint.len > maximum {
        return Err(format!(
            "{context} length is outside its {maximum}-byte limit"
        ));
    }
    Ok(())
}

fn validate_verification(
    report: &ProjectValidationReport,
    manifest: &ProjectManifest,
) -> Result<(), String> {
    if report.schema != PROJECT_VALIDATION_SCHEMA
        || report.schema_version != PROJECT_MANIFEST_SCHEMA_VERSION
        || report.project_id != manifest.project_id
        || report.manifest_digest != manifest.digest()?
        || report.sources_verified != manifest.sources.len() as u64
        || report.settings_verified != manifest.settings.len() as u64
        || report.presets_verified != manifest.presets.len() as u64
        || report.models_verified != manifest.models.len() as u64
        || report.plans_verified != manifest.plans.len() as u64
        || report.receipts_verified != manifest.receipts.len() as u64
        || report.timelines_verified != manifest.timelines.len() as u64
    {
        return Err("project bundle verification evidence does not match its manifest".into());
    }
    Ok(())
}

fn expected_bundle_len(header: &ProjectBundleHeader, header_len: u64) -> Result<u64, String> {
    let mut total = (PROJECT_BUNDLE_MAGIC.len() as u64)
        .checked_add(8)
        .and_then(|value| value.checked_add(header_len))
        .and_then(|value| value.checked_add(header.manifest.fingerprint.len))
        .and_then(|value| value.checked_add(header.verification.fingerprint.len))
        .ok_or_else(|| "project bundle length overflows".to_string())?;
    for file in &header.files {
        total = total
            .checked_add(file.fingerprint.len)
            .ok_or_else(|| "project bundle length overflows".to_string())?;
    }
    if total > MAX_BUNDLE_BYTES {
        return Err(format!(
            "project bundle exceeds the {MAX_BUNDLE_BYTES}-byte hard limit"
        ));
    }
    Ok(total)
}

fn fingerprint_bytes(bytes: &[u8]) -> FileFingerprint {
    FileFingerprint {
        len: bytes.len() as u64,
        digest: Digest::from_bytes(Sha256::digest(bytes).into()),
    }
}

fn write_hashed(output: &mut File, digest: &mut Sha256, bytes: &[u8]) -> Result<(), String> {
    output
        .write_all(bytes)
        .map_err(|error| format!("write staged project bundle: {error}"))?;
    digest.update(bytes);
    Ok(())
}

fn copy_open_component(
    output: &mut File,
    bundle_digest: &mut Sha256,
    component: &mut OpenBundleFile,
) -> Result<(), String> {
    component.file.seek(SeekFrom::Start(0)).map_err(|error| {
        format!(
            "rewind project component {}: {error}",
            component.path.display()
        )
    })?;
    let mut remaining = component.info.fingerprint.len;
    let mut component_digest = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "project bundle copy length does not fit this platform")?;
        let count = component
            .file
            .read(&mut buffer[..request])
            .map_err(|error| {
                format!(
                    "read project component {}: {error}",
                    component.path.display()
                )
            })?;
        if count == 0 {
            return Err(format!(
                "project component {} ended before its manifest length",
                component.info.locator
            ));
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| format!("write project bundle component: {error}"))?;
        bundle_digest.update(&buffer[..count]);
        component_digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    let mut extra = [0_u8; 1];
    if component
        .file
        .read(&mut extra)
        .map_err(|error| format!("finish project component read: {error}"))?
        != 0
        || Digest::from_bytes(component_digest.finalize().into())
            != component.info.fingerprint.digest
    {
        return Err(format!(
            "project component {} changed while the bundle was built",
            component.info.locator
        ));
    }
    Ok(())
}

fn prepare_project_bundle(
    path: &Path,
    staging_parent: Option<&Path>,
) -> Result<PreparedProjectBundle, String> {
    let (mut file, file_len) = crate::input::open_regular_file(path, "project bundle")?;
    if file_len == 0 || file_len > MAX_BUNDLE_BYTES {
        return Err(format!(
            "project bundle length must be in 1..={MAX_BUNDLE_BYTES} bytes"
        ));
    }
    let mut bundle_digest = Sha256::new();
    let mut magic = vec![0_u8; PROJECT_BUNDLE_MAGIC.len()];
    read_hashed_exact(
        &mut file,
        &mut bundle_digest,
        &mut magic,
        "project bundle magic",
    )?;
    if magic != PROJECT_BUNDLE_MAGIC {
        return Err("project bundle magic is invalid".into());
    }
    let mut length_bytes = [0_u8; 8];
    read_hashed_exact(
        &mut file,
        &mut bundle_digest,
        &mut length_bytes,
        "project bundle header length",
    )?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err("project bundle header length is invalid".into());
    }
    let mut header_bytes = vec![
        0_u8;
        usize::try_from(header_len).map_err(|_| {
            "project bundle header length does not fit this platform"
        })?
    ];
    read_hashed_exact(
        &mut file,
        &mut bundle_digest,
        &mut header_bytes,
        "project bundle header",
    )?;
    let header: ProjectBundleHeader = serde_json::from_slice(&header_bytes)
        .map_err(|error| format!("parse project bundle header: {error}"))?;
    validate_header_shape(&header)?;
    if expected_bundle_len(&header, header_len)? != file_len {
        return Err("project bundle length differs from its header".into());
    }

    let manifest_bytes = read_hashed_blob(
        &mut file,
        &mut bundle_digest,
        header.manifest.fingerprint,
        "project bundle manifest",
    )?;
    let verification_bytes = read_hashed_blob(
        &mut file,
        &mut bundle_digest,
        header.verification.fingerprint,
        "project bundle verification",
    )?;
    let manifest: ProjectManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("parse embedded project manifest: {error}"))?;
    manifest.validate()?;
    validate_reserved_locators(&manifest)?;
    if manifest.project_id != header.project_id || manifest.digest()? != header.manifest_digest {
        return Err("embedded project manifest identity differs from the bundle header".into());
    }
    let verification: ProjectValidationReport = serde_json::from_slice(&verification_bytes)
        .map_err(|error| format!("parse project bundle verification evidence: {error}"))?;
    validate_verification(&verification, &manifest)?;
    let expected_files = expected_bundle_files(
        &manifest,
        header.source_payloads_included,
        header.model_payloads_included,
    )?;
    if expected_files != header.files {
        return Err("project bundle file table differs from its manifest".into());
    }

    let staging = match staging_parent {
        Some(parent) => tempfile::Builder::new()
            .prefix(".denoize-project-import-")
            .tempdir_in(parent),
        None => tempfile::Builder::new()
            .prefix("denoize-project-inspect-")
            .tempdir(),
    }
    .map_err(|error| format!("create private project bundle staging directory: {error}"))?;
    write_private_bytes(staging.path(), IMPORTED_MANIFEST_LOCATOR, &manifest_bytes)?;
    write_private_bytes(
        staging.path(),
        IMPORTED_VERIFICATION_LOCATOR,
        &verification_bytes,
    )?;
    for entry in &header.files {
        extract_bundle_file(&mut file, &mut bundle_digest, staging.path(), entry)?;
    }
    if file
        .stream_position()
        .map_err(|error| format!("inspect project bundle position: {error}"))?
        != file_len
    {
        return Err("project bundle contains trailing bytes".into());
    }
    if file
        .metadata()
        .map_err(|error| format!("reinspect project bundle: {error}"))?
        .len()
        != file_len
    {
        return Err("project bundle changed while it was verified".into());
    }
    validate_staged_payloads(&manifest, &header, staging.path())?;
    let totals = validate_bundle_budgets(
        &header.files,
        header.source_payloads_included,
        header.source_payload_limit_bytes,
        header.model_payloads_included,
        header.model_payload_limit_bytes,
    )?;
    let bundle = FileFingerprint {
        len: file_len,
        digest: Digest::from_bytes(bundle_digest.finalize().into()),
    };
    Ok(PreparedProjectBundle {
        staging,
        info: bundle_info(&header, bundle, totals),
        manifest,
    })
}

fn read_hashed_exact(
    file: &mut File,
    digest: &mut Sha256,
    buffer: &mut [u8],
    context: &str,
) -> Result<(), String> {
    file.read_exact(buffer)
        .map_err(|error| format!("read {context}: {error}"))?;
    digest.update(buffer);
    Ok(())
}

fn read_hashed_blob(
    file: &mut File,
    digest: &mut Sha256,
    expected: FileFingerprint,
    context: &str,
) -> Result<Vec<u8>, String> {
    let length = usize::try_from(expected.len)
        .map_err(|_| format!("{context} length does not fit this platform"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| format!("reserve {context}: {error}"))?;
    bytes.resize(length, 0);
    read_hashed_exact(file, digest, &mut bytes, context)?;
    if fingerprint_bytes(&bytes) != expected {
        return Err(format!("{context} fingerprint differs from its header"));
    }
    Ok(bytes)
}

fn extract_bundle_file(
    input: &mut File,
    bundle_digest: &mut Sha256,
    staging_root: &Path,
    entry: &ProjectBundleFileInfo,
) -> Result<(), String> {
    let path = staged_locator_path(staging_root, &entry.locator)?;
    create_private_parent(staging_root, &path)?;
    let mut output = open_private_new(&path)?;
    let mut remaining = entry.fingerprint.len;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "project bundle extraction length does not fit this platform")?;
        input
            .read_exact(&mut buffer[..request])
            .map_err(|error| format!("read bundled file {}: {error}", entry.locator))?;
        output
            .write_all(&buffer[..request])
            .map_err(|error| format!("write bundled file {}: {error}", entry.locator))?;
        bundle_digest.update(&buffer[..request]);
        digest.update(&buffer[..request]);
        remaining -= request as u64;
    }
    if Digest::from_bytes(digest.finalize().into()) != entry.fingerprint.digest {
        return Err(format!(
            "bundled file {} fingerprint differs from its header",
            entry.locator
        ));
    }
    output
        .sync_all()
        .map_err(|error| format!("sync bundled file {}: {error}", entry.locator))?;
    Ok(())
}

fn write_private_bytes(root: &Path, locator: &str, bytes: &[u8]) -> Result<(), String> {
    let path = staged_locator_path(root, locator)?;
    create_private_parent(root, &path)?;
    let mut file = open_private_new(&path)?;
    file.write_all(bytes)
        .map_err(|error| format!("write imported project document {locator}: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("sync imported project document {locator}: {error}"))
}

fn staged_locator_path(root: &Path, locator: &str) -> Result<PathBuf, String> {
    validate_locator(locator)?;
    let mut path = root.to_path_buf();
    for component in locator.split('/') {
        path.push(component);
    }
    Ok(path)
}

fn create_private_parent(root: &Path, path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "project bundle entry has no parent".to_string())?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("create project bundle staging directory: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mut current = Some(parent);
        while let Some(directory) = current.filter(|directory| directory.starts_with(root)) {
            std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))
                .map_err(|error| format!("secure project bundle staging directory: {error}"))?;
            if directory == root {
                break;
            }
            current = directory.parent();
        }
    }
    Ok(())
}

fn open_private_new(path: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    options.open(path).map_err(|error| {
        format!(
            "create private project bundle file {}: {error}",
            path.display()
        )
    })
}

fn validate_staged_payloads(
    manifest: &ProjectManifest,
    header: &ProjectBundleHeader,
    root: &Path,
) -> Result<(), String> {
    for reference in &manifest.settings {
        let path = resolve_project_locator(root, &reference.locator, "bundled project setting")?;
        let bytes =
            read_bounded_regular(&path, "bundled project setting", MAX_DOCUMENT_FILE_BYTES)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| "bundled project setting is not UTF-8".to_string())?;
        text.parse::<toml::Value>()
            .map_err(|error| format!("parse bundled project setting: {error}"))?;
    }
    for reference in &manifest.presets {
        let path = resolve_project_locator(root, &reference.locator, "bundled project preset")?;
        crate::read_daw_preset(path)?;
    }
    for reference in &manifest.plans {
        let path = resolve_project_locator(root, &reference.locator, "bundled project plan")?;
        if ExecutionPlan::from_file(&path).is_err() {
            ProjectExecutionPlan::from_file(path)?;
        }
    }
    for reference in &manifest.receipts {
        let path = resolve_project_locator(root, &reference.locator, "bundled project receipt")?;
        if SignedExecutionReceipt::from_file(&path).is_err() {
            SignedProjectExecutionReceipt::from_file(path)?;
        }
    }
    if header.source_payloads_included {
        for source in &manifest.sources {
            let path = resolve_project_locator(root, &source.locator, "bundled project source")?;
            let observed = inspect_project_source(path, DecodeLimits::default())?;
            if observed.fingerprint != source.fingerprint
                || observed.timescale != source.timescale
                || observed.channels != source.channels
                || observed.presentation_frames != source.presentation_frames
            {
                return Err(format!(
                    "bundled source {} differs from its project manifest",
                    source.id
                ));
            }
        }
    }
    if header.model_payloads_included {
        for model in &manifest.models {
            validate_staged_model(model, root)?;
        }
    }
    Ok(())
}

fn validate_staged_model(model: &ProjectModelReference, root: &Path) -> Result<(), String> {
    let package = resolve_project_locator(root, &model.package.locator, "bundled model package")?;
    let public_key =
        resolve_project_locator(root, &model.public_key.locator, "bundled model public key")?;
    let runtime = RuntimeModelPackage::open(package, public_key)?;
    let info = runtime.info();
    if info.package_id != model.package_id
        || info.package_revision != model.package_revision
        || info.signing_key_id != model.signing_key_id
        || info.license_spdx != model.license_spdx
    {
        return Err(format!(
            "bundled model package {} differs from its trusted project reference",
            model.id
        ));
    }
    Ok(())
}

fn bundle_info(
    header: &ProjectBundleHeader,
    bundle: FileFingerprint,
    totals: BundleByteTotals,
) -> ProjectBundleInfo {
    ProjectBundleInfo {
        schema: PROJECT_BUNDLE_SCHEMA.into(),
        schema_version: PROJECT_BUNDLE_VERSION,
        project_id: header.project_id.clone(),
        manifest_digest: header.manifest_digest,
        bundle,
        manifest: header.manifest.fingerprint,
        verification: header.verification.fingerprint,
        source_payloads_included: header.source_payloads_included,
        source_payload_limit_bytes: header.source_payload_limit_bytes,
        source_payload_bytes: totals.sources,
        model_payloads_included: header.model_payloads_included,
        model_payload_limit_bytes: header.model_payload_limit_bytes,
        model_payload_bytes: totals.models,
        document_bytes: totals.documents,
        files: header.files.clone(),
    }
}

fn absolute_import_destination(path: &Path) -> Result<PathBuf, String> {
    let name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "project bundle import destination must name a directory".to_string())?;
    if name == "." || name == ".." {
        return Err("project bundle import destination is unsafe".into());
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|error| {
        format!(
            "resolve project bundle import parent {}: {error}",
            parent.display()
        )
    })?;
    if !parent.is_dir() {
        return Err("project bundle import parent is not a directory".into());
    }
    Ok(parent.join(name))
}

fn ensure_destination_absent(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(format!(
            "project bundle import destination already exists: {}",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "inspect project bundle import destination {}: {error}",
            path.display()
        )),
    }
}

#[cfg(target_os = "linux")]
fn rename_directory_noclobber(source: &Path, destination: &Path) -> Result<(), String> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt as _;
    const RENAME_NOREPLACE: libc::c_uint = 1;
    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| "project bundle staging path contains NUL".to_string())?;
    let destination = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| "project bundle destination contains NUL".to_string())?;
    // SAFETY: both pointers come from live `CString` values, are NUL-terminated,
    // and remain valid for the duration of the syscall.
    let result = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(format!(
            "publish imported project without clobbering: {}",
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(target_os = "macos")]
fn rename_directory_noclobber(source: &Path, destination: &Path) -> Result<(), String> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt as _;
    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| "project bundle staging path contains NUL".to_string())?;
    let destination = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| "project bundle destination contains NUL".to_string())?;
    // SAFETY: both pointers come from live `CString` values, are NUL-terminated,
    // and remain valid for the duration of the call.
    let result =
        unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
    if result == 0 {
        Ok(())
    } else {
        Err(format!(
            "publish imported project without clobbering: {}",
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(windows)]
fn rename_directory_noclobber(source: &Path, destination: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt as _;
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_WRITE_THROUGH};
    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: both buffers are NUL-terminated, remain live for the call, and
    // MoveFileExW receives no replacement flag, so an existing path is kept.
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_WRITE_THROUGH,
        )
    };
    if result != 0 {
        Ok(())
    } else {
        Err(format!(
            "publish imported project without clobbering: {}",
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn rename_directory_noclobber(_source: &Path, _destination: &Path) -> Result<(), String> {
    Err("atomic no-clobber project bundle import is unsupported on this platform".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PresentationRegion, ProjectSelection, ProjectSource, ProjectTimeline};
    use hound::{SampleFormat, WavSpec};

    fn write_wav(path: &Path) {
        let mut writer = hound::WavWriter::create(
            path,
            WavSpec {
                channels: 1,
                sample_rate: 8_000,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            },
        )
        .unwrap();
        for sample in [0.1_f32, 0.2, 0.3, 0.4] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn fixture(root: &Path) -> (PathBuf, ProjectManifest) {
        let source_path = root.join("source.wav");
        let setting_path = root.join("settings.toml");
        let license_path = root.join("source-license.txt");
        write_wav(&source_path);
        std::fs::write(&setting_path, "strength = 0.5\n").unwrap();
        std::fs::write(&license_path, "CC0-1.0\n").unwrap();
        let inspection = inspect_project_source(&source_path, DecodeLimits::default()).unwrap();
        let license =
            super::super::project_artifact_reference("source-license", &license_path, root)
                .unwrap();
        let source = ProjectSource::new("source", "source.wav", inspection, Some(license)).unwrap();
        let selection = ProjectSelection::new(
            "selection",
            "source",
            PresentationRegion::new(source.fingerprint, 8_000, 0, 4).unwrap(),
            vec![0],
            0,
            0,
            0,
        )
        .unwrap();
        let timeline = ProjectTimeline::new("main", 8_000, 1, vec![selection]).unwrap();
        let setting =
            super::super::project_artifact_reference("settings", &setting_path, root).unwrap();
        let manifest = ProjectManifest::new(
            "bundle-test",
            vec![source],
            vec![timeline],
            vec![setting],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .unwrap();
        let manifest_path = root.join("input-project.json");
        super::super::write_project_manifest(
            &manifest_path,
            &manifest,
            CommitMode::NoClobber,
            false,
        )
        .unwrap();
        (manifest_path, manifest)
    }

    #[test]
    fn references_only_bundle_omits_source_and_imports_documents_atomically() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest_path, manifest) = fixture(directory.path());
        let bundle = directory.path().join("project.dpb");
        let built = build_project_bundle(
            &manifest_path,
            directory.path(),
            &bundle,
            &ProjectBundleBuildOptions::default(),
            DecodeLimits::default(),
        )
        .unwrap();
        assert!(!built.source_payloads_included);
        assert_eq!(built.source_payload_bytes, 0);
        assert_eq!(inspect_project_bundle(&bundle).unwrap(), built);

        let imported = directory.path().join("imported");
        let report = import_project_bundle(&bundle, &imported).unwrap();
        assert_eq!(report.omitted_sources, vec!["source"]);
        assert!(imported.join(IMPORTED_MANIFEST_LOCATOR).is_file());
        assert!(imported.join(IMPORTED_VERIFICATION_LOCATOR).is_file());
        assert!(imported.join("settings.toml").is_file());
        assert!(imported.join("source-license.txt").is_file());
        assert!(!imported.join("source.wav").exists());
        let imported_manifest =
            ProjectManifest::from_file(imported.join(IMPORTED_MANIFEST_LOCATOR)).unwrap();
        assert_eq!(imported_manifest, manifest);
    }

    #[test]
    fn bounded_source_bundle_round_trips_and_tampering_never_publishes() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest_path, manifest) = fixture(directory.path());
        let bundle = directory.path().join("project-with-source.dpb");
        let options = ProjectBundleBuildOptions {
            include_sources: true,
            source_payload_limit_bytes: manifest.sources[0].fingerprint.len,
            ..ProjectBundleBuildOptions::default()
        };
        let built = build_project_bundle(
            &manifest_path,
            directory.path(),
            &bundle,
            &options,
            DecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(
            built.source_payload_bytes,
            manifest.sources[0].fingerprint.len
        );
        let imported = directory.path().join("complete-import");
        import_project_bundle(&bundle, &imported).unwrap();
        let imported_manifest =
            ProjectManifest::from_file(imported.join(IMPORTED_MANIFEST_LOCATOR)).unwrap();
        validate_project_files(&imported_manifest, &imported, DecodeLimits::default()).unwrap();

        let mut bytes = std::fs::read(&bundle).unwrap();
        *bytes.last_mut().unwrap() ^= 0x5a;
        let tampered = directory.path().join("tampered.dpb");
        std::fs::write(&tampered, bytes).unwrap();
        let rejected = directory.path().join("rejected-import");
        assert!(import_project_bundle(&tampered, &rejected).is_err());
        assert!(!rejected.exists());
    }

    #[test]
    fn import_never_changes_an_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let (manifest_path, _) = fixture(directory.path());
        let bundle = directory.path().join("project.dpb");
        build_project_bundle(
            &manifest_path,
            directory.path(),
            &bundle,
            &ProjectBundleBuildOptions::default(),
            DecodeLimits::default(),
        )
        .unwrap();
        let destination = directory.path().join("existing");
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(destination.join("keep.txt"), b"unchanged").unwrap();
        assert!(import_project_bundle(&bundle, &destination).is_err());
        assert_eq!(
            std::fs::read(destination.join("keep.txt")).unwrap(),
            b"unchanged"
        );
    }
}
