use super::{
    acquire_lock, cache_dir, ensure_provenance, open_existing_regular_file, path_for_catalog_model,
    provenance_directory, provenance_path, read_provenance, remove_file_if_present,
    sha256_open_file_exact, sidecar, update_catalog_model_with_options_and_progress,
    valid_provenance_url, validate_provenance, verify_bytes_at, CatalogModel, ModelDownloadOptions,
    ModelInstallationSource, ModelProvenance, ModelSpec, MAX_JSON_SAFE_INTEGER,
    MAX_MODEL_PROVENANCE_BYTES, MAX_PARTIAL_METADATA_BYTES, MODEL_PROVENANCE_VERSION,
    PARTIAL_METADATA_VERSION,
};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The integrity state of one package in the active authenticated catalog.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelCacheModelStatus {
    /// The optional package has not been installed.
    Missing,
    /// Artifact bytes and their content-addressed provenance are valid.
    Healthy,
    /// The artifact exists but its size or digest is wrong.
    Corrupt,
    /// Artifact bytes are valid, but the expected provenance is absent.
    ProvenanceMissing,
    /// Artifact bytes are valid, but the expected provenance is malformed or stale.
    ProvenanceInvalid,
    /// A symlink, directory, device, or other unsafe entry occupies an expected path.
    Unsafe,
}

/// A cache condition found by [`doctor_model_cache`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ModelCacheIssueKind {
    MissingArtifact,
    CorruptArtifact,
    MissingProvenance,
    InvalidProvenance,
    IncompleteDownload,
    StaleDownloadState,
    OrphanedEntry,
    UnsafeEntry,
}

/// One path-level diagnostic. `prunable` is true only when denoize can prove
/// ownership and remove the entry without following links or touching active
/// catalog/model state.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelCacheIssue {
    pub kind: ModelCacheIssueKind,
    pub path: PathBuf,
    pub model: Option<String>,
    pub detail: String,
    pub prunable: bool,
}

/// Health of one active-catalog package.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelCacheModel {
    pub name: String,
    pub path: PathBuf,
    pub status: ModelCacheModelStatus,
    /// Validated provenance for a healthy installed package.
    pub provenance: Option<ModelProvenance>,
    pub issues: Vec<ModelCacheIssue>,
}

/// Read-only inventory of the model cache.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelCacheReport {
    pub cache_dir: PathBuf,
    pub catalog_sequence: u64,
    pub catalog_sha256: String,
    pub models: Vec<ModelCacheModel>,
    pub issues: Vec<ModelCacheIssue>,
}

impl ModelCacheReport {
    /// True when no installed package or denoize-owned cache state needs
    /// attention. Optional packages that are simply missing do not make a
    /// fresh cache unhealthy.
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
            && self.models.iter().all(|model| {
                matches!(
                    model.status,
                    ModelCacheModelStatus::Missing | ModelCacheModelStatus::Healthy
                ) && model
                    .issues
                    .iter()
                    .all(|issue| matches!(issue.kind, ModelCacheIssueKind::MissingArtifact))
            })
    }
}

/// Result of repairing one active-catalog package.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelRepairOutcome {
    AlreadyHealthy,
    ProvenanceRebuilt,
    ArtifactInstalled,
}

/// Result of a cache prune. Dry runs populate `would_remove`; applying a prune
/// populates `removed`. Unsafe or unproven paths remain in `retained`.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelPruneReport {
    pub cache_dir: PathBuf,
    pub dry_run: bool,
    pub would_remove: Vec<PathBuf>,
    pub removed: Vec<PathBuf>,
    pub retained: Vec<ModelCacheIssue>,
}

enum PathKind {
    Missing,
    File(u64),
    Directory,
    Unsafe,
}

fn path_kind(path: &Path) -> Result<PathKind, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Ok(PathKind::Unsafe),
        Ok(metadata) if metadata.is_file() => Ok(PathKind::File(metadata.len())),
        Ok(metadata) if metadata.is_dir() => Ok(PathKind::Directory),
        Ok(_) => Ok(PathKind::Unsafe),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(PathKind::Missing),
        Err(error) => Err(format!("failed to inspect {}: {error}", path.display())),
    }
}

fn issue(
    kind: ModelCacheIssueKind,
    path: &Path,
    model: Option<&str>,
    detail: impl Into<String>,
    prunable: bool,
) -> ModelCacheIssue {
    ModelCacheIssue {
        kind,
        path: path.to_path_buf(),
        model: model.map(str::to_owned),
        detail: detail.into(),
        prunable,
    }
}

fn expected_provenance_path(model: &CatalogModel, destination: &Path) -> Result<PathBuf, String> {
    provenance_path(&ModelSpec::catalog(model), destination)
}

fn inspect_model(model: &CatalogModel, cache: &Path) -> Result<ModelCacheModel, String> {
    let spec = ModelSpec::catalog(model);
    let model_directory = cache.join(model.name());
    let destination = model_directory.join(model.filename());
    let mut issues = Vec::new();
    if matches!(
        path_kind(&model_directory)?,
        PathKind::File(_) | PathKind::Unsafe
    ) {
        issues.push(issue(
            ModelCacheIssueKind::UnsafeEntry,
            &model_directory,
            Some(model.name()),
            "model package path is not a regular directory",
            false,
        ));
        return Ok(ModelCacheModel {
            name: model.name().to_string(),
            path: destination,
            status: ModelCacheModelStatus::Unsafe,
            provenance: None,
            issues,
        });
    }
    let mut valid_provenance = None;
    let status = match path_kind(&destination)? {
        PathKind::Missing => {
            issues.push(issue(
                ModelCacheIssueKind::MissingArtifact,
                &destination,
                Some(model.name()),
                "model is not installed",
                false,
            ));
            ModelCacheModelStatus::Missing
        }
        PathKind::File(_) => match verify_bytes_at(&spec, &destination) {
            Err(error) => {
                issues.push(issue(
                    ModelCacheIssueKind::CorruptArtifact,
                    &destination,
                    Some(model.name()),
                    error,
                    false,
                ));
                ModelCacheModelStatus::Corrupt
            }
            Ok(_) => {
                let provenance_path = expected_provenance_path(model, &destination)?;
                match read_provenance(&provenance_path) {
                    Ok(None) => {
                        issues.push(issue(
                            ModelCacheIssueKind::MissingProvenance,
                            &provenance_path,
                            Some(model.name()),
                            "verified artifact has no authenticated provenance",
                            false,
                        ));
                        ModelCacheModelStatus::ProvenanceMissing
                    }
                    Ok(Some(provenance)) => match validate_provenance(&spec, &provenance) {
                        Ok(()) => {
                            valid_provenance = Some(provenance);
                            ModelCacheModelStatus::Healthy
                        }
                        Err(error) => {
                            issues.push(issue(
                                ModelCacheIssueKind::InvalidProvenance,
                                &provenance_path,
                                Some(model.name()),
                                error,
                                false,
                            ));
                            ModelCacheModelStatus::ProvenanceInvalid
                        }
                    },
                    Err(error) => {
                        issues.push(issue(
                            ModelCacheIssueKind::InvalidProvenance,
                            &provenance_path,
                            Some(model.name()),
                            error,
                            false,
                        ));
                        ModelCacheModelStatus::ProvenanceInvalid
                    }
                }
            }
        },
        PathKind::Directory | PathKind::Unsafe => {
            issues.push(issue(
                ModelCacheIssueKind::UnsafeEntry,
                &destination,
                Some(model.name()),
                "expected model path is not a regular file",
                false,
            ));
            ModelCacheModelStatus::Unsafe
        }
    };

    inspect_download_state(model, &destination, status, &mut issues)?;
    Ok(ModelCacheModel {
        name: model.name().to_string(),
        path: destination,
        status,
        provenance: valid_provenance,
        issues,
    })
}

fn read_partial_metadata(path: &Path) -> Result<Option<super::PartialMetadata>, String> {
    let Some(file) = open_existing_regular_file(path, "partial download metadata")? else {
        return Ok(None);
    };
    let length = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_PARTIAL_METADATA_BYTES {
        return Ok(None);
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(MAX_PARTIAL_METADATA_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_PARTIAL_METADATA_BYTES {
        return Ok(None);
    }
    Ok(serde_json::from_slice(&bytes).ok())
}

fn partial_metadata_matches(metadata: &super::PartialMetadata, model: &CatalogModel) -> bool {
    metadata.version == PARTIAL_METADATA_VERSION
        && !metadata.source_id.is_empty()
        && metadata.expected_sha256 == model.sha256()
        && metadata
            .total
            .is_none_or(|total| total == model.size_bytes())
}

fn inspect_download_state(
    model: &CatalogModel,
    destination: &Path,
    status: ModelCacheModelStatus,
    issues: &mut Vec<ModelCacheIssue>,
) -> Result<(), String> {
    let partial = sidecar(destination, ".part");
    let metadata = sidecar(destination, ".part.meta");
    let metadata_tmp = sidecar(&metadata, ".tmp");
    let partial_kind = path_kind(&partial)?;
    let metadata_kind = path_kind(&metadata)?;
    let temporary_kind = path_kind(&metadata_tmp)?;

    for (path, kind) in [
        (&partial, &partial_kind),
        (&metadata, &metadata_kind),
        (&metadata_tmp, &temporary_kind),
    ] {
        if matches!(kind, PathKind::Directory | PathKind::Unsafe) {
            issues.push(issue(
                ModelCacheIssueKind::UnsafeEntry,
                path,
                Some(model.name()),
                "download sidecar is not a regular file",
                false,
            ));
        }
    }

    let present = |kind: &PathKind| matches!(kind, PathKind::File(_));
    if matches!(status, ModelCacheModelStatus::Healthy) {
        for (path, kind) in [
            (&partial, &partial_kind),
            (&metadata, &metadata_kind),
            (&metadata_tmp, &temporary_kind),
        ] {
            if present(kind) {
                issues.push(issue(
                    ModelCacheIssueKind::StaleDownloadState,
                    path,
                    Some(model.name()),
                    "verified model has obsolete download state",
                    true,
                ));
            }
        }
        return Ok(());
    }

    if present(&temporary_kind) {
        issues.push(issue(
            ModelCacheIssueKind::StaleDownloadState,
            &metadata_tmp,
            Some(model.name()),
            "abandoned atomic metadata temporary",
            true,
        ));
    }

    match (&partial_kind, &metadata_kind) {
        (PathKind::File(partial_size), PathKind::File(_))
            if *partial_size <= model.size_bytes()
                && read_partial_metadata(&metadata)?
                    .as_ref()
                    .is_some_and(|value| partial_metadata_matches(value, model)) =>
        {
            issues.push(issue(
                ModelCacheIssueKind::IncompleteDownload,
                &partial,
                Some(model.name()),
                format!(
                    "resumable download contains {partial_size} of {} bytes",
                    model.size_bytes()
                ),
                false,
            ));
        }
        (PathKind::Missing, PathKind::Missing) => {}
        _ => {
            for (path, kind) in [(&partial, &partial_kind), (&metadata, &metadata_kind)] {
                if present(kind) {
                    issues.push(issue(
                        ModelCacheIssueKind::StaleDownloadState,
                        path,
                        Some(model.name()),
                        "download sidecars are incomplete or do not match the active package",
                        true,
                    ));
                }
            }
        }
    }
    Ok(())
}

fn basic_provenance_is_managed(provenance: &ModelProvenance, directory_name: &str) -> bool {
    provenance.version == MODEL_PROVENANCE_VERSION
        && provenance.model_name == directory_name
        && !provenance.filename.is_empty()
        && Path::new(&provenance.filename)
            .file_name()
            .is_some_and(|name| name == provenance.filename.as_str())
        && provenance.artifact_sha256.len() == 64
        && provenance
            .artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        && provenance.artifact_size_bytes > 0
        && provenance.catalog_sha256.len() == 64
        && provenance
            .catalog_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        && provenance.catalog_signing_key_id.len() == 16
        && provenance
            .catalog_signing_key_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
        && provenance.installed_at_unix_seconds <= MAX_JSON_SAFE_INTEGER
        && super::catalog::catalog_origin_is_safe(&provenance.catalog_origin)
        && match &provenance.installation_source {
            ModelInstallationSource::CatalogUrl { url }
            | ModelInstallationSource::AlternateUrl { url } => valid_provenance_url(url),
            ModelInstallationSource::LocalFile
            | ModelInstallationSource::CompletedPartial
            | ModelInstallationSource::ExistingCacheMigration => true,
        }
}

fn managed_provenance_entry(path: &Path, directory_name: &str) -> Result<bool, String> {
    if !matches!(path_kind(path)?, PathKind::File(length) if length <= MAX_MODEL_PROVENANCE_BYTES) {
        return Ok(false);
    }
    let provenance = match read_provenance(path) {
        Ok(Some(provenance)) => provenance,
        Ok(None) | Err(_) => return Ok(false),
    };
    if !basic_provenance_is_managed(&provenance, directory_name) {
        return Ok(false);
    }
    let expected_name = format!(
        "{}.{}.json",
        provenance.artifact_sha256, provenance.catalog_sha256
    );
    Ok(path.file_name().and_then(|value| value.to_str()) == Some(expected_name.as_str()))
}

fn managed_orphan_destination(path: &Path) -> Result<Option<PathBuf>, String> {
    let Some(directory_name) = path.file_name().and_then(|value| value.to_str()) else {
        return Ok(None);
    };
    if !matches!(path_kind(path)?, PathKind::Directory) {
        return Ok(None);
    }
    let provenance_dir = path.join(".provenance");
    if !matches!(path_kind(&provenance_dir)?, PathKind::Directory) {
        return Ok(None);
    }
    let mut matching_destination = None;
    for entry in fs::read_dir(&provenance_dir)
        .map_err(|error| format!("failed to read {}: {error}", provenance_dir.display()))?
    {
        let entry = entry
            .map_err(|error| format!("failed to read {}: {error}", provenance_dir.display()))?;
        let candidate = entry.path();
        if !managed_provenance_entry(&candidate, directory_name)? {
            return Ok(None);
        }
        let Some(provenance) = read_provenance(&candidate)? else {
            return Ok(None);
        };
        let destination = path.join(&provenance.filename);
        let Some(mut artifact) =
            open_existing_regular_file(&destination, "orphaned model artifact")?
        else {
            continue;
        };
        let artifact_matches =
            sha256_open_file_exact(&mut artifact, &destination, provenance.artifact_size_bytes)
                .is_ok_and(|actual| actual == provenance.artifact_sha256);
        if !artifact_matches {
            continue;
        }
        matching_destination.get_or_insert(destination);
    }
    let Some(destination) = matching_destination else {
        return Ok(None);
    };
    let partial = sidecar(&destination, ".part");
    let metadata = sidecar(&destination, ".part.meta");
    let metadata_tmp = sidecar(&metadata, ".tmp");
    for sidecar_path in [&partial, &metadata, &metadata_tmp] {
        if !matches!(
            path_kind(sidecar_path)?,
            PathKind::Missing | PathKind::File(_)
        ) {
            return Ok(None);
        }
    }
    let allowed: HashSet<PathBuf> = [
        destination.clone(),
        partial,
        metadata,
        metadata_tmp,
        provenance_dir,
    ]
    .into_iter()
    .collect();
    let layout_is_owned = fs::read_dir(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?
        .all(|entry| entry.is_ok_and(|entry| allowed.contains(&entry.path())));
    if layout_is_owned {
        return Ok(Some(destination));
    }
    Ok(None)
}

fn scan_known_directory(
    model: &CatalogModel,
    destination: &Path,
    issues: &mut Vec<ModelCacheIssue>,
) -> Result<(), String> {
    let directory = destination
        .parent()
        .ok_or_else(|| "invalid model cache path".to_string())?;
    if !matches!(path_kind(directory)?, PathKind::Directory) {
        return Ok(());
    }
    let partial = sidecar(destination, ".part");
    let metadata = sidecar(destination, ".part.meta");
    let metadata_tmp = sidecar(&metadata, ".tmp");
    let provenance_dir = provenance_directory(destination)?;
    let expected_provenance = expected_provenance_path(model, destination)?;
    let allowed: HashSet<PathBuf> = [
        destination.to_path_buf(),
        partial,
        metadata,
        metadata_tmp,
        provenance_dir.clone(),
    ]
    .into_iter()
    .collect();

    if matches!(
        path_kind(&provenance_dir)?,
        PathKind::File(_) | PathKind::Unsafe
    ) {
        issues.push(issue(
            ModelCacheIssueKind::UnsafeEntry,
            &provenance_dir,
            Some(model.name()),
            "model provenance path is not a regular directory",
            false,
        ));
    }

    for entry in fs::read_dir(directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
    {
        let entry =
            entry.map_err(|error| format!("failed to read {}: {error}", directory.display()))?;
        let path = entry.path();
        if allowed.contains(&path) {
            continue;
        }
        let regular_entry = !matches!(path_kind(&path)?, PathKind::Unsafe);
        issues.push(issue(
            if regular_entry {
                ModelCacheIssueKind::OrphanedEntry
            } else {
                ModelCacheIssueKind::UnsafeEntry
            },
            &path,
            Some(model.name()),
            "unknown package entry is retained because denoize ownership is not proven",
            false,
        ));
    }

    if matches!(path_kind(&provenance_dir)?, PathKind::Directory) {
        for entry in fs::read_dir(&provenance_dir)
            .map_err(|error| format!("failed to read {}: {error}", provenance_dir.display()))?
        {
            let entry = entry
                .map_err(|error| format!("failed to read {}: {error}", provenance_dir.display()))?;
            let path = entry.path();
            if path == expected_provenance {
                continue;
            }
            let prunable = managed_provenance_entry(&path, model.name())?;
            issues.push(issue(
                if prunable {
                    ModelCacheIssueKind::OrphanedEntry
                } else {
                    ModelCacheIssueKind::UnsafeEntry
                },
                &path,
                Some(model.name()),
                if prunable {
                    "provenance belongs to an inactive catalog identity"
                } else {
                    "unknown provenance entry is retained because denoize ownership is not proven"
                },
                prunable,
            ));
        }
    }
    Ok(())
}

fn scan_cache_root(
    cache: &Path,
    models: &[CatalogModel],
    issues: &mut Vec<ModelCacheIssue>,
) -> Result<(), String> {
    match path_kind(cache)? {
        PathKind::Missing => return Ok(()),
        PathKind::Directory => {}
        _ => {
            return Err(format!(
                "model cache is not a regular directory: {}",
                cache.display()
            ))
        }
    }
    let known: BTreeMap<&str, &CatalogModel> =
        models.iter().map(|model| (model.name(), model)).collect();
    for entry in fs::read_dir(cache)
        .map_err(|error| format!("failed to read {}: {error}", cache.display()))?
    {
        let entry =
            entry.map_err(|error| format!("failed to read {}: {error}", cache.display()))?;
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            issues.push(issue(
                ModelCacheIssueKind::UnsafeEntry,
                &path,
                None,
                "cache entry name is not valid UTF-8",
                false,
            ));
            continue;
        };
        if matches!(name.as_str(), ".catalog" | ".locks") {
            if !matches!(path_kind(&path)?, PathKind::Directory) {
                issues.push(issue(
                    ModelCacheIssueKind::UnsafeEntry,
                    &path,
                    None,
                    "reserved model state path is not a directory",
                    false,
                ));
            }
            continue;
        }
        if let Some(model) = known.get(name.as_str()) {
            let destination = path.join(model.filename());
            scan_known_directory(model, &destination, issues)?;
            continue;
        }
        let managed = managed_orphan_destination(&path)?;
        let prunable = managed.is_some();
        let regular_entry = !matches!(path_kind(&path)?, PathKind::Unsafe);
        issues.push(issue(
            if regular_entry {
                ModelCacheIssueKind::OrphanedEntry
            } else {
                ModelCacheIssueKind::UnsafeEntry
            },
            &path,
            None,
            if prunable {
                "package is no longer present in the active catalog"
            } else {
                "unknown cache entry is retained because denoize ownership is not proven"
            },
            prunable,
        ));
    }
    Ok(())
}

pub(super) fn doctor_catalog(catalog: &super::ModelCatalog) -> Result<ModelCacheReport, String> {
    let cache = cache_dir()?;
    super::reject_symlink(&cache)?;
    let mut models = Vec::with_capacity(catalog.models().len());
    for model in catalog.models() {
        models.push(inspect_model(model, &cache)?);
    }
    let mut issues = Vec::new();
    scan_cache_root(&cache, catalog.models(), &mut issues)?;
    Ok(ModelCacheReport {
        cache_dir: cache,
        catalog_sequence: catalog.sequence(),
        catalog_sha256: catalog.sha256().to_string(),
        models,
        issues,
    })
}

/// Inventory every active package and denoize-owned cache entry without
/// modifying package artifacts, provenance, or download sidecars. Resolving
/// the active catalog retains its normal authenticated cache-promotion rules.
pub fn doctor_model_cache() -> Result<ModelCacheReport, String> {
    let catalog = super::active_catalog()?;
    doctor_model_cache_for_catalog(&catalog)
}

/// Inventory against an already validated catalog, allowing callers to keep
/// the catalog rows and cache health bound to one authenticated identity.
pub fn doctor_model_cache_for_catalog(
    catalog: &super::ModelCatalog,
) -> Result<ModelCacheReport, String> {
    doctor_catalog(catalog)
}

fn repair_provenance<C>(
    model: &CatalogModel,
    destination: &Path,
    cancelled: &mut C,
) -> Result<(), String>
where
    C: FnMut() -> bool,
{
    let spec = ModelSpec::catalog(model);
    let lock = acquire_lock(destination, cancelled)?;
    if cancelled() {
        return Err("cancelled".into());
    }
    verify_bytes_at(&spec, destination)?;
    let path = expected_provenance_path(model, destination)?;
    match read_provenance(&path) {
        Ok(Some(provenance)) if validate_provenance(&spec, &provenance).is_ok() => {}
        Ok(None) => {
            ensure_provenance(
                &spec,
                destination,
                ModelInstallationSource::ExistingCacheMigration,
            )?;
        }
        Ok(Some(_)) | Err(_) => {
            remove_file_if_present(&path)?;
            ensure_provenance(
                &spec,
                destination,
                ModelInstallationSource::ExistingCacheMigration,
            )?;
        }
    }
    drop(lock);
    Ok(())
}

fn remove_invalid_provenance_before_install(
    model: &CatalogModel,
    destination: &Path,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<(), String> {
    let spec = ModelSpec::catalog(model);
    let path = expected_provenance_path(model, destination)?;
    let lock = acquire_lock(destination, cancelled)?;
    if cancelled() {
        return Err("cancelled".into());
    }
    let provenance_is_usable = match read_provenance(&path) {
        Ok(None) => true,
        Ok(Some(provenance)) => validate_provenance(&spec, &provenance).is_ok(),
        Err(_) => false,
    };
    if !provenance_is_usable {
        remove_file_if_present(&path)?;
    }
    drop(lock);
    Ok(())
}

/// Repair one active-catalog package. Verified bytes only need provenance
/// rebuilt; missing or corrupt bytes are reacquired and atomically replaced.
pub fn repair_catalog_model_with_options(
    model: &CatalogModel,
    options: &ModelDownloadOptions,
) -> Result<ModelRepairOutcome, String> {
    repair_catalog_model_with_options_and_progress(model, options, || false, |_, _| {})
}

/// Progress/cancellation-aware repair for interactive frontends.
pub fn repair_catalog_model_with_options_and_progress<C, P>(
    model: &CatalogModel,
    options: &ModelDownloadOptions,
    mut cancelled: C,
    mut progress: P,
) -> Result<ModelRepairOutcome, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    if cancelled() {
        return Err("cancelled".into());
    }
    let cache = cache_dir()?;
    let health = inspect_model(model, &cache)?;
    match health.status {
        ModelCacheModelStatus::Healthy => Ok(ModelRepairOutcome::AlreadyHealthy),
        ModelCacheModelStatus::ProvenanceMissing | ModelCacheModelStatus::ProvenanceInvalid => {
            repair_provenance(model, &health.path, &mut cancelled)?;
            Ok(ModelRepairOutcome::ProvenanceRebuilt)
        }
        ModelCacheModelStatus::Missing | ModelCacheModelStatus::Corrupt => {
            remove_invalid_provenance_before_install(model, &health.path, &mut cancelled)?;
            update_catalog_model_with_options_and_progress(
                model,
                options,
                &mut cancelled,
                &mut progress,
            )?;
            Ok(ModelRepairOutcome::ArtifactInstalled)
        }
        ModelCacheModelStatus::Unsafe => Err(format!(
            "refusing to repair unsafe model cache path: {}",
            health.path.display()
        )),
    }
}

fn remove_safe_tree(path: &Path) -> Result<(), String> {
    match path_kind(path)? {
        PathKind::Missing => Ok(()),
        PathKind::File(_) => {
            remove_file_if_present(path)?;
            Ok(())
        }
        // `remove_dir_all` does not follow symbolic links and uses a
        // descriptor-relative, TOCTOU-resistant implementation on Unix. The
        // orphan layout is re-proven under its per-model lock immediately
        // before this call.
        PathKind::Directory => fs::remove_dir_all(path)
            .map_err(|error| format!("failed to remove {}: {error}", path.display())),
        PathKind::Unsafe => Err(format!(
            "refusing to prune special or symbolic-link entry: {}",
            path.display()
        )),
    }
}

fn prune_known_model(
    model: &CatalogModel,
    dry_run: bool,
    would_remove: &mut Vec<PathBuf>,
    removed: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let destination = path_for_catalog_model(model)?;
    if dry_run {
        let cache = cache_dir()?;
        let mut health = inspect_model(model, &cache)?;
        scan_known_directory(model, &destination, &mut health.issues)?;
        would_remove.extend(
            health
                .issues
                .into_iter()
                .filter(|issue| issue.prunable)
                .map(|issue| issue.path),
        );
        return Ok(());
    }
    let mut never_cancelled = || false;
    let lock = acquire_lock(&destination, &mut never_cancelled)?;
    let cache = cache_dir()?;
    let mut health = inspect_model(model, &cache)?;
    scan_known_directory(model, &destination, &mut health.issues)?;
    let mut paths: Vec<_> = health
        .issues
        .iter()
        .filter(|issue| issue.prunable)
        .map(|issue| issue.path.clone())
        .collect();
    paths.sort();
    paths.dedup();
    for path in paths {
        if dry_run {
            would_remove.push(path);
        } else {
            remove_safe_tree(&path)?;
            removed.push(path);
        }
    }
    if !dry_run {
        if let Some(parent) = destination.parent() {
            let _ = fs::remove_dir(parent);
        }
    }
    drop(lock);
    Ok(())
}

fn prune_managed_orphan(
    path: &Path,
    dry_run: bool,
    would_remove: &mut Vec<PathBuf>,
    removed: &mut Vec<PathBuf>,
) -> Result<bool, String> {
    let Some(destination) = managed_orphan_destination(path)? else {
        return Ok(false);
    };
    if dry_run {
        would_remove.push(path.to_path_buf());
        return Ok(true);
    }
    let mut never_cancelled = || false;
    let lock = acquire_lock(&destination, &mut never_cancelled)?;
    if managed_orphan_destination(path)?.is_none() {
        drop(lock);
        return Ok(false);
    }
    remove_safe_tree(path)?;
    removed.push(path.to_path_buf());
    drop(lock);
    Ok(true)
}

/// Remove stale sidecars, superseded provenance, and whole orphan package
/// directories whose content-addressed provenance and artifact bytes match the
/// denoize-managed layout. Unknown and special entries are always retained.
/// Set `dry_run` to preview exact paths.
pub fn prune_model_cache(dry_run: bool) -> Result<ModelPruneReport, String> {
    let catalog = super::active_catalog()?;
    prune_catalog(&catalog, dry_run)
}

pub(super) fn prune_catalog(
    catalog: &super::ModelCatalog,
    dry_run: bool,
) -> Result<ModelPruneReport, String> {
    let report = doctor_catalog(catalog)?;
    let mut would_remove = Vec::new();
    let mut removed = Vec::new();
    for model in catalog.models() {
        prune_known_model(model, dry_run, &mut would_remove, &mut removed)?;
    }
    let mut retained: Vec<_> = report
        .models
        .iter()
        .flat_map(|model| model.issues.iter())
        .filter(|issue| !issue.prunable && issue.kind != ModelCacheIssueKind::MissingArtifact)
        .cloned()
        .collect();
    for issue in &report.issues {
        if issue.model.is_some() && issue.prunable {
            // Active-package stale state was already handled under that
            // package's lock by `prune_known_model`.
            continue;
        }
        if issue.prunable
            && prune_managed_orphan(&issue.path, dry_run, &mut would_remove, &mut removed)?
        {
            continue;
        }
        retained.push(issue.clone());
    }
    would_remove.sort();
    would_remove.dedup();
    removed.sort();
    removed.dedup();
    Ok(ModelPruneReport {
        cache_dir: report.cache_dir,
        dry_run,
        would_remove,
        removed,
        retained,
    })
}
