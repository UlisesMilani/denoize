//! Authenticated, length-delimited bundles for closed-network model installs.

#[cfg(test)]
use super::install_catalog_model_from_verified_bundle_for_test;
use super::{
    catalog, install_catalog_model_from_bundle, open_existing_regular_file, path_for_catalog_model,
    remove_catalog_model, validate_model_storage_path, verify_catalog_model, CatalogBundleFile,
    CatalogModel, ModelCatalog,
};
use crate::{AtomicOutput, CommitMode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use url::Url;

const BUNDLE_MAGIC: &[u8] = b"denoize-model-bundle-v1\n";
const BUNDLE_SCHEMA: &str = "denoize-model-bundle-v1";
const BUNDLE_VERSION: u32 = 1;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_CATALOG_BYTES: u64 = 1024 * 1024;
const MAX_SIGNATURE_BYTES: u64 = 16 * 1024;
const MAX_TRUST_ROOT_BYTES: u64 = 64 * 1024;
const MAX_MODELS: usize = 256;
const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 1024 * 1024;

/// Verified identity and contents of one closed-network model bundle.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OfflineBundleInfo {
    pub format_version: u32,
    pub bundle_sha256: String,
    pub size_bytes: u64,
    pub catalog_sequence: u64,
    pub catalog_sha256: String,
    pub catalog_signing_key_id: String,
    pub catalog_issued_at_unix_seconds: Option<u64>,
    pub catalog_expires_at_unix_seconds: Option<u64>,
    pub trust_root_version: u64,
    pub trust_root_sha256: String,
    pub models: Vec<OfflineBundleModelInfo>,
}

/// Verified model and notice files carried by an offline bundle.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OfflineBundleModelInfo {
    pub name: String,
    pub backend: String,
    pub artifact_filename: String,
    pub artifact_sha256: String,
    pub artifact_size_bytes: u64,
    pub license_filename: String,
    pub license_sha256: String,
    pub license_size_bytes: u64,
    pub provenance_filename: String,
    pub provenance_sha256: String,
    pub provenance_size_bytes: u64,
}

/// Result of importing every model from an authenticated bundle.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OfflineBundleImportReport {
    pub bundle: OfflineBundleInfo,
    pub installed: Vec<PathBuf>,
    pub already_present: Vec<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BundleManifest {
    schema: String,
    version: u32,
    catalog: BundleFileRecord,
    catalog_signature: BundleFileRecord,
    trust_root: BundleFileRecord,
    models: Vec<BundleModelRecord>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BundleFileRecord {
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BundleNamedFileRecord {
    filename: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BundleModelRecord {
    name: String,
    artifact: BundleNamedFileRecord,
    license: BundleNamedFileRecord,
    provenance: BundleNamedFileRecord,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceProvenanceDocument {
    schema: String,
    model_name: String,
    upstream_repository: String,
    upstream_revision: String,
    artifact: SourceArtifactDocument,
    license: SourceLicenseDocument,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceArtifactDocument {
    url: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceLicenseDocument {
    spdx: String,
    url: String,
    sha256: String,
    size_bytes: u64,
}

struct OpenBuildModel {
    record: BundleModelRecord,
    artifact: File,
    license: File,
    provenance: File,
    artifact_path: PathBuf,
    license_path: PathBuf,
    provenance_path: PathBuf,
}

struct PreparedModel {
    name: String,
    artifact: tempfile::NamedTempFile,
    existed: bool,
}

struct PreparedBundle {
    info: OfflineBundleInfo,
    catalog: ModelCatalog,
    catalog_bytes: Vec<u8>,
    signature: Vec<u8>,
    models: Vec<PreparedModel>,
}

/// Build a deterministic bundle from a signed catalog and a component tree.
///
/// `components_dir` must contain, for every catalog package, the exact paths
/// `<MODEL>/<artifact>`, `<MODEL>/<license>`, and `<MODEL>/<provenance>`.
pub fn build_offline_bundle(
    output: impl AsRef<Path>,
    catalog_path: impl AsRef<Path>,
    signature_path: impl AsRef<Path>,
    trust_root_path: impl AsRef<Path>,
    components_dir: impl AsRef<Path>,
) -> Result<OfflineBundleInfo, String> {
    build_offline_bundle_with_verifier(
        output.as_ref(),
        catalog_path.as_ref(),
        signature_path.as_ref(),
        trust_root_path.as_ref(),
        components_dir.as_ref(),
        catalog::verify_bundle_catalog,
    )
}

/// Authenticate every byte in a bundle without changing catalog or cache
/// state. The input must be a seekable regular file.
pub fn inspect_offline_bundle(path: impl AsRef<Path>) -> Result<OfflineBundleInfo, String> {
    Ok(
        prepare_offline_bundle_with_verifier(path.as_ref(), false, catalog::verify_bundle_catalog)?
            .info,
    )
}

/// Authenticate a bundle, activate its signed catalog, and atomically install
/// each missing model. All bundle bytes are verified before persistent model or
/// catalog state changes. A storage failure rolls back models created by this
/// invocation; the monotonic catalog rollback floor may remain advanced and a
/// retry is safe.
pub fn import_offline_bundle(path: impl AsRef<Path>) -> Result<OfflineBundleImportReport, String> {
    let prepared =
        prepare_offline_bundle_with_verifier(path.as_ref(), true, catalog::verify_bundle_catalog)?;
    commit_prepared_bundle(prepared)
}

/// Import only if the fully authenticated bundle still has `expected_sha256`.
/// This binds a prior user-visible inspection to the bytes committed by a later
/// confirmation step without trusting the pathname to remain unchanged.
pub fn import_offline_bundle_if_sha256(
    path: impl AsRef<Path>,
    expected_sha256: &str,
) -> Result<OfflineBundleImportReport, String> {
    if !valid_sha256(expected_sha256) {
        return Err(
            "expected offline bundle SHA-256 must be 64 lowercase hexadecimal characters".into(),
        );
    }
    let prepared =
        prepare_offline_bundle_with_verifier(path.as_ref(), true, catalog::verify_bundle_catalog)?;
    if prepared.info.bundle_sha256 != expected_sha256 {
        return Err(format!(
            "offline bundle changed after inspection: expected {expected_sha256}, got {}",
            prepared.info.bundle_sha256
        ));
    }
    commit_prepared_bundle(prepared)
}

fn commit_prepared_bundle(prepared: PreparedBundle) -> Result<OfflineBundleImportReport, String> {
    let active = catalog::activate_bundle_catalog(&prepared.catalog_bytes, &prepared.signature)?;
    if active.sequence() != prepared.catalog.sequence()
        || active.sha256() != prepared.catalog.sha256()
        || active.signing_key_id() != prepared.catalog.signing_key_id()
    {
        return Err("active model catalog changed while importing the offline bundle".into());
    }

    let mut installed = Vec::new();
    let mut already_present = Vec::new();
    let mut newly_installed = Vec::<CatalogModel>::new();
    let result = (|| {
        for staged in &prepared.models {
            let model = active.find(&staged.name).ok_or_else(|| {
                format!(
                    "offline bundle model disappeared from active catalog: {}",
                    staged.name
                )
            })?;
            if staged.existed {
                already_present.push(verify_catalog_model(model)?);
                continue;
            }
            let destination = install_catalog_model_from_bundle(
                model,
                staged.artifact.path(),
                &prepared.info.bundle_sha256,
            )?;
            newly_installed.push(model.clone());
            installed.push(destination);
        }
        Ok(())
    })();
    if let Err(error) = result {
        for model in newly_installed.iter().rev() {
            let _ = remove_catalog_model(model);
        }
        return Err(error);
    }
    Ok(OfflineBundleImportReport {
        bundle: prepared.info,
        installed,
        already_present,
    })
}

fn build_offline_bundle_with_verifier<F>(
    output: &Path,
    catalog_path: &Path,
    signature_path: &Path,
    trust_root_path: &Path,
    components_dir: &Path,
    mut verify_catalog: F,
) -> Result<OfflineBundleInfo, String>
where
    F: FnMut(&[u8], &[u8]) -> Result<catalog::VerifiedBundleCatalog, String>,
{
    let catalog_bytes = read_bounded_regular(catalog_path, MAX_CATALOG_BYTES, "model catalog")?;
    let signature = read_bounded_regular(
        signature_path,
        MAX_SIGNATURE_BYTES,
        "model catalog signature",
    )?;
    let trust_root =
        read_bounded_regular(trust_root_path, MAX_TRUST_ROOT_BYTES, "model trust root")?;
    let verified = verify_catalog(&catalog_bytes, &signature)?;
    let trust_root_record = file_record(&trust_root);
    if trust_root_record.sha256 != verified.trust_root_sha256 {
        return Err(format!(
            "bundle trust root does not match active trust root {}",
            verified.trust_root_sha256
        ));
    }

    let mut open_models = Vec::with_capacity(verified.catalog.models().len());
    for model in verified.catalog.models() {
        let metadata = require_bundle_metadata(model)?;
        let directory = components_dir.join(model.name());
        let artifact_path = directory.join(model.filename());
        let license_path = directory.join(metadata.license().filename());
        let provenance_path = directory.join(metadata.provenance().filename());
        let mut artifact = open_required_regular(&artifact_path, "bundle model artifact")?;
        let mut license = open_required_regular(&license_path, "bundle model license")?;
        let mut provenance = open_required_regular(&provenance_path, "bundle model provenance")?;
        let artifact_record = named_record_for_open_file(
            &mut artifact,
            &artifact_path,
            model.filename(),
            model.size_bytes(),
            model.sha256(),
        )?;
        let license_record =
            named_record_for_catalog_file(&mut license, &license_path, metadata.license())?;
        let provenance_record = named_record_for_catalog_file(
            &mut provenance,
            &provenance_path,
            metadata.provenance(),
        )?;
        let provenance_bytes = read_open_component_bytes(
            &mut provenance,
            &provenance_path,
            metadata.provenance().size_bytes(),
        )?;
        validate_source_provenance(model, metadata, &provenance_bytes)?;
        open_models.push(OpenBuildModel {
            record: BundleModelRecord {
                name: model.name().to_string(),
                artifact: artifact_record,
                license: license_record,
                provenance: provenance_record,
            },
            artifact,
            license,
            provenance,
            artifact_path,
            license_path,
            provenance_path,
        });
    }

    let manifest = BundleManifest {
        schema: BUNDLE_SCHEMA.to_string(),
        version: BUNDLE_VERSION,
        catalog: file_record(&catalog_bytes),
        catalog_signature: file_record(&signature),
        trust_root: trust_root_record,
        models: open_models
            .iter()
            .map(|model| model.record.clone())
            .collect(),
    };
    validate_manifest_shape(&manifest)?;
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to encode offline bundle manifest: {error}"))?;
    manifest_bytes.push(b'\n');
    if manifest_bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err("offline bundle manifest exceeds the 1 MiB limit".into());
    }

    let mut staged = AtomicOutput::new(output)?;
    staged
        .file_mut()
        .write_all(BUNDLE_MAGIC)
        .and_then(|_| {
            staged
                .file_mut()
                .write_all(&(manifest_bytes.len() as u64).to_be_bytes())
        })
        .and_then(|_| staged.file_mut().write_all(&manifest_bytes))
        .and_then(|_| staged.file_mut().write_all(&catalog_bytes))
        .and_then(|_| staged.file_mut().write_all(&signature))
        .and_then(|_| staged.file_mut().write_all(&trust_root))
        .map_err(|error| {
            format!(
                "failed to write offline bundle {}: {error}",
                output.display()
            )
        })?;
    for model in &mut open_models {
        copy_open_file(
            &mut model.artifact,
            &model.artifact_path,
            staged.file_mut(),
            model.record.artifact.size_bytes,
            &model.record.artifact.sha256,
        )?;
        copy_open_file(
            &mut model.license,
            &model.license_path,
            staged.file_mut(),
            model.record.license.size_bytes,
            &model.record.license.sha256,
        )?;
        copy_open_file(
            &mut model.provenance,
            &model.provenance_path,
            staged.file_mut(),
            model.record.provenance.size_bytes,
            &model.record.provenance.sha256,
        )?;
    }
    staged.file_mut().flush().map_err(|error| {
        format!(
            "failed to flush offline bundle {}: {error}",
            output.display()
        )
    })?;
    let staged_size = staged
        .file_mut()
        .metadata()
        .map_err(|error| format!("failed to inspect staged offline bundle: {error}"))?
        .len();
    let info = prepare_open_offline_bundle_with_verifier(
        staged.file_mut(),
        output,
        staged_size,
        false,
        &mut verify_catalog,
    )?
    .info;
    staged.commit(CommitMode::Replace)?;
    Ok(info)
}

#[cfg(test)]
pub(super) fn build_offline_bundle_for_test(
    output: &Path,
    catalog_path: &Path,
    signature_path: &Path,
    trust_root_path: &Path,
    components_dir: &Path,
) -> Result<OfflineBundleInfo, String> {
    let trust_root_sha256 = sha256_bytes(
        &std::fs::read(trust_root_path)
            .map_err(|error| format!("failed to read {}: {error}", trust_root_path.display()))?,
    );
    build_offline_bundle_with_verifier(
        output,
        catalog_path,
        signature_path,
        trust_root_path,
        components_dir,
        move |catalog_bytes, _signature| {
            Ok(catalog::VerifiedBundleCatalog {
                catalog: catalog::parse_catalog(
                    catalog_bytes,
                    super::CatalogOrigin::Signed {
                        source: "local-import".into(),
                    },
                )?,
                trust_root_version: 1,
                trust_root_sha256: trust_root_sha256.clone(),
            })
        },
    )
}

#[cfg(test)]
pub(super) fn inspect_offline_bundle_for_test(
    path: &Path,
    trust_root_sha256: &str,
) -> Result<OfflineBundleInfo, String> {
    prepare_offline_bundle_with_verifier(path, false, |catalog_bytes, _signature| {
        Ok(catalog::VerifiedBundleCatalog {
            catalog: catalog::parse_catalog(
                catalog_bytes,
                super::CatalogOrigin::Signed {
                    source: "local-import".into(),
                },
            )?,
            trust_root_version: 1,
            trust_root_sha256: trust_root_sha256.to_string(),
        })
    })
    .map(|prepared| prepared.info)
}

#[cfg(test)]
pub(super) fn import_offline_bundle_for_test(
    path: &Path,
    trust_root_sha256: &str,
) -> Result<OfflineBundleImportReport, String> {
    let prepared =
        prepare_offline_bundle_with_verifier(path, true, |catalog_bytes, _signature| {
            Ok(catalog::VerifiedBundleCatalog {
                catalog: catalog::parse_catalog(
                    catalog_bytes,
                    super::CatalogOrigin::Signed {
                        source: "local-import".into(),
                    },
                )?,
                trust_root_version: 1,
                trust_root_sha256: trust_root_sha256.to_string(),
            })
        })?;
    commit_prepared_bundle_for_test(prepared)
}

#[cfg(test)]
pub(super) fn import_offline_bundle_for_test_if_sha256(
    path: &Path,
    trust_root_sha256: &str,
    expected_sha256: &str,
) -> Result<OfflineBundleImportReport, String> {
    let prepared =
        prepare_offline_bundle_with_verifier(path, true, |catalog_bytes, _signature| {
            Ok(catalog::VerifiedBundleCatalog {
                catalog: catalog::parse_catalog(
                    catalog_bytes,
                    super::CatalogOrigin::Signed {
                        source: "local-import".into(),
                    },
                )?,
                trust_root_version: 1,
                trust_root_sha256: trust_root_sha256.to_string(),
            })
        })?;
    if prepared.info.bundle_sha256 != expected_sha256 {
        return Err(format!(
            "offline bundle changed after inspection: expected {expected_sha256}, got {}",
            prepared.info.bundle_sha256
        ));
    }
    commit_prepared_bundle_for_test(prepared)
}

#[cfg(test)]
fn commit_prepared_bundle_for_test(
    prepared: PreparedBundle,
) -> Result<OfflineBundleImportReport, String> {
    let mut installed = Vec::new();
    let mut already_present = Vec::new();
    let mut newly_installed = Vec::<CatalogModel>::new();
    let result = (|| {
        for staged in &prepared.models {
            let model = prepared.catalog.find(&staged.name).unwrap();
            if staged.existed {
                already_present.push(verify_catalog_model(model)?);
            } else {
                installed.push(install_catalog_model_from_verified_bundle_for_test(
                    model,
                    staged.artifact.path(),
                    &prepared.info.bundle_sha256,
                )?);
                newly_installed.push(model.clone());
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        for model in newly_installed.iter().rev() {
            let _ = remove_catalog_model(model);
        }
        return Err(error);
    }
    Ok(OfflineBundleImportReport {
        bundle: prepared.info,
        installed,
        already_present,
    })
}

fn prepare_offline_bundle_with_verifier<F>(
    path: &Path,
    stage_models: bool,
    mut verify_catalog: F,
) -> Result<PreparedBundle, String>
where
    F: FnMut(&[u8], &[u8]) -> Result<catalog::VerifiedBundleCatalog, String>,
{
    let mut input = open_required_regular(path, "offline model bundle")?;
    let size_bytes = input
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    prepare_open_offline_bundle_with_verifier(
        &mut input,
        path,
        size_bytes,
        stage_models,
        &mut verify_catalog,
    )
}

fn prepare_open_offline_bundle_with_verifier<F>(
    mut input: &mut File,
    path: &Path,
    size_bytes: u64,
    stage_models: bool,
    verify_catalog: &mut F,
) -> Result<PreparedBundle, String>
where
    F: FnMut(&[u8], &[u8]) -> Result<catalog::VerifiedBundleCatalog, String>,
{
    let bundle_sha256 = hash_open_file(&mut input, path, size_bytes)?;

    let mut magic = vec![0_u8; BUNDLE_MAGIC.len()];
    read_exact_described(&mut input, &mut magic, "offline bundle magic")?;
    if magic != BUNDLE_MAGIC {
        return Err("unsupported offline model bundle magic".into());
    }
    let manifest_size = read_u64(&mut input, "offline bundle manifest length")?;
    if manifest_size == 0 || manifest_size > MAX_MANIFEST_BYTES {
        return Err("offline bundle manifest size must be between 1 byte and 1 MiB".into());
    }
    let mut manifest_bytes = vec![
        0_u8;
        usize::try_from(manifest_size).map_err(|_| {
            "offline bundle manifest does not fit in memory".to_string()
        })?
    ];
    read_exact_described(&mut input, &mut manifest_bytes, "offline bundle manifest")?;
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("invalid offline bundle manifest JSON: {error}"))?;
    validate_manifest_shape(&manifest)?;
    let expected_size = expected_bundle_size(manifest_size, &manifest)?;
    if expected_size != size_bytes {
        return Err(format!(
            "offline bundle length mismatch: manifest describes {expected_size} bytes, file has {size_bytes}"
        ));
    }

    let catalog_bytes = read_record_bytes(
        &mut input,
        &manifest.catalog,
        MAX_CATALOG_BYTES,
        "model catalog",
    )?;
    let signature = read_record_bytes(
        &mut input,
        &manifest.catalog_signature,
        MAX_SIGNATURE_BYTES,
        "model catalog signature",
    )?;
    let _trust_root = read_record_bytes(
        &mut input,
        &manifest.trust_root,
        MAX_TRUST_ROOT_BYTES,
        "model trust root",
    )?;
    let verified = verify_catalog(&catalog_bytes, &signature)?;
    if manifest.catalog.sha256 != verified.catalog.sha256() {
        return Err("offline bundle catalog digest does not match its signed contents".into());
    }
    if manifest.trust_root.sha256 != verified.trust_root_sha256 {
        return Err(format!(
            "offline bundle trust root does not match active trust root {}",
            verified.trust_root_sha256
        ));
    }
    validate_manifest_catalog(&manifest, &verified.catalog)?;

    let mut staged_models = Vec::with_capacity(manifest.models.len());
    let mut model_infos = Vec::with_capacity(manifest.models.len());
    for (record, model) in manifest.models.iter().zip(verified.catalog.models()) {
        let existed = if stage_models {
            let destination = path_for_catalog_model(model)?;
            preflight_existing_model(model, &destination)?
        } else {
            false
        };
        if stage_models {
            let mut artifact = tempfile::NamedTempFile::new()
                .map_err(|error| format!("failed to create offline model staging file: {error}"))?;
            read_record_to_writer(
                &mut input,
                &record.artifact,
                MAX_MODEL_BYTES,
                "model artifact",
                artifact.as_file_mut(),
            )?;
            artifact.as_file_mut().flush().map_err(|error| {
                format!("failed to flush staged model {}: {error}", model.name())
            })?;
            staged_models.push(PreparedModel {
                name: model.name().to_string(),
                artifact,
                existed,
            });
        } else {
            let mut sink = std::io::sink();
            read_record_to_writer(
                &mut input,
                &record.artifact,
                MAX_MODEL_BYTES,
                "model artifact",
                &mut sink,
            )?;
        }
        let license = read_named_record_bytes(
            &mut input,
            &record.license,
            MAX_METADATA_BYTES,
            "model license",
        )?;
        let provenance = read_named_record_bytes(
            &mut input,
            &record.provenance,
            MAX_METADATA_BYTES,
            "model provenance",
        )?;
        validate_source_provenance(model, require_bundle_metadata(model)?, &provenance)?;
        // Keep the authenticated non-executable bytes in the verification
        // path even though installation provenance records their signed
        // digests rather than copying them into the executable model cache.
        drop((license, provenance));
        model_infos.push(model_info(model)?);
    }
    let mut trailing = [0_u8; 1];
    if input
        .read(&mut trailing)
        .map_err(|error| format!("failed to finish reading {}: {error}", path.display()))?
        != 0
    {
        return Err("offline bundle contains trailing bytes".into());
    }
    let final_sha256 = hash_open_file(&mut input, path, size_bytes)?;
    if final_sha256 != bundle_sha256 {
        return Err(format!(
            "offline bundle changed while reading {}",
            path.display()
        ));
    }
    Ok(PreparedBundle {
        info: OfflineBundleInfo {
            format_version: BUNDLE_VERSION,
            bundle_sha256,
            size_bytes,
            catalog_sequence: verified.catalog.sequence(),
            catalog_sha256: verified.catalog.sha256().to_string(),
            catalog_signing_key_id: verified.catalog.signing_key_id().to_string(),
            catalog_issued_at_unix_seconds: verified.catalog.issued_at_unix_seconds(),
            catalog_expires_at_unix_seconds: verified.catalog.expires_at_unix_seconds(),
            trust_root_version: verified.trust_root_version,
            trust_root_sha256: verified.trust_root_sha256,
            models: model_infos,
        },
        catalog: verified.catalog,
        catalog_bytes,
        signature,
        models: staged_models,
    })
}

fn validate_manifest_shape(manifest: &BundleManifest) -> Result<(), String> {
    if manifest.schema != BUNDLE_SCHEMA || manifest.version != BUNDLE_VERSION {
        return Err("unsupported offline model bundle manifest version".into());
    }
    if manifest.models.is_empty() || manifest.models.len() > MAX_MODELS {
        return Err(format!(
            "offline bundle must contain between 1 and {MAX_MODELS} models"
        ));
    }
    validate_record(&manifest.catalog, MAX_CATALOG_BYTES, "model catalog")?;
    validate_record(
        &manifest.catalog_signature,
        MAX_SIGNATURE_BYTES,
        "model catalog signature",
    )?;
    validate_record(
        &manifest.trust_root,
        MAX_TRUST_ROOT_BYTES,
        "model trust root",
    )?;
    let mut names = HashSet::with_capacity(manifest.models.len());
    for model in &manifest.models {
        if !names.insert(model.name.as_str()) {
            return Err(format!("duplicate offline bundle model: {}", model.name));
        }
        validate_named_record(&model.artifact, MAX_MODEL_BYTES, "model artifact")?;
        validate_named_record(&model.license, MAX_METADATA_BYTES, "model license")?;
        validate_named_record(&model.provenance, MAX_METADATA_BYTES, "model provenance")?;
    }
    Ok(())
}

fn validate_manifest_catalog(
    manifest: &BundleManifest,
    catalog: &ModelCatalog,
) -> Result<(), String> {
    if manifest.models.len() != catalog.models().len() {
        return Err("offline bundle model count does not match the signed catalog".into());
    }
    for (record, model) in manifest.models.iter().zip(catalog.models()) {
        let metadata = require_bundle_metadata(model)?;
        if record.name != model.name()
            || !named_record_matches(
                &record.artifact,
                model.filename(),
                model.size_bytes(),
                model.sha256(),
            )
            || !named_record_matches_catalog_file(&record.license, metadata.license())
            || !named_record_matches_catalog_file(&record.provenance, metadata.provenance())
        {
            return Err(format!(
                "offline bundle entry does not match signed catalog model {}",
                model.name()
            ));
        }
    }
    Ok(())
}

fn require_bundle_metadata(model: &CatalogModel) -> Result<&super::CatalogBundleMetadata, String> {
    model.offline_bundle().ok_or_else(|| {
        format!(
            "signed catalog model {} has no offline bundle license/provenance metadata",
            model.name()
        )
    })
}

fn preflight_existing_model(model: &CatalogModel, destination: &Path) -> Result<bool, String> {
    validate_model_storage_path(destination)?;
    let Some(mut existing) = open_existing_regular_file(destination, "installed model")? else {
        return Ok(false);
    };
    let actual = hash_open_file(&mut existing, destination, model.size_bytes())?;
    if actual != model.sha256() {
        return Err(format!(
            "refusing to replace an existing mismatched model during bundle import: {}",
            destination.display()
        ));
    }
    Ok(true)
}

fn model_info(model: &CatalogModel) -> Result<OfflineBundleModelInfo, String> {
    let metadata = require_bundle_metadata(model)?;
    Ok(OfflineBundleModelInfo {
        name: model.name().to_string(),
        backend: model.backend().to_string(),
        artifact_filename: model.filename().to_string(),
        artifact_sha256: model.sha256().to_string(),
        artifact_size_bytes: model.size_bytes(),
        license_filename: metadata.license().filename().to_string(),
        license_sha256: metadata.license().sha256().to_string(),
        license_size_bytes: metadata.license().size_bytes(),
        provenance_filename: metadata.provenance().filename().to_string(),
        provenance_sha256: metadata.provenance().sha256().to_string(),
        provenance_size_bytes: metadata.provenance().size_bytes(),
    })
}

fn validate_source_provenance(
    model: &CatalogModel,
    metadata: &super::CatalogBundleMetadata,
    bytes: &[u8],
) -> Result<(), String> {
    let document: SourceProvenanceDocument = serde_json::from_slice(bytes).map_err(|error| {
        format!(
            "invalid offline provenance JSON for model {}: {error}",
            model.name()
        )
    })?;
    if document.schema != "denoize-model-source-provenance-v1"
        || document.model_name != model.name()
        || document.upstream_revision != model.revision()
        || document.artifact.url != model.url()
        || document.artifact.sha256 != model.sha256()
        || document.artifact.size_bytes != model.size_bytes()
        || document.license.spdx != model.license()
        || document.license.sha256 != metadata.license().sha256()
        || document.license.size_bytes != metadata.license().size_bytes()
        || !valid_public_https_url(&document.upstream_repository)
        || !valid_public_https_url(&document.license.url)
    {
        return Err(format!(
            "offline provenance does not match signed catalog model {}",
            model.name()
        ));
    }
    Ok(())
}

fn valid_public_https_url(value: &str) -> bool {
    Url::parse(value).is_ok_and(|url| {
        url.scheme() == "https"
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
    })
}

fn expected_bundle_size(manifest_size: u64, manifest: &BundleManifest) -> Result<u64, String> {
    let mut total = (BUNDLE_MAGIC.len() as u64)
        .checked_add(8)
        .and_then(|value| value.checked_add(manifest_size))
        .ok_or_else(|| "offline bundle size overflow".to_string())?;
    for size in [
        manifest.catalog.size_bytes,
        manifest.catalog_signature.size_bytes,
        manifest.trust_root.size_bytes,
    ] {
        total = total
            .checked_add(size)
            .ok_or_else(|| "offline bundle size overflow".to_string())?;
    }
    for model in &manifest.models {
        for size in [
            model.artifact.size_bytes,
            model.license.size_bytes,
            model.provenance.size_bytes,
        ] {
            total = total
                .checked_add(size)
                .ok_or_else(|| "offline bundle size overflow".to_string())?;
        }
    }
    Ok(total)
}

fn validate_named_record(
    record: &BundleNamedFileRecord,
    maximum: u64,
    description: &str,
) -> Result<(), String> {
    if record.filename.is_empty()
        || record.filename.len() > 128
        || !record.filename.as_bytes()[0].is_ascii_alphanumeric()
        || !record
            .filename
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(format!(
            "offline bundle {description} has an invalid filename"
        ));
    }
    validate_record(
        &BundleFileRecord {
            size_bytes: record.size_bytes,
            sha256: record.sha256.clone(),
        },
        maximum,
        description,
    )
}

fn validate_record(
    record: &BundleFileRecord,
    maximum: u64,
    description: &str,
) -> Result<(), String> {
    if record.size_bytes == 0 || record.size_bytes > maximum {
        return Err(format!(
            "offline bundle {description} size must be between 1 and {maximum} bytes"
        ));
    }
    if !valid_sha256(&record.sha256) {
        return Err(format!(
            "offline bundle {description} has an invalid lowercase SHA-256"
        ));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn named_record_matches(
    record: &BundleNamedFileRecord,
    filename: &str,
    size_bytes: u64,
    sha256: &str,
) -> bool {
    record.filename == filename && record.size_bytes == size_bytes && record.sha256 == sha256
}

fn named_record_matches_catalog_file(
    record: &BundleNamedFileRecord,
    file: &CatalogBundleFile,
) -> bool {
    named_record_matches(record, file.filename(), file.size_bytes(), file.sha256())
}

fn file_record(bytes: &[u8]) -> BundleFileRecord {
    BundleFileRecord {
        size_bytes: bytes.len() as u64,
        sha256: sha256_bytes(bytes),
    }
}

fn named_record_for_catalog_file(
    file: &mut File,
    path: &Path,
    expected: &CatalogBundleFile,
) -> Result<BundleNamedFileRecord, String> {
    named_record_for_open_file(
        file,
        path,
        expected.filename(),
        expected.size_bytes(),
        expected.sha256(),
    )
}

fn named_record_for_open_file(
    file: &mut File,
    path: &Path,
    filename: &str,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<BundleNamedFileRecord, String> {
    let actual = hash_open_file(file, path, expected_size)?;
    if actual != expected_sha256 {
        return Err(format!(
            "offline bundle component checksum mismatch for {}: expected {expected_sha256}, got {actual}",
            path.display()
        ));
    }
    Ok(BundleNamedFileRecord {
        filename: filename.to_string(),
        size_bytes: expected_size,
        sha256: expected_sha256.to_string(),
    })
}

fn read_record_bytes(
    input: &mut File,
    record: &BundleFileRecord,
    maximum: u64,
    description: &str,
) -> Result<Vec<u8>, String> {
    validate_record(record, maximum, description)?;
    let mut bytes = vec![
        0_u8;
        usize::try_from(record.size_bytes).map_err(|_| format!(
            "offline bundle {description} does not fit in memory"
        ))?
    ];
    read_exact_described(input, &mut bytes, description)?;
    let actual = sha256_bytes(&bytes);
    if actual != record.sha256 {
        return Err(format!(
            "offline bundle {description} checksum mismatch: expected {}, got {actual}",
            record.sha256
        ));
    }
    Ok(bytes)
}

fn read_named_record_bytes(
    input: &mut File,
    record: &BundleNamedFileRecord,
    maximum: u64,
    description: &str,
) -> Result<Vec<u8>, String> {
    validate_named_record(record, maximum, description)?;
    read_record_bytes(
        input,
        &BundleFileRecord {
            size_bytes: record.size_bytes,
            sha256: record.sha256.clone(),
        },
        maximum,
        description,
    )
}

fn read_record_to_writer(
    input: &mut File,
    record: &BundleNamedFileRecord,
    maximum: u64,
    description: &str,
    output: &mut dyn Write,
) -> Result<(), String> {
    validate_named_record(record, maximum, description)?;
    let mut remaining = record.size_bytes;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    while remaining > 0 {
        let count = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "offline bundle chunk size overflow".to_string())?;
        read_exact_described(input, &mut buffer[..count], description)?;
        digest.update(&buffer[..count]);
        output
            .write_all(&buffer[..count])
            .map_err(|error| format!("failed to stage offline {description}: {error}"))?;
        remaining -= count as u64;
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != record.sha256 {
        return Err(format!(
            "offline bundle {description} checksum mismatch: expected {}, got {actual}",
            record.sha256
        ));
    }
    Ok(())
}

fn copy_open_file(
    input: &mut File,
    path: &Path,
    output: &mut File,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    input
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind {}: {error}", path.display()))?;
    let mut remaining = expected_size;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    while remaining > 0 {
        let count = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "offline bundle component size overflow".to_string())?;
        read_exact_described(input, &mut buffer[..count], "offline bundle component")?;
        digest.update(&buffer[..count]);
        output
            .write_all(&buffer[..count])
            .map_err(|error| format!("failed to write offline bundle: {error}"))?;
        remaining -= count as u64;
    }
    let after = input
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    let actual = format!("{:x}", digest.finalize());
    if after != expected_size || actual != expected_sha256 {
        return Err(format!(
            "offline bundle component changed while copying {}",
            path.display()
        ));
    }
    Ok(())
}

fn hash_open_file(file: &mut File, path: &Path, expected_size: u64) -> Result<String, String> {
    let before = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if before != expected_size {
        return Err(format!(
            "offline bundle component size mismatch for {}: expected {expected_size}, got {before}",
            path.display()
        ));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut read = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        read = read
            .checked_add(count as u64)
            .ok_or_else(|| "offline bundle component size overflow".to_string())?;
        if read > expected_size {
            return Err(format!(
                "offline bundle component grew while reading {}",
                path.display()
            ));
        }
        digest.update(&buffer[..count]);
    }
    let after = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if read != expected_size || after != expected_size {
        return Err(format!(
            "offline bundle component changed while reading {}",
            path.display()
        ));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind {}: {error}", path.display()))?;
    Ok(format!("{:x}", digest.finalize()))
}

fn read_open_component_bytes(
    file: &mut File,
    path: &Path,
    expected_size: u64,
) -> Result<Vec<u8>, String> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(usize::try_from(expected_size).map_err(|_| {
        format!(
            "offline component does not fit in memory: {}",
            path.display()
        )
    })?);
    file.take(expected_size + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if bytes.len() as u64 != expected_size {
        return Err(format!(
            "offline component changed while reading {}",
            path.display()
        ));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind {}: {error}", path.display()))?;
    Ok(bytes)
}

fn open_required_regular(path: &Path, description: &str) -> Result<File, String> {
    open_existing_regular_file(path, description)?
        .ok_or_else(|| format!("failed to open {}: file not found", path.display()))
}

fn read_bounded_regular(path: &Path, maximum: u64, description: &str) -> Result<Vec<u8>, String> {
    let mut file = open_required_regular(path, description)?;
    let length = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length == 0 || length > maximum {
        return Err(format!(
            "{description} size must be between 1 and {maximum} bytes"
        ));
    }
    let mut bytes = vec![
        0_u8;
        usize::try_from(length)
            .map_err(|_| format!("{description} does not fit in memory"))?
    ];
    read_exact_described(&mut file, &mut bytes, description)?;
    let after = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if after != length {
        return Err(format!(
            "{description} changed while reading {}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn read_u64(input: &mut File, description: &str) -> Result<u64, String> {
    let mut bytes = [0_u8; 8];
    read_exact_described(input, &mut bytes, description)?;
    Ok(u64::from_be_bytes(bytes))
}

fn read_exact_described(
    input: &mut File,
    buffer: &mut [u8],
    description: &str,
) -> Result<(), String> {
    input
        .read_exact(buffer)
        .map_err(|error| format!("failed to read {description}: {error}"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_bundle_evidence_matches_the_catalog() {
        let catalog = catalog::parse_catalog(
            include_bytes!("../../models/catalog-v1.json"),
            crate::models::CatalogOrigin::Embedded,
        )
        .unwrap();
        for (name, license, provenance) in [
            (
                "gtcrn-dns3",
                include_bytes!("../../models/licenses/gtcrn-dns3-MIT.txt").as_slice(),
                include_bytes!("../../models/provenance/gtcrn-dns3.json").as_slice(),
            ),
            (
                "dpdfnet2-48khz-hr",
                include_bytes!("../../models/licenses/dpdfnet2-48khz-hr-Apache-2.0.txt").as_slice(),
                include_bytes!("../../models/provenance/dpdfnet2-48khz-hr.json").as_slice(),
            ),
        ] {
            let model = catalog.find(name).unwrap();
            let metadata = require_bundle_metadata(model).unwrap();
            assert_eq!(license.len() as u64, metadata.license().size_bytes());
            assert_eq!(sha256_bytes(license), metadata.license().sha256());
            assert_eq!(provenance.len() as u64, metadata.provenance().size_bytes());
            assert_eq!(sha256_bytes(provenance), metadata.provenance().sha256());
            validate_source_provenance(model, metadata, provenance).unwrap();
        }
    }
}
