//! Signed external-model catalog, verified local cache, and installation provenance.

mod bundle;
mod catalog;
mod maintenance;

pub use bundle::{
    build_offline_bundle, import_offline_bundle, import_offline_bundle_if_sha256,
    inspect_offline_bundle, OfflineBundleImportReport, OfflineBundleInfo, OfflineBundleModelInfo,
};
#[cfg(any(feature = "gtcrn", feature = "dpdfnet"))]
pub(crate) use catalog::active_catalog_read_only;
pub use catalog::{
    active_catalog, catalog_status, embedded_catalog, import_catalog, import_trust_root,
    recover_embedded_trust_root, reset_trust_time_floor, trust_root_status, update_catalog,
    CatalogBundleFile, CatalogBundleMetadata, CatalogModel, CatalogOrigin, CatalogStatus,
    ModelCatalog, TrustRootOrigin, TrustRootStatus,
};
pub use maintenance::{
    doctor_model_cache, doctor_model_cache_for_catalog, prune_model_cache,
    repair_catalog_model_with_options, repair_catalog_model_with_options_and_progress,
    ModelCacheIssue, ModelCacheIssueKind, ModelCacheModel, ModelCacheModelStatus, ModelCacheReport,
    ModelPruneReport, ModelRepairOutcome,
};

use self::catalog::CatalogIdentity;
use crate::{AtomicOutput, CommitMode};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use fs2::FileExt;
use percent_encoding::percent_decode_str;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use url::Url;

const PARTIAL_METADATA_VERSION: u32 = 1;
const MAX_PARTIAL_METADATA_BYTES: u64 = 64 * 1024;
const MODEL_PROVENANCE_VERSION: u32 = 1;
const MAX_MODEL_PROVENANCE_BYTES: u64 = 64 * 1024;
const MAX_JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const USER_AGENT: &str = concat!("denoize-model-manager/", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Copy, Debug)]
pub struct ModelInfo {
    pub name: &'static str,
    pub backend: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub revision: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    pub license: &'static str,
    pub sample_rate: u32,
}

/// How model downloads reach the network.
#[derive(Clone, Default, Eq, PartialEq)]
pub enum ModelProxy {
    /// Honor `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, and `NO_PROXY`.
    #[default]
    Environment,
    /// Connect directly and ignore proxy environment variables.
    Disabled,
    /// Use one explicit HTTP proxy, optionally with URL credentials.
    Url(String),
}

impl fmt::Debug for ModelProxy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Environment => formatter.write_str("Environment"),
            Self::Disabled => formatter.write_str("Disabled"),
            Self::Url(value) => formatter
                .debug_tuple("Url")
                .field(&redact_proxy_url(value))
                .finish(),
        }
    }
}

/// Origin authentication kept separately from the model URL.
#[derive(Clone)]
pub enum ModelAuthentication {
    Bearer(String),
    Basic { username: String, password: String },
}

impl fmt::Debug for ModelAuthentication {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bearer(_) => formatter.write_str("Bearer(<redacted>)"),
            Self::Basic { .. } => formatter.write_str("Basic(<redacted>)"),
        }
    }
}

/// Network and source policy for model installation and updates.
#[derive(Clone)]
pub struct ModelDownloadOptions {
    /// Refuse every network request. Model operations may use verified cached
    /// data; catalog update revalidates the current embedded/cached catalog.
    pub offline: bool,
    /// Override the model-artifact or catalog URL for the operation, for
    /// example with an authenticated mirror. Signed catalog metadata remains
    /// authoritative.
    pub source_url: Option<String>,
    pub proxy: ModelProxy,
    pub authentication: Option<ModelAuthentication>,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub request_timeout: Duration,
}

impl Default for ModelDownloadOptions {
    fn default() -> Self {
        Self {
            offline: false,
            source_url: None,
            proxy: ModelProxy::Environment,
            authentication: None,
            connect_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(60),
            request_timeout: Duration::from_secs(30 * 60),
        }
    }
}

impl fmt::Debug for ModelDownloadOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelDownloadOptions")
            .field("offline", &self.offline)
            .field("source_url", &self.source_url.as_deref().map(redact_url))
            .field("proxy", &redacted_proxy(&self.proxy))
            .field("authentication", &self.authentication)
            .field("connect_timeout", &self.connect_timeout)
            .field("read_timeout", &self.read_timeout)
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}

impl ModelDownloadOptions {
    /// Build options from denoize-specific environment variables. Standard
    /// proxy variables are resolved lazily for the selected source URL.
    ///
    /// - `DENOIZE_MODEL_OFFLINE=1|true|yes|on`
    /// - `DENOIZE_MODEL_URL`
    /// - `DENOIZE_MODEL_PROXY` (empty means direct connection)
    /// - `DENOIZE_MODEL_BEARER_TOKEN`
    /// - `DENOIZE_MODEL_USERNAME` and `DENOIZE_MODEL_PASSWORD`
    pub fn from_env() -> Result<Self, String> {
        Self::from_env_with(|name| std::env::var(name).ok())
    }

    /// Build options with a caller-provided environment reader. Frontends can
    /// omit variables that are explicitly overridden by their own controls.
    pub fn from_env_with<F>(mut read_environment: F) -> Result<Self, String>
    where
        F: FnMut(&str) -> Option<String>,
    {
        let mut options = Self::default();
        if let Some(value) =
            read_nonempty_environment(&mut read_environment, "DENOIZE_MODEL_OFFLINE")
        {
            options.offline = parse_env_bool("DENOIZE_MODEL_OFFLINE", &value)?;
        }
        options.source_url = read_nonempty_environment(&mut read_environment, "DENOIZE_MODEL_URL");
        if let Some(value) = read_environment("DENOIZE_MODEL_PROXY") {
            options.proxy = if value.trim().is_empty() {
                ModelProxy::Disabled
            } else {
                ModelProxy::Url(value)
            };
        }

        let bearer = read_nonempty_environment(&mut read_environment, "DENOIZE_MODEL_BEARER_TOKEN");
        let username = read_nonempty_environment(&mut read_environment, "DENOIZE_MODEL_USERNAME");
        let password = read_nonempty_environment(&mut read_environment, "DENOIZE_MODEL_PASSWORD");
        if bearer.is_some() && (username.is_some() || password.is_some()) {
            return Err(
                "set either DENOIZE_MODEL_BEARER_TOKEN or basic model credentials, not both".into(),
            );
        }
        options.authentication =
            if let Some(token) = bearer {
                Some(ModelAuthentication::Bearer(token))
            } else {
                match (username, password) {
                    (Some(username), Some(password)) => {
                        Some(ModelAuthentication::Basic { username, password })
                    }
                    (None, None) => None,
                    _ => return Err(
                        "DENOIZE_MODEL_USERNAME and DENOIZE_MODEL_PASSWORD must be set together"
                            .into(),
                    ),
                }
            };
        Ok(options)
    }
}

pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "gtcrn-dns3",
        backend: "gtcrn",
        filename: "gtcrn_simple.onnx",
        url: "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/3862c44808dca492ea5a8a145d2dc2a1028d08c8/stream/onnx_models/gtcrn_simple.onnx",
        revision: "3862c44808dca492ea5a8a145d2dc2a1028d08c8",
        sha256: "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
        size_bytes: 535_190,
        license: "MIT",
        sample_rate: 16_000,
    },
    ModelInfo {
        name: "dpdfnet2-48khz-hr",
        backend: "dpdfnet",
        filename: "dpdfnet2_48khz_hr.onnx",
        url: "https://huggingface.co/Ceva-IP/DPDFNet/resolve/dd6818d00f50c836fed43a6243ebe49116de5964/onnx/dpdfnet2_48khz_hr.onnx",
        revision: "dd6818d00f50c836fed43a6243ebe49116de5964",
        sha256: "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
        size_bytes: 10_493_337,
        license: "Apache-2.0",
        sample_rate: 48_000,
    },
];

#[derive(Clone)]
struct ModelSpec<'a> {
    name: &'a str,
    backend: &'a str,
    filename: &'a str,
    url: &'a str,
    revision: &'a str,
    sha256: &'a str,
    size_bytes: u64,
    license: &'a str,
    sample_rate: u32,
    catalog: Option<CatalogIdentity>,
}

impl<'a> ModelSpec<'a> {
    fn legacy(model: &'a ModelInfo) -> Self {
        let catalog = embedded_catalog();
        let identity = catalog.models().iter().find(|candidate| {
            candidate.name() == model.name
                && candidate.backend() == model.backend
                && candidate.filename() == model.filename
                && candidate.url() == model.url
                && candidate.revision() == model.revision
                && candidate.sha256() == model.sha256
                && candidate.size_bytes() == model.size_bytes
                && candidate.license() == model.license
                && candidate.sample_rate() == model.sample_rate
        });
        Self {
            name: model.name,
            backend: model.backend,
            filename: model.filename,
            url: model.url,
            revision: model.revision,
            sha256: model.sha256,
            size_bytes: model.size_bytes,
            license: model.license,
            sample_rate: model.sample_rate,
            catalog: identity.map(|_| catalog.identity().clone()),
        }
    }

    fn catalog(model: &'a CatalogModel) -> Self {
        Self {
            name: &model.name,
            backend: &model.backend,
            filename: &model.filename,
            url: &model.url,
            revision: &model.revision,
            sha256: &model.sha256,
            size_bytes: model.size_bytes,
            license: &model.license,
            sample_rate: model.sample_rate,
            catalog: Some(model.catalog.clone()),
        }
    }
}

/// How the installed artifact bytes reached the local cache.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ModelInstallationSource {
    CatalogUrl {
        url: String,
    },
    AlternateUrl {
        url: String,
    },
    LocalFile,
    /// A checksum-valid completed `.part` created by an earlier invocation.
    CompletedPartial,
    ExistingCacheMigration,
    /// Installed from a fully authenticated closed-network bundle.
    OfflineBundle {
        bundle_sha256: String,
    },
}

impl<'de> Deserialize<'de> for ModelInstallationSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            kind: String,
            url: Option<String>,
            bundle_sha256: Option<String>,
        }

        let wire = Wire::deserialize(deserializer)?;
        match (wire.kind.as_str(), wire.url, wire.bundle_sha256) {
            ("catalog-url", Some(url), None) => Ok(Self::CatalogUrl { url }),
            ("alternate-url", Some(url), None) => Ok(Self::AlternateUrl { url }),
            ("local-file", None, None) => Ok(Self::LocalFile),
            ("completed-partial", None, None) => Ok(Self::CompletedPartial),
            ("existing-cache-migration", None, None) => Ok(Self::ExistingCacheMigration),
            ("offline-bundle", None, Some(bundle_sha256)) => {
                if bundle_sha256.len() != 64
                    || !bundle_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
                {
                    return Err(serde::de::Error::custom(
                        "offline bundle SHA-256 must be 64 lowercase hexadecimal characters",
                    ));
                }
                Ok(Self::OfflineBundle { bundle_sha256 })
            }
            ("catalog-url" | "alternate-url", None, None) => {
                Err(serde::de::Error::missing_field("url"))
            }
            ("offline-bundle", None, None) => Err(serde::de::Error::missing_field("bundle_sha256")),
            (
                "catalog-url"
                | "alternate-url"
                | "local-file"
                | "completed-partial"
                | "existing-cache-migration"
                | "offline-bundle",
                _,
                _,
            ) => Err(serde::de::Error::custom(
                "installation source contains fields that do not match its kind",
            )),
            _ => Err(serde::de::Error::unknown_variant(
                &wire.kind,
                &[
                    "catalog-url",
                    "alternate-url",
                    "local-file",
                    "completed-partial",
                    "existing-cache-migration",
                    "offline-bundle",
                ],
            )),
        }
    }
}

/// Package metadata bound to an authenticated catalog identity. Artifact and
/// catalog digests bind the package to exact bytes; origin, installation
/// source, and timestamp are local diagnostics rather than a signed attestation.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProvenance {
    pub version: u32,
    pub model_name: String,
    pub backend: String,
    pub filename: String,
    pub revision: String,
    pub license: String,
    pub sample_rate: u32,
    pub artifact_sha256: String,
    pub artifact_size_bytes: u64,
    pub catalog_sequence: u64,
    pub catalog_sha256: String,
    pub catalog_signing_key_id: String,
    pub catalog_origin: CatalogOrigin,
    pub installation_source: ModelInstallationSource,
    pub installed_at_unix_seconds: u64,
}

pub fn find(name: &str) -> Option<&'static ModelInfo> {
    MODELS
        .iter()
        .find(|model| model.name == name || model.backend == name)
}

pub fn cache_dir() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("DENOIZE_MODEL_DIR") {
        return Ok(PathBuf::from(path));
    }
    #[cfg(target_os = "windows")]
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return Ok(PathBuf::from(path).join("denoize").join("models"));
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path).join("denoize").join("models"));
    }
    std::env::var_os("HOME")
        .map(|path| PathBuf::from(path).join(".cache/denoize/models"))
        .ok_or_else(|| "cannot locate model cache; set DENOIZE_MODEL_DIR".into())
}

pub fn path(model: &ModelInfo) -> Result<PathBuf, String> {
    path_for_spec(&ModelSpec::legacy(model))
}

/// Resolve the cache path for a package from the active catalog.
pub fn path_for_catalog_model(model: &CatalogModel) -> Result<PathBuf, String> {
    path_for_spec(&ModelSpec::catalog(model))
}

fn path_for_spec(model: &ModelSpec<'_>) -> Result<PathBuf, String> {
    Ok(cache_dir()?.join(model.name).join(model.filename))
}

fn bundled_path_for_spec(model: &ModelSpec<'_>) -> Option<PathBuf> {
    // An explicit directory is an operator override and must never be
    // shadowed by a model carried inside a plug-in bundle.
    if std::env::var_os("DENOIZE_MODEL_DIR").is_some() {
        return None;
    }
    bundled_model_root().map(|root| root.join(model.name).join(model.filename))
}

#[cfg(target_os = "macos")]
fn bundled_model_root() -> Option<PathBuf> {
    let module = current_module_path()?;
    let root = bundled_model_root_from_module_path(&module)?;
    root.is_dir().then_some(root)
}

#[cfg(not(target_os = "macos"))]
fn bundled_model_root() -> Option<PathBuf> {
    None
}

#[cfg(any(target_os = "macos", test))]
fn bundled_model_root_from_module_path(module: &Path) -> Option<PathBuf> {
    let bundle = module.ancestors().find(|ancestor| {
        matches!(
            ancestor
                .extension()
                .and_then(|extension| extension.to_str()),
            Some("clap" | "appex")
        ) && module.starts_with(ancestor.join("Contents").join("MacOS"))
    })?;
    Some(
        bundle
            .join("Contents")
            .join("Resources")
            .join("denoize-models"),
    )
}

#[cfg(target_os = "macos")]
#[allow(
    unsafe_code,
    reason = "dladdr is the platform API for resolving the loaded plug-in bundle"
)]
#[inline(never)]
fn current_module_path() -> Option<PathBuf> {
    use std::ffi::{c_void, CStr, OsStr};
    use std::os::unix::ffi::OsStrExt;

    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
    let address = current_module_path as *const () as *const c_void;
    // SAFETY: info points to writable storage and address names this loaded
    // function. A successful dladdr call initializes the full Dl_info value.
    if unsafe { libc::dladdr(address, info.as_mut_ptr()) } == 0 {
        return None;
    }
    // SAFETY: dladdr returned success, so info is initialized.
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: a successful dladdr call returns a process-lifetime NUL-
    // terminated image path in dli_fname.
    let bytes = unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes();
    Some(PathBuf::from(OsStr::from_bytes(bytes)))
}

fn verify_runtime_spec(model: &ModelSpec<'_>) -> Result<PathBuf, String> {
    if let Some(destination) = bundled_path_for_spec(model) {
        return verify_bundled_spec_at(model, &destination).map_err(|error| {
            format!(
                "bundled model validation failed at {}: {error}",
                destination.display()
            )
        });
    }
    verify_spec_at(model, &path_for_spec(model)?)
}

fn verify_bundled_spec_at(model: &ModelSpec<'_>, destination: &Path) -> Result<PathBuf, String> {
    validate_model_storage_path(destination)?;
    let verified = verify_bytes_at(model, destination)?;
    if model.catalog.is_some() {
        let path = provenance_path(model, destination)?;
        let provenance = read_provenance(&path)?.ok_or_else(|| {
            format!(
                "authenticated bundled-model provenance is missing: {}",
                path.display()
            )
        })?;
        validate_provenance(model, &provenance)?;
    }
    Ok(verified)
}

pub fn verify(model: &ModelInfo) -> Result<PathBuf, String> {
    let model = ModelSpec::legacy(model);
    verify_runtime_spec(&model)
}

/// Verify an authenticated model carried by a caller-owned plug-in bundle.
///
/// `model_root` is the directory that directly contains catalog package
/// directories (for example, `denoize-models` in an LV2 bundle). Unlike
/// [`verify`], this function never consults the process environment or user
/// cache, so a plug-in can bind activation to the exact resources discovered
/// from its host-provided bundle path.
pub fn verify_bundled(model: &ModelInfo, model_root: &Path) -> Result<PathBuf, String> {
    let model = ModelSpec::legacy(model);
    let destination = model_root.join(model.name).join(model.filename);
    verify_bundled_spec_at(&model, &destination).map_err(|error| {
        format!(
            "bundled model validation failed at {}: {error}",
            destination.display()
        )
    })
}

/// Verify an installed catalog package and its authenticated provenance.
pub fn verify_catalog_model(model: &CatalogModel) -> Result<PathBuf, String> {
    let model = ModelSpec::catalog(model);
    verify_runtime_spec(&model)
}

/// Verify an installed catalog artifact without creating or migrating its
/// provenance record.
///
/// This crate-private path is for read-only inspection such as recommendation.
/// Installation and execution keep using [`verify_catalog_model`] so their
/// authenticated provenance requirements remain unchanged.
pub(crate) fn verify_catalog_model_read_only(model: &CatalogModel) -> Result<PathBuf, String> {
    let model = ModelSpec::catalog(model);
    if let Some(destination) = bundled_path_for_spec(&model) {
        return verify_bundled_spec_at(&model, &destination);
    }
    let destination = path_for_spec(&model)?;
    validate_model_storage_path(&destination)?;
    let verified = verify_bytes_at(&model, &destination)?;
    let provenance_path = provenance_path(&model, &destination)?;
    if let Some(provenance) = read_provenance(&provenance_path)? {
        validate_provenance(&model, &provenance)?;
    }
    Ok(verified)
}

#[cfg(test)]
fn verify_at(model: &ModelInfo, destination: &Path) -> Result<PathBuf, String> {
    verify_bytes_at(&ModelSpec::legacy(model), destination)
}

fn verify_spec_at(model: &ModelSpec<'_>, destination: &Path) -> Result<PathBuf, String> {
    validate_model_storage_path(destination)?;
    verify_bytes_at(model, destination)?;
    if model.catalog.is_none() {
        return Ok(destination.to_path_buf());
    }
    let (_, prepared) = prepare_provenance(
        model,
        destination,
        ModelInstallationSource::ExistingCacheMigration,
    )?;
    let result = verify_bytes_at(model, destination);
    if result.is_err() {
        cleanup_prepared_provenance(&prepared);
    }
    result
}

fn verify_bytes_at(model: &ModelSpec<'_>, destination: &Path) -> Result<PathBuf, String> {
    let Some(mut input) = open_existing_regular_file(destination, "installed model")? else {
        return Err(format!("model is not installed: {}", destination.display()));
    };
    let actual = sha256_open_file_exact(&mut input, destination, model.size_bytes)?;
    if actual != model.sha256 {
        return Err(format!(
            "checksum mismatch for {}: expected {}, got {}",
            destination.display(),
            model.sha256,
            actual
        ));
    }
    Ok(destination.to_path_buf())
}

/// Read and validate the provenance associated with a legacy manifest model.
/// Caller-constructed `ModelInfo` values that do not exactly match the embedded
/// authenticated catalog retain size/SHA verification but have no provenance.
pub fn provenance(model: &ModelInfo) -> Result<ModelProvenance, String> {
    let model = ModelSpec::legacy(model);
    provenance_for_spec(&model, &path_for_spec(&model)?)
}

/// Read and validate the provenance associated with an active catalog model.
pub fn catalog_model_provenance(model: &CatalogModel) -> Result<ModelProvenance, String> {
    let model = ModelSpec::catalog(model);
    provenance_for_spec(&model, &path_for_spec(&model)?)
}

fn provenance_for_spec(
    model: &ModelSpec<'_>,
    destination: &Path,
) -> Result<ModelProvenance, String> {
    if model.catalog.is_none() {
        return Err(format!(
            "model {} is not represented by the embedded authenticated catalog; provenance is unavailable",
            model.name
        ));
    }
    verify_spec_at(model, destination)?;
    let provenance = ensure_provenance(
        model,
        destination,
        ModelInstallationSource::ExistingCacheMigration,
    )?;
    verify_bytes_at(model, destination)?;
    Ok(provenance)
}

pub fn install(model: &ModelInfo) -> Result<PathBuf, String> {
    let options = ModelDownloadOptions::from_env()?;
    install_with_options(model, &options)
}

pub fn install_with_options(
    model: &ModelInfo,
    options: &ModelDownloadOptions,
) -> Result<PathBuf, String> {
    install_with_options_and_progress(model, options, || false, |_, _| {})
}

/// Install a model while reporting downloaded bytes and supporting
/// cancellation. Existing callers inherit environment-based download policy.
pub fn install_with_progress<C, P>(
    model: &ModelInfo,
    cancelled: C,
    progress: P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let options = ModelDownloadOptions::from_env()?;
    install_with_options_and_progress(model, &options, cancelled, progress)
}

/// Install using explicit policy. Interrupted transfers retain a verified,
/// source-bound `.part` sidecar and resume on the next invocation.
pub fn install_with_options_and_progress<C, P>(
    model: &ModelInfo,
    options: &ModelDownloadOptions,
    mut cancelled: C,
    mut progress: P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let model = ModelSpec::legacy(model);
    install_internal(&model, options, false, &mut cancelled, &mut progress)
}

pub fn install_catalog_model(model: &CatalogModel) -> Result<PathBuf, String> {
    let options = ModelDownloadOptions::from_env()?;
    install_catalog_model_with_options(model, &options)
}

pub fn install_catalog_model_with_options(
    model: &CatalogModel,
    options: &ModelDownloadOptions,
) -> Result<PathBuf, String> {
    install_catalog_model_with_options_and_progress(model, options, || false, |_, _| {})
}

pub fn install_catalog_model_with_options_and_progress<C, P>(
    model: &CatalogModel,
    options: &ModelDownloadOptions,
    mut cancelled: C,
    mut progress: P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let model = ModelSpec::catalog(model);
    install_internal(&model, options, false, &mut cancelled, &mut progress)
}

/// Install a model from a local file after checking the pinned byte length and
/// SHA-256. This is the air-gapped counterpart to a network install.
pub fn install_from_file(model: &ModelInfo, source: impl AsRef<Path>) -> Result<PathBuf, String> {
    install_from_file_with_progress(model, source, || false, |_, _| {})
}

pub fn install_from_file_with_progress<C, P>(
    model: &ModelInfo,
    source: impl AsRef<Path>,
    mut cancelled: C,
    mut progress: P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let model = ModelSpec::legacy(model);
    install_spec_from_file_with_progress(
        &model,
        source,
        ModelInstallationSource::LocalFile,
        true,
        &mut cancelled,
        &mut progress,
    )
}

pub fn install_catalog_model_from_file(
    model: &CatalogModel,
    source: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    install_catalog_model_from_file_with_progress(model, source, || false, |_, _| {})
}

pub fn install_catalog_model_from_file_with_progress<C, P>(
    model: &CatalogModel,
    source: impl AsRef<Path>,
    mut cancelled: C,
    mut progress: P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let model = ModelSpec::catalog(model);
    install_spec_from_file_with_progress(
        &model,
        source,
        ModelInstallationSource::LocalFile,
        true,
        &mut cancelled,
        &mut progress,
    )
}

fn install_spec_from_file_with_progress<C, P>(
    model: &ModelSpec<'_>,
    source: impl AsRef<Path>,
    installation_source: ModelInstallationSource,
    require_active_authority: bool,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    if require_active_authority {
        if let Some(identity) = model.catalog.as_ref() {
            catalog::require_catalog_acquisition(identity)?;
        }
    }
    let source = source.as_ref();
    let destination = path_for_spec(model)?;
    ensure_model_parent(&destination)?;
    let lock = acquire_lock(&destination, cancelled)?;
    let result = (|| {
        ensure_model_parent(&destination)?;
        let mut source_file = open_local_model(source)?;
        let actual = sha256_open_file_exact(&mut source_file, source, model.size_bytes)?;
        if actual != model.sha256 {
            return Err(format!(
                "local model checksum mismatch: expected {}, got {}",
                model.sha256, actual
            ));
        }
        let partial = sidecar(&destination, ".part");
        let metadata = sidecar(&destination, ".part.meta");
        reset_partial(&partial, &metadata)?;
        publish_model_open_file(
            model,
            installation_source,
            source_file,
            source,
            &destination,
            model.size_bytes,
            model.sha256,
            cancelled,
            progress,
        )?;
        Ok(destination.clone())
    })();
    drop(lock);
    result
}

pub(super) fn install_catalog_model_from_bundle(
    model: &CatalogModel,
    source: impl AsRef<Path>,
    bundle_sha256: &str,
) -> Result<PathBuf, String> {
    let model = ModelSpec::catalog(model);
    let mut never_cancelled = || false;
    let mut no_progress = |_, _| {};
    install_spec_from_file_with_progress(
        &model,
        source,
        ModelInstallationSource::OfflineBundle {
            bundle_sha256: bundle_sha256.to_string(),
        },
        true,
        &mut never_cancelled,
        &mut no_progress,
    )
}

#[cfg(test)]
pub(super) fn install_catalog_model_from_verified_bundle_for_test(
    model: &CatalogModel,
    source: impl AsRef<Path>,
    bundle_sha256: &str,
) -> Result<PathBuf, String> {
    let model = ModelSpec::catalog(model);
    let mut never_cancelled = || false;
    let mut no_progress = |_, _| {};
    install_spec_from_file_with_progress(
        &model,
        source,
        ModelInstallationSource::OfflineBundle {
            bundle_sha256: bundle_sha256.to_string(),
        },
        false,
        &mut never_cancelled,
        &mut no_progress,
    )
}

/// Remove an installed model and all interrupted-download state.
pub fn remove(model: &ModelInfo) -> Result<bool, String> {
    remove_spec(&ModelSpec::legacy(model))
}

pub fn remove_catalog_model(model: &CatalogModel) -> Result<bool, String> {
    remove_spec(&ModelSpec::catalog(model))
}

fn remove_spec(model: &ModelSpec<'_>) -> Result<bool, String> {
    let destination = path_for_spec(model)?;
    validate_model_storage_path(&destination)?;
    let Some(parent) = destination.parent() else {
        return Err("invalid model cache path".into());
    };
    let mut never_cancelled = || false;
    let lock = acquire_lock(&destination, &mut never_cancelled)?;
    validate_model_storage_path(&destination)?;
    let partial = sidecar(&destination, ".part");
    let metadata = sidecar(&destination, ".part.meta");
    let removed = remove_file_if_present(&destination)?
        | remove_file_if_present(&partial)?
        | remove_file_if_present(&metadata)?
        | remove_file_if_present(&sidecar(&metadata, ".tmp"))?
        | remove_provenance_state(&destination)?;
    match std::fs::remove_dir(parent) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) if error.kind() == ErrorKind::DirectoryNotEmpty => {}
        Err(error) => return Err(format!("failed to remove {}: {error}", parent.display())),
    }
    drop(lock);
    Ok(removed)
}

pub fn update(model: &ModelInfo) -> Result<PathBuf, String> {
    let options = ModelDownloadOptions::from_env()?;
    update_with_options(model, &options)
}

pub fn update_with_options(
    model: &ModelInfo,
    options: &ModelDownloadOptions,
) -> Result<PathBuf, String> {
    update_with_options_and_progress(model, options, || false, |_, _| {})
}

pub fn update_with_progress<C, P>(
    model: &ModelInfo,
    cancelled: C,
    progress: P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let options = ModelDownloadOptions::from_env()?;
    update_with_options_and_progress(model, &options, cancelled, progress)
}

/// Download a replacement while retaining the currently verified model until
/// the new bytes have passed integrity verification and can be atomically
/// committed.
pub fn update_with_options_and_progress<C, P>(
    model: &ModelInfo,
    options: &ModelDownloadOptions,
    mut cancelled: C,
    mut progress: P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let model = ModelSpec::legacy(model);
    install_internal(&model, options, true, &mut cancelled, &mut progress)
}

pub fn update_catalog_model_with_options(
    model: &CatalogModel,
    options: &ModelDownloadOptions,
) -> Result<PathBuf, String> {
    update_catalog_model_with_options_and_progress(model, options, || false, |_, _| {})
}

pub fn update_catalog_model_with_options_and_progress<C, P>(
    model: &CatalogModel,
    options: &ModelDownloadOptions,
    mut cancelled: C,
    mut progress: P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let model = ModelSpec::catalog(model);
    install_internal(&model, options, true, &mut cancelled, &mut progress)
}

fn install_internal<C, P>(
    model: &ModelSpec<'_>,
    options: &ModelDownloadOptions,
    force: bool,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let destination = path_for_spec(model)?;
    if !force {
        if let Ok(path) = verify_spec_at(model, &destination) {
            return Ok(path);
        }
    }
    if let Some(identity) = model.catalog.as_ref() {
        catalog::require_catalog_acquisition(identity)?;
    }
    ensure_model_parent(&destination)?;
    let lock = acquire_lock(&destination, cancelled)?;
    let result = (|| {
        ensure_model_parent(&destination)?;
        if !force {
            if let Ok(path) = verify_spec_at(model, &destination) {
                return Ok(path);
            }
        }
        let installation_source = installation_source_for_download(model, options);
        let partial = sidecar(&destination, ".part");
        let metadata_path = sidecar(&destination, ".part.meta");
        reject_symlink(&partial)?;
        reject_symlink(&metadata_path)?;

        if let Some(mut partial_file) =
            open_existing_regular_file(&partial, "partial model download")?
        {
            let partial_size = open_file_length(&partial_file, &partial)?;
            if partial_size == model.size_bytes {
                let valid =
                    match sha256_open_file_exact(&mut partial_file, &partial, model.size_bytes) {
                        Ok(actual) => actual == model.sha256,
                        Err(error) => {
                            drop(partial_file);
                            reset_partial(&partial, &metadata_path)?;
                            return Err(error);
                        }
                    };
                drop(partial_file);
                if valid {
                    let mut ignored_progress = |_, _| {};
                    publish_model_file(
                        model,
                        ModelInstallationSource::CompletedPartial,
                        &partial,
                        &destination,
                        model.size_bytes,
                        model.sha256,
                        cancelled,
                        &mut ignored_progress,
                    )?;
                    remove_file_if_present(&partial)?;
                    remove_file_if_present(&metadata_path)?;
                    return Ok(destination.clone());
                }
                reset_partial(&partial, &metadata_path)?;
            } else if partial_size > model.size_bytes {
                drop(partial_file);
                reset_partial(&partial, &metadata_path)?;
            }
        }
        if options.offline {
            if let Ok(path) = verify_spec_at(model, &destination) {
                return Ok(path);
            }
            return Err(format!(
                "offline mode: no verified model is available at {} (use `models install {} --from PATH`)",
                destination.display(), model.name
            ));
        }

        let raw_url = options.source_url.as_deref().unwrap_or(model.url);
        let source = Url::parse(raw_url).map_err(|_| "invalid model source URL".to_string())?;
        if !matches!(source.scheme(), "http" | "https") || source.host_str().is_none() {
            return Err(
                "model source URL must be an absolute http or https URL; use --from for local files"
                    .into(),
            );
        }
        validate_authentication(&source, options.authentication.as_ref())?;
        let source_id = source_identity(raw_url);
        download_spec(
            model,
            options,
            &source,
            &source_id,
            &partial,
            &metadata_path,
            cancelled,
            progress,
        )?;
        let mut ignored_progress = |_, _| {};
        publish_model_file(
            model,
            installation_source,
            &partial,
            &destination,
            model.size_bytes,
            model.sha256,
            cancelled,
            &mut ignored_progress,
        )?;
        remove_file_if_present(&partial)?;
        remove_file_if_present(&metadata_path)?;
        Ok(destination.clone())
    })();
    drop(lock);
    result
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn download<C, P>(
    model: &ModelInfo,
    options: &ModelDownloadOptions,
    source: &Url,
    source_id: &str,
    partial: &Path,
    metadata_path: &Path,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<(), String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let model = ModelSpec::legacy(model);
    download_spec(
        &model,
        options,
        source,
        source_id,
        partial,
        metadata_path,
        cancelled,
        progress,
    )
}

#[allow(clippy::too_many_arguments)]
fn download_spec<C, P>(
    model: &ModelSpec<'_>,
    options: &ModelDownloadOptions,
    source: &Url,
    source_id: &str,
    partial: &Path,
    metadata_path: &Path,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<(), String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let mut clean_retry_available = true;
    loop {
        if cancelled() {
            return Err("cancelled".into());
        }
        let mut metadata =
            load_matching_metadata(metadata_path, source_id, model.sha256, model.size_bytes)?;
        let mut partial_file = open_existing_regular_file(partial, "partial model download")?;
        let mut downloaded = partial_file
            .as_ref()
            .map(|file| open_file_length(file, partial))
            .transpose()?
            .unwrap_or(0);
        if downloaded == 0 {
            drop(partial_file.take());
            remove_file_if_present(metadata_path)?;
            metadata = None;
        } else if metadata.is_none() {
            drop(partial_file.take());
            reset_partial(partial, metadata_path)?;
            downloaded = 0;
        }

        if downloaded > model.size_bytes {
            drop(partial_file.take());
            reset_partial(partial, metadata_path)?;
            downloaded = 0;
            metadata = None;
        } else if downloaded == model.size_bytes {
            let mut input = partial_file
                .take()
                .expect("a non-empty partial has an open file");
            if sha256_open_file_exact(&mut input, partial, model.size_bytes)? == model.sha256 {
                return Ok(());
            }
            drop(input);
            reset_partial(partial, metadata_path)?;
            downloaded = 0;
            metadata = None;
        }
        drop(partial_file);

        let response = request_with_redirects(options, source, downloaded, metadata.as_ref())?;
        let status = response.status();
        if status == 416 && downloaded > 0 {
            if handle_range_not_satisfiable(
                &response,
                partial,
                downloaded,
                model.size_bytes,
                model.sha256,
            )? {
                return Ok(());
            }
            if !clean_retry_available {
                return Err("model server repeatedly rejected a clean download".into());
            }
            clean_retry_available = false;
            reset_partial(partial, metadata_path)?;
            continue;
        }
        if status >= 400 {
            return Err(format!(
                "model download from {} failed with HTTP {status}",
                redact_url(source.as_str())
            ));
        }
        let (resumed, expected_body_length) = match status {
            200 => {
                let content_length = parse_content_length(&response)?;
                if content_length.is_some_and(|length| length != model.size_bytes) {
                    return Err(format!(
                        "model response size mismatch: expected {} bytes, got {}",
                        model.size_bytes,
                        content_length.expect("checked as present")
                    ));
                }
                (false, content_length)
            }
            206 => match validate_partial_response(
                &response,
                downloaded,
                metadata.as_ref(),
                model.size_bytes,
            ) {
                Ok(length) => (downloaded > 0, Some(length)),
                Err(error) => {
                    if !clean_retry_available {
                        return Err(error);
                    }
                    clean_retry_available = false;
                    reset_partial(partial, metadata_path)?;
                    continue;
                }
            },
            _ => {
                return Err(format!(
                    "model download from {} returned unexpected HTTP {status}",
                    redact_url(source.as_str())
                ))
            }
        };
        let total = Some(model.size_bytes);

        let received_before = if resumed { downloaded } else { 0 };
        let next_metadata = PartialMetadata {
            version: PARTIAL_METADATA_VERSION,
            source_id: source_id.to_string(),
            expected_sha256: model.sha256.to_string(),
            etag: response.header("ETag").map(str::to_string).or_else(|| {
                resumed
                    .then(|| metadata.as_ref().and_then(|meta| meta.etag.clone()))
                    .flatten()
            }),
            last_modified: response
                .header("Last-Modified")
                .map(str::to_string)
                .or_else(|| {
                    resumed
                        .then(|| {
                            metadata
                                .as_ref()
                                .and_then(|meta| meta.last_modified.clone())
                        })
                        .flatten()
                }),
            total,
        };
        write_metadata(metadata_path, &next_metadata)?;

        let mut output = open_partial(partial, resumed)?;
        let opened_length = open_file_length(&output, partial)?;
        if opened_length != received_before {
            drop(output);
            reset_partial(partial, metadata_path)?;
            return Err(format!(
                "partial model size changed before writing: expected {received_before} bytes, got {opened_length}"
            ));
        }
        let mut reader = response.into_reader();
        let mut received = received_before;
        let mut body_received = 0_u64;
        progress(received, total);
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            if cancelled() {
                output
                    .flush()
                    .map_err(|error| format!("failed to flush {}: {error}", partial.display()))?;
                return Err("cancelled".into());
            }
            let count = reader.read(&mut buffer).map_err(|error| {
                format!(
                    "model download from {} was interrupted: {error}",
                    redact_url(source.as_str())
                )
            })?;
            if count == 0 {
                break;
            }
            let next_received = received
                .checked_add(count as u64)
                .ok_or_else(|| "model response size overflow".to_string())?;
            let next_body_received = body_received
                .checked_add(count as u64)
                .ok_or_else(|| "model response size overflow".to_string())?;
            if next_received > model.size_bytes
                || expected_body_length.is_some_and(|expected| next_body_received > expected)
            {
                drop(output);
                reset_partial(partial, metadata_path)?;
                return Err("model server sent more bytes than allowed by catalog metadata".into());
            }
            output
                .write_all(&buffer[..count])
                .map_err(|error| format!("failed to save {}: {error}", partial.display()))?;
            received = next_received;
            body_received = next_body_received;
            progress(received, total);
        }
        output
            .sync_all()
            .map_err(|error| format!("failed to sync {}: {error}", partial.display()))?;
        if let Some(expected) = expected_body_length {
            if body_received != expected {
                drop(output);
                reset_partial(partial, metadata_path)?;
                return Err(format!(
                    "model response body length mismatch: received {body_received}, expected {expected}"
                ));
            }
        }
        if received != model.size_bytes {
            return Err(format!(
                "model download is incomplete: received {received} of {} bytes",
                model.size_bytes
            ));
        }
        let actual = match sha256_open_file_exact(&mut output, partial, model.size_bytes) {
            Ok(actual) => actual,
            Err(error) => {
                drop(output);
                reset_partial(partial, metadata_path)?;
                return Err(error);
            }
        };
        if actual != model.sha256 {
            drop(output);
            reset_partial(partial, metadata_path)?;
            return Err(format!(
                "downloaded model checksum mismatch: expected {}, got {}; discarded corrupt partial download",
                model.sha256, actual
            ));
        }
        return Ok(());
    }
}

fn request_with_redirects(
    options: &ModelDownloadOptions,
    source: &Url,
    downloaded: u64,
    metadata: Option<&PartialMetadata>,
) -> Result<ureq::Response, String> {
    let mut current = source.clone();
    for redirect_count in 0..=5 {
        let client = build_client(&current, &options.proxy, options)?;
        let forwards_origin_authentication = may_forward_origin_authentication(source, &current);
        let has_url_credentials = !source.username().is_empty() || source.password().is_some();
        let has_sensitive_request = current.query().is_some()
            || (forwards_origin_authentication
                && (options.authentication.is_some() || has_url_credentials));
        validate_auth_transport(&current, has_sensitive_request, client.uses_proxy)?;
        let mut request = client
            .agent
            .get(current.as_str())
            .set("Accept-Encoding", "identity");
        if let Some(authorization) = client.proxy_authorization.as_deref() {
            request = request.set("Proxy-Authorization", authorization);
        }
        if forwards_origin_authentication {
            request = apply_authentication(request, options.authentication.as_ref());
            if options.authentication.is_none() {
                if let Some(authorization) = url_basic_authorization(source)? {
                    request = request.set("Authorization", &authorization);
                }
            }
        }
        if downloaded > 0 {
            request = request.set("Range", &format!("bytes={downloaded}-"));
            if let Some(validator) = metadata.and_then(PartialMetadata::if_range) {
                request = request.set("If-Range", validator);
            }
        }

        let response = match request.call() {
            Ok(response) => response,
            Err(ureq::Error::Status(_, response)) => response,
            Err(ureq::Error::Transport(error)) => {
                return Err(format!(
                    "model download from {} failed: {}",
                    redact_url(source.as_str()),
                    error.kind()
                ));
            }
        };
        if !matches!(response.status(), 301 | 302 | 303 | 307 | 308) {
            return Ok(response);
        }
        if redirect_count == 5 {
            return Err(format!(
                "model download from {} exceeded 5 redirects",
                redact_url(source.as_str())
            ));
        }
        let location = response
            .header("Location")
            .ok_or_else(|| "model redirect omitted Location".to_string())?;
        let next = current
            .join(location)
            .map_err(|_| "model redirect has an invalid Location".to_string())?;
        if !matches!(next.scheme(), "http" | "https") {
            return Err("model redirect must use http or https".into());
        }
        if current.scheme() == "https" && next.scheme() != "https" {
            return Err("refusing to follow an HTTPS model redirect to HTTP".into());
        }
        if (!next.username().is_empty() || next.password().is_some()) && !same_origin(source, &next)
        {
            return Err("refusing model redirect with cross-origin URL credentials".into());
        }
        current = next;
    }
    unreachable!("redirect loop returns at its configured limit")
}

fn may_forward_origin_authentication(source: &Url, destination: &Url) -> bool {
    source.host_str() == destination.host_str()
        && source.port_or_known_default() == destination.port_or_known_default()
        && (source.scheme() == destination.scheme()
            || (source.scheme() == "http" && destination.scheme() == "https"))
}

fn same_origin(left: &Url, right: &Url) -> bool {
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

struct ModelHttpClient {
    agent: ureq::Agent,
    uses_proxy: bool,
    /// ureq 2.x applies URL credentials to HTTPS CONNECT, but not to a plain
    /// HTTP proxy request. Only populate this for a selected HTTP proxy and an
    /// `http` origin so the header can never enter an HTTPS origin tunnel.
    proxy_authorization: Option<String>,
}

fn build_client(
    source: &Url,
    mode: &ModelProxy,
    options: &ModelDownloadOptions,
) -> Result<ModelHttpClient, String> {
    let proxy = resolve_proxy(source, mode)?;
    let mut builder = ureq::AgentBuilder::new()
        .try_proxy_from_env(false)
        .redirects(0)
        .user_agent(USER_AGENT)
        .timeout_connect(options.connect_timeout)
        .timeout_read(options.read_timeout)
        .timeout(options.request_timeout);
    let mut proxy_authorization = None;
    let uses_proxy = proxy.is_some();
    if let Some((proxy, authorization)) = proxy {
        builder = builder.proxy(proxy);
        proxy_authorization = authorization;
    }
    Ok(ModelHttpClient {
        agent: builder.build(),
        uses_proxy,
        proxy_authorization,
    })
}

fn resolve_proxy(
    source: &Url,
    mode: &ModelProxy,
) -> Result<Option<(ureq::Proxy, Option<String>)>, String> {
    let proxy = match mode {
        ModelProxy::Disabled => return Ok(None),
        ModelProxy::Url(value) => Some(("explicit proxy", value.clone())),
        ModelProxy::Environment => {
            if env_no_proxy_matches(source) {
                return Ok(None);
            }
            let keys: &[&str] = if source.scheme() == "https" {
                &["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]
            } else {
                &["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]
            };
            keys.iter()
                .find_map(|key| env_value(key).map(|value| (*key, value)))
        }
    };
    proxy
        .map(|(label, value)| {
            let (normalized, credentials) = normalize_proxy_url(&value)
                .map_err(|error| format!("invalid {label} URL (value redacted): {error}"))?;
            let proxy = ureq::Proxy::new(&normalized)
                .map_err(|_| format!("invalid {label} URL (value redacted)"))?;
            let authorization = (source.scheme() == "http").then_some(credentials).flatten();
            Ok((proxy, authorization))
        })
        .transpose()
}

fn normalize_proxy_url(value: &str) -> Result<(String, Option<String>), String> {
    let value = value.trim();
    let candidate = if value.contains("://") {
        value.to_string()
    } else {
        format!("http://{value}")
    };
    let proxy = Url::parse(&candidate).map_err(|_| "invalid proxy URL".to_string())?;
    if proxy.scheme() != "http" {
        return Err("model proxy URL must use http".into());
    }
    if !matches!(proxy.path(), "" | "/") || proxy.query().is_some() || proxy.fragment().is_some() {
        return Err("model proxy URL must not contain a path, query, or fragment".into());
    }
    if matches!(proxy.host(), Some(url::Host::Ipv6(_))) {
        return Err("literal IPv6 proxy addresses are not supported by the HTTP client".into());
    }
    let host = proxy
        .host_str()
        .ok_or_else(|| "model proxy URL requires a host".to_string())?;
    let port = proxy
        .port_or_known_default()
        .ok_or_else(|| "model proxy URL requires a valid port".to_string())?;
    let has_credentials = !proxy.username().is_empty() || proxy.password().is_some();
    if !has_credentials {
        return Ok((format!("http://{host}:{port}"), None));
    }
    let username = percent_decode_str(proxy.username())
        .decode_utf8()
        .map_err(|_| "proxy credentials must be valid UTF-8".to_string())?;
    let password = percent_decode_str(proxy.password().unwrap_or_default())
        .decode_utf8()
        .map_err(|_| "proxy credentials must be valid UTF-8".to_string())?;
    if username.is_empty()
        || password.is_empty()
        || username.contains(':')
        || contains_header_newline(&username)
        || contains_header_newline(&password)
    {
        return Err("proxy credentials contain invalid characters".into());
    }
    let normalized = format!("http://{username}:{password}@{host}:{port}");
    let authorization = format!(
        "Basic {}",
        BASE64_STANDARD.encode(format!("{username}:{password}"))
    );
    Ok((normalized, Some(authorization)))
}

fn env_no_proxy_matches(source: &Url) -> bool {
    let Some(host) = source.host_str() else {
        return false;
    };
    let port = source.port_or_known_default();
    let Some(value) = env_value("NO_PROXY").or_else(|| env_value("no_proxy")) else {
        return false;
    };
    no_proxy_matches(host, port, &value)
}

fn no_proxy_matches(host: &str, port: Option<u16>, value: &str) -> bool {
    let host = host.trim_matches(['[', ']']).trim_end_matches('.');
    value.split(',').any(|entry| {
        let entry = entry.trim();
        if entry == "*" {
            return true;
        }
        if entry.is_empty() {
            return false;
        }
        if let Some((network, prefix)) = entry.rsplit_once('/') {
            if let (Ok(address), Ok(network), Ok(prefix)) = (
                host.parse::<std::net::IpAddr>(),
                network.trim_matches(['[', ']']).parse::<std::net::IpAddr>(),
                prefix.parse::<u8>(),
            ) {
                return ip_in_network(address, network, prefix);
            }
        }
        let (entry_host, entry_port) = split_no_proxy_entry(entry);
        if entry_port.is_some() && entry_port != port {
            return false;
        }
        let entry_host = entry_host
            .trim_start_matches("*.")
            .trim_start_matches('.')
            .trim_matches(['[', ']'])
            .trim_end_matches('.');
        !entry_host.is_empty()
            && (host.eq_ignore_ascii_case(entry_host)
                || host
                    .to_ascii_lowercase()
                    .ends_with(&format!(".{}", entry_host.to_ascii_lowercase())))
    })
}

fn ip_in_network(address: std::net::IpAddr, network: std::net::IpAddr, prefix: u8) -> bool {
    match (address, network) {
        (std::net::IpAddr::V4(address), std::net::IpAddr::V4(network)) if prefix <= 32 => {
            let mask = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - prefix)
            };
            u32::from(address) & mask == u32::from(network) & mask
        }
        (std::net::IpAddr::V6(address), std::net::IpAddr::V6(network)) if prefix <= 128 => {
            let mask = if prefix == 0 {
                0
            } else {
                u128::MAX << (128 - prefix)
            };
            u128::from(address) & mask == u128::from(network) & mask
        }
        _ => false,
    }
}

fn split_no_proxy_entry(entry: &str) -> (&str, Option<u16>) {
    if entry.starts_with('[') {
        if let Some(end) = entry.find(']') {
            let port = entry
                .get(end + 1..)
                .and_then(|tail| tail.strip_prefix(':'))
                .and_then(|port| port.parse().ok());
            return (&entry[..=end], port);
        }
    }
    if entry.matches(':').count() == 1 {
        if let Some((host, port)) = entry.rsplit_once(':') {
            if let Ok(port) = port.parse() {
                return (host, Some(port));
            }
        }
    }
    (entry, None)
}

fn apply_authentication(
    mut request: ureq::Request,
    authentication: Option<&ModelAuthentication>,
) -> ureq::Request {
    if let Some(authentication) = authentication {
        let value = match authentication {
            ModelAuthentication::Bearer(token) => format!("Bearer {token}"),
            ModelAuthentication::Basic { username, password } => format!(
                "Basic {}",
                BASE64_STANDARD.encode(format!("{username}:{password}"))
            ),
        };
        request = request.set("Authorization", &value);
    }
    request
}

fn url_basic_authorization(source: &Url) -> Result<Option<String>, String> {
    if source.username().is_empty() && source.password().is_none() {
        return Ok(None);
    }
    let username = percent_decode_str(source.username())
        .decode_utf8()
        .map_err(|_| "model URL credentials must be valid UTF-8".to_string())?;
    let password = percent_decode_str(source.password().unwrap_or_default())
        .decode_utf8()
        .map_err(|_| "model URL credentials must be valid UTF-8".to_string())?;
    if username.is_empty()
        || password.is_empty()
        || username.contains(':')
        || contains_header_newline(&username)
        || contains_header_newline(&password)
    {
        return Err("model URL credentials contain invalid characters".into());
    }
    Ok(Some(format!(
        "Basic {}",
        BASE64_STANDARD.encode(format!("{username}:{password}"))
    )))
}

fn validate_authentication(
    source: &Url,
    authentication: Option<&ModelAuthentication>,
) -> Result<(), String> {
    if let Some(authentication) = authentication {
        match authentication {
            ModelAuthentication::Bearer(token) if token.is_empty() => {
                return Err("model bearer token cannot be empty".into())
            }
            ModelAuthentication::Bearer(token) if contains_header_newline(token) => {
                return Err("model bearer token contains invalid characters".into())
            }
            ModelAuthentication::Basic { username, password }
                if username.is_empty() || password.is_empty() =>
            {
                return Err("model basic username and password cannot be empty".into())
            }
            ModelAuthentication::Basic { username, password }
                if username.contains(':')
                    || contains_header_newline(username)
                    || contains_header_newline(password) =>
            {
                return Err("model basic credentials contain invalid characters".into())
            }
            _ => {}
        }
    }
    let has_url_credentials = !source.username().is_empty() || source.password().is_some();
    if has_url_credentials && authentication.is_some() {
        return Err("model URL credentials cannot be combined with explicit authentication".into());
    }
    let _ = url_basic_authorization(source)?;
    Ok(())
}

fn validate_auth_transport(
    destination: &Url,
    has_origin_credentials: bool,
    uses_proxy: bool,
) -> Result<(), String> {
    if has_origin_credentials
        && destination.scheme() != "https"
        && (!is_loopback(destination.host_str()) || uses_proxy)
    {
        return Err("refusing to send model credentials over non-HTTPS transport".into());
    }
    Ok(())
}

fn contains_header_newline(value: &str) -> bool {
    value.contains(['\r', '\n'])
}

fn is_loopback(host: Option<&str>) -> bool {
    match host {
        Some("localhost") => true,
        Some(host) => host
            .trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback()),
        None => false,
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PartialMetadata {
    version: u32,
    source_id: String,
    expected_sha256: String,
    etag: Option<String>,
    last_modified: Option<String>,
    total: Option<u64>,
}

impl PartialMetadata {
    fn if_range(&self) -> Option<&str> {
        self.etag
            .as_deref()
            .filter(|etag| !etag.trim_start().starts_with("W/"))
            .or(self.last_modified.as_deref())
    }
}

fn load_matching_metadata(
    path: &Path,
    source_id: &str,
    expected_sha256: &str,
    expected_size: u64,
) -> Result<Option<PartialMetadata>, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("failed to open {}: {error}", path.display())),
    };
    require_regular_file(&file, path, "partial download metadata")?;
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
    let metadata: PartialMetadata = match serde_json::from_slice(&bytes) {
        Ok(metadata) => metadata,
        Err(_) => return Ok(None),
    };
    Ok((metadata.version == PARTIAL_METADATA_VERSION
        && metadata.source_id == source_id
        && metadata.expected_sha256 == expected_sha256)
        .then_some(metadata)
        .filter(|metadata| metadata.total.is_none_or(|total| total == expected_size)))
}

fn write_metadata(path: &Path, metadata: &PartialMetadata) -> Result<(), String> {
    let bytes = serde_json::to_vec(metadata)
        .map_err(|error| format!("failed to encode download metadata: {error}"))?;
    let mut output = AtomicOutput::new(path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    output.commit(CommitMode::Replace)
}

fn parse_content_length(response: &ureq::Response) -> Result<Option<u64>, String> {
    response
        .header("Content-Length")
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|_| "model response has invalid Content-Length".to_string())
        })
        .transpose()
}

fn validate_partial_response(
    response: &ureq::Response,
    requested_start: u64,
    metadata: Option<&PartialMetadata>,
    expected_size: u64,
) -> Result<u64, String> {
    let value = response
        .header("Content-Range")
        .ok_or_else(|| "partial model response omitted Content-Range".to_string())?;
    let (start, end, total) = parse_satisfied_content_range(value)
        .ok_or_else(|| "partial model response has invalid Content-Range".to_string())?;
    if start != requested_start {
        return Err(format!(
            "partial model response starts at {start}, expected {requested_start}"
        ));
    }
    if start >= expected_size || end >= expected_size {
        return Err("partial model response exceeds the catalog package size".into());
    }
    if total.is_some_and(|total| total != expected_size) {
        return Err(format!(
            "partial model response size mismatch: expected {expected_size} bytes, got {}",
            total.expect("checked as present")
        ));
    }
    let range_length = end
        .checked_sub(start)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| "partial model response has invalid Content-Range".to_string())?;
    if let Some(length) = parse_content_length(response)? {
        if length != range_length {
            return Err("partial model response length disagrees with Content-Range".into());
        }
    }
    if metadata
        .and_then(|metadata| metadata.total)
        .is_some_and(|total| total != expected_size)
    {
        return Err("partial metadata conflicts with the catalog package size".into());
    }
    if let (Some(old), Some(new)) = (
        metadata.and_then(|meta| meta.etag.as_deref()),
        response.header("ETag"),
    ) {
        if old != new {
            return Err("model source ETag changed during resume".into());
        }
    }
    if let (Some(old), Some(new)) = (
        metadata.and_then(|meta| meta.last_modified.as_deref()),
        response.header("Last-Modified"),
    ) {
        if old != new {
            return Err("model source Last-Modified changed during resume".into());
        }
    }
    Ok(range_length)
}

fn parse_satisfied_content_range(value: &str) -> Option<(u64, u64, Option<u64>)> {
    let value = value.trim().strip_prefix("bytes ")?;
    let (range, total) = value.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let start: u64 = start.parse().ok()?;
    let end: u64 = end.parse().ok()?;
    if end < start {
        return None;
    }
    end.checked_sub(start)?.checked_add(1)?;
    let total = if total == "*" {
        None
    } else {
        let total: u64 = total.parse().ok()?;
        if total == 0 || end >= total {
            return None;
        }
        Some(total)
    };
    Some((start, end, total))
}

fn parse_unsatisfied_content_range(value: &str) -> Option<u64> {
    value.trim().strip_prefix("bytes */")?.parse().ok()
}

fn handle_range_not_satisfiable(
    response: &ureq::Response,
    partial: &Path,
    downloaded: u64,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<bool, String> {
    let total = response
        .header("Content-Range")
        .and_then(parse_unsatisfied_content_range);
    if downloaded != expected_size || total != Some(expected_size) {
        return Ok(false);
    }
    Ok(file_matches(partial, expected_size, expected_sha256))
}

fn installation_source_for_download(
    model: &ModelSpec<'_>,
    options: &ModelDownloadOptions,
) -> ModelInstallationSource {
    match options.source_url.as_deref() {
        Some(url) => ModelInstallationSource::AlternateUrl {
            url: redact_url(url),
        },
        None => ModelInstallationSource::CatalogUrl {
            url: redact_url(model.url),
        },
    }
}

struct PreparedProvenance {
    path: PathBuf,
    created: bool,
}

fn ensure_provenance(
    model: &ModelSpec<'_>,
    destination: &Path,
    installation_source: ModelInstallationSource,
) -> Result<ModelProvenance, String> {
    let (provenance, _) = prepare_provenance(model, destination, installation_source)?;
    Ok(provenance)
}

fn prepare_provenance(
    model: &ModelSpec<'_>,
    destination: &Path,
    installation_source: ModelInstallationSource,
) -> Result<(ModelProvenance, PreparedProvenance), String> {
    let catalog = model.catalog.as_ref().ok_or_else(|| {
        format!(
            "model {} is not represented by an authenticated catalog",
            model.name
        )
    })?;
    let path = provenance_path(model, destination)?;
    if let Some(provenance) = read_provenance(&path)? {
        validate_provenance(model, &provenance)?;
        return Ok((
            provenance,
            PreparedProvenance {
                path,
                created: false,
            },
        ));
    }

    let directory = provenance_directory(destination)?;
    reject_symlink(&directory)?;
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create {}: {error}", directory.display()))?;
    reject_symlink(&directory)?;
    let provenance = ModelProvenance {
        version: MODEL_PROVENANCE_VERSION,
        model_name: model.name.to_string(),
        backend: model.backend.to_string(),
        filename: model.filename.to_string(),
        revision: model.revision.to_string(),
        license: model.license.to_string(),
        sample_rate: model.sample_rate,
        artifact_sha256: model.sha256.to_string(),
        artifact_size_bytes: model.size_bytes,
        catalog_sequence: catalog.sequence,
        catalog_sha256: catalog.sha256.clone(),
        catalog_signing_key_id: catalog.signing_key_id.clone(),
        catalog_origin: catalog.origin.clone(),
        installation_source,
        installed_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "system clock is before the Unix epoch".to_string())?
            .as_secs(),
    };
    let bytes = serde_json::to_vec_pretty(&provenance)
        .map_err(|error| format!("failed to encode model provenance: {error}"))?;
    if bytes.len() as u64 > MAX_MODEL_PROVENANCE_BYTES {
        return Err("model provenance exceeds the 64 KiB limit".into());
    }
    let mut output = AtomicOutput::new(&path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    let created = match output.commit(CommitMode::NoClobber) {
        Ok(()) => true,
        Err(error) => {
            if let Some(existing) = read_provenance(&path)? {
                validate_provenance(model, &existing)?;
                return Ok((
                    existing,
                    PreparedProvenance {
                        path,
                        created: false,
                    },
                ));
            }
            return Err(error);
        }
    };
    let stored = read_provenance(&path)?
        .ok_or_else(|| format!("model provenance disappeared: {}", path.display()))?;
    validate_provenance(model, &stored)?;
    Ok((stored, PreparedProvenance { path, created }))
}

fn read_provenance(path: &Path) -> Result<Option<ModelProvenance>, String> {
    let Some(file) = open_existing_regular_file(path, "model provenance")? else {
        return Ok(None);
    };
    let length = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_MODEL_PROVENANCE_BYTES {
        return Err(format!(
            "model provenance exceeds the 64 KiB limit: {}",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(MAX_MODEL_PROVENANCE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_MODEL_PROVENANCE_BYTES {
        return Err(format!(
            "model provenance exceeds the 64 KiB limit: {}",
            path.display()
        ));
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| format!("invalid model provenance {}: {error}", path.display()))
}

fn validate_provenance(model: &ModelSpec<'_>, provenance: &ModelProvenance) -> Result<(), String> {
    let catalog = model.catalog.as_ref().ok_or_else(|| {
        format!(
            "model {} is not represented by an authenticated catalog",
            model.name
        )
    })?;
    // Origin is installation-time diagnostic context, not part of the signed
    // catalog identity. The same catalog may move from local import to HTTPS,
    // or become embedded in a later binary, without invalidating model bytes.
    let catalog_origin_is_valid = catalog::catalog_origin_is_safe(&provenance.catalog_origin);
    let installation_source_is_valid = match &provenance.installation_source {
        ModelInstallationSource::CatalogUrl { url }
        | ModelInstallationSource::AlternateUrl { url } => valid_provenance_url(url),
        ModelInstallationSource::LocalFile
        | ModelInstallationSource::CompletedPartial
        | ModelInstallationSource::ExistingCacheMigration => true,
        ModelInstallationSource::OfflineBundle { bundle_sha256 } => {
            bundle_sha256.len() == 64
                && bundle_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        }
    };
    if provenance.version != MODEL_PROVENANCE_VERSION
        || provenance.model_name != model.name
        || provenance.backend != model.backend
        || provenance.filename != model.filename
        || provenance.revision != model.revision
        || provenance.license != model.license
        || provenance.sample_rate != model.sample_rate
        || provenance.artifact_sha256 != model.sha256
        || provenance.artifact_size_bytes != model.size_bytes
        || provenance.catalog_sequence != catalog.sequence
        || provenance.catalog_sha256 != catalog.sha256
        || provenance.catalog_signing_key_id != catalog.signing_key_id
        || provenance.installed_at_unix_seconds > MAX_JSON_SAFE_INTEGER
        || !catalog_origin_is_valid
        || !installation_source_is_valid
    {
        return Err(format!(
            "installed model provenance does not match package {}",
            model.name
        ));
    }
    Ok(())
}

fn valid_provenance_url(value: &str) -> bool {
    if value.is_empty() || value.len() > 2048 || value.chars().any(char::is_control) {
        return false;
    }
    Url::parse(value).is_ok_and(|url| {
        matches!(url.scheme(), "http" | "https")
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
    })
}

fn provenance_directory(destination: &Path) -> Result<PathBuf, String> {
    Ok(model_parent(destination)?.join(".provenance"))
}

fn remove_provenance_state(destination: &Path) -> Result<bool, String> {
    let directory = provenance_directory(destination)?;
    reject_symlink(&directory)?;
    let entries = match std::fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(format!(
                "failed to read model provenance directory {}: {error}",
                directory.display()
            ))
        }
    };
    let mut removed = false;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "failed to read model provenance directory {}: {error}",
                directory.display()
            )
        })?;
        let path = entry.path();
        let metadata = entry.metadata().map_err(|error| {
            format!(
                "failed to inspect model provenance {}: {error}",
                path.display()
            )
        })?;
        if !metadata.is_file() || entry.file_type().is_ok_and(|kind| kind.is_symlink()) {
            return Err(format!(
                "refusing special entry in model provenance directory: {}",
                path.display()
            ));
        }
        removed |= remove_file_if_present(&path)?;
    }
    match std::fs::remove_dir(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "failed to remove model provenance directory {}: {error}",
                directory.display()
            ))
        }
    }
    Ok(removed)
}

fn provenance_path(model: &ModelSpec<'_>, destination: &Path) -> Result<PathBuf, String> {
    let catalog = model.catalog.as_ref().ok_or_else(|| {
        format!(
            "model {} is not represented by an authenticated catalog",
            model.name
        )
    })?;
    let directory = provenance_directory(destination)?;
    reject_symlink(&directory)?;
    Ok(directory.join(format!("{}.{}.json", model.sha256, catalog.sha256)))
}

fn cleanup_prepared_provenance(prepared: &PreparedProvenance) {
    if prepared.created {
        let _ = remove_file_if_present(&prepared.path);
        if let Some(parent) = prepared.path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn publish_model_file<C, P>(
    model: &ModelSpec<'_>,
    installation_source: ModelInstallationSource,
    source: &Path,
    destination: &Path,
    expected_size: u64,
    expected_sha256: &str,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<(), String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    if model.catalog.is_none() {
        return publish_file(
            source,
            destination,
            expected_size,
            expected_sha256,
            cancelled,
            progress,
        );
    }
    let (_, prepared) = prepare_provenance(model, destination, installation_source)?;
    let result = publish_file(
        source,
        destination,
        expected_size,
        expected_sha256,
        cancelled,
        progress,
    );
    if result.is_err() {
        cleanup_prepared_provenance(&prepared);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn publish_model_open_file<C, P>(
    model: &ModelSpec<'_>,
    installation_source: ModelInstallationSource,
    input: File,
    source: &Path,
    destination: &Path,
    expected_size: u64,
    expected_sha256: &str,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<(), String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    if model.catalog.is_none() {
        return publish_open_file(
            input,
            source,
            destination,
            expected_size,
            expected_sha256,
            cancelled,
            progress,
        );
    }
    let (_, prepared) = prepare_provenance(model, destination, installation_source)?;
    let result = publish_open_file(
        input,
        source,
        destination,
        expected_size,
        expected_sha256,
        cancelled,
        progress,
    );
    if result.is_err() {
        cleanup_prepared_provenance(&prepared);
    }
    result
}

fn publish_file<C, P>(
    source: &Path,
    destination: &Path,
    expected_size: u64,
    expected_sha256: &str,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<(), String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let input = open_existing_regular_file(source, "model source")?
        .ok_or_else(|| format!("failed to open {}: file not found", source.display()))?;
    publish_open_file(
        input,
        source,
        destination,
        expected_size,
        expected_sha256,
        cancelled,
        progress,
    )
}

#[allow(clippy::too_many_arguments)]
fn publish_open_file<C, P>(
    mut input: File,
    source: &Path,
    destination: &Path,
    expected_size: u64,
    expected_sha256: &str,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<(), String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    require_open_file_size(&input, source, expected_size)?;
    input
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind {}: {error}", source.display()))?;
    let mut output = AtomicOutput::new(destination)?;
    let mut copied = 0_u64;
    progress(0, Some(expected_size));
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        if cancelled() {
            return Err("cancelled".into());
        }
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", source.display()))?;
        if count == 0 {
            break;
        }
        let next_copied = copied
            .checked_add(count as u64)
            .ok_or_else(|| "model size overflow while staging".to_string())?;
        if next_copied > expected_size {
            return Err(format!(
                "model size exceeds catalog metadata while staging: expected {expected_size} bytes"
            ));
        }
        output
            .file_mut()
            .write_all(&buffer[..count])
            .map_err(|error| format!("failed to stage {}: {error}", destination.display()))?;
        copied = next_copied;
        progress(copied, Some(expected_size));
    }
    if copied != expected_size {
        return Err(format!(
            "model size mismatch while staging: expected {expected_size} bytes, got {copied}"
        ));
    }
    require_open_file_size(&input, source, expected_size)?;
    output.file_mut().flush().map_err(|error| {
        format!(
            "failed to flush staged model for {}: {error}",
            destination.display()
        )
    })?;
    require_open_file_size(output.file_mut(), destination, expected_size)?;
    let actual = sha256_open_file_exact(output.file_mut(), destination, expected_size)?;
    if actual != expected_sha256 {
        return Err(format!(
            "model checksum mismatch while staging: expected {expected_sha256}, got {actual}"
        ));
    }
    output.commit(CommitMode::Replace)
}

fn open_local_model(path: &Path) -> Result<File, String> {
    open_existing_regular_file(path, "local model source")?
        .ok_or_else(|| format!("failed to open {}: file not found", path.display()))
}

fn open_existing_regular_file(path: &Path, description: &str) -> Result<Option<File>, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("failed to open {}: {error}", path.display())),
    };
    require_regular_file(&file, path, description)?;
    Ok(Some(file))
}

fn open_file_length(file: &File, path: &Path) -> Result<u64, String> {
    file.metadata()
        .map(|metadata| metadata.len())
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))
}

fn require_open_file_size(file: &File, path: &Path, expected_size: u64) -> Result<(), String> {
    require_regular_file(file, path, "model source")?;
    let actual_size = open_file_length(file, path)?;
    if actual_size != expected_size {
        return Err(format!(
            "model size mismatch for {}: expected {expected_size} bytes, got {actual_size}",
            path.display()
        ));
    }
    Ok(())
}

fn sha256_open_file_exact(
    input: &mut File,
    path: &Path,
    expected_size: u64,
) -> Result<String, String> {
    require_open_file_size(input, path, expected_size)?;
    input
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut read = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        read = read
            .checked_add(count as u64)
            .ok_or_else(|| "model size overflow while hashing".to_string())?;
        if read > expected_size {
            return Err(format!(
                "model size exceeds catalog metadata while hashing: expected {expected_size} bytes"
            ));
        }
        digest.update(&buffer[..count]);
    }
    if read != expected_size {
        return Err(format!(
            "model size changed while hashing {}: expected {expected_size} bytes, read {read}",
            path.display()
        ));
    }
    require_open_file_size(input, path, expected_size)?;
    input
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("failed to rewind {}: {error}", path.display()))?;
    Ok(format!("{:x}", digest.finalize()))
}

fn file_matches(path: &Path, expected_size: u64, expected_sha256: &str) -> bool {
    let Ok(Some(mut input)) = open_existing_regular_file(path, "model source") else {
        return false;
    };
    sha256_open_file_exact(&mut input, path, expected_size)
        .is_ok_and(|actual| actual == expected_sha256)
}

fn acquire_lock<C>(destination: &Path, cancelled: &mut C) -> Result<File, String>
where
    C: FnMut() -> bool,
{
    let lock_path = model_lock_path(destination)?;
    let lock_directory = lock_path
        .parent()
        .ok_or_else(|| "invalid model lock path".to_string())?;
    reject_symlink(lock_directory)?;
    std::fs::create_dir_all(lock_directory).map_err(|error| {
        format!(
            "failed to create model lock directory {}: {error}",
            lock_directory.display()
        )
    })?;
    reject_symlink(lock_directory)?;
    reject_symlink(&lock_path)?;
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let lock = options
        .open(&lock_path)
        .map_err(|error| format!("failed to open model lock {}: {error}", lock_path.display()))?;
    require_regular_file(&lock, &lock_path, "model lock")?;
    loop {
        match FileExt::try_lock_exclusive(&lock) {
            Ok(()) => return Ok(lock),
            Err(error) if lock_is_contended(&error) => {
                if cancelled() {
                    return Err("cancelled".into());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                return Err(format!(
                    "failed to lock model cache {}: {error}",
                    lock_path.display()
                ))
            }
        }
    }
}

fn lock_is_contended(error: &std::io::Error) -> bool {
    error.kind() == ErrorKind::WouldBlock
        || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
}

fn open_partial(path: &Path, resumed: bool) -> Result<File, String> {
    reject_symlink(path)?;
    let mut options = OpenOptions::new();
    options
        .create(true)
        .read(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options
        .open(path)
        .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    require_regular_file(&file, path, "partial model download")?;
    Ok(file)
}

fn require_regular_file(file: &File, path: &Path, description: &str) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "{description} is not a regular file: {}",
            path.display()
        ));
    }
    Ok(())
}

fn model_lock_path(destination: &Path) -> Result<PathBuf, String> {
    let model_directory = model_parent(destination)?;
    let cache_directory = model_directory
        .parent()
        .ok_or_else(|| "invalid model cache path".to_string())?;
    let mut name = model_directory
        .file_name()
        .ok_or_else(|| "invalid model cache path".to_string())?
        .to_os_string();
    name.push(".lock");
    Ok(cache_directory.join(".locks").join(name))
}

fn model_parent(destination: &Path) -> Result<&Path, String> {
    destination
        .parent()
        .ok_or_else(|| "invalid model cache path".to_string())
}

pub(super) fn validate_model_storage_path(destination: &Path) -> Result<(), String> {
    let parent = model_parent(destination)?;
    let cache = parent
        .parent()
        .ok_or_else(|| "invalid model cache path".to_string())?;
    validate_model_directory(cache, "model cache")?;
    validate_model_directory(parent, "model package directory")?;
    Ok(())
}

fn ensure_model_parent(destination: &Path) -> Result<&Path, String> {
    let parent = model_parent(destination)?;
    let cache = parent
        .parent()
        .ok_or_else(|| "invalid model cache path".to_string())?;
    reject_symlink(cache)?;
    std::fs::create_dir_all(cache)
        .map_err(|error| format!("failed to create {}: {error}", cache.display()))?;
    require_model_directory(cache, "model cache")?;
    reject_symlink(parent)?;
    match std::fs::create_dir(parent) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(format!("failed to create {}: {error}", parent.display()));
        }
    }
    require_model_directory(parent, "model package directory")?;
    Ok(parent)
}

fn validate_model_directory(path: &Path, description: &str) -> Result<(), String> {
    reject_symlink(path)?;
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => Err(format!(
            "{description} is not a directory: {}",
            path.display()
        )),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to inspect {}: {error}", path.display())),
    }
}

fn require_model_directory(path: &Path, description: &str) -> Result<(), String> {
    reject_symlink(path)?;
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    if !metadata.is_dir() {
        return Err(format!(
            "{description} is not a directory: {}",
            path.display()
        ));
    }
    Ok(())
}

fn reject_symlink(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(format!(
            "refusing to use symbolic link for model state: {}",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to inspect {}: {error}", path.display())),
    }
}

fn reset_partial(partial: &Path, metadata: &Path) -> Result<(), String> {
    remove_file_if_present(partial)?;
    remove_file_if_present(metadata)?;
    remove_file_if_present(&sidecar(metadata, ".tmp"))?;
    Ok(())
}

fn remove_file_if_present(path: &Path) -> Result<bool, String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("failed to remove {}: {error}", path.display())),
    }
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn source_identity(source: &str) -> String {
    let mut digest = Sha256::new();
    // Credentials and expiring signed-query values may rotate while the
    // underlying object remains identical. Integrity and validators still
    // gate every append, so keep resume state bound to origin + path.
    digest.update(redact_url(source).as_bytes());
    format!("{:x}", digest.finalize())
}

/// Return a source suitable for diagnostics: userinfo, query, and fragment
/// are always discarded.
pub fn redact_url(value: &str) -> String {
    let Ok(mut url) = Url::parse(value) else {
        return "<invalid URL>".into();
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);
    url.to_string()
}

fn redacted_proxy(proxy: &ModelProxy) -> String {
    format!("{proxy:?}")
}

fn redact_proxy_url(value: &str) -> String {
    let value = value.trim();
    let candidate = if value.contains("://") {
        value.to_owned()
    } else {
        format!("http://{value}")
    };
    redact_url(&candidate)
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn read_nonempty_environment<F>(read_environment: &mut F, name: &str) -> Option<String>
where
    F: FnMut(&str) -> Option<String>,
{
    read_environment(name).filter(|value| !value.trim().is_empty())
}

fn parse_env_bool(name: &str, value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "invalid {name}: expected 1/0, true/false, yes/no, or on/off"
        )),
    }
}

#[cfg(test)]
mod tests;
