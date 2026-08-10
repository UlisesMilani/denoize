//! Versioned external-model manifest and verified local cache.

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
use std::time::Duration;
use url::Url;

const PARTIAL_METADATA_VERSION: u32 = 1;
const MAX_PARTIAL_METADATA_BYTES: u64 = 64 * 1024;
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
    /// Refuse every network request. A verified destination or completed
    /// verified `.part` file can still be used.
    pub offline: bool,
    /// Override the pinned manifest URL, for example with an authenticated
    /// mirror. Integrity remains pinned by the manifest size and SHA-256.
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

pub const MODELS: &[ModelInfo] = &[ModelInfo {
    name: "gtcrn-dns3",
    backend: "gtcrn",
    filename: "gtcrn_simple.onnx",
    url: "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/3862c44808dca492ea5a8a145d2dc2a1028d08c8/stream/onnx_models/gtcrn_simple.onnx",
    revision: "3862c44808dca492ea5a8a145d2dc2a1028d08c8",
    sha256: "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
    size_bytes: 535_190,
    license: "MIT",
    sample_rate: 16_000,
}];

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
    Ok(cache_dir()?.join(model.name).join(model.filename))
}

pub fn verify(model: &ModelInfo) -> Result<PathBuf, String> {
    verify_at(model, &path(model)?)
}

fn verify_at(model: &ModelInfo, destination: &Path) -> Result<PathBuf, String> {
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
    install_internal(model, options, false, &mut cancelled, &mut progress)
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
    let source = source.as_ref();
    let destination = path(model)?;
    let parent = model_parent(&destination)?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let lock = acquire_lock(&destination, &mut cancelled)?;
    let result = (|| {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
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
        publish_open_file(
            source_file,
            source,
            &destination,
            model.size_bytes,
            model.sha256,
            &mut cancelled,
            &mut progress,
        )?;
        Ok(destination.clone())
    })();
    drop(lock);
    result
}

/// Remove an installed model and all interrupted-download state.
pub fn remove(model: &ModelInfo) -> Result<bool, String> {
    let destination = path(model)?;
    let Some(parent) = destination.parent() else {
        return Err("invalid model cache path".into());
    };
    let mut never_cancelled = || false;
    let lock = acquire_lock(&destination, &mut never_cancelled)?;
    let partial = sidecar(&destination, ".part");
    let metadata = sidecar(&destination, ".part.meta");
    let removed = remove_file_if_present(&destination)?
        | remove_file_if_present(&partial)?
        | remove_file_if_present(&metadata)?
        | remove_file_if_present(&sidecar(&metadata, ".tmp"))?;
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
    install_internal(model, options, true, &mut cancelled, &mut progress)
}

fn install_internal<C, P>(
    model: &ModelInfo,
    options: &ModelDownloadOptions,
    force: bool,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<PathBuf, String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let destination = path(model)?;
    let parent = model_parent(&destination)?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let lock = acquire_lock(&destination, cancelled)?;
    let result = (|| {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        if !force {
            if let Ok(path) = verify_at(model, &destination) {
                return Ok(path);
            }
        }
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
                    publish_file(
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
            if let Ok(path) = verify_at(model, &destination) {
                return Ok(path);
            }
            return Err(format!(
                "offline mode: no verified model is available at {} (use `models install {} --from PATH`)",
                destination.display(), model.name
            ));
        }

        let raw_url = options.source_url.as_deref().unwrap_or(model.url);
        let source = Url::parse(raw_url).map_err(|_| "invalid model source URL".to_string())?;
        if !matches!(source.scheme(), "http" | "https") {
            return Err(
                "model source URL must use http or https; use --from for local files".into(),
            );
        }
        validate_authentication(&source, options.authentication.as_ref())?;
        let source_id = source_identity(raw_url);
        download(
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
        publish_file(
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
                return Err("model server sent more bytes than allowed by the manifest".into());
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
        return Err("partial model response exceeds the manifest size".into());
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
        return Err("partial metadata conflicts with the manifest size".into());
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
                "model size exceeds the manifest while staging: expected {expected_size} bytes"
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
                "model size exceeds the manifest while hashing: expected {expected_size} bytes"
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
