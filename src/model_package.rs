//! Signed, self-describing runtime packages for custom waveform models.
//!
//! A `.dmp` package is a length-delimited container, not an archive. Its
//! Minisign-authenticated manifest binds the exact ONNX and license bytes and
//! declares the frontend, tensor, accelerator, and resource contracts that are
//! checked again when the graph is prepared. The caller supplies the trusted
//! public key separately; embedding a replaceable key in the package would not
//! establish trust.

use base64::Engine as _;
use minisign_verify::{PublicKey, Signature};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::{AcceleratorRuntime, AtomicOutput, CommitMode, OnnxModelConfig};

pub const RUNTIME_MODEL_PACKAGE_SCHEMA: &str = "denoize-runtime-model-package-v1";
pub const RUNTIME_MODEL_PACKAGE_VERSION: u32 = 1;

const PACKAGE_MAGIC: &[u8] = b"denoize-runtime-model-package-v1\n";
const HEADER_LENGTH_FIELDS: u64 = 4 * 8;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_SIGNATURE_BYTES: u64 = 16 * 1024;
const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_LICENSE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PUBLIC_KEY_BYTES: u64 = 64 * 1024;
const MAX_CONTRACT_MEMORY_BYTES: u64 = 1024 * 1024 * 1024 * 1024;
const HASH_BUFFER_BYTES: usize = 64 * 1024;
const MAX_FIXED_TENSOR_SAMPLES: u64 = 1 << 40;

const RUNTIME_KIND: &str = "onnx-waveform-v1";
const FRONTEND_CHANNEL_MAPPING: &str = "independent-mono-v1";
const FRONTEND_NORMALIZATION: &str = "pcm-f32-minus-one-to-one-v1";
const FRONTEND_RESAMPLING: &str = "bandlimited-waveform-v1";
const FRONTEND_DURATION: &str = "preserve-input-frames-v1";
const TENSOR_ELEMENT_TYPE: &str = "float32";

/// Exact file identity recorded by a package manifest.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelFileContract {
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

/// License notice carried beside the model bytes.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelLicenseContract {
    pub spdx: String,
    pub file: RuntimeModelFileContract,
}

/// Runtime adapter and rate consumed by the package.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelRuntimeContract {
    pub kind: String,
    pub sample_rate_hz: u32,
}

/// Audio-domain transformations applied around inference.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelFrontendContract {
    pub channel_mapping: String,
    pub normalization: String,
    pub resampling: String,
    pub duration: String,
}

/// Waveform tensor shape declared by the package.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelTensorContract {
    pub element_type: String,
    /// `batch-samples` or `batch-channels-samples`.
    pub layout: String,
    pub fixed_input_samples: Option<u64>,
    pub fixed_output_samples: Option<u64>,
}

/// Conservative package-specific admission values.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelResourceContract {
    pub max_session_memory_bytes: u64,
    pub max_worker_memory_bytes: u64,
    pub max_gpu_session_memory_bytes: u64,
    pub max_gpu_worker_memory_bytes: u64,
    /// Concrete runtimes accepted by the package: `cpu`, `metal`, or `cuda`.
    pub accelerators: Vec<String>,
}

/// Signed manifest embedded in a `.dmp` package.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelPackageManifest {
    pub schema: String,
    pub format_version: u32,
    pub package_id: String,
    pub package_revision: String,
    pub signing_key_id: String,
    pub runtime: RuntimeModelRuntimeContract,
    pub frontend: RuntimeModelFrontendContract,
    pub tensor: RuntimeModelTensorContract,
    pub resources: RuntimeModelResourceContract,
    pub model: RuntimeModelFileContract,
    pub license: RuntimeModelLicenseContract,
}

/// Authenticated metadata returned without parsing the ONNX graph.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeModelPackageInfo {
    pub format_version: u32,
    pub package_sha256: String,
    pub size_bytes: u64,
    pub package_id: String,
    pub package_revision: String,
    pub signing_key_id: String,
    pub sample_rate_hz: u32,
    pub tensor_layout: String,
    pub fixed_input_samples: Option<u64>,
    pub fixed_output_samples: Option<u64>,
    pub model_filename: String,
    pub model_sha256: String,
    pub model_size_bytes: u64,
    pub license_filename: String,
    pub license_sha256: String,
    pub license_size_bytes: u64,
    pub license_spdx: String,
    pub max_session_memory_bytes: u64,
    pub max_worker_memory_bytes: u64,
    pub max_gpu_session_memory_bytes: u64,
    pub max_gpu_worker_memory_bytes: u64,
    pub accelerators: Vec<String>,
}

/// One verified package retained as an immutable runtime configuration.
///
/// Opening streams and hashes every component without retaining the model in
/// memory. Session preparation later reopens the regular file, requires the
/// complete package fingerprint to remain identical, and parses only the
/// authenticated model range.
#[derive(Clone, Eq, PartialEq)]
pub struct RuntimeModelPackage {
    package_path: PathBuf,
    public_key_path: PathBuf,
    package_sha256: String,
    package_size_bytes: u64,
    public_key_sha256: String,
    manifest: RuntimeModelPackageManifest,
    model_offset: u64,
    license_offset: u64,
}

/// A bounded package component reader that authenticates the bytes it returns.
///
/// Callers must either read through EOF or call [`Self::finish`]. A digest
/// mismatch is reported as an I/O error before the component is accepted.
pub struct RuntimeModelPackageReader {
    inner: BufReader<File>,
    remaining: u64,
    expected_sha256: String,
    hasher: Option<Sha256>,
    failed: bool,
}

impl RuntimeModelPackageReader {
    /// Drain any unread bytes and complete component authentication.
    pub fn finish(&mut self) -> std::io::Result<()> {
        let mut buffer = [0_u8; HASH_BUFFER_BYTES];
        while self.read(&mut buffer)? != 0 {}
        Ok(())
    }

    fn validate_complete(&mut self) -> std::io::Result<()> {
        let Some(hasher) = self.hasher.take() else {
            return if self.failed {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "runtime model package component authentication failed",
                ))
            } else {
                Ok(())
            };
        };
        let observed = format!("{:x}", hasher.finalize());
        if observed != self.expected_sha256 {
            self.failed = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "runtime model package component changed after verification",
            ));
        }
        Ok(())
    }
}

impl Read for RuntimeModelPackageReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if self.failed {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "runtime model package component authentication failed",
            ));
        }
        if self.remaining == 0 {
            self.validate_complete()?;
            return Ok(0);
        }
        if output.is_empty() {
            return Ok(0);
        }
        let limit =
            usize::try_from(self.remaining.min(output.len() as u64)).unwrap_or(output.len());
        let count = self.inner.read(&mut output[..limit])?;
        if count == 0 {
            self.failed = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "runtime model package component is truncated",
            ));
        }
        if let Some(hasher) = self.hasher.as_mut() {
            hasher.update(&output[..count]);
        }
        self.remaining -= count as u64;
        if self.remaining == 0 {
            self.validate_complete()?;
        }
        Ok(count)
    }
}

impl std::fmt::Debug for RuntimeModelPackage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeModelPackage")
            .field("package_path", &self.package_path)
            .field("public_key_path", &self.public_key_path)
            .field("package_sha256", &self.package_sha256)
            .field("package_id", &self.manifest.package_id)
            .field("package_revision", &self.manifest.package_revision)
            .finish_non_exhaustive()
    }
}

impl RuntimeModelPackage {
    /// Authenticate a regular-file package against a separately supplied
    /// Minisign key.
    ///
    /// This validates framing, signed contracts, and component identities but
    /// deliberately does not parse the ONNX graph. Backend session preparation
    /// performs that graph/tensor-contract check.
    pub fn open(
        package_path: impl AsRef<Path>,
        public_key_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let package_path = package_path.as_ref();
        let public_key_path = public_key_path.as_ref();
        let (public_key_bytes, public_key_sha256) = read_bounded_regular_file(
            public_key_path,
            "runtime model public key",
            MAX_PUBLIC_KEY_BYTES,
        )?;
        let public_key_text = std::str::from_utf8(&public_key_bytes)
            .map_err(|_| "runtime model public key is not UTF-8".to_string())?;
        let parsed_key = parse_public_key(public_key_text)?;
        let prepared = prepare_package(package_path, &parsed_key, |bytes, signature, key| {
            verify_manifest_signature(bytes, signature, key)
        })?;
        Ok(Self {
            package_path: package_path.to_path_buf(),
            public_key_path: public_key_path.to_path_buf(),
            package_sha256: prepared.info.package_sha256,
            package_size_bytes: prepared.info.size_bytes,
            public_key_sha256,
            manifest: prepared.manifest,
            model_offset: prepared.model_offset,
            license_offset: prepared.license_offset,
        })
    }

    #[must_use]
    pub fn package_path(&self) -> &Path {
        &self.package_path
    }

    #[must_use]
    pub fn public_key_path(&self) -> &Path {
        &self.public_key_path
    }

    #[must_use]
    pub fn package_sha256(&self) -> &str {
        &self.package_sha256
    }

    #[must_use]
    pub fn public_key_sha256(&self) -> &str {
        &self.public_key_sha256
    }

    #[must_use]
    pub fn manifest(&self) -> &RuntimeModelPackageManifest {
        &self.manifest
    }

    #[must_use]
    pub(crate) fn model_config(&self) -> OnnxModelConfig {
        OnnxModelConfig {
            path: self.package_path.clone(),
            sample_rate: self.manifest.runtime.sample_rate_hz,
        }
    }

    #[must_use]
    pub fn info(&self) -> RuntimeModelPackageInfo {
        package_info(
            &self.manifest,
            self.package_sha256.clone(),
            self.package_size_bytes,
        )
    }

    #[must_use]
    pub fn supports_accelerator(&self, runtime: AcceleratorRuntime) -> bool {
        self.manifest
            .resources
            .accelerators
            .iter()
            .any(|name| name == runtime.name())
    }

    #[cfg(feature = "onnx")]
    pub(crate) fn open_model_reader(&self) -> Result<RuntimeModelPackageReader, String> {
        usize::try_from(self.manifest.model.size_bytes).map_err(|_| {
            "runtime model package model length cannot be represented on this platform".to_string()
        })?;
        self.open_verified_range(
            self.model_offset,
            self.manifest.model.size_bytes,
            &self.manifest.model.sha256,
        )
    }

    /// Reverify the package and open its authenticated license-notice range.
    pub fn open_license_reader(&self) -> Result<RuntimeModelPackageReader, String> {
        self.open_verified_range(
            self.license_offset,
            self.manifest.license.file.size_bytes,
            &self.manifest.license.file.sha256,
        )
    }

    fn open_verified_range(
        &self,
        offset: u64,
        length: u64,
        expected_sha256: &str,
    ) -> Result<RuntimeModelPackageReader, String> {
        let (mut file, package_length) =
            crate::input::open_regular_file(&self.package_path, "runtime model package")?;
        if package_length != self.package_size_bytes {
            return Err(format!(
                "runtime model package changed after verification: {}",
                self.package_path.display()
            ));
        }
        let observed = sha256_open_file(&mut file, &self.package_path)?;
        if observed != self.package_sha256 {
            return Err(format!(
                "runtime model package changed after verification: {}",
                self.package_path.display()
            ));
        }
        file.seek(SeekFrom::Start(offset)).map_err(|error| {
            format!(
                "seek runtime model package {}: {error}",
                self.package_path.display()
            )
        })?;
        Ok(RuntimeModelPackageReader {
            inner: BufReader::with_capacity(HASH_BUFFER_BYTES, file),
            remaining: length,
            expected_sha256: expected_sha256.to_string(),
            hasher: Some(Sha256::new()),
            failed: false,
        })
    }

    #[cfg(all(test, feature = "onnx"))]
    pub(crate) fn for_onnx_contract_test(
        model_path: PathBuf,
        tensor: RuntimeModelTensorContract,
    ) -> Self {
        let (mut model, model_size_bytes) =
            crate::input::open_regular_file(&model_path, "test runtime model").unwrap();
        let model_sha256 = sha256_open_file(&mut model, &model_path).unwrap();
        let package_id = "test.runtime-model".to_string();
        let manifest = RuntimeModelPackageManifest {
            schema: RUNTIME_MODEL_PACKAGE_SCHEMA.into(),
            format_version: RUNTIME_MODEL_PACKAGE_VERSION,
            package_id,
            package_revision: "1".into(),
            signing_key_id: "0000000000000001".into(),
            runtime: RuntimeModelRuntimeContract {
                kind: RUNTIME_KIND.into(),
                sample_rate_hz: 16_000,
            },
            frontend: RuntimeModelFrontendContract {
                channel_mapping: FRONTEND_CHANNEL_MAPPING.into(),
                normalization: FRONTEND_NORMALIZATION.into(),
                resampling: FRONTEND_RESAMPLING.into(),
                duration: FRONTEND_DURATION.into(),
            },
            tensor,
            resources: RuntimeModelResourceContract {
                max_session_memory_bytes: crate::estimate_model_session_bytes(model_size_bytes)
                    .unwrap(),
                max_worker_memory_bytes: 0,
                max_gpu_session_memory_bytes: 0,
                max_gpu_worker_memory_bytes: 0,
                accelerators: vec!["cpu".into()],
            },
            model: RuntimeModelFileContract {
                filename: model_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                size_bytes: model_size_bytes,
                sha256: model_sha256.clone(),
            },
            license: RuntimeModelLicenseContract {
                spdx: "MIT".into(),
                file: RuntimeModelFileContract {
                    filename: "LICENSE".into(),
                    size_bytes: 1,
                    sha256: format!("{:x}", Sha256::digest(b"x")),
                },
            },
        };
        Self {
            package_path: model_path,
            public_key_path: PathBuf::from("test.pub"),
            package_sha256: model_sha256.clone(),
            package_size_bytes: model_size_bytes,
            public_key_sha256: format!("{:x}", Sha256::digest(b"test key")),
            manifest,
            model_offset: 0,
            license_offset: model_size_bytes,
        }
    }

    #[cfg(all(test, feature = "onnx"))]
    pub(crate) fn with_resources_for_test(
        mut self,
        resources: RuntimeModelResourceContract,
    ) -> Self {
        self.manifest.resources = resources;
        self
    }
}

#[derive(Debug)]
struct PreparedPackage {
    info: RuntimeModelPackageInfo,
    manifest: RuntimeModelPackageManifest,
    model_offset: u64,
    license_offset: u64,
}

/// Authenticate a package without retaining runtime configuration or parsing
/// its ONNX graph.
pub fn inspect_runtime_model_package(
    package_path: impl AsRef<Path>,
    public_key_path: impl AsRef<Path>,
) -> Result<RuntimeModelPackageInfo, String> {
    RuntimeModelPackage::open(package_path, public_key_path).map(|package| package.info())
}

/// Assemble a deterministic package from an already signed manifest.
pub fn build_runtime_model_package(
    output: impl AsRef<Path>,
    manifest_path: impl AsRef<Path>,
    signature_path: impl AsRef<Path>,
    public_key_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
    license_path: impl AsRef<Path>,
) -> Result<RuntimeModelPackageInfo, String> {
    let output = output.as_ref();
    let manifest_path = manifest_path.as_ref();
    let signature_path = signature_path.as_ref();
    let public_key_path = public_key_path.as_ref();
    let model_path = model_path.as_ref();
    let license_path = license_path.as_ref();

    let (manifest_bytes, _) =
        read_bounded_regular_file(manifest_path, "runtime model manifest", MAX_MANIFEST_BYTES)?;
    let (signature_bytes, _) = read_bounded_regular_file(
        signature_path,
        "runtime model manifest signature",
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
    let manifest = parse_and_validate_manifest(&manifest_bytes, public_key.key_id.as_str())?;

    require_filename(model_path, &manifest.model.filename, "runtime model")?;
    require_filename(
        license_path,
        &manifest.license.file.filename,
        "runtime model license",
    )?;
    let (mut model, model_len) =
        crate::input::open_regular_file(model_path, "runtime model component")?;
    require_component_identity(
        &mut model,
        model_path,
        model_len,
        &manifest.model,
        "runtime model",
    )?;
    let (mut license, license_len) =
        crate::input::open_regular_file(license_path, "runtime model license component")?;
    require_component_identity(
        &mut license,
        license_path,
        license_len,
        &manifest.license.file,
        "runtime model license",
    )?;

    let mut staged = AtomicOutput::new(output)?;
    staged
        .file_mut()
        .write_all(PACKAGE_MAGIC)
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
        .and_then(|_| staged.file_mut().write_all(&model_len.to_be_bytes()))
        .and_then(|_| staged.file_mut().write_all(&license_len.to_be_bytes()))
        .and_then(|_| staged.file_mut().write_all(&manifest_bytes))
        .and_then(|_| staged.file_mut().write_all(&signature_bytes))
        .map_err(|error| format!("write runtime model package {}: {error}", output.display()))?;
    copy_exact_component(&mut model, model_path, staged.file_mut(), &manifest.model)?;
    copy_exact_component(
        &mut license,
        license_path,
        staged.file_mut(),
        &manifest.license.file,
    )?;
    staged
        .file_mut()
        .flush()
        .map_err(|error| format!("flush runtime model package {}: {error}", output.display()))?;
    let size = staged
        .file_mut()
        .metadata()
        .map_err(|error| format!("inspect staged runtime model package: {error}"))?
        .len();
    let prepared = prepare_open_package(
        staged.file_mut(),
        output,
        size,
        &public_key,
        |bytes, signature, key| verify_manifest_signature(bytes, signature, key),
    )?;
    staged.commit(CommitMode::NoClobber)?;
    Ok(prepared.info)
}

#[derive(Clone)]
struct ParsedPublicKey {
    key: PublicKey,
    key_id: String,
}

fn parse_public_key(text: &str) -> Result<ParsedPublicKey, String> {
    match parse_public_key_text(text) {
        Ok(key) => Ok(key),
        Err(direct_error) if !text.trim().contains(['\n', '\r']) => {
            let decoded = decode_outer_minisign_base64(text, "runtime model public key")?;
            parse_public_key_text(&decoded).map_err(|_| direct_error)
        }
        Err(error) => Err(error),
    }
}

fn parse_public_key_text(text: &str) -> Result<ParsedPublicKey, String> {
    let mut key_lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("untrusted comment:"));
    let base64 = key_lines
        .next()
        .ok_or_else(|| "runtime model public key has no key data".to_string())?;
    if key_lines.next().is_some() {
        return Err("runtime model public key contains multiple key data lines".into());
    }
    let key = PublicKey::from_base64(base64)
        .map_err(|error| format!("invalid runtime model public key: {error}"))?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(base64.as_bytes())
        .map_err(|_| "invalid runtime model public key base64".to_string())?;
    if decoded.len() != 42 {
        return Err("invalid runtime model public key length".into());
    }
    let encoded_key_id = u64::from_le_bytes(
        decoded[2..10]
            .try_into()
            .map_err(|_| "invalid runtime model public key id".to_string())?,
    );
    Ok(ParsedPublicKey {
        key,
        key_id: format!("{encoded_key_id:016X}"),
    })
}

fn verify_manifest_signature(
    manifest: &[u8],
    signature_bytes: &[u8],
    public_key: &ParsedPublicKey,
) -> Result<(), String> {
    let signature_text = std::str::from_utf8(signature_bytes)
        .map_err(|_| "runtime model manifest signature is not UTF-8".to_string())?;
    let signature = match Signature::decode(signature_text) {
        Ok(signature) => signature,
        Err(direct_error) if !signature_text.trim().contains(['\n', '\r']) => {
            let decoded =
                decode_outer_minisign_base64(signature_text, "runtime model manifest signature")?;
            Signature::decode(&decoded)
                .map_err(|_| format!("invalid runtime model manifest signature: {direct_error}"))?
        }
        Err(error) => return Err(format!("invalid runtime model manifest signature: {error}")),
    };
    public_key
        .key
        .verify(manifest, &signature, false)
        .map_err(|error| format!("runtime model manifest signature verification failed: {error}"))
}

fn decode_outer_minisign_base64(text: &str, description: &str) -> Result<String, String> {
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(text.trim())
        .map_err(|_| format!("invalid outer Base64 wrapper for {description}"))?;
    String::from_utf8(decoded).map_err(|_| format!("decoded {description} is not UTF-8"))
}

fn prepare_package<F>(
    path: &Path,
    public_key: &ParsedPublicKey,
    verify: F,
) -> Result<PreparedPackage, String>
where
    F: FnMut(&[u8], &[u8], &ParsedPublicKey) -> Result<(), String>,
{
    let (mut file, size) = crate::input::open_regular_file(path, "runtime model package")?;
    prepare_open_package(&mut file, path, size, public_key, verify)
}

fn prepare_open_package<F>(
    file: &mut File,
    path: &Path,
    size: u64,
    public_key: &ParsedPublicKey,
    mut verify: F,
) -> Result<PreparedPackage, String>
where
    F: FnMut(&[u8], &[u8], &ParsedPublicKey) -> Result<(), String>,
{
    let minimum = (PACKAGE_MAGIC.len() as u64)
        .checked_add(HEADER_LENGTH_FIELDS)
        .ok_or_else(|| "runtime model package size accounting overflow".to_string())?;
    if size < minimum {
        return Err("runtime model package is truncated".into());
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind runtime model package {}: {error}", path.display()))?;
    let mut package_hasher = Sha256::new();
    let mut magic = vec![0_u8; PACKAGE_MAGIC.len()];
    file.read_exact(&mut magic).map_err(|error| {
        format!(
            "read runtime model package magic {}: {error}",
            path.display()
        )
    })?;
    if magic != PACKAGE_MAGIC {
        return Err("runtime model package has an unsupported magic/version".into());
    }
    package_hasher.update(&magic);
    let manifest_len = read_u64(file, "manifest length", &mut package_hasher)?;
    let signature_len = read_u64(file, "signature length", &mut package_hasher)?;
    let model_len = read_u64(file, "model length", &mut package_hasher)?;
    let license_len = read_u64(file, "license length", &mut package_hasher)?;
    require_bounded_length(manifest_len, 1, MAX_MANIFEST_BYTES, "manifest")?;
    require_bounded_length(signature_len, 1, MAX_SIGNATURE_BYTES, "signature")?;
    require_bounded_length(model_len, 1, MAX_MODEL_BYTES, "model")?;
    require_bounded_length(license_len, 1, MAX_LICENSE_BYTES, "license")?;
    let expected_size = minimum
        .checked_add(manifest_len)
        .and_then(|value| value.checked_add(signature_len))
        .and_then(|value| value.checked_add(model_len))
        .and_then(|value| value.checked_add(license_len))
        .ok_or_else(|| "runtime model package size accounting overflow".to_string())?;
    if expected_size != size {
        return Err(format!(
            "runtime model package length mismatch: header declares {expected_size} bytes, file has {size}"
        ));
    }
    let manifest_bytes = read_exact_bounded(file, manifest_len, "runtime model manifest")?;
    package_hasher.update(&manifest_bytes);
    let signature_bytes = read_exact_bounded(file, signature_len, "runtime model signature")?;
    package_hasher.update(&signature_bytes);
    verify(&manifest_bytes, &signature_bytes, public_key)?;
    let manifest = parse_and_validate_manifest(&manifest_bytes, &public_key.key_id)?;
    if manifest.model.size_bytes != model_len {
        return Err("runtime model manifest model size does not match package framing".into());
    }
    if manifest.license.file.size_bytes != license_len {
        return Err("runtime model manifest license size does not match package framing".into());
    }
    let model_offset = minimum
        .checked_add(manifest_len)
        .and_then(|value| value.checked_add(signature_len))
        .ok_or_else(|| "runtime model package model offset overflow".to_string())?;
    let license_offset = model_offset
        .checked_add(model_len)
        .ok_or_else(|| "runtime model package license offset overflow".to_string())?;
    require_stream_hash(
        file,
        model_len,
        &manifest.model.sha256,
        "runtime model",
        &mut package_hasher,
    )?;
    require_stream_hash(
        file,
        license_len,
        &manifest.license.file.sha256,
        "runtime model license",
        &mut package_hasher,
    )?;
    let package_sha256 = format!("{:x}", package_hasher.finalize());
    Ok(PreparedPackage {
        info: package_info(&manifest, package_sha256, size),
        manifest,
        model_offset,
        license_offset,
    })
}

fn parse_and_validate_manifest(
    bytes: &[u8],
    expected_key_id: &str,
) -> Result<RuntimeModelPackageManifest, String> {
    let manifest: RuntimeModelPackageManifest = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid runtime model manifest JSON: {error}"))?;
    validate_manifest(&manifest, expected_key_id)?;
    Ok(manifest)
}

fn validate_manifest(
    manifest: &RuntimeModelPackageManifest,
    expected_key_id: &str,
) -> Result<(), String> {
    if manifest.schema != RUNTIME_MODEL_PACKAGE_SCHEMA
        || manifest.format_version != RUNTIME_MODEL_PACKAGE_VERSION
    {
        return Err("runtime model manifest has an unsupported schema/version".into());
    }
    validate_identifier(&manifest.package_id, "package id")?;
    validate_identifier(&manifest.package_revision, "package revision")?;
    if !valid_key_id(&manifest.signing_key_id) {
        return Err("runtime model manifest has an invalid signing key id".into());
    }
    if manifest.signing_key_id != expected_key_id {
        return Err("runtime model manifest signing key id does not match the trusted key".into());
    }
    if manifest.runtime.kind != RUNTIME_KIND {
        return Err(format!(
            "unsupported runtime model adapter: {}",
            manifest.runtime.kind
        ));
    }
    if manifest.runtime.sample_rate_hz == 0
        || manifest.runtime.sample_rate_hz > crate::config::MAX_SAMPLE_RATE
    {
        return Err("runtime model sample rate is outside 1..=768000 Hz".into());
    }
    if manifest.frontend.channel_mapping != FRONTEND_CHANNEL_MAPPING
        || manifest.frontend.normalization != FRONTEND_NORMALIZATION
        || manifest.frontend.resampling != FRONTEND_RESAMPLING
        || manifest.frontend.duration != FRONTEND_DURATION
    {
        return Err("runtime model package declares an unsupported frontend contract".into());
    }
    if manifest.tensor.element_type != TENSOR_ELEMENT_TYPE {
        return Err("runtime model package tensor element type must be float32".into());
    }
    if !matches!(
        manifest.tensor.layout.as_str(),
        "batch-samples" | "batch-channels-samples"
    ) {
        return Err("runtime model package has an unsupported tensor layout".into());
    }
    validate_fixed_samples(manifest.tensor.fixed_input_samples, "input")?;
    validate_fixed_samples(manifest.tensor.fixed_output_samples, "output")?;
    validate_file_contract(&manifest.model, MAX_MODEL_BYTES, "runtime model")?;
    validate_file_contract(
        &manifest.license.file,
        MAX_LICENSE_BYTES,
        "runtime model license",
    )?;
    validate_spdx(&manifest.license.spdx)?;
    validate_resources(&manifest.resources, manifest.model.size_bytes)?;
    Ok(())
}

fn validate_resources(
    resources: &RuntimeModelResourceContract,
    model_bytes: u64,
) -> Result<(), String> {
    for (value, name) in [
        (resources.max_session_memory_bytes, "session memory"),
        (resources.max_worker_memory_bytes, "worker memory"),
        (resources.max_gpu_session_memory_bytes, "GPU session memory"),
        (resources.max_gpu_worker_memory_bytes, "GPU worker memory"),
    ] {
        if value > MAX_CONTRACT_MEMORY_BYTES {
            return Err(format!(
                "runtime model {name} contract exceeds the 1 TiB limit"
            ));
        }
    }
    let baseline = crate::estimate_model_session_bytes(model_bytes)?;
    if resources.max_session_memory_bytes < baseline {
        return Err(format!(
            "runtime model session memory contract {} is below the conservative {baseline}-byte baseline",
            resources.max_session_memory_bytes
        ));
    }
    if resources.accelerators.is_empty() {
        return Err("runtime model package must declare at least one accelerator".into());
    }
    let mut unique = HashSet::new();
    for accelerator in &resources.accelerators {
        if !matches!(accelerator.as_str(), "cpu" | "metal" | "cuda") {
            return Err(format!(
                "runtime model package declares unknown accelerator {accelerator}"
            ));
        }
        if !unique.insert(accelerator.as_str()) {
            return Err(format!(
                "runtime model package repeats accelerator {accelerator}"
            ));
        }
    }
    if !unique.contains("cpu") {
        return Err("runtime model package must retain CPU compatibility".into());
    }
    if unique.len() > 1 {
        let gpu_baseline = crate::estimate_gpu_session_bytes(model_bytes)?;
        if resources.max_gpu_session_memory_bytes < gpu_baseline {
            return Err(format!(
                "runtime model GPU memory contract {} is below the conservative {gpu_baseline}-byte baseline",
                resources.max_gpu_session_memory_bytes
            ));
        }
    }
    Ok(())
}

fn validate_file_contract(
    contract: &RuntimeModelFileContract,
    maximum: u64,
    description: &str,
) -> Result<(), String> {
    validate_filename(&contract.filename, description)?;
    require_bounded_length(contract.size_bytes, 1, maximum, description)?;
    if !valid_sha256(&contract.sha256) {
        return Err(format!("{description} has an invalid SHA-256"));
    }
    Ok(())
}

fn validate_filename(filename: &str, description: &str) -> Result<(), String> {
    if filename.is_empty()
        || filename.chars().count() > 255
        || filename.contains(['/', '\\', '\0'])
        || filename.chars().any(char::is_control)
        || matches!(filename, "." | "..")
    {
        return Err(format!("{description} filename is invalid"));
    }
    Ok(())
}

fn validate_identifier(value: &str, description: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'+' | b'-'))
    {
        return Err(format!("runtime model {description} is invalid"));
    }
    Ok(())
}

fn validate_spdx(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'.' | b'_' | b'+' | b'-' | b':' | b'(' | b')' | b' ')
        })
    {
        return Err("runtime model license has an invalid SPDX expression".into());
    }
    Ok(())
}

fn validate_fixed_samples(value: Option<u64>, description: &str) -> Result<(), String> {
    if value.is_some_and(|value| value == 0 || value > MAX_FIXED_TENSOR_SAMPLES) {
        return Err(format!(
            "runtime model fixed {description} sample count is invalid"
        ));
    }
    Ok(())
}

fn package_info(
    manifest: &RuntimeModelPackageManifest,
    package_sha256: String,
    size_bytes: u64,
) -> RuntimeModelPackageInfo {
    RuntimeModelPackageInfo {
        format_version: manifest.format_version,
        package_sha256,
        size_bytes,
        package_id: manifest.package_id.clone(),
        package_revision: manifest.package_revision.clone(),
        signing_key_id: manifest.signing_key_id.clone(),
        sample_rate_hz: manifest.runtime.sample_rate_hz,
        tensor_layout: manifest.tensor.layout.clone(),
        fixed_input_samples: manifest.tensor.fixed_input_samples,
        fixed_output_samples: manifest.tensor.fixed_output_samples,
        model_filename: manifest.model.filename.clone(),
        model_sha256: manifest.model.sha256.clone(),
        model_size_bytes: manifest.model.size_bytes,
        license_filename: manifest.license.file.filename.clone(),
        license_sha256: manifest.license.file.sha256.clone(),
        license_size_bytes: manifest.license.file.size_bytes,
        license_spdx: manifest.license.spdx.clone(),
        max_session_memory_bytes: manifest.resources.max_session_memory_bytes,
        max_worker_memory_bytes: manifest.resources.max_worker_memory_bytes,
        max_gpu_session_memory_bytes: manifest.resources.max_gpu_session_memory_bytes,
        max_gpu_worker_memory_bytes: manifest.resources.max_gpu_worker_memory_bytes,
        accelerators: manifest.resources.accelerators.clone(),
    }
}

fn read_u64(
    file: &mut File,
    description: &str,
    package_hasher: &mut Sha256,
) -> Result<u64, String> {
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("read runtime model package {description}: {error}"))?;
    package_hasher.update(bytes);
    Ok(u64::from_be_bytes(bytes))
}

fn read_exact_bounded(file: &mut File, length: u64, description: &str) -> Result<Vec<u8>, String> {
    let length = usize::try_from(length)
        .map_err(|_| format!("{description} length cannot be represented on this platform"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| format!("unable to reserve {description} bytes"))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)
        .map_err(|error| format!("read {description}: {error}"))?;
    Ok(bytes)
}

fn require_bounded_length(
    value: u64,
    minimum: u64,
    maximum: u64,
    description: &str,
) -> Result<(), String> {
    if value < minimum || value > maximum {
        return Err(format!(
            "runtime model package {description} length {value} is outside {minimum}..={maximum} bytes"
        ));
    }
    Ok(())
}

fn require_stream_hash(
    file: &mut File,
    length: u64,
    expected: &str,
    description: &str,
    package_hasher: &mut Sha256,
) -> Result<(), String> {
    let mut hasher = Sha256::new();
    let mut remaining = length;
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    while remaining > 0 {
        let limit = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        file.read_exact(&mut buffer[..limit])
            .map_err(|error| format!("read {description}: {error}"))?;
        hasher.update(&buffer[..limit]);
        package_hasher.update(&buffer[..limit]);
        remaining -= limit as u64;
    }
    let observed = format!("{:x}", hasher.finalize());
    if observed != expected {
        return Err(format!("{description} SHA-256 does not match its manifest"));
    }
    Ok(())
}

fn sha256_open_file(file: &mut File, path: &Path) -> Result<String, String> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {} for hashing: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_bounded_regular_file(
    path: &Path,
    description: &str,
    maximum: u64,
) -> Result<(Vec<u8>, String), String> {
    let (mut file, length) = crate::input::open_regular_file(path, description)?;
    require_bounded_length(length, 1, maximum, description)?;
    let bytes = read_exact_bounded(&mut file, length, description)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    Ok((bytes, digest))
}

fn require_filename(path: &Path, expected: &str, description: &str) -> Result<(), String> {
    let observed = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{description} path has no UTF-8 filename"))?;
    if observed != expected {
        return Err(format!(
            "{description} filename {observed} does not match manifest filename {expected}"
        ));
    }
    Ok(())
}

fn require_component_identity(
    file: &mut File,
    path: &Path,
    length: u64,
    contract: &RuntimeModelFileContract,
    description: &str,
) -> Result<(), String> {
    if length != contract.size_bytes {
        return Err(format!(
            "{description} length {length} does not match manifest length {}",
            contract.size_bytes
        ));
    }
    let observed = sha256_open_file(file, path)?;
    if observed != contract.sha256 {
        return Err(format!("{description} SHA-256 does not match its manifest"));
    }
    Ok(())
}

fn copy_exact_component(
    source: &mut File,
    path: &Path,
    destination: &mut File,
    contract: &RuntimeModelFileContract,
) -> Result<(), String> {
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind package component {}: {error}", path.display()))?;
    let copied = std::io::copy(&mut source.take(contract.size_bytes), destination)
        .map_err(|error| format!("copy package component {}: {error}", path.display()))?;
    if copied != contract.size_bytes {
        return Err(format!(
            "package component {} changed while it was copied",
            path.display()
        ));
    }
    Ok(())
}

fn valid_key_id(value: &str) -> bool {
    value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_fixture_model() -> Vec<u8> {
        base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            include_str!("model_package/testdata/model.onnx.base64").trim(),
        )
        .unwrap()
    }

    fn file_contract(filename: &str, bytes: &[u8]) -> RuntimeModelFileContract {
        RuntimeModelFileContract {
            filename: filename.into(),
            size_bytes: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    }

    fn manifest(model: &[u8], license: &[u8]) -> RuntimeModelPackageManifest {
        let model_contract = file_contract("model.onnx", model);
        RuntimeModelPackageManifest {
            schema: RUNTIME_MODEL_PACKAGE_SCHEMA.into(),
            format_version: RUNTIME_MODEL_PACKAGE_VERSION,
            package_id: "example.voice-cleaner".into(),
            package_revision: "2026.08.22".into(),
            signing_key_id: "0000000000000001".into(),
            runtime: RuntimeModelRuntimeContract {
                kind: RUNTIME_KIND.into(),
                sample_rate_hz: 48_000,
            },
            frontend: RuntimeModelFrontendContract {
                channel_mapping: FRONTEND_CHANNEL_MAPPING.into(),
                normalization: FRONTEND_NORMALIZATION.into(),
                resampling: FRONTEND_RESAMPLING.into(),
                duration: FRONTEND_DURATION.into(),
            },
            tensor: RuntimeModelTensorContract {
                element_type: TENSOR_ELEMENT_TYPE.into(),
                layout: "batch-samples".into(),
                fixed_input_samples: Some(16_000),
                fixed_output_samples: Some(16_000),
            },
            resources: RuntimeModelResourceContract {
                max_session_memory_bytes: crate::estimate_model_session_bytes(
                    model_contract.size_bytes,
                )
                .unwrap(),
                max_worker_memory_bytes: 8 * 1024 * 1024,
                max_gpu_session_memory_bytes: 0,
                max_gpu_worker_memory_bytes: 0,
                accelerators: vec!["cpu".into()],
            },
            model: model_contract,
            license: RuntimeModelLicenseContract {
                spdx: "Apache-2.0".into(),
                file: file_contract("LICENSE.txt", license),
            },
        }
    }

    fn test_key() -> ParsedPublicKey {
        parse_public_key("RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3").unwrap()
    }

    fn catalog_test_key() -> ParsedPublicKey {
        parse_public_key("RWRGXBPWng5f30bcoLrI1zJw2RyznBVNqkqjkCVztHv9cjqT3UAwuw1W").unwrap()
    }

    fn write_test_package(
        path: &Path,
        manifest: &RuntimeModelPackageManifest,
        model: &[u8],
        license: &[u8],
    ) {
        let manifest = serde_json::to_vec(manifest).unwrap();
        let signature = b"test signature bytes";
        let mut file = File::create(path).unwrap();
        file.write_all(PACKAGE_MAGIC).unwrap();
        file.write_all(&(manifest.len() as u64).to_be_bytes())
            .unwrap();
        file.write_all(&(signature.len() as u64).to_be_bytes())
            .unwrap();
        file.write_all(&(model.len() as u64).to_be_bytes()).unwrap();
        file.write_all(&(license.len() as u64).to_be_bytes())
            .unwrap();
        file.write_all(&manifest).unwrap();
        file.write_all(signature).unwrap();
        file.write_all(model).unwrap();
        file.write_all(license).unwrap();
    }

    #[test]
    fn length_delimited_package_verifies_components_and_contracts() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.dmp");
        let model = b"self-contained model bytes";
        let license = b"license notice";
        let mut manifest = manifest(model, license);
        manifest.signing_key_id = test_key().key_id;
        write_test_package(&path, &manifest, model, license);

        let (mut file, size) = crate::input::open_regular_file(&path, "test package").unwrap();
        let prepared =
            prepare_open_package(&mut file, &path, size, &test_key(), |_, _, _| Ok(())).unwrap();

        assert_eq!(prepared.info.package_id, "example.voice-cleaner");
        assert_eq!(prepared.info.model_sha256, manifest.model.sha256);
        assert_eq!(prepared.info.license_spdx, "Apache-2.0");
        assert_eq!(prepared.info.accelerators, vec!["cpu"]);
        assert!(prepared.model_offset < size);
    }

    #[test]
    fn component_tampering_is_rejected_after_manifest_verification() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.dmp");
        let model = b"self-contained model bytes";
        let license = b"license notice";
        let mut manifest = manifest(model, license);
        manifest.signing_key_id = test_key().key_id;
        write_test_package(&path, &manifest, b"self-contained model byte!", license);

        let (mut file, size) = crate::input::open_regular_file(&path, "test package").unwrap();
        let error = prepare_open_package(&mut file, &path, size, &test_key(), |_, _, _| Ok(()))
            .unwrap_err();
        assert!(
            error.contains("model SHA-256") || error.contains("model size"),
            "{error}"
        );
    }

    #[test]
    fn framing_rejects_oversized_lengths_and_trailing_data_before_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.dmp");
        let mut file = File::create(&path).unwrap();
        file.write_all(PACKAGE_MAGIC).unwrap();
        file.write_all(&(MAX_MANIFEST_BYTES + 1).to_be_bytes())
            .unwrap();
        file.write_all(&1_u64.to_be_bytes()).unwrap();
        file.write_all(&1_u64.to_be_bytes()).unwrap();
        file.write_all(&1_u64.to_be_bytes()).unwrap();
        drop(file);

        let (mut file, size) = crate::input::open_regular_file(&path, "test package").unwrap();
        let error = prepare_open_package(&mut file, &path, size, &test_key(), |_, _, _| {
            panic!("oversized framing must fail before signature verification")
        })
        .unwrap_err();
        assert!(error.contains("manifest length"), "{error}");

        let model = b"model";
        let license = b"license";
        let mut contract = manifest(model, license);
        contract.signing_key_id = test_key().key_id;
        write_test_package(&path, &contract, model, license);
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"trailing").unwrap();
        drop(file);
        let (mut file, size) = crate::input::open_regular_file(&path, "test package").unwrap();
        let error = prepare_open_package(&mut file, &path, size, &test_key(), |_, _, _| Ok(()))
            .unwrap_err();
        assert!(error.contains("length mismatch"), "{error}");
    }

    #[test]
    fn signed_fixture_builds_opens_and_reproduces_identical_package_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let manifest_path = directory.path().join("manifest.json");
        let signature_path = directory.path().join("manifest.json.sig");
        let public_key_path = directory.path().join("minisign.pub");
        let model_path = directory.path().join("model.onnx");
        let license_path = directory.path().join("LICENSE.txt");
        std::fs::write(
            &manifest_path,
            include_bytes!("model_package/testdata/manifest.json"),
        )
        .unwrap();
        std::fs::write(
            &signature_path,
            include_bytes!("model_package/testdata/manifest.json.sig"),
        )
        .unwrap();
        std::fs::write(
            &public_key_path,
            include_bytes!("model_package/testdata/minisign.pub"),
        )
        .unwrap();
        std::fs::write(&model_path, signed_fixture_model()).unwrap();
        std::fs::write(&license_path, b"fixture license").unwrap();

        let first = directory.path().join("first.dmp");
        let second = directory.path().join("second.dmp");
        let info = build_runtime_model_package(
            &first,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &model_path,
            &license_path,
        )
        .unwrap();
        build_runtime_model_package(
            &second,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &model_path,
            &license_path,
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&first).unwrap(),
            std::fs::read(&second).unwrap()
        );
        let original = std::fs::read(&first).unwrap();
        let error = build_runtime_model_package(
            &first,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &model_path,
            &license_path,
        )
        .unwrap_err();
        assert!(error.contains("exists"), "{error}");
        assert_eq!(std::fs::read(&first).unwrap(), original);

        let package = RuntimeModelPackage::open(&first, &public_key_path).unwrap();
        assert_eq!(package.info(), info);
        assert_eq!(package.manifest().package_id, "denoize.test.signed-fixture");
        assert_eq!(package.manifest().license.spdx, "MIT");
        assert_eq!(package.manifest().resources.accelerators, vec!["cpu"]);
        let mut license = Vec::new();
        package
            .open_license_reader()
            .unwrap()
            .read_to_end(&mut license)
            .unwrap();
        assert_eq!(license, b"fixture license");
    }

    #[test]
    fn authenticated_component_reader_rejects_same_inode_mutation() {
        let directory = tempfile::tempdir().unwrap();
        let manifest_path = directory.path().join("manifest.json");
        let signature_path = directory.path().join("manifest.json.sig");
        let public_key_path = directory.path().join("minisign.pub");
        let model_path = directory.path().join("model.onnx");
        let license_path = directory.path().join("LICENSE.txt");
        let package_path = directory.path().join("model.dmp");
        std::fs::write(
            &manifest_path,
            include_bytes!("model_package/testdata/manifest.json"),
        )
        .unwrap();
        std::fs::write(
            &signature_path,
            include_bytes!("model_package/testdata/manifest.json.sig"),
        )
        .unwrap();
        std::fs::write(
            &public_key_path,
            include_bytes!("model_package/testdata/minisign.pub"),
        )
        .unwrap();
        std::fs::write(&model_path, signed_fixture_model()).unwrap();
        std::fs::write(&license_path, b"fixture license").unwrap();
        build_runtime_model_package(
            &package_path,
            &manifest_path,
            &signature_path,
            &public_key_path,
            &model_path,
            &license_path,
        )
        .unwrap();

        let package = RuntimeModelPackage::open(&package_path, &public_key_path).unwrap();
        let mut reader = package.open_license_reader().unwrap();
        let mut writable = std::fs::OpenOptions::new()
            .write(true)
            .open(&package_path)
            .unwrap();
        writable
            .seek(SeekFrom::Start(package.license_offset))
            .unwrap();
        writable.write_all(b"tampered notice").unwrap();
        writable.sync_all().unwrap();

        let error = reader.read_to_end(&mut Vec::new()).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("changed after verification"));
    }

    #[test]
    fn manifest_rejects_underreported_resources_and_unsafe_names() {
        let mut manifest = manifest(b"model", b"license");
        manifest.resources.max_session_memory_bytes = 1;
        assert!(validate_manifest(&manifest, "0000000000000001")
            .unwrap_err()
            .contains("below the conservative"));

        manifest.resources.max_session_memory_bytes =
            crate::estimate_model_session_bytes(manifest.model.size_bytes).unwrap();
        manifest.model.filename = "../model.onnx".into();
        assert!(validate_manifest(&manifest, "0000000000000001")
            .unwrap_err()
            .contains("filename"));

        manifest.model.filename = "model\n.onnx".into();
        assert!(validate_manifest(&manifest, "0000000000000001")
            .unwrap_err()
            .contains("filename"));

        manifest.model.filename = "model.onnx".into();
        manifest.license.file.filename = "C:portable-license-name".into();
        validate_manifest(&manifest, "0000000000000001").unwrap();

        manifest.license.spdx = "DocumentRef-vendor:LicenseRef-model AND MIT".into();
        validate_manifest(&manifest, "0000000000000001").unwrap();
    }

    #[test]
    fn minisign_legacy_vector_is_rejected_by_modern_policy() {
        let key = test_key();
        let signature = b"untrusted comment: signature from minisign secret key\nRWQf6LRCGA9i59SLOFxz6NxvASXDJeRtuZykwQepbDEGt87ig1BNpWaVWuNrm73YiIiJbq71Wi+dP9eKL8OC351vwIasSSbXxwA=\ntrusted comment: timestamp:1555779966\tfile:test\nQtKMXWyYcwdpZAlPF7tE2ENJkRd1ujvKjlj1m9RtHTBnZPa5WKU5uWRs5GoP5M/VqE81QFuMKI5k/SfNQUaOAA==\n";
        let error = verify_manifest_signature(b"test", signature, &key).unwrap_err();
        assert!(error.contains("Unexpected signature algorithm"), "{error}");
    }

    #[test]
    fn modern_minisign_vector_is_accepted() {
        let signature = base64::engine::general_purpose::STANDARD
            .decode(
                std::str::from_utf8(include_bytes!("models/testdata/catalog-seq2.json.sig"))
                    .unwrap()
                    .trim(),
            )
            .unwrap();
        verify_manifest_signature(
            include_bytes!("models/testdata/catalog-seq2.json"),
            &signature,
            &catalog_test_key(),
        )
        .unwrap();
    }

    #[test]
    fn outer_base64_minisign_wrappers_are_accepted() {
        let raw_key = "untrusted comment: catalog fixture\nRWRGXBPWng5f30bcoLrI1zJw2RyznBVNqkqjkCVztHv9cjqT3UAwuw1W\n";
        let wrapped_key = base64::engine::general_purpose::STANDARD.encode(raw_key);
        let key = parse_public_key(&wrapped_key).unwrap();
        let wrapped_signature =
            std::str::from_utf8(include_bytes!("models/testdata/catalog-seq2.json.sig")).unwrap();
        verify_manifest_signature(
            include_bytes!("models/testdata/catalog-seq2.json"),
            wrapped_signature.as_bytes(),
            &key,
        )
        .unwrap();

        let ambiguous = format!("{raw_key}{}\n", raw_key.lines().nth(1).unwrap());
        let error = parse_public_key(&ambiguous)
            .err()
            .expect("multiple public keys must fail");
        assert!(error.contains("multiple key data lines"), "{error}");
    }
}
