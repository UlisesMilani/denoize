//! Authenticated model-catalog loading and rollback protection.

mod trust;

pub use trust::{
    import_trust_root, recover_embedded_trust_root, reset_trust_time_floor, trust_root_status,
    TrustRootOrigin, TrustRootStatus,
};

use super::{
    cache_dir, open_existing_regular_file, parse_content_length, redact_url,
    request_with_redirects, validate_authentication, ModelDownloadOptions,
};
use crate::{AtomicOutput, CommitMode};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use minisign_verify::{PublicKey, Signature};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use url::Url;

const CATALOG_SCHEMA: &str = "denoize-model-catalog-v1";
const CATALOG_STATE_VERSION: u32 = 2;
const LEGACY_CATALOG_STATE_VERSION: u32 = 1;
const CATALOG_ENVELOPE_VERSION: u32 = 1;
const MAX_CATALOG_BYTES: u64 = 1024 * 1024;
const MAX_SIGNATURE_BYTES: u64 = 16 * 1024;
const MAX_ENVELOPE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_STATE_BYTES: u64 = 64 * 1024;
const MAX_MODELS: usize = 256;
const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const DEFAULT_CATALOG_URL: &str =
    "https://github.com/penguin425/denoize/releases/latest/download/denoize-model-catalog-v1.json";
const LOCAL_IMPORT_SOURCE: &str = "local-import";
const EMBEDDED_CATALOG: &[u8] = include_bytes!("../../models/catalog-v1.json");

/// Where the active catalog obtained its authenticated contents.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum CatalogOrigin {
    /// Catalog bytes embedded in the installed denoize binary.
    Embedded,
    /// Detached-minisign catalog accepted from a local import or HTTPS source.
    Signed { source: String },
}

impl<'de> Deserialize<'de> for CatalogOrigin {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            kind: String,
            source: Option<String>,
        }

        let wire = Wire::deserialize(deserializer)?;
        match (wire.kind.as_str(), wire.source) {
            ("embedded", None) => Ok(Self::Embedded),
            ("signed", Some(source)) => Ok(Self::Signed { source }),
            ("embedded", Some(_)) => Err(serde::de::Error::custom(
                "embedded catalog origin must not contain source",
            )),
            ("signed", None) => Err(serde::de::Error::missing_field("source")),
            _ => Err(serde::de::Error::unknown_variant(
                &wire.kind,
                &["embedded", "signed"],
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CatalogIdentity {
    pub sequence: u64,
    pub sha256: String,
    pub signing_key_id: String,
    pub signing_public_key_base64: String,
    pub issued_at_unix_seconds: Option<u64>,
    pub expires_at_unix_seconds: Option<u64>,
    pub trust_root_version: u64,
    pub origin: CatalogOrigin,
}

/// One package selected from a verified model catalog.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogModel {
    pub(crate) name: String,
    pub(crate) backend: String,
    pub(crate) filename: String,
    pub(crate) url: String,
    pub(crate) revision: String,
    pub(crate) sha256: String,
    pub(crate) size_bytes: u64,
    pub(crate) license: String,
    pub(crate) sample_rate: u32,
    pub(crate) catalog: CatalogIdentity,
}

impl CatalogModel {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn license(&self) -> &str {
        &self.license
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn catalog_sequence(&self) -> u64 {
        self.catalog.sequence
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog.sha256
    }

    pub fn catalog_signing_key_id(&self) -> &str {
        &self.catalog.signing_key_id
    }

    pub fn catalog_issued_at_unix_seconds(&self) -> Option<u64> {
        self.catalog.issued_at_unix_seconds
    }

    pub fn catalog_expires_at_unix_seconds(&self) -> Option<u64> {
        self.catalog.expires_at_unix_seconds
    }

    pub fn catalog_trust_root_version(&self) -> u64 {
        self.catalog.trust_root_version
    }

    pub fn catalog_origin(&self) -> &CatalogOrigin {
        &self.catalog.origin
    }
}

/// A validated catalog whose entries all share one authenticated identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelCatalog {
    identity: CatalogIdentity,
    models: Vec<CatalogModel>,
}

impl ModelCatalog {
    pub(crate) fn identity(&self) -> &CatalogIdentity {
        &self.identity
    }

    pub fn sequence(&self) -> u64 {
        self.identity.sequence
    }

    pub fn sha256(&self) -> &str {
        &self.identity.sha256
    }

    pub fn signing_key_id(&self) -> &str {
        &self.identity.signing_key_id
    }

    pub fn issued_at_unix_seconds(&self) -> Option<u64> {
        self.identity.issued_at_unix_seconds
    }

    pub fn expires_at_unix_seconds(&self) -> Option<u64> {
        self.identity.expires_at_unix_seconds
    }

    pub fn trust_root_version(&self) -> u64 {
        self.identity.trust_root_version
    }

    pub fn origin(&self) -> &CatalogOrigin {
        &self.identity.origin
    }

    pub fn models(&self) -> &[CatalogModel] {
        &self.models
    }

    /// Find an exact package name, or an unambiguous backend alias.
    pub fn find(&self, name: &str) -> Option<&CatalogModel> {
        self.models
            .iter()
            .find(|model| model.name == name)
            .or_else(|| {
                let mut matching = self.models.iter().filter(|model| model.backend == name);
                let model = matching.next()?;
                matching.next().is_none().then_some(model)
            })
    }
}

/// Human- and UI-facing status for the active catalog and rollback floor.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CatalogStatus {
    pub sequence: u64,
    pub sha256: String,
    pub signing_key_id: String,
    pub origin: CatalogOrigin,
    pub model_count: usize,
    pub highest_accepted_sequence: u64,
    pub cached_catalog_path: PathBuf,
    pub issued_at_unix_seconds: Option<u64>,
    pub expires_at_unix_seconds: Option<u64>,
    pub trust_root_version: u64,
    pub trust_root_sha256: String,
    pub trust_root_expires_at_unix_seconds: u64,
    pub trust_root_highest_observed_unix_seconds: Option<u64>,
    pub acquisition_allowed: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogDocument {
    schema: String,
    sequence: u64,
    signing_key_id: String,
    #[serde(default)]
    issued_at_unix_seconds: Option<u64>,
    #[serde(default)]
    expires_at_unix_seconds: Option<u64>,
    models: Vec<CatalogModelDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogModelDocument {
    name: String,
    backend: String,
    filename: String,
    url: String,
    revision: String,
    sha256: String,
    size_bytes: u64,
    license: String,
    sample_rate: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CatalogState {
    version: u32,
    highest_sequence: u64,
    catalog_sha256: String,
    signing_key_id: String,
    #[serde(default = "legacy_trust_root_version")]
    trust_root_version: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SignedCatalogEnvelope {
    version: u32,
    catalog_base64: String,
    signature: String,
    source: String,
}

const fn legacy_trust_root_version() -> u64 {
    1
}

#[derive(Clone, Copy)]
enum CatalogVerificationMode {
    Historical { accepted_root_version: u64 },
    Current { now: u64 },
}

/// Return the catalog shipped inside this exact denoize build.
pub fn embedded_catalog() -> ModelCatalog {
    let root = trust::embedded_trust_root();
    parse_catalog_with_root(
        EMBEDDED_CATALOG,
        CatalogOrigin::Embedded,
        &root,
        CatalogVerificationMode::Historical {
            accepted_root_version: root.version(),
        },
    )
    .expect("the embedded model catalog is validated by the test suite")
}

/// Load the active signed catalog, falling back only to an equivalent embedded
/// catalog that does not violate the persisted rollback floor.
pub fn active_catalog() -> Result<ModelCatalog, String> {
    validate_catalog_storage_path()?;
    let directory = catalog_directory()?;
    match std::fs::symlink_metadata(&directory) {
        Ok(_) => {
            let lock_destination = directory.join("catalog.json");
            let mut never_cancelled = || false;
            let lock = super::acquire_lock(&lock_destination, &mut never_cancelled)?;
            let result = (|| {
                let root = trust::load_active_trust_root_locked()?;
                load_active_catalog_locked(embedded_catalog(), &root)
            })();
            drop(lock);
            result
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let embedded = embedded_catalog();
            if embedded.sequence() > 1 {
                promote_embedded_catalog(embedded)
            } else {
                Ok(embedded)
            }
        }
        Err(error) => Err(format!(
            "failed to inspect model catalog directory {}: {error}",
            directory.display()
        )),
    }
}

fn load_active_catalog_locked(
    embedded: ModelCatalog,
    root: &trust::ActiveTrustRoot,
) -> Result<ModelCatalog, String> {
    let state = load_state()?;
    // A newer binary-embedded catalog supersedes an older authenticated
    // cache. Do not let obsolete cache corruption prevent that upgrade. A
    // missing rollback state remains fail-closed below so deleting state
    // cannot silently reactivate an older embedded catalog.
    if state
        .as_ref()
        .is_some_and(|state| state.highest_sequence < embedded.sequence())
    {
        write_catalog_state(&embedded)?;
        return Ok(embedded);
    }
    if state
        .as_ref()
        .is_some_and(|state| state_matches_catalog(state, &embedded))
    {
        return Ok(embedded);
    }
    let envelope = load_envelope()?;
    match (state, envelope) {
        (None, None) => {
            if embedded.sequence() > 1 {
                write_catalog_state(&embedded)?;
            }
            Ok(embedded)
        }
        (Some(state), None) => Err(format!(
            "model catalog sequence {} was accepted previously, but its signed cache is missing; re-import that sequence or a newer catalog",
            state.highest_sequence
        )),
        (None, Some(_)) => Err(
            "signed model catalog cache exists without rollback state; re-import the catalog".into(),
        ),
        (Some(state), Some(envelope)) => {
            validate_state(&state)?;
            let catalog_bytes = BASE64_STANDARD
                .decode(envelope.catalog_base64.as_bytes())
                .map_err(|_| "cached model catalog has invalid base64".to_string())?;
            if catalog_bytes.len() as u64 > MAX_CATALOG_BYTES {
                return Err("cached model catalog exceeds the 1 MiB limit".into());
            }
            let catalog = verify_signed_catalog_with_root(
                &catalog_bytes,
                envelope.signature.as_bytes(),
                CatalogOrigin::Signed {
                    source: envelope.source,
                },
                root,
                CatalogVerificationMode::Historical {
                    accepted_root_version: state.trust_root_version,
                },
            )?;
            if catalog.sequence() < state.highest_sequence {
                return Err(format!(
                    "refusing model catalog rollback from sequence {} to {}",
                    state.highest_sequence,
                    catalog.sequence()
                ));
            }
            if catalog.sequence() != state.highest_sequence
                || catalog.sha256() != state.catalog_sha256
                || catalog.signing_key_id() != state.signing_key_id
            {
                return Err("signed model catalog does not match persisted rollback state".into());
            }
            if catalog.sequence() < embedded.sequence() {
                return Err(format!(
                    "signed model catalog sequence {} predates embedded sequence {}",
                    catalog.sequence(),
                    embedded.sequence()
                ));
            }
            if catalog.sequence() == embedded.sequence()
                && (catalog.sha256() != embedded.sha256()
                    || catalog.signing_key_id() != embedded.signing_key_id())
            {
                return Err(format!(
                    "signed model catalog conflicts with embedded content at sequence {}",
                    embedded.sequence()
                ));
            }
            Ok(catalog)
        }
    }
}

pub(super) fn promote_embedded_catalog(embedded: ModelCatalog) -> Result<ModelCatalog, String> {
    ensure_catalog_directory()?;
    let lock_destination = catalog_directory()?.join("catalog.json");
    let mut never_cancelled = || false;
    let lock = super::acquire_lock(&lock_destination, &mut never_cancelled)?;
    let result = (|| {
        let root = trust::load_active_trust_root_locked()?;
        load_active_catalog_locked(embedded, &root)
    })();
    drop(lock);
    result
}

pub fn catalog_status() -> Result<CatalogStatus, String> {
    let catalog = active_catalog()?;
    let root = trust_root_status()?;
    let acquisition_allowed = catalog_acquisition_allowed(&catalog)?;
    let highest_accepted_sequence = catalog.sequence();
    Ok(CatalogStatus {
        sequence: catalog.sequence(),
        sha256: catalog.sha256().to_string(),
        signing_key_id: catalog.signing_key_id().to_string(),
        origin: catalog.origin().clone(),
        model_count: catalog.models().len(),
        highest_accepted_sequence,
        cached_catalog_path: envelope_path()?,
        issued_at_unix_seconds: catalog.issued_at_unix_seconds(),
        expires_at_unix_seconds: catalog.expires_at_unix_seconds(),
        trust_root_version: root.version,
        trust_root_sha256: root.sha256,
        trust_root_expires_at_unix_seconds: root.expires_at_unix_seconds,
        trust_root_highest_observed_unix_seconds: root.highest_observed_unix_seconds,
        acquisition_allowed,
    })
}

/// Verify and atomically activate a detached-minisign catalog from local
/// regular files. This is the supported air-gapped update path.
pub fn import_catalog(
    catalog_path: impl AsRef<Path>,
    signature_path: impl AsRef<Path>,
) -> Result<ModelCatalog, String> {
    let catalog_path = catalog_path.as_ref();
    let signature_path = signature_path.as_ref();
    let catalog_bytes = read_bounded_file(catalog_path, MAX_CATALOG_BYTES, "model catalog")?;
    let signature = read_bounded_file(
        signature_path,
        MAX_SIGNATURE_BYTES,
        "model catalog signature",
    )?;
    activate_signed_catalog(&catalog_bytes, &signature, LOCAL_IMPORT_SOURCE)
}

/// Download, authenticate, and activate the latest signed catalog. Offline
/// callers simply revalidate the current embedded/cached state.
pub fn update_catalog(options: &ModelDownloadOptions) -> Result<ModelCatalog, String> {
    if options.offline {
        return active_catalog();
    }
    let raw_url = options.source_url.as_deref().unwrap_or(DEFAULT_CATALOG_URL);
    let catalog_url =
        Url::parse(raw_url).map_err(|_| "invalid model catalog URL: expected HTTPS".to_string())?;
    if catalog_url.scheme() != "https" || catalog_url.host_str().is_none() {
        return Err(
            "model catalog URL must be an absolute HTTPS URL; use catalog import for local files"
                .into(),
        );
    }
    if catalog_url.fragment().is_some() {
        return Err("model catalog URL must not contain a fragment".into());
    }
    validate_authentication(&catalog_url, options.authentication.as_ref())?;
    let signature_url = catalog_signature_url(&catalog_url)?;
    let catalog_bytes =
        download_bounded(options, &catalog_url, MAX_CATALOG_BYTES, "model catalog")?;
    let signature = download_bounded(
        options,
        &signature_url,
        MAX_SIGNATURE_BYTES,
        "model catalog signature",
    )?;
    activate_signed_catalog(&catalog_bytes, &signature, &redact_url(raw_url))
}

fn activate_signed_catalog(
    catalog_bytes: &[u8],
    signature: &[u8],
    source: &str,
) -> Result<ModelCatalog, String> {
    validate_catalog_source(source)?;
    ensure_catalog_directory()?;
    let lock_destination = catalog_directory()?.join("catalog.json");
    let mut never_cancelled = || false;
    let lock = super::acquire_lock(&lock_destination, &mut never_cancelled)?;
    let result = (|| {
        let root = trust::load_active_trust_root_locked()?;
        let now = trust::effective_now_and_record_locked(&root)?;
        trust::require_fresh_root(&root, now)?;
        let catalog = verify_signed_catalog_with_root(
            catalog_bytes,
            signature,
            CatalogOrigin::Signed {
                source: source.to_string(),
            },
            &root,
            CatalogVerificationMode::Current { now },
        )?;
        let embedded = embedded_catalog();
        if catalog.sequence() < embedded.sequence() {
            return Err(format!(
                "refusing model catalog sequence {} older than embedded sequence {}",
                catalog.sequence(),
                embedded.sequence()
            ));
        }
        if catalog.sequence() == embedded.sequence()
            && (catalog.sha256() != embedded.sha256()
                || catalog.signing_key_id() != embedded.signing_key_id())
        {
            return Err(format!(
                "refusing different model catalog content at embedded sequence {}",
                embedded.sequence()
            ));
        }
        if catalog.sequence() == embedded.sequence() {
            // The signature was still authenticated above, but persisting an
            // envelope identical to immutable embedded bytes adds no authority
            // or rollback protection. Prefer the build's trusted copy.
            return Ok(Some(embedded));
        }
        if let Some(state) = load_state()? {
            validate_state(&state)?;
            if catalog.sequence() < state.highest_sequence {
                return Err(format!(
                    "refusing model catalog rollback from sequence {} to {}",
                    state.highest_sequence,
                    catalog.sequence()
                ));
            }
            if catalog.sequence() == state.highest_sequence
                && (catalog.sha256() != state.catalog_sha256
                    || catalog.signing_key_id() != state.signing_key_id)
            {
                return Err(format!(
                    "refusing different model catalog content at already accepted sequence {}",
                    state.highest_sequence
                ));
            }
        }

        // Persist the rollback floor first. A crash between these commits can
        // require a retry, but can never make an older signed catalog active.
        write_catalog_state(&catalog)?;
        let signature = std::str::from_utf8(signature)
            .map_err(|_| "model catalog signature is not UTF-8".to_string())?;
        write_json_atomic(
            &envelope_path()?,
            &SignedCatalogEnvelope {
                version: CATALOG_ENVELOPE_VERSION,
                catalog_base64: BASE64_STANDARD.encode(catalog_bytes),
                signature: signature.to_string(),
                source: source.to_string(),
            },
        )?;
        Ok(None)
    })();
    drop(lock);
    match result? {
        Some(embedded) => Ok(embedded),
        None => active_catalog(),
    }
}

fn verify_signed_catalog_with_root(
    catalog_bytes: &[u8],
    signature_bytes: &[u8],
    origin: CatalogOrigin,
    root: &trust::ActiveTrustRoot,
    mode: CatalogVerificationMode,
) -> Result<ModelCatalog, String> {
    if catalog_bytes.len() as u64 > MAX_CATALOG_BYTES {
        return Err("model catalog exceeds the 1 MiB limit".into());
    }
    if signature_bytes.len() as u64 > MAX_SIGNATURE_BYTES {
        return Err("model catalog signature exceeds the 16 KiB limit".into());
    }
    let document: CatalogDocument = serde_json::from_slice(catalog_bytes)
        .map_err(|error| format!("invalid model catalog JSON: {error}"))?;
    let trusted_key = root.catalog_key(&document.signing_key_id).ok_or_else(|| {
        format!(
            "model catalog names untrusted signing key {}",
            document.signing_key_id
        )
    })?;
    if matches!(mode, CatalogVerificationMode::Current { .. })
        && !trusted_key.accepts(document.sequence)
    {
        return Err(format!(
            "model catalog signing key {} is not valid for sequence {}",
            trusted_key.key_id, document.sequence
        ));
    }
    let signature_text = decode_signature_text(signature_bytes)?;
    let signature = Signature::decode(signature_text.as_ref())
        .map_err(|error| format!("invalid model catalog signature: {error}"))?;
    let public_key = PublicKey::from_base64(&trusted_key.public_key_base64)
        .map_err(|error| format!("invalid embedded catalog public key: {error}"))?;
    public_key
        .verify(catalog_bytes, &signature, false)
        .map_err(|error| format!("model catalog signature verification failed: {error}"))?;
    validate_document(document, catalog_bytes, origin, root, trusted_key, mode)
}

#[cfg(test)]
fn verify_signed_catalog(
    catalog_bytes: &[u8],
    signature_bytes: &[u8],
    origin: CatalogOrigin,
) -> Result<ModelCatalog, String> {
    let root = trust::embedded_trust_root();
    verify_signed_catalog_with_root(
        catalog_bytes,
        signature_bytes,
        origin,
        &root,
        CatalogVerificationMode::Current {
            now: root.issued_at_unix_seconds() + 1,
        },
    )
}

fn decode_signature_text(signature_bytes: &[u8]) -> Result<Cow<'_, str>, String> {
    let signature_text = std::str::from_utf8(signature_bytes)
        .map_err(|_| "detached minisign signature is not UTF-8".to_string())?
        .trim();
    if signature_text.starts_with("untrusted comment:") {
        validate_signature_text(signature_text)?;
        return Ok(Cow::Borrowed(signature_text));
    }
    let decoded = BASE64_STANDARD
        .decode(signature_text.as_bytes())
        .map_err(|_| "detached signature is neither minisign text nor Tauri base64".to_string())?;
    if decoded.len() as u64 > MAX_SIGNATURE_BYTES {
        return Err("decoded minisign signature exceeds the 16 KiB limit".into());
    }
    let decoded = String::from_utf8(decoded)
        .map_err(|_| "decoded minisign signature is not UTF-8".to_string())?;
    validate_signature_text(decoded.trim())?;
    Ok(Cow::Owned(decoded.trim().to_string()))
}

fn validate_signature_text(signature: &str) -> Result<(), String> {
    let lines = signature.lines().collect::<Vec<_>>();
    if lines.len() != 4
        || !lines[0].starts_with("untrusted comment:")
        || !lines[2].starts_with("trusted comment: ")
    {
        return Err("detached signature must contain one minisign record".into());
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn parse_catalog(bytes: &[u8], origin: CatalogOrigin) -> Result<ModelCatalog, String> {
    let root = trust::embedded_trust_root();
    parse_catalog_with_root(
        bytes,
        origin,
        &root,
        CatalogVerificationMode::Historical {
            accepted_root_version: root.version(),
        },
    )
}

fn parse_catalog_with_root(
    bytes: &[u8],
    origin: CatalogOrigin,
    root: &trust::ActiveTrustRoot,
    mode: CatalogVerificationMode,
) -> Result<ModelCatalog, String> {
    let document: CatalogDocument = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid embedded model catalog JSON: {error}"))?;
    let trusted_key = root.catalog_key(&document.signing_key_id).ok_or_else(|| {
        format!(
            "model catalog names untrusted signing key {}",
            document.signing_key_id
        )
    })?;
    if matches!(mode, CatalogVerificationMode::Current { .. })
        && !trusted_key.accepts(document.sequence)
    {
        return Err(format!(
            "model catalog signing key {} is not valid for sequence {}",
            trusted_key.key_id, document.sequence
        ));
    }
    validate_document(document, bytes, origin, root, trusted_key, mode)
}

fn validate_document(
    document: CatalogDocument,
    bytes: &[u8],
    origin: CatalogOrigin,
    root: &trust::ActiveTrustRoot,
    trusted_key: &trust::CatalogTrustKey,
    mode: CatalogVerificationMode,
) -> Result<ModelCatalog, String> {
    if !catalog_origin_is_safe(&origin) {
        return Err("invalid model catalog origin".into());
    }
    if document.schema != CATALOG_SCHEMA {
        return Err(format!(
            "unsupported model catalog schema: {}",
            document.schema
        ));
    }
    if document.sequence == 0 || document.sequence > MAX_JSON_SAFE_INTEGER {
        return Err(format!(
            "model catalog sequence must be between 1 and {MAX_JSON_SAFE_INTEGER}"
        ));
    }
    validate_catalog_validity(&document, root, mode)?;
    if document.models.is_empty() || document.models.len() > MAX_MODELS {
        return Err(format!(
            "model catalog must contain between 1 and {MAX_MODELS} entries"
        ));
    }

    let identity = CatalogIdentity {
        sequence: document.sequence,
        sha256: sha256_bytes(bytes),
        signing_key_id: document.signing_key_id,
        signing_public_key_base64: trusted_key.public_key_base64.clone(),
        issued_at_unix_seconds: document.issued_at_unix_seconds,
        expires_at_unix_seconds: document.expires_at_unix_seconds,
        trust_root_version: match mode {
            CatalogVerificationMode::Historical {
                accepted_root_version,
            } => accepted_root_version,
            CatalogVerificationMode::Current { .. } => root.version(),
        },
        origin,
    };
    let mut names = HashSet::with_capacity(document.models.len());
    let mut models = Vec::with_capacity(document.models.len());
    for model in document.models {
        validate_identifier("model name", &model.name)?;
        validate_windows_device_name("model name", &model.name)?;
        validate_identifier("backend", &model.backend)?;
        if !names.insert(model.name.clone()) {
            return Err(format!("duplicate model catalog name: {}", model.name));
        }
        validate_filename(&model.filename)?;
        validate_bounded_text("revision", &model.revision, 128)?;
        validate_bounded_text("license", &model.license, 128)?;
        if model.sha256.len() != 64
            || !model
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(format!(
                "model {} has an invalid lowercase SHA-256",
                model.name
            ));
        }
        if model.size_bytes == 0 || model.size_bytes > MAX_MODEL_BYTES {
            return Err(format!(
                "model {} size must be between 1 byte and {MAX_MODEL_BYTES} bytes",
                model.name
            ));
        }
        if !(8_000..=768_000).contains(&model.sample_rate) {
            return Err(format!(
                "model {} sample rate must be between 8000 and 768000 Hz",
                model.name
            ));
        }
        let url = Url::parse(&model.url)
            .map_err(|_| format!("model {} has an invalid URL", model.name))?;
        if url.scheme() != "https"
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            return Err(format!(
                "model {} URL must be HTTPS without credentials or a fragment",
                model.name
            ));
        }
        models.push(CatalogModel {
            name: model.name,
            backend: model.backend,
            filename: model.filename,
            url: model.url,
            revision: model.revision,
            sha256: model.sha256,
            size_bytes: model.size_bytes,
            license: model.license,
            sample_rate: model.sample_rate,
            catalog: identity.clone(),
        });
    }
    Ok(ModelCatalog { identity, models })
}

fn validate_catalog_validity(
    document: &CatalogDocument,
    root: &trust::ActiveTrustRoot,
    mode: CatalogVerificationMode,
) -> Result<(), String> {
    let enforce_current_policy = matches!(mode, CatalogVerificationMode::Current { .. });
    let validity = match (
        document.issued_at_unix_seconds,
        document.expires_at_unix_seconds,
    ) {
        (Some(issued_at), Some(expires_at)) => {
            if issued_at == 0
                || issued_at > MAX_JSON_SAFE_INTEGER
                || expires_at == 0
                || expires_at > MAX_JSON_SAFE_INTEGER
                || expires_at <= issued_at
                || (enforce_current_policy
                    && expires_at - issued_at > root.max_catalog_validity_seconds())
            {
                return Err("model catalog has an invalid validity window".into());
            }
            Some((issued_at, expires_at))
        }
        (None, None) => None,
        _ => {
            return Err(
                "model catalog must provide issued_at_unix_seconds and expires_at_unix_seconds together"
                    .into(),
            );
        }
    };
    let CatalogVerificationMode::Current { now } = mode else {
        return Ok(());
    };
    if document.sequence >= root.expiration_required_from_sequence() && validity.is_none() {
        return Err(format!(
            "model catalog sequence {} must contain an expiration window",
            document.sequence
        ));
    }
    if let Some((issued_at, expires_at)) = validity {
        if issued_at > now.saturating_add(24 * 60 * 60) {
            return Err(format!(
                "model catalog sequence {} is not valid yet",
                document.sequence
            ));
        }
        if expires_at <= now {
            return Err(format!(
                "model catalog sequence {} expired at Unix time {expires_at}",
                document.sequence
            ));
        }
    }
    Ok(())
}

fn validate_identifier(description: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
        || !value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
    {
        return Err(format!(
            "{description} must be 1-64 lowercase ASCII letters, digits, '-' or '_'"
        ));
    }
    Ok(())
}

fn validate_filename(value: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 128
        || !bytes[0].is_ascii_alphanumeric()
        || bytes.last() == Some(&b'.')
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(
            "model filename must be 1-128 portable ASCII letters, digits, '.', '-' or '_'".into(),
        );
    }
    let mut components = Path::new(value).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err("model filename must be one ordinary path component".into());
    }
    validate_windows_device_name("model filename", value)
}

fn validate_windows_device_name(description: &str, value: &str) -> Result<(), String> {
    let stem = value
        .split('.')
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || stem.strip_prefix("COM").is_some_and(|suffix| {
            matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        })
        || stem.strip_prefix("LPT").is_some_and(|suffix| {
            matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
        })
    {
        return Err(format!("{description} uses a reserved Windows device name"));
    }
    Ok(())
}

fn validate_bounded_text(description: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(format!(
            "model {description} must be 1-{maximum} non-control UTF-8 bytes"
        ));
    }
    Ok(())
}

fn catalog_signature_url(catalog_url: &Url) -> Result<Url, String> {
    let mut signature_url = catalog_url.clone();
    let path = signature_url.path().to_string();
    if path.is_empty() || path.ends_with('/') {
        return Err("model catalog URL must name a JSON file".into());
    }
    signature_url.set_path(&format!("{path}.sig"));
    Ok(signature_url)
}

fn download_bounded(
    options: &ModelDownloadOptions,
    source: &Url,
    maximum: u64,
    description: &str,
) -> Result<Vec<u8>, String> {
    let response = request_with_redirects(options, source, 0, None)?;
    if response.status() != 200 {
        return Err(format!(
            "{description} download from {} returned HTTP {}",
            redact_url(source.as_str()),
            response.status()
        ));
    }
    if parse_content_length(&response)?.is_some_and(|length| length > maximum) {
        return Err(format!("{description} exceeds its {maximum}-byte limit"));
    }
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            format!(
                "failed to download {description} from {}: {error}",
                redact_url(source.as_str())
            )
        })?;
    if bytes.len() as u64 > maximum {
        return Err(format!("{description} exceeds its {maximum}-byte limit"));
    }
    Ok(bytes)
}

fn read_bounded_file(path: &Path, maximum: u64, description: &str) -> Result<Vec<u8>, String> {
    let file = open_existing_regular_file(path, description)?
        .ok_or_else(|| format!("failed to open {}: file not found", path.display()))?;
    let length = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > maximum {
        return Err(format!("{description} exceeds its {maximum}-byte limit"));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if bytes.len() as u64 > maximum {
        return Err(format!("{description} exceeds its {maximum}-byte limit"));
    }
    Ok(bytes)
}

fn load_state() -> Result<Option<CatalogState>, String> {
    let path = state_path()?;
    let Some(bytes) = read_optional_bounded(&path, MAX_STATE_BYTES, "model catalog state")? else {
        return Ok(None);
    };
    let state: CatalogState = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid model catalog state: {error}"))?;
    validate_state(&state)?;
    Ok(Some(state))
}

fn validate_state(state: &CatalogState) -> Result<(), String> {
    if !matches!(
        state.version,
        LEGACY_CATALOG_STATE_VERSION | CATALOG_STATE_VERSION
    ) || state.highest_sequence == 0
        || state.highest_sequence > MAX_JSON_SAFE_INTEGER
        || state.catalog_sha256.len() != 64
        || !state
            .catalog_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        || state.signing_key_id.len() != 16
        || !state
            .signing_key_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
        || state.trust_root_version == 0
        || state.trust_root_version > MAX_JSON_SAFE_INTEGER
        || (state.version == LEGACY_CATALOG_STATE_VERSION && state.trust_root_version != 1)
    {
        return Err("invalid model catalog rollback state".into());
    }
    Ok(())
}

fn state_matches_catalog(state: &CatalogState, catalog: &ModelCatalog) -> bool {
    state.highest_sequence == catalog.sequence()
        && state.catalog_sha256 == catalog.sha256()
        && state.signing_key_id == catalog.signing_key_id()
        && state.trust_root_version == catalog.trust_root_version()
}

fn load_envelope() -> Result<Option<SignedCatalogEnvelope>, String> {
    let path = envelope_path()?;
    let Some(bytes) = read_optional_bounded(&path, MAX_ENVELOPE_BYTES, "signed model catalog")?
    else {
        return Ok(None);
    };
    let envelope: SignedCatalogEnvelope = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid signed model catalog cache: {error}"))?;
    if envelope.version != CATALOG_ENVELOPE_VERSION
        || envelope.signature.len() as u64 > MAX_SIGNATURE_BYTES
    {
        return Err("invalid signed model catalog cache".into());
    }
    validate_catalog_source(&envelope.source)
        .map_err(|_| "invalid signed model catalog cache".to_string())?;
    Ok(Some(envelope))
}

fn validate_catalog_source(source: &str) -> Result<(), String> {
    if source == LOCAL_IMPORT_SOURCE {
        return Ok(());
    }
    if source.is_empty() || source.len() > 2048 || source.chars().any(char::is_control) {
        return Err("invalid model catalog source".into());
    }
    let url = Url::parse(source).map_err(|_| "invalid model catalog source".to_string())?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("invalid model catalog source".into());
    }
    Ok(())
}

pub(crate) fn catalog_origin_is_safe(origin: &CatalogOrigin) -> bool {
    match origin {
        CatalogOrigin::Embedded => true,
        CatalogOrigin::Signed { source } => validate_catalog_source(source).is_ok(),
    }
}

fn read_optional_bounded(
    path: &Path,
    maximum: u64,
    description: &str,
) -> Result<Option<Vec<u8>>, String> {
    let Some(file) = open_existing_regular_file(path, description)? else {
        return Ok(None);
    };
    let length = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > maximum {
        return Err(format!("{description} exceeds its {maximum}-byte limit"));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if bytes.len() as u64 > maximum {
        return Err(format!("{description} exceeds its {maximum}-byte limit"));
    }
    Ok(Some(bytes))
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
    let mut output = AtomicOutput::new(path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    output.commit(CommitMode::Replace)
}

fn write_catalog_state(catalog: &ModelCatalog) -> Result<(), String> {
    write_json_atomic(
        &state_path()?,
        &CatalogState {
            version: CATALOG_STATE_VERSION,
            highest_sequence: catalog.sequence(),
            catalog_sha256: catalog.sha256().to_string(),
            signing_key_id: catalog.signing_key_id().to_string(),
            trust_root_version: catalog.trust_root_version(),
        },
    )
}

pub(super) fn catalog_state_trust_root_version_for_recovery() -> Result<Option<u64>, String> {
    Ok(load_state()?.map(|state| state.trust_root_version))
}

pub(crate) fn require_catalog_acquisition(identity: &CatalogIdentity) -> Result<(), String> {
    validate_catalog_storage_path()?;
    ensure_catalog_directory()?;
    let lock_destination = catalog_directory()?.join("catalog.json");
    let mut never_cancelled = || false;
    let lock = super::acquire_lock(&lock_destination, &mut never_cancelled)?;
    let result = (|| {
        let root = trust::load_active_trust_root_locked()?;
        let now = trust::effective_now_and_record_locked(&root)?;
        validate_acquisition_authority(identity, &root, now, load_state()?.as_ref())
    })();
    drop(lock);
    result
}

fn catalog_acquisition_allowed(catalog: &ModelCatalog) -> Result<bool, String> {
    validate_catalog_storage_path()?;
    let directory = catalog_directory()?;
    match std::fs::symlink_metadata(&directory) {
        Ok(_) => {
            let lock_destination = directory.join("catalog.json");
            let mut never_cancelled = || false;
            let lock = super::acquire_lock(&lock_destination, &mut never_cancelled)?;
            let result = (|| {
                let root = trust::load_active_trust_root_locked()?;
                let now = trust::effective_now_locked(&root)?;
                Ok(validate_acquisition_authority(
                    catalog.identity(),
                    &root,
                    now,
                    load_state()?.as_ref(),
                )
                .is_ok())
            })();
            drop(lock);
            result
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let root = trust::embedded_trust_root();
            let now = trust::effective_now_locked(&root)?;
            Ok(validate_acquisition_authority(catalog.identity(), &root, now, None).is_ok())
        }
        Err(error) => Err(format!(
            "failed to inspect model catalog directory {}: {error}",
            directory.display()
        )),
    }
}

fn validate_acquisition_authority(
    identity: &CatalogIdentity,
    root: &trust::ActiveTrustRoot,
    now: u64,
    state: Option<&CatalogState>,
) -> Result<(), String> {
    trust::require_fresh_root(root, now)?;
    if let Some(state) = state {
        if state.highest_sequence != identity.sequence
            || state.catalog_sha256 != identity.sha256
            || state.signing_key_id != identity.signing_key_id
        {
            return Err(format!(
                "model catalog sequence {} is no longer active; reload the active catalog before acquiring packages",
                identity.sequence
            ));
        }
    } else if !matches!(identity.origin, CatalogOrigin::Embedded) {
        return Err("model catalog is not backed by active rollback state".into());
    }
    let key = root.catalog_key(&identity.signing_key_id).ok_or_else(|| {
        format!(
            "model catalog signing key {} is no longer trusted for new acquisitions",
            identity.signing_key_id
        )
    })?;
    if key.public_key_base64 != identity.signing_public_key_base64
        || !key.accepts(identity.sequence)
    {
        return Err(format!(
            "model catalog signing key {} is revoked or outside its allowed sequence window",
            identity.signing_key_id
        ));
    }
    if identity.sequence >= root.expiration_required_from_sequence()
        && identity.expires_at_unix_seconds.is_none()
    {
        return Err(format!(
            "model catalog sequence {} lacks the expiration required by trust-root version {}",
            identity.sequence,
            root.version()
        ));
    }
    match (
        identity.issued_at_unix_seconds,
        identity.expires_at_unix_seconds,
    ) {
        (Some(issued), Some(expires))
            if expires > issued && expires - issued <= root.max_catalog_validity_seconds() => {}
        (Some(_), Some(_)) => {
            return Err(format!(
                "model catalog sequence {} exceeds the active maximum validity interval",
                identity.sequence
            ));
        }
        (None, None) => {}
        _ => {
            return Err(format!(
                "model catalog sequence {} has an incomplete validity interval",
                identity.sequence
            ));
        }
    }
    if identity
        .issued_at_unix_seconds
        .is_some_and(|issued| issued > now.saturating_add(24 * 60 * 60))
    {
        return Err(format!(
            "model catalog sequence {} is not valid yet",
            identity.sequence
        ));
    }
    if identity
        .expires_at_unix_seconds
        .is_some_and(|expires| expires <= now)
    {
        return Err(format!(
            "model catalog sequence {} expired at Unix time {}",
            identity.sequence,
            identity.expires_at_unix_seconds.unwrap_or_default()
        ));
    }
    Ok(())
}

fn ensure_catalog_directory() -> Result<PathBuf, String> {
    let cache = cache_dir()?;
    super::reject_symlink(&cache)?;
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("failed to create {}: {error}", cache.display()))?;
    super::reject_symlink(&cache)?;
    let directory = cache.join(".catalog");
    super::reject_symlink(&directory)?;
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create {}: {error}", directory.display()))?;
    super::reject_symlink(&directory)?;
    Ok(directory)
}

fn validate_catalog_storage_path() -> Result<(), String> {
    let cache = cache_dir()?;
    super::reject_symlink(&cache)?;
    super::reject_symlink(&cache.join(".catalog"))
}

fn catalog_directory() -> Result<PathBuf, String> {
    Ok(cache_dir()?.join(".catalog"))
}

fn state_path() -> Result<PathBuf, String> {
    Ok(catalog_directory()?.join("state.json"))
}

fn envelope_path() -> Result<PathBuf, String> {
    Ok(catalog_directory()?.join("active.json"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ2: &[u8] = include_bytes!("testdata/catalog-seq2.json");
    const SEQ2_SIG: &[u8] = include_bytes!("testdata/catalog-seq2.json.sig");
    const SEQ3: &[u8] = include_bytes!("testdata/catalog-seq3.json");
    const SEQ3_SIG: &[u8] = include_bytes!("testdata/catalog-seq3.json.sig");
    const SEQ4: &[u8] = include_bytes!("testdata/catalog-seq4.json");
    const SEQ4_SIG: &[u8] = include_bytes!("testdata/catalog-seq4.json.sig");

    #[test]
    fn embedded_catalog_has_a_stable_valid_identity() {
        let root = trust::embedded_trust_root();
        let production_key = root.catalog_key("F5AE02E7593C64D9").unwrap();
        let public_key = BASE64_STANDARD
            .decode(&production_key.public_key_base64)
            .unwrap();
        let encoded_key_id = u64::from_le_bytes(public_key[2..10].try_into().unwrap());
        assert_eq!(format!("{encoded_key_id:016X}"), production_key.key_id);

        let catalog = embedded_catalog();
        assert_eq!(catalog.sequence(), 1);
        assert_eq!(catalog.signing_key_id(), production_key.key_id);
        assert_eq!(catalog.trust_root_version(), 1);
        assert_eq!(catalog.models().len(), 1);
        let model = catalog.find("gtcrn").unwrap();
        let legacy = &crate::models::MODELS[0];
        assert_eq!(model.name(), legacy.name);
        assert_eq!(model.backend(), legacy.backend);
        assert_eq!(model.filename(), legacy.filename);
        assert_eq!(model.url(), legacy.url);
        assert_eq!(model.revision(), legacy.revision);
        assert_eq!(model.sha256(), legacy.sha256);
        assert_eq!(model.size_bytes(), legacy.size_bytes);
        assert_eq!(model.license(), legacy.license);
        assert_eq!(model.sample_rate(), legacy.sample_rate);
        assert!(catalog
            .sha256()
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn verifies_tauri_base64_signatures_and_key_rotation() {
        let seq2 = verify_signed_catalog(
            SEQ2,
            SEQ2_SIG,
            CatalogOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap();
        assert_eq!(seq2.sequence(), 2);
        assert_eq!(seq2.find("gtcrn").unwrap().revision(), "catalog-sequence-2");

        let seq3 = verify_signed_catalog(
            SEQ3,
            SEQ3_SIG,
            CatalogOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap();
        assert_eq!(seq3.sequence(), 3);
        assert!(seq3.find("gtcrn").is_none(), "backend alias is ambiguous");
        assert_eq!(
            seq3.find("gtcrn-studio").unwrap().filename(),
            "gtcrn_studio.onnx"
        );

        let seq4 = verify_signed_catalog(
            SEQ4,
            SEQ4_SIG,
            CatalogOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap();
        assert_eq!(seq4.sequence(), 4);
        assert_eq!(seq4.signing_key_id(), "557E67D5F983C071");
    }

    #[test]
    fn rejects_tampering_wrong_keys_and_out_of_window_keys() {
        let mut tampered = SEQ2.to_vec();
        let index = tampered
            .windows(b"catalog-sequence-2".len())
            .position(|window| window == b"catalog-sequence-2")
            .unwrap();
        *tampered
            .get_mut(index + b"catalog-sequence-".len())
            .unwrap() = b'9';
        let error = verify_signed_catalog(
            &tampered,
            SEQ2_SIG,
            CatalogOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap_err();
        assert!(error.contains("signature verification failed"), "{error}");

        let error = verify_signed_catalog(
            SEQ2,
            SEQ4_SIG,
            CatalogOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap_err();
        assert!(error.contains("different key"), "{error}");

        let old_key_at_sequence_four = String::from_utf8(SEQ4.to_vec())
            .unwrap()
            .replace("557E67D5F983C071", "DF5F0E9ED6135C46");
        let error = verify_signed_catalog(
            old_key_at_sequence_four.as_bytes(),
            SEQ4_SIG,
            CatalogOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap_err();
        assert!(error.contains("not valid for sequence 4"), "{error}");
    }

    #[test]
    fn signature_decoder_accepts_raw_minisign_and_tauri_wrapping() {
        let wrapped = std::str::from_utf8(SEQ2_SIG).unwrap();
        let raw = String::from_utf8(BASE64_STANDARD.decode(wrapped.trim()).unwrap()).unwrap();
        assert!(decode_signature_text(SEQ2_SIG)
            .unwrap()
            .starts_with("untrusted comment:"));
        assert_eq!(
            decode_signature_text(raw.as_bytes()).unwrap().as_ref(),
            raw.trim()
        );
        let extra = format!("{}\nforged", raw.trim());
        assert!(decode_signature_text(extra.as_bytes()).is_err());
    }

    #[test]
    fn catalog_expiration_policy_has_exact_boundaries() {
        let root = trust::trust_root_with_catalog_expiration_for_test(2, 100);
        let mut document: serde_json::Value = serde_json::from_slice(SEQ2).unwrap();
        document["issued_at_unix_seconds"] = 1_000_u64.into();
        document["expires_at_unix_seconds"] = 1_100_u64.into();
        let bytes = serde_json::to_vec(&document).unwrap();
        parse_catalog_with_root(
            &bytes,
            CatalogOrigin::Embedded,
            &root,
            CatalogVerificationMode::Current { now: 1_099 },
        )
        .unwrap();

        let error = parse_catalog_with_root(
            &bytes,
            CatalogOrigin::Embedded,
            &root,
            CatalogVerificationMode::Current { now: 1_100 },
        )
        .unwrap_err();
        assert!(error.contains("expired"), "{error}");

        document
            .as_object_mut()
            .unwrap()
            .remove("expires_at_unix_seconds");
        document
            .as_object_mut()
            .unwrap()
            .remove("issued_at_unix_seconds");
        let error = parse_catalog_with_root(
            &serde_json::to_vec(&document).unwrap(),
            CatalogOrigin::Embedded,
            &root,
            CatalogVerificationMode::Current { now: 1_000 },
        )
        .unwrap_err();
        assert!(
            error.contains("must contain an expiration window"),
            "{error}"
        );

        let mut document: serde_json::Value = serde_json::from_slice(SEQ2).unwrap();
        let issued_at = root.issued_at_unix_seconds() + 1;
        document["issued_at_unix_seconds"] = issued_at.into();
        document["expires_at_unix_seconds"] = (issued_at + 101).into();
        let bytes = serde_json::to_vec(&document).unwrap();
        let historical = parse_catalog_with_root(
            &bytes,
            CatalogOrigin::Embedded,
            &root,
            CatalogVerificationMode::Historical {
                accepted_root_version: root.version(),
            },
        )
        .unwrap();
        let error =
            validate_acquisition_authority(historical.identity(), &root, issued_at + 50, None)
                .unwrap_err();
        assert!(error.contains("maximum validity interval"), "{error}");

        let error = parse_catalog_with_root(
            &bytes,
            CatalogOrigin::Embedded,
            &root,
            CatalogVerificationMode::Current {
                now: issued_at + 50,
            },
        )
        .unwrap_err();
        assert!(error.contains("invalid validity window"), "{error}");
    }

    #[test]
    fn revocation_blocks_acquisition_without_invalidating_loaded_catalog() {
        let catalog = embedded_catalog();
        let root = trust::trust_root_with_revocation_for_test("F5AE02E7593C64D9", 1);
        let error = validate_acquisition_authority(
            catalog.identity(),
            &root,
            root.issued_at_unix_seconds() + 1,
            None,
        )
        .unwrap_err();
        assert!(error.contains("revoked"), "{error}");
        assert_eq!(catalog.find("gtcrn").unwrap().catalog_sequence(), 1);
    }

    #[test]
    fn catalog_parser_rejects_duplicate_names_and_unsafe_filenames() {
        let duplicated = String::from_utf8(SEQ3.to_vec())
            .unwrap()
            .replace("\"gtcrn-studio\"", "\"gtcrn-dns3\"");
        let error = parse_catalog(duplicated.as_bytes(), CatalogOrigin::Embedded).unwrap_err();
        assert!(error.contains("duplicate model catalog name"), "{error}");

        let unsafe_filename = String::from_utf8(SEQ2.to_vec())
            .unwrap()
            .replace("gtcrn_simple.onnx", "../gtcrn.onnx");
        let error = parse_catalog(unsafe_filename.as_bytes(), CatalogOrigin::Embedded).unwrap_err();
        assert!(error.contains("portable ASCII"), "{error}");

        for filename in ["nested\\model.onnx", "NUL.onnx", "model.onnx."] {
            assert!(validate_filename(filename).is_err(), "{filename}");
        }

        let reserved_model_name = String::from_utf8(SEQ2.to_vec())
            .unwrap()
            .replace("\"gtcrn-dns3\"", "\"aux\"");
        let error =
            parse_catalog(reserved_model_name.as_bytes(), CatalogOrigin::Embedded).unwrap_err();
        assert!(error.contains("reserved Windows device name"), "{error}");

        let unsafe_sequence = String::from_utf8(SEQ2.to_vec())
            .unwrap()
            .replace("\"sequence\": 2", "\"sequence\": 18446744073709551615");
        let error = parse_catalog(unsafe_sequence.as_bytes(), CatalogOrigin::Embedded).unwrap_err();
        assert!(error.contains("sequence must be between"), "{error}");
    }

    #[test]
    fn persisted_catalog_sources_are_safe_diagnostic_labels() {
        assert!(validate_catalog_source(LOCAL_IMPORT_SOURCE).is_ok());
        assert!(validate_catalog_source("https://models.example.test/catalog.json").is_ok());
        for invalid in [
            "test",
            "http://models.example.test/catalog.json",
            "https://user:secret@models.example.test/catalog.json",
            "https://models.example.test/catalog.json?token=secret",
            "https://models.example.test/catalog.json#fragment",
            "https://models.example.test/catalog.json\nforged",
        ] {
            assert!(validate_catalog_source(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn catalog_update_rejects_non_absolute_and_fragment_urls_before_network_io() {
        for source_url in [
            "file:///catalog.json",
            "https://models.example.test/catalog.json#unsigned-fragment",
        ] {
            let options = ModelDownloadOptions {
                source_url: Some(source_url.into()),
                ..ModelDownloadOptions::default()
            };
            let error = update_catalog(&options).unwrap_err();
            assert!(error.contains("model catalog URL"), "{source_url}: {error}");
        }
    }
}
