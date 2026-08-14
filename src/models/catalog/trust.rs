//! Versioned trust roots for authenticated model catalogs.

use super::{
    catalog_directory, decode_signature_text, ensure_catalog_directory, read_bounded_file,
    read_optional_bounded, sha256_bytes, validate_catalog_source, write_json_atomic,
    LOCAL_IMPORT_SOURCE, MAX_JSON_SAFE_INTEGER, MAX_SIGNATURE_BYTES,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use minisign_verify::{PublicKey, Signature};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const TRUST_ROOT_SCHEMA: &str = "denoize-model-trust-root-v1";
const TRUST_SIGNATURE_SCHEMA: &str = "denoize-model-trust-signatures-v1";
const TRUST_STATE_VERSION: u32 = 1;
const TRUST_CHAIN_VERSION: u32 = 1;
const MAX_TRUST_ROOT_BYTES: u64 = 64 * 1024;
const MAX_TRUST_SIGNATURE_BUNDLE_BYTES: u64 = 256 * 1024;
const MAX_TRUST_STATE_BYTES: u64 = 64 * 1024;
const MAX_TRUST_CHAIN_BYTES: u64 = 2 * 1024 * 1024;
const MAX_ROOT_KEYS: usize = 16;
const MAX_CATALOG_KEYS: usize = 64;
const MAX_ROOT_CHAIN_LENGTH: usize = 32;
const MAX_ROOT_SIGNATURES: usize = 32;
const MAX_ROOT_VALIDITY_SECONDS: u64 = 5 * 366 * 24 * 60 * 60;
const MAX_CATALOG_VALIDITY_SECONDS: u64 = 366 * 24 * 60 * 60;
const MAX_ISSUED_AT_FUTURE_SKEW_SECONDS: u64 = 24 * 60 * 60;
const EMBEDDED_TRUST_ROOT: &[u8] = include_bytes!("../../../models/trust-root-v1.json");

/// Where the active model-catalog trust root came from.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum TrustRootOrigin {
    /// Trust root compiled into this exact denoize binary.
    Embedded,
    /// Sequentially rotated root accepted from a local import or HTTPS source.
    Signed { source: String },
}

/// Human- and UI-facing state for model-catalog trust policy.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TrustRootStatus {
    pub version: u64,
    pub sha256: String,
    pub issued_at_unix_seconds: u64,
    pub expires_at_unix_seconds: u64,
    pub expired: bool,
    pub signature_threshold: u16,
    pub root_key_ids: Vec<String>,
    pub catalog_signing_key_ids: Vec<String>,
    pub origin: TrustRootOrigin,
    pub highest_accepted_version: u64,
    pub highest_observed_unix_seconds: Option<u64>,
    pub cached_trust_chain_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustRootDocument {
    schema: String,
    version: u64,
    previous_root_sha256: Option<String>,
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
    signature_threshold: u16,
    root_keys: Vec<TrustKey>,
    catalog_policy: CatalogTrustPolicy,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustKey {
    key_id: String,
    public_key_base64: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogTrustPolicy {
    expiration_required_from_sequence: u64,
    max_validity_seconds: u64,
    keys: Vec<CatalogTrustKey>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CatalogTrustKey {
    pub(super) key_id: String,
    pub(super) public_key_base64: String,
    pub(super) first_sequence: u64,
    pub(super) last_sequence: Option<u64>,
    pub(super) revoked_at_sequence: Option<u64>,
}

impl CatalogTrustKey {
    pub(super) fn accepts(&self, sequence: u64) -> bool {
        sequence >= self.first_sequence
            && self
                .last_sequence
                .is_none_or(|last_sequence| sequence <= last_sequence)
            && self
                .revoked_at_sequence
                .is_none_or(|revoked_at| sequence < revoked_at)
    }
}

#[derive(Clone, Debug)]
pub(super) struct ActiveTrustRoot {
    document: TrustRootDocument,
    sha256: String,
    origin: TrustRootOrigin,
}

impl ActiveTrustRoot {
    pub(super) fn version(&self) -> u64 {
        self.document.version
    }

    pub(super) fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(super) fn issued_at_unix_seconds(&self) -> u64 {
        self.document.issued_at_unix_seconds
    }

    pub(super) fn expires_at_unix_seconds(&self) -> u64 {
        self.document.expires_at_unix_seconds
    }

    pub(super) fn origin(&self) -> &TrustRootOrigin {
        &self.origin
    }

    pub(super) fn catalog_key(&self, key_id: &str) -> Option<&CatalogTrustKey> {
        self.document
            .catalog_policy
            .keys
            .iter()
            .find(|key| key.key_id == key_id)
    }

    pub(super) fn expiration_required_from_sequence(&self) -> u64 {
        self.document
            .catalog_policy
            .expiration_required_from_sequence
    }

    pub(super) fn max_catalog_validity_seconds(&self) -> u64 {
        self.document.catalog_policy.max_validity_seconds
    }

    pub(super) fn is_expired_at(&self, now: u64) -> bool {
        now >= self.expires_at_unix_seconds()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustSignatureBundle {
    schema: String,
    signatures: Vec<TrustSignatureRecord>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrustSignatureRecord {
    key_id: String,
    signature: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrustState {
    version: u32,
    highest_root_version: u64,
    root_sha256: String,
    highest_observed_unix_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrustChain {
    version: u32,
    roots: Vec<SignedTrustRootEnvelope>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SignedTrustRootEnvelope {
    root_base64: String,
    signatures: Vec<TrustSignatureRecord>,
    source: String,
}

pub(super) fn embedded_trust_root() -> ActiveTrustRoot {
    let root = parse_trust_root(EMBEDDED_TRUST_ROOT, TrustRootOrigin::Embedded)
        .expect("the embedded model trust root is validated by the test suite");
    #[cfg(test)]
    {
        let mut root = root;
        // Existing detached catalog fixtures exercise two catalog signing-key
        // rotations. They are test-only authority and never enter release
        // binaries or the embedded trust-root digest.
        root.document
            .catalog_policy
            .expiration_required_from_sequence = 5;
        root.document.root_keys = vec![TrustKey {
            key_id: "D2AAF1CBF67BDFF4".into(),
            public_key_base64: "RWT033v2y/Gq0oCdWdPpTwQsFNNtrjZ8NU9nAkGXgUii7Mclfyp5SYr2".into(),
        }];
        root.document.catalog_policy.keys.extend([
            CatalogTrustKey {
                key_id: "DF5F0E9ED6135C46".into(),
                public_key_base64: "RWRGXBPWng5f30bcoLrI1zJw2RyznBVNqkqjkCVztHv9cjqT3UAwuw1W"
                    .into(),
                first_sequence: 2,
                last_sequence: Some(3),
                revoked_at_sequence: None,
            },
            CatalogTrustKey {
                key_id: "557E67D5F983C071".into(),
                public_key_base64: "RWRxwIP51Wd+VQD5W1g2IGJKbiO0tEjlMfR4V58VKkamn1A9MoOXy+g+"
                    .into(),
                first_sequence: 4,
                last_sequence: None,
                revoked_at_sequence: None,
            },
        ]);
        root
    }
    #[cfg(not(test))]
    {
        root
    }
}

#[cfg(test)]
pub(super) fn trust_root_with_catalog_expiration_for_test(
    required_from_sequence: u64,
    maximum_validity_seconds: u64,
) -> ActiveTrustRoot {
    let mut root = embedded_trust_root();
    root.document
        .catalog_policy
        .expiration_required_from_sequence = required_from_sequence;
    root.document.catalog_policy.max_validity_seconds = maximum_validity_seconds;
    root
}

#[cfg(test)]
pub(super) fn trust_root_with_revocation_for_test(
    key_id: &str,
    revoked_at_sequence: u64,
) -> ActiveTrustRoot {
    let mut root = embedded_trust_root();
    root.document
        .catalog_policy
        .keys
        .iter_mut()
        .find(|key| key.key_id == key_id)
        .expect("test catalog key exists")
        .revoked_at_sequence = Some(revoked_at_sequence);
    root
}

pub(super) fn load_active_trust_root_locked() -> Result<ActiveTrustRoot, String> {
    let embedded = embedded_trust_root();
    let state = load_trust_state()?;
    let chain = load_trust_chain()?;
    match (state, chain) {
        (None, None) => Ok(embedded),
        (None, Some(_)) => Err(
            "model trust-root chain exists without rollback state; re-import the trust root".into(),
        ),
        (Some(state), chain) => {
            validate_trust_state(&state)?;
            if state.highest_root_version < embedded.version() {
                // A newer binary-embedded root is an independent recovery and
                // upgrade channel. Preserve the monotonic time floor while
                // replacing obsolete cached chain state.
                write_trust_state(&embedded, state.highest_observed_unix_seconds)?;
                write_trust_chain(&TrustChain {
                    version: TRUST_CHAIN_VERSION,
                    roots: Vec::new(),
                })?;
                return Ok(embedded);
            }
            if state.highest_root_version == embedded.version() {
                if state.root_sha256 != embedded.sha256() {
                    return Err(format!(
                        "embedded model trust root conflicts with persisted version {}",
                        embedded.version()
                    ));
                }
                return Ok(embedded);
            }
            let chain = chain.ok_or_else(|| {
                format!(
                    "model trust-root version {} was accepted previously, but its signed chain is missing; re-import the chain or install a newer denoize binary",
                    state.highest_root_version
                )
            })?;
            let active = verify_chain_from_embedded(embedded, &chain)?;
            if active.version() != state.highest_root_version
                || active.sha256() != state.root_sha256
            {
                return Err("signed model trust-root chain does not match rollback state".into());
            }
            Ok(active)
        }
    }
}

/// Inspect the active trust root and its rollback/expiry state.
pub fn trust_root_status() -> Result<TrustRootStatus, String> {
    super::validate_catalog_storage_path()?;
    let directory = catalog_directory()?;
    let root = match std::fs::symlink_metadata(&directory) {
        Ok(_) => {
            let lock_destination = directory.join("catalog.json");
            let mut never_cancelled = || false;
            let lock = super::super::acquire_lock(&lock_destination, &mut never_cancelled)?;
            let result = load_active_trust_root_locked();
            drop(lock);
            result?
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => embedded_trust_root(),
        Err(error) => {
            return Err(format!(
                "failed to inspect model trust-root directory {}: {error}",
                directory.display()
            ));
        }
    };
    status_for_root(root)
}

/// Verify and atomically activate the next sequential trust root. The signature
/// bundle must satisfy both the current and candidate root thresholds.
pub fn import_trust_root(
    root_path: impl AsRef<Path>,
    signatures_path: impl AsRef<Path>,
) -> Result<TrustRootStatus, String> {
    let root_bytes =
        read_bounded_file(root_path.as_ref(), MAX_TRUST_ROOT_BYTES, "model trust root")?;
    let signature_bytes = read_bounded_file(
        signatures_path.as_ref(),
        MAX_TRUST_SIGNATURE_BUNDLE_BYTES,
        "model trust-root signature bundle",
    )?;
    let bundle = parse_signature_bundle(&signature_bytes)?;
    ensure_catalog_directory()?;
    let lock_destination = catalog_directory()?.join("catalog.json");
    let mut never_cancelled = || false;
    let lock = super::super::acquire_lock(&lock_destination, &mut never_cancelled)?;
    let result = (|| {
        let candidate = parse_trust_root(
            &root_bytes,
            TrustRootOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )?;
        let (current, mut chain) = match load_active_trust_root_locked() {
            Ok(current)
                if current.version() == candidate.version()
                    && current.sha256() == candidate.sha256() =>
            {
                return Ok(current);
            }
            Ok(current) => (
                current,
                load_trust_chain()?.unwrap_or(TrustChain {
                    version: TRUST_CHAIN_VERSION,
                    roots: Vec::new(),
                }),
            ),
            Err(original_error) => {
                // The rollback floor is committed before the chain. If a crash
                // lands between those writes, the exact same candidate can
                // safely repair the interrupted transaction after it is again
                // verified from the last complete chain root.
                let Some(state) = load_trust_state()? else {
                    return Err(original_error);
                };
                if state.highest_root_version != candidate.version()
                    || state.root_sha256 != candidate.sha256()
                {
                    return Err(original_error);
                }
                let chain = load_trust_chain()?.unwrap_or(TrustChain {
                    version: TRUST_CHAIN_VERSION,
                    roots: Vec::new(),
                });
                let previous = verify_chain_from_embedded(embedded_trust_root(), &chain)?;
                (previous, chain)
            }
        };
        verify_root_transition(&current, &candidate, &root_bytes, &bundle)?;
        let now = observed_now_locked()?;
        require_fresh_root(&candidate, now)?;
        if chain.roots.len() >= MAX_ROOT_CHAIN_LENGTH {
            return Err(format!(
                "model trust-root chain exceeds its {MAX_ROOT_CHAIN_LENGTH}-root limit; install a newer denoize binary"
            ));
        }
        chain.roots.push(SignedTrustRootEnvelope {
            root_base64: BASE64_STANDARD.encode(&root_bytes),
            signatures: bundle.signatures,
            source: LOCAL_IMPORT_SOURCE.into(),
        });

        // Persist the version/digest floor first. A crash can require the same
        // import to be retried, but cannot reactivate an older trust root.
        write_trust_state(&candidate, now)?;
        write_trust_chain(&chain)?;
        Ok(candidate)
    })();
    drop(lock);
    status_for_root(result?)
}

/// Reset corrupt or incomplete cached trust-root state to the root embedded in
/// this binary. Recovery never lowers the catalog's accepted sequence or a
/// valid newer trust-root version; those cases require a newer binary or the
/// missing signed chain.
pub fn recover_embedded_trust_root() -> Result<TrustRootStatus, String> {
    ensure_catalog_directory()?;
    let lock_destination = catalog_directory()?.join("catalog.json");
    let mut never_cancelled = || false;
    let lock = super::super::acquire_lock(&lock_destination, &mut never_cancelled)?;
    let result = (|| {
        let embedded = embedded_trust_root();
        if let Some(catalog_root_version) = super::catalog_state_trust_root_version_for_recovery()?
        {
            if catalog_root_version > embedded.version() {
                return Err(format!(
                    "refusing to reset trust root {} below active catalog authority {}; re-import the signed chain or install a newer denoize binary",
                    embedded.version(), catalog_root_version
                ));
            }
        }
        let state = match load_trust_state() {
            Ok(state) => state,
            // Recovery is specifically allowed to replace corrupt trust-root
            // state. A syntactically valid newer floor remains authoritative.
            Err(_) => None,
        };
        if let Some(state) = &state {
            if state.highest_root_version > embedded.version() {
                return Err(format!(
                    "refusing model trust-root rollback from version {} to embedded version {}; re-import the signed chain or install a newer denoize binary",
                    state.highest_root_version,
                    embedded.version()
                ));
            }
        }
        let observed = match state {
            Some(state) => state.highest_observed_unix_seconds,
            None => system_unix_seconds()?,
        };
        write_trust_state(&embedded, observed)?;
        write_trust_chain(&TrustChain {
            version: TRUST_CHAIN_VERSION,
            roots: Vec::new(),
        })?;
        Ok(embedded)
    })();
    drop(lock);
    status_for_root(result?)
}

/// Reset only the persisted trusted-time floor to the current system clock.
///
/// This exceptional recovery is intended for an accidental future clock jump
/// after the host clock has been corrected. It keeps the active signed root,
/// its chain, and both root and catalog rollback floors unchanged.
pub fn reset_trust_time_floor() -> Result<TrustRootStatus, String> {
    ensure_catalog_directory()?;
    let lock_destination = catalog_directory()?.join("catalog.json");
    let mut never_cancelled = || false;
    let lock = super::super::acquire_lock(&lock_destination, &mut never_cancelled)?;
    let result = (|| {
        let root = load_active_trust_root_locked()?;
        if let Some(catalog_root_version) = super::catalog_state_trust_root_version_for_recovery()?
        {
            if catalog_root_version > root.version() {
                return Err(format!(
                    "refusing to reset trusted time below active catalog authority {catalog_root_version}"
                ));
            }
        }
        let now = system_unix_seconds()?;
        require_fresh_root(&root, now)?;
        write_trust_state(&root, now)?;
        Ok(root)
    })();
    drop(lock);
    status_for_root(result?)
}

pub(super) fn effective_now_and_record_locked(root: &ActiveTrustRoot) -> Result<u64, String> {
    let now = effective_now_locked(root)?;
    let state = load_trust_state()?;
    if state
        .as_ref()
        .is_none_or(|state| state.highest_observed_unix_seconds < now)
    {
        write_trust_state(root, now)?;
    }
    Ok(now)
}

pub(super) fn effective_now_locked(root: &ActiveTrustRoot) -> Result<u64, String> {
    let system = system_unix_seconds()?;
    let Some(state) = load_trust_state()? else {
        return Ok(system);
    };
    validate_trust_state(&state)?;
    if state.highest_root_version != root.version() || state.root_sha256 != root.sha256() {
        return Err("active model trust root does not match persisted rollback state".into());
    }
    Ok(system.max(state.highest_observed_unix_seconds))
}

fn observed_now_locked() -> Result<u64, String> {
    let system = system_unix_seconds()?;
    Ok(match load_trust_state()? {
        Some(state) => system.max(state.highest_observed_unix_seconds),
        None => system,
    })
}

pub(super) fn require_fresh_root(root: &ActiveTrustRoot, now: u64) -> Result<(), String> {
    if root.issued_at_unix_seconds() > now.saturating_add(MAX_ISSUED_AT_FUTURE_SKEW_SECONDS) {
        return Err(format!(
            "model trust-root version {} is not valid yet",
            root.version()
        ));
    }
    if root.is_expired_at(now) {
        return Err(format!(
            "model trust-root version {} expired at Unix time {}; import a newer signed root or install a newer denoize binary",
            root.version(),
            root.expires_at_unix_seconds()
        ));
    }
    Ok(())
}

fn status_for_root(root: ActiveTrustRoot) -> Result<TrustRootStatus, String> {
    let state = load_trust_state()?;
    let now = match &state {
        Some(state) => system_unix_seconds()?.max(state.highest_observed_unix_seconds),
        None => system_unix_seconds()?,
    };
    let mut root_key_ids = root
        .document
        .root_keys
        .iter()
        .map(|key| key.key_id.clone())
        .collect::<Vec<_>>();
    root_key_ids.sort();
    let mut catalog_signing_key_ids = root
        .document
        .catalog_policy
        .keys
        .iter()
        .map(|key| key.key_id.clone())
        .collect::<Vec<_>>();
    catalog_signing_key_ids.sort();
    Ok(TrustRootStatus {
        version: root.version(),
        sha256: root.sha256().to_string(),
        issued_at_unix_seconds: root.issued_at_unix_seconds(),
        expires_at_unix_seconds: root.expires_at_unix_seconds(),
        expired: root.is_expired_at(now),
        signature_threshold: root.document.signature_threshold,
        root_key_ids,
        catalog_signing_key_ids,
        origin: root.origin().clone(),
        highest_accepted_version: root.version(),
        highest_observed_unix_seconds: state.map(|state| state.highest_observed_unix_seconds),
        cached_trust_chain_path: trust_chain_path()?,
    })
}

fn parse_trust_root(bytes: &[u8], origin: TrustRootOrigin) -> Result<ActiveTrustRoot, String> {
    if bytes.len() as u64 > MAX_TRUST_ROOT_BYTES {
        return Err("model trust root exceeds the 64 KiB limit".into());
    }
    let document: TrustRootDocument = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid model trust-root JSON: {error}"))?;
    validate_trust_root_document(&document)?;
    Ok(ActiveTrustRoot {
        document,
        sha256: sha256_bytes(bytes),
        origin,
    })
}

fn validate_trust_root_document(document: &TrustRootDocument) -> Result<(), String> {
    if document.schema != TRUST_ROOT_SCHEMA {
        return Err(format!(
            "unsupported model trust-root schema: {}",
            document.schema
        ));
    }
    if document.version == 0 || document.version > MAX_JSON_SAFE_INTEGER {
        return Err(format!(
            "model trust-root version must be between 1 and {MAX_JSON_SAFE_INTEGER}"
        ));
    }
    match (document.version, document.previous_root_sha256.as_deref()) {
        (1, None) => {}
        (1, Some(_)) => return Err("model trust-root version 1 must not name a predecessor".into()),
        (_, Some(digest)) if valid_sha256(digest) => {}
        (_, Some(_)) => return Err("model trust-root predecessor has an invalid SHA-256".into()),
        (_, None) => {
            return Err("rotated model trust root must name its predecessor SHA-256".into())
        }
    }
    validate_validity_window(
        "model trust root",
        document.issued_at_unix_seconds,
        document.expires_at_unix_seconds,
        MAX_ROOT_VALIDITY_SECONDS,
    )?;
    if document.root_keys.is_empty() || document.root_keys.len() > MAX_ROOT_KEYS {
        return Err(format!(
            "model trust root must contain between 1 and {MAX_ROOT_KEYS} root keys"
        ));
    }
    if document.signature_threshold == 0
        || usize::from(document.signature_threshold) > document.root_keys.len()
    {
        return Err("model trust-root signature threshold exceeds its distinct root keys".into());
    }
    let mut root_key_ids = HashSet::with_capacity(document.root_keys.len());
    for key in &document.root_keys {
        validate_trust_key(key)?;
        if !root_key_ids.insert(key.key_id.as_str()) {
            return Err(format!("duplicate model trust-root key: {}", key.key_id));
        }
    }
    let policy = &document.catalog_policy;
    if policy.expiration_required_from_sequence == 0
        || policy.expiration_required_from_sequence > MAX_JSON_SAFE_INTEGER
        || policy.max_validity_seconds == 0
        || policy.max_validity_seconds > MAX_CATALOG_VALIDITY_SECONDS
    {
        return Err("invalid model catalog expiration policy in trust root".into());
    }
    if policy.keys.is_empty() || policy.keys.len() > MAX_CATALOG_KEYS {
        return Err(format!(
            "model trust root must contain between 1 and {MAX_CATALOG_KEYS} catalog keys"
        ));
    }
    let mut catalog_key_ids = HashSet::with_capacity(policy.keys.len());
    for key in &policy.keys {
        validate_catalog_key(key)?;
        if !catalog_key_ids.insert(key.key_id.as_str()) {
            return Err(format!(
                "duplicate model catalog signing key in trust root: {}",
                key.key_id
            ));
        }
    }
    Ok(())
}

fn validate_trust_key(key: &TrustKey) -> Result<(), String> {
    validate_public_key_identity(&key.key_id, &key.public_key_base64, "trust-root key")
}

fn validate_catalog_key(key: &CatalogTrustKey) -> Result<(), String> {
    validate_public_key_identity(&key.key_id, &key.public_key_base64, "catalog signing key")?;
    if key.first_sequence == 0 || key.first_sequence > MAX_JSON_SAFE_INTEGER {
        return Err(format!(
            "catalog signing key {} has an invalid first sequence",
            key.key_id
        ));
    }
    if key
        .last_sequence
        .is_some_and(|last| last < key.first_sequence || last > MAX_JSON_SAFE_INTEGER)
    {
        return Err(format!(
            "catalog signing key {} has an invalid last sequence",
            key.key_id
        ));
    }
    if key
        .revoked_at_sequence
        .is_some_and(|revoked| revoked < key.first_sequence || revoked > MAX_JSON_SAFE_INTEGER)
    {
        return Err(format!(
            "catalog signing key {} has an invalid revocation sequence",
            key.key_id
        ));
    }
    Ok(())
}

fn validate_public_key_identity(
    key_id: &str,
    public_key_base64: &str,
    description: &str,
) -> Result<(), String> {
    if !valid_key_id(key_id) {
        return Err(format!("{description} has an invalid key id"));
    }
    PublicKey::from_base64(public_key_base64)
        .map_err(|error| format!("invalid {description} public key: {error}"))?;
    let decoded = BASE64_STANDARD
        .decode(public_key_base64.as_bytes())
        .map_err(|_| format!("invalid {description} public key base64"))?;
    if decoded.len() < 10 {
        return Err(format!("invalid {description} public key length"));
    }
    let encoded_key_id = u64::from_le_bytes(
        decoded[2..10]
            .try_into()
            .map_err(|_| format!("invalid {description} public key id"))?,
    );
    if format!("{encoded_key_id:016X}") != key_id {
        return Err(format!(
            "{description} public key id does not match key bytes"
        ));
    }
    Ok(())
}

fn valid_key_id(key_id: &str) -> bool {
    key_id.len() == 16
        && key_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
}

fn validate_validity_window(
    description: &str,
    issued_at: u64,
    expires_at: u64,
    maximum: u64,
) -> Result<(), String> {
    if issued_at == 0
        || issued_at > MAX_JSON_SAFE_INTEGER
        || expires_at == 0
        || expires_at > MAX_JSON_SAFE_INTEGER
        || expires_at <= issued_at
        || expires_at - issued_at > maximum
    {
        return Err(format!("{description} has an invalid validity window"));
    }
    Ok(())
}

fn verify_root_transition(
    current: &ActiveTrustRoot,
    candidate: &ActiveTrustRoot,
    candidate_bytes: &[u8],
    bundle: &TrustSignatureBundle,
) -> Result<(), String> {
    let expected_version = current
        .version()
        .checked_add(1)
        .ok_or_else(|| "model trust-root version overflow".to_string())?;
    if candidate.version() != expected_version {
        return Err(format!(
            "model trust-root rotation must advance exactly from version {} to {}",
            current.version(),
            expected_version
        ));
    }
    if candidate.document.previous_root_sha256.as_deref() != Some(current.sha256()) {
        return Err("model trust-root predecessor digest does not match active root".into());
    }
    if candidate.issued_at_unix_seconds() < current.issued_at_unix_seconds() {
        return Err("rotated model trust root predates the active root".into());
    }
    ensure_catalog_history_is_retained(current, candidate)?;
    verify_signature_threshold(
        candidate_bytes,
        bundle,
        &current.document.root_keys,
        current.document.signature_threshold,
        "current",
    )?;
    verify_signature_threshold(
        candidate_bytes,
        bundle,
        &candidate.document.root_keys,
        candidate.document.signature_threshold,
        "candidate",
    )
}

fn ensure_catalog_history_is_retained(
    current: &ActiveTrustRoot,
    candidate: &ActiveTrustRoot,
) -> Result<(), String> {
    if candidate.expiration_required_from_sequence() > current.expiration_required_from_sequence()
        || candidate.max_catalog_validity_seconds() > current.max_catalog_validity_seconds()
    {
        return Err("rotated model trust root weakened catalog expiration policy".into());
    }
    for previous in &current.document.catalog_policy.keys {
        let Some(next) = candidate.catalog_key(&previous.key_id) else {
            return Err(format!(
                "rotated model trust root removed historical catalog key {}",
                previous.key_id
            ));
        };
        if next.public_key_base64 != previous.public_key_base64
            || next.first_sequence != previous.first_sequence
        {
            return Err(format!(
                "rotated model trust root changed historical catalog key {}",
                previous.key_id
            ));
        }
        if widens_optional_ceiling(previous.last_sequence, next.last_sequence)
            || widens_optional_ceiling(previous.revoked_at_sequence, next.revoked_at_sequence)
        {
            return Err(format!(
                "rotated model trust root widened authority for catalog key {}",
                previous.key_id
            ));
        }
    }
    Ok(())
}

fn widens_optional_ceiling(previous: Option<u64>, next: Option<u64>) -> bool {
    match (previous, next) {
        (Some(previous), Some(next)) => next > previous,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

fn verify_signature_threshold(
    bytes: &[u8],
    bundle: &TrustSignatureBundle,
    keys: &[TrustKey],
    threshold: u16,
    description: &str,
) -> Result<(), String> {
    let mut verified = HashSet::new();
    for record in &bundle.signatures {
        let Some(key) = keys.iter().find(|key| key.key_id == record.key_id) else {
            continue;
        };
        let signature_text = decode_signature_text(record.signature.as_bytes())?;
        let signature = Signature::decode(signature_text.as_ref())
            .map_err(|error| format!("invalid model trust-root signature: {error}"))?;
        let public_key = PublicKey::from_base64(&key.public_key_base64)
            .map_err(|error| format!("invalid embedded trust-root public key: {error}"))?;
        public_key
            .verify(bytes, &signature, false)
            .map_err(|error| {
                format!(
                    "model trust-root signature verification failed for key {}: {error}",
                    key.key_id
                )
            })?;
        verified.insert(key.key_id.as_str());
    }
    if verified.len() < usize::from(threshold) {
        return Err(format!(
            "model trust-root signature bundle satisfies {} of {} required {description}-root signatures",
            verified.len(), threshold
        ));
    }
    Ok(())
}

fn parse_signature_bundle(bytes: &[u8]) -> Result<TrustSignatureBundle, String> {
    let bundle: TrustSignatureBundle = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid model trust-root signature bundle: {error}"))?;
    if bundle.schema != TRUST_SIGNATURE_SCHEMA
        || bundle.signatures.is_empty()
        || bundle.signatures.len() > MAX_ROOT_SIGNATURES
    {
        return Err("invalid model trust-root signature bundle".into());
    }
    validate_signature_records(&bundle.signatures)?;
    Ok(bundle)
}

fn validate_signature_records(signatures: &[TrustSignatureRecord]) -> Result<(), String> {
    if signatures.is_empty() || signatures.len() > MAX_ROOT_SIGNATURES {
        return Err("invalid model trust-root signature bundle".into());
    }
    let mut key_ids = HashSet::with_capacity(signatures.len());
    for signature in signatures {
        if !valid_key_id(&signature.key_id)
            || !key_ids.insert(signature.key_id.as_str())
            || signature.signature.len() as u64 > MAX_SIGNATURE_BYTES
        {
            return Err("invalid model trust-root signature bundle".into());
        }
        // Validate syntax before any state mutation. Cryptographic verification
        // is performed against both current and candidate threshold sets.
        decode_signature_text(signature.signature.as_bytes())?;
    }
    Ok(())
}

fn verify_chain_from_embedded(
    embedded: ActiveTrustRoot,
    chain: &TrustChain,
) -> Result<ActiveTrustRoot, String> {
    if chain.version != TRUST_CHAIN_VERSION || chain.roots.len() > MAX_ROOT_CHAIN_LENGTH {
        return Err("invalid model trust-root chain".into());
    }
    let mut current = embedded;
    for envelope in &chain.roots {
        validate_catalog_source(&envelope.source)
            .map_err(|_| "invalid model trust-root chain source".to_string())?;
        validate_signature_records(&envelope.signatures)
            .map_err(|_| "invalid model trust-root chain signatures".to_string())?;
        let root_bytes = BASE64_STANDARD
            .decode(envelope.root_base64.as_bytes())
            .map_err(|_| "cached model trust root has invalid base64".to_string())?;
        let candidate = parse_trust_root(
            &root_bytes,
            TrustRootOrigin::Signed {
                source: envelope.source.clone(),
            },
        )?;
        if candidate.version() <= current.version() {
            if candidate.version() == current.version() && candidate.sha256() != current.sha256() {
                return Err(format!(
                    "signed model trust-root chain conflicts at version {}",
                    current.version()
                ));
            }
            continue;
        }
        let bundle = TrustSignatureBundle {
            schema: TRUST_SIGNATURE_SCHEMA.into(),
            signatures: envelope.signatures.clone(),
        };
        verify_root_transition(&current, &candidate, &root_bytes, &bundle)?;
        current = candidate;
    }
    Ok(current)
}

fn load_trust_state() -> Result<Option<TrustState>, String> {
    let path = trust_state_path()?;
    let Some(bytes) =
        read_optional_bounded(&path, MAX_TRUST_STATE_BYTES, "model trust-root state")?
    else {
        return Ok(None);
    };
    let state: TrustState = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid model trust-root state: {error}"))?;
    validate_trust_state(&state)?;
    Ok(Some(state))
}

fn validate_trust_state(state: &TrustState) -> Result<(), String> {
    if state.version != TRUST_STATE_VERSION
        || state.highest_root_version == 0
        || state.highest_root_version > MAX_JSON_SAFE_INTEGER
        || !valid_sha256(&state.root_sha256)
        || state.highest_observed_unix_seconds == 0
        || state.highest_observed_unix_seconds > MAX_JSON_SAFE_INTEGER
    {
        return Err("invalid model trust-root rollback state".into());
    }
    Ok(())
}

fn load_trust_chain() -> Result<Option<TrustChain>, String> {
    let path = trust_chain_path()?;
    let Some(bytes) =
        read_optional_bounded(&path, MAX_TRUST_CHAIN_BYTES, "model trust-root chain")?
    else {
        return Ok(None);
    };
    let chain: TrustChain = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid model trust-root chain: {error}"))?;
    if chain.version != TRUST_CHAIN_VERSION || chain.roots.len() > MAX_ROOT_CHAIN_LENGTH {
        return Err("invalid model trust-root chain".into());
    }
    Ok(Some(chain))
}

fn write_trust_state(root: &ActiveTrustRoot, observed: u64) -> Result<(), String> {
    write_json_atomic(
        &trust_state_path()?,
        &TrustState {
            version: TRUST_STATE_VERSION,
            highest_root_version: root.version(),
            root_sha256: root.sha256().to_string(),
            highest_observed_unix_seconds: observed,
        },
    )
}

fn write_trust_chain(chain: &TrustChain) -> Result<(), String> {
    write_json_atomic(&trust_chain_path()?, chain)
}

fn trust_state_path() -> Result<PathBuf, String> {
    Ok(catalog_directory()?.join("trust-state.json"))
}

fn trust_chain_path() -> Result<PathBuf, String> {
    Ok(catalog_directory()?.join("trust-chain.json"))
}

fn system_unix_seconds() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| "system clock predates the Unix epoch".to_string())
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

    const ROOT_V2: &[u8] = include_bytes!("../testdata/trust-root-v2.json");
    const ROOT_V2_SIGNATURES: &[u8] = include_bytes!("../testdata/trust-root-v2.signatures.json");

    #[test]
    fn embedded_root_has_matching_key_identity_and_bounded_policy() {
        let root = embedded_trust_root();
        assert_eq!(root.version(), 1);
        assert_eq!(root.document.signature_threshold, 1);
        assert_eq!(root.document.root_keys.len(), 1);
        assert_eq!(root.document.root_keys[0].key_id, "D2AAF1CBF67BDFF4");
        assert!(root.catalog_key("F5AE02E7593C64D9").unwrap().accepts(1));
        assert!(root.expires_at_unix_seconds() > root.issued_at_unix_seconds());
    }

    #[test]
    fn optional_authority_ceilings_only_move_toward_revocation() {
        assert!(!widens_optional_ceiling(None, None));
        assert!(!widens_optional_ceiling(None, Some(7)));
        assert!(!widens_optional_ceiling(Some(7), Some(6)));
        assert!(widens_optional_ceiling(Some(7), Some(8)));
        assert!(widens_optional_ceiling(Some(7), None));
    }

    #[test]
    fn catalog_revocation_cutoff_is_explicit_and_non_retroactive() {
        let key = CatalogTrustKey {
            key_id: "F5AE02E7593C64D9".into(),
            public_key_base64: "RWTZZDxZ5wKu9QcABWE2Sy7ZEg6xQhQW+vVVclypgEu8QnjbnNbZmQvi".into(),
            first_sequence: 2,
            last_sequence: None,
            revoked_at_sequence: Some(4),
        };
        assert!(!key.accepts(1));
        assert!(key.accepts(2));
        assert!(key.accepts(3));
        assert!(!key.accepts(4));
    }

    #[test]
    fn rotation_requires_current_and_candidate_thresholds() {
        let current = embedded_trust_root();
        let candidate = parse_trust_root(
            ROOT_V2,
            TrustRootOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap();
        let bundle = parse_signature_bundle(ROOT_V2_SIGNATURES).unwrap();
        verify_root_transition(&current, &candidate, ROOT_V2, &bundle).unwrap();

        let old_only = TrustSignatureBundle {
            schema: TRUST_SIGNATURE_SCHEMA.into(),
            signatures: vec![bundle.signatures[0].clone()],
        };
        let error = verify_root_transition(&current, &candidate, ROOT_V2, &old_only).unwrap_err();
        assert!(error.contains("candidate-root signatures"), "{error}");

        let new_only = TrustSignatureBundle {
            schema: TRUST_SIGNATURE_SCHEMA.into(),
            signatures: vec![bundle.signatures[1].clone()],
        };
        let error = verify_root_transition(&current, &candidate, ROOT_V2, &new_only).unwrap_err();
        assert!(error.contains("current-root signatures"), "{error}");
    }

    #[test]
    fn rotation_signatures_bind_exact_root_bytes() {
        let current = embedded_trust_root();
        let candidate = parse_trust_root(
            ROOT_V2,
            TrustRootOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap();
        let bundle = parse_signature_bundle(ROOT_V2_SIGNATURES).unwrap();
        let mut tampered = ROOT_V2.to_vec();
        let offset = tampered
            .windows(b"15552000".len())
            .position(|window| window == b"15552000")
            .unwrap();
        tampered[offset] = b'2';
        let error = verify_root_transition(&current, &candidate, &tampered, &bundle).unwrap_err();
        assert!(error.contains("signature verification failed"), "{error}");
    }

    #[test]
    fn rotation_rejects_a_candidate_that_predates_the_active_root() {
        let current = embedded_trust_root();
        let mut document: serde_json::Value = serde_json::from_slice(ROOT_V2).unwrap();
        document["issued_at_unix_seconds"] = serde_json::json!(1786665599_u64);
        let bytes = serde_json::to_vec(&document).unwrap();
        let candidate = parse_trust_root(
            &bytes,
            TrustRootOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap();
        let bundle = parse_signature_bundle(ROOT_V2_SIGNATURES).unwrap();
        let error = verify_root_transition(&current, &candidate, &bytes, &bundle).unwrap_err();
        assert!(error.contains("predates the active root"), "{error}");
    }

    #[test]
    fn rotation_cannot_weaken_catalog_expiration_policy() {
        let current = embedded_trust_root();
        let bundle = parse_signature_bundle(ROOT_V2_SIGNATURES).unwrap();

        let mut document: serde_json::Value = serde_json::from_slice(ROOT_V2).unwrap();
        document["catalog_policy"]["expiration_required_from_sequence"] =
            serde_json::json!(current.expiration_required_from_sequence() + 1);
        let bytes = serde_json::to_vec(&document).unwrap();
        let candidate = parse_trust_root(
            &bytes,
            TrustRootOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap();
        let error = verify_root_transition(&current, &candidate, &bytes, &bundle).unwrap_err();
        assert!(
            error.contains("weakened catalog expiration policy"),
            "{error}"
        );

        let mut document: serde_json::Value = serde_json::from_slice(ROOT_V2).unwrap();
        document["catalog_policy"]["max_validity_seconds"] =
            serde_json::json!(current.max_catalog_validity_seconds() + 1);
        let bytes = serde_json::to_vec(&document).unwrap();
        let candidate = parse_trust_root(
            &bytes,
            TrustRootOrigin::Signed {
                source: LOCAL_IMPORT_SOURCE.into(),
            },
        )
        .unwrap();
        let error = verify_root_transition(&current, &candidate, &bytes, &bundle).unwrap_err();
        assert!(
            error.contains("weakened catalog expiration policy"),
            "{error}"
        );
    }

    #[test]
    fn signature_bundles_require_distinct_canonical_key_ids() {
        let mut document: serde_json::Value = serde_json::from_slice(ROOT_V2_SIGNATURES).unwrap();
        document["signatures"][0]["key_id"] = serde_json::json!("d2aaf1cbf67bdff4");
        let error = parse_signature_bundle(&serde_json::to_vec(&document).unwrap()).unwrap_err();
        assert!(error.contains("invalid model trust-root signature bundle"));

        let mut document: serde_json::Value = serde_json::from_slice(ROOT_V2_SIGNATURES).unwrap();
        let duplicate_key = document["signatures"][0]["key_id"].clone();
        document["signatures"][1]["key_id"] = duplicate_key;
        let error = parse_signature_bundle(&serde_json::to_vec(&document).unwrap()).unwrap_err();
        assert!(error.contains("invalid model trust-root signature bundle"));
    }

    #[test]
    fn root_expiration_boundary_is_closed() {
        let root = embedded_trust_root();
        require_fresh_root(&root, root.expires_at_unix_seconds() - 1).unwrap();
        let error = require_fresh_root(&root, root.expires_at_unix_seconds()).unwrap_err();
        assert!(error.contains("expired"), "{error}");
    }
}
