//! Authenticated, recoverable application-update transactions.
//!
//! The update engine deliberately does not replace a running executable.  It
//! verifies a signed release manifest, stages an exact candidate plus the
//! matching last-known-good payload, and atomically switches one small active
//! installation record.  A stable launcher or platform package integration
//! consumes that record.  Until startup health is confirmed, recovery switches
//! the record back without network access and without lowering the monotonic
//! anti-rollback floor.

use crate::{fault_injection, AtomicOutput, CommitMode};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use fs2::FileExt as _;
use minisign_verify::{PublicKey, Signature};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use url::Url;

pub const UPDATE_MANIFEST_SCHEMA: &str = "denoize-update-manifest-v1";
pub const UPDATE_MANIFEST_VERIFICATION_SCHEMA: &str = "denoize-update-manifest-verification-v1";
pub const UPDATE_BUNDLE_SCHEMA: &str = "denoize-update-bundle-v1";
pub const UPDATE_DOWNLOAD_SCHEMA: &str = "denoize-update-download-v1";
pub const UPDATE_CHECK_SCHEMA: &str = "denoize-update-check-v1";
pub const UPDATE_DRY_RUN_SCHEMA: &str = "denoize-update-dry-run-v1";
pub const UPDATE_STATUS_SCHEMA: &str = "denoize-update-status-v1";
pub const UPDATE_HEALTH_SCHEMA: &str = "denoize-update-health-v1";
pub const UPDATE_APPLY_SCHEMA: &str = "denoize-update-apply-v1";
pub const UPDATE_SCHEMA_VERSION: u32 = 1;

pub const DEFAULT_UPDATE_MANIFEST_URL: &str = "https://github.com/penguin425/denoize/releases/latest/download/denoize-update-manifest-v1.json";
pub const DEFAULT_UPDATE_MANIFEST_SIGNATURE_URL: &str = "https://github.com/penguin425/denoize/releases/latest/download/denoize-update-manifest-v1.json.sig";

const UPDATE_STATE_SCHEMA: &str = "denoize-update-state-v1";
const UPDATE_STATE_FILE: &str = "state-v1.json";
const UPDATE_LOCK_FILE: &str = "update-v1.lock";
const UPDATE_SLOT_DIRECTORY: &str = "slots-v1";
const UPDATE_STAGING_DIRECTORY: &str = "staging-v1";
const UPDATE_SLOT_MANIFEST_FILE: &str = "manifest-v1.json";
const UPDATE_SLOT_SIGNATURE_FILE: &str = "manifest-v1.json.sig";
const UPDATE_BUNDLE_MAGIC: &[u8] = b"denoize-update-bundle-v1\n";
const UPDATE_BUNDLE_HEADER_SCHEMA: &str = "denoize-update-bundle-header-v1";
const OFFICIAL_UPDATE_KEY_ID: &str = "F5AE02E7593C64D9";

/// The same public key embedded in the Desktop Tauri updater configuration.
/// It is an outer-Base64-wrapped minisign public-key document.
pub const OFFICIAL_UPDATE_PUBLIC_KEY: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IEY1QUUwMkU3NTkzQzY0RDkKUldUWlpEeFo1d0t1OVFjQUJXRTJTeTdaRWc2eFFoUVcrdlZWY2x5cGdFdThRbmpibk5iWm1RdmkK";

const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_SIGNATURE_BYTES: u64 = 16 * 1024;
const MAX_BUNDLE_HEADER_BYTES: u64 = 1024 * 1024;
const MAX_ARTIFACT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_EVIDENCE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_BUNDLE_BYTES: u64 = 5 * 1024 * 1024 * 1024;
const DEFAULT_MAX_STAGING_BYTES: u64 = 5 * 1024 * 1024 * 1024;
const MAX_PLATFORMS: usize = 64;
const MAX_ROLLBACKS: usize = 8;
const MAX_DIAGNOSTICS: usize = 128;
const MAX_FAILED_SLOTS: usize = 8;
const MAX_JSON_BYTES: usize = 4 * 1024 * 1024;
const COPY_BUFFER_BYTES: usize = 128 * 1024;
const MAX_UPDATE_REDIRECTS: usize = 5;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateFingerprint {
    pub len: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateRemoteFile {
    pub name: String,
    pub url: String,
    pub fingerprint: UpdateFingerprint,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UpdateActivationKind {
    PortableExecutable,
    MacosAppArchive,
    AppImage,
    DebPackage,
    NsisInstaller,
    MsiInstaller,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdatePayload {
    pub version: String,
    pub sequence: u64,
    pub activation: UpdateActivationKind,
    pub artifact: UpdateRemoteFile,
    pub sbom: UpdateRemoteFile,
    pub provenance: UpdateRemoteFile,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateRollbackPayload {
    pub from_version: String,
    pub from_sequence: u64,
    pub bundle_url: String,
    pub payload: UpdatePayload,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdatePlatform {
    pub platform: String,
    pub candidate: UpdatePayload,
    pub rollbacks: Vec<UpdateRollbackPayload>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateCompatibility {
    pub accepted_from_versions: Vec<String>,
    pub minimum_state_schema_version: u32,
    pub maximum_state_schema_version: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateRollbackPolicy {
    pub retained_last_known_good: u32,
    pub health_timeout_seconds: u64,
    pub maximum_start_attempts: u32,
    pub manual_recovery: bool,
    pub network_required_for_recovery: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateManifest {
    pub schema: String,
    pub schema_version: u32,
    pub channel: String,
    pub version: String,
    pub sequence: u64,
    pub published_unix_seconds: u64,
    pub source_commit: String,
    pub compatibility: UpdateCompatibility,
    pub rollback_policy: UpdateRollbackPolicy,
    pub platforms: Vec<UpdatePlatform>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateManifestVerification {
    pub schema: String,
    pub schema_version: u32,
    pub channel: String,
    pub version: String,
    pub sequence: u64,
    pub manifest_sha256: String,
    pub signing_key_id: String,
    pub platform_count: usize,
}

#[derive(Clone, Debug)]
pub struct VerifiedUpdateManifest {
    pub manifest: UpdateManifest,
    pub manifest_bytes: Vec<u8>,
    pub signature_bytes: Vec<u8>,
    pub verification: UpdateManifestVerification,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StableVersion {
    major: u64,
    minor: u64,
    patch: u64,
}

impl StableVersion {
    fn parse(raw: &str) -> Result<Self, String> {
        if raw.is_empty() || raw.len() > 64 || raw.starts_with('v') {
            return Err(format!("invalid stable update version: {raw}"));
        }
        let fields = raw.split('.').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!("update version must be MAJOR.MINOR.PATCH: {raw}"));
        }
        let parse = |field: &str| -> Result<u64, String> {
            if field.is_empty()
                || (field.len() > 1 && field.starts_with('0'))
                || !field.bytes().all(|byte| byte.is_ascii_digit())
            {
                return Err(format!("invalid stable update version: {raw}"));
            }
            field
                .parse::<u64>()
                .map_err(|_| format!("update version component is too large: {raw}"))
        };
        Ok(Self {
            major: parse(fields[0])?,
            minor: parse(fields[1])?,
            patch: parse(fields[2])?,
        })
    }

    fn sequence(&self) -> Result<u64, String> {
        if self.major > 9_000 || self.minor > 999_999 || self.patch > 999_999 {
            return Err("update version exceeds the v1 sequence range".into());
        }
        self.major
            .checked_mul(1_000_000_000_000)
            .and_then(|value| value.checked_add(self.minor * 1_000_000))
            .and_then(|value| value.checked_add(self.patch))
            .ok_or_else(|| "update version sequence overflowed".to_string())
    }
}

impl UpdateManifest {
    pub fn from_file(
        manifest_path: impl AsRef<Path>,
        signature_path: impl AsRef<Path>,
        public_key_path: Option<&Path>,
    ) -> Result<VerifiedUpdateManifest, String> {
        let manifest_bytes = read_bounded_regular(
            manifest_path.as_ref(),
            MAX_MANIFEST_BYTES,
            "update manifest",
        )?;
        let signature_bytes = read_bounded_regular(
            signature_path.as_ref(),
            MAX_SIGNATURE_BYTES,
            "update manifest signature",
        )?;
        let public_key = read_update_public_key(public_key_path)?;
        verify_update_manifest_bytes(manifest_bytes, signature_bytes, &public_key)
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        let encoded = serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize update manifest: {error}"))?;
        if encoded.len() >= MAX_MANIFEST_BYTES as usize {
            return Err("update manifest exceeds its JSON size limit".into());
        }
        Ok(encoded)
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != UPDATE_MANIFEST_SCHEMA || self.schema_version != UPDATE_SCHEMA_VERSION {
            return Err("update manifest has an unsupported schema or version".into());
        }
        validate_identifier("update channel", &self.channel, 64)?;
        let candidate_version = StableVersion::parse(&self.version)?;
        if candidate_version.sequence()? != self.sequence {
            return Err("update manifest sequence does not match its version".into());
        }
        if self.published_unix_seconds == 0 {
            return Err("update manifest publication time must be positive".into());
        }
        validate_source_commit(&self.source_commit)?;
        self.compatibility.validate(&self.version)?;
        self.rollback_policy.validate()?;
        if self.platforms.is_empty() || self.platforms.len() > MAX_PLATFORMS {
            return Err(format!(
                "update manifest must contain 1..={MAX_PLATFORMS} platforms"
            ));
        }
        let accepted = self
            .compatibility
            .accepted_from_versions
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut previous = None::<&str>;
        for platform in &self.platforms {
            validate_identifier("update platform", &platform.platform, 128)?;
            if previous.is_some_and(|value| value >= platform.platform.as_str()) {
                return Err("update manifest platforms must be unique and sorted".into());
            }
            previous = Some(&platform.platform);
            platform.candidate.validate(MAX_ARTIFACT_BYTES)?;
            if platform.candidate.version != self.version
                || platform.candidate.sequence != self.sequence
            {
                return Err(format!(
                    "candidate payload for {} does not match manifest version",
                    platform.platform
                ));
            }
            if platform.rollbacks.is_empty() || platform.rollbacks.len() > MAX_ROLLBACKS {
                return Err(format!(
                    "platform {} must contain 1..={MAX_ROLLBACKS} rollback payloads",
                    platform.platform
                ));
            }
            let mut rollback_versions = BTreeSet::new();
            for rollback in &platform.rollbacks {
                rollback.validate()?;
                if !accepted.contains(&rollback.from_version) {
                    return Err(format!(
                        "platform {} has an undeclared rollback version {}",
                        platform.platform, rollback.from_version
                    ));
                }
                if !rollback_versions.insert(rollback.from_version.clone()) {
                    return Err(format!(
                        "platform {} repeats rollback version {}",
                        platform.platform, rollback.from_version
                    ));
                }
            }
            if rollback_versions != accepted {
                return Err(format!(
                    "platform {} does not cover every accepted source version",
                    platform.platform
                ));
            }
        }
        ensure_json_size(self, "update manifest", MAX_MANIFEST_BYTES as usize)
    }

    pub fn platform(&self, platform: &str) -> Result<&UpdatePlatform, String> {
        self.platforms
            .iter()
            .find(|candidate| candidate.platform == platform)
            .ok_or_else(|| format!("update manifest does not support platform {platform}"))
    }
}

impl UpdateCompatibility {
    fn validate(&self, candidate: &str) -> Result<(), String> {
        if self.minimum_state_schema_version == 0
            || self.minimum_state_schema_version > self.maximum_state_schema_version
            || self.maximum_state_schema_version > UPDATE_SCHEMA_VERSION
        {
            return Err("update compatibility state-schema range is invalid".into());
        }
        if self.accepted_from_versions.is_empty()
            || self.accepted_from_versions.len() > MAX_ROLLBACKS
        {
            return Err("update compatibility must declare bounded source versions".into());
        }
        let candidate = StableVersion::parse(candidate)?;
        let mut previous = None::<StableVersion>;
        for raw in &self.accepted_from_versions {
            let version = StableVersion::parse(raw)?;
            if version.sequence()? >= candidate.sequence()? {
                return Err("accepted update source versions must precede the candidate".into());
            }
            let sequence = version.sequence()?;
            if previous
                .as_ref()
                .is_some_and(|value| value.sequence().is_ok_and(|previous| previous >= sequence))
            {
                return Err("accepted update source versions must be unique and sorted".into());
            }
            previous = Some(version);
        }
        Ok(())
    }
}

impl UpdateRollbackPolicy {
    fn validate(&self) -> Result<(), String> {
        if self.retained_last_known_good != 1 {
            return Err(
                "update rollback policy must retain exactly one last-known-good install".into(),
            );
        }
        if !(30..=7 * 24 * 60 * 60).contains(&self.health_timeout_seconds) {
            return Err("update health timeout must be between 30 seconds and 7 days".into());
        }
        if !(1..=16).contains(&self.maximum_start_attempts) {
            return Err("update maximum start attempts must be between 1 and 16".into());
        }
        if !self.manual_recovery || self.network_required_for_recovery {
            return Err(
                "update rollback policy must allow manual offline recovery without network".into(),
            );
        }
        Ok(())
    }
}

impl UpdatePayload {
    fn validate(&self, artifact_limit: u64) -> Result<(), String> {
        let version = StableVersion::parse(&self.version)?;
        if version.sequence()? != self.sequence {
            return Err("update payload sequence does not match its version".into());
        }
        self.artifact.validate("update artifact", artifact_limit)?;
        self.sbom.validate("update SBOM", MAX_EVIDENCE_BYTES)?;
        self.provenance
            .validate("update provenance", MAX_EVIDENCE_BYTES)?;
        let names = [
            self.artifact.name.as_str(),
            self.sbom.name.as_str(),
            self.provenance.name.as_str(),
            UPDATE_SLOT_MANIFEST_FILE,
            UPDATE_SLOT_SIGNATURE_FILE,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        if names.len() != 5 {
            return Err("update payload file names must be unique".into());
        }
        Ok(())
    }
}

impl UpdateRollbackPayload {
    fn validate(&self) -> Result<(), String> {
        let version = StableVersion::parse(&self.from_version)?;
        if version.sequence()? != self.from_sequence
            || self.payload.version != self.from_version
            || self.payload.sequence != self.from_sequence
        {
            return Err("rollback payload identity does not match its source version".into());
        }
        validate_https_url("update bundle URL", &self.bundle_url)?;
        self.payload.validate(MAX_ARTIFACT_BYTES)
    }
}

impl UpdateRemoteFile {
    fn validate(&self, context: &str, limit: u64) -> Result<(), String> {
        validate_filename(context, &self.name)?;
        validate_https_url(&format!("{context} URL"), &self.url)?;
        self.fingerprint.validate(context, limit)
    }
}

impl UpdateFingerprint {
    fn validate(&self, context: &str, limit: u64) -> Result<(), String> {
        if self.len == 0 || self.len > limit {
            return Err(format!("{context} length is outside its bounded range"));
        }
        validate_sha256(context, &self.sha256)
    }
}

pub fn verify_update_manifest_bytes(
    manifest_bytes: Vec<u8>,
    signature_bytes: Vec<u8>,
    public_key_text: &str,
) -> Result<VerifiedUpdateManifest, String> {
    if manifest_bytes.is_empty() || manifest_bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err("update manifest is empty or exceeds the 4 MiB limit".into());
    }
    if signature_bytes.is_empty() || signature_bytes.len() as u64 > MAX_SIGNATURE_BYTES {
        return Err("update manifest signature is empty or exceeds the 16 KiB limit".into());
    }
    let (public_key, key_id) = parse_minisign_public_key(public_key_text)?;
    let signature_text = decode_minisign_signature(&signature_bytes)?;
    let signature = Signature::decode(&signature_text)
        .map_err(|error| format!("invalid update manifest signature: {error}"))?;
    public_key
        .verify(&manifest_bytes, &signature, false)
        .map_err(|error| format!("update manifest signature verification failed: {error}"))?;
    let manifest: UpdateManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("parse update manifest: {error}"))?;
    manifest.validate()?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let verification = UpdateManifestVerification {
        schema: UPDATE_MANIFEST_VERIFICATION_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        channel: manifest.channel.clone(),
        version: manifest.version.clone(),
        sequence: manifest.sequence,
        manifest_sha256,
        signing_key_id: key_id,
        platform_count: manifest.platforms.len(),
    };
    Ok(VerifiedUpdateManifest {
        manifest,
        manifest_bytes,
        signature_bytes,
        verification,
    })
}

pub fn fetch_update_manifest(
    manifest_url: &str,
    signature_url: &str,
    public_key_path: Option<&Path>,
) -> Result<VerifiedUpdateManifest, String> {
    let manifest_bytes =
        download_update_bytes(manifest_url, MAX_MANIFEST_BYTES, "update manifest")?;
    let signature_bytes = download_update_bytes(
        signature_url,
        MAX_SIGNATURE_BYTES,
        "update manifest signature",
    )?;
    let public_key = read_update_public_key(public_key_path)?;
    verify_update_manifest_bytes(manifest_bytes, signature_bytes, &public_key)
}

fn parse_minisign_public_key(text: &str) -> Result<(PublicKey, String), String> {
    let direct = parse_minisign_public_key_text(text);
    match direct {
        Ok(value) => Ok(value),
        Err(direct_error) if !text.trim().contains(['\n', '\r']) => {
            let decoded = BASE64_STANDARD
                .decode(text.trim())
                .map_err(|_| "update public key is neither minisign text nor outer Base64")?;
            let decoded = String::from_utf8(decoded)
                .map_err(|_| "decoded update public key is not UTF-8".to_string())?;
            parse_minisign_public_key_text(&decoded).map_err(|_| direct_error)
        }
        Err(error) => Err(error),
    }
}

fn parse_minisign_public_key_text(text: &str) -> Result<(PublicKey, String), String> {
    let mut records = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("untrusted comment:"));
    let encoded = records
        .next()
        .ok_or_else(|| "update public key has no key data".to_string())?;
    if records.next().is_some() {
        return Err("update public key contains multiple key records".into());
    }
    let decoded = BASE64_STANDARD
        .decode(encoded)
        .map_err(|_| "update public key data is not Base64".to_string())?;
    if decoded.len() != 42 {
        return Err("update public key has an invalid length".into());
    }
    let raw_id = u64::from_le_bytes(
        decoded[2..10]
            .try_into()
            .map_err(|_| "update public key ID is invalid".to_string())?,
    );
    let key_id = format!("{raw_id:016X}");
    let key = PublicKey::from_base64(encoded)
        .map_err(|error| format!("invalid update public key: {error}"))?;
    Ok((key, key_id))
}

fn decode_minisign_signature(bytes: &[u8]) -> Result<String, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| "update manifest signature is not UTF-8".to_string())?
        .trim();
    let decoded = if text.starts_with("untrusted comment:") {
        text.to_string()
    } else {
        let decoded = BASE64_STANDARD.decode(text).map_err(|_| {
            "update signature is neither minisign text nor Tauri outer Base64".to_string()
        })?;
        if decoded.len() as u64 > MAX_SIGNATURE_BYTES {
            return Err("decoded update signature exceeds the 16 KiB limit".into());
        }
        String::from_utf8(decoded)
            .map_err(|_| "decoded update signature is not UTF-8".to_string())?
    };
    let lines = decoded.trim().lines().collect::<Vec<_>>();
    if lines.len() != 4
        || !lines[0].starts_with("untrusted comment:")
        || !lines[2].starts_with("trusted comment: ")
    {
        return Err("update signature must contain exactly one minisign record".into());
    }
    Ok(decoded.trim().to_string())
}

fn validate_identifier(context: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty()
        || value.len() > maximum
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b'+' | b'-')
        })
    {
        return Err(format!("{context} is not a bounded portable identifier"));
    }
    Ok(())
}

fn validate_filename(context: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 255
        || value == "."
        || value == ".."
        || value.contains(['/', '\\', ':'])
        || value
            .chars()
            .any(|character| character.is_control() || character == '\0')
    {
        return Err(format!("{context} name is not one portable file name"));
    }
    Ok(())
}

fn validate_sha256(context: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{context} SHA-256 must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_source_commit(value: &str) -> Result<(), String> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("update source commit must be a 40-character lowercase Git object ID".into());
    }
    Ok(())
}

fn validate_https_url(context: &str, raw: &str) -> Result<(), String> {
    if raw.len() > 4096 {
        return Err(format!("{context} exceeds the 4096-byte limit"));
    }
    let url = Url::parse(raw).map_err(|error| format!("invalid {context}: {error}"))?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.host_str().is_none()
        || url.fragment().is_some()
    {
        return Err(format!(
            "{context} must be credential-free HTTPS without a fragment"
        ));
    }
    Ok(())
}

fn update_http_response(
    source: &str,
    maximum: u64,
    context: &str,
) -> Result<(ureq::Response, Option<u64>), String> {
    validate_https_url(&format!("{context} URL"), source)?;
    let source = Url::parse(source).map_err(|error| format!("invalid {context} URL: {error}"))?;
    let agent = ureq::AgentBuilder::new()
        .redirects(0)
        .user_agent(concat!("denoize-update/", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .timeout(Duration::from_secs(60 * 60))
        .build();
    let mut current = source.clone();
    for redirect_count in 0..=MAX_UPDATE_REDIRECTS {
        let response = match agent
            .get(current.as_str())
            .set("Accept-Encoding", "identity")
            .call()
        {
            Ok(response) => response,
            Err(ureq::Error::Status(_, response)) => response,
            Err(ureq::Error::Transport(error)) => {
                return Err(format!(
                    "{context} download from {} failed: {}",
                    redacted_update_url(&current),
                    error.kind()
                ));
            }
        };
        if matches!(response.status(), 301 | 302 | 303 | 307 | 308) {
            if redirect_count == MAX_UPDATE_REDIRECTS {
                return Err(format!(
                    "{context} download exceeded {MAX_UPDATE_REDIRECTS} redirects"
                ));
            }
            let location = response
                .header("Location")
                .ok_or_else(|| format!("{context} redirect omitted Location"))?;
            let next = current
                .join(location)
                .map_err(|_| format!("{context} redirect has an invalid Location"))?;
            validate_https_url(&format!("{context} redirect URL"), next.as_str())?;
            current = next;
            continue;
        }
        if response.status() != 200 {
            return Err(format!(
                "{context} download from {} returned HTTP {}",
                redacted_update_url(&current),
                response.status()
            ));
        }
        if response
            .header("Content-Encoding")
            .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
        {
            return Err(format!(
                "{context} response used an unexpected content encoding"
            ));
        }
        let content_length = response
            .header("Content-Length")
            .map(|value| {
                value
                    .parse::<u64>()
                    .map_err(|_| format!("{context} response has invalid Content-Length"))
            })
            .transpose()?;
        if content_length.is_some_and(|length| length == 0 || length > maximum) {
            return Err(format!(
                "{context} exceeds its {maximum}-byte download limit"
            ));
        }
        return Ok((response, content_length));
    }
    unreachable!("bounded update redirect loop always returns")
}

fn download_update_bytes(source: &str, maximum: u64, context: &str) -> Result<Vec<u8>, String> {
    let (response, content_length) = update_http_response(source, maximum, context)?;
    let mut bytes = Vec::with_capacity(
        content_length
            .and_then(|length| usize::try_from(length).ok())
            .unwrap_or(0),
    );
    response
        .into_reader()
        .take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read downloaded {context}: {error}"))?;
    if bytes.is_empty() || bytes.len() as u64 > maximum {
        return Err(format!(
            "{context} exceeds its bounded non-empty download range"
        ));
    }
    if content_length.is_some_and(|length| length != bytes.len() as u64) {
        return Err(format!(
            "{context} response ended at a different Content-Length"
        ));
    }
    Ok(bytes)
}

fn download_update_to_atomic(
    source: &str,
    output: &Path,
    maximum: u64,
    context: &str,
) -> Result<(AtomicOutput, u64), String> {
    let (response, content_length) = update_http_response(source, maximum, context)?;
    let mut transaction = AtomicOutput::new(output)?;
    let mut reader = response.into_reader();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut downloaded = 0_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("read downloaded {context}: {error}"))?;
        if read == 0 {
            break;
        }
        downloaded = downloaded
            .checked_add(read as u64)
            .ok_or_else(|| format!("{context} download length overflowed"))?;
        if downloaded > maximum {
            return Err(format!(
                "{context} exceeds its {maximum}-byte download limit"
            ));
        }
        transaction
            .file_mut()
            .write_all(&buffer[..read])
            .map_err(|error| format!("stage downloaded {context}: {error}"))?;
    }
    if downloaded == 0 || content_length.is_some_and(|length| length != downloaded) {
        return Err(format!(
            "{context} response ended at a different bounded length"
        ));
    }
    Ok((transaction, downloaded))
}

fn redacted_update_url(source: &Url) -> String {
    let mut redacted = source.clone();
    redacted.set_query(None);
    redacted.set_fragment(None);
    redacted.to_string()
}

fn ensure_json_size<T: Serialize>(value: &T, context: &str, limit: usize) -> Result<(), String> {
    let encoded =
        serde_json::to_vec(value).map_err(|error| format!("serialize {context}: {error}"))?;
    if encoded.len() >= limit {
        return Err(format!("{context} exceeds its JSON size limit"));
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn now_unix_seconds() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .map_err(|_| "system clock is before the Unix epoch".to_string())
}

fn read_bounded_regular(path: &Path, maximum: u64, context: &str) -> Result<Vec<u8>, String> {
    let (file, len) = crate::input::open_regular_file(path, context)?;
    if len == 0 || len > maximum {
        return Err(format!(
            "{context} {} is empty or exceeds the {maximum}-byte limit",
            path.display()
        ));
    }
    let capacity =
        usize::try_from(len).map_err(|_| format!("{context} length does not fit this platform"))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {context} {}: {error}", path.display()))?;
    if bytes.len() as u64 != len {
        return Err(format!("{context} changed while it was read"));
    }
    Ok(bytes)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct UpdateBundleHeader {
    schema: String,
    schema_version: u32,
    platform: String,
    from_version: String,
    manifest: EmbeddedFile,
    signature: EmbeddedFile,
    candidate_artifact: EmbeddedFile,
    candidate_sbom: EmbeddedFile,
    candidate_provenance: EmbeddedFile,
    rollback_artifact: EmbeddedFile,
    rollback_sbom: EmbeddedFile,
    rollback_provenance: EmbeddedFile,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct EmbeddedFile {
    name: String,
    fingerprint: UpdateFingerprint,
}

#[derive(Clone, Debug)]
pub struct UpdateBundleBuildRequest {
    pub platform: String,
    pub from_version: String,
    pub manifest_path: PathBuf,
    pub signature_path: PathBuf,
    pub candidate_artifact_path: PathBuf,
    pub candidate_sbom_path: PathBuf,
    pub candidate_provenance_path: PathBuf,
    pub rollback_artifact_path: PathBuf,
    pub rollback_sbom_path: PathBuf,
    pub rollback_provenance_path: PathBuf,
    pub public_key_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateBundleInfo {
    pub schema: String,
    pub schema_version: u32,
    pub bundle_sha256: String,
    pub size_bytes: u64,
    pub platform: String,
    pub channel: String,
    pub from_version: String,
    pub from_sequence: u64,
    pub candidate_version: String,
    pub candidate_sequence: u64,
    pub manifest_sha256: String,
    pub signing_key_id: String,
    pub candidate_artifact: UpdateFingerprint,
    pub rollback_artifact: UpdateFingerprint,
    pub evidence_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateDownloadReport {
    pub schema: String,
    pub schema_version: u32,
    pub platform: String,
    pub from_version: String,
    pub candidate_version: String,
    pub candidate_sequence: u64,
    pub manifest_sha256: String,
    pub signing_key_id: String,
    pub bundle_sha256: String,
    pub size_bytes: u64,
    pub output_file_name: String,
    pub outcome: String,
}

struct OpenComponent {
    file: File,
    expected: UpdateRemoteFile,
}

struct PreparedUpdateBundle {
    file: File,
    file_len: u64,
    bundle_sha256: String,
    verified: VerifiedUpdateManifest,
    platform: UpdatePlatform,
    rollback: UpdateRollbackPayload,
    offsets: BundleOffsets,
}

#[derive(Clone, Copy)]
struct BundleOffsets {
    candidate_artifact: u64,
    candidate_sbom: u64,
    candidate_provenance: u64,
    rollback_artifact: u64,
    rollback_sbom: u64,
    rollback_provenance: u64,
}

pub fn build_update_bundle(
    output: impl AsRef<Path>,
    request: &UpdateBundleBuildRequest,
) -> Result<UpdateBundleInfo, String> {
    let verified = UpdateManifest::from_file(
        &request.manifest_path,
        &request.signature_path,
        request.public_key_path.as_deref(),
    )?;
    let platform = verified.manifest.platform(&request.platform)?.clone();
    let rollback = platform
        .rollbacks
        .iter()
        .find(|value| value.from_version == request.from_version)
        .cloned()
        .ok_or_else(|| {
            format!(
                "update platform {} does not accept source version {}",
                request.platform, request.from_version
            )
        })?;

    let mut candidate_artifact = open_verified_component(
        &request.candidate_artifact_path,
        &platform.candidate.artifact,
        MAX_ARTIFACT_BYTES,
        "candidate update artifact",
    )?;
    let mut candidate_sbom = open_verified_component(
        &request.candidate_sbom_path,
        &platform.candidate.sbom,
        MAX_EVIDENCE_BYTES,
        "candidate update SBOM",
    )?;
    let mut candidate_provenance = open_verified_component(
        &request.candidate_provenance_path,
        &platform.candidate.provenance,
        MAX_EVIDENCE_BYTES,
        "candidate update provenance",
    )?;
    let mut rollback_artifact = open_verified_component(
        &request.rollback_artifact_path,
        &rollback.payload.artifact,
        MAX_ARTIFACT_BYTES,
        "rollback update artifact",
    )?;
    let mut rollback_sbom = open_verified_component(
        &request.rollback_sbom_path,
        &rollback.payload.sbom,
        MAX_EVIDENCE_BYTES,
        "rollback update SBOM",
    )?;
    let mut rollback_provenance = open_verified_component(
        &request.rollback_provenance_path,
        &rollback.payload.provenance,
        MAX_EVIDENCE_BYTES,
        "rollback update provenance",
    )?;

    let header = UpdateBundleHeader {
        schema: UPDATE_BUNDLE_HEADER_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        platform: platform.platform.clone(),
        from_version: rollback.from_version.clone(),
        manifest: EmbeddedFile {
            name: "denoize-update-manifest-v1.json".into(),
            fingerprint: UpdateFingerprint {
                len: verified.manifest_bytes.len() as u64,
                sha256: verified.verification.manifest_sha256.clone(),
            },
        },
        signature: EmbeddedFile {
            name: "denoize-update-manifest-v1.json.sig".into(),
            fingerprint: UpdateFingerprint {
                len: verified.signature_bytes.len() as u64,
                sha256: sha256_bytes(&verified.signature_bytes),
            },
        },
        candidate_artifact: embedded_from_remote(&platform.candidate.artifact),
        candidate_sbom: embedded_from_remote(&platform.candidate.sbom),
        candidate_provenance: embedded_from_remote(&platform.candidate.provenance),
        rollback_artifact: embedded_from_remote(&rollback.payload.artifact),
        rollback_sbom: embedded_from_remote(&rollback.payload.sbom),
        rollback_provenance: embedded_from_remote(&rollback.payload.provenance),
    };
    validate_bundle_header(&header, &verified, &platform, &rollback)?;
    let mut header_bytes = serde_json::to_vec(&header)
        .map_err(|error| format!("serialize update bundle header: {error}"))?;
    if header_bytes.len() as u64 > MAX_BUNDLE_HEADER_BYTES {
        return Err("update bundle header exceeds the 1 MiB limit".into());
    }

    let expected_len = expected_bundle_len(&header, header_bytes.len() as u64)?;
    if expected_len > MAX_BUNDLE_BYTES {
        return Err("update bundle exceeds the 5 GiB limit".into());
    }
    let output = output.as_ref();
    let mut transaction = AtomicOutput::new(output)?;
    let mut digest = Sha256::new();
    write_hashed(transaction.file_mut(), &mut digest, UPDATE_BUNDLE_MAGIC)?;
    write_hashed(
        transaction.file_mut(),
        &mut digest,
        &(header_bytes.len() as u64).to_le_bytes(),
    )?;
    write_hashed(transaction.file_mut(), &mut digest, &header_bytes)?;
    header_bytes.fill(0);
    write_hashed(
        transaction.file_mut(),
        &mut digest,
        &verified.manifest_bytes,
    )?;
    write_hashed(
        transaction.file_mut(),
        &mut digest,
        &verified.signature_bytes,
    )?;
    copy_component(transaction.file_mut(), &mut digest, &mut candidate_artifact)?;
    copy_component(transaction.file_mut(), &mut digest, &mut candidate_sbom)?;
    copy_component(
        transaction.file_mut(),
        &mut digest,
        &mut candidate_provenance,
    )?;
    copy_component(transaction.file_mut(), &mut digest, &mut rollback_artifact)?;
    copy_component(transaction.file_mut(), &mut digest, &mut rollback_sbom)?;
    copy_component(
        transaction.file_mut(),
        &mut digest,
        &mut rollback_provenance,
    )?;
    if transaction
        .file_mut()
        .stream_position()
        .map_err(|error| format!("inspect staged update bundle length: {error}"))?
        != expected_len
    {
        return Err("staged update bundle length differs from its header".into());
    }
    let bundle_sha256 = format!("{:x}", digest.finalize());
    fault_injection::hit("update-bundle.before-commit")?;
    transaction.commit(CommitMode::NoClobber)?;
    let info = bundle_info(&verified, &platform, &rollback, bundle_sha256, expected_len)?;
    Ok(info)
}

pub fn inspect_update_bundle(
    path: impl AsRef<Path>,
    public_key_path: Option<&Path>,
) -> Result<UpdateBundleInfo, String> {
    let public_key = read_update_public_key(public_key_path)?;
    let prepared = prepare_update_bundle(path.as_ref(), &public_key)?;
    bundle_info(
        &prepared.verified,
        &prepared.platform,
        &prepared.rollback,
        prepared.bundle_sha256,
        prepared.file_len,
    )
}

pub fn download_update_bundle(
    verified: &VerifiedUpdateManifest,
    platform: &str,
    from_version: &str,
    output: impl AsRef<Path>,
    public_key_path: Option<&Path>,
) -> Result<UpdateDownloadReport, String> {
    verified.manifest.validate()?;
    validate_identifier("requested update platform", platform, 128)?;
    StableVersion::parse(from_version)?;
    let selected = verified.manifest.platform(platform)?;
    let rollback = selected
        .rollbacks
        .iter()
        .find(|candidate| candidate.from_version == from_version)
        .ok_or_else(|| {
            format!("update platform {platform} does not accept source version {from_version}")
        })?;
    let bundle_url = Url::parse(&rollback.bundle_url)
        .map_err(|error| format!("invalid update bundle URL: {error}"))?;
    let output_file_name = bundle_url
        .path_segments()
        .and_then(|segments| segments.filter(|segment| !segment.is_empty()).next_back())
        .ok_or_else(|| "update bundle URL does not name a file".to_string())?
        .to_string();
    validate_filename("update bundle URL", &output_file_name)?;
    let maximum_download =
        payload_staging_bytes(&selected.candidate, &rollback.payload)?.min(MAX_BUNDLE_BYTES);
    let (mut transaction, downloaded) = download_update_to_atomic(
        &rollback.bundle_url,
        output.as_ref(),
        maximum_download,
        "update bundle",
    )?;
    transaction
        .file_mut()
        .sync_all()
        .map_err(|error| format!("sync downloaded update bundle: {error}"))?;
    let info = inspect_update_bundle(transaction.staged_path(), public_key_path)?;
    if info.platform != platform
        || info.from_version != from_version
        || info.candidate_version != verified.manifest.version
        || info.candidate_sequence != verified.manifest.sequence
        || info.manifest_sha256 != verified.verification.manifest_sha256
        || info.signing_key_id != verified.verification.signing_key_id
    {
        return Err("downloaded update bundle differs from the selected signed manifest".into());
    }
    if info.size_bytes != downloaded {
        return Err("downloaded update bundle length changed during verification".into());
    }
    transaction.commit(CommitMode::NoClobber)?;
    Ok(UpdateDownloadReport {
        schema: UPDATE_DOWNLOAD_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        platform: info.platform,
        from_version: info.from_version,
        candidate_version: info.candidate_version,
        candidate_sequence: info.candidate_sequence,
        manifest_sha256: info.manifest_sha256,
        signing_key_id: info.signing_key_id,
        bundle_sha256: info.bundle_sha256,
        size_bytes: info.size_bytes,
        output_file_name,
        outcome: "downloaded-and-verified".into(),
    })
}

fn read_update_public_key(path: Option<&Path>) -> Result<String, String> {
    let text = match path {
        Some(path) => {
            String::from_utf8(read_bounded_regular(path, 64 * 1024, "update public key")?)
                .map_err(|_| "update public key is not UTF-8".to_string())
        }
        None => Ok(OFFICIAL_UPDATE_PUBLIC_KEY.to_string()),
    }?;
    let (_, key_id) = parse_minisign_public_key(&text)?;
    if path.is_none() && key_id != OFFICIAL_UPDATE_KEY_ID {
        return Err("embedded official update public key ID is invalid".into());
    }
    Ok(text)
}

fn prepare_update_bundle(path: &Path, public_key: &str) -> Result<PreparedUpdateBundle, String> {
    let (mut file, file_len) = crate::input::open_regular_file(path, "update bundle")?;
    if file_len == 0 || file_len > MAX_BUNDLE_BYTES {
        return Err("update bundle is empty or exceeds the 5 GiB limit".into());
    }
    let mut digest = Sha256::new();
    let mut magic = vec![0_u8; UPDATE_BUNDLE_MAGIC.len()];
    read_hashed(&mut file, &mut digest, &mut magic, "update bundle magic")?;
    if magic != UPDATE_BUNDLE_MAGIC {
        return Err("update bundle magic is invalid".into());
    }
    let mut length_bytes = [0_u8; 8];
    read_hashed(
        &mut file,
        &mut digest,
        &mut length_bytes,
        "update bundle header length",
    )?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_BUNDLE_HEADER_BYTES {
        return Err("update bundle header length is invalid".into());
    }
    let mut header_bytes = vec![
        0_u8;
        usize::try_from(header_len).map_err(|_| {
            "update bundle header does not fit this platform"
        })?
    ];
    read_hashed(
        &mut file,
        &mut digest,
        &mut header_bytes,
        "update bundle header",
    )?;
    let header: UpdateBundleHeader = serde_json::from_slice(&header_bytes)
        .map_err(|error| format!("parse update bundle header: {error}"))?;
    header_bytes.fill(0);
    let expected_len = expected_bundle_len(&header, header_len)?;
    if expected_len != file_len {
        return Err(format!(
            "update bundle length mismatch: header declares {expected_len}, file has {file_len}"
        ));
    }

    let manifest_bytes = read_bundle_blob(
        &mut file,
        &mut digest,
        &header.manifest,
        MAX_MANIFEST_BYTES,
        "embedded update manifest",
    )?;
    let signature_bytes = read_bundle_blob(
        &mut file,
        &mut digest,
        &header.signature,
        MAX_SIGNATURE_BYTES,
        "embedded update signature",
    )?;
    let verified = verify_update_manifest_bytes(manifest_bytes, signature_bytes, public_key)?;
    let platform = verified.manifest.platform(&header.platform)?.clone();
    let rollback = platform
        .rollbacks
        .iter()
        .find(|value| value.from_version == header.from_version)
        .cloned()
        .ok_or_else(|| {
            "update bundle source version is not accepted by its signed manifest".to_string()
        })?;
    validate_bundle_header(&header, &verified, &platform, &rollback)?;

    let candidate_artifact = file.stream_position().map_err(|error| error.to_string())?;
    hash_bundle_blob(
        &mut file,
        &mut digest,
        &header.candidate_artifact,
        "candidate update artifact",
    )?;
    let candidate_sbom = file.stream_position().map_err(|error| error.to_string())?;
    hash_bundle_blob(
        &mut file,
        &mut digest,
        &header.candidate_sbom,
        "candidate update SBOM",
    )?;
    let candidate_provenance = file.stream_position().map_err(|error| error.to_string())?;
    hash_bundle_blob(
        &mut file,
        &mut digest,
        &header.candidate_provenance,
        "candidate update provenance",
    )?;
    let rollback_artifact = file.stream_position().map_err(|error| error.to_string())?;
    hash_bundle_blob(
        &mut file,
        &mut digest,
        &header.rollback_artifact,
        "rollback update artifact",
    )?;
    let rollback_sbom = file.stream_position().map_err(|error| error.to_string())?;
    hash_bundle_blob(
        &mut file,
        &mut digest,
        &header.rollback_sbom,
        "rollback update SBOM",
    )?;
    let rollback_provenance = file.stream_position().map_err(|error| error.to_string())?;
    hash_bundle_blob(
        &mut file,
        &mut digest,
        &header.rollback_provenance,
        "rollback update provenance",
    )?;
    if file.stream_position().map_err(|error| error.to_string())? != file_len {
        return Err("update bundle contains trailing bytes".into());
    }
    let metadata_after = file
        .metadata()
        .map_err(|error| format!("inspect update bundle after verification: {error}"))?;
    if metadata_after.len() != file_len {
        return Err("update bundle changed while it was verified".into());
    }
    Ok(PreparedUpdateBundle {
        file,
        file_len,
        bundle_sha256: format!("{:x}", digest.finalize()),
        verified,
        platform,
        rollback,
        offsets: BundleOffsets {
            candidate_artifact,
            candidate_sbom,
            candidate_provenance,
            rollback_artifact,
            rollback_sbom,
            rollback_provenance,
        },
    })
}

fn embedded_from_remote(value: &UpdateRemoteFile) -> EmbeddedFile {
    EmbeddedFile {
        name: value.name.clone(),
        fingerprint: value.fingerprint.clone(),
    }
}

fn validate_bundle_header(
    header: &UpdateBundleHeader,
    verified: &VerifiedUpdateManifest,
    platform: &UpdatePlatform,
    rollback: &UpdateRollbackPayload,
) -> Result<(), String> {
    if header.schema != UPDATE_BUNDLE_HEADER_SCHEMA
        || header.schema_version != UPDATE_SCHEMA_VERSION
        || header.platform != platform.platform
        || header.from_version != rollback.from_version
    {
        return Err("update bundle header identity is invalid".into());
    }
    validate_filename("embedded update manifest", &header.manifest.name)?;
    validate_filename("embedded update signature", &header.signature.name)?;
    header
        .manifest
        .fingerprint
        .validate("embedded update manifest", MAX_MANIFEST_BYTES)?;
    header
        .signature
        .fingerprint
        .validate("embedded update signature", MAX_SIGNATURE_BYTES)?;
    if header.manifest.fingerprint.sha256 != verified.verification.manifest_sha256
        || header.manifest.fingerprint.len != verified.manifest_bytes.len() as u64
        || header.signature.fingerprint.sha256 != sha256_bytes(&verified.signature_bytes)
        || header.signature.fingerprint.len != verified.signature_bytes.len() as u64
    {
        return Err("update bundle manifest records do not match the authenticated bytes".into());
    }
    let expected = [
        (&header.candidate_artifact, &platform.candidate.artifact),
        (&header.candidate_sbom, &platform.candidate.sbom),
        (&header.candidate_provenance, &platform.candidate.provenance),
        (&header.rollback_artifact, &rollback.payload.artifact),
        (&header.rollback_sbom, &rollback.payload.sbom),
        (&header.rollback_provenance, &rollback.payload.provenance),
    ];
    for (embedded, remote) in expected {
        if embedded.name != remote.name || embedded.fingerprint != remote.fingerprint {
            return Err(format!(
                "update bundle component {} differs from the signed manifest",
                remote.name
            ));
        }
    }
    Ok(())
}

fn expected_bundle_len(header: &UpdateBundleHeader, header_len: u64) -> Result<u64, String> {
    let mut total = (UPDATE_BUNDLE_MAGIC.len() as u64)
        .checked_add(8)
        .and_then(|value| value.checked_add(header_len))
        .ok_or_else(|| "update bundle length overflowed".to_string())?;
    for blob in [
        &header.manifest,
        &header.signature,
        &header.candidate_artifact,
        &header.candidate_sbom,
        &header.candidate_provenance,
        &header.rollback_artifact,
        &header.rollback_sbom,
        &header.rollback_provenance,
    ] {
        total = total
            .checked_add(blob.fingerprint.len)
            .ok_or_else(|| "update bundle length overflowed".to_string())?;
    }
    Ok(total)
}

fn bundle_info(
    verified: &VerifiedUpdateManifest,
    platform: &UpdatePlatform,
    rollback: &UpdateRollbackPayload,
    bundle_sha256: String,
    size_bytes: u64,
) -> Result<UpdateBundleInfo, String> {
    validate_sha256("update bundle", &bundle_sha256)?;
    let evidence_bytes = platform
        .candidate
        .sbom
        .fingerprint
        .len
        .checked_add(platform.candidate.provenance.fingerprint.len)
        .and_then(|value| value.checked_add(rollback.payload.sbom.fingerprint.len))
        .and_then(|value| value.checked_add(rollback.payload.provenance.fingerprint.len))
        .ok_or_else(|| "update evidence byte total overflowed".to_string())?;
    Ok(UpdateBundleInfo {
        schema: UPDATE_BUNDLE_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        bundle_sha256,
        size_bytes,
        platform: platform.platform.clone(),
        channel: verified.manifest.channel.clone(),
        from_version: rollback.from_version.clone(),
        from_sequence: rollback.from_sequence,
        candidate_version: platform.candidate.version.clone(),
        candidate_sequence: platform.candidate.sequence,
        manifest_sha256: verified.verification.manifest_sha256.clone(),
        signing_key_id: verified.verification.signing_key_id.clone(),
        candidate_artifact: platform.candidate.artifact.fingerprint.clone(),
        rollback_artifact: rollback.payload.artifact.fingerprint.clone(),
        evidence_bytes,
    })
}

fn open_verified_component(
    path: &Path,
    expected: &UpdateRemoteFile,
    limit: u64,
    context: &str,
) -> Result<OpenComponent, String> {
    let (mut file, len) = crate::input::open_regular_file(path, context)?;
    if len != expected.fingerprint.len || len == 0 || len > limit {
        return Err(format!(
            "{context} length differs from signed manifest: expected {}, got {len}",
            expected.fingerprint.len
        ));
    }
    let digest = hash_open_file(&mut file, len, context)?;
    if digest != expected.fingerprint.sha256 {
        return Err(format!("{context} SHA-256 differs from signed manifest"));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {context}: {error}"))?;
    Ok(OpenComponent {
        file,
        expected: expected.clone(),
    })
}

fn copy_component(
    output: &mut File,
    bundle_digest: &mut Sha256,
    component: &mut OpenComponent,
) -> Result<(), String> {
    let expected_len = component.expected.fingerprint.len;
    let mut remaining = expected_len;
    let mut component_digest = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "update component read length does not fit this platform")?;
        let read = component
            .file
            .read(&mut buffer[..request])
            .map_err(|error| {
                format!("read update component {}: {error}", component.expected.name)
            })?;
        if read == 0 {
            return Err(format!(
                "update component {} ended before its signed length",
                component.expected.name
            ));
        }
        output
            .write_all(&buffer[..read])
            .map_err(|error| format!("write update bundle component: {error}"))?;
        bundle_digest.update(&buffer[..read]);
        component_digest.update(&buffer[..read]);
        remaining -= read as u64;
    }
    let mut extra = [0_u8; 1];
    if component
        .file
        .read(&mut extra)
        .map_err(|error| format!("finish update component read: {error}"))?
        != 0
    {
        return Err(format!(
            "update component {} grew while the bundle was built",
            component.expected.name
        ));
    }
    let digest = format!("{:x}", component_digest.finalize());
    if digest != component.expected.fingerprint.sha256 {
        return Err(format!(
            "update component {} changed after initial verification",
            component.expected.name
        ));
    }
    Ok(())
}

fn write_hashed(output: &mut File, digest: &mut Sha256, bytes: &[u8]) -> Result<(), String> {
    output
        .write_all(bytes)
        .map_err(|error| format!("write staged update bundle: {error}"))?;
    digest.update(bytes);
    Ok(())
}

fn read_hashed(
    input: &mut File,
    digest: &mut Sha256,
    bytes: &mut [u8],
    context: &str,
) -> Result<(), String> {
    input
        .read_exact(bytes)
        .map_err(|error| format!("read {context}: {error}"))?;
    digest.update(bytes);
    Ok(())
}

fn read_bundle_blob(
    input: &mut File,
    bundle_digest: &mut Sha256,
    record: &EmbeddedFile,
    limit: u64,
    context: &str,
) -> Result<Vec<u8>, String> {
    record.fingerprint.validate(context, limit)?;
    let len = usize::try_from(record.fingerprint.len)
        .map_err(|_| format!("{context} length does not fit this platform"))?;
    let mut bytes = vec![0_u8; len];
    read_hashed(input, bundle_digest, &mut bytes, context)?;
    if sha256_bytes(&bytes) != record.fingerprint.sha256 {
        return Err(format!("{context} SHA-256 differs from the bundle header"));
    }
    Ok(bytes)
}

fn hash_bundle_blob(
    input: &mut File,
    bundle_digest: &mut Sha256,
    record: &EmbeddedFile,
    context: &str,
) -> Result<(), String> {
    let mut remaining = record.fingerprint.len;
    let mut component_digest = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| format!("{context} read length does not fit this platform"))?;
        input
            .read_exact(&mut buffer[..request])
            .map_err(|error| format!("read {context}: {error}"))?;
        bundle_digest.update(&buffer[..request]);
        component_digest.update(&buffer[..request]);
        remaining -= request as u64;
    }
    let digest = format!("{:x}", component_digest.finalize());
    if digest != record.fingerprint.sha256 {
        return Err(format!(
            "{context} SHA-256 differs from the signed manifest"
        ));
    }
    Ok(())
}

fn hash_open_file(file: &mut File, expected_len: u64, context: &str) -> Result<String, String> {
    let mut remaining = expected_len;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| format!("{context} read length does not fit this platform"))?;
        let read = file
            .read(&mut buffer[..request])
            .map_err(|error| format!("read {context}: {error}"))?;
        if read == 0 {
            return Err(format!("{context} ended before its declared length"));
        }
        digest.update(&buffer[..read]);
        remaining -= read as u64;
    }
    let mut extra = [0_u8; 1];
    if file
        .read(&mut extra)
        .map_err(|error| format!("finish {context} read: {error}"))?
        != 0
    {
        return Err(format!("{context} grew while it was read"));
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct StoredUpdateSlot {
    slot_id: String,
    version: String,
    sequence: u64,
    platform: String,
    activation: UpdateActivationKind,
    manifest_sha256: String,
    artifact_name: String,
    artifact: UpdateFingerprint,
    sbom_name: String,
    sbom: UpdateFingerprint,
    provenance_name: String,
    provenance: UpdateFingerprint,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PendingUpdateHealth {
    candidate_slot_id: String,
    last_known_good_slot_id: String,
    health_token: String,
    applied_unix_seconds: u64,
    deadline_unix_seconds: u64,
    start_attempts: u32,
    maximum_start_attempts: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FailedUpdateSlot {
    slot_id: String,
    version: String,
    sequence: u64,
    artifact_sha256: String,
    failed_unix_seconds: u64,
    reason_code: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateDiagnostic {
    pub generation: u64,
    pub unix_seconds: u64,
    pub code: String,
    pub from_version: Option<String>,
    pub to_version: Option<String>,
    pub manifest_sha256: Option<String>,
    pub artifact_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct UpdateState {
    schema: String,
    schema_version: u32,
    generation: u64,
    channel: String,
    platform: String,
    highest_accepted_sequence: u64,
    highest_manifest_sha256: String,
    active: StoredUpdateSlot,
    last_known_good: StoredUpdateSlot,
    pending_health: Option<PendingUpdateHealth>,
    failed_slots: Vec<FailedUpdateSlot>,
    diagnostics: Vec<UpdateDiagnostic>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateSlotStatus {
    pub slot_id: String,
    pub version: String,
    pub sequence: u64,
    pub platform: String,
    pub activation: UpdateActivationKind,
    pub manifest_sha256: String,
    pub artifact_name: String,
    pub artifact: UpdateFingerprint,
    pub sbom: UpdateFingerprint,
    pub provenance: UpdateFingerprint,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateStatusReport {
    pub schema: String,
    pub schema_version: u32,
    pub managed: bool,
    pub generation: u64,
    pub channel: Option<String>,
    pub platform: Option<String>,
    pub phase: String,
    pub highest_accepted_sequence: Option<u64>,
    pub active: Option<UpdateSlotStatus>,
    pub last_known_good: Option<UpdateSlotStatus>,
    pub health_deadline_unix_seconds: Option<u64>,
    pub start_attempts: Option<u32>,
    pub maximum_start_attempts: Option<u32>,
    pub failed_slot_count: usize,
    pub diagnostics: Vec<UpdateDiagnostic>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateCheckReport {
    pub schema: String,
    pub schema_version: u32,
    pub channel: String,
    pub platform: String,
    pub current_version: String,
    pub current_sequence: u64,
    pub candidate_version: String,
    pub candidate_sequence: u64,
    pub manifest_sha256: String,
    pub signing_key_id: String,
    pub decision: String,
    pub reason_codes: Vec<String>,
    pub bundle_url: Option<String>,
    pub download_upper_bound_bytes: Option<u64>,
    pub read_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateDryRunReport {
    pub schema: String,
    pub schema_version: u32,
    pub channel: String,
    pub platform: String,
    pub current_version: String,
    pub candidate_version: String,
    pub candidate_sequence: u64,
    pub manifest_sha256: String,
    pub bundle_sha256: String,
    pub decision: String,
    pub reason_codes: Vec<String>,
    pub staging_bytes: u64,
    pub maximum_staging_bytes: u64,
    pub destination_actions: Vec<String>,
    pub preserves_last_known_good: bool,
    pub recovery_requires_network: bool,
    pub read_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateApplyReport {
    pub schema: String,
    pub schema_version: u32,
    pub channel: String,
    pub platform: String,
    pub from_version: String,
    pub candidate_version: String,
    pub candidate_sequence: u64,
    pub bundle_sha256: String,
    pub manifest_sha256: String,
    pub active_slot_id: String,
    pub last_known_good_slot_id: String,
    pub health_token: String,
    pub health_deadline_unix_seconds: u64,
    pub activation: UpdateActivationKind,
    pub outcome: String,
    pub relaunch_required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateHealthReport {
    pub schema: String,
    pub schema_version: u32,
    pub action: String,
    pub running_version: String,
    pub active_version: Option<String>,
    pub last_known_good_version: Option<String>,
    pub health_token: Option<String>,
    pub start_attempts: Option<u32>,
    pub maximum_start_attempts: Option<u32>,
    pub deadline_unix_seconds: Option<u64>,
    pub relaunch_required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpdateActivationTarget {
    pub version: String,
    pub platform: String,
    pub activation: UpdateActivationKind,
    pub artifact_path: PathBuf,
    pub artifact: UpdateFingerprint,
}

struct UpdateStore {
    root: PathBuf,
}

struct UpdateStoreLock {
    _file: File,
}

impl Drop for UpdateStoreLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self._file);
    }
}

impl StoredUpdateSlot {
    fn validate(&self) -> Result<(), String> {
        validate_identifier("update slot ID", &self.slot_id, 128)?;
        let version = StableVersion::parse(&self.version)?;
        if version.sequence()? != self.sequence {
            return Err("stored update slot sequence does not match its version".into());
        }
        validate_identifier("stored update platform", &self.platform, 128)?;
        validate_sha256("stored update manifest", &self.manifest_sha256)?;
        validate_filename("stored update artifact", &self.artifact_name)?;
        validate_filename("stored update SBOM", &self.sbom_name)?;
        validate_filename("stored update provenance", &self.provenance_name)?;
        let names = [
            self.artifact_name.as_str(),
            self.sbom_name.as_str(),
            self.provenance_name.as_str(),
            UPDATE_SLOT_MANIFEST_FILE,
            UPDATE_SLOT_SIGNATURE_FILE,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        if names.len() != 5 {
            return Err("stored update slot file names must be unique".into());
        }
        self.artifact
            .validate("stored update artifact", MAX_ARTIFACT_BYTES)?;
        self.sbom
            .validate("stored update SBOM", MAX_EVIDENCE_BYTES)?;
        self.provenance
            .validate("stored update provenance", MAX_EVIDENCE_BYTES)?;
        if self.slot_id != slot_id(self.sequence, &self.artifact.sha256) {
            return Err("stored update slot ID does not match its artifact".into());
        }
        Ok(())
    }
}

impl UpdateState {
    fn validate(&self) -> Result<(), String> {
        if self.schema != UPDATE_STATE_SCHEMA || self.schema_version != UPDATE_SCHEMA_VERSION {
            return Err("update state has an unsupported schema or version".into());
        }
        if self.generation == 0 {
            return Err("update state generation must be positive".into());
        }
        validate_identifier("update state channel", &self.channel, 64)?;
        validate_identifier("update state platform", &self.platform, 128)?;
        validate_sha256(
            "highest accepted update manifest",
            &self.highest_manifest_sha256,
        )?;
        self.active.validate()?;
        self.last_known_good.validate()?;
        if self.active.platform != self.platform || self.last_known_good.platform != self.platform {
            return Err("update slots do not match the state platform".into());
        }
        if self.active.sequence > self.highest_accepted_sequence
            || self.last_known_good.sequence > self.highest_accepted_sequence
        {
            return Err("update slot exceeds the monotonic accepted sequence".into());
        }
        if self.active.sequence == self.highest_accepted_sequence
            && self.active.manifest_sha256 != self.highest_manifest_sha256
        {
            return Err("active update does not match the highest accepted manifest".into());
        }
        match &self.pending_health {
            Some(pending) => {
                if pending.candidate_slot_id != self.active.slot_id
                    || pending.last_known_good_slot_id != self.last_known_good.slot_id
                    || self.active.slot_id == self.last_known_good.slot_id
                {
                    return Err(
                        "pending update health does not bind two distinct state slots".into(),
                    );
                }
                validate_health_token(&pending.health_token)?;
                if pending.applied_unix_seconds == 0
                    || pending.deadline_unix_seconds <= pending.applied_unix_seconds
                    || !(1..=16).contains(&pending.maximum_start_attempts)
                    || pending.start_attempts > pending.maximum_start_attempts
                {
                    return Err("pending update health policy is invalid".into());
                }
            }
            None if self.active != self.last_known_good => {
                return Err(
                    "healthy update state must make the active slot last-known-good".into(),
                );
            }
            None => {}
        }
        if self.failed_slots.len() > MAX_FAILED_SLOTS {
            return Err("update state contains too many failed-slot records".into());
        }
        for failed in &self.failed_slots {
            validate_identifier("failed update slot ID", &failed.slot_id, 128)?;
            let version = StableVersion::parse(&failed.version)?;
            if version.sequence()? != failed.sequence
                || failed.failed_unix_seconds == 0
                || failed.sequence > self.highest_accepted_sequence
            {
                return Err("failed update slot record has an invalid identity".into());
            }
            validate_sha256("failed update artifact", &failed.artifact_sha256)?;
            validate_reason_code(&failed.reason_code)?;
        }
        if self.diagnostics.len() > MAX_DIAGNOSTICS {
            return Err("update state contains too many diagnostics".into());
        }
        let mut previous_generation = 0;
        for diagnostic in &self.diagnostics {
            if diagnostic.generation == 0
                || diagnostic.generation < previous_generation
                || diagnostic.generation > self.generation
                || diagnostic.unix_seconds == 0
            {
                return Err("update diagnostic ordering is invalid".into());
            }
            previous_generation = diagnostic.generation;
            validate_identifier("update diagnostic code", &diagnostic.code, 128)?;
            if let Some(version) = &diagnostic.from_version {
                StableVersion::parse(version)?;
            }
            if let Some(version) = &diagnostic.to_version {
                StableVersion::parse(version)?;
            }
            if let Some(digest) = &diagnostic.manifest_sha256 {
                validate_sha256("diagnostic update manifest", digest)?;
            }
            if let Some(digest) = &diagnostic.artifact_sha256 {
                validate_sha256("diagnostic update artifact", digest)?;
            }
        }
        ensure_json_size(self, "update state", MAX_JSON_BYTES)
    }
}

impl UpdateStore {
    fn read_only(requested_root: &Path) -> Result<Self, String> {
        let root = resolve_state_root(requested_root)?;
        match std::fs::symlink_metadata(&root) {
            Ok(_) => {
                let root = require_private_update_directory(&root, "update state directory")?;
                Ok(Self { root })
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self { root }),
            Err(error) => Err(format!(
                "inspect update state directory {}: {error}",
                root.display()
            )),
        }
    }

    fn open(requested_root: &Path) -> Result<Self, String> {
        let root = resolve_state_root(requested_root)?;
        let root = prepare_private_update_directory(&root, "update state directory")?;
        prepare_private_update_directory(
            &root.join(UPDATE_SLOT_DIRECTORY),
            "update slot directory",
        )?;
        prepare_private_update_directory(
            &root.join(UPDATE_STAGING_DIRECTORY),
            "update staging directory",
        )?;
        Ok(Self { root })
    }

    fn open_existing(requested_root: &Path) -> Result<Self, String> {
        let root = resolve_state_root(requested_root)?;
        let root = require_private_update_directory(&root, "update state directory")?;
        require_private_update_directory(
            &root.join(UPDATE_SLOT_DIRECTORY),
            "update slot directory",
        )?;
        require_private_update_directory(
            &root.join(UPDATE_STAGING_DIRECTORY),
            "update staging directory",
        )?;
        Ok(Self { root })
    }

    fn state_path(&self) -> PathBuf {
        self.root.join(UPDATE_STATE_FILE)
    }

    fn lock_exclusive(&self) -> Result<UpdateStoreLock, String> {
        let path = self.root.join(UPDATE_LOCK_FILE);
        let file = open_or_create_private_control_file(&path)?;
        file.try_lock_exclusive().map_err(|error| {
            if error.kind() == std::io::ErrorKind::WouldBlock
                || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
            {
                format!("another updater holds the state lock: {}", path.display())
            } else {
                format!("lock update state {}: {error}", path.display())
            }
        })?;
        Ok(UpdateStoreLock { _file: file })
    }

    fn read_state_optional(&self) -> Result<Option<UpdateState>, String> {
        if !self.root.exists() {
            return Ok(None);
        }
        require_private_update_directory(&self.root, "update state directory")?;
        let path = self.state_path();
        let Some(file) =
            open_private_regular_optional(&path, "update state", MAX_JSON_BYTES as u64)?
        else {
            return Ok(None);
        };
        let mut bytes = Vec::new();
        file.take(MAX_JSON_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read update state {}: {error}", path.display()))?;
        if bytes.is_empty() || bytes.len() > MAX_JSON_BYTES {
            return Err("update state is empty or exceeds its JSON size limit".into());
        }
        let state: UpdateState = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse update state {}: {error}", path.display()))?;
        state.validate()?;
        Ok(Some(state))
    }

    fn write_state(&self, state: &UpdateState) -> Result<(), String> {
        state.validate()?;
        let mut bytes = serde_json::to_vec_pretty(state)
            .map_err(|error| format!("serialize update state: {error}"))?;
        bytes.push(b'\n');
        if bytes.len() > MAX_JSON_BYTES {
            return Err("update state exceeds its JSON size limit".into());
        }
        let path = self.state_path();
        let exists =
            open_private_regular_optional(&path, "update state", MAX_JSON_BYTES as u64)?.is_some();
        let mut output = if exists {
            AtomicOutput::new(&path)?
        } else {
            AtomicOutput::new_private(&path)?
        };
        output
            .file_mut()
            .write_all(&bytes)
            .map_err(|error| format!("write staged update state {}: {error}", path.display()))?;
        output.commit(if exists {
            CommitMode::Replace
        } else {
            CommitMode::NoClobber
        })?;
        sync_directory(&self.root, "update state directory")
    }

    fn cleanup_staging(&self) -> Result<(), String> {
        let staging = require_private_update_directory(
            &self.root.join(UPDATE_STAGING_DIRECTORY),
            "update staging directory",
        )?;
        for entry in std::fs::read_dir(&staging).map_err(|error| {
            format!(
                "read update staging directory {}: {error}",
                staging.display()
            )
        })? {
            let entry = entry.map_err(|error| format!("read update staging entry: {error}"))?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| "update staging entry name is not UTF-8".to_string())?;
            if !name.starts_with("stage-") {
                return Err(format!("unexpected update staging entry: {name}"));
            }
            validate_identifier("update staging entry", &name, 192)?;
            remove_private_slot_directory(&entry.path(), "update staging entry")?;
        }
        sync_directory(&staging, "update staging directory")
    }

    fn stage_slot(
        &self,
        prepared: &mut PreparedUpdateBundle,
        candidate: bool,
        payload: &UpdatePayload,
    ) -> Result<StoredUpdateSlot, String> {
        payload.validate(MAX_ARTIFACT_BYTES)?;
        let slot = StoredUpdateSlot {
            slot_id: slot_id(payload.sequence, &payload.artifact.fingerprint.sha256),
            version: payload.version.clone(),
            sequence: payload.sequence,
            platform: prepared.platform.platform.clone(),
            activation: payload.activation,
            manifest_sha256: prepared.verified.verification.manifest_sha256.clone(),
            artifact_name: payload.artifact.name.clone(),
            artifact: payload.artifact.fingerprint.clone(),
            sbom_name: payload.sbom.name.clone(),
            sbom: payload.sbom.fingerprint.clone(),
            provenance_name: payload.provenance.name.clone(),
            provenance: payload.provenance.fingerprint.clone(),
        };
        slot.validate()?;
        let slots = require_private_update_directory(
            &self.root.join(UPDATE_SLOT_DIRECTORY),
            "update slot directory",
        )?;
        let destination = slots.join(&slot.slot_id);
        if destination.exists() {
            self.verify_slot(&slot)?;
            return Ok(slot);
        }
        let nonce = now_unix_nanos()?;
        let staging_parent = require_private_update_directory(
            &self.root.join(UPDATE_STAGING_DIRECTORY),
            "update staging directory",
        )?;
        let stage_name = format!("stage-{}-{}-{nonce}", slot.slot_id, std::process::id());
        validate_identifier("update staging entry", &stage_name, 192)?;
        let stage = prepare_private_update_directory(
            &staging_parent.join(stage_name),
            "update slot stage",
        )?;
        let result = (|| {
            write_private_slot_bytes(
                &stage.join(UPDATE_SLOT_MANIFEST_FILE),
                &prepared.verified.manifest_bytes,
                "staged update manifest",
            )?;
            write_private_slot_bytes(
                &stage.join(UPDATE_SLOT_SIGNATURE_FILE),
                &prepared.verified.signature_bytes,
                "staged update signature",
            )?;
            let (artifact_offset, sbom_offset, provenance_offset) = if candidate {
                (
                    prepared.offsets.candidate_artifact,
                    prepared.offsets.candidate_sbom,
                    prepared.offsets.candidate_provenance,
                )
            } else {
                (
                    prepared.offsets.rollback_artifact,
                    prepared.offsets.rollback_sbom,
                    prepared.offsets.rollback_provenance,
                )
            };
            copy_bundle_component_to_slot(
                &mut prepared.file,
                artifact_offset,
                &stage.join(&slot.artifact_name),
                &slot.artifact,
                "staged update artifact",
            )?;
            copy_bundle_component_to_slot(
                &mut prepared.file,
                sbom_offset,
                &stage.join(&slot.sbom_name),
                &slot.sbom,
                "staged update SBOM",
            )?;
            copy_bundle_component_to_slot(
                &mut prepared.file,
                provenance_offset,
                &stage.join(&slot.provenance_name),
                &slot.provenance,
                "staged update provenance",
            )?;
            if prepared
                .file
                .metadata()
                .map_err(|error| format!("reinspect update bundle: {error}"))?
                .len()
                != prepared.file_len
            {
                return Err("update bundle changed while its slots were staged".into());
            }
            sync_directory(&stage, "staged update slot")?;
            std::fs::rename(&stage, &destination).map_err(|error| {
                format!(
                    "activate staged update slot {}: {error}",
                    destination.display()
                )
            })?;
            sync_directory(&slots, "update slot directory")?;
            self.verify_slot(&slot)
        })();
        if result.is_err() && stage.exists() {
            let _ = remove_private_slot_directory(&stage, "failed update slot stage");
        }
        result.map(|()| slot)
    }

    fn verify_slot(&self, slot: &StoredUpdateSlot) -> Result<(), String> {
        slot.validate()?;
        let directory = require_private_update_directory(
            &self.root.join(UPDATE_SLOT_DIRECTORY).join(&slot.slot_id),
            "stored update slot",
        )?;
        verify_private_file(
            &directory.join(&slot.artifact_name),
            &slot.artifact,
            "stored update artifact",
        )?;
        verify_private_file(
            &directory.join(&slot.sbom_name),
            &slot.sbom,
            "stored update SBOM",
        )?;
        verify_private_file(
            &directory.join(&slot.provenance_name),
            &slot.provenance,
            "stored update provenance",
        )?;
        let mut manifest = open_private_regular_optional(
            &directory.join(UPDATE_SLOT_MANIFEST_FILE),
            "stored update manifest",
            MAX_MANIFEST_BYTES,
        )?
        .ok_or_else(|| "stored update manifest is missing".to_string())?;
        let manifest_len = manifest
            .metadata()
            .map_err(|error| format!("inspect stored update manifest: {error}"))?
            .len();
        if manifest_len == 0
            || hash_open_file(&mut manifest, manifest_len, "stored update manifest")?
                != slot.manifest_sha256
        {
            return Err("stored update manifest fingerprint differs from update state".into());
        }
        open_private_regular_optional(
            &directory.join(UPDATE_SLOT_SIGNATURE_FILE),
            "stored update signature",
            MAX_SIGNATURE_BYTES,
        )?
        .ok_or_else(|| "stored update signature is missing".to_string())?;
        Ok(())
    }

    fn cleanup_unreferenced_slots(&self, state: &UpdateState) -> Result<(), String> {
        state.validate()?;
        let retained = [
            state.active.slot_id.as_str(),
            state.last_known_good.slot_id.as_str(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        let slots = require_private_update_directory(
            &self.root.join(UPDATE_SLOT_DIRECTORY),
            "update slot directory",
        )?;
        for entry in std::fs::read_dir(&slots)
            .map_err(|error| format!("read update slot directory {}: {error}", slots.display()))?
        {
            let entry = entry.map_err(|error| format!("read update slot entry: {error}"))?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| "update slot entry name is not UTF-8".to_string())?;
            validate_identifier("update slot entry", &name, 128)?;
            if !retained.contains(name.as_str()) {
                remove_private_slot_directory(&entry.path(), "unreferenced update slot")?;
            }
        }
        sync_directory(&slots, "update slot directory")
    }
}

fn resolve_state_root(requested: &Path) -> Result<PathBuf, String> {
    if !requested.is_absolute() {
        return Err("update state directory must be absolute".into());
    }
    if requested.file_name().is_none() {
        return Err("update state directory must name a bounded directory".into());
    }
    match std::fs::symlink_metadata(requested) {
        Ok(_) => Ok(requested.to_path_buf()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = requested
                .parent()
                .ok_or_else(|| "update state directory has no parent".to_string())?;
            let parent = std::fs::canonicalize(parent).map_err(|error| {
                format!(
                    "resolve update state parent directory {}: {error}",
                    parent.display()
                )
            })?;
            if !parent.is_dir() {
                return Err("update state parent path is not a directory".into());
            }
            Ok(parent.join(
                requested
                    .file_name()
                    .expect("state directory file name checked above"),
            ))
        }
        Err(error) => Err(format!(
            "inspect update state directory {}: {error}",
            requested.display()
        )),
    }
}

fn prepare_private_update_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    #[cfg(windows)]
    let create_result = crate::atomic_output::create_private_windows_directory(path);
    #[cfg(not(windows))]
    let create_result = std::fs::create_dir(path);
    match create_result {
        Ok(()) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                    .map_err(|error| format!("protect {label} {}: {error}", path.display()))?;
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(format!("create {label} {}: {error}", path.display())),
    }
    require_private_update_directory(path, label)
}

fn require_private_update_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(format!(
            "{label} must be a private non-symlink directory: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(format!(
                "{label} must be owned by the current user with mode 0700: {}",
                path.display()
            ));
        }
        crate::atomic_output::validate_unix_acl(path, path)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        };
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!(
                "{label} must not be a reparse point: {}",
                path.display()
            ));
        }
        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT);
        let directory = options
            .open(path)
            .map_err(|error| format!("open {label} {}: {error}", path.display()))?;
        crate::atomic_output::require_windows_private_acl(&directory).map_err(|error| {
            format!(
                "validate private ACL on {label} {}: {error}",
                path.display()
            )
        })?;
    }
    std::fs::canonicalize(path)
        .map_err(|error| format!("resolve {label} {}: {error}", path.display()))
}

fn configure_update_nofollow(options: &mut OpenOptions) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
}

fn require_private_update_file(file: &File, path: &Path, label: &str) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "{label} must be a regular file: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(format!("{label} must be owner-private: {}", path.display()));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!(
                "{label} must not be a reparse point: {}",
                path.display()
            ));
        }
        crate::atomic_output::require_windows_private_acl(file).map_err(|error| {
            format!(
                "validate private ACL on {label} {}: {error}",
                path.display()
            )
        })?;
    }
    Ok(())
}

fn open_private_regular_optional(
    path: &Path,
    label: &str,
    maximum_len: u64,
) -> Result<Option<File>, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    configure_update_nofollow(&mut options);
    let file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("open {label} {}: {error}", path.display())),
    };
    require_private_update_file(&file, path, label)?;
    let len = file
        .metadata()
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?
        .len();
    if len > maximum_len {
        return Err(format!("{label} exceeds its {maximum_len}-byte limit"));
    }
    Ok(Some(file))
}

fn open_or_create_private_control_file(path: &Path) -> Result<File, String> {
    #[cfg(windows)]
    let file = match crate::atomic_output::create_private_windows_control_file(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let mut options = OpenOptions::new();
            options.read(true).write(true).truncate(false);
            configure_update_nofollow(&mut options);
            options
                .open(path)
                .map_err(|error| format!("open update lock {}: {error}", path.display()))?
        }
        Err(error) => return Err(format!("create update lock {}: {error}", path.display())),
    };
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};
        let mut create = OpenOptions::new();
        create
            .create_new(true)
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
        match create.open(path) {
            Ok(file) => {
                file.set_permissions(std::fs::Permissions::from_mode(0o600))
                    .map_err(|error| format!("protect update lock {}: {error}", path.display()))?;
                file
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let mut options = OpenOptions::new();
                options.read(true).write(true).truncate(false);
                configure_update_nofollow(&mut options);
                options
                    .open(path)
                    .map_err(|error| format!("open update lock {}: {error}", path.display()))?
            }
            Err(error) => return Err(format!("create update lock {}: {error}", path.display())),
        }
    };
    #[cfg(not(any(unix, windows)))]
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|error| format!("open update lock {}: {error}", path.display()))?;
    require_private_update_file(&file, path, "update lock")?;
    Ok(file)
}

fn write_private_slot_bytes(path: &Path, bytes: &[u8], label: &str) -> Result<(), String> {
    if bytes.is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    let mut output = AtomicOutput::new_private(path)?;
    output
        .file_mut()
        .write_all(bytes)
        .map_err(|error| format!("write {label} {}: {error}", path.display()))?;
    output.commit(CommitMode::NoClobber)
}

fn copy_bundle_component_to_slot(
    bundle: &mut File,
    offset: u64,
    destination: &Path,
    expected: &UpdateFingerprint,
    label: &str,
) -> Result<(), String> {
    expected.validate(
        label,
        if label.contains("artifact") {
            MAX_ARTIFACT_BYTES
        } else {
            MAX_EVIDENCE_BYTES
        },
    )?;
    bundle
        .seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek to {label}: {error}"))?;
    let mut output = AtomicOutput::new_private(destination)?;
    let mut digest = Sha256::new();
    let mut remaining = expected.len;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| format!("{label} read length does not fit this platform"))?;
        bundle
            .read_exact(&mut buffer[..request])
            .map_err(|error| format!("read {label} from update bundle: {error}"))?;
        output
            .file_mut()
            .write_all(&buffer[..request])
            .map_err(|error| format!("write {label} {}: {error}", destination.display()))?;
        digest.update(&buffer[..request]);
        remaining -= request as u64;
    }
    if format!("{:x}", digest.finalize()) != expected.sha256 {
        return Err(format!("{label} changed after update bundle verification"));
    }
    output.commit(CommitMode::NoClobber)
}

fn verify_private_file(
    path: &Path,
    expected: &UpdateFingerprint,
    label: &str,
) -> Result<(), String> {
    let mut file = open_private_regular_optional(path, label, expected.len)?
        .ok_or_else(|| format!("{label} is missing: {}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?
        .len();
    if len != expected.len || hash_open_file(&mut file, len, label)? != expected.sha256 {
        return Err(format!("{label} fingerprint differs from update state"));
    }
    Ok(())
}

fn remove_private_slot_directory(path: &Path, label: &str) -> Result<(), String> {
    let directory = require_private_update_directory(path, label)?;
    for entry in std::fs::read_dir(&directory)
        .map_err(|error| format!("read {label} {}: {error}", directory.display()))?
    {
        let entry = entry.map_err(|error| format!("read {label} entry: {error}"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| format!("{label} entry name is not UTF-8"))?;
        validate_filename(label, &name)?;
        let file = open_private_regular_optional(&entry.path(), label, MAX_BUNDLE_BYTES)?
            .ok_or_else(|| format!("{label} entry disappeared during cleanup"))?;
        drop(file);
        std::fs::remove_file(entry.path())
            .map_err(|error| format!("remove {label} file {}: {error}", entry.path().display()))?;
    }
    std::fs::remove_dir(&directory)
        .map_err(|error| format!("remove {label} directory {}: {error}", directory.display()))
}

fn slot_id(sequence: u64, artifact_sha256: &str) -> String {
    format!("slot-{sequence}-{}", &artifact_sha256[..16])
}

fn now_unix_nanos() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .map_err(|_| "system clock is before the Unix epoch".to_string())
}

#[cfg(unix)]
fn sync_directory(path: &Path, label: &str) -> Result<(), String> {
    use std::os::unix::fs::OpenOptionsExt as _;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let directory = options
        .open(path)
        .map_err(|error| format!("open {label} {} for sync: {error}", path.display()))?;
    directory
        .sync_all()
        .map_err(|error| format!("sync {label} {}: {error}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path, _label: &str) -> Result<(), String> {
    Ok(())
}

pub fn check_update_manifest(
    verified: &VerifiedUpdateManifest,
    state_root: impl AsRef<Path>,
    channel: &str,
    platform: &str,
    current_version: &str,
) -> Result<UpdateCheckReport, String> {
    verified.manifest.validate()?;
    validate_identifier("requested update channel", channel, 64)?;
    validate_identifier("requested update platform", platform, 128)?;
    let current_sequence = StableVersion::parse(current_version)?.sequence()?;
    let selected = verified.manifest.platform(platform)?;
    let state = UpdateStore::read_only(state_root.as_ref())?.read_state_optional()?;
    let mut reason_codes = Vec::new();
    let mut bundle_url = None;
    let mut upper_bound = None;
    let decision = if verified.manifest.channel != channel {
        reason_codes.push("channel-mismatch".into());
        "incompatible"
    } else if state
        .as_ref()
        .is_some_and(|value| value.pending_health.is_some())
    {
        reason_codes.push("pending-health-confirmation".into());
        "pending-health"
    } else if state.as_ref().is_some_and(|value| {
        value.channel != verified.manifest.channel || value.platform != selected.platform
    }) {
        reason_codes.push("state-identity-mismatch".into());
        "incompatible"
    } else if state
        .as_ref()
        .is_some_and(|value| verified.manifest.sequence < value.highest_accepted_sequence)
    {
        reason_codes.push("anti-rollback-floor".into());
        "rollback-blocked"
    } else if state.as_ref().is_some_and(|value| {
        verified.manifest.sequence == value.highest_accepted_sequence
            && verified.verification.manifest_sha256 != value.highest_manifest_sha256
    }) {
        reason_codes.push("manifest-equivocation".into());
        "equivocation"
    } else if verified.manifest.sequence <= current_sequence {
        reason_codes.push(
            if verified.manifest.sequence == current_sequence {
                "already-current"
            } else {
                "candidate-is-older"
            }
            .into(),
        );
        "current"
    } else if selected
        .rollbacks
        .iter()
        .find(|value| value.from_version == current_version)
        .is_none()
    {
        reason_codes.push("current-version-not-supported".into());
        "incompatible"
    } else {
        let rollback = selected
            .rollbacks
            .iter()
            .find(|value| value.from_version == current_version)
            .expect("checked above");
        bundle_url = Some(rollback.bundle_url.clone());
        upper_bound = Some(payload_staging_bytes(
            &selected.candidate,
            &rollback.payload,
        )?);
        reason_codes.push("verified-update-available".into());
        "available"
    };
    Ok(UpdateCheckReport {
        schema: UPDATE_CHECK_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        channel: channel.into(),
        platform: platform.into(),
        current_version: current_version.into(),
        current_sequence,
        candidate_version: verified.manifest.version.clone(),
        candidate_sequence: verified.manifest.sequence,
        manifest_sha256: verified.verification.manifest_sha256.clone(),
        signing_key_id: verified.verification.signing_key_id.clone(),
        decision: decision.into(),
        reason_codes,
        bundle_url,
        download_upper_bound_bytes: upper_bound,
        read_only: true,
    })
}

pub fn dry_run_update_bundle(
    bundle_path: impl AsRef<Path>,
    state_root: impl AsRef<Path>,
    current_version: &str,
    maximum_staging_bytes: Option<u64>,
    public_key_path: Option<&Path>,
) -> Result<UpdateDryRunReport, String> {
    let public_key = read_update_public_key(public_key_path)?;
    let prepared = prepare_update_bundle(bundle_path.as_ref(), &public_key)?;
    let store = UpdateStore::read_only(state_root.as_ref())?;
    let state = store.read_state_optional()?;
    let maximum_staging_bytes = maximum_staging_bytes.unwrap_or(DEFAULT_MAX_STAGING_BYTES);
    let staging_bytes =
        payload_staging_bytes(&prepared.platform.candidate, &prepared.rollback.payload)?;
    let mut reason_codes = Vec::new();
    let decision = validate_apply_transition(
        &prepared,
        state.as_ref(),
        current_version,
        staging_bytes,
        maximum_staging_bytes,
    )
    .map(|()| {
        reason_codes.push("ready-to-stage".into());
        "ready"
    })
    .unwrap_or_else(|error| {
        reason_codes.push(stable_apply_reason(&error).into());
        "rejected"
    });
    Ok(UpdateDryRunReport {
        schema: UPDATE_DRY_RUN_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        channel: prepared.verified.manifest.channel.clone(),
        platform: prepared.platform.platform.clone(),
        current_version: current_version.into(),
        candidate_version: prepared.platform.candidate.version.clone(),
        candidate_sequence: prepared.platform.candidate.sequence,
        manifest_sha256: prepared.verified.verification.manifest_sha256.clone(),
        bundle_sha256: prepared.bundle_sha256,
        decision: decision.into(),
        reason_codes,
        staging_bytes,
        maximum_staging_bytes,
        destination_actions: vec![
            "stage-candidate-slot".into(),
            "stage-last-known-good-slot".into(),
            "atomically-activate-candidate".into(),
            "retain-offline-recovery".into(),
        ],
        preserves_last_known_good: true,
        recovery_requires_network: false,
        read_only: true,
    })
}

pub fn apply_update_bundle(
    bundle_path: impl AsRef<Path>,
    state_root: impl AsRef<Path>,
    current_version: &str,
    maximum_staging_bytes: Option<u64>,
    public_key_path: Option<&Path>,
) -> Result<UpdateApplyReport, String> {
    let public_key = read_update_public_key(public_key_path)?;
    let mut prepared = prepare_update_bundle(bundle_path.as_ref(), &public_key)?;
    let maximum_staging_bytes = maximum_staging_bytes.unwrap_or(DEFAULT_MAX_STAGING_BYTES);
    let staging_bytes =
        payload_staging_bytes(&prepared.platform.candidate, &prepared.rollback.payload)?;
    let store = UpdateStore::open(state_root.as_ref())?;
    let _lock = store.lock_exclusive()?;
    store.cleanup_staging()?;
    let state = store.read_state_optional()?;
    if let Some(existing) = &state {
        store.cleanup_unreferenced_slots(existing)?;
    }
    validate_apply_transition(
        &prepared,
        state.as_ref(),
        current_version,
        staging_bytes,
        maximum_staging_bytes,
    )?;
    let applied_at = now_unix_seconds()?;
    let candidate_payload = prepared.platform.candidate.clone();
    let candidate = store.stage_slot(&mut prepared, true, &candidate_payload)?;
    fault_injection::hit("update.after-candidate-slot-sync")?;
    let rollback_payload = prepared.rollback.payload.clone();
    let last_known_good = store.stage_slot(&mut prepared, false, &rollback_payload)?;
    fault_injection::hit("update.after-last-known-good-slot-sync")?;
    let token = health_token(
        &prepared.bundle_sha256,
        &candidate,
        &last_known_good,
        applied_at,
    );
    let deadline = applied_at
        .checked_add(verified_health_timeout(&prepared)?)
        .ok_or_else(|| "update health deadline overflowed".to_string())?;
    let previous_generation = state.as_ref().map_or(0, |value| value.generation);
    let from_version = state
        .as_ref()
        .map(|value| value.active.version.clone())
        .unwrap_or_else(|| current_version.to_string());
    let mut failed_slots = state
        .as_ref()
        .map(|value| value.failed_slots.clone())
        .unwrap_or_default();
    failed_slots.truncate(MAX_FAILED_SLOTS);
    let mut diagnostics = state
        .as_ref()
        .map(|value| value.diagnostics.clone())
        .unwrap_or_default();
    let generation = previous_generation
        .checked_add(1)
        .ok_or_else(|| "update state generation overflowed".to_string())?;
    push_diagnostic(
        &mut diagnostics,
        UpdateDiagnostic {
            generation,
            unix_seconds: applied_at,
            code: "candidate-activated-pending-health".into(),
            from_version: Some(from_version.clone()),
            to_version: Some(candidate.version.clone()),
            manifest_sha256: Some(candidate.manifest_sha256.clone()),
            artifact_sha256: Some(candidate.artifact.sha256.clone()),
        },
    );
    let new_state = UpdateState {
        schema: UPDATE_STATE_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        generation,
        channel: prepared.verified.manifest.channel.clone(),
        platform: prepared.platform.platform.clone(),
        highest_accepted_sequence: prepared.platform.candidate.sequence,
        highest_manifest_sha256: prepared.verified.verification.manifest_sha256.clone(),
        active: candidate.clone(),
        last_known_good: last_known_good.clone(),
        pending_health: Some(PendingUpdateHealth {
            candidate_slot_id: candidate.slot_id.clone(),
            last_known_good_slot_id: last_known_good.slot_id.clone(),
            health_token: token.clone(),
            applied_unix_seconds: applied_at,
            deadline_unix_seconds: deadline,
            start_attempts: 0,
            maximum_start_attempts: prepared
                .verified
                .manifest
                .rollback_policy
                .maximum_start_attempts,
        }),
        failed_slots,
        diagnostics,
    };
    new_state.validate()?;
    fault_injection::hit("update.before-activation-state-commit")?;
    store.write_state(&new_state)?;
    fault_injection::hit("update.after-activation-state-commit")?;
    store.cleanup_unreferenced_slots(&new_state)?;
    Ok(UpdateApplyReport {
        schema: UPDATE_APPLY_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        channel: new_state.channel,
        platform: new_state.platform,
        from_version,
        candidate_version: candidate.version,
        candidate_sequence: candidate.sequence,
        bundle_sha256: prepared.bundle_sha256,
        manifest_sha256: candidate.manifest_sha256,
        active_slot_id: candidate.slot_id,
        last_known_good_slot_id: last_known_good.slot_id,
        health_token: token,
        health_deadline_unix_seconds: deadline,
        activation: candidate.activation,
        outcome: "pending-health-confirmation".into(),
        relaunch_required: true,
    })
}

pub fn update_status(state_root: impl AsRef<Path>) -> Result<UpdateStatusReport, String> {
    let store = UpdateStore::read_only(state_root.as_ref())?;
    let Some(state) = store.read_state_optional()? else {
        return Ok(UpdateStatusReport {
            schema: UPDATE_STATUS_SCHEMA.into(),
            schema_version: UPDATE_SCHEMA_VERSION,
            managed: false,
            generation: 0,
            channel: None,
            platform: None,
            phase: "unmanaged".into(),
            highest_accepted_sequence: None,
            active: None,
            last_known_good: None,
            health_deadline_unix_seconds: None,
            start_attempts: None,
            maximum_start_attempts: None,
            failed_slot_count: 0,
            diagnostics: Vec::new(),
        });
    };
    state.validate()?;
    store.verify_slot(&state.active)?;
    store.verify_slot(&state.last_known_good)?;
    let pending = state.pending_health.as_ref();
    Ok(UpdateStatusReport {
        schema: UPDATE_STATUS_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        managed: true,
        generation: state.generation,
        channel: Some(state.channel),
        platform: Some(state.platform),
        phase: if pending.is_some() {
            "pending-health"
        } else {
            "healthy"
        }
        .into(),
        highest_accepted_sequence: Some(state.highest_accepted_sequence),
        active: Some(slot_status(&state.active)),
        last_known_good: Some(slot_status(&state.last_known_good)),
        health_deadline_unix_seconds: pending.map(|value| value.deadline_unix_seconds),
        start_attempts: pending.map(|value| value.start_attempts),
        maximum_start_attempts: pending.map(|value| value.maximum_start_attempts),
        failed_slot_count: state.failed_slots.len(),
        diagnostics: state.diagnostics,
    })
}

pub fn active_update_target(
    state_root: impl AsRef<Path>,
) -> Result<UpdateActivationTarget, String> {
    update_activation_target(state_root.as_ref(), false)
}

pub fn last_known_good_update_target(
    state_root: impl AsRef<Path>,
) -> Result<UpdateActivationTarget, String> {
    update_activation_target(state_root.as_ref(), true)
}

fn update_activation_target(
    state_root: &Path,
    last_known_good: bool,
) -> Result<UpdateActivationTarget, String> {
    let store = UpdateStore::open_existing(state_root)?;
    let _lock = store.lock_exclusive()?;
    let state = store
        .read_state_optional()?
        .ok_or_else(|| "application update state is not managed".to_string())?;
    let slot = if last_known_good {
        &state.last_known_good
    } else {
        &state.active
    };
    store.verify_slot(slot)?;
    let artifact_path = store
        .root
        .join(UPDATE_SLOT_DIRECTORY)
        .join(&slot.slot_id)
        .join(&slot.artifact_name);
    Ok(UpdateActivationTarget {
        version: slot.version.clone(),
        platform: slot.platform.clone(),
        activation: slot.activation,
        artifact_path,
        artifact: slot.artifact.clone(),
    })
}

pub fn begin_update_startup_health(
    state_root: impl AsRef<Path>,
    running_version: &str,
    now: Option<u64>,
) -> Result<UpdateHealthReport, String> {
    StableVersion::parse(running_version)?;
    if UpdateStore::read_only(state_root.as_ref())?
        .read_state_optional()?
        .is_none()
    {
        return Ok(unmanaged_health(running_version));
    }
    let store = UpdateStore::open_existing(state_root.as_ref())?;
    let _lock = store.lock_exclusive()?;
    let Some(mut state) = store.read_state_optional()? else {
        return Ok(unmanaged_health(running_version));
    };
    state.validate()?;
    store.verify_slot(&state.last_known_good)?;
    let active_integrity = store.verify_slot(&state.active);
    if let Err(error) = active_integrity {
        let Some(pending) = state.pending_health.clone() else {
            return Err(format!(
                "healthy active update slot failed integrity validation: {error}"
            ));
        };
        let now = now.unwrap_or(now_unix_seconds()?);
        recover_state(&store, &mut state, "candidate-integrity-failed", now)?;
        return Ok(UpdateHealthReport {
            schema: UPDATE_HEALTH_SCHEMA.into(),
            schema_version: UPDATE_SCHEMA_VERSION,
            action: "recovered-last-known-good".into(),
            running_version: running_version.into(),
            active_version: Some(state.active.version.clone()),
            last_known_good_version: Some(state.last_known_good.version.clone()),
            health_token: None,
            start_attempts: Some(pending.start_attempts),
            maximum_start_attempts: Some(pending.maximum_start_attempts),
            deadline_unix_seconds: Some(pending.deadline_unix_seconds),
            relaunch_required: running_version != state.active.version,
        });
    }
    let Some(pending) = state.pending_health.clone() else {
        if state.active.version != running_version {
            return Ok(UpdateHealthReport {
                schema: UPDATE_HEALTH_SCHEMA.into(),
                schema_version: UPDATE_SCHEMA_VERSION,
                action: "reactivate-managed-version".into(),
                running_version: running_version.into(),
                active_version: Some(state.active.version),
                last_known_good_version: Some(state.last_known_good.version),
                health_token: None,
                start_attempts: None,
                maximum_start_attempts: None,
                deadline_unix_seconds: None,
                relaunch_required: true,
            });
        }
        return Ok(UpdateHealthReport {
            schema: UPDATE_HEALTH_SCHEMA.into(),
            schema_version: UPDATE_SCHEMA_VERSION,
            action: "healthy".into(),
            running_version: running_version.into(),
            active_version: Some(state.active.version),
            last_known_good_version: Some(state.last_known_good.version),
            health_token: None,
            start_attempts: None,
            maximum_start_attempts: None,
            deadline_unix_seconds: None,
            relaunch_required: false,
        });
    };
    let now = now.unwrap_or(now_unix_seconds()?);
    if running_version != state.active.version
        || now > pending.deadline_unix_seconds
        || pending.start_attempts >= pending.maximum_start_attempts
    {
        let reason = if running_version != state.active.version {
            "running-version-mismatch"
        } else if now > pending.deadline_unix_seconds {
            "health-deadline-expired"
        } else {
            "health-start-attempts-exhausted"
        };
        recover_state(&store, &mut state, reason, now)?;
        return Ok(UpdateHealthReport {
            schema: UPDATE_HEALTH_SCHEMA.into(),
            schema_version: UPDATE_SCHEMA_VERSION,
            action: "recovered-last-known-good".into(),
            running_version: running_version.into(),
            active_version: Some(state.active.version.clone()),
            last_known_good_version: Some(state.last_known_good.version.clone()),
            health_token: None,
            start_attempts: Some(pending.start_attempts),
            maximum_start_attempts: Some(pending.maximum_start_attempts),
            deadline_unix_seconds: Some(pending.deadline_unix_seconds),
            relaunch_required: running_version != state.active.version,
        });
    }
    let next_attempt = pending
        .start_attempts
        .checked_add(1)
        .ok_or_else(|| "update health start-attempt counter overflowed".to_string())?;
    let generation = state
        .generation
        .checked_add(1)
        .ok_or_else(|| "update state generation overflowed".to_string())?;
    let pending_mut = state
        .pending_health
        .as_mut()
        .expect("pending health checked above");
    pending_mut.start_attempts = next_attempt;
    state.generation = generation;
    push_diagnostic(
        &mut state.diagnostics,
        UpdateDiagnostic {
            generation,
            unix_seconds: now,
            code: "startup-health-awaiting-confirmation".into(),
            from_version: Some(state.last_known_good.version.clone()),
            to_version: Some(state.active.version.clone()),
            manifest_sha256: Some(state.active.manifest_sha256.clone()),
            artifact_sha256: Some(state.active.artifact.sha256.clone()),
        },
    );
    store.write_state(&state)?;
    Ok(UpdateHealthReport {
        schema: UPDATE_HEALTH_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        action: "confirm-required".into(),
        running_version: running_version.into(),
        active_version: Some(state.active.version.clone()),
        last_known_good_version: Some(state.last_known_good.version.clone()),
        health_token: Some(pending.health_token.clone()),
        start_attempts: Some(next_attempt),
        maximum_start_attempts: Some(pending.maximum_start_attempts),
        deadline_unix_seconds: Some(pending.deadline_unix_seconds),
        relaunch_required: false,
    })
}

pub fn confirm_update_health(
    state_root: impl AsRef<Path>,
    running_version: &str,
    health_token: &str,
    now: Option<u64>,
) -> Result<UpdateHealthReport, String> {
    StableVersion::parse(running_version)?;
    validate_health_token(health_token)?;
    let store = UpdateStore::open_existing(state_root.as_ref())?;
    let _lock = store.lock_exclusive()?;
    let mut state = store
        .read_state_optional()?
        .ok_or_else(|| "update state is not initialized".to_string())?;
    state.validate()?;
    let pending = state
        .pending_health
        .clone()
        .ok_or_else(|| "no update is awaiting health confirmation".to_string())?;
    if state.active.version != running_version
        || pending.candidate_slot_id != state.active.slot_id
        || health_token != pending.health_token
    {
        return Err("update health confirmation does not match the pending candidate".into());
    }
    let now = now.unwrap_or(now_unix_seconds()?);
    if now > pending.deadline_unix_seconds {
        return Err("update health confirmation arrived after its deadline".into());
    }
    store.verify_slot(&state.active)?;
    let previous_lkg = state.last_known_good.clone();
    state.last_known_good = state.active.clone();
    state.pending_health = None;
    state.generation = state
        .generation
        .checked_add(1)
        .ok_or_else(|| "update state generation overflowed".to_string())?;
    push_diagnostic(
        &mut state.diagnostics,
        UpdateDiagnostic {
            generation: state.generation,
            unix_seconds: now,
            code: "candidate-health-confirmed".into(),
            from_version: Some(previous_lkg.version.clone()),
            to_version: Some(state.active.version.clone()),
            manifest_sha256: Some(state.active.manifest_sha256.clone()),
            artifact_sha256: Some(state.active.artifact.sha256.clone()),
        },
    );
    fault_injection::hit("update.before-health-confirm-state-commit")?;
    store.write_state(&state)?;
    fault_injection::hit("update.after-health-confirm-state-commit")?;
    store.cleanup_unreferenced_slots(&state)?;
    Ok(UpdateHealthReport {
        schema: UPDATE_HEALTH_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        action: "confirmed".into(),
        running_version: running_version.into(),
        active_version: Some(state.active.version.clone()),
        last_known_good_version: Some(state.last_known_good.version.clone()),
        health_token: None,
        start_attempts: Some(pending.start_attempts),
        maximum_start_attempts: Some(pending.maximum_start_attempts),
        deadline_unix_seconds: Some(pending.deadline_unix_seconds),
        relaunch_required: false,
    })
}

pub fn recover_update(
    state_root: impl AsRef<Path>,
    reason_code: &str,
    now: Option<u64>,
) -> Result<UpdateHealthReport, String> {
    validate_reason_code(reason_code)?;
    let store = UpdateStore::open_existing(state_root.as_ref())?;
    let _lock = store.lock_exclusive()?;
    let mut state = store
        .read_state_optional()?
        .ok_or_else(|| "update state is not initialized".to_string())?;
    state.validate()?;
    if state.pending_health.is_none() {
        return Err("manual recovery is available only while candidate health is pending".into());
    }
    let previous = state.active.version.clone();
    let now = now.unwrap_or(now_unix_seconds()?);
    recover_state(&store, &mut state, reason_code, now)?;
    Ok(UpdateHealthReport {
        schema: UPDATE_HEALTH_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        action: "recovered-last-known-good".into(),
        running_version: previous.clone(),
        active_version: Some(state.active.version.clone()),
        last_known_good_version: Some(state.last_known_good.version.clone()),
        health_token: None,
        start_attempts: None,
        maximum_start_attempts: None,
        deadline_unix_seconds: None,
        relaunch_required: previous != state.active.version,
    })
}

fn validate_apply_transition(
    prepared: &PreparedUpdateBundle,
    state: Option<&UpdateState>,
    current_version: &str,
    staging_bytes: u64,
    maximum_staging_bytes: u64,
) -> Result<(), String> {
    let current_sequence = StableVersion::parse(current_version)?.sequence()?;
    if current_version != prepared.rollback.from_version
        || current_sequence != prepared.rollback.from_sequence
    {
        return Err("current-version-mismatch: bundle rollback identity differs".into());
    }
    if staging_bytes > maximum_staging_bytes || maximum_staging_bytes > MAX_BUNDLE_BYTES {
        return Err("staging-limit-exceeded: update exceeds configured staging limit".into());
    }
    if prepared.platform.candidate.sequence <= current_sequence {
        return Err("candidate-not-newer: update candidate must be newer than current".into());
    }
    if let Some(state) = state {
        state.validate()?;
        if state.pending_health.is_some() {
            return Err("pending-health: confirm or recover the current candidate first".into());
        }
        if state.channel != prepared.verified.manifest.channel
            || state.platform != prepared.platform.platform
        {
            return Err("state-identity-mismatch: channel or platform changed".into());
        }
        if state.active.version != current_version {
            return Err(
                "current-version-mismatch: running version differs from active state".into(),
            );
        }
        if state.active.artifact != prepared.rollback.payload.artifact.fingerprint {
            return Err(
                "last-known-good-mismatch: rollback artifact differs from active state".into(),
            );
        }
        if prepared.platform.candidate.sequence < state.highest_accepted_sequence {
            return Err("anti-rollback-floor: candidate is below accepted sequence".into());
        }
        if prepared.platform.candidate.sequence == state.highest_accepted_sequence
            && prepared.verified.verification.manifest_sha256 != state.highest_manifest_sha256
        {
            return Err("manifest-equivocation: accepted sequence has different bytes".into());
        }
    }
    Ok(())
}

fn stable_apply_reason(error: &str) -> &'static str {
    for (prefix, code) in [
        ("current-version-mismatch", "current-version-mismatch"),
        ("staging-limit-exceeded", "staging-limit-exceeded"),
        ("candidate-not-newer", "candidate-not-newer"),
        ("pending-health", "pending-health"),
        ("state-identity-mismatch", "state-identity-mismatch"),
        ("last-known-good-mismatch", "last-known-good-mismatch"),
        ("anti-rollback-floor", "anti-rollback-floor"),
        ("manifest-equivocation", "manifest-equivocation"),
    ] {
        if error.starts_with(prefix) {
            return code;
        }
    }
    "invalid-update-transaction"
}

fn payload_staging_bytes(
    candidate: &UpdatePayload,
    rollback: &UpdatePayload,
) -> Result<u64, String> {
    let mut total = 0_u64;
    for file in [
        &candidate.artifact,
        &candidate.sbom,
        &candidate.provenance,
        &rollback.artifact,
        &rollback.sbom,
        &rollback.provenance,
    ] {
        total = total
            .checked_add(file.fingerprint.len)
            .ok_or_else(|| "update staging byte total overflowed".to_string())?;
    }
    total
        .checked_add(MAX_MANIFEST_BYTES.saturating_mul(2))
        .and_then(|value| value.checked_add(MAX_SIGNATURE_BYTES.saturating_mul(2)))
        .ok_or_else(|| "update staging byte total overflowed".to_string())
}

fn verified_health_timeout(prepared: &PreparedUpdateBundle) -> Result<u64, String> {
    prepared.verified.manifest.rollback_policy.validate()?;
    Ok(prepared
        .verified
        .manifest
        .rollback_policy
        .health_timeout_seconds)
}

fn health_token(
    bundle_sha256: &str,
    candidate: &StoredUpdateSlot,
    rollback: &StoredUpdateSlot,
    applied_at: u64,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"denoize-update-health-token-v1");
    digest.update(bundle_sha256.as_bytes());
    digest.update(candidate.slot_id.as_bytes());
    digest.update(rollback.slot_id.as_bytes());
    digest.update(applied_at.to_le_bytes());
    digest.update(std::process::id().to_le_bytes());
    format!("{:x}", digest.finalize())
}

fn validate_health_token(token: &str) -> Result<(), String> {
    validate_sha256("update health token", token)
}

fn validate_reason_code(value: &str) -> Result<(), String> {
    validate_identifier("update recovery reason code", value, 128)
}

fn push_diagnostic(diagnostics: &mut Vec<UpdateDiagnostic>, event: UpdateDiagnostic) {
    diagnostics.push(event);
    if diagnostics.len() > MAX_DIAGNOSTICS {
        let remove = diagnostics.len() - MAX_DIAGNOSTICS;
        diagnostics.drain(..remove);
    }
}

fn slot_status(slot: &StoredUpdateSlot) -> UpdateSlotStatus {
    UpdateSlotStatus {
        slot_id: slot.slot_id.clone(),
        version: slot.version.clone(),
        sequence: slot.sequence,
        platform: slot.platform.clone(),
        activation: slot.activation,
        manifest_sha256: slot.manifest_sha256.clone(),
        artifact_name: slot.artifact_name.clone(),
        artifact: slot.artifact.clone(),
        sbom: slot.sbom.clone(),
        provenance: slot.provenance.clone(),
    }
}

fn unmanaged_health(running_version: &str) -> UpdateHealthReport {
    UpdateHealthReport {
        schema: UPDATE_HEALTH_SCHEMA.into(),
        schema_version: UPDATE_SCHEMA_VERSION,
        action: "unmanaged".into(),
        running_version: running_version.into(),
        active_version: None,
        last_known_good_version: None,
        health_token: None,
        start_attempts: None,
        maximum_start_attempts: None,
        deadline_unix_seconds: None,
        relaunch_required: false,
    }
}

fn recover_state(
    store: &UpdateStore,
    state: &mut UpdateState,
    reason_code: &str,
    now: u64,
) -> Result<(), String> {
    validate_reason_code(reason_code)?;
    let pending = state
        .pending_health
        .clone()
        .ok_or_else(|| "no update is awaiting recovery".to_string())?;
    if pending.candidate_slot_id != state.active.slot_id
        || pending.last_known_good_slot_id != state.last_known_good.slot_id
    {
        return Err("pending update state does not bind active and last-known-good slots".into());
    }
    store.verify_slot(&state.last_known_good)?;
    let failed = state.active.clone();
    state.active = state.last_known_good.clone();
    state.pending_health = None;
    state.generation = state
        .generation
        .checked_add(1)
        .ok_or_else(|| "update state generation overflowed".to_string())?;
    state.failed_slots.push(FailedUpdateSlot {
        slot_id: failed.slot_id.clone(),
        version: failed.version.clone(),
        sequence: failed.sequence,
        artifact_sha256: failed.artifact.sha256.clone(),
        failed_unix_seconds: now,
        reason_code: reason_code.into(),
    });
    if state.failed_slots.len() > MAX_FAILED_SLOTS {
        let remove = state.failed_slots.len() - MAX_FAILED_SLOTS;
        state.failed_slots.drain(..remove);
    }
    push_diagnostic(
        &mut state.diagnostics,
        UpdateDiagnostic {
            generation: state.generation,
            unix_seconds: now,
            code: "last-known-good-recovered".into(),
            from_version: Some(failed.version),
            to_version: Some(state.active.version.clone()),
            manifest_sha256: Some(failed.manifest_sha256),
            artifact_sha256: Some(failed.artifact.sha256),
        },
    );
    // The redacted failed-slot record is retained; only the two state-bound
    // installation slots remain eligible for storage retention.
    fault_injection::hit("update.before-recovery-state-commit")?;
    store.write_state(state)?;
    fault_injection::hit("update.after-recovery-state-commit")?;
    store.cleanup_unreferenced_slots(state)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use minisign::{sign, KeyPair};
    use std::io::Cursor;

    const TEST_PLATFORM: &str = "linux-x86_64";

    struct SignedBundleFixture {
        bundle: PathBuf,
        manifest: PathBuf,
        signature: PathBuf,
        public_key: PathBuf,
        candidate_artifact_bytes: Vec<u8>,
    }

    fn remote_file(name: String, bytes: &[u8]) -> UpdateRemoteFile {
        UpdateRemoteFile {
            url: format!("https://updates.example.invalid/{name}"),
            name,
            fingerprint: UpdateFingerprint {
                len: bytes.len() as u64,
                sha256: sha256_bytes(bytes),
            },
        }
    }

    fn payload(version: &str, prefix: &str, artifact: &[u8]) -> (UpdatePayload, Vec<u8>, Vec<u8>) {
        let sbom = format!("SBOM for {version}\n").into_bytes();
        let provenance = format!("provenance for {version}\n").into_bytes();
        let artifact_name = format!("denoize-{version}-{prefix}.tar.gz");
        (
            UpdatePayload {
                version: version.into(),
                sequence: StableVersion::parse(version).unwrap().sequence().unwrap(),
                activation: UpdateActivationKind::PortableExecutable,
                artifact: remote_file(artifact_name.clone(), artifact),
                sbom: remote_file(format!("{artifact_name}.cdx.json"), &sbom),
                provenance: remote_file(format!("{artifact_name}.intoto.jsonl"), &provenance),
            },
            sbom,
            provenance,
        )
    }

    fn write_component(directory: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = directory.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn tauri_signature(key_pair: &KeyPair, manifest_bytes: &[u8]) -> Vec<u8> {
        let minisign_signature = sign(
            Some(&key_pair.pk),
            &key_pair.sk,
            Cursor::new(manifest_bytes),
            Some("timestamp:1800000000\tfile:update-manifest.json"),
            Some("denoize update manifest test signature"),
        )
        .unwrap()
        .to_bytes();
        // Tauri's signer publishes the minisign document as outer Base64.
        BASE64_STANDARD.encode(minisign_signature).into_bytes()
    }

    fn signed_bundle(
        directory: &Path,
        key_pair: &KeyPair,
        current_version: &str,
        candidate_version: &str,
    ) -> SignedBundleFixture {
        let candidate_artifact_bytes =
            format!("candidate payload {candidate_version}\n").into_bytes();
        let rollback_artifact_bytes = format!("installed payload {current_version}\n").into_bytes();
        let (candidate, candidate_sbom, candidate_provenance) =
            payload(candidate_version, TEST_PLATFORM, &candidate_artifact_bytes);
        let (rollback, rollback_sbom, rollback_provenance) =
            payload(current_version, TEST_PLATFORM, &rollback_artifact_bytes);
        let manifest = UpdateManifest {
            schema: UPDATE_MANIFEST_SCHEMA.into(),
            schema_version: UPDATE_SCHEMA_VERSION,
            channel: "stable".into(),
            version: candidate_version.into(),
            sequence: StableVersion::parse(candidate_version)
                .unwrap()
                .sequence()
                .unwrap(),
            published_unix_seconds: 1_800_000_000,
            source_commit: "ab".repeat(20),
            compatibility: UpdateCompatibility {
                accepted_from_versions: vec![current_version.into()],
                minimum_state_schema_version: 1,
                maximum_state_schema_version: 1,
            },
            rollback_policy: UpdateRollbackPolicy {
                retained_last_known_good: 1,
                health_timeout_seconds: 300,
                maximum_start_attempts: 3,
                manual_recovery: true,
                network_required_for_recovery: false,
            },
            platforms: vec![UpdatePlatform {
                platform: TEST_PLATFORM.into(),
                candidate: candidate.clone(),
                rollbacks: vec![UpdateRollbackPayload {
                    from_version: current_version.into(),
                    from_sequence: rollback.sequence,
                    bundle_url: format!(
                        "https://updates.example.invalid/denoize-{candidate_version}-from-{current_version}.dub"
                    ),
                    payload: rollback.clone(),
                }],
            }],
        };
        let manifest_bytes = format!("{}\n", manifest.to_pretty_json().unwrap()).into_bytes();
        // Exercise the exact Tauri release-signature representation in every
        // bundle and transaction test.
        let signature_bytes = tauri_signature(key_pair, &manifest_bytes);
        let suffix = format!("{current_version}-to-{candidate_version}");
        let manifest_path = write_component(
            directory,
            &format!("manifest-{suffix}.json"),
            &manifest_bytes,
        );
        let signature_path = write_component(
            directory,
            &format!("manifest-{suffix}.json.sig"),
            &signature_bytes,
        );
        let public_key_path = write_component(
            directory,
            "minisign.pub",
            &key_pair.pk.to_box().unwrap().to_bytes(),
        );
        let candidate_artifact = write_component(
            directory,
            &candidate.artifact.name,
            &candidate_artifact_bytes,
        );
        let candidate_sbom_path = write_component(directory, &candidate.sbom.name, &candidate_sbom);
        let candidate_provenance_path =
            write_component(directory, &candidate.provenance.name, &candidate_provenance);
        let rollback_artifact =
            write_component(directory, &rollback.artifact.name, &rollback_artifact_bytes);
        let rollback_sbom_path = write_component(directory, &rollback.sbom.name, &rollback_sbom);
        let rollback_provenance_path =
            write_component(directory, &rollback.provenance.name, &rollback_provenance);
        let bundle = directory.join(format!("update-{suffix}.dub"));
        build_update_bundle(
            &bundle,
            &UpdateBundleBuildRequest {
                platform: TEST_PLATFORM.into(),
                from_version: current_version.into(),
                manifest_path: manifest_path.clone(),
                signature_path: signature_path.clone(),
                candidate_artifact_path: candidate_artifact,
                candidate_sbom_path,
                candidate_provenance_path,
                rollback_artifact_path: rollback_artifact,
                rollback_sbom_path,
                rollback_provenance_path,
                public_key_path: Some(public_key_path.clone()),
            },
        )
        .unwrap();
        SignedBundleFixture {
            bundle,
            manifest: manifest_path,
            signature: signature_path,
            public_key: public_key_path,
            candidate_artifact_bytes,
        }
    }

    #[test]
    fn check_and_dry_run_are_read_only_then_apply_confirms_health() {
        let directory = tempfile::tempdir().unwrap();
        let key_pair = KeyPair::generate_unencrypted_keypair().unwrap();
        let fixture = signed_bundle(directory.path(), &key_pair, "1.1.0", "1.2.0");
        let state_root = directory.path().join("update-state");
        let verified = UpdateManifest::from_file(
            &fixture.manifest,
            &fixture.signature,
            Some(&fixture.public_key),
        )
        .unwrap();

        let check = check_update_manifest(&verified, &state_root, "stable", TEST_PLATFORM, "1.1.0")
            .unwrap();
        assert_eq!(check.decision, "available");
        assert!(check.read_only);
        assert!(!state_root.exists());

        let current =
            check_update_manifest(&verified, &state_root, "stable", TEST_PLATFORM, "1.2.0")
                .unwrap();
        assert_eq!(current.decision, "current");
        assert_eq!(current.reason_codes, ["already-current"]);
        let older = check_update_manifest(&verified, &state_root, "stable", TEST_PLATFORM, "1.3.0")
            .unwrap();
        assert_eq!(older.decision, "current");
        assert_eq!(older.reason_codes, ["candidate-is-older"]);
        assert!(!state_root.exists());

        let dry_run = dry_run_update_bundle(
            &fixture.bundle,
            &state_root,
            "1.1.0",
            None,
            Some(&fixture.public_key),
        )
        .unwrap();
        assert_eq!(dry_run.decision, "ready");
        assert!(dry_run.preserves_last_known_good);
        assert!(!dry_run.recovery_requires_network);
        assert!(!state_root.exists());

        let applied = apply_update_bundle(
            &fixture.bundle,
            &state_root,
            "1.1.0",
            None,
            Some(&fixture.public_key),
        )
        .unwrap();
        assert_eq!(applied.outcome, "pending-health-confirmation");
        assert!(applied.relaunch_required);
        let pending = update_status(&state_root).unwrap();
        assert_eq!(pending.phase, "pending-health");
        assert_eq!(pending.active.as_ref().unwrap().version, "1.2.0");
        assert_eq!(pending.last_known_good.as_ref().unwrap().version, "1.1.0");

        let candidate_target = active_update_target(&state_root).unwrap();
        assert_eq!(candidate_target.version, "1.2.0");
        assert_eq!(candidate_target.platform, TEST_PLATFORM);
        assert_eq!(
            std::fs::read(&candidate_target.artifact_path).unwrap(),
            fixture.candidate_artifact_bytes
        );
        assert_eq!(
            candidate_target.artifact.sha256,
            sha256_bytes(&fixture.candidate_artifact_bytes)
        );
        let rollback_target = last_known_good_update_target(&state_root).unwrap();
        assert_eq!(rollback_target.version, "1.1.0");
        assert_ne!(
            candidate_target.artifact_path,
            rollback_target.artifact_path
        );

        let begun = begin_update_startup_health(&state_root, "1.2.0", None).unwrap();
        assert_eq!(begun.action, "confirm-required");
        assert_eq!(
            begun.health_token.as_deref(),
            Some(applied.health_token.as_str())
        );
        let confirmed = confirm_update_health(
            &state_root,
            "1.2.0",
            begun.health_token.as_deref().unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(confirmed.action, "confirmed");
        let healthy = update_status(&state_root).unwrap();
        assert_eq!(healthy.phase, "healthy");
        assert_eq!(healthy.active, healthy.last_known_good);
        assert_eq!(
            last_known_good_update_target(&state_root).unwrap().version,
            "1.2.0"
        );
    }

    #[test]
    fn recovery_keeps_the_monotonic_floor_and_blocks_an_older_candidate() {
        let directory = tempfile::tempdir().unwrap();
        let key_pair = KeyPair::generate_unencrypted_keypair().unwrap();
        let high = signed_bundle(directory.path(), &key_pair, "1.0.0", "1.2.0");
        let state_root = directory.path().join("update-state");
        apply_update_bundle(
            &high.bundle,
            &state_root,
            "1.0.0",
            None,
            Some(&high.public_key),
        )
        .unwrap();
        let recovered = recover_update(&state_root, "manual-test-recovery", None).unwrap();
        assert_eq!(recovered.action, "recovered-last-known-good");
        let status = update_status(&state_root).unwrap();
        assert_eq!(status.active.as_ref().unwrap().version, "1.0.0");
        assert_eq!(status.highest_accepted_sequence, Some(1_000_002_000_000));
        assert_eq!(status.failed_slot_count, 1);
        let repair = begin_update_startup_health(&state_root, "1.2.0", None).unwrap();
        assert_eq!(repair.action, "reactivate-managed-version");
        assert_eq!(repair.active_version.as_deref(), Some("1.0.0"));
        assert!(repair.relaunch_required);

        let verified =
            UpdateManifest::from_file(&high.manifest, &high.signature, Some(&high.public_key))
                .unwrap();
        let mut equivocated_manifest = verified.manifest;
        equivocated_manifest.source_commit = "cd".repeat(20);
        let equivocated_bytes =
            format!("{}\n", equivocated_manifest.to_pretty_json().unwrap()).into_bytes();
        let public_key = String::from_utf8(key_pair.pk.to_box().unwrap().to_bytes()).unwrap();
        let equivocated = verify_update_manifest_bytes(
            equivocated_bytes.clone(),
            tauri_signature(&key_pair, &equivocated_bytes),
            &public_key,
        )
        .unwrap();
        let check =
            check_update_manifest(&equivocated, &state_root, "stable", TEST_PLATFORM, "1.0.0")
                .unwrap();
        assert_eq!(check.decision, "equivocation");
        assert_eq!(check.reason_codes, ["manifest-equivocation"]);

        let lower = signed_bundle(directory.path(), &key_pair, "1.0.0", "1.1.0");
        let dry_run = dry_run_update_bundle(
            &lower.bundle,
            &state_root,
            "1.0.0",
            None,
            Some(&lower.public_key),
        )
        .unwrap();
        assert_eq!(dry_run.decision, "rejected");
        assert_eq!(dry_run.reason_codes, ["anti-rollback-floor"]);
        let error = apply_update_bundle(
            &lower.bundle,
            &state_root,
            "1.0.0",
            None,
            Some(&lower.public_key),
        )
        .unwrap_err();
        assert!(error.contains("anti-rollback-floor"), "{error}");
    }

    #[test]
    fn staged_slot_tampering_blocks_status_but_offline_lkg_recovery_still_works() {
        let directory = tempfile::tempdir().unwrap();
        let key_pair = KeyPair::generate_unencrypted_keypair().unwrap();
        let fixture = signed_bundle(directory.path(), &key_pair, "2.0.0", "2.1.0");
        let state_root = directory.path().join("update-state");
        apply_update_bundle(
            &fixture.bundle,
            &state_root,
            "2.0.0",
            None,
            Some(&fixture.public_key),
        )
        .unwrap();
        let status = update_status(&state_root).unwrap();
        let active = status.active.unwrap();
        let artifact = state_root
            .join(UPDATE_SLOT_DIRECTORY)
            .join(active.slot_id)
            .join(active.artifact_name);
        let mut corrupted = fixture.candidate_artifact_bytes;
        corrupted[0] ^= 0xff;
        std::fs::write(&artifact, corrupted).unwrap();
        let error = update_status(&state_root).unwrap_err();
        assert!(error.contains("fingerprint differs"), "{error}");
        let error = active_update_target(&state_root).unwrap_err();
        assert!(error.contains("fingerprint differs"), "{error}");
        let recovered = begin_update_startup_health(&state_root, "2.1.0", None).unwrap();
        assert_eq!(recovered.action, "recovered-last-known-good");
        let status = update_status(&state_root).unwrap();
        assert_eq!(status.active.as_ref().unwrap().version, "2.0.0");
    }
}
