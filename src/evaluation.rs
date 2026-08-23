//! Reproducible licensed-corpus evaluation and signed release evidence.
//!
//! The manifest is deliberately a data contract rather than an arbitrary
//! command runner.  Every audio/model input is pinned by license, immutable
//! source revision, preparation recipe, length, and SHA-256.  A run consumes
//! only paths contained below an explicit corpus root and signs the canonical
//! result with the same independently trusted Ed25519 keys used by execution
//! receipts.

use crate::batch_resume::{self, Digest, FileFingerprint};
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
use crate::{
    hardware_capabilities, read_audio, read_wav_bytes, write_wav_bytes, AcceleratorPreference,
    Audio, Backend, BackendOptions, BackendSession, ChannelMode, ComparisonReport, OnnxModelConfig,
    Preset, SgmseProfile,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const EVALUATION_CORPUS_SCHEMA: &str = "denoize-evaluation-corpus-v1";
pub const EVALUATION_CORPUS_VERIFICATION_SCHEMA: &str = "denoize-evaluation-corpus-verification-v1";
pub const EVALUATION_RESULT_SCHEMA: &str = "denoize-evaluation-result-v1";
pub const EVALUATION_VERIFICATION_SCHEMA: &str = "denoize-evaluation-verification-v1";
pub const EVALUATION_COMPARISON_SCHEMA: &str = "denoize-evaluation-comparison-v1";
pub const LISTENING_RESULT_SCHEMA: &str = "denoize-listening-result-v1";
pub const EVALUATION_SCHEMA_VERSION: u32 = 1;

const EVALUATION_SIGNATURE_DOMAIN: &[u8] = b"denoize-evaluation-result-signature-v1";
const EVALUATION_MANIFEST_DIGEST_DOMAIN: &[u8] = b"denoize-evaluation-corpus-digest-v1";
const LISTENING_PROTOCOL_DIGEST_DOMAIN: &[u8] = b"denoize-listening-protocol-digest-v1";
const MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const MAX_CASES: usize = 10_000;
const MAX_TEXT_BYTES: usize = 4_096;
const MAX_LOCATOR_BYTES: usize = 4_096;
const SILENCE_FLOOR: f64 = 1e-12;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationLicense {
    pub spdx_id: String,
    pub name: String,
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationSource {
    pub uri: String,
    pub revision: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignalPreparation {
    pub description: String,
    pub tool: String,
    pub tool_version: String,
    pub parameters_digest: Digest,
}

/// One externally stored corpus or model artifact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationArtifact {
    pub path: String,
    pub fingerprint: FileFingerprint,
    pub license: EvaluationLicense,
    pub source: EvaluationSource,
    pub preparation: SignalPreparation,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCase {
    pub id: String,
    pub clean: EvaluationArtifact,
    pub noisy: EvaluationArtifact,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRecipe {
    pub backend: String,
    pub preset: String,
    pub accelerator: String,
    pub deterministic: bool,
    pub seed: Option<u64>,
    pub channel_mode: String,
    pub sgmse_profile: String,
    pub model: Option<EvaluationArtifact>,
    pub model_sample_rate: Option<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ThresholdAggregation {
    Minimum,
    Maximum,
    Mean,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ThresholdOperator {
    GreaterOrEqual,
    LessOrEqual,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegressionDirection {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationThreshold {
    pub metric: String,
    pub aggregation: ThresholdAggregation,
    pub operator: ThresholdOperator,
    pub value: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegressionTolerance {
    pub metric: String,
    pub aggregation: ThresholdAggregation,
    pub direction: RegressionDirection,
    pub max_regression: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ListeningProtocol {
    pub protocol_id: String,
    pub revision: String,
    pub method: String,
    pub instructions_uri: String,
    pub instructions_digest: Digest,
    pub scale_min: f64,
    pub scale_max: f64,
    pub minimum_listeners: u32,
    pub acceptance_score: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ListeningPolicy {
    pub required: bool,
    pub rationale: String,
    pub protocol: Option<ListeningProtocol>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPolicy {
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub silence_threshold_dbfs: f64,
    pub dropout_window_ms: u32,
    pub thresholds: Vec<EvaluationThreshold>,
    pub regression_tolerances: Vec<RegressionTolerance>,
    pub listening: ListeningPolicy,
}

/// Canonical input contract shared by local and CI evaluation runners.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationManifest {
    pub schema: String,
    pub schema_version: u32,
    pub corpus_id: String,
    pub corpus_version: String,
    pub title: String,
    pub cases: Vec<EvaluationCase>,
    pub recipe: EvaluationRecipe,
    pub policy: EvaluationPolicy,
}

impl EvaluationManifest {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = read_bounded_json(path, "evaluation corpus manifest")?;
        let manifest: Self = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "parse evaluation corpus manifest {}: {error}",
                path.display()
            )
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn digest(&self) -> Result<Digest, String> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize evaluation manifest for digest: {error}"))?;
        Ok(domain_digest(EVALUATION_MANIFEST_DIGEST_DOMAIN, &encoded))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize evaluation manifest: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            EVALUATION_CORPUS_SCHEMA,
            "evaluation corpus manifest",
        )?;
        validate_id("corpus ID", &self.corpus_id)?;
        validate_text("corpus version", &self.corpus_version)?;
        validate_text("corpus title", &self.title)?;
        if self.cases.is_empty() || self.cases.len() > MAX_CASES {
            return Err(format!(
                "evaluation corpus must contain 1..={MAX_CASES} cases"
            ));
        }
        let mut previous = None;
        for case in &self.cases {
            validate_id("evaluation case ID", &case.id)?;
            if previous.is_some_and(|value: &str| value >= case.id.as_str()) {
                return Err("evaluation cases must have unique, strictly sorted IDs".into());
            }
            previous = Some(&case.id);
            validate_artifact("clean audio", &case.clean)?;
            validate_artifact("noisy audio", &case.noisy)?;
            if case.clean.path == case.noisy.path {
                return Err(format!(
                    "evaluation case {} must use distinct clean and noisy artifacts",
                    case.id
                ));
            }
            let mut previous_tag = None;
            for tag in &case.tags {
                validate_id("evaluation tag", tag)?;
                if previous_tag.is_some_and(|value: &str| value >= tag.as_str()) {
                    return Err(format!(
                        "evaluation case {} tags must be unique and sorted",
                        case.id
                    ));
                }
                previous_tag = Some(tag);
            }
        }
        self.recipe.validate()?;
        self.policy.validate()?;
        ensure_json_size(self, "evaluation corpus manifest")
    }
}

impl EvaluationRecipe {
    fn validate(&self) -> Result<(), String> {
        validate_id("evaluation backend", &self.backend)?;
        if !known_backend_name(&self.backend) {
            return Err(format!("unknown evaluation backend: {}", self.backend));
        }
        let preset = Preset::parse(&self.preset)
            .ok_or_else(|| format!("unknown evaluation preset: {}", self.preset))?;
        if preset_name(preset) != self.preset {
            return Err("evaluation preset must use its canonical lowercase name".into());
        }
        let accelerator = AcceleratorPreference::parse(&self.accelerator)
            .ok_or_else(|| format!("unknown evaluation accelerator: {}", self.accelerator))?;
        if accelerator.name() != self.accelerator {
            return Err("evaluation accelerator must use its canonical lowercase name".into());
        }
        let channel_mode = ChannelMode::parse(&self.channel_mode)
            .ok_or_else(|| format!("unknown evaluation channel mode: {}", self.channel_mode))?;
        if channel_mode_name(channel_mode) != self.channel_mode {
            return Err("evaluation channel mode must use its canonical name".into());
        }
        let profile = SgmseProfile::parse(&self.sgmse_profile)
            .ok_or_else(|| format!("unknown evaluation SGMSE profile: {}", self.sgmse_profile))?;
        if sgmse_profile_name(profile) != self.sgmse_profile {
            return Err("evaluation SGMSE profile must use its canonical name".into());
        }
        if !self.deterministic {
            return Err("release evaluation recipes must enable deterministic processing".into());
        }
        if !matches!(self.accelerator.as_str(), "cpu" | "auto") {
            return Err(
                "deterministic evaluation recipes must request the cpu or auto accelerator".into(),
            );
        }
        if self.seed.is_some_and(|seed| seed > MAX_JSON_SAFE_INTEGER) {
            return Err("evaluation seed exceeds the JSON safe-integer limit".into());
        }
        match (self.backend.as_str(), self.seed) {
            ("sgmse", None) => return Err("SGMSE evaluation recipes must pin a seed".into()),
            ("sgmse", Some(_)) | (_, None) => {}
            (_, Some(_)) => return Err("only SGMSE evaluation recipes may define a seed".into()),
        }
        let needs_model = matches!(
            self.backend.as_str(),
            "onnx" | "mpsenet" | "bsrnn" | "mossformer2" | "sgmse" | "gtcrn"
        );
        if needs_model != self.model.is_some() || needs_model != self.model_sample_rate.is_some() {
            return Err(
                "external-model evaluation recipes must pin both model and model_sample_rate"
                    .into(),
            );
        }
        if let Some(model) = &self.model {
            validate_artifact("evaluation model", model)?;
        }
        if self
            .model_sample_rate
            .is_some_and(|rate| rate == 0 || rate > 768_000)
        {
            return Err("evaluation model_sample_rate must be in 1..=768000 Hz".into());
        }
        Ok(())
    }
}

impl EvaluationPolicy {
    fn validate(&self) -> Result<(), String> {
        if self.warmup_runs > 20 {
            return Err("evaluation warmup_runs must be in 0..=20".into());
        }
        if !(1..=50).contains(&self.measured_runs) {
            return Err("evaluation measured_runs must be in 1..=50".into());
        }
        if !self.silence_threshold_dbfs.is_finite()
            || !(-200.0..=-20.0).contains(&self.silence_threshold_dbfs)
        {
            return Err("silence_threshold_dbfs must be finite and in -200..=-20".into());
        }
        if !(10..=1_000).contains(&self.dropout_window_ms) {
            return Err("dropout_window_ms must be in 10..=1000".into());
        }
        validate_threshold_policy(&self.thresholds)?;
        validate_regression_policy(&self.regression_tolerances)?;
        self.listening.validate()
    }
}

fn validate_threshold_policy(thresholds: &[EvaluationThreshold]) -> Result<(), String> {
    if thresholds.is_empty() {
        return Err("evaluation policy must define accepted thresholds".into());
    }
    let mut keys = BTreeSet::new();
    let mut categories = BTreeSet::new();
    for threshold in thresholds {
        validate_metric(&threshold.metric)?;
        validate_finite("evaluation threshold", threshold.value)?;
        if !keys.insert((
            threshold.metric.as_str(),
            threshold.aggregation,
            threshold.operator,
        )) {
            return Err(format!(
                "duplicate evaluation threshold for {}",
                threshold.metric
            ));
        }
        categories.insert(metric_category(&threshold.metric));
    }
    for required in ["objective", "perceptual", "output", "performance"] {
        if !categories.contains(required) {
            return Err(format!(
                "evaluation policy must threshold at least one {required} metric"
            ));
        }
    }
    Ok(())
}

fn validate_regression_policy(tolerances: &[RegressionTolerance]) -> Result<(), String> {
    if tolerances.is_empty() {
        return Err("evaluation policy must define regression tolerances".into());
    }
    let mut regressions = BTreeSet::new();
    for tolerance in tolerances {
        validate_metric(&tolerance.metric)?;
        if !tolerance.max_regression.is_finite() || tolerance.max_regression < 0.0 {
            return Err("max_regression must be a finite non-negative value".into());
        }
        if !regressions.insert((tolerance.metric.as_str(), tolerance.aggregation)) {
            return Err(format!(
                "duplicate regression tolerance for {}",
                tolerance.metric
            ));
        }
    }
    Ok(())
}

impl ListeningPolicy {
    fn validate(&self) -> Result<(), String> {
        validate_text("listening policy rationale", &self.rationale)?;
        match (&self.protocol, self.required) {
            (Some(protocol), _) => protocol.validate(),
            (None, false) => Ok(()),
            (None, true) => Err("a required listening test must define its protocol".into()),
        }
    }
}

impl ListeningProtocol {
    fn validate(&self) -> Result<(), String> {
        validate_id("listening protocol ID", &self.protocol_id)?;
        validate_immutable_revision("listening protocol revision", &self.revision)?;
        validate_text("listening method", &self.method)?;
        validate_uri("listening instructions URI", &self.instructions_uri)?;
        if !self.scale_min.is_finite()
            || !self.scale_max.is_finite()
            || self.scale_min >= self.scale_max
            || !self.acceptance_score.is_finite()
            || !(self.scale_min..=self.scale_max).contains(&self.acceptance_score)
        {
            return Err("listening scale and acceptance score are invalid".into());
        }
        if self.minimum_listeners == 0 || self.minimum_listeners > 100_000 {
            return Err("minimum_listeners must be in 1..=100000".into());
        }
        Ok(())
    }

    /// Return the domain-separated digest a human listening-test result must
    /// bind. The corpus validation report exposes the same value.
    pub fn digest(&self) -> Result<Digest, String> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize listening protocol: {error}"))?;
        Ok(domain_digest(LISTENING_PROTOCOL_DIGEST_DOMAIN, &encoded))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ListeningTestResult {
    pub schema: String,
    pub schema_version: u32,
    pub corpus_id: String,
    pub manifest_digest: Digest,
    pub protocol_digest: Digest,
    pub listener_count: u32,
    pub aggregate_score: f64,
    pub accepted: bool,
}

impl ListeningTestResult {
    pub fn from_file(path: impl AsRef<Path>) -> Result<(Self, FileFingerprint), String> {
        let path = path.as_ref();
        let bytes = read_bounded_json(path, "listening-test result")?;
        let result: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse listening-test result {}: {error}", path.display()))?;
        result.validate()?;
        let fingerprint = fingerprint_bytes(&bytes);
        Ok((result, fingerprint))
    }

    fn validate(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            LISTENING_RESULT_SCHEMA,
            "listening-test result",
        )?;
        validate_id("listening corpus ID", &self.corpus_id)?;
        if self.listener_count == 0 || self.listener_count > 100_000 {
            return Err("listening-test listener_count must be in 1..=100000".into());
        }
        validate_finite("listening aggregate score", self.aggregate_score)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCorpusValidation {
    pub schema: String,
    pub schema_version: u32,
    pub manifest_digest: Digest,
    pub corpus_id: String,
    pub corpus_version: String,
    pub cases: usize,
    pub total_artifact_bytes: u64,
    pub total_audio_seconds: f64,
    pub listening_protocol_digest: Option<Digest>,
}

/// Verify provenance, containment, hashes, decode integrity, and clean/noisy
/// comparability without running a denoiser.
pub fn validate_evaluation_corpus(
    manifest: &EvaluationManifest,
    corpus_root: impl AsRef<Path>,
) -> Result<EvaluationCorpusValidation, String> {
    manifest.validate()?;
    let root = canonical_corpus_root(corpus_root.as_ref())?;
    let mut total_bytes = 0_u64;
    let mut total_seconds = 0.0;
    for case in &manifest.cases {
        let clean_path = verify_artifact(&root, &case.clean, "clean audio")?;
        let noisy_path = verify_artifact(&root, &case.noisy, "noisy audio")?;
        let clean = decode_stable_artifact(&clean_path, &case.clean, "clean audio")?;
        let noisy = decode_stable_artifact(&noisy_path, &case.noisy, "noisy audio")?;
        validate_audio_pair(&case.id, &clean, &noisy)?;
        total_bytes = total_bytes
            .checked_add(case.clean.fingerprint.len)
            .and_then(|value| value.checked_add(case.noisy.fingerprint.len))
            .ok_or_else(|| "evaluation artifact byte total overflows".to_string())?;
        total_seconds += clean.frames() as f64 / clean.sample_rate as f64;
    }
    if let Some(model) = &manifest.recipe.model {
        verify_artifact(&root, model, "evaluation model")?;
        total_bytes = total_bytes
            .checked_add(model.fingerprint.len)
            .ok_or_else(|| "evaluation artifact byte total overflows".to_string())?;
    }
    if total_bytes > MAX_JSON_SAFE_INTEGER {
        return Err("evaluation artifact byte total exceeds the JSON safe-integer limit".into());
    }
    Ok(EvaluationCorpusValidation {
        schema: EVALUATION_CORPUS_VERIFICATION_SCHEMA.into(),
        schema_version: EVALUATION_SCHEMA_VERSION,
        manifest_digest: manifest.digest()?,
        corpus_id: manifest.corpus_id.clone(),
        corpus_version: manifest.corpus_version.clone(),
        cases: manifest.cases.len(),
        total_artifact_bytes: total_bytes,
        total_audio_seconds: finite(total_seconds, "total audio duration")?,
        listening_protocol_digest: manifest
            .policy
            .listening
            .protocol
            .as_ref()
            .map(ListeningProtocol::digest)
            .transpose()?,
    })
}

fn validate_artifact(label: &str, artifact: &EvaluationArtifact) -> Result<(), String> {
    validate_locator(&artifact.path).map_err(|error| format!("invalid {label}: {error}"))?;
    if artifact.fingerprint.len == 0 || artifact.fingerprint.len > MAX_JSON_SAFE_INTEGER {
        return Err(format!(
            "{label} length must be in 1..={MAX_JSON_SAFE_INTEGER} bytes"
        ));
    }
    validate_license(&artifact.license)?;
    validate_source(&artifact.source)?;
    validate_preparation(&artifact.preparation)
}

fn validate_license(license: &EvaluationLicense) -> Result<(), String> {
    validate_spdx_id(&license.spdx_id)?;
    validate_text("license name", &license.name)?;
    validate_uri("license URL", &license.url)
}

fn validate_spdx_id(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'+' | b'-'))
    {
        return Err(
            "license SPDX ID must be an SPDX identifier or LicenseRef using ASCII letters, digits, '.', '+', or '-'"
                .into(),
        );
    }
    Ok(())
}

fn validate_source(source: &EvaluationSource) -> Result<(), String> {
    validate_uri("artifact source URI", &source.uri)?;
    validate_immutable_revision("artifact source revision", &source.revision)
}

fn validate_preparation(preparation: &SignalPreparation) -> Result<(), String> {
    validate_text("signal preparation description", &preparation.description)?;
    validate_text("signal preparation tool", &preparation.tool)?;
    validate_immutable_revision("signal preparation tool version", &preparation.tool_version)
}

fn validate_immutable_revision(label: &str, revision: &str) -> Result<(), String> {
    validate_text(label, revision)?;
    if revision.trim() != revision {
        return Err(format!(
            "{label} must not contain leading or trailing whitespace"
        ));
    }
    let normalized = revision.to_ascii_lowercase();
    if matches!(
        normalized.as_str(),
        "latest" | "head" | "main" | "master" | "stable" | "current"
    ) || normalized.starts_with("refs/heads/")
        || normalized.starts_with("heads/")
        || normalized.starts_with("branch:")
    {
        return Err(format!("{label} must pin an immutable revision"));
    }
    Ok(())
}

fn validate_text(label: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || value.len() > MAX_TEXT_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(format!(
            "{label} must contain 1..={MAX_TEXT_BYTES} printable bytes"
        ));
    }
    Ok(())
}

fn validate_id(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._+-".contains(&byte)
        })
    {
        return Err(format!(
            "{label} must be 1..=128 lowercase ASCII letters, digits, '.', '_', '+', or '-'"
        ));
    }
    Ok(())
}

fn validate_uri(label: &str, value: &str) -> Result<(), String> {
    validate_text(label, value)?;
    if value.trim() != value {
        return Err(format!("{label} must not contain surrounding whitespace"));
    }
    let Some((scheme, rest)) = value.split_once(':') else {
        return Err(format!("{label} must be an absolute URI"));
    };
    if !scheme
        .bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic())
        || rest.is_empty()
        || !scheme
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
    {
        return Err(format!("{label} must be an absolute URI"));
    }
    url::Url::parse(value).map_err(|_| format!("{label} must be an absolute URI"))?;
    Ok(())
}

fn require_schema(schema: &str, version: u32, expected: &str, label: &str) -> Result<(), String> {
    if schema != expected || version != EVALUATION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported {label} schema/version: {schema}/{version}"
        ));
    }
    Ok(())
}

fn validate_finite(label: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{label} must be finite"))
    }
}

fn finite(value: f64, label: &str) -> Result<f64, String> {
    validate_finite(label, value)?;
    Ok(value)
}

fn ensure_json_size<T: Serialize>(value: &T, label: &str) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|error| format!("serialize {label}: {error}"))?;
    if bytes.len() as u64 >= MAX_JSON_BYTES {
        Err(format!("{label} exceeds the {MAX_JSON_BYTES}-byte limit"))
    } else {
        Ok(())
    }
}

fn read_bounded_json(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let (file, len) = crate::input::open_regular_file(path, label)?;
    if len >= MAX_JSON_BYTES {
        return Err(format!(
            "{label} {} exceeds the {MAX_JSON_BYTES}-byte limit",
            path.display()
        ));
    }
    let capacity = usize::try_from(len)
        .map_err(|_| format!("{label} {} is too large for this platform", path.display()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| format!("reserve {label} bytes"))?;
    file.take(MAX_JSON_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {label} {}: {error}", path.display()))?;
    if bytes.len() as u64 != len {
        return Err(format!("{label} changed while reading: {}", path.display()));
    }
    Ok(bytes)
}

fn domain_digest(domain: &[u8], value: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
    Digest::from_bytes(hasher.finalize().into())
}

fn fingerprint_bytes(bytes: &[u8]) -> FileFingerprint {
    FileFingerprint {
        len: bytes.len() as u64,
        digest: Digest::from_bytes(Sha256::digest(bytes).into()),
    }
}

fn validate_locator(locator: &str) -> Result<(), String> {
    if locator.is_empty() || locator.len() > MAX_LOCATOR_BYTES {
        return Err(format!(
            "artifact locator length must be in 1..={MAX_LOCATOR_BYTES} bytes"
        ));
    }
    if locator.starts_with('/')
        || locator.ends_with('/')
        || locator.contains('\\')
        || locator.contains(':')
        || locator.chars().any(char::is_control)
    {
        return Err("artifact locator must be a portable relative path".into());
    }
    for part in locator.split('/') {
        if part.is_empty() || part == "." || part == ".." || part.len() > 255 {
            return Err("artifact locator contains an unsafe path component".into());
        }
    }
    Ok(())
}

fn canonical_corpus_root(root: &Path) -> Result<PathBuf, String> {
    let root = std::fs::canonicalize(root)
        .map_err(|error| format!("resolve corpus root {}: {error}", root.display()))?;
    if !root.is_dir() {
        return Err(format!(
            "corpus root is not a directory: {}",
            root.display()
        ));
    }
    Ok(root)
}

fn verify_artifact(
    root: &Path,
    artifact: &EvaluationArtifact,
    label: &str,
) -> Result<PathBuf, String> {
    validate_artifact(label, artifact)?;
    let mut candidate = root.to_path_buf();
    for component in Path::new(&artifact.path).components() {
        let Component::Normal(part) = component else {
            return Err(format!(
                "{label} contains an unsafe path: {}",
                artifact.path
            ));
        };
        candidate.push(part);
        let metadata = std::fs::symlink_metadata(&candidate)
            .map_err(|error| format!("inspect {label} {}: {error}", candidate.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "{label} must not traverse a symlink: {}",
                artifact.path
            ));
        }
    }
    let resolved = std::fs::canonicalize(&candidate)
        .map_err(|error| format!("resolve {label} {}: {error}", candidate.display()))?;
    if !resolved.starts_with(root) {
        return Err(format!(
            "{label} escapes the corpus root: {}",
            artifact.path
        ));
    }
    if !resolved.is_file() {
        return Err(format!("{label} is not a regular file: {}", artifact.path));
    }
    let observed = batch_resume::fingerprint_file(&resolved)?;
    if observed != artifact.fingerprint {
        return Err(format!("{label} fingerprint mismatch: {}", artifact.path));
    }
    Ok(resolved)
}

fn decode_stable_artifact(
    path: &Path,
    artifact: &EvaluationArtifact,
    label: &str,
) -> Result<Audio, String> {
    let audio = read_audio(path).map_err(|error| format!("decode {label}: {error}"))?;
    let after = batch_resume::fingerprint_file(path)?;
    if after != artifact.fingerprint {
        return Err(format!(
            "{label} changed while it was decoded: {}",
            artifact.path
        ));
    }
    validate_audio(label, &audio)?;
    Ok(audio)
}

fn validate_audio(label: &str, audio: &Audio) -> Result<(), String> {
    if audio.sample_rate == 0 || audio.channels() == 0 || audio.frames() == 0 {
        return Err(format!("{label} has empty audio geometry"));
    }
    let frames = audio.frames();
    if audio.channels.iter().any(|channel| channel.len() != frames) {
        return Err(format!("{label} has unequal channel lengths"));
    }
    if audio
        .channels
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite())
    {
        return Err(format!("{label} contains non-finite decoded samples"));
    }
    Ok(())
}

fn validate_audio_pair(id: &str, clean: &Audio, noisy: &Audio) -> Result<(), String> {
    if clean.sample_rate != noisy.sample_rate {
        return Err(format!(
            "evaluation case {id} clean/noisy sample rates differ"
        ));
    }
    if clean.channels() != noisy.channels() {
        return Err(format!(
            "evaluation case {id} clean/noisy channel counts differ"
        ));
    }
    if clean.frames() != noisy.frames() {
        return Err(format!("evaluation case {id} clean/noisy durations differ"));
    }
    if clean.channel_layout() != noisy.channel_layout() {
        return Err(format!(
            "evaluation case {id} clean/noisy channel layouts differ"
        ));
    }
    Ok(())
}

fn known_backend_name(name: &str) -> bool {
    matches!(
        name,
        "classical"
            | "rnnoise"
            | "deepfilter"
            | "onnx"
            | "mpsenet"
            | "bsrnn"
            | "mossformer2"
            | "sgmse"
            | "gtcrn"
    )
}

fn preset_name(preset: Preset) -> &'static str {
    match preset {
        Preset::Speech => "speech",
        Preset::Music => "music",
        Preset::Aggressive => "aggressive",
        Preset::Gentle => "gentle",
        Preset::Restore => "restore",
        Preset::HiFi => "hifi",
    }
}

fn channel_mode_name(mode: ChannelMode) -> &'static str {
    match mode {
        ChannelMode::Independent => "independent",
        ChannelMode::StereoLinked => "stereo-linked",
        ChannelMode::MidSide => "mid-side",
    }
}

fn sgmse_profile_name(profile: SgmseProfile) -> &'static str {
    match profile {
        SgmseProfile::Fast => "fast",
        SgmseProfile::Balanced => "balanced",
        SgmseProfile::Quality => "quality",
    }
}

fn metric_category(metric: &str) -> &str {
    metric.split_once('.').map(|value| value.0).unwrap_or("")
}

fn validate_metric(metric: &str) -> Result<(), String> {
    if known_metric(metric) {
        Ok(())
    } else {
        Err(format!("unknown evaluation metric: {metric}"))
    }
}

fn known_metric(metric: &str) -> bool {
    matches!(
        metric,
        "objective.si-sdr-db"
            | "objective.si-sdr-improvement-db"
            | "objective.si-snr-db"
            | "objective.snr-db"
            | "objective.segmental-snr-db"
            | "perceptual.stoi"
            | "perceptual.stoi-improvement"
            | "output.decode-integrity"
            | "performance.throughput-x"
            | "perceptual.musical-noise"
            | "perceptual.pumping"
            | "perceptual.transient-loss"
            | "perceptual.phase-distortion"
            | "output.duration-error-frames"
            | "output.sample-rate-mismatch"
            | "output.channel-mismatch"
            | "output.clipping-ratio"
            | "output.sample-peak-dbfs"
            | "output.true-peak-dbtp"
            | "output.dc-offset-abs"
            | "output.silence-ratio"
            | "output.dropout-ratio"
            | "output.integrated-lufs"
            | "output.non-finite-samples"
            | "performance.realtime-factor"
            | "performance.elapsed-ms"
            | "performance.peak-rss-bytes"
    )
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationEnvironment {
    pub os: String,
    pub architecture: String,
    pub logical_cpus: usize,
    pub cpu_features: Vec<String>,
    pub compiled_backends: Vec<String>,
    pub build_profile: String,
    pub backend: String,
    pub accelerator_requested: String,
    pub accelerator_effective: String,
    pub accelerator_fallback: Option<String>,
    pub accelerator_device: Option<String>,
    pub deterministic: bool,
    pub seed: Option<u64>,
    pub channel_mode: String,
    pub sgmse_profile: String,
    pub model: Option<FileFingerprint>,
    pub model_load_included_in_timing: bool,
    pub timing_scope: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveEvaluationMetrics {
    pub si_sdr_db: f64,
    pub si_sdr_improvement_db: f64,
    pub si_snr_db: f64,
    pub snr_db: f64,
    pub segmental_snr_db: f64,
    pub stoi: Option<f64>,
    pub stoi_improvement: Option<f64>,
    pub musical_noise: f64,
    pub pumping: f64,
    pub transient_loss: f64,
    pub phase_distortion: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputQualityMetrics {
    pub reference_sample_rate: u32,
    pub output_sample_rate: u32,
    pub sample_rate_matches: bool,
    pub reference_frames: u64,
    pub output_frames: u64,
    pub duration_error_frames: u64,
    pub reference_channels: u32,
    pub output_channels: u32,
    pub channel_mismatch: u32,
    pub reference_layout: String,
    pub output_layout: String,
    pub layout_matches: bool,
    pub clipping_samples: u64,
    pub clipping_ratio: f64,
    pub sample_peak_dbfs: f64,
    pub true_peak_dbtp: f64,
    pub dc_offset_per_channel: Vec<f64>,
    pub dc_offset_abs: f64,
    pub silence_ratio: f64,
    pub dropout_ratio: f64,
    pub integrated_lufs: Option<f64>,
    pub non_finite_samples: u64,
    pub decode_integrity: bool,
    pub fingerprint: FileFingerprint,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceMetrics {
    pub audio_seconds: f64,
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub elapsed_ms: Vec<f64>,
    pub median_elapsed_ms: f64,
    pub p95_elapsed_ms: f64,
    pub realtime_factor: f64,
    pub throughput_x: f64,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCaseResult {
    pub id: String,
    pub clean: FileFingerprint,
    pub noisy: FileFingerprint,
    pub sample_rate: u32,
    pub channels: u32,
    pub frames: u64,
    pub objective: ObjectiveEvaluationMetrics,
    pub output_quality: OutputQualityMetrics,
    pub performance: PerformanceMetrics,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdOutcome {
    pub metric: String,
    pub aggregation: ThresholdAggregation,
    pub operator: ThresholdOperator,
    pub limit: f64,
    pub observed: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ListeningEvidence {
    pub required: bool,
    pub protocol_digest: Option<Digest>,
    pub result_fingerprint: Option<FileFingerprint>,
    pub listener_count: Option<u32>,
    pub aggregate_score: Option<f64>,
    pub accepted: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationResultPayload {
    pub completed_at_unix_seconds: u64,
    pub manifest_digest: Digest,
    pub corpus_id: String,
    pub corpus_version: String,
    pub denoize_version: String,
    pub environment: EvaluationEnvironment,
    pub cases: Vec<EvaluationCaseResult>,
    pub thresholds: Vec<EvaluationThreshold>,
    pub regression_tolerances: Vec<RegressionTolerance>,
    pub threshold_outcomes: Vec<ThresholdOutcome>,
    pub listening: ListeningEvidence,
    pub accepted: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedEvaluationResult {
    pub schema: String,
    pub schema_version: u32,
    pub payload: EvaluationResultPayload,
    pub signature: ReceiptSignature,
}

impl SignedEvaluationResult {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = read_bounded_json(path, "signed evaluation result")?;
        let result: Self = serde_json::from_slice(&bytes).map_err(|error| {
            format!("parse signed evaluation result {}: {error}", path.display())
        })?;
        result.validate_structure()?;
        Ok(result)
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize signed evaluation result: {error}"))
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize evaluation payload for verification: {error}"))?;
        key.verify_domain_document(
            EVALUATION_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "evaluation result",
        )
    }

    pub fn verify_manifest(&self, manifest: &EvaluationManifest) -> Result<(), String> {
        manifest.validate()?;
        if self.payload.manifest_digest != manifest.digest()?
            || self.payload.corpus_id != manifest.corpus_id
            || self.payload.corpus_version != manifest.corpus_version
            || self.payload.thresholds != manifest.policy.thresholds
            || self.payload.regression_tolerances != manifest.policy.regression_tolerances
        {
            return Err("evaluation result does not match the supplied manifest".into());
        }
        Ok(())
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        require_schema(
            &self.schema,
            self.schema_version,
            EVALUATION_RESULT_SCHEMA,
            "signed evaluation result",
        )?;
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err(format!(
                "unsupported evaluation signature algorithm: {}",
                self.signature.algorithm
            ));
        }
        let _: Digest = self
            .signature
            .key_id
            .parse()
            .map_err(|error| format!("invalid evaluation key ID: {error}"))?;
        if self
            .signature
            .key_id
            .bytes()
            .any(|byte| byte.is_ascii_uppercase())
        {
            return Err("evaluation key ID must use lowercase hexadecimal".into());
        }
        ensure_json_size(self, "signed evaluation result")
    }
}

impl EvaluationResultPayload {
    fn validate(&self) -> Result<(), String> {
        if self.completed_at_unix_seconds > MAX_JSON_SAFE_INTEGER {
            return Err(
                "evaluation completion timestamp exceeds the JSON safe-integer limit".into(),
            );
        }
        validate_id("evaluation result corpus ID", &self.corpus_id)?;
        validate_text("evaluation result corpus version", &self.corpus_version)?;
        validate_text("denoize version", &self.denoize_version)?;
        self.environment.validate()?;
        if self.cases.is_empty() || self.cases.len() > MAX_CASES {
            return Err(format!(
                "evaluation result must contain 1..={MAX_CASES} cases"
            ));
        }
        let mut previous = None;
        for case in &self.cases {
            case.validate()?;
            if previous.is_some_and(|value: &str| value >= case.id.as_str()) {
                return Err("evaluation result cases must have unique, sorted IDs".into());
            }
            previous = Some(&case.id);
        }
        validate_threshold_policy(&self.thresholds)?;
        validate_regression_policy(&self.regression_tolerances)?;
        if self.threshold_outcomes.len() != self.thresholds.len() {
            return Err("evaluation result must contain one outcome per threshold".into());
        }
        for (threshold, outcome) in self.thresholds.iter().zip(&self.threshold_outcomes) {
            outcome.validate()?;
            if outcome.metric != threshold.metric
                || outcome.aggregation != threshold.aggregation
                || outcome.operator != threshold.operator
                || outcome.limit != threshold.value
            {
                return Err("evaluation threshold outcome does not match its policy".into());
            }
        }
        let expected_outcomes = evaluate_thresholds(&self.thresholds, &self.cases)?;
        if self.threshold_outcomes != expected_outcomes {
            return Err(
                "evaluation threshold outcomes do not match the signed case measurements".into(),
            );
        }
        self.listening.validate()?;
        let expected_accepted =
            self.threshold_outcomes.iter().all(|outcome| outcome.passed) && self.listening.accepted;
        if self.accepted != expected_accepted {
            return Err("evaluation accepted flag is inconsistent with its evidence".into());
        }
        Ok(())
    }
}

impl EvaluationEnvironment {
    fn validate(&self) -> Result<(), String> {
        for (label, value) in [
            ("environment OS", &self.os),
            ("environment architecture", &self.architecture),
            ("environment build profile", &self.build_profile),
            ("environment backend", &self.backend),
            ("requested accelerator", &self.accelerator_requested),
            ("effective accelerator", &self.accelerator_effective),
            ("environment channel mode", &self.channel_mode),
            ("environment SGMSE profile", &self.sgmse_profile),
            ("performance timing scope", &self.timing_scope),
        ] {
            validate_text(label, value)?;
        }
        if self.logical_cpus == 0 || self.logical_cpus as u64 > MAX_JSON_SAFE_INTEGER {
            return Err("evaluation environment logical CPU count is invalid".into());
        }
        validate_sorted_strings("CPU features", &self.cpu_features)?;
        validate_sorted_strings("compiled backends", &self.compiled_backends)?;
        if !matches!(self.build_profile.as_str(), "debug" | "release") {
            return Err("evaluation build profile must be debug or release".into());
        }
        if !known_backend_name(&self.backend)
            || self.compiled_backends.is_empty()
            || !self
                .compiled_backends
                .iter()
                .any(|value| value == &self.backend)
            || self
                .compiled_backends
                .iter()
                .any(|value| !known_backend_name(value))
        {
            return Err("evaluation backend is inconsistent with compiled backends".into());
        }
        if !self.deterministic {
            return Err("signed release evaluations must use deterministic processing".into());
        }
        let accelerator_is_consistent = match self.accelerator_requested.as_str() {
            "cpu" => self.accelerator_effective == "cpu" && self.accelerator_fallback.is_none(),
            "auto" => {
                self.accelerator_effective == "cpu"
                    && self.accelerator_fallback.as_deref() == Some("deterministic-mode")
            }
            _ => false,
        };
        if !accelerator_is_consistent || self.accelerator_device.is_some() {
            return Err("evaluation accelerator selection is inconsistent".into());
        }
        let canonical_channel_mode = ChannelMode::parse(&self.channel_mode)
            .map(channel_mode_name)
            .is_some_and(|value| value == self.channel_mode);
        let canonical_sgmse_profile = SgmseProfile::parse(&self.sgmse_profile)
            .map(sgmse_profile_name)
            .is_some_and(|value| value == self.sgmse_profile);
        if !canonical_channel_mode || !canonical_sgmse_profile {
            return Err("evaluation environment contains an unknown recipe control".into());
        }
        if self.seed.is_some_and(|seed| seed > MAX_JSON_SAFE_INTEGER) {
            return Err("evaluation environment seed exceeds the JSON safe-integer limit".into());
        }
        match (self.backend.as_str(), self.seed) {
            ("sgmse", None) => {
                return Err("SGMSE evaluation environment is missing its seed".into())
            }
            ("sgmse", Some(_)) | (_, None) => {}
            (_, Some(_)) => return Err("non-SGMSE evaluation environment contains a seed".into()),
        }
        let needs_model = matches!(
            self.backend.as_str(),
            "onnx" | "mpsenet" | "bsrnn" | "mossformer2" | "sgmse" | "gtcrn"
        );
        if needs_model != self.model.is_some() {
            return Err("evaluation environment model fingerprint is inconsistent".into());
        }
        if self.model_load_included_in_timing || self.timing_scope != "denoise-processing-only" {
            return Err("evaluation timing must exclude one-time model loading".into());
        }
        if let Some(device) = &self.accelerator_device {
            validate_text("accelerator device", device)?;
        }
        if let Some(fallback) = &self.accelerator_fallback {
            validate_text("accelerator fallback", fallback)?;
        }
        if self
            .model
            .is_some_and(|value| value.len == 0 || value.len > MAX_JSON_SAFE_INTEGER)
        {
            return Err("evaluation model fingerprint must be non-empty".into());
        }
        Ok(())
    }
}

impl EvaluationCaseResult {
    fn validate(&self) -> Result<(), String> {
        validate_id("evaluation result case ID", &self.id)?;
        for (label, fingerprint) in [
            ("clean audio", self.clean),
            ("noisy audio", self.noisy),
            ("output audio", self.output_quality.fingerprint),
        ] {
            if fingerprint.len == 0 || fingerprint.len > MAX_JSON_SAFE_INTEGER {
                return Err(format!("{label} fingerprint length is invalid"));
            }
        }
        if self.sample_rate == 0
            || self.channels == 0
            || self.frames == 0
            || self.frames > MAX_JSON_SAFE_INTEGER
        {
            return Err("evaluation result audio geometry must be non-zero".into());
        }
        self.objective.validate()?;
        self.output_quality.validate()?;
        self.performance.validate()?;
        if self.output_quality.reference_frames != self.frames
            || self.output_quality.reference_channels != self.channels
            || self.output_quality.reference_sample_rate != self.sample_rate
        {
            return Err("output-quality reference geometry does not match its case".into());
        }
        let expected_audio_seconds = self.frames as f64 / self.sample_rate as f64;
        if self.performance.audio_seconds != expected_audio_seconds {
            return Err("performance duration does not match its evaluation case".into());
        }
        Ok(())
    }
}

impl ObjectiveEvaluationMetrics {
    fn validate(&self) -> Result<(), String> {
        for (label, value) in [
            ("SI-SDR", self.si_sdr_db),
            ("SI-SDR improvement", self.si_sdr_improvement_db),
            ("SI-SNR", self.si_snr_db),
            ("SNR", self.snr_db),
            ("segmental SNR", self.segmental_snr_db),
            ("musical-noise score", self.musical_noise),
            ("pumping score", self.pumping),
            ("transient-loss score", self.transient_loss),
        ] {
            validate_finite(label, value)?;
        }
        for value in [self.musical_noise, self.pumping, self.transient_loss] {
            if !(0.0..=1.0).contains(&value) {
                return Err("perceptual artifact scores must be in 0..=1".into());
            }
        }
        if self
            .phase_distortion
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        {
            return Err("phase-distortion score must be in 0..=1".into());
        }
        for (label, value) in [
            ("STOI", self.stoi),
            ("STOI improvement", self.stoi_improvement),
            ("phase distortion", self.phase_distortion),
        ] {
            if let Some(value) = value {
                validate_finite(label, value)?;
            }
        }
        Ok(())
    }
}

impl OutputQualityMetrics {
    fn validate(&self) -> Result<(), String> {
        if self.reference_frames == 0
            || self.output_frames == 0
            || self.reference_frames > MAX_JSON_SAFE_INTEGER
            || self.output_frames > MAX_JSON_SAFE_INTEGER
            || self.reference_channels == 0
            || self.output_channels == 0
            || self.reference_sample_rate == 0
            || self.output_sample_rate == 0
        {
            return Err("output-quality audio geometry must be non-zero".into());
        }
        for (label, value) in [
            ("clipping ratio", self.clipping_ratio),
            ("sample peak", self.sample_peak_dbfs),
            ("true peak", self.true_peak_dbtp),
            ("absolute DC offset", self.dc_offset_abs),
            ("silence ratio", self.silence_ratio),
            ("dropout ratio", self.dropout_ratio),
        ] {
            validate_finite(label, value)?;
        }
        if !(0.0..=1.0).contains(&self.clipping_ratio)
            || !(0.0..=1.0).contains(&self.silence_ratio)
            || !(0.0..=1.0).contains(&self.dropout_ratio)
        {
            return Err("output-quality ratios must be in 0..=1".into());
        }
        if self.sample_rate_matches != (self.reference_sample_rate == self.output_sample_rate)
            || self.duration_error_frames != self.reference_frames.abs_diff(self.output_frames)
            || self.channel_mismatch != self.reference_channels.abs_diff(self.output_channels)
            || self.layout_matches != (self.reference_layout == self.output_layout)
        {
            return Err("output-quality geometry flags are inconsistent".into());
        }
        let total_samples = self
            .output_frames
            .checked_mul(u64::from(self.output_channels))
            .ok_or_else(|| "output-quality sample count overflows".to_string())?;
        if self.clipping_samples > total_samples
            || self.non_finite_samples > total_samples
            || self.clipping_samples > MAX_JSON_SAFE_INTEGER
            || self.non_finite_samples > MAX_JSON_SAFE_INTEGER
        {
            return Err("output-quality sample counters are invalid".into());
        }
        let expected_clipping_ratio = self.clipping_samples as f64 / total_samples as f64;
        if self.clipping_ratio != expected_clipping_ratio {
            return Err("output-quality clipping ratio is inconsistent".into());
        }
        if self.dc_offset_per_channel.len() != self.output_channels as usize {
            return Err("output-quality DC offsets must cover every channel".into());
        }
        for value in &self.dc_offset_per_channel {
            validate_finite("channel DC offset", *value)?;
        }
        let expected_dc_offset_abs = self
            .dc_offset_per_channel
            .iter()
            .fold(0.0_f64, |value, item| value.max(item.abs()));
        if self.dc_offset_abs != expected_dc_offset_abs {
            return Err("output-quality maximum DC offset is inconsistent".into());
        }
        if let Some(value) = self.integrated_lufs {
            validate_finite("integrated loudness", value)?;
        }
        if self.decode_integrity
            != (self.sample_rate_matches
                && self.duration_error_frames == 0
                && self.channel_mismatch == 0
                && self.layout_matches
                && self.non_finite_samples == 0)
        {
            return Err("output decode-integrity flag is inconsistent".into());
        }
        Ok(())
    }
}

impl PerformanceMetrics {
    fn validate(&self) -> Result<(), String> {
        if !self.audio_seconds.is_finite() || self.audio_seconds <= 0.0 {
            return Err("performance audio duration must be finite and positive".into());
        }
        if self.measured_runs == 0 || self.elapsed_ms.len() != self.measured_runs as usize {
            return Err("performance elapsed samples must match measured_runs".into());
        }
        if self.warmup_runs > 20 || self.measured_runs > 50 {
            return Err("performance run counts exceed evaluation bounds".into());
        }
        let mut previous = None;
        for value in &self.elapsed_ms {
            if !value.is_finite() || *value < 0.0 {
                return Err("performance elapsed samples must be finite and non-negative".into());
            }
            if previous.is_some_and(|prior| prior > *value) {
                return Err("performance elapsed samples must be sorted".into());
            }
            previous = Some(*value);
        }
        for (label, value) in [
            ("median elapsed time", self.median_elapsed_ms),
            ("p95 elapsed time", self.p95_elapsed_ms),
            ("realtime factor", self.realtime_factor),
            ("throughput", self.throughput_x),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{label} must be finite and non-negative"));
            }
        }
        let expected_median = percentile(&self.elapsed_ms, 0.5)?;
        let expected_p95 = percentile(&self.elapsed_ms, 0.95)?;
        let expected_realtime_factor = (expected_median / 1_000.0) / self.audio_seconds;
        let expected_throughput = 1.0 / expected_realtime_factor.max(f64::MIN_POSITIVE);
        if self.median_elapsed_ms != expected_median
            || self.p95_elapsed_ms != expected_p95
            || self.realtime_factor != expected_realtime_factor
            || self.throughput_x != expected_throughput
        {
            return Err("derived performance measurements are inconsistent".into());
        }
        if self
            .peak_rss_bytes
            .is_some_and(|value| value > MAX_JSON_SAFE_INTEGER)
        {
            return Err("peak RSS exceeds the JSON safe-integer limit".into());
        }
        Ok(())
    }
}

impl ThresholdOutcome {
    fn validate(&self) -> Result<(), String> {
        validate_metric(&self.metric)?;
        validate_finite("threshold limit", self.limit)?;
        validate_finite("threshold observation", self.observed)
    }
}

impl ListeningEvidence {
    fn validate(&self) -> Result<(), String> {
        if self.required {
            if self.protocol_digest.is_none()
                || self.result_fingerprint.is_none()
                || self.listener_count.is_none()
                || self.aggregate_score.is_none()
            {
                return Err("required listening evidence is incomplete".into());
            }
        } else if self.result_fingerprint.is_some()
            || self.listener_count.is_some()
            || self.aggregate_score.is_some()
        {
            return Err("non-required listening policy must not attach a result".into());
        }
        if let Some(value) = self.aggregate_score {
            validate_finite("listening evidence score", value)?;
        }
        if self.result_fingerprint.is_some_and(|value| value.len == 0) {
            return Err("listening result fingerprint must be non-empty".into());
        }
        if !self.required && !self.accepted {
            return Err("non-required listening evidence must be accepted".into());
        }
        Ok(())
    }
}

fn validate_sorted_strings(label: &str, values: &[String]) -> Result<(), String> {
    let mut previous = None;
    for value in values {
        validate_text(label, value)?;
        if previous.is_some_and(|prior: &str| prior >= value.as_str()) {
            return Err(format!("{label} must be unique and sorted"));
        }
        previous = Some(value);
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationVerificationReport {
    pub schema: String,
    pub schema_version: u32,
    pub key_id: String,
    pub manifest_digest: Digest,
    pub corpus_id: String,
    pub cases: usize,
    pub accepted: bool,
}

pub fn verify_evaluation_result(
    result: &SignedEvaluationResult,
    key: &ReceiptPublicKey,
    manifest: Option<&EvaluationManifest>,
) -> Result<EvaluationVerificationReport, String> {
    result.verify_signature(key)?;
    if let Some(manifest) = manifest {
        result.verify_manifest(manifest)?;
    }
    Ok(EvaluationVerificationReport {
        schema: EVALUATION_VERIFICATION_SCHEMA.into(),
        schema_version: EVALUATION_SCHEMA_VERSION,
        key_id: result.signature.key_id.clone(),
        manifest_digest: result.payload.manifest_digest,
        corpus_id: result.payload.corpus_id.clone(),
        cases: result.payload.cases.len(),
        accepted: result.payload.accepted,
    })
}

/// Execute one pinned corpus and return a signed result even when an accepted
/// threshold is missed.  Callers can publish the failed evidence before
/// returning a non-zero release-gate status.
pub fn run_evaluation(
    manifest: &EvaluationManifest,
    corpus_root: impl AsRef<Path>,
    signing_key: &ReceiptSecretKey,
    listening_result_path: Option<&Path>,
) -> Result<SignedEvaluationResult, String> {
    manifest.validate()?;
    let root = canonical_corpus_root(corpus_root.as_ref())?;
    let backend = Backend::parse(&manifest.recipe.backend).ok_or_else(|| {
        format!(
            "evaluation backend unavailable: {}",
            manifest.recipe.backend
        )
    })?;
    let model_path = manifest
        .recipe
        .model
        .as_ref()
        .map(|artifact| verify_artifact(&root, artifact, "evaluation model"))
        .transpose()?;
    let options = evaluation_backend_options(&manifest.recipe, model_path.as_deref())?;
    let session = BackendSession::prepare(backend, options)?;
    if let (Some(path), Some(artifact)) = (model_path.as_ref(), manifest.recipe.model.as_ref()) {
        if batch_resume::fingerprint_file(path)? != artifact.fingerprint {
            return Err("evaluation model changed while its backend was prepared".into());
        }
    }
    let environment = evaluation_environment(manifest, &session);
    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        cases.push(run_evaluation_case(manifest, case, &root, &session)?);
    }
    let threshold_outcomes = evaluate_thresholds(&manifest.policy.thresholds, &cases)?;
    let manifest_digest = manifest.digest()?;
    let listening = bind_listening_evidence(manifest, manifest_digest, listening_result_path)?;
    let accepted = threshold_outcomes.iter().all(|outcome| outcome.passed) && listening.accepted;
    let completed_at_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock is before the Unix epoch".to_string())?
        .as_secs();
    let payload = EvaluationResultPayload {
        completed_at_unix_seconds,
        manifest_digest,
        corpus_id: manifest.corpus_id.clone(),
        corpus_version: manifest.corpus_version.clone(),
        denoize_version: env!("CARGO_PKG_VERSION").into(),
        environment,
        cases,
        thresholds: manifest.policy.thresholds.clone(),
        regression_tolerances: manifest.policy.regression_tolerances.clone(),
        threshold_outcomes,
        listening,
        accepted,
    };
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize evaluation payload for signing: {error}"))?;
    let signature = signing_key.sign_domain_document(
        EVALUATION_SIGNATURE_DOMAIN,
        &document,
        "evaluation result",
    )?;
    let result = SignedEvaluationResult {
        schema: EVALUATION_RESULT_SCHEMA.into(),
        schema_version: EVALUATION_SCHEMA_VERSION,
        payload,
        signature,
    };
    result.validate_structure()?;
    Ok(result)
}

fn evaluation_backend_options(
    recipe: &EvaluationRecipe,
    model_path: Option<&Path>,
) -> Result<BackendOptions, String> {
    let model = match (model_path, recipe.model_sample_rate) {
        (Some(path), Some(sample_rate)) => Some(OnnxModelConfig {
            path: path.to_path_buf(),
            sample_rate,
        }),
        (None, None) => None,
        _ => return Err("evaluation model path/rate contract is incomplete".into()),
    };
    Ok(BackendOptions {
        onnx: model,
        runtime_package: None,
        channel_mode: ChannelMode::parse(&recipe.channel_mode)
            .ok_or("invalid evaluation channel mode")?,
        sgmse_profile: SgmseProfile::parse(&recipe.sgmse_profile)
            .ok_or("invalid evaluation SGMSE profile")?,
        deterministic: recipe.deterministic,
        accelerator: AcceleratorPreference::parse(&recipe.accelerator)
            .ok_or("invalid evaluation accelerator")?,
        seed: recipe.seed,
    })
}

fn evaluation_environment(
    manifest: &EvaluationManifest,
    session: &BackendSession,
) -> EvaluationEnvironment {
    let hardware = hardware_capabilities();
    let selection = session.accelerator();
    let mut cpu_features: Vec<String> = hardware
        .cpu_features()
        .iter()
        .map(|value| (*value).to_string())
        .collect();
    cpu_features.sort();
    cpu_features.dedup();
    let mut compiled_backends: Vec<String> = Backend::available_names()
        .iter()
        .map(|value| (*value).to_string())
        .collect();
    compiled_backends.sort();
    let accelerator_device = hardware
        .runtimes()
        .iter()
        .find(|runtime| runtime.runtime() == selection.effective())
        .and_then(|runtime| runtime.device())
        .map(str::to_string);
    EvaluationEnvironment {
        os: hardware.os().into(),
        architecture: hardware.architecture().into(),
        logical_cpus: hardware.logical_cpus(),
        cpu_features,
        compiled_backends,
        build_profile: if cfg!(debug_assertions) {
            "debug".into()
        } else {
            "release".into()
        },
        backend: manifest.recipe.backend.clone(),
        accelerator_requested: selection.requested().name().into(),
        accelerator_effective: selection.effective().name().into(),
        accelerator_fallback: selection.fallback().map(|value| value.name().into()),
        accelerator_device,
        deterministic: manifest.recipe.deterministic,
        seed: manifest.recipe.seed,
        channel_mode: manifest.recipe.channel_mode.clone(),
        sgmse_profile: manifest.recipe.sgmse_profile.clone(),
        model: manifest
            .recipe
            .model
            .as_ref()
            .map(|value| value.fingerprint),
        model_load_included_in_timing: false,
        timing_scope: "denoise-processing-only".into(),
    }
}

fn run_evaluation_case(
    manifest: &EvaluationManifest,
    case: &EvaluationCase,
    root: &Path,
    session: &BackendSession,
) -> Result<EvaluationCaseResult, String> {
    let clean_path = verify_artifact(root, &case.clean, "clean audio")?;
    let noisy_path = verify_artifact(root, &case.noisy, "noisy audio")?;
    let clean = decode_stable_artifact(&clean_path, &case.clean, "clean audio")?;
    let noisy = decode_stable_artifact(&noisy_path, &case.noisy, "noisy audio")?;
    validate_audio_pair(&case.id, &clean, &noisy)?;
    let config = Preset::parse(&manifest.recipe.preset)
        .ok_or("invalid evaluation preset")?
        .config(noisy.sample_rate);

    for _ in 0..manifest.policy.warmup_runs {
        let mut warmup = noisy.clone();
        crate::denoise_audio_with_backend_session(&mut warmup, config.clone(), session)?;
    }
    let rss_before = peak_rss_bytes();
    let mut elapsed = Vec::with_capacity(manifest.policy.measured_runs as usize);
    let mut enhanced = None;
    for _ in 0..manifest.policy.measured_runs {
        let mut output = noisy.clone();
        let duration =
            crate::denoise_audio_with_backend_session(&mut output, config.clone(), session)?;
        elapsed.push(duration.as_secs_f64() * 1_000.0);
        enhanced = Some(output);
    }
    elapsed.sort_by(f64::total_cmp);
    let enhanced = enhanced.ok_or("evaluation produced no measured output")?;
    let encoded = write_wav_bytes(&enhanced)?;
    let output_fingerprint = fingerprint_bytes(&encoded);
    let decoded =
        read_wav_bytes(encoded).map_err(|error| format!("decode evaluation output: {error}"))?;
    validate_audio("evaluation output", &decoded)?;
    let comparison = ComparisonReport::compare(&clean, &noisy, &decoded)?;
    let objective = objective_metrics(&comparison)?;
    let output_quality = output_quality_metrics(
        &clean,
        &decoded,
        output_fingerprint,
        manifest.policy.silence_threshold_dbfs,
        manifest.policy.dropout_window_ms,
    )?;
    let audio_seconds = noisy.frames() as f64 / noisy.sample_rate as f64;
    let median_elapsed_ms = percentile(&elapsed, 0.5)?;
    let p95_elapsed_ms = percentile(&elapsed, 0.95)?;
    let realtime_factor = (median_elapsed_ms / 1_000.0) / audio_seconds;
    let performance = PerformanceMetrics {
        audio_seconds: finite(audio_seconds, "case audio duration")?,
        warmup_runs: manifest.policy.warmup_runs,
        measured_runs: manifest.policy.measured_runs,
        elapsed_ms: elapsed,
        median_elapsed_ms: finite(median_elapsed_ms, "median elapsed time")?,
        p95_elapsed_ms: finite(p95_elapsed_ms, "p95 elapsed time")?,
        realtime_factor: finite(realtime_factor, "realtime factor")?,
        throughput_x: finite(1.0 / realtime_factor.max(f64::MIN_POSITIVE), "throughput")?,
        peak_rss_bytes: peak_rss_bytes().or(rss_before),
    };
    let result = EvaluationCaseResult {
        id: case.id.clone(),
        clean: case.clean.fingerprint,
        noisy: case.noisy.fingerprint,
        sample_rate: clean.sample_rate,
        channels: clean.channels() as u32,
        frames: clean.frames() as u64,
        objective,
        output_quality,
        performance,
    };
    result.validate()?;
    Ok(result)
}

fn objective_metrics(comparison: &ComparisonReport) -> Result<ObjectiveEvaluationMetrics, String> {
    let noisy = &comparison.noisy;
    let enhanced = &comparison.enhanced;
    Ok(ObjectiveEvaluationMetrics {
        si_sdr_db: finite(enhanced.si_sdr_db, "SI-SDR")?,
        si_sdr_improvement_db: finite(enhanced.si_sdr_db - noisy.si_sdr_db, "SI-SDR improvement")?,
        si_snr_db: finite(enhanced.si_snr_db, "SI-SNR")?,
        snr_db: finite(enhanced.snr_db, "SNR")?,
        segmental_snr_db: finite(enhanced.segmental_snr_db, "segmental SNR")?,
        stoi: enhanced
            .stoi
            .map(|value| finite(value, "STOI"))
            .transpose()?,
        stoi_improvement: optional_difference(enhanced.stoi, noisy.stoi, "STOI improvement")?,
        musical_noise: finite(
            enhanced.artifact_scores.musical_noise_score,
            "musical-noise score",
        )?,
        pumping: finite(enhanced.artifact_scores.pumping_score, "pumping score")?,
        transient_loss: finite(
            enhanced.artifact_scores.transient_loss_score,
            "transient-loss score",
        )?,
        phase_distortion: enhanced
            .artifact_scores
            .phase_distortion_score
            .map(|value| finite(value, "phase-distortion score"))
            .transpose()?,
    })
}

fn optional_difference(
    candidate: Option<f64>,
    baseline: Option<f64>,
    label: &str,
) -> Result<Option<f64>, String> {
    match (candidate, baseline) {
        (Some(candidate), Some(baseline)) => Ok(Some(finite(candidate - baseline, label)?)),
        _ => Ok(None),
    }
}

fn output_quality_metrics(
    reference: &Audio,
    output: &Audio,
    fingerprint: FileFingerprint,
    silence_threshold_dbfs: f64,
    dropout_window_ms: u32,
) -> Result<OutputQualityMetrics, String> {
    let frames_error = reference.frames().abs_diff(output.frames()) as u64;
    let channel_error = reference.channels().abs_diff(output.channels()) as u32;
    let sample_rate_matches = reference.sample_rate == output.sample_rate;
    let output_layout = output.channel_layout();
    let reference_layout = reference.channel_layout();
    let layout_matches = output_layout == reference_layout;
    let total_samples = output
        .frames()
        .checked_mul(output.channels())
        .ok_or_else(|| "output sample count overflows".to_string())?;
    let mut clipping_samples = 0_u64;
    let mut non_finite_samples = 0_u64;
    let mut peak = 0.0_f64;
    let mut offsets = Vec::with_capacity(output.channels());
    for channel in &output.channels {
        let mut sum = 0.0;
        for sample in channel {
            if !sample.is_finite() {
                non_finite_samples += 1;
                continue;
            }
            let magnitude = sample.abs();
            peak = peak.max(magnitude);
            if magnitude >= 1.0 {
                clipping_samples += 1;
            }
            sum += sample;
        }
        offsets.push(sum / channel.len().max(1) as f64);
    }
    let sample_peak_dbfs = amplitude_db(peak);
    let loudness = crate::loudness::measure_detailed(output).ok();
    let true_peak_dbtp = loudness
        .map(|value| value.true_peak_dbtp)
        .unwrap_or(sample_peak_dbfs);
    let integrated_lufs = loudness.map(|value| value.integrated_lufs);
    let threshold = 10_f64
        .powf(silence_threshold_dbfs / 20.0)
        .max(SILENCE_FLOOR);
    let (silence_ratio, dropout_ratio) = silence_and_dropout(output, threshold, dropout_window_ms)?;
    let dc_offset_abs = offsets
        .iter()
        .fold(0.0_f64, |value, item| value.max(item.abs()));
    let decode_integrity = frames_error == 0
        && sample_rate_matches
        && channel_error == 0
        && layout_matches
        && non_finite_samples == 0;
    Ok(OutputQualityMetrics {
        reference_sample_rate: reference.sample_rate,
        output_sample_rate: output.sample_rate,
        sample_rate_matches,
        reference_frames: reference.frames() as u64,
        output_frames: output.frames() as u64,
        duration_error_frames: frames_error,
        reference_channels: reference.channels() as u32,
        output_channels: output.channels() as u32,
        channel_mismatch: channel_error,
        reference_layout: reference_layout.to_string(),
        output_layout: output_layout.to_string(),
        layout_matches,
        clipping_samples,
        clipping_ratio: if total_samples == 0 {
            0.0
        } else {
            clipping_samples as f64 / total_samples as f64
        },
        sample_peak_dbfs: finite(sample_peak_dbfs, "sample peak")?,
        true_peak_dbtp: finite(true_peak_dbtp, "true peak")?,
        dc_offset_per_channel: offsets,
        dc_offset_abs: finite(dc_offset_abs, "absolute DC offset")?,
        silence_ratio: finite(silence_ratio, "silence ratio")?,
        dropout_ratio: finite(dropout_ratio, "dropout ratio")?,
        integrated_lufs: integrated_lufs
            .map(|value| finite(value, "integrated loudness"))
            .transpose()?,
        non_finite_samples,
        decode_integrity,
        fingerprint,
    })
}

fn amplitude_db(amplitude: f64) -> f64 {
    20.0 * amplitude.max(SILENCE_FLOOR).log10()
}

fn silence_and_dropout(
    audio: &Audio,
    threshold: f64,
    window_ms: u32,
) -> Result<(f64, f64), String> {
    let frames = audio.frames();
    if frames == 0 {
        return Err("cannot inspect silence in empty audio".into());
    }
    let silent_frames = (0..frames)
        .filter(|&frame| {
            audio
                .channels
                .iter()
                .all(|channel| channel[frame].abs() <= threshold)
        })
        .count();
    let window_frames =
        ((u64::from(audio.sample_rate) * u64::from(window_ms)) / 1_000).max(1) as usize;
    let mut silent_windows = Vec::new();
    for start in (0..frames).step_by(window_frames) {
        let end = (start + window_frames).min(frames);
        let samples = (end - start)
            .checked_mul(audio.channels())
            .ok_or_else(|| "dropout window sample count overflows".to_string())?;
        let energy: f64 = audio
            .channels
            .iter()
            .flat_map(|channel| &channel[start..end])
            .map(|sample| sample * sample)
            .sum();
        let rms = (energy / samples.max(1) as f64).sqrt();
        silent_windows.push(rms <= threshold);
    }
    let first_active = silent_windows.iter().position(|silent| !silent);
    let last_active = silent_windows.iter().rposition(|silent| !silent);
    let dropout_ratio = match (first_active, last_active) {
        (Some(first), Some(last)) if last >= first => {
            let span = &silent_windows[first..=last];
            span.iter().filter(|silent| **silent).count() as f64 / span.len() as f64
        }
        _ => 0.0,
    };
    Ok((silent_frames as f64 / frames as f64, dropout_ratio))
}

fn percentile(sorted: &[f64], quantile: f64) -> Result<f64, String> {
    if sorted.is_empty() {
        return Err("cannot calculate a percentile without samples".into());
    }
    let rank = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    finite(sorted[rank.min(sorted.len() - 1)], "performance percentile")
}

#[cfg(unix)]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let value = unsafe { usage.assume_init() }.ru_maxrss;
    let value = u64::try_from(value).ok()?;
    #[cfg(target_os = "macos")]
    return Some(value);
    #[cfg(not(target_os = "macos"))]
    Some(value.saturating_mul(1_024))
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

fn bind_listening_evidence(
    manifest: &EvaluationManifest,
    manifest_digest: Digest,
    result_path: Option<&Path>,
) -> Result<ListeningEvidence, String> {
    let policy = &manifest.policy.listening;
    let protocol_digest = policy
        .protocol
        .as_ref()
        .map(ListeningProtocol::digest)
        .transpose()?;
    if !policy.required {
        if result_path.is_some() {
            return Err(
                "--listening-result is only valid when listening evidence is required".into(),
            );
        }
        return Ok(ListeningEvidence {
            required: false,
            protocol_digest,
            result_fingerprint: None,
            listener_count: None,
            aggregate_score: None,
            accepted: true,
        });
    }
    let path = result_path.ok_or(
        "the evaluation manifest requires --listening-result; automation cannot substitute for its human protocol",
    )?;
    let (result, fingerprint) = ListeningTestResult::from_file(path)?;
    let protocol = policy
        .protocol
        .as_ref()
        .ok_or("required listening protocol is missing")?;
    if result.corpus_id != manifest.corpus_id
        || result.manifest_digest != manifest_digest
        || Some(result.protocol_digest) != protocol_digest
    {
        return Err("listening-test result does not match the evaluation manifest/protocol".into());
    }
    if result.listener_count < protocol.minimum_listeners
        || !(protocol.scale_min..=protocol.scale_max).contains(&result.aggregate_score)
    {
        return Err("listening-test result violates the pinned protocol bounds".into());
    }
    let accepted = result.accepted && result.aggregate_score >= protocol.acceptance_score;
    Ok(ListeningEvidence {
        required: true,
        protocol_digest,
        result_fingerprint: Some(fingerprint),
        listener_count: Some(result.listener_count),
        aggregate_score: Some(result.aggregate_score),
        accepted,
    })
}

fn evaluate_thresholds(
    thresholds: &[EvaluationThreshold],
    cases: &[EvaluationCaseResult],
) -> Result<Vec<ThresholdOutcome>, String> {
    thresholds
        .iter()
        .map(|threshold| {
            let observed = aggregate_metric(cases, &threshold.metric, threshold.aggregation)?;
            Ok(ThresholdOutcome {
                metric: threshold.metric.clone(),
                aggregation: threshold.aggregation,
                operator: threshold.operator,
                limit: threshold.value,
                observed,
                passed: compare_threshold(observed, threshold.operator, threshold.value),
            })
        })
        .collect()
}

fn compare_threshold(observed: f64, operator: ThresholdOperator, limit: f64) -> bool {
    match operator {
        ThresholdOperator::GreaterOrEqual => observed >= limit,
        ThresholdOperator::LessOrEqual => observed <= limit,
    }
}

fn aggregate_metric(
    cases: &[EvaluationCaseResult],
    metric: &str,
    aggregation: ThresholdAggregation,
) -> Result<f64, String> {
    if cases.is_empty() {
        return Err("cannot aggregate an empty evaluation result".into());
    }
    let values: Vec<f64> = cases
        .iter()
        .map(|case| case_metric(case, metric))
        .collect::<Result<_, _>>()?;
    let value = match aggregation {
        ThresholdAggregation::Minimum => values.iter().copied().fold(f64::INFINITY, f64::min),
        ThresholdAggregation::Maximum => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ThresholdAggregation::Mean => values.iter().sum::<f64>() / values.len() as f64,
    };
    finite(value, "aggregated evaluation metric")
}

fn case_metric(case: &EvaluationCaseResult, metric: &str) -> Result<f64, String> {
    let value = match metric {
        "objective.si-sdr-db" => case.objective.si_sdr_db,
        "objective.si-sdr-improvement-db" => case.objective.si_sdr_improvement_db,
        "objective.si-snr-db" => case.objective.si_snr_db,
        "objective.snr-db" => case.objective.snr_db,
        "objective.segmental-snr-db" => case.objective.segmental_snr_db,
        "perceptual.stoi" => case
            .objective
            .stoi
            .ok_or_else(|| format!("STOI is unavailable for evaluation case {}", case.id))?,
        "perceptual.stoi-improvement" => case.objective.stoi_improvement.ok_or_else(|| {
            format!(
                "STOI improvement is unavailable for evaluation case {}",
                case.id
            )
        })?,
        "perceptual.musical-noise" => case.objective.musical_noise,
        "perceptual.pumping" => case.objective.pumping,
        "perceptual.transient-loss" => case.objective.transient_loss,
        "perceptual.phase-distortion" => case.objective.phase_distortion.ok_or_else(|| {
            format!(
                "phase distortion is unavailable for mono evaluation case {}",
                case.id
            )
        })?,
        "output.duration-error-frames" => case.output_quality.duration_error_frames as f64,
        "output.sample-rate-mismatch" => {
            if case.output_quality.sample_rate_matches {
                0.0
            } else {
                1.0
            }
        }
        "output.channel-mismatch" => case.output_quality.channel_mismatch as f64,
        "output.clipping-ratio" => case.output_quality.clipping_ratio,
        "output.sample-peak-dbfs" => case.output_quality.sample_peak_dbfs,
        "output.true-peak-dbtp" => case.output_quality.true_peak_dbtp,
        "output.dc-offset-abs" => case.output_quality.dc_offset_abs,
        "output.silence-ratio" => case.output_quality.silence_ratio,
        "output.dropout-ratio" => case.output_quality.dropout_ratio,
        "output.integrated-lufs" => case.output_quality.integrated_lufs.ok_or_else(|| {
            format!(
                "integrated loudness is unavailable for evaluation case {}",
                case.id
            )
        })?,
        "output.non-finite-samples" => case.output_quality.non_finite_samples as f64,
        "output.decode-integrity" => {
            if case.output_quality.decode_integrity {
                1.0
            } else {
                0.0
            }
        }
        "performance.realtime-factor" => case.performance.realtime_factor,
        "performance.throughput-x" => case.performance.throughput_x,
        "performance.elapsed-ms" => case.performance.median_elapsed_ms,
        "performance.peak-rss-bytes" => case
            .performance
            .peak_rss_bytes
            .ok_or_else(|| format!("peak RSS is unavailable for evaluation case {}", case.id))?
            as f64,
        _ => return Err(format!("unknown evaluation metric: {metric}")),
    };
    finite(value, metric)
}

pub fn write_signed_evaluation_result(
    path: impl AsRef<Path>,
    result: &SignedEvaluationResult,
) -> Result<(), String> {
    use std::io::Write as _;
    result.validate_structure()?;
    let mut bytes = serde_json::to_vec_pretty(result)
        .map_err(|error| format!("serialize signed evaluation result: {error}"))?;
    bytes.push(b'\n');
    if bytes.len() as u64 >= MAX_JSON_BYTES {
        return Err("signed evaluation result exceeds its JSON size limit".into());
    }
    let path = path.as_ref();
    if path.exists() {
        return Err(format!(
            "signed evaluation result already exists: {}",
            path.display()
        ));
    }
    let mut output = crate::AtomicOutput::new(path)?;
    output
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("write signed evaluation result {}: {error}", path.display()))?;
    output
        .file_mut()
        .sync_data()
        .map_err(|error| format!("sync signed evaluation result {}: {error}", path.display()))?;
    output.commit(crate::CommitMode::NoClobber)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegressionOutcome {
    pub metric: String,
    pub aggregation: ThresholdAggregation,
    pub baseline: f64,
    pub candidate: f64,
    pub regression: f64,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationComparisonReport {
    pub schema: String,
    pub schema_version: u32,
    pub manifest_digest: Digest,
    pub baseline_key_id: String,
    pub candidate_key_id: String,
    pub baseline_version: String,
    pub candidate_version: String,
    pub environment_comparable: bool,
    pub regressions: Vec<RegressionOutcome>,
    pub passed: bool,
}

/// Authenticate and compare two results.  Performance comparisons fail closed
/// unless their recorded hardware, runtime, recipe, and timing scope match.
pub fn compare_evaluation_results(
    baseline: &SignedEvaluationResult,
    baseline_key: &ReceiptPublicKey,
    candidate: &SignedEvaluationResult,
    candidate_key: &ReceiptPublicKey,
) -> Result<EvaluationComparisonReport, String> {
    baseline.verify_signature(baseline_key)?;
    candidate.verify_signature(candidate_key)?;
    if baseline.payload.manifest_digest != candidate.payload.manifest_digest
        || baseline.payload.corpus_id != candidate.payload.corpus_id
        || baseline.payload.corpus_version != candidate.payload.corpus_version
        || baseline.payload.regression_tolerances != candidate.payload.regression_tolerances
    {
        return Err(
            "evaluation results use different corpus manifests or regression policies".into(),
        );
    }
    if baseline.payload.environment != candidate.payload.environment {
        return Err(
            "evaluation environments are incomparable; hardware, runtime, recipe, model, and timing scope must match"
                .into(),
        );
    }
    if baseline.payload.cases.len() != candidate.payload.cases.len()
        || baseline
            .payload
            .cases
            .iter()
            .zip(&candidate.payload.cases)
            .any(|(left, right)| {
                left.id != right.id || left.clean != right.clean || left.noisy != right.noisy
            })
    {
        return Err("evaluation results do not cover the same pinned cases".into());
    }
    let mut regressions = Vec::with_capacity(candidate.payload.regression_tolerances.len());
    for tolerance in &candidate.payload.regression_tolerances {
        let baseline_value = aggregate_metric(
            &baseline.payload.cases,
            &tolerance.metric,
            tolerance.aggregation,
        )?;
        let candidate_value = aggregate_metric(
            &candidate.payload.cases,
            &tolerance.metric,
            tolerance.aggregation,
        )?;
        let regression = match tolerance.direction {
            RegressionDirection::HigherIsBetter => baseline_value - candidate_value,
            RegressionDirection::LowerIsBetter => candidate_value - baseline_value,
        };
        let regression = finite(regression, "evaluation regression")?;
        regressions.push(RegressionOutcome {
            metric: tolerance.metric.clone(),
            aggregation: tolerance.aggregation,
            baseline: baseline_value,
            candidate: candidate_value,
            regression,
            limit: tolerance.max_regression,
            passed: regression <= tolerance.max_regression,
        });
    }
    let passed = candidate.payload.accepted && regressions.iter().all(|value| value.passed);
    Ok(EvaluationComparisonReport {
        schema: EVALUATION_COMPARISON_SCHEMA.into(),
        schema_version: EVALUATION_SCHEMA_VERSION,
        manifest_digest: candidate.payload.manifest_digest,
        baseline_key_id: baseline.signature.key_id.clone(),
        candidate_key_id: candidate.signature.key_id.clone(),
        baseline_version: baseline.payload.denoize_version.clone(),
        candidate_version: candidate.payload.denoize_version.clone(),
        environment_comparable: true,
        regressions,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn digest(byte: u8) -> Digest {
        Digest::from_bytes([byte; 32])
    }

    fn fixture_audio(noisy: bool) -> Audio {
        let sample_rate = 16_000;
        let frames = sample_rate as usize;
        let channel = (0..frames)
            .map(|index| {
                let time = index as f64 / sample_rate as f64;
                let clean = 0.35 * (2.0 * std::f64::consts::PI * 440.0 * time).sin()
                    + 0.12 * (2.0 * std::f64::consts::PI * 880.0 * time).sin();
                let noise = if noisy {
                    0.035 * (((index * 7_919) % 997) as f64 / 498.0 - 1.0)
                } else {
                    0.0
                };
                clean + noise
            })
            .collect();
        Audio {
            sample_rate,
            channels: vec![channel],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn artifact(path: &str, fingerprint: FileFingerprint) -> EvaluationArtifact {
        EvaluationArtifact {
            path: path.into(),
            fingerprint,
            license: EvaluationLicense {
                spdx_id: "CC0-1.0".into(),
                name: "Creative Commons Zero v1.0 Universal".into(),
                url: "https://creativecommons.org/publicdomain/zero/1.0/".into(),
            },
            source: EvaluationSource {
                uri: "https://example.invalid/denoize-evaluation-fixture".into(),
                revision: "sha256-0123456789abcdef".into(),
            },
            preparation: SignalPreparation {
                description: "Deterministic additive-noise test signal".into(),
                tool: "denoize-evaluation-test-generator".into(),
                tool_version: "1.0.0".into(),
                parameters_digest: digest(9),
            },
        }
    }

    fn manifest(clean: FileFingerprint, noisy: FileFingerprint) -> EvaluationManifest {
        EvaluationManifest {
            schema: EVALUATION_CORPUS_SCHEMA.into(),
            schema_version: EVALUATION_SCHEMA_VERSION,
            corpus_id: "cc0-synthetic-speech".into(),
            corpus_version: "1.0.0".into(),
            title: "CC0 deterministic evaluation fixture".into(),
            cases: vec![EvaluationCase {
                id: "tone-noise-001".into(),
                clean: artifact("audio/clean.wav", clean),
                noisy: artifact("audio/noisy.wav", noisy),
                tags: vec!["synthetic".into()],
            }],
            recipe: EvaluationRecipe {
                backend: "classical".into(),
                preset: "speech".into(),
                accelerator: "cpu".into(),
                deterministic: true,
                seed: None,
                channel_mode: "independent".into(),
                sgmse_profile: "balanced".into(),
                model: None,
                model_sample_rate: None,
            },
            policy: EvaluationPolicy {
                warmup_runs: 0,
                measured_runs: 1,
                silence_threshold_dbfs: -90.0,
                dropout_window_ms: 20,
                thresholds: vec![
                    EvaluationThreshold {
                        metric: "objective.si-sdr-improvement-db".into(),
                        aggregation: ThresholdAggregation::Minimum,
                        operator: ThresholdOperator::GreaterOrEqual,
                        value: -200.0,
                    },
                    EvaluationThreshold {
                        metric: "perceptual.musical-noise".into(),
                        aggregation: ThresholdAggregation::Maximum,
                        operator: ThresholdOperator::LessOrEqual,
                        value: 1.0,
                    },
                    EvaluationThreshold {
                        metric: "output.decode-integrity".into(),
                        aggregation: ThresholdAggregation::Minimum,
                        operator: ThresholdOperator::GreaterOrEqual,
                        value: 1.0,
                    },
                    EvaluationThreshold {
                        metric: "performance.realtime-factor".into(),
                        aggregation: ThresholdAggregation::Maximum,
                        operator: ThresholdOperator::LessOrEqual,
                        value: 10_000.0,
                    },
                ],
                regression_tolerances: vec![RegressionTolerance {
                    metric: "objective.si-sdr-improvement-db".into(),
                    aggregation: ThresholdAggregation::Minimum,
                    direction: RegressionDirection::HigherIsBetter,
                    max_regression: 0.0,
                }],
                listening: ListeningPolicy {
                    required: false,
                    rationale: "Synthetic contract fixture; human judgment is not claimed".into(),
                    protocol: None,
                },
            },
        }
    }

    fn write_fixture() -> (tempfile::TempDir, EvaluationManifest) {
        let directory = tempfile::tempdir().unwrap();
        let audio_dir = directory.path().join("audio");
        std::fs::create_dir(&audio_dir).unwrap();
        let clean_path = audio_dir.join("clean.wav");
        let noisy_path = audio_dir.join("noisy.wav");
        crate::write_wav(&clean_path, &fixture_audio(false)).unwrap();
        crate::write_wav(&noisy_path, &fixture_audio(true)).unwrap();
        let clean = batch_resume::fingerprint_file(&clean_path).unwrap();
        let noisy = batch_resume::fingerprint_file(&noisy_path).unwrap();
        (directory, manifest(clean, noisy))
    }

    #[test]
    fn corpus_validation_binds_provenance_hashes_and_geometry() {
        let (directory, manifest) = write_fixture();
        let report = validate_evaluation_corpus(&manifest, directory.path()).unwrap();
        assert_eq!(report.cases, 1);
        assert_eq!(report.corpus_id, manifest.corpus_id);
        assert!(report.total_audio_seconds > 0.9);

        let mut wrong_hash = manifest.clone();
        wrong_hash.cases[0].noisy.fingerprint.digest = digest(3);
        assert!(validate_evaluation_corpus(&wrong_hash, directory.path())
            .unwrap_err()
            .contains("fingerprint mismatch"));

        let mismatched_path = directory.path().join("audio/noisy-8khz.wav");
        let mut mismatched_audio = fixture_audio(true);
        mismatched_audio.sample_rate = 8_000;
        crate::write_wav(&mismatched_path, &mismatched_audio).unwrap();
        let mut mismatched = manifest.clone();
        mismatched.cases[0].noisy.path = "audio/noisy-8khz.wav".into();
        mismatched.cases[0].noisy.fingerprint =
            batch_resume::fingerprint_file(&mismatched_path).unwrap();
        assert!(validate_evaluation_corpus(&mismatched, directory.path())
            .unwrap_err()
            .contains("sample rates differ"));

        let mut floating = manifest.clone();
        floating.cases[0].clean.source.revision = "latest".into();
        assert!(floating
            .validate()
            .unwrap_err()
            .contains("immutable revision"));

        let mut branch = manifest.clone();
        branch.cases[0].clean.source.revision = "refs/heads/main".into();
        assert!(branch
            .validate()
            .unwrap_err()
            .contains("immutable revision"));

        let mut invalid_uri = manifest.clone();
        invalid_uri.cases[0].clean.source.uri = "1:not-a-uri-scheme".into();
        assert!(invalid_uri.validate().unwrap_err().contains("absolute URI"));

        let mut nondeterministic = manifest.clone();
        nondeterministic.recipe.deterministic = false;
        assert!(nondeterministic
            .validate()
            .unwrap_err()
            .contains("deterministic processing"));

        let mut irrelevant_seed = manifest.clone();
        irrelevant_seed.recipe.seed = Some(7);
        assert!(irrelevant_seed
            .validate()
            .unwrap_err()
            .contains("only SGMSE"));

        let mut incomplete = manifest.clone();
        incomplete
            .policy
            .thresholds
            .retain(|value| !value.metric.starts_with("performance."));
        assert!(incomplete
            .validate()
            .unwrap_err()
            .contains("performance metric"));

        let mut external_backend = manifest;
        external_backend.recipe.backend = "mpsenet".into();
        external_backend.recipe.model = Some(artifact(
            "models/mpsenet.onnx",
            external_backend.cases[0].clean.fingerprint,
        ));
        external_backend.recipe.model_sample_rate = Some(16_000);
        external_backend.validate().unwrap();
    }

    #[test]
    fn run_sign_verify_and_regression_compare_are_end_to_end() {
        let (directory, manifest) = write_fixture();
        let (secret, public) = crate::generate_receipt_keypair().unwrap();
        let result = run_evaluation(&manifest, directory.path(), &secret, None).unwrap();
        assert!(result.payload.accepted);
        assert!(result.payload.cases[0].output_quality.decode_integrity);
        assert_eq!(result.payload.threshold_outcomes.len(), 4);

        let report = verify_evaluation_result(&result, &public, Some(&manifest)).unwrap();
        assert!(report.accepted);
        let comparison = compare_evaluation_results(&result, &public, &result, &public).unwrap();
        assert!(comparison.passed);
        assert_eq!(comparison.regressions[0].regression, 0.0);

        let encoded = serde_json::to_vec_pretty(&result).unwrap();
        let parsed: SignedEvaluationResult = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            serde_json::to_vec(&result.payload).unwrap(),
            serde_json::to_vec(&parsed.payload).unwrap(),
            "evaluation payload must retain its canonical signing bytes across JSON"
        );
        parsed.verify_signature(&public).unwrap();

        let mut inconsistent_threshold = result.clone();
        inconsistent_threshold.payload.threshold_outcomes[0].observed += 1.0;
        assert!(inconsistent_threshold
            .validate_structure()
            .unwrap_err()
            .contains("signed case measurements"));

        let mut inconsistent_timing = result.clone();
        inconsistent_timing.payload.cases[0]
            .performance
            .median_elapsed_ms += 1.0;
        assert!(inconsistent_timing
            .validate_structure()
            .unwrap_err()
            .contains("performance measurements"));

        let mut tampered = result;
        tampered.payload.denoize_version.push_str("-tampered");
        assert!(tampered.verify_signature(&public).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn corpus_paths_reject_symlink_traversal_even_when_hash_matches() {
        use std::os::unix::fs::symlink;

        let (directory, mut manifest) = write_fixture();
        let target = directory.path().join("audio/clean.wav");
        let link = directory.path().join("audio/clean-link.wav");
        symlink(&target, &link).unwrap();
        manifest.cases[0].clean.path = "audio/clean-link.wav".into();
        let error = validate_evaluation_corpus(&manifest, directory.path()).unwrap_err();
        assert!(error.contains("symlink"), "{error}");
    }

    #[test]
    fn required_listening_protocol_cannot_be_silently_automated() {
        let (directory, mut manifest) = write_fixture();
        manifest.policy.listening = ListeningPolicy {
            required: true,
            rationale: "Human preference is a release criterion".into(),
            protocol: Some(ListeningProtocol {
                protocol_id: "mushra-speech".into(),
                revision: "1.0.0".into(),
                method: "Double-blind randomized MUSHRA".into(),
                instructions_uri: "https://example.invalid/protocol/1.0.0".into(),
                instructions_digest: digest(7),
                scale_min: 0.0,
                scale_max: 100.0,
                minimum_listeners: 8,
                acceptance_score: 70.0,
            }),
        };
        let (secret, _) = crate::generate_receipt_keypair().unwrap();
        let error = run_evaluation(&manifest, directory.path(), &secret, None).unwrap_err();
        assert!(error.contains("automation cannot substitute"), "{error}");

        let protocol_digest = manifest
            .policy
            .listening
            .protocol
            .as_ref()
            .unwrap()
            .digest()
            .unwrap();
        let listening = ListeningTestResult {
            schema: LISTENING_RESULT_SCHEMA.into(),
            schema_version: EVALUATION_SCHEMA_VERSION,
            corpus_id: manifest.corpus_id.clone(),
            manifest_digest: manifest.digest().unwrap(),
            protocol_digest,
            listener_count: 8,
            aggregate_score: 82.5,
            accepted: true,
        };
        let listening_path = directory.path().join("listening-result.json");
        std::fs::write(
            &listening_path,
            serde_json::to_vec_pretty(&listening).unwrap(),
        )
        .unwrap();
        let result =
            run_evaluation(&manifest, directory.path(), &secret, Some(&listening_path)).unwrap();
        assert!(result.payload.accepted);
        assert_eq!(result.payload.listening.listener_count, Some(8));
        assert_eq!(result.payload.listening.aggregate_score, Some(82.5));
        assert_eq!(
            result.payload.listening.result_fingerprint,
            Some(batch_resume::fingerprint_file(&listening_path).unwrap())
        );
    }
}
