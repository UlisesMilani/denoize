//! Fail-closed offline target-speaker extraction and promotion evidence.
//!
//! A target-speaker candidate is never silently treated as ordinary denoising
//! output. The signed package must expose mixture, enrollment, extracted-audio,
//! and calibrated three-state presence tensors through the dedicated adapter.
//! Separately signed evaluation evidence must bind the exact package and pass
//! protected target-present and target-absent strata. At runtime, audio is
//! published only for a confidently present target whose candidate also passes
//! bounded signal-safety checks.

use crate::audio::{estimate_audio_memory_bytes, Audio};
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
#[cfg(feature = "onnx")]
use crate::{
    AcceleratorPreference, AcceleratorSelection, Backend, BackendOptions, RuntimeModelPackage,
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "onnx")]
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::path::Path;
#[cfg(feature = "onnx")]
use zeroize::{Zeroize, Zeroizing};

pub const TARGET_SPEAKER_REPORT_SCHEMA: &str = "denoize-target-speaker-report-v1";
pub const TARGET_SPEAKER_PROMOTION_EVIDENCE_SCHEMA: &str =
    "denoize-target-speaker-promotion-evidence-v1";
pub const TARGET_SPEAKER_SCHEMA_VERSION: u32 = 1;
pub const MAX_TARGET_SPEAKER_EVIDENCE_STRATA: usize = 256;
pub const MAX_TARGET_SPEAKER_EVIDENCE_METRICS: usize = 64;
pub const MIN_TARGET_SPEAKER_ENROLLMENT_MILLIS: u64 = 500;
pub const MAX_TARGET_SPEAKER_ENROLLMENT_MILLIS: u64 = 30_000;
pub const MAX_TARGET_SPEAKER_MIXTURE_SECONDS: u64 = 3_600;

#[cfg(feature = "onnx")]
const MAX_CHANNELS: usize = 64;
const MAX_EVIDENCE_JSON_BYTES: u64 = 16 * 1024 * 1024;
const PROMOTION_SIGNATURE_DOMAIN: &[u8] = b"denoize-target-speaker-promotion-evidence-v1";
#[cfg(feature = "onnx")]
const MIXTURE_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-target-speaker-mixture-pcm-v1\0";
#[cfg(feature = "onnx")]
const OUTPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-target-speaker-output-pcm-v1\0";
#[cfg(feature = "onnx")]
const SILENCE_FLOOR: f64 = 1e-12;

const REQUIRED_STRATA: &[(&str, TargetSpeakerStratumKind)] = &[
    ("channel-mismatch", TargetSpeakerStratumKind::TargetPresent),
    ("child-speaker", TargetSpeakerStratumKind::TargetPresent),
    ("code-switching", TargetSpeakerStratumKind::TargetPresent),
    ("codec-enrollment", TargetSpeakerStratumKind::TargetPresent),
    ("different-sex", TargetSpeakerStratumKind::TargetPresent),
    ("many-interferers", TargetSpeakerStratumKind::TargetPresent),
    ("noisy-enrollment", TargetSpeakerStratumKind::TargetPresent),
    ("one-interferer", TargetSpeakerStratumKind::TargetPresent),
    (
        "real-t-conversation",
        TargetSpeakerStratumKind::TargetPresent,
    ),
    (
        "reverberant-enrollment",
        TargetSpeakerStratumKind::TargetPresent,
    ),
    ("same-sex", TargetSpeakerStratumKind::TargetPresent),
    ("same-words", TargetSpeakerStratumKind::TargetPresent),
    ("similar-voices", TargetSpeakerStratumKind::TargetPresent),
    ("singing", TargetSpeakerStratumKind::TargetPresent),
    ("speech-absent", TargetSpeakerStratumKind::TargetAbsent),
    ("target-absent", TargetSpeakerStratumKind::TargetAbsent),
    (
        "target-absent-same-words",
        TargetSpeakerStratumKind::TargetAbsent,
    ),
    (
        "target-absent-similar-interferer",
        TargetSpeakerStratumKind::TargetAbsent,
    ),
    (
        "target-present-clean",
        TargetSpeakerStratumKind::TargetPresent,
    ),
    ("ts-superb", TargetSpeakerStratumKind::TargetPresent),
    ("unseen-domain", TargetSpeakerStratumKind::TargetPresent),
    ("whisper", TargetSpeakerStratumKind::TargetPresent),
];

const PRESENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("content.target-word-error-rate", 0.35),
    MetricPolicy::at_least("extraction.si-sdr-improvement-db", 3.0),
    MetricPolicy::at_most("interferer.speaker-similarity", 0.30),
    MetricPolicy::at_most("interferer.word-leakage-rate", 0.02),
    MetricPolicy::at_most("output.duration-error-frames", 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0),
    MetricPolicy::at_least("perceptual.dnsmos-p808", 3.0),
    MetricPolicy::at_least("presence.recall", 0.95),
    MetricPolicy::at_least("speaker.target-similarity", 0.70),
];

const ABSENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("interferer.speaker-similarity", 0.30),
    MetricPolicy::at_most("interferer.word-leakage-rate", 0.01),
    MetricPolicy::at_most("output.duration-error-frames", 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0),
    MetricPolicy::at_most("output.rms-dbfs", -60.0),
    MetricPolicy::at_most("presence.false-positive-rate", 0.01),
];

#[derive(Clone, Copy)]
struct MetricPolicy {
    name: &'static str,
    operator: TargetSpeakerMetricOperator,
    hard_limit: f64,
}

impl MetricPolicy {
    const fn at_least(name: &'static str, hard_limit: f64) -> Self {
        Self {
            name,
            operator: TargetSpeakerMetricOperator::GreaterOrEqual,
            hard_limit,
        }
    }

    const fn at_most(name: &'static str, hard_limit: f64) -> Self {
        Self {
            name,
            operator: TargetSpeakerMetricOperator::LessOrEqual,
            hard_limit,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSpeakerStratumKind {
    TargetPresent,
    TargetAbsent,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSpeakerMetricOperator {
    GreaterOrEqual,
    LessOrEqual,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerMetricOutcome {
    pub metric: String,
    pub value: f64,
    pub operator: TargetSpeakerMetricOperator,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerStratumEvidence {
    pub id: String,
    pub kind: TargetSpeakerStratumKind,
    pub cases: u32,
    pub metrics: Vec<TargetSpeakerMetricOutcome>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub model_package_sha256: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub checkpoint_sha256: String,
    pub corpus_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub real_t_result_sha256: String,
    pub ts_superb_result_sha256: String,
    pub strata: Vec<TargetSpeakerStratumEvidence>,
    pub target_speaker_count: u32,
    pub interferer_speaker_count: u32,
    pub language_count: u32,
    pub presence_expected_calibration_error: f64,
    pub presence_expected_calibration_error_limit: f64,
    pub minimum_listeners: u32,
    pub listener_count: u32,
    pub listener_preference: f64,
    pub listener_preference_limit: f64,
    pub accepted: bool,
}

impl TargetSpeakerPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        for (label, digest) in [
            ("model package", self.model_package_sha256.as_str()),
            ("source", self.source_sha256.as_str()),
            ("checkpoint", self.checkpoint_sha256.as_str()),
            ("corpus manifest", self.corpus_manifest_sha256.as_str()),
            ("evaluation result", self.evaluation_result_sha256.as_str()),
            ("REAL-T result", self.real_t_result_sha256.as_str()),
            ("TS-SUPERB result", self.ts_superb_result_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
        validate_identifier("source revision", &self.source_revision)?;
        if self.completed_at_unix_seconds > (1_u64 << 53) - 1 {
            return Err(
                "target-speaker evidence timestamp exceeds the JSON safe-integer limit".into(),
            );
        }
        if self.strata.is_empty() || self.strata.len() > MAX_TARGET_SPEAKER_EVIDENCE_STRATA {
            return Err(format!(
                "target-speaker evidence must contain 1..={MAX_TARGET_SPEAKER_EVIDENCE_STRATA} strata"
            ));
        }
        let required: BTreeMap<_, _> = REQUIRED_STRATA.iter().copied().collect();
        let mut observed_strata = BTreeSet::new();
        let mut previous = None;
        let mut all_metrics_passed = true;
        for stratum in &self.strata {
            validate_identifier("target-speaker evidence stratum", &stratum.id)?;
            if previous.is_some_and(|value: &str| value >= stratum.id.as_str()) {
                return Err(
                    "target-speaker evidence strata must be unique and strictly sorted".into(),
                );
            }
            previous = Some(&stratum.id);
            observed_strata.insert(stratum.id.as_str());
            if required
                .get(stratum.id.as_str())
                .is_some_and(|expected| *expected != stratum.kind)
            {
                return Err(format!(
                    "target-speaker evidence stratum {} has the wrong presence kind",
                    stratum.id
                ));
            }
            if !(10..=1_000_000).contains(&stratum.cases) {
                return Err("target-speaker evidence stratum cases must be in 10..=1000000".into());
            }
            if stratum.metrics.is_empty()
                || stratum.metrics.len() > MAX_TARGET_SPEAKER_EVIDENCE_METRICS
            {
                return Err(format!(
                    "target-speaker evidence stratum metrics must be in 1..={MAX_TARGET_SPEAKER_EVIDENCE_METRICS}"
                ));
            }
            let policies = match stratum.kind {
                TargetSpeakerStratumKind::TargetPresent => PRESENT_METRICS,
                TargetSpeakerStratumKind::TargetAbsent => ABSENT_METRICS,
            };
            let policy_by_name: BTreeMap<_, _> = policies
                .iter()
                .map(|policy| (policy.name, policy))
                .collect();
            let mut observed_metrics = BTreeSet::new();
            let mut previous_metric = None;
            for metric in &stratum.metrics {
                validate_identifier("target-speaker evidence metric", &metric.metric)?;
                if previous_metric.is_some_and(|value: &str| value >= metric.metric.as_str()) {
                    return Err(
                        "target-speaker evidence metrics must be unique and strictly sorted".into(),
                    );
                }
                previous_metric = Some(&metric.metric);
                observed_metrics.insert(metric.metric.as_str());
                if !metric.value.is_finite() || !metric.limit.is_finite() {
                    return Err("target-speaker evidence metric values must be finite".into());
                }
                let expected = match metric.operator {
                    TargetSpeakerMetricOperator::GreaterOrEqual => metric.value >= metric.limit,
                    TargetSpeakerMetricOperator::LessOrEqual => metric.value <= metric.limit,
                };
                if metric.passed != expected {
                    return Err(format!(
                        "target-speaker evidence metric {} has an inconsistent passed flag",
                        metric.metric
                    ));
                }
                if let Some(policy) = policy_by_name.get(metric.metric.as_str()) {
                    validate_metric_policy(metric, policy)?;
                }
                all_metrics_passed &= metric.passed;
            }
            for policy in policies {
                if !observed_metrics.contains(policy.name) {
                    return Err(format!(
                        "target-speaker evidence stratum {} omits required metric {}",
                        stratum.id, policy.name
                    ));
                }
            }
        }
        for (id, _) in REQUIRED_STRATA {
            if !observed_strata.contains(id) {
                return Err(format!(
                    "target-speaker evidence omits required stratum {id}"
                ));
            }
        }
        if self.target_speaker_count < 100
            || self.target_speaker_count > 1_000_000
            || self.interferer_speaker_count < 100
            || self.interferer_speaker_count > 1_000_000
            || !(2..=1_000).contains(&self.language_count)
        {
            return Err(
                "target-speaker evidence requires at least 100 target and interferer speakers and two languages"
                    .into(),
            );
        }
        if !self.presence_expected_calibration_error.is_finite()
            || !self.presence_expected_calibration_error_limit.is_finite()
            || !(0.0..=1.0).contains(&self.presence_expected_calibration_error)
            || !(0.0..=0.05).contains(&self.presence_expected_calibration_error_limit)
        {
            return Err("target-speaker evidence presence calibration values are invalid".into());
        }
        if self.minimum_listeners < 20
            || self.minimum_listeners > 100_000
            || self.listener_count < self.minimum_listeners
            || self.listener_count > 100_000
            || !self.listener_preference.is_finite()
            || !self.listener_preference_limit.is_finite()
            || !(0.0..=1.0).contains(&self.listener_preference)
            || !(0.5..=1.0).contains(&self.listener_preference_limit)
        {
            return Err("target-speaker evidence listening values are invalid".into());
        }
        let expected_accepted = all_metrics_passed
            && self.presence_expected_calibration_error
                <= self.presence_expected_calibration_error_limit
            && self.listener_count >= self.minimum_listeners
            && self.listener_preference >= self.listener_preference_limit;
        if self.accepted != expected_accepted {
            return Err("target-speaker evidence accepted flag is inconsistent".into());
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize target-speaker evidence payload: {error}"))?;
        if bytes.len() as u64 >= MAX_EVIDENCE_JSON_BYTES {
            return Err("target-speaker evidence payload exceeds the bounded JSON limit".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedTargetSpeakerPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: TargetSpeakerPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedTargetSpeakerPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) =
            crate::input::open_regular_file(path, "target-speaker promotion evidence")?;
        if length >= MAX_EVIDENCE_JSON_BYTES {
            return Err(format!(
                "target-speaker promotion evidence {} exceeds the {MAX_EVIDENCE_JSON_BYTES}-byte limit",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve target-speaker evidence JSON".to_string())?;
        file.take(MAX_EVIDENCE_JSON_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read target-speaker promotion evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("target-speaker promotion evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse target-speaker promotion evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != TARGET_SPEAKER_PROMOTION_EVIDENCE_SCHEMA
            || self.schema_version != TARGET_SPEAKER_SCHEMA_VERSION
        {
            return Err("unsupported target-speaker promotion evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("target-speaker promotion evidence signature must use ed25519".into());
        }
        validate_sha256("evidence key ID", &self.signature.key_id)?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize target-speaker promotion evidence: {error}"))?;
        if bytes.len() as u64 >= MAX_EVIDENCE_JSON_BYTES {
            return Err("target-speaker promotion evidence exceeds the bounded JSON limit".into());
        }
        Ok(())
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload).map_err(|error| {
            format!("serialize target-speaker evidence for verification: {error}")
        })?;
        key.verify_domain_document(
            PROMOTION_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "target-speaker promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize target-speaker promotion evidence: {error}"))
    }
}

pub fn sign_target_speaker_promotion_evidence(
    payload: TargetSpeakerPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedTargetSpeakerPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize target-speaker evidence for signing: {error}"))?;
    let signature = key.sign_domain_document(
        PROMOTION_SIGNATURE_DOMAIN,
        &document,
        "target-speaker promotion evidence",
    )?;
    let evidence = SignedTargetSpeakerPromotionEvidence {
        schema: TARGET_SPEAKER_PROMOTION_EVIDENCE_SCHEMA.into(),
        schema_version: TARGET_SPEAKER_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSpeakerPresence {
    Present,
    Absent,
    Uncertain,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSpeakerDecision {
    AcceptedPresent,
    WithheldAbsent,
    WithheldUncertain,
    WithheldSafetyGate,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerExtractionConfig {
    pub minimum_present_probability: f64,
    pub minimum_absent_probability: f64,
    pub maximum_energy_gain_db: f64,
    pub maximum_peak_gain_db: f64,
    pub maximum_new_clipping_ratio: f64,
}

impl Default for TargetSpeakerExtractionConfig {
    fn default() -> Self {
        Self {
            minimum_present_probability: 0.90,
            minimum_absent_probability: 0.90,
            maximum_energy_gain_db: 3.0,
            maximum_peak_gain_db: 3.0,
            maximum_new_clipping_ratio: 0.0001,
        }
    }
}

impl TargetSpeakerExtractionConfig {
    pub fn validate(&self) -> Result<(), String> {
        validate_range(
            "minimum_present_probability",
            self.minimum_present_probability,
            0.5,
            1.0,
        )?;
        validate_range(
            "minimum_absent_probability",
            self.minimum_absent_probability,
            0.5,
            1.0,
        )?;
        validate_range(
            "maximum_energy_gain_db",
            self.maximum_energy_gain_db,
            0.0,
            12.0,
        )?;
        validate_range("maximum_peak_gain_db", self.maximum_peak_gain_db, 0.0, 12.0)?;
        validate_range(
            "maximum_new_clipping_ratio",
            self.maximum_new_clipping_ratio,
            0.0,
            0.01,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerModelIdentity {
    pub package_sha256: String,
    pub public_key_sha256: String,
    pub package_id: String,
    pub package_revision: String,
    pub precision_profile: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub source_license_spdx: String,
    pub checkpoint_sha256: String,
    pub checkpoint_license_spdx: String,
    pub accelerator: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerEvidenceIdentity {
    pub signing_key_id: String,
    pub corpus_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub real_t_result_sha256: String,
    pub ts_superb_result_sha256: String,
    pub strata: u32,
    pub target_speakers: u32,
    pub interferer_speakers: u32,
    pub languages: u32,
    pub accepted: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerEnrollmentSummary {
    pub input_sample_rate: u32,
    pub input_channels: usize,
    pub input_frames: usize,
    pub model_sample_rate: u32,
    pub model_samples: usize,
    pub mixdown_policy: String,
    pub raw_audio_retained: bool,
    pub embedding_retained: bool,
    pub digest_recorded: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerPresenceAssessment {
    pub state: TargetSpeakerPresence,
    pub absent_probability: f64,
    pub uncertain_probability: f64,
    pub present_probability: f64,
    pub minimum_absent_probability: f64,
    pub minimum_present_probability: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSpeakerSafetyGateKind {
    Geometry,
    FiniteNormalizedSamples,
    EnergyGain,
    PeakGain,
    NewClipping,
    TargetPresence,
    PromotionEvidence,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerSafetyGate {
    pub kind: TargetSpeakerSafetyGateKind,
    pub observed: f64,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerSafetyMeasurements {
    pub mixture_rms_dbfs: f64,
    pub candidate_rms_dbfs: f64,
    pub mixture_peak_dbfs: f64,
    pub candidate_peak_dbfs: f64,
    pub energy_delta_db: f64,
    pub mixture_clipping_ratio: f64,
    pub candidate_clipping_ratio: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpeakerExtractionReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub network_accessed: bool,
    pub deterministic: bool,
    pub model: TargetSpeakerModelIdentity,
    pub promotion_evidence: TargetSpeakerEvidenceIdentity,
    pub decision: TargetSpeakerDecision,
    pub model_invoked: bool,
    pub candidate_accepted: bool,
    pub output_published: bool,
    pub candidate_retained: bool,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_frames: usize,
    pub output_channels: usize,
    pub output_frames: Option<usize>,
    pub mixture_mixdown_policy: String,
    pub mixture_pcm_sha256: String,
    pub candidate_pcm_sha256: Option<String>,
    pub output_pcm_sha256: Option<String>,
    pub enrollment: TargetSpeakerEnrollmentSummary,
    pub presence: TargetSpeakerPresenceAssessment,
    pub measurements: TargetSpeakerSafetyMeasurements,
    pub safety_gates: Vec<TargetSpeakerSafetyGate>,
    pub runtime_speaker_identity_verified: bool,
    pub interferer_leakage_measured_at_runtime: bool,
    pub limitations: Vec<String>,
    pub warnings: Vec<String>,
}

impl TargetSpeakerExtractionReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|error| format!("serialize target-speaker extraction report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize target-speaker extraction report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct TargetSpeakerExtractionResult {
    /// `None` means publication was intentionally withheld. A caller must not
    /// substitute the mixture or the unverified candidate.
    pub audio: Option<Audio>,
    pub report: TargetSpeakerExtractionReport,
}

/// Conservative decoded mixture, enrollment, resampling, model input/output,
/// and report allowance. Signed model session resources are admitted
/// separately.
pub fn estimate_target_speaker_memory_bytes(mixture: &Audio, enrollment: &Audio) -> u64 {
    estimate_audio_memory_bytes(mixture)
        .saturating_mul(7)
        .saturating_add(estimate_audio_memory_bytes(enrollment).saturating_mul(5))
        .max(1024 * 1024)
}

#[cfg(feature = "onnx")]
pub struct TargetSpeakerSession {
    package: RuntimeModelPackage,
    model: crate::backend::target_speaker::TargetSpeakerModel,
    accelerator: AcceleratorSelection,
    evidence: TargetSpeakerEvidenceIdentity,
}

#[cfg(feature = "onnx")]
impl std::fmt::Debug for TargetSpeakerSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TargetSpeakerSession")
            .field("package_sha256", &self.package.package_sha256())
            .field("accelerator", &self.accelerator)
            .field("evidence", &self.evidence)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "onnx")]
impl TargetSpeakerSession {
    /// Authenticate evidence, bind it to the package, resolve the runtime, and
    /// validate the graph plus numerical vectors before user audio is decoded.
    pub fn prepare(
        package: RuntimeModelPackage,
        evidence: &SignedTargetSpeakerPromotionEvidence,
        evidence_key: &ReceiptPublicKey,
        requested: AcceleratorPreference,
    ) -> Result<Self, String> {
        evidence.verify_signature(evidence_key)?;
        if !evidence.payload.accepted {
            return Err(
                "target-speaker promotion evidence is authentic but does not pass promotion gates"
                    .into(),
            );
        }
        let manifest = package
            .manifest_v2()
            .ok_or("target-speaker extraction rejects runtime model package v1")?;
        for (label, observed, expected) in [
            (
                "model package SHA-256",
                evidence.payload.model_package_sha256.as_str(),
                package.package_sha256(),
            ),
            (
                "source revision",
                evidence.payload.source_revision.as_str(),
                manifest.provenance.source_revision.as_str(),
            ),
            (
                "source SHA-256",
                evidence.payload.source_sha256.as_str(),
                manifest.provenance.source_sha256.as_str(),
            ),
            (
                "checkpoint SHA-256",
                evidence.payload.checkpoint_sha256.as_str(),
                manifest.provenance.checkpoint_sha256.as_str(),
            ),
        ] {
            if observed != expected {
                return Err(format!(
                    "target-speaker promotion evidence {label} does not match the authenticated package"
                ));
            }
        }
        let mut options = BackendOptions::default().with_runtime_model_package(package.clone());
        options.deterministic = true;
        options.accelerator = requested;
        let accelerator = crate::select_accelerator_for_options(Backend::Onnx, &options)?;
        if !package.supports_accelerator(accelerator.effective()) {
            return Err(format!(
                "target-speaker package does not permit the {} accelerator",
                accelerator.effective().name()
            ));
        }
        let model = crate::backend::target_speaker::TargetSpeakerModel::load_runtime_package(
            &package,
            accelerator.effective(),
        )?;
        let payload = &evidence.payload;
        Ok(Self {
            package,
            model,
            accelerator,
            evidence: TargetSpeakerEvidenceIdentity {
                signing_key_id: evidence.signature.key_id.clone(),
                corpus_manifest_sha256: payload.corpus_manifest_sha256.clone(),
                evaluation_result_sha256: payload.evaluation_result_sha256.clone(),
                real_t_result_sha256: payload.real_t_result_sha256.clone(),
                ts_superb_result_sha256: payload.ts_superb_result_sha256.clone(),
                strata: payload.strata.len() as u32,
                target_speakers: payload.target_speaker_count,
                interferer_speakers: payload.interferer_speaker_count,
                languages: payload.language_count,
                accepted: true,
            },
        })
    }

    #[must_use]
    pub const fn accelerator(&self) -> AcceleratorSelection {
        self.accelerator
    }

    pub fn model_working_set_bytes(&self) -> Result<u64, String> {
        let profile = self
            .package
            .precision_profile_for(self.accelerator.effective())?
            .expect("target-speaker packages use v2 precision profiles");
        Ok(profile
            .resources
            .max_session_memory_bytes
            .saturating_add(profile.resources.max_worker_memory_bytes))
    }

    pub fn extract(
        &self,
        mixture: &Audio,
        enrollment: Audio,
        config: &TargetSpeakerExtractionConfig,
    ) -> Result<TargetSpeakerExtractionResult, String> {
        config.validate()?;
        validate_audio(mixture, "mixture", false)?;
        let enrollment = SensitiveEnrollment::new(enrollment);
        validate_audio(enrollment.audio(), "enrollment", false)?;
        let model_rate = self
            .package
            .manifest_v2()
            .expect("target-speaker session requires v2")
            .runtime
            .sample_rate_hz;
        let source_frames = mixture.frames();
        if source_frames as u64
            > u64::from(mixture.sample_rate).saturating_mul(MAX_TARGET_SPEAKER_MIXTURE_SECONDS)
        {
            return Err(format!(
                "target-speaker mixture exceeds the {MAX_TARGET_SPEAKER_MIXTURE_SECONDS}-second limit"
            ));
        }
        let enrollment_summary_input = (
            enrollment.audio().sample_rate,
            enrollment.audio().channels(),
            enrollment.audio().frames(),
        );
        let mixture_mono = mono_mix(mixture, "mixture")?;
        let enrollment_mono = Zeroizing::new(mono_mix(enrollment.audio(), "enrollment")?);
        let enrollment_model_f64 = Zeroizing::new(crate::resample::resample(
            &enrollment_mono,
            enrollment.audio().sample_rate,
            model_rate,
        )?);
        let enrollment_model = Zeroizing::new(
            enrollment_model_f64
                .iter()
                .map(|sample| *sample as f32)
                .collect::<Vec<_>>(),
        );
        validate_enrollment_duration(enrollment_model.len(), model_rate)?;
        if let Some(required) = self.model.fixed_enrollment_samples() {
            if enrollment_model.len() != required {
                return Err(format!(
                    "target-speaker package requires exactly {required} enrollment samples at {model_rate} Hz, got {}",
                    enrollment_model.len()
                ));
            }
        }
        let enrollment_model_samples = enrollment_model.len();
        let mixture_model_f64 =
            crate::resample::resample(&mixture_mono, mixture.sample_rate, model_rate)?;
        if mixture_model_f64.is_empty() {
            return Err("target-speaker mixture becomes empty at the model sample rate".into());
        }
        let mixture_model = mixture_model_f64
            .iter()
            .map(|sample| *sample as f32)
            .collect::<Vec<_>>();
        let inference = self.model.process(&mixture_model, &enrollment_model)?;
        drop(enrollment_model);
        drop(enrollment_model_f64);
        drop(enrollment_mono);
        drop(enrollment);
        let presence_values = inference.presence_probabilities;
        let candidate_model = Zeroizing::new(inference.audio);
        let candidate_resampled = Zeroizing::new(crate::resample::resample(
            &candidate_model
                .iter()
                .map(|sample| f64::from(*sample))
                .collect::<Vec<_>>(),
            model_rate,
            mixture.sample_rate,
        )?);
        let mut candidate = Zeroizing::new(Vec::new());
        candidate
            .try_reserve_exact(source_frames)
            .map_err(|_| "unable to reserve target-speaker candidate".to_string())?;
        candidate.extend(candidate_resampled.iter().copied().take(source_frames));
        candidate.resize(source_frames, 0.0);

        let presence = classify_presence(presence_values, config);
        let mixture_measurements = signal_measurements(&mixture_mono);
        let candidate_measurements = signal_measurements(&candidate);
        let geometry_passed = candidate.len() == source_frames;
        let finite_normalized_passed = candidate
            .iter()
            .all(|sample| sample.is_finite() && (-1.0..=1.0).contains(sample));
        let energy_delta_db = candidate_measurements.rms_dbfs - mixture_measurements.rms_dbfs;
        let peak_delta_db = candidate_measurements.peak_dbfs - mixture_measurements.peak_dbfs;
        let new_clipping =
            (candidate_measurements.clipping_ratio - mixture_measurements.clipping_ratio).max(0.0);
        let gates = vec![
            safety_gate(
                TargetSpeakerSafetyGateKind::Geometry,
                bool_value(geometry_passed),
                1.0,
                geometry_passed,
            ),
            safety_gate(
                TargetSpeakerSafetyGateKind::FiniteNormalizedSamples,
                bool_value(finite_normalized_passed),
                1.0,
                finite_normalized_passed,
            ),
            safety_gate(
                TargetSpeakerSafetyGateKind::EnergyGain,
                energy_delta_db,
                config.maximum_energy_gain_db,
                energy_delta_db <= config.maximum_energy_gain_db,
            ),
            safety_gate(
                TargetSpeakerSafetyGateKind::PeakGain,
                peak_delta_db,
                config.maximum_peak_gain_db,
                peak_delta_db <= config.maximum_peak_gain_db,
            ),
            safety_gate(
                TargetSpeakerSafetyGateKind::NewClipping,
                new_clipping,
                config.maximum_new_clipping_ratio,
                new_clipping <= config.maximum_new_clipping_ratio,
            ),
            safety_gate(
                TargetSpeakerSafetyGateKind::TargetPresence,
                f64::from(presence_values[2]),
                config.minimum_present_probability,
                presence == TargetSpeakerPresence::Present,
            ),
            safety_gate(
                TargetSpeakerSafetyGateKind::PromotionEvidence,
                1.0,
                1.0,
                true,
            ),
        ];
        let signal_gates_passed = gates
            .iter()
            .filter(|gate| gate.kind != TargetSpeakerSafetyGateKind::TargetPresence)
            .all(|gate| gate.passed);
        let decision = match presence {
            TargetSpeakerPresence::Absent => TargetSpeakerDecision::WithheldAbsent,
            TargetSpeakerPresence::Uncertain => TargetSpeakerDecision::WithheldUncertain,
            TargetSpeakerPresence::Present if !signal_gates_passed => {
                TargetSpeakerDecision::WithheldSafetyGate
            }
            TargetSpeakerPresence::Present => TargetSpeakerDecision::AcceptedPresent,
        };
        let accepted = decision == TargetSpeakerDecision::AcceptedPresent;
        let output = if accepted {
            Some(Audio {
                sample_rate: mixture.sample_rate,
                channels: vec![candidate.iter().copied().collect()],
                bits_per_sample: mixture.bits_per_sample,
                sample_format: mixture.sample_format,
                channel_mask: None,
            })
        } else {
            None
        };
        let output_digest = output
            .as_ref()
            .map(|audio| pcm_digest(audio, OUTPUT_PCM_DIGEST_DOMAIN));
        let mut warnings = Vec::new();
        match decision {
            TargetSpeakerDecision::AcceptedPresent => {}
            TargetSpeakerDecision::WithheldAbsent => warnings.push(
                "the calibrated presence head classified the target as absent; no audio was published"
                    .into(),
            ),
            TargetSpeakerDecision::WithheldUncertain => warnings.push(
                "target presence was uncertain; no mixture or candidate fallback was published"
                    .into(),
            ),
            TargetSpeakerDecision::WithheldSafetyGate => {
                let failed = gates
                    .iter()
                    .filter(|gate| !gate.passed)
                    .map(|gate| format!("{:?}", gate.kind).to_ascii_lowercase())
                    .collect::<Vec<_>>()
                    .join(", ");
                warnings.push(format!(
                    "target-speaker candidate failed safety gates ({failed}); no audio was published"
                ));
            }
        }
        let manifest = self
            .package
            .manifest_v2()
            .expect("target-speaker session requires v2");
        let profile = self
            .package
            .precision_profile_for(self.accelerator.effective())?
            .expect("target-speaker session selects one v2 profile");
        let report = TargetSpeakerExtractionReport {
            schema: TARGET_SPEAKER_REPORT_SCHEMA.into(),
            schema_version: TARGET_SPEAKER_SCHEMA_VERSION,
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            network_accessed: false,
            deterministic: true,
            model: TargetSpeakerModelIdentity {
                package_sha256: self.package.package_sha256().into(),
                public_key_sha256: self.package.public_key_sha256().into(),
                package_id: manifest.package_id.clone(),
                package_revision: manifest.package_revision.clone(),
                precision_profile: profile.id.clone(),
                source_revision: manifest.provenance.source_revision.clone(),
                source_sha256: manifest.provenance.source_sha256.clone(),
                source_license_spdx: manifest.provenance.source_license_spdx.clone(),
                checkpoint_sha256: manifest.provenance.checkpoint_sha256.clone(),
                checkpoint_license_spdx: manifest.provenance.checkpoint_license_spdx.clone(),
                accelerator: self.accelerator.effective().name().into(),
            },
            promotion_evidence: self.evidence.clone(),
            decision,
            model_invoked: true,
            candidate_accepted: accepted,
            output_published: accepted,
            candidate_retained: accepted,
            source_sample_rate: mixture.sample_rate,
            source_channels: mixture.channels(),
            source_frames,
            output_channels: 1,
            output_frames: accepted.then_some(source_frames),
            mixture_mixdown_policy: "arithmetic-mean-mono-v1".into(),
            mixture_pcm_sha256: pcm_digest(mixture, MIXTURE_PCM_DIGEST_DOMAIN),
            candidate_pcm_sha256: output_digest.clone(),
            output_pcm_sha256: output_digest,
            enrollment: TargetSpeakerEnrollmentSummary {
                input_sample_rate: enrollment_summary_input.0,
                input_channels: enrollment_summary_input.1,
                input_frames: enrollment_summary_input.2,
                model_sample_rate: model_rate,
                model_samples: enrollment_model_samples,
                mixdown_policy: "arithmetic-mean-mono-v1".into(),
                raw_audio_retained: false,
                embedding_retained: false,
                digest_recorded: false,
            },
            presence: TargetSpeakerPresenceAssessment {
                state: presence,
                absent_probability: f64::from(presence_values[0]),
                uncertain_probability: f64::from(presence_values[1]),
                present_probability: f64::from(presence_values[2]),
                minimum_absent_probability: config.minimum_absent_probability,
                minimum_present_probability: config.minimum_present_probability,
            },
            measurements: TargetSpeakerSafetyMeasurements {
                mixture_rms_dbfs: mixture_measurements.rms_dbfs,
                candidate_rms_dbfs: candidate_measurements.rms_dbfs,
                mixture_peak_dbfs: mixture_measurements.peak_dbfs,
                candidate_peak_dbfs: candidate_measurements.peak_dbfs,
                energy_delta_db,
                mixture_clipping_ratio: mixture_measurements.clipping_ratio,
                candidate_clipping_ratio: candidate_measurements.clipping_ratio,
            },
            safety_gates: gates,
            runtime_speaker_identity_verified: false,
            interferer_leakage_measured_at_runtime: false,
            limitations: limitations(),
            warnings,
        };
        Ok(TargetSpeakerExtractionResult {
            audio: output,
            report,
        })
    }
}

#[cfg(feature = "onnx")]
struct SensitiveEnrollment(Audio);

#[cfg(feature = "onnx")]
impl SensitiveEnrollment {
    fn new(audio: Audio) -> Self {
        Self(audio)
    }

    fn audio(&self) -> &Audio {
        &self.0
    }
}

#[cfg(feature = "onnx")]
impl Drop for SensitiveEnrollment {
    fn drop(&mut self) {
        for channel in &mut self.0.channels {
            channel.zeroize();
        }
    }
}

#[cfg(feature = "onnx")]
fn validate_audio(audio: &Audio, label: &str, allow_empty: bool) -> Result<(), String> {
    if audio.sample_rate == 0 {
        return Err(format!("target-speaker {label} sample rate is invalid"));
    }
    if audio.channels.is_empty() || audio.channels.len() > MAX_CHANNELS {
        return Err(format!(
            "target-speaker {label} channels must be in 1..={MAX_CHANNELS}"
        ));
    }
    let frames = audio.channels[0].len();
    if !allow_empty && frames == 0 {
        return Err(format!("target-speaker {label} must not be empty"));
    }
    if audio.channels.iter().any(|channel| channel.len() != frames) {
        return Err(format!(
            "target-speaker {label} channels must have equal lengths"
        ));
    }
    if audio
        .channels
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(format!(
            "target-speaker {label} contains an invalid normalized sample"
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn mono_mix(audio: &Audio, label: &str) -> Result<Vec<f64>, String> {
    let frames = audio.frames();
    let mut mono = Vec::new();
    mono.try_reserve_exact(frames)
        .map_err(|_| format!("unable to reserve target-speaker {label} mono mix"))?;
    let scale = 1.0 / audio.channels() as f64;
    for frame in 0..frames {
        let value = audio
            .channels
            .iter()
            .map(|channel| channel[frame])
            .sum::<f64>()
            * scale;
        mono.push(value);
    }
    Ok(mono)
}

#[cfg(feature = "onnx")]
fn validate_enrollment_duration(samples: usize, sample_rate: u32) -> Result<(), String> {
    let millis = (samples as u64)
        .saturating_mul(1000)
        .checked_div(u64::from(sample_rate))
        .unwrap_or(0);
    if !(MIN_TARGET_SPEAKER_ENROLLMENT_MILLIS..=MAX_TARGET_SPEAKER_ENROLLMENT_MILLIS)
        .contains(&millis)
    {
        return Err(format!(
            "target-speaker enrollment must be {MIN_TARGET_SPEAKER_ENROLLMENT_MILLIS}..={MAX_TARGET_SPEAKER_ENROLLMENT_MILLIS} ms after resampling, got {millis} ms"
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn classify_presence(
    probabilities: [f32; 3],
    config: &TargetSpeakerExtractionConfig,
) -> TargetSpeakerPresence {
    let absent = f64::from(probabilities[0]);
    let uncertain = f64::from(probabilities[1]);
    let present = f64::from(probabilities[2]);
    if present >= config.minimum_present_probability && present > absent && present > uncertain {
        TargetSpeakerPresence::Present
    } else if absent >= config.minimum_absent_probability && absent > present && absent > uncertain
    {
        TargetSpeakerPresence::Absent
    } else {
        TargetSpeakerPresence::Uncertain
    }
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy)]
struct SignalMeasurements {
    rms_dbfs: f64,
    peak_dbfs: f64,
    clipping_ratio: f64,
}

#[cfg(feature = "onnx")]
fn signal_measurements(samples: &[f64]) -> SignalMeasurements {
    if samples.is_empty() {
        return SignalMeasurements {
            rms_dbfs: -240.0,
            peak_dbfs: -240.0,
            clipping_ratio: 0.0,
        };
    }
    let energy = samples.iter().fold(0.0, |sum, sample| {
        if sample.is_finite() {
            sum + sample * sample
        } else {
            f64::INFINITY
        }
    });
    let rms = (energy / samples.len() as f64).sqrt();
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f64, f64::max);
    let clipping =
        samples.iter().filter(|sample| sample.abs() >= 1.0).count() as f64 / samples.len() as f64;
    SignalMeasurements {
        rms_dbfs: amplitude_dbfs(rms),
        peak_dbfs: amplitude_dbfs(peak),
        clipping_ratio: clipping,
    }
}

#[cfg(feature = "onnx")]
fn amplitude_dbfs(amplitude: f64) -> f64 {
    if !amplitude.is_finite() {
        240.0
    } else {
        (20.0 * amplitude.max(SILENCE_FLOOR).log10()).clamp(-240.0, 240.0)
    }
}

#[cfg(feature = "onnx")]
fn safety_gate(
    kind: TargetSpeakerSafetyGateKind,
    observed: f64,
    limit: f64,
    passed: bool,
) -> TargetSpeakerSafetyGate {
    TargetSpeakerSafetyGate {
        kind,
        observed: observed.clamp(-240.0, 240.0),
        limit: limit.clamp(-240.0, 240.0),
        passed,
    }
}

#[cfg(feature = "onnx")]
const fn bool_value(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

#[cfg(feature = "onnx")]
fn pcm_digest(audio: &Audio, domain: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(audio.sample_rate.to_be_bytes());
    digest.update((audio.channels() as u64).to_be_bytes());
    digest.update((audio.frames() as u64).to_be_bytes());
    for channel in &audio.channels {
        for sample in channel {
            digest.update(sample.to_bits().to_be_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

#[cfg(feature = "onnx")]
fn limitations() -> Vec<String> {
    vec![
        "the runtime presence head is not an independent speaker-verification system".into(),
        "interferer leakage and target identity are promotion-time measurements, not runtime measurements"
            .into(),
        "a valid evidence signature authenticates the evaluator's claim but cannot prove the underlying recordings or labels are truthful"
            .into(),
        "the v1 adapter mixes program channels to mono and does not preserve or infer spatial position"
            .into(),
        "enrollment buffers are zeroized on ordinary drop, but operating-system caches, allocator copies, swap, and crash dumps are outside this guarantee"
            .into(),
        "denoize does not bundle a target-speaker checkpoint until artifact-level redistribution and protected-stratum gates are independently satisfied"
            .into(),
    ]
}

fn validate_metric_policy(
    metric: &TargetSpeakerMetricOutcome,
    policy: &MetricPolicy,
) -> Result<(), String> {
    if metric.operator != policy.operator {
        return Err(format!(
            "target-speaker evidence metric {} uses the wrong operator",
            metric.metric
        ));
    }
    let strong_enough = match policy.operator {
        TargetSpeakerMetricOperator::GreaterOrEqual => metric.limit >= policy.hard_limit,
        TargetSpeakerMetricOperator::LessOrEqual => metric.limit <= policy.hard_limit,
    };
    if !strong_enough {
        return Err(format!(
            "target-speaker evidence metric {} uses a weaker limit than the release policy {}",
            metric.metric, policy.hard_limit
        ));
    }
    Ok(())
}

fn validate_range(label: &str, value: f64, minimum: f64, maximum: f64) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        Err(format!(
            "target-speaker {label} must be finite and in {minimum}..={maximum}"
        ))
    } else {
        Ok(())
    }
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
    {
        return Err(format!(
            "target-speaker evidence {label} must be lowercase SHA-256"
        ));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 256
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._+-".contains(&byte)
        })
    {
        return Err(format!(
            "{label} must use 1..=256 lowercase ASCII identifier characters"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_evidence_enforces_protected_strata_and_hard_limits() {
        let payload = passing_payload();
        payload.validate().unwrap();

        let mut missing = payload.clone();
        missing.strata.remove(0);
        assert!(missing
            .validate()
            .unwrap_err()
            .contains("omits required stratum"));

        let mut weak = payload.clone();
        let metric = weak
            .strata
            .iter_mut()
            .find(|stratum| stratum.kind == TargetSpeakerStratumKind::TargetPresent)
            .unwrap()
            .metrics
            .iter_mut()
            .find(|metric| metric.metric == "speaker.target-similarity")
            .unwrap();
        metric.limit = 0.1;
        metric.value = 0.1;
        assert!(weak.validate().unwrap_err().contains("weaker limit"));

        let mut uncalibrated = payload;
        uncalibrated.presence_expected_calibration_error = 0.051;
        assert!(uncalibrated
            .validate()
            .unwrap_err()
            .contains("accepted flag"));
    }

    #[test]
    fn extraction_config_is_closed_and_conservative() {
        let config = TargetSpeakerExtractionConfig::default();
        config.validate().unwrap();
        assert_eq!(config.minimum_present_probability, 0.90);
        assert_eq!(config.minimum_absent_probability, 0.90);
        let encoded = serde_json::to_string(&config).unwrap();
        let unknown = encoded.replace('{', "{\"unknown\":true,");
        assert!(serde_json::from_str::<TargetSpeakerExtractionConfig>(&unknown).is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn three_state_presence_never_promotes_ambiguous_probabilities() {
        let config = TargetSpeakerExtractionConfig::default();
        assert_eq!(
            classify_presence([0.01, 0.01, 0.98], &config),
            TargetSpeakerPresence::Present
        );
        assert_eq!(
            classify_presence([0.98, 0.01, 0.01], &config),
            TargetSpeakerPresence::Absent
        );
        assert_eq!(
            classify_presence([0.40, 0.20, 0.40], &config),
            TargetSpeakerPresence::Uncertain
        );
    }

    fn passing_payload() -> TargetSpeakerPromotionEvidencePayload {
        let strata = REQUIRED_STRATA
            .iter()
            .map(|(id, kind)| TargetSpeakerStratumEvidence {
                id: (*id).into(),
                kind: *kind,
                cases: 10,
                metrics: match kind {
                    TargetSpeakerStratumKind::TargetPresent => metric_outcomes(PRESENT_METRICS),
                    TargetSpeakerStratumKind::TargetAbsent => metric_outcomes(ABSENT_METRICS),
                },
            })
            .collect();
        TargetSpeakerPromotionEvidencePayload {
            completed_at_unix_seconds: 1_800_000_000,
            model_package_sha256: "0".repeat(64),
            source_revision: "0123456789abcdef".into(),
            source_sha256: "1".repeat(64),
            checkpoint_sha256: "2".repeat(64),
            corpus_manifest_sha256: "3".repeat(64),
            evaluation_result_sha256: "4".repeat(64),
            real_t_result_sha256: "5".repeat(64),
            ts_superb_result_sha256: "6".repeat(64),
            strata,
            target_speaker_count: 100,
            interferer_speaker_count: 100,
            language_count: 2,
            presence_expected_calibration_error: 0.05,
            presence_expected_calibration_error_limit: 0.05,
            minimum_listeners: 20,
            listener_count: 20,
            listener_preference: 0.5,
            listener_preference_limit: 0.5,
            accepted: true,
        }
    }

    fn metric_outcomes(policies: &[MetricPolicy]) -> Vec<TargetSpeakerMetricOutcome> {
        policies
            .iter()
            .map(|policy| TargetSpeakerMetricOutcome {
                metric: policy.name.into(),
                value: policy.hard_limit,
                operator: policy.operator,
                limit: policy.hard_limit,
                passed: true,
            })
            .collect()
    }
}
