//! Fail-closed causal target-speaker extraction.
//!
//! The streaming adapter retains the offline Stage 29 package/evidence gate and
//! adds per-stratum non-inferiority, measured latency, recurrent-state, target-
//! presence transition, and real-time callback evidence. A live block is
//! audible only after a conservative present hold; absent, uncertain, warm-up,
//! and signal-failure blocks are explicitly muted.

use crate::audio::Audio;
use crate::backend::causal_target_speaker::{CausalTargetSpeakerModel, CausalTargetSpeakerRuntime};
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
use crate::target_speaker::{
    SignedTargetSpeakerPromotionEvidence, TargetSpeakerMetricOperator, TargetSpeakerPresence,
    TargetSpeakerStratumKind, MAX_TARGET_SPEAKER_ENROLLMENT_MILLIS,
    MAX_TARGET_SPEAKER_MIXTURE_SECONDS, MIN_TARGET_SPEAKER_ENROLLMENT_MILLIS,
};
use crate::{
    AcceleratorPreference, AcceleratorSelection, Backend, BackendOptions, RuntimeModelPackage,
};
use crossbeam_queue::ArrayQueue;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use zeroize::{Zeroize, Zeroizing};

pub const CAUSAL_TARGET_SPEAKER_PROMOTION_EVIDENCE_SCHEMA: &str =
    "denoize-causal-target-speaker-promotion-evidence-v1";
pub const CAUSAL_TARGET_SPEAKER_REPORT_SCHEMA: &str = "denoize-causal-target-speaker-report-v1";
pub const CAUSAL_TARGET_SPEAKER_SCHEMA_VERSION: u32 = 1;

const CAUSAL_SIGNATURE_DOMAIN: &[u8] = b"denoize-causal-target-speaker-promotion-evidence-v1";
const MAX_EVIDENCE_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_STRATA: usize = 256;
const MAX_METRICS: usize = 64;
const MAX_EFFECTIVE_LATENCY_MILLISECONDS: f64 = 100.0;
const JSON_SAFE_INTEGER_MAX: u64 = (1_u64 << 53) - 1;
const MAX_CHANNELS: usize = 64;
const SILENCE_FLOOR: f64 = 1.0e-12;
const REALTIME_QUEUE_BLOCKS: usize = 16;
const REALTIME_POOL_BLOCKS: usize = 40;
const REALTIME_WORKER_POLL: Duration = Duration::from_micros(100);
const MIXTURE_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-causal-target-speaker-mixture-pcm-v1\0";
const OUTPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-causal-target-speaker-output-pcm-v1\0";

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

#[derive(Clone, Copy)]
struct MetricPolicy {
    name: &'static str,
    operator: TargetSpeakerMetricOperator,
    hard_limit: f64,
    maximum_regression: f64,
}

impl MetricPolicy {
    const fn at_least(name: &'static str, hard_limit: f64, maximum_regression: f64) -> Self {
        Self {
            name,
            operator: TargetSpeakerMetricOperator::GreaterOrEqual,
            hard_limit,
            maximum_regression,
        }
    }

    const fn at_most(name: &'static str, hard_limit: f64, maximum_regression: f64) -> Self {
        Self {
            name,
            operator: TargetSpeakerMetricOperator::LessOrEqual,
            hard_limit,
            maximum_regression,
        }
    }
}

const PRESENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("content.target-word-error-rate", 0.35, 0.02),
    MetricPolicy::at_least("extraction.si-sdr-improvement-db", 3.0, 0.5),
    MetricPolicy::at_most("interferer.speaker-similarity", 0.30, 0.02),
    MetricPolicy::at_most("interferer.word-leakage-rate", 0.02, 0.005),
    MetricPolicy::at_most("output.duration-error-frames", 0.0, 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0, 0.0),
    MetricPolicy::at_least("perceptual.dnsmos-p808", 3.0, 0.1),
    MetricPolicy::at_least("presence.recall", 0.95, 0.02),
    MetricPolicy::at_least("speaker.target-similarity", 0.70, 0.02),
];

const ABSENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("interferer.speaker-similarity", 0.30, 0.02),
    MetricPolicy::at_most("interferer.word-leakage-rate", 0.01, 0.005),
    MetricPolicy::at_most("output.duration-error-frames", 0.0, 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0, 0.0),
    MetricPolicy::at_most("output.rms-dbfs", -60.0, 3.0),
    MetricPolicy::at_most("presence.false-positive-rate", 0.01, 0.005),
];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerMetricEvidence {
    pub metric: String,
    pub operator: TargetSpeakerMetricOperator,
    pub offline_value: f64,
    pub causal_value: f64,
    pub hard_limit: f64,
    pub maximum_regression: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerStratumEvidence {
    pub id: String,
    pub kind: TargetSpeakerStratumKind,
    pub offline_cases: u32,
    pub causal_cases: u32,
    pub metrics: Vec<CausalTargetSpeakerMetricEvidence>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerRealtimeAudit {
    pub paced_blocks: u64,
    pub deadline_misses: u64,
    pub overload_blocks: u64,
    pub queue_capacity_blocks: u32,
    pub maximum_queue_depth_blocks: u32,
    pub callback_allocations: u64,
    pub callback_locks: u64,
    pub callback_waits: u64,
    pub callback_file_io_operations: u64,
    pub callback_network_operations: u64,
    pub callback_log_operations: u64,
    pub callback_inference_calls: u64,
}

impl CausalTargetSpeakerRealtimeAudit {
    fn validate_structure(&self) -> Result<(), String> {
        if self.paced_blocks < 10_000
            || self.paced_blocks > JSON_SAFE_INTEGER_MAX
            || self.deadline_misses > JSON_SAFE_INTEGER_MAX
            || self.overload_blocks > JSON_SAFE_INTEGER_MAX
            || !(16..=256).contains(&self.queue_capacity_blocks)
            || self.maximum_queue_depth_blocks > 255
            || [
                self.callback_allocations,
                self.callback_locks,
                self.callback_waits,
                self.callback_file_io_operations,
                self.callback_network_operations,
                self.callback_log_operations,
                self.callback_inference_calls,
            ]
            .into_iter()
            .any(|value| value > JSON_SAFE_INTEGER_MAX)
        {
            return Err("causal target-speaker realtime audit is outside schema bounds".into());
        }
        Ok(())
    }

    fn passed(&self) -> bool {
        self.paced_blocks >= 10_000
            && self.deadline_misses == 0
            && self.overload_blocks == 0
            && (16..=256).contains(&self.queue_capacity_blocks)
            && self.maximum_queue_depth_blocks < self.queue_capacity_blocks
            && self.callback_allocations == 0
            && self.callback_locks == 0
            && self.callback_waits == 0
            && self.callback_file_io_operations == 0
            && self.callback_network_operations == 0
            && self.callback_log_operations == 0
            && self.callback_inference_calls == 0
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerTransitionAudit {
    pub absent_to_present_cases: u32,
    pub present_to_absent_cases: u32,
    pub uncertain_transition_cases: u32,
    pub enrollment_mismatch_cases: u32,
    pub reference_loss_cases: u32,
    pub late_results_injected: u32,
    pub late_results_discarded: u32,
    pub stale_generation_results_injected: u32,
    pub stale_generation_results_discarded: u32,
    pub false_attribution_publications: u32,
}

impl CausalTargetSpeakerTransitionAudit {
    fn validate_structure(&self) -> Result<(), String> {
        for value in [
            self.absent_to_present_cases,
            self.present_to_absent_cases,
            self.uncertain_transition_cases,
            self.enrollment_mismatch_cases,
            self.reference_loss_cases,
            self.late_results_injected,
            self.stale_generation_results_injected,
        ] {
            if !(100..=1_000_000).contains(&value) {
                return Err(
                    "causal target-speaker transition audit case count is outside schema bounds"
                        .into(),
                );
            }
        }
        for value in [
            self.late_results_discarded,
            self.stale_generation_results_discarded,
            self.false_attribution_publications,
        ] {
            if value > 1_000_000 {
                return Err(
                    "causal target-speaker transition audit result is outside schema bounds".into(),
                );
            }
        }
        Ok(())
    }

    fn passed(&self) -> bool {
        self.absent_to_present_cases >= 100
            && self.present_to_absent_cases >= 100
            && self.uncertain_transition_cases >= 100
            && self.enrollment_mismatch_cases >= 100
            && self.reference_loss_cases >= 100
            && self.late_results_injected >= 100
            && self.late_results_discarded == self.late_results_injected
            && self.stale_generation_results_injected >= 100
            && self.stale_generation_results_discarded == self.stale_generation_results_injected
            && self.false_attribution_publications == 0
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub model_package_sha256: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub checkpoint_sha256: String,
    pub offline_evaluation_result_sha256: String,
    pub causal_evaluation_result_sha256: String,
    pub state_reset_flush_result_sha256: String,
    pub latency_result_sha256: String,
    pub realtime_callback_result_sha256: String,
    pub transition_result_sha256: String,
    pub strata: Vec<CausalTargetSpeakerStratumEvidence>,
    pub model_sample_rate_hz: u32,
    pub frame_samples: u64,
    pub algorithmic_latency_samples: u64,
    pub flush_samples: u64,
    pub perturbation_latency_cases: u32,
    pub effective_latency_milliseconds: f64,
    pub effective_latency_limit_milliseconds: f64,
    pub realtime: CausalTargetSpeakerRealtimeAudit,
    pub transitions: CausalTargetSpeakerTransitionAudit,
    pub accepted: bool,
}

impl CausalTargetSpeakerPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        for (label, digest) in [
            ("model package", self.model_package_sha256.as_str()),
            ("source", self.source_sha256.as_str()),
            ("checkpoint", self.checkpoint_sha256.as_str()),
            (
                "offline evaluation result",
                self.offline_evaluation_result_sha256.as_str(),
            ),
            (
                "causal evaluation result",
                self.causal_evaluation_result_sha256.as_str(),
            ),
            (
                "state/reset/flush result",
                self.state_reset_flush_result_sha256.as_str(),
            ),
            ("latency result", self.latency_result_sha256.as_str()),
            (
                "realtime callback result",
                self.realtime_callback_result_sha256.as_str(),
            ),
            ("transition result", self.transition_result_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
        validate_identifier("source revision", &self.source_revision)?;
        if self.completed_at_unix_seconds > JSON_SAFE_INTEGER_MAX {
            return Err(
                "causal target-speaker evidence timestamp exceeds the JSON safe-integer limit"
                    .into(),
            );
        }
        if self.strata.is_empty() || self.strata.len() > MAX_STRATA {
            return Err(format!(
                "causal target-speaker evidence must contain 1..={MAX_STRATA} strata"
            ));
        }
        let required: BTreeMap<_, _> = REQUIRED_STRATA.iter().copied().collect();
        let mut observed = BTreeSet::new();
        let mut previous = None;
        let mut metrics_passed = true;
        for stratum in &self.strata {
            validate_identifier("causal target-speaker stratum", &stratum.id)?;
            if previous.is_some_and(|value: &str| value >= stratum.id.as_str()) {
                return Err(
                    "causal target-speaker strata must be unique and strictly sorted".into(),
                );
            }
            previous = Some(&stratum.id);
            observed.insert(stratum.id.as_str());
            if required
                .get(stratum.id.as_str())
                .is_some_and(|expected| *expected != stratum.kind)
            {
                return Err(format!(
                    "causal target-speaker stratum {} has the wrong presence kind",
                    stratum.id
                ));
            }
            if !(10..=1_000_000).contains(&stratum.offline_cases)
                || !(10..=1_000_000).contains(&stratum.causal_cases)
            {
                return Err(
                    "causal target-speaker stratum case counts must be in 10..=1000000".into(),
                );
            }
            if stratum.metrics.is_empty() || stratum.metrics.len() > MAX_METRICS {
                return Err(format!(
                    "causal target-speaker stratum metrics must be in 1..={MAX_METRICS}"
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
            let mut metric_names = BTreeSet::new();
            let mut previous_metric = None;
            for metric in &stratum.metrics {
                validate_identifier("causal target-speaker metric", &metric.metric)?;
                if previous_metric.is_some_and(|value: &str| value >= metric.metric.as_str()) {
                    return Err(
                        "causal target-speaker metrics must be unique and strictly sorted".into(),
                    );
                }
                previous_metric = Some(&metric.metric);
                metric_names.insert(metric.metric.as_str());
                let policy = policy_by_name.get(metric.metric.as_str()).ok_or_else(|| {
                    format!(
                        "causal target-speaker stratum {} has unsupported metric {}",
                        stratum.id, metric.metric
                    )
                })?;
                validate_metric(metric, policy)?;
                metrics_passed &= metric.passed;
            }
            for policy in policies {
                if !metric_names.contains(policy.name) {
                    return Err(format!(
                        "causal target-speaker stratum {} omits required metric {}",
                        stratum.id, policy.name
                    ));
                }
            }
        }
        for (id, _) in REQUIRED_STRATA {
            if !observed.contains(id) {
                return Err(format!(
                    "causal target-speaker evidence omits required stratum {id}"
                ));
            }
        }
        if self.model_sample_rate_hz == 0
            || self.model_sample_rate_hz > 768_000
            || self.frame_samples == 0
            || self.frame_samples > 16_777_216
            || self.algorithmic_latency_samples > 16_777_216
            || self.flush_samples > 16_777_216
        {
            return Err("causal target-speaker signed stream geometry is invalid".into());
        }
        if !(100..=1_000_000).contains(&self.perturbation_latency_cases)
            || !self.effective_latency_milliseconds.is_finite()
            || !self.effective_latency_limit_milliseconds.is_finite()
            || !(0.0..=MAX_EFFECTIVE_LATENCY_MILLISECONDS)
                .contains(&self.effective_latency_limit_milliseconds)
            || self.effective_latency_milliseconds < 0.0
        {
            return Err("causal target-speaker effective-latency evidence is invalid".into());
        }
        self.realtime.validate_structure()?;
        self.transitions.validate_structure()?;
        let signed_latency_milliseconds = self
            .algorithmic_latency_samples
            .saturating_mul(1000)
            .div_ceil(u64::from(self.model_sample_rate_hz));
        let geometry_passed = signed_latency_milliseconds <= 100
            && self.flush_samples >= self.algorithmic_latency_samples;
        let latency_passed =
            self.effective_latency_milliseconds <= self.effective_latency_limit_milliseconds;
        let expected = metrics_passed
            && geometry_passed
            && latency_passed
            && self.realtime.passed()
            && self.transitions.passed();
        if self.accepted != expected {
            return Err("causal target-speaker evidence accepted flag is inconsistent".into());
        }
        let bytes = serde_json::to_vec(self).map_err(|error| {
            format!("serialize causal target-speaker evidence payload: {error}")
        })?;
        if bytes.len() as u64 >= MAX_EVIDENCE_JSON_BYTES {
            return Err(
                "causal target-speaker evidence payload exceeds the bounded JSON limit".into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedCausalTargetSpeakerPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: CausalTargetSpeakerPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedCausalTargetSpeakerPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) =
            crate::input::open_regular_file(path, "causal target-speaker promotion evidence")?;
        if length >= MAX_EVIDENCE_JSON_BYTES {
            return Err(format!(
                "causal target-speaker promotion evidence {} exceeds the {MAX_EVIDENCE_JSON_BYTES}-byte limit",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve causal target-speaker evidence JSON".to_string())?;
        file.take(MAX_EVIDENCE_JSON_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read causal target-speaker promotion evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("causal target-speaker promotion evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse causal target-speaker promotion evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != CAUSAL_TARGET_SPEAKER_PROMOTION_EVIDENCE_SCHEMA
            || self.schema_version != CAUSAL_TARGET_SPEAKER_SCHEMA_VERSION
        {
            return Err("unsupported causal target-speaker promotion evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("causal target-speaker evidence signature must use ed25519".into());
        }
        validate_sha256("evidence key ID", &self.signature.key_id)?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize causal target-speaker evidence: {error}"))?;
        if bytes.len() as u64 >= MAX_EVIDENCE_JSON_BYTES {
            return Err("causal target-speaker evidence exceeds the bounded JSON limit".into());
        }
        Ok(())
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload).map_err(|error| {
            format!("serialize causal target-speaker evidence for verification: {error}")
        })?;
        key.verify_domain_document(
            CAUSAL_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "causal target-speaker promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize causal target-speaker evidence: {error}"))
    }
}

pub fn sign_causal_target_speaker_promotion_evidence(
    payload: CausalTargetSpeakerPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedCausalTargetSpeakerPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload).map_err(|error| {
        format!("serialize causal target-speaker evidence for signing: {error}")
    })?;
    let signature = key.sign_domain_document(
        CAUSAL_SIGNATURE_DOMAIN,
        &document,
        "causal target-speaker promotion evidence",
    )?;
    let evidence = SignedCausalTargetSpeakerPromotionEvidence {
        schema: CAUSAL_TARGET_SPEAKER_PROMOTION_EVIDENCE_SCHEMA.into(),
        schema_version: CAUSAL_TARGET_SPEAKER_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerConfig {
    pub minimum_present_probability: f64,
    pub minimum_absent_probability: f64,
    pub present_hold_blocks: u32,
    pub maximum_energy_gain_db: f64,
    pub maximum_peak: f64,
}

impl Default for CausalTargetSpeakerConfig {
    fn default() -> Self {
        Self {
            minimum_present_probability: 0.90,
            minimum_absent_probability: 0.90,
            present_hold_blocks: 3,
            maximum_energy_gain_db: 3.0,
            maximum_peak: 1.0,
        }
    }
}

impl CausalTargetSpeakerConfig {
    pub fn validate(&self) -> Result<(), String> {
        validate_range(
            "minimum present probability",
            self.minimum_present_probability,
            0.5,
            1.0,
        )?;
        validate_range(
            "minimum absent probability",
            self.minimum_absent_probability,
            0.5,
            1.0,
        )?;
        if !(1..=100).contains(&self.present_hold_blocks) {
            return Err("causal target-speaker present hold must be in 1..=100 blocks".into());
        }
        validate_range(
            "maximum energy gain dB",
            self.maximum_energy_gain_db,
            0.0,
            12.0,
        )?;
        validate_range("maximum peak", self.maximum_peak, 0.5, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CausalTargetSpeakerBlockDecision {
    PublishedPresent,
    MutedAbsent,
    MutedUncertain,
    MutedPresentWarmup,
    MutedSafetyGate,
    MutedFlush,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerBlock {
    pub generation: u64,
    pub start_frame: u64,
    pub valid_frames: usize,
    pub audio: Vec<f32>,
    pub presence: TargetSpeakerPresence,
    pub absent_probability: f32,
    pub uncertain_probability: f32,
    pub present_probability: f32,
    pub decision: CausalTargetSpeakerBlockDecision,
    pub candidate_accepted: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerModelIdentity {
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
pub struct CausalTargetSpeakerEvidenceIdentity {
    pub offline_signing_key_id: String,
    pub causal_signing_key_id: String,
    pub offline_evaluation_result_sha256: String,
    pub causal_evaluation_result_sha256: String,
    pub state_reset_flush_result_sha256: String,
    pub latency_result_sha256: String,
    pub realtime_callback_result_sha256: String,
    pub transition_result_sha256: String,
    pub strata: u32,
    pub accepted: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerEnrollmentSummary {
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

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerDecisionCounts {
    pub published_present_blocks: u64,
    pub muted_absent_blocks: u64,
    pub muted_uncertain_blocks: u64,
    pub muted_present_warmup_blocks: u64,
    pub muted_safety_gate_blocks: u64,
    pub muted_flush_blocks: u64,
}

impl CausalTargetSpeakerDecisionCounts {
    fn observe(&mut self, decision: CausalTargetSpeakerBlockDecision) {
        match decision {
            CausalTargetSpeakerBlockDecision::PublishedPresent => {
                self.published_present_blocks += 1;
            }
            CausalTargetSpeakerBlockDecision::MutedAbsent => self.muted_absent_blocks += 1,
            CausalTargetSpeakerBlockDecision::MutedUncertain => self.muted_uncertain_blocks += 1,
            CausalTargetSpeakerBlockDecision::MutedPresentWarmup => {
                self.muted_present_warmup_blocks += 1;
            }
            CausalTargetSpeakerBlockDecision::MutedSafetyGate => {
                self.muted_safety_gate_blocks += 1;
            }
            CausalTargetSpeakerBlockDecision::MutedFlush => self.muted_flush_blocks += 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSpeakerRenderReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub network_accessed: bool,
    pub deterministic: bool,
    pub model: CausalTargetSpeakerModelIdentity,
    pub promotion_evidence: CausalTargetSpeakerEvidenceIdentity,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_frames: usize,
    pub output_channels: usize,
    pub output_frames: usize,
    pub model_sample_rate: u32,
    pub frame_samples: usize,
    pub algorithmic_latency_samples: usize,
    pub flush_samples: usize,
    pub input_blocks: u64,
    pub flush_blocks: u64,
    pub decision_counts: CausalTargetSpeakerDecisionCounts,
    pub presence_transitions: u64,
    pub rendered_audio_published: bool,
    pub mixture_mixdown_policy: String,
    pub mixture_pcm_sha256: String,
    pub output_pcm_sha256: String,
    pub enrollment: CausalTargetSpeakerEnrollmentSummary,
    pub runtime_speaker_identity_verified: bool,
    pub interferer_leakage_measured_at_runtime: bool,
    pub limitations: Vec<String>,
    pub warnings: Vec<String>,
}

impl CausalTargetSpeakerRenderReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|error| format!("serialize causal target-speaker report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize causal target-speaker report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct CausalTargetSpeakerRenderResult {
    pub audio: Audio,
    pub report: CausalTargetSpeakerRenderReport,
}

/// Conservative decoded, resampled, recurrent-render, alignment, and output
/// memory allowance. Signed model-session resources and decoder-session state
/// are admitted separately by callers.
#[must_use]
pub fn estimate_causal_target_speaker_memory_bytes(
    mixture: &Audio,
    enrollment: &Audio,
    model_sample_rate: u32,
    frame_samples: usize,
    flush_samples: usize,
) -> u64 {
    let frames_at_rate = |frames: usize, source_rate: u32| {
        let numerator = (frames as u128)
            .saturating_mul(u128::from(model_sample_rate))
            .saturating_add(u128::from(source_rate.saturating_sub(1)));
        let frames = numerator / u128::from(source_rate.max(1));
        u64::try_from(frames).unwrap_or(u64::MAX)
    };
    let source_frames = mixture.frames() as u64;
    let model_frames = frames_at_rate(mixture.frames(), mixture.sample_rate);
    let enrollment_source_frames = enrollment.frames() as u64;
    let enrollment_model_frames = frames_at_rate(enrollment.frames(), enrollment.sample_rate);
    let padded_model_frames = model_frames
        .saturating_add(frame_samples.saturating_sub(1) as u64)
        .checked_div(frame_samples.max(1) as u64)
        .unwrap_or(u64::MAX)
        .saturating_mul(frame_samples as u64)
        .saturating_add(flush_samples as u64);
    crate::audio::estimate_audio_memory_bytes(mixture)
        .saturating_add(crate::audio::estimate_audio_memory_bytes(enrollment))
        .saturating_add(source_frames.saturating_mul(24))
        .saturating_add(model_frames.saturating_mul(24))
        .saturating_add(padded_model_frames.saturating_mul(4))
        .saturating_add(enrollment_source_frames.saturating_mul(8))
        .saturating_add(enrollment_model_frames.saturating_mul(12))
        .saturating_add(1024 * 1024)
}

pub struct CausalTargetSpeakerSession {
    package: RuntimeModelPackage,
    model: CausalTargetSpeakerModel,
    accelerator: AcceleratorSelection,
    evidence: CausalTargetSpeakerEvidenceIdentity,
}

impl std::fmt::Debug for CausalTargetSpeakerSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CausalTargetSpeakerSession")
            .field("package_sha256", &self.package.package_sha256())
            .field("accelerator", &self.accelerator)
            .field("sample_rate_hz", &self.model.sample_rate_hz())
            .field("frame_samples", &self.model.frame_samples())
            .finish_non_exhaustive()
    }
}

impl CausalTargetSpeakerSession {
    pub fn prepare(
        package: RuntimeModelPackage,
        offline_evidence: &SignedTargetSpeakerPromotionEvidence,
        offline_evidence_key: &ReceiptPublicKey,
        causal_evidence: &SignedCausalTargetSpeakerPromotionEvidence,
        causal_evidence_key: &ReceiptPublicKey,
        requested: AcceleratorPreference,
    ) -> Result<Self, String> {
        offline_evidence.verify_signature(offline_evidence_key)?;
        causal_evidence.verify_signature(causal_evidence_key)?;
        if !offline_evidence.payload.accepted || !causal_evidence.payload.accepted {
            return Err("causal target-speaker extraction requires accepted offline and causal promotion evidence".into());
        }
        validate_offline_matrix_binding(offline_evidence, causal_evidence)?;
        let manifest = package
            .manifest_v2()
            .ok_or("causal target-speaker extraction rejects runtime model package v1")?;
        for (label, observed, expected) in [
            (
                "offline model package SHA-256",
                offline_evidence.payload.model_package_sha256.as_str(),
                package.package_sha256(),
            ),
            (
                "causal model package SHA-256",
                causal_evidence.payload.model_package_sha256.as_str(),
                package.package_sha256(),
            ),
            (
                "offline source revision",
                offline_evidence.payload.source_revision.as_str(),
                manifest.provenance.source_revision.as_str(),
            ),
            (
                "offline source SHA-256",
                offline_evidence.payload.source_sha256.as_str(),
                manifest.provenance.source_sha256.as_str(),
            ),
            (
                "offline checkpoint SHA-256",
                offline_evidence.payload.checkpoint_sha256.as_str(),
                manifest.provenance.checkpoint_sha256.as_str(),
            ),
            (
                "causal source revision",
                causal_evidence.payload.source_revision.as_str(),
                manifest.provenance.source_revision.as_str(),
            ),
            (
                "causal source SHA-256",
                causal_evidence.payload.source_sha256.as_str(),
                manifest.provenance.source_sha256.as_str(),
            ),
            (
                "causal checkpoint SHA-256",
                causal_evidence.payload.checkpoint_sha256.as_str(),
                manifest.provenance.checkpoint_sha256.as_str(),
            ),
            (
                "offline evaluation result SHA-256",
                causal_evidence
                    .payload
                    .offline_evaluation_result_sha256
                    .as_str(),
                offline_evidence.payload.evaluation_result_sha256.as_str(),
            ),
        ] {
            if observed != expected {
                return Err(format!(
                    "causal target-speaker promotion evidence {label} does not match the authenticated prerequisite"
                ));
            }
        }
        if causal_evidence.payload.model_sample_rate_hz != manifest.runtime.sample_rate_hz
            || causal_evidence.payload.frame_samples != manifest.latency.frame_samples
            || causal_evidence.payload.algorithmic_latency_samples
                != manifest.latency.algorithmic_latency_samples
            || causal_evidence.payload.flush_samples != manifest.latency.flush_samples
        {
            return Err("causal target-speaker evidence stream geometry does not match the authenticated package".into());
        }
        let mut options = BackendOptions::default().with_runtime_model_package(package.clone());
        options.deterministic = true;
        options.accelerator = requested;
        let accelerator = crate::select_accelerator_for_options(Backend::Onnx, &options)?;
        if !package.supports_accelerator(accelerator.effective()) {
            return Err(format!(
                "causal target-speaker package does not permit the {} accelerator",
                accelerator.effective().name()
            ));
        }
        let model =
            CausalTargetSpeakerModel::load_runtime_package(&package, accelerator.effective())?;
        let evidence = CausalTargetSpeakerEvidenceIdentity {
            offline_signing_key_id: offline_evidence.signature.key_id.clone(),
            causal_signing_key_id: causal_evidence.signature.key_id.clone(),
            offline_evaluation_result_sha256: causal_evidence
                .payload
                .offline_evaluation_result_sha256
                .clone(),
            causal_evaluation_result_sha256: causal_evidence
                .payload
                .causal_evaluation_result_sha256
                .clone(),
            state_reset_flush_result_sha256: causal_evidence
                .payload
                .state_reset_flush_result_sha256
                .clone(),
            latency_result_sha256: causal_evidence.payload.latency_result_sha256.clone(),
            realtime_callback_result_sha256: causal_evidence
                .payload
                .realtime_callback_result_sha256
                .clone(),
            transition_result_sha256: causal_evidence.payload.transition_result_sha256.clone(),
            strata: causal_evidence.payload.strata.len() as u32,
            accepted: true,
        };
        Ok(Self {
            package,
            model,
            accelerator,
            evidence,
        })
    }

    #[must_use]
    pub const fn accelerator(&self) -> AcceleratorSelection {
        self.accelerator
    }

    #[must_use]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.model.sample_rate_hz()
    }

    #[must_use]
    pub const fn frame_samples(&self) -> usize {
        self.model.frame_samples()
    }

    #[must_use]
    pub const fn algorithmic_latency_samples(&self) -> usize {
        self.model.algorithmic_latency_samples()
    }

    #[must_use]
    pub const fn flush_samples(&self) -> usize {
        self.model.flush_samples()
    }

    pub fn model_working_set_bytes(&self) -> Result<u64, String> {
        let profile = self
            .package
            .precision_profile_for(self.accelerator.effective())?
            .expect("causal target-speaker packages use v2 precision profiles");
        Ok(profile
            .resources
            .max_session_memory_bytes
            .saturating_add(profile.resources.max_worker_memory_bytes))
    }

    pub fn start(
        &self,
        enrollment: Audio,
        config: CausalTargetSpeakerConfig,
    ) -> Result<CausalTargetSpeakerStream, String> {
        config.validate()?;
        validate_enrollment_audio(&enrollment)?;
        let enrollment_input = (
            enrollment.sample_rate,
            enrollment.channels(),
            enrollment.frames(),
        );
        let enrollment = SensitiveEnrollment::new(enrollment);
        let mono = Zeroizing::new(mono_mix(enrollment.audio())?);
        let resampled = Zeroizing::new(crate::resample::resample(
            &mono,
            enrollment.audio().sample_rate,
            self.model.sample_rate_hz(),
        )?);
        validate_enrollment_duration(resampled.len(), self.model.sample_rate_hz())?;
        let mut model_samples = Zeroizing::new(
            resampled
                .iter()
                .map(|sample| sample.clamp(-1.0, 1.0) as f32)
                .collect::<Vec<_>>(),
        );
        if let Some(required) = self.model.fixed_enrollment_samples() {
            if model_samples.len() != required {
                return Err(format!(
                    "causal target-speaker package requires exactly {required} enrollment samples at {} Hz, got {}",
                    self.model.sample_rate_hz(),
                    model_samples.len()
                ));
            }
        }
        let enrollment_summary = CausalTargetSpeakerEnrollmentSummary {
            input_sample_rate: enrollment_input.0,
            input_channels: enrollment_input.1,
            input_frames: enrollment_input.2,
            model_sample_rate: self.model.sample_rate_hz(),
            model_samples: model_samples.len(),
            mixdown_policy: "arithmetic-mean-mono-v1".into(),
            raw_audio_retained: false,
            embedding_retained: false,
            digest_recorded: false,
        };
        let runtime = self.model.start(std::mem::take(&mut *model_samples))?;
        drop(resampled);
        drop(mono);
        drop(enrollment);
        Ok(CausalTargetSpeakerStream {
            runtime,
            config,
            enrollment_summary,
            frame_samples: self.model.frame_samples(),
            flush_samples: self.model.flush_samples(),
            generation: 1,
            next_frame: 0,
            present_streak: 0,
            finished: false,
        })
    }

    /// Render an ordinary audio object through the causal state machine while
    /// preserving exact source duration. Signed algorithmic latency is removed
    /// only after the authenticated flush tail has been produced.
    pub fn render(
        &self,
        mixture: &Audio,
        enrollment: Audio,
        config: CausalTargetSpeakerConfig,
    ) -> Result<CausalTargetSpeakerRenderResult, String> {
        validate_mixture_audio(mixture)?;
        let source_frames = mixture.frames();
        let mixture_mono = mono_mix(mixture)?;
        let model_f64 = crate::resample::resample(
            &mixture_mono,
            mixture.sample_rate,
            self.model.sample_rate_hz(),
        )?;
        if model_f64.is_empty() {
            return Err("causal target-speaker mixture becomes empty at the model rate".into());
        }
        let model_samples: Vec<f32> = model_f64
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) as f32)
            .collect();
        let mut stream = self.start(enrollment, config)?;
        let enrollment_summary = stream.enrollment_summary().clone();
        let frame_samples = stream.frame_samples();
        let input_blocks = model_samples.len().div_ceil(frame_samples);
        let expected_rendered = input_blocks
            .checked_mul(frame_samples)
            .and_then(|value| value.checked_add(self.model.flush_samples()))
            .ok_or_else(|| "causal target-speaker render size overflow".to_string())?;
        let mut rendered = Vec::new();
        rendered
            .try_reserve_exact(expected_rendered)
            .map_err(|_| "unable to reserve causal target-speaker render".to_string())?;
        let mut frame = vec![0.0_f32; frame_samples];
        let mut decision_counts = CausalTargetSpeakerDecisionCounts::default();
        let mut presence_transitions = 0_u64;
        let mut previous_presence = None;
        for chunk in model_samples.chunks(frame_samples) {
            frame.fill(0.0);
            frame[..chunk.len()].copy_from_slice(chunk);
            let block = stream.process(&frame)?;
            decision_counts.observe(block.decision);
            if previous_presence.is_some_and(|previous| previous != block.presence) {
                presence_transitions += 1;
            }
            previous_presence = Some(block.presence);
            rendered.extend_from_slice(&block.audio);
        }
        let flush = stream.finish()?;
        for block in &flush {
            decision_counts.observe(block.decision);
            if previous_presence.is_some_and(|previous| previous != block.presence) {
                presence_transitions += 1;
            }
            previous_presence = Some(block.presence);
            rendered.extend_from_slice(&block.audio[..block.valid_frames]);
        }
        let latency = self.model.algorithmic_latency_samples();
        let mut aligned = Vec::new();
        aligned
            .try_reserve_exact(model_samples.len())
            .map_err(|_| "unable to reserve aligned causal target-speaker output".to_string())?;
        for index in 0..model_samples.len() {
            aligned.push(rendered.get(latency + index).copied().unwrap_or(0.0));
        }
        let aligned_f64: Vec<f64> = aligned.iter().map(|sample| f64::from(*sample)).collect();
        let resampled = crate::resample::resample(
            &aligned_f64,
            self.model.sample_rate_hz(),
            mixture.sample_rate,
        )?;
        let mut output_samples = Vec::new();
        output_samples
            .try_reserve_exact(source_frames)
            .map_err(|_| "unable to reserve causal target-speaker output".to_string())?;
        output_samples.extend(resampled.iter().copied().take(source_frames));
        output_samples.resize(source_frames, 0.0);
        let output = Audio {
            sample_rate: mixture.sample_rate,
            channels: vec![output_samples],
            bits_per_sample: mixture.bits_per_sample,
            sample_format: mixture.sample_format,
            channel_mask: None,
        };
        let manifest = self
            .package
            .manifest_v2()
            .expect("causal target-speaker session requires package v2");
        let profile = self
            .package
            .precision_profile_for(self.accelerator.effective())?
            .expect("causal target-speaker session selects one precision profile");
        let mut warnings = Vec::new();
        if decision_counts.published_present_blocks == 0 {
            warnings.push(
                "no block passed target-presence and signal gates; the exact-duration output is silence"
                    .into(),
            );
        }
        let report = CausalTargetSpeakerRenderReport {
            schema: CAUSAL_TARGET_SPEAKER_REPORT_SCHEMA.into(),
            schema_version: CAUSAL_TARGET_SPEAKER_SCHEMA_VERSION,
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            network_accessed: false,
            deterministic: true,
            model: CausalTargetSpeakerModelIdentity {
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
            source_sample_rate: mixture.sample_rate,
            source_channels: mixture.channels(),
            source_frames,
            output_channels: 1,
            output_frames: source_frames,
            model_sample_rate: self.model.sample_rate_hz(),
            frame_samples,
            algorithmic_latency_samples: latency,
            flush_samples: self.model.flush_samples(),
            input_blocks: input_blocks as u64,
            flush_blocks: flush.len() as u64,
            decision_counts,
            presence_transitions,
            rendered_audio_published: true,
            mixture_mixdown_policy: "arithmetic-mean-mono-v1".into(),
            mixture_pcm_sha256: pcm_digest(mixture, MIXTURE_PCM_DIGEST_DOMAIN),
            output_pcm_sha256: pcm_digest(&output, OUTPUT_PCM_DIGEST_DOMAIN),
            enrollment: enrollment_summary,
            runtime_speaker_identity_verified: false,
            interferer_leakage_measured_at_runtime: false,
            limitations: causal_limitations(),
            warnings,
        };
        Ok(CausalTargetSpeakerRenderResult {
            audio: output,
            report,
        })
    }
}

fn validate_offline_matrix_binding(
    offline: &SignedTargetSpeakerPromotionEvidence,
    causal: &SignedCausalTargetSpeakerPromotionEvidence,
) -> Result<(), String> {
    if offline.payload.strata.len() != causal.payload.strata.len() {
        return Err(
            "causal target-speaker evidence does not reproduce the offline stratum matrix".into(),
        );
    }
    let offline_strata: BTreeMap<_, _> = offline
        .payload
        .strata
        .iter()
        .map(|stratum| (stratum.id.as_str(), stratum))
        .collect();
    for causal_stratum in &causal.payload.strata {
        let offline_stratum = offline_strata
            .get(causal_stratum.id.as_str())
            .ok_or_else(|| {
                format!(
                    "causal target-speaker evidence stratum {} is absent from offline evidence",
                    causal_stratum.id
                )
            })?;
        if causal_stratum.kind != offline_stratum.kind
            || causal_stratum.offline_cases != offline_stratum.cases
            || causal_stratum.metrics.len() != offline_stratum.metrics.len()
        {
            return Err(format!(
                "causal target-speaker evidence stratum {} does not reproduce offline kind/cases/metrics",
                causal_stratum.id
            ));
        }
        let offline_metrics: BTreeMap<_, _> = offline_stratum
            .metrics
            .iter()
            .map(|metric| (metric.metric.as_str(), metric))
            .collect();
        for causal_metric in &causal_stratum.metrics {
            let offline_metric = offline_metrics
                .get(causal_metric.metric.as_str())
                .ok_or_else(|| {
                    format!(
                        "causal target-speaker evidence metric {} is absent from offline stratum {}",
                        causal_metric.metric, causal_stratum.id
                    )
                })?;
            if causal_metric.operator != offline_metric.operator
                || causal_metric.offline_value != offline_metric.value
                || causal_metric.hard_limit != offline_metric.limit
            {
                return Err(format!(
                    "causal target-speaker evidence metric {} in stratum {} does not reproduce the offline operator/value/limit",
                    causal_metric.metric, causal_stratum.id
                ));
            }
        }
    }
    Ok(())
}

pub struct CausalTargetSpeakerStream {
    runtime: CausalTargetSpeakerRuntime,
    config: CausalTargetSpeakerConfig,
    enrollment_summary: CausalTargetSpeakerEnrollmentSummary,
    frame_samples: usize,
    flush_samples: usize,
    generation: u64,
    next_frame: u64,
    present_streak: u32,
    finished: bool,
}

impl std::fmt::Debug for CausalTargetSpeakerStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CausalTargetSpeakerStream")
            .field("frame_samples", &self.frame_samples)
            .field("flush_samples", &self.flush_samples)
            .field("generation", &self.generation)
            .field("next_frame", &self.next_frame)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl CausalTargetSpeakerStream {
    #[must_use]
    pub const fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    #[must_use]
    pub fn enrollment_summary(&self) -> &CausalTargetSpeakerEnrollmentSummary {
        &self.enrollment_summary
    }

    pub fn process(&mut self, mixture: &[f32]) -> Result<CausalTargetSpeakerBlock, String> {
        if self.finished {
            return Err("causal target-speaker stream has already been finished".into());
        }
        let start_frame = self.next_frame;
        let inference = self.runtime.process(mixture)?;
        self.next_frame = self
            .next_frame
            .checked_add(self.frame_samples as u64)
            .ok_or_else(|| "causal target-speaker frame clock overflow".to_string())?;
        self.classify_block(
            inference.audio,
            inference.presence_probabilities,
            mixture,
            start_frame,
        )
    }

    pub fn reset(&mut self) -> Result<u64, String> {
        self.runtime.reset()?;
        self.generation = self.generation.wrapping_add(1).max(1);
        self.next_frame = 0;
        self.present_streak = 0;
        self.finished = false;
        Ok(self.generation)
    }

    pub fn finish(&mut self) -> Result<Vec<CausalTargetSpeakerBlock>, String> {
        if self.finished {
            return Err("causal target-speaker stream has already been finished".into());
        }
        self.finished = true;
        let blocks = self.flush_samples.div_ceil(self.frame_samples);
        let mut output = Vec::new();
        output
            .try_reserve_exact(blocks)
            .map_err(|_| "unable to reserve causal target-speaker flush blocks".to_string())?;
        let zeros = vec![0.0_f32; self.frame_samples];
        let mut remaining = self.flush_samples;
        for _ in 0..blocks {
            let start_frame = self.next_frame;
            let inference = self.runtime.process(&zeros)?;
            self.next_frame = self
                .next_frame
                .checked_add(self.frame_samples as u64)
                .ok_or_else(|| "causal target-speaker flush clock overflow".to_string())?;
            let valid_frames = remaining.min(self.frame_samples);
            remaining -= valid_frames;
            let presence = classify_presence(inference.presence_probabilities, &self.config);
            if presence == TargetSpeakerPresence::Present {
                self.present_streak = self.present_streak.saturating_add(1);
            } else {
                self.present_streak = 0;
            }
            let safe = inference.audio.iter().all(|sample| {
                sample.is_finite() && f64::from(sample.abs()) <= self.config.maximum_peak
            });
            let accepted = presence == TargetSpeakerPresence::Present
                && self.present_streak >= self.config.present_hold_blocks
                && safe;
            let audio = if accepted {
                inference.audio
            } else {
                vec![0.0; self.frame_samples]
            };
            output.push(CausalTargetSpeakerBlock {
                generation: self.generation,
                start_frame,
                valid_frames,
                audio,
                presence,
                absent_probability: inference.presence_probabilities[0],
                uncertain_probability: inference.presence_probabilities[1],
                present_probability: inference.presence_probabilities[2],
                decision: if accepted {
                    CausalTargetSpeakerBlockDecision::PublishedPresent
                } else {
                    CausalTargetSpeakerBlockDecision::MutedFlush
                },
                candidate_accepted: accepted,
            });
        }
        Ok(output)
    }

    fn classify_block(
        &mut self,
        candidate: Vec<f32>,
        probabilities: [f32; 3],
        mixture: &[f32],
        start_frame: u64,
    ) -> Result<CausalTargetSpeakerBlock, String> {
        let presence = classify_presence(probabilities, &self.config);
        if presence == TargetSpeakerPresence::Present {
            self.present_streak = self.present_streak.saturating_add(1);
        } else {
            self.present_streak = 0;
        }
        let energy_gain = rms_dbfs(&candidate) - rms_dbfs(mixture);
        let safe = candidate.iter().all(|sample| {
            sample.is_finite() && f64::from(sample.abs()) <= self.config.maximum_peak
        }) && energy_gain <= self.config.maximum_energy_gain_db;
        let decision = match presence {
            TargetSpeakerPresence::Absent => CausalTargetSpeakerBlockDecision::MutedAbsent,
            TargetSpeakerPresence::Uncertain => CausalTargetSpeakerBlockDecision::MutedUncertain,
            TargetSpeakerPresence::Present if !safe => {
                CausalTargetSpeakerBlockDecision::MutedSafetyGate
            }
            TargetSpeakerPresence::Present
                if self.present_streak < self.config.present_hold_blocks =>
            {
                CausalTargetSpeakerBlockDecision::MutedPresentWarmup
            }
            TargetSpeakerPresence::Present => CausalTargetSpeakerBlockDecision::PublishedPresent,
        };
        let accepted = decision == CausalTargetSpeakerBlockDecision::PublishedPresent;
        let audio = if accepted {
            candidate
        } else {
            vec![0.0; self.frame_samples]
        };
        Ok(CausalTargetSpeakerBlock {
            generation: self.generation,
            start_frame,
            valid_frames: self.frame_samples,
            audio,
            presence,
            absent_probability: probabilities[0],
            uncertain_probability: probabilities[1],
            present_probability: probabilities[2],
            decision,
            candidate_accepted: accepted,
        })
    }
}

/// Absolute identity assigned to one block before it enters the worker queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalTargetSpeakerRealtimeToken {
    pub generation: u64,
    pub start_frame: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CausalTargetSpeakerRealtimeSubmitError {
    WrongFrameSize,
    NonFiniteInput,
    OutOfRangeInput,
    PoolExhausted,
    QueueFull,
    Stopped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CausalTargetSpeakerRealtimeReceiveError {
    WrongFrameSize,
    WrongGeneration,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausalTargetSpeakerRealtimeResult {
    pub token: CausalTargetSpeakerRealtimeToken,
    pub valid: bool,
    pub presence: TargetSpeakerPresence,
    pub absent_probability: f32,
    pub uncertain_probability: f32,
    pub present_probability: f32,
    pub decision: CausalTargetSpeakerBlockDecision,
    pub candidate_accepted: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CausalTargetSpeakerRealtimeMetrics {
    pub submitted_blocks: u64,
    pub overload_blocks: u64,
    pub late_blocks: u64,
    pub stale_generation_blocks: u64,
    pub invalid_blocks: u64,
    pub worker_errors: u64,
    pub maximum_input_queue_depth: u64,
}

#[derive(Default)]
struct RealtimeMetricAtoms {
    submitted_blocks: AtomicU64,
    overload_blocks: AtomicU64,
    late_blocks: AtomicU64,
    stale_generation_blocks: AtomicU64,
    invalid_blocks: AtomicU64,
    worker_errors: AtomicU64,
    maximum_input_queue_depth: AtomicU64,
}

impl RealtimeMetricAtoms {
    fn snapshot(&self) -> CausalTargetSpeakerRealtimeMetrics {
        CausalTargetSpeakerRealtimeMetrics {
            submitted_blocks: self.submitted_blocks.load(Ordering::Relaxed),
            overload_blocks: self.overload_blocks.load(Ordering::Relaxed),
            late_blocks: self.late_blocks.load(Ordering::Relaxed),
            stale_generation_blocks: self.stale_generation_blocks.load(Ordering::Relaxed),
            invalid_blocks: self.invalid_blocks.load(Ordering::Relaxed),
            worker_errors: self.worker_errors.load(Ordering::Relaxed),
            maximum_input_queue_depth: self.maximum_input_queue_depth.load(Ordering::Relaxed),
        }
    }
}

struct RealtimeBlock {
    token: CausalTargetSpeakerRealtimeToken,
    samples: Box<[f32]>,
    valid: bool,
    presence: TargetSpeakerPresence,
    probabilities: [f32; 3],
    decision: CausalTargetSpeakerBlockDecision,
    candidate_accepted: bool,
}

trait CausalRealtimeProcessor: Send {
    fn frame_samples(&self) -> usize;
    fn generation(&self) -> u64;
    fn process(&mut self, mixture: &[f32]) -> Result<CausalTargetSpeakerBlock, String>;
    fn reset(&mut self) -> Result<u64, String>;
}

impl CausalRealtimeProcessor for CausalTargetSpeakerStream {
    fn frame_samples(&self) -> usize {
        self.frame_samples()
    }

    fn generation(&self) -> u64 {
        self.generation()
    }

    fn process(&mut self, mixture: &[f32]) -> Result<CausalTargetSpeakerBlock, String> {
        self.process(mixture)
    }

    fn reset(&mut self) -> Result<u64, String> {
        self.reset()
    }
}

impl RealtimeBlock {
    fn new(frame_samples: usize) -> Result<Self, String> {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(frame_samples)
            .map_err(|_| "unable to reserve causal target-speaker real-time block".to_string())?;
        samples.resize(frame_samples, 0.0);
        Ok(Self {
            token: CausalTargetSpeakerRealtimeToken {
                generation: 1,
                start_frame: 0,
            },
            samples: samples.into_boxed_slice(),
            valid: false,
            presence: TargetSpeakerPresence::Uncertain,
            probabilities: [0.0, 1.0, 0.0],
            decision: CausalTargetSpeakerBlockDecision::MutedSafetyGate,
            candidate_accepted: false,
        })
    }
}

/// Bounded off-callback inference bridge.
///
/// Construction and destruction are control-thread operations. `try_submit`,
/// `try_receive_due`, and `reset` allocate no memory, acquire no mutex, perform
/// no I/O, and never wait for inference. The caller must render silence when a
/// due result is unavailable or invalid.
pub struct CausalTargetSpeakerRealtimeScheduler {
    frame_samples: usize,
    generation: u64,
    next_submit_frame: u64,
    input: Arc<ArrayQueue<RealtimeBlock>>,
    output: Arc<ArrayQueue<RealtimeBlock>>,
    free: Arc<ArrayQueue<RealtimeBlock>>,
    pending_output: Option<RealtimeBlock>,
    running: Arc<AtomicBool>,
    metrics: Arc<RealtimeMetricAtoms>,
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for CausalTargetSpeakerRealtimeScheduler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CausalTargetSpeakerRealtimeScheduler")
            .field("frame_samples", &self.frame_samples)
            .field("generation", &self.generation)
            .field("next_submit_frame", &self.next_submit_frame)
            .field("metrics", &self.metrics())
            .finish_non_exhaustive()
    }
}

impl CausalTargetSpeakerRealtimeScheduler {
    pub fn new(stream: CausalTargetSpeakerStream) -> Result<Self, String> {
        Self::new_with_processor(Box::new(stream))
    }

    fn new_with_processor(processor: Box<dyn CausalRealtimeProcessor>) -> Result<Self, String> {
        let frame_samples = processor.frame_samples();
        let generation = processor.generation();
        let input = Arc::new(ArrayQueue::new(REALTIME_QUEUE_BLOCKS));
        let output = Arc::new(ArrayQueue::new(REALTIME_QUEUE_BLOCKS));
        let free = Arc::new(ArrayQueue::new(REALTIME_POOL_BLOCKS));
        for _ in 0..REALTIME_POOL_BLOCKS {
            free.push(RealtimeBlock::new(frame_samples)?)
                .map_err(|_| "causal target-speaker real-time pool initialization failed")?;
        }
        let running = Arc::new(AtomicBool::new(true));
        let metrics = Arc::new(RealtimeMetricAtoms::default());
        let worker_input = Arc::clone(&input);
        let worker_output = Arc::clone(&output);
        let worker_free = Arc::clone(&free);
        let worker_running = Arc::clone(&running);
        let worker_metrics = Arc::clone(&metrics);
        let worker = thread::Builder::new()
            .name("denoize-causal-target-speaker".into())
            .spawn(move || {
                realtime_worker_loop(
                    processor,
                    worker_input,
                    worker_output,
                    worker_free,
                    worker_running,
                    worker_metrics,
                );
            })
            .map_err(|error| format!("start causal target-speaker worker: {error}"))?;
        Ok(Self {
            frame_samples,
            generation,
            next_submit_frame: 0,
            input,
            output,
            free,
            pending_output: None,
            running,
            metrics,
            worker: Some(worker),
        })
    }

    #[must_use]
    pub const fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Copy one exact model frame into the bounded worker queue. A queue/pool
    /// failure advances the absolute clock so later results cannot be mistaken
    /// for the missed frame.
    pub fn try_submit(
        &mut self,
        mixture: &[f32],
    ) -> Result<CausalTargetSpeakerRealtimeToken, CausalTargetSpeakerRealtimeSubmitError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(CausalTargetSpeakerRealtimeSubmitError::Stopped);
        }
        if mixture.len() != self.frame_samples {
            return Err(CausalTargetSpeakerRealtimeSubmitError::WrongFrameSize);
        }
        if mixture.iter().any(|sample| !sample.is_finite()) {
            return Err(CausalTargetSpeakerRealtimeSubmitError::NonFiniteInput);
        }
        if mixture.iter().any(|sample| !(-1.0..=1.0).contains(sample)) {
            return Err(CausalTargetSpeakerRealtimeSubmitError::OutOfRangeInput);
        }
        let token = CausalTargetSpeakerRealtimeToken {
            generation: self.generation,
            start_frame: self.next_submit_frame,
        };
        self.next_submit_frame = self
            .next_submit_frame
            .wrapping_add(self.frame_samples as u64);
        let Some(mut block) = self.free.pop() else {
            self.metrics.overload_blocks.fetch_add(1, Ordering::Relaxed);
            return Err(CausalTargetSpeakerRealtimeSubmitError::PoolExhausted);
        };
        block.token = token;
        block.samples.copy_from_slice(mixture);
        block.valid = false;
        match self.input.push(block) {
            Ok(()) => {
                self.metrics
                    .submitted_blocks
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .maximum_input_queue_depth
                    .fetch_max(self.input.len() as u64, Ordering::Relaxed);
                Ok(token)
            }
            Err(block) => {
                self.metrics.overload_blocks.fetch_add(1, Ordering::Relaxed);
                recycle_realtime_block(&self.free, block);
                Err(CausalTargetSpeakerRealtimeSubmitError::QueueFull)
            }
        }
    }

    /// Copy the result for exactly `token` into caller-owned storage. Older or
    /// prior-generation results are destroyed; a future result is retained in
    /// one preallocated pending slot.
    pub fn try_receive_due(
        &mut self,
        token: CausalTargetSpeakerRealtimeToken,
        destination: &mut [f32],
    ) -> Result<Option<CausalTargetSpeakerRealtimeResult>, CausalTargetSpeakerRealtimeReceiveError>
    {
        if destination.len() != self.frame_samples {
            return Err(CausalTargetSpeakerRealtimeReceiveError::WrongFrameSize);
        }
        if token.generation != self.generation {
            return Err(CausalTargetSpeakerRealtimeReceiveError::WrongGeneration);
        }
        loop {
            let Some(block) = self.pending_output.take().or_else(|| self.output.pop()) else {
                return Ok(None);
            };
            if block.token.generation != token.generation {
                self.metrics
                    .stale_generation_blocks
                    .fetch_add(1, Ordering::Relaxed);
                recycle_realtime_block(&self.free, block);
                continue;
            }
            if block.token.start_frame < token.start_frame {
                self.metrics.late_blocks.fetch_add(1, Ordering::Relaxed);
                recycle_realtime_block(&self.free, block);
                continue;
            }
            if block.token.start_frame > token.start_frame {
                self.pending_output = Some(block);
                return Ok(None);
            }
            if block.valid {
                destination.copy_from_slice(&block.samples);
            } else {
                destination.fill(0.0);
                self.metrics.invalid_blocks.fetch_add(1, Ordering::Relaxed);
            }
            let result = CausalTargetSpeakerRealtimeResult {
                token: block.token,
                valid: block.valid,
                presence: block.presence,
                absent_probability: block.probabilities[0],
                uncertain_probability: block.probabilities[1],
                present_probability: block.probabilities[2],
                decision: block.decision,
                candidate_accepted: block.candidate_accepted,
            };
            recycle_realtime_block(&self.free, block);
            return Ok(Some(result));
        }
    }

    /// Invalidate every queued result and begin a new zero-state generation.
    /// The worker observes the generation on the next submitted block.
    pub fn reset(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1).max(1);
        self.next_submit_frame = 0;
        if let Some(block) = self.pending_output.take() {
            self.metrics
                .stale_generation_blocks
                .fetch_add(1, Ordering::Relaxed);
            recycle_realtime_block(&self.free, block);
        }
        while let Some(block) = self.input.pop() {
            self.metrics
                .stale_generation_blocks
                .fetch_add(1, Ordering::Relaxed);
            recycle_realtime_block(&self.free, block);
        }
        while let Some(block) = self.output.pop() {
            self.metrics
                .stale_generation_blocks
                .fetch_add(1, Ordering::Relaxed);
            recycle_realtime_block(&self.free, block);
        }
        self.generation
    }

    #[must_use]
    pub fn metrics(&self) -> CausalTargetSpeakerRealtimeMetrics {
        self.metrics.snapshot()
    }

    /// Stop and join the permanent worker. Call only from a control thread.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for CausalTargetSpeakerRealtimeScheduler {
    fn drop(&mut self) {
        self.stop();
    }
}

fn realtime_worker_loop(
    mut processor: Box<dyn CausalRealtimeProcessor>,
    input: Arc<ArrayQueue<RealtimeBlock>>,
    output: Arc<ArrayQueue<RealtimeBlock>>,
    free: Arc<ArrayQueue<RealtimeBlock>>,
    running: Arc<AtomicBool>,
    metrics: Arc<RealtimeMetricAtoms>,
) {
    let mut generation = processor.generation();
    let mut next_frame = 0_u64;
    let mut pending = None;
    while running.load(Ordering::Acquire) {
        if let Some(block) = pending.take() {
            match output.push(block) {
                Ok(()) => continue,
                Err(block) => {
                    pending = Some(block);
                    thread::park_timeout(REALTIME_WORKER_POLL);
                    continue;
                }
            }
        }
        let Some(mut block) = input.pop() else {
            thread::park_timeout(REALTIME_WORKER_POLL);
            continue;
        };
        if block.token.generation != generation || block.token.start_frame != next_frame {
            generation = block.token.generation;
            if processor.reset().is_err() {
                metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        next_frame = block
            .token
            .start_frame
            .wrapping_add(block.samples.len() as u64);
        match processor.process(&block.samples) {
            Ok(result) => {
                block.samples.copy_from_slice(&result.audio);
                block.valid = true;
                block.presence = result.presence;
                block.probabilities = [
                    result.absent_probability,
                    result.uncertain_probability,
                    result.present_probability,
                ];
                block.decision = result.decision;
                block.candidate_accepted = result.candidate_accepted;
            }
            Err(_) => {
                block.samples.fill(0.0);
                block.valid = false;
                block.presence = TargetSpeakerPresence::Uncertain;
                block.probabilities = [0.0, 1.0, 0.0];
                block.decision = CausalTargetSpeakerBlockDecision::MutedSafetyGate;
                block.candidate_accepted = false;
                metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
                if processor.reset().is_err() {
                    metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        match output.push(block) {
            Ok(()) => {}
            Err(block) => pending = Some(block),
        }
    }
    if let Some(block) = pending {
        recycle_realtime_block(&free, block);
    }
}

#[inline]
fn recycle_realtime_block(queue: &ArrayQueue<RealtimeBlock>, block: RealtimeBlock) {
    if let Err(block) = queue.push(block) {
        // The fixed-pool ownership invariant makes this unreachable. Avoid a
        // deallocation on a real-time caller even if that invariant is broken.
        std::mem::forget(block);
    }
}

struct SensitiveEnrollment(Audio);

impl SensitiveEnrollment {
    fn new(audio: Audio) -> Self {
        Self(audio)
    }

    fn audio(&self) -> &Audio {
        &self.0
    }
}

impl Drop for SensitiveEnrollment {
    fn drop(&mut self) {
        for channel in &mut self.0.channels {
            channel.zeroize();
        }
    }
}

fn validate_metric(
    metric: &CausalTargetSpeakerMetricEvidence,
    policy: &MetricPolicy,
) -> Result<(), String> {
    let hard_limit_is_strong_enough = match policy.operator {
        TargetSpeakerMetricOperator::GreaterOrEqual => metric.hard_limit >= policy.hard_limit,
        TargetSpeakerMetricOperator::LessOrEqual => metric.hard_limit <= policy.hard_limit,
    };
    if metric.operator != policy.operator
        || !metric.offline_value.is_finite()
        || !metric.causal_value.is_finite()
        || !metric.hard_limit.is_finite()
        || !metric.maximum_regression.is_finite()
        || metric.maximum_regression < 0.0
        || metric.maximum_regression > policy.maximum_regression
        || !hard_limit_is_strong_enough
    {
        return Err(format!(
            "causal target-speaker metric {} has an invalid policy or value",
            metric.metric
        ));
    }
    let (offline_hard, causal_hard, non_inferior) = match policy.operator {
        TargetSpeakerMetricOperator::GreaterOrEqual => (
            metric.offline_value >= metric.hard_limit,
            metric.causal_value >= metric.hard_limit,
            metric.causal_value >= metric.offline_value - metric.maximum_regression,
        ),
        TargetSpeakerMetricOperator::LessOrEqual => (
            metric.offline_value <= metric.hard_limit,
            metric.causal_value <= metric.hard_limit,
            metric.causal_value <= metric.offline_value + metric.maximum_regression,
        ),
    };
    if metric.passed != (offline_hard && causal_hard && non_inferior) {
        return Err(format!(
            "causal target-speaker metric {} has an inconsistent passed flag",
            metric.metric
        ));
    }
    Ok(())
}

fn validate_enrollment_audio(audio: &Audio) -> Result<(), String> {
    if audio.sample_rate == 0 || audio.sample_rate > 768_000 {
        return Err("causal target-speaker enrollment sample rate is invalid".into());
    }
    if audio.channels.is_empty() || audio.channels.len() > MAX_CHANNELS {
        return Err(format!(
            "causal target-speaker enrollment channels must be in 1..={MAX_CHANNELS}"
        ));
    }
    let frames = audio.frames();
    if frames == 0 || audio.channels.iter().any(|channel| channel.len() != frames) {
        return Err(
            "causal target-speaker enrollment must be non-empty with equal channel lengths".into(),
        );
    }
    if audio
        .channels
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(
            "causal target-speaker enrollment contains an invalid normalized sample".into(),
        );
    }
    Ok(())
}

fn validate_mixture_audio(audio: &Audio) -> Result<(), String> {
    if audio.sample_rate == 0 || audio.sample_rate > 768_000 {
        return Err("causal target-speaker mixture sample rate is invalid".into());
    }
    if audio.channels.is_empty() || audio.channels.len() > MAX_CHANNELS {
        return Err(format!(
            "causal target-speaker mixture channels must be in 1..={MAX_CHANNELS}"
        ));
    }
    let frames = audio.frames();
    if frames == 0 || audio.channels.iter().any(|channel| channel.len() != frames) {
        return Err(
            "causal target-speaker mixture must be non-empty with equal channel lengths".into(),
        );
    }
    if frames as u64
        > u64::from(audio.sample_rate).saturating_mul(MAX_TARGET_SPEAKER_MIXTURE_SECONDS)
    {
        return Err(format!(
            "causal target-speaker mixture exceeds the {MAX_TARGET_SPEAKER_MIXTURE_SECONDS}-second limit"
        ));
    }
    if audio
        .channels
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err("causal target-speaker mixture contains an invalid normalized sample".into());
    }
    Ok(())
}

fn mono_mix(audio: &Audio) -> Result<Vec<f64>, String> {
    let frames = audio.frames();
    let mut mono = Vec::new();
    mono.try_reserve_exact(frames)
        .map_err(|_| "unable to reserve causal target-speaker enrollment mono mix".to_string())?;
    let scale = 1.0 / audio.channels() as f64;
    for frame in 0..frames {
        mono.push(
            audio
                .channels
                .iter()
                .map(|channel| channel[frame])
                .sum::<f64>()
                * scale,
        );
    }
    Ok(mono)
}

fn validate_enrollment_duration(samples: usize, sample_rate: u32) -> Result<(), String> {
    let millis = (samples as u64)
        .saturating_mul(1000)
        .checked_div(u64::from(sample_rate))
        .unwrap_or(0);
    if !(MIN_TARGET_SPEAKER_ENROLLMENT_MILLIS..=MAX_TARGET_SPEAKER_ENROLLMENT_MILLIS)
        .contains(&millis)
    {
        return Err(format!(
            "causal target-speaker enrollment must be {MIN_TARGET_SPEAKER_ENROLLMENT_MILLIS}..={MAX_TARGET_SPEAKER_ENROLLMENT_MILLIS} ms after resampling, got {millis} ms"
        ));
    }
    Ok(())
}

fn classify_presence(
    probabilities: [f32; 3],
    config: &CausalTargetSpeakerConfig,
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

fn rms_dbfs(samples: &[f32]) -> f64 {
    let energy = samples.iter().fold(0.0, |sum, sample| {
        sum + f64::from(*sample) * f64::from(*sample)
    });
    amplitude_dbfs((energy / samples.len().max(1) as f64).sqrt())
}

fn amplitude_dbfs(amplitude: f64) -> f64 {
    (20.0 * amplitude.max(SILENCE_FLOOR).log10()).clamp(-240.0, 240.0)
}

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

fn causal_limitations() -> Vec<String> {
    vec![
        "the runtime presence head is not an independent speaker-verification system".into(),
        "muted blocks preserve continuous timing with silence; they do not prove the target was absent"
            .into(),
        "interferer leakage and target identity are promotion-time measurements, not runtime measurements"
            .into(),
        "a valid evidence signature authenticates the evaluator's claim but cannot prove recordings, labels, consent, or benchmark independence"
            .into(),
        "the causal v1 adapter mixes program channels to mono and does not preserve spatial position"
            .into(),
        "enrollment buffers are zeroized on ordinary drop, but runtime copies, operating-system caches, allocator remnants, swap, and crash dumps are outside this guarantee"
            .into(),
        "denoize does not bundle a causal target-speaker checkpoint until artifact-level redistribution and every promotion gate pass independently"
            .into(),
    ]
}

fn validate_range(label: &str, value: f64, minimum: f64, maximum: f64) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        Err(format!(
            "causal target-speaker {label} must be finite and in {minimum}..={maximum}"
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
            "causal target-speaker evidence {label} must be lowercase SHA-256"
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

    struct FakeRealtimeProcessor {
        frame_samples: usize,
        generation: u64,
        next_frame: u64,
    }

    impl CausalRealtimeProcessor for FakeRealtimeProcessor {
        fn frame_samples(&self) -> usize {
            self.frame_samples
        }

        fn generation(&self) -> u64 {
            self.generation
        }

        fn process(&mut self, mixture: &[f32]) -> Result<CausalTargetSpeakerBlock, String> {
            let start_frame = self.next_frame;
            self.next_frame += self.frame_samples as u64;
            Ok(CausalTargetSpeakerBlock {
                generation: self.generation,
                start_frame,
                valid_frames: self.frame_samples,
                audio: mixture.to_vec(),
                presence: TargetSpeakerPresence::Present,
                absent_probability: 0.0,
                uncertain_probability: 0.0,
                present_probability: 1.0,
                decision: CausalTargetSpeakerBlockDecision::PublishedPresent,
                candidate_accepted: true,
            })
        }

        fn reset(&mut self) -> Result<u64, String> {
            self.generation = self.generation.wrapping_add(1).max(1);
            self.next_frame = 0;
            Ok(self.generation)
        }
    }

    #[test]
    fn causal_evidence_requires_non_inferiority_latency_and_callback_safety() {
        let payload = passing_payload();
        payload.validate().unwrap();

        let mut regression = payload.clone();
        let metric = regression
            .strata
            .iter_mut()
            .find(|stratum| stratum.kind == TargetSpeakerStratumKind::TargetPresent)
            .unwrap()
            .metrics
            .iter_mut()
            .find(|metric| metric.metric == "speaker.target-similarity")
            .unwrap();
        metric.causal_value = metric.offline_value - metric.maximum_regression - 0.001;
        assert!(regression.validate().unwrap_err().contains("passed flag"));

        let mut stricter = passing_payload();
        let metric = stricter
            .strata
            .iter_mut()
            .find(|stratum| stratum.kind == TargetSpeakerStratumKind::TargetPresent)
            .unwrap()
            .metrics
            .iter_mut()
            .find(|metric| metric.metric == "speaker.target-similarity")
            .unwrap();
        metric.hard_limit = 0.71;
        metric.offline_value = 0.71;
        metric.causal_value = 0.71;
        stricter.validate().unwrap();

        let mut latency = payload.clone();
        latency.effective_latency_milliseconds = 100.01;
        assert!(latency.validate().unwrap_err().contains("accepted flag"));

        let mut callback = payload;
        callback.realtime.callback_inference_calls = 1;
        assert!(callback.validate().unwrap_err().contains("accepted flag"));

        let mut oversized = passing_payload();
        oversized.realtime.callback_allocations = JSON_SAFE_INTEGER_MAX + 1;
        oversized.accepted = false;
        assert!(oversized.validate().unwrap_err().contains("schema bounds"));

        let mut geometry = passing_payload();
        geometry.algorithmic_latency_samples = 1_601;
        geometry.flush_samples = 1_601;
        assert!(geometry.validate().unwrap_err().contains("accepted flag"));
    }

    #[test]
    fn block_config_is_closed_and_fail_closed() {
        let config = CausalTargetSpeakerConfig::default();
        config.validate().unwrap();
        assert_eq!(config.present_hold_blocks, 3);
        assert_eq!(
            classify_presence([0.33, 0.33, 0.34], &config),
            TargetSpeakerPresence::Uncertain
        );
        let json = serde_json::to_string(&config).unwrap();
        assert!(serde_json::from_str::<CausalTargetSpeakerConfig>(
            &json.replace('{', "{\"unknown\":true,")
        )
        .is_err());
    }

    #[test]
    fn realtime_scheduler_is_bounded_and_discards_late_and_stale_results() {
        let processor = FakeRealtimeProcessor {
            frame_samples: 4,
            generation: 1,
            next_frame: 0,
        };
        let mut scheduler =
            CausalTargetSpeakerRealtimeScheduler::new_with_processor(Box::new(processor)).unwrap();
        assert_eq!(
            scheduler.try_submit(&[0.0; 3]),
            Err(CausalTargetSpeakerRealtimeSubmitError::WrongFrameSize)
        );
        assert_eq!(
            scheduler.try_submit(&[0.0, f32::NAN, 0.0, 0.0]),
            Err(CausalTargetSpeakerRealtimeSubmitError::NonFiniteInput)
        );
        assert_eq!(
            scheduler.try_submit(&[0.0, 1.01, 0.0, 0.0]),
            Err(CausalTargetSpeakerRealtimeSubmitError::OutOfRangeInput)
        );

        let first = scheduler.try_submit(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        let mut destination = [0.0_f32; 4];
        let first_result = wait_for_result(&mut scheduler, first, &mut destination);
        assert!(first_result.valid);
        assert_eq!(destination, [0.1, 0.2, 0.3, 0.4]);

        let late = scheduler.try_submit(&[0.5; 4]).unwrap();
        let future = CausalTargetSpeakerRealtimeToken {
            generation: late.generation,
            start_frame: late.start_frame + 4,
        };
        for _ in 0..200 {
            let _ = scheduler.try_receive_due(future, &mut destination).unwrap();
            if scheduler.metrics().late_blocks > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(scheduler.metrics().late_blocks, 1);

        let _old = scheduler.try_submit(&[0.6; 4]).unwrap();
        let generation = scheduler.reset();
        let current = scheduler.try_submit(&[0.7; 4]).unwrap();
        assert_eq!(current.generation, generation);
        let current_result = wait_for_result(&mut scheduler, current, &mut destination);
        assert!(current_result.valid);
        assert_eq!(destination, [0.7; 4]);
        assert!(scheduler.metrics().stale_generation_blocks >= 1);
        assert_eq!(scheduler.metrics().worker_errors, 0);
        scheduler.stop();
    }

    #[test]
    fn prepare_requires_two_valid_signatures_and_exact_package_binding() {
        let (_directory, package) = crate::backend::causal_target_speaker::tests::fixture_package();
        let package_sha256 = package.package_sha256().to_string();
        let mut offline_payload = offline_passing_payload();
        offline_payload.model_package_sha256 = package_sha256.clone();
        offline_payload.source_sha256 = "0".repeat(64);
        offline_payload.checkpoint_sha256 = "1".repeat(64);
        let (offline_secret, offline_public) = crate::generate_receipt_keypair().unwrap();
        let offline =
            crate::sign_target_speaker_promotion_evidence(offline_payload.clone(), &offline_secret)
                .unwrap();

        let mut causal_payload = passing_payload();
        causal_payload.model_package_sha256 = package_sha256;
        causal_payload.source_sha256 = "0".repeat(64);
        causal_payload.checkpoint_sha256 = "1".repeat(64);
        causal_payload.offline_evaluation_result_sha256 =
            offline_payload.evaluation_result_sha256.clone();
        causal_payload.model_sample_rate_hz = 16_000;
        causal_payload.frame_samples = 4;
        causal_payload.algorithmic_latency_samples = 4;
        causal_payload.flush_samples = 4;
        let (causal_secret, causal_public) = crate::generate_receipt_keypair().unwrap();
        let causal =
            sign_causal_target_speaker_promotion_evidence(causal_payload.clone(), &causal_secret)
                .unwrap();
        let session = CausalTargetSpeakerSession::prepare(
            package.clone(),
            &offline,
            &offline_public,
            &causal,
            &causal_public,
            AcceleratorPreference::Cpu,
        )
        .unwrap();
        assert_eq!(session.frame_samples(), 4);

        let mut matrix_payload = causal_payload.clone();
        matrix_payload.strata[0].offline_cases += 1;
        let mismatched_matrix =
            sign_causal_target_speaker_promotion_evidence(matrix_payload, &causal_secret).unwrap();
        assert!(CausalTargetSpeakerSession::prepare(
            package.clone(),
            &offline,
            &offline_public,
            &mismatched_matrix,
            &causal_public,
            AcceleratorPreference::Cpu,
        )
        .unwrap_err()
        .contains("does not reproduce"));

        causal_payload.model_package_sha256 = "f".repeat(64);
        let mismatched =
            sign_causal_target_speaker_promotion_evidence(causal_payload, &causal_secret).unwrap();
        assert!(CausalTargetSpeakerSession::prepare(
            package,
            &offline,
            &offline_public,
            &mismatched,
            &causal_public,
            AcceleratorPreference::Cpu,
        )
        .unwrap_err()
        .contains("does not match"));
    }

    #[test]
    fn render_preserves_source_geometry_and_emits_private_report() {
        let (_directory, package) = crate::backend::causal_target_speaker::tests::fixture_package();
        let model = CausalTargetSpeakerModel::load_runtime_package(
            &package,
            crate::AcceleratorRuntime::Cpu,
        )
        .unwrap();
        let session = CausalTargetSpeakerSession {
            package,
            model,
            accelerator: AcceleratorSelection::default(),
            evidence: CausalTargetSpeakerEvidenceIdentity {
                offline_signing_key_id: "3".repeat(64),
                causal_signing_key_id: "4".repeat(64),
                offline_evaluation_result_sha256: "5".repeat(64),
                causal_evaluation_result_sha256: "6".repeat(64),
                state_reset_flush_result_sha256: "7".repeat(64),
                latency_result_sha256: "8".repeat(64),
                realtime_callback_result_sha256: "9".repeat(64),
                transition_result_sha256: "a".repeat(64),
                strata: REQUIRED_STRATA.len() as u32,
                accepted: true,
            },
        };
        let mixture = Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let enrollment = Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.1; 8_000]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let result = session
            .render(
                &mixture,
                enrollment,
                CausalTargetSpeakerConfig {
                    present_hold_blocks: 1,
                    ..CausalTargetSpeakerConfig::default()
                },
            )
            .unwrap();

        assert_eq!(result.audio.sample_rate, mixture.sample_rate);
        assert_eq!(result.audio.channels(), 1);
        assert_eq!(result.audio.frames(), mixture.frames());
        assert!(result.audio.channels[0]
            .iter()
            .all(|sample| sample.is_finite()));
        assert_eq!(result.report.schema, CAUSAL_TARGET_SPEAKER_REPORT_SCHEMA);
        assert_eq!(result.report.source_frames, mixture.frames());
        assert_eq!(result.report.output_frames, mixture.frames());
        assert_eq!(result.report.algorithmic_latency_samples, 4);
        assert_eq!(result.report.flush_samples, 4);
        assert!(result.report.decision_counts.published_present_blocks > 0);
        assert!(!result.report.enrollment.raw_audio_retained);
        assert!(!result.report.enrollment.embedding_retained);
        assert!(!result.report.enrollment.digest_recorded);
        let report = result.report.to_json().unwrap();
        assert!(!report.contains("enrollment_path"));
        assert!(!report.contains("8000d"));
    }

    fn wait_for_result(
        scheduler: &mut CausalTargetSpeakerRealtimeScheduler,
        token: CausalTargetSpeakerRealtimeToken,
        destination: &mut [f32],
    ) -> CausalTargetSpeakerRealtimeResult {
        for _ in 0..500 {
            if let Some(result) = scheduler.try_receive_due(token, destination).unwrap() {
                return result;
            }
            thread::sleep(Duration::from_millis(1));
        }
        panic!("causal target-speaker worker did not return a bounded test block")
    }

    fn passing_payload() -> CausalTargetSpeakerPromotionEvidencePayload {
        let strata = REQUIRED_STRATA
            .iter()
            .map(|(id, kind)| {
                let policies = match kind {
                    TargetSpeakerStratumKind::TargetPresent => PRESENT_METRICS,
                    TargetSpeakerStratumKind::TargetAbsent => ABSENT_METRICS,
                };
                CausalTargetSpeakerStratumEvidence {
                    id: (*id).into(),
                    kind: *kind,
                    offline_cases: 10,
                    causal_cases: 10,
                    metrics: policies
                        .iter()
                        .map(|policy| CausalTargetSpeakerMetricEvidence {
                            metric: policy.name.into(),
                            operator: policy.operator,
                            offline_value: policy.hard_limit,
                            causal_value: policy.hard_limit,
                            hard_limit: policy.hard_limit,
                            maximum_regression: policy.maximum_regression,
                            passed: true,
                        })
                        .collect(),
                }
            })
            .collect();
        CausalTargetSpeakerPromotionEvidencePayload {
            completed_at_unix_seconds: 1_800_000_000,
            model_package_sha256: "0".repeat(64),
            source_revision: "0123456789abcdef".into(),
            source_sha256: "1".repeat(64),
            checkpoint_sha256: "2".repeat(64),
            offline_evaluation_result_sha256: "3".repeat(64),
            causal_evaluation_result_sha256: "4".repeat(64),
            state_reset_flush_result_sha256: "5".repeat(64),
            latency_result_sha256: "6".repeat(64),
            realtime_callback_result_sha256: "7".repeat(64),
            transition_result_sha256: "8".repeat(64),
            strata,
            model_sample_rate_hz: 16_000,
            frame_samples: 160,
            algorithmic_latency_samples: 1_440,
            flush_samples: 1_440,
            perturbation_latency_cases: 100,
            effective_latency_milliseconds: 90.0,
            effective_latency_limit_milliseconds: 100.0,
            realtime: CausalTargetSpeakerRealtimeAudit {
                paced_blocks: 10_000,
                deadline_misses: 0,
                overload_blocks: 0,
                queue_capacity_blocks: 16,
                maximum_queue_depth_blocks: 15,
                callback_allocations: 0,
                callback_locks: 0,
                callback_waits: 0,
                callback_file_io_operations: 0,
                callback_network_operations: 0,
                callback_log_operations: 0,
                callback_inference_calls: 0,
            },
            transitions: CausalTargetSpeakerTransitionAudit {
                absent_to_present_cases: 100,
                present_to_absent_cases: 100,
                uncertain_transition_cases: 100,
                enrollment_mismatch_cases: 100,
                reference_loss_cases: 100,
                late_results_injected: 100,
                late_results_discarded: 100,
                stale_generation_results_injected: 100,
                stale_generation_results_discarded: 100,
                false_attribution_publications: 0,
            },
            accepted: true,
        }
    }

    fn offline_passing_payload() -> crate::TargetSpeakerPromotionEvidencePayload {
        let strata = REQUIRED_STRATA
            .iter()
            .map(|(id, kind)| {
                let policies = match kind {
                    TargetSpeakerStratumKind::TargetPresent => PRESENT_METRICS,
                    TargetSpeakerStratumKind::TargetAbsent => ABSENT_METRICS,
                };
                crate::TargetSpeakerStratumEvidence {
                    id: (*id).into(),
                    kind: *kind,
                    cases: 10,
                    metrics: policies
                        .iter()
                        .map(|policy| crate::TargetSpeakerMetricOutcome {
                            metric: policy.name.into(),
                            value: policy.hard_limit,
                            operator: policy.operator,
                            limit: policy.hard_limit,
                            passed: true,
                        })
                        .collect(),
                }
            })
            .collect();
        crate::TargetSpeakerPromotionEvidencePayload {
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
}
