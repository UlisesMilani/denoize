//! Typed far-end-reference acoustic echo cancellation.
//!
//! The promoted baseline deliberately keeps alignment and the linear echo path
//! inspectable: a signed delay estimate feeds a partitioned frequency-domain
//! normalized-LMS filter, a double-talk controller freezes unsafe adaptation,
//! and a conservative residual suppressor may only attenuate the linear error.
//! A missing or low-confidence reference returns the microphone unchanged.

use crate::audio::Audio;
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::path::Path;
use std::sync::Arc;

pub const AEC_REPORT_SCHEMA: &str = "denoize-aec-report-v1";
pub const AEC_PROMOTION_EVIDENCE_SCHEMA: &str = "denoize-aec-promotion-evidence-v1";
pub const AEC_SCHEMA_VERSION: u32 = 1;

const IMPLEMENTATION_ID: &str = "native-pfdnlms-v1";
const EVIDENCE_SIGNATURE_DOMAIN: &[u8] = b"denoize-aec-promotion-evidence-v1";
const CONFIG_DIGEST_DOMAIN: &[u8] = b"denoize-aec-config-v1\0";
const MICROPHONE_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-aec-microphone-pcm-v1\0";
const REFERENCE_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-aec-reference-pcm-v1\0";
const OUTPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-aec-output-pcm-v1\0";
const MAX_EVIDENCE_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EVIDENCE_STRATA: usize = 128;
const MAX_EVIDENCE_METRICS: usize = 32;
const MAX_AUDIO_SECONDS: u64 = 3_600;
const JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const POWER_FLOOR: f32 = 1.0e-12;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecConfig {
    pub sample_rate: u32,
    pub block_size_samples: usize,
    pub tail_samples: usize,
    pub maximum_delay_samples: usize,
    pub delay_analysis_samples: usize,
    pub minimum_delay_confidence: f32,
    pub adaptation_rate: f32,
    pub filter_leakage: f32,
    pub adaptation_regularization: f32,
    pub double_talk_correlation_threshold: f32,
    pub reference_activation_rms: f32,
    pub residual_suppression: f32,
    pub minimum_far_end_gain: f32,
    pub minimum_double_talk_gain: f32,
    pub maximum_peak: f32,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            block_size_samples: 256,
            tail_samples: 24_000,
            maximum_delay_samples: 48_000,
            delay_analysis_samples: 144_000,
            minimum_delay_confidence: 0.10,
            adaptation_rate: 0.15,
            filter_leakage: 0.9999,
            adaptation_regularization: 1.0e-3,
            double_talk_correlation_threshold: 0.18,
            reference_activation_rms: 1.0e-3,
            residual_suppression: 0.60,
            minimum_far_end_gain: 0.08,
            minimum_double_talk_gain: 0.85,
            maximum_peak: 1.0,
        }
    }
}

impl AecConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) = crate::input::open_regular_file(path, "AEC configuration")?;
        const MAX_CONFIG_BYTES: u64 = 64 * 1024;
        if length >= MAX_CONFIG_BYTES {
            return Err(format!(
                "AEC configuration {} exceeds the {MAX_CONFIG_BYTES}-byte limit",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve AEC configuration JSON".to_string())?;
        file.take(MAX_CONFIG_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read AEC configuration: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("AEC configuration changed while reading".into());
        }
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse AEC configuration: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(8_000..=192_000).contains(&self.sample_rate) {
            return Err("AEC sample_rate must be in 8000..=192000".into());
        }
        if !self.block_size_samples.is_power_of_two()
            || self.block_size_samples < 64
            || self.block_size_samples > 4_096
        {
            return Err("AEC block_size_samples must be a power of two in 64..=4096".into());
        }
        if self.algorithmic_plus_buffering_milliseconds() > 20.0 {
            return Err(
                "AEC algorithmic-plus-buffering latency must not exceed 20 milliseconds".into(),
            );
        }
        let maximum_tail = (self.sample_rate as usize)
            .checked_mul(2)
            .ok_or("AEC tail bound overflow")?;
        if self.tail_samples < self.block_size_samples || self.tail_samples > maximum_tail {
            return Err(
                "AEC tail_samples must cover one block and no more than two seconds".into(),
            );
        }
        let maximum_delay = (self.sample_rate as usize)
            .checked_mul(2)
            .ok_or("AEC delay bound overflow")?;
        if self.maximum_delay_samples > maximum_delay {
            return Err("AEC maximum_delay_samples must not exceed two seconds".into());
        }
        let minimum_analysis = self
            .maximum_delay_samples
            .saturating_mul(2)
            .saturating_add(self.block_size_samples);
        let maximum_analysis = (self.sample_rate as usize)
            .checked_mul(10)
            .ok_or("AEC delay-analysis bound overflow")?;
        if self.delay_analysis_samples < minimum_analysis
            || self.delay_analysis_samples > maximum_analysis
        {
            return Err(
                "AEC delay_analysis_samples must cover both delay signs and no more than ten seconds"
                    .into(),
            );
        }
        validate_f32_range(
            "minimum_delay_confidence",
            self.minimum_delay_confidence,
            0.01,
            1.0,
        )?;
        validate_f32_range("adaptation_rate", self.adaptation_rate, 0.001, 0.5)?;
        validate_f32_range("filter_leakage", self.filter_leakage, 0.95, 1.0)?;
        validate_f32_range(
            "adaptation_regularization",
            self.adaptation_regularization,
            1.0e-9,
            1.0,
        )?;
        validate_f32_range(
            "double_talk_correlation_threshold",
            self.double_talk_correlation_threshold,
            0.01,
            0.95,
        )?;
        validate_f32_range(
            "reference_activation_rms",
            self.reference_activation_rms,
            1.0e-6,
            0.1,
        )?;
        validate_f32_range("residual_suppression", self.residual_suppression, 0.0, 1.0)?;
        validate_f32_range("minimum_far_end_gain", self.minimum_far_end_gain, 0.0, 1.0)?;
        validate_f32_range(
            "minimum_double_talk_gain",
            self.minimum_double_talk_gain,
            0.5,
            1.0,
        )?;
        if self.minimum_double_talk_gain < self.minimum_far_end_gain {
            return Err("AEC double-talk gain must not be below the far-end-only gain".into());
        }
        validate_f32_range("maximum_peak", self.maximum_peak, 0.5, 1.0)?;
        estimate_aec_memory_bytes(self)?;
        Ok(())
    }

    pub fn algorithmic_plus_buffering_milliseconds(&self) -> f64 {
        self.block_size_samples as f64 * 1_000.0 / self.sample_rate as f64
    }

    pub fn digest(&self) -> Result<String, String> {
        let document = serde_json::to_vec(self)
            .map_err(|error| format!("serialize AEC configuration: {error}"))?;
        let mut digest = Sha256::new();
        digest.update(CONFIG_DIGEST_DOMAIN);
        digest.update(document);
        Ok(format!("{:x}", digest.finalize()))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecClockMapping {
    pub microphone_sample_rate: u32,
    pub reference_sample_rate: u32,
    pub reference_clock_ppm: f64,
    pub initial_delay_samples: i32,
    pub route_generation: u64,
}

impl AecClockMapping {
    pub fn validate(&self, config: &AecConfig) -> Result<(), String> {
        if self.microphone_sample_rate != config.sample_rate {
            return Err(
                "AEC clock mapping microphone rate does not match the promoted configuration"
                    .into(),
            );
        }
        if !(8_000..=192_000).contains(&self.reference_sample_rate) {
            return Err("AEC reference sample rate must be in 8000..=192000".into());
        }
        if !self.reference_clock_ppm.is_finite()
            || !(-2_000.0..=2_000.0).contains(&self.reference_clock_ppm)
        {
            return Err("AEC reference_clock_ppm must be finite and in -2000..=2000".into());
        }
        if self.initial_delay_samples.unsigned_abs() as usize > config.maximum_delay_samples {
            return Err("AEC initial signed delay exceeds maximum_delay_samples".into());
        }
        if self.route_generation > JSON_SAFE_INTEGER {
            return Err("AEC route generation exceeds the JSON safe-integer limit".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AecEvidenceStratumKind {
    FarEndOnly,
    NearEndOnly,
    DoubleTalk,
    Transition,
    Impairment,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AecEvidenceMetricOperator {
    GreaterOrEqual,
    LessOrEqual,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecEvidenceMetric {
    pub metric: String,
    pub value: f64,
    pub operator: AecEvidenceMetricOperator,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecEvidenceStratum {
    pub id: String,
    pub kind: AecEvidenceStratumKind,
    pub cases: u32,
    pub metrics: Vec<AecEvidenceMetric>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub implementation: String,
    pub implementation_source_revision: String,
    pub implementation_source_sha256: String,
    pub configuration_sha256: String,
    pub corpus_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub listening_result_sha256: String,
    pub sample_rate: u32,
    pub block_size_samples: usize,
    pub tail_samples: usize,
    pub maximum_delay_samples: usize,
    pub strata: Vec<AecEvidenceStratum>,
    pub real_device_cases: u32,
    pub nonlinear_device_cases: u32,
    pub delay_transition_cases: u32,
    pub paced_realtime_blocks: u64,
    pub worst_case_realtime_factor: f64,
    pub callback_allocations: u64,
    pub callback_locks: u64,
    pub callback_waits: u64,
    pub callback_io_operations: u64,
    pub callback_log_operations: u64,
    pub deadline_misses: u64,
    pub stale_frames_after_reset: u64,
    pub minimum_listeners: u32,
    pub listener_count: u32,
    pub listener_preference: f64,
    pub listener_preference_limit: f64,
    pub accepted: bool,
}

#[derive(Clone, Copy)]
struct MetricPolicy {
    name: &'static str,
    operator: AecEvidenceMetricOperator,
    hard_limit: f64,
}

impl MetricPolicy {
    const fn at_least(name: &'static str, hard_limit: f64) -> Self {
        Self {
            name,
            operator: AecEvidenceMetricOperator::GreaterOrEqual,
            hard_limit,
        }
    }

    const fn at_most(name: &'static str, hard_limit: f64) -> Self {
        Self {
            name,
            operator: AecEvidenceMetricOperator::LessOrEqual,
            hard_limit,
        }
    }
}

const COMMON_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("latency.algorithmic-plus-buffering-ms", 20.0),
    MetricPolicy::at_most("output.duration-error-frames", 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0),
];
const FAR_END_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_least("echo.erle-db", 10.0),
    MetricPolicy::at_least("perceptual.aecmos-far-end", 3.5),
];
const NEAR_END_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("content.word-accuracy-regression", 0.02),
    MetricPolicy::at_most("near-end.attenuation-db", 1.0),
];
const DOUBLE_TALK_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("content.word-accuracy-regression", 0.02),
    MetricPolicy::at_most("near-end.attenuation-db", 1.5),
    MetricPolicy::at_least("perceptual.aecmos-double-talk", 3.2),
];
const TRANSITION_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("near-end.attenuation-db", 1.5),
    MetricPolicy::at_most("reset.stale-output-frames", 0.0),
    MetricPolicy::at_most("transition.reconvergence-ms", 500.0),
];
const IMPAIRMENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("content.word-accuracy-regression", 0.02),
    MetricPolicy::at_least("echo.erle-db", 6.0),
    MetricPolicy::at_most("near-end.attenuation-db", 1.5),
    MetricPolicy::at_least("perceptual.aecmos", 3.0),
];

const REQUIRED_STRATA: &[(&str, AecEvidenceStratumKind)] = &[
    ("background-noise", AecEvidenceStratumKind::Impairment),
    ("clipping", AecEvidenceStratumKind::Impairment),
    ("clock-drift-negative", AecEvidenceStratumKind::Transition),
    ("clock-drift-positive", AecEvidenceStratumKind::Transition),
    ("delay-jump", AecEvidenceStratumKind::Transition),
    ("delay-negative", AecEvidenceStratumKind::Transition),
    ("delay-positive", AecEvidenceStratumKind::Transition),
    ("double-talk", AecEvidenceStratumKind::DoubleTalk),
    ("far-end-clean", AecEvidenceStratumKind::FarEndOnly),
    ("linear-path", AecEvidenceStratumKind::FarEndOnly),
    ("music-playback", AecEvidenceStratumKind::Impairment),
    ("near-end-clean", AecEvidenceStratumKind::NearEndOnly),
    ("nonlinear-speaker", AecEvidenceStratumKind::Impairment),
    ("real-device", AecEvidenceStratumKind::Impairment),
    ("reference-loss", AecEvidenceStratumKind::Transition),
    ("room-change", AecEvidenceStratumKind::Transition),
    ("route-change", AecEvidenceStratumKind::Transition),
];

impl AecPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        if self.completed_at_unix_seconds > JSON_SAFE_INTEGER {
            return Err("AEC evidence timestamp exceeds the JSON safe-integer limit".into());
        }
        if self.implementation != IMPLEMENTATION_ID {
            return Err("AEC evidence names an unsupported implementation".into());
        }
        validate_identifier(
            "AEC implementation source revision",
            &self.implementation_source_revision,
        )?;
        for (label, value) in [
            (
                "implementation source",
                self.implementation_source_sha256.as_str(),
            ),
            ("configuration", self.configuration_sha256.as_str()),
            ("corpus manifest", self.corpus_manifest_sha256.as_str()),
            ("evaluation result", self.evaluation_result_sha256.as_str()),
            ("listening result", self.listening_result_sha256.as_str()),
        ] {
            validate_sha256(label, value)?;
        }
        let geometry = AecConfig {
            sample_rate: self.sample_rate,
            block_size_samples: self.block_size_samples,
            tail_samples: self.tail_samples,
            maximum_delay_samples: self.maximum_delay_samples,
            ..AecConfig::default()
        };
        if !(8_000..=192_000).contains(&geometry.sample_rate)
            || !geometry.block_size_samples.is_power_of_two()
            || geometry.block_size_samples < 64
            || geometry.block_size_samples as f64 * 1_000.0 / geometry.sample_rate as f64 > 20.0
            || geometry.tail_samples < geometry.block_size_samples
            || geometry.tail_samples > geometry.sample_rate as usize * 2
            || geometry.maximum_delay_samples > geometry.sample_rate as usize * 2
        {
            return Err("AEC evidence contains invalid promoted geometry".into());
        }
        if self.strata.len() != REQUIRED_STRATA.len() || self.strata.len() > MAX_EVIDENCE_STRATA {
            return Err(format!(
                "AEC evidence must contain exactly {} required strata",
                REQUIRED_STRATA.len()
            ));
        }
        let required: BTreeMap<_, _> = REQUIRED_STRATA.iter().copied().collect();
        let mut observed = BTreeSet::new();
        let mut previous = None;
        let mut all_metrics_passed = true;
        for stratum in &self.strata {
            validate_identifier("AEC evidence stratum", &stratum.id)?;
            if previous.is_some_and(|value: &str| value >= stratum.id.as_str()) {
                return Err("AEC evidence strata must be unique and strictly sorted".into());
            }
            previous = Some(&stratum.id);
            let expected_kind = required
                .get(stratum.id.as_str())
                .ok_or_else(|| format!("AEC evidence contains unknown stratum {}", stratum.id))?;
            if *expected_kind != stratum.kind {
                return Err(format!(
                    "AEC evidence stratum {} has the wrong kind",
                    stratum.id
                ));
            }
            observed.insert(stratum.id.as_str());
            if !(10..=1_000_000).contains(&stratum.cases) {
                return Err("AEC evidence stratum cases must be in 10..=1000000".into());
            }
            if stratum.metrics.is_empty() || stratum.metrics.len() > MAX_EVIDENCE_METRICS {
                return Err(format!(
                    "AEC evidence stratum metrics must be in 1..={MAX_EVIDENCE_METRICS}"
                ));
            }
            let mut policies = BTreeMap::new();
            for policy in COMMON_METRICS
                .iter()
                .chain(kind_metrics(stratum.kind).iter())
            {
                policies.insert(policy.name, policy);
            }
            let mut observed_metrics = BTreeSet::new();
            let mut previous_metric = None;
            for metric in &stratum.metrics {
                validate_identifier("AEC evidence metric", &metric.metric)?;
                if previous_metric.is_some_and(|value: &str| value >= metric.metric.as_str()) {
                    return Err("AEC evidence metrics must be unique and strictly sorted".into());
                }
                previous_metric = Some(&metric.metric);
                if !metric.value.is_finite() || !metric.limit.is_finite() {
                    return Err("AEC evidence metric values must be finite".into());
                }
                let policy = policies.get(metric.metric.as_str()).ok_or_else(|| {
                    format!("AEC evidence contains unknown metric {}", metric.metric)
                })?;
                if metric.operator != policy.operator {
                    return Err(format!(
                        "AEC evidence metric {} has the wrong operator",
                        metric.metric
                    ));
                }
                let strict_enough = match metric.operator {
                    AecEvidenceMetricOperator::GreaterOrEqual => metric.limit >= policy.hard_limit,
                    AecEvidenceMetricOperator::LessOrEqual => metric.limit <= policy.hard_limit,
                };
                if !strict_enough {
                    return Err(format!(
                        "AEC evidence metric {} weakens its hard limit",
                        metric.metric
                    ));
                }
                let expected_passed = match metric.operator {
                    AecEvidenceMetricOperator::GreaterOrEqual => metric.value >= metric.limit,
                    AecEvidenceMetricOperator::LessOrEqual => metric.value <= metric.limit,
                };
                if metric.passed != expected_passed {
                    return Err(format!(
                        "AEC evidence metric {} has an inconsistent passed flag",
                        metric.metric
                    ));
                }
                observed_metrics.insert(metric.metric.as_str());
                all_metrics_passed &= metric.passed;
            }
            for policy in policies.values() {
                if !observed_metrics.contains(policy.name) {
                    return Err(format!(
                        "AEC evidence stratum {} omits required metric {}",
                        stratum.id, policy.name
                    ));
                }
            }
        }
        for (id, _) in REQUIRED_STRATA {
            if !observed.contains(id) {
                return Err(format!("AEC evidence omits required stratum {id}"));
            }
        }
        if !(100..=1_000_000).contains(&self.real_device_cases)
            || !(100..=1_000_000).contains(&self.nonlinear_device_cases)
            || !(100..=1_000_000).contains(&self.delay_transition_cases)
        {
            return Err(
                "AEC evidence requires at least 100 real-device, nonlinear, and transition cases"
                    .into(),
            );
        }
        if !(10_000..=JSON_SAFE_INTEGER).contains(&self.paced_realtime_blocks)
            || !self.worst_case_realtime_factor.is_finite()
            || !(0.0..=0.5).contains(&self.worst_case_realtime_factor)
        {
            return Err("AEC evidence real-time audit is outside the promotion limits".into());
        }
        let callback_violations = self.callback_allocations
            | self.callback_locks
            | self.callback_waits
            | self.callback_io_operations
            | self.callback_log_operations
            | self.deadline_misses
            | self.stale_frames_after_reset;
        if callback_violations != 0 {
            return Err(
                "AEC evidence records a callback, deadline, or stale-reset violation".into(),
            );
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
            return Err("AEC evidence listening audit is invalid".into());
        }
        let expected_accepted =
            all_metrics_passed && self.listener_preference >= self.listener_preference_limit;
        if self.accepted != expected_accepted {
            return Err("AEC evidence accepted flag is inconsistent".into());
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize AEC evidence payload: {error}"))?;
        if bytes.len() as u64 >= MAX_EVIDENCE_JSON_BYTES {
            return Err("AEC evidence payload exceeds the bounded JSON limit".into());
        }
        Ok(())
    }
}

fn kind_metrics(kind: AecEvidenceStratumKind) -> &'static [MetricPolicy] {
    match kind {
        AecEvidenceStratumKind::FarEndOnly => FAR_END_METRICS,
        AecEvidenceStratumKind::NearEndOnly => NEAR_END_METRICS,
        AecEvidenceStratumKind::DoubleTalk => DOUBLE_TALK_METRICS,
        AecEvidenceStratumKind::Transition => TRANSITION_METRICS,
        AecEvidenceStratumKind::Impairment => IMPAIRMENT_METRICS,
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedAecPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: AecPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedAecPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) = crate::input::open_regular_file(path, "AEC promotion evidence")?;
        if length >= MAX_EVIDENCE_JSON_BYTES {
            return Err(format!(
                "AEC promotion evidence {} exceeds the {MAX_EVIDENCE_JSON_BYTES}-byte limit",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve AEC evidence JSON".to_string())?;
        file.take(MAX_EVIDENCE_JSON_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read AEC promotion evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("AEC promotion evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse AEC promotion evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != AEC_PROMOTION_EVIDENCE_SCHEMA || self.schema_version != AEC_SCHEMA_VERSION
        {
            return Err("unsupported AEC promotion evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("AEC promotion evidence signature must use ed25519".into());
        }
        validate_sha256("AEC evidence key ID", &self.signature.key_id)?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize AEC promotion evidence: {error}"))?;
        if bytes.len() as u64 >= MAX_EVIDENCE_JSON_BYTES {
            return Err("AEC promotion evidence exceeds the bounded JSON limit".into());
        }
        Ok(())
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize AEC evidence for verification: {error}"))?;
        key.verify_domain_document(
            EVIDENCE_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "AEC promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize AEC promotion evidence: {error}"))
    }
}

pub fn sign_aec_promotion_evidence(
    payload: AecPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedAecPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize AEC evidence for signing: {error}"))?;
    let signature = key.sign_domain_document(
        EVIDENCE_SIGNATURE_DOMAIN,
        &document,
        "AEC promotion evidence",
    )?;
    let evidence = SignedAecPromotionEvidence {
        schema: AEC_PROMOTION_EVIDENCE_SCHEMA.into(),
        schema_version: AEC_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AecTalkState {
    Silence,
    FarEndOnly,
    NearEndOnly,
    DoubleTalk,
    ReferenceUncertain,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AecResetReason {
    Initial,
    RouteChange,
    ReferenceDiscontinuity,
    ClockJump,
    DelayJump,
    NonFiniteState,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecResetCounts {
    pub initial: u64,
    pub route_change: u64,
    pub reference_discontinuity: u64,
    pub clock_jump: u64,
    pub delay_jump: u64,
    pub non_finite_state: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecDelayEstimate {
    pub signed_delay_samples: i32,
    pub confidence: f32,
    pub polarity_inverted: bool,
    pub analyzed_samples: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecRenderReport {
    pub schema: String,
    pub schema_version: u32,
    pub implementation: String,
    pub configuration_sha256: String,
    pub evidence_signing_key_id: String,
    pub evidence_evaluation_result_sha256: String,
    pub microphone_pcm_sha256: String,
    pub reference_pcm_sha256: String,
    pub output_pcm_sha256: String,
    pub microphone_frames: usize,
    pub reference_frames: usize,
    pub output_frames: usize,
    pub microphone_sample_rate: u32,
    pub reference_sample_rate: u32,
    pub reference_clock_ppm: f64,
    pub route_generation: u64,
    pub delay: AecDelayEstimate,
    pub block_size_samples: usize,
    pub tail_samples: usize,
    pub maximum_delay_samples: usize,
    pub algorithmic_plus_buffering_milliseconds: f64,
    pub silence_blocks: u64,
    pub far_end_only_blocks: u64,
    pub near_end_only_blocks: u64,
    pub double_talk_blocks: u64,
    pub reference_uncertain_blocks: u64,
    pub adaptation_blocks: u64,
    pub reset_count: u64,
    pub reset_reasons: AecResetCounts,
    pub clipped_samples: u64,
    pub non_finite_output_samples: u64,
    pub far_end_erle_db: Option<f64>,
    pub exact_output_duration: bool,
    pub paths_recorded: u32,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct AecRenderResult {
    pub audio: Audio,
    pub report: AecRenderReport,
}

impl AecRenderReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != AEC_REPORT_SCHEMA || self.schema_version != AEC_SCHEMA_VERSION {
            return Err("unsupported AEC report schema".into());
        }
        if self.implementation != IMPLEMENTATION_ID {
            return Err("AEC report names an unsupported implementation".into());
        }
        for (label, digest) in [
            ("configuration", self.configuration_sha256.as_str()),
            ("evidence key", self.evidence_signing_key_id.as_str()),
            (
                "evidence evaluation result",
                self.evidence_evaluation_result_sha256.as_str(),
            ),
            ("microphone PCM", self.microphone_pcm_sha256.as_str()),
            ("reference PCM", self.reference_pcm_sha256.as_str()),
            ("output PCM", self.output_pcm_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
        if self.microphone_frames == 0
            || self.output_frames != self.microphone_frames
            || !self.exact_output_duration
        {
            return Err("AEC report must bind exact non-empty microphone/output geometry".into());
        }
        if !(8_000..=192_000).contains(&self.microphone_sample_rate)
            || !(8_000..=192_000).contains(&self.reference_sample_rate)
            || !self.reference_clock_ppm.is_finite()
            || !(-2_000.0..=2_000.0).contains(&self.reference_clock_ppm)
        {
            return Err("AEC report clock mapping is invalid".into());
        }
        if self.route_generation > JSON_SAFE_INTEGER
            || self.delay.signed_delay_samples.unsigned_abs() as usize > self.maximum_delay_samples
            || !self.delay.confidence.is_finite()
            || !(0.0..=1.0).contains(&self.delay.confidence)
        {
            return Err("AEC report delay or route identity is invalid".into());
        }
        if !self.block_size_samples.is_power_of_two()
            || !(64..=4_096).contains(&self.block_size_samples)
            || self.tail_samples < self.block_size_samples
            || self.tail_samples > self.microphone_sample_rate as usize * 2
            || self.maximum_delay_samples > self.microphone_sample_rate as usize * 2
        {
            return Err("AEC report filter geometry is invalid".into());
        }
        let expected_latency =
            self.block_size_samples as f64 * 1_000.0 / self.microphone_sample_rate as f64;
        if !self.algorithmic_plus_buffering_milliseconds.is_finite()
            || self.algorithmic_plus_buffering_milliseconds > 20.0
            || (self.algorithmic_plus_buffering_milliseconds - expected_latency).abs() > 1.0e-9
        {
            return Err("AEC report latency does not match its block geometry".into());
        }
        let classified_blocks = self
            .silence_blocks
            .checked_add(self.far_end_only_blocks)
            .and_then(|value| value.checked_add(self.near_end_only_blocks))
            .and_then(|value| value.checked_add(self.double_talk_blocks))
            .and_then(|value| value.checked_add(self.reference_uncertain_blocks))
            .ok_or("AEC report block-count overflow")?;
        let expected_blocks =
            u64::try_from(self.microphone_frames.div_ceil(self.block_size_samples))
                .map_err(|_| "AEC report block count is not representable".to_string())?;
        if classified_blocks != expected_blocks || self.adaptation_blocks > self.far_end_only_blocks
        {
            return Err(
                "AEC report block classifications or adaptation counts are inconsistent".into(),
            );
        }
        for value in [
            self.silence_blocks,
            self.far_end_only_blocks,
            self.near_end_only_blocks,
            self.double_talk_blocks,
            self.reference_uncertain_blocks,
            self.adaptation_blocks,
            self.reset_count,
            self.clipped_samples,
            self.non_finite_output_samples,
            self.reset_reasons.initial,
            self.reset_reasons.route_change,
            self.reset_reasons.reference_discontinuity,
            self.reset_reasons.clock_jump,
            self.reset_reasons.delay_jump,
            self.reset_reasons.non_finite_state,
        ] {
            if value > JSON_SAFE_INTEGER {
                return Err("AEC report count exceeds the JSON safe-integer limit".into());
            }
        }
        let reset_reason_total = self
            .reset_reasons
            .initial
            .checked_add(self.reset_reasons.route_change)
            .and_then(|value| value.checked_add(self.reset_reasons.reference_discontinuity))
            .and_then(|value| value.checked_add(self.reset_reasons.clock_jump))
            .and_then(|value| value.checked_add(self.reset_reasons.delay_jump))
            .and_then(|value| value.checked_add(self.reset_reasons.non_finite_state))
            .ok_or("AEC report reset-reason count overflow")?;
        if reset_reason_total != self.reset_count {
            return Err("AEC report reset reasons do not sum to reset_count".into());
        }
        if self.non_finite_output_samples != 0 || self.paths_recorded != 0 {
            return Err("AEC report records a non-finite output or filesystem path".into());
        }
        if self
            .far_end_erle_db
            .is_some_and(|value| !value.is_finite() || !(-240.0..=240.0).contains(&value))
        {
            return Err("AEC report far-end ERLE is invalid".into());
        }
        if !(5..=8).contains(&self.limitations.len())
            || self
                .limitations
                .iter()
                .any(|value| value.is_empty() || value.len() > 1_024)
            || self.limitations.iter().collect::<BTreeSet<_>>().len() != self.limitations.len()
        {
            return Err("AEC report limitations must be bounded and unique".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| format!("serialize AEC report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|error| format!("serialize AEC report: {error}"))
    }
}

#[derive(Clone, Debug)]
struct AecEvidenceIdentity {
    signing_key_id: String,
    evaluation_result_sha256: String,
}

#[derive(Clone, Debug)]
pub struct AecSession {
    config: AecConfig,
    evidence: AecEvidenceIdentity,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecBlockDiagnostics {
    pub talk_state: Option<AecTalkState>,
    pub reference_rms: f32,
    pub microphone_rms: f32,
    pub linear_error_rms: f32,
    pub residual_gain: f32,
    pub adapted: bool,
    pub state_reset: bool,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AecRealtimeDiagnostics {
    pub completed_blocks: u64,
    pub last_block: Option<AecBlockDiagnostics>,
    pub latency_samples: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct AecStreamCounters {
    silence_blocks: u64,
    far_end_only_blocks: u64,
    near_end_only_blocks: u64,
    double_talk_blocks: u64,
    reference_uncertain_blocks: u64,
    adaptation_blocks: u64,
    reset_count: u64,
    reset_reasons: AecResetCounts,
    clipped_samples: u64,
    non_finite_output_samples: u64,
    far_microphone_energy: f64,
    far_error_energy: f64,
}

/// Preallocated causal AEC stream.
///
/// Construction and reset may take ordinary control-thread time. Once built,
/// [`process_block`](Self::process_block) accepts exactly one promoted block
/// and performs no allocation, locks, waits, I/O, network access, or logging.
pub struct AecStream {
    config: AecConfig,
    fft_size: usize,
    partitions: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex32>,
    reference_fft: Vec<Complex32>,
    error_fft: Vec<Complex32>,
    estimate_fft: Vec<Complex32>,
    projection_fft: Vec<Complex32>,
    reference_history: Vec<Complex32>,
    filter: Vec<Complex32>,
    previous_reference: Vec<f32>,
    echo_estimate: Vec<f32>,
    linear_error: Vec<f32>,
    ring_position: usize,
    processed_blocks: u64,
    route_generation: u64,
    reference_confident: bool,
    reference_was_active: bool,
    pending_reset_marker: bool,
    counters: AecStreamCounters,
}

/// Arbitrary-quantum mono sidechain adapter with one fixed AEC-block latency.
///
/// Construction allocates every input, reference, processed, and latency
/// buffer. Successful [`process`](Self::process) calls allocate, lock, wait,
/// log, and perform I/O zero times.
pub struct AecRealtimeAdapter {
    stream: AecStream,
    microphone_block: Vec<f32>,
    reference_block: Vec<f32>,
    processed_block: Vec<f32>,
    latency_block: Vec<f32>,
    input_fill: usize,
    latency_index: usize,
}

impl std::fmt::Debug for AecRealtimeAdapter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AecRealtimeAdapter")
            .field("stream", &self.stream)
            .field("input_fill", &self.input_fill)
            .field("latency_index", &self.latency_index)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for AecStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AecStream")
            .field("config", &self.config)
            .field("fft_size", &self.fft_size)
            .field("partitions", &self.partitions)
            .field("processed_blocks", &self.processed_blocks)
            .field("route_generation", &self.route_generation)
            .field("reference_confident", &self.reference_confident)
            .finish_non_exhaustive()
    }
}

impl AecSession {
    pub fn prepare(
        evidence: &SignedAecPromotionEvidence,
        evidence_key: &ReceiptPublicKey,
        config: AecConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        evidence.verify_signature(evidence_key)?;
        if !evidence.payload.accepted {
            return Err(
                "AEC promotion evidence is authentic but does not pass promotion gates".into(),
            );
        }
        let payload = &evidence.payload;
        if payload.implementation != IMPLEMENTATION_ID
            || payload.sample_rate != config.sample_rate
            || payload.block_size_samples != config.block_size_samples
            || payload.tail_samples != config.tail_samples
            || payload.maximum_delay_samples != config.maximum_delay_samples
            || payload.configuration_sha256 != config.digest()?
        {
            return Err("AEC promotion evidence does not bind the requested implementation and configuration".into());
        }
        Ok(Self {
            config,
            evidence: AecEvidenceIdentity {
                signing_key_id: evidence.signature.key_id.clone(),
                evaluation_result_sha256: payload.evaluation_result_sha256.clone(),
            },
        })
    }

    pub fn config(&self) -> &AecConfig {
        &self.config
    }

    pub fn stream(
        &self,
        route_generation: u64,
        reference_confident: bool,
    ) -> Result<AecStream, String> {
        AecStream::new(self.config.clone(), route_generation, reference_confident)
    }

    pub fn realtime_adapter(
        &self,
        route_generation: u64,
        reference_confident: bool,
    ) -> Result<AecRealtimeAdapter, String> {
        AecRealtimeAdapter::new(self.stream(route_generation, reference_confident)?)
    }

    pub fn render(
        &self,
        microphone: &Audio,
        reference: &Audio,
        mapping: &AecClockMapping,
    ) -> Result<AecRenderResult, String> {
        validate_mono_audio(microphone, "AEC microphone", true)?;
        validate_mono_audio(reference, "AEC far-end reference", false)?;
        mapping.validate(&self.config)?;
        if microphone.sample_rate != mapping.microphone_sample_rate
            || reference.sample_rate != mapping.reference_sample_rate
        {
            return Err("AEC audio sample rates do not match the explicit clock mapping".into());
        }
        let microphone_frames = microphone.frames();
        let reference_frames = reference.frames();
        let mapped_length = microphone_frames
            .checked_add(self.config.maximum_delay_samples)
            .and_then(|value| value.checked_add(self.config.block_size_samples))
            .ok_or("AEC mapped reference length overflow")?;
        let mapped_reference = map_reference_clock(&reference.channels[0], mapping, mapped_length)?;
        let delay = estimate_delay(
            &microphone.channels[0],
            &mapped_reference,
            &self.config,
            mapping.initial_delay_samples,
        );
        let reference_confident = delay.confidence >= self.config.minimum_delay_confidence;
        let mut aligned_reference = Vec::new();
        aligned_reference
            .try_reserve_exact(microphone_frames)
            .map_err(|_| "unable to reserve aligned AEC reference".to_string())?;
        for frame in 0..microphone_frames {
            let reference_index = frame as i64 - delay.signed_delay_samples as i64;
            let sample = if reference_index >= 0 {
                mapped_reference
                    .get(reference_index as usize)
                    .copied()
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            aligned_reference.push(sample as f32);
        }
        let mut stream = self.stream(mapping.route_generation, reference_confident)?;
        let block_size = self.config.block_size_samples;
        let mut microphone_block = vec![0.0_f32; block_size];
        let mut reference_block = vec![0.0_f32; block_size];
        let mut output_block = vec![0.0_f32; block_size];
        let mut output = Vec::new();
        output
            .try_reserve_exact(microphone_frames)
            .map_err(|_| "unable to reserve AEC output".to_string())?;
        let mut offset = 0usize;
        while offset < microphone_frames {
            let valid = (microphone_frames - offset).min(block_size);
            microphone_block.fill(0.0);
            reference_block.fill(0.0);
            for frame in 0..valid {
                microphone_block[frame] = microphone.channels[0][offset + frame] as f32;
                reference_block[frame] = aligned_reference[offset + frame];
            }
            stream.process_block(&microphone_block, &reference_block, &mut output_block)?;
            output.extend(output_block[..valid].iter().map(|sample| *sample as f64));
            offset += valid;
        }
        if microphone_frames == 0 {
            return Err("AEC microphone must contain at least one frame".into());
        }
        let counters = stream.counters;
        let far_end_erle_db = if counters.far_microphone_energy > POWER_FLOOR as f64
            && counters.far_error_energy > POWER_FLOOR as f64
        {
            Some(
                (10.0 * (counters.far_microphone_energy / counters.far_error_energy).log10())
                    .clamp(-240.0, 240.0),
            )
        } else {
            None
        };
        let microphone_digest = digest_pcm(
            MICROPHONE_PCM_DIGEST_DOMAIN,
            microphone.sample_rate,
            &microphone.channels[0],
        );
        let reference_digest = digest_pcm(
            REFERENCE_PCM_DIGEST_DOMAIN,
            reference.sample_rate,
            &reference.channels[0],
        );
        let output_digest = digest_pcm(OUTPUT_PCM_DIGEST_DOMAIN, microphone.sample_rate, &output);
        let report = AecRenderReport {
            schema: AEC_REPORT_SCHEMA.into(),
            schema_version: AEC_SCHEMA_VERSION,
            implementation: IMPLEMENTATION_ID.into(),
            configuration_sha256: self.config.digest()?,
            evidence_signing_key_id: self.evidence.signing_key_id.clone(),
            evidence_evaluation_result_sha256: self.evidence.evaluation_result_sha256.clone(),
            microphone_pcm_sha256: microphone_digest,
            reference_pcm_sha256: reference_digest,
            output_pcm_sha256: output_digest,
            microphone_frames,
            reference_frames,
            output_frames: output.len(),
            microphone_sample_rate: microphone.sample_rate,
            reference_sample_rate: reference.sample_rate,
            reference_clock_ppm: mapping.reference_clock_ppm,
            route_generation: mapping.route_generation,
            delay,
            block_size_samples: self.config.block_size_samples,
            tail_samples: self.config.tail_samples,
            maximum_delay_samples: self.config.maximum_delay_samples,
            algorithmic_plus_buffering_milliseconds: self
                .config
                .algorithmic_plus_buffering_milliseconds(),
            silence_blocks: counters.silence_blocks,
            far_end_only_blocks: counters.far_end_only_blocks,
            near_end_only_blocks: counters.near_end_only_blocks,
            double_talk_blocks: counters.double_talk_blocks,
            reference_uncertain_blocks: counters.reference_uncertain_blocks,
            adaptation_blocks: counters.adaptation_blocks,
            reset_count: counters.reset_count,
            reset_reasons: counters.reset_reasons,
            clipped_samples: counters.clipped_samples,
            non_finite_output_samples: counters.non_finite_output_samples,
            far_end_erle_db,
            exact_output_duration: output.len() == microphone_frames,
            paths_recorded: 0,
            limitations: vec![
                "the promoted baseline accepts one microphone channel and one typed far-end reference channel".into(),
                "constant clock drift is mapped explicitly; abrupt clock jumps require a cold reset".into(),
                "low-confidence or missing reference blocks preserve the microphone instead of suppressing it".into(),
                "ERLE is reported only from blocks classified as far-end-only".into(),
                "the native linear baseline does not claim nonlinear neural residual-filter parity".into(),
            ],
        };
        if !report.exact_output_duration || report.non_finite_output_samples != 0 {
            return Err("AEC render violated exact-duration or finite-output safety".into());
        }
        report.validate()?;
        Ok(AecRenderResult {
            audio: Audio {
                sample_rate: microphone.sample_rate,
                channels: vec![output],
                bits_per_sample: microphone.bits_per_sample,
                sample_format: microphone.sample_format,
                channel_mask: microphone.channel_mask,
            },
            report,
        })
    }
}

impl AecStream {
    fn new(
        config: AecConfig,
        route_generation: u64,
        reference_confident: bool,
    ) -> Result<Self, String> {
        config.validate()?;
        if route_generation > JSON_SAFE_INTEGER {
            return Err("AEC route generation exceeds the JSON safe-integer limit".into());
        }
        let fft_size = config
            .block_size_samples
            .checked_mul(2)
            .ok_or("AEC FFT size overflow")?;
        let partitions = config.tail_samples.div_ceil(config.block_size_samples);
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft_size);
        let inverse = planner.plan_fft_inverse(fft_size);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let complex_zero = Complex32::new(0.0, 0.0);
        let history_len = partitions
            .checked_mul(fft_size)
            .ok_or("AEC filter history size overflow")?;
        let block_size = config.block_size_samples;
        Ok(Self {
            config,
            fft_size,
            partitions,
            forward,
            inverse,
            scratch: vec![complex_zero; scratch_len],
            reference_fft: vec![complex_zero; fft_size],
            error_fft: vec![complex_zero; fft_size],
            estimate_fft: vec![complex_zero; fft_size],
            projection_fft: vec![complex_zero; fft_size],
            reference_history: vec![complex_zero; history_len],
            filter: vec![complex_zero; history_len],
            previous_reference: vec![0.0; block_size],
            echo_estimate: vec![0.0; block_size],
            linear_error: vec![0.0; block_size],
            ring_position: 0,
            processed_blocks: 0,
            route_generation,
            reference_confident,
            reference_was_active: false,
            pending_reset_marker: true,
            counters: AecStreamCounters {
                reset_count: 1,
                reset_reasons: AecResetCounts {
                    initial: 1,
                    ..AecResetCounts::default()
                },
                ..AecStreamCounters::default()
            },
        })
    }

    pub fn block_size_samples(&self) -> usize {
        self.config.block_size_samples
    }

    pub fn route_generation(&self) -> u64 {
        self.route_generation
    }

    pub fn set_reference_confident(&mut self, confident: bool) {
        if self.reference_confident && !confident {
            self.reset(AecResetReason::ReferenceDiscontinuity);
        }
        self.reference_confident = confident;
    }

    pub fn set_route_generation(&mut self, generation: u64) -> Result<(), String> {
        if generation > JSON_SAFE_INTEGER {
            return Err("AEC route generation exceeds the JSON safe-integer limit".into());
        }
        if generation != self.route_generation {
            self.route_generation = generation;
            self.reset(AecResetReason::RouteChange);
        }
        Ok(())
    }

    pub fn reset(&mut self, reason: AecResetReason) {
        self.scratch.fill(Complex32::new(0.0, 0.0));
        self.reference_fft.fill(Complex32::new(0.0, 0.0));
        self.error_fft.fill(Complex32::new(0.0, 0.0));
        self.estimate_fft.fill(Complex32::new(0.0, 0.0));
        self.projection_fft.fill(Complex32::new(0.0, 0.0));
        self.reference_history.fill(Complex32::new(0.0, 0.0));
        self.filter.fill(Complex32::new(0.0, 0.0));
        self.previous_reference.fill(0.0);
        self.echo_estimate.fill(0.0);
        self.linear_error.fill(0.0);
        self.ring_position = 0;
        self.processed_blocks = 0;
        self.reference_was_active = false;
        self.pending_reset_marker = true;
        self.counters.reset_count = self.counters.reset_count.saturating_add(1);
        let counter = match reason {
            AecResetReason::Initial => &mut self.counters.reset_reasons.initial,
            AecResetReason::RouteChange => &mut self.counters.reset_reasons.route_change,
            AecResetReason::ReferenceDiscontinuity => {
                &mut self.counters.reset_reasons.reference_discontinuity
            }
            AecResetReason::ClockJump => &mut self.counters.reset_reasons.clock_jump,
            AecResetReason::DelayJump => &mut self.counters.reset_reasons.delay_jump,
            AecResetReason::NonFiniteState => &mut self.counters.reset_reasons.non_finite_state,
        };
        *counter = counter.saturating_add(1);
    }

    pub fn process_block(
        &mut self,
        microphone: &[f32],
        reference: &[f32],
        output: &mut [f32],
    ) -> Result<AecBlockDiagnostics, String> {
        let block_size = self.config.block_size_samples;
        if microphone.len() != block_size
            || reference.len() != block_size
            || output.len() != block_size
        {
            return Err(format!(
                "AEC stream requires exact {block_size}-sample microphone, reference, and output blocks"
            ));
        }
        if microphone
            .iter()
            .chain(reference.iter())
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("AEC stream input samples must be finite normalized PCM".into());
        }
        let microphone_rms = rms(microphone);
        let reference_rms = rms(reference);
        if !self.reference_confident {
            output.copy_from_slice(microphone);
            self.counters.reference_uncertain_blocks =
                self.counters.reference_uncertain_blocks.saturating_add(1);
            let reset = std::mem::take(&mut self.pending_reset_marker);
            return Ok(AecBlockDiagnostics {
                talk_state: Some(AecTalkState::ReferenceUncertain),
                reference_rms,
                microphone_rms,
                linear_error_rms: microphone_rms,
                residual_gain: 1.0,
                adapted: false,
                state_reset: reset,
            });
        }
        let reference_active = reference_rms >= self.config.reference_activation_rms;
        if reference_active {
            self.reference_was_active = true;
        } else if self.reference_was_active {
            self.reset(AecResetReason::ReferenceDiscontinuity);
        }
        self.ring_position = if self.ring_position == 0 {
            self.partitions - 1
        } else {
            self.ring_position - 1
        };
        for index in 0..block_size {
            self.reference_fft[index] = Complex32::new(self.previous_reference[index], 0.0);
            self.reference_fft[block_size + index] = Complex32::new(reference[index], 0.0);
        }
        self.previous_reference.copy_from_slice(reference);
        self.forward
            .process_with_scratch(&mut self.reference_fft, &mut self.scratch);
        let current_history = self.ring_position * self.fft_size;
        self.reference_history[current_history..current_history + self.fft_size]
            .copy_from_slice(&self.reference_fft);
        self.estimate_fft.fill(Complex32::new(0.0, 0.0));
        for partition in 0..self.partitions {
            let history_partition = (self.ring_position + partition) % self.partitions;
            let history_offset = history_partition * self.fft_size;
            let filter_offset = partition * self.fft_size;
            for bin in 0..self.fft_size {
                self.estimate_fft[bin] +=
                    self.filter[filter_offset + bin] * self.reference_history[history_offset + bin];
            }
        }
        self.inverse
            .process_with_scratch(&mut self.estimate_fft, &mut self.scratch);
        let inverse_scale = 1.0 / self.fft_size as f32;
        for index in 0..block_size {
            let estimate = self.estimate_fft[block_size + index].re * inverse_scale;
            self.echo_estimate[index] = if estimate.is_finite() { estimate } else { 0.0 };
            self.linear_error[index] = microphone[index] - self.echo_estimate[index];
        }
        let linear_error_rms = rms(&self.linear_error);
        let echo_rms = rms(&self.echo_estimate);
        let correlation = normalized_correlation(microphone, reference).abs();
        let microphone_active = microphone_rms >= self.config.reference_activation_rms;
        let talk_state = if !reference_active && !microphone_active {
            AecTalkState::Silence
        } else if !reference_active {
            AecTalkState::NearEndOnly
        } else if !microphone_active {
            AecTalkState::FarEndOnly
        } else if correlation >= self.config.double_talk_correlation_threshold
            || (echo_rms > self.config.reference_activation_rms
                && linear_error_rms <= microphone_rms * 0.75)
        {
            AecTalkState::FarEndOnly
        } else {
            AecTalkState::DoubleTalk
        };
        match talk_state {
            AecTalkState::Silence => {
                self.counters.silence_blocks = self.counters.silence_blocks.saturating_add(1)
            }
            AecTalkState::FarEndOnly => {
                self.counters.far_end_only_blocks =
                    self.counters.far_end_only_blocks.saturating_add(1)
            }
            AecTalkState::NearEndOnly => {
                self.counters.near_end_only_blocks =
                    self.counters.near_end_only_blocks.saturating_add(1)
            }
            AecTalkState::DoubleTalk => {
                self.counters.double_talk_blocks =
                    self.counters.double_talk_blocks.saturating_add(1)
            }
            AecTalkState::ReferenceUncertain => {
                self.counters.reference_uncertain_blocks =
                    self.counters.reference_uncertain_blocks.saturating_add(1)
            }
        }
        let adapted = talk_state == AecTalkState::FarEndOnly;
        if adapted {
            self.error_fft.fill(Complex32::new(0.0, 0.0));
            for index in 0..block_size {
                self.error_fft[block_size + index] = Complex32::new(self.linear_error[index], 0.0);
            }
            self.forward
                .process_with_scratch(&mut self.error_fft, &mut self.scratch);
            for bin in 0..self.fft_size {
                let mut normalization = self.config.adaptation_regularization;
                for partition in 0..self.partitions {
                    let history_partition = (self.ring_position + partition) % self.partitions;
                    normalization +=
                        self.reference_history[history_partition * self.fft_size + bin].norm_sqr();
                }
                let step = self.config.adaptation_rate / normalization.max(POWER_FLOOR);
                for partition in 0..self.partitions {
                    let history_partition = (self.ring_position + partition) % self.partitions;
                    let history = self.reference_history[history_partition * self.fft_size + bin];
                    let filter_index = partition * self.fft_size + bin;
                    self.filter[filter_index] = self.filter[filter_index]
                        * self.config.filter_leakage
                        + history.conj() * self.error_fft[bin] * step;
                }
            }
            let constrained_partition = self.processed_blocks as usize % self.partitions;
            let filter_offset = constrained_partition * self.fft_size;
            self.projection_fft
                .copy_from_slice(&self.filter[filter_offset..filter_offset + self.fft_size]);
            self.inverse
                .process_with_scratch(&mut self.projection_fft, &mut self.scratch);
            for value in &mut self.projection_fft[..block_size] {
                *value *= inverse_scale;
            }
            self.projection_fft[block_size..].fill(Complex32::new(0.0, 0.0));
            self.forward
                .process_with_scratch(&mut self.projection_fft, &mut self.scratch);
            self.filter[filter_offset..filter_offset + self.fft_size]
                .copy_from_slice(&self.projection_fft);
            self.counters.adaptation_blocks = self.counters.adaptation_blocks.saturating_add(1);
        }
        let echo_power = echo_rms * echo_rms;
        let error_power = linear_error_rms * linear_error_rms;
        let echo_fraction = echo_power / (echo_power + error_power + POWER_FLOOR);
        let raw_gain = 1.0 - self.config.residual_suppression * echo_fraction.sqrt();
        let residual_gain = match talk_state {
            AecTalkState::FarEndOnly => raw_gain.clamp(self.config.minimum_far_end_gain, 1.0),
            AecTalkState::DoubleTalk => raw_gain.clamp(self.config.minimum_double_talk_gain, 1.0),
            _ => 1.0,
        };
        for index in 0..block_size {
            let candidate = self.linear_error[index] * residual_gain;
            let safe = if candidate.is_finite() {
                candidate.clamp(-self.config.maximum_peak, self.config.maximum_peak)
            } else {
                self.counters.non_finite_output_samples =
                    self.counters.non_finite_output_samples.saturating_add(1);
                microphone[index]
            };
            if safe != candidate && candidate.is_finite() {
                self.counters.clipped_samples = self.counters.clipped_samples.saturating_add(1);
            }
            output[index] = safe;
        }
        if talk_state == AecTalkState::FarEndOnly {
            self.counters.far_microphone_energy += microphone
                .iter()
                .map(|sample| (*sample as f64) * (*sample as f64))
                .sum::<f64>();
            self.counters.far_error_energy += output
                .iter()
                .map(|sample| (*sample as f64) * (*sample as f64))
                .sum::<f64>();
        }
        if self
            .filter
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            output.copy_from_slice(microphone);
            self.reset(AecResetReason::NonFiniteState);
        } else {
            self.processed_blocks = self.processed_blocks.saturating_add(1);
        }
        let state_reset = std::mem::take(&mut self.pending_reset_marker);
        Ok(AecBlockDiagnostics {
            talk_state: Some(talk_state),
            reference_rms,
            microphone_rms,
            linear_error_rms,
            residual_gain,
            adapted,
            state_reset,
        })
    }
}

impl AecRealtimeAdapter {
    fn new(stream: AecStream) -> Result<Self, String> {
        let block_size = stream.block_size_samples();
        let required = block_size
            .checked_mul(4)
            .ok_or("AEC real-time adapter buffer size overflow")?;
        if required > 16 * 1024 * 1024 {
            return Err("AEC real-time adapter exceeds its scalar buffer limit".into());
        }
        Ok(Self {
            stream,
            microphone_block: vec![0.0; block_size],
            reference_block: vec![0.0; block_size],
            processed_block: vec![0.0; block_size],
            latency_block: vec![0.0; block_size],
            input_fill: 0,
            latency_index: 0,
        })
    }

    pub fn latency_samples(&self) -> usize {
        self.latency_block.len()
    }

    pub fn route_generation(&self) -> u64 {
        self.stream.route_generation()
    }

    pub fn set_route_generation(&mut self, generation: u64) -> Result<(), String> {
        if generation != self.stream.route_generation() {
            self.stream.set_route_generation(generation)?;
            self.clear_partial_and_latency();
        }
        Ok(())
    }

    pub fn set_reference_confident(&mut self, confident: bool) {
        if confident != self.stream.reference_confident {
            self.stream.set_reference_confident(confident);
            self.clear_partial_and_latency();
        }
    }

    pub fn reset(&mut self, reason: AecResetReason) {
        self.stream.reset(reason);
        self.clear_partial_and_latency();
    }

    fn clear_partial_and_latency(&mut self) {
        self.microphone_block.fill(0.0);
        self.reference_block.fill(0.0);
        self.processed_block.fill(0.0);
        self.latency_block.fill(0.0);
        self.input_fill = 0;
        self.latency_index = 0;
    }

    pub fn process(
        &mut self,
        microphone: &[f32],
        reference: &[f32],
        output: &mut [f32],
    ) -> Result<AecRealtimeDiagnostics, String> {
        if microphone.len() != reference.len() || microphone.len() != output.len() {
            return Err(
                "AEC real-time adapter requires equal microphone, reference, and output quanta"
                    .into(),
            );
        }
        if microphone
            .iter()
            .chain(reference.iter())
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err("AEC real-time adapter input must be finite normalized PCM".into());
        }
        let block_size = self.latency_block.len();
        let mut completed_blocks = 0_u64;
        let mut last_block = None;
        for frame in 0..microphone.len() {
            output[frame] = self.latency_block[self.latency_index];
            self.latency_index += 1;
            if self.latency_index == block_size {
                self.latency_index = 0;
            }
            self.microphone_block[self.input_fill] = microphone[frame];
            self.reference_block[self.input_fill] = reference[frame];
            self.input_fill += 1;
            if self.input_fill == block_size {
                let diagnostics = self.stream.process_block(
                    &self.microphone_block,
                    &self.reference_block,
                    &mut self.processed_block,
                )?;
                self.latency_block.copy_from_slice(&self.processed_block);
                self.input_fill = 0;
                completed_blocks = completed_blocks.saturating_add(1);
                last_block = Some(diagnostics);
            }
        }
        Ok(AecRealtimeDiagnostics {
            completed_blocks,
            last_block,
            latency_samples: block_size,
        })
    }
}

pub fn estimate_aec_memory_bytes(config: &AecConfig) -> Result<u64, String> {
    if config.block_size_samples == 0 {
        return Err("AEC block size must be non-zero".into());
    }
    let fft_size = config
        .block_size_samples
        .checked_mul(2)
        .ok_or("AEC FFT size overflow")?;
    let partitions = config
        .tail_samples
        .div_ceil(config.block_size_samples.max(1));
    let complex_values = partitions
        .checked_mul(fft_size)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(fft_size.saturating_mul(7)))
        .ok_or("AEC complex working-set overflow")?;
    let scalar_values = config
        .block_size_samples
        .checked_mul(4)
        .ok_or("AEC scalar working-set overflow")?;
    let stream_bytes = (complex_values as u64)
        .checked_mul(std::mem::size_of::<Complex32>() as u64)
        .and_then(|value| {
            value.checked_add(
                (scalar_values as u64).saturating_mul(std::mem::size_of::<f32>() as u64),
            )
        })
        .ok_or("AEC working-set byte count overflow")?;
    let delay_reference_samples = config
        .delay_analysis_samples
        .checked_add(config.maximum_delay_samples)
        .ok_or("AEC delay reference size overflow")?;
    let delay_convolution_samples = config
        .delay_analysis_samples
        .checked_add(delay_reference_samples)
        .ok_or("AEC delay convolution size overflow")?;
    let delay_fft_size = delay_convolution_samples
        .max(2)
        .checked_next_power_of_two()
        .ok_or("AEC delay FFT size overflow")?;
    let delay_complex_bytes = (delay_fft_size as u64)
        .checked_mul(3)
        .and_then(|value| value.checked_mul(std::mem::size_of::<Complex32>() as u64))
        .ok_or("AEC delay FFT byte count overflow")?;
    let delay_prefix_bytes = config
        .delay_analysis_samples
        .checked_add(delay_reference_samples)
        .and_then(|value| value.checked_add(2))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
        .ok_or("AEC delay prefix byte count overflow")? as u64;
    let bytes = stream_bytes
        .checked_add(delay_complex_bytes)
        .and_then(|value| value.checked_add(delay_prefix_bytes))
        .ok_or("AEC aggregate working-set byte count overflow")?;
    if bytes > 512 * 1024 * 1024 {
        return Err("AEC working set exceeds the 512 MiB hard limit".into());
    }
    Ok(bytes)
}

fn map_reference_clock(
    reference: &[f64],
    mapping: &AecClockMapping,
    output_length: usize,
) -> Result<Vec<f64>, String> {
    let ratio = mapping.reference_sample_rate as f64
        * (1.0 + mapping.reference_clock_ppm / 1_000_000.0)
        / mapping.microphone_sample_rate as f64;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err("AEC clock mapping produced an invalid rate ratio".into());
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_length)
        .map_err(|_| "unable to reserve clock-mapped AEC reference".to_string())?;
    for index in 0..output_length {
        let position = index as f64 * ratio;
        let lower = position.floor() as usize;
        let fraction = position - lower as f64;
        let first = reference.get(lower).copied().unwrap_or(0.0);
        let second = reference
            .get(lower.saturating_add(1))
            .copied()
            .unwrap_or(first);
        output.push((first + (second - first) * fraction).clamp(-1.0, 1.0));
    }
    Ok(output)
}

fn estimate_delay(
    microphone: &[f64],
    reference: &[f64],
    config: &AecConfig,
    initial_delay_samples: i32,
) -> AecDelayEstimate {
    let analyzed = microphone.len().min(config.delay_analysis_samples);
    if analyzed < config.block_size_samples || reference.is_empty() {
        return AecDelayEstimate {
            signed_delay_samples: initial_delay_samples,
            confidence: 0.0,
            polarity_inverted: false,
            analyzed_samples: analyzed,
        };
    }
    let maximum_delay = config.maximum_delay_samples.min(i32::MAX as usize) as i32;
    let reference_samples = reference
        .len()
        .min(analyzed.saturating_add(maximum_delay as usize));
    let fft_size = analyzed
        .saturating_add(reference_samples)
        .max(2)
        .next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(fft_size);
    let inverse = planner.plan_fft_inverse(fft_size);
    let scratch_len = forward
        .get_inplace_scratch_len()
        .max(inverse.get_inplace_scratch_len());
    let zero = Complex32::new(0.0, 0.0);
    let mut microphone_fft = vec![zero; fft_size];
    let mut reference_fft = vec![zero; fft_size];
    let mut scratch = vec![zero; scratch_len];
    for (destination, sample) in microphone_fft.iter_mut().zip(&microphone[..analyzed]) {
        destination.re = *sample as f32;
    }
    for (destination, sample) in reference_fft
        .iter_mut()
        .zip(&reference[..reference_samples])
    {
        destination.re = *sample as f32;
    }
    forward.process_with_scratch(&mut microphone_fft, &mut scratch);
    forward.process_with_scratch(&mut reference_fft, &mut scratch);
    for bin in 0..fft_size {
        microphone_fft[bin] *= reference_fft[bin].conj();
    }
    inverse.process_with_scratch(&mut microphone_fft, &mut scratch);
    let mut microphone_energy = vec![0.0_f64; analyzed + 1];
    for (index, sample) in microphone[..analyzed].iter().enumerate() {
        microphone_energy[index + 1] = microphone_energy[index] + sample * sample;
    }
    let mut reference_energy = vec![0.0_f64; reference_samples + 1];
    for (index, sample) in reference[..reference_samples].iter().enumerate() {
        reference_energy[index + 1] = reference_energy[index] + sample * sample;
    }
    let minimum_overlap = analyzed.div_ceil(4).max(64);
    let inverse_scale = 1.0 / fft_size as f64;
    let mut best_lag = initial_delay_samples.clamp(-maximum_delay, maximum_delay);
    let mut best_correlation = 0.0_f64;
    for candidate in -maximum_delay..=maximum_delay {
        let microphone_start = candidate.max(0) as usize;
        let microphone_end =
            (reference_samples as i64 + candidate as i64).clamp(0, analyzed as i64) as usize;
        if microphone_end <= microphone_start || microphone_end - microphone_start < minimum_overlap
        {
            continue;
        }
        let reference_start = (microphone_start as i64 - candidate as i64) as usize;
        let reference_end = (microphone_end as i64 - candidate as i64) as usize;
        let microphone_power =
            microphone_energy[microphone_end] - microphone_energy[microphone_start];
        let reference_power = reference_energy[reference_end] - reference_energy[reference_start];
        if microphone_power <= 1.0e-12 || reference_power <= 1.0e-12 {
            continue;
        }
        let correlation_index = if candidate >= 0 {
            candidate as usize
        } else {
            fft_size - candidate.unsigned_abs() as usize
        };
        let cross = microphone_fft[correlation_index].re as f64 * inverse_scale;
        let correlation = cross / (microphone_power * reference_power).sqrt();
        if correlation.abs() > best_correlation.abs() {
            best_correlation = correlation;
            best_lag = candidate;
        }
    }
    AecDelayEstimate {
        signed_delay_samples: best_lag,
        confidence: best_correlation.abs().clamp(0.0, 1.0) as f32,
        polarity_inverted: best_correlation < 0.0,
        analyzed_samples: analyzed,
    }
}

fn validate_mono_audio(audio: &Audio, label: &str, require_nonempty: bool) -> Result<(), String> {
    if audio.channels() != 1 {
        return Err(format!(
            "{label} must contain exactly one typed mono channel"
        ));
    }
    if require_nonempty && audio.frames() == 0 {
        return Err(format!("{label} must contain at least one frame"));
    }
    let maximum_frames = (audio.sample_rate as u64)
        .checked_mul(MAX_AUDIO_SECONDS)
        .ok_or_else(|| format!("{label} duration bound overflow"))?;
    if audio.frames() as u64 > maximum_frames {
        return Err(format!("{label} exceeds the one-hour duration limit"));
    }
    if audio.channels[0]
        .iter()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(format!("{label} samples must be finite normalized PCM"));
    }
    Ok(())
}

fn digest_pcm(domain: &[u8], sample_rate: u32, samples: &[f64]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(sample_rate.to_le_bytes());
    digest.update((samples.len() as u64).to_le_bytes());
    for sample in samples {
        digest.update(sample.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn normalized_correlation(first: &[f32], second: &[f32]) -> f32 {
    let mut cross = 0.0_f64;
    let mut first_energy = 0.0_f64;
    let mut second_energy = 0.0_f64;
    for (first, second) in first.iter().zip(second) {
        cross += *first as f64 * *second as f64;
        first_energy += *first as f64 * *first as f64;
        second_energy += *second as f64 * *second as f64;
    }
    if first_energy <= POWER_FLOOR as f64 || second_energy <= POWER_FLOOR as f64 {
        0.0
    } else {
        (cross / (first_energy * second_energy).sqrt()) as f32
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let energy = samples
        .iter()
        .map(|sample| *sample as f64 * *sample as f64)
        .sum::<f64>();
    (energy / samples.len() as f64).sqrt() as f32
}

fn validate_f32_range(label: &str, value: f32, minimum: f32, maximum: f32) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "AEC {label} must be finite and in {minimum}..={maximum}"
        ));
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 256
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/' | b'+')
        })
    {
        return Err(format!("{label} is not a bounded portable identifier"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn test_config() -> AecConfig {
        AecConfig {
            sample_rate: 16_000,
            block_size_samples: 128,
            tail_samples: 2_048,
            maximum_delay_samples: 512,
            delay_analysis_samples: 4_096,
            ..AecConfig::default()
        }
    }

    fn passing_metric(policy: &MetricPolicy) -> AecEvidenceMetric {
        AecEvidenceMetric {
            metric: policy.name.into(),
            value: policy.hard_limit,
            operator: policy.operator,
            limit: policy.hard_limit,
            passed: true,
        }
    }

    fn passing_payload(config: &AecConfig) -> AecPromotionEvidencePayload {
        let strata = REQUIRED_STRATA
            .iter()
            .map(|(id, kind)| {
                let mut metrics = COMMON_METRICS
                    .iter()
                    .chain(kind_metrics(*kind).iter())
                    .map(passing_metric)
                    .collect::<Vec<_>>();
                metrics.sort_by(|left, right| left.metric.cmp(&right.metric));
                AecEvidenceStratum {
                    id: (*id).into(),
                    kind: *kind,
                    cases: 100,
                    metrics,
                }
            })
            .collect();
        AecPromotionEvidencePayload {
            completed_at_unix_seconds: 1_700_000_000,
            implementation: IMPLEMENTATION_ID.into(),
            implementation_source_revision: "0123456789abcdef".into(),
            implementation_source_sha256: "11".repeat(32),
            configuration_sha256: config.digest().unwrap(),
            corpus_manifest_sha256: "22".repeat(32),
            evaluation_result_sha256: "33".repeat(32),
            listening_result_sha256: "44".repeat(32),
            sample_rate: config.sample_rate,
            block_size_samples: config.block_size_samples,
            tail_samples: config.tail_samples,
            maximum_delay_samples: config.maximum_delay_samples,
            strata,
            real_device_cases: 100,
            nonlinear_device_cases: 100,
            delay_transition_cases: 100,
            paced_realtime_blocks: 10_000,
            worst_case_realtime_factor: 0.5,
            callback_allocations: 0,
            callback_locks: 0,
            callback_waits: 0,
            callback_io_operations: 0,
            callback_log_operations: 0,
            deadline_misses: 0,
            stale_frames_after_reset: 0,
            minimum_listeners: 20,
            listener_count: 20,
            listener_preference: 0.5,
            listener_preference_limit: 0.5,
            accepted: true,
        }
    }

    fn signed_session(config: AecConfig) -> AecSession {
        let (secret, public) = crate::generate_receipt_keypair().unwrap();
        let signed = sign_aec_promotion_evidence(passing_payload(&config), &secret).unwrap();
        AecSession::prepare(&signed, &public, config).unwrap()
    }

    fn pseudo_random(frames: usize) -> Vec<f64> {
        let mut state = 0x5eed_1234_u32;
        (0..frames)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f64 / u32::MAX as f64) * 1.6 - 0.8
            })
            .collect()
    }

    fn audio(sample_rate: u32, samples: Vec<f64>) -> Audio {
        Audio {
            sample_rate,
            channels: vec![samples],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    #[test]
    fn default_configuration_meets_latency_and_memory_limits() {
        let config = AecConfig::default();
        config.validate().unwrap();
        assert!(config.algorithmic_plus_buffering_milliseconds() <= 20.0);
        assert!(estimate_aec_memory_bytes(&config).unwrap() < 512 * 1024 * 1024);
    }

    #[test]
    fn evidence_is_signed_and_binds_the_exact_configuration() {
        let config = test_config();
        let (secret, public) = crate::generate_receipt_keypair().unwrap();
        let mut signed = sign_aec_promotion_evidence(passing_payload(&config), &secret).unwrap();
        AecSession::prepare(&signed, &public, config.clone()).unwrap();
        signed.payload.tail_samples += config.block_size_samples;
        assert!(signed.verify_signature(&public).is_err());

        let signed = sign_aec_promotion_evidence(passing_payload(&config), &secret).unwrap();
        let mut different = config;
        different.residual_suppression = 0.5;
        assert!(AecSession::prepare(&signed, &public, different).is_err());
    }

    #[test]
    fn evidence_rejects_weaker_limits_and_missing_required_strata() {
        let config = test_config();
        let mut payload = passing_payload(&config);
        let metric = payload.strata[0]
            .metrics
            .iter_mut()
            .find(|metric| metric.metric == "latency.algorithmic-plus-buffering-ms")
            .unwrap();
        metric.limit = 21.0;
        metric.value = 21.0;
        assert!(payload.validate().is_err());

        let mut payload = passing_payload(&config);
        payload.strata.pop();
        assert!(payload.validate().is_err());
    }

    #[test]
    fn signed_delay_estimator_represents_both_signs() {
        let config = test_config();
        let reference = pseudo_random(5_000);
        for expected_delay in [137_i32, -91_i32] {
            let mut microphone = vec![0.0; 4_096];
            for (frame, sample) in microphone.iter_mut().enumerate() {
                let reference_frame = frame as i64 - expected_delay as i64;
                if reference_frame >= 0 {
                    *sample = reference
                        .get(reference_frame as usize)
                        .copied()
                        .unwrap_or(0.0)
                        * 0.5;
                }
            }
            let estimate = estimate_delay(&microphone, &reference, &config, 0);
            assert_eq!(estimate.signed_delay_samples, expected_delay);
            assert!(estimate.confidence > 0.99);
        }
    }

    #[test]
    fn uncertain_reference_and_near_end_blocks_preserve_microphone() {
        let session = signed_session(test_config());
        let block_size = session.config().block_size_samples;
        let microphone = vec![0.25_f32; block_size];
        let reference = vec![0.0_f32; block_size];
        let mut output = vec![0.0_f32; block_size];

        let mut uncertain = session.stream(1, false).unwrap();
        let diagnostics = uncertain
            .process_block(&microphone, &reference, &mut output)
            .unwrap();
        assert_eq!(output, microphone);
        assert_eq!(
            diagnostics.talk_state,
            Some(AecTalkState::ReferenceUncertain)
        );
        assert!(!diagnostics.adapted);

        let mut confident = session.stream(1, true).unwrap();
        let diagnostics = confident
            .process_block(&microphone, &reference, &mut output)
            .unwrap();
        assert_eq!(output, microphone);
        assert_eq!(diagnostics.talk_state, Some(AecTalkState::NearEndOnly));
        assert!(!diagnostics.adapted);
    }

    #[test]
    fn route_change_cold_resets_filter_state() {
        let session = signed_session(test_config());
        let block_size = session.config().block_size_samples;
        let reference = pseudo_random(block_size)
            .into_iter()
            .map(|sample| sample as f32)
            .collect::<Vec<_>>();
        let microphone = reference
            .iter()
            .map(|sample| sample * 0.5)
            .collect::<Vec<_>>();
        let mut output = vec![0.0_f32; block_size];
        let mut stream = session.stream(1, true).unwrap();
        let first = stream
            .process_block(&microphone, &reference, &mut output)
            .unwrap();
        assert!(first.adapted);
        stream.set_route_generation(2).unwrap();
        let reset = stream
            .process_block(&microphone, &reference, &mut output)
            .unwrap();
        assert!(reset.state_reset);
        assert_eq!(stream.route_generation(), 2);
    }

    #[test]
    fn reference_loss_cold_resets_before_preserving_near_end() {
        let session = signed_session(test_config());
        let block_size = session.config().block_size_samples;
        let reference = pseudo_random(block_size)
            .into_iter()
            .map(|sample| sample as f32)
            .collect::<Vec<_>>();
        let microphone = reference
            .iter()
            .map(|sample| sample * 0.5)
            .collect::<Vec<_>>();
        let mut output = vec![0.0_f32; block_size];
        let mut stream = session.stream(1, true).unwrap();
        stream
            .process_block(&microphone, &reference, &mut output)
            .unwrap();

        let near_end = vec![0.125_f32; block_size];
        let missing_reference = vec![0.0_f32; block_size];
        let diagnostics = stream
            .process_block(&near_end, &missing_reference, &mut output)
            .unwrap();
        assert!(diagnostics.state_reset);
        assert_eq!(diagnostics.talk_state, Some(AecTalkState::NearEndOnly));
        assert_eq!(output, near_end);
    }

    #[test]
    fn realtime_adapter_accepts_arbitrary_quanta_with_fixed_block_latency() {
        let session = signed_session(test_config());
        let latency = session.config().block_size_samples;
        let microphone = (0..333)
            .map(|frame| ((frame as f32 * 0.031).sin() * 0.25).clamp(-1.0, 1.0))
            .collect::<Vec<_>>();
        let reference = vec![0.0_f32; microphone.len()];
        let mut output = vec![0.0_f32; microphone.len()];
        let mut adapter = session.realtime_adapter(3, true).unwrap();
        let mut offset = 0usize;
        for quantum in [17usize, 73, 5, 128, 64, 46] {
            let end = (offset + quantum).min(microphone.len());
            adapter
                .process(
                    &microphone[offset..end],
                    &reference[offset..end],
                    &mut output[offset..end],
                )
                .unwrap();
            offset = end;
            if offset == microphone.len() {
                break;
            }
        }
        assert_eq!(offset, microphone.len());
        assert!(output[..latency].iter().all(|sample| *sample == 0.0));
        assert_eq!(
            &output[latency..],
            &microphone[..microphone.len() - latency]
        );
        assert_eq!(adapter.latency_samples(), latency);

        adapter.set_route_generation(4).unwrap();
        let mut reset_output = vec![1.0_f32; latency];
        adapter
            .process(
                &microphone[..latency],
                &reference[..latency],
                &mut reset_output,
            )
            .unwrap();
        assert!(reset_output.iter().all(|sample| *sample == 0.0));
        assert_eq!(adapter.route_generation(), 4);
    }

    #[test]
    fn render_preserves_exact_geometry_and_never_records_paths() {
        let config = test_config();
        let session = signed_session(config.clone());
        let reference_samples = pseudo_random(5_000);
        let delay = 83_i32;
        let mut microphone_samples = vec![0.0; 3_333];
        for (frame, sample) in microphone_samples.iter_mut().enumerate() {
            let reference_frame = frame as i64 - delay as i64;
            if reference_frame >= 0 {
                *sample = reference_samples[reference_frame as usize] * 0.4;
            }
        }
        let microphone = audio(config.sample_rate, microphone_samples);
        let reference = audio(config.sample_rate, reference_samples);
        let result = session
            .render(
                &microphone,
                &reference,
                &AecClockMapping {
                    microphone_sample_rate: config.sample_rate,
                    reference_sample_rate: config.sample_rate,
                    reference_clock_ppm: 0.0,
                    initial_delay_samples: 0,
                    route_generation: 7,
                },
            )
            .unwrap();
        assert_eq!(result.audio.frames(), microphone.frames());
        assert_eq!(result.report.output_frames, microphone.frames());
        assert_eq!(result.report.delay.signed_delay_samples, delay);
        assert!(result.report.exact_output_duration);
        assert_eq!(result.report.non_finite_output_samples, 0);
        assert_eq!(result.report.paths_recorded, 0);
        assert!(!serde_json::to_string(&result.report)
            .unwrap()
            .contains("/tmp/"));
    }
}
