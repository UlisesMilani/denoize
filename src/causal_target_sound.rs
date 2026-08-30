//! Fail-closed causal extraction of one sound from an authenticated finite catalog.
//!
//! Every block publishes a complete decomposition. Accepted model candidates
//! publish `target` and an exact `input - target` residual. Absent, uncertain,
//! warm-up, unsafe, late, and overloaded blocks publish the conservative pair
//! `target = 0`, `residual = input`; partial semantic removal is never a fallback.

use crate::audio::{estimate_audio_memory_bytes, Audio};
use crate::backend::causal_target_sound::{
    CausalTargetSoundBackendSnapshot, CausalTargetSoundModel, CausalTargetSoundRuntime,
    CausalTargetSoundStateValue as BackendStateValue,
};
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
use crate::target_sound::{
    SignedTargetSoundPromotionEvidence, TargetSoundMetricOperator, TargetSoundMode,
    TargetSoundPresence, TargetSoundQuery, TargetSoundStratumKind, MAX_TARGET_SOUND_AUDIO_SECONDS,
};
use crate::{
    AcceleratorPreference, AcceleratorSelection, Backend, BackendOptions, RuntimeModelPackage,
};
use crossbeam_queue::ArrayQueue;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub const CAUSAL_TARGET_SOUND_EVIDENCE_SCHEMA: &str =
    "denoize-causal-target-sound-promotion-evidence-v1";
pub const CAUSAL_TARGET_SOUND_REPORT_SCHEMA: &str = "denoize-causal-target-sound-report-v1";
pub const CAUSAL_TARGET_SOUND_SNAPSHOT_SCHEMA: &str = "denoize-causal-target-sound-snapshot-v1";
pub const CAUSAL_TARGET_SOUND_SCHEMA_VERSION: u32 = 1;

const EVIDENCE_SIGNATURE_DOMAIN: &[u8] = b"denoize-causal-target-sound-promotion-evidence-v1";
const CONFIG_DIGEST_DOMAIN: &[u8] = b"denoize-causal-target-sound-config-v1\0";
const INPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-causal-target-sound-input-pcm-v1\0";
const TARGET_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-causal-target-sound-target-pcm-v1\0";
const RESIDUAL_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-causal-target-sound-residual-pcm-v1\0";
const OUTPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-causal-target-sound-output-pcm-v1\0";
const MAX_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;
const JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const MAX_EFFECTIVE_LATENCY_MILLISECONDS: f64 = 100.0;
const MAX_METRICS: usize = 16;
const REALTIME_QUEUE_BLOCKS: usize = 16;
const REALTIME_POOL_BLOCKS: usize = 40;
const REALTIME_WORKER_POLL: Duration = Duration::from_micros(100);
const SILENCE_FLOOR: f64 = 1.0e-12;

const REQUIRED_STRATA: &[(&str, TargetSoundStratumKind)] = &[
    ("binaural-spatial", TargetSoundStratumKind::BinauralSpatial),
    ("class-confusable", TargetSoundStratumKind::TargetPresent),
    ("clean-bypass", TargetSoundStratumKind::TargetAbsent),
    ("low-snr", TargetSoundStratumKind::TargetPresent),
    ("multi-instance", TargetSoundStratumKind::TargetPresent),
    (
        "music-foreground",
        TargetSoundStratumKind::ProtectedForeground,
    ),
    ("query-alias", TargetSoundStratumKind::TargetPresent),
    (
        "speech-foreground",
        TargetSoundStratumKind::ProtectedForeground,
    ),
    ("target-absent", TargetSoundStratumKind::TargetAbsent),
    ("target-present", TargetSoundStratumKind::TargetPresent),
    ("tonal-target", TargetSoundStratumKind::TargetPresent),
    ("transient-target", TargetSoundStratumKind::TargetPresent),
    ("unseen-domain", TargetSoundStratumKind::TargetPresent),
    ("unseen-interferer", TargetSoundStratumKind::TargetPresent),
];

#[derive(Clone, Copy)]
struct MetricPolicy {
    name: &'static str,
    operator: TargetSoundMetricOperator,
    hard_limit: f64,
    maximum_regression: f64,
}

impl MetricPolicy {
    const fn at_least(name: &'static str, hard_limit: f64, maximum_regression: f64) -> Self {
        Self {
            name,
            operator: TargetSoundMetricOperator::GreaterOrEqual,
            hard_limit,
            maximum_regression,
        }
    }

    const fn at_most(name: &'static str, hard_limit: f64, maximum_regression: f64) -> Self {
        Self {
            name,
            operator: TargetSoundMetricOperator::LessOrEqual,
            hard_limit,
            maximum_regression,
        }
    }
}

const PRESENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_least("extraction.target-si-sdr-improvement-db", 3.0, 0.5),
    MetricPolicy::at_most("output.clipped-samples", 0.0, 0.0),
    MetricPolicy::at_most("output.duration-mismatch-samples", 0.0, 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0, 0.0),
    MetricPolicy::at_least("output.protected-foreground-sdr-db", 20.0, 1.0),
    MetricPolicy::at_most("presence.expected-calibration-error", 0.05, 0.005),
    MetricPolicy::at_most("presence.false-negative-rate", 0.05, 0.005),
    MetricPolicy::at_most("recombination.maximum-absolute-error", 1.0e-5, 1.0e-6),
    MetricPolicy::at_most("residual.target-leakage-db", -20.0, 1.0),
];

const ABSENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("output.clipped-samples", 0.0, 0.0),
    MetricPolicy::at_most("output.duration-mismatch-samples", 0.0, 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0, 0.0),
    MetricPolicy::at_most("presence.expected-calibration-error", 0.05, 0.005),
    MetricPolicy::at_most("presence.false-positive-rate", 0.01, 0.002),
    MetricPolicy::at_most("recombination.maximum-absolute-error", 1.0e-5, 1.0e-6),
    MetricPolicy::at_most("target.output-rms-dbfs", -60.0, 3.0),
];

const BINAURAL_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_least("extraction.target-si-sdr-improvement-db", 3.0, 0.5),
    MetricPolicy::at_most("output.clipped-samples", 0.0, 0.0),
    MetricPolicy::at_most("output.duration-mismatch-samples", 0.0, 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0, 0.0),
    MetricPolicy::at_most("presence.expected-calibration-error", 0.05, 0.005),
    MetricPolicy::at_most("presence.false-negative-rate", 0.05, 0.005),
    MetricPolicy::at_most("recombination.maximum-absolute-error", 1.0e-5, 1.0e-6),
    MetricPolicy::at_most("residual.target-leakage-db", -20.0, 1.0),
    MetricPolicy::at_most("spatial.ild-error-db", 1.0, 0.2),
    MetricPolicy::at_most("spatial.itd-error-microseconds", 100.0, 20.0),
];

fn metric_policies(kind: TargetSoundStratumKind) -> &'static [MetricPolicy] {
    match kind {
        TargetSoundStratumKind::TargetPresent | TargetSoundStratumKind::ProtectedForeground => {
            PRESENT_METRICS
        }
        TargetSoundStratumKind::TargetAbsent => ABSENT_METRICS,
        TargetSoundStratumKind::BinauralSpatial => BINAURAL_METRICS,
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundMetricEvidence {
    pub metric: String,
    pub operator: TargetSoundMetricOperator,
    pub offline_value: f64,
    pub causal_value: f64,
    pub hard_limit: f64,
    pub maximum_regression: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundStratumEvidence {
    pub id: String,
    pub kind: TargetSoundStratumKind,
    pub offline_cases: u64,
    pub causal_cases: u64,
    pub metrics: Vec<CausalTargetSoundMetricEvidence>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundDeviceLatencyMeasurement {
    pub device_id: String,
    pub device_class: String,
    pub operating_system: String,
    pub audio_stack: String,
    pub sample_rate_hz: u32,
    pub channels: u32,
    pub capture_milliseconds: f64,
    pub chunk_milliseconds: f64,
    pub lookahead_milliseconds: f64,
    pub resampling_milliseconds: f64,
    pub inference_milliseconds: f64,
    pub buffering_milliseconds: f64,
    pub host_milliseconds: f64,
    pub output_milliseconds: f64,
    pub total_milliseconds: f64,
}

impl CausalTargetSoundDeviceLatencyMeasurement {
    fn validate(&self, limit: f64) -> Result<(), String> {
        validate_identifier("device ID", &self.device_id)?;
        for (label, value) in [
            ("device class", self.device_class.as_str()),
            ("operating system", self.operating_system.as_str()),
            ("audio stack", self.audio_stack.as_str()),
        ] {
            validate_text(label, value, 160)?;
        }
        let components = [
            self.capture_milliseconds,
            self.chunk_milliseconds,
            self.lookahead_milliseconds,
            self.resampling_milliseconds,
            self.inference_milliseconds,
            self.buffering_milliseconds,
            self.host_milliseconds,
            self.output_milliseconds,
        ];
        if !(8_000..=192_000).contains(&self.sample_rate_hz)
            || !(1..=2).contains(&self.channels)
            || components
                .iter()
                .chain(std::iter::once(&self.total_milliseconds))
                .any(|value| !value.is_finite() || !(0.0..=1_000.0).contains(value))
            || (components.iter().sum::<f64>() - self.total_milliseconds).abs() > 0.1
            || self.total_milliseconds > limit
        {
            return Err("causal target-sound device latency measurement is inconsistent".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundRealtimeAudit {
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

impl CausalTargetSoundRealtimeAudit {
    fn validate(&self) -> Result<(), String> {
        if self.paced_blocks < 10_000
            || self.paced_blocks > JSON_SAFE_INTEGER
            || self.deadline_misses > JSON_SAFE_INTEGER
            || self.overload_blocks > JSON_SAFE_INTEGER
            || !(16..=256).contains(&self.queue_capacity_blocks)
            || self.maximum_queue_depth_blocks >= self.queue_capacity_blocks
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
            .any(|value| value > JSON_SAFE_INTEGER)
        {
            return Err("causal target-sound realtime audit is outside schema bounds".into());
        }
        Ok(())
    }

    fn passed(&self) -> bool {
        self.deadline_misses == 0
            && self.overload_blocks == 0
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
pub struct CausalTargetSoundTransitionAudit {
    pub reset_cases: u32,
    pub discontinuity_cases: u32,
    pub dropout_cases: u32,
    pub overload_fallback_cases: u32,
    pub snapshot_roundtrip_cases: u32,
    pub resampler_boundary_cases: u32,
    pub query_mutation_rejections: u32,
    pub late_results_injected: u32,
    pub late_results_discarded: u32,
    pub stale_generation_results_injected: u32,
    pub stale_generation_results_discarded: u32,
    pub partial_semantic_removal_publications: u32,
    pub recombination_violations: u32,
}

impl CausalTargetSoundTransitionAudit {
    fn validate(&self) -> Result<(), String> {
        for value in [
            self.reset_cases,
            self.discontinuity_cases,
            self.dropout_cases,
            self.overload_fallback_cases,
            self.snapshot_roundtrip_cases,
            self.resampler_boundary_cases,
            self.query_mutation_rejections,
            self.late_results_injected,
            self.stale_generation_results_injected,
        ] {
            if !(100..=1_000_000).contains(&value) {
                return Err(
                    "causal target-sound transition audit case count is outside bounds".into(),
                );
            }
        }
        for value in [
            self.late_results_discarded,
            self.stale_generation_results_discarded,
            self.partial_semantic_removal_publications,
            self.recombination_violations,
        ] {
            if value > 1_000_000 {
                return Err("causal target-sound transition audit result is outside bounds".into());
            }
        }
        Ok(())
    }

    fn passed(&self) -> bool {
        self.late_results_discarded == self.late_results_injected
            && self.stale_generation_results_discarded == self.stale_generation_results_injected
            && self.partial_semantic_removal_publications == 0
            && self.recombination_violations == 0
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub offline_model_package_sha256: String,
    pub causal_model_package_sha256: String,
    pub causal_source_revision: String,
    pub causal_source_sha256: String,
    pub causal_checkpoint_sha256: String,
    pub offline_configuration_sha256: String,
    pub causal_configuration_sha256: String,
    pub query_catalog_sha256: String,
    pub query_catalog_revision: String,
    pub query_class_ids_sha256: String,
    pub query_class_count: u32,
    pub offline_evaluation_result_sha256: String,
    pub causal_evaluation_result_sha256: String,
    pub state_reset_flush_result_sha256: String,
    pub snapshot_roundtrip_result_sha256: String,
    pub recombination_result_sha256: String,
    pub latency_result_sha256: String,
    pub realtime_callback_result_sha256: String,
    pub transition_result_sha256: String,
    pub strata: Vec<CausalTargetSoundStratumEvidence>,
    pub model_sample_rate_hz: u32,
    pub model_channels: u32,
    pub frame_samples: u64,
    pub algorithmic_latency_samples: u64,
    pub flush_samples: u64,
    pub perturbation_latency_cases: u32,
    pub effective_latency_limit_milliseconds: f64,
    pub worst_effective_latency_milliseconds: f64,
    pub device_measurements: Vec<CausalTargetSoundDeviceLatencyMeasurement>,
    pub realtime: CausalTargetSoundRealtimeAudit,
    pub transitions: CausalTargetSoundTransitionAudit,
    pub accepted: bool,
}

impl CausalTargetSoundPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        if self.completed_at_unix_seconds > JSON_SAFE_INTEGER {
            return Err("causal target-sound evidence timestamp exceeds JSON safe integer".into());
        }
        validate_identifier("causal source revision", &self.causal_source_revision)?;
        validate_identifier("query catalog revision", &self.query_catalog_revision)?;
        for (label, value) in [
            (
                "offline model package",
                self.offline_model_package_sha256.as_str(),
            ),
            (
                "causal model package",
                self.causal_model_package_sha256.as_str(),
            ),
            ("causal source", self.causal_source_sha256.as_str()),
            ("causal checkpoint", self.causal_checkpoint_sha256.as_str()),
            (
                "offline configuration",
                self.offline_configuration_sha256.as_str(),
            ),
            (
                "causal configuration",
                self.causal_configuration_sha256.as_str(),
            ),
            ("query catalog", self.query_catalog_sha256.as_str()),
            ("query class IDs", self.query_class_ids_sha256.as_str()),
            (
                "offline evaluation result",
                self.offline_evaluation_result_sha256.as_str(),
            ),
            (
                "causal evaluation result",
                self.causal_evaluation_result_sha256.as_str(),
            ),
            (
                "state reset flush result",
                self.state_reset_flush_result_sha256.as_str(),
            ),
            (
                "snapshot roundtrip result",
                self.snapshot_roundtrip_result_sha256.as_str(),
            ),
            (
                "recombination result",
                self.recombination_result_sha256.as_str(),
            ),
            ("latency result", self.latency_result_sha256.as_str()),
            (
                "realtime callback result",
                self.realtime_callback_result_sha256.as_str(),
            ),
            ("transition result", self.transition_result_sha256.as_str()),
        ] {
            validate_sha256(label, value)?;
        }
        if !(2..=4096).contains(&self.query_class_count)
            || !(8_000..=192_000).contains(&self.model_sample_rate_hz)
            || !(1..=2).contains(&self.model_channels)
            || self.frame_samples == 0
            || self.frame_samples > 262_144
            || self.algorithmic_latency_samples > 262_144
            || self.flush_samples > 262_144
            || self.flush_samples < self.algorithmic_latency_samples
            || !(100..=1_000_000).contains(&self.perturbation_latency_cases)
            || !self.effective_latency_limit_milliseconds.is_finite()
            || !(0.0..=MAX_EFFECTIVE_LATENCY_MILLISECONDS)
                .contains(&self.effective_latency_limit_milliseconds)
            || !self.worst_effective_latency_milliseconds.is_finite()
            || self.worst_effective_latency_milliseconds < 0.0
        {
            return Err("causal target-sound signed geometry or latency is invalid".into());
        }
        let signed_latency = self
            .algorithmic_latency_samples
            .saturating_mul(1000)
            .div_ceil(u64::from(self.model_sample_rate_hz));
        if signed_latency > 100 {
            return Err("causal target-sound signed algorithmic latency exceeds 100 ms".into());
        }
        if self.strata.len() != REQUIRED_STRATA.len() {
            return Err("causal target-sound evidence has the wrong stratum count".into());
        }
        let mut metrics_passed = true;
        for (stratum, (id, kind)) in self.strata.iter().zip(REQUIRED_STRATA) {
            if stratum.id != *id || stratum.kind != *kind {
                return Err("causal target-sound evidence strata must be exact and sorted".into());
            }
            if !(50..=1_000_000).contains(&stratum.offline_cases)
                || !(50..=1_000_000).contains(&stratum.causal_cases)
            {
                return Err("causal target-sound stratum case counts are outside bounds".into());
            }
            let policies = metric_policies(stratum.kind);
            if stratum.metrics.len() != policies.len() || stratum.metrics.len() > MAX_METRICS {
                return Err("causal target-sound stratum has the wrong metric set".into());
            }
            for (metric, policy) in stratum.metrics.iter().zip(policies) {
                validate_metric(metric, policy)?;
                metrics_passed &= metric.passed;
            }
        }
        if !(3..=64).contains(&self.device_measurements.len()) {
            return Err(
                "causal target-sound evidence requires 3..=64 named device measurements".into(),
            );
        }
        let mut device_ids = BTreeSet::new();
        let mut worst = 0.0_f64;
        for device in &self.device_measurements {
            device.validate(self.effective_latency_limit_milliseconds)?;
            if !device_ids.insert(device.device_id.as_str()) {
                return Err("causal target-sound device IDs must be unique".into());
            }
            worst = worst.max(device.total_milliseconds);
        }
        if (worst - self.worst_effective_latency_milliseconds).abs() > 0.001 {
            return Err("causal target-sound worst effective latency is inconsistent".into());
        }
        self.realtime.validate()?;
        self.transitions.validate()?;
        let expected = metrics_passed
            && worst <= self.effective_latency_limit_milliseconds
            && self.realtime.passed()
            && self.transitions.passed();
        if self.accepted != expected {
            return Err("causal target-sound accepted flag is inconsistent".into());
        }
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize causal target-sound evidence payload: {error}"))?;
        if encoded.len() as u64 >= MAX_EVIDENCE_BYTES {
            return Err("causal target-sound evidence exceeds the bounded JSON limit".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedCausalTargetSoundPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: CausalTargetSoundPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedCausalTargetSoundPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) =
            crate::input::open_regular_file(path, "causal target-sound promotion evidence")?;
        if length >= MAX_EVIDENCE_BYTES {
            return Err(format!(
                "causal target-sound evidence {} exceeds {MAX_EVIDENCE_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve causal target-sound evidence JSON".to_string())?;
        file.take(MAX_EVIDENCE_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read causal target-sound evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("causal target-sound evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse causal target-sound evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != CAUSAL_TARGET_SOUND_EVIDENCE_SCHEMA
            || self.schema_version != CAUSAL_TARGET_SOUND_SCHEMA_VERSION
        {
            return Err("unsupported causal target-sound evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("causal target-sound evidence signature must use ed25519".into());
        }
        validate_sha256("evidence key ID", &self.signature.key_id)?;
        Ok(())
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize causal target-sound evidence: {error}"))?;
        key.verify_domain_document(
            EVIDENCE_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "causal target-sound promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize causal target-sound evidence: {error}"))
    }
}

pub fn sign_causal_target_sound_promotion_evidence(
    payload: CausalTargetSoundPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedCausalTargetSoundPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize causal target-sound evidence: {error}"))?;
    let signature = key.sign_domain_document(
        EVIDENCE_SIGNATURE_DOMAIN,
        &document,
        "causal target-sound promotion evidence",
    )?;
    let evidence = SignedCausalTargetSoundPromotionEvidence {
        schema: CAUSAL_TARGET_SOUND_EVIDENCE_SCHEMA.into(),
        schema_version: CAUSAL_TARGET_SOUND_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundConfig {
    pub mode: TargetSoundMode,
    pub minimum_present_probability: f64,
    pub minimum_absent_probability: f64,
    pub present_hold_blocks: u32,
    pub maximum_model_recombination_error: f64,
    pub maximum_publication_recombination_error: f64,
    pub maximum_target_peak: f64,
    pub maximum_residual_peak: f64,
    pub maximum_energy_gain_db: f64,
    pub maximum_stereo_correlation_delta: f64,
    pub maximum_mid_side_energy_ratio_delta_db: f64,
}

impl Default for CausalTargetSoundConfig {
    fn default() -> Self {
        Self {
            mode: TargetSoundMode::Preserve,
            minimum_present_probability: 0.90,
            minimum_absent_probability: 0.90,
            present_hold_blocks: 3,
            maximum_model_recombination_error: 0.01,
            maximum_publication_recombination_error: 1.0e-6,
            maximum_target_peak: 1.0,
            maximum_residual_peak: 1.0,
            maximum_energy_gain_db: 3.0,
            maximum_stereo_correlation_delta: 0.05,
            maximum_mid_side_energy_ratio_delta_db: 1.5,
        }
    }
}

impl CausalTargetSoundConfig {
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
            return Err("causal target-sound present hold must be in 1..=100 blocks".into());
        }
        validate_range(
            "maximum model recombination error",
            self.maximum_model_recombination_error,
            0.0,
            0.10,
        )?;
        validate_range(
            "maximum publication recombination error",
            self.maximum_publication_recombination_error,
            0.0,
            1.0e-5,
        )?;
        validate_range("maximum target peak", self.maximum_target_peak, 0.5, 1.0)?;
        validate_range(
            "maximum residual peak",
            self.maximum_residual_peak,
            0.5,
            1.0,
        )?;
        validate_range(
            "maximum energy gain dB",
            self.maximum_energy_gain_db,
            0.0,
            12.0,
        )?;
        validate_range(
            "maximum stereo correlation delta",
            self.maximum_stereo_correlation_delta,
            0.0,
            0.25,
        )?;
        validate_range(
            "maximum mid/side energy ratio delta dB",
            self.maximum_mid_side_energy_ratio_delta_db,
            0.0,
            6.0,
        )
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        digest_json(
            CONFIG_DIGEST_DOMAIN,
            self,
            "causal target-sound configuration",
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CausalTargetSoundBlockDecision {
    PublishedPresent,
    FallbackAbsent,
    FallbackUncertain,
    FallbackPresentWarmup,
    FallbackSafetyGate,
    FallbackFlush,
    FallbackOverload,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundBlock {
    pub generation: u64,
    pub start_frame: u64,
    pub valid_frames: usize,
    pub channels: usize,
    pub target: Vec<f32>,
    pub residual: Vec<f32>,
    pub presence: TargetSoundPresence,
    pub absent_probability: f32,
    pub uncertain_probability: f32,
    pub present_probability: f32,
    pub model_recombination_maximum_absolute_error: f64,
    pub publication_recombination_maximum_absolute_error: f64,
    pub decision: CausalTargetSoundBlockDecision,
    pub candidate_accepted: bool,
}

impl CausalTargetSoundBlock {
    #[must_use]
    pub fn selected(&self, mode: TargetSoundMode) -> &[f32] {
        match mode {
            TargetSoundMode::Preserve => &self.target,
            TargetSoundMode::Remove => &self.residual,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "element_type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum CausalTargetSoundSnapshotState {
    Float32 { shape: Vec<usize>, values: Vec<f32> },
    Int64 { shape: Vec<usize>, values: Vec<i64> },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundSnapshot {
    pub schema: String,
    pub schema_version: u32,
    pub model_package_sha256: String,
    pub configuration_sha256: String,
    pub query_sha256: String,
    pub query_catalog_sha256: String,
    pub selected_class_id: String,
    pub snapshot_generation: u64,
    pub next_frame: u64,
    pub present_streak: u32,
    pub states: Vec<CausalTargetSoundSnapshotState>,
}

impl CausalTargetSoundSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != CAUSAL_TARGET_SOUND_SNAPSHOT_SCHEMA
            || self.schema_version != CAUSAL_TARGET_SOUND_SCHEMA_VERSION
        {
            return Err("unsupported causal target-sound snapshot schema".into());
        }
        for (label, digest) in [
            ("model package", self.model_package_sha256.as_str()),
            ("configuration", self.configuration_sha256.as_str()),
            ("query", self.query_sha256.as_str()),
            ("query catalog", self.query_catalog_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
        validate_identifier("selected class ID", &self.selected_class_id)?;
        if self.snapshot_generation == 0
            || self.snapshot_generation > JSON_SAFE_INTEGER
            || self.next_frame > JSON_SAFE_INTEGER
            || self.present_streak > 100
            || self.states.is_empty()
            || self.states.len() > 64
        {
            return Err("causal target-sound snapshot metadata is outside bounds".into());
        }
        let mut elements = 0_usize;
        for state in &self.states {
            let (shape, length, finite) = match state {
                CausalTargetSoundSnapshotState::Float32 { shape, values } => (
                    shape,
                    values.len(),
                    values.iter().all(|value| value.is_finite()),
                ),
                CausalTargetSoundSnapshotState::Int64 { shape, values } => {
                    (shape, values.len(), true)
                }
            };
            if shape.is_empty()
                || shape.len() > 8
                || shape.iter().any(|axis| *axis == 0 || *axis > 16_777_216)
                || shape
                    .iter()
                    .try_fold(1_usize, |total, axis| total.checked_mul(*axis))
                    != Some(length)
                || !finite
            {
                return Err("causal target-sound snapshot state is invalid".into());
            }
            elements = elements
                .checked_add(length)
                .ok_or_else(|| "causal target-sound snapshot size overflow".to_string())?;
            if elements > 8_388_608 {
                return Err("causal target-sound snapshot has too many state elements".into());
            }
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize causal target-sound snapshot: {error}"))?;
        if bytes.len() >= MAX_SNAPSHOT_BYTES {
            return Err("causal target-sound snapshot exceeds the bounded JSON limit".into());
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() >= MAX_SNAPSHOT_BYTES {
            return Err("causal target-sound snapshot exceeds the bounded JSON limit".into());
        }
        let snapshot: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse causal target-sound snapshot: {error}"))?;
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize causal target-sound snapshot: {error}"))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundQueryIdentity {
    pub query_sha256: String,
    pub catalog_sha256: String,
    pub catalog_revision: String,
    pub class_ids_sha256: String,
    pub class_count: u32,
    pub class_id: String,
    pub class_index: u32,
    pub canonical_label: String,
    pub encoding: String,
    pub open_text_accepted: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundModelIdentity {
    pub package_sha256: String,
    pub public_key_sha256: String,
    pub package_id: String,
    pub package_revision: String,
    pub precision_profile: String,
    pub package_license_spdx: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub source_license_spdx: String,
    pub checkpoint_sha256: String,
    pub checkpoint_license_spdx: String,
    pub accelerator: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundEvidenceIdentity {
    pub offline_signing_key_id: String,
    pub causal_signing_key_id: String,
    pub offline_model_package_sha256: String,
    pub offline_evaluation_result_sha256: String,
    pub causal_evaluation_result_sha256: String,
    pub state_reset_flush_result_sha256: String,
    pub snapshot_roundtrip_result_sha256: String,
    pub recombination_result_sha256: String,
    pub latency_result_sha256: String,
    pub realtime_callback_result_sha256: String,
    pub transition_result_sha256: String,
    pub device_measurements: u32,
    pub strata: u32,
    pub accepted: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundDecisionCounts {
    pub published_present_blocks: u64,
    pub fallback_absent_blocks: u64,
    pub fallback_uncertain_blocks: u64,
    pub fallback_present_warmup_blocks: u64,
    pub fallback_safety_gate_blocks: u64,
    pub fallback_flush_blocks: u64,
    pub fallback_overload_blocks: u64,
}

impl CausalTargetSoundDecisionCounts {
    fn observe(&mut self, decision: CausalTargetSoundBlockDecision) {
        match decision {
            CausalTargetSoundBlockDecision::PublishedPresent => self.published_present_blocks += 1,
            CausalTargetSoundBlockDecision::FallbackAbsent => self.fallback_absent_blocks += 1,
            CausalTargetSoundBlockDecision::FallbackUncertain => {
                self.fallback_uncertain_blocks += 1;
            }
            CausalTargetSoundBlockDecision::FallbackPresentWarmup => {
                self.fallback_present_warmup_blocks += 1;
            }
            CausalTargetSoundBlockDecision::FallbackSafetyGate => {
                self.fallback_safety_gate_blocks += 1;
            }
            CausalTargetSoundBlockDecision::FallbackFlush => self.fallback_flush_blocks += 1,
            CausalTargetSoundBlockDecision::FallbackOverload => self.fallback_overload_blocks += 1,
        }
    }

    fn total(&self) -> u64 {
        self.published_present_blocks
            .saturating_add(self.fallback_absent_blocks)
            .saturating_add(self.fallback_uncertain_blocks)
            .saturating_add(self.fallback_present_warmup_blocks)
            .saturating_add(self.fallback_safety_gate_blocks)
            .saturating_add(self.fallback_flush_blocks)
            .saturating_add(self.fallback_overload_blocks)
    }

    #[must_use]
    pub fn fallback_blocks(&self) -> u64 {
        self.total().saturating_sub(self.published_present_blocks)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTargetSoundRenderReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub network_accessed: bool,
    pub deterministic: bool,
    pub mode: TargetSoundMode,
    pub configuration_sha256: String,
    pub query: CausalTargetSoundQueryIdentity,
    pub model: CausalTargetSoundModelIdentity,
    pub promotion_evidence: CausalTargetSoundEvidenceIdentity,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_frames: usize,
    pub model_sample_rate: u32,
    pub model_channels: usize,
    pub frame_samples: usize,
    pub algorithmic_latency_samples: usize,
    pub flush_samples: usize,
    pub input_blocks: u64,
    pub flush_blocks: u64,
    pub decision_counts: CausalTargetSoundDecisionCounts,
    pub presence_transitions: u64,
    pub source_clock_withheld_frames: usize,
    pub source_clock_conservative_fallback: bool,
    pub target_published: bool,
    pub residual_published: bool,
    pub output_published: bool,
    pub input_pcm_sha256: String,
    pub target_pcm_sha256: String,
    pub residual_pcm_sha256: String,
    pub output_pcm_sha256: String,
    pub maximum_model_recombination_error: f64,
    pub maximum_publication_recombination_error: f64,
    pub partial_semantic_removal_fallbacks: u64,
    pub path_fields_recorded: u32,
    pub limitations: Vec<String>,
    pub warnings: Vec<String>,
}

impl CausalTargetSoundRenderReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != CAUSAL_TARGET_SOUND_REPORT_SCHEMA
            || self.schema_version != CAUSAL_TARGET_SOUND_SCHEMA_VERSION
            || self.denoize_version != env!("CARGO_PKG_VERSION")
            || self.network_accessed
            || !self.deterministic
            || self.configuration_sha256.len() != 64
            || !(8_000..=192_000).contains(&self.source_sample_rate)
            || !(1..=2).contains(&self.source_channels)
            || self.source_frames == 0
            || self.source_frames as u64
                > u64::from(self.source_sample_rate).saturating_mul(MAX_TARGET_SOUND_AUDIO_SECONDS)
            || !(8_000..=192_000).contains(&self.model_sample_rate)
            || self.model_channels != self.source_channels
            || self.frame_samples == 0
            || self.input_blocks == 0
            || self.decision_counts.total() != self.input_blocks + self.flush_blocks
            || self.source_clock_withheld_frames > self.source_frames
            || !self.target_published
            || !self.residual_published
            || !self.output_published
            || !self.maximum_model_recombination_error.is_finite()
            || !self.maximum_publication_recombination_error.is_finite()
            || !(0.0..=2.0).contains(&self.maximum_model_recombination_error)
            || !(0.0..=2.0).contains(&self.maximum_publication_recombination_error)
            || self.partial_semantic_removal_fallbacks != 0
            || self.path_fields_recorded != 0
            || self.limitations.is_empty()
            || self.limitations.len() > 32
            || self.warnings.len() > 32
        {
            return Err("causal target-sound report violates bounded result contracts".into());
        }
        for (label, digest) in [
            ("configuration", self.configuration_sha256.as_str()),
            ("input PCM", self.input_pcm_sha256.as_str()),
            ("target PCM", self.target_pcm_sha256.as_str()),
            ("residual PCM", self.residual_pcm_sha256.as_str()),
            ("output PCM", self.output_pcm_sha256.as_str()),
            ("model package", self.model.package_sha256.as_str()),
            ("model public key", self.model.public_key_sha256.as_str()),
            ("query", self.query.query_sha256.as_str()),
            ("catalog", self.query.catalog_sha256.as_str()),
            ("class IDs", self.query.class_ids_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
        if self.query.open_text_accepted
            || self.query.encoding != "one-hot-v1"
            || self.query.class_count < 2
            || self.query.class_count > 4096
            || self.query.class_index >= self.query.class_count
            || self.query.class_id.is_empty()
            || self.query.canonical_label.is_empty()
            || !self.promotion_evidence.accepted
            || self.promotion_evidence.strata != REQUIRED_STRATA.len() as u32
            || self.promotion_evidence.device_measurements < 3
        {
            return Err("causal target-sound report identity is inconsistent".into());
        }
        for text in self.limitations.iter().chain(&self.warnings) {
            validate_text("report text", text, 512)?;
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize causal target-sound report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize causal target-sound report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct CausalTargetSoundRenderResult {
    pub target: Audio,
    pub residual: Audio,
    pub output: Audio,
    pub report: CausalTargetSoundRenderReport,
}

pub fn estimate_causal_target_sound_memory_bytes(
    input: &Audio,
    model_sample_rate: u32,
    model_channels: usize,
    frame_samples: usize,
    flush_samples: usize,
) -> Result<u64, String> {
    if input.sample_rate == 0
        || input.channels.is_empty()
        || !(8_000..=192_000).contains(&model_sample_rate)
        || !(1..=2).contains(&model_channels)
        || frame_samples == 0
    {
        return Err("causal target-sound memory geometry is invalid".into());
    }
    let model_frames = crate::resample::planned_output_frames(
        input.frames(),
        input.sample_rate,
        model_sample_rate,
    )?;
    let model = (model_frames as u128)
        .checked_add(flush_samples as u128)
        .and_then(|value| value.checked_mul(model_channels as u128))
        .and_then(|value| value.checked_mul(24))
        .ok_or_else(|| "causal target-sound memory estimate overflow".to_string())?;
    let frames = (frame_samples as u128)
        .checked_mul(model_channels as u128)
        .and_then(|value| value.checked_mul(16))
        .ok_or_else(|| "causal target-sound frame estimate overflow".to_string())?;
    let source = u128::from(estimate_audio_memory_bytes(input)).saturating_mul(8);
    let bytes = model
        .checked_add(frames)
        .and_then(|value| value.checked_add(source))
        .and_then(|value| value.checked_add(1024 * 1024))
        .ok_or_else(|| "causal target-sound memory estimate overflow".to_string())?;
    u64::try_from(bytes).map_err(|_| "causal target-sound memory estimate exceeds u64".into())
}

pub struct CausalTargetSoundSession {
    package: RuntimeModelPackage,
    model: CausalTargetSoundModel,
    accelerator: AcceleratorSelection,
    config: CausalTargetSoundConfig,
    configuration_sha256: String,
    query: CausalTargetSoundQueryIdentity,
    evidence: CausalTargetSoundEvidenceIdentity,
}

impl std::fmt::Debug for CausalTargetSoundSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CausalTargetSoundSession")
            .field("package_sha256", &self.package.package_sha256())
            .field("accelerator", &self.accelerator)
            .field("query", &self.query.class_id)
            .field("sample_rate_hz", &self.model.sample_rate_hz())
            .field("frame_samples", &self.model.frame_samples())
            .finish_non_exhaustive()
    }
}

impl CausalTargetSoundSession {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        package: RuntimeModelPackage,
        offline_evidence: &SignedTargetSoundPromotionEvidence,
        offline_evidence_key: &ReceiptPublicKey,
        causal_evidence: &SignedCausalTargetSoundPromotionEvidence,
        causal_evidence_key: &ReceiptPublicKey,
        query: &TargetSoundQuery,
        config: &CausalTargetSoundConfig,
        requested: AcceleratorPreference,
    ) -> Result<Self, String> {
        query.validate()?;
        config.validate()?;
        offline_evidence.verify_signature(offline_evidence_key)?;
        causal_evidence.verify_signature(causal_evidence_key)?;
        if !offline_evidence.payload.accepted || !causal_evidence.payload.accepted {
            return Err(
                "causal target-sound requires accepted offline and causal promotion evidence"
                    .into(),
            );
        }
        validate_offline_matrix_binding(offline_evidence, causal_evidence)?;
        let manifest = package
            .manifest_v2()
            .ok_or("causal target-sound rejects runtime model package v1")?;
        let payload = &causal_evidence.payload;
        let configuration_sha256 = config.digest()?;
        let query_sha256 = query.digest()?;
        let catalog_sha256 = query.catalog_sha256()?;
        let class_ids_sha256 = query.class_ids_sha256()?;
        for (label, observed, expected) in [
            (
                "offline model package SHA-256",
                payload.offline_model_package_sha256.as_str(),
                offline_evidence.payload.model_package_sha256.as_str(),
            ),
            (
                "causal model package SHA-256",
                payload.causal_model_package_sha256.as_str(),
                package.package_sha256(),
            ),
            (
                "causal source revision",
                payload.causal_source_revision.as_str(),
                manifest.provenance.source_revision.as_str(),
            ),
            (
                "causal source SHA-256",
                payload.causal_source_sha256.as_str(),
                manifest.provenance.source_sha256.as_str(),
            ),
            (
                "causal checkpoint SHA-256",
                payload.causal_checkpoint_sha256.as_str(),
                manifest.provenance.checkpoint_sha256.as_str(),
            ),
            (
                "offline configuration SHA-256",
                payload.offline_configuration_sha256.as_str(),
                offline_evidence.payload.configuration_sha256.as_str(),
            ),
            (
                "causal configuration SHA-256",
                payload.causal_configuration_sha256.as_str(),
                configuration_sha256.as_str(),
            ),
            (
                "query catalog SHA-256",
                payload.query_catalog_sha256.as_str(),
                catalog_sha256.as_str(),
            ),
            (
                "query catalog revision",
                payload.query_catalog_revision.as_str(),
                query.catalog_revision.as_str(),
            ),
            (
                "query class IDs SHA-256",
                payload.query_class_ids_sha256.as_str(),
                class_ids_sha256.as_str(),
            ),
            (
                "offline evaluation result SHA-256",
                payload.offline_evaluation_result_sha256.as_str(),
                offline_evidence.payload.evaluation_result_sha256.as_str(),
            ),
        ] {
            if observed != expected {
                return Err(format!(
                    "causal target-sound promotion evidence {label} does not match the authenticated prerequisite"
                ));
            }
        }
        if payload.query_class_count as usize != query.classes.len()
            || offline_evidence.payload.query_catalog_sha256 != catalog_sha256
            || offline_evidence.payload.query_catalog_revision != query.catalog_revision
            || offline_evidence.payload.query_class_ids_sha256 != class_ids_sha256
            || offline_evidence.payload.query_class_count as usize != query.classes.len()
            || payload.model_sample_rate_hz != manifest.runtime.sample_rate_hz
            || payload.frame_samples != manifest.latency.frame_samples
            || payload.algorithmic_latency_samples != manifest.latency.algorithmic_latency_samples
            || payload.flush_samples != manifest.latency.flush_samples
        {
            return Err(
                "causal target-sound evidence catalog or stream geometry does not match".into(),
            );
        }
        let mut options = BackendOptions::default().with_runtime_model_package(package.clone());
        options.deterministic = true;
        options.accelerator = requested;
        let accelerator = crate::select_accelerator_for_options(Backend::Onnx, &options)?;
        if !package.supports_accelerator(accelerator.effective()) {
            return Err(format!(
                "causal target-sound package does not permit the {} accelerator",
                accelerator.effective().name()
            ));
        }
        let model =
            CausalTargetSoundModel::load_runtime_package(&package, accelerator.effective())?;
        if model.query_classes() != query.classes.len()
            || model.channels() as u32 != payload.model_channels
        {
            return Err(
                "causal target-sound graph catalog or channel count differs from evidence".into(),
            );
        }
        let selected_index = query.selected_index()?;
        let selected_class = query.selected_class()?;
        let query_identity = CausalTargetSoundQueryIdentity {
            query_sha256,
            catalog_sha256,
            catalog_revision: query.catalog_revision.clone(),
            class_ids_sha256,
            class_count: query.classes.len() as u32,
            class_id: selected_class.id.clone(),
            class_index: selected_index as u32,
            canonical_label: selected_class.canonical_label.clone(),
            encoding: "one-hot-v1".into(),
            open_text_accepted: false,
        };
        let evidence = CausalTargetSoundEvidenceIdentity {
            offline_signing_key_id: offline_evidence.signature.key_id.clone(),
            causal_signing_key_id: causal_evidence.signature.key_id.clone(),
            offline_model_package_sha256: payload.offline_model_package_sha256.clone(),
            offline_evaluation_result_sha256: payload.offline_evaluation_result_sha256.clone(),
            causal_evaluation_result_sha256: payload.causal_evaluation_result_sha256.clone(),
            state_reset_flush_result_sha256: payload.state_reset_flush_result_sha256.clone(),
            snapshot_roundtrip_result_sha256: payload.snapshot_roundtrip_result_sha256.clone(),
            recombination_result_sha256: payload.recombination_result_sha256.clone(),
            latency_result_sha256: payload.latency_result_sha256.clone(),
            realtime_callback_result_sha256: payload.realtime_callback_result_sha256.clone(),
            transition_result_sha256: payload.transition_result_sha256.clone(),
            device_measurements: payload.device_measurements.len() as u32,
            strata: payload.strata.len() as u32,
            accepted: true,
        };
        Ok(Self {
            package,
            model,
            accelerator,
            config: config.clone(),
            configuration_sha256,
            query: query_identity,
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
    pub const fn channels(&self) -> usize {
        self.model.channels()
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
            .expect("causal target-sound package uses v2 precision profiles");
        Ok(profile
            .resources
            .max_session_memory_bytes
            .saturating_add(profile.resources.max_worker_memory_bytes))
    }

    pub fn start(&self) -> Result<CausalTargetSoundStream, String> {
        let runtime = self.model.start(self.query.class_index as usize)?;
        Ok(CausalTargetSoundStream {
            runtime,
            config: self.config.clone(),
            model_package_sha256: self.package.package_sha256().into(),
            configuration_sha256: self.configuration_sha256.clone(),
            query_sha256: self.query.query_sha256.clone(),
            query_catalog_sha256: self.query.catalog_sha256.clone(),
            selected_class_id: self.query.class_id.clone(),
            channels: self.model.channels(),
            frame_samples: self.model.frame_samples(),
            flush_samples: self.model.flush_samples(),
            generation: 1,
            next_frame: 0,
            present_streak: 0,
            finished: false,
        })
    }
}

pub struct CausalTargetSoundStream {
    runtime: CausalTargetSoundRuntime,
    config: CausalTargetSoundConfig,
    model_package_sha256: String,
    configuration_sha256: String,
    query_sha256: String,
    query_catalog_sha256: String,
    selected_class_id: String,
    channels: usize,
    frame_samples: usize,
    flush_samples: usize,
    generation: u64,
    next_frame: u64,
    present_streak: u32,
    finished: bool,
}

impl std::fmt::Debug for CausalTargetSoundStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CausalTargetSoundStream")
            .field("channels", &self.channels)
            .field("frame_samples", &self.frame_samples)
            .field("generation", &self.generation)
            .field("next_frame", &self.next_frame)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl CausalTargetSoundStream {
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    #[must_use]
    pub const fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn process(&mut self, mixture: &[Vec<f32>]) -> Result<CausalTargetSoundBlock, String> {
        if self.finished {
            return Err("causal target-sound stream has already been finished".into());
        }
        let start_frame = self.next_frame;
        let inference = self.runtime.process(mixture)?;
        self.next_frame = self
            .next_frame
            .checked_add(self.frame_samples as u64)
            .ok_or_else(|| "causal target-sound frame clock overflow".to_string())?;
        self.classify_block(
            inference.target,
            inference.residual,
            inference.presence_probabilities,
            mixture,
            start_frame,
            false,
        )
    }

    pub fn reset(&mut self) -> Result<u64, String> {
        let generation = self
            .generation
            .checked_add(1)
            .filter(|generation| *generation <= JSON_SAFE_INTEGER)
            .ok_or_else(|| "causal target-sound stream generation is exhausted".to_string())?;
        self.reset_generation_at(generation, 0)?;
        Ok(self.generation)
    }

    fn reset_generation_at(&mut self, generation: u64, next_frame: u64) -> Result<(), String> {
        self.runtime.reset()?;
        self.generation = generation.max(1);
        self.next_frame = next_frame;
        self.present_streak = 0;
        self.finished = false;
        Ok(())
    }

    pub fn snapshot(&self) -> Result<CausalTargetSoundSnapshot, String> {
        if self.finished {
            return Err("cannot snapshot a finished causal target-sound stream".into());
        }
        let backend = self.runtime.snapshot()?;
        let snapshot = CausalTargetSoundSnapshot {
            schema: CAUSAL_TARGET_SOUND_SNAPSHOT_SCHEMA.into(),
            schema_version: CAUSAL_TARGET_SOUND_SCHEMA_VERSION,
            model_package_sha256: self.model_package_sha256.clone(),
            configuration_sha256: self.configuration_sha256.clone(),
            query_sha256: self.query_sha256.clone(),
            query_catalog_sha256: self.query_catalog_sha256.clone(),
            selected_class_id: self.selected_class_id.clone(),
            snapshot_generation: self.generation,
            next_frame: self.next_frame,
            present_streak: self.present_streak,
            states: backend.states.into_iter().map(public_state).collect(),
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn restore(&mut self, snapshot: &CausalTargetSoundSnapshot) -> Result<u64, String> {
        snapshot.validate()?;
        for (label, observed, expected) in [
            (
                "model package",
                snapshot.model_package_sha256.as_str(),
                self.model_package_sha256.as_str(),
            ),
            (
                "configuration",
                snapshot.configuration_sha256.as_str(),
                self.configuration_sha256.as_str(),
            ),
            (
                "query",
                snapshot.query_sha256.as_str(),
                self.query_sha256.as_str(),
            ),
            (
                "query catalog",
                snapshot.query_catalog_sha256.as_str(),
                self.query_catalog_sha256.as_str(),
            ),
            (
                "selected class",
                snapshot.selected_class_id.as_str(),
                self.selected_class_id.as_str(),
            ),
        ] {
            if observed != expected {
                return Err(format!(
                    "causal target-sound snapshot {label} does not match the active stream"
                ));
            }
        }
        let restored_generation = self
            .generation
            .max(snapshot.snapshot_generation)
            .checked_add(1)
            .filter(|generation| *generation <= JSON_SAFE_INTEGER)
            .ok_or_else(|| "causal target-sound snapshot generation is exhausted".to_string())?;
        let backend = CausalTargetSoundBackendSnapshot {
            states: snapshot.states.iter().cloned().map(backend_state).collect(),
        };
        self.runtime.restore(&backend)?;
        self.generation = restored_generation;
        self.next_frame = snapshot.next_frame;
        self.present_streak = snapshot.present_streak;
        self.finished = false;
        Ok(self.generation)
    }

    pub fn finish(&mut self) -> Result<Vec<CausalTargetSoundBlock>, String> {
        if self.finished {
            return Err("causal target-sound stream has already been finished".into());
        }
        self.finished = true;
        let blocks = self.flush_samples.div_ceil(self.frame_samples);
        let mut output = Vec::new();
        output
            .try_reserve_exact(blocks)
            .map_err(|_| "unable to reserve causal target-sound flush blocks".to_string())?;
        let zeros = vec![vec![0.0_f32; self.frame_samples]; self.channels];
        let mut remaining = self.flush_samples;
        for _ in 0..blocks {
            let start_frame = self.next_frame;
            let inference = self.runtime.process(&zeros)?;
            self.next_frame = self
                .next_frame
                .checked_add(self.frame_samples as u64)
                .ok_or_else(|| "causal target-sound flush clock overflow".to_string())?;
            let valid_frames = remaining.min(self.frame_samples);
            remaining -= valid_frames;
            let mut block = self.classify_block(
                inference.target,
                inference.residual,
                inference.presence_probabilities,
                &zeros,
                start_frame,
                true,
            )?;
            block.valid_frames = valid_frames;
            output.push(block);
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn classify_block(
        &mut self,
        target: Vec<Vec<f32>>,
        model_residual: Vec<Vec<f32>>,
        probabilities: [f32; 3],
        mixture: &[Vec<f32>],
        start_frame: u64,
        flush: bool,
    ) -> Result<CausalTargetSoundBlock, String> {
        let presence = classify_presence(probabilities, &self.config);
        if presence == TargetSoundPresence::Present {
            self.present_streak = self.present_streak.saturating_add(1);
        } else {
            self.present_streak = 0;
        }
        let mut exact_residual = vec![vec![0.0_f32; self.frame_samples]; self.channels];
        let mut model_error = 0.0_f64;
        let mut publication_error = 0.0_f64;
        for channel in 0..self.channels {
            for frame in 0..self.frame_samples {
                let input = mixture[channel][frame];
                let candidate = target[channel][frame];
                model_error = model_error.max(
                    (f64::from(input)
                        - f64::from(candidate)
                        - f64::from(model_residual[channel][frame]))
                    .abs(),
                );
                exact_residual[channel][frame] = input - candidate;
                publication_error = publication_error.max(
                    (f64::from(candidate) + f64::from(exact_residual[channel][frame])
                        - f64::from(input))
                    .abs(),
                );
            }
        }
        let target_peak = peak(&target);
        let residual_peak = peak(&exact_residual);
        let target_gain = rms_dbfs(&target) - rms_dbfs(mixture);
        let residual_gain = rms_dbfs(&exact_residual) - rms_dbfs(mixture);
        let spatial_safe = spatial_safe(
            &model_residual,
            &exact_residual,
            self.config.maximum_stereo_correlation_delta,
            self.config.maximum_mid_side_energy_ratio_delta_db,
        )?;
        let signal_safe = model_error <= self.config.maximum_model_recombination_error
            && publication_error <= self.config.maximum_publication_recombination_error
            && target_peak <= self.config.maximum_target_peak
            && residual_peak <= self.config.maximum_residual_peak
            && target_gain <= self.config.maximum_energy_gain_db
            && residual_gain <= self.config.maximum_energy_gain_db
            && spatial_safe;
        let decision = if flush && presence != TargetSoundPresence::Present {
            CausalTargetSoundBlockDecision::FallbackFlush
        } else {
            match presence {
                TargetSoundPresence::Absent => CausalTargetSoundBlockDecision::FallbackAbsent,
                TargetSoundPresence::Uncertain => CausalTargetSoundBlockDecision::FallbackUncertain,
                TargetSoundPresence::Present if !signal_safe => {
                    CausalTargetSoundBlockDecision::FallbackSafetyGate
                }
                TargetSoundPresence::Present
                    if self.present_streak < self.config.present_hold_blocks =>
                {
                    CausalTargetSoundBlockDecision::FallbackPresentWarmup
                }
                TargetSoundPresence::Present => CausalTargetSoundBlockDecision::PublishedPresent,
            }
        };
        let accepted = decision == CausalTargetSoundBlockDecision::PublishedPresent;
        let (target, residual) = if accepted {
            (flatten(&target), flatten(&exact_residual))
        } else {
            (
                vec![0.0_f32; self.channels * self.frame_samples],
                flatten(mixture),
            )
        };
        let published_error =
            maximum_recombination_error_flat(&target, &residual, &flatten(mixture))?;
        if published_error > self.config.maximum_publication_recombination_error {
            return Err("causal target-sound conservative publication failed recombination".into());
        }
        Ok(CausalTargetSoundBlock {
            generation: self.generation,
            start_frame,
            valid_frames: self.frame_samples,
            channels: self.channels,
            target,
            residual,
            presence,
            absent_probability: probabilities[0],
            uncertain_probability: probabilities[1],
            present_probability: probabilities[2],
            model_recombination_maximum_absolute_error: model_error,
            publication_recombination_maximum_absolute_error: published_error,
            decision,
            candidate_accepted: accepted,
        })
    }
}

impl CausalTargetSoundSession {
    /// Render through the causal state machine, remove the signed latency, and
    /// reconstruct the residual at the source clock so conservation survives
    /// both chunk and resampler boundaries.
    pub fn render(&self, input: &Audio) -> Result<CausalTargetSoundRenderResult, String> {
        validate_audio(input, self.model.channels())?;
        let model_rate = self.model.sample_rate_hz();
        let model_input =
            crate::resample::resample_channels(&input.channels, input.sample_rate, model_rate)?;
        let model_frames = model_input
            .first()
            .map(Vec::len)
            .ok_or("causal target-sound resampling produced no channels")?;
        if model_frames == 0
            || model_input.len() != self.model.channels()
            || model_input
                .iter()
                .any(|channel| channel.len() != model_frames)
        {
            return Err("causal target-sound model-rate geometry is invalid".into());
        }
        let frame_samples = self.model.frame_samples();
        let input_blocks = model_frames.div_ceil(frame_samples);
        let rendered_capacity = input_blocks
            .checked_mul(frame_samples)
            .and_then(|value| value.checked_add(self.model.flush_samples()))
            .ok_or_else(|| "causal target-sound render size overflow".to_string())?;
        let mut rendered_target = vec![Vec::new(); self.model.channels()];
        for channel in &mut rendered_target {
            channel
                .try_reserve_exact(rendered_capacity)
                .map_err(|_| "unable to reserve causal target-sound render".to_string())?;
        }
        let mut rendered_publication_mask = Vec::new();
        rendered_publication_mask
            .try_reserve_exact(rendered_capacity)
            .map_err(|_| "unable to reserve causal target-sound publication mask".to_string())?;
        let mut stream = self.start()?;
        let mut frame = vec![vec![0.0_f32; frame_samples]; self.model.channels()];
        let mut decision_counts = CausalTargetSoundDecisionCounts::default();
        let mut previous_presence = None;
        let mut presence_transitions = 0_u64;
        let mut maximum_model_recombination_error = 0.0_f64;
        let mut maximum_publication_recombination_error = 0.0_f64;
        for block_index in 0..input_blocks {
            let start = block_index * frame_samples;
            let available = model_frames.saturating_sub(start).min(frame_samples);
            for channel in 0..self.model.channels() {
                frame[channel].fill(0.0);
                for offset in 0..available {
                    frame[channel][offset] =
                        model_input[channel][start + offset].clamp(-1.0, 1.0) as f32;
                }
            }
            let block = stream.process(&frame)?;
            observe_render_block(
                &block,
                &mut decision_counts,
                &mut previous_presence,
                &mut presence_transitions,
                &mut maximum_model_recombination_error,
                &mut maximum_publication_recombination_error,
            );
            append_planar_block(
                &mut rendered_target,
                &block.target,
                self.model.channels(),
                frame_samples,
                frame_samples,
            )?;
            rendered_publication_mask.extend(std::iter::repeat_n(
                if block.candidate_accepted { 1.0 } else { 0.0 },
                frame_samples,
            ));
        }
        let flush = stream.finish()?;
        for block in &flush {
            observe_render_block(
                block,
                &mut decision_counts,
                &mut previous_presence,
                &mut presence_transitions,
                &mut maximum_model_recombination_error,
                &mut maximum_publication_recombination_error,
            );
            append_planar_block(
                &mut rendered_target,
                &block.target,
                self.model.channels(),
                frame_samples,
                block.valid_frames,
            )?;
            rendered_publication_mask.extend(std::iter::repeat_n(
                if block.candidate_accepted { 1.0 } else { 0.0 },
                block.valid_frames,
            ));
        }
        let latency = self.model.algorithmic_latency_samples();
        let mut aligned = vec![vec![0.0_f64; model_frames]; self.model.channels()];
        for (channel, aligned_channel) in aligned.iter_mut().enumerate() {
            for (frame, output) in aligned_channel.iter_mut().enumerate() {
                *output = f64::from(
                    rendered_target[channel]
                        .get(latency + frame)
                        .copied()
                        .unwrap_or(0.0),
                );
            }
        }
        let mut aligned_publication_mask = vec![0.0_f64; model_frames];
        for (frame, mask) in aligned_publication_mask.iter_mut().enumerate() {
            *mask = rendered_publication_mask
                .get(latency + frame)
                .copied()
                .unwrap_or(0.0);
        }
        let mut target_source =
            crate::resample::resample_channels(&aligned, model_rate, input.sample_rate)?;
        if target_source.len() != input.channels() {
            return Err("causal target-sound source-clock channel count changed".into());
        }
        for channel in &mut target_source {
            channel.resize(input.frames(), 0.0);
            channel.truncate(input.frames());
        }
        let mut source_publication_mask =
            crate::resample::resample(&aligned_publication_mask, model_rate, input.sample_rate)?;
        source_publication_mask.resize(input.frames(), 0.0);
        source_publication_mask.truncate(input.frames());
        let mut source_clock_withheld_frames = 0_usize;
        for frame in 0..input.frames() {
            let model_interval_start =
                (frame as u128).saturating_mul(model_rate as u128) / input.sample_rate as u128;
            let model_interval_end = ((frame + 1) as u128)
                .saturating_mul(model_rate as u128)
                .saturating_add(input.sample_rate as u128 - 1)
                / input.sample_rate as u128;
            let direct_interval_accepted = (model_interval_start
                ..model_interval_end.max(model_interval_start.saturating_add(1)))
                .all(|model_frame| {
                    usize::try_from(model_frame)
                        .ok()
                        .and_then(|index| aligned_publication_mask.get(index))
                        .is_some_and(|mask| *mask == 1.0)
                });
            let filtered_mask_accepted = source_publication_mask
                .get(frame)
                .is_some_and(|mask| mask.is_finite() && *mask >= 0.999_999);
            if !direct_interval_accepted || !filtered_mask_accepted {
                for channel in &mut target_source {
                    channel[frame] = 0.0;
                }
                source_clock_withheld_frames = source_clock_withheld_frames.saturating_add(1);
            }
        }
        let mut residual_source = vec![vec![0.0_f64; input.frames()]; input.channels()];
        for channel in 0..input.channels() {
            for frame in 0..input.frames() {
                residual_source[channel][frame] =
                    input.channels[channel][frame] - target_source[channel][frame];
            }
        }
        let source_publication_error =
            maximum_recombination_error_f64(&target_source, &residual_source, &input.channels)?;
        maximum_publication_recombination_error =
            maximum_publication_recombination_error.max(source_publication_error);
        let source_safe = finite_normalized(&target_source)
            && finite_normalized(&residual_source)
            && peak_f64(&target_source) <= self.config.maximum_target_peak
            && peak_f64(&residual_source) <= self.config.maximum_residual_peak
            && rms_dbfs_f64(&target_source) - rms_dbfs_f64(&input.channels)
                <= self.config.maximum_energy_gain_db
            && rms_dbfs_f64(&residual_source) - rms_dbfs_f64(&input.channels)
                <= self.config.maximum_energy_gain_db
            && source_publication_error <= self.config.maximum_publication_recombination_error;
        let source_clock_conservative_fallback = !source_safe;
        if source_clock_conservative_fallback {
            for channel in &mut target_source {
                channel.fill(0.0);
            }
            residual_source.clone_from(&input.channels);
            maximum_publication_recombination_error = 0.0;
        }
        let target = output_audio(input, target_source);
        let residual = output_audio(input, residual_source);
        let output = match self.config.mode {
            TargetSoundMode::Preserve => target.clone(),
            TargetSoundMode::Remove => residual.clone(),
        };
        let manifest = self
            .package
            .manifest_v2()
            .expect("causal target-sound session requires v2");
        let profile = self
            .package
            .precision_profile_for(self.accelerator.effective())?
            .expect("causal target-sound session selects one precision profile");
        let mut warnings = Vec::new();
        if decision_counts.published_present_blocks == 0 {
            warnings.push(
                "no block passed presence, hold, and signal gates; target is silence and residual is the untouched input"
                    .into(),
            );
        }
        if source_clock_conservative_fallback {
            warnings.push(
                "source-clock resampling failed a publication gate; the entire render uses the conservative decomposition"
                    .into(),
            );
        }
        let report = CausalTargetSoundRenderReport {
            schema: CAUSAL_TARGET_SOUND_REPORT_SCHEMA.into(),
            schema_version: CAUSAL_TARGET_SOUND_SCHEMA_VERSION,
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            network_accessed: false,
            deterministic: true,
            mode: self.config.mode,
            configuration_sha256: self.configuration_sha256.clone(),
            query: self.query.clone(),
            model: CausalTargetSoundModelIdentity {
                package_sha256: self.package.package_sha256().into(),
                public_key_sha256: self.package.public_key_sha256().into(),
                package_id: manifest.package_id.clone(),
                package_revision: manifest.package_revision.clone(),
                precision_profile: profile.id.clone(),
                package_license_spdx: manifest.license.spdx.clone(),
                source_revision: manifest.provenance.source_revision.clone(),
                source_sha256: manifest.provenance.source_sha256.clone(),
                source_license_spdx: manifest.provenance.source_license_spdx.clone(),
                checkpoint_sha256: manifest.provenance.checkpoint_sha256.clone(),
                checkpoint_license_spdx: manifest.provenance.checkpoint_license_spdx.clone(),
                accelerator: self.accelerator.effective().name().into(),
            },
            promotion_evidence: self.evidence.clone(),
            source_sample_rate: input.sample_rate,
            source_channels: input.channels(),
            source_frames: input.frames(),
            model_sample_rate: model_rate,
            model_channels: self.model.channels(),
            frame_samples,
            algorithmic_latency_samples: latency,
            flush_samples: self.model.flush_samples(),
            input_blocks: input_blocks as u64,
            flush_blocks: flush.len() as u64,
            decision_counts,
            presence_transitions,
            source_clock_withheld_frames,
            source_clock_conservative_fallback,
            target_published: true,
            residual_published: true,
            output_published: true,
            input_pcm_sha256: pcm_digest(input, INPUT_PCM_DIGEST_DOMAIN),
            target_pcm_sha256: pcm_digest(&target, TARGET_PCM_DIGEST_DOMAIN),
            residual_pcm_sha256: pcm_digest(&residual, RESIDUAL_PCM_DIGEST_DOMAIN),
            output_pcm_sha256: pcm_digest(&output, OUTPUT_PCM_DIGEST_DOMAIN),
            maximum_model_recombination_error,
            maximum_publication_recombination_error,
            partial_semantic_removal_fallbacks: 0,
            path_fields_recorded: 0,
            limitations: causal_limitations(),
            warnings,
        };
        report.validate()?;
        Ok(CausalTargetSoundRenderResult {
            target,
            residual,
            output,
            report,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn observe_render_block(
    block: &CausalTargetSoundBlock,
    decision_counts: &mut CausalTargetSoundDecisionCounts,
    previous_presence: &mut Option<TargetSoundPresence>,
    presence_transitions: &mut u64,
    maximum_model_recombination_error: &mut f64,
    maximum_publication_recombination_error: &mut f64,
) {
    decision_counts.observe(block.decision);
    if previous_presence.is_some_and(|previous| previous != block.presence) {
        *presence_transitions = presence_transitions.saturating_add(1);
    }
    *previous_presence = Some(block.presence);
    *maximum_model_recombination_error =
        maximum_model_recombination_error.max(block.model_recombination_maximum_absolute_error);
    *maximum_publication_recombination_error = maximum_publication_recombination_error
        .max(block.publication_recombination_maximum_absolute_error);
}

fn append_planar_block(
    destination: &mut [Vec<f32>],
    flat: &[f32],
    channels: usize,
    frame_samples: usize,
    valid_frames: usize,
) -> Result<(), String> {
    if destination.len() != channels
        || flat.len() != channels.saturating_mul(frame_samples)
        || valid_frames > frame_samples
    {
        return Err("causal target-sound rendered block geometry is invalid".into());
    }
    for (channel, destination_channel) in destination.iter_mut().enumerate() {
        let start = channel * frame_samples;
        destination_channel.extend_from_slice(&flat[start..start + valid_frames]);
    }
    Ok(())
}

/// Write the only permitted overload/dropout fallback into caller-owned planar
/// buffers. This function allocates no memory and performs no synchronization.
pub fn write_causal_target_sound_conservative_fallback(
    input: &[f32],
    target: &mut [f32],
    residual: &mut [f32],
) -> Result<(), String> {
    if input.len() != target.len() || input.len() != residual.len() {
        return Err("causal target-sound fallback buffers have different lengths".into());
    }
    if input
        .iter()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err("causal target-sound fallback input is not finite normalized PCM".into());
    }
    target.fill(0.0);
    residual.copy_from_slice(input);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalTargetSoundRealtimeToken {
    pub generation: u64,
    pub start_frame: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CausalTargetSoundRealtimeSubmitError {
    WrongFrameSize,
    NonFiniteInput,
    OutOfRangeInput,
    PoolExhausted,
    QueueFull,
    Stopped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CausalTargetSoundRealtimeReceiveError {
    WrongFrameSize,
    WrongGeneration,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausalTargetSoundRealtimeResult {
    pub token: CausalTargetSoundRealtimeToken,
    pub valid: bool,
    pub presence: TargetSoundPresence,
    pub absent_probability: f32,
    pub uncertain_probability: f32,
    pub present_probability: f32,
    pub model_recombination_maximum_absolute_error: f64,
    pub publication_recombination_maximum_absolute_error: f64,
    pub decision: CausalTargetSoundBlockDecision,
    pub candidate_accepted: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CausalTargetSoundRealtimeMetrics {
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
    fn snapshot(&self) -> CausalTargetSoundRealtimeMetrics {
        CausalTargetSoundRealtimeMetrics {
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
    token: CausalTargetSoundRealtimeToken,
    input: Box<[f32]>,
    target: Box<[f32]>,
    residual: Box<[f32]>,
    valid: bool,
    presence: TargetSoundPresence,
    probabilities: [f32; 3],
    model_recombination_error: f64,
    publication_recombination_error: f64,
    decision: CausalTargetSoundBlockDecision,
    candidate_accepted: bool,
}

impl RealtimeBlock {
    fn new(samples_per_block: usize) -> Result<Self, String> {
        let allocate = || -> Result<Box<[f32]>, String> {
            let mut samples = Vec::new();
            samples
                .try_reserve_exact(samples_per_block)
                .map_err(|_| "unable to reserve causal target-sound realtime block".to_string())?;
            samples.resize(samples_per_block, 0.0);
            Ok(samples.into_boxed_slice())
        };
        Ok(Self {
            token: CausalTargetSoundRealtimeToken {
                generation: 1,
                start_frame: 0,
            },
            input: allocate()?,
            target: allocate()?,
            residual: allocate()?,
            valid: false,
            presence: TargetSoundPresence::Uncertain,
            probabilities: [0.0, 1.0, 0.0],
            model_recombination_error: 0.0,
            publication_recombination_error: 0.0,
            decision: CausalTargetSoundBlockDecision::FallbackSafetyGate,
            candidate_accepted: false,
        })
    }
}

trait CausalRealtimeProcessor: Send {
    fn channels(&self) -> usize;
    fn frame_samples(&self) -> usize;
    fn generation(&self) -> u64;
    fn process_flat(&mut self, mixture: &[f32]) -> Result<CausalTargetSoundBlock, String>;
    fn reset_generation_at(&mut self, generation: u64, next_frame: u64) -> Result<(), String>;
}

impl CausalRealtimeProcessor for CausalTargetSoundStream {
    fn channels(&self) -> usize {
        self.channels()
    }

    fn frame_samples(&self) -> usize {
        self.frame_samples()
    }

    fn generation(&self) -> u64 {
        self.generation()
    }

    fn process_flat(&mut self, mixture: &[f32]) -> Result<CausalTargetSoundBlock, String> {
        if mixture.len() != self.channels.saturating_mul(self.frame_samples) {
            return Err("causal target-sound realtime worker received the wrong frame size".into());
        }
        let planar = mixture
            .chunks_exact(self.frame_samples)
            .map(|channel| channel.to_vec())
            .collect::<Vec<_>>();
        self.process(&planar)
    }

    fn reset_generation_at(&mut self, generation: u64, next_frame: u64) -> Result<(), String> {
        self.reset_generation_at(generation, next_frame)
    }
}

/// Fixed-pool off-callback inference bridge.
///
/// Construction/destruction are control-thread operations. `try_submit`,
/// `try_receive_due`, `reset`, and the conservative fallback allocate no
/// memory, acquire no mutex, perform no I/O, and never wait for inference.
pub struct CausalTargetSoundRealtimeScheduler {
    channels: usize,
    frame_samples: usize,
    samples_per_block: usize,
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

impl std::fmt::Debug for CausalTargetSoundRealtimeScheduler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CausalTargetSoundRealtimeScheduler")
            .field("channels", &self.channels)
            .field("frame_samples", &self.frame_samples)
            .field("generation", &self.generation)
            .field("next_submit_frame", &self.next_submit_frame)
            .field("metrics", &self.metrics())
            .finish_non_exhaustive()
    }
}

impl CausalTargetSoundRealtimeScheduler {
    pub fn new(stream: CausalTargetSoundStream) -> Result<Self, String> {
        Self::new_with_processor(Box::new(stream))
    }

    fn new_with_processor(processor: Box<dyn CausalRealtimeProcessor>) -> Result<Self, String> {
        let channels = processor.channels();
        let frame_samples = processor.frame_samples();
        let samples_per_block = channels
            .checked_mul(frame_samples)
            .ok_or_else(|| "causal target-sound realtime block size overflow".to_string())?;
        let generation = processor.generation();
        let input = Arc::new(ArrayQueue::new(REALTIME_QUEUE_BLOCKS));
        let output = Arc::new(ArrayQueue::new(REALTIME_QUEUE_BLOCKS));
        let free = Arc::new(ArrayQueue::new(REALTIME_POOL_BLOCKS));
        for _ in 0..REALTIME_POOL_BLOCKS {
            free.push(RealtimeBlock::new(samples_per_block)?)
                .map_err(|_| "causal target-sound realtime pool initialization failed")?;
        }
        let running = Arc::new(AtomicBool::new(true));
        let metrics = Arc::new(RealtimeMetricAtoms::default());
        let worker_input = Arc::clone(&input);
        let worker_output = Arc::clone(&output);
        let worker_free = Arc::clone(&free);
        let worker_running = Arc::clone(&running);
        let worker_metrics = Arc::clone(&metrics);
        let worker = thread::Builder::new()
            .name("denoize-causal-target-sound".into())
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
            .map_err(|error| format!("start causal target-sound worker: {error}"))?;
        Ok(Self {
            channels,
            frame_samples,
            samples_per_block,
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
    pub const fn channels(&self) -> usize {
        self.channels
    }

    #[must_use]
    pub const fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Submit one planar block. Queue/pool failure advances the absolute clock;
    /// the caller must immediately use the conservative fallback for that block.
    pub fn try_submit(
        &mut self,
        mixture: &[f32],
    ) -> Result<CausalTargetSoundRealtimeToken, CausalTargetSoundRealtimeSubmitError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(CausalTargetSoundRealtimeSubmitError::Stopped);
        }
        if mixture.len() != self.samples_per_block {
            return Err(CausalTargetSoundRealtimeSubmitError::WrongFrameSize);
        }
        if mixture.iter().any(|sample| !sample.is_finite()) {
            return Err(CausalTargetSoundRealtimeSubmitError::NonFiniteInput);
        }
        if mixture.iter().any(|sample| !(-1.0..=1.0).contains(sample)) {
            return Err(CausalTargetSoundRealtimeSubmitError::OutOfRangeInput);
        }
        let token = CausalTargetSoundRealtimeToken {
            generation: self.generation,
            start_frame: self.next_submit_frame,
        };
        self.next_submit_frame = self
            .next_submit_frame
            .wrapping_add(self.frame_samples as u64);
        let Some(mut block) = self.free.pop() else {
            self.metrics.overload_blocks.fetch_add(1, Ordering::Relaxed);
            return Err(CausalTargetSoundRealtimeSubmitError::PoolExhausted);
        };
        block.token = token;
        block.input.copy_from_slice(mixture);
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
                Err(CausalTargetSoundRealtimeSubmitError::QueueFull)
            }
        }
    }

    /// Receive exactly one due decomposition into caller-owned planar buffers.
    /// A missing result returns `None`; the caller must use the conservative
    /// fallback with the original input for that due block.
    pub fn try_receive_due(
        &mut self,
        token: CausalTargetSoundRealtimeToken,
        target: &mut [f32],
        residual: &mut [f32],
    ) -> Result<Option<CausalTargetSoundRealtimeResult>, CausalTargetSoundRealtimeReceiveError>
    {
        if target.len() != self.samples_per_block || residual.len() != self.samples_per_block {
            return Err(CausalTargetSoundRealtimeReceiveError::WrongFrameSize);
        }
        if token.generation != self.generation {
            return Err(CausalTargetSoundRealtimeReceiveError::WrongGeneration);
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
            target.copy_from_slice(&block.target);
            residual.copy_from_slice(&block.residual);
            if !block.valid {
                self.metrics.invalid_blocks.fetch_add(1, Ordering::Relaxed);
            }
            let result = CausalTargetSoundRealtimeResult {
                token: block.token,
                valid: block.valid,
                presence: block.presence,
                absent_probability: block.probabilities[0],
                uncertain_probability: block.probabilities[1],
                present_probability: block.probabilities[2],
                model_recombination_maximum_absolute_error: block.model_recombination_error,
                publication_recombination_maximum_absolute_error: block
                    .publication_recombination_error,
                decision: block.decision,
                candidate_accepted: block.candidate_accepted,
            };
            recycle_realtime_block(&self.free, block);
            return Ok(Some(result));
        }
    }

    /// Invalidate all queued results and begin a new zero-state generation.
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
    pub fn metrics(&self) -> CausalTargetSoundRealtimeMetrics {
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

impl Drop for CausalTargetSoundRealtimeScheduler {
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
        let reset_failed =
            if block.token.generation != generation || block.token.start_frame != next_frame {
                generation = block.token.generation;
                if processor
                    .reset_generation_at(generation, block.token.start_frame)
                    .is_err()
                {
                    metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
                    true
                } else {
                    false
                }
            } else {
                false
            };
        next_frame = block
            .token
            .start_frame
            .wrapping_add(processor.frame_samples() as u64);
        let inference = if reset_failed {
            Err("causal target-sound worker state reset failed".to_string())
        } else {
            processor.process_flat(&block.input)
        };
        match inference {
            Ok(result) => {
                block.target.copy_from_slice(&result.target);
                block.residual.copy_from_slice(&result.residual);
                block.valid = true;
                block.presence = result.presence;
                block.probabilities = [
                    result.absent_probability,
                    result.uncertain_probability,
                    result.present_probability,
                ];
                block.model_recombination_error = result.model_recombination_maximum_absolute_error;
                block.publication_recombination_error =
                    result.publication_recombination_maximum_absolute_error;
                block.decision = result.decision;
                block.candidate_accepted = result.candidate_accepted;
            }
            Err(_) => {
                block.target.fill(0.0);
                block.residual.copy_from_slice(&block.input);
                block.valid = false;
                block.presence = TargetSoundPresence::Uncertain;
                block.probabilities = [0.0, 1.0, 0.0];
                block.model_recombination_error = 0.0;
                block.publication_recombination_error = 0.0;
                block.decision = CausalTargetSoundBlockDecision::FallbackSafetyGate;
                block.candidate_accepted = false;
                metrics.worker_errors.fetch_add(1, Ordering::Relaxed);
                if processor
                    .reset_generation_at(generation, next_frame)
                    .is_err()
                {
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
        // Preserve callback-side no-deallocation even if the fixed-pool
        // ownership invariant is broken by a future change.
        std::mem::forget(block);
    }
}

fn validate_offline_matrix_binding(
    offline: &SignedTargetSoundPromotionEvidence,
    causal: &SignedCausalTargetSoundPromotionEvidence,
) -> Result<(), String> {
    if offline.payload.strata.len() != causal.payload.strata.len() {
        return Err(
            "causal target-sound evidence does not reproduce the offline stratum matrix".into(),
        );
    }
    for (offline_stratum, causal_stratum) in
        offline.payload.strata.iter().zip(&causal.payload.strata)
    {
        if causal_stratum.id != offline_stratum.id
            || causal_stratum.kind != offline_stratum.kind
            || causal_stratum.offline_cases != offline_stratum.cases
            || causal_stratum.metrics.len() != offline_stratum.metrics.len()
        {
            return Err(format!(
                "causal target-sound stratum {} does not reproduce offline kind, cases, and metrics",
                causal_stratum.id
            ));
        }
        for (offline_metric, causal_metric) in
            offline_stratum.metrics.iter().zip(&causal_stratum.metrics)
        {
            if causal_metric.metric != offline_metric.metric
                || causal_metric.operator != offline_metric.operator
                || causal_metric.offline_value != offline_metric.value
                || causal_metric.hard_limit != offline_metric.limit
            {
                return Err(format!(
                    "causal target-sound metric {} in {} does not reproduce the offline claim",
                    causal_metric.metric, causal_stratum.id
                ));
            }
        }
    }
    Ok(())
}

fn validate_metric(
    metric: &CausalTargetSoundMetricEvidence,
    policy: &MetricPolicy,
) -> Result<(), String> {
    validate_identifier("causal metric", &metric.metric)?;
    let hard_limit_is_strong_enough = match policy.operator {
        TargetSoundMetricOperator::GreaterOrEqual => metric.hard_limit >= policy.hard_limit,
        TargetSoundMetricOperator::LessOrEqual => metric.hard_limit <= policy.hard_limit,
    };
    if metric.metric != policy.name
        || metric.operator != policy.operator
        || !metric.offline_value.is_finite()
        || !metric.causal_value.is_finite()
        || !metric.hard_limit.is_finite()
        || !metric.maximum_regression.is_finite()
        || !(0.0..=policy.maximum_regression).contains(&metric.maximum_regression)
        || !hard_limit_is_strong_enough
    {
        return Err(format!(
            "causal target-sound metric {} has an invalid policy or value",
            metric.metric
        ));
    }
    let (offline_hard, causal_hard, non_inferior) = match policy.operator {
        TargetSoundMetricOperator::GreaterOrEqual => (
            metric.offline_value >= metric.hard_limit,
            metric.causal_value >= metric.hard_limit,
            metric.causal_value >= metric.offline_value - metric.maximum_regression,
        ),
        TargetSoundMetricOperator::LessOrEqual => (
            metric.offline_value <= metric.hard_limit,
            metric.causal_value <= metric.hard_limit,
            metric.causal_value <= metric.offline_value + metric.maximum_regression,
        ),
    };
    if metric.passed != (offline_hard && causal_hard && non_inferior) {
        return Err(format!(
            "causal target-sound metric {} has an inconsistent passed flag",
            metric.metric
        ));
    }
    Ok(())
}

fn public_state(value: BackendStateValue) -> CausalTargetSoundSnapshotState {
    match value {
        BackendStateValue::Float32 { shape, values } => {
            CausalTargetSoundSnapshotState::Float32 { shape, values }
        }
        BackendStateValue::Int64 { shape, values } => {
            CausalTargetSoundSnapshotState::Int64 { shape, values }
        }
    }
}

fn backend_state(value: CausalTargetSoundSnapshotState) -> BackendStateValue {
    match value {
        CausalTargetSoundSnapshotState::Float32 { shape, values } => {
            BackendStateValue::Float32 { shape, values }
        }
        CausalTargetSoundSnapshotState::Int64 { shape, values } => {
            BackendStateValue::Int64 { shape, values }
        }
    }
}

fn classify_presence(
    probabilities: [f32; 3],
    config: &CausalTargetSoundConfig,
) -> TargetSoundPresence {
    let absent = f64::from(probabilities[0]);
    let uncertain = f64::from(probabilities[1]);
    let present = f64::from(probabilities[2]);
    if present >= config.minimum_present_probability && present > absent && present > uncertain {
        TargetSoundPresence::Present
    } else if absent >= config.minimum_absent_probability && absent > present && absent > uncertain
    {
        TargetSoundPresence::Absent
    } else {
        TargetSoundPresence::Uncertain
    }
}

fn flatten(channels: &[Vec<f32>]) -> Vec<f32> {
    channels.iter().flatten().copied().collect()
}

fn peak(channels: &[Vec<f32>]) -> f64 {
    channels
        .iter()
        .flatten()
        .fold(0.0_f64, |peak, sample| peak.max(f64::from(sample.abs())))
}

fn peak_f64(channels: &[Vec<f64>]) -> f64 {
    channels
        .iter()
        .flatten()
        .fold(0.0_f64, |peak, sample| peak.max(sample.abs()))
}

fn rms_dbfs(channels: &[Vec<f32>]) -> f64 {
    let mut count = 0_usize;
    let mut energy = 0.0_f64;
    for sample in channels.iter().flatten() {
        count += 1;
        energy += f64::from(*sample) * f64::from(*sample);
    }
    amplitude_dbfs((energy / count.max(1) as f64).sqrt())
}

fn rms_dbfs_f64(channels: &[Vec<f64>]) -> f64 {
    let mut count = 0_usize;
    let mut energy = 0.0_f64;
    for sample in channels.iter().flatten() {
        count += 1;
        energy += sample * sample;
    }
    amplitude_dbfs((energy / count.max(1) as f64).sqrt())
}

fn amplitude_dbfs(value: f64) -> f64 {
    (20.0 * value.max(SILENCE_FLOOR).log10()).clamp(-240.0, 240.0)
}

fn spatial_safe(
    model_residual: &[Vec<f32>],
    exact_residual: &[Vec<f32>],
    maximum_correlation_delta: f64,
    maximum_mid_side_delta_db: f64,
) -> Result<bool, String> {
    if model_residual.len() == 1 && exact_residual.len() == 1 {
        return Ok(true);
    }
    if model_residual.len() != 2
        || exact_residual.len() != 2
        || model_residual[0].len() != model_residual[1].len()
        || exact_residual[0].len() != exact_residual[1].len()
        || model_residual[0].len() != exact_residual[0].len()
    {
        return Err("causal target-sound spatial geometry is invalid".into());
    }
    let correlation_delta = (normalized_correlation_f32(&model_residual[0], &model_residual[1])
        - normalized_correlation_f32(&exact_residual[0], &exact_residual[1]))
    .abs();
    let mid_side_delta =
        (mid_side_ratio_f32(model_residual)? - mid_side_ratio_f32(exact_residual)?).abs();
    Ok(correlation_delta <= maximum_correlation_delta
        && mid_side_delta <= maximum_mid_side_delta_db)
}

fn normalized_correlation_f32(left: &[f32], right: &[f32]) -> f64 {
    let (mut dot, mut left_energy, mut right_energy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        left_energy += left * left;
        right_energy += right * right;
    }
    if left_energy <= 1.0e-24 || right_energy <= 1.0e-24 {
        0.0
    } else {
        (dot / (left_energy.sqrt() * right_energy.sqrt())).clamp(-1.0, 1.0)
    }
}

fn mid_side_ratio_f32(channels: &[Vec<f32>]) -> Result<f64, String> {
    if channels.len() != 2 || channels[0].len() != channels[1].len() {
        return Err("causal target-sound mid/side geometry is invalid".into());
    }
    let (mut mid_energy, mut side_energy) = (0.0_f64, 0.0_f64);
    for (&left, &right) in channels[0].iter().zip(&channels[1]) {
        let left = f64::from(left);
        let right = f64::from(right);
        let mid = (left + right) * std::f64::consts::FRAC_1_SQRT_2;
        let side = (left - right) * std::f64::consts::FRAC_1_SQRT_2;
        mid_energy += mid * mid;
        side_energy += side * side;
    }
    Ok((10.0 * ((side_energy + 1.0e-24) / (mid_energy + 1.0e-24)).log10()).clamp(-240.0, 240.0))
}

fn maximum_recombination_error_flat(
    target: &[f32],
    residual: &[f32],
    input: &[f32],
) -> Result<f64, String> {
    if target.len() != residual.len() || target.len() != input.len() {
        return Err("causal target-sound publication geometry differs".into());
    }
    Ok(target.iter().zip(residual).zip(input).fold(
        0.0_f64,
        |maximum, ((target, residual), input)| {
            maximum.max((f64::from(*target) + f64::from(*residual) - f64::from(*input)).abs())
        },
    ))
}

fn maximum_recombination_error_f64(
    target: &[Vec<f64>],
    residual: &[Vec<f64>],
    input: &[Vec<f64>],
) -> Result<f64, String> {
    if target.len() != residual.len()
        || target.len() != input.len()
        || target
            .iter()
            .zip(residual)
            .zip(input)
            .any(|((target, residual), input)| {
                target.len() != residual.len() || target.len() != input.len()
            })
    {
        return Err("causal target-sound source publication geometry differs".into());
    }
    Ok(target
        .iter()
        .zip(residual)
        .zip(input)
        .flat_map(|((target, residual), input)| target.iter().zip(residual).zip(input))
        .fold(0.0_f64, |maximum, ((target, residual), input)| {
            maximum.max((target + residual - input).abs())
        }))
}

fn finite_normalized(channels: &[Vec<f64>]) -> bool {
    channels
        .iter()
        .flatten()
        .all(|sample| sample.is_finite() && (-1.0..=1.0).contains(sample))
}

fn validate_audio(audio: &Audio, channels: usize) -> Result<(), String> {
    if !(8_000..=192_000).contains(&audio.sample_rate)
        || audio.channels() != channels
        || audio.frames() == 0
        || audio.frames() as u64
            > u64::from(audio.sample_rate).saturating_mul(MAX_TARGET_SOUND_AUDIO_SECONDS)
        || audio
            .channels
            .iter()
            .any(|channel| channel.len() != audio.frames())
        || !finite_normalized(&audio.channels)
    {
        return Err(
            "causal target-sound input violates its bounded normalized mono/stereo contract".into(),
        );
    }
    Ok(())
}

fn output_audio(input: &Audio, channels: Vec<Vec<f64>>) -> Audio {
    Audio {
        sample_rate: input.sample_rate,
        channels,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
        channel_mask: input.channel_mask,
    }
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
        "the runtime accepts only an authenticated finite catalog and never sends natural language to the graph".into(),
        "the calibrated presence head is not independent acoustic-event ground truth".into(),
        "target leakage, protected-foreground quality, and spatial accuracy are promotion-time measurements".into(),
        "a valid evidence signature authenticates an evaluator claim but cannot prove recordings, labels, licenses, device measurements, or benchmark independence".into(),
        "accepted blocks publish a model target and a derived exact residual; conservative blocks publish target silence and untouched input".into(),
        "callback safety requires the fixed-pool scheduler API and the host must use the provided conservative fallback on submit or receive failure".into(),
        "snapshots contain model state and must be protected as potentially sensitive application data".into(),
        "no model, checkpoint, catalog, or upstream dataset is bundled; each artifact and license remains separately audited".into(),
    ]
}

fn digest_json<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
    label: &str,
) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| format!("serialize {label}: {error}"))?;
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_range(label: &str, value: f64, minimum: f64, maximum: f64) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        Err(format!(
            "causal target-sound {label} must be finite and in {minimum}..={maximum}"
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
            "causal target-sound {label} must be lowercase SHA-256"
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
            "causal target-sound {label} must use 1..=256 lowercase ASCII identifier characters"
        ));
    }
    Ok(())
}

fn validate_text(label: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty()
        || value.len() > maximum
        || value.chars().any(|character| character.is_control())
    {
        Err(format!(
            "causal target-sound {label} must contain 1..={maximum} non-control UTF-8 bytes"
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AcceleratorRuntime;

    #[test]
    fn evidence_requires_named_devices_and_no_partial_removal() {
        let payload = accepted_payload();
        payload.validate().unwrap();

        let mut invalid = payload.clone();
        invalid.transitions.partial_semantic_removal_publications = 1;
        invalid.accepted = false;
        invalid.validate().unwrap();
        invalid.accepted = true;
        assert!(invalid.validate().unwrap_err().contains("accepted flag"));

        let mut invalid = payload;
        invalid.device_measurements.pop();
        assert!(invalid.validate().unwrap_err().contains("3..=64"));
    }

    #[test]
    fn stream_fallback_snapshot_restore_and_acceptance_conserve_input() {
        let (_directory, package) = crate::backend::causal_target_sound::tests::fixture_package();
        let model = CausalTargetSoundModel::load_runtime_package(&package, AcceleratorRuntime::Cpu)
            .unwrap();
        let mut stream = fixture_stream(&package, &model, CausalTargetSoundConfig::default());
        let input = vec![vec![-0.5, 0.0, 0.25, 0.75]];
        let first = stream.process(&input).unwrap();
        assert_eq!(
            first.decision,
            CausalTargetSoundBlockDecision::FallbackPresentWarmup
        );
        assert_eq!(first.target, vec![0.0; 4]);
        assert_eq!(first.residual, input[0]);
        let snapshot = stream.snapshot().unwrap();
        let encoded = snapshot.to_pretty_json().unwrap();
        let decoded = CausalTargetSoundSnapshot::from_json(encoded.as_bytes()).unwrap();
        let mut invalid_generation = decoded.clone();
        invalid_generation.snapshot_generation = JSON_SAFE_INTEGER + 1;
        assert!(invalid_generation
            .validate()
            .unwrap_err()
            .contains("metadata"));
        let mut exhausted_generation = decoded.clone();
        exhausted_generation.snapshot_generation = JSON_SAFE_INTEGER;
        assert!(stream
            .restore(&exhausted_generation)
            .unwrap_err()
            .contains("generation is exhausted"));
        stream.process(&input).unwrap();
        let restored_generation = stream.restore(&decoded).unwrap();
        assert!(restored_generation > decoded.snapshot_generation);
        let second = stream.process(&input).unwrap();
        assert_eq!(
            second.decision,
            CausalTargetSoundBlockDecision::FallbackPresentWarmup
        );
        let third = stream.process(&input).unwrap();
        assert_eq!(
            third.decision,
            CausalTargetSoundBlockDecision::PublishedPresent
        );
        assert_eq!(third.target, input[0]);
        assert_eq!(third.residual, vec![0.0; 4]);
        assert_eq!(third.publication_recombination_maximum_absolute_error, 0.0);
    }

    #[test]
    fn realtime_scheduler_and_direct_fallback_publish_complete_pairs() {
        let (_directory, package) = crate::backend::causal_target_sound::tests::fixture_package();
        let model = CausalTargetSoundModel::load_runtime_package(&package, AcceleratorRuntime::Cpu)
            .unwrap();
        let mut config = CausalTargetSoundConfig::default();
        config.present_hold_blocks = 1;
        let stream = fixture_stream(&package, &model, config);
        let mut scheduler = CausalTargetSoundRealtimeScheduler::new(stream).unwrap();
        let input = [-0.5, 0.0, 0.25, 0.75];
        let token = scheduler.try_submit(&input).unwrap();
        let mut target = [0.0; 4];
        let mut residual = [0.0; 4];
        let mut received = None;
        for _ in 0..10_000 {
            received = scheduler
                .try_receive_due(token, &mut target, &mut residual)
                .unwrap();
            if received.is_some() {
                break;
            }
            std::thread::yield_now();
        }
        let result = received.expect("worker returned a bounded result");
        assert!(result.valid);
        assert_eq!(
            result.decision,
            CausalTargetSoundBlockDecision::PublishedPresent
        );
        for ((target, residual), input) in target.iter().zip(residual).zip(input) {
            assert_eq!(*target + residual, input);
        }
        write_causal_target_sound_conservative_fallback(&input, &mut target, &mut residual)
            .unwrap();
        assert_eq!(target, [0.0; 4]);
        assert_eq!(residual, input);
        scheduler.stop();
    }

    #[test]
    fn file_render_has_exact_source_clock_conservation() {
        let (_directory, package) = crate::backend::causal_target_sound::tests::fixture_package();
        let model = CausalTargetSoundModel::load_runtime_package(&package, AcceleratorRuntime::Cpu)
            .unwrap();
        let mut config = CausalTargetSoundConfig::default();
        config.present_hold_blocks = 1;
        let configuration_sha256 = config.digest().unwrap();
        let session = CausalTargetSoundSession {
            package,
            model,
            accelerator: AcceleratorSelection::default(),
            config,
            configuration_sha256,
            query: query_identity(),
            evidence: evidence_identity(),
        };
        let input = Audio {
            sample_rate: 16_000,
            channels: vec![vec![-0.5, 0.0, 0.25, 0.75, 0.5, -0.25, 0.0, 0.25]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let result = session.render(&input).unwrap();
        assert_eq!(result.target.frames(), input.frames());
        assert_eq!(result.residual.frames(), input.frames());
        for frame in 0..input.frames() {
            assert_eq!(
                result.target.channels[0][frame] + result.residual.channels[0][frame],
                input.channels[0][frame]
            );
        }
        result.report.validate().unwrap();
    }

    #[test]
    fn file_render_withholds_source_samples_across_resampler_boundary() {
        let (_directory, package) = crate::backend::causal_target_sound::tests::fixture_package();
        let model = CausalTargetSoundModel::load_runtime_package(&package, AcceleratorRuntime::Cpu)
            .unwrap();
        let config = CausalTargetSoundConfig::default();
        let configuration_sha256 = config.digest().unwrap();
        let session = CausalTargetSoundSession {
            package,
            model,
            accelerator: AcceleratorSelection::default(),
            config,
            configuration_sha256,
            query: query_identity(),
            evidence: evidence_identity(),
        };
        let input = Audio {
            sample_rate: 8_000,
            channels: vec![vec![0.25; 8]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let result = session.render(&input).unwrap();
        assert!(result.report.source_clock_withheld_frames >= 2);
        assert_eq!(&result.target.channels[0][..2], &[0.0, 0.0]);
        assert_eq!(&result.residual.channels[0][..2], &[0.25, 0.25]);
        for frame in 0..input.frames() {
            assert_eq!(
                result.target.channels[0][frame] + result.residual.channels[0][frame],
                input.channels[0][frame]
            );
        }
        result.report.validate().unwrap();
    }

    fn fixture_stream(
        package: &RuntimeModelPackage,
        model: &CausalTargetSoundModel,
        config: CausalTargetSoundConfig,
    ) -> CausalTargetSoundStream {
        CausalTargetSoundStream {
            runtime: model.start(1).unwrap(),
            config: config.clone(),
            model_package_sha256: package.package_sha256().into(),
            configuration_sha256: config.digest().unwrap(),
            query_sha256: "a".repeat(64),
            query_catalog_sha256: "b".repeat(64),
            selected_class_id: "rain".into(),
            channels: model.channels(),
            frame_samples: model.frame_samples(),
            flush_samples: model.flush_samples(),
            generation: 1,
            next_frame: 0,
            present_streak: 0,
            finished: false,
        }
    }

    fn query_identity() -> CausalTargetSoundQueryIdentity {
        CausalTargetSoundQueryIdentity {
            query_sha256: "a".repeat(64),
            catalog_sha256: "b".repeat(64),
            catalog_revision: "fixture-1".into(),
            class_ids_sha256: "c".repeat(64),
            class_count: 2,
            class_id: "rain".into(),
            class_index: 1,
            canonical_label: "Rain".into(),
            encoding: "one-hot-v1".into(),
            open_text_accepted: false,
        }
    }

    fn evidence_identity() -> CausalTargetSoundEvidenceIdentity {
        CausalTargetSoundEvidenceIdentity {
            offline_signing_key_id: "d".repeat(64),
            causal_signing_key_id: "e".repeat(64),
            offline_model_package_sha256: "f".repeat(64),
            offline_evaluation_result_sha256: "1".repeat(64),
            causal_evaluation_result_sha256: "2".repeat(64),
            state_reset_flush_result_sha256: "3".repeat(64),
            snapshot_roundtrip_result_sha256: "4".repeat(64),
            recombination_result_sha256: "5".repeat(64),
            latency_result_sha256: "6".repeat(64),
            realtime_callback_result_sha256: "7".repeat(64),
            transition_result_sha256: "8".repeat(64),
            device_measurements: 3,
            strata: REQUIRED_STRATA.len() as u32,
            accepted: true,
        }
    }

    fn accepted_payload() -> CausalTargetSoundPromotionEvidencePayload {
        let strata = REQUIRED_STRATA
            .iter()
            .map(|(id, kind)| CausalTargetSoundStratumEvidence {
                id: (*id).into(),
                kind: *kind,
                offline_cases: 100,
                causal_cases: 100,
                metrics: metric_policies(*kind)
                    .iter()
                    .map(|policy| CausalTargetSoundMetricEvidence {
                        metric: policy.name.into(),
                        operator: policy.operator,
                        offline_value: passing_value(policy),
                        causal_value: passing_value(policy),
                        hard_limit: policy.hard_limit,
                        maximum_regression: 0.0,
                        passed: true,
                    })
                    .collect(),
            })
            .collect();
        let devices = (0..3)
            .map(|index| CausalTargetSoundDeviceLatencyMeasurement {
                device_id: format!("fixture-{index}"),
                device_class: "desktop".into(),
                operating_system: "Test OS".into(),
                audio_stack: "Test Audio".into(),
                sample_rate_hz: 48_000,
                channels: 2,
                capture_milliseconds: 1.0,
                chunk_milliseconds: 1.0,
                lookahead_milliseconds: 1.0,
                resampling_milliseconds: 1.0,
                inference_milliseconds: 1.0,
                buffering_milliseconds: 1.0,
                host_milliseconds: 1.0,
                output_milliseconds: 1.0,
                total_milliseconds: 8.0,
            })
            .collect();
        CausalTargetSoundPromotionEvidencePayload {
            completed_at_unix_seconds: 1,
            offline_model_package_sha256: "a".repeat(64),
            causal_model_package_sha256: "b".repeat(64),
            causal_source_revision: "fixture-1".into(),
            causal_source_sha256: "c".repeat(64),
            causal_checkpoint_sha256: "d".repeat(64),
            offline_configuration_sha256: "e".repeat(64),
            causal_configuration_sha256: "f".repeat(64),
            query_catalog_sha256: "1".repeat(64),
            query_catalog_revision: "fixture-1".into(),
            query_class_ids_sha256: "2".repeat(64),
            query_class_count: 2,
            offline_evaluation_result_sha256: "3".repeat(64),
            causal_evaluation_result_sha256: "4".repeat(64),
            state_reset_flush_result_sha256: "5".repeat(64),
            snapshot_roundtrip_result_sha256: "6".repeat(64),
            recombination_result_sha256: "7".repeat(64),
            latency_result_sha256: "8".repeat(64),
            realtime_callback_result_sha256: "9".repeat(64),
            transition_result_sha256: "0".repeat(64),
            strata,
            model_sample_rate_hz: 48_000,
            model_channels: 2,
            frame_samples: 480,
            algorithmic_latency_samples: 960,
            flush_samples: 960,
            perturbation_latency_cases: 100,
            effective_latency_limit_milliseconds: 100.0,
            worst_effective_latency_milliseconds: 8.0,
            device_measurements: devices,
            realtime: CausalTargetSoundRealtimeAudit {
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
            transitions: CausalTargetSoundTransitionAudit {
                reset_cases: 100,
                discontinuity_cases: 100,
                dropout_cases: 100,
                overload_fallback_cases: 100,
                snapshot_roundtrip_cases: 100,
                resampler_boundary_cases: 100,
                query_mutation_rejections: 100,
                late_results_injected: 100,
                late_results_discarded: 100,
                stale_generation_results_injected: 100,
                stale_generation_results_discarded: 100,
                partial_semantic_removal_publications: 0,
                recombination_violations: 0,
            },
            accepted: true,
        }
    }

    fn passing_value(policy: &MetricPolicy) -> f64 {
        match policy.operator {
            TargetSoundMetricOperator::GreaterOrEqual => policy.hard_limit + 1.0,
            TargetSoundMetricOperator::LessOrEqual if policy.hard_limit > 0.0 => {
                policy.hard_limit * 0.5
            }
            TargetSoundMetricOperator::LessOrEqual if policy.hard_limit < 0.0 => {
                policy.hard_limit - 1.0
            }
            TargetSoundMetricOperator::LessOrEqual => 0.0,
        }
    }
}
