//! Fail-closed offline extraction of one sound from a finite semantic catalog.
//!
//! The model never receives open text. A query carries the complete ordered
//! class catalog, promotion evidence authenticates that exact catalog, and the
//! adapter sends only the selected one-hot class index to the graph. The graph
//! must return target, residual, and calibrated absent/uncertain/present
//! probabilities. Nothing is published for an absent or uncertain target, or
//! when conservation, signal, geometry, or spatial checks fail.

use crate::audio::{estimate_audio_memory_bytes, Audio};
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
#[cfg(feature = "onnx")]
use crate::{
    AcceleratorPreference, AcceleratorSelection, Backend, BackendOptions, RuntimeModelPackage,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::path::Path;

pub const TARGET_SOUND_QUERY_SCHEMA: &str = "denoize-target-sound-query-v1";
pub const TARGET_SOUND_EVIDENCE_SCHEMA: &str = "denoize-target-sound-promotion-evidence-v1";
pub const TARGET_SOUND_REPORT_SCHEMA: &str = "denoize-target-sound-report-v1";
pub const TARGET_SOUND_SCHEMA_VERSION: u32 = 1;
pub const MAX_TARGET_SOUND_CLASSES: usize = 4096;
pub const MAX_TARGET_SOUND_AUDIO_SECONDS: u64 = 6 * 60 * 60;
pub const MAX_TARGET_SOUND_WINDOWS: usize = 500_000;

const MAX_QUERY_BYTES: u64 = 4 * 1024 * 1024;
const MAX_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const QUERY_DIGEST_DOMAIN: &[u8] = b"denoize-target-sound-query-v1\0";
const CATALOG_DIGEST_DOMAIN: &[u8] = b"denoize-target-sound-catalog-v1\0";
const CLASS_IDS_DIGEST_DOMAIN: &[u8] = b"denoize-target-sound-class-ids-v1\0";
const CONFIG_DIGEST_DOMAIN: &[u8] = b"denoize-target-sound-config-v1\0";
const EVIDENCE_SIGNATURE_DOMAIN: &[u8] = b"denoize-target-sound-promotion-evidence-v1";
#[cfg(feature = "onnx")]
const INPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-target-sound-input-pcm-v1\0";
#[cfg(feature = "onnx")]
const TARGET_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-target-sound-target-pcm-v1\0";
#[cfg(feature = "onnx")]
const RESIDUAL_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-target-sound-residual-pcm-v1\0";
#[cfg(feature = "onnx")]
const OUTPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-target-sound-output-pcm-v1\0";

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

const PRESENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_least("extraction.target-si-sdr-improvement-db", 3.0),
    MetricPolicy::at_most("output.clipped-samples", 0.0),
    MetricPolicy::at_most("output.duration-mismatch-samples", 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0),
    MetricPolicy::at_least("output.protected-foreground-sdr-db", 20.0),
    MetricPolicy::at_most("presence.expected-calibration-error", 0.05),
    MetricPolicy::at_most("presence.false-negative-rate", 0.05),
    MetricPolicy::at_most("recombination.maximum-absolute-error", 1.0e-5),
    MetricPolicy::at_most("residual.target-leakage-db", -20.0),
];

const ABSENT_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_most("output.clipped-samples", 0.0),
    MetricPolicy::at_most("output.duration-mismatch-samples", 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0),
    MetricPolicy::at_most("presence.expected-calibration-error", 0.05),
    MetricPolicy::at_most("presence.false-positive-rate", 0.01),
    MetricPolicy::at_most("recombination.maximum-absolute-error", 1.0e-5),
    MetricPolicy::at_most("target.output-rms-dbfs", -60.0),
];

const BINAURAL_METRICS: &[MetricPolicy] = &[
    MetricPolicy::at_least("extraction.target-si-sdr-improvement-db", 3.0),
    MetricPolicy::at_most("output.clipped-samples", 0.0),
    MetricPolicy::at_most("output.duration-mismatch-samples", 0.0),
    MetricPolicy::at_most("output.non-finite-samples", 0.0),
    MetricPolicy::at_most("presence.expected-calibration-error", 0.05),
    MetricPolicy::at_most("presence.false-negative-rate", 0.05),
    MetricPolicy::at_most("recombination.maximum-absolute-error", 1.0e-5),
    MetricPolicy::at_most("residual.target-leakage-db", -20.0),
    MetricPolicy::at_most("spatial.ild-error-db", 1.0),
    MetricPolicy::at_most("spatial.itd-error-microseconds", 100.0),
];

#[derive(Clone, Copy)]
struct MetricPolicy {
    name: &'static str,
    operator: TargetSoundMetricOperator,
    hard_limit: f64,
}

impl MetricPolicy {
    const fn at_least(name: &'static str, hard_limit: f64) -> Self {
        Self {
            name,
            operator: TargetSoundMetricOperator::GreaterOrEqual,
            hard_limit,
        }
    }

    const fn at_most(name: &'static str, hard_limit: f64) -> Self {
        Self {
            name,
            operator: TargetSoundMetricOperator::LessOrEqual,
            hard_limit,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundCatalogClass {
    pub id: String,
    pub canonical_label: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundQuery {
    pub schema: String,
    pub schema_version: u32,
    pub catalog_revision: String,
    /// Array order is the authenticated one-hot index order.
    pub classes: Vec<TargetSoundCatalogClass>,
    pub selected_class_id: String,
}

#[derive(Serialize)]
struct CatalogDigestDocument<'a> {
    catalog_revision: &'a str,
    classes: &'a [TargetSoundCatalogClass],
}

impl TargetSoundQuery {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) = crate::input::open_regular_file(path, "target-sound query")?;
        if length >= MAX_QUERY_BYTES {
            return Err(format!(
                "target-sound query {} exceeds {MAX_QUERY_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve target-sound query JSON".to_string())?;
        file.take(MAX_QUERY_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read target-sound query: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("target-sound query changed while reading".into());
        }
        let query: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse target-sound query: {error}"))?;
        query.validate()?;
        Ok(query)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != TARGET_SOUND_QUERY_SCHEMA
            || self.schema_version != TARGET_SOUND_SCHEMA_VERSION
        {
            return Err("unsupported target-sound query schema".into());
        }
        validate_identifier("catalog revision", &self.catalog_revision)?;
        if !(2..=MAX_TARGET_SOUND_CLASSES).contains(&self.classes.len()) {
            return Err(format!(
                "target-sound query catalog must contain 2..={MAX_TARGET_SOUND_CLASSES} classes"
            ));
        }
        validate_identifier("selected class ID", &self.selected_class_id)?;
        let mut ids = BTreeSet::new();
        for class in &self.classes {
            validate_identifier("catalog class ID", &class.id)?;
            validate_bounded_text("catalog canonical label", &class.canonical_label, 160)?;
            if !ids.insert(class.id.as_str()) {
                return Err("target-sound catalog class IDs must be unique".into());
            }
        }
        if !ids.contains(self.selected_class_id.as_str()) {
            return Err("target-sound selected class is absent from the catalog".into());
        }
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize target-sound query: {error}"))?;
        if encoded.len() as u64 >= MAX_QUERY_BYTES {
            return Err("target-sound query exceeds the bounded JSON limit".into());
        }
        Ok(())
    }

    pub fn selected_index(&self) -> Result<usize, String> {
        self.validate()?;
        self.classes
            .iter()
            .position(|class| class.id == self.selected_class_id)
            .ok_or_else(|| "target-sound selected class is absent from the catalog".into())
    }

    pub fn selected_class(&self) -> Result<&TargetSoundCatalogClass, String> {
        let index = self.selected_index()?;
        Ok(&self.classes[index])
    }

    pub fn catalog_sha256(&self) -> Result<String, String> {
        self.validate()?;
        digest_json(
            CATALOG_DIGEST_DOMAIN,
            &CatalogDigestDocument {
                catalog_revision: &self.catalog_revision,
                classes: &self.classes,
            },
            "target-sound catalog",
        )
    }

    pub fn class_ids_sha256(&self) -> Result<String, String> {
        self.validate()?;
        let ids = self
            .classes
            .iter()
            .map(|class| class.id.as_str())
            .collect::<Vec<_>>();
        digest_json(CLASS_IDS_DIGEST_DOMAIN, &ids, "target-sound class IDs")
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        digest_json(QUERY_DIGEST_DOMAIN, self, "target-sound query")
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize target-sound query: {error}"))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSoundMode {
    Preserve,
    Remove,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundConfig {
    pub mode: TargetSoundMode,
    pub minimum_present_probability: f64,
    pub minimum_absent_probability: f64,
    pub maximum_model_recombination_error: f64,
    pub maximum_publication_recombination_error: f64,
    pub maximum_target_peak: f64,
    pub maximum_residual_peak: f64,
    pub maximum_energy_gain_db: f64,
    pub maximum_stereo_correlation_delta: f64,
    pub maximum_mid_side_energy_ratio_delta_db: f64,
}

impl Default for TargetSoundConfig {
    fn default() -> Self {
        Self {
            mode: TargetSoundMode::Preserve,
            minimum_present_probability: 0.90,
            minimum_absent_probability: 0.90,
            maximum_model_recombination_error: 0.01,
            maximum_publication_recombination_error: 1.0e-12,
            maximum_target_peak: 1.0,
            maximum_residual_peak: 1.0,
            maximum_energy_gain_db: 3.0,
            maximum_stereo_correlation_delta: 0.05,
            maximum_mid_side_energy_ratio_delta_db: 1.5,
        }
    }
}

impl TargetSoundConfig {
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
            "maximum_model_recombination_error",
            self.maximum_model_recombination_error,
            0.0,
            0.10,
        )?;
        validate_range(
            "maximum_publication_recombination_error",
            self.maximum_publication_recombination_error,
            0.0,
            1.0e-6,
        )?;
        validate_range("maximum_target_peak", self.maximum_target_peak, 0.5, 1.0)?;
        validate_range(
            "maximum_residual_peak",
            self.maximum_residual_peak,
            0.5,
            1.0,
        )?;
        validate_range(
            "maximum_energy_gain_db",
            self.maximum_energy_gain_db,
            0.0,
            12.0,
        )?;
        validate_range(
            "maximum_stereo_correlation_delta",
            self.maximum_stereo_correlation_delta,
            0.0,
            0.25,
        )?;
        validate_range(
            "maximum_mid_side_energy_ratio_delta_db",
            self.maximum_mid_side_energy_ratio_delta_db,
            0.0,
            6.0,
        )
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        digest_json(CONFIG_DIGEST_DOMAIN, self, "target-sound configuration")
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSoundStratumKind {
    TargetPresent,
    TargetAbsent,
    ProtectedForeground,
    BinauralSpatial,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSoundMetricOperator {
    GreaterOrEqual,
    LessOrEqual,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundMetricOutcome {
    pub metric: String,
    pub value: f64,
    pub operator: TargetSoundMetricOperator,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundEvidenceStratum {
    pub id: String,
    pub kind: TargetSoundStratumKind,
    pub cases: u64,
    pub metrics: Vec<TargetSoundMetricOutcome>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub model_package_sha256: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub checkpoint_sha256: String,
    pub configuration_sha256: String,
    pub query_catalog_sha256: String,
    pub query_catalog_revision: String,
    pub query_class_ids_sha256: String,
    pub query_class_count: u32,
    pub class_coverage_manifest_sha256: String,
    pub evaluated_class_count: u32,
    pub minimum_present_cases_per_class: u64,
    pub minimum_absent_cases_per_class: u64,
    pub worst_class_false_positive_rate: f64,
    pub worst_class_false_negative_rate: f64,
    pub artifact_bom_sha256: String,
    pub training_dataset_license_manifest_sha256: String,
    pub evaluation_corpus_manifest_sha256: String,
    pub evaluation_corpus_license_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub listening_result_sha256: String,
    pub strata: Vec<TargetSoundEvidenceStratum>,
    pub paired_cases: u64,
    pub target_absent_cases: u64,
    pub protected_foreground_cases: u64,
    pub binaural_cases: u64,
    pub listener_count: u64,
    pub listener_preference: f64,
    pub redistributed_restricted_artifacts: u64,
    pub unresolved_artifact_licenses: u64,
    pub unresolved_training_dataset_licenses: u64,
    pub unresolved_evaluation_dataset_licenses: u64,
    pub accepted: bool,
}

impl TargetSoundPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        if self.completed_at_unix_seconds > JSON_SAFE_INTEGER {
            return Err("target-sound evidence timestamp exceeds JSON safe integer".into());
        }
        validate_identifier("source revision", &self.source_revision)?;
        validate_identifier("query catalog revision", &self.query_catalog_revision)?;
        for (label, value) in [
            ("model package", self.model_package_sha256.as_str()),
            ("source", self.source_sha256.as_str()),
            ("checkpoint", self.checkpoint_sha256.as_str()),
            ("configuration", self.configuration_sha256.as_str()),
            ("query catalog", self.query_catalog_sha256.as_str()),
            ("query class IDs", self.query_class_ids_sha256.as_str()),
            (
                "class coverage manifest",
                self.class_coverage_manifest_sha256.as_str(),
            ),
            ("artifact BOM", self.artifact_bom_sha256.as_str()),
            (
                "training dataset license manifest",
                self.training_dataset_license_manifest_sha256.as_str(),
            ),
            (
                "evaluation corpus manifest",
                self.evaluation_corpus_manifest_sha256.as_str(),
            ),
            (
                "evaluation corpus license manifest",
                self.evaluation_corpus_license_manifest_sha256.as_str(),
            ),
            ("evaluation result", self.evaluation_result_sha256.as_str()),
            ("listening result", self.listening_result_sha256.as_str()),
        ] {
            validate_sha256(label, value)?;
        }
        if !(2..=MAX_TARGET_SOUND_CLASSES as u32).contains(&self.query_class_count) {
            return Err("target-sound evidence query class count is outside 2..=4096".into());
        }
        let class_case_floor = u64::from(self.evaluated_class_count)
            .checked_mul(
                self.minimum_present_cases_per_class
                    .saturating_add(self.minimum_absent_cases_per_class),
            )
            .ok_or_else(|| "target-sound per-class coverage count overflow".to_string())?;
        let class_coverage_valid = self.evaluated_class_count == self.query_class_count
            && (20..=1_000_000).contains(&self.minimum_present_cases_per_class)
            && (20..=1_000_000).contains(&self.minimum_absent_cases_per_class)
            && self.worst_class_false_positive_rate.is_finite()
            && (0.0..=0.01).contains(&self.worst_class_false_positive_rate)
            && self.worst_class_false_negative_rate.is_finite()
            && (0.0..=0.05).contains(&self.worst_class_false_negative_rate);
        if !class_coverage_valid {
            return Err("target-sound evidence does not cover every catalog class".into());
        }
        if self.strata.len() != REQUIRED_STRATA.len() {
            return Err(format!(
                "target-sound evidence requires exactly {} strata",
                REQUIRED_STRATA.len()
            ));
        }
        let mut all_metrics_passed = true;
        for (index, stratum) in self.strata.iter().enumerate() {
            let (expected_id, expected_kind) = REQUIRED_STRATA[index];
            if stratum.id != expected_id || stratum.kind != expected_kind {
                return Err("target-sound evidence strata must be exact and sorted".into());
            }
            if !(50..=1_000_000).contains(&stratum.cases) {
                return Err("target-sound evidence stratum cases must be in 50..=1000000".into());
            }
            let policies = metric_policies(stratum.kind);
            if stratum.metrics.len() != policies.len() {
                return Err(format!(
                    "target-sound evidence stratum {} has the wrong metric count",
                    stratum.id
                ));
            }
            for (metric, policy) in stratum.metrics.iter().zip(policies) {
                validate_identifier("evidence metric", &metric.metric)?;
                if metric.metric != policy.name {
                    return Err(format!(
                        "target-sound evidence stratum {} metrics must be exact and sorted",
                        stratum.id
                    ));
                }
                validate_metric_policy(metric, policy)?;
                all_metrics_passed &= metric.passed;
            }
        }
        let counts_valid = (1_000..=10_000_000).contains(&self.paired_cases)
            && self.paired_cases >= class_case_floor
            && (200..=1_000_000).contains(&self.target_absent_cases)
            && (200..=1_000_000).contains(&self.protected_foreground_cases)
            && (200..=1_000_000).contains(&self.binaural_cases)
            && (20..=100_000).contains(&self.listener_count)
            && self.listener_preference.is_finite()
            && (0.5..=1.0).contains(&self.listener_preference);
        if !counts_valid {
            return Err("target-sound global evaluation counts are outside hard limits".into());
        }
        let licenses_clear = self.redistributed_restricted_artifacts == 0
            && self.unresolved_artifact_licenses == 0
            && self.unresolved_training_dataset_licenses == 0
            && self.unresolved_evaluation_dataset_licenses == 0;
        if !licenses_clear {
            return Err(
                "target-sound promotion evidence contains unresolved or restricted artifacts"
                    .into(),
            );
        }
        let expected_accepted = all_metrics_passed
            && class_coverage_valid
            && counts_valid
            && licenses_clear
            && self.listener_preference >= 0.5;
        if self.accepted != expected_accepted {
            return Err("target-sound accepted flag is inconsistent".into());
        }
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize target-sound evidence payload: {error}"))?;
        if encoded.len() as u64 >= MAX_EVIDENCE_BYTES {
            return Err("target-sound evidence payload exceeds the bounded JSON limit".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedTargetSoundPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: TargetSoundPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedTargetSoundPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) =
            crate::input::open_regular_file(path, "target-sound promotion evidence")?;
        if length >= MAX_EVIDENCE_BYTES {
            return Err(format!(
                "target-sound promotion evidence {} exceeds {MAX_EVIDENCE_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve target-sound evidence JSON".to_string())?;
        file.take(MAX_EVIDENCE_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read target-sound promotion evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("target-sound promotion evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse target-sound promotion evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != TARGET_SOUND_EVIDENCE_SCHEMA
            || self.schema_version != TARGET_SOUND_SCHEMA_VERSION
        {
            return Err("unsupported target-sound promotion evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("target-sound promotion evidence must use ed25519".into());
        }
        validate_sha256("evidence key ID", &self.signature.key_id)?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize target-sound evidence: {error}"))?;
        if encoded.len() as u64 >= MAX_EVIDENCE_BYTES {
            return Err("target-sound promotion evidence exceeds the bounded JSON limit".into());
        }
        Ok(())
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize target-sound evidence: {error}"))?;
        key.verify_domain_document(
            EVIDENCE_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "target-sound promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize target-sound evidence: {error}"))
    }
}

pub fn sign_target_sound_promotion_evidence(
    payload: TargetSoundPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedTargetSoundPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize target-sound evidence: {error}"))?;
    let signature = key.sign_domain_document(
        EVIDENCE_SIGNATURE_DOMAIN,
        &document,
        "target-sound promotion evidence",
    )?;
    let evidence = SignedTargetSoundPromotionEvidence {
        schema: TARGET_SOUND_EVIDENCE_SCHEMA.into(),
        schema_version: TARGET_SOUND_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSoundPresence {
    Present,
    Absent,
    Uncertain,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSoundDecision {
    AcceptedPresent,
    WithheldAbsent,
    WithheldUncertain,
    WithheldSafetyGate,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetSoundSafetyGateKind {
    QueryCatalog,
    Geometry,
    FiniteNormalizedSamples,
    ModelRecombination,
    PublishedRecombination,
    TargetPeak,
    ResidualPeak,
    TargetEnergyGain,
    ResidualEnergyGain,
    TargetStereoCorrelation,
    TargetMidSideEnergy,
    ResidualStereoCorrelation,
    ResidualMidSideEnergy,
    TargetPresence,
    PromotionEvidence,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundSafetyGate {
    pub kind: TargetSoundSafetyGateKind,
    pub observed: f64,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundPresenceAssessment {
    pub state: TargetSoundPresence,
    pub absent_probability: f64,
    pub uncertain_probability: f64,
    pub present_probability: f64,
    pub minimum_absent_probability: f64,
    pub minimum_present_probability: f64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundQueryIdentity {
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundTrainingDatasetIdentity {
    pub id: String,
    pub revision: String,
    pub sha256: Option<String>,
    pub license_spdx: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundModelIdentity {
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
    pub training_datasets: Vec<TargetSoundTrainingDatasetIdentity>,
    pub accelerator: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundEvidenceIdentity {
    pub signing_key_id: String,
    pub class_coverage_manifest_sha256: String,
    pub evaluated_class_count: u32,
    pub minimum_present_cases_per_class: u64,
    pub minimum_absent_cases_per_class: u64,
    pub worst_class_false_positive_rate: f64,
    pub worst_class_false_negative_rate: f64,
    pub artifact_bom_sha256: String,
    pub training_dataset_license_manifest_sha256: String,
    pub evaluation_corpus_manifest_sha256: String,
    pub evaluation_corpus_license_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub listening_result_sha256: String,
    pub strata: u32,
    pub paired_cases: u64,
    pub target_absent_cases: u64,
    pub protected_foreground_cases: u64,
    pub binaural_cases: u64,
    pub listener_count: u64,
    pub accepted: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundSafetyMeasurements {
    pub input_rms_dbfs: f64,
    pub target_rms_dbfs: f64,
    pub residual_rms_dbfs: f64,
    pub input_peak: f64,
    pub target_peak: f64,
    pub residual_peak: f64,
    pub target_energy_gain_db: f64,
    pub residual_energy_gain_db: f64,
    pub model_recombination_maximum_absolute_error: f64,
    pub publication_recombination_maximum_absolute_error: f64,
    pub target_stereo_correlation_delta: Option<f64>,
    pub target_mid_side_energy_ratio_delta_db: Option<f64>,
    pub residual_stereo_correlation_delta: Option<f64>,
    pub residual_mid_side_energy_ratio_delta_db: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSoundReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub configuration_sha256: String,
    pub mode: TargetSoundMode,
    pub network_accessed: bool,
    pub deterministic: bool,
    pub closed_class_query: bool,
    pub model_invoked: bool,
    pub query: TargetSoundQueryIdentity,
    pub model: TargetSoundModelIdentity,
    pub promotion_evidence: TargetSoundEvidenceIdentity,
    pub decision: TargetSoundDecision,
    pub candidate_accepted: bool,
    pub target_published: bool,
    pub residual_published: bool,
    pub output_published: bool,
    pub candidates_retained: bool,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_frames: usize,
    pub model_sample_rate: u32,
    pub model_channels: usize,
    pub model_window_samples: usize,
    pub model_hop_samples: usize,
    pub model_windows: usize,
    pub input_pcm_sha256: String,
    pub target_pcm_sha256: Option<String>,
    pub residual_pcm_sha256: Option<String>,
    pub output_pcm_sha256: Option<String>,
    pub presence: TargetSoundPresenceAssessment,
    pub measurements: TargetSoundSafetyMeasurements,
    pub safety_gates: Vec<TargetSoundSafetyGate>,
    pub path_fields_recorded: u64,
    pub limitations: Vec<String>,
    pub warnings: Vec<String>,
}

impl TargetSoundReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != TARGET_SOUND_REPORT_SCHEMA
            || self.schema_version != TARGET_SOUND_SCHEMA_VERSION
            || !is_semver_triplet(&self.denoize_version)
            || self.network_accessed
            || !self.deterministic
            || !self.closed_class_query
            || !self.model_invoked
        {
            return Err("unsupported target-sound report header".into());
        }
        for (label, value) in [
            ("configuration", self.configuration_sha256.as_str()),
            ("query", self.query.query_sha256.as_str()),
            ("query catalog", self.query.catalog_sha256.as_str()),
            ("query class IDs", self.query.class_ids_sha256.as_str()),
            ("model package", self.model.package_sha256.as_str()),
            ("model public key", self.model.public_key_sha256.as_str()),
            ("model source", self.model.source_sha256.as_str()),
            ("model checkpoint", self.model.checkpoint_sha256.as_str()),
            (
                "evidence signing key",
                self.promotion_evidence.signing_key_id.as_str(),
            ),
            (
                "evidence artifact BOM",
                self.promotion_evidence.artifact_bom_sha256.as_str(),
            ),
            (
                "evidence class coverage manifest",
                self.promotion_evidence
                    .class_coverage_manifest_sha256
                    .as_str(),
            ),
            (
                "evidence training license manifest",
                self.promotion_evidence
                    .training_dataset_license_manifest_sha256
                    .as_str(),
            ),
            (
                "evidence evaluation corpus",
                self.promotion_evidence
                    .evaluation_corpus_manifest_sha256
                    .as_str(),
            ),
            (
                "evidence evaluation corpus license",
                self.promotion_evidence
                    .evaluation_corpus_license_manifest_sha256
                    .as_str(),
            ),
            (
                "evidence evaluation",
                self.promotion_evidence.evaluation_result_sha256.as_str(),
            ),
            (
                "evidence listening",
                self.promotion_evidence.listening_result_sha256.as_str(),
            ),
            ("input PCM", self.input_pcm_sha256.as_str()),
        ] {
            validate_sha256(label, value)?;
        }
        validate_identifier("query catalog revision", &self.query.catalog_revision)?;
        validate_identifier("query class ID", &self.query.class_id)?;
        validate_bounded_text("query canonical label", &self.query.canonical_label, 160)?;
        if self.query.encoding != "one-hot-v1"
            || self.query.open_text_accepted
            || !(2..=MAX_TARGET_SOUND_CLASSES as u32).contains(&self.query.class_count)
            || self.query.class_index >= self.query.class_count
        {
            return Err("target-sound report query identity is invalid".into());
        }
        for (label, value, maximum) in [
            ("model package ID", self.model.package_id.as_str(), 256),
            (
                "model package revision",
                self.model.package_revision.as_str(),
                256,
            ),
            (
                "model precision profile",
                self.model.precision_profile.as_str(),
                256,
            ),
            (
                "model package license",
                self.model.package_license_spdx.as_str(),
                512,
            ),
            (
                "model source revision",
                self.model.source_revision.as_str(),
                512,
            ),
            (
                "model source license",
                self.model.source_license_spdx.as_str(),
                512,
            ),
            (
                "model checkpoint license",
                self.model.checkpoint_license_spdx.as_str(),
                512,
            ),
        ] {
            validate_bounded_text(label, value, maximum)?;
        }
        if self.model.training_datasets.len() > 64 {
            return Err("target-sound report has too many training datasets".into());
        }
        let mut dataset_ids = BTreeSet::new();
        for dataset in &self.model.training_datasets {
            validate_bounded_text("training dataset ID", &dataset.id, 256)?;
            validate_bounded_text("training dataset revision", &dataset.revision, 512)?;
            validate_bounded_text("training dataset license", &dataset.license_spdx, 512)?;
            if !dataset_ids.insert(dataset.id.as_str()) {
                return Err("target-sound training dataset IDs must be unique".into());
            }
            if let Some(sha256) = &dataset.sha256 {
                validate_sha256("training dataset", sha256)?;
            }
        }
        let accepted = self.decision == TargetSoundDecision::AcceptedPresent;
        let digests_present = self.target_pcm_sha256.is_some()
            && self.residual_pcm_sha256.is_some()
            && self.output_pcm_sha256.is_some();
        if accepted != self.candidate_accepted
            || accepted != self.target_published
            || accepted != self.residual_published
            || accepted != self.output_published
            || accepted != self.candidates_retained
            || accepted != digests_present
        {
            return Err("target-sound publication flags are inconsistent".into());
        }
        for (label, digest) in [
            ("target PCM", self.target_pcm_sha256.as_deref()),
            ("residual PCM", self.residual_pcm_sha256.as_deref()),
            ("output PCM", self.output_pcm_sha256.as_deref()),
        ] {
            if let Some(digest) = digest {
                validate_sha256(label, digest)?;
            }
        }
        let report_class_case_floor = u64::from(self.promotion_evidence.evaluated_class_count)
            .checked_mul(
                self.promotion_evidence
                    .minimum_present_cases_per_class
                    .saturating_add(self.promotion_evidence.minimum_absent_cases_per_class),
            )
            .ok_or_else(|| "target-sound report class coverage count overflow".to_string())?;
        if !self.promotion_evidence.accepted
            || self.promotion_evidence.strata != REQUIRED_STRATA.len() as u32
            || self.promotion_evidence.evaluated_class_count != self.query.class_count
            || self.promotion_evidence.minimum_present_cases_per_class < 20
            || self.promotion_evidence.minimum_present_cases_per_class > 1_000_000
            || self.promotion_evidence.minimum_absent_cases_per_class < 20
            || self.promotion_evidence.minimum_absent_cases_per_class > 1_000_000
            || !self
                .promotion_evidence
                .worst_class_false_positive_rate
                .is_finite()
            || !(0.0..=0.01).contains(&self.promotion_evidence.worst_class_false_positive_rate)
            || !self
                .promotion_evidence
                .worst_class_false_negative_rate
                .is_finite()
            || !(0.0..=0.05).contains(&self.promotion_evidence.worst_class_false_negative_rate)
            || self.promotion_evidence.paired_cases < 1_000
            || self.promotion_evidence.paired_cases > 10_000_000
            || self.promotion_evidence.paired_cases < report_class_case_floor
            || self.promotion_evidence.target_absent_cases < 200
            || self.promotion_evidence.target_absent_cases > 1_000_000
            || self.promotion_evidence.protected_foreground_cases < 200
            || self.promotion_evidence.protected_foreground_cases > 1_000_000
            || self.promotion_evidence.binaural_cases < 200
            || self.promotion_evidence.binaural_cases > 1_000_000
            || self.promotion_evidence.listener_count < 20
            || self.promotion_evidence.listener_count > 100_000
            || !matches!(self.model.accelerator.as_str(), "cpu" | "metal" | "cuda")
            || !(8_000..=192_000).contains(&self.source_sample_rate)
            || !(1..=2).contains(&self.source_channels)
            || self.source_frames == 0
            || self.source_frames as u64
                > u64::from(self.source_sample_rate).saturating_mul(MAX_TARGET_SOUND_AUDIO_SECONDS)
            || !(8_000..=192_000).contains(&self.model_sample_rate)
            || self.model_channels != self.source_channels
            || !(256..=16_777_216).contains(&self.model_window_samples)
            || self.model_hop_samples == 0
            || self.model_hop_samples > self.model_window_samples
            || self.model_windows == 0
            || self.model_windows > MAX_TARGET_SOUND_WINDOWS
            || self.path_fields_recorded != 0
            || self.limitations.is_empty()
            || self.limitations.len() > 32
            || self.warnings.len() > 32
        {
            return Err("target-sound report violates bounded result contracts".into());
        }
        validate_presence(&self.presence)?;
        validate_measurements(&self.measurements, self.source_channels)?;
        validate_gates(
            &self.safety_gates,
            self.source_channels,
            accepted,
            &self.presence,
            &self.measurements,
        )?;
        match (self.presence.state, self.decision) {
            (TargetSoundPresence::Present, TargetSoundDecision::AcceptedPresent)
            | (TargetSoundPresence::Present, TargetSoundDecision::WithheldSafetyGate)
            | (TargetSoundPresence::Absent, TargetSoundDecision::WithheldAbsent)
            | (TargetSoundPresence::Uncertain, TargetSoundDecision::WithheldUncertain) => {}
            _ => return Err("target-sound decision conflicts with presence state".into()),
        }
        for text in self.limitations.iter().chain(&self.warnings) {
            validate_bounded_text("target-sound report text", text, 512)?;
        }
        Ok(())
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize target-sound report: {error}"))
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize target-sound report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct TargetSoundResult {
    /// All three values are `None` when publication is withheld. Callers must
    /// not substitute the mixture or retain unverified model candidates.
    pub target: Option<Audio>,
    pub residual: Option<Audio>,
    pub output: Option<Audio>,
    pub report: TargetSoundReport,
}

pub fn estimate_target_sound_memory_bytes(
    input: &Audio,
    model_sample_rate: u32,
    model_channels: usize,
    model_window_samples: usize,
) -> Result<u64, String> {
    if input.sample_rate == 0
        || input.channels.is_empty()
        || !(8_000..=192_000).contains(&model_sample_rate)
        || !(1..=2).contains(&model_channels)
        || !(256..=16_777_216).contains(&model_window_samples)
    {
        return Err("target-sound memory geometry is invalid".into());
    }
    let model_frames = crate::resample::planned_output_frames(
        input.frames(),
        input.sample_rate,
        model_sample_rate,
    )?;
    let retained = (model_frames as u128)
        .checked_mul(model_channels as u128)
        .and_then(|value| value.checked_mul(12))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() as u128))
        .ok_or_else(|| "target-sound memory estimate overflow".to_string())?;
    let windows = (model_window_samples as u128)
        .checked_mul(model_channels as u128)
        .and_then(|value| value.checked_mul(6))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u128))
        .ok_or_else(|| "target-sound memory estimate overflow".to_string())?;
    let source = u128::from(estimate_audio_memory_bytes(input)).saturating_mul(8);
    let resampler = u128::from(crate::resample::resampler_plan_bytes(
        model_channels,
        input.sample_rate,
        model_sample_rate,
    )?);
    let bytes = retained
        .checked_add(windows)
        .and_then(|value| value.checked_add(source))
        .and_then(|value| value.checked_add(resampler))
        .ok_or_else(|| "target-sound memory estimate overflow".to_string())?;
    u64::try_from(bytes).map_err(|_| "target-sound memory estimate exceeds u64".to_string())
}

#[cfg(feature = "onnx")]
pub struct TargetSoundSession {
    package: RuntimeModelPackage,
    model: crate::backend::target_sound::TargetSoundModel,
    accelerator: AcceleratorSelection,
    evidence: TargetSoundEvidenceIdentity,
    configuration_sha256: String,
    catalog_sha256: String,
    catalog_revision: String,
    class_ids_sha256: String,
    class_count: usize,
}

#[cfg(feature = "onnx")]
impl std::fmt::Debug for TargetSoundSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TargetSoundSession")
            .field("package_sha256", &self.package.package_sha256())
            .field("model", &self.model)
            .field("accelerator", &self.accelerator)
            .field("catalog_sha256", &self.catalog_sha256)
            .field("class_count", &self.class_count)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "onnx")]
impl TargetSoundSession {
    /// Authenticate and bind the package, finite catalog, configuration, and
    /// promotion evidence before user audio is decoded.
    pub fn prepare(
        package: RuntimeModelPackage,
        evidence: &SignedTargetSoundPromotionEvidence,
        evidence_key: &ReceiptPublicKey,
        query: &TargetSoundQuery,
        config: &TargetSoundConfig,
        requested: AcceleratorPreference,
    ) -> Result<Self, String> {
        config.validate()?;
        query.validate()?;
        evidence.verify_signature(evidence_key)?;
        if !evidence.payload.accepted {
            return Err(
                "target-sound evidence is authentic but does not pass promotion gates".into(),
            );
        }
        let manifest = package
            .manifest_v2()
            .ok_or("target-sound extraction rejects runtime model package v1")?;
        let configuration_sha256 = config.digest()?;
        let catalog_sha256 = query.catalog_sha256()?;
        let class_ids_sha256 = query.class_ids_sha256()?;
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
            (
                "configuration SHA-256",
                evidence.payload.configuration_sha256.as_str(),
                configuration_sha256.as_str(),
            ),
            (
                "query catalog SHA-256",
                evidence.payload.query_catalog_sha256.as_str(),
                catalog_sha256.as_str(),
            ),
            (
                "query catalog revision",
                evidence.payload.query_catalog_revision.as_str(),
                query.catalog_revision.as_str(),
            ),
            (
                "query class IDs SHA-256",
                evidence.payload.query_class_ids_sha256.as_str(),
                class_ids_sha256.as_str(),
            ),
        ] {
            if observed != expected {
                return Err(format!(
                    "target-sound promotion evidence {label} does not match the authenticated runtime inputs"
                ));
            }
        }
        if evidence.payload.query_class_count as usize != query.classes.len() {
            return Err("target-sound evidence class count does not match the catalog".into());
        }
        let mut options = BackendOptions::default().with_runtime_model_package(package.clone());
        options.deterministic = true;
        options.accelerator = requested;
        let accelerator = crate::select_accelerator_for_options(Backend::Onnx, &options)?;
        if !package.supports_accelerator(accelerator.effective()) {
            return Err(format!(
                "target-sound package does not permit the {} accelerator",
                accelerator.effective().name()
            ));
        }
        let model = crate::backend::target_sound::TargetSoundModel::load_runtime_package(
            &package,
            accelerator.effective(),
        )?;
        if model.query_classes() != query.classes.len() {
            return Err(format!(
                "target-sound model exposes {} query classes but the authenticated catalog contains {}",
                model.query_classes(),
                query.classes.len()
            ));
        }
        let payload = &evidence.payload;
        Ok(Self {
            package,
            model,
            accelerator,
            evidence: TargetSoundEvidenceIdentity {
                signing_key_id: evidence.signature.key_id.clone(),
                class_coverage_manifest_sha256: payload.class_coverage_manifest_sha256.clone(),
                evaluated_class_count: payload.evaluated_class_count,
                minimum_present_cases_per_class: payload.minimum_present_cases_per_class,
                minimum_absent_cases_per_class: payload.minimum_absent_cases_per_class,
                worst_class_false_positive_rate: payload.worst_class_false_positive_rate,
                worst_class_false_negative_rate: payload.worst_class_false_negative_rate,
                artifact_bom_sha256: payload.artifact_bom_sha256.clone(),
                training_dataset_license_manifest_sha256: payload
                    .training_dataset_license_manifest_sha256
                    .clone(),
                evaluation_corpus_manifest_sha256: payload
                    .evaluation_corpus_manifest_sha256
                    .clone(),
                evaluation_corpus_license_manifest_sha256: payload
                    .evaluation_corpus_license_manifest_sha256
                    .clone(),
                evaluation_result_sha256: payload.evaluation_result_sha256.clone(),
                listening_result_sha256: payload.listening_result_sha256.clone(),
                strata: payload.strata.len() as u32,
                paired_cases: payload.paired_cases,
                target_absent_cases: payload.target_absent_cases,
                protected_foreground_cases: payload.protected_foreground_cases,
                binaural_cases: payload.binaural_cases,
                listener_count: payload.listener_count,
                accepted: true,
            },
            configuration_sha256,
            catalog_sha256,
            catalog_revision: query.catalog_revision.clone(),
            class_ids_sha256,
            class_count: query.classes.len(),
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
            .expect("target-sound packages use v2 precision profiles");
        Ok(profile
            .resources
            .max_session_memory_bytes
            .saturating_add(profile.resources.max_worker_memory_bytes))
    }

    pub fn processing_working_set_bytes(&self, input: &Audio) -> Result<u64, String> {
        let manifest = self
            .package
            .manifest_v2()
            .expect("target-sound session requires v2");
        estimate_target_sound_memory_bytes(
            input,
            manifest.runtime.sample_rate_hz,
            self.model.channels(),
            self.model.window_samples(),
        )
    }
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy)]
struct SignalMeasurements {
    rms_dbfs: f64,
    peak: f64,
}

#[cfg(feature = "onnx")]
fn signal_measurements(channels: &[Vec<f64>]) -> SignalMeasurements {
    let mut count = 0usize;
    let mut energy = 0.0_f64;
    let mut peak = 0.0_f64;
    for sample in channels.iter().flatten().copied() {
        count = count.saturating_add(1);
        if sample.is_finite() {
            energy += sample * sample;
            peak = peak.max(sample.abs());
        } else {
            energy = f64::INFINITY;
            peak = f64::INFINITY;
        }
    }
    let rms = if count == 0 {
        0.0
    } else {
        (energy / count as f64).sqrt()
    };
    SignalMeasurements {
        rms_dbfs: amplitude_dbfs(rms),
        peak: if peak.is_finite() {
            peak.clamp(0.0, 240.0)
        } else {
            240.0
        },
    }
}

#[cfg(feature = "onnx")]
fn amplitude_dbfs(amplitude: f64) -> f64 {
    if !amplitude.is_finite() {
        240.0
    } else {
        (20.0 * amplitude.max(1.0e-12).log10()).clamp(-240.0, 240.0)
    }
}

#[cfg(feature = "onnx")]
fn bounded_dbfs_delta(output_dbfs: f64, input_dbfs: f64) -> f64 {
    (output_dbfs - input_dbfs).clamp(-240.0, 240.0)
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy)]
struct SpatialDeltas {
    target_correlation_delta: f64,
    target_mid_side_delta_db: f64,
    residual_correlation_delta: f64,
    residual_mid_side_delta_db: f64,
}

#[cfg(feature = "onnx")]
fn spatial_deltas(
    target_model: &[Vec<f64>],
    residual_model: &[Vec<f64>],
    target_source: &[Vec<f64>],
    residual_source: &[Vec<f64>],
) -> Result<Option<SpatialDeltas>, String> {
    if target_model.len() == 1 {
        return Ok(None);
    }
    for channels in [target_model, residual_model, target_source, residual_source] {
        if channels.len() != 2 || channels[0].len() != channels[1].len() {
            return Err("target-sound spatial measurement geometry is invalid".into());
        }
    }
    Ok(Some(SpatialDeltas {
        target_correlation_delta: (normalized_correlation(&target_model[0], &target_model[1])
            - normalized_correlation(&target_source[0], &target_source[1]))
        .abs()
        .min(240.0),
        target_mid_side_delta_db: (mid_side_energy_ratio_db(target_model)?
            - mid_side_energy_ratio_db(target_source)?)
        .abs()
        .min(240.0),
        residual_correlation_delta: (normalized_correlation(
            &residual_model[0],
            &residual_model[1],
        ) - normalized_correlation(
            &residual_source[0],
            &residual_source[1],
        ))
        .abs()
        .min(240.0),
        residual_mid_side_delta_db: (mid_side_energy_ratio_db(residual_model)?
            - mid_side_energy_ratio_db(residual_source)?)
        .abs()
        .min(240.0),
    }))
}

#[cfg(feature = "onnx")]
fn normalized_correlation(left: &[f64], right: &[f64]) -> f64 {
    let (mut dot, mut left_energy, mut right_energy) = (0.0, 0.0, 0.0);
    for (&left, &right) in left.iter().zip(right) {
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

#[cfg(feature = "onnx")]
fn mid_side_energy_ratio_db(channels: &[Vec<f64>]) -> Result<f64, String> {
    if channels.len() != 2 || channels[0].len() != channels[1].len() {
        return Err("target-sound mid/side geometry is invalid".into());
    }
    let (mut mid_energy, mut side_energy) = (0.0, 0.0);
    for (&left, &right) in channels[0].iter().zip(&channels[1]) {
        let mid = (left + right) * std::f64::consts::FRAC_1_SQRT_2;
        let side = (left - right) * std::f64::consts::FRAC_1_SQRT_2;
        mid_energy += mid * mid;
        side_energy += side * side;
    }
    Ok((10.0 * ((side_energy + 1.0e-24) / (mid_energy + 1.0e-24)).log10()).clamp(-240.0, 240.0))
}

#[cfg(feature = "onnx")]
fn classify_presence(probabilities: [f64; 3], config: &TargetSoundConfig) -> TargetSoundPresence {
    presence_state(
        probabilities,
        config.minimum_absent_probability,
        config.minimum_present_probability,
    )
}

fn presence_state(
    probabilities: [f64; 3],
    minimum_absent_probability: f64,
    minimum_present_probability: f64,
) -> TargetSoundPresence {
    if probabilities[2] >= minimum_present_probability
        && probabilities[2] > probabilities[0]
        && probabilities[2] > probabilities[1]
    {
        TargetSoundPresence::Present
    } else if probabilities[0] >= minimum_absent_probability
        && probabilities[0] > probabilities[1]
        && probabilities[0] > probabilities[2]
    {
        TargetSoundPresence::Absent
    } else {
        TargetSoundPresence::Uncertain
    }
}

#[cfg(feature = "onnx")]
fn validate_audio(audio: &Audio) -> Result<(), String> {
    if !(8_000..=192_000).contains(&audio.sample_rate)
        || !(1..=2).contains(&audio.channels.len())
        || audio.frames() == 0
        || audio.frames() as u64
            > u64::from(audio.sample_rate).saturating_mul(MAX_TARGET_SOUND_AUDIO_SECONDS)
        || audio
            .channels
            .iter()
            .any(|channel| channel.len() != audio.frames())
        || audio
            .channels
            .iter()
            .flatten()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(
            "target-sound input violates its bounded normalized mono/stereo contract".into(),
        );
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn window_starts(frames: usize, window: usize, hop: usize) -> Result<Vec<usize>, String> {
    if frames == 0 || window == 0 || hop == 0 || hop > window {
        return Err("target-sound window geometry is invalid".into());
    }
    let count = if frames <= window {
        1
    } else {
        (frames - window).div_ceil(hop) + 1
    };
    if count > MAX_TARGET_SOUND_WINDOWS {
        return Err("target-sound window count exceeds the bounded limit".into());
    }
    let mut starts = Vec::new();
    starts
        .try_reserve_exact(count)
        .map_err(|_| "unable to reserve target-sound windows".to_string())?;
    for index in 0..count {
        starts.push(index.saturating_mul(hop));
    }
    Ok(starts)
}

#[cfg(feature = "onnx")]
fn allocate_matrix(rows: usize, columns: usize, label: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .map_err(|_| format!("unable to reserve target-sound {label}"))?;
    for _ in 0..rows {
        let mut row = Vec::new();
        row.try_reserve_exact(columns)
            .map_err(|_| format!("unable to reserve target-sound {label}"))?;
        row.resize(columns, 0.0);
        output.push(row);
    }
    Ok(output)
}

#[cfg(feature = "onnx")]
fn allocate_f32_matrix(rows: usize, columns: usize, label: &str) -> Result<Vec<Vec<f32>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .map_err(|_| format!("unable to reserve target-sound {label}"))?;
    for _ in 0..rows {
        let mut row = Vec::new();
        row.try_reserve_exact(columns)
            .map_err(|_| format!("unable to reserve target-sound {label}"))?;
        row.resize(columns, 0.0);
        output.push(row);
    }
    Ok(output)
}

#[cfg(feature = "onnx")]
fn pcm_digest(audio: &Audio, domain: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(audio.sample_rate.to_le_bytes());
    digest.update((audio.channels() as u64).to_le_bytes());
    digest.update((audio.frames() as u64).to_le_bytes());
    for frame in 0..audio.frames() {
        for channel in &audio.channels {
            digest.update(channel[frame].to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

#[cfg(feature = "onnx")]
fn safety_gate(
    kind: TargetSoundSafetyGateKind,
    observed: f64,
    limit: f64,
    passed: bool,
) -> TargetSoundSafetyGate {
    TargetSoundSafetyGate {
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
fn limitations() -> Vec<String> {
    vec![
        "the v1 runtime accepts only a signed finite catalog and never sends natural language to the graph".into(),
        "the calibrated presence head is not an independent acoustic-event detector".into(),
        "source identity, target leakage, and binaural localization quality are promotion-time measurements, not runtime ground truth".into(),
        "a valid evidence signature authenticates the evaluator's claim but cannot prove that recordings, labels, licenses, or listening results are truthful".into(),
        "target and residual are model estimates; only their bounded conservation with the observed mixture is checked at runtime".into(),
        "the offline finite-window adapter makes no causal or real-time latency claim".into(),
        "no checkpoint or upstream dataset is bundled; every package, checkpoint, source, training dataset, evaluation corpus, and license must be supplied and audited separately".into(),
    ]
}

fn metric_policies(kind: TargetSoundStratumKind) -> &'static [MetricPolicy] {
    match kind {
        TargetSoundStratumKind::TargetPresent | TargetSoundStratumKind::ProtectedForeground => {
            PRESENT_METRICS
        }
        TargetSoundStratumKind::TargetAbsent => ABSENT_METRICS,
        TargetSoundStratumKind::BinauralSpatial => BINAURAL_METRICS,
    }
}

fn validate_metric_policy(
    metric: &TargetSoundMetricOutcome,
    policy: &MetricPolicy,
) -> Result<(), String> {
    if !metric.value.is_finite() || !metric.limit.is_finite() {
        return Err("target-sound evidence metric values must be finite".into());
    }
    if metric.operator != policy.operator {
        return Err(format!(
            "target-sound evidence metric {} uses the wrong operator",
            metric.metric
        ));
    }
    let weaker = match policy.operator {
        TargetSoundMetricOperator::GreaterOrEqual => metric.limit < policy.hard_limit,
        TargetSoundMetricOperator::LessOrEqual => metric.limit > policy.hard_limit,
    };
    if weaker {
        return Err(format!(
            "target-sound evidence metric {} declares a weaker limit",
            metric.metric
        ));
    }
    let expected = match metric.operator {
        TargetSoundMetricOperator::GreaterOrEqual => metric.value >= metric.limit,
        TargetSoundMetricOperator::LessOrEqual => metric.value <= metric.limit,
    };
    if metric.passed != expected {
        return Err(format!(
            "target-sound evidence metric {} has an inconsistent passed flag",
            metric.metric
        ));
    }
    Ok(())
}

fn validate_presence(presence: &TargetSoundPresenceAssessment) -> Result<(), String> {
    let values = [
        presence.absent_probability,
        presence.uncertain_probability,
        presence.present_probability,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (values.iter().sum::<f64>() - 1.0).abs() > 0.001
        || !(0.5..=1.0).contains(&presence.minimum_absent_probability)
        || !(0.5..=1.0).contains(&presence.minimum_present_probability)
    {
        return Err("target-sound report presence assessment is invalid".into());
    }
    let expected = presence_state(
        values,
        presence.minimum_absent_probability,
        presence.minimum_present_probability,
    );
    if presence.state != expected {
        return Err("target-sound report presence state conflicts with its probabilities".into());
    }
    Ok(())
}

fn validate_measurements(
    measurements: &TargetSoundSafetyMeasurements,
    channels: usize,
) -> Result<(), String> {
    let required = [
        measurements.input_rms_dbfs,
        measurements.target_rms_dbfs,
        measurements.residual_rms_dbfs,
        measurements.input_peak,
        measurements.target_peak,
        measurements.residual_peak,
        measurements.target_energy_gain_db,
        measurements.residual_energy_gain_db,
        measurements.model_recombination_maximum_absolute_error,
        measurements.publication_recombination_maximum_absolute_error,
    ];
    if required
        .iter()
        .any(|value| !value.is_finite() || !(-240.0..=240.0).contains(value))
        || measurements.input_peak < 0.0
        || measurements.target_peak < 0.0
        || measurements.residual_peak < 0.0
        || measurements.model_recombination_maximum_absolute_error < 0.0
        || measurements.publication_recombination_maximum_absolute_error < 0.0
    {
        return Err("target-sound report safety measurements are invalid".into());
    }
    let spatial = [
        measurements.target_stereo_correlation_delta,
        measurements.target_mid_side_energy_ratio_delta_db,
        measurements.residual_stereo_correlation_delta,
        measurements.residual_mid_side_energy_ratio_delta_db,
    ];
    match channels {
        1 if spatial.iter().any(Option::is_some) => {
            Err("mono target-sound reports must omit spatial measurements".into())
        }
        2 if spatial.iter().any(|value| {
            value.is_none_or(|value| !value.is_finite() || !(0.0..=240.0).contains(&value))
        }) =>
        {
            Err("stereo target-sound reports require bounded spatial measurements".into())
        }
        1 | 2 => Ok(()),
        _ => Err("target-sound report channel count is invalid".into()),
    }
}

fn validate_gates(
    gates: &[TargetSoundSafetyGate],
    channels: usize,
    accepted: bool,
    presence: &TargetSoundPresenceAssessment,
    measurements: &TargetSoundSafetyMeasurements,
) -> Result<(), String> {
    let required = [
        TargetSoundSafetyGateKind::QueryCatalog,
        TargetSoundSafetyGateKind::Geometry,
        TargetSoundSafetyGateKind::FiniteNormalizedSamples,
        TargetSoundSafetyGateKind::ModelRecombination,
        TargetSoundSafetyGateKind::PublishedRecombination,
        TargetSoundSafetyGateKind::TargetPeak,
        TargetSoundSafetyGateKind::ResidualPeak,
        TargetSoundSafetyGateKind::TargetEnergyGain,
        TargetSoundSafetyGateKind::ResidualEnergyGain,
        TargetSoundSafetyGateKind::TargetPresence,
        TargetSoundSafetyGateKind::PromotionEvidence,
    ];
    let stereo = [
        TargetSoundSafetyGateKind::TargetStereoCorrelation,
        TargetSoundSafetyGateKind::TargetMidSideEnergy,
        TargetSoundSafetyGateKind::ResidualStereoCorrelation,
        TargetSoundSafetyGateKind::ResidualMidSideEnergy,
    ];
    let expected_len = required.len() + if channels == 2 { stereo.len() } else { 0 };
    if gates.len() != expected_len {
        return Err("target-sound report has the wrong safety-gate set".into());
    }
    let mut observed = BTreeMap::new();
    for gate in gates {
        if !gate.observed.is_finite()
            || !gate.limit.is_finite()
            || !(-240.0..=240.0).contains(&gate.observed)
            || !(-240.0..=240.0).contains(&gate.limit)
            || !valid_gate_limit(gate.kind, gate.limit)
            || observed.insert(gate.kind, gate).is_some()
        {
            return Err("target-sound report safety gate is invalid or duplicated".into());
        }
    }
    for kind in required
        .into_iter()
        .chain((channels == 2).then_some(stereo).into_iter().flatten())
    {
        if !observed.contains_key(&kind) {
            return Err("target-sound report omits a required safety gate".into());
        }
    }
    for gate in observed.values() {
        let expected_passed = match gate.kind {
            TargetSoundSafetyGateKind::QueryCatalog
            | TargetSoundSafetyGateKind::Geometry
            | TargetSoundSafetyGateKind::FiniteNormalizedSamples
            | TargetSoundSafetyGateKind::PromotionEvidence => {
                gate.observed == 1.0 && gate.limit == 1.0
            }
            TargetSoundSafetyGateKind::TargetPresence => {
                gate.observed == presence.present_probability
                    && gate.limit == presence.minimum_present_probability
                    && presence.state == TargetSoundPresence::Present
            }
            _ => gate.observed <= gate.limit,
        };
        if gate.passed != expected_passed {
            return Err("target-sound report safety-gate result is inconsistent".into());
        }
    }
    let measurement_bindings = [
        (
            TargetSoundSafetyGateKind::ModelRecombination,
            measurements.model_recombination_maximum_absolute_error,
        ),
        (
            TargetSoundSafetyGateKind::PublishedRecombination,
            measurements.publication_recombination_maximum_absolute_error,
        ),
        (
            TargetSoundSafetyGateKind::TargetPeak,
            measurements.target_peak,
        ),
        (
            TargetSoundSafetyGateKind::ResidualPeak,
            measurements.residual_peak,
        ),
        (
            TargetSoundSafetyGateKind::TargetEnergyGain,
            measurements.target_energy_gain_db,
        ),
        (
            TargetSoundSafetyGateKind::ResidualEnergyGain,
            measurements.residual_energy_gain_db,
        ),
    ];
    for (kind, expected_observed) in measurement_bindings {
        if observed.get(&kind).expect("required gate checked").observed != expected_observed {
            return Err("target-sound safety gate is not bound to its measurement".into());
        }
    }
    if channels == 2 {
        for (kind, expected_observed) in [
            (
                TargetSoundSafetyGateKind::TargetStereoCorrelation,
                measurements.target_stereo_correlation_delta,
            ),
            (
                TargetSoundSafetyGateKind::TargetMidSideEnergy,
                measurements.target_mid_side_energy_ratio_delta_db,
            ),
            (
                TargetSoundSafetyGateKind::ResidualStereoCorrelation,
                measurements.residual_stereo_correlation_delta,
            ),
            (
                TargetSoundSafetyGateKind::ResidualMidSideEnergy,
                measurements.residual_mid_side_energy_ratio_delta_db,
            ),
        ] {
            if observed.get(&kind).expect("stereo gate checked").observed
                != expected_observed.expect("stereo measurement checked")
            {
                return Err("target-sound spatial gate is not bound to its measurement".into());
            }
        }
    }
    if accepted != observed.values().all(|gate| gate.passed) {
        return Err("target-sound accepted decision conflicts with safety gates".into());
    }
    Ok(())
}

fn valid_gate_limit(kind: TargetSoundSafetyGateKind, limit: f64) -> bool {
    match kind {
        TargetSoundSafetyGateKind::QueryCatalog
        | TargetSoundSafetyGateKind::Geometry
        | TargetSoundSafetyGateKind::FiniteNormalizedSamples
        | TargetSoundSafetyGateKind::PromotionEvidence => limit == 1.0,
        TargetSoundSafetyGateKind::ModelRecombination => (0.0..=0.10).contains(&limit),
        TargetSoundSafetyGateKind::PublishedRecombination => (0.0..=1.0e-6).contains(&limit),
        TargetSoundSafetyGateKind::TargetPeak | TargetSoundSafetyGateKind::ResidualPeak => {
            (0.5..=1.0).contains(&limit)
        }
        TargetSoundSafetyGateKind::TargetEnergyGain
        | TargetSoundSafetyGateKind::ResidualEnergyGain => (0.0..=12.0).contains(&limit),
        TargetSoundSafetyGateKind::TargetStereoCorrelation
        | TargetSoundSafetyGateKind::ResidualStereoCorrelation => (0.0..=0.25).contains(&limit),
        TargetSoundSafetyGateKind::TargetMidSideEnergy
        | TargetSoundSafetyGateKind::ResidualMidSideEnergy => (0.0..=6.0).contains(&limit),
        TargetSoundSafetyGateKind::TargetPresence => (0.5..=1.0).contains(&limit),
    }
}

fn digest_json<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
    label: &str,
) -> Result<String, String> {
    let document = serde_json::to_vec(value)
        .map_err(|error| format!("serialize {label} for digest: {error}"))?;
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(document);
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_range(label: &str, value: f64, minimum: f64, maximum: f64) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "target-sound {label} must be finite and in {minimum}..={maximum}"
        ));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'+' | b'-'))
    {
        return Err(format!("target-sound {label} is invalid"));
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("target-sound {label} SHA-256 is invalid"));
    }
    Ok(())
}

fn validate_bounded_text(label: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(format!("target-sound {label} is invalid"));
    }
    Ok(())
}

fn is_semver_triplet(value: &str) -> bool {
    let mut parts = value.split('.');
    (0..3).all(|_| {
        parts
            .next()
            .is_some_and(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    }) && parts.next().is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> String {
        "ab".repeat(32)
    }

    fn query() -> TargetSoundQuery {
        TargetSoundQuery {
            schema: TARGET_SOUND_QUERY_SCHEMA.into(),
            schema_version: TARGET_SOUND_SCHEMA_VERSION,
            catalog_revision: "catalog-1".into(),
            classes: vec![
                TargetSoundCatalogClass {
                    id: "alarm".into(),
                    canonical_label: "Alarm".into(),
                },
                TargetSoundCatalogClass {
                    id: "baby-cry".into(),
                    canonical_label: "Baby cry".into(),
                },
            ],
            selected_class_id: "baby-cry".into(),
        }
    }

    fn metric_outcomes(policies: &[MetricPolicy]) -> Vec<TargetSoundMetricOutcome> {
        policies
            .iter()
            .map(|policy| TargetSoundMetricOutcome {
                metric: policy.name.into(),
                value: policy.hard_limit,
                operator: policy.operator,
                limit: policy.hard_limit,
                passed: true,
            })
            .collect()
    }

    fn evidence() -> TargetSoundPromotionEvidencePayload {
        let query = query();
        TargetSoundPromotionEvidencePayload {
            completed_at_unix_seconds: 1,
            model_package_sha256: digest(),
            source_revision: "revision-1".into(),
            source_sha256: digest(),
            checkpoint_sha256: digest(),
            configuration_sha256: TargetSoundConfig::default().digest().unwrap(),
            query_catalog_sha256: query.catalog_sha256().unwrap(),
            query_catalog_revision: query.catalog_revision.clone(),
            query_class_ids_sha256: query.class_ids_sha256().unwrap(),
            query_class_count: query.classes.len() as u32,
            class_coverage_manifest_sha256: digest(),
            evaluated_class_count: query.classes.len() as u32,
            minimum_present_cases_per_class: 20,
            minimum_absent_cases_per_class: 20,
            worst_class_false_positive_rate: 0.01,
            worst_class_false_negative_rate: 0.05,
            artifact_bom_sha256: digest(),
            training_dataset_license_manifest_sha256: digest(),
            evaluation_corpus_manifest_sha256: digest(),
            evaluation_corpus_license_manifest_sha256: digest(),
            evaluation_result_sha256: digest(),
            listening_result_sha256: digest(),
            strata: REQUIRED_STRATA
                .iter()
                .map(|(id, kind)| TargetSoundEvidenceStratum {
                    id: (*id).into(),
                    kind: *kind,
                    cases: 50,
                    metrics: metric_outcomes(metric_policies(*kind)),
                })
                .collect(),
            paired_cases: 1_000,
            target_absent_cases: 200,
            protected_foreground_cases: 200,
            binaural_cases: 200,
            listener_count: 20,
            listener_preference: 0.5,
            redistributed_restricted_artifacts: 0,
            unresolved_artifact_licenses: 0,
            unresolved_training_dataset_licenses: 0,
            unresolved_evaluation_dataset_licenses: 0,
            accepted: true,
        }
    }

    #[test]
    fn query_binds_order_labels_and_selection_without_open_text() {
        let query = query();
        query.validate().unwrap();
        assert_eq!(query.selected_index().unwrap(), 1);
        let catalog = query.catalog_sha256().unwrap();
        let mut reordered = query.clone();
        reordered.classes.swap(0, 1);
        assert_ne!(catalog, reordered.catalog_sha256().unwrap());
        let mut relabeled = query.clone();
        relabeled.classes[0].canonical_label = "Different".into();
        assert_ne!(catalog, relabeled.catalog_sha256().unwrap());
        let mut unknown = query;
        unknown.selected_class_id = "free-form-request".into();
        assert!(unknown.validate().is_err());
    }

    #[test]
    fn evidence_requires_exact_strata_metrics_counts_and_license_clearance() {
        evidence().validate().unwrap();
        let mut invalid = evidence();
        invalid.strata.swap(0, 1);
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.strata[0].metrics[0].limit = -999.0;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.target_absent_cases = 199;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.evaluated_class_count = 3;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.worst_class_false_positive_rate = 0.011;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.unresolved_training_dataset_licenses = 1;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn configuration_is_mode_bound_and_conservative() {
        let preserve = TargetSoundConfig::default();
        preserve.validate().unwrap();
        let mut remove = preserve.clone();
        remove.mode = TargetSoundMode::Remove;
        assert_ne!(preserve.digest().unwrap(), remove.digest().unwrap());
        let mut invalid = preserve;
        invalid.maximum_model_recombination_error = 0.11;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn memory_estimate_charges_rate_channels_and_window() {
        let input = Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; 48_000], vec![0.0; 48_000]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let baseline = estimate_target_sound_memory_bytes(&input, 16_000, 2, 32_000).unwrap();
        let higher_rate = estimate_target_sound_memory_bytes(&input, 48_000, 2, 32_000).unwrap();
        let larger_window =
            estimate_target_sound_memory_bytes(&input, 16_000, 2, 16_777_216).unwrap();
        assert!(higher_rate > baseline);
        assert!(larger_window > baseline);
        assert!(is_semver_triplet("0.89.0"));
        assert!(!is_semver_triplet("v0.89.0"));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn three_state_presence_never_promotes_ambiguous_probabilities() {
        let config = TargetSoundConfig::default();
        assert_eq!(
            classify_presence([0.01, 0.01, 0.98], &config),
            TargetSoundPresence::Present
        );
        assert_eq!(
            classify_presence([0.98, 0.01, 0.01], &config),
            TargetSoundPresence::Absent
        );
        assert_eq!(
            classify_presence([0.40, 0.20, 0.40], &config),
            TargetSoundPresence::Uncertain
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn dbfs_deltas_remain_reportable_at_the_measurement_floor() {
        assert_eq!(bounded_dbfs_delta(0.25, -240.0), 240.0);
        assert_eq!(bounded_dbfs_delta(-240.0, 0.25), -240.0);
        assert_eq!(bounded_dbfs_delta(-10.0, -20.0), 10.0);
    }

    #[test]
    fn report_gate_limits_cannot_weaken_runtime_configuration_bounds() {
        assert!(valid_gate_limit(
            TargetSoundSafetyGateKind::PublishedRecombination,
            1.0e-6
        ));
        assert!(!valid_gate_limit(
            TargetSoundSafetyGateKind::PublishedRecombination,
            1.0
        ));
        assert!(valid_gate_limit(TargetSoundSafetyGateKind::TargetPeak, 0.5));
        assert!(!valid_gate_limit(
            TargetSoundSafetyGateKind::TargetPeak,
            0.49
        ));
        assert!(!valid_gate_limit(
            TargetSoundSafetyGateKind::PromotionEvidence,
            0.0
        ));
    }
}

#[cfg(feature = "onnx")]
impl TargetSoundSession {
    pub fn extract(
        &self,
        input: &Audio,
        query: &TargetSoundQuery,
        config: &TargetSoundConfig,
    ) -> Result<TargetSoundResult, String> {
        config.validate()?;
        query.validate()?;
        validate_audio(input)?;
        let query_catalog_sha256 = query.catalog_sha256()?;
        let query_class_ids_sha256 = query.class_ids_sha256()?;
        if config.digest()? != self.configuration_sha256
            || query_catalog_sha256 != self.catalog_sha256
            || query.catalog_revision != self.catalog_revision
            || query_class_ids_sha256 != self.class_ids_sha256
            || query.classes.len() != self.class_count
        {
            return Err(
                "target-sound query or configuration changed after session preparation".into(),
            );
        }
        let class_index = query.selected_index()?;
        let selected_class = query.selected_class()?;
        let manifest = self
            .package
            .manifest_v2()
            .expect("target-sound session requires v2");
        let model_rate = manifest.runtime.sample_rate_hz;
        let channels = self.model.channels();
        let window = self.model.window_samples();
        let hop = usize::try_from(manifest.latency.hop_samples)
            .map_err(|_| "target-sound hop is too large".to_string())?;
        if input.channels() != channels {
            return Err(format!(
                "target-sound package requires {channels} input channels, got {}",
                input.channels()
            ));
        }
        let model_input =
            crate::resample::resample_channels(&input.channels, input.sample_rate, model_rate)?;
        let model_frames = model_input
            .first()
            .map(Vec::len)
            .ok_or("target-sound resampling produced no channels")?;
        let expected_model_frames =
            crate::resample::planned_output_frames(input.frames(), input.sample_rate, model_rate)?;
        if model_frames == 0
            || model_frames != expected_model_frames
            || model_input
                .iter()
                .any(|channel| channel.len() != model_frames)
        {
            return Err("target-sound resampling produced invalid geometry".into());
        }
        let starts = window_starts(model_frames, window, hop)?;
        let mut target_sum = allocate_matrix(channels, model_frames, "target accumulation")?;
        let mut residual_sum = allocate_matrix(channels, model_frames, "residual accumulation")?;
        let mut weights = vec![0_u32; model_frames];
        let mut probability_sum = [0.0_f64; 3];
        let mut probability_weight = 0.0_f64;
        let mut model_recombination_maximum_absolute_error = 0.0_f64;
        for &start in &starts {
            let mut model_window = allocate_f32_matrix(channels, window, "model window")?;
            for channel in 0..channels {
                let available = model_frames.saturating_sub(start).min(window);
                for offset in 0..available {
                    model_window[channel][offset] = model_input[channel][start + offset] as f32;
                }
            }
            let inference = self.model.process(&model_window, class_index)?;
            let available = model_frames.saturating_sub(start).min(window);
            let window_weight = available as f64;
            for (sum, probability) in probability_sum
                .iter_mut()
                .zip(inference.presence_probabilities)
            {
                *sum += f64::from(probability) * window_weight;
            }
            probability_weight += window_weight;
            for channel in 0..channels {
                for offset in 0..available {
                    let source = model_input[channel][start + offset];
                    let target = f64::from(inference.target[channel][offset]);
                    let residual = f64::from(inference.residual[channel][offset]);
                    model_recombination_maximum_absolute_error =
                        model_recombination_maximum_absolute_error
                            .max((source - target - residual).abs());
                    target_sum[channel][start + offset] += target;
                    residual_sum[channel][start + offset] += residual;
                }
            }
            for weight in &mut weights[start..start + available] {
                *weight = weight.saturating_add(1);
            }
        }
        if weights.contains(&0) {
            return Err("target-sound windows did not cover the model input".into());
        }
        for channel in 0..channels {
            for frame in 0..model_frames {
                let weight = f64::from(weights[frame]);
                target_sum[channel][frame] /= weight;
                residual_sum[channel][frame] /= weight;
            }
        }
        let probabilities = [
            probability_sum[0] / probability_weight,
            probability_sum[1] / probability_weight,
            probability_sum[2] / probability_weight,
        ];
        let presence = classify_presence(probabilities, config);

        let mut target_source =
            crate::resample::resample_channels(&target_sum, model_rate, input.sample_rate)?;
        let expected_source_frames =
            crate::resample::planned_output_frames(model_frames, model_rate, input.sample_rate)?;
        if target_source.len() != channels
            || target_source
                .iter()
                .any(|channel| channel.len() != expected_source_frames)
        {
            return Err("target-sound source-clock resampling produced invalid geometry".into());
        }
        for channel in &mut target_source {
            channel.resize(input.frames(), 0.0);
            channel.truncate(input.frames());
        }
        let mut residual_source = allocate_matrix(channels, input.frames(), "exact residual")?;
        let mut publication_recombination_maximum_absolute_error = 0.0_f64;
        for channel in 0..channels {
            for frame in 0..input.frames() {
                residual_source[channel][frame] =
                    input.channels[channel][frame] - target_source[channel][frame];
                publication_recombination_maximum_absolute_error =
                    publication_recombination_maximum_absolute_error.max(
                        (target_source[channel][frame] + residual_source[channel][frame]
                            - input.channels[channel][frame])
                            .abs(),
                    );
            }
        }
        let input_signal = signal_measurements(&input.channels);
        let target_signal = signal_measurements(&target_source);
        let residual_signal = signal_measurements(&residual_source);
        let target_energy_gain_db =
            bounded_dbfs_delta(target_signal.rms_dbfs, input_signal.rms_dbfs);
        let residual_energy_gain_db =
            bounded_dbfs_delta(residual_signal.rms_dbfs, input_signal.rms_dbfs);
        let finite_normalized = target_source
            .iter()
            .chain(&residual_source)
            .flatten()
            .all(|sample| sample.is_finite() && (-1.0..=1.0).contains(sample));
        let geometry = target_source.len() == channels
            && residual_source.len() == channels
            && target_source
                .iter()
                .all(|channel| channel.len() == input.frames())
            && residual_source
                .iter()
                .all(|channel| channel.len() == input.frames());
        let spatial = spatial_deltas(&target_sum, &residual_sum, &target_source, &residual_source)?;
        let mut gates = vec![
            safety_gate(TargetSoundSafetyGateKind::QueryCatalog, 1.0, 1.0, true),
            safety_gate(
                TargetSoundSafetyGateKind::Geometry,
                bool_value(geometry),
                1.0,
                geometry,
            ),
            safety_gate(
                TargetSoundSafetyGateKind::FiniteNormalizedSamples,
                bool_value(finite_normalized),
                1.0,
                finite_normalized,
            ),
            safety_gate(
                TargetSoundSafetyGateKind::ModelRecombination,
                model_recombination_maximum_absolute_error,
                config.maximum_model_recombination_error,
                model_recombination_maximum_absolute_error
                    <= config.maximum_model_recombination_error,
            ),
            safety_gate(
                TargetSoundSafetyGateKind::PublishedRecombination,
                publication_recombination_maximum_absolute_error,
                config.maximum_publication_recombination_error,
                publication_recombination_maximum_absolute_error
                    <= config.maximum_publication_recombination_error,
            ),
            safety_gate(
                TargetSoundSafetyGateKind::TargetPeak,
                target_signal.peak,
                config.maximum_target_peak,
                target_signal.peak <= config.maximum_target_peak,
            ),
            safety_gate(
                TargetSoundSafetyGateKind::ResidualPeak,
                residual_signal.peak,
                config.maximum_residual_peak,
                residual_signal.peak <= config.maximum_residual_peak,
            ),
            safety_gate(
                TargetSoundSafetyGateKind::TargetEnergyGain,
                target_energy_gain_db,
                config.maximum_energy_gain_db,
                target_energy_gain_db <= config.maximum_energy_gain_db,
            ),
            safety_gate(
                TargetSoundSafetyGateKind::ResidualEnergyGain,
                residual_energy_gain_db,
                config.maximum_energy_gain_db,
                residual_energy_gain_db <= config.maximum_energy_gain_db,
            ),
        ];
        if let Some(spatial) = spatial {
            gates.extend([
                safety_gate(
                    TargetSoundSafetyGateKind::TargetStereoCorrelation,
                    spatial.target_correlation_delta,
                    config.maximum_stereo_correlation_delta,
                    spatial.target_correlation_delta <= config.maximum_stereo_correlation_delta,
                ),
                safety_gate(
                    TargetSoundSafetyGateKind::TargetMidSideEnergy,
                    spatial.target_mid_side_delta_db,
                    config.maximum_mid_side_energy_ratio_delta_db,
                    spatial.target_mid_side_delta_db
                        <= config.maximum_mid_side_energy_ratio_delta_db,
                ),
                safety_gate(
                    TargetSoundSafetyGateKind::ResidualStereoCorrelation,
                    spatial.residual_correlation_delta,
                    config.maximum_stereo_correlation_delta,
                    spatial.residual_correlation_delta <= config.maximum_stereo_correlation_delta,
                ),
                safety_gate(
                    TargetSoundSafetyGateKind::ResidualMidSideEnergy,
                    spatial.residual_mid_side_delta_db,
                    config.maximum_mid_side_energy_ratio_delta_db,
                    spatial.residual_mid_side_delta_db
                        <= config.maximum_mid_side_energy_ratio_delta_db,
                ),
            ]);
        }
        gates.extend([
            safety_gate(
                TargetSoundSafetyGateKind::TargetPresence,
                probabilities[2],
                config.minimum_present_probability,
                presence == TargetSoundPresence::Present,
            ),
            safety_gate(TargetSoundSafetyGateKind::PromotionEvidence, 1.0, 1.0, true),
        ]);
        let signal_gates_passed = gates
            .iter()
            .filter(|gate| gate.kind != TargetSoundSafetyGateKind::TargetPresence)
            .all(|gate| gate.passed);
        let decision = match presence {
            TargetSoundPresence::Absent => TargetSoundDecision::WithheldAbsent,
            TargetSoundPresence::Uncertain => TargetSoundDecision::WithheldUncertain,
            TargetSoundPresence::Present if !signal_gates_passed => {
                TargetSoundDecision::WithheldSafetyGate
            }
            TargetSoundPresence::Present => TargetSoundDecision::AcceptedPresent,
        };
        let accepted = decision == TargetSoundDecision::AcceptedPresent;
        let target_candidate = Audio {
            sample_rate: input.sample_rate,
            channels: target_source,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: input.channel_mask,
        };
        let residual_candidate = Audio {
            sample_rate: input.sample_rate,
            channels: residual_source,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: input.channel_mask,
        };
        let selected_candidate = match config.mode {
            TargetSoundMode::Preserve => target_candidate.clone(),
            TargetSoundMode::Remove => residual_candidate.clone(),
        };
        let target_digest =
            accepted.then(|| pcm_digest(&target_candidate, TARGET_PCM_DIGEST_DOMAIN));
        let residual_digest =
            accepted.then(|| pcm_digest(&residual_candidate, RESIDUAL_PCM_DIGEST_DOMAIN));
        let output_digest =
            accepted.then(|| pcm_digest(&selected_candidate, OUTPUT_PCM_DIGEST_DOMAIN));
        let mut warnings = Vec::new();
        match decision {
            TargetSoundDecision::AcceptedPresent => {}
            TargetSoundDecision::WithheldAbsent => warnings.push(
                "the calibrated presence head classified the selected sound as absent; no audio was published"
                    .into(),
            ),
            TargetSoundDecision::WithheldUncertain => warnings.push(
                "target-sound presence was uncertain; no target, residual, or mixture fallback was published"
                    .into(),
            ),
            TargetSoundDecision::WithheldSafetyGate => {
                let failed = gates
                    .iter()
                    .filter(|gate| !gate.passed)
                    .map(|gate| format!("{:?}", gate.kind).to_ascii_lowercase())
                    .collect::<Vec<_>>()
                    .join(", ");
                warnings.push(format!(
                    "target-sound candidates failed safety gates ({failed}); no audio was published"
                ));
            }
        }
        let profile = self
            .package
            .precision_profile_for(self.accelerator.effective())?
            .expect("target-sound session selects one v2 profile");
        let training_datasets = manifest
            .provenance
            .training_datasets
            .iter()
            .map(|dataset| TargetSoundTrainingDatasetIdentity {
                id: dataset.id.clone(),
                revision: dataset.revision.clone(),
                sha256: dataset.sha256.clone(),
                license_spdx: dataset.license_spdx.clone(),
            })
            .collect();
        let report = TargetSoundReport {
            schema: TARGET_SOUND_REPORT_SCHEMA.into(),
            schema_version: TARGET_SOUND_SCHEMA_VERSION,
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            configuration_sha256: self.configuration_sha256.clone(),
            mode: config.mode,
            network_accessed: false,
            deterministic: true,
            closed_class_query: true,
            model_invoked: true,
            query: TargetSoundQueryIdentity {
                query_sha256: query.digest()?,
                catalog_sha256: self.catalog_sha256.clone(),
                catalog_revision: self.catalog_revision.clone(),
                class_ids_sha256: self.class_ids_sha256.clone(),
                class_count: self.class_count as u32,
                class_id: selected_class.id.clone(),
                class_index: class_index as u32,
                canonical_label: selected_class.canonical_label.clone(),
                encoding: "one-hot-v1".into(),
                open_text_accepted: false,
            },
            model: TargetSoundModelIdentity {
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
                training_datasets,
                accelerator: self.accelerator.effective().name().into(),
            },
            promotion_evidence: self.evidence.clone(),
            decision,
            candidate_accepted: accepted,
            target_published: accepted,
            residual_published: accepted,
            output_published: accepted,
            candidates_retained: accepted,
            source_sample_rate: input.sample_rate,
            source_channels: input.channels(),
            source_frames: input.frames(),
            model_sample_rate: model_rate,
            model_channels: channels,
            model_window_samples: window,
            model_hop_samples: hop,
            model_windows: starts.len(),
            input_pcm_sha256: pcm_digest(input, INPUT_PCM_DIGEST_DOMAIN),
            target_pcm_sha256: target_digest,
            residual_pcm_sha256: residual_digest,
            output_pcm_sha256: output_digest,
            presence: TargetSoundPresenceAssessment {
                state: presence,
                absent_probability: probabilities[0],
                uncertain_probability: probabilities[1],
                present_probability: probabilities[2],
                minimum_absent_probability: config.minimum_absent_probability,
                minimum_present_probability: config.minimum_present_probability,
            },
            measurements: TargetSoundSafetyMeasurements {
                input_rms_dbfs: input_signal.rms_dbfs,
                target_rms_dbfs: target_signal.rms_dbfs,
                residual_rms_dbfs: residual_signal.rms_dbfs,
                input_peak: input_signal.peak,
                target_peak: target_signal.peak,
                residual_peak: residual_signal.peak,
                target_energy_gain_db,
                residual_energy_gain_db,
                model_recombination_maximum_absolute_error,
                publication_recombination_maximum_absolute_error,
                target_stereo_correlation_delta: spatial
                    .map(|value| value.target_correlation_delta),
                target_mid_side_energy_ratio_delta_db: spatial
                    .map(|value| value.target_mid_side_delta_db),
                residual_stereo_correlation_delta: spatial
                    .map(|value| value.residual_correlation_delta),
                residual_mid_side_energy_ratio_delta_db: spatial
                    .map(|value| value.residual_mid_side_delta_db),
            },
            safety_gates: gates,
            path_fields_recorded: 0,
            limitations: limitations(),
            warnings,
        };
        report.validate()?;
        Ok(TargetSoundResult {
            target: accepted.then_some(target_candidate),
            residual: accepted.then_some(residual_candidate),
            output: accepted.then_some(selected_candidate),
            report,
        })
    }
}
