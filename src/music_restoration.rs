//! Bounded candidate restoration for complete mono or stereo music programs.
//!
//! This operation repairs one finished mixture. It does not estimate stems,
//! undo artistic mastering intent, or claim that a generated high-frequency
//! component is recovered ground truth. Uncertain model regions preserve the
//! input, and every applied change is exposed as an exact correction residual.

use crate::audio::Audio;
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
#[cfg(feature = "onnx")]
use crate::{
    AcceleratorPreference, AcceleratorSelection, Backend, BackendOptions, RuntimeModelPackage,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::Path;

pub const MUSIC_RESTORATION_EVIDENCE_SCHEMA: &str =
    "denoize-music-restoration-promotion-evidence-v1";
pub const MUSIC_RESTORATION_REPORT_SCHEMA: &str = "denoize-music-restoration-report-v1";
pub const MUSIC_RESTORATION_SCHEMA_VERSION: u32 = 1;

const MAX_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_AUDIO_SECONDS: u64 = 6 * 60 * 60;
const MAX_WINDOWS: usize = 500_000;
#[cfg(feature = "onnx")]
const MAX_WORKING_BYTES: u128 = 2 * 1024 * 1024 * 1024;
const JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const EVIDENCE_SIGNATURE_DOMAIN: &[u8] = b"denoize-music-restoration-promotion-evidence-v1";
const CONFIG_DIGEST_DOMAIN: &[u8] = b"denoize-music-restoration-config-v1\0";
#[cfg(feature = "onnx")]
const INPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-music-restoration-input-pcm-v1\0";
#[cfg(feature = "onnx")]
const OUTPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-music-restoration-output-pcm-v1\0";
#[cfg(feature = "onnx")]
const CORRECTION_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-music-restoration-correction-pcm-v1\0";

const REQUIRED_STRATA: [&str; 12] = [
    "aac-64k",
    "clean-bypass",
    "genre-unseen",
    "long-form",
    "mono",
    "mp3-64k",
    "neural-codec",
    "percussion-transients",
    "phase-critical",
    "stereo-image",
    "unseen-codec",
    "wideband-reference",
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MusicRestorationTask {
    CodecRepair,
    BandwidthExtension,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRestorationConfig {
    pub task: MusicRestorationTask,
    pub minimum_apply_probability: f64,
    pub minimum_bypass_probability: f64,
    pub minimum_apply_frames: usize,
    pub maximum_output_peak: f64,
    pub maximum_absolute_correction: f64,
    pub maximum_stereo_correlation_delta: f64,
    pub maximum_mid_side_energy_ratio_delta_db: f64,
}

impl Default for MusicRestorationConfig {
    fn default() -> Self {
        Self {
            task: MusicRestorationTask::CodecRepair,
            minimum_apply_probability: 0.80,
            minimum_bypass_probability: 0.80,
            minimum_apply_frames: 2,
            maximum_output_peak: 1.0,
            maximum_absolute_correction: 0.50,
            maximum_stereo_correlation_delta: 0.05,
            maximum_mid_side_energy_ratio_delta_db: 1.5,
        }
    }
}

impl MusicRestorationConfig {
    pub fn validate(&self) -> Result<(), String> {
        validate_range(
            "minimum_apply_probability",
            self.minimum_apply_probability,
            0.5,
            1.0,
        )?;
        validate_range(
            "minimum_bypass_probability",
            self.minimum_bypass_probability,
            0.5,
            1.0,
        )?;
        if !(1..=100).contains(&self.minimum_apply_frames) {
            return Err("music-restoration minimum_apply_frames must be in 1..=100".into());
        }
        validate_range("maximum_output_peak", self.maximum_output_peak, 0.5, 1.0)?;
        validate_range(
            "maximum_absolute_correction",
            self.maximum_absolute_correction,
            0.01,
            1.0,
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
        let document = serde_json::to_vec(self)
            .map_err(|error| format!("serialize music-restoration configuration: {error}"))?;
        let mut digest = Sha256::new();
        digest.update(CONFIG_DIGEST_DOMAIN);
        digest.update(document);
        Ok(format!("{:x}", digest.finalize()))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRestorationEvidenceStratum {
    pub id: String,
    pub cases: u64,
    pub multi_mel_snr_improvement_db: f64,
    pub zimtohrli_regression: f64,
    pub fad_clap_regression: f64,
    pub low_band_snr_db: f64,
    pub transient_loss_rate: f64,
    pub stereo_correlation_error: f64,
    pub phase_error_radians: f64,
    pub duration_mismatch_samples: u64,
    pub clipped_samples: u64,
    pub non_finite_samples: u64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRestorationPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub task: MusicRestorationTask,
    pub model_package_sha256: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub checkpoint_sha256: String,
    pub configuration_sha256: String,
    pub artifact_bom_sha256: String,
    pub training_dataset_license_manifest_sha256: String,
    pub evaluation_corpus_manifest_sha256: String,
    pub evaluation_corpus_license_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub listening_result_sha256: String,
    pub strata: Vec<MusicRestorationEvidenceStratum>,
    pub paired_clips: u64,
    pub full_length_tracks: u64,
    pub instrument_classes: u64,
    pub genres: u64,
    pub clean_bypass_cases: u64,
    pub mono_cases: u64,
    pub stereo_cases: u64,
    pub listener_count: u64,
    pub listener_preference: f64,
    pub redistributed_restricted_artifacts: u64,
    pub accepted: bool,
}

impl MusicRestorationPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        if self.completed_at_unix_seconds > JSON_SAFE_INTEGER {
            return Err("music-restoration evidence timestamp exceeds JSON safe integer".into());
        }
        validate_identifier("source revision", &self.source_revision)?;
        for (label, value) in [
            ("model package", self.model_package_sha256.as_str()),
            ("source", self.source_sha256.as_str()),
            ("checkpoint", self.checkpoint_sha256.as_str()),
            ("configuration", self.configuration_sha256.as_str()),
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
        if self.strata.len() != REQUIRED_STRATA.len() {
            return Err(format!(
                "music-restoration evidence requires exactly {} strata",
                REQUIRED_STRATA.len()
            ));
        }
        let mut all_passed = true;
        for (index, stratum) in self.strata.iter().enumerate() {
            if stratum.id != REQUIRED_STRATA[index] {
                return Err("music-restoration evidence strata must be exact and sorted".into());
            }
            if !(10..=1_000_000).contains(&stratum.cases)
                || stratum.duration_mismatch_samples > JSON_SAFE_INTEGER
                || stratum.clipped_samples > JSON_SAFE_INTEGER
                || stratum.non_finite_samples > JSON_SAFE_INTEGER
            {
                return Err("music-restoration stratum counts are outside bounded limits".into());
            }
            let finite = [
                stratum.multi_mel_snr_improvement_db,
                stratum.zimtohrli_regression,
                stratum.fad_clap_regression,
                stratum.low_band_snr_db,
                stratum.transient_loss_rate,
                stratum.stereo_correlation_error,
                stratum.phase_error_radians,
            ]
            .iter()
            .all(|value| value.is_finite());
            let expected = finite
                && (0.0..=240.0).contains(&stratum.multi_mel_snr_improvement_db)
                && (-1.0..=0.01).contains(&stratum.zimtohrli_regression)
                && (-1_000.0..=0.02).contains(&stratum.fad_clap_regression)
                && (40.0..=240.0).contains(&stratum.low_band_snr_db)
                && (0.0..=0.02).contains(&stratum.transient_loss_rate)
                && (0.0..=0.02).contains(&stratum.stereo_correlation_error)
                && (0.0..=0.20).contains(&stratum.phase_error_radians)
                && stratum.duration_mismatch_samples == 0
                && stratum.clipped_samples == 0
                && stratum.non_finite_samples == 0;
            if stratum.passed != expected {
                return Err(format!(
                    "music-restoration stratum {} has inconsistent promotion status",
                    stratum.id
                ));
            }
            all_passed &= stratum.passed;
        }
        if !(1_000..=10_000_000).contains(&self.paired_clips)
            || !(50..=1_000_000).contains(&self.full_length_tracks)
            || !(8..=1_000).contains(&self.instrument_classes)
            || !(8..=10_000).contains(&self.genres)
            || !(100..=1_000_000).contains(&self.clean_bypass_cases)
            || !(100..=1_000_000).contains(&self.mono_cases)
            || !(100..=1_000_000).contains(&self.stereo_cases)
            || !(20..=100_000).contains(&self.listener_count)
            || !self.listener_preference.is_finite()
            || !(0.5..=1.0).contains(&self.listener_preference)
            || self.redistributed_restricted_artifacts != 0
        {
            return Err(
                "music-restoration global promotion evidence is outside hard limits".into(),
            );
        }
        let expected_accepted = all_passed
            && self.listener_preference >= 0.5
            && self.redistributed_restricted_artifacts == 0;
        if self.accepted != expected_accepted {
            return Err("music-restoration accepted flag is inconsistent".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedMusicRestorationPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: MusicRestorationPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedMusicRestorationPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) =
            crate::input::open_regular_file(path, "music-restoration promotion evidence")?;
        if length >= MAX_EVIDENCE_BYTES {
            return Err(format!(
                "music-restoration promotion evidence {} exceeds {MAX_EVIDENCE_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve music-restoration evidence JSON".to_string())?;
        file.take(MAX_EVIDENCE_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read music-restoration promotion evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("music-restoration promotion evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse music-restoration promotion evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != MUSIC_RESTORATION_EVIDENCE_SCHEMA
            || self.schema_version != MUSIC_RESTORATION_SCHEMA_VERSION
        {
            return Err("unsupported music-restoration promotion evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("music-restoration promotion evidence must use ed25519".into());
        }
        validate_sha256("evidence key ID", &self.signature.key_id)
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize music-restoration evidence: {error}"))?;
        key.verify_domain_document(
            EVIDENCE_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "music-restoration promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize music-restoration evidence: {error}"))
    }
}

pub fn sign_music_restoration_promotion_evidence(
    payload: MusicRestorationPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedMusicRestorationPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize music-restoration evidence: {error}"))?;
    let signature = key.sign_domain_document(
        EVIDENCE_SIGNATURE_DOMAIN,
        &document,
        "music-restoration promotion evidence",
    )?;
    let evidence = SignedMusicRestorationPromotionEvidence {
        schema: MUSIC_RESTORATION_EVIDENCE_SCHEMA.into(),
        schema_version: MUSIC_RESTORATION_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MusicRestorationDecision {
    Apply,
    Uncertain,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRestorationRegion {
    pub start_sample: u64,
    pub end_sample: u64,
    pub decision: MusicRestorationDecision,
    pub confidence: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRestorationTrainingDatasetIdentity {
    pub id: String,
    pub revision: String,
    pub sha256: Option<String>,
    pub license_spdx: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRestorationModelIdentity {
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
    pub training_datasets: Vec<MusicRestorationTrainingDatasetIdentity>,
    pub accelerator: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRestorationEvidenceIdentity {
    pub signing_key_id: String,
    pub artifact_bom_sha256: String,
    pub training_dataset_license_manifest_sha256: String,
    pub evaluation_corpus_manifest_sha256: String,
    pub evaluation_corpus_license_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub listening_result_sha256: String,
    pub paired_clips: u64,
    pub full_length_tracks: u64,
    pub listener_count: u64,
    pub accepted: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRestorationReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub task: MusicRestorationTask,
    pub configuration_sha256: String,
    pub network_accessed: bool,
    pub deterministic: bool,
    pub candidate_render: bool,
    pub recovered_ground_truth_claimed: bool,
    pub dry_stems_produced: bool,
    pub creative_mastering_applied: bool,
    pub model: MusicRestorationModelIdentity,
    pub promotion_evidence: MusicRestorationEvidenceIdentity,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_frames: usize,
    pub output_sample_rate: u32,
    pub output_channels: usize,
    pub output_frames: usize,
    pub model_sample_rate: u32,
    pub model_channels: usize,
    pub model_window_samples: usize,
    pub model_hop_samples: usize,
    pub model_state_frames_per_window: usize,
    pub model_windows: usize,
    pub decision_frames: usize,
    pub applied_decision_frames: usize,
    pub bypassed_decision_frames: usize,
    pub uncertain_decision_frames: usize,
    pub applied_source_samples: u64,
    pub changed_samples: u64,
    pub regions: Vec<MusicRestorationRegion>,
    pub input_pcm_sha256: String,
    pub output_pcm_sha256: String,
    pub correction_pcm_sha256: String,
    pub correction_recombination_maximum_absolute_error: f64,
    pub maximum_absolute_correction: f64,
    pub maximum_output_peak: f64,
    pub input_stereo_correlation: Option<f64>,
    pub output_stereo_correlation: Option<f64>,
    pub stereo_correlation_delta: Option<f64>,
    pub input_mid_side_energy_ratio_db: Option<f64>,
    pub output_mid_side_energy_ratio_db: Option<f64>,
    pub mid_side_energy_ratio_delta_db: Option<f64>,
    pub exact_output_geometry: bool,
    pub path_fields_recorded: u64,
    pub limitations: Vec<String>,
}

impl MusicRestorationReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MUSIC_RESTORATION_REPORT_SCHEMA
            || self.schema_version != MUSIC_RESTORATION_SCHEMA_VERSION
            || !is_semver_triplet(&self.denoize_version)
            || self.network_accessed
            || !self.deterministic
            || !self.candidate_render
            || self.recovered_ground_truth_claimed
            || self.dry_stems_produced
            || self.creative_mastering_applied
        {
            return Err("unsupported music-restoration report header".into());
        }
        for (label, value) in [
            ("configuration", self.configuration_sha256.as_str()),
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
            ("output PCM", self.output_pcm_sha256.as_str()),
            ("correction PCM", self.correction_pcm_sha256.as_str()),
        ] {
            validate_sha256(label, value)?;
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
            return Err("music-restoration report has too many training datasets".into());
        }
        let mut dataset_ids = BTreeSet::new();
        for dataset in &self.model.training_datasets {
            validate_bounded_text("training dataset ID", &dataset.id, 256)?;
            validate_bounded_text("training dataset revision", &dataset.revision, 512)?;
            validate_bounded_text("training dataset license", &dataset.license_spdx, 512)?;
            if !dataset_ids.insert(dataset.id.as_str()) {
                return Err("music-restoration training dataset IDs must be unique".into());
            }
            if let Some(sha256) = &dataset.sha256 {
                validate_sha256("training dataset", sha256)?;
            }
        }
        let model_frames = crate::resample::planned_output_frames(
            self.source_frames,
            self.source_sample_rate,
            self.model_sample_rate,
        )?;
        let state_hop = self
            .model_window_samples
            .checked_div(self.model_state_frames_per_window.max(1))
            .ok_or_else(|| "music-restoration report state clock overflow".to_string())?;
        let expected_decisions = model_frames.div_ceil(state_hop.max(1));
        let expected_windows = window_count(
            model_frames,
            self.model_window_samples,
            self.model_hop_samples,
        )?;
        if !self.promotion_evidence.accepted
            || !(1_000..=10_000_000).contains(&self.promotion_evidence.paired_clips)
            || !(50..=1_000_000).contains(&self.promotion_evidence.full_length_tracks)
            || !(20..=100_000).contains(&self.promotion_evidence.listener_count)
            || !matches!(self.model.accelerator.as_str(), "cpu" | "metal" | "cuda")
            || !(8_000..=192_000).contains(&self.source_sample_rate)
            || !(1..=2).contains(&self.source_channels)
            || self.source_frames == 0
            || self.source_frames as u64
                > u64::from(self.source_sample_rate).saturating_mul(MAX_AUDIO_SECONDS)
            || self.output_sample_rate != self.source_sample_rate
            || self.output_channels != self.source_channels
            || self.output_frames != self.source_frames
            || !(8_000..=192_000).contains(&self.model_sample_rate)
            || self.model_channels != self.source_channels
            || self.model_window_samples < 256
            || self.model_window_samples > 16_777_216
            || self.model_hop_samples == 0
            || self.model_hop_samples > self.model_window_samples
            || self.model_state_frames_per_window == 0
            || self.model_state_frames_per_window > self.model_window_samples
            || !self
                .model_window_samples
                .is_multiple_of(self.model_state_frames_per_window)
            || !self.model_hop_samples.is_multiple_of(state_hop.max(1))
            || self.model_windows != expected_windows
            || self.decision_frames != expected_decisions
            || self
                .applied_decision_frames
                .checked_add(self.bypassed_decision_frames)
                .and_then(|value| value.checked_add(self.uncertain_decision_frames))
                != Some(self.decision_frames)
            || self.applied_source_samples > self.source_frames as u64
            || self.changed_samples
                > (self.source_frames as u64).saturating_mul(self.source_channels as u64)
            || !self
                .correction_recombination_maximum_absolute_error
                .is_finite()
            || !(0.0..=1.0e-12).contains(&self.correction_recombination_maximum_absolute_error)
            || !self.maximum_absolute_correction.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_absolute_correction)
            || !self.maximum_output_peak.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_output_peak)
            || !self.exact_output_geometry
            || self.path_fields_recorded != 0
            || self.regions.len() > MAX_WINDOWS
            || self.limitations.is_empty()
            || self.limitations.len() > 32
        {
            return Err("music-restoration report violates bounded result contracts".into());
        }
        validate_regions(&self.regions, self.source_frames)?;
        for limitation in &self.limitations {
            validate_bounded_text("music-restoration limitation", limitation, 512)?;
        }
        match self.source_channels {
            1 => {
                if self.input_stereo_correlation.is_some()
                    || self.output_stereo_correlation.is_some()
                    || self.stereo_correlation_delta.is_some()
                    || self.input_mid_side_energy_ratio_db.is_some()
                    || self.output_mid_side_energy_ratio_db.is_some()
                    || self.mid_side_energy_ratio_delta_db.is_some()
                {
                    return Err("mono music-restoration reports must omit stereo metrics".into());
                }
            }
            2 => {
                let correlations = [
                    self.input_stereo_correlation,
                    self.output_stereo_correlation,
                    self.stereo_correlation_delta,
                ];
                let ratios = [
                    self.input_mid_side_energy_ratio_db,
                    self.output_mid_side_energy_ratio_db,
                    self.mid_side_energy_ratio_delta_db,
                ];
                if correlations
                    .iter()
                    .any(|value| value.is_none_or(|value| !value.is_finite()))
                    || ratios
                        .iter()
                        .any(|value| value.is_none_or(|value| !value.is_finite()))
                    || !(-1.0..=1.0).contains(&self.input_stereo_correlation.unwrap())
                    || !(-1.0..=1.0).contains(&self.output_stereo_correlation.unwrap())
                    || !(0.0..=2.0).contains(&self.stereo_correlation_delta.unwrap())
                    || !(-240.0..=240.0).contains(&self.input_mid_side_energy_ratio_db.unwrap())
                    || !(-240.0..=240.0).contains(&self.output_mid_side_energy_ratio_db.unwrap())
                    || !(0.0..=480.0).contains(&self.mid_side_energy_ratio_delta_db.unwrap())
                {
                    return Err("stereo music-restoration metrics are invalid".into());
                }
            }
            _ => unreachable!("source channel count was validated"),
        }
        Ok(())
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize music-restoration report: {error}"))
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize music-restoration report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct MusicRestorationResult {
    pub output: Audio,
    pub correction: Audio,
    pub report: MusicRestorationReport,
}

pub fn estimate_music_restoration_memory_bytes(
    input: &Audio,
    model_sample_rate: u32,
    model_channels: usize,
    model_window_samples: usize,
) -> Result<u64, String> {
    if input.sample_rate == 0
        || input.channels.is_empty()
        || !(1..=2).contains(&model_channels)
        || !(256..=16_777_216).contains(&model_window_samples)
    {
        return Err("music-restoration memory geometry is invalid".into());
    }
    let model_frames = crate::resample::planned_output_frames(
        input.frames(),
        input.sample_rate,
        model_sample_rate,
    )?;
    let model_scalars_per_frame = (model_channels as u128)
        .checked_mul(7)
        .and_then(|value| value.checked_add(12))
        .ok_or_else(|| "music-restoration memory estimate overflow".to_string())?;
    let source_scalars_per_frame = (input.channels() as u128)
        .checked_mul(5)
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| "music-restoration memory estimate overflow".to_string())?;
    let scalar_bytes = (model_frames as u128)
        .checked_mul(model_scalars_per_frame)
        .and_then(|value| {
            value.checked_add((input.frames() as u128).saturating_mul(source_scalars_per_frame))
        })
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() as u128))
        .ok_or_else(|| "music-restoration memory estimate overflow".to_string())?;
    let window_bytes = (model_window_samples as u128)
        .checked_mul(
            (model_channels as u128)
                .saturating_mul(6)
                .saturating_add(12),
        )
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() as u128))
        .ok_or_else(|| "music-restoration window memory estimate overflow".to_string())?;
    let input_resampler = crate::resample::resampler_plan_bytes(
        model_channels,
        input.sample_rate,
        model_sample_rate,
    )?;
    let correction_resampler = crate::resample::resampler_plan_bytes(
        model_channels,
        model_sample_rate,
        input.sample_rate,
    )?;
    let bytes = scalar_bytes
        .checked_add(window_bytes)
        .and_then(|value| {
            value.checked_add(crate::audio::estimate_audio_memory_bytes(input) as u128)
        })
        .and_then(|value| value.checked_add(u128::from(input_resampler.max(correction_resampler))))
        .ok_or_else(|| "music-restoration memory estimate overflow".to_string())?;
    u64::try_from(bytes).map_err(|_| "music-restoration memory estimate exceeds u64".to_string())
}

#[cfg(feature = "onnx")]
pub struct MusicRestorationSession {
    package: RuntimeModelPackage,
    model: crate::backend::music_restoration::MusicRestorationModel,
    accelerator: AcceleratorSelection,
    evidence: MusicRestorationEvidenceIdentity,
    configuration_sha256: String,
    task: MusicRestorationTask,
}

#[cfg(feature = "onnx")]
impl std::fmt::Debug for MusicRestorationSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MusicRestorationSession")
            .field("package_sha256", &self.package.package_sha256())
            .field("model", &self.model)
            .field("accelerator", &self.accelerator)
            .field("task", &self.task)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "onnx")]
impl MusicRestorationSession {
    pub fn prepare(
        package: RuntimeModelPackage,
        evidence: &SignedMusicRestorationPromotionEvidence,
        evidence_key: &ReceiptPublicKey,
        config: &MusicRestorationConfig,
        requested: AcceleratorPreference,
    ) -> Result<Self, String> {
        config.validate()?;
        evidence.verify_signature(evidence_key)?;
        if !evidence.payload.accepted {
            return Err(
                "music-restoration evidence is authentic but does not pass promotion gates".into(),
            );
        }
        if evidence.payload.task != config.task {
            return Err("music-restoration evidence task does not match the requested task".into());
        }
        let manifest = package
            .manifest_v2()
            .ok_or("music restoration rejects runtime model package v1")?;
        let configuration_sha256 = config.digest()?;
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
        ] {
            if observed != expected {
                return Err(format!(
                    "music-restoration evidence {label} does not match the authenticated package/configuration"
                ));
            }
        }
        let mut options = BackendOptions::default().with_runtime_model_package(package.clone());
        options.deterministic = true;
        options.accelerator = requested;
        let accelerator = crate::select_accelerator_for_options(Backend::Onnx, &options)?;
        if !package.supports_accelerator(accelerator.effective()) {
            return Err(format!(
                "music-restoration package does not permit the {} accelerator",
                accelerator.effective().name()
            ));
        }
        let model = crate::backend::music_restoration::MusicRestorationModel::load_runtime_package(
            &package,
            accelerator.effective(),
        )?;
        let payload = &evidence.payload;
        Ok(Self {
            package,
            model,
            accelerator,
            evidence: MusicRestorationEvidenceIdentity {
                signing_key_id: evidence.signature.key_id.clone(),
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
                paired_clips: payload.paired_clips,
                full_length_tracks: payload.full_length_tracks,
                listener_count: payload.listener_count,
                accepted: true,
            },
            configuration_sha256,
            task: config.task,
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
            .expect("music-restoration packages use v2 precision profiles");
        Ok(profile
            .resources
            .max_session_memory_bytes
            .saturating_add(profile.resources.max_worker_memory_bytes))
    }

    pub fn processing_working_set_bytes(&self, input: &Audio) -> Result<u64, String> {
        let manifest = self
            .package
            .manifest_v2()
            .expect("music-restoration session requires v2");
        estimate_music_restoration_memory_bytes(
            input,
            manifest.runtime.sample_rate_hz,
            self.model.channels(),
            self.model.window_samples(),
        )
    }

    pub fn restore(
        &self,
        input: &Audio,
        config: &MusicRestorationConfig,
    ) -> Result<MusicRestorationResult, String> {
        config.validate()?;
        validate_audio(input)?;
        if config.digest()? != self.configuration_sha256 || config.task != self.task {
            return Err(
                "music-restoration configuration changed after evidence-bound preparation".into(),
            );
        }
        if input.channels() != self.model.channels() {
            return Err(format!(
                "music-restoration package requires {} program channels, got {}",
                self.model.channels(),
                input.channels()
            ));
        }
        if u128::from(self.processing_working_set_bytes(input)?) > MAX_WORKING_BYTES {
            return Err("music-restoration working set exceeds the 2-GiB hard limit".into());
        }
        let manifest = self
            .package
            .manifest_v2()
            .expect("music-restoration session requires v2");
        let model_rate = manifest.runtime.sample_rate_hz;
        let model_input =
            crate::resample::resample_channels(&input.channels, input.sample_rate, model_rate)?;
        let model_frames = model_input.first().map_or(0, Vec::len);
        if model_frames == 0 {
            return Err("music-restoration input becomes empty at the model rate".into());
        }
        let window = self.model.window_samples();
        let hop = usize::try_from(manifest.latency.hop_samples)
            .map_err(|_| "music-restoration hop is too large".to_string())?;
        let state_frames_per_window = self.model.state_frames();
        let state_hop = window / state_frames_per_window;
        let starts = window_starts(model_frames, window, hop)?;
        let decision_frames = model_frames.div_ceil(state_hop);
        let channels = self.model.channels();
        let mut delta_sum = allocate_matrix(channels, model_frames, "candidate correction")?;
        let mut delta_weight = vec![0_u32; model_frames];
        let mut state_sum = vec![[0.0_f64; 3]; decision_frames];
        let mut state_weight = vec![0_u32; decision_frames];

        for &start in &starts {
            let mut block = allocate_f32_matrix(channels, window, "model input window")?;
            let available = window.min(model_frames - start);
            for (destination, source) in block.iter_mut().zip(&model_input) {
                for (output, sample) in destination[..available]
                    .iter_mut()
                    .zip(&source[start..start + available])
                {
                    *output = *sample as f32;
                }
            }
            let inference = self.model.process(&block)?;
            for offset in 0..available {
                let target = start + offset;
                delta_weight[target] = delta_weight[target].saturating_add(1);
                for channel in 0..channels {
                    let delta = f64::from(inference.candidate[channel][offset])
                        - model_input[channel][target];
                    if !delta.is_finite() || delta.abs() > config.maximum_absolute_correction {
                        return Err(
                            "music-restoration candidate exceeds the authenticated correction limit"
                                .into(),
                        );
                    }
                    delta_sum[channel][target] += delta;
                }
            }
            for frame in 0..state_frames_per_window {
                let sample = start.saturating_add(frame.saturating_mul(state_hop));
                if sample >= model_frames {
                    break;
                }
                let global = sample / state_hop;
                for class in 0..3 {
                    state_sum[global][class] +=
                        f64::from(inference.state_probabilities[frame][class]);
                }
                state_weight[global] = state_weight[global].saturating_add(1);
            }
        }
        if delta_weight.iter().any(|weight| *weight == 0)
            || state_weight.iter().any(|weight| *weight == 0)
        {
            return Err("music-restoration windows did not cover the authenticated clocks".into());
        }
        for (values, weight) in state_sum.iter_mut().zip(&state_weight) {
            for value in values {
                *value /= f64::from(*weight);
            }
        }
        let mut decisions = classify_decisions(&state_sum, config);
        enforce_minimum_apply_run(&mut decisions, config.minimum_apply_frames);
        let mut correction_model = allocate_matrix(channels, model_frames, "gated correction")?;
        for sample in 0..model_frames {
            if decisions[(sample / state_hop).min(decisions.len() - 1)] == InternalDecision::Apply {
                let weight = f64::from(delta_weight[sample]);
                for channel in 0..channels {
                    correction_model[channel][sample] = delta_sum[channel][sample] / weight;
                }
            }
        }
        let applied_decision_frames = decisions
            .iter()
            .filter(|decision| **decision == InternalDecision::Apply)
            .count();
        let bypassed_decision_frames = decisions
            .iter()
            .filter(|decision| **decision == InternalDecision::Bypass)
            .count();
        let uncertain_decision_frames =
            decisions.len() - applied_decision_frames - bypassed_decision_frames;

        let mut correction_source = if applied_decision_frames == 0 {
            allocate_matrix(channels, input.frames(), "zero source correction")?
        } else {
            crate::resample::resample_channels(&correction_model, model_rate, input.sample_rate)?
        };
        for channel in &mut correction_source {
            channel.resize(input.frames(), 0.0);
            channel.truncate(input.frames());
        }
        let mut output_channels = allocate_matrix(channels, input.frames(), "restored output")?;
        let mut maximum_absolute_correction = 0.0_f64;
        let mut maximum_output_peak = 0.0_f64;
        let mut changed_samples = 0_u64;
        for channel in 0..channels {
            for frame in 0..input.frames() {
                let correction = correction_source[channel][frame];
                if !correction.is_finite()
                    || correction.abs() > config.maximum_absolute_correction + 1.0e-12
                {
                    return Err(
                        "music-restoration resampled correction exceeds its hard limit".into(),
                    );
                }
                let value = input.channels[channel][frame] + correction;
                if !value.is_finite() || value.abs() > config.maximum_output_peak {
                    return Err(
                        "music-restoration candidate violates the finite output peak gate".into(),
                    );
                }
                output_channels[channel][frame] = value;
                maximum_absolute_correction = maximum_absolute_correction.max(correction.abs());
                maximum_output_peak = maximum_output_peak.max(value.abs());
                if correction.abs() > 1.0e-12 {
                    changed_samples = changed_samples.saturating_add(1);
                }
            }
        }
        let output = Audio {
            sample_rate: input.sample_rate,
            channels: output_channels,
            bits_per_sample: input.bits_per_sample,
            sample_format: input.sample_format,
            channel_mask: input.channel_mask,
        };
        let mut exact_correction = allocate_matrix(channels, input.frames(), "exact correction")?;
        let mut correction_recombination_maximum_absolute_error = 0.0_f64;
        for channel in 0..channels {
            for frame in 0..input.frames() {
                exact_correction[channel][frame] =
                    output.channels[channel][frame] - input.channels[channel][frame];
                let recombined = input.channels[channel][frame] + exact_correction[channel][frame];
                correction_recombination_maximum_absolute_error =
                    correction_recombination_maximum_absolute_error
                        .max((recombined - output.channels[channel][frame]).abs());
            }
        }
        if correction_recombination_maximum_absolute_error > 1.0e-12 {
            return Err("music-restoration correction does not exactly recombine".into());
        }
        let correction = Audio {
            sample_rate: input.sample_rate,
            channels: exact_correction,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: input.channel_mask,
        };
        let stereo = stereo_metrics(input, &output)?;
        if let Some(metrics) = stereo {
            if metrics.correlation_delta > config.maximum_stereo_correlation_delta
                || metrics.mid_side_ratio_delta_db > config.maximum_mid_side_energy_ratio_delta_db
            {
                return Err(
                    "music-restoration candidate violates stereo preservation gates".into(),
                );
            }
        }
        let regions = build_regions(
            &decisions,
            &state_sum,
            state_hop,
            model_rate,
            input.sample_rate,
            input.frames(),
        )?;
        let applied_source_samples = regions
            .iter()
            .filter(|region| region.decision == MusicRestorationDecision::Apply)
            .map(|region| region.end_sample - region.start_sample)
            .sum();
        let profile = self
            .package
            .precision_profile_for(self.accelerator.effective())?
            .expect("music-restoration session selects one v2 profile");
        let training_datasets = manifest
            .provenance
            .training_datasets
            .iter()
            .map(|dataset| MusicRestorationTrainingDatasetIdentity {
                id: dataset.id.clone(),
                revision: dataset.revision.clone(),
                sha256: dataset.sha256.clone(),
                license_spdx: dataset.license_spdx.clone(),
            })
            .collect();
        let report = MusicRestorationReport {
            schema: MUSIC_RESTORATION_REPORT_SCHEMA.into(),
            schema_version: MUSIC_RESTORATION_SCHEMA_VERSION,
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            task: config.task,
            configuration_sha256: self.configuration_sha256.clone(),
            network_accessed: false,
            deterministic: true,
            candidate_render: true,
            recovered_ground_truth_claimed: false,
            dry_stems_produced: false,
            creative_mastering_applied: false,
            model: MusicRestorationModelIdentity {
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
            source_sample_rate: input.sample_rate,
            source_channels: input.channels(),
            source_frames: input.frames(),
            output_sample_rate: output.sample_rate,
            output_channels: output.channels(),
            output_frames: output.frames(),
            model_sample_rate: model_rate,
            model_channels: channels,
            model_window_samples: window,
            model_hop_samples: hop,
            model_state_frames_per_window: state_frames_per_window,
            model_windows: starts.len(),
            decision_frames: decisions.len(),
            applied_decision_frames,
            bypassed_decision_frames,
            uncertain_decision_frames,
            applied_source_samples,
            changed_samples,
            regions,
            input_pcm_sha256: pcm_digest(input, INPUT_PCM_DIGEST_DOMAIN),
            output_pcm_sha256: pcm_digest(&output, OUTPUT_PCM_DIGEST_DOMAIN),
            correction_pcm_sha256: pcm_digest(&correction, CORRECTION_PCM_DIGEST_DOMAIN),
            correction_recombination_maximum_absolute_error,
            maximum_absolute_correction,
            maximum_output_peak,
            input_stereo_correlation: stereo.map(|metrics| metrics.input_correlation),
            output_stereo_correlation: stereo.map(|metrics| metrics.output_correlation),
            stereo_correlation_delta: stereo.map(|metrics| metrics.correlation_delta),
            input_mid_side_energy_ratio_db: stereo.map(|metrics| metrics.input_mid_side_ratio_db),
            output_mid_side_energy_ratio_db: stereo
                .map(|metrics| metrics.output_mid_side_ratio_db),
            mid_side_energy_ratio_delta_db: stereo
                .map(|metrics| metrics.mid_side_ratio_delta_db),
            exact_output_geometry: output.sample_rate == input.sample_rate
                && output.channels() == input.channels()
                && output.frames() == input.frames(),
            path_fields_recorded: 0,
            limitations: vec![
                "the output is a model-generated candidate render, not recovered ground truth".into(),
                "the operation preserves one complete mono or stereo mixture and never produces dry stems".into(),
                "uncertain and confidently clean decision frames receive no model correction".into(),
                "codec repair and bandwidth extension cannot prove the original missing phase or spectrum".into(),
                "creative EQ, compression, stereo widening, and mastering are outside this operation".into(),
                "the correction residual is required to expose every applied sample-domain change".into(),
                "no checkpoint is bundled; the exact artifact BOM, licenses, and evaluation evidence must be supplied".into(),
            ],
        };
        report.validate()?;
        Ok(MusicRestorationResult {
            output,
            correction,
            report,
        })
    }
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InternalDecision {
    Apply,
    Bypass,
    Uncertain,
}

#[cfg(feature = "onnx")]
fn classify_decisions(
    probabilities: &[[f64; 3]],
    config: &MusicRestorationConfig,
) -> Vec<InternalDecision> {
    probabilities
        .iter()
        .map(|values| {
            if values[2] >= config.minimum_apply_probability
                && values[2] > values[0]
                && values[2] > values[1]
            {
                InternalDecision::Apply
            } else if values[0] >= config.minimum_bypass_probability
                && values[0] > values[1]
                && values[0] > values[2]
            {
                InternalDecision::Bypass
            } else {
                InternalDecision::Uncertain
            }
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn enforce_minimum_apply_run(decisions: &mut [InternalDecision], minimum: usize) {
    let mut start = 0usize;
    while start < decisions.len() {
        if decisions[start] != InternalDecision::Apply {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < decisions.len() && decisions[end] == InternalDecision::Apply {
            end += 1;
        }
        if end - start < minimum {
            decisions[start..end].fill(InternalDecision::Uncertain);
        }
        start = end;
    }
}

#[cfg(feature = "onnx")]
fn build_regions(
    decisions: &[InternalDecision],
    probabilities: &[[f64; 3]],
    state_hop: usize,
    model_rate: u32,
    source_rate: u32,
    source_frames: usize,
) -> Result<Vec<MusicRestorationRegion>, String> {
    let mut regions = Vec::new();
    let mut start = 0usize;
    while start < decisions.len() {
        let state = decisions[start];
        if state == InternalDecision::Bypass {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < decisions.len() && decisions[end] == state {
            end += 1;
        }
        let confidence_index = if state == InternalDecision::Apply {
            2
        } else {
            1
        };
        let confidence = probabilities[start..end]
            .iter()
            .map(|values| values[confidence_index])
            .sum::<f64>()
            / (end - start) as f64;
        regions.push(MusicRestorationRegion {
            start_sample: map_sample(start.saturating_mul(state_hop), model_rate, source_rate),
            end_sample: map_sample(end.saturating_mul(state_hop), model_rate, source_rate)
                .min(source_frames as u64),
            decision: if state == InternalDecision::Apply {
                MusicRestorationDecision::Apply
            } else {
                MusicRestorationDecision::Uncertain
            },
            confidence,
        });
        if regions.len() > MAX_WINDOWS {
            return Err("music-restoration region count exceeds the bounded limit".into());
        }
        start = end;
    }
    regions.retain(|region| region.start_sample < region.end_sample);
    Ok(regions)
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy)]
struct StereoMetrics {
    input_correlation: f64,
    output_correlation: f64,
    correlation_delta: f64,
    input_mid_side_ratio_db: f64,
    output_mid_side_ratio_db: f64,
    mid_side_ratio_delta_db: f64,
}

#[cfg(feature = "onnx")]
fn stereo_metrics(input: &Audio, output: &Audio) -> Result<Option<StereoMetrics>, String> {
    if input.channels() == 1 {
        return Ok(None);
    }
    if input.channels() != 2 || output.channels() != 2 || input.frames() != output.frames() {
        return Err("music-restoration stereo metric geometry is invalid".into());
    }
    let input_correlation = normalized_correlation(&input.channels[0], &input.channels[1]);
    let output_correlation = normalized_correlation(&output.channels[0], &output.channels[1]);
    let input_mid_side_ratio_db = mid_side_energy_ratio_db(input)?;
    let output_mid_side_ratio_db = mid_side_energy_ratio_db(output)?;
    Ok(Some(StereoMetrics {
        input_correlation,
        output_correlation,
        correlation_delta: (output_correlation - input_correlation).abs(),
        input_mid_side_ratio_db,
        output_mid_side_ratio_db,
        mid_side_ratio_delta_db: (output_mid_side_ratio_db - input_mid_side_ratio_db).abs(),
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
fn mid_side_energy_ratio_db(audio: &Audio) -> Result<f64, String> {
    if audio.channels() != 2 {
        return Err("mid/side energy requires stereo audio".into());
    }
    let (mut mid_energy, mut side_energy) = (0.0, 0.0);
    for (&left, &right) in audio.channels[0].iter().zip(&audio.channels[1]) {
        let mid = (left + right) * std::f64::consts::FRAC_1_SQRT_2;
        let side = (left - right) * std::f64::consts::FRAC_1_SQRT_2;
        mid_energy += mid * mid;
        side_energy += side * side;
    }
    Ok((10.0 * ((side_energy + 1.0e-24) / (mid_energy + 1.0e-24)).log10()).clamp(-240.0, 240.0))
}

#[cfg(feature = "onnx")]
fn validate_audio(audio: &Audio) -> Result<(), String> {
    if !(8_000..=192_000).contains(&audio.sample_rate)
        || !(1..=2).contains(&audio.channels.len())
        || audio.frames() == 0
        || audio.frames() as u64 > u64::from(audio.sample_rate).saturating_mul(MAX_AUDIO_SECONDS)
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
            "music-restoration input violates its bounded normalized mono/stereo contract".into(),
        );
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn window_starts(frames: usize, window: usize, hop: usize) -> Result<Vec<usize>, String> {
    let count = window_count(frames, window, hop)?;
    let mut starts = Vec::new();
    starts
        .try_reserve_exact(count)
        .map_err(|_| "unable to reserve music-restoration windows".to_string())?;
    for index in 0..count {
        starts.push(index.saturating_mul(hop));
    }
    Ok(starts)
}

fn window_count(frames: usize, window: usize, hop: usize) -> Result<usize, String> {
    if frames == 0 || window == 0 || hop == 0 || hop > window {
        return Err("music-restoration window geometry is invalid".into());
    }
    let count = if frames <= window {
        1
    } else {
        (frames - window).div_ceil(hop) + 1
    };
    if count > MAX_WINDOWS {
        return Err("music-restoration window count exceeds the bounded limit".into());
    }
    Ok(count)
}

#[cfg(feature = "onnx")]
fn allocate_matrix(rows: usize, columns: usize, label: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .map_err(|_| format!("unable to reserve music-restoration {label}"))?;
    for _ in 0..rows {
        let mut row = Vec::new();
        row.try_reserve_exact(columns)
            .map_err(|_| format!("unable to reserve music-restoration {label}"))?;
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
        .map_err(|_| format!("unable to reserve music-restoration {label}"))?;
    for _ in 0..rows {
        let mut row = Vec::new();
        row.try_reserve_exact(columns)
            .map_err(|_| format!("unable to reserve music-restoration {label}"))?;
        row.resize(columns, 0.0);
        output.push(row);
    }
    Ok(output)
}

#[cfg(feature = "onnx")]
fn map_sample(sample: usize, from_rate: u32, to_rate: u32) -> u64 {
    (sample as u128)
        .saturating_mul(u128::from(to_rate))
        .saturating_add(u128::from(from_rate) / 2)
        .checked_div(u128::from(from_rate))
        .unwrap_or(0)
        .min(u128::from(u64::MAX)) as u64
}

fn validate_regions(regions: &[MusicRestorationRegion], frames: usize) -> Result<(), String> {
    if regions.len() > MAX_WINDOWS {
        return Err("music-restoration region count exceeds the bounded limit".into());
    }
    let mut previous_end = 0u64;
    for region in regions {
        if region.start_sample < previous_end
            || region.start_sample >= region.end_sample
            || region.end_sample > frames as u64
            || !region.confidence.is_finite()
            || !(0.0..=1.0).contains(&region.confidence)
        {
            return Err("music-restoration region geometry is invalid".into());
        }
        previous_end = region.end_sample;
    }
    Ok(())
}

fn validate_range(label: &str, value: f64, minimum: f64, maximum: f64) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "music-restoration {label} must be finite and in {minimum}..={maximum}"
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
        return Err(format!("music-restoration {label} is invalid"));
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("music-restoration {label} SHA-256 is invalid"));
    }
    Ok(())
}

fn validate_bounded_text(label: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(format!("music-restoration {label} is invalid"));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> String {
        "ab".repeat(32)
    }

    fn evidence() -> MusicRestorationPromotionEvidencePayload {
        MusicRestorationPromotionEvidencePayload {
            completed_at_unix_seconds: 1,
            task: MusicRestorationTask::CodecRepair,
            model_package_sha256: digest(),
            source_revision: "revision-1".into(),
            source_sha256: digest(),
            checkpoint_sha256: digest(),
            configuration_sha256: MusicRestorationConfig::default().digest().unwrap(),
            artifact_bom_sha256: digest(),
            training_dataset_license_manifest_sha256: digest(),
            evaluation_corpus_manifest_sha256: digest(),
            evaluation_corpus_license_manifest_sha256: digest(),
            evaluation_result_sha256: digest(),
            listening_result_sha256: digest(),
            strata: REQUIRED_STRATA
                .iter()
                .map(|id| MusicRestorationEvidenceStratum {
                    id: (*id).into(),
                    cases: 100,
                    multi_mel_snr_improvement_db: 1.0,
                    zimtohrli_regression: 0.0,
                    fad_clap_regression: 0.0,
                    low_band_snr_db: 60.0,
                    transient_loss_rate: 0.01,
                    stereo_correlation_error: 0.01,
                    phase_error_radians: 0.10,
                    duration_mismatch_samples: 0,
                    clipped_samples: 0,
                    non_finite_samples: 0,
                    passed: true,
                })
                .collect(),
            paired_clips: 1_000,
            full_length_tracks: 50,
            instrument_classes: 8,
            genres: 8,
            clean_bypass_cases: 100,
            mono_cases: 100,
            stereo_cases: 100,
            listener_count: 20,
            listener_preference: 0.5,
            redistributed_restricted_artifacts: 0,
            accepted: true,
        }
    }

    #[test]
    fn evidence_requires_exact_strata_fidelity_and_license_gates() {
        evidence().validate().unwrap();
        let mut invalid = evidence();
        invalid.strata.swap(0, 1);
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.strata[0].transient_loss_rate = 0.03;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.redistributed_restricted_artifacts = 1;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.source_sha256 = "AB".repeat(32);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn configuration_is_task_bound_and_conservative() {
        let codec = MusicRestorationConfig::default();
        codec.validate().unwrap();
        let mut bandwidth = codec.clone();
        bandwidth.task = MusicRestorationTask::BandwidthExtension;
        assert_ne!(codec.digest().unwrap(), bandwidth.digest().unwrap());
        let mut invalid = codec;
        invalid.maximum_absolute_correction = 1.1;
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
        let baseline = estimate_music_restoration_memory_bytes(&input, 16_000, 2, 32_000).unwrap();
        let higher_rate =
            estimate_music_restoration_memory_bytes(&input, 48_000, 2, 32_000).unwrap();
        let larger_window =
            estimate_music_restoration_memory_bytes(&input, 16_000, 2, 16_777_216).unwrap();
        assert!(higher_rate > baseline);
        assert!(larger_window > baseline);
        assert!(is_semver_triplet("0.88.0"));
        assert!(!is_semver_triplet("v0.88.0"));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn short_apply_runs_become_uncertain() {
        let config = MusicRestorationConfig::default();
        let probabilities = [
            [0.05, 0.05, 0.90],
            [0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90],
            [0.05, 0.05, 0.90],
        ];
        let mut decisions = classify_decisions(&probabilities, &config);
        enforce_minimum_apply_run(&mut decisions, 2);
        assert_eq!(
            decisions,
            vec![
                InternalDecision::Uncertain,
                InternalDecision::Bypass,
                InternalDecision::Apply,
                InternalDecision::Apply,
            ]
        );
    }
}
