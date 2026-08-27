//! Bounded continuous speech separation and anonymous meeting diarization.
//!
//! The default output is an anonymous, capped track set plus an unassigned
//! residual that exactly reconstructs the mono reference.  No speaker
//! embedding or enrollment audio is stored. Optional labels can be attached
//! only through an explicit Stage 29 consent receipt; this module never
//! invents an identity from diarization.

use crate::audio::Audio;
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
#[cfg(feature = "onnx")]
use crate::{
    AcceleratorPreference, AcceleratorSelection, Backend, BackendOptions, RuntimeModelPackage,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
#[cfg(feature = "onnx")]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::Path;

pub const MEETING_SPEAKER_EVIDENCE_SCHEMA: &str = "denoize-meeting-speaker-promotion-evidence-v1";
pub const MEETING_SPEAKER_REPORT_SCHEMA: &str = "denoize-meeting-speaker-report-v1";
pub const MEETING_TRACK_LABELS_SCHEMA: &str = "denoize-meeting-track-labels-v1";
pub const MEETING_SPEAKER_SCHEMA_VERSION: u32 = 1;
pub const MAX_MEETING_SPEAKER_TRACKS: usize = 8;

const MAX_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MEETING_SECONDS: u64 = 1_800;
const MAX_SEGMENTS: usize = 250_000;
#[cfg(feature = "onnx")]
const MAX_WORKING_BYTES: u128 = 2 * 1024 * 1024 * 1024;
const JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const EVIDENCE_SIGNATURE_DOMAIN: &[u8] = b"denoize-meeting-speaker-promotion-evidence-v1";
const CONFIG_DIGEST_DOMAIN: &[u8] = b"denoize-meeting-speaker-config-v1\0";
#[cfg(feature = "onnx")]
const INPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-meeting-speaker-input-pcm-v1\0";
#[cfg(feature = "onnx")]
const TRACK_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-meeting-speaker-track-pcm-v1\0";
#[cfg(feature = "onnx")]
const RESIDUAL_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-meeting-speaker-residual-pcm-v1\0";

const REQUIRED_STRATA: [&str; 12] = [
    "array-available",
    "cross-talk",
    "far-field",
    "four-plus-speakers",
    "language-switch",
    "long-meeting",
    "overlap",
    "real-meeting",
    "single-channel",
    "speaker-count",
    "unknown-speech",
    "unseen-room",
];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingSpeakerConfig {
    pub minimum_active_probability: f64,
    pub minimum_inactive_probability: f64,
    pub minimum_unknown_probability: f64,
    pub minimum_active_frames: usize,
    pub permutation_minimum_correlation: f64,
    pub permutation_minimum_margin: f64,
    pub maximum_track_peak: f64,
    pub maximum_residual_peak: f64,
}

impl Default for MeetingSpeakerConfig {
    fn default() -> Self {
        Self {
            minimum_active_probability: 0.80,
            minimum_inactive_probability: 0.80,
            minimum_unknown_probability: 0.80,
            minimum_active_frames: 2,
            permutation_minimum_correlation: 0.20,
            permutation_minimum_margin: 0.05,
            maximum_track_peak: 1.0,
            maximum_residual_peak: 1.0,
        }
    }
}

impl MeetingSpeakerConfig {
    pub fn validate(&self) -> Result<(), String> {
        validate_range(
            "minimum_active_probability",
            self.minimum_active_probability,
            0.5,
            1.0,
        )?;
        validate_range(
            "minimum_inactive_probability",
            self.minimum_inactive_probability,
            0.5,
            1.0,
        )?;
        validate_range(
            "minimum_unknown_probability",
            self.minimum_unknown_probability,
            0.5,
            1.0,
        )?;
        if !(1..=100).contains(&self.minimum_active_frames) {
            return Err("meeting-speaker minimum_active_frames must be in 1..=100".into());
        }
        validate_range(
            "permutation_minimum_correlation",
            self.permutation_minimum_correlation,
            0.0,
            1.0,
        )?;
        validate_range(
            "permutation_minimum_margin",
            self.permutation_minimum_margin,
            0.0,
            1.0,
        )?;
        validate_range("maximum_track_peak", self.maximum_track_peak, 0.5, 1.0)?;
        validate_range(
            "maximum_residual_peak",
            self.maximum_residual_peak,
            0.5,
            1.0,
        )
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        let document = serde_json::to_vec(self)
            .map_err(|error| format!("serialize meeting-speaker configuration: {error}"))?;
        let mut digest = Sha256::new();
        digest.update(CONFIG_DIGEST_DOMAIN);
        digest.update(document);
        Ok(format!("{:x}", digest.finalize()))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingSpeakerEvidenceStratum {
    pub id: String,
    pub cases: u64,
    pub permutation_si_sdr_improvement_db: f64,
    pub diarization_error_rate: f64,
    pub jaccard_error_rate: f64,
    pub overlap_f1: f64,
    pub track_swap_rate: f64,
    pub tcp_wer_regression: f64,
    pub unknown_false_assignment_rate: f64,
    pub non_finite_samples: u64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingSpeakerPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub model_package_sha256: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub checkpoint_sha256: String,
    pub configuration_sha256: String,
    pub corpus_manifest_sha256: String,
    pub corpus_license_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub listening_result_sha256: String,
    pub strata: Vec<MeetingSpeakerEvidenceStratum>,
    pub real_meeting_cases: u64,
    pub distinct_speakers: u64,
    pub language_count: u64,
    pub speaker_count_expected_calibration_error: f64,
    pub listener_count: u64,
    pub listener_preference: f64,
    pub retained_enrollment_recordings: u64,
    pub retained_speaker_embeddings: u64,
    pub accepted: bool,
}

impl MeetingSpeakerPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        if self.completed_at_unix_seconds > JSON_SAFE_INTEGER {
            return Err("meeting-speaker evidence timestamp exceeds JSON safe integer".into());
        }
        validate_identifier("source revision", &self.source_revision)?;
        for (label, value) in [
            ("model package", self.model_package_sha256.as_str()),
            ("source", self.source_sha256.as_str()),
            ("checkpoint", self.checkpoint_sha256.as_str()),
            ("configuration", self.configuration_sha256.as_str()),
            ("corpus manifest", self.corpus_manifest_sha256.as_str()),
            (
                "corpus license manifest",
                self.corpus_license_manifest_sha256.as_str(),
            ),
            ("evaluation result", self.evaluation_result_sha256.as_str()),
            ("listening result", self.listening_result_sha256.as_str()),
        ] {
            validate_sha256(label, value)?;
        }
        if self.strata.len() != REQUIRED_STRATA.len() {
            return Err(format!(
                "meeting-speaker evidence requires exactly {} strata",
                REQUIRED_STRATA.len()
            ));
        }
        let mut all_passed = true;
        for (index, stratum) in self.strata.iter().enumerate() {
            if stratum.id != REQUIRED_STRATA[index] {
                return Err("meeting-speaker evidence strata must be exact and sorted".into());
            }
            if !(10..=1_000_000).contains(&stratum.cases)
                || stratum.non_finite_samples > JSON_SAFE_INTEGER
            {
                return Err("meeting-speaker stratum counts are outside bounded limits".into());
            }
            let finite = [
                stratum.permutation_si_sdr_improvement_db,
                stratum.diarization_error_rate,
                stratum.jaccard_error_rate,
                stratum.overlap_f1,
                stratum.track_swap_rate,
                stratum.tcp_wer_regression,
                stratum.unknown_false_assignment_rate,
            ]
            .iter()
            .all(|value| value.is_finite());
            let expected = finite
                && (0.0..=240.0).contains(&stratum.permutation_si_sdr_improvement_db)
                && (0.0..=0.30).contains(&stratum.diarization_error_rate)
                && (0.0..=0.40).contains(&stratum.jaccard_error_rate)
                && (0.60..=1.0).contains(&stratum.overlap_f1)
                && (0.0..=0.02).contains(&stratum.track_swap_rate)
                && (-1.0..=0.02).contains(&stratum.tcp_wer_regression)
                && (0.0..=0.01).contains(&stratum.unknown_false_assignment_rate)
                && stratum.non_finite_samples == 0;
            if stratum.passed != expected {
                return Err(format!(
                    "meeting-speaker stratum {} has inconsistent promotion status",
                    stratum.id
                ));
            }
            all_passed &= stratum.passed;
        }
        if !(100..=1_000_000).contains(&self.real_meeting_cases)
            || !(100..=1_000_000).contains(&self.distinct_speakers)
            || !(2..=1_000).contains(&self.language_count)
            || !self.speaker_count_expected_calibration_error.is_finite()
            || !(0.0..=0.05).contains(&self.speaker_count_expected_calibration_error)
            || !(20..=100_000).contains(&self.listener_count)
            || !self.listener_preference.is_finite()
            || !(0.5..=1.0).contains(&self.listener_preference)
            || self.retained_enrollment_recordings != 0
            || self.retained_speaker_embeddings != 0
        {
            return Err("meeting-speaker global promotion evidence is outside hard limits".into());
        }
        let expected_accepted = all_passed && self.listener_preference >= 0.5;
        if self.accepted != expected_accepted {
            return Err("meeting-speaker accepted flag is inconsistent".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedMeetingSpeakerPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: MeetingSpeakerPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedMeetingSpeakerPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) =
            crate::input::open_regular_file(path, "meeting-speaker promotion evidence")?;
        if length >= MAX_EVIDENCE_BYTES {
            return Err(format!(
                "meeting-speaker promotion evidence {} exceeds {MAX_EVIDENCE_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve meeting-speaker evidence JSON".to_string())?;
        file.take(MAX_EVIDENCE_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read meeting-speaker promotion evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("meeting-speaker promotion evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse meeting-speaker promotion evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != MEETING_SPEAKER_EVIDENCE_SCHEMA
            || self.schema_version != MEETING_SPEAKER_SCHEMA_VERSION
        {
            return Err("unsupported meeting-speaker promotion evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("meeting-speaker promotion evidence must use ed25519".into());
        }
        validate_sha256("evidence key ID", &self.signature.key_id)
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize meeting-speaker evidence: {error}"))?;
        key.verify_domain_document(
            EVIDENCE_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "meeting-speaker promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize meeting-speaker evidence: {error}"))
    }
}

pub fn sign_meeting_speaker_promotion_evidence(
    payload: MeetingSpeakerPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedMeetingSpeakerPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize meeting-speaker evidence: {error}"))?;
    let signature = key.sign_domain_document(
        EVIDENCE_SIGNATURE_DOMAIN,
        &document,
        "meeting-speaker promotion evidence",
    )?;
    let evidence = SignedMeetingSpeakerPromotionEvidence {
        schema: MEETING_SPEAKER_EVIDENCE_SCHEMA.into(),
        schema_version: MEETING_SPEAKER_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeetingActivityState {
    Active,
    Uncertain,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingSpeakerSegment {
    pub start_sample: u64,
    pub end_sample: u64,
    pub state: MeetingActivityState,
    pub confidence: f64,
    pub overlap: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingRegion {
    pub start_sample: u64,
    pub end_sample: u64,
    pub confidence: f64,
}

/// Explicit Stage 29 handoff. The hashes bind consent and an accepted target-
/// speaker report; raw enrollment and embeddings remain outside this record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingTrackLabel {
    pub track_id: String,
    pub label: String,
    pub consent_record_sha256: String,
    pub target_speaker_report_sha256: String,
    pub raw_enrollment_retained: bool,
    pub speaker_embedding_retained: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingTrackLabelsDocument {
    pub schema: String,
    pub schema_version: u32,
    pub labels: Vec<MeetingTrackLabel>,
}

impl MeetingTrackLabelsDocument {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) = crate::input::open_regular_file(path, "meeting track labels")?;
        if length >= 1024 * 1024 {
            return Err(format!(
                "meeting track labels {} exceed 1048576 bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve meeting track labels".to_string())?;
        file.take(1024 * 1024)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read meeting track labels: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("meeting track labels changed while reading".into());
        }
        let document: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse meeting track labels: {error}"))?;
        document.validate()?;
        Ok(document)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MEETING_TRACK_LABELS_SCHEMA
            || self.schema_version != MEETING_SPEAKER_SCHEMA_VERSION
        {
            return Err("unsupported meeting track-label schema".into());
        }
        validate_labels(&self.labels, MAX_MEETING_SPEAKER_TRACKS)
    }
}

impl MeetingTrackLabel {
    fn validate(&self) -> Result<(), String> {
        validate_track_id(&self.track_id)?;
        if self.label.is_empty()
            || self.label.len() > 128
            || self.label.chars().any(char::is_control)
        {
            return Err("meeting track label must be 1..=128 printable bytes".into());
        }
        validate_sha256("consent record", &self.consent_record_sha256)?;
        validate_sha256("target-speaker report", &self.target_speaker_report_sha256)?;
        if self.raw_enrollment_retained || self.speaker_embedding_retained {
            return Err(
                "meeting track labels reject retained enrollment recordings or embeddings".into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingSpeakerModelIdentity {
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingSpeakerEvidenceIdentity {
    pub signing_key_id: String,
    pub corpus_manifest_sha256: String,
    pub corpus_license_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub listening_result_sha256: String,
    pub real_meeting_cases: u64,
    pub distinct_speakers: u64,
    pub languages: u64,
    pub accepted: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingSpeakerTrackSummary {
    pub id: String,
    pub label: Option<String>,
    pub pcm_sha256: String,
    pub active_samples: u64,
    pub uncertain_samples: u64,
    pub segments: Vec<MeetingSpeakerSegment>,
    pub consent_record_sha256: Option<String>,
    pub target_speaker_report_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeetingSpeakerReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub configuration_sha256: String,
    pub network_accessed: bool,
    pub deterministic: bool,
    pub model: MeetingSpeakerModelIdentity,
    pub promotion_evidence: MeetingSpeakerEvidenceIdentity,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_frames: usize,
    pub model_sample_rate: u32,
    pub model_input_channels: usize,
    pub model_window_samples: usize,
    pub model_hop_samples: usize,
    pub model_activity_frames: usize,
    pub model_windows: usize,
    pub maximum_tracks: usize,
    pub published_tracks: usize,
    pub track_summaries: Vec<MeetingSpeakerTrackSummary>,
    pub unknown_regions: Vec<MeetingRegion>,
    pub overlap_regions: Vec<MeetingRegion>,
    pub permutation_ambiguous_windows: usize,
    pub mixture_pcm_sha256: String,
    pub unassigned_pcm_sha256: String,
    pub recombination_maximum_absolute_error: f64,
    pub exact_output_duration: bool,
    pub raw_enrollment_retained: bool,
    pub speaker_embeddings_retained: bool,
    pub path_fields_recorded: u64,
    pub limitations: Vec<String>,
}

impl MeetingSpeakerReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MEETING_SPEAKER_REPORT_SCHEMA
            || self.schema_version != MEETING_SPEAKER_SCHEMA_VERSION
            || !is_semver_triplet(&self.denoize_version)
            || self.network_accessed
            || !self.deterministic
        {
            return Err("unsupported meeting-speaker report header".into());
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
                "evidence corpus",
                self.promotion_evidence.corpus_manifest_sha256.as_str(),
            ),
            (
                "evidence corpus license",
                self.promotion_evidence
                    .corpus_license_manifest_sha256
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
            ("mixture PCM", self.mixture_pcm_sha256.as_str()),
            ("unassigned PCM", self.unassigned_pcm_sha256.as_str()),
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
        if !self.promotion_evidence.accepted
            || !(100..=1_000_000).contains(&self.promotion_evidence.real_meeting_cases)
            || !(100..=1_000_000).contains(&self.promotion_evidence.distinct_speakers)
            || !(2..=1_000).contains(&self.promotion_evidence.languages)
            || !matches!(self.model.accelerator.as_str(), "cpu" | "metal" | "cuda")
            || !(8_000..=192_000).contains(&self.source_sample_rate)
            || self.source_channels == 0
            || self.source_channels > 64
            || self.source_frames == 0
            || self.source_frames as u64
                > u64::from(self.source_sample_rate).saturating_mul(MAX_MEETING_SECONDS)
            || !(8_000..=192_000).contains(&self.model_sample_rate)
            || self.model_input_channels == 0
            || self.model_input_channels > 64
            || self.model_window_samples < 256
            || self.model_window_samples > 16_777_216
            || self.model_hop_samples == 0
            || self.model_hop_samples > self.model_window_samples
            || self.model_activity_frames == 0
            || self.model_activity_frames > self.model_window_samples
            || !self
                .model_window_samples
                .is_multiple_of(self.model_activity_frames)
            || !self
                .model_hop_samples
                .is_multiple_of(self.model_window_samples / self.model_activity_frames.max(1))
            || self.model_windows == 0
            || self.model_windows > MAX_SEGMENTS
            || self.permutation_ambiguous_windows > self.model_windows.saturating_sub(1)
            || !(1..=MAX_MEETING_SPEAKER_TRACKS).contains(&self.maximum_tracks)
            || self.published_tracks == 0
            || self.published_tracks > self.maximum_tracks
            || self.track_summaries.len() != self.published_tracks
            || !self.exact_output_duration
            || !self.recombination_maximum_absolute_error.is_finite()
            || self.recombination_maximum_absolute_error < 0.0
            || self.recombination_maximum_absolute_error > 1.0e-12
            || self.raw_enrollment_retained
            || self.speaker_embeddings_retained
            || self.path_fields_recorded != 0
        {
            return Err("meeting-speaker report violates closed invariants".into());
        }
        if self.model_input_channels != 1 && self.model_input_channels != self.source_channels {
            return Err(
                "meeting-speaker fixed-array report channel geometry is inconsistent".into(),
            );
        }
        let model_frames = crate::resample::planned_output_frames(
            self.source_frames,
            self.source_sample_rate,
            self.model_sample_rate,
        )?;
        let expected_windows = if model_frames <= self.model_window_samples {
            1
        } else {
            (model_frames - self.model_window_samples).div_ceil(self.model_hop_samples) + 1
        };
        if expected_windows != self.model_windows {
            return Err("meeting-speaker report window count is inconsistent".into());
        }
        let mut ids = BTreeSet::new();
        let mut labels = BTreeSet::new();
        let mut previous_id = None;
        for track in &self.track_summaries {
            validate_track_id(&track.id)?;
            if previous_id.is_some_and(|value: &str| value >= track.id.as_str())
                || !ids.insert(track.id.as_str())
            {
                return Err("meeting-speaker track summaries must be sorted and unique".into());
            }
            previous_id = Some(&track.id);
            validate_sha256("track PCM", &track.pcm_sha256)?;
            if track.segments.is_empty() || track.segments.len() > MAX_SEGMENTS {
                return Err("meeting-speaker published tracks require bounded segments".into());
            }
            validate_segments(&track.segments, self.source_frames)?;
            let expected_active = track
                .segments
                .iter()
                .filter(|segment| segment.state == MeetingActivityState::Active)
                .try_fold(0_u64, |total, segment| {
                    total.checked_add(segment.end_sample - segment.start_sample)
                })
                .ok_or_else(|| "meeting-speaker active duration overflow".to_string())?;
            let expected_uncertain = track
                .segments
                .iter()
                .filter(|segment| segment.state == MeetingActivityState::Uncertain)
                .try_fold(0_u64, |total, segment| {
                    total.checked_add(segment.end_sample - segment.start_sample)
                })
                .ok_or_else(|| "meeting-speaker uncertain duration overflow".to_string())?;
            if expected_active == 0
                || track.active_samples != expected_active
                || track.uncertain_samples != expected_uncertain
            {
                return Err("meeting-speaker track duration summary is inconsistent".into());
            }
            match (
                &track.label,
                &track.consent_record_sha256,
                &track.target_speaker_report_sha256,
            ) {
                (None, None, None) => {}
                (Some(label), Some(consent), Some(report)) => {
                    if label.is_empty() || label.len() > 128 || label.chars().any(char::is_control)
                    {
                        return Err("meeting-speaker report label is invalid".into());
                    }
                    if !labels.insert(label.as_str()) {
                        return Err("meeting-speaker report labels must be unique".into());
                    }
                    validate_sha256("track consent", consent)?;
                    validate_sha256("target-speaker report", report)?;
                }
                _ => {
                    return Err(
                        "meeting-speaker label metadata must be all-present or all-absent".into(),
                    )
                }
            }
        }
        validate_regions(&self.unknown_regions, self.source_frames)?;
        validate_regions(&self.overlap_regions, self.source_frames)?;
        let limitations: BTreeSet<&str> = self.limitations.iter().map(String::as_str).collect();
        if !(6..=10).contains(&self.limitations.len())
            || limitations.len() != self.limitations.len()
            || self.limitations.iter().any(|value| {
                value.is_empty() || value.len() > 1_024 || value.chars().any(char::is_control)
            })
        {
            return Err("meeting-speaker report limitations are invalid".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize meeting-speaker report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize meeting-speaker report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct MeetingSpeakerResult {
    pub tracks: Vec<Audio>,
    pub unassigned: Audio,
    pub report: MeetingSpeakerReport,
}

pub fn estimate_meeting_speaker_memory_bytes(
    input: &Audio,
    model_sample_rate: u32,
    model_input_channels: usize,
    model_window_samples: usize,
    tracks: usize,
) -> Result<u64, String> {
    if input.sample_rate == 0
        || input.channels.is_empty()
        || model_input_channels == 0
        || model_input_channels > 64
        || !(256..=16_777_216).contains(&model_window_samples)
        || !(1..=MAX_MEETING_SPEAKER_TRACKS).contains(&tracks)
    {
        return Err("meeting-speaker memory geometry is invalid".into());
    }
    let model_frames = crate::resample::planned_output_frames(
        input.frames(),
        input.sample_rate,
        model_sample_rate,
    )?;
    let model_scalars_per_frame = (tracks as u128)
        .checked_mul(8)
        .and_then(|value| value.checked_add((model_input_channels as u128).saturating_mul(2)))
        .and_then(|value| value.checked_add(8))
        .ok_or_else(|| "meeting-speaker memory estimate overflow".to_string())?;
    let source_scalars_per_frame = (tracks as u128)
        .checked_mul(2)
        .and_then(|value| value.checked_add(6))
        .ok_or_else(|| "meeting-speaker memory estimate overflow".to_string())?;
    let scalar_bytes = (model_frames as u128)
        .checked_mul(model_scalars_per_frame)
        .and_then(|value| {
            value.checked_add((input.frames() as u128).saturating_mul(source_scalars_per_frame))
        })
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() as u128))
        .ok_or_else(|| "meeting-speaker memory estimate overflow".to_string())?;
    let window_scalars_per_frame = (tracks as u128)
        .checked_mul(6)
        .and_then(|value| value.checked_add((model_input_channels as u128).saturating_mul(2)))
        .and_then(|value| value.checked_add(8))
        .ok_or_else(|| "meeting-speaker window memory estimate overflow".to_string())?;
    let window_bytes = (model_window_samples as u128)
        .checked_mul(window_scalars_per_frame)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() as u128))
        .ok_or_else(|| "meeting-speaker window memory estimate overflow".to_string())?;
    let input_resampler = crate::resample::resampler_plan_bytes(
        model_input_channels,
        input.sample_rate,
        model_sample_rate,
    )?;
    let output_resampler =
        crate::resample::resampler_plan_bytes(1, model_sample_rate, input.sample_rate)?;
    let bytes = scalar_bytes
        .checked_add(window_bytes)
        .and_then(|value| {
            value.checked_add(crate::audio::estimate_audio_memory_bytes(input) as u128)
        })
        .and_then(|value| value.checked_add(u128::from(input_resampler.max(output_resampler))))
        .ok_or_else(|| "meeting-speaker memory estimate overflow".to_string())?;
    u64::try_from(bytes).map_err(|_| "meeting-speaker memory estimate exceeds u64".to_string())
}

#[cfg(feature = "onnx")]
pub struct MeetingSpeakerSession {
    package: RuntimeModelPackage,
    model: crate::backend::meeting_speaker::MeetingSpeakerModel,
    accelerator: AcceleratorSelection,
    evidence: MeetingSpeakerEvidenceIdentity,
}

#[cfg(feature = "onnx")]
impl std::fmt::Debug for MeetingSpeakerSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MeetingSpeakerSession")
            .field("package_sha256", &self.package.package_sha256())
            .field("model", &self.model)
            .field("accelerator", &self.accelerator)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "onnx")]
impl MeetingSpeakerSession {
    pub fn prepare(
        package: RuntimeModelPackage,
        evidence: &SignedMeetingSpeakerPromotionEvidence,
        evidence_key: &ReceiptPublicKey,
        config: &MeetingSpeakerConfig,
        requested: AcceleratorPreference,
    ) -> Result<Self, String> {
        config.validate()?;
        evidence.verify_signature(evidence_key)?;
        if !evidence.payload.accepted {
            return Err(
                "meeting-speaker evidence is authentic but does not pass promotion gates".into(),
            );
        }
        let manifest = package
            .manifest_v2()
            .ok_or("meeting speaker tracks reject runtime model package v1")?;
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
                config.digest()?.as_str(),
            ),
        ] {
            if observed != expected {
                return Err(format!(
                    "meeting-speaker evidence {label} does not match the authenticated package/configuration"
                ));
            }
        }
        let mut options = BackendOptions::default().with_runtime_model_package(package.clone());
        options.deterministic = true;
        options.accelerator = requested;
        let accelerator = crate::select_accelerator_for_options(Backend::Onnx, &options)?;
        if !package.supports_accelerator(accelerator.effective()) {
            return Err(format!(
                "meeting-speaker package does not permit the {} accelerator",
                accelerator.effective().name()
            ));
        }
        let model = crate::backend::meeting_speaker::MeetingSpeakerModel::load_runtime_package(
            &package,
            accelerator.effective(),
        )?;
        let payload = &evidence.payload;
        Ok(Self {
            package,
            model,
            accelerator,
            evidence: MeetingSpeakerEvidenceIdentity {
                signing_key_id: evidence.signature.key_id.clone(),
                corpus_manifest_sha256: payload.corpus_manifest_sha256.clone(),
                corpus_license_manifest_sha256: payload.corpus_license_manifest_sha256.clone(),
                evaluation_result_sha256: payload.evaluation_result_sha256.clone(),
                listening_result_sha256: payload.listening_result_sha256.clone(),
                real_meeting_cases: payload.real_meeting_cases,
                distinct_speakers: payload.distinct_speakers,
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
            .expect("meeting-speaker packages use v2 precision profiles");
        Ok(profile
            .resources
            .max_session_memory_bytes
            .saturating_add(profile.resources.max_worker_memory_bytes))
    }

    pub fn processing_working_set_bytes(&self, input: &Audio) -> Result<u64, String> {
        let manifest = self
            .package
            .manifest_v2()
            .expect("meeting-speaker session requires v2");
        estimate_meeting_speaker_memory_bytes(
            input,
            manifest.runtime.sample_rate_hz,
            self.model.input_channels(),
            self.model.window_samples(),
            self.model.tracks(),
        )
    }

    pub fn separate(
        &self,
        input: &Audio,
        config: &MeetingSpeakerConfig,
        labels: &[MeetingTrackLabel],
    ) -> Result<MeetingSpeakerResult, String> {
        config.validate()?;
        validate_audio(input)?;
        validate_labels(labels, self.model.tracks())?;
        if u128::from(self.processing_working_set_bytes(input)?) > MAX_WORKING_BYTES {
            return Err("meeting-speaker working set exceeds the 2-GiB hard limit".into());
        }
        let manifest = self
            .package
            .manifest_v2()
            .expect("meeting-speaker session requires v2");
        let model_rate = manifest.runtime.sample_rate_hz;
        let mut model_input = if self.model.input_channels() == 1 {
            vec![mono_mix(input)?]
        } else {
            if input.channels() != self.model.input_channels() {
                return Err(format!(
                    "meeting-speaker fixed array package requires {} channels, got {}",
                    self.model.input_channels(),
                    input.channels()
                ));
            }
            input.channels.clone()
        };
        model_input =
            crate::resample::resample_channels(&model_input, input.sample_rate, model_rate)?;
        let model_frames = model_input.first().map_or(0, Vec::len);
        if model_frames == 0 {
            return Err("meeting-speaker input becomes empty at the model rate".into());
        }
        let window = self.model.window_samples();
        let hop = usize::try_from(manifest.latency.hop_samples)
            .map_err(|_| "meeting-speaker hop is too large".to_string())?;
        let activity_frames = self.model.activity_frames();
        let activity_hop = window / activity_frames;
        debug_assert_eq!(hop % activity_hop, 0);
        let windows = window_starts(model_frames, window, hop)?;
        let tracks = self.model.tracks();
        let global_activity_frames = model_frames.div_ceil(activity_hop);
        let mut track_sum = allocate_matrix(tracks, model_frames, "speaker-track accumulator")?;
        let mut track_weight = allocate_matrix(tracks, model_frames, "speaker-track weights")?;
        let mut activity_sum = allocate_probability_cube(
            tracks,
            global_activity_frames,
            "speaker activity probabilities",
        )?;
        let mut activity_weight =
            allocate_matrix(tracks, global_activity_frames, "speaker activity weights")?;
        let mut meeting_sum = vec![[0.0_f64; 3]; global_activity_frames];
        let mut meeting_weight = vec![0_u32; global_activity_frames];
        let mut ambiguous_activity = vec![false; global_activity_frames];
        let mut previous_mapping: Vec<usize> = (0..tracks).collect();
        let overlap = window.saturating_sub(hop);
        let mut permutation_ambiguous_windows = 0usize;

        for (window_index, &start) in windows.iter().enumerate() {
            let mut block = allocate_f32_matrix(
                self.model.input_channels(),
                window,
                "meeting model input window",
            )?;
            let available = window.min(model_frames - start);
            for (destination, source) in block.iter_mut().zip(&model_input) {
                for (out, sample) in destination[..available]
                    .iter_mut()
                    .zip(&source[start..start + available])
                {
                    *out = *sample as f32;
                }
            }
            let inference = self.model.process(&block)?;
            let mapping = if window_index == 0 || overlap == 0 {
                previous_mapping.clone()
            } else {
                let comparison = overlap.min(available);
                let scores = correlation_matrix(
                    &track_sum,
                    &track_weight,
                    &inference.tracks,
                    start,
                    comparison,
                );
                let assignment = best_assignment(&scores)?;
                let confident = assignment.best_average >= config.permutation_minimum_correlation
                    && assignment.best_average - assignment.second_average
                        >= config.permutation_minimum_margin;
                if confident {
                    assignment.local_to_global
                } else {
                    permutation_ambiguous_windows += 1;
                    let first = start / activity_hop;
                    let count = available.div_ceil(activity_hop);
                    for value in ambiguous_activity
                        .iter_mut()
                        .skip(first)
                        .take(count.min(global_activity_frames.saturating_sub(first)))
                    {
                        *value = true;
                    }
                    previous_mapping.clone()
                }
            };
            previous_mapping = mapping.clone();
            for (local, &global) in mapping.iter().enumerate() {
                for offset in 0..available {
                    let value = f64::from(inference.tracks[local][offset]);
                    track_sum[global][start + offset] += value;
                    track_weight[global][start + offset] += 1.0;
                }
                for frame in 0..activity_frames {
                    let model_sample = start.saturating_add(frame * activity_hop);
                    if model_sample >= model_frames {
                        break;
                    }
                    let global_frame = model_sample / activity_hop;
                    for class in 0..3 {
                        activity_sum[global][global_frame][class] +=
                            f64::from(inference.track_activity_probabilities[local][frame][class]);
                    }
                    activity_weight[global][global_frame] += 1.0;
                }
            }
            for frame in 0..activity_frames {
                let model_sample = start.saturating_add(frame * activity_hop);
                if model_sample >= model_frames {
                    break;
                }
                let global_frame = model_sample / activity_hop;
                for class in 0..3 {
                    meeting_sum[global_frame][class] +=
                        f64::from(inference.meeting_state_probabilities[frame][class]);
                }
                meeting_weight[global_frame] = meeting_weight[global_frame].saturating_add(1);
            }
        }

        for track in 0..tracks {
            for frame in 0..model_frames {
                let weight = track_weight[track][frame];
                if weight > 0.0 {
                    track_sum[track][frame] /= weight;
                }
            }
            for frame in 0..global_activity_frames {
                let weight = activity_weight[track][frame];
                if weight > 0.0 {
                    for class in 0..3 {
                        activity_sum[track][frame][class] /= weight;
                    }
                }
            }
        }
        for frame in 0..global_activity_frames {
            let weight = f64::from(meeting_weight[frame]);
            if weight > 0.0 {
                for class in 0..3 {
                    meeting_sum[frame][class] /= weight;
                }
            }
        }

        let activity = classify_activity(&activity_sum, config);
        let overlap_flags = overlap_flags(&activity);
        let unknown_flags = unknown_flags(
            &meeting_sum,
            &ambiguous_activity,
            config.minimum_unknown_probability,
        );
        let publish = published_tracks(&activity, config.minimum_active_frames);
        if !publish.iter().any(|value| *value) {
            return Err(
                "meeting-speaker model did not produce a bounded confidently active track".into(),
            );
        }
        let reference_model = arithmetic_mean(&model_input)?;
        let mut published_model = Vec::new();
        let mut published_indices = Vec::new();
        for (index, should_publish) in publish.iter().copied().enumerate() {
            if should_publish {
                ensure_peak(
                    &track_sum[index],
                    config.maximum_track_peak,
                    "meeting speaker track",
                )?;
                published_indices.push(index);
                published_model.push(track_sum[index].clone());
            }
        }
        let mut residual_model = reference_model.clone();
        for track in &published_model {
            for (residual, sample) in residual_model.iter_mut().zip(track) {
                *residual -= sample;
            }
        }
        ensure_peak(
            &residual_model,
            config.maximum_residual_peak,
            "meeting unassigned residual",
        )?;

        let mut output_tracks = Vec::new();
        output_tracks
            .try_reserve_exact(published_model.len())
            .map_err(|_| "unable to reserve meeting output tracks".to_string())?;
        for track in &published_model {
            let mut samples = crate::resample::resample(track, model_rate, input.sample_rate)?;
            samples.resize(input.frames(), 0.0);
            samples.truncate(input.frames());
            ensure_peak(&samples, config.maximum_track_peak, "meeting speaker track")?;
            output_tracks.push(Audio {
                sample_rate: input.sample_rate,
                channels: vec![samples],
                bits_per_sample: input.bits_per_sample,
                sample_format: input.sample_format,
                channel_mask: None,
            });
        }
        let reference_source = mono_mix(input)?;
        let mut residual_source = reference_source.clone();
        for track in &output_tracks {
            for (residual, sample) in residual_source.iter_mut().zip(&track.channels[0]) {
                *residual -= sample;
            }
        }
        ensure_peak(
            &residual_source,
            config.maximum_residual_peak,
            "meeting unassigned residual",
        )?;
        let recombination_maximum_absolute_error = reference_source
            .iter()
            .enumerate()
            .map(|(frame, reference)| {
                let sum = output_tracks
                    .iter()
                    .map(|track| track.channels[0][frame])
                    .sum::<f64>()
                    + residual_source[frame];
                (sum - reference).abs()
            })
            .fold(0.0_f64, f64::max);
        if recombination_maximum_absolute_error > 1.0e-12 {
            return Err("meeting speaker tracks do not exactly recombine with the residual".into());
        }
        let unassigned = Audio {
            sample_rate: input.sample_rate,
            channels: vec![residual_source],
            bits_per_sample: input.bits_per_sample,
            sample_format: input.sample_format,
            channel_mask: None,
        };

        let published_ids: BTreeSet<String> = published_indices
            .iter()
            .map(|index| track_id(*index))
            .collect();
        if labels
            .iter()
            .any(|label| !published_ids.contains(&label.track_id))
        {
            return Err(
                "meeting track labels must reference tracks that were confidently published".into(),
            );
        }
        let label_map: BTreeMap<&str, &MeetingTrackLabel> = labels
            .iter()
            .map(|label| (label.track_id.as_str(), label))
            .collect();
        let mut track_summaries = Vec::new();
        for (published_position, &model_track) in published_indices.iter().enumerate() {
            let id = track_id(model_track);
            let segments = build_segments(
                &activity[model_track],
                &overlap_flags,
                &activity_sum[model_track],
                activity_hop,
                model_rate,
                input.sample_rate,
                input.frames(),
            )?;
            let active_samples = segments
                .iter()
                .filter(|segment| segment.state == MeetingActivityState::Active)
                .map(|segment| segment.end_sample - segment.start_sample)
                .sum();
            let uncertain_samples = segments
                .iter()
                .filter(|segment| segment.state == MeetingActivityState::Uncertain)
                .map(|segment| segment.end_sample - segment.start_sample)
                .sum();
            let label = label_map.get(id.as_str()).copied();
            track_summaries.push(MeetingSpeakerTrackSummary {
                id,
                label: label.map(|value| value.label.clone()),
                pcm_sha256: pcm_digest(&output_tracks[published_position], TRACK_PCM_DIGEST_DOMAIN),
                active_samples,
                uncertain_samples,
                segments,
                consent_record_sha256: label.map(|value| value.consent_record_sha256.clone()),
                target_speaker_report_sha256: label
                    .map(|value| value.target_speaker_report_sha256.clone()),
            });
        }
        let unknown_regions = build_regions(
            &unknown_flags,
            &meeting_sum,
            2,
            activity_hop,
            model_rate,
            input.sample_rate,
            input.frames(),
        )?;
        let overlap_confidence = overlap_confidences(&activity_sum);
        let overlap_regions = build_regions(
            &overlap_flags,
            &overlap_confidence,
            0,
            activity_hop,
            model_rate,
            input.sample_rate,
            input.frames(),
        )?;
        let profile = self
            .package
            .precision_profile_for(self.accelerator.effective())?
            .expect("meeting-speaker session selects one v2 profile");
        let report = MeetingSpeakerReport {
            schema: MEETING_SPEAKER_REPORT_SCHEMA.into(),
            schema_version: MEETING_SPEAKER_SCHEMA_VERSION,
            denoize_version: env!("CARGO_PKG_VERSION").into(),
            configuration_sha256: config.digest()?,
            network_accessed: false,
            deterministic: true,
            model: MeetingSpeakerModelIdentity {
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
            source_sample_rate: input.sample_rate,
            source_channels: input.channels(),
            source_frames: input.frames(),
            model_sample_rate: model_rate,
            model_input_channels: self.model.input_channels(),
            model_window_samples: window,
            model_hop_samples: hop,
            model_activity_frames: activity_frames,
            model_windows: windows.len(),
            maximum_tracks: tracks,
            published_tracks: output_tracks.len(),
            track_summaries,
            unknown_regions,
            overlap_regions,
            permutation_ambiguous_windows,
            mixture_pcm_sha256: pcm_digest(input, INPUT_PCM_DIGEST_DOMAIN),
            unassigned_pcm_sha256: pcm_digest(&unassigned, RESIDUAL_PCM_DIGEST_DOMAIN),
            recombination_maximum_absolute_error,
            exact_output_duration: output_tracks
                .iter()
                .all(|track| track.frames() == input.frames())
                && unassigned.frames() == input.frames(),
            raw_enrollment_retained: false,
            speaker_embeddings_retained: false,
            path_fields_recorded: 0,
            limitations: vec![
                "speaker tracks are anonymous unless an explicit Stage 29 consent receipt labels them".into(),
                "unknown speech is never forced into a named or anonymous identity".into(),
                "the track count is capped at eight and overflow remains in the unassigned residual".into(),
                "window permutations use bounded waveform continuity and expose every ambiguous window".into(),
                "the unassigned residual is required for exact mixture reconstruction".into(),
                "transcription is evaluation-only and is not produced by this operation".into(),
                "no checkpoint is bundled; the exact package and licensed-corpus evidence must be supplied".into(),
            ],
        };
        report.validate()?;
        Ok(MeetingSpeakerResult {
            tracks: output_tracks,
            unassigned,
            report,
        })
    }
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameActivity {
    Inactive,
    Active,
    Uncertain,
}

#[cfg(feature = "onnx")]
fn classify_activity(
    probabilities: &[Vec<[f64; 3]>],
    config: &MeetingSpeakerConfig,
) -> Vec<Vec<FrameActivity>> {
    probabilities
        .iter()
        .map(|track| {
            track
                .iter()
                .map(|values| {
                    if values[2] >= config.minimum_active_probability
                        && values[2] > values[0]
                        && values[2] > values[1]
                    {
                        FrameActivity::Active
                    } else if values[0] >= config.minimum_inactive_probability
                        && values[0] > values[1]
                        && values[0] > values[2]
                    {
                        FrameActivity::Inactive
                    } else {
                        FrameActivity::Uncertain
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn published_tracks(activity: &[Vec<FrameActivity>], minimum: usize) -> Vec<bool> {
    activity
        .iter()
        .map(|track| {
            let mut run = 0usize;
            track.iter().any(|state| {
                run = if *state == FrameActivity::Active {
                    run.saturating_add(1)
                } else {
                    0
                };
                run >= minimum
            })
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn overlap_flags(activity: &[Vec<FrameActivity>]) -> Vec<bool> {
    let frames = activity.first().map_or(0, Vec::len);
    (0..frames)
        .map(|frame| {
            activity
                .iter()
                .filter(|track| track[frame] == FrameActivity::Active)
                .take(2)
                .count()
                >= 2
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn unknown_flags(meeting: &[[f64; 3]], ambiguous: &[bool], threshold: f64) -> Vec<bool> {
    meeting
        .iter()
        .zip(ambiguous)
        .map(|(values, ambiguous)| {
            *ambiguous || (values[2] >= threshold && values[2] > values[0] && values[2] > values[1])
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn overlap_confidences(activity: &[Vec<[f64; 3]>]) -> Vec<[f64; 3]> {
    let frames = activity.first().map_or(0, Vec::len);
    (0..frames)
        .map(|frame| {
            let mut active = activity
                .iter()
                .map(|track| track[frame][2])
                .collect::<Vec<_>>();
            active.sort_by(|left, right| right.total_cmp(left));
            [active.get(1).copied().unwrap_or(0.0), 0.0, 0.0]
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn build_segments(
    activity: &[FrameActivity],
    overlap: &[bool],
    probabilities: &[[f64; 3]],
    activity_hop: usize,
    model_rate: u32,
    source_rate: u32,
    source_frames: usize,
) -> Result<Vec<MeetingSpeakerSegment>, String> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    while start < activity.len() {
        if activity[start] == FrameActivity::Inactive {
            start += 1;
            continue;
        }
        let state = activity[start];
        let is_overlap = overlap[start];
        let mut end = start + 1;
        while end < activity.len() && activity[end] == state && overlap[end] == is_overlap {
            end += 1;
        }
        let confidence_index = if state == FrameActivity::Active { 2 } else { 1 };
        let confidence = probabilities[start..end]
            .iter()
            .map(|values| values[confidence_index])
            .sum::<f64>()
            / (end - start) as f64;
        segments.push(MeetingSpeakerSegment {
            start_sample: map_sample(start * activity_hop, model_rate, source_rate),
            end_sample: map_sample(end * activity_hop, model_rate, source_rate)
                .min(source_frames as u64),
            state: if state == FrameActivity::Active {
                MeetingActivityState::Active
            } else {
                MeetingActivityState::Uncertain
            },
            confidence,
            overlap: is_overlap,
        });
        if segments.len() > MAX_SEGMENTS {
            return Err("meeting-speaker segment count exceeds the bounded limit".into());
        }
        start = end;
    }
    segments.retain(|segment| segment.start_sample < segment.end_sample);
    Ok(segments)
}

#[cfg(feature = "onnx")]
fn build_regions(
    flags: &[bool],
    probabilities: &[[f64; 3]],
    confidence_index: usize,
    activity_hop: usize,
    model_rate: u32,
    source_rate: u32,
    source_frames: usize,
) -> Result<Vec<MeetingRegion>, String> {
    let mut regions = Vec::new();
    let mut start = 0usize;
    while start < flags.len() {
        if !flags[start] {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < flags.len() && flags[end] {
            end += 1;
        }
        let confidence = probabilities[start..end]
            .iter()
            .map(|values| values[confidence_index])
            .sum::<f64>()
            / (end - start) as f64;
        regions.push(MeetingRegion {
            start_sample: map_sample(start * activity_hop, model_rate, source_rate),
            end_sample: map_sample(end * activity_hop, model_rate, source_rate)
                .min(source_frames as u64),
            confidence,
        });
        if regions.len() > MAX_SEGMENTS {
            return Err("meeting-speaker region count exceeds the bounded limit".into());
        }
        start = end;
    }
    regions.retain(|region| region.start_sample < region.end_sample);
    Ok(regions)
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

#[cfg(feature = "onnx")]
struct Assignment {
    local_to_global: Vec<usize>,
    best_average: f64,
    second_average: f64,
}

#[cfg(feature = "onnx")]
fn best_assignment(scores: &[Vec<f64>]) -> Result<Assignment, String> {
    let tracks = scores.len();
    if tracks == 0
        || tracks > MAX_MEETING_SPEAKER_TRACKS
        || scores.iter().any(|row| row.len() != tracks)
        || scores.iter().flatten().any(|score| !score.is_finite())
    {
        return Err("meeting-speaker permutation matrix is invalid".into());
    }
    let states = 1usize << tracks;
    let mut best = vec![f64::NEG_INFINITY; states];
    let mut second = vec![f64::NEG_INFINITY; states];
    let mut parent = vec![None; states];
    best[0] = 0.0;
    for mask in 0..states {
        let local = mask.count_ones() as usize;
        if local >= tracks || !best[mask].is_finite() {
            continue;
        }
        for global in 0..tracks {
            if mask & (1 << global) != 0 {
                continue;
            }
            let next = mask | (1 << global);
            let candidate = best[mask] + scores[local][global];
            if candidate > best[next] {
                second[next] = best[next].max(second[next]).max(candidate.min(best[next]));
                best[next] = candidate;
                parent[next] = Some((mask, global));
            } else if candidate > second[next] {
                second[next] = candidate;
            }
            if second[mask].is_finite() {
                let alternative = second[mask] + scores[local][global];
                if alternative < best[next] && alternative > second[next] {
                    second[next] = alternative;
                }
            }
        }
    }
    let full = states - 1;
    if !best[full].is_finite() {
        return Err("meeting-speaker permutation assignment failed".into());
    }
    let mut local_to_global = vec![0usize; tracks];
    let mut mask = full;
    for local in (0..tracks).rev() {
        let (previous, global) = parent[mask]
            .ok_or_else(|| "meeting-speaker permutation assignment is incomplete".to_string())?;
        local_to_global[local] = global;
        mask = previous;
    }
    Ok(Assignment {
        local_to_global,
        best_average: best[full] / tracks as f64,
        second_average: if second[full].is_finite() {
            second[full] / tracks as f64
        } else {
            f64::NEG_INFINITY
        },
    })
}

#[cfg(feature = "onnx")]
fn correlation_matrix(
    accumulated: &[Vec<f64>],
    weights: &[Vec<f64>],
    local: &[Vec<f32>],
    start: usize,
    count: usize,
) -> Vec<Vec<f64>> {
    local
        .iter()
        .map(|candidate| {
            accumulated
                .iter()
                .zip(weights)
                .map(|(reference, weight)| {
                    normalized_correlation(reference, weight, candidate, start, count)
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn normalized_correlation(
    reference: &[f64],
    weights: &[f64],
    candidate: &[f32],
    start: usize,
    count: usize,
) -> f64 {
    let mut dot = 0.0;
    let mut left = 0.0;
    let mut right = 0.0;
    for offset in 0..count {
        if weights[start + offset] <= 0.0 {
            continue;
        }
        let a = reference[start + offset] / weights[start + offset];
        let b = f64::from(candidate[offset]);
        dot += a * b;
        left += a * a;
        right += b * b;
    }
    if left <= 1.0e-10 || right <= 1.0e-10 {
        0.0
    } else {
        (dot / (left.sqrt() * right.sqrt())).clamp(-1.0, 1.0)
    }
}

#[cfg(feature = "onnx")]
fn window_starts(frames: usize, window: usize, hop: usize) -> Result<Vec<usize>, String> {
    if frames == 0 || window == 0 || hop == 0 || hop > window {
        return Err("meeting-speaker window geometry is invalid".into());
    }
    let count = if frames <= window {
        1
    } else {
        (frames - window).div_ceil(hop) + 1
    };
    if count > MAX_SEGMENTS {
        return Err("meeting-speaker window count exceeds the bounded limit".into());
    }
    let mut starts = Vec::new();
    starts
        .try_reserve_exact(count)
        .map_err(|_| "unable to reserve meeting-speaker windows".to_string())?;
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
        .map_err(|_| format!("unable to reserve {label}"))?;
    for _ in 0..rows {
        let mut row = Vec::new();
        row.try_reserve_exact(columns)
            .map_err(|_| format!("unable to reserve {label}"))?;
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
        .map_err(|_| format!("unable to reserve {label}"))?;
    for _ in 0..rows {
        let mut row = Vec::new();
        row.try_reserve_exact(columns)
            .map_err(|_| format!("unable to reserve {label}"))?;
        row.resize(columns, 0.0);
        output.push(row);
    }
    Ok(output)
}

#[cfg(feature = "onnx")]
fn allocate_probability_cube(
    tracks: usize,
    frames: usize,
    label: &str,
) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(tracks)
        .map_err(|_| format!("unable to reserve {label}"))?;
    for _ in 0..tracks {
        let mut row = Vec::new();
        row.try_reserve_exact(frames)
            .map_err(|_| format!("unable to reserve {label}"))?;
        row.resize(frames, [0.0; 3]);
        output.push(row);
    }
    Ok(output)
}

#[cfg(feature = "onnx")]
fn validate_audio(audio: &Audio) -> Result<(), String> {
    if !(8_000..=192_000).contains(&audio.sample_rate)
        || audio.channels.is_empty()
        || audio.channels.len() > 64
        || audio.frames() == 0
        || audio.frames() as u64 > u64::from(audio.sample_rate).saturating_mul(MAX_MEETING_SECONDS)
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
        return Err("meeting-speaker input violates its bounded normalized-audio contract".into());
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn mono_mix(audio: &Audio) -> Result<Vec<f64>, String> {
    let mut mono = Vec::new();
    mono.try_reserve_exact(audio.frames())
        .map_err(|_| "unable to reserve meeting mono reference".to_string())?;
    let scale = 1.0 / audio.channels() as f64;
    for frame in 0..audio.frames() {
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

#[cfg(feature = "onnx")]
fn arithmetic_mean(channels: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    if channels.is_empty()
        || channels
            .iter()
            .any(|channel| channel.len() != channels[0].len())
    {
        return Err("meeting model channels have inconsistent geometry".into());
    }
    let frames = channels[0].len();
    let mut mono = Vec::new();
    mono.try_reserve_exact(frames)
        .map_err(|_| "unable to reserve meeting model reference".to_string())?;
    let scale = 1.0 / channels.len() as f64;
    for frame in 0..frames {
        mono.push(channels.iter().map(|channel| channel[frame]).sum::<f64>() * scale);
    }
    Ok(mono)
}

#[cfg(feature = "onnx")]
fn ensure_peak(samples: &[f64], maximum: f64, label: &str) -> Result<(), String> {
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || sample.abs() > maximum)
    {
        return Err(format!(
            "{label} contains a non-finite sample or exceeds the configured peak"
        ));
    }
    Ok(())
}

fn validate_labels(labels: &[MeetingTrackLabel], maximum_tracks: usize) -> Result<(), String> {
    if labels.len() > maximum_tracks {
        return Err("meeting track labels exceed the model track cap".into());
    }
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    for label in labels {
        label.validate()?;
        let index = label
            .track_id
            .strip_prefix("speaker-")
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or("meeting track label has an invalid anonymous track ID")?;
        if index == 0
            || index > maximum_tracks
            || !ids.insert(label.track_id.as_str())
            || !names.insert(label.label.as_str())
        {
            return Err(
                "meeting track labels must be unique and within the model track cap".into(),
            );
        }
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn track_id(index: usize) -> String {
    format!("speaker-{:03}", index + 1)
}

fn validate_track_id(value: &str) -> Result<(), String> {
    let Some(number) = value.strip_prefix("speaker-") else {
        return Err("meeting track ID must use speaker-NNN".into());
    };
    if number.len() != 3
        || !number.bytes().all(|byte| byte.is_ascii_digit())
        || number
            .parse::<usize>()
            .ok()
            .is_none_or(|index| index == 0 || index > MAX_MEETING_SPEAKER_TRACKS)
    {
        return Err("meeting track ID must use speaker-NNN within the track cap".into());
    }
    Ok(())
}

fn validate_segments(segments: &[MeetingSpeakerSegment], frames: usize) -> Result<(), String> {
    let mut previous_end = 0u64;
    for segment in segments {
        if segment.start_sample < previous_end
            || segment.start_sample >= segment.end_sample
            || segment.end_sample > frames as u64
            || !segment.confidence.is_finite()
            || !(0.0..=1.0).contains(&segment.confidence)
        {
            return Err("meeting-speaker segment geometry is invalid".into());
        }
        previous_end = segment.end_sample;
    }
    Ok(())
}

fn validate_regions(regions: &[MeetingRegion], frames: usize) -> Result<(), String> {
    if regions.len() > MAX_SEGMENTS {
        return Err("meeting-speaker region count exceeds the bounded limit".into());
    }
    let mut previous_end = 0u64;
    for region in regions {
        if region.start_sample < previous_end
            || region.start_sample >= region.end_sample
            || region.end_sample > frames as u64
            || !region.confidence.is_finite()
            || !(0.0..=1.0).contains(&region.confidence)
        {
            return Err("meeting-speaker region geometry is invalid".into());
        }
        previous_end = region.end_sample;
    }
    Ok(())
}

fn validate_range(label: &str, value: f64, minimum: f64, maximum: f64) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "meeting-speaker {label} must be finite and in {minimum}..={maximum}"
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
        return Err(format!("meeting-speaker {label} is invalid"));
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("meeting-speaker {label} SHA-256 is invalid"));
    }
    Ok(())
}

fn validate_bounded_text(label: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(format!("meeting-speaker {label} is invalid"));
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

    fn evidence() -> MeetingSpeakerPromotionEvidencePayload {
        MeetingSpeakerPromotionEvidencePayload {
            completed_at_unix_seconds: 1,
            model_package_sha256: digest(),
            source_revision: "revision-1".into(),
            source_sha256: digest(),
            checkpoint_sha256: digest(),
            configuration_sha256: MeetingSpeakerConfig::default().digest().unwrap(),
            corpus_manifest_sha256: digest(),
            corpus_license_manifest_sha256: digest(),
            evaluation_result_sha256: digest(),
            listening_result_sha256: digest(),
            strata: REQUIRED_STRATA
                .iter()
                .map(|id| MeetingSpeakerEvidenceStratum {
                    id: (*id).into(),
                    cases: 100,
                    permutation_si_sdr_improvement_db: 1.0,
                    diarization_error_rate: 0.20,
                    jaccard_error_rate: 0.30,
                    overlap_f1: 0.70,
                    track_swap_rate: 0.01,
                    tcp_wer_regression: 0.0,
                    unknown_false_assignment_rate: 0.005,
                    non_finite_samples: 0,
                    passed: true,
                })
                .collect(),
            real_meeting_cases: 100,
            distinct_speakers: 100,
            language_count: 2,
            speaker_count_expected_calibration_error: 0.04,
            listener_count: 20,
            listener_preference: 0.5,
            retained_enrollment_recordings: 0,
            retained_speaker_embeddings: 0,
            accepted: true,
        }
    }

    #[test]
    fn evidence_requires_exact_strata_and_privacy_gates() {
        evidence().validate().unwrap();
        let mut invalid = evidence();
        invalid.retained_speaker_embeddings = 1;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.strata.swap(0, 1);
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.strata[0].permutation_si_sdr_improvement_db = 241.0;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.strata[0].tcp_wer_regression = -2.0;
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.source_sha256 = "AB".repeat(32);
        assert!(invalid.validate().is_err());
        let mut invalid = evidence();
        invalid.source_revision = "revision with spaces".into();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn memory_estimate_charges_model_rate_and_authenticated_window() {
        let input = Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; 48_000]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let baseline = estimate_meeting_speaker_memory_bytes(&input, 16_000, 1, 32_000, 2).unwrap();
        let higher_rate =
            estimate_meeting_speaker_memory_bytes(&input, 48_000, 1, 32_000, 2).unwrap();
        let larger_window =
            estimate_meeting_speaker_memory_bytes(&input, 16_000, 1, 16_777_216, 2).unwrap();
        assert!(higher_rate > baseline);
        assert!(larger_window > baseline);
        assert!(is_semver_triplet("0.87.0"));
        assert!(!is_semver_triplet("0.87"));
        assert!(!is_semver_triplet("v0.87.0"));
    }

    #[test]
    fn labels_require_consent_and_never_retain_biometrics() {
        let valid = MeetingTrackLabel {
            track_id: "speaker-001".into(),
            label: "facilitator".into(),
            consent_record_sha256: digest(),
            target_speaker_report_sha256: digest(),
            raw_enrollment_retained: false,
            speaker_embedding_retained: false,
        };
        validate_labels(std::slice::from_ref(&valid), 2).unwrap();
        let mut invalid = valid;
        invalid.speaker_embedding_retained = true;
        assert!(validate_labels(&[invalid], 2).is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn exact_assignment_finds_permuted_tracks_and_margin() {
        let assignment = best_assignment(&[vec![0.1, 0.9], vec![0.8, 0.2]]).unwrap();
        assert_eq!(assignment.local_to_global, vec![1, 0]);
        assert!((assignment.best_average - 0.85).abs() < 1.0e-12);
        assert!(assignment.best_average > assignment.second_average);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn exact_assignment_matches_brute_force_top_two() {
        fn enumerate(
            scores: &[Vec<f64>],
            local: usize,
            used: usize,
            total: f64,
            totals: &mut Vec<f64>,
        ) {
            if local == scores.len() {
                totals.push(total);
                return;
            }
            for global in 0..scores.len() {
                if used & (1 << global) == 0 {
                    enumerate(
                        scores,
                        local + 1,
                        used | (1 << global),
                        total + scores[local][global],
                        totals,
                    );
                }
            }
        }

        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        for tracks in 1..=5 {
            for _ in 0..32 {
                let mut scores = vec![vec![0.0; tracks]; tracks];
                for row in &mut scores {
                    for score in row {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        *score = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                    }
                }
                let assignment = best_assignment(&scores).unwrap();
                let observed = assignment
                    .local_to_global
                    .iter()
                    .enumerate()
                    .map(|(local, global)| scores[local][*global])
                    .sum::<f64>();
                let mut totals = Vec::new();
                enumerate(&scores, 0, 0, 0.0, &mut totals);
                totals.sort_by(|left, right| right.total_cmp(left));
                assert!((observed - totals[0]).abs() < 1.0e-12);
                assert!((assignment.best_average - totals[0] / tracks as f64).abs() < 1.0e-12);
                if tracks == 1 {
                    assert_eq!(assignment.second_average, f64::NEG_INFINITY);
                } else {
                    assert!(
                        (assignment.second_average - totals[1] / tracks as f64).abs() < 1.0e-12
                    );
                }
            }
        }
        assert!(best_assignment(&[vec![f64::NAN]]).is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn activity_preserves_unknown_and_overlap() {
        let probabilities = vec![
            vec![[0.05, 0.05, 0.90], [0.05, 0.05, 0.90]],
            vec![[0.05, 0.05, 0.90], [0.90, 0.05, 0.05]],
        ];
        let activity = classify_activity(&probabilities, &MeetingSpeakerConfig::default());
        assert_eq!(overlap_flags(&activity), vec![true, false]);
        assert_eq!(published_tracks(&activity, 2), vec![true, false]);
        assert_eq!(
            unknown_flags(
                &[[0.05, 0.05, 0.90], [0.90, 0.05, 0.05]],
                &[false, true],
                0.8
            ),
            vec![true, true]
        );
    }
}
