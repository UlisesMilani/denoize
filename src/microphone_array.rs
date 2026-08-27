//! Explicit-geometry microphone-array enhancement.
//!
//! Ordinary stereo and surround are never inferred to be microphone arrays.
//! A caller must authenticate promotion evidence and provide a closed geometry
//! whose channel IDs, coordinates, calibration, and reference microphone bind
//! the exact processing configuration before audio is accepted.

use crate::audio::Audio;
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
use crate::fft::Complex;
use crate::restoration::{
    restore_audio, RestorationConfig, RestorationMode, RestorationOperation, WpeChannelMode,
    WpeConfig,
};
use crate::stft::{Stft, StftConfig};
use crate::window::{WindowParams, WindowType};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::Path;

pub const MICROPHONE_ARRAY_EVIDENCE_SCHEMA: &str = "denoize-microphone-array-promotion-evidence-v1";
pub const MICROPHONE_ARRAY_REPORT_SCHEMA: &str = "denoize-microphone-array-report-v1";
pub const MICROPHONE_ARRAY_SCHEMA_VERSION: u32 = 1;

const IMPLEMENTATION_ID: &str = "native-wpe-mask-mvdr-v1";
const CONFIG_DIGEST_DOMAIN: &[u8] = b"denoize-microphone-array-config-v1\0";
const EVIDENCE_SIGNATURE_DOMAIN: &[u8] = b"denoize-microphone-array-promotion-evidence-v1";
const INPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-microphone-array-input-pcm-v1\0";
const OUTPUT_PCM_DIGEST_DOMAIN: &[u8] = b"denoize-microphone-array-output-pcm-v1\0";
const MAX_CONFIG_BYTES: u64 = 256 * 1024;
const MAX_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CHANNELS: usize = 4;
const JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
const POWER_FLOOR: f64 = 1.0e-15;

const REQUIRED_STRATA: [&str; 12] = [
    "bad-channel",
    "channel-permutation",
    "clock-skew",
    "diffuse-noise",
    "directional-noise",
    "gain-phase-mismatch",
    "moving-source",
    "program-stereo",
    "real-meeting",
    "simulated-rir",
    "two-microphone",
    "unseen-geometry",
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArrayCoordinateUnit {
    Meters,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArrayHandedness {
    RightHandedXForwardYLeftZUp,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArrayInputSemantics {
    MicrophoneArray,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MicrophonePosition {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub sample_skew: i32,
    pub gain_mismatch_db: f64,
    pub phase_mismatch_degrees: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MicrophoneArrayGeometry {
    pub input_semantics: ArrayInputSemantics,
    pub coordinate_unit: ArrayCoordinateUnit,
    pub handedness: ArrayHandedness,
    pub reference_microphone_id: String,
    pub microphones: Vec<MicrophonePosition>,
}

impl MicrophoneArrayGeometry {
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=MAX_CHANNELS).contains(&self.microphones.len()) {
            return Err(format!(
                "microphone-array geometry requires 2..={MAX_CHANNELS} microphones"
            ));
        }
        validate_identifier("reference microphone ID", &self.reference_microphone_id)?;
        let mut ids = BTreeSet::new();
        let mut reference_count = 0usize;
        for microphone in &self.microphones {
            validate_identifier("microphone ID", &microphone.id)?;
            if !ids.insert(microphone.id.as_str()) {
                return Err("microphone-array IDs must be unique".into());
            }
            reference_count += usize::from(microphone.id == self.reference_microphone_id);
            for (axis, coordinate) in [
                ("x", microphone.x),
                ("y", microphone.y),
                ("z", microphone.z),
            ] {
                if !coordinate.is_finite() || !(-100.0..=100.0).contains(&coordinate) {
                    return Err(format!(
                        "microphone {axis} coordinate must be finite and in -100..=100 meters"
                    ));
                }
            }
            if microphone.sample_skew.unsigned_abs() > 32 {
                return Err("microphone sample_skew must be in -32..=32 samples".into());
            }
            if !microphone.gain_mismatch_db.is_finite()
                || !(-24.0..=24.0).contains(&microphone.gain_mismatch_db)
            {
                return Err("microphone gain mismatch must be finite and in -24..=24 dB".into());
            }
            if !microphone.phase_mismatch_degrees.is_finite()
                || !(-90.0..=90.0).contains(&microphone.phase_mismatch_degrees)
            {
                return Err(
                    "microphone phase mismatch must be finite and in -90..=90 degrees".into(),
                );
            }
        }
        if reference_count != 1 {
            return Err("reference_microphone_id must name exactly one microphone".into());
        }
        for left in 0..self.microphones.len() {
            for right in left + 1..self.microphones.len() {
                let a = &self.microphones[left];
                let b = &self.microphones[right];
                let distance =
                    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt();
                if distance < 1.0e-4 {
                    return Err("microphone positions must be distinct by at least 0.1 mm".into());
                }
            }
        }
        Ok(())
    }

    fn canonicalized(&self) -> Self {
        let mut geometry = self.clone();
        geometry
            .microphones
            .sort_by(|left, right| left.id.cmp(&right.id));
        geometry
    }

    fn canonical_channel_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.microphones.len()).collect();
        order.sort_by(|left, right| self.microphones[*left].id.cmp(&self.microphones[*right].id));
        order
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MicrophoneArrayConfig {
    pub sample_rate: u32,
    pub geometry: MicrophoneArrayGeometry,
    pub frame_size: usize,
    pub hop_size: usize,
    pub wpe_prediction_delay_frames: usize,
    pub wpe_prediction_taps: usize,
    pub wpe_iterations: usize,
    pub diagonal_loading: f64,
    pub maximum_condition_number: f64,
    pub covariance_smoothing: f64,
    pub inactive_channel_rms: f64,
    pub maximum_peak: f64,
}

impl Default for MicrophoneArrayConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            geometry: MicrophoneArrayGeometry {
                input_semantics: ArrayInputSemantics::MicrophoneArray,
                coordinate_unit: ArrayCoordinateUnit::Meters,
                handedness: ArrayHandedness::RightHandedXForwardYLeftZUp,
                reference_microphone_id: "mic-0".into(),
                microphones: vec![
                    MicrophonePosition {
                        id: "mic-0".into(),
                        x: -0.04,
                        y: 0.0,
                        z: 0.0,
                        sample_skew: 0,
                        gain_mismatch_db: 0.0,
                        phase_mismatch_degrees: 0.0,
                    },
                    MicrophonePosition {
                        id: "mic-1".into(),
                        x: 0.04,
                        y: 0.0,
                        z: 0.0,
                        sample_skew: 0,
                        gain_mismatch_db: 0.0,
                        phase_mismatch_degrees: 0.0,
                    },
                ],
            },
            frame_size: 512,
            hop_size: 128,
            wpe_prediction_delay_frames: 3,
            wpe_prediction_taps: 8,
            wpe_iterations: 3,
            diagonal_loading: 1.0e-3,
            maximum_condition_number: 1.0e6,
            covariance_smoothing: 0.05,
            inactive_channel_rms: 1.0e-7,
            maximum_peak: 1.0,
        }
    }
}

impl MicrophoneArrayConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) =
            crate::input::open_regular_file(path, "microphone-array configuration")?;
        if length >= MAX_CONFIG_BYTES {
            return Err(format!(
                "microphone-array configuration {} exceeds {MAX_CONFIG_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve microphone-array configuration".to_string())?;
        file.take(MAX_CONFIG_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read microphone-array configuration: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("microphone-array configuration changed while reading".into());
        }
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse microphone-array configuration: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(8_000..=192_000).contains(&self.sample_rate) {
            return Err("microphone-array sample_rate must be in 8000..=192000".into());
        }
        self.geometry.validate()?;
        if !self.frame_size.is_power_of_two() || !(128..=4096).contains(&self.frame_size) {
            return Err("microphone-array frame_size must be a power of two in 128..=4096".into());
        }
        if self.hop_size == 0 || self.hop_size > self.frame_size / 2 {
            return Err("microphone-array hop_size must be in 1..=frame_size/2".into());
        }
        if !(1..=16).contains(&self.wpe_prediction_delay_frames)
            || !(1..=24).contains(&self.wpe_prediction_taps)
            || !(1..=10).contains(&self.wpe_iterations)
        {
            return Err("microphone-array WPE geometry is outside bounded limits".into());
        }
        validate_f64_range("diagonal_loading", self.diagonal_loading, 1.0e-8, 0.25)?;
        validate_f64_range(
            "maximum_condition_number",
            self.maximum_condition_number,
            10.0,
            1.0e12,
        )?;
        validate_f64_range("covariance_smoothing", self.covariance_smoothing, 0.0, 1.0)?;
        validate_f64_range("inactive_channel_rms", self.inactive_channel_rms, 0.0, 0.1)?;
        validate_f64_range("maximum_peak", self.maximum_peak, 0.5, 1.0)?;
        if self.algorithmic_latency_milliseconds() > 100.0 {
            return Err("microphone-array STFT latency must not exceed 100 milliseconds".into());
        }
        estimate_microphone_array_memory_bytes(self, self.sample_rate as usize * 3)?;
        Ok(())
    }

    pub fn algorithmic_latency_milliseconds(&self) -> f64 {
        (self.frame_size - self.hop_size) as f64 * 1000.0 / self.sample_rate as f64
    }

    pub fn digest(&self) -> Result<String, String> {
        let mut canonical = self.clone();
        canonical.geometry = canonical.geometry.canonicalized();
        let document = serde_json::to_vec(&canonical)
            .map_err(|error| format!("serialize microphone-array configuration: {error}"))?;
        let mut digest = Sha256::new();
        digest.update(CONFIG_DIGEST_DOMAIN);
        digest.update(document);
        Ok(format!("{:x}", digest.finalize()))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MicrophoneArrayEvidenceStratum {
    pub id: String,
    pub cases: u64,
    pub si_sdr_improvement_db: f64,
    pub wer_regression: f64,
    pub doa_error_degrees: f64,
    pub reference_coloration_db: f64,
    pub target_leakage_db: f64,
    pub non_finite_samples: u64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MicrophoneArrayPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub implementation: String,
    pub implementation_source_revision: String,
    pub implementation_source_sha256: String,
    pub configuration_sha256: String,
    pub corpus_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub listening_result_sha256: String,
    pub strata: Vec<MicrophoneArrayEvidenceStratum>,
    pub real_meeting_cases: u64,
    pub unseen_geometry_cases: u64,
    pub permutation_cases: u64,
    pub paced_realtime_blocks: u64,
    pub worst_case_realtime_factor: f64,
    pub callback_allocations: u64,
    pub callback_locks: u64,
    pub callback_waits: u64,
    pub deadline_misses: u64,
    pub listener_count: u64,
    pub listener_preference: f64,
    pub accepted: bool,
}

impl MicrophoneArrayPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        if self.completed_at_unix_seconds > JSON_SAFE_INTEGER {
            return Err("microphone-array evidence timestamp exceeds JSON safe integer".into());
        }
        if self.implementation != IMPLEMENTATION_ID {
            return Err("unsupported microphone-array implementation".into());
        }
        validate_identifier(
            "implementation source revision",
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
        if self.strata.len() != REQUIRED_STRATA.len() {
            return Err(format!(
                "microphone-array evidence requires exactly {} strata",
                REQUIRED_STRATA.len()
            ));
        }
        let mut all_passed = true;
        for (index, stratum) in self.strata.iter().enumerate() {
            if stratum.id != REQUIRED_STRATA[index] {
                return Err("microphone-array evidence strata must be exact and sorted".into());
            }
            if !(10..=1_000_000).contains(&stratum.cases) {
                return Err("microphone-array stratum cases must be in 10..=1000000".into());
            }
            if stratum.non_finite_samples > JSON_SAFE_INTEGER {
                return Err(
                    "microphone-array non-finite count exceeds the JSON safe-integer limit".into(),
                );
            }
            let finite = [
                stratum.si_sdr_improvement_db,
                stratum.wer_regression,
                stratum.doa_error_degrees,
                stratum.reference_coloration_db,
                stratum.target_leakage_db,
            ]
            .iter()
            .all(|value| value.is_finite());
            let expected = finite
                && stratum.si_sdr_improvement_db >= 0.0
                && stratum.wer_regression <= 0.02
                && stratum.doa_error_degrees <= 20.0
                && stratum.reference_coloration_db.abs() <= 1.5
                && stratum.target_leakage_db <= -3.0
                && stratum.non_finite_samples == 0;
            if stratum.passed != expected {
                return Err(format!(
                    "microphone-array stratum {} has inconsistent promotion status",
                    stratum.id
                ));
            }
            all_passed &= stratum.passed;
        }
        if !(100..=1_000_000).contains(&self.real_meeting_cases)
            || !(100..=1_000_000).contains(&self.unseen_geometry_cases)
            || !(100..=1_000_000).contains(&self.permutation_cases)
            || !(10_000..=JSON_SAFE_INTEGER).contains(&self.paced_realtime_blocks)
            || !self.worst_case_realtime_factor.is_finite()
            || !(0.0..=0.5).contains(&self.worst_case_realtime_factor)
            || self.callback_allocations != 0
            || self.callback_locks != 0
            || self.callback_waits != 0
            || self.deadline_misses != 0
            || !(20..=100_000).contains(&self.listener_count)
            || !self.listener_preference.is_finite()
            || !(0.5..=1.0).contains(&self.listener_preference)
        {
            return Err("microphone-array global promotion evidence is outside hard limits".into());
        }
        let expected_accepted = all_passed && self.listener_preference >= 0.5;
        if self.accepted != expected_accepted {
            return Err("microphone-array accepted flag is inconsistent".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedMicrophoneArrayPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: MicrophoneArrayPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedMicrophoneArrayPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) = crate::input::open_regular_file(path, "microphone-array evidence")?;
        if length >= MAX_EVIDENCE_BYTES {
            return Err(format!(
                "microphone-array evidence {} exceeds {MAX_EVIDENCE_BYTES} bytes",
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve microphone-array evidence".to_string())?;
        file.take(MAX_EVIDENCE_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read microphone-array evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("microphone-array evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse microphone-array evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != MICROPHONE_ARRAY_EVIDENCE_SCHEMA
            || self.schema_version != MICROPHONE_ARRAY_SCHEMA_VERSION
        {
            return Err("unsupported microphone-array evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("microphone-array evidence must use ed25519".into());
        }
        validate_sha256("microphone-array evidence key ID", &self.signature.key_id)
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize microphone-array evidence: {error}"))?;
        key.verify_domain_document(
            EVIDENCE_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "microphone-array promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize microphone-array evidence: {error}"))
    }
}

pub fn sign_microphone_array_promotion_evidence(
    payload: MicrophoneArrayPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedMicrophoneArrayPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize microphone-array evidence: {error}"))?;
    let signature = key.sign_domain_document(
        EVIDENCE_SIGNATURE_DOMAIN,
        &document,
        "microphone-array promotion evidence",
    )?;
    let evidence = SignedMicrophoneArrayPromotionEvidence {
        schema: MICROPHONE_ARRAY_EVIDENCE_SCHEMA.into(),
        schema_version: MICROPHONE_ARRAY_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MicrophoneArrayReport {
    pub schema: String,
    pub schema_version: u32,
    pub implementation: String,
    pub configuration_sha256: String,
    pub evidence_signing_key_id: String,
    pub evidence_evaluation_result_sha256: String,
    pub input_pcm_sha256: String,
    pub output_pcm_sha256: String,
    pub sample_rate: u32,
    pub input_channels: usize,
    pub input_frames: usize,
    pub output_channels: usize,
    pub output_frames: usize,
    pub reference_microphone_id: String,
    pub canonical_microphone_ids: Vec<String>,
    pub active_microphones: usize,
    pub inactive_microphone_ids: Vec<String>,
    pub frame_size: usize,
    pub hop_size: usize,
    pub algorithmic_latency_milliseconds: f64,
    pub solved_frequency_bins: usize,
    pub fallback_frequency_bins: usize,
    pub maximum_observed_condition_number: f64,
    pub clipped_samples: u64,
    pub non_finite_samples: u64,
    pub exact_output_duration: bool,
    pub paths_recorded: u64,
    pub limitations: Vec<String>,
}

impl MicrophoneArrayReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MICROPHONE_ARRAY_REPORT_SCHEMA
            || self.schema_version != MICROPHONE_ARRAY_SCHEMA_VERSION
            || self.implementation != IMPLEMENTATION_ID
        {
            return Err("unsupported microphone-array report schema".into());
        }
        validate_sha256("configuration", &self.configuration_sha256)?;
        validate_sha256("evidence signing key", &self.evidence_signing_key_id)?;
        validate_sha256("evidence result", &self.evidence_evaluation_result_sha256)?;
        validate_sha256("input PCM", &self.input_pcm_sha256)?;
        validate_sha256("output PCM", &self.output_pcm_sha256)?;
        validate_identifier("reference microphone ID", &self.reference_microphone_id)?;
        for id in self
            .canonical_microphone_ids
            .iter()
            .chain(&self.inactive_microphone_ids)
        {
            validate_identifier("report microphone ID", id)?;
        }
        if !(8_000..=192_000).contains(&self.sample_rate)
            || !(2..=MAX_CHANNELS).contains(&self.input_channels)
            || self.input_frames == 0
            || self.input_frames > self.sample_rate as usize * 3_600
            || self.output_channels != 1
            || self.output_frames != self.input_frames
            || !self.exact_output_duration
            || self.non_finite_samples != 0
            || self.paths_recorded != 0
            || self
                .active_microphones
                .checked_add(self.inactive_microphone_ids.len())
                != Some(self.input_channels)
            || self
                .solved_frequency_bins
                .checked_add(self.fallback_frequency_bins)
                != Some(self.frame_size / 2 + 1)
            || !self.maximum_observed_condition_number.is_finite()
            || !(1.0..=1.0e12).contains(&self.maximum_observed_condition_number)
        {
            return Err("microphone-array report violates its closed invariants".into());
        }
        if !self.frame_size.is_power_of_two()
            || !(128..=4096).contains(&self.frame_size)
            || self.hop_size == 0
            || self.hop_size > self.frame_size / 2
        {
            return Err("microphone-array report has invalid STFT geometry".into());
        }
        let expected_latency =
            (self.frame_size - self.hop_size) as f64 * 1000.0 / self.sample_rate as f64;
        if !self.algorithmic_latency_milliseconds.is_finite()
            || (self.algorithmic_latency_milliseconds - expected_latency).abs() > 1.0e-9
        {
            return Err("microphone-array report latency does not match STFT geometry".into());
        }
        if self.canonical_microphone_ids.len() != self.input_channels
            || self
                .canonical_microphone_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || !self
                .canonical_microphone_ids
                .iter()
                .any(|id| id == &self.reference_microphone_id)
        {
            return Err("microphone-array report has invalid canonical microphone IDs".into());
        }
        let canonical: BTreeSet<&str> = self
            .canonical_microphone_ids
            .iter()
            .map(String::as_str)
            .collect();
        let inactive: BTreeSet<&str> = self
            .inactive_microphone_ids
            .iter()
            .map(String::as_str)
            .collect();
        if inactive.len() != self.inactive_microphone_ids.len()
            || !inactive.is_subset(&canonical)
            || inactive.contains(self.reference_microphone_id.as_str())
        {
            return Err("microphone-array report has invalid inactive microphone IDs".into());
        }
        if self.clipped_samples > u64::try_from(self.output_frames).unwrap_or(u64::MAX) {
            return Err("microphone-array report clipped-sample count exceeds output".into());
        }
        let limitations: BTreeSet<&str> = self.limitations.iter().map(String::as_str).collect();
        if !(5..=8).contains(&self.limitations.len())
            || limitations.len() != self.limitations.len()
            || self
                .limitations
                .iter()
                .any(|limitation| limitation.is_empty() || limitation.len() > 1_024)
        {
            return Err("microphone-array report has invalid limitations".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("serialize microphone-array report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize microphone-array report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct MicrophoneArrayResult {
    pub audio: Audio,
    pub report: MicrophoneArrayReport,
}

#[derive(Clone, Debug)]
pub struct MicrophoneArraySession {
    config: MicrophoneArrayConfig,
    evidence_signing_key_id: String,
    evidence_evaluation_result_sha256: String,
}

impl MicrophoneArraySession {
    pub fn prepare(
        evidence: &SignedMicrophoneArrayPromotionEvidence,
        evidence_key: &ReceiptPublicKey,
        config: MicrophoneArrayConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        evidence.verify_signature(evidence_key)?;
        if !evidence.payload.accepted {
            return Err("microphone-array evidence is authentic but not accepted".into());
        }
        if evidence.payload.implementation != IMPLEMENTATION_ID
            || evidence.payload.configuration_sha256 != config.digest()?
        {
            return Err(
                "microphone-array evidence does not bind the requested configuration".into(),
            );
        }
        Ok(Self {
            config,
            evidence_signing_key_id: evidence.signature.key_id.clone(),
            evidence_evaluation_result_sha256: evidence.payload.evaluation_result_sha256.clone(),
        })
    }

    pub fn config(&self) -> &MicrophoneArrayConfig {
        &self.config
    }

    pub fn enhance(&self, input: &Audio) -> Result<MicrophoneArrayResult, String> {
        validate_input(input, &self.config)?;
        let frames = input.frames();
        let order = self.config.geometry.canonical_channel_order();
        let canonical_microphones: Vec<&MicrophonePosition> = order
            .iter()
            .map(|index| &self.config.geometry.microphones[*index])
            .collect();
        let reference_index = canonical_microphones
            .iter()
            .position(|microphone| microphone.id == self.config.geometry.reference_microphone_id)
            .ok_or("canonical microphone order lost the reference microphone")?;
        let calibrated = calibrate_channels(input, &order, &canonical_microphones)?;

        let mut inactive_microphone_ids = Vec::new();
        let mut active = Vec::new();
        for (index, channel) in calibrated.iter().enumerate() {
            if rms(channel) >= self.config.inactive_channel_rms {
                active.push(index);
            } else {
                inactive_microphone_ids.push(canonical_microphones[index].id.clone());
            }
        }
        if !active.contains(&reference_index) {
            return Err("reference microphone is inactive; enhancement fails closed".into());
        }

        let calibrated_audio = Audio {
            sample_rate: input.sample_rate,
            channels: calibrated,
            bits_per_sample: input.bits_per_sample,
            sample_format: input.sample_format,
            channel_mask: None,
        };
        let mut restoration = RestorationConfig {
            mode: RestorationMode::Apply,
            operations: vec![RestorationOperation::Dereverb],
            dereverb: WpeConfig {
                minimum_confidence: 0.0,
                channel_mode: WpeChannelMode::Multichannel,
                frame_size: self.config.frame_size,
                hop_size: self.config.hop_size,
                prediction_delay_frames: self.config.wpe_prediction_delay_frames,
                prediction_taps: self.config.wpe_prediction_taps,
                iterations: self.config.wpe_iterations,
                regularization: self.config.diagonal_loading.max(1.0e-8),
                maximum_attenuation_db: 12.0,
            },
            ..RestorationConfig::default()
        };
        // The array contract owns geometry and channel semantics. Keep WPE's
        // generic confidence gate open but retain every numerical bound.
        restoration.dereverb.minimum_confidence = 0.0;
        let dereverberated = restore_audio(&calibrated_audio, &restoration)?.audio;
        let beamformed = mask_mvdr(
            &dereverberated.channels,
            &canonical_microphones,
            reference_index,
            &active,
            &self.config,
        )?;
        if beamformed.output.len() != frames {
            return Err("microphone-array synthesis changed output duration".into());
        }
        let mut clipped_samples = 0_u64;
        let mut non_finite_samples = 0_u64;
        let output: Vec<f64> = beamformed
            .output
            .into_iter()
            .map(|sample| {
                if !sample.is_finite() {
                    non_finite_samples = non_finite_samples.saturating_add(1);
                    0.0
                } else {
                    let clipped = sample.clamp(-self.config.maximum_peak, self.config.maximum_peak);
                    clipped_samples += u64::from(clipped != sample);
                    clipped
                }
            })
            .collect();
        if non_finite_samples != 0 {
            return Err("microphone-array processing produced non-finite output".into());
        }
        let audio = Audio {
            sample_rate: input.sample_rate,
            channels: vec![output],
            bits_per_sample: input.bits_per_sample,
            sample_format: input.sample_format,
            channel_mask: None,
        };
        let report = MicrophoneArrayReport {
            schema: MICROPHONE_ARRAY_REPORT_SCHEMA.into(),
            schema_version: MICROPHONE_ARRAY_SCHEMA_VERSION,
            implementation: IMPLEMENTATION_ID.into(),
            configuration_sha256: self.config.digest()?,
            evidence_signing_key_id: self.evidence_signing_key_id.clone(),
            evidence_evaluation_result_sha256: self.evidence_evaluation_result_sha256.clone(),
            input_pcm_sha256: digest_audio(INPUT_PCM_DIGEST_DOMAIN, input),
            output_pcm_sha256: digest_audio(OUTPUT_PCM_DIGEST_DOMAIN, &audio),
            sample_rate: input.sample_rate,
            input_channels: input.channels(),
            input_frames: frames,
            output_channels: 1,
            output_frames: audio.frames(),
            reference_microphone_id: self.config.geometry.reference_microphone_id.clone(),
            canonical_microphone_ids: canonical_microphones
                .iter()
                .map(|microphone| microphone.id.clone())
                .collect(),
            active_microphones: active.len(),
            inactive_microphone_ids,
            frame_size: self.config.frame_size,
            hop_size: self.config.hop_size,
            algorithmic_latency_milliseconds: self.config.algorithmic_latency_milliseconds(),
            solved_frequency_bins: beamformed.solved_bins,
            fallback_frequency_bins: beamformed.fallback_bins,
            maximum_observed_condition_number: beamformed.maximum_condition_number,
            clipped_samples,
            non_finite_samples,
            exact_output_duration: audio.frames() == frames,
            paths_recorded: 0,
            limitations: vec![
                "input channels are accepted only as an explicitly declared microphone array".into(),
                "the deterministic baseline uses multichannel WPE and mask-estimated MVDR".into(),
                "ill-conditioned bins and arrays with fewer than two active microphones use the declared reference channel".into(),
                "no neural spatial checkpoint is bundled or promoted".into(),
                "moving-source and streaming claims require separate real-device evidence".into(),
            ],
        };
        report.validate()?;
        Ok(MicrophoneArrayResult { audio, report })
    }
}

struct BeamformResult {
    output: Vec<f64>,
    solved_bins: usize,
    fallback_bins: usize,
    maximum_condition_number: f64,
}

fn mask_mvdr(
    channels: &[Vec<f64>],
    microphones: &[&MicrophonePosition],
    reference_index: usize,
    active: &[usize],
    config: &MicrophoneArrayConfig,
) -> Result<BeamformResult, String> {
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    if frames == 0 {
        return Err("microphone-array input must contain at least one frame".into());
    }
    let stft = Stft::try_new(StftConfig {
        frame_size: config.frame_size,
        hop: config.hop_size,
        window: WindowType::Hann,
        window_params: WindowParams::default(),
    })
    .map_err(|error| format!("construct microphone-array STFT: {error}"))?;
    let pad = config.frame_size / 2;
    let padded_length = frames
        .checked_add(pad.saturating_mul(2))
        .ok_or("microphone-array padded length overflow")?
        .max(config.frame_size);
    let frame_count = 1 + padded_length
        .saturating_sub(config.frame_size)
        .div_ceil(config.hop_size);
    let synthesis_length = (frame_count - 1)
        .checked_mul(config.hop_size)
        .and_then(|value| value.checked_add(config.frame_size))
        .ok_or("microphone-array synthesis length overflow")?;
    let bins = stft.nbins();
    let mut spectra = Vec::new();
    spectra
        .try_reserve_exact(channels.len())
        .map_err(|_| "unable to reserve microphone-array spectra".to_string())?;
    for (channel_index, channel) in channels.iter().enumerate() {
        let mut channel_spectra = vec![Complex::default(); frame_count * bins];
        let phase = -microphones[channel_index]
            .phase_mismatch_degrees
            .to_radians();
        let calibration = Complex::new(phase.cos(), phase.sin());
        let mut time = vec![0.0; config.frame_size];
        let mut spectrum = vec![Complex::default(); config.frame_size];
        for frame_index in 0..frame_count {
            let start = frame_index * config.hop_size;
            for (offset, value) in time.iter_mut().enumerate() {
                let padded_index = start + offset;
                *value = padded_index
                    .checked_sub(pad)
                    .and_then(|source| channel.get(source))
                    .copied()
                    .unwrap_or(0.0);
            }
            stft.analyze(&time, &mut spectrum);
            for bin in 0..bins {
                channel_spectra[frame_index * bins + bin] = complex_mul(spectrum[bin], calibration);
            }
        }
        spectra.push(channel_spectra);
    }

    let mut output_spectra = vec![Complex::default(); frame_count * config.frame_size];
    let mut solved_bins = 0usize;
    let mut fallback_bins = 0usize;
    let mut maximum_condition_number: f64 = 1.0;
    for bin in 0..bins {
        if active.len() < 2 {
            for frame_index in 0..frame_count {
                output_spectra[frame_index * config.frame_size + bin] =
                    spectra[reference_index][frame_index * bins + bin];
            }
            fallback_bins += 1;
            continue;
        }
        let mut powers: Vec<f64> = (0..frame_count)
            .map(|frame_index| complex_norm_sqr(spectra[reference_index][frame_index * bins + bin]))
            .collect();
        powers.sort_by(f64::total_cmp);
        let noise_floor = powers[powers.len() / 5].max(POWER_FLOOR);
        let dimension = active.len();
        let mut speech = vec![Complex::default(); dimension * dimension];
        let mut noise = vec![Complex::default(); dimension * dimension];
        let mut speech_weight_sum = 0.0;
        let mut noise_weight_sum = 0.0;
        for frame_index in 0..frame_count {
            let reference_power =
                complex_norm_sqr(spectra[reference_index][frame_index * bins + bin]);
            let raw = reference_power / (reference_power + noise_floor);
            let speech_weight = raw
                .mul_add(
                    1.0 - config.covariance_smoothing,
                    config.covariance_smoothing * 0.5,
                )
                .clamp(0.02, 0.98);
            let noise_weight = 1.0 - speech_weight;
            speech_weight_sum += speech_weight;
            noise_weight_sum += noise_weight;
            for row in 0..dimension {
                let x_row = spectra[active[row]][frame_index * bins + bin];
                for column in 0..dimension {
                    let x_column = spectra[active[column]][frame_index * bins + bin];
                    let covariance = complex_mul(x_row, complex_conj(x_column));
                    speech[row * dimension + column] = complex_add(
                        speech[row * dimension + column],
                        complex_scale(covariance, speech_weight),
                    );
                    noise[row * dimension + column] = complex_add(
                        noise[row * dimension + column],
                        complex_scale(covariance, noise_weight),
                    );
                }
            }
        }
        for value in &mut speech {
            *value = complex_scale(*value, 1.0 / speech_weight_sum.max(POWER_FLOOR));
        }
        for value in &mut noise {
            *value = complex_scale(*value, 1.0 / noise_weight_sum.max(POWER_FLOOR));
        }
        let trace = (0..dimension)
            .map(|index| noise[index * dimension + index].re.max(0.0))
            .sum::<f64>();
        let loading = config.diagonal_loading * (trace / dimension as f64).max(POWER_FLOOR);
        for index in 0..dimension {
            noise[index * dimension + index].re += loading;
        }
        let active_reference = active.iter().position(|index| *index == reference_index);
        let Some(active_reference) = active_reference else {
            return Err("active microphone set lost the reference microphone".into());
        };
        let steering = principal_vector(&speech, dimension, active_reference);
        let Some((solution, condition)) = solve_complex_system(
            &noise,
            &steering,
            dimension,
            config.maximum_condition_number,
        ) else {
            for frame_index in 0..frame_count {
                output_spectra[frame_index * config.frame_size + bin] =
                    spectra[reference_index][frame_index * bins + bin];
            }
            fallback_bins += 1;
            continue;
        };
        let denominator =
            steering
                .iter()
                .zip(&solution)
                .fold(Complex::default(), |sum, (steering, solution)| {
                    complex_add(sum, complex_mul(complex_conj(*steering), *solution))
                });
        if complex_norm_sqr(denominator) <= POWER_FLOOR || !complex_is_finite(denominator) {
            for frame_index in 0..frame_count {
                output_spectra[frame_index * config.frame_size + bin] =
                    spectra[reference_index][frame_index * bins + bin];
            }
            fallback_bins += 1;
            continue;
        }
        let weights: Vec<Complex> = solution
            .iter()
            .map(|value| complex_div(*value, denominator))
            .collect();
        for frame_index in 0..frame_count {
            let output =
                weights
                    .iter()
                    .zip(active)
                    .fold(Complex::default(), |sum, (weight, channel)| {
                        complex_add(
                            sum,
                            complex_mul(
                                complex_conj(*weight),
                                spectra[*channel][frame_index * bins + bin],
                            ),
                        )
                    });
            output_spectra[frame_index * config.frame_size + bin] = output;
        }
        maximum_condition_number = maximum_condition_number.max(condition);
        solved_bins += 1;
    }
    for frame_index in 0..frame_count {
        for bin in 1..bins.saturating_sub(1) {
            let value = output_spectra[frame_index * config.frame_size + bin];
            output_spectra[frame_index * config.frame_size + config.frame_size - bin] =
                complex_conj(value);
        }
    }
    let mut synthesized = vec![0.0; synthesis_length];
    let mut normalization = vec![0.0; synthesis_length];
    for frame_index in 0..frame_count {
        let offset = frame_index * config.frame_size;
        stft.synthesize(
            &mut output_spectra[offset..offset + config.frame_size],
            &mut synthesized,
            &mut normalization,
            frame_index * config.hop_size,
        );
    }
    for (sample, weight) in synthesized.iter_mut().zip(&normalization) {
        if *weight > POWER_FLOOR {
            *sample /= *weight;
        }
    }
    let end = pad
        .checked_add(frames)
        .ok_or("microphone-array synthesis crop overflow")?;
    if end > synthesized.len() {
        return Err("microphone-array synthesis crop exceeds output".into());
    }
    Ok(BeamformResult {
        output: synthesized[pad..end].to_vec(),
        solved_bins,
        fallback_bins,
        maximum_condition_number,
    })
}

fn principal_vector(matrix: &[Complex], dimension: usize, reference: usize) -> Vec<Complex> {
    let mut vector = vec![Complex::default(); dimension];
    vector[reference] = Complex::new(1.0, 0.0);
    for _ in 0..12 {
        let mut next = vec![Complex::default(); dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                next[row] = complex_add(
                    next[row],
                    complex_mul(matrix[row * dimension + column], vector[column]),
                );
            }
        }
        let norm = next
            .iter()
            .map(|value| complex_norm_sqr(*value))
            .sum::<f64>()
            .sqrt();
        if norm <= POWER_FLOOR || !norm.is_finite() {
            break;
        }
        for value in &mut next {
            *value = complex_scale(*value, 1.0 / norm);
        }
        vector = next;
    }
    let reference_phase = vector[reference];
    if complex_norm_sqr(reference_phase) > POWER_FLOOR {
        let phase = complex_scale(
            reference_phase,
            1.0 / complex_norm_sqr(reference_phase).sqrt(),
        );
        for value in &mut vector {
            *value = complex_mul(*value, complex_conj(phase));
        }
    }
    vector
}

fn solve_complex_system(
    matrix: &[Complex],
    right: &[Complex],
    dimension: usize,
    maximum_condition_number: f64,
) -> Option<(Vec<Complex>, f64)> {
    let mut augmented = vec![Complex::default(); dimension * (dimension + 1)];
    for row in 0..dimension {
        for column in 0..dimension {
            augmented[row * (dimension + 1) + column] = matrix[row * dimension + column];
        }
        augmented[row * (dimension + 1) + dimension] = right[row];
    }
    let mut minimum_pivot = f64::INFINITY;
    let mut maximum_pivot: f64 = 0.0;
    for column in 0..dimension {
        let pivot_row = (column..dimension).max_by(|left, right| {
            complex_norm_sqr(augmented[*left * (dimension + 1) + column]).total_cmp(
                &complex_norm_sqr(augmented[*right * (dimension + 1) + column]),
            )
        })?;
        let pivot_norm = complex_norm_sqr(augmented[pivot_row * (dimension + 1) + column]).sqrt();
        if pivot_norm <= POWER_FLOOR || !pivot_norm.is_finite() {
            return None;
        }
        minimum_pivot = minimum_pivot.min(pivot_norm);
        maximum_pivot = maximum_pivot.max(pivot_norm);
        if maximum_pivot / minimum_pivot > maximum_condition_number {
            return None;
        }
        if pivot_row != column {
            for index in column..=dimension {
                augmented.swap(
                    column * (dimension + 1) + index,
                    pivot_row * (dimension + 1) + index,
                );
            }
        }
        let pivot = augmented[column * (dimension + 1) + column];
        for index in column..=dimension {
            let location = column * (dimension + 1) + index;
            augmented[location] = complex_div(augmented[location], pivot);
        }
        for row in 0..dimension {
            if row == column {
                continue;
            }
            let factor = augmented[row * (dimension + 1) + column];
            for index in column..=dimension {
                let location = row * (dimension + 1) + index;
                augmented[location] = complex_sub(
                    augmented[location],
                    complex_mul(factor, augmented[column * (dimension + 1) + index]),
                );
            }
        }
    }
    let solution = (0..dimension)
        .map(|row| augmented[row * (dimension + 1) + dimension])
        .collect::<Vec<_>>();
    if solution.iter().all(|value| complex_is_finite(*value)) {
        Some((solution, maximum_pivot / minimum_pivot))
    } else {
        None
    }
}

pub fn estimate_microphone_array_memory_bytes(
    config: &MicrophoneArrayConfig,
    frames: usize,
) -> Result<u64, String> {
    let channels = config.geometry.microphones.len();
    let padded = frames
        .checked_add(config.frame_size)
        .ok_or("microphone-array memory frame overflow")?;
    let frame_count = 1 + padded
        .saturating_sub(config.frame_size)
        .div_ceil(config.hop_size.max(1));
    let bins = config.frame_size / 2 + 1;
    let scalar_samples = channels
        .checked_mul(frames)
        .and_then(|value| value.checked_mul(8))
        .ok_or("microphone-array scalar memory overflow")?;
    let spectra = channels
        .checked_mul(frame_count)
        .and_then(|value| value.checked_mul(bins))
        .and_then(|value| value.checked_mul(std::mem::size_of::<Complex>()))
        .ok_or("microphone-array spectrum memory overflow")?;
    let wpe = scalar_samples.saturating_mul(12);
    let total = scalar_samples
        .saturating_mul(3)
        .saturating_add(spectra.saturating_mul(3))
        .saturating_add(wpe)
        .saturating_add(8 * 1024 * 1024);
    let total =
        u64::try_from(total).map_err(|_| "microphone-array memory exceeds u64".to_string())?;
    if total > 2 * 1024 * 1024 * 1024 {
        return Err("microphone-array estimated working set exceeds 2 GiB".into());
    }
    Ok(total)
}

fn calibrate_channels(
    input: &Audio,
    order: &[usize],
    microphones: &[&MicrophonePosition],
) -> Result<Vec<Vec<f64>>, String> {
    let frames = input.frames();
    let mut output = Vec::new();
    output
        .try_reserve_exact(order.len())
        .map_err(|_| "unable to reserve calibrated microphone channels".to_string())?;
    for (canonical_index, input_index) in order.iter().enumerate() {
        let microphone = microphones[canonical_index];
        let gain = 10.0_f64.powf(-microphone.gain_mismatch_db / 20.0);
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|_| "unable to reserve calibrated microphone samples".to_string())?;
        for frame in 0..frames {
            let source = frame as i64 + microphone.sample_skew as i64;
            let sample = if source >= 0 {
                input.channels[*input_index]
                    .get(source as usize)
                    .copied()
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let sample = sample * gain;
            if !sample.is_finite() {
                return Err("microphone calibration produced a non-finite sample".into());
            }
            channel.push(sample.clamp(-1.0, 1.0));
        }
        output.push(channel);
    }
    Ok(output)
}

fn validate_input(input: &Audio, config: &MicrophoneArrayConfig) -> Result<(), String> {
    if input.sample_rate != config.sample_rate {
        return Err(
            "microphone-array input rate does not match the authenticated configuration".into(),
        );
    }
    if input.channels() != config.geometry.microphones.len() {
        return Err("microphone-array input channel count does not match geometry".into());
    }
    if input.frames() == 0 || input.frames() > input.sample_rate as usize * 3_600 {
        return Err("microphone-array input duration must be in (0, 3600] seconds".into());
    }
    if input
        .channels
        .iter()
        .any(|channel| channel.len() != input.frames())
    {
        return Err("microphone-array input channels must be sample-aligned".into());
    }
    if input
        .channels
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err("microphone-array input must be finite normalized PCM".into());
    }
    estimate_microphone_array_memory_bytes(config, input.frames())?;
    Ok(())
}

fn digest_audio(domain: &[u8], audio: &Audio) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(audio.sample_rate.to_le_bytes());
    digest.update((audio.channels() as u64).to_le_bytes());
    digest.update((audio.frames() as u64).to_le_bytes());
    for channel in &audio.channels {
        for sample in channel {
            digest.update(sample.to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f64>() / samples.len() as f64).sqrt()
}

fn complex_add(left: Complex, right: Complex) -> Complex {
    Complex::new(left.re + right.re, left.im + right.im)
}

fn complex_sub(left: Complex, right: Complex) -> Complex {
    Complex::new(left.re - right.re, left.im - right.im)
}

fn complex_mul(left: Complex, right: Complex) -> Complex {
    Complex::new(
        left.re * right.re - left.im * right.im,
        left.re * right.im + left.im * right.re,
    )
}

fn complex_conj(value: Complex) -> Complex {
    Complex::new(value.re, -value.im)
}

fn complex_scale(value: Complex, scale: f64) -> Complex {
    Complex::new(value.re * scale, value.im * scale)
}

fn complex_norm_sqr(value: Complex) -> f64 {
    value.re * value.re + value.im * value.im
}

fn complex_div(numerator: Complex, denominator: Complex) -> Complex {
    let norm = complex_norm_sqr(denominator).max(POWER_FLOOR);
    complex_scale(
        complex_mul(numerator, complex_conj(denominator)),
        1.0 / norm,
    )
}

fn complex_is_finite(value: Complex) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn validate_f64_range(name: &str, value: f64, minimum: f64, maximum: f64) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "microphone-array {name} must be finite and in {minimum}..={maximum}"
        ));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("{label} must be a bounded ASCII identifier"));
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{label} must be a lowercase 64-character SHA-256 hex digest"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(config: &MicrophoneArrayConfig) -> MicrophoneArrayPromotionEvidencePayload {
        MicrophoneArrayPromotionEvidencePayload {
            completed_at_unix_seconds: 1_800_000_000,
            implementation: IMPLEMENTATION_ID.into(),
            implementation_source_revision: "0123456789abcdef".into(),
            implementation_source_sha256: "1".repeat(64),
            configuration_sha256: config.digest().unwrap(),
            corpus_manifest_sha256: "2".repeat(64),
            evaluation_result_sha256: "3".repeat(64),
            listening_result_sha256: "4".repeat(64),
            strata: REQUIRED_STRATA
                .iter()
                .map(|id| MicrophoneArrayEvidenceStratum {
                    id: (*id).into(),
                    cases: 100,
                    si_sdr_improvement_db: 1.0,
                    wer_regression: 0.0,
                    doa_error_degrees: 5.0,
                    reference_coloration_db: 0.1,
                    target_leakage_db: -6.0,
                    non_finite_samples: 0,
                    passed: true,
                })
                .collect(),
            real_meeting_cases: 100,
            unseen_geometry_cases: 100,
            permutation_cases: 100,
            paced_realtime_blocks: 10_000,
            worst_case_realtime_factor: 0.25,
            callback_allocations: 0,
            callback_locks: 0,
            callback_waits: 0,
            deadline_misses: 0,
            listener_count: 20,
            listener_preference: 0.6,
            accepted: true,
        }
    }

    fn session(config: &MicrophoneArrayConfig) -> MicrophoneArraySession {
        let (secret, public) = crate::generate_receipt_keypair().unwrap();
        let evidence = sign_microphone_array_promotion_evidence(evidence(config), &secret).unwrap();
        MicrophoneArraySession::prepare(&evidence, &public, config.clone()).unwrap()
    }

    fn fixture(config: &MicrophoneArrayConfig, frames: usize) -> Audio {
        let mut channels = vec![vec![0.0; frames]; config.geometry.microphones.len()];
        for frame in 0..frames {
            let clean = (2.0 * std::f64::consts::PI * 440.0 * frame as f64
                / config.sample_rate as f64)
                .sin()
                * 0.2;
            let noise = (2.0 * std::f64::consts::PI * 173.0 * frame as f64
                / config.sample_rate as f64)
                .sin()
                * 0.03;
            channels[0][frame] = clean + noise;
            channels[1][frame] = clean - noise;
        }
        Audio {
            sample_rate: config.sample_rate,
            channels,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        }
    }

    #[test]
    fn geometry_rejects_duplicates_and_program_stereo_has_no_representation() {
        let mut config = MicrophoneArrayConfig::default();
        config.geometry.microphones[1].x = config.geometry.microphones[0].x;
        assert!(config.validate().is_err());
    }

    #[test]
    fn signed_evidence_binds_canonical_geometry() {
        let config = MicrophoneArrayConfig::default();
        let mut permuted = config.clone();
        permuted.geometry.microphones.swap(0, 1);
        assert_eq!(config.digest().unwrap(), permuted.digest().unwrap());
        let (secret, public) = crate::generate_receipt_keypair().unwrap();
        let signed = sign_microphone_array_promotion_evidence(evidence(&config), &secret).unwrap();
        MicrophoneArraySession::prepare(&signed, &public, permuted).unwrap();
    }

    #[test]
    fn evidence_rejects_unbounded_or_inconsistent_counts() {
        let config = MicrophoneArrayConfig::default();
        let mut payload = evidence(&config);
        payload.real_meeting_cases = 1_000_001;
        assert!(payload.validate().is_err());
        let mut payload = evidence(&config);
        payload.paced_realtime_blocks = JSON_SAFE_INTEGER + 1;
        assert!(payload.validate().is_err());
        let mut payload = evidence(&config);
        payload.strata[0].non_finite_samples = JSON_SAFE_INTEGER + 1;
        payload.strata[0].passed = false;
        payload.accepted = false;
        assert!(payload.validate().is_err());
    }

    #[test]
    fn enhancement_is_exact_duration_finite_and_path_free() {
        let config = MicrophoneArrayConfig::default();
        let input = fixture(&config, 3_211);
        let result = session(&config).enhance(&input).unwrap();
        assert_eq!(result.audio.channels(), 1);
        assert_eq!(result.audio.frames(), input.frames());
        assert!(result.audio.channels[0]
            .iter()
            .all(|sample| sample.is_finite()));
        assert!(result.report.exact_output_duration);
        assert_eq!(result.report.paths_recorded, 0);
        result.report.validate().unwrap();
        let mut invalid = result.report.clone();
        invalid.inactive_microphone_ids = vec![invalid.reference_microphone_id.clone()];
        invalid.active_microphones = invalid.input_channels - 1;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn channel_and_geometry_permutation_is_equivalent() {
        let config = MicrophoneArrayConfig::default();
        let input = fixture(&config, 2_048);
        let first = session(&config).enhance(&input).unwrap();
        let mut permuted_config = config.clone();
        permuted_config.geometry.microphones.swap(0, 1);
        let mut permuted_input = input.clone();
        permuted_input.channels.swap(0, 1);
        let second = session(&permuted_config).enhance(&permuted_input).unwrap();
        let maximum_delta = first.audio.channels[0]
            .iter()
            .zip(&second.audio.channels[0])
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_delta <= 1.0e-10,
            "permutation delta {maximum_delta}"
        );
    }

    #[test]
    fn inactive_reference_fails_closed() {
        let config = MicrophoneArrayConfig::default();
        let mut input = fixture(&config, 1_024);
        input.channels[0].fill(0.0);
        assert!(session(&config).enhance(&input).is_err());
    }

    #[test]
    fn complex_solver_rejects_singular_systems() {
        let matrix = vec![Complex::default(); 4];
        let right = vec![Complex::new(1.0, 0.0); 2];
        assert!(solve_complex_system(&matrix, &right, 2, 1.0e6).is_none());
    }
}
