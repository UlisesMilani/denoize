//! Fail-closed orchestration and evidence contracts for universal speech
//! restoration.
//!
//! The runtime deliberately separates a candidate render from the published
//! output. A signed model package v2 is authenticated and numerically checked
//! while preparing the BSRNN session. Clean inputs bypass inference, and a
//! candidate that violates geometry, finite-sample, energy, peak, clipping,
//! silence-injection, or native-quality gates is discarded in favour of the
//! bit-exact input. These signal gates cannot prove word, phoneme, prosody, or
//! speaker fidelity; promotion therefore uses a separately signed, stratified
//! evidence document.

use crate::audio::{estimate_audio_memory_bytes, Audio};
#[cfg(feature = "bsrnn")]
use crate::diagnostics::{diagnose_audio, DiagnosticOptions, DiagnosticReport};
use crate::execution::{ReceiptPublicKey, ReceiptSecretKey, ReceiptSignature};
#[cfg(feature = "bsrnn")]
use crate::{Backend, BackendSession, DenoiserConfig};
use serde::{Deserialize, Serialize};
#[cfg(feature = "bsrnn")]
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::Path;

pub const UNIVERSAL_RESTORATION_REPORT_SCHEMA: &str = "denoize-universal-restoration-report-v1";
pub const UNIVERSAL_RESTORATION_MASK_SCHEMA: &str = "denoize-universal-restoration-mask-v1";
pub const UNIVERSAL_PROMOTION_EVIDENCE_SCHEMA: &str = "denoize-universal-promotion-evidence-v1";
pub const UNIVERSAL_RESTORATION_SCHEMA_VERSION: u32 = 1;
pub const MAX_UNIVERSAL_MASK_RUNS: usize = 4_000_000;
pub const MAX_UNIVERSAL_EVIDENCE_STRATA: usize = 256;
pub const MAX_UNIVERSAL_EVIDENCE_METRICS: usize = 64;

const MAX_CHANNELS: usize = 64;
const MAX_EVIDENCE_JSON_BYTES: u64 = 16 * 1024 * 1024;
const PROMOTION_SIGNATURE_DOMAIN: &[u8] = b"denoize-universal-promotion-evidence-v1";
#[cfg(feature = "bsrnn")]
const PCM_DIGEST_DOMAIN: &[u8] = b"denoize-universal-restoration-pcm-v1\0";
#[cfg(feature = "bsrnn")]
const SILENCE_FLOOR: f64 = 1e-12;

const REQUIRED_STRATA: &[&str] = &[
    "accent",
    "age",
    "clean-bypass",
    "degradation-additive-noise",
    "degradation-bandwidth-limitation",
    "degradation-clipping",
    "degradation-codec-distortion",
    "degradation-packet-loss",
    "degradation-reverberation",
    "degradation-wind",
    "emotion",
    "language",
    "near-clean-bypass",
    "non-speech",
    "seen-corpus",
    "sex",
    "singing",
    "speech",
    "unseen-corpus",
    "whisper",
];

const REQUIRED_METRICS: &[&str] = &[
    "content.phoneme-similarity-delta",
    "content.word-error-rate-delta",
    "hallucination.new-word-rate",
    "objective.si-sdr-improvement-db",
    "output.duration-error-frames",
    "output.non-finite-samples",
    "perceptual.quality-delta",
    "performance.realtime-factor",
    "speaker.similarity-delta",
];

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UniversalModelFamily {
    #[default]
    Discriminative,
    Hybrid,
    Generative,
}

impl UniversalModelFamily {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "discriminative" | "safe" => Some(Self::Discriminative),
            "hybrid" => Some(Self::Hybrid),
            "generative" | "generation" => Some(Self::Generative),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UniversalRenderRole {
    #[default]
    Primary,
    Alternate,
}

impl UniversalRenderRole {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "primary" | "default" => Some(Self::Primary),
            "alternate" | "comparison" => Some(Self::Alternate),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UniversalDegradation {
    AdditiveNoise,
    Reverberation,
    Clipping,
    BandwidthLimitation,
    CodecDistortion,
    PacketLoss,
    Wind,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UniversalRestorationDecision {
    BypassedClean,
    Accepted,
    RejectedSafetyGate,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalRestorationConfig {
    pub model_family: UniversalModelFamily,
    pub render_role: UniversalRenderRole,
    pub allow_experimental: bool,
    pub analysis_seconds: u32,
    pub minimum_degradation_score: f64,
    pub maximum_energy_gain_db: f64,
    pub maximum_peak_gain_db: f64,
    pub maximum_new_clipping_ratio: f64,
    pub maximum_quality_score_regression: f64,
}

impl Default for UniversalRestorationConfig {
    fn default() -> Self {
        Self {
            model_family: UniversalModelFamily::Discriminative,
            render_role: UniversalRenderRole::Primary,
            allow_experimental: false,
            analysis_seconds: 12,
            minimum_degradation_score: 0.08,
            maximum_energy_gain_db: 6.0,
            maximum_peak_gain_db: 6.0,
            maximum_new_clipping_ratio: 0.0001,
            maximum_quality_score_regression: 5.0,
        }
    }
}

impl UniversalRestorationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=60).contains(&self.analysis_seconds) {
            return Err("universal analysis_seconds must be in 1..=60".into());
        }
        validate_range(
            "minimum_degradation_score",
            self.minimum_degradation_score,
            0.0,
            1.0,
        )?;
        validate_range(
            "maximum_energy_gain_db",
            self.maximum_energy_gain_db,
            0.0,
            24.0,
        )?;
        validate_range("maximum_peak_gain_db", self.maximum_peak_gain_db, 0.0, 24.0)?;
        validate_range(
            "maximum_new_clipping_ratio",
            self.maximum_new_clipping_ratio,
            0.0,
            0.1,
        )?;
        validate_range(
            "maximum_quality_score_regression",
            self.maximum_quality_score_regression,
            0.0,
            25.0,
        )?;
        if self.model_family != UniversalModelFamily::Discriminative
            && (!self.allow_experimental || self.render_role != UniversalRenderRole::Alternate)
        {
            return Err(
                "hybrid and generative universal models require allow_experimental=true and render_role=alternate"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalDegradationEvidence {
    pub degradation: UniversalDegradation,
    pub detected: bool,
    pub confidence: f64,
    pub severity: f64,
    pub score: f64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalModelIdentity {
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UniversalSafetyGateKind {
    Geometry,
    FiniteSamples,
    EnergyGain,
    PeakGain,
    NewClipping,
    SilenceInjection,
    NativeQualityRegression,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalSafetyGate {
    pub kind: UniversalSafetyGateKind,
    pub observed: f64,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalSafetyMeasurements {
    pub input_rms_dbfs: f64,
    pub candidate_rms_dbfs: Option<f64>,
    pub input_peak_dbfs: f64,
    pub candidate_peak_dbfs: Option<f64>,
    pub energy_delta_db: Option<f64>,
    pub input_clipping_ratio: f64,
    pub candidate_clipping_ratio: Option<f64>,
    pub native_quality_score_delta: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UniversalMaskState {
    Untouched,
    Replaced,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalMaskRun {
    pub channel: usize,
    pub start_frame: usize,
    pub frame_count: usize,
    pub state: UniversalMaskState,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalRestorationMask {
    pub schema: String,
    pub schema_version: u32,
    pub channels: usize,
    pub frames: usize,
    pub runs: Vec<UniversalMaskRun>,
}

impl UniversalRestorationMask {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != UNIVERSAL_RESTORATION_MASK_SCHEMA
            || self.schema_version != UNIVERSAL_RESTORATION_SCHEMA_VERSION
        {
            return Err("unsupported universal restoration mask schema".into());
        }
        if self.channels == 0 || self.channels > MAX_CHANNELS {
            return Err(format!(
                "universal restoration mask channels must be in 1..={MAX_CHANNELS}"
            ));
        }
        if self.runs.len() > MAX_UNIVERSAL_MASK_RUNS {
            return Err(format!(
                "universal restoration mask exceeds the {MAX_UNIVERSAL_MASK_RUNS}-run limit"
            ));
        }
        let mut cursor = vec![0usize; self.channels];
        let mut previous = None;
        for run in &self.runs {
            if run.channel >= self.channels || run.frame_count == 0 {
                return Err("universal restoration mask run has invalid geometry".into());
            }
            let position = (run.channel, run.start_frame);
            if previous.is_some_and(|value| value >= position) {
                return Err("universal restoration mask runs are not strictly ordered".into());
            }
            previous = Some(position);
            if run.start_frame != cursor[run.channel]
                || run.start_frame.saturating_add(run.frame_count) > self.frames
            {
                return Err("universal restoration mask does not provide exact coverage".into());
            }
            cursor[run.channel] = cursor[run.channel]
                .checked_add(run.frame_count)
                .ok_or("universal restoration mask coverage overflows")?;
        }
        if cursor.iter().any(|covered| *covered != self.frames) {
            return Err("universal restoration mask omits channel frames".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalRestorationReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub network_accessed: bool,
    pub model_family: UniversalModelFamily,
    pub render_role: UniversalRenderRole,
    pub model: UniversalModelIdentity,
    pub decision: UniversalRestorationDecision,
    pub model_invoked: bool,
    pub candidate_accepted: bool,
    pub deterministic: bool,
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: usize,
    pub input_pcm_sha256: String,
    pub candidate_pcm_sha256: Option<String>,
    pub output_pcm_sha256: String,
    pub mask_sha256: String,
    pub changed_samples: usize,
    pub degradations: Vec<UniversalDegradationEvidence>,
    pub measurements: UniversalSafetyMeasurements,
    pub safety_gates: Vec<UniversalSafetyGate>,
    pub semantic_fidelity_assessed: bool,
    pub speaker_identity_assessed: bool,
    pub promotion_evidence_verified: bool,
    pub limitations: Vec<String>,
    pub warnings: Vec<String>,
}

impl UniversalRestorationReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|error| format!("serialize universal restoration report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize universal restoration report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct UniversalRestorationResult {
    pub audio: Audio,
    pub report: UniversalRestorationReport,
    pub mask: UniversalRestorationMask,
}

/// Conservative decoded-audio, candidate, diagnostic, and worst-case RLE
/// allowance. Model session memory is admitted separately from its signed
/// package resource contract.
pub fn estimate_universal_restoration_memory_bytes(audio: &Audio) -> u64 {
    let base = estimate_audio_memory_bytes(audio);
    let samples = (audio.channels() as u64).saturating_mul(audio.frames() as u64);
    let runs = samples
        .min(MAX_UNIVERSAL_MASK_RUNS as u64)
        .saturating_mul(std::mem::size_of::<UniversalMaskRun>() as u64);
    base.saturating_mul(7)
        .saturating_add(samples)
        .saturating_add(runs)
        .max(1024 * 1024)
}

#[cfg(feature = "bsrnn")]
pub fn restore_universal_audio(
    input: &Audio,
    session: &BackendSession,
    config: &UniversalRestorationConfig,
) -> Result<UniversalRestorationResult, String> {
    config.validate()?;
    validate_audio(input)?;
    if session.backend() != Backend::Bsrnn {
        return Err("universal restoration requires the dedicated bsrnn backend".into());
    }
    if !session.options().deterministic {
        return Err("universal restoration requires deterministic backend processing".into());
    }
    let package = session.options().runtime_package.as_ref().ok_or(
        "universal restoration requires an authenticated runtime model package v2; raw ONNX is not accepted",
    )?;
    let manifest = package
        .manifest_v2()
        .ok_or("universal restoration rejects runtime model package v1")?;
    let profile = package
        .precision_profile_for(session.accelerator().effective())?
        .expect("v2 packages always select a precision profile");
    let model = UniversalModelIdentity {
        package_sha256: package.package_sha256().into(),
        public_key_sha256: package.public_key_sha256().into(),
        package_id: manifest.package_id.clone(),
        package_revision: manifest.package_revision.clone(),
        precision_profile: profile.id.clone(),
        source_revision: manifest.provenance.source_revision.clone(),
        source_sha256: manifest.provenance.source_sha256.clone(),
        source_license_spdx: manifest.provenance.source_license_spdx.clone(),
        checkpoint_sha256: manifest.provenance.checkpoint_sha256.clone(),
        checkpoint_license_spdx: manifest.provenance.checkpoint_license_spdx.clone(),
        accelerator: session.accelerator().effective().name().into(),
    };

    let diagnostic_options =
        DiagnosticOptions::new().with_analysis_seconds(config.analysis_seconds);
    let baseline = diagnose_audio(input, diagnostic_options)?;
    let degradations = degradation_evidence(&baseline);
    let should_invoke = degradations
        .iter()
        .any(|item| item.detected && item.score >= config.minimum_degradation_score);
    let input_digest = pcm_digest(input);
    let input_measurements = signal_measurements(input);

    if !should_invoke {
        let output = input.try_clone_fallible("universal clean bypass")?;
        let mask = encode_mask(input, &output)?;
        let mask_sha256 = json_sha256(&mask)?;
        return Ok(UniversalRestorationResult {
            report: UniversalRestorationReport {
                schema: UNIVERSAL_RESTORATION_REPORT_SCHEMA.into(),
                schema_version: UNIVERSAL_RESTORATION_SCHEMA_VERSION,
                denoize_version: env!("CARGO_PKG_VERSION").into(),
                network_accessed: false,
                model_family: config.model_family,
                render_role: config.render_role,
                model,
                decision: UniversalRestorationDecision::BypassedClean,
                model_invoked: false,
                candidate_accepted: false,
                deterministic: true,
                sample_rate: input.sample_rate,
                channels: input.channels(),
                frames: input.frames(),
                input_pcm_sha256: input_digest.clone(),
                candidate_pcm_sha256: None,
                output_pcm_sha256: input_digest,
                mask_sha256,
                changed_samples: 0,
                degradations,
                measurements: UniversalSafetyMeasurements {
                    input_rms_dbfs: input_measurements.rms_dbfs,
                    candidate_rms_dbfs: None,
                    input_peak_dbfs: input_measurements.peak_dbfs,
                    candidate_peak_dbfs: None,
                    energy_delta_db: None,
                    input_clipping_ratio: input_measurements.clipping_ratio,
                    candidate_clipping_ratio: None,
                    native_quality_score_delta: None,
                },
                safety_gates: Vec::new(),
                semantic_fidelity_assessed: false,
                speaker_identity_assessed: false,
                promotion_evidence_verified: false,
                limitations: limitations(),
                warnings: vec![
                    "no supported degradation crossed the conservative invocation threshold; output is bit-exact input"
                        .into(),
                ],
            },
            audio: output,
            mask,
        });
    }

    let candidate_channels = session.process(
        &input.channels,
        input.sample_rate,
        &DenoiserConfig::default(input.sample_rate),
    )?;
    let candidate = Audio {
        sample_rate: input.sample_rate,
        channels: candidate_channels,
        bits_per_sample: input.bits_per_sample,
        sample_format: input.sample_format,
        channel_mask: input.channel_mask,
    };
    let geometry_passed = same_geometry(input, &candidate);
    let finite_passed = candidate
        .channels
        .iter()
        .flatten()
        .all(|sample| sample.is_finite() && (-1.0..=1.0).contains(sample));
    let candidate_measurements = signal_measurements(&candidate);
    let candidate_diagnosis = if geometry_passed && finite_passed {
        Some(diagnose_audio(&candidate, diagnostic_options)?)
    } else {
        None
    };
    let energy_gain = candidate_measurements.rms_dbfs - input_measurements.rms_dbfs;
    let peak_gain = candidate_measurements.peak_dbfs - input_measurements.peak_dbfs;
    let new_clipping =
        (candidate_measurements.clipping_ratio - input_measurements.clipping_ratio).max(0.0);
    let quality_delta = candidate_diagnosis
        .as_ref()
        .map(|report| report.quality.score - baseline.quality.score);
    let silence_limit = if input_measurements.rms_dbfs <= -55.0 {
        (input_measurements.rms_dbfs + 6.0).min(-45.0)
    } else {
        0.0
    };
    let silence_passed =
        input_measurements.rms_dbfs > -55.0 || candidate_measurements.rms_dbfs <= silence_limit;
    let quality_passed =
        quality_delta.is_some_and(|delta| delta >= -config.maximum_quality_score_regression);
    let safety_gates = vec![
        gate(
            UniversalSafetyGateKind::Geometry,
            if geometry_passed { 1.0 } else { 0.0 },
            1.0,
            geometry_passed,
        ),
        gate(
            UniversalSafetyGateKind::FiniteSamples,
            if finite_passed { 1.0 } else { 0.0 },
            1.0,
            finite_passed,
        ),
        gate(
            UniversalSafetyGateKind::EnergyGain,
            energy_gain,
            config.maximum_energy_gain_db,
            energy_gain <= config.maximum_energy_gain_db,
        ),
        gate(
            UniversalSafetyGateKind::PeakGain,
            peak_gain,
            config.maximum_peak_gain_db,
            peak_gain <= config.maximum_peak_gain_db,
        ),
        gate(
            UniversalSafetyGateKind::NewClipping,
            new_clipping,
            config.maximum_new_clipping_ratio,
            new_clipping <= config.maximum_new_clipping_ratio,
        ),
        gate(
            UniversalSafetyGateKind::SilenceInjection,
            candidate_measurements.rms_dbfs,
            silence_limit,
            silence_passed,
        ),
        gate(
            UniversalSafetyGateKind::NativeQualityRegression,
            quality_delta.unwrap_or(f64::NEG_INFINITY),
            -config.maximum_quality_score_regression,
            quality_passed,
        ),
    ];
    let accepted = safety_gates.iter().all(|item| item.passed);
    let candidate_digest = pcm_digest(&candidate);
    let output = if accepted {
        candidate
    } else {
        input.try_clone_fallible("universal safety-gate rollback")?
    };
    let output_digest = if accepted {
        candidate_digest.clone()
    } else {
        input_digest.clone()
    };
    let mask = encode_mask(input, &output)?;
    let changed_samples = mask
        .runs
        .iter()
        .filter(|run| run.state == UniversalMaskState::Replaced)
        .map(|run| run.frame_count)
        .sum();
    let mut warnings = Vec::new();
    if !accepted {
        let failed = safety_gates
            .iter()
            .filter(|item| !item.passed)
            .map(|item| format!("{:?}", item.kind).to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(", ");
        warnings.push(format!(
            "candidate failed safety gates ({failed}); published output is bit-exact input"
        ));
    }
    if config.model_family != UniversalModelFamily::Discriminative {
        warnings.push(
            "hybrid/generative result is an explicitly experimental alternate render and is not eligible for silent primary selection"
                .into(),
        );
    }
    let report = UniversalRestorationReport {
        schema: UNIVERSAL_RESTORATION_REPORT_SCHEMA.into(),
        schema_version: UNIVERSAL_RESTORATION_SCHEMA_VERSION,
        denoize_version: env!("CARGO_PKG_VERSION").into(),
        network_accessed: false,
        model_family: config.model_family,
        render_role: config.render_role,
        model,
        decision: if accepted {
            UniversalRestorationDecision::Accepted
        } else {
            UniversalRestorationDecision::RejectedSafetyGate
        },
        model_invoked: true,
        candidate_accepted: accepted,
        deterministic: true,
        sample_rate: input.sample_rate,
        channels: input.channels(),
        frames: input.frames(),
        input_pcm_sha256: input_digest,
        candidate_pcm_sha256: Some(candidate_digest),
        output_pcm_sha256: output_digest,
        mask_sha256: json_sha256(&mask)?,
        changed_samples,
        degradations,
        measurements: UniversalSafetyMeasurements {
            input_rms_dbfs: input_measurements.rms_dbfs,
            candidate_rms_dbfs: Some(candidate_measurements.rms_dbfs),
            input_peak_dbfs: input_measurements.peak_dbfs,
            candidate_peak_dbfs: Some(candidate_measurements.peak_dbfs),
            energy_delta_db: Some(energy_gain),
            input_clipping_ratio: input_measurements.clipping_ratio,
            candidate_clipping_ratio: Some(candidate_measurements.clipping_ratio),
            native_quality_score_delta: quality_delta,
        },
        safety_gates,
        semantic_fidelity_assessed: false,
        speaker_identity_assessed: false,
        promotion_evidence_verified: false,
        limitations: limitations(),
        warnings,
    };
    Ok(UniversalRestorationResult {
        audio: output,
        report,
        mask,
    })
}

#[cfg(feature = "bsrnn")]
#[derive(Clone, Copy)]
struct SignalMeasurements {
    rms_dbfs: f64,
    peak_dbfs: f64,
    clipping_ratio: f64,
}

#[cfg(feature = "bsrnn")]
fn signal_measurements(audio: &Audio) -> SignalMeasurements {
    let mut energy = 0.0;
    let mut peak = 0.0_f64;
    let mut clipping = 0usize;
    let mut samples = 0usize;
    for sample in audio.channels.iter().flatten() {
        samples = samples.saturating_add(1);
        if sample.is_finite() {
            energy += sample * sample;
            peak = peak.max(sample.abs());
            clipping = clipping.saturating_add(usize::from(sample.abs() >= 0.999));
        }
    }
    let rms = (energy / samples.max(1) as f64).sqrt();
    SignalMeasurements {
        rms_dbfs: amplitude_db(rms),
        peak_dbfs: amplitude_db(peak),
        clipping_ratio: clipping as f64 / samples.max(1) as f64,
    }
}

#[cfg(feature = "bsrnn")]
fn amplitude_db(value: f64) -> f64 {
    (20.0 * value.max(SILENCE_FLOOR).log10()).clamp(-240.0, 24.0)
}

#[cfg(feature = "bsrnn")]
fn degradation_evidence(report: &DiagnosticReport) -> Vec<UniversalDegradationEvidence> {
    let mapping = [
        ("additive-noise", UniversalDegradation::AdditiveNoise),
        ("reverberation", UniversalDegradation::Reverberation),
        ("clipping", UniversalDegradation::Clipping),
        (
            "bandwidth-limitation",
            UniversalDegradation::BandwidthLimitation,
        ),
        ("codec-distortion", UniversalDegradation::CodecDistortion),
        ("packet-loss-or-dropout", UniversalDegradation::PacketLoss),
        ("wind-or-plosive", UniversalDegradation::Wind),
    ];
    mapping
        .into_iter()
        .map(|(kind, degradation)| {
            let finding = report
                .findings
                .iter()
                .find(|finding| finding.kind == kind)
                .expect("native diagnosis always reports every supported degradation");
            UniversalDegradationEvidence {
                degradation,
                detected: finding.detected,
                confidence: finding.confidence,
                severity: finding.severity,
                score: (finding.confidence * finding.severity).clamp(0.0, 1.0),
            }
        })
        .collect()
}

#[cfg(feature = "bsrnn")]
fn gate(
    kind: UniversalSafetyGateKind,
    observed: f64,
    limit: f64,
    passed: bool,
) -> UniversalSafetyGate {
    UniversalSafetyGate {
        kind,
        observed: if observed.is_finite() {
            observed
        } else {
            -240.0
        },
        limit,
        passed,
    }
}

#[cfg(any(feature = "bsrnn", test))]
fn same_geometry(input: &Audio, output: &Audio) -> bool {
    input.sample_rate == output.sample_rate
        && input.channels() == output.channels()
        && input.frames() == output.frames()
        && input.channel_mask == output.channel_mask
        && output
            .channels
            .iter()
            .all(|channel| channel.len() == input.frames())
}

#[cfg(any(feature = "bsrnn", test))]
fn encode_mask(input: &Audio, output: &Audio) -> Result<UniversalRestorationMask, String> {
    if !same_geometry(input, output) {
        return Err("cannot encode universal mask for mismatched audio geometry".into());
    }
    let mut runs = Vec::new();
    for (channel_index, (before, after)) in input.channels.iter().zip(&output.channels).enumerate()
    {
        if before.is_empty() {
            continue;
        }
        let mut start = 0usize;
        while start < before.len() {
            let changed = before[start].to_bits() != after[start].to_bits();
            let mut end = start + 1;
            while end < before.len() && (before[end].to_bits() != after[end].to_bits()) == changed {
                end += 1;
            }
            if runs.len() >= MAX_UNIVERSAL_MASK_RUNS {
                return Err(format!(
                    "universal restoration mask exceeds the {MAX_UNIVERSAL_MASK_RUNS}-run limit"
                ));
            }
            runs.try_reserve(1)
                .map_err(|_| "unable to reserve universal mask run".to_string())?;
            runs.push(UniversalMaskRun {
                channel: channel_index,
                start_frame: start,
                frame_count: end - start,
                state: if changed {
                    UniversalMaskState::Replaced
                } else {
                    UniversalMaskState::Untouched
                },
            });
            start = end;
        }
    }
    let mask = UniversalRestorationMask {
        schema: UNIVERSAL_RESTORATION_MASK_SCHEMA.into(),
        schema_version: UNIVERSAL_RESTORATION_SCHEMA_VERSION,
        channels: input.channels(),
        frames: input.frames(),
        runs,
    };
    mask.validate()?;
    Ok(mask)
}

#[cfg(any(feature = "bsrnn", test))]
fn validate_audio(audio: &Audio) -> Result<(), String> {
    if audio.sample_rate == 0 || audio.sample_rate > crate::config::MAX_SAMPLE_RATE {
        return Err("universal restoration input sample rate is invalid".into());
    }
    if audio.channels.is_empty() || audio.channels.len() > MAX_CHANNELS {
        return Err(format!(
            "universal restoration channels must be in 1..={MAX_CHANNELS}"
        ));
    }
    let frames = audio.frames();
    if audio.channels.iter().any(|channel| channel.len() != frames) {
        return Err("universal restoration input channels have unequal lengths".into());
    }
    if audio
        .channels
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err("universal restoration input contains an invalid normalized sample".into());
    }
    Ok(())
}

#[cfg(feature = "bsrnn")]
fn pcm_digest(audio: &Audio) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PCM_DIGEST_DOMAIN);
    hasher.update(audio.sample_rate.to_le_bytes());
    hasher.update((audio.channels() as u64).to_le_bytes());
    hasher.update((audio.frames() as u64).to_le_bytes());
    for channel in &audio.channels {
        for sample in channel {
            hasher.update(sample.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(feature = "bsrnn")]
fn json_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("serialize universal restoration evidence for digest: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[cfg(feature = "bsrnn")]
fn limitations() -> Vec<String> {
    vec![
        "native signal gates do not assess word, phoneme, prosody, speaker identity, or factual fidelity"
            .into(),
        "the current spectral adapter resamples through a signed 48000 Hz, 481-bin contract; it does not claim parity with an upstream sample-frequency-independent graph"
            .into(),
        "model quality and redistribution rights require separately signed stratified promotion evidence"
            .into(),
    ]
}

fn validate_range(label: &str, value: f64, minimum: f64, maximum: f64) -> Result<(), String> {
    if !value.is_finite() || value < minimum || value > maximum {
        Err(format!(
            "universal {label} must be finite and in {minimum}..={maximum}"
        ))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UniversalMetricOperator {
    GreaterOrEqual,
    LessOrEqual,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalMetricOutcome {
    pub metric: String,
    pub value: f64,
    pub operator: UniversalMetricOperator,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalStratumEvidence {
    pub id: String,
    pub cases: u32,
    pub metrics: Vec<UniversalMetricOutcome>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniversalPromotionEvidencePayload {
    pub completed_at_unix_seconds: u64,
    pub model_package_sha256: String,
    pub model_family: UniversalModelFamily,
    pub source_revision: String,
    pub source_sha256: String,
    pub checkpoint_sha256: String,
    pub corpus_manifest_sha256: String,
    pub evaluation_result_sha256: String,
    pub strata: Vec<UniversalStratumEvidence>,
    pub minimum_listeners: u32,
    pub listener_count: u32,
    pub listener_preference: f64,
    pub listener_preference_limit: f64,
    pub accepted: bool,
}

impl UniversalPromotionEvidencePayload {
    pub fn validate(&self) -> Result<(), String> {
        for (label, digest) in [
            ("model package", self.model_package_sha256.as_str()),
            ("source", self.source_sha256.as_str()),
            ("checkpoint", self.checkpoint_sha256.as_str()),
            ("corpus manifest", self.corpus_manifest_sha256.as_str()),
            ("evaluation result", self.evaluation_result_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
        validate_identifier("source revision", &self.source_revision)?;
        if self.completed_at_unix_seconds > (1_u64 << 53) - 1 {
            return Err("universal evidence timestamp exceeds the JSON safe-integer limit".into());
        }
        if self.strata.is_empty() || self.strata.len() > MAX_UNIVERSAL_EVIDENCE_STRATA {
            return Err(format!(
                "universal evidence must contain 1..={MAX_UNIVERSAL_EVIDENCE_STRATA} strata"
            ));
        }
        let mut previous = None;
        let mut observed_strata = BTreeSet::new();
        let mut all_metrics_passed = true;
        for stratum in &self.strata {
            validate_identifier("universal evidence stratum", &stratum.id)?;
            if previous.is_some_and(|value: &str| value >= stratum.id.as_str()) {
                return Err("universal evidence strata must be unique and strictly sorted".into());
            }
            previous = Some(&stratum.id);
            observed_strata.insert(stratum.id.as_str());
            if stratum.cases == 0 || stratum.cases > 1_000_000 {
                return Err("universal evidence stratum cases must be in 1..=1000000".into());
            }
            if stratum.metrics.is_empty() || stratum.metrics.len() > MAX_UNIVERSAL_EVIDENCE_METRICS
            {
                return Err(format!(
                    "universal evidence stratum metrics must be in 1..={MAX_UNIVERSAL_EVIDENCE_METRICS}"
                ));
            }
            let mut previous_metric = None;
            let mut observed_metrics = BTreeSet::new();
            for metric in &stratum.metrics {
                validate_identifier("universal evidence metric", &metric.metric)?;
                if previous_metric.is_some_and(|value: &str| value >= metric.metric.as_str()) {
                    return Err(
                        "universal evidence metrics must be unique and strictly sorted".into(),
                    );
                }
                previous_metric = Some(&metric.metric);
                observed_metrics.insert(metric.metric.as_str());
                if !metric.value.is_finite() || !metric.limit.is_finite() {
                    return Err("universal evidence metric values must be finite".into());
                }
                let expected = match metric.operator {
                    UniversalMetricOperator::GreaterOrEqual => metric.value >= metric.limit,
                    UniversalMetricOperator::LessOrEqual => metric.value <= metric.limit,
                };
                if metric.passed != expected {
                    return Err(format!(
                        "universal evidence metric {} has an inconsistent passed flag",
                        metric.metric
                    ));
                }
                all_metrics_passed &= metric.passed;
            }
            for required in REQUIRED_METRICS {
                if !observed_metrics.contains(required) {
                    return Err(format!(
                        "universal evidence stratum {} omits required metric {required}",
                        stratum.id
                    ));
                }
            }
        }
        for required in REQUIRED_STRATA {
            if !observed_strata.contains(required) {
                return Err(format!(
                    "universal evidence omits required stratum {required}"
                ));
            }
        }
        if self.minimum_listeners == 0
            || self.minimum_listeners > 100_000
            || self.listener_count < self.minimum_listeners
            || self.listener_count > 100_000
            || !self.listener_preference.is_finite()
            || !self.listener_preference_limit.is_finite()
            || !(0.0..=1.0).contains(&self.listener_preference)
            || !(0.0..=1.0).contains(&self.listener_preference_limit)
        {
            return Err("universal evidence listening values are invalid".into());
        }
        let expected_accepted = all_metrics_passed
            && self.listener_count >= self.minimum_listeners
            && self.listener_preference >= self.listener_preference_limit;
        if self.accepted != expected_accepted {
            return Err("universal evidence accepted flag is inconsistent".into());
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize universal evidence payload: {error}"))?;
        if bytes.len() as u64 >= MAX_EVIDENCE_JSON_BYTES {
            return Err("universal evidence payload exceeds the bounded JSON limit".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignedUniversalPromotionEvidence {
    pub schema: String,
    pub schema_version: u32,
    pub payload: UniversalPromotionEvidencePayload,
    pub signature: ReceiptSignature,
}

impl SignedUniversalPromotionEvidence {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let (file, length) = crate::input::open_regular_file(path, "universal promotion evidence")?;
        if length >= MAX_EVIDENCE_JSON_BYTES {
            return Err(format!(
                "universal promotion evidence {} exceeds the {}-byte limit",
                path.display(),
                MAX_EVIDENCE_JSON_BYTES
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length as usize)
            .map_err(|_| "unable to reserve universal evidence JSON".to_string())?;
        file.take(MAX_EVIDENCE_JSON_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read universal promotion evidence: {error}"))?;
        if bytes.len() as u64 != length {
            return Err("universal promotion evidence changed while reading".into());
        }
        let evidence: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse universal promotion evidence: {error}"))?;
        evidence.validate_structure()?;
        Ok(evidence)
    }

    pub fn validate_structure(&self) -> Result<(), String> {
        if self.schema != UNIVERSAL_PROMOTION_EVIDENCE_SCHEMA
            || self.schema_version != UNIVERSAL_RESTORATION_SCHEMA_VERSION
        {
            return Err("unsupported universal promotion evidence schema".into());
        }
        self.payload.validate()?;
        if self.signature.algorithm != "ed25519" {
            return Err("universal promotion evidence signature must use ed25519".into());
        }
        validate_sha256("universal evidence key ID", &self.signature.key_id)?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize universal promotion evidence: {error}"))?;
        if bytes.len() as u64 >= MAX_EVIDENCE_JSON_BYTES {
            return Err("universal promotion evidence exceeds the bounded JSON limit".into());
        }
        Ok(())
    }

    pub fn verify_signature(&self, key: &ReceiptPublicKey) -> Result<(), String> {
        self.validate_structure()?;
        let document = serde_json::to_vec(&self.payload)
            .map_err(|error| format!("serialize universal evidence for verification: {error}"))?;
        key.verify_domain_document(
            PROMOTION_SIGNATURE_DOMAIN,
            &document,
            &self.signature,
            "universal promotion evidence",
        )
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize universal promotion evidence: {error}"))
    }
}

pub fn sign_universal_promotion_evidence(
    payload: UniversalPromotionEvidencePayload,
    key: &ReceiptSecretKey,
) -> Result<SignedUniversalPromotionEvidence, String> {
    payload.validate()?;
    let document = serde_json::to_vec(&payload)
        .map_err(|error| format!("serialize universal evidence for signing: {error}"))?;
    let signature = key.sign_domain_document(
        PROMOTION_SIGNATURE_DOMAIN,
        &document,
        "universal promotion evidence",
    )?;
    let evidence = SignedUniversalPromotionEvidence {
        schema: UNIVERSAL_PROMOTION_EVIDENCE_SCHEMA.into(),
        schema_version: UNIVERSAL_RESTORATION_SCHEMA_VERSION,
        payload,
        signature,
    };
    evidence.validate_structure()?;
    Ok(evidence)
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
    {
        return Err(format!(
            "universal evidence {label} must be lowercase SHA-256"
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
    use hound::SampleFormat;

    fn audio(channels: Vec<Vec<f64>>) -> Audio {
        Audio {
            sample_rate: 48_000,
            channels,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    #[test]
    fn experimental_models_can_only_be_alternate_renders() {
        let mut config = UniversalRestorationConfig {
            model_family: UniversalModelFamily::Generative,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        config.allow_experimental = true;
        config.render_role = UniversalRenderRole::Alternate;
        config.validate().unwrap();
    }

    #[test]
    fn complete_mask_is_exact_and_bounded() {
        let input = audio(vec![vec![0.0, 0.0, 0.0, 0.0]]);
        let output = audio(vec![vec![0.0, 0.25, 0.5, 0.0]]);
        let mask = encode_mask(&input, &output).unwrap();
        mask.validate().unwrap();
        assert_eq!(mask.runs.len(), 3);
        assert_eq!(mask.runs[1].state, UniversalMaskState::Replaced);
        assert_eq!(mask.runs[1].frame_count, 2);
    }

    #[test]
    fn malformed_audio_and_closed_config_are_rejected() {
        assert!(validate_audio(&audio(vec![vec![0.0; 2], vec![0.0; 1]])).is_err());
        assert!(validate_audio(&audio(vec![vec![f64::NAN]])).is_err());
        let encoded = serde_json::to_string(&UniversalRestorationConfig::default()).unwrap();
        let unknown = encoded.replacen('{', "{\"unknown\":true,", 1);
        assert!(serde_json::from_str::<UniversalRestorationConfig>(&unknown).is_err());
    }

    #[test]
    fn promotion_evidence_requires_every_stratum_and_metric() {
        let metrics = || {
            REQUIRED_METRICS
                .iter()
                .map(|metric| UniversalMetricOutcome {
                    metric: (*metric).into(),
                    value: 0.0,
                    operator: UniversalMetricOperator::GreaterOrEqual,
                    limit: 0.0,
                    passed: true,
                })
                .collect()
        };
        let payload = UniversalPromotionEvidencePayload {
            completed_at_unix_seconds: 1,
            model_package_sha256: "0".repeat(64),
            model_family: UniversalModelFamily::Discriminative,
            source_revision: "b1dc3ad1e86419ff0bd666f455bda7936bff0e9a".into(),
            source_sha256: "1".repeat(64),
            checkpoint_sha256: "2".repeat(64),
            corpus_manifest_sha256: "3".repeat(64),
            evaluation_result_sha256: "4".repeat(64),
            strata: REQUIRED_STRATA
                .iter()
                .map(|id| UniversalStratumEvidence {
                    id: (*id).into(),
                    cases: 1,
                    metrics: metrics(),
                })
                .collect(),
            minimum_listeners: 1,
            listener_count: 1,
            listener_preference: 0.5,
            listener_preference_limit: 0.5,
            accepted: true,
        };
        payload.validate().unwrap();
        let mut incomplete = payload.clone();
        incomplete.strata.pop();
        assert!(incomplete
            .validate()
            .unwrap_err()
            .contains("required stratum"));
        let mut inconsistent = payload;
        inconsistent.strata[0].metrics[0].passed = false;
        assert!(inconsistent.validate().is_err());
    }

    #[test]
    fn memory_estimate_covers_multiple_audio_copies() {
        let input = audio(vec![vec![0.0; 48_000]]);
        assert!(
            estimate_universal_restoration_memory_bytes(&input)
                >= estimate_audio_memory_bytes(&input).saturating_mul(7)
        );
    }
}
