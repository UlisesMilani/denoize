//! Network-free, explainable processing recommendations.
//!
//! Recommendations combine a bounded signal summary, the locally compiled
//! backends, read-only verification of models in the embedded signed catalog,
//! detected compute runtimes, configured resource ceilings, and optional
//! on-device calibration. No catalog/cache mutation, model download, or remote
//! service is used by this module.

use crate::decode::DecodeBudget;
use crate::hardware::{
    hardware_capabilities_read_only, select_accelerator_from_capabilities, HardwareCapabilities,
};
use crate::service::requires_external_model;
use crate::{
    AcceleratorPreference, AcceleratorRuntime, Audio, AudioCodec, AudioFormat, AudioInputSession,
    AudioStreamReader, Backend, BackendOptions, DecodeLimits, OnnxModelConfig, Preset,
    ProcessingMode,
};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

/// Stable identifier embedded in recommendation reports.
pub const RECOMMENDATION_SCHEMA: &str = "denoize-recommendation-v1";
/// Current recommendation report schema version.
pub const RECOMMENDATION_SCHEMA_VERSION: u32 = 1;

const DEFAULT_ANALYSIS_SECONDS: u32 = 12;
const MAX_ANALYSIS_SECONDS: u32 = 60;
const ANALYSIS_BLOCK_FRAMES: usize = 4_096;
const DEFAULT_CALIBRATION_RUNS: u8 = 3;
const MAX_CALIBRATION_RUNS: u8 = 9;
const CALIBRATION_SAMPLE_RATE: u32 = 48_000;
const CALIBRATION_FRAMES: usize = 24_000;
const CALIBRATION_SCRATCH_BYTES: u64 = 1024 * 1024;
const CALIBRATION_WORKLOAD: &str = "classical-hifi-v1";
const CALIBRATION_DOMAIN: &[u8] = b"denoize-device-calibration-v1\0";
const ANALYSIS_DOMAIN: &[u8] = b"denoize-recommendation-analysis-v1\0";
#[cfg(test)]
const CALIBRATION_FIXTURE_SHA256: &str =
    "5f64cb9074291ee8688f2f8d432dfb926ca37a0be33e41e3875d71d468a1e479";

/// Optimization intent used to rank otherwise runnable candidates.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecommendationGoal {
    /// Balance expected restoration quality, latency, and retained memory.
    #[default]
    Balanced,
    /// Prefer the strongest suitable locally runnable backend.
    Quality,
    /// Prefer low latency and high realtime headroom.
    Speed,
    /// Prefer the smallest denoize-owned model/runtime reservation.
    LowMemory,
}

impl RecommendationGoal {
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "balanced" | "default" => Some(Self::Balanced),
            "quality" | "best" | "highest" => Some(Self::Quality),
            "speed" | "fast" | "realtime" => Some(Self::Speed),
            "low-memory" | "low_memory" | "memory" => Some(Self::LowMemory),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Quality => "quality",
            Self::Speed => "speed",
            Self::LowMemory => "low-memory",
        }
    }
}

/// Coarse material class inferred from a bounded signal prefix.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecommendationMaterial {
    Speech,
    Music,
    Mixed,
    Quiet,
}

impl RecommendationMaterial {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Speech => "speech",
            Self::Music => "music",
            Self::Mixed => "mixed",
            Self::Quiet => "quiet",
        }
    }
}

/// Configuration for one recommendation operation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecommendationOptions {
    goal: RecommendationGoal,
    analysis_seconds: u32,
    calibration_runs: Option<u8>,
    decode_limits: DecodeLimits,
    max_gpu_memory_bytes: Option<u64>,
    accelerator: AcceleratorPreference,
    deterministic: bool,
}

impl Default for RecommendationOptions {
    fn default() -> Self {
        Self {
            goal: RecommendationGoal::Balanced,
            analysis_seconds: DEFAULT_ANALYSIS_SECONDS,
            calibration_runs: None,
            decode_limits: DecodeLimits::default(),
            max_gpu_memory_bytes: None,
            accelerator: AcceleratorPreference::Auto,
            deterministic: false,
        }
    }
}

impl RecommendationOptions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_goal(mut self, goal: RecommendationGoal) -> Self {
        self.goal = goal;
        self
    }

    #[must_use]
    pub const fn with_analysis_seconds(mut self, seconds: u32) -> Self {
        self.analysis_seconds = seconds;
        self
    }

    /// Enable calibration with the given number of measured runs.
    #[must_use]
    pub const fn with_calibration_runs(mut self, runs: Option<u8>) -> Self {
        self.calibration_runs = runs;
        self
    }

    /// Enable or disable the fixed on-device calibration workload.
    #[must_use]
    pub const fn with_calibration(mut self, enabled: bool) -> Self {
        self.calibration_runs = if enabled {
            Some(DEFAULT_CALIBRATION_RUNS)
        } else {
            None
        };
        self
    }

    #[must_use]
    pub const fn with_decode_limits(mut self, limits: DecodeLimits) -> Self {
        self.decode_limits = limits;
        self
    }

    /// Limit the conservative GPU-side session reservation used for admission.
    #[must_use]
    pub const fn with_max_gpu_memory_bytes(mut self, limit: Option<u64>) -> Self {
        self.max_gpu_memory_bytes = limit;
        self
    }

    #[must_use]
    pub const fn with_accelerator(mut self, accelerator: AcceleratorPreference) -> Self {
        self.accelerator = accelerator;
        self
    }

    #[must_use]
    pub const fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    #[must_use]
    pub const fn goal(self) -> RecommendationGoal {
        self.goal
    }

    #[must_use]
    pub const fn analysis_seconds(self) -> u32 {
        self.analysis_seconds
    }

    #[must_use]
    pub const fn calibration_runs(self) -> Option<u8> {
        self.calibration_runs
    }

    #[must_use]
    pub const fn decode_limits(self) -> DecodeLimits {
        self.decode_limits
    }

    #[must_use]
    pub const fn max_gpu_memory_bytes(self) -> Option<u64> {
        self.max_gpu_memory_bytes
    }

    #[must_use]
    pub const fn accelerator(self) -> AcceleratorPreference {
        self.accelerator
    }

    #[must_use]
    pub const fn deterministic(self) -> bool {
        self.deterministic
    }

    /// Validate option-only bounds without opening an input or model.
    pub fn validate(self) -> Result<(), String> {
        if !(1..=MAX_ANALYSIS_SECONDS).contains(&self.analysis_seconds) {
            return Err(format!(
                "recommendation analysis duration must be between 1 and {MAX_ANALYSIS_SECONDS} seconds"
            ));
        }
        if self
            .calibration_runs
            .is_some_and(|runs| !(1..=MAX_CALIBRATION_RUNS).contains(&runs))
        {
            return Err(format!(
                "recommendation calibration runs must be between 1 and {MAX_CALIBRATION_RUNS}"
            ));
        }
        Ok(())
    }
}

/// Deterministic measurements derived from a bounded input prefix.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RecommendationInput {
    pub format: String,
    pub codec: String,
    pub sample_rate: u32,
    pub channels: usize,
    pub total_frames: Option<u64>,
    pub analyzed_frames: usize,
    pub analysis_mode: String,
    pub analysis_sha256: String,
    pub rms_dbfs: f64,
    pub peak_dbfs: f64,
    pub crest_db: f64,
    pub active_ratio: f64,
    pub zero_crossing_rate: f64,
    pub transient_ratio: f64,
    pub stereo_correlation: Option<f64>,
    pub material: RecommendationMaterial,
    pub material_confidence: f64,
}

/// Network-free device facts used by the decision.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RecommendationDevice {
    pub os: String,
    pub architecture: String,
    pub logical_cpus: usize,
    pub requested_accelerator: String,
    pub available_runtimes: Vec<String>,
}

/// Reproducible evidence from the fixed local calibration workload.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CalibrationEvidence {
    pub workload: String,
    pub fixture_sha256: String,
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: usize,
    pub warmup_runs: u8,
    pub measured_runs: u8,
    pub elapsed_ms: Vec<f64>,
    pub median_elapsed_ms: f64,
    pub baseline_realtime_headroom: f64,
}

/// One stable explanation attached to a candidate score or exclusion.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RecommendationReason {
    pub code: String,
    pub impact: i16,
    pub detail: String,
}

/// One compiled backend considered by the recommendation engine.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RecommendationCandidate {
    pub backend: String,
    pub preset: String,
    pub model: Option<String>,
    pub eligible: bool,
    pub score: u16,
    pub requested_accelerator: String,
    pub effective_accelerator: Option<String>,
    pub accelerator_fallback: Option<String>,
    pub estimated_memory_bytes: Option<u64>,
    pub estimated_gpu_memory_bytes: Option<u64>,
    pub calibrated_realtime_headroom: Option<f64>,
    pub reasons: Vec<RecommendationReason>,
}

/// Effective settings selected from the candidate list.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RecommendationDecision {
    pub backend: String,
    pub preset: String,
    pub processing_mode: String,
    pub strength: f64,
    pub adaptive_noise: bool,
    pub vad: bool,
    pub accelerator: String,
    pub model: Option<String>,
    pub arguments: Vec<String>,
}

/// Stable, explainable report returned by file and decoded-audio entry points.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RecommendationReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub network_accessed: bool,
    pub goal: RecommendationGoal,
    pub input: RecommendationInput,
    pub device: RecommendationDevice,
    pub calibration: Option<CalibrationEvidence>,
    pub decision: RecommendationDecision,
    pub candidates: Vec<RecommendationCandidate>,
}

impl RecommendationReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|error| format!("serialize recommendation report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize recommendation report: {error}"))
    }
}

/// Recommend runnable settings for a regular-file input with default options.
pub fn recommend_file(path: impl AsRef<Path>) -> Result<RecommendationReport, String> {
    recommend_file_with_options(path, RecommendationOptions::default())
}

/// Recommend runnable settings without updating a catalog/model cache or
/// downloading a model.
pub fn recommend_file_with_options(
    path: impl AsRef<Path>,
    options: RecommendationOptions,
) -> Result<RecommendationReport, String> {
    options.validate()?;
    let session = AudioInputSession::open(path)?;
    let input = analyze_session(session, options)?;
    recommend_from_input(input, options)
}

/// Recommend settings for already-decoded audio.
///
/// When a working-set ceiling is supplied, the caller-owned channel capacities
/// and recommendation analysis state must fit it before analysis begins.
pub fn recommend_audio(
    audio: &Audio,
    options: RecommendationOptions,
) -> Result<RecommendationReport, String> {
    options.validate()?;
    validate_audio(audio)?;
    let limit = analysis_frame_limit(audio.sample_rate, options.analysis_seconds)?;
    let frames = audio.frames().min(limit);
    let scratch = analysis_scratch_bytes(audio.channels.len())?;
    DecodeBudget::new(options.decode_limits).check_planar_capacities(
        &audio.channels,
        scratch,
        "recommendation decoded-audio analysis",
    )?;
    let mut accumulator = SignalAccumulator::try_new(audio.sample_rate, audio.channels.len())?;
    accumulator.ingest(&audio.channels, frames)?;
    let input = accumulator.finish(
        "decoded-audio",
        "pcm",
        Some(audio.frames() as u64),
        "decoded-audio",
    )?;
    recommend_from_input(input, options)
}

fn analyze_session(
    mut session: AudioInputSession,
    options: RecommendationOptions,
) -> Result<RecommendationInput, String> {
    let probe = crate::probe_file_from_session_with_limits(&mut session, options.decode_limits)?;
    if matches!(
        probe.format,
        AudioFormat::Wav
            | AudioFormat::Flac
            | AudioFormat::OggVorbis
            | AudioFormat::OggOpus
            | AudioFormat::Mp3
            | AudioFormat::AacAdts
            | AudioFormat::M4a
    ) {
        let mut reader = AudioStreamReader::from_session(session, options.decode_limits)?;
        let info = reader.info();
        let frame_limit = analysis_frame_limit(info.sample_rate(), options.analysis_seconds)?;
        let block_frames = ANALYSIS_BLOCK_FRAMES.min(frame_limit.max(1));
        let analysis_scratch = analysis_scratch_bytes(info.channels())?;
        let temporary_bytes = info
            .decoder_additional_bytes
            .checked_add(analysis_scratch)
            .ok_or_else(|| "recommendation stream scratch byte count overflows".to_string())?;
        DecodeBudget::new(options.decode_limits).check_planar_frames(
            info.channels(),
            block_frames,
            temporary_bytes,
            "recommendation stream analysis",
        )?;
        let mut accumulator = SignalAccumulator::try_new(info.sample_rate(), info.channels())?;
        while accumulator.frames < frame_limit {
            let remaining = frame_limit - accumulator.frames;
            let Some(block) = reader.next_block(block_frames.min(remaining))? else {
                break;
            };
            let frames = block.first().map_or(0, Vec::len);
            accumulator.ingest(&block, frames)?;
        }
        return accumulator.finish(
            format_name(info.format),
            codec_name(info.codec),
            info.total_frames,
            "bounded-stream",
        );
    }

    let audio = crate::read_audio_from_session_with_limits(&mut session, options.decode_limits)?;
    validate_audio(&audio)?;
    let frame_limit = analysis_frame_limit(audio.sample_rate, options.analysis_seconds)?;
    let frames = audio.frames().min(frame_limit);
    let scratch = analysis_scratch_bytes(audio.channels.len())?;
    DecodeBudget::new(options.decode_limits).check_planar_capacities(
        &audio.channels,
        scratch,
        "recommendation whole-file analysis",
    )?;
    let mut accumulator = SignalAccumulator::try_new(audio.sample_rate, audio.channels.len())?;
    accumulator.ingest(&audio.channels, frames)?;
    accumulator.finish(
        format_name(probe.format),
        codec_name(probe.codec),
        Some(audio.frames() as u64),
        "whole-file-fallback",
    )
}

fn validate_audio(audio: &Audio) -> Result<(), String> {
    if audio.sample_rate == 0 {
        return Err("recommendation input sample rate is zero".into());
    }
    if audio.channels.is_empty() {
        return Err("recommendation input has no channels".into());
    }
    let frames = audio.channels[0].len();
    if frames == 0 {
        return Err("recommendation input has no frames".into());
    }
    if let Some((index, channel)) = audio
        .channels
        .iter()
        .enumerate()
        .find(|(_, channel)| channel.len() != frames)
    {
        return Err(format!(
            "recommendation input channel {index} has {} frames but channel 0 has {frames}",
            channel.len()
        ));
    }
    Ok(())
}

fn analysis_frame_limit(sample_rate: u32, seconds: u32) -> Result<usize, String> {
    u64::from(sample_rate)
        .checked_mul(u64::from(seconds))
        .and_then(|frames| usize::try_from(frames).ok())
        .ok_or_else(|| "recommendation analysis frame limit overflows".to_string())
}

struct SignalAccumulator {
    sample_rate: u32,
    channels: usize,
    frames: usize,
    sample_count: u64,
    sum_squares: f64,
    peak: f64,
    active: u64,
    zero_crossings: u64,
    differences: u64,
    transients: u64,
    previous: Vec<Option<f64>>,
    stereo_count: u64,
    stereo_x: f64,
    stereo_y: f64,
    stereo_x2: f64,
    stereo_y2: f64,
    stereo_xy: f64,
    hash: Sha256,
}

impl SignalAccumulator {
    fn try_new(sample_rate: u32, channels: usize) -> Result<Self, String> {
        let mut hash = Sha256::new();
        hash.update(ANALYSIS_DOMAIN);
        hash.update(sample_rate.to_le_bytes());
        hash.update((channels as u64).to_le_bytes());
        let mut previous = Vec::new();
        previous
            .try_reserve_exact(channels)
            .map_err(|error| format!("recommendation analysis state reserve: {error}"))?;
        previous.resize(channels, None);
        Ok(Self {
            sample_rate,
            channels,
            frames: 0,
            sample_count: 0,
            sum_squares: 0.0,
            peak: 0.0,
            active: 0,
            zero_crossings: 0,
            differences: 0,
            transients: 0,
            previous,
            stereo_count: 0,
            stereo_x: 0.0,
            stereo_y: 0.0,
            stereo_x2: 0.0,
            stereo_y2: 0.0,
            stereo_xy: 0.0,
            hash,
        })
    }

    fn ingest(&mut self, channels: &[Vec<f64>], frames: usize) -> Result<(), String> {
        if channels.len() != self.channels {
            return Err("recommendation input channel count changed during analysis".into());
        }
        if channels.iter().any(|channel| channel.len() < frames) {
            return Err("recommendation input block has inconsistent channel lengths".into());
        }
        for frame in 0..frames {
            for (index, channel) in channels.iter().enumerate() {
                let sample = crate::sanitize_sample(channel[frame]);
                self.hash.update(sample.to_bits().to_le_bytes());
                let absolute = sample.abs();
                self.sum_squares += sample * sample;
                self.peak = self.peak.max(absolute);
                self.active += u64::from(absolute >= 0.001);
                if let Some(previous) = self.previous[index] {
                    self.differences += 1;
                    self.zero_crossings += u64::from(
                        (previous < 0.0 && sample >= 0.0) || (previous >= 0.0 && sample < 0.0),
                    );
                    self.transients += u64::from((sample - previous).abs() >= 0.15);
                }
                self.previous[index] = Some(sample);
                self.sample_count += 1;
            }
            if self.channels >= 2 {
                let left = crate::sanitize_sample(channels[0][frame]);
                let right = crate::sanitize_sample(channels[1][frame]);
                self.stereo_x += left;
                self.stereo_y += right;
                self.stereo_x2 += left * left;
                self.stereo_y2 += right * right;
                self.stereo_xy += left * right;
                self.stereo_count += 1;
            }
        }
        self.frames = self
            .frames
            .checked_add(frames)
            .ok_or_else(|| "recommendation analyzed frame count overflows".to_string())?;
        Ok(())
    }

    fn finish(
        self,
        format: impl Into<String>,
        codec: impl Into<String>,
        total_frames: Option<u64>,
        analysis_mode: impl Into<String>,
    ) -> Result<RecommendationInput, String> {
        if self.frames == 0 || self.sample_count == 0 {
            return Err("recommendation input has no decodable frames".into());
        }
        let rms = (self.sum_squares / self.sample_count as f64).sqrt();
        let rms_dbfs = amplitude_db(rms);
        let peak_dbfs = amplitude_db(self.peak);
        let crest_db = if rms > 0.0 {
            20.0 * (self.peak.max(rms) / rms).log10()
        } else {
            0.0
        };
        let active_ratio = self.active as f64 / self.sample_count as f64;
        let zero_crossing_rate = ratio(self.zero_crossings, self.differences);
        let transient_ratio = ratio(self.transients, self.differences);
        let stereo_correlation = correlation(&self);
        let (material, material_confidence) = classify_material(
            self.channels,
            rms_dbfs,
            crest_db,
            active_ratio,
            zero_crossing_rate,
            transient_ratio,
            stereo_correlation,
        );
        Ok(RecommendationInput {
            format: format.into(),
            codec: codec.into(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            total_frames,
            analyzed_frames: self.frames,
            analysis_mode: analysis_mode.into(),
            analysis_sha256: format!("{:x}", self.hash.finalize()),
            rms_dbfs: round_metric(rms_dbfs),
            peak_dbfs: round_metric(peak_dbfs),
            crest_db: round_metric(crest_db),
            active_ratio: round_metric(active_ratio),
            zero_crossing_rate: round_metric(zero_crossing_rate),
            transient_ratio: round_metric(transient_ratio),
            stereo_correlation: stereo_correlation.map(round_metric),
            material,
            material_confidence: round_metric(material_confidence),
        })
    }
}

fn analysis_scratch_bytes(channels: usize) -> Result<u64, String> {
    let channel_state = u64::try_from(channels)
        .ok()
        .and_then(|channels| channels.checked_mul(std::mem::size_of::<Option<f64>>() as u64))
        .ok_or_else(|| "recommendation analysis state byte count overflows".to_string())?;
    channel_state
        .checked_add(std::mem::size_of::<SignalAccumulator>() as u64)
        .ok_or_else(|| "recommendation analysis scratch byte count overflows".to_string())
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn amplitude_db(amplitude: f64) -> f64 {
    if amplitude > 0.0 {
        (20.0 * amplitude.log10()).max(-120.0)
    } else {
        -120.0
    }
}

fn correlation(accumulator: &SignalAccumulator) -> Option<f64> {
    if accumulator.stereo_count < 2 {
        return None;
    }
    let count = accumulator.stereo_count as f64;
    let covariance = accumulator.stereo_xy - accumulator.stereo_x * accumulator.stereo_y / count;
    let variance_x = accumulator.stereo_x2 - accumulator.stereo_x * accumulator.stereo_x / count;
    let variance_y = accumulator.stereo_y2 - accumulator.stereo_y * accumulator.stereo_y / count;
    let denominator = (variance_x.max(0.0) * variance_y.max(0.0)).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some((covariance / denominator).clamp(-1.0, 1.0))
    }
}

fn classify_material(
    channels: usize,
    rms_dbfs: f64,
    crest_db: f64,
    active_ratio: f64,
    zero_crossing_rate: f64,
    transient_ratio: f64,
    stereo_correlation: Option<f64>,
) -> (RecommendationMaterial, f64) {
    if rms_dbfs <= -55.0 || active_ratio <= 0.03 {
        return (RecommendationMaterial::Quiet, 0.9);
    }
    let mono_bias = if channels == 1 { 0.18 } else { 0.0 };
    let speech_zcr = triangular(zero_crossing_rate, 0.015, 0.09, 0.28);
    let speech_crest = triangular(crest_db, 4.0, 12.0, 26.0);
    let speech_activity = triangular(active_ratio, 0.08, 0.58, 1.0);
    let speech_score =
        (0.36 * speech_zcr + 0.26 * speech_crest + 0.20 * speech_activity + mono_bias)
            .clamp(0.0, 1.0);

    let stereo_width = stereo_correlation.map_or(0.0, |value| (1.0 - value.abs()).clamp(0.0, 1.0));
    let music_zcr = triangular(zero_crossing_rate, 0.0, 0.035, 0.18);
    let music_crest = triangular(crest_db, 3.0, 10.0, 24.0);
    let music_transients = triangular(transient_ratio, 0.0, 0.035, 0.22);
    let music_score =
        (0.30 * music_zcr + 0.25 * music_crest + 0.20 * music_transients + 0.25 * stereo_width)
            .clamp(0.0, 1.0);
    let difference = (speech_score - music_score).abs();
    if difference < 0.14 {
        (RecommendationMaterial::Mixed, 1.0 - difference / 0.14)
    } else if speech_score > music_score {
        (RecommendationMaterial::Speech, difference.clamp(0.0, 1.0))
    } else {
        (RecommendationMaterial::Music, difference.clamp(0.0, 1.0))
    }
}

fn triangular(value: f64, low: f64, center: f64, high: f64) -> f64 {
    if value <= low || value >= high {
        0.0
    } else if value <= center {
        (value - low) / (center - low)
    } else {
        (high - value) / (high - center)
    }
}

fn round_metric(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn recommend_from_input(
    input: RecommendationInput,
    options: RecommendationOptions,
) -> Result<RecommendationReport, String> {
    let hardware = hardware_capabilities_read_only();
    let mut available_runtimes = hardware
        .runtimes()
        .iter()
        .filter(|runtime| runtime.available())
        .map(|runtime| runtime.runtime().name().to_string())
        .collect::<Vec<_>>();
    available_runtimes.sort();
    let device = RecommendationDevice {
        os: hardware.os().into(),
        architecture: hardware.architecture().into(),
        logical_cpus: hardware.logical_cpus(),
        requested_accelerator: options.accelerator.name().into(),
        available_runtimes,
    };
    let calibration = if let Some(runs) = options.calibration_runs {
        let temporary_bytes = CALIBRATION_SCRATCH_BYTES
            .checked_add(analysis_scratch_bytes(1)?)
            .ok_or_else(|| "recommendation calibration scratch byte count overflows".to_string())?;
        DecodeBudget::new(options.decode_limits).check_planar_frames(
            1,
            CALIBRATION_FRAMES,
            temporary_bytes,
            "recommendation device calibration",
        )?;
        Some(run_device_calibration(runs)?)
    } else {
        None
    };
    let preset = recommended_preset(input.material, options.goal);
    let mode = recommended_mode(input.material);
    let mut denoiser = preset.config(input.sample_rate);
    ProcessingMode::parse(mode)
        .expect("recommended processing mode must parse")
        .apply(&mut denoiser);
    let mut candidates =
        build_candidates(&input, preset, options, calibration.as_ref(), &hardware)?;
    candidates.sort_by(candidate_order);
    let selected = candidates
        .iter()
        .find(|candidate| candidate.eligible)
        .ok_or_else(|| {
            "no compiled backend satisfies the requested recommendation constraints".to_string()
        })?;
    let accelerator = selected
        .effective_accelerator
        .as_deref()
        .unwrap_or("cpu")
        .to_string();
    let mut arguments = vec![
        "--backend".into(),
        selected.backend.clone(),
        "--preset".into(),
        selected.preset.clone(),
        "--mode".into(),
        mode.into(),
        "--strength".into(),
        denoiser.strength.to_string(),
        "--accelerator".into(),
        accelerator.clone(),
    ];
    if options.deterministic {
        arguments.push("--deterministic".into());
    }
    let decision = RecommendationDecision {
        backend: selected.backend.clone(),
        preset: selected.preset.clone(),
        processing_mode: mode.into(),
        strength: denoiser.strength,
        adaptive_noise: denoiser.adaptive_noise,
        vad: denoiser.vad,
        accelerator,
        model: selected.model.clone(),
        arguments,
    };
    Ok(RecommendationReport {
        schema: RECOMMENDATION_SCHEMA.into(),
        schema_version: RECOMMENDATION_SCHEMA_VERSION,
        denoize_version: env!("CARGO_PKG_VERSION").into(),
        network_accessed: false,
        goal: options.goal,
        input,
        device,
        calibration,
        decision,
        candidates,
    })
}

fn build_candidates(
    input: &RecommendationInput,
    preset: Preset,
    options: RecommendationOptions,
    calibration: Option<&CalibrationEvidence>,
    hardware: &HardwareCapabilities,
) -> Result<Vec<RecommendationCandidate>, String> {
    let managed_catalog = Backend::available_names()
        .contains(&"gtcrn")
        .then(crate::models::embedded_catalog);
    let mut candidates = Vec::new();
    for &name in Backend::available_names() {
        let backend = Backend::parse(name).expect("available backend name must parse");
        let traits = backend_traits(name);
        let mut reasons = vec![reason(
            "compiled",
            0,
            "backend is compiled into this binary",
        )];
        let mut backend_options = BackendOptions {
            accelerator: options.accelerator,
            deterministic: options.deterministic,
            ..BackendOptions::default()
        };
        let mut model_name = None;
        let mut eligible = true;
        let mut model_size = 0_u64;
        if requires_external_model(backend) {
            eligible = false;
            reasons.push(reason(
                "explicit-model-required",
                -100,
                "backend requires a caller-supplied model path, which recommendation reports intentionally do not serialize",
            ));
        } else if name == "gtcrn" {
            let catalog = managed_catalog
                .as_ref()
                .expect("GTCRN availability created an embedded catalog");
            match catalog.find(name) {
                Some(model) => {
                    model_name = Some(model.name().to_string());
                    model_size = model.size_bytes();
                    match crate::models::verify_catalog_model_read_only(model) {
                        Ok(path) => {
                            backend_options.onnx = Some(OnnxModelConfig {
                                path,
                                sample_rate: model.sample_rate(),
                            });
                            reasons.push(reason(
                                "verified-model",
                                5,
                                format!("verified managed model {} is installed", model.name()),
                            ));
                        }
                        Err(_) => {
                            eligible = false;
                            reasons.push(reason(
                            "model-unavailable",
                            -100,
                            format!(
                                "managed model {} is not installed or failed read-only integrity verification; run denoize models doctor",
                                model.name()
                            ),
                        ));
                        }
                    }
                }
                None => {
                    eligible = false;
                    reasons.push(reason(
                        "model-unavailable",
                        -100,
                        "no unambiguous managed model is present in the embedded signed catalog",
                    ));
                }
            }
        }

        let mut effective_accelerator = None;
        let mut effective_runtime = None;
        let mut accelerator_fallback = None;
        if eligible {
            if let Err(error) = backend_options.validate_resolved_resources(backend) {
                eligible = false;
                reasons.push(reason("invalid-backend-options", -100, error));
            }
        }
        if eligible {
            match select_accelerator_from_capabilities(
                backend,
                backend_options.accelerator,
                backend_options.deterministic,
                hardware,
            ) {
                Ok(selection) => {
                    effective_runtime = Some(selection.effective());
                    effective_accelerator = Some(selection.effective().name().to_string());
                    accelerator_fallback =
                        selection.fallback().map(|value| value.name().to_string());
                    reasons.push(reason(
                        "runtime",
                        i16::from(selection.effective() != crate::AcceleratorRuntime::Cpu) * 4,
                        format!(
                            "{} resolves to {}{}",
                            selection.requested().name(),
                            selection.effective().name(),
                            selection
                                .fallback()
                                .map(|fallback| format!(" ({})", fallback.name()))
                                .unwrap_or_default()
                        ),
                    ));
                }
                Err(error) => {
                    eligible = false;
                    reasons.push(reason("runtime-unavailable", -100, error));
                }
            }
        }

        let estimated_memory = if name == "classical" {
            Some(0)
        } else if requires_external_model(backend) {
            None
        } else {
            crate::estimate_model_session_bytes(model_size).ok()
        };
        if let (Some(limit), Some(estimate)) = (
            options.decode_limits.max_working_set_bytes,
            estimated_memory,
        ) {
            if estimate > limit {
                eligible = false;
                reasons.push(reason(
                    "memory-limit",
                    -100,
                    format!(
                        "estimated model/runtime reservation {estimate} bytes exceeds the {limit}-byte limit"
                    ),
                ));
            }
        }

        let mut estimated_gpu_memory = None;
        if effective_runtime.is_some_and(|runtime| runtime != AcceleratorRuntime::Cpu) {
            match crate::estimate_gpu_session_bytes(model_size) {
                Ok(estimate) => estimated_gpu_memory = Some(estimate),
                Err(error) => {
                    eligible = false;
                    reasons.push(reason("gpu-memory-estimate", -100, error));
                }
            }
        }
        if let Some(estimate) = estimated_gpu_memory {
            let runtime = effective_runtime.expect("GPU estimate requires an effective runtime");
            let device_memory_bytes = hardware
                .runtimes()
                .iter()
                .find(|capability| capability.runtime() == runtime)
                .and_then(|capability| capability.memory_bytes());
            apply_gpu_memory_constraints(
                estimate,
                options.max_gpu_memory_bytes,
                device_memory_bytes,
                &mut eligible,
                &mut reasons,
            );
            reasons.push(reason(
                "runtime-read-only-probe",
                0,
                "recommendation does not create or test a runtime cache; processing revalidates cache writability before model preparation",
            ));
        }

        let quality = material_quality(traits, input.material);
        let scored_memory =
            estimated_memory.map(|bytes| bytes.saturating_add(estimated_gpu_memory.unwrap_or(0)));
        let memory_score = scored_memory.map_or(40, |bytes| {
            100_i32.saturating_sub((bytes / (16 * 1024 * 1024)).min(80) as i32)
        });
        let (quality_weight, speed_weight, memory_weight) = goal_weights(options.goal);
        let mut score =
            (quality * quality_weight + traits.speed * speed_weight + memory_score * memory_weight)
                / 100;
        let material_adjustment = material_adjustment(name, input.material);
        score += material_adjustment;
        reasons.push(reason(
            "material-fit",
            material_adjustment as i16,
            format!(
                "{} material contributes quality score {quality}",
                input.material.name()
            ),
        ));
        let calibrated_headroom = calibration
            .map(|evidence| evidence.baseline_realtime_headroom / f64::from(traits.cost_units));
        if let Some(headroom) = calibrated_headroom {
            let impact = if headroom < 1.0 {
                -35
            } else if headroom < 1.5 {
                -18
            } else if headroom >= 8.0 {
                5
            } else {
                0
            };
            score += impact;
            reasons.push(reason(
                "calibrated-headroom",
                impact as i16,
                format!(
                    "fixed device calibration estimates {:.3}x heuristic realtime headroom for cost class {}",
                    headroom, traits.cost_units
                ),
            ));
        } else {
            reasons.push(reason(
                "uncalibrated",
                0,
                "candidate uses static cost class because on-device calibration was not requested",
            ));
        }
        if !eligible {
            score = 0;
        }
        candidates.push(RecommendationCandidate {
            backend: name.into(),
            preset: preset_name(preset).into(),
            model: model_name,
            eligible,
            score: score.clamp(0, 100) as u16,
            requested_accelerator: options.accelerator.name().into(),
            effective_accelerator,
            accelerator_fallback,
            estimated_memory_bytes: estimated_memory,
            estimated_gpu_memory_bytes: estimated_gpu_memory,
            calibrated_realtime_headroom: calibrated_headroom.map(round_metric),
            reasons,
        });
    }
    Ok(candidates)
}

fn apply_gpu_memory_constraints(
    estimate: u64,
    configured_limit: Option<u64>,
    device_limit: Option<u64>,
    eligible: &mut bool,
    reasons: &mut Vec<RecommendationReason>,
) {
    if let Some(limit) = configured_limit {
        if estimate > limit {
            *eligible = false;
            reasons.push(reason(
                "gpu-memory-limit",
                -100,
                format!(
                    "estimated GPU session reservation {estimate} bytes exceeds the configured {limit}-byte GPU limit"
                ),
            ));
        }
    }
    match device_limit {
        Some(available) if estimate > available => {
            *eligible = false;
            reasons.push(reason(
                "device-gpu-memory",
                -100,
                format!(
                    "estimated GPU session reservation {estimate} bytes exceeds the device-reported {available}-byte limit"
                ),
            ));
        }
        Some(available) => reasons.push(reason(
            "gpu-memory-fit",
            0,
            format!(
                "estimated GPU session reservation {estimate} bytes fits the device-reported {available}-byte limit"
            ),
        )),
        None => reasons.push(reason(
            "gpu-memory-unreported",
            0,
            "the runtime did not report a GPU memory limit; processing admission will revalidate configured limits",
        )),
    }
}

#[derive(Clone, Copy)]
struct BackendTraits {
    speech_quality: i32,
    music_quality: i32,
    mixed_quality: i32,
    quiet_quality: i32,
    speed: i32,
    cost_units: u16,
}

fn backend_traits(name: &str) -> BackendTraits {
    match name {
        "classical" => traits(62, 82, 78, 92, 96, 1),
        "rnnoise" => traits(78, 48, 62, 55, 92, 2),
        "deepfilter" => traits(94, 68, 82, 64, 68, 10),
        "gtcrn" => traits(91, 58, 76, 60, 76, 7),
        "mpsenet" => traits(96, 58, 78, 58, 38, 28),
        "bsrnn" => traits(95, 67, 83, 60, 52, 18),
        "mossformer2" => traits(97, 62, 82, 58, 30, 45),
        "sgmse" => traits(99, 72, 88, 62, 8, 180),
        "onnx" => traits(55, 55, 55, 50, 45, 24),
        _ => traits(50, 50, 50, 50, 40, 30),
    }
}

const fn traits(
    speech_quality: i32,
    music_quality: i32,
    mixed_quality: i32,
    quiet_quality: i32,
    speed: i32,
    cost_units: u16,
) -> BackendTraits {
    BackendTraits {
        speech_quality,
        music_quality,
        mixed_quality,
        quiet_quality,
        speed,
        cost_units,
    }
}

const fn material_quality(traits: BackendTraits, material: RecommendationMaterial) -> i32 {
    match material {
        RecommendationMaterial::Speech => traits.speech_quality,
        RecommendationMaterial::Music => traits.music_quality,
        RecommendationMaterial::Mixed => traits.mixed_quality,
        RecommendationMaterial::Quiet => traits.quiet_quality,
    }
}

const fn goal_weights(goal: RecommendationGoal) -> (i32, i32, i32) {
    match goal {
        RecommendationGoal::Balanced => (55, 35, 10),
        RecommendationGoal::Quality => (78, 12, 10),
        RecommendationGoal::Speed => (25, 65, 10),
        RecommendationGoal::LowMemory => (25, 15, 60),
    }
}

fn material_adjustment(name: &str, material: RecommendationMaterial) -> i32 {
    match (name, material) {
        ("classical", RecommendationMaterial::Music | RecommendationMaterial::Quiet) => 8,
        (
            "rnnoise" | "deepfilter" | "gtcrn" | "mpsenet" | "bsrnn" | "mossformer2" | "sgmse",
            RecommendationMaterial::Speech,
        ) => 8,
        ("rnnoise" | "gtcrn" | "mpsenet" | "mossformer2", RecommendationMaterial::Music) => -12,
        ("sgmse", RecommendationMaterial::Quiet) => -10,
        _ => 0,
    }
}

fn recommended_preset(material: RecommendationMaterial, goal: RecommendationGoal) -> Preset {
    match (material, goal) {
        (RecommendationMaterial::Speech, RecommendationGoal::Speed) => Preset::Gentle,
        (RecommendationMaterial::Speech, _) => Preset::Speech,
        (RecommendationMaterial::Music, RecommendationGoal::Quality) => Preset::HiFi,
        (RecommendationMaterial::Music, _) => Preset::Music,
        (RecommendationMaterial::Quiet, _) => Preset::Restore,
        (RecommendationMaterial::Mixed, RecommendationGoal::Quality) => Preset::HiFi,
        (RecommendationMaterial::Mixed, _) => Preset::Gentle,
    }
}

const fn recommended_mode(material: RecommendationMaterial) -> &'static str {
    match material {
        RecommendationMaterial::Speech => "speech",
        RecommendationMaterial::Music => "music",
        RecommendationMaterial::Mixed | RecommendationMaterial::Quiet => "ambient",
    }
}

const fn preset_name(preset: Preset) -> &'static str {
    match preset {
        Preset::Speech => "speech",
        Preset::Music => "music",
        Preset::Aggressive => "aggressive",
        Preset::Gentle => "gentle",
        Preset::Restore => "restore",
        Preset::HiFi => "hifi",
    }
}

fn reason(code: &str, impact: i16, detail: impl Into<String>) -> RecommendationReason {
    RecommendationReason {
        code: code.into(),
        impact,
        detail: detail.into(),
    }
}

fn candidate_order(left: &RecommendationCandidate, right: &RecommendationCandidate) -> Ordering {
    right
        .eligible
        .cmp(&left.eligible)
        .then_with(|| right.score.cmp(&left.score))
        .then_with(|| left.backend.cmp(&right.backend))
}

/// Run the fixed, network-free device calibration workload.
pub fn run_device_calibration(runs: u8) -> Result<CalibrationEvidence, String> {
    if !(1..=MAX_CALIBRATION_RUNS).contains(&runs) {
        return Err(format!(
            "recommendation calibration runs must be between 1 and {MAX_CALIBRATION_RUNS}"
        ));
    }
    let (fixture, fixture_sha256) = calibration_fixture();
    let config = Preset::HiFi.config(CALIBRATION_SAMPLE_RATE);
    let _ = crate::backend::process_classical(&fixture, &config);
    let mut elapsed_ms = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let started = Instant::now();
        let output = crate::backend::process_classical(&fixture, &config);
        std::hint::black_box(output);
        let elapsed = started.elapsed().as_secs_f64().max(1e-9);
        elapsed_ms.push(round_metric(elapsed * 1_000.0));
    }
    let mut ordered = elapsed_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let median_elapsed_ms = ordered[ordered.len() / 2];
    let fixture_seconds = CALIBRATION_FRAMES as f64 / f64::from(CALIBRATION_SAMPLE_RATE);
    let baseline_realtime_headroom = fixture_seconds / (median_elapsed_ms / 1_000.0).max(1e-9);
    Ok(CalibrationEvidence {
        workload: CALIBRATION_WORKLOAD.into(),
        fixture_sha256,
        sample_rate: CALIBRATION_SAMPLE_RATE,
        channels: 1,
        frames: CALIBRATION_FRAMES,
        warmup_runs: 1,
        measured_runs: runs,
        elapsed_ms,
        median_elapsed_ms,
        baseline_realtime_headroom: round_metric(baseline_realtime_headroom),
    })
}

fn calibration_fixture() -> (Vec<Vec<f64>>, String) {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    let mut channel = Vec::with_capacity(CALIBRATION_FRAMES);
    let mut hash = Sha256::new();
    hash.update(CALIBRATION_DOMAIN);
    hash.update(CALIBRATION_SAMPLE_RATE.to_le_bytes());
    hash.update((CALIBRATION_FRAMES as u64).to_le_bytes());
    for frame in 0..CALIBRATION_FRAMES {
        state = splitmix64(state);
        // Integer-only synthesis plus an exact power-of-two conversion keeps
        // the fixture bytes stable across libm implementations and targets.
        let phase = (frame % 512) as i32;
        let triangle = (if phase < 256 { phase } else { 511 - phase }) * 256 - 32_640;
        let noise = i32::from((state >> 48) as u16) - 32_768;
        let envelope = if frame % 9_600 < 7_200 { 3 } else { 1 };
        let fixed_sample = triangle * envelope + noise / 8;
        let sample = f64::from(fixed_sample) / 131_072.0;
        hash.update(sample.to_bits().to_le_bytes());
        channel.push(sample);
    }
    (vec![channel], format!("{:x}", hash.finalize()))
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

const fn format_name(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::Rf64 => "rf64",
        AudioFormat::Aiff => "aiff",
        AudioFormat::Caf => "caf",
        AudioFormat::Flac => "flac",
        AudioFormat::OggOpus => "ogg-opus",
        AudioFormat::OggVorbis => "ogg-vorbis",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::M4a => "m4a",
        AudioFormat::AacAdts => "aac-adts",
        AudioFormat::Unknown => "unknown",
    }
}

const fn codec_name(codec: AudioCodec) -> &'static str {
    match codec {
        AudioCodec::Pcm => "pcm",
        AudioCodec::Flac => "flac",
        AudioCodec::Opus => "opus",
        AudioCodec::Vorbis => "vorbis",
        AudioCodec::Mp3 => "mp3",
        AudioCodec::Aac => "aac",
        AudioCodec::Alac => "alac",
        AudioCodec::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn speech_like() -> Audio {
        let frames = 48_000;
        let channel = (0..frames)
            .map(|frame| {
                let time = frame as f64 / 48_000.0;
                let envelope = if frame % 9_600 < 7_200 { 1.0 } else { 0.03 };
                ((std::f64::consts::TAU * 155.0 * time).sin() * 0.24
                    + (std::f64::consts::TAU * 2_100.0 * time).sin() * 0.04)
                    * envelope
            })
            .collect();
        Audio {
            sample_rate: 48_000,
            channels: vec![channel],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn music_like() -> Audio {
        let frames = 48_000;
        let left = (0..frames)
            .map(|frame| {
                let time = frame as f64 / 48_000.0;
                (std::f64::consts::TAU * 220.0 * time).sin() * 0.35
                    + (std::f64::consts::TAU * 440.0 * time).sin() * 0.15
            })
            .collect();
        let right = (0..frames)
            .map(|frame| {
                let time = frame as f64 / 48_000.0;
                (std::f64::consts::TAU * 277.0 * time).sin() * 0.31
                    + (std::f64::consts::TAU * 554.0 * time).sin() * 0.13
            })
            .collect();
        Audio {
            sample_rate: 48_000,
            channels: vec![left, right],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    #[test]
    fn options_validate_bounded_analysis_and_calibration() {
        assert!(RecommendationOptions::new()
            .with_analysis_seconds(0)
            .validate()
            .is_err());
        assert!(RecommendationOptions::new()
            .with_analysis_seconds(MAX_ANALYSIS_SECONDS + 1)
            .validate()
            .is_err());
        assert!(RecommendationOptions::new()
            .with_calibration_runs(Some(0))
            .validate()
            .is_err());
        assert!(RecommendationOptions::new()
            .with_calibration_runs(Some(MAX_CALIBRATION_RUNS + 1))
            .validate()
            .is_err());
    }

    #[test]
    fn decoded_audio_report_is_stable_and_network_free() {
        let report = recommend_audio(
            &speech_like(),
            RecommendationOptions::new().with_accelerator(AcceleratorPreference::Cpu),
        )
        .expect("recommend speech");
        assert_eq!(report.schema, RECOMMENDATION_SCHEMA);
        assert!(!report.network_accessed);
        assert_eq!(report.input.analysis_mode, "decoded-audio");
        assert_eq!(report.input.analysis_sha256.len(), 64);
        assert!(!report.candidates.is_empty());
        assert!(report.candidates[0].eligible);
        assert!(report
            .candidates
            .iter()
            .filter(|candidate| candidate.effective_accelerator.as_deref() == Some("cpu"))
            .all(|candidate| candidate.estimated_gpu_memory_bytes.is_none()));
        assert_eq!(report.decision.backend, report.candidates[0].backend);
        let mut effective = Preset::parse(&report.decision.preset)
            .expect("decision preset parses")
            .config(report.input.sample_rate);
        ProcessingMode::parse(&report.decision.processing_mode)
            .expect("decision mode parses")
            .apply(&mut effective);
        assert_eq!(report.decision.strength, effective.strength);
        assert_eq!(report.decision.adaptive_noise, effective.adaptive_noise);
        assert_eq!(report.decision.vad, effective.vad);
        let expected_strength = effective.strength.to_string();
        assert!(report
            .decision
            .arguments
            .windows(2)
            .any(|pair| pair[0] == "--strength" && pair[1] == expected_strength));
        assert!(report.candidates.iter().all(|candidate| {
            !candidate.eligible
                || !requires_external_model(
                    Backend::parse(&candidate.backend).expect("reported backend parses"),
                )
        }));
        assert!(report
            .to_json()
            .expect("serialize")
            .contains(RECOMMENDATION_SCHEMA));
    }

    #[test]
    fn signal_hash_and_metrics_are_deterministic() {
        let options = RecommendationOptions::new().with_accelerator(AcceleratorPreference::Cpu);
        let first = recommend_audio(&music_like(), options).expect("first report");
        let second = recommend_audio(&music_like(), options).expect("second report");
        assert_eq!(first.input, second.input);
        assert_eq!(first.decision, second.decision);
        assert_eq!(first.input.material, RecommendationMaterial::Music);
        assert!(first.input.stereo_correlation.is_some());
    }

    #[test]
    fn mp3_file_recommendation_uses_the_bounded_stream_reader() {
        let directory = tempfile::tempdir().expect("create recommendation directory");
        let path = directory.path().join("input.mp3");
        let audio = speech_like();
        crate::encode::write_audio(&path, &audio, crate::EncodeOptions::default())
            .expect("encode recommendation MP3");

        let report = recommend_file_with_options(
            &path,
            RecommendationOptions::new().with_accelerator(AcceleratorPreference::Cpu),
        )
        .expect("recommend MP3");
        assert_eq!(report.input.format, "mp3");
        assert_eq!(report.input.codec, "mp3");
        assert_eq!(report.input.analysis_mode, "bounded-stream");
        assert_eq!(report.input.sample_rate, audio.sample_rate);
        assert_eq!(report.input.channels, audio.channels());
        assert!(report.input.analyzed_frames > 0);
    }

    #[test]
    fn opus_file_recommendation_uses_the_granule_aware_stream_reader() {
        let directory = tempfile::tempdir().expect("create recommendation directory");
        let path = directory.path().join("input.opus");
        let audio = speech_like();
        crate::encode::write_audio(&path, &audio, crate::EncodeOptions::default())
            .expect("encode recommendation Opus");

        let report = recommend_file_with_options(
            &path,
            RecommendationOptions::new().with_accelerator(AcceleratorPreference::Cpu),
        )
        .expect("recommend Opus");
        assert_eq!(report.input.format, "ogg-opus");
        assert_eq!(report.input.codec, "opus");
        assert_eq!(report.input.analysis_mode, "bounded-stream");
        assert_eq!(report.input.sample_rate, 48_000);
        assert_eq!(report.input.channels, audio.channels());
        assert!(report.input.analyzed_frames > 0);
    }

    #[test]
    fn adts_aac_file_recommendation_uses_the_frame_aware_stream_reader() {
        const SILENT_STEREO_ADTS: [u8; 13] = [
            0xff, 0xf1, 0x50, 0x80, 0x01, 0xbf, 0xfc, 0x21, 0x00, 0x00, 0x00, 0x00, 0x1c,
        ];
        let directory = tempfile::tempdir().expect("create recommendation directory");
        let path = directory.path().join("input.aac");
        std::fs::write(&path, SILENT_STEREO_ADTS.repeat(3)).expect("write recommendation ADTS AAC");

        let report = recommend_file_with_options(
            &path,
            RecommendationOptions::new().with_accelerator(AcceleratorPreference::Cpu),
        )
        .expect("recommend ADTS AAC");
        assert_eq!(report.input.format, "aac-adts");
        assert_eq!(report.input.codec, "aac");
        assert_eq!(report.input.analysis_mode, "bounded-stream");
        assert_eq!(report.input.sample_rate, 44_100);
        assert_eq!(report.input.channels, 2);
        assert_eq!(report.input.analyzed_frames, 3 * 1_024);
    }

    #[test]
    fn low_memory_goal_keeps_a_runnable_fallback() {
        let limits = DecodeLimits::default().with_max_working_set_bytes(Some(2 * 1024 * 1024));
        let report = recommend_audio(
            &speech_like(),
            RecommendationOptions::new()
                .with_goal(RecommendationGoal::LowMemory)
                .with_decode_limits(limits)
                .with_accelerator(AcceleratorPreference::Cpu),
        )
        .expect("low-memory recommendation");
        assert_eq!(report.decision.backend, "classical");
        assert!(report
            .candidates
            .iter()
            .filter(|candidate| candidate.backend != "classical")
            .all(|candidate| !candidate.eligible));
    }

    #[test]
    fn fixed_calibration_fixture_has_stable_identity() {
        let (first, first_hash) = calibration_fixture();
        let (second, second_hash) = calibration_fixture();
        assert_eq!(first_hash, second_hash);
        assert_eq!(first, second);
        assert_eq!(first_hash, CALIBRATION_FIXTURE_SHA256);
    }

    #[test]
    fn calibration_produces_finite_positive_evidence() {
        let evidence = run_device_calibration(1).expect("calibrate");
        assert_eq!(evidence.workload, CALIBRATION_WORKLOAD);
        assert_eq!(evidence.measured_runs, 1);
        assert!(evidence.median_elapsed_ms > 0.0);
        assert!(evidence.baseline_realtime_headroom > 0.0);
        assert!(evidence.median_elapsed_ms.is_finite());
        assert!(evidence.baseline_realtime_headroom.is_finite());
    }

    #[test]
    fn calibration_respects_the_decode_working_set_before_running() {
        let audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0]],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        };
        let limits = DecodeLimits::default().with_max_working_set_bytes(Some(1024 * 1024));
        let error = recommend_audio(
            &audio,
            RecommendationOptions::new()
                .with_calibration(true)
                .with_decode_limits(limits),
        )
        .unwrap_err();
        assert!(
            error.contains("recommendation device calibration"),
            "{error}"
        );
    }

    #[test]
    fn malformed_audio_is_rejected_before_candidate_discovery() {
        let mut audio = speech_like();
        audio.channels.push(vec![0.0; 7]);
        let error = recommend_audio(&audio, RecommendationOptions::new()).unwrap_err();
        assert!(error.contains("channel 1"));
    }

    #[test]
    fn gpu_memory_constraints_are_inclusive_and_explain_failures() {
        let mut eligible = true;
        let mut reasons = Vec::new();
        apply_gpu_memory_constraints(128, Some(128), Some(128), &mut eligible, &mut reasons);
        assert!(eligible);
        assert!(reasons.iter().any(|reason| reason.code == "gpu-memory-fit"));

        let mut eligible = true;
        let mut reasons = Vec::new();
        apply_gpu_memory_constraints(128, Some(127), Some(127), &mut eligible, &mut reasons);
        assert!(!eligible);
        assert!(reasons
            .iter()
            .any(|reason| reason.code == "gpu-memory-limit"));
        assert!(reasons
            .iter()
            .any(|reason| reason.code == "device-gpu-memory"));

        let mut eligible = true;
        let mut reasons = Vec::new();
        apply_gpu_memory_constraints(128, None, None, &mut eligible, &mut reasons);
        assert!(eligible);
        assert!(reasons
            .iter()
            .any(|reason| reason.code == "gpu-memory-unreported"));
    }

    #[test]
    fn candidate_sort_is_deterministic() {
        let mut candidates = vec![
            RecommendationCandidate {
                backend: "z".into(),
                preset: "hifi".into(),
                model: None,
                eligible: true,
                score: 50,
                requested_accelerator: "cpu".into(),
                effective_accelerator: Some("cpu".into()),
                accelerator_fallback: None,
                estimated_memory_bytes: Some(0),
                estimated_gpu_memory_bytes: None,
                calibrated_realtime_headroom: None,
                reasons: vec![],
            },
            RecommendationCandidate {
                backend: "a".into(),
                preset: "hifi".into(),
                model: None,
                eligible: true,
                score: 50,
                requested_accelerator: "cpu".into(),
                effective_accelerator: Some("cpu".into()),
                accelerator_fallback: None,
                estimated_memory_bytes: Some(0),
                estimated_gpu_memory_bytes: None,
                calibrated_realtime_headroom: None,
                reasons: vec![],
            },
        ];
        candidates.sort_by(candidate_order);
        assert_eq!(candidates[0].backend, "a");
    }
}
