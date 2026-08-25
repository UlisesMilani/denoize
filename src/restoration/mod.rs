//! Deterministic, inspectable audio restoration.
//!
//! Restoration is deliberately separate from the denoiser configuration. Each
//! operation reports what it detected, what it actually changed, its bounded
//! parameters, confidence, energy delta, and warnings. The exported mask is a
//! complete run-length encoding of every input frame and channel, so automation
//! can distinguish detection, context padding, and replacement without looking
//! at the rendered audio.

mod declick;
mod declip;
mod dehum;
mod wind;
mod wpe;

use crate::audio::{estimate_audio_memory_bytes, Audio};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

pub const RESTORATION_REPORT_SCHEMA: &str = "denoize-restoration-report-v1";
pub const RESTORATION_MASK_SCHEMA: &str = "denoize-restoration-mask-v1";
pub const RESTORATION_SCHEMA_VERSION: u32 = 1;
pub const MAX_RESTORATION_CHANNELS: usize = 64;
pub const MAX_RESTORATION_OPERATIONS: usize = 5;
pub const MAX_RESTORATION_MASK_RUNS: usize = 4_000_000;

// State priority matters when operation masks overlap: a true detection must
// not be downgraded to context padding from a later operation.
const MASK_PADDED: u8 = 1;
const MASK_DETECTED: u8 = 2;
const MASK_REPLACED: u8 = 3;

/// Whether detected damage is repaired or only reported.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RestorationMode {
    #[default]
    Apply,
    DetectOnly,
}

/// Deterministic restoration operations in their canonical execution order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RestorationOperation {
    Declip,
    Declick,
    Dehum,
    Dereverb,
    WindPlosive,
}

impl RestorationOperation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Declip => "declip",
            Self::Declick => "declick",
            Self::Dehum => "dehum",
            Self::Dereverb => "dereverb",
            Self::WindPlosive => "wind-plosive",
        }
    }

    fn bit(self) -> u8 {
        match self {
            Self::Declip => 1,
            Self::Declick => 2,
            Self::Dehum => 4,
            Self::Dereverb => 8,
            Self::WindPlosive => 16,
        }
    }

    fn canonical() -> [Self; MAX_RESTORATION_OPERATIONS] {
        [
            Self::Declip,
            Self::Declick,
            Self::Dehum,
            Self::Dereverb,
            Self::WindPlosive,
        ]
    }
}

/// Robust sliding harmonic-regression settings.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DehumConfig {
    pub minimum_confidence: f64,
    pub maximum_harmonics: usize,
    pub maximum_frequency_hz: f64,
    pub block_seconds: f64,
    pub attenuation_db: f64,
}

impl Default for DehumConfig {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.55,
            maximum_harmonics: 20,
            maximum_frequency_hz: 2_000.0,
            block_seconds: 1.0,
            attenuation_db: 30.0,
        }
    }
}

/// Robust prediction-residual click detection and interpolation settings.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclickConfig {
    pub minimum_confidence: f64,
    pub residual_threshold_mad: f64,
    pub maximum_gap_ms: f64,
    pub merge_gap_ms: f64,
    pub context_ms: f64,
    pub prediction_order: usize,
}

impl Default for DeclickConfig {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.6,
            residual_threshold_mad: 10.0,
            maximum_gap_ms: 3.0,
            merge_gap_ms: 0.15,
            context_ms: 8.0,
            prediction_order: 12,
        }
    }
}

/// Analysis-sparse clipping reconstruction settings.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclipConfig {
    pub minimum_confidence: f64,
    pub threshold_tolerance: f64,
    pub minimum_run_samples: usize,
    pub maximum_region_ms: f64,
    pub context_ms: f64,
    pub iterations: usize,
}

impl Default for DeclipConfig {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.65,
            threshold_tolerance: 0.002,
            minimum_run_samples: 2,
            maximum_region_ms: 20.0,
            context_ms: 12.0,
            iterations: 24,
        }
    }
}

/// Finite weighted-prediction-error dereverberation settings.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WpeChannelMode {
    #[default]
    Independent,
    Multichannel,
}

/// Finite weighted-prediction-error dereverberation settings.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WpeConfig {
    pub minimum_confidence: f64,
    pub channel_mode: WpeChannelMode,
    pub frame_size: usize,
    pub hop_size: usize,
    pub prediction_delay_frames: usize,
    pub prediction_taps: usize,
    pub iterations: usize,
    pub regularization: f64,
    pub maximum_attenuation_db: f64,
}

impl Default for WpeConfig {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.35,
            channel_mode: WpeChannelMode::Independent,
            frame_size: 512,
            hop_size: 128,
            prediction_delay_frames: 3,
            prediction_taps: 8,
            iterations: 3,
            regularization: 1e-5,
            maximum_attenuation_db: 12.0,
        }
    }
}

/// Conservative low-frequency burst repair settings.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WindPlosiveConfig {
    pub minimum_confidence: f64,
    pub window_ms: f64,
    pub maximum_burst_ms: f64,
    pub low_band_hz: f64,
    pub ratio_threshold: f64,
    pub maximum_attenuation_db: f64,
}

impl Default for WindPlosiveConfig {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.68,
            window_ms: 20.0,
            maximum_burst_ms: 350.0,
            low_band_hz: 180.0,
            ratio_threshold: 5.0,
            maximum_attenuation_db: 18.0,
        }
    }
}

/// Complete deterministic restoration configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RestorationConfig {
    pub mode: RestorationMode,
    pub operations: Vec<RestorationOperation>,
    pub dehum: DehumConfig,
    pub declick: DeclickConfig,
    pub declip: DeclipConfig,
    pub dereverb: WpeConfig,
    pub wind_plosive: WindPlosiveConfig,
}

impl Default for RestorationConfig {
    fn default() -> Self {
        Self {
            mode: RestorationMode::Apply,
            operations: RestorationOperation::canonical().to_vec(),
            dehum: DehumConfig::default(),
            declick: DeclickConfig::default(),
            declip: DeclipConfig::default(),
            dereverb: WpeConfig::default(),
            wind_plosive: WindPlosiveConfig::default(),
        }
    }
}

impl RestorationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.operations.is_empty() || self.operations.len() > MAX_RESTORATION_OPERATIONS {
            return Err("restoration operations must contain between 1 and 5 entries".into());
        }
        let mut unique = BTreeSet::new();
        for operation in &self.operations {
            if !unique.insert(*operation) {
                return Err(format!(
                    "restoration operation {} is duplicated",
                    operation.name()
                ));
            }
        }
        validate_unit_interval("dehum.minimum_confidence", self.dehum.minimum_confidence)?;
        validate_unit_interval(
            "declick.minimum_confidence",
            self.declick.minimum_confidence,
        )?;
        validate_unit_interval("declip.minimum_confidence", self.declip.minimum_confidence)?;
        validate_unit_interval(
            "dereverb.minimum_confidence",
            self.dereverb.minimum_confidence,
        )?;
        validate_unit_interval(
            "wind_plosive.minimum_confidence",
            self.wind_plosive.minimum_confidence,
        )?;
        validate_finite_range("dehum.block_seconds", self.dehum.block_seconds, 0.25, 4.0)?;
        if !(1..=64).contains(&self.dehum.maximum_harmonics) {
            return Err("dehum.maximum_harmonics must be in 1..=64".into());
        }
        validate_finite_range(
            "dehum.maximum_frequency_hz",
            self.dehum.maximum_frequency_hz,
            100.0,
            20_000.0,
        )?;
        validate_finite_range("dehum.attenuation_db", self.dehum.attenuation_db, 0.0, 80.0)?;
        validate_finite_range(
            "declick.residual_threshold_mad",
            self.declick.residual_threshold_mad,
            4.0,
            40.0,
        )?;
        validate_finite_range(
            "declick.maximum_gap_ms",
            self.declick.maximum_gap_ms,
            0.02,
            20.0,
        )?;
        validate_finite_range("declick.merge_gap_ms", self.declick.merge_gap_ms, 0.0, 2.0)?;
        validate_finite_range("declick.context_ms", self.declick.context_ms, 1.0, 100.0)?;
        if !(2..=64).contains(&self.declick.prediction_order) {
            return Err("declick.prediction_order must be in 2..=64".into());
        }
        validate_finite_range(
            "declip.threshold_tolerance",
            self.declip.threshold_tolerance,
            1e-6,
            0.05,
        )?;
        if !(2..=64).contains(&self.declip.minimum_run_samples) {
            return Err("declip.minimum_run_samples must be in 2..=64".into());
        }
        validate_finite_range(
            "declip.maximum_region_ms",
            self.declip.maximum_region_ms,
            0.02,
            100.0,
        )?;
        validate_finite_range("declip.context_ms", self.declip.context_ms, 1.0, 200.0)?;
        if !(1..=128).contains(&self.declip.iterations) {
            return Err("declip.iterations must be in 1..=128".into());
        }
        if !(128..=4096).contains(&self.dereverb.frame_size)
            || !self.dereverb.frame_size.is_power_of_two()
        {
            return Err("dereverb.frame_size must be a power of two in 128..=4096".into());
        }
        if self.dereverb.hop_size == 0
            || self.dereverb.hop_size > self.dereverb.frame_size / 2
            || self.dereverb.frame_size % self.dereverb.hop_size != 0
        {
            return Err(
                "dereverb.hop_size must divide frame_size and be at most half a frame".into(),
            );
        }
        if !(1..=20).contains(&self.dereverb.prediction_delay_frames) {
            return Err("dereverb.prediction_delay_frames must be in 1..=20".into());
        }
        if !(1..=24).contains(&self.dereverb.prediction_taps) {
            return Err("dereverb.prediction_taps must be in 1..=24".into());
        }
        if !(1..=10).contains(&self.dereverb.iterations) {
            return Err("dereverb.iterations must be in 1..=10".into());
        }
        validate_finite_range(
            "dereverb.regularization",
            self.dereverb.regularization,
            1e-12,
            1.0,
        )?;
        validate_finite_range(
            "dereverb.maximum_attenuation_db",
            self.dereverb.maximum_attenuation_db,
            0.0,
            40.0,
        )?;
        validate_finite_range(
            "wind_plosive.window_ms",
            self.wind_plosive.window_ms,
            2.0,
            100.0,
        )?;
        validate_finite_range(
            "wind_plosive.maximum_burst_ms",
            self.wind_plosive.maximum_burst_ms,
            self.wind_plosive.window_ms,
            2_000.0,
        )?;
        validate_finite_range(
            "wind_plosive.low_band_hz",
            self.wind_plosive.low_band_hz,
            40.0,
            500.0,
        )?;
        validate_finite_range(
            "wind_plosive.ratio_threshold",
            self.wind_plosive.ratio_threshold,
            1.25,
            50.0,
        )?;
        validate_finite_range(
            "wind_plosive.maximum_attenuation_db",
            self.wind_plosive.maximum_attenuation_db,
            0.0,
            40.0,
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RestorationStatus {
    Applied,
    Detected,
    Skipped,
}

/// Operation-specific, closed report details.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum RestorationOperationDetails {
    Dehum {
        fundamental_hz: Option<f64>,
        tracked_blocks: usize,
        harmonic_count: usize,
        attenuation_db: f64,
    },
    Declick {
        regions: usize,
        rejected_regions: usize,
        prediction_order: usize,
        maximum_gap_samples: usize,
    },
    Declip {
        regions: usize,
        rejected_regions: usize,
        positive_threshold: Option<f64>,
        negative_threshold: Option<f64>,
        iterations: usize,
        converged_regions: usize,
    },
    Dereverb {
        channel_mode: WpeChannelMode,
        frame_size: usize,
        hop_size: usize,
        prediction_delay_frames: usize,
        prediction_taps: usize,
        effective_context_frames: usize,
        iterations: usize,
        solved_bins: usize,
        ill_conditioned_bins: usize,
        convergence: f64,
    },
    WindPlosive {
        regions: usize,
        rejected_regions: usize,
        low_band_hz: f64,
        maximum_attenuation_db: f64,
        stereo_coherence: Option<f64>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RestorationOperationReport {
    pub operation: RestorationOperation,
    pub status: RestorationStatus,
    pub detected_samples: usize,
    pub changed_samples: usize,
    pub confidence: f64,
    pub energy_delta_db: f64,
    pub warnings: Vec<String>,
    pub details: RestorationOperationDetails,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RestorationMaskState {
    Untouched,
    Detected,
    Padded,
    Replaced,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RestorationMaskRun {
    pub channel: usize,
    pub start_frame: usize,
    pub frame_count: usize,
    pub state: RestorationMaskState,
    pub operations: Vec<RestorationOperation>,
    pub confidence: f64,
}

/// Complete same-length RLE mask. Runs cover every frame exactly once per
/// channel and appear in channel/start order.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RestorationMask {
    pub schema: String,
    pub schema_version: u32,
    pub channels: usize,
    pub frames: usize,
    pub runs: Vec<RestorationMaskRun>,
}

impl RestorationMask {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != RESTORATION_MASK_SCHEMA
            || self.schema_version != RESTORATION_SCHEMA_VERSION
        {
            return Err("unsupported restoration mask schema".into());
        }
        if self.channels == 0 || self.channels > MAX_RESTORATION_CHANNELS {
            return Err(format!(
                "restoration mask channel count must be in 1..={MAX_RESTORATION_CHANNELS}"
            ));
        }
        if self.runs.len() > MAX_RESTORATION_MASK_RUNS {
            return Err(format!(
                "restoration mask exceeds the bounded {MAX_RESTORATION_MASK_RUNS}-run limit"
            ));
        }
        let mut cursor = vec![0usize; self.channels];
        let mut previous_position = None;
        for run in &self.runs {
            if run.channel >= self.channels || run.frame_count == 0 {
                return Err("restoration mask run has invalid geometry".into());
            }
            if run.start_frame != cursor[run.channel]
                || run.start_frame.saturating_add(run.frame_count) > self.frames
            {
                return Err("restoration mask runs do not form exact channel coverage".into());
            }
            let position = (run.channel, run.start_frame);
            if previous_position.is_some_and(|previous| previous >= position) {
                return Err("restoration mask runs must be in channel/start order".into());
            }
            previous_position = Some(position);
            if !run.confidence.is_finite() || !(0.0..=1.0).contains(&run.confidence) {
                return Err("restoration mask confidence must be finite and in 0..=1".into());
            }
            if run.state == RestorationMaskState::Untouched {
                if !run.operations.is_empty() || run.confidence != 0.0 {
                    return Err("untouched restoration mask runs cannot name operations".into());
                }
            } else if run.operations.is_empty() {
                return Err("changed restoration mask runs must name an operation".into());
            }
            if run.operations.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err("restoration mask operations must be unique and sorted".into());
            }
            cursor[run.channel] = cursor[run.channel]
                .checked_add(run.frame_count)
                .ok_or("restoration mask coverage overflows usize")?;
        }
        if cursor.iter().any(|covered| *covered != self.frames) {
            return Err("restoration mask does not cover every channel frame".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RestorationReport {
    pub schema: String,
    pub schema_version: u32,
    pub mode: RestorationMode,
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: usize,
    pub input_pcm_sha256: String,
    pub mask_sha256: String,
    pub deterministic: bool,
    pub bypassed: bool,
    pub detected_samples: usize,
    pub changed_samples: usize,
    pub confidence: f64,
    pub energy_delta_db: f64,
    pub operations: Vec<RestorationOperationReport>,
    pub warnings: Vec<String>,
}

impl RestorationReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|error| format!("serialize restoration report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize restoration report: {error}"))
    }
}

#[derive(Clone, Debug)]
pub struct RestorationResult {
    pub audio: Audio,
    pub report: RestorationReport,
    pub mask: RestorationMask,
}

#[derive(Clone, Copy, Debug, Default)]
struct MaskCell {
    state: u8,
    operations: u8,
    detected: bool,
    confidence: f64,
}

struct OperationOutcome {
    detected_samples: usize,
    changed_samples: usize,
    confidence: f64,
    warnings: Vec<String>,
    details: RestorationOperationDetails,
}

/// Conservative working-set estimate for admission control.
pub fn estimate_restoration_memory_bytes(audio: &Audio, config: &RestorationConfig) -> u64 {
    let base = estimate_audio_memory_bytes(audio);
    let samples = (audio.channels() as u64).saturating_mul(audio.frames() as u64);
    let cells = samples.saturating_mul(std::mem::size_of::<MaskCell>() as u64);
    // WPE holds input/output PCM, two full STFT planes, a synthesized
    // candidate, and bounded regression scratch concurrently. The other
    // operations peak at input/output plus at most three signal-sized work
    // buffers. Keep admission deliberately conservative.
    let processing_peak = if config.operations.contains(&RestorationOperation::Dereverb) {
        base.saturating_mul(12).saturating_add(cells)
    } else {
        base.saturating_mul(6).saturating_add(cells)
    };
    // A highly alternating detector result can produce one RLE run per
    // sample. Account for both the run object and its maximum closed operation
    // list at the separate mask-encoding peak.
    let maximum_runs = samples.min(MAX_RESTORATION_MASK_RUNS as u64);
    let run_bytes = maximum_runs.saturating_mul(
        (std::mem::size_of::<RestorationMaskRun>()
            + MAX_RESTORATION_OPERATIONS * std::mem::size_of::<RestorationOperation>())
            as u64,
    );
    let mask_peak = base
        .saturating_mul(2)
        .saturating_add(cells)
        .saturating_add(run_bytes);
    processing_peak.max(mask_peak).max(1024 * 1024)
}

/// Restore an in-memory signal without changing its sample rate, channel count,
/// or frame count.
pub fn restore_audio(
    input: &Audio,
    config: &RestorationConfig,
) -> Result<RestorationResult, String> {
    config.validate()?;
    validate_audio(input)?;
    let input_digest = input_pcm_digest(input);
    let mut output = input.try_clone_fallible("restoration output")?;
    let mut mask = allocate_mask(input.channels(), input.frames())?;
    let selected: BTreeSet<_> = config.operations.iter().copied().collect();
    let mut operation_reports = Vec::new();
    operation_reports
        .try_reserve_exact(selected.len())
        .map_err(|_| "unable to reserve restoration operation reports".to_string())?;

    for operation in RestorationOperation::canonical() {
        if !selected.contains(&operation) {
            continue;
        }
        let before = signal_energy(&output.channels);
        let outcome = match operation {
            RestorationOperation::Declip => declip::process(
                &mut output.channels,
                input.sample_rate,
                config.mode,
                &mut mask,
                &config.declip,
            )?,
            RestorationOperation::Declick => declick::process(
                &mut output.channels,
                input.sample_rate,
                config.mode,
                &mut mask,
                &config.declick,
            )?,
            RestorationOperation::Dehum => dehum::process(
                &mut output.channels,
                input.sample_rate,
                config.mode,
                &mut mask,
                &config.dehum,
            )?,
            RestorationOperation::Dereverb => wpe::process(
                &mut output.channels,
                input.sample_rate,
                config.mode,
                &mut mask,
                &config.dereverb,
            )?,
            RestorationOperation::WindPlosive => wind::process(
                &mut output.channels,
                input.sample_rate,
                config.mode,
                &mut mask,
                &config.wind_plosive,
            )?,
        };
        let after = signal_energy(&output.channels);
        let status = if outcome.detected_samples == 0 {
            RestorationStatus::Skipped
        } else if config.mode == RestorationMode::DetectOnly || outcome.changed_samples == 0 {
            RestorationStatus::Detected
        } else {
            RestorationStatus::Applied
        };
        operation_reports.push(RestorationOperationReport {
            operation,
            status,
            detected_samples: outcome.detected_samples,
            changed_samples: outcome.changed_samples,
            confidence: finite_confidence(outcome.confidence),
            energy_delta_db: energy_delta_db(before, after),
            warnings: outcome.warnings,
            details: outcome.details,
        });
    }

    if output.channels.len() != input.channels.len()
        || output
            .channels
            .iter()
            .zip(&input.channels)
            .any(|(after, before)| after.len() != before.len())
    {
        return Err("restoration operation changed the signal geometry".into());
    }
    if output
        .channels
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err("restoration produced an invalid normalized sample".into());
    }
    if config.mode == RestorationMode::DetectOnly
        && output
            .channels
            .iter()
            .flatten()
            .zip(input.channels.iter().flatten())
            .any(|(after, before)| after.to_bits() != before.to_bits())
    {
        return Err("detect-only restoration modified audio".into());
    }

    let detected_samples = mask.iter().flatten().filter(|cell| cell.detected).count();
    let confidence = weighted_detection_confidence(&mask);
    let mask = encode_mask(&mask, input.frames())?;
    mask.validate()?;
    let mask_bytes = serde_json::to_vec(&mask)
        .map_err(|error| format!("serialize restoration mask for digest: {error}"))?;
    let mask_digest = hex_digest(&mask_bytes);
    let changed_samples = output
        .channels
        .iter()
        .flatten()
        .zip(input.channels.iter().flatten())
        .filter(|(after, before)| after.to_bits() != before.to_bits())
        .count();
    let mut warnings: Vec<String> = operation_reports
        .iter()
        .flat_map(|operation| operation.warnings.iter().cloned())
        .collect();
    warnings.sort();
    warnings.dedup();
    let report = RestorationReport {
        schema: RESTORATION_REPORT_SCHEMA.into(),
        schema_version: RESTORATION_SCHEMA_VERSION,
        mode: config.mode,
        sample_rate: input.sample_rate,
        channels: input.channels(),
        frames: input.frames(),
        input_pcm_sha256: input_digest,
        mask_sha256: mask_digest,
        deterministic: true,
        bypassed: changed_samples == 0,
        detected_samples,
        changed_samples,
        confidence,
        energy_delta_db: energy_delta_db(
            signal_energy(&input.channels),
            signal_energy(&output.channels),
        ),
        operations: operation_reports,
        warnings,
    };
    Ok(RestorationResult {
        audio: output,
        report,
        mask,
    })
}

fn validate_audio(audio: &Audio) -> Result<(), String> {
    if audio.sample_rate == 0 || audio.sample_rate > crate::config::MAX_SAMPLE_RATE {
        return Err(format!(
            "restoration sample rate must be in 1..={}",
            crate::config::MAX_SAMPLE_RATE
        ));
    }
    if audio.channels.is_empty() || audio.channels.len() > MAX_RESTORATION_CHANNELS {
        return Err(format!(
            "restoration channel count must be in 1..={MAX_RESTORATION_CHANNELS}"
        ));
    }
    let frames = audio.frames();
    if audio.channels.iter().any(|channel| channel.len() != frames) {
        return Err("restoration input channels must have equal frame counts".into());
    }
    if audio
        .channels
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err("restoration input must contain finite normalized samples".into());
    }
    Ok(())
}

fn allocate_mask(channels: usize, frames: usize) -> Result<Vec<Vec<MaskCell>>, String> {
    let mut mask = Vec::new();
    mask.try_reserve_exact(channels)
        .map_err(|_| "unable to reserve restoration mask channels".to_string())?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|_| "unable to reserve restoration mask frames".to_string())?;
        channel.resize(frames, MaskCell::default());
        mask.push(channel);
    }
    Ok(mask)
}

fn mark_range(
    mask: &mut [Vec<MaskCell>],
    channel: usize,
    start: usize,
    end: usize,
    state: u8,
    operation: RestorationOperation,
    confidence: f64,
) {
    let Some(channel_mask) = mask.get_mut(channel) else {
        return;
    };
    let length = channel_mask.len();
    for cell in &mut channel_mask[start.min(length)..end.min(length)] {
        if state == MASK_DETECTED {
            cell.detected = true;
        }
        cell.state = cell.state.max(state);
        cell.operations |= operation.bit();
        cell.confidence = cell.confidence.max(finite_confidence(confidence));
    }
}

fn encode_mask(mask: &[Vec<MaskCell>], frames: usize) -> Result<RestorationMask, String> {
    let mut runs = Vec::new();
    for (channel_index, channel) in mask.iter().enumerate() {
        if channel.is_empty() {
            continue;
        }
        let mut start = 0;
        while start < channel.len() {
            let seed = channel[start];
            let mut end = start + 1;
            while end < channel.len()
                && channel[end].state == seed.state
                && channel[end].operations == seed.operations
                && channel[end].confidence.to_bits() == seed.confidence.to_bits()
            {
                end += 1;
            }
            if runs.len() >= MAX_RESTORATION_MASK_RUNS {
                return Err(format!(
                    "restoration mask exceeds the bounded {MAX_RESTORATION_MASK_RUNS}-run limit"
                ));
            }
            runs.try_reserve(1)
                .map_err(|_| "unable to reserve restoration mask runs".to_string())?;
            runs.push(RestorationMaskRun {
                channel: channel_index,
                start_frame: start,
                frame_count: end - start,
                state: match seed.state {
                    0 => RestorationMaskState::Untouched,
                    MASK_PADDED => RestorationMaskState::Padded,
                    MASK_DETECTED => RestorationMaskState::Detected,
                    MASK_REPLACED => RestorationMaskState::Replaced,
                    _ => return Err("restoration mask has an invalid state".into()),
                },
                operations: operations_from_bits(seed.operations)?,
                confidence: finite_confidence(seed.confidence),
            });
            start = end;
        }
    }
    Ok(RestorationMask {
        schema: RESTORATION_MASK_SCHEMA.into(),
        schema_version: RESTORATION_SCHEMA_VERSION,
        channels: mask.len(),
        frames,
        runs,
    })
}

fn operations_from_bits(bits: u8) -> Result<Vec<RestorationOperation>, String> {
    let mut operations = Vec::new();
    let operation_count = bits.count_ones() as usize;
    operations
        .try_reserve_exact(operation_count)
        .map_err(|_| "unable to reserve restoration mask operations".to_string())?;
    operations.extend(
        RestorationOperation::canonical()
            .into_iter()
            .filter(|operation| bits & operation.bit() != 0),
    );
    Ok(operations)
}

fn input_pcm_digest(audio: &Audio) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"denoize-restoration-input-pcm-v1\0");
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

fn hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn signal_energy(channels: &[Vec<f64>]) -> f64 {
    channels
        .iter()
        .flatten()
        .map(|sample| sample * sample)
        .sum::<f64>()
}

fn energy_delta_db(before: f64, after: f64) -> f64 {
    if before <= f64::MIN_POSITIVE && after <= f64::MIN_POSITIVE {
        return 0.0;
    }
    (10.0 * ((after + 1e-30) / (before + 1e-30)).log10()).clamp(-240.0, 240.0)
}

fn finite_confidence(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn weighted_detection_confidence(mask: &[Vec<MaskCell>]) -> f64 {
    let mut weight = 0usize;
    let mut total = 0.0;
    for cell in mask.iter().flatten() {
        if cell.detected {
            weight = weight.saturating_add(1);
            total += cell.confidence;
        }
    }
    if weight == 0 {
        0.0
    } else {
        finite_confidence(total / weight as f64)
    }
}

fn validate_unit_interval(field: &str, value: f64) -> Result<(), String> {
    validate_finite_range(field, value, 0.0, 1.0)
}

fn validate_finite_range(field: &str, value: f64, min: f64, max: f64) -> Result<(), String> {
    if !value.is_finite() || value < min || value > max {
        Err(format!("{field} must be finite and in {min}..={max}"))
    } else {
        Ok(())
    }
}

fn milliseconds_to_samples(milliseconds: f64, sample_rate: u32, minimum: usize) -> usize {
    ((milliseconds * sample_rate as f64 / 1_000.0).round() as usize).max(minimum)
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

fn median_absolute_deviation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut center_values = values.to_vec();
    let center = median(&mut center_values);
    let mut deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    median(&mut deviations)
}

fn mark_changed_samples(
    before: &[f64],
    after: &[f64],
    mask: &mut [Vec<MaskCell>],
    channel: usize,
    operation: RestorationOperation,
    confidence: f64,
) -> usize {
    let mut changed = 0;
    for index in 0..before.len().min(after.len()) {
        if before[index].to_bits() != after[index].to_bits() {
            changed += 1;
            mark_range(
                mask,
                channel,
                index,
                index + 1,
                MASK_REPLACED,
                operation,
                confidence,
            );
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn audio(channels: Vec<Vec<f64>>, sample_rate: u32) -> Audio {
        Audio {
            sample_rate,
            channels,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    #[test]
    fn clean_silence_bypasses_bit_exactly() {
        let input = audio(vec![vec![0.0; 4_800]], 48_000);
        let result = restore_audio(&input, &RestorationConfig::default()).unwrap();
        assert!(result.report.bypassed);
        assert_eq!(result.report.changed_samples, 0);
        assert_eq!(result.audio.channels[0], input.channels[0]);
        result.mask.validate().unwrap();
    }

    #[test]
    fn detect_only_never_modifies_audio() {
        let mut samples = vec![0.0; 4_800];
        samples[2_400] = 0.95;
        let input = audio(vec![samples], 48_000);
        let mut config = RestorationConfig::default();
        config.mode = RestorationMode::DetectOnly;
        config.operations = vec![RestorationOperation::Declick];
        let result = restore_audio(&input, &config).unwrap();
        assert_eq!(result.audio.channels[0], input.channels[0]);
        assert_eq!(result.report.changed_samples, 0);
        assert!(result.report.detected_samples > 0);
        assert!(result
            .mask
            .runs
            .iter()
            .all(|run| run.state != RestorationMaskState::Replaced));
    }

    #[test]
    fn configuration_is_closed_and_bounded() {
        let mut config = RestorationConfig::default();
        config.operations.push(RestorationOperation::Dehum);
        assert!(config.validate().is_err());
        let json = serde_json::to_string(&RestorationConfig::default()).unwrap();
        let with_unknown = json.replacen('{', "{\"unknown\":true,", 1);
        assert!(serde_json::from_str::<RestorationConfig>(&with_unknown).is_err());
    }

    #[test]
    fn malformed_audio_is_rejected_before_allocation_heavy_dsp() {
        let ragged = audio(vec![vec![0.0; 8], vec![0.0; 7]], 48_000);
        assert!(restore_audio(&ragged, &RestorationConfig::default()).is_err());
        let invalid = audio(vec![vec![f64::NAN]], 48_000);
        assert!(restore_audio(&invalid, &RestorationConfig::default()).is_err());
    }

    #[test]
    fn mask_validation_requires_exact_per_channel_coverage() {
        let mask = RestorationMask {
            schema: RESTORATION_MASK_SCHEMA.into(),
            schema_version: RESTORATION_SCHEMA_VERSION,
            channels: 1,
            frames: 10,
            runs: vec![RestorationMaskRun {
                channel: 0,
                start_frame: 0,
                frame_count: 9,
                state: RestorationMaskState::Untouched,
                operations: vec![],
                confidence: 0.0,
            }],
        };
        assert!(mask.validate().is_err());
    }

    #[test]
    fn mask_validation_rejects_out_of_order_and_unbounded_geometry() {
        let untouched = |channel| RestorationMaskRun {
            channel,
            start_frame: 0,
            frame_count: 10,
            state: RestorationMaskState::Untouched,
            operations: vec![],
            confidence: 0.0,
        };
        let out_of_order = RestorationMask {
            schema: RESTORATION_MASK_SCHEMA.into(),
            schema_version: RESTORATION_SCHEMA_VERSION,
            channels: 2,
            frames: 10,
            runs: vec![untouched(1), untouched(0)],
        };
        assert!(out_of_order.validate().is_err());
        let unbounded_channels = RestorationMask {
            schema: RESTORATION_MASK_SCHEMA.into(),
            schema_version: RESTORATION_SCHEMA_VERSION,
            channels: MAX_RESTORATION_CHANNELS + 1,
            frames: 0,
            runs: vec![],
        };
        assert!(unbounded_channels.validate().is_err());
    }

    #[test]
    fn memory_admission_accounts_for_wpe_and_worst_case_mask_encoding() {
        let input = audio(vec![vec![0.0; 48_000]], 48_000);
        let mut local = RestorationConfig::default();
        local.operations = vec![RestorationOperation::Declick];
        let mut wpe = local.clone();
        wpe.operations = vec![RestorationOperation::Dereverb];
        let local_bytes = estimate_restoration_memory_bytes(&input, &local);
        let wpe_bytes = estimate_restoration_memory_bytes(&input, &wpe);
        assert!(local_bytes >= 1024 * 1024);
        assert!(wpe_bytes > local_bytes);
    }
}
