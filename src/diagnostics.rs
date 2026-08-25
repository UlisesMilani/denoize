//! Bounded, network-free audio degradation diagnosis and no-reference quality
//! assessment.
//!
//! The native estimator is intentionally transparent: it reports independent
//! signal measurements, confidence, and uncertainty instead of presenting a
//! learned MOS predictor as ground truth.  It is suitable for triage and
//! regression screening; release acceptance still requires the reference and
//! listening-test evidence in [`crate::evaluation`].

use crate::decode::DecodeBudget;
use crate::fft::{Complex, Fft};
use crate::resample::{resample, resampler_plan_bytes, StreamingResampler};
use crate::{
    sanitize_sample, Audio, AudioCodec, AudioFormat, AudioInputSession, AudioStreamReader,
    DecodeLimits,
};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::cmp::Ordering;
use std::path::Path;

/// Stable schema identifier for a degradation diagnosis.
pub const DIAGNOSTIC_SCHEMA: &str = "denoize-diagnostic-v1";
/// Stable schema identifier for a no-reference assessment or before/after
/// comparison.
pub const ASSESSMENT_SCHEMA: &str = "denoize-assessment-v1";
pub const DIAGNOSTIC_SCHEMA_VERSION: u32 = 1;
pub const ASSESSMENT_SCHEMA_VERSION: u32 = 1;

const DEFAULT_ANALYSIS_SECONDS: u32 = 12;
const MAX_ANALYSIS_SECONDS: u32 = 60;
const MAX_ANALYSIS_SAMPLE_RATE: u32 = 48_000;
const ANALYSIS_BLOCK_FRAMES: usize = 4_096;
const ANALYSIS_FIXED_SCRATCH_BYTES: u64 = 128 * 1024;
const ANALYSIS_DOMAIN: &[u8] = b"denoize-native-diagnostic-v1\0";
const EPSILON: f64 = 1.0e-15;

/// Bounded input and memory options shared by diagnosis and assessment.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiagnosticOptions {
    analysis_seconds: u32,
    decode_limits: DecodeLimits,
}

impl Default for DiagnosticOptions {
    fn default() -> Self {
        Self {
            analysis_seconds: DEFAULT_ANALYSIS_SECONDS,
            decode_limits: DecodeLimits::default(),
        }
    }
}

impl DiagnosticOptions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_analysis_seconds(mut self, seconds: u32) -> Self {
        self.analysis_seconds = seconds;
        self
    }

    #[must_use]
    pub const fn with_decode_limits(mut self, limits: DecodeLimits) -> Self {
        self.decode_limits = limits;
        self
    }

    #[must_use]
    pub const fn analysis_seconds(self) -> u32 {
        self.analysis_seconds
    }

    #[must_use]
    pub const fn decode_limits(self) -> DecodeLimits {
        self.decode_limits
    }

    pub fn validate(self) -> Result<(), String> {
        if !(1..=MAX_ANALYSIS_SECONDS).contains(&self.analysis_seconds) {
            return Err(format!(
                "diagnostic analysis duration must be between 1 and {MAX_ANALYSIS_SECONDS} seconds"
            ));
        }
        Ok(())
    }
}

/// One independently reported degradation family.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DiagnosticFinding {
    pub kind: String,
    pub detected: bool,
    pub confidence: f64,
    pub severity: f64,
    pub evidence: String,
    pub recommended_action: String,
}

/// Direct measurements used by the native estimator.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DiagnosticMetrics {
    pub rms_dbfs: f64,
    pub peak_dbfs: f64,
    pub dc_offset: f64,
    pub clipped_sample_ratio: f64,
    pub flat_clip_run_ratio: f64,
    pub click_rate_per_second: f64,
    pub dropout_seconds: f64,
    pub spectral_flatness: f64,
    pub low_frequency_energy_ratio: f64,
    pub high_frequency_energy_ratio: f64,
    pub estimated_bandwidth_hz: f64,
    pub noise_floor_dbfs: f64,
    pub estimated_snr_db: f64,
    pub late_energy_ratio: f64,
    pub hum_frequency_hz: Option<f64>,
    pub hum_prominence_db: f64,
}

/// No-reference quality dimensions.  These are calibrated native proxy scores,
/// not human MOS labels.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NoReferenceQuality {
    pub method: String,
    pub score: f64,
    pub estimated_mos_proxy: f64,
    pub uncertainty: f64,
    pub noise_cleanliness: f64,
    pub distortion_freedom: f64,
    pub spectral_completeness: f64,
    pub continuity: f64,
}

/// Bounded input facts without a filesystem pathname.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DiagnosticInput {
    pub format: String,
    pub codec: String,
    pub sample_rate: u32,
    pub analysis_sample_rate: u32,
    pub channels: usize,
    pub total_frames: Option<u64>,
    pub source_analyzed_frames: usize,
    pub analyzed_frames: usize,
    pub analyzed_seconds: f64,
    pub analysis_mode: String,
    pub analysis_sha256: String,
}

/// Complete native degradation report.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DiagnosticReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub network_accessed: bool,
    pub input: DiagnosticInput,
    pub quality: NoReferenceQuality,
    pub metrics: DiagnosticMetrics,
    pub findings: Vec<DiagnosticFinding>,
    pub recommended_pipeline: Vec<String>,
    pub limitations: Vec<String>,
}

impl DiagnosticReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|error| format!("serialize diagnostic report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize diagnostic report: {error}"))
    }
}

/// Geometry and score changes in a before/after comparison.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AssessmentComparison {
    pub quality_score_delta: f64,
    pub estimated_mos_proxy_delta: f64,
    pub noise_cleanliness_delta: f64,
    pub distortion_freedom_delta: f64,
    pub spectral_completeness_delta: f64,
    pub continuity_delta: f64,
    pub sample_rate_equal: bool,
    pub channel_count_equal: bool,
    pub total_frame_delta: Option<i64>,
    pub duration_delta_milliseconds: Option<f64>,
    pub presentation_preserved: bool,
    pub semantic_fidelity_assessed: bool,
}

/// Single-input no-reference assessment or a before/after comparison.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AssessmentReport {
    pub schema: String,
    pub schema_version: u32,
    pub denoize_version: String,
    pub network_accessed: bool,
    pub baseline: Option<DiagnosticReport>,
    pub candidate: DiagnosticReport,
    pub comparison: Option<AssessmentComparison>,
    pub verdict: String,
    pub warnings: Vec<String>,
}

impl AssessmentReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|error| format!("serialize assessment report: {error}"))
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize assessment report: {error}"))
    }
}

#[derive(Clone, Debug, Default)]
struct RawPcmStats {
    samples: u64,
    clipped: u64,
    flat_clipped: u64,
    previous: Vec<Option<f64>>,
}

impl RawPcmStats {
    fn ingest(&mut self, channels: &[Vec<f64>], frames: usize) {
        if self.previous.len() != channels.len() {
            self.previous.resize(channels.len(), None);
        }
        for frame in 0..frames {
            for (index, channel) in channels.iter().enumerate() {
                let sample = sanitize_sample(channel[frame]);
                let clipped = sample.abs() >= 0.999;
                self.samples += 1;
                self.clipped += u64::from(clipped);
                if clipped
                    && self.previous[index]
                        .is_some_and(|previous| (sample - previous).abs() <= 1.0e-7)
                {
                    self.flat_clipped += 1;
                }
                self.previous[index] = Some(sample);
            }
        }
    }
}

struct AnalysisPcm {
    samples: Vec<f64>,
    format: AudioFormat,
    codec: AudioCodec,
    source_rate: u32,
    analysis_rate: u32,
    channels: usize,
    total_frames: Option<u64>,
    source_analyzed_frames: usize,
    analysis_mode: &'static str,
    raw: RawPcmStats,
}

#[derive(Default)]
struct SpectralSummary {
    flatness: f64,
    low_ratio: f64,
    high_ratio: f64,
    bandwidth_hz: f64,
    noise_floor_dbfs: f64,
    estimated_snr_db: f64,
    hum_50_db: f64,
    hum_60_db: f64,
    hum_50_support: usize,
    hum_60_support: usize,
    late_energy_ratio: f64,
}

/// Diagnose one regular audio file without network or model-cache access.
pub fn diagnose_file(path: impl AsRef<Path>) -> Result<DiagnosticReport, String> {
    diagnose_file_with_options(path, DiagnosticOptions::default())
}

/// Diagnose one regular audio file with explicit bounded-analysis options.
pub fn diagnose_file_with_options(
    path: impl AsRef<Path>,
    options: DiagnosticOptions,
) -> Result<DiagnosticReport, String> {
    options.validate()?;
    let analysis = load_analysis_pcm(path.as_ref(), options)?;
    diagnose_pcm(analysis)
}

/// Diagnose already-decoded audio.  Only the configured prefix is retained.
pub fn diagnose_audio(
    audio: &Audio,
    options: DiagnosticOptions,
) -> Result<DiagnosticReport, String> {
    options.validate()?;
    validate_audio(audio)?;
    let source_limit = frame_limit(audio.sample_rate, options.analysis_seconds)?;
    let source_frames = audio.frames().min(source_limit);
    let analysis_rate = audio.sample_rate.min(MAX_ANALYSIS_SAMPLE_RATE);
    let analysis_limit = frame_limit(analysis_rate, options.analysis_seconds)?;
    let scratch = analysis_temporary_bytes(audio.sample_rate, analysis_rate, analysis_limit)?;
    DecodeBudget::new(options.decode_limits).check_planar_capacities(
        &audio.channels,
        scratch,
        "diagnostic decoded-audio analysis",
    )?;
    let raw = raw_stats(&audio.channels, source_frames);
    let mixed = mix_to_mono(&audio.channels, source_frames)?;
    let mut samples = if analysis_rate == audio.sample_rate {
        mixed
    } else {
        resample(&mixed, audio.sample_rate, analysis_rate)?
    };
    samples.truncate(analysis_limit);
    diagnose_pcm(AnalysisPcm {
        samples,
        format: AudioFormat::Wav,
        codec: AudioCodec::Pcm,
        source_rate: audio.sample_rate,
        analysis_rate,
        channels: audio.channels(),
        total_frames: Some(audio.frames() as u64),
        source_analyzed_frames: source_frames,
        analysis_mode: "decoded-audio",
        raw,
    })
}

/// Produce a single-input no-reference assessment.
pub fn assess_file_with_options(
    path: impl AsRef<Path>,
    options: DiagnosticOptions,
) -> Result<AssessmentReport, String> {
    let candidate = diagnose_file_with_options(path, options)?;
    Ok(AssessmentReport {
        schema: ASSESSMENT_SCHEMA.into(),
        schema_version: ASSESSMENT_SCHEMA_VERSION,
        denoize_version: env!("CARGO_PKG_VERSION").into(),
        network_accessed: false,
        baseline: None,
        candidate,
        comparison: None,
        verdict: "single-input".into(),
        warnings: assessment_warnings(false),
    })
}

/// Compare two regular audio files with identical native no-reference metrics.
pub fn compare_files_with_options(
    baseline: impl AsRef<Path>,
    candidate: impl AsRef<Path>,
    options: DiagnosticOptions,
) -> Result<AssessmentReport, String> {
    let baseline = diagnose_file_with_options(baseline, options)?;
    let candidate = diagnose_file_with_options(candidate, options)?;
    let comparison = compare_reports(&baseline, &candidate);
    let verdict = assessment_verdict(&baseline, &candidate, &comparison).to_string();
    let mut warnings = assessment_warnings(true);
    if !comparison.presentation_preserved {
        warnings.push(
            "sample rate, channel count, or presentation duration changed; review alignment before accepting the quality delta"
                .into(),
        );
    }
    Ok(AssessmentReport {
        schema: ASSESSMENT_SCHEMA.into(),
        schema_version: ASSESSMENT_SCHEMA_VERSION,
        denoize_version: env!("CARGO_PKG_VERSION").into(),
        network_accessed: false,
        baseline: Some(baseline),
        candidate,
        comparison: Some(comparison),
        verdict,
        warnings,
    })
}

fn assessment_warnings(comparison: bool) -> Vec<String> {
    let mut warnings = vec![
        "native no-reference scores are triage proxies, not human MOS or release acceptance evidence"
            .into(),
        "semantic, phoneme, and speaker-identity fidelity are not assessed without an explicit reference model"
            .into(),
    ];
    if comparison {
        warnings.push(
            "a higher proxy score cannot by itself authorize generated or hallucinated replacement content"
                .into(),
        );
    }
    warnings
}

fn compare_reports(
    baseline: &DiagnosticReport,
    candidate: &DiagnosticReport,
) -> AssessmentComparison {
    let sample_rate_equal = baseline.input.sample_rate == candidate.input.sample_rate;
    let channel_count_equal = baseline.input.channels == candidate.input.channels;
    let total_frame_delta = match (baseline.input.total_frames, candidate.input.total_frames) {
        (Some(before), Some(after)) => {
            let before = i128::from(before);
            let after = i128::from(after);
            i64::try_from(after - before).ok()
        }
        _ => None,
    };
    let duration_delta_milliseconds =
        match (baseline.input.total_frames, candidate.input.total_frames) {
            (Some(before), Some(after)) => Some(
                (after as f64 / f64::from(candidate.input.sample_rate)
                    - before as f64 / f64::from(baseline.input.sample_rate))
                    * 1_000.0,
            ),
            _ => None,
        };
    let presentation_preserved = sample_rate_equal
        && channel_count_equal
        && duration_delta_milliseconds.is_some_and(|delta| delta.abs() <= 1.0);
    AssessmentComparison {
        quality_score_delta: candidate.quality.score - baseline.quality.score,
        estimated_mos_proxy_delta: candidate.quality.estimated_mos_proxy
            - baseline.quality.estimated_mos_proxy,
        noise_cleanliness_delta: candidate.quality.noise_cleanliness
            - baseline.quality.noise_cleanliness,
        distortion_freedom_delta: candidate.quality.distortion_freedom
            - baseline.quality.distortion_freedom,
        spectral_completeness_delta: candidate.quality.spectral_completeness
            - baseline.quality.spectral_completeness,
        continuity_delta: candidate.quality.continuity - baseline.quality.continuity,
        sample_rate_equal,
        channel_count_equal,
        total_frame_delta,
        duration_delta_milliseconds,
        presentation_preserved,
        semantic_fidelity_assessed: false,
    }
}

fn assessment_verdict(
    baseline: &DiagnosticReport,
    candidate: &DiagnosticReport,
    comparison: &AssessmentComparison,
) -> &'static str {
    if !comparison.presentation_preserved {
        return "incomparable";
    }
    let new_clipping =
        candidate.metrics.clipped_sample_ratio > baseline.metrics.clipped_sample_ratio + 1.0e-4;
    let new_dropouts = candidate.metrics.dropout_seconds > baseline.metrics.dropout_seconds + 0.01;
    if comparison.quality_score_delta >= 3.0 && !new_clipping && !new_dropouts {
        "improved"
    } else if comparison.quality_score_delta <= -3.0 || new_clipping || new_dropouts {
        "degraded"
    } else if comparison.quality_score_delta.abs() < 1.0 {
        "unchanged"
    } else {
        "mixed"
    }
}

fn load_analysis_pcm(path: &Path, options: DiagnosticOptions) -> Result<AnalysisPcm, String> {
    let mut session = AudioInputSession::open(path)?;
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
        return load_stream_analysis(session, options);
    }
    let audio = crate::read_audio_from_session_with_limits(&mut session, options.decode_limits)?;
    validate_audio(&audio)?;
    let source_limit = frame_limit(audio.sample_rate, options.analysis_seconds)?;
    let source_frames = audio.frames().min(source_limit);
    let analysis_rate = audio.sample_rate.min(MAX_ANALYSIS_SAMPLE_RATE);
    let analysis_limit = frame_limit(analysis_rate, options.analysis_seconds)?;
    let scratch = analysis_temporary_bytes(audio.sample_rate, analysis_rate, analysis_limit)?;
    DecodeBudget::new(options.decode_limits).check_planar_capacities(
        &audio.channels,
        scratch,
        "diagnostic whole-file analysis",
    )?;
    let raw = raw_stats(&audio.channels, source_frames);
    let mixed = mix_to_mono(&audio.channels, source_frames)?;
    let mut samples = if analysis_rate == audio.sample_rate {
        mixed
    } else {
        resample(&mixed, audio.sample_rate, analysis_rate)?
    };
    samples.truncate(analysis_limit);
    Ok(AnalysisPcm {
        samples,
        format: probe.format,
        codec: probe.codec,
        source_rate: audio.sample_rate,
        analysis_rate,
        channels: audio.channels(),
        total_frames: Some(audio.frames() as u64),
        source_analyzed_frames: source_frames,
        analysis_mode: "whole-file-fallback",
        raw,
    })
}

fn load_stream_analysis(
    session: AudioInputSession,
    options: DiagnosticOptions,
) -> Result<AnalysisPcm, String> {
    let mut reader = AudioStreamReader::from_session(session, options.decode_limits)?;
    let info = reader.info();
    let source_rate = info.sample_rate();
    let channels = info.channels();
    let analysis_rate = source_rate.min(MAX_ANALYSIS_SAMPLE_RATE);
    let source_limit = frame_limit(source_rate, options.analysis_seconds)?;
    let analysis_limit = frame_limit(analysis_rate, options.analysis_seconds)?;
    let block_frames = ANALYSIS_BLOCK_FRAMES.min(source_limit.max(1));
    let analysis_bytes = analysis_temporary_bytes(source_rate, analysis_rate, analysis_limit)?;
    let temporary_bytes = info
        .decoder_additional_bytes
        .checked_add(analysis_bytes)
        .ok_or_else(|| "diagnostic temporary byte count overflows".to_string())?;
    DecodeBudget::new(options.decode_limits).check_planar_frames(
        channels,
        block_frames,
        temporary_bytes,
        "diagnostic bounded analysis",
    )?;
    let mut converter = StreamingResampler::new(1, source_rate, analysis_rate)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(analysis_limit)
        .map_err(|_| "unable to reserve diagnostic analysis samples".to_string())?;
    let mut raw = RawPcmStats::default();
    let mut source_analyzed_frames = 0usize;
    while source_analyzed_frames < source_limit {
        let request = block_frames.min(source_limit - source_analyzed_frames);
        let Some(block) = reader.next_block(request)? else {
            break;
        };
        let frames = block.first().map_or(0, Vec::len).min(request);
        if frames == 0 {
            break;
        }
        let mixed = mix_to_mono(&block, frames)?;
        raw.ingest(&block, frames);
        let converted = converter.process(&[mixed])?;
        if let Some(channel) = converted.first() {
            extend_bounded(&mut samples, channel, analysis_limit);
        }
        source_analyzed_frames = source_analyzed_frames
            .checked_add(frames)
            .ok_or_else(|| "diagnostic source frame count overflows".to_string())?;
    }
    let tail = converter.finish()?;
    if let Some(channel) = tail.first() {
        extend_bounded(&mut samples, channel, analysis_limit);
    }
    Ok(AnalysisPcm {
        samples,
        format: info.format,
        codec: info.codec,
        source_rate,
        analysis_rate,
        channels,
        total_frames: info.total_frames,
        source_analyzed_frames,
        analysis_mode: "bounded-stream",
        raw,
    })
}

fn validate_audio(audio: &Audio) -> Result<(), String> {
    if audio.sample_rate == 0 {
        return Err("diagnostic input sample rate is zero".into());
    }
    if audio.channels.is_empty() {
        return Err("diagnostic input has no channels".into());
    }
    let frames = audio.frames();
    if frames == 0 {
        return Err("diagnostic input has no frames".into());
    }
    if audio.channels.iter().any(|channel| channel.len() != frames) {
        return Err("diagnostic input channels have inconsistent frame counts".into());
    }
    Ok(())
}

fn frame_limit(sample_rate: u32, seconds: u32) -> Result<usize, String> {
    u64::from(sample_rate)
        .checked_mul(u64::from(seconds))
        .and_then(|frames| usize::try_from(frames).ok())
        .ok_or_else(|| "diagnostic analysis frame limit overflows".to_string())
}

fn analysis_temporary_bytes(
    source_rate: u32,
    analysis_rate: u32,
    analysis_frames: usize,
) -> Result<u64, String> {
    let retained_samples = u64::try_from(analysis_frames)
        .ok()
        .and_then(|frames| frames.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| "diagnostic analysis byte count overflows".to_string())?;
    let curvature_samples = analysis_frames.saturating_sub(2) / 8 + 1;
    let curvature_bytes = u64::try_from(curvature_samples)
        .ok()
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| "diagnostic curvature byte count overflows".to_string())?;
    let resampler_bytes = resampler_plan_bytes(1, source_rate, analysis_rate)?;
    retained_samples
        .checked_add(curvature_bytes)
        .and_then(|bytes| bytes.checked_add(resampler_bytes))
        .and_then(|bytes| bytes.checked_add(ANALYSIS_FIXED_SCRATCH_BYTES))
        .ok_or_else(|| "diagnostic analysis working-set byte count overflows".to_string())
}

fn extend_bounded(destination: &mut Vec<f64>, source: &[f64], limit: usize) {
    let remaining = limit.saturating_sub(destination.len());
    destination.extend_from_slice(&source[..source.len().min(remaining)]);
}

fn raw_stats(channels: &[Vec<f64>], frames: usize) -> RawPcmStats {
    let mut stats = RawPcmStats::default();
    stats.ingest(channels, frames);
    stats
}

fn mix_to_mono(channels: &[Vec<f64>], frames: usize) -> Result<Vec<f64>, String> {
    if channels.is_empty() || channels.iter().any(|channel| channel.len() < frames) {
        return Err("diagnostic input block has inconsistent channels".into());
    }
    let mut mixed = Vec::new();
    mixed
        .try_reserve_exact(frames)
        .map_err(|_| "unable to reserve diagnostic mono samples".to_string())?;
    let scale = 1.0 / channels.len() as f64;
    for frame in 0..frames {
        let value = channels
            .iter()
            .map(|channel| sanitize_sample(channel[frame]))
            .sum::<f64>()
            * scale;
        mixed.push(value.clamp(-1.0, 1.0));
    }
    Ok(mixed)
}

fn diagnose_pcm(analysis: AnalysisPcm) -> Result<DiagnosticReport, String> {
    if analysis.samples.is_empty() {
        return Err("diagnostic input has no decodable frames in the analysis window".into());
    }
    let spectral = spectral_summary(&analysis.samples, analysis.analysis_rate)?;
    let sample_metrics = sample_metrics(&analysis.samples, analysis.analysis_rate);
    let clipped_ratio = ratio(analysis.raw.clipped, analysis.raw.samples);
    let flat_clip_ratio = ratio(analysis.raw.flat_clipped, analysis.raw.samples);
    let hum = if spectral.hum_60_db >= spectral.hum_50_db {
        (60.0, spectral.hum_60_db, spectral.hum_60_support)
    } else {
        (50.0, spectral.hum_50_db, spectral.hum_50_support)
    };
    let metrics = DiagnosticMetrics {
        rms_dbfs: amplitude_db(sample_metrics.rms),
        peak_dbfs: amplitude_db(sample_metrics.peak),
        dc_offset: sample_metrics.dc,
        clipped_sample_ratio: clipped_ratio,
        flat_clip_run_ratio: flat_clip_ratio,
        click_rate_per_second: sample_metrics.click_rate,
        dropout_seconds: sample_metrics.dropout_seconds,
        spectral_flatness: spectral.flatness,
        low_frequency_energy_ratio: spectral.low_ratio,
        high_frequency_energy_ratio: spectral.high_ratio,
        estimated_bandwidth_hz: spectral.bandwidth_hz,
        noise_floor_dbfs: spectral.noise_floor_dbfs,
        estimated_snr_db: spectral.estimated_snr_db,
        late_energy_ratio: spectral.late_energy_ratio,
        hum_frequency_hz: (hum.2 >= 2 && hum.1 >= 3.0).then_some(hum.0),
        hum_prominence_db: hum.1.max(0.0),
    };
    let findings = build_findings(&analysis, &metrics, hum.2);
    let rms_dbfs = metrics.rms_dbfs;
    let quality = build_quality(&findings, metrics.rms_dbfs);
    let mut recommended_pipeline = Vec::new();
    for action in findings
        .iter()
        .filter(|finding| finding.detected)
        .map(|finding| &finding.recommended_action)
        .filter(|action| action.as_str() != "none")
    {
        if !recommended_pipeline.contains(action) {
            recommended_pipeline.push(action.clone());
        }
    }
    if recommended_pipeline.is_empty() {
        recommended_pipeline.push("assess-only".into());
    }
    let mut hash = Sha256::new();
    hash.update(ANALYSIS_DOMAIN);
    hash.update(analysis.analysis_rate.to_le_bytes());
    hash.update((analysis.channels as u64).to_le_bytes());
    for sample in &analysis.samples {
        hash.update(sample.to_bits().to_le_bytes());
    }
    let analyzed_seconds = analysis.source_analyzed_frames as f64 / f64::from(analysis.source_rate);
    Ok(DiagnosticReport {
        schema: DIAGNOSTIC_SCHEMA.into(),
        schema_version: DIAGNOSTIC_SCHEMA_VERSION,
        denoize_version: env!("CARGO_PKG_VERSION").into(),
        network_accessed: false,
        input: DiagnosticInput {
            format: format_name(analysis.format).into(),
            codec: codec_name(analysis.codec).into(),
            sample_rate: analysis.source_rate,
            analysis_sample_rate: analysis.analysis_rate,
            channels: analysis.channels,
            total_frames: analysis.total_frames,
            source_analyzed_frames: analysis.source_analyzed_frames,
            analyzed_frames: analysis.samples.len(),
            analyzed_seconds,
            analysis_mode: analysis.analysis_mode.into(),
            analysis_sha256: format!("{:x}", hash.finalize()),
        },
        quality,
        metrics,
        findings,
        recommended_pipeline,
        limitations: diagnostic_limitations(rms_dbfs),
    })
}

fn diagnostic_limitations(rms_dbfs: f64) -> Vec<String> {
    let mut limitations = vec![
        "native heuristic estimates are content-dependent and require corpus calibration".into(),
        "phoneme, word, speaker-identity, and generative hallucination fidelity are not measured"
            .into(),
        "human listening and reference metrics remain mandatory release gates".into(),
    ];
    if rms_dbfs <= -70.0 {
        limitations.push(
            "the analyzed signal is below -70 dBFS; degradation classification is indeterminate"
                .into(),
        );
    }
    limitations
}

struct SampleMetrics {
    rms: f64,
    peak: f64,
    dc: f64,
    click_rate: f64,
    dropout_seconds: f64,
}

fn sample_metrics(samples: &[f64], sample_rate: u32) -> SampleMetrics {
    let count = samples.len().max(1) as f64;
    let sum = samples.iter().sum::<f64>();
    let sum_squares = samples.iter().map(|sample| sample * sample).sum::<f64>();
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f64::max);
    let mut curvature = Vec::with_capacity(samples.len().saturating_sub(2) / 8 + 1);
    for window in samples.windows(3).step_by(8) {
        curvature.push((window[2] - 2.0 * window[1] + window[0]).abs());
    }
    let median_curvature = percentile(&mut curvature, 0.5);
    let threshold = (median_curvature * 24.0).max(0.10);
    let mut clicks = 0usize;
    let mut index = 2usize;
    while index + 2 < samples.len() {
        let left = samples[index] - samples[index - 1];
        let right = samples[index + 1] - samples[index];
        let curvature = (right - left).abs();
        if curvature >= threshold
            && left.signum() != right.signum()
            && samples[index].abs() >= samples[index - 1].abs().max(samples[index + 1].abs())
        {
            clicks += 1;
            index += 4;
        } else {
            index += 1;
        }
    }
    let seconds = samples.len() as f64 / f64::from(sample_rate);
    let dropout_seconds = detect_dropouts(samples, sample_rate);
    SampleMetrics {
        rms: (sum_squares / count).sqrt(),
        peak,
        dc: sum / count,
        click_rate: if seconds > 0.0 {
            clicks as f64 / seconds
        } else {
            0.0
        },
        dropout_seconds,
    }
}

fn detect_dropouts(samples: &[f64], sample_rate: u32) -> f64 {
    let frame = (sample_rate as usize / 100).max(1);
    if samples.len() < frame * 5 {
        return 0.0;
    }
    let levels: Vec<f64> = samples
        .chunks(frame)
        .map(|chunk| {
            let energy =
                chunk.iter().map(|sample| sample * sample).sum::<f64>() / chunk.len().max(1) as f64;
            amplitude_db(energy.sqrt())
        })
        .collect();
    let mut dropout_frames = 0usize;
    let mut index = 1usize;
    while index + 1 < levels.len() {
        if levels[index] > -72.0 {
            index += 1;
            continue;
        }
        let start = index;
        while index < levels.len() && levels[index] <= -72.0 {
            index += 1;
        }
        let run = index - start;
        let before = levels[start.saturating_sub(5)..start]
            .iter()
            .copied()
            .fold(-120.0, f64::max);
        let after_end = (index + 5).min(levels.len());
        let after = levels[index..after_end]
            .iter()
            .copied()
            .fold(-120.0, f64::max);
        if (1..=25).contains(&run) && before >= -42.0 && after >= -42.0 {
            dropout_frames += run;
        }
    }
    dropout_frames as f64 * frame as f64 / f64::from(sample_rate)
}

fn spectral_summary(samples: &[f64], sample_rate: u32) -> Result<SpectralSummary, String> {
    let frame_size = if sample_rate >= 32_000 {
        2_048
    } else if sample_rate >= 16_000 {
        1_024
    } else {
        512
    };
    let hop = frame_size / 2;
    let fft = Fft::new(frame_size);
    let bins = fft.nbins();
    let mut average_power = vec![0.0; bins];
    let mut flatness_sum = 0.0;
    let mut low_energy = 0.0;
    let mut high_energy = 0.0;
    let mut total_energy = 0.0;
    let mut frame_db = Vec::new();
    let mut spectrum = vec![Complex::default(); frame_size];
    let mut frames = 0usize;
    let mut start = 0usize;
    while start < samples.len() {
        let available = (samples.len() - start).min(frame_size);
        if available < frame_size / 4 && frames > 0 {
            break;
        }
        let mut time_energy = 0.0;
        for index in 0..frame_size {
            let value = if index < available {
                samples[start + index]
            } else {
                0.0
            };
            let window =
                0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / frame_size as f64).cos();
            spectrum[index] = Complex::new(value * window, 0.0);
            if index < available {
                time_energy += value * value;
            }
        }
        frame_db.push(amplitude_db((time_energy / available.max(1) as f64).sqrt()));
        fft.forward(&mut spectrum);
        let mut arithmetic = 0.0;
        let mut logarithmic = 0.0;
        for bin in 0..bins {
            let power = spectrum[bin].re * spectrum[bin].re + spectrum[bin].im * spectrum[bin].im;
            average_power[bin] += power;
            arithmetic += power;
            logarithmic += (power + EPSILON).ln();
            let frequency = bin as f64 * f64::from(sample_rate) / frame_size as f64;
            total_energy += power;
            if frequency <= 200.0 {
                low_energy += power;
            }
            let high_boundary = (8_000.0_f64).min(f64::from(sample_rate) * 0.35);
            if frequency >= high_boundary {
                high_energy += power;
            }
        }
        let arithmetic_mean = arithmetic / bins as f64;
        let geometric_mean = (logarithmic / bins as f64).exp();
        flatness_sum += (geometric_mean / (arithmetic_mean + EPSILON)).clamp(0.0, 1.0);
        frames += 1;
        if start + hop <= start {
            return Err("diagnostic spectral frame offset overflows".into());
        }
        start += hop;
    }
    if frames == 0 {
        return Err("diagnostic input is too short for spectral analysis".into());
    }
    for power in &mut average_power {
        *power /= frames as f64;
    }
    let spectrum_total = average_power.iter().sum::<f64>();
    let target = spectrum_total * 0.995;
    let mut cumulative = 0.0;
    let mut occupied_bin = 0usize;
    for (bin, power) in average_power.iter().enumerate() {
        cumulative += *power;
        if cumulative <= target {
            occupied_bin = bin;
        }
    }
    let bandwidth_hz = occupied_bin as f64 * f64::from(sample_rate) / frame_size as f64;
    let (hum_50_db, hum_50_support) = harmonic_prominence(&average_power, sample_rate, 50.0);
    let (hum_60_db, hum_60_support) = harmonic_prominence(&average_power, sample_rate, 60.0);
    let mut sorted_levels = frame_db.clone();
    let noise_floor_dbfs = percentile(&mut sorted_levels, 0.10);
    let active_dbfs = percentile(&mut sorted_levels, 0.75);
    let estimated_snr_db = (active_dbfs - noise_floor_dbfs).clamp(0.0, 60.0);
    let late_energy_ratio = estimate_late_energy_ratio(&frame_db);
    Ok(SpectralSummary {
        flatness: (flatness_sum / frames as f64).clamp(0.0, 1.0),
        low_ratio: (low_energy / (total_energy + EPSILON)).clamp(0.0, 1.0),
        high_ratio: (high_energy / (total_energy + EPSILON)).clamp(0.0, 1.0),
        bandwidth_hz,
        noise_floor_dbfs,
        estimated_snr_db,
        hum_50_db,
        hum_60_db,
        hum_50_support,
        hum_60_support,
        late_energy_ratio,
    })
}

fn harmonic_prominence(power: &[f64], sample_rate: u32, fundamental: f64) -> (f64, usize) {
    if power.len() < 16 {
        return (0.0, 0);
    }
    let fft_size = (power.len() - 1) * 2;
    let nyquist = f64::from(sample_rate) / 2.0;
    let harmonics = ((nyquist / fundamental) as usize).min(12);
    let mut prominences = Vec::new();
    for harmonic in 1..=harmonics {
        let frequency = fundamental * harmonic as f64;
        if frequency > 1_200.0 {
            break;
        }
        let bin = (frequency * fft_size as f64 / f64::from(sample_rate)).round() as usize;
        if bin < 5 || bin + 5 >= power.len() {
            continue;
        }
        let signal = power[bin - 1..=bin + 1].iter().sum::<f64>() / 3.0;
        let local = power[bin - 5..bin - 2]
            .iter()
            .chain(power[bin + 3..=bin + 5].iter())
            .sum::<f64>()
            / 6.0;
        let prominence = 10.0 * ((signal + EPSILON) / (local + EPSILON)).log10();
        if prominence >= 3.0 {
            prominences.push(prominence.min(40.0));
        }
    }
    let support = prominences.len();
    let mean = if support == 0 {
        0.0
    } else {
        prominences.iter().sum::<f64>() / support as f64
    };
    (mean, support)
}

fn estimate_late_energy_ratio(frame_db: &[f64]) -> f64 {
    if frame_db.len() < 24 {
        return 0.0;
    }
    let mut ratios = Vec::new();
    for index in 1..frame_db.len().saturating_sub(20) {
        if frame_db[index] >= frame_db[index - 1] + 6.0 && frame_db[index] >= -45.0 {
            let onset = 10.0_f64.powf(frame_db[index] / 10.0);
            let late = frame_db[index + 5..index + 20]
                .iter()
                .map(|value| 10.0_f64.powf(*value / 10.0))
                .sum::<f64>()
                / 15.0;
            ratios.push((late / (onset + EPSILON)).clamp(0.0, 1.0));
        }
    }
    percentile(&mut ratios, 0.5)
}

fn build_findings(
    analysis: &AnalysisPcm,
    metrics: &DiagnosticMetrics,
    hum_support: usize,
) -> Vec<DiagnosticFinding> {
    let mut findings = Vec::new();
    let dynamic_noise = (1.0 - metrics.estimated_snr_db / 30.0).clamp(0.0, 1.0);
    let noise_severity = (metrics.spectral_flatness * 0.8 + dynamic_noise * 0.2).clamp(0.0, 1.0);
    findings.push(finding(
        "additive-noise",
        noise_severity,
        (0.45 + metrics.spectral_flatness * 0.45).clamp(0.0, 0.95),
        format!(
            "spectral flatness {:.3}; frame-level SNR proxy {:.1} dB; noise floor {:.1} dBFS",
            metrics.spectral_flatness, metrics.estimated_snr_db, metrics.noise_floor_dbfs
        ),
        "enhance",
    ));

    let clipping_severity = (metrics.clipped_sample_ratio * 500.0
        + metrics.flat_clip_run_ratio * 1_000.0)
        .clamp(0.0, 1.0);
    findings.push(finding(
        "clipping",
        clipping_severity,
        if metrics.flat_clip_run_ratio > 0.0 {
            0.98
        } else {
            0.75
        },
        format!(
            "{:.5}% samples at full scale; {:.5}% repeated flat-top samples",
            metrics.clipped_sample_ratio * 100.0,
            metrics.flat_clip_run_ratio * 100.0
        ),
        "declip",
    ));

    let hum_severity = ((metrics.hum_prominence_db - 4.0) / 18.0).clamp(0.0, 1.0)
        * (hum_support as f64 / 4.0).clamp(0.25, 1.0);
    findings.push(finding(
        "hum",
        hum_severity,
        (0.35 + hum_support as f64 * 0.14).clamp(0.0, 0.95),
        format!(
            "{} Hz family has {:.1} dB mean prominence across {hum_support} harmonics",
            metrics.hum_frequency_hz.unwrap_or(0.0),
            metrics.hum_prominence_db
        ),
        "dehum",
    ));

    let click_severity = (metrics.click_rate_per_second / 8.0).clamp(0.0, 1.0);
    findings.push(finding(
        "clicks",
        click_severity,
        (0.55 + click_severity * 0.35).clamp(0.0, 0.95),
        format!(
            "robust isolated-curvature detector found {:.2} events per second",
            metrics.click_rate_per_second
        ),
        "declick",
    ));

    let reverb_severity = ((metrics.late_energy_ratio - 0.08) / 0.45).clamp(0.0, 1.0);
    let onset_count = spectral_onset_count_hint(metrics.late_energy_ratio);
    findings.push(finding(
        "reverberation",
        reverb_severity,
        onset_count,
        format!(
            "median 50–200 ms post-onset energy ratio {:.3}",
            metrics.late_energy_ratio
        ),
        "dereverb",
    ));

    let nyquist = f64::from(analysis.analysis_rate) / 2.0;
    let relative_bandwidth = (metrics.estimated_bandwidth_hz / nyquist.max(1.0)).clamp(0.0, 1.0);
    let bandwidth_severity = ((0.78 - relative_bandwidth) / 0.58).clamp(0.0, 1.0);
    let bandwidth_confidence = (0.30 + metrics.spectral_flatness * 0.45).clamp(0.0, 0.85);
    findings.push(finding(
        "bandwidth-limitation",
        bandwidth_severity,
        bandwidth_confidence,
        format!(
            "99.5% occupied bandwidth {:.0} Hz ({:.1}% of Nyquist); high-band energy {:.4}",
            metrics.estimated_bandwidth_hz,
            relative_bandwidth * 100.0,
            metrics.high_frequency_energy_ratio
        ),
        "universal-restore",
    ));

    let dropout_severity = (metrics.dropout_seconds / 0.20).clamp(0.0, 1.0);
    findings.push(finding(
        "packet-loss-or-dropout",
        dropout_severity,
        if dropout_severity > 0.0 { 0.86 } else { 0.55 },
        format!(
            "{:.3} seconds of short interior near-silence gaps surrounded by active audio",
            metrics.dropout_seconds
        ),
        "universal-restore",
    ));

    let wind_severity = ((metrics.low_frequency_energy_ratio - 0.42) / 0.45).clamp(0.0, 1.0)
        * (metrics.spectral_flatness * 1.6).clamp(0.0, 1.0);
    findings.push(finding(
        "wind-or-plosive",
        wind_severity,
        (0.35 + metrics.spectral_flatness * 0.35).clamp(0.0, 0.80),
        format!(
            "sub-200 Hz energy ratio {:.3} with spectral flatness {:.3}",
            metrics.low_frequency_energy_ratio, metrics.spectral_flatness
        ),
        "dewind",
    ));

    let lossy = matches!(
        analysis.codec,
        AudioCodec::Opus | AudioCodec::Vorbis | AudioCodec::Mp3 | AudioCodec::Aac
    );
    let codec_severity = if lossy {
        (0.08 + bandwidth_severity * 0.55).clamp(0.0, 0.70)
    } else {
        0.0
    };
    findings.push(finding(
        "codec-distortion",
        codec_severity,
        if lossy && bandwidth_severity >= 0.25 {
            0.62
        } else if lossy {
            0.35
        } else {
            0.20
        },
        format!(
            "input codec {}{}",
            codec_name(analysis.codec),
            if lossy {
                "; lossy transport alone is risk evidence, not proof of audible damage"
            } else {
                "; no lossy-container evidence"
            }
        ),
        "universal-restore",
    ));

    if metrics.rms_dbfs <= -70.0 {
        for finding in &mut findings {
            finding.detected = false;
            finding.confidence = finding.confidence.min(0.20);
            finding.severity = 0.0;
            finding.evidence.push_str("; insufficient active signal");
        }
    }

    findings
}

fn spectral_onset_count_hint(late_energy_ratio: f64) -> f64 {
    if late_energy_ratio > 0.0 {
        0.62
    } else {
        0.25
    }
}

fn finding(
    kind: &str,
    severity: f64,
    confidence: f64,
    evidence: String,
    recommended_action: &str,
) -> DiagnosticFinding {
    let severity = finite_unit(severity);
    let confidence = finite_unit(confidence);
    DiagnosticFinding {
        kind: kind.into(),
        detected: severity >= 0.12 && confidence >= 0.55,
        confidence,
        severity,
        evidence,
        recommended_action: recommended_action.into(),
    }
}

fn build_quality(findings: &[DiagnosticFinding], rms_dbfs: f64) -> NoReferenceQuality {
    if rms_dbfs <= -70.0 {
        return NoReferenceQuality {
            method: "denoize-native-no-reference-v1".into(),
            score: 50.0,
            estimated_mos_proxy: 3.0,
            uncertainty: 0.50,
            noise_cleanliness: 50.0,
            distortion_freedom: 50.0,
            spectral_completeness: 50.0,
            continuity: 50.0,
        };
    }
    let severity = |kind: &str| {
        findings
            .iter()
            .find(|finding| finding.kind == kind)
            .map_or(0.0, |finding| finding.severity * finding.confidence)
    };
    let noise = severity("additive-noise");
    let clipping = severity("clipping");
    let hum = severity("hum");
    let clicks = severity("clicks");
    let reverb = severity("reverberation");
    let bandwidth = severity("bandwidth-limitation");
    let dropout = severity("packet-loss-or-dropout");
    let wind = severity("wind-or-plosive");
    let codec = severity("codec-distortion");
    let noise_cleanliness = 100.0 * (1.0 - (noise * 0.70 + hum * 0.18 + wind * 0.12));
    let distortion_freedom =
        100.0 * (1.0 - (clipping * 0.45 + clicks * 0.25 + reverb * 0.20 + codec * 0.10));
    let spectral_completeness = 100.0 * (1.0 - (bandwidth * 0.70 + codec * 0.20 + wind * 0.10));
    let continuity = 100.0 * (1.0 - (dropout * 0.75 + clicks * 0.15 + clipping * 0.10));
    let penalty = noise * 22.0
        + clipping * 22.0
        + hum * 12.0
        + clicks * 12.0
        + reverb * 12.0
        + bandwidth * 10.0
        + dropout * 18.0
        + wind * 8.0
        + codec * 5.0;
    let score = (100.0 - penalty).clamp(0.0, 100.0);
    let estimated_mos_proxy = 1.0 + 4.0 * (score / 100.0).powf(0.72);
    let ambiguity = findings
        .iter()
        .filter(|finding| (0.10..0.65).contains(&finding.severity))
        .map(|finding| 1.0 - finding.confidence)
        .sum::<f64>()
        / findings.len().max(1) as f64;
    NoReferenceQuality {
        method: "denoize-native-no-reference-v1".into(),
        score: finite_score(score),
        estimated_mos_proxy: estimated_mos_proxy.clamp(1.0, 5.0),
        uncertainty: (0.12 + ambiguity).clamp(0.12, 0.50),
        noise_cleanliness: finite_score(noise_cleanliness),
        distortion_freedom: finite_score(distortion_freedom),
        spectral_completeness: finite_score(spectral_completeness),
        continuity: finite_score(continuity),
    }
}

fn format_name(format: AudioFormat) -> &'static str {
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

fn codec_name(codec: AudioCodec) -> &'static str {
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

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn amplitude_db(value: f64) -> f64 {
    (20.0 * value.max(1.0e-6).log10()).clamp(-120.0, 0.0)
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let index = ((values.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn finite_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn finite_score(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 100.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn audio(samples: Vec<f64>, sample_rate: u32) -> Audio {
        Audio {
            sample_rate,
            channels: vec![samples],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    #[test]
    fn detects_clipping_and_hum_with_stable_contract() {
        let rate = 48_000;
        let mut samples = Vec::new();
        for index in 0..rate * 2 {
            let time = index as f64 / f64::from(rate);
            let value = 0.85 * (std::f64::consts::TAU * 220.0 * time).sin()
                + 0.25 * (std::f64::consts::TAU * 60.0 * time).sin()
                + 0.12 * (std::f64::consts::TAU * 120.0 * time).sin()
                + 0.08 * (std::f64::consts::TAU * 180.0 * time).sin();
            samples.push(value.clamp(-1.0, 1.0));
        }
        let report = diagnose_audio(
            &audio(samples, rate),
            DiagnosticOptions::new().with_analysis_seconds(2),
        )
        .unwrap();
        assert_eq!(report.schema, DIAGNOSTIC_SCHEMA);
        assert!(!report.network_accessed);
        assert_eq!(report.input.analysis_sha256.len(), 64);
        assert!(report.metrics.clipped_sample_ratio > 0.001);
        assert!(report
            .findings
            .iter()
            .any(|finding| { finding.kind == "clipping" && finding.detected }));
        assert!(report
            .findings
            .iter()
            .any(|finding| { finding.kind == "hum" && finding.confidence >= 0.55 }));
        assert!(report
            .recommended_pipeline
            .iter()
            .any(|action| action == "declip"));
    }

    #[test]
    fn no_reference_comparison_rejects_geometry_changes() {
        let rate = 16_000;
        let samples: Vec<f64> = (0..rate)
            .map(|index| (std::f64::consts::TAU * 440.0 * index as f64 / rate as f64).sin() * 0.2)
            .collect();
        let before =
            diagnose_audio(&audio(samples.clone(), rate), DiagnosticOptions::default()).unwrap();
        let after =
            diagnose_audio(&audio(samples, rate * 2), DiagnosticOptions::default()).unwrap();
        let comparison = compare_reports(&before, &after);
        assert!(!comparison.sample_rate_equal);
        assert!(!comparison.presentation_preserved);
        assert_eq!(
            assessment_verdict(&before, &after, &comparison),
            "incomparable"
        );
    }

    #[test]
    fn diagnostic_option_bounds_precede_analysis() {
        let error = diagnose_audio(
            &audio(vec![0.0; 100], 16_000),
            DiagnosticOptions::new().with_analysis_seconds(0),
        )
        .unwrap_err();
        assert!(error.contains("between 1 and 60"));
    }

    #[test]
    fn silence_is_indeterminate_without_restoration_actions() {
        let report = diagnose_audio(
            &audio(vec![0.0; 48_000], 48_000),
            DiagnosticOptions::default(),
        )
        .unwrap();

        assert_eq!(report.quality.score, 50.0);
        assert_eq!(report.quality.uncertainty, 0.50);
        assert!(report.findings.iter().all(|finding| !finding.detected));
        assert_eq!(report.recommended_pipeline, ["assess-only"]);
        assert!(report
            .limitations
            .iter()
            .any(|limitation| limitation.contains("below -70 dBFS")));
    }
}
