//! EBU R128 / ITU-R BS.1770 loudness measurement and normalization.
//!
//! [`measure`] keeps the compact integrated-LUFS/true-peak interface used by
//! the normalization pipeline. [`measure_detailed`] exposes the standard
//! momentary, short-term, integrated, and loudness-range measurements as well
//! as the relative gate threshold and sample/true peaks.

use crate::{
    channel_layout::{ChannelLayout, ChannelMask, ChannelPosition},
    config::{checked_resource_add, checked_resource_multiply, ConfigError},
    sanitize_sample, Audio,
};
use ebur128::{Channel as EbuChannel, EbuR128, Mode};

#[derive(Clone, Copy, Debug)]
pub struct LoudnessReport {
    pub input_lufs: f64,
    pub output_lufs: f64,
    pub true_peak_dbtp: f64,
    pub gain_db: f64,
}

/// Gain calculated by the bounded first pass of stream normalization.
#[derive(Clone, Copy, Debug)]
pub struct StreamingLoudnessGain {
    report: LoudnessReport,
    linear: f64,
}

impl StreamingLoudnessGain {
    #[must_use]
    pub const fn report(self) -> LoudnessReport {
        self.report
    }

    #[must_use]
    pub const fn linear(self) -> f64 {
        self.linear
    }

    /// Apply the measured constant gain without changing block geometry.
    pub fn apply(self, channels: &mut [Vec<f64>]) {
        for sample in channels.iter_mut().flatten() {
            *sample = sanitize_sample(*sample * self.linear);
        }
    }
}

/// Fixed-memory EBU R128 analyzer for the first pass of a long stream.
///
/// Histogram mode keeps 1,000 loudness bins instead of retaining one energy
/// value per programme block. True-peak and K-weighting state are bounded by
/// channel count and sample rate; `add_block` reuses one caller-sized scratch
/// allocation.
pub struct StreamingLoudnessAnalyzer {
    analyzer: EbuR128,
    channels: usize,
    scratch: Vec<f64>,
    frames: u64,
}

impl StreamingLoudnessAnalyzer {
    pub fn new(
        channels: usize,
        sample_rate: u32,
        channel_mask: Option<ChannelMask>,
    ) -> Result<Self, String> {
        if channels == 0 {
            return Err("streaming loudness requires at least one channel".into());
        }
        let mut analyzer = EbuR128::new(
            channels as u32,
            sample_rate,
            Mode::I | Mode::TRUE_PEAK | Mode::HISTOGRAM,
        )
        .map_err(|error| format!("initialize streaming loudness analyzer: {error}"))?;
        if let Some(channel_map) = ebur_channel_map_for(channels, channel_mask) {
            analyzer
                .set_channel_map(&channel_map)
                .map_err(|error| format!("configure streaming loudness channel map: {error}"))?;
        }
        Ok(Self {
            analyzer,
            channels,
            scratch: Vec::new(),
            frames: 0,
        })
    }

    pub fn add_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if channels.len() != self.channels {
            return Err(format!(
                "streaming loudness expected {} channels, got {}",
                self.channels,
                channels.len()
            ));
        }
        let frames = channels.first().map(Vec::len).unwrap_or(0);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("streaming loudness blocks must have equal channel lengths".into());
        }
        let samples = frames
            .checked_mul(self.channels)
            .ok_or_else(|| "streaming loudness block size overflows".to_string())?;
        self.scratch.clear();
        self.scratch.try_reserve_exact(samples).map_err(|_| {
            ConfigError::allocation_failed("streaming loudness scratch").to_string()
        })?;
        for frame in 0..frames {
            for channel in channels {
                self.scratch.push(sanitize_sample(channel[frame]));
            }
        }
        self.analyzer
            .add_frames_f64(&self.scratch)
            .map_err(|error| format!("analyze streaming loudness: {error}"))?;
        self.frames = self
            .frames
            .checked_add(frames as u64)
            .ok_or_else(|| "streaming loudness frame count overflows".to_string())?;
        Ok(())
    }

    pub fn finish(
        self,
        target_lufs: f64,
        peak_limit_dbtp: f64,
    ) -> Result<StreamingLoudnessGain, String> {
        validate_normalization_targets(target_lufs, peak_limit_dbtp)?;
        if self.frames == 0 {
            return Err("cannot measure empty streaming audio".into());
        }
        let input_lufs = integrated_loudness(&self.analyzer)?;
        let input_peak = true_peak_dbtp(&self.analyzer, self.channels)?;
        let loudness_gain = target_lufs - input_lufs;
        let peak_gain = peak_limit_dbtp - input_peak;
        let gain_db = loudness_gain.min(peak_gain);
        let linear = 10f64.powf(gain_db / 20.0);
        Ok(StreamingLoudnessGain {
            report: LoudnessReport {
                input_lufs,
                output_lufs: input_lufs + gain_db,
                true_peak_dbtp: input_peak + gain_db,
                gain_db,
            },
            linear,
        })
    }
}

/// Conservative denoize-owned and analyzer state for a bounded loudness pass.
pub fn estimate_streaming_loudness_bytes(
    channels: usize,
    sample_rate: u32,
    block_frames: usize,
) -> Result<u64, ConfigError> {
    if channels == 0 {
        return Err(ConfigError::invalid("channels", "a positive channel count"));
    }
    let scratch_samples = checked_resource_multiply(
        "streaming loudness scratch",
        channels as u64,
        block_frames as u64,
    )?;
    let scratch_bytes = checked_resource_multiply(
        "streaming loudness scratch",
        scratch_samples,
        std::mem::size_of::<f64>() as u64,
    )?;
    // ebur128 retains a 400 ms channel ring, K-weighting/true-peak state, and
    // two fixed 1,000-bin histograms. One second per channel is conservative.
    let analyzer_samples = checked_resource_multiply(
        "streaming loudness analyzer",
        channels as u64,
        sample_rate as u64,
    )?;
    let analyzer_bytes = checked_resource_multiply(
        "streaming loudness analyzer",
        analyzer_samples,
        std::mem::size_of::<f64>() as u64,
    )?;
    checked_resource_add(
        "streaming loudness analyzer",
        checked_resource_add("streaming loudness analyzer", scratch_bytes, analyzer_bytes)?,
        2 * 1_000 * std::mem::size_of::<u64>() as u64,
    )
}

/// Loudness values calculated according to EBU R128 / ITU-R BS.1770.
///
/// Momentary loudness requires at least a 400 ms programme and short-term
/// loudness and loudness range require at least 3 seconds. Those fields are
/// therefore optional for short inputs instead of returning a value based on
/// the analyzer's zero-filled warm-up buffer. The integrated loudness remains
/// the required measurement and is rejected when the signal has no gated
/// blocks (for example, silence or an input shorter than the first 400 ms
/// block).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoudnessMetrics {
    /// Gated programme loudness in LUFS.
    pub integrated_lufs: f64,
    /// Loudness over the most recent 400 ms, in LUFS.
    pub momentary_lufs: Option<f64>,
    /// Loudness over the most recent 3 seconds, in LUFS.
    pub short_term_lufs: Option<f64>,
    /// EBU Tech 3342 loudness range in LU.
    pub loudness_range_lu: Option<f64>,
    /// Relative gate threshold used by the integrated measurement, in LUFS.
    pub relative_threshold_lufs: Option<f64>,
    /// Maximum sample peak across channels, in dBFS.
    pub sample_peak_dbfs: f64,
    /// Maximum 4x oversampled true peak across channels, in dBTP.
    pub true_peak_dbtp: f64,
}

/// Apply a constant gain toward `target_lufs`, constrained by `peak_limit_dbtp`.
pub fn normalize(
    audio: &mut Audio,
    target_lufs: f64,
    peak_limit_dbtp: f64,
) -> Result<LoudnessReport, String> {
    validate_normalization_targets(target_lufs, peak_limit_dbtp)?;
    let (input_lufs, input_peak) = measure(audio)?;
    let loudness_gain = target_lufs - input_lufs;
    let peak_gain = peak_limit_dbtp - input_peak;
    let gain_db = loudness_gain.min(peak_gain);
    let gain = 10f64.powf(gain_db / 20.0);
    for channel in &mut audio.channels {
        for sample in channel {
            *sample = sanitize_sample(*sample * gain);
        }
    }
    let (output_lufs, true_peak_dbtp) = measure(audio)?;
    Ok(LoudnessReport {
        input_lufs,
        output_lufs,
        true_peak_dbtp,
        gain_db,
    })
}

fn validate_normalization_targets(target_lufs: f64, peak_limit_dbtp: f64) -> Result<(), String> {
    if !target_lufs.is_finite() || !(-70.0..=0.0).contains(&target_lufs) {
        return Err("loudness target must be between -70 and 0 LUFS".into());
    }
    if !peak_limit_dbtp.is_finite() || !(-20.0..=0.0).contains(&peak_limit_dbtp) {
        return Err("true-peak limit must be between -20 and 0 dBTP".into());
    }
    Ok(())
}

pub fn measure(audio: &Audio) -> Result<(f64, f64), String> {
    let analyzer = create_analyzer(audio, Mode::I | Mode::TRUE_PEAK)?;
    let loudness = integrated_loudness(&analyzer)?;
    Ok((loudness, true_peak_dbtp(&analyzer, audio.channels())?))
}

/// Measure the complete set of EBU R128 / ITU-R BS.1770 programme metrics.
///
/// The analyzer uses K-weighting, the absolute and relative gates for
/// integrated loudness, the 400 ms and 3 s windows for momentary and
/// short-term loudness, EBU Tech 3342 for LRA, and 4x true-peak scanning.
/// Short inputs expose unavailable windowed metrics as `None`; the integrated
/// measurement still follows the existing error behavior when no gated block
/// can be calculated.
pub fn measure_detailed(audio: &Audio) -> Result<LoudnessMetrics, String> {
    let analyzer = create_analyzer(
        audio,
        Mode::I | Mode::S | Mode::LRA | Mode::SAMPLE_PEAK | Mode::TRUE_PEAK,
    )?;
    let integrated_lufs = integrated_loudness(&analyzer)?;
    let momentary_lufs = if has_duration(audio, 400) {
        finite_metric(analyzer.loudness_momentary().ok())
    } else {
        None
    };
    let short_term_lufs = if has_duration(audio, 3_000) {
        finite_metric(analyzer.loudness_shortterm().ok())
    } else {
        None
    };
    let loudness_range_lu = if has_duration(audio, 3_000) {
        finite_metric(analyzer.loudness_range().ok())
    } else {
        None
    };
    let relative_threshold_lufs = finite_metric(analyzer.relative_threshold().ok());

    let mut sample_peak = 0.0f64;
    for channel in 0..audio.channels() {
        sample_peak = sample_peak.max(
            analyzer
                .sample_peak(channel as u32)
                .map_err(|error| format!("measure sample peak: {error}"))?,
        );
    }

    Ok(LoudnessMetrics {
        integrated_lufs,
        momentary_lufs,
        short_term_lufs,
        loudness_range_lu,
        relative_threshold_lufs,
        sample_peak_dbfs: amplitude_to_db(sample_peak),
        true_peak_dbtp: true_peak_dbtp(&analyzer, audio.channels())?,
    })
}

fn create_analyzer(audio: &Audio, mode: Mode) -> Result<EbuR128, String> {
    let channels = audio.channels();
    if channels == 0 || audio.frames() == 0 {
        return Err("cannot measure empty audio".into());
    }
    let mut analyzer = EbuR128::new(channels as u32, audio.sample_rate, mode)
        .map_err(|error| format!("initialize loudness analyzer: {error}"))?;
    if let Some(channel_map) = ebur_channel_map(audio) {
        analyzer
            .set_channel_map(&channel_map)
            .map_err(|error| format!("configure loudness channel map: {error}"))?;
    }
    let mut interleaved = Vec::with_capacity(audio.frames() * channels);
    for frame in 0..audio.frames() {
        for channel in &audio.channels {
            interleaved.push(sanitize_sample(channel.get(frame).copied().unwrap_or(0.0)));
        }
    }
    analyzer
        .add_frames_f64(&interleaved)
        .map_err(|error| format!("analyze loudness: {error}"))?;
    Ok(analyzer)
}

/// Translate the WAVE speaker mask into the channel positions understood by
/// `ebur128`. In particular, LFE is explicitly marked `Unused`; treating a
/// 2.1/5.1/7.1 LFE channel as a center channel would incorrectly raise LUFS.
/// When an input has no usable mask, a known layout inferred from its channel
/// count supplies the conventional positions. Unknown layouts retain the
/// analyzer's conservative defaults.
fn ebur_channel_map(audio: &Audio) -> Option<Vec<EbuChannel>> {
    let channels = audio.channels();
    ebur_channel_map_for(channels, audio.effective_channel_mask())
}

fn ebur_channel_map_for(
    channels: usize,
    channel_mask: Option<ChannelMask>,
) -> Option<Vec<EbuChannel>> {
    let positions = channel_mask
        .filter(|mask| mask.bits() != 0 && mask.channels() == channels)
        .map(|mask| mask.positions())
        .or_else(|| {
            ChannelLayout::from_channel_count(channels)
                .mask()
                .map(|mask| mask.positions())
        })?;
    if positions.len() != channels {
        return None;
    }
    Some(positions.into_iter().map(ebu_channel_position).collect())
}

fn ebu_channel_position(position: ChannelPosition) -> EbuChannel {
    match position {
        ChannelPosition::FrontLeft => EbuChannel::Left,
        ChannelPosition::FrontRight => EbuChannel::Right,
        ChannelPosition::FrontCenter => EbuChannel::Center,
        ChannelPosition::Lfe1 => EbuChannel::Unused,
        ChannelPosition::RearLeft => EbuChannel::LeftSurround,
        ChannelPosition::RearRight => EbuChannel::RightSurround,
        ChannelPosition::FrontLeftCenter => EbuChannel::MpSC,
        ChannelPosition::FrontRightCenter => EbuChannel::MmSC,
        ChannelPosition::RearCenter => EbuChannel::Mp180,
        ChannelPosition::SideLeft => EbuChannel::Mp090,
        ChannelPosition::SideRight => EbuChannel::Mm090,
        ChannelPosition::TopCenter => EbuChannel::Up000,
        ChannelPosition::TopFrontLeft => EbuChannel::Up030,
        ChannelPosition::TopFrontCenter => EbuChannel::Up000,
        ChannelPosition::TopFrontRight => EbuChannel::Um030,
        ChannelPosition::TopRearLeft => EbuChannel::Up110,
        ChannelPosition::TopRearCenter => EbuChannel::Up180,
        ChannelPosition::TopRearRight => EbuChannel::Um110,
    }
}

fn integrated_loudness(analyzer: &EbuR128) -> Result<f64, String> {
    let loudness = analyzer
        .loudness_global()
        .map_err(|error| format!("measure integrated loudness: {error}"))?;
    if !loudness.is_finite() {
        return Err("integrated loudness is undefined (audio may be silent or too short)".into());
    }
    Ok(loudness)
}

fn true_peak_dbtp(analyzer: &EbuR128, channels: usize) -> Result<f64, String> {
    let mut peak = 0.0f64;
    for channel in 0..channels {
        peak = peak.max(
            analyzer
                .true_peak(channel as u32)
                .map_err(|error| format!("measure true peak: {error}"))?,
        );
    }
    Ok(amplitude_to_db(peak))
}

fn amplitude_to_db(amplitude: f64) -> f64 {
    20.0 * amplitude.max(1e-10).log10()
}

fn finite_metric(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn has_duration(audio: &Audio, milliseconds: usize) -> bool {
    let required_frames = (audio.sample_rate as usize)
        .saturating_mul(milliseconds)
        .saturating_add(999)
        / 1_000;
    audio.frames() >= required_frames
}

#[cfg(test)]
mod tests {
    use crate::channel_layout::ChannelLayout;

    use super::*;

    fn reference_stereo_sine() -> Audio {
        let sample_rate = 48_000;
        let frames = sample_rate as usize * 5;
        let mut channel = Vec::with_capacity(frames);
        let mut accumulator = 0.0f32;
        let step = 2.0 * std::f32::consts::PI * 440.0 / sample_rate as f32;
        for _ in 0..frames {
            channel.push(accumulator.sin() as f64);
            accumulator += step;
        }
        Audio {
            sample_rate,
            channels: vec![channel.clone(), channel],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn sine_channel(sample_rate: u32, seconds: usize, amplitude: f64) -> Vec<f64> {
        (0..sample_rate as usize * seconds)
            .map(|index| {
                let time = index as f64 / sample_rate as f64;
                amplitude * (2.0 * std::f64::consts::PI * 440.0 * time).sin()
            })
            .collect()
    }

    fn multichannel_audio(
        channels: Vec<Vec<f64>>,
        channel_mask: Option<crate::channel_layout::ChannelMask>,
    ) -> Audio {
        Audio {
            sample_rate: 48_000,
            channels,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask,
        }
    }

    #[test]
    fn reaches_loudness_target_without_exceeding_true_peak() {
        let sample_rate = 48_000;
        let channel = (0..sample_rate * 2)
            .map(|index| {
                let time = index as f64 / sample_rate as f64;
                0.08 * (2.0 * std::f64::consts::PI * 440.0 * time).sin()
            })
            .collect();
        let mut audio = Audio {
            sample_rate,
            channels: vec![channel],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let report = normalize(&mut audio, -20.0, -1.0).unwrap();
        assert!((report.output_lufs + 20.0).abs() < 0.1);
        assert!(report.true_peak_dbtp <= -1.0 + 1e-6);
    }

    #[test]
    fn true_peak_measurement_catches_intersample_overshoot() {
        let sample_rate = 48_000;
        let mut channel = Vec::with_capacity(sample_rate as usize * 2);
        for index in 0..sample_rate as usize * 2 {
            channel.push(if index % 2 == 0 { 0.9 } else { -0.9 });
        }
        let sample_peak = channel.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let audio = Audio {
            sample_rate,
            channels: vec![channel],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let (_, true_peak_dbtp) = measure(&audio).unwrap();
        let sample_peak_dbtp = 20.0 * sample_peak.log10();
        assert!(
            true_peak_dbtp > sample_peak_dbtp + 0.01,
            "true peak {true_peak_dbtp:.3} dBTP did not exceed sample peak {sample_peak_dbtp:.3} dB"
        );
    }

    #[test]
    fn reports_ebu_r128_reference_metrics() {
        let metrics = measure_detailed(&reference_stereo_sine()).unwrap();
        assert!((metrics.integrated_lufs + 0.6826).abs() < 1e-3);
        assert!(metrics
            .momentary_lufs
            .is_some_and(|value| (value + 0.6813).abs() < 1e-3));
        assert!(metrics
            .short_term_lufs
            .is_some_and(|value| (value + 0.6828).abs() < 1e-3));
        assert!(metrics
            .loudness_range_lu
            .is_some_and(|value| value.abs() < 1e-3));
        assert!(metrics
            .relative_threshold_lufs
            .is_some_and(|value| (value + 10.6826).abs() < 1e-3));
        assert!(metrics.sample_peak_dbfs > -1e-6);
        assert!(metrics.true_peak_dbtp > 0.0);
    }

    #[test]
    fn omits_windowed_metrics_when_the_input_is_short() {
        let sample_rate = 48_000;
        let channel = (0..sample_rate as usize * 2)
            .map(|index| {
                let time = index as f64 / sample_rate as f64;
                0.08 * (2.0 * std::f64::consts::PI * 440.0 * time).sin()
            })
            .collect();
        let audio = Audio {
            sample_rate,
            channels: vec![channel],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let metrics = measure_detailed(&audio).unwrap();
        assert!(metrics.momentary_lufs.is_some());
        assert!(metrics.short_term_lufs.is_none());
        assert!(metrics.loudness_range_lu.is_none());
    }

    #[test]
    fn excludes_lfe_from_multichannel_loudness() {
        let sample_rate = 48_000;
        let frames = sample_rate as usize * 2;
        let lfe_only = multichannel_audio(
            vec![vec![0.0; frames], vec![0.0; frames], vec![1.0; frames]],
            ChannelLayout::TwoPointOne.mask(),
        );
        assert!(measure(&lfe_only).is_err());

        let stereo = multichannel_audio(
            vec![
                sine_channel(sample_rate, 2, 0.1),
                sine_channel(sample_rate, 2, 0.1),
            ],
            ChannelLayout::Stereo.mask(),
        );
        let stereo_with_lfe = multichannel_audio(
            vec![
                stereo.channels[0].clone(),
                stereo.channels[1].clone(),
                vec![1.0; frames],
            ],
            ChannelLayout::TwoPointOne.mask(),
        );
        let (stereo_lufs, _) = measure(&stereo).unwrap();
        let (with_lfe_lufs, _) = measure(&stereo_with_lfe).unwrap();
        assert!((stereo_lufs - with_lfe_lufs).abs() < 1e-6);
    }

    #[test]
    fn applies_bs1770_surround_weight_and_scans_every_true_peak_channel() {
        let sample_rate = 48_000;
        let frames = sample_rate as usize * 2;
        let mask = ChannelLayout::FivePointZero.mask();
        let mut front_channels = vec![vec![0.0; frames]; 5];
        front_channels[0] = sine_channel(sample_rate, 2, 0.1);
        let mut surround_channels = vec![vec![0.0; frames]; 5];
        surround_channels[3] = sine_channel(sample_rate, 2, 0.1);
        let (front_lufs, _) = measure(&multichannel_audio(front_channels, mask)).unwrap();
        let (surround_lufs, _) = measure(&multichannel_audio(surround_channels, mask)).unwrap();
        let surround_gain_db = 10.0 * 1.41f64.log10();
        assert!((surround_lufs - front_lufs - surround_gain_db).abs() < 0.05);

        let mut true_peak_channels = vec![vec![0.0; frames]; 8];
        for (index, sample) in true_peak_channels[7].iter_mut().enumerate() {
            *sample = if index % 2 == 0 { 0.9 } else { -0.9 };
        }
        let metrics = measure_detailed(&multichannel_audio(
            true_peak_channels,
            ChannelLayout::SevenPointOne.mask(),
        ))
        .unwrap();
        assert!((metrics.sample_peak_dbfs - 20.0 * 0.9f64.log10()).abs() < 1e-6);
        assert!(metrics.true_peak_dbtp > metrics.sample_peak_dbfs + 0.01);
    }

    #[test]
    fn normalize_sanitizes_nonfinite_and_extreme_samples() {
        let mut audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.1; 48_000]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        audio.channels[0][0] = f64::NAN;
        audio.channels[0][1] = f64::INFINITY;
        audio.channels[0][2] = 1e300;
        normalize(&mut audio, -20.0, -1.0).unwrap();
        assert!(audio.channels[0].iter().all(|sample| sample.is_finite()));
        assert!(audio.channels[0].iter().all(|sample| sample.abs() <= 1.0));
    }

    #[test]
    fn bounded_streaming_analyzer_matches_constant_gain_contract() {
        let audio = reference_stereo_sine();
        let (input_lufs, input_peak) = measure(&audio).unwrap();
        let mut analyzer = StreamingLoudnessAnalyzer::new(
            audio.channels(),
            audio.sample_rate,
            audio.effective_channel_mask(),
        )
        .unwrap();
        for start in (0..audio.frames()).step_by(997) {
            let end = (start + 997).min(audio.frames());
            let block: Vec<Vec<f64>> = audio
                .channels
                .iter()
                .map(|channel| channel[start..end].to_vec())
                .collect();
            analyzer.add_block(&block).unwrap();
        }
        let gain = analyzer.finish(-18.0, -1.0).unwrap();
        let report = gain.report();
        assert!((report.input_lufs - input_lufs).abs() < 0.05);
        let expected_gain = (-18.0 - report.input_lufs).min(-1.0 - input_peak);
        assert!((report.gain_db - expected_gain).abs() < 0.05);
        assert!((report.output_lufs - (report.input_lufs + report.gain_db)).abs() < 1e-12);
        assert!(report.true_peak_dbtp <= -1.0 + 1e-9);
    }

    #[test]
    fn bounded_streaming_gain_preserves_block_geometry() {
        let mut block = vec![vec![0.5, -0.5], vec![0.25, -0.25]];
        let gain = StreamingLoudnessGain {
            report: LoudnessReport {
                input_lufs: -10.0,
                output_lufs: -16.0,
                true_peak_dbtp: -7.0,
                gain_db: -6.0,
            },
            linear: 0.5,
        };
        gain.apply(&mut block);
        assert_eq!(block, vec![vec![0.25, -0.25], vec![0.125, -0.125]]);
    }
}
