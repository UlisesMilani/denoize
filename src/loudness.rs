//! EBU R128 / ITU-R BS.1770 loudness measurement and normalization.
//!
//! [`measure`] keeps the compact integrated-LUFS/true-peak interface used by
//! the normalization pipeline. [`measure_detailed`] exposes the standard
//! momentary, short-term, integrated, and loudness-range measurements as well
//! as the relative gate threshold and sample/true peaks.

use crate::{sanitize_sample, Audio};
use ebur128::{EbuR128, Mode};

#[derive(Clone, Copy, Debug)]
pub struct LoudnessReport {
    pub input_lufs: f64,
    pub output_lufs: f64,
    pub true_peak_dbtp: f64,
    pub gain_db: f64,
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
    if !target_lufs.is_finite() || !(-70.0..=0.0).contains(&target_lufs) {
        return Err("loudness target must be between -70 and 0 LUFS".into());
    }
    if !peak_limit_dbtp.is_finite() || !(-20.0..=0.0).contains(&peak_limit_dbtp) {
        return Err("true-peak limit must be between -20 and 0 dBTP".into());
    }
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
}
