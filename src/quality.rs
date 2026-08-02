//! Full-reference perceptual quality metrics.
//!
//! SI-SDR/SI-SNR and the artifact indicators in [`crate::benchmark`] are
//! useful, deterministic engineering signals, but they do not model speech
//! intelligibility or a listener's quality judgement.  This module contains
//! the reference-based metrics that can be calculated without a model file.
//! STOI is implemented locally from the published 10 kHz, one-third-octave
//! algorithm.  ViSQOL is available behind the `visqol` feature and uses the
//! MIT-licensed pure-Rust implementation from `audio_samples_qoe`.

use crate::audio::Audio;
use crate::fft::{Complex, Fft};
use crate::resample::resample;

const STOI_RATE: u32 = 10_000;
const STOI_FRAME: usize = 256;
const STOI_FFT: usize = 512;
const STOI_HOP: usize = STOI_FRAME / 2;
const STOI_SEGMENT_FRAMES: usize = 30;
const STOI_BANDS: usize = 15;
const STOI_MIN_DB: f64 = -15.0;

/// Full-reference quality metrics emitted by a benchmark report.
///
/// `stoi` is a normalized intelligibility estimate in `[0, 1]`.  ViSQOL is
/// a MOS-LQO estimate in `[1, 5]` when the crate is built with `--features
/// visqol`; it is `None` otherwise.  PESQ intentionally remains `None`: the
/// ITU-T P.862 reference implementation and its conformance material are not
/// distributed under a license that can be bundled by this project.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QualityMetrics {
    /// Short-Time Objective Intelligibility, normalized to `[0, 1]`.
    pub stoi: Option<f64>,
    /// ITU-T P.862 PESQ.  Unavailable unless an external licensed adapter is
    /// supplied by the caller.
    pub pesq: Option<f64>,
    /// ViSQOL MOS-LQO in `[1, 5]`, available with the `visqol` feature.
    pub visqol: Option<f64>,
}

impl QualityMetrics {
    /// Compare a reference/test pair.
    ///
    /// Metric availability is represented with `Option` so an unsupported
    /// sample rate, an input that is too short for a metric, or an omitted
    /// optional implementation never prevents the regular benchmark report
    /// from being generated.
    pub fn compare(reference: &Audio, test: &Audio) -> Self {
        Self {
            stoi: stoi(reference, test),
            pesq: None,
            visqol: visqol(reference, test),
        }
    }
}

/// Calculate the normalized STOI score for a reference/test pair.
///
/// This follows the original STOI feature extraction: both signals are
/// downmixed and resampled to 10 kHz, analysed with a 256-sample Hann window
/// and 50% hop, grouped into 15 one-third-octave bands, then scored over
/// 30-frame temporal segments with 15 dB clipping.  The implementation uses
/// the project's own FFT and resampler so it is deterministic and has no
/// external model or runtime dependency.
pub fn stoi(reference: &Audio, test: &Audio) -> Option<f64> {
    if reference.channels.is_empty()
        || test.channels.is_empty()
        || reference.sample_rate == 0
        || test.sample_rate == 0
    {
        return None;
    }

    let reference_rate = reference.sample_rate;
    let test_rate = test.sample_rate;
    let reference = downmix(reference);
    let test = downmix(test);
    let reference = resample(&reference, reference_rate, STOI_RATE).ok()?;
    let test = resample(&test, test_rate, STOI_RATE).ok()?;
    let frames = reference.len().min(test.len());
    if frames < STOI_FRAME + STOI_HOP * (STOI_SEGMENT_FRAMES - 1) {
        return None;
    }

    let reference = &reference[..frames];
    let test = &test[..frames];
    let reference_envelopes = band_envelopes(reference);
    let test_envelopes = band_envelopes(test);
    let frame_count = reference_envelopes.len().min(test_envelopes.len());
    if frame_count < STOI_SEGMENT_FRAMES {
        return None;
    }

    // The clipping limit is the 15 dB tolerance used by STOI.  Expressing it
    // as a linear ratio avoids repeatedly evaluating a logarithm in the hot
    // loop.
    let clipping_ratio = 10.0_f64.powf(STOI_MIN_DB / 20.0);
    let mut total = 0.0;
    let mut count = 0usize;
    for start in 0..=frame_count - STOI_SEGMENT_FRAMES {
        for band in 0..STOI_BANDS {
            let mut clean = [0.0; STOI_SEGMENT_FRAMES];
            let mut degraded = [0.0; STOI_SEGMENT_FRAMES];
            for offset in 0..STOI_SEGMENT_FRAMES {
                clean[offset] = reference_envelopes[start + offset][band];
                degraded[offset] = test_envelopes[start + offset][band];
            }
            if let Some(score) = segment_correlation(&clean, &degraded, clipping_ratio) {
                total += score;
                count += 1;
            }
        }
    }
    (count > 0).then(|| (total / count as f64).clamp(0.0, 1.0))
}

fn downmix(audio: &Audio) -> Vec<f64> {
    let frames = audio.frames();
    if audio.channels.is_empty() {
        return Vec::new();
    }
    (0..frames)
        .map(|index| {
            audio
                .channels
                .iter()
                .map(|channel| channel[index])
                .sum::<f64>()
                / audio.channels.len() as f64
        })
        .collect()
}

fn band_envelopes(signal: &[f64]) -> Vec<[f64; STOI_BANDS]> {
    let fft = Fft::new(STOI_FFT);
    let window = (0..STOI_FRAME)
        .map(|index| {
            0.5 - 0.5
                * (2.0 * std::f64::consts::PI * index as f64
                    / (STOI_FRAME.saturating_sub(1) as f64))
                    .cos()
        })
        .collect::<Vec<_>>();
    let starts = frame_starts(signal.len());
    let bands = band_bins();
    let mut buffer = vec![Complex::default(); STOI_FFT];
    let mut output = Vec::with_capacity(starts.len());

    for start in starts {
        buffer.fill(Complex::default());
        for index in 0..STOI_FRAME {
            if let Some(sample) = signal.get(start + index).copied() {
                let sample = if sample.is_finite() { sample } else { 0.0 };
                buffer[index] = Complex::new(sample * window[index], 0.0);
            }
        }
        fft.forward(&mut buffer);
        let mut envelope = [0.0; STOI_BANDS];
        for (band, &(first, last)) in bands.iter().enumerate() {
            envelope[band] = (first..=last)
                .map(|bin| buffer[bin].re.hypot(buffer[bin].im))
                .sum();
        }
        output.push(envelope);
    }
    output
}

fn frame_starts(frames: usize) -> Vec<usize> {
    if frames == 0 {
        return Vec::new();
    }
    let mut starts = Vec::with_capacity(frames / STOI_HOP + 1);
    let mut start = 0;
    while start < frames {
        starts.push(start);
        if start >= frames.saturating_sub(STOI_FRAME) {
            break;
        }
        start += STOI_HOP;
    }
    starts
}

fn band_bins() -> [(usize, usize); STOI_BANDS] {
    let mut bands = [(0usize, 0usize); STOI_BANDS];
    for (band, slot) in bands.iter_mut().enumerate() {
        let lower_hz = 150.0 * 2.0_f64.powf(band as f64 / 3.0);
        let upper_hz = 150.0 * 2.0_f64.powf((band + 1) as f64 / 3.0);
        let first = (lower_hz * STOI_FFT as f64 / STOI_RATE as f64).ceil() as usize;
        let last = (upper_hz * STOI_FFT as f64 / STOI_RATE as f64)
            .floor()
            .min((STOI_FFT / 2) as f64) as usize;
        *slot = (first.min(STOI_FFT / 2), last.max(first).min(STOI_FFT / 2));
    }
    bands
}

fn segment_correlation(
    clean: &[f64; STOI_SEGMENT_FRAMES],
    degraded: &[f64; STOI_SEGMENT_FRAMES],
    clipping_ratio: f64,
) -> Option<f64> {
    let clean_energy = clean.iter().map(|value| value * value).sum::<f64>();
    let degraded_energy = degraded.iter().map(|value| value * value).sum::<f64>();
    if clean_energy <= 1e-20 {
        return None;
    }

    let scale = if degraded_energy <= 1e-20 {
        0.0
    } else {
        (clean_energy / degraded_energy).sqrt()
    };
    let mut clipped = [0.0; STOI_SEGMENT_FRAMES];
    for (index, value) in clipped.iter_mut().enumerate() {
        let scaled = degraded[index] * scale;
        *value = scaled.min(clean[index] * (1.0 + clipping_ratio));
    }

    let clean_mean = clean.iter().sum::<f64>() / clean.len() as f64;
    let degraded_mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let mut numerator = 0.0;
    let mut clean_variance = 0.0;
    let mut degraded_variance = 0.0;
    for (&clean_value, &degraded_value) in clean.iter().zip(clipped.iter()) {
        let clean_value = clean_value - clean_mean;
        let degraded_value = degraded_value - degraded_mean;
        numerator += clean_value * degraded_value;
        clean_variance += clean_value * clean_value;
        degraded_variance += degraded_value * degraded_value;
    }

    if clean_variance <= 1e-20 && degraded_variance <= 1e-20 {
        // A stationary band has no variance for Pearson correlation.  It is
        // still a perfect match when the normalized envelopes agree.
        let error = clean
            .iter()
            .zip(clipped.iter())
            .map(|(clean, degraded)| (clean - degraded).abs())
            .sum::<f64>()
            / clean.iter().sum::<f64>().max(1e-20);
        return Some(if error <= 1e-6 { 1.0 } else { 0.0 });
    }
    if clean_variance <= 1e-20 || degraded_variance <= 1e-20 {
        return Some(0.0);
    }
    Some((numerator / (clean_variance * degraded_variance).sqrt()).clamp(-1.0, 1.0))
}

#[cfg(feature = "visqol")]
fn visqol(reference: &Audio, test: &Audio) -> Option<f64> {
    use audio_samples::AudioSamples;
    use audio_samples_qoe::{visqol as calculate_visqol, VisqolOptions};
    use ndarray_visqol::Array1;
    use std::num::NonZeroU32;

    let rate = NonZeroU32::new(reference.sample_rate)?;
    let reference = AudioSamples::new_mono(Array1::from_vec(downmix(reference)), rate).ok()?;
    let test = AudioSamples::new_mono(Array1::from_vec(downmix(test)), rate).ok()?;
    let score = calculate_visqol(&reference, test, &VisqolOptions::audio()).ok()?;
    score.is_finite().then(|| score.clamp(1.0, 5.0))
}

#[cfg(not(feature = "visqol"))]
fn visqol(_reference: &Audio, _test: &Audio) -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn mono(samples: Vec<f64>, sample_rate: u32) -> Audio {
        Audio {
            sample_rate,
            channels: vec![samples],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn speech_like(seconds: f64) -> Vec<f64> {
        let rate = 16_000.0;
        (0..(seconds * rate) as usize)
            .map(|index| {
                let t = index as f64 / rate;
                let carrier = (2.0 * std::f64::consts::PI * 180.0 * t).sin();
                let harmonic = (2.0 * std::f64::consts::PI * 540.0 * t).sin() * 0.35;
                let syllable = (2.0 * std::f64::consts::PI * 3.2 * t)
                    .sin()
                    .mul_add(0.35, 0.65);
                (carrier + harmonic) * syllable * 0.3
            })
            .collect()
    }

    #[test]
    fn stoi_is_high_for_identical_audio_and_lower_for_noise() {
        let clean = speech_like(2.0);
        let noisy = clean
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                sample
                    + (2.0 * std::f64::consts::PI * 2_731.0 * index as f64 / 16_000.0).sin() * 0.25
            })
            .collect::<Vec<_>>();
        let clean_audio = mono(clean.clone(), 16_000);
        let identical = stoi(&clean_audio, &clean_audio).expect("STOI for a valid fixture");
        let degraded = stoi(&clean_audio, &mono(noisy, 16_000)).expect("STOI for a valid fixture");
        assert!(identical > 0.99, "identical STOI: {identical}");
        assert!(degraded < identical - 0.05, "degraded STOI: {degraded}");
    }

    #[test]
    fn short_audio_reports_unavailable_stoi() {
        let audio = mono(vec![0.0; STOI_FRAME], 16_000);
        assert_eq!(stoi(&audio, &audio), None);
    }

    #[test]
    fn quality_metrics_leave_licensed_pesq_unmeasured() {
        let audio = mono(speech_like(1.0), 16_000);
        let metrics = QualityMetrics::compare(&audio, &audio);
        assert!(metrics.stoi.is_some());
        assert_eq!(metrics.pesq, None);
        #[cfg(not(feature = "visqol"))]
        assert_eq!(metrics.visqol, None);
    }

    #[cfg(feature = "visqol")]
    #[test]
    fn visqol_reports_mos_for_a_valid_fixture() {
        let audio = mono(speech_like(3.0), 16_000);
        let metrics = QualityMetrics::compare(&audio, &audio);
        let score = metrics.visqol.expect("ViSQOL for a valid fixture");
        assert!((1.0..=5.0).contains(&score), "ViSQOL score: {score}");
        assert!(score > 4.0, "identical ViSQOL score: {score}");
    }
}
