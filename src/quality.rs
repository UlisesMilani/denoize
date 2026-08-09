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
use crate::stoi_resample::resample;

const STOI_RATE: u32 = 10_000;
const STOI_FRAME: usize = 256;
const STOI_FFT: usize = 512;
const STOI_HOP: usize = STOI_FRAME / 2;
const STOI_SEGMENT_FRAMES: usize = 30;
const STOI_BANDS: usize = 15;
const STOI_MIN_DB: f64 = -15.0;
const STOI_DYNAMIC_RANGE_DB: f64 = 40.0;

/// Full-reference quality metrics emitted by a benchmark report.
///
/// `stoi` is an intelligibility estimate in `[-1, 1]`.  ViSQOL is
/// a MOS-LQO estimate in `[1, 5]` when the crate is built with `--features
/// visqol`; it is `None` otherwise.  PESQ intentionally remains `None`: the
/// ITU-T P.862 reference implementation and its conformance material are not
/// distributed under a license that can be bundled by this project.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QualityMetrics {
    /// Short-Time Objective Intelligibility in `[-1, 1]`.
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

/// Calculate the STOI score for a reference/test pair.
///
/// This follows the original STOI feature extraction: both signals are
/// downmixed and resampled to 10 kHz, reference-silent frames more than 40 dB
/// below the peak are removed, and the retained signal is analysed with a
/// 256-sample Hann window and 50% hop. The spectra are grouped into 15
/// one-third-octave bands and scored over 30-frame temporal segments with
/// 15 dB clipping. The implementation uses the project's own FFT and
/// resampler so it is deterministic and has no external model or runtime
/// dependency.
pub fn stoi(reference: &Audio, test: &Audio) -> Option<f64> {
    if reference.channels.is_empty()
        || test.channels.is_empty()
        || reference.sample_rate == 0
        || test.sample_rate == 0
        || reference.sample_rate != test.sample_rate
        || !has_uniform_channels(reference)
        || !has_uniform_channels(test)
        || reference.frames() != test.frames()
    {
        return None;
    }

    let reference_rate = reference.sample_rate;
    let test_rate = test.sample_rate;
    let reference = downmix(reference);
    let test = downmix(test);
    let reference = resample(&reference, reference_rate, STOI_RATE)?;
    let test = resample(&test, test_rate, STOI_RATE)?;
    if reference.len() != test.len() {
        return None;
    }
    let frames = reference.len();
    if frames <= STOI_FRAME {
        return None;
    }

    let reference = &reference[..frames];
    let test = &test[..frames];
    let (reference, test) = remove_silent_frames(reference, test)?;
    let reference_envelopes = band_envelopes(&reference);
    let test_envelopes = band_envelopes(&test);
    let frame_count = reference_envelopes.len().min(test_envelopes.len());
    if frame_count < STOI_SEGMENT_FRAMES {
        return None;
    }

    // The clipping limit is the 15 dB tolerance used by STOI.  Expressing it
    // as a linear ratio avoids repeatedly evaluating a logarithm in the hot
    // loop.
    let clipping_ratio = 10.0_f64.powf(-STOI_MIN_DB / 20.0);
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
            total += segment_correlation(&clean, &degraded, clipping_ratio);
            count += 1;
        }
    }
    let score = total / count as f64;
    score.is_finite().then(|| score.clamp(-1.0, 1.0))
}

fn has_uniform_channels(audio: &Audio) -> bool {
    let Some(frames) = audio.channels.first().map(Vec::len) else {
        return false;
    };
    audio.channels.iter().all(|channel| channel.len() == frames)
}

fn downmix(audio: &Audio) -> Vec<f64> {
    let Some(frames) = audio.channels.iter().map(Vec::len).min() else {
        return Vec::new();
    };
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

fn hann_window() -> [f64; STOI_FRAME] {
    std::array::from_fn(|index| {
        0.5 - 0.5
            * (2.0 * std::f64::consts::PI * (index + 1) as f64 / (STOI_FRAME + 1) as f64).cos()
    })
}

fn finite_sample(sample: f64) -> f64 {
    if sample.is_finite() {
        sample
    } else {
        0.0
    }
}

fn remove_silent_frames(reference: &[f64], test: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let starts = frame_starts(reference.len().min(test.len()));
    if starts.is_empty() {
        return None;
    }

    let window = hann_window();
    let energies = starts
        .iter()
        .map(|&start| {
            let norm = (0..STOI_FRAME)
                .map(|index| {
                    let sample = finite_sample(reference[start + index]) * window[index];
                    sample * sample
                })
                .sum::<f64>()
                .sqrt();
            20.0 * (norm + f64::EPSILON).log10()
        })
        .collect::<Vec<_>>();
    let max_energy = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let retained = energies
        .iter()
        .filter(|&&energy| energy > max_energy - STOI_DYNAMIC_RANGE_DB)
        .count();
    if retained == 0 {
        return None;
    }

    let output_len = STOI_FRAME + STOI_HOP * (retained - 1);
    let mut reference_output = vec![0.0; output_len];
    let mut test_output = vec![0.0; output_len];
    let mut output_frame = 0usize;
    for (&start, &energy) in starts.iter().zip(energies.iter()) {
        if energy <= max_energy - STOI_DYNAMIC_RANGE_DB {
            continue;
        }
        let output_start = output_frame * STOI_HOP;
        for index in 0..STOI_FRAME {
            reference_output[output_start + index] +=
                finite_sample(reference[start + index]) * window[index];
            test_output[output_start + index] += finite_sample(test[start + index]) * window[index];
        }
        output_frame += 1;
    }
    Some((reference_output, test_output))
}

fn band_envelopes(signal: &[f64]) -> Vec<[f64; STOI_BANDS]> {
    let fft = Fft::new(STOI_FFT);
    let window = hann_window();
    let starts = frame_starts(signal.len());
    let bands = band_bins();
    let mut buffer = vec![Complex::default(); STOI_FFT];
    let mut output = Vec::with_capacity(starts.len());

    for start in starts {
        buffer.fill(Complex::default());
        for index in 0..STOI_FRAME {
            let sample = finite_sample(signal[start + index]);
            buffer[index] = Complex::new(sample * window[index], 0.0);
        }
        fft.forward(&mut buffer);
        let mut envelope = [0.0; STOI_BANDS];
        for (band, &(first, end)) in bands.iter().enumerate() {
            envelope[band] = (first..end)
                .map(|bin| {
                    buffer[bin]
                        .re
                        .mul_add(buffer[bin].re, buffer[bin].im.powi(2))
                })
                .sum::<f64>()
                .sqrt();
        }
        output.push(envelope);
    }
    output
}

fn frame_starts(frames: usize) -> Vec<usize> {
    let mut starts = Vec::with_capacity(frames / STOI_HOP);
    let mut start = 0;
    while start < frames.saturating_sub(STOI_FRAME) {
        starts.push(start);
        start += STOI_HOP;
    }
    starts
}

fn band_bins() -> [(usize, usize); STOI_BANDS] {
    let mut bands = [(0usize, 0usize); STOI_BANDS];
    for (band, slot) in bands.iter_mut().enumerate() {
        let center_hz = 150.0 * 2.0_f64.powf(band as f64 / 3.0);
        let lower_hz = center_hz * 2.0_f64.powf(-1.0 / 6.0);
        let upper_hz = center_hz * 2.0_f64.powf(1.0 / 6.0);
        let first = (lower_hz * STOI_FFT as f64 / STOI_RATE as f64).round() as usize;
        let end = (upper_hz * STOI_FFT as f64 / STOI_RATE as f64).round() as usize;
        *slot = (
            first.min(STOI_FFT / 2),
            end.max(first + 1).min(STOI_FFT / 2 + 1),
        );
    }
    bands
}

fn segment_correlation(
    clean: &[f64; STOI_SEGMENT_FRAMES],
    degraded: &[f64; STOI_SEGMENT_FRAMES],
    clipping_ratio: f64,
) -> f64 {
    let clean_energy = clean.iter().map(|value| value * value).sum::<f64>();
    let degraded_energy = degraded.iter().map(|value| value * value).sum::<f64>();
    let scale = clean_energy.sqrt() / (degraded_energy.sqrt() + f64::EPSILON);
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

    let denominator =
        (clean_variance.sqrt() + f64::EPSILON) * (degraded_variance.sqrt() + f64::EPSILON);
    (numerator / denominator).clamp(-1.0, 1.0)
}

#[cfg(feature = "visqol")]
fn visqol(reference: &Audio, test: &Audio) -> Option<f64> {
    use audio_samples::AudioSamples;
    use audio_samples_qoe::{visqol as calculate_visqol, VisqolOptions};
    use ndarray::Array1;
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

    fn reference_fixture(seconds: f64, sample_rate: u32) -> Vec<f64> {
        let rate = sample_rate as f64;
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
    fn stoi_matches_reference_implementation_fixture() {
        // Precomputed with pystoi 0.4.1. The non-10 kHz cases also verify its
        // Octave-compatible polyphase resampling and signal alignment.
        let cases = [
            (8_000, 0.884_994_483_670_439_3),
            (10_000, 0.765_102_111_482_166_7),
            (16_000, 0.838_727_788_659_411_9),
            (44_100, 0.775_014_630_251_397_7),
            (48_000, 0.776_641_575_127_054_3),
        ];
        for (sample_rate, expected) in cases {
            let clean = reference_fixture(2.0, sample_rate);
            let noisy = clean
                .iter()
                .enumerate()
                .map(|(index, sample)| {
                    sample
                        + (2.0 * std::f64::consts::PI * 2_731.0 * index as f64 / sample_rate as f64)
                            .sin()
                            * 0.25
                })
                .collect::<Vec<_>>();
            let score = stoi(&mono(clean, sample_rate), &mono(noisy, sample_rate))
                .expect("STOI for the reference fixture");
            assert!(
                (score - expected).abs() < 1e-9,
                "{sample_rate} Hz reference score: expected {expected}, got {score}"
            );
        }
    }

    #[test]
    fn third_octave_bins_match_the_reference_filterbank() {
        assert_eq!(
            band_bins(),
            [
                (7, 9),
                (9, 11),
                (11, 14),
                (14, 17),
                (17, 22),
                (22, 27),
                (27, 34),
                (34, 43),
                (43, 55),
                (55, 69),
                (69, 87),
                (87, 109),
                (109, 138),
                (138, 174),
                (174, 219),
            ]
        );
    }

    #[test]
    fn band_envelopes_use_root_sum_square_energy() {
        let mut impulse = vec![0.0; STOI_FRAME + 1];
        impulse[STOI_FRAME / 2] = 1.0;
        let envelopes = band_envelopes(&impulse);
        let impulse_magnitude = hann_window()[STOI_FRAME / 2];
        assert_eq!(envelopes.len(), 1);
        assert!((envelopes[0][0] - impulse_magnitude * 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((envelopes[0][4] - impulse_magnitude * 5.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn silent_frame_removal_uses_the_reference_dynamic_range() {
        let mut reference = vec![1e-4; 641];
        reference[..STOI_FRAME].fill(1.0);
        let (reference, test) =
            remove_silent_frames(&reference, &reference).expect("retained active frames");
        assert_eq!(reference.len(), STOI_FRAME + STOI_HOP);
        assert_eq!(test.len(), reference.len());
    }

    #[test]
    fn stft_discards_incomplete_trailing_frames() {
        assert_eq!(frame_starts(3_968).len(), 29);
        assert_eq!(frame_starts(3_969).len(), 30);
        assert_eq!(frame_starts(3_969).last(), Some(&3_712));
    }

    #[test]
    fn clipping_uses_the_negative_fifteen_db_lower_bound() {
        let ratio = 10.0_f64.powf(-STOI_MIN_DB / 20.0);
        assert!((ratio - 5.623_413_251_903_491).abs() < 1e-12);

        let mut clean = [0.0; STOI_SEGMENT_FRAMES];
        for (index, value) in clean.iter_mut().enumerate() {
            *value = 1.0 + ((7 * index) % 11) as f64;
        }
        let mut degraded = clean;
        degraded[5] *= 5.0;
        degraded[17] *= 2.5;
        let score = segment_correlation(&clean, &degraded, ratio);
        assert!((score - 0.710_572_998_770_498_7).abs() < 1e-12);
    }

    #[test]
    fn ragged_channels_report_unavailable_stoi() {
        let ragged = Audio {
            sample_rate: STOI_RATE,
            channels: vec![vec![0.0; 5_000], vec![0.0; 4_999]],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        };
        assert_eq!(stoi(&ragged, &ragged), None);
    }

    #[test]
    fn mismatched_lengths_report_unavailable_stoi() {
        let reference = mono(reference_fixture(1.0, STOI_RATE), STOI_RATE);
        let mut truncated = reference_fixture(1.0, STOI_RATE);
        truncated.pop();
        assert_eq!(stoi(&reference, &mono(truncated, STOI_RATE)), None);

        let reference = mono(vec![0.0; 48_000], 48_000);
        let truncated = mono(vec![0.0; 47_999], 48_000);
        assert_eq!(stoi(&reference, &truncated), None);

        let different_rate = mono(vec![0.0; 10_000], 10_000);
        assert_eq!(stoi(&reference, &different_rate), None);
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
