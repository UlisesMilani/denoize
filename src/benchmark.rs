//! Objective quality and stereo-imaging reports.

use crate::fft::{Complex, Fft};
use crate::Audio;
use std::f64::consts::PI;

/// Normalized artifact-screening indicators derived from a reference/test pair.
///
/// Every score is in `[0, 1]`, where zero means that the corresponding
/// artifact was not detected and one means a strong indication. These are
/// deterministic, dependency-free screening signals rather than perceptual
/// listening-test scores; use them to catch regressions and inspect the audio
/// when a score rises.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtifactReport {
    /// Narrow spectral excesses in the test signal that are absent from the
    /// reference (a common musical-noise / "birdie" signature).
    pub musical_noise_score: f64,
    /// Frame-to-frame modulation of the test/reference level ratio.
    pub pumping_score: f64,
    /// Fraction of strong reference onsets whose spectral flux is missing in
    /// the test signal.
    pub transient_loss_score: f64,
    /// Inter-channel phase relationship error. `None` for mono input.
    pub phase_distortion_score: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct BenchmarkReport {
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: usize,
    pub si_sdr_db: f64,
    pub si_snr_db: f64,
    pub snr_db: f64,
    pub segmental_snr_db: f64,
    pub stereo_side_sdr_db: Option<f64>,
    pub correlation_error: Option<f64>,
    /// Dependency-free artifact-screening indicators.
    pub artifact_scores: ArtifactReport,
    /// Native STOI score in `[0, 1]` when the input is long enough.
    pub stoi: Option<f64>,
    /// PESQ is `None` unless a separately licensed external adapter is used.
    pub pesq: Option<f64>,
    /// ViSQOL MOS-LQO (`[1, 5]`) when built with the `visqol` feature.
    pub visqol: Option<f64>,
    pub elapsed_ms: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
}

impl BenchmarkReport {
    pub fn compare(reference: &Audio, test: &Audio) -> Result<Self, String> {
        validate_pair(reference, test)?;
        let frames = reference.frames().min(test.frames());
        let r = downmix(reference, frames);
        let t = downmix(test, frames);
        let artifact_scores = ArtifactReport::compare(reference, test)?;
        let quality_metrics = crate::quality::QualityMetrics::compare(reference, test);
        let (side_sdr, correlation_error) = if reference.channels.len() == 2 {
            let rs = side(reference, frames);
            let ts = side(test, frames);
            (
                Some(finite_db(si_sdr(&rs, &ts))),
                Some(
                    (correlation(
                        &reference.channels[0][..frames],
                        &reference.channels[1][..frames],
                    ) - correlation(&test.channels[0][..frames], &test.channels[1][..frames]))
                    .abs(),
                )
                .filter(|value| value.is_finite()),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            frames,
            sample_rate: reference.sample_rate,
            channels: reference.channels.len(),
            si_sdr_db: finite_db(si_sdr(&r, &t)),
            si_snr_db: finite_db(si_snr(&r, &t)),
            snr_db: finite_db(snr(&r, &t)),
            segmental_snr_db: finite_db(segmental_snr(&r, &t, reference.sample_rate)),
            stereo_side_sdr_db: side_sdr,
            correlation_error,
            artifact_scores,
            stoi: quality_metrics.stoi,
            pesq: quality_metrics.pesq,
            visqol: quality_metrics.visqol,
            elapsed_ms: None,
            peak_rss_bytes: None,
        })
    }

    pub fn json(&self) -> String {
        format!("{{\"frames\":{},\"sample_rate\":{},\"channels\":{},\"si_sdr_db\":{},\"si_snr_db\":{},\"snr_db\":{},\"segmental_snr_db\":{},\"stereo_side_sdr_db\":{},\"correlation_error\":{},\"artifact_scores\":{},\"stoi\":{},\"pesq\":{},\"visqol\":{},\"elapsed_ms\":{},\"peak_rss_bytes\":{}}}", self.frames, self.sample_rate, self.channels, json_number(self.si_sdr_db), json_number(self.si_snr_db), json_number(self.snr_db), json_number(self.segmental_snr_db), optional(self.stereo_side_sdr_db), optional(self.correlation_error), self.artifact_scores.json(), optional(self.stoi), optional(self.pesq), optional(self.visqol), optional(self.elapsed_ms), self.peak_rss_bytes.map_or_else(|| "null".into(), |v| v.to_string()))
    }

    pub fn markdown(&self) -> String {
        format!("| Metric | Value |\n|---|---:|\n| SI-SDR | {:.3} dB |\n| SI-SNR | {:.3} dB |\n| SNR | {:.3} dB |\n| Segmental SNR | {:.3} dB |\n| Stereo side SDR | {} |\n| Correlation error | {} |\n| Musical-noise score (0=none) | {:.3} |\n| Pumping score (0=none) | {:.3} |\n| Transient-loss score (0=none) | {:.3} |\n| Phase-distortion score (0=none) | {} |\n| STOI (0–1, higher is better) | {} |\n| PESQ (licensed adapter required) | {} |\n| ViSQOL MOS-LQO (1–5) | {} |", self.si_sdr_db, self.si_snr_db, self.snr_db, self.segmental_snr_db, db(self.stereo_side_sdr_db), display(self.correlation_error, 6), self.artifact_scores.musical_noise_score, self.artifact_scores.pumping_score, self.artifact_scores.transient_loss_score, display(self.artifact_scores.phase_distortion_score, 3), display(self.stoi, 4), display(self.pesq, 3), display(self.visqol, 3))
    }
}

impl ArtifactReport {
    /// Compare a reference signal with a test signal and calculate artifact
    /// screening indicators.
    pub fn compare(reference: &Audio, test: &Audio) -> Result<Self, String> {
        validate_pair(reference, test)?;
        let frames = reference.frames().min(test.frames());
        let ref_mix = downmix(reference, frames);
        let test_mix = downmix(test, frames);
        let observations = collect_artifact_observations(reference, test, &ref_mix, &test_mix);

        let musical_noise_score = normalized_score(
            observations.musical_noise_excess,
            observations.test_spectral_energy,
        );
        let pumping_score = pumping_score(&observations.rms_pairs);
        let transient_loss_score =
            transient_loss_score(&observations.ref_flux, &observations.test_flux);
        let phase_distortion_score = if reference.channels.len() == 2 {
            Some(normalized_score(
                observations.phase_error,
                observations.phase_weight,
            ))
        } else {
            None
        };

        Ok(Self {
            musical_noise_score,
            pumping_score,
            transient_loss_score,
            phase_distortion_score,
        })
    }

    /// Return the machine-readable representation used by benchmark reports.
    pub fn json(&self) -> String {
        format!(
            "{{\"musical_noise_score\":{},\"pumping_score\":{},\"transient_loss_score\":{},\"phase_distortion_score\":{}}}",
            json_number(self.musical_noise_score),
            json_number(self.pumping_score),
            json_number(self.transient_loss_score),
            optional(self.phase_distortion_score),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ComparisonReport {
    pub noisy: BenchmarkReport,
    pub enhanced: BenchmarkReport,
}

impl ComparisonReport {
    pub fn compare(clean: &Audio, noisy: &Audio, enhanced: &Audio) -> Result<Self, String> {
        Ok(Self {
            noisy: BenchmarkReport::compare(clean, noisy)?,
            enhanced: BenchmarkReport::compare(clean, enhanced)?,
        })
    }

    pub fn json(&self) -> String {
        format!(
            "{{\"noisy\":{},\"enhanced\":{},\"improvement\":{{\"si_sdr_db\":{},\"si_snr_db\":{},\"snr_db\":{},\"segmental_snr_db\":{},\"stereo_side_sdr_db\":{},\"correlation_error\":{},\"stoi\":{},\"pesq\":{},\"visqol\":{},\"musical_noise_score\":{},\"pumping_score\":{},\"transient_loss_score\":{},\"phase_distortion_score\":{}}}}}",
            self.noisy.json(), self.enhanced.json(),
            json_number(self.enhanced.si_sdr_db - self.noisy.si_sdr_db),
            json_number(self.enhanced.si_snr_db - self.noisy.si_snr_db),
            json_number(self.enhanced.snr_db - self.noisy.snr_db),
            json_number(self.enhanced.segmental_snr_db - self.noisy.segmental_snr_db),
            optional_difference(self.enhanced.stereo_side_sdr_db, self.noisy.stereo_side_sdr_db),
            optional_difference(self.noisy.correlation_error, self.enhanced.correlation_error),
            optional_difference(self.enhanced.stoi, self.noisy.stoi),
            optional_difference(self.enhanced.pesq, self.noisy.pesq),
            optional_difference(self.enhanced.visqol, self.noisy.visqol),
            json_number(self.noisy.artifact_scores.musical_noise_score - self.enhanced.artifact_scores.musical_noise_score),
            json_number(self.noisy.artifact_scores.pumping_score - self.enhanced.artifact_scores.pumping_score),
            json_number(self.noisy.artifact_scores.transient_loss_score - self.enhanced.artifact_scores.transient_loss_score),
            optional_difference(self.noisy.artifact_scores.phase_distortion_score, self.enhanced.artifact_scores.phase_distortion_score),
        )
    }

    pub fn markdown(&self) -> String {
        format!(
            "| Metric | Noisy | Enhanced | Improvement |\n|---|---:|---:|---:|\n| SI-SDR | {:.3} dB | {:.3} dB | {:+.3} dB |\n| SI-SNR | {:.3} dB | {:.3} dB | {:+.3} dB |\n| SNR | {:.3} dB | {:.3} dB | {:+.3} dB |\n| Segmental SNR | {:.3} dB | {:.3} dB | {:+.3} dB |\n| Stereo side SDR (higher is better) | {} | {} | {} |\n| Correlation error (lower is better) | {} | {} | {} |\n| STOI (higher is better) | {} | {} | {} |\n| PESQ (higher is better; licensed adapter) | {} | {} | {} |\n| ViSQOL MOS-LQO (higher is better) | {} | {} | {} |\n| Musical-noise score (lower is better) | {:.3} | {:.3} | {:+.3} |\n| Pumping score (lower is better) | {:.3} | {:.3} | {:+.3} |\n| Transient-loss score (lower is better) | {:.3} | {:.3} | {:+.3} |\n| Phase-distortion score (lower is better) | {} | {} | {} |\n\nArtifact scores are deterministic screening indicators in [0, 1], not perceptual listening-test scores. STOI is implemented natively. ViSQOL is measured when the `visqol` feature is enabled. PESQ remains unavailable because its ITU-T reference implementation requires a separately licensed external adapter.",
            self.noisy.si_sdr_db, self.enhanced.si_sdr_db, self.enhanced.si_sdr_db - self.noisy.si_sdr_db,
            self.noisy.si_snr_db, self.enhanced.si_snr_db, self.enhanced.si_snr_db - self.noisy.si_snr_db,
            self.noisy.snr_db, self.enhanced.snr_db, self.enhanced.snr_db - self.noisy.snr_db,
            self.noisy.segmental_snr_db, self.enhanced.segmental_snr_db, self.enhanced.segmental_snr_db - self.noisy.segmental_snr_db,
            db(self.noisy.stereo_side_sdr_db), db(self.enhanced.stereo_side_sdr_db), db(optional_difference_value(self.enhanced.stereo_side_sdr_db, self.noisy.stereo_side_sdr_db)),
            display(self.noisy.correlation_error, 6), display(self.enhanced.correlation_error, 6), display(optional_difference_value(self.noisy.correlation_error, self.enhanced.correlation_error), 6),
            display(self.noisy.stoi, 4), display(self.enhanced.stoi, 4), display(optional_difference_value(self.enhanced.stoi, self.noisy.stoi), 4),
            display(self.noisy.pesq, 3), display(self.enhanced.pesq, 3), display(optional_difference_value(self.enhanced.pesq, self.noisy.pesq), 3),
            display(self.noisy.visqol, 3), display(self.enhanced.visqol, 3), display(optional_difference_value(self.enhanced.visqol, self.noisy.visqol), 3),
            self.noisy.artifact_scores.musical_noise_score, self.enhanced.artifact_scores.musical_noise_score, self.noisy.artifact_scores.musical_noise_score - self.enhanced.artifact_scores.musical_noise_score,
            self.noisy.artifact_scores.pumping_score, self.enhanced.artifact_scores.pumping_score, self.noisy.artifact_scores.pumping_score - self.enhanced.artifact_scores.pumping_score,
            self.noisy.artifact_scores.transient_loss_score, self.enhanced.artifact_scores.transient_loss_score, self.noisy.artifact_scores.transient_loss_score - self.enhanced.artifact_scores.transient_loss_score,
            display(self.noisy.artifact_scores.phase_distortion_score, 3), display(self.enhanced.artifact_scores.phase_distortion_score, 3), display(optional_difference_value(self.noisy.artifact_scores.phase_distortion_score, self.enhanced.artifact_scores.phase_distortion_score), 3),
        )
    }

    pub fn html(&self) -> String {
        let rows = self
            .markdown()
            .lines()
            .skip(2)
            // Keep the HTML export in lockstep with every metric row in the
            // Markdown report.  A fixed row count silently dropped the
            // transient-loss and phase-distortion metrics when they were
            // added to the comparison report.
            .take_while(|line| !line.is_empty())
            .map(|line| {
                let cells = line
                    .trim_matches('|')
                    .split('|')
                    .map(str::trim)
                    .map(|cell| format!("<td>{cell}</td>"))
                    .collect::<String>();
                format!("<tr>{cells}</tr>")
            })
            .collect::<String>();
        format!("<!doctype html><meta charset=\"utf-8\"><title>denoize comparison</title><style>body{{font-family:system-ui;max-width:900px;margin:3rem auto}}table{{border-collapse:collapse}}td,th{{padding:.5rem 1rem;border:1px solid #ccc;text-align:right}}td:first-child{{text-align:left}}</style><h1>denoize quality comparison</h1><table><thead><tr><th>Metric</th><th>Noisy</th><th>Enhanced</th><th>Improvement</th></tr></thead><tbody>{rows}</tbody></table><p>Artifact scores are deterministic screening indicators in [0, 1], lower is better. STOI is native; ViSQOL requires the <code>visqol</code> feature; PESQ requires a separately licensed external adapter.</p>")
    }
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        "null".into()
    }
}

fn finite_db(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        -120.0
    }
}

fn optional(v: Option<f64>) -> String {
    v.map_or_else(|| "null".into(), json_number)
}
fn display(v: Option<f64>, precision: usize) -> String {
    v.filter(|value| value.is_finite())
        .map_or_else(|| "n/a".into(), |v| format!("{v:.precision$}"))
}
fn db(v: Option<f64>) -> String {
    v.filter(|value| value.is_finite())
        .map_or_else(|| "n/a".into(), |v| format!("{v:.3} dB"))
}

fn optional_difference(a: Option<f64>, b: Option<f64>) -> String {
    optional(optional_difference_value(a, b))
}

fn optional_difference_value(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
            let difference = a - b;
            difference.is_finite().then_some(difference)
        }
        _ => None,
    }
}

fn validate_pair(reference: &Audio, test: &Audio) -> Result<(), String> {
    if reference.sample_rate != test.sample_rate {
        return Err("benchmark sample rates differ".into());
    }
    if reference.channels.len() != test.channels.len() || reference.channels.is_empty() {
        return Err("benchmark channel counts differ or are empty".into());
    }
    if reference.frames().min(test.frames()) == 0 {
        return Err("benchmark inputs are empty".into());
    }
    Ok(())
}

#[derive(Default)]
struct ArtifactObservations {
    musical_noise_excess: f64,
    test_spectral_energy: f64,
    rms_pairs: Vec<(f64, f64)>,
    ref_flux: Vec<f64>,
    test_flux: Vec<f64>,
    phase_error: f64,
    phase_weight: f64,
}

/// Collect all four artifact signals in one shared STFT pass. The frame size
/// is intentionally bounded so a long benchmark stays predictable in both
/// memory and runtime.
fn collect_artifact_observations(
    reference: &Audio,
    test: &Audio,
    ref_mix: &[f64],
    test_mix: &[f64],
) -> ArtifactObservations {
    let frames = ref_mix.len().min(test_mix.len());
    let frame_size = artifact_frame_size(frames);
    let hop = frame_size / 2;
    let window = hann_window(frame_size);
    let fft = Fft::new(frame_size);
    let nbins = fft.nbins();
    let starts = frame_starts(frames, frame_size, hop);

    let mut ref_buffer = vec![Complex::default(); frame_size];
    let mut test_buffer = vec![Complex::default(); frame_size];
    let mut ref_left_buffer = vec![Complex::default(); frame_size];
    let mut ref_right_buffer = vec![Complex::default(); frame_size];
    let mut test_left_buffer = vec![Complex::default(); frame_size];
    let mut test_right_buffer = vec![Complex::default(); frame_size];
    let mut ref_magnitude = vec![0.0; nbins];
    let mut test_magnitude = vec![0.0; nbins];
    let mut prev_ref_magnitude = vec![0.0; nbins];
    let mut prev_test_magnitude = vec![0.0; nbins];
    let stereo = reference.channels.len() == 2;
    let mut observations = ArtifactObservations {
        rms_pairs: Vec::with_capacity(starts.len()),
        ref_flux: Vec::with_capacity(starts.len()),
        test_flux: Vec::with_capacity(starts.len()),
        ..ArtifactObservations::default()
    };

    for (frame_index, &start) in starts.iter().enumerate() {
        fill_windowed(&mut ref_buffer, ref_mix, start, &window);
        fill_windowed(&mut test_buffer, test_mix, start, &window);
        fft.forward(&mut ref_buffer);
        fft.forward(&mut test_buffer);
        magnitudes(&ref_buffer, &mut ref_magnitude);
        magnitudes(&test_buffer, &mut test_magnitude);

        observations.ref_flux.push(if frame_index == 0 {
            0.0
        } else {
            spectral_flux(&ref_magnitude, &prev_ref_magnitude)
        });
        observations.test_flux.push(if frame_index == 0 {
            0.0
        } else {
            spectral_flux(&test_magnitude, &prev_test_magnitude)
        });
        prev_ref_magnitude.copy_from_slice(&ref_magnitude);
        prev_test_magnitude.copy_from_slice(&test_magnitude);

        let ref_rms = frame_rms(ref_mix, start, frame_size);
        let test_rms = frame_rms(test_mix, start, frame_size);
        observations.rms_pairs.push((ref_rms, test_rms));

        // Musical noise is represented by narrow-band energy that is both
        // absent from the reference and much stronger than its neighbours.
        for k in 1..nbins.saturating_sub(1) {
            let test_power = test_magnitude[k] * test_magnitude[k];
            let ref_power = ref_magnitude[k] * ref_magnitude[k];
            let excess = (test_power - 1.15 * ref_power).max(0.0);
            let neighbour_power = 0.5
                * (test_magnitude[k - 1] * test_magnitude[k - 1]
                    + test_magnitude[k + 1] * test_magnitude[k + 1]);
            let prominence = ((test_power / (neighbour_power + 1e-30)) - 1.0).clamp(0.0, 4.0) / 4.0;
            observations.musical_noise_excess += excess * prominence;
        }
        observations.test_spectral_energy += test_magnitude.iter().map(|m| m * m).sum::<f64>();

        if stereo {
            fill_windowed(&mut ref_left_buffer, &reference.channels[0], start, &window);
            fill_windowed(
                &mut ref_right_buffer,
                &reference.channels[1],
                start,
                &window,
            );
            fill_windowed(&mut test_left_buffer, &test.channels[0], start, &window);
            fill_windowed(&mut test_right_buffer, &test.channels[1], start, &window);
            fft.forward(&mut ref_left_buffer);
            fft.forward(&mut ref_right_buffer);
            fft.forward(&mut test_left_buffer);
            fft.forward(&mut test_right_buffer);
            let (error, weight) = phase_error_for_frame(
                &ref_left_buffer,
                &ref_right_buffer,
                &test_left_buffer,
                &test_right_buffer,
            );
            observations.phase_error += error;
            observations.phase_weight += weight;
        }
    }

    observations
}

fn artifact_frame_size(frames: usize) -> usize {
    // At least 64 samples keeps the FFT meaningful for very short fixtures;
    // longer inputs use a power-of-two frame no larger than 1024 samples.
    frames.min(1024).max(64).next_power_of_two().min(1024)
}

fn frame_starts(frames: usize, frame_size: usize, hop: usize) -> Vec<usize> {
    let mut starts = Vec::with_capacity((frames / hop).saturating_add(1));
    let mut start = 0;
    while start < frames {
        starts.push(start);
        if start >= frames.saturating_sub(frame_size) {
            break;
        }
        start += hop;
    }
    starts
}

fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f64 / (size.saturating_sub(1) as f64)).cos())
        .collect()
}

fn fill_windowed(buffer: &mut [Complex], signal: &[f64], start: usize, window: &[f64]) {
    for (index, slot) in buffer.iter_mut().enumerate() {
        let sample = signal
            .get(start + index)
            .copied()
            .filter(|sample| sample.is_finite())
            .unwrap_or(0.0);
        *slot = Complex::new(sample * window[index], 0.0);
    }
}

fn magnitudes(spectrum: &[Complex], output: &mut [f64]) {
    for (index, magnitude) in output.iter_mut().enumerate() {
        let value = spectrum[index];
        *magnitude = value.re.hypot(value.im);
    }
}

fn spectral_flux(current: &[f64], previous: &[f64]) -> f64 {
    let rise = current
        .iter()
        .zip(previous)
        .map(|(current, previous)| (current - previous).max(0.0))
        .sum::<f64>();
    let previous_energy = previous.iter().sum::<f64>();
    (rise / (previous_energy + 1e-12)).clamp(0.0, 100.0)
}

fn frame_rms(signal: &[f64], start: usize, frame_size: usize) -> f64 {
    let end = (start + frame_size).min(signal.len());
    if start >= end {
        return 0.0;
    }
    let sum = signal[start..end]
        .iter()
        .filter(|sample| sample.is_finite())
        .map(|sample| sample * sample)
        .sum::<f64>();
    (sum / (end - start) as f64).sqrt()
}

fn phase_error_for_frame(
    reference_left: &[Complex],
    reference_right: &[Complex],
    test_left: &[Complex],
    test_right: &[Complex],
) -> (f64, f64) {
    let nbins = reference_left.len() / 2 + 1;
    let mut error = 0.0;
    let mut weight = 0.0;
    for k in 1..nbins.saturating_sub(1) {
        let reference_cross = cross_spectrum(reference_left[k], reference_right[k]);
        let test_cross = cross_spectrum(test_left[k], test_right[k]);
        let reference_magnitude = reference_cross.re.hypot(reference_cross.im);
        let test_magnitude = test_cross.re.hypot(test_cross.im);
        if reference_magnitude <= 1e-12 || test_magnitude <= 1e-12 {
            continue;
        }
        let cosine = ((reference_cross.re * test_cross.re + reference_cross.im * test_cross.im)
            / (reference_magnitude * test_magnitude))
            .clamp(-1.0, 1.0);
        let bin_weight = reference_magnitude.min(test_magnitude);
        error += 0.5 * (1.0 - cosine) * bin_weight;
        weight += bin_weight;
    }
    (error, weight)
}

fn cross_spectrum(left: Complex, right: Complex) -> Complex {
    // left * conjugate(right), preserving the inter-channel phase relation.
    Complex::new(
        left.re * right.re + left.im * right.im,
        left.im * right.re - left.re * right.im,
    )
}

fn normalized_score(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 1e-30 || !numerator.is_finite() || !denominator.is_finite() {
        0.0
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

fn pumping_score(rms_pairs: &[(f64, f64)]) -> f64 {
    let reference_peak = rms_pairs
        .iter()
        .map(|(reference, _)| *reference)
        .fold(0.0, f64::max);
    if reference_peak <= 1e-9 {
        return 0.0;
    }
    let floor = reference_peak * 0.01;
    let mut previous_gain: Option<f64> = None;
    let mut total_change = 0.0;
    let mut count = 0usize;
    for &(reference, test) in rms_pairs {
        if reference <= floor {
            previous_gain = None;
            continue;
        }
        let gain_db = (20.0 * ((test + floor) / (reference + floor)).log10()).clamp(-60.0, 60.0);
        if let Some(previous) = previous_gain {
            total_change += (gain_db - previous).abs();
            count += 1;
        }
        previous_gain = Some(gain_db);
    }
    if count == 0 {
        0.0
    } else {
        (total_change / count as f64 / 12.0).clamp(0.0, 1.0)
    }
}

fn transient_loss_score(reference_flux: &[f64], test_flux: &[f64]) -> f64 {
    let maximum = reference_flux.iter().copied().fold(0.0, f64::max);
    let threshold = (maximum * 0.15).max(0.05);
    let mut weighted_loss = 0.0;
    let mut total_weight = 0.0;
    for (&reference, &test) in reference_flux.iter().zip(test_flux) {
        if reference < threshold {
            continue;
        }
        weighted_loss += ((reference - test).max(0.0) / reference.max(1e-12)) * reference;
        total_weight += reference;
    }
    normalized_score(weighted_loss, total_weight)
}

fn downmix(a: &Audio, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a.channels.iter().map(|c| c[i]).sum::<f64>() / a.channels.len() as f64)
        .collect()
}
fn side(a: &Audio, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (a.channels[0][i] - a.channels[1][i]) * 0.5)
        .collect()
}

pub fn si_sdr(reference: &[f64], estimate: &[f64]) -> f64 {
    let dot = reference
        .iter()
        .zip(estimate)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let scale = dot / reference.iter().map(|x| x * x).sum::<f64>().max(1e-30);
    let target_energy = reference.iter().map(|x| (x * scale).powi(2)).sum::<f64>();
    let noise_energy = reference
        .iter()
        .zip(estimate)
        .map(|(a, b)| (a * scale - b).powi(2))
        .sum::<f64>();
    10.0 * (target_energy / noise_energy.max(1e-30)).log10()
}

pub fn si_snr(reference: &[f64], estimate: &[f64]) -> f64 {
    let rm = reference.iter().sum::<f64>() / reference.len() as f64;
    let em = estimate.iter().sum::<f64>() / estimate.len() as f64;
    si_sdr(
        &reference.iter().map(|x| x - rm).collect::<Vec<_>>(),
        &estimate.iter().map(|x| x - em).collect::<Vec<_>>(),
    )
}

pub fn snr(reference: &[f64], estimate: &[f64]) -> f64 {
    let signal = reference.iter().map(|sample| sample * sample).sum::<f64>();
    let noise = reference
        .iter()
        .zip(estimate)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    10.0 * (signal / noise.max(1e-30)).log10()
}

pub fn segmental_snr(reference: &[f64], estimate: &[f64], sample_rate: u32) -> f64 {
    let window = (sample_rate as usize / 50).max(1);
    let mut values = Vec::new();
    for (r, e) in reference.chunks(window).zip(estimate.chunks(window)) {
        let signal = r.iter().map(|sample| sample * sample).sum::<f64>();
        if signal > 1e-12 {
            values.push(snr(r, e).clamp(-10.0, 35.0));
        }
    }
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let dot = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    dot / (a.iter().map(|x| x * x).sum::<f64>() * b.iter().map(|x| x * x).sum::<f64>())
        .sqrt()
        .max(1e-30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_ignore_gain() {
        let reference = [1.0, -1.0, 0.5, -0.5];
        let estimate = [0.5, -0.5, 0.25, -0.25];
        assert!(si_sdr(&reference, &estimate) > 250.0);
        assert!(si_snr(&reference, &estimate) > 250.0);
    }

    #[test]
    fn comparison_reports_quality_improvement_in_all_formats() {
        let clean = Audio {
            sample_rate: 16_000,
            channels: vec![(0..1600)
                .map(|index| (index as f64 * 0.031).sin() * 0.5)
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        let noisy = Audio {
            sample_rate: clean.sample_rate,
            channels: vec![clean.channels[0]
                .iter()
                .enumerate()
                .map(|(index, sample)| sample + if index % 2 == 0 { 0.1 } else { -0.1 })
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        let enhanced = Audio {
            sample_rate: clean.sample_rate,
            channels: vec![clean.channels[0]
                .iter()
                .enumerate()
                .map(|(index, sample)| sample + if index % 2 == 0 { 0.02 } else { -0.02 })
                .collect()],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };

        let report = ComparisonReport::compare(&clean, &noisy, &enhanced).unwrap();
        assert!(report.enhanced.snr_db > report.noisy.snr_db);
        assert!(report.json().contains("\"improvement\""));
        assert!(report.json().contains("\"artifact_scores\""));
        assert!(report.json().contains("\"stoi\""));
        assert!(report.json().contains("\"pesq\""));
        assert!(report.json().contains("\"stereo_side_sdr_db\""));
        assert!(report.json().contains("\"correlation_error\""));
        assert!(report.json().contains("\"visqol\""));
        assert!(report.markdown().contains("Segmental SNR"));
        assert!(report.markdown().contains("Stereo side SDR"));
        assert!(report.markdown().contains("Correlation error"));
        assert!(report.markdown().contains("STOI"));
        assert!(report.markdown().contains("PESQ"));
        assert!(report.markdown().contains("ViSQOL"));
        assert!(report.markdown().contains("Musical-noise score"));
        let html = report.html();
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("Transient-loss score"));
        assert!(html.contains("Phase-distortion score"));
    }

    #[test]
    fn silent_comparison_metrics_stay_finite_and_json_safe() {
        let audio = mono(vec![0.0; 1600]);
        let report = ComparisonReport::compare(&audio, &audio, &audio).unwrap();
        for metrics in [&report.noisy, &report.enhanced] {
            assert!(metrics.si_sdr_db.is_finite());
            assert!(metrics.si_snr_db.is_finite());
            assert!(metrics.snr_db.is_finite());
            assert!(metrics.segmental_snr_db.is_finite());
        }
        let json = report.json();
        assert!(!json.contains("NaN"));
        assert!(!json.contains("inf"));
        assert!(json.contains("\"si_sdr_db\":-120.000000"));
    }

    fn mono(samples: Vec<f64>) -> Audio {
        Audio {
            sample_rate: 16_000,
            channels: vec![samples],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn stereo(left: Vec<f64>, right: Vec<f64>) -> Audio {
        Audio {
            sample_rate: 16_000,
            channels: vec![left, right],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        }
    }

    #[test]
    fn identical_audio_has_no_artifact_scores() {
        let signal = (0..16_384)
            .map(|index| {
                (2.0 * PI * 440.0 * index as f64 / 16_000.0).sin() * 0.4
                    + (2.0 * PI * 913.0 * index as f64 / 16_000.0).sin() * 0.1
            })
            .collect::<Vec<_>>();
        let audio = mono(signal);
        let report = ArtifactReport::compare(&audio, &audio).unwrap();
        assert!(report.musical_noise_score < 1e-12);
        assert!(report.pumping_score < 1e-12);
        assert!(report.transient_loss_score < 1e-12);
        assert_eq!(report.phase_distortion_score, None);
    }

    #[test]
    fn detects_frame_level_pumping() {
        let reference = (0..16_384)
            .map(|index| (2.0 * PI * 440.0 * index as f64 / 16_000.0).sin() * 0.5)
            .collect::<Vec<_>>();
        let test = reference
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let gain = if (index / 2_048) % 2 == 0 { 1.0 } else { 0.25 };
                sample * gain
            })
            .collect::<Vec<_>>();
        let report = ArtifactReport::compare(&mono(reference), &mono(test)).unwrap();
        assert!(
            report.pumping_score > 0.2,
            "pumping score: {}",
            report.pumping_score
        );
    }

    #[test]
    fn detects_transient_loss() {
        let mut reference = vec![0.0; 16_384];
        for index in (1_024..16_384).step_by(2_048) {
            reference[index] = 0.95;
            if index + 1 < reference.len() {
                reference[index + 1] = -0.7;
            }
        }
        let test = vec![0.0; reference.len()];
        let report = ArtifactReport::compare(&mono(reference), &mono(test)).unwrap();
        assert!(
            report.transient_loss_score > 0.2,
            "transient-loss score: {}",
            report.transient_loss_score
        );
    }

    #[test]
    fn detects_narrowband_musical_noise() {
        let reference = vec![0.0; 16_384];
        let test = (0..16_384)
            .map(|index| (2.0 * PI * 1_000.0 * index as f64 / 16_000.0).sin() * 0.5)
            .collect::<Vec<_>>();
        let report = ArtifactReport::compare(&mono(reference), &mono(test)).unwrap();
        assert!(
            report.musical_noise_score > 0.1,
            "musical-noise score: {}",
            report.musical_noise_score
        );
    }

    #[test]
    fn detects_stereo_phase_inversion() {
        let left = (0..16_384)
            .map(|index| (2.0 * PI * 440.0 * index as f64 / 16_000.0).sin() * 0.5)
            .collect::<Vec<_>>();
        let right = left.clone();
        let inverted = right.iter().map(|sample| -*sample).collect::<Vec<_>>();
        let report =
            ArtifactReport::compare(&stereo(left.clone(), right), &stereo(left, inverted)).unwrap();
        assert!(
            report.phase_distortion_score.unwrap_or(0.0) > 0.8,
            "phase-distortion score: {:?}",
            report.phase_distortion_score
        );
    }
}
