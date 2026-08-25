use super::{
    finite_confidence, mark_changed_samples, mark_range, MaskCell, OperationOutcome,
    RestorationMode, RestorationOperation, RestorationOperationDetails, WpeChannelMode, WpeConfig,
    MASK_DETECTED,
};
use crate::fft::Complex;
use crate::stft::{Stft, StftConfig};
use crate::window::{WindowParams, WindowType};

const MAX_MULTICHANNEL_CHANNELS: usize = 4;

pub(super) fn process(
    channels: &mut [Vec<f64>],
    _sample_rate: u32,
    mode: RestorationMode,
    mask: &mut [Vec<MaskCell>],
    config: &WpeConfig,
) -> Result<OperationOutcome, String> {
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    let mut warnings = Vec::new();
    if config.channel_mode == WpeChannelMode::Multichannel
        && channels.len() > MAX_MULTICHANNEL_CHANNELS
    {
        return Err(format!(
            "multichannel WPE supports at most {MAX_MULTICHANNEL_CHANNELS} channels; select independent mode"
        ));
    }
    if frames < config.frame_size {
        warnings.push("input is shorter than one WPE analysis frame".into());
        return Ok(empty_outcome(config, warnings));
    }
    let stft = Stft::try_new(StftConfig {
        frame_size: config.frame_size,
        hop: config.hop_size,
        window: WindowType::Hann,
        window_params: WindowParams::default(),
    })
    .map_err(|error| format!("construct WPE STFT: {error}"))?;
    let analysis = analyze(channels, &stft)?;
    let first_usable = config
        .prediction_delay_frames
        .saturating_add(config.prediction_taps)
        .saturating_sub(1);
    if analysis.frame_count <= first_usable + 1 {
        warnings.push("input has insufficient WPE frames for the selected delay and taps".into());
        return Ok(empty_outcome(config, warnings));
    }

    let mut enhanced = analysis.spectra.clone();
    let mut solved_bins = 0usize;
    let mut ill_conditioned_bins = 0usize;
    let mut convergence_sum = 0.0;
    for target_channel in 0..channels.len() {
        let sources: Vec<usize> = match config.channel_mode {
            WpeChannelMode::Independent => vec![target_channel],
            WpeChannelMode::Multichannel => (0..channels.len()).collect(),
        };
        for bin in 1..analysis.bin_count.saturating_sub(1) {
            match enhance_bin(
                &analysis.spectra,
                target_channel,
                &sources,
                bin,
                analysis.frame_count,
                analysis.bin_count,
                config,
            ) {
                Some((values, convergence)) => {
                    for (frame, value) in values.into_iter().enumerate() {
                        enhanced[target_channel][frame * analysis.bin_count + bin] = value;
                    }
                    solved_bins += 1;
                    convergence_sum += convergence;
                }
                None => ill_conditioned_bins += 1,
            }
        }
    }
    let candidate = synthesize(
        &enhanced,
        channels.len(),
        frames,
        analysis.frame_count,
        analysis.padded_length,
        &stft,
    )?;
    let input_energy = channels
        .iter()
        .flatten()
        .map(|sample| sample * sample)
        .sum::<f64>();
    let correction_energy = channels
        .iter()
        .flatten()
        .zip(candidate.iter().flatten())
        .map(|(before, after)| (before - after).powi(2))
        .sum::<f64>();
    let solve_ratio =
        solved_bins as f64 / (solved_bins.saturating_add(ill_conditioned_bins).max(1)) as f64;
    let prediction_support = if input_energy <= 1e-20 {
        0.0
    } else {
        (correction_energy / input_energy).sqrt()
    };
    let confidence = finite_confidence(prediction_support * 2.0) * solve_ratio;
    if confidence < config.minimum_confidence {
        warnings.push(format!(
            "late-prediction confidence {:.3} is below the {:.3} repair gate",
            confidence, config.minimum_confidence
        ));
        return Ok(OperationOutcome {
            detected_samples: 0,
            changed_samples: 0,
            confidence,
            warnings,
            details: details(config, solved_bins, ill_conditioned_bins, convergence_sum),
        });
    }

    let mut detected_samples = 0usize;
    for channel_index in 0..channels.len() {
        let mut index = 0usize;
        while index < frames {
            if (channels[channel_index][index] - candidate[channel_index][index]).abs() <= 1e-10 {
                index += 1;
                continue;
            }
            let start = index;
            while index < frames
                && (channels[channel_index][index] - candidate[channel_index][index]).abs() > 1e-10
            {
                index += 1;
            }
            mark_range(
                mask,
                channel_index,
                start,
                index,
                MASK_DETECTED,
                RestorationOperation::Dereverb,
                confidence,
            );
            detected_samples += index - start;
        }
    }
    let mut changed_samples = 0usize;
    if mode == RestorationMode::Apply {
        for channel_index in 0..channels.len() {
            let before = channels[channel_index].clone();
            channels[channel_index].copy_from_slice(&candidate[channel_index]);
            changed_samples += mark_changed_samples(
                &before,
                &channels[channel_index],
                mask,
                channel_index,
                RestorationOperation::Dereverb,
                confidence,
            );
        }
    }
    if ill_conditioned_bins > 0 {
        warnings.push(format!(
            "{ill_conditioned_bins} WPE frequency-bin solve(s) were bypassed as ill-conditioned"
        ));
    }
    Ok(OperationOutcome {
        detected_samples,
        changed_samples,
        confidence,
        warnings,
        details: details(config, solved_bins, ill_conditioned_bins, convergence_sum),
    })
}

fn empty_outcome(config: &WpeConfig, warnings: Vec<String>) -> OperationOutcome {
    OperationOutcome {
        detected_samples: 0,
        changed_samples: 0,
        confidence: 0.0,
        warnings,
        details: details(config, 0, 0, 0.0),
    }
}

fn details(
    config: &WpeConfig,
    solved_bins: usize,
    ill_conditioned_bins: usize,
    convergence_sum: f64,
) -> RestorationOperationDetails {
    RestorationOperationDetails::Dereverb {
        channel_mode: config.channel_mode,
        frame_size: config.frame_size,
        hop_size: config.hop_size,
        prediction_delay_frames: config.prediction_delay_frames,
        prediction_taps: config.prediction_taps,
        effective_context_frames: (config.prediction_delay_frames + config.prediction_taps)
            * config.hop_size,
        iterations: config.iterations,
        solved_bins,
        ill_conditioned_bins,
        convergence: if solved_bins == 0 {
            0.0
        } else {
            (convergence_sum / solved_bins as f64).clamp(0.0, 1.0)
        },
    }
}

struct Analysis {
    spectra: Vec<Vec<Complex>>,
    frame_count: usize,
    bin_count: usize,
    padded_length: usize,
}

fn analyze(channels: &[Vec<f64>], stft: &Stft) -> Result<Analysis, String> {
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    let pad = stft.frame_size() / 2;
    let requested = frames.saturating_add(2 * pad).max(stft.frame_size());
    let remainder = requested.saturating_sub(stft.frame_size()) % stft.hop();
    let padded_length = if remainder == 0 {
        requested
    } else {
        requested + (stft.hop() - remainder)
    };
    let frame_count = 1 + (padded_length - stft.frame_size()) / stft.hop();
    let bin_count = stft.nbins();
    let mut spectra = Vec::with_capacity(channels.len());
    let mut time = vec![0.0; stft.frame_size()];
    let mut spectrum = vec![Complex::default(); stft.frame_size()];
    for channel in channels {
        let mut channel_spectrum = Vec::new();
        channel_spectrum
            .try_reserve_exact(frame_count.saturating_mul(bin_count))
            .map_err(|_| "unable to reserve WPE spectra".to_string())?;
        for frame in 0..frame_count {
            let start = frame * stft.hop();
            for offset in 0..stft.frame_size() {
                let logical = start as isize + offset as isize - pad as isize;
                time[offset] = channel[reflect_signed(logical, channel.len())];
            }
            stft.analyze(&time, &mut spectrum);
            channel_spectrum.extend_from_slice(&spectrum[..bin_count]);
        }
        spectra.push(channel_spectrum);
    }
    Ok(Analysis {
        spectra,
        frame_count,
        bin_count,
        padded_length,
    })
}

fn synthesize(
    spectra: &[Vec<Complex>],
    channels: usize,
    output_frames: usize,
    frame_count: usize,
    padded_length: usize,
    stft: &Stft,
) -> Result<Vec<Vec<f64>>, String> {
    let pad = stft.frame_size() / 2;
    let bins = stft.nbins();
    let mut output = Vec::with_capacity(channels);
    for channel_spectrum in spectra.iter().take(channels) {
        let mut padded = vec![0.0; padded_length];
        let mut norm = vec![0.0; padded_length];
        let mut spectrum = vec![Complex::default(); stft.frame_size()];
        for frame in 0..frame_count {
            for bin in 0..bins {
                spectrum[bin] = channel_spectrum[frame * bins + bin];
            }
            for bin in bins..stft.frame_size() {
                spectrum[bin] = complex_conjugate(spectrum[stft.frame_size() - bin]);
            }
            stft.synthesize(&mut spectrum, &mut padded, &mut norm, frame * stft.hop());
        }
        for (sample, weight) in padded.iter_mut().zip(norm) {
            if weight > 1e-12 {
                *sample /= weight;
            }
        }
        let end = pad.saturating_add(output_frames);
        if end > padded.len() {
            return Err("WPE synthesis crop exceeds the padded signal".into());
        }
        let rendered: Vec<f64> = padded[pad..end]
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0))
            .collect();
        if rendered.iter().any(|sample| !sample.is_finite()) {
            return Err("WPE synthesis produced a non-finite sample".into());
        }
        output.push(rendered);
    }
    Ok(output)
}

fn enhance_bin(
    spectra: &[Vec<Complex>],
    target_channel: usize,
    source_channels: &[usize],
    bin: usize,
    frame_count: usize,
    bin_count: usize,
    config: &WpeConfig,
) -> Option<(Vec<Complex>, f64)> {
    let original: Vec<Complex> = (0..frame_count)
        .map(|frame| spectra[target_channel][frame * bin_count + bin])
        .collect();
    let mut enhanced = original.clone();
    let dimension = source_channels.len().checked_mul(config.prediction_taps)?;
    let first = config.prediction_delay_frames + config.prediction_taps - 1;
    let mut convergence = 1.0;
    for _ in 0..config.iterations {
        let average_power = enhanced
            .iter()
            .map(|value| complex_norm_squared(*value))
            .sum::<f64>()
            / frame_count as f64;
        let floor = average_power * 1e-6 + 1e-18;
        let mut matrix = vec![vec![Complex::default(); dimension]; dimension];
        let mut right = vec![Complex::default(); dimension];
        let mut predictor = vec![Complex::default(); dimension];
        for frame in first..frame_count {
            fill_predictor(
                &mut predictor,
                spectra,
                source_channels,
                frame,
                bin,
                bin_count,
                config,
            );
            let weight = 1.0 / complex_norm_squared(enhanced[frame]).max(floor);
            for row in 0..dimension {
                right[row] = complex_add(
                    right[row],
                    complex_scale(
                        complex_mul(predictor[row], complex_conjugate(original[frame])),
                        weight,
                    ),
                );
                for column in 0..dimension {
                    matrix[row][column] = complex_add(
                        matrix[row][column],
                        complex_scale(
                            complex_mul(predictor[row], complex_conjugate(predictor[column])),
                            weight,
                        ),
                    );
                }
            }
        }
        let trace = (0..dimension)
            .map(|index| matrix[index][index].re.abs())
            .sum::<f64>()
            / dimension as f64;
        let diagonal = config.regularization * trace.max(1e-18);
        for index in 0..dimension {
            matrix[index][index].re += diagonal;
        }
        let filter = solve_complex(matrix, right)?;
        let mut next = original.clone();
        let minimum_ratio = 10.0f64.powf(-config.maximum_attenuation_db / 20.0);
        let mut change_energy = 0.0;
        let mut base_energy = 0.0;
        for frame in first..frame_count {
            fill_predictor(
                &mut predictor,
                spectra,
                source_channels,
                frame,
                bin,
                bin_count,
                config,
            );
            let prediction = filter.iter().zip(&predictor).fold(
                Complex::default(),
                |sum, (coefficient, value)| {
                    complex_add(sum, complex_mul(complex_conjugate(*coefficient), *value))
                },
            );
            let mut residual = complex_sub(original[frame], prediction);
            let original_magnitude = complex_norm_squared(original[frame]).sqrt();
            let residual_magnitude = complex_norm_squared(residual).sqrt();
            let minimum_magnitude = original_magnitude * minimum_ratio;
            if residual_magnitude < minimum_magnitude {
                residual = if residual_magnitude > 1e-18 {
                    complex_scale(residual, minimum_magnitude / residual_magnitude)
                } else {
                    complex_scale(original[frame], minimum_ratio)
                };
            }
            change_energy += complex_norm_squared(complex_sub(residual, enhanced[frame]));
            base_energy += complex_norm_squared(enhanced[frame]);
            next[frame] = residual;
        }
        convergence = (change_energy / (base_energy + 1e-18)).sqrt().min(1.0);
        enhanced = next;
    }
    if enhanced
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        None
    } else {
        Some((enhanced, convergence))
    }
}

fn fill_predictor(
    target: &mut [Complex],
    spectra: &[Vec<Complex>],
    source_channels: &[usize],
    frame: usize,
    bin: usize,
    bin_count: usize,
    config: &WpeConfig,
) {
    let mut index = 0;
    for source in source_channels {
        for tap in 0..config.prediction_taps {
            let source_frame = frame - config.prediction_delay_frames - tap;
            target[index] = spectra[*source][source_frame * bin_count + bin];
            index += 1;
        }
    }
}

fn solve_complex(mut matrix: Vec<Vec<Complex>>, mut right: Vec<Complex>) -> Option<Vec<Complex>> {
    let dimension = right.len();
    for column in 0..dimension {
        let pivot = (column..dimension).max_by(|left, right_index| {
            complex_norm_squared(matrix[*left][column])
                .total_cmp(&complex_norm_squared(matrix[*right_index][column]))
        })?;
        if complex_norm_squared(matrix[pivot][column]) <= 1e-24 {
            return None;
        }
        matrix.swap(column, pivot);
        right.swap(column, pivot);
        let divisor = matrix[column][column];
        for entry in &mut matrix[column][column..] {
            *entry = complex_div(*entry, divisor)?;
        }
        right[column] = complex_div(right[column], divisor)?;
        for row in 0..dimension {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for target_column in column..dimension {
                matrix[row][target_column] = complex_sub(
                    matrix[row][target_column],
                    complex_mul(factor, matrix[column][target_column]),
                );
            }
            right[row] = complex_sub(right[row], complex_mul(factor, right[column]));
        }
    }
    Some(right)
}

fn reflect_signed(mut index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let last = length as isize - 1;
    while index < 0 || index > last {
        if index < 0 {
            index = -index;
        }
        if index > last {
            index = 2 * last - index;
        }
    }
    index as usize
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

fn complex_scale(value: Complex, scale: f64) -> Complex {
    Complex::new(value.re * scale, value.im * scale)
}

fn complex_conjugate(value: Complex) -> Complex {
    Complex::new(value.re, -value.im)
}

fn complex_norm_squared(value: Complex) -> f64 {
    value.re * value.re + value.im * value.im
}

fn complex_div(numerator: Complex, denominator: Complex) -> Option<Complex> {
    let norm = complex_norm_squared(denominator);
    if norm <= 1e-30 || !norm.is_finite() {
        return None;
    }
    Some(Complex::new(
        (numerator.re * denominator.re + numerator.im * denominator.im) / norm,
        (numerator.im * denominator.re - numerator.re * denominator.im) / norm,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_solver_recovers_known_solution() {
        let matrix = vec![
            vec![Complex::new(3.0, 0.0), Complex::new(1.0, -1.0)],
            vec![Complex::new(1.0, 1.0), Complex::new(4.0, 0.0)],
        ];
        let expected = [Complex::new(0.5, 0.25), Complex::new(-0.2, 0.1)];
        let right = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(expected)
                    .fold(Complex::default(), |sum, (coefficient, value)| {
                        complex_add(sum, complex_mul(*coefficient, value))
                    })
            })
            .collect();
        let solution = solve_complex(matrix, right).unwrap();
        for (actual, expected) in solution.iter().zip(expected) {
            assert!((actual.re - expected.re).abs() < 1e-10);
            assert!((actual.im - expected.im).abs() < 1e-10);
        }
    }

    #[test]
    fn stft_roundtrip_preserves_duration_and_signal() {
        let stft = Stft::try_new(StftConfig {
            frame_size: 256,
            hop: 64,
            window: WindowType::Hann,
            window_params: WindowParams::default(),
        })
        .unwrap();
        let channels = vec![(0..3_117)
            .map(|index| (index as f64 * 0.02).sin())
            .collect()];
        let analysis = analyze(&channels, &stft).unwrap();
        let output = synthesize(
            &analysis.spectra,
            1,
            channels[0].len(),
            analysis.frame_count,
            analysis.padded_length,
            &stft,
        )
        .unwrap();
        assert_eq!(output[0].len(), channels[0].len());
        let error = output[0]
            .iter()
            .zip(&channels[0])
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        assert!(error < 1e-8, "roundtrip error {error}");
    }
}
