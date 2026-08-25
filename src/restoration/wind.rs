use super::{
    finite_confidence, mark_changed_samples, mark_range, median, milliseconds_to_samples, MaskCell,
    OperationOutcome, RestorationMode, RestorationOperation, RestorationOperationDetails,
    WindPlosiveConfig, MASK_DETECTED, MASK_PADDED,
};
use std::f64::consts::PI;

#[derive(Clone, Copy)]
struct Region {
    start: usize,
    end: usize,
    confidence: f64,
    coherence: Option<f64>,
}

pub(super) fn process(
    channels: &mut [Vec<f64>],
    sample_rate: u32,
    mode: RestorationMode,
    mask: &mut [Vec<MaskCell>],
    config: &WindPlosiveConfig,
) -> Result<OperationOutcome, String> {
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    let window = milliseconds_to_samples(config.window_ms, sample_rate, 8);
    let maximum_burst = milliseconds_to_samples(config.maximum_burst_ms, sample_rate, window);
    let mut warnings = Vec::new();
    if frames < window.saturating_mul(3) {
        warnings.push("input is too short for a bounded wind/plosive baseline".into());
        return Ok(empty_outcome(config, warnings));
    }
    let lowpassed: Vec<Vec<f64>> = channels
        .iter()
        .map(|channel| one_pole_lowpass(channel, sample_rate, config.low_band_hz))
        .collect();
    let mut all_regions = Vec::new();
    let mut rejected_regions = 0usize;
    for channel_index in 0..channels.len() {
        let candidates = detect_windows(channels, &lowpassed, channel_index, window, config);
        for region in merge_windows(candidates, window / 2, frames) {
            if region.end - region.start > maximum_burst {
                rejected_regions += 1;
            } else {
                all_regions.push((channel_index, region));
            }
        }
    }
    let snapshot = if mode == RestorationMode::Apply {
        Some(channels.to_vec())
    } else {
        None
    };
    let mut confidence_sum = 0.0;
    let mut detected_samples = 0usize;
    let mut coherence_sum = 0.0;
    let mut coherence_count = 0usize;
    for &(channel_index, region) in &all_regions {
        detected_samples += region.end - region.start;
        confidence_sum += region.confidence;
        if let Some(coherence) = region.coherence {
            coherence_sum += coherence;
            coherence_count += 1;
        }
        let fade = milliseconds_to_samples(5.0, sample_rate, 1)
            .min((region.end - region.start) / 4)
            .max(1);
        mark_range(
            mask,
            channel_index,
            region.start.saturating_sub(fade),
            region.start,
            MASK_PADDED,
            RestorationOperation::WindPlosive,
            region.confidence,
        );
        mark_range(
            mask,
            channel_index,
            region.start,
            region.end,
            MASK_DETECTED,
            RestorationOperation::WindPlosive,
            region.confidence,
        );
        mark_range(
            mask,
            channel_index,
            region.end,
            region.end.saturating_add(fade),
            MASK_PADDED,
            RestorationOperation::WindPlosive,
            region.confidence,
        );
        if mode == RestorationMode::Apply {
            attenuate_region(
                &mut channels[channel_index],
                &lowpassed[channel_index],
                region,
                fade,
                config.maximum_attenuation_db,
            );
        }
    }
    let mut changed_samples = 0usize;
    if mode == RestorationMode::Apply {
        for channel_index in 0..channels.len() {
            changed_samples += mark_changed_samples(
                &snapshot.as_ref().expect("apply mode has a snapshot")[channel_index],
                &channels[channel_index],
                mask,
                channel_index,
                RestorationOperation::WindPlosive,
                if all_regions.is_empty() {
                    0.0
                } else {
                    confidence_sum / all_regions.len() as f64
                },
            );
        }
    }
    if all_regions.is_empty() {
        warnings.push("no short low-frequency burst passed the wind/plosive gate".into());
    }
    if rejected_regions > 0 {
        warnings.push(format!(
            "{rejected_regions} low-frequency region(s) were left untouched because they exceeded the burst-duration limit"
        ));
    }
    Ok(OperationOutcome {
        detected_samples,
        changed_samples,
        confidence: if all_regions.is_empty() {
            0.0
        } else {
            finite_confidence(confidence_sum / all_regions.len() as f64)
        },
        warnings,
        details: RestorationOperationDetails::WindPlosive {
            regions: all_regions.len(),
            rejected_regions,
            low_band_hz: config.low_band_hz,
            maximum_attenuation_db: config.maximum_attenuation_db,
            stereo_coherence: (coherence_count > 0)
                .then_some(coherence_sum / coherence_count as f64),
        },
    })
}

fn empty_outcome(config: &WindPlosiveConfig, warnings: Vec<String>) -> OperationOutcome {
    OperationOutcome {
        detected_samples: 0,
        changed_samples: 0,
        confidence: 0.0,
        warnings,
        details: RestorationOperationDetails::WindPlosive {
            regions: 0,
            rejected_regions: 0,
            low_band_hz: config.low_band_hz,
            maximum_attenuation_db: config.maximum_attenuation_db,
            stereo_coherence: None,
        },
    }
}

fn detect_windows(
    channels: &[Vec<f64>],
    lowpassed: &[Vec<f64>],
    channel_index: usize,
    window: usize,
    config: &WindPlosiveConfig,
) -> Vec<Region> {
    let hop = (window / 2).max(1);
    let frames = channels[channel_index].len();
    let mut measurements = Vec::new();
    let mut start = 0usize;
    while start + window <= frames {
        let end = start + window;
        let mut low_energy = 0.0;
        let mut high_energy = 0.0;
        for index in start..end {
            let low = lowpassed[channel_index][index];
            let high = channels[channel_index][index] - low;
            low_energy += low * low;
            high_energy += high * high;
        }
        measurements.push((
            start,
            low_energy / window as f64,
            high_energy / window as f64,
        ));
        start += hop;
    }
    let mut baseline_values: Vec<f64> = measurements.iter().map(|(_, low, _)| *low).collect();
    let baseline = median(&mut baseline_values).max(1e-16);
    let mut candidates = Vec::new();
    for (window_index, &(start, low_energy, high_energy)) in measurements.iter().enumerate() {
        let ratio = low_energy / high_energy.max(1e-16);
        let burst = low_energy / baseline;
        let previous = window_index
            .checked_sub(1)
            .map(|index| measurements[index].1)
            .unwrap_or(baseline);
        let next = measurements
            .get(window_index + 1)
            .map(|measurement| measurement.1)
            .unwrap_or(baseline);
        let modulation = low_energy / previous.min(next).max(baseline * 0.5).max(1e-16);
        if ratio < config.ratio_threshold || burst < 2.5 || modulation < 1.35 {
            continue;
        }
        let coherence = if channels.len() > 1 {
            Some(window_coherence(
                lowpassed,
                channel_index,
                start,
                start + window,
            ))
        } else {
            None
        };
        let ratio_support = (ratio / config.ratio_threshold - 1.0).min(2.0) / 2.0;
        let burst_support = (burst / 2.5 - 1.0).min(2.0) / 2.0;
        let coherence_support = coherence
            .map(|value| (1.0 - value).clamp(0.0, 1.0))
            .unwrap_or(0.55);
        let confidence = finite_confidence(
            0.45 + 0.2 * ratio_support + 0.2 * burst_support + 0.15 * coherence_support,
        );
        if confidence >= config.minimum_confidence {
            candidates.push(Region {
                start,
                end: start + window,
                confidence,
                coherence,
            });
        }
    }
    candidates
}

fn merge_windows(mut windows: Vec<Region>, maximum_gap: usize, frames: usize) -> Vec<Region> {
    if windows.is_empty() {
        return windows;
    }
    windows.sort_by_key(|region| region.start);
    let mut merged = Vec::new();
    let mut current = windows[0];
    for region in windows.into_iter().skip(1) {
        if region.start <= current.end.saturating_add(maximum_gap) {
            current.end = current.end.max(region.end).min(frames);
            current.confidence = current.confidence.max(region.confidence);
            current.coherence = match (current.coherence, region.coherence) {
                (Some(left), Some(right)) => Some(0.5 * (left + right)),
                (left, right) => left.or(right),
            };
        } else {
            merged.push(current);
            current = region;
        }
    }
    merged.push(current);
    merged
}

fn attenuate_region(
    samples: &mut [f64],
    lowpassed: &[f64],
    region: Region,
    fade: usize,
    maximum_attenuation_db: f64,
) {
    let start = region.start.saturating_sub(fade);
    let end = region.end.saturating_add(fade).min(samples.len());
    let amount = (1.0 - 10.0f64.powf(-maximum_attenuation_db / 20.0)) * region.confidence;
    for index in start..end {
        let envelope = if index < region.start {
            raised_fade(index - start, region.start - start)
        } else if index >= region.end {
            raised_fade(end - index - 1, end - region.end)
        } else {
            1.0
        };
        let highpassed = samples[index] - lowpassed[index];
        let mix = (amount * envelope).clamp(0.0, 1.0);
        samples[index] = (samples[index] * (1.0 - mix) + highpassed * mix).clamp(-1.0, 1.0);
    }
}

fn raised_fade(index: usize, length: usize) -> f64 {
    if length <= 1 {
        1.0
    } else {
        0.5 - 0.5 * (PI * index as f64 / (length - 1) as f64).cos()
    }
}

fn one_pole_lowpass(samples: &[f64], sample_rate: u32, cutoff_hz: f64) -> Vec<f64> {
    let alpha = 1.0 - (-2.0 * PI * cutoff_hz / sample_rate as f64).exp();
    let mut output = Vec::with_capacity(samples.len());
    let mut state = samples.first().copied().unwrap_or(0.0);
    for sample in samples {
        state += alpha * (*sample - state);
        output.push(state);
    }
    output
}

fn window_coherence(lowpassed: &[Vec<f64>], channel_index: usize, start: usize, end: usize) -> f64 {
    let reference = &lowpassed[channel_index][start..end];
    let mut best = 0.0f64;
    for (other_index, other) in lowpassed.iter().enumerate() {
        if other_index == channel_index {
            continue;
        }
        let other = &other[start..end];
        let dot = reference
            .iter()
            .zip(other)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let left_energy = reference.iter().map(|sample| sample * sample).sum::<f64>();
        let right_energy = other.iter().map(|sample| sample * sample).sum::<f64>();
        let coherence = dot.abs() / (left_energy * right_energy).sqrt().max(1e-16);
        best = best.max(coherence.clamp(0.0, 1.0));
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_bass_is_not_a_burst() {
        let rate = 48_000u32;
        let samples: Vec<f64> = (0..rate as usize)
            .map(|index| (2.0 * PI * 80.0 * index as f64 / rate as f64).sin() * 0.2)
            .collect();
        let channels = vec![samples];
        let lowpassed = vec![one_pole_lowpass(&channels[0], rate, 180.0)];
        let candidates =
            detect_windows(&channels, &lowpassed, 0, 960, &WindPlosiveConfig::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn isolated_low_frequency_burst_is_detected() {
        let rate = 48_000u32;
        let mut samples = vec![0.001; rate as usize];
        for index in 20_000..22_000 {
            samples[index] += (2.0 * PI * 60.0 * index as f64 / rate as f64).sin() * 0.7;
        }
        let channels = vec![samples];
        let lowpassed = vec![one_pole_lowpass(&channels[0], rate, 180.0)];
        let candidates =
            detect_windows(&channels, &lowpassed, 0, 960, &WindPlosiveConfig::default());
        assert!(!candidates.is_empty());
    }
}
