use super::{
    finite_confidence, mark_changed_samples, mark_range, median, milliseconds_to_samples,
    DeclipConfig, MaskCell, OperationOutcome, RestorationMode, RestorationOperation,
    RestorationOperationDetails, MASK_DETECTED, MASK_PADDED,
};
use crate::fft::{Complex, Fft};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClipPolarity {
    Positive,
    Negative,
}

#[derive(Clone, Copy)]
struct ClipRegion {
    start: usize,
    end: usize,
    polarity: ClipPolarity,
    threshold: f64,
    confidence: f64,
}

pub(super) fn process(
    channels: &mut [Vec<f64>],
    sample_rate: u32,
    mode: RestorationMode,
    mask: &mut [Vec<MaskCell>],
    config: &DeclipConfig,
) -> Result<OperationOutcome, String> {
    let maximum_region = milliseconds_to_samples(config.maximum_region_ms, sample_rate, 1);
    let context = milliseconds_to_samples(config.context_ms, sample_rate, 8).min(16_384);
    let mut accepted_regions = 0usize;
    let mut rejected_regions = 0usize;
    let mut detected_samples = 0usize;
    let mut converged_regions = 0usize;
    let mut confidence_sum = 0.0;
    let mut changed_samples = 0usize;
    let mut positive_thresholds = Vec::new();
    let mut negative_thresholds = Vec::new();
    let mut warnings = Vec::new();

    for channel_index in 0..channels.len() {
        let snapshot = channels[channel_index].clone();
        let detection = detect_regions(&snapshot, config);
        rejected_regions += detection.rejected;
        for region in detection.regions {
            if region.end - region.start > maximum_region
                || region.start < 2
                || region.end.saturating_add(2) > snapshot.len()
            {
                rejected_regions += 1;
                continue;
            }
            accepted_regions += 1;
            detected_samples += region.end - region.start;
            confidence_sum += region.confidence;
            match region.polarity {
                ClipPolarity::Positive => positive_thresholds.push(region.threshold),
                ClipPolarity::Negative => negative_thresholds.push(region.threshold),
            }
            let padding = context.min(region.start).min(snapshot.len() - region.end);
            mark_range(
                mask,
                channel_index,
                region.start.saturating_sub(padding),
                region.start,
                MASK_PADDED,
                RestorationOperation::Declip,
                region.confidence,
            );
            mark_range(
                mask,
                channel_index,
                region.start,
                region.end,
                MASK_DETECTED,
                RestorationOperation::Declip,
                region.confidence,
            );
            mark_range(
                mask,
                channel_index,
                region.end,
                region.end.saturating_add(padding),
                MASK_PADDED,
                RestorationOperation::Declip,
                region.confidence,
            );
            if mode == RestorationMode::Apply
                && reconstruct_region(
                    &mut channels[channel_index],
                    region,
                    context,
                    config.iterations,
                )?
            {
                converged_regions += 1;
            }
        }
        if mode == RestorationMode::Apply {
            changed_samples += mark_changed_samples(
                &snapshot,
                &channels[channel_index],
                mask,
                channel_index,
                RestorationOperation::Declip,
                if accepted_regions == 0 {
                    0.0
                } else {
                    confidence_sum / accepted_regions as f64
                },
            );
        }
    }
    if accepted_regions == 0 {
        warnings.push("no flat-top region passed the clipping confidence gate".into());
    }
    if rejected_regions > 0 {
        warnings.push(format!(
            "{rejected_regions} possible clipped region(s) were left untouched as ambiguous or overlong"
        ));
    }
    if mode == RestorationMode::Apply && accepted_regions > converged_regions {
        warnings.push(format!(
            "{} declipping region(s) reached the iteration cap; hard sample constraints were still preserved",
            accepted_regions - converged_regions
        ));
    }
    Ok(OperationOutcome {
        detected_samples,
        changed_samples,
        confidence: if accepted_regions == 0 {
            0.0
        } else {
            finite_confidence(confidence_sum / accepted_regions as f64)
        },
        warnings,
        details: RestorationOperationDetails::Declip {
            regions: accepted_regions,
            rejected_regions,
            positive_threshold: threshold_summary(&mut positive_thresholds),
            negative_threshold: threshold_summary(&mut negative_thresholds),
            iterations: config.iterations,
            converged_regions,
        },
    })
}

struct Detection {
    regions: Vec<ClipRegion>,
    rejected: usize,
}

fn detect_regions(samples: &[f64], config: &DeclipConfig) -> Detection {
    if samples.len() < config.minimum_run_samples.saturating_add(4) {
        return Detection {
            regions: Vec::new(),
            rejected: 0,
        };
    }
    let positive = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let negative = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let mut raw = Vec::new();
    if positive >= 0.1 {
        collect_plateaus(samples, ClipPolarity::Positive, positive, config, &mut raw);
    }
    if negative <= -0.1 {
        collect_plateaus(samples, ClipPolarity::Negative, negative, config, &mut raw);
    }
    raw.sort_by_key(|region| region.start);
    let clipped_samples: usize = raw.iter().map(|region| region.end - region.start).sum();
    if clipped_samples.saturating_mul(5) > samples.len() {
        return Detection {
            regions: Vec::new(),
            rejected: raw.len(),
        };
    }
    let mut regions = Vec::new();
    let mut rejected = 0usize;
    for region in raw {
        let left_slope = samples[region.start] - samples[region.start - 1];
        let right_slope = samples[region.end] - samples[region.end - 1];
        let approaches_plateau = match region.polarity {
            ClipPolarity::Positive => {
                left_slope >= -config.threshold_tolerance
                    && right_slope <= config.threshold_tolerance
            }
            ClipPolarity::Negative => {
                left_slope <= config.threshold_tolerance
                    && right_slope >= -config.threshold_tolerance
            }
        };
        if approaches_plateau && region.confidence >= config.minimum_confidence {
            regions.push(region);
        } else {
            rejected += 1;
        }
    }
    Detection { regions, rejected }
}

fn collect_plateaus(
    samples: &[f64],
    polarity: ClipPolarity,
    threshold: f64,
    config: &DeclipConfig,
    regions: &mut Vec<ClipRegion>,
) {
    let tolerance = config
        .threshold_tolerance
        .max(threshold.abs() * config.threshold_tolerance);
    let is_plateau = |sample: f64| match polarity {
        ClipPolarity::Positive => (threshold - sample).abs() <= tolerance,
        ClipPolarity::Negative => (sample - threshold).abs() <= tolerance,
    };
    let mut index = 1usize;
    while index + 1 < samples.len() {
        if !is_plateau(samples[index]) {
            index += 1;
            continue;
        }
        let start = index;
        while index < samples.len() && is_plateau(samples[index]) {
            index += 1;
        }
        let end = index;
        if end - start < config.minimum_run_samples || end >= samples.len() {
            continue;
        }
        let plateau_span = samples[start..end]
            .iter()
            .copied()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
                (min.min(value), max.max(value))
            });
        let flatness = 1.0 - ((plateau_span.1 - plateau_span.0) / tolerance.max(1e-12)).min(1.0);
        let length_support =
            ((end - start) as f64 / (config.minimum_run_samples * 2) as f64).clamp(0.0, 1.0);
        let confidence = finite_confidence(0.55 + 0.25 * flatness + 0.2 * length_support);
        regions.push(ClipRegion {
            start,
            end,
            polarity,
            threshold,
            confidence,
        });
    }
}

fn reconstruct_region(
    samples: &mut [f64],
    region: ClipRegion,
    context: usize,
    iterations: usize,
) -> Result<bool, String> {
    let segment_start = region.start.saturating_sub(context);
    let segment_end = region.end.saturating_add(context).min(samples.len());
    let source = samples[segment_start..segment_end].to_vec();
    let gap_start = region.start - segment_start;
    let gap_end = region.end - segment_start;
    let logical_length = source.len();
    let fft_size = logical_length.next_power_of_two().max(8);
    if fft_size > 65_536 {
        return Err("declipping analysis window exceeds the bounded 65536-sample FFT".into());
    }
    let mut working = source.clone();
    initialize_gap(&mut working, gap_start, gap_end, region);
    let mut spectrum = vec![Complex::default(); fft_size];
    let fft = Fft::new(fft_size);
    let mut previous = working[gap_start..gap_end].to_vec();
    let mut converged = false;
    for iteration in 0..iterations {
        for index in 0..fft_size {
            let value = if index < working.len() {
                working[index]
            } else {
                working[reflect_index(index, working.len())]
            };
            spectrum[index] = Complex::new(value, 0.0);
        }
        fft.forward(&mut spectrum);
        let mut magnitudes: Vec<f64> = spectrum
            .iter()
            .take(fft_size / 2 + 1)
            .map(|value| value.re.hypot(value.im))
            .collect();
        magnitudes.sort_by(f64::total_cmp);
        let progress = (iteration + 1) as f64 / iterations as f64;
        let retained_fraction = 0.15 + 0.75 * progress;
        let cutoff_index = ((1.0 - retained_fraction) * magnitudes.len() as f64) as usize;
        let cutoff = magnitudes[cutoff_index.min(magnitudes.len() - 1)];
        for coefficient in &mut spectrum {
            if coefficient.re.hypot(coefficient.im) < cutoff {
                *coefficient = Complex::default();
            }
        }
        fft.inverse(&mut spectrum);
        for index in 0..logical_length {
            if index < gap_start || index >= gap_end {
                working[index] = source[index];
            } else {
                working[index] = project_clipped(spectrum[index].re, region).clamp(-1.0, 1.0);
            }
        }
        let mut delta = 0.0f64;
        for (old, new) in previous.iter_mut().zip(&working[gap_start..gap_end]) {
            delta = delta.max((*old - *new).abs());
            *old = *new;
        }
        if delta < 1e-7 {
            converged = true;
            break;
        }
    }
    samples[region.start..region.end].copy_from_slice(&working[gap_start..gap_end]);
    Ok(converged)
}

fn initialize_gap(working: &mut [f64], start: usize, end: usize, region: ClipRegion) {
    let left = working[start - 1];
    let right = working[end];
    let headroom = (1.0 - region.threshold.abs()).min(0.12).max(0.005);
    let gap = end - start;
    for offset in 0..gap {
        let position = (offset + 1) as f64 / (gap + 1) as f64;
        let linear = left * (1.0 - position) + right * position;
        let arch = (PI * position).sin() * headroom;
        working[start + offset] = match region.polarity {
            ClipPolarity::Positive => linear.max(region.threshold + arch).clamp(-1.0, 1.0),
            ClipPolarity::Negative => linear.min(region.threshold - arch).clamp(-1.0, 1.0),
        };
    }
}

fn project_clipped(value: f64, region: ClipRegion) -> f64 {
    match region.polarity {
        ClipPolarity::Positive => value.max(region.threshold),
        ClipPolarity::Negative => value.min(region.threshold),
    }
}

fn reflect_index(mut index: usize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    while index >= length {
        index = 2 * length - index - 2;
    }
    index
}

fn threshold_summary(values: &mut [f64]) -> Option<f64> {
    (!values.is_empty()).then(|| median(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_flat_top_is_found() {
        let mut samples: Vec<f64> = (0..512)
            .map(|index| ((index as f64 * 0.04).sin() * 0.8).min(0.7))
            .collect();
        for sample in &mut samples[120..124] {
            *sample = 0.7;
        }
        let detection = detect_regions(&samples, &DeclipConfig::default());
        assert!(detection
            .regions
            .iter()
            .any(|region| region.start <= 120 && region.end >= 124));
    }

    #[test]
    fn reconstruction_changes_only_unknown_samples_and_obeys_inequality() {
        let mut samples: Vec<f64> = (0..256)
            .map(|index| (index as f64 * 0.03).sin() * 0.8)
            .collect();
        for sample in &mut samples[100..104] {
            *sample = 0.5;
        }
        let before = samples.clone();
        let region = ClipRegion {
            start: 100,
            end: 104,
            polarity: ClipPolarity::Positive,
            threshold: 0.5,
            confidence: 0.9,
        };
        reconstruct_region(&mut samples, region, 32, 12).unwrap();
        assert_eq!(&samples[..100], &before[..100]);
        assert_eq!(&samples[104..], &before[104..]);
        assert!(samples[100..104].iter().all(|sample| *sample >= 0.5));
        assert!(samples[100..104]
            .iter()
            .zip(&before[100..104])
            .any(|(after, before)| after.to_bits() != before.to_bits()));
    }

    #[test]
    fn sustained_square_wave_is_rejected_as_ambiguous() {
        let samples: Vec<f64> = (0..2_000)
            .map(|index| if (index / 20) % 2 == 0 { 0.7 } else { -0.7 })
            .collect();
        let detection = detect_regions(&samples, &DeclipConfig::default());
        assert!(detection.regions.is_empty());
        assert!(detection.rejected > 0);
    }
}
