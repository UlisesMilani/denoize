use super::{
    finite_confidence, mark_changed_samples, mark_range, median, median_absolute_deviation,
    milliseconds_to_samples, DeclickConfig, MaskCell, OperationOutcome, RestorationMode,
    RestorationOperation, RestorationOperationDetails, MASK_DETECTED, MASK_PADDED,
};

#[derive(Clone, Copy)]
struct Region {
    start: usize,
    end: usize,
    confidence: f64,
}

pub(super) fn process(
    channels: &mut [Vec<f64>],
    sample_rate: u32,
    mode: RestorationMode,
    mask: &mut [Vec<MaskCell>],
    config: &DeclickConfig,
) -> Result<OperationOutcome, String> {
    let maximum_gap = milliseconds_to_samples(config.maximum_gap_ms, sample_rate, 1);
    let merge_gap = milliseconds_to_samples(config.merge_gap_ms, sample_rate, 0);
    let context =
        milliseconds_to_samples(config.context_ms, sample_rate, config.prediction_order + 2);
    let mut accepted_regions = 0usize;
    let mut rejected_regions = 0usize;
    let mut detected_samples = 0usize;
    let mut confidence_sum = 0.0;
    let mut changed_samples = 0usize;
    let mut warnings = Vec::new();

    for channel_index in 0..channels.len() {
        let snapshot = channels[channel_index].clone();
        let candidates = detect_candidates(&snapshot, config);
        let regions = merge_candidates(&candidates, merge_gap);
        for region in regions {
            if region.end - region.start > maximum_gap
                || region.start < context
                || region.end.saturating_add(context) > snapshot.len()
            {
                rejected_regions += 1;
                continue;
            }
            accepted_regions += 1;
            detected_samples += region.end - region.start;
            confidence_sum += region.confidence;
            let padding = config
                .prediction_order
                .min(region.start)
                .min(snapshot.len() - region.end);
            mark_range(
                mask,
                channel_index,
                region.start.saturating_sub(padding),
                region.start,
                MASK_PADDED,
                RestorationOperation::Declick,
                region.confidence,
            );
            mark_range(
                mask,
                channel_index,
                region.start,
                region.end,
                MASK_DETECTED,
                RestorationOperation::Declick,
                region.confidence,
            );
            mark_range(
                mask,
                channel_index,
                region.end,
                region.end.saturating_add(padding),
                MASK_PADDED,
                RestorationOperation::Declick,
                region.confidence,
            );
            if mode == RestorationMode::Apply {
                interpolate_region(
                    &mut channels[channel_index],
                    region.start,
                    region.end,
                    context,
                    config.prediction_order,
                )?;
            }
        }
        if mode == RestorationMode::Apply {
            changed_samples += mark_changed_samples(
                &snapshot,
                &channels[channel_index],
                mask,
                channel_index,
                RestorationOperation::Declick,
                if accepted_regions == 0 {
                    0.0
                } else {
                    confidence_sum / accepted_regions as f64
                },
            );
        }
    }
    if accepted_regions == 0 {
        warnings.push("no short prediction-residual outlier passed the click gate".into());
    }
    if rejected_regions > 0 {
        warnings.push(format!(
            "{rejected_regions} click region(s) were left untouched because context or duration limits failed"
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
        details: RestorationOperationDetails::Declick {
            regions: accepted_regions,
            rejected_regions,
            prediction_order: config.prediction_order,
            maximum_gap_samples: maximum_gap,
        },
    })
}

fn detect_candidates(samples: &[f64], config: &DeclickConfig) -> Vec<(usize, f64)> {
    if samples.len() < config.prediction_order.saturating_mul(2).saturating_add(8) {
        return Vec::new();
    }
    let warped = warped_signal(samples, 0.35);
    let coefficients = fit_ar(&warped, config.prediction_order);
    let mut residual = vec![0.0; samples.len()];
    for index in config.prediction_order..warped.len() {
        residual[index] = warped[index] - predict(&warped[..index], &coefficients);
    }
    let valid = &residual[config.prediction_order..];
    let mut centers = valid.to_vec();
    let center = median(&mut centers);
    let mad = median_absolute_deviation(valid).max(1e-12);
    let robust_sigma = 1.4826 * mad;
    let signal_rms =
        (samples.iter().map(|sample| sample * sample).sum::<f64>() / samples.len() as f64).sqrt();
    let absolute_floor = (signal_rms * 0.015).max(1e-7);
    let threshold = config.residual_threshold_mad * robust_sigma;
    let mut candidates = Vec::new();
    for index in config.prediction_order..samples.len().saturating_sub(2) {
        let deviation = (residual[index] - center).abs();
        if deviation <= threshold.max(absolute_floor) {
            continue;
        }
        let local_peak = deviation >= (residual[index - 1] - center).abs()
            && deviation >= (residual[index + 1] - center).abs();
        let reversal = residual[index] * residual[index + 1] < 0.0
            || residual[index] * residual[index - 1] < 0.0;
        let isolated = (residual[index - 1] - center)
            .abs()
            .max((residual[index + 1] - center).abs())
            < deviation * 0.35;
        if !local_peak || (!reversal && !isolated) {
            continue;
        }
        let normalized = deviation / threshold.max(absolute_floor);
        let confidence = finite_confidence(0.55 + 0.18 * (normalized - 1.0).min(2.5));
        if confidence >= config.minimum_confidence {
            candidates.push((index, confidence));
        }
    }
    candidates
}

fn warped_signal(samples: &[f64], lambda: f64) -> Vec<f64> {
    let mut output = vec![0.0; samples.len()];
    for index in 1..samples.len() {
        output[index] = samples[index] - lambda * samples[index - 1] + lambda * output[index - 1];
    }
    output
}

fn merge_candidates(candidates: &[(usize, f64)], merge_gap: usize) -> Vec<Region> {
    let Some(&(first, confidence)) = candidates.first() else {
        return Vec::new();
    };
    let mut regions = Vec::new();
    let mut current = Region {
        start: first,
        end: first + 1,
        confidence,
    };
    for &(index, candidate_confidence) in &candidates[1..] {
        if index <= current.end.saturating_add(merge_gap) {
            current.end = index + 1;
            current.confidence = current.confidence.max(candidate_confidence);
        } else {
            regions.push(current);
            current = Region {
                start: index,
                end: index + 1,
                confidence: candidate_confidence,
            };
        }
    }
    regions.push(current);
    regions
}

fn interpolate_region(
    samples: &mut [f64],
    start: usize,
    end: usize,
    context: usize,
    order: usize,
) -> Result<(), String> {
    if start >= end || end > samples.len() {
        return Err("declick interpolation region is out of bounds".into());
    }
    let context_start = start.saturating_sub(context);
    let context_end = end.saturating_add(context).min(samples.len());
    let mut training = Vec::with_capacity((start - context_start) + (context_end - end));
    training.extend_from_slice(&samples[context_start..start]);
    training.extend_from_slice(&samples[end..context_end]);
    let coefficients = fit_ar(&training, order.min(training.len().saturating_sub(1)));
    if coefficients.is_empty() {
        return Ok(());
    }
    let gap = end - start;
    let mut forward_history = samples[context_start..start].to_vec();
    let mut forward = Vec::with_capacity(gap);
    for _ in 0..gap {
        let value = predict(&forward_history, &coefficients).clamp(-1.0, 1.0);
        forward.push(value);
        forward_history.push(value);
    }
    let reversed_right: Vec<f64> = samples[end..context_end].iter().rev().copied().collect();
    let reverse_coefficients = fit_ar(
        &reversed_right,
        order.min(reversed_right.len().saturating_sub(1)),
    );
    let mut backward_history = reversed_right;
    let mut backward = Vec::with_capacity(gap);
    for _ in 0..gap {
        let value = predict(&backward_history, &reverse_coefficients).clamp(-1.0, 1.0);
        backward.push(value);
        backward_history.push(value);
    }
    backward.reverse();
    for offset in 0..gap {
        let blend = (offset + 1) as f64 / (gap + 1) as f64;
        samples[start + offset] =
            (forward[offset] * (1.0 - blend) + backward[offset] * blend).clamp(-1.0, 1.0);
    }
    Ok(())
}

fn fit_ar(samples: &[f64], order: usize) -> Vec<f64> {
    let order = order.min(samples.len().saturating_sub(1));
    if order == 0 {
        return Vec::new();
    }
    let mut autocorrelation = vec![0.0; order + 1];
    for lag in 0..=order {
        for index in lag..samples.len() {
            autocorrelation[lag] += samples[index] * samples[index - lag];
        }
    }
    autocorrelation[0] += autocorrelation[0].abs() * 1e-6 + 1e-12;
    let mut polynomial = vec![0.0; order + 1];
    polynomial[0] = 1.0;
    let mut error = autocorrelation[0];
    if error <= 1e-18 {
        return vec![0.0; order];
    }
    for current_order in 1..=order {
        let mut numerator = autocorrelation[current_order];
        for index in 1..current_order {
            numerator += polynomial[index] * autocorrelation[current_order - index];
        }
        let reflection = (-numerator / error).clamp(-0.995, 0.995);
        let previous = polynomial.clone();
        polynomial[current_order] = reflection;
        for index in 1..current_order {
            polynomial[index] = previous[index] + reflection * previous[current_order - index];
        }
        error *= 1.0 - reflection * reflection;
        if error <= 1e-18 {
            break;
        }
    }
    polynomial[1..].iter().map(|value| -*value).collect()
}

fn predict(history: &[f64], coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .filter_map(|(offset, coefficient)| {
            history
                .len()
                .checked_sub(offset + 1)
                .map(|index| coefficient * history[index])
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_impulse_is_detected_but_sine_is_not_blanketed() {
        let rate = 48_000u32;
        let mut samples: Vec<f64> = (0..4_800)
            .map(|index| {
                (2.0 * std::f64::consts::PI * 440.0 * index as f64 / rate as f64).sin() * 0.1
            })
            .collect();
        samples[2_400] += 0.8;
        let candidates = detect_candidates(&samples, &DeclickConfig::default());
        assert!(candidates
            .iter()
            .any(|(index, _)| index.abs_diff(2_400) <= 1));
        assert!(candidates.len() < 12);
    }

    #[test]
    fn ar_interpolation_is_finite_and_local() {
        let mut samples: Vec<f64> = (0..512).map(|index| (index as f64 * 0.03).sin()).collect();
        let outside = samples.clone();
        interpolate_region(&mut samples, 250, 253, 64, 12).unwrap();
        assert_eq!(&samples[..250], &outside[..250]);
        assert_eq!(&samples[253..], &outside[253..]);
        assert!(samples[250..253].iter().all(|sample| sample.is_finite()));
    }
}
