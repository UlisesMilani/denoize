use super::{
    finite_confidence, mark_changed_samples, mark_range, DehumConfig, MaskCell, OperationOutcome,
    RestorationMode, RestorationOperation, RestorationOperationDetails, MASK_DETECTED,
};
use std::f64::consts::PI;

const SEARCH_STARTS_HZ: [f64; 2] = [49.0, 59.0];
const SEARCH_STEPS: usize = 21;
const SEARCH_STEP_HZ: f64 = 0.1;
const SEARCH_HARMONICS: usize = 6;

pub(super) fn process(
    channels: &mut [Vec<f64>],
    sample_rate: u32,
    mode: RestorationMode,
    mask: &mut [Vec<MaskCell>],
    config: &DehumConfig,
) -> Result<OperationOutcome, String> {
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    let mut warnings = Vec::new();
    if frames < 64 {
        warnings.push("input is too short for stable harmonic regression".into());
        return Ok(empty_outcome(config, warnings));
    }
    let requested = (config.block_seconds * sample_rate as f64).round() as usize;
    let block_length = requested.clamp(64, frames);
    let hop = (block_length / 2).max(1);
    let starts = block_starts(frames, block_length, hop);
    let mut aggregate = vec![0.0; frames];
    for channel in channels.iter() {
        for (target, sample) in aggregate.iter_mut().zip(channel) {
            *target += *sample / channels.len() as f64;
        }
    }
    let before = if mode == RestorationMode::Apply {
        Some(channels.to_vec())
    } else {
        None
    };
    let mut corrections: Vec<Vec<f64>> = channels
        .iter()
        .map(|channel| vec![0.0; channel.len()])
        .collect();
    let mut weights = vec![0.0; frames];
    let mut tracked_blocks = 0usize;
    let mut confidence_sum = 0.0;
    let mut fundamental_sum = 0.0;
    let mut reported_harmonics = 0usize;
    let attenuation = 1.0 - 10.0f64.powf(-config.attenuation_db / 20.0);

    for start in starts {
        let end = start + block_length;
        let block = &aggregate[start..end];
        let Some(estimate) = estimate_fundamental(block, sample_rate) else {
            continue;
        };
        if estimate.confidence < config.minimum_confidence {
            continue;
        }
        let harmonic_count = ((config.maximum_frequency_hz / estimate.frequency_hz).floor()
            as usize)
            .min(config.maximum_harmonics)
            .min(((sample_rate as f64 * 0.48) / estimate.frequency_hz).floor() as usize)
            .max(1);
        tracked_blocks += 1;
        confidence_sum += estimate.confidence;
        fundamental_sum += estimate.frequency_hz;
        reported_harmonics = reported_harmonics.max(harmonic_count);
        for channel_index in 0..channels.len() {
            mark_range(
                mask,
                channel_index,
                start,
                end,
                MASK_DETECTED,
                RestorationOperation::Dehum,
                estimate.confidence,
            );
            for harmonic in 1..=harmonic_count {
                let frequency = estimate.frequency_hz * harmonic as f64;
                let (sin_coefficient, cos_coefficient) = robust_sinusoid_fit(
                    &channels[channel_index][start..end],
                    sample_rate,
                    frequency,
                );
                let amplitude = sin_coefficient.hypot(cos_coefficient);
                let rms = root_mean_square(&channels[channel_index][start..end]);
                if amplitude <= (rms * 0.001).max(1e-10) {
                    continue;
                }
                for offset in 0..block_length {
                    let phase = 2.0 * PI * frequency * offset as f64 / sample_rate as f64;
                    let window = raised_cosine(offset, block_length);
                    let predicted = sin_coefficient * phase.sin() + cos_coefficient * phase.cos();
                    corrections[channel_index][start + offset] += predicted * window * attenuation;
                }
            }
        }
        for offset in 0..block_length {
            weights[start + offset] += raised_cosine(offset, block_length);
        }
    }

    let confidence = if tracked_blocks == 0 {
        0.0
    } else {
        finite_confidence(confidence_sum / tracked_blocks as f64)
    };
    let mut changed_samples = 0usize;
    if mode == RestorationMode::Apply && tracked_blocks > 0 {
        for channel_index in 0..channels.len() {
            for frame in 0..frames {
                if weights[frame] > 1e-12 {
                    let corrected = channels[channel_index][frame]
                        - corrections[channel_index][frame] / weights[frame];
                    channels[channel_index][frame] = corrected.clamp(-1.0, 1.0);
                }
            }
            changed_samples += mark_changed_samples(
                &before.as_ref().expect("apply mode has a snapshot")[channel_index],
                &channels[channel_index],
                mask,
                channel_index,
                RestorationOperation::Dehum,
                confidence,
            );
        }
    }
    if tracked_blocks == 0 {
        warnings.push("no stable 50/60 Hz harmonic complex passed the confidence gate".into());
    }
    let detected_samples = mask
        .iter()
        .flatten()
        .filter(|cell| cell.operations & RestorationOperation::Dehum.bit() != 0)
        .count();
    Ok(OperationOutcome {
        detected_samples,
        changed_samples,
        confidence,
        warnings,
        details: RestorationOperationDetails::Dehum {
            fundamental_hz: (tracked_blocks > 0).then_some(fundamental_sum / tracked_blocks as f64),
            tracked_blocks,
            harmonic_count: reported_harmonics,
            attenuation_db: config.attenuation_db,
        },
    })
}

fn empty_outcome(config: &DehumConfig, warnings: Vec<String>) -> OperationOutcome {
    OperationOutcome {
        detected_samples: 0,
        changed_samples: 0,
        confidence: 0.0,
        warnings,
        details: RestorationOperationDetails::Dehum {
            fundamental_hz: None,
            tracked_blocks: 0,
            harmonic_count: 0,
            attenuation_db: config.attenuation_db,
        },
    }
}

struct FundamentalEstimate {
    frequency_hz: f64,
    confidence: f64,
}

fn estimate_fundamental(block: &[f64], sample_rate: u32) -> Option<FundamentalEstimate> {
    let rms = root_mean_square(block);
    if rms <= 1e-10 {
        return None;
    }
    let stride = ((sample_rate as usize) / 4_000).max(1);
    let mut best_frequency = 0.0;
    let mut best_score = 0.0;
    let mut best_active = 0usize;
    for search_start in SEARCH_STARTS_HZ {
        for step in 0..SEARCH_STEPS {
            let frequency = search_start + step as f64 * SEARCH_STEP_HZ;
            let mut score = 0.0;
            let mut active = 0usize;
            for harmonic in 1..=SEARCH_HARMONICS {
                let harmonic_frequency = frequency * harmonic as f64;
                if harmonic_frequency >= sample_rate as f64 * 0.48 {
                    break;
                }
                let amplitude =
                    sampled_sinusoid_amplitude(block, sample_rate, harmonic_frequency, stride);
                let relative = amplitude / rms;
                if relative > 0.01 {
                    active += 1;
                }
                score += relative * relative / harmonic as f64;
            }
            if score > best_score {
                best_score = score;
                best_frequency = frequency;
                best_active = active;
            }
        }
    }
    if best_frequency == 0.0 {
        return None;
    }
    let harmonic_support = if best_active >= 2 {
        1.0
    } else if best_active == 1 {
        0.35
    } else {
        0.0
    };
    let confidence = finite_confidence(best_score.sqrt() * 3.0 * harmonic_support);
    Some(FundamentalEstimate {
        frequency_hz: best_frequency,
        confidence,
    })
}

fn sampled_sinusoid_amplitude(
    block: &[f64],
    sample_rate: u32,
    frequency: f64,
    stride: usize,
) -> f64 {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    let mut weight_sum = 0.0;
    for index in (0..block.len()).step_by(stride) {
        let weight = raised_cosine(index, block.len());
        let phase = 2.0 * PI * frequency * index as f64 / sample_rate as f64;
        sin_sum += block[index] * phase.sin() * weight;
        cos_sum += block[index] * phase.cos() * weight;
        weight_sum += weight;
    }
    if weight_sum <= 1e-12 {
        0.0
    } else {
        2.0 * sin_sum.hypot(cos_sum) / weight_sum
    }
}

fn robust_sinusoid_fit(block: &[f64], sample_rate: u32, frequency: f64) -> (f64, f64) {
    let (initial_sin, initial_cos) = weighted_fit(block, sample_rate, frequency, None);
    let mut residuals = Vec::with_capacity(block.len());
    for (index, sample) in block.iter().enumerate() {
        let phase = 2.0 * PI * frequency * index as f64 / sample_rate as f64;
        residuals.push(*sample - initial_sin * phase.sin() - initial_cos * phase.cos());
    }
    let scale = super::median_absolute_deviation(&residuals) * 1.4826;
    if scale <= 1e-12 {
        return (initial_sin, initial_cos);
    }
    weighted_fit(block, sample_rate, frequency, Some((&residuals, scale)))
}

fn weighted_fit(
    block: &[f64],
    sample_rate: u32,
    frequency: f64,
    robust: Option<(&[f64], f64)>,
) -> (f64, f64) {
    let mut ss = 0.0;
    let mut cc = 0.0;
    let mut sc = 0.0;
    let mut ys = 0.0;
    let mut yc = 0.0;
    for (index, sample) in block.iter().enumerate() {
        let phase = 2.0 * PI * frequency * index as f64 / sample_rate as f64;
        let sin = phase.sin();
        let cos = phase.cos();
        let mut weight = raised_cosine(index, block.len());
        if let Some((residuals, scale)) = robust {
            let normalized = residuals[index].abs() / (1.5 * scale);
            if normalized > 1.0 {
                weight /= normalized;
            }
        }
        ss += weight * sin * sin;
        cc += weight * cos * cos;
        sc += weight * sin * cos;
        ys += weight * *sample * sin;
        yc += weight * *sample * cos;
    }
    let determinant = ss * cc - sc * sc;
    if determinant.abs() <= 1e-18 {
        (0.0, 0.0)
    } else {
        (
            (ys * cc - yc * sc) / determinant,
            (yc * ss - ys * sc) / determinant,
        )
    }
}

fn block_starts(frames: usize, block: usize, hop: usize) -> Vec<usize> {
    if frames <= block {
        return vec![0];
    }
    let mut starts = Vec::new();
    let mut start = 0;
    while start + block < frames {
        starts.push(start);
        start = start.saturating_add(hop);
    }
    let last = frames - block;
    if starts.last().copied() != Some(last) {
        starts.push(last);
    }
    starts
}

fn raised_cosine(index: usize, length: usize) -> f64 {
    if length <= 1 {
        1.0
    } else {
        0.5 - 0.5 * (2.0 * PI * index as f64 / (length - 1) as f64).cos()
    }
}

fn root_mean_square(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        (samples.iter().map(|sample| sample * sample).sum::<f64>() / samples.len() as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_missing_fundamental_from_harmonics() {
        let rate = 8_000;
        let samples: Vec<f64> = (0..rate)
            .map(|index| {
                let time = index as f64 / rate as f64;
                0.08 * (2.0 * PI * 100.0 * time).sin() + 0.05 * (2.0 * PI * 150.0 * time).sin()
            })
            .collect();
        let estimate = estimate_fundamental(&samples, rate as u32).unwrap();
        assert!((estimate.frequency_hz - 50.0).abs() <= 0.2);
        assert!(estimate.confidence > 0.55);
    }
}
