//! Band-limited, channel-synchronous sample-rate conversion.

use rubato::{FftFixedIn, Resampler};

use crate::audio::sanitize_sample;
use crate::config::MAX_SAMPLE_RATE;

const CHUNK_FRAMES: usize = 1024;
const SUB_CHUNKS: usize = 2;
const MAX_RESAMPLE_WORKING_BYTES: u128 = 512 * 1024 * 1024;
// Covers rubato's FFT plans, spectra, filters, scratch buffers, and temporary
// constructor storage in addition to the explicitly-sized buffers below.
const FFT_PLAN_SCALAR_SAFETY_FACTOR: u128 = 64;

pub fn resample(input: &[f64], from_rate: u32, to_rate: u32) -> Result<Vec<f64>, String> {
    validate_sample_rates(from_rate, to_rate)?;
    validate_resampler_plan(1, from_rate, to_rate)?;
    let mut channel = Vec::new();
    channel
        .try_reserve_exact(input.len())
        .map_err(|_| "unable to reserve sample-rate conversion input".to_string())?;
    channel.extend_from_slice(input);
    let channels = resample_channels(&[channel], from_rate, to_rate)?;
    Ok(channels.into_iter().next().unwrap_or_default())
}

/// Resample every channel through one shared clock so stereo phase and timing
/// cannot drift. The FFT resampler includes a band-limiting filter, unlike the
/// linear interpolation previously used here.
pub fn resample_channels(
    input: &[Vec<f64>],
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<Vec<f64>>, String> {
    validate_sample_rates(from_rate, to_rate)?;
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let frames = input[0].len();
    if input.iter().any(|channel| channel.len() != frames) {
        return Err("all channels must contain the same number of frames".into());
    }
    if frames == 0 {
        return clone_channels(input, false);
    }
    validate_resampler_plan(input.len(), from_rate, to_rate)?;

    // The converter performs floating-point FFT arithmetic, so one invalid
    // input sample would otherwise contaminate an entire block.  Keep the
    // common finite path allocation-free while still making direct callers
    // safe for NaN, infinity, and out-of-range amplitudes.
    let needs_sanitization = input
        .iter()
        .flatten()
        .any(|sample| !sample.is_finite() || *sample < -1.0 || *sample > 1.0);
    let sanitized;
    let input = if needs_sanitization {
        sanitized = clone_channels(input, true)?;
        &sanitized
    } else {
        input
    };
    if from_rate == to_rate {
        return clone_channels(input, false);
    }

    let expected = planned_output_frames(frames, from_rate, to_rate)?;
    let mut converter = FftFixedIn::<f64>::new(
        from_rate as usize,
        to_rate as usize,
        CHUNK_FRAMES,
        SUB_CHUNKS,
        input.len(),
    )
    .map_err(|error| format!("failed to create sample-rate converter: {error}"))?;
    let delay = converter.output_delay();
    let output_target = expected
        .checked_add(delay)
        .ok_or_else(|| "sample-rate conversion output capacity overflow".to_string())?;
    // A call may return substantially more than CHUNK_FRAMES when upsampling
    // (for example, 48 kHz -> 768 kHz returns up to 16,384 frames). Keep a
    // checked bound for the final flush call, but retain only the samples that
    // belong to the delayed output target in the accumulator.
    let append_limit = output_target
        .checked_add(converter.output_frames_max())
        .ok_or_else(|| "sample-rate conversion output capacity overflow".to_string())?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| "unable to reserve sample-rate conversion channels".to_string())?;
    for _ in 0..input.len() {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(output_target)
            .map_err(|_| "unable to reserve sample-rate conversion output".to_string())?;
        output.push(channel);
    }
    let mut position = 0;

    while frames - position >= converter.input_frames_next() {
        let count = converter.input_frames_next();
        let chunk: Vec<&[f64]> = input
            .iter()
            .map(|channel| &channel[position..position + count])
            .collect();
        let converted = converter
            .process(&chunk, None)
            .map_err(|error| format!("sample-rate conversion failed: {error}"))?;
        append(&mut output, &converted, output_target, append_limit)?;
        position += count;
    }
    if position < frames {
        let tail: Vec<&[f64]> = input.iter().map(|channel| &channel[position..]).collect();
        let converted = converter
            .process_partial(Some(&tail), None)
            .map_err(|error| format!("sample-rate conversion failed: {error}"))?;
        append(&mut output, &converted, output_target, append_limit)?;
    }
    while output.first().map_or(0, Vec::len) < output_target {
        let converted = converter
            .process_partial::<&[f64]>(None, None)
            .map_err(|error| format!("sample-rate conversion flush failed: {error}"))?;
        append(&mut output, &converted, output_target, append_limit)?;
    }

    for channel in &mut output {
        channel.drain(..delay.min(channel.len()));
        channel.truncate(expected);
        channel.resize(expected, 0.0);
        for sample in channel {
            *sample = sanitize_sample(*sample);
        }
    }
    Ok(output)
}

fn validate_sample_rates(from_rate: u32, to_rate: u32) -> Result<(), String> {
    if from_rate == 0 || to_rate == 0 || from_rate > MAX_SAMPLE_RATE || to_rate > MAX_SAMPLE_RATE {
        return Err(format!(
            "sample rates must be between 1 and {MAX_SAMPLE_RATE} Hz"
        ));
    }
    Ok(())
}

fn planned_output_frames(frames: usize, from_rate: u32, to_rate: u32) -> Result<usize, String> {
    let numerator = (frames as u128)
        .checked_mul(to_rate as u128)
        .and_then(|value| value.checked_add(from_rate as u128 / 2))
        .ok_or_else(|| "sample-rate conversion output length overflow".to_string())?;
    usize::try_from(numerator / from_rate as u128)
        .map_err(|_| "sample-rate conversion output length is too large".to_string())
}

pub(crate) fn validate_resampler_plan(
    channels: usize,
    from_rate: u32,
    to_rate: u32,
) -> Result<(), String> {
    resampler_plan_bytes(channels, from_rate, to_rate).map(|_| ())
}

pub(crate) fn resampler_plan_bytes(
    channels: usize,
    from_rate: u32,
    to_rate: u32,
) -> Result<u64, String> {
    validate_sample_rates(from_rate, to_rate)?;
    if channels == 0 {
        return Err("sample-rate conversion requires at least one channel".into());
    }
    if from_rate == to_rate {
        return Ok(0);
    }
    let gcd = greatest_common_divisor(from_rate as u128, to_rate as u128);
    let minimum_input_chunk = from_rate as u128 / gcd;
    let wanted_subchunk = (CHUNK_FRAMES / SUB_CHUNKS) as u128;
    let fft_chunks = wanted_subchunk
        .checked_add(minimum_input_chunk - 1)
        .map(|value| value / minimum_input_chunk)
        .ok_or_else(|| "sample-rate conversion FFT plan overflow".to_string())?;
    let fft_size_in = fft_chunks
        .checked_mul(from_rate as u128 / gcd)
        .ok_or_else(|| "sample-rate conversion FFT plan overflow".to_string())?;
    let fft_size_out = fft_chunks
        .checked_mul(to_rate as u128 / gcd)
        .ok_or_else(|| "sample-rate conversion FFT plan overflow".to_string())?;
    let maximum_available = fft_size_in
        .checked_sub(1)
        .and_then(|value| value.checked_add(CHUNK_FRAMES as u128))
        .ok_or_else(|| "sample-rate conversion FFT plan overflow".to_string())?;
    let maximum_output = (maximum_available / fft_size_in)
        .checked_mul(fft_size_out)
        .ok_or_else(|| "sample-rate conversion FFT plan overflow".to_string())?;

    let per_channel = (CHUNK_FRAMES as u128)
        .checked_add(fft_size_in)
        .and_then(|value| value.checked_add(fft_size_out))
        .and_then(|value| value.checked_add(maximum_output))
        .ok_or_else(|| "sample-rate conversion buffer plan overflow".to_string())?;
    let channel_samples = per_channel
        .checked_mul(channels as u128)
        .ok_or_else(|| "sample-rate conversion buffer plan overflow".to_string())?;
    let shared_samples = fft_size_in
        .checked_add(fft_size_out)
        .and_then(|value| value.checked_mul(FFT_PLAN_SCALAR_SAFETY_FACTOR))
        .ok_or_else(|| "sample-rate conversion FFT plan overflow".to_string())?;
    let bytes = channel_samples
        .checked_add(shared_samples)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() as u128))
        .ok_or_else(|| "sample-rate conversion working-set plan overflow".to_string())?;
    if bytes > MAX_RESAMPLE_WORKING_BYTES {
        return Err(format!(
            "sample-rate conversion working set requires {bytes} bytes, limit is {MAX_RESAMPLE_WORKING_BYTES} bytes"
        ));
    }
    u64::try_from(bytes).map_err(|_| "sample-rate conversion working-set plan overflow".to_string())
}

fn greatest_common_divisor(mut lhs: u128, mut rhs: u128) -> u128 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn clone_channels(input: &[Vec<f64>], sanitize: bool) -> Result<Vec<Vec<f64>>, String> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(input.len())
        .map_err(|_| "unable to reserve sample-rate conversion channels".to_string())?;
    for source in input {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(source.len())
            .map_err(|_| "unable to reserve sample-rate conversion samples".to_string())?;
        if sanitize {
            channel.extend(source.iter().copied().map(sanitize_sample));
        } else {
            channel.extend_from_slice(source);
        }
        cloned.push(channel);
    }
    Ok(cloned)
}

fn append(
    output: &mut [Vec<f64>],
    chunk: &[Vec<f64>],
    output_target: usize,
    append_limit: usize,
) -> Result<(), String> {
    for (output, chunk) in output.iter_mut().zip(chunk) {
        let produced = output
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| "sample-rate conversion output length overflow".to_string())?;
        if produced > append_limit {
            return Err("sample-rate conversion output exceeded its planned bound".into());
        }
        let remaining = output_target
            .checked_sub(output.len())
            .ok_or_else(|| "sample-rate conversion output exceeded its target".to_string())?;
        let retained = remaining.min(chunk.len());
        output
            .try_reserve_exact(retained)
            .map_err(|_| "unable to grow sample-rate conversion output".to_string())?;
        output.extend_from_slice(&chunk[..retained]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn same_rate_is_an_exact_identity() {
        let input = vec![0.25, -0.5, 1.0];
        assert_eq!(resample(&input, 48_000, 48_000).unwrap(), input);
    }

    #[test]
    fn preserves_requested_duration() {
        let input = vec![0.0; 44_100];
        assert_eq!(resample(&input, 44_100, 16_000).unwrap().len(), 16_000);
        assert_eq!(resample(&input, 44_100, 48_000).unwrap().len(), 48_000);
    }

    #[test]
    fn high_ratio_flush_retains_only_the_requested_output() {
        let input = vec![0.0; CHUNK_FRAMES * 2];
        let output = resample(&input, 48_000, 768_000).unwrap();
        assert_eq!(output.len(), input.len() * 16);
    }

    #[test]
    fn append_discards_flush_overshoot_at_the_checked_target() {
        let mut output = vec![Vec::new()];
        output[0].try_reserve_exact(4).unwrap();
        let capacity = output[0].capacity();
        append(&mut output, &[vec![1.0; 16]], 4, 20).unwrap();
        assert_eq!(output[0], vec![1.0; 4]);
        assert_eq!(output[0].capacity(), capacity);
    }

    #[test]
    fn downsampling_rejects_content_above_nyquist() {
        let tone = |frequency: f64| {
            (0..48_000)
                .map(|i| (TAU * frequency * i as f64 / 48_000.0).sin())
                .collect::<Vec<_>>()
        };
        let passband = resample(&tone(1_000.0), 48_000, 16_000).unwrap();
        let stopband = resample(&tone(12_000.0), 48_000, 16_000).unwrap();
        let rms = |samples: &[f64]| {
            (samples.iter().map(|x| x * x).sum::<f64>() / samples.len() as f64).sqrt()
        };
        assert!(rms(&stopband) < rms(&passband) * 0.01);
    }

    #[test]
    fn linked_channels_remain_sample_identical() {
        let channel: Vec<f64> = (0..4_410)
            .map(|i| (TAU * 997.0 * i as f64 / 44_100.0).sin())
            .collect();
        let output = resample_channels(&[channel.clone(), channel], 44_100, 48_000).unwrap();
        assert_eq!(output[0], output[1]);
    }

    #[test]
    fn supports_arbitrary_sample_rate_pairs() {
        // Include both conventional rates and a deliberately non-standard
        // rate to make sure conversion does not rely on a fixed codec table.
        let rates = [8_000, 12_345, 22_050, 32_000, 44_100, 48_000, 96_000];
        let frames = 1_001usize;
        let input: Vec<f64> = (0..frames)
            .map(|frame| (TAU * 997.0 * frame as f64 / 44_100.0).sin() * 0.5)
            .collect();

        for &from_rate in &rates {
            for &to_rate in &rates {
                if from_rate == to_rate {
                    continue;
                }
                let output = resample(&input, from_rate, to_rate).unwrap_or_else(|error| {
                    panic!("{from_rate} Hz -> {to_rate} Hz conversion failed: {error}")
                });
                let expected = ((frames as u128 * to_rate as u128 + from_rate as u128 / 2)
                    / from_rate as u128) as usize;
                assert_eq!(
                    output.len(),
                    expected,
                    "{from_rate} Hz -> {to_rate} Hz output length"
                );
                assert!(
                    output.iter().all(|sample| sample.is_finite()),
                    "{from_rate} Hz -> {to_rate} Hz produced a non-finite sample"
                );
            }
        }
    }

    #[test]
    fn sanitizes_nonfinite_and_extreme_samples_before_conversion() {
        let input = vec![vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.0,
            -2.0,
            0.25,
        ]];
        let output = resample_channels(&input, 48_000, 48_000).unwrap();
        assert_eq!(output[0], vec![0.0, 0.0, 0.0, 1.0, -1.0, 0.25]);
    }

    #[test]
    fn hostile_rate_and_capacity_plans_fail_without_allocating() {
        assert!(resample(&[0.0], 48_000, MAX_SAMPLE_RATE + 1).is_err());
        assert!(resample(&[], 0, 48_000).is_err());
        assert!(validate_resampler_plan(usize::MAX, 1, MAX_SAMPLE_RATE).is_err());
        let tiny_many_channel_input = vec![vec![0.0]; 100];
        let error = resample_channels(&tiny_many_channel_input, MAX_SAMPLE_RATE, 1).unwrap_err();
        assert!(error.contains("working set"), "unexpected error: {error}");
        if usize::BITS < 128 {
            assert!(planned_output_frames(usize::MAX, 1, MAX_SAMPLE_RATE).is_err());
        }
    }
}
