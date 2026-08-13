//! RNNoise backend via `nnnoiseless` (pure-Rust port of Xiph RNNoise).
//!
//! Operates at 48 kHz, 480-sample frames. Other sample rates are converted with
//! a band-limited FFT resampler.

use nnnoiseless::DenoiseState;

const RN_SR: u32 = 48_000;
const FRAME: usize = DenoiseState::FRAME_SIZE;

/// Stateful RNNoise processing for a continuous stream of arbitrary-sized
/// channel-planar blocks.
///
/// The processor retains one RNNoise model state per channel, incomplete
/// 480-sample model frames, and both sample-rate converters. Consequently a
/// block can legitimately produce zero frames while bounded internal latency
/// is accumulated. Call [`finish`](Self::finish) to emit the final partial
/// frame and resampler delay.
pub(crate) struct StreamingProcessor {
    channels: usize,
    to_48k: crate::resample::StreamingResampler,
    from_48k: crate::resample::StreamingResampler,
    denoise: Vec<Box<DenoiseState<'static>>>,
    pending_48k: Vec<Vec<f32>>,
    input_frames: usize,
    output_frames: usize,
    finished: bool,
}

impl StreamingProcessor {
    pub(crate) fn new(sample_rate: u32, channels: usize) -> Result<Self, String> {
        if channels == 0 || channels > crate::config::MAX_STREAM_CHANNELS {
            return Err(format!(
                "RNNoise streaming channels must be between 1 and {}",
                crate::config::MAX_STREAM_CHANNELS
            ));
        }
        let to_48k = crate::resample::StreamingResampler::new(channels, sample_rate, RN_SR)?;
        let from_48k = crate::resample::StreamingResampler::new(channels, RN_SR, sample_rate)?;

        let mut denoise = Vec::new();
        denoise
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve RNNoise channel states".to_string())?;
        let mut pending_48k = Vec::new();
        pending_48k
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve RNNoise pending channels".to_string())?;
        for _ in 0..channels {
            denoise.push(DenoiseState::new());
            let mut pending = Vec::new();
            pending
                .try_reserve_exact(FRAME)
                .map_err(|_| "unable to reserve RNNoise pending samples".to_string())?;
            pending_48k.push(pending);
        }

        Ok(Self {
            channels,
            to_48k,
            from_48k,
            denoise,
            pending_48k,
            input_frames: 0,
            output_frames: 0,
            finished: false,
        })
    }

    /// Process another block. Input and output are channel-planar; all input
    /// channels must have the same frame count.
    pub(crate) fn process_block(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("RNNoise stream is finished; reset it before processing more input".into());
        }
        let frames = validate_stream_block(input, self.channels)?;
        let input_frames = self
            .input_frames
            .checked_add(frames)
            .ok_or_else(|| "RNNoise streaming input length overflow".to_string())?;

        let at_48k = self.to_48k.process(input)?;
        let enhanced_48k = self.process_48k(&at_48k)?;
        let output = self.from_48k.process(&enhanced_48k)?;
        let produced = output.first().map_or(0, Vec::len);
        if output.iter().any(|channel| channel.len() != produced) {
            return Err("RNNoise stream produced unaligned channels".into());
        }
        let output_frames = self
            .output_frames
            .checked_add(produced)
            .ok_or_else(|| "RNNoise streaming output length overflow".to_string())?;
        if output_frames > input_frames {
            return Err("RNNoise stream produced samples ahead of its input clock".into());
        }
        self.input_frames = input_frames;
        self.output_frames = output_frames;
        Ok(output)
    }

    /// Flush all incomplete model and resampler frames. The concatenated output
    /// from every process call plus this tail is exactly as long as the accepted
    /// input stream. Calling finish again returns one empty vector per channel.
    // The current device loop drops queued audio at shutdown, but keep an
    // explicit lifecycle for embedders and future graceful playback draining.
    #[allow(dead_code)]
    pub(crate) fn finish(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .input_frames
            .checked_sub(self.output_frames)
            .ok_or_else(|| "RNNoise stream exceeded its input clock".to_string())?;
        let mut output = empty_output(self.channels, remaining)?;
        if self.finished {
            return Ok(output);
        }

        let tail_48k = self.to_48k.finish()?;
        let enhanced = self.process_48k(&tail_48k)?;
        let converted = self.from_48k.process(&enhanced)?;
        append_limited(&mut output, &converted, remaining)?;

        let final_model_frame = self.finish_48k()?;
        let converted = self.from_48k.process(&final_model_frame)?;
        append_limited(&mut output, &converted, remaining)?;

        let converted = self.from_48k.finish()?;
        append_limited(&mut output, &converted, remaining)?;

        let produced = output.first().map_or(0, Vec::len);
        if produced < remaining {
            for channel in &mut output {
                channel.resize(remaining, 0.0);
            }
        }
        self.output_frames = self.input_frames;
        self.finished = true;
        Ok(output)
    }

    /// Start a logically independent stream with the same format.
    #[allow(dead_code)]
    pub(crate) fn reset(&mut self) {
        self.to_48k.reset();
        self.from_48k.reset();
        for state in &mut self.denoise {
            *state = DenoiseState::new();
        }
        for pending in &mut self.pending_48k {
            pending.clear();
        }
        self.input_frames = 0;
        self.output_frames = 0;
        self.finished = false;
    }

    fn process_48k(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let frames = validate_stream_block(input, self.channels)?;
        let pending_frames = self.pending_48k.first().map_or(0, Vec::len);
        let available = pending_frames
            .checked_add(frames)
            .ok_or_else(|| "RNNoise model input length overflow".to_string())?;
        let complete_frames = (available / FRAME)
            .checked_mul(FRAME)
            .ok_or_else(|| "RNNoise model output length overflow".to_string())?;
        let mut output = empty_output(self.channels, complete_frames)?;
        if frames == 0 {
            return Ok(output);
        }

        let mut position = 0usize;
        if pending_frames > 0 {
            let copied = (FRAME - pending_frames).min(frames);
            for (pending, source) in self.pending_48k.iter_mut().zip(input) {
                pending.extend(source[..copied].iter().copied().map(to_rnnoise_sample));
            }
            position = copied;
            if pending_frames + copied == FRAME {
                for channel in 0..self.channels {
                    let mut denoised = [0.0f32; FRAME];
                    self.denoise[channel].process_frame(&mut denoised, &self.pending_48k[channel]);
                    output[channel].extend(denoised.iter().map(|sample| *sample as f64 / 32768.0));
                    self.pending_48k[channel].clear();
                }
            }
        }

        while frames - position >= FRAME {
            for channel in 0..self.channels {
                let mut model_input = [0.0f32; FRAME];
                for (destination, source) in model_input
                    .iter_mut()
                    .zip(&input[channel][position..position + FRAME])
                {
                    *destination = to_rnnoise_sample(*source);
                }
                let mut denoised = [0.0f32; FRAME];
                self.denoise[channel].process_frame(&mut denoised, &model_input);
                output[channel].extend(denoised.iter().map(|sample| *sample as f64 / 32768.0));
            }
            position += FRAME;
        }

        if position < frames {
            for (pending, source) in self.pending_48k.iter_mut().zip(input) {
                pending.extend(source[position..].iter().copied().map(to_rnnoise_sample));
            }
        }
        Ok(output)
    }

    #[allow(dead_code)]
    fn finish_48k(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let retained = self.pending_48k.first().map_or(0, Vec::len);
        let mut output = empty_output(self.channels, retained)?;
        if retained == 0 {
            return Ok(output);
        }
        if self
            .pending_48k
            .iter()
            .any(|channel| channel.len() != retained)
        {
            return Err("RNNoise pending channels became unaligned".into());
        }
        for channel in 0..self.channels {
            let mut model_input = [0.0f32; FRAME];
            model_input[..retained].copy_from_slice(&self.pending_48k[channel]);
            let mut denoised = [0.0f32; FRAME];
            self.denoise[channel].process_frame(&mut denoised, &model_input);
            output[channel].extend(
                denoised[..retained]
                    .iter()
                    .map(|sample| *sample as f64 / 32768.0),
            );
            self.pending_48k[channel].clear();
        }
        Ok(output)
    }
}

/// Denoise channels using RNNoise.
pub fn process(channels: &[Vec<f64>], sample_rate: u32) -> Result<Vec<Vec<f64>>, String> {
    let mut out = Vec::with_capacity(channels.len());
    for ch in channels {
        out.push(process_channel(ch, sample_rate)?);
    }
    Ok(out)
}

fn process_channel(input: &[f64], sample_rate: u32) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Resample to 48 kHz if needed.
    let at_48k: Vec<f32> = if sample_rate == RN_SR {
        input.iter().map(|&x| (x as f32) * 32768.0).collect()
    } else {
        crate::resample::resample(input, sample_rate, RN_SR)?
            .into_iter()
            .map(|x| (x as f32) * 32768.0)
            .collect()
    };

    let mut denoise = DenoiseState::new();
    let mut out_buf = [0.0f32; FRAME];
    let mut output = Vec::with_capacity(at_48k.len());
    let mut i = 0;
    while i < at_48k.len() {
        let end = (i + FRAME).min(at_48k.len());
        let mut frame = [0.0f32; FRAME];
        frame[..end - i].copy_from_slice(&at_48k[i..end]);
        denoise.process_frame(&mut out_buf, &frame);
        // Keep every frame aligned with its input. Discarding the first frame
        // shortens the stream by 10 ms, shifts all remaining audio earlier,
        // and turns inputs <= FRAME into silence.
        let n = if end - i == FRAME { FRAME } else { end - i };
        output.extend_from_slice(&out_buf[..n]);
        i += FRAME;
    }

    // Resample back to original rate.
    let normalized: Vec<f64> = output.iter().map(|&x| (x as f64) / 32768.0).collect();
    let result = if sample_rate == RN_SR {
        normalized
    } else {
        crate::resample::resample(&normalized, RN_SR, sample_rate)?
    };

    // Match input length.
    let mut trimmed = result;
    trimmed.truncate(input.len());
    if trimmed.len() < input.len() {
        trimmed.resize(input.len(), 0.0);
    }
    Ok(trimmed)
}

fn validate_stream_block(input: &[Vec<f64>], channels: usize) -> Result<usize, String> {
    if input.len() != channels {
        return Err(format!(
            "RNNoise stream expected {channels} channels, received {}",
            input.len()
        ));
    }
    let frames = input.first().map_or(0, Vec::len);
    if input.iter().any(|channel| channel.len() != frames) {
        return Err("RNNoise stream channels must contain the same number of frames".into());
    }
    Ok(frames)
}

fn empty_output(channels: usize, capacity: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| "unable to reserve RNNoise output channels".to_string())?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(capacity)
            .map_err(|_| "unable to reserve RNNoise output samples".to_string())?;
        output.push(channel);
    }
    Ok(output)
}

#[allow(dead_code)]
fn append_limited(
    destination: &mut [Vec<f64>],
    source: &[Vec<f64>],
    frame_limit: usize,
) -> Result<(), String> {
    if source.len() != destination.len() {
        return Err("RNNoise stream produced an invalid channel count".into());
    }
    let destination_frames = destination.first().map_or(0, Vec::len);
    if destination
        .iter()
        .any(|channel| channel.len() != destination_frames)
    {
        return Err("RNNoise destination channels became unaligned".into());
    }
    let source_frames = source.first().map_or(0, Vec::len);
    if source.iter().any(|channel| channel.len() != source_frames) {
        return Err("RNNoise stream produced unaligned channels".into());
    }
    let retained = frame_limit
        .checked_sub(destination_frames)
        .ok_or_else(|| "RNNoise streaming output exceeded its target".to_string())?
        .min(source_frames);
    for (destination, source) in destination.iter_mut().zip(source) {
        destination
            .try_reserve_exact(retained)
            .map_err(|_| "unable to grow RNNoise output".to_string())?;
        destination.extend_from_slice(&source[..retained]);
    }
    Ok(())
}

fn to_rnnoise_sample(sample: f64) -> f32 {
    (crate::audio::sanitize_sample(sample) as f32) * 32768.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert!(process_channel(&[], 48_000).unwrap().is_empty());
    }

    #[test]
    fn short_input_keeps_length_and_audio() {
        let input: Vec<f64> = (0..FRAME)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / RN_SR as f64).sin() * 0.5)
            .collect();
        let output = process_channel(&input, RN_SR).unwrap();
        assert_eq!(output.len(), input.len());
        assert!(output.iter().any(|x| x.abs() > 1e-6));
    }

    #[test]
    fn streaming_arbitrary_chunks_match_offline_at_48k() {
        let left: Vec<f64> = (0..2_137)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / RN_SR as f64).sin() * 0.5)
            .collect();
        let right: Vec<f64> = left.iter().map(|sample| -*sample * 0.7).collect();
        let input = vec![left, right];
        let expected = process(&input, RN_SR).unwrap();
        let mut processor = StreamingProcessor::new(RN_SR, 2).unwrap();
        let mut actual = vec![Vec::new(), Vec::new()];
        let chunk_sizes = [1usize, FRAME - 2, 3, 17, 701, 5];
        let mut position = 0usize;
        let mut chunk = 0usize;
        while position < input[0].len() {
            let end = (position + chunk_sizes[chunk % chunk_sizes.len()]).min(input[0].len());
            let block = input
                .iter()
                .map(|channel| channel[position..end].to_vec())
                .collect::<Vec<_>>();
            let enhanced = processor.process_block(&block).unwrap();
            for (destination, source) in actual.iter_mut().zip(enhanced) {
                destination.extend(source);
            }
            position = end;
            chunk += 1;
        }
        let tail = processor.finish().unwrap();
        for (destination, source) in actual.iter_mut().zip(tail) {
            destination.extend(source);
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn streaming_resampled_chunks_match_offline_and_keep_channels_aligned() {
        let rate = 44_100;
        let channel: Vec<f64> = (0..5_003)
            .map(|i| (2.0 * std::f64::consts::PI * 997.0 * i as f64 / rate as f64).sin() * 0.25)
            .collect();
        let input = vec![channel.clone(), channel];
        let expected = process(&input, rate).unwrap();
        let mut processor = StreamingProcessor::new(rate, 2).unwrap();
        let mut actual = vec![Vec::new(), Vec::new()];
        for block_start in (0..input[0].len()).step_by(333) {
            let block_end = (block_start + 333).min(input[0].len());
            let block = input
                .iter()
                .map(|channel| channel[block_start..block_end].to_vec())
                .collect::<Vec<_>>();
            let enhanced = processor.process_block(&block).unwrap();
            assert_eq!(enhanced[0].len(), enhanced[1].len());
            for (destination, source) in actual.iter_mut().zip(enhanced) {
                destination.extend(source);
            }
        }
        let tail = processor.finish().unwrap();
        for (destination, source) in actual.iter_mut().zip(tail) {
            destination.extend(source);
        }
        assert_eq!(actual, expected);
        assert_eq!(actual[0], actual[1]);
    }

    #[test]
    fn streaming_partial_frame_finish_and_reset_are_explicit() {
        let input = vec![vec![0.25; FRAME - 1]];
        let mut processor = StreamingProcessor::new(RN_SR, 1).unwrap();
        assert!(processor.process_block(&input).unwrap()[0].is_empty());
        assert_eq!(processor.finish().unwrap()[0].len(), FRAME - 1);
        assert!(processor.finish().unwrap()[0].is_empty());
        assert!(processor.process_block(&input).is_err());

        processor.reset();
        assert!(processor.process_block(&input).unwrap()[0].is_empty());
        assert_eq!(processor.finish().unwrap()[0].len(), FRAME - 1);
    }

    #[test]
    fn streaming_rejects_invalid_channels_and_rates() {
        assert!(StreamingProcessor::new(0, 1).is_err());
        assert!(StreamingProcessor::new(RN_SR, 0).is_err());
        assert!(StreamingProcessor::new(RN_SR, crate::config::MAX_STREAM_CHANNELS + 1).is_err());
        let mut processor = StreamingProcessor::new(RN_SR, 2).unwrap();
        assert!(processor.process_block(&[vec![0.0]]).is_err());
        assert!(processor
            .process_block(&[vec![0.0], vec![0.0, 0.0]])
            .is_err());
    }
}
