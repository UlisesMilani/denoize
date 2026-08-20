//! ClearerVoice MossFormer2 48 kHz speech-enhancement adapter.
//!
//! The converted graph maps `[1, 496, 180]` Kaldi fbank/delta features to a
//! real-valued `[1, 496, 961]` spectral mask.  This module reproduces the
//! official four-second segmentation, 40 ms/8 ms frontend, non-centred
//! symmetric-Hamming STFT, mask application, and edge-discard stitching.

use super::tract_runtime::SharedRunnable;
use super::OnnxModelConfig;
use crate::AcceleratorRuntime;
use kaldi_native_fbank::{
    mel::MelOptions, FbankComputer, FbankOptions, FrameOptions, OnlineFeature,
};
use rustfft::{num_complex::Complex32, FftPlanner};
use tract_onnx::prelude::*;

const MODEL_RATE: u32 = 48_000;
const WINDOW_SAMPLES: usize = 192_000;
const STRIDE_SAMPLES: usize = 144_000;
const GIVE_UP_SAMPLES: usize = 24_000;
const FFT_SIZE: usize = 1_920;
const HOP_SIZE: usize = 384;
const FRAMES: usize = 496;
const BINS: usize = FFT_SIZE / 2 + 1;
const MEL_BINS: usize = 60;
const FEATURES: usize = MEL_BINS * 3;
const STREAM_SCRATCH_ALLOWANCE_BYTES: u64 = 16 * 1024 * 1024;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<Vec<f64>>, String> {
    Mossformer2Model::load(config, AcceleratorRuntime::Cpu)?.process(channels, input_sample_rate)
}

pub(crate) struct Mossformer2Model {
    model: SharedRunnable,
}

impl Mossformer2Model {
    pub(crate) fn load(
        config: &OnnxModelConfig,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        if config.sample_rate != MODEL_RATE {
            return Err(format!(
                "MossFormer2 expects a {MODEL_RATE} Hz model, got {} Hz",
                config.sample_rate
            ));
        }
        if !config.path.is_file() {
            return Err(format!(
                "MossFormer2 ONNX model does not exist or is not a file: {}",
                config.path.display()
            ));
        }
        Ok(Self {
            model: load_model(config, runtime)?,
        })
    }

    pub(crate) fn process(
        &self,
        channels: &[Vec<f64>],
        input_sample_rate: u32,
    ) -> Result<Vec<Vec<f64>>, String> {
        if channels.is_empty() {
            return Ok(Vec::new());
        }
        channels
            .iter()
            .map(|channel| process_channel(channel, input_sample_rate, self.model.as_ref()))
            .collect()
    }
}

/// Continuous bounded-window MossFormer2 processing.
///
/// The official four-second window is retained until complete. Each following
/// inference advances by the three-second stride and emits only the samples
/// outside the model's discarded half-second edges. At end of stream one
/// partial window is zero-padded exactly like the offline adapter.
pub(crate) struct StreamingProcessor {
    channels: usize,
    to_model_rate: crate::resample::StreamingResampler,
    from_model_rate: crate::resample::StreamingResampler,
    model: SharedRunnable,
    pending_model_rate: Vec<Vec<f64>>,
    windows_processed: usize,
    model_source_frames: usize,
    model_output_frames: usize,
    input_frames: usize,
    output_frames: usize,
    finished: bool,
}

impl StreamingProcessor {
    pub(crate) fn new_with_accelerator(
        config: &OnnxModelConfig,
        sample_rate: u32,
        channels: usize,
        runtime: AcceleratorRuntime,
    ) -> Result<Self, String> {
        if channels == 0 || channels > crate::config::MAX_STREAM_CHANNELS {
            return Err(format!(
                "MossFormer2 streaming channels must be between 1 and {}",
                crate::config::MAX_STREAM_CHANNELS
            ));
        }
        let model = Mossformer2Model::load(config, runtime)?.model;
        let to_model_rate =
            crate::resample::StreamingResampler::new(channels, sample_rate, MODEL_RATE)?;
        let from_model_rate =
            crate::resample::StreamingResampler::new(channels, MODEL_RATE, sample_rate)?;
        let mut pending_model_rate = Vec::new();
        pending_model_rate
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve MossFormer2 pending channels".to_string())?;
        for _ in 0..channels {
            let mut pending = Vec::new();
            pending
                .try_reserve_exact(WINDOW_SAMPLES)
                .map_err(|_| "unable to reserve MossFormer2 pending samples".to_string())?;
            pending_model_rate.push(pending);
        }
        Ok(Self {
            channels,
            to_model_rate,
            from_model_rate,
            model,
            pending_model_rate,
            windows_processed: 0,
            model_source_frames: 0,
            model_output_frames: 0,
            input_frames: 0,
            output_frames: 0,
            finished: false,
        })
    }

    pub(crate) fn process_block(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err(
                "MossFormer2 stream is finished; reset it before processing more input".into(),
            );
        }
        let frames = validate_stream_block(input, self.channels)?;
        let input_frames = self
            .input_frames
            .checked_add(frames)
            .ok_or_else(|| "MossFormer2 streaming input length overflow".to_string())?;
        let at_model_rate = self.to_model_rate.process(input)?;
        let enhanced_model_rate = self.process_model_rate(&at_model_rate)?;
        let output = self.from_model_rate.process(&enhanced_model_rate)?;
        let produced = validate_stream_block(&output, self.channels)?;
        let output_frames = self
            .output_frames
            .checked_add(produced)
            .ok_or_else(|| "MossFormer2 streaming output length overflow".to_string())?;
        if output_frames > input_frames {
            return Err("MossFormer2 stream produced samples ahead of its input clock".into());
        }
        self.input_frames = input_frames;
        self.output_frames = output_frames;
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .input_frames
            .checked_sub(self.output_frames)
            .ok_or_else(|| "MossFormer2 stream exceeded its input clock".to_string())?;
        let mut output = empty_stream_output(self.channels, remaining)?;
        if self.finished {
            return Ok(output);
        }

        let model_input_tail = self.to_model_rate.finish()?;
        let enhanced = self.process_model_rate(&model_input_tail)?;
        let converted = self.from_model_rate.process(&enhanced)?;
        append_stream_output(&mut output, &converted, remaining)?;

        let enhanced = self.finish_model_rate()?;
        let converted = self.from_model_rate.process(&enhanced)?;
        append_stream_output(&mut output, &converted, remaining)?;

        let converted = self.from_model_rate.finish()?;
        append_stream_output(&mut output, &converted, remaining)?;
        if output.first().map_or(0, Vec::len) < remaining {
            for channel in &mut output {
                channel.resize(remaining, 0.0);
            }
        }
        self.output_frames = self.input_frames;
        self.finished = true;
        Ok(output)
    }

    pub(crate) fn reset(&mut self) {
        self.to_model_rate.reset();
        self.from_model_rate.reset();
        for pending in &mut self.pending_model_rate {
            pending.clear();
        }
        self.windows_processed = 0;
        self.model_source_frames = 0;
        self.model_output_frames = 0;
        self.input_frames = 0;
        self.output_frames = 0;
        self.finished = false;
    }

    fn process_model_rate(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let frames = validate_stream_block(input, self.channels)?;
        self.model_source_frames = self
            .model_source_frames
            .checked_add(frames)
            .ok_or_else(|| "MossFormer2 model input length overflow".to_string())?;
        let mut output = empty_stream_output(self.channels, frames)?;
        let mut position = 0usize;
        while position < frames {
            let pending = self.pending_model_rate.first().map_or(0, Vec::len);
            let copied = (WINDOW_SAMPLES - pending).min(frames - position);
            for (destination, source) in self.pending_model_rate.iter_mut().zip(input) {
                destination.extend(
                    source[position..position + copied]
                        .iter()
                        .copied()
                        .map(crate::sanitize_sample),
                );
            }
            position += copied;
            if pending + copied == WINDOW_SAMPLES {
                self.run_window(&mut output)?;
            }
        }
        Ok(output)
    }

    fn finish_model_rate(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .model_source_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "MossFormer2 model output exceeded its input clock".to_string())?;
        let mut output = empty_stream_output(self.channels, remaining)?;
        if self.model_source_frames == 0 {
            return Ok(output);
        }
        let last_complete_end = if self.windows_processed == 0 {
            0
        } else {
            WINDOW_SAMPLES
                .checked_add(
                    (self.windows_processed - 1)
                        .checked_mul(STRIDE_SAMPLES)
                        .ok_or_else(|| "MossFormer2 window position overflow".to_string())?,
                )
                .ok_or_else(|| "MossFormer2 window position overflow".to_string())?
        };
        if self.model_source_frames > last_complete_end {
            for pending in &mut self.pending_model_rate {
                pending.resize(WINDOW_SAMPLES, 0.0);
            }
            self.run_window(&mut output)?;
        }
        let produced = output.first().map_or(0, Vec::len);
        if produced < remaining {
            for channel in &mut output {
                channel.resize(remaining, 0.0);
            }
            self.model_output_frames = self.model_source_frames;
        }
        for pending in &mut self.pending_model_rate {
            pending.clear();
        }
        if self.model_output_frames != self.model_source_frames {
            return Err("MossFormer2 stream did not flush to its model clock".into());
        }
        Ok(output)
    }

    fn run_window(&mut self, output: &mut [Vec<f64>]) -> Result<(), String> {
        if self
            .pending_model_rate
            .iter()
            .any(|channel| channel.len() != WINDOW_SAMPLES)
        {
            return Err("MossFormer2 streaming window has an invalid length".into());
        }
        let source_start = if self.windows_processed == 0 {
            0
        } else {
            GIVE_UP_SAMPLES
        };
        let source_end = WINDOW_SAMPLES - GIVE_UP_SAMPLES;
        let available = source_end - source_start;
        let remaining = self
            .model_source_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "MossFormer2 model output exceeded its input clock".to_string())?;
        let take = available.min(remaining);
        for (channel, destination) in output.iter_mut().enumerate() {
            let segment: Vec<f32> = self.pending_model_rate[channel]
                .iter()
                .map(|sample| (*sample * 32_768.0) as f32)
                .collect();
            let enhanced = enhance_segment(&segment, self.model.as_ref())?;
            destination.extend(
                enhanced[source_start..source_start + take]
                    .iter()
                    .map(|sample| crate::sanitize_sample(*sample as f64 / 32_768.0)),
            );
        }
        self.model_output_frames = self
            .model_output_frames
            .checked_add(take)
            .ok_or_else(|| "MossFormer2 model output length overflow".to_string())?;
        self.windows_processed = self
            .windows_processed
            .checked_add(1)
            .ok_or_else(|| "MossFormer2 window count overflow".to_string())?;
        for pending in &mut self.pending_model_rate {
            pending.copy_within(STRIDE_SAMPLES..WINDOW_SAMPLES, 0);
            pending.truncate(WINDOW_SAMPLES - STRIDE_SAMPLES);
        }
        Ok(())
    }
}

pub(crate) fn streaming_state_bytes(
    processor_channels: usize,
    input_sample_rate: u32,
    input_channels: usize,
) -> Result<u64, crate::config::ConfigError> {
    use crate::config::{checked_resource_add, checked_resource_multiply, ConfigError};

    if processor_channels == 0
        || processor_channels > crate::config::MAX_STREAM_CHANNELS
        || input_channels == 0
        || input_channels > crate::config::MAX_STREAM_CHANNELS
    {
        return Err(ConfigError::invalid("channels", "an integer in 1..=64"));
    }
    let pending_samples = checked_resource_multiply(
        "MossFormer2 stream window",
        processor_channels as u64,
        WINDOW_SAMPLES as u64,
    )?;
    let pending_bytes = checked_resource_multiply(
        "MossFormer2 stream window",
        pending_samples,
        std::mem::size_of::<f64>() as u64,
    )?;
    let model_state = checked_resource_add(
        "MossFormer2 stream state",
        pending_bytes,
        STREAM_SCRATCH_ALLOWANCE_BYTES,
    )?;
    // The first result is emitted after one four-second model window. Reserve
    // the wrapper's possible stereo-link and VAD alignment queues as three
    // simultaneous input-rate f64 copies, even when those modes are disabled.
    let alignment_frames = checked_resource_multiply(
        "MossFormer2 stream alignment",
        u64::from(input_sample_rate),
        4,
    )?;
    let alignment_samples = checked_resource_multiply(
        "MossFormer2 stream alignment",
        alignment_frames,
        input_channels as u64,
    )?;
    let alignment_bytes = checked_resource_multiply(
        "MossFormer2 stream alignment",
        alignment_samples,
        3 * std::mem::size_of::<f64>() as u64,
    )?;
    checked_resource_add("MossFormer2 stream state", model_state, alignment_bytes)
}

fn validate_stream_block(input: &[Vec<f64>], channels: usize) -> Result<usize, String> {
    if input.len() != channels {
        return Err(format!(
            "MossFormer2 stream expected {channels} channels, received {}",
            input.len()
        ));
    }
    let frames = input.first().map_or(0, Vec::len);
    if input.iter().any(|channel| channel.len() != frames) {
        return Err("MossFormer2 stream channels must contain the same number of frames".into());
    }
    Ok(frames)
}

fn empty_stream_output(channels: usize, capacity: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| "unable to reserve MossFormer2 output channels".to_string())?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(capacity)
            .map_err(|_| "unable to reserve MossFormer2 output samples".to_string())?;
        output.push(channel);
    }
    Ok(output)
}

fn append_stream_output(
    destination: &mut [Vec<f64>],
    source: &[Vec<f64>],
    limit: usize,
) -> Result<(), String> {
    if destination.len() != source.len() {
        return Err("MossFormer2 stream output channel count changed".into());
    }
    let frames = source.first().map_or(0, Vec::len);
    if source.iter().any(|channel| channel.len() != frames) {
        return Err("MossFormer2 stream output channels became unaligned".into());
    }
    let retained = limit.saturating_sub(destination.first().map_or(0, Vec::len));
    let take = retained.min(frames);
    for (output, input) in destination.iter_mut().zip(source) {
        output.extend(input.iter().take(take).copied().map(crate::sanitize_sample));
    }
    Ok(())
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    model: &dyn tract_onnx::tract_core::runtime::Runnable,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let at_model_rate = crate::resample::resample(input, input_sample_rate, MODEL_RATE)?;
    let original_model_length = at_model_rate.len();
    let padded_length = segmentation_length(original_model_length);
    let mut padded = at_model_rate;
    padded.resize(padded_length, 0.0);
    let mut enhanced = vec![0.0f32; padded_length];

    let mut start = 0;
    while start + WINDOW_SAMPLES <= padded_length {
        let segment: Vec<f32> = padded[start..start + WINDOW_SAMPLES]
            .iter()
            .map(|sample| (*sample * 32_768.0) as f32)
            .collect();
        let output = enhance_segment(&segment, model)?;
        let (source_start, source_end, target_start) = if start == 0 {
            (0, WINDOW_SAMPLES - GIVE_UP_SAMPLES, start)
        } else {
            (
                GIVE_UP_SAMPLES,
                WINDOW_SAMPLES - GIVE_UP_SAMPLES,
                start + GIVE_UP_SAMPLES,
            )
        };
        enhanced[target_start..target_start + source_end - source_start]
            .copy_from_slice(&output[source_start..source_end]);
        start += STRIDE_SAMPLES;
    }

    let enhanced: Vec<f64> = enhanced[..original_model_length]
        .iter()
        .map(|sample| *sample as f64 / 32_768.0)
        .collect();
    let mut output = crate::resample::resample(&enhanced, MODEL_RATE, input_sample_rate)?;
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

fn segmentation_length(input_length: usize) -> usize {
    if input_length <= WINDOW_SAMPLES {
        return WINDOW_SAMPLES;
    }
    let extra = input_length - WINDOW_SAMPLES;
    WINDOW_SAMPLES + extra.div_ceil(STRIDE_SAMPLES) * STRIDE_SAMPLES
}

fn enhance_segment(
    samples: &[f32],
    model: &dyn tract_onnx::tract_core::runtime::Runnable,
) -> Result<Vec<f32>, String> {
    let features = fbank_with_deltas(samples)?;
    let mask = run_model(&features, model)?;
    let spectrum = stft(samples);
    let masked: Vec<Complex32> = spectrum
        .into_iter()
        .zip(mask)
        .map(|(value, gain)| value * gain)
        .collect();
    istft(&masked, samples.len())
}

fn fbank_with_deltas(samples: &[f32]) -> Result<Vec<f32>, String> {
    let frame_opts = FrameOptions {
        samp_freq: MODEL_RATE as f32,
        frame_shift_ms: 8.0,
        frame_length_ms: 40.0,
        // Upstream requests one PCM-unit of random dither. Deployment uses
        // zero dither so identical audio has deterministic model features.
        dither: 0.0,
        preemph_coeff: 0.97,
        remove_dc_offset: true,
        window_type: "hamming".into(),
        round_to_power_of_two: true,
        blackman_coeff: 0.42,
        snip_edges: true,
    };
    let mut mel_opts = MelOptions::default();
    mel_opts.num_bins = MEL_BINS;
    let options = FbankOptions {
        frame_opts,
        mel_opts,
        use_energy: false,
        raw_energy: true,
        htk_compat: false,
        energy_floor: 1.0,
        use_log_fbank: true,
        use_power: true,
    };
    let computer = FbankComputer::new(options)
        .map_err(|error| format!("MossFormer2 fbank setup failed: {error}"))?;
    let mut online =
        OnlineFeature::new(kaldi_native_fbank::online::FeatureComputer::Fbank(computer));
    online.accept_waveform(MODEL_RATE as f32, samples);
    online.input_finished();
    if online.num_frames_ready() != FRAMES {
        return Err(format!(
            "MossFormer2 frontend produced {} frames; expected {FRAMES}",
            online.num_frames_ready()
        ));
    }
    let base: Vec<f32> = online.features.into_iter().flatten().collect();
    let delta = deltas(&base, FRAMES, MEL_BINS);
    let delta_delta = deltas(&delta, FRAMES, MEL_BINS);
    let mut result = vec![0.0; FRAMES * FEATURES];
    for frame in 0..FRAMES {
        let output = &mut result[frame * FEATURES..(frame + 1) * FEATURES];
        output[..MEL_BINS].copy_from_slice(&base[frame * MEL_BINS..(frame + 1) * MEL_BINS]);
        output[MEL_BINS..2 * MEL_BINS]
            .copy_from_slice(&delta[frame * MEL_BINS..(frame + 1) * MEL_BINS]);
        output[2 * MEL_BINS..]
            .copy_from_slice(&delta_delta[frame * MEL_BINS..(frame + 1) * MEL_BINS]);
    }
    Ok(result)
}

fn deltas(input: &[f32], frames: usize, bins: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for frame in 0..frames {
        for bin in 0..bins {
            let mut numerator = 0.0;
            for distance in 1..=2 {
                let before = frame.saturating_sub(distance);
                let after = (frame + distance).min(frames - 1);
                numerator +=
                    distance as f32 * (input[after * bins + bin] - input[before * bins + bin]);
            }
            output[frame * bins + bin] = numerator / 10.0;
        }
    }
    output
}

fn symmetric_hamming() -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|index| {
            0.54 - 0.46 * (2.0 * std::f32::consts::PI * index as f32 / (FFT_SIZE - 1) as f32).cos()
        })
        .collect()
}

fn stft(input: &[f32]) -> Vec<Complex32> {
    let window = symmetric_hamming();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut output = vec![Complex32::default(); FRAMES * BINS];
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..FRAMES {
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            buffer[index] = Complex32::new(input[start + index] * window[index], 0.0);
        }
        fft.process(&mut buffer);
        output[frame * BINS..(frame + 1) * BINS].copy_from_slice(&buffer[..BINS]);
    }
    output
}

fn istft(spectrum: &[Complex32], output_length: usize) -> Result<Vec<f32>, String> {
    if spectrum.len() != FRAMES * BINS {
        return Err("MossFormer2 mask has an unexpected spectrum size".into());
    }
    let window = symmetric_hamming();
    let reconstructed_length = (FRAMES - 1) * HOP_SIZE + FFT_SIZE;
    let mut signal = vec![0.0f32; reconstructed_length];
    let mut envelope = vec![0.0f32; reconstructed_length];
    let mut planner = FftPlanner::new();
    let inverse = planner.plan_fft_inverse(FFT_SIZE);
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    for frame in 0..FRAMES {
        buffer[..BINS].copy_from_slice(&spectrum[frame * BINS..(frame + 1) * BINS]);
        for bin in BINS..FFT_SIZE {
            buffer[bin] = buffer[FFT_SIZE - bin].conj();
        }
        inverse.process(&mut buffer);
        let start = frame * HOP_SIZE;
        for index in 0..FFT_SIZE {
            signal[start + index] += buffer[index].re / FFT_SIZE as f32 * window[index];
            envelope[start + index] += window[index] * window[index];
        }
    }
    for (sample, weight) in signal.iter_mut().zip(envelope) {
        if weight > 1e-8 {
            *sample /= weight;
        }
    }
    signal.resize(output_length, 0.0);
    if signal.iter().any(|sample| !sample.is_finite()) {
        return Err("MossFormer2 reconstruction produced a non-finite sample".into());
    }
    Ok(signal)
}

fn load_model(
    config: &OnnxModelConfig,
    runtime: AcceleratorRuntime,
) -> Result<SharedRunnable, String> {
    let mut model = tract_onnx::onnx()
        .model_for_path(&config.path)
        .map_err(|error| model_error("load", error))?;
    if model
        .input_outlets()
        .map_err(|e| model_error("inspect", e))?
        .len()
        != 1
        || model
            .output_outlets()
            .map_err(|e| model_error("inspect", e))?
            .len()
            != 1
    {
        return Err("MossFormer2 ONNX model must have one input and one output".into());
    }
    model
        .set_input_fact(0, f32::fact(tvec!(1, FRAMES, FEATURES)).into())
        .map_err(|error| model_error("configure input", error))?;
    model
        .set_output_fact(0, f32::fact(tvec!(1, FRAMES, BINS)).into())
        .map_err(|error| model_error("configure output", error))?;
    let model = model
        .into_typed()
        .map_err(|error| model_error("type", error))?;
    super::tract_runtime::prepare(model, runtime, "MossFormer2 model")
}

fn run_model(
    features: &[f32],
    model: &dyn tract_onnx::tract_core::runtime::Runnable,
) -> Result<Vec<f32>, String> {
    let input = Tensor::from_shape(&[1, FRAMES, FEATURES], features)
        .map_err(|error| model_error("create feature tensor", error))?;
    let outputs = model
        .run(tvec!(input.into_tvalue()))
        .map_err(|error| model_error("run", error))?;
    let output = outputs[0]
        .to_plain_array_view::<f32>()
        .map_err(|error| model_error("read output", error))?;
    if output.len() != FRAMES * BINS || output.iter().any(|value| !value.is_finite()) {
        return Err("MossFormer2 model returned an invalid mask".into());
    }
    Ok(output.iter().copied().collect())
}

fn model_error(stage: &str, error: impl std::fmt::Display) -> String {
    format!("MossFormer2 ONNX {stage} failed: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use tract_onnx::pb::{
        attribute_proto, tensor_proto, tensor_shape_proto, type_proto, AttributeProto, GraphProto,
        ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto,
        ValueInfoProto,
    };

    #[test]
    fn official_window_has_496_frames() {
        assert_eq!(1 + (WINDOW_SAMPLES - FFT_SIZE) / HOP_SIZE, FRAMES);
    }

    #[test]
    fn segmentation_covers_the_requested_duration() {
        for length in [1, WINDOW_SAMPLES, WINDOW_SAMPLES + 1, 1_000_000] {
            let padded = segmentation_length(length);
            assert!(padded >= length);
            assert_eq!((padded - WINDOW_SAMPLES) % STRIDE_SAMPLES, 0);
        }
    }

    #[test]
    fn deltas_replicate_boundary_frames() {
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let actual = deltas(&input, 4, 1);
        assert_eq!(actual, vec![0.5, 0.8, 0.8, 0.5]);
    }

    #[test]
    fn stft_identity_reconstruction_is_transparent() {
        let input: Vec<f32> = (0..WINDOW_SAMPLES)
            .map(|index| (index as f32 * 0.013).sin())
            .collect();
        let output = istft(&stft(&input), input.len()).unwrap();
        let maximum = input
            .iter()
            .zip(output)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum < 2e-4, "maximum reconstruction error: {maximum}");
    }

    #[test]
    fn bounded_streaming_matches_offline_window_stitching_and_reset() {
        let mut bytes = Vec::new();
        constant_unity_mask_model().encode(&mut bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "denoize-mossformer-stream-{}-{}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        let config = OnnxModelConfig {
            path: path.clone(),
            sample_rate: MODEL_RATE,
        };
        let input: Vec<f64> = (0..WINDOW_SAMPLES + 777)
            .map(|index| {
                let time = index as f64 / MODEL_RATE as f64;
                0.12 * (std::f64::consts::TAU * 230.0 * time).sin()
            })
            .collect();
        let offline = process(&[input.clone()], MODEL_RATE, &config).unwrap();

        let run = |stream: &mut StreamingProcessor| {
            let mut output = Vec::new();
            for block in input.chunks(7_919) {
                output.extend_from_slice(&stream.process_block(&[block.to_vec()]).unwrap()[0]);
            }
            output.extend_from_slice(&stream.finish().unwrap()[0]);
            output
        };
        let mut stream = StreamingProcessor::new_with_accelerator(
            &config,
            MODEL_RATE,
            1,
            AcceleratorRuntime::Cpu,
        )
        .unwrap();
        let first = run(&mut stream);
        assert_eq!(first.len(), input.len());
        assert_eq!(first, offline[0]);

        stream.reset();
        assert_eq!(run(&mut stream), first);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn streaming_state_estimate_covers_window_scratch_and_alignment() {
        let mono = streaming_state_bytes(1, 48_000, 1).unwrap();
        let stereo = streaming_state_bytes(2, 48_000, 2).unwrap();
        assert!(mono > STREAM_SCRATCH_ALLOWANCE_BYTES);
        assert!(stereo > mono);
        assert!(streaming_state_bytes(0, 48_000, 1).is_err());
    }

    fn constant_unity_mask_model() -> ModelProto {
        let value_info = |name: &str, shape: Vec<i64>| ValueInfoProto {
            name: name.into(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: shape.into_iter().map(dimension_value).collect(),
                    }),
                })),
            }),
            doc_string: String::new(),
        };
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            producer_name: "denoize-test".into(),
            graph: Some(GraphProto {
                name: "mossformer-unity-mask".into(),
                node: vec![NodeProto {
                    output: vec!["mask".into()],
                    name: "constant-mask".into(),
                    op_type: "Constant".into(),
                    attribute: vec![AttributeProto {
                        name: "value".into(),
                        r#type: attribute_proto::AttributeType::Tensor as i32,
                        t: Some(TensorProto {
                            dims: vec![1, FRAMES as i64, BINS as i64],
                            data_type: tensor_proto::DataType::Float as i32,
                            float_data: vec![1.0; FRAMES * BINS],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                input: vec![value_info(
                    "features",
                    vec![1, FRAMES as i64, FEATURES as i64],
                )],
                output: vec![value_info("mask", vec![1, FRAMES as i64, BINS as i64])],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn dimension_value(value: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
            denotation: String::new(),
        }
    }
}
