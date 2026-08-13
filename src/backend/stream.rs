//! Stateful backend session shared by bounded file and realtime processing.

use std::borrow::Cow;
use std::collections::VecDeque;

use super::{Backend, BackendOptions, ChannelMode};
use crate::config::{ConfigError, MAX_STREAM_BLOCK_FRAMES, MAX_STREAM_CHANNELS};
use crate::{DenoiserConfig, StreamingDenoiser};

#[cfg(feature = "rnnoise")]
const RNNOISE_STATE_ALLOWANCE_PER_CHANNEL: u64 = 2 * 1024 * 1024;

/// A reusable stateful denoising session for continuous planar audio.
///
/// Supported backends retain their overlap, recurrent model, and resampler
/// state between calls. Model-backed sessions load and optimize their graph at
/// construction and never reopen it while processing or resetting the stream.
pub struct StreamingBackendSession {
    backend: Backend,
    input_channels: usize,
    processor_channels: usize,
    channel_mode: ChannelMode,
    denoiser: DenoiserConfig,
    processor: StreamingBackend,
    linked_original: VecDeque<(f64, f64)>,
    finished: bool,
}

enum StreamingBackend {
    Classical(StreamingDenoiser),
    #[cfg(feature = "rnnoise")]
    Rnnoise(Box<super::rnnoise::StreamingProcessor>),
    #[cfg(feature = "gtcrn")]
    Gtcrn(Box<super::gtcrn::StreamingProcessor>),
}

impl StreamingBackendSession {
    /// Return whether a compiled backend has a continuous stateful adapter.
    #[allow(unreachable_patterns)]
    pub fn supports(backend: Backend) -> bool {
        match backend {
            Backend::Classical => true,
            #[cfg(feature = "rnnoise")]
            Backend::Rnnoise => true,
            #[cfg(feature = "gtcrn")]
            Backend::Gtcrn => true,
            _ => false,
        }
    }

    /// Construct a stream and allocate every backend state before accepting
    /// audio. `backend_options` must already contain any managed model path.
    pub fn new(
        backend: Backend,
        sample_rate: u32,
        channels: usize,
        mut denoiser: DenoiserConfig,
        backend_options: BackendOptions,
    ) -> Result<Self, String> {
        if channels == 0 || channels > MAX_STREAM_CHANNELS {
            return Err(format!(
                "streaming backend channels must be between 1 and {MAX_STREAM_CHANNELS}"
            ));
        }
        if !Self::supports(backend) {
            return Err("selected backend does not support stateful streaming".into());
        }
        denoiser.sample_rate = sample_rate;
        denoiser
            .validate_config()
            .map_err(|error| error.to_string())?;
        backend_options.validate_resolved_resources(backend)?;
        let stereo_mode = channels == 2 && backend_options.channel_mode != ChannelMode::Independent;
        let processor_channels =
            if stereo_mode && backend_options.channel_mode == ChannelMode::StereoLinked {
                1
            } else {
                channels
            };
        let _ = (sample_rate, processor_channels);
        let processor = Self::build_processor(
            backend,
            sample_rate,
            processor_channels,
            &denoiser,
            &backend_options,
        )?;
        Ok(Self {
            backend,
            input_channels: channels,
            processor_channels,
            channel_mode: if stereo_mode {
                backend_options.channel_mode
            } else {
                ChannelMode::Independent
            },
            denoiser,
            processor,
            linked_original: VecDeque::new(),
            finished: false,
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Conservative backend-specific state beyond the classical stream and
    /// caller-owned input/output blocks.
    pub fn estimate_additional_bytes(
        backend: Backend,
        sample_rate: u32,
        channels: usize,
        channel_mode: ChannelMode,
    ) -> Result<u64, ConfigError> {
        if channels == 0 || channels > MAX_STREAM_CHANNELS {
            return Err(ConfigError::invalid("channels", "an integer in 1..=64"));
        }
        let processor_channels = if channels == 2 && channel_mode == ChannelMode::StereoLinked {
            1
        } else {
            channels
        };
        let _ = (sample_rate, processor_channels);
        match backend {
            Backend::Classical => Ok(0),
            #[cfg(feature = "rnnoise")]
            Backend::Rnnoise => {
                let resamplers = resampler_pair_bytes(
                    processor_channels,
                    sample_rate,
                    48_000,
                    "RNNoise stream resamplers",
                )?;
                let state = u64::try_from(processor_channels)
                    .ok()
                    .and_then(|channels| channels.checked_mul(RNNOISE_STATE_ALLOWANCE_PER_CHANNEL))
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "RNNoise stream state",
                    })?;
                resamplers
                    .checked_add(state)
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "RNNoise stream state",
                    })
            }
            #[cfg(feature = "gtcrn")]
            Backend::Gtcrn => {
                let resamplers = resampler_pair_bytes(
                    processor_channels,
                    sample_rate,
                    super::gtcrn::SAMPLE_RATE,
                    "GTCRN stream resamplers",
                )?;
                resamplers
                    .checked_add(super::gtcrn::streaming_state_bytes(processor_channels)?)
                    .ok_or(ConfigError::ResourceOverflow {
                        resource: "GTCRN stream state",
                    })
            }
            #[allow(unreachable_patterns)]
            _ => Err(ConfigError::invalid(
                "backend",
                "a compiled backend with stateful streaming support",
            )),
        }
    }

    /// Process a block. The returned block can be empty while bounded model or
    /// sample-rate-converter latency is retained.
    pub fn process_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("streaming backend session has already been finished".into());
        }
        self.process_block_with_limit(channels, MAX_STREAM_BLOCK_FRAMES)
    }

    /// Flush every pending model frame and converter delay exactly once.
    pub fn finish(&mut self) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("streaming backend session has already been finished".into());
        }
        let processed = match &mut self.processor {
            StreamingBackend::Classical(processor) => processor.finish(),
            #[cfg(feature = "rnnoise")]
            StreamingBackend::Rnnoise(processor) => processor.finish(),
            #[cfg(feature = "gtcrn")]
            StreamingBackend::Gtcrn(processor) => processor.finish(),
        }?;
        let output = self.restore_channel_mode(processed)?;
        if self.channel_mode == ChannelMode::StereoLinked && !self.linked_original.is_empty() {
            return Err("linked streaming backend did not flush every input frame".into());
        }
        self.finished = true;
        Ok(output)
    }

    /// Start an independent stream while retaining any already-loaded model.
    pub fn reset(&mut self) -> Result<(), String> {
        match &mut self.processor {
            StreamingBackend::Classical(processor) => {
                let replacement =
                    StreamingDenoiser::new(self.denoiser.clone(), self.processor_channels)?;
                *processor = replacement;
            }
            #[cfg(feature = "rnnoise")]
            StreamingBackend::Rnnoise(processor) => processor.reset(),
            #[cfg(feature = "gtcrn")]
            StreamingBackend::Gtcrn(processor) => processor.reset(),
        }
        self.linked_original.clear();
        self.finished = false;
        Ok(())
    }

    fn build_processor(
        backend: Backend,
        sample_rate: u32,
        channels: usize,
        denoiser: &DenoiserConfig,
        backend_options: &BackendOptions,
    ) -> Result<StreamingBackend, String> {
        let _ = (sample_rate, backend_options);
        match backend {
            Backend::Classical => Ok(StreamingBackend::Classical(StreamingDenoiser::new(
                denoiser.clone(),
                channels,
            )?)),
            #[cfg(feature = "rnnoise")]
            Backend::Rnnoise => Ok(StreamingBackend::Rnnoise(Box::new(
                super::rnnoise::StreamingProcessor::new(sample_rate, channels)?,
            ))),
            #[cfg(feature = "gtcrn")]
            Backend::Gtcrn => {
                let model = backend_options
                    .onnx
                    .as_ref()
                    .ok_or_else(|| "GTCRN streaming requires the managed ONNX model".to_string())?;
                Ok(StreamingBackend::Gtcrn(Box::new(
                    super::gtcrn::StreamingProcessor::new(model, sample_rate, channels)?,
                )))
            }
            #[allow(unreachable_patterns)]
            _ => Err("selected backend does not support stateful streaming".into()),
        }
    }

    pub(crate) fn process_block_with_limit(
        &mut self,
        channels: &[Vec<f64>],
        block_limit: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        validate_block(channels, self.input_channels)?;
        if block_limit == 0 {
            return Err("streaming backend block limit must be positive".into());
        }
        let frames = channels.first().map(Vec::len).unwrap_or(0);
        if frames <= block_limit {
            return self.process_bounded_block(channels);
        }

        let mut output = empty_channels(self.input_channels)?;
        let mut position = 0usize;
        while position < frames {
            let end = position.saturating_add(block_limit).min(frames);
            let block = clone_range(channels, position, end)?;
            let ready = self.process_bounded_block(&block)?;
            append_channels(&mut output, &ready, self.input_channels)?;
            position = end;
        }
        Ok(output)
    }

    fn process_bounded_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let backend_input = match self.channel_mode {
            ChannelMode::Independent => Cow::Borrowed(channels),
            ChannelMode::StereoLinked => {
                let frames = channels[0].len();
                self.linked_original.try_reserve(frames).map_err(|_| {
                    ConfigError::allocation_failed("linked stream alignment").to_string()
                })?;
                let mut mid = Vec::new();
                mid.try_reserve_exact(frames)
                    .map_err(|_| ConfigError::allocation_failed("linked stream mid").to_string())?;
                for (&left, &right) in channels[0].iter().zip(&channels[1]) {
                    let left = crate::audio::sanitize_sample(left);
                    let right = crate::audio::sanitize_sample(right);
                    mid.push((left + right) * 0.5);
                    self.linked_original.push_back((left, right));
                }
                Cow::Owned(vec![mid])
            }
            ChannelMode::MidSide => {
                let (mid, side) = super::encode_mid_side(&channels[0], &channels[1])?;
                Cow::Owned(vec![mid, side])
            }
        };
        let processed = match &mut self.processor {
            StreamingBackend::Classical(processor) => processor.process_block(&backend_input),
            #[cfg(feature = "rnnoise")]
            StreamingBackend::Rnnoise(processor) => processor.process_block(&backend_input),
            #[cfg(feature = "gtcrn")]
            StreamingBackend::Gtcrn(processor) => processor.process_block(&backend_input),
        }?;
        self.restore_channel_mode(processed)
    }

    fn restore_channel_mode(
        &mut self,
        mut processed: Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, String> {
        match self.channel_mode {
            ChannelMode::Independent => {
                validate_block(&processed, self.input_channels)?;
                Ok(processed)
            }
            ChannelMode::StereoLinked => {
                if processed.len() != 1 {
                    return Err("linked streaming backend must return one channel".into());
                }
                let enhanced = processed.pop().unwrap_or_default();
                if enhanced.len() > self.linked_original.len() {
                    return Err("linked streaming backend returned unaligned frames".into());
                }
                let mut left = Vec::new();
                let mut right = Vec::new();
                left.try_reserve_exact(enhanced.len()).map_err(|_| {
                    ConfigError::allocation_failed("linked stream output").to_string()
                })?;
                right.try_reserve_exact(enhanced.len()).map_err(|_| {
                    ConfigError::allocation_failed("linked stream output").to_string()
                })?;
                for clean in enhanced {
                    let (original_left, original_right) = self
                        .linked_original
                        .pop_front()
                        .ok_or_else(|| "linked streaming alignment queue underflow".to_string())?;
                    let original_mid = (original_left + original_right) * 0.5;
                    let correction = clean - original_mid;
                    left.push(crate::audio::sanitize_sample(original_left + correction));
                    right.push(crate::audio::sanitize_sample(original_right + correction));
                }
                Ok(vec![left, right])
            }
            ChannelMode::MidSide => {
                if processed.len() != 2 {
                    return Err("mid-side streaming backend must return two channels".into());
                }
                let (left, right) = super::decode_mid_side(&processed[0], &processed[1])?;
                Ok(vec![left, right])
            }
        }
    }
}

#[cfg(any(feature = "rnnoise", feature = "gtcrn"))]
fn resampler_pair_bytes(
    channels: usize,
    source_rate: u32,
    model_rate: u32,
    resource: &'static str,
) -> Result<u64, ConfigError> {
    let forward = crate::resample::resampler_plan_bytes(channels, source_rate, model_rate)
        .map_err(|_| ConfigError::invalid("sample_rate", "a bounded resampler plan"))?;
    let reverse = crate::resample::resampler_plan_bytes(channels, model_rate, source_rate)
        .map_err(|_| ConfigError::invalid("sample_rate", "a bounded resampler plan"))?;
    forward
        .checked_add(reverse)
        .ok_or(ConfigError::ResourceOverflow { resource })
}

fn validate_block(channels: &[Vec<f64>], expected_channels: usize) -> Result<usize, String> {
    if channels.len() != expected_channels {
        return Err(format!(
            "expected {expected_channels} streaming channels, got {}",
            channels.len()
        ));
    }
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    if channels.iter().any(|channel| channel.len() != frames) {
        return Err("streaming blocks must have equal channel lengths".into());
    }
    Ok(frames)
}

fn empty_channels(channels: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| ConfigError::allocation_failed("stream output channels").to_string())?;
    for _ in 0..channels {
        output.push(Vec::new());
    }
    Ok(output)
}

fn clone_range(channels: &[Vec<f64>], start: usize, end: usize) -> Result<Vec<Vec<f64>>, String> {
    let frames = end.checked_sub(start).ok_or_else(|| {
        ConfigError::ResourceOverflow {
            resource: "stream split block",
        }
        .to_string()
    })?;
    let mut block = Vec::new();
    block
        .try_reserve_exact(channels.len())
        .map_err(|_| ConfigError::allocation_failed("stream split channels").to_string())?;
    for channel in channels {
        let mut split = Vec::new();
        split
            .try_reserve_exact(frames)
            .map_err(|_| ConfigError::allocation_failed("stream split samples").to_string())?;
        split.extend_from_slice(&channel[start..end]);
        block.push(split);
    }
    Ok(block)
}

fn append_channels(
    output: &mut [Vec<f64>],
    block: &[Vec<f64>],
    expected_channels: usize,
) -> Result<(), String> {
    validate_block(block, expected_channels)?;
    if output.len() != expected_channels {
        return Err("stream split output channel count changed".into());
    }
    for (output, block) in output.iter_mut().zip(block) {
        output
            .try_reserve_exact(block.len())
            .map_err(|_| ConfigError::allocation_failed("stream split output").to_string())?;
    }
    for (output, block) in output.iter_mut().zip(block) {
        output.extend_from_slice(block);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classical_stream_preserves_channel_geometry() {
        let mut config = DenoiserConfig::default(48_000);
        config.profile_ms = -1.0;
        let mut session = StreamingBackendSession::new(
            Backend::Classical,
            48_000,
            2,
            config,
            BackendOptions::default(),
        )
        .unwrap();
        let input = vec![vec![0.1; 2048], vec![-0.1; 2048]];
        let mut output = session.process_block(&input).unwrap();
        let tail = session.finish().unwrap();
        for (channel, tail) in output.iter_mut().zip(tail) {
            channel.extend(tail);
        }
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|channel| channel.len() == 2048));
    }

    #[test]
    fn unsupported_compiled_backend_is_rejected() {
        #[cfg(feature = "onnx")]
        assert!(!StreamingBackendSession::supports(Backend::Onnx));
    }
}
