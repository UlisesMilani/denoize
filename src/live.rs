//! Realtime system-audio capture, denoising, and playback.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, Sample, SampleFormat, Stream, StreamConfig};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::audio::Audio;
#[cfg(test)]
use crate::config::MAX_STREAM_BLOCK_FRAMES;
use crate::config::{
    checked_profile_target_samples, checked_resource_add, checked_resource_multiply, ConfigError,
    MAX_SAMPLE_RATE, MAX_STREAM_CHANNELS, MAX_STREAM_STATE_BYTES,
};
use crate::denoiser::DenoiserConfig;
use crate::{
    denoise_audio_with_backend_config, select_accelerator_for_options, AcceleratorSelection,
    Backend, BackendOptions, ChannelMode, ResourcePlan, StreamingBackendSession,
};

const MIN_CHUNK_MS: u32 = 10;
const MAX_CHUNK_MS: u32 = 2_000;
const MIN_TARGET_LATENCY_MS: u32 = 20;
const MAX_TARGET_LATENCY_MS: u32 = 5_000;
const MAX_DRIFT_PPM: u32 = 10_000;
const MAX_RECONNECT_TIMEOUT_MS: u32 = 300_000;
const DEFAULT_MAX_DRIFT_PPM: u32 = 2_500;
const DEFAULT_RECONNECT_TIMEOUT_MS: u32 = 30_000;
const RECONNECT_INITIAL_BACKOFF_MS: u64 = 100;
const RECONNECT_MAX_BACKOFF_MS: u64 = 2_000;
const LIVE_STATUS_INTERVAL: Duration = Duration::from_millis(100);
const PRIME_TIMEOUT: Duration = Duration::from_secs(10);
const ASYNC_SINC_LEN: usize = 128;
const ASYNC_SINC_OVERSAMPLING: usize = 128;
const CAPTURE_QUEUE_CHUNKS: usize = 4;
// The callback can simultaneously retain its full pending chunk and the next
// freshly allocated chunk while the bounded channel retains four more and the
// worker owns the chunk it is currently processing.
const CAPTURE_PIPELINE_CHUNKS: u64 = CAPTURE_QUEUE_CHUNKS as u64 + 3;
// Transactional processing and channel transforms can retain the input,
// sanitized working input, processed output, and channel-mode scratch.
const BASE_WORKER_AUDIO_COPIES: u64 = 9;
// VAD additionally retains its attenuated output and a region input/output.
const VAD_WORKER_AUDIO_COPIES: u64 = 12;
const PLAYBACK_QUEUE_CHUNKS: u64 = 8;
#[cfg(feature = "rnnoise")]
const RNNOISE_SAMPLE_RATE: u64 = 48_000;
#[cfg(feature = "rnnoise")]
const RNNOISE_FRAME_FRAMES: u64 = 480;
#[cfg(any(feature = "rnnoise", feature = "gtcrn"))]
const STREAM_RESAMPLER_CHUNK_FRAMES: u64 = 1_024;
#[cfg(any(feature = "rnnoise", feature = "gtcrn"))]
const STREAM_RESAMPLER_SUB_CHUNKS: u64 = 2;
#[cfg(feature = "gtcrn")]
const GTCRN_SAMPLE_RATE: u64 = crate::backend::gtcrn::SAMPLE_RATE as u64;
#[cfg(feature = "gtcrn")]
const GTCRN_HOP_FRAMES: u64 = crate::backend::gtcrn::HOP_SIZE as u64;

static CTRL_C_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();
static CTRL_C_SESSION: OnceLock<Mutex<Option<Weak<AtomicBool>>>> = OnceLock::new();

/// Settings for a realtime capture-to-playback session.
#[derive(Clone, Debug)]
pub struct LiveConfig {
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub chunk_ms: u32,
    pub backend: Backend,
    pub backend_options: BackendOptions,
    pub denoiser: DenoiserConfig,
}

/// Runtime policy for independent capture/playback clocks and device recovery.
///
/// A zero target latency selects a chunk-aware default of two capture chunks,
/// with a 40 ms minimum. A zero reconnect timeout disables automatic hotplug
/// recovery. Clock conversion remains active when drift correction is disabled
/// with `max_drift_ppm == 0`, so devices may still use different nominal rates.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveResilienceConfig {
    pub target_latency_ms: u32,
    pub max_drift_ppm: u32,
    pub reconnect_timeout_ms: u32,
}

impl Default for LiveResilienceConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 0,
            max_drift_ppm: DEFAULT_MAX_DRIFT_PPM,
            reconnect_timeout_ms: DEFAULT_RECONNECT_TIMEOUT_MS,
        }
    }
}

impl LiveResilienceConfig {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target_latency_ms: 0,
            max_drift_ppm: DEFAULT_MAX_DRIFT_PPM,
            reconnect_timeout_ms: DEFAULT_RECONNECT_TIMEOUT_MS,
        }
    }

    #[must_use]
    pub const fn with_target_latency_ms(mut self, target_latency_ms: u32) -> Self {
        self.target_latency_ms = target_latency_ms;
        self
    }

    #[must_use]
    pub const fn with_max_drift_ppm(mut self, max_drift_ppm: u32) -> Self {
        self.max_drift_ppm = max_drift_ppm;
        self
    }

    #[must_use]
    pub const fn with_reconnect_timeout_ms(mut self, reconnect_timeout_ms: u32) -> Self {
        self.reconnect_timeout_ms = reconnect_timeout_ms;
        self
    }

    fn validate(self) -> Result<(), ConfigError> {
        if self.target_latency_ms != 0
            && !(MIN_TARGET_LATENCY_MS..=MAX_TARGET_LATENCY_MS).contains(&self.target_latency_ms)
        {
            return Err(ConfigError::invalid(
                "target_latency_ms",
                "zero for automatic or an integer in 20..=5000 ms",
            ));
        }
        if self.max_drift_ppm > MAX_DRIFT_PPM {
            return Err(ConfigError::invalid(
                "max_drift_ppm",
                "an integer in 0..=10000 ppm",
            ));
        }
        if self.reconnect_timeout_ms > MAX_RECONNECT_TIMEOUT_MS {
            return Err(ConfigError::invalid(
                "reconnect_timeout_ms",
                "an integer in 0..=300000 ms",
            ));
        }
        Ok(())
    }

    fn resolved_target_latency_ms(self, chunk_ms: u32) -> Result<u32, ConfigError> {
        self.validate()?;
        if self.target_latency_ms != 0 {
            return Ok(self.target_latency_ms);
        }
        chunk_ms
            .checked_mul(2)
            .map(|latency| latency.max(40).min(MAX_TARGET_LATENCY_MS))
            .ok_or(ConfigError::ResourceOverflow {
                resource: "live target latency",
            })
    }
}

impl LiveConfig {
    /// Validate caller-controlled values without enumerating audio hardware or
    /// inspecting a model path. The denoiser's sample rate is deliberately
    /// replaced with a safe placeholder because live hardware supplies the
    /// effective rate after device selection.
    pub fn validate_config(&self) -> Result<(), ConfigError> {
        if !(MIN_CHUNK_MS..=MAX_CHUNK_MS).contains(&self.chunk_ms) {
            return Err(ConfigError::invalid(
                "chunk_ms",
                "an integer in 10..=2000 ms",
            ));
        }
        let mut denoiser = self.denoiser.clone();
        denoiser.sample_rate = 1;
        denoiser.validate_config()?;
        if !backend_is_live_capable(self.backend) {
            return Err(ConfigError::invalid(
                "backend",
                "a compiled backend with stateful realtime support",
            ));
        }
        #[cfg(feature = "gtcrn")]
        if self.backend == Backend::Gtcrn && self.denoiser.vad {
            return Err(ConfigError::invalid(
                "vad",
                "disabled for the causal GTCRN realtime backend",
            ));
        }
        self.backend_options.validate_config(self.backend)
    }
}

/// Whether a backend has a bounded, low-latency implementation suitable for
/// the current live capture worker.
#[allow(unreachable_patterns)]
pub fn backend_is_live_capable(backend: Backend) -> bool {
    match backend {
        Backend::Classical => true,
        #[cfg(feature = "rnnoise")]
        Backend::Rnnoise => true,
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => true,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveBufferPlan {
    chunk_frames: usize,
    input_capacity: usize,
    queue_capacity: usize,
    target_queue_frames: usize,
    maximum_resampled_frames: usize,
    resampler_delay_frames: usize,
    backend_delay_frames: usize,
    required_bytes: u64,
}

fn maximum_ready_burst_frames(
    config: &LiveConfig,
    sample_rate: u32,
    chunk_frames: u64,
) -> Result<u64, ConfigError> {
    // The compatibility path returns exactly one input chunk. Stateful paths
    // can release samples retained by profiling, frame quantization, model
    // framing, and sample-rate conversion together with the current chunk.
    if config.denoiser.vad {
        return Ok(chunk_frames);
    }
    match config.backend {
        Backend::Classical => {
            // Live automatic profiling starts the adaptive estimator
            // immediately, so only explicit positive profiles retain a
            // prefix. Once initialized, at most one classical FFT frame of
            // original samples can remain unreturned between calls.
            let profile_target = if config.denoiser.profile_ms > 0.0 {
                u64::try_from(checked_profile_target_samples(
                    config.denoiser.profile_ms,
                    sample_rate,
                    config.denoiser.frame_size,
                )?)
                .map_err(|_| ConfigError::ResourceOverflow {
                    resource: "live classical ready burst",
                })?
            } else {
                0
            };
            let backlog = profile_target.max(config.denoiser.frame_size as u64);
            checked_resource_add("live classical ready burst", chunk_frames, backlog)
        }
        #[cfg(feature = "rnnoise")]
        Backend::Rnnoise => {
            let first_src_debt = stream_resampler_output_debt(sample_rate as u64, 48_000)?;
            let model_debt = RNNOISE_FRAME_FRAMES - 1;
            let debt_at_48k =
                checked_resource_add("live RNNoise ready burst", first_src_debt, model_debt)?;
            let upstream_debt = checked_ceil_scale(
                "live RNNoise ready burst",
                debt_at_48k,
                sample_rate as u64,
                RNNOISE_SAMPLE_RATE,
            )?;
            let second_src_debt =
                stream_resampler_output_debt(RNNOISE_SAMPLE_RATE, sample_rate as u64)?;
            let backlog =
                checked_resource_add("live RNNoise ready burst", upstream_debt, second_src_debt)?;
            checked_resource_add("live RNNoise ready burst", chunk_frames, backlog)
        }
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => {
            let first_src_debt =
                stream_resampler_output_debt(sample_rate as u64, GTCRN_SAMPLE_RATE)?;
            // A partial input hop and the causal WOLA hop can be retained at
            // the same time before the first aligned output is available.
            let model_debt = GTCRN_HOP_FRAMES
                .checked_mul(2)
                .and_then(|frames| frames.checked_sub(1))
                .ok_or(ConfigError::ResourceOverflow {
                    resource: "live GTCRN ready burst",
                })?;
            let debt_at_model_rate =
                checked_resource_add("live GTCRN ready burst", first_src_debt, model_debt)?;
            let upstream_debt = checked_ceil_scale(
                "live GTCRN ready burst",
                debt_at_model_rate,
                sample_rate as u64,
                GTCRN_SAMPLE_RATE,
            )?;
            let second_src_debt =
                stream_resampler_output_debt(GTCRN_SAMPLE_RATE, sample_rate as u64)?;
            let backlog =
                checked_resource_add("live GTCRN ready burst", upstream_debt, second_src_debt)?;
            checked_resource_add("live GTCRN ready burst", chunk_frames, backlog)
        }
        #[allow(unreachable_patterns)]
        _ => Err(ConfigError::invalid(
            "backend",
            "a compiled backend with stateful realtime support",
        )),
    }
}

#[cfg(any(feature = "rnnoise", feature = "gtcrn"))]
fn stream_resampler_output_debt(from_rate: u64, to_rate: u64) -> Result<u64, ConfigError> {
    if from_rate == to_rate {
        return Ok(0);
    }
    let gcd = greatest_common_divisor(from_rate, to_rate);
    let minimum_input_chunk = from_rate / gcd;
    let wanted_subchunk = STREAM_RESAMPLER_CHUNK_FRAMES / STREAM_RESAMPLER_SUB_CHUNKS;
    let fft_chunks = checked_ceil_div(
        "live RNNoise resampler quantum",
        wanted_subchunk,
        minimum_input_chunk,
    )?;
    let fft_size_in = checked_resource_multiply(
        "live RNNoise resampler quantum",
        fft_chunks,
        from_rate / gcd,
    )?;
    let fft_size_out =
        checked_resource_multiply("live RNNoise resampler quantum", fft_chunks, to_rate / gcd)?;
    let external_pending = STREAM_RESAMPLER_CHUNK_FRAMES - 1;
    let internal_pending = fft_size_in - 1;
    let held_input = checked_resource_add(
        "live RNNoise resampler quantum",
        external_pending,
        internal_pending,
    )?;
    let held_output = checked_ceil_scale(
        "live RNNoise resampler quantum",
        held_input,
        to_rate,
        from_rate,
    )?;
    let filter_delay = fft_size_out / 2;
    // One frame covers each nearest-rounded stream clock boundary.
    let delayed =
        checked_resource_add("live RNNoise resampler quantum", held_output, filter_delay)?;
    checked_resource_add("live RNNoise resampler quantum", delayed, 2)
}

#[cfg(any(feature = "rnnoise", feature = "gtcrn"))]
fn checked_ceil_scale(
    resource: &'static str,
    value: u64,
    numerator: u64,
    denominator: u64,
) -> Result<u64, ConfigError> {
    let product = checked_resource_multiply(resource, value, numerator)?;
    checked_ceil_div(resource, product, denominator)
}

fn checked_ceil_div(
    resource: &'static str,
    numerator: u64,
    denominator: u64,
) -> Result<u64, ConfigError> {
    let adjusted = checked_resource_add(resource, numerator, denominator - 1)?;
    Ok(adjusted / denominator)
}

#[cfg(any(feature = "rnnoise", feature = "gtcrn"))]
fn greatest_common_divisor(mut lhs: u64, mut rhs: u64) -> u64 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn plan_live_buffers_with_resilience(
    config: &LiveConfig,
    resilience: LiveResilienceConfig,
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_channels: usize,
    output_channels: usize,
) -> Result<LiveBufferPlan, ConfigError> {
    config.validate_config()?;
    resilience.validate()?;
    if output_channels == 0 || output_channels > MAX_STREAM_CHANNELS {
        return Err(ConfigError::invalid(
            "output_channels",
            "an integer in 1..=64",
        ));
    }
    let mut denoiser = config.denoiser.clone();
    denoiser.sample_rate = output_sample_rate;
    denoiser.validate_config()?;
    if input_sample_rate == 0 || input_sample_rate > MAX_SAMPLE_RATE {
        return Err(ConfigError::invalid(
            "input_sample_rate",
            "an integer in 1..=768000 Hz",
        ));
    }
    let backend_additional_bytes = StreamingBackendSession::estimate_additional_bytes(
        config.backend,
        output_sample_rate,
        input_channels,
        config.backend_options.channel_mode,
    )?;
    let processor = ResourcePlan::for_stream(
        input_channels,
        denoiser.frame_size,
        output_sample_rate,
        denoiser.profile_ms,
    )?;

    let chunk_numerator = checked_resource_multiply(
        "live chunk frames",
        input_sample_rate as u64,
        config.chunk_ms as u64,
    )?;
    let chunk_frames_u64 = (chunk_numerator / 1_000).max(1);
    let nominal_output = checked_resource_multiply(
        "live asynchronous resampler",
        chunk_frames_u64,
        output_sample_rate as u64,
    )?;
    let drifted_output =
        checked_resource_multiply("live asynchronous resampler", nominal_output, 1_000_000)?;
    let drift_denominator = 1_000_000u64
        .checked_sub(resilience.max_drift_ppm as u64)
        .ok_or(ConfigError::ResourceOverflow {
            resource: "live asynchronous resampler",
        })?;
    let resample_denominator = checked_resource_multiply(
        "live asynchronous resampler",
        input_sample_rate as u64,
        drift_denominator,
    )?;
    let maximum_resampled_frames = checked_resource_add(
        "live asynchronous resampler",
        checked_ceil_div(
            "live asynchronous resampler",
            drifted_output,
            resample_denominator,
        )?,
        10,
    )?;
    let ready_burst_frames =
        maximum_ready_burst_frames(config, output_sample_rate, maximum_resampled_frames)?;
    let input_samples =
        checked_resource_multiply("live input buffer", chunk_frames_u64, input_channels as u64)?;
    let target_latency_ms = resilience.resolved_target_latency_ms(config.chunk_ms)?;
    let target_numerator = checked_resource_multiply(
        "live target latency",
        output_sample_rate as u64,
        target_latency_ms as u64,
    )?;
    let target_queue_frames = checked_ceil_div("live target latency", target_numerator, 1_000)?;
    let scheduling_queue_frames = checked_resource_multiply(
        "live playback queue",
        maximum_resampled_frames,
        PLAYBACK_QUEUE_CHUNKS,
    )?;
    let target_headroom = checked_resource_multiply("live playback queue", target_queue_frames, 2)?;
    let steady_queue_frames = scheduling_queue_frames.max(target_headroom);
    // Keep the historical eight-chunk scheduling cushion plus one complete
    // stateful release. The latter can be much larger than an input chunk.
    let queue_frames = checked_resource_add(
        "live playback queue",
        steady_queue_frames,
        ready_burst_frames,
    )?;
    let queue_samples =
        checked_resource_multiply("live playback queue", queue_frames, output_channels as u64)?;

    let captured_chunk_bytes = checked_resource_multiply(
        "live working set",
        input_samples,
        std::mem::size_of::<f32>() as u64,
    )?;
    let captured_bytes = checked_resource_multiply(
        "live working set",
        captured_chunk_bytes,
        CAPTURE_PIPELINE_CHUNKS,
    )?;
    let input_planar_bytes = checked_resource_multiply(
        "live working set",
        input_samples,
        std::mem::size_of::<f64>() as u64,
    )?;
    let processed_samples = checked_resource_multiply(
        "live working set",
        maximum_resampled_frames,
        input_channels as u64,
    )?;
    let worker_chunk_bytes = checked_resource_multiply(
        "live working set",
        processed_samples,
        std::mem::size_of::<f64>() as u64,
    )?;
    let worker_audio_copies = if denoiser.vad {
        VAD_WORKER_AUDIO_COPIES
    } else {
        BASE_WORKER_AUDIO_COPIES
    };
    // One of the historical chunk-sized copies is the processed output. Size
    // that copy by its real maximum burst and retain the other conservative
    // chunk copies unchanged.
    let regular_worker_bytes = checked_resource_multiply(
        "live working set",
        worker_chunk_bytes,
        worker_audio_copies - 1,
    )?;
    let ready_samples = checked_resource_multiply(
        "live working set",
        ready_burst_frames,
        input_channels as u64,
    )?;
    let ready_bytes = checked_resource_multiply(
        "live working set",
        ready_samples,
        std::mem::size_of::<f64>() as u64,
    )?;
    let backend_worker_bytes =
        checked_resource_add("live working set", regular_worker_bytes, ready_bytes)?;
    let worker_bytes =
        checked_resource_add("live working set", input_planar_bytes, backend_worker_bytes)?;
    let linked_alignment_bytes = if !denoiser.vad
        && input_channels == 2
        && config.backend_options.channel_mode == ChannelMode::StereoLinked
    {
        let retained_frames = ready_burst_frames
            .checked_sub(maximum_resampled_frames)
            .ok_or(ConfigError::ResourceOverflow {
                resource: "live linked alignment",
            })?;
        let retained_samples = checked_resource_multiply(
            "live linked alignment",
            retained_frames,
            input_channels as u64,
        )?;
        checked_resource_multiply(
            "live linked alignment",
            retained_samples,
            std::mem::size_of::<f64>() as u64,
        )?
    } else {
        0
    };
    let playback_bytes = checked_resource_multiply(
        "live working set",
        queue_samples,
        std::mem::size_of::<f32>() as u64,
    )?;
    let async_resampler_bytes =
        async_resampler_plan_bytes(chunk_frames_u64, maximum_resampled_frames, input_channels)?;
    let worker_and_resampler =
        checked_resource_add("live working set", worker_bytes, async_resampler_bytes)?;
    let worker_and_alignment = checked_resource_add(
        "live working set",
        worker_and_resampler,
        linked_alignment_bytes,
    )?;
    let input_bytes =
        checked_resource_add("live working set", captured_bytes, worker_and_alignment)?;
    let buffer_bytes = checked_resource_add("live working set", input_bytes, playback_bytes)?;
    let stream_and_buffers = checked_resource_add(
        "live working set",
        processor.estimated_bytes(),
        buffer_bytes,
    )?;
    let required_bytes = checked_resource_add(
        "live working set",
        stream_and_buffers,
        backend_additional_bytes,
    )?;
    if required_bytes > MAX_STREAM_STATE_BYTES {
        return Err(ConfigError::ResourceLimitExceeded {
            resource: "live working set",
            required_bytes,
            limit_bytes: MAX_STREAM_STATE_BYTES,
        });
    }

    let resampler_delay_numerator = checked_resource_multiply(
        "live resampler latency",
        (ASYNC_SINC_LEN / 2) as u64,
        output_sample_rate as u64,
    )?;
    let resampler_delay_frames = checked_ceil_div(
        "live resampler latency",
        resampler_delay_numerator,
        input_sample_rate as u64,
    )?;
    let backend_delay_frames = ready_burst_frames
        .checked_sub(maximum_resampled_frames)
        .ok_or(ConfigError::ResourceOverflow {
            resource: "live backend latency",
        })?;

    Ok(LiveBufferPlan {
        chunk_frames: usize::try_from(chunk_frames_u64).map_err(|_| {
            ConfigError::ResourceOverflow {
                resource: "live chunk frames",
            }
        })?,
        input_capacity: usize::try_from(input_samples).map_err(|_| {
            ConfigError::ResourceOverflow {
                resource: "live input buffer",
            }
        })?,
        queue_capacity: usize::try_from(queue_samples).map_err(|_| {
            ConfigError::ResourceOverflow {
                resource: "live playback queue",
            }
        })?,
        target_queue_frames: usize::try_from(target_queue_frames).map_err(|_| {
            ConfigError::ResourceOverflow {
                resource: "live target latency",
            }
        })?,
        maximum_resampled_frames: usize::try_from(maximum_resampled_frames).map_err(|_| {
            ConfigError::ResourceOverflow {
                resource: "live asynchronous resampler",
            }
        })?,
        resampler_delay_frames: usize::try_from(resampler_delay_frames).map_err(|_| {
            ConfigError::ResourceOverflow {
                resource: "live resampler latency",
            }
        })?,
        backend_delay_frames: usize::try_from(backend_delay_frames).map_err(|_| {
            ConfigError::ResourceOverflow {
                resource: "live backend latency",
            }
        })?,
        required_bytes,
    })
}

#[cfg(test)]
fn plan_live_buffers(
    config: &LiveConfig,
    sample_rate: u32,
    input_channels: usize,
    output_channels: usize,
) -> Result<LiveBufferPlan, ConfigError> {
    plan_live_buffers_with_resilience(
        config,
        LiveResilienceConfig::default(),
        sample_rate,
        sample_rate,
        input_channels,
        output_channels,
    )
}

fn async_resampler_plan_bytes(
    input_frames: u64,
    output_frames: u64,
    channels: usize,
) -> Result<u64, ConfigError> {
    let channels = u64::try_from(channels).map_err(|_| ConfigError::ResourceOverflow {
        resource: "live asynchronous resampler",
    })?;
    let internal_frames = checked_resource_add(
        "live asynchronous resampler",
        input_frames,
        (ASYNC_SINC_LEN * 2) as u64,
    )?;
    let channel_frames = checked_resource_add(
        "live asynchronous resampler",
        checked_resource_multiply("live asynchronous resampler", internal_frames, channels)?,
        checked_resource_multiply("live asynchronous resampler", output_frames, channels)?,
    )?;
    let channel_bytes = checked_resource_multiply(
        "live asynchronous resampler",
        channel_frames,
        std::mem::size_of::<f64>() as u64,
    )?;
    let filter_values = checked_resource_multiply(
        "live asynchronous resampler",
        ASYNC_SINC_LEN as u64,
        ASYNC_SINC_OVERSAMPLING as u64,
    )?;
    let filter_bytes = checked_resource_multiply(
        "live asynchronous resampler",
        filter_values,
        std::mem::size_of::<f64>() as u64,
    )?;
    checked_resource_add("live asynchronous resampler", channel_bytes, filter_bytes)
}

/// Connection phase reported by [`LiveStatus`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveConnectionState {
    Connecting,
    Priming,
    Running,
    Recovering,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct LiveStatus {
    /// Processing clock. This remains the output-device rate for compatibility.
    pub sample_rate: u32,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub input_channels: usize,
    pub output_channels: usize,
    pub chunk_frames: usize,
    pub input_level: f32,
    pub output_level: f32,
    pub processed_chunks: u64,
    pub dropped_chunks: u64,
    pub underrun_frames: u64,
    pub overflow_frames: u64,
    pub queued_frames: usize,
    pub target_queue_frames: usize,
    pub queue_latency_ms: f64,
    pub processing_latency_ms: f64,
    pub input_device_latency_ms: f64,
    pub output_device_latency_ms: f64,
    /// Estimated capture-to-playback latency, including algorithmic delay.
    pub estimated_total_latency_ms: f64,
    /// Current output/input ratio correction. Positive values grow the queue.
    pub drift_correction_ppm: f64,
    pub reconnect_attempts: u64,
    pub device_generation: u64,
    pub connection_state: LiveConnectionState,
    /// Concrete runtime used by the live processor.
    pub accelerator: AcceleratorSelection,
}

impl LiveStatus {
    fn connection(
        state: LiveConnectionState,
        accelerator: AcceleratorSelection,
        reconnect_attempts: u64,
        device_generation: u64,
    ) -> Self {
        Self {
            sample_rate: 0,
            input_sample_rate: 0,
            output_sample_rate: 0,
            input_channels: 0,
            output_channels: 0,
            chunk_frames: 0,
            input_level: 0.0,
            output_level: 0.0,
            processed_chunks: 0,
            dropped_chunks: 0,
            underrun_frames: 0,
            overflow_frames: 0,
            queued_frames: 0,
            target_queue_frames: 0,
            queue_latency_ms: 0.0,
            processing_latency_ms: 0.0,
            input_device_latency_ms: 0.0,
            output_device_latency_ms: 0.0,
            estimated_total_latency_ms: 0.0,
            drift_correction_ppm: 0.0,
            reconnect_attempts,
            device_generation,
            connection_state: state,
            accelerator,
        }
    }
}

struct ClockDriftController {
    target_frames: f64,
    max_correction_ppm: f64,
    integral_error_seconds: f64,
    correction_ppm: f64,
}

impl ClockDriftController {
    fn new(target_frames: usize, max_correction_ppm: u32) -> Self {
        Self {
            target_frames: target_frames.max(1) as f64,
            max_correction_ppm: max_correction_ppm as f64,
            integral_error_seconds: 0.0,
            correction_ppm: 0.0,
        }
    }

    fn update(&mut self, queued_frames: usize, elapsed_seconds: f64) -> f64 {
        if self.max_correction_ppm == 0.0 {
            self.correction_ppm = 0.0;
            return 0.0;
        }
        let normalized_error =
            ((self.target_frames - queued_frames as f64) / self.target_frames).clamp(-2.0, 2.0);
        self.integral_error_seconds = (self.integral_error_seconds
            + normalized_error * elapsed_seconds.max(0.0))
        .clamp(-5.0, 5.0);
        let requested = 5_000.0 * normalized_error + 500.0 * self.integral_error_seconds;
        self.correction_ppm = requested.clamp(-self.max_correction_ppm, self.max_correction_ppm);
        self.correction_ppm
    }

    fn reset(&mut self) {
        self.integral_error_seconds = 0.0;
        self.correction_ppm = 0.0;
    }
}

struct AdaptiveClockResampler {
    converter: SincFixedIn<f64>,
    output: Vec<Vec<f64>>,
    maximum_output_frames: usize,
}

impl AdaptiveClockResampler {
    fn new(
        input_sample_rate: u32,
        output_sample_rate: u32,
        input_frames: usize,
        channels: usize,
        maximum_output_frames: usize,
        max_drift_ppm: u32,
    ) -> Result<Self, String> {
        let nominal_ratio = output_sample_rate as f64 / input_sample_rate as f64;
        let maximum_relative_ratio = 1.0 / (1.0 - max_drift_ppm as f64 / 1_000_000.0);
        let converter = SincFixedIn::<f64>::new(
            nominal_ratio,
            maximum_relative_ratio,
            SincInterpolationParameters {
                sinc_len: ASYNC_SINC_LEN,
                f_cutoff: 0.95,
                oversampling_factor: ASYNC_SINC_OVERSAMPLING,
                interpolation: SincInterpolationType::Cubic,
                window: WindowFunction::BlackmanHarris2,
            },
            input_frames,
            channels,
        )
        .map_err(|error| format!("initialize live asynchronous resampler: {error}"))?;
        if converter.output_frames_max() > maximum_output_frames {
            return Err(format!(
                "live asynchronous resampler requires {} output frames, planned maximum is {maximum_output_frames}",
                converter.output_frames_max()
            ));
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(channels)
            .map_err(|_| ConfigError::allocation_failed("live resampler output").to_string())?;
        for _ in 0..channels {
            let mut channel = Vec::new();
            channel
                .try_reserve_exact(maximum_output_frames)
                .map_err(|_| ConfigError::allocation_failed("live resampler output").to_string())?;
            channel.resize(maximum_output_frames, 0.0);
            output.push(channel);
        }
        Ok(Self {
            converter,
            output,
            maximum_output_frames,
        })
    }

    fn process(
        &mut self,
        input: &[Vec<f64>],
        correction_ppm: f64,
    ) -> Result<Vec<Vec<f64>>, String> {
        let relative_ratio = 1.0 + correction_ppm / 1_000_000.0;
        self.converter
            .set_resample_ratio_relative(relative_ratio, true)
            .map_err(|error| format!("adjust live asynchronous resampler: {error}"))?;
        let (_, output_frames) = self
            .converter
            .process_into_buffer(input, &mut self.output, None)
            .map_err(|error| format!("resample live capture: {error}"))?;
        if output_frames > self.maximum_output_frames {
            return Err("live asynchronous resampler exceeded its planned output".into());
        }
        let mut converted = Vec::new();
        converted
            .try_reserve_exact(self.output.len())
            .map_err(|_| ConfigError::allocation_failed("live resampled channels").to_string())?;
        for source in &self.output {
            let mut channel = Vec::new();
            channel.try_reserve_exact(output_frames).map_err(|_| {
                ConfigError::allocation_failed("live resampled samples").to_string()
            })?;
            channel.extend_from_slice(&source[..output_frames]);
            converted.push(channel);
        }
        Ok(converted)
    }

    fn reset(&mut self) {
        self.converter.reset();
    }
}

struct CapturedChunk {
    sequence: u64,
    samples: Vec<f32>,
}

#[derive(Clone, Debug)]
struct GenerationFailure {
    message: String,
    recoverable: bool,
    reached_running: bool,
}

impl GenerationFailure {
    fn recoverable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            recoverable: true,
            reached_running: false,
        }
    }

    fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            recoverable: false,
            reached_running: false,
        }
    }

    fn after_start(mut self) -> Self {
        self.reached_running = true;
        self
    }
}

struct RecoverySchedule {
    timeout: Duration,
    timeout_ms: u32,
    started: Option<Instant>,
    backoff: Duration,
    reconnect_attempts: u64,
    device_generation: u64,
}

impl RecoverySchedule {
    fn new(timeout_ms: u32) -> Self {
        Self {
            timeout: Duration::from_millis(timeout_ms as u64),
            timeout_ms,
            started: None,
            backoff: Duration::from_millis(RECONNECT_INITIAL_BACKOFF_MS),
            reconnect_attempts: 0,
            device_generation: 1,
        }
    }

    fn schedule(&mut self, failure: &GenerationFailure, now: Instant) -> Result<Duration, String> {
        if !failure.recoverable || self.timeout.is_zero() {
            return Err(failure.message.clone());
        }
        if failure.reached_running || self.started.is_none() {
            self.started = Some(now);
            self.backoff = Duration::from_millis(RECONNECT_INITIAL_BACKOFF_MS);
        }
        let elapsed = now.saturating_duration_since(self.started.expect("recovery start exists"));
        if elapsed >= self.timeout {
            return Err(format!(
                "live device recovery timed out after {} ms: {}",
                self.timeout_ms, failure.message
            ));
        }
        self.reconnect_attempts = self
            .reconnect_attempts
            .checked_add(1)
            .ok_or_else(|| "live reconnect attempt counter exhausted".to_string())?;
        self.device_generation = self
            .device_generation
            .checked_add(1)
            .ok_or_else(|| "live device generation counter exhausted".to_string())?;
        let remaining = self.timeout.saturating_sub(elapsed);
        let delay = self.backoff.min(remaining);
        self.backoff = self
            .backoff
            .saturating_mul(2)
            .min(Duration::from_millis(RECONNECT_MAX_BACKOFF_MS));
        Ok(delay)
    }
}

fn record_generation_failure(
    slot: &Mutex<Option<GenerationFailure>>,
    failure: GenerationFailure,
    generation_running: &AtomicBool,
) {
    if let Ok(mut current) = slot.lock() {
        if current.is_none() {
            *current = Some(failure);
        }
    }
    generation_running.store(false, Ordering::Release);
}

#[derive(Clone)]
struct LiveMetrics {
    input_level: Arc<AtomicU32>,
    output_level: Arc<AtomicU32>,
    dropped_chunks: Arc<AtomicU64>,
    processed_chunks: Arc<AtomicU64>,
    underrun_frames: Arc<AtomicU64>,
    overflow_frames: Arc<AtomicU64>,
    input_device_latency_us: Arc<AtomicU64>,
    output_device_latency_us: Arc<AtomicU64>,
    processing_latency_us: Arc<AtomicU64>,
    drift_correction_bits: Arc<AtomicU64>,
}

impl LiveMetrics {
    fn new() -> Self {
        Self {
            input_level: Arc::new(AtomicU32::new(0)),
            output_level: Arc::new(AtomicU32::new(0)),
            dropped_chunks: Arc::new(AtomicU64::new(0)),
            processed_chunks: Arc::new(AtomicU64::new(0)),
            underrun_frames: Arc::new(AtomicU64::new(0)),
            overflow_frames: Arc::new(AtomicU64::new(0)),
            input_device_latency_us: Arc::new(AtomicU64::new(0)),
            output_device_latency_us: Arc::new(AtomicU64::new(0)),
            processing_latency_us: Arc::new(AtomicU64::new(0)),
            drift_correction_bits: Arc::new(AtomicU64::new(0.0f64.to_bits())),
        }
    }
}

#[derive(Clone)]
struct LiveProcessorSpec {
    backend: Backend,
    backend_options: BackendOptions,
    accelerator: AcceleratorSelection,
    denoiser: DenoiserConfig,
    channels: usize,
}

struct ProcessedLiveBlock {
    channels: Vec<Vec<f64>>,
    reset_for_gap: bool,
}

/// All DSP state owned by one live session. Rebuilding `kind` before swapping
/// it in makes a discontinuity reset transactional: a failed allocation leaves
/// the old state untouched and terminates the session instead of advancing a
/// subset of the channel, overlap-add, or resampler state.
struct LiveProcessor {
    spec: LiveProcessorSpec,
    kind: LiveProcessorKind,
    next_sequence: u64,
}

enum LiveProcessorKind {
    Stateful(Box<StatefulLiveProcessor>),
    Compatibility(CompatibilityLiveProcessor),
}

struct CompatibilityLiveProcessor {
    backend: Backend,
    backend_options: BackendOptions,
    denoiser: DenoiserConfig,
}

struct StatefulLiveProcessor {
    processor: StreamingBackendSession,
}

impl LiveProcessor {
    #[cfg(test)]
    fn new(config: &LiveConfig, channels: usize) -> Result<Self, String> {
        let accelerator = select_accelerator_for_options(config.backend, &config.backend_options)?;
        Self::new_with_accelerator(config, channels, accelerator)
    }

    fn new_with_accelerator(
        config: &LiveConfig,
        channels: usize,
        accelerator: AcceleratorSelection,
    ) -> Result<Self, String> {
        let spec = LiveProcessorSpec {
            backend: config.backend,
            backend_options: config.backend_options.clone(),
            accelerator,
            denoiser: config.denoiser.clone(),
            channels,
        };
        let kind = LiveProcessorKind::new(&spec, true)?;
        Ok(Self {
            spec,
            kind,
            next_sequence: 0,
        })
    }

    fn process_chunk(
        &mut self,
        sequence: u64,
        channels: Vec<Vec<f64>>,
    ) -> Result<ProcessedLiveBlock, String> {
        validate_live_block(&channels, self.spec.channels)?;
        let following = sequence
            .checked_add(1)
            .ok_or_else(|| "live capture sequence exhausted".to_string())?;
        let reset_for_gap = sequence != self.next_sequence;
        if reset_for_gap {
            self.kind.reset()?;
        }
        self.next_sequence = following;
        let mut channels = self.kind.process_block(channels)?;
        validate_live_block(&channels, self.spec.channels)?;
        for sample in channels.iter_mut().flatten() {
            *sample = crate::audio::sanitize_sample(*sample);
        }
        Ok(ProcessedLiveBlock {
            channels,
            reset_for_gap,
        })
    }
}

impl LiveProcessorKind {
    fn new(spec: &LiveProcessorSpec, warn_compatibility: bool) -> Result<Self, String> {
        if spec.denoiser.vad {
            if warn_compatibility {
                eprintln!(
                    "denoize: live VAD uses chunk-compatible processing; persistent DSP state is unavailable"
                );
            }
            return Ok(Self::Compatibility(CompatibilityLiveProcessor {
                backend: spec.backend,
                backend_options: spec.backend_options.clone(),
                denoiser: spec.denoiser.clone(),
            }));
        }
        Ok(Self::Stateful(Box::new(StatefulLiveProcessor::new(spec)?)))
    }

    fn process_block(&mut self, channels: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Self::Stateful(processor) => processor.process_block(&channels),
            Self::Compatibility(processor) => processor.process_block(channels),
        }
    }

    fn reset(&mut self) -> Result<(), String> {
        match self {
            Self::Stateful(processor) => processor.processor.reset(),
            Self::Compatibility(_) => Ok(()),
        }
    }
}

impl CompatibilityLiveProcessor {
    fn process_block(&mut self, channels: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        let mut audio = Audio {
            sample_rate: self.denoiser.sample_rate,
            channels,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        denoise_audio_with_backend_config(
            &mut audio,
            self.denoiser.clone(),
            self.backend,
            &self.backend_options,
        )?;
        Ok(audio.channels)
    }
}

impl StatefulLiveProcessor {
    fn new(spec: &LiveProcessorSpec) -> Result<Self, String> {
        let mut config = spec.denoiser.clone();
        // Automatic profiling intentionally buffers up to 1.5 seconds for
        // offline analysis. A realtime classical session instead starts the
        // adaptive estimator immediately so its first chunk sees only the
        // normal overlap-add latency.
        if spec.backend == Backend::Classical && config.profile_ms == 0.0 {
            config.profile_ms = -1.0;
        }
        Ok(Self {
            processor: StreamingBackendSession::new_with_accelerator(
                spec.backend,
                spec.denoiser.sample_rate,
                spec.channels,
                config,
                spec.backend_options.clone(),
                spec.accelerator,
            )?,
        })
    }

    fn process_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        self.processor.process_block(channels)
    }
}

fn validate_live_block(channels: &[Vec<f64>], expected_channels: usize) -> Result<(), String> {
    if channels.len() != expected_channels {
        return Err(format!(
            "expected {expected_channels} live channels, got {}",
            channels.len()
        ));
    }
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    if channels.iter().any(|channel| channel.len() != frames) {
        return Err("live blocks must have equal channel lengths".into());
    }
    Ok(())
}

/// Return the input and output device names exposed by the default host.
pub fn device_names() -> Result<(Vec<String>, Vec<String>), String> {
    let host = cpal::default_host();
    let inputs = host
        .input_devices()
        .map_err(|e| format!("enumerate input devices: {e}"))?
        .map(|d| d.name().unwrap_or_else(|_| "<unknown>".into()))
        .collect();
    let outputs = host
        .output_devices()
        .map_err(|e| format!("enumerate output devices: {e}"))?
        .map(|d| d.name().unwrap_or_else(|_| "<unknown>".into()))
        .collect();
    Ok((inputs, outputs))
}

/// Run until Ctrl-C, processing bounded chunks away from the audio callbacks.
pub fn run(config: LiveConfig) -> Result<(), String> {
    let config = PreparedLiveConfig::new(config)?;
    let running = Arc::new(AtomicBool::new(true));
    let _signal_session = register_ctrl_c_session(Arc::clone(&running))?;
    run_prepared_with_status_impl(config, running, |_| {}, None)
}

/// Run until Ctrl-C under one process-wide resource governor.
pub fn run_with_governor(
    config: LiveConfig,
    governor: &crate::ResourceGovernor,
) -> Result<(), String> {
    let config = PreparedLiveConfig::new(config)?;
    let running = Arc::new(AtomicBool::new(true));
    let _signal_session = register_ctrl_c_session(Arc::clone(&running))?;
    run_prepared_with_status_impl(config, running, |_| {}, Some(governor))
}

/// Run a live session with status diagnostics until Ctrl-C.
pub fn run_with_governor_and_status<F>(
    config: LiveConfig,
    governor: &crate::ResourceGovernor,
    report: F,
) -> Result<(), String>
where
    F: FnMut(LiveStatus),
{
    let prepared = PreparedLiveConfig::new(config)?;
    run_prepared_with_governor_and_status(prepared, governor, report)
}

/// Run an already-prepared resilient live session until Ctrl-C.
pub fn run_prepared_with_governor_and_status<F>(
    prepared: PreparedLiveConfig,
    governor: &crate::ResourceGovernor,
    report: F,
) -> Result<(), String>
where
    F: FnMut(LiveStatus),
{
    let running = Arc::new(AtomicBool::new(true));
    let _signal_session = register_ctrl_c_session(Arc::clone(&running))?;
    run_prepared_with_status_impl(prepared, running, report, Some(governor))
}

/// Run a live session controlled by the caller and periodically report levels.
pub fn run_with_status<F>(
    config: LiveConfig,
    running: Arc<AtomicBool>,
    report: F,
) -> Result<(), String>
where
    F: FnMut(LiveStatus),
{
    let config = PreparedLiveConfig::new(config)?;
    run_prepared_with_status_impl(config, running, report, None)
}

/// Validated live configuration with one captured accelerator decision.
///
/// Frontends can prepare this value before registering a live job, then pass
/// it to [`run_prepared_with_status`] without probing hardware a second time.
#[derive(Debug)]
pub struct PreparedLiveConfig {
    config: LiveConfig,
    accelerator: AcceleratorSelection,
    resilience: LiveResilienceConfig,
}

impl PreparedLiveConfig {
    /// Validate resources and capture the runtime that execution will use.
    pub fn new(mut config: LiveConfig) -> Result<Self, String> {
        config
            .validate_config()
            .map_err(|error| error.to_string())?;
        config.backend_options =
            crate::service::resolve_backend_options(config.backend, config.backend_options)?;
        let accelerator = select_accelerator_for_options(config.backend, &config.backend_options)?;
        Ok(Self {
            config,
            accelerator,
            resilience: LiveResilienceConfig::default(),
        })
    }

    /// Override clock/reconnect policy after model and backend preparation.
    pub fn with_resilience(mut self, resilience: LiveResilienceConfig) -> Result<Self, String> {
        resilience.validate().map_err(|error| error.to_string())?;
        resilience
            .resolved_target_latency_ms(self.config.chunk_ms)
            .map_err(|error| error.to_string())?;
        self.resilience = resilience;
        Ok(self)
    }

    /// Return the concrete runtime captured during preparation.
    #[must_use]
    pub const fn accelerator(&self) -> AcceleratorSelection {
        self.accelerator
    }

    #[must_use]
    pub const fn resilience(&self) -> LiveResilienceConfig {
        self.resilience
    }
}

struct CtrlCSession {
    running: Arc<AtomicBool>,
}

impl Drop for CtrlCSession {
    fn drop(&mut self) {
        let slot = CTRL_C_SESSION.get_or_init(|| Mutex::new(None));
        if let Ok(mut active) = slot.lock() {
            let belongs_to_this_session = active
                .as_ref()
                .and_then(Weak::upgrade)
                .is_some_and(|running| Arc::ptr_eq(&running, &self.running));
            if belongs_to_this_session {
                *active = None;
            }
        }
    }
}

fn register_ctrl_c_session(running: Arc<AtomicBool>) -> Result<CtrlCSession, String> {
    let installed = CTRL_C_HANDLER.get_or_init(|| {
        ctrlc::set_handler(|| {
            let slot = CTRL_C_SESSION.get_or_init(|| Mutex::new(None));
            if let Ok(active) = slot.lock() {
                if let Some(running) = active.as_ref().and_then(Weak::upgrade) {
                    running.store(false, Ordering::SeqCst);
                }
            }
        })
        .map_err(|error| format!("install Ctrl-C handler: {error}"))
    });
    if let Err(error) = installed {
        return Err(error.clone());
    }

    let slot = CTRL_C_SESSION.get_or_init(|| Mutex::new(None));
    let mut active = slot
        .lock()
        .map_err(|_| "Ctrl-C session lock poisoned".to_string())?;
    if active.as_ref().and_then(Weak::upgrade).is_some() {
        return Err("another Ctrl-C controlled live session is already running".into());
    }
    *active = Some(Arc::downgrade(&running));
    Ok(CtrlCSession { running })
}

/// Run an already-prepared live session and periodically report levels.
pub fn run_prepared_with_status<F>(
    prepared: PreparedLiveConfig,
    running: Arc<AtomicBool>,
    report: F,
) -> Result<(), String>
where
    F: FnMut(LiveStatus),
{
    run_prepared_with_status_impl(prepared, running, report, None)
}

/// Run an already-prepared live session under aggregate resource admission.
pub fn run_prepared_with_status_and_governor<F>(
    prepared: PreparedLiveConfig,
    running: Arc<AtomicBool>,
    governor: &crate::ResourceGovernor,
    report: F,
) -> Result<(), String>
where
    F: FnMut(LiveStatus),
{
    run_prepared_with_status_impl(prepared, running, report, Some(governor))
}

fn run_prepared_with_status_impl<F>(
    prepared: PreparedLiveConfig,
    running: Arc<AtomicBool>,
    mut report: F,
    governor: Option<&crate::ResourceGovernor>,
) -> Result<(), String>
where
    F: FnMut(LiveStatus),
{
    let accelerator = prepared.accelerator;
    let mut recovery = RecoverySchedule::new(prepared.resilience.reconnect_timeout_ms);
    let mut last_status = None::<LiveStatus>;

    report(LiveStatus::connection(
        LiveConnectionState::Connecting,
        accelerator,
        recovery.reconnect_attempts,
        recovery.device_generation,
    ));
    loop {
        if !running.load(Ordering::Acquire) {
            return Ok(());
        }
        let result = run_device_generation(
            &prepared,
            Arc::clone(&running),
            governor,
            recovery.reconnect_attempts,
            recovery.device_generation,
            |status| {
                last_status = Some(status);
                report(status);
            },
        );
        match result {
            Ok(()) => return Ok(()),
            Err(_failure) if !running.load(Ordering::Acquire) => return Ok(()),
            Err(failure) => {
                let delay = recovery.schedule(&failure, Instant::now())?;
                let mut status = last_status.unwrap_or_else(|| {
                    LiveStatus::connection(
                        LiveConnectionState::Recovering,
                        accelerator,
                        recovery.reconnect_attempts,
                        recovery.device_generation,
                    )
                });
                status.connection_state = LiveConnectionState::Recovering;
                status.reconnect_attempts = recovery.reconnect_attempts;
                status.device_generation = recovery.device_generation;
                report(status);
                eprintln!(
                    "denoize: live device interrupted; reconnect attempt {}: {}",
                    recovery.reconnect_attempts, failure.message
                );
                interruptible_sleep(delay, &running);
            }
        }
    }
}

fn run_device_generation<F>(
    prepared: &PreparedLiveConfig,
    session_running: Arc<AtomicBool>,
    governor: Option<&crate::ResourceGovernor>,
    reconnect_attempts: u64,
    device_generation: u64,
    mut report: F,
) -> Result<(), GenerationFailure>
where
    F: FnMut(LiveStatus),
{
    let mut config = prepared.config.clone();
    let accelerator = prepared.accelerator;
    let resilience = prepared.resilience;
    let host = cpal::default_host();
    let input = select_device(&host, true, config.input_device.as_deref())
        .map_err(GenerationFailure::recoverable)?;
    let output = select_device(&host, false, config.output_device.as_deref())
        .map_err(GenerationFailure::recoverable)?;
    let input_supported = input
        .default_input_config()
        .map_err(|error| GenerationFailure::recoverable(format!("input config: {error}")))?;
    let output_supported = output
        .default_output_config()
        .map_err(|error| GenerationFailure::recoverable(format!("output config: {error}")))?;
    let input_cfg: StreamConfig = input_supported.clone().into();
    let output_cfg: StreamConfig = output_supported.clone().into();
    let input_rate = input_cfg.sample_rate.0;
    let output_rate = output_cfg.sample_rate.0;
    let in_channels = input_cfg.channels as usize;
    let out_channels = output_cfg.channels as usize;
    let buffer_plan = plan_live_buffers_with_resilience(
        &config,
        resilience,
        input_rate,
        output_rate,
        in_channels,
        out_channels,
    )
    .map_err(|error| GenerationFailure::fatal(error.to_string()))?;
    let worker_memory = buffer_plan
        .required_bytes
        .checked_add(crate::estimate_backend_worker_memory_bytes(
            &config.backend_options,
        ))
        .ok_or_else(|| GenerationFailure::fatal("live worker memory reservation overflow"))?;
    let mut worker_request = crate::ResourceRequest::worker(worker_memory, 0);
    if accelerator.effective() != crate::AcceleratorRuntime::Cpu {
        let gpu_memory = buffer_plan
            .required_bytes
            .checked_mul(2)
            .and_then(|bytes| {
                bytes.checked_add(crate::estimate_backend_worker_gpu_memory_bytes(
                    &config.backend_options,
                ))
            })
            .ok_or_else(|| GenerationFailure::fatal("live GPU reservation overflow"))?;
        worker_request = worker_request
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(gpu_memory);
    }
    let request = worker_request
        .checked_add(
            crate::estimate_backend_session_request(
                config.backend,
                &config.backend_options,
                accelerator,
            )
            .map_err(GenerationFailure::fatal)?,
        )
        .map_err(GenerationFailure::fatal)?;
    let _resource_permit = governor
        .map(|governor| {
            governor.acquire_with_cancel(request, || !session_running.load(Ordering::Acquire))
        })
        .transpose()
        .map_err(GenerationFailure::fatal)?;
    config.denoiser.sample_rate = output_rate;
    let chunk_frames = buffer_plan.chunk_frames;
    let queue_capacity = buffer_plan.queue_capacity;
    let mut playback_queue = VecDeque::<f32>::new();
    playback_queue
        .try_reserve_exact(queue_capacity)
        .map_err(|_| {
            GenerationFailure::fatal(
                ConfigError::allocation_failed("live playback queue").to_string(),
            )
        })?;
    let playback = Arc::new(Mutex::new(playback_queue));
    let mut pending_input = Vec::<f32>::new();
    pending_input
        .try_reserve_exact(buffer_plan.input_capacity)
        .map_err(|_| {
            GenerationFailure::fatal(
                ConfigError::allocation_failed("live input buffer").to_string(),
            )
        })?;
    let (tx, rx) = mpsc::sync_channel::<CapturedChunk>(CAPTURE_QUEUE_CHUNKS);
    let (worker_ready_tx, worker_ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);
    let metrics = LiveMetrics::new();
    let generation_failure = Arc::new(Mutex::new(None::<GenerationFailure>));
    let generation_running = Arc::new(AtomicBool::new(true));
    let worker_running = Arc::clone(&generation_running);
    let worker_playback = Arc::clone(&playback);
    let worker_metrics = metrics.clone();
    let worker_failure = Arc::clone(&generation_failure);
    let worker_config = config.clone();
    let worker = std::thread::spawn(move || {
        // Some compiled stream backends intentionally own thread-affine model
        // state. Construct the live processor on its permanent worker thread,
        // then acknowledge readiness before either device stream is opened.
        let mut live_processor =
            match LiveProcessor::new_with_accelerator(&worker_config, in_channels, accelerator) {
                Ok(processor) => processor,
                Err(error) => {
                    let _ = worker_ready_tx.send(Err(error));
                    return;
                }
            };
        let mut clock_resampler = match AdaptiveClockResampler::new(
            input_rate,
            output_rate,
            chunk_frames,
            in_channels,
            buffer_plan.maximum_resampled_frames,
            resilience.max_drift_ppm,
        ) {
            Ok(resampler) => resampler,
            Err(error) => {
                let _ = worker_ready_tx.send(Err(error));
                return;
            }
        };
        let mut drift_controller =
            ClockDriftController::new(buffer_plan.target_queue_frames, resilience.max_drift_ppm);
        let mut next_resampler_sequence = 0u64;
        let chunk_seconds = chunk_frames as f64 / input_rate as f64;
        let mut processing_average_us = 0.0f64;
        if worker_ready_tx.send(Ok(())).is_err() {
            return;
        }
        while worker_running.load(Ordering::Acquire) {
            let captured = match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(captured) => captured,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };
            let processing_started = Instant::now();
            let mut channels = Vec::new();
            if channels.try_reserve_exact(in_channels).is_err() {
                record_generation_failure(
                    &worker_failure,
                    GenerationFailure::fatal(
                        ConfigError::allocation_failed("live worker channels").to_string(),
                    ),
                    &worker_running,
                );
                break;
            }
            let mut allocation_failed = false;
            for _ in 0..in_channels {
                let mut channel = Vec::new();
                if channel.try_reserve_exact(chunk_frames).is_err() {
                    allocation_failed = true;
                    break;
                }
                channels.push(channel);
            }
            if allocation_failed {
                record_generation_failure(
                    &worker_failure,
                    GenerationFailure::fatal(
                        ConfigError::allocation_failed("live worker samples").to_string(),
                    ),
                    &worker_running,
                );
                break;
            }
            for frame in captured.samples.chunks_exact(in_channels) {
                for (channel, sample) in channels.iter_mut().zip(frame) {
                    channel.push(*sample as f64);
                }
            }
            let following_sequence = match captured.sequence.checked_add(1) {
                Some(following) => following,
                None => {
                    record_generation_failure(
                        &worker_failure,
                        GenerationFailure::fatal("live capture sequence exhausted"),
                        &worker_running,
                    );
                    break;
                }
            };
            let reset_for_gap = captured.sequence != next_resampler_sequence;
            next_resampler_sequence = following_sequence;
            if reset_for_gap {
                clock_resampler.reset();
                drift_controller.reset();
            }
            let queued_frames = match worker_playback.lock() {
                Ok(queue) => queue.len() / out_channels,
                Err(_) => {
                    record_generation_failure(
                        &worker_failure,
                        GenerationFailure::fatal("live playback queue lock poisoned"),
                        &worker_running,
                    );
                    break;
                }
            };
            let correction_ppm = if reset_for_gap {
                0.0
            } else {
                drift_controller.update(queued_frames, chunk_seconds)
            };
            worker_metrics
                .drift_correction_bits
                .store(correction_ppm.to_bits(), Ordering::Relaxed);
            let channels = match clock_resampler.process(&channels, correction_ppm) {
                Ok(channels) => channels,
                Err(error) => {
                    record_generation_failure(
                        &worker_failure,
                        GenerationFailure::fatal(error),
                        &worker_running,
                    );
                    break;
                }
            };
            let processed = match live_processor.process_chunk(captured.sequence, channels) {
                Ok(processed) => processed,
                Err(error) => {
                    record_generation_failure(
                        &worker_failure,
                        GenerationFailure::fatal(error),
                        &worker_running,
                    );
                    break;
                }
            };
            let audio = processed.channels;
            let enqueue_result = match worker_playback.lock() {
                Ok(mut queue) => {
                    if processed.reset_for_gap {
                        queue.clear();
                    }
                    enqueue_playback_block(
                        &mut queue,
                        &audio,
                        out_channels,
                        queue_capacity,
                        &worker_metrics.output_level,
                    )
                }
                Err(_) => Err("live playback queue lock poisoned".into()),
            };
            match enqueue_result {
                Ok(overflow_frames) => {
                    worker_metrics
                        .overflow_frames
                        .fetch_add(overflow_frames as u64, Ordering::Relaxed);
                }
                Err(error) => {
                    record_generation_failure(
                        &worker_failure,
                        GenerationFailure::fatal(error),
                        &worker_running,
                    );
                    break;
                }
            }
            worker_metrics
                .processed_chunks
                .fetch_add(1, Ordering::Relaxed);
            let elapsed_us = processing_started.elapsed().as_secs_f64() * 1_000_000.0;
            processing_average_us = if processing_average_us == 0.0 {
                elapsed_us
            } else {
                0.9 * processing_average_us + 0.1 * elapsed_us
            };
            worker_metrics
                .processing_latency_us
                .store(processing_average_us.round() as u64, Ordering::Relaxed);
        }
    });
    match worker_ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let _ = worker.join();
            return Err(GenerationFailure::fatal(error));
        }
        Err(_) => {
            let _ = worker.join();
            return Err(GenerationFailure::fatal(
                "live worker exited before processor initialization completed",
            ));
        }
    }

    let input_stream = match build_input(
        &input,
        &input_cfg,
        input_supported.sample_format(),
        tx,
        chunk_frames,
        pending_input,
        Arc::clone(&metrics.input_level),
        Arc::clone(&metrics.dropped_chunks),
        Arc::clone(&metrics.input_device_latency_us),
        Arc::clone(&generation_failure),
        Arc::clone(&generation_running),
    ) {
        Ok(stream) => stream,
        Err(error) => {
            generation_running.store(false, Ordering::Release);
            let _ = worker.join();
            return Err(GenerationFailure::recoverable(error));
        }
    };
    let output_stream = match build_output(
        &output,
        &output_cfg,
        output_supported.sample_format(),
        Arc::clone(&playback),
        out_channels,
        Arc::clone(&metrics.underrun_frames),
        Arc::clone(&metrics.output_device_latency_us),
        Arc::clone(&generation_failure),
        Arc::clone(&generation_running),
    ) {
        Ok(stream) => stream,
        Err(error) => {
            drop(input_stream);
            generation_running.store(false, Ordering::Release);
            let _ = worker.join();
            return Err(GenerationFailure::recoverable(error));
        }
    };
    if let Err(error) = input_stream.play() {
        drop(input_stream);
        drop(output_stream);
        generation_running.store(false, Ordering::Release);
        let _ = worker.join();
        return Err(GenerationFailure::recoverable(format!(
            "start input: {error}"
        )));
    }

    let prime_timeout = PRIME_TIMEOUT.saturating_add(Duration::from_secs_f64(
        (config.denoiser.profile_ms.max(0.0) / 1_000.0).min(60.0),
    ));
    let prime_started = Instant::now();
    report(runtime_status(
        LiveConnectionState::Priming,
        input_rate,
        output_rate,
        in_channels,
        out_channels,
        &buffer_plan,
        &playback,
        &metrics,
        accelerator,
        reconnect_attempts,
        device_generation,
    ));
    let mut last_report = Instant::now();
    while session_running.load(Ordering::Acquire)
        && generation_running.load(Ordering::Acquire)
        && !worker.is_finished()
    {
        let queued_frames = playback
            .lock()
            .map(|queue| queue.len() / out_channels)
            .unwrap_or(0);
        if queued_frames >= buffer_plan.target_queue_frames {
            break;
        }
        if prime_started.elapsed() >= prime_timeout {
            record_generation_failure(
                &generation_failure,
                GenerationFailure::fatal(format!(
                    "live playback queue did not reach its {} frame target",
                    buffer_plan.target_queue_frames
                )),
                &generation_running,
            );
            break;
        }
        if last_report.elapsed() >= LIVE_STATUS_INTERVAL {
            report(runtime_status(
                LiveConnectionState::Priming,
                input_rate,
                output_rate,
                in_channels,
                out_channels,
                &buffer_plan,
                &playback,
                &metrics,
                accelerator,
                reconnect_attempts,
                device_generation,
            ));
            last_report = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if !session_running.load(Ordering::Acquire) {
        generation_running.store(false, Ordering::Release);
        drop(input_stream);
        drop(output_stream);
        let _ = worker.join();
        return Ok(());
    }
    if !generation_running.load(Ordering::Acquire) || worker.is_finished() {
        drop(input_stream);
        drop(output_stream);
        generation_running.store(false, Ordering::Release);
        let _ = worker.join();
        return Err(
            take_generation_failure(&generation_failure).unwrap_or_else(|| {
                GenerationFailure::fatal("live worker stopped while priming playback")
            }),
        );
    }
    if let Err(error) = output_stream.play() {
        drop(input_stream);
        drop(output_stream);
        generation_running.store(false, Ordering::Release);
        let _ = worker.join();
        return Err(GenerationFailure::recoverable(format!(
            "start output: {error}"
        )));
    }
    let fallback = accelerator
        .fallback()
        .map(|reason| format!(", fallback {}", reason.name()))
        .unwrap_or_default();
    eprintln!(
        "denoize: live input {input_rate} Hz -> output {output_rate} Hz, {in_channels} input channel(s), {chunk_frames} frames/chunk, target {} frames; accelerator {}{fallback}; press Ctrl-C to stop",
        buffer_plan.target_queue_frames,
        accelerator.effective().name()
    );
    report(runtime_status(
        LiveConnectionState::Running,
        input_rate,
        output_rate,
        in_channels,
        out_channels,
        &buffer_plan,
        &playback,
        &metrics,
        accelerator,
        reconnect_attempts,
        device_generation,
    ));
    while session_running.load(Ordering::Acquire)
        && generation_running.load(Ordering::Acquire)
        && !worker.is_finished()
    {
        std::thread::sleep(LIVE_STATUS_INTERVAL);
        report(runtime_status(
            LiveConnectionState::Running,
            input_rate,
            output_rate,
            in_channels,
            out_channels,
            &buffer_plan,
            &playback,
            &metrics,
            accelerator,
            reconnect_attempts,
            device_generation,
        ));
    }
    generation_running.store(false, Ordering::Release);
    drop(input_stream);
    drop(output_stream);
    worker
        .join()
        .map_err(|_| GenerationFailure::fatal("live worker panicked"))?;
    if !session_running.load(Ordering::Acquire) {
        return Ok(());
    }
    Err(take_generation_failure(&generation_failure)
        .unwrap_or_else(|| GenerationFailure::fatal("live worker stopped unexpectedly"))
        .after_start())
}

fn take_generation_failure(
    failure: &Mutex<Option<GenerationFailure>>,
) -> Option<GenerationFailure> {
    failure.lock().ok()?.take()
}

fn interruptible_sleep(duration: Duration, running: &AtomicBool) {
    let deadline = Instant::now().checked_add(duration);
    while running.load(Ordering::Acquire) {
        let Some(remaining) =
            deadline.and_then(|deadline| deadline.checked_duration_since(Instant::now()))
        else {
            break;
        };
        if remaining.is_zero() {
            break;
        }
        std::thread::sleep(remaining.min(Duration::from_millis(25)));
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_status(
    connection_state: LiveConnectionState,
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_channels: usize,
    output_channels: usize,
    buffer_plan: &LiveBufferPlan,
    playback: &Mutex<VecDeque<f32>>,
    metrics: &LiveMetrics,
    accelerator: AcceleratorSelection,
    reconnect_attempts: u64,
    device_generation: u64,
) -> LiveStatus {
    let queued_frames = playback
        .lock()
        .map(|queue| queue.len() / output_channels.max(1))
        .unwrap_or(0);
    let queue_latency_ms = frames_to_milliseconds(queued_frames, output_sample_rate);
    let processing_latency_ms =
        metrics.processing_latency_us.load(Ordering::Relaxed) as f64 / 1_000.0;
    let input_device_latency_ms =
        metrics.input_device_latency_us.load(Ordering::Relaxed) as f64 / 1_000.0;
    let output_device_latency_ms =
        metrics.output_device_latency_us.load(Ordering::Relaxed) as f64 / 1_000.0;
    let capture_latency_ms = frames_to_milliseconds(buffer_plan.chunk_frames, input_sample_rate);
    let resampler_latency_ms =
        frames_to_milliseconds(buffer_plan.resampler_delay_frames, output_sample_rate);
    let backend_latency_ms =
        frames_to_milliseconds(buffer_plan.backend_delay_frames, output_sample_rate);
    LiveStatus {
        sample_rate: output_sample_rate,
        input_sample_rate,
        output_sample_rate,
        input_channels,
        output_channels,
        chunk_frames: buffer_plan.chunk_frames,
        input_level: f32::from_bits(metrics.input_level.swap(0, Ordering::Relaxed)),
        output_level: f32::from_bits(metrics.output_level.swap(0, Ordering::Relaxed)),
        processed_chunks: metrics.processed_chunks.load(Ordering::Relaxed),
        dropped_chunks: metrics.dropped_chunks.load(Ordering::Relaxed),
        underrun_frames: metrics.underrun_frames.load(Ordering::Relaxed),
        overflow_frames: metrics.overflow_frames.load(Ordering::Relaxed),
        queued_frames,
        target_queue_frames: buffer_plan.target_queue_frames,
        queue_latency_ms,
        processing_latency_ms,
        input_device_latency_ms,
        output_device_latency_ms,
        estimated_total_latency_ms: input_device_latency_ms
            + capture_latency_ms
            + resampler_latency_ms
            + backend_latency_ms
            + processing_latency_ms
            + queue_latency_ms
            + output_device_latency_ms,
        drift_correction_ppm: f64::from_bits(metrics.drift_correction_bits.load(Ordering::Relaxed)),
        reconnect_attempts,
        device_generation,
        connection_state,
        accelerator,
    }
}

fn frames_to_milliseconds(frames: usize, sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        0.0
    } else {
        frames as f64 * 1_000.0 / sample_rate as f64
    }
}

fn select_device(
    host: &cpal::Host,
    input: bool,
    requested: Option<&str>,
) -> Result<Device, String> {
    if let Some(name) = requested {
        let kind = if input { "input" } else { "output" };
        let devices = if input {
            host.input_devices()
        } else {
            host.output_devices()
        }
        .map_err(|e| format!("enumerate devices: {e}"))?;
        return select_unique_named_device(
            devices.filter_map(|device| device.name().ok().map(|name| (name, device))),
            name,
            kind,
        );
    }
    if input {
        host.default_input_device()
    } else {
        host.default_output_device()
    }
    .ok_or_else(|| {
        format!(
            "no default {} device",
            if input { "input" } else { "output" }
        )
    })
}

fn select_unique_named_device<T>(
    devices: impl IntoIterator<Item = (String, T)>,
    requested: &str,
    kind: &str,
) -> Result<T, String> {
    let mut selected = None;
    for (name, device) in devices {
        if name != requested {
            continue;
        }
        if selected.is_some() {
            return Err(format!(
                "{kind} device name is ambiguous: {requested} (multiple exact matches)"
            ));
        }
        selected = Some(device);
    }
    selected.ok_or_else(|| format!("{kind} device not found: {requested}"))
}

fn build_input(
    device: &Device,
    cfg: &StreamConfig,
    format: SampleFormat,
    tx: mpsc::SyncSender<CapturedChunk>,
    chunk_frames: usize,
    pending: Vec<f32>,
    input_level: Arc<AtomicU32>,
    dropped_chunks: Arc<AtomicU64>,
    device_latency_us: Arc<AtomicU64>,
    generation_failure: Arc<Mutex<Option<GenerationFailure>>>,
    generation_running: Arc<AtomicBool>,
) -> Result<Stream, String> {
    let channels = cfg.channels as usize;
    let capacity = chunk_frames.checked_mul(channels).ok_or_else(|| {
        ConfigError::ResourceOverflow {
            resource: "live input buffer",
        }
        .to_string()
    })?;
    macro_rules! stream {
        ($ty:ty, $convert:expr) => {{
            let mut pending = pending;
            let mut next_sequence = 0u64;
            let data_failure = Arc::clone(&generation_failure);
            let stream_failure = Arc::clone(&generation_failure);
            let data_running = Arc::clone(&generation_running);
            let stream_running = Arc::clone(&generation_running);
            let device_latency_us = Arc::clone(&device_latency_us);
            device.build_input_stream(
                cfg,
                move |data: &[$ty], info| {
                    let timestamp = info.timestamp();
                    if let Some(latency) = timestamp.callback.duration_since(&timestamp.capture) {
                        store_duration_us(&device_latency_us, latency);
                    }
                    for sample in data.iter().map($convert) {
                        pending.push(sample);
                        if pending.len() == capacity {
                            let sequence = next_sequence;
                            let Some(following) = next_sequence.checked_add(1) else {
                                record_generation_failure(
                                    &data_failure,
                                    GenerationFailure::fatal("live capture sequence exhausted")
                                        .after_start(),
                                    &data_running,
                                );
                                pending.clear();
                                return;
                            };
                            next_sequence = following;
                            let mut chunk = Vec::new();
                            if chunk.try_reserve_exact(capacity).is_err() {
                                pending.clear();
                                dropped_chunks.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            chunk.extend(pending.drain(..));
                            for sample in &chunk {
                                store_peak(&input_level, sample.abs());
                            }
                            if tx
                                .try_send(CapturedChunk {
                                    sequence,
                                    samples: chunk,
                                })
                                .is_err()
                            {
                                dropped_chunks.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                },
                move |error| {
                    let message = format!("input stream error: {error}");
                    record_generation_failure(
                        &stream_failure,
                        GenerationFailure::recoverable(message.clone()),
                        &stream_running,
                    );
                    eprintln!("denoize: {message}");
                },
                None,
            )
        }};
    }
    let result = match format {
        SampleFormat::I8 => stream!(i8, |x: &i8| input_sample_to_f32(*x)),
        SampleFormat::F32 => stream!(f32, |x: &f32| *x),
        SampleFormat::I16 => stream!(i16, |x: &i16| input_sample_to_f32(*x)),
        SampleFormat::I32 => stream!(i32, |x: &i32| input_sample_to_f32(*x)),
        SampleFormat::I64 => stream!(i64, |x: &i64| input_sample_to_f32(*x)),
        SampleFormat::U8 => stream!(u8, |x: &u8| input_sample_to_f32(*x)),
        SampleFormat::U16 => stream!(u16, |x: &u16| input_sample_to_f32(*x)),
        SampleFormat::U32 => stream!(u32, |x: &u32| input_sample_to_f32(*x)),
        SampleFormat::U64 => stream!(u64, |x: &u64| input_sample_to_f32(*x)),
        SampleFormat::F64 => stream!(f64, |x: &f64| *x as f32),
        other => return Err(format!("unsupported live input sample format: {other:?}")),
    };
    result.map_err(|e| format!("build input stream: {e}"))
}

fn input_sample_to_f32<T>(sample: T) -> f32
where
    T: Sample,
    f32: FromSample<T>,
{
    sample.to_sample::<f32>()
}

fn output_sample_from_f32<T>(sample: f32) -> T
where
    T: Sample + FromSample<f32>,
{
    T::from_sample(crate::audio::sanitize_sample(sample as f64) as f32)
}

fn store_duration_us(target: &AtomicU64, duration: Duration) {
    target.store(
        u64::try_from(duration.as_micros()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

fn store_peak(target: &AtomicU32, value: f32) {
    let mut current = target.load(Ordering::Relaxed);
    while value > f32::from_bits(current) {
        match target.compare_exchange_weak(
            current,
            value.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => current = next,
        }
    }
}

fn enqueue_playback_block(
    queue: &mut VecDeque<f32>,
    audio: &[Vec<f64>],
    output_channels: usize,
    queue_capacity: usize,
    output_level: &AtomicU32,
) -> Result<usize, String> {
    let frames = audio.first().map(Vec::len).unwrap_or(0);
    if audio.is_empty() || audio.iter().any(|channel| channel.len() != frames) {
        return Err("live processor returned invalid playback channels".into());
    }
    let samples = frames
        .checked_mul(output_channels)
        .ok_or_else(|| "live playback block size overflow".to_string())?;
    let required = queue
        .len()
        .checked_add(samples)
        .ok_or_else(|| "live playback queue length overflow".to_string())?;
    if samples > queue_capacity || queue.len() % output_channels != 0 {
        return Err(format!(
            "live playback queue invariant exceeded: block {samples}, queued {}, capacity {queue_capacity}",
            queue.len()
        ));
    }
    let overflow_samples = required.saturating_sub(queue_capacity);
    let overflow_frames = overflow_samples.div_ceil(output_channels);
    let discard_samples = overflow_frames
        .checked_mul(output_channels)
        .ok_or_else(|| "live playback overflow length overflow".to_string())?;
    if discard_samples > queue.len() {
        return Err(format!(
            "live playback queue cannot discard {discard_samples} stale samples from {} queued samples",
            queue.len()
        ));
    }
    for _ in 0..discard_samples {
        queue.pop_front();
    }
    for frame in 0..frames {
        for out_ch in 0..output_channels {
            let source = out_ch.min(audio.len() - 1);
            let sample = audio[source][frame] as f32;
            store_peak(output_level, sample.abs());
            queue.push_back(sample);
        }
    }
    Ok(overflow_frames)
}

fn build_output(
    device: &Device,
    cfg: &StreamConfig,
    format: SampleFormat,
    queue: Arc<Mutex<VecDeque<f32>>>,
    output_channels: usize,
    underrun_frames: Arc<AtomicU64>,
    device_latency_us: Arc<AtomicU64>,
    generation_failure: Arc<Mutex<Option<GenerationFailure>>>,
    generation_running: Arc<AtomicBool>,
) -> Result<Stream, String> {
    macro_rules! stream {
        ($ty:ty, $convert:expr) => {{
            let queue = Arc::clone(&queue);
            let data_failure = Arc::clone(&generation_failure);
            let stream_failure = Arc::clone(&generation_failure);
            let data_running = Arc::clone(&generation_running);
            let stream_running = Arc::clone(&generation_running);
            let underrun_frames = Arc::clone(&underrun_frames);
            let device_latency_us = Arc::clone(&device_latency_us);
            device.build_output_stream(
                cfg,
                move |data: &mut [$ty], info| {
                    let timestamp = info.timestamp();
                    if let Some(latency) = timestamp.playback.duration_since(&timestamp.callback) {
                        store_duration_us(&device_latency_us, latency);
                    }
                    match queue.try_lock() {
                        Ok(mut queue) => {
                            let missing_samples = data.len().saturating_sub(queue.len());
                            if missing_samples != 0 {
                                underrun_frames.fetch_add(
                                    missing_samples.div_ceil(output_channels) as u64,
                                    Ordering::Relaxed,
                                );
                            }
                            for sample in data {
                                *sample = $convert(queue.pop_front().unwrap_or(0.0));
                            }
                        }
                        Err(std::sync::TryLockError::WouldBlock) => {
                            underrun_frames.fetch_add(
                                data.len().div_ceil(output_channels) as u64,
                                Ordering::Relaxed,
                            );
                            for sample in data {
                                *sample = $convert(0.0);
                            }
                        }
                        Err(std::sync::TryLockError::Poisoned(_)) => {
                            for sample in data {
                                *sample = $convert(0.0);
                            }
                            record_generation_failure(
                                &data_failure,
                                GenerationFailure::fatal("live playback queue lock poisoned")
                                    .after_start(),
                                &data_running,
                            );
                        }
                    }
                },
                move |error| {
                    let message = format!("output stream error: {error}");
                    record_generation_failure(
                        &stream_failure,
                        GenerationFailure::recoverable(message.clone()),
                        &stream_running,
                    );
                    eprintln!("denoize: {message}");
                },
                None,
            )
        }};
    }
    let result = match format {
        SampleFormat::I8 => stream!(i8, |x: f32| output_sample_from_f32::<i8>(x)),
        SampleFormat::F32 => stream!(f32, |x: f32| output_sample_from_f32::<f32>(x)),
        SampleFormat::I16 => stream!(i16, |x: f32| output_sample_from_f32::<i16>(x)),
        SampleFormat::I32 => stream!(i32, |x: f32| output_sample_from_f32::<i32>(x)),
        SampleFormat::I64 => stream!(i64, |x: f32| output_sample_from_f32::<i64>(x)),
        SampleFormat::U8 => stream!(u8, |x: f32| output_sample_from_f32::<u8>(x)),
        SampleFormat::U16 => stream!(u16, |x: f32| output_sample_from_f32::<u16>(x)),
        SampleFormat::U32 => stream!(u32, |x: f32| output_sample_from_f32::<u32>(x)),
        SampleFormat::U64 => stream!(u64, |x: f32| output_sample_from_f32::<u64>(x)),
        SampleFormat::F64 => stream!(f64, |x: f32| output_sample_from_f32::<f64>(x)),
        other => return Err(format!("unsupported live output sample format: {other:?}")),
    };
    result.map_err(|e| format!("build output stream: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> LiveConfig {
        LiveConfig {
            input_device: None,
            output_device: None,
            chunk_ms: 100,
            backend: Backend::Classical,
            backend_options: BackendOptions::default(),
            denoiser: DenoiserConfig::default(0),
        }
    }

    fn processor_config(channel_mode: ChannelMode) -> LiveConfig {
        let mut config = config();
        config.denoiser.sample_rate = 48_000;
        config.denoiser.frame_size = 256;
        config.denoiser.overlap = 0.75;
        config.backend_options.channel_mode = channel_mode;
        config
    }

    #[cfg(feature = "rnnoise")]
    fn rnnoise_processor_config(channel_mode: ChannelMode) -> LiveConfig {
        let mut config = processor_config(channel_mode);
        config.backend = Backend::Rnnoise;
        config.denoiser.sample_rate = 44_100;
        config
    }

    fn stereo_signal(frames: usize) -> Vec<Vec<f64>> {
        vec![
            (0..frames)
                .map(|frame| (frame as f64 * 0.031).sin() * 0.08)
                .collect(),
            (0..frames)
                .map(|frame| (frame as f64 * 0.047).cos() * 0.06)
                .collect(),
        ]
    }

    fn process_partitions(
        config: &LiveConfig,
        input: &[Vec<f64>],
        partitions: &[usize],
    ) -> Vec<Vec<f64>> {
        let mut processor = LiveProcessor::new(config, input.len()).unwrap();
        let mut output = vec![Vec::new(); input.len()];
        let mut position = 0usize;
        let mut sequence = 0u64;
        let mut partition = 0usize;
        while position < input[0].len() {
            let frames = partitions[partition % partitions.len()]
                .min(input[0].len().saturating_sub(position));
            let block: Vec<Vec<f64>> = input
                .iter()
                .map(|channel| channel[position..position + frames].to_vec())
                .collect();
            let processed = processor.process_chunk(sequence, block).unwrap();
            assert!(!processed.reset_for_gap);
            for (output, block) in output.iter_mut().zip(processed.channels) {
                output.extend(block);
            }
            position += frames;
            sequence += 1;
            partition += 1;
        }
        output
    }

    #[test]
    fn session_processor_is_partition_invariant_for_all_channel_modes() {
        let input = stereo_signal(8_193);
        for mode in [
            ChannelMode::Independent,
            ChannelMode::StereoLinked,
            ChannelMode::MidSide,
        ] {
            let config = processor_config(mode);
            let contiguous = process_partitions(&config, &input, &[input[0].len()]);
            let irregular = process_partitions(&config, &input, &[1, 17, 3, 511, 64, 2, 997]);
            assert_eq!(irregular, contiguous, "partition mismatch for {mode:?}");
        }
    }

    #[test]
    fn oversized_live_blocks_split_at_the_stream_cap_without_changing_output() {
        let input = stereo_signal(1_003);
        for mode in [
            ChannelMode::Independent,
            ChannelMode::StereoLinked,
            ChannelMode::MidSide,
        ] {
            let config = processor_config(mode);
            let spec = LiveProcessorSpec {
                backend: config.backend,
                backend_options: config.backend_options.clone(),
                accelerator: AcceleratorSelection::default(),
                denoiser: config.denoiser.clone(),
                channels: 2,
            };
            let mut contiguous = StatefulLiveProcessor::new(&spec).unwrap();
            let mut split = StatefulLiveProcessor::new(&spec).unwrap();
            let expected = contiguous
                .processor
                .process_block_with_limit(&input, input[0].len())
                .unwrap();
            let actual = split
                .processor
                .process_block_with_limit(&input, 37)
                .unwrap();
            assert_eq!(actual, expected, "internal split mismatch for {mode:?}");
        }
    }

    #[test]
    fn linked_stream_keeps_original_side_samples_aligned_with_delayed_output() {
        let input = stereo_signal(4_097);
        let output = process_partitions(
            &processor_config(ChannelMode::StereoLinked),
            &input,
            &[37, 1, 509, 8, 113],
        );
        assert_eq!(output[0].len(), output[1].len());
        assert!(!output[0].is_empty());
        for frame in 0..output[0].len() {
            let original_side = input[0][frame] - input[1][frame];
            let processed_side = output[0][frame] - output[1][frame];
            assert!((processed_side - original_side).abs() < 1e-12);
        }
    }

    #[cfg(feature = "rnnoise")]
    #[test]
    fn rnnoise_live_resamplers_are_partition_invariant_in_all_channel_modes() {
        let input = stereo_signal(10_003);
        for mode in [
            ChannelMode::Independent,
            ChannelMode::StereoLinked,
            ChannelMode::MidSide,
        ] {
            let config = rnnoise_processor_config(mode);
            let contiguous = process_partitions(&config, &input, &[input[0].len()]);
            let irregular = process_partitions(&config, &input, &[1, 441, 7, 1_777, 32, 2, 997]);
            assert_eq!(irregular, contiguous, "RNNoise mismatch for {mode:?}");
            assert!(!irregular[0].is_empty());
        }
    }

    #[cfg(feature = "rnnoise")]
    #[test]
    fn rnnoise_gap_reset_is_cold_across_model_and_both_resamplers() {
        let config = rnnoise_processor_config(ChannelMode::StereoLinked);
        let prelude = stereo_signal(5_001);
        let post_gap = stereo_signal(7_003);
        let mut interrupted = LiveProcessor::new(&config, 2).unwrap();
        interrupted.process_chunk(0, prelude).unwrap();
        let reset = interrupted.process_chunk(3, post_gap.clone()).unwrap();
        assert!(reset.reset_for_gap);

        let mut cold = LiveProcessor::new(&config, 2).unwrap();
        let expected = cold.process_chunk(0, post_gap.clone()).unwrap();
        assert_eq!(reset.channels, expected.channels);
        for frame in 0..reset.channels[0].len() {
            let original_side = post_gap[0][frame] - post_gap[1][frame];
            let processed_side = reset.channels[0][frame] - reset.channels[1][frame];
            assert!((processed_side - original_side).abs() < 1e-12);
        }
    }

    #[test]
    fn sequence_gap_replaces_every_session_state_with_a_cold_processor() {
        let config = processor_config(ChannelMode::MidSide);
        let prelude = stereo_signal(2_048);
        let post_gap = stereo_signal(3_001);
        let mut interrupted = LiveProcessor::new(&config, 2).unwrap();
        let first = interrupted.process_chunk(0, prelude).unwrap();
        assert!(!first.reset_for_gap);
        let reset = interrupted.process_chunk(2, post_gap.clone()).unwrap();
        assert!(reset.reset_for_gap);

        let mut cold = LiveProcessor::new(&config, 2).unwrap();
        let expected = cold.process_chunk(0, post_gap).unwrap();
        assert_eq!(reset.channels, expected.channels);
    }

    #[test]
    fn new_live_sessions_never_inherit_previous_dsp_state() {
        let config = processor_config(ChannelMode::Independent);
        let dirtying_input = stereo_signal(4_000);
        let session_input = stereo_signal(2_503);
        let mut old_session = LiveProcessor::new(&config, 2).unwrap();
        old_session.process_chunk(0, dirtying_input).unwrap();

        let mut first_new_session = LiveProcessor::new(&config, 2).unwrap();
        let mut second_new_session = LiveProcessor::new(&config, 2).unwrap();
        let first = first_new_session
            .process_chunk(0, session_input.clone())
            .unwrap();
        let second = second_new_session.process_chunk(0, session_input).unwrap();
        assert_eq!(first.channels, second.channels);
    }

    #[test]
    fn automatic_profile_live_bootstrap_emits_from_the_first_chunk() {
        let mut config = config();
        config.denoiser.sample_rate = 48_000;
        assert_eq!(config.denoiser.profile_ms, 0.0);
        let input = vec![(0..4_800)
            .map(|frame| (frame as f64 * 0.029).sin() * 0.1)
            .collect()];
        let mut processor = LiveProcessor::new(&config, 1).unwrap();
        let output = processor.process_chunk(0, input).unwrap().channels;
        assert!(
            !output[0].is_empty(),
            "automatic live profiling must not buffer 1.5 seconds"
        );
    }

    #[test]
    fn vad_retains_the_chunk_compatible_processing_path() {
        let mut config = processor_config(ChannelMode::Independent);
        config.denoiser.vad = true;
        let processor = LiveProcessor::new(&config, 1).unwrap();
        assert!(matches!(
            processor.kind,
            LiveProcessorKind::Compatibility(_)
        ));
    }

    #[test]
    fn peak_store_only_moves_upward() {
        let peak = AtomicU32::new(0.0_f32.to_bits());
        store_peak(&peak, 0.4);
        store_peak(&peak, 0.2);
        store_peak(&peak, 0.8);
        assert_eq!(f32::from_bits(peak.load(Ordering::Relaxed)), 0.8);
    }

    #[test]
    fn every_cpal_sample_format_has_centered_live_conversion() {
        assert_eq!(input_sample_to_f32(0_i8), 0.0);
        assert_eq!(input_sample_to_f32(0_i16), 0.0);
        assert_eq!(input_sample_to_f32(0_i32), 0.0);
        assert_eq!(input_sample_to_f32(0_i64), 0.0);
        assert_eq!(input_sample_to_f32(128_u8), 0.0);
        assert_eq!(input_sample_to_f32(32_768_u16), 0.0);
        assert_eq!(input_sample_to_f32(2_147_483_648_u32), 0.0);
        assert_eq!(input_sample_to_f32(9_223_372_036_854_775_808_u64), 0.0);
        assert_eq!(input_sample_to_f32(0.25_f32), 0.25);
        assert_eq!(input_sample_to_f32(0.25_f64), 0.25);

        assert_eq!(output_sample_from_f32::<i8>(0.0), 0);
        assert_eq!(output_sample_from_f32::<i16>(0.0), 0);
        assert_eq!(output_sample_from_f32::<i32>(0.0), 0);
        assert_eq!(output_sample_from_f32::<i64>(0.0), 0);
        assert_eq!(output_sample_from_f32::<u8>(0.0), 128);
        assert_eq!(output_sample_from_f32::<u16>(0.0), 32_768);
        assert_eq!(output_sample_from_f32::<u32>(0.0), 2_147_483_648);
        assert_eq!(
            output_sample_from_f32::<u64>(0.0),
            9_223_372_036_854_775_808
        );
        assert_eq!(output_sample_from_f32::<f32>(f32::NAN), 0.0);
        assert_eq!(output_sample_from_f32::<f64>(0.25), 0.25);
    }

    #[test]
    fn named_device_selection_rejects_missing_and_ambiguous_matches() {
        assert_eq!(
            select_unique_named_device(
                [("first".to_string(), 1_u8), ("second".to_string(), 2)],
                "second",
                "input",
            ),
            Ok(2)
        );
        assert_eq!(
            select_unique_named_device([("first".to_string(), 1_u8)], "missing", "input")
                .unwrap_err(),
            "input device not found: missing"
        );
        assert_eq!(
            select_unique_named_device(
                [("same".to_string(), 1_u8), ("same".to_string(), 2)],
                "same",
                "output",
            )
            .unwrap_err(),
            "output device name is ambiguous: same (multiple exact matches)"
        );
    }

    #[test]
    fn resilience_policy_has_bounded_explicit_and_chunk_aware_defaults() {
        let automatic = LiveResilienceConfig::default();
        assert_eq!(automatic.resolved_target_latency_ms(10), Ok(40));
        assert_eq!(automatic.resolved_target_latency_ms(100), Ok(200));
        assert!(automatic.validate().is_ok());

        let explicit = LiveResilienceConfig::new()
            .with_target_latency_ms(MIN_TARGET_LATENCY_MS)
            .with_max_drift_ppm(MAX_DRIFT_PPM)
            .with_reconnect_timeout_ms(MAX_RECONNECT_TIMEOUT_MS);
        assert_eq!(
            explicit.resolved_target_latency_ms(MAX_CHUNK_MS),
            Ok(MIN_TARGET_LATENCY_MS)
        );
        for invalid in [
            LiveResilienceConfig::new().with_target_latency_ms(MIN_TARGET_LATENCY_MS - 1),
            LiveResilienceConfig::new().with_target_latency_ms(MAX_TARGET_LATENCY_MS + 1),
            LiveResilienceConfig::new().with_max_drift_ppm(MAX_DRIFT_PPM + 1),
            LiveResilienceConfig::new().with_reconnect_timeout_ms(MAX_RECONNECT_TIMEOUT_MS + 1),
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn asynchronous_plan_accepts_independent_device_rates() {
        let config = config();
        let plan = plan_live_buffers_with_resilience(
            &config,
            LiveResilienceConfig::default(),
            44_100,
            48_000,
            2,
            2,
        )
        .unwrap();
        assert_eq!(plan.chunk_frames, 4_410);
        assert_eq!(plan.input_capacity, 8_820);
        assert_eq!(plan.target_queue_frames, 9_600);
        assert!(plan.maximum_resampled_frames >= 4_810);
        assert!(plan.resampler_delay_frames > 0);
        assert!(plan.required_bytes > 0);

        assert!(matches!(
            plan_live_buffers_with_resilience(
                &config,
                LiveResilienceConfig::default(),
                0,
                48_000,
                1,
                1,
            ),
            Err(ConfigError::InvalidValue {
                field: "input_sample_rate",
                ..
            })
        ));
    }

    #[test]
    fn asynchronous_resampler_tracks_ratio_and_keeps_channels_aligned() {
        let config = config();
        let plan = plan_live_buffers_with_resilience(
            &config,
            LiveResilienceConfig::default(),
            44_100,
            48_000,
            2,
            2,
        )
        .unwrap();
        let input = vec![
            (0..plan.chunk_frames)
                .map(|frame| (frame as f64 * 0.017).sin())
                .collect::<Vec<_>>(),
            (0..plan.chunk_frames)
                .map(|frame| -(frame as f64 * 0.017).sin())
                .collect::<Vec<_>>(),
        ];
        let mut faster = AdaptiveClockResampler::new(
            44_100,
            48_000,
            plan.chunk_frames,
            2,
            plan.maximum_resampled_frames,
            DEFAULT_MAX_DRIFT_PPM,
        )
        .unwrap();
        let mut slower = AdaptiveClockResampler::new(
            44_100,
            48_000,
            plan.chunk_frames,
            2,
            plan.maximum_resampled_frames,
            DEFAULT_MAX_DRIFT_PPM,
        )
        .unwrap();
        let grown = faster
            .process(&input, DEFAULT_MAX_DRIFT_PPM as f64)
            .unwrap();
        let shrunk = slower
            .process(&input, -(DEFAULT_MAX_DRIFT_PPM as f64))
            .unwrap();
        assert!(grown[0].len() >= shrunk[0].len());
        assert!(grown[0].len() <= plan.maximum_resampled_frames);
        assert_eq!(grown[0].len(), grown[1].len());
        for (left, right) in grown[0].iter().zip(&grown[1]) {
            assert!((left + right).abs() < 1e-10);
        }
    }

    #[test]
    fn asynchronous_resampler_plan_covers_supported_rate_directions() {
        let config = config();
        for (input_rate, output_rate) in [
            (8_000, 192_000),
            (192_000, 8_000),
            (44_100, 48_000),
            (48_000, 44_100),
            (96_000, 192_000),
        ] {
            for max_drift_ppm in [0, MAX_DRIFT_PPM] {
                let resilience = LiveResilienceConfig::new().with_max_drift_ppm(max_drift_ppm);
                let plan = plan_live_buffers_with_resilience(
                    &config,
                    resilience,
                    input_rate,
                    output_rate,
                    2,
                    2,
                )
                .unwrap();
                let converter = AdaptiveClockResampler::new(
                    input_rate,
                    output_rate,
                    plan.chunk_frames,
                    2,
                    plan.maximum_resampled_frames,
                    max_drift_ppm,
                )
                .unwrap();
                assert!(converter.converter.output_frames_max() <= plan.maximum_resampled_frames);
            }
        }
    }

    #[test]
    fn drift_controller_is_bounded_and_moves_queue_toward_target() {
        let mut controller = ClockDriftController::new(4_800, DEFAULT_MAX_DRIFT_PPM);
        let low = controller.update(0, 0.1);
        assert!(low > 0.0);
        assert!(low <= DEFAULT_MAX_DRIFT_PPM as f64);
        controller.reset();
        let high = controller.update(9_600, 0.1);
        assert!(high < 0.0);
        assert!(high >= -(DEFAULT_MAX_DRIFT_PPM as f64));
        controller.reset();
        assert_eq!(controller.update(4_800, 0.1), 0.0);

        let mut disabled = ClockDriftController::new(4_800, 0);
        assert_eq!(disabled.update(0, 1.0), 0.0);
    }

    #[test]
    fn drift_controller_converges_for_independent_clock_error() {
        const TARGET: f64 = 4_800.0;
        const FRAMES_PER_STEP: f64 = 480.0;
        const PLAYBACK_CLOCK_ERROR_PPM: f64 = 1_000.0;

        let mut controller = ClockDriftController::new(TARGET as usize, 2_500);
        let mut queued = TARGET;
        let mut correction = 0.0;
        for _ in 0..20_000 {
            correction = controller.update(queued.max(0.0) as usize, 0.01);
            queued += FRAMES_PER_STEP * (1.0 + correction / 1_000_000.0)
                - FRAMES_PER_STEP * (1.0 + PLAYBACK_CLOCK_ERROR_PPM / 1_000_000.0);
        }

        assert!((queued - TARGET).abs() < 10.0, "queue settled at {queued}");
        assert!(
            (correction - PLAYBACK_CLOCK_ERROR_PPM).abs() < 100.0,
            "correction settled at {correction} ppm"
        );
    }

    #[test]
    fn reconnect_schedule_is_finite_exponential_and_resets_after_a_live_generation() {
        let start = Instant::now();
        let failure = GenerationFailure::recoverable("device missing");
        let mut schedule = RecoverySchedule::new(1_000);
        assert_eq!(
            schedule.schedule(&failure, start).unwrap(),
            Duration::from_millis(100)
        );
        assert_eq!(
            schedule
                .schedule(&failure, start + Duration::from_millis(50))
                .unwrap(),
            Duration::from_millis(200)
        );
        assert_eq!(schedule.reconnect_attempts, 2);
        assert_eq!(schedule.device_generation, 3);

        let interrupted = GenerationFailure::recoverable("unplugged").after_start();
        assert_eq!(
            schedule
                .schedule(&interrupted, start + Duration::from_millis(500))
                .unwrap(),
            Duration::from_millis(100)
        );
        assert!(schedule
            .schedule(&failure, start + Duration::from_millis(1_500))
            .unwrap_err()
            .contains("timed out"));

        assert_eq!(
            RecoverySchedule::new(0)
                .schedule(&failure, start)
                .unwrap_err(),
            "device missing"
        );
        assert_eq!(
            RecoverySchedule::new(1_000)
                .schedule(&GenerationFailure::fatal("processor failed"), start)
                .unwrap_err(),
            "processor failed"
        );
    }

    #[test]
    fn latency_status_combines_device_queue_processing_and_algorithmic_delay() {
        let config = config();
        let plan = plan_live_buffers(&config, 48_000, 1, 2).unwrap();
        let playback = Mutex::new(VecDeque::from(vec![0.0; plan.target_queue_frames * 2]));
        let metrics = LiveMetrics::new();
        metrics
            .input_level
            .store(0.5f32.to_bits(), Ordering::Relaxed);
        metrics
            .output_level
            .store(0.25f32.to_bits(), Ordering::Relaxed);
        metrics
            .input_device_latency_us
            .store(2_000, Ordering::Relaxed);
        metrics
            .output_device_latency_us
            .store(3_000, Ordering::Relaxed);
        metrics
            .processing_latency_us
            .store(1_000, Ordering::Relaxed);
        metrics
            .drift_correction_bits
            .store(125.0f64.to_bits(), Ordering::Relaxed);
        let accelerator =
            select_accelerator_for_options(config.backend, &config.backend_options).unwrap();
        let status = runtime_status(
            LiveConnectionState::Running,
            48_000,
            48_000,
            1,
            2,
            &plan,
            &playback,
            &metrics,
            accelerator,
            2,
            3,
        );
        assert_eq!(status.queued_frames, plan.target_queue_frames);
        assert_eq!(status.drift_correction_ppm, 125.0);
        assert_eq!(status.reconnect_attempts, 2);
        assert_eq!(status.device_generation, 3);
        assert_eq!(status.connection_state, LiveConnectionState::Running);
        assert!(status.estimated_total_latency_ms > status.queue_latency_ms + 5.0);
        assert_eq!(status.input_level, 0.5);
        assert_eq!(status.output_level, 0.25);
    }

    #[test]
    fn classical_ready_burst_plan_covers_frame_quantization_and_profiles() {
        let mut config = config();
        config.chunk_ms = MIN_CHUNK_MS;
        config.denoiser.sample_rate = 8_000;
        config.denoiser.frame_size = 4_096;
        config.denoiser.overlap = 0.5;
        let plan = plan_live_buffers(&config, 8_000, 1, 1).unwrap();
        assert_eq!(plan.chunk_frames, 80);
        assert_eq!(maximum_ready_burst_frames(&config, 8_000, 80), Ok(4_176));
        assert!(plan.maximum_resampled_frames >= plan.chunk_frames);
        assert!(plan.queue_capacity >= 4_816);

        let mut processor = LiveProcessor::new(&config, 1).unwrap();
        let mut largest = Vec::new();
        for sequence in 0..64u64 {
            let input = vec![(0..80)
                .map(|frame| ((sequence * 80 + frame as u64) as f64 * 0.019).sin() * 0.05)
                .collect()];
            let ready = processor.process_chunk(sequence, input).unwrap().channels;
            if ready[0].len() > largest.len() {
                largest = ready[0].clone();
            }
        }
        let old_capacity = 80 * PLAYBACK_QUEUE_CHUNKS as usize;
        assert_eq!(largest.len(), 2_048);
        assert!(largest.len() > old_capacity);
        let level = AtomicU32::new(0.0f32.to_bits());
        let mut old_queue = VecDeque::new();
        assert!(enqueue_playback_block(
            &mut old_queue,
            &[largest.clone()],
            1,
            old_capacity,
            &level
        )
        .is_err());
        assert!(old_queue.is_empty());
        let mut planned_queue = VecDeque::new();
        enqueue_playback_block(
            &mut planned_queue,
            &[largest],
            1,
            plan.queue_capacity,
            &level,
        )
        .unwrap();

        let mut full_planned_queue = VecDeque::from(vec![0.0; plan.queue_capacity]);
        assert_eq!(full_planned_queue.len(), plan.queue_capacity);
        let discarded = enqueue_playback_block(
            &mut full_planned_queue,
            &[vec![0.0]],
            1,
            plan.queue_capacity,
            &level,
        )
        .unwrap();
        assert_eq!(discarded, 1);
        assert_eq!(full_planned_queue.len(), plan.queue_capacity);

        config.denoiser.profile_ms = 100.0;
        assert_eq!(maximum_ready_burst_frames(&config, 8_000, 80), Ok(4_976));
        assert!(
            plan_live_buffers(&config, 8_000, 1, 1)
                .unwrap()
                .queue_capacity
                >= 5_616
        );
    }

    #[test]
    fn playback_overflow_discards_only_oldest_complete_frames() {
        let level = AtomicU32::new(0.0f32.to_bits());
        let mut queue = VecDeque::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let audio = vec![vec![10.0, 20.0, 30.0], vec![11.0, 21.0, 31.0]];

        let overflow = enqueue_playback_block(&mut queue, &audio, 2, 8, &level).unwrap();

        assert_eq!(overflow, 2);
        assert_eq!(queue.len(), 8);
        assert_eq!(
            queue.into_iter().collect::<Vec<_>>(),
            vec![4.0, 5.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]
        );
    }

    #[cfg(feature = "rnnoise")]
    #[test]
    fn rnnoise_low_rate_ready_burst_plan_covers_both_resamplers() {
        let mut config = config();
        config.backend = Backend::Rnnoise;
        config.chunk_ms = MIN_CHUNK_MS;
        config.denoiser.sample_rate = 8_000;
        config.denoiser.profile_ms = -1.0;
        let plan = plan_live_buffers(&config, 8_000, 1, 1).unwrap();
        assert_eq!(maximum_ready_burst_frames(&config, 8_000, 80), Ok(2_253));
        assert!(plan.queue_capacity >= 2_893);

        let mut processor = LiveProcessor::new(&config, 1).unwrap();
        let mut largest = Vec::new();
        for sequence in 0..160u64 {
            let input = vec![(0..80)
                .map(|frame| ((sequence * 80 + frame as u64) as f64 * 0.023).sin() * 0.05)
                .collect()];
            let ready = processor.process_chunk(sequence, input).unwrap().channels;
            if ready[0].len() > largest.len() {
                largest = ready[0].clone();
            }
        }
        let old_capacity = 80 * PLAYBACK_QUEUE_CHUNKS as usize;
        assert!(largest.len() > old_capacity, "largest={}", largest.len());
        assert!(largest.len() <= 2_253);
        let level = AtomicU32::new(0.0f32.to_bits());
        let mut queue = VecDeque::new();
        enqueue_playback_block(&mut queue, &[largest], 1, plan.queue_capacity, &level).unwrap();
    }

    #[test]
    fn live_config_accepts_chunk_boundaries_and_a_rate_placeholder() {
        for chunk_ms in [MIN_CHUNK_MS, MAX_CHUNK_MS] {
            let mut config = config();
            config.chunk_ms = chunk_ms;
            config.denoiser.sample_rate = u32::MAX;
            assert!(config.validate_config().is_ok());
        }
        for chunk_ms in [MIN_CHUNK_MS - 1, MAX_CHUNK_MS + 1] {
            let mut config = config();
            config.chunk_ms = chunk_ms;
            assert!(matches!(
                config.validate_config(),
                Err(ConfigError::InvalidValue {
                    field: "chunk_ms",
                    ..
                })
            ));
        }
    }

    #[test]
    fn live_config_rejects_core_nan_and_model_rate_before_hardware() {
        let mut invalid_core = config();
        invalid_core.denoiser.strength = f64::NAN;
        assert!(matches!(
            invalid_core.validate_config(),
            Err(ConfigError::InvalidValue {
                field: "strength",
                ..
            })
        ));

        let mut invalid_model = config();
        invalid_model.backend_options.onnx = Some(crate::OnnxModelConfig {
            path: "model-that-must-not-be-opened.onnx".into(),
            sample_rate: 0,
        });
        assert!(matches!(
            invalid_model.validate_config(),
            Err(ConfigError::InvalidValue {
                field: "backend_options.onnx.sample_rate",
                ..
            })
        ));
    }

    #[cfg(feature = "gtcrn")]
    #[test]
    fn gtcrn_is_live_capable_but_rejects_chunk_compatibility_vad() {
        let mut config = config();
        config.backend = Backend::Gtcrn;
        config.backend_options.onnx = Some(crate::OnnxModelConfig {
            path: "model-path-is-not-opened-during-validation.onnx".into(),
            sample_rate: crate::backend::gtcrn::SAMPLE_RATE,
        });
        assert!(backend_is_live_capable(Backend::Gtcrn));
        assert!(config.validate_config().is_ok());
        config.denoiser.vad = true;
        assert!(matches!(
            config.validate_config(),
            Err(ConfigError::InvalidValue { field: "vad", .. })
        ));
    }

    #[cfg(any(feature = "deepfilter", feature = "mossformer2"))]
    #[test]
    fn bounded_file_stream_backends_do_not_claim_low_latency_live_support() {
        #[cfg(feature = "deepfilter")]
        assert!(!backend_is_live_capable(Backend::DeepFilter));
        #[cfg(feature = "mossformer2")]
        assert!(!backend_is_live_capable(Backend::Mossformer2));
    }

    #[test]
    fn hardware_plan_uses_effective_rate_and_checked_capacities() {
        let mut config = config();
        let normal = plan_live_buffers(&config, 48_000, 2, 2).unwrap();
        assert_eq!(normal.chunk_frames, 4_800);
        assert_eq!(normal.input_capacity, 9_600);
        assert!(normal.queue_capacity >= 90_496);
        assert!(normal.target_queue_frames > 0);
        assert!(normal.resampler_delay_frames > 0);
        assert!(normal.required_bytes > 0);
        assert!(plan_live_buffers(&config, crate::config::MAX_SAMPLE_RATE, 1, 1).is_ok());
        config.chunk_ms = MAX_CHUNK_MS;
        let oversized = plan_live_buffers(&config, crate::config::MAX_SAMPLE_RATE, 1, 1).unwrap();
        assert_eq!(oversized.chunk_frames, 1_536_000);
        assert!(oversized.chunk_frames > MAX_STREAM_BLOCK_FRAMES);
        config.chunk_ms = 100;
        for rate in [0, crate::config::MAX_SAMPLE_RATE + 1] {
            assert!(matches!(
                plan_live_buffers(&config, rate, 1, 1),
                Err(ConfigError::InvalidValue {
                    field: "sample_rate",
                    ..
                })
            ));
        }
        assert!(plan_live_buffers(
            &config,
            48_000,
            crate::config::MAX_STREAM_CHANNELS,
            crate::config::MAX_STREAM_CHANNELS,
        )
        .is_ok());
        for channels in [0, crate::config::MAX_STREAM_CHANNELS + 1] {
            assert!(plan_live_buffers(&config, 48_000, channels, 1).is_err());
            assert!(matches!(
                plan_live_buffers(&config, 48_000, 1, channels),
                Err(ConfigError::InvalidValue {
                    field: "output_channels",
                    ..
                })
            ));
        }
    }

    #[cfg(feature = "rnnoise")]
    #[test]
    fn rnnoise_resampler_is_validated_before_live_setup() {
        let mut config = config();
        config.backend = Backend::Rnnoise;

        assert!(matches!(
            plan_live_buffers(&config, 1, 1, 1),
            Err(ConfigError::InvalidValue {
                field: "sample_rate",
                ..
            })
        ));
        assert!(plan_live_buffers(&config, 48_000, 1, 1).is_ok());

        config.chunk_ms = MAX_CHUNK_MS;
        config.denoiser.profile_ms = -1.0;
        assert!(matches!(
            plan_live_buffers(&config, 767_999, 1, 1),
            Err(ConfigError::ResourceLimitExceeded {
                resource: "live working set",
                ..
            })
        ));
    }

    #[test]
    fn aggregate_live_buffers_share_the_streaming_hard_limit() {
        let mut config = config();
        config.chunk_ms = MAX_CHUNK_MS;
        assert!(matches!(
            plan_live_buffers(
                &config,
                crate::config::MAX_SAMPLE_RATE,
                1,
                crate::config::MAX_STREAM_CHANNELS,
            ),
            Err(ConfigError::ResourceLimitExceeded {
                resource: "live working set",
                ..
            })
        ));

        // This fits if only the callback and worker's current chunks are
        // counted, but exceeds the limit once all four bounded channel slots
        // and both simultaneously-live f64 Audio buffers are included.
        assert!(matches!(
            plan_live_buffers(&config, 96_000, crate::config::MAX_STREAM_CHANNELS, 1,),
            Err(ConfigError::ResourceLimitExceeded {
                resource: "live working set",
                ..
            })
        ));

        let mut vad = config;
        vad.denoiser.vad = true;
        vad.denoiser.profile_ms = -1.0;
        let mut without_vad = vad.clone();
        without_vad.denoiser.vad = false;
        let boundary = (1..=crate::config::MAX_STREAM_CHANNELS)
            .find(|channels| {
                plan_live_buffers(&without_vad, 96_000, *channels, 1).is_ok()
                    && matches!(
                        plan_live_buffers(&vad, 96_000, *channels, 1),
                        Err(ConfigError::ResourceLimitExceeded {
                            resource: "live working set",
                            ..
                        })
                    )
            })
            .expect("VAD copies must cross the aggregate cap before the base worker");
        assert!(boundary > 1);
    }

    #[test]
    fn invalid_config_precedes_cpal_device_selection() {
        let mut config = config();
        config.chunk_ms = MIN_CHUNK_MS - 1;
        config.input_device = Some("device-that-must-not-be-enumerated".into());
        let error = run_with_status(config, Arc::new(AtomicBool::new(false)), |_| {}).unwrap_err();
        assert!(error.contains("`chunk_ms`"), "unexpected error: {error}");
    }

    #[test]
    fn live_accelerator_is_resolved_before_cpal_device_selection() {
        let mut config = config();
        config.backend_options.accelerator = crate::AcceleratorPreference::Auto;
        let prepared = PreparedLiveConfig::new(config.clone()).unwrap();
        assert_eq!(
            prepared.accelerator.fallback(),
            Some(crate::AcceleratorFallback::BackendCpuOnly)
        );

        config.backend_options.accelerator = crate::AcceleratorPreference::Gpu;
        config.input_device = Some("device-that-must-not-be-enumerated".into());
        let error = run_with_status(config, Arc::new(AtomicBool::new(false)), |_| {}).unwrap_err();
        assert!(
            error.contains("backend classical does not support accelerator gpu"),
            "unexpected error: {error}"
        );
        assert!(!error.contains("device not found"));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn unsupported_backend_precedes_model_and_cpal_device_selection() {
        let mut config = config();
        config.backend = Backend::Onnx;
        config.backend_options.onnx = Some(crate::OnnxModelConfig {
            path: "model-that-does-not-exist.onnx".into(),
            sample_rate: 48_000,
        });
        config.input_device = Some("device-that-must-not-be-enumerated".into());

        let preparation_error = PreparedLiveConfig::new(config.clone()).unwrap_err();
        assert!(
            preparation_error.contains("`backend`"),
            "unexpected error: {preparation_error}"
        );

        let error = run_with_status(config, Arc::new(AtomicBool::new(false)), |_| {}).unwrap_err();

        assert!(error.contains("`backend`"), "unexpected error: {error}");
        assert!(!error.contains("model does not exist"));
        assert!(!error.contains("device not found"));
    }
}
