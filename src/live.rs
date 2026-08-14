//! Realtime system-audio capture, denoising, and playback.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};

use crate::audio::Audio;
#[cfg(test)]
use crate::config::MAX_STREAM_BLOCK_FRAMES;
use crate::config::{
    checked_profile_target_samples, checked_resource_add, checked_resource_multiply, ConfigError,
    MAX_STREAM_CHANNELS, MAX_STREAM_STATE_BYTES,
};
use crate::denoiser::DenoiserConfig;
use crate::{
    denoise_audio_with_backend_config, select_accelerator, AcceleratorSelection, Backend,
    BackendOptions, ChannelMode, ResourcePlan, StreamingBackendSession,
};

const MIN_CHUNK_MS: u32 = 10;
const MAX_CHUNK_MS: u32 = 2_000;
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
    StreamingBackendSession::supports(backend)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveBufferPlan {
    chunk_frames: usize,
    input_capacity: usize,
    queue_capacity: usize,
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

#[cfg(any(feature = "rnnoise", feature = "gtcrn"))]
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

fn plan_live_buffers(
    config: &LiveConfig,
    sample_rate: u32,
    input_channels: usize,
    output_channels: usize,
) -> Result<LiveBufferPlan, ConfigError> {
    config.validate_config()?;
    if output_channels == 0 || output_channels > MAX_STREAM_CHANNELS {
        return Err(ConfigError::invalid(
            "output_channels",
            "an integer in 1..=64",
        ));
    }
    let mut denoiser = config.denoiser.clone();
    denoiser.sample_rate = sample_rate;
    denoiser.validate_config()?;
    let backend_additional_bytes = StreamingBackendSession::estimate_additional_bytes(
        config.backend,
        sample_rate,
        input_channels,
        config.backend_options.channel_mode,
    )?;
    let processor = ResourcePlan::for_stream(
        input_channels,
        denoiser.frame_size,
        sample_rate,
        denoiser.profile_ms,
    )?;

    let chunk_numerator = checked_resource_multiply(
        "live chunk frames",
        sample_rate as u64,
        config.chunk_ms as u64,
    )?;
    let chunk_frames_u64 = (chunk_numerator / 1_000).max(1);
    let ready_burst_frames = maximum_ready_burst_frames(config, sample_rate, chunk_frames_u64)?;
    let input_samples =
        checked_resource_multiply("live input buffer", chunk_frames_u64, input_channels as u64)?;
    let steady_queue_frames = checked_resource_multiply(
        "live playback queue",
        chunk_frames_u64,
        PLAYBACK_QUEUE_CHUNKS,
    )?;
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
    let worker_chunk_bytes = checked_resource_multiply(
        "live working set",
        input_samples,
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
    let worker_bytes = checked_resource_add("live working set", regular_worker_bytes, ready_bytes)?;
    let linked_alignment_bytes = if !denoiser.vad
        && input_channels == 2
        && config.backend_options.channel_mode == ChannelMode::StereoLinked
    {
        let retained_frames = ready_burst_frames.checked_sub(chunk_frames_u64).ok_or(
            ConfigError::ResourceOverflow {
                resource: "live linked alignment",
            },
        )?;
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
    let worker_and_alignment =
        checked_resource_add("live working set", worker_bytes, linked_alignment_bytes)?;
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
        required_bytes,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct LiveStatus {
    pub sample_rate: u32,
    pub input_channels: usize,
    pub output_channels: usize,
    pub chunk_frames: usize,
    pub input_level: f32,
    pub output_level: f32,
    pub processed_chunks: u64,
    pub dropped_chunks: u64,
    /// Concrete runtime used by the live processor.
    pub accelerator: AcceleratorSelection,
}

struct CapturedChunk {
    sequence: u64,
    samples: Vec<f32>,
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
        let accelerator = select_accelerator(
            config.backend,
            config.backend_options.accelerator,
            config.backend_options.deterministic,
        )?;
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
}

impl PreparedLiveConfig {
    /// Validate resources and capture the runtime that execution will use.
    pub fn new(mut config: LiveConfig) -> Result<Self, String> {
        config
            .validate_config()
            .map_err(|error| error.to_string())?;
        config.backend_options =
            crate::service::resolve_backend_options(config.backend, config.backend_options)?;
        let accelerator = select_accelerator(
            config.backend,
            config.backend_options.accelerator,
            config.backend_options.deterministic,
        )?;
        Ok(Self {
            config,
            accelerator,
        })
    }

    /// Return the concrete runtime captured during preparation.
    #[must_use]
    pub const fn accelerator(&self) -> AcceleratorSelection {
        self.accelerator
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
    let PreparedLiveConfig {
        mut config,
        accelerator,
    } = prepared;
    let host = cpal::default_host();
    let input = select_device(&host, true, config.input_device.as_deref())?;
    let output = select_device(&host, false, config.output_device.as_deref())?;
    let input_supported = input
        .default_input_config()
        .map_err(|e| format!("input config: {e}"))?;
    let output_supported = output
        .default_output_config()
        .map_err(|e| format!("output config: {e}"))?;
    let input_cfg: StreamConfig = input_supported.clone().into();
    let output_cfg: StreamConfig = output_supported.clone().into();
    if input_cfg.sample_rate != output_cfg.sample_rate {
        return Err(format!(
            "input/output sample rates differ ({} vs {} Hz); select devices with a common default rate",
            input_cfg.sample_rate.0, output_cfg.sample_rate.0
        ));
    }

    let rate = input_cfg.sample_rate.0;
    let in_channels = input_cfg.channels as usize;
    let out_channels = output_cfg.channels as usize;
    let buffer_plan = plan_live_buffers(&config, rate, in_channels, out_channels)
        .map_err(|error| error.to_string())?;
    let mut worker_request = crate::ResourceRequest::worker(buffer_plan.required_bytes, 0);
    if accelerator.effective() != crate::AcceleratorRuntime::Cpu {
        worker_request = worker_request.with_gpu_jobs(1).with_gpu_memory_bytes(
            buffer_plan
                .required_bytes
                .checked_mul(2)
                .ok_or_else(|| "live GPU reservation overflow".to_string())?,
        );
    }
    let request = worker_request.checked_add(crate::estimate_backend_session_request(
        config.backend,
        &config.backend_options,
        accelerator,
    )?)?;
    let _resource_permit = governor
        .map(|governor| governor.acquire(request))
        .transpose()?;
    config.denoiser.sample_rate = rate;
    let chunk_frames = buffer_plan.chunk_frames;
    let queue_capacity = buffer_plan.queue_capacity;
    let mut playback_queue = VecDeque::<f32>::new();
    playback_queue
        .try_reserve_exact(queue_capacity)
        .map_err(|_| ConfigError::allocation_failed("live playback queue").to_string())?;
    let playback = Arc::new(Mutex::new(playback_queue));
    let mut pending_input = Vec::<f32>::new();
    pending_input
        .try_reserve_exact(buffer_plan.input_capacity)
        .map_err(|_| ConfigError::allocation_failed("live input buffer").to_string())?;
    let (tx, rx) = mpsc::sync_channel::<CapturedChunk>(CAPTURE_QUEUE_CHUNKS);
    let live_processor = LiveProcessor::new_with_accelerator(&config, in_channels, accelerator)?;
    let input_level = Arc::new(AtomicU32::new(0));
    let output_level = Arc::new(AtomicU32::new(0));
    let dropped_chunks = Arc::new(AtomicU64::new(0));
    let processed_chunks = Arc::new(AtomicU64::new(0));
    let worker_error = Arc::new(Mutex::new(None::<String>));
    let worker_running = Arc::clone(&running);
    let worker_playback = Arc::clone(&playback);
    let worker_output_level = Arc::clone(&output_level);
    let worker_processed = Arc::clone(&processed_chunks);
    let worker_failure = Arc::clone(&worker_error);
    let worker = std::thread::spawn(move || {
        let mut live_processor = live_processor;
        while worker_running.load(Ordering::Relaxed) {
            let captured = match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(captured) => captured,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };
            let mut channels = Vec::new();
            if channels.try_reserve_exact(in_channels).is_err() {
                if let Ok(mut error) = worker_failure.lock() {
                    *error =
                        Some(ConfigError::allocation_failed("live worker channels").to_string());
                }
                worker_running.store(false, Ordering::Relaxed);
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
                if let Ok(mut error) = worker_failure.lock() {
                    *error =
                        Some(ConfigError::allocation_failed("live worker samples").to_string());
                }
                worker_running.store(false, Ordering::Relaxed);
                break;
            }
            for frame in captured.samples.chunks_exact(in_channels) {
                for (channel, sample) in channels.iter_mut().zip(frame) {
                    channel.push(*sample as f64);
                }
            }
            let processed = match live_processor.process_chunk(captured.sequence, channels) {
                Ok(processed) => processed,
                Err(error) => {
                    if let Ok(mut failure) = worker_failure.lock() {
                        *failure = Some(error);
                    }
                    worker_running.store(false, Ordering::Relaxed);
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
                        &worker_output_level,
                    )
                }
                Err(_) => Err("live playback queue lock poisoned".into()),
            };
            if let Err(error) = enqueue_result {
                if let Ok(mut failure) = worker_failure.lock() {
                    *failure = Some(error);
                }
                worker_running.store(false, Ordering::Relaxed);
                break;
            }
            worker_processed.fetch_add(1, Ordering::Relaxed);
        }
    });

    let input_stream = build_input(
        &input,
        &input_cfg,
        input_supported.sample_format(),
        tx,
        chunk_frames,
        pending_input,
        Arc::clone(&input_level),
        Arc::clone(&dropped_chunks),
        Arc::clone(&worker_error),
        Arc::clone(&running),
    )?;
    let output_stream = build_output(
        &output,
        &output_cfg,
        output_supported.sample_format(),
        playback,
        Arc::clone(&worker_error),
        Arc::clone(&running),
    )?;
    output_stream
        .play()
        .map_err(|e| format!("start output: {e}"))?;
    input_stream
        .play()
        .map_err(|e| format!("start input: {e}"))?;
    let fallback = accelerator
        .fallback()
        .map(|reason| format!(", fallback {}", reason.name()))
        .unwrap_or_default();
    eprintln!(
        "denoize: live at {rate} Hz, {in_channels} input channel(s), {chunk_frames} frames/chunk; accelerator {}{fallback}; press Ctrl-C to stop",
        accelerator.effective().name()
    );
    while running.load(Ordering::Relaxed) && !worker.is_finished() {
        std::thread::sleep(Duration::from_millis(100));
        report(LiveStatus {
            sample_rate: rate,
            input_channels: in_channels,
            output_channels: out_channels,
            chunk_frames,
            input_level: f32::from_bits(input_level.swap(0, Ordering::Relaxed)),
            output_level: f32::from_bits(output_level.swap(0, Ordering::Relaxed)),
            processed_chunks: processed_chunks.load(Ordering::Relaxed),
            dropped_chunks: dropped_chunks.load(Ordering::Relaxed),
            accelerator,
        });
    }
    drop(input_stream);
    drop(output_stream);
    worker
        .join()
        .map_err(|_| "live worker panicked".to_string())?;
    if let Some(error) = worker_error
        .lock()
        .map_err(|_| "live worker status lock poisoned".to_string())?
        .take()
    {
        return Err(error);
    }
    Ok(())
}

fn select_device(
    host: &cpal::Host,
    input: bool,
    requested: Option<&str>,
) -> Result<Device, String> {
    if let Some(name) = requested {
        let devices = if input {
            host.input_devices()
        } else {
            host.output_devices()
        }
        .map_err(|e| format!("enumerate devices: {e}"))?;
        return devices
            .filter_map(|device| device.name().ok().map(|n| (n, device)))
            .find(|(n, _)| n == name)
            .map(|(_, device)| device)
            .ok_or_else(|| {
                format!(
                    "{} device not found: {name}",
                    if input { "input" } else { "output" }
                )
            });
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

fn build_input(
    device: &Device,
    cfg: &StreamConfig,
    format: SampleFormat,
    tx: mpsc::SyncSender<CapturedChunk>,
    chunk_frames: usize,
    pending: Vec<f32>,
    input_level: Arc<AtomicU32>,
    dropped_chunks: Arc<AtomicU64>,
    session_error: Arc<Mutex<Option<String>>>,
    running: Arc<AtomicBool>,
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
            let data_session_error = Arc::clone(&session_error);
            let stream_session_error = Arc::clone(&session_error);
            let data_running = Arc::clone(&running);
            let stream_running = Arc::clone(&running);
            device.build_input_stream(
                cfg,
                move |data: &[$ty], _| {
                    for sample in data.iter().map($convert) {
                        pending.push(sample);
                        if pending.len() == capacity {
                            let sequence = next_sequence;
                            let Some(following) = next_sequence.checked_add(1) else {
                                if let Ok(mut failure) = data_session_error.lock() {
                                    if failure.is_none() {
                                        *failure = Some("live capture sequence exhausted".into());
                                    }
                                }
                                data_running.store(false, Ordering::Relaxed);
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
                    if let Ok(mut failure) = stream_session_error.lock() {
                        if failure.is_none() {
                            *failure = Some(message.clone());
                        }
                    }
                    stream_running.store(false, Ordering::Relaxed);
                    eprintln!("denoize: {message}");
                },
                None,
            )
        }};
    }
    let result = match format {
        SampleFormat::F32 => stream!(f32, |x: &f32| *x),
        SampleFormat::I16 => stream!(i16, |x: &i16| *x as f32 / 32768.0),
        SampleFormat::U16 => stream!(u16, |x: &u16| *x as f32 / 32767.5 - 1.0),
        other => return Err(format!("unsupported live input sample format: {other:?}")),
    };
    result.map_err(|e| format!("build input stream: {e}"))
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
) -> Result<(), String> {
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
    if required > queue_capacity {
        return Err(format!(
            "live playback queue invariant exceeded: {required} samples ready, capacity {queue_capacity}"
        ));
    }
    for frame in 0..frames {
        for out_ch in 0..output_channels {
            let source = out_ch.min(audio.len() - 1);
            let sample = audio[source][frame] as f32;
            store_peak(output_level, sample.abs());
            queue.push_back(sample);
        }
    }
    Ok(())
}

fn build_output(
    device: &Device,
    cfg: &StreamConfig,
    format: SampleFormat,
    queue: Arc<Mutex<VecDeque<f32>>>,
    session_error: Arc<Mutex<Option<String>>>,
    running: Arc<AtomicBool>,
) -> Result<Stream, String> {
    macro_rules! stream {
        ($ty:ty, $convert:expr) => {{
            let queue = Arc::clone(&queue);
            let session_error = Arc::clone(&session_error);
            let running = Arc::clone(&running);
            device.build_output_stream(
                cfg,
                move |data: &mut [$ty], _| {
                    if let Ok(mut queue) = queue.lock() {
                        for sample in data {
                            *sample = $convert(queue.pop_front().unwrap_or(0.0));
                        }
                    }
                },
                move |error| {
                    let message = format!("output stream error: {error}");
                    if let Ok(mut failure) = session_error.lock() {
                        if failure.is_none() {
                            *failure = Some(message.clone());
                        }
                    }
                    running.store(false, Ordering::Relaxed);
                    eprintln!("denoize: {message}");
                },
                None,
            )
        }};
    }
    let result = match format {
        SampleFormat::F32 => stream!(f32, |x: f32| x),
        SampleFormat::I16 => stream!(i16, |x: f32| {
            (crate::audio::sanitize_sample(x as f64) * 32767.0) as i16
        }),
        SampleFormat::U16 => stream!(u16, |x: f32| {
            ((crate::audio::sanitize_sample(x as f64) + 1.0) * 32767.5) as u16
        }),
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
    fn classical_ready_burst_plan_covers_frame_quantization_and_profiles() {
        let mut config = config();
        config.chunk_ms = MIN_CHUNK_MS;
        config.denoiser.sample_rate = 8_000;
        config.denoiser.frame_size = 4_096;
        config.denoiser.overlap = 0.5;
        let plan = plan_live_buffers(&config, 8_000, 1, 1).unwrap();
        assert_eq!(plan.chunk_frames, 80);
        assert_eq!(maximum_ready_burst_frames(&config, 8_000, 80), Ok(4_176));
        assert_eq!(plan.queue_capacity, 4_816);

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

        let mut full_planned_queue = VecDeque::from(vec![0.0; old_capacity]);
        enqueue_playback_block(
            &mut full_planned_queue,
            &[vec![0.0; 4_176]],
            1,
            plan.queue_capacity,
            &level,
        )
        .unwrap();
        assert_eq!(full_planned_queue.len(), plan.queue_capacity);
        assert!(enqueue_playback_block(
            &mut full_planned_queue,
            &[vec![0.0]],
            1,
            plan.queue_capacity,
            &level,
        )
        .is_err());
        assert_eq!(full_planned_queue.len(), plan.queue_capacity);

        config.denoiser.profile_ms = 100.0;
        assert_eq!(maximum_ready_burst_frames(&config, 8_000, 80), Ok(4_976));
        assert_eq!(
            plan_live_buffers(&config, 8_000, 1, 1)
                .unwrap()
                .queue_capacity,
            5_616
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
        assert_eq!(plan.queue_capacity, 2_893);

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

    #[test]
    fn hardware_plan_uses_effective_rate_and_checked_capacities() {
        let mut config = config();
        let normal = plan_live_buffers(&config, 48_000, 2, 2).unwrap();
        assert_eq!(normal.chunk_frames, 4_800);
        assert_eq!(normal.input_capacity, 9_600);
        assert_eq!(normal.queue_capacity, 90_496);
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
        assert!(plan_live_buffers(&without_vad, 96_000, 22, 1).is_ok());
        assert!(matches!(
            plan_live_buffers(&vad, 96_000, 22, 1),
            Err(ConfigError::ResourceLimitExceeded {
                resource: "live working set",
                ..
            })
        ));
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
