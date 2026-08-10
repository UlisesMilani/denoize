//! Realtime system-audio capture, denoising, and playback.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};

use crate::audio::Audio;
use crate::config::{
    checked_resource_add, checked_resource_multiply, ConfigError, MAX_STREAM_CHANNELS,
    MAX_STREAM_STATE_BYTES,
};
use crate::denoiser::DenoiserConfig;
use crate::{denoise_audio_with_backend_config, Backend, BackendOptions, ResourcePlan};

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
                "classical or rnnoise for realtime processing",
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
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveBufferPlan {
    chunk_frames: usize,
    input_capacity: usize,
    queue_capacity: usize,
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
    #[cfg(feature = "rnnoise")]
    let rnnoise_resampler_bytes = if config.backend == Backend::Rnnoise {
        let mut maximum = 0u64;
        for (from_rate, to_rate) in [(sample_rate, 48_000), (48_000, sample_rate)] {
            let bytes =
                crate::resample::resampler_plan_bytes(1, from_rate, to_rate).map_err(|_| {
                    ConfigError::invalid(
                        "sample_rate",
                        "a rate with a bounded RNNoise 48 kHz resampler plan",
                    )
                })?;
            maximum = maximum.max(bytes);
        }
        maximum
    } else {
        0
    };
    #[cfg(not(feature = "rnnoise"))]
    let rnnoise_resampler_bytes = 0u64;
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
    let input_samples =
        checked_resource_multiply("live input buffer", chunk_frames_u64, input_channels as u64)?;
    let queue_samples = checked_resource_multiply(
        "live playback queue",
        chunk_frames_u64,
        output_channels as u64,
    )?;
    let queue_samples =
        checked_resource_multiply("live playback queue", queue_samples, PLAYBACK_QUEUE_CHUNKS)?;

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
    let worker_bytes =
        checked_resource_multiply("live working set", worker_chunk_bytes, worker_audio_copies)?;
    let playback_bytes = checked_resource_multiply(
        "live working set",
        queue_samples,
        std::mem::size_of::<f32>() as u64,
    )?;
    let input_bytes = checked_resource_add("live working set", captured_bytes, worker_bytes)?;
    let buffer_bytes = checked_resource_add("live working set", input_bytes, playback_bytes)?;
    let stream_and_buffers = checked_resource_add(
        "live working set",
        processor.estimated_bytes(),
        buffer_bytes,
    )?;
    let required_bytes = checked_resource_add(
        "live working set",
        stream_and_buffers,
        rnnoise_resampler_bytes,
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
    let config = prepare_live_config(config)?;
    let running = Arc::new(AtomicBool::new(true));
    let _signal_session = register_ctrl_c_session(Arc::clone(&running))?;
    run_prepared_with_status(config, running, |_| {})
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
    let config = prepare_live_config(config)?;
    run_prepared_with_status(config, running, report)
}

fn prepare_live_config(mut config: LiveConfig) -> Result<LiveConfig, String> {
    config
        .validate_config()
        .map_err(|error| error.to_string())?;
    config.backend_options =
        crate::service::resolve_backend_options(config.backend, config.backend_options)?;
    Ok(config)
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

fn run_prepared_with_status<F>(
    mut config: LiveConfig,
    running: Arc<AtomicBool>,
    mut report: F,
) -> Result<(), String>
where
    F: FnMut(LiveStatus),
{
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
    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(CAPTURE_QUEUE_CHUNKS);
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
        while worker_running.load(Ordering::Relaxed) {
            let samples = match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(samples) => samples,
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
            for frame in samples.chunks_exact(in_channels) {
                for (channel, sample) in channels.iter_mut().zip(frame) {
                    channel.push(*sample as f64);
                }
            }
            let mut audio = Audio {
                sample_rate: rate,
                channels,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
                channel_mask: None,
            };
            if let Err(error) = denoise_audio_with_backend_config(
                &mut audio,
                config.denoiser.clone(),
                config.backend,
                &config.backend_options,
            ) {
                eprintln!("denoize: live processing error: {error}");
                if let Ok(mut failure) = worker_failure.lock() {
                    *failure = Some(error);
                }
                worker_running.store(false, Ordering::Relaxed);
                break;
            }
            let frames = audio.frames();
            if let Ok(mut queue) = worker_playback.lock() {
                for frame in 0..frames {
                    for out_ch in 0..out_channels {
                        let source = out_ch.min(audio.channels().saturating_sub(1));
                        if queue.len() == queue_capacity {
                            queue.pop_front();
                        }
                        let sample = audio.channels[source][frame] as f32;
                        store_peak(&worker_output_level, sample.abs());
                        queue.push_back(sample);
                    }
                }
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
    eprintln!("denoize: live at {rate} Hz, {in_channels} input channel(s), {chunk_frames} frames/chunk; press Ctrl-C to stop");
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
    tx: mpsc::SyncSender<Vec<f32>>,
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
            let session_error = Arc::clone(&session_error);
            let running = Arc::clone(&running);
            device.build_input_stream(
                cfg,
                move |data: &[$ty], _| {
                    for sample in data.iter().map($convert) {
                        pending.push(sample);
                        if pending.len() == capacity {
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
                            if tx.try_send(chunk).is_err() {
                                dropped_chunks.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                },
                move |error| {
                    let message = format!("input stream error: {error}");
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

    #[test]
    fn peak_store_only_moves_upward() {
        let peak = AtomicU32::new(0.0_f32.to_bits());
        store_peak(&peak, 0.4);
        store_peak(&peak, 0.2);
        store_peak(&peak, 0.8);
        assert_eq!(f32::from_bits(peak.load(Ordering::Relaxed)), 0.8);
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

    #[test]
    fn hardware_plan_uses_effective_rate_and_checked_capacities() {
        let config = config();
        assert_eq!(
            plan_live_buffers(&config, 48_000, 2, 2).unwrap(),
            LiveBufferPlan {
                chunk_frames: 4_800,
                input_capacity: 9_600,
                queue_capacity: 76_800,
            }
        );
        assert!(plan_live_buffers(&config, crate::config::MAX_SAMPLE_RATE, 1, 1).is_ok());
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

        let preparation_error = prepare_live_config(config.clone()).unwrap_err();
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
