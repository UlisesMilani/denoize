//! Browser-safe scalar denoising compiled from the same DSP sources as the
//! native library.
//!
//! Only the thin binding and interleaving code lives in this crate. The FFT,
//! STFT, gain, noise-estimation, window, and streaming denoiser modules are
//! included directly from the repository's canonical `src/` files so scalar
//! native/WASM behavior cannot drift into two implementations.

#![allow(dead_code)]

use wasm_bindgen::prelude::*;

#[path = "../../../src/bessel.rs"]
mod bessel;
#[path = "../../../src/config.rs"]
mod config;
#[path = "../../../src/denoiser.rs"]
mod denoiser;
#[path = "../../../src/fft.rs"]
mod fft;
#[path = "../../../src/gain.rs"]
mod gain;
#[path = "../../../src/noise.rs"]
mod noise;
#[path = "../../../src/perceptual.rs"]
mod perceptual;
#[path = "../../../src/postfilter.rs"]
mod postfilter;
#[path = "../../../src/stft.rs"]
mod stft;
#[path = "../../../src/window.rs"]
mod window;

// `denoiser.rs` needs only this leaf helper from the native audio module. Keep
// the exact fail-safe PCM rule here without importing codecs, filesystems, TLS,
// or platform libraries into wasm32-unknown-unknown.
mod audio {
    #[inline]
    pub fn sanitize_sample(sample: f64) -> f64 {
        if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

use denoiser::{DenoiserConfig, StreamingDenoiser};

const MAX_CHANNELS: usize = 32;
const MAX_FRAMES_PER_CALL: usize = 1_048_576;
const MAX_BUFFERED_FRAMES: usize = 4_194_304;
const MAX_RENDER_QUANTUM: usize = 32_768;

struct ScalarProcessor {
    config: DenoiserConfig,
    stream: StreamingDenoiser,
    channels: usize,
    max_frames_per_call: usize,
    max_buffered_frames: usize,
    total_input_frames: u64,
    total_output_frames: u64,
    finished: bool,
    cancelled: bool,
}

impl ScalarProcessor {
    fn new(
        sample_rate: u32,
        channels: u32,
        strength: f32,
        frame_size: u32,
        max_frames_per_call: u32,
        max_buffered_frames: u32,
    ) -> Result<Self, String> {
        let channels = usize::try_from(channels)
            .map_err(|_| "channels do not fit this platform".to_string())?;
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return Err("channels must be in 1..=32".into());
        }
        let max_frames_per_call = usize::try_from(max_frames_per_call)
            .map_err(|_| "max_frames_per_call does not fit this platform".to_string())?;
        if !(1..=MAX_FRAMES_PER_CALL).contains(&max_frames_per_call) {
            return Err("max_frames_per_call must be in 1..=1048576".into());
        }
        let max_buffered_frames = usize::try_from(max_buffered_frames)
            .map_err(|_| "max_buffered_frames does not fit this platform".to_string())?;
        if !(max_frames_per_call..=MAX_BUFFERED_FRAMES).contains(&max_buffered_frames) {
            return Err("max_buffered_frames must cover one call and be at most 4194304".into());
        }
        let mut config = DenoiserConfig::default(sample_rate);
        config.strength = f64::from(strength);
        config.frame_size = usize::try_from(frame_size)
            .map_err(|_| "frame_size does not fit this platform".to_string())?;
        // Browser APIs never retain an implicit 1.5-second profile prefix.
        // Finite callers can prime the stream explicitly if desired.
        config.profile_ms = -1.0;
        config
            .validate_config()
            .map_err(|error| error.to_string())?;
        let stream = StreamingDenoiser::new(config.clone(), channels)?;
        Ok(Self {
            config,
            stream,
            channels,
            max_frames_per_call,
            max_buffered_frames,
            total_input_frames: 0,
            total_output_frames: 0,
            finished: false,
            cancelled: false,
        })
    }

    fn buffered_frames(&self) -> Result<usize, String> {
        let buffered = self
            .total_input_frames
            .checked_sub(self.total_output_frames)
            .ok_or_else(|| "WASM processor frame accounting underflow".to_string())?;
        usize::try_from(buffered)
            .map_err(|_| "WASM buffered frame count does not fit this platform".to_string())
    }

    fn reserve_interleaved(&self, frames: usize) -> Result<Vec<f32>, String> {
        let samples = frames
            .checked_mul(self.channels)
            .ok_or_else(|| "WASM output sample count overflows".to_string())?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(samples)
            .map_err(|_| "unable to reserve bounded WASM output".to_string())?;
        Ok(output)
    }

    fn planar_input(&self, input: &[f32], frames: usize) -> Result<Vec<Vec<f64>>, String> {
        let mut planar = Vec::new();
        planar
            .try_reserve_exact(self.channels)
            .map_err(|_| "unable to reserve WASM channel table".to_string())?;
        for _ in 0..self.channels {
            let mut channel = Vec::new();
            channel
                .try_reserve_exact(frames)
                .map_err(|_| "unable to reserve bounded WASM input channel".to_string())?;
            planar.push(channel);
        }
        for frame in 0..frames {
            for (channel, values) in planar.iter_mut().enumerate() {
                values.push(f64::from(input[frame * self.channels + channel]));
            }
        }
        Ok(planar)
    }

    fn append_interleaved(output: &mut Vec<f32>, channels: &[Vec<f64>]) -> Result<usize, String> {
        let frames = channels.first().map(Vec::len).unwrap_or(0);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("scalar core returned unequal channel lengths".into());
        }
        for frame in 0..frames {
            for channel in channels {
                output.push(channel[frame] as f32);
            }
        }
        Ok(frames)
    }

    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if self.finished {
            return Err("WASM processor has already been finished".into());
        }
        if self.cancelled {
            return Err("WASM processing was cancelled".into());
        }
        if !input.len().is_multiple_of(self.channels) {
            return Err("interleaved input length is not divisible by channels".into());
        }
        let frames = input.len() / self.channels;
        if frames > self.max_frames_per_call {
            return Err("input exceeds max_frames_per_call".into());
        }
        let buffered = self.buffered_frames()?;
        let required = buffered
            .checked_add(frames)
            .ok_or_else(|| "WASM required output frame count overflows".to_string())?;
        if required > self.max_buffered_frames {
            return Err("processing would exceed max_buffered_frames".into());
        }
        let next_total_input = self
            .total_input_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "WASM total input frame count overflows".to_string())?;
        self.total_output_frames
            .checked_add(required as u64)
            .ok_or_else(|| "WASM total output frame count can overflow".to_string())?;
        let planar = self.planar_input(input, frames)?;
        // Reserve the conservative maximum before advancing DSP state.
        let mut interleaved = self.reserve_interleaved(required)?;
        let processed = self.stream.process_block(&planar)?;
        let output_frames = Self::append_interleaved(&mut interleaved, &processed)?;
        self.total_input_frames = next_total_input;
        self.total_output_frames = self
            .total_output_frames
            .checked_add(output_frames as u64)
            .ok_or_else(|| "WASM total output frame count overflows".to_string())?;
        Ok(interleaved)
    }

    fn finish(&mut self) -> Result<Vec<f32>, String> {
        if self.finished {
            return Err("WASM processor has already been finished".into());
        }
        if self.cancelled {
            return Err("WASM processing was cancelled".into());
        }
        let required = self.buffered_frames()?;
        self.total_output_frames
            .checked_add(required as u64)
            .ok_or_else(|| "WASM total output frame count can overflow".to_string())?;
        let mut interleaved = self.reserve_interleaved(required)?;
        let processed = self.stream.finish()?;
        let output_frames = Self::append_interleaved(&mut interleaved, &processed)?;
        self.total_output_frames = self
            .total_output_frames
            .checked_add(output_frames as u64)
            .ok_or_else(|| "WASM total output frame count overflows".to_string())?;
        self.finished = true;
        if self.total_input_frames != self.total_output_frames {
            return Err("scalar core did not preserve exact finite duration".into());
        }
        Ok(interleaved)
    }

    fn reset(&mut self) -> Result<(), String> {
        let stream = StreamingDenoiser::new(self.config.clone(), self.channels)?;
        self.stream = stream;
        self.total_input_frames = 0;
        self.total_output_frames = 0;
        self.finished = false;
        self.cancelled = false;
        Ok(())
    }
}

fn js_error(error: String) -> JsValue {
    JsValue::from_str(&error)
}

/// Owned scalar processor intended for a browser Worker, never an
/// `AudioWorkletProcessor` rendering callback.
#[wasm_bindgen]
pub struct DenoizeWasmProcessor {
    inner: ScalarProcessor,
}

#[wasm_bindgen]
impl DenoizeWasmProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sample_rate: u32,
        channels: u32,
        strength: f32,
        frame_size: u32,
        max_frames_per_call: u32,
        max_buffered_frames: u32,
    ) -> Result<DenoizeWasmProcessor, JsValue> {
        ScalarProcessor::new(
            sample_rate,
            channels,
            strength,
            frame_size,
            max_frames_per_call,
            max_buffered_frames,
        )
        .map(|inner| Self { inner })
        .map_err(js_error)
    }

    pub fn process_interleaved(&mut self, input: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.inner.process(input).map_err(js_error)
    }

    pub fn finish(&mut self) -> Result<Vec<f32>, JsValue> {
        self.inner.finish().map_err(js_error)
    }

    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.inner.reset().map_err(js_error)
    }

    pub fn cancel(&mut self) {
        self.inner.cancelled = true;
    }

    pub fn buffered_frames(&self) -> Result<u32, JsValue> {
        self.inner
            .buffered_frames()
            .and_then(|frames| {
                u32::try_from(frames)
                    .map_err(|_| "WASM buffered frame count does not fit u32".to_string())
            })
            .map_err(js_error)
    }

    pub fn total_input_frames(&self) -> u64 {
        self.inner.total_input_frames
    }

    pub fn total_output_frames(&self) -> u64 {
        self.inner.total_output_frames
    }
}

/// Validate the render quantum observed from the current Web Audio output
/// buffer. The API deliberately has no constant 128-frame assumption.
#[wasm_bindgen]
pub fn validate_render_quantum(render_quantum_size: u32, channels: u32) -> Result<(), JsValue> {
    let quantum = usize::try_from(render_quantum_size)
        .map_err(|_| js_error("render quantum does not fit this platform".into()))?;
    let channels = usize::try_from(channels)
        .map_err(|_| js_error("channel count does not fit this platform".into()))?;
    if !(1..=MAX_RENDER_QUANTUM).contains(&quantum) {
        return Err(js_error(
            "render quantum must be observed in 1..=32768".into(),
        ));
    }
    if !(1..=MAX_CHANNELS).contains(&channels) {
        return Err(js_error("channels must be in 1..=32".into()));
    }
    quantum
        .checked_mul(channels)
        .ok_or_else(|| js_error("render quantum sample count overflows".into()))?;
    Ok(())
}

#[wasm_bindgen]
pub fn denoize_wasm_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[wasm_bindgen]
pub fn denoize_wasm_capabilities_json() -> String {
    include_str!("../capabilities.json").trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_processor_is_incremental_bounded_and_duration_exact() {
        let mut processor = ScalarProcessor::new(16_000, 2, 0.6, 256, 512, 2_048)
            .unwrap_or_else(|error| panic!("create scalar processor: {error}"));
        let mut source = vec![0.0f32; 2 * 1_000];
        for frame in 0..1_000 {
            source[2 * frame] = (frame as f32 * 0.01).sin() * 0.1;
            source[2 * frame + 1] = (frame as f32 * 0.013).cos() * 0.1;
        }
        let mut output = Vec::new();
        for block in source.chunks(2 * 173) {
            output.extend(
                processor
                    .process(block)
                    .unwrap_or_else(|error| panic!("process scalar block: {error}")),
            );
        }
        output.extend(
            processor
                .finish()
                .unwrap_or_else(|error| panic!("finish scalar stream: {error}")),
        );
        assert_eq!(output.len(), source.len());
        assert_eq!(processor.total_input_frames, 1_000);
        assert_eq!(processor.total_output_frames, 1_000);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn scalar_output_is_independent_of_input_block_partition() {
        let source = (0..2_113)
            .map(|frame| (frame as f32 * 0.019).sin() * 0.05)
            .collect::<Vec<_>>();
        let render = |block_frames: usize| {
            let mut processor = ScalarProcessor::new(16_000, 1, 0.4, 256, 512, 4_096)
                .unwrap_or_else(|error| panic!("create scalar processor: {error}"));
            let mut output = Vec::new();
            for block in source.chunks(block_frames) {
                output.extend(
                    processor
                        .process(block)
                        .unwrap_or_else(|error| panic!("process scalar block: {error}")),
                );
            }
            output.extend(
                processor
                    .finish()
                    .unwrap_or_else(|error| panic!("finish scalar stream: {error}")),
            );
            output
        };
        assert_eq!(render(113), render(251));
    }

    #[test]
    fn capability_contract_has_no_silent_backend_or_quantum_fallback() {
        let capability: serde_json::Value = serde_json::from_str(&denoize_wasm_capabilities_json())
            .unwrap_or_else(|error| panic!("parse capability JSON: {error}"));
        assert_eq!(capability["backend"], "classical-scalar");
        assert_eq!(
            capability["default_render_quantum"],
            serde_json::Value::Null
        );
        assert_eq!(capability["observed_render_quantum_required"], true);
        assert_eq!(capability["implicit_model_downloads"], false);
    }

    #[test]
    fn cancellation_and_limits_fail_without_advancing_state() {
        let mut processor = ScalarProcessor::new(48_000, 1, 0.6, 256, 32, 128)
            .unwrap_or_else(|error| panic!("create scalar processor: {error}"));
        assert!(processor.process(&[0.0; 33]).is_err());
        assert_eq!(processor.total_input_frames, 0);
        processor.cancelled = true;
        assert!(processor.process(&[0.0; 1]).is_err());
        assert_eq!(processor.total_input_frames, 0);
        processor
            .reset()
            .unwrap_or_else(|error| panic!("reset scalar processor: {error}"));
        assert!(!processor.cancelled);
    }
}
