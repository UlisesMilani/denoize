//! DeepFilterNet v3 backend via the official `deep_filter` crate (tract ONNX).
//!
//! Requires `--features deepfilter` at build time. Uses the embedded DFN3 model.

use df::tract::{DfParams, DfTract, RuntimeParams};
use df::transforms::resample;
use ndarray::{Array2, Axis};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;

/// Target sample rate for DeepFilterNet (48 kHz).
const DF_SR: usize = 48_000;
const DF_DEFAULT_HOP_SIZE: u64 = 480;
const STREAM_MODEL_ALLOWANCE_BYTES: u64 = 128 * 1024 * 1024;
const STREAM_WORKER_STACK_BYTES: u64 = 2 * 1024 * 1024;

/// Denoise channels using DeepFilterNet v3.
pub fn process(channels: &[Vec<f64>], sample_rate: u32) -> Result<Vec<Vec<f64>>, String> {
    DeepFilterModel::load()?.process(channels, sample_rate)
}

pub(crate) struct DeepFilterModel {
    id: u64,
}

struct ThreadModels {
    session_id: u64,
    templates: HashMap<usize, DfTract>,
}

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static THREAD_MODELS: RefCell<Option<ThreadModels>> = const { RefCell::new(None) };
}

impl DeepFilterModel {
    pub(crate) fn load() -> Result<Self, String> {
        let id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        let model = new_model(1)?;
        THREAD_MODELS.with(|cache| {
            let mut templates = HashMap::new();
            templates.insert(1, model);
            *cache.borrow_mut() = Some(ThreadModels {
                session_id: id,
                templates,
            });
        });
        Ok(Self { id })
    }

    pub(crate) fn process(
        &self,
        channels: &[Vec<f64>],
        sample_rate: u32,
    ) -> Result<Vec<Vec<f64>>, String> {
        let n_ch = channels.len().max(1);
        let max_len = channels.iter().map(|c| c.len()).max().unwrap_or(0);
        if max_len == 0 {
            return Ok(channels.to_vec());
        }

        // Build f32 array [channels, samples] at 48 kHz.
        let mut ch_data: Vec<Vec<f32>> = Vec::with_capacity(n_ch);
        for ch in channels {
            let f32_in: Vec<f32> = ch.iter().map(|&x| x as f32).collect();
            let at_48k = if sample_rate as usize == DF_SR {
                f32_in
            } else {
                resample_to_48k(&f32_in, sample_rate as usize)?
            };
            ch_data.push(at_48k);
        }

        let mut model = THREAD_MODELS.with(|cache| -> Result<DfTract, String> {
            let mut cache = cache.borrow_mut();
            if cache
                .as_ref()
                .is_none_or(|models| models.session_id != self.id)
            {
                *cache = Some(ThreadModels {
                    session_id: self.id,
                    templates: HashMap::new(),
                });
            }
            let models = cache
                .as_mut()
                .expect("DeepFilterNet thread cache was initialized");
            if !models.templates.contains_key(&n_ch) {
                models.templates.insert(n_ch, new_model(n_ch)?);
            }
            Ok(models
                .templates
                .get(&n_ch)
                .expect("DeepFilterNet channel template was inserted")
                .clone())
        })?;

        // Flush and remove the STFT/model lookahead latency. Processing only the
        // source frames leaves this delay at the beginning and truncates the same
        // number of samples from the end.
        let source_len_48k = ch_data.iter().map(|c| c.len()).max().unwrap_or(0);
        let stft_delay = model
            .fft_size
            .checked_sub(model.hop_size)
            .ok_or_else(|| "DeepFilterNet reported an invalid FFT size".to_string())?;
        let model_delay = model
            .lookahead
            .checked_mul(model.hop_size)
            .ok_or_else(|| "DeepFilterNet latency overflow".to_string())?;
        let delay_48k = stft_delay
            .checked_add(model_delay)
            .ok_or_else(|| "DeepFilterNet latency overflow".to_string())?;
        let flush_len_48k = source_len_48k
            .checked_add(delay_48k)
            .ok_or_else(|| "DeepFilterNet input is too long".to_string())?;
        let len_48k = padded_hop_len(flush_len_48k, model.hop_size);
        for c in &mut ch_data {
            c.resize(len_48k, 0.0);
        }

        let noisy = Array2::from_shape_fn((n_ch, len_48k), |(ch, i)| ch_data[ch][i]);
        let mut enh = Array2::zeros((n_ch, len_48k));

        for (ns_chunk, enh_chunk) in noisy
            .view()
            .axis_chunks_iter(Axis(1), model.hop_size)
            .zip(enh.view_mut().axis_chunks_iter_mut(Axis(1), model.hop_size))
        {
            debug_assert_eq!(ns_chunk.len_of(Axis(1)), model.hop_size);
            model
                .process(ns_chunk, enh_chunk)
                .map_err(|e| format!("DeepFilterNet process failed: {e}"))?;
        }

        // Extract per-channel output and resample back.
        let mut result = Vec::with_capacity(n_ch);
        for ch in 0..n_ch {
            let row: Vec<f32> = enh
                .row(ch)
                .iter()
                .skip(delay_48k)
                .take(source_len_48k)
                .copied()
                .collect();
            let f64_out: Vec<f64> = if sample_rate as usize == DF_SR {
                row.iter().map(|&x| x as f64).collect()
            } else {
                resample_from_48k(&row, sample_rate as usize)?
                    .iter()
                    .map(|&x| x as f64)
                    .collect()
            };
            let orig_len = channels.get(ch).map(|c| c.len()).unwrap_or(len_48k);
            let mut trimmed = f64_out;
            trimmed.truncate(orig_len);
            if trimmed.len() < orig_len {
                trimmed.resize(orig_len, 0.0);
            }
            result.push(trimmed);
        }
        Ok(result)
    }
}

/// Continuous, bounded DeepFilterNet processing at an arbitrary input rate.
///
/// The embedded graph is loaded once, while only one incomplete model hop,
/// STFT/model lookahead, and two sample-rate-converter clocks are retained.
/// Output can therefore be empty until the fixed model latency has elapsed.
pub(crate) struct StreamingProcessor {
    channels: usize,
    to_model_rate: crate::resample::StreamingResampler,
    from_model_rate: crate::resample::StreamingResampler,
    model: DeepFilterStreamWorker,
    pending_model_rate: Vec<Vec<f64>>,
    hop_input: Array2<f32>,
    hop_output: Array2<f32>,
    hop_size: usize,
    delay_model_frames: usize,
    discard_model_frames: usize,
    model_source_frames: usize,
    model_processed_frames: usize,
    model_output_frames: usize,
    input_frames: usize,
    output_frames: usize,
    finished: bool,
}

#[derive(Clone, Copy)]
struct DeepFilterStreamGeometry {
    sample_rate: usize,
    hop_size: usize,
    fft_size: usize,
    lookahead: usize,
}

enum DeepFilterStreamCommand {
    Process {
        input: Array2<f32>,
        output: Array2<f32>,
    },
    Reset,
    Shutdown,
}

enum DeepFilterStreamResponse {
    Processed {
        input: Array2<f32>,
        output: Array2<f32>,
        error: Option<String>,
    },
    Reset,
}

struct DeepFilterStreamWorker {
    commands: SyncSender<DeepFilterStreamCommand>,
    responses: Receiver<DeepFilterStreamResponse>,
    thread: Option<JoinHandle<()>>,
}

impl DeepFilterStreamWorker {
    fn new(channels: usize) -> Result<(Self, DeepFilterStreamGeometry), String> {
        let (commands, command_rx) = mpsc::sync_channel(0);
        let (response_tx, responses) = mpsc::sync_channel(0);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let stack_size = usize::try_from(STREAM_WORKER_STACK_BYTES)
            .map_err(|_| "DeepFilterNet worker stack size does not fit this platform")?;
        let thread = std::thread::Builder::new()
            .name("denoize-deepfilter-stream".into())
            .stack_size(stack_size)
            .spawn(move || {
                let template = match new_model(channels) {
                    Ok(model) => model,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                let geometry = DeepFilterStreamGeometry {
                    sample_rate: template.sr,
                    hop_size: template.hop_size,
                    fft_size: template.fft_size,
                    lookahead: template.lookahead,
                };
                let mut model = template.clone();
                if ready_tx.send(Ok(geometry)).is_err() {
                    return;
                }
                while let Ok(command) = command_rx.recv() {
                    match command {
                        DeepFilterStreamCommand::Process { input, mut output } => {
                            output.fill(0.0);
                            let error = model
                                .process(input.view(), output.view_mut())
                                .err()
                                .map(|error| format!("DeepFilterNet process failed: {error}"));
                            if response_tx
                                .send(DeepFilterStreamResponse::Processed {
                                    input,
                                    output,
                                    error,
                                })
                                .is_err()
                            {
                                return;
                            }
                        }
                        DeepFilterStreamCommand::Reset => {
                            model = template.clone();
                            if response_tx.send(DeepFilterStreamResponse::Reset).is_err() {
                                return;
                            }
                        }
                        DeepFilterStreamCommand::Shutdown => return,
                    }
                }
            })
            .map_err(|error| format!("start DeepFilterNet stream worker: {error}"))?;
        let geometry = match ready_rx.recv() {
            Ok(Ok(geometry)) => geometry,
            Ok(Err(error)) => {
                let _ = thread.join();
                return Err(error);
            }
            Err(_) => {
                let _ = thread.join();
                return Err("DeepFilterNet stream worker exited during initialization".into());
            }
        };
        Ok((
            Self {
                commands,
                responses,
                thread: Some(thread),
            },
            geometry,
        ))
    }

    fn process(
        &mut self,
        input: Array2<f32>,
        output: Array2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>, Option<String>), String> {
        if let Err(error) = self
            .commands
            .send(DeepFilterStreamCommand::Process { input, output })
        {
            if let DeepFilterStreamCommand::Process { input, output } = error.0 {
                return Ok((
                    input,
                    output,
                    Some("DeepFilterNet stream worker is unavailable".into()),
                ));
            }
            unreachable!("failed DeepFilterNet process send returned another command");
        }
        match self.responses.recv() {
            Ok(DeepFilterStreamResponse::Processed {
                input,
                output,
                error,
            }) => Ok((input, output, error)),
            Ok(DeepFilterStreamResponse::Reset) => {
                Err("DeepFilterNet stream worker returned an unexpected reset response".into())
            }
            Err(_) => Err("DeepFilterNet stream worker exited during inference".into()),
        }
    }

    fn reset(&mut self) -> Result<(), String> {
        self.commands
            .send(DeepFilterStreamCommand::Reset)
            .map_err(|_| "DeepFilterNet stream worker is unavailable".to_string())?;
        match self.responses.recv() {
            Ok(DeepFilterStreamResponse::Reset) => Ok(()),
            Ok(DeepFilterStreamResponse::Processed { .. }) => {
                Err("DeepFilterNet stream worker returned an unexpected process response".into())
            }
            Err(_) => Err("DeepFilterNet stream worker exited during reset".into()),
        }
    }
}

impl Drop for DeepFilterStreamWorker {
    fn drop(&mut self) {
        let _ = self.commands.send(DeepFilterStreamCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl StreamingProcessor {
    pub(crate) fn new(sample_rate: u32, channels: usize) -> Result<Self, String> {
        if channels == 0 || channels > crate::config::MAX_STREAM_CHANNELS {
            return Err(format!(
                "DeepFilterNet streaming channels must be between 1 and {}",
                crate::config::MAX_STREAM_CHANNELS
            ));
        }
        let (model, geometry) = DeepFilterStreamWorker::new(channels)?;
        if geometry.sample_rate != DF_SR {
            return Err(format!(
                "embedded DeepFilterNet model expects {} Hz, not {DF_SR} Hz",
                geometry.sample_rate
            ));
        }
        let hop_size = geometry.hop_size;
        if hop_size == 0 {
            return Err("DeepFilterNet reported a zero hop size".into());
        }
        let stft_delay = geometry
            .fft_size
            .checked_sub(hop_size)
            .ok_or_else(|| "DeepFilterNet reported an invalid FFT size".to_string())?;
        let model_delay = geometry
            .lookahead
            .checked_mul(hop_size)
            .ok_or_else(|| "DeepFilterNet latency overflow".to_string())?;
        let delay_model_frames = stft_delay
            .checked_add(model_delay)
            .ok_or_else(|| "DeepFilterNet latency overflow".to_string())?;
        let to_model_rate =
            crate::resample::StreamingResampler::new(channels, sample_rate, DF_SR as u32)?;
        let from_model_rate =
            crate::resample::StreamingResampler::new(channels, DF_SR as u32, sample_rate)?;
        let mut pending_model_rate = Vec::new();
        pending_model_rate
            .try_reserve_exact(channels)
            .map_err(|_| "unable to reserve DeepFilterNet pending channels".to_string())?;
        for _ in 0..channels {
            let mut pending = Vec::new();
            pending
                .try_reserve_exact(hop_size)
                .map_err(|_| "unable to reserve DeepFilterNet pending samples".to_string())?;
            pending_model_rate.push(pending);
        }
        Ok(Self {
            channels,
            to_model_rate,
            from_model_rate,
            model,
            pending_model_rate,
            hop_input: Array2::zeros((channels, hop_size)),
            hop_output: Array2::zeros((channels, hop_size)),
            hop_size,
            delay_model_frames,
            discard_model_frames: delay_model_frames,
            model_source_frames: 0,
            model_processed_frames: 0,
            model_output_frames: 0,
            input_frames: 0,
            output_frames: 0,
            finished: false,
        })
    }

    pub(crate) fn process_block(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err(
                "DeepFilterNet stream is finished; reset it before processing more input".into(),
            );
        }
        let frames = validate_stream_block(input, self.channels)?;
        let input_frames = self
            .input_frames
            .checked_add(frames)
            .ok_or_else(|| "DeepFilterNet streaming input length overflow".to_string())?;
        let at_model_rate = self.to_model_rate.process(input)?;
        let enhanced_model_rate = self.process_model_rate(&at_model_rate)?;
        let output = self.from_model_rate.process(&enhanced_model_rate)?;
        let produced = validate_stream_block(&output, self.channels)?;
        let output_frames = self
            .output_frames
            .checked_add(produced)
            .ok_or_else(|| "DeepFilterNet streaming output length overflow".to_string())?;
        if output_frames > input_frames {
            return Err("DeepFilterNet stream produced samples ahead of its input clock".into());
        }
        self.input_frames = input_frames;
        self.output_frames = output_frames;
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .input_frames
            .checked_sub(self.output_frames)
            .ok_or_else(|| "DeepFilterNet stream exceeded its input clock".to_string())?;
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

    pub(crate) fn reset(&mut self) -> Result<(), String> {
        self.to_model_rate.reset();
        self.from_model_rate.reset();
        self.model.reset()?;
        for pending in &mut self.pending_model_rate {
            pending.clear();
        }
        self.hop_input.fill(0.0);
        self.hop_output.fill(0.0);
        self.discard_model_frames = self.delay_model_frames;
        self.model_source_frames = 0;
        self.model_processed_frames = 0;
        self.model_output_frames = 0;
        self.input_frames = 0;
        self.output_frames = 0;
        self.finished = false;
        Ok(())
    }

    fn process_model_rate(&mut self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let frames = validate_stream_block(input, self.channels)?;
        self.model_source_frames = self
            .model_source_frames
            .checked_add(frames)
            .ok_or_else(|| "DeepFilterNet model input length overflow".to_string())?;
        let mut output = empty_stream_output(self.channels, frames)?;
        if frames == 0 {
            return Ok(output);
        }

        let mut position = 0usize;
        let pending_frames = self.pending_model_rate.first().map_or(0, Vec::len);
        if pending_frames > 0 {
            let copied = (self.hop_size - pending_frames).min(frames);
            for (pending, source) in self.pending_model_rate.iter_mut().zip(input) {
                pending.extend(source[..copied].iter().copied().map(crate::sanitize_sample));
            }
            position = copied;
            if pending_frames + copied == self.hop_size {
                for channel in 0..self.channels {
                    for frame in 0..self.hop_size {
                        self.hop_input[(channel, frame)] =
                            self.pending_model_rate[channel][frame] as f32;
                    }
                    self.pending_model_rate[channel].clear();
                }
                self.run_model_hop(&mut output)?;
            }
        }

        while frames - position >= self.hop_size {
            for channel in 0..self.channels {
                for frame in 0..self.hop_size {
                    self.hop_input[(channel, frame)] =
                        crate::sanitize_sample(input[channel][position + frame]) as f32;
                }
            }
            self.run_model_hop(&mut output)?;
            position += self.hop_size;
        }

        if position < frames {
            for (pending, source) in self.pending_model_rate.iter_mut().zip(input) {
                pending.extend(
                    source[position..]
                        .iter()
                        .copied()
                        .map(crate::sanitize_sample),
                );
            }
        }
        Ok(output)
    }

    fn finish_model_rate(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let remaining = self
            .model_source_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "DeepFilterNet model output exceeded its input clock".to_string())?;
        let mut output = empty_stream_output(self.channels, remaining)?;
        if self.model_source_frames == 0 {
            return Ok(output);
        }
        let flush_frames = self
            .model_source_frames
            .checked_add(self.delay_model_frames)
            .ok_or_else(|| "DeepFilterNet flush length overflow".to_string())?;
        let process_target = padded_hop_len(flush_frames, self.hop_size);

        if self
            .pending_model_rate
            .first()
            .is_some_and(|channel| !channel.is_empty())
        {
            self.hop_input.fill(0.0);
            for channel in 0..self.channels {
                for (frame, sample) in self.pending_model_rate[channel].iter().enumerate() {
                    self.hop_input[(channel, frame)] = *sample as f32;
                }
                self.pending_model_rate[channel].clear();
            }
            self.run_model_hop(&mut output)?;
        }
        while self.model_processed_frames < process_target {
            self.hop_input.fill(0.0);
            self.run_model_hop(&mut output)?;
        }
        if self.model_processed_frames != process_target
            || self.model_output_frames != self.model_source_frames
        {
            return Err("DeepFilterNet stream did not flush to its model clock".into());
        }
        Ok(output)
    }

    fn run_model_hop(&mut self, output: &mut [Vec<f64>]) -> Result<(), String> {
        if self.hop_input.len() != self.channels * self.hop_size
            || self.hop_output.len() != self.channels * self.hop_size
        {
            return Err("DeepFilterNet stream hop buffers are unavailable".into());
        }
        let hop_input = std::mem::take(&mut self.hop_input);
        let hop_output = std::mem::take(&mut self.hop_output);
        let (hop_input, hop_output, error) = self.model.process(hop_input, hop_output)?;
        self.hop_input = hop_input;
        self.hop_output = hop_output;
        if let Some(error) = error {
            return Err(error);
        }
        self.model_processed_frames = self
            .model_processed_frames
            .checked_add(self.hop_size)
            .ok_or_else(|| "DeepFilterNet processed frame count overflow".to_string())?;
        let skipped = self.discard_model_frames.min(self.hop_size);
        self.discard_model_frames -= skipped;
        let available = self.hop_size - skipped;
        let remaining = self
            .model_source_frames
            .checked_sub(self.model_output_frames)
            .ok_or_else(|| "DeepFilterNet model output exceeded its input clock".to_string())?;
        let take = available.min(remaining);
        for (channel, destination) in output.iter_mut().enumerate() {
            destination.extend(
                self.hop_output
                    .row(channel)
                    .iter()
                    .skip(skipped)
                    .take(take)
                    .map(|sample| crate::sanitize_sample(*sample as f64)),
            );
        }
        self.model_output_frames = self
            .model_output_frames
            .checked_add(take)
            .ok_or_else(|| "DeepFilterNet model output length overflow".to_string())?;
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
    let samples = checked_resource_multiply(
        "DeepFilterNet stream buffers",
        processor_channels as u64,
        DF_DEFAULT_HOP_SIZE,
    )?;
    let f64_bytes = checked_resource_multiply(
        "DeepFilterNet stream buffers",
        samples,
        2 * std::mem::size_of::<f64>() as u64,
    )?;
    let f32_bytes = checked_resource_multiply(
        "DeepFilterNet stream buffers",
        samples,
        2 * std::mem::size_of::<f32>() as u64,
    )?;
    let model_state = checked_resource_add(
        "DeepFilterNet stream state",
        checked_resource_add(
            "DeepFilterNet stream state",
            STREAM_MODEL_ALLOWANCE_BYTES,
            STREAM_WORKER_STACK_BYTES,
        )?,
        checked_resource_add("DeepFilterNet stream buffers", f64_bytes, f32_bytes)?,
    )?;
    // One input-rate second is well above the embedded graph's fixed STFT,
    // lookahead, and resampler latency. Reserve three simultaneous copies for
    // linked-stereo restoration and optional VAD original/processed alignment.
    let alignment_samples = checked_resource_multiply(
        "DeepFilterNet stream alignment",
        u64::from(input_sample_rate),
        input_channels as u64,
    )?;
    let alignment_bytes = checked_resource_multiply(
        "DeepFilterNet stream alignment",
        alignment_samples,
        3 * std::mem::size_of::<f64>() as u64,
    )?;
    checked_resource_add("DeepFilterNet stream state", model_state, alignment_bytes)
}

fn validate_stream_block(input: &[Vec<f64>], channels: usize) -> Result<usize, String> {
    if input.len() != channels {
        return Err(format!(
            "DeepFilterNet stream expected {channels} channels, received {}",
            input.len()
        ));
    }
    let frames = input.first().map_or(0, Vec::len);
    if input.iter().any(|channel| channel.len() != frames) {
        return Err("DeepFilterNet stream channels must contain the same number of frames".into());
    }
    Ok(frames)
}

fn empty_stream_output(channels: usize, capacity: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| "unable to reserve DeepFilterNet output channels".to_string())?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(capacity)
            .map_err(|_| "unable to reserve DeepFilterNet output samples".to_string())?;
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
        return Err("DeepFilterNet stream output channel count changed".into());
    }
    let source_frames = source.first().map_or(0, Vec::len);
    if source.iter().any(|channel| channel.len() != source_frames) {
        return Err("DeepFilterNet stream output channels became unaligned".into());
    }
    let retained = limit.saturating_sub(destination.first().map_or(0, Vec::len));
    let take = retained.min(source_frames);
    for (output, input) in destination.iter_mut().zip(source) {
        output.extend(input.iter().take(take).copied().map(crate::sanitize_sample));
    }
    Ok(())
}

fn new_model(channels: usize) -> Result<DfTract, String> {
    let r_params = RuntimeParams::default_with_ch(channels)
        .with_atten_lim(100.0)
        .with_thresholds(-15.0, 35.0, 35.0);
    DfTract::new(DfParams::default(), &r_params)
        .map_err(|error| format!("DeepFilterNet init failed: {error}"))
}

fn padded_hop_len(input_len: usize, hop_size: usize) -> usize {
    input_len.div_ceil(hop_size) * hop_size
}

fn resample_to_48k(input: &[f32], from_sr: usize) -> Result<Vec<f32>, String> {
    if from_sr == DF_SR {
        return Ok(input.to_vec());
    }
    let arr = ndarray::Array2::from_shape_fn((1, input.len()), |(_, i)| input[i]);
    let out = resample(arr.view(), from_sr, DF_SR, None)
        .map_err(|e| format!("resample to 48k failed: {e}"))?;
    Ok(out.row(0).iter().copied().collect())
}

fn resample_from_48k(input: &[f32], to_sr: usize) -> Result<Vec<f32>, String> {
    if to_sr == DF_SR {
        return Ok(input.to_vec());
    }
    let arr = ndarray::Array2::from_shape_fn((1, input.len()), |(_, i)| input[i]);
    let out = resample(arr.view(), DF_SR, to_sr, None)
        .map_err(|e| format!("resample from 48k failed: {e}"))?;
    Ok(out.row(0).iter().copied().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn correlation(left: &[f64], right: &[f64]) -> f64 {
        let dot: f64 = left.iter().zip(right).map(|(a, b)| a * b).sum();
        let left_energy: f64 = left.iter().map(|sample| sample * sample).sum();
        let right_energy: f64 = right.iter().map(|sample| sample * sample).sum();
        dot / (left_energy * right_energy).sqrt()
    }

    #[test]
    fn final_partial_hop_is_padded() {
        assert_eq!(padded_hop_len(1, 480), 480);
        assert_eq!(padded_hop_len(480, 480), 480);
        assert_eq!(padded_hop_len(481, 480), 960);
    }

    #[test]
    fn embedded_model_runs_end_to_end() {
        let sample_count = DF_SR / 4;
        let noisy: Vec<f64> = (0..sample_count)
            .map(|index| {
                let time = index as f64 / DF_SR as f64;
                let voiced = 0.18 * (std::f64::consts::TAU * 180.0 * time).sin()
                    + 0.08 * (std::f64::consts::TAU * 360.0 * time).sin();
                let noise = (((index * 37) % 101) as f64 / 50.0 - 1.0) * 0.03;
                voiced + noise
            })
            .collect();
        let input_energy: f64 = noisy.iter().map(|sample| sample * sample).sum();

        let enhanced = process(&[noisy], DF_SR as u32).expect("embedded model inference failed");

        assert_eq!(enhanced.len(), 1);
        assert_eq!(enhanced[0].len(), sample_count);
        assert!(enhanced[0].iter().all(|sample| sample.is_finite()));
        let output_energy: f64 = enhanced[0].iter().map(|sample| sample * sample).sum();
        assert!(output_energy > 1e-6, "embedded model produced silence");
        assert!(
            output_energy < input_energy * 4.0,
            "embedded model produced unbounded output"
        );
    }

    #[test]
    fn embedded_model_handles_resampled_stereo() {
        const INPUT_RATE: u32 = 16_000;
        let sample_count = INPUT_RATE as usize / 4;
        let channel = |frequency: f64, phase: f64| {
            (0..sample_count)
                .map(|index| {
                    let time = index as f64 / INPUT_RATE as f64;
                    0.2 * (std::f64::consts::TAU * frequency * time + phase).sin()
                })
                .collect::<Vec<_>>()
        };
        let input = [channel(180.0, 0.0), channel(240.0, 0.4)];

        let enhanced = process(&input, INPUT_RATE).expect("resampled stereo inference failed");

        assert_eq!(enhanced.len(), input.len());
        for (output, source) in enhanced.iter().zip(input.iter()) {
            assert_eq!(output.len(), source.len());
            assert!(output.iter().all(|sample| sample.is_finite()));
            let output_energy: f64 = output.iter().map(|sample| sample * sample).sum();
            let input_energy: f64 = source.iter().map(|sample| sample * sample).sum();
            assert!(output_energy > 1e-6, "resampled output was silent");
            assert!(
                output_energy < input_energy * 4.0,
                "resampled output was unbounded"
            );
            let tail_energy: f64 = output
                .iter()
                .rev()
                .take(INPUT_RATE as usize / 100)
                .map(|sample| sample * sample)
                .sum();
            assert!(tail_energy > 1e-6, "resampled output tail was truncated");
        }
        let left_own = correlation(&enhanced[0], &input[0]).abs();
        let left_other = correlation(&enhanced[0], &input[1]).abs();
        let right_own = correlation(&enhanced[1], &input[1]).abs();
        let right_other = correlation(&enhanced[1], &input[0]).abs();
        assert!(left_own > left_other + 0.2, "left channel was mixed up");
        assert!(right_own > right_other + 0.2, "right channel was mixed up");
    }

    #[test]
    fn streaming_matches_offline_across_irregular_blocks_and_reset() {
        let sample_count = DF_SR / 5 + 173;
        let noisy: Vec<f64> = (0..sample_count)
            .map(|index| {
                let time = index as f64 / DF_SR as f64;
                0.16 * (std::f64::consts::TAU * 210.0 * time).sin()
                    + (((index * 29) % 97) as f64 / 48.0 - 1.0) * 0.025
            })
            .collect();
        let expected = process(&[noisy.clone()], DF_SR as u32).unwrap();

        let run = |processor: &mut StreamingProcessor| {
            let mut output = Vec::new();
            let mut position = 0usize;
            for block_frames in [113, 997, 17, 2_049, 480, 731] {
                if position == noisy.len() {
                    break;
                }
                let end = position.saturating_add(block_frames).min(noisy.len());
                let ready = processor
                    .process_block(&[noisy[position..end].to_vec()])
                    .unwrap();
                output.extend_from_slice(&ready[0]);
                position = end;
            }
            if position < noisy.len() {
                let ready = processor
                    .process_block(&[noisy[position..].to_vec()])
                    .unwrap();
                output.extend_from_slice(&ready[0]);
            }
            output.extend_from_slice(&processor.finish().unwrap()[0]);
            output
        };

        let mut processor = StreamingProcessor::new(DF_SR as u32, 1).unwrap();
        let first = run(&mut processor);
        assert_eq!(first.len(), noisy.len());
        assert_eq!(first, expected[0]);

        processor.reset().unwrap();
        let second = run(&mut processor);
        assert_eq!(second, first);
    }

    #[test]
    fn streaming_processor_remains_send_with_thread_affine_tract_state() {
        fn assert_send<T: Send>() {}
        assert_send::<StreamingProcessor>();
    }

    #[test]
    fn streaming_resampling_preserves_exact_input_clock() {
        const INPUT_RATE: u32 = 16_000;
        let noisy: Vec<f64> = (0..(INPUT_RATE as usize / 2 + 37))
            .map(|index| {
                let time = index as f64 / INPUT_RATE as f64;
                0.18 * (std::f64::consts::TAU * 240.0 * time).sin()
            })
            .collect();
        let mut processor = StreamingProcessor::new(INPUT_RATE, 1).unwrap();
        let mut output = Vec::new();
        for block in noisy.chunks(127) {
            output.extend_from_slice(&processor.process_block(&[block.to_vec()]).unwrap()[0]);
        }
        output.extend_from_slice(&processor.finish().unwrap()[0]);

        assert_eq!(output.len(), noisy.len());
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().map(|sample| sample * sample).sum::<f64>() > 1e-6);
    }

    #[test]
    fn streaming_state_estimate_includes_bounded_model_and_channel_buffers() {
        let mono = streaming_state_bytes(1, 48_000, 1).unwrap();
        let stereo = streaming_state_bytes(2, 48_000, 2).unwrap();
        assert!(mono >= STREAM_MODEL_ALLOWANCE_BYTES);
        assert!(stereo > mono);
        assert!(streaming_state_bytes(0, 48_000, 1).is_err());
    }
}
