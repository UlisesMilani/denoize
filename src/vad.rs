//! Lightweight energy VAD and speech-region segmentation.

use std::collections::VecDeque;

use crate::audio::sanitize_sample;
use crate::config::{
    checked_profile_target_samples, checked_resource_add, checked_resource_multiply, ConfigError,
    MAX_STREAM_BLOCK_FRAMES, MAX_STREAM_CHANNELS, MAX_STREAM_STATE_BYTES,
};

const VAD_WINDOW_HZ: usize = 50;
const VAD_HISTORY_SECONDS: usize = 5;
const VAD_HANGOVER_WINDOWS: usize = 10;

pub(crate) fn estimate_streaming_bytes(
    sample_rate: u32,
    channels: usize,
    block_frames: usize,
    frame_size: usize,
    profile_ms: f64,
) -> Result<u64, ConfigError> {
    if channels == 0 || channels > MAX_STREAM_CHANNELS {
        return Err(ConfigError::invalid("channels", "an integer in 1..=64"));
    }
    if !(1..=MAX_STREAM_BLOCK_FRAMES).contains(&block_frames) {
        return Err(ConfigError::invalid(
            "block_frames",
            "an integer in 1..=1048576",
        ));
    }
    let profile_frames = checked_profile_target_samples(profile_ms, sample_rate, frame_size)?;
    let window_frames = (sample_rate as u64 / VAD_WINDOW_HZ as u64).max(1);
    let backend_debt = checked_resource_add(
        "streaming VAD alignment",
        profile_frames as u64,
        checked_resource_multiply("streaming VAD alignment", frame_size as u64, 2)?,
    )?;
    let alignment_frames = checked_resource_add(
        "streaming VAD alignment",
        checked_resource_add("streaming VAD alignment", backend_debt, block_frames as u64)?,
        window_frames,
    )?;
    let channel_values = checked_resource_multiply(
        "streaming VAD alignment",
        alignment_frames,
        checked_resource_multiply("streaming VAD alignment", channels as u64, 2)?,
    )?;
    let values = checked_resource_add("streaming VAD alignment", channel_values, alignment_frames)?;
    let queue_bytes = checked_resource_multiply(
        "streaming VAD alignment",
        values,
        std::mem::size_of::<f64>() as u64,
    )?;
    let history_values = (VAD_WINDOW_HZ * VAD_HISTORY_SECONDS * 2) as u64;
    let history_bytes = checked_resource_multiply(
        "streaming VAD history",
        history_values,
        std::mem::size_of::<f64>() as u64,
    )?;
    let total = checked_resource_add("streaming VAD state", queue_bytes, history_bytes)?;
    if total > MAX_STREAM_STATE_BYTES {
        return Err(ConfigError::ResourceLimitExceeded {
            resource: "streaming VAD state",
            required_bytes: total,
            limit_bytes: MAX_STREAM_STATE_BYTES,
        });
    }
    Ok(total)
}

/// Inclusive-exclusive sample range containing speech plus context padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeechRegion {
    pub start: usize,
    pub end: usize,
}

/// Detect speech regions from planar audio using 20 ms RMS frames.
pub fn speech_regions(channels: &[Vec<f64>], sample_rate: u32) -> Vec<SpeechRegion> {
    let frames = channels.iter().map(Vec::len).max().unwrap_or(0);
    if frames == 0 || channels.is_empty() {
        return Vec::new();
    }
    let window = (sample_rate as usize / 50).max(1);
    let mut levels = Vec::with_capacity(frames.div_ceil(window));
    for start in (0..frames).step_by(window) {
        let end = (start + window).min(frames);
        let mut energy = 0.0;
        let mut count = 0usize;
        for channel in channels {
            for sample in &channel[start.min(channel.len())..end.min(channel.len())] {
                let sample = sanitize_sample(*sample);
                energy += sample * sample;
                count += 1;
            }
        }
        let rms = (energy / count.max(1) as f64).sqrt();
        levels.push(20.0 * rms.max(1e-10).log10());
    }
    let mut sorted = levels.clone();
    sorted.sort_by(f64::total_cmp);
    let floor = sorted[sorted.len() / 5];
    let peak = sorted.last().copied().unwrap_or(-200.0);
    let threshold = if peak - floor < 6.0 {
        if peak > -50.0 {
            floor - 1.0
        } else {
            -50.0
        }
    } else {
        (floor + 6.0).clamp(-55.0, -25.0)
    };
    let hangover_frames = 10; // 200 ms
    let mut active = vec![false; levels.len()];
    let mut hangover = 0usize;
    for (index, level) in levels.iter().enumerate() {
        if *level >= threshold {
            hangover = hangover_frames;
            active[index] = true;
        } else if hangover > 0 {
            active[index] = true;
            hangover -= 1;
        }
    }

    let padding = sample_rate as usize / 10; // 100 ms
    let merge_gap = sample_rate as usize * 3 / 10; // 300 ms
    let mut regions: Vec<SpeechRegion> = Vec::new();
    let mut index = 0usize;
    while index < active.len() {
        if !active[index] {
            index += 1;
            continue;
        }
        let first = index;
        while index < active.len() && active[index] {
            index += 1;
        }
        let mut region = SpeechRegion {
            start: (first * window).saturating_sub(padding),
            end: (index * window + padding).min(frames),
        };
        if let Some(previous) = regions.last_mut() {
            if region.start.saturating_sub(previous.end) <= merge_gap {
                previous.end = region.end;
                continue;
            }
        }
        region.end = region.end.max(region.start);
        regions.push(region);
    }
    regions
}

/// Bounded online VAD and alignment state for a latency-bearing stream.
///
/// Input decisions are made from 20 ms RMS windows and a bounded five-second
/// noise-floor history. Original samples, decisions, and processed samples are
/// retained only until the backend releases the corresponding presentation
/// frames. The caller remains responsible for including that backend latency
/// in its resource plan.
pub(crate) struct StreamingVad {
    channels: usize,
    window_frames: usize,
    fade_frames: usize,
    silence_gain: f64,
    speech_mix: f64,
    history_limit: usize,
    levels: VecDeque<f64>,
    level_scratch: Vec<f64>,
    pending_input: Vec<Vec<f64>>,
    pending_energy: f64,
    pending_samples: usize,
    originals: Vec<VecDeque<f64>>,
    weights: VecDeque<f64>,
    processed: Vec<VecDeque<f64>>,
    hangover: usize,
    weight: f64,
    input_frames: u64,
    processed_frames: u64,
    output_frames: u64,
    input_finished: bool,
}

impl StreamingVad {
    pub(crate) fn new(
        sample_rate: u32,
        channels: usize,
        silence_gain: f64,
        speech_mix: f64,
    ) -> Result<Self, String> {
        if sample_rate == 0 {
            return Err("streaming VAD sample rate must be positive".into());
        }
        if channels == 0 || channels > MAX_STREAM_CHANNELS {
            return Err(format!(
                "streaming VAD channels must be between 1 and {MAX_STREAM_CHANNELS}"
            ));
        }
        if !silence_gain.is_finite() || !(0.0..=1.0).contains(&silence_gain) {
            return Err("streaming VAD silence gain must be finite and in 0..=1".into());
        }
        if !speech_mix.is_finite() || !(0.0..=1.0).contains(&speech_mix) {
            return Err("streaming VAD speech mix must be finite and in 0..=1".into());
        }
        let window_frames = (sample_rate as usize / VAD_WINDOW_HZ).max(1);
        let fade_frames = window_frames;
        let history_limit = VAD_WINDOW_HZ * VAD_HISTORY_SECONDS;
        let pending_input = empty_vec_channels(channels, window_frames, "VAD input window")?;
        let originals = empty_deque_channels(channels, "VAD original alignment")?;
        let processed = empty_deque_channels(channels, "VAD processed alignment")?;
        let mut levels = VecDeque::new();
        levels
            .try_reserve_exact(history_limit)
            .map_err(|_| ConfigError::allocation_failed("VAD level history").to_string())?;
        let mut level_scratch = Vec::new();
        level_scratch
            .try_reserve_exact(history_limit)
            .map_err(|_| ConfigError::allocation_failed("VAD level scratch").to_string())?;
        Ok(Self {
            channels,
            window_frames,
            fade_frames,
            silence_gain,
            speech_mix,
            history_limit,
            levels,
            level_scratch,
            pending_input,
            pending_energy: 0.0,
            pending_samples: 0,
            originals,
            weights: VecDeque::new(),
            processed,
            hangover: 0,
            weight: 0.0,
            input_frames: 0,
            processed_frames: 0,
            output_frames: 0,
            input_finished: false,
        })
    }

    pub(crate) fn push_input(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if self.input_finished {
            return Err("streaming VAD input has already been finished".into());
        }
        let frames = validate_channels(channels, self.channels, "VAD input")?;
        self.input_frames = self
            .input_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "streaming VAD input frame count overflows".to_string())?;
        for frame in 0..frames {
            for (pending, channel) in self.pending_input.iter_mut().zip(channels) {
                let sample = sanitize_sample(channel[frame]);
                pending.push(sample);
                self.pending_energy += sample * sample;
                self.pending_samples = self
                    .pending_samples
                    .checked_add(1)
                    .ok_or_else(|| "streaming VAD sample count overflows".to_string())?;
            }
            if self.pending_input[0].len() == self.window_frames {
                self.complete_window()?;
            }
        }
        Ok(())
    }

    pub(crate) fn push_processed(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        let frames = validate_channels(channels, self.channels, "VAD processed output")?;
        for (queue, channel) in self.processed.iter_mut().zip(channels) {
            queue.try_reserve(frames).map_err(|_| {
                ConfigError::allocation_failed("VAD processed alignment").to_string()
            })?;
            queue.extend(channel.iter().copied().map(sanitize_sample));
        }
        self.processed_frames = self
            .processed_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "streaming VAD processed frame count overflows".to_string())?;
        if self.processed_frames > self.input_frames {
            return Err("streaming VAD backend produced more frames than it received".into());
        }
        Ok(())
    }

    pub(crate) fn drain_ready(&mut self) -> Result<Vec<Vec<f64>>, String> {
        let frames = self
            .weights
            .len()
            .min(self.originals[0].len())
            .min(self.processed[0].len());
        let mut output = empty_vec_channels(self.channels, frames, "VAD output block")?;
        for _ in 0..frames {
            let weight = self
                .weights
                .pop_front()
                .ok_or_else(|| "streaming VAD decision queue underflow".to_string())?;
            for channel in 0..self.channels {
                let original = self.originals[channel]
                    .pop_front()
                    .ok_or_else(|| "streaming VAD original queue underflow".to_string())?;
                let processed = self.processed[channel]
                    .pop_front()
                    .ok_or_else(|| "streaming VAD processed queue underflow".to_string())?;
                let attenuated = original * self.silence_gain;
                let speech = processed * self.speech_mix + original * (1.0 - self.speech_mix);
                output[channel].push(sanitize_sample(
                    attenuated * (1.0 - weight) + speech * weight,
                ));
            }
        }
        self.output_frames = self
            .output_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "streaming VAD output frame count overflows".to_string())?;
        Ok(output)
    }

    pub(crate) fn finish_input(&mut self) -> Result<(), String> {
        if self.input_finished {
            return Err("streaming VAD input has already been finished".into());
        }
        if !self.pending_input[0].is_empty() {
            self.complete_window()?;
        }
        self.input_finished = true;
        Ok(())
    }

    pub(crate) fn finish_output(&self) -> Result<(), String> {
        if !self.input_finished {
            return Err("streaming VAD input was not finished".into());
        }
        if self.input_frames != self.processed_frames || self.input_frames != self.output_frames {
            return Err(format!(
                "streaming VAD alignment ended at {} input, {} processed, and {} output frames",
                self.input_frames, self.processed_frames, self.output_frames
            ));
        }
        if !self.weights.is_empty()
            || self.originals.iter().any(|queue| !queue.is_empty())
            || self.processed.iter().any(|queue| !queue.is_empty())
        {
            return Err("streaming VAD alignment queues were not fully drained".into());
        }
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.levels.clear();
        self.level_scratch.clear();
        for channel in &mut self.pending_input {
            channel.clear();
        }
        for channel in &mut self.originals {
            channel.clear();
        }
        for channel in &mut self.processed {
            channel.clear();
        }
        self.weights.clear();
        self.pending_energy = 0.0;
        self.pending_samples = 0;
        self.hangover = 0;
        self.weight = 0.0;
        self.input_frames = 0;
        self.processed_frames = 0;
        self.output_frames = 0;
        self.input_finished = false;
    }

    fn complete_window(&mut self) -> Result<(), String> {
        let frames = self.pending_input[0].len();
        if frames == 0
            || self
                .pending_input
                .iter()
                .any(|channel| channel.len() != frames)
        {
            return Err("streaming VAD input window is unaligned".into());
        }
        let rms = (self.pending_energy / self.pending_samples.max(1) as f64).sqrt();
        let level = 20.0 * rms.max(1e-10).log10();
        if self.levels.len() == self.history_limit {
            self.levels.pop_front();
        }
        self.levels.push_back(level);
        self.level_scratch.clear();
        self.level_scratch.extend(self.levels.iter().copied());
        self.level_scratch.sort_by(f64::total_cmp);
        let floor = self.level_scratch[self.level_scratch.len() / 5];
        let peak = self.level_scratch.last().copied().unwrap_or(-200.0);
        let threshold = vad_threshold(floor, peak);
        let active = if level >= threshold {
            self.hangover = VAD_HANGOVER_WINDOWS;
            true
        } else if self.hangover > 0 {
            self.hangover -= 1;
            true
        } else {
            false
        };
        for original in &mut self.originals {
            original.try_reserve(frames).map_err(|_| {
                ConfigError::allocation_failed("VAD original alignment").to_string()
            })?;
        }
        self.weights
            .try_reserve(frames)
            .map_err(|_| ConfigError::allocation_failed("VAD decision alignment").to_string())?;
        let target = if active { 1.0 } else { 0.0 };
        let step = 1.0 / self.fade_frames.max(1) as f64;
        for frame in 0..frames {
            if self.weight < target {
                self.weight = (self.weight + step).min(target);
            } else if self.weight > target {
                self.weight = (self.weight - step).max(target);
            }
            self.weights.push_back(self.weight);
            for channel in 0..self.channels {
                self.originals[channel].push_back(self.pending_input[channel][frame]);
            }
        }
        for channel in &mut self.pending_input {
            channel.clear();
        }
        self.pending_energy = 0.0;
        self.pending_samples = 0;
        Ok(())
    }
}

fn vad_threshold(floor: f64, peak: f64) -> f64 {
    if peak - floor < 6.0 {
        if peak > -50.0 {
            floor - 1.0
        } else {
            -50.0
        }
    } else {
        (floor + 6.0).clamp(-55.0, -25.0)
    }
}

fn validate_channels(
    channels: &[Vec<f64>],
    expected: usize,
    context: &str,
) -> Result<usize, String> {
    if channels.len() != expected {
        return Err(format!(
            "{context} expected {expected} channels, got {}",
            channels.len()
        ));
    }
    let frames = channels.first().map(Vec::len).unwrap_or(0);
    if channels.iter().any(|channel| channel.len() != frames) {
        return Err(format!("{context} channels must have equal lengths"));
    }
    Ok(frames)
}

fn empty_vec_channels(
    channels: usize,
    frames: usize,
    resource: &'static str,
) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| ConfigError::allocation_failed(resource).to_string())?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|_| ConfigError::allocation_failed(resource).to_string())?;
        output.push(channel);
    }
    Ok(output)
}

fn empty_deque_channels(
    channels: usize,
    resource: &'static str,
) -> Result<Vec<VecDeque<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|_| ConfigError::allocation_failed(resource).to_string())?;
    for _ in 0..channels {
        output.push(VecDeque::new());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_separated_speech_and_skips_long_silence() {
        let mut audio = vec![0.0; 48_000 * 3];
        for sample in &mut audio[48_000..48_000 + 9_600] {
            *sample = 0.2;
        }
        let regions = speech_regions(&[audio], 48_000);
        assert_eq!(regions.len(), 1);
        assert!(regions[0].start < 48_000);
        assert!(regions[0].end < 48_000 * 2);
    }

    #[test]
    fn silence_has_no_regions() {
        assert!(speech_regions(&[vec![0.0; 16_000]], 16_000).is_empty());
    }

    #[test]
    fn streaming_vad_aligns_delayed_backend_output() {
        let mut vad = StreamingVad::new(1_000, 1, 0.1, 0.5).unwrap();
        let quiet = vec![vec![0.001; 40]];
        vad.push_input(&quiet).unwrap();
        vad.push_processed(&[vec![0.5; 20]]).unwrap();
        let first = vad.drain_ready().unwrap();
        assert_eq!(first[0].len(), 20);
        assert!(first[0]
            .iter()
            .all(|sample| (*sample - 0.0001).abs() < 1e-12));
        vad.push_processed(&[vec![0.5; 20]]).unwrap();
        vad.finish_input().unwrap();
        let second = vad.drain_ready().unwrap();
        assert_eq!(second[0].len(), 20);
        vad.finish_output().unwrap();
    }

    #[test]
    fn streaming_vad_fades_into_detected_speech() {
        let mut vad = StreamingVad::new(1_000, 1, 0.0, 1.0).unwrap();
        vad.push_input(&[vec![0.25; 20]]).unwrap();
        vad.push_processed(&[vec![0.5; 20]]).unwrap();
        vad.finish_input().unwrap();
        let output = vad.drain_ready().unwrap();
        assert!(output[0][0] > 0.0);
        assert!(output[0][0] < output[0][19]);
        assert!((output[0][19] - 0.5).abs() < 1e-12);
        vad.finish_output().unwrap();
    }

    #[test]
    fn streaming_vad_reset_discards_prior_alignment_state() {
        let mut vad = StreamingVad::new(1_000, 2, 0.08, 0.85).unwrap();
        vad.push_input(&[vec![0.2; 10], vec![0.2; 10]]).unwrap();
        vad.reset();
        vad.push_input(&[vec![0.2; 20], vec![0.2; 20]]).unwrap();
        vad.push_processed(&[vec![0.2; 20], vec![0.2; 20]]).unwrap();
        vad.finish_input().unwrap();
        assert_eq!(vad.drain_ready().unwrap()[0].len(), 20);
        vad.finish_output().unwrap();
    }
}
