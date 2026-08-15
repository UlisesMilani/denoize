//! Raw AAC-LC in ADTS encoding.

use super::pcm::StreamPcmLayout;
use super::{AacEncoder, DownmixMode, EncodeOptions, OutputFormat};
use crate::Audio;
use oxideav_aac_encoder::encoder::{EncoderConfig, StreamEncoder, FRAME_LEN};
use std::io::Write;
use std::path::Path;

use crate::atomic_output::{AtomicOutput, CommitMode};

/// Bounded block-oriented raw ADTS AAC encoder.
pub(super) struct AdtsAacStreamWriter<W: Write> {
    output: std::io::BufWriter<W>,
    encoder: StreamEncoder,
    layout: StreamPcmLayout,
    converted: Vec<i16>,
    pending: Vec<i16>,
    frame_samples: usize,
    input_frames: u64,
}

impl<W: Write> AdtsAacStreamWriter<W> {
    pub(super) fn new(
        output: W,
        sample_rate: u32,
        input_channels: usize,
        channel_mask: Option<crate::ChannelMask>,
        bitrate_bps: u32,
        downmix: DownmixMode,
    ) -> Result<Self, String> {
        let layout = StreamPcmLayout::new(input_channels, channel_mask, downmix)?;
        let encoder = StreamEncoder::new(EncoderConfig {
            sample_rate,
            channels: layout.output().count,
            bitrate: bitrate_bps,
        })
        .map_err(|error| format!("AAC encoder init: {error}"))?;
        let frame_samples = FRAME_LEN
            .checked_mul(layout.output().count as usize)
            .ok_or_else(|| "AAC frame sample count overflows".to_string())?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(frame_samples)
            .map_err(|error| format!("reserve AAC stream frame: {error}"))?;
        Ok(Self {
            output: std::io::BufWriter::new(output),
            encoder,
            layout,
            converted: Vec::new(),
            pending,
            frame_samples,
            input_frames: 0,
        })
    }

    pub(super) fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        let frames = self
            .layout
            .fill_interleaved_i16(channels, &mut self.converted)?;
        self.input_frames = self
            .input_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "AAC stream frame count overflows".to_string())?;
        let mut position = 0usize;
        while position < self.converted.len() {
            let take =
                (self.frame_samples - self.pending.len()).min(self.converted.len() - position);
            self.pending
                .extend_from_slice(&self.converted[position..position + take]);
            position += take;
            if self.pending.len() == self.frame_samples {
                self.write_pending_frame()?;
            }
        }
        Ok(())
    }

    fn write_pending_frame(&mut self) -> Result<(), String> {
        let frame = self
            .encoder
            .encode_frame(&self.pending)
            .map_err(|error| format!("AAC encode: {error}"))?;
        self.output
            .write_all(&frame)
            .map_err(|error| format!("write AAC: {error}"))?;
        self.pending.clear();
        Ok(())
    }

    pub(super) fn finalize(mut self) -> Result<(), String> {
        if self.input_frames == 0 {
            return Err("AAC output requires at least one frame".into());
        }
        if !self.pending.is_empty() {
            self.write_pending_frame()?;
        }
        let final_frame = self
            .encoder
            .finish()
            .map_err(|error| format!("AAC finish: {error}"))?;
        self.output
            .write_all(&final_frame)
            .map_err(|error| format!("write AAC: {error}"))?;
        self.output
            .flush()
            .map_err(|error| format!("flush AAC: {error}"))
    }
}

pub fn write_adts_aac<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    bitrate_bps: u32,
) -> Result<(), String> {
    write_adts_aac_with_downmix(path, audio, bitrate_bps, DownmixMode::Preserve)
}

pub fn write_adts_aac_with_downmix<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    bitrate_bps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    EncodeOptions {
        m4a_bitrate_bps: bitrate_bps,
        aac_encoder: AacEncoder::Oxide,
        downmix,
        ..EncodeOptions::default()
    }
    .validate_config(OutputFormat::AacAdts, audio)?;
    let mut output = AtomicOutput::new(path)?;
    write_adts_aac_to_writer(output.file_mut(), audio, bitrate_bps, downmix)?;
    output.commit(CommitMode::Replace)
}

pub(super) fn write_adts_aac_to_writer<W: Write>(
    output: W,
    audio: &Audio,
    bitrate_bps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    let mut writer = AdtsAacStreamWriter::new(
        output,
        audio.sample_rate,
        audio.channels(),
        audio.channel_mask,
        bitrate_bps,
        downmix,
    )?;
    writer.write_block(&audio.channels)?;
    writer.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adts_roundtrip() {
        let sample_rate = 44_100;
        let audio = Audio {
            sample_rate,
            channels: vec![(0..sample_rate / 2)
                .map(|index| {
                    let time = index as f64 / sample_rate as f64;
                    0.2 * (2.0 * std::f64::consts::PI * 330.0 * time).sin()
                })
                .collect()],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let path = std::env::temp_dir().join(format!("denoize-adts-{}.aac", std::process::id()));
        write_adts_aac(&path, &audio, 128_000).unwrap();
        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, sample_rate);
        assert!(decoded.frames() > 10_000);
        std::fs::remove_file(path).unwrap();
    }
}
