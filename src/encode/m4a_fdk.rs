//! Optional Fraunhofer FDK-AAC encoder with MP4/M4A muxing.

use std::io::{Seek, Write};
use std::path::Path;

use fdk_aac_rust::encoder::{
    ConfiguredPureRustEncoder, EncoderParameter, PureRustEncoderParameters,
};
use mp4::ChannelConfig;

use crate::atomic_output::{AtomicOutput, CommitMode};
use crate::audio::Audio;

use super::m4a::{sample_rate_to_index, BoundedM4aMuxer, DEFAULT_MAX_TABLE_BYTES};
use super::pcm::StreamPcmLayout;
use super::{AacEncoder, DownmixMode, EncodeOptions, OutputFormat};

/// Block-oriented FDK-AAC writer using the same bounded MP4 sample-table spool
/// as the default Pure-Rust AAC encoder.
pub(super) struct FdkM4aStreamWriter<W: Write + Seek> {
    muxer: BoundedM4aMuxer<W>,
    encoder: ConfiguredPureRustEncoder,
    layout: StreamPcmLayout,
    converted: Vec<i16>,
    pending: Vec<f32>,
    frame_samples: usize,
    frame_length: u32,
    encoder_delay: u64,
    input_frames: u64,
    encoded_media_frames: u64,
}

impl<W: Write + Seek> FdkM4aStreamWriter<W> {
    pub(super) fn new(
        output: W,
        sample_rate: u32,
        input_channels: usize,
        channel_mask: Option<crate::ChannelMask>,
        bitrate_bps: u32,
        downmix: DownmixMode,
        max_table_bytes: Option<u64>,
    ) -> Result<Self, String> {
        let layout = StreamPcmLayout::new(input_channels, channel_mask, downmix)?;
        let mut encoder =
            configured_encoder(layout.output().count as usize, sample_rate, bitrate_bps)?;
        let frame_length = u32::try_from(encoder.input_samples_per_channel())
            .map_err(|_| "FDK-AAC frame length exceeds u32".to_string())?;
        // The Pure-Rust LC backend starts its CBR reservoir empty. Encode and
        // retain one silent preroll access unit so the first real block cannot
        // exceed the nominal budget by its element/header bits. The edit list
        // trims this explicit preroll in addition to the encoder's reported
        // analysis delay.
        let encoder_delay = u64::from(encoder.encoder_delay())
            .checked_add(u64::from(frame_length))
            .ok_or_else(|| "FDK-AAC encoder delay overflows".to_string())?;
        let frame_samples = usize::try_from(frame_length)
            .ok()
            .and_then(|frames| frames.checked_mul(layout.output().count as usize))
            .ok_or_else(|| "FDK-AAC frame sample count overflows".to_string())?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(frame_samples)
            .map_err(|error| format!("reserve FDK-AAC frame: {error}"))?;
        let chan_conf = if layout.output().is_stereo {
            ChannelConfig::Stereo
        } else {
            ChannelConfig::Mono
        };
        let mut muxer = BoundedM4aMuxer::new(
            output,
            sample_rate,
            bitrate_bps,
            sample_rate_to_index(sample_rate)?,
            chan_conf,
            frame_length,
            encoder_delay,
            max_table_bytes.unwrap_or(DEFAULT_MAX_TABLE_BYTES),
        )?;
        let preroll = encoder
            .encode_interleaved_f32(&vec![0.0; frame_samples])
            .map_err(|error| format!("FDK-AAC preroll encode: {error}"))?;
        if preroll.is_empty() {
            return Err("FDK-AAC encoder produced an empty preroll access unit".into());
        }
        muxer.write_raw_access_unit(&preroll)?;
        Ok(Self {
            muxer,
            encoder,
            layout,
            converted: Vec::new(),
            pending,
            frame_samples,
            frame_length,
            encoder_delay,
            input_frames: 0,
            encoded_media_frames: u64::from(frame_length),
        })
    }

    pub(super) fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        let frames = self
            .layout
            .fill_interleaved_i16(channels, &mut self.converted)?;
        self.input_frames = self
            .input_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "FDK-AAC source frame count overflows".to_string())?;
        let mut position = 0usize;
        while position < self.converted.len() {
            let take =
                (self.frame_samples - self.pending.len()).min(self.converted.len() - position);
            self.pending.extend(
                self.converted[position..position + take]
                    .iter()
                    .map(|sample| f32::from(*sample)),
            );
            position += take;
            if self.pending.len() == self.frame_samples {
                self.encode_pending()?;
            }
        }
        Ok(())
    }

    fn encode_pending(&mut self) -> Result<(), String> {
        let encoded = self
            .encoder
            .encode_interleaved_f32(&self.pending)
            .map_err(|error| format!("FDK-AAC encode: {error}"))?;
        if encoded.is_empty() {
            return Err("FDK-AAC encoder produced an empty access unit".into());
        }
        self.muxer.write_raw_access_unit(&encoded)?;
        self.encoded_media_frames = self
            .encoded_media_frames
            .checked_add(u64::from(self.frame_length))
            .ok_or_else(|| "FDK-AAC media duration overflows".to_string())?;
        self.pending.clear();
        Ok(())
    }

    pub(super) fn finalize(mut self) -> Result<(), String> {
        if self.input_frames == 0 {
            return Err("M4A output requires at least one frame".into());
        }
        if !self.pending.is_empty() {
            self.pending.resize(self.frame_samples, 0.0);
            self.encode_pending()?;
        }
        let required_media_frames = self
            .input_frames
            .checked_add(self.encoder_delay)
            .ok_or_else(|| "FDK-AAC presentation duration overflows".to_string())?;
        while self.encoded_media_frames < required_media_frames {
            self.pending.resize(self.frame_samples, 0.0);
            self.encode_pending()?;
        }
        self.muxer.finalize(self.input_frames)
    }
}

pub(super) fn stream_timing(
    channels: usize,
    sample_rate: u32,
    bitrate_bps: u32,
) -> Result<(u32, u64), String> {
    let encoder = configured_encoder(channels, sample_rate, bitrate_bps)?;
    let frame_length = u32::try_from(encoder.input_samples_per_channel())
        .map_err(|_| "FDK-AAC frame length exceeds u32".to_string())?;
    let encoder_delay = u64::from(encoder.encoder_delay())
        .checked_add(u64::from(frame_length))
        .ok_or_else(|| "FDK-AAC encoder delay overflows".to_string())?;
    Ok((frame_length, encoder_delay))
}

fn configured_encoder(
    channels: usize,
    sample_rate: u32,
    bitrate_bps: u32,
) -> Result<ConfiguredPureRustEncoder, String> {
    let mut parameters = PureRustEncoderParameters::new(channels);
    for (parameter, value) in [
        (EncoderParameter::AudioObjectType, 2),
        (EncoderParameter::SampleRate, sample_rate),
        (EncoderParameter::Bitrate, bitrate_bps),
        (EncoderParameter::BitrateMode, 0),
        (
            EncoderParameter::ChannelMode,
            if channels == 2 { 2 } else { 1 },
        ),
        (EncoderParameter::ChannelOrder, 1),
        (EncoderParameter::Afterburner, 1),
        (EncoderParameter::TransportMux, 0),
    ] {
        parameters
            .set_parameter(parameter, value)
            .map_err(|error| format!("FDK-AAC parameter: {error}"))?;
    }
    ConfiguredPureRustEncoder::from_parameters(&parameters)
        .map_err(|error| format!("FDK-AAC encoder init: {error}"))
}

pub fn write_m4a_fdk<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    bitrate_bps: u32,
) -> Result<(), String> {
    write_m4a_fdk_with_downmix(path, audio, bitrate_bps, DownmixMode::Preserve)
}

pub fn write_m4a_fdk_with_downmix<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    bitrate_bps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    EncodeOptions {
        m4a_bitrate_bps: bitrate_bps,
        aac_encoder: AacEncoder::Fdk,
        downmix,
        ..EncodeOptions::default()
    }
    .validate_config(OutputFormat::M4a, audio)?;
    let mut output = AtomicOutput::new(path)?;
    write_m4a_fdk_to_writer(output.file_mut(), audio, bitrate_bps, downmix)?;
    output.commit(CommitMode::Replace)
}

pub(super) fn write_m4a_fdk_to_writer<W: Write + Seek>(
    output: W,
    audio: &Audio,
    bitrate_bps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    let mut writer = FdkM4aStreamWriter::new(
        output,
        audio.sample_rate,
        audio.channels(),
        audio.channel_mask,
        bitrate_bps,
        downmix,
        None,
    )?;
    writer.write_block(&audio.channels)?;
    writer.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdk_m4a_roundtrip() {
        let sample_rate = 44_100;
        let samples = (0..sample_rate / 2)
            .map(|index| {
                let time = index as f64 / sample_rate as f64;
                0.2 * (2.0 * std::f64::consts::PI * 440.0 * time).sin()
            })
            .collect();
        let audio = Audio {
            sample_rate,
            channels: vec![samples],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let path = std::env::temp_dir().join(format!("denoize-fdk-{}.m4a", std::process::id()));
        write_m4a_fdk(&path, &audio, 128_000).unwrap();
        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, sample_rate);
        assert_eq!(decoded.frames(), audio.frames());
        let leading_rms = decoded.channels[0][..1024]
            .iter()
            .map(|sample| sample * sample)
            .sum::<f64>()
            / 1024.0;
        let first_nonzero = decoded.channels[0]
            .iter()
            .position(|sample| sample.abs() > 1e-8);
        let peak = decoded.channels[0]
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(
            leading_rms > 1e-8,
            "FDK encoder priming was not trimmed: leading mean square {leading_rms:e}, first nonzero {first_nonzero:?}, peak {peak:e}"
        );
        std::fs::remove_file(path).unwrap();
    }
}
