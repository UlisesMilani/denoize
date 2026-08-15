//! MP3 encode via `shine-rs` (Pure Rust, LGPL-2.0).

use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::path::Path;

use shine_rs::{Mp3Encoder, Mp3EncoderConfig, StereoMode, SUPPORTED_BITRATES};

use crate::atomic_output::{AtomicOutput, CommitMode};
use crate::audio::Audio;

use super::pcm::{lossy_channel_layout, EncodeChannels, StreamPcmLayout};
use super::{DownmixMode, EncodeOptions, OutputFormat};

/// Default MP3 bitrate (kbps).
pub const DEFAULT_MP3_BITRATE: u32 = 192;

/// Bounded block-oriented MP3 encoder.
///
/// `shine-rs` retains at most one MPEG frame internally. Encoded frames are
/// written as soon as they are returned instead of accumulating the complete
/// MP3 in memory.
pub(super) struct Mp3StreamWriter<W: Write> {
    output: BufWriter<W>,
    encoder: Mp3Encoder,
    layout: StreamPcmLayout,
    pcm: Vec<i16>,
    pending: VecDeque<i16>,
    frame: Vec<i16>,
    input_samples: usize,
    encoded_frames: usize,
    finished: bool,
}

impl<W: Write> Mp3StreamWriter<W> {
    pub(super) fn new(
        output: W,
        sample_rate: u32,
        input_channels: usize,
        channel_mask: Option<crate::ChannelMask>,
        bitrate_kbps: u32,
        downmix: DownmixMode,
    ) -> Result<Self, String> {
        let layout = StreamPcmLayout::new(input_channels, channel_mask, downmix)?;
        let config = effective_mp3_stream_config(sample_rate, layout.output(), bitrate_kbps)?;
        let encoder = Mp3Encoder::new(config).map_err(|error| format!("mp3 encoder: {error}"))?;
        Ok(Self {
            output: BufWriter::new(output),
            encoder,
            layout,
            pcm: Vec::new(),
            pending: VecDeque::new(),
            frame: Vec::new(),
            input_samples: 0,
            encoded_frames: 0,
            finished: false,
        })
    }

    pub(super) fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if self.finished {
            return Err("MP3 stream encoder is already finalized".into());
        }
        self.layout.fill_interleaved_i16(channels, &mut self.pcm)?;
        self.input_samples = self
            .input_samples
            .checked_add(self.pcm.len())
            .ok_or_else(|| "MP3 stream sample count overflows".to_string())?;
        self.pending.extend(self.pcm.iter().copied());
        let samples_per_frame = self.encoder.samples_per_frame();
        while self.pending.len() >= samples_per_frame {
            self.encode_pending_frame(samples_per_frame)?;
        }
        Ok(())
    }

    fn encode_pending_frame(&mut self, samples_per_frame: usize) -> Result<(), String> {
        self.frame.clear();
        self.frame.extend(
            self.pending
                .drain(..samples_per_frame.min(self.pending.len())),
        );
        self.frame.resize(samples_per_frame, 0);

        let encoded = self
            .encoder
            .encode_interleaved(&self.frame)
            .map_err(|error| format!("mp3 encode: {error}"))?;
        if encoded.len() != 1 || self.encoder.buffered_samples() != 0 {
            return Err(format!(
                "mp3 encoder returned {} frames with {} buffered samples for one complete input frame",
                encoded.len(),
                self.encoder.buffered_samples()
            ));
        }
        let encoded_bytes = encoded[0].len();
        self.output
            .write_all(&encoded[0])
            .map_err(|error| format!("write mp3: {error}"))?;

        // shine-rs 0.1.3 increases Layer III's part2_3_length to account for
        // reservoir stuffing, but does not emit those stuffing bits.  That is
        // observable for simple MPEG-2/2.5 input: the next header otherwise
        // begins before the byte length declared by the previous header.  Feed
        // one complete frame per call so the missing tail can be derived from
        // bits_per_frame, then materialize it before encoding another frame.
        let config = self.encoder.shine_config();
        let cached_bits = 32_i32
            .checked_sub(config.bs.cache_bits)
            .ok_or_else(|| "mp3 encoder reported an invalid bit cache".to_string())?;
        if !(0..32).contains(&cached_bits) {
            return Err(format!(
                "mp3 encoder reported {cached_bits} cached bits after a frame"
            ));
        }
        let actual_bits = encoded_bytes
            .checked_mul(8)
            .and_then(|bits| bits.checked_add(cached_bits as usize))
            .ok_or_else(|| "MP3 encoded frame bit count overflows".to_string())?;
        let declared_bits = usize::try_from(config.mpeg.bits_per_frame)
            .map_err(|_| "mp3 encoder reported a negative frame size".to_string())?;
        let stuffing_bits = declared_bits.checked_sub(actual_bits).ok_or_else(|| {
            format!("mp3 encoder emitted {actual_bits} bits for a {declared_bits}-bit frame")
        })?;
        let mut remaining = stuffing_bits;
        while remaining != 0 {
            let bits = remaining.min(32);
            config
                .bs
                .put_bits(0, bits as i32)
                .map_err(|error| format!("mp3 frame stuffing: {error}"))?;
            remaining -= bits;
        }
        if (32 - config.bs.cache_bits) % 8 != 0 {
            return Err("mp3 encoder did not end a declared frame on a byte boundary".into());
        }
        config
            .bs
            .flush()
            .map_err(|error| format!("mp3 frame bitstream flush: {error}"))?;
        let (tail, tail_bytes) = shine_rs::shine_flush(config);
        let declared_bytes = declared_bits.div_ceil(8);
        if encoded_bytes.checked_add(tail_bytes) != Some(declared_bytes) {
            return Err(format!(
                "mp3 encoder materialized {} bytes for a {declared_bytes}-byte frame",
                encoded_bytes.saturating_add(tail_bytes)
            ));
        }
        self.output
            .write_all(&tail[..tail_bytes])
            .map_err(|error| format!("write mp3 frame stuffing: {error}"))?;
        self.encoded_frames = self
            .encoded_frames
            .checked_add(1)
            .ok_or_else(|| "MP3 encoded frame count overflows".to_string())?;
        Ok(())
    }

    pub(super) fn finalize(mut self) -> Result<(), String> {
        if self.input_samples == 0 {
            return Err("MP3 output requires at least one frame".into());
        }
        let samples_per_frame = self.encoder.samples_per_frame();
        if !self.pending.is_empty() {
            self.encode_pending_frame(samples_per_frame)?;
        }
        while self.encoded_frames < 2 {
            self.encode_pending_frame(samples_per_frame)?;
        }
        let final_bytes = self
            .encoder
            .finish()
            .map_err(|error| format!("mp3 finish: {error}"))?;
        self.output
            .write_all(&final_bytes)
            .map_err(|error| format!("write mp3: {error}"))?;
        self.finished = true;
        self.output
            .flush()
            .map_err(|error| format!("flush mp3: {error}"))
    }
}

/// Write planar `f64` audio to an MP3 file.
pub fn write_mp3<P: AsRef<Path>>(path: P, audio: &Audio, bitrate_kbps: u32) -> Result<(), String> {
    write_mp3_with_downmix(path, audio, bitrate_kbps, DownmixMode::Preserve)
}

/// Write planar `f64` audio to MP3 with an explicit surround downmix policy.
pub fn write_mp3_with_downmix<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    bitrate_kbps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    EncodeOptions {
        mp3_bitrate_kbps: bitrate_kbps,
        downmix,
        ..EncodeOptions::default()
    }
    .validate_config(OutputFormat::Mp3, audio)?;
    let mut output = AtomicOutput::new(path)?;
    write_mp3_to_writer(output.file_mut(), audio, bitrate_kbps, downmix)?;
    output.commit(CommitMode::Replace)
}

pub(super) fn write_mp3_to_writer<W: Write>(
    output: W,
    audio: &Audio,
    bitrate_kbps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    let mut writer = Mp3StreamWriter::new(
        output,
        audio.sample_rate,
        audio.channels(),
        audio.channel_mask,
        bitrate_kbps,
        downmix,
    )?;
    writer.write_block(&audio.channels)?;
    writer.finalize()
}

pub(super) fn effective_mp3_config(
    audio: &Audio,
    requested_bitrate_kbps: u32,
    downmix: DownmixMode,
) -> Result<(EncodeChannels, Mp3EncoderConfig), String> {
    if audio.frames() == 0 {
        return Err("MP3 output requires at least one frame".into());
    }
    let layout = lossy_channel_layout(audio, downmix)?;
    let config = effective_mp3_stream_config(audio.sample_rate, layout, requested_bitrate_kbps)?;
    Ok((layout, config))
}

pub(super) fn effective_mp3_stream_config(
    sample_rate: u32,
    layout: EncodeChannels,
    requested_bitrate_kbps: u32,
) -> Result<Mp3EncoderConfig, String> {
    let stereo_mode = if layout.is_stereo {
        StereoMode::JointStereo
    } else {
        StereoMode::Mono
    };
    let build_config = |bitrate| Mp3EncoderConfig {
        sample_rate,
        bitrate,
        channels: layout.count,
        stereo_mode,
        copyright: false,
        original: true,
    };
    let bitrate = effective_mp3_bitrate_kbps(sample_rate, requested_bitrate_kbps)?;
    let config = build_config(bitrate);
    config
        .validate()
        .map_err(|error| format!("MP3 encoder config: {error}"))?;
    Ok(config)
}

pub(crate) fn effective_mp3_bitrate_kbps(
    sample_rate: u32,
    requested_bitrate_kbps: u32,
) -> Result<u32, String> {
    if !shine_rs::SUPPORTED_SAMPLE_RATES.contains(&sample_rate) {
        return Err(format!(
            "MP3 encode: unsupported sample rate {sample_rate} Hz (supported: {:?})",
            shine_rs::SUPPORTED_SAMPLE_RATES,
        ));
    }
    let build_config = |bitrate| Mp3EncoderConfig {
        sample_rate,
        bitrate,
        channels: 1,
        stereo_mode: StereoMode::Mono,
        copyright: false,
        original: true,
    };
    SUPPORTED_BITRATES
        .iter()
        .copied()
        .filter(|bitrate| *bitrate <= requested_bitrate_kbps)
        .rev()
        .find(|bitrate| build_config(*bitrate).validate().is_ok())
        .or_else(|| {
            SUPPORTED_BITRATES
                .iter()
                .copied()
                .find(|bitrate| build_config(*bitrate).validate().is_ok())
        })
        .ok_or_else(|| format!("MP3 encode: no compatible bitrate for {sample_rate} Hz"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn sine_stereo(sr: u32, secs: f32) -> Audio {
        let frames = (sr as f32 * secs) as usize;
        let mut l = Vec::with_capacity(frames);
        let mut r = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f64 / sr as f64;
            let v = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.25;
            l.push(v);
            r.push(v * 0.8);
        }
        Audio {
            sample_rate: sr,
            channels: vec![l, r],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("denoize_mp3_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn mp3_roundtrip_decode() {
        let path = tmp("rt.mp3");
        let audio = sine_stereo(44100, 0.5);
        write_mp3(&path, &audio, 128).unwrap();
        assert!(path.metadata().unwrap().len() > 100);

        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.n_channels(), 2);
        assert_eq!(
            decoded.channel_mask,
            crate::channel_layout::ChannelLayout::Stereo.mask()
        );
        assert!(decoded.frames() > 10000);

        // Lossy but should retain energy
        let rms_in: f64 =
            audio.channels[0].iter().map(|s| s * s).sum::<f64>() / audio.frames() as f64;
        let rms_out: f64 =
            decoded.channels[0].iter().map(|s| s * s).sum::<f64>() / decoded.frames() as f64;
        assert!(rms_out > 0.01);
        assert!(rms_out < rms_in * 2.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mpeg2_stream_materializes_each_declared_frame() {
        let audio = sine_stereo(16_000, 0.2);
        let mut encoded = Vec::new();
        let mut writer = Mp3StreamWriter::new(
            &mut encoded,
            audio.sample_rate,
            audio.channels(),
            audio.channel_mask,
            DEFAULT_MP3_BITRATE,
            DownmixMode::Preserve,
        )
        .unwrap();
        for start in (0..audio.frames()).step_by(127) {
            let end = (start + 127).min(audio.frames());
            let block: Vec<Vec<f64>> = audio
                .channels
                .iter()
                .map(|channel| channel[start..end].to_vec())
                .collect();
            writer.write_block(&block).unwrap();
        }
        writer.finalize().unwrap();

        // MPEG-2 Layer III carries 576 samples/channel per packet.  At
        // 16 kHz and the selected 160 kbps CBR fallback, each header declares
        // 720 bytes.  Every header must begin exactly at that boundary.
        let packet_bytes = 720;
        let packets = audio.frames().div_ceil(576);
        assert_eq!(encoded.len(), packets * packet_bytes);
        for packet in encoded.chunks_exact(packet_bytes) {
            assert_eq!(&packet[..2], &[0xff, 0xf3]);
        }
    }

    #[test]
    fn direct_writer_validates_before_staging() {
        let path = tmp("preserve.mp3");
        std::fs::write(&path, b"existing output").unwrap();
        let audio = sine_stereo(12_345, 0.1);

        let error = write_mp3(&path, &audio, 128).unwrap_err();

        assert!(error.contains("unsupported sample rate"));
        assert_eq!(std::fs::read(&path).unwrap(), b"existing output");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn subminimum_bitrate_uses_the_lowest_rate_compatible_fallback() {
        for sample_rate in [32_000, 44_100, 48_000] {
            let audio = sine_stereo(sample_rate, 0.01);
            for requested in [0, 8, 16, 24, 31] {
                let (_, config) =
                    effective_mp3_config(&audio, requested, DownmixMode::Preserve).unwrap();
                assert_eq!(config.bitrate, 32);
                config.validate().unwrap();
            }
        }
    }

    #[test]
    fn compatible_bitrates_still_round_down() {
        let audio = sine_stereo(44_100, 0.01);
        for (requested, expected) in [(33, 32), (191, 160), (u32::MAX, 320)] {
            let (_, config) =
                effective_mp3_config(&audio, requested, DownmixMode::Preserve).unwrap();
            assert_eq!(config.bitrate, expected);
            config.validate().unwrap();
        }
    }
}
