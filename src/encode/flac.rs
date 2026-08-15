use std::io::{BufWriter, Seek, SeekFrom, Write};

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, Stream, StreamInfo};
use flacenc::error::{Verified, Verify};
use flacenc::source::{Context, Fill, FrameBuf};

use crate::audio::{sanitize_sample, Audio};

/// Bounded block-oriented FLAC encoder.
///
/// Only one configured FLAC frame, one interleaved PCM block, and one encoded
/// frame are retained. STREAMINFO is written as a fixed-size placeholder and
/// rewritten in place after the final frame, so output memory does not grow
/// with the duration of the input.
pub(super) struct FlacStreamWriter<W: Write + Seek> {
    output: BufWriter<W>,
    config: Verified<flacenc::config::Encoder>,
    stream_info: StreamInfo,
    frame_buffer: FrameBuf,
    context: Context,
    encoded: ByteSink,
    pending: Vec<i32>,
    channels: usize,
    bits_per_sample: usize,
    block_samples: usize,
    frame_number: usize,
    input_frames: u64,
}

impl<W: Write + Seek> FlacStreamWriter<W> {
    pub(super) fn new(
        output: W,
        sample_rate: u32,
        channels: usize,
        bits_per_sample: u16,
    ) -> Result<Self, String> {
        validate_geometry(sample_rate, channels, bits_per_sample)?;
        let bits_per_sample = effective_bits(bits_per_sample)?;
        let config = flacenc::config::Encoder::default()
            .into_verified()
            .map_err(|error| format!("FLAC config: {:?}", error.1))?;
        let block_samples = config
            .block_size
            .checked_mul(channels)
            .ok_or_else(|| "FLAC block sample count overflows".to_string())?;
        let stream_info = StreamInfo::new(sample_rate as usize, channels, bits_per_sample)
            .map_err(|error| format!("FLAC stream info: {error}"))?;
        let frame_buffer = FrameBuf::with_size(channels, config.block_size)
            .map_err(|error| format!("FLAC frame buffer: {error}"))?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(block_samples)
            .map_err(|error| format!("reserve FLAC PCM block: {error}"))?;
        let mut output = BufWriter::new(output);
        let header = serialize_header(&stream_info)?;
        output
            .write_all(&header)
            .map_err(|error| format!("FLAC write header: {error}"))?;
        Ok(Self {
            output,
            config,
            stream_info,
            frame_buffer,
            context: Context::new(bits_per_sample, channels),
            encoded: ByteSink::new(),
            pending,
            channels,
            bits_per_sample,
            block_samples,
            frame_number: 0,
            input_frames: 0,
        })
    }

    pub(super) fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if channels.len() != self.channels {
            return Err(format!(
                "FLAC stream encoder expected {} channels, received {}",
                self.channels,
                channels.len()
            ));
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("FLAC stream encode blocks must have equal channel lengths".into());
        }
        let next_frames = self
            .input_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "FLAC stream frame count overflows".to_string())?;
        if next_frames >= 1_u64 << 36 {
            return Err("FLAC stream exceeds the 36-bit total-sample limit".into());
        }
        let scale = (1_i64 << (self.bits_per_sample - 1)) as f64;
        for frame in 0..frames {
            for channel in channels {
                self.pending.push(
                    (sanitize_sample(channel[frame]) * scale)
                        .round()
                        .clamp(-scale, scale - 1.0) as i32,
                );
            }
            if self.pending.len() == self.block_samples {
                self.encode_pending()?;
            }
        }
        self.input_frames = next_frames;
        Ok(())
    }

    fn encode_pending(&mut self) -> Result<(), String> {
        if self.pending.is_empty() {
            return Ok(());
        }
        if self.frame_number >= 1usize << 31 {
            return Err("FLAC stream frame number exceeds the format limit".into());
        }
        self.frame_buffer
            .fill_interleaved(&self.pending)
            .map_err(|error| format!("fill FLAC frame: {error}"))?;
        self.context
            .fill_interleaved(&self.pending)
            .map_err(|error| format!("update FLAC digest: {error}"))?;
        let frame = flacenc::encode_fixed_size_frame(
            &self.config,
            &self.frame_buffer,
            self.frame_number,
            &self.stream_info,
        )
        .map_err(|error| format!("FLAC encode: {error}"))?;
        self.encoded.clear();
        frame
            .write(&mut self.encoded)
            .map_err(|error| format!("FLAC serialize: {error}"))?;
        self.output
            .write_all(self.encoded.as_slice())
            .map_err(|error| format!("FLAC write: {error}"))?;
        self.stream_info.update_frame_info(&frame);
        self.frame_number += 1;
        self.pending.clear();
        Ok(())
    }

    pub(super) fn finalize(mut self) -> Result<(), String> {
        if self.input_frames == 0 {
            return Err("FLAC output requires at least one frame".into());
        }
        self.encode_pending()?;
        self.stream_info.set_md5_digest(&self.context.md5_digest());
        let header = serialize_header(&self.stream_info)?;
        self.output
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("seek FLAC header: {error}"))?;
        self.output
            .write_all(&header)
            .map_err(|error| format!("rewrite FLAC header: {error}"))?;
        self.output
            .seek(SeekFrom::End(0))
            .map_err(|error| format!("seek FLAC end: {error}"))?;
        self.output
            .flush()
            .map_err(|error| format!("FLAC flush: {error}"))
    }
}

fn serialize_header(stream_info: &StreamInfo) -> Result<Vec<u8>, String> {
    let stream = Stream::with_stream_info(stream_info.clone());
    let mut sink = ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|error| format!("FLAC serialize header: {error}"))?;
    let bytes = sink.into_inner();
    if bytes.len() != 42 {
        return Err(format!(
            "FLAC STREAMINFO header has unexpected {}-byte length",
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn effective_bits(bits_per_sample: u16) -> Result<usize, String> {
    let bits = bits_per_sample.clamp(8, 24) as usize;
    if !matches!(bits, 8 | 12 | 16 | 20 | 24) {
        return Err(format!(
            "FLAC encode: unsupported effective bit depth {bits} (supported: 8, 12, 16, 20, 24)"
        ));
    }
    Ok(bits)
}

pub(super) fn validate_geometry(
    sample_rate: u32,
    channels: usize,
    bits_per_sample: u16,
) -> Result<(), String> {
    if channels == 0 || channels > 8 {
        return Err(format!(
            "FLAC encode: unsupported channel count {channels} (supported: 1..=8)"
        ));
    }
    if sample_rate == 0 || sample_rate > 96_000 {
        return Err(format!(
            "FLAC encode: unsupported sample rate {sample_rate} Hz (supported: 1..=96000)"
        ));
    }
    effective_bits(bits_per_sample).map(|_| ())
}

pub(super) fn write_flac_to_writer<W: Write + Seek>(
    output: W,
    audio: &Audio,
) -> Result<(), String> {
    let mut writer = FlacStreamWriter::new(
        output,
        audio.sample_rate,
        audio.channels(),
        audio.bits_per_sample,
    )?;
    writer.write_block(&audio.channels)?;
    writer.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn block_writer_roundtrips_without_retaining_the_whole_stream() {
        let mut output = Cursor::new(Vec::new());
        {
            let mut writer = FlacStreamWriter::new(&mut output, 48_000, 2, 16).unwrap();
            for block in 0..20 {
                let start = block * 733;
                let left = (start..start + 733)
                    .map(|index| (index as f64 / 47.0).sin() * 0.2)
                    .collect::<Vec<_>>();
                writer.write_block(&[left.clone(), left]).unwrap();
            }
            writer.finalize().unwrap();
        }
        assert!(output.get_ref().starts_with(b"fLaC"));
        output.set_position(0);
        let decoded = claxon::FlacReader::new(output).unwrap();
        assert_eq!(decoded.streaminfo().sample_rate, 48_000);
        assert_eq!(decoded.streaminfo().channels, 2);
        assert_eq!(decoded.streaminfo().samples, Some(20 * 733));
    }
}
