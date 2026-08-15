use crate::Audio;
use ogg::writing::{PacketWriteEndInfo, PacketWriter};
use opus::{Application, Bitrate, Channels, Encoder};
use std::borrow::Cow;
use std::io::{BufWriter, Write};

use super::pcm::StreamPcmLayout;
use super::DownmixMode;

const OPUS_FRAME_SIZE: usize = 960;
const OPUS_PACKET_BYTES: usize = 4_000;
const OGG_SERIAL: u32 = 0x444e_5a45;

/// Bounded block-oriented Ogg Opus writer.
pub(super) struct OggOpusStreamWriter<W: Write> {
    writer: PacketWriter<'static, BufWriter<W>>,
    encoder: Encoder,
    resampler: crate::resample::StreamingResampler,
    layout: StreamPcmLayout,
    pcm: Vec<f32>,
    pending_packet: Option<(Vec<u8>, u64)>,
    channels: usize,
    encoded_frames: u64,
    input_frames: u64,
}

impl<W: Write> OggOpusStreamWriter<W> {
    pub(super) fn new(
        output: W,
        sample_rate: u32,
        input_channels: usize,
        channel_mask: Option<crate::ChannelMask>,
        bitrate: u32,
        downmix: DownmixMode,
    ) -> Result<Self, String> {
        let layout = StreamPcmLayout::new(input_channels, channel_mask, downmix)?;
        let channels = layout.output().count as usize;
        let opus_channels = if channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };
        let mut encoder = Encoder::new(48_000, opus_channels, Application::Audio)
            .map_err(|error| format!("Opus encoder: {error}"))?;
        encoder
            .set_bitrate(Bitrate::Bits(bitrate as i32))
            .map_err(|error| format!("Opus bitrate: {error}"))?;
        let pre_skip = encoder
            .get_lookahead()
            .map_err(|error| format!("Opus lookahead: {error}"))? as u16;
        let mut writer = PacketWriter::new(BufWriter::new(output));
        let mut head = b"OpusHead".to_vec();
        head.extend([1, channels as u8]);
        head.extend(pre_skip.to_le_bytes());
        head.extend(sample_rate.to_le_bytes());
        head.extend(0_i16.to_le_bytes());
        head.push(0);
        writer
            .write_packet(Cow::Owned(head), OGG_SERIAL, PacketWriteEndInfo::EndPage, 0)
            .map_err(|error| format!("Ogg header: {error}"))?;
        let vendor = b"denoize";
        let mut tags = b"OpusTags".to_vec();
        tags.extend((vendor.len() as u32).to_le_bytes());
        tags.extend(vendor);
        tags.extend(0_u32.to_le_bytes());
        writer
            .write_packet(Cow::Owned(tags), OGG_SERIAL, PacketWriteEndInfo::EndPage, 0)
            .map_err(|error| format!("Ogg tags: {error}"))?;
        let frame_samples = OPUS_FRAME_SIZE
            .checked_mul(channels)
            .ok_or_else(|| "Opus frame sample count overflows".to_string())?;
        let mut pcm = Vec::new();
        pcm.try_reserve_exact(frame_samples)
            .map_err(|error| format!("reserve Opus frame: {error}"))?;
        // libopus reports the number of decoded samples that must be discarded
        // before the first presentation sample. Feed that many leading zeroes
        // into the encoder so discarding `pre_skip` removes codec warm-up, not
        // the beginning of the caller's audio. Granule positions then count
        // this complete decoded timeline and the EOS granule trims only the
        // padded tail of the final Opus packet.
        let leading_samples = usize::from(pre_skip)
            .checked_mul(channels)
            .ok_or_else(|| "Opus pre-skip sample count overflows".to_string())?;
        pcm.resize(leading_samples, 0.0);
        Ok(Self {
            writer,
            encoder,
            resampler: crate::resample::StreamingResampler::new(channels, sample_rate, 48_000)?,
            layout,
            pcm,
            pending_packet: None,
            channels,
            encoded_frames: 0,
            input_frames: 0,
        })
    }

    pub(super) fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        let frames = self.layout.validate_block(channels)?;
        self.input_frames = self
            .input_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "Opus source frame count overflows".to_string())?;
        let converted_input = self.layout.convert_planar_f64(channels)?;
        let converted = self.resampler.process(&converted_input)?;
        self.append_resampled(&converted)
    }

    fn append_resampled(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if channels.len() != self.channels {
            return Err("Opus resampler returned an invalid channel count".into());
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("Opus resampler returned unaligned channels".into());
        }
        for frame in 0..frames {
            for channel in channels {
                self.pcm.push(crate::sanitize_sample(channel[frame]) as f32);
            }
            if self.pcm.len() == OPUS_FRAME_SIZE * self.channels {
                self.encode_pending(OPUS_FRAME_SIZE)?;
            }
        }
        Ok(())
    }

    fn encode_pending(&mut self, actual_frames: usize) -> Result<(), String> {
        let packet = self
            .encoder
            .encode_vec_float(&self.pcm, OPUS_PACKET_BYTES)
            .map_err(|error| format!("Opus encode: {error}"))?;
        self.encoded_frames = self
            .encoded_frames
            .checked_add(actual_frames as u64)
            .ok_or_else(|| "Opus granule position overflows".to_string())?;
        let granule = self.encoded_frames;
        if let Some((previous, previous_granule)) = self.pending_packet.replace((packet, granule)) {
            self.writer
                .write_packet(
                    Cow::Owned(previous),
                    OGG_SERIAL,
                    PacketWriteEndInfo::EndPage,
                    previous_granule,
                )
                .map_err(|error| format!("Ogg write: {error}"))?;
        }
        self.pcm.clear();
        Ok(())
    }

    pub(super) fn finalize(mut self) -> Result<(), String> {
        if self.input_frames == 0 {
            return Err("Opus output requires at least one frame".into());
        }
        let tail = self.resampler.finish()?;
        self.append_resampled(&tail)?;
        if !self.pcm.is_empty() {
            let actual_frames = self.pcm.len() / self.channels;
            self.pcm.resize(OPUS_FRAME_SIZE * self.channels, 0.0);
            self.encode_pending(actual_frames)?;
        }
        let (packet, granule) = self
            .pending_packet
            .take()
            .ok_or_else(|| "Opus encoder produced no audio packet".to_string())?;
        self.writer
            .write_packet(
                Cow::Owned(packet),
                OGG_SERIAL,
                PacketWriteEndInfo::EndStream,
                granule,
            )
            .map_err(|error| format!("Ogg write: {error}"))?;
        let mut output = self.writer.into_inner();
        output
            .flush()
            .map_err(|error| format!("Ogg flush: {error}"))
    }
}

pub(super) fn write_ogg_opus_to_writer<W: Write>(
    output: W,
    audio: &Audio,
    bitrate: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    let mut writer = OggOpusStreamWriter::new(
        output,
        audio.sample_rate,
        audio.channels(),
        audio.channel_mask,
        bitrate,
        downmix,
    )?;
    writer.write_block(&audio.channels)?;
    writer.finalize()
}
