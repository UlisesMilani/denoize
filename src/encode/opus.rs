use crate::Audio;
use ogg::writing::{PacketWriteEndInfo, PacketWriter};
use opus::{Application, Bitrate, Channels, Encoder};
use std::borrow::Cow;
use std::io::{BufWriter, Write};

use super::pcm::{downmix_to_stereo, lossy_channel_layout};
use super::DownmixMode;

pub(super) fn write_ogg_opus_to_writer<W: Write>(
    output: W,
    audio: &Audio,
    bitrate: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    if audio.frames() == 0 {
        return Err("Opus output requires at least one frame".into());
    }
    let source_channels = if audio.channels() > 2 {
        // Validate the explicit policy and known layout before rendering.
        lossy_channel_layout(audio, downmix)?;
        downmix_to_stereo(audio)?
    } else {
        if audio.channels() == 0 {
            return Err("Opus requires at least one channel".into());
        }
        audio.channels.clone()
    };
    let converted =
        crate::resample::resample_channels(&source_channels, audio.sample_rate, 48_000)?;
    let count = converted.len();
    let channels = if count == 1 {
        Channels::Mono
    } else {
        Channels::Stereo
    };
    let mut encoder = Encoder::new(48_000, channels, Application::Audio)
        .map_err(|e| format!("Opus encoder: {e}"))?;
    encoder
        .set_bitrate(Bitrate::Bits(bitrate as i32))
        .map_err(|e| format!("Opus bitrate: {e}"))?;
    let pre_skip = encoder
        .get_lookahead()
        .map_err(|e| format!("Opus lookahead: {e}"))? as u16;
    let mut writer = PacketWriter::new(BufWriter::new(output));
    let serial = 0x444e_5a45;
    let mut head = b"OpusHead".to_vec();
    head.extend([1, count as u8]);
    head.extend(pre_skip.to_le_bytes());
    head.extend(audio.sample_rate.to_le_bytes());
    head.extend(0_i16.to_le_bytes());
    head.push(0);
    writer
        .write_packet(Cow::Owned(head), serial, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| format!("Ogg header: {e}"))?;
    let vendor = b"denoize";
    let mut tags = b"OpusTags".to_vec();
    tags.extend((vendor.len() as u32).to_le_bytes());
    tags.extend(vendor);
    tags.extend(0_u32.to_le_bytes());
    writer
        .write_packet(Cow::Owned(tags), serial, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| format!("Ogg tags: {e}"))?;
    let frame_size = 960;
    let total = converted[0].len();
    let mut position = 0;
    let mut granule = pre_skip as u64;
    while position < total {
        let actual = (total - position).min(frame_size);
        let mut pcm = vec![0.0f32; frame_size * count];
        for frame in 0..actual {
            for channel in 0..count {
                pcm[frame * count + channel] = converted[channel][position + frame] as f32;
            }
        }
        let packet = encoder
            .encode_vec_float(&pcm, 4_000)
            .map_err(|e| format!("Opus encode: {e}"))?;
        granule += actual as u64;
        position += actual;
        let end = if position == total {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::EndPage
        };
        writer
            .write_packet(Cow::Owned(packet), serial, end, granule)
            .map_err(|e| format!("Ogg write: {e}"))?;
    }
    let mut output = writer.into_inner();
    output.flush().map_err(|e| format!("Ogg flush: {e}"))
}
