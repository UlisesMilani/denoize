//! M4A (AAC-LC in MP4) encode — Pure-Rust `oxideav-aac` + `mp4` muxer.

use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use mp4::{
    AacConfig, AudioObjectType as Mp4Aot, ChannelConfig, FourCC, MediaConfig, Mp4Config, Mp4Writer,
    SampleFreqIndex, TrackConfig, TrackType,
};
use oxideav_aac_encoder::adts::ADTS_HEADER_BYTES_NO_CRC;
use oxideav_aac_encoder::encoder::{EncoderConfig, StreamEncoder, FRAME_LEN};

use crate::atomic_output::{AtomicOutput, CommitMode};
use crate::audio::Audio;

use super::pcm::StreamPcmLayout;
use super::{AacEncoder, DownmixMode, EncodeOptions, OutputFormat};

pub(super) const TABLE_RECORD_BYTES: u64 = 12;
pub(super) const DEFAULT_MAX_TABLE_BYTES: u64 = 1024 * 1024 * 1024;
const MP4_BOX_HEADER_BYTES: u64 = 8;
const MP4_FULL_BOX_HEADER_BYTES: u64 = 4;

/// Bounded block-oriented AAC-in-MP4 writer.
///
/// AAC access units are written directly to `mdat`. Their fixed-width size and
/// offset records are stored in an anonymous file rather than a duration-sized
/// RAM vector; final `stsz` and `co64` boxes are streamed from that file.
pub(super) struct M4aStreamWriter<W: Write + Seek> {
    muxer: BoundedM4aMuxer<W>,
    encoder: StreamEncoder,
    layout: StreamPcmLayout,
    converted: Vec<i16>,
    pending: Vec<i16>,
    frame_samples: usize,
    input_frames: u64,
}

impl<W: Write + Seek> M4aStreamWriter<W> {
    pub(super) fn new(
        output: W,
        sample_rate: u32,
        input_channels: usize,
        channel_mask: Option<crate::ChannelMask>,
        bitrate_bps: u32,
        downmix: DownmixMode,
        max_table_bytes: Option<u64>,
    ) -> Result<Self, String> {
        let freq_index = sample_rate_to_index(sample_rate)?;
        let layout = StreamPcmLayout::new(input_channels, channel_mask, downmix)?;
        let encoder = StreamEncoder::new(EncoderConfig {
            sample_rate,
            channels: layout.output().count,
            bitrate: bitrate_bps,
        })
        .map_err(|error| format!("aac encoder init: {error}"))?;
        let frame_samples = FRAME_LEN
            .checked_mul(layout.output().count as usize)
            .ok_or_else(|| "M4A AAC frame sample count overflows".to_string())?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(frame_samples)
            .map_err(|error| format!("reserve M4A AAC frame: {error}"))?;
        let chan_conf = if layout.output().is_stereo {
            ChannelConfig::Stereo
        } else {
            ChannelConfig::Mono
        };
        Ok(Self {
            muxer: BoundedM4aMuxer::new(
                output,
                sample_rate,
                bitrate_bps,
                freq_index,
                chan_conf,
                FRAME_LEN as u32,
                FRAME_LEN as u64,
                max_table_bytes.unwrap_or(DEFAULT_MAX_TABLE_BYTES),
            )?,
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
            .ok_or_else(|| "M4A source frame count overflows".to_string())?;
        let mut position = 0usize;
        while position < self.converted.len() {
            let take =
                (self.frame_samples - self.pending.len()).min(self.converted.len() - position);
            self.pending
                .extend_from_slice(&self.converted[position..position + take]);
            position += take;
            if self.pending.len() == self.frame_samples {
                self.encode_pending()?;
            }
        }
        Ok(())
    }

    fn encode_pending(&mut self) -> Result<(), String> {
        let adts = self
            .encoder
            .encode_frame(&self.pending)
            .map_err(|error| format!("aac encode: {error}"))?;
        self.muxer.write_adts_frame(&adts)?;
        self.pending.clear();
        Ok(())
    }

    pub(super) fn finalize(mut self) -> Result<(), String> {
        if self.input_frames == 0 {
            return Err("M4A output requires at least one frame".into());
        }
        if !self.pending.is_empty() {
            self.encode_pending()?;
        }
        let final_frame = self
            .encoder
            .finish()
            .map_err(|error| format!("aac finish: {error}"))?;
        self.muxer.write_adts_frame(&final_frame)?;
        self.muxer.finalize(self.input_frames)
    }
}

pub(super) struct BoundedM4aMuxer<W: Write + Seek> {
    output: BufWriter<W>,
    table: File,
    template: M4aTemplate,
    mdat_position: u64,
    output_position: u64,
    sample_rate: u32,
    bitrate_bps: u32,
    freq_index: SampleFreqIndex,
    chan_conf: ChannelConfig,
    sample_duration: u32,
    encoder_delay: u64,
    sample_count: u32,
    max_sample_size: u32,
    max_chunk_offset: u64,
    table_bytes: u64,
    max_table_bytes: u64,
}

impl<W: Write + Seek> BoundedM4aMuxer<W> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        output: W,
        sample_rate: u32,
        bitrate_bps: u32,
        freq_index: SampleFreqIndex,
        chan_conf: ChannelConfig,
        sample_duration: u32,
        encoder_delay: u64,
        max_table_bytes: u64,
    ) -> Result<Self, String> {
        if max_table_bytes < TABLE_RECORD_BYTES {
            return Err(format!(
                "M4A sample-table limit must be at least {TABLE_RECORD_BYTES} bytes"
            ));
        }
        if sample_duration == 0 {
            return Err("M4A sample duration must be greater than zero".into());
        }
        let template = M4aTemplate::build(sample_rate, bitrate_bps, freq_index, chan_conf)?;
        let mut output = BufWriter::new(output);
        output
            .write_all(&template.ftyp)
            .map_err(|error| format!("write M4A ftyp: {error}"))?;
        let mdat_position = template.ftyp.len() as u64;
        // Match the conventional `mdat` + `wide` placeholder used by the mp4
        // crate. A small stream keeps the 32-bit `mdat` header (and therefore
        // remains compatible with metadata tools which cannot skip an
        // extended atom); a stream crossing 4 GiB overwrites `wide` with the
        // 64-bit size without shifting any media bytes.
        output
            .write_all(&0_u32.to_be_bytes())
            .and_then(|_| output.write_all(b"mdat"))
            .and_then(|_| output.write_all(&8_u32.to_be_bytes()))
            .and_then(|_| output.write_all(b"wide"))
            .map_err(|error| format!("write M4A mdat header: {error}"))?;
        let output_position = mdat_position
            .checked_add(16)
            .ok_or_else(|| "M4A output position overflows".to_string())?;
        let table = tempfile::tempfile()
            .map_err(|error| format!("create bounded M4A sample table: {error}"))?;
        Ok(Self {
            output,
            table,
            template,
            mdat_position,
            output_position,
            sample_rate,
            bitrate_bps,
            freq_index,
            chan_conf,
            sample_duration,
            encoder_delay,
            sample_count: 0,
            max_sample_size: 0,
            max_chunk_offset: 0,
            table_bytes: 0,
            max_table_bytes,
        })
    }

    fn write_adts_frame(&mut self, adts: &[u8]) -> Result<(), String> {
        if adts.len() <= ADTS_HEADER_BYTES_NO_CRC {
            return Ok(());
        }
        self.write_raw_access_unit(&adts[ADTS_HEADER_BYTES_NO_CRC..])
    }

    pub(super) fn write_raw_access_unit(&mut self, raw: &[u8]) -> Result<(), String> {
        if raw.is_empty() {
            return Ok(());
        }
        let size = u32::try_from(raw.len())
            .map_err(|_| "M4A AAC access unit exceeds the MP4 size field".to_string())?;
        if size > 0x00ff_ffff {
            return Err("M4A AAC access unit exceeds the 24-bit decoder buffer field".into());
        }
        let next_table_bytes = self
            .table_bytes
            .checked_add(TABLE_RECORD_BYTES)
            .ok_or_else(|| "M4A sample-table byte count overflows".to_string())?;
        if next_table_bytes > self.max_table_bytes {
            return Err(format!(
                "M4A sample table requires {next_table_bytes} bytes, exceeding its {}-byte limit",
                self.max_table_bytes
            ));
        }
        let next_sample_count = self
            .sample_count
            .checked_add(1)
            .ok_or_else(|| "M4A sample count exceeds the MP4 limit".to_string())?;
        let next_position = self
            .output_position
            .checked_add(raw.len() as u64)
            .ok_or_else(|| "M4A output length overflows".to_string())?;
        self.table
            .write_all(&size.to_be_bytes())
            .and_then(|_| self.table.write_all(&self.output_position.to_be_bytes()))
            .map_err(|error| format!("write bounded M4A sample table: {error}"))?;
        self.output
            .write_all(raw)
            .map_err(|error| format!("write M4A access unit: {error}"))?;
        self.max_chunk_offset = self.max_chunk_offset.max(self.output_position);
        self.output_position = next_position;
        self.sample_count = next_sample_count;
        self.max_sample_size = self.max_sample_size.max(size);
        self.table_bytes = next_table_bytes;
        Ok(())
    }

    pub(super) fn finalize(mut self, presentation_frames: u64) -> Result<(), String> {
        if self.sample_count == 0 {
            return Err("M4A encoder produced no AAC access units".into());
        }
        if presentation_frames == 0 {
            return Err("M4A output requires at least one presentation frame".into());
        }
        self.output
            .flush()
            .map_err(|error| format!("flush M4A media data: {error}"))?;
        let media_end = self
            .output
            .seek(SeekFrom::End(0))
            .map_err(|error| format!("seek M4A media end: {error}"))?;
        if media_end != self.output_position {
            return Err("M4A media position changed while muxing".into());
        }
        let mdat_size = media_end
            .checked_sub(self.mdat_position)
            .ok_or_else(|| "M4A mdat size underflows".to_string())?;
        self.output
            .seek(SeekFrom::Start(self.mdat_position))
            .and_then(|_| {
                if let Ok(size) = u32::try_from(mdat_size) {
                    self.output.write_all(&size.to_be_bytes())?;
                    self.output.write_all(b"mdat")
                } else {
                    self.output.write_all(&1_u32.to_be_bytes())?;
                    self.output.write_all(b"mdat")?;
                    self.output.write_all(&mdat_size.to_be_bytes())
                }
            })
            .map_err(|error| format!("rewrite M4A mdat size: {error}"))?;
        self.output
            .seek(SeekFrom::Start(media_end))
            .map_err(|error| format!("seek M4A metadata position: {error}"))?;
        self.table
            .flush()
            .and_then(|_| self.table.seek(SeekFrom::Start(0)).map(|_| ()))
            .map_err(|error| format!("rewind bounded M4A sample table: {error}"))?;
        self.template.write_moov(
            &mut self.output,
            &mut self.table,
            self.sample_count,
            presentation_frames,
            self.sample_duration,
            self.encoder_delay,
            self.sample_rate,
            self.bitrate_bps,
            self.freq_index,
            self.chan_conf,
            self.max_sample_size,
            self.max_chunk_offset,
        )?;
        self.output
            .flush()
            .map_err(|error| format!("mp4 flush: {error}"))
    }
}

struct M4aTemplate {
    ftyp: Vec<u8>,
    mvhd: Vec<u8>,
    tkhd: Vec<u8>,
    mdhd: Vec<u8>,
    hdlr: Vec<u8>,
    smhd: Vec<u8>,
    dinf: Vec<u8>,
    stsd: Vec<u8>,
}

impl M4aTemplate {
    fn build(
        sample_rate: u32,
        bitrate_bps: u32,
        freq_index: SampleFreqIndex,
        chan_conf: ChannelConfig,
    ) -> Result<Self, String> {
        let brand = |value: &str| {
            value
                .parse::<FourCC>()
                .map_err(|error| format!("mp4 brand '{value}': {error}"))
        };
        let cursor = Cursor::new(Vec::new());
        let mut writer = Mp4Writer::write_start(
            cursor,
            &Mp4Config {
                major_brand: brand("M4A ")?,
                minor_version: 0,
                compatible_brands: vec![brand("M4A ")?, brand("mp42")?, brand("isom")?],
                timescale: sample_rate,
            },
        )
        .map_err(|error| format!("build bounded M4A template: {error}"))?;
        writer
            .add_track(&TrackConfig {
                track_type: TrackType::Audio,
                timescale: sample_rate,
                language: "und".into(),
                media_conf: MediaConfig::AacConfig(AacConfig {
                    bitrate: bitrate_bps,
                    profile: Mp4Aot::AacLowComplexity,
                    freq_index,
                    chan_conf,
                }),
            })
            .map_err(|error| format!("build bounded M4A track template: {error}"))?;
        writer
            .write_end()
            .map_err(|error| format!("finish bounded M4A template: {error}"))?;
        let bytes = writer.into_writer().into_inner();
        let ftyp = find_box(&bytes, b"ftyp")?.to_vec();
        let moov = find_box(&bytes, b"moov")?;
        let mvhd = find_child(moov, b"mvhd")?.to_vec();
        let trak = find_child(moov, b"trak")?;
        let tkhd = find_child(trak, b"tkhd")?.to_vec();
        let mdia = find_child(trak, b"mdia")?;
        let mdhd = find_child(mdia, b"mdhd")?.to_vec();
        let hdlr = find_child(mdia, b"hdlr")?.to_vec();
        let minf = find_child(mdia, b"minf")?;
        let smhd = find_child(minf, b"smhd")?.to_vec();
        let dinf = find_child(minf, b"dinf")?.to_vec();
        let stbl = find_child(minf, b"stbl")?;
        let stsd = find_child(stbl, b"stsd")?.to_vec();
        Ok(Self {
            ftyp,
            mvhd,
            tkhd,
            mdhd,
            hdlr,
            smhd,
            dinf,
            stsd,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn write_moov<W: Write>(
        &self,
        output: &mut W,
        table: &mut File,
        sample_count: u32,
        presentation_frames: u64,
        sample_duration: u32,
        encoder_delay: u64,
        sample_rate: u32,
        _bitrate_bps: u32,
        _freq_index: SampleFreqIndex,
        _chan_conf: ChannelConfig,
        max_sample_size: u32,
        max_chunk_offset: u64,
    ) -> Result<(), String> {
        let media_duration = u64::from(sample_count)
            .checked_mul(u64::from(sample_duration))
            .ok_or_else(|| "M4A duration overflows".to_string())?;
        let required_media_duration = presentation_frames
            .checked_add(encoder_delay)
            .ok_or_else(|| "M4A presentation duration overflows".to_string())?;
        if required_media_duration > media_duration {
            return Err(format!(
                "M4A encoder produced {media_duration} media frames, fewer than the {required_media_duration} required for its delay and presentation"
            ));
        }
        let mvhd = patch_duration_box(&self.mvhd, b"mvhd", Some(sample_rate), presentation_frames)?;
        let tkhd = patch_duration_box(&self.tkhd, b"tkhd", None, presentation_frames)?;
        let mdhd = patch_duration_box(&self.mdhd, b"mdhd", Some(sample_rate), media_duration)?;
        let stsd = patch_stsd_buffer_size(&self.stsd, max_sample_size)?;
        let elst_size = if presentation_frames <= u64::from(u32::MAX) {
            28_u64
        } else {
            36_u64
        };
        let edts_size = checked_box_size(&[elst_size])?;
        let stts_size = 24_u64;
        let stsc_size = 28_u64;
        let stsz_size = 20_u64
            .checked_add(u64::from(sample_count) * 4)
            .ok_or_else(|| "M4A stsz size overflows".to_string())?;
        let use_stco = max_chunk_offset <= u64::from(u32::MAX);
        let chunk_offset_size = 16_u64
            .checked_add(
                u64::from(sample_count)
                    .checked_mul(if use_stco { 4 } else { 8 })
                    .ok_or_else(|| "M4A chunk-offset table size overflows".to_string())?,
            )
            .ok_or_else(|| "M4A chunk-offset table size overflows".to_string())?;
        let stbl_size = checked_box_size(&[
            stsd.len() as u64,
            stts_size,
            stsc_size,
            stsz_size,
            chunk_offset_size,
        ])?;
        let minf_size =
            checked_box_size(&[self.smhd.len() as u64, self.dinf.len() as u64, stbl_size])?;
        let mdia_size = checked_box_size(&[mdhd.len() as u64, self.hdlr.len() as u64, minf_size])?;
        let trak_size = checked_box_size(&[tkhd.len() as u64, edts_size, mdia_size])?;
        let moov_size = checked_box_size(&[mvhd.len() as u64, trak_size])?;

        write_box_header(output, moov_size, b"moov")?;
        output
            .write_all(&mvhd)
            .map_err(|error| format!("write M4A mvhd: {error}"))?;
        write_box_header(output, trak_size, b"trak")?;
        output
            .write_all(&tkhd)
            .map_err(|error| format!("write M4A tkhd: {error}"))?;
        write_box_header(output, edts_size, b"edts")?;
        write_elst(output, presentation_frames, encoder_delay, elst_size)?;
        write_box_header(output, mdia_size, b"mdia")?;
        output
            .write_all(&mdhd)
            .and_then(|_| output.write_all(&self.hdlr))
            .map_err(|error| format!("write M4A media header: {error}"))?;
        write_box_header(output, minf_size, b"minf")?;
        output
            .write_all(&self.smhd)
            .and_then(|_| output.write_all(&self.dinf))
            .map_err(|error| format!("write M4A media info: {error}"))?;
        write_box_header(output, stbl_size, b"stbl")?;
        output
            .write_all(&stsd)
            .map_err(|error| format!("write M4A sample description: {error}"))?;
        write_stts(output, sample_count, sample_duration)?;
        write_stsc(output)?;
        write_stsz(output, table, sample_count, stsz_size)?;
        table
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind M4A offset table: {error}"))?;
        if use_stco {
            write_stco(output, table, sample_count, chunk_offset_size)?;
        } else {
            write_co64(output, table, sample_count, chunk_offset_size)?;
        }
        Ok(())
    }
}

fn patch_stsd_buffer_size(template: &[u8], max_sample_size: u32) -> Result<Vec<u8>, String> {
    if max_sample_size > 0x00ff_ffff {
        return Err("M4A decoder buffer size exceeds its 24-bit field".into());
    }
    let (_, stsd_header) = parse_box_header(template)?;
    let entries_start = stsd_header
        .checked_add(8)
        .ok_or_else(|| "M4A stsd entry offset overflows".to_string())?;
    if entries_start > template.len() {
        return Err("truncated bounded M4A stsd template".into());
    }
    let entries = &template[entries_start..];
    let mp4a = find_box_sequence(entries, b"mp4a")?;
    let mp4a_start = entries_start + slice_offset(entries, mp4a)?;
    let (_, mp4a_header) = parse_box_header(mp4a)?;
    let children_start = mp4a_header
        .checked_add(28)
        .ok_or_else(|| "M4A mp4a child offset overflows".to_string())?;
    if children_start > mp4a.len() {
        return Err("truncated bounded M4A mp4a template".into());
    }
    let children = &mp4a[children_start..];
    let esds = find_box_sequence(children, b"esds")?;
    let esds_start = mp4a_start + children_start + slice_offset(children, esds)?;
    let (_, esds_header) = parse_box_header(esds)?;
    let descriptor_start = esds_header
        .checked_add(4)
        .ok_or_else(|| "M4A esds descriptor offset overflows".to_string())?;
    let (es_tag, es_payload, es_end) = parse_descriptor_header(esds, descriptor_start)?;
    if es_tag != 0x03 || es_payload.checked_add(3).is_none_or(|next| next > es_end) {
        return Err("invalid bounded M4A ES descriptor".into());
    }
    let (decoder_tag, decoder_payload, decoder_end) =
        parse_descriptor_header(esds, es_payload + 3)?;
    if decoder_tag != 0x04
        || decoder_payload
            .checked_add(5)
            .is_none_or(|next| next > decoder_end)
    {
        return Err("invalid bounded M4A decoder descriptor".into());
    }
    let buffer_offset = esds_start
        .checked_add(decoder_payload)
        .and_then(|offset| offset.checked_add(2))
        .ok_or_else(|| "M4A decoder buffer offset overflows".to_string())?;
    let mut output = template.to_vec();
    let buffer = output
        .get_mut(buffer_offset..buffer_offset + 3)
        .ok_or_else(|| "M4A decoder buffer field is truncated".to_string())?;
    let bytes = max_sample_size.to_be_bytes();
    buffer.copy_from_slice(&bytes[1..]);
    Ok(output)
}

fn parse_descriptor_header(bytes: &[u8], start: usize) -> Result<(u8, usize, usize), String> {
    let tag = *bytes
        .get(start)
        .ok_or_else(|| "truncated bounded M4A descriptor tag".to_string())?;
    let mut position = start + 1;
    let mut size = 0usize;
    let mut ended = false;
    for _ in 0..4 {
        let byte = *bytes
            .get(position)
            .ok_or_else(|| "truncated bounded M4A descriptor length".to_string())?;
        position += 1;
        size = size
            .checked_shl(7)
            .and_then(|size| size.checked_add(usize::from(byte & 0x7f)))
            .ok_or_else(|| "M4A descriptor length overflows".to_string())?;
        if byte & 0x80 == 0 {
            ended = true;
            break;
        }
    }
    if !ended {
        return Err("invalid bounded M4A descriptor length".into());
    }
    let end = position
        .checked_add(size)
        .ok_or_else(|| "M4A descriptor end overflows".to_string())?;
    if end > bytes.len() {
        return Err("bounded M4A descriptor extends beyond its box".into());
    }
    Ok((tag, position, end))
}

fn slice_offset(parent: &[u8], child: &[u8]) -> Result<usize, String> {
    let parent_start = parent.as_ptr() as usize;
    let child_start = child.as_ptr() as usize;
    child_start
        .checked_sub(parent_start)
        .filter(|offset| *offset <= parent.len())
        .ok_or_else(|| "bounded M4A child does not belong to its parent".to_string())
}

fn write_elst<W: Write>(
    output: &mut W,
    presentation_frames: u64,
    encoder_delay: u64,
    size: u64,
) -> Result<(), String> {
    write_box_header(output, size, b"elst")?;
    let version = if presentation_frames <= u64::from(u32::MAX) {
        0_u8
    } else {
        1_u8
    };
    output
        .write_all(&[version, 0, 0, 0])
        .and_then(|_| output.write_all(&1_u32.to_be_bytes()))
        .map_err(|error| format!("write M4A edit-list header: {error}"))?;
    if version == 0 {
        let encoder_delay = i32::try_from(encoder_delay)
            .map_err(|_| "M4A encoder delay exceeds the version-0 edit field".to_string())?;
        output
            .write_all(&(presentation_frames as u32).to_be_bytes())
            .and_then(|_| output.write_all(&encoder_delay.to_be_bytes()))
            .map_err(|error| format!("write M4A version-0 edit: {error}"))?;
    } else {
        let encoder_delay = i64::try_from(encoder_delay)
            .map_err(|_| "M4A encoder delay exceeds the version-1 edit field".to_string())?;
        output
            .write_all(&presentation_frames.to_be_bytes())
            .and_then(|_| output.write_all(&encoder_delay.to_be_bytes()))
            .map_err(|error| format!("write M4A version-1 edit: {error}"))?;
    }
    output
        .write_all(&1_i16.to_be_bytes())
        .and_then(|_| output.write_all(&0_i16.to_be_bytes()))
        .map_err(|error| format!("write M4A edit rate: {error}"))
}

fn checked_box_size(children: &[u64]) -> Result<u64, String> {
    let size = children
        .iter()
        .try_fold(MP4_BOX_HEADER_BYTES, |size, child| size.checked_add(*child));
    let size = size.ok_or_else(|| "M4A metadata box size overflows".to_string())?;
    if size > u64::from(u32::MAX) {
        return Err("M4A metadata box exceeds the 32-bit bounded muxer limit".into());
    }
    Ok(size)
}

fn write_box_header<W: Write>(output: &mut W, size: u64, kind: &[u8; 4]) -> Result<(), String> {
    let size = u32::try_from(size)
        .map_err(|_| "M4A metadata box exceeds the 32-bit bounded muxer limit".to_string())?;
    output
        .write_all(&size.to_be_bytes())
        .and_then(|_| output.write_all(kind))
        .map_err(|error| format!("write M4A {} box header: {error}", fourcc(kind)))
}

fn write_full_box_header<W: Write>(
    output: &mut W,
    size: u64,
    kind: &[u8; 4],
) -> Result<(), String> {
    write_box_header(output, size, kind)?;
    output
        .write_all(&[0; MP4_FULL_BOX_HEADER_BYTES as usize])
        .map_err(|error| format!("write M4A {} full-box header: {error}", fourcc(kind)))
}

fn write_stts<W: Write>(
    output: &mut W,
    sample_count: u32,
    sample_duration: u32,
) -> Result<(), String> {
    write_full_box_header(output, 24, b"stts")?;
    output
        .write_all(&1_u32.to_be_bytes())
        .and_then(|_| output.write_all(&sample_count.to_be_bytes()))
        .and_then(|_| output.write_all(&sample_duration.to_be_bytes()))
        .map_err(|error| format!("write M4A time-to-sample table: {error}"))
}

fn write_stsc<W: Write>(output: &mut W) -> Result<(), String> {
    write_full_box_header(output, 28, b"stsc")?;
    for value in [1_u32, 1, 1, 1] {
        output
            .write_all(&value.to_be_bytes())
            .map_err(|error| format!("write M4A sample-to-chunk table: {error}"))?;
    }
    Ok(())
}

fn write_stsz<W: Write>(
    output: &mut W,
    table: &mut File,
    sample_count: u32,
    size: u64,
) -> Result<(), String> {
    write_full_box_header(output, size, b"stsz")?;
    output
        .write_all(&0_u32.to_be_bytes())
        .and_then(|_| output.write_all(&sample_count.to_be_bytes()))
        .map_err(|error| format!("write M4A sample-size header: {error}"))?;
    let mut record = [0_u8; TABLE_RECORD_BYTES as usize];
    for _ in 0..sample_count {
        table
            .read_exact(&mut record)
            .map_err(|error| format!("read bounded M4A sample size: {error}"))?;
        output
            .write_all(&record[..4])
            .map_err(|error| format!("write M4A sample size: {error}"))?;
    }
    Ok(())
}

fn write_co64<W: Write>(
    output: &mut W,
    table: &mut File,
    sample_count: u32,
    size: u64,
) -> Result<(), String> {
    write_full_box_header(output, size, b"co64")?;
    output
        .write_all(&sample_count.to_be_bytes())
        .map_err(|error| format!("write M4A chunk-offset header: {error}"))?;
    let mut record = [0_u8; TABLE_RECORD_BYTES as usize];
    for _ in 0..sample_count {
        table
            .read_exact(&mut record)
            .map_err(|error| format!("read bounded M4A chunk offset: {error}"))?;
        output
            .write_all(&record[4..])
            .map_err(|error| format!("write M4A chunk offset: {error}"))?;
    }
    Ok(())
}

fn write_stco<W: Write>(
    output: &mut W,
    table: &mut File,
    sample_count: u32,
    size: u64,
) -> Result<(), String> {
    write_full_box_header(output, size, b"stco")?;
    output
        .write_all(&sample_count.to_be_bytes())
        .map_err(|error| format!("write M4A chunk-offset header: {error}"))?;
    let mut record = [0_u8; TABLE_RECORD_BYTES as usize];
    for _ in 0..sample_count {
        table
            .read_exact(&mut record)
            .map_err(|error| format!("read bounded M4A sample offset: {error}"))?;
        let offset = u64::from_be_bytes(record[4..].try_into().expect("eight-byte offset"));
        let offset = u32::try_from(offset)
            .map_err(|_| "M4A sample offset exceeds the 32-bit stco field".to_string())?;
        output
            .write_all(&offset.to_be_bytes())
            .map_err(|error| format!("write M4A sample offset: {error}"))?;
    }
    Ok(())
}

fn patch_duration_box(
    template: &[u8],
    kind: &[u8; 4],
    timescale: Option<u32>,
    duration: u64,
) -> Result<Vec<u8>, String> {
    if template.len() < 32 || &template[4..8] != kind || template[8] != 0 {
        return Err(format!("invalid bounded M4A {} template", fourcc(kind)));
    }
    if duration <= u64::from(u32::MAX) {
        let mut output = template.to_vec();
        match kind {
            b"mvhd" | b"mdhd" => {
                output[20..24].copy_from_slice(
                    &timescale
                        .ok_or_else(|| format!("{} timescale is missing", fourcc(kind)))?
                        .to_be_bytes(),
                );
                output[24..28].copy_from_slice(&(duration as u32).to_be_bytes());
            }
            b"tkhd" => output[28..32].copy_from_slice(&(duration as u32).to_be_bytes()),
            _ => return Err("unsupported M4A duration template".into()),
        }
        return Ok(output);
    }

    let tail_offset = match kind {
        b"mvhd" | b"mdhd" => 28,
        b"tkhd" => 32,
        _ => return Err("unsupported M4A duration template".into()),
    };
    let new_size = template
        .len()
        .checked_add(12)
        .ok_or_else(|| "M4A version-1 header size overflows".to_string())?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(new_size)
        .map_err(|error| format!("reserve M4A version-1 header: {error}"))?;
    output.extend_from_slice(&(new_size as u32).to_be_bytes());
    output.extend_from_slice(kind);
    output.push(1);
    output.extend_from_slice(&template[9..12]);
    output.extend_from_slice(&0_u64.to_be_bytes());
    output.extend_from_slice(&0_u64.to_be_bytes());
    match kind {
        b"mvhd" | b"mdhd" => {
            output.extend_from_slice(
                &timescale
                    .ok_or_else(|| format!("{} timescale is missing", fourcc(kind)))?
                    .to_be_bytes(),
            );
            output.extend_from_slice(&duration.to_be_bytes());
        }
        b"tkhd" => {
            output.extend_from_slice(&template[20..24]);
            output.extend_from_slice(&0_u32.to_be_bytes());
            output.extend_from_slice(&duration.to_be_bytes());
        }
        _ => unreachable!(),
    }
    output.extend_from_slice(&template[tail_offset..]);
    if output.len() != new_size {
        return Err(format!(
            "M4A {} version-1 header size changed unexpectedly",
            fourcc(kind)
        ));
    }
    Ok(output)
}

fn find_box<'a>(bytes: &'a [u8], kind: &[u8; 4]) -> Result<&'a [u8], String> {
    find_box_sequence(bytes, kind)
}

fn find_child<'a>(parent: &'a [u8], kind: &[u8; 4]) -> Result<&'a [u8], String> {
    let (_, header_size) = parse_box_header(parent)?;
    find_box_sequence(&parent[header_size..], kind)
}

fn find_box_sequence<'a>(mut bytes: &'a [u8], kind: &[u8; 4]) -> Result<&'a [u8], String> {
    while !bytes.is_empty() {
        let (size, _) = parse_box_header(bytes)?;
        let size =
            usize::try_from(size).map_err(|_| "M4A template box is too large".to_string())?;
        if size > bytes.len() {
            return Err("M4A template box extends beyond its parent".into());
        }
        if &bytes[4..8] == kind {
            return Ok(&bytes[..size]);
        }
        bytes = &bytes[size..];
    }
    Err(format!(
        "bounded M4A template is missing its {} box",
        fourcc(kind)
    ))
}

fn parse_box_header(bytes: &[u8]) -> Result<(u64, usize), String> {
    if bytes.len() < 8 {
        return Err("truncated bounded M4A template box header".into());
    }
    let size32 = u32::from_be_bytes(bytes[..4].try_into().expect("four-byte size"));
    match size32 {
        0 => Ok((bytes.len() as u64, 8)),
        1 => {
            if bytes.len() < 16 {
                return Err("truncated bounded M4A large box header".into());
            }
            let size = u64::from_be_bytes(bytes[8..16].try_into().expect("eight-byte size"));
            if size < 16 {
                return Err("invalid bounded M4A large box size".into());
            }
            Ok((size, 16))
        }
        size if size < 8 => Err("invalid bounded M4A box size".into()),
        size => Ok((u64::from(size), 8)),
    }
}

fn fourcc(kind: &[u8; 4]) -> String {
    String::from_utf8_lossy(kind).into_owned()
}

/// Write planar `f64` audio to an M4A file.
pub fn write_m4a<P: AsRef<Path>>(path: P, audio: &Audio, bitrate_bps: u32) -> Result<(), String> {
    write_m4a_with_downmix(path, audio, bitrate_bps, DownmixMode::Preserve)
}

/// Write planar `f64` audio to M4A with an explicit surround downmix policy.
pub fn write_m4a_with_downmix<P: AsRef<Path>>(
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
    .validate_config(OutputFormat::M4a, audio)?;
    let mut output = AtomicOutput::new(path)?;
    write_m4a_to_writer(output.file_mut(), audio, bitrate_bps, downmix)?;
    output.commit(CommitMode::Replace)
}

pub(super) fn write_m4a_to_writer<W: Write + Seek>(
    output: W,
    audio: &Audio,
    bitrate_bps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    let mut writer = M4aStreamWriter::new(
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

pub(super) fn sample_rate_to_index(sr: u32) -> Result<SampleFreqIndex, String> {
    match sr {
        96000 => Ok(SampleFreqIndex::Freq96000),
        88200 => Ok(SampleFreqIndex::Freq88200),
        64000 => Ok(SampleFreqIndex::Freq64000),
        48000 => Ok(SampleFreqIndex::Freq48000),
        44100 => Ok(SampleFreqIndex::Freq44100),
        32000 => Ok(SampleFreqIndex::Freq32000),
        24000 => Ok(SampleFreqIndex::Freq24000),
        22050 => Ok(SampleFreqIndex::Freq22050),
        16000 => Ok(SampleFreqIndex::Freq16000),
        12000 => Ok(SampleFreqIndex::Freq12000),
        11025 => Ok(SampleFreqIndex::Freq11025),
        8000 => Ok(SampleFreqIndex::Freq8000),
        7350 => Ok(SampleFreqIndex::Freq7350),
        _ => Err(format!(
            "M4A encode: unsupported sample rate {sr} Hz (AAC standard rates only)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn sine_mono(sr: u32, secs: f32) -> Audio {
        let frames = (sr as f32 * secs) as usize;
        let mut ch = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f64 / sr as f64;
            ch.push((2.0 * std::f64::consts::PI * 330.0 * t).sin() * 0.3);
        }
        Audio {
            sample_rate: sr,
            channels: vec![ch],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("denoize_m4a_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn m4a_roundtrip_decode() {
        let path = tmp("rt.m4a");
        let audio = sine_mono(44100, 0.5);
        write_m4a(&path, &audio, 128_000).unwrap();
        assert!(path.metadata().unwrap().len() > 100);

        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.frames(), audio.frames());

        let rms_out: f64 =
            decoded.channels[0].iter().map(|s| s * s).sum::<f64>() / decoded.frames() as f64;
        assert!(rms_out > 0.005);

        let leading_rms = decoded.channels[0][..1024]
            .iter()
            .map(|sample| sample * sample)
            .sum::<f64>()
            / 1024.0;
        assert!(leading_rms > 0.005, "encoder priming was not trimmed");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sample_table_spool_limit_is_enforced_before_the_next_record() {
        let audio = sine_mono(44_100, 0.01);
        let mut output = Cursor::new(Vec::new());
        let mut writer = M4aStreamWriter::new(
            &mut output,
            audio.sample_rate,
            audio.channels(),
            audio.channel_mask,
            128_000,
            DownmixMode::Preserve,
            Some(TABLE_RECORD_BYTES),
        )
        .unwrap();
        writer.write_block(&audio.channels).unwrap();
        let error = writer.finalize().unwrap_err();
        assert!(error.contains("sample table requires 24 bytes"), "{error}");
    }
}
