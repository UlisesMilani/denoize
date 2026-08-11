//! M4A / MP4-AAC decoder — `mp4` demux + Pure-Rust `oxideav-aac` AAC-LC decode.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use mp4::{ChannelConfig, MediaType, Mp4Reader, Mp4Track, TrackType};
use oxideav_aac::decode::{DecodedFrame, StreamDecoder};

use super::pcm::DecodedPcm;

/// `bufferSizeDB` in MPEG-4 systems descriptors is 24 bits. Keeping access
/// units within that representable range also prevents a corrupt `stsz` from
/// turning one tiny input into a multi-gigabyte allocation.
const MAX_AAC_ACCESS_UNIT_SIZE: u32 = 0x00ff_ffff;
const MAX_MP4_BOX_DEPTH: usize = 32;
const MAX_EMPTY_TRUN_SAMPLES: u64 = 10_000_000;

#[derive(Debug)]
struct ScanBudget {
    empty_trun_samples: u64,
}

impl Default for ScanBudget {
    fn default() -> Self {
        Self {
            empty_trun_samples: MAX_EMPTY_TRUN_SAMPLES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoxListKind {
    Top,
    Moov,
    Trak,
    Mdia,
    Minf,
    Dinf,
    Dref,
    Stbl,
    Stsd,
    Edts,
    Udta,
    Meta,
    Ilst,
    IlstItem,
    Mvex,
    Moof,
    Traf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleEntryKind {
    Mp4a,
    Avc,
    Hevc,
    Vp9,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RawBoxHeader {
    start: u64,
    body_start: u64,
    end: u64,
    name: [u8; 4],
    extends_to_end: bool,
}

impl RawBoxHeader {
    fn body_size(self) -> u64 {
        self.end - self.body_start
    }
}

/// Validate the box boundaries that `mp4 0.14` assumes before giving it an
/// untrusted file. The dependency's nested container loops accept zero-sized
/// children and seek back to the same header, and several count-bearing boxes
/// reserve from an unchecked count before reaching EOF. This bounded walker
/// skips media payloads, but proves forward progress, parent containment, and
/// allocation-count bounds for every box family the M4A path asks `mp4` to
/// parse.
pub(super) fn validate_mp4_structure<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
) -> Result<(), String> {
    if file_size < 8 {
        return Err("file is too short for an MP4 box header".to_string());
    }
    scan_box_list(reader, 0, file_size, BoxListKind::Top, 0).map(|_| ())
}

fn scan_box_list<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    end: u64,
    kind: BoxListKind,
    depth: usize,
) -> Result<u32, String> {
    let mut budget = ScanBudget::default();
    scan_box_list_with_budget(reader, start, end, kind, depth, &mut budget)
}

fn scan_box_list_with_budget<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    end: u64,
    kind: BoxListKind,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<u32, String> {
    if depth > MAX_MP4_BOX_DEPTH {
        return Err(format!("MP4 box nesting exceeds {MAX_MP4_BOX_DEPTH}"));
    }
    if start > end {
        return Err("MP4 child range starts after its parent".to_string());
    }

    let mut cursor = start;
    let mut count = 0u32;
    while cursor < end {
        let header = read_raw_box_header(reader, cursor, end, kind == BoxListKind::Top)?;
        validate_raw_box(reader, header, kind, depth, budget)?;
        if header.end <= cursor {
            return Err(format!(
                "MP4 box {:?} at byte {cursor} made no forward progress",
                header.name
            ));
        }
        cursor = header.end;
        count = count
            .checked_add(1)
            .ok_or_else(|| "MP4 box count overflow".to_string())?;
    }
    if cursor != end {
        return Err(format!(
            "MP4 child boxes end at byte {cursor}, expected parent end {end}"
        ));
    }
    Ok(count)
}

fn read_raw_box_header<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    parent_end: u64,
    top_level: bool,
) -> Result<RawBoxHeader, String> {
    let header_end = start
        .checked_add(8)
        .ok_or_else(|| "MP4 box header offset overflow".to_string())?;
    if header_end > parent_end {
        return Err(format!(
            "truncated MP4 box header at byte {start} within parent ending at {parent_end}"
        ));
    }

    let mut bytes = [0u8; 8];
    read_exact_at(reader, start, &mut bytes)?;
    let size32 = u32::from_be_bytes(bytes[..4].try_into().unwrap());
    let name: [u8; 4] = bytes[4..].try_into().unwrap();
    let (box_size, body_start) = match size32 {
        0 => {
            if !top_level {
                return Err(format!(
                    "zero-sized MP4 box {} is only supported as a final top-level box",
                    fourcc_text(name)
                ));
            }
            (parent_end - start, header_end)
        }
        1 => {
            let large_header_end = start
                .checked_add(16)
                .ok_or_else(|| "large MP4 box header offset overflow".to_string())?;
            if large_header_end > parent_end {
                return Err(format!("truncated large MP4 box header at byte {start}"));
            }
            let mut large = [0u8; 8];
            read_exact_at(reader, header_end, &mut large)?;
            let size = u64::from_be_bytes(large);
            if size < 16 {
                return Err(format!(
                    "large MP4 box {} at byte {start} has invalid size {size}",
                    fourcc_text(name)
                ));
            }
            (size, large_header_end)
        }
        size => {
            if size < 8 {
                return Err(format!(
                    "MP4 box {} at byte {start} is shorter than its header ({size})",
                    fourcc_text(name)
                ));
            }
            (u64::from(size), header_end)
        }
    };

    let end = start.checked_add(box_size).ok_or_else(|| {
        format!(
            "MP4 box {} at byte {start} overflows its end offset",
            fourcc_text(name)
        )
    })?;
    if end > parent_end {
        return Err(format!(
            "MP4 box {} range {start}..{end} exceeds parent end {parent_end}",
            fourcc_text(name)
        ));
    }
    Ok(RawBoxHeader {
        start,
        body_start,
        end,
        name,
        extends_to_end: size32 == 0,
    })
}

fn validate_raw_box<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    parent: BoxListKind,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    // `mp4 0.14` deliberately stops at any top-level size-zero header without
    // parsing its payload. Match that compatibility behavior while continuing
    // to reject size-zero children, which make its container loops stall.
    if header.extends_to_end {
        return Ok(());
    }

    match parent {
        BoxListKind::Top => match &header.name {
            b"moov" => scan_children(reader, header, BoxListKind::Moov, depth, budget),
            b"moof" => scan_children(reader, header, BoxListKind::Moof, depth, budget),
            b"emsg" => validate_emsg(reader, header),
            b"ftyp" => require_body_size(header, 8),
            _ => Ok(()),
        },
        BoxListKind::Moov => match &header.name {
            b"mvhd" => validate_versioned_body(reader, header, 100, 112),
            b"trak" => scan_children(reader, header, BoxListKind::Trak, depth, budget),
            b"meta" => scan_meta_children(reader, header, depth, budget),
            b"mvex" => scan_children(reader, header, BoxListKind::Mvex, depth, budget),
            b"udta" => scan_children(reader, header, BoxListKind::Udta, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Trak => match &header.name {
            b"tkhd" => validate_versioned_body(reader, header, 84, 96),
            b"edts" => scan_required_children(reader, header, BoxListKind::Edts, depth, budget),
            b"meta" => scan_meta_children(reader, header, depth, budget),
            b"mdia" => scan_children(reader, header, BoxListKind::Mdia, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Mdia => match &header.name {
            b"mdhd" => validate_versioned_body(reader, header, 24, 36),
            b"hdlr" => validate_hdlr(reader, header),
            b"minf" => scan_children(reader, header, BoxListKind::Minf, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Minf => match &header.name {
            b"smhd" => require_body_size(header, 8),
            b"vmhd" => require_body_size(header, 12),
            b"dinf" => scan_children(reader, header, BoxListKind::Dinf, depth, budget),
            b"stbl" => scan_children(reader, header, BoxListKind::Stbl, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Dinf => match &header.name {
            b"dref" => scan_counted_children(reader, header, BoxListKind::Dref, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Dref => match &header.name {
            b"url " => require_body_size(header, 4),
            _ => Ok(()),
        },
        BoxListKind::Stbl => match &header.name {
            b"stsd" => scan_stsd_entries(reader, header, depth, budget),
            b"stts" | b"ctts" => validate_counted_leaf(reader, header, 8, 8),
            b"stsc" => validate_counted_leaf(reader, header, 12, 8),
            b"stss" | b"stco" => validate_counted_leaf(reader, header, 4, 8),
            b"co64" => validate_counted_leaf(reader, header, 8, 8),
            b"stsz" => validate_stsz(reader, header),
            _ => Ok(()),
        },
        BoxListKind::Stsd => match &header.name {
            b"mp4a" => validate_sample_entry(reader, header, 28, SampleEntryKind::Mp4a),
            b"avc1" => validate_sample_entry(reader, header, 78, SampleEntryKind::Avc),
            b"hev1" => validate_sample_entry(reader, header, 78, SampleEntryKind::Hevc),
            b"vp09" => validate_sample_entry(reader, header, 78, SampleEntryKind::Vp9),
            b"tx3g" => require_body_size(header, 38),
            _ => Ok(()),
        },
        BoxListKind::Edts => match &header.name {
            b"elst" => validate_elst(reader, header),
            _ => Ok(()),
        },
        BoxListKind::Udta => match &header.name {
            b"meta" => scan_meta_children(reader, header, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Meta => match &header.name {
            b"hdlr" => validate_hdlr(reader, header),
            b"ilst" => scan_children(reader, header, BoxListKind::Ilst, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Ilst => match &header.name {
            b"\xa9nam" | b"\xa9day" | b"covr" | b"desc" => {
                scan_children(reader, header, BoxListKind::IlstItem, depth, budget)
            }
            _ => Ok(()),
        },
        BoxListKind::IlstItem => match &header.name {
            b"data" => require_body_size(header, 8),
            _ => Ok(()),
        },
        BoxListKind::Mvex => match &header.name {
            b"mehd" => validate_versioned_body(reader, header, 8, 12),
            b"trex" => require_body_size(header, 24),
            _ => Ok(()),
        },
        BoxListKind::Moof => match &header.name {
            b"mfhd" => require_body_size(header, 8),
            b"traf" => scan_children(reader, header, BoxListKind::Traf, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Traf => match &header.name {
            b"tfhd" => validate_tfhd(reader, header),
            b"tfdt" => validate_tfdt(reader, header),
            b"trun" => validate_trun(reader, header, budget),
            _ => Ok(()),
        },
    }
}

fn scan_children<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    kind: BoxListKind,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    scan_box_list_with_budget(
        reader,
        header.body_start,
        header.end,
        kind,
        depth + 1,
        budget,
    )
    .map(|_| ())
}

fn scan_required_children<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    kind: BoxListKind,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    require_body_size(header, 8)?;
    let count = scan_box_list_with_budget(
        reader,
        header.body_start,
        header.end,
        kind,
        depth + 1,
        budget,
    )?;
    if count == 0 {
        return Err(format!(
            "MP4 box {} requires at least one child",
            fourcc_text(header.name)
        ));
    }
    Ok(())
}

fn scan_counted_children<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    kind: BoxListKind,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    require_body_size(header, 8)?;
    let expected = read_u32_at(reader, header.body_start + 4)?;
    if kind == BoxListKind::Stsd && expected == 0 {
        return Err("stsd entry_count must be at least one".to_string());
    }
    let actual = scan_box_list_with_budget(
        reader,
        header.body_start + 8,
        header.end,
        kind,
        depth + 1,
        budget,
    )?;
    if actual != expected {
        return Err(format!(
            "MP4 box {} declares {expected} children but contains {actual}",
            fourcc_text(header.name)
        ));
    }
    Ok(())
}

fn scan_stsd_entries<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    require_body_size(header, 8)?;
    let expected = read_u32_at(reader, header.body_start + 4)?;
    if expected == 0 {
        return Err("stsd entry_count must be at least one".to_string());
    }

    let child_depth = depth
        .checked_add(1)
        .ok_or_else(|| "MP4 box nesting depth overflow".to_string())?;
    if child_depth > MAX_MP4_BOX_DEPTH {
        return Err(format!("MP4 box nesting exceeds {MAX_MP4_BOX_DEPTH}"));
    }

    let mut cursor = header.body_start + 8;
    let mut actual = 0u32;
    while cursor < header.end {
        let entry = read_raw_box_header(reader, cursor, header.end, false)?;
        // `mp4 0.14` parses only the first sample entry and then seeks to the
        // end of stsd. Validate every entry's raw boundary and forward
        // progress, but mirror that parser by interpreting only entry #1.
        // This keeps unused QuickTime AudioSampleEntry v1/v2 extensions from
        // being mistaken for child box headers.
        if actual == 0 {
            validate_raw_box(reader, entry, BoxListKind::Stsd, child_depth, budget)?;
        }
        if entry.end <= cursor {
            return Err(format!(
                "MP4 sample entry {:?} at byte {cursor} made no forward progress",
                entry.name
            ));
        }
        cursor = entry.end;
        actual = actual
            .checked_add(1)
            .ok_or_else(|| "MP4 sample-entry count overflow".to_string())?;
    }
    if cursor != header.end {
        return Err(format!(
            "MP4 sample entries end at byte {cursor}, expected stsd end {}",
            header.end
        ));
    }
    if actual != expected {
        return Err(format!(
            "MP4 box stsd declares {expected} children but contains {actual}"
        ));
    }
    Ok(())
}

fn validate_sample_entry<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    child_offset: u64,
    kind: SampleEntryKind,
) -> Result<(), String> {
    require_body_size(header, child_offset)?;
    let start = header
        .body_start
        .checked_add(child_offset)
        .ok_or_else(|| "MP4 sample-entry child offset overflow".to_string())?;
    if start == header.end {
        return match kind {
            SampleEntryKind::Mp4a => Ok(()),
            _ => Err(format!(
                "MP4 sample entry {} requires a codec configuration child",
                fourcc_text(header.name)
            )),
        };
    }

    let child = read_raw_box_header(reader, start, header.end, false)?;
    match kind {
        SampleEntryKind::Mp4a if child.name == *b"esds" => validate_esds(reader, child),
        SampleEntryKind::Mp4a => Ok(()),
        SampleEntryKind::Avc if child.name == *b"avcC" => validate_avcc(reader, child),
        SampleEntryKind::Avc => Err("avc1 sample entry does not begin with avcC".to_string()),
        SampleEntryKind::Hevc if child.name == *b"hvcC" => require_body_size(child, 1),
        SampleEntryKind::Hevc => Err("hev1 sample entry does not begin with hvcC".to_string()),
        // `mp4 0.14` does not check the first child FourCC before parsing its
        // body as VpccBox, so validate the same body layout regardless of name.
        SampleEntryKind::Vp9 => require_body_size(child, 11),
    }
}

fn scan_meta_children<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    require_body_size(header, 4)?;
    let extended = read_u32_at(reader, header.body_start)?;
    let child_start = if extended == 0 {
        header.body_start + 4
    } else {
        require_body_size(header, 8)?;
        let possible_hdlr = read_fourcc_at(reader, header.body_start + 4)?;
        if possible_hdlr != *b"hdlr" {
            return Err(format!(
                "MP4 meta box at byte {} has unsupported version/flags {extended:#010x}",
                header.start
            ));
        }
        header.body_start
    };
    scan_box_list_with_budget(
        reader,
        child_start,
        header.end,
        BoxListKind::Meta,
        depth + 1,
        budget,
    )
    .map(|_| ())
}

fn validate_counted_leaf<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    entry_size: u64,
    entries_offset: u64,
) -> Result<(), String> {
    require_body_size(header, entries_offset)?;
    let count = u64::from(read_u32_at(reader, header.body_start + 4)?);
    let required = entries_offset
        .checked_add(
            count
                .checked_mul(entry_size)
                .ok_or_else(|| "MP4 table byte count overflow".to_string())?,
        )
        .ok_or_else(|| "MP4 table size overflow".to_string())?;
    require_body_size(header, required)
}

fn validate_stsz<R: Read + Seek>(reader: &mut R, header: RawBoxHeader) -> Result<(), String> {
    require_body_size(header, 12)?;
    let sample_size = read_u32_at(reader, header.body_start + 4)?;
    if sample_size == 0 {
        let count = u64::from(read_u32_at(reader, header.body_start + 8)?);
        let required = 12u64
            .checked_add(
                count
                    .checked_mul(4)
                    .ok_or_else(|| "stsz byte count overflow".to_string())?,
            )
            .ok_or_else(|| "stsz size overflow".to_string())?;
        require_body_size(header, required)?;
    }
    Ok(())
}

fn validate_elst<R: Read + Seek>(reader: &mut R, header: RawBoxHeader) -> Result<(), String> {
    require_body_size(header, 8)?;
    let version = read_u8_at(reader, header.body_start)?;
    let entry_size = match version {
        0 => 12,
        1 => 20,
        _ => {
            return Err(format!(
                "unsupported MP4 edit-list version {version} at byte {}",
                header.start
            ))
        }
    };
    validate_counted_leaf(reader, header, entry_size, 8)
}

fn validate_avcc<R: Read + Seek>(reader: &mut R, header: RawBoxHeader) -> Result<(), String> {
    require_body_size(header, 7)?;
    let sps_count = read_u8_at(reader, header.body_start + 5)? & 0x1f;
    let mut cursor = header.body_start + 6;
    for _ in 0..sps_count {
        cursor = validate_nal_unit(reader, cursor, header.end, "avcC SPS")?;
    }
    require_range_end(cursor, 1, header.end, "avcC PPS count")?;
    let pps_count = read_u8_at(reader, cursor)?;
    cursor = cursor
        .checked_add(1)
        .ok_or_else(|| "avcC PPS count offset overflow".to_string())?;
    for _ in 0..pps_count {
        cursor = validate_nal_unit(reader, cursor, header.end, "avcC PPS")?;
    }
    Ok(())
}

fn validate_nal_unit<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    end: u64,
    context: &str,
) -> Result<u64, String> {
    let length_end = require_range_end(offset, 2, end, context)?;
    let length = u64::from(read_u16_at(reader, offset)?);
    require_range_end(length_end, length, end, context)
}

fn validate_esds<R: Read + Seek>(reader: &mut R, header: RawBoxHeader) -> Result<(), String> {
    require_body_size(header, 4)?;
    let mut cursor = header.body_start + 4;
    while cursor < header.end {
        let descriptor = read_descriptor_header(reader, cursor, header.end)?;
        if descriptor.tag != 0x03 {
            // This is exactly where `mp4 0.14` stops and reports that the
            // required ESDescriptor is absent.
            return Ok(());
        }
        cursor = validate_es_descriptor(reader, descriptor)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct DescriptorRange {
    tag: u8,
    payload_start: u64,
    declared_end: u64,
}

fn read_descriptor_header<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    parent_end: u64,
) -> Result<DescriptorRange, String> {
    require_range_end(offset, 1, parent_end, "descriptor tag")?;
    let tag = read_u8_at(reader, offset)?;
    let mut cursor = offset + 1;
    let mut size = 0u32;
    for _ in 0..4 {
        require_range_end(cursor, 1, parent_end, "descriptor length")?;
        let byte = read_u8_at(reader, cursor)?;
        cursor += 1;
        size = (size << 7) | u32::from(byte & 0x7f);
        if byte & 0x80 == 0 {
            break;
        }
    }
    let declared_end =
        require_range_end(cursor, u64::from(size), parent_end, "descriptor payload")?;
    Ok(DescriptorRange {
        tag,
        payload_start: cursor,
        declared_end,
    })
}

fn validate_es_descriptor<R: Read + Seek>(
    reader: &mut R,
    descriptor: DescriptorRange,
) -> Result<u64, String> {
    require_range_end(
        descriptor.payload_start,
        3,
        descriptor.declared_end,
        "ESDescriptor fixed fields",
    )?;
    let mut cursor = descriptor.payload_start + 3;
    while cursor < descriptor.declared_end {
        let child = read_descriptor_header(reader, cursor, descriptor.declared_end)?;
        cursor = match child.tag {
            0x04 => validate_decoder_config_descriptor(reader, child)?,
            // The pinned writer declares an SLConfigDescriptor length of zero
            // but the reader always consumes its one predefined byte.
            0x06 => require_range_end(
                child.payload_start,
                1,
                descriptor.declared_end,
                "SLConfigDescriptor",
            )?,
            _ => child.declared_end,
        };
    }
    Ok(cursor)
}

fn validate_decoder_config_descriptor<R: Read + Seek>(
    reader: &mut R,
    descriptor: DescriptorRange,
) -> Result<u64, String> {
    require_range_end(
        descriptor.payload_start,
        13,
        descriptor.declared_end,
        "DecoderConfigDescriptor fixed fields",
    )?;
    let mut cursor = descriptor.payload_start + 13;
    while cursor < descriptor.declared_end {
        let child = read_descriptor_header(reader, cursor, descriptor.declared_end)?;
        cursor = match child.tag {
            0x05 => validate_decoder_specific_descriptor(reader, child)?,
            _ => child.declared_end,
        };
    }
    Ok(cursor)
}

fn validate_decoder_specific_descriptor<R: Read + Seek>(
    reader: &mut R,
    descriptor: DescriptorRange,
) -> Result<u64, String> {
    let first_two_end = require_range_end(
        descriptor.payload_start,
        2,
        descriptor.declared_end,
        "DecoderSpecificDescriptor",
    )?;
    let byte_a = read_u8_at(reader, descriptor.payload_start)?;
    let byte_b = read_u8_at(reader, descriptor.payload_start + 1)?;
    let profile = byte_a >> 3;
    let extended_profile = profile == 31;
    let frequency_index = if extended_profile {
        (byte_b >> 1) & 0x0f
    } else {
        ((byte_a & 0x07) << 1) | (byte_b >> 7)
    };
    let extra = if frequency_index == 15 {
        3
    } else if extended_profile {
        1
    } else {
        0
    };
    require_range_end(
        first_two_end,
        extra,
        descriptor.declared_end,
        "DecoderSpecificDescriptor extension",
    )?;
    // The pinned reader needs only the AudioSpecificConfig prefix, but the
    // descriptor may legally carry additional codec-specific bytes. They are
    // still part of this descriptor, not sibling descriptor headers.
    Ok(descriptor.declared_end)
}

fn validate_emsg<R: Read + Seek>(reader: &mut R, header: RawBoxHeader) -> Result<(), String> {
    require_body_size(header, 4)?;
    let version = read_u8_at(reader, header.body_start)?;
    match version {
        0 => {
            let scheme_end = find_nul(reader, header.body_start + 4, header.end, "emsg scheme")?;
            let value_end = find_nul(reader, scheme_end, header.end, "emsg value")?;
            require_range_end(value_end, 16, header.end, "emsg version 0 fixed fields")?;
        }
        1 => {
            let strings_start = require_range_end(
                header.body_start,
                24,
                header.end,
                "emsg version 1 fixed fields",
            )?;
            let scheme_end = find_nul(reader, strings_start, header.end, "emsg scheme")?;
            find_nul(reader, scheme_end, header.end, "emsg value")?;
        }
        _ => {
            return Err(format!(
                "unsupported emsg version {version} at byte {}",
                header.start
            ));
        }
    }
    Ok(())
}

fn validate_tfhd<R: Read + Seek>(reader: &mut R, header: RawBoxHeader) -> Result<(), String> {
    require_body_size(header, 8)?;
    let (_, flags) = read_full_box_header_at(reader, header.body_start)?;
    let optional_size = if flags & 0x000001 != 0 { 8 } else { 0 }
        + if flags & 0x000002 != 0 { 4 } else { 0 }
        + if flags & 0x000008 != 0 { 4 } else { 0 }
        + if flags & 0x000010 != 0 { 4 } else { 0 }
        + if flags & 0x000020 != 0 { 4 } else { 0 };
    require_body_size(header, 8 + optional_size)
}

fn validate_tfdt<R: Read + Seek>(reader: &mut R, header: RawBoxHeader) -> Result<(), String> {
    require_body_size(header, 4)?;
    let version = read_u8_at(reader, header.body_start)?;
    match version {
        0 => require_body_size(header, 8),
        1 => require_body_size(header, 12),
        _ => Err(format!(
            "unsupported tfdt version {version} at byte {}",
            header.start
        )),
    }
}

fn validate_trun<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    require_body_size(header, 8)?;
    let (_, flags) = read_full_box_header_at(reader, header.body_start)?;
    let sample_count = u64::from(read_u32_at(reader, header.body_start + 4)?);
    let optional_size =
        if flags & 0x000001 != 0 { 4 } else { 0 } + if flags & 0x000004 != 0 { 4 } else { 0 };
    let entry_size = if flags & 0x000100 != 0 { 4 } else { 0 }
        + if flags & 0x000200 != 0 { 4 } else { 0 }
        + if flags & 0x000400 != 0 { 4 } else { 0 }
        + if flags & 0x000800 != 0 { 4 } else { 0 };

    // With no per-sample fields the dependency still loops `sample_count`
    // times. Charge every such run against one shared budget so many tiny
    // boxes cannot multiply parser work beyond a fixed bound.
    if entry_size == 0 {
        budget.empty_trun_samples = budget
            .empty_trun_samples
            .checked_sub(sample_count)
            .ok_or_else(|| {
                format!("aggregate empty trun sample_count exceeds {MAX_EMPTY_TRUN_SAMPLES}")
            })?;
    }
    let entries_size = sample_count
        .checked_mul(entry_size)
        .ok_or_else(|| "trun entry byte count overflow".to_string())?;
    let required = 8u64
        .checked_add(optional_size)
        .and_then(|size| size.checked_add(entries_size))
        .ok_or_else(|| "trun body size overflow".to_string())?;
    require_body_size(header, required)
}

fn find_nul<R: Read + Seek>(
    reader: &mut R,
    mut offset: u64,
    end: u64,
    context: &str,
) -> Result<u64, String> {
    let mut buffer = [0u8; 4096];
    while offset < end {
        let remaining = end - offset;
        let wanted = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| format!("{context} scan length cannot be represented"))?;
        read_exact_at(reader, offset, &mut buffer[..wanted])?;
        if let Some(index) = buffer[..wanted].iter().position(|byte| *byte == 0) {
            return offset
                .checked_add(index as u64 + 1)
                .ok_or_else(|| format!("{context} end offset overflow"));
        }
        offset = offset
            .checked_add(wanted as u64)
            .ok_or_else(|| format!("{context} scan offset overflow"))?;
    }
    Err(format!(
        "{context} is not NUL-terminated within its MP4 box"
    ))
}

fn require_range_end(start: u64, len: u64, end: u64, context: &str) -> Result<u64, String> {
    let required_end = start
        .checked_add(len)
        .ok_or_else(|| format!("{context} range overflow"))?;
    if required_end > end {
        return Err(format!("{context} exceeds its MP4 box boundary"));
    }
    Ok(required_end)
}

fn validate_versioned_body<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    version_zero_size: u64,
    version_one_size: u64,
) -> Result<(), String> {
    require_body_size(header, 1)?;
    let version = read_u8_at(reader, header.body_start)?;
    match version {
        0 => require_body_size(header, version_zero_size),
        1 => require_body_size(header, version_one_size),
        _ => Err(format!(
            "unsupported MP4 box {} version {version} at byte {}",
            fourcc_text(header.name),
            header.start
        )),
    }
}

fn validate_hdlr<R: Read + Seek>(reader: &mut R, header: RawBoxHeader) -> Result<(), String> {
    let _ = reader;
    // QuickTime permits a counted component-name string here, whereas ISO
    // metadata handlers commonly use NUL-terminated UTF-8. The dependency
    // consumes exactly the declared body in either form, so length is the
    // containment invariant and content must remain compatible with both.
    require_body_size(header, 25)
}

fn require_body_size(header: RawBoxHeader, minimum: u64) -> Result<(), String> {
    if header.body_size() < minimum {
        return Err(format!(
            "MP4 box {} at byte {} has body size {}, minimum is {minimum}",
            fourcc_text(header.name),
            header.start,
            header.body_size()
        ));
    }
    Ok(())
}

fn read_exact_at<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    bytes: &mut [u8],
) -> Result<(), String> {
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|e| format!("seek MP4 structure at byte {offset}: {e}"))?;
    reader
        .read_exact(bytes)
        .map_err(|e| format!("read MP4 structure at byte {offset}: {e}"))
}

fn read_u8_at<R: Read + Seek>(reader: &mut R, offset: u64) -> Result<u8, String> {
    let mut byte = [0u8; 1];
    read_exact_at(reader, offset, &mut byte)?;
    Ok(byte[0])
}

fn read_u16_at<R: Read + Seek>(reader: &mut R, offset: u64) -> Result<u16, String> {
    let mut bytes = [0u8; 2];
    read_exact_at(reader, offset, &mut bytes)?;
    Ok(u16::from_be_bytes(bytes))
}

fn read_u32_at<R: Read + Seek>(reader: &mut R, offset: u64) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    read_exact_at(reader, offset, &mut bytes)?;
    Ok(u32::from_be_bytes(bytes))
}

fn read_full_box_header_at<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
) -> Result<(u8, u32), String> {
    let mut bytes = [0u8; 4];
    read_exact_at(reader, offset, &mut bytes)?;
    Ok((
        bytes[0],
        u32::from_be_bytes([0, bytes[1], bytes[2], bytes[3]]),
    ))
}

fn read_fourcc_at<R: Read + Seek>(reader: &mut R, offset: u64) -> Result<[u8; 4], String> {
    let mut bytes = [0u8; 4];
    read_exact_at(reader, offset, &mut bytes)?;
    Ok(bytes)
}

fn fourcc_text(name: [u8; 4]) -> String {
    String::from_utf8_lossy(&name).into_owned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampleDescriptor {
    offset: u64,
    size: u32,
    index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StszFields {
    sample_size: u32,
    sample_count: u32,
    variable_size_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StscFields {
    first_chunk: u32,
    samples_per_chunk: u32,
    sample_description_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkOffsetsKind {
    Missing,
    Stco(usize),
    Co64(usize),
    Both,
}

/// Primitive-only view of the sample-table fields used by this decoder.
///
/// The `mp4` crate exposes `Mp4Track`, but the concrete sample-table box types
/// live in private modules. This interface lets the production path inspect
/// those public fields without naming their types, and keeps all arithmetic in
/// this module instead of relying on `mp4`'s derived `first_sample` values.
trait SampleTable {
    fn stsz_fields(&self) -> StszFields;
    fn variable_sample_size(&self, index: usize) -> Option<u32>;

    fn chunk_offsets_kind(&self) -> ChunkOffsetsKind;
    fn chunk_offset(&self, index: usize) -> Option<u64>;

    fn stsc_len(&self) -> usize;
    fn stsc_entry(&self, index: usize) -> Option<StscFields>;

    fn stts_len(&self) -> usize;
    fn stts_sample_count(&self, index: usize) -> Option<u32>;

    fn has_ctts(&self) -> bool;
    fn ctts_len(&self) -> usize;
    fn ctts_sample_count(&self, index: usize) -> Option<u32>;
}

impl SampleTable for Mp4Track {
    fn stsz_fields(&self) -> StszFields {
        let stsz = &self.trak.mdia.minf.stbl.stsz;
        StszFields {
            sample_size: stsz.sample_size,
            sample_count: stsz.sample_count,
            variable_size_count: stsz.sample_sizes.len(),
        }
    }

    fn variable_sample_size(&self, index: usize) -> Option<u32> {
        self.trak
            .mdia
            .minf
            .stbl
            .stsz
            .sample_sizes
            .get(index)
            .copied()
    }

    fn chunk_offsets_kind(&self) -> ChunkOffsetsKind {
        let stbl = &self.trak.mdia.minf.stbl;
        match (&stbl.stco, &stbl.co64) {
            (None, None) => ChunkOffsetsKind::Missing,
            (Some(stco), None) => ChunkOffsetsKind::Stco(stco.entries.len()),
            (None, Some(co64)) => ChunkOffsetsKind::Co64(co64.entries.len()),
            (Some(_), Some(_)) => ChunkOffsetsKind::Both,
        }
    }

    fn chunk_offset(&self, index: usize) -> Option<u64> {
        let stbl = &self.trak.mdia.minf.stbl;
        match (&stbl.stco, &stbl.co64) {
            (Some(stco), None) => stco.entries.get(index).copied().map(u64::from),
            (None, Some(co64)) => co64.entries.get(index).copied(),
            _ => None,
        }
    }

    fn stsc_len(&self) -> usize {
        self.trak.mdia.minf.stbl.stsc.entries.len()
    }

    fn stsc_entry(&self, index: usize) -> Option<StscFields> {
        self.trak
            .mdia
            .minf
            .stbl
            .stsc
            .entries
            .get(index)
            .map(|entry| StscFields {
                first_chunk: entry.first_chunk,
                samples_per_chunk: entry.samples_per_chunk,
                sample_description_index: entry.sample_description_index,
            })
    }

    fn stts_len(&self) -> usize {
        self.trak.mdia.minf.stbl.stts.entries.len()
    }

    fn stts_sample_count(&self, index: usize) -> Option<u32> {
        self.trak
            .mdia
            .minf
            .stbl
            .stts
            .entries
            .get(index)
            .map(|entry| entry.sample_count)
    }

    fn has_ctts(&self) -> bool {
        self.trak.mdia.minf.stbl.ctts.is_some()
    }

    fn ctts_len(&self) -> usize {
        self.trak
            .mdia
            .minf
            .stbl
            .ctts
            .as_ref()
            .map_or(0, |ctts| ctts.entries.len())
    }

    fn ctts_sample_count(&self, index: usize) -> Option<u32> {
        self.trak
            .mdia
            .minf
            .stbl
            .ctts
            .as_ref()?
            .entries
            .get(index)
            .map(|entry| entry.sample_count)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValidatedSampleTable {
    sample_count: u32,
}

/// Decode M4A/MP4-AAC from path.
pub fn decode_m4a(path: &Path) -> Result<DecodedPcm, String> {
    // Keep the original handle for payload reads. Parsing happens through a
    // clone because each access unit below is read with an absolute seek.
    let mut payload_reader = File::open(path).map_err(|e| format!("open m4a: {e}"))?;
    let file_size = payload_reader
        .metadata()
        .map_err(|e| format!("stat m4a: {e}"))?
        .len();
    let mut structure_file = payload_reader
        .try_clone()
        .map_err(|e| format!("clone m4a handle for structural validation: {e}"))?;
    validate_mp4_structure(&mut structure_file, file_size)
        .map_err(|e| format!("mp4 structure: {e}"))?;

    let mut header_file = payload_reader
        .try_clone()
        .map_err(|e| format!("clone m4a handle for header parsing: {e}"))?;
    header_file
        .seek(SeekFrom::Start(0))
        .map_err(|e| format!("rewind m4a header: {e}"))?;
    let mp4 = Mp4Reader::read_header(BufReader::new(header_file), file_size)
        .map_err(|e| format!("mp4 parse: {e}"))?;

    let track = select_aac_track(&mp4)?;
    if !track.trafs.is_empty() {
        return Err(format!(
            "fragmented AAC track {} is not supported; a regular M4A sample table is required",
            track.track_id()
        ));
    }

    let validated = validate_sample_table(track, file_size)
        .map_err(|e| format!("AAC track {} sample table: {e}", track.track_id()))?;

    let profile = track
        .audio_profile()
        .map_err(|e| format!("aac profile: {e}"))?;
    let freq_index = track
        .sample_freq_index()
        .map_err(|e| format!("aac sample rate: {e}"))?;
    let channel_config = track
        .channel_config()
        .map_err(|e| format!("aac channels: {e}"))?;

    let sample_rate = freq_index.freq();
    let aot = profile as u8;
    let fs_index = freq_index as u8;
    let chan_conf = channel_config as u8;
    let n_ch = channel_config_to_count(channel_config);

    let mut decoder = StreamDecoder::new();
    let mut channels = Vec::new();
    channels
        .try_reserve_exact(n_ch)
        .map_err(|e| format!("reserve M4A output channels: {e}"))?;
    channels.resize_with(n_ch, Vec::new);
    let mut access_unit = Vec::new();
    let mut decoded_frames = 0usize;

    visit_sample_descriptors(track, validated.sample_count, |descriptor| {
        // Zero-sized entries still participate in stsc accounting. They carry
        // no AAC payload, so retain the existing behavior of skipping decode.
        if descriptor.size == 0 {
            return Ok(());
        }

        let size = usize::try_from(descriptor.size).map_err(|_| {
            format!(
                "AAC sample {} size cannot be represented on this platform",
                descriptor.index
            )
        })?;
        access_unit.clear();
        access_unit.try_reserve_exact(size).map_err(|e| {
            format!(
                "reserve AAC sample {} ({} bytes): {e}",
                descriptor.index, descriptor.size
            )
        })?;
        access_unit.resize(size, 0);

        payload_reader
            .seek(SeekFrom::Start(descriptor.offset))
            .map_err(|e| format!("seek AAC sample {}: {e}", descriptor.index))?;
        payload_reader
            .read_exact(&mut access_unit)
            .map_err(|e| format!("read AAC sample {}: {e}", descriptor.index))?;

        let frame = decoder
            .decode_raw_data_block(aot, fs_index, sample_rate, chan_conf, 1, &access_unit)
            .map_err(|e| format!("decode AAC sample {}: {e}", descriptor.index))?;
        append_decoded_frame(
            &mut channels,
            &frame,
            n_ch,
            sample_rate,
            &mut decoded_frames,
        )
        .map_err(|e| format!("AAC sample {}: {e}", descriptor.index))
    })?;

    if decoded_frames == 0 {
        return Err("M4A decode produced no samples".into());
    }

    Ok(DecodedPcm {
        sample_rate,
        channels,
        channel_mask: crate::channel_layout::ChannelLayout::from_channel_count(n_ch).mask(),
    })
}

fn select_aac_track<R: Read + Seek>(mp4: &Mp4Reader<R>) -> Result<&Mp4Track, String> {
    // `Mp4Reader` stores tracks in a HashMap. Validate IDs and then follow the
    // original moov/trak ordering so selection is deterministic.
    if mp4.tracks().len() != mp4.moov.traks.len() {
        return Err("duplicate MP4 track IDs in moov".into());
    }

    for trak in &mp4.moov.traks {
        let track_id = trak.tkhd.track_id;
        let track = mp4
            .tracks()
            .get(&track_id)
            .ok_or_else(|| format!("MP4 track {track_id} metadata missing"))?;
        if track.track_type().ok() == Some(TrackType::Audio)
            && track.media_type().ok() == Some(MediaType::AAC)
        {
            return Ok(track);
        }
    }

    Err("no AAC audio track found in M4A/MP4".into())
}

fn validate_sample_table(track: &Mp4Track, file_size: u64) -> Result<ValidatedSampleTable, String> {
    validate_table(track, file_size)
}

/// Validate every table relationship and byte range before allocating an AAC
/// access-unit buffer. The success path of this pass performs no allocation.
fn validate_table<T: SampleTable + ?Sized>(
    table: &T,
    file_size: u64,
) -> Result<ValidatedSampleTable, String> {
    let stsz = table.stsz_fields();
    if stsz.sample_count == 0 {
        return Err("stsz declares no samples".into());
    }

    let sample_count = usize::try_from(stsz.sample_count)
        .map_err(|_| "stsz sample_count cannot be represented on this platform")?;
    if stsz.sample_size == 0 {
        if stsz.variable_size_count != sample_count {
            return Err(format!(
                "stsz variable-size entry count {} does not match sample_count {}",
                stsz.variable_size_count, stsz.sample_count
            ));
        }
    } else if stsz.variable_size_count != 0 {
        return Err(format!(
            "stsz fixed sample_size {} must not have {} variable-size entries",
            stsz.sample_size, stsz.variable_size_count
        ));
    }

    let chunk_count = match table.chunk_offsets_kind() {
        ChunkOffsetsKind::Missing => return Err("missing stco/co64 chunk-offset table".into()),
        ChunkOffsetsKind::Both => {
            return Err("both stco and co64 are present; exactly one is required".into())
        }
        ChunkOffsetsKind::Stco(len) | ChunkOffsetsKind::Co64(len) => len,
    };
    if chunk_count == 0 {
        return Err("chunk-offset table is empty".into());
    }
    let chunk_count_u32 =
        u32::try_from(chunk_count).map_err(|_| "chunk-offset count exceeds the MP4 u32 limit")?;

    let stsc_len = table.stsc_len();
    if stsc_len == 0 {
        return Err("stsc has no entries".into());
    }
    let mut mapped_samples = 0u64;
    for index in 0..stsc_len {
        let entry = table
            .stsc_entry(index)
            .ok_or("stsc entry disappeared during validation")?;
        if index == 0 && entry.first_chunk != 1 {
            return Err(format!(
                "stsc first entry starts at chunk {}, expected 1",
                entry.first_chunk
            ));
        }
        if entry.first_chunk == 0 || entry.first_chunk > chunk_count_u32 {
            return Err(format!(
                "stsc entry {} first_chunk {} is outside 1..={}",
                index + 1,
                entry.first_chunk,
                chunk_count
            ));
        }
        if entry.samples_per_chunk == 0 {
            return Err(format!(
                "stsc entry {} has zero samples_per_chunk",
                index + 1
            ));
        }
        if entry.sample_description_index != 1 {
            return Err(format!(
                "stsc entry {} references unsupported sample description {} (expected 1)",
                index + 1,
                entry.sample_description_index
            ));
        }

        let next_first_chunk = if index + 1 < stsc_len {
            let next = table
                .stsc_entry(index + 1)
                .ok_or("stsc entry disappeared during validation")?;
            if next.first_chunk <= entry.first_chunk {
                return Err(format!(
                    "stsc first_chunk values are not strictly increasing at entry {}",
                    index + 2
                ));
            }
            u64::from(next.first_chunk)
        } else {
            u64::from(chunk_count_u32) + 1
        };
        let run_chunks = next_first_chunk
            .checked_sub(u64::from(entry.first_chunk))
            .ok_or("stsc chunk run underflows")?;
        let run_samples = run_chunks
            .checked_mul(u64::from(entry.samples_per_chunk))
            .ok_or("stsc sample total overflows")?;
        mapped_samples = mapped_samples
            .checked_add(run_samples)
            .ok_or("stsc sample total overflows")?;
    }
    if mapped_samples != u64::from(stsz.sample_count) {
        return Err(format!(
            "stsc maps {mapped_samples} samples but stsz declares {}",
            stsz.sample_count
        ));
    }

    let stts_len = table.stts_len();
    if stts_len == 0 {
        return Err("stts has no entries".into());
    }
    let stts_samples =
        sum_positive_counts(stts_len, |index| table.stts_sample_count(index), "stts")?;
    if stts_samples != u64::from(stsz.sample_count) {
        return Err(format!(
            "stts covers {stts_samples} samples but stsz declares {}",
            stsz.sample_count
        ));
    }

    if table.has_ctts() {
        let ctts_len = table.ctts_len();
        if ctts_len == 0 {
            return Err("ctts is present but has no entries".into());
        }
        let ctts_samples =
            sum_positive_counts(ctts_len, |index| table.ctts_sample_count(index), "ctts")?;
        if ctts_samples != u64::from(stsz.sample_count) {
            return Err(format!(
                "ctts covers {ctts_samples} samples but stsz declares {}",
                stsz.sample_count
            ));
        }
    }

    visit_sample_descriptors(table, stsz.sample_count, |descriptor| {
        if descriptor.size > MAX_AAC_ACCESS_UNIT_SIZE {
            return Err(format!(
                "AAC sample {} is {} bytes, exceeding the {}-byte safety limit",
                descriptor.index, descriptor.size, MAX_AAC_ACCESS_UNIT_SIZE
            ));
        }
        let end = descriptor
            .offset
            .checked_add(u64::from(descriptor.size))
            .ok_or_else(|| format!("AAC sample {} byte range overflows", descriptor.index))?;
        if descriptor.offset > file_size || end > file_size {
            return Err(format!(
                "AAC sample {} byte range {}..{} exceeds file size {}",
                descriptor.index, descriptor.offset, end, file_size
            ));
        }
        Ok(())
    })?;

    Ok(ValidatedSampleTable {
        sample_count: stsz.sample_count,
    })
}

fn sum_positive_counts<F>(len: usize, mut count_at: F, table_name: &str) -> Result<u64, String>
where
    F: FnMut(usize) -> Option<u32>,
{
    let mut total = 0u64;
    for index in 0..len {
        let count = count_at(index)
            .ok_or_else(|| format!("{table_name} entry disappeared during validation"))?;
        if count == 0 {
            return Err(format!(
                "{table_name} entry {} has zero sample_count",
                index + 1
            ));
        }
        total = total
            .checked_add(u64::from(count))
            .ok_or_else(|| format!("{table_name} sample total overflows"))?;
    }
    Ok(total)
}

/// Re-walk the validated tables using only scalar cursor state. Descriptors are
/// consumed immediately instead of being collected into a sample-count-sized
/// allocation.
fn visit_sample_descriptors<T, F>(
    table: &T,
    expected_sample_count: u32,
    mut visit: F,
) -> Result<(), String>
where
    T: SampleTable + ?Sized,
    F: FnMut(SampleDescriptor) -> Result<(), String>,
{
    let chunk_count = match table.chunk_offsets_kind() {
        ChunkOffsetsKind::Stco(len) | ChunkOffsetsKind::Co64(len) => len,
        ChunkOffsetsKind::Missing => return Err("missing stco/co64 chunk-offset table".into()),
        ChunkOffsetsKind::Both => {
            return Err("both stco and co64 are present; exactly one is required".into())
        }
    };
    let stsz = table.stsz_fields();
    let mut stsc_index = 0usize;
    let mut emitted = 0u32;

    for chunk_index in 0..chunk_count {
        let chunk_number = u32::try_from(chunk_index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .ok_or("chunk number exceeds the MP4 u32 limit")?;
        while stsc_index + 1 < table.stsc_len() {
            let next = table
                .stsc_entry(stsc_index + 1)
                .ok_or("stsc entry disappeared while walking samples")?;
            if chunk_number < next.first_chunk {
                break;
            }
            stsc_index += 1;
        }
        let stsc = table
            .stsc_entry(stsc_index)
            .ok_or("stsc entry missing while walking samples")?;
        let mut offset = table
            .chunk_offset(chunk_index)
            .ok_or("chunk offset missing while walking samples")?;

        for _ in 0..stsc.samples_per_chunk {
            let sample_index = usize::try_from(emitted)
                .map_err(|_| "sample index cannot be represented on this platform")?;
            let size = if stsz.sample_size == 0 {
                table
                    .variable_sample_size(sample_index)
                    .ok_or("stsz entry missing while walking samples")?
            } else {
                stsz.sample_size
            };
            let index = emitted.checked_add(1).ok_or("AAC sample index overflows")?;
            visit(SampleDescriptor {
                offset,
                size,
                index,
            })?;
            offset = offset
                .checked_add(u64::from(size))
                .ok_or_else(|| format!("AAC sample {index} byte range overflows"))?;
            emitted = index;
        }
    }

    if emitted != expected_sample_count {
        return Err(format!(
            "sample-table walk produced {emitted} samples, expected {expected_sample_count}"
        ));
    }
    Ok(())
}

fn channel_config_to_count(cfg: ChannelConfig) -> usize {
    match cfg {
        ChannelConfig::Mono => 1,
        ChannelConfig::Stereo => 2,
        ChannelConfig::Three => 3,
        ChannelConfig::Four => 4,
        ChannelConfig::Five => 5,
        ChannelConfig::FiveOne => 6,
        ChannelConfig::SevenOne => 8,
    }
}

fn append_decoded_frame(
    channels: &mut [Vec<f64>],
    frame: &DecodedFrame,
    expected_channels: usize,
    expected_sample_rate: u32,
    total_frames: &mut usize,
) -> Result<(), String> {
    if frame.channels == 0 {
        if frame.pcm.is_empty() {
            return Ok(());
        }
        return Err("zero-channel AAC frame unexpectedly contains PCM samples".into());
    }
    // Tolerate an empty channel-bearing frame as a decoder priming/no-output
    // marker, matching the raw ADTS adapter.
    if frame.pcm.is_empty() {
        return Ok(());
    }
    if channels.len() != expected_channels || frame.channels != expected_channels {
        return Err(format!(
            "decoded channel count {} does not match AAC configuration {}",
            frame.channels, expected_channels
        ));
    }
    if frame.sample_rate != expected_sample_rate {
        return Err(format!(
            "decoded sample rate {} does not match AAC configuration {}",
            frame.sample_rate, expected_sample_rate
        ));
    }
    if frame.pcm.len() % expected_channels != 0 {
        return Err(format!(
            "decoded PCM length {} is not divisible by {} channels",
            frame.pcm.len(),
            expected_channels
        ));
    }

    let frame_count = frame.pcm.len() / expected_channels;
    let next_total = total_frames
        .checked_add(frame_count)
        .ok_or("decoded M4A frame count overflows")?;
    for channel in channels.iter_mut() {
        channel
            .try_reserve(frame_count)
            .map_err(|e| format!("reserve decoded M4A PCM: {e}"))?;
    }
    for samples in frame.pcm.chunks_exact(expected_channels) {
        for (channel, sample) in channels.iter_mut().zip(samples) {
            let value = *sample as f64 / 32768.0;
            channel.push(crate::audio::sanitize_sample(value));
        }
    }
    *total_frames = next_total;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[derive(Clone, Debug)]
    struct TestTable {
        fixed_size: u32,
        sample_count: u32,
        variable_sizes: Vec<u32>,
        chunk_kind: ChunkOffsetsKind,
        chunk_offsets: Vec<u64>,
        stsc: Vec<StscFields>,
        stts: Vec<u32>,
        ctts: Option<Vec<u32>>,
    }

    impl TestTable {
        fn variable_stco() -> Self {
            Self {
                fixed_size: 0,
                sample_count: 3,
                variable_sizes: vec![2, 3, 4],
                chunk_kind: ChunkOffsetsKind::Stco(2),
                chunk_offsets: vec![10, 20],
                stsc: vec![
                    StscFields {
                        first_chunk: 1,
                        samples_per_chunk: 2,
                        sample_description_index: 1,
                    },
                    StscFields {
                        first_chunk: 2,
                        samples_per_chunk: 1,
                        sample_description_index: 1,
                    },
                ],
                stts: vec![3],
                ctts: None,
            }
        }
    }

    impl SampleTable for TestTable {
        fn stsz_fields(&self) -> StszFields {
            StszFields {
                sample_size: self.fixed_size,
                sample_count: self.sample_count,
                variable_size_count: self.variable_sizes.len(),
            }
        }

        fn variable_sample_size(&self, index: usize) -> Option<u32> {
            self.variable_sizes.get(index).copied()
        }

        fn chunk_offsets_kind(&self) -> ChunkOffsetsKind {
            self.chunk_kind
        }

        fn chunk_offset(&self, index: usize) -> Option<u64> {
            self.chunk_offsets.get(index).copied()
        }

        fn stsc_len(&self) -> usize {
            self.stsc.len()
        }

        fn stsc_entry(&self, index: usize) -> Option<StscFields> {
            self.stsc.get(index).copied()
        }

        fn stts_len(&self) -> usize {
            self.stts.len()
        }

        fn stts_sample_count(&self, index: usize) -> Option<u32> {
            self.stts.get(index).copied()
        }

        fn has_ctts(&self) -> bool {
            self.ctts.is_some()
        }

        fn ctts_len(&self) -> usize {
            self.ctts.as_ref().map_or(0, Vec::len)
        }

        fn ctts_sample_count(&self, index: usize) -> Option<u32> {
            self.ctts.as_ref()?.get(index).copied()
        }
    }

    fn descriptors(table: &TestTable) -> Result<Vec<SampleDescriptor>, String> {
        let mut descriptors = Vec::new();
        visit_sample_descriptors(table, table.sample_count, |descriptor| {
            descriptors.push(descriptor);
            Ok(())
        })?;
        Ok(descriptors)
    }

    fn encoded_aac_table(sample_sizes: &[usize]) -> (Vec<u8>, Mp4Reader<Cursor<Vec<u8>>>) {
        let config = mp4::Mp4Config {
            major_brand: "M4A ".parse().unwrap(),
            minor_version: 0,
            compatible_brands: vec!["M4A ".parse().unwrap(), "isom".parse().unwrap()],
            timescale: 48_000,
        };
        let cursor = Cursor::new(Vec::new());
        let mut writer = mp4::Mp4Writer::write_start(cursor, &config).unwrap();
        writer
            .add_track(&mp4::TrackConfig::from(mp4::AacConfig::default()))
            .unwrap();
        let mut start_time = 0u64;
        for &size in sample_sizes {
            writer
                .write_sample(
                    1,
                    &mp4::Mp4Sample {
                        start_time,
                        duration: 1024,
                        rendering_offset: 0,
                        is_sync: true,
                        bytes: mp4::Bytes::from(vec![size as u8; size]),
                    },
                )
                .unwrap();
            start_time += 1024;
        }
        writer.write_end().unwrap();
        let bytes = writer.into_writer().into_inner();
        let reader =
            Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64).unwrap();
        (bytes, reader)
    }

    fn encoded_aac_tracks(count: usize) -> Vec<u8> {
        let config = mp4::Mp4Config {
            major_brand: "M4A ".parse().unwrap(),
            minor_version: 0,
            compatible_brands: vec!["M4A ".parse().unwrap(), "isom".parse().unwrap()],
            timescale: 48_000,
        };
        let cursor = Cursor::new(Vec::new());
        let mut writer = mp4::Mp4Writer::write_start(cursor, &config).unwrap();
        for _ in 0..count {
            writer
                .add_track(&mp4::TrackConfig::from(mp4::AacConfig::default()))
                .unwrap();
        }
        writer.write_end().unwrap();
        writer.into_writer().into_inner()
    }

    fn raw_box(name: [u8; 4], body: &[u8]) -> Vec<u8> {
        let size = u32::try_from(8 + body.len()).unwrap();
        let mut bytes = Vec::with_capacity(size as usize);
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(&name);
        bytes.extend_from_slice(body);
        bytes
    }

    fn raw_large_box(name: [u8; 4], body: &[u8]) -> Vec<u8> {
        let size = u64::try_from(16 + body.len()).unwrap();
        let mut bytes = Vec::with_capacity(size as usize);
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&name);
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(body);
        bytes
    }

    fn descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() < 128);
        let mut bytes = vec![tag, payload.len() as u8];
        bytes.extend_from_slice(payload);
        bytes
    }

    fn canonical_esds_body() -> Vec<u8> {
        let mut decoder_config_payload = vec![0; 13];
        decoder_config_payload.extend_from_slice(&descriptor(0x05, &[0x12, 0x10]));
        let decoder_config = descriptor(0x04, &decoder_config_payload);

        let mut es_payload = vec![0; 3];
        es_payload.extend_from_slice(&decoder_config);
        // `mp4 0.14` intentionally writes a zero declared length followed by
        // the one predefined SLConfig byte and reads it the same way.
        es_payload.extend_from_slice(&[0x06, 0x00, 0x00]);

        let mut body = vec![0; 4];
        body.extend_from_slice(&descriptor(0x03, &es_payload));
        body
    }

    #[test]
    fn validates_variable_stco_and_walks_exact_offsets() {
        let table = TestTable::variable_stco();
        assert_eq!(
            validate_table(&table, 100).unwrap(),
            ValidatedSampleTable { sample_count: 3 }
        );
        assert_eq!(
            descriptors(&table).unwrap(),
            vec![
                SampleDescriptor {
                    offset: 10,
                    size: 2,
                    index: 1,
                },
                SampleDescriptor {
                    offset: 12,
                    size: 3,
                    index: 2,
                },
                SampleDescriptor {
                    offset: 20,
                    size: 4,
                    index: 3,
                },
            ]
        );
    }

    #[test]
    fn validates_fixed_co64_and_walks_exact_offsets() {
        let mut table = TestTable::variable_stco();
        table.fixed_size = 4;
        table.variable_sizes.clear();
        table.chunk_kind = ChunkOffsetsKind::Co64(2);
        validate_table(&table, 100).unwrap();
        assert_eq!(
            descriptors(&table).unwrap(),
            vec![
                SampleDescriptor {
                    offset: 10,
                    size: 4,
                    index: 1,
                },
                SampleDescriptor {
                    offset: 14,
                    size: 4,
                    index: 2,
                },
                SampleDescriptor {
                    offset: 20,
                    size: 4,
                    index: 3,
                },
            ]
        );
    }

    #[test]
    fn validates_real_mp4_writer_tables_without_derived_sample_math() {
        for sizes in [&[4usize, 4][..], &[2usize, 5][..]] {
            let (bytes, reader) = encoded_aac_table(sizes);
            validate_mp4_structure(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
            let track = select_aac_track(&reader).unwrap();
            let validated = validate_sample_table(track, bytes.len() as u64).unwrap();
            assert_eq!(validated.sample_count, 2);

            let mut actual_sizes = Vec::new();
            visit_sample_descriptors(track, validated.sample_count, |descriptor| {
                actual_sizes.push(descriptor.size as usize);
                let start = descriptor.offset as usize;
                let end = start + descriptor.size as usize;
                assert_eq!(&bytes[start..end], vec![descriptor.size as u8; end - start]);
                Ok(())
            })
            .unwrap();
            assert_eq!(actual_sizes, sizes);
        }
    }

    #[test]
    fn selects_first_aac_in_moov_order_and_rejects_duplicate_track_ids() {
        let mut bytes = encoded_aac_tracks(2);
        let reader =
            Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64).unwrap();
        assert_eq!(select_aac_track(&reader).unwrap().track_id(), 1);

        let tkhd_offsets = bytes
            .windows(4)
            .enumerate()
            .filter_map(|(offset, bytes)| (bytes == b"tkhd").then_some(offset + 16))
            .collect::<Vec<_>>();
        assert_eq!(tkhd_offsets.len(), 2);
        bytes[tkhd_offsets[1]..tkhd_offsets[1] + 4].copy_from_slice(&1u32.to_be_bytes());
        let duplicate_reader =
            Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64).unwrap();
        let error = select_aac_track(&duplicate_reader).unwrap_err();
        assert!(error.contains("duplicate MP4 track IDs"), "{error}");
    }

    #[test]
    fn rejects_oversized_access_unit_without_allocating_it() {
        let mut table = TestTable::variable_stco();
        table.fixed_size = MAX_AAC_ACCESS_UNIT_SIZE + 1;
        table.variable_sizes.clear();
        let error = validate_table(&table, u64::MAX).unwrap_err();
        assert!(error.contains("safety limit"), "{error}");
    }

    #[test]
    fn rejects_inconsistent_stsz_semantics() {
        let mut variable = TestTable::variable_stco();
        variable.variable_sizes.pop();
        assert!(validate_table(&variable, 100)
            .unwrap_err()
            .contains("variable-size entry count"));

        let mut fixed = TestTable::variable_stco();
        fixed.fixed_size = 3;
        assert!(validate_table(&fixed, 100)
            .unwrap_err()
            .contains("must not have"));
    }

    #[test]
    fn rejects_missing_ambiguous_or_empty_chunk_offsets() {
        for kind in [
            ChunkOffsetsKind::Missing,
            ChunkOffsetsKind::Both,
            ChunkOffsetsKind::Stco(0),
        ] {
            let mut table = TestTable::variable_stco();
            table.chunk_kind = kind;
            assert!(validate_table(&table, 100).is_err(), "{kind:?}");
        }
    }

    #[test]
    fn rejects_zero_unordered_or_mismatched_stsc_runs() {
        let mut zero = TestTable::variable_stco();
        zero.stsc[0].samples_per_chunk = 0;
        assert!(validate_table(&zero, 100)
            .unwrap_err()
            .contains("zero samples_per_chunk"));

        let mut unordered = TestTable::variable_stco();
        unordered.stsc[1].first_chunk = 1;
        assert!(validate_table(&unordered, 100)
            .unwrap_err()
            .contains("not strictly increasing"));

        let mut mismatch = TestTable::variable_stco();
        mismatch.stsc[0].samples_per_chunk = 1;
        assert!(validate_table(&mismatch, 100)
            .unwrap_err()
            .contains("stsc maps"));
    }

    #[test]
    fn rejects_stts_and_ctts_count_mismatches() {
        let mut stts = TestTable::variable_stco();
        stts.stts = vec![2];
        assert!(validate_table(&stts, 100)
            .unwrap_err()
            .contains("stts covers"));

        let mut ctts = TestTable::variable_stco();
        ctts.ctts = Some(vec![1, 1]);
        assert!(validate_table(&ctts, 100)
            .unwrap_err()
            .contains("ctts covers"));

        ctts.ctts = Some(vec![3, 0]);
        assert!(validate_table(&ctts, 100)
            .unwrap_err()
            .contains("zero sample_count"));
    }

    #[test]
    fn rejects_sample_range_past_end_of_file() {
        let mut table = TestTable::variable_stco();
        table.chunk_offsets[1] = 99;
        let error = validate_table(&table, 100).unwrap_err();
        assert!(error.contains("exceeds file size"), "{error}");
    }

    #[test]
    fn accepts_individually_bounded_reused_chunk_offsets() {
        let mut table = TestTable::variable_stco();
        table.sample_count = 2;
        table.variable_sizes = vec![60, 60];
        table.chunk_offsets = vec![10, 10];
        table.stsc = vec![StscFields {
            first_chunk: 1,
            samples_per_chunk: 1,
            sample_description_index: 1,
        }];
        table.stts = vec![2];

        // The aggregate is 120 bytes, but both legal table references are the
        // individually bounded range 10..70 in the same 100-byte file.
        validate_table(&table, 100).unwrap();
    }

    #[test]
    fn zero_size_sample_keeps_mapping_and_offset() {
        let mut table = TestTable::variable_stco();
        table.variable_sizes = vec![2, 0, 4];
        validate_table(&table, 100).unwrap();
        let descriptors = descriptors(&table).unwrap();
        assert_eq!(descriptors[1].offset, 12);
        assert_eq!(descriptors[2].offset, 20);
    }

    #[test]
    fn validates_decoded_geometry_before_appending() {
        let mut channels = vec![Vec::new(), Vec::new()];
        let mut total_frames = 0;
        append_decoded_frame(
            &mut channels,
            &DecodedFrame {
                pcm: vec![i16::MIN, i16::MAX, 0, 16_384],
                channels: 2,
                sample_rate: 48_000,
            },
            2,
            48_000,
            &mut total_frames,
        )
        .unwrap();
        assert_eq!(total_frames, 2);
        assert_eq!(channels[0], vec![-1.0, 0.0]);
        assert_eq!(channels[1], vec![32767.0 / 32768.0, 0.5]);

        for frame in [
            DecodedFrame {
                pcm: vec![0, 0],
                channels: 1,
                sample_rate: 48_000,
            },
            DecodedFrame {
                pcm: vec![0, 0],
                channels: 2,
                sample_rate: 44_100,
            },
            DecodedFrame {
                pcm: vec![0, 0, 0],
                channels: 2,
                sample_rate: 48_000,
            },
            DecodedFrame {
                pcm: vec![1],
                channels: 0,
                sample_rate: 48_000,
            },
        ] {
            let before = channels.clone();
            assert!(
                append_decoded_frame(&mut channels, &frame, 2, 48_000, &mut total_frames,).is_err()
            );
            assert_eq!(channels, before);
        }
    }

    #[test]
    fn nested_zero_size_box_is_rejected_without_stalling() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("zero-size-child.m4a");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&16u32.to_be_bytes());
        bytes.extend_from_slice(b"ftyp");
        bytes.extend_from_slice(b"isom");
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(&16u32.to_be_bytes());
        bytes.extend_from_slice(b"moov");
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(b"free");
        std::fs::write(&path, bytes).unwrap();

        let error = decode_m4a(&path).unwrap_err();
        assert!(error.contains("zero-sized MP4 box free"), "{error}");
    }

    #[test]
    fn structural_preflight_rejects_nonprogress_and_parent_overrun() {
        let mut too_short = Vec::new();
        too_short.extend_from_slice(&4u32.to_be_bytes());
        too_short.extend_from_slice(b"free");
        let error = scan_box_list(
            &mut Cursor::new(&too_short),
            0,
            too_short.len() as u64,
            BoxListKind::Moov,
            0,
        )
        .unwrap_err();
        assert!(error.contains("shorter than its header"), "{error}");

        let mut overrun = Vec::new();
        overrun.extend_from_slice(&16u32.to_be_bytes());
        overrun.extend_from_slice(b"free");
        let error = scan_box_list(
            &mut Cursor::new(&overrun),
            0,
            overrun.len() as u64,
            BoxListKind::Moov,
            0,
        )
        .unwrap_err();
        assert!(error.contains("exceeds parent end"), "{error}");
    }

    #[test]
    fn structural_preflight_keeps_fixed_box_reads_inside_their_siblings() {
        let cases = [
            (BoxListKind::Moov, *b"mvhd", vec![0], 100u64),
            (BoxListKind::Trak, *b"tkhd", vec![0], 84),
            (BoxListKind::Mdia, *b"mdhd", vec![0], 24),
            (BoxListKind::Mdia, *b"hdlr", vec![0; 24], 25),
            (BoxListKind::Minf, *b"smhd", vec![0; 7], 8),
            (BoxListKind::Minf, *b"vmhd", vec![0; 11], 12),
            (BoxListKind::Mvex, *b"mehd", vec![0], 8),
            (BoxListKind::Mvex, *b"trex", vec![0; 23], 24),
        ];
        for (kind, name, body, minimum) in cases {
            let mut bytes = raw_box(name, &body);
            bytes.extend_from_slice(&raw_box(*b"free", &[0; 128]));
            let error = scan_box_list(&mut Cursor::new(&bytes), 0, bytes.len() as u64, kind, 0)
                .expect_err("short fixed box must not borrow its sibling bytes");
            assert!(
                error.contains(&format!("minimum is {minimum}")),
                "{}: {error}",
                fourcc_text(name)
            );
        }

        let hdlr_without_terminator = raw_box(*b"hdlr", &[b'x'; 25]);
        scan_box_list(
            &mut Cursor::new(&hdlr_without_terminator),
            0,
            hdlr_without_terminator.len() as u64,
            BoxListKind::Meta,
            0,
        )
        .expect("QuickTime counted-string hdlr names remain compatible");
    }

    #[test]
    fn structural_preflight_rejects_empty_stsd_and_edts_boxes() {
        let stsd = raw_box(*b"stsd", &[0; 8]);
        let error = scan_box_list(
            &mut Cursor::new(&stsd),
            0,
            stsd.len() as u64,
            BoxListKind::Stbl,
            0,
        )
        .expect_err("mp4 parser always reads one stsd child");
        assert!(
            error.contains("entry_count must be at least one"),
            "{error}"
        );

        let edts = raw_box(*b"edts", &[]);
        let error = scan_box_list(
            &mut Cursor::new(&edts),
            0,
            edts.len() as u64,
            BoxListKind::Trak,
            0,
        )
        .expect_err("mp4 parser always reads one edts child");
        assert!(error.contains("minimum is 8"), "{error}");
    }

    #[test]
    fn structural_preflight_interprets_only_the_first_stsd_entry() {
        let first = raw_box(*b"mp4a", &[0; 28]);

        // A valid, unused QuickTime AudioSampleEntry v1. Its extension starts
        // where a v0 entry would end, and therefore is not a child-box list.
        let mut second_body = vec![0; 28];
        second_body[8..10].copy_from_slice(&1u16.to_be_bytes());
        second_body.extend_from_slice(&1024u32.to_be_bytes()); // samples/packet
        second_body.extend_from_slice(&0u32.to_be_bytes()); // bytes/packet
        second_body.extend_from_slice(&0u32.to_be_bytes()); // bytes/frame
        second_body.extend_from_slice(&2u32.to_be_bytes()); // bytes/sample
        let second = raw_box(*b"mp4a", &second_body);

        let mut body = vec![0; 4];
        body.extend_from_slice(&2u32.to_be_bytes());
        body.extend_from_slice(&first);
        body.extend_from_slice(&second);
        let stsd = raw_box(*b"stsd", &body);
        scan_box_list(
            &mut Cursor::new(&stsd),
            0,
            stsd.len() as u64,
            BoxListKind::Stbl,
            0,
        )
        .expect("unused QuickTime sample entries remain parser-compatible");
    }

    #[test]
    fn structural_preflight_bounds_required_sample_entry_children() {
        for (name, expected) in [
            (*b"avc1", "requires a codec configuration child"),
            (*b"hev1", "requires a codec configuration child"),
            (*b"vp09", "requires a codec configuration child"),
        ] {
            let entry = raw_box(name, &[0; 78]);
            let error = scan_box_list(
                &mut Cursor::new(&entry),
                0,
                entry.len() as u64,
                BoxListKind::Stsd,
                0,
            )
            .expect_err("video sample entry must have its codec child");
            assert!(error.contains(expected), "{}: {error}", fourcc_text(name));
        }

        let mp4a = raw_box(*b"mp4a", &[0; 28]);
        scan_box_list(
            &mut Cursor::new(&mp4a),
            0,
            mp4a.len() as u64,
            BoxListKind::Stsd,
            0,
        )
        .expect("mp4a codec child remains optional to match mp4 0.14");

        for (entry_name, child_name, child_body) in [
            (*b"avc1", *b"avcC", vec![0; 7]),
            (*b"hev1", *b"hvcC", vec![1]),
            (*b"vp09", *b"vpcC", vec![0; 11]),
        ] {
            let mut body = vec![0; 78];
            body.extend_from_slice(&raw_box(child_name, &child_body));
            let entry = raw_box(entry_name, &body);
            scan_box_list(
                &mut Cursor::new(&entry),
                0,
                entry.len() as u64,
                BoxListKind::Stsd,
                0,
            )
            .unwrap();
        }
    }

    #[test]
    fn structural_preflight_bounds_codec_configuration_payloads() {
        let avcc_short = raw_box(*b"avcC", &[0; 6]);
        let error = validate_avcc(
            &mut Cursor::new(&avcc_short),
            read_raw_box_header(
                &mut Cursor::new(&avcc_short),
                0,
                avcc_short.len() as u64,
                false,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(error.contains("minimum is 7"), "{error}");

        let mut truncated_sps = vec![1, 0, 0, 0, 0, 1];
        truncated_sps.extend_from_slice(&5u16.to_be_bytes());
        truncated_sps.extend_from_slice(&[1, 2, 3]);
        let avcc = raw_box(*b"avcC", &truncated_sps);
        let mut cursor = Cursor::new(&avcc);
        let header = read_raw_box_header(&mut cursor, 0, avcc.len() as u64, false).unwrap();
        let error = validate_avcc(&mut cursor, header).unwrap_err();
        assert!(error.contains("avcC SPS exceeds"), "{error}");

        let mut valid_avcc = vec![1, 0, 0, 0, 0, 1];
        valid_avcc.extend_from_slice(&2u16.to_be_bytes());
        valid_avcc.extend_from_slice(&[1, 2]);
        valid_avcc.push(1);
        valid_avcc.extend_from_slice(&1u16.to_be_bytes());
        valid_avcc.push(3);
        valid_avcc.extend_from_slice(&[0xaa, 0xbb]);
        let avcc = raw_box(*b"avcC", &valid_avcc);
        let mut cursor = Cursor::new(&avcc);
        let header = read_raw_box_header(&mut cursor, 0, avcc.len() as u64, false).unwrap();
        validate_avcc(&mut cursor, header).unwrap();

        for (name, body_len, minimum) in [(*b"hvcC", 0, 1), (*b"vpcC", 10, 11)] {
            let config = raw_box(name, &vec![0; body_len]);
            let mut cursor = Cursor::new(&config);
            let header = read_raw_box_header(&mut cursor, 0, config.len() as u64, false).unwrap();
            let error = require_body_size(header, minimum).unwrap_err();
            assert!(error.contains(&format!("minimum is {minimum}")), "{error}");
        }
    }

    #[test]
    fn structural_preflight_bounds_esds_descriptor_hierarchy() {
        let valid = raw_box(*b"esds", &canonical_esds_body());
        let mut cursor = Cursor::new(&valid);
        let header = read_raw_box_header(&mut cursor, 0, valid.len() as u64, false).unwrap();
        validate_esds(&mut cursor, header).unwrap();

        // FFmpeg commonly emits a five-byte AudioSpecificConfig. Only its
        // prefix is interpreted by mp4 0.14, but all five bytes belong to the
        // same DecoderSpecificDescriptor.
        let mut config = vec![0; 13];
        config.extend_from_slice(&descriptor(0x05, &[0x11, 0x90, 0x56, 0xe5, 0x00]));
        let mut es = vec![0; 3];
        es.extend_from_slice(&descriptor(0x04, &config));
        let mut extended_asc_body = vec![0; 4];
        extended_asc_body.extend_from_slice(&descriptor(0x03, &es));
        let extended_asc = raw_box(*b"esds", &extended_asc_body);
        let mut cursor = Cursor::new(&extended_asc);
        let header = read_raw_box_header(&mut cursor, 0, extended_asc.len() as u64, false).unwrap();
        validate_esds(&mut cursor, header).expect("extended ASC bytes stay inside tag 0x05");

        let cases = [
            (vec![0, 0, 0, 0, 0x03, 0x80], "descriptor length"),
            (vec![0, 0, 0, 0, 0x03, 0x7f], "descriptor payload exceeds"),
            (
                {
                    let mut body = vec![0; 4];
                    body.extend_from_slice(&descriptor(0x03, &[0; 2]));
                    body
                },
                "ESDescriptor fixed fields",
            ),
            (
                {
                    let mut es = vec![0; 3];
                    es.extend_from_slice(&descriptor(0x04, &[0; 12]));
                    let mut body = vec![0; 4];
                    body.extend_from_slice(&descriptor(0x03, &es));
                    body
                },
                "DecoderConfigDescriptor fixed fields",
            ),
            (
                {
                    let mut config = vec![0; 13];
                    config.extend_from_slice(&descriptor(0x05, &[0]));
                    let mut es = vec![0; 3];
                    es.extend_from_slice(&descriptor(0x04, &config));
                    let mut body = vec![0; 4];
                    body.extend_from_slice(&descriptor(0x03, &es));
                    body
                },
                "DecoderSpecificDescriptor",
            ),
            (
                {
                    let mut es = vec![0; 3];
                    es.extend_from_slice(&[0x06, 0x00]);
                    let mut body = vec![0; 4];
                    body.extend_from_slice(&descriptor(0x03, &es));
                    body
                },
                "SLConfigDescriptor",
            ),
        ];
        for (body, expected) in cases {
            let bytes = raw_box(*b"esds", &body);
            let mut cursor = Cursor::new(&bytes);
            let header = read_raw_box_header(&mut cursor, 0, bytes.len() as u64, false).unwrap();
            let error = validate_esds(&mut cursor, header).expect_err("malformed esds must fail");
            assert!(error.contains(expected), "expected {expected}: {error}");
        }
    }

    #[test]
    fn structural_preflight_uses_physical_offsets_for_large_codec_boxes() {
        let esds = raw_large_box(*b"esds", &canonical_esds_body());
        let mut mp4a_body = vec![0; 28];
        mp4a_body.extend_from_slice(&esds);
        let mp4a = raw_large_box(*b"mp4a", &mp4a_body);
        scan_box_list(
            &mut Cursor::new(&mp4a),
            0,
            mp4a.len() as u64,
            BoxListKind::Stsd,
            0,
        )
        .unwrap();

        let avcc = raw_large_box(*b"avcC", &[0; 7]);
        let mut avc_body = vec![0; 78];
        avc_body.extend_from_slice(&avcc);
        let avc = raw_large_box(*b"avc1", &avc_body);
        scan_box_list(
            &mut Cursor::new(&avc),
            0,
            avc.len() as u64,
            BoxListKind::Stsd,
            0,
        )
        .unwrap();
    }

    #[test]
    fn structural_preflight_bounds_table_counts_and_metadata_payloads() {
        let mut stts_body = Vec::new();
        stts_body.extend_from_slice(&0u32.to_be_bytes());
        stts_body.extend_from_slice(&u32::MAX.to_be_bytes());
        let stts = raw_box(*b"stts", &stts_body);
        let error = scan_box_list(
            &mut Cursor::new(&stts),
            0,
            stts.len() as u64,
            BoxListKind::Stbl,
            0,
        )
        .unwrap_err();
        assert!(error.contains("body size"), "{error}");

        let data = raw_box(*b"data", &[]);
        let error = scan_box_list(
            &mut Cursor::new(&data),
            0,
            data.len() as u64,
            BoxListKind::IlstItem,
            0,
        )
        .unwrap_err();
        assert!(error.contains("minimum is 8"), "{error}");
    }

    #[test]
    fn structural_preflight_accepts_a_bounded_large_box_header() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(b"free");
        bytes.extend_from_slice(&16u64.to_be_bytes());
        validate_mp4_structure(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
    }

    #[test]
    fn structural_preflight_preserves_terminal_zero_size_box_compatibility() {
        let mut bytes = encoded_aac_tracks(1);
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(b"free");
        bytes.extend_from_slice(b"opaque terminal payload");

        validate_mp4_structure(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
        Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64).unwrap();
    }

    #[test]
    fn structural_preflight_accepts_bounded_emsg_and_fragment_boxes() {
        let mut version_zero_emsg = vec![0, 0, 0, 0];
        version_zero_emsg.extend_from_slice(b"urn:example\0value\0");
        version_zero_emsg.extend_from_slice(&48_000u32.to_be_bytes());
        version_zero_emsg.extend_from_slice(&0u32.to_be_bytes());
        version_zero_emsg.extend_from_slice(&960u32.to_be_bytes());
        version_zero_emsg.extend_from_slice(&1u32.to_be_bytes());
        version_zero_emsg.extend_from_slice(b"message");

        let mut version_one_emsg = vec![1, 0, 0, 0];
        version_one_emsg.extend_from_slice(&48_000u32.to_be_bytes());
        version_one_emsg.extend_from_slice(&960u64.to_be_bytes());
        version_one_emsg.extend_from_slice(&960u32.to_be_bytes());
        version_one_emsg.extend_from_slice(&2u32.to_be_bytes());
        version_one_emsg.extend_from_slice(b"urn:example\0value\0message");

        let mut mfhd_body = vec![0, 0, 0, 0];
        mfhd_body.extend_from_slice(&1u32.to_be_bytes());
        let mfhd = raw_box(*b"mfhd", &mfhd_body);
        let mut tfhd_body = vec![0, 0, 0, 0];
        tfhd_body.extend_from_slice(&1u32.to_be_bytes());
        let tfhd = raw_box(*b"tfhd", &tfhd_body);
        let mut trun_body = vec![0, 0, 2, 0];
        trun_body.extend_from_slice(&1u32.to_be_bytes());
        trun_body.extend_from_slice(&10u32.to_be_bytes());
        let trun = raw_box(*b"trun", &trun_body);
        let mut traf_body = tfhd;
        traf_body.extend_from_slice(&trun);
        let traf = raw_box(*b"traf", &traf_body);
        let mut moof_body = mfhd;
        moof_body.extend_from_slice(&traf);

        let mut bytes = encoded_aac_tracks(1);
        bytes.extend_from_slice(&raw_box(*b"emsg", &version_zero_emsg));
        bytes.extend_from_slice(&raw_box(*b"emsg", &version_one_emsg));
        bytes.extend_from_slice(&raw_box(*b"moof", &moof_body));

        validate_mp4_structure(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
        Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64).unwrap();
    }

    #[test]
    fn structural_preflight_bounds_emsg_strings_and_zero_stride_trun_work() {
        let emsg = raw_box(*b"emsg", &[0, 0, 0, 0, b'x']);
        let error = validate_mp4_structure(&mut Cursor::new(&emsg), emsg.len() as u64)
            .expect_err("unterminated emsg strings must fail");
        assert!(error.contains("not NUL-terminated"), "{error}");

        let mut trun_body = vec![0, 0, 0, 0];
        trun_body.extend_from_slice(&u32::MAX.to_be_bytes());
        let trun = raw_box(*b"trun", &trun_body);
        let error = scan_box_list(
            &mut Cursor::new(&trun),
            0,
            trun.len() as u64,
            BoxListKind::Traf,
            0,
        )
        .expect_err("tiny trun must not request billions of loop iterations");
        assert!(error.contains("aggregate empty trun"), "{error}");

        let mut bounded_body = vec![0, 0, 0, 0];
        bounded_body.extend_from_slice(&6_000_000u32.to_be_bytes());
        let bounded = raw_box(*b"trun", &bounded_body);
        let mut repeated = bounded.clone();
        repeated.extend_from_slice(&bounded);
        let error = scan_box_list(
            &mut Cursor::new(&repeated),
            0,
            repeated.len() as u64,
            BoxListKind::Traf,
            0,
        )
        .expect_err("many empty trun boxes must share one work budget");
        assert!(error.contains("aggregate empty trun"), "{error}");
    }

    #[test]
    fn rejects_missing_file() {
        assert!(decode_m4a(Path::new("/nonexistent/file.m4a")).is_err());
    }
}
