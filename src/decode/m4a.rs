//! M4A / MP4-AAC decoder — `mp4` demux + Pure-Rust `oxideav-aac` AAC-LC decode.
//!
//! Version 0/1 unity-rate edit lists define presentation timing after decode.
//! Multiple media edits and leading/interior empty edits are composed exactly;
//! unsupported or malformed edit timelines fail closed instead of returning
//! untrimmed audio.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use mp4::{ChannelConfig, MediaType, Mp4Reader, Mp4Track, TrackType};
use oxideav_aac::decode::{DecodedFrame, StreamDecoder};

use super::budget::{channel_descriptor_bytes, DecodeBudget};
use super::pcm::DecodedPcm;
use super::DecodeLimits;

/// MPEG-4 `bufferSizeDB` is 24 bits. Using the same finite ceiling for MP4 AAC
/// samples prevents a corrupt `stsz` from requesting an unbounded access-unit
/// buffer. When a working-set limit is configured, the payload-proportional
/// decoder allowance below is checked before allocation or decoder entry.
const MAX_AAC_ACCESS_UNIT_SIZE: u32 = 0x00ff_ffff;
const MAX_MP4_BOX_DEPTH: usize = 32;
const MAX_EMPTY_TRUN_SAMPLES: u64 = 10_000_000;
const MAX_EDIT_WORKING_BYTES: u128 = 512 * 1024 * 1024;
// `oxideav-aac` keeps at most one decoder state per 4-bit element slot and
// bounds every transform/QMF dimension to AAC's fixed 1024-line geometry.
// This allowance covers all 48 possible SCE/LFE/CPE slots, SBR/PS state, and
// their largest simultaneous frame-local f64 work vectors before we enter the
// dependency. It is intentionally independent of the declared output layout,
// because a hostile raw_data_block can carry extra element slots before our
// returned-geometry check rejects it.
const AAC_DECODER_INTERNAL_BYTES: u64 = 128 * 1024 * 1024;
// See the matching ADTS derivation in `aac.rs`: the smallest complete accepted
// SCE is 29 bits and the largest two-channel per-occurrence live set is bounded
// below 56 KiB, so 64 KiB/input byte retains over 4x headroom for hostile
// repeated tags, spectra, returned PCM, and Vec descriptors.
const AAC_DECODER_BYTES_PER_ACCESS_UNIT_BYTE: u64 = 64 * 1024;
/// Covers retained box structs, collection buckets, and the parser's buffered
/// reader for each structurally visited box. Variable-size payload/table
/// allocations are charged separately below. The deliberately large per-box
/// allowance also covers the second `TrakBox` copy held by `Mp4Track`.
const MP4_PARSER_BASE_BYTES: u64 = 64 * 1024;
const MP4_PARSER_BYTES_PER_BOX: u64 = 2 * 1024;
// `extract_edit_context` can format an error after reserving its table clones.
// Charge a small fixed allowance before entering that fallible construction so
// even malformed edit metadata cannot allocate beyond an exact decode cap.
const EDIT_CONTEXT_ERROR_BYTES: u64 = 1024;
const FALLBACK_REASON_BYTES: u64 = 64;

#[derive(Debug)]
struct ScanBudget {
    empty_trun_samples: u64,
    parser_retained_bytes: u64,
}

impl Default for ScanBudget {
    fn default() -> Self {
        Self {
            empty_trun_samples: MAX_EMPTY_TRUN_SAMPLES,
            parser_retained_bytes: MP4_PARSER_BASE_BYTES,
        }
    }
}

impl ScanBudget {
    fn charge_parser_bytes(&mut self, bytes: u64, context: &str) -> Result<(), String> {
        self.parser_retained_bytes = self
            .parser_retained_bytes
            .checked_add(bytes)
            .ok_or_else(|| format!("{context} parser allocation byte count overflows"))?;
        Ok(())
    }

    fn charge_parser_entries(
        &mut self,
        count: u64,
        bytes_per_entry: u64,
        context: &str,
    ) -> Result<(), String> {
        let bytes = count
            .checked_mul(bytes_per_entry)
            .ok_or_else(|| format!("{context} parser allocation byte count overflows"))?;
        self.charge_parser_bytes(bytes, context)
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
    SampleEntry,
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
#[cfg(test)]
pub(super) fn validate_mp4_structure<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
) -> Result<(), String> {
    scan_mp4_structure(reader, file_size).map(|_| ())
}

/// Validate the raw box graph and reject parser-owned allocations before
/// `mp4::Mp4Reader` is allowed to materialize them. The returned byte count is
/// retained by the parsed header throughout native M4A decode.
pub(super) fn preflight_mp4_parser<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    budget: DecodeBudget,
) -> Result<u64, String> {
    let scan = scan_mp4_structure(reader, file_size)?;
    budget.check_peak(0, scan.parser_retained_bytes, "M4A/MP4 header parser")?;
    Ok(scan.parser_retained_bytes)
}

fn scan_mp4_structure<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
) -> Result<ScanBudget, String> {
    if file_size < 8 {
        return Err("file is too short for an MP4 box header".to_string());
    }
    let mut budget = ScanBudget::default();
    scan_box_list_with_budget(reader, 0, file_size, BoxListKind::Top, 0, &mut budget)?;
    Ok(budget)
}

#[cfg(test)]
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
        let header = read_charged_raw_box_header(
            reader,
            cursor,
            end,
            kind == BoxListKind::Top,
            kind,
            budget,
        )?;
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

/// Read and account one structurally visited box exactly once.
///
/// Some container families use manual child scanners to enforce relationships
/// which the generic list walker cannot express. Routing both paths through
/// this helper keeps their parser graph and variable allocations under the
/// same aggregate budget without double-charging generic traversal.
fn read_charged_raw_box_header<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    parent_end: u64,
    top_level: bool,
    parent: BoxListKind,
    budget: &mut ScanBudget,
) -> Result<RawBoxHeader, String> {
    let header = read_raw_box_header(reader, start, parent_end, top_level)?;
    budget.charge_parser_bytes(MP4_PARSER_BYTES_PER_BOX, "MP4 box graph")?;
    charge_variable_parser_allocation(reader, header, parent, budget)?;
    Ok(header)
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

fn charge_variable_parser_allocation<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    parent: BoxListKind,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    match parent {
        BoxListKind::Top => match &header.name {
            b"ftyp" => budget.charge_parser_bytes(header.body_size(), "MP4 ftyp brands"),
            // `EmsgBox` retains two strings and the remaining message body.
            b"emsg" => budget.charge_parser_entries(header.body_size(), 2, "MP4 emsg payload"),
            _ => Ok(()),
        },
        BoxListKind::Stbl => match &header.name {
            // Every track table is retained once in `MoovBox` and cloned once
            // into `Mp4Track`; charge both copies at their Rust entry sizes.
            b"stts" | b"ctts" => charge_counted_parser_table(reader, header, 16, budget),
            b"stsc" => charge_counted_parser_table(reader, header, 24, budget),
            b"stss" | b"stco" => charge_counted_parser_table(reader, header, 8, budget),
            b"co64" => charge_counted_parser_table(reader, header, 16, budget),
            b"stsz" => {
                require_body_size(header, 12)?;
                if read_u32_at(reader, header.body_start + 4)? == 0 {
                    let count = u64::from(read_u32_at(reader, header.body_start + 8)?);
                    budget.charge_parser_entries(count, 8, "MP4 stsz table")
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        },
        BoxListKind::Mdia | BoxListKind::Meta if header.name == *b"hdlr" => {
            // HdlrBox first builds a body-sized byte vector, converts it into
            // its retained String, and the enclosing TrakBox is cloned into
            // Mp4Track. Two body sizes cover the retained copies; a third
            // covers conversion-time overlap before invalid UTF-8 is dropped.
            budget.charge_parser_entries(header.body_size(), 3, "MP4 handler name")
        }
        BoxListKind::Dref if header.name == *b"url " => {
            budget.charge_parser_entries(header.body_size(), 3, "MP4 data-reference URL")
        }
        BoxListKind::Edts if header.name == *b"elst" => {
            require_body_size(header, 8)?;
            let count = u64::from(read_u32_at(reader, header.body_start + 4)?);
            // `mp4::ElstEntry` is cloned with its containing track. Use a
            // padded 32-byte entry allowance per copy.
            budget.charge_parser_entries(count, 64, "MP4 edit-list table")
        }
        BoxListKind::Traf if header.name == *b"trun" => {
            require_body_size(header, 8)?;
            let (_, flags) = read_full_box_header_at(reader, header.body_start)?;
            let count = u64::from(read_u32_at(reader, header.body_start + 4)?);
            let fields = u64::from((flags & 0x000100 != 0) as u8)
                + u64::from((flags & 0x000200 != 0) as u8)
                + u64::from((flags & 0x000400 != 0) as u8)
                + u64::from((flags & 0x000800 != 0) as u8);
            // Moof owns one TrafBox and the selected track clones it.
            budget.charge_parser_entries(count, fields * 8, "MP4 trun tables")
        }
        // An unknown metadata handler retains every child payload. Charging
        // every non-handler child is conservative for the mdir path and exact
        // enough for the dependency's Unknown variant. Track metadata is
        // cloned into `Mp4Track`, hence the factor of two.
        BoxListKind::Meta if header.name != *b"hdlr" => {
            budget.charge_parser_entries(header.body_size(), 2, "MP4 metadata payload")
        }
        BoxListKind::IlstItem if header.name == *b"data" => {
            budget.charge_parser_entries(header.body_size(), 2, "MP4 ilst data payload")
        }
        // AVC parameter-set payloads are retained as nested vectors and then
        // cloned with the track. Other supported sample entries retain only
        // fixed-size configuration fields.
        BoxListKind::Stsd if header.name == *b"avc1" => {
            budget.charge_parser_entries(header.body_size(), 2, "MP4 AVC sample entry")
        }
        _ => Ok(()),
    }
}

fn charge_counted_parser_table<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    retained_bytes_per_entry: u64,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    require_body_size(header, 8)?;
    let count = u64::from(read_u32_at(reader, header.body_start + 4)?);
    budget.charge_parser_entries(count, retained_bytes_per_entry, "MP4 counted table")
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
            b"trak" => scan_trak_children(reader, header, depth, budget),
            b"meta" => scan_meta_children(reader, header, depth, budget),
            b"mvex" => scan_children(reader, header, BoxListKind::Mvex, depth, budget),
            b"udta" => scan_children(reader, header, BoxListKind::Udta, depth, budget),
            _ => Ok(()),
        },
        BoxListKind::Trak => match &header.name {
            b"tkhd" => validate_versioned_body(reader, header, 84, 96),
            b"edts" => validate_edts(reader, header, depth, budget),
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
            b"mp4a" => validate_sample_entry(reader, header, 28, SampleEntryKind::Mp4a, budget),
            b"avc1" => validate_sample_entry(reader, header, 78, SampleEntryKind::Avc, budget),
            b"hev1" => validate_sample_entry(reader, header, 78, SampleEntryKind::Hevc, budget),
            b"vp09" => validate_sample_entry(reader, header, 78, SampleEntryKind::Vp9, budget),
            b"tx3g" => require_body_size(header, 38),
            _ => Ok(()),
        },
        BoxListKind::SampleEntry => Ok(()),
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

fn scan_trak_children<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    let child_depth = depth
        .checked_add(1)
        .ok_or_else(|| "MP4 box nesting depth overflow".to_string())?;
    if child_depth > MAX_MP4_BOX_DEPTH {
        return Err(format!("MP4 box nesting exceeds {MAX_MP4_BOX_DEPTH}"));
    }

    let mut cursor = header.body_start;
    let mut edts_count = 0u32;
    while cursor < header.end {
        let child = read_charged_raw_box_header(
            reader,
            cursor,
            header.end,
            false,
            BoxListKind::Trak,
            budget,
        )?;
        if child.name == *b"edts" {
            edts_count = edts_count
                .checked_add(1)
                .ok_or_else(|| "trak edts count overflow".to_string())?;
            if edts_count > 1 {
                return Err(format!(
                    "MP4 trak at byte {} contains duplicate edts boxes",
                    header.start
                ));
            }
        }
        validate_raw_box(reader, child, BoxListKind::Trak, child_depth, budget)?;
        if child.end <= cursor {
            return Err(format!(
                "MP4 box {:?} at byte {cursor} made no forward progress",
                child.name
            ));
        }
        cursor = child.end;
    }
    if cursor != header.end {
        return Err(format!(
            "MP4 trak children end at byte {cursor}, expected parent end {}",
            header.end
        ));
    }
    Ok(())
}

fn validate_edts<R: Read + Seek>(
    reader: &mut R,
    header: RawBoxHeader,
    depth: usize,
    budget: &mut ScanBudget,
) -> Result<(), String> {
    require_body_size(header, 8)?;
    let child_depth = depth
        .checked_add(1)
        .ok_or_else(|| "MP4 box nesting depth overflow".to_string())?;
    if child_depth > MAX_MP4_BOX_DEPTH {
        return Err(format!("MP4 box nesting exceeds {MAX_MP4_BOX_DEPTH}"));
    }

    let child = read_charged_raw_box_header(
        reader,
        header.body_start,
        header.end,
        false,
        BoxListKind::Edts,
        budget,
    )?;
    if child.name != *b"elst" {
        return Err(format!(
            "MP4 edts at byte {} must contain exactly one elst child, found {}",
            header.start,
            fourcc_text(child.name)
        ));
    }
    validate_raw_box(reader, child, BoxListKind::Edts, child_depth, budget)?;
    if child.end != header.end {
        return Err(format!(
            "MP4 edts at byte {} must contain exactly one elst child",
            header.start
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
        let entry = if actual == 0 {
            read_charged_raw_box_header(
                reader,
                cursor,
                header.end,
                false,
                BoxListKind::Stsd,
                budget,
            )?
        } else {
            // `mp4 0.14` seeks over later entries without materializing them.
            // Validate their physical boundaries without charging retained
            // parser allocations which the dependency never creates.
            read_raw_box_header(reader, cursor, header.end, false)?
        };
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
    budget: &mut ScanBudget,
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

    let child = read_charged_raw_box_header(
        reader,
        start,
        header.end,
        false,
        BoxListKind::SampleEntry,
        budget,
    )?;
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
    let (version, flags) = read_full_box_header_at(reader, header.body_start)?;
    if flags != 0 {
        return Err(format!(
            "MP4 elst at byte {} has nonzero flags {flags:#08x}",
            header.start
        ));
    }
    let entry_size = match version {
        0 => 12u64,
        1 => 20u64,
        _ => {
            return Err(format!(
                "unsupported MP4 edit-list version {version} at byte {}",
                header.start
            ));
        }
    };
    let count = u64::from(read_u32_at(reader, header.body_start + 4)?);
    if count == 0 {
        return Err(format!(
            "MP4 elst at byte {} must contain at least one entry",
            header.start
        ));
    }
    let required = 8u64
        .checked_add(
            count
                .checked_mul(entry_size)
                .ok_or_else(|| "elst byte count overflow".to_string())?,
        )
        .ok_or_else(|| "elst size overflow".to_string())?;
    if header.body_size() != required {
        return Err(format!(
            "MP4 elst at byte {} has {} body bytes, expected exactly {required}",
            header.start,
            header.body_size()
        ));
    }
    Ok(())
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
struct RawEdit {
    segment_duration: u64,
    media_time: i64,
    media_rate: i16,
    media_rate_fraction: i16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RawEditList {
    version: u8,
    flags: u32,
    entries: Vec<RawEdit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RawSttsEntry {
    sample_count: u32,
    sample_delta: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct EditContext {
    edit_list: RawEditList,
    movie_timescale: u32,
    media_timescale: u32,
    composition_offset: i64,
    sample_count: u32,
    stts_entries: Vec<RawSttsEntry>,
}

#[derive(Debug)]
pub(super) struct FallbackTrackEdit {
    track_id: u32,
    edit: Result<Option<EditContext>, String>,
}

#[derive(Debug)]
pub(super) enum M4aDecodeError {
    TryOtherCodec {
        reason: String,
        track_edits: Vec<FallbackTrackEdit>,
        /// Denoize-owned edit metadata kept alive during fallback decode.
        retained_bytes: u64,
    },
    Fatal(String),
}

impl M4aDecodeError {
    #[cfg(test)]
    fn contains(&self, pattern: &str) -> bool {
        self.to_string().contains(pattern)
    }
}

impl std::fmt::Display for M4aDecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TryOtherCodec { reason, .. } | Self::Fatal(reason) => formatter.write_str(reason),
        }
    }
}

impl From<String> for M4aDecodeError {
    fn from(error: String) -> Self {
        Self::Fatal(error)
    }
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

    fn stts_version(&self) -> u8;
    fn stts_flags(&self) -> u32;
    fn stts_len(&self) -> usize;
    fn stts_sample_count(&self, index: usize) -> Option<u32>;
    fn stts_sample_delta(&self, index: usize) -> Option<u32>;

    fn has_ctts(&self) -> bool;
    fn ctts_version(&self) -> Option<u8>;
    fn ctts_flags(&self) -> Option<u32>;
    fn ctts_len(&self) -> usize;
    fn ctts_sample_count(&self, index: usize) -> Option<u32>;
    fn ctts_sample_offset(&self, index: usize) -> Option<i32>;
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

    fn stts_version(&self) -> u8 {
        self.trak.mdia.minf.stbl.stts.version
    }

    fn stts_flags(&self) -> u32 {
        self.trak.mdia.minf.stbl.stts.flags
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

    fn stts_sample_delta(&self, index: usize) -> Option<u32> {
        self.trak
            .mdia
            .minf
            .stbl
            .stts
            .entries
            .get(index)
            .map(|entry| entry.sample_delta)
    }

    fn has_ctts(&self) -> bool {
        self.trak.mdia.minf.stbl.ctts.is_some()
    }

    fn ctts_version(&self) -> Option<u8> {
        self.trak
            .mdia
            .minf
            .stbl
            .ctts
            .as_ref()
            .map(|ctts| ctts.version)
    }

    fn ctts_flags(&self) -> Option<u32> {
        self.trak
            .mdia
            .minf
            .stbl
            .ctts
            .as_ref()
            .map(|ctts| ctts.flags)
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

    fn ctts_sample_offset(&self, index: usize) -> Option<i32> {
        self.trak
            .mdia
            .minf
            .stbl
            .ctts
            .as_ref()?
            .entries
            .get(index)
            .map(|entry| entry.sample_offset)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValidatedSampleTable {
    sample_count: u32,
}

/// Decode M4A/MP4-AAC from an already-open input.
pub(super) fn decode_m4a(
    mut payload_reader: File,
    limits: DecodeLimits,
) -> Result<DecodedPcm, M4aDecodeError> {
    let budget = DecodeBudget::new(limits);
    // Keep the original handle for payload reads. Parsing happens through a
    // clone because each access unit below is read with an absolute seek.
    let file_size = payload_reader
        .metadata()
        .map_err(|e| format!("stat m4a: {e}"))?
        .len();
    let mut structure_file = payload_reader
        .try_clone()
        .map_err(|e| format!("clone m4a handle for structural validation: {e}"))?;
    let parser_retained_bytes = preflight_mp4_parser(&mut structure_file, file_size, budget)
        .map_err(|e| format!("mp4 structure: {e}"))?;

    let mut header_file = payload_reader
        .try_clone()
        .map_err(|e| format!("clone m4a handle for header parsing: {e}"))?;
    header_file
        .seek(SeekFrom::Start(0))
        .map_err(|e| format!("rewind m4a header: {e}"))?;
    let mp4 = Mp4Reader::read_header(BufReader::new(header_file), file_size)
        .map_err(|e| format!("mp4 parse: {e}"))?;

    let track = match select_aac_track(&mp4)? {
        Some(track) => track,
        None => {
            let (track_edits, edit_retained_bytes) =
                fallback_track_edits(&mp4, budget, parser_retained_bytes)?;
            let retained_bytes = edit_retained_bytes
                .checked_add(FALLBACK_REASON_BYTES)
                .ok_or_else(|| "M4A fallback retained byte count overflows".to_string())?;
            budget.check_peak(
                0,
                parser_retained_bytes
                    .checked_add(retained_bytes)
                    .ok_or_else(|| "M4A fallback parser byte count overflows".to_string())?,
                "M4A fallback metadata",
            )?;
            return Err(M4aDecodeError::TryOtherCodec {
                reason: "no AAC audio track found in M4A/MP4".into(),
                track_edits,
                retained_bytes,
            });
        }
    };
    if !track.trafs.is_empty() {
        return Err(format!(
            "fragmented AAC track {} is not supported; a regular M4A sample table is required",
            track.track_id()
        )
        .into());
    }

    let validated = validate_sample_table(track, file_size)
        .map_err(|e| format!("AAC track {} sample table: {e}", track.track_id()))?;
    let edit_clone_bytes = edit_context_requested_bytes(track)?;
    budget.check_peak(
        0,
        parser_retained_bytes
            .checked_add(edit_clone_bytes)
            .ok_or_else(|| "M4A edit metadata byte count overflows".to_string())?,
        "M4A edit metadata",
    )?;
    let edit = extract_edit_context(&mp4, track)?;
    let edit_retained_bytes = edit
        .as_ref()
        .map(edit_context_capacity_bytes)
        .transpose()?
        .unwrap_or(0);
    let native_retained_bytes = parser_retained_bytes
        .checked_add(edit_retained_bytes)
        .ok_or_else(|| "M4A retained metadata byte count overflows".to_string())?;

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

    let decoder_internal_bytes = aac_decoder_retained_bytes(n_ch)?;
    let decoder_temporary_bytes = native_retained_bytes
        .checked_add(decoder_internal_bytes)
        .ok_or_else(|| "M4A/AAC decoder byte count overflows".to_string())?;
    budget.check_planar_frames(n_ch, 0, decoder_temporary_bytes, "M4A/AAC decode")?;
    let mut decoder = StreamDecoder::new();
    let mut channels = Vec::new();
    channels
        .try_reserve_exact(n_ch)
        .map_err(|e| format!("reserve M4A output channels: {e}"))?;
    channels.resize_with(n_ch, Vec::new);
    let mut access_unit = Vec::new();
    let mut decoded_frames = 0usize;
    let mut stts_timeline = SttsTimeline::default();
    let edit_active = edit.is_some();

    visit_sample_descriptors(track, validated.sample_count, |descriptor| {
        // Zero-sized entries still participate in stsc accounting. They carry
        // no AAC payload, so retain the existing behavior of skipping decode.
        if descriptor.size == 0 {
            if edit_active {
                return Err(format!(
                    "AAC sample {} has zero size while an edit list is active",
                    descriptor.index
                ));
            }
            return Ok(());
        }

        let size = usize::try_from(descriptor.size).map_err(|_| {
            format!(
                "AAC sample {} size cannot be represented on this platform",
                descriptor.index
            )
        })?;
        let existing_access_unit_bytes = allocation_capacity_bytes::<u8>(
            access_unit.capacity(),
            "M4A/AAC retained access-unit buffer",
        )?;
        let access_unit_bytes = existing_access_unit_bytes.max(u64::from(descriptor.size));
        let maximum_frame_bytes = maximum_aac_frame_bytes(n_ch)?;
        let access_unit_decoder_bytes = aac_access_unit_decoder_bytes(descriptor.size)?;
        let predecode_temporary = access_unit_bytes
            .checked_add(maximum_frame_bytes)
            .and_then(|bytes| bytes.checked_add(native_retained_bytes))
            .and_then(|bytes| bytes.checked_add(decoder_internal_bytes))
            .and_then(|bytes| bytes.checked_add(access_unit_decoder_bytes))
            .ok_or("M4A/AAC temporary byte count overflows")?;
        budget.check_planar_frames(
            n_ch,
            decoded_frames,
            predecode_temporary,
            "M4A/AAC packet decode",
        )?;
        budget.check_planar_capacities(&channels, predecode_temporary, "M4A/AAC packet decode")?;
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
        let returned_frame_bytes =
            allocation_capacity_bytes::<i16>(frame.pcm.capacity(), "M4A/AAC frame")?;
        let live_access_unit_bytes = allocation_capacity_bytes::<u8>(
            access_unit.capacity(),
            "M4A/AAC retained access-unit buffer",
        )?;
        let frame_count = append_decoded_frame(
            &mut channels,
            &frame,
            n_ch,
            sample_rate,
            &mut decoded_frames,
            budget,
            live_access_unit_bytes
                .checked_add(returned_frame_bytes)
                .and_then(|bytes| bytes.checked_add(native_retained_bytes))
                .and_then(|bytes| bytes.checked_add(decoder_internal_bytes))
                .and_then(|bytes| bytes.checked_add(access_unit_decoder_bytes))
                .ok_or("M4A/AAC temporary byte count overflows")?,
        )
        .map_err(|e| format!("AAC sample {}: {e}", descriptor.index))?;

        if edit_active {
            if frame_count == 0 {
                return Err(format!(
                    "AAC sample {} decoded to zero frames while an edit list is active",
                    descriptor.index
                ));
            }
            let media_units = stts_timeline.advance(track)?;
            let nominal_frames = round_rescaled(
                media_units,
                u128::from(sample_rate),
                u128::from(track.timescale()),
                "AAC stts timeline",
            )?;
            let actual_frames = decoded_frames as u128;
            validate_edit_timeline_position(
                descriptor.index,
                validated.sample_count,
                actual_frames,
                nominal_frames,
            )?;
        }
        Ok(())
    })?;

    if decoded_frames == 0 {
        return Err("M4A decode produced no samples".to_string().into());
    }

    let mut decoded = DecodedPcm {
        sample_rate,
        channels,
        channel_mask: crate::channel_layout::ChannelLayout::from_channel_count(n_ch).mask(),
    };
    if let Some(edit) = &edit {
        apply_edit_to_decoded_with_budget(
            &mut decoded,
            edit,
            budget,
            native_retained_bytes
                .checked_add(allocation_capacity_bytes::<u8>(
                    access_unit.capacity(),
                    "M4A/AAC retained access-unit buffer",
                )?)
                .and_then(|bytes| bytes.checked_add(decoder_internal_bytes))
                .ok_or_else(|| "M4A/AAC retained byte count overflows".to_string())?,
        )?;
    }
    Ok(decoded)
}

fn select_aac_track<R: Read + Seek>(mp4: &Mp4Reader<R>) -> Result<Option<&Mp4Track>, String> {
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
            return Ok(Some(track));
        }
    }

    Ok(None)
}

fn fallback_track_edits<R: Read + Seek>(
    mp4: &Mp4Reader<R>,
    budget: DecodeBudget,
    parser_retained_bytes: u64,
) -> Result<(Vec<FallbackTrackEdit>, u64), String> {
    let outer_bytes =
        allocation_bytes::<FallbackTrackEdit>(mp4.moov.traks.len(), "M4A fallback track metadata")?;
    let mut planned_bytes = outer_bytes;
    for trak in &mp4.moov.traks {
        let Some(track) = mp4.tracks().get(&trak.tkhd.track_id) else {
            continue;
        };
        if track.track_type().ok() == Some(TrackType::Audio) {
            planned_bytes = planned_bytes
                .checked_add(edit_context_requested_bytes(track)?)
                .ok_or("M4A fallback edit metadata byte count overflows")?;
        }
    }
    budget.check_peak(
        0,
        parser_retained_bytes
            .checked_add(planned_bytes)
            .ok_or("M4A fallback parser byte count overflows")?,
        "M4A fallback edit metadata",
    )?;

    let mut track_edits = Vec::new();
    track_edits
        .try_reserve_exact(mp4.moov.traks.len())
        .map_err(|error| format!("reserve M4A fallback track metadata: {error}"))?;
    for trak in &mp4.moov.traks {
        let Some(track) = mp4.tracks().get(&trak.tkhd.track_id) else {
            continue;
        };
        if track.track_type().ok() == Some(TrackType::Audio) {
            track_edits.push(FallbackTrackEdit {
                track_id: track.track_id(),
                edit: extract_edit_context(mp4, track),
            });
        }
    }
    let retained_bytes = fallback_track_edit_capacity_bytes(&track_edits, track_edits.capacity())?;
    Ok((track_edits, retained_bytes))
}

fn edit_context_requested_bytes(track: &Mp4Track) -> Result<u64, String> {
    let Some(edts) = track.trak.edts.as_ref() else {
        return Ok(0);
    };
    let clone_bytes = match edts.elst.as_ref() {
        Some(elst) => allocation_bytes::<RawEdit>(elst.entries.len(), "M4A edit-list clone")?
            .checked_add(allocation_bytes::<RawSttsEntry>(
                track.stts_len(),
                "M4A stts clone",
            )?)
            .ok_or_else(|| "M4A edit context byte count overflows".to_string())?,
        None => 0,
    };
    clone_bytes
        .checked_add(EDIT_CONTEXT_ERROR_BYTES)
        .ok_or_else(|| "M4A edit context byte count overflows".to_string())
}

fn edit_context_capacity_bytes(context: &EditContext) -> Result<u64, String> {
    allocation_capacity_bytes::<RawEdit>(
        context.edit_list.entries.capacity(),
        "M4A edit-list clone",
    )?
    .checked_add(allocation_capacity_bytes::<RawSttsEntry>(
        context.stts_entries.capacity(),
        "M4A stts clone",
    )?)
    .ok_or_else(|| "M4A retained edit context byte count overflows".to_string())
}

fn fallback_track_edit_capacity_bytes(
    track_edits: &[FallbackTrackEdit],
    outer_capacity: usize,
) -> Result<u64, String> {
    let mut bytes = allocation_capacity_bytes::<FallbackTrackEdit>(
        outer_capacity,
        "M4A fallback track metadata",
    )?;
    for track in track_edits {
        let nested = match &track.edit {
            Ok(Some(context)) => edit_context_capacity_bytes(context)?,
            Err(error) => u64::try_from(error.capacity())
                .map_err(|_| "M4A fallback edit error capacity does not fit in u64")?,
            Ok(None) => 0,
        };
        bytes = bytes
            .checked_add(nested)
            .ok_or("M4A retained fallback edit byte count overflows")?;
    }
    Ok(bytes)
}

fn extract_edit_context<R: Read + Seek>(
    mp4: &Mp4Reader<R>,
    track: &Mp4Track,
) -> Result<Option<EditContext>, String> {
    let Some(edts) = track.trak.edts.as_ref() else {
        return Ok(None);
    };
    let elst = edts
        .elst
        .as_ref()
        .ok_or_else(|| format!("M4A audio track {} edts is missing elst", track.track_id()))?;
    if !matches!(elst.version, 0 | 1) {
        return Err(format!(
            "unsupported edit-list version {} on track {}",
            elst.version,
            track.track_id()
        ));
    }

    let mut entries = Vec::new();
    entries
        .try_reserve_exact(elst.entries.len())
        .map_err(|error| format!("reserve edit-list entries: {error}"))?;
    for entry in &elst.entries {
        let media_time = restore_edit_media_time(elst.version, entry.media_time)?;
        entries.push(RawEdit {
            segment_duration: entry.segment_duration,
            media_time,
            media_rate: entry.media_rate as i16,
            media_rate_fraction: entry.media_rate_fraction as i16,
        });
    }
    let edit_list = RawEditList {
        version: elst.version,
        flags: elst.flags,
        entries,
    };
    validate_raw_edit_list(&edit_list)?;

    let movie_timescale = mp4.timescale();
    if movie_timescale == 0 {
        return Err("M4A edit list has a zero movie timescale".into());
    }
    let media_timescale = track.timescale();
    if media_timescale == 0 {
        return Err(format!(
            "M4A edit list track {} has a zero media timescale",
            track.track_id()
        ));
    }
    let source = validate_edit_active_source_table(track)?;

    Ok(Some(EditContext {
        edit_list,
        movie_timescale,
        media_timescale,
        composition_offset: source.composition_offset,
        sample_count: source.sample_count,
        stts_entries: source.stts_entries,
    }))
}

fn restore_edit_media_time(version: u8, raw: u64) -> Result<i64, String> {
    match version {
        0 => Ok(i64::from(raw as u32 as i32)),
        1 => Ok(raw as i64),
        _ => Err(format!("unsupported M4A edit-list version {version}")),
    }
}

fn validate_raw_edit_list(edit_list: &RawEditList) -> Result<(), String> {
    if !matches!(edit_list.version, 0 | 1) {
        return Err(format!(
            "unsupported M4A edit-list version {}",
            edit_list.version
        ));
    }
    if edit_list.flags != 0 {
        return Err(format!(
            "M4A edit-list flags must be zero, found {:#08x}",
            edit_list.flags
        ));
    }
    if edit_list.entries.is_empty() {
        return Err("M4A edit list has no entries".into());
    }
    for (index, entry) in edit_list.entries.iter().enumerate() {
        if entry.segment_duration == 0 {
            return Err(format!(
                "M4A edit-list entry {} has zero segment duration",
                index + 1
            ));
        }
        if entry.media_rate != 1 || entry.media_rate_fraction != 0 {
            return Err(format!(
                "M4A edit-list entry {} has unsupported media rate {}+{}/65536; only 1+0/65536 is supported",
                index + 1,
                entry.media_rate,
                entry.media_rate_fraction
            ));
        }
        if entry.media_time < -1 {
            return Err(format!(
                "M4A edit-list entry {} has unsupported negative media time {}",
                index + 1,
                entry.media_time
            ));
        }
    }
    if edit_list.entries.last().unwrap().media_time == -1 {
        return Err("M4A edit list must not end with an empty edit".into());
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct ValidatedEditSource {
    composition_offset: i64,
    sample_count: u32,
    stts_entries: Vec<RawSttsEntry>,
}

fn validate_edit_active_source_table<T: SampleTable + ?Sized>(
    table: &T,
) -> Result<ValidatedEditSource, String> {
    let stsz = table.stsz_fields();
    if stsz.sample_count == 0 {
        return Err("edit-active M4A track has no access units".into());
    }
    if stsz.sample_size == 0 {
        let count = usize::try_from(stsz.sample_count)
            .map_err(|_| "edit-active sample count cannot be represented on this platform")?;
        if stsz.variable_size_count != count {
            return Err(format!(
                "edit-active stsz has {} sizes for {} access units",
                stsz.variable_size_count, stsz.sample_count
            ));
        }
        for index in 0..count {
            if table.variable_sample_size(index) == Some(0) {
                return Err(format!(
                    "M4A audio sample {} has zero size while an edit list is active",
                    index + 1
                ));
            }
        }
    }
    if table.stts_version() != 0 {
        return Err(format!(
            "unsupported edit-active stts version {}",
            table.stts_version()
        ));
    }
    if table.stts_flags() != 0 {
        return Err(format!(
            "edit-active stts flags must be zero, found {:#08x}",
            table.stts_flags()
        ));
    }
    let stts_len = table.stts_len();
    if stts_len == 0 {
        return Err("edit-active stts has no entries".into());
    }
    let mut stts_entries = Vec::new();
    stts_entries
        .try_reserve_exact(stts_len)
        .map_err(|error| format!("reserve edit-active stts entries: {error}"))?;
    let mut stts_samples = 0u64;
    for index in 0..stts_len {
        let count = table
            .stts_sample_count(index)
            .ok_or("stts entry disappeared while validating edit timing")?;
        if count == 0 {
            return Err(format!("stts entry {} has zero sample_count", index + 1));
        }
        let delta = table
            .stts_sample_delta(index)
            .ok_or("stts delta disappeared while validating edit timing")?;
        if delta == 0 {
            return Err(format!("stts entry {} has zero sample_delta", index + 1));
        }
        stts_samples = stts_samples
            .checked_add(u64::from(count))
            .ok_or("stts sample total overflows")?;
        stts_entries.push(RawSttsEntry {
            sample_count: count,
            sample_delta: delta,
        });
    }
    if stts_samples != u64::from(stsz.sample_count) {
        return Err(format!(
            "stts covers {stts_samples} samples but stsz declares {}",
            stsz.sample_count
        ));
    }
    let composition_offset = constant_composition_offset(table, stsz.sample_count)?;
    Ok(ValidatedEditSource {
        composition_offset,
        sample_count: stsz.sample_count,
        stts_entries,
    })
}

fn constant_composition_offset<T: SampleTable + ?Sized>(
    table: &T,
    sample_count: u32,
) -> Result<i64, String> {
    if !table.has_ctts() {
        return Ok(0);
    }
    let version = table
        .ctts_version()
        .ok_or("ctts disappeared while reading its version")?;
    if !matches!(version, 0 | 1) {
        return Err(format!("unsupported M4A ctts version {version}"));
    }
    let flags = table
        .ctts_flags()
        .ok_or("ctts disappeared while reading its flags")?;
    if flags != 0 {
        return Err(format!("M4A ctts flags must be zero, found {flags:#08x}"));
    }
    let len = table.ctts_len();
    if len == 0 {
        return Err("ctts is present but has no entries".into());
    }

    let mut covered = 0u64;
    let mut constant = None;
    for index in 0..len {
        let count = table
            .ctts_sample_count(index)
            .ok_or("ctts entry disappeared while validating edit timing")?;
        if count == 0 {
            return Err(format!("ctts entry {} has zero sample_count", index + 1));
        }
        covered = covered
            .checked_add(u64::from(count))
            .ok_or("ctts sample total overflows")?;
        let raw = table
            .ctts_sample_offset(index)
            .ok_or("ctts offset disappeared while validating edit timing")?;
        let offset = if version == 0 {
            i64::from(raw as u32)
        } else {
            i64::from(raw)
        };
        if let Some(expected) = constant {
            if offset != expected {
                return Err(format!(
                    "M4A edit list requires a constant ctts offset, found {expected} then {offset}"
                ));
            }
        } else {
            constant = Some(offset);
        }
    }
    if covered != u64::from(sample_count) {
        return Err(format!(
            "ctts covers {covered} samples but stsz declares {sample_count}"
        ));
    }
    Ok(constant.unwrap())
}

#[derive(Default)]
struct SttsTimeline {
    entry_index: usize,
    remaining_in_entry: u32,
    current_delta: u32,
    cumulative_media_units: u128,
}

impl SttsTimeline {
    fn advance<T: SampleTable + ?Sized>(&mut self, table: &T) -> Result<u128, String> {
        if self.remaining_in_entry == 0 {
            let count = table
                .stts_sample_count(self.entry_index)
                .ok_or("stts ended before the final AAC access unit")?;
            if count == 0 {
                return Err(format!(
                    "stts entry {} has zero sample_count",
                    self.entry_index + 1
                ));
            }
            self.current_delta = table
                .stts_sample_delta(self.entry_index)
                .ok_or("stts sample delta disappeared during AAC decode")?;
            self.remaining_in_entry = count;
            self.entry_index = self
                .entry_index
                .checked_add(1)
                .ok_or("stts entry index overflows")?;
        }
        self.remaining_in_entry -= 1;
        self.cumulative_media_units = self
            .cumulative_media_units
            .checked_add(u128::from(self.current_delta))
            .ok_or("stts cumulative duration overflows")?;
        Ok(self.cumulative_media_units)
    }
}

fn validate_edit_timeline_position(
    sample_index: u32,
    sample_count: u32,
    actual_frames: u128,
    nominal_frames: u128,
) -> Result<(), String> {
    if sample_index < sample_count {
        if actual_frames != nominal_frames {
            return Err(format!(
                "AAC sample {sample_index} cumulative decode is {actual_frames} frames, but stts requires {nominal_frames}"
            ));
        }
    } else if actual_frames < nominal_frames {
        return Err(format!(
            "final AAC decode is {actual_frames} frames, shorter than the stts timeline {nominal_frames}"
        ));
    }
    Ok(())
}

fn round_rescaled(
    value: u128,
    multiplier: u128,
    divisor: u128,
    label: &str,
) -> Result<u128, String> {
    if divisor == 0 {
        return Err(format!("{label} has a zero timescale"));
    }
    let numerator = value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("{label} rescale overflows"))?;
    let quotient = numerator / divisor;
    let remainder = numerator % divisor;
    let half_up_threshold = divisor / 2 + divisor % 2;
    if remainder >= half_up_threshold {
        quotient
            .checked_add(1)
            .ok_or_else(|| format!("{label} rounded result overflows"))
    } else {
        Ok(quotient)
    }
}

fn stts_total_media_units(context: &EditContext) -> Result<u128, String> {
    let mut covered_samples = 0u64;
    let mut media_units = 0u128;
    for (index, entry) in context.stts_entries.iter().enumerate() {
        if entry.sample_count == 0 {
            return Err(format!(
                "M4A edit-active stts entry {} has zero sample_count",
                index + 1
            ));
        }
        if entry.sample_delta == 0 {
            return Err(format!(
                "M4A edit-active stts entry {} has zero sample_delta",
                index + 1
            ));
        }
        covered_samples = covered_samples
            .checked_add(u64::from(entry.sample_count))
            .ok_or("M4A edit-active stts sample total overflows")?;
        let run_duration = u128::from(entry.sample_count)
            .checked_mul(u128::from(entry.sample_delta))
            .ok_or("M4A edit-active stts run duration overflows")?;
        media_units = media_units
            .checked_add(run_duration)
            .ok_or("M4A edit-active stts total duration overflows")?;
    }
    if covered_samples != u64::from(context.sample_count) {
        return Err(format!(
            "M4A edit-active stts covers {covered_samples} samples but stsz declares {}",
            context.sample_count
        ));
    }
    Ok(media_units)
}

fn ceil_rescaled(
    value: u128,
    multiplier: u128,
    divisor: u128,
    label: &str,
) -> Result<u128, String> {
    if divisor == 0 {
        return Err(format!("{label} has a zero timescale"));
    }
    let numerator = value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("{label} rescale overflows"))?;
    let quotient = numerator / divisor;
    if numerator % divisor == 0 {
        Ok(quotient)
    } else {
        quotient
            .checked_add(1)
            .ok_or_else(|| format!("{label} rounded result overflows"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedEdit {
    source_start: Option<usize>,
    frames: usize,
    copy_frames: usize,
}

#[cfg(test)]
fn plan_edits(
    context: &EditContext,
    sample_rate: u32,
    raw_frames: usize,
    channel_count: usize,
) -> Result<(Vec<PlannedEdit>, usize), String> {
    plan_edits_with_budget(
        context,
        sample_rate,
        raw_frames,
        &vec![Vec::new(); channel_count],
        DecodeBudget::new(DecodeLimits::default()),
        0,
    )
}

fn plan_edits_with_budget(
    context: &EditContext,
    sample_rate: u32,
    raw_frames: usize,
    channels: &[Vec<f64>],
    budget: DecodeBudget,
    retained_temporary_bytes: u64,
) -> Result<(Vec<PlannedEdit>, usize), String> {
    let channel_count = channels.len();
    validate_raw_edit_list(&context.edit_list)?;
    if context.movie_timescale == 0 {
        return Err("M4A edit list has a zero movie timescale".into());
    }
    if context.media_timescale == 0 {
        return Err("M4A edit list has a zero media timescale".into());
    }
    if channel_count == 0 {
        return Err("M4A edit list cannot be applied to zero channels".into());
    }
    let stts_media_units = stts_total_media_units(context)?;
    let stts_frame_end = round_rescaled(
        stts_media_units,
        u128::from(sample_rate),
        u128::from(context.media_timescale),
        "M4A edit-list stts frame boundary",
    )?;
    let stts_movie_end = ceil_rescaled(
        stts_media_units,
        u128::from(context.movie_timescale),
        u128::from(context.media_timescale),
        "M4A edit-list stts movie boundary",
    )?;
    let stts_scaled_end = stts_movie_end
        .checked_mul(u128::from(context.media_timescale))
        .ok_or("M4A edit-list stts movie boundary overflows")?;

    let plan_bytes =
        allocation_bytes::<PlannedEdit>(context.edit_list.entries.len(), "M4A edit plan")?;
    let planning_temporary = retained_temporary_bytes
        .checked_add(plan_bytes)
        .ok_or("M4A edit planning byte count overflows")?;
    budget.check_planar_frames(
        channel_count,
        raw_frames,
        planning_temporary,
        "M4A edit plan",
    )?;
    budget.check_planar_capacities(channels, planning_temporary, "M4A edit plan")?;
    let mut planned = Vec::new();
    planned
        .try_reserve_exact(context.edit_list.entries.len())
        .map_err(|error| format!("reserve M4A edit plan: {error}"))?;
    let actual_plan_bytes =
        allocation_capacity_bytes::<PlannedEdit>(planned.capacity(), "M4A edit plan")?;
    let actual_planning_temporary = retained_temporary_bytes
        .checked_add(actual_plan_bytes)
        .ok_or("M4A edit planning byte count overflows")?;
    budget.check_planar_capacities(channels, actual_planning_temporary, "M4A edit plan")?;
    let mut cumulative_duration = 0u128;
    let mut previous_boundary = 0u128;
    let mut output_frames = 0usize;
    for (index, entry) in context.edit_list.entries.iter().enumerate() {
        cumulative_duration = cumulative_duration
            .checked_add(u128::from(entry.segment_duration))
            .ok_or("M4A edit-list movie duration overflows")?;
        let boundary = round_rescaled(
            cumulative_duration,
            u128::from(sample_rate),
            u128::from(context.movie_timescale),
            "M4A edit-list movie boundary",
        )?;
        let segment_frames = boundary
            .checked_sub(previous_boundary)
            .ok_or("M4A edit-list segment boundary underflows")?;
        previous_boundary = boundary;
        let frames = usize::try_from(segment_frames).map_err(|_| {
            format!(
                "M4A edit-list entry {} frame count cannot be represented on this platform",
                index + 1
            )
        })?;
        output_frames = output_frames
            .checked_add(frames)
            .ok_or("M4A edit-list output frame count overflows")?;

        let (source_start, copy_frames) = if entry.media_time == -1 {
            (None, 0)
        } else {
            let adjusted = i128::from(entry.media_time)
                .checked_sub(i128::from(context.composition_offset))
                .ok_or("M4A edit media-time subtraction overflows")?;
            if adjusted < 0 {
                return Err(format!(
                    "M4A edit-list entry {} media source underflows after ctts offset",
                    index + 1
                ));
            }
            let adjusted = adjusted as u128;
            if adjusted >= stts_media_units {
                return Err(format!(
                    "M4A edit-list entry {} media start {adjusted} is at or beyond stts media end {stts_media_units}",
                    index + 1
                ));
            }
            let requested_scaled_end = adjusted
                .checked_mul(u128::from(context.movie_timescale))
                .and_then(|start| {
                    u128::from(entry.segment_duration)
                        .checked_mul(u128::from(context.media_timescale))
                        .and_then(|duration| start.checked_add(duration))
                })
                .ok_or("M4A edit-list media endpoint calculation overflows")?;
            if requested_scaled_end > stts_scaled_end {
                return Err(format!(
                    "M4A edit-list entry {} ends beyond the globally quantized stts media boundary at movie tick {stts_movie_end}",
                    index + 1,
                ));
            }
            let source_frame = round_rescaled(
                adjusted,
                u128::from(sample_rate),
                u128::from(context.media_timescale),
                "M4A edit-list media source",
            )?;
            let copy_frames = segment_frames.min(stts_frame_end.saturating_sub(source_frame));
            let source = usize::try_from(source_frame).map_err(|_| {
                format!(
                    "M4A edit-list entry {} media source cannot be represented on this platform",
                    index + 1
                )
            })?;
            let copy_frames = usize::try_from(copy_frames).map_err(|_| {
                format!(
                    "M4A edit-list entry {} copy length cannot be represented on this platform",
                    index + 1
                )
            })?;
            let source_end = source.checked_add(copy_frames).ok_or_else(|| {
                format!("M4A edit-list entry {} source range overflows", index + 1)
            })?;
            if source_end > raw_frames {
                return Err(format!(
                    "M4A edit-list entry {} source range {source}..{source_end} exceeds decoded length {raw_frames}",
                    index + 1
                ));
            }
            (Some(source), copy_frames)
        };
        planned.push(PlannedEdit {
            source_start,
            frames,
            copy_frames,
        });
    }
    if output_frames == 0 {
        return Err("M4A edit list produces zero output frames".into());
    }

    let is_single_in_place =
        planned.len() == 1 && planned[0].source_start.is_some() && output_frames <= raw_frames;
    if !is_single_in_place {
        let working_bytes = (raw_frames as u128)
            .checked_add(output_frames as u128)
            .and_then(|frames| frames.checked_mul(channel_count as u128))
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u128))
            .ok_or("M4A edit-list working-set calculation overflows")?;
        if working_bytes > MAX_EDIT_WORKING_BYTES {
            return Err(format!(
                "M4A edit-list working set requires {working_bytes} bytes, limit is {MAX_EDIT_WORKING_BYTES} bytes"
            ));
        }
    }

    Ok((planned, output_frames))
}

#[cfg(test)]
fn apply_edit_to_channels(
    channels: &mut Vec<Vec<f64>>,
    sample_rate: u32,
    context: &EditContext,
) -> Result<(), String> {
    apply_edit_to_channels_with_budget(
        channels,
        sample_rate,
        context,
        DecodeBudget::new(DecodeLimits::default()),
        0,
    )
}

fn apply_edit_to_channels_with_budget(
    channels: &mut Vec<Vec<f64>>,
    sample_rate: u32,
    context: &EditContext,
    budget: DecodeBudget,
    retained_temporary_bytes: u64,
) -> Result<(), String> {
    let raw_frames = channels
        .first()
        .map(Vec::len)
        .ok_or("M4A edit list cannot be applied to zero channels")?;
    if let Some((index, length)) = channels.iter().enumerate().find_map(|(index, channel)| {
        (channel.len() != raw_frames).then_some((index, channel.len()))
    }) {
        return Err(format!(
            "M4A edit source channel {} has {length} frames, expected {raw_frames}",
            index + 1
        ));
    }
    let (planned, output_frames) = plan_edits_with_budget(
        context,
        sample_rate,
        raw_frames,
        channels,
        budget,
        retained_temporary_bytes,
    )?;
    let plan_bytes = allocation_capacity_bytes::<PlannedEdit>(planned.capacity(), "M4A edit plan")?;
    let source_temporary = retained_temporary_bytes
        .checked_add(plan_bytes)
        .ok_or("M4A edit source byte count overflows")?;
    budget.check_planar_capacities(channels, source_temporary, "M4A edit source")?;

    if planned.len() == 1 {
        let edit = planned[0];
        let source_start = edit
            .source_start
            .expect("validated one-entry edit list cannot be empty");
        if edit.frames <= raw_frames {
            if source_start == 0 && edit.frames == raw_frames && edit.copy_frames == raw_frames {
                return Ok(());
            }
            let source_end = source_start + edit.copy_frames;
            for channel in channels.iter_mut() {
                channel.copy_within(source_start..source_end, 0);
                channel[edit.copy_frames..edit.frames].fill(0.0);
                channel.truncate(edit.frames);
            }
            return Ok(());
        }
    }

    let source_capacity_samples = channels.iter().try_fold(0u64, |total, channel| {
        let capacity = u64::try_from(channel.capacity())
            .map_err(|_| "M4A edit source capacity does not fit in u64")?;
        total
            .checked_add(capacity)
            .ok_or("M4A edit source capacity count overflows")
    })?;
    let source_descriptor_bytes = channel_descriptor_bytes(channels.len(), "M4A edit source")?;
    let raw_bytes = source_capacity_samples
        .checked_mul(std::mem::size_of::<f64>() as u64)
        .and_then(|bytes| bytes.checked_add(source_descriptor_bytes))
        .and_then(|bytes| bytes.checked_add(plan_bytes))
        .and_then(|bytes| bytes.checked_add(retained_temporary_bytes))
        .ok_or("M4A edit composition byte count overflows")?;
    budget.check_planar_frames(
        channels.len(),
        output_frames,
        raw_bytes,
        "M4A edit composition",
    )?;
    let mut composed = Vec::new();
    composed
        .try_reserve_exact(channels.len())
        .map_err(|error| format!("reserve M4A edited channels: {error}"))?;
    composed.resize_with(channels.len(), Vec::new);
    budget.reserve_planar_frames(
        &mut composed,
        output_frames,
        raw_bytes,
        "M4A edit composition",
    )?;
    budget.check_planar_capacities(&composed, raw_bytes, "M4A edit composition")?;
    for (source, output) in channels.iter().zip(composed.iter_mut()) {
        for edit in &planned {
            if let Some(source_start) = edit.source_start {
                output.extend_from_slice(&source[source_start..source_start + edit.copy_frames]);
                output.resize(output.len() + edit.frames - edit.copy_frames, 0.0);
            } else {
                output.resize(output.len() + edit.frames, 0.0);
            }
        }
    }
    std::mem::swap(channels, &mut composed);
    Ok(())
}

fn apply_edit_to_decoded_with_budget(
    decoded: &mut DecodedPcm,
    context: &EditContext,
    budget: DecodeBudget,
    retained_temporary_bytes: u64,
) -> Result<(), String> {
    apply_edit_to_channels_with_budget(
        &mut decoded.channels,
        decoded.sample_rate,
        context,
        budget,
        retained_temporary_bytes,
    )
}

pub(super) fn fallback_track_has_edit(
    track_edits: &[FallbackTrackEdit],
    selected_track_id: u32,
) -> Result<bool, String> {
    let selected = track_edits
        .iter()
        .find(|track| track.track_id == selected_track_id)
        .ok_or_else(|| {
            format!(
                "fallback decoder selected MP4 audio track {selected_track_id}, but its primary track metadata is unavailable"
            )
        })?;
    selected
        .edit
        .as_ref()
        .map(Option::is_some)
        .map_err(Clone::clone)
}

/// Validate the packet timeline selected by Symphonia before its decoded PCM
/// is interpreted through an edit list. This is intentionally absent for
/// fallback tracks without edits so their historical recovery behavior stays
/// unchanged.
pub(super) fn fallback_timeline_verifier(
    track_edits: &[FallbackTrackEdit],
    selected_track_id: u32,
) -> Result<Option<FallbackTimelineVerifier<'_>>, String> {
    let selected = track_edits
        .iter()
        .find(|track| track.track_id == selected_track_id)
        .ok_or_else(|| {
            format!(
                "fallback decoder selected MP4 audio track {selected_track_id}, but its primary track metadata is unavailable"
            )
        })?;
    let Some(context) = selected.edit.as_ref().map_err(Clone::clone)? else {
        return Ok(None);
    };
    Ok(Some(FallbackTimelineVerifier {
        track_id: selected_track_id,
        entries: &context.stts_entries,
        sample_count: context.sample_count,
        media_timescale: context.media_timescale,
        entry_index: 0,
        remaining_in_entry: 0,
        current_delta: 0,
        observed_packets: 0,
        cumulative_media_units: 0,
        actual_frames: 0,
        sample_rate: None,
    }))
}

pub(super) struct FallbackTimelineVerifier<'a> {
    track_id: u32,
    entries: &'a [RawSttsEntry],
    sample_count: u32,
    media_timescale: u32,
    entry_index: usize,
    remaining_in_entry: u32,
    current_delta: u32,
    observed_packets: u32,
    cumulative_media_units: u128,
    actual_frames: u128,
    sample_rate: Option<u32>,
}

impl FallbackTimelineVerifier<'_> {
    pub(super) fn observe_packet(
        &mut self,
        decoded_frames: usize,
        sample_rate: u32,
    ) -> Result<(), String> {
        if self.observed_packets >= self.sample_count {
            return Err(format!(
                "M4A fallback track {} decoded an extra packet beyond the {} samples declared by stts",
                self.track_id, self.sample_count
            ));
        }
        if decoded_frames == 0 {
            return Err(format!(
                "M4A fallback track {} packet {} decoded to zero frames while an edit list is active",
                self.track_id,
                self.observed_packets + 1
            ));
        }
        if sample_rate == 0 {
            return Err(format!(
                "M4A fallback track {} decoded at a zero sample rate",
                self.track_id
            ));
        }
        if let Some(expected) = self.sample_rate {
            if sample_rate != expected {
                return Err(format!(
                    "M4A fallback track {} sample rate changed from {expected} to {sample_rate}",
                    self.track_id
                ));
            }
        } else {
            self.sample_rate = Some(sample_rate);
        }

        if self.remaining_in_entry == 0 {
            let entry = self.entries.get(self.entry_index).ok_or_else(|| {
                format!(
                    "M4A fallback track {} stts ended before packet {}",
                    self.track_id,
                    self.observed_packets + 1
                )
            })?;
            self.entry_index = self
                .entry_index
                .checked_add(1)
                .ok_or("M4A fallback stts entry index overflows")?;
            self.remaining_in_entry = entry.sample_count;
            self.current_delta = entry.sample_delta;
        }
        self.remaining_in_entry -= 1;
        self.observed_packets = self
            .observed_packets
            .checked_add(1)
            .ok_or("M4A fallback packet count overflows")?;
        self.cumulative_media_units = self
            .cumulative_media_units
            .checked_add(u128::from(self.current_delta))
            .ok_or("M4A fallback stts cumulative duration overflows")?;
        self.actual_frames = self
            .actual_frames
            .checked_add(decoded_frames as u128)
            .ok_or("M4A fallback decoded-frame total overflows")?;

        let nominal_frames = round_rescaled(
            self.cumulative_media_units,
            u128::from(sample_rate),
            u128::from(self.media_timescale),
            "M4A fallback stts timeline",
        )?;
        if self.observed_packets < self.sample_count {
            if self.actual_frames != nominal_frames {
                return Err(format!(
                    "M4A fallback track {} packet {} cumulative decode is {} frames, but stts requires {nominal_frames}",
                    self.track_id, self.observed_packets, self.actual_frames
                ));
            }
        } else if self.actual_frames < nominal_frames {
            return Err(format!(
                "M4A fallback track {} final decode is {} frames, shorter than the stts timeline {nominal_frames}",
                self.track_id, self.actual_frames
            ));
        }
        Ok(())
    }

    pub(super) fn finish(&self) -> Result<(), String> {
        if self.observed_packets != self.sample_count {
            return Err(format!(
                "M4A fallback track {} decoded {} packets, but stts declares {}",
                self.track_id, self.observed_packets, self.sample_count
            ));
        }
        if self.remaining_in_entry != 0 || self.entry_index != self.entries.len() {
            return Err(format!(
                "M4A fallback track {} did not consume its complete stts timeline",
                self.track_id
            ));
        }
        Ok(())
    }
}

pub(super) fn apply_fallback_track_edit(
    decoded: &mut DecodedPcm,
    selected_track_id: u32,
    track_edits: Vec<FallbackTrackEdit>,
    budget: DecodeBudget,
) -> Result<(), String> {
    let selected = track_edits
        .into_iter()
        .find(|track| track.track_id == selected_track_id)
        .ok_or_else(|| {
            format!(
                "fallback decoder selected MP4 audio track {selected_track_id}, but its primary track metadata is unavailable"
            )
    })?;
    if let Some(context) = selected.edit? {
        apply_edit_to_decoded_with_budget(decoded, &context, budget, 0)?;
    }
    Ok(())
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
            return Err("both stco and co64 are present; exactly one is required".into());
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
            return Err("both stco and co64 are present; exactly one is required".into());
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
    budget: DecodeBudget,
    temporary_bytes: u64,
) -> Result<usize, String> {
    if frame.channels == 0 {
        if frame.pcm.is_empty() {
            return Ok(0);
        }
        return Err("zero-channel AAC frame unexpectedly contains PCM samples".into());
    }
    // Tolerate an empty channel-bearing frame as a decoder priming/no-output
    // marker, matching the raw ADTS adapter.
    if frame.pcm.is_empty() {
        return Ok(0);
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
    budget.reserve_planar_additional(channels, frame_count, temporary_bytes, "M4A/AAC decode")?;
    for samples in frame.pcm.chunks_exact(expected_channels) {
        for (channel, sample) in channels.iter_mut().zip(samples) {
            let value = *sample as f64 / 32768.0;
            channel.push(crate::audio::sanitize_sample(value));
        }
    }
    *total_frames = next_total;
    Ok(frame_count)
}

fn maximum_aac_frame_bytes(channels: usize) -> Result<u64, String> {
    let frames = oxideav_aac::decode::FRAME_LEN
        .checked_mul(2)
        .ok_or("M4A/AAC maximum frame count overflows")?;
    let samples = channels
        .checked_mul(frames)
        .ok_or("M4A/AAC maximum sample count overflows")?;
    allocation_bytes::<i16>(samples, "M4A/AAC maximum decoded frame")
}

fn aac_decoder_retained_bytes(_declared_channels: usize) -> Result<u64, String> {
    Ok(AAC_DECODER_INTERNAL_BYTES)
}

fn aac_access_unit_decoder_bytes(access_unit_bytes: u32) -> Result<u64, String> {
    u64::from(access_unit_bytes)
        .checked_mul(AAC_DECODER_BYTES_PER_ACCESS_UNIT_BYTE)
        .ok_or_else(|| "M4A/AAC access-unit decoder byte count overflows".to_string())
}

fn allocation_bytes<T>(len: usize, context: &str) -> Result<u64, String> {
    u64::try_from(len)
        .ok()
        .and_then(|len| len.checked_mul(std::mem::size_of::<T>() as u64))
        .ok_or_else(|| format!("{context} byte count overflows"))
}

fn allocation_capacity_bytes<T>(capacity: usize, context: &str) -> Result<u64, String> {
    allocation_bytes::<T>(capacity, context)
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
        stts_version: u8,
        stts_flags: u32,
        stts: Vec<u32>,
        stts_deltas: Vec<u32>,
        ctts_version: u8,
        ctts_flags: u32,
        ctts: Option<Vec<u32>>,
        ctts_offsets: Vec<i32>,
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
                stts_version: 0,
                stts_flags: 0,
                stts: vec![3],
                stts_deltas: vec![1_024],
                ctts_version: 0,
                ctts_flags: 0,
                ctts: None,
                ctts_offsets: Vec::new(),
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

        fn stts_version(&self) -> u8 {
            self.stts_version
        }

        fn stts_flags(&self) -> u32 {
            self.stts_flags
        }

        fn stts_sample_count(&self, index: usize) -> Option<u32> {
            self.stts.get(index).copied()
        }

        fn stts_sample_delta(&self, index: usize) -> Option<u32> {
            self.stts_deltas.get(index).copied()
        }

        fn has_ctts(&self) -> bool {
            self.ctts.is_some()
        }

        fn ctts_version(&self) -> Option<u8> {
            self.ctts.as_ref().map(|_| self.ctts_version)
        }

        fn ctts_flags(&self) -> Option<u32> {
            self.ctts.as_ref().map(|_| self.ctts_flags)
        }

        fn ctts_len(&self) -> usize {
            self.ctts.as_ref().map_or(0, Vec::len)
        }

        fn ctts_sample_count(&self, index: usize) -> Option<u32> {
            self.ctts.as_ref()?.get(index).copied()
        }

        fn ctts_sample_offset(&self, index: usize) -> Option<i32> {
            self.ctts.as_ref()?.get(index)?;
            Some(self.ctts_offsets.get(index).copied().unwrap_or(0))
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

    fn raw_edit(segment_duration: u64, media_time: i64) -> RawEdit {
        RawEdit {
            segment_duration,
            media_time,
            media_rate: 1,
            media_rate_fraction: 0,
        }
    }

    fn edit_context(
        entries: Vec<RawEdit>,
        movie_timescale: u32,
        media_timescale: u32,
        composition_offset: i64,
    ) -> EditContext {
        EditContext {
            edit_list: RawEditList {
                version: 1,
                flags: 0,
                entries,
            },
            movie_timescale,
            media_timescale,
            composition_offset,
            sample_count: 1,
            stts_entries: vec![RawSttsEntry {
                sample_count: 1,
                sample_delta: u32::MAX,
            }],
        }
    }

    fn raw_elst(version: u8, flags: u32, entries: &[(u64, i64, i16, i16)]) -> Vec<u8> {
        let mut body = vec![
            version,
            ((flags >> 16) & 0xff) as u8,
            ((flags >> 8) & 0xff) as u8,
            (flags & 0xff) as u8,
        ];
        body.extend_from_slice(&(entries.len() as u32).to_be_bytes());
        for &(duration, media_time, rate, fraction) in entries {
            if version == 1 {
                body.extend_from_slice(&duration.to_be_bytes());
                body.extend_from_slice(&media_time.to_be_bytes());
            } else {
                body.extend_from_slice(&(duration as u32).to_be_bytes());
                body.extend_from_slice(&(media_time as i32).to_be_bytes());
            }
            body.extend_from_slice(&rate.to_be_bytes());
            body.extend_from_slice(&fraction.to_be_bytes());
        }
        raw_box(*b"elst", &body)
    }

    fn append_child_to_first_trak(bytes: &mut Vec<u8>, child: &[u8]) {
        let moov_start = bytes
            .windows(4)
            .position(|window| window == b"moov")
            .unwrap()
            - 4;
        let trak_start = bytes
            .windows(4)
            .position(|window| window == b"trak")
            .unwrap()
            - 4;
        let moov_size = u32::from_be_bytes(bytes[moov_start..moov_start + 4].try_into().unwrap());
        let trak_size = u32::from_be_bytes(bytes[trak_start..trak_start + 4].try_into().unwrap());
        let insert_at = trak_start + trak_size as usize;
        bytes.splice(insert_at..insert_at, child.iter().copied());
        let added = u32::try_from(child.len()).unwrap();
        bytes[moov_start..moov_start + 4]
            .copy_from_slice(&moov_size.checked_add(added).unwrap().to_be_bytes());
        bytes[trak_start..trak_start + 4]
            .copy_from_slice(&trak_size.checked_add(added).unwrap().to_be_bytes());
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
            let track = select_aac_track(&reader).unwrap().unwrap();
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
        assert_eq!(select_aac_track(&reader).unwrap().unwrap().track_id(), 1);

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
    fn enforces_mpeg4_access_unit_size_ceiling_without_allocating_payload() {
        let mut table = TestTable::variable_stco();
        table.variable_sizes.clear();

        for accepted_size in [8_192, 64 * 1024, MAX_AAC_ACCESS_UNIT_SIZE] {
            table.fixed_size = accepted_size;
            validate_table(&table, u64::MAX).unwrap_or_else(|error| {
                panic!("{accepted_size}-byte MP4 AAC access unit must be accepted: {error}")
            });
        }

        table.fixed_size = MAX_AAC_ACCESS_UNIT_SIZE + 1;
        let error = validate_table(&table, u64::MAX).unwrap_err();
        assert!(error.contains("safety limit"), "{error}");
    }

    #[test]
    fn checks_large_access_unit_amplification_against_exact_working_budget() {
        const LARGE_ACCESS_UNIT_SIZE: u32 = 64 * 1024;
        let amplified = aac_access_unit_decoder_bytes(LARGE_ACCESS_UNIT_SIZE).unwrap();

        let exact =
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(amplified)));
        exact
            .check_peak(0, amplified, "M4A large AAC access unit")
            .expect("the exact payload-proportional working budget must be accepted");

        let one_byte_short = DecodeBudget::new(
            DecodeLimits::default().with_max_working_set_bytes(Some(amplified - 1)),
        );
        let error = one_byte_short
            .check_peak(0, amplified, "M4A large AAC access unit")
            .expect_err("the pre-entry budget check must reject before allocation or decode");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn hostile_access_unit_amplification_remains_bounded() {
        let amplified = aac_access_unit_decoder_bytes(MAX_AAC_ACCESS_UNIT_SIZE).unwrap();
        let budget = DecodeBudget::new(
            DecodeLimits::default().with_max_working_set_bytes(Some(amplified.saturating_sub(1))),
        );
        let error = budget
            .check_peak(0, amplified, "M4A hostile repeated AAC elements")
            .expect_err("payload-proportional oxideav work must be rejected before decode");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn spare_pcm_capacity_is_combined_with_the_next_m4a_access_unit_peak() {
        const MIB: u64 = 1024 * 1024;
        let mut channels = vec![Vec::with_capacity(32_768)];
        channels[0].resize(1_024, 0.0);
        let logical_bytes = allocation_bytes::<f64>(channels[0].len(), "test M4A length").unwrap()
            + std::mem::size_of::<Vec<f64>>() as u64;
        let next_access_unit_temporary = MIB - logical_bytes;
        let budget =
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(MIB)));

        budget
            .check_planar_frames(
                1,
                channels[0].len(),
                next_access_unit_temporary,
                "logical M4A",
            )
            .expect("logical frames fit the crafted cap");
        let error = budget
            .check_planar_capacities(
                &channels,
                next_access_unit_temporary,
                "M4A/AAC packet decode",
            )
            .expect_err("retained spare capacity plus next access unit must be rejected");
        assert!(error.contains("working-set limit"), "{error}");
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
    fn restores_signed_edit_fields_and_rounds_half_up_from_cumulative_boundaries() {
        assert_eq!(restore_edit_media_time(0, u64::from(u32::MAX)).unwrap(), -1);
        assert_eq!(
            restore_edit_media_time(0, u64::from(0x8000_0000u32)).unwrap(),
            i64::from(i32::MIN)
        );
        assert_eq!(restore_edit_media_time(1, u64::MAX).unwrap(), -1);
        assert_eq!(
            restore_edit_media_time(1, i64::MIN as u64).unwrap(),
            i64::MIN
        );
        assert!(restore_edit_media_time(2, 0).is_err());

        assert_eq!(round_rescaled(1, 4, 10, "test").unwrap(), 0);
        assert_eq!(round_rescaled(1, 5, 10, "test").unwrap(), 1);
        assert_eq!(round_rescaled(1, 6, 10, "test").unwrap(), 1);
        assert_eq!(round_rescaled(21, 48_000, 1_001, "test").unwrap(), 1_007);

        let cumulative = edit_context(vec![raw_edit(1, -1), raw_edit(1, 0)], 2, 1, 0);
        let (planned, total) = plan_edits(&cumulative, 1, 1, 1).unwrap();
        assert_eq!(planned[0].frames, 1);
        assert_eq!(planned[1].frames, 0);
        assert_eq!(
            total, 1,
            "two half-frame edits must not round to two frames"
        );

        let thirds = edit_context(
            vec![raw_edit(1, -1), raw_edit(1, -1), raw_edit(1, 0)],
            3,
            1,
            0,
        );
        let (planned, total) = plan_edits(&thirds, 2, 1, 1).unwrap();
        assert_eq!(
            planned.iter().map(|edit| edit.frames).collect::<Vec<_>>(),
            vec![1, 0, 1]
        );
        assert_eq!(total, 2);
    }

    #[test]
    fn applies_single_edit_in_place_and_preserves_input_on_validation_failure() {
        let identity = edit_context(vec![raw_edit(6, 0)], 1, 1, 0);
        let mut unchanged = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]];
        let pointer = unchanged[0].as_ptr();
        apply_edit_to_channels(&mut unchanged, 1, &identity).unwrap();
        assert_eq!(unchanged, vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]);
        assert_eq!(unchanged[0].as_ptr(), pointer);

        let context = edit_context(vec![raw_edit(3, 2)], 1, 1, 0);
        let mut channels = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]];
        apply_edit_to_channels(&mut channels, 1, &context).unwrap();
        assert_eq!(channels, vec![vec![2.0, 3.0, 4.0]]);

        let invalid = edit_context(vec![raw_edit(3, 4)], 1, 1, 0);
        let mut channels = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
        let before = channels.clone();
        let error = apply_edit_to_channels(&mut channels, 1, &invalid).unwrap_err();
        assert!(error.contains("exceeds decoded length"), "{error}");
        assert_eq!(channels, before);

        let mut unequal = vec![vec![0.0, 1.0], vec![0.0]];
        let before = unequal.clone();
        assert!(apply_edit_to_channels(&mut unequal, 1, &context).is_err());
        assert_eq!(unequal, before);
    }

    #[test]
    fn edit_composition_charges_spare_source_and_plan_capacities() {
        const MIB: u64 = 1024 * 1024;
        const RETAINED_TEMPORARY_BYTES: u64 = 2 * MIB;
        let context = edit_context(vec![raw_edit(1, -1), raw_edit(1, 0)], 1, 1, 0);
        let mut source = Vec::with_capacity(32_768);
        source.push(1.0);
        let channels = vec![source];

        let (planned, output_frames) = plan_edits(&context, 1, 1, 1).unwrap();
        assert_eq!(output_frames, 2);
        let plan_bytes =
            allocation_capacity_bytes::<PlannedEdit>(planned.capacity(), "M4A edit plan test")
                .unwrap();
        let source_bytes = channels[0].capacity() as u64 * std::mem::size_of::<f64>() as u64
            + channel_descriptor_bytes(channels.len(), "M4A edit source test").unwrap();
        let output_bytes = crate::decode::budget::planar_pcm_bytes(
            channels.len(),
            output_frames,
            "M4A edit output test",
        )
        .unwrap()
            + channel_descriptor_bytes(channels.len(), "M4A edit output test").unwrap();
        let exact_peak = source_bytes + plan_bytes + output_bytes + RETAINED_TEMPORARY_BYTES;

        let mut exact_channels = channels.clone();
        // `Clone` may compact capacity; restore the crafted spare source.
        let additional_capacity = 32_768 - exact_channels[0].capacity();
        exact_channels[0].reserve_exact(additional_capacity);
        apply_edit_to_channels_with_budget(
            &mut exact_channels,
            1,
            &context,
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(exact_peak))),
            RETAINED_TEMPORARY_BYTES,
        )
        .expect("exact source, plan, and output capacity boundary");

        let mut tight_channels = Vec::new();
        tight_channels.push(Vec::with_capacity(32_768));
        tight_channels[0].push(1.0);
        let error = apply_edit_to_channels_with_budget(
            &mut tight_channels,
            1,
            &context,
            DecodeBudget::new(
                DecodeLimits::default().with_max_working_set_bytes(Some(exact_peak - 1)),
            ),
            RETAINED_TEMPORARY_BYTES,
        )
        .expect_err("spare source plus live plan and output must exceed the tight cap");
        assert!(error.contains("working-set limit"), "{error}");
        assert_eq!(tight_channels[0], vec![1.0]);
    }

    #[test]
    fn edit_planning_rejects_spare_source_capacity_before_plan_reserve() {
        const RETAINED_TEMPORARY_BYTES: u64 = 2 * 1024 * 1024;
        let context = edit_context(vec![raw_edit(1, -1), raw_edit(1, 0)], 1, 1, 0);
        let mut source = Vec::with_capacity(32_768);
        source.push(1.0);
        let channels = vec![source];
        let requested_plan_bytes =
            allocation_bytes::<PlannedEdit>(context.edit_list.entries.len(), "M4A edit plan test")
                .unwrap();
        let source_bytes = channels[0].capacity() as u64 * std::mem::size_of::<f64>() as u64
            + channel_descriptor_bytes(channels.len(), "M4A edit source test").unwrap();
        let exact_peak = source_bytes + requested_plan_bytes + RETAINED_TEMPORARY_BYTES;

        plan_edits_with_budget(
            &context,
            1,
            1,
            &channels,
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(exact_peak))),
            RETAINED_TEMPORARY_BYTES,
        )
        .expect("exact source and requested plan boundary");
        let error = plan_edits_with_budget(
            &context,
            1,
            1,
            &channels,
            DecodeBudget::new(
                DecodeLimits::default().with_max_working_set_bytes(Some(exact_peak - 1)),
            ),
            RETAINED_TEMPORARY_BYTES,
        )
        .expect_err("spare source plus requested plan must fail before plan reserve");
        assert!(error.contains("M4A edit plan"), "{error}");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn zero_fills_movie_tick_rounding_past_the_stts_media_end() {
        let mut leading_like = edit_context(vec![raw_edit(122, 0)], 1_000, 48_000, 0);
        leading_like.sample_count = 1;
        leading_like.stts_entries = vec![RawSttsEntry {
            sample_count: 1,
            sample_delta: 5_824,
        }];
        let (planned, output_frames) = plan_edits(&leading_like, 48_000, 5_824, 1).unwrap();
        assert_eq!(output_frames, 5_856);
        assert_eq!(planned[0].copy_frames, 5_824);
        let mut shifted_leading = leading_like.clone();
        shifted_leading.edit_list.entries[0].media_time = 1;
        assert!(plan_edits(&shifted_leading, 48_000, 5_824, 1)
            .unwrap_err()
            .contains("globally quantized stts media boundary"));

        let mut channels = vec![vec![1.0; 5_824]];
        apply_edit_to_channels(&mut channels, 48_000, &leading_like).unwrap();
        assert_eq!(channels[0].len(), 5_856);
        assert!(channels[0][..5_824].iter().all(|sample| *sample == 1.0));
        assert!(channels[0][5_824..].iter().all(|sample| *sample == 0.0));

        let mut alac_tick = edit_context(vec![raw_edit(1, 152)], 1_000, 8_000, 0);
        alac_tick.sample_count = 1;
        alac_tick.stts_entries = vec![RawSttsEntry {
            sample_count: 1,
            sample_delta: 159,
        }];
        let mut channels = vec![(0..160).map(f64::from).collect::<Vec<_>>()];
        apply_edit_to_channels(&mut channels, 8_000, &alac_tick).unwrap();
        assert_eq!(
            channels[0],
            vec![152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 0.0]
        );

        let mut stereo_end = edit_context(vec![raw_edit(21, 1_024)], 1_001, 48_000, 0);
        stereo_end.sample_count = 1;
        stereo_end.stts_entries = vec![RawSttsEntry {
            sample_count: 1,
            sample_delta: 2_033,
        }];
        let (stereo_plan, _) = plan_edits(&stereo_end, 48_000, 2_033, 1).unwrap();
        assert_eq!(stereo_plan[0].copy_frames, 1_007);
        stereo_end.edit_list.entries[0].segment_duration = 22;
        assert!(plan_edits(&stereo_end, 48_000, 2_033, 1)
            .unwrap_err()
            .contains("globally quantized stts media boundary"));

        let mut ctts_end = edit_context(vec![raw_edit(122, 37)], 1_000, 48_000, 37);
        ctts_end.sample_count = 1;
        ctts_end.stts_entries = vec![RawSttsEntry {
            sample_count: 1,
            sample_delta: 5_824,
        }];
        plan_edits(&ctts_end, 48_000, 5_824, 1).unwrap();
        ctts_end.edit_list.entries[0].media_time = 38;
        assert!(plan_edits(&ctts_end, 48_000, 5_824, 1)
            .unwrap_err()
            .contains("globally quantized stts media boundary"));
    }

    #[test]
    fn composes_leading_and_intermediate_empty_edits_atomically() {
        let context = edit_context(
            vec![
                raw_edit(2, -1),
                raw_edit(2, 1),
                raw_edit(1, -1),
                raw_edit(2, 4),
            ],
            1,
            1,
            0,
        );
        let mut channels = vec![
            vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        ];
        apply_edit_to_channels(&mut channels, 1, &context).unwrap();
        assert_eq!(channels[0], vec![0.0, 0.0, 11.0, 12.0, 0.0, 14.0, 15.0]);
        assert_eq!(channels[1], vec![0.0, 0.0, 21.0, 22.0, 0.0, 24.0, 25.0]);

        let reordered = edit_context(
            vec![raw_edit(2, 2), raw_edit(2, 0), raw_edit(2, 2)],
            1,
            1,
            0,
        );
        let mut channels = vec![vec![0.0, 1.0, 2.0, 3.0]];
        apply_edit_to_channels(&mut channels, 1, &reordered).unwrap();
        assert_eq!(channels[0], vec![2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_invalid_edit_shapes_timescales_overflow_ranges_and_zero_output() {
        let mut trailing = edit_context(vec![raw_edit(1, 0), raw_edit(1, -1)], 1, 1, 0);
        assert!(plan_edits(&trailing, 1, 2, 1)
            .unwrap_err()
            .contains("must not end"));

        trailing.edit_list.entries[1].media_time = 1;
        trailing.edit_list.entries[0].media_rate = -1;
        assert!(plan_edits(&trailing, 1, 2, 1)
            .unwrap_err()
            .contains("unsupported media rate"));

        trailing.edit_list.entries[0].media_rate = 1;
        trailing.edit_list.entries[0].media_rate_fraction = 1;
        assert!(plan_edits(&trailing, 1, 2, 1)
            .unwrap_err()
            .contains("unsupported media rate"));
        trailing.edit_list.entries[0].media_rate_fraction = 0;
        trailing.edit_list.entries[0].media_time = -2;
        assert!(plan_edits(&trailing, 1, 2, 1)
            .unwrap_err()
            .contains("unsupported negative media time"));
        trailing.edit_list.entries[0].media_time = 0;
        trailing.edit_list.entries[0].segment_duration = 0;
        assert!(plan_edits(&trailing, 1, 2, 1)
            .unwrap_err()
            .contains("zero segment duration"));

        let zero_movie = edit_context(vec![raw_edit(1, 0)], 0, 1, 0);
        assert!(plan_edits(&zero_movie, 1, 1, 1)
            .unwrap_err()
            .contains("zero movie"));
        let zero_media = edit_context(vec![raw_edit(1, 0)], 1, 0, 0);
        assert!(plan_edits(&zero_media, 1, 1, 1)
            .unwrap_err()
            .contains("zero media"));

        let underflow = edit_context(vec![raw_edit(1, 0)], 1, 1, 1);
        assert!(plan_edits(&underflow, 1, 1, 1)
            .unwrap_err()
            .contains("underflows"));
        let positive_ctts = edit_context(vec![raw_edit(1, 150)], 1, 1, 100);
        assert_eq!(
            plan_edits(&positive_ctts, 1, 51, 1).unwrap().0[0].source_start,
            Some(50)
        );
        let negative_ctts = edit_context(vec![raw_edit(1, 0)], 1, 1, -100);
        assert_eq!(
            plan_edits(&negative_ctts, 1, 101, 1).unwrap().0[0].source_start,
            Some(100)
        );
        let range = edit_context(vec![raw_edit(2, 1)], 1, 1, 0);
        assert!(plan_edits(&range, 1, 2, 1)
            .unwrap_err()
            .contains("exceeds decoded length"));

        let mut final_padding = edit_context(vec![raw_edit(24, 1_000)], 1, 1, 0);
        final_padding.sample_count = 1;
        final_padding.stts_entries = vec![RawSttsEntry {
            sample_count: 1,
            sample_delta: 1_000,
        }];
        let error = plan_edits(&final_padding, 1, 1_024, 1).unwrap_err();
        assert!(
            error.contains("media start 1000 is at or beyond stts media end 1000"),
            "{error}"
        );
        final_padding.edit_list.entries[0].media_time = 976;
        let (boundary, _) = plan_edits(&final_padding, 1, 1_024, 1).unwrap();
        assert_eq!(boundary[0].source_start, Some(976));
        assert_eq!(boundary[0].frames, 24);

        let mut quantized_end = edit_context(vec![raw_edit(3, 0)], 1, 2, 0);
        quantized_end.sample_count = 1;
        quantized_end.stts_entries = vec![RawSttsEntry {
            sample_count: 1,
            sample_delta: 5,
        }];
        let (rounded_up, _) = plan_edits(&quantized_end, 2, 6, 1).unwrap();
        assert_eq!(rounded_up[0].frames, 6);
        quantized_end.edit_list.entries[0].segment_duration = 4;
        let error = plan_edits(&quantized_end, 2, 8, 1).unwrap_err();
        assert!(
            error.contains("globally quantized stts media boundary"),
            "{error}"
        );

        let zero_output = edit_context(vec![raw_edit(1, 0)], 3, 1, 0);
        assert!(plan_edits(&zero_output, 1, 1, 1)
            .unwrap_err()
            .contains("zero output"));

        let huge_v1 = edit_context(vec![raw_edit(u64::MAX, -1), raw_edit(1, 0)], 1, 1, 0);
        assert!(plan_edits(&huge_v1, 1, 1, 1).is_err());
    }

    #[test]
    fn bounds_only_allocating_edit_paths_to_512_mib() {
        let general = edit_context(
            vec![raw_edit(20_000_000, 0), raw_edit(20_000_000, 20_000_000)],
            1,
            1,
            0,
        );
        let error = plan_edits(&general, 1, 40_000_000, 1).unwrap_err();
        assert!(error.contains("working set"), "{error}");

        let in_place = edit_context(vec![raw_edit(40_000_000, 0)], 1, 1, 0);
        assert!(plan_edits(&in_place, 1, 40_000_000, 1).is_ok());

        let mut single_zero_tail = edit_context(vec![raw_edit(1, 0)], 1, u32::MAX, 0);
        single_zero_tail.sample_count = 1;
        single_zero_tail.stts_entries = vec![RawSttsEntry {
            sample_count: 1,
            sample_delta: 1,
        }];
        let error = plan_edits(&single_zero_tail, u32::MAX, 1, 1).unwrap_err();
        assert!(error.contains("working set"), "{error}");
    }

    #[test]
    fn restores_constant_ctts_offsets_by_version_and_rejects_variation() {
        let mut table = TestTable::variable_stco();
        table.ctts = Some(vec![3]);
        table.ctts_offsets = vec![i32::MIN];
        assert_eq!(
            constant_composition_offset(&table, 3).unwrap(),
            2_147_483_648
        );

        table.ctts_version = 1;
        assert_eq!(
            constant_composition_offset(&table, 3).unwrap(),
            -2_147_483_648
        );

        table.ctts = Some(vec![1, 2]);
        table.ctts_offsets = vec![7, 8];
        assert!(constant_composition_offset(&table, 3)
            .unwrap_err()
            .contains("constant ctts"));
        table.ctts_version = 2;
        assert!(constant_composition_offset(&table, 3)
            .unwrap_err()
            .contains("unsupported"));
    }

    #[test]
    fn enforces_edit_active_access_unit_and_stts_timeline_safety() {
        let mut zero_size = TestTable::variable_stco();
        zero_size.variable_sizes[1] = 0;
        assert!(validate_edit_active_source_table(&zero_size)
            .unwrap_err()
            .contains("zero size"));

        let mut zero_delta = TestTable::variable_stco();
        zero_delta.stts_deltas[0] = 0;
        assert!(validate_edit_active_source_table(&zero_delta)
            .unwrap_err()
            .contains("zero sample_delta"));

        let mut runs = TestTable::variable_stco();
        runs.stts = vec![2, 1];
        runs.stts_deltas = vec![1_024, 960];
        let mut timeline = SttsTimeline::default();
        assert_eq!(timeline.advance(&runs).unwrap(), 1_024);
        assert_eq!(timeline.advance(&runs).unwrap(), 2_048);
        assert_eq!(timeline.advance(&runs).unwrap(), 3_008);

        assert!(validate_edit_timeline_position(1, 3, 1_023, 1_024).is_err());
        assert!(validate_edit_timeline_position(2, 3, 2_048, 2_048).is_ok());
        assert!(validate_edit_timeline_position(3, 3, 3_007, 3_008).is_err());
        assert!(validate_edit_timeline_position(3, 3, 4_032, 3_008).is_ok());

        let mut channels = vec![Vec::new(), Vec::new()];
        let mut total = 0;
        let frames = append_decoded_frame(
            &mut channels,
            &DecodedFrame {
                pcm: Vec::new(),
                channels: 2,
                sample_rate: 48_000,
            },
            2,
            48_000,
            &mut total,
            DecodeBudget::new(DecodeLimits::default()),
            0,
        )
        .unwrap();
        assert_eq!(frames, 0);
        assert_eq!(total, 0);
    }

    #[test]
    fn fallback_edit_is_matched_to_the_decoder_selected_track_id() {
        let first = edit_context(vec![raw_edit(1, 0)], 1, 1, 0);
        let second = edit_context(vec![raw_edit(1, 2)], 1, 1, 0);
        let edits = vec![
            FallbackTrackEdit {
                track_id: 7,
                edit: Ok(Some(first)),
            },
            FallbackTrackEdit {
                track_id: 9,
                edit: Ok(Some(second)),
            },
        ];
        let mut decoded = DecodedPcm {
            sample_rate: 1,
            channels: vec![vec![10.0, 11.0, 12.0]],
            channel_mask: None,
        };
        assert!(fallback_track_has_edit(&edits, 9).unwrap());
        apply_fallback_track_edit(
            &mut decoded,
            9,
            edits,
            DecodeBudget::new(DecodeLimits::default()),
        )
        .unwrap();
        assert_eq!(decoded.channels, vec![vec![12.0]]);

        let before = decoded.channels.clone();
        let error = apply_fallback_track_edit(
            &mut decoded,
            11,
            Vec::new(),
            DecodeBudget::new(DecodeLimits::default()),
        )
        .unwrap_err();
        assert!(error.contains("metadata is unavailable"), "{error}");
        assert_eq!(decoded.channels, before);

        let no_edit = vec![FallbackTrackEdit {
            track_id: 11,
            edit: Ok(None),
        }];
        assert!(!fallback_track_has_edit(&no_edit, 11).unwrap());
    }

    #[test]
    fn fallback_edit_timeline_validates_every_packet_and_stts_coverage() {
        let timeline = |sample_count, stts_entries| {
            let mut context = edit_context(vec![raw_edit(1, 0)], 1, 8_000, 0);
            context.sample_count = sample_count;
            context.stts_entries = stts_entries;
            vec![FallbackTrackEdit {
                track_id: 9,
                edit: Ok(Some(context)),
            }]
        };

        let exact = timeline(
            2,
            vec![RawSttsEntry {
                sample_count: 2,
                sample_delta: 160,
            }],
        );
        let mut verifier = fallback_timeline_verifier(&exact, 9).unwrap().unwrap();
        verifier.observe_packet(160, 8_000).unwrap();
        verifier.observe_packet(160, 8_000).unwrap();
        verifier.finish().unwrap();

        let final_short = timeline(
            1,
            vec![RawSttsEntry {
                sample_count: 1,
                sample_delta: 161,
            }],
        );
        let error = fallback_timeline_verifier(&final_short, 9)
            .unwrap()
            .unwrap()
            .observe_packet(160, 8_000)
            .unwrap_err();
        assert!(
            error.contains("shorter than the stts timeline 161"),
            "{error}"
        );

        let nonfinal_mismatch = timeline(
            2,
            vec![RawSttsEntry {
                sample_count: 2,
                sample_delta: 160,
            }],
        );
        let error = fallback_timeline_verifier(&nonfinal_mismatch, 9)
            .unwrap()
            .unwrap()
            .observe_packet(159, 8_000)
            .unwrap_err();
        assert!(error.contains("stts requires 160"), "{error}");

        let one_packet = timeline(
            1,
            vec![RawSttsEntry {
                sample_count: 1,
                sample_delta: 160,
            }],
        );
        let mut extra = fallback_timeline_verifier(&one_packet, 9).unwrap().unwrap();
        extra.observe_packet(160, 8_000).unwrap();
        assert!(extra
            .observe_packet(160, 8_000)
            .unwrap_err()
            .contains("extra packet"));

        let missing = timeline(
            2,
            vec![RawSttsEntry {
                sample_count: 2,
                sample_delta: 160,
            }],
        );
        let mut missing = fallback_timeline_verifier(&missing, 9).unwrap().unwrap();
        missing.observe_packet(160, 8_000).unwrap();
        assert!(missing.finish().unwrap_err().contains("decoded 1 packets"));

        let no_edit = vec![FallbackTrackEdit {
            track_id: 9,
            edit: Ok(None),
        }];
        assert!(fallback_timeline_verifier(&no_edit, 9).unwrap().is_none());
    }

    #[test]
    fn selected_aac_edit_error_is_fatal_and_never_reaches_symphonia_fallback() {
        let (mut bytes, _) = encoded_aac_table(&[4]);
        let non_unity_edit = raw_box(*b"edts", &raw_elst(0, 0, &[(1, 0, 2, 0)]));
        append_child_to_first_trak(&mut bytes, &non_unity_edit);

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fatal-edit.m4a");
        std::fs::write(&path, bytes).unwrap();

        let native_error =
            decode_m4a(File::open(&path).unwrap(), DecodeLimits::default()).unwrap_err();
        assert!(
            matches!(native_error, M4aDecodeError::Fatal(_)),
            "{native_error}"
        );
        assert!(
            native_error.contains("unsupported media rate"),
            "{native_error}"
        );

        let public_error = crate::decode::decode_file(&path).unwrap_err();
        assert!(
            public_error.contains("unsupported media rate"),
            "{public_error}"
        );
        assert!(!public_error.contains("ALAC/other"), "{public_error}");
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
            DecodeBudget::new(DecodeLimits::default()),
            0,
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
            assert!(append_decoded_frame(
                &mut channels,
                &frame,
                2,
                48_000,
                &mut total_frames,
                DecodeBudget::new(DecodeLimits::default()),
                0,
            )
            .is_err());
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

        let error = decode_m4a(File::open(&path).unwrap(), DecodeLimits::default()).unwrap_err();
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
    fn structural_preflight_charges_manual_elst_children_before_parser_entry() {
        const ENTRY_COUNT: usize = 4_096;
        let entries = vec![(1, 0, 1, 0); ENTRY_COUNT];
        let elst = raw_elst(0, 0, &entries);
        let edts = raw_box(*b"edts", &elst);
        let trak = raw_box(*b"trak", &edts);
        let bytes = raw_box(*b"moov", &trak);

        let scan = scan_mp4_structure(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
        let expected =
            MP4_PARSER_BASE_BYTES + 4 * MP4_PARSER_BYTES_PER_BOX + ENTRY_COUNT as u64 * 64;
        assert_eq!(scan.parser_retained_bytes, expected);

        preflight_mp4_parser(
            &mut Cursor::new(&bytes),
            bytes.len() as u64,
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(expected))),
        )
        .expect("exact edit-list parser boundary");
        let error = preflight_mp4_parser(
            &mut Cursor::new(&bytes),
            bytes.len() as u64,
            DecodeBudget::new(
                DecodeLimits::default().with_max_working_set_bytes(Some(expected - 1)),
            ),
        )
        .expect_err("edit-list parser allocation must be rejected before dependency entry");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn structural_preflight_charges_manual_avc_sample_children_before_parser_entry() {
        let mut avcc_body = vec![0; 256 * 1024];
        avcc_body[0] = 1;
        let avcc = raw_box(*b"avcC", &avcc_body);
        let mut avc1_body = vec![0; 78];
        avc1_body.extend_from_slice(&avcc);
        let avc1 = raw_box(*b"avc1", &avc1_body);
        let mut stsd_body = vec![0; 4];
        stsd_body.extend_from_slice(&1u32.to_be_bytes());
        stsd_body.extend_from_slice(&avc1);
        let stsd = raw_box(*b"stsd", &stsd_body);
        let stbl = raw_box(*b"stbl", &stsd);
        let minf = raw_box(*b"minf", &stbl);
        let mdia = raw_box(*b"mdia", &minf);
        let trak = raw_box(*b"trak", &mdia);
        let bytes = raw_box(*b"moov", &trak);

        let scan = scan_mp4_structure(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
        let expected =
            MP4_PARSER_BASE_BYTES + 8 * MP4_PARSER_BYTES_PER_BOX + avc1_body.len() as u64 * 2;
        assert_eq!(scan.parser_retained_bytes, expected);

        preflight_mp4_parser(
            &mut Cursor::new(&bytes),
            bytes.len() as u64,
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(expected))),
        )
        .expect("exact AVC parser boundary");
        let error = preflight_mp4_parser(
            &mut Cursor::new(&bytes),
            bytes.len() as u64,
            DecodeBudget::new(
                DecodeLimits::default().with_max_working_set_bytes(Some(expected - 1)),
            ),
        )
        .expect_err("AVC parser allocation must be rejected before dependency entry");
        assert!(error.contains("working-set limit"), "{error}");
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
    fn structural_preflight_requires_one_exact_elst_and_rejects_duplicate_edts() {
        let elst = raw_elst(0, 0, &[(10, 0, 1, 0)]);
        let edts = raw_box(*b"edts", &elst);

        let mut duplicate_body = edts.clone();
        duplicate_body.extend_from_slice(&edts);
        let duplicate_trak = raw_box(*b"trak", &duplicate_body);
        let error = scan_box_list(
            &mut Cursor::new(&duplicate_trak),
            0,
            duplicate_trak.len() as u64,
            BoxListKind::Moov,
            0,
        )
        .unwrap_err();
        assert!(error.contains("duplicate edts"), "{error}");

        let wrong = raw_box(*b"edts", &raw_box(*b"free", &[0; 12]));
        let error = scan_box_list(
            &mut Cursor::new(&wrong),
            0,
            wrong.len() as u64,
            BoxListKind::Trak,
            0,
        )
        .unwrap_err();
        assert!(error.contains("exactly one elst"), "{error}");

        let mut two_children = elst.clone();
        two_children.extend_from_slice(&raw_box(*b"free", &[]));
        let extra = raw_box(*b"edts", &two_children);
        let error = scan_box_list(
            &mut Cursor::new(&extra),
            0,
            extra.len() as u64,
            BoxListKind::Trak,
            0,
        )
        .unwrap_err();
        assert!(error.contains("exactly one elst"), "{error}");

        let no_edts = raw_box(*b"trak", &raw_box(*b"free", &[]));
        scan_box_list(
            &mut Cursor::new(&no_edts),
            0,
            no_edts.len() as u64,
            BoxListKind::Moov,
            0,
        )
        .unwrap();
    }

    #[test]
    fn structural_preflight_requires_exact_nonempty_zero_flag_elst_body() {
        let valid = raw_elst(1, 0, &[(10, -1, 1, 0), (20, 0, 1, 0)]);
        scan_box_list(
            &mut Cursor::new(&valid),
            0,
            valid.len() as u64,
            BoxListKind::Edts,
            0,
        )
        .unwrap();

        for malformed in [raw_elst(0, 0, &[]), raw_elst(0, 1, &[(1, 0, 1, 0)])] {
            assert!(scan_box_list(
                &mut Cursor::new(&malformed),
                0,
                malformed.len() as u64,
                BoxListKind::Edts,
                0,
            )
            .is_err());
        }

        let mut trailing = raw_elst(0, 0, &[(1, 0, 1, 0)]);
        trailing.push(0xaa);
        let size = u32::try_from(trailing.len()).unwrap();
        trailing[..4].copy_from_slice(&size.to_be_bytes());
        let error = scan_box_list(
            &mut Cursor::new(&trailing),
            0,
            trailing.len() as u64,
            BoxListKind::Edts,
            0,
        )
        .unwrap_err();
        assert!(error.contains("expected exactly"), "{error}");
    }

    #[test]
    fn rejects_empty_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        assert!(decode_m4a(File::open(file.path()).unwrap(), DecodeLimits::default(),).is_err());
    }
}
