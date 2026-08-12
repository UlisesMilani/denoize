//! Bounded, page-level Ogg Vorbis Comment handling.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};

use super::{
    parse_comment_body, serialize_comment_body, MetadataBudget, MetadataLimits,
    VorbisCommentsSnapshot,
};

pub(super) fn read_file(
    input: &mut File,
    limits: &MetadataLimits,
    budget: &mut MetadataBudget,
) -> Result<Option<VorbisCommentsSnapshot>, String> {
    if !is_ogg(input)? {
        return Ok(None);
    }
    let scan = scan(input, limits, budget, ScanMode::HeadersOnly)
        .map_err(|error| format!("read Ogg metadata: {error}"))?;
    Ok(scan.target.and_then(|target| target.comments))
}

pub(super) fn validate(file: &mut File, limits: &MetadataLimits) -> Result<(), String> {
    if !is_ogg(file)? {
        return Err("input is not an Ogg stream".into());
    }
    let mut budget = MetadataBudget::new(*limits);
    scan(file, limits, &mut budget, ScanMode::Decode)
        .map(|_| ())
        .map_err(|error| format!("validate Ogg stream: {error}"))
}

pub(super) fn rewrite(
    output: &mut File,
    comments: &VorbisCommentsSnapshot,
    limits: &MetadataLimits,
) -> Result<(), String> {
    if !is_ogg(output)? {
        return Ok(());
    }

    let mut destination_budget = MetadataBudget::new(*limits);
    let scan = scan(output, limits, &mut destination_budget, ScanMode::Rewrite)
        .map_err(|error| format!("read Ogg metadata: {error}"))?;
    let Some(target) = scan.target.as_ref() else {
        return Ok(());
    };
    if target.comments.is_none() {
        return Err("Ogg rewrite requires retained destination comments".into());
    }
    if target.shares_header_page_with_audio {
        return Err(
            "Ogg audio data shares the final metadata header page; refusing unsafe rewrite".into(),
        );
    }

    let mut serialize_budget = MetadataBudget::new(*limits);
    let body = serialize_comment_body(comments, &mut serialize_budget)?;
    serialize_budget.charge_bytes(target.tail.len(), "OpusTags tail")?;
    rewrite_scanned(output, &scan, target, body, limits)
        .map_err(|error| format!("write Ogg metadata: {error}"))
}

fn is_ogg(file: &mut File) -> Result<bool, String> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind raw Ogg metadata: {error}"))?;
    let mut capture = [0_u8; 4];
    let mut read = 0;
    while read < capture.len() {
        match file.read(&mut capture[read..]) {
            Ok(0) => break,
            Ok(count) => read += count,
            Err(error) => return Err(format!("read raw Ogg metadata: {error}")),
        }
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind raw Ogg metadata: {error}"))?;
    Ok(read == capture.len() && capture == *b"OggS")
}

#[derive(Debug)]
struct Scan {
    target: Option<Target>,
}

#[derive(Clone, Debug)]
struct Target {
    serial: u32,
    bos_page: u64,
    header_end_page: u64,
    eos_page: Option<u64>,
    codec: Codec,
    comments: Option<VorbisCommentsSnapshot>,
    tail: Vec<u8>,
    shares_header_page_with_audio: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Codec {
    Vorbis,
    Opus,
    Speex { header_packets: usize },
}

impl Codec {
    fn header_packets(self) -> usize {
        match self {
            Self::Vorbis => 3,
            Self::Opus => 2,
            Self::Speex { header_packets } => header_packets,
        }
    }

    fn comment_prefix_len(self) -> usize {
        match self {
            Self::Vorbis => 7,
            Self::Opus => 8,
            Self::Speex { .. } => 0,
        }
    }
}

const CAPTURE: &[u8; 4] = b"OggS";
const HEADER_LEN: usize = 27;
const MAX_SEGMENTS: usize = 255;
const MAX_PAGE_BODY: usize = MAX_SEGMENTS * 255;
const CONTINUED: u8 = 0x01;
const BOS: u8 = 0x02;
const EOS: u8 = 0x04;
const NO_GRANULE: u64 = u64::MAX;

#[derive(Clone, Debug)]
struct Page {
    header_type: u8,
    granule: u64,
    serial: u32,
    sequence: u32,
    checksum: u32,
    lacing: Vec<u8>,
    body: Vec<u8>,
}

impl Page {
    fn is_continued(&self) -> bool {
        self.header_type & CONTINUED != 0
    }

    fn is_bos(&self) -> bool {
        self.header_type & BOS != 0
    }

    fn is_eos(&self) -> bool {
        self.header_type & EOS != 0
    }

    fn completed_packets(&self) -> usize {
        self.lacing.iter().filter(|lace| **lace < 255).count()
    }

    fn encoded_len(&self) -> usize {
        HEADER_LEN + self.lacing.len() + self.body.len()
    }

    fn header(&self, checksum: u32) -> Result<[u8; HEADER_LEN], String> {
        let segment_count = u8::try_from(self.lacing.len())
            .map_err(|_| "Ogg page has too many lacing values".to_owned())?;
        let mut header = [0_u8; HEADER_LEN];
        header[..4].copy_from_slice(CAPTURE);
        header[4] = 0;
        header[5] = self.header_type;
        header[6..14].copy_from_slice(&self.granule.to_le_bytes());
        header[14..18].copy_from_slice(&self.serial.to_le_bytes());
        header[18..22].copy_from_slice(&self.sequence.to_le_bytes());
        header[22..26].copy_from_slice(&checksum.to_le_bytes());
        header[26] = segment_count;
        Ok(header)
    }

    fn calculated_checksum(&self) -> Result<u32, String> {
        let header = self.header(0)?;
        Ok(ogg_crc(&[&header, &self.lacing, &self.body]))
    }

    fn write_original<W: Write>(&self, writer: &mut W) -> Result<(), String> {
        self.write_with_checksum(writer, self.checksum)
    }

    fn write_recomputed<W: Write>(&self, writer: &mut W) -> Result<(), String> {
        let checksum = self.calculated_checksum()?;
        self.write_with_checksum(writer, checksum)
    }

    fn write_with_checksum<W: Write>(&self, writer: &mut W, checksum: u32) -> Result<(), String> {
        let header = self.header(checksum)?;
        writer
            .write_all(&header)
            .and_then(|()| writer.write_all(&self.lacing))
            .and_then(|()| writer.write_all(&self.body))
            .map_err(|error| format!("write Ogg page: {error}"))
    }
}

struct PageReader<R> {
    inner: R,
    pages: u64,
}

impl<R: Read> PageReader<R> {
    fn new(inner: R) -> Self {
        Self { inner, pages: 0 }
    }

    fn next_page(&mut self) -> Result<Option<Page>, String> {
        let mut header = [0_u8; HEADER_LEN];
        let Some(()) = read_exact_or_eof(&mut self.inner, &mut header[..4])
            .map_err(|error| format!("read Ogg capture pattern: {error}"))?
        else {
            return Ok(None);
        };
        self.inner
            .read_exact(&mut header[4..])
            .map_err(|error| format!("truncated Ogg page header: {error}"))?;
        if &header[..4] != CAPTURE {
            return Err("invalid Ogg capture pattern".into());
        }
        if header[4] != 0 {
            return Err(format!("unsupported Ogg stream version {}", header[4]));
        }
        if header[5] & !(CONTINUED | BOS | EOS) != 0 {
            return Err(format!("invalid Ogg header type 0x{:02x}", header[5]));
        }

        let segment_count = usize::from(header[26]);
        let mut lacing = Vec::new();
        lacing
            .try_reserve_exact(segment_count)
            .map_err(|error| format!("reserve Ogg lacing table: {error}"))?;
        lacing.resize(segment_count, 0);
        self.inner
            .read_exact(&mut lacing)
            .map_err(|error| format!("truncated Ogg lacing table: {error}"))?;
        let body_len = lacing.iter().try_fold(0_usize, |total, lace| {
            total
                .checked_add(usize::from(*lace))
                .ok_or_else(|| "Ogg page body size overflow".to_owned())
        })?;
        if body_len > MAX_PAGE_BODY {
            return Err("Ogg page body exceeds format limit".into());
        }
        let mut body = Vec::new();
        body.try_reserve_exact(body_len)
            .map_err(|error| format!("reserve Ogg page body: {error}"))?;
        body.resize(body_len, 0);
        self.inner
            .read_exact(&mut body)
            .map_err(|error| format!("truncated Ogg page body: {error}"))?;

        let stored_checksum = u32::from_le_bytes(
            header[22..26]
                .try_into()
                .map_err(|_| "invalid Ogg checksum".to_owned())?,
        );
        header[22..26].fill(0);
        let calculated = ogg_crc(&[&header, &lacing, &body]);
        if calculated != stored_checksum {
            return Err(format!(
                "Ogg CRC mismatch on page {} (stored {stored_checksum:08x}, calculated {calculated:08x})",
                self.pages
            ));
        }

        let page = Page {
            header_type: header[5],
            granule: u64::from_le_bytes(
                header[6..14]
                    .try_into()
                    .map_err(|_| "invalid Ogg granule position".to_owned())?,
            ),
            serial: u32::from_le_bytes(
                header[14..18]
                    .try_into()
                    .map_err(|_| "invalid Ogg serial number".to_owned())?,
            ),
            sequence: u32::from_le_bytes(
                header[18..22]
                    .try_into()
                    .map_err(|_| "invalid Ogg page sequence".to_owned())?,
            ),
            checksum: stored_checksum,
            lacing,
            body,
        };
        self.pages = self
            .pages
            .checked_add(1)
            .ok_or_else(|| "Ogg page count overflow".to_owned())?;
        Ok(Some(page))
    }
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buffer: &mut [u8]) -> io::Result<Option<()>> {
    let mut filled = 0_usize;
    while filled < buffer.len() {
        let count = reader.read(&mut buffer[filled..])?;
        if count == 0 {
            if filled == 0 {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "partial Ogg capture pattern",
            ));
        }
        filled += count;
    }
    Ok(Some(()))
}

const fn crc_table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut index = 0_usize;
    while index < table.len() {
        let mut value = (index as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 0x8000_0000 != 0 {
                (value << 1) ^ 0x04c1_1db7
            } else {
                value << 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

const CRC_TABLE: [u32; 256] = crc_table();

fn ogg_crc(parts: &[&[u8]]) -> u32 {
    let mut crc = 0_u32;
    for part in parts {
        for byte in *part {
            let index = usize::from(((crc >> 24) as u8) ^ *byte);
            crc = (crc << 8) ^ CRC_TABLE[index];
        }
    }
    crc
}

#[derive(Debug)]
struct StreamState {
    expected_sequence: u32,
    packet_open: bool,
    packet_len: usize,
    packet_pages: usize,
    packet_buffer: Vec<u8>,
    packet_index: usize,
    codec: Option<Codec>,
    bos_page: u64,
    first_packet_page: Option<u64>,
    target: bool,
    eos: bool,
}

impl StreamState {
    fn new(bos_page: u64) -> Self {
        Self {
            expected_sequence: 0,
            packet_open: false,
            packet_len: 0,
            packet_pages: 0,
            packet_buffer: Vec::new(),
            packet_index: 0,
            codec: None,
            bos_page,
            first_packet_page: None,
            target: false,
            eos: false,
        }
    }

    fn should_collect(&self) -> bool {
        self.packet_index == 0
            || (self.codec.is_some() && self.packet_index == 1)
            || (self.codec == Some(Codec::Vorbis) && self.packet_index == 2)
    }
}

#[derive(Debug)]
struct PendingTarget {
    serial: u32,
    bos_page: u64,
    codec: Codec,
    comments: Option<VorbisCommentsSnapshot>,
    comment_seen: bool,
    tail: Vec<u8>,
    header_end_page: Option<u64>,
    eos_page: Option<u64>,
    shares_header_page_with_audio: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanMode {
    /// Stop after the selected codec's complete header set has been checked.
    HeadersOnly,
    /// Check every page and packet needed by a decoder, without requiring EOS.
    Decode,
    /// Check the complete container and require every logical stream to end.
    Rewrite,
}

fn scan<R: Read>(
    reader: &mut R,
    limits: &MetadataLimits,
    budget: &mut MetadataBudget,
    mode: ScanMode,
) -> Result<Scan, String> {
    let mut pages = PageReader::new(reader);
    let mut streams: HashMap<u32, StreamState> = HashMap::new();
    streams
        .try_reserve(limits.max_ogg_streams.min(64))
        .map_err(|error| format!("reserve Ogg stream state: {error}"))?;
    let mut stream_count = 0_usize;
    let mut target: Option<PendingTarget> = None;
    let mut page_index = 0_u64;
    let mut buffered_header_bytes = 0_usize;
    let retain_comments =
        limits.max_total_bytes != 0 && limits.max_item_bytes != 0 && limits.max_items != 0;
    let header_buffer_limit = if retain_comments {
        limits.max_ogg_packet_bytes.min(limits.max_total_bytes)
    } else {
        // Codec identification, comment framing, and setup headers remain
        // mandatory even when the caller has no budget for retained optional
        // metadata. Their transient packet assembly is bounded structurally.
        limits.max_ogg_packet_bytes
    };
    let mut stopped_after_headers = false;

    while let Some(page) = pages.next_page()? {
        if page.is_bos() {
            if page.is_continued() {
                return Err(format!(
                    "Ogg BOS page {page_index} cannot continue a packet"
                ));
            }
            if page.sequence != 0 {
                return Err(format!(
                    "Ogg BOS page {page_index} starts at sequence {} instead of 0",
                    page.sequence
                ));
            }
            if let Some(previous) = streams.get(&page.serial) {
                if !previous.eos {
                    return Err(format!(
                        "duplicate Ogg BOS for active serial {} on page {page_index}",
                        page.serial
                    ));
                }
            }
            stream_count = stream_count
                .checked_add(1)
                .ok_or_else(|| "Ogg logical stream count overflow".to_owned())?;
            if stream_count > limits.max_ogg_streams {
                return Err(format!(
                    "Ogg logical stream count exceeds limit ({} > {})",
                    stream_count, limits.max_ogg_streams
                ));
            }
            if !streams.contains_key(&page.serial) {
                streams
                    .try_reserve(1)
                    .map_err(|error| format!("reserve Ogg stream state: {error}"))?;
            }
            streams.insert(page.serial, StreamState::new(page_index));
        } else if !streams.contains_key(&page.serial) {
            return Err(format!(
                "Ogg page {page_index} for serial {} appears before BOS",
                page.serial
            ));
        }

        let state = streams
            .get_mut(&page.serial)
            .ok_or_else(|| "missing Ogg stream state".to_owned())?;
        if state.eos {
            return Err(format!(
                "Ogg page {page_index} follows EOS for serial {}",
                page.serial
            ));
        }
        if page.sequence != state.expected_sequence {
            return Err(format!(
                "Ogg sequence mismatch for serial {} on page {page_index}: expected {}, got {}",
                page.serial, state.expected_sequence, page.sequence
            ));
        }
        state.expected_sequence = state.expected_sequence.wrapping_add(1);
        if page.is_continued() != state.packet_open {
            return Err(format!(
                "Ogg continuation flag mismatch for serial {} on page {page_index}",
                page.serial
            ));
        }

        let mut body_offset = 0_usize;
        let mut packet_on_page = state.packet_open;
        if packet_on_page {
            state.packet_pages = state
                .packet_pages
                .checked_add(1)
                .ok_or_else(|| "Ogg packet page count overflow".to_owned())?;
            check_packet_pages(state.packet_pages, limits)?;
        }

        for (segment_index, lace) in page.lacing.iter().copied().enumerate() {
            if !packet_on_page {
                state.packet_pages = 1;
                check_packet_pages(state.packet_pages, limits)?;
                packet_on_page = true;
                if state.packet_index == 0 && state.first_packet_page.is_none() {
                    state.first_packet_page = Some(page_index);
                }
            }
            let length = usize::from(lace);
            let body_end = body_offset
                .checked_add(length)
                .ok_or_else(|| "Ogg page body offset overflow".to_owned())?;
            let segment = page
                .body
                .get(body_offset..body_end)
                .ok_or_else(|| "Ogg lacing exceeds page body".to_owned())?;
            body_offset = body_end;
            state.packet_len = state
                .packet_len
                .checked_add(length)
                .ok_or_else(|| "Ogg packet size overflow".to_owned())?;
            if state.packet_len > limits.max_ogg_packet_bytes {
                return Err(format!(
                    "Ogg packet exceeds limit ({} > {} bytes)",
                    state.packet_len, limits.max_ogg_packet_bytes
                ));
            }
            if state.should_collect() {
                let next_buffered = buffered_header_bytes
                    .checked_add(length)
                    .ok_or_else(|| "Ogg header buffer size overflow".to_owned())?;
                if next_buffered > header_buffer_limit {
                    return Err(format!(
                        "Ogg simultaneous header buffers exceed allocation limit ({} > {} bytes)",
                        next_buffered, header_buffer_limit
                    ));
                }
                if retain_comments && state.target && state.packet_index == 1 {
                    budget.check_bytes(state.packet_len, "Ogg comment packet")?;
                }
                state
                    .packet_buffer
                    .try_reserve(length)
                    .map_err(|error| format!("reserve Ogg header packet: {error}"))?;
                state.packet_buffer.extend_from_slice(segment);
                buffered_header_bytes = next_buffered;
            }

            state.packet_open = lace == 255;
            if state.packet_open {
                continue;
            }

            let packet = std::mem::take(&mut state.packet_buffer);
            if state.packet_index == 0 {
                let codec = identify_codec(&packet, limits)?;
                state.codec = codec;
                if let Some(codec) = codec {
                    if state.first_packet_page != Some(state.bos_page) {
                        return Err(format!(
                            "Ogg codec identification for serial {} did not begin on its BOS page",
                            page.serial
                        ));
                    }
                    if target.is_none() {
                        state.target = true;
                        target = Some(PendingTarget {
                            serial: page.serial,
                            bos_page: state.bos_page,
                            codec,
                            comments: None,
                            comment_seen: false,
                            tail: Vec::new(),
                            header_end_page: None,
                            eos_page: None,
                            shares_header_page_with_audio: false,
                        });
                    }
                }
            } else if state.packet_index == 1 {
                if let Some(codec) = state.codec {
                    let parts = comment_parts(codec, &packet, limits, retain_comments)?;
                    if state.target {
                        let pending = target
                            .as_mut()
                            .ok_or_else(|| "missing Ogg target state".to_owned())?;
                        pending.comment_seen = true;
                        if retain_comments {
                            let comments = parse_comment_body(parts.body, budget)?;
                            budget.charge_bytes(parts.tail.len(), "OpusTags tail")?;
                            pending
                                .tail
                                .try_reserve_exact(parts.tail.len())
                                .map_err(|error| format!("reserve Ogg comment tail: {error}"))?;
                            pending.tail.extend_from_slice(parts.tail);
                            pending.comments = Some(comments);
                        } else {
                            validate_tagless_comment(parts.body, parts.tail)?;
                        }
                    }
                }
            } else if state.packet_index == 2
                && state.codec == Some(Codec::Vorbis)
                && !packet.starts_with(b"\x05vorbis")
            {
                return Err("invalid Vorbis setup packet prefix".into());
            }

            state.packet_index = state
                .packet_index
                .checked_add(1)
                .ok_or_else(|| "Ogg packet count overflow".to_owned())?;
            if let Some(codec) = state.codec {
                if state.packet_index == codec.header_packets() {
                    if state.target {
                        let pending = target
                            .as_mut()
                            .ok_or_else(|| "missing Ogg target state".to_owned())?;
                        pending.header_end_page = Some(page_index);
                        pending.shares_header_page_with_audio =
                            segment_index + 1 != page.lacing.len();
                    }
                }
            }
            buffered_header_bytes = buffered_header_bytes
                .checked_sub(packet.len())
                .ok_or_else(|| "Ogg header buffer accounting underflow".to_owned())?;
            state.packet_len = 0;
            state.packet_pages = 0;
            packet_on_page = false;
        }
        if body_offset != page.body.len() {
            return Err("Ogg page body has unreferenced bytes".into());
        }

        if page.is_eos() {
            if state.packet_open {
                return Err(format!(
                    "Ogg EOS page {page_index} ends with an unfinished packet"
                ));
            }
            if let Some(codec) = state.codec {
                if state.packet_index < codec.header_packets() {
                    return Err(format!(
                        "Ogg stream {} ends before all codec headers",
                        page.serial
                    ));
                }
            }
            state.eos = true;
            if state.target {
                let pending = target
                    .as_mut()
                    .ok_or_else(|| "missing Ogg target state".to_owned())?;
                pending.eos_page = Some(page_index);
            }
        }

        if target.as_ref().is_none_or(|pending| !pending.comment_seen) {
            let inspected = usize::try_from(page_index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .unwrap_or(usize::MAX);
            if inspected > limits.max_ogg_pages {
                return Err(format!(
                    "Ogg metadata search exceeds page limit ({} > {})",
                    inspected, limits.max_ogg_pages
                ));
            }
        }
        if let Some(pending) = &target {
            let last_header_page = pending.header_end_page.unwrap_or(page_index);
            let header_pages = last_header_page
                .checked_sub(pending.bos_page)
                .and_then(|count| count.checked_add(1))
                .unwrap_or(u64::MAX);
            if header_pages > limits.max_ogg_pages as u64 {
                return Err(format!(
                    "Ogg codec headers exceed page limit ({} > {})",
                    header_pages, limits.max_ogg_pages
                ));
            }
        }

        page_index = page_index
            .checked_add(1)
            .ok_or_else(|| "Ogg page index overflow".to_owned())?;
        if mode == ScanMode::HeadersOnly
            && target
                .as_ref()
                .is_some_and(|pending| pending.header_end_page.is_some())
        {
            stopped_after_headers = true;
            break;
        }
    }

    if page_index == 0 {
        return Err("empty Ogg stream".into());
    }
    for (serial, state) in &streams {
        if !stopped_after_headers && state.packet_open {
            return Err(format!(
                "truncated Ogg packet at end of stream for serial {serial}"
            ));
        }
        if mode == ScanMode::Rewrite && !state.eos {
            return Err(format!("Ogg stream {serial} is missing EOS"));
        }
    }

    let target = target
        .map(|pending| {
            if !pending.comment_seen {
                return Err("Ogg codec stream is missing its comment header".to_owned());
            }
            let header_end_page = pending
                .header_end_page
                .ok_or_else(|| "Ogg codec stream is missing required headers".to_owned())?;
            Ok::<Target, String>(Target {
                serial: pending.serial,
                bos_page: pending.bos_page,
                header_end_page,
                eos_page: pending.eos_page,
                codec: pending.codec,
                comments: pending.comments,
                tail: pending.tail,
                shares_header_page_with_audio: pending.shares_header_page_with_audio,
            })
        })
        .transpose()?;
    Ok(Scan { target })
}

fn check_packet_pages(pages: usize, limits: &MetadataLimits) -> Result<(), String> {
    if pages > limits.max_ogg_pages {
        return Err(format!(
            "continued Ogg packet exceeds page limit ({} > {})",
            pages, limits.max_ogg_pages
        ));
    }
    Ok(())
}

fn identify_codec(packet: &[u8], limits: &MetadataLimits) -> Result<Option<Codec>, String> {
    if packet.starts_with(b"\x01vorbis") {
        if packet.len() != 30 {
            return Err(format!(
                "invalid Vorbis identification header length {}",
                packet.len()
            ));
        }
        if packet[7..11] != [0, 0, 0, 0]
            || packet[11] == 0
            || packet[12..16] == [0, 0, 0, 0]
            || packet[29] != 1
        {
            return Err("invalid Vorbis identification header".into());
        }
        let small = packet[28] & 0x0f;
        let large = packet[28] >> 4;
        if !(6..=13).contains(&small) || !(6..=13).contains(&large) || small > large {
            return Err("invalid Vorbis block sizes".into());
        }
        return Ok(Some(Codec::Vorbis));
    }
    if packet.starts_with(b"OpusHead") {
        if packet.len() < 19 || !(1..=15).contains(&packet[8]) || packet[9] == 0 {
            return Err("invalid Opus identification header".into());
        }
        return Ok(Some(Codec::Opus));
    }
    if packet.starts_with(b"Speex   ") {
        if packet.len() < 80 {
            return Err("truncated Speex identification header".into());
        }
        let header_size = read_i32_at(packet, 32, "Speex header size")?;
        if header_size < 80 || usize::try_from(header_size).ok() != Some(packet.len()) {
            return Err("invalid Speex identification header size".into());
        }
        let extra_headers = read_i32_at(packet, 68, "Speex extra-header count")?;
        let extra_headers = usize::try_from(extra_headers)
            .map_err(|_| "negative Speex extra-header count".to_owned())?;
        let header_packets = extra_headers
            .checked_add(2)
            .ok_or_else(|| "Speex header count overflow".to_owned())?;
        if header_packets > limits.max_items || header_packets > limits.max_ogg_pages {
            return Err("Speex header count exceeds metadata limits".into());
        }
        return Ok(Some(Codec::Speex { header_packets }));
    }
    Ok(None)
}

fn read_i32_at(data: &[u8], offset: usize, context: &str) -> Result<i32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| format!("{context} offset overflow"))?;
    let bytes = data
        .get(offset..end)
        .ok_or_else(|| format!("truncated {context}"))?;
    Ok(i32::from_le_bytes(
        bytes.try_into().map_err(|_| format!("invalid {context}"))?,
    ))
}

struct CommentParts<'a> {
    body: &'a [u8],
    tail: &'a [u8],
}

fn comment_parts<'a>(
    codec: Codec,
    packet: &'a [u8],
    limits: &MetadataLimits,
    retain_comments: bool,
) -> Result<CommentParts<'a>, String> {
    let (body_start, require_framing, allow_tail) = match codec {
        Codec::Vorbis => {
            if !packet.starts_with(b"\x03vorbis") {
                return Err("invalid Vorbis comment packet prefix".into());
            }
            (7, true, false)
        }
        Codec::Opus => {
            if !packet.starts_with(b"OpusTags") {
                return Err("invalid OpusTags packet prefix".into());
            }
            (8, false, true)
        }
        Codec::Speex { .. } => (0, false, false),
    };
    let payload = packet
        .get(body_start..)
        .ok_or_else(|| "truncated Ogg comment packet".to_owned())?;
    let body_len = if retain_comments {
        locate_comment_body(payload, limits)?
    } else {
        locate_tagless_comment_body(payload)?
    };
    let body_end = body_start
        .checked_add(body_len)
        .ok_or_else(|| "Ogg comment body offset overflow".to_owned())?;
    let body = packet
        .get(body_start..body_end)
        .ok_or_else(|| "truncated Ogg comment body".to_owned())?;
    let remainder = packet
        .get(body_end..)
        .ok_or_else(|| "invalid Ogg comment tail".to_owned())?;
    if require_framing {
        if remainder != [1] {
            return Err("Vorbis comment packet has invalid framing byte".into());
        }
        return Ok(CommentParts { body, tail: &[] });
    }
    if !allow_tail && !remainder.is_empty() {
        return Err("Speex comment packet has trailing bytes".into());
    }
    Ok(CommentParts {
        body,
        tail: remainder,
    })
}

fn locate_tagless_comment_body(data: &[u8]) -> Result<usize, String> {
    let mut offset = 0_usize;
    let vendor_len = read_u32(&mut offset, data, "Vorbis vendor length")?;
    let vendor_end = offset
        .checked_add(vendor_len)
        .ok_or_else(|| "Vorbis vendor length overflow".to_owned())?;
    let vendor = data
        .get(offset..vendor_end)
        .ok_or_else(|| "truncated Vorbis vendor".to_owned())?;
    std::str::from_utf8(vendor).map_err(|error| format!("Vorbis vendor is not UTF-8: {error}"))?;
    offset = vendor_end;
    let item_count = read_u32(&mut offset, data, "Vorbis comment count")?;
    if item_count != 0 {
        return Err("Vorbis fields exceed zero retained metadata budget".into());
    }
    Ok(offset)
}

fn validate_tagless_comment(body: &[u8], tail: &[u8]) -> Result<(), String> {
    if !tail.is_empty() {
        return Err("OpusTags tail exceeds zero retained metadata budget".into());
    }
    let mut offset = 0_usize;
    let vendor_len = read_u32(&mut offset, body, "Vorbis vendor length")?;
    let vendor_end = offset
        .checked_add(vendor_len)
        .ok_or_else(|| "Vorbis vendor length overflow".to_owned())?;
    let vendor = body
        .get(offset..vendor_end)
        .ok_or_else(|| "truncated Vorbis vendor".to_owned())?;
    std::str::from_utf8(vendor).map_err(|error| format!("Vorbis vendor is not UTF-8: {error}"))?;
    offset = vendor_end;
    let item_count = read_u32(&mut offset, body, "Vorbis comment count")?;
    if item_count != 0 {
        return Err("Vorbis fields exceed zero retained metadata budget".into());
    }
    if offset != body.len() {
        return Err(format!(
            "Vorbis comment body has {} trailing bytes",
            body.len() - offset
        ));
    }
    Ok(())
}

fn locate_comment_body(data: &[u8], limits: &MetadataLimits) -> Result<usize, String> {
    let mut offset = 0_usize;
    let vendor_len = read_u32(&mut offset, data, "Vorbis vendor length")?;
    if vendor_len > limits.max_item_bytes {
        return Err(format!(
            "Vorbis vendor exceeds metadata item limit ({} > {} bytes)",
            vendor_len, limits.max_item_bytes
        ));
    }
    offset = offset
        .checked_add(vendor_len)
        .ok_or_else(|| "Vorbis vendor length overflow".to_owned())?;
    if offset > data.len() {
        return Err("truncated Vorbis vendor".into());
    }
    let count = read_u32(&mut offset, data, "Vorbis comment count")?;
    if count > limits.max_items {
        return Err(format!(
            "Vorbis comment count exceeds metadata item count limit ({} > {})",
            count, limits.max_items
        ));
    }
    for _ in 0..count {
        let length = read_u32(&mut offset, data, "Vorbis comment length")?;
        if length > limits.max_item_bytes {
            return Err(format!(
                "Vorbis comment exceeds metadata item limit ({} > {} bytes)",
                length, limits.max_item_bytes
            ));
        }
        offset = offset
            .checked_add(length)
            .ok_or_else(|| "Vorbis comment length overflow".to_owned())?;
        if offset > data.len() {
            return Err("truncated Vorbis comment".into());
        }
    }
    Ok(offset)
}

fn read_u32(offset: &mut usize, data: &[u8], context: &str) -> Result<usize, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| format!("{context} offset overflow"))?;
    let bytes = data
        .get(*offset..end)
        .ok_or_else(|| format!("truncated {context}"))?;
    *offset = end;
    usize::try_from(u32::from_le_bytes(
        bytes.try_into().map_err(|_| format!("invalid {context}"))?,
    ))
    .map_err(|_| format!("{context} does not fit in memory"))
}

fn rewrite_scanned(
    output: &mut File,
    scan: &Scan,
    target: &Target,
    body: Vec<u8>,
    limits: &MetadataLimits,
) -> Result<(), String> {
    debug_assert_eq!(
        scan.target.as_ref().map(|candidate| candidate.bos_page),
        Some(target.bos_page)
    );
    let eos_page = target
        .eos_page
        .ok_or_else(|| "Ogg rewrite target is missing EOS".to_owned())?;
    let original_comments = target
        .comments
        .as_ref()
        .ok_or_else(|| "Ogg rewrite target comments were not retained".to_owned())?;
    let original_comment_len = target
        .codec
        .comment_prefix_len()
        .checked_add(comment_body_len(original_comments)?)
        .and_then(|size| size.checked_add(target.tail.len()))
        .and_then(|size| size.checked_add(usize::from(target.codec == Codec::Vorbis)))
        .ok_or_else(|| "original Ogg comment packet size overflow".to_owned())?;
    let target_retained_bytes = comment_body_len(original_comments)?
        .checked_add(target.tail.len())
        .ok_or_else(|| "retained Ogg target size overflow".to_owned())?;
    let serialized_and_target = body
        .len()
        .checked_add(target_retained_bytes)
        .ok_or_else(|| "Ogg serialized and retained target size overflow".to_owned())?;
    check_rewrite_allocation(
        &[serialized_and_target],
        limits,
        "Ogg serialized body and retained target",
    )?;
    let pages = collect_header_pages(output, target, limits, serialized_and_target)?;
    let page_bytes = pages.iter().try_fold(0_usize, |total, page| {
        total
            .checked_add(page.encoded_len())
            .ok_or_else(|| "Ogg header page aggregate overflow".to_owned())
    })?;
    let serialized_target_and_pages = serialized_and_target
        .checked_add(page_bytes)
        .ok_or_else(|| "Ogg retained rewrite allocation overflow".to_owned())?;
    let (mut packets, capacities, header_had_eos) =
        extract_header_packets(&pages, target, limits, serialized_target_and_pages)?;
    // Extraction duplicates the target header bodies. Release the encoded
    // pages before allocating the replacement and generated page set so each
    // phase is bounded by the aggregate limit instead of retaining three
    // representations at once.
    drop(pages);
    let packet_bytes = packets.iter().try_fold(0_usize, |total, packet| {
        total
            .checked_add(packet.len())
            .ok_or_else(|| "Ogg header packet aggregate overflow".to_owned())
    })?;
    let replacement_len = comment_packet_len(target, &body)?;
    check_rewrite_allocation(
        &[
            body.len(),
            target_retained_bytes,
            packet_bytes,
            replacement_len,
        ],
        limits,
        "Ogg serialized body, target, packets, and replacement",
    )?;
    let replacement = build_comment_packet(target, &body, limits)?;
    drop(body);
    let packet_bytes_after = packet_bytes
        .checked_sub(original_comment_len)
        .and_then(|size| size.checked_add(replacement.len()))
        .ok_or_else(|| "rewritten Ogg header packet aggregate overflow".to_owned())?;
    let comment = packets
        .get_mut(1)
        .ok_or_else(|| "Ogg codec stream has no comment packet".to_owned())?;
    if *comment == replacement {
        return Ok(());
    }
    *comment = replacement;
    let generated = paginate_headers(
        &packets,
        &capacities,
        target.serial,
        header_had_eos,
        limits,
        packet_bytes_after
            .checked_add(target_retained_bytes)
            .ok_or_else(|| "Ogg retained rewrite allocation overflow".to_owned())?,
    )?;
    let generated_bytes = generated.iter().try_fold(0_usize, |total, page| {
        total
            .checked_add(page.encoded_len())
            .ok_or_else(|| "rewritten Ogg page aggregate overflow".to_owned())
    })?;
    check_rewrite_allocation(
        &[target_retained_bytes, packet_bytes_after, generated_bytes],
        limits,
        "Ogg extracted header packets and generated pages",
    )?;
    let old_header_pages = capacities.len();
    drop(capacities);
    drop(packets);
    stream_rewritten_file(output, target, eos_page, old_header_pages, &generated)
}

fn comment_body_len(snapshot: &VorbisCommentsSnapshot) -> Result<usize, String> {
    let mut length = 4_usize
        .checked_add(snapshot.vendor.len())
        .and_then(|size| size.checked_add(4))
        .ok_or_else(|| "Vorbis comment body size overflow".to_owned())?;
    for (key, value) in &snapshot.items {
        length = length
            .checked_add(4)
            .and_then(|size| size.checked_add(key.len()))
            .and_then(|size| size.checked_add(1))
            .and_then(|size| size.checked_add(value.len()))
            .ok_or_else(|| "Vorbis comment body size overflow".to_owned())?;
    }
    Ok(length)
}

fn check_rewrite_allocation(
    amounts: &[usize],
    limits: &MetadataLimits,
    context: &str,
) -> Result<(), String> {
    let total = amounts.iter().try_fold(0_usize, |total, amount| {
        total
            .checked_add(*amount)
            .ok_or_else(|| format!("{context} size overflow"))
    })?;
    if total > limits.max_total_bytes {
        return Err(format!(
            "{context} exceed metadata aggregate limit ({} > {} bytes)",
            total, limits.max_total_bytes
        ));
    }
    Ok(())
}

fn build_comment_packet(
    target: &Target,
    body: &[u8],
    limits: &MetadataLimits,
) -> Result<Vec<u8>, String> {
    let prefix: &[u8] = match target.codec {
        Codec::Vorbis => b"\x03vorbis",
        Codec::Opus => b"OpusTags",
        Codec::Speex { .. } => &[],
    };
    let length = comment_packet_len(target, body)?;
    if length > limits.max_ogg_packet_bytes {
        return Err(format!(
            "Ogg comment packet exceeds limit ({} > {} bytes)",
            length, limits.max_ogg_packet_bytes
        ));
    }
    let mut packet = Vec::new();
    packet
        .try_reserve_exact(length)
        .map_err(|error| format!("reserve Ogg comment packet: {error}"))?;
    packet.extend_from_slice(prefix);
    packet.extend_from_slice(body);
    packet.extend_from_slice(&target.tail);
    if target.codec == Codec::Vorbis {
        packet.push(1);
    }
    Ok(packet)
}

fn comment_packet_len(target: &Target, body: &[u8]) -> Result<usize, String> {
    target
        .codec
        .comment_prefix_len()
        .checked_add(body.len())
        .and_then(|size| size.checked_add(target.tail.len()))
        .and_then(|size| size.checked_add(usize::from(target.codec == Codec::Vorbis)))
        .ok_or_else(|| "Ogg comment packet size overflow".to_owned())
}

fn collect_header_pages(
    output: &mut File,
    target: &Target,
    limits: &MetadataLimits,
    retained_bytes: usize,
) -> Result<Vec<Page>, String> {
    output
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind Ogg metadata: {error}"))?;
    let mut reader = PageReader::new(&mut *output);
    let mut pages = Vec::new();
    let mut buffered = 0_usize;
    let buffer_limit = limits.max_total_bytes;
    let mut page_index = 0_u64;
    while let Some(page) = reader.next_page()? {
        if page_index >= target.bos_page
            && page_index <= target.header_end_page
            && page.serial == target.serial
        {
            if pages.len() >= limits.max_ogg_pages {
                return Err("Ogg header window exceeds page limit".into());
            }
            buffered = buffered
                .checked_add(page.encoded_len())
                .ok_or_else(|| "Ogg header window size overflow".to_owned())?;
            let simultaneous = retained_bytes
                .checked_add(buffered)
                .ok_or_else(|| "Ogg header window allocation overflow".to_owned())?;
            if simultaneous > buffer_limit {
                return Err(format!(
                    "Ogg retained data and header window exceed allocation limit ({} > {} bytes)",
                    simultaneous, buffer_limit
                ));
            }
            pages
                .try_reserve(1)
                .map_err(|error| format!("reserve Ogg header pages: {error}"))?;
            pages.push(page);
        }
        if page_index == target.header_end_page {
            break;
        }
        page_index = page_index
            .checked_add(1)
            .ok_or_else(|| "Ogg page index overflow".to_owned())?;
    }
    if pages.is_empty() {
        return Err("Ogg target header pages disappeared during rewrite".into());
    }
    Ok(pages)
}

fn extract_header_packets(
    pages: &[Page],
    target: &Target,
    limits: &MetadataLimits,
    retained_bytes: usize,
) -> Result<(Vec<Vec<u8>>, Vec<usize>, bool), String> {
    let header_had_eos = pages.last().is_some_and(Page::is_eos);
    let mut packets: Vec<Vec<u8>> = Vec::new();
    packets
        .try_reserve_exact(target.codec.header_packets())
        .map_err(|error| format!("reserve Ogg header packets: {error}"))?;
    let mut capacities = Vec::new();
    capacities
        .try_reserve_exact(pages.len())
        .map_err(|error| format!("reserve Ogg page capacities: {error}"))?;
    let mut packet = Vec::new();
    let mut packet_open = false;
    let mut aggregate = 0_usize;
    let aggregate_limit = limits.max_total_bytes;

    for (page_offset, page) in pages.iter().enumerate() {
        if page.serial != target.serial {
            return Err("non-target serial entered Ogg header packet set".into());
        }
        if page_offset == 0 {
            if !page.is_bos() || page.is_continued() || page.sequence != 0 {
                return Err("invalid first Ogg target header page".into());
            }
        } else if page.is_bos() {
            return Err("unexpected BOS inside Ogg header window".into());
        }
        if page.is_continued() != packet_open {
            return Err("Ogg header continuation mismatch".into());
        }
        if page.lacing.is_empty() {
            return Err("empty Ogg page inside codec header window".into());
        }
        let completed = page.completed_packets();
        if (completed == 0 && page.granule != NO_GRANULE) || (completed != 0 && page.granule != 0) {
            return Err("invalid granule position on Ogg codec header page".into());
        }
        capacities.push(page.lacing.len());
        let mut body_offset = 0_usize;
        for lace in &page.lacing {
            let length = usize::from(*lace);
            let end = body_offset
                .checked_add(length)
                .ok_or_else(|| "Ogg header body offset overflow".to_owned())?;
            let bytes = page
                .body
                .get(body_offset..end)
                .ok_or_else(|| "Ogg header lacing exceeds body".to_owned())?;
            body_offset = end;
            let next_len = packet
                .len()
                .checked_add(length)
                .ok_or_else(|| "Ogg header packet size overflow".to_owned())?;
            if next_len > limits.max_ogg_packet_bytes {
                return Err("Ogg header packet exceeds packet limit".into());
            }
            check_rewrite_allocation(
                &[retained_bytes, aggregate, next_len],
                limits,
                "Ogg retained header pages and extracted packets",
            )?;
            packet
                .try_reserve(length)
                .map_err(|error| format!("reserve Ogg header packet: {error}"))?;
            packet.extend_from_slice(bytes);
            packet_open = *lace == 255;
            if !packet_open {
                aggregate = aggregate
                    .checked_add(packet.len())
                    .ok_or_else(|| "Ogg header allocation overflow".to_owned())?;
                if aggregate > aggregate_limit {
                    return Err("Ogg header packets exceed allocation limit".into());
                }
                packets
                    .try_reserve(1)
                    .map_err(|error| format!("reserve Ogg header packet list: {error}"))?;
                packets.push(std::mem::take(&mut packet));
            }
        }
        if body_offset != page.body.len() {
            return Err("Ogg header page has unreferenced bytes".into());
        }
    }
    if packet_open || !packet.is_empty() {
        return Err("Ogg codec header window ends mid-packet".into());
    }
    if packets.len() != target.codec.header_packets() {
        return Err(format!(
            "Ogg codec header count changed during rewrite ({} != {})",
            packets.len(),
            target.codec.header_packets()
        ));
    }
    Ok((packets, capacities, header_had_eos))
}

#[derive(Clone, Copy)]
struct PacketSegmentCursor<'a> {
    packets: &'a [Vec<u8>],
    packet: usize,
    offset: usize,
}

impl<'a> PacketSegmentCursor<'a> {
    fn new(packets: &'a [Vec<u8>]) -> Self {
        Self {
            packets,
            packet: 0,
            offset: 0,
        }
    }

    fn next(&mut self) -> Option<(u8, &'a [u8])> {
        let packet = self.packets.get(self.packet)?;
        if self.offset == packet.len() {
            self.packet += 1;
            self.offset = 0;
            return Some((0, &[]));
        }
        let remaining = packet.len() - self.offset;
        let length = remaining.min(255);
        let start = self.offset;
        self.offset += length;
        if length < 255 {
            self.packet += 1;
            self.offset = 0;
        }
        Some((length as u8, &packet[start..start + length]))
    }
}

fn packet_segment_count(packet: &[u8]) -> Result<usize, String> {
    packet
        .len()
        .checked_div(255)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| "Ogg packet segment count overflow".to_owned())
}

fn paginate_headers(
    packets: &[Vec<u8>],
    capacities: &[usize],
    serial: u32,
    eos: bool,
    limits: &MetadataLimits,
    retained_bytes: usize,
) -> Result<Vec<Page>, String> {
    let total_segments = packets.iter().try_fold(0_usize, |total, packet| {
        total
            .checked_add(packet_segment_count(packet)?)
            .ok_or_else(|| "Ogg header segment count overflow".to_owned())
    })?;
    let mut cursor = PacketSegmentCursor::new(packets);
    let mut generated = Vec::new();
    let mut generated_bytes = 0_usize;
    let mut emitted = 0_usize;
    let mut previous_lace = 0_u8;
    while emitted < total_segments {
        if generated.len() >= limits.max_ogg_pages {
            return Err("rewritten Ogg headers exceed page limit".into());
        }
        let capacity = capacities.get(generated.len()).copied().unwrap_or(255);
        if capacity == 0 || capacity > MAX_SEGMENTS {
            return Err("invalid Ogg header page capacity".into());
        }
        let count = capacity.min(total_segments - emitted);
        let continued = emitted != 0 && previous_lace == 255;
        let mut lacing = Vec::new();
        lacing
            .try_reserve_exact(count)
            .map_err(|error| format!("reserve rewritten Ogg lacing: {error}"))?;
        let mut preview = cursor;
        let page_body_len = (0..count).try_fold(0_usize, |total, _| {
            let (_, bytes) = preview
                .next()
                .ok_or_else(|| "Ogg segment preview ended early".to_owned())?;
            total
                .checked_add(bytes.len())
                .ok_or_else(|| "rewritten Ogg page body size overflow".to_owned())
        })?;
        let encoded_len = HEADER_LEN
            .checked_add(count)
            .and_then(|size| size.checked_add(page_body_len))
            .ok_or_else(|| "rewritten Ogg page size overflow".to_owned())?;
        check_rewrite_allocation(
            &[retained_bytes, generated_bytes, encoded_len],
            limits,
            "Ogg retained headers and generated pages",
        )?;
        let mut page_body = Vec::new();
        page_body
            .try_reserve_exact(page_body_len)
            .map_err(|error| format!("reserve rewritten Ogg page body: {error}"))?;
        let mut completed = false;
        for _ in 0..count {
            let (lace, bytes) = cursor
                .next()
                .ok_or_else(|| "Ogg segment iterator ended early".to_owned())?;
            lacing.push(lace);
            page_body.extend_from_slice(bytes);
            previous_lace = lace;
            completed |= lace < 255;
            emitted += 1;
        }
        let last = emitted == total_segments;
        let mut header_type = 0_u8;
        if continued {
            header_type |= CONTINUED;
        }
        if generated.is_empty() {
            header_type |= BOS;
        }
        if eos && last {
            header_type |= EOS;
        }
        let sequence = u32::try_from(generated.len())
            .map_err(|_| "rewritten Ogg page sequence overflow".to_owned())?;
        let generated_page = Page {
            header_type,
            granule: if completed { 0 } else { NO_GRANULE },
            serial,
            sequence,
            checksum: 0,
            lacing,
            body: page_body,
        };
        generated_bytes = generated_bytes
            .checked_add(generated_page.encoded_len())
            .ok_or_else(|| "rewritten Ogg page aggregate overflow".to_owned())?;
        if generated_bytes > limits.max_total_bytes {
            return Err(format!(
                "rewritten Ogg pages exceed metadata aggregate limit ({} > {} bytes)",
                generated_bytes, limits.max_total_bytes
            ));
        }
        generated
            .try_reserve(1)
            .map_err(|error| format!("reserve rewritten Ogg pages: {error}"))?;
        generated.push(generated_page);
    }
    if cursor.next().is_some() {
        return Err("Ogg segment iterator has trailing data".into());
    }
    Ok(generated)
}

fn stream_rewritten_file(
    output: &mut File,
    target: &Target,
    eos_page: u64,
    old_header_pages: usize,
    generated: &[Page],
) -> Result<(), String> {
    let old_header_pages = u32::try_from(old_header_pages)
        .map_err(|_| "Ogg header page count does not fit sequence numbers".to_owned())?;
    let new_header_pages = u32::try_from(generated.len())
        .map_err(|_| "rewritten Ogg page count does not fit sequence numbers".to_owned())?;
    output
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind Ogg metadata: {error}"))?;
    let mut reader = PageReader::new(&mut *output);
    let mut staged =
        tempfile::tempfile().map_err(|error| format!("create staged Ogg metadata: {error}"))?;
    let mut generated_index = 0_usize;
    let mut page_index = 0_u64;
    while let Some(mut page) = reader.next_page()? {
        let header_slot = page_index >= target.bos_page
            && page_index <= target.header_end_page
            && page.serial == target.serial;
        if header_slot {
            if let Some(replacement) = generated.get(generated_index) {
                replacement.write_recomputed(&mut staged)?;
                generated_index += 1;
            }
            if page_index == target.header_end_page {
                while let Some(replacement) = generated.get(generated_index) {
                    replacement.write_recomputed(&mut staged)?;
                    generated_index += 1;
                }
            }
        } else if page_index > target.header_end_page
            && page_index <= eos_page
            && page.serial == target.serial
            && old_header_pages != new_header_pages
        {
            page.sequence = page
                .sequence
                .wrapping_add(new_header_pages)
                .wrapping_sub(old_header_pages);
            page.write_recomputed(&mut staged)?;
        } else {
            page.write_original(&mut staged)?;
        }
        page_index = page_index
            .checked_add(1)
            .ok_or_else(|| "Ogg page index overflow".to_owned())?;
    }
    if generated_index != generated.len() {
        return Err("not all rewritten Ogg pages were emitted".into());
    }
    staged
        .flush()
        .map_err(|error| format!("flush staged Ogg metadata: {error}"))?;
    staged
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind staged Ogg metadata: {error}"))?;
    output
        .set_len(0)
        .map_err(|error| format!("truncate Ogg metadata: {error}"))?;
    output
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind Ogg output: {error}"))?;
    io::copy(&mut staged, output)
        .map_err(|error| format!("publish staged Ogg metadata: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("flush Ogg metadata: {error}"))?;
    output
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind rewritten Ogg metadata: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn snapshot(vendor: &str, value: &str) -> VorbisCommentsSnapshot {
        VorbisCommentsSnapshot::new(vendor.into(), vec![("X-CUSTOM".into(), value.into())])
    }

    fn comment_body(snapshot: &VorbisCommentsSnapshot) -> Vec<u8> {
        let limits = MetadataLimits::default();
        let mut budget = MetadataBudget::new(limits);
        serialize_comment_body(snapshot, &mut budget).unwrap()
    }

    fn vorbis_identification() -> Vec<u8> {
        let mut packet = vec![0_u8; 30];
        packet[..7].copy_from_slice(b"\x01vorbis");
        packet[11] = 2;
        packet[12..16].copy_from_slice(&48_000_u32.to_le_bytes());
        packet[28] = 0xb8;
        packet[29] = 1;
        packet
    }

    fn opus_identification() -> Vec<u8> {
        let mut packet = vec![0_u8; 19];
        packet[..8].copy_from_slice(b"OpusHead");
        packet[8] = 1;
        packet[9] = 2;
        packet[12..16].copy_from_slice(&48_000_u32.to_le_bytes());
        packet
    }

    fn speex_identification(extra_headers: i32) -> Vec<u8> {
        let mut packet = vec![0_u8; 80];
        packet[..8].copy_from_slice(b"Speex   ");
        packet[32..36].copy_from_slice(&80_i32.to_le_bytes());
        packet[36..40].copy_from_slice(&16_000_i32.to_le_bytes());
        packet[48..52].copy_from_slice(&1_i32.to_le_bytes());
        packet[68..72].copy_from_slice(&extra_headers.to_le_bytes());
        packet
    }

    fn framed_comment(codec: Codec, comments: &VorbisCommentsSnapshot, tail: &[u8]) -> Vec<u8> {
        let body = comment_body(comments);
        let mut packet = Vec::new();
        match codec {
            Codec::Vorbis => packet.extend_from_slice(b"\x03vorbis"),
            Codec::Opus => packet.extend_from_slice(b"OpusTags"),
            Codec::Speex { .. } => {}
        }
        packet.extend_from_slice(&body);
        packet.extend_from_slice(tail);
        if codec == Codec::Vorbis {
            packet.push(1);
        }
        packet
    }

    fn packet_lacing(packet: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut lacing = Vec::new();
        let mut body = Vec::new();
        for chunk in packet.chunks(255) {
            lacing.push(chunk.len() as u8);
            body.extend_from_slice(chunk);
        }
        if packet.len() % 255 == 0 {
            lacing.push(0);
        }
        (lacing, body)
    }

    fn packets_lacing(packets: &[Vec<u8>]) -> (Vec<u8>, Vec<u8>) {
        let mut lacing = Vec::new();
        let mut body = Vec::new();
        for packet in packets {
            let (packet_lacing, packet_body) = packet_lacing(packet);
            lacing.extend(packet_lacing);
            body.extend(packet_body);
        }
        (lacing, body)
    }

    fn page(
        serial: u32,
        sequence: u32,
        header_type: u8,
        granule: u64,
        lacing: Vec<u8>,
        body: Vec<u8>,
    ) -> Vec<u8> {
        let page = Page {
            header_type,
            granule,
            serial,
            sequence,
            checksum: 0,
            lacing,
            body,
        };
        let mut encoded = Vec::new();
        page.write_recomputed(&mut encoded).unwrap();
        encoded
    }

    fn packet_page(
        serial: u32,
        sequence: u32,
        header_type: u8,
        granule: u64,
        packets: &[Vec<u8>],
    ) -> Vec<u8> {
        let (lacing, body) = packets_lacing(packets);
        page(serial, sequence, header_type, granule, lacing, body)
    }

    fn scan_bytes(bytes: &[u8], limits: MetadataLimits) -> Result<Scan, String> {
        let mut budget = MetadataBudget::new(limits);
        scan(
            &mut Cursor::new(bytes),
            &limits,
            &mut budget,
            ScanMode::Rewrite,
        )
    }

    fn decoded_pages(bytes: &[u8]) -> Vec<Page> {
        let mut reader = PageReader::new(Cursor::new(bytes));
        let mut pages = Vec::new();
        while let Some(page) = reader.next_page().unwrap() {
            pages.push(page);
        }
        pages
    }

    fn rewrite_bytes(
        bytes: &[u8],
        comments: &VorbisCommentsSnapshot,
        limits: MetadataLimits,
    ) -> Result<Vec<u8>, String> {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(bytes).unwrap();
        rewrite(&mut file, comments, &limits)?;
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut rewritten = Vec::new();
        file.read_to_end(&mut rewritten).unwrap();
        Ok(rewritten)
    }

    fn header_packets(bytes: &[u8]) -> (Target, Vec<Vec<u8>>) {
        let limits = MetadataLimits::default();
        let scan = scan_bytes(bytes, limits).unwrap();
        let target = scan.target.unwrap();
        let pages: Vec<_> = decoded_pages(bytes)
            .into_iter()
            .enumerate()
            .filter_map(|(index, page)| {
                let index = index as u64;
                (index >= target.bos_page
                    && index <= target.header_end_page
                    && page.serial == target.serial)
                    .then_some(page)
            })
            .collect();
        let page_bytes = pages.iter().map(Page::encoded_len).sum();
        let (packets, _, _) = extract_header_packets(&pages, &target, &limits, page_bytes).unwrap();
        (target, packets)
    }

    #[test]
    fn rewrites_vorbis_multipacket_page_and_keeps_framing() {
        let original = snapshot("destination", "old");
        let packets = vec![
            vorbis_identification(),
            framed_comment(Codec::Vorbis, &original, &[]),
            b"\x05vorbis-setup".to_vec(),
        ];
        let bytes = packet_page(7, 0, BOS | EOS, 0, &packets);
        let replacement = snapshot("source", "retained");
        let rewritten = rewrite_bytes(&bytes, &replacement, MetadataLimits::default()).unwrap();

        let (target, packets) = header_packets(&rewritten);
        assert_eq!(target.comments, Some(replacement.clone()));
        assert_eq!(packets[0], vorbis_identification());
        assert_eq!(packets[2], b"\x05vorbis-setup");
        assert!(packets[1].starts_with(b"\x03vorbis"));
        assert_eq!(packets[1].last(), Some(&1));
    }

    #[test]
    fn opus_exact_255_continuation_shrinks_and_preserves_tail() {
        let tail = b"legal-opus-tail";
        // OpusTags(8) + comment body(vendor + 8) + tail = exactly 255.
        let original = VorbisCommentsSnapshot::new("v".repeat(239 - tail.len()), Vec::new());
        let comment = framed_comment(Codec::Opus, &original, tail);
        assert_eq!(comment.len(), 255);
        let mut bytes = packet_page(9, 0, BOS, 0, &[opus_identification()]);
        bytes.extend(page(
            9,
            1,
            0,
            NO_GRANULE,
            vec![255],
            comment[..255].to_vec(),
        ));
        bytes.extend(page(
            9,
            2,
            CONTINUED | EOS,
            0,
            vec![0],
            comment[255..].to_vec(),
        ));

        let replacement = snapshot("short", "new");
        let rewritten = rewrite_bytes(&bytes, &replacement, MetadataLimits::default()).unwrap();
        let (target, packets) = header_packets(&rewritten);
        assert_eq!(target.comments, Some(replacement.clone()));
        assert!(packets[1].starts_with(b"OpusTags"));
        assert!(packets[1].ends_with(tail));
        assert_eq!(decoded_pages(&rewritten).len(), 2);
    }

    #[test]
    fn speex_comment_is_raw_without_a_fake_prefix() {
        let original = snapshot("speex", "old");
        let codec = Codec::Speex { header_packets: 2 };
        let mut bytes = packet_page(11, 0, BOS, 0, &[speex_identification(0)]);
        bytes.extend(packet_page(
            11,
            1,
            EOS,
            0,
            &[framed_comment(codec, &original, &[])],
        ));
        let replacement = snapshot("raw", "new");
        let rewritten = rewrite_bytes(&bytes, &replacement, MetadataLimits::default()).unwrap();
        let (target, packets) = header_packets(&rewritten);
        assert_eq!(target.comments, Some(replacement.clone()));
        assert!(!packets[1].starts_with(b"SpeexTags"));
        assert_eq!(packets[1], comment_body(&replacement));
    }

    #[test]
    fn growth_preserves_multiplexed_pages_and_only_resequences_target_audio() {
        let original = snapshot("small", "old");
        let target_serial = 17;
        let other_serial = 99;
        let mut bytes = packet_page(target_serial, 0, BOS, 0, &[opus_identification()]);
        let other_bos = packet_page(other_serial, 0, BOS, 0, &[b"other-head".to_vec()]);
        bytes.extend(&other_bos);
        bytes.extend(packet_page(
            target_serial,
            1,
            0,
            0,
            &[framed_comment(Codec::Opus, &original, b"tail")],
        ));
        let other_eos = packet_page(other_serial, 1, EOS, 41, &[b"other-data".to_vec()]);
        bytes.extend(&other_eos);
        let target_audio = packet_page(
            target_serial,
            2,
            EOS,
            1234,
            &[b"target-audio-packet".to_vec()],
        );
        bytes.extend(&target_audio);
        let chained = packet_page(target_serial, 0, BOS | EOS, 7, &[b"later-chain".to_vec()]);
        bytes.extend(&chained);

        let replacement = VorbisCommentsSnapshot::new("g".repeat(70_000), Vec::new());
        let rewritten = rewrite_bytes(&bytes, &replacement, MetadataLimits::default()).unwrap();
        let pages = decoded_pages(&rewritten);

        let other_pages: Vec<_> = pages
            .iter()
            .filter(|page| page.serial == other_serial)
            .cloned()
            .collect();
        let expected_other = decoded_pages(&[other_bos, other_eos].concat());
        assert_eq!(other_pages.len(), expected_other.len());
        for (actual, expected) in other_pages.iter().zip(&expected_other) {
            assert_eq!(actual.header_type, expected.header_type);
            assert_eq!(actual.granule, expected.granule);
            assert_eq!(actual.sequence, expected.sequence);
            assert_eq!(actual.checksum, expected.checksum);
            assert_eq!(actual.lacing, expected.lacing);
            assert_eq!(actual.body, expected.body);
        }

        let audio = pages
            .iter()
            .find(|page| page.serial == target_serial && page.granule == 1234)
            .unwrap();
        let original_audio = decoded_pages(&target_audio).pop().unwrap();
        assert_eq!(audio.header_type, original_audio.header_type);
        assert_eq!(audio.granule, original_audio.granule);
        assert_eq!(audio.lacing, original_audio.lacing);
        assert_eq!(audio.body, original_audio.body);
        assert!(audio.sequence > original_audio.sequence);

        let later = pages.last().unwrap();
        let original_later = decoded_pages(&chained).pop().unwrap();
        assert_eq!(later.sequence, 0);
        assert_eq!(later.checksum, original_later.checksum);
        assert_eq!(later.body, original_later.body);
    }

    #[test]
    fn malformed_crc_truncation_sequence_and_continuation_fail_closed() {
        let original = snapshot("vendor", "old");
        let mut bytes = packet_page(3, 0, BOS, 0, &[opus_identification()]);
        bytes.extend(packet_page(
            3,
            1,
            EOS,
            0,
            &[framed_comment(Codec::Opus, &original, &[])],
        ));

        let mut bad_crc = bytes.clone();
        *bad_crc.last_mut().unwrap() ^= 1;
        assert!(scan_bytes(&bad_crc, MetadataLimits::default())
            .unwrap_err()
            .contains("CRC mismatch"));

        let mut truncated = bytes.clone();
        truncated.pop();
        assert!(scan_bytes(&truncated, MetadataLimits::default())
            .unwrap_err()
            .contains("truncated Ogg page body"));

        let id = packet_page(3, 0, BOS, 0, &[opus_identification()]);
        let bad_sequence = [
            id,
            packet_page(3, 2, EOS, 0, &[framed_comment(Codec::Opus, &original, &[])]),
        ]
        .concat();
        assert!(scan_bytes(&bad_sequence, MetadataLimits::default())
            .unwrap_err()
            .contains("sequence mismatch"));

        let bad_continuation = [
            packet_page(3, 0, BOS, 0, &[opus_identification()]),
            packet_page(
                3,
                1,
                CONTINUED | EOS,
                0,
                &[framed_comment(Codec::Opus, &original, &[])],
            ),
        ]
        .concat();
        assert!(scan_bytes(&bad_continuation, MetadataLimits::default())
            .unwrap_err()
            .contains("continuation flag mismatch"));
    }

    #[test]
    fn rejects_wrong_vorbis_framing_and_fake_speex_prefix() {
        let comments = snapshot("vendor", "old");
        let mut vorbis = framed_comment(Codec::Vorbis, &comments, &[]);
        *vorbis.last_mut().unwrap() = 0;
        let bytes = packet_page(
            5,
            0,
            BOS | EOS,
            0,
            &[vorbis_identification(), vorbis, b"\x05setup".to_vec()],
        );
        assert!(scan_bytes(&bytes, MetadataLimits::default())
            .unwrap_err()
            .contains("framing byte"));

        let mut fake = b"SpeexTags".to_vec();
        fake.extend(comment_body(&comments));
        let speex = [
            packet_page(6, 0, BOS, 0, &[speex_identification(0)]),
            packet_page(6, 1, EOS, 0, &[fake]),
        ]
        .concat();
        assert!(scan_bytes(&speex, MetadataLimits::default()).is_err());
    }

    #[test]
    fn validates_capture_version_flags_bos_eos_and_resource_bounds() {
        let valid = packet_page(1, 0, BOS | EOS, 0, &[b"unknown".to_vec()]);

        let mut bad_capture = valid.clone();
        bad_capture[0] = b'X';
        assert!(scan_bytes(&bad_capture, MetadataLimits::default())
            .unwrap_err()
            .contains("capture pattern"));

        let mut bad_version = valid.clone();
        bad_version[4] = 1;
        assert!(scan_bytes(&bad_version, MetadataLimits::default())
            .unwrap_err()
            .contains("version"));

        let mut bad_flags = valid.clone();
        bad_flags[5] |= 0x08;
        assert!(scan_bytes(&bad_flags, MetadataLimits::default())
            .unwrap_err()
            .contains("header type"));

        let no_bos = packet_page(1, 0, EOS, 0, &[b"unknown".to_vec()]);
        assert!(scan_bytes(&no_bos, MetadataLimits::default())
            .unwrap_err()
            .contains("before BOS"));

        let no_eos = packet_page(1, 0, BOS, 0, &[b"unknown".to_vec()]);
        assert!(scan_bytes(&no_eos, MetadataLimits::default())
            .unwrap_err()
            .contains("missing EOS"));

        let two_streams = [
            valid,
            packet_page(2, 0, BOS | EOS, 0, &[b"unknown".to_vec()]),
        ]
        .concat();
        let mut limits = MetadataLimits::default();
        limits.max_ogg_streams = 1;
        assert!(scan_bytes(&two_streams, limits)
            .unwrap_err()
            .contains("stream count exceeds limit"));

        let oversized = packet_page(1, 0, BOS | EOS, 0, &[vec![0; 20]]);
        let mut limits = MetadataLimits::default();
        limits.max_ogg_packet_bytes = 10;
        assert!(scan_bytes(&oversized, limits)
            .unwrap_err()
            .contains("packet exceeds limit"));
    }

    #[test]
    fn rewrite_validation_failure_leaves_file_unchanged() {
        let original = snapshot("vendor", "old");
        let mut bytes = [
            packet_page(3, 0, BOS, 0, &[opus_identification()]),
            packet_page(3, 1, EOS, 0, &[framed_comment(Codec::Opus, &original, &[])]),
        ]
        .concat();
        *bytes.last_mut().unwrap() ^= 1;
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&bytes).unwrap();
        assert!(rewrite(
            &mut file,
            &snapshot("replacement", "new"),
            &MetadataLimits::default()
        )
        .is_err());
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut after = Vec::new();
        file.read_to_end(&mut after).unwrap();
        assert_eq!(after, bytes);
    }

    struct CountingReader<R> {
        inner: R,
        bytes: usize,
    }

    impl<R: Read> Read for CountingReader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let count = self.inner.read(buffer)?;
            self.bytes += count;
            Ok(count)
        }
    }

    #[test]
    fn page_search_limit_stops_before_unneeded_trailing_input() {
        let first = packet_page(42, 0, BOS, 0, &[b"unknown-one".to_vec()]);
        let second = packet_page(42, 1, 0, 1, &[b"unknown-two".to_vec()]);
        let third = packet_page(42, 2, EOS, 2, &[vec![0xaa; 60_000]]);
        let bytes = [first.clone(), second.clone(), third].concat();
        let mut reader = CountingReader {
            inner: Cursor::new(bytes),
            bytes: 0,
        };
        let mut limits = MetadataLimits::default();
        limits.max_ogg_pages = 1;
        let mut budget = MetadataBudget::new(limits);
        assert!(
            scan(&mut reader, &limits, &mut budget, ScanMode::HeadersOnly,)
                .unwrap_err()
                .contains("page limit")
        );
        assert_eq!(reader.bytes, first.len() + second.len());
    }

    #[test]
    fn header_only_accepts_shared_final_header_page_but_rewrite_is_atomic() {
        let comments = snapshot("vendor", "old");
        let bytes = packet_page(
            7,
            0,
            BOS | EOS,
            11,
            &[
                vorbis_identification(),
                framed_comment(Codec::Vorbis, &comments, &[]),
                b"\x05vorbis-setup".to_vec(),
                b"audio".to_vec(),
            ],
        );
        let limits = MetadataLimits::default();
        let mut budget = MetadataBudget::new(limits);
        let header_scan = scan(
            &mut Cursor::new(&bytes),
            &limits,
            &mut budget,
            ScanMode::HeadersOnly,
        )
        .unwrap();
        assert!(
            header_scan
                .target
                .as_ref()
                .unwrap()
                .shares_header_page_with_audio
        );
        let mut budget = MetadataBudget::new(limits);
        scan(
            &mut Cursor::new(&bytes),
            &limits,
            &mut budget,
            ScanMode::Decode,
        )
        .unwrap();

        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&bytes).unwrap();
        let error = rewrite(&mut file, &snapshot("new", "value"), &limits).unwrap_err();
        assert!(error.contains("shares the final metadata header page"));
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut after = Vec::new();
        file.read_to_end(&mut after).unwrap();
        assert_eq!(after, bytes);
    }

    #[test]
    fn header_only_stops_after_headers_while_decode_checks_trailing_pages() {
        let comments = snapshot("vendor", "old");
        let first = packet_page(3, 0, BOS, 0, &[opus_identification()]);
        let second = packet_page(3, 1, 0, 0, &[framed_comment(Codec::Opus, &comments, &[])]);
        let mut corrupt = packet_page(3, 2, EOS, 9, &[b"audio".to_vec()]);
        *corrupt.last_mut().unwrap() ^= 1;
        let bytes = [first.clone(), second.clone(), corrupt].concat();
        let limits = MetadataLimits::default();
        let mut reader = CountingReader {
            inner: Cursor::new(&bytes),
            bytes: 0,
        };
        let mut budget = MetadataBudget::new(limits);
        scan(&mut reader, &limits, &mut budget, ScanMode::HeadersOnly).unwrap();
        assert_eq!(reader.bytes, first.len() + second.len());

        let mut budget = MetadataBudget::new(limits);
        assert!(scan(
            &mut Cursor::new(&bytes),
            &limits,
            &mut budget,
            ScanMode::Decode,
        )
        .unwrap_err()
        .contains("CRC mismatch"));
    }

    #[test]
    fn header_only_allows_an_interleaved_non_target_packet_to_remain_open() {
        let comments = snapshot("vendor", "old");
        let other_open = page(99, 0, BOS, NO_GRANULE, vec![255], vec![0xaa; 255]);
        let bytes = [
            packet_page(3, 0, BOS, 0, &[opus_identification()]),
            other_open,
            packet_page(3, 1, EOS, 0, &[framed_comment(Codec::Opus, &comments, &[])]),
        ]
        .concat();
        let limits = MetadataLimits::default();
        let mut budget = MetadataBudget::new(limits);
        scan(
            &mut Cursor::new(bytes),
            &limits,
            &mut budget,
            ScanMode::HeadersOnly,
        )
        .unwrap();

        let only_truncated_unknown = page(99, 0, BOS, NO_GRANULE, vec![255], vec![0xaa; 255]);
        let mut budget = MetadataBudget::new(limits);
        assert!(scan(
            &mut Cursor::new(only_truncated_unknown),
            &limits,
            &mut budget,
            ScanMode::HeadersOnly,
        )
        .unwrap_err()
        .contains("truncated Ogg packet"));
    }

    #[test]
    fn decode_accepts_closed_packets_without_eos_but_rewrite_rejects() {
        let comments = snapshot("vendor", "old");
        let bytes = [
            packet_page(3, 0, BOS, 0, &[opus_identification()]),
            packet_page(3, 1, 0, 0, &[framed_comment(Codec::Opus, &comments, &[])]),
            packet_page(3, 2, 0, 9, &[b"audio".to_vec()]),
        ]
        .concat();
        let limits = MetadataLimits::default();
        let mut budget = MetadataBudget::new(limits);
        scan(
            &mut Cursor::new(&bytes),
            &limits,
            &mut budget,
            ScanMode::Decode,
        )
        .unwrap();
        let mut budget = MetadataBudget::new(limits);
        assert!(scan(
            &mut Cursor::new(&bytes),
            &limits,
            &mut budget,
            ScanMode::Rewrite,
        )
        .unwrap_err()
        .contains("missing EOS"));
    }

    #[test]
    fn simultaneous_header_buffers_and_rewrite_copies_obey_total_limit() {
        let comments = VorbisCommentsSnapshot::new("v".repeat(40), Vec::new());
        let bytes = packet_page(
            7,
            0,
            BOS | EOS,
            0,
            &[
                vorbis_identification(),
                framed_comment(Codec::Vorbis, &comments, &[]),
                b"\x05vorbis-setup-payload".to_vec(),
            ],
        );
        let mut limits = MetadataLimits::default();
        limits.max_total_bytes = 200;
        let error = rewrite_bytes(&bytes, &snapshot("new", "value"), limits).unwrap_err();
        assert!(
            error.contains("aggregate limit") || error.contains("allocation limit"),
            "{error}"
        );

        limits.max_total_bytes = 400;
        let first_open = page(3, 0, BOS, NO_GRANULE, vec![255], vec![0; 255]);
        let second_open = page(4, 0, BOS, NO_GRANULE, vec![255], vec![0; 255]);
        let interleaved = [first_open, second_open].concat();
        let mut budget = MetadataBudget::new(limits);
        assert!(scan(
            &mut Cursor::new(interleaved),
            &limits,
            &mut budget,
            ScanMode::Decode,
        )
        .unwrap_err()
        .contains("simultaneous header buffers"));
    }

    #[test]
    fn zero_length_extra_headers_do_not_require_full_page_body_capacity() {
        let original = VorbisCommentsSnapshot::new(String::new(), Vec::new());
        let codec = Codec::Speex {
            header_packets: 202,
        };
        let mut identification = speex_identification(200);
        identification[68..72].copy_from_slice(&200_i32.to_le_bytes());
        let mut packets = Vec::new();
        packets.push(identification);
        packets.push(framed_comment(codec, &original, &[]));
        packets.extend((0..200).map(|_| Vec::new()));
        let bytes = packet_page(11, 0, BOS | EOS, 0, &packets);
        let mut limits = MetadataLimits::default();
        limits.max_total_bytes = 4_096;
        limits.max_items = 512;
        let rewritten = rewrite_bytes(&bytes, &snapshot("v", "new"), limits).unwrap();
        let (_, packets) = header_packets(&rewritten);
        assert_eq!(packets.len(), 202);
    }

    fn zero_optional_limits() -> MetadataLimits {
        MetadataLimits {
            max_total_bytes: 0,
            max_item_bytes: 0,
            max_items: 0,
            ..MetadataLimits::default()
        }
    }

    #[test]
    fn zero_optional_budget_accepts_tagless_opus_and_vorbis_without_snapshot() {
        let tagless = VorbisCommentsSnapshot::new("mandatory codec vendor".into(), Vec::new());
        let opus = [
            packet_page(3, 0, BOS, 0, &[opus_identification()]),
            packet_page(3, 1, EOS, 0, &[framed_comment(Codec::Opus, &tagless, &[])]),
        ]
        .concat();
        let limits = zero_optional_limits();
        let mut budget = MetadataBudget::new(limits);
        let opus_scan = scan(
            &mut Cursor::new(opus),
            &limits,
            &mut budget,
            ScanMode::HeadersOnly,
        )
        .unwrap();
        assert!(opus_scan.target.unwrap().comments.is_none());

        let vorbis = packet_page(
            7,
            0,
            BOS | EOS,
            0,
            &[
                vorbis_identification(),
                framed_comment(Codec::Vorbis, &tagless, &[]),
                b"\x05vorbis-setup".to_vec(),
            ],
        );
        let mut budget = MetadataBudget::new(limits);
        let vorbis_scan = scan(
            &mut Cursor::new(vorbis),
            &limits,
            &mut budget,
            ScanMode::HeadersOnly,
        )
        .unwrap();
        assert!(vorbis_scan.target.unwrap().comments.is_none());
    }

    #[test]
    fn zero_optional_budget_rejects_fields_tail_and_invalid_vendor() {
        let with_field = snapshot("vendor", "optional");
        let with_tail = VorbisCommentsSnapshot::new("vendor".into(), Vec::new());
        for packet in [
            framed_comment(Codec::Opus, &with_field, &[]),
            framed_comment(Codec::Opus, &with_tail, b"optional-tail"),
        ] {
            let bytes = [
                packet_page(3, 0, BOS, 0, &[opus_identification()]),
                packet_page(3, 1, EOS, 0, &[packet]),
            ]
            .concat();
            let limits = zero_optional_limits();
            let mut budget = MetadataBudget::new(limits);
            assert!(scan(
                &mut Cursor::new(bytes),
                &limits,
                &mut budget,
                ScanMode::HeadersOnly,
            )
            .is_err());
        }

        let mut invalid_body = Vec::new();
        invalid_body.extend(1_u32.to_le_bytes());
        invalid_body.push(0xff);
        invalid_body.extend(0_u32.to_le_bytes());
        let mut invalid_packet = b"OpusTags".to_vec();
        invalid_packet.extend(invalid_body);
        let bytes = [
            packet_page(3, 0, BOS, 0, &[opus_identification()]),
            packet_page(3, 1, EOS, 0, &[invalid_packet]),
        ]
        .concat();
        let limits = zero_optional_limits();
        let mut budget = MetadataBudget::new(limits);
        assert!(scan(
            &mut Cursor::new(bytes),
            &limits,
            &mut budget,
            ScanMode::HeadersOnly,
        )
        .unwrap_err()
        .contains("not UTF-8"));
    }

    #[test]
    fn rewrite_peak_accounts_for_the_live_serialized_body() {
        let original = VorbisCommentsSnapshot::new("old".into(), Vec::new());
        let bytes = [
            packet_page(3, 0, BOS, 0, &[opus_identification()]),
            packet_page(3, 1, EOS, 0, &[framed_comment(Codec::Opus, &original, &[])]),
        ]
        .concat();
        let replacement = VorbisCommentsSnapshot::new("x".repeat(500), Vec::new());
        let mut limits = MetadataLimits::default();
        limits.max_total_bytes = 1_100;
        let error = rewrite_bytes(&bytes, &replacement, limits).unwrap_err();
        assert!(
            error.contains("serialized body")
                || error.contains("allocation limit")
                || error.contains("aggregate limit"),
            "{error}"
        );
    }
}
