//! Bounded FLAC metadata inspection and Vorbis-comment rewriting.
//!
//! FLAC metadata blocks are described by fixed four-byte headers, so blocks
//! which are not retained can be validated and skipped without allocating
//! their bodies. Rewrites are assembled in a temporary file with a fixed-size
//! copy buffer before the caller's staged output is modified.

use std::cmp;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};

use lofty::picture::{Picture, PictureInformation};

use super::{
    parse_comment_body, picture_block_len, serialize_comment_body, write_picture_block,
    MetadataBudget, MetadataLimits, VorbisCommentsSnapshot,
};

const FLAC_MARKER: &[u8; 4] = b"fLaC";
const STREAMINFO: u8 = 0;
const PADDING: u8 = 1;
const APPLICATION: u8 = 2;
const VORBIS_COMMENT: u8 = 4;
const PICTURE: u8 = 6;
const RESERVED: u8 = 127;
const STREAMINFO_BYTES: u32 = 34;
const MAX_BLOCK_BYTES: u32 = 0x00ff_ffff;
const COPY_BUFFER_BYTES: usize = 8 * 1024;

#[derive(Clone, Copy, Debug)]
struct BlockDescriptor {
    block_type: u8,
    body_offset: u64,
    body_len: u32,
}

#[derive(Debug)]
struct Layout {
    blocks: Vec<BlockDescriptor>,
    comments: Option<VorbisCommentsSnapshot>,
    audio_offset: u64,
    file_len: u64,
}

/// Inspect FLAC metadata through an existing staged file handle.
pub(super) fn read_file(
    input: &mut File,
    limits: &MetadataLimits,
    budget: &mut MetadataBudget,
) -> Result<Option<VorbisCommentsSnapshot>, String> {
    let Some(layout) = scan(input, limits, budget)? else {
        return Ok(None);
    };
    Ok(layout.comments)
}

/// Validate a staged FLAC through an existing handle for decoder preflight.
///
/// The caller must rewind the handle before handing it to a decoder. Parsed
/// comments are discarded after all block, item, and aggregate limits have
/// been enforced.
pub(super) fn validate(input: &mut File, limits: &MetadataLimits) -> Result<(), String> {
    let mut budget = MetadataBudget::new(*limits);
    if scan(input, limits, &mut budget)?.is_none() {
        return Err("raw metadata input does not start with the FLAC marker".into());
    }
    Ok(())
}

/// Merge `source` into the destination FLAC's Vorbis comments.
///
/// Non-comment metadata block bodies retain their order and bytes. When the
/// size delta fits in existing padding, the audio offset also remains stable.
/// Otherwise both the rebuilt metadata prefix and the audio suffix are copied
/// through a temporary file using a fixed-size buffer.
pub(super) fn rewrite(
    output: &mut File,
    comments: &VorbisCommentsSnapshot,
    pictures: &[(Picture, PictureInformation)],
    limits: &MetadataLimits,
) -> Result<(), String> {
    let mut scan_budget = MetadataBudget::new(*limits);
    let Some(layout) = scan(output, limits, &mut scan_budget)? else {
        return Ok(());
    };

    let Layout {
        blocks,
        comments: _,
        audio_offset,
        file_len,
    } = layout;
    if pictures.len() > limits.max_items {
        return Err(format!(
            "FLAC pictures exceed the {} item limit",
            limits.max_items
        ));
    }
    let minimum_blocks = pictures
        .len()
        .checked_add(2)
        .ok_or("FLAC picture block count overflow")?;
    if minimum_blocks > limits.max_flac_blocks {
        return Err(format!(
            "rewritten FLAC metadata exceeds the {} block limit",
            limits.max_flac_blocks
        ));
    }
    let mut final_budget = MetadataBudget::new(*limits);
    let comment = serialize_comment_body(comments, &mut final_budget)?;
    validate_output_block_len(comment.len(), limits, "FLAC Vorbis comment")?;

    let mut picture_lengths = Vec::new();
    picture_lengths
        .try_reserve_exact(pictures.len())
        .map_err(|_| "unable to reserve bounded FLAC picture descriptors")?;
    for (picture, information) in pictures {
        let length = picture_block_len(picture, *information)?;
        validate_output_block_len(length, limits, "FLAC picture")?;
        final_budget.check_item(length, "FLAC picture block")?;
        final_budget.charge_bytes(length, "FLAC picture block")?;
        picture_lengths.push(
            u32::try_from(length).map_err(|_| "FLAC picture length does not fit in 32 bits")?,
        );
    }

    let mut plan = build_plan(&blocks, comment, &picture_lengths, limits)?;
    let keeps_audio_offset = adjust_existing_padding(&mut plan, audio_offset, limits)?;
    let planned_metadata_len = plan_metadata_len(&plan)?;
    if keeps_audio_offset && planned_metadata_len != audio_offset {
        return Err("internal FLAC padding adjustment did not preserve the audio offset".into());
    }

    let mut scratch = tempfile::tempfile()
        .map_err(|error| format!("create FLAC metadata rewrite scratch file: {error}"))?;
    write_metadata_plan(output, &mut scratch, &plan, pictures)?;

    if !keeps_audio_offset {
        let suffix_len = file_len
            .checked_sub(audio_offset)
            .ok_or("invalid FLAC audio suffix range")?;
        copy_exact_at(
            output,
            audio_offset,
            suffix_len,
            &mut scratch,
            "copy FLAC audio suffix to rewrite scratch file",
        )?;
    }
    scratch
        .flush()
        .map_err(|error| format!("flush FLAC metadata rewrite scratch file: {error}"))?;
    let scratch_len = scratch
        .stream_position()
        .map_err(|error| format!("inspect FLAC metadata rewrite scratch file: {error}"))?;

    scratch
        .rewind()
        .map_err(|error| format!("rewind FLAC metadata rewrite scratch file: {error}"))?;
    output
        .rewind()
        .map_err(|error| format!("rewind raw FLAC metadata output: {error}"))?;
    if !keeps_audio_offset {
        output
            .set_len(0)
            .map_err(|error| format!("truncate raw FLAC metadata output: {error}"))?;
    }
    copy_exact(
        &mut scratch,
        scratch_len,
        output,
        "publish FLAC metadata rewrite",
    )?;
    if !keeps_audio_offset {
        output
            .set_len(scratch_len)
            .map_err(|error| format!("set rewritten FLAC output length: {error}"))?;
    }
    output
        .flush()
        .map_err(|error| format!("flush raw FLAC metadata output: {error}"))
}

fn scan<R: Read + Seek>(
    reader: &mut R,
    limits: &MetadataLimits,
    budget: &mut MetadataBudget,
) -> Result<Option<Layout>, String> {
    let file_len = reader
        .seek(SeekFrom::End(0))
        .map_err(|error| format!("inspect raw FLAC metadata length: {error}"))?;
    reader
        .rewind()
        .map_err(|error| format!("rewind raw FLAC metadata: {error}"))?;

    let mut marker = [0_u8; 4];
    match reader.read_exact(&mut marker) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(format!("read raw FLAC marker: {error}")),
    }
    if &marker != FLAC_MARKER {
        return Ok(None);
    }

    let initial_capacity = cmp::min(limits.max_flac_blocks, 16);
    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(initial_capacity)
        .map_err(|_| "unable to reserve bounded FLAC block descriptors")?;
    let mut comments: Option<VorbisCommentsSnapshot> = None;
    let mut metadata_offset = 4_u64;

    loop {
        let index = blocks.len();
        if index >= limits.max_flac_blocks {
            return Err(format!(
                "FLAC metadata exceeds the {} block limit",
                limits.max_flac_blocks
            ));
        }

        let header_end = metadata_offset
            .checked_add(4)
            .ok_or("FLAC metadata offset overflow")?;
        if header_end > file_len {
            return Err("truncated FLAC metadata header or missing last-block flag".into());
        }
        reader
            .seek(SeekFrom::Start(metadata_offset))
            .map_err(|error| format!("seek to FLAC metadata header: {error}"))?;
        let mut header = [0_u8; 4];
        reader
            .read_exact(&mut header)
            .map_err(|error| format!("read FLAC metadata header: {error}"))?;

        let is_last = header[0] & 0x80 != 0;
        let block_type = header[0] & 0x7f;
        if block_type == RESERVED {
            return Err("FLAC metadata uses reserved block type 127".into());
        }
        let body_len = u32::from_be_bytes([0, header[1], header[2], header[3]]);
        let body_len_usize = usize::try_from(body_len)
            .map_err(|_| "FLAC metadata block length does not fit this platform")?;
        if body_len_usize > limits.max_flac_block_bytes {
            return Err(format!(
                "FLAC metadata block exceeds the {} byte limit",
                limits.max_flac_block_bytes
            ));
        }
        if index == 0 {
            if block_type != STREAMINFO {
                return Err("FLAC STREAMINFO must be the first metadata block".into());
            }
            if body_len != STREAMINFO_BYTES {
                return Err("FLAC STREAMINFO block must contain exactly 34 bytes".into());
            }
        } else if block_type == STREAMINFO {
            return Err("FLAC contains more than one STREAMINFO block".into());
        }
        if block_type == APPLICATION && body_len < 4 {
            return Err("FLAC APPLICATION block is shorter than its four-byte identifier".into());
        }

        let body_offset = header_end;
        let body_end = body_offset
            .checked_add(u64::from(body_len))
            .ok_or("FLAC metadata block range overflow")?;
        if body_end > file_len {
            return Err("truncated FLAC metadata block body".into());
        }

        match block_type {
            VORBIS_COMMENT => {
                if comments.is_some() {
                    return Err("FLAC contains more than one Vorbis comment block".into());
                }
                budget.check_bytes(body_len_usize, "FLAC Vorbis comment block")?;
                let body = read_body(reader, body_offset, body_len_usize)?;
                comments = Some(parse_comment_body(&body, budget)?);
            }
            PICTURE => {
                budget.check_item(body_len_usize, "FLAC picture block")?;
                budget.charge_bytes(body_len_usize, "FLAC picture block")?;
                reader
                    .seek(SeekFrom::Start(body_end))
                    .map_err(|error| format!("skip FLAC picture block: {error}"))?;
            }
            _ => {
                reader
                    .seek(SeekFrom::Start(body_end))
                    .map_err(|error| format!("skip FLAC metadata block: {error}"))?;
            }
        }

        if blocks.len() == blocks.capacity() {
            blocks
                .try_reserve(1)
                .map_err(|_| "unable to reserve bounded FLAC block descriptors")?;
        }
        blocks.push(BlockDescriptor {
            block_type,
            body_offset,
            body_len,
        });
        metadata_offset = body_end;
        if is_last {
            break;
        }
    }

    Ok(Some(Layout {
        blocks,
        comments,
        audio_offset: metadata_offset,
        file_len,
    }))
}

fn read_body<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    body.try_reserve_exact(length)
        .map_err(|_| "unable to reserve bounded FLAC metadata block")?;
    body.resize(length, 0);
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek to FLAC metadata block: {error}"))?;
    reader
        .read_exact(&mut body)
        .map_err(|error| format!("read FLAC metadata block: {error}"))?;
    Ok(body)
}

enum PlannedBody {
    Copy {
        offset: u64,
        len: u32,
    },
    Comment(Vec<u8>),
    Picture {
        index: usize,
        len: u32,
    },
    Padding {
        offset: u64,
        original_len: u32,
        output_len: u32,
    },
}

impl PlannedBody {
    fn len(&self) -> Result<u32, String> {
        match self {
            Self::Copy { len, .. } => Ok(*len),
            Self::Comment(body) => u32::try_from(body.len())
                .map_err(|_| "FLAC comment length does not fit in 32 bits".into()),
            Self::Picture { len, .. } => Ok(*len),
            Self::Padding { output_len, .. } => Ok(*output_len),
        }
    }
}

struct PlannedBlock {
    block_type: u8,
    body: PlannedBody,
}

fn build_plan(
    blocks: &[BlockDescriptor],
    comment: Vec<u8>,
    picture_lengths: &[u32],
    limits: &MetadataLimits,
) -> Result<Vec<PlannedBlock>, String> {
    let retained_count = blocks
        .iter()
        .filter(|block| !matches!(block.block_type, VORBIS_COMMENT | PICTURE))
        .count();
    let target_count = picture_lengths
        .len()
        .checked_add(1)
        .ok_or("FLAC rewrite target block count overflow")?;
    let unadjusted_count = retained_count
        .checked_add(target_count)
        .ok_or("FLAC rewrite block count overflow")?;
    let padding_to_remove = unadjusted_count.saturating_sub(limits.max_flac_blocks);
    let available_padding = blocks
        .iter()
        .filter(|block| block.block_type == PADDING)
        .count();
    if padding_to_remove > available_padding {
        return Err(format!(
            "rewritten FLAC metadata exceeds the {} block limit",
            limits.max_flac_blocks
        ));
    }
    let requested_capacity = unadjusted_count
        .checked_sub(padding_to_remove)
        .ok_or("FLAC rewrite block count underflow")?;

    let mut plan = Vec::new();
    plan.try_reserve_exact(requested_capacity)
        .map_err(|_| "unable to reserve bounded FLAC rewrite plan")?;
    let streaminfo = blocks
        .first()
        .ok_or("FLAC rewrite layout has no STREAMINFO block")?;
    plan.push(PlannedBlock {
        block_type: STREAMINFO,
        body: PlannedBody::Copy {
            offset: streaminfo.body_offset,
            len: streaminfo.body_len,
        },
    });
    plan.push(PlannedBlock {
        block_type: VORBIS_COMMENT,
        body: PlannedBody::Comment(comment),
    });
    for (index, length) in picture_lengths.iter().enumerate() {
        plan.push(PlannedBlock {
            block_type: PICTURE,
            body: PlannedBody::Picture {
                index,
                len: *length,
            },
        });
    }

    let mut removed_padding = 0_usize;
    for block in blocks.iter().skip(1) {
        if matches!(block.block_type, VORBIS_COMMENT | PICTURE) {
            continue;
        }
        if block.block_type == PADDING && removed_padding < padding_to_remove {
            removed_padding += 1;
            continue;
        }
        plan.push(PlannedBlock {
            block_type: block.block_type,
            body: if block.block_type == PADDING {
                PlannedBody::Padding {
                    offset: block.body_offset,
                    original_len: block.body_len,
                    output_len: block.body_len,
                }
            } else {
                PlannedBody::Copy {
                    offset: block.body_offset,
                    len: block.body_len,
                }
            },
        });
    }
    if removed_padding != padding_to_remove || plan.len() != requested_capacity {
        return Err("internal FLAC rewrite block plan is inconsistent".into());
    }
    Ok(plan)
}

fn adjust_existing_padding(
    plan: &mut [PlannedBlock],
    target_len: u64,
    limits: &MetadataLimits,
) -> Result<bool, String> {
    let current_len = plan_metadata_len(plan)?;
    if current_len == target_len {
        return Ok(true);
    }

    if current_len > target_len {
        let excess = current_len - target_len;
        let available = plan.iter().try_fold(0_u64, |total, block| {
            let len = match &block.body {
                PlannedBody::Padding { output_len, .. } => u64::from(*output_len),
                _ => 0,
            };
            total.checked_add(len).ok_or("FLAC padding size overflow")
        })?;
        if available < excess {
            return Ok(false);
        }

        let mut remaining = excess;
        for block in plan {
            let PlannedBody::Padding { output_len, .. } = &mut block.body else {
                continue;
            };
            let reduction = cmp::min(u64::from(*output_len), remaining);
            *output_len -= u32::try_from(reduction)
                .map_err(|_| "FLAC padding reduction does not fit in 24 bits")?;
            remaining -= reduction;
            if remaining == 0 {
                break;
            }
        }
        return Ok(true);
    }

    let deficit = target_len - current_len;
    let maximum_len = u64::try_from(limits.max_flac_block_bytes.min(MAX_BLOCK_BYTES as usize))
        .map_err(|_| "FLAC block limit does not fit in 64 bits")?;
    let available = plan.iter().try_fold(0_u64, |total, block| {
        let capacity = match &block.body {
            PlannedBody::Padding { output_len, .. } => maximum_len
                .checked_sub(u64::from(*output_len))
                .ok_or("FLAC padding exceeds its configured block limit")?,
            _ => 0,
        };
        total
            .checked_add(capacity)
            .ok_or("FLAC padding capacity overflow")
    })?;
    if available < deficit {
        return Ok(false);
    }

    let mut remaining = deficit;
    for block in plan {
        let PlannedBody::Padding { output_len, .. } = &mut block.body else {
            continue;
        };
        let capacity = maximum_len
            .checked_sub(u64::from(*output_len))
            .ok_or("FLAC padding exceeds its configured block limit")?;
        let increase = cmp::min(capacity, remaining);
        let increased = u64::from(*output_len)
            .checked_add(increase)
            .ok_or("FLAC padding length overflow")?;
        *output_len =
            u32::try_from(increased).map_err(|_| "FLAC padding length exceeds the 24-bit limit")?;
        remaining -= increase;
        if remaining == 0 {
            break;
        }
    }
    Ok(true)
}

fn plan_metadata_len(plan: &[PlannedBlock]) -> Result<u64, String> {
    plan.iter().try_fold(4_u64, |total, block| {
        let body_len = block.body.len()?;
        total
            .checked_add(4)
            .and_then(|value| value.checked_add(u64::from(body_len)))
            .ok_or_else(|| "FLAC rewrite metadata length overflow".into())
    })
}

fn validate_output_block_len(
    length: usize,
    limits: &MetadataLimits,
    context: &str,
) -> Result<(), String> {
    if length > MAX_BLOCK_BYTES as usize {
        return Err(format!(
            "{context} exceeds the FLAC 24-bit block size limit"
        ));
    }
    if length > limits.max_flac_block_bytes {
        return Err(format!(
            "{context} exceeds the {} byte block limit",
            limits.max_flac_block_bytes
        ));
    }
    Ok(())
}

fn write_metadata_plan(
    source: &mut File,
    scratch: &mut File,
    plan: &[PlannedBlock],
    pictures: &[(Picture, PictureInformation)],
) -> Result<(), String> {
    scratch
        .write_all(FLAC_MARKER)
        .map_err(|error| format!("write FLAC marker to rewrite scratch file: {error}"))?;
    for (index, block) in plan.iter().enumerate() {
        let length = block.body.len()?;
        let mut first = block.block_type;
        if index + 1 == plan.len() {
            first |= 0x80;
        }
        let header = [
            first,
            (length >> 16) as u8,
            (length >> 8) as u8,
            length as u8,
        ];
        scratch
            .write_all(&header)
            .map_err(|error| format!("write FLAC metadata header to scratch file: {error}"))?;

        let body_start = scratch
            .stream_position()
            .map_err(|error| format!("inspect FLAC rewrite scratch position: {error}"))?;
        match &block.body {
            PlannedBody::Copy { offset, len } => copy_exact_at(
                source,
                *offset,
                u64::from(*len),
                scratch,
                "copy FLAC metadata block to rewrite scratch file",
            )?,
            PlannedBody::Comment(body) => scratch.write_all(body).map_err(|error| {
                format!("write FLAC Vorbis comment to rewrite scratch file: {error}")
            })?,
            PlannedBody::Picture { index, .. } => {
                let (picture, information) = pictures
                    .get(*index)
                    .ok_or("internal FLAC picture rewrite index is invalid")?;
                write_picture_block(picture, *information, scratch).map_err(|error| {
                    format!("write FLAC picture to rewrite scratch file: {error}")
                })?;
            }
            PlannedBody::Padding {
                offset,
                original_len,
                output_len,
            } => {
                let retained = cmp::min(*original_len, *output_len);
                copy_exact_at(
                    source,
                    *offset,
                    u64::from(retained),
                    scratch,
                    "copy FLAC padding to rewrite scratch file",
                )?;
                write_zeros(
                    scratch,
                    u64::from(*output_len - retained),
                    "extend FLAC padding in rewrite scratch file",
                )?;
            }
        }
        let body_end = scratch
            .stream_position()
            .map_err(|error| format!("inspect FLAC rewrite scratch position: {error}"))?;
        if body_end.checked_sub(body_start) != Some(u64::from(length)) {
            return Err("FLAC rewrite block writer produced an inconsistent length".into());
        }
    }
    Ok(())
}

fn copy_exact_at<R: Read + Seek, W: Write>(
    source: &mut R,
    offset: u64,
    length: u64,
    destination: &mut W,
    context: &str,
) -> Result<(), String> {
    source
        .seek(SeekFrom::Start(offset))
        .map_err(|error| format!("{context}: {error}"))?;
    copy_exact(source, length, destination, context)
}

fn copy_exact<R: Read, W: Write>(
    source: &mut R,
    mut remaining: u64,
    destination: &mut W,
    context: &str,
) -> Result<(), String> {
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    while remaining != 0 {
        let length = cmp::min(remaining, buffer.len() as u64) as usize;
        source
            .read_exact(&mut buffer[..length])
            .map_err(|error| format!("{context}: {error}"))?;
        destination
            .write_all(&buffer[..length])
            .map_err(|error| format!("{context}: {error}"))?;
        remaining -= length as u64;
    }
    Ok(())
}

fn write_zeros<W: Write>(
    destination: &mut W,
    mut remaining: u64,
    context: &str,
) -> Result<(), String> {
    let zeros = [0_u8; COPY_BUFFER_BYTES];
    while remaining != 0 {
        let length = cmp::min(remaining, zeros.len() as u64) as usize;
        destination
            .write_all(&zeros[..length])
            .map_err(|error| format!("{context}: {error}"))?;
        remaining -= length as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::io::Cursor;
    use std::rc::Rc;

    use lofty::picture::{MimeType, PictureType};

    use super::*;

    fn limits() -> MetadataLimits {
        MetadataLimits {
            max_total_bytes: 4 * 1024 * 1024,
            max_item_bytes: 2 * 1024 * 1024,
            max_items: 1_024,
            max_flac_block_bytes: 2 * 1024 * 1024,
            max_flac_blocks: 64,
            ..MetadataLimits::default()
        }
    }

    fn snapshot(vendor: &str, items: &[(&str, &str)]) -> VorbisCommentsSnapshot {
        VorbisCommentsSnapshot::new(
            vendor.into(),
            items
                .iter()
                .map(|(key, value)| ((*key).into(), (*value).into()))
                .collect(),
        )
    }

    fn comment_body(value: &VorbisCommentsSnapshot, limits: MetadataLimits) -> Vec<u8> {
        serialize_comment_body(value, &mut MetadataBudget::new(limits)).unwrap()
    }

    fn metadata_block(block_type: u8, last: bool, body: &[u8]) -> Vec<u8> {
        assert!(body.len() <= MAX_BLOCK_BYTES as usize);
        let length = body.len() as u32;
        let mut output = Vec::new();
        output.extend_from_slice(&[
            block_type | if last { 0x80 } else { 0 },
            (length >> 16) as u8,
            (length >> 8) as u8,
            length as u8,
        ]);
        output.extend_from_slice(body);
        output
    }

    fn flac(blocks: &[(u8, Vec<u8>)], audio: &[u8]) -> Vec<u8> {
        let mut output = FLAC_MARKER.to_vec();
        for (index, (block_type, body)) in blocks.iter().enumerate() {
            output.extend(metadata_block(*block_type, index + 1 == blocks.len(), body));
        }
        output.extend_from_slice(audio);
        output
    }

    fn scan_bytes(bytes: Vec<u8>, limits: MetadataLimits) -> Result<Option<Layout>, String> {
        scan(
            &mut Cursor::new(bytes),
            &limits,
            &mut MetadataBudget::new(limits),
        )
    }

    fn streaminfo() -> Vec<u8> {
        vec![0_u8; STREAMINFO_BYTES as usize]
    }

    fn picture(data: Vec<u8>, description: &str) -> (Picture, PictureInformation) {
        (
            Picture::unchecked(data)
                .pic_type(PictureType::CoverFront)
                .mime_type(MimeType::Png)
                .description(description.to_owned())
                .build(),
            PictureInformation {
                width: 32,
                height: 24,
                color_depth: 24,
                num_colors: 0,
            },
        )
    }

    #[test]
    fn block_count_limit_accepts_below_and_exact_but_rejects_above() {
        let mut bounded = limits();
        bounded.max_flac_blocks = 2;
        assert!(
            scan_bytes(flac(&[(STREAMINFO, streaminfo())], b""), bounded)
                .unwrap()
                .is_some()
        );
        assert!(scan_bytes(
            flac(&[(STREAMINFO, streaminfo()), (PADDING, vec![])], b""),
            bounded,
        )
        .unwrap()
        .is_some());
        assert!(scan_bytes(
            flac(
                &[
                    (STREAMINFO, streaminfo()),
                    (PADDING, vec![]),
                    (PADDING, vec![]),
                ],
                b"",
            ),
            bounded,
        )
        .is_err());
    }

    #[test]
    fn block_size_limit_accepts_minus_one_and_exact_but_rejects_plus_one() {
        let mut bounded = limits();
        bounded.max_flac_block_bytes = STREAMINFO_BYTES as usize + 1;
        for length in [STREAMINFO_BYTES as usize, STREAMINFO_BYTES as usize + 1] {
            assert!(scan_bytes(
                flac(
                    &[(STREAMINFO, streaminfo()), (PADDING, vec![0; length])],
                    b"",
                ),
                bounded,
            )
            .unwrap()
            .is_some());
        }
        assert!(scan_bytes(
            flac(
                &[
                    (STREAMINFO, streaminfo()),
                    (PADDING, vec![0; STREAMINFO_BYTES as usize + 2]),
                ],
                b"",
            ),
            bounded,
        )
        .is_err());
    }

    #[test]
    fn rejects_every_truncation_after_the_flac_marker() {
        let comment = comment_body(&snapshot("vendor", &[("TITLE", "value")]), limits());
        let complete = flac(
            &[(STREAMINFO, streaminfo()), (VORBIS_COMMENT, comment)],
            b"audio",
        );
        let audio_offset = complete.len() - b"audio".len();
        for length in FLAC_MARKER.len()..audio_offset {
            assert!(
                scan_bytes(complete[..length].to_vec(), limits()).is_err(),
                "truncation at {length} was accepted"
            );
        }
    }

    #[test]
    fn rejects_nonfirst_duplicate_and_wrong_sized_streaminfo() {
        assert!(scan_bytes(flac(&[(PADDING, vec![0; 34])], b""), limits()).is_err());
        assert!(scan_bytes(flac(&[(STREAMINFO, vec![0; 33])], b""), limits()).is_err());
        assert!(scan_bytes(flac(&[(STREAMINFO, vec![0; 35])], b""), limits()).is_err());
        assert!(scan_bytes(
            flac(
                &[(STREAMINFO, streaminfo()), (STREAMINFO, streaminfo())],
                b"",
            ),
            limits(),
        )
        .is_err());
    }

    #[test]
    fn rejects_reserved_type_and_missing_last_flag() {
        assert!(scan_bytes(
            flac(&[(STREAMINFO, streaminfo()), (RESERVED, vec![])], b""),
            limits(),
        )
        .is_err());

        let mut missing_last = FLAC_MARKER.to_vec();
        missing_last.extend(metadata_block(STREAMINFO, false, &streaminfo()));
        assert!(scan_bytes(missing_last, limits()).is_err());
    }

    #[test]
    fn application_block_requires_a_complete_identifier() {
        for length in 0..4 {
            let error = scan_bytes(
                flac(
                    &[(STREAMINFO, streaminfo()), (APPLICATION, vec![0; length])],
                    b"",
                ),
                limits(),
            )
            .unwrap_err();
            assert!(error.contains("four-byte identifier"), "{error}");
        }
        assert!(scan_bytes(
            flac(
                &[(STREAMINFO, streaminfo()), (APPLICATION, vec![0; 4])],
                b"",
            ),
            limits(),
        )
        .unwrap()
        .is_some());
    }

    struct CountingReader {
        inner: Cursor<Vec<u8>>,
        read_bytes: Rc<Cell<usize>>,
    }

    impl Read for CountingReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let read = self.inner.read(buffer)?;
            self.read_bytes.set(self.read_bytes.get() + read);
            Ok(read)
        }
    }

    impl Seek for CountingReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }

    #[test]
    fn large_untouched_block_is_skipped_without_reading_its_body() {
        let application = vec![0xa5; 1024 * 1024];
        let bytes = flac(&[(STREAMINFO, streaminfo()), (2, application)], b"audio");
        let read_bytes = Rc::new(Cell::new(0));
        let mut reader = CountingReader {
            inner: Cursor::new(bytes),
            read_bytes: Rc::clone(&read_bytes),
        };
        let configured = limits();
        assert!(scan(
            &mut reader,
            &configured,
            &mut MetadataBudget::new(configured),
        )
        .unwrap()
        .is_some());
        assert_eq!(read_bytes.get(), FLAC_MARKER.len() + 2 * 4);
    }

    #[test]
    fn retained_total_limit_is_checked_before_comment_body_allocation() {
        let configured = limits();
        let comment = comment_body(&snapshot("vendor", &[("TITLE", "value")]), configured);
        let bytes = flac(
            &[
                (STREAMINFO, streaminfo()),
                (VORBIS_COMMENT, comment.clone()),
            ],
            b"",
        );

        for maximum in [comment.len(), comment.len() + 1] {
            let mut accepted = configured;
            accepted.max_total_bytes = maximum;
            assert!(scan_bytes(bytes.clone(), accepted).unwrap().is_some());
        }

        let read_bytes = Rc::new(Cell::new(0));
        let mut reader = CountingReader {
            inner: Cursor::new(bytes),
            read_bytes: Rc::clone(&read_bytes),
        };
        let mut rejected = configured;
        rejected.max_total_bytes = comment.len() - 1;
        assert!(scan(&mut reader, &rejected, &mut MetadataBudget::new(rejected),).is_err());
        assert_eq!(read_bytes.get(), FLAC_MARKER.len() + 2 * 4);
    }

    #[test]
    fn picture_blocks_obey_per_item_and_aggregate_limits_without_being_read() {
        let bytes = flac(&[(STREAMINFO, streaminfo()), (PICTURE, vec![7; 33])], b"");
        let mut item_limited = limits();
        item_limited.max_item_bytes = 32;
        assert!(scan_bytes(bytes.clone(), item_limited).is_err());

        let mut total_limited = limits();
        total_limited.max_total_bytes = 32;
        assert!(scan_bytes(bytes.clone(), total_limited).is_err());

        for maximum in [33, 34] {
            let mut accepted = limits();
            accepted.max_item_bytes = maximum;
            accepted.max_total_bytes = maximum;
            assert!(scan_bytes(bytes.clone(), accepted).unwrap().is_some());
        }
    }

    #[test]
    fn rewrite_budgets_supplied_pictures_with_the_new_comment_body() {
        let configured = limits();
        let old_comment = comment_body(&snapshot("v", &[]), configured);
        let pictures = [picture(vec![7; 33], "bounded")];
        let picture_len = picture_block_len(&pictures[0].0, pictures[0].1).unwrap();
        let original = flac(
            &[
                (STREAMINFO, streaminfo()),
                (VORBIS_COMMENT, old_comment.clone()),
            ],
            b"audio",
        );
        let mut bounded = configured;
        bounded.max_total_bytes = old_comment.len() + picture_len;
        assert!(scan_bytes(original.clone(), bounded).unwrap().is_some());

        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&original).unwrap();
        let error = rewrite(
            &mut output,
            &snapshot("longer-vendor", &[("TITLE", "new retained title")]),
            &pictures,
            &bounded,
        )
        .unwrap_err();
        assert!(error.contains("aggregate limit"), "{error}");

        let mut after = Vec::new();
        after.try_reserve_exact(original.len()).unwrap();
        after.resize(original.len(), 0);
        output.rewind().unwrap();
        output.read_exact(&mut after).unwrap();
        assert_eq!(after, original);
    }

    #[test]
    fn rewrite_replaces_all_native_picture_blocks_in_canonical_order() {
        let configured = limits();
        let destination = comment_body(&snapshot("old", &[("TITLE", "old")]), configured);
        let application = vec![0x42; 21];
        let audio = b"audio-after-native-pictures";
        let original = flac(
            &[
                (STREAMINFO, streaminfo()),
                (2, application.clone()),
                (PICTURE, vec![0x11; 40]),
                (VORBIS_COMMENT, destination),
                (PICTURE, vec![0x22; 48]),
                (PADDING, vec![0; 96]),
            ],
            audio,
        );
        let pictures = [
            picture(vec![0xa1; 17], "front"),
            picture(vec![0xb2; 23], "alternate"),
        ];
        let mut expected = Vec::new();
        for (picture, information) in &pictures {
            let mut body = Vec::new();
            write_picture_block(picture, *information, &mut body).unwrap();
            expected.push(body);
        }

        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&original).unwrap();
        rewrite(
            &mut output,
            &snapshot("new", &[("TITLE", "new")]),
            &pictures,
            &configured,
        )
        .unwrap();

        output.rewind().unwrap();
        let layout = scan(
            &mut output,
            &configured,
            &mut MetadataBudget::new(configured),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            layout
                .blocks
                .iter()
                .map(|block| block.block_type)
                .collect::<Vec<_>>(),
            vec![STREAMINFO, VORBIS_COMMENT, PICTURE, PICTURE, 2, PADDING]
        );
        let actual = layout
            .blocks
            .iter()
            .filter(|block| block.block_type == PICTURE)
            .map(|block| {
                read_body(&mut output, block.body_offset, block.body_len as usize).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        let application_block = layout
            .blocks
            .iter()
            .find(|block| block.block_type == 2)
            .unwrap();
        assert_eq!(
            read_body(
                &mut output,
                application_block.body_offset,
                application_block.body_len as usize,
            )
            .unwrap(),
            application
        );
        assert_eq!(
            read_body(&mut output, layout.audio_offset, audio.len()).unwrap(),
            audio
        );
    }

    #[test]
    fn rewrite_with_no_supplied_pictures_removes_all_native_picture_blocks() {
        let configured = limits();
        let destination = comment_body(&snapshot("old", &[]), configured);
        let audio = b"audio-after-removed-pictures";
        let original = flac(
            &[
                (STREAMINFO, streaminfo()),
                (PICTURE, vec![0x11; 40]),
                (VORBIS_COMMENT, destination),
                (PICTURE, vec![0x22; 48]),
                (PADDING, vec![0; 32]),
            ],
            audio,
        );
        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&original).unwrap();
        rewrite(
            &mut output,
            &snapshot("new", &[("TITLE", "picture-free")]),
            &[],
            &configured,
        )
        .unwrap();

        output.rewind().unwrap();
        let layout = scan(
            &mut output,
            &configured,
            &mut MetadataBudget::new(configured),
        )
        .unwrap()
        .unwrap();
        assert!(layout
            .blocks
            .iter()
            .all(|block| block.block_type != PICTURE));
        assert_eq!(
            read_body(&mut output, layout.audio_offset, audio.len()).unwrap(),
            audio
        );
    }

    #[test]
    fn rewrite_absorbs_growth_into_padding_and_preserves_other_bytes() {
        let configured = limits();
        let destination = comment_body(&snapshot("v", &[("TITLE", "old")]), configured);
        let application = vec![0x5a; 73];
        let audio = b"\xff\xf8audio-suffix-must-remain-exact";
        let original = flac(
            &[
                (STREAMINFO, streaminfo()),
                (2, application.clone()),
                (VORBIS_COMMENT, destination),
                (PADDING, vec![0xcc; 128]),
            ],
            audio,
        );
        let old_layout = scan_bytes(original.clone(), configured).unwrap().unwrap();
        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&original).unwrap();
        rewrite(
            &mut output,
            &snapshot(
                "source-vendor",
                &[("X-CUSTOM", "a deliberately longer retained value")],
            ),
            &[],
            &configured,
        )
        .unwrap();

        output.rewind().unwrap();
        let new_layout = scan(
            &mut output,
            &configured,
            &mut MetadataBudget::new(configured),
        )
        .unwrap()
        .unwrap();
        assert_eq!(new_layout.audio_offset, old_layout.audio_offset);
        assert_eq!(new_layout.file_len, old_layout.file_len);
        assert_eq!(new_layout.comments.unwrap().vendor, "source-vendor");

        let application_block = new_layout
            .blocks
            .iter()
            .find(|block| block.block_type == 2)
            .unwrap();
        let copied = read_body(
            &mut output,
            application_block.body_offset,
            application_block.body_len as usize,
        )
        .unwrap();
        assert_eq!(copied, application);
        let suffix = read_body(&mut output, new_layout.audio_offset, audio.len()).unwrap();
        assert_eq!(suffix, audio);
    }

    #[test]
    fn rewrite_expands_existing_padding_when_comment_shrinks() {
        let configured = limits();
        let destination = comment_body(
            &snapshot(
                "destination-vendor-is-deliberately-long",
                &[("TITLE", "old")],
            ),
            configured,
        );
        let audio = b"audio-after-stable-offset";
        let original = flac(
            &[
                (STREAMINFO, streaminfo()),
                (VORBIS_COMMENT, destination),
                (PADDING, vec![0xa7; 9]),
            ],
            audio,
        );
        let old_layout = scan_bytes(original.clone(), configured).unwrap().unwrap();
        let old_padding = old_layout
            .blocks
            .iter()
            .find(|block| block.block_type == PADDING)
            .unwrap()
            .body_len;
        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&original).unwrap();
        rewrite(
            &mut output,
            &snapshot("v", &[("TITLE", "old")]),
            &[],
            &configured,
        )
        .unwrap();

        output.rewind().unwrap();
        let layout = scan(
            &mut output,
            &configured,
            &mut MetadataBudget::new(configured),
        )
        .unwrap()
        .unwrap();
        assert_eq!(layout.audio_offset, old_layout.audio_offset);
        assert!(
            layout
                .blocks
                .iter()
                .find(|block| block.block_type == PADDING)
                .unwrap()
                .body_len
                > old_padding
        );
        let suffix = read_body(&mut output, layout.audio_offset, audio.len()).unwrap();
        assert_eq!(suffix, audio);
    }

    #[test]
    fn rewrite_streams_suffix_when_growth_exceeds_padding() {
        let configured = limits();
        let destination = comment_body(&snapshot("v", &[]), configured);
        let mut audio = vec![0_u8; COPY_BUFFER_BYTES * 3 + 17];
        for (index, byte) in audio.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        audio[..2].copy_from_slice(b"\xff\xf8");
        let original = flac(
            &[
                (STREAMINFO, streaminfo()),
                (VORBIS_COMMENT, destination),
                (PADDING, vec![0; 2]),
            ],
            &audio,
        );
        let old_layout = scan_bytes(original.clone(), configured).unwrap().unwrap();
        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&original).unwrap();
        rewrite(
            &mut output,
            &snapshot("much-longer-source-vendor", &[("COMMENT", "expanded")]),
            &[],
            &configured,
        )
        .unwrap();

        output.rewind().unwrap();
        let new_layout = scan(
            &mut output,
            &configured,
            &mut MetadataBudget::new(configured),
        )
        .unwrap()
        .unwrap();
        assert!(new_layout.audio_offset > old_layout.audio_offset);
        let suffix = read_body(&mut output, new_layout.audio_offset, audio.len()).unwrap();
        assert_eq!(suffix, audio);
    }

    #[test]
    fn rewrite_inserts_the_only_comment_after_streaminfo() {
        let configured = limits();
        let application = vec![0x31; 19];
        let audio = b"audio-after-inserted-comment";
        let original = flac(
            &[(STREAMINFO, streaminfo()), (2, application.clone())],
            audio,
        );
        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&original).unwrap();
        rewrite(
            &mut output,
            &snapshot("source", &[("TITLE", "inserted")]),
            &[],
            &configured,
        )
        .unwrap();

        output.rewind().unwrap();
        let layout = scan(
            &mut output,
            &configured,
            &mut MetadataBudget::new(configured),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            layout
                .blocks
                .iter()
                .map(|block| block.block_type)
                .collect::<Vec<_>>(),
            vec![STREAMINFO, VORBIS_COMMENT, 2]
        );
        let application_block = &layout.blocks[2];
        assert_eq!(
            read_body(
                &mut output,
                application_block.body_offset,
                application_block.body_len as usize,
            )
            .unwrap(),
            application
        );
        assert_eq!(
            read_body(&mut output, layout.audio_offset, audio.len()).unwrap(),
            audio
        );
    }

    #[test]
    fn comment_insertion_reuses_padding_slot_at_the_block_count_limit() {
        let mut configured = limits();
        configured.max_flac_blocks = 2;
        let audio = b"audio-after-padding-slot-reuse";
        let original = flac(&[(STREAMINFO, streaminfo()), (PADDING, vec![0; 48])], audio);
        assert!(scan_bytes(original.clone(), configured).unwrap().is_some());
        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&original).unwrap();
        rewrite(
            &mut output,
            &snapshot("v", &[("TITLE", "fits in the old padding slot")]),
            &[],
            &configured,
        )
        .unwrap();

        output.rewind().unwrap();
        let layout = scan(
            &mut output,
            &configured,
            &mut MetadataBudget::new(configured),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            layout
                .blocks
                .iter()
                .map(|block| block.block_type)
                .collect::<Vec<_>>(),
            vec![STREAMINFO, VORBIS_COMMENT]
        );
        assert_eq!(
            read_body(&mut output, layout.audio_offset, audio.len()).unwrap(),
            audio
        );
    }

    #[test]
    fn duplicate_comment_blocks_are_rejected() {
        let configured = limits();
        let first = comment_body(&snapshot("first", &[("ONE", "1")]), configured);
        let second = comment_body(&snapshot("second", &[("TWO", "2")]), configured);
        let original = flac(
            &[
                (STREAMINFO, streaminfo()),
                (VORBIS_COMMENT, first),
                (2, vec![9; 17]),
                (VORBIS_COMMENT, second),
                (PADDING, vec![0; 64]),
            ],
            b"audio",
        );
        let error = scan_bytes(original, configured).unwrap_err();
        assert!(error.contains("more than one Vorbis comment"), "{error}");
    }
}
