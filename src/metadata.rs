//! Cross-container audio metadata preservation.
//!
//! Lofty provides a useful format-agnostic view of tags, but converting a
//! concrete tag to that view is intentionally lossy for fields which have no
//! [`lofty::tag::ItemKey`] mapping.  In particular, arbitrary Vorbis Comment
//! fields (including the de-facto `CHAPTER*` fields) are not represented by a
//! [`lofty::tag::Tag`].  This module therefore keeps both the generic tag and a
//! lossless snapshot of Vorbis Comments when the source and destination use a
//! Vorbis-compatible container.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::Path;

use lofty::config::{GlobalOptions, WriteOptions, apply_global_options};
use lofty::error::ErrorKind;
use lofty::file::{AudioFile, TaggedFileExt};
use lofty::tag::Tag;

/// Metadata captured from an input file.
///
/// The generic tag remains available through [`Metadata::tag`].  The private
/// Vorbis snapshot is applied only when the destination is FLAC or Ogg, where
/// the original arbitrary comment keys are representable.
#[derive(Clone)]
pub struct Metadata {
    tag: Tag,
    vorbis_comments: Option<VorbisCommentsSnapshot>,
}

impl Metadata {
    /// Returns the format-agnostic tag.
    #[must_use]
    pub fn tag(&self) -> &Tag {
        &self.tag
    }

    /// Consumes the snapshot and returns its format-agnostic tag.
    #[must_use]
    pub fn into_tag(self) -> Tag {
        self.tag
    }
}

/// Read the input metadata, including all generic tags and arbitrary Vorbis
/// Comment fields.
pub fn read_extended(input: &Path) -> Result<Option<Metadata>, String> {
    configure_lofty();

    let tag = match lofty::read_from_path(input) {
        Ok(source) => merge_tags(&source),
        // Lofty intentionally does not claim every audio container. Audio
        // decoding still handles those containers, so an absent tag reader is
        // equivalent to “no metadata” rather than a processing failure.
        Err(error) if matches!(error.kind(), ErrorKind::UnknownFormat) => None,
        Err(error) => return Err(format!("read metadata from {}: {error}", input.display())),
    };

    let vorbis_comments = if tag.is_some() {
        read_vorbis_comments(input)?
    } else {
        None
    };
    Ok(tag.map(|tag| Metadata {
        tag,
        vorbis_comments,
    }))
}

/// Read the generic representation of the input tag.
///
/// This compatibility wrapper is retained for embedders that used the
/// original API. Processing paths should use [`read_extended`] so that custom
/// Vorbis fields and chapter comments are retained as well.
pub fn read(input: &Path) -> Result<Option<Tag>, String> {
    read_extended(input).map(|metadata| metadata.map(Metadata::into_tag))
}

/// Write an extended metadata snapshot to an encoded output file.
pub fn write_extended(metadata: Metadata, output: &Path) -> Result<(), String> {
    configure_lofty();

    let Metadata {
        mut tag,
        vorbis_comments,
    } = metadata;
    let mut destination = lofty::read_from_path(output)
        .map_err(|error| format!("read output metadata from {}: {error}", output.display()))?;
    let target_type = destination.primary_tag_type();
    if tag.tag_type() != target_type {
        tag.re_map(target_type);
    }
    let has_pictures = tag.picture_count() > 0;
    destination.insert_tag(tag);
    destination
        .save_to_path(output, WriteOptions::default())
        .map_err(|error| format!("write metadata to {}: {error}", output.display()))?;

    if let Some(source_comments) = vorbis_comments {
        // A generic Vorbis writer already serializes pictures from Tag. Avoid
        // adding a second METADATA_BLOCK_PICTURE copy when merging raw fields.
        apply_vorbis_comments(&source_comments, output, has_pictures)?;
    }

    Ok(())
}

/// Write a generic tag to an encoded output file.
///
/// This compatibility wrapper intentionally has no raw-container snapshot.
pub fn write(tag: Tag, output: &Path) -> Result<(), String> {
    write_extended(
        Metadata {
            tag,
            vorbis_comments: None,
        },
        output,
    )
}

/// Copy all supported metadata from `input` to `output`.
/// Returns `false` when the input has no metadata.
pub fn copy(input: &Path, output: &Path) -> Result<bool, String> {
    let Some(metadata) = read_extended(input)? else {
        return Ok(false);
    };
    write_extended(metadata, output)?;
    Ok(true)
}

fn configure_lofty() {
    // This is Lofty's default, but making it explicit protects native ID3v2
    // frames (CHAP/CTOC, TXXX variants, and other binary frames) and MP4
    // atoms when the tag is written back without a type remap.
    let mut options = GlobalOptions::new();
    options.preserve_format_specific_items(true);
    apply_global_options(options);
}

fn merge_tags(source: &lofty::file::TaggedFile) -> Option<Tag> {
    let primary = source.primary_tag().or_else(|| source.first_tag())?;
    let mut merged = primary.clone();

    // Files such as MP3 can contain ID3v2, ID3v1, and APE tags at once. Keep
    // the primary tag's native companion, then add values from every other
    // tag which are not already present. Duplicated values are meaningful for
    // artist/comment fields, so only exact TagItem duplicates are removed.
    for tag in source.tags() {
        if std::ptr::eq(tag, primary) {
            continue;
        }
        for item in tag.items() {
            if !merged.items().any(|existing| existing == item) {
                merged.push_unchecked(item.clone());
            }
        }
        for picture in tag.pictures() {
            if !merged.pictures().contains(picture) {
                merged.push_picture(picture.clone());
            }
        }
    }

    Some(merged)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VorbisCommentsSnapshot {
    vendor: String,
    items: Vec<(String, String)>,
}

impl VorbisCommentsSnapshot {
    fn new(vendor: String, items: Vec<(String, String)>) -> Self {
        Self { vendor, items }
    }
}

fn read_vorbis_comments(input: &Path) -> Result<Option<VorbisCommentsSnapshot>, String> {
    // Avoid loading an entire MP3/WAV/M4A into memory just to discover that it
    // is not a Vorbis-compatible container. The raw parser is only needed for
    // FLAC and Ogg files.
    let mut prefix = [0_u8; 4];
    File::open(input)
        .and_then(|mut file| file.read_exact(&mut prefix))
        .map_err(|error| format!("read raw metadata from {}: {error}", input.display()))?;
    if !prefix.eq(b"fLaC") && !prefix.eq(b"OggS") {
        return Ok(None);
    }
    let bytes = fs::read(input)
        .map_err(|error| format!("read raw metadata from {}: {error}", input.display()))?;

    if bytes.starts_with(b"fLaC") {
        // A malformed block is ignored here; Lofty has already parsed the
        // file, and generic metadata should remain usable in that case.
        return Ok(parse_flac_comments(&bytes));
    }
    if bytes.starts_with(b"OggS") {
        return Ok(parse_ogg_comments(&bytes));
    }
    Ok(None)
}

fn parse_comment_body(body: &[u8]) -> Option<VorbisCommentsSnapshot> {
    let mut offset = 0;
    let vendor_len = read_u32_le(body, &mut offset)? as usize;
    let vendor =
        String::from_utf8(body.get(offset..offset.checked_add(vendor_len)?)?.to_vec()).ok()?;
    offset = offset.checked_add(vendor_len)?;

    let count = read_u32_le(body, &mut offset)? as usize;
    let mut items = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let length = read_u32_le(body, &mut offset)? as usize;
        let value =
            String::from_utf8(body.get(offset..offset.checked_add(length)?)?.to_vec()).ok()?;
        offset = offset.checked_add(length)?;
        let (key, value) = value.split_once('=')?;
        if key.is_empty() {
            return None;
        }
        items.push((key.to_owned(), value.to_owned()));
    }

    Some(VorbisCommentsSnapshot::new(vendor, items))
}

fn serialize_comment_body(snapshot: &VorbisCommentsSnapshot) -> Result<Vec<u8>, String> {
    let vendor = snapshot.vendor.as_bytes();
    let mut items = Vec::with_capacity(snapshot.items.len());
    for (key, value) in &snapshot.items {
        if key.is_empty() || key.contains('=') || key.bytes().any(|byte| byte > 0x7f) {
            continue;
        }
        items.push(format!("{key}={value}"));
    }
    let vendor_len = u32::try_from(vendor.len()).map_err(|_| "Vorbis vendor is too large")?;
    let item_count = u32::try_from(items.len()).map_err(|_| "too many Vorbis comments")?;
    let mut output = Vec::new();
    output.extend(vendor_len.to_le_bytes());
    output.extend(vendor);
    output.extend(item_count.to_le_bytes());
    for item in items {
        let bytes = item.as_bytes();
        let length = u32::try_from(bytes.len()).map_err(|_| "Vorbis comment is too large")?;
        output.extend(length.to_le_bytes());
        output.extend(bytes);
    }
    Ok(output)
}

fn read_u32_le(data: &[u8], offset: &mut usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let bytes = data.get(*offset..end)?;
    *offset = end;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

#[derive(Clone, Copy)]
enum OggCommentPrefix {
    Vorbis,
    Opus,
    Speex,
}

impl OggCommentPrefix {
    fn identification(data: &[u8]) -> Option<Self> {
        if data.starts_with(b"\x01vorbis") {
            Some(Self::Vorbis)
        } else if data.starts_with(b"OpusHead") {
            Some(Self::Opus)
        } else if data.starts_with(b"Speex   ") {
            Some(Self::Speex)
        } else {
            None
        }
    }

    fn comment_prefix(self) -> &'static [u8] {
        match self {
            Self::Vorbis => b"\x03vorbis",
            Self::Opus => b"OpusTags",
            Self::Speex => b"SpeexTags",
        }
    }
}

fn parse_ogg_comments(bytes: &[u8]) -> Option<VorbisCommentsSnapshot> {
    let mut reader = ogg::PacketReader::new(Cursor::new(bytes));
    let mut candidate: Option<(u32, OggCommentPrefix)> = None;
    while let Ok(Some(packet)) = reader.read_packet() {
        if candidate.is_none() {
            if let Some(prefix) = OggCommentPrefix::identification(&packet.data) {
                candidate = Some((packet.stream_serial(), prefix));
            }
            continue;
        }
        let (serial, prefix) = candidate.expect("candidate is set");
        if packet.stream_serial() != serial {
            continue;
        }
        let comment_prefix = prefix.comment_prefix();
        if packet.data.starts_with(comment_prefix) {
            return parse_comment_body(&packet.data[comment_prefix.len()..]);
        }
        return None;
    }
    None
}

fn parse_flac_comments(bytes: &[u8]) -> Option<VorbisCommentsSnapshot> {
    let blocks = parse_flac_blocks(bytes)?;
    blocks
        .iter()
        .find(|(block_type, _)| *block_type == 4)
        .and_then(|(_, body)| parse_comment_body(body))
}

fn parse_flac_blocks(bytes: &[u8]) -> Option<Vec<(u8, Vec<u8>)>> {
    if !bytes.starts_with(b"fLaC") {
        return None;
    }
    let mut offset: usize = 4;
    let mut blocks = Vec::new();
    loop {
        let header = bytes.get(offset..offset.checked_add(4)?)?;
        let is_last = header[0] & 0x80 != 0;
        let block_type = header[0] & 0x7f;
        let length = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;
        offset = offset.checked_add(4)?;
        let end = offset.checked_add(length)?;
        blocks.push((block_type, bytes.get(offset..end)?.to_vec()));
        offset = end;
        if is_last {
            break;
        }
    }
    Some(blocks)
}

fn merge_vorbis_comments(
    source: &VorbisCommentsSnapshot,
    mut destination: VorbisCommentsSnapshot,
    skip_picture_comments: bool,
) -> VorbisCommentsSnapshot {
    if !source.vendor.is_empty() {
        destination.vendor.clone_from(&source.vendor);
    }

    // The generic writer has already emitted all mapped fields. Treat those
    // values as a multiset and append only the source occurrences which are
    // still missing; this retains custom fields and repeated artist/chapter
    // values without duplicating ordinary fields or cover art.
    let mut remaining: HashMap<(String, String), usize> = HashMap::new();
    for (key, value) in &destination.items {
        *remaining
            .entry((key.to_ascii_lowercase(), value.clone()))
            .or_default() += 1;
    }
    for (key, value) in &source.items {
        if skip_picture_comments && key.eq_ignore_ascii_case("METADATA_BLOCK_PICTURE") {
            continue;
        }
        let map_key = (key.to_ascii_lowercase(), value.clone());
        if let Some(count) = remaining.get_mut(&map_key) {
            if *count > 0 {
                *count -= 1;
                continue;
            }
        }
        destination.items.push((key.clone(), value.clone()));
    }
    destination
}

fn apply_vorbis_comments(
    source: &VorbisCommentsSnapshot,
    output: &Path,
    skip_picture_comments: bool,
) -> Result<(), String> {
    let bytes = fs::read(output)
        .map_err(|error| format!("read output metadata from {}: {error}", output.display()))?;
    if bytes.starts_with(b"fLaC") {
        return rewrite_flac_comments(&bytes, source, output, skip_picture_comments);
    }
    if bytes.starts_with(b"OggS") {
        return rewrite_ogg_comments(&bytes, source, output, skip_picture_comments);
    }
    Ok(())
}

fn rewrite_flac_comments(
    bytes: &[u8],
    source: &VorbisCommentsSnapshot,
    output: &Path,
    skip_picture_comments: bool,
) -> Result<(), String> {
    let Some(mut blocks) = parse_flac_blocks(bytes) else {
        return Ok(());
    };
    let existing = blocks
        .iter()
        .find(|(block_type, _)| *block_type == 4)
        .and_then(|(_, body)| parse_comment_body(body))
        .unwrap_or_else(|| VorbisCommentsSnapshot::new(String::new(), Vec::new()));
    let merged = merge_vorbis_comments(source, existing, skip_picture_comments);
    let comment = serialize_comment_body(&merged)?;
    if let Some((_, body)) = blocks.iter_mut().find(|(block_type, _)| *block_type == 4) {
        *body = comment;
    } else {
        let insert_at = blocks.len().min(1);
        blocks.insert(insert_at, (4, comment));
    }

    let audio_offset = flac_audio_offset(bytes).ok_or("invalid FLAC metadata blocks")?;
    let mut rewritten = Vec::with_capacity(bytes.len() + 64);
    rewritten.extend(b"fLaC");
    for (index, (block_type, body)) in blocks.iter().enumerate() {
        let length = u32::try_from(body.len()).map_err(|_| "FLAC metadata block is too large")?;
        if length > 0x00ff_ffff {
            return Err("FLAC metadata block exceeds the 24-bit size limit".into());
        }
        let first = if index + 1 == blocks.len() {
            0x80 | block_type
        } else {
            *block_type
        };
        rewritten.push(first);
        rewritten.push((length >> 16) as u8);
        rewritten.push((length >> 8) as u8);
        rewritten.push(length as u8);
        rewritten.extend(body);
    }
    rewritten.extend(&bytes[audio_offset..]);
    replace_file(output, &rewritten)
}

fn flac_audio_offset(bytes: &[u8]) -> Option<usize> {
    if !bytes.starts_with(b"fLaC") {
        return None;
    }
    let mut offset: usize = 4;
    loop {
        let header = bytes.get(offset..offset.checked_add(4)?)?;
        let is_last = header[0] & 0x80 != 0;
        let length = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;
        offset = offset.checked_add(4)?.checked_add(length)?;
        if is_last {
            return Some(offset);
        }
    }
}

fn rewrite_ogg_comments(
    bytes: &[u8],
    source: &VorbisCommentsSnapshot,
    output: &Path,
    skip_picture_comments: bool,
) -> Result<(), String> {
    let mut reader = ogg::PacketReader::new(Cursor::new(bytes));
    let mut packets = Vec::new();
    let mut target_serial = None;
    let mut target_prefix = None;
    let mut replaced = false;

    while let Some(packet) = reader
        .read_packet()
        .map_err(|error| format!("read Ogg metadata: {error}"))?
    {
        let serial = packet.stream_serial();
        if target_serial.is_none() {
            if let Some(prefix) = OggCommentPrefix::identification(&packet.data) {
                target_serial = Some(serial);
                target_prefix = Some(prefix);
            }
        } else if !replaced && Some(serial) == target_serial {
            let prefix = target_prefix.expect("target prefix is set");
            let comment_prefix = prefix.comment_prefix();
            if packet.data.starts_with(comment_prefix) {
                let existing = parse_comment_body(&packet.data[comment_prefix.len()..])
                    .unwrap_or_else(|| VorbisCommentsSnapshot::new(String::new(), Vec::new()));
                let merged = merge_vorbis_comments(source, existing, skip_picture_comments);
                let mut data = comment_prefix.to_vec();
                data.extend(serialize_comment_body(&merged)?);
                packets.push((data, serial, packet.absgp_page(), packet.last_in_stream()));
                replaced = true;
                continue;
            }
        }
        let granule = packet.absgp_page();
        let last = packet.last_in_stream();
        packets.push((packet.data, serial, granule, last));
    }

    if !replaced {
        return Ok(());
    }

    let temp = temporary_path(output);
    let file = File::create(&temp)
        .map_err(|error| format!("create temporary metadata file {}: {error}", temp.display()))?;
    let mut writer = ogg::PacketWriter::new(std::io::BufWriter::new(file));
    for (data, serial, granule, last) in packets {
        let end = if last {
            ogg::PacketWriteEndInfo::EndStream
        } else {
            ogg::PacketWriteEndInfo::EndPage
        };
        writer
            .write_packet(Cow::Owned(data), serial, end, granule)
            .map_err(|error| format!("write Ogg metadata: {error}"))?;
    }
    let mut file = writer
        .into_inner()
        .into_inner()
        .map_err(|error| format!("flush temporary metadata file: {error}"))?;
    file.flush()
        .map_err(|error| format!("flush Ogg metadata: {error}"))?;
    drop(file);
    fs::rename(&temp, output)
        .map_err(|error| format!("replace Ogg metadata in {}: {error}", output.display()))
}

fn temporary_path(output: &Path) -> std::path::PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    output.with_file_name(format!(".{name}.metadata.part"))
}

fn replace_file(output: &Path, bytes: &[u8]) -> Result<(), String> {
    let temp = temporary_path(output);
    fs::write(&temp, bytes)
        .map_err(|error| format!("write temporary metadata file {}: {error}", temp.display()))?;
    fs::rename(&temp, output)
        .map_err(|error| format!("replace metadata in {}: {error}", output.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lofty::tag::{Accessor, Tag, TagType};

    #[test]
    fn comment_body_round_trips_unknown_and_chapter_fields() {
        let source = VorbisCommentsSnapshot::new(
            "vendor".into(),
            vec![
                ("TITLE".into(), "A title".into()),
                ("CHAPTER001".into(), "00:00:01.000".into()),
                ("CHAPTER001NAME".into(), "Intro".into()),
                ("X-CUSTOM".into(), "retained".into()),
            ],
        );
        let encoded = serialize_comment_body(&source).unwrap();
        assert_eq!(parse_comment_body(&encoded), Some(source));
    }

    #[test]
    fn merges_all_tags_and_pictures_without_exact_duplicates() {
        let mut primary = Tag::new(TagType::Id3v2);
        primary.set_title("title".into());
        let mut secondary = Tag::new(TagType::Id3v1);
        secondary.set_artist("artist".into());
        let mut destination = primary.clone();
        for item in secondary.items() {
            if !destination.items().any(|existing| existing == item) {
                destination.push_unchecked(item.clone());
            }
        }
        assert_eq!(destination.title().as_deref(), Some("title"));
        assert_eq!(destination.artist().as_deref(), Some("artist"));
    }

    #[test]
    fn flac_blocks_are_parsed() {
        let mut bytes = b"fLaC".to_vec();
        bytes.extend([0x80, 0, 0, 4]);
        bytes.extend([0, 0, 0, 0]);
        let blocks = parse_flac_blocks(&bytes).unwrap();
        assert_eq!(blocks, vec![(0, vec![0, 0, 0, 0])]);
    }
}
