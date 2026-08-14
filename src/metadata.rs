//! Cross-container audio metadata preservation.
//!
//! Lofty provides a useful format-agnostic view of tags, but converting a
//! concrete tag to that view is intentionally lossy for fields which have no
//! [`lofty::tag::ItemKey`] mapping. In particular, arbitrary Vorbis Comment
//! fields (including the de-facto `CHAPTER*` fields) are retained separately
//! and written back losslessly to Vorbis-compatible outputs.

mod flac;
mod limits;
mod ogg;

pub use limits::MetadataLimits;

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::read::DecoderReader;
use base64::write::EncoderWriter;
use lofty::config::{apply_global_options, GlobalOptions, ParseOptions, WriteOptions};
use lofty::error::ErrorKind;
use lofty::file::{AudioFile, TaggedFileExt};
use lofty::ogg::{OggPictureStorage, VorbisComments};
use lofty::picture::{Picture, PictureInformation};
use lofty::probe::Probe;
use lofty::tag::{ItemValue, Tag, TagType};

use crate::atomic_output::{AtomicOutput, CommitMode};

const RESIDENT_REPRESENTATION_MULTIPLIER: u64 = 16;
const RESIDENT_ITEM_DESCRIPTOR_BYTES: u64 = 256;
const RESIDENT_BASE_BYTES: u64 = 1024;

/// Metadata captured from an input file.
///
/// The generic tag remains available through [`Metadata::tag`]. The private
/// Vorbis snapshot is applied only when the destination is FLAC or Ogg, where
/// the original vendor and arbitrary comment keys are representable.
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

    /// Conservatively estimate the denoize-owned memory retained by this snapshot.
    ///
    /// The estimate charges generic and raw Vorbis representations separately,
    /// includes per-item allocation overhead, and leaves room for the temporary
    /// representations created while rewriting a destination container. It is
    /// admission accounting rather than an allocator-exact heap measurement.
    #[must_use]
    pub fn estimated_memory_bytes(&self) -> u64 {
        let mut payload_bytes = 0u64;
        let mut items = 0u64;
        for item in self.tag.items() {
            let value_len = match item.value() {
                ItemValue::Text(value) | ItemValue::Locator(value) => value.len(),
                ItemValue::Binary(value) => value.len(),
            };
            payload_bytes = payload_bytes
                .saturating_add(item.description().len() as u64)
                .saturating_add(value_len as u64);
            items = items.saturating_add(1);
        }
        for picture in self.tag.pictures() {
            payload_bytes = payload_bytes
                .saturating_add(picture.data().len() as u64)
                .saturating_add(picture.description().map_or(0, str::len) as u64)
                .saturating_add(picture.mime_type().map_or(0, |mime| mime.as_str().len()) as u64);
            items = items.saturating_add(1);
        }
        if let Some(vorbis) = &self.vorbis_comments {
            payload_bytes = payload_bytes.saturating_add(vorbis.vendor.len() as u64);
            items = items.saturating_add(1);
            for (key, value) in &vorbis.items {
                payload_bytes = payload_bytes
                    .saturating_add(key.len() as u64)
                    .saturating_add(value.len() as u64);
                items = items.saturating_add(1);
            }
        }
        payload_bytes
            .saturating_add(items.saturating_mul(RESIDENT_ITEM_DESCRIPTOR_BYTES))
            .saturating_add(RESIDENT_BASE_BYTES)
            .saturating_mul(RESIDENT_REPRESENTATION_MULTIPLIER)
    }
}

/// Read metadata from a regular-file input with finite default resource limits.
///
/// FIFOs, directories, and device files are rejected before parsing.
pub fn read_extended(input: &Path) -> Result<Option<Metadata>, String> {
    read_extended_with_limits(input, MetadataLimits::default())
}

/// Read metadata from a regular-file input with caller-selected resource limits.
///
/// FLAC blocks and Ogg pages are inspected with streaming readers before
/// Lofty is allowed to parse the generic tag. This bounds attacker-controlled
/// allocations without retaining an entire audio file in memory.
pub fn read_extended_with_limits(
    input: &Path,
    limits: MetadataLimits,
) -> Result<Option<Metadata>, String> {
    let mut session = crate::input::AudioInputSession::open(input)?;
    session.read_metadata_with_limits(limits)
}

/// Read metadata through an already validated regular-file handle.
///
/// The caller keeps ownership of the handle and may rewind it for subsequent
/// probing or decoding. Detection, raw-container validation, and Lofty's
/// generic parser all use this same open file description.
pub(crate) fn read_extended_from_file_with_limits(
    source_file: &mut File,
    input: &Path,
    limits: MetadataLimits,
) -> Result<Option<Metadata>, String> {
    // Keep detection, the raw-container scan, and Lofty's generic read on one
    // open file description. Besides bounding each parser, this prevents a
    // pathname replacement from mixing raw comments from one inode with a
    // generic tag from another.
    let container = detect_file_container(source_file)
        .map_err(|error| format!("read metadata signature from {}: {error}", input.display()))?;
    let mut raw_budget = MetadataBudget::new(limits);
    let vorbis_comments = match container {
        RawContainer::Flac => flac::read_file(source_file, &limits, &mut raw_budget)?,
        RawContainer::Ogg => ogg::read_file(source_file, &limits, &mut raw_budget)?,
        RawContainer::Other => None,
    };

    // A zero optional-payload budget is still sufficient for a tagless raw
    // FLAC or Ogg stream: the raw scan above has proved that there is no
    // optional comment or picture to retain. Lofty's per-allocation limit
    // treats zero as an immediate error even though there is no generic tag to
    // retain, so avoid invoking it in this exact case.
    if matches!(container, RawContainer::Flac | RawContainer::Ogg)
        && vorbis_comments.is_none()
        && limits.max_item_bytes.min(limits.max_total_bytes) == 0
    {
        return Ok(None);
    }

    configure_lofty(limits);
    let source = match read_lofty_file(source_file, limits) {
        Ok(source) => source,
        // Audio decoding intentionally supports some containers for which
        // Lofty has no tag reader. Treat those as having no generic metadata.
        Err(error) if matches!(error.kind(), ErrorKind::UnknownFormat) => {
            return Ok(vorbis_comments.map(|vorbis_comments| Metadata {
                tag: Tag::new(TagType::VorbisComments),
                vorbis_comments: Some(vorbis_comments),
            }));
        }
        Err(error) => return Err(format!("read metadata from {}: {error}", input.display())),
    };

    preflight_source_tags(&source, limits)?;
    let tag = merge_tags(&source);
    let Some(tag) = tag.or_else(|| {
        vorbis_comments
            .as_ref()
            .map(|_| Tag::new(TagType::VorbisComments))
    }) else {
        return Ok(None);
    };

    // Raw FLAC/Ogg scanners have already accounted for the exact comment and
    // picture representations. Other formats are accounted from Lofty's
    // bounded generic representation.
    if container == RawContainer::Other {
        let mut budget = MetadataBudget::new(limits);
        charge_tag(&tag, &mut budget)?;
    }

    Ok(Some(Metadata {
        tag,
        vorbis_comments,
    }))
}

/// Read the generic representation of the input tag.
///
/// This compatibility wrapper is retained for embedders that used the
/// original API. Processing paths should use [`read_extended`] so custom
/// Vorbis fields and chapter comments are retained as well.
pub fn read(input: &Path) -> Result<Option<Tag>, String> {
    read_extended(input).map(|metadata| metadata.map(Metadata::into_tag))
}

/// Write an extended metadata snapshot with finite default resource limits.
pub fn write_extended(metadata: Metadata, output: &Path) -> Result<(), String> {
    write_extended_with_limits(metadata, output, MetadataLimits::default())
}

/// Write an extended metadata snapshot with caller-selected resource limits.
pub fn write_extended_with_limits(
    metadata: Metadata,
    output: &Path,
    limits: MetadataLimits,
) -> Result<(), String> {
    validate_metadata(&metadata, limits)?;

    let mut transaction = AtomicOutput::new(output)?;
    // Reject FIFOs, devices, and directories before reading any destination
    // bytes. Opening the transaction first fixes the canonical destination
    // parent, while dropping it after a rejection removes only the empty stage.
    let fixed_output = transaction.destination_path().to_path_buf();
    let source_session = crate::input::AudioInputSession::open(&fixed_output)?;
    let mut source = source_session.into_file_rewound("output metadata source")?;
    io::copy(&mut source, transaction.file_mut())
        .map_err(|error| format!("stage output metadata from {}: {error}", output.display()))?;
    drop(source);
    write_extended_to_file_with_limits(metadata, transaction.file_mut(), limits)?;
    transaction.commit(CommitMode::Replace)
}

/// Write an extended snapshot through an existing file handle with defaults.
///
/// The caller should provide a private staged file: failures can leave this
/// handle partially modified, while a separately managed destination remains
/// untouched until the stage is committed.
pub fn write_extended_to_file(metadata: Metadata, output: &mut File) -> Result<(), String> {
    write_extended_to_file_with_limits(metadata, output, MetadataLimits::default())
}

/// Write an extended snapshot through a handle with caller-selected limits.
///
/// The caller should provide a private staged file. As with
/// [`write_extended_to_file`], a failure may leave this handle modified.
pub fn write_extended_to_file_with_limits(
    metadata: Metadata,
    output: &mut File,
    limits: MetadataLimits,
) -> Result<(), String> {
    validate_metadata(&metadata, limits)?;

    let container = detect_file_container(output)?;
    if container != RawContainer::Other {
        return write_raw_metadata(metadata, output, container, limits);
    }

    let Metadata {
        mut tag,
        vorbis_comments: _,
    } = metadata;
    output
        .rewind()
        .map_err(|error| format!("rewind output metadata: {error}"))?;
    configure_lofty(limits);
    let mut destination = read_lofty_file(output, limits)
        .map_err(|error| format!("read output metadata: {error}"))?;
    preflight_source_tags(&destination, limits)
        .map_err(|error| format!("output metadata exceeds limits: {error}"))?;
    let mut destination_budget = MetadataBudget::new(limits);
    for destination_tag in destination.tags() {
        charge_tag(destination_tag, &mut destination_budget)
            .map_err(|error| format!("output metadata exceeds limits: {error}"))?;
    }
    let target_type = destination.primary_tag_type();
    if tag.tag_type() != target_type {
        tag.re_map(target_type);
    }
    destination.insert_tag(tag);
    let mut final_tag_budget = MetadataBudget::new(limits);
    for destination_tag in destination.tags() {
        charge_tag(destination_tag, &mut final_tag_budget)
            .map_err(|error| format!("rewritten output metadata exceeds limits: {error}"))?;
    }
    destination
        .save_to(&mut *output, WriteOptions::default())
        .map_err(|error| format!("write output metadata: {error}"))?;

    Ok(())
}

fn write_raw_metadata(
    metadata: Metadata,
    output: &mut File,
    container: RawContainer,
    limits: MetadataLimits,
) -> Result<(), String> {
    let mut destination_budget = MetadataBudget::new(limits);
    let destination = match container {
        RawContainer::Flac => flac::read_file(output, &limits, &mut destination_budget)?,
        RawContainer::Ogg => ogg::read_file(output, &limits, &mut destination_budget)?,
        RawContainer::Other => return Err("internal raw metadata container mismatch".into()),
    };
    if container == RawContainer::Ogg && destination.is_none() {
        return Err("Ogg output has no supported codec comment stream".into());
    }

    let Metadata {
        mut tag,
        vorbis_comments,
    } = metadata;
    let pictures = take_tag_pictures(&mut tag)?;
    let mut converted = VorbisComments::from(tag);
    debug_assert!(converted.pictures().is_empty());
    let mut generic = snapshot_from_lofty_comments(&mut converted, limits)?;

    // Lofty's FLAC/Ogg writers retain the encoded destination's codec vendor,
    // rather than treating EncoderSoftware as a replacement vendor. Match that
    // behavior without allowing those writers to read the audio payload.
    if let Some(destination) = destination {
        generic.vendor = destination.vendor;
    }

    let preserve_source_ogg_pictures = container == RawContainer::Ogg
        && vorbis_comments.as_ref().is_some_and(|snapshot| {
            snapshot
                .items
                .iter()
                .any(|(key, _)| is_vorbis_picture_key(key))
        });
    if container == RawContainer::Ogg && !preserve_source_ogg_pictures {
        append_ogg_picture_comments(&mut generic, &pictures, limits)?;
    }
    let has_replacement_pictures = !pictures.is_empty() && !preserve_source_ogg_pictures;
    let final_comments = if let Some(source) = vorbis_comments.as_ref() {
        merge_vorbis_comments(
            source,
            generic,
            has_replacement_pictures,
            &mut MetadataBudget::new(limits),
        )?
    } else {
        generic
    };

    match container {
        RawContainer::Flac => flac::rewrite(output, &final_comments, &pictures, &limits),
        RawContainer::Ogg => ogg::rewrite(output, &final_comments, &limits),
        RawContainer::Other => unreachable!("raw container checked above"),
    }
}

fn take_tag_pictures(tag: &mut Tag) -> Result<Vec<(Picture, PictureInformation)>, String> {
    let count = usize::try_from(tag.picture_count())
        .map_err(|_| "metadata picture count does not fit in memory".to_owned())?;
    let mut pictures = Vec::new();
    try_reserve_vec(&mut pictures, count, "converted metadata pictures")?;
    // Removing backwards avoids shifting the still-owned Tag contents. Reverse
    // once afterward to preserve the source picture order exactly.
    for index in (0..count).rev() {
        let picture = tag.remove_picture(index);
        let information = PictureInformation::from_picture(&picture).unwrap_or_default();
        pictures.push((picture, information));
    }
    pictures.reverse();
    Ok(pictures)
}

fn snapshot_from_lofty_comments(
    comments: &mut VorbisComments,
    limits: MetadataLimits,
) -> Result<VorbisCommentsSnapshot, String> {
    if comments.vendor().len() > limits.max_item_bytes {
        return Err(format!(
            "Vorbis vendor exceeds metadata item limit ({} > {} bytes)",
            comments.vendor().len(),
            limits.max_item_bytes
        ));
    }
    let vendor = try_clone_string(comments.vendor(), "converted Vorbis vendor")?;
    let count = comments.items().len();
    if count > limits.max_items {
        return Err(format!(
            "converted Vorbis comment count exceeds limit ({} > {})",
            count, limits.max_items
        ));
    }
    let mut items = Vec::new();
    try_reserve_vec(&mut items, count, "converted Vorbis comments")?;
    for (key, value) in comments.take_items() {
        validate_comment_key(key.as_bytes())?;
        let length = key
            .len()
            .checked_add(1)
            .and_then(|size| size.checked_add(value.len()))
            .ok_or_else(|| "converted Vorbis comment size overflow".to_owned())?;
        if length > limits.max_item_bytes {
            return Err(format!(
                "converted Vorbis comment exceeds metadata item limit ({} > {} bytes)",
                length, limits.max_item_bytes
            ));
        }
        items.push((key, value));
    }
    Ok(VorbisCommentsSnapshot::new(vendor, items))
}

fn append_ogg_picture_comments(
    comments: &mut VorbisCommentsSnapshot,
    pictures: &[(Picture, PictureInformation)],
    limits: MetadataLimits,
) -> Result<(), String> {
    let final_count = comments
        .items
        .len()
        .checked_add(pictures.len())
        .ok_or_else(|| "Ogg picture comment count overflow".to_owned())?;
    if final_count > limits.max_items {
        return Err(format!(
            "Ogg picture comments exceed metadata item count limit ({} > {})",
            final_count, limits.max_items
        ));
    }
    comments
        .items
        .try_reserve_exact(pictures.len())
        .map_err(|error| format!("reserve Ogg picture comments: {error}"))?;
    for (picture, information) in pictures {
        let value = encode_ogg_picture(picture, *information, limits)?;
        comments.items.push((
            try_clone_string("METADATA_BLOCK_PICTURE", "Ogg picture comment key")?,
            value,
        ));
    }
    Ok(())
}

fn encode_ogg_picture(
    picture: &Picture,
    information: PictureInformation,
    limits: MetadataLimits,
) -> Result<String, String> {
    let raw_len = picture_block_len(picture, information)?;
    if raw_len > limits.max_item_bytes {
        return Err(format!(
            "Ogg decoded picture exceeds metadata item limit ({} > {} bytes)",
            raw_len, limits.max_item_bytes
        ));
    }
    let encoded_len = base64::encoded_len(raw_len, true)
        .ok_or_else(|| "Ogg picture base64 length overflow".to_owned())?;
    let field_len = "METADATA_BLOCK_PICTURE"
        .len()
        .checked_add(1)
        .and_then(|size| size.checked_add(encoded_len))
        .ok_or_else(|| "Ogg picture comment size overflow".to_owned())?;
    if field_len > limits.max_item_bytes {
        return Err(format!(
            "Ogg picture comment exceeds metadata item limit ({} > {} bytes)",
            field_len, limits.max_item_bytes
        ));
    }

    let mut encoded = Vec::new();
    try_reserve_vec(&mut encoded, encoded_len, "Ogg picture base64")?;
    {
        let mut encoder = EncoderWriter::new(&mut encoded, &BASE64_STANDARD);
        write_picture_block(picture, information, &mut encoder)?;
        encoder
            .finish()
            .map_err(|error| format!("finish Ogg picture base64: {error}"))?;
    }
    if encoded.len() != encoded_len {
        return Err(format!(
            "internal Ogg picture base64 length mismatch ({} != {})",
            encoded.len(),
            encoded_len
        ));
    }
    String::from_utf8(encoded).map_err(|error| format!("encode Ogg picture base64: {error}"))
}

pub(super) fn picture_block_len(
    picture: &Picture,
    _information: PictureInformation,
) -> Result<usize, String> {
    let mime = picture.mime_type().map_or("", |mime| mime.as_str());
    let description = picture.description().unwrap_or("");
    let _ = u32::try_from(mime.len()).map_err(|_| "picture MIME type is too large")?;
    let _ = u32::try_from(description.len()).map_err(|_| "picture description is too large")?;
    let _ = u32::try_from(picture.data().len()).map_err(|_| "picture data is too large")?;
    32_usize
        .checked_add(mime.len())
        .and_then(|size| size.checked_add(description.len()))
        .and_then(|size| size.checked_add(picture.data().len()))
        .ok_or_else(|| "FLAC picture block size overflow".to_owned())
}

pub(super) fn write_picture_block<W: Write>(
    picture: &Picture,
    information: PictureInformation,
    output: &mut W,
) -> Result<(), String> {
    let _ = picture_block_len(picture, information)?;
    let mime = picture.mime_type().map_or("", |mime| mime.as_str());
    let description = picture.description().unwrap_or("");
    let mime_len = u32::try_from(mime.len()).map_err(|_| "picture MIME type is too large")?;
    let description_len =
        u32::try_from(description.len()).map_err(|_| "picture description is too large")?;
    let data_len = u32::try_from(picture.data().len()).map_err(|_| "picture data is too large")?;

    output
        .write_all(&u32::from(picture.pic_type().as_u8()).to_be_bytes())
        .and_then(|()| output.write_all(&mime_len.to_be_bytes()))
        .and_then(|()| output.write_all(mime.as_bytes()))
        .and_then(|()| output.write_all(&description_len.to_be_bytes()))
        .and_then(|()| output.write_all(description.as_bytes()))
        .and_then(|()| output.write_all(&information.width.to_be_bytes()))
        .and_then(|()| output.write_all(&information.height.to_be_bytes()))
        .and_then(|()| output.write_all(&information.color_depth.to_be_bytes()))
        .and_then(|()| output.write_all(&information.num_colors.to_be_bytes()))
        .and_then(|()| output.write_all(&data_len.to_be_bytes()))
        .and_then(|()| output.write_all(picture.data()))
        .map_err(|error| format!("write FLAC picture block: {error}"))
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

/// Validate FLAC container metadata before passing the same handle to the
/// decoder. The caller is responsible for seeking the handle after this call,
/// on both success and failure.
pub(crate) fn preflight_flac_decode(file: &mut File, limits: MetadataLimits) -> Result<(), String> {
    flac::validate(file, &limits)
}

/// Validate Ogg page, stream, packet, and metadata bounds before passing the
/// same handle to the decoder. The caller is responsible for seeking the
/// handle after this call, on both success and failure.
pub(crate) fn preflight_ogg_decode(file: &mut File, limits: MetadataLimits) -> Result<(), String> {
    ogg::validate(file, &limits)
}

fn configure_lofty(limits: MetadataLimits) {
    // Lofty applies these options per thread. The explicit allocation cap
    // complements our aggregate and count budgets for every individual read.
    let mut options = GlobalOptions::new();
    options.preserve_format_specific_items(true);
    options.allocation_limit(limits.max_item_bytes.min(limits.max_total_bytes));
    apply_global_options(options);
}

fn read_lofty_file(
    output: &mut File,
    limits: MetadataLimits,
) -> lofty::error::Result<lofty::file::TaggedFile> {
    output.rewind()?;
    read_lofty_reader(&mut *output, limits)
}

fn read_lofty_reader<R>(
    reader: R,
    limits: MetadataLimits,
) -> lofty::error::Result<lofty::file::TaggedFile>
where
    R: Read + Seek,
{
    // Metadata parsers also read small container headers. The fixed allowance
    // is an I/O bound, not an allocation allowance; allocations remain subject
    // to the exact item and aggregate checks.
    const CONTAINER_READ_ALLOWANCE: usize = 1024 * 1024;
    let read_limit = limits
        .max_total_bytes
        .saturating_add(CONTAINER_READ_ALLOWANCE);
    let reader = BoundedReader::new(reader, read_limit);
    let options = ParseOptions::new().read_properties(false);
    Probe::new(reader)
        .options(options)
        .guess_file_type()?
        .read()
}

fn merge_tags(source: &lofty::file::TaggedFile) -> Option<Tag> {
    let primary = source.primary_tag().or_else(|| source.first_tag())?;
    let mut merged = primary.clone();

    // Keep the primary tag's native companion, then add values from every
    // other tag which are not already present. Exact duplicates are omitted;
    // repeated non-identical values remain meaningful.
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

fn preflight_source_tags(
    source: &lofty::file::TaggedFile,
    limits: MetadataLimits,
) -> Result<(), String> {
    // Validate before cloning/merging. This intentionally counts every parsed
    // source item, which bounds work even when many tags contain duplicates.
    let mut budget = MetadataBudget::new(limits);
    for tag in source.tags() {
        check_tag_items(tag, &mut budget)?;
    }
    Ok(())
}

fn validate_metadata(metadata: &Metadata, limits: MetadataLimits) -> Result<(), String> {
    let mut tag_budget = MetadataBudget::new(limits);
    charge_tag(&metadata.tag, &mut tag_budget)?;
    if let Some(snapshot) = &metadata.vorbis_comments {
        let mut snapshot_budget = MetadataBudget::new(limits);
        validate_snapshot(snapshot, &mut snapshot_budget)?;
    }
    Ok(())
}

fn check_tag_items(tag: &Tag, budget: &mut MetadataBudget) -> Result<(), String> {
    for item in tag.items() {
        let value_len = match item.value() {
            ItemValue::Text(value) | ItemValue::Locator(value) => value.len(),
            ItemValue::Binary(value) => value.len(),
        };
        let bytes = item
            .description()
            .len()
            .checked_add(value_len)
            .ok_or_else(|| "metadata item size overflow".to_owned())?;
        budget.check_item(bytes, "generic metadata item")?;
    }
    for picture in tag.pictures() {
        let bytes = picture
            .data()
            .len()
            .checked_add(picture.description().map_or(0, str::len))
            .and_then(|size| {
                size.checked_add(picture.mime_type().map_or(0, |mime| mime.as_str().len()))
            })
            .ok_or_else(|| "metadata picture size overflow".to_owned())?;
        budget.check_item(bytes, "metadata picture")?;
    }
    Ok(())
}

fn charge_tag(tag: &Tag, budget: &mut MetadataBudget) -> Result<(), String> {
    for item in tag.items() {
        let value_len = match item.value() {
            ItemValue::Text(value) | ItemValue::Locator(value) => value.len(),
            ItemValue::Binary(value) => value.len(),
        };
        let bytes = item
            .description()
            .len()
            .checked_add(value_len)
            .ok_or_else(|| "metadata item size overflow".to_owned())?;
        budget.charge_item(bytes, "generic metadata item")?;
    }
    for picture in tag.pictures() {
        let bytes = picture
            .data()
            .len()
            .checked_add(picture.description().map_or(0, str::len))
            .and_then(|size| {
                size.checked_add(picture.mime_type().map_or(0, |mime| mime.as_str().len()))
            })
            .ok_or_else(|| "metadata picture size overflow".to_owned())?;
        budget.charge_item(bytes, "metadata picture")?;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct VorbisCommentsSnapshot {
    pub(super) vendor: String,
    pub(super) items: Vec<(String, String)>,
}

impl VorbisCommentsSnapshot {
    pub(super) fn new(vendor: String, items: Vec<(String, String)>) -> Self {
        Self { vendor, items }
    }
}

pub(super) struct MetadataBudget {
    limits: MetadataLimits,
    total_bytes: usize,
    items: usize,
}

impl MetadataBudget {
    pub(super) fn new(limits: MetadataLimits) -> Self {
        Self {
            limits,
            total_bytes: 0,
            items: 0,
        }
    }

    pub(super) fn check_bytes(&self, bytes: usize, context: &str) -> Result<(), String> {
        let total = self
            .total_bytes
            .checked_add(bytes)
            .ok_or_else(|| format!("{context} aggregate size overflow"))?;
        if total > self.limits.max_total_bytes {
            return Err(format!(
                "{context} exceeds metadata aggregate limit ({} > {} bytes)",
                total, self.limits.max_total_bytes
            ));
        }
        Ok(())
    }

    pub(super) fn charge_bytes(&mut self, bytes: usize, context: &str) -> Result<(), String> {
        self.check_bytes(bytes, context)?;
        let total = self
            .total_bytes
            .checked_add(bytes)
            .ok_or_else(|| format!("{context} aggregate size overflow"))?;
        self.total_bytes = total;
        Ok(())
    }

    pub(super) fn check_item(&mut self, bytes: usize, context: &str) -> Result<(), String> {
        if bytes > self.limits.max_item_bytes {
            return Err(format!(
                "{context} exceeds metadata item limit ({} > {} bytes)",
                bytes, self.limits.max_item_bytes
            ));
        }
        let items = self
            .items
            .checked_add(1)
            .ok_or_else(|| format!("{context} count overflow"))?;
        if items > self.limits.max_items {
            return Err(format!(
                "{context} exceeds metadata item count limit ({} > {})",
                items, self.limits.max_items
            ));
        }
        self.items = items;
        Ok(())
    }

    pub(super) fn charge_item(&mut self, bytes: usize, context: &str) -> Result<(), String> {
        self.check_item(bytes, context)?;
        self.charge_bytes(bytes, context)
    }

    pub(super) fn charge_picture_base64(
        &mut self,
        value: &str,
        context: &str,
    ) -> Result<(), String> {
        if value.len() > self.limits.max_item_bytes {
            return Err(format!(
                "{context} base64 value exceeds metadata item limit ({} > {} bytes)",
                value.len(),
                self.limits.max_item_bytes
            ));
        }

        let mut decoder = DecoderReader::new(value.as_bytes(), &BASE64_STANDARD);
        let mut scratch = [0_u8; 8 * 1024];
        let mut decoded = 0_usize;
        loop {
            let count = decoder
                .read(&mut scratch)
                .map_err(|error| format!("invalid {context} base64: {error}"))?;
            if count == 0 {
                break;
            }
            decoded = decoded
                .checked_add(count)
                .ok_or_else(|| format!("{context} decoded size overflow"))?;
            if decoded > self.limits.max_item_bytes {
                return Err(format!(
                    "{context} decoded picture exceeds metadata item limit ({} > {} bytes)",
                    decoded, self.limits.max_item_bytes
                ));
            }
        }
        self.charge_bytes(decoded, context)
    }
}

pub(super) fn parse_comment_body(
    body: &[u8],
    budget: &mut MetadataBudget,
) -> Result<VorbisCommentsSnapshot, String> {
    budget.charge_bytes(body.len(), "Vorbis comment body")?;
    let mut offset = 0_usize;
    let vendor_len = read_u32_le(body, &mut offset)?;
    let vendor_len = usize::try_from(vendor_len)
        .map_err(|_| "Vorbis vendor length does not fit in memory".to_owned())?;
    if vendor_len > budget.limits.max_item_bytes {
        return Err(format!(
            "Vorbis vendor exceeds metadata item limit ({} > {} bytes)",
            vendor_len, budget.limits.max_item_bytes
        ));
    }
    let vendor_end = offset
        .checked_add(vendor_len)
        .ok_or_else(|| "Vorbis vendor length overflow".to_owned())?;
    let vendor_bytes = body
        .get(offset..vendor_end)
        .ok_or_else(|| "truncated Vorbis vendor".to_owned())?;
    let vendor_text = std::str::from_utf8(vendor_bytes)
        .map_err(|error| format!("Vorbis vendor is not UTF-8: {error}"))?;
    let mut vendor = String::new();
    try_reserve_string(&mut vendor, vendor_len, "Vorbis vendor")?;
    vendor.push_str(vendor_text);
    offset = vendor_end;

    let count = read_u32_le(body, &mut offset)?;
    let count = usize::try_from(count)
        .map_err(|_| "Vorbis comment count does not fit in memory".to_owned())?;
    if count > budget.limits.max_items {
        return Err(format!(
            "Vorbis comment count exceeds metadata item count limit ({} > {})",
            count, budget.limits.max_items
        ));
    }
    let mut items = Vec::new();
    try_reserve_vec(&mut items, count, "Vorbis comment list")?;

    for _ in 0..count {
        let length = read_u32_le(body, &mut offset)?;
        let length = usize::try_from(length)
            .map_err(|_| "Vorbis comment length does not fit in memory".to_owned())?;
        budget.check_item(length, "Vorbis comment")?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| "Vorbis comment length overflow".to_owned())?;
        let field = body
            .get(offset..end)
            .ok_or_else(|| "truncated Vorbis comment".to_owned())?;
        offset = end;
        let separator = field
            .iter()
            .position(|byte| *byte == b'=')
            .ok_or_else(|| "Vorbis comment is missing '='".to_owned())?;
        let key = &field[..separator];
        validate_comment_key(key)?;
        let value = &field[separator + 1..];
        let key = std::str::from_utf8(key)
            .map_err(|error| format!("Vorbis comment key is not UTF-8: {error}"))?;
        let value = std::str::from_utf8(value)
            .map_err(|error| format!("Vorbis comment value is not UTF-8: {error}"))?;

        let mut owned_key = String::new();
        try_reserve_string(&mut owned_key, key.len(), "Vorbis comment key")?;
        owned_key.push_str(key);
        let mut owned_value = String::new();
        try_reserve_string(&mut owned_value, value.len(), "Vorbis comment value")?;
        owned_value.push_str(value);
        if is_vorbis_picture_key(key) {
            budget.charge_picture_base64(value, "Vorbis picture comment")?;
        }
        items.push((owned_key, owned_value));
    }

    if offset != body.len() {
        return Err(format!(
            "Vorbis comment body has {} trailing bytes",
            body.len() - offset
        ));
    }

    Ok(VorbisCommentsSnapshot::new(vendor, items))
}

pub(super) fn serialize_comment_body(
    snapshot: &VorbisCommentsSnapshot,
    budget: &mut MetadataBudget,
) -> Result<Vec<u8>, String> {
    let vendor = snapshot.vendor.as_bytes();
    if vendor.len() > budget.limits.max_item_bytes {
        return Err(format!(
            "Vorbis vendor exceeds metadata item limit ({} > {} bytes)",
            vendor.len(),
            budget.limits.max_item_bytes
        ));
    }
    let vendor_len = u32::try_from(vendor.len()).map_err(|_| "Vorbis vendor is too large")?;
    let item_count =
        u32::try_from(snapshot.items.len()).map_err(|_| "too many Vorbis comments".to_owned())?;
    if snapshot.items.len() > budget.limits.max_items {
        return Err(format!(
            "Vorbis comment count exceeds metadata item count limit ({} > {})",
            snapshot.items.len(),
            budget.limits.max_items
        ));
    }

    let mut output_len = 4_usize
        .checked_add(vendor.len())
        .and_then(|size| size.checked_add(4))
        .ok_or_else(|| "Vorbis comment body size overflow".to_owned())?;
    for (key, value) in &snapshot.items {
        validate_comment_key(key.as_bytes())?;
        let length = key
            .len()
            .checked_add(1)
            .and_then(|size| size.checked_add(value.len()))
            .ok_or_else(|| "Vorbis comment size overflow".to_owned())?;
        budget.check_item(length, "Vorbis comment")?;
        let _ = u32::try_from(length).map_err(|_| "Vorbis comment is too large")?;
        output_len = output_len
            .checked_add(4)
            .and_then(|size| size.checked_add(length))
            .ok_or_else(|| "Vorbis comment body size overflow".to_owned())?;
        if is_vorbis_picture_key(key) {
            budget.charge_picture_base64(value, "Vorbis picture comment")?;
        }
    }
    budget.charge_bytes(output_len, "Vorbis comment body")?;

    let mut output = Vec::new();
    try_reserve_vec(&mut output, output_len, "serialized Vorbis comment body")?;
    output.extend(vendor_len.to_le_bytes());
    output.extend(vendor);
    output.extend(item_count.to_le_bytes());
    for (key, value) in &snapshot.items {
        let length = key
            .len()
            .checked_add(1)
            .and_then(|size| size.checked_add(value.len()))
            .ok_or_else(|| "Vorbis comment size overflow".to_owned())?;
        let length = u32::try_from(length).map_err(|_| "Vorbis comment is too large")?;
        output.extend(length.to_le_bytes());
        output.extend(key.as_bytes());
        output.push(b'=');
        output.extend(value.as_bytes());
    }
    Ok(output)
}

fn validate_snapshot(
    snapshot: &VorbisCommentsSnapshot,
    budget: &mut MetadataBudget,
) -> Result<(), String> {
    // Serialization performs all strict syntax and size checks. It is bounded
    // by the supplied budget and uses fallible reservation.
    let _ = serialize_comment_body(snapshot, budget)?;
    Ok(())
}

fn validate_comment_key(key: &[u8]) -> Result<(), String> {
    // Vorbis field names are printable ASCII 0x20..=0x7D except '='.
    if key.is_empty()
        || key
            .iter()
            .any(|byte| !(0x20..=0x7d).contains(byte) || *byte == b'=')
    {
        return Err("invalid Vorbis comment key".into());
    }
    Ok(())
}

fn is_vorbis_picture_key(key: &str) -> bool {
    key.eq_ignore_ascii_case("METADATA_BLOCK_PICTURE") || key.eq_ignore_ascii_case("COVERART")
}

fn read_u32_le(data: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "Vorbis comment offset overflow".to_owned())?;
    let bytes = data
        .get(*offset..end)
        .ok_or_else(|| "truncated Vorbis comment length".to_owned())?;
    *offset = end;
    Ok(u32::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| "invalid Vorbis comment length".to_owned())?,
    ))
}

pub(super) fn merge_vorbis_comments(
    source: &VorbisCommentsSnapshot,
    destination: VorbisCommentsSnapshot,
    skip_picture_comments: bool,
    budget: &mut MetadataBudget,
) -> Result<VorbisCommentsSnapshot, String> {
    let maximum = source
        .items
        .len()
        .checked_add(destination.items.len())
        .ok_or_else(|| "Vorbis comment count overflow".to_owned())?;
    if maximum > budget.limits.max_items.saturating_mul(2) {
        return Err("Vorbis comment merge input exceeds bounded count".into());
    }

    let mut source_occurrences: HashMap<(String, String), usize> = HashMap::new();
    source_occurrences
        .try_reserve(source.items.len())
        .map_err(|error| format!("reserve Vorbis merge index: {error}"))?;
    let mut items = Vec::new();
    try_reserve_vec(
        &mut items,
        maximum.min(budget.limits.max_items),
        "merged Vorbis comments",
    )?;

    // The source snapshot is authoritative. Retain its exact vendor, order,
    // key casing, duplicates, and empty values, then append only genuinely new
    // values emitted by the generic destination writer.
    for (key, value) in &source.items {
        if skip_picture_comments && is_vorbis_picture_key(key) {
            continue;
        }
        let item_len = key
            .len()
            .checked_add(1)
            .and_then(|size| size.checked_add(value.len()))
            .ok_or_else(|| "Vorbis comment size overflow".to_owned())?;
        if item_len > budget.limits.max_item_bytes {
            return Err("source Vorbis comment exceeds metadata item limit".into());
        }
        if items.len() >= budget.limits.max_items {
            return Err("merged Vorbis comments exceed metadata item count limit".into());
        }
        let map_key = (
            try_ascii_lowercase(key, "Vorbis merge key")?,
            try_clone_string(value, "Vorbis merge value")?,
        );
        *source_occurrences.entry(map_key).or_default() += 1;
        items.push((
            try_clone_string(key, "merged Vorbis comment key")?,
            try_clone_string(value, "merged Vorbis comment value")?,
        ));
    }

    for (key, value) in destination.items {
        let map_key = (
            try_ascii_lowercase(&key, "Vorbis merge key")?,
            try_clone_string(&value, "Vorbis merge value")?,
        );
        if let Some(count) = source_occurrences.get_mut(&map_key) {
            if *count > 0 {
                *count -= 1;
                continue;
            }
        }
        let item_len = key
            .len()
            .checked_add(1)
            .and_then(|size| size.checked_add(value.len()))
            .ok_or_else(|| "Vorbis comment size overflow".to_owned())?;
        if item_len > budget.limits.max_item_bytes {
            return Err("destination Vorbis comment exceeds metadata item limit".into());
        }
        if items.len() >= budget.limits.max_items {
            return Err("merged Vorbis comments exceed metadata item count limit".into());
        }
        items.push((key, value));
    }

    Ok(VorbisCommentsSnapshot::new(
        try_clone_string(&source.vendor, "Vorbis vendor")?,
        items,
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RawContainer {
    Flac,
    Ogg,
    Other,
}

fn detect_file_container(file: &mut File) -> Result<RawContainer, String> {
    file.rewind()
        .map_err(|error| format!("rewind metadata signature: {error}"))?;
    let mut prefix = [0_u8; 10];
    let mut filled = 0_usize;
    while filled < prefix.len() {
        let count = file
            .read(&mut prefix[filled..])
            .map_err(|error| format!("read metadata signature: {error}"))?;
        if count == 0 {
            break;
        }
        filled += count;
    }
    if filled >= 3 && prefix[..3] == *b"ID3" {
        if filled < prefix.len() {
            file.rewind()
                .map_err(|error| format!("rewind metadata signature: {error}"))?;
            return Err("truncated ID3v2 prefix before audio metadata".into());
        }
        let file_len = file
            .metadata()
            .map_err(|error| format!("stat metadata input: {error}"))?
            .len();
        let payload_offset = crate::decode::id3v2_payload_offset(&prefix, file_len)?
            .ok_or_else(|| "internal ID3v2 metadata detection mismatch".to_owned())?;
        file.seek(SeekFrom::Start(payload_offset))
            .map_err(|error| format!("seek past ID3v2 metadata prefix: {error}"))?;
        let mut payload = [0_u8; 4];
        file.read_exact(&mut payload)
            .map_err(|error| format!("read metadata after ID3v2 prefix: {error}"))?;
        file.rewind()
            .map_err(|error| format!("rewind metadata signature: {error}"))?;
        if matches!(&payload, b"fLaC" | b"OggS") {
            return Err(
                "ID3v2-prefixed FLAC/Ogg metadata is unsupported by the bounded raw container path"
                    .into(),
            );
        }
        return Ok(RawContainer::Other);
    }

    file.rewind()
        .map_err(|error| format!("rewind metadata signature: {error}"))?;
    Ok(if filled >= 4 && prefix[..4] == *b"fLaC" {
        RawContainer::Flac
    } else if filled >= 4 && prefix[..4] == *b"OggS" {
        RawContainer::Ogg
    } else {
        RawContainer::Other
    })
}

pub(super) fn try_reserve_vec<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &str,
) -> Result<(), String> {
    values
        .try_reserve_exact(additional)
        .map_err(|error| format!("reserve {context}: {error}"))
}

fn try_reserve_string(value: &mut String, additional: usize, context: &str) -> Result<(), String> {
    value
        .try_reserve_exact(additional)
        .map_err(|error| format!("reserve {context}: {error}"))
}

fn try_clone_string(value: &str, context: &str) -> Result<String, String> {
    let mut cloned = String::new();
    try_reserve_string(&mut cloned, value.len(), context)?;
    cloned.push_str(value);
    Ok(cloned)
}

fn try_ascii_lowercase(value: &str, context: &str) -> Result<String, String> {
    let mut lowercase = String::new();
    try_reserve_string(&mut lowercase, value.len(), context)?;
    for byte in value.bytes() {
        lowercase.push(char::from(byte.to_ascii_lowercase()));
    }
    Ok(lowercase)
}

struct BoundedReader<R> {
    inner: R,
    remaining: usize,
}

impl<R> BoundedReader<R> {
    fn new(inner: R, maximum_read_bytes: usize) -> Self {
        Self {
            inner,
            remaining: maximum_read_bytes,
        }
    }
}

impl<R: Read> Read for BoundedReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "metadata parser read budget exceeded",
            ));
        }
        let allowed = buffer.len().min(self.remaining);
        let read = self.inner.read(&mut buffer[..allowed])?;
        self.remaining -= read;
        Ok(read)
    }
}

impl<R: Seek> Seek for BoundedReader<R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    use lofty::picture::{MimeType, PictureType};
    use lofty::tag::{Accessor, ItemKey, Tag, TagType};

    fn body(snapshot: &VorbisCommentsSnapshot) -> Vec<u8> {
        serialize_comment_body(
            snapshot,
            &mut MetadataBudget::new(MetadataLimits::default()),
        )
        .unwrap()
    }

    #[test]
    fn default_limits_are_finite_and_internally_consistent() {
        let limits = MetadataLimits::default();
        assert!(limits.max_total_bytes < usize::MAX);
        assert!(limits.max_item_bytes <= limits.max_total_bytes);
        assert!(limits.max_flac_block_bytes <= 0x00ff_ffff);
        assert!(limits.max_items > 0);
        assert!(limits.max_flac_blocks > 0);
        assert!(limits.max_ogg_packet_bytes > 0);
        assert!(limits.max_ogg_pages > 0);
        assert!(limits.max_ogg_streams > 0);
    }

    #[test]
    fn retained_metadata_estimate_counts_generic_and_raw_representations() {
        let empty = Metadata {
            tag: Tag::new(TagType::Id3v2),
            vorbis_comments: None,
        };
        assert_eq!(
            empty.estimated_memory_bytes(),
            RESIDENT_BASE_BYTES * RESIDENT_REPRESENTATION_MULTIPLIER
        );

        let mut tag = Tag::new(TagType::Id3v2);
        tag.insert_text(ItemKey::EncoderSoftware, "source encoder".into());
        let generic = Metadata {
            tag,
            vorbis_comments: None,
        };
        assert!(generic.estimated_memory_bytes() > empty.estimated_memory_bytes());

        let raw = Metadata {
            tag: generic.tag.clone(),
            vorbis_comments: Some(VorbisCommentsSnapshot::new(
                "vendor".into(),
                vec![("CUSTOM".into(), "value".into())],
            )),
        };
        assert!(raw.estimated_memory_bytes() > generic.estimated_memory_bytes());
    }

    #[test]
    fn malformed_picture_bytes_survive_conversion_with_default_information() {
        let first = Picture::unchecked(vec![1, 2, 3])
            .pic_type(PictureType::CoverFront)
            .mime_type(MimeType::Png)
            .description("malformed")
            .build();
        let second = Picture::unchecked(vec![9, 8])
            .pic_type(PictureType::CoverBack)
            .build();
        let mut tag = Tag::new(TagType::Id3v2);
        tag.push_picture(first.clone());
        tag.push_picture(second.clone());

        let pictures = take_tag_pictures(&mut tag).unwrap();
        assert_eq!(tag.picture_count(), 0);
        assert_eq!(pictures.len(), 2);
        assert_eq!(pictures[0].0, first);
        assert_eq!(pictures[1].0, second);
        assert_eq!(pictures[0].1, PictureInformation::default());
        assert_eq!(pictures[1].1, PictureInformation::default());
    }

    #[test]
    fn streaming_picture_serialization_matches_lofty_layout() {
        let picture = Picture::unchecked(vec![1, 2, 3, 4, 5])
            .pic_type(PictureType::CoverFront)
            .mime_type(MimeType::Png)
            .description("front")
            .build();
        let information = PictureInformation {
            width: 3,
            height: 2,
            color_depth: 24,
            num_colors: 0,
        };
        let mut raw = Vec::new();
        raw.try_reserve_exact(picture_block_len(&picture, information).unwrap())
            .unwrap();
        write_picture_block(&picture, information, &mut raw).unwrap();
        assert_eq!(raw, picture.as_flac_bytes(information, false));

        let encoded = encode_ogg_picture(&picture, information, MetadataLimits::default()).unwrap();
        assert_eq!(encoded.as_bytes(), picture.as_flac_bytes(information, true));
    }

    #[test]
    fn commentless_flac_keeps_converted_encoder_vendor() {
        let mut output = tempfile::tempfile().unwrap();
        output.write_all(b"fLaC").unwrap();
        output.write_all(&[0x80, 0, 0, 34]).unwrap();
        output.write_all(&[0_u8; 34]).unwrap();
        output.write_all(b"audio").unwrap();

        let mut tag = Tag::new(TagType::VorbisComments);
        tag.insert_text(ItemKey::EncoderSoftware, "source encoder".into());
        write_extended_to_file_with_limits(
            Metadata {
                tag,
                vorbis_comments: None,
            },
            &mut output,
            MetadataLimits::default(),
        )
        .unwrap();

        let limits = MetadataLimits::default();
        let comments = flac::read_file(&mut output, &limits, &mut MetadataBudget::new(limits))
            .unwrap()
            .unwrap();
        assert_eq!(comments.vendor, "source encoder");
    }

    #[test]
    fn id3_prefixed_flac_is_rejected_before_bounded_raw_limits_can_be_bypassed() {
        let mut bytes = b"ID3\x04\0\0\0\0\0\0".to_vec();
        bytes.extend_from_slice(b"fLaC");
        bytes.extend_from_slice(&[0, 0, 0, 34]);
        bytes.extend_from_slice(&[0_u8; 34]);
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&[0x80 | 1, 0, 0, 0]);

        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("prefixed.flac");
        std::fs::write(&input, &bytes).unwrap();
        let mut limits = MetadataLimits::default();
        limits.max_flac_blocks = 1;

        let error = match read_extended_with_limits(&input, limits) {
            Ok(_) => panic!("ID3v2-prefixed FLAC unexpectedly passed bounded read"),
            Err(error) => error,
        };
        assert!(error.contains("ID3v2-prefixed FLAC/Ogg"), "{error}");

        let mut output = tempfile::tempfile().unwrap();
        output.write_all(&bytes).unwrap();
        let original = std::fs::read(&input).unwrap();
        let error = write_extended_to_file_with_limits(
            Metadata {
                tag: Tag::new(TagType::VorbisComments),
                vorbis_comments: None,
            },
            &mut output,
            limits,
        )
        .unwrap_err();
        assert!(error.contains("ID3v2-prefixed FLAC/Ogg"), "{error}");
        output.rewind().unwrap();
        let mut after = Vec::new();
        output.read_to_end(&mut after).unwrap();
        assert_eq!(after, original);
    }

    #[test]
    fn comment_body_round_trips_order_case_duplicates_and_empty_values() {
        let source = VorbisCommentsSnapshot::new(
            "vendor".into(),
            vec![
                ("TITLE".into(), "A title".into()),
                ("x-Custom".into(), "".into()),
                ("TITLE".into(), "A title".into()),
                ("CHAPTER001".into(), "00:00:01.000".into()),
            ],
        );
        let encoded = body(&source);
        let parsed = parse_comment_body(
            &encoded,
            &mut MetadataBudget::new(MetadataLimits::default()),
        )
        .unwrap();
        assert_eq!(parsed, source);
    }

    #[test]
    fn comment_parser_rejects_trailing_and_invalid_keys() {
        let source =
            VorbisCommentsSnapshot::new("vendor".into(), vec![("TITLE".into(), "value".into())]);
        let mut encoded = body(&source);
        encoded.push(1);
        assert!(parse_comment_body(
            &encoded,
            &mut MetadataBudget::new(MetadataLimits::default())
        )
        .unwrap_err()
        .contains("trailing"));

        let invalid =
            VorbisCommentsSnapshot::new("vendor".into(), vec![("BAD=KEY".into(), "value".into())]);
        assert!(serialize_comment_body(
            &invalid,
            &mut MetadataBudget::new(MetadataLimits::default())
        )
        .is_err());
    }

    #[test]
    fn comment_parser_enforces_item_count_and_total_limits() {
        let source = VorbisCommentsSnapshot::new(
            "v".into(),
            vec![("A".into(), "1".into()), ("B".into(), "2".into())],
        );
        let encoded = body(&source);
        let count_limits = MetadataLimits {
            max_items: 1,
            ..MetadataLimits::default()
        };
        assert!(
            parse_comment_body(&encoded, &mut MetadataBudget::new(count_limits))
                .unwrap_err()
                .contains("count")
        );

        let total_limits = MetadataLimits {
            max_total_bytes: encoded.len() - 1,
            ..MetadataLimits::default()
        };
        assert!(
            parse_comment_body(&encoded, &mut MetadataBudget::new(total_limits))
                .unwrap_err()
                .contains("aggregate")
        );
    }

    #[test]
    fn picture_base64_decoded_bytes_are_charged() {
        for key in ["METADATA_BLOCK_PICTURE", "COVERART"] {
            let source =
                VorbisCommentsSnapshot::new("v".into(), vec![(key.into(), "QUJDRA==".into())]);
            let encoded = body(&source);
            let limits = MetadataLimits {
                max_total_bytes: encoded.len() + 3,
                ..MetadataLimits::default()
            };
            assert!(parse_comment_body(&encoded, &mut MetadataBudget::new(limits)).is_err());

            let limits = MetadataLimits {
                max_total_bytes: encoded.len() + 4,
                ..MetadataLimits::default()
            };
            assert!(parse_comment_body(&encoded, &mut MetadataBudget::new(limits)).is_ok());
        }
    }

    #[test]
    fn source_comment_order_vendor_case_and_duplicates_are_authoritative() {
        let source = VorbisCommentsSnapshot::new(
            "".into(),
            vec![
                ("x-Key".into(), "".into()),
                ("TITLE".into(), "same".into()),
                ("TITLE".into(), "same".into()),
            ],
        );
        let destination = VorbisCommentsSnapshot::new(
            "destination".into(),
            vec![
                ("title".into(), "same".into()),
                ("title".into(), "same".into()),
                ("ARTIST".into(), "new".into()),
            ],
        );
        let merged = merge_vorbis_comments(
            &source,
            destination,
            false,
            &mut MetadataBudget::new(MetadataLimits::default()),
        )
        .unwrap();
        assert_eq!(merged.vendor, "");
        assert_eq!(
            merged.items,
            vec![
                ("x-Key".into(), "".into()),
                ("TITLE".into(), "same".into()),
                ("TITLE".into(), "same".into()),
                ("ARTIST".into(), "new".into()),
            ]
        );
    }

    #[test]
    fn merges_all_tags_without_exact_duplicates() {
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
    fn write_extended_commits_metadata_without_leaving_a_stage() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("output.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&output, spec).unwrap();
        writer.write_sample(0_i16).unwrap();
        writer.finalize().unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&output, fs::Permissions::from_mode(0o600)).unwrap();
        }

        let mut tag = Tag::new(TagType::RiffInfo);
        tag.set_title("committed title".into());
        write(tag, &output).unwrap();

        let saved = read(&output).unwrap().unwrap();
        assert_eq!(saved.title().as_deref(), Some("committed title"));
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&output).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn write_extended_rolls_back_when_staged_metadata_fails() {
        let mut output = tempfile::NamedTempFile::new().unwrap();
        let original = b"not an audio container";
        output.write_all(original).unwrap();
        output.flush().unwrap();

        let mut tag = Tag::new(TagType::Id3v2);
        tag.set_title("must not be published".into());
        assert!(write_extended(
            Metadata {
                tag,
                vorbis_comments: Some(VorbisCommentsSnapshot::new(
                    "vendor".into(),
                    vec![("X-CUSTOM".into(), "value".into())],
                )),
            },
            output.path(),
        )
        .is_err());
        assert_eq!(fs::read(output.path()).unwrap(), original.to_vec());
    }

    #[test]
    fn limited_write_rejects_metadata_before_touching_destination() {
        let mut output = tempfile::NamedTempFile::new().unwrap();
        let original = b"destination remains byte exact";
        output.write_all(original).unwrap();
        output.flush().unwrap();

        let mut tag = Tag::new(TagType::Id3v2);
        tag.set_title("larger than the selected aggregate budget".into());
        let limits = MetadataLimits {
            max_total_bytes: 4,
            ..MetadataLimits::default()
        };
        let error = write_extended_with_limits(
            Metadata {
                tag,
                vorbis_comments: None,
            },
            output.path(),
            limits,
        )
        .unwrap_err();
        assert!(error.contains("aggregate limit"));
        assert_eq!(fs::read(output.path()).unwrap(), original);
    }

    #[cfg(unix)]
    #[test]
    fn path_write_rejects_fifo_without_replacing_or_opening_it_for_io() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt as _;
        use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};

        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("output.fifo");
        let output_name = CString::new(output.as_os_str().as_bytes()).unwrap();
        // SAFETY: `output_name` is NUL terminated and names a path in the
        // private test directory.
        assert_eq!(unsafe { libc::mkfifo(output_name.as_ptr(), 0o600) }, 0);
        let before = fs::symlink_metadata(&output).unwrap();

        let mut tag = Tag::new(TagType::Id3v2);
        tag.set_title("must not be published".into());
        let error = write_extended(
            Metadata {
                tag,
                vorbis_comments: None,
            },
            &output,
        )
        .unwrap_err();

        assert!(error.contains("not a regular file"), "{error}");
        let after = fs::symlink_metadata(&output).unwrap();
        assert!(after.file_type().is_fifo());
        assert_eq!((after.dev(), after.ino()), (before.dev(), before.ino()));
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
