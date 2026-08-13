//! denoize 自作デコード層 — lossless / lossy audio containers を高品質 PCM (`f64`) へ。
//!
//! # 設計方針（劣化最小）
//! - デコード出力は `f32` → `f64` へ拡張のみ（再量子化なし）
//! - サンプルレート変換なし（ソースレートを維持）
//! - 内部パイプラインは 32-bit float 相当精度で denoise へ渡す
//!
//! # バックエンド
//! | 形式 | 実装 |
//! |------|------|
//! | WAV / BWF | `hound`（既存） |
//! | RF64 | bounded native PCM reader |
//! | MP3  | `symphonia`（Xing / LAME gapless timing）+ bounded `nanomp3` compatibility fallback |
//! | M4A  | `mp4` demux + `oxideav-aac` AAC-LC / `symphonia` ALAC decode; v0/v1 unity-rate edit-list presentation timing (multiple media edits and leading/interior empty edits; malformed/unsupported edits fail closed) |
//! | AIFF / CAF / Ogg Vorbis / ALAC | `symphonia` |

mod aac;
mod budget;
mod m4a;
mod mp3;
mod opus;
mod pcm;

pub(crate) use budget::DecodeBudget;
pub use budget::DecodeLimits;
pub use pcm::DecodedPcm;

use std::path::Path;

const FORMAT_SNIFF_BYTES: usize = 4096;
const ID3V2_HEADER_BYTES: usize = 10;
const MAX_RF64_SIZE_TABLE_ENTRIES: usize = 65_536;
// A conservative allowance for the key/value, hash bucket, and allocator
// bookkeeping retained per RF64 `ds64` table entry.
const RF64_SIZE_TABLE_ENTRY_BYTES: u64 = 64;

fn rf64_table_working_bytes(
    retained_entries: usize,
    incoming_entries: usize,
    transient_body_bytes: usize,
    context: &str,
) -> Result<u64, String> {
    let total_entries = retained_entries
        .checked_add(incoming_entries)
        .ok_or_else(|| format!("{context} size-table entry count overflows"))?;
    if total_entries > MAX_RF64_SIZE_TABLE_ENTRIES {
        return Err(format!(
            "{context} size table exceeds {MAX_RF64_SIZE_TABLE_ENTRIES} entries"
        ));
    }
    u64::try_from(total_entries)
        .ok()
        .and_then(|entries| entries.checked_mul(RF64_SIZE_TABLE_ENTRY_BYTES))
        .and_then(|bytes| bytes.checked_add(u64::try_from(transient_body_bytes).ok()?))
        .ok_or_else(|| format!("{context} working-set byte count overflows"))
}

fn read_rf64_ds64_table(
    reader: &mut impl std::io::Read,
    body_len: usize,
    size_table: &mut std::collections::HashMap<[u8; 4], u64>,
    entries_seen: &mut usize,
    budget: DecodeBudget,
    context: &str,
) -> Result<u64, String> {
    if body_len < 28 || body_len > 1 << 20 {
        return Err(format!("{context} ds64 chunk has an invalid size"));
    }
    let mut fixed = [0u8; 28];
    reader
        .read_exact(&mut fixed)
        .map_err(|error| format!("read {context} ds64 header: {error}"))?;
    let data_size = u64::from_le_bytes(fixed[8..16].try_into().expect("fixed ds64 data size"));
    let table_len = usize::try_from(u32::from_le_bytes(
        fixed[24..28].try_into().expect("fixed table length"),
    ))
    .map_err(|_| format!("{context} ds64 table count is too large"))?;
    let required = table_len
        .checked_mul(12)
        .and_then(|bytes| bytes.checked_add(fixed.len()))
        .ok_or_else(|| format!("{context} ds64 table size overflows"))?;
    if required > body_len {
        return Err(format!("{context} ds64 chunk ends before its table"));
    }
    let working_bytes =
        rf64_table_working_bytes(*entries_seen, table_len, fixed.len() + 12, context)?;
    budget.check_peak(0, working_bytes, context)?;
    size_table
        .try_reserve(table_len)
        .map_err(|error| format!("reserve {context} size table: {error}"))?;
    *entries_seen = entries_seen
        .checked_add(table_len)
        .ok_or_else(|| format!("{context} size-table entry count overflows"))?;
    for _ in 0..table_len {
        let mut entry = [0u8; 12];
        reader
            .read_exact(&mut entry)
            .map_err(|error| format!("read {context} ds64 table: {error}"))?;
        let mut chunk_id = [0u8; 4];
        chunk_id.copy_from_slice(&entry[..4]);
        size_table.insert(
            chunk_id,
            u64::from_le_bytes(entry[4..12].try_into().expect("fixed table entry")),
        );
    }
    Ok(data_size)
}

fn declared_id3v2_payload_offset(header: &[u8]) -> Result<Option<u64>, String> {
    if !header.starts_with(b"ID3") {
        return Ok(None);
    }
    let header = header
        .get(..ID3V2_HEADER_BYTES)
        .ok_or_else(|| "truncated ID3v2 header".to_string())?;
    if !(2..=4).contains(&header[3]) || header[4] == 0xff {
        return Err(format!(
            "unsupported ID3v2 version {}.{}",
            header[3], header[4]
        ));
    }
    if header[6..10].iter().any(|byte| byte & 0x80 != 0) {
        return Err("invalid ID3v2 synchsafe size".into());
    }

    let tag_size = header[6..10]
        .iter()
        .fold(0u64, |size, byte| (size << 7) | u64::from(*byte));
    // Only ID3v2.4 assigns bit 4 to a ten-byte footer. In v2.3 and
    // earlier that bit is reserved and must never move the audio payload.
    let footer_size = u64::from(header[3] == 4 && header[5] & 0x10 != 0) * 10;
    let payload_offset = (ID3V2_HEADER_BYTES as u64)
        .checked_add(tag_size)
        .and_then(|offset| offset.checked_add(footer_size))
        .ok_or_else(|| "ID3v2 size overflows".to_string())?;
    Ok(Some(payload_offset))
}

pub(super) fn id3v2_payload_offset(header: &[u8], file_len: u64) -> Result<Option<u64>, String> {
    let Some(payload_offset) = declared_id3v2_payload_offset(header)? else {
        return Ok(None);
    };
    if payload_offset > file_len {
        return Err(format!(
            "ID3v2 tag extends beyond the file ({payload_offset} bytes declared, {file_len} bytes available)"
        ));
    }
    Ok(Some(payload_offset))
}

/// Remove one leading ID3v2 tag from an in-memory stream without interpreting
/// its frames. This is shared by raw-stream decoders so routing and decoding
/// agree on footer and bounds semantics.
pub(super) fn strip_id3v2_prefix(bytes: &[u8]) -> Result<&[u8], String> {
    let file_len = u64::try_from(bytes.len()).map_err(|_| "ID3v2 input length overflows u64")?;
    let Some(payload_offset) = id3v2_payload_offset(bytes, file_len)? else {
        return Ok(bytes);
    };
    let payload_offset = usize::try_from(payload_offset)
        .map_err(|_| "ID3v2 payload offset does not fit in memory")?;
    Ok(&bytes[payload_offset..])
}

enum OggBosDetection {
    Detected(AudioFormat),
    Unknown,
    Incomplete,
}

fn detect_ogg_bos_format(header: &[u8]) -> OggBosDetection {
    let Some(fixed_header) = header.get(..27) else {
        return OggBosDetection::Incomplete;
    };
    // Identification packets begin on a non-continuation BOS page. Inspect
    // only that first packet, never comments or encoded payload bytes.
    if &fixed_header[..4] != b"OggS"
        || fixed_header[4] != 0
        || fixed_header[5] & 0x02 == 0
        || fixed_header[5] & 0x01 != 0
    {
        return OggBosDetection::Unknown;
    }

    let segment_count = usize::from(fixed_header[26]);
    let Some(lacing_end) = 27usize.checked_add(segment_count) else {
        return OggBosDetection::Unknown;
    };
    let Some(lacing) = header.get(27..lacing_end) else {
        return OggBosDetection::Incomplete;
    };
    if lacing.is_empty() {
        return OggBosDetection::Unknown;
    }

    let mut first_packet_len = 0usize;
    let mut packet_complete = false;
    for &segment_len in lacing {
        let Some(next_len) = first_packet_len.checked_add(usize::from(segment_len)) else {
            return OggBosDetection::Unknown;
        };
        first_packet_len = next_len;
        if segment_len < 255 {
            packet_complete = true;
            break;
        }
    }
    let Some(packet_end) = lacing_end.checked_add(first_packet_len) else {
        return OggBosDetection::Unknown;
    };
    // Codec identification is entirely in the packet prefix. Do not require
    // a compatible future OpusHead extension to fit in the bounded sniff
    // buffer, but never inspect bytes beyond the first packet's lacing span.
    let available_end = packet_end.min(header.len());
    let first_packet = &header[lacing_end..available_end];
    if first_packet.starts_with(b"OpusHead") {
        OggBosDetection::Detected(AudioFormat::OggOpus)
    } else if first_packet.starts_with(b"\x01vorbis") {
        OggBosDetection::Detected(AudioFormat::OggVorbis)
    } else if packet_complete && first_packet.len() == first_packet_len {
        OggBosDetection::Unknown
    } else if first_packet.len() >= b"OpusHead".len() {
        // Eight leading bytes suffice to rule out both supported signatures.
        OggBosDetection::Unknown
    } else {
        OggBosDetection::Incomplete
    }
}

/// Detected container / codec family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Rf64,
    Aiff,
    Caf,
    Flac,
    OggOpus,
    OggVorbis,
    Mp3,
    M4a,
    AacAdts,
    Unknown,
}

/// Codec carried by an audio container.
///
/// This is deliberately separate from [`AudioFormat`]: containers such as Ogg
/// and ISO BMFF (`.m4a` / `.mp4`) can carry more than one codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioCodec {
    Pcm,
    Flac,
    Opus,
    Vorbis,
    Mp3,
    Aac,
    Alac,
    Unknown,
}

/// Result of inspecting an audio file without decoding its sample payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioProbe {
    /// Detected container / format family.
    pub format: AudioFormat,
    /// Codec reported by the container's audio track metadata.
    pub codec: AudioCodec,
    /// Number of audio tracks reported by the container.
    pub audio_tracks: usize,
    /// Whether the container also reports video, subtitle, or other non-audio tracks.
    pub has_non_audio_tracks: bool,
    /// Whether a RIFF/RF64 WAVE carries Broadcast Wave (`bext`) metadata.
    pub is_broadcast_wave: bool,
}

impl AudioFormat {
    /// Sniff from file content and extension.
    pub fn detect(path: &Path, header: &[u8]) -> Self {
        let mut header = header;
        while header.starts_with(b"ID3") {
            match strip_id3v2_prefix(header) {
                Ok(payload) if payload.len() < header.len() => header = payload,
                // A bounded caller may not have supplied the whole tag. Do not
                // guess MP3 from ID3 alone; only the caller's extension remains
                // available as a fallback in that case.
                _ => return Self::from_extension(path),
            }
        }

        if header.starts_with(b"OggS") {
            // A recognized Ogg container with an unknown first BOS packet is
            // not implicitly Opus, even when its filename ends in `.ogg`.
            return match detect_ogg_bos_format(header) {
                OggBosDetection::Detected(format) => format,
                OggBosDetection::Unknown => AudioFormat::Unknown,
                // `detect` is also a public bounded-header helper. Preserve its
                // extension fallback when the caller supplied only part of a
                // structurally valid BOS page.
                OggBosDetection::Incomplete => Self::from_extension(path),
            };
        }

        if header.len() >= 12 {
            if &header[0..4] == b"RIFF" && &header[8..12] == b"WAVE" {
                return AudioFormat::Wav;
            }
            if &header[0..4] == b"RF64" && &header[8..12] == b"WAVE" {
                return AudioFormat::Rf64;
            }
            if &header[0..4] == b"FORM" && (&header[8..12] == b"AIFF" || &header[8..12] == b"AIFC")
            {
                return AudioFormat::Aiff;
            }
            if &header[0..4] == b"caff" {
                return AudioFormat::Caf;
            }
            if &header[0..4] == b"fLaC" {
                return AudioFormat::Flac;
            }
            if &header[4..8] == b"ftyp" {
                return AudioFormat::M4a;
            }
            // ADTS has a 12-bit sync word and its two layer bits are always 0.
            // Check it before the broader 11-bit MPEG audio sync test.
            if header[0] == 0xFF && (header[1] & 0xF6) == 0xF0 {
                return AudioFormat::AacAdts;
            }
            if header[0] == 0xFF && (header[1] & 0xE0) == 0xE0 {
                return AudioFormat::Mp3;
            }
        }

        Self::from_extension(path)
    }

    fn from_extension(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("wav" | "bwf") => AudioFormat::Wav,
            Some("rf64") => AudioFormat::Rf64,
            Some("aif" | "aiff" | "aifc") => AudioFormat::Aiff,
            Some("caf") => AudioFormat::Caf,
            Some("flac") => AudioFormat::Flac,
            Some("opus" | "ogg") => AudioFormat::OggOpus,
            Some("oga" | "vorbis") => AudioFormat::OggVorbis,
            Some("mp3") => AudioFormat::Mp3,
            Some("m4a" | "m4b" | "m4p" | "mp4") => AudioFormat::M4a,
            Some("aac") => AudioFormat::AacAdts,
            _ => AudioFormat::Unknown,
        }
    }

    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            AudioFormat::Wav => &["wav"],
            AudioFormat::Rf64 => &["rf64"],
            AudioFormat::Aiff => &["aif", "aiff", "aifc"],
            AudioFormat::Caf => &["caf"],
            AudioFormat::Flac => &["flac"],
            AudioFormat::OggOpus => &["opus", "ogg"],
            AudioFormat::OggVorbis => &["oga", "vorbis"],
            AudioFormat::Mp3 => &["mp3"],
            AudioFormat::M4a => &["m4a", "m4b", "mp4", "aac"],
            AudioFormat::AacAdts => &["aac"],
            AudioFormat::Unknown => &[],
        }
    }
}

fn detect_file_format_from_file(
    path: &Path,
    use_extension_fallback: bool,
    file: &mut std::fs::File,
) -> Result<AudioFormat, String> {
    use std::io::{Read, Seek, SeekFrom};

    let file_len = file
        .metadata()
        .map_err(|error| format!("stat {} for format detection: {error}", path.display()))?
        .len();
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {} for format detection: {error}", path.display()))?;

    let mut header = Vec::with_capacity(FORMAT_SNIFF_BYTES);
    file.by_ref()
        .take(FORMAT_SNIFF_BYTES as u64)
        .read_to_end(&mut header)
        .map_err(|error| format!("read {} for format detection: {error}", path.display()))?;
    if let Some(payload_offset) = id3v2_payload_offset(&header, file_len)? {
        file.seek(SeekFrom::Start(payload_offset))
            .map_err(|error| {
                format!(
                    "seek past ID3v2 tag in {} for format detection: {error}",
                    path.display()
                )
            })?;
        header.clear();
        file.take(FORMAT_SNIFF_BYTES as u64)
            .read_to_end(&mut header)
            .map_err(|error| {
                format!(
                    "read payload of {} for format detection: {error}",
                    path.display()
                )
            })?;
    }

    Ok(AudioFormat::detect(
        if use_extension_fallback {
            path
        } else {
            Path::new("")
        },
        &header,
    ))
}

/// Inspect an audio file's container and codec without decoding its samples.
///
/// Container detection is content-based. Ogg and ISO BMFF containers are then
/// demuxed with Symphonia so that `.ogg` is not implicitly treated as Opus and
/// `.m4a` / `.mp4` is not implicitly treated as AAC. The path must resolve to
/// a regular file; FIFOs, directories, and device files are rejected.
pub fn probe_file(path: &Path) -> Result<AudioProbe, String> {
    probe_file_with_limits(path, DecodeLimits::default())
}

/// Inspect a regular-file audio input with explicit FLAC/Ogg metadata limits.
///
/// The limits are applied before a decoder or demuxer can materialize
/// container metadata. Other formats retain their existing probe behavior.
pub fn probe_file_with_metadata_limits(
    path: &Path,
    metadata_limits: crate::metadata::MetadataLimits,
) -> Result<AudioProbe, String> {
    probe_file_with_limits(
        path,
        DecodeLimits {
            metadata: metadata_limits,
            ..DecodeLimits::default()
        },
    )
}

/// Inspect a regular-file audio input with explicit decode resource limits.
pub fn probe_file_with_limits(path: &Path, limits: DecodeLimits) -> Result<AudioProbe, String> {
    let mut session = crate::input::AudioInputSession::open(path)?;
    probe_file_from_session_with_limits(&mut session, limits)
}

/// Inspect the regular file held by an existing input session.
///
/// The clone used for probing refers to the same opened filesystem object as
/// later session operations, even if the original pathname is replaced.
pub fn probe_file_from_session_with_limits(
    session: &mut crate::input::AudioInputSession,
    limits: DecodeLimits,
) -> Result<AudioProbe, String> {
    let path = session.path().to_path_buf();
    let source = session.try_clone_rewound("audio probe")?;
    probe_file_from_file_with_limits(&path, source, limits)
}

fn probe_file_from_file_with_limits(
    path: &Path,
    mut source: std::fs::File,
    limits: DecodeLimits,
) -> Result<AudioProbe, String> {
    let budget = DecodeBudget::new(limits);
    budget.check_peak(0, FORMAT_SNIFF_BYTES as u64, "audio probe")?;
    // Suppress extension fallback here. A probe must describe the file's
    // contents, not what its name claims the contents are.
    let format = detect_file_format_from_file(path, false, &mut source)?;
    use std::io::{Seek, SeekFrom};
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {} after format detection: {error}", path.display()))?;

    match format {
        AudioFormat::Wav | AudioFormat::Rf64 => {
            probe_wave_file_from_file(path, format, source, budget)
        }
        AudioFormat::Aiff | AudioFormat::Caf => Ok(single_audio_track(format, AudioCodec::Pcm)),
        AudioFormat::Flac => {
            crate::metadata::preflight_flac_decode(&mut source, limits.metadata)?;
            Ok(single_audio_track(format, AudioCodec::Flac))
        }
        AudioFormat::Mp3 => Ok(single_audio_track(format, AudioCodec::Mp3)),
        AudioFormat::AacAdts => Ok(single_audio_track(format, AudioCodec::Aac)),
        AudioFormat::OggOpus | AudioFormat::OggVorbis => {
            probe_ogg_tracks(path, format, limits.metadata, source)
        }
        AudioFormat::M4a => probe_mp4_tracks_from_file(path, source, limits),
        AudioFormat::Unknown => Err(format!(
            "could not identify the audio container from the file header ({}); verify that the file is a supported, non-truncated audio file",
            path.display()
        )),
    }
}

fn single_audio_track(format: AudioFormat, codec: AudioCodec) -> AudioProbe {
    AudioProbe {
        format,
        codec,
        audio_tracks: 1,
        has_non_audio_tracks: false,
        is_broadcast_wave: false,
    }
}

fn probe_wave_file_from_file(
    path: &Path,
    format: AudioFormat,
    mut file: std::fs::File,
    budget: DecodeBudget,
) -> Result<AudioProbe, String> {
    use std::io::{Read, Seek, SeekFrom};

    let file_len = file
        .metadata()
        .map_err(|error| format!("stat {} for WAVE codec probe: {error}", path.display()))?
        .len();
    if file_len < 12 {
        return Err(format!("truncated WAVE header ({})", path.display()));
    }

    let mut offset = 12u64;
    let mut codec = AudioCodec::Unknown;
    let mut found_fmt = false;
    let mut found_data = false;
    let mut is_broadcast_wave = false;
    let mut rf64_data_size = None;
    let mut rf64_chunk_sizes = std::collections::HashMap::<[u8; 4], u64>::new();
    let mut rf64_table_entries_seen = 0usize;
    budget.check_peak(0, 0, "WAVE/RF64 probe")?;
    for _ in 0..4096 {
        let Some(header_end) = offset.checked_add(8) else {
            break;
        };
        if header_end > file_len {
            break;
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("seek WAVE chunk {}: {error}", path.display()))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)
            .map_err(|error| format!("read WAVE chunk {}: {error}", path.display()))?;
        let chunk_size_32 = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let chunk_size = u64::from(chunk_size_32);
        let data_offset = header_end;
        let effective_chunk_size = if format == AudioFormat::Rf64 && chunk_size_32 == u32::MAX {
            rf64_chunk_sizes
                .get(&header[0..4])
                .copied()
                .or_else(|| {
                    (&header[0..4] == b"data")
                        .then_some(rf64_data_size)
                        .flatten()
                })
                .ok_or_else(|| {
                    format!(
                        "RF64 chunk {:?} has no ds64 size ({})",
                        String::from_utf8_lossy(&header[0..4]),
                        path.display()
                    )
                })?
        } else {
            chunk_size
        };

        if &header[0..4] == b"ds64" && format == AudioFormat::Rf64 {
            if !(28..=1 << 20).contains(&chunk_size)
                || data_offset
                    .checked_add(chunk_size)
                    .is_none_or(|end| end > file_len)
            {
                return Err(format!("RF64 ds64 chunk is truncated ({})", path.display()));
            }
            rf64_data_size = Some(read_rf64_ds64_table(
                &mut file,
                usize::try_from(chunk_size).unwrap(),
                &mut rf64_chunk_sizes,
                &mut rf64_table_entries_seen,
                budget,
                "RF64 probe",
            )?);
        } else if &header[0..4] == b"fmt " {
            if chunk_size < 16 {
                return Err(format!("WAVE fmt chunk is truncated ({})", path.display()));
            }
            let read_len = usize::try_from(chunk_size.min(64)).unwrap();
            if data_offset
                .checked_add(read_len as u64)
                .is_none_or(|end| end > file_len)
            {
                return Err(format!(
                    "WAVE fmt chunk exceeds the file ({})",
                    path.display()
                ));
            }
            let mut fmt = [0u8; 64];
            file.read_exact(&mut fmt[..read_len])
                .map_err(|error| format!("read WAVE fmt chunk {}: {error}", path.display()))?;
            codec = wave_codec_from_fmt(&fmt[..read_len]);
            found_fmt = true;
        } else if &header[0..4] == b"bext" {
            is_broadcast_wave = true;
        } else if &header[0..4] == b"data" {
            if data_offset
                .checked_add(effective_chunk_size)
                .is_none_or(|end| end > file_len)
            {
                return Err(format!(
                    "WAVE data chunk exceeds the file ({})",
                    path.display()
                ));
            }
            found_data = true;
        }

        let Some(next) = data_offset
            .checked_add(effective_chunk_size)
            .and_then(|end| end.checked_add(effective_chunk_size & 1))
        else {
            return Err(format!("WAVE chunk size overflows ({})", path.display()));
        };
        if next <= offset || next > file_len {
            return Err(format!("WAVE chunk exceeds the file ({})", path.display()));
        }
        offset = next;
    }

    if !found_fmt {
        return Err(format!(
            "WAVE file has no valid fmt chunk ({})",
            path.display()
        ));
    }
    if !found_data {
        return Err(format!("WAVE file has no data chunk ({})", path.display()));
    }
    Ok(AudioProbe {
        format,
        codec,
        audio_tracks: 1,
        has_non_audio_tracks: false,
        is_broadcast_wave,
    })
}

fn wave_codec_from_fmt(fmt: &[u8]) -> AudioCodec {
    if fmt.len() < 16 {
        return AudioCodec::Unknown;
    }
    match u16::from_le_bytes([fmt[0], fmt[1]]) {
        1 | 3 => AudioCodec::Pcm,
        0xfffe if fmt.len() >= 40 => {
            const PCM_GUID: [u8; 16] = [
                1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71,
            ];
            const FLOAT_GUID: [u8; 16] = [
                3, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71,
            ];
            if fmt[24..40] == PCM_GUID || fmt[24..40] == FLOAT_GUID {
                AudioCodec::Pcm
            } else {
                AudioCodec::Unknown
            }
        }
        _ => AudioCodec::Unknown,
    }
}

fn probe_mp4_tracks_from_file(
    path: &Path,
    mut file: std::fs::File,
    limits: DecodeLimits,
) -> Result<AudioProbe, String> {
    use std::io::{BufReader, Seek, SeekFrom};

    let size = file
        .metadata()
        .map_err(|error| format!("stat {} for MP4 codec probe: {error}", path.display()))?
        .len();
    let budget = DecodeBudget::new(limits);
    let mut fallback_file = file
        .try_clone()
        .map_err(|error| format!("clone {} for MP4 fallback probe: {error}", path.display()))?;
    let primary = (|| {
        m4a::preflight_mp4_parser(&mut file, size, budget)
            .map_err(|error| format!("validate M4A/MP4 structure ({}): {error}", path.display()))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind {} for MP4 codec probe: {error}", path.display()))?;
        mp4::Mp4Reader::read_header(BufReader::new(file), size)
            .map(|reader| summarize_mp4_tracks(reader.tracks()))
            .map_err(|error| format!("parse M4A/MP4 track metadata ({}): {error}", path.display()))
    })();

    if let Ok(probe) = primary {
        if probe.audio_tracks > 0 && probe.codec != AudioCodec::Unknown {
            return Ok(probe);
        }
    }

    fallback_file
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {} for MP4 fallback probe: {error}", path.display()))?;
    match probe_container_tracks_from_file(path, AudioFormat::M4a, fallback_file) {
        Ok(probe) if probe.codec == AudioCodec::Alac => Ok(probe),
        Ok(mut probe) => {
            // The generic demuxer does not establish that an `mp4a` sample
            // entry is the AAC-LC profile supported by this project. Keep the
            // primary parser's conservative result instead of upgrading an
            // unsupported AAC/USAC/ALS profile to preservable AAC.
            if let Ok(primary_probe) = &primary {
                Ok(*primary_probe)
            } else {
                probe.codec = AudioCodec::Unknown;
                Ok(probe)
            }
        }
        Err(symphonia_error) => match primary {
            // The primary parser still gives a safe answer for an unknown or
            // unsupported audio sample entry. Let planning reject it as
            // ambiguous instead of guessing AAC.
            Ok(probe) => Ok(probe),
            Err(mp4_error) => Err(format!(
                "{mp4_error}; fallback probe failed: {symphonia_error}"
            )),
        },
    }
}

fn summarize_mp4_tracks(tracks: &std::collections::HashMap<u32, mp4::Mp4Track>) -> AudioProbe {
    let mut audio_tracks = 0;
    let mut has_non_audio_tracks = false;
    let mut codec = None;
    let mut codecs_disagree = false;

    for track in tracks.values() {
        match track.track_type() {
            Ok(mp4::TrackType::Audio) => {
                audio_tracks += 1;
                let track_codec = if matches!(track.media_type(), Ok(mp4::MediaType::AAC))
                    && matches!(
                        track.audio_profile(),
                        Ok(mp4::AudioObjectType::AacLowComplexity)
                    )
                    && track.sample_freq_index().is_ok()
                    && track.channel_config().is_ok()
                {
                    AudioCodec::Aac
                } else {
                    AudioCodec::Unknown
                };
                if let Some(previous) = codec {
                    codecs_disagree |= previous != track_codec;
                } else {
                    codec = Some(track_codec);
                }
            }
            _ => has_non_audio_tracks = true,
        }
    }

    AudioProbe {
        format: AudioFormat::M4a,
        codec: if codecs_disagree {
            AudioCodec::Unknown
        } else {
            codec.unwrap_or(AudioCodec::Unknown)
        },
        audio_tracks,
        has_non_audio_tracks,
        is_broadcast_wave: false,
    }
}

fn probe_ogg_tracks(
    path: &Path,
    header_format: AudioFormat,
    metadata_limits: crate::metadata::MetadataLimits,
    mut source: std::fs::File,
) -> Result<AudioProbe, String> {
    use std::io::{Seek, SeekFrom};

    // Ask the demuxer to validate the first physical link, then scan every BOS
    // page so chained or multiplexed logical streams cannot be silently lost.
    // Preflight, BOS inspection, and demuxing all use handles cloned from this
    // one open file, so a path replacement cannot make the stages disagree.
    crate::metadata::preflight_ogg_decode(&mut source, metadata_limits)?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {} for Ogg codec probe: {error}", path.display()))?;
    let mut bos_source = source
        .try_clone()
        .map_err(|error| format!("clone {} for Ogg stream probe: {error}", path.display()))?;
    let codecs = scan_ogg_bos_codecs(&mut bos_source, path, &metadata_limits)?;
    // File::try_clone may share a cursor with the original handle. Rewind only
    // after the clone scan is complete, immediately before handing it off.
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {} for Ogg demux probe: {error}", path.display()))?;
    probe_container_tracks_from_file(path, header_format, source)?;
    let audio_tracks = codecs
        .iter()
        .filter(|codec| matches!(codec, AudioCodec::Opus | AudioCodec::Vorbis))
        .count();
    let has_non_audio_tracks = codecs.iter().any(|codec| *codec == AudioCodec::Unknown);
    let mut known = codecs
        .iter()
        .copied()
        .filter(|codec| *codec != AudioCodec::Unknown);
    let first = known.next().unwrap_or(AudioCodec::Unknown);
    let codecs_disagree = known.any(|codec| codec != first);
    let codec = if has_non_audio_tracks || codecs_disagree {
        AudioCodec::Unknown
    } else {
        first
    };
    let format = match codec {
        AudioCodec::Opus => AudioFormat::OggOpus,
        AudioCodec::Vorbis => AudioFormat::OggVorbis,
        _ => AudioFormat::Unknown,
    };
    Ok(AudioProbe {
        format,
        codec,
        audio_tracks,
        has_non_audio_tracks,
        is_broadcast_wave: false,
    })
}

fn scan_ogg_bos_codecs(
    file: &mut std::fs::File,
    path: &Path,
    metadata_limits: &crate::metadata::MetadataLimits,
) -> Result<Vec<AudioCodec>, String> {
    use std::io::{Read, Seek, SeekFrom};

    let file_len = file
        .metadata()
        .map_err(|error| format!("stat {} for Ogg stream probe: {error}", path.display()))?
        .len();
    let mut offset = 0u64;
    let mut codecs = Vec::new();
    while offset < file_len {
        if offset
            .checked_add(27)
            .is_none_or(|header_end| header_end > file_len)
        {
            return Err(format!("truncated Ogg page header ({})", path.display()));
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("seek Ogg page {}: {error}", path.display()))?;
        let mut header = [0u8; 27];
        file.read_exact(&mut header)
            .map_err(|error| format!("read Ogg page {}: {error}", path.display()))?;
        if &header[0..4] != b"OggS" || header[4] != 0 {
            return Err(format!("invalid Ogg page header ({})", path.display()));
        }
        if header[5] & !0x07 != 0 {
            return Err(format!("invalid Ogg page flags ({})", path.display()));
        }
        let lacing_len = usize::from(header[26]);
        let mut lacing = [0u8; 255];
        let lacing = &mut lacing[..lacing_len];
        file.read_exact(lacing)
            .map_err(|error| format!("read Ogg lacing table {}: {error}", path.display()))?;
        let body_size = lacing.iter().map(|value| *value as u64).sum::<u64>();
        let body_offset = offset
            .checked_add(27)
            .and_then(|value| value.checked_add(lacing.len() as u64))
            .ok_or_else(|| format!("Ogg page offset overflows ({})", path.display()))?;
        let body_end = body_offset
            .checked_add(body_size)
            .ok_or_else(|| format!("Ogg page size overflows ({})", path.display()))?;
        if body_end > file_len || body_end <= offset {
            return Err(format!("Ogg page exceeds the file ({})", path.display()));
        }

        if header[5] & 0x02 != 0 {
            if header[5] & 0x01 != 0 {
                return Err(format!(
                    "Ogg BOS page cannot continue an earlier packet ({})",
                    path.display()
                ));
            }
            if codecs.len() >= metadata_limits.max_ogg_streams {
                return Err(format!(
                    "Ogg logical stream count exceeds the configured limit of {} ({})",
                    metadata_limits.max_ogg_streams,
                    path.display()
                ));
            }
            let mut first_packet_size = 0_u64;
            for segment in lacing.iter().copied() {
                first_packet_size = first_packet_size
                    .checked_add(u64::from(segment))
                    .ok_or_else(|| format!("Ogg BOS packet size overflows ({})", path.display()))?;
                if segment < 255 {
                    break;
                }
            }
            let mut prefix = [0u8; 8];
            let prefix_len = usize::try_from(first_packet_size.min(prefix.len() as u64)).unwrap();
            file.read_exact(&mut prefix[..prefix_len])
                .map_err(|error| format!("read Ogg BOS packet {}: {error}", path.display()))?;
            codecs
                .try_reserve(1)
                .map_err(|_| format!("allocate Ogg stream probe ({})", path.display()))?;
            codecs.push(if prefix.starts_with(b"OpusHead") {
                AudioCodec::Opus
            } else if prefix.starts_with(b"\x01vorbis") {
                AudioCodec::Vorbis
            } else {
                AudioCodec::Unknown
            });
        }
        offset = body_end;
    }
    if codecs.is_empty() {
        return Err(format!(
            "Ogg container has no BOS stream ({})",
            path.display()
        ));
    }
    Ok(codecs)
}

fn probe_container_tracks_from_file(
    path: &Path,
    header_format: AudioFormat,
    source: std::fs::File,
) -> Result<AudioProbe, String> {
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let stream = MediaSourceStream::new(Box::new(source), Default::default());
    let mut hint = Hint::new();
    hint.with_extension(if header_format == AudioFormat::M4a {
        "mp4"
    } else {
        "ogg"
    });
    let format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| {
            format!(
                "the {} header was recognized, but its track metadata could not be parsed ({}): {error}",
                if header_format == AudioFormat::M4a {
                    "M4A/MP4"
                } else {
                    "Ogg"
                },
                path.display()
            )
        })?;

    Ok(summarize_container_tracks(header_format, format.tracks()))
}

fn summarize_container_tracks(
    header_format: AudioFormat,
    tracks: &[symphonia::core::formats::Track],
) -> AudioProbe {
    let mut audio_tracks = 0;
    let mut has_non_audio_tracks = false;
    let mut codec = None;
    let mut codecs_disagree = false;

    for track in tracks {
        match track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
        {
            Some(params) => {
                audio_tracks += 1;
                let track_codec = audio_codec_from_symphonia(params.codec);
                if let Some(previous) = codec {
                    codecs_disagree |= previous != track_codec;
                } else {
                    codec = Some(track_codec);
                }
            }
            None => {
                // Preserve mode must not silently discard video, subtitle, or
                // unclassified tracks. Treat every non-audio track as extra,
                // including one whose parameters are absent.
                has_non_audio_tracks = true;
            }
        }
    }

    let codec = if codecs_disagree {
        AudioCodec::Unknown
    } else {
        codec.unwrap_or(AudioCodec::Unknown)
    };
    let format = match header_format {
        AudioFormat::OggOpus | AudioFormat::OggVorbis => match codec {
            AudioCodec::Opus => AudioFormat::OggOpus,
            AudioCodec::Vorbis => AudioFormat::OggVorbis,
            // AudioFormat has no generic Ogg variant. Do not mislabel an
            // unknown or mixed Ogg stream as one of the supported codecs.
            _ => AudioFormat::Unknown,
        },
        _ => header_format,
    };

    AudioProbe {
        format,
        codec,
        audio_tracks,
        has_non_audio_tracks,
        is_broadcast_wave: false,
    }
}

fn audio_codec_from_symphonia(codec: symphonia::core::codecs::audio::AudioCodecId) -> AudioCodec {
    use symphonia::core::codecs::audio::well_known::{
        CODEC_ID_AAC, CODEC_ID_ALAC, CODEC_ID_FLAC, CODEC_ID_MP3, CODEC_ID_OPUS, CODEC_ID_VORBIS,
    };

    match codec {
        CODEC_ID_FLAC => AudioCodec::Flac,
        CODEC_ID_OPUS => AudioCodec::Opus,
        CODEC_ID_VORBIS => AudioCodec::Vorbis,
        CODEC_ID_MP3 => AudioCodec::Mp3,
        CODEC_ID_AAC => AudioCodec::Aac,
        CODEC_ID_ALAC => AudioCodec::Alac,
        _ => AudioCodec::Unknown,
    }
}

/// Decode any supported regular audio file to high-fidelity planar PCM.
///
/// FIFOs, directories, and device files are rejected.
pub fn decode_file(path: &Path) -> Result<DecodedPcm, String> {
    decode_file_with_limits(path, DecodeLimits::default())
}

/// Decode a regular-file audio input with explicit FLAC/Ogg metadata limits.
///
/// FLAC and Ogg structures are validated on the same open file handle that is
/// subsequently passed to the decoder. This prevents oversized metadata from
/// reaching third-party parsers and avoids a path-reopen race between the
/// validation and decode stages.
pub fn decode_file_with_metadata_limits(
    path: &Path,
    metadata_limits: crate::metadata::MetadataLimits,
) -> Result<DecodedPcm, String> {
    decode_file_with_limits(
        path,
        DecodeLimits {
            metadata: metadata_limits,
            ..DecodeLimits::default()
        },
    )
}

/// Decode a regular-file audio input with explicit resource limits.
pub fn decode_file_with_limits(path: &Path, limits: DecodeLimits) -> Result<DecodedPcm, String> {
    let mut session = crate::input::AudioInputSession::open(path)?;
    decode_file_from_session_with_limits(&mut session, limits)
}

/// Decode the regular file held by an existing input session.
pub fn decode_file_from_session_with_limits(
    session: &mut crate::input::AudioInputSession,
    limits: DecodeLimits,
) -> Result<DecodedPcm, String> {
    let path = session.path().to_path_buf();
    let source = session.try_clone_rewound("audio decode")?;
    decode_file_from_file_with_limits(&path, source, limits)
}

/// Detect and decode an already-open input without reopening its pathname.
pub(crate) fn decode_file_from_file_with_limits(
    path: &Path,
    mut source: std::fs::File,
    limits: DecodeLimits,
) -> Result<DecodedPcm, String> {
    use std::io::{Seek, SeekFrom};

    DecodeBudget::new(limits).check_planar_frames(0, 0, 0, "audio decode")?;
    let fmt = detect_file_format_from_file(path, true, &mut source)?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {} after format detection: {error}", path.display()))?;
    decode_file_from_detected_file(path, fmt, source, limits)
}

fn decode_file_from_detected_file(
    path: &Path,
    fmt: AudioFormat,
    mut source: std::fs::File,
    limits: DecodeLimits,
) -> Result<DecodedPcm, String> {
    match fmt {
        AudioFormat::Wav => {
            crate::audio::read_wav_from_file_with_limits(source, limits).map(|audio| DecodedPcm {
                sample_rate: audio.sample_rate,
                channels: audio.channels,
                channel_mask: audio.channel_mask,
            })
        }
        AudioFormat::Rf64 => decode_rf64(source, limits),
        AudioFormat::Aiff | AudioFormat::Caf => {
            decode_symphonia(path, source, DecodeBudget::new(limits))
        }
        AudioFormat::OggVorbis => decode_symphonia_ogg(path, source, limits),
        AudioFormat::Flac => decode_flac(source, limits),
        AudioFormat::OggOpus => opus::decode_ogg_opus_with_limits(source, limits),
        AudioFormat::Mp3 => decode_mp3(path, source, limits),
        AudioFormat::M4a => {
            let m4a_source = clone_rewound(&mut source, path, "M4A primary decode")?;
            match m4a::decode_m4a(m4a_source, limits) {
                Ok(decoded) => Ok(decoded),
                Err(m4a::M4aDecodeError::Fatal(error)) => {
                    Err(format!("M4A/AAC decode failed: {error}"))
                }
                Err(m4a::M4aDecodeError::TryOtherCodec {
                    reason,
                    track_edits,
                    retained_bytes,
                }) => {
                    let fallback_budget = DecodeBudget::new(limits)
                        .with_retained_bytes(retained_bytes)
                        .map_err(|error| format!("M4A fallback metadata: {error}"))?;
                    let fallback_source = clone_rewound(&mut source, path, "M4A fallback decode")?;
                    let (mut decoded, selected_track_id, packet_errors) =
                        decode_symphonia_with_track_id(
                            path,
                            fallback_source,
                            &track_edits,
                            fallback_budget,
                        )
                        .map_err(|symphonia_error| {
                            format!(
                                "M4A/AAC decode failed: {reason}; ALAC/other decoder: {symphonia_error}"
                            )
                        })?;
                    let edit_active = m4a::fallback_track_has_edit(
                        &track_edits,
                        selected_track_id,
                    )
                    .map_err(|edit_error| {
                            format!(
                                "M4A/AAC decode failed: {reason}; ALAC/other edit list: {edit_error}"
                            )
                    })?;
                    if edit_active {
                        reject_m4a_fallback_packet_errors(&packet_errors).map_err(|edit_error| {
                            format!(
                                "M4A/AAC decode failed: {reason}; ALAC/other edit list: {edit_error}"
                            )
                        })?;
                    }
                    m4a::apply_fallback_track_edit(
                        &mut decoded,
                        selected_track_id,
                        track_edits,
                        fallback_budget,
                    )
                    .map_err(|edit_error| {
                        format!(
                            "M4A/AAC decode failed: {reason}; ALAC/other edit list: {edit_error}"
                        )
                    })?;
                    Ok(decoded)
                }
            }
        }
        AudioFormat::AacAdts => aac::decode_adts(source, limits),
        AudioFormat::Unknown => Err(format!(
            "unsupported audio format ({}); supported input: wav, rf64/bwf, aiff, caf, flac, opus/vorbis, mp3, m4a/alac, aac",
            path.display()
        )),
    }
}

fn clone_rewound(
    source: &mut std::fs::File,
    path: &Path,
    context: &str,
) -> Result<std::fs::File, String> {
    use std::io::{Seek, SeekFrom};

    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {context} {}: {error}", path.display()))?;
    let mut clone = source
        .try_clone()
        .map_err(|error| format!("clone {context} {}: {error}", path.display()))?;
    clone
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind cloned {context} {}: {error}", path.display()))?;
    Ok(clone)
}

/// Decode a container supported by Symphonia into planar `f64` PCM.
///
/// Symphonia returns a packet-local audio buffer whose sample type depends on
/// the source codec. Its generic planar conversion keeps the conversion in one
/// place and avoids assuming that AIFF/CAF/ALAC are always integer PCM.
#[derive(Default)]
struct RecoverablePacketErrors {
    invalid_main_data_offset: bool,
    other: Option<String>,
    reset_required: bool,
    empty_decoded_packet: bool,
}

struct SymphoniaDecodeOutcome {
    decoded: Option<DecodedPcm>,
    selected_track_id: u32,
    expected_sample_rate: Option<u32>,
    expected_channel_count: Option<usize>,
    packet_errors: RecoverablePacketErrors,
}

impl SymphoniaDecodeOutcome {
    fn into_decoded(self) -> Result<DecodedPcm, String> {
        self.decoded
            .ok_or_else(|| "no audio packets decoded".to_string())
    }
}

fn decode_symphonia(
    path: &Path,
    source: std::fs::File,
    budget: DecodeBudget,
) -> Result<DecodedPcm, String> {
    decode_symphonia_with_report(path, source, budget)?.into_decoded()
}

fn decode_symphonia_ogg(
    path: &Path,
    mut source: std::fs::File,
    limits: DecodeLimits,
) -> Result<DecodedPcm, String> {
    use std::io::{Seek, SeekFrom};

    crate::metadata::preflight_ogg_decode(&mut source, limits.metadata)?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind Ogg input {}: {error}", path.display()))?;
    decode_symphonia_with_report_inner(path, None, source, DecodeBudget::new(limits))?
        .into_decoded()
}

fn decode_symphonia_with_track_id(
    path: &Path,
    source: std::fs::File,
    track_edits: &[m4a::FallbackTrackEdit],
    budget: DecodeBudget,
) -> Result<(DecodedPcm, u32, RecoverablePacketErrors), String> {
    let SymphoniaDecodeOutcome {
        decoded,
        selected_track_id,
        packet_errors,
        ..
    } = decode_symphonia_with_report_inner(path, Some(track_edits), source, budget)?;
    let decoded = decoded.ok_or_else(|| "no audio packets decoded".to_string())?;
    Ok((decoded, selected_track_id, packet_errors))
}

fn reject_m4a_fallback_packet_errors(errors: &RecoverablePacketErrors) -> Result<(), String> {
    if errors.reset_required {
        return Err("M4A fallback decoder stopped early because a reset was required".into());
    }
    if errors.empty_decoded_packet {
        return Err("M4A fallback decoder produced an empty packet on the edited timeline".into());
    }
    if errors.invalid_main_data_offset {
        return Err(
            "M4A fallback decoder skipped a packet with an invalid main-data offset".into(),
        );
    }
    if let Some(error) = &errors.other {
        return Err(format!(
            "M4A fallback decoder skipped a packet error: {error}"
        ));
    }
    Ok(())
}

fn decode_symphonia_with_report(
    path: &Path,
    source: std::fs::File,
    budget: DecodeBudget,
) -> Result<SymphoniaDecodeOutcome, String> {
    decode_symphonia_with_report_inner(path, None, source, budget)
}

fn decode_symphonia_with_report_inner(
    path: &Path,
    fallback_track_edits: Option<&[m4a::FallbackTrackEdit]>,
    source: std::fs::File,
    budget: DecodeBudget,
) -> Result<SymphoniaDecodeOutcome, String> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let stream = MediaSourceStream::new(Box::new(source), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| format!("probe: {error}"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| "no audio track found".to_string())?;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| "audio track has no codec parameters".to_string())?;
    let channel_mask = codec_params
        .channels
        .as_ref()
        .and_then(|channels| match channels {
            symphonia::core::audio::Channels::Positioned(position) => {
                u32::try_from(position.bits())
                    .ok()
                    .and_then(crate::channel_layout::ChannelMask::from_bits)
            }
            _ => None,
        });
    let expected_sample_rate = codec_params.sample_rate;
    let expected_channel_count = codec_params
        .channels
        .as_ref()
        .map(|channels| channels.count())
        .filter(|count| *count > 0);
    let declared_packet_temporary =
        match (expected_channel_count, codec_params.max_frames_per_packet) {
            (Some(channel_count), Some(frames)) => {
                symphonia_temporary_bytes(channel_count, frames)?
            }
            _ => 0,
        };
    if let (Some(channel_count), Some(frame_count)) = (expected_channel_count, track.num_frames) {
        let frame_count = usize::try_from(frame_count)
            .map_err(|_| "Symphonia track frame count is too large for this platform")?;
        budget.check_planar_frames(
            channel_count,
            frame_count,
            declared_packet_temporary,
            "Symphonia decode",
        )?;
    }
    let track_id = track.id;
    let mut timeline_verifier = match fallback_track_edits {
        Some(track_edits) => m4a::fallback_timeline_verifier(track_edits, track_id)?,
        None => None,
    };
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
        .map_err(|error| format!("decoder: {error}"))?;
    let mut sample_rate = None;
    let mut channels: Vec<Vec<f64>> = Vec::new();
    let mut packet_errors = RecoverablePacketErrors::default();

    loop {
        // The format reader may allocate its next packet internally. Charge
        // all currently reserved output capacity and the declared decoder
        // packet scratch before entering that dependency.
        budget.check_planar_capacities(
            &channels,
            declared_packet_temporary,
            "Symphonia packet read",
        )?;
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => {
                packet_errors.reset_required = true;
                break;
            }
            Err(error) => return Err(format!("read packet: {error}")),
        };
        let packet_data_bytes = u64::try_from(packet.data.len())
            .map_err(|_| "Symphonia packet byte count does not fit in u64")?;
        let predecode_temporary = packet_data_bytes
            .checked_add(declared_packet_temporary)
            .ok_or_else(|| "Symphonia pre-decode temporary byte count overflows".to_string())?;
        budget.check_planar_capacities(
            &channels,
            predecode_temporary,
            "Symphonia packet decode",
        )?;
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError("mpa: invalid main_data offset")) => {
                packet_errors.invalid_main_data_offset = true;
                continue;
            }
            Err(error @ SymphoniaError::IoError(_))
            | Err(error @ SymphoniaError::DecodeError(_)) => {
                packet_errors.other.get_or_insert_with(|| error.to_string());
                continue;
            }
            Err(error) => return Err(format!("decode packet: {error}")),
        };
        let rate = decoded.spec().rate();
        let decoded_frames = decoded.frames();
        if let Some(verifier) = timeline_verifier.as_mut() {
            verifier.observe_packet(decoded_frames, rate)?;
        }
        if decoded_frames == 0 {
            packet_errors.empty_decoded_packet = true;
        }
        let plane_count = decoded.num_planes();
        if plane_count == 0 {
            packet_errors.empty_decoded_packet = true;
            continue;
        }
        if let Some(expected_rate) = sample_rate {
            if expected_rate != rate {
                return Err(format!(
                    "audio sample rate changed from {expected_rate} to {rate}"
                ));
            }
        } else {
            sample_rate = Some(rate);
            let packet_temporary = symphonia_temporary_bytes(
                plane_count,
                u64::try_from(decoded.capacity())
                    .map_err(|_| "Symphonia packet capacity does not fit in u64")?,
            )?
            .checked_add(packet_data_bytes)
            .ok_or_else(|| "Symphonia packet temporary byte count overflows".to_string())?;
            budget.check_planar_frames(
                plane_count,
                decoded_frames,
                packet_temporary,
                "Symphonia decode",
            )?;
            channels
                .try_reserve_exact(plane_count)
                .map_err(|error| format!("reserve Symphonia channel list: {error}"))?;
            channels.resize_with(plane_count, Vec::new);
        }
        if plane_count != channels.len() {
            return Err("audio channel count changed during decode".into());
        }

        // Symphonia owns the current decoded packet. Count its capacity as a
        // conservative f64-width codec temporary, then reserve every output
        // plane before extending any of them.
        let packet_temporary = symphonia_temporary_bytes(
            plane_count,
            u64::try_from(decoded.capacity())
                .map_err(|_| "Symphonia packet capacity does not fit in u64")?,
        )?
        .checked_add(packet_data_bytes)
        .ok_or_else(|| "Symphonia packet temporary byte count overflows".to_string())?;
        let previous_frames = channels.first().map(Vec::len).unwrap_or(0);
        let target_frames = budget.reserve_planar_additional(
            &mut channels,
            decoded_frames,
            packet_temporary,
            "Symphonia decode",
        )?;
        for destination in &mut channels {
            destination.resize(target_frames, 0.0);
        }
        let mut destinations = Vec::new();
        destinations
            .try_reserve_exact(plane_count)
            .map_err(|error| format!("reserve Symphonia plane views: {error}"))?;
        for destination in &mut channels {
            destinations.push(&mut destination[previous_frames..target_frames]);
        }
        decoded.copy_to_slice_planar::<f64, _>(&mut destinations);
        drop(destinations);
        for destination in &mut channels {
            for sample in &mut destination[previous_frames..target_frames] {
                *sample = crate::audio::sanitize_sample(*sample);
            }
        }
    }

    if let Some(verifier) = timeline_verifier.as_ref() {
        verifier.finish()?;
    }

    let decoded = sample_rate.map(|sample_rate| DecodedPcm {
        sample_rate,
        channels,
        channel_mask,
    });
    Ok(SymphoniaDecodeOutcome {
        decoded,
        selected_track_id: track_id,
        expected_sample_rate,
        expected_channel_count,
        packet_errors,
    })
}

fn symphonia_temporary_bytes(channel_count: usize, frames: u64) -> Result<u64, String> {
    let channels =
        u64::try_from(channel_count).map_err(|_| "Symphonia channel count does not fit in u64")?;
    channels
        .checked_mul(frames)
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
        .and_then(|sample_bytes| {
            channels
                .checked_mul(std::mem::size_of::<&mut [f64]>() as u64)
                .and_then(|views| sample_bytes.checked_add(views))
        })
        .ok_or_else(|| "Symphonia packet temporary byte count overflows".to_string())
}

fn decode_mp3(
    path: &Path,
    mut source: std::fs::File,
    limits: DecodeLimits,
) -> Result<DecodedPcm, String> {
    let primary_source = clone_rewound(&mut source, path, "MP3 primary decode")?;
    let attempt = decode_symphonia_with_report(path, primary_source, DecodeBudget::new(limits))?;
    if !attempt.packet_errors.invalid_main_data_offset {
        return attempt.into_decoded().map(normalize_mp3_layout);
    }

    if let Some(other) = &attempt.packet_errors.other {
        return Err(format!(
            "MP3 decode encountered Symphonia's invalid main-data offset and another packet error ({other}); compatibility fallback is unsafe"
        ));
    }

    let timing_source = clone_rewound(&mut source, path, "MP3 fallback timing probe")?;
    let (classified_rate, classified_channels) = match mp3::inspect_timing_metadata(
        timing_source,
        limits,
    ) {
        mp3::TimingMetadata::Absent {
            sample_rate,
            channel_count,
        } => (sample_rate, channel_count),
        mp3::TimingMetadata::Present => {
            return Err(
                "MP3 decode encountered an invalid main-data offset; compatibility fallback is disabled because the first frame contains Xing/Info/VBRI timing metadata"
                    .into(),
            );
        }
        mp3::TimingMetadata::Undetermined(reason) => {
            return Err(format!(
                "MP3 decode encountered an invalid main-data offset; compatibility fallback is disabled because timing metadata could not be ruled out: {reason}"
            ));
        }
    };

    let SymphoniaDecodeOutcome {
        decoded,
        expected_sample_rate,
        expected_channel_count,
        ..
    } = attempt;
    let primary_frames = decoded.as_ref().map(DecodedPcm::frames).unwrap_or(0);
    // Do not retain a potentially long partial Symphonia buffer while the
    // compatibility decoder accumulates a replacement whole-stream PCM.
    drop(decoded);

    let fallback_source = clone_rewound(&mut source, path, "MP3 compatibility fallback")?;
    let decoded = mp3::decode_mp3_file_compatibility(fallback_source, limits).map_err(|fallback_error| {
        format!(
            "MP3 compatibility fallback failed after Symphonia's invalid main-data offset: {fallback_error}"
        )
    })?;
    let fallback_frames = decoded.frames();
    let fallback_channels = decoded.n_channels();
    if decoded.sample_rate != classified_rate || fallback_channels != classified_channels {
        return Err(format!(
            "MP3 compatibility fallback changed the first-frame format ({} Hz/{classified_channels} ch to {} Hz/{fallback_channels} ch)",
            classified_rate, decoded.sample_rate
        ));
    }
    if expected_sample_rate.is_some_and(|rate| rate != decoded.sample_rate) {
        return Err(format!(
            "MP3 compatibility fallback sample rate {} does not match Symphonia track metadata {:?}",
            decoded.sample_rate, expected_sample_rate
        ));
    }
    if expected_channel_count.is_some_and(|count| count != fallback_channels) {
        return Err(format!(
            "MP3 compatibility fallback channel count {fallback_channels} does not match Symphonia track metadata {:?}",
            expected_channel_count
        ));
    }
    if fallback_frames <= primary_frames {
        return Err(format!(
            "MP3 compatibility fallback did not recover additional audio ({fallback_frames} frames versus {primary_frames} from Symphonia)"
        ));
    }

    Ok(normalize_mp3_layout(decoded))
}

fn normalize_mp3_layout(mut decoded: DecodedPcm) -> DecodedPcm {
    // MPEG audio only represents mono or stereo. Symphonia's MPEG track
    // parameters currently label mono as FRONT_LEFT even though the decoder
    // emits its standard centered mono layout. Preserve the public layout
    // contract used by every other one-channel decoder.
    decoded.channel_mask =
        crate::channel_layout::ChannelLayout::from_channel_count(decoded.channels.len()).mask();
    decoded
}

/// Decode RF64 PCM without materialising the encoded file in memory.
///
/// RF64 is the 64-bit extension of RIFF/WAVE. The `ds64` chunk carries the
/// sizes that cannot fit in the legacy 32-bit RIFF fields; the actual sample
/// payload is still ordinary little-endian WAVE PCM. Broadcast-WAVE (BWF) is
/// handled by the normal RIFF/WAVE reader because its `bext` chunk is metadata
/// that can be skipped by `hound`.
fn checked_rf64_decoded_bytes(frame_count: usize, channel_count: usize) -> Result<usize, String> {
    frame_count
        .checked_mul(channel_count)
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| "RF64 decoded byte count overflows".to_string())
}

fn decode_rf64(mut file: std::fs::File, limits: DecodeLimits) -> Result<DecodedPcm, String> {
    use std::collections::HashMap;
    use std::io::{Read, Seek, SeekFrom};

    let budget = DecodeBudget::new(limits);
    let file_len = file
        .metadata()
        .map_err(|error| format!("stat RF64: {error}"))?
        .len();
    if file_len < 12 {
        return Err("RF64 header is truncated".into());
    }
    let mut header = [0u8; 12];
    file.read_exact(&mut header)
        .map_err(|error| format!("read RF64 header: {error}"))?;
    if &header[0..4] != b"RF64" || &header[8..12] != b"WAVE" {
        return Err("invalid RF64/WAVE header".into());
    }

    let mut data_size_from_ds64 = None;
    let mut extended_chunk_sizes = HashMap::<[u8; 4], u64>::new();
    let mut extended_table_entries_seen = 0usize;
    let mut format = None;
    let mut data_offset = None;
    let mut data_size = None;

    loop {
        let chunk_offset = file
            .stream_position()
            .map_err(|error| format!("locate RF64 chunk: {error}"))?;
        if chunk_offset == file_len {
            break;
        }
        let chunk_data_offset = chunk_offset
            .checked_add(8)
            .ok_or_else(|| "RF64 chunk header offset overflows".to_string())?;
        if chunk_data_offset > file_len {
            return Err("RF64 chunk header is truncated".into());
        }

        let mut id = [0u8; 4];
        file.read_exact(&mut id)
            .map_err(|error| format!("read RF64 chunk: {error}"))?;
        let size32 = read_u32_le(&mut file, "RF64 chunk size")?;
        let declared_size = if size32 == u32::MAX {
            *extended_chunk_sizes
                .get(&id)
                .or_else(|| {
                    (id == *b"data")
                        .then_some(&data_size_from_ds64)
                        .and_then(Option::as_ref)
                })
                .ok_or_else(|| {
                    format!(
                        "RF64 chunk {:?} uses 0xffffffff without a ds64 size",
                        String::from_utf8_lossy(&id)
                    )
                })?
        } else {
            u64::from(size32)
        };
        let chunk_end = chunk_data_offset
            .checked_add(declared_size)
            .ok_or_else(|| {
                format!(
                    "RF64 chunk {:?} size overflows the file offset",
                    String::from_utf8_lossy(&id)
                )
            })?;
        let padded_end = chunk_end
            .checked_add(declared_size & 1)
            .ok_or_else(|| "RF64 chunk padding offset overflows".to_string())?;
        if chunk_end > file_len {
            return Err(format!(
                "RF64 chunk {:?} exceeds the file length",
                String::from_utf8_lossy(&id)
            ));
        }
        if padded_end > file_len {
            return Err(format!(
                "RF64 chunk {:?} has truncated padding",
                String::from_utf8_lossy(&id)
            ));
        }

        match &id {
            b"ds64" => {
                let body_len = usize::try_from(u64::from(size32))
                    .map_err(|_| "RF64 ds64 chunk is too large".to_string())?;
                data_size_from_ds64 = Some(read_rf64_ds64_table(
                    &mut file,
                    body_len,
                    &mut extended_chunk_sizes,
                    &mut extended_table_entries_seen,
                    budget,
                    "RF64 decode",
                )?);
            }
            b"fmt " => {
                let body_len = usize::try_from(declared_size)
                    .map_err(|_| "RF64 fmt chunk is too large".to_string())?;
                if body_len < 16 || body_len > 1 << 20 {
                    return Err("RF64 fmt chunk has an invalid size".into());
                }
                let retained_table_bytes = rf64_table_working_bytes(
                    extended_table_entries_seen,
                    0,
                    body_len,
                    "RF64 decode",
                )?;
                budget.check_peak(0, retained_table_bytes, "RF64 format parse")?;
                let mut body = Vec::new();
                body.try_reserve_exact(body_len)
                    .map_err(|error| format!("reserve RF64 fmt body: {error}"))?;
                body.resize(body_len, 0);
                file.read_exact(&mut body)
                    .map_err(|error| format!("read RF64 fmt: {error}"))?;
                let format_tag =
                    u16::from_le_bytes(body[0..2].try_into().expect("fixed format tag"));
                let channels =
                    u16::from_le_bytes(body[2..4].try_into().expect("fixed channel count"));
                let sample_rate =
                    u32::from_le_bytes(body[4..8].try_into().expect("fixed sample rate"));
                let block_align =
                    u16::from_le_bytes(body[12..14].try_into().expect("fixed block align"));
                let container_bits_per_sample =
                    u16::from_le_bytes(body[14..16].try_into().expect("fixed bit depth"));
                let extensible = format_tag == 0xfffe;
                let (format_tag, valid_bits_per_sample) = if extensible {
                    if body.len() < 40 {
                        return Err("RF64 extensible fmt chunk is truncated".into());
                    }
                    let subformat =
                        u16::from_le_bytes(body[24..26].try_into().expect("fixed subformat"));
                    (
                        subformat,
                        u16::from_le_bytes(body[18..20].try_into().expect("fixed valid bits")),
                    )
                } else {
                    (format_tag, container_bits_per_sample)
                };
                if channels == 0
                    || sample_rate == 0
                    || block_align == 0
                    || container_bits_per_sample == 0
                    || valid_bits_per_sample == 0
                {
                    return Err("RF64 fmt chunk contains invalid audio parameters".into());
                }
                if valid_bits_per_sample > container_bits_per_sample {
                    return Err("RF64 valid sample depth exceeds its PCM container width".into());
                }
                if !matches!(format_tag, 1 | 3) {
                    return Err(format!("RF64 codec format 0x{format_tag:04x} is unsupported; only PCM and IEEE float are supported"));
                }
                let channel_mask = if extensible {
                    let bits =
                        u32::from_le_bytes(body[20..24].try_into().expect("fixed channel mask"));
                    let mask = crate::channel_layout::ChannelMask::from_bits(bits)
                        .ok_or_else(|| format!("RF64 channel mask 0x{bits:08x} is invalid"))?;
                    if mask.bits() != 0 && mask.channels() != channels as usize {
                        return Err(format!(
                            "RF64 channel mask has {} positions but fmt declares {channels} channels",
                            mask.channels()
                        ));
                    }
                    Some(mask)
                } else {
                    None
                };
                format = Some((
                    format_tag == 3,
                    channels as usize,
                    sample_rate,
                    block_align as usize,
                    container_bits_per_sample,
                    valid_bits_per_sample,
                    channel_mask,
                ));
            }
            b"data" => {
                data_offset = Some(chunk_data_offset);
                data_size = Some(declared_size);
            }
            _ => {}
        }
        file.seek(SeekFrom::Start(padded_end))
            .map_err(|error| format!("skip RF64 chunk: {error}"))?;
        if format.is_some() && data_offset.is_some() {
            break;
        }
    }

    let (
        is_float,
        channel_count,
        sample_rate,
        block_align,
        container_bits_per_sample,
        valid_bits_per_sample,
        channel_mask,
    ) = format.ok_or_else(|| "RF64 fmt chunk not found".to_string())?;
    let data_offset = data_offset.ok_or_else(|| "RF64 data chunk not found".to_string())?;
    let data_size = data_size.ok_or_else(|| "RF64 data size not found".to_string())?;
    if block_align == 0 || data_size % block_align as u64 != 0 {
        return Err("RF64 data size is not aligned to complete audio frames".into());
    }
    let frame_count = usize::try_from(data_size / block_align as u64)
        .map_err(|_| "RF64 audio is too large for this platform".to_string())?;
    let _decoded_bytes = checked_rf64_decoded_bytes(frame_count, channel_count)?;
    if container_bits_per_sample > 64
        || valid_bits_per_sample > 64
        || (is_float
            && (container_bits_per_sample != valid_bits_per_sample
                || !matches!(container_bits_per_sample, 32 | 64)))
    {
        return Err(format!(
            "RF64 sample depth {valid_bits_per_sample} in a {container_bits_per_sample}-bit container is unsupported"
        ));
    }
    let bytes_per_sample = (container_bits_per_sample as usize).div_ceil(8);
    if bytes_per_sample == 0 || bytes_per_sample.saturating_mul(channel_count) > block_align {
        return Err("RF64 fmt block alignment is invalid".into());
    }

    // Extended sizes are no longer needed once the data and format locations
    // have been resolved; release the table before allocating whole-stream PCM.
    drop(extended_chunk_sizes);
    file.seek(SeekFrom::Start(data_offset))
        .map_err(|error| format!("seek RF64 data: {error}"))?;
    let block_frames = (64 * 1024 / block_align).max(1).min(frame_count.max(1));
    let buffer_len = block_frames
        .checked_mul(block_align)
        .ok_or_else(|| "RF64 decode buffer size overflows".to_string())?;
    let buffer_bytes =
        u64::try_from(buffer_len).map_err(|_| "RF64 buffer size does not fit in u64")?;
    budget.check_planar_frames(channel_count, frame_count, buffer_bytes, "RF64 decode")?;
    let mut channels = Vec::new();
    channels
        .try_reserve_exact(channel_count)
        .map_err(|_| "unable to reserve RF64 channel list".to_string())?;
    channels.resize_with(channel_count, Vec::new);
    budget.reserve_planar_frames(&mut channels, frame_count, buffer_bytes, "RF64 decode")?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(buffer_len)
        .map_err(|_| "unable to reserve RF64 decode buffer".to_string())?;
    buffer.resize(buffer_len, 0);
    let mut remaining = data_size;
    while remaining > 0 {
        let wanted =
            usize::try_from(remaining.min(buffer.len() as u64)).expect("buffer length fits");
        file.read_exact(&mut buffer[..wanted])
            .map_err(|error| format!("read RF64 samples: {error}"))?;
        for frame in buffer[..wanted].chunks_exact(block_align) {
            for channel in 0..channel_count {
                let sample = &frame[channel * bytes_per_sample..][..bytes_per_sample];
                let value = if is_float {
                    match container_bits_per_sample {
                        32 => f32::from_le_bytes(sample.try_into().expect("float32 sample")) as f64,
                        64 => f64::from_le_bytes(sample.try_into().expect("float64 sample")),
                        _ => unreachable!("validated float depth"),
                    }
                } else {
                    let mut raw = 0u64;
                    for (index, byte) in sample.iter().enumerate() {
                        raw |= u64::from(*byte) << (index * 8);
                    }
                    if container_bits_per_sample < 64 {
                        raw &= (1u64 << container_bits_per_sample) - 1;
                    }
                    let value_bits = raw >> (container_bits_per_sample - valid_bits_per_sample);
                    let midpoint = 1u64 << (valid_bits_per_sample - 1);
                    if container_bits_per_sample == 8 {
                        (value_bits as f64 - midpoint as f64) / midpoint as f64
                    } else {
                        let signed = if value_bits & midpoint != 0 {
                            i128::from(value_bits) - (1i128 << valid_bits_per_sample)
                        } else {
                            i128::from(value_bits)
                        };
                        signed as f64 / midpoint as f64
                    }
                };
                channels[channel].push(crate::audio::sanitize_sample(value));
            }
        }
        remaining -= wanted as u64;
    }

    Ok(DecodedPcm {
        sample_rate,
        channels,
        channel_mask,
    })
}

fn read_u32_le(file: &mut impl std::io::Read, context: &str) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("{context}: {error}"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn decode_flac(mut source: std::fs::File, limits: DecodeLimits) -> Result<DecodedPcm, String> {
    use std::io::{Seek, SeekFrom};

    crate::metadata::preflight_flac_decode(&mut source, limits.metadata)?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("FLAC rewind: {error}"))?;
    let reader_options = claxon::FlacReaderOptions {
        read_vorbis_comment: false,
        ..claxon::FlacReaderOptions::default()
    };
    let mut reader = claxon::FlacReader::new_ext(source, reader_options)
        .map_err(|e| format!("FLAC open: {e}"))?;
    let info = reader.streaminfo();
    let channels = info.channels as usize;
    let scale = 1.0 / (1_u64 << (info.bits_per_sample - 1)) as f64;
    let budget = DecodeBudget::new(limits);
    let frame_temporary = u64::from(info.max_block_size)
        .checked_mul(u64::from(info.channels))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<i32>() as u64))
        .ok_or_else(|| "FLAC frame temporary byte count overflows".to_string())?;
    let mut output = Vec::new();
    let planned_frames = info
        .samples
        .map(usize::try_from)
        .transpose()
        .map_err(|_| "FLAC decoded frame count is too large for this platform")?;
    budget.check_planar_frames(
        channels,
        planned_frames.unwrap_or(0),
        frame_temporary,
        "FLAC decode",
    )?;
    output
        .try_reserve_exact(channels)
        .map_err(|error| format!("reserve FLAC channel list: {error}"))?;
    output.resize_with(channels, Vec::new);
    if let Some(frames) = planned_frames {
        budget.reserve_planar_frames(&mut output, frames, frame_temporary, "FLAC decode")?;
    }
    let mut retained_frames = 0usize;
    for (index, sample) in reader.samples().enumerate() {
        if index % channels == 0 {
            retained_frames = retained_frames
                .checked_add(1)
                .ok_or_else(|| "FLAC decoded frame count overflows".to_string())?;
            if output[0].capacity() < retained_frames {
                let retained_before = retained_frames - 1;
                debug_assert!(output.iter().all(|plane| plane.len() == retained_before));
                budget.reserve_planar_additional(&mut output, 1, frame_temporary, "FLAC decode")?;
            }
        }
        output[index % channels]
            .push(sample.map_err(|e| format!("FLAC decode: {e}"))? as f64 * scale);
    }
    Ok(DecodedPcm {
        sample_rate: info.sample_rate,
        channels: output,
        channel_mask: crate::channel_layout::ChannelLayout::from_channel_count(channels).mask(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SILENT_STEREO_ADTS: [u8; 13] = [
        0xff, 0xf1, 0x50, 0x80, 0x01, 0xbf, 0xfc, 0x21, 0x00, 0x00, 0x00, 0x00, 0x1c,
    ];

    fn synchsafe(size: usize) -> [u8; 4] {
        assert!(size <= 0x0fff_ffff);
        [
            ((size >> 21) & 0x7f) as u8,
            ((size >> 14) & 0x7f) as u8,
            ((size >> 7) & 0x7f) as u8,
            (size & 0x7f) as u8,
        ]
    }

    fn id3v2_tag(version: u8, flags: u8, body_len: usize, footer: bool) -> Vec<u8> {
        let size = synchsafe(body_len);
        let mut tag = b"ID3".to_vec();
        tag.extend([version, 0, flags]);
        tag.extend(size);
        tag.resize(ID3V2_HEADER_BYTES + body_len, 0);
        if footer {
            tag.extend(b"3DI");
            tag.extend([version, 0, flags]);
            tag.extend(size);
        }
        tag
    }

    fn flac_with_vendor(vendor_len: usize) -> Vec<u8> {
        let vendor_len = u32::try_from(vendor_len).expect("vendor fixture fits u32");
        let comment_len = 4_u32
            .checked_add(vendor_len)
            .and_then(|length| length.checked_add(4))
            .expect("comment fixture length");
        assert!(comment_len <= 0x00ff_ffff);

        let mut bytes = b"fLaC".to_vec();
        bytes.extend([0, 0, 0, 34]);
        bytes.extend([0; 34]);
        bytes.push(0x80 | 4);
        let comment_len_bytes = comment_len.to_be_bytes();
        bytes.extend_from_slice(&comment_len_bytes[1..]);
        bytes.extend(vendor_len.to_le_bytes());
        bytes.resize(bytes.len() + vendor_len as usize, b'v');
        bytes.extend(0_u32.to_le_bytes());
        bytes
    }

    fn rf64_ds64_only(table_entries: usize) -> Vec<u8> {
        let body_len = 28usize
            .checked_add(table_entries.checked_mul(12).unwrap())
            .unwrap();
        let mut bytes = b"RF64".to_vec();
        bytes.extend(u32::MAX.to_le_bytes());
        bytes.extend(b"WAVE");
        bytes.extend(b"ds64");
        bytes.extend(u32::try_from(body_len).unwrap().to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(u32::try_from(table_entries).unwrap().to_le_bytes());
        for index in 0..table_entries {
            bytes.extend(b"JUNK");
            bytes.extend(u64::try_from(index).unwrap().to_le_bytes());
        }
        bytes
    }

    fn ogg_opus_with_misleading_tags() -> Vec<u8> {
        use ogg::writing::{PacketWriteEndInfo, PacketWriter};
        use std::borrow::Cow;

        let serial = 0x51_0aa_c51;
        let mut writer = PacketWriter::new(std::io::Cursor::new(Vec::new()));
        let mut head = b"OpusHead".to_vec();
        head.extend([1, 1]);
        head.extend(0u16.to_le_bytes());
        head.extend(48_000u32.to_le_bytes());
        head.extend(0i16.to_le_bytes());
        head.push(0);
        writer
            .write_packet(Cow::Owned(head), serial, PacketWriteEndInfo::EndPage, 0)
            .unwrap();

        let vendor = b"\x01vorbis";
        let mut tags = b"OpusTags".to_vec();
        tags.extend((vendor.len() as u32).to_le_bytes());
        tags.extend(vendor);
        tags.extend(0u32.to_le_bytes());
        writer
            .write_packet(Cow::Owned(tags), serial, PacketWriteEndInfo::EndPage, 0)
            .unwrap();

        let mut encoder =
            ::opus::Encoder::new(48_000, ::opus::Channels::Mono, ::opus::Application::Audio)
                .unwrap();
        let packet = encoder.encode_vec_float(&[0.0; 960], 4_000).unwrap();
        writer
            .write_packet(
                Cow::Owned(packet),
                serial,
                PacketWriteEndInfo::EndStream,
                960,
            )
            .unwrap();
        writer.into_inner().into_inner()
    }

    fn symphonia_audio_track(
        id: u32,
        codec: symphonia::core::codecs::audio::AudioCodecId,
    ) -> symphonia::core::formats::Track {
        use symphonia::core::codecs::audio::AudioCodecParameters;
        use symphonia::core::codecs::CodecParameters;

        let mut params = AudioCodecParameters::new();
        params.for_codec(codec);
        let mut track = symphonia::core::formats::Track::new(id);
        track.with_codec_params(CodecParameters::Audio(params));
        track
    }

    #[test]
    fn detect_wav() {
        let h = b"RIFF\x00\x00\x00\x00WAVE";
        assert_eq!(AudioFormat::detect(Path::new("x.wav"), h), AudioFormat::Wav);
    }

    #[test]
    fn detects_payload_after_in_memory_id3() {
        let mut mp3 = id3v2_tag(4, 0, 3, false);
        mp3.extend(b"\xff\xfb\x90\x64\0\0\0\0\0\0\0\0");
        assert_eq!(AudioFormat::detect(Path::new(""), &mp3), AudioFormat::Mp3);

        let mut aac = id3v2_tag(4, 0, 3, false);
        aac.extend(SILENT_STEREO_ADTS);
        assert_eq!(
            AudioFormat::detect(Path::new(""), &aac),
            AudioFormat::AacAdts
        );
    }

    #[test]
    fn detect_m4a_ftyp() {
        let h = b"\x00\x00\x00\x20ftypM4A ";
        assert_eq!(AudioFormat::detect(Path::new("x.m4a"), h), AudioFormat::M4a);
    }

    #[test]
    fn m4a_fallback_never_slices_pcm_after_skipped_packet_errors() {
        let mut errors = RecoverablePacketErrors {
            invalid_main_data_offset: true,
            other: None,
            ..RecoverablePacketErrors::default()
        };
        assert!(reject_m4a_fallback_packet_errors(&errors)
            .unwrap_err()
            .contains("invalid main-data offset"));

        errors.invalid_main_data_offset = false;
        errors.other = Some("corrupt ALAC packet".into());
        assert!(reject_m4a_fallback_packet_errors(&errors)
            .unwrap_err()
            .contains("corrupt ALAC packet"));

        errors.other = None;
        errors.reset_required = true;
        assert!(reject_m4a_fallback_packet_errors(&errors)
            .unwrap_err()
            .contains("reset was required"));
        errors.reset_required = false;
        errors.empty_decoded_packet = true;
        assert!(reject_m4a_fallback_packet_errors(&errors)
            .unwrap_err()
            .contains("empty packet"));
        assert!(reject_m4a_fallback_packet_errors(&RecoverablePacketErrors::default()).is_ok());
    }

    #[test]
    fn detect_adts_before_mp3() {
        let h = b"\xff\xf1\x50\x80\x00\x1f\xfc\x00\x00\x00\x00\x00";
        assert_eq!(
            AudioFormat::detect(Path::new("x.aac"), h),
            AudioFormat::AacAdts
        );
        assert_eq!(
            AudioFormat::detect(Path::new("x.aac"), b""),
            AudioFormat::AacAdts
        );
    }

    #[test]
    fn id3v2_prefix_obeys_versioned_footer_synchsafe_and_file_bounds() {
        let mut v23 = id3v2_tag(3, 0x10, 4, false);
        v23.push(0xaa);
        assert_eq!(strip_id3v2_prefix(&v23).unwrap(), &[0xaa]);

        let mut v24 = id3v2_tag(4, 0x10, 4, true);
        v24.push(0xbb);
        assert_eq!(strip_id3v2_prefix(&v24).unwrap(), &[0xbb]);

        let exact = id3v2_tag(4, 0, 4, false);
        assert_eq!(
            id3v2_payload_offset(&exact, exact.len() as u64).unwrap(),
            Some(exact.len() as u64)
        );
        assert!(id3v2_payload_offset(&exact, exact.len() as u64 - 1)
            .unwrap_err()
            .contains("extends beyond"));

        let mut non_synchsafe = id3v2_tag(4, 0, 0, false);
        non_synchsafe[6] = 0x80;
        assert!(strip_id3v2_prefix(&non_synchsafe)
            .unwrap_err()
            .contains("synchsafe"));
        assert!(strip_id3v2_prefix(b"ID3\x04")
            .unwrap_err()
            .contains("truncated"));
    }

    #[test]
    fn large_id3v2_adts_is_probed_and_decoded_from_its_payload() {
        let mut bytes = id3v2_tag(4, 0, FORMAT_SNIFF_BYTES + 17, false);
        bytes.extend(SILENT_STEREO_ADTS);
        assert_eq!(
            AudioFormat::detect(Path::new(""), &bytes),
            AudioFormat::AacAdts
        );

        let file = tempfile::NamedTempFile::new().expect("create tagged ADTS fixture");
        std::fs::write(file.path(), &bytes).expect("write tagged ADTS fixture");
        let probe = probe_file(file.path()).expect("probe tagged ADTS payload");
        assert_eq!(
            probe,
            single_audio_track(AudioFormat::AacAdts, AudioCodec::Aac)
        );

        let decoded = decode_file(file.path()).expect("decode tagged ADTS payload");
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.n_channels(), 2);
        assert_eq!(decoded.frames(), 1_024);
    }

    #[test]
    fn oversized_flac_vendor_is_rejected_before_probe_or_decode_parser() {
        let file = tempfile::NamedTempFile::new().expect("create oversized FLAC fixture");
        std::fs::write(file.path(), flac_with_vendor(64)).expect("write oversized FLAC fixture");
        let mut limits = crate::metadata::MetadataLimits::default();
        limits.max_item_bytes = 32;

        let probe_error = probe_file_with_metadata_limits(file.path(), limits)
            .expect_err("bounded probe must reject oversized FLAC vendor");
        assert!(
            probe_error.contains("32") || probe_error.contains("limit"),
            "{probe_error}"
        );

        let decode_error = decode_file_with_metadata_limits(file.path(), limits)
            .expect_err("bounded decode must reject oversized FLAC vendor");
        assert!(
            decode_error.contains("32") || decode_error.contains("limit"),
            "{decode_error}"
        );
    }

    #[test]
    fn compressed_flac_silence_is_rejected_from_streaminfo_before_pcm_reserve() {
        let directory = tempfile::tempdir().expect("create FLAC fixture directory");
        let path = directory.path().join("compressed-silence.flac");
        let frames = 100_000usize;
        let audio = crate::audio::Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; frames], vec![0.0; frames]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        crate::encode::write_audio(&path, &audio, crate::encode::EncodeOptions::default())
            .expect("encode compressed FLAC silence fixture");

        const CAP_BYTES: u64 = 2 * 1024 * 1024;
        let encoded_bytes = std::fs::metadata(&path)
            .expect("stat compressed FLAC fixture")
            .len();
        assert!(
            encoded_bytes < CAP_BYTES,
            "fixture is not compressed enough"
        );

        let error = decode_file_with_limits(
            &path,
            DecodeLimits::new(crate::metadata::MetadataLimits::default(), Some(CAP_BYTES)),
        )
        .expect_err("STREAMINFO geometry must reject decoded PCM beyond the cap");
        assert!(error.contains("FLAC decode"), "{error}");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn rf64_ds64_table_is_rejected_before_hostile_reserve() {
        let entry_count = 20_000;
        let bytes = rf64_ds64_only(entry_count);
        let file = tempfile::NamedTempFile::new().expect("create RF64 table fixture");
        std::fs::write(file.path(), bytes).expect("write RF64 table fixture");
        let limits = DecodeLimits::new(
            crate::metadata::MetadataLimits::default(),
            Some(1024 * 1024),
        );

        let probe_error = probe_file_with_limits(file.path(), limits)
            .expect_err("RF64 probe must reject table beyond one-MiB cap");
        assert!(probe_error.contains("RF64 probe"), "{probe_error}");
        assert!(probe_error.contains("working-set limit"), "{probe_error}");

        let decode_error = decode_file_with_limits(file.path(), limits)
            .expect_err("RF64 decode must reject table beyond one-MiB cap");
        assert!(decode_error.contains("RF64 decode"), "{decode_error}");
        assert!(decode_error.contains("working-set limit"), "{decode_error}");
    }

    #[test]
    fn rf64_ds64_table_has_finite_entry_ceiling() {
        let error =
            rf64_table_working_bytes(MAX_RF64_SIZE_TABLE_ENTRIES, 1, 0, "RF64 test").unwrap_err();
        assert!(error.contains("exceeds"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn detected_ogg_handle_survives_path_replacement_before_decode() {
        use std::io::{Seek, SeekFrom};

        let directory = tempfile::tempdir().expect("create replacement fixture directory");
        let path = directory.path().join("input.ogg");
        let replacement = directory.path().join("replacement.bin");
        std::fs::write(&path, ogg_opus_with_misleading_tags()).expect("write original Ogg file");

        let mut source = std::fs::File::open(&path).expect("open original Ogg handle");
        let format = detect_file_format_from_file(&path, true, &mut source)
            .expect("detect original Ogg handle");
        assert_eq!(format, AudioFormat::OggOpus);
        source
            .seek(SeekFrom::Start(0))
            .expect("rewind original Ogg handle");

        std::fs::write(&replacement, flac_with_vendor(64)).expect("write replacement FLAC file");
        std::fs::rename(&replacement, &path).expect("atomically replace the path");

        let decoded =
            decode_file_from_detected_file(&path, format, source, DecodeLimits::default())
                .expect("decode must continue from the already-open Ogg inode");
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.n_channels(), 1);
        assert_eq!(decoded.frames(), 960);
        assert!(std::fs::read(&path)
            .expect("read replacement path")
            .starts_with(b"fLaC"));
    }

    #[test]
    fn ogg_detection_uses_only_the_first_bos_packet() {
        let opus = ogg_opus_with_misleading_tags();
        assert!(opus
            .windows(b"\x01vorbis".len())
            .any(|window| window == b"\x01vorbis"));
        assert_eq!(
            AudioFormat::detect(Path::new("misleading.ogg"), &opus),
            AudioFormat::OggOpus
        );

        let file = tempfile::NamedTempFile::new().expect("create misleading Opus fixture");
        std::fs::write(file.path(), &opus).expect("write misleading Opus fixture");
        let probe = probe_file(file.path()).expect("probe misleading Opus fixture");
        assert_eq!(
            probe,
            single_audio_track(AudioFormat::OggOpus, AudioCodec::Opus)
        );
        let decoded = decode_file(file.path()).expect("decode misleading Opus fixture");
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.n_channels(), 1);
        assert_eq!(decoded.frames(), 960);

        let mut unknown = ogg_opus_with_misleading_tags();
        let body_offset = 27 + usize::from(unknown[26]);
        unknown[body_offset..body_offset + 8].copy_from_slice(b"Unknown!");
        assert_eq!(
            AudioFormat::detect(Path::new("unknown.ogg"), &unknown),
            AudioFormat::Unknown
        );

        assert_eq!(
            AudioFormat::detect(Path::new("partial.ogg"), b"OggS\0"),
            AudioFormat::OggOpus
        );

        use ogg::writing::{PacketWriteEndInfo, PacketWriter};
        use std::borrow::Cow;
        let mut extended_head = b"OpusHead".to_vec();
        extended_head.resize(FORMAT_SNIFF_BYTES + 1_000, 0);
        let mut writer = PacketWriter::new(Vec::new());
        writer
            .write_packet(
                Cow::Owned(extended_head),
                51,
                PacketWriteEndInfo::EndPage,
                0,
            )
            .unwrap();
        let extended_stream = writer.into_inner();
        assert_eq!(
            AudioFormat::detect(
                Path::new("extended.opus"),
                &extended_stream[..FORMAT_SNIFF_BYTES]
            ),
            AudioFormat::OggOpus
        );
    }

    #[test]
    fn summarize_ogg_uses_track_codec_instead_of_header_guess() {
        use symphonia::core::codecs::audio::well_known::CODEC_ID_VORBIS;

        let tracks = [symphonia_audio_track(0, CODEC_ID_VORBIS)];
        let probe = summarize_container_tracks(AudioFormat::OggOpus, &tracks);

        assert_eq!(probe.format, AudioFormat::OggVorbis);
        assert_eq!(probe.codec, AudioCodec::Vorbis);
        assert_eq!(probe.audio_tracks, 1);
        assert!(!probe.has_non_audio_tracks);
    }

    #[test]
    fn summarize_m4a_reports_all_tracks_and_alac() {
        use symphonia::core::codecs::audio::well_known::CODEC_ID_ALAC;
        use symphonia::core::codecs::video::VideoCodecParameters;
        use symphonia::core::codecs::CodecParameters;

        let audio = symphonia_audio_track(0, CODEC_ID_ALAC);
        let mut video = symphonia::core::formats::Track::new(1);
        video.with_codec_params(CodecParameters::Video(VideoCodecParameters::default()));

        let probe = summarize_container_tracks(AudioFormat::M4a, &[audio, video]);

        assert_eq!(probe.format, AudioFormat::M4a);
        assert_eq!(probe.codec, AudioCodec::Alac);
        assert_eq!(probe.audio_tracks, 1);
        assert!(probe.has_non_audio_tracks);
    }

    #[test]
    fn summarize_unclassified_track_prevents_implicit_preservation() {
        use symphonia::core::codecs::audio::well_known::CODEC_ID_AAC;

        let audio = symphonia_audio_track(0, CODEC_ID_AAC);
        let unclassified = symphonia::core::formats::Track::new(1);

        let probe = summarize_container_tracks(AudioFormat::M4a, &[audio, unclassified]);

        assert_eq!(probe.codec, AudioCodec::Aac);
        assert_eq!(probe.audio_tracks, 1);
        assert!(probe.has_non_audio_tracks);
    }

    #[test]
    fn summarize_mixed_audio_codecs_is_unknown() {
        use symphonia::core::codecs::audio::well_known::{CODEC_ID_AAC, CODEC_ID_ALAC};

        let tracks = [
            symphonia_audio_track(0, CODEC_ID_AAC),
            symphonia_audio_track(1, CODEC_ID_ALAC),
        ];
        let probe = summarize_container_tracks(AudioFormat::M4a, &tracks);

        assert_eq!(probe.codec, AudioCodec::Unknown);
        assert_eq!(probe.audio_tracks, 2);
        assert!(!probe.has_non_audio_tracks);
    }

    #[test]
    fn probe_mp4_counts_multiple_aac_and_non_audio_tracks() {
        let config = mp4::Mp4Config {
            major_brand: "M4A ".parse().unwrap(),
            minor_version: 0,
            compatible_brands: vec!["M4A ".parse().unwrap(), "isom".parse().unwrap()],
            timescale: 48_000,
        };
        let cursor = std::io::Cursor::new(Vec::new());
        let mut writer = mp4::Mp4Writer::write_start(cursor, &config).unwrap();
        let track = mp4::TrackConfig::from(mp4::AacConfig::default());
        writer.add_track(&track).unwrap();
        writer.add_track(&track).unwrap();
        writer
            .add_track(&mp4::TrackConfig::from(mp4::TtxtConfig::default()))
            .unwrap();
        writer.write_end().unwrap();
        let bytes = writer.into_writer().into_inner();
        let path = std::env::temp_dir().join(format!(
            "denoize-probe-{}-multiple-audio.m4a",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();

        let probe = probe_file(&path).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(probe.format, AudioFormat::M4a);
        assert_eq!(probe.codec, AudioCodec::Aac);
        assert_eq!(probe.audio_tracks, 2);
        assert!(probe.has_non_audio_tracks);
    }

    #[test]
    fn probe_file_rejects_nested_zero_size_mp4_box_without_stalling() {
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

        let error = probe_file(&path).expect_err("malformed MP4 probe must fail");
        assert!(error.contains("zero-sized MP4 box free"), "{error}");
    }

    #[test]
    fn probe_file_uses_content_instead_of_extension() {
        let path =
            std::env::temp_dir().join(format!("denoize-probe-{}-content.mp4", std::process::id()));
        let bytes = crate::audio::write_wav_bytes(&crate::audio::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 16]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        })
        .expect("encode probe fixture");
        std::fs::write(&path, bytes).expect("write probe fixture");

        let probe = probe_file(&path).expect("probe WAV header");
        std::fs::remove_file(&path).expect("remove probe fixture");

        assert_eq!(probe, single_audio_track(AudioFormat::Wav, AudioCodec::Pcm));
    }

    #[test]
    fn probe_file_does_not_assume_codec_from_extension() {
        use std::io::Write;

        let path =
            std::env::temp_dir().join(format!("denoize-probe-{}-unknown.m4a", std::process::id()));
        let mut file = std::fs::File::create(&path).expect("create probe fixture");
        file.write_all(b"not an audio file")
            .expect("write probe fixture");
        drop(file);

        let error = probe_file(&path).expect_err("extension alone must not determine a codec");
        std::fs::remove_file(&path).expect("remove probe fixture");

        assert!(error.contains("could not identify the audio container"));
    }

    #[test]
    fn rf64_decoded_byte_plan_rejects_arithmetic_overflow() {
        assert_eq!(checked_rf64_decoded_bytes(2, 3).unwrap(), 48);
        assert!(checked_rf64_decoded_bytes(usize::MAX, 2)
            .unwrap_err()
            .contains("decoded byte count overflows"));
    }
}
