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
//! | M4A  | `mp4` demux + `oxideav-aac` Pure-Rust AAC-LC decode |
//! | AIFF / CAF / Ogg Vorbis / ALAC | `symphonia` |

mod aac;
mod m4a;
mod mp3;
mod opus;
mod pcm;

pub use pcm::DecodedPcm;

use std::path::Path;

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
        if header.len() >= 12 {
            if &header[0..4] == b"RIFF" && header.len() >= 12 && &header[8..12] == b"WAVE" {
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
            if &header[0..4] == b"OggS" {
                if header.windows(7).any(|window| window == b"\x01vorbis") {
                    return AudioFormat::OggVorbis;
                }
                return AudioFormat::OggOpus;
            }
            if &header[4..8] == b"ftyp" {
                return AudioFormat::M4a;
            }
            // ADTS has a 12-bit sync word and its two layer bits are always 0.
            // Check it before the broader 11-bit MPEG audio sync test.
            if header[0] == 0xFF && (header[1] & 0xF6) == 0xF0 {
                return AudioFormat::AacAdts;
            }
            if &header[0..3] == b"ID3" {
                return AudioFormat::Mp3;
            }
            if header[0] == 0xFF && (header[1] & 0xE0) == 0xE0 {
                return AudioFormat::Mp3;
            }
        }

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

/// Inspect an audio file's container and codec without decoding its samples.
///
/// Container detection is content-based. Ogg and ISO BMFF containers are then
/// demuxed with Symphonia so that `.ogg` is not implicitly treated as Opus and
/// `.m4a` / `.mp4` is not implicitly treated as AAC.
pub fn probe_file(path: &Path) -> Result<AudioProbe, String> {
    let header = read_header(path, 4096)?;
    // Suppress extension fallback here. A probe must describe the file's
    // contents, not what its name claims the contents are.
    let format = AudioFormat::detect(Path::new(""), &header);

    match format {
        AudioFormat::Wav | AudioFormat::Rf64 => probe_wave_file(path, format),
        AudioFormat::Aiff | AudioFormat::Caf => Ok(single_audio_track(format, AudioCodec::Pcm)),
        AudioFormat::Flac => Ok(single_audio_track(format, AudioCodec::Flac)),
        AudioFormat::Mp3 => Ok(single_audio_track(format, AudioCodec::Mp3)),
        AudioFormat::AacAdts => Ok(single_audio_track(format, AudioCodec::Aac)),
        AudioFormat::OggOpus | AudioFormat::OggVorbis => probe_ogg_tracks(path, format),
        AudioFormat::M4a => probe_mp4_tracks(path),
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

fn probe_wave_file(path: &Path, format: AudioFormat) -> Result<AudioProbe, String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("open {} for WAVE codec probe: {error}", path.display()))?;
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
            let mut body = vec![0u8; usize::try_from(chunk_size).unwrap()];
            file.read_exact(&mut body)
                .map_err(|error| format!("read RF64 ds64 chunk {}: {error}", path.display()))?;
            rf64_data_size = Some(u64::from_le_bytes(body[8..16].try_into().unwrap()));
            let table_len = u32::from_le_bytes(body[24..28].try_into().unwrap()) as usize;
            let required = 28usize.saturating_add(table_len.saturating_mul(12));
            if required > body.len() {
                return Err(format!(
                    "RF64 ds64 chunk ends before its table ({})",
                    path.display()
                ));
            }
            for entry in body[28..required].chunks_exact(12) {
                let mut chunk_id = [0u8; 4];
                chunk_id.copy_from_slice(&entry[..4]);
                rf64_chunk_sizes.insert(
                    chunk_id,
                    u64::from_le_bytes(entry[4..12].try_into().unwrap()),
                );
            }
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
            let mut fmt = vec![0u8; read_len];
            file.read_exact(&mut fmt)
                .map_err(|error| format!("read WAVE fmt chunk {}: {error}", path.display()))?;
            codec = wave_codec_from_fmt(&fmt);
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

fn probe_mp4_tracks(path: &Path) -> Result<AudioProbe, String> {
    use std::io::BufReader;

    let file = std::fs::File::open(path)
        .map_err(|error| format!("open {} for MP4 codec probe: {error}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|error| format!("stat {} for MP4 codec probe: {error}", path.display()))?
        .len();
    let primary = mp4::Mp4Reader::read_header(BufReader::new(file), size)
        .map(|reader| summarize_mp4_tracks(reader.tracks()))
        .map_err(|error| format!("parse M4A/MP4 track metadata ({}): {error}", path.display()));

    if let Ok(probe) = primary {
        if probe.audio_tracks > 0 && probe.codec != AudioCodec::Unknown {
            return Ok(probe);
        }
    }

    match probe_container_tracks(path, AudioFormat::M4a) {
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

fn probe_ogg_tracks(path: &Path, header_format: AudioFormat) -> Result<AudioProbe, String> {
    // Ask the demuxer to validate the first physical link, then scan every BOS
    // page so chained or multiplexed logical streams cannot be silently lost.
    probe_container_tracks(path, header_format)?;
    let codecs = scan_ogg_bos_codecs(path)?;
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

fn scan_ogg_bos_codecs(path: &Path) -> Result<Vec<AudioCodec>, String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("open {} for Ogg stream probe: {error}", path.display()))?;
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
        let mut lacing = vec![0u8; header[26] as usize];
        file.read_exact(&mut lacing)
            .map_err(|error| format!("read Ogg lacing table {}: {error}", path.display()))?;
        let body_size = lacing.iter().map(|value| *value as u64).sum::<u64>();
        let body_offset = offset + 27 + lacing.len() as u64;
        let body_end = body_offset
            .checked_add(body_size)
            .ok_or_else(|| format!("Ogg page size overflows ({})", path.display()))?;
        if body_end > file_len || body_end <= offset {
            return Err(format!("Ogg page exceeds the file ({})", path.display()));
        }

        if header[5] & 0x02 != 0 {
            let mut prefix = [0u8; 8];
            let prefix_len = usize::try_from(body_size.min(prefix.len() as u64)).unwrap();
            file.read_exact(&mut prefix[..prefix_len])
                .map_err(|error| format!("read Ogg BOS packet {}: {error}", path.display()))?;
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

fn probe_container_tracks(path: &Path, header_format: AudioFormat) -> Result<AudioProbe, String> {
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let source = std::fs::File::open(path)
        .map_err(|error| format!("open {} for codec probe: {error}", path.display()))?;
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

/// Decode any supported audio file to high-fidelity planar PCM.
pub fn decode_file(path: &Path) -> Result<DecodedPcm, String> {
    let header = read_header(path, 4096)?;
    let fmt = AudioFormat::detect(path, &header);

    match fmt {
        AudioFormat::Wav => decode_wav(path),
        AudioFormat::Rf64 => decode_rf64(path),
        AudioFormat::Aiff | AudioFormat::Caf | AudioFormat::OggVorbis => decode_symphonia(path),
        AudioFormat::Flac => decode_flac(path),
        AudioFormat::OggOpus => opus::decode_ogg_opus(path),
        AudioFormat::Mp3 => decode_mp3(path),
        AudioFormat::M4a => m4a::decode_m4a(path).or_else(|aac_error| {
            decode_symphonia(path).map_err(|symphonia_error| {
                format!("M4A/AAC decode failed: {aac_error}; ALAC/other decoder: {symphonia_error}")
            })
        }),
        AudioFormat::AacAdts => aac::decode_adts(path),
        AudioFormat::Unknown => Err(format!(
            "unsupported audio format ({}); supported input: wav, rf64/bwf, aiff, caf, flac, opus/vorbis, mp3, m4a/alac, aac",
            path.display()
        )),
    }
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
}

struct SymphoniaDecodeOutcome {
    decoded: Option<DecodedPcm>,
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

fn decode_symphonia(path: &Path) -> Result<DecodedPcm, String> {
    decode_symphonia_with_report(path)?.into_decoded()
}

fn decode_symphonia_with_report(path: &Path) -> Result<SymphoniaDecodeOutcome, String> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let source = std::fs::File::open(path).map_err(|error| format!("open: {error}"))?;
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
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
        .map_err(|error| format!("decoder: {error}"))?;
    let track_id = track.id;
    let mut sample_rate = None;
    let mut channels: Vec<Vec<f64>> = Vec::new();
    let mut packet_errors = RecoverablePacketErrors::default();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => break,
            Err(error) => return Err(format!("read packet: {error}")),
        };
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
        let mut planes = Vec::<Vec<f32>>::new();
        decoded.copy_to_vecs_planar(&mut planes);
        if planes.is_empty() {
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
            channels = vec![Vec::new(); planes.len()];
        }
        if planes.len() != channels.len() {
            return Err("audio channel count changed during decode".into());
        }
        for (destination, source) in channels.iter_mut().zip(planes) {
            destination.extend(
                source
                    .into_iter()
                    .map(|sample| crate::audio::sanitize_sample(sample as f64)),
            );
        }
    }

    let decoded = sample_rate.map(|sample_rate| DecodedPcm {
        sample_rate,
        channels,
        channel_mask,
    });
    Ok(SymphoniaDecodeOutcome {
        decoded,
        expected_sample_rate,
        expected_channel_count,
        packet_errors,
    })
}

fn decode_mp3(path: &Path) -> Result<DecodedPcm, String> {
    let attempt = decode_symphonia_with_report(path)?;
    if !attempt.packet_errors.invalid_main_data_offset {
        return attempt.into_decoded().map(normalize_mp3_layout);
    }

    if let Some(other) = &attempt.packet_errors.other {
        return Err(format!(
            "MP3 decode encountered Symphonia's invalid main-data offset and another packet error ({other}); compatibility fallback is unsafe"
        ));
    }

    let (classified_rate, classified_channels) = match mp3::inspect_timing_metadata(path) {
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

    let decoded = mp3::decode_mp3_file_compatibility(path).map_err(|fallback_error| {
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

fn decode_rf64(path: &Path) -> Result<DecodedPcm, String> {
    use std::collections::HashMap;
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path).map_err(|error| format!("open RF64: {error}"))?;
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
                if body_len < 28 || body_len > 1 << 20 {
                    return Err("RF64 ds64 chunk has an invalid size".into());
                }
                let mut body = vec![0u8; body_len];
                file.read_exact(&mut body)
                    .map_err(|error| format!("read RF64 ds64: {error}"))?;
                data_size_from_ds64 = Some(u64::from_le_bytes(
                    body[8..16].try_into().expect("fixed ds64 data size"),
                ));
                let table_len =
                    u32::from_le_bytes(body[24..28].try_into().expect("fixed ds64 table length"))
                        as usize;
                let required = table_len
                    .checked_mul(12)
                    .and_then(|bytes| bytes.checked_add(28))
                    .ok_or_else(|| "RF64 ds64 table size overflows".to_string())?;
                if required > body.len() {
                    return Err("RF64 ds64 chunk ends before its table".into());
                }
                for entry in body[28..required].chunks_exact(12) {
                    let mut chunk_id = [0u8; 4];
                    chunk_id.copy_from_slice(&entry[..4]);
                    let size =
                        u64::from_le_bytes(entry[4..12].try_into().expect("fixed table size"));
                    extended_chunk_sizes.insert(chunk_id, size);
                }
            }
            b"fmt " => {
                let body_len = usize::try_from(declared_size)
                    .map_err(|_| "RF64 fmt chunk is too large".to_string())?;
                if body_len < 16 || body_len > 1 << 20 {
                    return Err("RF64 fmt chunk has an invalid size".into());
                }
                let mut body = vec![0u8; body_len];
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

    file.seek(SeekFrom::Start(data_offset))
        .map_err(|error| format!("seek RF64 data: {error}"))?;
    let mut channels = Vec::new();
    channels
        .try_reserve_exact(channel_count)
        .map_err(|_| "unable to reserve RF64 channel list".to_string())?;
    for _ in 0..channel_count {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frame_count)
            .map_err(|_| "unable to reserve RF64 decoded samples".to_string())?;
        channels.push(channel);
    }
    let block_frames = (64 * 1024 / block_align).max(1).min(frame_count.max(1));
    let buffer_len = block_frames
        .checked_mul(block_align)
        .ok_or_else(|| "RF64 decode buffer size overflows".to_string())?;
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

fn decode_flac(path: &Path) -> Result<DecodedPcm, String> {
    let mut reader = claxon::FlacReader::open(path).map_err(|e| format!("FLAC open: {e}"))?;
    let info = reader.streaminfo();
    let channels = info.channels as usize;
    let scale = 1.0 / (1_u64 << (info.bits_per_sample - 1)) as f64;
    let mut output = vec![Vec::new(); channels];
    for (index, sample) in reader.samples().enumerate() {
        output[index % channels]
            .push(sample.map_err(|e| format!("FLAC decode: {e}"))? as f64 * scale);
    }
    Ok(DecodedPcm {
        sample_rate: info.sample_rate,
        channels: output,
        channel_mask: crate::channel_layout::ChannelLayout::from_channel_count(channels).mask(),
    })
}

fn read_header(path: &Path, n: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut buf = vec![0u8; n];
    let got = f.read(&mut buf).map_err(|e| format!("read: {e}"))?;
    buf.truncate(got);
    Ok(buf)
}

fn decode_wav(path: &Path) -> Result<DecodedPcm, String> {
    let audio = crate::audio::read_wav(path)?;
    Ok(DecodedPcm {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        channel_mask: audio.channel_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn detect_mp3_id3() {
        assert_eq!(
            AudioFormat::detect(Path::new("x.mp3"), b"ID3"),
            AudioFormat::Mp3
        );
    }

    #[test]
    fn detect_m4a_ftyp() {
        let h = b"\x00\x00\x00\x20ftypM4A ";
        assert_eq!(AudioFormat::detect(Path::new("x.m4a"), h), AudioFormat::M4a);
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
