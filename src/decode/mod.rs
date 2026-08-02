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
//! | MP3  | `nanomp3`（Pure Rust / minimp3 移植） |
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
        AudioFormat::Mp3 => mp3::decode_mp3_file(path),
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
fn decode_symphonia(path: &Path) -> Result<DecodedPcm, String> {
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
                crate::channel_layout::ChannelMask::from_bits(position.bits() as u32)
            }
            _ => None,
        });
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
        .map_err(|error| format!("decoder: {error}"))?;
    let track_id = track.id;
    let mut sample_rate = None;
    let mut channels: Vec<Vec<f64>> = Vec::new();

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
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::DecodeError(_)) => continue,
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
                    .map(|sample| (sample as f64).clamp(-1.0, 1.0)),
            );
        }
    }

    let sample_rate = sample_rate.ok_or_else(|| "no audio packets decoded".to_string())?;
    Ok(DecodedPcm {
        sample_rate,
        channels,
        channel_mask,
    })
}

/// Decode RF64 PCM without materialising the encoded file in memory.
///
/// RF64 is the 64-bit extension of RIFF/WAVE. The `ds64` chunk carries the
/// sizes that cannot fit in the legacy 32-bit RIFF fields; the actual sample
/// payload is still ordinary little-endian WAVE PCM. Broadcast-WAVE (BWF) is
/// handled by the normal RIFF/WAVE reader because its `bext` chunk is metadata
/// that can be skipped by `hound`.
fn decode_rf64(path: &Path) -> Result<DecodedPcm, String> {
    use std::collections::HashMap;
    use std::io::{ErrorKind, Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path).map_err(|error| format!("open RF64: {error}"))?;
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
        let mut id = [0u8; 4];
        match file.read_exact(&mut id) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(format!("read RF64 chunk: {error}")),
        }
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
                let required = 28usize.saturating_add(table_len.saturating_mul(12));
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
                let bits_per_sample =
                    u16::from_le_bytes(body[14..16].try_into().expect("fixed bit depth"));
                let extensible = format_tag == 0xfffe;
                let (format_tag, bits_per_sample) = if extensible {
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
                    (format_tag, bits_per_sample)
                };
                if channels == 0 || sample_rate == 0 || block_align == 0 || bits_per_sample == 0 {
                    return Err("RF64 fmt chunk contains invalid audio parameters".into());
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
                    bits_per_sample,
                    channel_mask,
                ));
            }
            b"data" => {
                data_offset = Some(
                    file.stream_position()
                        .map_err(|error| format!("locate RF64 data: {error}"))?,
                );
                data_size = Some(declared_size);
                file.seek(SeekFrom::Current(i64::try_from(declared_size).map_err(
                    |_| "RF64 data chunk is too large to seek".to_string(),
                )?))
                .map_err(|error| format!("skip RF64 data: {error}"))?;
            }
            _ => {
                file.seek(SeekFrom::Current(
                    i64::try_from(declared_size)
                        .map_err(|_| "RF64 chunk is too large to seek".to_string())?,
                ))
                .map_err(|error| format!("skip RF64 chunk: {error}"))?;
            }
        }
        if declared_size & 1 != 0 {
            file.seek(SeekFrom::Current(1))
                .map_err(|error| format!("skip RF64 chunk padding: {error}"))?;
        }
        if format.is_some() && data_offset.is_some() {
            break;
        }
    }

    let (is_float, channel_count, sample_rate, block_align, bits_per_sample, channel_mask) =
        format.ok_or_else(|| "RF64 fmt chunk not found".to_string())?;
    let data_offset = data_offset.ok_or_else(|| "RF64 data chunk not found".to_string())?;
    let data_size = data_size.ok_or_else(|| "RF64 data size not found".to_string())?;
    if block_align == 0 || data_size % block_align as u64 != 0 {
        return Err("RF64 data size is not aligned to complete audio frames".into());
    }
    let frame_count = usize::try_from(data_size / block_align as u64)
        .map_err(|_| "RF64 audio is too large for this platform".to_string())?;
    if bits_per_sample > 64 || (is_float && !matches!(bits_per_sample, 32 | 64)) {
        return Err(format!(
            "RF64 sample depth {bits_per_sample} is unsupported"
        ));
    }
    let bytes_per_sample = (bits_per_sample as usize).div_ceil(8);
    if bytes_per_sample == 0 || bytes_per_sample.saturating_mul(channel_count) > block_align {
        return Err("RF64 fmt block alignment is invalid".into());
    }

    file.seek(SeekFrom::Start(data_offset))
        .map_err(|error| format!("seek RF64 data: {error}"))?;
    let mut channels = vec![Vec::with_capacity(frame_count); channel_count];
    let block_frames = (64 * 1024 / block_align).max(1).min(frame_count.max(1));
    let mut buffer = vec![0u8; block_frames * block_align];
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
                    match bits_per_sample {
                        32 => f32::from_le_bytes(sample.try_into().expect("float32 sample")) as f64,
                        64 => f64::from_le_bytes(sample.try_into().expect("float64 sample")),
                        _ => unreachable!("validated float depth"),
                    }
                } else if bits_per_sample == 8 {
                    (sample[0] as f64 - 128.0) / 128.0
                } else {
                    let mut raw = 0u64;
                    for (index, byte) in sample.iter().enumerate() {
                        raw |= u64::from(*byte) << (index * 8);
                    }
                    let sign_bit = 1u64 << (bits_per_sample - 1);
                    let signed = if raw & sign_bit != 0 {
                        (raw as i64) - (1i64 << bits_per_sample)
                    } else {
                        raw as i64
                    };
                    signed as f64 / (1u64 << (bits_per_sample - 1)) as f64
                };
                channels[channel].push(value.clamp(-1.0, 1.0));
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
}
