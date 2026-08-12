//! Bounded MP3 compatibility decoder.
//!
//! Symphonia remains the primary MP3 decoder because it applies Xing/Info +
//! LAME delay and padding. This module is deliberately narrower: it streams a
//! fixed-size encoded input window through `nanomp3` for raw MPEG streams that
//! trigger Symphonia's known bit-reservoir compatibility error.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[cfg(test)]
use nanomp3::Channels;
use nanomp3::{Decoder, FrameInfo, MAX_SAMPLES_PER_FRAME};

use super::budget::DecodeBudget;
use super::pcm::DecodedPcm;
use super::DecodeLimits;

/// Refill below this threshold so the bounded carry normally holds several
/// complete MPEG frames before each exact-frame decode.
const MIN_DECODE_WINDOW_BYTES: usize = 16 * 1024;
/// The encoded input working set stays fixed regardless of the file size.
const INPUT_BUFFER_BYTES: usize = 32 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum TimingMetadata {
    /// A valid first MPEG frame was inspected and has no recognised timing tag.
    Absent {
        sample_rate: u32,
        channel_count: usize,
    },
    /// Xing, Info, or VBRI was found in the first MPEG frame.
    Present,
    /// The first MPEG frame could not be classified within the bounded window.
    Undetermined(String),
}

/// Conservatively decide whether a raw-frame fallback may discard no timing
/// metadata. Unknown inputs are intentionally not eligible.
pub(super) fn inspect_timing_metadata(mut input: File, limits: DecodeLimits) -> TimingMetadata {
    match inspect_timing_metadata_inner(&mut input, DecodeBudget::new(limits)) {
        Ok(result) => result,
        Err(error) => TimingMetadata::Undetermined(error),
    }
}

fn inspect_timing_metadata_inner(
    input: &mut File,
    budget: DecodeBudget,
) -> Result<TimingMetadata, String> {
    seek_past_id3v2(input)?;

    budget.check_peak(0, INPUT_BUFFER_BYTES as u64, "MP3 fallback timing probe")?;
    let mut encoded = vec![0u8; INPUT_BUFFER_BYTES];
    let mut buffered = 0usize;
    while buffered < encoded.len() {
        let read = input
            .read(&mut encoded[buffered..])
            .map_err(|error| format!("read MP3 fallback probe: {error}"))?;
        if read == 0 {
            break;
        }
        buffered += read;
    }
    if buffered == 0 {
        return Ok(TimingMetadata::Undetermined(
            "no MPEG payload after ID3v2 metadata".into(),
        ));
    }

    let Some(header) = encoded[..buffered].get(..4).and_then(parse_frame_header) else {
        return Ok(TimingMetadata::Undetermined(format!(
            "no MPEG Layer III header at the start of the bounded payload"
        )));
    };
    if header.frame_bytes > buffered {
        return Ok(TimingMetadata::Undetermined(
            "the first MPEG frame is incomplete within the bounded payload".into(),
        ));
    }
    if buffered > header.frame_bytes {
        let Some(next) = encoded[header.frame_bytes..]
            .get(..4)
            .and_then(parse_frame_header)
        else {
            return Ok(TimingMetadata::Undetermined(
                "the first MPEG frame is not followed by a contiguous frame".into(),
            ));
        };
        if next.sample_rate != header.sample_rate || next.version != header.version {
            return Ok(TimingMetadata::Undetermined(
                "the first two MPEG frame headers are incompatible".into(),
            ));
        }
    }

    let first_frame = &encoded[..header.frame_bytes];
    if contains_timing_tag(first_frame) {
        Ok(TimingMetadata::Present)
    } else {
        Ok(TimingMetadata::Absent {
            sample_rate: header.sample_rate,
            channel_count: header.channel_count,
        })
    }
}

fn contains_timing_tag(bytes: &[u8]) -> bool {
    [b"Xing".as_slice(), b"Info".as_slice(), b"VBRI".as_slice()]
        .into_iter()
        .any(|tag| bytes.windows(tag.len()).any(|window| window == tag))
}

/// Decode a raw MPEG stream without materialising the encoded file in memory.
///
/// The decoded PCM is necessarily retained because [`DecodedPcm`] is an owned
/// whole-stream value, but encoded input buffering is capped at 32 KiB.
pub(super) fn decode_mp3_file_compatibility(
    mut input: File,
    limits: DecodeLimits,
) -> Result<DecodedPcm, String> {
    seek_past_id3v2(&mut input)?;
    decode_mp3_stream(&mut input, limits)
}

fn decode_mp3_stream<R: Read>(input: &mut R, limits: DecodeLimits) -> Result<DecodedPcm, String> {
    let budget = DecodeBudget::new(limits);
    let scratch_bytes = mp3_scratch_bytes()?;
    budget.check_peak(0, scratch_bytes, "MP3 fallback buffers")?;
    let mut decoder = Decoder::new();
    let mut pcm = vec![0.0f32; MAX_SAMPLES_PER_FRAME];
    let mut encoded = vec![0u8; INPUT_BUFFER_BYTES];
    let mut start = 0usize;
    let mut end = 0usize;
    let mut eof = false;

    let mut channels: Vec<Vec<f64>> = Vec::new();
    let mut sample_rate = None;
    let mut channel_count = None;

    loop {
        while !eof && end - start < MIN_DECODE_WINDOW_BYTES {
            if start == end {
                start = 0;
                end = 0;
            } else if end == encoded.len() {
                // Compact only when the write end has no space. Decoded
                // frames otherwise advance `start` without moving bytes, so
                // long streams remain amortized linear in encoded size.
                encoded.copy_within(start..end, 0);
                end -= start;
                start = 0;
            }
            let read = input
                .read(&mut encoded[end..])
                .map_err(|error| format!("read MP3 fallback: {error}"))?;
            if read == 0 {
                eof = true;
            } else {
                end += read;
            }
        }

        if start == end {
            break;
        }

        let buffered = &encoded[start..end];
        if eof && recognized_trailing_metadata(buffered) {
            break;
        }

        let header = buffered.get(..4).and_then(parse_frame_header).ok_or_else(|| {
                "MP3 fallback only accepts contiguous MPEG Layer III frames (or recognised trailing metadata)"
                    .to_string()
            })?;
        if header.frame_bytes > buffered.len() {
            if eof {
                return Err(format!(
                    "truncated final MP3 frame: expected {} bytes, found {}",
                    header.frame_bytes,
                    buffered.len()
                ));
            }
            return Err(format!(
                "MP3 frame exceeds the {INPUT_BUFFER_BYTES}-byte fallback input window"
            ));
        }

        // Re-check on the same cloned session handle used for fallback decode.
        // This keeps format/timing eligibility tied to the selected bytes and
        // prevents Xing/Info/VBRI delay metadata from reaching the raw path.
        if channels.is_empty() && contains_timing_tag(&buffered[..header.frame_bytes]) {
            return Err("MP3 fallback first frame contains Xing/Info/VBRI timing metadata".into());
        }

        budget.check_planar_capacities(&channels, scratch_bytes, "MP3 fallback decode")?;
        let (consumed, info) = decoder.decode(&buffered[..header.frame_bytes], &mut pcm);
        let info = info.ok_or_else(|| {
            "MP3 fallback could not decode a complete contiguous MPEG frame".to_string()
        })?;
        if consumed != header.frame_bytes {
            return Err(format!(
                "MP3 fallback consumed {consumed} bytes for a {}-byte MPEG frame",
                header.frame_bytes
            ));
        }
        if info.sample_rate != header.sample_rate
            || usize::from(info.channels.num()) != header.channel_count
        {
            return Err("MP3 fallback decoder disagrees with the MPEG frame header".into());
        }
        append_frame(
            &info,
            &pcm,
            &mut channels,
            &mut sample_rate,
            &mut channel_count,
            budget,
            scratch_bytes,
        )?;

        start += header.frame_bytes;
    }

    let sample_rate = sample_rate.ok_or_else(|| "no valid MP3 frames found".to_string())?;
    let channel_count = channel_count.expect("sample rate and channel count are set together");
    if channels.len() != channel_count || channels.iter().any(|channel| channel.is_empty()) {
        return Err("MP3 fallback produced an incomplete channel set".into());
    }

    let channel_mask =
        crate::channel_layout::ChannelLayout::from_channel_count(channel_count).mask();
    Ok(DecodedPcm {
        sample_rate,
        channels,
        channel_mask,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MpegFrameHeader {
    frame_bytes: usize,
    sample_rate: u32,
    channel_count: usize,
    version: u8,
}

fn parse_frame_header(bytes: &[u8]) -> Option<MpegFrameHeader> {
    let header = u32::from_be_bytes(bytes.get(..4)?.try_into().ok()?);
    if header & 0xffe0_0000 != 0xffe0_0000 {
        return None;
    }
    let version = ((header >> 19) & 0x3) as u8;
    let layer = ((header >> 17) & 0x3) as u8;
    let bitrate_index = ((header >> 12) & 0xf) as usize;
    let sample_rate_index = ((header >> 10) & 0x3) as usize;
    if version == 1
        || layer != 1
        || bitrate_index == 0
        || bitrate_index == 15
        || sample_rate_index == 3
    {
        return None;
    }

    const MPEG1_LAYER3_KBPS: [u32; 16] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ];
    const MPEG2_LAYER3_KBPS: [u32; 16] = [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
    ];
    const MPEG1_SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 32_000];

    let sample_rate = match version {
        3 => MPEG1_SAMPLE_RATES[sample_rate_index],
        2 => MPEG1_SAMPLE_RATES[sample_rate_index] / 2,
        0 => MPEG1_SAMPLE_RATES[sample_rate_index] / 4,
        _ => return None,
    };
    let bitrate_kbps = if version == 3 {
        MPEG1_LAYER3_KBPS[bitrate_index]
    } else {
        MPEG2_LAYER3_KBPS[bitrate_index]
    };
    let coefficient = if version == 3 { 144_000 } else { 72_000 };
    let padding = usize::from((header & 0x200) != 0);
    let frame_bytes = usize::try_from(coefficient * bitrate_kbps / sample_rate)
        .ok()?
        .checked_add(padding)?;
    if frame_bytes < 4 || frame_bytes > INPUT_BUFFER_BYTES {
        return None;
    }

    Some(MpegFrameHeader {
        frame_bytes,
        sample_rate,
        channel_count: if (header >> 6) & 0x3 == 3 { 1 } else { 2 },
        version,
    })
}

fn recognized_trailing_metadata(bytes: &[u8]) -> bool {
    let without_id3v1 =
        if bytes.len() >= 128 && &bytes[bytes.len() - 128..bytes.len() - 125] == b"TAG" {
            &bytes[..bytes.len() - 128]
        } else {
            bytes
        };
    if without_id3v1.is_empty() {
        return true;
    }
    if without_id3v1.len() < 32 {
        return false;
    }
    let footer = &without_id3v1[without_id3v1.len() - 32..];
    if &footer[..8] != b"APETAGEX" {
        return false;
    }
    let tag_size = u32::from_le_bytes(footer[12..16].try_into().unwrap()) as usize;
    tag_size >= 32 && tag_size == without_id3v1.len()
}

fn append_frame(
    info: &FrameInfo,
    pcm: &[f32],
    channels: &mut Vec<Vec<f64>>,
    sample_rate: &mut Option<u32>,
    channel_count: &mut Option<usize>,
    budget: DecodeBudget,
    temporary_bytes: u64,
) -> Result<(), String> {
    let current_channel_count = usize::from(info.channels.num());
    let interleaved_len = info
        .samples_produced
        .checked_mul(current_channel_count)
        .ok_or_else(|| "MP3 fallback sample count overflows".to_string())?;
    let interleaved = pcm
        .get(..interleaved_len)
        .ok_or_else(|| "MP3 fallback produced more samples than its PCM buffer".to_string())?;

    match (*sample_rate, *channel_count) {
        (None, None) => {
            budget.check_planar_frames(
                current_channel_count,
                info.samples_produced,
                temporary_bytes,
                "MP3 fallback decode",
            )?;
            *sample_rate = Some(info.sample_rate);
            *channel_count = Some(current_channel_count);
            channels
                .try_reserve_exact(current_channel_count)
                .map_err(|error| format!("reserve MP3 fallback channels: {error}"))?;
            channels.resize_with(current_channel_count, Vec::new);
        }
        (Some(expected_rate), Some(expected_channels)) => {
            if info.sample_rate != expected_rate {
                return Err(format!(
                    "MP3 fallback sample rate changed from {expected_rate} to {}",
                    info.sample_rate
                ));
            }
            if current_channel_count != expected_channels {
                return Err(format!(
                    "MP3 fallback channel count changed from {expected_channels} to {current_channel_count}"
                ));
            }
        }
        _ => return Err("MP3 fallback decoder state is inconsistent".into()),
    }

    budget.reserve_planar_additional(
        channels,
        info.samples_produced,
        temporary_bytes,
        "MP3 fallback decode",
    )?;
    for frame in interleaved.chunks_exact(current_channel_count) {
        for (destination, &sample) in channels.iter_mut().zip(frame) {
            destination.push(crate::audio::sanitize_sample(f64::from(sample)));
        }
    }
    Ok(())
}

fn mp3_scratch_bytes() -> Result<u64, String> {
    let pcm_bytes = u64::try_from(MAX_SAMPLES_PER_FRAME)
        .ok()
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| "MP3 fallback PCM scratch byte count overflows".to_string())?;
    (INPUT_BUFFER_BYTES as u64)
        .checked_add(pcm_bytes)
        .ok_or_else(|| "MP3 fallback scratch byte count overflows".to_string())
}

fn seek_past_id3v2(input: &mut File) -> Result<(), String> {
    let file_len = input
        .metadata()
        .map_err(|error| format!("stat MP3 fallback input: {error}"))?
        .len();
    input
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind MP3 fallback input: {error}"))?;
    let mut header = [0u8; 10];
    let mut read = 0usize;
    while read < header.len() {
        let count = input
            .read(&mut header[read..])
            .map_err(|error| format!("read MP3 ID3v2 header: {error}"))?;
        if count == 0 {
            break;
        }
        read += count;
    }

    let payload_offset = super::id3v2_payload_offset(&header[..read], file_len)?.unwrap_or(0);
    input
        .seek(SeekFrom::Start(payload_offset))
        .map_err(|error| format!("seek past MP3 ID3v2 tag: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_tag_scan_is_conservative() {
        assert!(contains_timing_tag(b"frame-prefix-Xing-frame-data"));
        assert!(contains_timing_tag(b"frame-prefix-Info-frame-data"));
        assert!(contains_timing_tag(b"frame-prefix-VBRI-frame-data"));
        assert!(!contains_timing_tag(b"ordinary MPEG frame bytes"));
    }

    #[test]
    fn parses_mpeg1_and_mpeg2_layer3_headers() {
        let mpeg1 = parse_frame_header(&[0xff, 0xfb, 0x90, 0x64]).unwrap();
        assert_eq!(mpeg1.sample_rate, 44_100);
        assert_eq!(mpeg1.channel_count, 2);
        assert_eq!(mpeg1.frame_bytes, 417);

        let mpeg2 = parse_frame_header(&[0xff, 0xf3, 0x80, 0xc0]).unwrap();
        assert_eq!(mpeg2.sample_rate, 22_050);
        assert_eq!(mpeg2.channel_count, 1);
        assert_eq!(mpeg2.frame_bytes, 208);
    }

    #[test]
    fn recognised_trailing_tags_are_exact() {
        let mut id3v1 = vec![0u8; 128];
        id3v1[..3].copy_from_slice(b"TAG");
        assert!(recognized_trailing_metadata(&id3v1));

        let mut ape = vec![0u8; 32];
        ape[..8].copy_from_slice(b"APETAGEX");
        ape[12..16].copy_from_slice(&32u32.to_le_bytes());
        assert!(recognized_trailing_metadata(&ape));
        ape.push(0);
        assert!(!recognized_trailing_metadata(&ape));
    }

    #[test]
    fn stream_decoder_rejects_empty_and_large_garbage_inputs() {
        assert!(decode_mp3_stream(
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            DecodeLimits::default(),
        )
        .is_err());
        let mut garbage = std::io::Cursor::new(vec![0u8; INPUT_BUFFER_BYTES * 2 + 17]);
        assert!(decode_mp3_stream(&mut garbage, DecodeLimits::default()).is_err());
    }

    #[test]
    fn fixed_mp3_scratch_budget_has_an_exact_boundary() {
        let scratch = mp3_scratch_bytes().unwrap();
        let exact_error = decode_mp3_stream(
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            DecodeLimits::default().with_max_working_set_bytes(Some(scratch)),
        )
        .unwrap_err();
        assert!(exact_error.contains("no valid MP3 frames"), "{exact_error}");

        let error = decode_mp3_stream(
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            DecodeLimits::default().with_max_working_set_bytes(Some(scratch - 1)),
        )
        .unwrap_err();
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn decoded_pcm_budget_is_checked_across_mp3_frames_before_growth() {
        const MIB: u64 = 1024 * 1024;
        let budget =
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(MIB)));
        let scratch = mp3_scratch_bytes().unwrap();
        let info = FrameInfo {
            samples_produced: 1_152,
            channels: Channels::Stereo,
            sample_rate: 44_100,
            bitrate: 128,
        };
        let pcm = vec![0.0; MAX_SAMPLES_PER_FRAME];
        let mut channels = Vec::new();
        let mut sample_rate = None;
        let mut channel_count = None;
        for _ in 0..18 {
            append_frame(
                &info,
                &pcm,
                &mut channels,
                &mut sample_rate,
                &mut channel_count,
                budget,
                scratch,
            )
            .expect("PCM below the one-MiB normal-work boundary");
        }
        let before = channels[0].len();
        let error = append_frame(
            &info,
            &pcm,
            &mut channels,
            &mut sample_rate,
            &mut channel_count,
            budget,
            scratch,
        )
        .expect_err("the next complete MP3 frame must cross the cap");
        assert!(error.contains("working-set limit"), "{error}");
        assert!(channels.iter().all(|channel| channel.len() == before));
    }

    #[test]
    fn spare_pcm_capacity_is_combined_with_mp3_decoder_scratch() {
        const MIB: u64 = 1024 * 1024;
        let mut channels = vec![Vec::with_capacity(32_768)];
        channels[0].resize(1_024, 0.0);
        let logical_bytes = u64::try_from(channels[0].len() * std::mem::size_of::<f64>()).unwrap()
            + std::mem::size_of::<Vec<f64>>() as u64;
        let scratch = MIB - logical_bytes;
        let budget =
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(MIB)));

        budget
            .check_planar_frames(1, channels[0].len(), scratch, "logical MP3")
            .expect("logical frames fit the crafted cap");
        let error = budget
            .check_planar_capacities(&channels, scratch, "MP3 fallback decode")
            .expect_err("retained spare capacity plus decoder scratch must be rejected");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn frame_append_rejects_rate_and_channel_changes() {
        let mut channels = Vec::new();
        let mut sample_rate = None;
        let mut channel_count = None;
        let pcm = vec![0.0; MAX_SAMPLES_PER_FRAME];
        append_frame(
            &FrameInfo {
                samples_produced: 1,
                channels: Channels::Stereo,
                sample_rate: 44_100,
                bitrate: 128,
            },
            &pcm,
            &mut channels,
            &mut sample_rate,
            &mut channel_count,
            DecodeBudget::new(DecodeLimits::default()),
            0,
        )
        .unwrap();
        let error = append_frame(
            &FrameInfo {
                samples_produced: 1,
                channels: Channels::Mono,
                sample_rate: 48_000,
                bitrate: 128,
            },
            &pcm,
            &mut channels,
            &mut sample_rate,
            &mut channel_count,
            DecodeBudget::new(DecodeLimits::default()),
            0,
        )
        .unwrap_err();
        assert!(error.contains("sample rate changed"));
    }

    #[test]
    fn compatibility_stream_decodes_the_last_frame_before_id3v1_and_rejects_truncation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("short-stereo.mp3");
        let audio = crate::Audio {
            sample_rate: 44_100,
            channels: vec![vec![0.25], vec![-0.25]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: crate::ChannelLayout::Stereo.mask(),
        };
        crate::write_audio(&path, &audio, crate::EncodeOptions::default()).unwrap();
        let encoded = std::fs::read(path).unwrap();

        let decoded =
            decode_mp3_stream(&mut std::io::Cursor::new(&encoded), DecodeLimits::default())
                .unwrap();
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.channels.len(), 2);
        assert_eq!(decoded.frames(), 2_304);

        let mut with_id3v1 = encoded.clone();
        let mut id3v1 = [0u8; 128];
        id3v1[..3].copy_from_slice(b"TAG");
        with_id3v1.extend_from_slice(&id3v1);
        let tagged = decode_mp3_stream(
            &mut std::io::Cursor::new(with_id3v1),
            DecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(tagged.frames(), decoded.frames());

        let truncated = &encoded[..encoded.len() - 1];
        let error = decode_mp3_stream(
            &mut std::io::Cursor::new(truncated),
            DecodeLimits::default(),
        )
        .unwrap_err();
        assert!(error.contains("truncated final MP3 frame"));

        let mut timing_tagged = encoded;
        timing_tagged[40..44].copy_from_slice(b"Xing");
        let error = decode_mp3_stream(
            &mut std::io::Cursor::new(timing_tagged),
            DecodeLimits::default(),
        )
        .unwrap_err();
        assert!(error.contains("timing metadata"));
    }

    #[test]
    fn compatibility_stream_crosses_multiple_bounded_input_windows() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("long-stereo.mp3");
        let frames = 100_000usize;
        let audio = crate::Audio {
            sample_rate: 44_100,
            channels: vec![vec![0.125; frames], vec![-0.125; frames]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: crate::ChannelLayout::Stereo.mask(),
        };
        crate::write_audio(&path, &audio, crate::EncodeOptions::default()).unwrap();
        let encoded = std::fs::read(path).unwrap();
        assert!(encoded.len() > INPUT_BUFFER_BYTES);

        let decoded =
            decode_mp3_stream(&mut std::io::Cursor::new(encoded), DecodeLimits::default()).unwrap();
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.channels.len(), 2);
        assert!(decoded.frames() >= frames);
    }
}
