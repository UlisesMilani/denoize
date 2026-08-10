//! Raw ADTS AAC decoding.

use super::pcm::DecodedPcm;
use oxideav_aac::decode::{DecodedFrame, StreamDecoder};
use std::path::Path;

pub fn decode_adts(path: &Path) -> Result<DecodedPcm, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read AAC: {error}"))?;
    let payload = super::strip_id3v2_prefix(&bytes)
        .map_err(|error| format!("parse leading AAC ID3v2 tag: {error}"))?;
    let frames = StreamDecoder::new()
        .decode_all(payload)
        .map_err(|error| format!("decode ADTS AAC: {error}"))?;
    decoded_frames_to_pcm(&frames)
}

fn decoded_frames_to_pcm(frames: &[DecodedFrame]) -> Result<DecodedPcm, String> {
    for frame in frames {
        if frame.channels == 0 && !frame.pcm.is_empty() {
            return Err("ADTS AAC non-audio frame unexpectedly contains PCM samples".into());
        }
    }
    let first = frames
        .iter()
        .find(|frame| frame.channels > 0 && !frame.pcm.is_empty())
        .ok_or("ADTS AAC decode produced no samples")?;
    let sample_rate = first.sample_rate;
    let channel_count = first.channels;
    let mut frame_count = 0usize;

    for frame in frames {
        // The current decoder uses zero-channel empty frames for fill-only raw
        // data blocks. Also tolerate a future channel-bearing empty frame as a
        // priming/no-output marker, matching the MP4 AAC adapter.
        if frame.channels == 0 || frame.pcm.is_empty() {
            continue;
        }
        if frame.sample_rate != sample_rate || frame.channels != channel_count {
            return Err("ADTS AAC changes sample rate or channel count mid-stream".into());
        }
        if frame.pcm.len() % channel_count != 0 {
            return Err("ADTS AAC frame has incomplete interleaved PCM".into());
        }
        frame_count = frame_count
            .checked_add(frame.pcm.len() / channel_count)
            .ok_or("ADTS AAC decoded frame count overflows")?;
    }

    let mut channels = Vec::new();
    channels
        .try_reserve_exact(channel_count)
        .map_err(|error| format!("reserve ADTS AAC channels: {error}"))?;
    channels.resize_with(channel_count, Vec::new);
    for channel in &mut channels {
        channel
            .try_reserve_exact(frame_count)
            .map_err(|error| format!("reserve ADTS AAC PCM: {error}"))?;
    }

    for frame in frames
        .iter()
        .filter(|frame| frame.channels > 0 && !frame.pcm.is_empty())
    {
        for samples in frame.pcm.chunks_exact(channel_count) {
            for (channel, sample) in channels.iter_mut().zip(samples) {
                channel.push(*sample as f64 / 32768.0);
            }
        }
    }
    let channel_mask =
        crate::channel_layout::ChannelLayout::from_channel_count(channels.len()).mask();
    Ok(DecodedPcm {
        sample_rate,
        channels,
        channel_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_aac::raw_data_block::FrameAssembler;

    fn frame(sample_rate: u32, channels: usize, pcm: &[i16]) -> DecodedFrame {
        DecodedFrame {
            pcm: pcm.to_vec(),
            channels,
            sample_rate,
        }
    }

    fn adts_frame(payload: &[u8]) -> Vec<u8> {
        const HEADER_LEN: usize = 7;
        let frame_len = HEADER_LEN + payload.len();
        assert!(frame_len <= 0x1fff);
        let profile = 1u8; // AAC LC is encoded as audioObjectType - 1.
        let frequency_index = 4u8; // 44.1 kHz.
        let channel_configuration = 2u8;
        let fullness = 0x7ffu16;
        let mut output = vec![
            0xff,
            0xf1,
            (profile << 6) | (frequency_index << 2) | (channel_configuration >> 2),
            ((channel_configuration & 3) << 6) | (((frame_len >> 11) & 3) as u8),
            ((frame_len >> 3) & 0xff) as u8,
            (((frame_len & 7) as u8) << 5) | ((fullness >> 6) as u8),
            ((fullness & 0x3f) << 2) as u8,
        ];
        output.extend_from_slice(payload);
        output
    }

    #[test]
    fn skips_non_audio_frames_without_changing_audio_geometry() {
        let frames = [
            frame(22_050, 0, &[]),
            frame(44_100, 1, &[8_192, -8_192]),
            frame(96_000, 0, &[]),
            frame(44_100, 1, &[16_384]),
        ];

        let decoded = decoded_frames_to_pcm(&frames).expect("collect AAC audio frames");
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.channels, vec![vec![0.25, -0.25, 0.5]]);
    }

    #[test]
    fn rejects_invalid_decoded_frame_geometry() {
        let non_audio_with_pcm = [frame(44_100, 0, &[1])];
        assert!(decoded_frames_to_pcm(&non_audio_with_pcm)
            .unwrap_err()
            .contains("non-audio frame"));

        let empty_audio_marker = [frame(22_050, 1, &[]), frame(44_100, 1, &[8_192])];
        let decoded =
            decoded_frames_to_pcm(&empty_audio_marker).expect("empty audio marker must be skipped");
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.channels, vec![vec![0.25]]);

        let incomplete_stereo = [frame(44_100, 2, &[1, 2, 3])];
        assert!(decoded_frames_to_pcm(&incomplete_stereo)
            .unwrap_err()
            .contains("incomplete interleaved PCM"));
    }

    #[test]
    fn fill_only_adts_returns_an_error_instead_of_panicking() {
        let mut assembler = FrameAssembler::new();
        assembler.push_fill(&[]).expect("write AAC fill element");
        let bytes = adts_frame(&assembler.push_end());
        let file = tempfile::NamedTempFile::new().expect("create AAC fixture");
        std::fs::write(file.path(), bytes).expect("write AAC fixture");

        let result = std::panic::catch_unwind(|| decode_adts(file.path()));
        let error = result
            .expect("fill-only AAC must not panic")
            .expect_err("fill-only AAC has no output audio");
        assert!(error.contains("decode produced no samples"), "{error}");
    }

    #[cfg(feature = "m4a-encode")]
    #[test]
    fn leading_fill_frame_preserves_runtime_encoded_pcm() {
        use oxideav_aac_encoder::encoder::{EncoderConfig, StreamEncoder, FRAME_LEN};

        let mut encoder = StreamEncoder::new(EncoderConfig {
            sample_rate: 44_100,
            channels: 1,
            bitrate: 96_000,
        })
        .expect("create AAC encoder");
        let baseline_bytes = encoder
            .encode_all(&vec![0i16; FRAME_LEN])
            .expect("encode AAC fixture");

        let mut assembler = FrameAssembler::new();
        assembler.push_fill(&[]).expect("write AAC fill element");
        let mut prefixed_bytes = adts_frame(&assembler.push_end());
        prefixed_bytes.extend_from_slice(&baseline_bytes);

        let baseline_file = tempfile::NamedTempFile::new().expect("create baseline AAC fixture");
        let prefixed_file = tempfile::NamedTempFile::new().expect("create prefixed AAC fixture");
        std::fs::write(baseline_file.path(), baseline_bytes).expect("write baseline AAC fixture");
        std::fs::write(prefixed_file.path(), prefixed_bytes).expect("write prefixed AAC fixture");

        let baseline = decode_adts(baseline_file.path()).expect("decode baseline AAC");
        let prefixed = decode_adts(prefixed_file.path()).expect("decode fill-prefixed AAC");
        assert_eq!(prefixed.sample_rate, baseline.sample_rate);
        assert_eq!(prefixed.channel_mask, baseline.channel_mask);
        assert_eq!(prefixed.channels, baseline.channels);
    }
}
