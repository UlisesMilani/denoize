//! Raw ADTS AAC decoding.

use super::pcm::DecodedPcm;
use super::{budget::DecodeBudget, DecodeLimits};
use oxideav_aac::adts::{AdtsHeader, ADTS_HEADER_BYTES_NO_CRC};
use oxideav_aac::decode::{DecodedFrame, StreamDecoder, FRAME_LEN};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

// The decoder's element-slot maps and transform/SBR state are bounded by AAC
// syntax, not the advertised output geometry. Match the conservative native
// M4A allowance so a capped raw ADTS decode fails before entering oxideav-aac
// if that bounded third-party working set cannot fit.
const AAC_DECODER_INTERNAL_BYTES: u64 = 128 * 1024 * 1024;
// oxideav-aac currently retains every decoded element until it interleaves
// the frame, including non-conforming repeated tags. The smallest complete
// output element accepted by it is a 29-bit SCE (a CPE is at least 43 bits).
// Even charging 56 KiB per occurrence for two core and SBR f64 outputs,
// returned i16 PCM, spectra, and descriptors gives <16 KiB/input byte.
// 64 KiB/input byte leaves over 4x headroom. Persistent syntax-bounded maps
// remain covered by the fixed allowance above.
const AAC_DECODER_BYTES_PER_PAYLOAD_BYTE: u64 = 64 * 1024;

/// Decode raw ADTS AAC without retaining the encoded file or every decoded
/// `i16` frame in memory. An ADTS frame is at most 8,191 bytes by construction,
/// so the encoded working set remains independent of input length.
pub(super) fn decode_adts(mut input: File, limits: DecodeLimits) -> Result<DecodedPcm, String> {
    seek_past_id3v2(&mut input)?;

    let budget = DecodeBudget::new(limits);
    budget.check_peak(0, AAC_DECODER_INTERNAL_BYTES, "ADTS AAC decoder state")?;
    let mut decoder = StreamDecoder::new();
    let mut collector = DecodedFrameCollector::default();
    loop {
        let mut fixed_header = [0u8; ADTS_HEADER_BYTES_NO_CRC];
        let fixed_bytes = read_up_to(&mut input, &mut fixed_header)
            .map_err(|error| format!("read ADTS AAC header: {error}"))?;
        if fixed_bytes < fixed_header.len() {
            // Match oxideav-aac's whole-buffer adapter: fewer than seven
            // trailing bytes do not form another ADTS frame and are ignored.
            break;
        }

        let protection_absent = fixed_header[1] & 1 != 0;
        let mut header_bytes = [0u8; 9];
        header_bytes[..fixed_header.len()].copy_from_slice(&fixed_header);
        let header_len = if protection_absent {
            fixed_header.len()
        } else {
            input
                .read_exact(&mut header_bytes[fixed_header.len()..])
                .map_err(|error| format!("decode ADTS AAC: {error}"))?;
            header_bytes.len()
        };
        let (header, payload_offset) = AdtsHeader::parse(&header_bytes[..header_len])
            .map_err(|error| format!("decode ADTS AAC: {error}"))?;
        let frame_len = usize::from(header.aac_frame_length);
        let payload_len = frame_len
            .checked_sub(payload_offset)
            .ok_or("decode ADTS AAC: frame length is shorter than its header")?;
        let payload_bytes = u64::try_from(payload_len)
            .map_err(|_| "ADTS AAC frame size does not fit in u64".to_string())?;
        // `DecodedFrame::pcm` is allocated inside oxideav-aac. Bound its
        // largest syntax-representable frame before entering that dependency;
        // the actual returned size is accounted again before planar growth.
        let frame_scratch = maximum_decoded_frame_bytes(&header)?;
        let decoder_bytes = aac_decoder_working_bytes(payload_bytes)?;
        let temporary_bytes = payload_bytes
            .checked_add(frame_scratch)
            .and_then(|bytes| bytes.checked_add(decoder_bytes))
            .ok_or("ADTS AAC temporary byte count overflows")?;
        budget.check_planar_frames(
            collector.channels.len(),
            collector.frame_count,
            temporary_bytes,
            "ADTS AAC decode",
        )?;
        budget.check_planar_capacities(&collector.channels, temporary_bytes, "ADTS AAC decode")?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(payload_len)
            .map_err(|error| format!("reserve ADTS AAC frame: {error}"))?;
        payload.resize(payload_len, 0);
        input
            .read_exact(&mut payload)
            .map_err(|error| format!("decode ADTS AAC: {error}"))?;

        let frame = decoder
            .decode_frame(&header, &payload)
            .map_err(|error| format!("decode ADTS AAC: {error}"))?;
        let returned_frame_bytes = allocation_bytes::<i16>(frame.pcm.capacity(), "ADTS AAC frame")?;
        collector.push(
            &frame,
            budget,
            payload_bytes
                .checked_add(returned_frame_bytes)
                .and_then(|bytes| bytes.checked_add(decoder_bytes))
                .ok_or("ADTS AAC temporary byte count overflows")?,
        )?;
    }
    collector.finish()
}

fn aac_decoder_working_bytes(payload_bytes: u64) -> Result<u64, String> {
    payload_bytes
        .checked_mul(AAC_DECODER_BYTES_PER_PAYLOAD_BYTE)
        .and_then(|bytes| bytes.checked_add(AAC_DECODER_INTERNAL_BYTES))
        .ok_or_else(|| "ADTS AAC decoder byte count overflows".to_string())
}

#[cfg(test)]
fn decoded_frames_to_pcm(frames: &[DecodedFrame]) -> Result<DecodedPcm, String> {
    let mut collector = DecodedFrameCollector::default();
    let budget = DecodeBudget::new(DecodeLimits::default());
    for frame in frames {
        let temporary_bytes = allocation_bytes::<i16>(frame.pcm.len(), "ADTS AAC frame")?;
        collector.push(frame, budget, temporary_bytes)?;
    }
    collector.finish()
}

#[derive(Default)]
struct DecodedFrameCollector {
    sample_rate: Option<u32>,
    channel_count: Option<usize>,
    channels: Vec<Vec<f64>>,
    frame_count: usize,
}

impl DecodedFrameCollector {
    fn push(
        &mut self,
        frame: &DecodedFrame,
        budget: DecodeBudget,
        temporary_bytes: u64,
    ) -> Result<(), String> {
        if frame.channels == 0 && !frame.pcm.is_empty() {
            return Err("ADTS AAC non-audio frame unexpectedly contains PCM samples".into());
        }
        // The current decoder uses zero-channel empty frames for fill-only raw
        // data blocks. Also tolerate a future channel-bearing empty frame as a
        // priming/no-output marker, matching the MP4 AAC adapter.
        if frame.channels == 0 || frame.pcm.is_empty() {
            return Ok(());
        }

        let channel_count = frame.channels;
        if frame.pcm.len() % channel_count != 0 {
            return Err("ADTS AAC frame has incomplete interleaved PCM".into());
        }
        let frame_count = frame.pcm.len() / channel_count;
        let next_total = self
            .frame_count
            .checked_add(frame_count)
            .ok_or("ADTS AAC decoded frame count overflows")?;

        match (self.sample_rate, self.channel_count) {
            (None, None) => {
                budget.check_planar_frames(
                    frame.channels,
                    next_total,
                    temporary_bytes,
                    "ADTS AAC decode",
                )?;
                self.sample_rate = Some(frame.sample_rate);
                self.channel_count = Some(frame.channels);
                self.channels
                    .try_reserve_exact(frame.channels)
                    .map_err(|error| format!("reserve ADTS AAC channels: {error}"))?;
                self.channels.resize_with(frame.channels, Vec::new);
            }
            (Some(sample_rate), Some(channel_count)) => {
                if frame.sample_rate != sample_rate || frame.channels != channel_count {
                    return Err("ADTS AAC changes sample rate or channel count mid-stream".into());
                }
            }
            _ => return Err("ADTS AAC decoder state is inconsistent".into()),
        }

        budget.reserve_planar_additional(
            &mut self.channels,
            frame_count,
            temporary_bytes,
            "ADTS AAC decode",
        )?;
        for samples in frame.pcm.chunks_exact(channel_count) {
            for (channel, sample) in self.channels.iter_mut().zip(samples) {
                channel.push(*sample as f64 / 32768.0);
            }
        }
        self.frame_count = next_total;
        Ok(())
    }

    fn finish(self) -> Result<DecodedPcm, String> {
        let sample_rate = self
            .sample_rate
            .ok_or("ADTS AAC decode produced no samples")?;
        let channel_count = self
            .channel_count
            .expect("ADTS AAC sample rate and channel count are set together");
        if self.frame_count == 0
            || self.channels.len() != channel_count
            || self
                .channels
                .iter()
                .any(|channel| channel.len() != self.frame_count)
        {
            return Err("ADTS AAC decode produced an incomplete channel set".into());
        }
        let channel_mask =
            crate::channel_layout::ChannelLayout::from_channel_count(channel_count).mask();
        Ok(DecodedPcm {
            sample_rate,
            channels: self.channels,
            channel_mask,
        })
    }
}

fn maximum_decoded_frame_bytes(header: &AdtsHeader) -> Result<u64, String> {
    // A PCE can signal at most 15 front, 15 side, and 15 back channel
    // elements (two channels each) plus three LFE elements.
    const MAX_PCE_CHANNELS: usize = (15 + 15 + 15) * 2 + 3;
    let channels_per_block = match header.channel_configuration {
        0 => MAX_PCE_CHANNELS,
        1..=6 => usize::from(header.channel_configuration),
        7 => 8,
        _ => unreachable!("ADTS channel_configuration is a three-bit field"),
    };
    let channels = channels_per_block
        .checked_mul(usize::from(header.number_of_raw_data_blocks_in_frame))
        .ok_or("ADTS AAC maximum channel count overflows")?;
    let frames = FRAME_LEN
        .checked_mul(2)
        .ok_or("ADTS AAC maximum frame count overflows")?;
    let samples = channels
        .checked_mul(frames)
        .ok_or("ADTS AAC maximum sample count overflows")?;
    allocation_bytes::<i16>(samples, "ADTS AAC maximum decoded frame")
}

fn allocation_bytes<T>(len: usize, context: &str) -> Result<u64, String> {
    u64::try_from(len)
        .ok()
        .and_then(|len| len.checked_mul(std::mem::size_of::<T>() as u64))
        .ok_or_else(|| format!("{context} byte count overflows"))
}

fn read_up_to(input: &mut File, output: &mut [u8]) -> std::io::Result<usize> {
    let mut read = 0usize;
    while read < output.len() {
        let count = input.read(&mut output[read..])?;
        if count == 0 {
            break;
        }
        read += count;
    }
    Ok(read)
}

fn seek_past_id3v2(input: &mut File) -> Result<(), String> {
    let file_len = input
        .metadata()
        .map_err(|error| format!("stat AAC input: {error}"))?
        .len();
    input
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind AAC input: {error}"))?;
    let mut header = [0u8; 10];
    let read = read_up_to(input, &mut header)
        .map_err(|error| format!("read AAC ID3v2 header: {error}"))?;
    let payload_offset = super::id3v2_payload_offset(&header[..read], file_len)
        .map_err(|error| format!("parse leading AAC ID3v2 tag: {error}"))?
        .unwrap_or(0);
    input
        .seek(SeekFrom::Start(payload_offset))
        .map_err(|error| format!("seek past AAC ID3v2 tag: {error}"))?;
    Ok(())
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
    fn decoded_pcm_budget_is_checked_across_adts_frames_before_growth() {
        const MIB: u64 = 1024 * 1024;
        let limits = DecodeLimits::default().with_max_working_set_bytes(Some(MIB));
        let budget = DecodeBudget::new(limits);
        let frame = frame(44_100, 2, &vec![0; FRAME_LEN * 2]);
        let temporary = allocation_bytes::<i16>(frame.pcm.len(), "test frame").unwrap();
        let mut collector = DecodedFrameCollector::default();

        for _ in 0..21 {
            collector
                .push(&frame, budget, temporary)
                .expect("PCM below the exact one-MiB normal-work boundary");
        }
        let before = collector.frame_count;
        let error = collector
            .push(&frame, budget, temporary)
            .expect_err("the next complete AAC frame must cross the cap");
        assert!(error.contains("working-set limit"), "{error}");
        assert_eq!(collector.frame_count, before);
        assert!(collector
            .channels
            .iter()
            .all(|channel| channel.len() == before));

        let mut below_floor = DecodedFrameCollector::default();
        let error = below_floor
            .push(
                &frame,
                DecodeBudget::new(
                    DecodeLimits::default().with_max_working_set_bytes(Some(MIB - 1)),
                ),
                temporary,
            )
            .expect_err("a sub-MiB cap cannot admit normal whole-file processing");
        assert!(error.contains("approximately 1 MiB"), "{error}");
        assert!(below_floor.channels.is_empty());
    }

    #[test]
    fn decoded_frame_count_overflow_fails_without_appending() {
        let mut collector = DecodedFrameCollector {
            sample_rate: Some(44_100),
            channel_count: Some(1),
            channels: vec![Vec::new()],
            frame_count: usize::MAX,
        };
        let error = collector
            .push(
                &frame(44_100, 1, &[0]),
                DecodeBudget::new(DecodeLimits::default()),
                2,
            )
            .unwrap_err();
        assert!(error.contains("frame count overflows"), "{error}");
        assert!(collector.channels[0].is_empty());
    }

    #[test]
    fn repeated_minimal_elements_are_budgeted_before_oxideav_decode() {
        // All-zero bits repeatedly form a tag-0 SCE with max_sfb == 0.
        // oxideav retains each decoded element until frame interleave, so the
        // payload-proportional allowance must reject this amplification before
        // the dependency is entered.
        let payload = vec![0u8; 4_096];
        let bytes = adts_frame(&payload);
        let file = tempfile::NamedTempFile::new().expect("create hostile AAC fixture");
        std::fs::write(file.path(), bytes).expect("write hostile AAC fixture");
        let required = aac_decoder_working_bytes(payload.len() as u64).unwrap();
        let error = decode_adts(
            File::open(file.path()).expect("open hostile AAC fixture"),
            DecodeLimits::default().with_max_working_set_bytes(Some(required.saturating_sub(1))),
        )
        .expect_err("hostile element amplification must fail before decode");
        assert!(error.contains("ADTS AAC decode"), "{error}");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn spare_pcm_capacity_is_combined_with_the_next_aac_frame_peak() {
        const MIB: u64 = 1024 * 1024;
        let mut collector = DecodedFrameCollector {
            sample_rate: Some(44_100),
            channel_count: Some(1),
            channels: vec![Vec::with_capacity(32_768)],
            frame_count: 1_024,
        };
        collector.channels[0].resize(1_024, 0.0);
        let logical_bytes = allocation_bytes::<f64>(1_024, "test AAC length").unwrap()
            + std::mem::size_of::<Vec<f64>>() as u64;
        let next_frame_temporary = MIB - logical_bytes;
        let budget =
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(MIB)));

        // Logical length alone fits this peak, but the already-retained spare
        // capacity does not. This is the exact check performed immediately
        // before entering oxideav.
        budget
            .check_planar_frames(
                1,
                collector.frame_count,
                next_frame_temporary,
                "logical AAC",
            )
            .expect("logical frames fit the crafted cap");
        let error = budget
            .check_planar_capacities(&collector.channels, next_frame_temporary, "ADTS AAC decode")
            .expect_err("actual retained capacity plus next frame must be rejected");
        assert!(error.contains("working-set limit"), "{error}");
    }

    #[test]
    fn fill_only_adts_returns_an_error_instead_of_panicking() {
        let mut assembler = FrameAssembler::new();
        assembler.push_fill(&[]).expect("write AAC fill element");
        let bytes = adts_frame(&assembler.push_end());
        let file = tempfile::NamedTempFile::new().expect("create AAC fixture");
        std::fs::write(file.path(), bytes).expect("write AAC fixture");

        let result = std::panic::catch_unwind(|| {
            decode_adts(
                File::open(file.path()).expect("open AAC fixture"),
                DecodeLimits::default(),
            )
        });
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

        let baseline = decode_adts(
            File::open(baseline_file.path()).expect("open baseline AAC"),
            DecodeLimits::default(),
        )
        .expect("decode baseline AAC");
        let prefixed = decode_adts(
            File::open(prefixed_file.path()).expect("open fill-prefixed AAC"),
            DecodeLimits::default(),
        )
        .expect("decode fill-prefixed AAC");
        assert_eq!(prefixed.sample_rate, baseline.sample_rate);
        assert_eq!(prefixed.channel_mask, baseline.channel_mask);
        assert_eq!(prefixed.channels, baseline.channels);
    }
}
