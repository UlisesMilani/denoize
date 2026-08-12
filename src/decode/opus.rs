use super::DecodedPcm;
use super::{budget::DecodeBudget, DecodeLimits};
use opus::{Channels, Decoder};
use std::ops::Range;

const MIN_OPUS_RETAINED_SAMPLES: usize = 1024;
const OGG_MAX_PAGE_BODY_BYTES: u64 = 255 * 255;
// Page lacing/position vectors, one page body, and fixed HashMap/reader state
// for the single logical stream proven by our allocation-free preflight.
const OGG_PACKET_READER_FIXED_BYTES: u64 = 128 * 1024;

pub(super) fn decode_ogg_opus_with_limits(
    mut file: std::fs::File,
    limits: DecodeLimits,
) -> Result<DecodedPcm, String> {
    use std::io::{Seek, SeekFrom};

    let budget = DecodeBudget::new(limits);
    // PacketReader materializes a complete packet before returning it. Check
    // the structurally enforced maximum before constructing or entering that
    // third-party reader, including for a real zero-byte decode cap.
    let packet_allocation_bytes = u64::try_from(limits.metadata.max_ogg_packet_bytes)
        .map_err(|_| "Ogg packet limit does not fit in u64".to_string())?;
    let packet_reader_internal_bytes = packet_allocation_bytes
        .checked_add(OGG_MAX_PAGE_BODY_BYTES)
        .and_then(|bytes| bytes.checked_add(OGG_PACKET_READER_FIXED_BYTES))
        .ok_or("Opus packet reader byte count overflows")?;
    let maximum_packet_reader_bytes = packet_reader_internal_bytes
        .checked_add(packet_allocation_bytes)
        .ok_or("Opus packet reader byte count overflows")?;
    budget.check_peak(0, maximum_packet_reader_bytes, "Opus packet reader")?;
    crate::metadata::preflight_ogg_decode(&mut file, limits.metadata)?;
    preflight_single_logical_stream(&mut file)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("Opus rewind: {error}"))?;
    let mut packets = ogg::PacketReader::new(std::io::BufReader::new(file));
    let head = packets
        .read_packet()
        .map_err(|e| format!("Ogg read: {e}"))?
        .ok_or("missing OpusHead")?;
    if head.data.len() < 19 || &head.data[..8] != b"OpusHead" {
        return Err("Ogg stream is not Opus".into());
    }
    let count = head.data[9] as usize;
    if !(1..=2).contains(&count) {
        return Err("only mono/stereo Opus is supported".into());
    }
    let pre_skip = u16::from_le_bytes([head.data[10], head.data[11]]) as usize;
    let stream_serial = head.stream_serial();
    let channels = if count == 1 {
        Channels::Mono
    } else {
        Channels::Stereo
    };
    drop(head);
    let tags = packets
        .read_packet()
        .map_err(|e| format!("Ogg tags: {e}"))?;
    drop(tags);
    let mut decoded = Vec::<f32>::new();
    let mut granules = OpusGranuleTracker::new(pre_skip);
    let scratch_samples = 5_760usize
        .checked_mul(count)
        .ok_or("Opus decode scratch sample count overflows")?;
    let scratch_bytes = allocation_bytes::<f32>(scratch_samples, "Opus decode scratch")?;
    let packet_and_scratch = maximum_packet_reader_bytes
        .checked_add(scratch_bytes)
        .ok_or("Opus packet scratch byte count overflows")?;
    budget.check_peak(0, packet_and_scratch, "Opus decode scratch")?;
    let mut decoder = Decoder::new(48_000, channels).map_err(|e| format!("Opus decoder: {e}"))?;
    let mut buffer = vec![0.0f32; scratch_samples];
    loop {
        // Combine already-retained f32 capacity with the maximum packet
        // allocation before every PacketReader entry. The raw Ogg pass proved
        // that the configured maximum bounds each packet in this file.
        let retained_f32 = allocation_bytes::<f32>(decoded.capacity(), "Opus retained PCM")?;
        budget.check_peak(retained_f32, packet_and_scratch, "Opus packet reader")?;
        let Some(packet) = packets
            .read_packet()
            .map_err(|e| format!("Ogg read: {e}"))?
        else {
            break;
        };
        if packet.stream_serial() != stream_serial {
            return Err("chained or multiplexed Ogg Opus streams are not supported".into());
        }
        let packet_bytes = u64::try_from(packet.data.capacity())
            .map_err(|_| "Opus packet capacity does not fit in u64".to_string())?;
        let retained_f32 = allocation_bytes::<f32>(decoded.capacity(), "Opus retained PCM")?;
        let live_packet_scratch = packet_reader_internal_bytes
            .checked_add(packet_bytes)
            .and_then(|bytes| bytes.checked_add(scratch_bytes))
            .ok_or("Opus packet scratch byte count overflows")?;
        budget.check_peak(retained_f32, live_packet_scratch, "Opus packet decode")?;
        let frames = decoder
            .decode_float(&packet.data, &mut buffer, false)
            .map_err(|e| format!("Opus decode: {e}"))?;
        let packet_samples = frames
            .checked_mul(count)
            .ok_or("Opus decoded packet sample count overflows")?;
        reserve_interleaved_additional(
            &mut decoded,
            packet_samples,
            budget,
            live_packet_scratch,
            "Opus retained PCM",
        )?;
        decoded.extend_from_slice(&buffer[..packet_samples]);
        granules.push_packet(
            frames,
            packet.last_in_page(),
            packet.last_in_stream(),
            packet.absgp_page(),
        )?;
    }
    let range = granules.decoded_sample_range(decoded.len(), count)?;
    let output_frames = range.len() / count;
    drop(buffer);
    drop(decoder);
    drop(packets);
    let retained_f32 = allocation_bytes::<f32>(decoded.capacity(), "Opus retained PCM")?;
    budget.check_planar_frames(count, output_frames, retained_f32, "Opus output conversion")?;
    let mut output = Vec::new();
    budget.check_planar_frames(count, 0, retained_f32, "Opus output channels")?;
    output
        .try_reserve_exact(count)
        .map_err(|error| format!("reserve Opus output channels: {error}"))?;
    output.resize_with(count, Vec::new);
    budget.reserve_planar_frames(
        &mut output,
        output_frames,
        retained_f32,
        "Opus output conversion",
    )?;
    for (index, sample) in decoded[range].iter().enumerate() {
        output[index % count].push(*sample as f64);
    }
    Ok(DecodedPcm {
        sample_rate: 48_000,
        channels: output,
        channel_mask: crate::channel_layout::ChannelLayout::from_channel_count(count).mask(),
    })
}

/// PacketReader retains an unfinished packet for every logical stream. The
/// decoder rejects chained/multiplexed input anyway, so prove the file uses a
/// single serial with fixed-size buffers before handing it to PacketReader.
fn preflight_single_logical_stream(file: &mut std::fs::File) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};

    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind Opus stream preflight: {error}"))?;
    let mut first_serial = None;
    loop {
        let mut header = [0u8; 27];
        let mut read = 0usize;
        while read < header.len() {
            let count = file
                .read(&mut header[read..])
                .map_err(|error| format!("read Opus page header: {error}"))?;
            if count == 0 {
                break;
            }
            read += count;
        }
        if read == 0 {
            break;
        }
        if read != header.len() || &header[..4] != b"OggS" {
            return Err("invalid Ogg page during Opus stream preflight".into());
        }
        let serial = u32::from_le_bytes(header[14..18].try_into().unwrap());
        match first_serial {
            Some(expected) if serial != expected => {
                return Err("chained or multiplexed Ogg Opus streams are not supported".into());
            }
            None => first_serial = Some(serial),
            _ => {}
        }
        let segment_count = usize::from(header[26]);
        let mut lacing = [0u8; 255];
        file.read_exact(&mut lacing[..segment_count])
            .map_err(|error| format!("read Opus page lacing: {error}"))?;
        let body_bytes = lacing[..segment_count]
            .iter()
            .map(|length| u64::from(*length))
            .sum::<u64>();
        let offset = i64::try_from(body_bytes)
            .map_err(|_| "Opus page body size does not fit in i64".to_string())?;
        file.seek(SeekFrom::Current(offset))
            .map_err(|error| format!("skip Opus page body: {error}"))?;
    }
    Ok(())
}

fn allocation_bytes<T>(len: usize, context: &str) -> Result<u64, String> {
    u64::try_from(len)
        .ok()
        .and_then(|len| len.checked_mul(std::mem::size_of::<T>() as u64))
        .ok_or_else(|| format!("{context} byte count overflows"))
}

/// Reserve an interleaved f32 accumulator with checked geometric growth.
/// Near a cap, use all available headroom between the required and doubled
/// capacity so repeated small packets remain amortized instead of reallocating
/// for every append.
fn reserve_interleaved_additional(
    samples: &mut Vec<f32>,
    additional: usize,
    budget: DecodeBudget,
    temporary_bytes: u64,
    context: &str,
) -> Result<usize, String> {
    let required = samples
        .len()
        .checked_add(additional)
        .ok_or_else(|| format!("{context} sample count overflows"))?;
    if samples.capacity() >= required {
        let retained = allocation_bytes::<f32>(samples.capacity(), context)?;
        budget.check_peak(retained, temporary_bytes, context)?;
        return Ok(required);
    }

    let geometric = samples
        .capacity()
        .checked_mul(2)
        .unwrap_or(usize::MAX)
        .max(MIN_OPUS_RETAINED_SAMPLES)
        .max(required);
    let retained = allocation_bytes::<f32>(geometric, context)?;
    let reserve_capacity = if budget
        .check_peak(retained, temporary_bytes, context)
        .is_ok()
    {
        geometric
    } else {
        let required_bytes = allocation_bytes::<f32>(required, context)?;
        budget.check_peak(required_bytes, temporary_bytes, context)?;
        let mut fits = required;
        let mut fails = geometric;
        while fails - fits > 1 {
            let candidate = fits + (fails - fits) / 2;
            let candidate_bytes = allocation_bytes::<f32>(candidate, context)?;
            if budget
                .check_peak(candidate_bytes, temporary_bytes, context)
                .is_ok()
            {
                fits = candidate;
            } else {
                fails = candidate;
            }
        }
        fits
    };
    samples
        .try_reserve_exact(reserve_capacity - samples.len())
        .map_err(|error| format!("reserve {context}: {error}"))?;
    let actual_bytes = allocation_bytes::<f32>(samples.capacity(), context)?;
    budget.check_peak(actual_bytes, temporary_bytes, context)?;
    Ok(required)
}

#[derive(Debug)]
struct OpusGranuleTracker {
    pre_skip: usize,
    total_frames: usize,
    page_start_frame: usize,
    page_frames: usize,
    previous_page_granule: Option<u64>,
    eos_end_frame: Option<usize>,
}

impl OpusGranuleTracker {
    fn new(pre_skip: usize) -> Self {
        Self {
            pre_skip,
            total_frames: 0,
            page_start_frame: 0,
            page_frames: 0,
            previous_page_granule: None,
            eos_end_frame: None,
        }
    }

    fn push_packet(
        &mut self,
        frames: usize,
        last_in_page: bool,
        last_in_stream: bool,
        page_granule: u64,
    ) -> Result<(), String> {
        if self.eos_end_frame.is_some() {
            return Err("Ogg Opus contains audio packets after the end of stream".into());
        }
        self.page_frames = self
            .page_frames
            .checked_add(frames)
            .ok_or("Opus decoded page frame count overflows")?;
        self.total_frames = self
            .total_frames
            .checked_add(frames)
            .ok_or("Opus decoded frame count overflows")?;

        if last_in_stream && !last_in_page {
            return Err("Ogg Opus end-of-stream packet is not last in its page".into());
        }
        if !last_in_page {
            return Ok(());
        }
        // Ogg encodes an unset granule position as all one bits. A page that
        // completes an Opus packet must carry a real granule position.
        if page_granule == u64::MAX {
            return Err("Opus audio page has an unset granule position".into());
        }

        if last_in_stream {
            let kept_on_page = match self.previous_page_granule {
                Some(previous) => {
                    let kept = page_granule
                        .checked_sub(previous)
                        .ok_or("Opus end granule precedes the previous audio page")?;
                    if kept > self.page_frames as u64 {
                        return Err("Opus end granule exceeds decoded frames on its page".into());
                    }
                    kept as usize
                }
                None => {
                    if page_granule < self.pre_skip as u64 {
                        return Err("Opus end granule is smaller than pre-skip".into());
                    }
                    if page_granule >= self.page_frames as u64 {
                        self.page_frames
                    } else {
                        page_granule as usize
                    }
                }
            };
            self.eos_end_frame = Some(
                self.page_start_frame
                    .checked_add(kept_on_page)
                    .ok_or("Opus end frame overflows")?,
            );
        } else if let Some(previous) = self.previous_page_granule {
            let expected = previous
                .checked_add(self.page_frames as u64)
                .ok_or("Opus page granule overflows")?;
            if page_granule != expected {
                return Err("Opus page granules do not match decoded frame durations".into());
            }
            self.previous_page_granule = Some(page_granule);
        } else {
            // The first completed audio page may use a positive granule origin
            // for cropping or a live-stream join. It still cannot claim fewer
            // samples than that page decoded unless it is the EOS page above.
            if page_granule < self.page_frames as u64 {
                return Err("initial Opus granule is smaller than its decoded audio page".into());
            }
            self.previous_page_granule = Some(page_granule);
        }

        self.page_start_frame = self.total_frames;
        self.page_frames = 0;
        Ok(())
    }

    fn decoded_sample_range(
        &self,
        decoded_samples: usize,
        channel_count: usize,
    ) -> Result<Range<usize>, String> {
        if channel_count == 0 || decoded_samples % channel_count != 0 {
            return Err("Opus decoder returned incomplete interleaved PCM".into());
        }
        let decoded_frames = decoded_samples / channel_count;
        if decoded_frames != self.total_frames {
            return Err("Opus granule accounting does not match decoded PCM".into());
        }

        let start_frame = self.pre_skip.min(decoded_frames);
        let end_frame = self
            .eos_end_frame
            .unwrap_or(decoded_frames)
            .min(decoded_frames)
            .max(start_frame);
        Ok(start_frame * channel_count..end_frame * channel_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogg::writing::{PacketWriteEndInfo, PacketWriter};
    use opus::{Application, Encoder};
    use std::borrow::Cow;

    fn opus_ogg_with_granules(first_granule: Option<u64>, final_granule: u64) -> Vec<u8> {
        opus_ogg_with_tag_padding(first_granule, final_granule, 0)
    }

    fn opus_ogg_with_tag_padding(
        first_granule: Option<u64>,
        final_granule: u64,
        tag_padding: usize,
    ) -> Vec<u8> {
        let channel_count = 2u8;
        let pre_skip = 312u16;
        let mut head = b"OpusHead".to_vec();
        head.extend([1, channel_count]);
        head.extend(pre_skip.to_le_bytes());
        head.extend(48_000u32.to_le_bytes());
        head.extend(0i16.to_le_bytes());
        head.push(0);

        let mut tags = b"OpusTags".to_vec();
        tags.extend(0u32.to_le_bytes());
        tags.extend(0u32.to_le_bytes());
        tags.resize(tags.len() + tag_padding, 0);

        let mut encoder = Encoder::new(48_000, Channels::Stereo, Application::Audio)
            .expect("create Opus encoder");
        let first_packet = encoder
            .encode_vec_float(&vec![0.0f32; 960 * usize::from(channel_count)], 4_000)
            .expect("encode Opus packet");
        let final_packet = first_granule.map(|_| {
            encoder
                .encode_vec_float(&vec![0.0f32; 960 * usize::from(channel_count)], 4_000)
                .expect("encode final Opus packet")
        });

        let mut writer = PacketWriter::new(Vec::new());
        let serial = 0x5445_5354;
        writer
            .write_packet(Cow::Owned(head), serial, PacketWriteEndInfo::EndPage, 0)
            .expect("write OpusHead");
        writer
            .write_packet(Cow::Owned(tags), serial, PacketWriteEndInfo::EndPage, 0)
            .expect("write OpusTags");
        if let Some(first_granule) = first_granule {
            writer
                .write_packet(
                    Cow::Owned(first_packet),
                    serial,
                    PacketWriteEndInfo::EndPage,
                    first_granule,
                )
                .expect("write first Opus audio packet");
            writer
                .write_packet(
                    Cow::Owned(final_packet.expect("second encoded packet")),
                    serial,
                    PacketWriteEndInfo::EndStream,
                    final_granule,
                )
                .expect("write final Opus audio packet");
        } else {
            writer
                .write_packet(
                    Cow::Owned(first_packet),
                    serial,
                    PacketWriteEndInfo::EndStream,
                    final_granule,
                )
                .expect("write Opus audio packet");
        }
        writer.into_inner()
    }

    #[test]
    fn opus_sample_window_handles_large_granules_without_overflow() {
        let mut tracker = OpusGranuleTracker::new(312);
        tracker
            .push_packet(960, true, false, u64::MAX - 1)
            .expect("large positive initial granule is valid");
        assert!(tracker
            .push_packet(960, true, true, u64::MAX)
            .unwrap_err()
            .contains("unset granule"));
    }

    #[test]
    fn opus_sample_window_never_ends_before_pre_skip() {
        let mut tracker = OpusGranuleTracker::new(312);
        assert!(tracker
            .push_packet(960, true, true, 100)
            .unwrap_err()
            .contains("smaller than pre-skip"));

        let tracker = OpusGranuleTracker::new(0);
        assert!(tracker
            .decoded_sample_range(3, 2)
            .unwrap_err()
            .contains("incomplete interleaved PCM"));
    }

    #[test]
    fn retained_pcm_uses_checked_headroom_instead_of_per_packet_reallocation() {
        let budget =
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(3_000)));
        let mut samples = Vec::new();
        reserve_interleaved_additional(&mut samples, 100, budget, 0, "test Opus PCM")
            .expect("tight cap still reserves useful headroom");
        assert!(samples.capacity() > 100);
        assert!(samples.capacity() <= 750);
        let initial_capacity = samples.capacity();
        samples.resize(100, 0.0);

        for _ in 0..100 {
            reserve_interleaved_additional(&mut samples, 1, budget, 0, "test Opus PCM")
                .expect("small packet fits retained headroom");
            samples.push(0.0);
            assert_eq!(samples.capacity(), initial_capacity);
        }

        let mut rejected = Vec::new();
        let error = reserve_interleaved_additional(
            &mut rejected,
            1,
            DecodeBudget::new(DecodeLimits::default().with_max_working_set_bytes(Some(0))),
            0,
            "test Opus PCM",
        )
        .unwrap_err();
        assert!(error.contains("working-set limit"), "{error}");
        assert!(rejected.is_empty());
    }

    #[test]
    fn packet_reader_budget_is_checked_before_reading_opus_head() {
        let file = tempfile::NamedTempFile::new().expect("create Opus fixture");
        std::fs::write(file.path(), opus_ogg_with_granules(None, 1_860))
            .expect("write Opus fixture");
        let error = decode_ogg_opus_with_limits(
            std::fs::File::open(file.path()).expect("open Opus fixture"),
            DecodeLimits::default().with_max_working_set_bytes(Some(0)),
        )
        .expect_err("zero decode cap must fail before PacketReader");
        assert!(error.contains("Opus packet reader"), "{error}");
    }

    #[test]
    fn unset_ogg_granule_returns_an_error_instead_of_panicking() {
        let file = tempfile::NamedTempFile::new().expect("create Opus fixture");
        std::fs::write(file.path(), opus_ogg_with_granules(None, u64::MAX))
            .expect("write Opus fixture");

        let result = std::panic::catch_unwind(|| {
            decode_ogg_opus_with_limits(
                std::fs::File::open(file.path()).expect("open Opus fixture"),
                DecodeLimits::default(),
            )
        });
        let error = result
            .expect("unset Opus granule must not panic")
            .expect_err("unset Opus granule must fail");
        assert!(error.contains("unset granule"), "{error}");
    }

    #[test]
    fn positive_initial_granule_does_not_hide_end_trimming() {
        let standard_file = tempfile::NamedTempFile::new().expect("create standard Opus fixture");
        let offset_file = tempfile::NamedTempFile::new().expect("create offset Opus fixture");
        std::fs::write(
            standard_file.path(),
            opus_ogg_with_granules(Some(960), 1_860),
        )
        .expect("write standard Opus fixture");
        std::fs::write(
            offset_file.path(),
            opus_ogg_with_granules(Some(10_000), 10_900),
        )
        .expect("write positive-origin Opus fixture");

        let standard = decode_ogg_opus_with_limits(
            std::fs::File::open(standard_file.path()).expect("open standard Opus fixture"),
            DecodeLimits::default(),
        )
        .expect("decode standard Opus");
        let offset = decode_ogg_opus_with_limits(
            std::fs::File::open(offset_file.path()).expect("open offset Opus fixture"),
            DecodeLimits::default(),
        )
        .expect("decode offset Opus");
        assert_eq!(standard.channels[0].len(), 1_548);
        assert_eq!(standard.channels, offset.channels);
    }

    #[test]
    fn oversized_opus_tags_are_rejected_before_packet_decode() {
        let file = tempfile::NamedTempFile::new().expect("create oversized OpusTags fixture");
        std::fs::write(file.path(), opus_ogg_with_tag_padding(None, 1_860, 64))
            .expect("write oversized OpusTags fixture");
        let mut limits = DecodeLimits::default();
        limits.metadata.max_ogg_packet_bytes = 32;
        limits.metadata.max_item_bytes = 32;

        let error = decode_ogg_opus_with_limits(
            std::fs::File::open(file.path()).expect("open oversized OpusTags fixture"),
            limits,
        )
        .expect_err("oversized OpusTags must fail during bounded preflight");
        assert!(error.contains("limit") || error.contains("32"), "{error}");
    }
}
