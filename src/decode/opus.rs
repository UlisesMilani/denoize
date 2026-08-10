use super::DecodedPcm;
use opus::{Channels, Decoder};
use std::ops::Range;
use std::path::Path;

pub fn decode_ogg_opus(path: &Path) -> Result<DecodedPcm, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Opus open: {e}"))?;
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
    let mut decoder = Decoder::new(48_000, channels).map_err(|e| format!("Opus decoder: {e}"))?;
    let _tags = packets
        .read_packet()
        .map_err(|e| format!("Ogg tags: {e}"))?;
    let mut decoded = Vec::<f32>::new();
    let mut granules = OpusGranuleTracker::new(pre_skip);
    let mut buffer = vec![0.0f32; 5_760 * count];
    while let Some(packet) = packets
        .read_packet()
        .map_err(|e| format!("Ogg read: {e}"))?
    {
        if packet.stream_serial() != stream_serial {
            return Err("chained or multiplexed Ogg Opus streams are not supported".into());
        }
        let frames = decoder
            .decode_float(&packet.data, &mut buffer, false)
            .map_err(|e| format!("Opus decode: {e}"))?;
        decoded.extend_from_slice(&buffer[..frames * count]);
        granules.push_packet(
            frames,
            packet.last_in_page(),
            packet.last_in_stream(),
            packet.absgp_page(),
        )?;
    }
    let range = granules.decoded_sample_range(decoded.len(), count)?;
    let mut output = vec![Vec::new(); count];
    for (index, sample) in decoded[range].iter().enumerate() {
        output[index % count].push(*sample as f64);
    }
    Ok(DecodedPcm {
        sample_rate: 48_000,
        channels: output,
        channel_mask: crate::channel_layout::ChannelLayout::from_channel_count(count).mask(),
    })
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
    fn unset_ogg_granule_returns_an_error_instead_of_panicking() {
        let file = tempfile::NamedTempFile::new().expect("create Opus fixture");
        std::fs::write(file.path(), opus_ogg_with_granules(None, u64::MAX))
            .expect("write Opus fixture");

        let result = std::panic::catch_unwind(|| decode_ogg_opus(file.path()));
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

        let standard = decode_ogg_opus(standard_file.path()).expect("decode standard Opus");
        let offset = decode_ogg_opus(offset_file.path()).expect("decode offset Opus");
        assert_eq!(standard.channels[0].len(), 1_548);
        assert_eq!(standard.channels, offset.channels);
    }
}
