//! M4A / MP4-AAC decoder — `mp4` demux + Pure-Rust `oxideav-aac` AAC-LC decode.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use mp4::{ChannelConfig, MediaType, Mp4Reader, Mp4Track, TrackType};
use oxideav_aac::decode::{DecodedFrame, StreamDecoder};

use super::pcm::DecodedPcm;

/// `bufferSizeDB` in MPEG-4 systems descriptors is 24 bits. Keeping access
/// units within that representable range also prevents a corrupt `stsz` from
/// turning one tiny input into a multi-gigabyte allocation.
const MAX_AAC_ACCESS_UNIT_SIZE: u32 = 0x00ff_ffff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampleDescriptor {
    offset: u64,
    size: u32,
    index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StszFields {
    sample_size: u32,
    sample_count: u32,
    variable_size_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StscFields {
    first_chunk: u32,
    samples_per_chunk: u32,
    sample_description_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkOffsetsKind {
    Missing,
    Stco(usize),
    Co64(usize),
    Both,
}

/// Primitive-only view of the sample-table fields used by this decoder.
///
/// The `mp4` crate exposes `Mp4Track`, but the concrete sample-table box types
/// live in private modules. This interface lets the production path inspect
/// those public fields without naming their types, and keeps all arithmetic in
/// this module instead of relying on `mp4`'s derived `first_sample` values.
trait SampleTable {
    fn stsz_fields(&self) -> StszFields;
    fn variable_sample_size(&self, index: usize) -> Option<u32>;

    fn chunk_offsets_kind(&self) -> ChunkOffsetsKind;
    fn chunk_offset(&self, index: usize) -> Option<u64>;

    fn stsc_len(&self) -> usize;
    fn stsc_entry(&self, index: usize) -> Option<StscFields>;

    fn stts_len(&self) -> usize;
    fn stts_sample_count(&self, index: usize) -> Option<u32>;

    fn has_ctts(&self) -> bool;
    fn ctts_len(&self) -> usize;
    fn ctts_sample_count(&self, index: usize) -> Option<u32>;
}

impl SampleTable for Mp4Track {
    fn stsz_fields(&self) -> StszFields {
        let stsz = &self.trak.mdia.minf.stbl.stsz;
        StszFields {
            sample_size: stsz.sample_size,
            sample_count: stsz.sample_count,
            variable_size_count: stsz.sample_sizes.len(),
        }
    }

    fn variable_sample_size(&self, index: usize) -> Option<u32> {
        self.trak
            .mdia
            .minf
            .stbl
            .stsz
            .sample_sizes
            .get(index)
            .copied()
    }

    fn chunk_offsets_kind(&self) -> ChunkOffsetsKind {
        let stbl = &self.trak.mdia.minf.stbl;
        match (&stbl.stco, &stbl.co64) {
            (None, None) => ChunkOffsetsKind::Missing,
            (Some(stco), None) => ChunkOffsetsKind::Stco(stco.entries.len()),
            (None, Some(co64)) => ChunkOffsetsKind::Co64(co64.entries.len()),
            (Some(_), Some(_)) => ChunkOffsetsKind::Both,
        }
    }

    fn chunk_offset(&self, index: usize) -> Option<u64> {
        let stbl = &self.trak.mdia.minf.stbl;
        match (&stbl.stco, &stbl.co64) {
            (Some(stco), None) => stco.entries.get(index).copied().map(u64::from),
            (None, Some(co64)) => co64.entries.get(index).copied(),
            _ => None,
        }
    }

    fn stsc_len(&self) -> usize {
        self.trak.mdia.minf.stbl.stsc.entries.len()
    }

    fn stsc_entry(&self, index: usize) -> Option<StscFields> {
        self.trak
            .mdia
            .minf
            .stbl
            .stsc
            .entries
            .get(index)
            .map(|entry| StscFields {
                first_chunk: entry.first_chunk,
                samples_per_chunk: entry.samples_per_chunk,
                sample_description_index: entry.sample_description_index,
            })
    }

    fn stts_len(&self) -> usize {
        self.trak.mdia.minf.stbl.stts.entries.len()
    }

    fn stts_sample_count(&self, index: usize) -> Option<u32> {
        self.trak
            .mdia
            .minf
            .stbl
            .stts
            .entries
            .get(index)
            .map(|entry| entry.sample_count)
    }

    fn has_ctts(&self) -> bool {
        self.trak.mdia.minf.stbl.ctts.is_some()
    }

    fn ctts_len(&self) -> usize {
        self.trak
            .mdia
            .minf
            .stbl
            .ctts
            .as_ref()
            .map_or(0, |ctts| ctts.entries.len())
    }

    fn ctts_sample_count(&self, index: usize) -> Option<u32> {
        self.trak
            .mdia
            .minf
            .stbl
            .ctts
            .as_ref()?
            .entries
            .get(index)
            .map(|entry| entry.sample_count)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValidatedSampleTable {
    sample_count: u32,
}

/// Decode M4A/MP4-AAC from path.
pub fn decode_m4a(path: &Path) -> Result<DecodedPcm, String> {
    // Keep the original handle for payload reads. Parsing happens through a
    // clone because each access unit below is read with an absolute seek.
    let mut payload_reader = File::open(path).map_err(|e| format!("open m4a: {e}"))?;
    let file_size = payload_reader
        .metadata()
        .map_err(|e| format!("stat m4a: {e}"))?
        .len();
    let header_file = payload_reader
        .try_clone()
        .map_err(|e| format!("clone m4a handle for header parsing: {e}"))?;
    let mp4 = Mp4Reader::read_header(BufReader::new(header_file), file_size)
        .map_err(|e| format!("mp4 parse: {e}"))?;

    let track = select_aac_track(&mp4)?;
    if !track.trafs.is_empty() {
        return Err(format!(
            "fragmented AAC track {} is not supported; a regular M4A sample table is required",
            track.track_id()
        ));
    }

    let validated = validate_sample_table(track, file_size)
        .map_err(|e| format!("AAC track {} sample table: {e}", track.track_id()))?;

    let profile = track
        .audio_profile()
        .map_err(|e| format!("aac profile: {e}"))?;
    let freq_index = track
        .sample_freq_index()
        .map_err(|e| format!("aac sample rate: {e}"))?;
    let channel_config = track
        .channel_config()
        .map_err(|e| format!("aac channels: {e}"))?;

    let sample_rate = freq_index.freq();
    let aot = profile as u8;
    let fs_index = freq_index as u8;
    let chan_conf = channel_config as u8;
    let n_ch = channel_config_to_count(channel_config);

    let mut decoder = StreamDecoder::new();
    let mut channels = Vec::new();
    channels
        .try_reserve_exact(n_ch)
        .map_err(|e| format!("reserve M4A output channels: {e}"))?;
    channels.resize_with(n_ch, Vec::new);
    let mut access_unit = Vec::new();
    let mut decoded_frames = 0usize;

    visit_sample_descriptors(track, validated.sample_count, |descriptor| {
        // Zero-sized entries still participate in stsc accounting. They carry
        // no AAC payload, so retain the existing behavior of skipping decode.
        if descriptor.size == 0 {
            return Ok(());
        }

        let size = usize::try_from(descriptor.size).map_err(|_| {
            format!(
                "AAC sample {} size cannot be represented on this platform",
                descriptor.index
            )
        })?;
        access_unit.clear();
        access_unit.try_reserve_exact(size).map_err(|e| {
            format!(
                "reserve AAC sample {} ({} bytes): {e}",
                descriptor.index, descriptor.size
            )
        })?;
        access_unit.resize(size, 0);

        payload_reader
            .seek(SeekFrom::Start(descriptor.offset))
            .map_err(|e| format!("seek AAC sample {}: {e}", descriptor.index))?;
        payload_reader
            .read_exact(&mut access_unit)
            .map_err(|e| format!("read AAC sample {}: {e}", descriptor.index))?;

        let frame = decoder
            .decode_raw_data_block(aot, fs_index, sample_rate, chan_conf, 1, &access_unit)
            .map_err(|e| format!("decode AAC sample {}: {e}", descriptor.index))?;
        append_decoded_frame(
            &mut channels,
            &frame,
            n_ch,
            sample_rate,
            &mut decoded_frames,
        )
        .map_err(|e| format!("AAC sample {}: {e}", descriptor.index))
    })?;

    if decoded_frames == 0 {
        return Err("M4A decode produced no samples".into());
    }

    Ok(DecodedPcm {
        sample_rate,
        channels,
        channel_mask: crate::channel_layout::ChannelLayout::from_channel_count(n_ch).mask(),
    })
}

fn select_aac_track<R: Read + Seek>(mp4: &Mp4Reader<R>) -> Result<&Mp4Track, String> {
    // `Mp4Reader` stores tracks in a HashMap. Validate IDs and then follow the
    // original moov/trak ordering so selection is deterministic.
    if mp4.tracks().len() != mp4.moov.traks.len() {
        return Err("duplicate MP4 track IDs in moov".into());
    }

    for trak in &mp4.moov.traks {
        let track_id = trak.tkhd.track_id;
        let track = mp4
            .tracks()
            .get(&track_id)
            .ok_or_else(|| format!("MP4 track {track_id} metadata missing"))?;
        if track.track_type().ok() == Some(TrackType::Audio)
            && track.media_type().ok() == Some(MediaType::AAC)
        {
            return Ok(track);
        }
    }

    Err("no AAC audio track found in M4A/MP4".into())
}

fn validate_sample_table(track: &Mp4Track, file_size: u64) -> Result<ValidatedSampleTable, String> {
    validate_table(track, file_size)
}

/// Validate every table relationship and byte range before allocating an AAC
/// access-unit buffer. The success path of this pass performs no allocation.
fn validate_table<T: SampleTable + ?Sized>(
    table: &T,
    file_size: u64,
) -> Result<ValidatedSampleTable, String> {
    let stsz = table.stsz_fields();
    if stsz.sample_count == 0 {
        return Err("stsz declares no samples".into());
    }

    let sample_count = usize::try_from(stsz.sample_count)
        .map_err(|_| "stsz sample_count cannot be represented on this platform")?;
    if stsz.sample_size == 0 {
        if stsz.variable_size_count != sample_count {
            return Err(format!(
                "stsz variable-size entry count {} does not match sample_count {}",
                stsz.variable_size_count, stsz.sample_count
            ));
        }
    } else if stsz.variable_size_count != 0 {
        return Err(format!(
            "stsz fixed sample_size {} must not have {} variable-size entries",
            stsz.sample_size, stsz.variable_size_count
        ));
    }

    let chunk_count = match table.chunk_offsets_kind() {
        ChunkOffsetsKind::Missing => return Err("missing stco/co64 chunk-offset table".into()),
        ChunkOffsetsKind::Both => {
            return Err("both stco and co64 are present; exactly one is required".into())
        }
        ChunkOffsetsKind::Stco(len) | ChunkOffsetsKind::Co64(len) => len,
    };
    if chunk_count == 0 {
        return Err("chunk-offset table is empty".into());
    }
    let chunk_count_u32 =
        u32::try_from(chunk_count).map_err(|_| "chunk-offset count exceeds the MP4 u32 limit")?;

    let stsc_len = table.stsc_len();
    if stsc_len == 0 {
        return Err("stsc has no entries".into());
    }
    let mut mapped_samples = 0u64;
    for index in 0..stsc_len {
        let entry = table
            .stsc_entry(index)
            .ok_or("stsc entry disappeared during validation")?;
        if index == 0 && entry.first_chunk != 1 {
            return Err(format!(
                "stsc first entry starts at chunk {}, expected 1",
                entry.first_chunk
            ));
        }
        if entry.first_chunk == 0 || entry.first_chunk > chunk_count_u32 {
            return Err(format!(
                "stsc entry {} first_chunk {} is outside 1..={}",
                index + 1,
                entry.first_chunk,
                chunk_count
            ));
        }
        if entry.samples_per_chunk == 0 {
            return Err(format!(
                "stsc entry {} has zero samples_per_chunk",
                index + 1
            ));
        }
        if entry.sample_description_index != 1 {
            return Err(format!(
                "stsc entry {} references unsupported sample description {} (expected 1)",
                index + 1,
                entry.sample_description_index
            ));
        }

        let next_first_chunk = if index + 1 < stsc_len {
            let next = table
                .stsc_entry(index + 1)
                .ok_or("stsc entry disappeared during validation")?;
            if next.first_chunk <= entry.first_chunk {
                return Err(format!(
                    "stsc first_chunk values are not strictly increasing at entry {}",
                    index + 2
                ));
            }
            u64::from(next.first_chunk)
        } else {
            u64::from(chunk_count_u32) + 1
        };
        let run_chunks = next_first_chunk
            .checked_sub(u64::from(entry.first_chunk))
            .ok_or("stsc chunk run underflows")?;
        let run_samples = run_chunks
            .checked_mul(u64::from(entry.samples_per_chunk))
            .ok_or("stsc sample total overflows")?;
        mapped_samples = mapped_samples
            .checked_add(run_samples)
            .ok_or("stsc sample total overflows")?;
    }
    if mapped_samples != u64::from(stsz.sample_count) {
        return Err(format!(
            "stsc maps {mapped_samples} samples but stsz declares {}",
            stsz.sample_count
        ));
    }

    let stts_len = table.stts_len();
    if stts_len == 0 {
        return Err("stts has no entries".into());
    }
    let stts_samples =
        sum_positive_counts(stts_len, |index| table.stts_sample_count(index), "stts")?;
    if stts_samples != u64::from(stsz.sample_count) {
        return Err(format!(
            "stts covers {stts_samples} samples but stsz declares {}",
            stsz.sample_count
        ));
    }

    if table.has_ctts() {
        let ctts_len = table.ctts_len();
        if ctts_len == 0 {
            return Err("ctts is present but has no entries".into());
        }
        let ctts_samples =
            sum_positive_counts(ctts_len, |index| table.ctts_sample_count(index), "ctts")?;
        if ctts_samples != u64::from(stsz.sample_count) {
            return Err(format!(
                "ctts covers {ctts_samples} samples but stsz declares {}",
                stsz.sample_count
            ));
        }
    }

    visit_sample_descriptors(table, stsz.sample_count, |descriptor| {
        if descriptor.size > MAX_AAC_ACCESS_UNIT_SIZE {
            return Err(format!(
                "AAC sample {} is {} bytes, exceeding the {}-byte safety limit",
                descriptor.index, descriptor.size, MAX_AAC_ACCESS_UNIT_SIZE
            ));
        }
        let end = descriptor
            .offset
            .checked_add(u64::from(descriptor.size))
            .ok_or_else(|| format!("AAC sample {} byte range overflows", descriptor.index))?;
        if descriptor.offset > file_size || end > file_size {
            return Err(format!(
                "AAC sample {} byte range {}..{} exceeds file size {}",
                descriptor.index, descriptor.offset, end, file_size
            ));
        }
        Ok(())
    })?;

    Ok(ValidatedSampleTable {
        sample_count: stsz.sample_count,
    })
}

fn sum_positive_counts<F>(len: usize, mut count_at: F, table_name: &str) -> Result<u64, String>
where
    F: FnMut(usize) -> Option<u32>,
{
    let mut total = 0u64;
    for index in 0..len {
        let count = count_at(index)
            .ok_or_else(|| format!("{table_name} entry disappeared during validation"))?;
        if count == 0 {
            return Err(format!(
                "{table_name} entry {} has zero sample_count",
                index + 1
            ));
        }
        total = total
            .checked_add(u64::from(count))
            .ok_or_else(|| format!("{table_name} sample total overflows"))?;
    }
    Ok(total)
}

/// Re-walk the validated tables using only scalar cursor state. Descriptors are
/// consumed immediately instead of being collected into a sample-count-sized
/// allocation.
fn visit_sample_descriptors<T, F>(
    table: &T,
    expected_sample_count: u32,
    mut visit: F,
) -> Result<(), String>
where
    T: SampleTable + ?Sized,
    F: FnMut(SampleDescriptor) -> Result<(), String>,
{
    let chunk_count = match table.chunk_offsets_kind() {
        ChunkOffsetsKind::Stco(len) | ChunkOffsetsKind::Co64(len) => len,
        ChunkOffsetsKind::Missing => return Err("missing stco/co64 chunk-offset table".into()),
        ChunkOffsetsKind::Both => {
            return Err("both stco and co64 are present; exactly one is required".into())
        }
    };
    let stsz = table.stsz_fields();
    let mut stsc_index = 0usize;
    let mut emitted = 0u32;

    for chunk_index in 0..chunk_count {
        let chunk_number = u32::try_from(chunk_index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .ok_or("chunk number exceeds the MP4 u32 limit")?;
        while stsc_index + 1 < table.stsc_len() {
            let next = table
                .stsc_entry(stsc_index + 1)
                .ok_or("stsc entry disappeared while walking samples")?;
            if chunk_number < next.first_chunk {
                break;
            }
            stsc_index += 1;
        }
        let stsc = table
            .stsc_entry(stsc_index)
            .ok_or("stsc entry missing while walking samples")?;
        let mut offset = table
            .chunk_offset(chunk_index)
            .ok_or("chunk offset missing while walking samples")?;

        for _ in 0..stsc.samples_per_chunk {
            let sample_index = usize::try_from(emitted)
                .map_err(|_| "sample index cannot be represented on this platform")?;
            let size = if stsz.sample_size == 0 {
                table
                    .variable_sample_size(sample_index)
                    .ok_or("stsz entry missing while walking samples")?
            } else {
                stsz.sample_size
            };
            let index = emitted.checked_add(1).ok_or("AAC sample index overflows")?;
            visit(SampleDescriptor {
                offset,
                size,
                index,
            })?;
            offset = offset
                .checked_add(u64::from(size))
                .ok_or_else(|| format!("AAC sample {index} byte range overflows"))?;
            emitted = index;
        }
    }

    if emitted != expected_sample_count {
        return Err(format!(
            "sample-table walk produced {emitted} samples, expected {expected_sample_count}"
        ));
    }
    Ok(())
}

fn channel_config_to_count(cfg: ChannelConfig) -> usize {
    match cfg {
        ChannelConfig::Mono => 1,
        ChannelConfig::Stereo => 2,
        ChannelConfig::Three => 3,
        ChannelConfig::Four => 4,
        ChannelConfig::Five => 5,
        ChannelConfig::FiveOne => 6,
        ChannelConfig::SevenOne => 8,
    }
}

fn append_decoded_frame(
    channels: &mut [Vec<f64>],
    frame: &DecodedFrame,
    expected_channels: usize,
    expected_sample_rate: u32,
    total_frames: &mut usize,
) -> Result<(), String> {
    if frame.channels == 0 {
        if frame.pcm.is_empty() {
            return Ok(());
        }
        return Err("zero-channel AAC frame unexpectedly contains PCM samples".into());
    }
    // Tolerate an empty channel-bearing frame as a decoder priming/no-output
    // marker, matching the raw ADTS adapter.
    if frame.pcm.is_empty() {
        return Ok(());
    }
    if channels.len() != expected_channels || frame.channels != expected_channels {
        return Err(format!(
            "decoded channel count {} does not match AAC configuration {}",
            frame.channels, expected_channels
        ));
    }
    if frame.sample_rate != expected_sample_rate {
        return Err(format!(
            "decoded sample rate {} does not match AAC configuration {}",
            frame.sample_rate, expected_sample_rate
        ));
    }
    if frame.pcm.len() % expected_channels != 0 {
        return Err(format!(
            "decoded PCM length {} is not divisible by {} channels",
            frame.pcm.len(),
            expected_channels
        ));
    }

    let frame_count = frame.pcm.len() / expected_channels;
    let next_total = total_frames
        .checked_add(frame_count)
        .ok_or("decoded M4A frame count overflows")?;
    for channel in channels.iter_mut() {
        channel
            .try_reserve(frame_count)
            .map_err(|e| format!("reserve decoded M4A PCM: {e}"))?;
    }
    for samples in frame.pcm.chunks_exact(expected_channels) {
        for (channel, sample) in channels.iter_mut().zip(samples) {
            let value = *sample as f64 / 32768.0;
            channel.push(crate::audio::sanitize_sample(value));
        }
    }
    *total_frames = next_total;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[derive(Clone, Debug)]
    struct TestTable {
        fixed_size: u32,
        sample_count: u32,
        variable_sizes: Vec<u32>,
        chunk_kind: ChunkOffsetsKind,
        chunk_offsets: Vec<u64>,
        stsc: Vec<StscFields>,
        stts: Vec<u32>,
        ctts: Option<Vec<u32>>,
    }

    impl TestTable {
        fn variable_stco() -> Self {
            Self {
                fixed_size: 0,
                sample_count: 3,
                variable_sizes: vec![2, 3, 4],
                chunk_kind: ChunkOffsetsKind::Stco(2),
                chunk_offsets: vec![10, 20],
                stsc: vec![
                    StscFields {
                        first_chunk: 1,
                        samples_per_chunk: 2,
                        sample_description_index: 1,
                    },
                    StscFields {
                        first_chunk: 2,
                        samples_per_chunk: 1,
                        sample_description_index: 1,
                    },
                ],
                stts: vec![3],
                ctts: None,
            }
        }
    }

    impl SampleTable for TestTable {
        fn stsz_fields(&self) -> StszFields {
            StszFields {
                sample_size: self.fixed_size,
                sample_count: self.sample_count,
                variable_size_count: self.variable_sizes.len(),
            }
        }

        fn variable_sample_size(&self, index: usize) -> Option<u32> {
            self.variable_sizes.get(index).copied()
        }

        fn chunk_offsets_kind(&self) -> ChunkOffsetsKind {
            self.chunk_kind
        }

        fn chunk_offset(&self, index: usize) -> Option<u64> {
            self.chunk_offsets.get(index).copied()
        }

        fn stsc_len(&self) -> usize {
            self.stsc.len()
        }

        fn stsc_entry(&self, index: usize) -> Option<StscFields> {
            self.stsc.get(index).copied()
        }

        fn stts_len(&self) -> usize {
            self.stts.len()
        }

        fn stts_sample_count(&self, index: usize) -> Option<u32> {
            self.stts.get(index).copied()
        }

        fn has_ctts(&self) -> bool {
            self.ctts.is_some()
        }

        fn ctts_len(&self) -> usize {
            self.ctts.as_ref().map_or(0, Vec::len)
        }

        fn ctts_sample_count(&self, index: usize) -> Option<u32> {
            self.ctts.as_ref()?.get(index).copied()
        }
    }

    fn descriptors(table: &TestTable) -> Result<Vec<SampleDescriptor>, String> {
        let mut descriptors = Vec::new();
        visit_sample_descriptors(table, table.sample_count, |descriptor| {
            descriptors.push(descriptor);
            Ok(())
        })?;
        Ok(descriptors)
    }

    fn encoded_aac_table(sample_sizes: &[usize]) -> (Vec<u8>, Mp4Reader<Cursor<Vec<u8>>>) {
        let config = mp4::Mp4Config {
            major_brand: "M4A ".parse().unwrap(),
            minor_version: 0,
            compatible_brands: vec!["M4A ".parse().unwrap(), "isom".parse().unwrap()],
            timescale: 48_000,
        };
        let cursor = Cursor::new(Vec::new());
        let mut writer = mp4::Mp4Writer::write_start(cursor, &config).unwrap();
        writer
            .add_track(&mp4::TrackConfig::from(mp4::AacConfig::default()))
            .unwrap();
        let mut start_time = 0u64;
        for &size in sample_sizes {
            writer
                .write_sample(
                    1,
                    &mp4::Mp4Sample {
                        start_time,
                        duration: 1024,
                        rendering_offset: 0,
                        is_sync: true,
                        bytes: mp4::Bytes::from(vec![size as u8; size]),
                    },
                )
                .unwrap();
            start_time += 1024;
        }
        writer.write_end().unwrap();
        let bytes = writer.into_writer().into_inner();
        let reader =
            Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64).unwrap();
        (bytes, reader)
    }

    fn encoded_aac_tracks(count: usize) -> Vec<u8> {
        let config = mp4::Mp4Config {
            major_brand: "M4A ".parse().unwrap(),
            minor_version: 0,
            compatible_brands: vec!["M4A ".parse().unwrap(), "isom".parse().unwrap()],
            timescale: 48_000,
        };
        let cursor = Cursor::new(Vec::new());
        let mut writer = mp4::Mp4Writer::write_start(cursor, &config).unwrap();
        for _ in 0..count {
            writer
                .add_track(&mp4::TrackConfig::from(mp4::AacConfig::default()))
                .unwrap();
        }
        writer.write_end().unwrap();
        writer.into_writer().into_inner()
    }

    #[test]
    fn validates_variable_stco_and_walks_exact_offsets() {
        let table = TestTable::variable_stco();
        assert_eq!(
            validate_table(&table, 100).unwrap(),
            ValidatedSampleTable { sample_count: 3 }
        );
        assert_eq!(
            descriptors(&table).unwrap(),
            vec![
                SampleDescriptor {
                    offset: 10,
                    size: 2,
                    index: 1,
                },
                SampleDescriptor {
                    offset: 12,
                    size: 3,
                    index: 2,
                },
                SampleDescriptor {
                    offset: 20,
                    size: 4,
                    index: 3,
                },
            ]
        );
    }

    #[test]
    fn validates_fixed_co64_and_walks_exact_offsets() {
        let mut table = TestTable::variable_stco();
        table.fixed_size = 4;
        table.variable_sizes.clear();
        table.chunk_kind = ChunkOffsetsKind::Co64(2);
        validate_table(&table, 100).unwrap();
        assert_eq!(
            descriptors(&table).unwrap(),
            vec![
                SampleDescriptor {
                    offset: 10,
                    size: 4,
                    index: 1,
                },
                SampleDescriptor {
                    offset: 14,
                    size: 4,
                    index: 2,
                },
                SampleDescriptor {
                    offset: 20,
                    size: 4,
                    index: 3,
                },
            ]
        );
    }

    #[test]
    fn validates_real_mp4_writer_tables_without_derived_sample_math() {
        for sizes in [&[4usize, 4][..], &[2usize, 5][..]] {
            let (bytes, reader) = encoded_aac_table(sizes);
            let track = select_aac_track(&reader).unwrap();
            let validated = validate_sample_table(track, bytes.len() as u64).unwrap();
            assert_eq!(validated.sample_count, 2);

            let mut actual_sizes = Vec::new();
            visit_sample_descriptors(track, validated.sample_count, |descriptor| {
                actual_sizes.push(descriptor.size as usize);
                let start = descriptor.offset as usize;
                let end = start + descriptor.size as usize;
                assert_eq!(&bytes[start..end], vec![descriptor.size as u8; end - start]);
                Ok(())
            })
            .unwrap();
            assert_eq!(actual_sizes, sizes);
        }
    }

    #[test]
    fn selects_first_aac_in_moov_order_and_rejects_duplicate_track_ids() {
        let mut bytes = encoded_aac_tracks(2);
        let reader =
            Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64).unwrap();
        assert_eq!(select_aac_track(&reader).unwrap().track_id(), 1);

        let tkhd_offsets = bytes
            .windows(4)
            .enumerate()
            .filter_map(|(offset, bytes)| (bytes == b"tkhd").then_some(offset + 16))
            .collect::<Vec<_>>();
        assert_eq!(tkhd_offsets.len(), 2);
        bytes[tkhd_offsets[1]..tkhd_offsets[1] + 4].copy_from_slice(&1u32.to_be_bytes());
        let duplicate_reader =
            Mp4Reader::read_header(Cursor::new(bytes.clone()), bytes.len() as u64).unwrap();
        let error = select_aac_track(&duplicate_reader).unwrap_err();
        assert!(error.contains("duplicate MP4 track IDs"), "{error}");
    }

    #[test]
    fn rejects_oversized_access_unit_without_allocating_it() {
        let mut table = TestTable::variable_stco();
        table.fixed_size = MAX_AAC_ACCESS_UNIT_SIZE + 1;
        table.variable_sizes.clear();
        let error = validate_table(&table, u64::MAX).unwrap_err();
        assert!(error.contains("safety limit"), "{error}");
    }

    #[test]
    fn rejects_inconsistent_stsz_semantics() {
        let mut variable = TestTable::variable_stco();
        variable.variable_sizes.pop();
        assert!(validate_table(&variable, 100)
            .unwrap_err()
            .contains("variable-size entry count"));

        let mut fixed = TestTable::variable_stco();
        fixed.fixed_size = 3;
        assert!(validate_table(&fixed, 100)
            .unwrap_err()
            .contains("must not have"));
    }

    #[test]
    fn rejects_missing_ambiguous_or_empty_chunk_offsets() {
        for kind in [
            ChunkOffsetsKind::Missing,
            ChunkOffsetsKind::Both,
            ChunkOffsetsKind::Stco(0),
        ] {
            let mut table = TestTable::variable_stco();
            table.chunk_kind = kind;
            assert!(validate_table(&table, 100).is_err(), "{kind:?}");
        }
    }

    #[test]
    fn rejects_zero_unordered_or_mismatched_stsc_runs() {
        let mut zero = TestTable::variable_stco();
        zero.stsc[0].samples_per_chunk = 0;
        assert!(validate_table(&zero, 100)
            .unwrap_err()
            .contains("zero samples_per_chunk"));

        let mut unordered = TestTable::variable_stco();
        unordered.stsc[1].first_chunk = 1;
        assert!(validate_table(&unordered, 100)
            .unwrap_err()
            .contains("not strictly increasing"));

        let mut mismatch = TestTable::variable_stco();
        mismatch.stsc[0].samples_per_chunk = 1;
        assert!(validate_table(&mismatch, 100)
            .unwrap_err()
            .contains("stsc maps"));
    }

    #[test]
    fn rejects_stts_and_ctts_count_mismatches() {
        let mut stts = TestTable::variable_stco();
        stts.stts = vec![2];
        assert!(validate_table(&stts, 100)
            .unwrap_err()
            .contains("stts covers"));

        let mut ctts = TestTable::variable_stco();
        ctts.ctts = Some(vec![1, 1]);
        assert!(validate_table(&ctts, 100)
            .unwrap_err()
            .contains("ctts covers"));

        ctts.ctts = Some(vec![3, 0]);
        assert!(validate_table(&ctts, 100)
            .unwrap_err()
            .contains("zero sample_count"));
    }

    #[test]
    fn rejects_sample_range_past_end_of_file() {
        let mut table = TestTable::variable_stco();
        table.chunk_offsets[1] = 99;
        let error = validate_table(&table, 100).unwrap_err();
        assert!(error.contains("exceeds file size"), "{error}");
    }

    #[test]
    fn accepts_individually_bounded_reused_chunk_offsets() {
        let mut table = TestTable::variable_stco();
        table.sample_count = 2;
        table.variable_sizes = vec![60, 60];
        table.chunk_offsets = vec![10, 10];
        table.stsc = vec![StscFields {
            first_chunk: 1,
            samples_per_chunk: 1,
            sample_description_index: 1,
        }];
        table.stts = vec![2];

        // The aggregate is 120 bytes, but both legal table references are the
        // individually bounded range 10..70 in the same 100-byte file.
        validate_table(&table, 100).unwrap();
    }

    #[test]
    fn zero_size_sample_keeps_mapping_and_offset() {
        let mut table = TestTable::variable_stco();
        table.variable_sizes = vec![2, 0, 4];
        validate_table(&table, 100).unwrap();
        let descriptors = descriptors(&table).unwrap();
        assert_eq!(descriptors[1].offset, 12);
        assert_eq!(descriptors[2].offset, 20);
    }

    #[test]
    fn validates_decoded_geometry_before_appending() {
        let mut channels = vec![Vec::new(), Vec::new()];
        let mut total_frames = 0;
        append_decoded_frame(
            &mut channels,
            &DecodedFrame {
                pcm: vec![i16::MIN, i16::MAX, 0, 16_384],
                channels: 2,
                sample_rate: 48_000,
            },
            2,
            48_000,
            &mut total_frames,
        )
        .unwrap();
        assert_eq!(total_frames, 2);
        assert_eq!(channels[0], vec![-1.0, 0.0]);
        assert_eq!(channels[1], vec![32767.0 / 32768.0, 0.5]);

        for frame in [
            DecodedFrame {
                pcm: vec![0, 0],
                channels: 1,
                sample_rate: 48_000,
            },
            DecodedFrame {
                pcm: vec![0, 0],
                channels: 2,
                sample_rate: 44_100,
            },
            DecodedFrame {
                pcm: vec![0, 0, 0],
                channels: 2,
                sample_rate: 48_000,
            },
            DecodedFrame {
                pcm: vec![1],
                channels: 0,
                sample_rate: 48_000,
            },
        ] {
            let before = channels.clone();
            assert!(
                append_decoded_frame(&mut channels, &frame, 2, 48_000, &mut total_frames,).is_err()
            );
            assert_eq!(channels, before);
        }
    }

    #[test]
    fn rejects_missing_file() {
        assert!(decode_m4a(Path::new("/nonexistent/file.m4a")).is_err());
    }
}
