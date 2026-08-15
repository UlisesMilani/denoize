//! Bounded block decoders for long regular-file inputs.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use claxon::{Block, FlacReader, FlacReaderOptions};
use hound::{SampleFormat, WavSpec};
use symphonia::core::codecs::audio::well_known::CODEC_ID_VORBIS;
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use super::{AudioCodec, AudioFormat, DecodeBudget, DecodeLimits};
use crate::config::{MAX_SAMPLE_RATE, MAX_STREAM_BLOCK_FRAMES, MAX_STREAM_CHANNELS};
use crate::{AudioInputSession, ChannelMask, WavStreamReader};

const F64_BYTES: u64 = std::mem::size_of::<f64>() as u64;
const I32_BYTES: u64 = std::mem::size_of::<i32>() as u64;
const FLAC_ABSOLUTE_MAX_BLOCK_FRAMES: usize = u16::MAX as usize;

/// Geometry and conservative decoder accounting for a bounded input stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct AudioStreamInfo {
    /// Content-detected container family.
    pub format: AudioFormat,
    /// Content-detected codec.
    pub codec: AudioCodec,
    /// WAV representation used for the streamed output.
    pub output_spec: WavSpec,
    /// Speaker mask exposed by the input container, when available.
    pub channel_mask: Option<ChannelMask>,
    /// Declared playable frame count, when the container supplies one.
    pub total_frames: Option<u64>,
    /// Largest decoded codec block retained in addition to the caller block.
    pub max_decoder_frames: usize,
    /// Conservative bytes retained by the compressed decoder beyond the
    /// normal input, enhanced, and output blocks.
    pub decoder_additional_bytes: u64,
}

impl AudioStreamInfo {
    #[must_use]
    pub const fn channels(self) -> usize {
        self.output_spec.channels as usize
    }

    #[must_use]
    pub const fn sample_rate(self) -> u32 {
        self.output_spec.sample_rate
    }
}

/// Inspect one already-open regular input without decoding its audio frames.
///
/// WAV, native FLAC, and Ogg Vorbis are accepted. Metadata structures are
/// validated with the supplied finite limits before a compressed parser is
/// constructed.
pub fn inspect_audio_stream_session(
    session: &mut AudioInputSession,
    limits: DecodeLimits,
) -> Result<AudioStreamInfo, String> {
    let path = session.path().to_path_buf();
    let mut source = session.try_clone_rewound("inspect bounded audio stream")?;
    let format = super::detect_file_format_from_file(&path, false, &mut source)?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind stream input {}: {error}", path.display()))?;
    match format {
        AudioFormat::Wav => {
            let wav = crate::inspect_wav_session(session)?;
            Ok(AudioStreamInfo {
                format,
                codec: AudioCodec::Pcm,
                output_spec: wav.spec,
                channel_mask: wav.channel_mask,
                total_frames: Some(wav.total_frames),
                max_decoder_frames: 0,
                decoder_additional_bytes: 0,
            })
        }
        AudioFormat::Flac => inspect_flac(source, limits),
        AudioFormat::OggVorbis => inspect_ogg_vorbis(source, limits),
        other => Err(format!(
            "--stream does not support {other:?} input yet; use WAV, FLAC, or Ogg Vorbis"
        )),
    }
}

fn validate_stream_geometry(
    channels: usize,
    sample_rate: u32,
    context: &str,
) -> Result<(), String> {
    if channels == 0 || channels > MAX_STREAM_CHANNELS {
        return Err(format!(
            "{context} channel count must be between 1 and {MAX_STREAM_CHANNELS}"
        ));
    }
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return Err(format!(
            "{context} sample rate must be between 1 and {MAX_SAMPLE_RATE} Hz"
        ));
    }
    Ok(())
}

fn planar_bytes(channels: usize, frames: usize, bytes_per_sample: u64) -> Result<u64, String> {
    u64::try_from(channels)
        .ok()
        .and_then(|channels| channels.checked_mul(u64::try_from(frames).ok()?))
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or_else(|| "stream decoder byte count overflows".to_string())
}

fn channel_descriptor_bytes(channels: usize) -> Result<u64, String> {
    u64::try_from(channels)
        .ok()
        .and_then(|channels| channels.checked_mul(std::mem::size_of::<Vec<f64>>() as u64))
        .ok_or_else(|| "stream decoder channel descriptor count overflows".to_string())
}

fn float_output_spec(channels: usize, sample_rate: u32) -> Result<WavSpec, String> {
    validate_stream_geometry(channels, sample_rate, "stream input")?;
    Ok(WavSpec {
        channels: u16::try_from(channels)
            .map_err(|_| "stream channel count does not fit in WAV".to_string())?,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    })
}

fn inspect_flac(mut source: File, limits: DecodeLimits) -> Result<AudioStreamInfo, String> {
    crate::metadata::preflight_flac_decode(&mut source, limits.metadata)?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind FLAC stream after metadata validation: {error}"))?;
    let reader = FlacReader::new_ext(
        source,
        FlacReaderOptions {
            metadata_only: true,
            read_vorbis_comment: false,
        },
    )
    .map_err(|error| format!("inspect FLAC stream: {error}"))?;
    let stream = reader.streaminfo();
    let channels = usize::try_from(stream.channels)
        .map_err(|_| "FLAC channel count does not fit in memory".to_string())?;
    validate_stream_geometry(channels, stream.sample_rate, "FLAC stream")?;
    if !(4..=32).contains(&stream.bits_per_sample) {
        return Err(format!(
            "unsupported FLAC sample width: {} bits",
            stream.bits_per_sample
        ));
    }
    let decoded = planar_bytes(channels, FLAC_ABSOLUTE_MAX_BLOCK_FRAMES, I32_BYTES)?;
    let decoder_additional_bytes = decoded
        .checked_add(channel_descriptor_bytes(channels)?)
        .ok_or_else(|| "FLAC stream decoder byte count overflows".to_string())?;
    DecodeBudget::new(limits).check_peak(0, decoder_additional_bytes, "FLAC stream decoder")?;
    Ok(AudioStreamInfo {
        format: AudioFormat::Flac,
        codec: AudioCodec::Flac,
        output_spec: float_output_spec(channels, stream.sample_rate)?,
        channel_mask: None,
        total_frames: stream.samples,
        max_decoder_frames: FLAC_ABSOLUTE_MAX_BLOCK_FRAMES,
        decoder_additional_bytes,
    })
}

fn inspect_ogg_vorbis(mut source: File, limits: DecodeLimits) -> Result<AudioStreamInfo, String> {
    crate::metadata::preflight_ogg_decode(&mut source, limits.metadata)?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind Ogg Vorbis stream after metadata validation: {error}"))?;
    let mut fixed = [0_u8; 27];
    source
        .read_exact(&mut fixed)
        .map_err(|error| format!("read Ogg Vorbis identification page: {error}"))?;
    if &fixed[..4] != b"OggS" || fixed[4] != 0 || fixed[5] & 0x02 == 0 || fixed[5] & 0x01 != 0 {
        return Err("Ogg Vorbis stream does not begin with a complete BOS page".into());
    }
    let segment_count = usize::from(fixed[26]);
    let mut lacing = vec![0_u8; segment_count];
    source
        .read_exact(&mut lacing)
        .map_err(|error| format!("read Ogg Vorbis identification lacing: {error}"))?;
    let mut packet_len = 0usize;
    let mut complete = false;
    for length in lacing {
        packet_len = packet_len
            .checked_add(usize::from(length))
            .ok_or_else(|| "Ogg Vorbis identification packet size overflows".to_string())?;
        if length < 255 {
            complete = true;
            break;
        }
    }
    if !complete || packet_len < 30 || packet_len > limits.metadata.max_ogg_packet_bytes {
        return Err("Ogg Vorbis identification packet is incomplete or exceeds its limit".into());
    }
    let mut packet = Vec::new();
    packet
        .try_reserve_exact(packet_len)
        .map_err(|error| format!("reserve Ogg Vorbis identification packet: {error}"))?;
    packet.resize(packet_len, 0);
    source
        .read_exact(&mut packet)
        .map_err(|error| format!("read Ogg Vorbis identification packet: {error}"))?;
    if !packet.starts_with(b"\x01vorbis") || packet[7..11] != [0, 0, 0, 0] {
        return Err("invalid Ogg Vorbis identification header".into());
    }
    let channels = usize::from(packet[11]);
    let sample_rate = u32::from_le_bytes(packet[12..16].try_into().expect("fixed sample rate"));
    validate_stream_geometry(channels, sample_rate, "Ogg Vorbis stream")?;
    let block_sizes = packet[28];
    let small_power = block_sizes & 0x0f;
    let large_power = block_sizes >> 4;
    if !(6..=13).contains(&small_power)
        || !(6..=13).contains(&large_power)
        || small_power > large_power
        || packet[29] & 1 == 0
    {
        return Err("invalid Ogg Vorbis block sizes or framing flag".into());
    }
    let max_decoder_frames = 1usize << large_power;
    let decoded = planar_bytes(channels, max_decoder_frames, F64_BYTES)?;
    // Symphonia retains its typed packet buffer while denoize copies one
    // planar packet. Charge both representations plus the largest encoded
    // packet admitted by the structural preflight.
    let decoder_additional_bytes = decoded
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(limits.metadata.max_ogg_packet_bytes as u64))
        .and_then(|bytes| bytes.checked_add(channel_descriptor_bytes(channels).ok()?))
        .ok_or_else(|| "Ogg Vorbis stream decoder byte count overflows".to_string())?;
    DecodeBudget::new(limits).check_peak(
        0,
        decoder_additional_bytes,
        "Ogg Vorbis stream decoder",
    )?;
    Ok(AudioStreamInfo {
        format: AudioFormat::OggVorbis,
        codec: AudioCodec::Vorbis,
        output_spec: float_output_spec(channels, sample_rate)?,
        channel_mask: None,
        total_frames: None,
        max_decoder_frames,
        decoder_additional_bytes,
    })
}

/// A block-oriented decoder for WAV, FLAC, and Ogg Vorbis regular files.
pub struct AudioStreamReader {
    path: PathBuf,
    identity: File,
    info: AudioStreamInfo,
    inner: StreamReader,
}

enum StreamReader {
    Wav(WavStreamReader<BufReader<File>>),
    Flac(FlacStreamReader),
    Vorbis(VorbisStreamReader),
}

impl AudioStreamReader {
    /// Open and consume a validated regular-file session.
    pub fn from_session(
        mut session: AudioInputSession,
        limits: DecodeLimits,
    ) -> Result<Self, String> {
        let path = session.path().to_path_buf();
        let info = inspect_audio_stream_session(&mut session, limits)?;
        let source = session.into_file_rewound("open bounded audio stream")?;
        let identity = source
            .try_clone()
            .map_err(|error| format!("clone stream input {}: {error}", path.display()))?;
        let inner = match info.format {
            AudioFormat::Wav => StreamReader::Wav(WavStreamReader::from_file(source)?),
            AudioFormat::Flac => StreamReader::Flac(FlacStreamReader::new(source, info, limits)?),
            AudioFormat::OggVorbis => {
                StreamReader::Vorbis(VorbisStreamReader::new(&path, source, info, limits)?)
            }
            _ => unreachable!("stream inspection rejected unsupported input"),
        };
        Ok(Self {
            path,
            identity,
            info,
            inner,
        })
    }

    #[must_use]
    pub const fn info(&self) -> AudioStreamInfo {
        self.info
    }

    /// Read at most `max_frames` complete frames.
    pub fn next_block(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<f64>>>, String> {
        if !(1..=MAX_STREAM_BLOCK_FRAMES).contains(&max_frames) {
            return Err(format!(
                "stream block size must be between 1 and {MAX_STREAM_BLOCK_FRAMES} frames"
            ));
        }
        match &mut self.inner {
            StreamReader::Wav(reader) => reader.next_block(max_frames),
            StreamReader::Flac(reader) => reader.next_block(max_frames),
            StreamReader::Vorbis(reader) => reader.next_block(max_frames),
        }
    }

    /// Re-hash the exact opened input without reopening its pathname.
    pub fn fingerprint_input(&self) -> Result<crate::batch_resume::FileFingerprint, String> {
        crate::batch_resume::fingerprint_open_file_at(&self.identity, &self.path)
    }
}

fn empty_planar(channels: usize, frames: usize, context: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|error| format!("reserve {context} channel list: {error}"))?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|error| format!("reserve {context} samples: {error}"))?;
        output.push(channel);
    }
    Ok(output)
}

struct FlacStreamReader {
    reader: FlacReader<File>,
    info: AudioStreamInfo,
    pending: Option<Block>,
    pending_offset: usize,
    recycle: Vec<i32>,
    decoded_frames: u64,
    eof: bool,
}

impl FlacStreamReader {
    fn new(source: File, info: AudioStreamInfo, limits: DecodeLimits) -> Result<Self, String> {
        let reader = FlacReader::new_ext(
            source,
            FlacReaderOptions {
                metadata_only: false,
                read_vorbis_comment: false,
            },
        )
        .map_err(|error| format!("open FLAC stream: {error}"))?;
        let stream = reader.streaminfo();
        if stream.sample_rate != info.sample_rate()
            || usize::try_from(stream.channels).ok() != Some(info.channels())
            || stream.samples != info.total_frames
        {
            return Err("FLAC stream geometry changed between inspection and decode".into());
        }
        let max_samples = FLAC_ABSOLUTE_MAX_BLOCK_FRAMES
            .checked_mul(info.channels())
            .ok_or_else(|| "FLAC stream sample buffer size overflows".to_string())?;
        DecodeBudget::new(limits).check_peak(
            0,
            info.decoder_additional_bytes,
            "FLAC stream decoder",
        )?;
        let mut recycle = Vec::new();
        recycle
            .try_reserve_exact(max_samples)
            .map_err(|error| format!("reserve FLAC stream sample buffer: {error}"))?;
        Ok(Self {
            reader,
            info,
            pending: None,
            pending_offset: 0,
            recycle,
            decoded_frames: 0,
            eof: false,
        })
    }

    fn decode_next_block(&mut self) -> Result<bool, String> {
        if self.eof {
            return Ok(false);
        }
        let buffer = std::mem::take(&mut self.recycle);
        let next = self
            .reader
            .blocks()
            .read_next_or_eof(buffer)
            .map_err(|error| format!("decode FLAC stream block: {error}"))?;
        let Some(block) = next else {
            self.eof = true;
            if self
                .info
                .total_frames
                .is_some_and(|frames| frames != self.decoded_frames)
            {
                return Err(format!(
                    "FLAC stream decoded {} frames but STREAMINFO declared {}",
                    self.decoded_frames,
                    self.info.total_frames.unwrap_or(0)
                ));
            }
            return Ok(false);
        };
        if block.channels() as usize != self.info.channels() {
            return Err("FLAC channel count changed while streaming".into());
        }
        let duration = block.duration() as usize;
        if duration == 0 || duration > FLAC_ABSOLUTE_MAX_BLOCK_FRAMES {
            return Err("FLAC stream block has an invalid duration".into());
        }
        // Claxon exposes the encoded frame/sample number through `time()`,
        // but fixed-block streams may report the final short frame in units
        // derived from that shorter block. CRC validation and the declared
        // aggregate sample count remain authoritative for continuity here.
        self.decoded_frames = self
            .decoded_frames
            .checked_add(duration as u64)
            .ok_or_else(|| "FLAC stream frame count overflows".to_string())?;
        if self
            .info
            .total_frames
            .is_some_and(|frames| self.decoded_frames > frames)
        {
            return Err("FLAC stream exceeds its declared frame count".into());
        }
        self.pending = Some(block);
        self.pending_offset = 0;
        Ok(true)
    }

    fn next_block(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<f64>>>, String> {
        let mut output = empty_planar(self.info.channels(), max_frames, "FLAC stream block")?;
        while output[0].len() < max_frames {
            if self.pending.is_none() && !self.decode_next_block()? {
                break;
            }
            let pending = self.pending.as_ref().expect("decoded block is present");
            let available = pending.duration() as usize - self.pending_offset;
            let take = available.min(max_frames - output[0].len());
            let scale = 1.0
                / (1_u64 << (self.reader.streaminfo().bits_per_sample.saturating_sub(1))) as f64;
            for (index, destination) in output.iter_mut().enumerate() {
                let source = pending.channel(index as u32);
                destination.extend(
                    source[self.pending_offset..self.pending_offset + take]
                        .iter()
                        .map(|sample| crate::sanitize_sample(f64::from(*sample) * scale)),
                );
            }
            self.pending_offset += take;
            if self.pending_offset == pending.duration() as usize {
                let block = self.pending.take().expect("pending FLAC block");
                self.recycle = block.into_buffer();
                self.pending_offset = 0;
            }
        }
        if output[0].is_empty() {
            Ok(None)
        } else {
            Ok(Some(output))
        }
    }
}

struct VorbisStreamReader {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    info: AudioStreamInfo,
    limits: DecodeLimits,
    pending: Vec<Vec<f64>>,
    pending_offset: usize,
    eof: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_file, Audio, EncodeOptions};
    use base64::Engine as _;

    fn fixture(frames: usize) -> Audio {
        let left = (0..frames)
            .map(|frame| ((frame as f64 * 0.017).sin() * 0.7).clamp(-1.0, 1.0))
            .collect::<Vec<_>>();
        let right = (0..frames)
            .map(|frame| ((frame as f64 * 0.011).cos() * 0.5).clamp(-1.0, 1.0))
            .collect::<Vec<_>>();
        Audio {
            sample_rate: 48_000,
            channels: vec![left, right],
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
            channel_mask: None,
        }
    }

    fn collect(mut reader: AudioStreamReader, block_frames: usize) -> Vec<Vec<f64>> {
        let mut output = vec![Vec::new(); reader.info().channels()];
        while let Some(block) = reader.next_block(block_frames).expect("read stream block") {
            assert!(block[0].len() <= block_frames);
            for (destination, source) in output.iter_mut().zip(block) {
                destination.extend(source);
            }
        }
        output
    }

    #[test]
    fn flac_stream_matches_whole_file_decode_across_block_boundaries() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("input.flac");
        let input = fixture(20_123);
        crate::encode::write_audio(&path, &input, EncodeOptions::default())
            .expect("encode FLAC fixture");

        let session = AudioInputSession::open(&path).expect("open FLAC session");
        let reader = AudioStreamReader::from_session(session, DecodeLimits::default())
            .expect("open FLAC stream");
        assert_eq!(reader.info().format, AudioFormat::Flac);
        assert_eq!(reader.info().codec, AudioCodec::Flac);
        assert_eq!(reader.info().total_frames, Some(input.frames() as u64));
        let streamed = collect(reader, 257);
        let whole = decode_file(&path).expect("decode whole FLAC");
        assert_eq!(streamed.len(), whole.channels.len());
        for (streamed, whole) in streamed.iter().zip(&whole.channels) {
            assert_eq!(streamed.len(), whole.len());
            let error = streamed
                .iter()
                .zip(whole)
                .map(|(streamed, whole)| (streamed - whole).abs())
                .fold(0.0, f64::max);
            assert!(error <= f64::EPSILON, "stream/whole FLAC error {error}");
        }
    }

    #[test]
    fn flac_stream_decoder_allowance_has_an_exact_boundary() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("input.flac");
        crate::encode::write_audio(&path, &fixture(4_096), EncodeOptions::default())
            .expect("encode FLAC fixture");
        let mut session = AudioInputSession::open(&path).expect("open FLAC session");
        let info = inspect_audio_stream_session(&mut session, DecodeLimits::default())
            .expect("inspect FLAC stream");
        let exact =
            DecodeLimits::default().with_max_working_set_bytes(Some(info.decoder_additional_bytes));
        inspect_audio_stream_session(&mut session, exact).expect("exact decoder allowance");
        let error = inspect_audio_stream_session(
            &mut session,
            DecodeLimits::default()
                .with_max_working_set_bytes(Some(info.decoder_additional_bytes - 1)),
        )
        .expect_err("one byte below the decoder allowance must fail");
        assert!(error.contains("FLAC stream decoder"));
    }

    #[test]
    fn ogg_vorbis_stream_matches_whole_file_decode() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("input.ogg");
        let encoded = base64::engine::general_purpose::STANDARD
            .decode(include_str!("testdata/tiny-vorbis.ogg.b64").trim())
            .expect("decode embedded Ogg Vorbis fixture");
        std::fs::write(&path, encoded).expect("write Ogg Vorbis fixture");

        let session = AudioInputSession::open(&path).expect("open Ogg Vorbis session");
        let reader = AudioStreamReader::from_session(session, DecodeLimits::default())
            .expect("open Ogg Vorbis stream");
        assert_eq!(reader.info().format, AudioFormat::OggVorbis);
        assert_eq!(reader.info().codec, AudioCodec::Vorbis);
        assert_eq!(reader.info().sample_rate(), 16_000);
        assert_eq!(reader.info().channels(), 2);
        let streamed = collect(reader, 73);
        let whole = decode_file(&path).expect("decode whole Ogg Vorbis fixture");
        assert_eq!(streamed.len(), whole.channels.len());
        for (streamed, whole) in streamed.iter().zip(&whole.channels) {
            assert_eq!(streamed.len(), whole.len());
            let error = streamed
                .iter()
                .zip(whole)
                .map(|(streamed, whole)| (streamed - whole).abs())
                .fold(0.0, f64::max);
            assert!(error <= f64::EPSILON, "stream/whole Vorbis error {error}");
        }
    }

    #[test]
    fn unsupported_compressed_stream_is_rejected_during_inspection() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("input.opus");
        crate::encode::write_audio(&path, &fixture(960), EncodeOptions::default())
            .expect("encode Ogg Opus fixture");
        let mut session = AudioInputSession::open(&path).expect("open Ogg Opus session");
        let error = inspect_audio_stream_session(&mut session, DecodeLimits::default())
            .expect_err("Ogg Opus stream must remain unsupported until granule semantics land");
        assert!(error.contains("use WAV, FLAC, or Ogg Vorbis"));
    }

    #[cfg(unix)]
    #[test]
    fn stream_session_keeps_decoding_the_opened_inode_after_path_replacement() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("input.flac");
        let moved = directory.path().join("opened.flac");
        let original = fixture(4_321);
        crate::encode::write_audio(&path, &original, EncodeOptions::default())
            .expect("encode original FLAC");
        let session = AudioInputSession::open(&path).expect("open original stream session");
        std::fs::rename(&path, &moved).expect("move opened input pathname");
        crate::encode::write_audio(&path, &fixture(777), EncodeOptions::default())
            .expect("write replacement FLAC");

        let reader = AudioStreamReader::from_session(session, DecodeLimits::default())
            .expect("decode held original inode");
        let streamed = collect(reader, 113);
        assert_eq!(streamed[0].len(), original.frames());
        assert_eq!(streamed[1].len(), original.frames());
    }
}

impl VorbisStreamReader {
    fn new(
        path: &Path,
        source: File,
        info: AudioStreamInfo,
        limits: DecodeLimits,
    ) -> Result<Self, String> {
        DecodeBudget::new(limits).check_peak(
            0,
            info.decoder_additional_bytes,
            "Ogg Vorbis stream decoder",
        )?;
        let stream = MediaSourceStream::new(Box::new(source), Default::default());
        let mut hint = Hint::new();
        if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
            hint.with_extension(extension);
        }
        let format = symphonia::default::get_probe()
            .probe(
                &hint,
                stream,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|error| format!("probe Ogg Vorbis stream: {error}"))?;
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| "Ogg Vorbis stream has no audio track".to_string())?;
        let codec_params = track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .ok_or_else(|| "Ogg Vorbis stream has no audio codec parameters".to_string())?;
        if codec_params.codec != CODEC_ID_VORBIS
            || codec_params.sample_rate != Some(info.sample_rate())
            || codec_params.channels.as_ref().map(|value| value.count()) != Some(info.channels())
        {
            return Err("Ogg Vorbis codec geometry changed after native inspection".into());
        }
        let track_id = track.id;
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(codec_params, &AudioDecoderOptions::default())
            .map_err(|error| format!("open Ogg Vorbis decoder: {error}"))?;
        let pending = empty_planar(
            info.channels(),
            info.max_decoder_frames,
            "Ogg Vorbis packet",
        )?;
        Ok(Self {
            format,
            decoder,
            track_id,
            info,
            limits,
            pending,
            pending_offset: 0,
            eof: false,
        })
    }

    fn pending_capacity_bytes(&self) -> Result<u64, String> {
        let samples = self.pending.iter().try_fold(0_u64, |total, channel| {
            total
                .checked_add(channel.capacity() as u64)
                .ok_or_else(|| "Ogg Vorbis pending capacity overflows".to_string())
        })?;
        samples
            .checked_mul(F64_BYTES)
            .and_then(|bytes| bytes.checked_add(channel_descriptor_bytes(self.pending.len()).ok()?))
            .ok_or_else(|| "Ogg Vorbis pending byte count overflows".to_string())
    }

    fn decode_next_packet(&mut self) -> Result<bool, String> {
        if self.eof {
            return Ok(false);
        }
        loop {
            DecodeBudget::new(self.limits).check_peak(
                self.pending_capacity_bytes()?,
                self.info.decoder_additional_bytes,
                "Ogg Vorbis packet read",
            )?;
            let packet = match self.format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    self.eof = true;
                    return Ok(false);
                }
                Err(SymphoniaError::ResetRequired) => {
                    return Err("Ogg Vorbis stream requires an unsupported decoder reset".into())
                }
                Err(error) => return Err(format!("read Ogg Vorbis packet: {error}")),
            };
            if packet.data.len() > self.limits.metadata.max_ogg_packet_bytes {
                return Err("Ogg Vorbis packet exceeds the configured packet limit".into());
            }
            if packet.track_id != self.track_id {
                continue;
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(SymphoniaError::DecodeError(_)) | Err(SymphoniaError::IoError(_)) => continue,
                Err(SymphoniaError::ResetRequired) => {
                    return Err("Ogg Vorbis decoder requested an unsupported reset".into())
                }
                Err(error) => return Err(format!("decode Ogg Vorbis packet: {error}")),
            };
            if decoded.spec().rate() != self.info.sample_rate()
                || decoded.num_planes() != self.info.channels()
            {
                return Err("Ogg Vorbis geometry changed while streaming".into());
            }
            let frames = decoded.frames();
            if frames == 0 {
                continue;
            }
            if frames > self.info.max_decoder_frames
                || decoded.capacity() > self.info.max_decoder_frames
            {
                return Err(format!(
                    "Ogg Vorbis decoder packet exceeds the {}-frame bounded stream limit",
                    self.info.max_decoder_frames
                ));
            }
            for channel in &mut self.pending {
                channel.resize(frames, 0.0);
            }
            let mut destinations = self
                .pending
                .iter_mut()
                .map(Vec::as_mut_slice)
                .collect::<Vec<_>>();
            decoded.copy_to_slice_planar::<f64, _>(&mut destinations);
            drop(destinations);
            for channel in &mut self.pending {
                for sample in channel {
                    *sample = crate::sanitize_sample(*sample);
                }
            }
            self.pending_offset = 0;
            return Ok(true);
        }
    }

    fn next_block(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<f64>>>, String> {
        let mut output = empty_planar(self.info.channels(), max_frames, "Ogg Vorbis stream block")?;
        while output[0].len() < max_frames {
            let pending_frames = self.pending.first().map(Vec::len).unwrap_or(0);
            if self.pending_offset == pending_frames && !self.decode_next_packet()? {
                break;
            }
            let pending_frames = self.pending[0].len();
            let available = pending_frames - self.pending_offset;
            let take = available.min(max_frames - output[0].len());
            for (destination, source) in output.iter_mut().zip(&self.pending) {
                destination
                    .extend_from_slice(&source[self.pending_offset..self.pending_offset + take]);
            }
            self.pending_offset += take;
            if self.pending_offset == pending_frames {
                for channel in &mut self.pending {
                    channel.clear();
                }
                self.pending_offset = 0;
            }
        }
        if output[0].is_empty() {
            Ok(None)
        } else {
            Ok(Some(output))
        }
    }
}
