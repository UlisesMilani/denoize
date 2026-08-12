//! Audio file I/O: WAV read/write (`hound`) + unified decode for MP3/M4A/WAV.
//!
//! Decoded compressed audio is promoted to `f64` planar PCM at native sample rate
//! (see [`crate::decode`]) before denoising. WAV write preserves bit depth.

use crate::atomic_output::{AtomicOutput, CommitMode};
use crate::channel_layout::{ChannelLayout, ChannelMask, PanInfo};
use crate::config::{
    checked_stream_memory_bytes, ConfigError, MAX_STREAM_BLOCK_FRAMES, MAX_STREAM_STATE_BYTES,
};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

const BYTES_PER_MIB: u64 = 1024 * 1024;
const MIN_MEMORY_ESTIMATE_BYTES: u64 = BYTES_PER_MIB;
// RIFF stores the file size minus the leading eight bytes in a u32. Hound's
// classic PCM header contributes 36 bytes to that field, while its extensible
// header contributes 60 bytes.
const PCM_WAV_RIFF_OVERHEAD: u64 = 36;
const EXTENSIBLE_WAV_RIFF_OVERHEAD: u64 = 60;
// Hound's supported integer/float sample encodings occupy at most four bytes.
// `WavWriter` does not expose the container width chosen by `new_with_spec_ex`,
// so wrapping an arbitrary writer must account for this conservative maximum.
const MAX_WAV_CONTAINER_BYTES_PER_SAMPLE: u64 = 4;

/// Keep a decoded sample in the range accepted by the DSP and codecs.
///
/// Non-finite values cannot represent an audio sample and would otherwise
/// poison FFT state, noise estimates, and container encoders.  Treat them as
/// a short, safe silence instead of allowing them to propagate.
#[inline]
pub fn sanitize_sample(sample: f64) -> f64 {
    if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// In-memory audio: one `Vec<f64>` per channel, plus format metadata.
#[derive(Clone, Debug)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f64>>,
    pub bits_per_sample: u16,
    pub sample_format: SampleFormat,
    /// WAVE speaker mask, when the source container supplied one.
    pub channel_mask: Option<ChannelMask>,
}

impl Audio {
    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    pub fn frames(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Fallibly clone the planar samples so callers can preserve transactional
    /// semantics when later processing can still fail.
    pub(crate) fn try_clone_fallible(&self, context: &str) -> Result<Self, String> {
        let mut channels = Vec::new();
        channels
            .try_reserve_exact(self.channels.len())
            .map_err(|_| format!("unable to reserve {context} channels"))?;
        for source in &self.channels {
            let mut channel = Vec::new();
            channel
                .try_reserve_exact(source.len())
                .map_err(|_| format!("unable to reserve {context} samples"))?;
            channel.extend_from_slice(source);
            channels.push(channel);
        }
        Ok(Self {
            sample_rate: self.sample_rate,
            channels,
            bits_per_sample: self.bits_per_sample,
            sample_format: self.sample_format,
            channel_mask: self.channel_mask,
        })
    }

    /// Sanitize all in-memory samples in place.
    ///
    /// Finite values outside the normalized PCM range are clipped and NaN or
    /// infinity is replaced by silence.  The return value is the number of
    /// samples changed, which can be used for diagnostics without requiring a
    /// second pass.
    pub fn sanitize_samples(&mut self) -> usize {
        let mut changed = 0;
        for channel in &mut self.channels {
            for sample in channel {
                let sanitized = sanitize_sample(*sample);
                if *sample != sanitized || !sample.is_finite() {
                    changed += 1;
                }
                *sample = sanitized;
            }
        }
        changed
    }

    /// Return the conventional layout for the planar channel order.
    ///
    /// The channel count is always preserved by denoize's lossless paths.  A
    /// file-specific channel mask, when present, is additional metadata and
    /// does not change the PCM channel order inferred here.
    pub fn channel_layout(&self) -> ChannelLayout {
        self.channel_mask
            .filter(|mask| mask.channels() == self.channels())
            .map(ChannelLayout::from_channel_mask)
            .unwrap_or_else(|| ChannelLayout::from_channel_count(self.channels()))
    }

    /// Return the source mask, or the conventional mask for a known layout.
    pub fn effective_channel_mask(&self) -> Option<ChannelMask> {
        match self.channel_mask {
            Some(mask) if mask.bits() == 0 || mask.channels() == self.channels() => Some(mask),
            Some(_) => None,
            None => self.channel_layout().mask(),
        }
    }

    /// Return one speaker pan coordinate per planar channel.
    pub fn pan_info(&self) -> Option<Vec<PanInfo>> {
        self.effective_channel_mask().map(ChannelMask::pan)
    }

    /// A `WavSpec` matching this audio for writing.
    fn wav_spec(&self) -> WavSpec {
        WavSpec {
            channels: self.channels() as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: self.bits_per_sample,
            sample_format: self.sample_format,
        }
    }
}

/// Estimate the bytes occupied by decoded planar samples and their small
/// per-channel allocations.
///
/// This intentionally uses the vector lengths rather than the input file
/// size: compressed inputs can expand substantially when decoded, and WAV
/// files are promoted to `f64` samples before processing.
pub fn estimate_audio_memory_bytes(audio: &Audio) -> u64 {
    let samples = audio.channels.iter().fold(0u64, |total, channel| {
        total.saturating_add(channel.len() as u64)
    });
    samples
        .saturating_mul(std::mem::size_of::<f64>() as u64)
        .saturating_add((audio.channels.len() as u64).saturating_mul(256))
}

/// Estimate the in-memory working set for the normal (non-streaming) path.
///
/// Processing retains the decoded input while constructing a second set of
/// channel buffers and uses FFT scratch space. A conservative three-times
/// multiplier gives `--max-memory` a useful guard without pretending to be an
/// allocator-level measurement.
pub fn estimate_audio_working_set_bytes(audio: &Audio) -> u64 {
    estimate_audio_memory_bytes(audio)
        .saturating_mul(3)
        .max(MIN_MEMORY_ESTIMATE_BYTES)
}

/// Estimate the bounded working set of [`crate::denoiser::StreamingDenoiser`].
///
/// The stream keeps per-channel STFT state, a bounded profiling prefix, and a
/// current input/output block. The estimate is deliberately conservative and
/// scales with the configured block size rather than the total recording
/// length.
pub fn estimate_stream_memory_bytes(
    channels: usize,
    block_frames: usize,
    frame_size: usize,
    sample_rate: u32,
) -> u64 {
    checked_stream_memory_bytes(channels, block_frames, frame_size, sample_rate, 0.0)
        .unwrap_or(u64::MAX)
}

/// Checked, profile-aware estimate of a streaming working set.
///
/// Unlike [`estimate_stream_memory_bytes`], this entry point rejects invalid
/// dimensions, arithmetic overflow, and aggregate state-plus-block plans over
/// the streaming hard limit instead of saturating to `u64::MAX`.
pub fn estimate_stream_memory_bytes_checked(
    channels: usize,
    block_frames: usize,
    frame_size: usize,
    sample_rate: u32,
    profile_ms: f64,
) -> Result<u64, ConfigError> {
    checked_stream_memory_bytes(channels, block_frames, frame_size, sample_rate, profile_ms)
}

/// Conservative preflight estimate for a filesystem input.
///
/// The normal path decodes to planar `f64` and may retain multiple channel
/// buffers while processing. Eight times the encoded file size is therefore a
/// useful upper-bound heuristic for rejecting obviously oversized inputs
/// before decoding; a one-MiB floor keeps tiny files usable with a 1-MiB cap.
pub fn estimate_file_memory_bytes<P: AsRef<std::path::Path>>(path: P) -> Result<u64, String> {
    let size = std::fs::metadata(path.as_ref())
        .map_err(|error| format!("read input metadata: {error}"))?
        .len();
    Ok(size.saturating_mul(8).max(MIN_MEMORY_ESTIMATE_BYTES))
}

/// Enforce an optional memory cap in MiB, returning a user-facing diagnostic.
pub fn ensure_memory_limit(
    estimated_bytes: u64,
    max_memory_mb: Option<usize>,
    context: &str,
) -> Result<(), String> {
    let Some(max_memory_mb) = max_memory_mb else {
        return Ok(());
    };
    if max_memory_mb == 0 {
        return Err("--max-memory must be at least 1 MiB".into());
    }
    let max_memory_mb = u64::try_from(max_memory_mb)
        .map_err(|_| "--max-memory is too large for this platform".to_string())?;
    let limit = max_memory_mb
        .checked_mul(BYTES_PER_MIB)
        .ok_or_else(|| "--max-memory byte limit overflows u64".to_string())?;
    if estimated_bytes > limit {
        let estimated_mib = estimated_bytes
            .saturating_add(BYTES_PER_MIB - 1)
            .saturating_div(BYTES_PER_MIB);
        return Err(format!(
            "{context} requires approximately {estimated_mib} MiB, but --max-memory allows {max_memory_mb} MiB; use --stream for WAV or raise the limit"
        ));
    }
    Ok(())
}

/// Read any supported audio file (WAV, MP3, M4A) into de-interleaved `f64` channels.
///
/// Compressed formats are decoded losslessly to float precision (no rate conversion).
pub fn read_audio<P: AsRef<std::path::Path>>(path: P) -> Result<Audio, String> {
    read_audio_with_metadata_limits(path, crate::metadata::MetadataLimits::default())
}

/// Read any supported audio file while enforcing explicit FLAC/Ogg metadata
/// limits before the compressed decoder is allowed to parse the container.
pub fn read_audio_with_metadata_limits<P: AsRef<std::path::Path>>(
    path: P,
    metadata_limits: crate::metadata::MetadataLimits,
) -> Result<Audio, String> {
    let path = path.as_ref();
    // Keep the original WAV representation so WAV -> WAV processing preserves
    // integer/float sample format and bit depth. Compressed decoders do not
    // have equivalent PCM container metadata and are promoted to f32 PCM.
    let header = {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let mut header = [0u8; 12];
        let n = file.read(&mut header).map_err(|e| format!("read: {e}"))?;
        header[..n].to_vec()
    };
    if crate::decode::AudioFormat::detect(path, &header) == crate::decode::AudioFormat::Wav {
        return read_wav(path);
    }
    let pcm = crate::decode::decode_file_with_metadata_limits(path, metadata_limits)?;
    Ok(pcm.into_audio())
}

/// Read a WAV file into de-interleaved `f64` channels.
pub fn read_wav<P: AsRef<std::path::Path>>(path: P) -> Result<Audio, String> {
    let channel_mask = read_wav_channel_mask(path.as_ref())?;
    let reader = WavReader::open(&path).map_err(|e| format!("open: {e}"))?;
    read_wav_reader(reader, channel_mask)
}

/// Read WAV data supplied by a pipe or another in-memory source.
pub fn read_wav_bytes(bytes: Vec<u8>) -> Result<Audio, String> {
    let channel_mask = read_wav_channel_mask_bytes(&bytes)?;
    let reader = WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| format!("open: {e}"))?;
    read_wav_reader(reader, channel_mask)
}

fn read_wav_reader<R: std::io::Read>(
    mut reader: WavReader<R>,
    channel_mask: Option<ChannelMask>,
) -> Result<Audio, String> {
    let spec = reader.spec();
    validate_readable_wav_spec(spec)?;
    let nchan = spec.channels as usize;

    let mut channels: Vec<Vec<f64>> = (0..nchan).map(|_| Vec::new()).collect();

    match spec.sample_format {
        SampleFormat::Float => {
            let samples: Result<Vec<f32>, String> = reader
                .samples::<f32>()
                .map(|s| s.map_err(|e| format!("read: {e}")))
                .collect();
            for (i, v) in samples?.iter().enumerate() {
                channels[i % nchan].push(sanitize_sample(*v as f64));
            }
        }
        SampleFormat::Int => {
            let max = (1u64 << (spec.bits_per_sample - 1)) as f64; // 2^(bits-1)
            let inv = 1.0 / max;
            if spec.bits_per_sample <= 16 {
                let samples: Result<Vec<i16>, String> = reader
                    .samples::<i16>()
                    .map(|s| s.map_err(|e| format!("read: {e}")))
                    .collect();
                for (i, v) in samples?.iter().enumerate() {
                    channels[i % nchan].push(sanitize_sample(*v as f64 * inv));
                }
            } else {
                let samples: Result<Vec<i32>, String> = reader
                    .samples::<i32>()
                    .map(|s| s.map_err(|e| format!("read: {e}")))
                    .collect();
                for (i, v) in samples?.iter().enumerate() {
                    channels[i % nchan].push(sanitize_sample(*v as f64 * inv));
                }
            }
        }
    }

    Ok(Audio {
        sample_rate: spec.sample_rate,
        channels,
        bits_per_sample: spec.bits_per_sample,
        sample_format: spec.sample_format,
        channel_mask,
    })
}

fn validate_readable_wav_spec(spec: WavSpec) -> Result<(), String> {
    if spec.channels == 0 {
        return Err("0 channels".into());
    }
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 8 | 16 | 24 | 32) | (SampleFormat::Float, 32) => Ok(()),
        (SampleFormat::Int, bits) => Err(format!(
            "unsupported integer WAV bit depth: {bits} (supported: 8, 16, 24, 32)"
        )),
        (SampleFormat::Float, bits) => Err(format!(
            "unsupported float WAV bit depth: {bits} (supported: 32)"
        )),
    }
}

/// Read the optional WAVE_FORMAT_EXTENSIBLE speaker mask without relying on
/// hound's intentionally small `WavSpec` abstraction.
fn read_wav_channel_mask(path: &std::path::Path) -> Result<Option<ChannelMask>, String> {
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("open WAV header: {error}"))?;
    let mut header = [0u8; 12];
    if file.read_exact(&mut header).is_err() || &header[8..12] != b"WAVE" {
        return Ok(None);
    }
    loop {
        let mut chunk = [0u8; 8];
        match file.read_exact(&mut chunk) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(error) => return Err(format!("read WAV chunk header: {error}")),
        }
        let size = u32::from_le_bytes(chunk[4..8].try_into().expect("WAV chunk size")) as usize;
        if &chunk[..4] == b"fmt " {
            if size < 40 {
                return Ok(None);
            }
            if size > 1 << 20 {
                return Err("WAV fmt chunk is too large".into());
            }
            let mut body = vec![0u8; size];
            file.read_exact(&mut body)
                .map_err(|error| format!("read WAV fmt chunk: {error}"))?;
            return parse_wav_channel_mask_fmt(&body);
        }
        use std::io::{Seek, SeekFrom};
        let skip = size.saturating_add(size & 1);
        file.seek(SeekFrom::Current(
            i64::try_from(skip).map_err(|_| "WAV chunk is too large to seek".to_string())?,
        ))
        .map_err(|error| format!("skip WAV chunk: {error}"))?;
    }
}

fn read_wav_channel_mask_bytes(bytes: &[u8]) -> Result<Option<ChannelMask>, String> {
    if bytes.len() < 12 || &bytes[8..12] != b"WAVE" {
        return Ok(None);
    }
    let mut offset = 12usize;
    while offset.saturating_add(8) <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let body_start = offset + 8;
        let body_end = body_start.saturating_add(size);
        if body_end > bytes.len() {
            break;
        }
        if id == b"fmt " && size >= 40 {
            let body = &bytes[body_start..body_end];
            return parse_wav_channel_mask_fmt(body);
        }
        offset = body_end.saturating_add(size & 1);
    }
    Ok(None)
}

fn parse_wav_channel_mask_fmt(body: &[u8]) -> Result<Option<ChannelMask>, String> {
    if body.len() < 40 {
        return Ok(None);
    }
    let format_tag = u16::from_le_bytes(body[0..2].try_into().expect("WAV format tag"));
    if format_tag != 0xfffe {
        return Ok(None);
    }
    let channels = u16::from_le_bytes(body[2..4].try_into().expect("WAV channel count")) as usize;
    let mask_bits = u32::from_le_bytes(body[20..24].try_into().expect("WAV channel mask"));
    let mask = ChannelMask::from_bits(mask_bits)
        .ok_or_else(|| format!("WAV channel mask 0x{mask_bits:08x} is invalid"))?;
    if mask.bits() != 0 && mask.channels() != channels {
        return Err(format!(
            "WAV channel mask has {} positions but fmt declares {channels} channels",
            mask.channels()
        ));
    }
    Ok(Some(mask))
}

/// Write an [`Audio`] to a file; format is inferred from the extension (`.wav`, `.mp3`, `.m4a`).
pub fn write_audio<P: AsRef<std::path::Path>>(
    path: P,
    audio: &Audio,
    options: crate::encode::EncodeOptions,
) -> Result<(), String> {
    crate::encode::write_audio(path, audio, options)
}

/// Write an [`Audio`] to a WAV file, preserving its bit depth / format.
pub fn write_wav<P: AsRef<std::path::Path>>(path: P, audio: &Audio) -> Result<(), String> {
    let path = path.as_ref();
    crate::encode::OutputFormat::Wav
        .validate_config(audio, &crate::encode::EncodeOptions::default())?;
    let mut output = AtomicOutput::new(path)?;
    write_wav_to_file(output.file_mut(), audio)?;
    output.commit(CommitMode::Replace)
}

/// Write an [`Audio`] to an already-open WAV file without reopening its path.
///
/// The file is rewound and truncated before encoding. It must be open for
/// reading as well as writing when a multichannel speaker mask is required.
pub fn write_wav_to_file(file: &mut File, audio: &Audio) -> Result<(), String> {
    crate::encode::OutputFormat::Wav
        .validate_config(audio, &crate::encode::EncodeOptions::default())?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek WAV output: {e}"))?;
    file.set_len(0)
        .map_err(|e| format!("truncate WAV output: {e}"))?;
    let spec = audio.wav_spec();
    {
        let writer =
            WavWriter::new(BufWriter::new(&mut *file), spec).map_err(|e| format!("create: {e}"))?;
        write_wav_writer(writer, audio)?;
    }
    write_wav_channel_mask_to_file(file, audio.channels(), audio.effective_channel_mask())
}

/// Encode a complete WAV into memory for stdout and network transports.
pub fn write_wav_bytes(audio: &Audio) -> Result<Vec<u8>, String> {
    crate::encode::OutputFormat::Wav
        .validate_config(audio, &crate::encode::EncodeOptions::default())?;
    let mut bytes = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut bytes);
        let writer =
            WavWriter::new(cursor, audio.wav_spec()).map_err(|e| format!("create: {e}"))?;
        write_wav_writer(writer, audio)?;
    }
    patch_wav_channel_mask_bytes(&mut bytes, audio)?;
    Ok(bytes)
}

/// Set the WAVE speaker mask in an already-finalized multichannel WAV file.
/// This is used by the bounded streaming path, whose writer receives only a
/// `WavSpec` while the input mask is kept as lightweight header metadata.
pub fn write_wav_channel_mask(
    path: impl AsRef<std::path::Path>,
    channels: usize,
    channel_mask: Option<ChannelMask>,
) -> Result<(), String> {
    if channels <= 2 {
        return Ok(());
    }
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path.as_ref())
        .map_err(|error| format!("open WAV header for channel mask: {error}"))?;
    write_wav_channel_mask_to_file(&mut file, channels, channel_mask)
}

/// Set the WAVE speaker mask through an already-open file handle.
///
/// Keeping this operation on the original handle prevents the finalized WAV
/// from being replaced between encoding and the header update.
pub fn write_wav_channel_mask_to_file(
    file: &mut File,
    channels: usize,
    channel_mask: Option<ChannelMask>,
) -> Result<(), String> {
    if channels <= 2 {
        return Ok(());
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("seek WAV header for channel mask: {error}"))?;
    let mut header = [0u8; 44];
    file.read_exact(&mut header)
        .map_err(|error| format!("read WAV header for channel mask: {error}"))?;
    if &header[..4] != b"RIFF"
        || &header[8..12] != b"WAVE"
        || &header[12..16] != b"fmt "
        || u32::from_le_bytes(header[16..20].try_into().expect("WAV fmt size")) < 40
        || u16::from_le_bytes(header[20..22].try_into().expect("WAV format tag")) != 0xfffe
    {
        return Err("multichannel WAV output is not WAVE_FORMAT_EXTENSIBLE".into());
    }
    file.seek(SeekFrom::Start(40))
        .map_err(|error| format!("seek WAV channel mask: {error}"))?;
    let bits = channel_mask
        .filter(|mask| mask.bits() == 0 || mask.channels() == channels)
        .map_or(0, ChannelMask::bits);
    file.write_all(&bits.to_le_bytes())
        .map_err(|error| format!("write WAV channel mask: {error}"))?;
    file.flush()
        .map_err(|error| format!("flush WAV channel mask: {error}"))
}

fn patch_wav_channel_mask_bytes(bytes: &mut [u8], audio: &Audio) -> Result<(), String> {
    if audio.channels() <= 2 {
        return Ok(());
    }
    if bytes.len() < 44 || &bytes[12..16] != b"fmt " {
        return Err("WAV output has no fmt chunk to store channel mask".into());
    }
    let fmt_size = u32::from_le_bytes(bytes[16..20].try_into().expect("WAV fmt size"));
    if fmt_size < 40
        || u16::from_le_bytes(bytes[20..22].try_into().expect("WAV format tag")) != 0xfffe
    {
        return Err("multichannel WAV output is not WAVE_FORMAT_EXTENSIBLE".into());
    }
    let bits = audio
        .effective_channel_mask()
        .filter(|mask| mask.bits() == 0 || mask.channels() == audio.channels())
        .map_or(0, ChannelMask::bits);
    bytes[40..44].copy_from_slice(&bits.to_le_bytes());
    Ok(())
}

fn write_wav_writer<W: std::io::Write + std::io::Seek>(
    mut writer: WavWriter<W>,
    audio: &Audio,
) -> Result<(), String> {
    let nchan = audio.channels();
    let frames = audio.frames();

    match audio.sample_format {
        SampleFormat::Float => {
            for f in 0..frames {
                for ch in 0..nchan {
                    let v = sanitize_sample(audio.channels[ch].get(f).copied().unwrap_or(0.0));
                    writer
                        .write_sample(v as f32)
                        .map_err(|e| format!("write: {e}"))?;
                }
            }
        }
        SampleFormat::Int => {
            let max = (1i64 << (audio.bits_per_sample - 1)) as f64;
            let hi = (max - 1.0) as i64;
            let lo = -max as i64;
            if audio.bits_per_sample <= 16 {
                for f in 0..frames {
                    for ch in 0..nchan {
                        let v = sanitize_sample(audio.channels[ch].get(f).copied().unwrap_or(0.0));
                        let q = ((v * max).round() as i64).min(hi).max(lo);
                        writer
                            .write_sample(q as i16)
                            .map_err(|e| format!("write: {e}"))?;
                    }
                }
            } else {
                for f in 0..frames {
                    for ch in 0..nchan {
                        let v = sanitize_sample(audio.channels[ch].get(f).copied().unwrap_or(0.0));
                        let q = ((v * max).round() as i64).min(hi).max(lo);
                        writer
                            .write_sample(q as i32)
                            .map_err(|e| format!("write: {e}"))?;
                    }
                }
            }
        }
    }
    writer.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(())
}

/// Block-oriented WAV reader. Samples are returned as planar `f64` channels,
/// keeping at most `max_frames` frames in memory per call.
pub struct WavStreamReader<R: Read + Seek> {
    reader: WavReader<R>,
    spec: WavSpec,
    channel_mask: Option<ChannelMask>,
}

#[derive(Clone, Copy, Debug)]
struct WavStreamBlockPlan {
    max_samples: usize,
    peak_bytes: u64,
}

fn plan_wav_stream_block(channels: usize, max_frames: usize) -> Result<WavStreamBlockPlan, String> {
    if channels == 0 {
        return Err("stream WAV requires at least one channel".into());
    }
    if !(1..=MAX_STREAM_BLOCK_FRAMES).contains(&max_frames) {
        return Err(format!(
            "stream block size must be between 1 and {MAX_STREAM_BLOCK_FRAMES} frames"
        ));
    }
    let max_samples = max_frames
        .checked_mul(channels)
        .ok_or_else(|| "stream block sample count overflow".to_string())?;
    let max_samples_u64 = u64::try_from(max_samples)
        .map_err(|_| "stream block sample count is too large".to_string())?;
    let sample_bytes = max_samples_u64
        .checked_mul(std::mem::size_of::<f64>() as u64)
        .ok_or_else(|| "stream block byte count overflow".to_string())?;
    let channel_headers = u64::try_from(channels)
        .map_err(|_| "stream channel count is too large".to_string())?
        .checked_mul(std::mem::size_of::<Vec<f64>>() as u64)
        .ok_or_else(|| "stream channel header byte count overflow".to_string())?;
    let vector_headers =
        (std::mem::size_of::<Vec<f64>>() + std::mem::size_of::<Vec<Vec<f64>>>()) as u64;
    let peak_bytes = sample_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(channel_headers))
        .and_then(|bytes| bytes.checked_add(vector_headers))
        .ok_or_else(|| "stream block peak byte count overflow".to_string())?;
    if peak_bytes > MAX_STREAM_STATE_BYTES {
        return Err(format!(
            "stream block requires {peak_bytes} bytes while deinterleaving, limit is {MAX_STREAM_STATE_BYTES} bytes"
        ));
    }
    Ok(WavStreamBlockPlan {
        max_samples,
        peak_bytes,
    })
}

impl WavStreamReader<BufReader<std::fs::File>> {
    /// Open a filesystem WAV for bounded-memory reading.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let channel_mask = read_wav_channel_mask(path.as_ref())?;
        let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let reader = WavReader::new(BufReader::new(file)).map_err(|e| format!("open: {e}"))?;
        Self::from_reader_with_mask(reader, channel_mask)
    }
}

impl<R: Read + Seek> WavStreamReader<R> {
    /// Wrap an existing seekable WAV source.
    pub fn from_reader(reader: WavReader<R>) -> Result<Self, String> {
        Self::from_reader_with_mask(reader, None)
    }

    fn from_reader_with_mask(
        reader: WavReader<R>,
        channel_mask: Option<ChannelMask>,
    ) -> Result<Self, String> {
        let spec = reader.spec();
        validate_readable_wav_spec(spec)?;
        Ok(Self {
            reader,
            spec,
            channel_mask,
        })
    }

    pub fn spec(&self) -> WavSpec {
        self.spec
    }

    pub fn channel_mask(&self) -> Option<ChannelMask> {
        self.channel_mask
    }

    /// Read up to `max_frames`, returning `None` only at clean end-of-file.
    pub fn next_block(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<f64>>>, String> {
        let nchan = self.spec.channels as usize;
        let plan = plan_wav_stream_block(nchan, max_frames)?;
        debug_assert!(plan.peak_bytes <= MAX_STREAM_STATE_BYTES);

        // Reserve every allocation before advancing the streaming reader. If
        // memory pressure prevents the second representation from being
        // reserved, the caller can retry without losing an input block.
        let mut interleaved = Vec::new();
        interleaved
            .try_reserve_exact(plan.max_samples)
            .map_err(|_| "unable to reserve stream input block".to_string())?;
        let mut channels = Vec::new();
        channels
            .try_reserve_exact(nchan)
            .map_err(|_| "unable to reserve stream channel list".to_string())?;
        for _ in 0..nchan {
            let mut channel = Vec::new();
            channel
                .try_reserve_exact(max_frames)
                .map_err(|_| "unable to reserve stream channel block".to_string())?;
            channels.push(channel);
        }

        match self.spec.sample_format {
            SampleFormat::Float => {
                for sample in self.reader.samples::<f32>().take(plan.max_samples) {
                    interleaved.push(sanitize_sample(
                        sample.map_err(|e| format!("read: {e}"))? as f64
                    ));
                }
            }
            SampleFormat::Int if self.spec.bits_per_sample <= 16 => {
                let max = (1u64 << (self.spec.bits_per_sample - 1)) as f64;
                let inv = 1.0 / max;
                for sample in self.reader.samples::<i16>().take(plan.max_samples) {
                    interleaved.push(sanitize_sample(
                        sample.map_err(|e| format!("read: {e}"))? as f64 * inv,
                    ));
                }
            }
            SampleFormat::Int => {
                let max = (1u64 << (self.spec.bits_per_sample - 1)) as f64;
                let inv = 1.0 / max;
                for sample in self.reader.samples::<i32>().take(plan.max_samples) {
                    interleaved.push(sanitize_sample(
                        sample.map_err(|e| format!("read: {e}"))? as f64 * inv,
                    ));
                }
            }
        }
        if interleaved.is_empty() {
            return Ok(None);
        }
        if interleaved.len() % nchan != 0 {
            return Err("truncated WAV frame at end of input".into());
        }
        for (index, sample) in interleaved.into_iter().enumerate() {
            channels[index % nchan].push(sample);
        }
        Ok(Some(channels))
    }
}

/// Block-oriented WAV writer. The output format is fixed by the supplied
/// [`WavSpec`], so integer bit depth and floating-point WAVs are preserved.
pub struct WavStreamWriter<W: Write + Seek> {
    writer: WavWriter<W>,
    spec: WavSpec,
    bytes_per_sample: u64,
    data_bytes_written: u64,
    data_byte_limit: u64,
}

impl WavStreamWriter<BufWriter<std::fs::File>> {
    /// Create a filesystem WAV for bounded-memory writing.
    pub fn create<P: AsRef<std::path::Path>>(path: P, spec: WavSpec) -> Result<Self, String> {
        validate_wav_stream_spec(spec)?;
        let file = std::fs::File::create(path).map_err(|e| format!("create: {e}"))?;
        Self::from_sink(BufWriter::new(file), spec)
    }
}

impl<W: Write + Seek> WavStreamWriter<W> {
    /// Create a streaming WAV writer over an existing seekable sink.
    pub fn from_sink(sink: W, spec: WavSpec) -> Result<Self, String> {
        validate_wav_stream_spec(spec)?;
        let writer = WavWriter::new(sink, spec).map_err(|e| format!("create: {e}"))?;
        let bytes_per_sample = u64::from(spec.bits_per_sample / 8);
        Self::from_writer_with_accounting(
            writer,
            spec,
            bytes_per_sample,
            wav_stream_data_byte_limit(spec),
        )
    }

    /// Wrap an existing WAV sink.
    ///
    /// Hound does not expose an existing writer's container width. This
    /// constructor therefore charges every sample at the maximum supported
    /// four-byte width and assumes the larger extensible header. Writers made
    /// by [`Self::from_sink`] use their exact known layout instead.
    pub fn from_writer(writer: WavWriter<W>, spec: WavSpec) -> Result<Self, String> {
        validate_wav_stream_spec(spec)?;
        Self::from_writer_with_accounting(
            writer,
            spec,
            MAX_WAV_CONTAINER_BYTES_PER_SAMPLE,
            u64::from(u32::MAX) - EXTENSIBLE_WAV_RIFF_OVERHEAD,
        )
    }

    fn from_writer_with_accounting(
        writer: WavWriter<W>,
        spec: WavSpec,
        bytes_per_sample: u64,
        data_byte_limit: u64,
    ) -> Result<Self, String> {
        if writer.spec() != spec {
            return Err("WAV stream writer spec does not match the supplied spec".into());
        }
        let data_bytes_written = u64::from(writer.len())
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| "existing WAV stream data length overflow".to_string())?;
        if data_bytes_written > data_byte_limit {
            return Err("existing WAV stream data exceeds the RIFF container limit".into());
        }
        Ok(Self {
            writer,
            spec,
            bytes_per_sample,
            data_bytes_written,
            data_byte_limit,
        })
    }

    /// Write a planar block, interleaving it in the WAV container.
    pub fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        let nchan = self.spec.channels as usize;
        if channels.len() != nchan {
            return Err(format!("expected {nchan} channels, got {}", channels.len()));
        }
        let frames = channels.first().map(Vec::len).unwrap_or(0);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("stream blocks must have equal channel lengths".into());
        }
        let block_samples = frames
            .checked_mul(nchan)
            .ok_or_else(|| "WAV stream block sample count overflow".to_string())?;
        let block_bytes = u64::try_from(block_samples)
            .map_err(|_| "WAV stream block sample count is too large".to_string())?
            .checked_mul(self.bytes_per_sample)
            .ok_or_else(|| "WAV stream block byte count overflow".to_string())?;
        let next_data_bytes = self
            .data_bytes_written
            .checked_add(block_bytes)
            .ok_or_else(|| "WAV stream data length overflow".to_string())?;
        if next_data_bytes > self.data_byte_limit {
            return Err(format!(
                "WAV stream data would exceed the RIFF container limit of {} bytes",
                self.data_byte_limit
            ));
        }
        match self.spec.sample_format {
            SampleFormat::Float => {
                for frame in 0..frames {
                    for channel in channels {
                        self.write_sample(sanitize_sample(channel[frame]) as f32)?;
                    }
                }
            }
            SampleFormat::Int if self.spec.bits_per_sample <= 16 => {
                let max = (1i64 << (self.spec.bits_per_sample - 1)) as f64;
                let hi = (max - 1.0) as i64;
                let lo = -max as i64;
                for frame in 0..frames {
                    for channel in channels {
                        let value = sanitize_sample(channel[frame]);
                        let quantized = ((value * max).round() as i64).min(hi).max(lo);
                        self.write_sample(quantized as i16)?;
                    }
                }
            }
            SampleFormat::Int => {
                let max = (1i64 << (self.spec.bits_per_sample - 1)) as f64;
                let hi = (max - 1.0) as i64;
                let lo = -max as i64;
                for frame in 0..frames {
                    for channel in channels {
                        let value = sanitize_sample(channel[frame]);
                        let quantized = ((value * max).round() as i64).min(hi).max(lo);
                        self.write_sample(quantized as i32)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Finalize the WAV header and flush the underlying sink.
    pub fn finalize(self) -> Result<(), String> {
        self.writer.finalize().map_err(|e| format!("finalize: {e}"))
    }

    fn write_sample<S: hound::Sample>(&mut self, sample: S) -> Result<(), String> {
        self.writer
            .write_sample(sample)
            .map_err(|e| format!("write: {e}"))?;
        // The complete block was checked against `data_byte_limit` before its
        // first sample was written, so this increment cannot overflow.
        self.data_bytes_written += self.bytes_per_sample;
        Ok(())
    }
}

fn wav_stream_data_byte_limit(spec: WavSpec) -> u64 {
    let riff_overhead = if spec.channels > 2 || spec.bits_per_sample > 16 {
        EXTENSIBLE_WAV_RIFF_OVERHEAD
    } else {
        PCM_WAV_RIFF_OVERHEAD
    };
    u64::from(u32::MAX) - riff_overhead
}

fn validate_wav_stream_spec(spec: WavSpec) -> Result<(), String> {
    if spec.channels == 0 {
        return Err("WAV stream requires at least one channel".into());
    }
    if spec.sample_rate == 0 {
        return Err("WAV stream sample rate must be greater than zero".into());
    }
    let bytes_per_sample = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 8) => 1u64,
        (SampleFormat::Int, 16) => 2,
        (SampleFormat::Int, 24) => 3,
        (SampleFormat::Int, 32) | (SampleFormat::Float, 32) => 4,
        (SampleFormat::Int, bits) => {
            return Err(format!(
                "unsupported integer WAV bit depth: {bits} (supported: 8, 16, 24, 32)"
            ));
        }
        (SampleFormat::Float, bits) => {
            return Err(format!(
                "unsupported float WAV bit depth: {bits} (supported: 32)"
            ));
        }
    };
    let block_align = u64::from(spec.channels)
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| "WAV stream block alignment overflow".to_string())?;
    if block_align > u16::MAX as u64 {
        return Err("WAV stream block alignment exceeds the WAV header limit".into());
    }
    let byte_rate = u64::from(spec.sample_rate)
        .checked_mul(block_align)
        .ok_or_else(|| "WAV stream byte-rate overflow".to_string())?;
    if byte_rate > u32::MAX as u64 {
        return Err("WAV stream byte rate exceeds the WAV header limit".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("denoize_audio_{}_{}", std::process::id(), name));
        p
    }

    fn extensible_float_wav_with_valid_bits(valid_bits: u16) -> Vec<u8> {
        let audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.25]],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        };
        let mut bytes = write_wav_bytes(&audio).unwrap();
        let fmt = bytes
            .windows(4)
            .position(|window| window == b"fmt ")
            .expect("fmt chunk");
        let body = fmt + 8;
        assert_eq!(&bytes[body..body + 2], &0xfffeu16.to_le_bytes());
        bytes[body + 18..body + 20].copy_from_slice(&valid_bits.to_le_bytes());
        bytes
    }

    #[test]
    fn sanitize_samples_maps_nonfinite_and_extreme_values_to_safe_pcm() {
        let mut audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                2.0,
                -2.0,
                0.25,
            ]],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        };
        assert_eq!(audio.sanitize_samples(), 5);
        assert_eq!(audio.channels[0], vec![0.0, 0.0, 0.0, 1.0, -1.0, 0.25]);
    }

    #[test]
    fn finite_pcm_clipping_is_symmetric_and_idempotent() {
        let input = [-1.25, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25];
        let clipped: Vec<_> = input.iter().copied().map(sanitize_sample).collect();
        assert_eq!(clipped, [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0]);
        assert_eq!(
            clipped,
            clipped
                .iter()
                .copied()
                .map(sanitize_sample)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn wav_write_sanitizes_nonfinite_samples_and_supports_empty_frames() {
        let path = tmp("nonfinite.wav");
        let audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                2.0,
                -2.0,
                0.25,
            ]],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        };
        write_wav(&path, &audio).unwrap();
        let decoded = read_wav(&path).unwrap();
        assert_eq!(decoded.channels[0][..5], [0.0, 0.0, 0.0, 1.0, -1.0]);
        assert!((decoded.channels[0][5] - 0.25).abs() < 1e-6);
        std::fs::remove_file(&path).unwrap();

        let empty_path = tmp("empty.wav");
        let empty = Audio {
            sample_rate: 48_000,
            channels: vec![Vec::new()],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        };
        write_wav(&empty_path, &empty).unwrap();
        let decoded_empty = read_wav(&empty_path).unwrap();
        assert_eq!(decoded_empty.channels(), 1);
        assert_eq!(decoded_empty.frames(), 0);
        std::fs::remove_file(empty_path).unwrap();
    }

    #[test]
    fn wav_stream_writer_sanitizes_nonfinite_samples() {
        let path = tmp("stream_nonfinite.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavStreamWriter::create(&path, spec).unwrap();
        writer
            .write_block(&[vec![f64::NAN, f64::INFINITY, -2.0, 0.5]])
            .unwrap();
        writer.finalize().unwrap();
        let decoded = read_wav(&path).unwrap();
        assert_eq!(decoded.channels[0][..3], [0.0, 0.0, -1.0]);
        assert!((decoded.channels[0][3] - 0.5).abs() < 1e-6);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn wav_16bit_roundtrip() {
        let path = tmp("rt16.wav");
        let sr = 16000u32;
        let spec = WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        let mut signal = Vec::new();
        for i in 0..sr as usize {
            let v = (2.0 * std::f64::consts::PI * 220.0 * i as f64 / sr as f64).sin() * 0.5;
            signal.push(v);
            w.write_sample((v * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();

        let audio = read_wav(&path).unwrap();
        assert_eq!(audio.sample_rate, sr);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.frames(), sr as usize);
        for (i, &sig) in signal.iter().enumerate() {
            assert!((audio.channels[0][i] - sig).abs() < 1e-3, "@{i}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_audio_preserves_wav_format() {
        let path = tmp("preserve16.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        w.write_sample(123i16).unwrap();
        w.finalize().unwrap();

        let audio = read_audio(&path).unwrap();
        assert_eq!(audio.bits_per_sample, 16);
        assert_eq!(audio.sample_format, SampleFormat::Int);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wav_stream_reader_and_writer_roundtrip_blocks() {
        let input = tmp("stream_in.wav");
        let output = tmp("stream_out.wav");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&input, spec).unwrap();
        for frame in 0..257 {
            writer.write_sample((frame as i16).wrapping_mul(3)).unwrap();
            writer
                .write_sample(-(frame as i16).wrapping_mul(2))
                .unwrap();
        }
        writer.finalize().unwrap();

        let mut reader = WavStreamReader::open(&input).unwrap();
        assert_eq!(reader.spec(), spec);
        assert!(reader.next_block(0).unwrap_err().contains("between 1"));
        assert!(reader
            .next_block(MAX_STREAM_BLOCK_FRAMES + 1)
            .unwrap_err()
            .contains("stream block size"));
        let first = reader.next_block(100).unwrap().unwrap();
        assert_eq!(
            first.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![100, 100]
        );
        let second = reader.next_block(100).unwrap().unwrap();
        assert_eq!(
            second.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![100, 100]
        );
        let third = reader.next_block(100).unwrap().unwrap();
        assert_eq!(third.iter().map(Vec::len).collect::<Vec<_>>(), vec![57, 57]);
        assert!(reader.next_block(100).unwrap().is_none());

        let mut stream_writer = WavStreamWriter::create(&output, spec).unwrap();
        stream_writer.write_block(&first).unwrap();
        stream_writer.write_block(&second).unwrap();
        stream_writer.write_block(&third).unwrap();
        stream_writer.finalize().unwrap();
        let roundtrip = read_wav(&output).unwrap();
        assert_eq!(roundtrip.frames(), 257);
        assert_eq!(roundtrip.channels(), 2);
        assert!((roundtrip.channels[0][120] - (360.0 / 32_768.0)).abs() < 1e-5);
        assert!((roundtrip.channels[1][120] + (240.0 / 32_768.0)).abs() < 1e-5);

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn extensible_valid_bits_are_rejected_before_sample_conversion() {
        let bytes = extensible_float_wav_with_valid_bits(65);
        let error = read_wav_bytes(bytes.clone()).unwrap_err();
        assert!(error.contains("unsupported float WAV bit depth: 65"));

        let reader = WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        let error = WavStreamReader::from_reader(reader).err().unwrap();
        assert!(error.contains("unsupported float WAV bit depth: 65"));

        let int_twelve = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 12,
            sample_format: SampleFormat::Int,
        };
        assert!(validate_readable_wav_spec(int_twelve).is_err());
    }

    #[test]
    fn stream_block_plan_counts_planar_vector_headers() {
        let normal = plan_wav_stream_block(2, 8_192).unwrap();
        assert!(normal.peak_bytes <= MAX_STREAM_STATE_BYTES);

        // The two f64 payloads alone are 8 KiB below the hard limit. The
        // per-channel Vec headers must still make this plan fail closed.
        let error = plan_wav_stream_block(u16::MAX as usize, 512).unwrap_err();
        assert!(error.contains("while deinterleaving"));
    }

    #[test]
    fn stream_writer_preflights_riff_limit_and_existing_writer_state() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer =
            WavStreamWriter::from_sink(std::io::Cursor::new(Vec::new()), spec).unwrap();
        assert_eq!(writer.bytes_per_sample, 2);
        assert_eq!(
            writer.data_byte_limit,
            u64::from(u32::MAX) - PCM_WAV_RIFF_OVERHEAD
        );
        writer.data_bytes_written = writer.data_byte_limit - 1;
        let underlying_samples = writer.writer.len();
        let error = writer.write_block(&[vec![0.0]]).unwrap_err();
        assert!(error.contains("RIFF container limit"));
        assert_eq!(writer.writer.len(), underlying_samples);

        let mut existing = WavWriter::new(std::io::Cursor::new(Vec::new()), spec).unwrap();
        existing.write_sample(1i16).unwrap();
        let wrapped = WavStreamWriter::from_writer(existing, spec).unwrap();
        assert_eq!(wrapped.bytes_per_sample, MAX_WAV_CONTAINER_BYTES_PER_SAMPLE);
        assert_eq!(wrapped.data_bytes_written, 4);
        assert_eq!(
            wrapped.data_byte_limit,
            u64::from(u32::MAX) - EXTENSIBLE_WAV_RIFF_OVERHEAD
        );
        wrapped.finalize().unwrap();

        let padded_spec = WavSpec {
            bits_per_sample: 24,
            ..spec
        };
        let mut padded = WavWriter::new_with_spec_ex(
            std::io::Cursor::new(Vec::new()),
            hound::WavSpecEx {
                spec: padded_spec,
                bytes_per_sample: 4,
            },
        )
        .unwrap();
        padded.write_sample(1i32).unwrap();
        let padded = WavStreamWriter::from_writer(padded, padded_spec).unwrap();
        assert_eq!(padded.data_bytes_written, 4);
        padded.finalize().unwrap();

        let actual = WavWriter::new(std::io::Cursor::new(Vec::new()), spec).unwrap();
        let mismatched = WavSpec {
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            ..spec
        };
        let error = WavStreamWriter::from_writer(actual, mismatched)
            .err()
            .unwrap();
        assert!(error.contains("does not match"));
    }

    #[test]
    fn invalid_wav_configuration_is_rejected_before_replacing_or_truncating_output() {
        let path = tmp("invalid_preflight.wav");
        std::fs::write(&path, b"existing output").unwrap();
        let invalid_spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Float,
        };
        assert!(WavStreamWriter::create(&path, invalid_spec).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"existing output");

        let invalid_audio = Audio {
            sample_rate: 0,
            channels: vec![vec![0.0]],
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
            channel_mask: None,
        };
        assert!(write_wav(&path, &invalid_audio).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"existing output");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn memory_estimates_scale_with_audio_and_stream_blocks() {
        let small = Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_000]],
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
            channel_mask: None,
        };
        let large = Audio {
            channels: vec![vec![0.0; 2_000], vec![0.0; 2_000]],
            ..small.clone()
        };
        assert!(estimate_audio_memory_bytes(&small) > 0);
        assert!(estimate_audio_memory_bytes(&large) > estimate_audio_memory_bytes(&small));
        assert!(estimate_audio_working_set_bytes(&large) >= estimate_audio_memory_bytes(&large));
        assert!(
            estimate_stream_memory_bytes(2, 4_096, 2_048, 48_000)
                > estimate_stream_memory_bytes(2, 1_024, 2_048, 48_000)
        );

        let without_profile =
            estimate_stream_memory_bytes_checked(2, 4_096, 2_048, 48_000, -1.0).unwrap();
        let with_profile =
            estimate_stream_memory_bytes_checked(2, 4_096, 2_048, 48_000, 10_000.0).unwrap();
        assert!(with_profile > without_profile);
    }

    #[test]
    fn checked_stream_estimate_rejects_overflow_and_oversized_working_sets() {
        assert!(matches!(
            estimate_stream_memory_bytes_checked(1, usize::MAX, 2_048, 48_000, -1.0),
            Err(ConfigError::InvalidValue {
                field: "block_frames",
                ..
            })
        ));
        assert!(estimate_stream_memory_bytes_checked(1, 0, 2_048, 48_000, -1.0).is_err());
        assert!(matches!(
            estimate_stream_memory_bytes_checked(64, 1_048_576, 2_048, 48_000, -1.0),
            Err(ConfigError::ResourceLimitExceeded { .. })
        ));
    }

    #[test]
    fn reports_standard_surround_layouts_without_mixing_channels() {
        let audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; 2]; 6],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        };
        assert_eq!(
            audio.channel_layout(),
            crate::channel_layout::ChannelLayout::FivePointOne
        );
        assert_eq!(audio.channels(), audio.channel_layout().channels());
    }

    #[test]
    fn multichannel_wav_roundtrip_preserves_explicit_speaker_mask() {
        let path = tmp("mask_roundtrip.wav");
        let mask = ChannelMask::from_bits(
            ChannelMask::FRONT_LEFT
                | ChannelMask::FRONT_RIGHT
                | ChannelMask::FRONT_CENTER
                | ChannelMask::LFE1
                | ChannelMask::SIDE_LEFT
                | ChannelMask::SIDE_RIGHT,
        )
        .unwrap();
        let audio = Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0, 0.1]; 6],
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
            channel_mask: Some(mask),
        };
        write_wav(&path, &audio).unwrap();
        let decoded = read_wav(&path).unwrap();
        assert_eq!(decoded.channel_mask, Some(mask));
        assert_eq!(decoded.channel_layout(), ChannelLayout::Unknown(6));
        let pan = decoded.pan_info().unwrap();
        assert_eq!(pan.len(), 6);
        assert_eq!(pan[4].azimuth_degrees, -90.0);
        let bytes = write_wav_bytes(&audio).unwrap();
        assert_eq!(read_wav_bytes(bytes).unwrap().channel_mask, Some(mask));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn memory_limit_reports_clear_overflow() {
        let error = ensure_memory_limit(2 * 1024 * 1024, Some(1), "decoded audio").unwrap_err();
        assert!(error.contains("decoded audio"));
        assert!(error.contains("--max-memory allows 1 MiB"));
        ensure_memory_limit(1024, Some(1), "decoded audio").unwrap();
        ensure_memory_limit(2 * 1024 * 1024, None, "decoded audio").unwrap();
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn memory_limit_rejects_byte_conversion_overflow() {
        let error = ensure_memory_limit(0, Some(usize::MAX), "decoded audio").unwrap_err();
        assert!(error.contains("overflows u64"), "unexpected error: {error}");
    }
}
