use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use hound::{SampleFormat, WavSpec};

use crate::audio::WavStreamWriter;
use crate::channel_layout::ChannelMask;

use super::pcm::StreamPcmLayout;
use super::{EncodeOptions, OutputFormat};

const IO_BUFFER_BYTES: u64 = 64 * 1024;
const MP3_PRIVATE_ALLOWANCE_BYTES: u64 = 8 * 1024 * 1024;
const OPUS_PRIVATE_ALLOWANCE_BYTES: u64 = 8 * 1024 * 1024;
const FLAC_PRIVATE_ALLOWANCE_BYTES: u64 = 32 * 1024 * 1024;
const AAC_PRIVATE_ALLOWANCE_BYTES: u64 = 128 * 1024 * 1024;
const DEFAULT_MAX_AUXILIARY_TEMPORARY_BYTES: u64 = 1024 * 1024 * 1024;
const AAC_LC_FRAME_FRAMES: u64 = 1024;
const M4A_TABLE_RECORD_BYTES: u64 = 12;
const STREAM_CONTAINER_ALLOWANCE_BYTES: u64 = 1024 * 1024;
const FLAC_MAX_BYTES_PER_SAMPLE: u64 = 8;
const MP3_MAX_FRAME_BYTES: u64 = 2 * 1024;
const MP3_MIN_FRAMES_PER_PACKET: u64 = 576;
const OPUS_FRAME_FRAMES: u64 = 960;
const OGG_OPUS_MAX_PACKET_WITH_CONTAINER_BYTES: u64 = 8 * 1024;
const DEFAULT_SPOOL_REPLAY_FRAMES: usize = 8_192;

/// Duration-independent temporary-file bounds used by stream encoders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct StreamEncodeLimits {
    max_auxiliary_temporary_bytes: u64,
}

impl StreamEncodeLimits {
    #[must_use]
    pub const fn new(max_auxiliary_temporary_bytes: u64) -> Self {
        Self {
            max_auxiliary_temporary_bytes,
        }
    }

    #[must_use]
    pub const fn max_auxiliary_temporary_bytes(self) -> u64 {
        self.max_auxiliary_temporary_bytes
    }
}

impl Default for StreamEncodeLimits {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_AUXILIARY_TEMPORARY_BYTES)
    }
}

/// PCM geometry used to configure a block-oriented output encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamEncodeSpec {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub sample_format: SampleFormat,
    pub channel_mask: Option<ChannelMask>,
    /// Exact presentation frames when known before encoding.
    pub total_frames: Option<u64>,
}

/// Geometry observed while validating a completed private stream output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamOutputVerification {
    pub format: crate::AudioFormat,
    pub codec: crate::AudioCodec,
    pub sample_rate: u32,
    pub channels: u16,
    pub presentation_frames: u64,
}

impl StreamEncodeSpec {
    #[must_use]
    pub const fn new(
        wav: WavSpec,
        channel_mask: Option<ChannelMask>,
        total_frames: Option<u64>,
    ) -> Self {
        Self {
            sample_rate: wav.sample_rate,
            channels: wav.channels,
            bits_per_sample: wav.bits_per_sample,
            sample_format: wav.sample_format,
            channel_mask,
            total_frames,
        }
    }

    #[must_use]
    pub const fn wav_spec(self) -> WavSpec {
        WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: self.bits_per_sample,
            sample_format: self.sample_format,
        }
    }

    pub(crate) fn validate_structure(self) -> Result<(), String> {
        if self.channels == 0 {
            return Err("stream output requires at least one channel".into());
        }
        if self.sample_rate == 0 {
            return Err("stream output sample rate must be greater than zero".into());
        }
        if let Some(mask) = self.channel_mask {
            if mask.bits() != 0 && mask.channels() != self.channels as usize {
                return Err(format!(
                    "stream output channel mask describes {} channels, but PCM has {}",
                    mask.channels(),
                    self.channels
                ));
            }
        }
        Ok(())
    }
}

pub(super) fn validate_stream_config(
    format: OutputFormat,
    spec: StreamEncodeSpec,
    options: EncodeOptions,
) -> Result<(), String> {
    options.validate_options(format)?;
    spec.validate_structure()?;
    match format {
        OutputFormat::Wav => {
            crate::audio::validate_wav_stream_spec(spec.wav_spec())?;
            if let Some(frames) = spec.total_frames {
                let bytes_per_sample = u64::from(spec.bits_per_sample / 8);
                let data_bytes = frames
                    .checked_mul(u64::from(spec.channels))
                    .and_then(|samples| samples.checked_mul(bytes_per_sample))
                    .ok_or_else(|| "WAV stream data length overflows".to_string())?;
                super::validate_wav_container_size(
                    data_bytes,
                    u64::from(spec.channels),
                    spec.bits_per_sample,
                )?;
            }
            Ok(())
        }
        OutputFormat::Flac => {
            super::flac::validate_geometry(
                spec.sample_rate,
                spec.channels as usize,
                spec.bits_per_sample,
            )?;
            reject_known_empty(spec, "FLAC")
        }
        OutputFormat::OggOpus => {
            reject_known_empty(spec, "Opus")?;
            if spec.sample_rate > crate::config::MAX_SAMPLE_RATE {
                return Err(format!(
                    "Opus encode: unsupported source sample rate {} Hz (supported: 1..={})",
                    spec.sample_rate,
                    crate::config::MAX_SAMPLE_RATE
                ));
            }
            let layout =
                StreamPcmLayout::new(spec.channels as usize, spec.channel_mask, options.downmix)?;
            crate::resample::validate_resampler_plan(
                layout.output().count as usize,
                spec.sample_rate,
                48_000,
            )
        }
        OutputFormat::Mp3 => {
            reject_known_empty(spec, "MP3")?;
            let layout =
                StreamPcmLayout::new(spec.channels as usize, spec.channel_mask, options.downmix)?;
            super::mp3::effective_mp3_stream_config(
                spec.sample_rate,
                layout.output(),
                options.mp3_bitrate_kbps,
            )
            .map(|_| ())
        }
        OutputFormat::M4a | OutputFormat::AacAdts => {
            let codec = if format == OutputFormat::M4a {
                "M4A"
            } else {
                "AAC"
            };
            reject_known_empty(spec, codec)?;
            let layout =
                StreamPcmLayout::new(spec.channels as usize, spec.channel_mask, options.downmix)?;
            if !super::AAC_ENCODER_SAMPLE_RATES.contains(&spec.sample_rate) {
                return Err(format!(
                    "{codec} encode: unsupported sample rate {} Hz (AAC standard rates only)",
                    spec.sample_rate
                ));
            }
            #[cfg(feature = "fdk-aac-encoder")]
            if options.aac_encoder == super::AacEncoder::Fdk {
                super::validate_fdk_aac_config(
                    layout.output().count as usize,
                    spec.sample_rate,
                    options.m4a_bitrate_bps,
                )?;
            }
            let _ = layout;
            Ok(())
        }
    }
}

fn reject_known_empty(spec: StreamEncodeSpec, codec: &str) -> Result<(), String> {
    if spec.total_frames == Some(0) {
        Err(format!("{codec} output requires at least one frame"))
    } else {
        Ok(())
    }
}

/// Conservative denoize-owned and explicitly-accounted codec bytes retained
/// by a stream encoder in addition to the caller's input/enhanced blocks.
pub fn estimate_stream_encode_additional_bytes(
    format: OutputFormat,
    spec: StreamEncodeSpec,
    block_frames: usize,
    options: EncodeOptions,
) -> Result<u64, String> {
    validate_stream_config(format, spec, options)?;
    let layout = if matches!(
        format,
        OutputFormat::OggOpus | OutputFormat::Mp3 | OutputFormat::M4a | OutputFormat::AacAdts
    ) {
        Some(StreamPcmLayout::new(
            spec.channels as usize,
            spec.channel_mask,
            options.downmix,
        )?)
    } else {
        None
    };
    let output_channels = layout.as_ref().map_or(spec.channels as usize, |layout| {
        layout.output().count as usize
    });
    let block_samples = checked_samples(block_frames, output_channels)?;
    let conversion_bytes = block_samples
        .checked_mul(std::mem::size_of::<f64>() as u64)
        .ok_or_else(|| "stream encoder conversion byte count overflows".to_string())?;
    let fixed = match format {
        OutputFormat::Wav => IO_BUFFER_BYTES,
        OutputFormat::Flac => FLAC_PRIVATE_ALLOWANCE_BYTES,
        OutputFormat::Mp3 => MP3_PRIVATE_ALLOWANCE_BYTES,
        OutputFormat::OggOpus => OPUS_PRIVATE_ALLOWANCE_BYTES
            .checked_add(crate::resample::resampler_plan_bytes(
                output_channels,
                spec.sample_rate,
                48_000,
            )?)
            .ok_or_else(|| "Opus stream encoder byte count overflows".to_string())?,
        OutputFormat::M4a | OutputFormat::AacAdts => AAC_PRIVATE_ALLOWANCE_BYTES,
    };
    fixed
        .checked_add(IO_BUFFER_BYTES)
        .and_then(|bytes| bytes.checked_add(conversion_bytes))
        .ok_or_else(|| "stream encoder byte count overflows".to_string())
}

/// Conservative peak bytes used while decoding a completed private output for
/// pre-publication verification.
///
/// Encoding and verification are sequential phases. Callers should reserve
/// the greater of this value and their live input/backend/encoder phase rather
/// than summing both phases.
pub fn estimate_stream_output_verification_bytes(
    format: OutputFormat,
    spec: StreamEncodeSpec,
    block_frames: usize,
    options: EncodeOptions,
    encode_limits: StreamEncodeLimits,
    decode_limits: crate::DecodeLimits,
) -> Result<u64, String> {
    validate_stream_config(format, spec, options)?;
    if block_frames == 0 || block_frames > crate::config::MAX_STREAM_BLOCK_FRAMES {
        return Err(format!(
            "stream output verification block size must be between 1 and {} frames",
            crate::config::MAX_STREAM_BLOCK_FRAMES
        ));
    }
    let channels = if matches!(format, OutputFormat::Wav | OutputFormat::Flac) {
        usize::from(spec.channels)
    } else {
        StreamPcmLayout::new(spec.channels as usize, spec.channel_mask, options.downmix)?
            .output()
            .count as usize
    };
    let block_bytes = checked_samples(block_frames, channels)?
        .checked_mul(std::mem::size_of::<f64>() as u64)
        .ok_or_else(|| "stream verification block byte count overflows".to_string())?;
    let descriptors = u64::try_from(channels)
        .ok()
        .and_then(|channels| channels.checked_mul(std::mem::size_of::<Vec<f64>>() as u64))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<Vec<f64>>>() as u64))
        .ok_or_else(|| "stream verification descriptor byte count overflows".to_string())?;
    let add = |left: u64, right: u64, context: &str| {
        left.checked_add(right)
            .ok_or_else(|| format!("{context} byte count overflows"))
    };
    let verification = match format {
        OutputFormat::Wav => add(
            block_bytes
                .checked_mul(2)
                .ok_or_else(|| "WAV verification byte count overflows".to_string())?,
            descriptors,
            "WAV verification",
        )?,
        OutputFormat::Flac => {
            let decoded = checked_samples(65_535, channels)?
                .checked_mul(std::mem::size_of::<i32>() as u64)
                .ok_or_else(|| "FLAC verification byte count overflows".to_string())?;
            add(
                add(decoded, block_bytes, "FLAC verification")?,
                descriptors,
                "FLAC verification",
            )?
        }
        OutputFormat::Mp3 => {
            let packet_pcm = checked_samples(1_152, channels)?
                .checked_mul(std::mem::size_of::<f64>() as u64)
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| "MP3 verification byte count overflows".to_string())?;
            let id3 = u64::try_from(decode_limits.metadata.max_total_bytes)
                .map_err(|_| "MP3 metadata limit does not fit in u64".to_string())?
                .checked_mul(4)
                .ok_or_else(|| "MP3 metadata verification byte count overflows".to_string())?;
            [
                packet_pcm,
                block_bytes,
                descriptors,
                id3,
                2 * 1024,
                128 * 1024,
            ]
            .into_iter()
            .try_fold(0_u64, |total, bytes| add(total, bytes, "MP3 verification"))?
        }
        OutputFormat::OggOpus => {
            let scratch = checked_samples(5_760, channels)?
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| "Opus verification scratch byte count overflows".to_string())?;
            let page_pcm = checked_samples(5_760 * 255, channels)?
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| "Opus verification page byte count overflows".to_string())?;
            let packet = u64::try_from(decode_limits.metadata.max_ogg_packet_bytes)
                .map_err(|_| "Opus packet limit does not fit in u64".to_string())?;
            let packet_buffers = packet
                .checked_mul(2)
                .ok_or_else(|| "Opus packet verification byte count overflows".to_string())?;
            [
                scratch,
                page_pcm,
                block_bytes,
                descriptors,
                packet_buffers,
                255 * 255,
                256 * 1024,
            ]
            .into_iter()
            .try_fold(0_u64, |total, bytes| add(total, bytes, "Opus verification"))?
        }
        OutputFormat::AacAdts | OutputFormat::M4a => {
            let access_unit_bytes = if format == OutputFormat::M4a
                && options.aac_encoder == super::AacEncoder::Fdk
            {
                #[cfg(feature = "fdk-aac-encoder")]
                {
                    let layout = StreamPcmLayout::new(
                        spec.channels as usize,
                        spec.channel_mask,
                        options.downmix,
                    )?;
                    let (frame_length, _) = super::m4a_fdk::stream_timing(
                        layout.output().count as usize,
                        spec.sample_rate,
                        options.m4a_bitrate_bps,
                    )?;
                    fdk_access_unit_ceiling(
                        layout.output().count as usize,
                        spec.sample_rate,
                        frame_length,
                        options.m4a_bitrate_bps,
                    )?
                }
                #[cfg(not(feature = "fdk-aac-encoder"))]
                {
                    return Err(
                        "FDK-AAC is unavailable in this build; rebuild with --features fdk-aac-encoder"
                            .into(),
                    );
                }
            } else {
                oxide_adts_frame_ceiling(spec.sample_rate, options.m4a_bitrate_bps)?
                    .saturating_sub(7)
            };
            let decoder = access_unit_bytes
                .checked_mul(64 * 1024)
                .and_then(|bytes| bytes.checked_add(AAC_PRIVATE_ALLOWANCE_BYTES))
                .ok_or_else(|| "AAC verification decoder byte count overflows".to_string())?;
            let decoded_packet = checked_samples(2_048, channels)?
                .checked_mul((std::mem::size_of::<i16>() + std::mem::size_of::<f64>()) as u64)
                .ok_or_else(|| "AAC verification frame byte count overflows".to_string())?;
            let parser = if format == OutputFormat::M4a {
                let table =
                    estimate_stream_encode_temporary_bytes(format, spec, options, encode_limits)?;
                let metadata = u64::try_from(decode_limits.metadata.max_total_bytes)
                    .map_err(|_| "M4A metadata limit does not fit in u64".to_string())?;
                let metadata = metadata
                    .checked_mul(4)
                    .ok_or_else(|| "M4A metadata verification byte count overflows".to_string())?;
                table
                    .checked_mul(8)
                    .and_then(|bytes| bytes.checked_add(metadata))
                    .and_then(|bytes| bytes.checked_add(STREAM_CONTAINER_ALLOWANCE_BYTES))
                    .ok_or_else(|| "M4A verification parser byte count overflows".to_string())?
            } else {
                0
            };
            [
                decoder,
                access_unit_bytes,
                decoded_packet,
                block_bytes,
                descriptors,
                parser,
            ]
            .into_iter()
            .try_fold(0_u64, |total, bytes| add(total, bytes, "AAC verification"))?
        }
    };
    Ok(verification)
}

/// Temporary bytes retained outside the destination staging file.
///
/// M4A stores fixed-width sample-size/offset records in an anonymous bounded
/// spool. A known presentation length yields the exact record bytes; an
/// unknown length reserves the configured spool ceiling. Other encoders do
/// not retain duration-sized auxiliary files and return zero.
pub fn estimate_stream_encode_temporary_bytes(
    format: OutputFormat,
    spec: StreamEncodeSpec,
    options: EncodeOptions,
    limits: StreamEncodeLimits,
) -> Result<u64, String> {
    validate_stream_config(format, spec, options)?;
    if format != OutputFormat::M4a {
        return Ok(0);
    }
    let Some(frames) = spec.total_frames else {
        return Ok(limits.max_auxiliary_temporary_bytes);
    };
    let (sample_duration, encoder_delay) = if options.aac_encoder == super::AacEncoder::Fdk {
        #[cfg(feature = "fdk-aac-encoder")]
        {
            let layout =
                StreamPcmLayout::new(spec.channels as usize, spec.channel_mask, options.downmix)?;
            super::m4a_fdk::stream_timing(
                layout.output().count as usize,
                spec.sample_rate,
                options.m4a_bitrate_bps,
            )?
        }
        #[cfg(not(feature = "fdk-aac-encoder"))]
        {
            return Err(
                "FDK-AAC is unavailable in this build; rebuild with --features fdk-aac-encoder"
                    .into(),
            );
        }
    } else {
        (AAC_LC_FRAME_FRAMES as u32, AAC_LC_FRAME_FRAMES)
    };
    let media_frames = frames
        .checked_add(encoder_delay)
        .ok_or_else(|| "M4A sample-table duration overflows".to_string())?;
    let access_units = media_frames.div_ceil(u64::from(sample_duration));
    let required = access_units
        .checked_mul(M4A_TABLE_RECORD_BYTES)
        .ok_or_else(|| "M4A sample-table byte count overflows".to_string())?;
    if required > limits.max_auxiliary_temporary_bytes {
        return Err(format!(
            "M4A sample table requires {required} bytes, exceeding its {}-byte limit",
            limits.max_auxiliary_temporary_bytes
        ));
    }
    Ok(required)
}

/// Conservative staged-file bytes for a stream whose presentation length is
/// known, or `None` when the input container does not expose that length.
///
/// Bounds use each configured encoder's actual packet ceiling rather than a
/// bitrate-average guess. Callers should add metadata and M4A auxiliary-table
/// bytes, then still inspect the staged file before atomic publication.
pub fn estimate_stream_encode_output_bytes(
    format: OutputFormat,
    spec: StreamEncodeSpec,
    options: EncodeOptions,
    limits: StreamEncodeLimits,
) -> Result<Option<u64>, String> {
    validate_stream_config(format, spec, options)?;
    let Some(frames) = spec.total_frames else {
        return Ok(None);
    };
    let channels = u64::from(spec.channels);
    let bytes = match format {
        OutputFormat::Wav => frames
            .checked_mul(channels)
            .and_then(|samples| samples.checked_mul(u64::from(spec.bits_per_sample / 8)))
            .and_then(|data| data.checked_add(68))
            .ok_or_else(|| "WAV stream output byte count overflows".to_string())?,
        OutputFormat::Flac => frames
            .checked_mul(channels)
            .and_then(|samples| samples.checked_mul(FLAC_MAX_BYTES_PER_SAMPLE))
            .and_then(|data| data.checked_add(STREAM_CONTAINER_ALLOWANCE_BYTES))
            .ok_or_else(|| "FLAC stream output byte count overflows".to_string())?,
        OutputFormat::Mp3 => frames
            .div_ceil(MP3_MIN_FRAMES_PER_PACKET)
            .checked_add(4)
            .and_then(|packets| packets.checked_mul(MP3_MAX_FRAME_BYTES))
            .and_then(|data| data.checked_add(STREAM_CONTAINER_ALLOWANCE_BYTES))
            .ok_or_else(|| "MP3 stream output byte count overflows".to_string())?,
        OutputFormat::OggOpus => {
            let resampled_frames = rounded_resampled_frames(frames, spec.sample_rate, 48_000)?;
            resampled_frames
                .div_ceil(OPUS_FRAME_FRAMES)
                .checked_add(2)
                .and_then(|packets| packets.checked_mul(OGG_OPUS_MAX_PACKET_WITH_CONTAINER_BYTES))
                .and_then(|data| data.checked_add(STREAM_CONTAINER_ALLOWANCE_BYTES))
                .ok_or_else(|| "Ogg Opus stream output byte count overflows".to_string())?
        }
        OutputFormat::AacAdts => {
            let access_units = frames
                .div_ceil(AAC_LC_FRAME_FRAMES)
                .checked_add(1)
                .ok_or_else(|| "ADTS AAC stream access-unit count overflows".to_string())?;
            access_units
                .checked_mul(oxide_adts_frame_ceiling(
                    spec.sample_rate,
                    options.m4a_bitrate_bps,
                )?)
                .and_then(|data| data.checked_add(STREAM_CONTAINER_ALLOWANCE_BYTES))
                .ok_or_else(|| "ADTS AAC stream output byte count overflows".to_string())?
        }
        OutputFormat::M4a => {
            let table_bytes =
                estimate_stream_encode_temporary_bytes(format, spec, options, limits)?;
            let access_units = table_bytes / M4A_TABLE_RECORD_BYTES;
            let access_unit_bytes = if options.aac_encoder == super::AacEncoder::Fdk {
                #[cfg(feature = "fdk-aac-encoder")]
                {
                    let layout = StreamPcmLayout::new(
                        spec.channels as usize,
                        spec.channel_mask,
                        options.downmix,
                    )?;
                    let (frame_length, _) = super::m4a_fdk::stream_timing(
                        layout.output().count as usize,
                        spec.sample_rate,
                        options.m4a_bitrate_bps,
                    )?;
                    fdk_access_unit_ceiling(
                        layout.output().count as usize,
                        spec.sample_rate,
                        frame_length,
                        options.m4a_bitrate_bps,
                    )?
                }
                #[cfg(not(feature = "fdk-aac-encoder"))]
                {
                    return Err(
                        "FDK-AAC is unavailable in this build; rebuild with --features fdk-aac-encoder"
                            .into(),
                    );
                }
            } else {
                oxide_adts_frame_ceiling(spec.sample_rate, options.m4a_bitrate_bps)?
                    .saturating_sub(7)
            };
            access_units
                .checked_mul(access_unit_bytes)
                .and_then(|data| data.checked_add(table_bytes))
                .and_then(|data| data.checked_add(STREAM_CONTAINER_ALLOWANCE_BYTES))
                .ok_or_else(|| "M4A stream output byte count overflows".to_string())?
        }
    };
    Ok(Some(bytes))
}

/// Conservative anonymous-spool bytes for a non-seekable output sink.
///
/// The spooled writer retains interleaved `f64` PCM until final geometry is
/// known, then encodes into a second anonymous file and validates that file
/// before copying any bytes to the caller's `Write` sink. Both live spools and
/// codec auxiliary data are included here.
pub fn estimate_spooled_stream_output_bytes(
    format: OutputFormat,
    spec: StreamEncodeSpec,
    options: EncodeOptions,
    limits: StreamEncodeLimits,
) -> Result<Option<u64>, String> {
    validate_stream_config(format, spec, options)?;
    spec.total_frames
        .map(|frames| {
            estimate_spooled_stream_output_for_frames(format, spec, frames, options, limits)
        })
        .transpose()
}

fn estimate_spooled_stream_output_for_frames(
    format: OutputFormat,
    mut spec: StreamEncodeSpec,
    frames: u64,
    options: EncodeOptions,
    limits: StreamEncodeLimits,
) -> Result<u64, String> {
    spec.total_frames = Some(frames);
    let pcm = frames
        .checked_mul(u64::from(spec.channels))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| "non-seekable PCM spool byte count overflows".to_string())?;
    let encoded = estimate_stream_encode_output_bytes(format, spec, options, limits)?
        .ok_or_else(|| "known non-seekable output length produced no encoded bound".to_string())?;
    let auxiliary = estimate_stream_encode_temporary_bytes(format, spec, options, limits)?;
    pcm.checked_add(encoded)
        .and_then(|bytes| bytes.checked_add(auxiliary))
        .ok_or_else(|| "non-seekable output spool byte count overflows".to_string())
}

fn rounded_resampled_frames(frames: u64, from_rate: u32, to_rate: u32) -> Result<u64, String> {
    let numerator = u128::from(frames)
        .checked_mul(u128::from(to_rate))
        .and_then(|value| value.checked_add(u128::from(from_rate) / 2))
        .ok_or_else(|| "stream resampled frame count overflows".to_string())?;
    u64::try_from(numerator / u128::from(from_rate))
        .map_err(|_| "stream resampled frame count exceeds u64".to_string())
}

fn expected_stream_output_frames(
    format: OutputFormat,
    source_rate: u32,
    encoded_input_frames: u64,
) -> Result<u64, String> {
    match format {
        OutputFormat::Wav | OutputFormat::Flac | OutputFormat::M4a => Ok(encoded_input_frames),
        OutputFormat::OggOpus => {
            rounded_resampled_frames(encoded_input_frames, source_rate, 48_000)
        }
        OutputFormat::Mp3 => {
            // MPEG-1 Layer III carries 1,152 PCM frames per packet; MPEG-2 and
            // 2.5 carry 576. The Shine writer emits at least two complete
            // packets so even a tiny file is independently demuxable.
            let packet_frames = if source_rate >= 32_000 { 1_152 } else { 576 };
            let minimum = packet_frames * 2;
            Ok(encoded_input_frames.max(minimum).div_ceil(packet_frames) * packet_frames)
        }
        OutputFormat::AacAdts => {
            // Raw ADTS cannot signal codec delay or final padding. Oxide emits
            // one delayed access unit after the padded source access units.
            encoded_input_frames
                .div_ceil(AAC_LC_FRAME_FRAMES)
                .checked_add(1)
                .and_then(|units| units.checked_mul(AAC_LC_FRAME_FRAMES))
                .ok_or_else(|| "ADTS AAC presentation frame count overflows".to_string())
        }
    }
}

fn expected_stream_output_identity(
    format: OutputFormat,
) -> (crate::AudioFormat, crate::AudioCodec) {
    match format {
        OutputFormat::Wav => (crate::AudioFormat::Wav, crate::AudioCodec::Pcm),
        OutputFormat::Flac => (crate::AudioFormat::Flac, crate::AudioCodec::Flac),
        OutputFormat::OggOpus => (crate::AudioFormat::OggOpus, crate::AudioCodec::Opus),
        OutputFormat::Mp3 => (crate::AudioFormat::Mp3, crate::AudioCodec::Mp3),
        OutputFormat::M4a => (crate::AudioFormat::M4a, crate::AudioCodec::Aac),
        OutputFormat::AacAdts => (crate::AudioFormat::AacAdts, crate::AudioCodec::Aac),
    }
}

/// Decode and validate a completed private stream output before publication.
///
/// The supplied handle is authoritative; `display_path` is used only in error
/// messages, so a concurrent pathname replacement cannot redirect validation.
/// Validation walks the complete encoded stream with the bounded block decoder
/// and rejects container/codec, channel, sample-rate, or presentation-duration
/// drift. The caller may then fingerprint and atomically publish the same open
/// file. The file cursor is unspecified on return.
pub fn verify_stream_output_file(
    file: &mut File,
    display_path: &Path,
    format: OutputFormat,
    spec: StreamEncodeSpec,
    encoded_input_frames: u64,
    options: EncodeOptions,
    limits: crate::DecodeLimits,
    block_frames: usize,
) -> Result<StreamOutputVerification, String> {
    validate_stream_config(format, spec, options)?;
    if block_frames == 0 || block_frames > crate::config::MAX_STREAM_BLOCK_FRAMES {
        return Err(format!(
            "stream output verification block size must be between 1 and {} frames",
            crate::config::MAX_STREAM_BLOCK_FRAMES
        ));
    }
    if spec
        .total_frames
        .is_some_and(|declared| declared != encoded_input_frames)
    {
        return Err(format!(
            "stream output verification received {encoded_input_frames} frames, but the encoder declared {}",
            spec.total_frames.unwrap_or(0)
        ));
    }

    file.flush()
        .map_err(|error| format!("flush staged stream output: {error}"))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind staged stream output: {error}"))?;
    let mut source = file
        .try_clone()
        .map_err(|error| format!("clone staged stream output: {error}"))?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind cloned staged stream output: {error}"))?;
    let session =
        crate::AudioInputSession::from_open_file(display_path, source, "staged stream output")?;
    let mut reader = crate::AudioStreamReader::from_session(session, limits)?;
    let info = reader.info();
    let (expected_format, expected_codec) = expected_stream_output_identity(format);
    if info.format != expected_format || info.codec != expected_codec {
        return Err(format!(
            "staged stream output identifies as {:?}/{:?}, expected {:?}/{:?}",
            info.format, info.codec, expected_format, expected_codec
        ));
    }

    let expected_channels = if matches!(format, OutputFormat::Wav | OutputFormat::Flac) {
        spec.channels
    } else {
        u16::from(
            StreamPcmLayout::new(spec.channels as usize, spec.channel_mask, options.downmix)?
                .output()
                .count,
        )
    };
    let expected_sample_rate = if format == OutputFormat::OggOpus {
        48_000
    } else {
        spec.sample_rate
    };
    if info.output_spec.channels != expected_channels
        || info.output_spec.sample_rate != expected_sample_rate
    {
        return Err(format!(
            "staged stream output geometry is {}ch at {} Hz, expected {expected_channels}ch at {expected_sample_rate} Hz",
            info.output_spec.channels, info.output_spec.sample_rate
        ));
    }

    let mut presentation_frames = 0_u64;
    while let Some(block) = reader.next_block(block_frames)? {
        if block.len() != usize::from(expected_channels) {
            return Err("staged stream output channel count changed while decoding".into());
        }
        let frames = block.first().map_or(0, Vec::len);
        if frames == 0 || block.iter().any(|channel| channel.len() != frames) {
            return Err("staged stream output decoder returned an invalid block".into());
        }
        presentation_frames = presentation_frames
            .checked_add(frames as u64)
            .ok_or_else(|| "staged stream output frame count overflows".to_string())?;
    }
    let expected_frames =
        expected_stream_output_frames(format, spec.sample_rate, encoded_input_frames)?;
    if presentation_frames != expected_frames {
        return Err(format!(
            "staged stream output decodes to {presentation_frames} presentation frames, expected {expected_frames}"
        ));
    }
    Ok(StreamOutputVerification {
        format: info.format,
        codec: info.codec,
        sample_rate: info.output_spec.sample_rate,
        channels: info.output_spec.channels,
        presentation_frames,
    })
}

fn oxide_adts_frame_ceiling(sample_rate: u32, bitrate_bps: u32) -> Result<u64, String> {
    let bits = u64::from(bitrate_bps)
        .checked_mul(AAC_LC_FRAME_FRAMES)
        .ok_or_else(|| "AAC per-frame bitrate budget overflows".to_string())?
        / u64::from(sample_rate);
    Ok((bits / 8).max(23))
}

#[cfg(feature = "fdk-aac-encoder")]
fn fdk_access_unit_ceiling(
    channels: usize,
    sample_rate: u32,
    frame_length: u32,
    bitrate_bps: u32,
) -> Result<u64, String> {
    let nominal_bits = u64::from(bitrate_bps)
        .checked_mul(u64::from(frame_length))
        .and_then(|bits| bits.checked_add(u64::from(sample_rate) - 1))
        .ok_or_else(|| "FDK-AAC per-frame bitrate budget overflows".to_string())?
        / u64::from(sample_rate);
    let reservoir_bits = u64::try_from(channels)
        .ok()
        .and_then(|channels| channels.checked_mul(6_144))
        .ok_or_else(|| "FDK-AAC reservoir budget overflows".to_string())?;
    Ok(nominal_bits.max(reservoir_bits).div_ceil(8))
}

fn checked_samples(frames: usize, channels: usize) -> Result<u64, String> {
    u64::try_from(frames)
        .ok()
        .and_then(|frames| frames.checked_mul(u64::try_from(channels).ok()?))
        .ok_or_else(|| "stream encoder sample count overflows".to_string())
}

/// One seekable block-oriented output transaction.
///
/// The writer borrows the caller's private staging sink. [`finalize`](Self::finalize)
/// completes container headers and releases that borrow before metadata or an
/// atomic publication step is applied by the caller.
pub struct AudioStreamWriter<'a, W: Write + Seek> {
    inner: StreamWriterInner<'a, W>,
    spec: StreamEncodeSpec,
    frames_written: u64,
}

enum StreamWriterInner<'a, W: Write + Seek> {
    Wav(WavStreamWriter<BufWriter<&'a mut W>>),
    Flac(super::flac::FlacStreamWriter<&'a mut W>),
    OggOpus(super::opus::OggOpusStreamWriter<&'a mut W>),
    Mp3(super::mp3::Mp3StreamWriter<&'a mut W>),
    #[cfg(feature = "m4a-encode")]
    M4a(super::m4a::M4aStreamWriter<&'a mut W>),
    #[cfg(feature = "fdk-aac-encoder")]
    M4aFdk(super::m4a_fdk::FdkM4aStreamWriter<&'a mut W>),
    #[cfg(feature = "m4a-encode")]
    AacAdts(super::aac::AdtsAacStreamWriter<&'a mut W>),
}

impl<'a, W: Write + Seek> AudioStreamWriter<'a, W> {
    pub fn new(
        sink: &'a mut W,
        format: OutputFormat,
        spec: StreamEncodeSpec,
        options: EncodeOptions,
    ) -> Result<Self, String> {
        Self::new_with_limits(sink, format, spec, options, StreamEncodeLimits::default())
    }

    pub fn new_with_limits(
        sink: &'a mut W,
        format: OutputFormat,
        spec: StreamEncodeSpec,
        options: EncodeOptions,
        limits: StreamEncodeLimits,
    ) -> Result<Self, String> {
        validate_stream_config(format, spec, options)?;
        estimate_stream_encode_temporary_bytes(format, spec, options, limits)?;
        let inner = match format {
            OutputFormat::Wav => StreamWriterInner::Wav(WavStreamWriter::from_sink(
                BufWriter::new(sink),
                spec.wav_spec(),
            )?),
            OutputFormat::Flac => StreamWriterInner::Flac(super::flac::FlacStreamWriter::new(
                sink,
                spec.sample_rate,
                spec.channels as usize,
                spec.bits_per_sample,
            )?),
            OutputFormat::OggOpus => {
                StreamWriterInner::OggOpus(super::opus::OggOpusStreamWriter::new(
                    sink,
                    spec.sample_rate,
                    spec.channels as usize,
                    spec.channel_mask,
                    128_000,
                    options.downmix,
                )?)
            }
            OutputFormat::Mp3 => StreamWriterInner::Mp3(super::mp3::Mp3StreamWriter::new(
                sink,
                spec.sample_rate,
                spec.channels as usize,
                spec.channel_mask,
                options.mp3_bitrate_kbps,
                options.downmix,
            )?),
            OutputFormat::M4a => {
                #[cfg(feature = "m4a-encode")]
                {
                    if options.aac_encoder == super::AacEncoder::Fdk {
                        #[cfg(feature = "fdk-aac-encoder")]
                        {
                            StreamWriterInner::M4aFdk(super::m4a_fdk::FdkM4aStreamWriter::new(
                                sink,
                                spec.sample_rate,
                                spec.channels as usize,
                                spec.channel_mask,
                                options.m4a_bitrate_bps,
                                options.downmix,
                                Some(limits.max_auxiliary_temporary_bytes),
                            )?)
                        }
                        #[cfg(not(feature = "fdk-aac-encoder"))]
                        {
                            return Err(
                                "FDK-AAC is unavailable in this build; rebuild with --features fdk-aac-encoder"
                                    .into(),
                            );
                        }
                    } else {
                        StreamWriterInner::M4a(super::m4a::M4aStreamWriter::new(
                            sink,
                            spec.sample_rate,
                            spec.channels as usize,
                            spec.channel_mask,
                            options.m4a_bitrate_bps,
                            options.downmix,
                            Some(limits.max_auxiliary_temporary_bytes),
                        )?)
                    }
                }
                #[cfg(not(feature = "m4a-encode"))]
                {
                    return Err(
                        "M4A output is unavailable in this build; rebuild with --features m4a-encode"
                            .into(),
                    );
                }
            }
            OutputFormat::AacAdts => {
                #[cfg(feature = "m4a-encode")]
                {
                    if options.aac_encoder == super::AacEncoder::Fdk {
                        return Err(
                            "FDK-AAC ADTS output is not available; use M4A or --aac-encoder oxide"
                                .into(),
                        );
                    }
                    StreamWriterInner::AacAdts(super::aac::AdtsAacStreamWriter::new(
                        sink,
                        spec.sample_rate,
                        spec.channels as usize,
                        spec.channel_mask,
                        options.m4a_bitrate_bps,
                        options.downmix,
                    )?)
                }
                #[cfg(not(feature = "m4a-encode"))]
                {
                    return Err(
                        "AAC output is unavailable in this build; rebuild with --features m4a-encode"
                            .into(),
                    );
                }
            }
        };
        Ok(Self {
            inner,
            spec,
            frames_written: 0,
        })
    }

    pub fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if channels.len() != self.spec.channels as usize {
            return Err(format!(
                "stream output expected {} channels, received {}",
                self.spec.channels,
                channels.len()
            ));
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("stream output blocks must have equal channel lengths".into());
        }
        // Stateful processors and resamplers may legitimately emit an empty
        // block while retaining overlap. Treat it as a no-op after validating
        // the channel geometry; several codec APIs reject a zero-length call.
        if frames == 0 {
            return Ok(());
        }
        let next = self
            .frames_written
            .checked_add(frames as u64)
            .ok_or_else(|| "stream output frame count overflows".to_string())?;
        if self.spec.total_frames.is_some_and(|total| next > total) {
            return Err("stream output exceeds its declared presentation length".into());
        }
        match &mut self.inner {
            StreamWriterInner::Wav(writer) => writer.write_block(channels),
            StreamWriterInner::Flac(writer) => writer.write_block(channels),
            StreamWriterInner::OggOpus(writer) => writer.write_block(channels),
            StreamWriterInner::Mp3(writer) => writer.write_block(channels),
            #[cfg(feature = "m4a-encode")]
            StreamWriterInner::M4a(writer) => writer.write_block(channels),
            #[cfg(feature = "fdk-aac-encoder")]
            StreamWriterInner::M4aFdk(writer) => writer.write_block(channels),
            #[cfg(feature = "m4a-encode")]
            StreamWriterInner::AacAdts(writer) => writer.write_block(channels),
        }?;
        self.frames_written = next;
        Ok(())
    }

    pub fn finalize(self) -> Result<(), String> {
        if self
            .spec
            .total_frames
            .is_some_and(|total| total != self.frames_written)
        {
            return Err(format!(
                "stream output wrote {} frames, expected {}",
                self.frames_written,
                self.spec.total_frames.unwrap_or(0)
            ));
        }
        match self.inner {
            StreamWriterInner::Wav(writer) => writer.finalize(),
            StreamWriterInner::Flac(writer) => writer.finalize(),
            StreamWriterInner::OggOpus(writer) => writer.finalize(),
            StreamWriterInner::Mp3(writer) => writer.finalize(),
            #[cfg(feature = "m4a-encode")]
            StreamWriterInner::M4a(writer) => writer.finalize(),
            #[cfg(feature = "fdk-aac-encoder")]
            StreamWriterInner::M4aFdk(writer) => writer.finalize(),
            #[cfg(feature = "m4a-encode")]
            StreamWriterInner::AacAdts(writer) => writer.finalize(),
        }
    }
}

/// Block-oriented encoder for a plain non-seekable [`Write`] sink.
///
/// Input PCM is retained in a finite anonymous file. On [`finalize`](Self::finalize),
/// it is replayed through [`AudioStreamWriter`], fully decoded and validated,
/// and only then copied to the supplied sink. Consequently an encode or
/// verification error writes no caller-visible bytes. A sink I/O error during
/// the final copy can still leave a partial external stream; non-seekable
/// destinations cannot provide filesystem atomicity.
pub struct SpooledAudioStreamWriter<W: Write> {
    sink: W,
    pcm: File,
    format: OutputFormat,
    spec: StreamEncodeSpec,
    options: EncodeOptions,
    encode_limits: StreamEncodeLimits,
    decode_limits: crate::DecodeLimits,
    spool_limits: crate::StreamSpoolLimits,
    replay_frames: usize,
    frames_written: u64,
    pcm_bytes: u64,
}

/// Finite anonymous interleaved-f64 spool for multi-pass stream processing.
///
/// This is used when a first pass must finish before encoding can begin, such
/// as integrated-loudness normalization. It never names or publishes a file;
/// dropping it removes the operating-system-managed anonymous storage.
pub struct StreamPcmSpool {
    pcm: File,
    channels: usize,
    frames: u64,
    bytes: u64,
    max_bytes: u64,
    read_frames: u64,
    reading: bool,
}

impl StreamPcmSpool {
    pub fn new(channels: usize, max_bytes: u64) -> Result<Self, String> {
        if channels == 0 || channels > crate::config::MAX_STREAM_CHANNELS {
            return Err(format!(
                "PCM stream spool channels must be between 1 and {}",
                crate::config::MAX_STREAM_CHANNELS
            ));
        }
        let pcm = tempfile::tempfile()
            .map_err(|error| format!("create anonymous PCM stream spool: {error}"))?;
        Ok(Self {
            pcm,
            channels,
            frames: 0,
            bytes: 0,
            max_bytes,
            read_frames: 0,
            reading: false,
        })
    }

    pub fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if self.reading {
            return Err("PCM stream spool cannot append after replay has begun".into());
        }
        if channels.len() != self.channels {
            return Err(format!(
                "PCM stream spool expected {} channels, received {}",
                self.channels,
                channels.len()
            ));
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("PCM stream spool blocks must have equal channel lengths".into());
        }
        let block_bytes = (frames as u64)
            .checked_mul(self.channels as u64)
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
            .ok_or_else(|| "PCM stream spool block size overflows".to_string())?;
        let next_bytes = self
            .bytes
            .checked_add(block_bytes)
            .ok_or_else(|| "PCM stream spool size overflows".to_string())?;
        if next_bytes > self.max_bytes {
            return Err(format!(
                "PCM stream spool requires {next_bytes} bytes, exceeding its {}-byte limit",
                self.max_bytes
            ));
        }
        for frame in 0..frames {
            for channel in channels {
                self.pcm
                    .write_all(&crate::sanitize_sample(channel[frame]).to_le_bytes())
                    .map_err(|error| format!("write anonymous PCM stream spool: {error}"))?;
            }
        }
        self.frames = self
            .frames
            .checked_add(frames as u64)
            .ok_or_else(|| "PCM stream spool frame count overflows".to_string())?;
        self.bytes = next_bytes;
        Ok(())
    }

    pub fn prepare_read(&mut self) -> Result<(), String> {
        self.pcm
            .flush()
            .and_then(|_| self.pcm.seek(SeekFrom::Start(0)).map(|_| ()))
            .map_err(|error| format!("rewind anonymous PCM stream spool: {error}"))?;
        self.read_frames = 0;
        self.reading = true;
        Ok(())
    }

    pub fn next_block(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<f64>>>, String> {
        if max_frames == 0 || max_frames > crate::config::MAX_STREAM_BLOCK_FRAMES {
            return Err(format!(
                "PCM stream spool replay size must be between 1 and {} frames",
                crate::config::MAX_STREAM_BLOCK_FRAMES
            ));
        }
        if !self.reading {
            return Err("PCM stream spool must be prepared before replay".into());
        }
        let remaining = self.frames.saturating_sub(self.read_frames);
        if remaining == 0 {
            return Ok(None);
        }
        let frames = remaining.min(max_frames as u64) as usize;
        let block = read_interleaved_pcm_block(&mut self.pcm, self.channels, frames)?;
        self.read_frames += frames as u64;
        Ok(Some(block))
    }

    #[must_use]
    pub const fn frames(&self) -> u64 {
        self.frames
    }

    #[must_use]
    pub const fn len(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames == 0
    }
}

impl<W: Write> SpooledAudioStreamWriter<W> {
    /// Construct a non-seekable writer with finite default spool/codec limits.
    pub fn new(
        sink: W,
        format: OutputFormat,
        spec: StreamEncodeSpec,
        options: EncodeOptions,
    ) -> Result<Self, String> {
        Self::new_with_limits(
            sink,
            format,
            spec,
            options,
            StreamEncodeLimits::default(),
            crate::DecodeLimits::default(),
            crate::StreamSpoolLimits::default(),
            DEFAULT_SPOOL_REPLAY_FRAMES,
        )
    }

    /// Construct a non-seekable writer with explicit finite limits.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_limits(
        sink: W,
        format: OutputFormat,
        spec: StreamEncodeSpec,
        options: EncodeOptions,
        encode_limits: StreamEncodeLimits,
        decode_limits: crate::DecodeLimits,
        spool_limits: crate::StreamSpoolLimits,
        replay_frames: usize,
    ) -> Result<Self, String> {
        validate_stream_config(format, spec, options)?;
        if !(1..=crate::config::MAX_STREAM_BLOCK_FRAMES).contains(&replay_frames) {
            return Err(format!(
                "non-seekable output replay size must be between 1 and {} frames",
                crate::config::MAX_STREAM_BLOCK_FRAMES
            ));
        }
        if let Some(required) =
            estimate_spooled_stream_output_bytes(format, spec, options, encode_limits)?
        {
            ensure_spool_limit(required, spool_limits, "declared non-seekable output")?;
        }
        let pcm = tempfile::tempfile()
            .map_err(|error| format!("create anonymous PCM output spool: {error}"))?;
        Ok(Self {
            sink,
            pcm,
            format,
            spec,
            options,
            encode_limits,
            decode_limits,
            spool_limits,
            replay_frames,
            frames_written: 0,
            pcm_bytes: 0,
        })
    }

    /// Append one planar block without retaining it in memory after return.
    pub fn write_block(&mut self, channels: &[Vec<f64>]) -> Result<(), String> {
        if channels.len() != usize::from(self.spec.channels) {
            return Err(format!(
                "non-seekable stream output expected {} channels, received {}",
                self.spec.channels,
                channels.len()
            ));
        }
        let frames = channels.first().map_or(0, Vec::len);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("non-seekable stream blocks must have equal channel lengths".into());
        }
        if frames == 0 {
            return Ok(());
        }
        let next_frames = self
            .frames_written
            .checked_add(frames as u64)
            .ok_or_else(|| "non-seekable output frame count overflows".to_string())?;
        if self
            .spec
            .total_frames
            .is_some_and(|declared| next_frames > declared)
        {
            return Err("non-seekable output exceeds its declared presentation length".into());
        }
        let required = estimate_spooled_stream_output_for_frames(
            self.format,
            self.spec,
            next_frames,
            self.options,
            self.encode_limits,
        )?;
        ensure_spool_limit(required, self.spool_limits, "non-seekable output")?;
        for frame in 0..frames {
            for channel in channels {
                self.pcm
                    .write_all(&channel[frame].to_le_bytes())
                    .map_err(|error| format!("write anonymous PCM output spool: {error}"))?;
            }
        }
        let block_bytes = (frames as u64)
            .checked_mul(u64::from(self.spec.channels))
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
            .ok_or_else(|| "non-seekable PCM block byte count overflows".to_string())?;
        self.pcm_bytes = self
            .pcm_bytes
            .checked_add(block_bytes)
            .ok_or_else(|| "non-seekable PCM spool byte count overflows".to_string())?;
        self.frames_written = next_frames;
        Ok(())
    }

    /// Complete, validate, and copy the encoded stream to the caller's sink.
    pub fn finalize(self) -> Result<W, String> {
        self.finalize_with_fingerprint().map(|(sink, _)| sink)
    }

    /// Complete, validate, fingerprint, and copy the encoded stream.
    ///
    /// The fingerprint describes the exact verified encoded spool. It is
    /// returned only after the sink accepted and flushed every byte, allowing a
    /// caller to sign a stdout receipt without authenticating a partial copy.
    pub fn finalize_with_fingerprint(
        self,
    ) -> Result<(W, crate::batch_resume::FileFingerprint), String> {
        self.finalize_with_metadata_and_loudness(
            None,
            crate::metadata::MetadataLimits::default(),
            None,
        )
        .map(|(sink, fingerprint, _)| (sink, fingerprint))
    }

    /// Complete optional two-pass normalization and metadata preservation,
    /// verify the encoded spool, and only then copy it to the sink.
    ///
    /// `loudness` is `(target_lufs, true_peak_dbtp)`. The returned report is
    /// present exactly when normalization was requested.
    pub fn finalize_with_metadata_and_loudness(
        mut self,
        metadata: Option<crate::metadata::Metadata>,
        metadata_limits: crate::metadata::MetadataLimits,
        loudness: Option<(f64, f64)>,
    ) -> Result<
        (
            W,
            crate::batch_resume::FileFingerprint,
            Option<crate::loudness::LoudnessReport>,
        ),
        String,
    > {
        if self
            .spec
            .total_frames
            .is_some_and(|declared| declared != self.frames_written)
        {
            return Err(format!(
                "non-seekable output wrote {} frames, expected {}",
                self.frames_written,
                self.spec.total_frames.unwrap_or(0)
            ));
        }
        let mut final_spec = self.spec;
        final_spec.total_frames = Some(self.frames_written);
        let reserved = estimate_spooled_stream_output_for_frames(
            self.format,
            final_spec,
            self.frames_written,
            self.options,
            self.encode_limits,
        )?;
        ensure_spool_limit(reserved, self.spool_limits, "non-seekable output")?;

        self.pcm
            .flush()
            .and_then(|_| self.pcm.seek(SeekFrom::Start(0)).map(|_| ()))
            .map_err(|error| format!("rewind anonymous PCM output spool: {error}"))?;
        let loudness_gain = if let Some((target_lufs, peak_limit_dbtp)) = loudness {
            let mut analyzer = crate::loudness::StreamingLoudnessAnalyzer::new(
                usize::from(self.spec.channels),
                self.spec.sample_rate,
                self.spec.channel_mask,
            )?;
            let mut remaining = self.frames_written;
            while remaining != 0 {
                let frames = remaining.min(self.replay_frames as u64) as usize;
                let block = read_interleaved_pcm_block(
                    &mut self.pcm,
                    usize::from(self.spec.channels),
                    frames,
                )?;
                analyzer.add_block(&block)?;
                remaining -= frames as u64;
            }
            self.pcm
                .seek(SeekFrom::Start(0))
                .map_err(|error| format!("rewind analyzed PCM output spool: {error}"))?;
            Some(analyzer.finish(target_lufs, peak_limit_dbtp)?)
        } else {
            None
        };
        let mut encoded = tempfile::tempfile()
            .map_err(|error| format!("create anonymous encoded output spool: {error}"))?;
        {
            let mut writer = AudioStreamWriter::new_with_limits(
                &mut encoded,
                self.format,
                final_spec,
                self.options,
                self.encode_limits,
            )?;
            let mut remaining = self.frames_written;
            while remaining != 0 {
                let frames = remaining.min(self.replay_frames as u64) as usize;
                let mut block = read_interleaved_pcm_block(
                    &mut self.pcm,
                    usize::from(self.spec.channels),
                    frames,
                )?;
                if let Some(gain) = loudness_gain {
                    gain.apply(&mut block);
                }
                writer.write_block(&block)?;
                remaining -= frames as u64;
            }
            writer.finalize()?;
        }
        if self.format == OutputFormat::Wav {
            crate::audio::write_wav_channel_mask_to_file(
                &mut encoded,
                usize::from(self.spec.channels),
                self.spec.channel_mask,
            )?;
        }
        if let Some(metadata) = metadata {
            crate::metadata::write_extended_to_file_with_limits(
                metadata,
                &mut encoded,
                metadata_limits,
            )?;
        }
        let encoded_bytes = encoded
            .metadata()
            .map_err(|error| format!("inspect anonymous encoded output spool: {error}"))?
            .len();
        let auxiliary = estimate_stream_encode_temporary_bytes(
            self.format,
            final_spec,
            self.options,
            self.encode_limits,
        )?;
        let actual = self
            .pcm_bytes
            .checked_add(encoded_bytes)
            .and_then(|bytes| bytes.checked_add(auxiliary))
            .ok_or_else(|| "non-seekable output actual spool byte count overflows".to_string())?;
        ensure_spool_limit(actual, self.spool_limits, "non-seekable output")?;
        verify_stream_output_file(
            &mut encoded,
            spooled_output_display_path(self.format),
            self.format,
            final_spec,
            self.frames_written,
            self.options,
            self.decode_limits,
            self.replay_frames,
        )?;
        let fingerprint = crate::batch_resume::fingerprint_open_file_at(
            &encoded,
            spooled_output_display_path(self.format),
        )?;
        encoded
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind verified encoded output spool: {error}"))?;
        std::io::copy(&mut encoded, &mut self.sink)
            .map_err(|error| format!("copy verified audio to non-seekable output: {error}"))?;
        self.sink
            .flush()
            .map_err(|error| format!("flush non-seekable audio output: {error}"))?;
        Ok((
            self.sink,
            fingerprint,
            loudness_gain.map(crate::loudness::StreamingLoudnessGain::report),
        ))
    }
}

fn ensure_spool_limit(
    required: u64,
    limits: crate::StreamSpoolLimits,
    context: &str,
) -> Result<(), String> {
    if required > limits.max_bytes() {
        return Err(format!(
            "{context} requires {required} bytes across its PCM, encoded, and auxiliary spools, exceeding the {}-byte spool limit",
            limits.max_bytes()
        ));
    }
    Ok(())
}

fn read_interleaved_pcm_block(
    source: &mut File,
    channels: usize,
    frames: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let mut block = Vec::new();
    block
        .try_reserve_exact(channels)
        .map_err(|error| format!("reserve spooled output channel list: {error}"))?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|error| format!("reserve spooled output channel: {error}"))?;
        block.push(channel);
    }
    let mut bytes = [0_u8; 8];
    for _ in 0..frames {
        for channel in &mut block {
            source
                .read_exact(&mut bytes)
                .map_err(|error| format!("read anonymous PCM output spool: {error}"))?;
            channel.push(f64::from_le_bytes(bytes));
        }
    }
    Ok(block)
}

fn spooled_output_display_path(format: OutputFormat) -> &'static Path {
    Path::new(match format {
        OutputFormat::Wav => "<writer>.wav",
        OutputFormat::Flac => "<writer>.flac",
        OutputFormat::OggOpus => "<writer>.opus",
        OutputFormat::Mp3 => "<writer>.mp3",
        OutputFormat::M4a => "<writer>.m4a",
        OutputFormat::AacAdts => "<writer>.aac",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn spec(sample_rate: u32, frames: u64) -> StreamEncodeSpec {
        StreamEncodeSpec::new(
            WavSpec {
                channels: 2,
                sample_rate,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            },
            crate::ChannelLayout::Stereo.mask(),
            Some(frames),
        )
    }

    #[test]
    fn bounded_writers_accept_multiple_blocks() {
        let formats = vec![
            OutputFormat::Wav,
            OutputFormat::Flac,
            OutputFormat::OggOpus,
            OutputFormat::Mp3,
            #[cfg(feature = "m4a-encode")]
            OutputFormat::M4a,
            #[cfg(feature = "m4a-encode")]
            OutputFormat::AacAdts,
        ];
        for format in formats {
            let mut output = Cursor::new(Vec::new());
            let frames = 4_321usize;
            let output_bound = estimate_stream_encode_output_bytes(
                format,
                spec(48_000, frames as u64),
                EncodeOptions::default(),
                StreamEncodeLimits::default(),
            )
            .unwrap()
            .unwrap();
            {
                let mut writer = AudioStreamWriter::new(
                    &mut output,
                    format,
                    spec(48_000, frames as u64),
                    EncodeOptions::default(),
                )
                .unwrap();
                for start in (0..frames).step_by(317) {
                    let len = (frames - start).min(317);
                    let left = (start..start + len)
                        .map(|index| (index as f64 / 31.0).sin() * 0.2)
                        .collect::<Vec<_>>();
                    writer.write_block(&[left.clone(), left]).unwrap();
                }
                writer.finalize().unwrap();
            }
            assert!(output.get_ref().len() > 64, "{format:?} output is empty");
            assert!(
                output.get_ref().len() as u64 <= output_bound,
                "{format:?} output exceeded its {output_bound}-byte bound"
            );
        }
    }

    #[test]
    fn finite_pcm_spool_replays_exact_blocks_and_enforces_limit() {
        let bytes = 3_u64 * 2 * std::mem::size_of::<f64>() as u64;
        let mut spool = StreamPcmSpool::new(2, bytes).unwrap();
        spool
            .write_block(&[vec![0.1, 0.2], vec![-0.1, -0.2]])
            .unwrap();
        spool.write_block(&[vec![0.3], vec![-0.3]]).unwrap();
        assert_eq!(spool.frames(), 3);
        assert_eq!(spool.len(), bytes);
        assert!(spool.write_block(&[vec![0.4], vec![-0.4]]).is_err());
        assert!(spool.next_block(2).is_err());
        spool.prepare_read().unwrap();
        assert!(spool.write_block(&[vec![0.4], vec![-0.4]]).is_err());
        assert_eq!(
            spool.next_block(2).unwrap().unwrap(),
            vec![vec![0.1, 0.2], vec![-0.1, -0.2]]
        );
        assert_eq!(
            spool.next_block(2).unwrap().unwrap(),
            vec![vec![0.3], vec![-0.3]]
        );
        assert!(spool.next_block(2).unwrap().is_none());
    }

    #[test]
    fn spooled_writer_publishes_to_a_plain_write_sink_after_verification() {
        let frames = 4_321usize;
        let input = (0..frames)
            .map(|index| (index as f64 / 31.0).sin() * 0.2)
            .collect::<Vec<_>>();
        let mut writer = SpooledAudioStreamWriter::new(
            Vec::new(),
            OutputFormat::Flac,
            spec(48_000, frames as u64),
            EncodeOptions::default(),
        )
        .unwrap();
        for start in (0..frames).step_by(317) {
            let end = (start + 317).min(frames);
            writer
                .write_block(&[input[start..end].to_vec(), input[start..end].to_vec()])
                .unwrap();
        }
        let (encoded, fingerprint) = writer.finalize_with_fingerprint().unwrap();
        assert!(encoded.starts_with(b"fLaC"));
        let root = tempfile::tempdir().unwrap();
        let captured = root.path().join("captured.flac");
        std::fs::write(&captured, &encoded).unwrap();
        assert_eq!(
            fingerprint,
            crate::batch_resume::fingerprint_file(&captured).unwrap()
        );

        let mut reader = crate::AudioStreamReader::from_reader(std::io::Cursor::new(encoded))
            .expect("decode published plain Write sink");
        let mut decoded_frames = 0usize;
        while let Some(block) = reader.next_block(113).unwrap() {
            assert_eq!(block.len(), 2);
            decoded_frames += block[0].len();
        }
        assert_eq!(decoded_frames, frames);
    }

    #[test]
    fn spooled_writer_declared_bound_has_an_exact_prewrite_boundary() {
        let stream_spec = spec(48_000, 321);
        let required = estimate_spooled_stream_output_bytes(
            OutputFormat::Flac,
            stream_spec,
            EncodeOptions::default(),
            StreamEncodeLimits::default(),
        )
        .unwrap()
        .unwrap();
        SpooledAudioStreamWriter::new_with_limits(
            Vec::new(),
            OutputFormat::Flac,
            stream_spec,
            EncodeOptions::default(),
            StreamEncodeLimits::default(),
            crate::DecodeLimits::default(),
            crate::StreamSpoolLimits::new(required),
            73,
        )
        .unwrap();
        let error = match SpooledAudioStreamWriter::new_with_limits(
            Vec::new(),
            OutputFormat::Flac,
            stream_spec,
            EncodeOptions::default(),
            StreamEncodeLimits::default(),
            crate::DecodeLimits::default(),
            crate::StreamSpoolLimits::new(required - 1),
            73,
        ) {
            Ok(_) => panic!("one byte below the declared spool bound must fail"),
            Err(error) => error,
        };
        assert!(error.contains("spool limit"), "{error}");
    }

    #[test]
    fn exact_declared_length_is_enforced_before_finalize() {
        let mut output = Cursor::new(Vec::new());
        let mut writer = AudioStreamWriter::new(
            &mut output,
            OutputFormat::Wav,
            spec(48_000, 2),
            EncodeOptions::default(),
        )
        .unwrap();
        let error = writer
            .write_block(&[vec![0.0; 3], vec![0.0; 3]])
            .unwrap_err();
        assert!(error.contains("declared presentation length"));
    }

    #[cfg(feature = "m4a-encode")]
    #[test]
    fn m4a_auxiliary_table_limit_is_checked_before_touching_the_sink() {
        let stream_spec = spec(48_000, 1_024);
        let options = EncodeOptions::default();
        let exact = 2 * M4A_TABLE_RECORD_BYTES;
        assert_eq!(
            estimate_stream_encode_temporary_bytes(
                OutputFormat::M4a,
                stream_spec,
                options,
                StreamEncodeLimits::new(exact),
            )
            .unwrap(),
            exact
        );

        let mut output = Cursor::new(Vec::new());
        let error = match AudioStreamWriter::new_with_limits(
            &mut output,
            OutputFormat::M4a,
            stream_spec,
            options,
            StreamEncodeLimits::new(exact - 1),
        ) {
            Ok(_) => panic!("undersized M4A sample-table limit was accepted"),
            Err(error) => error,
        };
        assert!(error.contains("requires 24 bytes"), "{error}");
        assert!(output.get_ref().is_empty());
    }

    #[cfg(feature = "m4a-encode")]
    #[test]
    fn unknown_m4a_duration_reserves_the_configured_auxiliary_ceiling() {
        let mut stream_spec = spec(48_000, 1);
        stream_spec.total_frames = None;
        let limit = 7_777;
        assert_eq!(
            estimate_stream_encode_temporary_bytes(
                OutputFormat::M4a,
                stream_spec,
                EncodeOptions::default(),
                StreamEncodeLimits::new(limit),
            )
            .unwrap(),
            limit
        );
    }

    #[cfg(feature = "fdk-aac-encoder")]
    #[test]
    fn bounded_fdk_m4a_writer_accepts_multiple_blocks() {
        let mut output = Cursor::new(Vec::new());
        let frames = 3_217usize;
        let mut options = EncodeOptions::default();
        options.aac_encoder = crate::encode::AacEncoder::Fdk;
        let stream_spec = spec(48_000, frames as u64);
        let exact_table_bytes = estimate_stream_encode_temporary_bytes(
            OutputFormat::M4a,
            stream_spec,
            options,
            StreamEncodeLimits::default(),
        )
        .unwrap();
        let mut writer = AudioStreamWriter::new_with_limits(
            &mut output,
            OutputFormat::M4a,
            stream_spec,
            options,
            StreamEncodeLimits::new(exact_table_bytes),
        )
        .unwrap();
        for start in (0..frames).step_by(211) {
            let len = (frames - start).min(211);
            let left = (start..start + len)
                .map(|index| (index as f64 / 17.0).sin() * 0.2)
                .collect::<Vec<_>>();
            writer.write_block(&[left.clone(), left]).unwrap();
        }
        writer.finalize().unwrap();
        assert!(output.get_ref().len() > 64);
        let output_bound = estimate_stream_encode_output_bytes(
            OutputFormat::M4a,
            stream_spec,
            options,
            StreamEncodeLimits::new(exact_table_bytes),
        )
        .unwrap()
        .unwrap();
        assert!(output.get_ref().len() as u64 <= output_bound);
    }
}
