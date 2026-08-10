//! Audio encode layer — WAV / MP3 / M4A output.
//!
//! | Format | Backend |
//! |--------|---------|
//! | WAV | `hound` (lossless, preserves bit depth) |
//! | MP3 | `shine-rs` (Pure Rust) |
//! | M4A | `oxideav-aac` + `mp4` mux (Pure-Rust AAC-LC) |

#[cfg(feature = "m4a-encode")]
mod aac;
#[cfg(feature = "m4a-encode")]
mod m4a;
#[cfg(feature = "fdk-aac-encoder")]
mod m4a_fdk;
mod mp3;
mod opus;
mod pcm;

#[cfg(feature = "m4a-encode")]
pub use aac::{write_adts_aac, write_adts_aac_with_downmix};
#[cfg(feature = "m4a-encode")]
pub use m4a::{write_m4a, write_m4a_with_downmix};
#[cfg(feature = "fdk-aac-encoder")]
pub use m4a_fdk::{write_m4a_fdk, write_m4a_fdk_with_downmix};
pub use mp3::{write_mp3, write_mp3_with_downmix, DEFAULT_MP3_BITRATE};

/// Default AAC bitrate (bps, not kbps).
pub const DEFAULT_M4A_BITRATE: u32 = 192_000;

/// Policy for reducing a surround input to a codec's supported channel count.
///
/// `Preserve` is deliberately the default: a 5.1/7.1 input must never be
/// silently mixed into stereo.  Select `Stereo` only when that loss of spatial
/// channels is intentional.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DownmixMode {
    /// Refuse an output codec that cannot represent the input layout.
    #[default]
    Preserve,
    /// Explicitly render a standard surround layout to stereo.
    Stereo,
}

impl DownmixMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "preserve" | "none" | "off" => Some(Self::Preserve),
            "stereo" | "2" | "on" => Some(Self::Stereo),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AacEncoder {
    #[default]
    Oxide,
    Fdk,
}

impl AacEncoder {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "oxide" | "oxideav" | "rust" => Some(Self::Oxide),
            "fdk" | "fdk-aac" => Some(Self::Fdk),
            _ => None,
        }
    }
}

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::atomic_output::{AtomicOutput, CommitMode};
use crate::audio::{sanitize_sample, write_wav_to_file, Audio};
use crate::config::MAX_SAMPLE_RATE;

const AAC_ENCODER_SAMPLE_RATES: [u32; 12] = [
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 64_000, 88_200, 96_000,
];

// hound stores `file length - 8` in RIFF's 32-bit size field. Its canonical
// PCM header contributes 36 bytes to that value, while WAVEFORMATEXTENSIBLE
// contributes 60 bytes.
const WAV_PCM_RIFF_OVERHEAD_BYTES: u64 = 36;
const WAV_EXTENSIBLE_RIFF_OVERHEAD_BYTES: u64 = 60;

/// Output container inferred from file extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Wav,
    Flac,
    OggOpus,
    Mp3,
    M4a,
    AacAdts,
}

impl OutputFormat {
    pub fn from_path(path: &Path) -> Result<Self, String> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("wav") => Ok(OutputFormat::Wav),
            Some("flac") => Ok(OutputFormat::Flac),
            Some("opus" | "ogg" | "oga") => Ok(OutputFormat::OggOpus),
            Some("mp3") => Ok(OutputFormat::Mp3),
            Some("m4a" | "m4b" | "mp4") => Ok(OutputFormat::M4a),
            Some("aac") => Ok(OutputFormat::AacAdts),
            Some(ext) => Err(format!(
                "unsupported output format '.{ext}'; use .wav, .flac, .opus/.ogg/.oga, .mp3, .m4a, or .aac"
            )),
            None => Err(
                "output path has no extension; use .wav, .flac, .opus/.ogg/.oga, .mp3, .m4a, or .aac".into(),
            ),
        }
    }

    /// Validate that this output format and encoder selection are available
    /// in the current build before any audio processing or output staging.
    pub fn validate_encoder(self, aac_encoder: AacEncoder) -> Result<(), String> {
        match self {
            OutputFormat::M4a => {
                #[cfg(not(feature = "m4a-encode"))]
                {
                    let _ = aac_encoder;
                    Err("M4A output is unavailable in this build; rebuild with --features m4a-encode or use WAV/MP3".into())
                }
                #[cfg(feature = "m4a-encode")]
                {
                    if aac_encoder == AacEncoder::Fdk {
                        #[cfg(not(feature = "fdk-aac-encoder"))]
                        {
                            return Err("FDK-AAC is unavailable in this build; rebuild with --features fdk-aac-encoder".into());
                        }
                    }
                    Ok(())
                }
            }
            OutputFormat::AacAdts => {
                #[cfg(not(feature = "m4a-encode"))]
                {
                    let _ = aac_encoder;
                    Err(
                        "AAC output is unavailable in this build; rebuild with --features m4a-encode"
                            .into(),
                    )
                }
                #[cfg(feature = "m4a-encode")]
                {
                    if aac_encoder == AacEncoder::Fdk {
                        return Err(
                            "FDK-AAC ADTS output is not available; use M4A or --aac-encoder oxide"
                                .into(),
                        );
                    }
                    Ok(())
                }
            }
            OutputFormat::Wav | OutputFormat::Flac | OutputFormat::OggOpus | OutputFormat::Mp3 => {
                Ok(())
            }
        }
    }

    /// Validate an encode request without opening, seeking, or modifying an
    /// output file.
    pub fn validate_config(self, audio: &Audio, options: &EncodeOptions) -> Result<(), String> {
        options.validate_config(self, audio)
    }
}

/// Encoding options for lossy outputs.
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// MP3 constant bitrate in kbps.
    pub mp3_bitrate_kbps: u32,
    /// AAC constant bitrate in bps.
    pub m4a_bitrate_bps: u32,
    pub aac_encoder: AacEncoder,
    /// Explicit policy for multichannel lossy outputs.
    pub downmix: DownmixMode,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            mp3_bitrate_kbps: DEFAULT_MP3_BITRATE,
            m4a_bitrate_bps: DEFAULT_M4A_BITRATE,
            aac_encoder: AacEncoder::Oxide,
            downmix: DownmixMode::Preserve,
        }
    }
}

impl EncodeOptions {
    /// Validate encoder availability and options whose contract does not
    /// depend on decoded PCM. This is suitable for filesystem preflight
    /// before input metadata or audio is read.
    pub fn validate_options(self, format: OutputFormat) -> Result<(), String> {
        format.validate_encoder(self.aac_encoder)?;
        if matches!(format, OutputFormat::M4a | OutputFormat::AacAdts) && self.m4a_bitrate_bps == 0
        {
            let codec = if format == OutputFormat::M4a {
                "M4A"
            } else {
                "AAC"
            };
            return Err(format!("{codec} encode: bitrate must be greater than zero"));
        }
        Ok(())
    }

    /// Validate encoder availability, PCM structure, and codec-specific
    /// constraints without creating an encoder or touching an output file.
    pub fn validate_config(self, format: OutputFormat, audio: &Audio) -> Result<(), String> {
        // Availability is independent of the input and must win over later
        // codec checks for formats omitted from this build.
        self.validate_options(format)?;
        validate_audio_structure(audio)?;

        match format {
            OutputFormat::Wav => validate_wav_config(audio),
            OutputFormat::Flac => validate_flac_config(audio),
            OutputFormat::OggOpus => {
                require_frames(audio, "Opus")?;
                if audio.sample_rate > MAX_SAMPLE_RATE {
                    return Err(format!(
                        "Opus encode: unsupported source sample rate {} Hz (supported: 1..={MAX_SAMPLE_RATE})",
                        audio.sample_rate
                    ));
                }
                let layout = pcm::lossy_channel_layout(audio, self.downmix)?;
                crate::resample::validate_resampler_plan(
                    layout.count as usize,
                    audio.sample_rate,
                    48_000,
                )?;
                Ok(())
            }
            OutputFormat::Mp3 => {
                // Resolve and validate the exact configuration that the writer
                // will pass to shine-rs. Requests between supported rates keep
                // the established round-down behavior; values below a codec's
                // minimum use its lowest compatible bitrate.
                mp3::effective_mp3_config(audio, self.mp3_bitrate_kbps, self.downmix)?;
                Ok(())
            }
            OutputFormat::M4a | OutputFormat::AacAdts => {
                let codec = if format == OutputFormat::M4a {
                    "M4A"
                } else {
                    "AAC"
                };
                require_frames(audio, codec)?;
                let _layout = pcm::lossy_channel_layout(audio, self.downmix)?;
                if !AAC_ENCODER_SAMPLE_RATES.contains(&audio.sample_rate) {
                    return Err(format!(
                        "{codec} encode: unsupported sample rate {} Hz (AAC standard rates only)",
                        audio.sample_rate
                    ));
                }
                #[cfg(feature = "fdk-aac-encoder")]
                if self.aac_encoder == AacEncoder::Fdk {
                    validate_fdk_aac_config(
                        _layout.count as usize,
                        audio.sample_rate,
                        self.m4a_bitrate_bps,
                    )?;
                }

                Ok(())
            }
        }
    }
}

fn validate_audio_structure(audio: &Audio) -> Result<(), String> {
    if audio.channels.is_empty() {
        return Err("audio output requires at least one channel".into());
    }
    if audio.sample_rate == 0 {
        return Err("audio output sample rate must be greater than zero".into());
    }

    let frames = audio.channels[0].len();
    if let Some((index, channel)) = audio
        .channels
        .iter()
        .enumerate()
        .find(|(_, channel)| channel.len() != frames)
    {
        return Err(format!(
            "audio channel {index} has {} frames, expected {frames}",
            channel.len()
        ));
    }
    frames
        .checked_mul(audio.channels.len())
        .ok_or_else(|| "audio sample count is too large to encode".to_string())?;
    Ok(())
}

fn require_frames(audio: &Audio, codec: &str) -> Result<(), String> {
    if audio.frames() == 0 {
        return Err(format!("{codec} output requires at least one frame"));
    }
    Ok(())
}

fn validate_wav_container_size(
    data_bytes: u64,
    channels: u64,
    bits_per_sample: u16,
) -> Result<(), String> {
    // Keep this predicate synchronized with hound's WavWriter format choice.
    let riff_overhead = if channels > 2 || bits_per_sample > 16 {
        WAV_EXTENSIBLE_RIFF_OVERHEAD_BYTES
    } else {
        WAV_PCM_RIFF_OVERHEAD_BYTES
    };
    let max_data_bytes = u64::from(u32::MAX) - riff_overhead;
    if data_bytes > max_data_bytes {
        return Err("WAV encode: PCM data plus RIFF header exceeds the WAV container limit".into());
    }
    Ok(())
}

fn validate_wav_config(audio: &Audio) -> Result<(), String> {
    let bytes_per_sample = match (audio.sample_format, audio.bits_per_sample) {
        (hound::SampleFormat::Int, 8) => 1u64,
        (hound::SampleFormat::Int, 16) => 2,
        (hound::SampleFormat::Int, 24) => 3,
        (hound::SampleFormat::Int, 32) => 4,
        (hound::SampleFormat::Float, 32) => 4,
        (hound::SampleFormat::Int, bits) => {
            return Err(format!(
                "WAV encode: unsupported integer bit depth {bits} (supported: 8, 16, 24, 32)"
            ));
        }
        (hound::SampleFormat::Float, bits) => {
            return Err(format!(
                "WAV encode: unsupported float bit depth {bits} (supported: 32)"
            ));
        }
    };
    let channels = u64::try_from(audio.channels())
        .map_err(|_| "WAV encode: channel count is too large".to_string())?;
    if channels > u16::MAX as u64 {
        return Err(format!(
            "WAV encode: {} channels exceed the WAV header limit of {}",
            audio.channels(),
            u16::MAX
        ));
    }
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| "WAV encode: block alignment overflow".to_string())?;
    if block_align > u16::MAX as u64 {
        return Err("WAV encode: block alignment exceeds the WAV header limit".into());
    }
    let bytes_per_second = u64::from(audio.sample_rate)
        .checked_mul(block_align)
        .ok_or_else(|| "WAV encode: byte-rate overflow".to_string())?;
    if bytes_per_second > u32::MAX as u64 {
        return Err("WAV encode: byte rate exceeds the WAV header limit".into());
    }
    let data_bytes = u64::try_from(audio.frames())
        .ok()
        .and_then(|frames| frames.checked_mul(block_align))
        .ok_or_else(|| "WAV encode: data length overflow".to_string())?;
    validate_wav_container_size(data_bytes, channels, audio.bits_per_sample)
}

fn validate_flac_config(audio: &Audio) -> Result<(), String> {
    require_frames(audio, "FLAC")?;
    if audio.channels() > 8 {
        return Err(format!(
            "FLAC encode: unsupported channel count {} (supported: 1..=8)",
            audio.channels()
        ));
    }
    if audio.sample_rate > 96_000 {
        return Err(format!(
            "FLAC encode: unsupported sample rate {} Hz (supported: 1..=96000)",
            audio.sample_rate
        ));
    }
    let bits = audio.bits_per_sample.clamp(8, 24);
    if !matches!(bits, 8 | 12 | 16 | 20 | 24) {
        return Err(format!(
            "FLAC encode: unsupported effective bit depth {bits} (supported: 8, 12, 16, 20, 24)"
        ));
    }
    Ok(())
}

#[cfg(feature = "fdk-aac-encoder")]
fn validate_fdk_aac_config(
    channels: usize,
    sample_rate: u32,
    bitrate_bps: u32,
) -> Result<(), String> {
    use fdk_aac_rust::encoder::{EncoderParameter, PureRustEncoderParameters};

    let mut parameters = PureRustEncoderParameters::new(channels);
    for (parameter, value) in [
        (EncoderParameter::AudioObjectType, 2),
        (EncoderParameter::SampleRate, sample_rate),
        (EncoderParameter::Bitrate, bitrate_bps),
        (EncoderParameter::BitrateMode, 0),
        (
            EncoderParameter::ChannelMode,
            if channels == 2 { 2 } else { 1 },
        ),
        (EncoderParameter::ChannelOrder, 1),
        (EncoderParameter::Afterburner, 1),
        (EncoderParameter::TransportMux, 0),
    ] {
        parameters
            .set_parameter(parameter, value)
            .map_err(|error| format!("FDK-AAC parameter: {error}"))?;
    }
    parameters
        .resolve()
        .map(|_| ())
        .map_err(|error| format!("FDK-AAC config: {error}"))
}

/// Write audio to a file; format is chosen from the path extension.
pub fn write_audio<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    options: EncodeOptions,
) -> Result<(), String> {
    let path = path.as_ref();
    let format = OutputFormat::from_path(path)?;
    format.validate_config(audio, &options)?;
    let mut output = AtomicOutput::new(path)?;
    write_audio_to_file(output.file_mut(), format, audio, options)?;
    output.commit(CommitMode::Replace)
}

/// Write audio to an already-open file using an explicitly selected format.
///
/// The request is validated before the file is rewound and truncated. After
/// that preflight, callers can safely use a securely-created staging file
/// without reopening a predictable path.
pub fn write_audio_to_file(
    file: &mut File,
    format: OutputFormat,
    audio: &Audio,
    options: EncodeOptions,
) -> Result<(), String> {
    format.validate_config(audio, &options)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("seek output: {error}"))?;
    file.set_len(0)
        .map_err(|error| format!("truncate output: {error}"))?;

    match format {
        OutputFormat::Wav => write_wav_to_file(file, audio),
        OutputFormat::Flac => write_flac_to_writer(&mut *file, audio),
        OutputFormat::OggOpus => {
            opus::write_ogg_opus_to_writer(&mut *file, audio, 128_000, options.downmix)
        }
        OutputFormat::Mp3 => {
            mp3::write_mp3_to_writer(&mut *file, audio, options.mp3_bitrate_kbps, options.downmix)
        }
        OutputFormat::M4a => {
            #[cfg(feature = "m4a-encode")]
            {
                match options.aac_encoder {
                    AacEncoder::Oxide => m4a::write_m4a_to_writer(
                        &mut *file,
                        audio,
                        options.m4a_bitrate_bps,
                        options.downmix,
                    ),
                    AacEncoder::Fdk => {
                        #[cfg(feature = "fdk-aac-encoder")]
                        {
                            m4a_fdk::write_m4a_fdk_to_writer(
                                &mut *file,
                                audio,
                                options.m4a_bitrate_bps,
                                options.downmix,
                            )
                        }
                        #[cfg(not(feature = "fdk-aac-encoder"))]
                        {
                            Err("FDK-AAC is unavailable in this build; rebuild with --features fdk-aac-encoder".into())
                        }
                    }
                }
            }
            #[cfg(not(feature = "m4a-encode"))]
            {
                Err("M4A output is unavailable in the crates.io build; use WAV/MP3 or a GitHub release binary".into())
            }
        }
        OutputFormat::AacAdts => {
            #[cfg(feature = "m4a-encode")]
            {
                if options.aac_encoder == AacEncoder::Fdk {
                    return Err(
                        "FDK-AAC ADTS output is not available; use M4A or --aac-encoder oxide"
                            .into(),
                    );
                }
                aac::write_adts_aac_to_writer(
                    &mut *file,
                    audio,
                    options.m4a_bitrate_bps,
                    options.downmix,
                )
            }
            #[cfg(not(feature = "m4a-encode"))]
            {
                Err(
                    "AAC output is unavailable in this build; rebuild with --features m4a-encode"
                        .into(),
                )
            }
        }
    }?;

    file.flush()
        .map_err(|error| format!("flush output: {error}"))
}

fn write_flac_to_writer<W: Write>(mut output: W, audio: &Audio) -> Result<(), String> {
    if audio.channels() == 0 {
        return Err("FLAC output requires at least one channel".into());
    }
    if audio.frames() == 0 {
        return Err("FLAC output requires at least one frame".into());
    }
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;
    let bits = audio.bits_per_sample.clamp(8, 24) as usize;
    let scale = (1_i64 << (bits - 1)) as f64;
    let mut samples = Vec::with_capacity(audio.frames() * audio.channels());
    for frame in 0..audio.frames() {
        for channel in &audio.channels {
            samples.push(
                (sanitize_sample(channel[frame]) * scale)
                    .round()
                    .clamp(-scale, scale - 1.0) as i32,
            );
        }
    }
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| format!("FLAC config: {:?}", e.1))?;
    let source = flacenc::source::MemSource::from_samples(
        &samples,
        audio.channels(),
        bits,
        audio.sample_rate as usize,
    );
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| format!("FLAC encode: {e}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| format!("FLAC serialize: {e}"))?;
    output
        .write_all(sink.as_slice())
        .map_err(|e| format!("FLAC write: {e}"))?;
    output.flush().map_err(|e| format!("FLAC flush: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek};

    fn test_audio(sample_rate: u32, channels: usize, frames: usize) -> Audio {
        Audio {
            sample_rate,
            channels: vec![vec![0.0; frames]; channels],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        }
    }

    #[test]
    fn wav_container_limit_reserves_hounds_exact_riff_header_size() {
        let pcm_limit = u64::from(u32::MAX) - WAV_PCM_RIFF_OVERHEAD_BYTES;
        validate_wav_container_size(pcm_limit, 2, 16).unwrap();
        assert!(validate_wav_container_size(pcm_limit + 1, 2, 16)
            .unwrap_err()
            .contains("RIFF header"));

        let extensible_limit = u64::from(u32::MAX) - WAV_EXTENSIBLE_RIFF_OVERHEAD_BYTES;
        for (channels, bits_per_sample) in [(1, 24), (3, 16)] {
            validate_wav_container_size(extensible_limit, channels, bits_per_sample).unwrap();
            assert!(
                validate_wav_container_size(extensible_limit + 1, channels, bits_per_sample)
                    .unwrap_err()
                    .contains("RIFF header")
            );
        }
    }

    #[test]
    fn detect_output_formats() {
        assert_eq!(
            OutputFormat::from_path(Path::new("out.mp3")).unwrap(),
            OutputFormat::Mp3
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("out.m4a")).unwrap(),
            OutputFormat::M4a
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("out.aac")).unwrap(),
            OutputFormat::AacAdts
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("out.flac")).unwrap(),
            OutputFormat::Flac
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("out.opus")).unwrap(),
            OutputFormat::OggOpus
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("out.oga")).unwrap(),
            OutputFormat::OggOpus
        );
    }

    #[test]
    fn validates_encoder_availability_before_writing() {
        for format in [
            OutputFormat::Wav,
            OutputFormat::Flac,
            OutputFormat::OggOpus,
            OutputFormat::Mp3,
        ] {
            assert!(format.validate_encoder(AacEncoder::Oxide).is_ok());
            assert!(format.validate_encoder(AacEncoder::Fdk).is_ok());
        }

        #[cfg(feature = "m4a-encode")]
        {
            assert!(OutputFormat::M4a
                .validate_encoder(AacEncoder::Oxide)
                .is_ok());
            assert!(OutputFormat::AacAdts
                .validate_encoder(AacEncoder::Oxide)
                .is_ok());
        }
        #[cfg(not(feature = "m4a-encode"))]
        {
            assert!(OutputFormat::M4a
                .validate_encoder(AacEncoder::Oxide)
                .is_err());
            assert!(OutputFormat::AacAdts
                .validate_encoder(AacEncoder::Oxide)
                .is_err());
        }

        #[cfg(all(feature = "m4a-encode", feature = "fdk-aac-encoder"))]
        assert!(OutputFormat::M4a.validate_encoder(AacEncoder::Fdk).is_ok());
        #[cfg(not(all(feature = "m4a-encode", feature = "fdk-aac-encoder")))]
        assert!(OutputFormat::M4a.validate_encoder(AacEncoder::Fdk).is_err());
        assert!(OutputFormat::AacAdts
            .validate_encoder(AacEncoder::Fdk)
            .is_err());

        let _invalid_audio = test_audio(0, 0, 0);
        #[cfg(not(feature = "m4a-encode"))]
        assert!(OutputFormat::M4a
            .validate_config(&_invalid_audio, &EncodeOptions::default())
            .unwrap_err()
            .contains("unavailable"));
        #[cfg(all(feature = "m4a-encode", not(feature = "fdk-aac-encoder")))]
        assert!(OutputFormat::M4a
            .validate_config(
                &_invalid_audio,
                &EncodeOptions {
                    aac_encoder: AacEncoder::Fdk,
                    ..EncodeOptions::default()
                },
            )
            .unwrap_err()
            .contains("unavailable"));
    }

    #[test]
    fn default_options_validate_for_built_in_formats() {
        let audio = test_audio(48_000, 2, 1);
        let options = EncodeOptions::default();

        for format in [
            OutputFormat::Wav,
            OutputFormat::Flac,
            OutputFormat::OggOpus,
            OutputFormat::Mp3,
        ] {
            format.validate_config(&audio, &options).unwrap();
            options.validate_config(format, &audio).unwrap();
        }

        #[cfg(feature = "m4a-encode")]
        for format in [OutputFormat::M4a, OutputFormat::AacAdts] {
            format.validate_config(&audio, &options).unwrap();
        }

        #[cfg(feature = "fdk-aac-encoder")]
        OutputFormat::M4a
            .validate_config(
                &audio,
                &EncodeOptions {
                    aac_encoder: AacEncoder::Fdk,
                    ..options
                },
            )
            .unwrap();
    }

    #[test]
    fn options_only_validation_checks_selected_codec_contract() {
        for bitrate in [0, u32::MAX] {
            EncodeOptions {
                mp3_bitrate_kbps: bitrate,
                ..EncodeOptions::default()
            }
            .validate_options(OutputFormat::Mp3)
            .unwrap();
        }

        #[cfg(feature = "m4a-encode")]
        for format in [OutputFormat::M4a, OutputFormat::AacAdts] {
            for bitrate in [1, u32::MAX] {
                EncodeOptions {
                    m4a_bitrate_bps: bitrate,
                    ..EncodeOptions::default()
                }
                .validate_options(format)
                .unwrap();
            }
            assert!(EncodeOptions {
                m4a_bitrate_bps: 0,
                ..EncodeOptions::default()
            }
            .validate_options(format)
            .unwrap_err()
            .contains("bitrate"));
        }

        #[cfg(not(feature = "m4a-encode"))]
        assert!(EncodeOptions {
            m4a_bitrate_bps: 0,
            ..EncodeOptions::default()
        }
        .validate_options(OutputFormat::M4a)
        .unwrap_err()
        .contains("unavailable"));

        #[cfg(all(feature = "m4a-encode", not(feature = "fdk-aac-encoder")))]
        assert!(EncodeOptions {
            aac_encoder: AacEncoder::Fdk,
            ..EncodeOptions::default()
        }
        .validate_options(OutputFormat::M4a)
        .unwrap_err()
        .contains("unavailable"));

        #[cfg(feature = "fdk-aac-encoder")]
        EncodeOptions {
            aac_encoder: AacEncoder::Fdk,
            ..EncodeOptions::default()
        }
        .validate_options(OutputFormat::M4a)
        .unwrap();
    }

    #[test]
    fn codec_boundaries_are_validated_without_narrowing_mp3_bitrate() {
        let mut mp3 = test_audio(44_100, 1, 1);
        for &sample_rate in shine_rs::SUPPORTED_SAMPLE_RATES {
            mp3.sample_rate = sample_rate;
            for bitrate in [0, u32::MAX] {
                let options = EncodeOptions {
                    mp3_bitrate_kbps: bitrate,
                    ..EncodeOptions::default()
                };
                options.validate_config(OutputFormat::Mp3, &mp3).unwrap();
            }
        }
        mp3.sample_rate = 12_345;
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::Mp3, &mp3)
            .unwrap_err()
            .contains("unsupported sample rate"));

        let mut opus = test_audio(8_000, 1, 1);
        EncodeOptions::default()
            .validate_config(OutputFormat::OggOpus, &opus)
            .unwrap();
        opus.sample_rate = 1;
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::OggOpus, &opus)
            .unwrap_err()
            .contains("working set"));
        opus.sample_rate = MAX_SAMPLE_RATE;
        EncodeOptions::default()
            .validate_config(OutputFormat::OggOpus, &opus)
            .unwrap();
        opus.sample_rate = MAX_SAMPLE_RATE + 1;
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::OggOpus, &opus)
            .is_err());

        let flac = test_audio(96_000, 8, 1);
        EncodeOptions::default()
            .validate_config(OutputFormat::Flac, &flac)
            .unwrap();
        let nine_channel_flac = test_audio(96_000, 9, 1);
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::Flac, &nine_channel_flac)
            .is_err());
        let too_fast_flac = test_audio(96_001, 1, 1);
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::Flac, &too_fast_flac)
            .is_err());

        let mut unsupported_flac_depth = test_audio(48_000, 1, 1);
        unsupported_flac_depth.bits_per_sample = 10;
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::Flac, &unsupported_flac_depth)
            .is_err());
        unsupported_flac_depth.bits_per_sample = 7;
        EncodeOptions::default()
            .validate_config(OutputFormat::Flac, &unsupported_flac_depth)
            .unwrap();
        unsupported_flac_depth.bits_per_sample = u16::MAX;
        EncodeOptions::default()
            .validate_config(OutputFormat::Flac, &unsupported_flac_depth)
            .unwrap();
    }

    #[cfg(feature = "m4a-encode")]
    #[test]
    fn aac_rates_and_bitrate_follow_the_encoder_contract() {
        let options = EncodeOptions::default();
        let mut audio = test_audio(48_000, 1, 1);
        for sample_rate in AAC_ENCODER_SAMPLE_RATES {
            audio.sample_rate = sample_rate;
            options.validate_config(OutputFormat::M4a, &audio).unwrap();
            options
                .validate_config(OutputFormat::AacAdts, &audio)
                .unwrap();
        }

        audio.sample_rate = 7_350;
        assert!(options.validate_config(OutputFormat::M4a, &audio).is_err());
        audio.sample_rate = 48_000;
        assert!(EncodeOptions {
            m4a_bitrate_bps: 0,
            ..options
        }
        .validate_config(OutputFormat::M4a, &audio)
        .unwrap_err()
        .contains("bitrate"));
    }

    #[test]
    fn planar_structure_and_downmix_policy_are_validated() {
        let mut uneven = test_audio(44_100, 2, 2);
        uneven.channels[1].pop();
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::Wav, &uneven)
            .unwrap_err()
            .contains("expected 2"));

        let zero_rate = test_audio(0, 1, 1);
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::Wav, &zero_rate)
            .unwrap_err()
            .contains("greater than zero"));

        // Empty PCM is a valid WAV, while frame-oriented codecs require at
        // least one frame.
        let empty_pcm = test_audio(44_100, 1, 0);
        EncodeOptions::default()
            .validate_config(OutputFormat::Wav, &empty_pcm)
            .unwrap();
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::Mp3, &empty_pcm)
            .is_err());

        let surround = test_audio(44_100, 6, 1);
        assert!(EncodeOptions::default()
            .validate_config(OutputFormat::Mp3, &surround)
            .unwrap_err()
            .contains("without mixing"));
        EncodeOptions {
            downmix: DownmixMode::Stereo,
            ..EncodeOptions::default()
        }
        .validate_config(OutputFormat::Mp3, &surround)
        .unwrap();
    }

    #[test]
    fn failed_path_encode_preserves_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.wav");
        std::fs::write(&destination, b"existing output").unwrap();
        let invalid_audio = Audio {
            sample_rate: 48_000,
            channels: Vec::new(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };

        let error =
            write_audio(&destination, &invalid_audio, EncodeOptions::default()).unwrap_err();

        assert!(error.contains("at least one channel"));
        assert_eq!(std::fs::read(&destination).unwrap(), b"existing output");
        assert!(std::fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .all(|entry| !entry.file_name().to_string_lossy().starts_with(".denoize-")));
    }

    #[test]
    fn invalid_request_does_not_seek_or_truncate_open_file() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("staging.bin");
        let original = b"existing staging bytes";
        std::fs::write(&destination, original).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&destination)
            .unwrap();
        file.seek(SeekFrom::End(0)).unwrap();
        let position = file.stream_position().unwrap();
        let invalid_audio = test_audio(12_345, 1, 1);

        let error = write_audio_to_file(
            &mut file,
            OutputFormat::Mp3,
            &invalid_audio,
            EncodeOptions::default(),
        )
        .unwrap_err();

        assert!(error.contains("unsupported sample rate"));
        assert_eq!(file.stream_position().unwrap(), position);
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, original);
    }

    #[test]
    fn oversized_opus_resampler_is_rejected_before_truncating_open_file() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("staging.bin");
        let original = b"existing staging bytes";
        std::fs::write(&destination, original).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&destination)
            .unwrap();
        file.seek(SeekFrom::End(0)).unwrap();
        let position = file.stream_position().unwrap();

        let error = write_audio_to_file(
            &mut file,
            OutputFormat::OggOpus,
            &test_audio(1, 1, 1),
            EncodeOptions::default(),
        )
        .unwrap_err();

        assert!(error.contains("working set"), "unexpected error: {error}");
        assert_eq!(file.stream_position().unwrap(), position);
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, original);
    }

    #[test]
    fn invalid_structure_is_rejected_before_output_staging() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.mp3");
        std::fs::write(&destination, b"existing output").unwrap();
        let mut invalid_audio = test_audio(44_100, 2, 2);
        invalid_audio.channels[1].pop();

        let error =
            write_audio(&destination, &invalid_audio, EncodeOptions::default()).unwrap_err();

        assert!(error.contains("expected 2"));
        assert_eq!(std::fs::read(&destination).unwrap(), b"existing output");
        assert!(std::fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .all(|entry| !entry.file_name().to_string_lossy().starts_with(".denoize-")));
    }
}
