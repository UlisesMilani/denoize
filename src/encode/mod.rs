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
            Some("opus" | "ogg") => Ok(OutputFormat::OggOpus),
            Some("mp3") => Ok(OutputFormat::Mp3),
            Some("m4a" | "m4b" | "mp4") => Ok(OutputFormat::M4a),
            Some("aac") => Ok(OutputFormat::AacAdts),
            Some(ext) => Err(format!(
                "unsupported output format '.{ext}'; use .wav, .flac, .opus, .mp3, .m4a, or .aac"
            )),
            None => Err(
                "output path has no extension; use .wav, .flac, .opus, .mp3, .m4a, or .aac".into(),
            ),
        }
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

/// Write audio to a file; format is chosen from the path extension.
pub fn write_audio<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    options: EncodeOptions,
) -> Result<(), String> {
    let path = path.as_ref();
    let format = OutputFormat::from_path(path)?;
    let mut output = AtomicOutput::new(path)?;
    write_audio_to_file(output.file_mut(), format, audio, options)?;
    output.commit(CommitMode::Replace)
}

/// Write audio to an already-open file using an explicitly selected format.
///
/// The file is rewound and truncated before any encoded bytes are written, so
/// callers can safely use a securely-created staging file without reopening a
/// predictable path.
pub fn write_audio_to_file(
    file: &mut File,
    format: OutputFormat,
    audio: &Audio,
    options: EncodeOptions,
) -> Result<(), String> {
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
}
