//! MP3 encode via `shine-rs` (Pure Rust, LGPL-2.0).

use std::io::Write;
use std::path::Path;

use shine_rs::{Mp3Encoder, Mp3EncoderConfig, StereoMode, SUPPORTED_BITRATES};

use crate::atomic_output::{AtomicOutput, CommitMode};
use crate::audio::Audio;

use super::pcm::{lossy_channel_layout, planar_f64_to_interleaved_i16, EncodeChannels};
use super::{DownmixMode, EncodeOptions, OutputFormat};

/// Default MP3 bitrate (kbps).
pub const DEFAULT_MP3_BITRATE: u32 = 192;

/// Write planar `f64` audio to an MP3 file.
pub fn write_mp3<P: AsRef<Path>>(path: P, audio: &Audio, bitrate_kbps: u32) -> Result<(), String> {
    write_mp3_with_downmix(path, audio, bitrate_kbps, DownmixMode::Preserve)
}

/// Write planar `f64` audio to MP3 with an explicit surround downmix policy.
pub fn write_mp3_with_downmix<P: AsRef<Path>>(
    path: P,
    audio: &Audio,
    bitrate_kbps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    EncodeOptions {
        mp3_bitrate_kbps: bitrate_kbps,
        downmix,
        ..EncodeOptions::default()
    }
    .validate_config(OutputFormat::Mp3, audio)?;
    let mut output = AtomicOutput::new(path)?;
    write_mp3_to_writer(output.file_mut(), audio, bitrate_kbps, downmix)?;
    output.commit(CommitMode::Replace)
}

pub(super) fn write_mp3_to_writer<W: Write>(
    mut output: W,
    audio: &Audio,
    bitrate_kbps: u32,
    downmix: DownmixMode,
) -> Result<(), String> {
    let (layout, config) = effective_mp3_config(audio, bitrate_kbps, downmix)?;
    let mut pcm = planar_f64_to_interleaved_i16(audio, layout)?;
    let mut encoder = Mp3Encoder::new(config).map_err(|e| format!("mp3 encoder: {e}"))?;

    // A single MPEG frame is too short for common demuxers to establish frame
    // continuity (FFmpeg and Symphonia both reject it). MP3 has no raw-stream
    // field for an exact sub-frame duration, so emit at least two complete
    // frames and let standards-aware Xing/LAME files carry exact timing when
    // that metadata is available.
    let minimum_samples = encoder
        .samples_per_frame()
        .checked_mul(2)
        .ok_or_else(|| "MP3 minimum frame count overflows".to_string())?;
    if pcm.len() < minimum_samples {
        pcm.resize(minimum_samples, 0);
    }

    let mut mp3 = Vec::new();
    for frame in encoder
        .encode_interleaved(&pcm)
        .map_err(|e| format!("mp3 encode: {e}"))?
    {
        mp3.extend(frame);
    }
    mp3.extend(encoder.finish().map_err(|e| format!("mp3 finish: {e}"))?);

    // `shine-rs` 0.1.3 mirrors the C `shine_flush()` helper, which only
    // returns whole bytes already written to the output buffer. The Rust
    // bitstream writer can still hold the final few bytes in its bit cache,
    // leaving the last MPEG frame shorter than the length declared in its
    // header. Flush that cache before collecting the remaining bytes so short
    // clips and the final partial frame remain valid MP3 bitstreams.
    encoder
        .shine_config()
        .bs
        .flush()
        .map_err(|e| format!("mp3 bitstream flush: {e}"))?;
    let config = encoder.shine_config();
    let (flush_data, flush_written) = shine_rs::shine_flush(config);
    mp3.extend_from_slice(&flush_data[..flush_written]);

    output
        .write_all(&mp3)
        .map_err(|e| format!("write mp3: {e}"))?;
    output.flush().map_err(|e| format!("flush mp3: {e}"))
}

pub(super) fn effective_mp3_config(
    audio: &Audio,
    requested_bitrate_kbps: u32,
    downmix: DownmixMode,
) -> Result<(EncodeChannels, Mp3EncoderConfig), String> {
    if audio.frames() == 0 {
        return Err("MP3 output requires at least one frame".into());
    }
    let layout = lossy_channel_layout(audio, downmix)?;
    let stereo_mode = if layout.is_stereo {
        StereoMode::JointStereo
    } else {
        StereoMode::Mono
    };
    let build_config = |bitrate| Mp3EncoderConfig {
        sample_rate: audio.sample_rate,
        bitrate,
        channels: layout.count,
        stereo_mode,
        copyright: false,
        original: true,
    };
    let bitrate = effective_mp3_bitrate_kbps(audio.sample_rate, requested_bitrate_kbps)?;
    let config = build_config(bitrate);
    config
        .validate()
        .map_err(|error| format!("MP3 encoder config: {error}"))?;
    Ok((layout, config))
}

pub(crate) fn effective_mp3_bitrate_kbps(
    sample_rate: u32,
    requested_bitrate_kbps: u32,
) -> Result<u32, String> {
    if !shine_rs::SUPPORTED_SAMPLE_RATES.contains(&sample_rate) {
        return Err(format!(
            "MP3 encode: unsupported sample rate {sample_rate} Hz (supported: {:?})",
            shine_rs::SUPPORTED_SAMPLE_RATES,
        ));
    }
    let build_config = |bitrate| Mp3EncoderConfig {
        sample_rate,
        bitrate,
        channels: 1,
        stereo_mode: StereoMode::Mono,
        copyright: false,
        original: true,
    };
    SUPPORTED_BITRATES
        .iter()
        .copied()
        .filter(|bitrate| *bitrate <= requested_bitrate_kbps)
        .rev()
        .find(|bitrate| build_config(*bitrate).validate().is_ok())
        .or_else(|| {
            SUPPORTED_BITRATES
                .iter()
                .copied()
                .find(|bitrate| build_config(*bitrate).validate().is_ok())
        })
        .ok_or_else(|| format!("MP3 encode: no compatible bitrate for {sample_rate} Hz"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::SampleFormat;

    fn sine_stereo(sr: u32, secs: f32) -> Audio {
        let frames = (sr as f32 * secs) as usize;
        let mut l = Vec::with_capacity(frames);
        let mut r = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f64 / sr as f64;
            let v = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.25;
            l.push(v);
            r.push(v * 0.8);
        }
        Audio {
            sample_rate: sr,
            channels: vec![l, r],
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("denoize_mp3_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn mp3_roundtrip_decode() {
        let path = tmp("rt.mp3");
        let audio = sine_stereo(44100, 0.5);
        write_mp3(&path, &audio, 128).unwrap();
        assert!(path.metadata().unwrap().len() > 100);

        let decoded = crate::decode::decode_file(&path).unwrap();
        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.n_channels(), 2);
        assert_eq!(
            decoded.channel_mask,
            crate::channel_layout::ChannelLayout::Stereo.mask()
        );
        assert!(decoded.frames() > 10000);

        // Lossy but should retain energy
        let rms_in: f64 =
            audio.channels[0].iter().map(|s| s * s).sum::<f64>() / audio.frames() as f64;
        let rms_out: f64 =
            decoded.channels[0].iter().map(|s| s * s).sum::<f64>() / decoded.frames() as f64;
        assert!(rms_out > 0.01);
        assert!(rms_out < rms_in * 2.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_writer_validates_before_staging() {
        let path = tmp("preserve.mp3");
        std::fs::write(&path, b"existing output").unwrap();
        let audio = sine_stereo(12_345, 0.1);

        let error = write_mp3(&path, &audio, 128).unwrap_err();

        assert!(error.contains("unsupported sample rate"));
        assert_eq!(std::fs::read(&path).unwrap(), b"existing output");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn subminimum_bitrate_uses_the_lowest_rate_compatible_fallback() {
        for sample_rate in [32_000, 44_100, 48_000] {
            let audio = sine_stereo(sample_rate, 0.01);
            for requested in [0, 8, 16, 24, 31] {
                let (_, config) =
                    effective_mp3_config(&audio, requested, DownmixMode::Preserve).unwrap();
                assert_eq!(config.bitrate, 32);
                config.validate().unwrap();
            }
        }
    }

    #[test]
    fn compatible_bitrates_still_round_down() {
        let audio = sine_stereo(44_100, 0.01);
        for (requested, expected) in [(33, 32), (191, 160), (u32::MAX, 320)] {
            let (_, config) =
                effective_mp3_config(&audio, requested, DownmixMode::Preserve).unwrap();
            assert_eq!(config.bitrate, expected);
            config.validate().unwrap();
        }
    }
}
