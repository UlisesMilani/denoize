//! `denoize` — pure-Rust audio denoiser built for the world's highest fidelity.
//!
//! Goal: transparent, artifact-free restoration that preserves timbre,
//! transients, dynamics, and "air" better than any classical offline tool.
//!
//! ## Implemented technologies
//!
//! ### Classical DSP (always available)
//! - STFT/ISTFT + Perfect Reconstruction OLA（高オーバーラップ対応）
//! - IMCRA/MCRA ノイズ推定 + Spectral Flatness プロファイル + Anchoring
//! - Ephraim-Malah Decision-Directed SNR
//! - 8種類のゲイン推定器（OMLSA, LogMMSE, MMSE-STSA, Wiener, SpecSub + 非線形/幾何学的）
//! - Attack/Release + Cepstral Smoothing + Transient Protection
//! - 高度窓関数: Kaiser / Flat-top / DPSS
//! - マルチバンドスペクトルサブトラクション
//! - 知覚重み付け（Bark帯域）+ 音楽ノイズ抑制ポストフィルタ
//!
//! ### Input / output codecs (built-in, no ffmpeg)
//! - **Decode**: WAV / MP3 (`nanomp3`) / M4A (Pure Rust AAC-LC)
//! - **Encode**: WAV / MP3 (`shine-rs`) / M4A (`oxideav-aac` Pure-Rust AAC-LC)
//! - Decoded to `f64` PCM at native sample rate (no extra quantisation)
//!
//! ### Optional AI backends (feature-gated)
//! - `rnnoise` feature: RNNoise via nnnoiseless (pure-Rust)
//! - `deepfilter` feature: DeepFilterNet v3 via tract ONNX
//! - `onnx` feature: user-supplied waveform ONNX models via tract
//! - `mpsenet` feature: MP-SENet compressed-magnitude/phase ONNX adapter
//! - `bsrnn` feature: ESPnet BSRNN spectral ONNX adapter
//! - `mossformer2` feature: ClearerVoice MossFormer2 48 kHz ONNX adapter
//! - `sgmse` feature: SGMSE+ iterative diffusion ONNX adapter
//!
//! Build with all backends: `cargo build --release --features full`

pub mod atomic_output;
pub mod audio;
pub mod backend;
pub mod benchmark;
pub mod bessel;
pub mod channel_layout;
pub mod decode;
pub mod denoiser;
pub mod encode;
pub mod fft;
pub mod gain;
#[cfg(feature = "live")]
pub mod live;
pub mod loudness;
pub mod metadata;
pub mod models;
pub mod noise;
pub mod perceptual;
pub mod postfilter;
pub mod quality;
pub mod resample;
pub mod service;
pub mod stft;
mod stoi_resample;
pub mod stream;
pub mod vad;
pub mod window;

pub use atomic_output::{AtomicOutput, CommitMode};
pub use audio::{
    ensure_memory_limit, estimate_audio_memory_bytes, estimate_audio_working_set_bytes,
    estimate_file_memory_bytes, estimate_stream_memory_bytes, read_audio, read_wav, read_wav_bytes,
    sanitize_sample, write_audio, write_wav, write_wav_bytes, write_wav_channel_mask, Audio,
    WavStreamReader, WavStreamWriter,
};
pub use backend::{
    decode_mid_side, encode_mid_side, Backend, BackendOptions, ChannelMode, OnnxModelConfig,
    SgmseProfile,
};
pub use benchmark::{ArtifactReport, BenchmarkReport, ComparisonReport};
pub use channel_layout::{ChannelLayout, ChannelMask, ChannelPosition, PanInfo};
pub use decode::{decode_file, probe_file, AudioCodec, AudioFormat, AudioProbe, DecodedPcm};
pub use denoiser::{Denoiser, DenoiserConfig, Preset, ProcessingMode, StreamingDenoiser};
pub use encode::{AacEncoder, DownmixMode, EncodeOptions, OutputFormat};
pub use gain::{Algorithm, SpecSubLaw};
pub use quality::QualityMetrics;
pub use window::{WindowParams, WindowType};

/// Encode audio and optional metadata into a staged file, then publish it in
/// one filesystem commit.
pub fn write_audio_transactional(
    output: impl AsRef<std::path::Path>,
    audio: &Audio,
    encode_options: EncodeOptions,
    metadata_snapshot: Option<metadata::Metadata>,
    commit_mode: CommitMode,
) -> Result<(), String> {
    let output = output.as_ref();
    let format = OutputFormat::from_path(output)?;
    write_audio_transactional_as(
        output,
        format,
        audio,
        encode_options,
        metadata_snapshot,
        commit_mode,
    )
}

/// Encode audio using a format selected during preflight, then publish it in
/// one filesystem commit without re-inferring the codec from the path.
pub fn write_audio_transactional_as(
    output: impl AsRef<std::path::Path>,
    format: OutputFormat,
    audio: &Audio,
    encode_options: EncodeOptions,
    metadata_snapshot: Option<metadata::Metadata>,
    commit_mode: CommitMode,
) -> Result<(), String> {
    let output = output.as_ref();
    format.validate_encoder(encode_options.aac_encoder)?;
    let mut transaction = AtomicOutput::new(output)?;
    encode::write_audio_to_file(transaction.file_mut(), format, audio, encode_options)?;
    if let Some(metadata_snapshot) = metadata_snapshot {
        metadata::write_extended_to_file(metadata_snapshot, transaction.file_mut())?;
    }
    transaction.commit(commit_mode)
}

/// Denoise a WAV file end-to-end, writing the result to `output`.
pub fn denoise_file<P1, P2>(input: P1, output: P2, config: DenoiserConfig) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend(input, output, config, Backend::Classical)
}

/// Denoise with an explicit backend (classical / rnnoise / deepfilter).
pub fn denoise_file_with_backend<P1, P2>(
    input: P1,
    output: P2,
    config: DenoiserConfig,
    backend: Backend,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend_opts(input, output, config, backend, EncodeOptions::default())
}

/// Denoise with explicit backend and output encode options.
pub fn denoise_file_with_backend_opts<P1, P2>(
    input: P1,
    output: P2,
    config: DenoiserConfig,
    backend: Backend,
    encode_opts: EncodeOptions,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    denoise_file_with_backend_config(
        input,
        output,
        config,
        backend,
        encode_opts,
        BackendOptions::default(),
    )
}

/// Denoise with explicit backend, encoder, and backend-specific model options.
pub fn denoise_file_with_backend_config<P1, P2>(
    input: P1,
    output: P2,
    config: DenoiserConfig,
    backend: Backend,
    encode_opts: EncodeOptions,
    backend_options: BackendOptions,
) -> Result<Audio, String>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    let input = input.as_ref();
    let output = output.as_ref();
    if backend == Backend::Classical {
        config.validate()?;
    }
    let metadata = metadata::read_extended(input)?;
    let mut audio = read_audio(input)?;
    denoise_audio_with_backend_config(&mut audio, config, backend, &backend_options)?;
    write_audio_transactional(output, &audio, encode_opts, metadata, CommitMode::Replace)?;
    Ok(audio)
}

/// Process already-decoded audio in place. This is the path used by stdin and
/// embedders that do not have filesystem-backed input.
pub fn denoise_audio_with_backend_config(
    audio: &mut Audio,
    mut config: DenoiserConfig,
    backend: Backend,
    backend_options: &BackendOptions,
) -> Result<std::time::Duration, String> {
    config.sample_rate = audio.sample_rate;
    if backend == Backend::Classical {
        config.validate()?;
    }
    audio.sanitize_samples();
    let t0 = std::time::Instant::now();
    audio.channels = if config.vad {
        process_with_vad(
            backend,
            &audio.channels,
            audio.sample_rate,
            &config,
            backend_options,
        )?
    } else {
        backend::process_channels(
            backend,
            &audio.channels,
            audio.sample_rate,
            &config,
            backend_options,
        )?
    };
    audio.sanitize_samples();
    let elapsed = t0.elapsed();
    eprintln!(
        "denoize: {:?} | {}ch x {} frames ({:.2}s) in {:.2?} ({:.1}x realtime)",
        backend,
        audio.channels(),
        audio.frames(),
        audio.frames() as f64 / audio.sample_rate as f64,
        elapsed,
        (audio.frames() as f64 / audio.sample_rate as f64) / elapsed.as_secs_f64().max(1e-9),
    );
    Ok(elapsed)
}

fn process_with_vad(
    backend: Backend,
    channels: &[Vec<f64>],
    sample_rate: u32,
    config: &DenoiserConfig,
    backend_options: &BackendOptions,
) -> Result<Vec<Vec<f64>>, String> {
    let regions = vad::speech_regions(channels, sample_rate);
    let fade_frames = (sample_rate as usize / 50).max(1); // 20 ms
    let silence_gain = config.vad_silence_gain;
    let speech_mix = config.vad_speech_mix;
    let mut output: Vec<Vec<f64>> = channels
        .iter()
        .map(|channel| channel.iter().map(|sample| sample * silence_gain).collect())
        .collect();
    for region in regions {
        let input: Vec<Vec<f64>> = channels
            .iter()
            .map(|channel| {
                channel[region.start.min(channel.len())..region.end.min(channel.len())].to_vec()
            })
            .collect();
        let enhanced =
            backend::process_channels(backend, &input, sample_rate, config, backend_options)?;
        for (channel_index, enhanced_channel) in enhanced.iter().enumerate() {
            let Some(destination) = output.get_mut(channel_index) else {
                continue;
            };
            let original = &channels[channel_index];
            for (offset, sample) in enhanced_channel.iter().enumerate() {
                let index = region.start + offset;
                if index >= destination.len() || index >= original.len() || index >= region.end {
                    break;
                }
                let target = sample * speech_mix + original[index] * (1.0 - speech_mix);
                let weight = vad_mix_weight(offset, region.end - region.start, fade_frames);
                destination[index] = destination[index] * (1.0 - weight) + target * weight;
            }
        }
    }
    Ok(output)
}

fn vad_mix_weight(offset: usize, length: usize, fade_frames: usize) -> f64 {
    // Start and end at the attenuated signal so a processed region cannot
    // introduce a discontinuity at either handoff.
    let from_start = offset.min(fade_frames) as f64 / fade_frames.max(1) as f64;
    let from_end =
        length.saturating_sub(offset + 1).min(fade_frames) as f64 / fade_frames.max(1) as f64;
    from_start.min(from_end).clamp(0.0, 1.0)
}

#[cfg(test)]
mod vad_mix_tests {
    use super::{process_with_vad, vad, vad_mix_weight, Backend, BackendOptions, DenoiserConfig};

    #[test]
    fn fades_vad_region_edges_without_exceeding_unity() {
        assert_eq!(vad_mix_weight(0, 100, 10), 0.0);
        assert_eq!(vad_mix_weight(99, 100, 10), 0.0);
        assert_eq!(vad_mix_weight(50, 100, 10), 1.0);
        assert!((vad_mix_weight(5, 100, 10) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn fade_weights_are_bounded_monotonic_and_slope_limited() {
        let fade_frames = 10;
        let weights: Vec<_> = (0..100)
            .map(|offset| vad_mix_weight(offset, 100, fade_frames))
            .collect();

        assert!(weights.iter().all(|weight| (0.0..=1.0).contains(weight)));
        assert!(weights.windows(2).all(|pair| {
            (pair[1] - pair[0]).abs() <= 1.0 / fade_frames as f64 + f64::EPSILON
        }));
        assert!(weights[..=fade_frames]
            .windows(2)
            .all(|pair| pair[1] >= pair[0]));
        assert!(weights[fade_frames..]
            .windows(2)
            .all(|pair| pair[1] <= pair[0]));
        assert_eq!(weights.first().copied(), Some(0.0));
        assert_eq!(weights.last().copied(), Some(0.0));
    }

    fn test_config(sample_rate: u32) -> DenoiserConfig {
        let mut config = DenoiserConfig::default(sample_rate);
        config.vad = true;
        config.vad_silence_gain = 0.2;
        config.vad_speech_mix = 0.0;
        config.sanitized()
    }

    #[test]
    fn vad_applies_configured_gain_to_non_speech_audio() {
        let sample_rate = 16_000;
        let input: Vec<f64> = (0..sample_rate)
            .map(|index| {
                1.0e-5 * (2.0 * std::f64::consts::PI * 37.0 * index as f64
                    / sample_rate as f64)
                    .sin()
            })
            .collect();
        assert!(vad::speech_regions(std::slice::from_ref(&input), sample_rate).is_empty());

        let output = process_with_vad(
            Backend::Classical,
            std::slice::from_ref(&input),
            sample_rate,
            &test_config(sample_rate),
            &BackendOptions::default(),
        )
        .unwrap();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), input.len());
        for (actual, original) in output[0].iter().zip(&input) {
            assert!((actual - original * 0.2).abs() < 1e-20);
        }
    }

    #[test]
    fn vad_crossfade_matches_expected_edges_without_clicks() {
        let sample_rate = 16_000;
        let frames = sample_rate as usize * 2;
        let active_start = sample_rate as usize / 2;
        let active_end = sample_rate as usize * 3 / 2;
        let transition = sample_rate as usize / 20;
        let input: Vec<f64> = (0..frames)
            .map(|index| {
                let envelope = if index < active_start.saturating_sub(transition) {
                    0.0
                } else if index < active_start {
                    let position = (index - (active_start - transition)) as f64
                        / transition as f64;
                    let smooth = position * position * (3.0 - 2.0 * position);
                    0.3 * smooth
                } else if index < active_end {
                    0.3
                } else if index < active_end + transition {
                    let position = (index - active_end) as f64 / transition as f64;
                    let smooth = position * position * (3.0 - 2.0 * position);
                    0.3 * (1.0 - smooth)
                } else {
                    0.0
                };
                envelope
                    * (2.0 * std::f64::consts::PI * 80.0 * index as f64
                        / sample_rate as f64)
                        .sin()
            })
            .collect();
        let regions = vad::speech_regions(std::slice::from_ref(&input), sample_rate);
        assert!(!regions.is_empty());

        let output = process_with_vad(
            Backend::Classical,
            std::slice::from_ref(&input),
            sample_rate,
            &test_config(sample_rate),
            &BackendOptions::default(),
        )
        .unwrap();
        assert_eq!(output[0].len(), input.len());
        assert!(output[0].iter().all(|sample| sample.is_finite()));

        let silence_gain = 0.2;
        let mut expected: Vec<f64> = input.iter().map(|sample| sample * silence_gain).collect();
        for region in &regions {
            for offset in 0..region.end.saturating_sub(region.start) {
                let index = region.start + offset;
                if index >= expected.len() {
                    break;
                }
                let weight = vad_mix_weight(
                    offset,
                    region.end - region.start,
                    sample_rate as usize / 50,
                );
                expected[index] = expected[index] * (1.0 - weight) + input[index] * weight;
            }
        }
        for (actual, expected) in output[0].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }

        for region in regions {
            if region.start > 0 {
                let jump = (output[0][region.start] - output[0][region.start - 1]).abs();
                assert!(jump < 0.02, "VAD start boundary jump: {jump}");
            }
            if region.end < output[0].len() {
                let jump = (output[0][region.end] - output[0][region.end - 1]).abs();
                assert!(jump < 0.02, "VAD end boundary jump: {jump}");
            }
        }
    }
}

#[cfg(test)]
mod input_safety_tests {
    use super::*;

    #[test]
    fn high_level_processing_sanitizes_nonfinite_samples_and_keeps_empty_audio_safe() {
        let mut audio = Audio {
            sample_rate: 16_000,
            channels: vec![vec![f64::NAN, f64::INFINITY, -f64::INFINITY, 2.0, -2.0]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        denoise_audio_with_backend_config(
            &mut audio,
            DenoiserConfig::default(16_000),
            Backend::Classical,
            &BackendOptions::default(),
        )
        .unwrap();
        assert!(audio.channels[0].iter().all(|sample| sample.is_finite()));
        assert!(audio.channels[0].iter().all(|sample| sample.abs() <= 1.0));

        let mut empty = Audio {
            sample_rate: 16_000,
            channels: vec![Vec::new()],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        denoise_audio_with_backend_config(
            &mut empty,
            DenoiserConfig::default(16_000),
            Backend::Classical,
            &BackendOptions::default(),
        )
        .unwrap();
        assert_eq!(empty.frames(), 0);
    }

    #[test]
    fn classical_high_level_processing_rejects_invalid_dpss_bandwidth() {
        let mut audio = Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.25; 512]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        };
        let original = audio.channels.clone();
        let mut config = DenoiserConfig::default(audio.sample_rate);
        config.window = WindowType::Dpss;
        config.window_params.dpss_bandwidth = crate::window::MAX_DENOISER_DPSS_NW + 0.5;

        let error = denoise_audio_with_backend_config(
            &mut audio,
            config,
            Backend::Classical,
            &BackendOptions::default(),
        )
        .unwrap_err();

        assert!(
            error.contains("DPSS bandwidth"),
            "unexpected error: {error}"
        );
        assert_eq!(audio.channels, original);
    }

    #[test]
    fn file_processing_validates_dpss_before_reading_input() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("missing.wav");
        let output = root.path().join("output.wav");
        let mut config = DenoiserConfig::default(16_000);
        config.window = WindowType::Dpss;
        config.window_params.dpss_bandwidth = crate::window::MAX_DENOISER_DPSS_NW + 0.5;

        let error = denoise_file_with_backend_config(
            &input,
            &output,
            config,
            Backend::Classical,
            EncodeOptions::default(),
            BackendOptions::default(),
        )
        .unwrap_err();

        assert!(
            error.contains("DPSS bandwidth"),
            "unexpected error: {error}"
        );
        assert!(!output.exists());
    }
}
