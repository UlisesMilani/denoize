//! Optional AI denoising backends (feature-gated).

#[cfg(feature = "onnx")]
pub(crate) mod causal_target_speaker;
mod classical;
#[cfg(feature = "onnx")]
pub(crate) mod meeting_speaker;
mod session;
mod stream;
#[cfg(feature = "onnx")]
pub(crate) mod target_speaker;
#[cfg(any(
    feature = "onnx",
    feature = "mpsenet",
    feature = "bsrnn",
    feature = "mossformer2",
    feature = "sgmse",
    feature = "gtcrn"
))]
mod tract_runtime;

use std::path::PathBuf;

use crate::audio::sanitize_sample;
use crate::config::{ConfigError, MAX_SAMPLE_RATE};

#[inline]
fn finite_sample(sample: f64) -> f64 {
    if sample.is_finite() {
        sample
    } else {
        0.0
    }
}

const MID_SIDE_SCALE: f64 = std::f64::consts::FRAC_1_SQRT_2;

pub use classical::process_classical;
pub use session::BackendSession;
pub use stream::StreamingBackendSession;

/// Denoising backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Enhanced classical DSP pipeline (default).
    Classical,
    /// RNNoise (nnnoiseless — pure-Rust port of Xiph RNNoise).
    #[cfg(feature = "rnnoise")]
    Rnnoise,
    /// DeepFilterNet v3 (tract ONNX, embedded default model).
    #[cfg(feature = "deepfilter")]
    DeepFilter,
    /// User-supplied waveform-to-waveform ONNX model (pure-Rust tract runtime).
    #[cfg(feature = "onnx")]
    Onnx,
    /// MP-SENet magnitude/phase speech enhancement model.
    #[cfg(feature = "mpsenet")]
    MpSenet,
    /// ESPnet band-split recurrent speech enhancement model.
    #[cfg(feature = "bsrnn")]
    Bsrnn,
    /// ClearerVoice MossFormer2 48 kHz speech enhancement model.
    #[cfg(feature = "mossformer2")]
    Mossformer2,
    /// SGMSE+ diffusion speech enhancement model.
    #[cfg(feature = "sgmse")]
    Sgmse,
    /// Official streaming GTCRN speech enhancement model.
    #[cfg(feature = "gtcrn")]
    Gtcrn,
}

/// Configuration for a waveform-to-waveform ONNX enhancement model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxModelConfig {
    /// Path to the ONNX model file.
    pub path: PathBuf,
    /// Sample rate expected and produced by the model.
    pub sample_rate: u32,
}

impl OnnxModelConfig {
    /// Validate model metadata without opening or parsing the model file.
    pub fn validate_config(&self) -> Result<(), ConfigError> {
        if self.sample_rate == 0 || self.sample_rate > MAX_SAMPLE_RATE {
            return Err(ConfigError::invalid(
                "backend_options.onnx.sample_rate",
                "an integer in 1..=768000 Hz",
            ));
        }
        Ok(())
    }
}

/// How a stereo pair is presented to a denoising backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelMode {
    /// Process channels separately (legacy behavior).
    #[default]
    Independent,
    /// Estimate one common correction from the stereo mid signal and apply it
    /// equally to left and right. This preserves the side signal exactly.
    StereoLinked,
    /// Transform left/right to mid/side, denoise both, then reconstruct.
    MidSide,
}

/// Encode equal-length stereo channels into an energy-preserving mid/side pair.
///
/// Both components use the same `1/sqrt(2)` normalization, so the transform is
/// orthonormal and the original left/right samples can be recovered exactly
/// (up to floating-point roundoff) with [`decode_mid_side`].
pub fn encode_mid_side(left: &[f64], right: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
    if left.len() != right.len() {
        return Err("stereo channels must contain the same number of frames".into());
    }
    let (mut mid, mut side) = (
        Vec::with_capacity(left.len()),
        Vec::with_capacity(left.len()),
    );
    for (&left, &right) in left.iter().zip(right) {
        let left = sanitize_sample(left);
        let right = sanitize_sample(right);
        mid.push(finite_sample((left + right) * MID_SIDE_SCALE));
        side.push(finite_sample((left - right) * MID_SIDE_SCALE));
    }
    Ok((mid, side))
}

/// Decode an energy-preserving mid/side pair back to equal-length stereo.
pub fn decode_mid_side(mid: &[f64], side: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
    if mid.len() != side.len() {
        return Err("mid and side channels must contain the same number of frames".into());
    }
    let (mut left, mut right) = (Vec::with_capacity(mid.len()), Vec::with_capacity(mid.len()));
    for (&mid, &side) in mid.iter().zip(side) {
        let mid = finite_sample(mid);
        let side = finite_sample(side);
        left.push(finite_sample((mid + side) * MID_SIDE_SCALE));
        right.push(finite_sample((mid - side) * MID_SIDE_SCALE));
    }
    Ok((left, right))
}

impl ChannelMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "independent" | "separate" => Some(Self::Independent),
            "linked" | "stereo-linked" | "stereo_linked" => Some(Self::StereoLinked),
            "mid-side" | "midside" | "ms" => Some(Self::MidSide),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SgmseProfile {
    Fast,
    #[default]
    Balanced,
    Quality,
}

impl SgmseProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "balanced" | "default" => Some(Self::Balanced),
            "quality" | "high" => Some(Self::Quality),
            _ => None,
        }
    }
    #[cfg(feature = "sgmse")]
    pub(crate) fn steps(self) -> usize {
        match self {
            Self::Fast => 8,
            Self::Balanced => 20,
            Self::Quality => 30,
        }
    }
}

/// Options used by backends that require external model configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackendOptions {
    /// Model configuration used by the `onnx` backend when that feature is enabled.
    pub onnx: Option<OnnxModelConfig>,
    /// Authenticated custom-model package used by the generic `onnx` backend
    /// or the dedicated BSRNN spectral adapter.
    ///
    /// Prefer [`BackendOptions::with_runtime_model_package`] so the compatible
    /// path/rate identity is populated atomically. Raw ONNX configuration and
    /// runtime packages cannot be mixed.
    pub runtime_package: Option<crate::RuntimeModelPackage>,
    /// Stereo channel coupling strategy.
    pub channel_mode: ChannelMode,
    /// SGMSE+ diffusion budget.
    pub sgmse_profile: SgmseProfile,
    /// Serialize backend work that can otherwise run in parallel.
    ///
    /// This is the processing-side part of reproducible output mode. It keeps
    /// channel inference and batch scheduling in a stable order. Container
    /// timestamps and diagnostic timings are not affected by this flag.
    pub deterministic: bool,
    /// Requested inference accelerator policy.
    ///
    /// CPU remains the compatibility default. `Auto` may select a usable GPU
    /// for tract-backed adapters and records an explicit CPU fallback reason.
    pub accelerator: crate::AcceleratorPreference,
    /// Optional seed for stochastic backends such as SGMSE+.
    ///
    /// Supplying a seed makes the stochastic sampler repeatable. `None` uses
    /// the backend's stable default seed for backwards-compatible output.
    pub seed: Option<u64>,
}

impl BackendOptions {
    /// Select one already verified custom-model package.
    #[must_use]
    pub fn with_runtime_model_package(mut self, package: crate::RuntimeModelPackage) -> Self {
        self.onnx = Some(package.model_config());
        self.runtime_package = Some(package);
        self
    }

    /// Validate backend options before any model path is inspected.
    ///
    /// A missing GTCRN model remains valid at this stage because application
    /// services may resolve it from the managed model library. Backends whose
    /// models must be supplied by the caller reject a missing configuration.
    pub fn validate_config(&self, backend: Backend) -> Result<(), ConfigError> {
        if let Some(package) = &self.runtime_package {
            if !backend_accepts_runtime_package(backend) {
                return Err(ConfigError::invalid(
                    "backend_options.runtime_package",
                    "a package used by the generic onnx or dedicated bsrnn backend",
                ));
            }
            if self.onnx.as_ref() != Some(&package.model_config()) {
                return Err(ConfigError::invalid(
                    "backend_options.runtime_package",
                    "a package selected through with_runtime_model_package without a conflicting raw ONNX model",
                ));
            }
        }
        if let Some(model) = &self.onnx {
            model.validate_config()?;
            validate_named_model_rate(backend, model.sample_rate)?;
        } else if requires_caller_model(backend) {
            return Err(missing_model_error());
        }
        Ok(())
    }

    /// Validate options after managed model resolution has completed.
    pub(crate) fn validate_resolved_config(&self, backend: Backend) -> Result<(), ConfigError> {
        self.validate_config(backend)?;
        if requires_any_model(backend) && self.onnx.is_none() {
            return Err(missing_model_error());
        }
        Ok(())
    }

    /// Validate the selected model resource after managed resolution.
    ///
    /// Adapters still validate the file they open. This early check provides
    /// deterministic error ordering, but is not a substitute for consuming a
    /// retained file handle when stronger path-race protection is required.
    pub fn validate_resolved_resources(&self, backend: Backend) -> Result<(), String> {
        self.validate_resolved_config(backend)
            .map_err(|error| error.to_string())?;
        if requires_any_model(backend) {
            if let Some(package) = &self.runtime_package {
                if !package.package_path().is_file() {
                    return Err(format!(
                        "selected runtime model package does not exist or is not a file: {}",
                        package.package_path().display()
                    ));
                }
                return Ok(());
            }
            let model = self
                .onnx
                .as_ref()
                .expect("resolved model presence was validated");
            if !model.path.is_file() {
                return Err(format!(
                    "selected backend model does not exist or is not a file: {}",
                    model.path.display()
                ));
            }
        }
        Ok(())
    }
}

fn backend_accepts_runtime_package(backend: Backend) -> bool {
    #[cfg(feature = "onnx")]
    {
        if backend == Backend::Onnx {
            return true;
        }
        #[cfg(feature = "bsrnn")]
        if backend == Backend::Bsrnn {
            return true;
        }
        false
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = backend;
        false
    }
}

fn missing_model_error() -> ConfigError {
    ConfigError::invalid(
        "backend_options.onnx",
        "a model configuration for the selected backend",
    )
}

fn requires_caller_model(backend: Backend) -> bool {
    match backend {
        #[cfg(feature = "onnx")]
        Backend::Onnx => true,
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => true,
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => true,
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => true,
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => true,
        _ => false,
    }
}

fn requires_any_model(backend: Backend) -> bool {
    if requires_caller_model(backend) {
        return true;
    }
    match backend {
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => true,
        _ => false,
    }
}

fn validate_named_model_rate(backend: Backend, sample_rate: u32) -> Result<(), ConfigError> {
    let expected: Option<(u32, &'static str)> = match backend {
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => Some((16_000, "exactly 16000 Hz for MP-SENet")),
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => Some((48_000, "exactly 48000 Hz for BSRNN")),
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => Some((48_000, "exactly 48000 Hz for MossFormer2")),
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => Some((16_000, "exactly 16000 Hz for SGMSE+")),
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => Some((16_000, "exactly 16000 Hz for GTCRN")),
        _ => None,
    };
    if let Some((required, description)) = expected {
        if sample_rate != required {
            return Err(ConfigError::invalid(
                "backend_options.onnx.sample_rate",
                description,
            ));
        }
    }
    Ok(())
}

impl Backend {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "classical" | "dsp" | "stft" => Backend::Classical,
            #[cfg(feature = "rnnoise")]
            "rnnoise" | "rnn" => Backend::Rnnoise,
            #[cfg(feature = "deepfilter")]
            "deepfilter" | "deepfilternet" | "dfn" | "dfn3" => Backend::DeepFilter,
            #[cfg(feature = "onnx")]
            "onnx" | "model" => Backend::Onnx,
            #[cfg(feature = "mpsenet")]
            "mpsenet" | "mp-senet" | "mp_senet" => Backend::MpSenet,
            #[cfg(feature = "bsrnn")]
            "bsrnn" | "bs-rnn" | "bs_rnn" => Backend::Bsrnn,
            #[cfg(feature = "mossformer2")]
            "mossformer2" | "moss-former2" | "mossformer" => Backend::Mossformer2,
            #[cfg(feature = "sgmse")]
            "sgmse" | "sgmse+" | "sgmse-plus" => Backend::Sgmse,
            #[cfg(feature = "gtcrn")]
            "gtcrn" => Backend::Gtcrn,
            #[cfg(not(feature = "rnnoise"))]
            "rnnoise" | "rnn" => return None,
            #[cfg(not(feature = "deepfilter"))]
            "deepfilter" | "deepfilternet" | "dfn" | "dfn3" => return None,
            #[cfg(not(feature = "onnx"))]
            "onnx" | "model" => return None,
            #[cfg(not(feature = "mpsenet"))]
            "mpsenet" | "mp-senet" | "mp_senet" => return None,
            #[cfg(not(feature = "bsrnn"))]
            "bsrnn" | "bs-rnn" | "bs_rnn" => return None,
            #[cfg(not(feature = "mossformer2"))]
            "mossformer2" | "moss-former2" | "mossformer" => return None,
            #[cfg(not(feature = "sgmse"))]
            "sgmse" | "sgmse+" | "sgmse-plus" => return None,
            #[cfg(not(feature = "gtcrn"))]
            "gtcrn" => return None,
            _ => return None,
        })
    }

    pub fn available_names() -> &'static [&'static str] {
        &[
            "classical",
            #[cfg(feature = "rnnoise")]
            "rnnoise",
            #[cfg(feature = "deepfilter")]
            "deepfilter",
            #[cfg(feature = "onnx")]
            "onnx",
            #[cfg(feature = "mpsenet")]
            "mpsenet",
            #[cfg(feature = "bsrnn")]
            "bsrnn",
            #[cfg(feature = "mossformer2")]
            "mossformer2",
            #[cfg(feature = "sgmse")]
            "sgmse",
            #[cfg(feature = "gtcrn")]
            "gtcrn",
        ]
    }
}

/// Process all channels with the selected backend.
pub fn process_channels(
    backend: Backend,
    channels: &[Vec<f64>],
    sample_rate: u32,
    classical_cfg: &crate::denoiser::DenoiserConfig,
    backend_options: &BackendOptions,
) -> Result<Vec<Vec<f64>>, String> {
    BackendSession::prepare(backend, backend_options.clone())?.process(
        channels,
        sample_rate,
        classical_cfg,
    )
}

#[cfg(feature = "rnnoise")]
pub mod rnnoise;

#[cfg(feature = "deepfilter")]
pub mod deepfilter;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "mpsenet")]
pub mod mpsenet;

#[cfg(feature = "bsrnn")]
pub mod bsrnn;

#[cfg(feature = "mossformer2")]
pub mod mossformer2;

#[cfg(feature = "sgmse")]
pub mod sgmse;

#[cfg(feature = "gtcrn")]
pub mod gtcrn;

#[cfg(test)]
mod channel_tests {
    use super::*;

    fn model_options(sample_rate: u32) -> BackendOptions {
        BackendOptions {
            onnx: Some(OnnxModelConfig {
                path: PathBuf::from("model-that-must-not-be-opened.onnx"),
                sample_rate,
            }),
            ..BackendOptions::default()
        }
    }

    #[test]
    fn parses_channel_modes() {
        assert_eq!(
            ChannelMode::parse("linked"),
            Some(ChannelMode::StereoLinked)
        );
        assert_eq!(ChannelMode::parse("mid-side"), Some(ChannelMode::MidSide));
        assert_eq!(
            ChannelMode::parse("independent"),
            Some(ChannelMode::Independent)
        );
    }

    #[test]
    fn mid_side_transform_roundtrips_stereo() {
        let left = vec![0.0, 0.25, -0.75, 1.0];
        let right = vec![0.5, -0.25, 0.75, -1.0];
        let (mid, side) = encode_mid_side(&left, &right).unwrap();
        let (decoded_left, decoded_right) = decode_mid_side(&mid, &side).unwrap();
        for (actual, expected) in decoded_left.iter().zip(&left) {
            assert!((actual - expected).abs() < 1e-14);
        }
        for (actual, expected) in decoded_right.iter().zip(&right) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn mid_side_rejects_mismatched_lengths() {
        assert!(encode_mid_side(&[0.0], &[]).is_err());
        assert!(decode_mid_side(&[0.0], &[]).is_err());
    }

    #[test]
    fn stereo_linked_preserves_the_side_signal() {
        let frames = 4_096;
        let left: Vec<f64> = (0..frames)
            .map(|i| (i as f64 * 0.013).sin() * 0.4)
            .collect();
        let right: Vec<f64> = (0..frames)
            .map(|i| (i as f64 * 0.017).sin() * 0.3)
            .collect();
        let input = vec![left, right];
        let options = BackendOptions {
            channel_mode: ChannelMode::StereoLinked,
            ..BackendOptions::default()
        };
        let output = process_channels(
            Backend::Classical,
            &input,
            48_000,
            &crate::denoiser::DenoiserConfig::default(48_000),
            &options,
        )
        .unwrap();
        for index in 0..frames {
            let before = input[0][index] - input[1][index];
            let after = output[0][index] - output[1][index];
            assert!((before - after).abs() < 1e-12);
        }
    }

    #[test]
    fn model_rate_is_bounded_even_when_the_option_is_dormant() {
        assert!(model_options(1).validate_config(Backend::Classical).is_ok());
        assert!(model_options(MAX_SAMPLE_RATE)
            .validate_config(Backend::Classical)
            .is_ok());
        for invalid in [0, MAX_SAMPLE_RATE + 1] {
            assert!(matches!(
                model_options(invalid).validate_config(Backend::Classical),
                Err(ConfigError::InvalidValue {
                    field: "backend_options.onnx.sample_rate",
                    ..
                })
            ));
        }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn authenticated_package_is_bound_to_the_generic_onnx_backend_and_model_identity() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.onnx");
        std::fs::write(&model, b"test model bytes").unwrap();
        let package = crate::RuntimeModelPackage::for_onnx_contract_test(
            model,
            crate::RuntimeModelTensorContract {
                element_type: "float32".into(),
                layout: "batch-samples".into(),
                fixed_input_samples: None,
                fixed_output_samples: None,
            },
        );
        let mut options = BackendOptions::default().with_runtime_model_package(package);
        options.validate_config(Backend::Onnx).unwrap();
        assert!(options
            .validate_config(Backend::Classical)
            .unwrap_err()
            .to_string()
            .contains("generic onnx"));

        options.onnx.as_mut().unwrap().sample_rate = 48_000;
        assert!(options
            .validate_config(Backend::Onnx)
            .unwrap_err()
            .to_string()
            .contains("conflicting raw ONNX"));
    }

    #[test]
    fn processing_validates_effective_core_config_before_samples() {
        let samples = vec![vec![f64::INFINITY]];
        let mut config = crate::denoiser::DenoiserConfig::default(48_000);
        config.strength = f64::NAN;
        let error = process_channels(
            Backend::Classical,
            &samples,
            48_000,
            &config,
            &BackendOptions::default(),
        )
        .unwrap_err();
        assert!(error.contains("`strength`"));

        let mut stale_rate = crate::denoiser::DenoiserConfig::default(0);
        stale_rate.strength = 0.0;
        assert!(process_channels(
            Backend::Classical,
            &[vec![0.0]],
            48_000,
            &stale_rate,
            &BackendOptions::default(),
        )
        .is_ok());
        assert!(process_channels(
            Backend::Classical,
            &[vec![0.0]],
            0,
            &stale_rate,
            &BackendOptions::default(),
        )
        .unwrap_err()
        .contains("`sample_rate`"));
    }

    #[test]
    fn named_model_rates_are_checked_before_model_paths() {
        #[allow(unused_mut)]
        let mut cases: Vec<(Backend, u32)> = Vec::new();
        #[cfg(feature = "mpsenet")]
        cases.push((Backend::MpSenet, 16_000));
        #[cfg(feature = "bsrnn")]
        cases.push((Backend::Bsrnn, 48_000));
        #[cfg(feature = "mossformer2")]
        cases.push((Backend::Mossformer2, 48_000));
        #[cfg(feature = "sgmse")]
        cases.push((Backend::Sgmse, 16_000));
        #[cfg(feature = "gtcrn")]
        cases.push((Backend::Gtcrn, 16_000));

        for (backend, required) in cases {
            assert!(model_options(required).validate_config(backend).is_ok());
            let wrong = if required == 16_000 { 48_000 } else { 16_000 };
            assert!(matches!(
                model_options(wrong).validate_config(backend),
                Err(ConfigError::InvalidValue {
                    field: "backend_options.onnx.sample_rate",
                    ..
                })
            ));
        }
    }

    #[test]
    fn caller_supplied_model_backends_reject_missing_configuration() {
        let options = BackendOptions::default();
        #[allow(unused_mut)]
        let mut backends: Vec<Backend> = Vec::new();
        #[cfg(feature = "onnx")]
        backends.push(Backend::Onnx);
        #[cfg(feature = "mpsenet")]
        backends.push(Backend::MpSenet);
        #[cfg(feature = "bsrnn")]
        backends.push(Backend::Bsrnn);
        #[cfg(feature = "mossformer2")]
        backends.push(Backend::Mossformer2);
        #[cfg(feature = "sgmse")]
        backends.push(Backend::Sgmse);
        for backend in backends {
            assert!(matches!(
                options.validate_config(backend),
                Err(ConfigError::InvalidValue {
                    field: "backend_options.onnx",
                    ..
                })
            ));
        }
    }

    #[cfg(feature = "gtcrn")]
    #[test]
    fn managed_gtcrn_may_be_missing_only_before_resolution() {
        let options = BackendOptions::default();
        assert!(options.validate_config(Backend::Gtcrn).is_ok());
        assert!(matches!(
            options.validate_resolved_config(Backend::Gtcrn),
            Err(ConfigError::InvalidValue {
                field: "backend_options.onnx",
                ..
            })
        ));
    }
}

#[cfg(all(
    test,
    any(
        feature = "mpsenet",
        feature = "bsrnn",
        feature = "mossformer2",
        feature = "sgmse"
    )
))]
mod tests {
    use super::*;

    #[cfg(feature = "mpsenet")]
    #[test]
    fn parses_mp_senet_aliases() {
        assert_eq!(Backend::parse("mpsenet"), Some(Backend::MpSenet));
        assert_eq!(Backend::parse("mp-senet"), Some(Backend::MpSenet));
        assert!(Backend::available_names().contains(&"mpsenet"));
    }

    #[cfg(feature = "bsrnn")]
    #[test]
    fn parses_bsrnn_aliases() {
        assert_eq!(Backend::parse("bsrnn"), Some(Backend::Bsrnn));
        assert_eq!(Backend::parse("bs-rnn"), Some(Backend::Bsrnn));
        assert!(Backend::available_names().contains(&"bsrnn"));
    }

    #[cfg(feature = "mossformer2")]
    #[test]
    fn parses_mossformer2_aliases() {
        assert_eq!(Backend::parse("mossformer2"), Some(Backend::Mossformer2));
        assert_eq!(Backend::parse("moss-former2"), Some(Backend::Mossformer2));
        assert!(Backend::available_names().contains(&"mossformer2"));
    }

    #[cfg(feature = "sgmse")]
    #[test]
    fn parses_sgmse_aliases() {
        assert_eq!(Backend::parse("sgmse"), Some(Backend::Sgmse));
        assert_eq!(Backend::parse("sgmse+"), Some(Backend::Sgmse));
        assert!(Backend::available_names().contains(&"sgmse"));
    }
}
