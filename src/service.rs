//! Shared application service used by the CLI and graphical frontends.

use crate::loudness::LoudnessReport;
#[cfg(feature = "gtcrn")]
use crate::OnnxModelConfig;
use crate::{Audio, Backend, BackendOptions, BackendSession, ConfigError, DenoiserConfig};
use std::time::Duration;

/// User-facing backend choice shared by every application frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Explicit(Backend),
}

/// Options for processing decoded audio.
#[derive(Clone, Debug)]
pub struct ProcessingOptions {
    pub backend: BackendChoice,
    pub quality: Option<String>,
    pub denoiser: DenoiserConfig,
    pub backend_options: BackendOptions,
    pub loudness_lufs: Option<f64>,
    pub true_peak_dbtp: f64,
}

/// Fully selected and validated processing options.
///
/// This is the single effective configuration consumed by both execution and
/// batch recipe fingerprinting. In particular, the denoiser sample rate has
/// already been replaced with the decoded rate, compatibility sanitization
/// has already been applied, and managed backend resources have been resolved.
#[derive(Clone, Debug)]
pub struct ResolvedProcessingOptions {
    pub backend: Backend,
    pub denoiser: DenoiserConfig,
    pub backend_options: BackendOptions,
    pub loudness_lufs: Option<f64>,
    pub true_peak_dbtp: f64,
}

impl ResolvedProcessingOptions {
    /// Validate an already-resolved plan without opening or modifying audio.
    pub fn validate_config(&self) -> Result<(), String> {
        self.denoiser
            .validate_config()
            .map_err(|error| error.to_string())?;
        self.backend_options
            .validate_resolved_resources(self.backend)?;
        if let Some(target) = self.loudness_lufs {
            validate_finite_range(
                "loudness_lufs",
                target,
                -70.0,
                0.0,
                "a finite value in -70..=0 LUFS",
            )
            .map_err(|error| error.to_string())?;
        }
        validate_finite_range(
            "true_peak_dbtp",
            self.true_peak_dbtp,
            -20.0,
            0.0,
            "a finite value in -20..=0 dBTP",
        )
        .map_err(|error| error.to_string())
    }
}

impl ProcessingOptions {
    /// Validate effective processing options without opening a model or
    /// modifying decoded audio. The selected backend is returned so callers
    /// can resolve managed model configuration only after validation succeeds.
    pub fn validate_config(&self, audio: &Audio) -> Result<Backend, ConfigError> {
        if let Some(quality) = self.quality.as_deref() {
            let quality = quality.to_ascii_lowercase();
            if !matches!(quality.as_str(), "high" | "ultra" | "max" | "highest") {
                return Err(ConfigError::invalid(
                    "quality",
                    "one of high, ultra, max, or highest",
                ));
            }
        }

        let duration = audio.frames() as f64 / audio.sample_rate.max(1) as f64;
        let backend = select_backend(self.backend, duration, self.quality.as_deref());
        let mut denoiser = self.denoiser.clone();
        denoiser.sample_rate = audio.sample_rate;
        denoiser.validate_config()?;
        self.backend_options.validate_config(backend)?;

        if let Some(target) = self.loudness_lufs {
            validate_finite_range(
                "loudness_lufs",
                target,
                -70.0,
                0.0,
                "a finite value in -70..=0 LUFS",
            )?;
        }
        validate_finite_range(
            "true_peak_dbtp",
            self.true_peak_dbtp,
            -20.0,
            0.0,
            "a finite value in -20..=0 dBTP",
        )?;
        Ok(backend)
    }

    /// Resolve every effective option without modifying decoded audio.
    pub fn resolve(self, audio: &Audio) -> Result<ResolvedProcessingOptions, String> {
        resolve_processing_options(audio, self)
    }
}

fn validate_finite_range(
    field: &'static str,
    value: f64,
    min: f64,
    max: f64,
    expected: &'static str,
) -> Result<(), ConfigError> {
    if !value.is_finite() || value < min || value > max {
        return Err(ConfigError::invalid(field, expected));
    }
    Ok(())
}

/// Information produced by a completed processing operation.
#[derive(Clone, Copy, Debug)]
pub struct ProcessingResult {
    pub backend: Backend,
    pub elapsed: Duration,
    pub loudness: Option<LoudnessReport>,
}

/// Stable display/configuration name for a compiled backend.
pub fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Classical => "classical",
        #[cfg(feature = "rnnoise")]
        Backend::Rnnoise => "rnnoise",
        #[cfg(feature = "deepfilter")]
        Backend::DeepFilter => "deepfilter",
        #[cfg(feature = "onnx")]
        Backend::Onnx => "onnx",
        #[cfg(feature = "mpsenet")]
        Backend::MpSenet => "mpsenet",
        #[cfg(feature = "bsrnn")]
        Backend::Bsrnn => "bsrnn",
        #[cfg(feature = "mossformer2")]
        Backend::Mossformer2 => "mossformer2",
        #[cfg(feature = "sgmse")]
        Backend::Sgmse => "sgmse",
        #[cfg(feature = "gtcrn")]
        Backend::Gtcrn => "gtcrn",
    }
}

/// Whether a backend needs a user-selected ONNX file rather than embedded or
/// managed weights.
pub fn requires_external_model(backend: Backend) -> bool {
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

/// Select a backend consistently for CLI and graphical processing.
pub fn select_backend(
    choice: BackendChoice,
    _duration_seconds: f64,
    _quality: Option<&str>,
) -> Backend {
    if let BackendChoice::Explicit(backend) = choice {
        return backend;
    }
    #[cfg(feature = "deepfilter")]
    {
        let high_quality = _quality.is_some_and(|quality| {
            matches!(
                quality.to_ascii_lowercase().as_str(),
                "high" | "ultra" | "max" | "highest"
            )
        });
        if high_quality || _duration_seconds <= 10.0 * 60.0 {
            return Backend::DeepFilter;
        }
    }
    #[cfg(feature = "rnnoise")]
    {
        return Backend::Rnnoise;
    }
    #[allow(unreachable_code)]
    Backend::Classical
}

/// Select the low-latency backend preferred for realtime sessions.
pub fn select_live_backend() -> Backend {
    #[cfg(feature = "rnnoise")]
    {
        return Backend::Rnnoise;
    }
    #[allow(unreachable_code)]
    Backend::Classical
}

/// Fill backend options that can be resolved from the managed model library.
pub fn resolve_backend_options(
    _backend: Backend,
    #[allow(unused_mut)] mut options: BackendOptions,
) -> Result<BackendOptions, String> {
    options
        .validate_config(_backend)
        .map_err(|error| error.to_string())?;
    #[cfg(feature = "gtcrn")]
    if _backend == Backend::Gtcrn && options.onnx.is_none() {
        let catalog = crate::models::active_catalog()?;
        let model = catalog
            .find("gtcrn")
            .ok_or_else(|| "active model catalog has no unambiguous GTCRN package".to_string())?;
        options.onnx = Some(OnnxModelConfig {
            path: crate::models::verify_catalog_model(model).map_err(|error| {
                format!(
                    "GTCRN managed model is unavailable ({error}); run `denoize models install gtcrn`"
                )
            })?,
            sample_rate: model.sample_rate(),
        });
    }
    options.validate_resolved_resources(_backend)?;
    Ok(options)
}

/// Select, validate, sanitize, and resolve the exact processing configuration
/// that execution will consume.
pub fn resolve_processing_options(
    audio: &Audio,
    options: ProcessingOptions,
) -> Result<ResolvedProcessingOptions, String> {
    let backend = options
        .validate_config(audio)
        .map_err(|error| error.to_string())?;
    let mut denoiser = options.denoiser;
    denoiser.sample_rate = audio.sample_rate;
    // `Denoiser::try_new`, used by the classical backend, validates first and
    // then applies this compatibility sanitization. Resolve it here so recipe
    // hashing and execution cannot observe two different configurations.
    denoiser
        .validate_config()
        .map_err(|error| error.to_string())?;
    let denoiser = denoiser.sanitized();
    let backend_options = resolve_backend_options(backend, options.backend_options)?;
    Ok(ResolvedProcessingOptions {
        backend,
        denoiser,
        backend_options,
        loudness_lufs: options.loudness_lufs,
        true_peak_dbtp: options.true_peak_dbtp,
    })
}

/// Process decoded audio using an already-resolved effective configuration.
pub fn process_audio_resolved(
    audio: &mut Audio,
    options: &ResolvedProcessingOptions,
) -> Result<ProcessingResult, String> {
    let session = BackendSession::prepare(options.backend, options.backend_options.clone())?;
    process_audio_resolved_with_session(audio, options, &session)
}

/// Process decoded audio with an already-prepared backend session.
pub fn process_audio_resolved_with_session(
    audio: &mut Audio,
    options: &ResolvedProcessingOptions,
    session: &BackendSession,
) -> Result<ProcessingResult, String> {
    if audio.sample_rate != options.denoiser.sample_rate {
        return Err(format!(
            "resolved processing sample rate {} Hz does not match decoded audio rate {} Hz",
            options.denoiser.sample_rate, audio.sample_rate
        ));
    }
    options.validate_config()?;
    if session.backend() != options.backend || session.options() != &options.backend_options {
        return Err("prepared backend session does not match resolved processing options".into());
    }
    let (mut working, elapsed) =
        crate::process_audio_copy_with_backend_session(audio, options.denoiser.clone(), session)?;
    let loudness = options
        .loudness_lufs
        .map(|target| crate::loudness::normalize(&mut working, target, options.true_peak_dbtp))
        .transpose()?;
    *audio = working;
    Ok(ProcessingResult {
        backend: options.backend,
        elapsed,
        loudness,
    })
}

/// Process already-decoded audio with common backend and delivery behavior.
pub fn process_audio(
    audio: &mut Audio,
    options: ProcessingOptions,
) -> Result<ProcessingResult, String> {
    let resolved = resolve_processing_options(audio, options)?;
    process_audio_resolved(audio, &resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio(sample_rate: u32) -> Audio {
        Audio {
            sample_rate,
            channels: vec![vec![0.25; 256]],
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            channel_mask: None,
        }
    }

    fn options() -> ProcessingOptions {
        ProcessingOptions {
            backend: BackendChoice::Explicit(Backend::Classical),
            quality: None,
            denoiser: DenoiserConfig::default(48_000),
            backend_options: BackendOptions::default(),
            loudness_lufs: None,
            true_peak_dbtp: -1.0,
        }
    }

    #[test]
    fn explicit_backend_is_preserved() {
        assert_eq!(
            select_backend(
                BackendChoice::Explicit(Backend::Classical),
                10.0,
                Some("ultra")
            ),
            Backend::Classical
        );
    }

    #[test]
    fn automatic_backend_is_compiled() {
        let selected = select_backend(BackendChoice::Auto, 10.0, None);
        assert!(Backend::available_names().contains(&backend_name(selected)));
    }

    #[test]
    fn live_backend_is_compiled() {
        assert!(Backend::available_names().contains(&backend_name(select_live_backend())));
    }

    #[test]
    fn classical_does_not_require_external_weights() {
        assert!(!requires_external_model(Backend::Classical));
    }

    #[test]
    fn processing_validation_uses_the_decoded_sample_rate() {
        let mut options = options();
        options.denoiser.sample_rate = 0;
        assert_eq!(
            options.validate_config(&audio(48_000)).unwrap(),
            Backend::Classical
        );
        assert!(matches!(
            options.validate_config(&audio(0)),
            Err(ConfigError::InvalidValue {
                field: "sample_rate",
                ..
            })
        ));
        assert!(options
            .validate_config(&audio(crate::config::MAX_SAMPLE_RATE))
            .is_ok());
        assert!(options
            .validate_config(&audio(crate::config::MAX_SAMPLE_RATE + 1))
            .is_err());
    }

    #[test]
    fn resolution_records_the_exact_sanitized_execution_config() {
        let mut options = options();
        options.denoiser.sample_rate = 1;
        options.denoiser.smoothing = 1.0;

        let resolved = resolve_processing_options(&audio(44_100), options).unwrap();

        assert_eq!(resolved.backend, Backend::Classical);
        assert_eq!(resolved.denoiser.sample_rate, 44_100);
        assert_eq!(resolved.denoiser.smoothing, 0.95);
        resolved.validate_config().unwrap();
    }

    #[test]
    fn resolved_sample_rate_mismatch_is_transactional() {
        let mut decoded = audio(48_000);
        decoded.channels[0][0] = 2.0;
        let before = decoded.clone();
        let resolved = resolve_processing_options(&decoded, options()).unwrap();
        decoded.sample_rate = 44_100;

        let error = process_audio_resolved(&mut decoded, &resolved).unwrap_err();

        assert!(error.contains("does not match decoded audio rate"));
        assert_eq!(decoded.channels, before.channels);
        assert_eq!(decoded.sample_rate, 44_100);
    }

    #[test]
    fn quality_is_a_closed_case_insensitive_contract() {
        for quality in ["high", "ultra", "max", "highest", "HIGH"] {
            let mut options = options();
            options.quality = Some(quality.into());
            assert!(options.validate_config(&audio(48_000)).is_ok());
        }
        let mut invalid = options();
        invalid.quality = Some("fastest".into());
        assert!(matches!(
            invalid.validate_config(&audio(48_000)),
            Err(ConfigError::InvalidValue {
                field: "quality",
                ..
            })
        ));
    }

    #[test]
    fn loudness_and_true_peak_ranges_are_checked() {
        for target in [-70.0, 0.0] {
            let mut options = options();
            options.loudness_lufs = Some(target);
            assert!(options.validate_config(&audio(48_000)).is_ok());
        }
        for peak in [-20.0, 0.0] {
            let mut options = options();
            options.true_peak_dbtp = peak;
            assert!(options.validate_config(&audio(48_000)).is_ok());
        }
        for target in [f64::NAN, f64::INFINITY, -70.01, 0.01] {
            let mut options = options();
            options.loudness_lufs = Some(target);
            assert!(matches!(
                options.validate_config(&audio(48_000)),
                Err(ConfigError::InvalidValue {
                    field: "loudness_lufs",
                    ..
                })
            ));
        }
        for peak in [f64::NAN, f64::INFINITY, -20.01, 0.01] {
            let mut options = options();
            options.true_peak_dbtp = peak;
            assert!(matches!(
                options.validate_config(&audio(48_000)),
                Err(ConfigError::InvalidValue {
                    field: "true_peak_dbtp",
                    ..
                })
            ));
        }
    }

    #[test]
    fn invalid_options_do_not_mutate_decoded_audio() {
        let mut decoded = audio(48_000);
        decoded.channels[0][0] = 2.0;
        let before = decoded.channels.clone();
        let mut options = options();
        options.denoiser.strength = f64::NAN;

        let error = process_audio(&mut decoded, options).unwrap_err();

        assert!(error.contains("`strength`"));
        assert_eq!(decoded.channels, before);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn missing_model_resource_does_not_mutate_decoded_audio() {
        let mut decoded = audio(48_000);
        decoded.channels[0][0] = 2.0;
        let before = decoded.channels.clone();
        let mut options = options();
        options.backend = BackendChoice::Explicit(Backend::Onnx);
        options.backend_options.onnx = Some(crate::OnnxModelConfig {
            path: "model-that-does-not-exist.onnx".into(),
            sample_rate: 48_000,
        });

        let error = process_audio(&mut decoded, options).unwrap_err();

        assert!(
            error.contains("model does not exist"),
            "unexpected error: {error}"
        );
        assert_eq!(decoded.channels, before);
    }

    #[test]
    fn loudness_failure_does_not_commit_processed_audio() {
        let mut decoded = audio(48_000);
        decoded.channels[0].fill(0.0);
        decoded.channels[0][0] = 0.5;
        let before = decoded.clone();
        let mut options = options();
        options.loudness_lufs = Some(-23.0);

        let error = process_audio(&mut decoded, options).unwrap_err();

        assert!(
            error.contains("undefined") || error.contains("too short"),
            "unexpected error: {error}"
        );
        assert_eq!(decoded.sample_rate, before.sample_rate);
        assert_eq!(decoded.channels, before.channels);
        assert_eq!(decoded.bits_per_sample, before.bits_per_sample);
        assert_eq!(decoded.sample_format, before.sample_format);
        assert_eq!(decoded.channel_mask, before.channel_mask);
    }

    #[test]
    fn invalid_quality_precedes_missing_model_configuration() {
        #[cfg(feature = "onnx")]
        {
            let mut options = options();
            options.backend = BackendChoice::Explicit(Backend::Onnx);
            options.quality = Some("unknown".into());
            let error = options.validate_config(&audio(48_000)).unwrap_err();
            assert!(matches!(
                error,
                ConfigError::InvalidValue {
                    field: "quality",
                    ..
                }
            ));
        }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn generic_onnx_requires_external_weights() {
        assert!(requires_external_model(Backend::Onnx));
    }
}
