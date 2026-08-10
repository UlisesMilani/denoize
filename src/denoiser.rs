//! The de-noizer: ties together STFT, IMCRA noise estimation, the
//! decision-directed a-priori SNR estimator, the selected spectral gain
//! estimator, attack/release + cepstral gain smoothing, transient protection,
//! and optional pre-emphasis.
//!
//! 現在の実装でサポートしているノイズ除去技術（完全版）:
//!
//! 1. 基本フレームワーク
//!    - STFT + ISTFT（自前 radix-2 FFT）
//!    - Perfect Reconstruction OLA（窓エネルギー累積正規化）
//!    - 高オーバーラップ（0.5〜0.95）対応
//!    - 窓関数: Hann / Hamming / Sine / Blackman / Kaiser / Flat-top / DPSS
//!
//! 2. ノイズ推定
//!    - IMCRA/MCRA スタイル（minima-controlled recursive averaging）
//!    - 指数忘却型2トラッカー最小値追跡
//!    - Speech Presence Probability (SPP) 推定
//!    - Spectral Flatness による自動ノイズプロファイル検出
//!    - Profile Anchoring + 上昇率制限
//!
//! 3. SNR推定
//!    - Ephraim-Malah Decision-Directed a-priori SNR
//!
//! 4. スペクトルゲイン推定器（5種類）
//!    - OMLSA (Cohen 2001, デフォルト)
//!    - LogMMSE (Ephraim-Malah 1985)
//!    - MMSE-STSA (Ephraim-Malah 1984)
//!    - Wiener
//!    - Spectral Subtraction (+ nonlinear / geometric / multiband variants)
//!
//! 5. 後処理・平滑化
//!    - Attack/Release ゲイン平滑化
//!    - Gain Floor
//!    - DC Blocking
//!    - Makeup Gain
//!
//! 6. 高音質化拡張（本プロジェクトの目玉）
//!    - Transient Protection（オンセット保護）
//!    - Cepstral Smoothing（ミュージカルノイズ抑制）
//!    - Perceptual Bark weighting + Musical-noise post-filter
//!    - Pre-emphasis / De-emphasis（オプション）
//!
//! 全体パイプライン（per channel）:
//!   1. Optional DC-blocking high-pass filter
//!   2. Optional noise profile seed（先頭無音 or 指定）
//!   3. STFT analysis（任意の窓 + 高オーバーラップ対応）
//!   4. IMCRA ノイズPSD + SPP 更新
//!   5. Decision-directed a-priori SNR 推定
//!   6. 選択したゲイン推定器で `g[k]` を計算
//!   7. Transient Protection（フラックスベース）
//!   8. Attack/Release 平滑化
//!   9. Cepstral Smoothing（本格的低ケフレンシ除去）
//!  10. ゲイン適用（位相保持）
//!  11. ISTFT + 完全再構成 OLA 正規化
//!  12. Optional de-emphasis + makeup gain

use std::collections::VecDeque;

use crate::audio::sanitize_sample;
use crate::config::{
    checked_stream_memory_bytes, ConfigError, ResourcePlan, MAX_DENOISER_FRAME_SIZE,
    MAX_KAISER_BETA, MAX_MAKEUP_GAIN_DB, MAX_PROFILE_MS, MAX_SAMPLE_RATE, MIN_DENOISER_FRAME_SIZE,
    MIN_MAKEUP_GAIN_DB,
};
use crate::fft::Complex;
use crate::gain::{compute_gain, multiband_specsub_gains, Algorithm, GainParams, SpecSubLaw};
use crate::noise::{NoiseConfig, NoiseEstimator};
use crate::perceptual::{apply_perceptual_weights, bin_to_bark_band, N_BARK_BANDS};
use crate::postfilter::{MusicalNoisePostFilter, PostFilterConfig};
use crate::stft::{Stft, StftConfig};
use crate::window::{validate_dpss_bandwidth, WindowParams, WindowType, MAX_DENOISER_DPSS_NW};

/// Top-level configuration.
///
/// Aimed at the highest possible sound quality: artifact-free, transparent
/// denoising that preserves transients, timbre, stereo image, and "air".
/// All parameters default toward fidelity; increase strength only when needed.
#[derive(Clone, Debug)]
pub struct DenoiserConfig {
    /// Gain-estimation algorithm.
    pub algorithm: Algorithm,
    /// Denoising strength in `[0, 1]` (higher = more aggressive). Start low
    /// (0.2-0.5) for music/mastering to preserve fidelity.
    pub strength: f64,
    /// FFT frame size (power of two). Larger = better freq resolution / less
    /// musical noise, but more time smearing. 2048-8192 recommended for hi-fi.
    pub frame_size: usize,
    /// Overlap ratio in `[0.5, 0.95]`. Higher overlap (0.75-0.875) dramatically
    /// reduces artifacts and pre-echo at modest CPU cost.
    pub overlap: f64,
    /// Analysis/synthesis window.
    pub window: WindowType,
    /// Noise profile. `>0`: learn from first N ms. `0`: auto-detect leading
    /// silence. `<0`: none (rely on blind IMCRA bootstrap).
    pub profile_ms: f64,
    /// Allow the noise PSD to adapt over time.
    pub adapt: bool,
    /// Continuously learn a profile from confidently noise-only regions.
    pub adaptive_noise: bool,
    /// Segment processing around detected speech and strongly attenuate silence.
    pub vad: bool,
    /// Attenuation gain in `[0, 1]` applied to non-speech regions when VAD is enabled.
    /// Default is 0.08 (~ -22 dB).
    pub vad_silence_gain: f64,
    /// Blend factor in `[0, 1]` for speech regions: `processed * mix + original * (1 - mix)`.
    /// Default is 0.85 (85% denoised speech, 15% natural original speech blend).
    pub vad_speech_mix: f64,
    /// Gain release-smoothing coefficient in `[0, 1]` (higher = slower).
    /// Higher values help kill musical noise for transparent results.
    pub smoothing: f64,
    /// Apply a DC-blocking high-pass filter before processing.
    pub dc_block: bool,
    /// Makeup gain in dB applied to the output (`-120..=120`).
    pub makeup_gain_db: f64,
    /// Sample rate of the signal to be processed.
    pub sample_rate: u32,

    // === High-fidelity extensions (for world's best sound quality) ===
    /// Protect transients/onsets: reduce suppression during detected attacks
    /// to preserve punch, clarity, and natural dynamics (music, percussion, speech plosives).
    pub transient_protect: bool,
    /// Apply light cepstral smoothing to the per-frame gain curve.
    /// Strongly suppresses musical noise / "birdies" while preserving overall timbre.
    pub cepstral_smoothing: bool,
    /// Apply first-order pre-emphasis before analysis and matching de-emphasis
    /// after synthesis. Helps control high-frequency noise without dulling the signal.
    pub pre_emphasis: bool,
    /// Coefficient for pre-emphasis (0.0 = disabled effect, typical 0.9-0.97).
    pub pre_emphasis_alpha: f64,

    // === Advanced DSP (roadmap items 3–5) ===
    /// Kaiser β / DPSS NW parameters for advanced windows.
    pub window_params: WindowParams,
    /// Use multiband spectral subtraction (per-Bark-band noise estimate).
    pub multiband: bool,
    /// Apply Bark-scale perceptual gain weighting after estimation.
    pub perceptual_weighting: bool,
    /// Enable musical-noise suppression post-filter.
    pub musical_noise_postfilter: bool,
}

/// Named presets for common material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Speech,
    Music,
    Aggressive,
    Gentle,
    Restore,
    /// Highest-fidelity preset: minimal artifacts, maximum transparency and
    /// preservation of musicality/transients.
    /// Uses proper spectral-flux Transient Protection + FFT-based Cepstral liftering.
    HiFi,
}

/// High-level intent that coordinates denoising features for the material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessingMode {
    Speech,
    Music,
    Ambient,
}

impl ProcessingMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "speech" | "voice" => Some(Self::Speech),
            "music" => Some(Self::Music),
            "ambient" | "environment" => Some(Self::Ambient),
            _ => None,
        }
    }

    pub fn apply(self, config: &mut DenoiserConfig) {
        match self {
            Self::Speech => {
                config.strength = config.strength.max(0.7);
                config.vad = true;
                config.adaptive_noise = true;
                config.transient_protect = true;
                config.cepstral_smoothing = true;
            }
            Self::Music => {
                config.strength = config.strength.min(0.35);
                config.vad = false;
                config.adaptive_noise = false;
                config.transient_protect = true;
                config.perceptual_weighting = true;
                config.musical_noise_postfilter = true;
                config.smoothing = config.smoothing.max(0.75);
            }
            Self::Ambient => {
                config.strength = config.strength.min(0.4);
                config.vad = false;
                config.adaptive_noise = true;
                config.transient_protect = true;
                config.perceptual_weighting = true;
                config.smoothing = config.smoothing.max(0.7);
            }
        }
    }
}

impl Preset {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "speech" | "voice" => Preset::Speech,
            "music" => Preset::Music,
            "aggressive" => Preset::Aggressive,
            "gentle" => Preset::Gentle,
            "restore" => Preset::Restore,
            "hifi" | "mastering" | "hi-fi" | "highfidelity" => Preset::HiFi,
            _ => return None,
        })
    }

    /// Build a [`DenoiserConfig`] from this preset at the given sample rate.
    ///
    /// HiFi preset is tuned for maximum transparency and fidelity: gentlest
    /// suppression, large frames, high overlap, full transient + cepstral
    /// protections, and pre-emphasis. Use for music, mastering, or when
    /// "world's best sound quality" is the goal over maximum noise removal.
    pub fn config(self, sample_rate: u32) -> DenoiserConfig {
        let mut c = DenoiserConfig::default(sample_rate);
        match self {
            Preset::Speech => {
                c.algorithm = Algorithm::Omlsa;
                c.strength = 0.6;
                c.frame_size = 2048;
                c.smoothing = 0.6;
            }
            Preset::Music => {
                c.algorithm = Algorithm::Omlsa;
                c.strength = 0.4;
                c.frame_size = 4096;
                c.smoothing = 0.5;
                c.overlap = 0.8;
                c.transient_protect = true;
                c.cepstral_smoothing = true;
                c.perceptual_weighting = true;
                c.musical_noise_postfilter = true;
            }
            Preset::Aggressive => {
                c.algorithm = Algorithm::Omlsa;
                c.strength = 0.85;
                c.frame_size = 2048;
                c.smoothing = 0.72;
            }
            Preset::Gentle => {
                c.algorithm = Algorithm::LogMmse;
                c.strength = 0.3;
                c.frame_size = 2048;
                c.smoothing = 0.45;
            }
            Preset::Restore => {
                c.algorithm = Algorithm::LogMmse;
                c.strength = 0.2;
                c.frame_size = 2048;
                c.smoothing = 0.4;
            }
            Preset::HiFi => {
                // The "world's best sound quality" preset: prioritize transparency,
                // natural timbre, transient fidelity, minimal artifacts.
                // OMLSA + low strength + protections gives excellent balance.
                c.algorithm = Algorithm::Omlsa;
                c.strength = 0.28;
                c.frame_size = 4096;
                c.overlap = 0.875;
                c.window = WindowType::Kaiser;
                c.window_params.kaiser_beta = 10.0;
                c.smoothing = 0.65;
                c.transient_protect = true;
                c.cepstral_smoothing = true;
                c.perceptual_weighting = true;
                c.musical_noise_postfilter = true;
                // Pre-emphasis is powerful for HF noise but can color clean signals
                // when combined with spectral processing. Enable explicitly with --pre-emphasis.
                c.pre_emphasis = false;
                c.pre_emphasis_alpha = 0.72;
            }
        }
        c
    }
}

impl DenoiserConfig {
    pub fn default(sample_rate: u32) -> Self {
        DenoiserConfig {
            algorithm: Algorithm::Omlsa,
            strength: 0.6,
            frame_size: 2048,
            overlap: 0.75,
            window: WindowType::Hann,
            profile_ms: 0.0,
            adapt: true,
            adaptive_noise: false,
            vad: false,
            vad_silence_gain: 0.08,
            vad_speech_mix: 0.85,
            smoothing: 0.6,
            dc_block: true,
            makeup_gain_db: 0.0,
            sample_rate,
            // Hi-fi defaults (enable features that push toward best possible quality)
            transient_protect: true,
            cepstral_smoothing: false, // opt-in for max quality; adds a bit of CPU
            pre_emphasis: false,
            pre_emphasis_alpha: 0.92,
            window_params: WindowParams::default(),
            multiband: false,
            perceptual_weighting: false,
            musical_noise_postfilter: false,
        }
    }

    /// Strictly validate configuration received from an external caller.
    ///
    /// Window-specific parameters are checked only when their window is
    /// selected, so dormant values retain the behavior of the established
    /// infallible API.
    pub fn validate_config(&self) -> Result<(), ConfigError> {
        validate_finite_range(
            "strength",
            self.strength,
            0.0,
            1.0,
            "a finite value in 0..=1",
        )?;
        validate_finite_range(
            "overlap",
            self.overlap,
            0.5,
            0.95,
            "a finite value in 0.5..=0.95",
        )?;
        validate_finite_range(
            "vad_silence_gain",
            self.vad_silence_gain,
            0.0,
            1.0,
            "a finite value in 0..=1",
        )?;
        validate_finite_range(
            "vad_speech_mix",
            self.vad_speech_mix,
            0.0,
            1.0,
            "a finite value in 0..=1",
        )?;
        validate_finite_range(
            "smoothing",
            self.smoothing,
            0.0,
            1.0,
            "a finite value in 0..=1",
        )?;
        if !self.profile_ms.is_finite() || self.profile_ms > MAX_PROFILE_MS {
            return Err(ConfigError::invalid(
                "profile_ms",
                "a finite non-positive mode or at most 60000 ms",
            ));
        }
        validate_finite_range(
            "makeup_gain_db",
            self.makeup_gain_db,
            MIN_MAKEUP_GAIN_DB,
            MAX_MAKEUP_GAIN_DB,
            "a finite value in -120..=120 dB",
        )?;
        if !self.frame_size.is_power_of_two()
            || !(MIN_DENOISER_FRAME_SIZE..=MAX_DENOISER_FRAME_SIZE).contains(&self.frame_size)
        {
            return Err(ConfigError::invalid(
                "frame_size",
                "a power of two in 256..=65536",
            ));
        }
        if self.sample_rate == 0 || self.sample_rate > MAX_SAMPLE_RATE {
            return Err(ConfigError::invalid(
                "sample_rate",
                "an integer in 1..=768000 Hz",
            ));
        }
        validate_finite_range(
            "pre_emphasis_alpha",
            self.pre_emphasis_alpha,
            0.0,
            0.99,
            "a finite value in 0..=0.99",
        )?;

        match self.window {
            WindowType::Kaiser => validate_finite_range(
                "window_params.kaiser_beta",
                self.window_params.kaiser_beta,
                0.0,
                MAX_KAISER_BETA,
                "a finite value in 0..=50",
            )?,
            WindowType::Dpss => {
                let bandwidth = self.window_params.dpss_bandwidth;
                if !bandwidth.is_finite() || bandwidth <= 0.0 || bandwidth > MAX_DENOISER_DPSS_NW {
                    return Err(ConfigError::invalid(
                        "window_params.dpss_bandwidth",
                        "DPSS bandwidth (NW), finite and greater than 0 and at most 8",
                    ));
                }
                validate_dpss_bandwidth(self.frame_size, bandwidth)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Compatibility wrapper returning the established string error type.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_config().map_err(|error| error.to_string())
    }

    /// Clamp user-supplied values into safe ranges.
    pub fn sanitized(mut self) -> Self {
        let defaults = DenoiserConfig::default(48_000);
        self.strength = finite_clamp(self.strength, defaults.strength, 0.0, 1.0);
        // Preserve the established effective ceiling of the infallible API.
        self.smoothing = finite_clamp(self.smoothing, defaults.smoothing, 0.0, 0.95);
        self.overlap = finite_clamp(self.overlap, defaults.overlap, 0.5, 0.95);
        self.vad_silence_gain =
            finite_clamp(self.vad_silence_gain, defaults.vad_silence_gain, 0.0, 1.0);
        self.vad_speech_mix = finite_clamp(self.vad_speech_mix, defaults.vad_speech_mix, 0.0, 1.0);
        if !self.profile_ms.is_finite() {
            self.profile_ms = defaults.profile_ms;
        } else if self.profile_ms > MAX_PROFILE_MS {
            self.profile_ms = MAX_PROFILE_MS;
        }
        self.makeup_gain_db = finite_clamp(
            self.makeup_gain_db,
            defaults.makeup_gain_db,
            MIN_MAKEUP_GAIN_DB,
            MAX_MAKEUP_GAIN_DB,
        );
        if !self.frame_size.is_power_of_two()
            || !(MIN_DENOISER_FRAME_SIZE..=MAX_DENOISER_FRAME_SIZE).contains(&self.frame_size)
        {
            self.frame_size = 2048;
        }
        if self.sample_rate == 0 || self.sample_rate > MAX_SAMPLE_RATE {
            self.sample_rate = 48_000;
        }
        if self.window == WindowType::Kaiser
            && (!self.window_params.kaiser_beta.is_finite()
                || !(0.0..=MAX_KAISER_BETA).contains(&self.window_params.kaiser_beta))
        {
            self.window_params.kaiser_beta = WindowParams::default().kaiser_beta;
        }
        if self.window == WindowType::Dpss
            && (!self.window_params.dpss_bandwidth.is_finite()
                || self.window_params.dpss_bandwidth > MAX_DENOISER_DPSS_NW
                || validate_dpss_bandwidth(self.frame_size, self.window_params.dpss_bandwidth)
                    .is_err())
        {
            self.window_params.dpss_bandwidth = WindowParams::default().dpss_bandwidth;
        }
        self.pre_emphasis_alpha = finite_clamp(
            self.pre_emphasis_alpha,
            defaults.pre_emphasis_alpha,
            0.0,
            0.99,
        );
        // Always enable quality features by default for best results unless explicitly off
        self
    }
}

fn validate_finite_range(
    field: &'static str,
    value: f64,
    minimum: f64,
    maximum: f64,
    expected: &'static str,
) -> Result<(), ConfigError> {
    if value.is_finite() && (minimum..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(ConfigError::invalid(field, expected))
    }
}

fn finite_clamp(value: f64, fallback: f64, minimum: f64, maximum: f64) -> f64 {
    if value.is_finite() {
        value.clamp(minimum, maximum)
    } else {
        fallback
    }
}

pub struct Denoiser {
    config: DenoiserConfig,
    stft: Stft,
    noise: NoiseEstimator,
    noise_cfg: NoiseConfig,
    gain_params: GainParams,
    sample_rate: u32,
    frame_size: usize,
    hop: usize,
    m: usize, // number of unique bins
    alpha_dd: f64,
    xi_min: f64,
    makeup: f64,

    // --- per-channel recursion / smoothing state (length `m`) ---
    prev_g: Vec<f64>,
    prev_y2: Vec<f64>,
    prev_lambda_d: Vec<f64>,
    prev_gsmooth: Vec<f64>,

    // --- reusable scratch (length `frame_size` / `m`) ---
    spec: Vec<Complex>,
    frame: Vec<f64>,
    y2: Vec<f64>,
    g: Vec<f64>,
    lambda_d_buf: Vec<f64>,
    spp_buf: Vec<f64>,
    y2_snapshot_buf: Vec<f64>,

    // High-fidelity state
    prev_frame_energy: f64,
    prev_mag: Vec<f64>, // previous frame magnitude for spectral flux
    pre_emph_prev: f64, // for pre-emphasis filter state
    de_emph_prev: f64,  // for de-emphasis filter state
    dc_prev_x: f64,
    dc_prev_y: f64,
    cepstral_fft: crate::fft::Fft,
    cepstral_spec: Vec<Complex>,
    cepstral_orig: Vec<f64>,

    // Advanced DSP state
    bark_bands: Vec<usize>,
    postfilter: MusicalNoisePostFilter,
}

impl Denoiser {
    /// Construct a denoiser after repairing invalid values for compatibility.
    ///
    /// External callers that need invalid input reported rather than repaired
    /// should use [`Denoiser::try_new`].
    pub fn new(config: DenoiserConfig) -> Self {
        let config = config.sanitized();
        Self::build(config).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Strictly validate and construct a denoiser from an external config.
    pub fn try_new(config: DenoiserConfig) -> Result<Self, ConfigError> {
        config.validate_config()?;
        Self::build(config.sanitized())
    }

    fn build(config: DenoiserConfig) -> Result<Self, ConfigError> {
        let strength = config.strength;
        let musical_pf = config.musical_noise_postfilter;
        let sample_rate = config.sample_rate;
        let frame_size = config.frame_size;
        let hop = (frame_size as f64 * (1.0 - config.overlap)).round() as usize;
        let hop = hop.max(1);
        let stft = Stft::try_new(StftConfig {
            frame_size,
            hop,
            window: config.window,
            window_params: config.window_params,
        })?;
        let m = stft.nbins();

        // Strength -> estimator floors / oversubtraction.
        let xi_min = 10f64.powf(-25.0 / 10.0); // -25 dB a-priori floor
        let g_min_db = -20.0 - 25.0 * config.strength;
        let g_min = 10f64.powf(g_min_db / 20.0);
        let alpha_os = 1.0 + 2.0 * config.strength; // 1..3
        let beta_floor = 0.02;
        let gain_params = GainParams {
            xi_min,
            g_min,
            alpha_os,
            beta_floor,
        };

        let noise_cfg = NoiseConfig {
            adaptive_profile: config.adaptive_noise,
            ..NoiseConfig::default()
        };
        let noise = NoiseEstimator::new(noise_cfg, m, sample_rate, hop);
        let makeup = 10f64.powf(config.makeup_gain_db / 20.0);

        let cepstral_fft_size = (2 * m).next_power_of_two().max(32);
        let cepstral_fft = crate::fft::Fft::new(cepstral_fft_size);

        Ok(Denoiser {
            config,
            stft,
            noise,
            noise_cfg,
            gain_params,
            sample_rate,
            frame_size,
            hop,
            m,
            alpha_dd: 0.98,
            xi_min,
            makeup,
            prev_g: vec![0.0; m],
            prev_y2: vec![0.0; m],
            prev_lambda_d: vec![1e-12; m],
            prev_gsmooth: vec![1.0; m],
            spec: vec![Complex::default(); frame_size],
            frame: vec![0.0; frame_size],
            y2: vec![0.0; m],
            g: vec![0.0; m],
            lambda_d_buf: vec![0.0; m],
            spp_buf: vec![0.0; m],
            y2_snapshot_buf: vec![0.0; m],
            prev_frame_energy: 0.0,
            prev_mag: vec![0.0; m],
            pre_emph_prev: 0.0,
            de_emph_prev: 0.0,
            dc_prev_x: 0.0,
            dc_prev_y: 0.0,
            cepstral_fft,
            cepstral_spec: vec![Complex::default(); cepstral_fft_size],
            cepstral_orig: vec![0.0; m],
            bark_bands: bin_to_bark_band(m, sample_rate),
            postfilter: MusicalNoisePostFilter::new(
                m,
                PostFilterConfig {
                    enabled: musical_pf,
                    strength,
                    ..PostFilterConfig::default()
                },
            ),
        })
    }

    pub fn config(&self) -> &DenoiserConfig {
        &self.config
    }

    /// Reset per-channel recursion / smoothing state and rebuild the noise
    /// estimator so each channel is processed independently.
    fn reset_for_channel(&mut self) {
        self.noise = NoiseEstimator::new(self.noise_cfg, self.m, self.sample_rate, self.hop);
        self.noise.adapt = self.config.adapt;
        for v in &mut self.prev_g {
            *v = 0.0;
        }
        for v in &mut self.prev_y2 {
            *v = 0.0;
        }
        for v in &mut self.prev_lambda_d {
            *v = 1e-12;
        }
        for v in &mut self.prev_gsmooth {
            *v = 1.0;
        }
        self.prev_frame_energy = 0.0;
        self.prev_mag.fill(0.0);
        self.pre_emph_prev = 0.0;
        self.de_emph_prev = 0.0;
        self.dc_prev_x = 0.0;
        self.dc_prev_y = 0.0;
        self.postfilter.reset();
    }

    /// One-pole DC-blocking high-pass filter: `y = x - x[n-1] + R*y[n-1]`.
    fn dc_block(input: &[f64]) -> Vec<f64> {
        let r = 0.999;
        let mut out = Vec::with_capacity(input.len());
        let mut prev_x = 0.0;
        let mut prev_y = 0.0;
        for &x in input {
            let y = x - prev_x + r * prev_y;
            out.push(y);
            prev_x = x;
            prev_y = y;
        }
        out
    }

    /// Process one sample through the stateful DC blocker used by streaming
    /// input. The batch path keeps the equivalent vectorized implementation
    /// above for backwards-compatible output.
    fn dc_block_sample(&mut self, x: f64) -> f64 {
        let r = 0.999;
        let y = x - self.dc_prev_x + r * self.dc_prev_y;
        self.dc_prev_x = x;
        self.dc_prev_y = y;
        y
    }

    /// First-order pre-emphasis: y[n] = x[n] - alpha * x[n-1]
    fn pre_emphasize(&mut self, input: &[f64]) -> Vec<f64> {
        let alpha = self.config.pre_emphasis_alpha;
        let mut out = Vec::with_capacity(input.len());
        let mut prev = self.pre_emph_prev;
        for &x in input {
            let y = x - alpha * prev;
            out.push(y);
            prev = x;
        }
        self.pre_emph_prev = prev;
        out
    }

    fn pre_emphasize_sample(&mut self, x: f64) -> f64 {
        let y = x - self.config.pre_emphasis_alpha * self.pre_emph_prev;
        self.pre_emph_prev = x;
        y
    }

    /// Matching de-emphasis (inverse): x[n] = y[n] + alpha * x[n-1]
    fn de_emphasize(&mut self, input: &[f64]) -> Vec<f64> {
        let alpha = self.config.pre_emphasis_alpha;
        let mut out = Vec::with_capacity(input.len());
        let mut prev = self.de_emph_prev;
        for &y in input {
            let x = y + alpha * prev;
            out.push(x);
            prev = x;
        }
        self.de_emph_prev = prev;
        out
    }

    fn de_emphasize_sample(&mut self, y: f64) -> f64 {
        let x = y + self.config.pre_emphasis_alpha * self.de_emph_prev;
        self.de_emph_prev = x;
        x
    }

    /// Compute transient / onset score using proper **spectral flux**.
    /// Spectral flux = sum_k | |Y[k]| - |Y_prev[k]| |
    /// Combined with total energy delta for robustness.
    /// Returns value in [0, 1] (higher = stronger transient).
    fn compute_transient_score(&mut self) -> f64 {
        let m = self.m;
        let mut flux = 0.0;
        let mut energy = 0.0;

        for (k, &y2_k) in self.y2_snapshot_buf.iter().enumerate().take(m) {
            let mag = y2_k.sqrt();
            energy += y2_k;
            let prev_mag = self.prev_mag[k];
            flux += (mag - prev_mag).abs();
            self.prev_mag[k] = mag * 0.7 + prev_mag * 0.3; // light temporal smoothing on mag
        }

        // Update smoothed energy
        let delta_e = (energy - self.prev_frame_energy).max(0.0);
        self.prev_frame_energy = energy * 0.6 + self.prev_frame_energy * 0.4;

        // Normalize flux
        let norm_flux = if energy > 1e-12 {
            (flux / (energy.sqrt() + 1e-9)).clamp(0.0, 8.0) / 8.0
        } else {
            0.0
        };

        // Combine flux and energy rise
        let energy_rise = if energy > 1e-12 {
            (delta_e / (energy + 1e-9)).clamp(0.0, 3.0) / 3.0
        } else {
            0.0
        };

        // Weighted combination. Flux is more reliable for musical transients.
        (0.75 * norm_flux + 0.25 * energy_rise).clamp(0.0, 1.0)
    }

    /// Proper **cepstral smoothing** (liftering) of the gain vector.
    ///
    /// Full implementation:
    ///   log(G) → FFT(cepstrum) → zero high quefrency (lifter) → IFFT → exp
    ///
    /// Then a conservative blend back to the original gains.
    /// This prevents over-smoothing on clean signals while strongly
    /// suppressing musical noise when it appears.
    fn cepstral_smooth_gains(&mut self) {
        let m = self.m;
        if m < 8 {
            return;
        }

        // Compute variation to decide how strongly to apply smoothing.
        // On clean signals gains are nearly flat → almost no smoothing.
        let mut min_g = 1.0f64;
        let mut max_g = 0.0f64;
        let mut sum = 0.0;
        for &v in self.g.iter() {
            min_g = min_g.min(v);
            max_g = max_g.max(v);
            sum += v;
        }
        let mean = sum / m as f64;
        let variation = (max_g - min_g) / mean.max(1e-6);

        if variation < 0.04 {
            // Almost no variation → this is clean or very high SNR.
            // Do almost nothing to preserve amplitude perfectly.
            return;
        }

        // Save original
        self.cepstral_orig.copy_from_slice(&self.g);

        let fft_size = self.cepstral_fft.size();
        let keep = 6.min(fft_size / 10);

        self.cepstral_spec.fill(Complex::default());
        for (i, &gi) in self.g.iter().enumerate().take(m) {
            self.cepstral_spec[i] = Complex::new(gi.max(1e-8).ln(), 0.0);
        }

        self.cepstral_fft.forward(&mut self.cepstral_spec);

        for slot in self
            .cepstral_spec
            .iter_mut()
            .take(fft_size - keep)
            .skip(keep)
        {
            *slot = Complex::default();
        }

        self.cepstral_fft.inverse(&mut self.cepstral_spec);

        // Dynamic blend: more smoothing when there is more variation (more noise)
        let blend = (0.35 + 0.45 * variation.min(1.0)).min(0.75);

        for (i, gi) in self.g.iter_mut().enumerate().take(m) {
            let liftered = self.cepstral_spec[i].re.exp().clamp(1e-6, 1.0);
            *gi = (blend * liftered + (1.0 - blend) * self.cepstral_orig[i]).clamp(1e-6, 1.0);
        }
    }

    /// Auto-detect the number of leading "noise-only" frames for profiling.
    ///
    /// Uses *spectral flatness* (Wiener entropy): white/background noise has a
    /// flat spectrum (flatness ≈ 1) while any tonal or voiced signal is spectrally
    /// peaky (flatness < 1). This works at *any* broadband SNR, unlike a pure
    /// energy threshold which fails when the signal is only a few dB above the
    /// noise. Returns the count of leading flat frames, or 0 if there is no
    /// clear noise-only segment followed by signal.
    fn detect_profile_frames(&mut self, input: &[f64]) -> usize {
        let n = self.frame_size;
        let m = self.m;
        let hop = self.hop;
        let frames_15s = (1.5 * self.sample_rate as f64 / hop as f64) as usize;
        let max_check = frames_15s.max(8);

        let mut observed = 0usize;
        let mut fmax = 0.0f64;
        let mut fmin = 1.0f64;
        let mut start = 0;
        while start + n <= input.len() && observed < max_check {
            let flatness = self.profile_frame_flatness(input, start, n, m);
            fmax = fmax.max(flatness);
            fmin = fmin.min(flatness);
            observed += 1;
            start += hop;
        }
        if observed == 0 {
            return 0;
        }

        // Spectral flatness of white noise is well below 1 in practice (~0.5,
        // because |FFT bin|^2 is exponentially distributed), so an absolute
        // threshold near 1 does not work. Instead, threshold adaptively relative
        // to the observed flatness range: the leading noise-only frames have the
        // highest flatness, and the signal onset shows up as a drop.
        // Need a meaningful flatness contrast to trust a profile.
        if fmax - fmin < 0.08 {
            return 0;
        }
        // 60% of the way from the minimum to the maximum flatness.
        let flat_thr = fmin + 0.6 * (fmax - fmin);
        let mut run = 0;
        start = 0;
        while run < observed {
            if self.profile_frame_flatness(input, start, n, m) >= flat_thr {
                run += 1;
            } else {
                break;
            }
            start += hop;
        }
        let min_frames = ((0.08 * self.sample_rate as f64 / hop as f64).round() as usize).max(1);
        // Trust the profile only if there is a signal onset after it.
        if run >= min_frames && run < observed {
            run
        } else {
            0
        }
    }

    fn profile_frame_flatness(&mut self, input: &[f64], start: usize, n: usize, m: usize) -> f64 {
        self.frame[..n].copy_from_slice(&input[start..start + n]);
        self.stft.analyze(&self.frame, &mut self.spec);
        let mut sum_p = 0.0;
        let mut sum_logp = 0.0;
        let mut nonzero = 0usize;
        for &bin in self.spec.iter().take(m) {
            let power = bin.re * bin.re + bin.im * bin.im;
            if power > 1e-20 {
                sum_p += power;
                sum_logp += power.ln();
                nonzero += 1;
            }
        }
        if nonzero == 0 {
            return 0.0;
        }
        let geometric_mean = (sum_logp / nonzero as f64).exp();
        let arithmetic_mean = sum_p / nonzero as f64;
        (geometric_mean / arithmetic_mean.max(1e-300)).clamp(0.0, 1.0)
    }

    /// Analyze the first `n_frames` frames and return their per-bin power.
    fn collect_profile_y2(&mut self, input: &[f64], n_frames: usize) -> Vec<Vec<f64>> {
        let n = self.frame_size;
        let m = self.m;
        let available_frames = if input.len() < n {
            0
        } else {
            1 + (input.len() - n) / self.hop
        };
        let frame_count = n_frames.min(available_frames);
        if frame_count == 0 {
            return Vec::new();
        }

        // Accumulate one mean spectrum instead of retaining every requested
        // frame. This bounds profile memory by O(nbins), and the chronological
        // sum for each bin is identical to NoiseEstimator's frame averaging.
        let mut mean = vec![0.0; m];
        let mut start = 0;
        for _ in 0..frame_count {
            self.frame[..n].copy_from_slice(&input[start..start + n]);
            self.stft.analyze(&self.frame, &mut self.spec);
            for (k, total) in mean.iter_mut().enumerate() {
                let c = self.spec[k];
                *total += c.re * c.re + c.im * c.im;
            }
            start += self.hop;
        }
        for value in &mut mean {
            *value /= frame_count as f64;
        }
        vec![mean]
    }

    fn explicit_profile_frames(&self) -> usize {
        if self.config.profile_ms <= 0.0 {
            return 0;
        }
        // Keep the established single-rounding formula for batch/profile DSP.
        // `checked_profile_target_samples` is intentionally used only for the
        // retained streaming prefix, whose unit is whole samples.
        ((self.config.profile_ms / 1000.0 * self.sample_rate as f64 / self.hop as f64).round()
            as usize)
            .max(1)
    }

    /// Apply a real per-bin gain `g` (length `m`) to the full spectrum,
    /// preserving Hermitian symmetry so the ISTFT stays real.
    fn apply_gain(&mut self) {
        let n = self.frame_size;
        let m = self.m;
        // DC bin.
        self.spec[0] = self.spec[0].mul_real(self.g[0]);
        // Bins 1 .. n/2-1, mirrored to n-k.
        for k in 1..m - 1 {
            let gk = self.g[k];
            self.spec[k] = self.spec[k].mul_real(gk);
            let mir = n - k;
            self.spec[mir] = self.spec[mir].mul_real(gk);
        }
        // Nyquist bin.
        self.spec[n / 2] = self.spec[n / 2].mul_real(self.g[m - 1]);
    }

    /// Process a single frame at sample offset `start` (zero-padded at the
    /// tail if needed) and overlap-add its synthesis into `out`/`norm`.
    fn process_frame(
        &mut self,
        input: &[f64],
        start: usize,
        frame_idx: usize,
        out: &mut [f64],
        norm: &mut [f64],
    ) {
        let n = self.frame_size;
        let m = self.m;
        for i in 0..n {
            self.frame[i] = if start + i < input.len() {
                input[start + i]
            } else {
                0.0
            };
        }
        self.stft.analyze(&self.frame, &mut self.spec);

        for k in 0..m {
            let c = self.spec[k];
            self.y2[k] = c.re * c.re + c.im * c.im;
        }
        self.noise.update(&self.y2);

        // Strong fidelity bypass for very clean frames
        let frame_energy: f64 = self.y2.iter().sum();
        let noise_energy: f64 = self.noise.noise_psd().iter().sum();
        if frame_energy > noise_energy * 50.0 {
            // Almost certainly no noise — pass the frame through untouched
            for k in 0..m {
                self.g[k] = 1.0;
            }
            self.apply_gain();
            self.stft.synthesize(&mut self.spec, out, norm, start);
            // still update some state lightly
            for k in 0..m {
                self.prev_g[k] = 1.0;
                self.prev_y2[k] = self.y2[k];
                self.prev_lambda_d[k] = self.noise.noise_psd()[k];
                self.prev_gsmooth[k] = 1.0;
            }
            return;
        }

        // Copy out the noise estimate / SPP so we don't hold a borrow of
        // `self.noise` while mutating the per-bin recursion state.
        self.lambda_d_buf.copy_from_slice(self.noise.noise_psd());
        self.spp_buf.copy_from_slice(self.noise.speech_presence());

        let g_min = self.gain_params.g_min;
        let alpha_dd = self.alpha_dd;
        let xi_min = self.xi_min;
        let algo = self.config.algorithm;
        let gp = self.gain_params;
        let smoothing = self.config.smoothing;

        // Transient score for this frame (protects onsets for fidelity)
        // Uses proper spectral flux (not just total energy)
        let tscore = if self.config.transient_protect {
            self.y2_snapshot_buf.copy_from_slice(&self.y2);
            self.compute_transient_score()
        } else {
            0.0
        };

        // Per-bin gamma / xi for this frame.
        let mut gamma_frame = vec![0.0f64; m];
        let mut xi_frame = vec![0.0f64; m];
        for (k, &y2_k) in self.y2.iter().enumerate().take(m) {
            let lam = self.lambda_d_buf[k].max(1e-12);
            let gamma = y2_k / lam;
            let xi_hat = if frame_idx == 0 {
                (gamma - 1.0).max(xi_min)
            } else {
                let prev_sig = self.prev_g[k] * self.prev_g[k] * self.prev_y2[k]
                    / self.prev_lambda_d[k].max(1e-12);
                alpha_dd * prev_sig + (1.0 - alpha_dd) * (gamma - 1.0).max(xi_min)
            };
            gamma_frame[k] = gamma;
            xi_frame[k] = xi_hat.max(xi_min);
        }

        // Multiband spectral subtraction path (SpecSub family only).
        let use_mb_specsub = self.config.multiband
            && matches!(
                algo,
                Algorithm::SpectralSubtraction
                    | Algorithm::SpecSubNonlinear
                    | Algorithm::SpecSubGeometric
            );
        if use_mb_specsub {
            let law = match algo {
                Algorithm::SpecSubNonlinear => SpecSubLaw::PowerLaw(0.75),
                Algorithm::SpecSubGeometric => SpecSubLaw::Geometric,
                _ => SpecSubLaw::Linear,
            };
            let mb = multiband_specsub_gains(&gamma_frame, &self.bark_bands, N_BARK_BANDS, gp, law);
            for (k, &mb_k) in mb.iter().enumerate().take(m) {
                self.g[k] = mb_k.max(g_min);
            }
        } else {
            for (k, &spp_k) in self.spp_buf.iter().enumerate().take(m) {
                let mut gk = compute_gain(algo, xi_frame[k], gamma_frame[k], spp_k, gp);
                if gk < g_min {
                    gk = g_min;
                }

                // Transient protection (spectral flux based):
                if tscore > 0.03 {
                    let protect = (tscore * 0.85).min(0.96);
                    gk = gk * (1.0 - protect) + 1.0 * protect;
                    gk = gk.clamp(g_min, 1.0);
                }

                // Attack/release smoothing.
                let gs = if gk >= self.prev_gsmooth[k] {
                    gk
                } else {
                    smoothing * self.prev_gsmooth[k] + (1.0 - smoothing) * gk
                };
                self.prev_gsmooth[k] = gs;
                self.g[k] = gs;
            }
        }

        // Perceptual Bark weighting.
        if self.config.perceptual_weighting {
            apply_perceptual_weights(&mut self.g, &self.bark_bands, self.config.strength, g_min);
        }

        // Musical-noise post-filter.
        if self.config.musical_noise_postfilter {
            self.postfilter
                .apply(&self.y2, &self.lambda_d_buf, &mut self.g);
        }

        // Stash for decision-directed recursion.
        for (k, &lam_k) in self.lambda_d_buf.iter().enumerate().take(m) {
            self.prev_g[k] = self.g[k];
            self.prev_y2[k] = self.y2[k];
            self.prev_lambda_d[k] = lam_k;
        }

        // Cepstral smoothing on the final gain curve (after temporal smoothing)
        // for superior musical-noise suppression while retaining timbre.
        // Uses full FFT-based cepstral liftering (proper implementation).
        if self.config.cepstral_smoothing {
            self.cepstral_smooth_gains();
            // Re-apply min floor after smoothing
            let gmin = g_min;
            for gi in &mut self.g {
                if *gi < gmin {
                    *gi = gmin;
                }
            }
        }

        self.apply_gain();
        self.stft.synthesize(&mut self.spec, out, norm, start);
    }

    /// Denoise a single (mono) channel of `f64` samples in `[-1, 1]`.
    pub fn process_channel(&mut self, input: &[f64]) -> Vec<f64> {
        self.reset_for_channel();
        let sanitized: Vec<f64> = input.iter().copied().map(sanitize_sample).collect();
        let mut x: Vec<f64> = if self.config.dc_block {
            Self::dc_block(&sanitized)
        } else {
            sanitized
        };
        if self.config.pre_emphasis {
            x = self.pre_emphasize(&x);
        }
        let total = x.len();

        // Noise profiling.
        let profile_frames = if self.config.profile_ms > 0.0 {
            self.explicit_profile_frames()
        } else if self.config.profile_ms == 0.0 {
            self.detect_profile_frames(&x)
        } else {
            0
        };
        if profile_frames > 0 {
            let prof = self.collect_profile_y2(&x, profile_frames);
            if !prof.is_empty() {
                self.noise.seed_from_profile(&prof);
            }
        }

        // Pad the signal by one frame of zeros at each end so every original
        // sample lies in the fully-overlapped interior of the overlap-add. This
        // avoids the edge blow-up where a single frame overlaps and the Hann
        // window value is ~0 (which would make out/norm = IFFT(spec*w)/w
        // explode). The zero-padding frames are skipped by the noise estimator's
        // bootstrap (see `NoiseEstimator::update`) so they do not corrupt the
        // noise estimate.
        let n = self.frame_size;
        let hop = self.hop;
        let plen = total + 2 * n;
        let mut padded = vec![0.0; plen];
        padded[n..n + total].copy_from_slice(&x);

        let mut out = vec![0.0; plen];
        let mut norm = vec![0.0; plen];

        let mut start = 0usize;
        let mut frame_idx = 0usize;
        while start + n <= plen {
            self.process_frame(&padded, start, frame_idx, &mut out, &mut norm);
            start += hop;
            frame_idx += 1;
        }

        // Perfect-reconstruction OLA normalization + makeup gain, over the
        // original (interior) sample range only.
        let makeup = self.makeup;
        let mut result = vec![0.0; total];
        for i in 0..total {
            let nv = norm[n + i];
            if nv > 1e-9 {
                result[i] = (out[n + i] / nv) * makeup;
            } else {
                result[i] = 0.0;
            }
        }

        // De-emphasis (must be applied after reconstruction to invert pre-emphasis correctly)
        if self.config.pre_emphasis {
            result = self.de_emphasize(&result);
        }
        for sample in &mut result {
            *sample = sanitize_sample(*sample);
        }
        result
    }

    /// Denoise `channels` (one `Vec<f64>` per channel), processed independently.
    pub fn process(&mut self, channels: &[Vec<f64>]) -> Vec<Vec<f64>> {
        channels.iter().map(|ch| self.process_channel(ch)).collect()
    }
}

/// Stateful classical denoiser for bounded-memory, block-by-block processing.
///
/// The stream has the same one-frame zero padding and overlap-add semantics as
/// [`Denoiser::process_channel`].  A small, bounded prefix is retained while
/// automatic or explicit noise profiling is initialized; after that, only the
/// STFT overlap and the current input block are resident in memory.
pub struct StreamingDenoiser {
    channels: Vec<ChannelStream>,
    finished: bool,
}

struct ChannelStream {
    denoiser: Denoiser,
    input: VecDeque<f64>,
    profile: Vec<f64>,
    profile_target: usize,
    profile_ready: bool,
    frame: Vec<f64>,
    frame_out: Vec<f64>,
    frame_norm: Vec<f64>,
    ola_out: Vec<f64>,
    ola_norm: Vec<f64>,
    pending: VecDeque<f64>,
    frame_idx: usize,
    input_frames: usize,
    emitted_padded: usize,
    discarded_left: usize,
    returned_frames: usize,
    finished: bool,
}

impl ChannelStream {
    fn try_new(config: DenoiserConfig, profile_target: usize) -> Result<Self, ConfigError> {
        let denoiser = Denoiser::try_new(config)?;
        let n = denoiser.frame_size;
        let doubled_frame = n.checked_mul(2).ok_or(ConfigError::ResourceOverflow {
            resource: "stream input",
        })?;
        let profiled_input =
            profile_target
                .checked_add(n)
                .ok_or(ConfigError::ResourceOverflow {
                    resource: "stream input",
                })?;
        let input_capacity = doubled_frame.max(profiled_input);
        let mut input = VecDeque::new();
        input
            .try_reserve_exact(input_capacity)
            .map_err(|_| ConfigError::allocation_failed("stream input"))?;
        if profile_target == 0 {
            input.extend(std::iter::repeat(0.0).take(n));
        }
        let mut profile = Vec::new();
        profile
            .try_reserve_exact(profile_target)
            .map_err(|_| ConfigError::allocation_failed("stream profile"))?;
        let mut pending = VecDeque::new();
        pending
            .try_reserve_exact(n)
            .map_err(|_| ConfigError::allocation_failed("stream output"))?;
        Ok(Self {
            denoiser,
            input,
            profile,
            profile_ready: profile_target == 0,
            profile_target,
            frame: try_zeroed_f64(n, "stream frame")?,
            frame_out: try_zeroed_f64(n, "stream frame output")?,
            frame_norm: try_zeroed_f64(n, "stream frame normalization")?,
            ola_out: try_zeroed_f64(n, "stream overlap output")?,
            ola_norm: try_zeroed_f64(n, "stream overlap normalization")?,
            pending,
            frame_idx: 0,
            input_frames: 0,
            emitted_padded: 0,
            discarded_left: 0,
            returned_frames: 0,
            finished: false,
        })
    }

    fn try_reserve_block(&mut self, frames: usize) -> Result<Vec<f64>, ConfigError> {
        let next_input_frames =
            self.input_frames
                .checked_add(frames)
                .ok_or(ConfigError::ResourceOverflow {
                    resource: "stream frame count",
                })?;
        let returned_capacity = next_input_frames.checked_sub(self.returned_frames).ok_or(
            ConfigError::ResourceOverflow {
                resource: "stream returned block",
            },
        )?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(returned_capacity)
            .map_err(|_| ConfigError::allocation_failed("stream returned block"))?;

        let n = self.denoiser.frame_size;
        let profile_remaining = if self.profile_ready {
            0
        } else {
            self.profile_target.saturating_sub(self.profile.len())
        };
        let profile_samples = frames.min(profile_remaining);
        self.profile
            .try_reserve_exact(profile_samples)
            .map_err(|_| ConfigError::allocation_failed("stream profile block"))?;

        let crosses_profile = !self.profile_ready && frames >= profile_remaining;
        let pending_samples = if crosses_profile {
            // initialize_profile() transfers the entire retained prefix after
            // adding a left-padding frame, then this block contributes every
            // sample (including any remainder after the boundary).
            let transition_samples = n
                .checked_add(self.profile.len())
                .and_then(|samples| samples.checked_add(frames))
                .ok_or(ConfigError::ResourceOverflow {
                    resource: "stream profile transition",
                })?;
            self.input
                .try_reserve_exact(transition_samples)
                .map_err(|_| ConfigError::allocation_failed("stream input block"))?;
            transition_samples
        } else {
            if self.profile_ready {
                self.input
                    .try_reserve_exact(frames)
                    .map_err(|_| ConfigError::allocation_failed("stream input block"))?;
            }
            // A pre-existing partial frame can make processing emit up to one
            // frame more than the new block length.
            frames.checked_add(n).ok_or(ConfigError::ResourceOverflow {
                resource: "stream output block",
            })?
        };
        self.pending
            .try_reserve_exact(pending_samples)
            .map_err(|_| ConfigError::allocation_failed("stream output block"))?;
        self.emitted_padded
            .checked_add(pending_samples)
            .ok_or(ConfigError::ResourceOverflow {
                resource: "stream frame count",
            })?;
        Ok(output)
    }

    #[inline]
    fn transform_sample(&mut self, sample: f64) -> f64 {
        let mut value = sanitize_sample(sample);
        if self.denoiser.config.dc_block {
            value = self.denoiser.dc_block_sample(value);
        }
        if self.denoiser.config.pre_emphasis {
            value = self.denoiser.pre_emphasize_sample(value);
        }
        value
    }

    fn initialize_profile(&mut self) {
        if self.profile_ready {
            return;
        }
        let profile = std::mem::take(&mut self.profile);
        let profile_frames = if self.denoiser.config.profile_ms > 0.0 {
            self.denoiser.explicit_profile_frames()
        } else if self.denoiser.config.profile_ms == 0.0 {
            self.denoiser.detect_profile_frames(&profile)
        } else {
            0
        };
        if profile_frames > 0 {
            let frames = self.denoiser.collect_profile_y2(&profile, profile_frames);
            if !frames.is_empty() {
                self.denoiser.noise.seed_from_profile(&frames);
            }
        }
        self.profile_ready = true;
        let n = self.denoiser.frame_size;
        self.input.extend(std::iter::repeat(0.0).take(n));
        self.input.extend(profile);
    }

    fn push_samples(&mut self, samples: &[f64]) {
        for &sample in samples {
            self.input_frames += 1;
            let value = self.transform_sample(sample);
            if self.profile_ready {
                self.input.push_back(value);
            } else {
                self.profile.push(value);
                if self.profile.len() >= self.profile_target {
                    self.initialize_profile();
                }
            }
        }
        if self.profile_ready {
            self.process_available();
        }
    }

    fn process_available(&mut self) {
        let n = self.denoiser.frame_size;
        let hop = self.denoiser.hop;
        while self.input.len() >= n {
            for i in 0..n {
                self.frame[i] = self.input[i];
            }
            self.frame_out.fill(0.0);
            self.frame_norm.fill(0.0);
            self.denoiser.process_frame(
                &self.frame,
                0,
                self.frame_idx,
                &mut self.frame_out,
                &mut self.frame_norm,
            );
            for i in 0..n {
                self.ola_out[i] += self.frame_out[i];
                self.ola_norm[i] += self.frame_norm[i];
            }
            let makeup = self.denoiser.makeup;
            for i in 0..hop {
                let norm = self.ola_norm[i];
                let value = if norm > 1e-9 {
                    (self.ola_out[i] / norm) * makeup
                } else {
                    0.0
                };
                self.pending.push_back(value);
            }
            self.ola_out.copy_within(hop..n, 0);
            self.ola_norm.copy_within(hop..n, 0);
            self.ola_out[n - hop..].fill(0.0);
            self.ola_norm[n - hop..].fill(0.0);
            for _ in 0..hop {
                self.input.pop_front();
            }
            self.frame_idx += 1;
            self.emitted_padded += hop;
        }
    }

    fn drain_ready_into(&mut self, output: &mut Vec<f64>) {
        let n = self.denoiser.frame_size;
        while self.discarded_left < n {
            if self.pending.pop_front().is_none() {
                break;
            }
            self.discarded_left += 1;
        }
        while self.returned_frames < self.input_frames {
            let Some(value) = self.pending.pop_front() else {
                break;
            };
            let value = if self.denoiser.config.pre_emphasis {
                self.denoiser.de_emphasize_sample(value)
            } else {
                value
            };
            output.push(value);
            self.returned_frames += 1;
        }
    }

    fn try_reserve_finish(&mut self) -> Result<Vec<f64>, ConfigError> {
        if self.finished {
            return Ok(Vec::new());
        }
        let n = self.denoiser.frame_size;
        let unreturned = self.input_frames.checked_sub(self.returned_frames).ok_or(
            ConfigError::ResourceOverflow {
                resource: "stream finish buffer",
            },
        )?;
        self.input_frames
            .checked_add(n)
            .ok_or(ConfigError::ResourceOverflow {
                resource: "stream finish frame count",
            })?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(unreturned)
            .map_err(|_| ConfigError::allocation_failed("stream returned finish"))?;

        let input_samples = if self.profile_ready {
            n
        } else {
            n.checked_mul(2)
                .and_then(|samples| samples.checked_add(self.profile.len()))
                .ok_or(ConfigError::ResourceOverflow {
                    resource: "stream finish input",
                })?
        };
        self.input
            .try_reserve_exact(input_samples)
            .map_err(|_| ConfigError::allocation_failed("stream finish input"))?;

        // Completing the final full frames can overshoot the exact padded
        // target by one hop, so keep two frames beyond all unreturned input.
        let finish_samples = n
            .checked_mul(2)
            .and_then(|samples| samples.checked_add(unreturned))
            .ok_or(ConfigError::ResourceOverflow {
                resource: "stream finish buffer",
            })?;
        self.pending
            .try_reserve_exact(finish_samples)
            .map_err(|_| ConfigError::allocation_failed("stream finish output"))?;

        self.emitted_padded
            .checked_add(finish_samples)
            .ok_or(ConfigError::ResourceOverflow {
                resource: "stream finish frame count",
            })?;
        Ok(output)
    }

    fn finish_into(&mut self, output: &mut Vec<f64>) {
        if self.finished {
            return;
        }
        let n = self.denoiser.frame_size;
        if !self.profile_ready {
            self.initialize_profile();
        }
        self.input.extend(std::iter::repeat(0.0).take(n));
        self.process_available();
        // Checked by try_reserve_finish before any channel is advanced.
        let target = n + self.input_frames;
        if self.emitted_padded < target {
            let remaining = (target - self.emitted_padded).min(n);
            let makeup = self.denoiser.makeup;
            for i in 0..remaining {
                let norm = self.ola_norm[i];
                let value = if norm > 1e-9 {
                    (self.ola_out[i] / norm) * makeup
                } else {
                    0.0
                };
                self.pending.push_back(value);
            }
            self.emitted_padded += remaining;
        }
        self.drain_ready_into(output);
        self.finished = true;
    }
}

fn try_zeroed_f64(length: usize, resource: &'static str) -> Result<Vec<f64>, ConfigError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ConfigError::allocation_failed(resource))?;
    values.resize(length, 0.0);
    Ok(values)
}

impl StreamingDenoiser {
    /// Create a stateful denoiser with one independent processor per channel.
    pub fn new(config: DenoiserConfig, channels: usize) -> Result<Self, String> {
        config
            .validate_config()
            .map_err(|error| error.to_string())?;
        let plan = ResourcePlan::for_stream(
            channels,
            config.frame_size,
            config.sample_rate,
            config.profile_ms,
        )
        .map_err(|error| error.to_string())?;
        let mut channel_streams = Vec::new();
        channel_streams
            .try_reserve_exact(channels)
            .map_err(|_| ConfigError::allocation_failed("stream channels").to_string())?;
        for _ in 0..channels {
            channel_streams.push(
                ChannelStream::try_new(config.clone(), plan.profile_target_samples())
                    .map_err(|error| error.to_string())?,
            );
        }
        Ok(Self {
            channels: channel_streams,
            finished: false,
        })
    }

    /// Process one interleaved-time, planar block and return any output that
    /// is ready without waiting for the stream to finish.
    pub fn process_block(&mut self, channels: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("streaming denoiser has already been finished".into());
        }
        if channels.len() != self.channels.len() {
            return Err(format!(
                "expected {} channels, got {}",
                self.channels.len(),
                channels.len()
            ));
        }
        let frames = channels.first().map(Vec::len).unwrap_or(0);
        if channels.iter().any(|channel| channel.len() != frames) {
            return Err("streaming blocks must have equal channel lengths".into());
        }
        if frames > 0 {
            let config = &self.channels[0].denoiser.config;
            checked_stream_memory_bytes(
                self.channels.len(),
                frames,
                config.frame_size,
                config.sample_rate,
                config.profile_ms,
            )
            .map_err(|error| error.to_string())?;
        }

        // Phase one reserves every internal queue and every returned vector
        // for every channel.  Capacity changes are harmless if a later
        // reservation fails; no samples or DSP state have been consumed yet.
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.channels.len())
            .map_err(|_| ConfigError::allocation_failed("stream returned channels").to_string())?;
        for stream in &mut self.channels {
            output.push(
                stream
                    .try_reserve_block(frames)
                    .map_err(|error| error.to_string())?,
            );
        }

        // Phase two is allocation-free for the stream queues and return
        // buffers, so all channels advance together.
        for ((stream, channel), channel_output) in
            self.channels.iter_mut().zip(channels).zip(&mut output)
        {
            stream.push_samples(channel);
            stream.drain_ready_into(channel_output);
        }
        Ok(output)
    }

    /// Flush the overlap-add tail and return the final output block.
    pub fn finish(&mut self) -> Result<Vec<Vec<f64>>, String> {
        if self.finished {
            return Err("streaming denoiser has already been finished".into());
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.channels.len())
            .map_err(|_| ConfigError::allocation_failed("stream returned channels").to_string())?;
        for stream in &mut self.channels {
            output.push(
                stream
                    .try_reserve_finish()
                    .map_err(|error| error.to_string())?,
            );
        }
        for (stream, channel_output) in self.channels.iter_mut().zip(&mut output) {
            stream.finish_into(channel_output);
        }
        self.finished = true;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple deterministic uniform-noise generator (no `rand` dependency).
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed.wrapping_add(0x9e3779b97f4a7c15))
        }
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Use the top 32 bits -> uniform in [0,1).
            let u = (self.0 >> 32) as f64 / (u32::MAX as f64 + 1.0);
            u * 2.0 - 1.0 // [-1, 1)
        }
    }

    fn invalid_field(error: ConfigError) -> &'static str {
        match error {
            ConfigError::InvalidValue { field, .. } => field,
            other => panic!("expected invalid field, got {other:?}"),
        }
    }

    #[test]
    fn strict_validation_rejects_nonfinite_major_float_fields() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut config = DenoiserConfig::default(48_000);
            config.strength = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "strength"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.overlap = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "overlap"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.vad_silence_gain = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "vad_silence_gain"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.vad_speech_mix = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "vad_speech_mix"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.smoothing = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "smoothing"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.profile_ms = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "profile_ms"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.makeup_gain_db = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "makeup_gain_db"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.pre_emphasis_alpha = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "pre_emphasis_alpha"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.window = WindowType::Kaiser;
            config.window_params.kaiser_beta = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "window_params.kaiser_beta"
            );

            let mut config = DenoiserConfig::default(48_000);
            config.window = WindowType::Dpss;
            config.window_params.dpss_bandwidth = value;
            assert_eq!(
                invalid_field(config.validate_config().unwrap_err()),
                "window_params.dpss_bandwidth"
            );
        }
    }

    #[test]
    fn strict_validation_accepts_boundaries_and_rejects_hostile_sizes() {
        let mut config = DenoiserConfig::default(1);
        config.strength = 0.0;
        config.overlap = 0.5;
        config.vad_silence_gain = 1.0;
        config.vad_speech_mix = 0.0;
        config.smoothing = 1.0;
        config.profile_ms = -f64::MAX;
        config.frame_size = MIN_DENOISER_FRAME_SIZE;
        config.pre_emphasis_alpha = 0.99;
        config.makeup_gain_db = MIN_MAKEUP_GAIN_DB;
        config.validate_config().unwrap();

        config.sample_rate = MAX_SAMPLE_RATE;
        config.frame_size = MAX_DENOISER_FRAME_SIZE;
        config.profile_ms = MAX_PROFILE_MS;
        config.strength = 1.0;
        config.overlap = 0.95;
        config.makeup_gain_db = MAX_MAKEUP_GAIN_DB;
        config.validate_config().unwrap();

        config.window = WindowType::Kaiser;
        config.window_params.kaiser_beta = MAX_KAISER_BETA;
        config.validate_config().unwrap();
        config.window_params.kaiser_beta = MAX_KAISER_BETA + 1.0;
        assert_eq!(
            invalid_field(config.validate_config().unwrap_err()),
            "window_params.kaiser_beta"
        );
        config.window = WindowType::Hann;
        config.validate_config().unwrap();

        config.makeup_gain_db = MAX_MAKEUP_GAIN_DB + 1.0;
        assert_eq!(
            invalid_field(config.validate_config().unwrap_err()),
            "makeup_gain_db"
        );
        config.makeup_gain_db = 0.0;

        config.frame_size = 131_072;
        assert_eq!(
            invalid_field(config.validate_config().unwrap_err()),
            "frame_size"
        );
        config.frame_size = 300;
        assert_eq!(
            invalid_field(config.validate_config().unwrap_err()),
            "frame_size"
        );
        config.frame_size = 2_048;
        config.sample_rate = 0;
        assert_eq!(
            invalid_field(config.validate_config().unwrap_err()),
            "sample_rate"
        );
        config.sample_rate = MAX_SAMPLE_RATE + 1;
        assert_eq!(
            invalid_field(config.validate_config().unwrap_err()),
            "sample_rate"
        );
        config.sample_rate = 48_000;
        config.profile_ms = MAX_PROFILE_MS + 1.0;
        assert_eq!(
            invalid_field(config.validate_config().unwrap_err()),
            "profile_ms"
        );
    }

    #[test]
    fn legacy_new_repairs_nonfinite_values_and_huge_frames_without_panicking() {
        let mut config = DenoiserConfig::default(0);
        config.strength = f64::NAN;
        config.overlap = f64::INFINITY;
        config.vad_silence_gain = f64::NEG_INFINITY;
        config.vad_speech_mix = f64::NAN;
        config.smoothing = f64::INFINITY;
        config.profile_ms = f64::NAN;
        config.makeup_gain_db = f64::INFINITY;
        config.pre_emphasis_alpha = f64::NAN;
        config.frame_size = 1 << 20;

        let denoiser = std::panic::catch_unwind(|| Denoiser::new(config))
            .expect("legacy constructor must repair hostile configuration");
        assert_eq!(denoiser.config().frame_size, 2_048);
        assert_eq!(denoiser.config().sample_rate, 48_000);
        assert!(denoiser.config().validate_config().is_ok());

        let mut strict = DenoiserConfig::default(48_000);
        strict.frame_size = 1 << 20;
        assert_eq!(
            invalid_field(Denoiser::try_new(strict).err().unwrap()),
            "frame_size"
        );

        let mut invalid_kaiser = DenoiserConfig::default(48_000);
        invalid_kaiser.window = WindowType::Kaiser;
        invalid_kaiser.window_params.kaiser_beta = f64::INFINITY;
        let repaired = Denoiser::new(invalid_kaiser.clone());
        assert_eq!(
            repaired.config().window_params.kaiser_beta,
            WindowParams::default().kaiser_beta
        );
        assert_eq!(
            invalid_field(Denoiser::try_new(invalid_kaiser).err().unwrap()),
            "window_params.kaiser_beta"
        );

        let mut excessive_makeup = DenoiserConfig::default(48_000);
        excessive_makeup.makeup_gain_db = f64::MAX;
        let repaired = Denoiser::new(excessive_makeup.clone());
        assert_eq!(repaired.config().makeup_gain_db, MAX_MAKEUP_GAIN_DB);
        assert_eq!(
            invalid_field(Denoiser::try_new(excessive_makeup).err().unwrap()),
            "makeup_gain_db"
        );
    }

    #[test]
    fn checked_and_legacy_constructors_share_effective_valid_config() {
        let mut config = DenoiserConfig::default(48_000);
        config.smoothing = 1.0;
        config.makeup_gain_db = MAX_MAKEUP_GAIN_DB;

        let legacy = Denoiser::new(config.clone());
        let checked = Denoiser::try_new(config).unwrap();
        assert_eq!(legacy.config().smoothing, 0.95);
        assert_eq!(checked.config().smoothing, legacy.config().smoothing);
        assert_eq!(
            checked.config().makeup_gain_db,
            legacy.config().makeup_gain_db
        );
        assert!(checked.makeup.is_finite());
    }

    #[test]
    fn streaming_rejects_channels_profiles_and_aggregate_resource_exhaustion() {
        let config = DenoiserConfig::default(48_000);
        assert!(StreamingDenoiser::new(config.clone(), 0)
            .err()
            .unwrap()
            .contains("channels"));
        assert!(
            StreamingDenoiser::new(config.clone(), crate::config::MAX_STREAM_CHANNELS + 1)
                .err()
                .unwrap()
                .contains("channels")
        );

        let mut invalid_profile = config.clone();
        invalid_profile.profile_ms = f64::INFINITY;
        assert!(StreamingDenoiser::new(invalid_profile, 1)
            .err()
            .unwrap()
            .contains("profile_ms"));

        let mut oversized = config;
        oversized.frame_size = MAX_DENOISER_FRAME_SIZE;
        oversized.sample_rate = MAX_SAMPLE_RATE;
        oversized.profile_ms = MAX_PROFILE_MS;
        assert!(StreamingDenoiser::new(oversized, 1)
            .err()
            .unwrap()
            .contains("streaming state"));
    }

    #[test]
    fn streaming_rejects_oversized_blocks_before_mutating_state() {
        let mut config = DenoiserConfig::default(48_000);
        config.profile_ms = -1.0;
        let mut stream = StreamingDenoiser::new(config, 1).unwrap();
        let oversized = vec![vec![0.0; crate::config::MAX_STREAM_BLOCK_FRAMES + 1]];
        let error = stream.process_block(&oversized).unwrap_err();
        assert!(error.contains("block_frames"), "unexpected error: {error}");

        let empty = stream.process_block(&[Vec::new()]).unwrap();
        assert_eq!(empty, vec![Vec::<f64>::new()]);
        assert!(stream.process_block(&[vec![0.0; 32]]).is_ok());
    }

    #[test]
    fn profile_crossing_preflight_covers_every_queue_push() {
        let mut config = DenoiserConfig::default(8_000);
        config.frame_size = MIN_DENOISER_FRAME_SIZE;
        config.overlap = 0.5;
        config.profile_ms = 1.0;
        config.dc_block = false;
        let profile_target =
            ResourcePlan::for_stream(1, config.frame_size, config.sample_rate, config.profile_ms)
                .unwrap()
                .profile_target_samples();
        let mut stream = ChannelStream::try_new(config, profile_target).unwrap();

        let prefix = vec![0.0; profile_target - 1];
        let mut prefix_output = stream.try_reserve_block(prefix.len()).unwrap();
        stream.push_samples(&prefix);
        stream.drain_ready_into(&mut prefix_output);
        assert!(!stream.profile_ready);
        assert_eq!(stream.profile.len(), profile_target - 1);

        let crossing = [0.0, 0.0];
        let transition_samples = stream.denoiser.frame_size + stream.profile.len() + crossing.len();
        let mut output = stream.try_reserve_block(crossing.len()).unwrap();
        assert!(stream.input.capacity() - stream.input.len() >= transition_samples);
        assert!(stream.pending.capacity() - stream.pending.len() >= transition_samples);
        assert!(output.capacity() - output.len() >= profile_target + 1);

        let input_capacity = stream.input.capacity();
        let pending_capacity = stream.pending.capacity();
        stream.push_samples(&crossing);
        stream.drain_ready_into(&mut output);
        assert!(stream.profile_ready);
        assert_eq!(stream.input.capacity(), input_capacity);
        assert_eq!(stream.pending.capacity(), pending_capacity);
    }

    #[test]
    fn incomplete_profile_finish_is_fully_preflighted() {
        let mut config = DenoiserConfig::default(8_000);
        config.frame_size = MIN_DENOISER_FRAME_SIZE;
        config.overlap = 0.5;
        config.profile_ms = 100.0;
        config.dc_block = false;
        let profile_target =
            ResourcePlan::for_stream(1, config.frame_size, config.sample_rate, config.profile_ms)
                .unwrap()
                .profile_target_samples();
        let mut stream = ChannelStream::try_new(config, profile_target).unwrap();
        // Stop one sample before the profile boundary.  At this point finish
        // must hold the retained prefix plus both padding frames at once.
        let prefix = vec![0.0; profile_target - 1];
        let mut prefix_output = stream.try_reserve_block(prefix.len()).unwrap();
        stream.push_samples(&prefix);
        stream.drain_ready_into(&mut prefix_output);

        let mut output = stream.try_reserve_finish().unwrap();
        let input_capacity = stream.input.capacity();
        let pending_capacity = stream.pending.capacity();
        stream.finish_into(&mut output);

        assert_eq!(output.len(), prefix.len());
        assert_eq!(stream.input.capacity(), input_capacity);
        assert_eq!(stream.pending.capacity(), pending_capacity);
    }

    #[test]
    fn multichannel_preflight_error_does_not_advance_an_earlier_channel() {
        let mut config = DenoiserConfig::default(8_000);
        config.frame_size = MIN_DENOISER_FRAME_SIZE;
        config.overlap = 0.5;
        config.profile_ms = -1.0;
        config.dc_block = false;
        let mut stream = StreamingDenoiser::new(config, 2).unwrap();

        // Make only the second channel's return reservation impossible.  This
        // models a late-channel allocation failure without asking the test
        // process to commit an enormous allocation.
        stream.channels[1].input_frames = usize::MAX / 2;
        let first_before = (
            stream.channels[0].input_frames,
            stream.channels[0].frame_idx,
            stream.channels[0].input.len(),
            stream.channels[0].pending.len(),
        );
        let error = stream
            .process_block(&[vec![0.25; 1], vec![0.25; 1]])
            .unwrap_err();
        assert!(
            error.contains("stream returned block"),
            "unexpected error: {error}"
        );
        assert_eq!(
            (
                stream.channels[0].input_frames,
                stream.channels[0].frame_idx,
                stream.channels[0].input.len(),
                stream.channels[0].pending.len(),
            ),
            first_before
        );

        // Restore the synthetic counter and prove both DSP states still
        // produce identical output for identical subsequent input.
        stream.channels[1].input_frames = 0;
        let samples = vec![0.25; 3 * MIN_DENOISER_FRAME_SIZE];
        let mut output = stream.process_block(&[samples.clone(), samples]).unwrap();
        let mut tail = stream.finish().unwrap();
        output[0].append(&mut tail[0]);
        output[1].append(&mut tail[1]);
        assert_eq!(output[0], output[1]);
    }

    #[test]
    fn multichannel_finish_error_does_not_finish_an_earlier_channel() {
        let mut config = DenoiserConfig::default(8_000);
        config.frame_size = MIN_DENOISER_FRAME_SIZE;
        config.overlap = 0.5;
        config.profile_ms = -1.0;
        config.dc_block = false;
        let mut stream = StreamingDenoiser::new(config, 2).unwrap();
        let samples = vec![0.125; MIN_DENOISER_FRAME_SIZE + 17];
        stream.process_block(&[samples.clone(), samples]).unwrap();

        let second_input_frames = stream.channels[1].input_frames;
        stream.channels[1].input_frames = usize::MAX / 2;
        let first_before = (
            stream.channels[0].frame_idx,
            stream.channels[0].input.len(),
            stream.channels[0].pending.len(),
            stream.channels[0].emitted_padded,
            stream.channels[0].returned_frames,
            stream.channels[0].finished,
        );
        let error = stream.finish().unwrap_err();
        assert!(
            error.contains("stream returned finish"),
            "unexpected error: {error}"
        );
        assert!(!stream.finished);
        assert_eq!(
            (
                stream.channels[0].frame_idx,
                stream.channels[0].input.len(),
                stream.channels[0].pending.len(),
                stream.channels[0].emitted_padded,
                stream.channels[0].returned_frames,
                stream.channels[0].finished,
            ),
            first_before
        );

        stream.channels[1].input_frames = second_input_frames;
        let output = stream.finish().unwrap();
        assert_eq!(output[0], output[1]);
    }

    #[test]
    fn default_and_named_presets_keep_expected_windows() {
        let sample_rate = 48_000;
        assert_eq!(
            DenoiserConfig::default(sample_rate).window,
            WindowType::Hann
        );
        for preset in [
            Preset::Speech,
            Preset::Music,
            Preset::Aggressive,
            Preset::Gentle,
            Preset::Restore,
        ] {
            assert_eq!(
                preset.config(sample_rate).window,
                WindowType::Hann,
                "{preset:?} changed its established window"
            );
        }

        let hifi = Preset::HiFi.config(sample_rate);
        assert_eq!(hifi.window, WindowType::Kaiser);
        assert_eq!(hifi.window_params.kaiser_beta, 10.0);
    }

    #[test]
    fn dpss_configuration_validation_and_sanitization_are_safe() {
        let mut base = DenoiserConfig::default(48_000);
        base.frame_size = 256;
        base.window = WindowType::Dpss;

        for invalid in [
            0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            MAX_DENOISER_DPSS_NW + 0.5,
        ] {
            let mut config = base.clone();
            config.window_params.dpss_bandwidth = invalid;
            let error = config.validate().unwrap_err();
            assert!(
                error.contains("DPSS bandwidth"),
                "unexpected error: {error}"
            );

            // The legacy infallible constructor must repair bad values before
            // it reaches the checked window generator.
            let denoiser = Denoiser::new(config);
            assert_eq!(
                denoiser.config().window_params.dpss_bandwidth,
                WindowParams::default().dpss_bandwidth
            );
        }

        for valid in [0.5, 3.0, MAX_DENOISER_DPSS_NW] {
            let mut config = base.clone();
            config.window_params.dpss_bandwidth = valid;
            config.validate().unwrap();
            assert_eq!(config.sanitized().window_params.dpss_bandwidth, valid);
        }

        let mut repaired_frame = base.clone();
        repaired_frame.frame_size = 300;
        repaired_frame.window_params.dpss_bandwidth = f64::NAN;
        let repaired_frame = repaired_frame.sanitized();
        assert_eq!(repaired_frame.frame_size, 2048);
        assert_eq!(
            repaired_frame.window_params.dpss_bandwidth,
            WindowParams::default().dpss_bandwidth
        );

        let mut dormant = DenoiserConfig::default(48_000);
        dormant.window_params.dpss_bandwidth = f64::NAN;
        dormant.validate().unwrap();
        let dormant = Denoiser::new(dormant);
        assert_eq!(dormant.config().window, WindowType::Hann);
        assert!(dormant.config().window_params.dpss_bandwidth.is_nan());
    }

    #[test]
    fn streaming_constructor_rejects_invalid_dpss_bandwidth() {
        let mut config = DenoiserConfig::default(48_000);
        config.window = WindowType::Dpss;
        config.window_params.dpss_bandwidth = MAX_DENOISER_DPSS_NW + 0.5;

        let error = StreamingDenoiser::new(config, 1).err().unwrap();
        assert!(
            error.contains("DPSS bandwidth"),
            "unexpected error: {error}"
        );
    }

    fn snr_db(clean: &[f64], test: &[f64]) -> f64 {
        let mut sc = 0.0;
        let mut sn = 0.0;
        for i in 0..clean.len() {
            sc += clean[i] * clean[i];
            let e = test[i] - clean[i];
            sn += e * e;
        }
        10.0 * (sc / sn.max(1e-300)).log10()
    }

    #[test]
    fn nonfinite_extreme_and_silent_inputs_remain_safe() {
        let mut config = DenoiserConfig::default(16_000);
        config.frame_size = 256;
        config.overlap = 0.75;
        config.profile_ms = -1.0;
        config.dc_block = true;
        config.pre_emphasis = true;
        let mut input = vec![0.0; 1_023];
        input[0] = f64::NAN;
        input[1] = f64::INFINITY;
        input[2] = f64::NEG_INFINITY;
        input[3] = 1e300;
        input[4] = -1e300;
        input[5] = 0.35;

        let mut denoiser = Denoiser::new(config.clone());
        let output = denoiser.process_channel(&input);
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().all(|sample| sample.abs() <= 1.0));

        let mut silent_denoiser = Denoiser::new(config);
        let silence = silent_denoiser.process_channel(&vec![0.0; 1_023]);
        assert!(silence.iter().all(|sample| sample.is_finite()));
        assert!(silence.iter().all(|sample| sample.abs() <= 1e-12));

        let mut empty_denoiser = Denoiser::new(DenoiserConfig::default(16_000));
        assert!(empty_denoiser.process_channel(&[]).is_empty());
    }

    #[test]
    fn dc_block_removes_constant_offset_and_matches_streaming_state() {
        let input = vec![0.25; 16_000];
        let batch = Denoiser::dc_block(&input);
        let tail_mean = batch[15_000..].iter().sum::<f64>() / 1_000.0;
        assert!(tail_mean.abs() < 1e-5, "residual DC offset: {tail_mean}");

        let mut stream = Denoiser::new(DenoiserConfig::default(16_000));
        let streaming: Vec<_> = input
            .iter()
            .copied()
            .map(|sample| stream.dc_block_sample(sample))
            .collect();
        assert_eq!(streaming, batch);
    }

    #[test]
    fn denoising_improves_snr() {
        let sr: u32 = 16000;
        let dur = 2.0;
        let n = (sr as f64 * dur) as usize;
        let silence = (sr as f64 * 0.3) as usize; // 0.3 s leading noise-only

        // Clean: silence then a two-tone signal.
        let mut clean = vec![0.0; n];
        for (i, c) in clean.iter_mut().enumerate().take(n).skip(silence) {
            let t = i as f64 / sr as f64;
            *c = 0.30 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 880.0 * t).sin();
        }

        // Noise scaled to ~0 dB SNR in the tone region.
        let pc: f64 = clean[silence..].iter().map(|s| s * s).sum::<f64>() / (n - silence) as f64;
        let pn = pc; // 0 dB
        let scale = (3.0 * pn).sqrt(); // uniform[-1,1] variance = 1/3
        let mut rng = Lcg::new(12345);
        let noise: Vec<f64> = (0..n).map(|_| scale * rng.uniform()).collect();

        let noisy: Vec<f64> = (0..n).map(|i| clean[i] + noise[i]).collect();
        let in_snr = snr_db(&clean[silence..], &noisy[silence..]);

        let mut den = Denoiser::new(Preset::Speech.config(sr));
        let out = den.process_channel(&noisy);
        assert_eq!(out.len(), noisy.len());

        // Compare over the interior of the tone region (avoid edge effects).
        let edge = 4096;
        let lo = silence + edge;
        let hi = n - edge;
        let out_snr = snr_db(&clean[lo..hi], &out[lo..hi]);

        assert!(
            out_snr > in_snr + 3.0,
            "expected SNR improvement > 3 dB, got in={in_snr:.2} out={out_snr:.2}"
        );
    }

    #[test]
    fn clean_signal_is_preserved() {
        let sr: u32 = 16000;
        let n = sr as usize * 2;
        let silence = sr as usize / 3;
        let mut clean = vec![0.0; n];
        for (i, c) in clean.iter_mut().enumerate().take(n).skip(silence) {
            let t = i as f64 / sr as f64;
            *c = 0.25 * (2.0 * std::f64::consts::PI * 660.0 * t).sin();
        }
        let mut den = Denoiser::new(Preset::Restore.config(sr));
        let out = den.process_channel(&clean);

        // The tone amplitude should be preserved to within a few percent.
        let lo = silence + 4096;
        let hi = n - 4096;
        let in_rms = (clean[lo..hi].iter().map(|s| s * s).sum::<f64>() / (hi - lo) as f64).sqrt();
        let out_rms = (out[lo..hi].iter().map(|s| s * s).sum::<f64>() / (hi - lo) as f64).sqrt();
        let rel = (out_rms - in_rms).abs() / in_rms;
        assert!(rel < 0.06, "tone amplitude changed by {rel:.3}");
    }

    #[test]
    fn streaming_matches_batch_with_bounded_blocks() {
        let sr = 16_000;
        let mut config = Preset::Gentle.config(sr);
        config.frame_size = 512;
        config.overlap = 0.75;
        config.profile_ms = -1.0;
        config.dc_block = false;
        config.pre_emphasis = true;
        let signal: Vec<f64> = (0..sr as usize * 2)
            .map(|i| {
                let t = i as f64 / sr as f64;
                0.25 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
                    + 0.04 * (2.0 * std::f64::consts::PI * 2_700.0 * t).sin()
            })
            .collect();

        let mut batch = Denoiser::new(config.clone());
        let expected = batch.process_channel(&signal);
        let mut streaming = StreamingDenoiser::new(config, 1).unwrap();
        let mut actual = Vec::new();
        let mut offset = 0;
        for block_size in [37, 1_003, 257, 4_096, 89, 777] {
            if offset >= signal.len() {
                break;
            }
            let end = (offset + block_size).min(signal.len());
            let block = vec![signal[offset..end].to_vec()];
            actual.extend(streaming.process_block(&block).unwrap().remove(0));
            offset = end;
        }
        if offset < signal.len() {
            actual.extend(
                streaming
                    .process_block(&[signal[offset..].to_vec()])
                    .unwrap()
                    .remove(0),
            );
        }
        actual.extend(streaming.finish().unwrap().remove(0));
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_error < 1e-9,
            "streaming drifted from batch by {max_error}"
        );
    }

    #[test]
    fn dpss_streaming_matches_batch_with_irregular_blocks() {
        let sample_rate = 16_000;
        let mut config = DenoiserConfig::default(sample_rate);
        config.frame_size = 256;
        config.overlap = 0.5;
        config.window = WindowType::Dpss;
        config.window_params.dpss_bandwidth = 3.0;
        config.profile_ms = -1.0;
        config.dc_block = false;
        config.pre_emphasis = false;
        let signal_len = 5 * config.frame_size + 37;
        let signal: Vec<f64> = (0..signal_len)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                0.25 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
                    + 0.04 * (2.0 * std::f64::consts::PI * 2_700.0 * t).sin()
            })
            .collect();

        let mut batch = Denoiser::new(config.clone());
        let expected = batch.process_channel(&signal);
        let mut streaming = StreamingDenoiser::new(config, 1).unwrap();
        let block_sizes = [1, 37, 513, 2, 89, 257, 11];
        let mut actual = Vec::new();
        let mut offset = 0;
        let mut block_index = 0;
        while offset < signal.len() {
            let end = (offset + block_sizes[block_index % block_sizes.len()]).min(signal.len());
            actual.extend(
                streaming
                    .process_block(&[signal[offset..end].to_vec()])
                    .unwrap()
                    .remove(0),
            );
            offset = end;
            block_index += 1;
        }
        actual.extend(streaming.finish().unwrap().remove(0));

        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(max_error < 1e-9, "DPSS streaming drifted by {max_error}");
    }

    #[test]
    fn hifi_preset_preserves_clean_and_enables_features() {
        let sr: u32 = 48000;
        let n = (sr as usize) * 2;
        // Use a short leading silence like the other preservation test
        let silence = sr as usize / 4;
        let mut clean = vec![0.0; n];
        for (i, c) in clean.iter_mut().enumerate().take(n).skip(silence) {
            let t = i as f64 / sr as f64;
            *c = 0.18 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
                + 0.09 * (2.0 * std::f64::consts::PI * 1760.0 * t).sin();
        }

        let mut cfg = Preset::HiFi.config(sr);
        // Enable the signature hi-fi features
        cfg.cepstral_smoothing = true;
        cfg.transient_protect = true;
        cfg.pre_emphasis = false;
        cfg.strength = 0.28;

        let mut den = Denoiser::new(cfg);
        let out = den.process_channel(&clean);

        // Compare on the interior active region (avoid edges and leading silence)
        let edge = 4096;
        let lo = silence + edge;
        let hi = n - edge;
        let in_rms = (clean[lo..hi].iter().map(|s| s * s).sum::<f64>() / (hi - lo) as f64).sqrt();
        let out_rms = (out[lo..hi].iter().map(|s| s * s).sum::<f64>() / (hi - lo) as f64).sqrt();
        let rel = (out_rms - in_rms).abs() / in_rms;

        // HiFi mode with cepstral + transient can have small level shifts on pure tones.
        // We still want it under ~12% for good fidelity.
        assert!(
            rel < 0.12,
            "hifi changed clean amplitude by {rel:.3} (too much for fidelity mode)"
        );

        // At least verify that the HiFi preset enables the main quality features by default
        let c = Preset::HiFi.config(sr);
        assert!(c.transient_protect);
        assert!(c.cepstral_smoothing);
        assert!(c.perceptual_weighting);
        assert!(c.musical_noise_postfilter);
        assert_eq!(c.window, WindowType::Kaiser);
    }

    #[test]
    fn content_modes_coordinate_processing_controls() {
        let mut speech = DenoiserConfig::default(48_000);
        ProcessingMode::Speech.apply(&mut speech);
        assert!(speech.vad && speech.adaptive_noise);
        assert!(speech.strength >= 0.7);

        let mut music = DenoiserConfig::default(48_000);
        ProcessingMode::Music.apply(&mut music);
        assert!(!music.vad && music.transient_protect);
        assert!(music.strength <= 0.35);
        assert!(music.perceptual_weighting);

        let mut ambient = DenoiserConfig::default(48_000);
        ProcessingMode::Ambient.apply(&mut ambient);
        assert!(ambient.adaptive_noise && !ambient.vad);
        assert!(ambient.strength <= 0.4);
    }
}
