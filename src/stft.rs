//! Short-time Fourier transform engine with perfect-reconstruction
//! overlap-add (OLA) synthesis.
//!
//! The analysis window `w_a` and synthesis window `w_s` are applied on the way
//! in and out. Perfect reconstruction of an *unmodified* spectrum is achieved
//! by normalizing each output sample by the accumulated
//! `sum_k w_a[n-kH] * w_s[n-kH]`, which is tracked in a parallel buffer. This
//! makes the OLA exact for *any* window and any overlap ratio, including the
//! 75%-overlap Hann configuration used by default (where the sum is not 1.0).

use crate::fft::{Complex, Fft};
use crate::window::{make_with_params, WindowParams, WindowType};

/// Configuration for the STFT engine.
#[derive(Clone, Copy, Debug)]
pub struct StftConfig {
    pub frame_size: usize,
    pub hop: usize,
    pub window: WindowType,
    pub window_params: WindowParams,
}

pub struct Stft {
    cfg: StftConfig,
    analysis: Vec<f64>,
    synthesis: Vec<f64>,
    fft: Fft,
}

impl Stft {
    pub fn new(cfg: StftConfig) -> Self {
        assert!(cfg.frame_size.is_power_of_two());
        assert!(cfg.hop > 0 && cfg.hop <= cfg.frame_size);
        // Identical analysis and synthesis windows: the normalization buffer
        // then holds sum_k w[n-kH]^2 which is smooth and strictly positive.
        let w = make_with_params(cfg.window, cfg.frame_size, &cfg.window_params);
        Stft {
            cfg,
            analysis: w.clone(),
            synthesis: w,
            fft: Fft::new(cfg.frame_size),
        }
    }

    #[inline]
    pub fn frame_size(&self) -> usize {
        self.cfg.frame_size
    }

    #[inline]
    pub fn hop(&self) -> usize {
        self.cfg.hop
    }

    #[inline]
    pub fn nbins(&self) -> usize {
        self.fft.nbins()
    }

    #[inline]
    pub fn fft(&self) -> &Fft {
        &self.fft
    }

    /// Window a time-domain frame (`len == frame_size`) and forward-transform
    /// it into `spec` (`len == frame_size`), with the imaginary part set up.
    pub fn analyze(&self, time: &[f64], spec: &mut [Complex]) {
        debug_assert_eq!(time.len(), self.cfg.frame_size);
        debug_assert_eq!(spec.len(), self.cfg.frame_size);
        for i in 0..self.cfg.frame_size {
            spec[i] = Complex::new(time[i] * self.analysis[i], 0.0);
        }
        self.fft.forward(spec);
    }

    /// Inverse-transform `spec`, apply the synthesis window, and overlap-add
    /// into `out` while accumulating the normalization weight into `norm`, at
    /// sample offset `start`.
    pub fn synthesize(
        &self,
        spec: &mut [Complex],
        out: &mut [f64],
        norm: &mut [f64],
        start: usize,
    ) {
        debug_assert_eq!(spec.len(), self.cfg.frame_size);
        self.fft.inverse(spec);
        let n = self.cfg.frame_size;
        // Guard against writing past the end of the output buffers.
        let end = (start + n).min(out.len());
        let lim = end - start;
        for i in 0..lim {
            let s = spec[i].re * self.synthesis[i];
            out[start + i] += s;
            norm[start + i] += self.analysis[i] * self.synthesis[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_dpss_roundtrip(nw: f64, hop: usize, min_norm_floor: f64, max_norm_ceiling: f64) {
        let n = 256;
        let signal_len = 5 * n + 37;
        let mut signal: Vec<f64> = (0..signal_len)
            .map(|i| {
                let i = i as f64;
                0.31 * (0.071 * i).sin()
                    + 0.17 * (0.193 * i + 0.2).cos()
                    + 0.08 * (0.417 * i - 0.4).sin()
            })
            .collect();
        signal[0] += 0.75;
        signal[signal_len - 1] -= 0.625;

        let mut window_params = WindowParams::default();
        window_params.dpss_bandwidth = nw;
        let stft = Stft::new(StftConfig {
            frame_size: n,
            hop,
            window: WindowType::Dpss,
            window_params,
        });

        // Match the production denoiser's one-frame padding on both sides.
        // The deliberately unaligned signal length also exercises the partial
        // overlap pattern at the right endpoint.
        let mut padded = vec![0.0; signal_len + 2 * n];
        padded[n..n + signal_len].copy_from_slice(&signal);
        let mut out = vec![0.0; padded.len()];
        let mut norm = vec![0.0; padded.len()];
        let mut spec = vec![Complex::default(); n];

        let mut start = 0;
        while start + n <= padded.len() {
            stft.analyze(&padded[start..start + n], &mut spec);
            stft.synthesize(&mut spec, &mut out, &mut norm, start);
            start += hop;
        }

        let covered_norm = &norm[n..n + signal_len];
        assert!(covered_norm.iter().all(|value| value.is_finite()));
        assert!(out[n..n + signal_len].iter().all(|value| value.is_finite()));
        let min_norm = covered_norm.iter().copied().fold(f64::INFINITY, f64::min);
        let max_norm = covered_norm
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            min_norm > min_norm_floor,
            "NW={nw}, hop={hop}: minimum OLA norm {min_norm} <= {min_norm_floor}"
        );
        assert!(
            max_norm < max_norm_ceiling,
            "NW={nw}, hop={hop}: maximum OLA norm {max_norm} >= {max_norm_ceiling}"
        );

        let max_error = signal
            .iter()
            .enumerate()
            .map(|(i, expected)| {
                let index = n + i;
                (out[index] / norm[index] - expected).abs()
            })
            .fold(0.0, f64::max);
        assert!(
            max_error < 1e-9,
            "NW={nw}, hop={hop}: reconstruction error {max_error}"
        );
    }

    #[test]
    fn perfect_reconstruction_hann_75pct() {
        let n = 1024;
        let hop = n / 4; // 75% overlap
        let stft = Stft::new(StftConfig {
            frame_size: n,
            hop,
            window: WindowType::Hann,
            window_params: WindowParams::default(),
        });

        // Build a test signal longer than one frame.
        let total = 8 * n;
        let signal: Vec<f64> = (0..total)
            .map(|i| (0.017 * i as f64).sin() + 0.5 * (0.003 * i as f64).cos())
            .collect();

        let mut out = vec![0.0; total];
        let mut norm = vec![0.0; total];
        let mut spec = vec![Complex::default(); n];
        let mut frame = vec![0.0; n];

        let mut start = 0;
        while start + n <= total {
            frame.copy_from_slice(&signal[start..start + n]);
            stft.analyze(&frame, &mut spec);
            // No modification -> must reconstruct exactly.
            stft.synthesize(&mut spec, &mut out, &mut norm, start);
            start += hop;
        }

        // Normalize by the OLA weight and compare in the fully-covered interior.
        let interior = n..total - n;
        let mut max_err: f64 = 0.0;
        for i in interior {
            let r = out[i] / norm[i];
            max_err = max_err.max((r - signal[i]).abs());
        }
        assert!(max_err < 1e-6, "reconstruction error too high: {max_err}");
    }

    #[test]
    fn dpss_roundtrip_preserves_unaligned_signal() {
        assert_dpss_roundtrip(3.0, 128, 0.19, f64::INFINITY);
        assert_dpss_roundtrip(3.0, 64, 1.15, f64::INFINITY);
        assert_dpss_roundtrip(4.0, 128, 0.08, 1.01);
        // Protect the application-level NW cap: even the strongest accepted
        // taper stays well above the runtime's 1e-9 normalization cutoff.
        assert_dpss_roundtrip(8.0, 128, 0.003, 1.01);
    }
}
