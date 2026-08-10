//! Window functions for short-time Fourier analysis/synthesis.
//!
//! All windows returned here are *periodic* with `N` samples, matching the
//! overlap convention used by the STFT.  Analytic families use a periodic
//! phase directly; DPSS uses the equivalent conventional construction of a
//! symmetric `N+1` sequence followed by truncation.  Exact COLA depends on the
//! selected family and hop size.

use std::f64::consts::PI;
use std::fmt;

/// Maximum DPSS time-bandwidth product accepted by denoiser configurations.
///
/// The DPSS generator itself supports the full mathematical range
/// `0 < NW < N / 2`.  This application-level upper limit keeps denoiser
/// configurations in the range useful for audio STFT processing.
pub const MAX_DENOISER_DPSS_NW: f64 = 8.0;

/// Error returned by checked window construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WindowError {
    /// The DPSS time-bandwidth product is not finite or is outside
    /// `0 < NW < N / 2`.
    InvalidDpssBandwidth {
        /// Requested periodic window length.
        n_total: usize,
        /// Requested time-bandwidth product.
        bandwidth: f64,
    },
    /// The DPSS eigensolver could not meet its bounded convergence criteria.
    DpssSolverDidNotConverge {
        /// Requested periodic window length.
        n_total: usize,
        /// Requested time-bandwidth product.
        bandwidth: f64,
    },
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            WindowError::InvalidDpssBandwidth { n_total, bandwidth } => write!(
                f,
                "DPSS bandwidth NW must be finite and satisfy 0 < NW < N/2 \
                 (N={n_total}, NW={bandwidth})"
            ),
            WindowError::DpssSolverDidNotConverge { n_total, bandwidth } => write!(
                f,
                "DPSS eigensolver did not converge (N={n_total}, NW={bandwidth})"
            ),
        }
    }
}

impl std::error::Error for WindowError {}

/// Supported window families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowType {
    /// Periodic Hann window. Default; COLA-exact at 50% overlap.
    Hann,
    /// Periodic Hamming window (slightly higher sidelobe attenuation trade-off).
    Hamming,
    /// Sine (a.k.a. half-sine / sqrt-Hann) window.
    Sine,
    /// Blackman window (low spectral leakage, narrower main lobe).
    Blackman,
    /// Kaiser-Bessel window. Adjustable sidelobe suppression via `kaiser_beta`.
    Kaiser,
    /// Flat-top window. Excellent amplitude accuracy; wider main lobe.
    FlatTop,
    /// DPSS (Discrete Prolate Spheroidal Sequence) / Slepian window.
    /// Excellent energy concentration; `dpss_bandwidth` sets time-bandwidth product NW.
    Dpss,
}

impl WindowType {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "hann" => WindowType::Hann,
            "hamming" => WindowType::Hamming,
            "sine" => WindowType::Sine,
            "blackman" => WindowType::Blackman,
            "kaiser" | "kaiser-bessel" => WindowType::Kaiser,
            "flattop" | "flat-top" | "flat_top" => WindowType::FlatTop,
            "dpss" | "slepian" => WindowType::Dpss,
            _ => return None,
        })
    }
}

/// Parameters for advanced windows.
#[derive(Clone, Copy, Debug)]
pub struct WindowParams {
    /// Kaiser β (typical 5–12 for audio; higher = lower sidelobes).
    pub kaiser_beta: f64,
    /// DPSS time-bandwidth product NW (typical 2.5–4.0).
    pub dpss_bandwidth: f64,
}

impl Default for WindowParams {
    fn default() -> Self {
        WindowParams {
            kaiser_beta: 8.0,
            dpss_bandwidth: 3.0,
        }
    }
}

/// Evaluate a single window sample `w(n)` for `n in 0..N` (periodic form).
///
/// # Panics
///
/// For DPSS lengths of at least two, panics if the default bandwidth is
/// invalid for `n_total` or the bounded eigensolver does not converge.
pub fn sample(kind: WindowType, n: usize, n_total: usize) -> f64 {
    sample_with_params(kind, n, n_total, &WindowParams::default())
}

/// Evaluate with explicit parameters (Kaiser β, DPSS NW).
///
/// A DPSS sample is an entry of a global eigenvector, so it cannot be
/// evaluated independently.  For [`WindowType::Dpss`] this function builds
/// the complete window once and selects `n`, using O(`n_total`) work and
/// memory.  Consequently its result agrees with [`make_with_params`] at every
/// in-range index.  Prefer building the window once when several samples are
/// needed.
///
/// # Panics
///
/// For DPSS lengths of at least two, panics if its parameters are invalid or
/// the bounded eigensolver does not converge.
pub fn sample_with_params(
    kind: WindowType,
    n: usize,
    n_total: usize,
    params: &WindowParams,
) -> f64 {
    if kind == WindowType::Dpss {
        return make_with_params(kind, n_total, params)
            .get(n)
            .copied()
            .unwrap_or(0.0);
    }

    let n = n as f64;
    let nn = n_total as f64;
    match kind {
        WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * n / nn).cos()),
        WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * n / nn).cos(),
        WindowType::Sine => (PI * (n + 0.5) / nn).sin(),
        WindowType::Blackman => {
            0.42 - 0.5 * (2.0 * PI * n / nn).cos() + 0.08 * (4.0 * PI * n / nn).cos()
        }
        WindowType::Kaiser => kaiser(n, nn, params.kaiser_beta),
        WindowType::FlatTop => flat_top(n, nn),
        WindowType::Dpss => unreachable!("DPSS handled before scalar evaluation"),
    }
}

/// Build a window of length `n_total`.
///
/// # Panics
///
/// For DPSS lengths of at least two, panics if the default bandwidth is
/// invalid for `n_total` or the bounded eigensolver does not converge.
pub fn make(kind: WindowType, n_total: usize) -> Vec<f64> {
    make_with_params(kind, n_total, &WindowParams::default())
}

/// Build with explicit parameters.
///
/// # Panics
///
/// For DPSS lengths of at least two, panics if its parameters are invalid or
/// the bounded eigensolver does not converge.  Use
/// [`make_with_params_checked`] when parameters come from an external source
/// or construction failure must be handled.
pub fn make_with_params(kind: WindowType, n_total: usize, params: &WindowParams) -> Vec<f64> {
    if kind == WindowType::Dpss {
        if n_total == 0 {
            return Vec::new();
        }
        if n_total == 1 {
            return vec![1.0];
        }
        // Preserve the established infallible API while making numerical
        // failure explicit instead of silently substituting another window.
        return make_with_params_checked(kind, n_total, params)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    (0..n_total)
        .map(|i| sample_with_params(kind, i, n_total, params))
        .collect()
}

/// Validate a periodic DPSS time-bandwidth product.
///
/// An empty window is handled as a harmless special case by
/// [`make_with_params_checked`].  For lengths of at least two,
/// SciPy-compatible periodic DPSS parameters must be finite and satisfy
/// `0 < NW < N / 2`.
pub fn validate_dpss_bandwidth(n_total: usize, bandwidth: f64) -> Result<(), WindowError> {
    let upper = n_total as f64 / 2.0;
    if bandwidth.is_finite() && bandwidth > 0.0 && bandwidth < upper {
        Ok(())
    } else {
        Err(WindowError::InvalidDpssBandwidth { n_total, bandwidth })
    }
}

/// Build a window while reporting invalid DPSS parameters or solver failure.
///
/// Non-DPSS windows are identical to [`make_with_params`].  A periodic DPSS
/// is obtained from the principal symmetric sequence of length `N + 1`, then
/// truncating its last sample, matching SciPy's `sym=False` convention.
/// Lengths zero and one conventionally return `[]` and `[1]`, respectively,
/// without consulting DPSS parameters.
pub fn make_with_params_checked(
    kind: WindowType,
    n_total: usize,
    params: &WindowParams,
) -> Result<Vec<f64>, WindowError> {
    if kind != WindowType::Dpss {
        return Ok(make_with_params(kind, n_total, params));
    }
    match n_total {
        0 => return Ok(Vec::new()),
        1 => return Ok(vec![1.0]),
        _ => {}
    }
    validate_dpss_bandwidth(n_total, params.dpss_bandwidth)?;
    dpss_periodic(n_total, params.dpss_bandwidth)
}

/// Kaiser-Bessel window (periodic form).
fn kaiser(n: f64, n_total: f64, beta: f64) -> f64 {
    let alpha = 0.5 * (n_total - 1.0);
    let x = (n - alpha) / alpha;
    bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / bessel_i0(beta)
}

/// Modified Bessel I0 for Kaiser window (simple series).
fn bessel_i0(x: f64) -> f64 {
    if x.abs() < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + 3.5156229 * t2
            + 3.0899424 * t2.powi(2)
            + 1.2067492 * t2.powi(3)
            + 0.2659732 * t2.powi(4)
            + 0.0360768 * t2.powi(5)
            + 0.0045813 * t2.powi(6)
    } else {
        let t = 3.75 / x.abs();
        let ax = x.abs().exp() / x.abs().sqrt();
        let c = 0.39894228 + 0.01328592 * t + 0.00225319 * t * t - 0.00157565 * t.powi(3)
            + 0.00916281 * t.powi(4)
            - 0.02057706 * t.powi(5)
            + 0.02635537 * t.powi(6)
            - 0.01647633 * t.powi(7)
            + 0.00392377 * t.powi(8);
        ax * c
    }
}

/// Flat-top window (Heinzel et al. 5-term, periodic, clamped ≥ 0 for STFT).
fn flat_top(n: f64, n_total: f64) -> f64 {
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;
    let phi = 2.0 * PI * n / n_total;
    let w = a0 - a1 * phi.cos() + a2 * (2.0 * phi).cos() - a3 * (3.0 * phi).cos()
        + a4 * (4.0 * phi).cos();
    w.max(0.0)
}

/// Principal periodic DPSS, peak-normalised to one.
fn dpss_periodic(n_total: usize, nw: f64) -> Result<Vec<f64>, WindowError> {
    let full_len = n_total
        .checked_add(1)
        .ok_or(WindowError::DpssSolverDidNotConverge {
            n_total,
            bandwidth: nw,
        })?;
    let (diagonal, off_diagonal) = dpss_even_tridiagonal(full_len, nw);
    let (eigenvalue_upper, bracket_width, matrix_norm) =
        largest_eigenvalue_upper(&diagonal, &off_diagonal);
    let reduced = principal_eigenvector(
        &diagonal,
        &off_diagonal,
        eigenvalue_upper,
        bracket_width,
        matrix_norm,
    )
    .ok_or(WindowError::DpssSolverDidNotConverge {
        n_total,
        bandwidth: nw,
    })?;

    let mut full = expand_even_eigenvector(&reduced, full_len);
    let peak = full.iter().copied().fold(0.0_f64, f64::max);
    if !peak.is_finite() || peak <= 0.0 {
        return Err(WindowError::DpssSolverDidNotConverge {
            n_total,
            bandwidth: nw,
        });
    }
    for value in &mut full {
        // Perron-Frobenius makes the principal vector strictly positive.  The
        // clamp only preserves that mathematical property if a tail rounds
        // all the way down to zero in binary64.
        *value = (*value / peak).max(f64::MIN_POSITIVE);
    }
    full.truncate(n_total);
    Ok(full)
}

/// Build the scaled tridiagonal commuting with the DPSS concentration
/// operator, projected into the even centrosymmetric half-space.
///
/// Scaling the full matrix by `4 / M²` keeps every entry O(1) without
/// changing its eigenvectors.  The projection is orthonormal: for even `M`
/// the central off-diagonal is added to the last diagonal entry; for odd `M`
/// the final off-diagonal is multiplied by sqrt(2).
fn dpss_even_tridiagonal(full_len: usize, nw: f64) -> (Vec<f64>, Vec<f64>) {
    let m = full_len as f64;
    let dimension = full_len.div_ceil(2);
    let scale = 4.0 / (m * m);
    let cosine = (2.0 * PI * nw / m).cos();
    let center = 0.5 * (m - 1.0);

    let mut diagonal = Vec::with_capacity(dimension);
    for i in 0..dimension {
        let offset = center - i as f64;
        diagonal.push(offset * offset * cosine * scale);
    }

    let mut off_diagonal = Vec::with_capacity(dimension.saturating_sub(1));
    for i in 0..dimension.saturating_sub(1) {
        let edge = (i + 1) as f64;
        off_diagonal.push(0.5 * edge * (m - edge) * scale);
    }

    if full_len.is_multiple_of(2) {
        let central_edge = dimension as f64;
        diagonal[dimension - 1] += 0.5 * central_edge * (m - central_edge) * scale;
    } else if dimension > 1 {
        off_diagonal[dimension - 2] *= 2.0_f64.sqrt();
    }

    (diagonal, off_diagonal)
}

/// Return a tight upper bracket for the largest eigenvalue, its final width,
/// and the matrix infinity norm.  Gershgorin bounds provide the initial
/// interval and Sturm counts isolate its final eigenvalue without allocating
/// eigenvectors.
fn largest_eigenvalue_upper(diagonal: &[f64], off_diagonal: &[f64]) -> (f64, f64, f64) {
    debug_assert!(!diagonal.is_empty());
    if diagonal.len() == 1 {
        let norm = diagonal[0].abs().max(1.0);
        let width = 16.0 * f64::EPSILON * norm;
        return (diagonal[0] + width, width, norm);
    }

    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    let mut matrix_norm = 0.0_f64;
    for i in 0..diagonal.len() {
        let mut radius = 0.0;
        if i > 0 {
            radius += off_diagonal[i - 1].abs();
        }
        if i < off_diagonal.len() {
            radius += off_diagonal[i].abs();
        }
        lower = lower.min(diagonal[i] - radius);
        upper = upper.max(diagonal[i] + radius);
        matrix_norm = matrix_norm.max(diagonal[i].abs() + radius);
    }
    matrix_norm = matrix_norm.max(1.0);
    let guard = 64.0 * f64::EPSILON * matrix_norm;
    lower -= guard;
    upper += guard;

    // Ninety-six steps are enough to reduce any finite binary64 Gershgorin
    // interval below one ulp.  Stop if the midpoint itself can no longer be
    // represented between the endpoints.
    for _ in 0..96 {
        let midpoint = lower + 0.5 * (upper - lower);
        if midpoint == lower || midpoint == upper {
            break;
        }
        if sturm_count_below(diagonal, off_diagonal, midpoint) < diagonal.len() {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    (upper, (upper - lower).max(0.0), matrix_norm)
}

/// Number of eigenvalues strictly below `x`, using the signed LDLᵀ pivots
/// of `T - xI`.
fn sturm_count_below(diagonal: &[f64], off_diagonal: &[f64], x: f64) -> usize {
    // This mirrors the limiting sign of a zero pivot in a Sturm sequence and
    // remains small enough that e²/pivot stays finite for the scaled matrix.
    const PIVOT_FLOOR: f64 = f64::MIN_POSITIVE * 16.0;

    let mut pivot = diagonal[0] - x;
    if pivot.abs() < PIVOT_FLOOR {
        pivot = -PIVOT_FLOOR;
    }
    let mut count = usize::from(pivot < 0.0);
    for i in 1..diagonal.len() {
        pivot = diagonal[i] - x - off_diagonal[i - 1].powi(2) / pivot;
        if pivot.abs() < PIVOT_FLOOR {
            pivot = -PIVOT_FLOOR;
        }
        count += usize::from(pivot < 0.0);
    }
    count
}

/// Principal eigenvector from bounded shifted inverse iteration.
///
/// Every shift is above the bisection bracket, so `shift * I - T` is SPD and
/// admits an LDLᵀ factorisation without pivoting.  Starting from a positive
/// vector preserves positivity because this matrix is an irreducible
/// M-matrix.  We require at least three solves and check both iterate change
/// and the eigen-residual before accepting the result.  A larger shift is
/// tried only when factorisation proves the current shift is not numerically
/// SPD; once factorisation succeeds, increasing it would only slow inverse
/// iteration toward a clustered principal eigenvalue.
fn principal_eigenvector(
    diagonal: &[f64],
    off_diagonal: &[f64],
    eigenvalue_upper: f64,
    bracket_width: f64,
    matrix_norm: f64,
) -> Option<Vec<f64>> {
    const MIN_ITERATIONS: usize = 3;
    const MAX_ITERATIONS: usize = 96;
    const FACTOR_RETRIES: usize = 5;
    const RESIDUAL_TOLERANCE: f64 = 2.5e-13;
    const ITERATE_TOLERANCE: f64 = 2.5e-13;

    let dimension = diagonal.len();
    let mut padding = (64.0 * f64::EPSILON * matrix_norm).max(2.0 * bracket_width);
    let mut factor = None;
    for _ in 0..FACTOR_RETRIES {
        let shift = eigenvalue_upper + padding;
        match factor_shifted_ldlt(diagonal, off_diagonal, shift) {
            Some((pivots, lower)) => {
                factor = Some((pivots, lower));
                break;
            }
            None => padding *= 8.0,
        }
    }
    let (pivots, lower) = factor?;

    let mut current = vec![1.0; dimension];
    normalize_l2(&mut current)?;
    let mut next = vec![0.0; dimension];
    let mut work = vec![0.0; dimension];
    let mut product = vec![0.0; dimension];

    for iteration in 0..MAX_ITERATIONS {
        solve_shifted_ldlt(&pivots, &lower, &current, &mut work, &mut next);
        normalize_l2(&mut next)?;
        if dot(&current, &next) < 0.0 {
            for value in &mut next {
                *value = -*value;
            }
        }

        let iterate_change = l2_difference(&current, &next);
        tridiagonal_product(diagonal, off_diagonal, &next, &mut product);
        let rayleigh = dot(&next, &product);
        for (value, &vector_value) in product.iter_mut().zip(&next) {
            *value -= rayleigh * vector_value;
        }
        let residual = l2_norm(&product);
        std::mem::swap(&mut current, &mut next);

        if iteration + 1 >= MIN_ITERATIONS
            && iterate_change <= ITERATE_TOLERANCE
            && residual <= RESIDUAL_TOLERANCE * matrix_norm
        {
            if current
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            {
                return Some(current);
            }
            break;
        }
    }
    None
}

/// LDLᵀ factorisation of `shift * I - T`.
fn factor_shifted_ldlt(
    diagonal: &[f64],
    off_diagonal: &[f64],
    shift: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut pivots = vec![0.0; diagonal.len()];
    let mut lower = vec![0.0; off_diagonal.len()];
    pivots[0] = shift - diagonal[0];
    if !pivots[0].is_finite() || pivots[0] <= 0.0 {
        return None;
    }
    for i in 1..diagonal.len() {
        lower[i - 1] = -off_diagonal[i - 1] / pivots[i - 1];
        pivots[i] = shift - diagonal[i] - off_diagonal[i - 1].powi(2) / pivots[i - 1];
        if !lower[i - 1].is_finite() || !pivots[i].is_finite() || pivots[i] <= 0.0 {
            return None;
        }
    }
    Some((pivots, lower))
}

fn solve_shifted_ldlt(
    pivots: &[f64],
    lower: &[f64],
    right_hand_side: &[f64],
    work: &mut [f64],
    solution: &mut [f64],
) {
    work[0] = right_hand_side[0];
    for i in 1..pivots.len() {
        work[i] = right_hand_side[i] - lower[i - 1] * work[i - 1];
    }
    for (value, &pivot) in work.iter_mut().zip(pivots) {
        *value /= pivot;
    }
    let last = pivots.len() - 1;
    solution[last] = work[last];
    for i in (0..last).rev() {
        solution[i] = work[i] - lower[i] * solution[i + 1];
    }
}

fn tridiagonal_product(
    diagonal: &[f64],
    off_diagonal: &[f64],
    vector: &[f64],
    product: &mut [f64],
) {
    for i in 0..diagonal.len() {
        let mut value = diagonal[i] * vector[i];
        if i > 0 {
            value += off_diagonal[i - 1] * vector[i - 1];
        }
        if i < off_diagonal.len() {
            value += off_diagonal[i] * vector[i + 1];
        }
        product[i] = value;
    }
}

fn expand_even_eigenvector(reduced: &[f64], full_len: usize) -> Vec<f64> {
    let mut full = vec![0.0; full_len];
    let paired = full_len / 2;
    let pair_scale = 1.0 / 2.0_f64.sqrt();
    for i in 0..paired {
        let value = reduced[i] * pair_scale;
        full[i] = value;
        full[full_len - 1 - i] = value;
    }
    if !full_len.is_multiple_of(2) {
        full[paired] = reduced[paired];
    }
    full
}

fn normalize_l2(vector: &mut [f64]) -> Option<()> {
    let scale = vector
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    for value in vector.iter_mut() {
        *value /= scale;
    }
    let norm = l2_norm(vector);
    if !norm.is_finite() || norm == 0.0 {
        return None;
    }
    for value in vector {
        *value /= norm;
    }
    Some(())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    compensated_sum(left.iter().zip(right).map(|(a, b)| a * b))
}

fn l2_norm(vector: &[f64]) -> f64 {
    dot(vector, vector).sqrt()
}

fn l2_difference(left: &[f64], right: &[f64]) -> f64 {
    compensated_sum(left.iter().zip(right).map(|(a, b)| (a - b).powi(2))).sqrt()
}

/// Neumaier-compensated accumulation.  Unlike plain summation this remains
/// reliable for the Rayleigh quotient and convergence tests at large window
/// lengths, including cancellation in residual-related dot products.
fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        let next = sum + value;
        if sum.abs() >= value.abs() {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }
        sum = next;
    }
    sum + correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_cola_at_50pct() {
        let n = 1024;
        let w = make(WindowType::Hann, n);
        let hop = n / 2;
        let len = n + hop * 3;
        let mut cola = vec![0.0; len];
        let mut start = 0;
        while start + n <= len {
            for i in 0..n {
                cola[start + i] += w[i];
            }
            start += hop;
        }
        for &c in cola.iter().take(len - n).skip(n) {
            assert!((c - 1.0).abs() < 1e-12, "cola={c}");
        }
    }

    #[test]
    fn advanced_windows_bounded() {
        let n = 512;
        for kind in [WindowType::Kaiser, WindowType::FlatTop, WindowType::Dpss] {
            let w = make(kind, n);
            for &v in &w {
                assert!(v.is_finite());
                // Flat-top has tiny negative sidelobes at edges; clamp check.
                assert!((-0.01..=1.05).contains(&v), "{kind:?} value {v}");
            }
        }
    }

    #[test]
    fn dpss_periodic_matches_reference_values() {
        let params = WindowParams {
            dpss_bandwidth: 2.0,
            ..WindowParams::default()
        };
        let actual = make_with_params_checked(WindowType::Dpss, 8, &params).unwrap();
        // Generated with SciPy v1.18.0, peeled commit
        // 54ef5423f2e4376230ec3bfda6912a07a50958e3, using
        // `dpss(8, 2, Kmax=1, sym=False, norm=2)` followed by peak
        // normalisation.  Only these numeric values are retained here.
        let expected = [
            0.07409072028476879,
            0.27352337476511301,
            0.58217417973918195,
            0.87726572978085537,
            1.0,
            0.87726572978085537,
            0.58217417973918195,
            0.27352337476511301,
        ];
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-14,
                "index {index}: actual={actual:.17e}, expected={expected:.17e}"
            );
        }
    }

    fn concentration_ratio(window: &[f64], half_bandwidth: f64) -> f64 {
        let energy = dot(window, window);
        let mut concentrated = 2.0 * half_bandwidth * energy;
        for i in 0..window.len() {
            for j in i + 1..window.len() {
                let distance = (j - i) as f64;
                let kernel = (2.0 * PI * half_bandwidth * distance).sin() / (PI * distance);
                concentrated += 2.0 * window[i] * window[j] * kernel;
            }
        }
        concentrated / energy
    }

    #[test]
    fn dpss_periodic_concentration_matches_reference() {
        let params = WindowParams::default();
        let window = make_with_params_checked(WindowType::Dpss, 64, &params).unwrap();
        let mut full = window.clone();
        full.push(window[0]);
        let ratio = concentration_ratio(&full, 3.0 / 65.0);
        assert!((ratio - 0.9999998728070119).abs() <= 5.0e-10);
    }

    #[test]
    fn non_dpss_window_golden_values() {
        let params = WindowParams::default();
        let golden = [
            (WindowType::Hann, 4_605_863_345_008_909_798),
            (WindowType::Hamming, 4_605_968_870_912_198_407),
            (WindowType::Sine, 4_607_009_347_991_985_328),
            (WindowType::Blackman, 4_605_142_769_068_530_518),
            (WindowType::Kaiser, 4_606_517_306_813_646_917),
            (WindowType::FlatTop, 4_601_672_451_236_213_111),
        ];
        for (kind, expected_bits) in golden {
            let actual = sample_with_params(kind, 3, 8, &params);
            let expected = f64::from_bits(expected_bits);
            assert!(
                (actual - expected).abs() <= 1.0e-13,
                "{kind:?} changed: actual={actual:.17e}, expected={expected:.17e}"
            );
        }
    }

    #[test]
    fn dpss_shapes_are_positive_symmetric_and_peak_normalised() {
        for bandwidth in [2.5, 3.0, 4.0] {
            let params = WindowParams {
                dpss_bandwidth: bandwidth,
                ..WindowParams::default()
            };
            let window = make_with_params_checked(WindowType::Dpss, 256, &params).unwrap();
            assert!(window.iter().all(|value| value.is_finite() && *value > 0.0));
            assert_eq!(window[128], 1.0);
            for k in 1..256 {
                assert_eq!(
                    window[k].to_bits(),
                    window[256 - k].to_bits(),
                    "NW={bandwidth}, k={k}"
                );
            }
            assert!(
                window[..=128].windows(2).all(|pair| pair[0] < pair[1]),
                "NW={bandwidth} does not rise strictly to its principal peak"
            );
        }

        // Odd output N exercises the even full length M=N+1, whose
        // half-space reduction absorbs the central edge into its diagonal.
        let odd =
            make_with_params_checked(WindowType::Dpss, 255, &WindowParams::default()).unwrap();
        assert_eq!(odd[127], 1.0);
        assert_eq!(odd[128], 1.0);
        for k in 1..255 {
            assert_eq!(odd[k].to_bits(), odd[255 - k].to_bits(), "odd N, k={k}");
        }
        assert!(full_scaled_residual_inf(&odd, 3.0) <= 5.0e-13);
    }

    #[test]
    fn dpss_checked_validation_and_small_lengths() {
        assert_eq!(MAX_DENOISER_DPSS_NW, 8.0);
        for bandwidth in [
            f64::NEG_INFINITY,
            -1.0,
            0.0,
            32.0,
            33.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert!(matches!(
                validate_dpss_bandwidth(64, bandwidth),
                Err(WindowError::InvalidDpssBandwidth { .. })
            ));
            let params = WindowParams {
                dpss_bandwidth: bandwidth,
                ..WindowParams::default()
            };
            assert!(matches!(
                make_with_params_checked(WindowType::Dpss, 64, &params),
                Err(WindowError::InvalidDpssBandwidth { .. })
            ));
        }
        assert!(validate_dpss_bandwidth(64, 31.999).is_ok());
        for (n_total, bandwidth) in [
            (2, 0.001),
            (2, 0.999),
            (3, 1.499),
            (64, 0.001),
            (64, 31.999),
        ] {
            let params = WindowParams {
                dpss_bandwidth: bandwidth,
                ..WindowParams::default()
            };
            let window = make_with_params_checked(WindowType::Dpss, n_total, &params).unwrap();
            assert_eq!(window.len(), n_total);
            assert!(window.iter().all(|value| value.is_finite() && *value > 0.0));
            assert_eq!(window.iter().copied().fold(0.0_f64, f64::max), 1.0);
        }

        let invalid = WindowParams {
            dpss_bandwidth: f64::NAN,
            ..WindowParams::default()
        };
        assert!(make_with_params_checked(WindowType::Dpss, 0, &invalid)
            .unwrap()
            .is_empty());
        assert_eq!(
            make_with_params_checked(WindowType::Dpss, 1, &invalid).unwrap(),
            [1.0]
        );
        assert!(make(WindowType::Dpss, 0).is_empty());
        assert_eq!(make(WindowType::Dpss, 1), [1.0]);
    }

    #[test]
    fn dpss_sample_matches_built_window() {
        let params = WindowParams {
            dpss_bandwidth: 2.5,
            ..WindowParams::default()
        };
        let window = make_with_params_checked(WindowType::Dpss, 16, &params).unwrap();
        for (index, expected) in window.iter().copied().enumerate() {
            assert_eq!(
                sample_with_params(WindowType::Dpss, index, 16, &params).to_bits(),
                expected.to_bits()
            );
        }
        assert_eq!(sample_with_params(WindowType::Dpss, 16, 16, &params), 0.0);
    }

    fn full_scaled_residual_inf(periodic: &[f64], bandwidth: f64) -> f64 {
        let full_len = periodic.len() + 1;
        let m = full_len as f64;
        let scale = 4.0 / (m * m);
        let cosine = (2.0 * PI * bandwidth / m).cos();
        let center = 0.5 * (m - 1.0);
        let mut vector = periodic.to_vec();
        vector.push(periodic[0]);
        let mut product = vec![0.0; full_len];
        let mut matrix_norm = 0.0_f64;

        for i in 0..full_len {
            let offset = center - i as f64;
            let diagonal = offset * offset * cosine * scale;
            let left = if i > 0 {
                0.5 * i as f64 * (m - i as f64) * scale
            } else {
                0.0
            };
            let right = if i + 1 < full_len {
                let edge = (i + 1) as f64;
                0.5 * edge * (m - edge) * scale
            } else {
                0.0
            };
            product[i] = diagonal * vector[i];
            if i > 0 {
                product[i] += left * vector[i - 1];
            }
            if i + 1 < full_len {
                product[i] += right * vector[i + 1];
            }
            matrix_norm = matrix_norm.max(diagonal.abs() + left + right);
        }

        let rayleigh = dot(&vector, &product) / dot(&vector, &vector);
        let residual_inf = product
            .iter()
            .zip(&vector)
            .map(|(product, vector)| (product - rayleigh * vector).abs())
            .fold(0.0_f64, f64::max);
        let vector_inf = vector.iter().map(|value| value.abs()).fold(0.0, f64::max);
        residual_inf / (matrix_norm * vector_inf)
    }

    #[test]
    fn dpss_large_full_eigenproblem_residual() {
        // N=255 exercises the even internal M=N+1 reduction. N=262144 is a
        // regression for clustered eigenvalues: successful checked
        // construction proves both residual and iterate-change convergence.
        for n_total in [255, 4096, 8192, 65_536, 262_144] {
            let params = WindowParams::default();
            let window = make_with_params_checked(WindowType::Dpss, n_total, &params).unwrap();
            if n_total == 262_144 {
                assert!(window.iter().all(|value| value.is_finite() && *value > 0.0));
                assert_eq!(window[n_total / 2], 1.0);
                assert!(window[..=n_total / 2]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]));
                for k in 1..n_total {
                    assert_eq!(window[k].to_bits(), window[n_total - k].to_bits());
                }
            }
            let residual = full_scaled_residual_inf(&window, params.dpss_bandwidth);
            assert!(
                residual <= 5.0e-13,
                "N={n_total}, scaled full residual={residual:.17e}"
            );
        }
    }
}
