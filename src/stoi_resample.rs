//! Resampling compatible with the Octave-style filter used by `pystoi`.
//!
//! STOI's reference Python implementation does not use SciPy's default
//! `resample_poly` window.  It builds a longer Kaiser-windowed sinc filter in
//! `utils._resample_window_oct`, normalizes it, and then lets
//! `scipy.signal.resample_poly` apply its zero-phase padding.  Keeping that
//! process here avoids small, but measurable, differences in STOI fixtures.

const REJECTION_DB: f64 = 60.0;
const KAISER_BETA: f64 = 0.1102 * (REJECTION_DB - 8.7);

// Enough for every conventional audio-rate conversion by a wide margin.  A
// bound is necessary because rates are caller-provided u32 values and filter
// length grows linearly with the reduced rate ratio.
const MAX_FILTER_TAPS: usize = 10_000_001;

/// Resample with the filter and alignment used by `pystoi` 0.4.1.
///
/// The result has `ceil(input.len() * to_rate / from_rate)` samples, matching
/// `scipy.signal.resample_poly`. Invalid rates and impractically large reduced
/// ratios return `None` rather than overflowing or attempting an unbounded
/// allocation.
pub(crate) fn resample(input: &[f64], from_rate: u32, to_rate: u32) -> Option<Vec<f64>> {
    if from_rate == 0 || to_rate == 0 {
        return None;
    }
    if from_rate == to_rate {
        return Some(input.to_vec());
    }
    if input.is_empty() {
        return Some(Vec::new());
    }

    let divisor = gcd(from_rate, to_rate);
    let up = usize::try_from(to_rate / divisor).ok()?;
    let down = usize::try_from(from_rate / divisor).ok()?;
    let mut filter = octave_window(up, down)?;
    let half_len = (filter.len() - 1) / 2;
    // resample_poly scales a caller-supplied window before filtering. Doing
    // this as a separate pass preserves the same floating-point operations.
    for coefficient in &mut filter {
        *coefficient *= up as f64;
    }

    let scaled_len = input.len().checked_mul(up)?;
    let output_len = scaled_len / down + usize::from(scaled_len % down != 0);
    let mut output = Vec::new();
    output.try_reserve_exact(output_len).ok()?;

    // This is deliberately `down`, rather than zero, when half_len is an
    // exact multiple. It is the centering rule used by resample_poly.
    let pre_pad = down.checked_sub(half_len % down)?;
    let pre_remove = half_len.checked_add(pre_pad)?.checked_div(down)?;
    let last_filter_index = pre_pad.checked_add(filter.len() - 1)?;

    // Evaluate only the requested, centered portion of upfirdn's output. For
    // a raw output sample j, coefficient index is j*down-k*up. Iterating k in
    // ascending order also matches SciPy's floating-point accumulation order.
    for output_index in 0..output_len {
        let raw_index = pre_remove.checked_add(output_index)?;
        let time = raw_index.checked_mul(down)?;

        if time < pre_pad {
            output.push(0.0);
            continue;
        }

        let first_input = if time > last_filter_index {
            ceil_div(time - last_filter_index, up)?
        } else {
            0
        };
        let last_input = ((time - pre_pad) / up).min(input.len() - 1);

        let mut value = 0.0;
        if first_input <= last_input {
            for input_index in first_input..=last_input {
                let filter_index = time - input_index * up - pre_pad;
                value += input[input_index] * filter[filter_index];
            }
        }
        output.push(value);
    }

    Some(output)
}

fn octave_window(up: usize, down: usize) -> Option<Vec<f64>> {
    let max_rate = up.max(down) as f64;
    let stopband_cutoff = 1.0 / (2.0 * max_rate);
    let roll_off_width = stopband_cutoff / 10.0;
    let half_len_f64 = ((REJECTION_DB - 8.0) / (28.714 * roll_off_width)).ceil();
    if !half_len_f64.is_finite() || half_len_f64 < 1.0 || half_len_f64 > usize::MAX as f64 {
        return None;
    }
    let half_len = half_len_f64 as usize;
    let filter_len = half_len.checked_mul(2)?.checked_add(1)?;
    if filter_len > MAX_FILTER_TAPS {
        return None;
    }

    let mut window = Vec::new();
    window.try_reserve_exact(filter_len).ok()?;
    let beta_denominator = modified_bessel_i0(KAISER_BETA);
    let ideal_scale = 2.0 * up as f64 * stopband_cutoff;

    for index in 0..filter_len {
        let t = index as f64 - half_len as f64;
        let sinc_argument = 2.0 * stopband_cutoff * t;
        let sinc = if sinc_argument == 0.0 {
            1.0
        } else {
            let angle = std::f64::consts::PI * sinc_argument;
            angle.sin() / angle
        };
        let relative_position = t / half_len as f64;
        let kaiser_argument = KAISER_BETA * (1.0 - relative_position.powi(2)).max(0.0).sqrt();
        let kaiser = modified_bessel_i0(kaiser_argument) / beta_denominator;
        window.push(kaiser * ideal_scale * sinc);
    }

    let sum = window.iter().sum::<f64>();
    if !sum.is_finite() || sum == 0.0 {
        return None;
    }
    for coefficient in &mut window {
        *coefficient /= sum;
    }
    Some(window)
}

fn modified_bessel_i0(value: f64) -> f64 {
    // I0(x) = sum((x^2 / 4)^k / (k!)^2). The STOI Kaiser beta is only
    // 5.65326, for which this converges to f64 precision in a few terms.
    let squared_quarter = value * value * 0.25;
    let mut sum = 1.0;
    let mut term = 1.0;
    for order in 1..=64 {
        let order = order as f64;
        term *= squared_quarter / (order * order);
        let next = sum + term;
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

fn ceil_div(numerator: usize, denominator: usize) -> Option<usize> {
    let quotient = numerator.checked_div(denominator)?;
    quotient.checked_add(usize::from(numerator % denominator != 0))
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::resample;

    const FIXTURE: [f64; 16] = [
        0.25, -0.5, 0.75, -1.0, 0.5, 0.125, -0.25, 0.0, 0.375, -0.625, 0.875, -0.125, 0.25, -0.75,
        0.5, 0.1,
    ];

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 1.0e-15,
                "sample {index}: expected {expected:.17e}, got {actual:.17e}, error {error:.3e}"
            );
        }
    }

    #[test]
    fn equal_rates_clone_exactly() {
        let input = [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        let output = resample(&input, 10_000, 10_000).expect("valid rates");
        assert_eq!(output.len(), input.len());
        for (&actual, &expected) in output.iter().zip(&input) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn zero_rate_is_rejected() {
        assert_eq!(resample(&FIXTURE, 0, 10_000), None);
        assert_eq!(resample(&FIXTURE, 16_000, 0), None);
        assert_eq!(resample(&FIXTURE, 0, 0), None);
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(resample(&[], 16_000, 10_000), Some(Vec::new()));
    }

    #[test]
    fn output_length_is_ceil_of_rate_ratio() {
        for input_len in 1..=33 {
            let input = vec![0.0; input_len];
            for &(from_rate, to_rate) in &[(16_000, 10_000), (48_000, 10_000), (10_000, 44_100)] {
                let expected = (input_len * to_rate as usize).div_ceil(from_rate as usize);
                assert_eq!(
                    resample(&input, from_rate, to_rate)
                        .expect("valid conversion")
                        .len(),
                    expected
                );
            }
        }
    }

    #[test]
    fn matches_pystoi_0_4_1_from_16_khz() {
        // Generated by pystoi.utils.resample_oct with SciPy's resample_poly.
        let expected = [
            0.04027566031699189,
            -0.02047008354880606,
            -0.14447095456229669,
            0.13196267009841994,
            -0.06199107929389283,
            0.01957429945466374,
            0.11501789453329479,
            0.30188943296067816,
            -0.39557252770425644,
            0.3120579915145172,
        ];
        assert_close(
            &resample(&FIXTURE, 16_000, 10_000).expect("valid conversion"),
            &expected,
        );
    }

    #[test]
    fn matches_pystoi_0_4_1_from_48_khz() {
        // This ratio exercises a different filter phase and ceil output length.
        let expected = [
            -0.005164767708441263,
            -0.01382896956013285,
            0.07671091822375789,
            0.025623339367896514,
        ];
        assert_close(
            &resample(&FIXTURE, 48_000, 10_000).expect("valid conversion"),
            &expected,
        );
    }
}
