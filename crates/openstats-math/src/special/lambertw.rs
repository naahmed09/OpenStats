//! Lambert W function: the inverse of w ↦ w e^w.
//!
//! * [`lambert_w0`] — principal branch on x ≥ −1/e, W₀(x) ≥ −1.
//! * [`lambert_wm1`] — lower real branch on −1/e ≤ x < 0, W₋₁(x) ≤ −1.
//!
//! Initial estimates come from the branch-point series in
//! p = √(2(e·x + 1)) near −1/e, from the logarithmic asymptotics for large |ln x|
//! and from low-order Taylor expansions near 0; Halley iteration then
//! converges to full precision in a few steps.

use std::f64::consts::E;

const NEG_INV_E: f64 = -0.367_879_441_171_442_33;
/// −1/e − NEG_INV_E (low word of the double-double −1/e).
const NEG_INV_E_LO: f64 = 1.242_875_367_278_836_3e-17;

/// p = √(2(ex+1)) with ex+1 computed without cancellation error.
fn branch_p(x: f64) -> f64 {
    let d = (x - NEG_INV_E) - NEG_INV_E_LO;
    (2.0 * E * d).max(0.0).sqrt()
}

/// Branch-point series coefficients: W = −1 + p − p²/3 + 11p³/72 − 43p⁴/540 + 769p⁵/17280 − …
fn branch_series(p: f64) -> f64 {
    -1.0 + p
        * (1.0
            + p * (-1.0 / 3.0
                + p * (11.0 / 72.0 + p * (-43.0 / 540.0 + p * (769.0 / 17280.0 + p * (-221.0 / 8505.0))))))
}

fn halley(mut w: f64, x: f64) -> f64 {
    for _ in 0..40 {
        let ew = w.exp();
        let f = w * ew - x;
        let wp1 = w + 1.0;
        if wp1 == 0.0 {
            break;
        }
        let step = f / (ew * wp1 - (w + 2.0) * f / (2.0 * wp1));
        let nw = w - step;
        if !nw.is_finite() {
            break;
        }
        if (nw - w).abs() <= 4.0 * f64::EPSILON * nw.abs() {
            return nw;
        }
        w = nw;
    }
    w
}

/// Principal branch W₀(x); NaN for x < −1/e.
pub fn lambert_w0(x: f64) -> f64 {
    if x.is_nan() || x < NEG_INV_E {
        return f64::NAN;
    }
    if x == 0.0 || x == f64::INFINITY {
        return x;
    }
    let guess = if x < -0.25 {
        let p = branch_p(x);
        if p < 1e-3 {
            return branch_series(p);
        }
        branch_series(p)
    } else if x.abs() < 0.3 {
        x * (1.0 - x * (1.0 - 1.5 * x))
    } else if x < 3.0 {
        let l = x.ln_1p();
        l * (1.0 - (1.0 + l).ln() / (2.0 + l))
    } else {
        let l1 = x.ln();
        let l2 = l1.ln();
        l1 - l2 + l2 / l1
    };
    halley(guess, x)
}

/// Lower branch W₋₁(x) for −1/e ≤ x < 0 (−∞ at 0⁻); NaN elsewhere.
pub fn lambert_wm1(x: f64) -> f64 {
    if x.is_nan() || !(NEG_INV_E..=0.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    let guess = if x < -0.25 {
        let p = branch_p(x);
        if p < 1e-3 {
            return branch_series(-p);
        }
        branch_series(-p)
    } else {
        let l1 = (-x).ln();
        let l2 = (-l1).ln();
        l1 - l2 + l2 / l1
    };
    halley(guess, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(lambert_w0(0.0), 0.0);
        assert!((lambert_w0(1.0) - 0.567_143_290_409_783_8).abs() < 1e-16);
        assert!((lambert_w0(E) - 1.0).abs() < 1e-15);
        assert!((lambert_w0(-1.0 / E) + 1.0).abs() < 1e-8);
        assert!((lambert_wm1(-2.0 * (-2f64).exp()) + 2.0).abs() < 1e-14);
        assert!((lambert_w0(-0.2) + 0.259_171_101_819_073_4).abs() < 1e-15);
        assert!((lambert_wm1(-0.2) + 2.542_641_357_773_526_5).abs() < 1e-14);
        assert!(lambert_w0(-0.4).is_nan());
        assert!(lambert_wm1(0.1).is_nan());
        assert_eq!(lambert_wm1(0.0), f64::NEG_INFINITY);
        assert_eq!(lambert_w0(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn inverse_property() {
        let xs =
            [-0.36, -0.3, -0.2, -0.05, 1e-300, 1e-10, 1e-3, 0.1, 0.5, 1.0, 2.9, 3.1, 10.0, 100.0, 1e5, 1e100, 1e300];
        for &x in &xs {
            let w = lambert_w0(x);
            let back = w * w.exp();
            assert!((back - x).abs() <= 4e-16 * (1.0 + w.abs()) * x.abs(), "W0({x}) = {w}: back {back}");
        }
        for &x in &[-0.367, -0.3, -0.2, -0.1, -1e-3, -1e-10, -1e-100, -1e-300] {
            let w = lambert_wm1(x);
            assert!(w <= -1.0);
            let back = w * w.exp();
            assert!((back - x).abs() <= 4e-16 * (1.0 + w.abs()) * x.abs(), "W-1({x}) = {w}: back {back}");
        }
    }

    #[test]
    fn branch_point_behaviour() {
        // Near −1/e: W ≈ −1 ± √(2(1+ex)); the two branches are symmetric to leading order.
        let x = NEG_INV_E + 1e-10;
        let (a, b) = (lambert_w0(x), lambert_wm1(x));
        assert!(a > -1.0 && b < -1.0);
        let p2 = 2.0 * (E * x + 1.0);
        assert!(((a + 1.0) + (b + 1.0) + 2.0 * p2 / 3.0).abs() < 2e-15);
        assert_eq!(lambert_w0(NEG_INV_E), -1.0);
        assert_eq!(lambert_wm1(NEG_INV_E), -1.0);
    }
}
