//! Line searches along a descent direction.

/// Outcome of a line search on φ(α) = f(x + α·d).
#[derive(Debug, Clone, Copy)]
pub struct LineSearch {
    pub alpha: f64,
    /// φ(α).
    pub value: f64,
    /// φ′(α) = ∇f(x + αd)·d when the search used derivatives, else NaN.
    pub slope: f64,
    pub evals: usize,
}

/// Armijo backtracking: shrink α by `rho` until
/// φ(α) ≤ φ(0) + c₁·α·φ′(0). Gives up after `max_evals` trials.
pub fn backtracking(
    mut phi: impl FnMut(f64) -> f64,
    phi0: f64,
    dphi0: f64,
    alpha0: f64,
    c1: f64,
    rho: f64,
    max_evals: usize,
) -> Option<LineSearch> {
    let mut alpha = alpha0;
    for k in 1..=max_evals {
        let v = phi(alpha);
        if v.is_finite() && v <= phi0 + c1 * alpha * dphi0 {
            return Some(LineSearch { alpha, value: v, slope: f64::NAN, evals: k });
        }
        alpha *= rho;
    }
    None
}

/// Minimiser of the cubic through (a, fa, ga) and (b, fb, gb), restricted to
/// the interval; `None` when it is not real or falls outside.
fn cubic_min(a: f64, fa: f64, ga: f64, b: f64, fb: f64, gb: f64) -> Option<f64> {
    let d1 = ga + gb - 3.0 * (fa - fb) / (a - b);
    let disc = d1 * d1 - ga * gb;
    if disc < 0.0 {
        return None;
    }
    let d2 = (if b > a { 1.0 } else { -1.0 }) * disc.sqrt();
    let t = b - (b - a) * (gb + d2 - d1) / (gb - ga + 2.0 * d2);
    if t.is_finite() {
        Some(t)
    } else {
        None
    }
}

/// Line search satisfying the strong Wolfe conditions
/// φ(α) ≤ φ(0) + c₁αφ′(0) and |φ′(α)| ≤ c₂|φ′(0)| (Nocedal & Wright,
/// Algorithms 3.5 and 3.6, with cubic interpolation in the zoom phase).
/// `phi(α)` returns `(φ(α), φ′(α))`. Requires φ′(0) < 0 and 0 < c₁ < c₂ < 1.
pub fn strong_wolfe(
    mut phi: impl FnMut(f64) -> (f64, f64),
    phi0: f64,
    dphi0: f64,
    alpha0: f64,
    c1: f64,
    c2: f64,
    max_evals: usize,
) -> Option<LineSearch> {
    if dphi0.is_nan() || dphi0 >= 0.0 || alpha0 <= 0.0 {
        return None;
    }
    let alpha_max = 1e100;
    let mut evals = 0usize;
    let (mut a_prev, mut f_prev, mut g_prev) = (0.0, phi0, dphi0);
    let mut a = alpha0;
    for i in 0..max_evals {
        let (fa, ga) = phi(a);
        evals += 1;
        if !fa.is_finite() {
            // Overshot into a region where f is undefined: shrink toward a_prev.
            a = 0.5 * (a_prev + a);
            if (a - a_prev).abs() < 1e-300 {
                return None;
            }
            continue;
        }
        if fa > phi0 + c1 * a * dphi0 || (i > 0 && fa >= f_prev) {
            return zoom(&mut phi, phi0, dphi0, c1, c2, (a_prev, f_prev, g_prev), (a, fa, ga), &mut evals, max_evals);
        }
        if ga.abs() <= -c2 * dphi0 {
            return Some(LineSearch { alpha: a, value: fa, slope: ga, evals });
        }
        if ga >= 0.0 {
            return zoom(&mut phi, phi0, dphi0, c1, c2, (a, fa, ga), (a_prev, f_prev, g_prev), &mut evals, max_evals);
        }
        a_prev = a;
        f_prev = fa;
        g_prev = ga;
        a = (2.0 * a).min(alpha_max);
    }
    None
}

type Pt = (f64, f64, f64);

#[allow(clippy::too_many_arguments)]
fn zoom(
    phi: &mut impl FnMut(f64) -> (f64, f64),
    phi0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
    mut lo: Pt,
    mut hi: Pt,
    evals: &mut usize,
    max_evals: usize,
) -> Option<LineSearch> {
    // Invariant: lo satisfies the sufficient decrease condition with the
    // smallest value so far; the minimiser of φ lies between lo and hi.
    while *evals < max_evals + 40 {
        let (width_lo, width_hi) = (lo.0.min(hi.0), lo.0.max(hi.0));
        let width = width_hi - width_lo;
        if width <= 1e-16 * width_hi.abs().max(1e-300) {
            return None;
        }
        let mut a = cubic_min(lo.0, lo.1, lo.2, hi.0, hi.1, hi.2).unwrap_or(0.5 * (lo.0 + hi.0));
        // Safeguard: stay at least 10% inside the interval.
        let (l, u) = (width_lo + 0.1 * width, width_hi - 0.1 * width);
        if !(a >= l && a <= u) {
            a = 0.5 * (lo.0 + hi.0);
        }
        let (fa, ga) = phi(a);
        *evals += 1;
        if !fa.is_finite() {
            hi = (a, f64::INFINITY, 0.0);
            continue;
        }
        if fa > phi0 + c1 * a * dphi0 || fa >= lo.1 {
            hi = (a, fa, ga);
        } else {
            if ga.abs() <= -c2 * dphi0 {
                return Some(LineSearch { alpha: a, value: fa, slope: ga, evals: *evals });
            }
            if ga * (hi.0 - lo.0) >= 0.0 {
                hi = lo;
            }
            lo = (a, fa, ga);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strong_wolfe_on_quartic_and_quadratic() {
        let c1 = 1e-4;
        let c2 = 0.9;
        // φ(α) = (α − 3)² − 9 ⇒ φ′(0) = −6.
        let q = |a: f64| ((a - 3.0) * (a - 3.0) - 9.0, 2.0 * (a - 3.0));
        for a0 in [1.0, 0.001, 50.0] {
            let r = strong_wolfe(q, 0.0, -6.0, a0, c1, c2, 50).expect("quadratic");
            assert!(r.value <= c1 * r.alpha * -6.0 && r.slope.abs() <= c2 * 6.0, "α0 = {a0}: {r:?}");
        }
        // Steep quartic with a short minimiser; first trial overshoots wildly.
        let f4 = |a: f64| ((a - 0.01).powi(4) * 1e6 - 0.5 * a, 4e6 * (a - 0.01).powi(3) - 0.5);
        let (f0, g0) = f4(0.0);
        let r = strong_wolfe(f4, f0, g0, 10.0, c1, 0.1, 80).expect("quartic");
        assert!(r.value <= f0 + c1 * r.alpha * g0 && r.slope.abs() <= 0.1 * g0.abs());
        // Not a descent direction.
        assert!(strong_wolfe(q, 0.0, 1.0, 1.0, c1, c2, 10).is_none());
    }

    #[test]
    fn backtracking_armijo() {
        let r = backtracking(|a| (a - 0.2) * (a - 0.2) - 0.04, 0.0, -0.4, 4.0, 1e-4, 0.5, 40).unwrap();
        assert!(r.value <= 1e-4 * r.alpha * -0.4);
        assert!(backtracking(|_| 1.0, 0.0, -1.0, 1.0, 1e-4, 0.5, 20).is_none());
    }
}
