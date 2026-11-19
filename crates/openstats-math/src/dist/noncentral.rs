//! Non-central χ², beta, F and t distributions.
//!
//! All four are evaluated through their exact Poisson-mixture (or, for t, the
//! Guenther / Lenth Owen-type) series — no normal or other approximations:
//!
//! * **χ′²(ν, λ)** = Σⱼ Pois(j; λ/2) · χ²_{ν+2j}
//! * **Beta′(a, b, λ)** = Σⱼ Pois(j; λ/2) · Beta(a+j, b)
//! * **F′(ν₁, ν₂, λ)**: F = (ν₂/ν₁)·B/(1−B) with B ~ Beta′(ν₁/2, ν₂/2, λ)
//! * **t′(ν, δ)**: F(t) = Φ(−δ) + ½ Σⱼ [pⱼ I_x(j+½, ν/2) + qⱼ I_x(j+1, ν/2)],
//!   x = t²/(t²+ν), pⱼ = e^{−δ²/2}(δ²/2)ʲ/j!, qⱼ = δ e^{−δ²/2}(δ²/2)ʲ/(√2 Γ(j+3/2)).
//!
//! The mixture indices are summed outward from the Poisson mode and stop when
//! the weights are below 1e-18 of the running sum, so the work is
//! O(√λ) incomplete-gamma/beta evaluations. The cdf is the sum of positive
//! terms, and the survival function is summed from the *complementary*
//! incomplete functions, so both tails keep full relative accuracy (for t′
//! this holds whenever δ·t ≥ 0 after the reflection t′(t; δ) = −t′(−t; −δ)
//! used for negative arguments; the opposite tail loses at most the
//! cancellation between the p- and q-series). Densities are sums of positive
//! terms accumulated in the log domain from the largest term, so they neither
//! overflow nor underflow before the true value does. Quantiles invert the
//! cdf (p ≤ ½) or the survival function (p > ½) with Brent's method.

use super::{require, valid_p, ContinuousDistribution};
use crate::error::Result;
use crate::integrate::GaussLegendre;
use crate::roots::brent;
use crate::special::erf::{erfcx, norm_cdf, norm_sf};
use crate::special::gamma::{lgamma_correction, ln_beta, ln_gamma, log1pmx, LN_SQRT_2PI};
use crate::special::incbeta::{beta_inc_pair, beta_prefix};
use crate::special::incgamma::{gamma_pq, gamma_prefix};
use std::f64::consts::{LN_2, PI};

use super::discrete::ln_dpois_raw;

const TOL: f64 = 1e-18;
const MAX_TERMS: u64 = 20_000_000;
/// ln(1e-18): terms this far below the largest are negligible.
const LN_TOL: f64 = -41.5;

fn ln_w(j: u64, m: f64) -> f64 {
    ln_dpois_raw(j as f64, m)
}

/// Σⱼ term(j), summed outward from `start`; `f(j)` returns the term and a
/// bound on |term(k)| for every k further from the mode than j.
fn mixture_sum(start: u64, f: impl Fn(u64) -> (f64, f64)) -> f64 {
    let (mut sum, _) = f(start);
    let mut j = start;
    for _ in 0..MAX_TERMS {
        j += 1;
        let (t, b) = f(j);
        sum += t;
        if b <= TOL * sum.abs() {
            break;
        }
    }
    let mut j = start;
    while j > 0 {
        j -= 1;
        let (t, b) = f(j);
        sum += t;
        if b <= TOL * sum.abs() {
            break;
        }
    }
    sum
}

/// ln Σⱼ tⱼ for a log-concave positive sequence given by the log of its
/// ratios: `ln_ratio(j) = ln(t_{j+1}/t_j)`, anchored at `ln_t(j0)`. The
/// peak is located exactly by walking from `j0`; returns −∞ if the anchor is.
fn ln_unimodal_sum(j0: u64, ln_t0: f64, ln_ratio: impl Fn(u64) -> f64) -> f64 {
    if ln_t0 == f64::NEG_INFINITY || ln_t0.is_nan() {
        return ln_t0;
    }
    // Move to the mode.
    let (mut peak, mut ln_peak) = (j0, ln_t0);
    while ln_ratio(peak) > 0.0 {
        ln_peak += ln_ratio(peak);
        peak += 1;
        if peak > MAX_TERMS {
            return f64::NAN;
        }
    }
    while peak > 0 && ln_ratio(peak - 1) < 0.0 {
        ln_peak -= ln_ratio(peak - 1);
        peak -= 1;
    }
    let mut sum = 1.0;
    let mut lt = 0.0;
    for j in (peak..).take(MAX_TERMS as usize) {
        lt += ln_ratio(j);
        sum += lt.exp();
        if lt < LN_TOL {
            break;
        }
    }
    let mut lt = 0.0;
    let mut j = peak;
    while j > 0 {
        lt -= ln_ratio(j - 1);
        j -= 1;
        sum += lt.exp();
        if lt < LN_TOL {
            break;
        }
    }
    ln_peak + sum.ln()
}

/// Expand [a, b] (inside [lo, hi]) by doubling the span until `f` changes sign.
fn bracket(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, lo: f64, hi: f64) -> Option<(f64, f64)> {
    let big = f64::MAX / 4.0;
    let (lo, hi) = (lo.max(-big), hi.min(big));
    let (mut fa, mut fb) = (f(a), f(b));
    let mut span = b - a;
    for _ in 0..2200 {
        if fa == 0.0 || fb == 0.0 || (fa < 0.0) != (fb < 0.0) {
            return Some((a, b));
        }
        span *= 2.0;
        // Move the end whose residual is smaller in magnitude (closer to the root).
        if fa.abs() < fb.abs() {
            let na = (a - span).max(lo);
            if na == a {
                return None;
            }
            a = na;
            fa = f(a);
        } else {
            let nb = (b + span).min(hi);
            if nb == b {
                return None;
            }
            b = nb;
            fb = f(b);
        }
    }
    None
}

/// Solve cdf(x) = p (p ≤ ½) or sf(x) = q (p > ½) on (lo, hi).
#[allow(clippy::too_many_arguments)]
fn solve_quantile(
    cdf: &dyn Fn(f64) -> f64,
    sf: &dyn Fn(f64) -> f64,
    p: f64,
    q: f64,
    lo: f64,
    hi: f64,
    center: f64,
    scale: f64,
) -> f64 {
    if !valid_p(p) || !valid_p(q) {
        return f64::NAN;
    }
    if p == 0.0 {
        return lo;
    }
    if q == 0.0 {
        return hi;
    }
    let use_cdf = p <= 0.5;
    let f = |x: f64| if use_cdf { cdf(x) - p } else { q - sf(x) };
    let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let a = if center.is_finite() { (center - scale).max(lo) } else { lo.max(-scale) };
    let mut b = if center.is_finite() { (center + scale).min(hi) } else { a + scale };
    if b <= a {
        b = a + scale;
    }
    match bracket(&f, a, b.min(hi), lo, hi) {
        Some((l, h)) => brent(f, l, h, 0.0, 800).unwrap_or(f64::NAN),
        None => f64::NAN,
    }
}

// ---------------------------------------------------------------------------
// Quadrature over the scale variable of a t-type mixture
// ---------------------------------------------------------------------------

/// ln of the density of S = √(χ²_ν/ν) at s > 0.
fn ln_chi_scaled_pdf(nu: f64, s: f64) -> f64 {
    let a = 0.5 * nu;
    if a >= 10.0 {
        let z = s * s - 1.0;
        let dev = if z.abs() < 0.5 { a * log1pmx(z) } else { a * (1.0 - s * s + 2.0 * s.ln()) };
        LN_2 + 0.5 * a.ln() - LN_SQRT_2PI - lgamma_correction(a) + dev - s.ln()
    } else {
        LN_2 + a * a.ln() - ln_gamma(a) + (nu - 1.0) * s.ln() - a * s * s
    }
}

/// ln Φ̄(z) for z ≥ 0 without underflow.
fn ln_norm_sf(z: f64) -> f64 {
    (0.5 * erfcx(z / std::f64::consts::SQRT_2)).ln() - 0.5 * z * z
}

/// ln ∫ exp(ln_h(w)) dw for a smooth unimodal ln_h on (−700, 40): golden-section
/// peak search, the ±46 drop points by bisection, then 24 panels of
/// Gauss–Legendre-32. Peak-normalised, so no overflow or underflow.
fn ln_integral_unimodal(ln_h: &dyn Fn(f64) -> f64) -> f64 {
    let h = |w: f64| {
        let v = ln_h(w);
        if v.is_nan() {
            f64::NEG_INFINITY
        } else {
            v
        }
    };
    let (mut a, mut b) = (-700.0f64, 40.0f64);
    let gr = 0.618_033_988_749_894_9;
    let (mut c, mut d) = (b - gr * (b - a), a + gr * (b - a));
    let (mut fc, mut fd) = (h(c), h(d));
    for _ in 0..300 {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = h(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = h(d);
        }
    }
    let wp = 0.5 * (a + b);
    let m0 = h(wp).max(fc).max(fd);
    if m0 == f64::NEG_INFINITY {
        return m0;
    }
    let edge = |dir: f64| -> f64 {
        let mut step = 0.25;
        let mut far = wp + dir * step;
        while h(far) > m0 - 46.0 && (-700.0..=40.0).contains(&(far + dir * step)) {
            step *= 2.0;
            far = wp + dir * step;
        }
        let (mut near, mut out) = (wp, far);
        if h(out) > m0 - 46.0 {
            return out; // reached the allowed range
        }
        for _ in 0..70 {
            let mid = 0.5 * (near + out);
            if h(mid) > m0 - 46.0 {
                near = mid;
            } else {
                out = mid;
            }
        }
        out
    };
    let (wl, wr) = (edge(-1.0), edge(1.0));
    let rule = GaussLegendre::new(32);
    let panels = 24;
    let width = (wr - wl) / panels as f64;
    let mut sum = 0.0;
    for k in 0..panels {
        let (c, half) = (wl + width * (k as f64 + 0.5), 0.5 * width);
        for (x, wt) in rule.nodes.iter().zip(&rule.weights) {
            sum += wt * half * (h(c + half * x) - m0).exp();
        }
    }
    m0 + sum.ln()
}

// ---------------------------------------------------------------------------
// Non-central chi-squared
// ---------------------------------------------------------------------------

/// Non-central χ² with ν degrees of freedom and non-centrality λ = Σ μᵢ².
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonCentralChiSquared {
    pub df: f64,
    pub lambda: f64,
}

impl NonCentralChiSquared {
    pub fn new(df: f64, lambda: f64) -> Result<Self> {
        require(df > 0.0 && df.is_finite(), || format!("NonCentralChiSquared: need df > 0 (got {df})"))?;
        require(lambda >= 0.0 && lambda.is_finite(), || {
            format!("NonCentralChiSquared: need non-centrality ≥ 0 (got {lambda})")
        })?;
        Ok(NonCentralChiSquared { df, lambda })
    }
}

impl ContinuousDistribution for NonCentralChiSquared {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::nc_chi_square(rng, self.df, self.lambda)
    }
    fn name(&self) -> &'static str {
        "NonCentralChiSquared"
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < 0.0 || x == f64::INFINITY {
            return f64::NEG_INFINITY;
        }
        let (a, m) = (0.5 * self.df, 0.5 * self.lambda);
        if x == 0.0 {
            return if self.df < 2.0 {
                f64::INFINITY
            } else if self.df == 2.0 {
                -m - LN_2
            } else {
                f64::NEG_INFINITY
            };
        }
        let y = 0.5 * x;
        // t_j = w_j · g_{a+j}(y) / 2, g_s(y) = yˢ⁻¹e⁻ʸ/Γ(s) = gamma_prefix(s, y)/y.
        let ln_t = |j: u64| ln_w(j, m) + (gamma_prefix(a + j as f64, y) / (2.0 * y)).ln();
        let ratio = |j: u64| (m / (j as f64 + 1.0)).ln() + (y / (a + j as f64)).ln();
        // Peak of the log-concave sequence: (j+1)(a+j) ≈ m·y.
        let b = a + 1.0;
        let disc = (b * b + 4.0 * (m * y - a)).max(0.0);
        let j0 = (0.5 * (-b + disc.sqrt())).max(0.0).floor() as u64;
        ln_unimodal_sum(j0, ln_t(j0), ratio)
    }
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 0.0;
        }
        if x == f64::INFINITY {
            return 1.0;
        }
        let (a, y, m) = (0.5 * self.df, 0.5 * x, 0.5 * self.lambda);
        mixture_sum(m.floor() as u64, |j| {
            let w = ln_w(j, m).exp();
            (w * gamma_pq(a + j as f64, y).0, w)
        })
        .min(1.0)
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 1.0;
        }
        if x == f64::INFINITY {
            return 0.0;
        }
        let (a, y, m) = (0.5 * self.df, 0.5 * x, 0.5 * self.lambda);
        mixture_sum(m.floor() as u64, |j| {
            let w = ln_w(j, m).exp();
            (w * gamma_pq(a + j as f64, y).1, w)
        })
        .min(1.0)
    }
    fn quantile(&self, p: f64) -> f64 {
        solve_quantile(&|x| self.cdf(x), &|x| self.sf(x), p, 1.0 - p, 0.0, f64::INFINITY, self.mean(), self.std_dev())
    }
    fn isf(&self, q: f64) -> f64 {
        solve_quantile(&|x| self.cdf(x), &|x| self.sf(x), 1.0 - q, q, 0.0, f64::INFINITY, self.mean(), self.std_dev())
    }
    fn mean(&self) -> f64 {
        self.df + self.lambda
    }
    fn variance(&self) -> f64 {
        2.0 * (self.df + 2.0 * self.lambda)
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
}

// ---------------------------------------------------------------------------
// Non-central beta
// ---------------------------------------------------------------------------

/// Non-central beta Beta′(a, b, λ) on [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonCentralBeta {
    pub a: f64,
    pub b: f64,
    pub lambda: f64,
}

impl NonCentralBeta {
    pub fn new(a: f64, b: f64, lambda: f64) -> Result<Self> {
        require(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite(), || {
            format!("NonCentralBeta: need shapes > 0 (got {a}, {b})")
        })?;
        require(lambda >= 0.0 && lambda.is_finite(), || {
            format!("NonCentralBeta: need non-centrality ≥ 0 (got {lambda})")
        })?;
        Ok(NonCentralBeta { a, b, lambda })
    }

    fn m(&self) -> f64 {
        0.5 * self.lambda
    }

    /// P(X ≤ x) given x and y = 1 − x.
    pub(crate) fn cdf_xy(&self, x: f64, y: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if y <= 0.0 {
            return 1.0;
        }
        let m = self.m();
        mixture_sum(m.floor() as u64, |j| {
            let w = ln_w(j, m).exp();
            (w * beta_inc_pair(self.a + j as f64, self.b, x, y).0, w)
        })
        .min(1.0)
    }

    pub(crate) fn sf_xy(&self, x: f64, y: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if y <= 0.0 {
            return 0.0;
        }
        let m = self.m();
        mixture_sum(m.floor() as u64, |j| {
            let w = ln_w(j, m).exp();
            (w * beta_inc_pair(self.a + j as f64, self.b, x, y).1, w)
        })
        .min(1.0)
    }

    /// ln of the density at (x, y = 1 − x), including the boundaries.
    pub(crate) fn ln_pdf_xy(&self, x: f64, y: f64) -> f64 {
        if x.is_nan() || y.is_nan() {
            return f64::NAN;
        }
        if x < 0.0 || y < 0.0 {
            return f64::NEG_INFINITY;
        }
        let (a, b, m) = (self.a, self.b, self.m());
        if x == 0.0 {
            return if a < 1.0 {
                f64::INFINITY
            } else if a == 1.0 {
                -m + b.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
        if y == 0.0 {
            // Σ w_j (a+j)·[b = 1] = a + m for b = 1.
            return if b < 1.0 {
                f64::INFINITY
            } else if b == 1.0 {
                (a + m).ln()
            } else {
                f64::NEG_INFINITY
            };
        }
        let ln_t = |j: u64| ln_w(j, m) + (beta_prefix(a + j as f64, b, x, y) / (x * y)).ln();
        let ratio = |j: u64| {
            let aj = a + j as f64;
            (m / (j as f64 + 1.0)).ln() + (x * (aj + b) / aj).ln()
        };
        // Peak of the log-concave sequence: m·x·(a+b+j) = (j+1)(a+j).
        let mx = m * x;
        let (c1, c0) = (a + 1.0 - mx, a - mx * (a + b));
        let disc = (c1 * c1 - 4.0 * c0).max(0.0);
        let j0 = (0.5 * (-c1 + disc.sqrt())).max(0.0).floor() as u64;
        ln_unimodal_sum(j0, ln_t(j0), ratio)
    }

    /// Σ w_j · h(j) for 0 ≤ h ≤ 1.
    fn expect(&self, h: impl Fn(f64) -> f64) -> f64 {
        let m = self.m();
        mixture_sum(m.floor() as u64, |j| {
            let w = ln_w(j, m).exp();
            (w * h(self.a + j as f64), w)
        })
    }
}

impl ContinuousDistribution for NonCentralBeta {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::nc_beta(rng, self.a, self.b, self.lambda)
    }
    fn name(&self) -> &'static str {
        "NonCentralBeta"
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }
        self.ln_pdf_xy(x, 1.0 - x)
    }
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        self.cdf_xy(x.clamp(0.0, 1.0), (1.0 - x).clamp(0.0, 1.0))
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        self.sf_xy(x.clamp(0.0, 1.0), (1.0 - x).clamp(0.0, 1.0))
    }
    fn quantile(&self, p: f64) -> f64 {
        solve_quantile(&|x| self.cdf(x), &|x| self.sf(x), p, 1.0 - p, 0.0, 1.0, self.mean(), self.std_dev())
    }
    fn isf(&self, q: f64) -> f64 {
        solve_quantile(&|x| self.cdf(x), &|x| self.sf(x), 1.0 - q, q, 0.0, 1.0, self.mean(), self.std_dev())
    }
    fn mean(&self) -> f64 {
        self.expect(|a| a / (a + self.b))
    }
    fn variance(&self) -> f64 {
        let ex2 = self.expect(|a| a * (a + 1.0) / ((a + self.b) * (a + self.b + 1.0)));
        let m1 = self.mean();
        (ex2 - m1 * m1).max(0.0)
    }
    fn support(&self) -> (f64, f64) {
        (0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Non-central F
// ---------------------------------------------------------------------------

/// Non-central F with numerator/denominator degrees of freedom ν₁, ν₂ and
/// non-centrality λ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonCentralF {
    pub df1: f64,
    pub df2: f64,
    pub lambda: f64,
}

impl NonCentralF {
    pub fn new(df1: f64, df2: f64, lambda: f64) -> Result<Self> {
        require(df1 > 0.0 && df2 > 0.0 && df1.is_finite() && df2.is_finite(), || {
            format!("NonCentralF: need positive finite degrees of freedom (got {df1}, {df2})")
        })?;
        require(lambda >= 0.0 && lambda.is_finite(), || {
            format!("NonCentralF: need non-centrality ≥ 0 (got {lambda})")
        })?;
        Ok(NonCentralF { df1, df2, lambda })
    }
    fn beta(&self) -> NonCentralBeta {
        NonCentralBeta { a: 0.5 * self.df1, b: 0.5 * self.df2, lambda: self.lambda }
    }
    /// (y, 1 − y) with y = ν₁x/(ν₁x + ν₂), the beta-scale abscissa.
    fn xy(&self, x: f64) -> (f64, f64) {
        let d = self.df1 * x + self.df2;
        if d.is_infinite() {
            (1.0, 0.0)
        } else {
            (self.df1 * x / d, self.df2 / d)
        }
    }
}

impl ContinuousDistribution for NonCentralF {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::nc_f(rng, self.df1, self.df2, self.lambda)
    }
    fn name(&self) -> &'static str {
        "NonCentralF"
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < 0.0 || x == f64::INFINITY {
            return f64::NEG_INFINITY;
        }
        let (y, cy) = self.xy(x);
        let d = self.df1 * x + self.df2;
        // dy/dx = ν₁ν₂/(ν₁x + ν₂)²
        self.beta().ln_pdf_xy(y, cy) + self.df1.ln() + self.df2.ln() - 2.0 * d.ln()
    }
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 0.0;
        }
        let (y, cy) = self.xy(x);
        self.beta().cdf_xy(y, cy)
    }
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 1.0;
        }
        let (y, cy) = self.xy(x);
        self.beta().sf_xy(y, cy)
    }
    fn quantile(&self, p: f64) -> f64 {
        let (c, s) = self.search_scale();
        solve_quantile(&|x| self.cdf(x), &|x| self.sf(x), p, 1.0 - p, 0.0, f64::INFINITY, c, s)
    }
    fn isf(&self, q: f64) -> f64 {
        let (c, s) = self.search_scale();
        solve_quantile(&|x| self.cdf(x), &|x| self.sf(x), 1.0 - q, q, 0.0, f64::INFINITY, c, s)
    }
    fn mean(&self) -> f64 {
        if self.df2 > 2.0 {
            self.df2 * (self.df1 + self.lambda) / (self.df1 * (self.df2 - 2.0))
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        let (n1, n2, l) = (self.df1, self.df2, self.lambda);
        if n2 > 4.0 {
            2.0 * (n2 / n1).powi(2) * ((n1 + l).powi(2) + (n1 + 2.0 * l) * (n2 - 2.0))
                / ((n2 - 2.0).powi(2) * (n2 - 4.0))
        } else {
            f64::NAN
        }
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
}

impl NonCentralF {
    /// Starting point and width for quantile bracketing (the moments may not exist).
    fn search_scale(&self) -> (f64, f64) {
        let c = (1.0 + self.lambda / self.df1) * if self.df2 > 2.0 { self.df2 / (self.df2 - 2.0) } else { 1.0 };
        (c, 0.5 * c.max(1.0))
    }
}

// ---------------------------------------------------------------------------
// Non-central t
// ---------------------------------------------------------------------------

/// Non-central t with ν degrees of freedom (∞ allowed) and non-centrality δ:
/// T = (Z + δ)/√(χ²_ν/ν).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonCentralT {
    pub df: f64,
    pub delta: f64,
}

impl NonCentralT {
    pub fn new(df: f64, delta: f64) -> Result<Self> {
        require(df > 0.0, || format!("NonCentralT: need df > 0 (got {df})"))?;
        require(delta.is_finite(), || format!("NonCentralT: need finite non-centrality (got {delta})"))?;
        Ok(NonCentralT { df, delta })
    }

    /// (x, y) = (t²/(t²+ν), ν/(t²+ν)) for t ≥ 0.
    fn xy(&self, t: f64) -> (f64, f64) {
        let t2 = t * t;
        if t2.is_infinite() {
            let y = (self.df / t) / t;
            (1.0 - y, y)
        } else {
            (t2 / (t2 + self.df), self.df / (t2 + self.df))
        }
    }

    /// ½ Σⱼ [pⱼ Iⱼ + qⱼ I′ⱼ] for t > 0 where `upper` selects the
    /// complementary incomplete beta functions. q carries the sign of δ.
    fn series(&self, t: f64, delta: f64, upper: bool) -> f64 {
        let (x, y) = self.xy(t);
        let b = 0.5 * self.df;
        let m = 0.5 * delta * delta;
        let ln_abs_delta = delta.abs().ln();
        let sign = delta.signum();
        let ln_pi_half = 0.5 * PI.ln();
        mixture_sum(m.floor() as u64, |j| {
            let jf = j as f64;
            let lw = ln_w(j, m);
            let p = lw.exp();
            // q_j = δ p_j · j!/(√2 Γ(j+3/2)) = δ p_j B(j+1, ½)/(√2 √π)
            let q = sign * (ln_abs_delta + lw + ln_beta(jf + 1.0, 0.5) - 0.5 * LN_2 - ln_pi_half).exp();
            let (i1, c1) = beta_inc_pair(jf + 0.5, b, x, y);
            let (i2, c2) = beta_inc_pair(jf + 1.0, b, x, y);
            let (v1, v2) = if upper { (c1, c2) } else { (i1, i2) };
            let term = 0.5 * (p * v1 + q * v2);
            let bound = 0.5 * (p + q.abs()) * 2.0;
            (term, bound)
        })
    }

    /// (P(T ≤ t), P(T > t)) for t ≥ 0, one of them computed by series.
    fn lower_for_nonneg(&self, t: f64, delta: f64) -> f64 {
        if t == 0.0 {
            return norm_cdf(-delta);
        }
        (norm_cdf(-delta) + self.series(t, delta, false)).clamp(0.0, 1.0)
    }
    fn upper_for_nonneg(&self, t: f64, delta: f64) -> f64 {
        if t == 0.0 {
            return norm_cdf(delta);
        }
        if delta < 0.0 {
            // P(T > t) = E[Φ̄(tS + |δ|)]: positive integrand, no cancellation.
            let nu = self.df;
            let ln_i = ln_integral_unimodal(&|w: f64| {
                let s = w.exp();
                ln_chi_scaled_pdf(nu, s) + w + ln_norm_sf(t * s - delta)
            });
            return ln_i.exp().clamp(0.0, 1.0);
        }
        self.series(t, delta, true).clamp(0.0, 1.0)
    }
}

impl ContinuousDistribution for NonCentralT {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::nc_t(rng, self.df, self.delta)
    }
    fn name(&self) -> &'static str {
        "NonCentralT"
    }
    fn ln_pdf(&self, t: f64) -> f64 {
        if t.is_nan() {
            return f64::NAN;
        }
        let (nu, delta) = (self.df, self.delta);
        if nu.is_infinite() {
            return -0.5 * (t - delta) * (t - delta) - 0.5 * (2.0 * PI).ln();
        }
        if t.is_infinite() {
            return f64::NEG_INFINITY;
        }
        let ln_b = ln_beta(0.5 * nu, 0.5);
        if t == 0.0 {
            return -0.5 * delta * delta - 0.5 * nu.ln() - ln_b;
        }
        if delta * t < 0.0 {
            // f(t) = ∫ s φ(ts − δ) f_S(s) ds, with φ(ts − δ) = φ(|t|s + |δ|) a Gaussian
            // tail: a positive, peaked integrand handled in the log domain.
            return ln_integral_unimodal(&|w: f64| {
                let s = w.exp();
                ln_chi_scaled_pdf(nu, s) + 2.0 * w - 0.5 * (t * s - delta).powi(2) - LN_SQRT_2PI
            });
        }
        // δt ≥ 0: all-positive series. ln pdf = lnK0 + ln Σ r_j.
        let s = nu + t * t;
        let ln_k0 = -0.5 * s.ln() - 0.5 * nu * (t * t / nu).ln_1p() - 0.5 * delta * delta - ln_b;
        let y = delta * t * std::f64::consts::SQRT_2 / s.sqrt();
        if y == 0.0 {
            return ln_k0;
        }
        let ln_y = y.ln();
        // ρ_j = Γ((ν+j+2)/2)/Γ((ν+j+1)/2); ρ₀ = (ν/2) B(ν/2, ½)/√π, ρ_{j+1} = ((ν+j+1)/2)/ρ_j.
        let mut rho = 0.5 * nu * ln_b.exp() / PI.sqrt();
        // r_j recurrence in the log domain with an online log-sum-exp.
        let (mut ln_r, mut max, mut sum) = (0.0f64, 0.0f64, 1.0f64);
        let mut j = 0u64;
        loop {
            let step = ln_y + rho.ln() - (j as f64 + 1.0).ln();
            ln_r += step;
            rho = 0.5 * (nu + j as f64 + 1.0) / rho;
            j += 1;
            if ln_r > max {
                sum = sum * (max - ln_r).exp() + 1.0;
                max = ln_r;
            } else {
                sum += (ln_r - max).exp();
            }
            if (step < 0.0 && ln_r < max + LN_TOL) || !ln_r.is_finite() {
                break;
            }
            if j > MAX_TERMS {
                return f64::NAN;
            }
        }
        ln_k0 + max + sum.ln()
    }
    fn pdf(&self, t: f64) -> f64 {
        self.ln_pdf(t).exp()
    }
    fn cdf(&self, t: f64) -> f64 {
        if t.is_nan() {
            return f64::NAN;
        }
        if self.df.is_infinite() {
            return norm_cdf(t - self.delta);
        }
        if t == f64::INFINITY {
            return 1.0;
        }
        if t == f64::NEG_INFINITY {
            return 0.0;
        }
        if t >= 0.0 {
            self.lower_for_nonneg(t, self.delta)
        } else {
            self.upper_for_nonneg(-t, -self.delta)
        }
    }
    fn sf(&self, t: f64) -> f64 {
        if t.is_nan() {
            return f64::NAN;
        }
        if self.df.is_infinite() {
            return norm_sf(t - self.delta);
        }
        if t == f64::INFINITY {
            return 0.0;
        }
        if t == f64::NEG_INFINITY {
            return 1.0;
        }
        if t >= 0.0 {
            self.upper_for_nonneg(t, self.delta)
        } else {
            self.lower_for_nonneg(-t, -self.delta)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        let s = 1.0 + self.delta.abs();
        solve_quantile(&|x| self.cdf(x), &|x| self.sf(x), p, 1.0 - p, f64::NEG_INFINITY, f64::INFINITY, self.delta, s)
    }
    fn isf(&self, q: f64) -> f64 {
        let s = 1.0 + self.delta.abs();
        solve_quantile(&|x| self.cdf(x), &|x| self.sf(x), 1.0 - q, q, f64::NEG_INFINITY, f64::INFINITY, self.delta, s)
    }
    fn mean(&self) -> f64 {
        let nu = self.df;
        if nu.is_infinite() {
            return self.delta;
        }
        if nu <= 1.0 {
            return f64::NAN;
        }
        // δ √(ν/2) Γ((ν−1)/2)/Γ(ν/2),  Γ(a−½)/Γ(a) = B(a−½, ½)/√π
        self.delta * (0.5 * nu).sqrt() * ln_beta(0.5 * (nu - 1.0), 0.5).exp() / PI.sqrt()
    }
    fn variance(&self) -> f64 {
        let nu = self.df;
        if nu.is_infinite() {
            return 1.0;
        }
        if nu <= 2.0 {
            return f64::NAN;
        }
        let m = self.mean();
        nu * (1.0 + self.delta * self.delta) / (nu - 2.0) - m * m
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{Beta, ChiSquared, FisherF, StudentT};

    fn close(a: f64, b: f64, rel: f64) -> bool {
        a == b || (a - b).abs() <= rel * a.abs().max(b.abs())
    }

    /// Cheap cdf ↔ pdf ↔ quantile consistency for any continuous distribution.
    fn check<D: ContinuousDistribution>(d: &D, xs: &[f64]) {
        for &x in xs {
            let (c, s) = (d.cdf(x), d.sf(x));
            assert!((c + s - 1.0).abs() < 1e-13, "{}: cdf+sf at {x}: {c} {s}", d.name());
            let pd = d.pdf(x);
            assert!(
                pd >= 0.0 && (d.ln_pdf(x) - pd.ln()).abs() <= 1e-12 * pd.ln().abs().max(1.0),
                "{}: pdf at {x}",
                d.name()
            );
            let (lo, hi) = d.support();
            let mut h = 1e-5 * x.abs().max(1.0);
            if lo.is_finite() {
                h = h.min(1e-3 * (x - lo));
            }
            if hi.is_finite() {
                h = h.min(1e-3 * (hi - x));
            }
            if pd > 1e-300 && x - h > lo && x + h < hi {
                let num = (d.cdf(x + h) - d.cdf(x - h)) / (2.0 * h);
                assert!((num - pd).abs() < 1e-6 * pd.max(1e-6), "{}: pdf vs dcdf at {x}: {pd} vs {num}", d.name());
            }
            if c > 1e-280 && c < 0.999 {
                let q = d.quantile(c);
                assert!(close(d.cdf(q), c, 1e-11), "{}: quantile round trip at {x}", d.name());
            }
            if s > 1e-280 && s < 0.999 {
                let q = d.isf(s);
                assert!(close(d.sf(q), s, 1e-11), "{}: isf round trip at {x}", d.name());
            }
        }
        assert_eq!(d.quantile(0.0), d.support().0);
        assert_eq!(d.quantile(1.0), d.support().1);
        assert!(d.quantile(1.5).is_nan() && d.quantile(f64::NAN).is_nan());
    }

    #[test]
    fn zero_non_centrality_reduces_to_central() {
        let (nu, x) = (7.0, 3.1);
        let c = NonCentralChiSquared::new(nu, 0.0).unwrap();
        let cc = ChiSquared::new(nu).unwrap();
        assert!(
            close(c.cdf(x), cc.cdf(x), 1e-14) && close(c.sf(x), cc.sf(x), 1e-14) && close(c.pdf(x), cc.pdf(x), 1e-14)
        );
        let b = NonCentralBeta::new(2.5, 4.0, 0.0).unwrap();
        let bb = Beta::new(2.5, 4.0).unwrap();
        assert!(close(b.cdf(0.3), bb.cdf(0.3), 1e-14) && close(b.pdf(0.3), bb.pdf(0.3), 1e-14));
        let f = NonCentralF::new(5.0, 9.0, 0.0).unwrap();
        let ff = FisherF::new(5.0, 9.0).unwrap();
        for x in [0.2, 1.0, 4.0] {
            assert!(
                close(f.cdf(x), ff.cdf(x), 1e-13)
                    && close(f.sf(x), ff.sf(x), 1e-13)
                    && close(f.pdf(x), ff.pdf(x), 1e-13)
            );
        }
        let t = NonCentralT::new(6.0, 0.0).unwrap();
        let tt = StudentT::new(6.0).unwrap();
        for x in [-4.0, -1.0, 0.0, 0.5, 2.5] {
            assert!(
                close(t.cdf(x), tt.cdf(x), 1e-13)
                    && close(t.sf(x), tt.sf(x), 1e-13)
                    && close(t.pdf(x), tt.pdf(x), 1e-13),
                "x={x}"
            );
        }
    }

    #[test]
    fn chi_squared_identities() {
        // df = 1: X = (Z + μ)², so cdf(x) = Φ(√x − μ) − Φ(−√x − μ) with λ = μ².
        let mu = 1.7f64;
        let d = NonCentralChiSquared::new(1.0, mu * mu).unwrap();
        for x in [0.05f64, 0.5, 2.0, 9.0, 30.0] {
            let r = x.sqrt();
            let want = norm_cdf(r - mu) - norm_cdf(-r - mu);
            assert!((d.cdf(x) - want).abs() < 1e-15, "x={x}");
            let want_sf = norm_sf(r - mu) + norm_cdf(-r - mu);
            assert!(close(d.sf(x), want_sf, 1e-14));
        }
        // df = 2: the Marcum Q function; cdf(x) = 1 − Q₁(μ, √x) and pdf(x) = ½ e^{−(x+λ)/2} I₀(√(λx)).
        let lam = 3.0f64;
        let d = NonCentralChiSquared::new(2.0, lam).unwrap();
        for x in [0.4, 2.0, 7.5, 20.0] {
            let want = 0.5 * (-(x + lam) / 2.0).exp() * crate::special::bessel::i0((lam * x).sqrt());
            assert!(close(d.pdf(x), want, 1e-13), "x={x}");
        }
        // Moments.
        let d = NonCentralChiSquared::new(4.0, 6.0).unwrap();
        assert_eq!((d.mean(), d.variance()), (10.0, 32.0));
        // Boundary behaviour of the density at 0.
        assert_eq!(NonCentralChiSquared::new(1.0, 2.0).unwrap().pdf(0.0), f64::INFINITY);
        assert!(close(NonCentralChiSquared::new(2.0, 2.0).unwrap().pdf(0.0), 0.5 * (-1.0f64).exp(), 1e-15));
        assert_eq!(NonCentralChiSquared::new(3.0, 2.0).unwrap().pdf(0.0), 0.0);
        assert_eq!(d.cdf(-1.0), 0.0);
        assert_eq!(d.sf(-1.0), 1.0);
        check(&d, &[0.5, 3.0, 9.0, 10.0, 25.0, 60.0]);
        check(&NonCentralChiSquared::new(0.7, 30.0).unwrap(), &[0.5, 10.0, 30.0, 50.0]);
        check(&NonCentralChiSquared::new(20.0, 2000.0).unwrap(), &[1900.0, 2020.0, 2150.0]);
        // Large non-centrality keeps working (Poisson weights summed outward from the mode).
        let big = NonCentralChiSquared::new(3.0, 1e6).unwrap();
        assert!(close(big.cdf(1e6 + 3.0), 0.5, 2e-3));
        assert!(NonCentralChiSquared::new(0.0, 1.0).is_err() && NonCentralChiSquared::new(2.0, -1.0).is_err());
    }

    #[test]
    fn beta_and_f_identities() {
        let b = NonCentralBeta::new(2.0, 3.5, 4.0).unwrap();
        check(&b, &[0.05, 0.2, 0.5, 0.8, 0.97]);
        // Mixture identity: Σ_j Pois(j; 2)·Beta(2+j, 3.5).cdf(x) with an independent loop.
        let x = 0.4;
        let mut want = 0.0;
        let mut w = (-2.0f64).exp();
        for j in 0..80 {
            want += w * Beta::new(2.0 + j as f64, 3.5).unwrap().cdf(x);
            w *= 2.0 / (j as f64 + 1.0);
        }
        assert!(close(b.cdf(x), want, 1e-14));
        // F via beta: P(F ≤ x) = P(B ≤ ν₁x/(ν₁x+ν₂)).
        let f = NonCentralF::new(6.0, 11.0, 4.0).unwrap();
        let bb = NonCentralBeta::new(3.0, 5.5, 4.0).unwrap();
        for x in [0.3, 1.0, 2.5, 9.0] {
            assert!(close(f.cdf(x), bb.cdf(6.0 * x / (6.0 * x + 11.0)), 1e-13));
        }
        check(&f, &[0.1, 0.5, 1.0, 2.0, 5.0, 20.0]);
        // Mean of the non-central F: ν₂(ν₁+λ)/(ν₁(ν₂−2)); variance formula against numerical moments.
        assert!(close(f.mean(), 11.0 * 10.0 / (6.0 * 9.0), 1e-15));
        let m = integrate_moment(&f, 0.0);
        assert!(close(m, f.mean(), 1e-7), "numerical mean {m} vs {}", f.mean());
        assert!(NonCentralF::new(4.0, 2.0, 1.0).unwrap().mean().is_nan());
        assert!(NonCentralF::new(4.0, 4.0, 1.0).unwrap().variance().is_nan());
        // Non-central beta moments by direct mixture sums.
        let mb = integrate_moment(&b, 0.0);
        assert!(close(b.mean(), mb, 1e-9));
        let var_num = integrate_moment(&b, 2.0) - mb * mb;
        assert!(close(b.variance(), var_num, 1e-7), "{} vs {var_num}", b.variance());
        // Boundary densities.
        assert_eq!(NonCentralBeta::new(0.5, 2.0, 1.0).unwrap().pdf(0.0), f64::INFINITY);
        assert!(close(NonCentralBeta::new(1.0, 2.0, 1.0).unwrap().pdf(0.0), 2.0 * (-0.5f64).exp(), 1e-15));
        assert_eq!(NonCentralBeta::new(2.0, 2.0, 1.0).unwrap().pdf(1.0), 0.0);
        assert!(close(NonCentralBeta::new(2.0, 1.0, 1.0).unwrap().pdf(1.0), 2.5, 1e-15));
        assert!(NonCentralBeta::new(0.0, 1.0, 1.0).is_err() && NonCentralF::new(1.0, 0.0, 1.0).is_err());
    }

    /// E[(X − shift)^k-style moment] by composite Gauss–Legendre on the cdf's own
    /// density, where `shift = 0` gives E[X] and `shift = 2` gives E[X²].
    fn integrate_moment<D: ContinuousDistribution>(d: &D, order: f64) -> f64 {
        let (lo, hi) = d.support();
        let hi = if hi.is_finite() { hi } else { d.quantile(1.0 - 1e-13) };
        let rule = crate::integrate::GaussLegendre::new(32);
        let panels = 400;
        let mut sum = 0.0;
        let w = (hi - lo) / panels as f64;
        for k in 0..panels {
            let (c, half) = (lo + w * (k as f64 + 0.5), 0.5 * w);
            for (x, wt) in rule.nodes.iter().zip(&rule.weights) {
                let v = c + half * x;
                let p = if order == 2.0 { v * v } else { v };
                sum += wt * half * p * d.pdf(v);
            }
        }
        sum
    }

    #[test]
    fn t_identities() {
        // ν = ∞: normal shift.
        let t = NonCentralT::new(f64::INFINITY, 1.5).unwrap();
        assert!(close(t.cdf(0.7), norm_cdf(0.7 - 1.5), 1e-15) && close(t.sf(3.0), norm_sf(1.5), 1e-15));
        assert_eq!((t.mean(), t.variance()), (1.5, 1.0));
        // A very large ν approaches it.
        let big = NonCentralT::new(1e7, 1.5).unwrap();
        assert!(close(big.cdf(0.7), t.cdf(0.7), 1e-6));
        // Symmetry T(ν, δ) = −T(ν, −δ).
        let (a, b) = (NonCentralT::new(8.0, 2.2).unwrap(), NonCentralT::new(8.0, -2.2).unwrap());
        for x in [-3.0, -0.5, 0.0, 0.9, 4.0] {
            assert!(close(a.cdf(x), b.sf(-x), 1e-13) && close(a.pdf(x), b.pdf(-x), 1e-12), "x={x}");
        }
        // cdf(0) = Φ(−δ) exactly.
        assert_eq!(a.cdf(0.0), norm_cdf(-2.2));
        assert_eq!(a.sf(0.0), norm_cdf(2.2));
        // Moments: E[T] and Var[T] against numerical integration of the density.
        let d = NonCentralT::new(9.0, 1.3).unwrap();
        let m = integrate_t_moment(&d, 1);
        assert!(close(d.mean(), m, 1e-6), "mean {} vs {m}", d.mean());
        let v = integrate_t_moment(&d, 2) - m * m;
        assert!(close(d.variance(), v, 1e-5), "var {} vs {v}", d.variance());
        assert!(NonCentralT::new(1.0, 1.0).unwrap().mean().is_nan());
        assert!(NonCentralT::new(2.0, 1.0).unwrap().variance().is_nan());
        // pdf at 0 closed form: e^{−δ²/2} Γ((ν+1)/2) / (√(πν) Γ(ν/2)).
        let want = (-0.5f64 * 1.3 * 1.3).exp() * crate::special::gamma::gamma(5.0)
            / ((9.0 * PI).sqrt() * crate::special::gamma::gamma(4.5));
        assert!(close(d.pdf(0.0), want, 1e-14));
        // The two density branches (series for δt ≥ 0, quadrature for δt < 0) are continuous across t = 0.
        assert!(close(d.pdf(1e-9), d.pdf(-1e-9), 1e-8));
        check(&d, &[-6.0, -2.0, -0.5, 0.5, 1.3, 3.0, 8.0]);
        check(&NonCentralT::new(2.5, -1.0).unwrap(), &[-5.0, -1.0, 0.7, 4.0]);
        check(&NonCentralT::new(40.0, 6.0).unwrap(), &[0.0, 4.0, 6.0, 9.0]);
        assert!(NonCentralT::new(0.0, 1.0).is_err() && NonCentralT::new(5.0, f64::NAN).is_err());
    }

    fn integrate_t_moment(d: &NonCentralT, k: i32) -> f64 {
        let rule = crate::integrate::GaussLegendre::new(32);
        let (lo, hi) = (d.quantile(1e-14), d.quantile(1.0 - 1e-14));
        let panels = 600;
        let w = (hi - lo) / panels as f64;
        let mut sum = 0.0;
        for p in 0..panels {
            let (c, half) = (lo + w * (p as f64 + 0.5), 0.5 * w);
            for (x, wt) in rule.nodes.iter().zip(&rule.weights) {
                let v = c + half * x;
                sum += wt * half * v.powi(k) * d.pdf(v);
            }
        }
        sum
    }

    #[test]
    fn t_series_matches_independent_quadrature() {
        // The series (cdf, sf) against the positive-integrand representation E[Φ(tS − δ)].
        let rule = crate::integrate::GaussLegendre::new(32);
        for &(nu, delta, t) in
            &[(5.0, 1.0, 2.0), (12.0, 3.0, 4.0), (3.0, -2.0, 1.0), (30.0, 6.0, 7.0), (2.0, 0.5, -1.5)]
        {
            let d = NonCentralT::new(nu, delta).unwrap();
            let ln_h = |w: f64| ln_chi_scaled_pdf(nu, w.exp()) + w;
            let (mut cdf, mut sf) = (0.0, 0.0);
            let panels = 900;
            let (wl, wr) = (-45.0, 2.5);
            let width = (wr - wl) / panels as f64;
            for k in 0..panels {
                let (c, half) = (wl + width * (k as f64 + 0.5), 0.5 * width);
                for (x, wt) in rule.nodes.iter().zip(&rule.weights) {
                    let w = c + half * x;
                    let dens = ln_h(w).exp();
                    let z = t * w.exp() - delta;
                    cdf += wt * half * dens * norm_cdf(z);
                    sf += wt * half * dens * norm_sf(z);
                }
            }
            assert!(close(d.cdf(t), cdf, 1e-11), "cdf({t}; {nu}, {delta}): {} vs {cdf}", d.cdf(t));
            assert!(close(d.sf(t), sf, 1e-11), "sf({t}; {nu}, {delta}): {} vs {sf}", d.sf(t));
        }
    }

    #[test]
    fn t_tails_keep_relative_accuracy() {
        // Left tail for δ > 0: P(T ≤ t) ≤ P(T ≤ 0) = Φ(−δ), and strictly positive.
        let d = NonCentralT::new(100.0, 10.0).unwrap();
        assert!(close(d.cdf(-30.0), 6.801_053_238_105_849e-107, 1e-11)); // adaptive-quadrature value
                                                                         // Upper tail with δ < 0.
        let d = NonCentralT::new(30.0, -5.0).unwrap();
        let s = d.sf(30.0);
        assert!(s > 0.0 && s < norm_cdf(-5.0));
        assert!(close(d.cdf(-30.0 * 0.0 - 1.0), 1.0 - d.sf(-1.0), 1e-13));
    }

    #[test]
    fn solver_helpers() {
        // Peak-normalised integral of a Gaussian in w: ∫ exp(−(w−3)²/2 + 700) dw = √(2π) e^{700}.
        let v = ln_integral_unimodal(&|w: f64| -0.5 * (w - 3.0) * (w - 3.0) + 700.0);
        assert!((v - (700.0 + 0.5 * (2.0 * PI).ln())).abs() < 1e-13);
        // ln χ-density integrates to 1 over s for several ν (checks the Stirling-correction branch too).
        for &nu in &[0.6, 3.0, 19.0, 21.0, 400.0] {
            let v = ln_integral_unimodal(&|w: f64| ln_chi_scaled_pdf(nu, w.exp()) + w);
            assert!(v.abs() < 1e-12, "ν={nu}: ln ∫ = {v}");
        }
        assert_eq!(ln_integral_unimodal(&|_| f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(close(ln_norm_sf(3.0).exp(), norm_sf(3.0), 1e-14));
        // Bracket expansion finds roots far from the start and stops at bounds.
        let (a, b) = bracket(&|x: f64| x - 1e30, 0.0, 1.0, 0.0, f64::INFINITY).unwrap();
        assert!(a <= 1e30 && b >= 1e30);
        assert!(bracket(&|x: f64| x + 1.0, 0.0, 1.0, 0.0, 10.0).is_none());
    }
}
