//! Location–scale life distributions.
//!
//! Every family is `z = (g(t) − μ)/σ` with a standard base distribution of `z`: Weibull and the
//! exponential use the smallest extreme value law of `ln t`, the lognormal the normal law of
//! `ln t`, the log-logistic the logistic law of `ln t`; the normal and logistic fit `t` itself.

use openstats_math::special::erf::{norm_cdf, norm_ln_pdf, norm_ppf, norm_sf};

/// A life-time distribution family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Two-parameter Weibull (shape β = 1/σ, scale η = e^μ).
    Weibull,
    /// Lognormal (location μ and scale σ of ln t).
    Lognormal,
    /// Exponential (mean e^μ; σ fixed at 1).
    Exponential,
    /// Normal.
    Normal,
    /// Logistic.
    Logistic,
    /// Log-logistic (location μ and scale σ of ln t).
    Loglogistic,
}

/// All families, in the order used for ranking.
pub const FAMILIES: [Family; 6] =
    [Family::Weibull, Family::Lognormal, Family::Exponential, Family::Loglogistic, Family::Normal, Family::Logistic];

#[derive(Clone, Copy)]
enum Base {
    Sev,
    Norm,
    Logis,
}

/// `ln(1 + eˣ)` without overflow.
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

impl Family {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Family::Weibull => "Weibull",
            Family::Lognormal => "Lognormal",
            Family::Exponential => "Exponential",
            Family::Normal => "Normal",
            Family::Logistic => "Logistic",
            Family::Loglogistic => "Loglogistic",
        }
    }

    /// `true` when the model is for `ln t` (times must be positive).
    pub fn is_log(self) -> bool {
        !matches!(self, Family::Normal | Family::Logistic)
    }

    /// `true` when the scale is fixed (exponential).
    pub fn fixed_scale(self) -> bool {
        self == Family::Exponential
    }

    fn base(self) -> Base {
        match self {
            Family::Weibull | Family::Exponential => Base::Sev,
            Family::Lognormal | Family::Normal => Base::Norm,
            Family::Logistic | Family::Loglogistic => Base::Logis,
        }
    }

    /// The transformation `g(t)` (`ln t` or the identity); `ln 0 = −∞`.
    pub fn g(self, t: f64) -> f64 {
        if self.is_log() {
            t.ln()
        } else {
            t
        }
    }

    /// Inverse of [`Family::g`].
    pub fn g_inv(self, x: f64) -> f64 {
        if self.is_log() {
            x.exp()
        } else {
            x
        }
    }

    /// Log density of the standard variate.
    pub fn ln_pdf0(self, z: f64) -> f64 {
        match self.base() {
            Base::Sev => z - z.exp(),
            Base::Norm => norm_ln_pdf(z),
            Base::Logis => -z.abs() - 2.0 * (-z.abs()).exp().ln_1p(),
        }
    }

    /// Log of the standard cdf.
    pub fn ln_cdf0(self, z: f64) -> f64 {
        match self.base() {
            Base::Sev => (-(-z.exp()).exp_m1()).ln(),
            Base::Norm => {
                let c = norm_cdf(z);
                if c > 0.0 {
                    c.ln()
                } else {
                    norm_ln_pdf(z) - (-z).ln() // asymptotic far lower tail
                }
            }
            Base::Logis => -softplus(-z),
        }
    }

    /// Log of the standard survival function.
    pub fn ln_sf0(self, z: f64) -> f64 {
        match self.base() {
            Base::Sev => -z.exp(),
            Base::Norm => {
                let s = norm_sf(z);
                if s > 0.0 {
                    s.ln()
                } else {
                    norm_ln_pdf(z) - z.ln()
                }
            }
            Base::Logis => -softplus(z),
        }
    }

    /// Standard cdf.
    pub fn cdf0(self, z: f64) -> f64 {
        match self.base() {
            Base::Sev => -(-z.exp()).exp_m1(),
            Base::Norm => norm_cdf(z),
            Base::Logis => 1.0 / (1.0 + (-z).exp()),
        }
    }

    /// Standard survival function.
    pub fn sf0(self, z: f64) -> f64 {
        self.ln_sf0(z).exp()
    }

    /// Standard density.
    pub fn pdf0(self, z: f64) -> f64 {
        self.ln_pdf0(z).exp()
    }

    /// Standard quantile.
    pub fn ppf0(self, p: f64) -> f64 {
        match self.base() {
            Base::Sev => (-(-p).ln_1p()).ln(),
            Base::Norm => norm_ppf(p),
            Base::Logis => (p / (1.0 - p)).ln(),
        }
    }

    /// `ln(F(zu) − F(zl))`, computed in the tail where it is accurate.
    pub fn ln_interval(self, zl: f64, zu: f64) -> f64 {
        if zl > 0.0 {
            let (a, b) = (self.ln_sf0(zl), self.ln_sf0(zu));
            a + (-(b - a).exp_m1()).ln()
        } else {
            let (a, b) = (self.ln_cdf0(zu), self.ln_cdf0(zl));
            a + (-(b - a).exp_m1()).ln()
        }
    }
}
