//! Scale mappers from data values to the unit interval (and back).

use super::ticks::{extended_ticks, log_ticks, probability_ticks, time_ticks, Tick};
use openstats_math::special::erf::norm_ppf;

/// Which distribution a probability axis is straightened for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilityKind {
    /// Normal: `z = Φ⁻¹(p)`.
    Normal,
    /// Weibull: `ln(−ln(1 − p))`.
    Weibull,
    /// Lognormal (probit axis; pair with a log data axis).
    Lognormal,
}

/// A mapping from data values to `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub enum Scale {
    /// Linear between `min` and `max`.
    Linear {
        /// Lower bound.
        min: f64,
        /// Upper bound.
        max: f64,
    },
    /// Base-10 logarithmic (both bounds > 0).
    Log {
        /// Lower bound (> 0).
        min: f64,
        /// Upper bound.
        max: f64,
    },
    /// Probability axis for fractions in (0, 1).
    Probability {
        /// Distribution.
        kind: ProbabilityKind,
        /// Lower bound (fraction).
        min: f64,
        /// Upper bound (fraction).
        max: f64,
    },
    /// `n` categories at the integer positions 0..n, each in its own band.
    Band {
        /// Number of categories.
        n: usize,
    },
    /// Date/time axis (Excel serial days).
    Time {
        /// Start.
        min: f64,
        /// End.
        max: f64,
    },
}

fn transform(kind: ProbabilityKind, p: f64) -> f64 {
    match kind {
        ProbabilityKind::Normal | ProbabilityKind::Lognormal => norm_ppf(p),
        ProbabilityKind::Weibull => (-(1.0 - p).ln()).ln(),
    }
}

fn untransform(kind: ProbabilityKind, t: f64) -> f64 {
    match kind {
        ProbabilityKind::Normal | ProbabilityKind::Lognormal => {
            0.5 * (1.0 + openstats_math::special::erf::erf(t / std::f64::consts::SQRT_2))
        }
        ProbabilityKind::Weibull => 1.0 - (-t.exp()).exp(),
    }
}

impl Scale {
    /// Linear scale padded by `pad` (fraction of the range) on both sides; a
    /// degenerate range is widened.
    pub fn linear_padded(min: f64, max: f64, pad: f64) -> Scale {
        let (mut lo, mut hi) = if min <= max { (min, max) } else { (max, min) };
        if !(lo.is_finite() && hi.is_finite()) {
            lo = 0.0;
            hi = 1.0;
        }
        if hi - lo < 1e-12 * lo.abs().max(1.0) {
            let d = if lo == 0.0 { 1.0 } else { lo.abs() * 0.05 };
            lo -= d;
            hi += d;
        }
        let p = (hi - lo) * pad;
        Scale::Linear { min: lo - p, max: hi + p }
    }

    /// Data-space bounds `(min, max)`; a band scale spans `(-0.5, n - 0.5)`.
    pub fn domain(&self) -> (f64, f64) {
        match *self {
            Scale::Linear { min, max } | Scale::Log { min, max } | Scale::Time { min, max } => (min, max),
            Scale::Probability { min, max, .. } => (min, max),
            Scale::Band { n } => (-0.5, n as f64 - 0.5),
        }
    }

    /// Same scale type over new bounds (used for zooming).
    pub fn with_domain(&self, lo: f64, hi: f64) -> Scale {
        match *self {
            Scale::Linear { .. } => Scale::Linear { min: lo, max: hi },
            Scale::Log { .. } => Scale::Log { min: lo.max(1e-300), max: hi.max(lo.max(1e-300) * 1.0001) },
            Scale::Probability { kind, .. } => {
                Scale::Probability { kind, min: lo.clamp(1e-9, 0.5), max: hi.clamp(0.5, 1.0 - 1e-9) }
            }
            Scale::Time { .. } => Scale::Time { min: lo, max: hi },
            Scale::Band { n } => Scale::Band { n },
        }
    }

    fn fwd(&self, v: f64) -> f64 {
        match *self {
            Scale::Linear { .. } | Scale::Time { .. } | Scale::Band { .. } => v,
            Scale::Log { .. } => {
                if v > 0.0 {
                    v.log10()
                } else {
                    f64::NAN
                }
            }
            Scale::Probability { kind, .. } => {
                if v > 0.0 && v < 1.0 {
                    transform(kind, v)
                } else {
                    f64::NAN
                }
            }
        }
    }

    /// Position of `v` in `[0, 1]` (outside for out-of-range values, NaN for invalid).
    pub fn norm(&self, v: f64) -> f64 {
        let (lo, hi) = self.domain();
        let (a, b) = (self.fwd(lo), self.fwd(hi));
        if b == a {
            return 0.5;
        }
        (self.fwd(v) - a) / (b - a)
    }

    /// Data value at position `t` in `[0, 1]`.
    pub fn invert(&self, t: f64) -> f64 {
        let (lo, hi) = self.domain();
        let (a, b) = (self.fwd(lo), self.fwd(hi));
        let u = a + t * (b - a);
        match *self {
            Scale::Linear { .. } | Scale::Time { .. } | Scale::Band { .. } => u,
            Scale::Log { .. } => 10f64.powf(u),
            Scale::Probability { kind, .. } => untransform(kind, u),
        }
    }

    /// Map `v` to pixels between `p0` (at `min`) and `p1` (at `max`).
    pub fn map(&self, v: f64, p0: f64, p1: f64) -> f64 {
        p0 + self.norm(v) * (p1 - p0)
    }

    /// Ticks for an axis about `target` ticks long.
    pub fn ticks(&self, target: usize) -> Vec<Tick> {
        match *self {
            Scale::Linear { min, max } => extended_ticks(min, max, target),
            Scale::Log { min, max } => log_ticks(min, max),
            Scale::Probability { min, max, .. } => probability_ticks(min, max),
            Scale::Time { min, max } => time_ticks(min, max, target),
            Scale::Band { n } => (0..n).map(|i| Tick { value: i as f64, label: String::new(), major: true }).collect(),
        }
    }

    /// Width of one band (in `[0, 1]`) of a band scale.
    pub fn band_width(&self) -> f64 {
        match *self {
            Scale::Band { n } if n > 0 => 1.0 / n as f64,
            _ => 0.0,
        }
    }

    /// Pad a domain to enclose a probability in (0,1) data range sensibly.
    pub fn probability_for(kind: ProbabilityKind, lo: f64, hi: f64) -> Scale {
        let lo = lo.clamp(1e-9, 0.5);
        let hi = hi.clamp(0.5, 1.0 - 1e-9);
        // Round outwards to the next tick decade.
        let down = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            .iter()
            .rev()
            .find(|&&p| p <= lo)
            .copied()
            .unwrap_or(1e-4);
        let up =
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999].iter().find(|&&p| p >= hi).copied().unwrap_or(0.9999);
        Scale::Probability { kind, min: down, max: up }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_and_log_round_trip() {
        let s = Scale::Linear { min: 10.0, max: 20.0 };
        assert!((s.norm(15.0) - 0.5).abs() < 1e-15);
        assert!((s.map(15.0, 100.0, 300.0) - 200.0).abs() < 1e-12);
        let l = Scale::Log { min: 1.0, max: 1000.0 };
        assert!((l.norm(10.0) - 1.0 / 3.0).abs() < 1e-12);
        assert!((l.invert(l.norm(37.5)) - 37.5).abs() < 1e-9);
        assert!(l.norm(-1.0).is_nan());
    }

    #[test]
    fn probability_scales() {
        let n = Scale::Probability { kind: ProbabilityKind::Normal, min: 0.001, max: 0.999 };
        assert!((n.norm(0.5) - 0.5).abs() < 1e-12);
        assert!((n.norm(0.84134474606854) + n.norm(0.15865525393146) - 1.0).abs() < 1e-9);
        assert!((n.invert(n.norm(0.9)) - 0.9).abs() < 1e-9);
        let w = Scale::Probability { kind: ProbabilityKind::Weibull, min: 0.01, max: 0.99 };
        assert!((w.invert(w.norm(0.632)) - 0.632).abs() < 1e-9);
        assert!(w.norm(1.0).is_nan());
        let p = Scale::probability_for(ProbabilityKind::Normal, 0.02, 0.97);
        assert_eq!(p.domain(), (0.01, 0.99));
    }

    #[test]
    fn band_and_padding() {
        let b = Scale::Band { n: 4 };
        assert_eq!(b.domain(), (-0.5, 3.5));
        assert!((b.norm(0.0) - 0.125).abs() < 1e-15);
        assert!((b.band_width() - 0.25).abs() < 1e-15);
        let s = Scale::linear_padded(5.0, 5.0, 0.05);
        let (lo, hi) = s.domain();
        assert!(lo < 5.0 && hi > 5.0);
        assert_eq!(Scale::linear_padded(f64::NAN, 1.0, 0.0).domain(), (0.0, 1.0));
    }
}
