//! Random numbers: deterministic generators and variate sampling.
//!
//! | item | contents |
//! |---|---|
//! | [`Rng`], [`SeedableRng`] | object-safe generator trait; uniform doubles, unbiased integers |
//! | [`Mt19937`], [`Mt19937_64`] | Mersenne Twister (Matsumoto–Nishimura), 32- and 64-bit |
//! | [`Pcg64`] | PCG XSL-RR 128/64 (O'Neill), 2¹²⁷ independent streams |
//! | [`Xoshiro256StarStar`] | xoshiro256** (Blackman–Vigna), seeded through SplitMix64 |
//! | [`variates`] | exact samplers: normal, exponential, gamma, beta, Poisson, binomial, … |
//! | [`multivariate`] | multivariate normal, Wishart, Dirichlet, multinomial |
//!
//! Every distribution in [`crate::dist`] has a `sample` method on its trait
//! ([`crate::dist::ContinuousDistribution::sample`],
//! [`crate::dist::DiscreteDistribution::sample`]); distributions without a
//! dedicated algorithm are sampled by inversion of their quantile function.

mod generators;
pub mod multivariate;
pub mod variates;

pub use generators::{Mt19937, Mt19937_64, Pcg64, SplitMix64, Xoshiro256StarStar};

/// The generator used when none is chosen.
pub type DefaultRng = Pcg64;

/// A source of random bits. Object safe, so samplers take `&mut dyn Rng`.
pub trait Rng {
    /// Next 64 uniformly distributed bits.
    fn next_u64(&mut self) -> u64;

    /// Next 32 uniformly distributed bits.
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform on [0, 1) with 53 random bits (every multiple of 2⁻⁵³).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform on the open interval (0, 1): the 2⁵² odd multiples of 2⁻⁵³.
    /// Safe to pass to `ln` and to quantile functions.
    fn next_f64_open(&mut self) -> f64 {
        ((self.next_u64() >> 12) as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0)
    }

    /// Unbiased uniform integer in `0..n` (Lemire's multiply-and-reject
    /// method). Returns 0 when `n == 0`.
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Fill a slice with uniforms on [0, 1).
    fn fill_f64(&mut self, out: &mut [f64]) {
        for v in out {
            *v = self.next_f64();
        }
    }
}

impl<R: Rng + ?Sized> Rng for &mut R {
    fn next_u64(&mut self) -> u64 {
        (**self).next_u64()
    }
    fn next_u32(&mut self) -> u32 {
        (**self).next_u32()
    }
}

/// Generators that can be created from a single 64-bit seed. Equal seeds
/// give equal streams on every platform.
pub trait SeedableRng: Sized {
    fn seed_from_u64(seed: u64) -> Self;
}
