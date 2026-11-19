//! Pseudo-random generators with published reference outputs.

use super::{Rng, SeedableRng};

/// SplitMix64 (Steele, Lea & Flood): used to expand seeds.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
}

impl Rng for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl SeedableRng for SplitMix64 {
    fn seed_from_u64(seed: u64) -> Self {
        SplitMix64::new(seed)
    }
}

const MT_N: usize = 624;
const MT_M: usize = 397;

/// 32-bit Mersenne Twister MT19937. `Mt19937::new(5489)` is the reference
/// default seed; its first output is 3499211612.
#[derive(Debug, Clone)]
pub struct Mt19937 {
    mt: [u32; MT_N],
    idx: usize,
}

impl Mt19937 {
    /// Reference `init_genrand`.
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; MT_N];
        mt[0] = seed;
        for i in 1..MT_N {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: MT_N }
    }

    /// Reference `init_by_array`.
    pub fn from_key(key: &[u32]) -> Self {
        let mut g = Mt19937::new(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        let klen = key.len().max(1);
        for _ in 0..MT_N.max(klen) {
            let k = key.get(j).copied().unwrap_or(0);
            g.mt[i] = (g.mt[i] ^ (g.mt[i - 1] ^ (g.mt[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(k)
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                g.mt[0] = g.mt[MT_N - 1];
                i = 1;
            }
            if j >= klen {
                j = 0;
            }
        }
        for _ in 0..MT_N - 1 {
            g.mt[i] =
                (g.mt[i] ^ (g.mt[i - 1] ^ (g.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                g.mt[0] = g.mt[MT_N - 1];
                i = 1;
            }
        }
        g.mt[0] = 0x8000_0000;
        g.idx = MT_N;
        g
    }

    fn refill(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        for i in 0..MT_N {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % MT_N] & LOWER);
            let mut v = self.mt[(i + MT_M) % MT_N] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.mt[i] = v;
        }
        self.idx = 0;
    }

    fn next32(&mut self) -> u32 {
        if self.idx >= MT_N {
            self.refill();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

impl Rng for Mt19937 {
    fn next_u32(&mut self) -> u32 {
        self.next32()
    }
    /// Two consecutive outputs, the first in the high word.
    fn next_u64(&mut self) -> u64 {
        let hi = u64::from(self.next32());
        (hi << 32) | u64::from(self.next32())
    }
}

impl SeedableRng for Mt19937 {
    fn seed_from_u64(seed: u64) -> Self {
        Mt19937::from_key(&[seed as u32, (seed >> 32) as u32])
    }
}

const MT64_N: usize = 312;
const MT64_M: usize = 156;

/// 64-bit Mersenne Twister MT19937-64. `Mt19937_64::new(5489)` starts with
/// 14514284786278117030.
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Mt19937_64 {
    mt: [u64; MT64_N],
    idx: usize,
}

impl Mt19937_64 {
    /// Reference `init_genrand64`.
    pub fn new(seed: u64) -> Self {
        let mut mt = [0u64; MT64_N];
        mt[0] = seed;
        for i in 1..MT64_N {
            mt[i] = 6_364_136_223_846_793_005u64.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62)).wrapping_add(i as u64);
        }
        Mt19937_64 { mt, idx: MT64_N }
    }

    /// Reference `init_by_array64`.
    pub fn from_key(key: &[u64]) -> Self {
        let mut g = Mt19937_64::new(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        let klen = key.len().max(1);
        for _ in 0..MT64_N.max(klen) {
            let k = key.get(j).copied().unwrap_or(0);
            g.mt[i] = (g.mt[i] ^ (g.mt[i - 1] ^ (g.mt[i - 1] >> 62)).wrapping_mul(3_935_559_000_370_003_845))
                .wrapping_add(k)
                .wrapping_add(j as u64);
            i += 1;
            j += 1;
            if i >= MT64_N {
                g.mt[0] = g.mt[MT64_N - 1];
                i = 1;
            }
            if j >= klen {
                j = 0;
            }
        }
        for _ in 0..MT64_N - 1 {
            g.mt[i] = (g.mt[i] ^ (g.mt[i - 1] ^ (g.mt[i - 1] >> 62)).wrapping_mul(2_862_933_555_777_941_757))
                .wrapping_sub(i as u64);
            i += 1;
            if i >= MT64_N {
                g.mt[0] = g.mt[MT64_N - 1];
                i = 1;
            }
        }
        g.mt[0] = 1u64 << 63;
        g.idx = MT64_N;
        g
    }

    fn refill(&mut self) {
        const UPPER: u64 = 0xFFFF_FFFF_8000_0000;
        const LOWER: u64 = 0x7FFF_FFFF;
        for i in 0..MT64_N {
            let x = (self.mt[i] & UPPER) | (self.mt[(i + 1) % MT64_N] & LOWER);
            let mut v = self.mt[(i + MT64_M) % MT64_N] ^ (x >> 1);
            if x & 1 != 0 {
                v ^= 0xB502_6F5A_A966_19E9;
            }
            self.mt[i] = v;
        }
        self.idx = 0;
    }
}

impl Rng for Mt19937_64 {
    fn next_u64(&mut self) -> u64 {
        if self.idx >= MT64_N {
            self.refill();
        }
        let mut x = self.mt[self.idx];
        self.idx += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^ (x >> 43)
    }
}

impl SeedableRng for Mt19937_64 {
    fn seed_from_u64(seed: u64) -> Self {
        Mt19937_64::from_key(&[seed])
    }
}

const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

/// PCG XSL-RR 128/64 (`pcg64`): a 128-bit linear congruential generator with
/// a xorshift-low / random-rotate output permutation and a selectable odd
/// increment (stream).
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    /// Reference `pcg_setseq_128_srandom_r(initstate, initseq)`.
    pub fn new(initstate: u128, initseq: u128) -> Self {
        let mut g = Pcg64 { state: 0, inc: (initseq << 1) | 1 };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    /// Raw state and (odd) increment, e.g. to restore a saved generator.
    pub fn from_raw(state: u128, inc: u128) -> Self {
        Pcg64 { state, inc: inc | 1 }
    }

    /// `(state, inc)`, the complete generator state.
    pub fn raw(&self) -> (u128, u128) {
        (self.state, self.inc)
    }

    /// A generator for stream number `stream` of a seed; different streams
    /// are statistically independent sequences.
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let s = (u128::from(sm.next_u64()) << 64) | u128::from(sm.next_u64());
        Pcg64::new(s, u128::from(stream))
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }
}

impl Rng for Pcg64 {
    fn next_u64(&mut self) -> u64 {
        self.step();
        let rot = (self.state >> 122) as u32;
        let xored = ((self.state >> 64) as u64) ^ (self.state as u64);
        xored.rotate_right(rot)
    }
}

impl SeedableRng for Pcg64 {
    fn seed_from_u64(seed: u64) -> Self {
        Pcg64::with_stream(seed, 0xda3e_39cb_94b9_5bdb)
    }
}

/// xoshiro256** 1.0 (Blackman & Vigna): 256 bits of state, period 2²⁵⁶ − 1.
#[derive(Debug, Clone)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// From explicit state; the all-zero state is replaced by a fixed
    /// non-zero one because it is a fixed point of the recurrence.
    pub fn from_state(s: [u64; 4]) -> Self {
        if s == [0; 4] {
            return Xoshiro256StarStar::seed_from_u64(0);
        }
        Xoshiro256StarStar { s }
    }
}

impl Rng for Xoshiro256StarStar {
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

impl SeedableRng for Xoshiro256StarStar {
    fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        Xoshiro256StarStar { s: [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_twister_reference_outputs() {
        let mut g = Mt19937::new(5489);
        let want = [3_499_211_612u32, 581_869_302, 3_890_346_734, 3_586_334_585, 545_404_204];
        for w in want {
            assert_eq!(g.next_u32(), w);
        }
        // The 10000th output of the default-seeded generator (C++11 standard).
        let mut g = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10_000 {
            last = g.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
        // init_by_array reference vector (mt19937ar.out, first outputs).
        let mut g = Mt19937::from_key(&[0x123, 0x234, 0x345, 0x456]);
        assert_eq!([g.next_u32(), g.next_u32(), g.next_u32()], [1_067_595_299, 955_945_823, 477_289_528]);
    }

    #[test]
    fn mersenne_twister_64_reference_outputs() {
        let mut g = Mt19937_64::new(5489);
        let want = [
            14_514_284_786_278_117_030u64,
            4_620_546_740_167_642_908,
            13_109_570_281_517_897_720,
            17_462_938_647_148_434_322,
        ];
        for w in want {
            assert_eq!(g.next_u64(), w);
        }
        // mt19937-64.out for init_by_array64({0x12345, 0x23456, 0x34567, 0x45678}).
        let mut g = Mt19937_64::from_key(&[0x12345, 0x23456, 0x34567, 0x45678]);
        assert_eq!(g.next_u64(), 7_266_447_313_870_364_031);
        assert_eq!(g.next_u64(), 4_946_485_549_665_804_864);
    }

    #[test]
    fn pcg64_matches_numpy_and_reference_seeding() {
        let mut g = Pcg64::from_raw(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef, 5);
        let want = [
            14_033_088_863_580_555_306u64,
            330_374_187_581_533_017,
            13_490_766_448_995_287_451,
            7_840_894_061_775_059_679,
        ];
        for w in want {
            assert_eq!(g.next_u64(), w);
        }
        let mut g = Pcg64::new(42, 54);
        let want = [
            9_705_778_491_962_043_240u64,
            1_370_407_407_632_858_425,
            11_774_395_822_783_136_600,
            17_944_889_938_176_486_912,
        ];
        for w in want {
            assert_eq!(g.next_u64(), w);
        }
    }

    #[test]
    fn xoshiro_and_splitmix_reference_outputs() {
        let mut g = Xoshiro256StarStar::from_state([1, 2, 3, 4]);
        assert_eq!(
            [g.next_u64(), g.next_u64(), g.next_u64(), g.next_u64()],
            [11_520, 0, 1_509_978_240, 1_215_971_899_390_074_240]
        );
        let mut s = SplitMix64::new(1_234_567);
        assert_eq!(
            [s.next_u64(), s.next_u64(), s.next_u64()],
            [6_457_827_717_110_365_317, 3_203_168_211_198_807_973, 9_817_491_932_198_370_423]
        );
    }
}
